//! LTX-2 19B (ltx2-rapid) audio vocoder: a plain HiFiGAN generator (log-mel ->
//! 16kHz stereo waveform). Distinct from the LTX-2.3 `VocoderWithBWE` (BigVGAN v2
//! main + BWE + mel-STFT + snakebeta + anti-alias filters): the 19B merge ships a
//! single HiFiGAN generator with `resblock="1"`, leaky_relu activations, a final
//! tanh, and NO band-width-extension / resampling. Whole stack runs f32.
//!
//! Ground truth (byte-identical across three clones): the plain `Vocoder` class in
//! `lightx2v/.../ltx2/audio_vae/vocoder.py` (+ `LTX-2/.../audio_vae/vocoder.py`
//! `Vocoder` at 271-416, `resnet.py` `ResBlock1`). Config from the on-disk
//! `LTX2_audio_vae_bf16.safetensors` `vocoder` block: init 1024, up_rates
//! [6,5,2,2,2] (x120), up_kernels [16,15,8,4,4], resblock kernels [3,7,11] /
//! dilations [1,3,5], stereo, 16kHz out. Tensors: `vocoder.{conv_pre, ups.i,
//! resblocks.j.convs1/2.m, conv_post}` (conv_post has NO bias).
//!
//! Forward (vocoder.py:376-416): conv_pre -> for each of 5 up stages {
//! leaky_relu(0.1); ConvTranspose1d(pad=(k-stride)/2); mean of the 3 kernel-size
//! ResBlock1 } -> leaky_relu(0.01) [act_post default slope] -> conv_post ->
//! tanh. ResBlock1 (resnet.py:73-80): for each of 3 dilations, `xt =
//! conv1(leaky_relu(x,0.1)); xt = conv2(leaky_relu(xt,0.1)); x = x + xt` with
//! convs1 dilated (pad `(k*d-d)/2`), convs2 dilation-1 (pad `(k-1)/2`).
//!
//! Reuses the audio VAE decoder unchanged (its `audio_vae.decoder.*` keys are
//! byte-identical to the 22B file); only the vocoder tail differs.

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError};
use thinfer_core::ops::{
    ActDtype, AddF32, Conv1dF32, Conv1dOp, ConvTranspose1dF32, ConvTranspose1dOp, LeakyReluF32,
    LeakyReluOp, Op, ScaleF32, ScaleOp, TanhF32, WeightDtype, WgslConfig, conv_transpose1d_lout,
    conv1d_lout, conv1d_uniform_bytes, leaky_relu_uniform_bytes, scale_uniform_bytes,
};
use thinfer_core::residency::{GpuView, ResidencyError, WeightHandle, WeightResidency};
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace, WsBuf};

use crate::common::loader::{LoadError, register_passthrough};
use crate::common::seq::{act_readback_to_f32, act_upload_bytes};

// ============================================================================
// Config (disk-verified)
// ============================================================================

pub const MEL_BINS: usize = 64;
pub const OUT_CHANNELS: usize = 2;
/// The 19B HiFiGAN vocoder emits at 24kHz. The mel runs at 100 frames/sec
/// (16kHz / hop 160) and the generator upsamples it by `prod(UP_RATES)` = 240,
/// so 100 * 240 = 24000 samples/sec (ref `vocoder.py` `output_sampling_rate`
/// default 24000). No BWE / resample. Tagging this 16kHz stretched playback 1.5x
/// (low pitch + over-long container).
pub const OUTPUT_SR: u32 = 24000;

const INIT_CH: usize = 1024;
const UP_RATES: [usize; 5] = [6, 5, 2, 2, 2];
const UP_KERNELS: [usize; 5] = [16, 15, 8, 4, 4];
const RESBLOCK_KERNELS: [usize; 3] = [3, 7, 11];
const RESBLOCK_DILATIONS: [usize; 3] = [1, 3, 5];
/// Up-stage / ResBlock1 activation slope.
const LRELU_SLOPE: f32 = 0.1;
/// `act_post` is `nn.LeakyReLU()` with no arg -> PyTorch default 0.01 (NOT 0.1).
const ACT_POST_SLOPE: f32 = 0.01;

/// `padding="same"` for an odd kernel at unit stride.
fn same_pad(k: usize, dilation: usize) -> u32 {
    ((k * dilation - dilation) / 2) as u32
}

// ============================================================================
// Pipelines
// ============================================================================

pub struct RapidVocoderPipelines {
    conv1d: thinfer_core::backend::WgpuPipeline,
    convt1d: thinfer_core::backend::WgpuPipeline,
    leaky: thinfer_core::backend::WgpuPipeline,
    tanh: thinfer_core::backend::WgpuPipeline,
    add: thinfer_core::backend::WgpuPipeline,
    scale: thinfer_core::backend::WgpuPipeline,
}

impl RapidVocoderPipelines {
    pub async fn compile(backend: &WgpuBackend) -> Result<Self, WgpuError> {
        // f32 acts, bf16 weights (all convs dequant to f32).
        let cfg = &WgslConfig {
            bf16_quant_writes: false,
            act_dtype: ActDtype::F32,
            weight_dtype: WeightDtype::Bf16,
        };
        Ok(Self {
            conv1d: backend
                .create_pipeline(
                    "ltx_rapid_vocoder_conv1d",
                    &<Conv1dF32 as Conv1dOp>::wgsl(cfg),
                    "main",
                    <Conv1dF32 as Conv1dOp>::layout(),
                )
                .await?,
            convt1d: backend
                .create_pipeline(
                    "ltx_rapid_vocoder_convt1d",
                    &<ConvTranspose1dF32 as ConvTranspose1dOp>::wgsl(cfg),
                    "main",
                    <ConvTranspose1dF32 as ConvTranspose1dOp>::layout(),
                )
                .await?,
            leaky: backend
                .create_pipeline(
                    "ltx_rapid_vocoder_leaky_relu",
                    &<LeakyReluF32 as LeakyReluOp>::wgsl(cfg),
                    "main",
                    <LeakyReluF32 as LeakyReluOp>::layout(),
                )
                .await?,
            tanh: backend
                .create_pipeline(
                    "ltx_rapid_vocoder_tanh",
                    TanhF32::wgsl(cfg),
                    "main",
                    TanhF32::layout(),
                )
                .await?,
            add: backend
                .create_pipeline(
                    "ltx_rapid_vocoder_add",
                    AddF32::wgsl(cfg),
                    "main",
                    AddF32::layout(),
                )
                .await?,
            scale: backend
                .create_pipeline(
                    "ltx_rapid_vocoder_scale",
                    &<ScaleF32 as ScaleOp>::wgsl(cfg),
                    "main",
                    <ScaleF32 as ScaleOp>::layout(),
                )
                .await?,
        })
    }
}

// ============================================================================
// A [C, L] activation buffer with explicit dims (B=1 throughout).
// ============================================================================

#[derive(Clone, Copy)]
struct Cl<'w> {
    buf: BatchBuf<'w>,
    c: u32,
    l: u32,
}

impl Cl<'_> {
    fn n(&self) -> u32 {
        self.c * self.l
    }
    fn bytes(&self) -> u64 {
        self.n() as u64 * 4
    }
}

/// A `[C, L]` activation copied into a workspace buffer that outlives one
/// `BatchScope` submit, carrying its dims so the next scope can re-import it.
struct Persisted {
    buf: WsBuf<WgpuBackend>,
    c: u32,
    l: u32,
}

impl Persisted {
    fn import<'w>(&self, scope: &BatchScope<'w, WgpuBackend>) -> Cl<'w> {
        Cl {
            buf: scope.import_copy(self.buf.as_buf_ref()),
            c: self.c,
            l: self.l,
        }
    }
}

fn persist_cl<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    workspace: &Workspace<WgpuBackend>,
    x: Cl<'w>,
) -> Result<Persisted, WgpuError> {
    let ws = workspace.alloc(x.bytes())?;
    let dst = scope.import_copy(ws.as_buf_ref());
    scope.copy_buffer_to_buffer(x.buf, 0, dst, 0, x.bytes())?;
    Ok(Persisted {
        buf: ws,
        c: x.c,
        l: x.l,
    })
}

// ============================================================================
// GPU op helpers (operate on Cl within a BatchScope)
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn conv1d<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &RapidVocoderPipelines,
    x: Cl<'w>,
    w: BufRef,
    bias: BufRef,
    cout: u32,
    k: u32,
    stride: u32,
    dilation: u32,
    pad: u32,
) -> Result<Cl<'w>, WgpuError> {
    let lout = conv1d_lout(x.l, k, stride, dilation, pad);
    let out = scope.alloc((cout * lout) as u64 * 4)?;
    let u = scope.write_uniform(&conv1d_uniform_bytes(
        1, x.c, cout, x.l, lout, k, stride, dilation, pad, 1,
    ))?;
    scope.conv1d::<Conv1dF32>(
        &pl.conv1d,
        x.buf,
        scope.import_copy(w),
        scope.import_copy(bias),
        u,
        out,
        cout * lout,
    )?;
    Ok(Cl {
        buf: out,
        c: cout,
        l: lout,
    })
}

#[allow(clippy::too_many_arguments)]
fn conv_transpose1d<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &RapidVocoderPipelines,
    x: Cl<'w>,
    w: BufRef,
    bias: BufRef,
    cout: u32,
    k: u32,
    stride: u32,
    pad: u32,
) -> Result<Cl<'w>, WgpuError> {
    let lout = conv_transpose1d_lout(x.l, k, stride, 1, pad);
    let out = scope.alloc((cout * lout) as u64 * 4)?;
    let u = scope.write_uniform(&conv1d_uniform_bytes(
        1, x.c, cout, x.l, lout, k, stride, 1, pad, 1,
    ))?;
    scope.conv_transpose1d::<ConvTranspose1dF32>(
        &pl.convt1d,
        x.buf,
        scope.import_copy(w),
        scope.import_copy(bias),
        u,
        out,
        cout * lout,
    )?;
    Ok(Cl {
        buf: out,
        c: cout,
        l: lout,
    })
}

fn leaky<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &RapidVocoderPipelines,
    x: Cl<'w>,
    slope: f32,
) -> Result<Cl<'w>, WgpuError> {
    let out = scope.alloc(x.bytes())?;
    let u = scope.write_uniform(&leaky_relu_uniform_bytes(x.n(), slope))?;
    scope.leaky_relu::<LeakyReluF32>(&pl.leaky, x.buf, u, out, x.n())?;
    Ok(Cl {
        buf: out,
        c: x.c,
        l: x.l,
    })
}

fn tanh<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &RapidVocoderPipelines,
    x: Cl<'w>,
) -> Result<Cl<'w>, WgpuError> {
    let out = scope.alloc(x.bytes())?;
    scope.dispatch_op::<TanhF32>(&pl.tanh, &[x.buf], out)?;
    Ok(Cl {
        buf: out,
        c: x.c,
        l: x.l,
    })
}

fn add<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &RapidVocoderPipelines,
    a: Cl<'w>,
    b: Cl<'w>,
) -> Result<Cl<'w>, WgpuError> {
    let out = scope.alloc(a.bytes())?;
    scope.dispatch_op::<AddF32>(&pl.add, &[a.buf, b.buf], out)?;
    Ok(Cl {
        buf: out,
        c: a.c,
        l: a.l,
    })
}

fn scale<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &RapidVocoderPipelines,
    x: Cl<'w>,
    s: f32,
) -> Result<Cl<'w>, WgpuError> {
    let out = scope.alloc(x.bytes())?;
    let u = scope.write_uniform(&scale_uniform_bytes(x.n(), s))?;
    scope.scale::<ScaleF32>(&pl.scale, x.buf, u, out, x.n())?;
    Ok(Cl {
        buf: out,
        c: x.c,
        l: x.l,
    })
}

/// One ResBlock1 (resnet.py:73-80): 3 dilated sub-blocks, each
/// `conv1(leaky(x)) -> conv2(leaky(.))` with a residual add.
fn res_block<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &RapidVocoderPipelines,
    mut x: Cl<'w>,
    rb: &ResBlockBufs,
    kernel: u32,
) -> Result<Cl<'w>, WgpuError> {
    for (m, &dil) in RESBLOCK_DILATIONS.iter().enumerate() {
        let dil = dil as u32;
        let pad1 = same_pad(kernel as usize, dil as usize);
        let pad2 = same_pad(kernel as usize, 1);
        let xt = leaky(scope, pl, x, LRELU_SLOPE)?;
        let xt = conv1d(
            scope,
            pl,
            xt,
            rb.convs1[m].weight,
            rb.convs1[m].bias.unwrap(),
            x.c,
            kernel,
            1,
            dil,
            pad1,
        )?;
        let xt = leaky(scope, pl, xt, LRELU_SLOPE)?;
        let xt = conv1d(
            scope,
            pl,
            xt,
            rb.convs2[m].weight,
            rb.convs2[m].bias.unwrap(),
            x.c,
            kernel,
            1,
            1,
            pad2,
        )?;
        x = add(scope, pl, x, xt)?;
    }
    Ok(x)
}

// ============================================================================
// Weight handles
// ============================================================================

#[derive(Clone, Copy)]
struct ConvH {
    weight: WeightHandle,
    bias: Option<WeightHandle>,
}

struct ResBlockH {
    convs1: Vec<ConvH>,
    convs2: Vec<ConvH>,
}

fn reg_conv<S: WeightSource>(
    res: &WeightResidency<S>,
    prefix: &str,
    has_bias: bool,
) -> Result<ConvH, LoadError> {
    Ok(ConvH {
        weight: register_passthrough(res, &WeightId(format!("{prefix}.weight")))?,
        bias: has_bias
            .then(|| register_passthrough(res, &WeightId(format!("{prefix}.bias"))))
            .transpose()?,
    })
}

// Acquired buffers --------------------------------------------------------

struct ConvBufs {
    weight: BufRef,
    bias: Option<BufRef>,
}
struct ResBlockBufs {
    convs1: Vec<ConvBufs>,
    convs2: Vec<ConvBufs>,
}
struct GenBufs {
    conv_pre: ConvBufs,
    ups: Vec<ConvBufs>,
    resblocks: Vec<ResBlockBufs>,
    conv_post: ConvBufs,
}

async fn acq_conv<'r, S: WeightSource>(
    res: &'r WeightResidency<S>,
    backend: &WgpuBackend,
    h: &ConvH,
    pins: &mut Vec<GpuView<'r>>,
) -> Result<ConvBufs, ResidencyError<S::Error, WgpuError>> {
    let wv = res.acquire(h.weight, backend).await?;
    let weight = wv.buf();
    pins.push(wv);
    let bias = match h.bias {
        Some(b) => {
            let bv = res.acquire(b, backend).await?;
            let buf = bv.buf();
            pins.push(bv);
            Some(buf)
        }
        None => None,
    };
    Ok(ConvBufs { weight, bias })
}

// ============================================================================
// Decoder
// ============================================================================

pub struct RapidVocoder {
    pipelines: RapidVocoderPipelines,
    conv_pre: ConvH,
    ups: Vec<ConvH>,
    resblocks: Vec<ResBlockH>,
    conv_post: ConvH,
    /// per-up-stage output channel count (`init >> (i+1)`).
    channels: Vec<u32>,
}

#[derive(Debug)]
pub enum RapidVocoderError<SE: core::fmt::Debug> {
    Wgpu(WgpuError),
    Load(LoadError),
    Residency(ResidencyError<SE, WgpuError>),
}
impl<SE: core::fmt::Debug> From<WgpuError> for RapidVocoderError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}
impl<SE: core::fmt::Debug> From<LoadError> for RapidVocoderError<SE> {
    fn from(e: LoadError) -> Self {
        Self::Load(e)
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for RapidVocoderError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

impl RapidVocoder {
    pub fn new<S: WeightSource>(
        pipelines: RapidVocoderPipelines,
        residency: &WeightResidency<S>,
    ) -> Result<Self, RapidVocoderError<S::Error>> {
        let n_up = UP_RATES.len();
        let channels: Vec<u32> = (0..n_up).map(|i| (INIT_CH >> (i + 1)) as u32).collect();
        let conv_pre = reg_conv(residency, "vocoder.conv_pre", true)?;
        let ups = (0..n_up)
            .map(|i| reg_conv(residency, &format!("vocoder.ups.{i}"), true))
            .collect::<Result<_, _>>()?;
        let mut resblocks = Vec::with_capacity(n_up * RESBLOCK_KERNELS.len());
        for i in 0..n_up {
            for kidx in 0..RESBLOCK_KERNELS.len() {
                let j = i * RESBLOCK_KERNELS.len() + kidx;
                let bp = format!("vocoder.resblocks.{j}");
                resblocks.push(ResBlockH {
                    convs1: (0..3)
                        .map(|m| reg_conv(residency, &format!("{bp}.convs1.{m}"), true))
                        .collect::<Result<_, _>>()?,
                    convs2: (0..3)
                        .map(|m| reg_conv(residency, &format!("{bp}.convs2.{m}"), true))
                        .collect::<Result<_, _>>()?,
                });
            }
        }
        // conv_post has no bias (`use_bias_at_final=false`).
        let conv_post = reg_conv(residency, "vocoder.conv_post", false)?;
        Ok(Self {
            pipelines,
            conv_pre,
            ups,
            resblocks,
            conv_post,
            channels,
        })
    }

    /// Decode a log-mel `[2, frames, 64]` (host f32) into a 16kHz stereo waveform
    /// `[2, T_wav]` (host f32, planar `[c, t]`, in `[-1, 1]` from the final tanh).
    pub async fn decode<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        mel: &[f32],
        frames: usize,
    ) -> Result<Vec<f32>, RapidVocoderError<S::Error>> {
        assert_eq!(mel.len(), OUT_CHANNELS * frames * MEL_BINS, "mel size");
        let cin = OUT_CHANNELS * MEL_BINS; // 128
        // Input rearrange (host): mel[s,t,c] -> x0[s*64+c, t] (stereo-outer).
        let mut x0 = vec![0.0f32; cin * frames];
        for s in 0..OUT_CHANNELS {
            for c in 0..MEL_BINS {
                for t in 0..frames {
                    x0[(s * MEL_BINS + c) * frames + t] = mel[(s * frames + t) * MEL_BINS + c];
                }
            }
        }

        let mut pins: Vec<GpuView> = Vec::new();
        let bufs = self.acquire(residency, backend, &mut pins).await?;

        // conv_post has no bias -> a small bf16 zero-bias buffer.
        let zero_bias = workspace.alloc((OUT_CHANNELS as u64) * 2)?;
        backend.write_buffer(
            zero_bias.id(),
            0,
            &act_upload_bytes(ActDtype::Bf16, &[0.0f32; OUT_CHANNELS]),
        )?;
        let zero_bias_ref = zero_bias.as_buf_ref();

        let in_buf = workspace.alloc((x0.len() * 4) as u64)?;
        backend.write_buffer(in_buf.id(), 0, &act_upload_bytes(ActDtype::F32, &x0))?;

        // Split the generator into per-stage / per-resblock submits (like the 22B
        // vocoder): keeping every `[C,L]` intermediate live in one scope grows the
        // VRAM peak with audio length and OOMs the 8GB card. Persist only the
        // inter-stage activation across boundaries and drain the pool between.
        // Numerically identical (same ops/order; the boundary is a verbatim copy).
        let out_host;
        let out_n;
        {
            let pl = &self.pipelines;
            // conv_pre: 128 -> init_ch, k7 pad3.
            let mut cur = {
                let scope = workspace.batch();
                let x = Cl {
                    buf: scope.import_copy(in_buf.as_buf_ref()),
                    c: cin as u32,
                    l: frames as u32,
                };
                let x = conv1d(
                    &scope,
                    pl,
                    x,
                    bufs.conv_pre.weight,
                    bufs.conv_pre.bias.unwrap(),
                    INIT_CH as u32,
                    7,
                    1,
                    1,
                    3,
                )?;
                let p = persist_cl(&scope, workspace, x)?;
                scope.submit_void().await?;
                p
            };
            workspace.drain_pool();

            for i in 0..self.ups.len() {
                let cout = self.channels[i];
                let rate = UP_RATES[i] as u32;
                let upk = UP_KERNELS[i] as u32;
                let pad_t = (upk - rate) / 2;
                // leaky_relu(0.1) -> conv_transpose -> shared input to the 3 resblocks.
                let xt = {
                    let scope = workspace.batch();
                    let x = cur.import(&scope);
                    let x = leaky(&scope, pl, x, LRELU_SLOPE)?;
                    let x = conv_transpose1d(
                        &scope,
                        pl,
                        x,
                        bufs.ups[i].weight,
                        bufs.ups[i].bias.unwrap(),
                        cout,
                        upk,
                        rate,
                        pad_t,
                    )?;
                    let p = persist_cl(&scope, workspace, x)?;
                    scope.submit_void().await?;
                    p
                };
                drop(cur);
                workspace.drain_pool();
                // Each kernel-size resblock reads the SAME `xt`; mean over the 3.
                let mut ys: Vec<Persisted> = Vec::with_capacity(RESBLOCK_KERNELS.len());
                for (kidx, &kernel) in RESBLOCK_KERNELS.iter().enumerate() {
                    let rb = &bufs.resblocks[i * RESBLOCK_KERNELS.len() + kidx];
                    let scope = workspace.batch();
                    let x = xt.import(&scope);
                    let y = res_block(&scope, pl, x, rb, kernel as u32)?;
                    let p = persist_cl(&scope, workspace, y)?;
                    scope.submit_void().await?;
                    ys.push(p);
                    workspace.drain_pool();
                }
                drop(xt);
                workspace.drain_pool();
                // Mean of the 3 resblocks.
                cur = {
                    let scope = workspace.batch();
                    let mut acc: Option<Cl> = None;
                    for y in &ys {
                        let yc = y.import(&scope);
                        acc = Some(match acc {
                            None => yc,
                            Some(a) => add(&scope, pl, a, yc)?,
                        });
                    }
                    let x = scale(
                        &scope,
                        pl,
                        acc.expect("resblocks"),
                        1.0 / RESBLOCK_KERNELS.len() as f32,
                    )?;
                    let p = persist_cl(&scope, workspace, x)?;
                    scope.submit_void().await?;
                    p
                };
                drop(ys);
                workspace.drain_pool();
            }

            // act_post leaky_relu(0.01) -> conv_post (no bias) -> tanh -> readback.
            {
                let scope = workspace.batch();
                let x = cur.import(&scope);
                let x = leaky(&scope, pl, x, ACT_POST_SLOPE)?;
                let x = conv1d(
                    &scope,
                    pl,
                    x,
                    bufs.conv_post.weight,
                    zero_bias_ref,
                    OUT_CHANNELS as u32,
                    7,
                    1,
                    1,
                    3,
                )?;
                let x = tanh(&scope, pl, x)?;
                out_n = x.n() as usize;
                let out_buf = workspace.alloc(x.bytes())?;
                let dst = scope.import_copy(out_buf.as_buf_ref());
                scope.copy_buffer_to_buffer(x.buf, 0, dst, 0, x.bytes())?;
                scope.submit_void().await?;
                let bytes = backend.read_buffer(out_buf.id(), 0, x.bytes()).await?;
                out_host = act_readback_to_f32(ActDtype::F32, &bytes, out_n);
            }
            drop(cur);
        }
        workspace.drain_pool();
        drop(pins);
        Ok(out_host)
    }

    async fn acquire<'r, S: WeightSource>(
        &self,
        res: &'r WeightResidency<S>,
        backend: &WgpuBackend,
        pins: &mut Vec<GpuView<'r>>,
    ) -> Result<GenBufs, RapidVocoderError<S::Error>> {
        let conv_pre = acq_conv(res, backend, &self.conv_pre, pins).await?;
        let mut ups = Vec::with_capacity(self.ups.len());
        for h in &self.ups {
            ups.push(acq_conv(res, backend, h, pins).await?);
        }
        let mut resblocks = Vec::with_capacity(self.resblocks.len());
        for rb in &self.resblocks {
            let mut convs1 = Vec::with_capacity(3);
            let mut convs2 = Vec::with_capacity(3);
            for m in 0..3 {
                convs1.push(acq_conv(res, backend, &rb.convs1[m], pins).await?);
                convs2.push(acq_conv(res, backend, &rb.convs2[m], pins).await?);
            }
            resblocks.push(ResBlockBufs { convs1, convs2 });
        }
        let conv_post = acq_conv(res, backend, &self.conv_post, pins).await?;
        Ok(GenBufs {
            conv_pre,
            ups,
            resblocks,
            conv_post,
        })
    }
}
