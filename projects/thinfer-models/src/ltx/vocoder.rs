//! LTX-2.3 audio vocoder (`VocoderWithBWE`, log-mel -> 48kHz stereo waveform).
//! BigVGAN v2 main generator (init 1536, 6 up stages, x160 -> 16kHz) + a BWE
//! generator (init 512, 5 stages, x120 -> 48kHz residual) + a causal STFT mel
//! front-end + a Hann sinc resampler. Whole tail runs f32 (the upstream autocast;
//! bf16 accumulation over ~108 sequential convs degrades spectral metrics 40-90%).
//!
//! Ground truth: `ltx_core/model/audio_vae/vocoder.py` (`Vocoder`, `AMPBlock1`,
//! `Activation1d`, `UpSample1d`/`DownSample1d`, `MelSTFT`, `VocoderWithBWE`).
//! Config from the on-disk `audio_vae.safetensors` `vocoder`/`bwe` blocks; tensors
//! under `vocoder.vocoder.*`, `vocoder.bwe_generator.*`, `vocoder.mel_stft.*`
//! (the extra `vocoder.` prefix is on disk; pyref strips ONE).
//!
//! GPU/host split: the `Vocoder` conv stack (main + BWE) runs on GPU via the
//! net-new f32 ops (`Conv1dF32`, `ConvTranspose1dF32`, `SnakeBetaF32`,
//! `ReplicatePad1dF32` doubling as crop via signed offset, `ScaleF32`, `AddF32`).
//! The lighter `VocoderWithBWE` glue (main-output clamp, causal STFT -> mel, Hann
//! resampler, residual+skip+clamp) runs on host. Anti-alias up/down filters load
//! from the checkpoint ([1,1,12], identical kaiser everywhere -> loaded once,
//! tiled to [C,1,12]); the Hann resampler filter is `persistent=False` upstream so
//! it is regenerated here.

use std::collections::HashMap;

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError};
use thinfer_core::ops::{
    ActDtype, AddF32, Conv1dF32, Conv1dOp, ConvTranspose1dF32, ConvTranspose1dOp, Op,
    ReplicatePad1dF32, ReplicatePad1dOp, ScaleF32, ScaleOp, SnakeBetaF32, SnakeBetaOp, WeightDtype,
    WgslConfig, conv_transpose1d_lout, conv1d_lout, conv1d_uniform_bytes, crop1d_uniform_bytes,
    replicate_pad1d_uniform_bytes, scale_uniform_bytes, snake_beta_uniform_bytes,
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
pub const INPUT_SR: usize = 16000;
pub const OUTPUT_SR: usize = 48000;
/// mel_stft: n_fft 512, hop 80, win 512, mel 64.
pub const N_FFT: usize = 512;
pub const HOP: usize = 80;
const SNAKE_EPS: f32 = 1e-9;
const RESBLOCK_KERNELS: [usize; 3] = [3, 7, 11];
const RESBLOCK_DILATIONS: [usize; 3] = [1, 3, 5];
const FILTER_K: usize = 12; // anti-alias kaiser kernel size.
const ACT_RATIO: usize = 2; // Activation1d up/down ratio.

/// One `Vocoder` generator's structural config.
#[derive(Clone)]
struct GenCfg {
    prefix: &'static str,
    init_ch: usize,
    up_rates: &'static [usize],
    up_kernels: &'static [usize],
}

const MAIN: GenCfg = GenCfg {
    prefix: "vocoder.vocoder",
    init_ch: 1536,
    up_rates: &[5, 2, 2, 2, 2, 2],
    up_kernels: &[11, 4, 4, 4, 4, 4],
};
const BWE: GenCfg = GenCfg {
    prefix: "vocoder.bwe_generator",
    init_ch: 512,
    up_rates: &[6, 5, 2, 2, 2],
    up_kernels: &[12, 11, 4, 4, 4],
};

fn get_padding(k: usize, dilation: usize) -> usize {
    (k * dilation - dilation) / 2
}

// ============================================================================
// Pipelines
// ============================================================================

pub struct VocoderPipelines {
    conv1d: thinfer_core::backend::WgpuPipeline,
    convt1d: thinfer_core::backend::WgpuPipeline,
    snake: thinfer_core::backend::WgpuPipeline,
    pad: thinfer_core::backend::WgpuPipeline,
    scale: thinfer_core::backend::WgpuPipeline,
    add: thinfer_core::backend::WgpuPipeline,
}

impl VocoderPipelines {
    pub async fn compile(backend: &WgpuBackend) -> Result<Self, WgpuError> {
        // f32 acts; bf16 weights (learned convs + tiled filters all dequant to f32).
        let cfg = &WgslConfig {
            bf16_quant_writes: false,
            act_dtype: ActDtype::F32,
            weight_dtype: WeightDtype::Bf16,
        };
        Ok(Self {
            conv1d: backend
                .create_pipeline(
                    "ltx_vocoder_conv1d",
                    &<Conv1dF32 as Conv1dOp>::wgsl(cfg),
                    "main",
                    <Conv1dF32 as Conv1dOp>::layout(),
                )
                .await?,
            convt1d: backend
                .create_pipeline(
                    "ltx_vocoder_convt1d",
                    &<ConvTranspose1dF32 as ConvTranspose1dOp>::wgsl(cfg),
                    "main",
                    <ConvTranspose1dF32 as ConvTranspose1dOp>::layout(),
                )
                .await?,
            snake: backend
                .create_pipeline(
                    "ltx_vocoder_snake",
                    &<SnakeBetaF32 as SnakeBetaOp>::wgsl(cfg),
                    "main",
                    <SnakeBetaF32 as SnakeBetaOp>::layout(),
                )
                .await?,
            pad: backend
                .create_pipeline(
                    "ltx_vocoder_pad",
                    &<ReplicatePad1dF32 as ReplicatePad1dOp>::wgsl(cfg),
                    "main",
                    <ReplicatePad1dF32 as ReplicatePad1dOp>::layout(),
                )
                .await?,
            scale: backend
                .create_pipeline(
                    "ltx_vocoder_scale",
                    &<ScaleF32 as ScaleOp>::wgsl(cfg),
                    "main",
                    <ScaleF32 as ScaleOp>::layout(),
                )
                .await?,
            add: backend
                .create_pipeline(
                    "ltx_vocoder_add",
                    AddF32::wgsl(cfg),
                    "main",
                    AddF32::layout(),
                )
                .await?,
        })
    }
}

// ============================================================================
// A [C, L] activation buffer with explicit dims (B=1 throughout the stack).
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
/// Used to split the generator into bounded per-stage submits.
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

/// Copy a scope-local `Cl` into a workspace buffer that survives the submit.
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
    pl: &VocoderPipelines,
    x: Cl<'w>,
    w: BufRef,
    bias: BufRef,
    cout: u32,
    k: u32,
    stride: u32,
    dilation: u32,
    pad: u32,
    groups: u32,
) -> Result<Cl<'w>, WgpuError> {
    let lout = conv1d_lout(x.l, k, stride, dilation, pad);
    let out = scope.alloc((cout * lout) as u64 * 4)?;
    let u = scope.write_uniform(&conv1d_uniform_bytes(
        1, x.c, cout, x.l, lout, k, stride, dilation, pad, groups,
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
    pl: &VocoderPipelines,
    x: Cl<'w>,
    w: BufRef,
    bias: BufRef,
    cout: u32,
    k: u32,
    stride: u32,
    pad: u32,
    groups: u32,
) -> Result<Cl<'w>, WgpuError> {
    let lout = conv_transpose1d_lout(x.l, k, stride, 1, pad);
    let out = scope.alloc((cout * lout) as u64 * 4)?;
    let u = scope.write_uniform(&conv1d_uniform_bytes(
        1, x.c, cout, x.l, lout, k, stride, 1, pad, groups,
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

fn add<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &VocoderPipelines,
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
    pl: &VocoderPipelines,
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

fn snake<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &VocoderPipelines,
    x: Cl<'w>,
    act: &ActBufs,
) -> Result<Cl<'w>, WgpuError> {
    let out = scope.alloc(x.bytes())?;
    let u = scope.write_uniform(&snake_beta_uniform_bytes(x.n(), x.c, x.l, SNAKE_EPS))?;
    scope.snake_beta::<SnakeBetaF32>(
        &pl.snake,
        x.buf,
        scope.import_copy(act.alpha),
        scope.import_copy(act.beta),
        u,
        out,
        x.n(),
    )?;
    Ok(Cl {
        buf: out,
        c: x.c,
        l: x.l,
    })
}

/// Edge-replicate pad `lpad`/`rpad` samples (out len `L+lpad+rpad`).
fn pad<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &VocoderPipelines,
    x: Cl<'w>,
    lpad: u32,
    rpad: u32,
) -> Result<Cl<'w>, WgpuError> {
    let lout = x.l + lpad + rpad;
    let out = scope.alloc((x.c * lout) as u64 * 4)?;
    let u = scope.write_uniform(&replicate_pad1d_uniform_bytes(1, x.c, x.l, lpad, rpad))?;
    scope.replicate_pad1d::<ReplicatePad1dF32>(&pl.pad, x.buf, u, out, x.c * lout)?;
    Ok(Cl {
        buf: out,
        c: x.c,
        l: lout,
    })
}

/// Crop `lout` samples starting at `start`.
fn crop<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &VocoderPipelines,
    x: Cl<'w>,
    start: u32,
    lout: u32,
) -> Result<Cl<'w>, WgpuError> {
    let out = scope.alloc((x.c * lout) as u64 * 4)?;
    let u = scope.write_uniform(&crop1d_uniform_bytes(1, x.c, x.l, start, lout))?;
    scope.replicate_pad1d::<ReplicatePad1dF32>(&pl.pad, x.buf, u, out, x.c * lout)?;
    Ok(Cl {
        buf: out,
        c: x.c,
        l: lout,
    })
}

/// Anti-aliased Activation1d: UpSample1d(2) -> SnakeBeta -> DownSample1d(2),
/// length-preserving. `flt` carries the per-C depthwise [C,1,12] filters + zero
/// bias. Replicate pads/crops match `UpSample1d`/`LowPassFilter1d` (kaiser k12).
#[allow(clippy::too_many_arguments)]
fn activation1d<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &VocoderPipelines,
    x: Cl<'w>,
    act: &ActBufs,
    flt: &FilterBufs,
    zero_bias: BufRef,
) -> Result<Cl<'w>, WgpuError> {
    let c = x.c;
    let l = x.l;
    // UpSample1d(ratio 2, k12): pad(5,5) -> convt(stride2,k12,pad0,depthwise) ->
    // scale*2 -> crop[15 : 15+2L].
    let p = pad(scope, pl, x, 5, 5)?;
    let up = conv_transpose1d(scope, pl, p, flt.up, zero_bias, c, FILTER_K as u32, 2, 0, c)?;
    let up = scale(scope, pl, up, ACT_RATIO as f32)?;
    let up = crop(scope, pl, up, 15, 2 * l)?;
    // SnakeBeta at 2L.
    let s = snake(scope, pl, up, act)?;
    // DownSample1d(ratio 2, k12): pad(5,6) -> conv1d(stride2,k12,pad0,depthwise).
    let p = pad(scope, pl, s, 5, 6)?;
    let down = conv1d(
        scope,
        pl,
        p,
        flt.down,
        zero_bias,
        c,
        FILTER_K as u32,
        2,
        1,
        0,
        c,
    )?;
    debug_assert_eq!(down.l, l, "activation1d not length-preserving");
    Ok(down)
}

/// One AMPBlock1 (BigVGAN v2 resblock): 3 dilated sub-blocks, each
/// `act -> conv1(dil) -> act -> conv2(dil1)` with a residual add.
#[allow(clippy::too_many_arguments)]
fn amp_block<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &VocoderPipelines,
    mut x: Cl<'w>,
    rb: &AmpBufs,
    kernel: u32,
    flt: &FilterBufs,
    zero_bias: BufRef,
) -> Result<Cl<'w>, WgpuError> {
    for (m, &dil_taps) in RESBLOCK_DILATIONS.iter().enumerate() {
        let dil = dil_taps as u32;
        let pad1 = get_padding(kernel as usize, dil as usize) as u32;
        let pad2 = get_padding(kernel as usize, 1) as u32;
        let xt = activation1d(scope, pl, x, &rb.acts1[m], flt, zero_bias)?;
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
            1,
        )?;
        let xt = activation1d(scope, pl, xt, &rb.acts2[m], flt, zero_bias)?;
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
            1,
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

#[derive(Clone, Copy)]
struct ActH {
    alpha: WeightHandle,
    beta: WeightHandle,
}

struct AmpH {
    convs1: Vec<ConvH>,
    convs2: Vec<ConvH>,
    acts1: Vec<ActH>,
    acts2: Vec<ActH>,
}

struct GenH {
    cfg: GenCfg,
    conv_pre: ConvH,
    ups: Vec<ConvH>,
    resblocks: Vec<AmpH>,
    act_post: ActH,
    conv_post: ConvH,
    /// per-up-stage output channel count (`init >> (i+1)`).
    channels: Vec<u32>,
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

fn reg_act<S: WeightSource>(res: &WeightResidency<S>, prefix: &str) -> Result<ActH, LoadError> {
    Ok(ActH {
        alpha: register_passthrough(res, &WeightId(format!("{prefix}.act.alpha")))?,
        beta: register_passthrough(res, &WeightId(format!("{prefix}.act.beta")))?,
    })
}

impl GenH {
    fn register<S: WeightSource>(
        res: &WeightResidency<S>,
        cfg: &GenCfg,
    ) -> Result<Self, LoadError> {
        let p = cfg.prefix;
        let n_up = cfg.up_rates.len();
        let channels: Vec<u32> = (0..n_up).map(|i| (cfg.init_ch >> (i + 1)) as u32).collect();
        let conv_pre = reg_conv(res, &format!("{p}.conv_pre"), true)?;
        let ups = (0..n_up)
            .map(|i| reg_conv(res, &format!("{p}.ups.{i}"), true))
            .collect::<Result<_, _>>()?;
        let mut resblocks = Vec::with_capacity(n_up * RESBLOCK_KERNELS.len());
        for i in 0..n_up {
            for kidx in 0..RESBLOCK_KERNELS.len() {
                let j = i * RESBLOCK_KERNELS.len() + kidx;
                let bp = format!("{p}.resblocks.{j}");
                resblocks.push(AmpH {
                    convs1: (0..3)
                        .map(|m| reg_conv(res, &format!("{bp}.convs1.{m}"), true))
                        .collect::<Result<_, _>>()?,
                    convs2: (0..3)
                        .map(|m| reg_conv(res, &format!("{bp}.convs2.{m}"), true))
                        .collect::<Result<_, _>>()?,
                    acts1: (0..3)
                        .map(|m| reg_act(res, &format!("{bp}.acts1.{m}")))
                        .collect::<Result<_, _>>()?,
                    acts2: (0..3)
                        .map(|m| reg_act(res, &format!("{bp}.acts2.{m}")))
                        .collect::<Result<_, _>>()?,
                });
            }
        }
        Ok(Self {
            cfg: cfg.clone(),
            conv_pre,
            ups,
            resblocks,
            act_post: reg_act(res, &format!("{p}.act_post"))?,
            conv_post: reg_conv(res, &format!("{p}.conv_post"), false)?,
            channels,
        })
    }

    fn final_ch(&self) -> u32 {
        (self.cfg.init_ch >> self.cfg.up_rates.len()) as u32
    }
}

// Acquired buffers ---------------------------------------------------------

struct ConvBufs {
    weight: BufRef,
    bias: Option<BufRef>,
}
struct ActBufs {
    alpha: BufRef,
    beta: BufRef,
}
struct AmpBufs {
    convs1: Vec<ConvBufs>,
    convs2: Vec<ConvBufs>,
    acts1: Vec<ActBufs>,
    acts2: Vec<ActBufs>,
}
struct GenBufs {
    conv_pre: ConvBufs,
    ups: Vec<ConvBufs>,
    resblocks: Vec<AmpBufs>,
    act_post: ActBufs,
    conv_post: ConvBufs,
}

/// Per-channel-count depthwise filter buffers ([C,1,12] bf16) + zero bias.
#[derive(Clone, Copy)]
struct FilterBufs {
    up: BufRef,
    down: BufRef,
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

/// Dequant a bf16 `[C]` weight to f32 and upload it to a fresh workspace buffer
/// (the SnakeBeta op reads alpha/beta as `array<f32>`; the whole tail is f32).
async fn upload_f32_from_bf16<S: WeightSource>(
    res: &WeightResidency<S>,
    backend: &WgpuBackend,
    workspace: &Workspace<WgpuBackend>,
    wh: WeightHandle,
    held: &mut Vec<WsBuf<WgpuBackend>>,
) -> Result<BufRef, VocoderError<S::Error>> {
    let view = res.acquire(wh, backend).await?;
    let nbytes = view.buf().len;
    let bytes = backend
        .read_buffer(view.buf().id, view.buf().offset, nbytes)
        .await?;
    let f32v = act_readback_to_f32(ActDtype::Bf16, &bytes, (nbytes / 2) as usize);
    drop(view);
    let buf = workspace.alloc((f32v.len() * 4) as u64)?;
    backend.write_buffer(buf.id(), 0, &act_upload_bytes(ActDtype::F32, &f32v))?;
    let r = buf.as_buf_ref();
    held.push(buf);
    Ok(r)
}

async fn acq_act<S: WeightSource>(
    res: &WeightResidency<S>,
    backend: &WgpuBackend,
    workspace: &Workspace<WgpuBackend>,
    h: &ActH,
    held: &mut Vec<WsBuf<WgpuBackend>>,
) -> Result<ActBufs, VocoderError<S::Error>> {
    Ok(ActBufs {
        alpha: upload_f32_from_bf16(res, backend, workspace, h.alpha, held).await?,
        beta: upload_f32_from_bf16(res, backend, workspace, h.beta, held).await?,
    })
}

// ============================================================================
// Decoder
// ============================================================================

pub struct Vocoder {
    pub pipelines: VocoderPipelines,
    main: GenH,
    bwe: GenH,
    /// Anti-alias kaiser taps (12 each), identical across all Activation1d.
    up_filter: Vec<f32>,
    down_filter: Vec<f32>,
    /// STFT bases (host f32): forward_basis [514,1,512], mel_basis [64,257].
    forward_basis: Vec<f32>,
    mel_basis: Vec<f32>,
}

#[derive(Debug)]
pub enum VocoderError<SE: core::fmt::Debug> {
    Wgpu(WgpuError),
    Load(LoadError),
    Residency(ResidencyError<SE, WgpuError>),
}
impl<SE: core::fmt::Debug> From<WgpuError> for VocoderError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}
impl<SE: core::fmt::Debug> From<LoadError> for VocoderError<SE> {
    fn from(e: LoadError) -> Self {
        Self::Load(e)
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for VocoderError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

const N_FREQS: usize = N_FFT / 2 + 1; // 257

impl Vocoder {
    pub async fn new<S: WeightSource>(
        pipelines: VocoderPipelines,
        residency: &WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<Self, VocoderError<S::Error>> {
        let main = GenH::register(residency, &MAIN)?;
        let bwe = GenH::register(residency, &BWE)?;
        let up_filter = read_weight_f32(
            backend,
            residency,
            "vocoder.vocoder.act_post.upsample.filter",
            FILTER_K,
        )
        .await?;
        let down_filter = read_weight_f32(
            backend,
            residency,
            "vocoder.vocoder.act_post.downsample.lowpass.filter",
            FILTER_K,
        )
        .await?;
        let forward_basis = read_weight_f32(
            backend,
            residency,
            "vocoder.mel_stft.stft_fn.forward_basis",
            N_FREQS * 2 * N_FFT,
        )
        .await?;
        let mel_basis = read_weight_f32(
            backend,
            residency,
            "vocoder.mel_stft.mel_basis",
            MEL_BINS * N_FREQS,
        )
        .await?;
        Ok(Self {
            pipelines,
            main,
            bwe,
            up_filter,
            down_filter,
            forward_basis,
            mel_basis,
        })
    }

    /// Decode a log-mel `[2, T, 64]` (host f32) into a 48kHz stereo waveform
    /// `[2, T_wav]` (host f32, clamped [-1,1]).
    pub async fn decode<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        mel: &[f32],
        frames: usize,
    ) -> Result<Vec<f32>, VocoderError<S::Error>> {
        assert_eq!(mel.len(), OUT_CHANNELS * frames * MEL_BINS, "mel size");
        let mut x_main = self
            .run_generator(backend, residency, workspace, &self.main, mel, frames)
            .await?;
        let l16 = x_main.len() / OUT_CHANNELS;
        for v in &mut x_main {
            *v = v.clamp(-1.0, 1.0); // main use_tanh_at_final=false -> clamp.
        }
        let output_length = l16 * OUTPUT_SR / INPUT_SR;

        // Pad x_main to a hop multiple, causal STFT -> log-mel for the BWE drive.
        let rem = l16 % HOP;
        let l16_pad = if rem != 0 { l16 + (HOP - rem) } else { l16 };
        let padded = if l16_pad != l16 {
            let mut p = vec![0.0f32; OUT_CHANNELS * l16_pad];
            for c in 0..OUT_CHANNELS {
                p[c * l16_pad..c * l16_pad + l16].copy_from_slice(&x_main[c * l16..(c + 1) * l16]);
            }
            p
        } else {
            x_main.clone()
        };
        let (bwe_mel, t_frames) = self.compute_mel(&padded, l16_pad);

        let residual = self
            .run_generator(backend, residency, workspace, &self.bwe, &bwe_mel, t_frames)
            .await?;
        let skip = self.hann_resample(&x_main, l16);
        let l48 = residual.len() / OUT_CHANNELS;
        assert_eq!(
            skip.len(),
            OUT_CHANNELS * l48,
            "resampler vs residual length"
        );

        let mut wav = vec![0.0f32; OUT_CHANNELS * output_length];
        for c in 0..OUT_CHANNELS {
            for t in 0..output_length {
                let v = residual[c * l48 + t] + skip[c * l48 + t];
                wav[c * output_length + t] = v.clamp(-1.0, 1.0);
            }
        }
        Ok(wav)
    }

    /// Run one `Vocoder` generator on GPU: mel `[2, frames, 64]` (host) ->
    /// waveform `[2, Lout]` (host f32, pre final activation).
    async fn run_generator<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        gn: &GenH,
        mel: &[f32],
        frames: usize,
    ) -> Result<Vec<f32>, VocoderError<S::Error>> {
        let cin = OUT_CHANNELS * MEL_BINS; // 128
        // Input rearrange (host): mel[s,t,c] -> x0[s*64+c, t].
        let mut x0 = vec![0.0f32; cin * frames];
        for s in 0..OUT_CHANNELS {
            for c in 0..MEL_BINS {
                for t in 0..frames {
                    x0[(s * MEL_BINS + c) * frames + t] = mel[(s * frames + t) * MEL_BINS + c];
                }
            }
        }

        let mut pins: Vec<GpuView> = Vec::new();
        // f32-dequant'd snake alpha/beta buffers, held until the scope submits.
        let mut act_held: Vec<WsBuf<WgpuBackend>> = Vec::new();
        let bufs = self
            .acquire(residency, backend, workspace, gn, &mut pins, &mut act_held)
            .await?;

        // Per-distinct-C depthwise filter buffers + a shared zero-bias buffer.
        let mut filter_pins: Vec<WsBuf<WgpuBackend>> = Vec::new();
        let mut filters: HashMap<u32, FilterBufs> = HashMap::new();
        let mut distinct: Vec<u32> = gn.channels.clone();
        distinct.push(gn.final_ch());
        distinct.sort_unstable();
        distinct.dedup();
        let max_c = *distinct.iter().max().unwrap();
        for &c in &distinct {
            let up = self.tile_filter(workspace, backend, &self.up_filter, c)?;
            let down = self.tile_filter(workspace, backend, &self.down_filter, c)?;
            let bufs = FilterBufs {
                up: up.as_buf_ref(),
                down: down.as_buf_ref(),
            };
            filter_pins.push(up);
            filter_pins.push(down);
            filters.insert(c, bufs);
        }
        // Zero bias (bf16) for the no-bias convs (filters + conv_post).
        let zero_bias = workspace.alloc((max_c.max(gn.final_ch()) as u64) * 2)?;
        backend.write_buffer(
            zero_bias.id(),
            0,
            &act_upload_bytes(ActDtype::Bf16, &vec![0.0f32; max_c as usize]),
        )?;
        let zero_bias_ref = zero_bias.as_buf_ref();

        let in_buf = workspace.alloc((x0.len() * 4) as u64)?;
        backend.write_buffer(in_buf.id(), 0, &act_upload_bytes(ActDtype::F32, &x0))?;

        // The whole generator (~108 convs) in ONE scope keeps every intermediate
        // `[C,L]` live until submit -> a peak that grows unbounded with audio
        // length (5s OOM'd the 8GB card on a 15MB workspace alloc). Split it into
        // per-stage / per-resblock submits, persisting only the inter-stage
        // activation across the boundary and draining the pool between. The peak
        // drops from "sum of all buffers" to "one resblock's working set". This is
        // NUMERICALLY IDENTICAL (same ops/order; the boundary is a verbatim
        // buffer copy), so `vocoder_parity` is unaffected.
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
                    gn.cfg.init_ch as u32,
                    7,
                    1,
                    1,
                    3,
                    1,
                )?;
                let p = persist_cl(&scope, workspace, x)?;
                scope.submit_void().await?;
                p
            };
            workspace.drain_pool();

            for i in 0..gn.ups.len() {
                let cout = gn.channels[i];
                let rate = gn.cfg.up_rates[i] as u32;
                let upk = gn.cfg.up_kernels[i] as u32;
                let pad_t = (upk - rate) / 2;
                let flt = filters[&cout];
                // conv_transpose -> the shared input to all 3 resblocks.
                let xt = {
                    let scope = workspace.batch();
                    let x = cur.import(&scope);
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
                        1,
                    )?;
                    let p = persist_cl(&scope, workspace, x)?;
                    scope.submit_void().await?;
                    p
                };
                drop(cur);
                workspace.drain_pool();
                // Each kernel-size resblock reads the SAME `xt`; persisted for the mean.
                let mut ys: Vec<Persisted> = Vec::with_capacity(RESBLOCK_KERNELS.len());
                for (kidx, &kernel) in RESBLOCK_KERNELS.iter().enumerate() {
                    let rb = &bufs.resblocks[i * RESBLOCK_KERNELS.len() + kidx];
                    let scope = workspace.batch();
                    let x = xt.import(&scope);
                    let y = amp_block(&scope, pl, x, rb, kernel as u32, &flt, zero_bias_ref)?;
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

            // act_post Activation1d -> conv_post final_ch -> 2 (no bias) -> readback.
            {
                let scope = workspace.batch();
                let flt = filters[&gn.final_ch()];
                let x = cur.import(&scope);
                let x = activation1d(&scope, pl, x, &bufs.act_post, &flt, zero_bias_ref)?;
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
                    1,
                )?;
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
        drop(filter_pins);
        drop(act_held);
        drop(pins);
        Ok(out_host)
    }

    /// Tile a 12-tap filter to a depthwise `[C,1,12]` bf16 conv weight buffer.
    fn tile_filter(
        &self,
        workspace: &Workspace<WgpuBackend>,
        backend: &WgpuBackend,
        taps: &[f32],
        c: u32,
    ) -> Result<WsBuf<WgpuBackend>, WgpuError> {
        let mut w = Vec::with_capacity(c as usize * taps.len());
        for _ in 0..c {
            w.extend_from_slice(taps);
        }
        let buf = workspace.alloc((w.len() as u64) * 2)?; // bf16
        backend.write_buffer(buf.id(), 0, &act_upload_bytes(ActDtype::Bf16, &w))?;
        Ok(buf)
    }

    #[allow(clippy::too_many_arguments)]
    async fn acquire<'r, S: WeightSource>(
        &self,
        res: &'r WeightResidency<S>,
        backend: &WgpuBackend,
        workspace: &Workspace<WgpuBackend>,
        gn: &GenH,
        pins: &mut Vec<GpuView<'r>>,
        act_held: &mut Vec<WsBuf<WgpuBackend>>,
    ) -> Result<GenBufs, VocoderError<S::Error>> {
        let conv_pre = acq_conv(res, backend, &gn.conv_pre, pins).await?;
        let mut ups = Vec::with_capacity(gn.ups.len());
        for h in &gn.ups {
            ups.push(acq_conv(res, backend, h, pins).await?);
        }
        let mut resblocks = Vec::with_capacity(gn.resblocks.len());
        for rb in &gn.resblocks {
            let mut convs1 = Vec::with_capacity(3);
            let mut convs2 = Vec::with_capacity(3);
            let mut acts1 = Vec::with_capacity(3);
            let mut acts2 = Vec::with_capacity(3);
            for m in 0..3 {
                convs1.push(acq_conv(res, backend, &rb.convs1[m], pins).await?);
                convs2.push(acq_conv(res, backend, &rb.convs2[m], pins).await?);
                acts1.push(acq_act(res, backend, workspace, &rb.acts1[m], act_held).await?);
                acts2.push(acq_act(res, backend, workspace, &rb.acts2[m], act_held).await?);
            }
            resblocks.push(AmpBufs {
                convs1,
                convs2,
                acts1,
                acts2,
            });
        }
        let act_post = acq_act(res, backend, workspace, &gn.act_post, act_held).await?;
        let conv_post = acq_conv(res, backend, &gn.conv_post, pins).await?;
        Ok(GenBufs {
            conv_pre,
            ups,
            resblocks,
            act_post,
            conv_post,
        })
    }

    // -- host STFT mel ------------------------------------------------------

    /// Causal log-mel: pad left (win-hop), conv with `forward_basis` (stride hop),
    /// magnitude, mel_basis projection, log(clamp(.,1e-5)). Returns `([2, T, 64]`,
    /// t_frames) in the BWE input layout.
    fn compute_mel(&self, x: &[f32], l: usize) -> (Vec<f32>, usize) {
        let left_pad = N_FFT - HOP; // causal
        let t_frames = (l + left_pad - N_FFT) / HOP + 1;
        let mut out = vec![0.0f32; OUT_CHANNELS * t_frames * MEL_BINS];
        for ch in 0..OUT_CHANNELS {
            for f in 0..t_frames {
                let start = f as isize * HOP as isize - left_pad as isize;
                let mut spec = vec![0.0f32; N_FREQS * 2];
                for (b, sp) in spec.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    let wbase = b * N_FFT;
                    for n in 0..N_FFT {
                        let idx = start + n as isize;
                        let xv = if idx >= 0 && (idx as usize) < l {
                            x[ch * l + idx as usize]
                        } else {
                            0.0
                        };
                        acc += self.forward_basis[wbase + n] * xv;
                    }
                    *sp = acc;
                }
                for m in 0..MEL_BINS {
                    let mut mel_v = 0.0f32;
                    let mbase = m * N_FREQS;
                    for k in 0..N_FREQS {
                        let re = spec[k];
                        let im = spec[N_FREQS + k];
                        mel_v += self.mel_basis[mbase + k] * (re * re + im * im).sqrt();
                    }
                    out[(ch * t_frames + f) * MEL_BINS + m] = mel_v.max(1e-5).ln();
                }
            }
        }
        (out, t_frames)
    }

    // -- host Hann resampler (UpSample1d ratio 3, persistent=False) ---------

    fn hann_resample(&self, x: &[f32], l: usize) -> Vec<f32> {
        let ratio = OUTPUT_SR / INPUT_SR; // 3
        let rolloff = 0.99f64;
        let lp_width = 6usize;
        let width = (lp_width as f64 / rolloff).ceil() as usize; // 7
        let ksize = 2 * width * ratio + 1; // 43
        let pad_amt = width; // 7
        let pad_left = 2 * width * ratio; // 42
        let pad_right = ksize - ratio; // 40
        let mut filt = vec![0.0f32; ksize];
        for (i, fv) in filt.iter_mut().enumerate() {
            let t = (i as f64 / ratio as f64 - width as f64) * rolloff;
            let tc = t.clamp(-(lp_width as f64), lp_width as f64);
            let window = (tc * std::f64::consts::PI / lp_width as f64 / 2.0)
                .cos()
                .powi(2);
            let sinc = if t == 0.0 {
                1.0
            } else {
                (std::f64::consts::PI * t).sin() / (std::f64::consts::PI * t)
            };
            *fv = (sinc * window * rolloff / ratio as f64) as f32;
        }
        let lp = l + 2 * pad_amt;
        let lout_t = (lp - 1) * ratio + ksize;
        let crop_len = lout_t - pad_left - pad_right;
        let mut out = vec![0.0f32; OUT_CHANNELS * crop_len];
        for ch in 0..OUT_CHANNELS {
            let get = |idx: isize| -> f32 { x[ch * l + idx.clamp(0, l as isize - 1) as usize] };
            for lo in 0..crop_len {
                let pos = lo + pad_left;
                let mut acc = 0.0f32;
                for (kk, &fk) in filt.iter().enumerate() {
                    if pos >= kk && (pos - kk).is_multiple_of(ratio) {
                        let li = (pos - kk) / ratio;
                        if li < lp {
                            acc += fk * get(li as isize - pad_amt as isize);
                        }
                    }
                }
                out[ch * crop_len + lo] = acc * ratio as f32;
            }
        }
        out
    }
}

async fn read_weight_f32<S: WeightSource>(
    backend: &WgpuBackend,
    residency: &WeightResidency<S>,
    id: &str,
    n: usize,
) -> Result<Vec<f32>, VocoderError<S::Error>> {
    let h = register_passthrough(residency, &WeightId(id.into()))?;
    let view = residency.acquire(h, backend).await?;
    let bytes = backend
        .read_buffer(view.buf().id, 0, (n * 2) as u64)
        .await?;
    Ok(act_readback_to_f32(ActDtype::Bf16, &bytes, n))
}
