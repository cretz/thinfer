//! BigVGAN v2 generator: the shared upsampling conv tower behind both the
//! LTX-2.3 vocoder and the MiniMax-H3 audio VAE decoder.
//!
//! Both checkpoints are the same architecture with different hyperparameters
//! and different tensor names, so everything structural lives here and each
//! model contributes only a [`BigVganCfg`]. The two ship IDENTICAL
//! `resblock_kernel_sizes [3, 7, 11]` / `resblock_dilation_sizes [1, 3, 5]` and
//! the same 12-tap Kaiser anti-alias filters, which is why this is a
//! parameterization rather than a second implementation.
//!
//! The whole tower runs f32. That is not a default: bf16 accumulation over the
//! ~108 sequential convolutions degrades spectral metrics 40-90%.
//!
//! **Peak memory is why the generator is split into per-stage submits.** Run as
//! one scope, every intermediate `[C, L]` stays live until submit and the peak
//! grows without bound in audio length (5 seconds OOM'd an 8 GB card). Each
//! stage and each resblock submits on its own and only the inter-stage
//! activation is persisted across the boundary, so the peak is one resblock's
//! working set. This is NUMERICALLY IDENTICAL: same ops in the same order, and
//! the boundary is a verbatim buffer copy.

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

pub const SNAKE_EPS: f32 = 1e-9;
/// Shared across every released BigVGAN v2 checkpoint we load.
pub const RESBLOCK_KERNELS: [usize; 3] = [3, 7, 11];
pub const RESBLOCK_DILATIONS: [usize; 3] = [1, 3, 5];
/// Anti-alias Kaiser kernel size, and the `Activation1d` up/down ratio.
pub const FILTER_K: usize = 12;
pub const ACT_RATIO: usize = 2;

fn get_padding(k: usize, dilation: usize) -> usize {
    (k * dilation - dilation) / 2
}

/// A k=1 projection applied to the host input before `conv_pre`.
#[derive(Clone, Debug)]
pub struct InProj {
    pub name: String,
    /// Channels it emits, which is what `conv_pre` consumes.
    pub out_channels: usize,
}

/// One generator's structure and tensor naming.
///
/// The naming flags exist because the two checkpoints nest the same modules
/// differently; they are checkpoint facts, not options.
#[derive(Clone, Debug)]
pub struct BigVganCfg {
    pub prefix: String,
    pub in_channels: usize,
    pub out_channels: usize,
    pub init_ch: usize,
    pub up_rates: Vec<usize>,
    pub up_kernels: Vec<usize>,
    /// `ups.{i}.0` (MiniMax-H3) vs `ups.{i}` (LTX).
    pub ups_nested: bool,
    /// One interleaved `activations.{2m}` / `activations.{2m + 1}` list
    /// (MiniMax-H3) vs separate `acts1.{m}` / `acts2.{m}` (LTX).
    pub acts_interleaved: bool,
    /// Module name of the post activation: `activation_post` vs `act_post`.
    pub act_post: String,
    /// Optional k=1 projection ahead of `conv_pre` (H3's `dec_in_proj`, which
    /// lifts 32 latent channels to the 2048 `conv_pre` expects).
    pub in_proj: Option<InProj>,
    /// Tensors holding the shared Kaiser taps. Identical at every activation
    /// site, so they are read once and tiled per channel count.
    pub up_filter: String,
    pub down_filter: String,
}

impl BigVganCfg {
    fn channels(&self) -> Vec<u32> {
        (0..self.up_rates.len())
            .map(|i| (self.init_ch >> (i + 1)) as u32)
            .collect()
    }

    fn final_ch(&self) -> u32 {
        (self.init_ch >> self.up_rates.len()) as u32
    }

    fn up_name(&self, i: usize) -> String {
        let p = &self.prefix;
        if self.ups_nested {
            format!("{p}.ups.{i}.0")
        } else {
            format!("{p}.ups.{i}")
        }
    }

    /// Activation `m` of the first/second slot of resblock `j`. Interleaved
    /// checkpoints store both slots in one list, even indices first.
    fn act_name(&self, j: usize, m: usize, second: bool) -> String {
        let p = &self.prefix;
        if self.acts_interleaved {
            let idx = 2 * m + usize::from(second);
            format!("{p}.resblocks.{j}.activations.{idx}")
        } else {
            let slot = if second { 2 } else { 1 };
            format!("{p}.resblocks.{j}.acts{slot}.{m}")
        }
    }
}

// ============================================================================
// Pipelines
// ============================================================================

pub struct BigVganPipelines {
    conv1d: thinfer_core::backend::WgpuPipeline,
    convt1d: thinfer_core::backend::WgpuPipeline,
    snake: thinfer_core::backend::WgpuPipeline,
    pad: thinfer_core::backend::WgpuPipeline,
    scale: thinfer_core::backend::WgpuPipeline,
    add: thinfer_core::backend::WgpuPipeline,
}

impl BigVganPipelines {
    pub async fn compile(backend: &WgpuBackend, label: &str) -> Result<Self, WgpuError> {
        // f32 acts; bf16 weights (learned convs + tiled filters all dequant to f32).
        let cfg = &WgslConfig {
            bf16_quant_writes: false,
            act_dtype: ActDtype::F32,
            weight_dtype: WeightDtype::Bf16,
        };
        Ok(Self {
            conv1d: backend
                .create_pipeline(
                    &format!("{label}_conv1d"),
                    &<Conv1dF32 as Conv1dOp>::wgsl(cfg),
                    "main",
                    <Conv1dF32 as Conv1dOp>::layout(),
                )
                .await?,
            convt1d: backend
                .create_pipeline(
                    &format!("{label}_convt1d"),
                    &<ConvTranspose1dF32 as ConvTranspose1dOp>::wgsl(cfg),
                    "main",
                    <ConvTranspose1dF32 as ConvTranspose1dOp>::layout(),
                )
                .await?,
            snake: backend
                .create_pipeline(
                    &format!("{label}_snake"),
                    &<SnakeBetaF32 as SnakeBetaOp>::wgsl(cfg),
                    "main",
                    <SnakeBetaF32 as SnakeBetaOp>::layout(),
                )
                .await?,
            pad: backend
                .create_pipeline(
                    &format!("{label}_pad"),
                    &<ReplicatePad1dF32 as ReplicatePad1dOp>::wgsl(cfg),
                    "main",
                    <ReplicatePad1dF32 as ReplicatePad1dOp>::layout(),
                )
                .await?,
            scale: backend
                .create_pipeline(
                    &format!("{label}_scale"),
                    &<ScaleF32 as ScaleOp>::wgsl(cfg),
                    "main",
                    <ScaleF32 as ScaleOp>::layout(),
                )
                .await?,
            add: backend
                .create_pipeline(
                    &format!("{label}_add"),
                    AddF32::wgsl(cfg),
                    "main",
                    AddF32::layout(),
                )
                .await?,
        })
    }
}

// ============================================================================
// A [C, L] activation buffer with explicit dims (B = 1 throughout the stack).
// ============================================================================

#[derive(Clone, Copy)]
pub struct Cl<'w> {
    pub buf: BatchBuf<'w>,
    pub c: u32,
    pub l: u32,
}

impl Cl<'_> {
    pub fn n(&self) -> u32 {
        self.c * self.l
    }
    pub fn bytes(&self) -> u64 {
        self.n() as u64 * 4
    }
}

/// A `[C, L]` activation copied into a workspace buffer that outlives one
/// `BatchScope`, carrying its dims so the next scope can re-import it.
pub struct Persisted {
    buf: WsBuf<WgpuBackend>,
    c: u32,
    l: u32,
}

impl Persisted {
    pub fn import<'w>(&self, scope: &BatchScope<'w, WgpuBackend>) -> Cl<'w> {
        Cl {
            buf: scope.import_copy(self.buf.as_buf_ref()),
            c: self.c,
            l: self.l,
        }
    }
}

/// Copy a scope-local `Cl` into a workspace buffer that survives the submit.
pub fn persist_cl<'w>(
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
pub fn conv1d<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &BigVganPipelines,
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
pub fn conv_transpose1d<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &BigVganPipelines,
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

pub fn add<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &BigVganPipelines,
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

pub fn scale<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &BigVganPipelines,
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
    pl: &BigVganPipelines,
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

/// Edge-replicate pad `lpad`/`rpad` samples (out len `L + lpad + rpad`).
pub fn pad<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &BigVganPipelines,
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
pub fn crop<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &BigVganPipelines,
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

/// Anti-aliased `Activation1d`: `UpSample1d(2) -> SnakeBeta -> DownSample1d(2)`,
/// length-preserving. `flt` carries the per-C depthwise `[C, 1, 12]` filters and
/// a zero bias; the pads and crops match `UpSample1d` / `LowPassFilter1d`.
pub fn activation1d<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &BigVganPipelines,
    x: Cl<'w>,
    act: &ActBufs,
    flt: &FilterBufs,
    zero_bias: BufRef,
) -> Result<Cl<'w>, WgpuError> {
    let c = x.c;
    let l = x.l;
    let p = pad(scope, pl, x, 5, 5)?;
    let up = conv_transpose1d(scope, pl, p, flt.up, zero_bias, c, FILTER_K as u32, 2, 0, c)?;
    let up = scale(scope, pl, up, ACT_RATIO as f32)?;
    let up = crop(scope, pl, up, 15, 2 * l)?;
    let s = snake(scope, pl, up, act)?;
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

/// One `AMPBlock1`: three dilated sub-blocks, each
/// `act -> conv1(dilated) -> act -> conv2(dilation 1)` with a residual add.
pub fn amp_block<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &BigVganPipelines,
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
            rb.convs1[m].bias.expect("resblock conv1 bias"),
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
            rb.convs2[m].bias.expect("resblock conv2 bias"),
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
    in_proj: Option<ConvH>,
    conv_pre: ConvH,
    ups: Vec<ConvH>,
    resblocks: Vec<AmpH>,
    act_post: ActH,
    conv_post: ConvH,
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
        cfg: &BigVganCfg,
    ) -> Result<Self, LoadError> {
        let p = &cfg.prefix;
        let n_up = cfg.up_rates.len();
        let in_proj = cfg
            .in_proj
            .as_ref()
            .map(|ip| reg_conv(res, &ip.name, true))
            .transpose()?;
        let conv_pre = reg_conv(res, &format!("{p}.conv_pre"), true)?;
        let ups = (0..n_up)
            .map(|i| reg_conv(res, &cfg.up_name(i), true))
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
                        .map(|m| reg_act(res, &cfg.act_name(j, m, false)))
                        .collect::<Result<_, _>>()?,
                    acts2: (0..3)
                        .map(|m| reg_act(res, &cfg.act_name(j, m, true)))
                        .collect::<Result<_, _>>()?,
                });
            }
        }
        Ok(Self {
            in_proj,
            conv_pre,
            ups,
            resblocks,
            act_post: reg_act(res, &format!("{p}.{}", cfg.act_post))?,
            conv_post: reg_conv(res, &format!("{p}.conv_post"), false)?,
        })
    }
}

// Acquired buffers ---------------------------------------------------------

struct ConvBufs {
    weight: BufRef,
    bias: Option<BufRef>,
}
pub struct ActBufs {
    alpha: BufRef,
    beta: BufRef,
}
pub struct AmpBufs {
    convs1: Vec<ConvBufs>,
    convs2: Vec<ConvBufs>,
    acts1: Vec<ActBufs>,
    acts2: Vec<ActBufs>,
}
struct GenBufs {
    in_proj: Option<ConvBufs>,
    conv_pre: ConvBufs,
    ups: Vec<ConvBufs>,
    resblocks: Vec<AmpBufs>,
    act_post: ActBufs,
    conv_post: ConvBufs,
}

/// Per-channel-count depthwise filter buffers (`[C, 1, 12]` bf16) + zero bias.
#[derive(Clone, Copy)]
pub struct FilterBufs {
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
) -> Result<BufRef, BigVganError<S::Error>> {
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
) -> Result<ActBufs, BigVganError<S::Error>> {
    Ok(ActBufs {
        alpha: upload_f32_from_bf16(res, backend, workspace, h.alpha, held).await?,
        beta: upload_f32_from_bf16(res, backend, workspace, h.beta, held).await?,
    })
}

#[derive(Debug)]
pub enum BigVganError<SE: core::fmt::Debug> {
    Wgpu(WgpuError),
    Load(LoadError),
    Residency(ResidencyError<SE, WgpuError>),
}
impl<SE: core::fmt::Debug> From<WgpuError> for BigVganError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}
impl<SE: core::fmt::Debug> From<LoadError> for BigVganError<SE> {
    fn from(e: LoadError) -> Self {
        Self::Load(e)
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for BigVganError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

// ============================================================================
// Generator
// ============================================================================

pub struct BigVgan {
    cfg: BigVganCfg,
    handles: GenH,
    /// Anti-alias Kaiser taps, identical at every `Activation1d` site.
    up_filter: Vec<f32>,
    down_filter: Vec<f32>,
}

impl BigVgan {
    pub async fn new<S: WeightSource>(
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        cfg: BigVganCfg,
    ) -> Result<Self, BigVganError<S::Error>> {
        let handles = GenH::register(residency, &cfg)?;
        let up_filter = read_weight_f32(backend, residency, &cfg.up_filter, FILTER_K).await?;
        let down_filter = read_weight_f32(backend, residency, &cfg.down_filter, FILTER_K).await?;
        Ok(Self {
            cfg,
            handles,
            up_filter,
            down_filter,
        })
    }

    pub fn cfg(&self) -> &BigVganCfg {
        &self.cfg
    }

    /// Run the tower: `x0` is `[in_channels, l]` host f32, the result is
    /// `[out_channels, l_out]` host f32 with NO final activation applied.
    pub async fn run<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        pipelines: &BigVganPipelines,
        x0: &[f32],
        l: usize,
    ) -> Result<Vec<f32>, BigVganError<S::Error>> {
        let cfg = &self.cfg;
        assert_eq!(x0.len(), cfg.in_channels * l, "bigvgan input size");

        let mut pins: Vec<GpuView> = Vec::new();
        let mut act_held: Vec<WsBuf<WgpuBackend>> = Vec::new();
        let bufs = self
            .acquire(residency, backend, workspace, &mut pins, &mut act_held)
            .await?;

        // Per-distinct-C depthwise filter buffers + a shared zero-bias buffer.
        let mut filter_pins: Vec<WsBuf<WgpuBackend>> = Vec::new();
        let mut filters: HashMap<u32, FilterBufs> = HashMap::new();
        let mut distinct = cfg.channels();
        distinct.push(cfg.final_ch());
        distinct.sort_unstable();
        distinct.dedup();
        let max_c = *distinct.iter().max().expect("at least one up stage");
        for &c in &distinct {
            let up = tile_filter(workspace, backend, &self.up_filter, c)?;
            let down = tile_filter(workspace, backend, &self.down_filter, c)?;
            filters.insert(
                c,
                FilterBufs {
                    up: up.as_buf_ref(),
                    down: down.as_buf_ref(),
                },
            );
            filter_pins.push(up);
            filter_pins.push(down);
        }
        // Zero bias (bf16) for the no-bias convs (filters + conv_post).
        let zero_bias = workspace.alloc((max_c.max(cfg.final_ch()) as u64) * 2)?;
        backend.write_buffer(
            zero_bias.id(),
            0,
            &act_upload_bytes(ActDtype::Bf16, &vec![0.0f32; max_c as usize]),
        )?;
        let zero_bias_ref = zero_bias.as_buf_ref();

        let in_buf = workspace.alloc((x0.len() * 4) as u64)?;
        backend.write_buffer(in_buf.id(), 0, &act_upload_bytes(ActDtype::F32, x0))?;

        let out_host;
        {
            let pl = pipelines;
            // Optional k=1 input projection, then conv_pre (k7, pad3).
            let mut cur = {
                let scope = workspace.batch();
                let mut x = Cl {
                    buf: scope.import_copy(in_buf.as_buf_ref()),
                    c: cfg.in_channels as u32,
                    l: l as u32,
                };
                if let Some(ip) = &bufs.in_proj {
                    let cout = cfg
                        .in_proj
                        .as_ref()
                        .expect("in_proj cfg present when its buffers are")
                        .out_channels as u32;
                    x = conv1d(
                        &scope,
                        pl,
                        x,
                        ip.weight,
                        ip.bias.expect("in_proj bias"),
                        cout,
                        1,
                        1,
                        1,
                        0,
                        1,
                    )?;
                }
                let x = conv1d(
                    &scope,
                    pl,
                    x,
                    bufs.conv_pre.weight,
                    bufs.conv_pre.bias.expect("conv_pre bias"),
                    cfg.init_ch as u32,
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

            for (i, &cout) in cfg.channels().iter().enumerate() {
                let rate = cfg.up_rates[i] as u32;
                let upk = cfg.up_kernels[i] as u32;
                let pad_t = (upk - rate) / 2;
                let flt = filters[&cout];
                let xt = {
                    let scope = workspace.batch();
                    let x = cur.import(&scope);
                    let x = conv_transpose1d(
                        &scope,
                        pl,
                        x,
                        bufs.ups[i].weight,
                        bufs.ups[i].bias.expect("up bias"),
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
                }
                drop(xt);
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
            }

            // act_post Activation1d -> conv_post (no bias) -> readback.
            {
                let scope = workspace.batch();
                let flt = filters[&cfg.final_ch()];
                let x = cur.import(&scope);
                let x = activation1d(&scope, pl, x, &bufs.act_post, &flt, zero_bias_ref)?;
                let x = conv1d(
                    &scope,
                    pl,
                    x,
                    bufs.conv_post.weight,
                    zero_bias_ref,
                    cfg.out_channels as u32,
                    7,
                    1,
                    1,
                    3,
                    1,
                )?;
                let out_n = x.n() as usize;
                let out_buf = workspace.alloc(x.bytes())?;
                let dst = scope.import_copy(out_buf.as_buf_ref());
                scope.copy_buffer_to_buffer(x.buf, 0, dst, 0, x.bytes())?;
                scope.submit_void().await?;
                let bytes = backend.read_buffer(out_buf.id(), 0, x.bytes()).await?;
                out_host = act_readback_to_f32(ActDtype::F32, &bytes, out_n);
            }
            drop(cur);
        }
        // Stage buffers stay pooled across scopes (each resblock rents ~100,
        // and its siblings and the next channel reuse them); the arbiter
        // reclaims idle ones under budget pressure. Free them all at the end.
        workspace.drain_pool();
        drop(filter_pins);
        drop(act_held);
        drop(pins);
        Ok(out_host)
    }

    async fn acquire<'r, S: WeightSource>(
        &self,
        res: &'r WeightResidency<S>,
        backend: &WgpuBackend,
        workspace: &Workspace<WgpuBackend>,
        pins: &mut Vec<GpuView<'r>>,
        act_held: &mut Vec<WsBuf<WgpuBackend>>,
    ) -> Result<GenBufs, BigVganError<S::Error>> {
        let gn = &self.handles;
        let in_proj = match &gn.in_proj {
            Some(h) => Some(acq_conv(res, backend, h, pins).await?),
            None => None,
        };
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
            in_proj,
            conv_pre,
            ups,
            resblocks,
            act_post,
            conv_post,
        })
    }
}

/// Tile a 12-tap filter to a depthwise `[C, 1, 12]` bf16 conv weight buffer.
fn tile_filter(
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

pub async fn read_weight_f32<S: WeightSource>(
    backend: &WgpuBackend,
    residency: &WeightResidency<S>,
    id: &str,
    n: usize,
) -> Result<Vec<f32>, BigVganError<S::Error>> {
    let h = register_passthrough(residency, &WeightId(id.into()))?;
    let view = residency.acquire(h, backend).await?;
    let bytes = backend
        .read_buffer(view.buf().id, 0, (n * 2) as u64)
        .await?;
    Ok(act_readback_to_f32(ActDtype::Bf16, &bytes, n))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h3() -> BigVganCfg {
        BigVganCfg {
            prefix: "decoder".into(),
            in_channels: 32,
            out_channels: 1,
            init_ch: 1024,
            up_rates: vec![5, 5, 2, 2, 2, 2, 2],
            up_kernels: vec![9, 9, 4, 4, 4, 4, 4],
            ups_nested: true,
            acts_interleaved: true,
            act_post: "activation_post".into(),
            in_proj: Some(InProj {
                name: "dec_in_proj".into(),
                out_channels: 2048,
            }),
            up_filter: "decoder.resblocks.0.activations.0.upsample.filter".into(),
            down_filter: "decoder.resblocks.0.activations.0.downsample.lowpass.filter".into(),
        }
    }

    fn ltx_main() -> BigVganCfg {
        BigVganCfg {
            prefix: "vocoder.vocoder".into(),
            in_channels: 128,
            out_channels: 2,
            init_ch: 1536,
            up_rates: vec![5, 2, 2, 2, 2, 2],
            up_kernels: vec![11, 4, 4, 4, 4, 4],
            ups_nested: false,
            acts_interleaved: false,
            act_post: "act_post".into(),
            in_proj: None,
            up_filter: "vocoder.vocoder.act_post.upsample.filter".into(),
            down_filter: "vocoder.vocoder.act_post.downsample.lowpass.filter".into(),
        }
    }

    /// The checkpoints nest the same modules differently; getting either wrong
    /// is a load failure, not a silent one, but pin both anyway.
    #[test]
    fn tensor_names_follow_each_checkpoint() {
        let h = h3();
        assert_eq!(h.up_name(0), "decoder.ups.0.0");
        // Interleaved: slot-1 activations are the even indices.
        assert_eq!(h.act_name(3, 0, false), "decoder.resblocks.3.activations.0");
        assert_eq!(h.act_name(3, 0, true), "decoder.resblocks.3.activations.1");
        assert_eq!(h.act_name(3, 2, false), "decoder.resblocks.3.activations.4");

        let l = ltx_main();
        assert_eq!(l.up_name(0), "vocoder.vocoder.ups.0");
        assert_eq!(
            l.act_name(3, 2, false),
            "vocoder.vocoder.resblocks.3.acts1.2"
        );
        assert_eq!(
            l.act_name(3, 2, true),
            "vocoder.vocoder.resblocks.3.acts2.2"
        );
    }

    /// The channel ladder halves per stage and must land on the width the
    /// checkpoint's `conv_post` expects.
    #[test]
    fn channel_ladder_ends_where_conv_post_starts() {
        // H3: conv_post.weight is [1, 8, 7] -> 1024 >> 7 == 8.
        assert_eq!(h3().final_ch(), 8);
        assert_eq!(h3().channels(), vec![512, 256, 128, 64, 32, 16, 8]);
        // LTX main: 1536 >> 6 == 24.
        assert_eq!(ltx_main().final_ch(), 24);
    }

    /// The up rates must multiply to the encoder hop length, or the decoder
    /// emits the wrong sample rate (the LTX `OUTPUT_SR` mistag cost 1.5x
    /// playback speed once already).
    #[test]
    fn h3_up_rates_multiply_to_the_hop_length() {
        assert_eq!(h3().up_rates.iter().product::<usize>(), 800);
        assert_eq!(h3().up_kernels.len(), h3().up_rates.len());
    }
}
