//! LTX-2.3 latent spatial upscaler x2 (`LatentUpsampler`, dims=3, spatial-only).
//! A latent-space CNN that doubles H,W of the video VAE latent between the two
//! distilled stages: stage 1 denoises at half resolution, this upsamples the
//! latent x2, then stage 2 re-noises + refines at full resolution.
//!
//! Ground truth: `ltx_core/model/upsampler/{model,res_block,pixel_shuffle}.py`.
//! Checkpoint `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` `__metadata__.config`
//! (disk-verified): `in_channels=128, mid_channels=1024, num_blocks_per_stage=4,
//! dims=3, spatial_upsample=true, temporal_upsample=false, rational_resampler=
//! false`. All weights BF16.
//!
//! Graph (un-normalized latent in, B=1, layout `[C,F,H,W]` throughout):
//! ```text
//! initial_conv(k3 128->1024) -> initial_norm(GN32) -> silu
//! -> 4x ResBlock                                  (channels 1024, k3)
//! -> upsampler.0(k3 1024->4096, per-frame) -> pixel_shuffle2d(x2) -> [1024,F,2H,2W]
//! -> 4x post_upsample ResBlock
//! -> final_conv(k3 1024->128)
//! ```
//! ResBlock: `silu(norm2(conv2(silu(norm1(conv1(x))))) + x)`.
//!
//! All convs run through the core `Conv3dF32`. The k3 spatial convs use standard
//! `nn.Conv3d(padding=1)` = SYMMETRIC ZERO pad on every axis; the kernel only
//! front-pads time, so [`zero_pad_time`] pre-pads a zero frame each side (cf the
//! video VAE's `replicate_pad_time`, which differs by padding REPLICATE). The
//! per-frame `upsampler.0` Conv2d is run as a `(1,3,3)` Conv3d: a `[Cout,Cin,3,3]`
//! conv2d weight is byte-identical to `[Cout,Cin,1,3,3]`, so the same buffer feeds
//! the 3D kernel with no time mixing and no transpose. The pixel-shuffle reuses
//! `DepthToSpace3dF32` with `p=[1,2,2]` (matches `PixelShuffleND(2)` channel order).

use thinfer_core::backend::{BufRef, WgpuBackend, WgpuError};
use thinfer_core::ops::{
    ActDtype, AddF32, ConcatTimeF32, ConcatTimeOp, Conv3dConfig, Conv3dF32, Conv3dOp,
    DepthToSpace3dF32, DepthToSpace3dOp, GroupNormF32, GroupNormOp, Op, SiluF32, WeightDtype,
    WgslConfig,
};
use thinfer_core::residency::{GpuView, ResidencyError, WeightHandle, WeightResidency};
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace};

use crate::common::loader::{LoadError, register_passthrough};

/// Fixed config (disk-verified; the distilled checkpoint is the only variant).
pub const IN_CHANNELS: usize = 128;
pub const MID_CHANNELS: usize = 1024;
pub const N_BLOCKS_PER_STAGE: usize = 4;
const GN_GROUPS: u32 = 32;
/// `nn.GroupNorm` default eps.
const GN_EPS: f32 = 1e-5;

// ============================================================================
// Pipelines
// ============================================================================

pub struct LtxUpsamplerPipelines {
    conv3d: Conv3dF32,
    conv3d_pipeline: thinfer_core::backend::WgpuPipeline,
    group_norm: thinfer_core::backend::WgpuPipeline,
    silu: thinfer_core::backend::WgpuPipeline,
    add: thinfer_core::backend::WgpuPipeline,
    depth_to_space: thinfer_core::backend::WgpuPipeline,
    concat_time: thinfer_core::backend::WgpuPipeline,
}

impl LtxUpsamplerPipelines {
    pub async fn compile(backend: &WgpuBackend) -> Result<Self, WgpuError> {
        use thinfer_core::backend::Backend;
        // F32 acts (tightest parity), bf16 safetensors weights (norm + conv both).
        let cfg = &WgslConfig {
            bf16_quant_writes: false,
            act_dtype: ActDtype::F32,
            weight_dtype: WeightDtype::Bf16,
        };
        let conv3d = Conv3dF32::new(Conv3dConfig::DEFAULT);
        Ok(Self {
            conv3d_pipeline: backend
                .create_pipeline(
                    "ltx_upsampler_conv3d",
                    &conv3d.wgsl(cfg),
                    "main",
                    <Conv3dF32 as Conv3dOp>::layout(),
                )
                .await?,
            conv3d,
            group_norm: backend
                .create_pipeline(
                    "ltx_upsampler_group_norm",
                    <GroupNormF32 as GroupNormOp>::wgsl(cfg),
                    "main",
                    <GroupNormF32 as GroupNormOp>::layout(),
                )
                .await?,
            silu: backend
                .create_pipeline(
                    "ltx_upsampler_silu",
                    SiluF32::wgsl(cfg),
                    "main",
                    SiluF32::layout(),
                )
                .await?,
            add: backend
                .create_pipeline(
                    "ltx_upsampler_add",
                    AddF32::wgsl(cfg),
                    "main",
                    AddF32::layout(),
                )
                .await?,
            depth_to_space: backend
                .create_pipeline(
                    "ltx_upsampler_depth_to_space3d",
                    &<DepthToSpace3dF32 as DepthToSpace3dOp>::wgsl(cfg),
                    "main",
                    <DepthToSpace3dF32 as DepthToSpace3dOp>::layout(),
                )
                .await?,
            concat_time: backend
                .create_pipeline(
                    "ltx_upsampler_concat_time",
                    &<ConcatTimeF32 as ConcatTimeOp>::wgsl(cfg),
                    "main",
                    <ConcatTimeF32 as ConcatTimeOp>::layout(),
                )
                .await?,
        })
    }
}

// ============================================================================
// Shapes / uniforms (B=1, [C,T,H,W] row-major f32 acts)
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape {
    c: u32,
    t: u32,
    h: u32,
    w: u32,
}

impl Shape {
    fn elems(&self) -> u32 {
        self.c * self.t * self.h * self.w
    }
    fn bytes(&self) -> u64 {
        self.elems() as u64 * 4
    }
    fn thw(&self) -> u32 {
        self.t * self.h * self.w
    }
}

#[allow(clippy::too_many_arguments)]
fn conv3d_uniform_bytes(
    cin: u32,
    cout: u32,
    t_in: u32,
    h_in: u32,
    w_in: u32,
    t_out: u32,
    h_out: u32,
    w_out: u32,
    ker: (u32, u32, u32),
    pad: (u32, u32, u32),
) -> [u8; 80] {
    let fields: [u32; 20] = [
        1, cin, cout, t_in, h_in, w_in, t_out, h_out, w_out, ker.0, ker.1, ker.2, pad.0, pad.1,
        pad.2, 1, 1, 1, 0, 0,
    ];
    let mut bytes = [0u8; 80];
    for (i, v) in fields.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    bytes
}

#[allow(clippy::too_many_arguments)]
fn concat_time_uniform_bytes(
    c: u32,
    h: u32,
    w: u32,
    a_t: u32,
    b_t: u32,
    a_start: u32,
    a_count: u32,
    b_start: u32,
    b_count: u32,
) -> [u8; 48] {
    let fields: [u32; 12] = [
        1, c, h, w, a_t, b_t, a_start, a_count, b_start, b_count, 0, 0,
    ];
    let mut bytes = [0u8; 48];
    for (i, v) in fields.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn group_norm_uniform_bytes(c: u32, g: u32, h: u32, w: u32, eps: f32) -> [u8; 32] {
    let fields = [1u32, c, g, h, w, eps.to_bits(), 0, 0];
    let mut bytes = [0u8; 32];
    for (i, v) in fields.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn depth_to_space_uniform_bytes(
    cin: u32,
    s: Shape,
    p: [u32; 3],
    cout: u32,
    out: Shape,
) -> [u8; 48] {
    let fields: [u32; 12] = [
        cin, s.t, s.h, s.w, p[0], p[1], p[2], 0, cout, out.t, out.h, out.w,
    ];
    let mut bytes = [0u8; 48];
    for (i, v) in fields.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    bytes
}

// ============================================================================
// Op wrappers (run inside one BatchScope)
// ============================================================================

struct ConvBufs {
    weight: BufRef,
    bias: BufRef,
}

struct NormBufs {
    weight: BufRef,
    bias: BufRef,
}

/// Symmetric ZERO temporal pad by 1: `[C,T,H,W]` -> `[C,T+2,H,W]` = `[0, x, 0]`,
/// via two `concat_time` passes against a zero frame. `zeros` must cover at least
/// one `[C,1,H,W]` frame (its leading `C*H*W` elements are read as the pad frame).
fn zero_pad_time<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &LtxUpsamplerPipelines,
    x: BatchBuf<'w>,
    s: Shape,
    zeros: BufRef,
) -> Result<BatchBuf<'w>, WgpuError> {
    let z = scope.import_copy(zeros);
    // tmp [C,T+1,H,W] = [zero, x].
    let tmp_shape = Shape { t: s.t + 1, ..s };
    let tmp = scope.alloc(tmp_shape.bytes())?;
    let u1 = scope.write_uniform(&concat_time_uniform_bytes(
        s.c, s.h, s.w, 1, s.t, 0, 1, 0, s.t,
    ))?;
    scope.concat_time::<ConcatTimeF32>(&pl.concat_time, z, x, u1, tmp, tmp_shape.elems())?;
    // padded [C,T+2,H,W] = [tmp, zero].
    let pad_shape = Shape { t: s.t + 2, ..s };
    let padded = scope.alloc(pad_shape.bytes())?;
    let z2 = scope.import_copy(zeros);
    let u2 = scope.write_uniform(&concat_time_uniform_bytes(
        s.c,
        s.h,
        s.w,
        s.t + 1,
        1,
        0,
        s.t + 1,
        0,
        1,
    ))?;
    scope.concat_time::<ConcatTimeF32>(&pl.concat_time, tmp, z2, u2, padded, pad_shape.elems())?;
    Ok(padded)
}

/// k3x3x3 conv with symmetric zero pad (standard `nn.Conv3d(padding=1)`):
/// `[s.c,T,H,W]` -> `[cout,T,H,W]`.
fn conv3d_k3<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &LtxUpsamplerPipelines,
    x: BatchBuf<'w>,
    s: Shape,
    w: &ConvBufs,
    cout: u32,
    zeros: BufRef,
) -> Result<(BatchBuf<'w>, Shape), WgpuError> {
    let padded = zero_pad_time(scope, pl, x, s, zeros)?;
    let out_shape = Shape { c: cout, ..s };
    let out = scope.alloc(out_shape.bytes())?;
    let u = scope.write_uniform(&conv3d_uniform_bytes(
        s.c,
        cout,
        s.t + 2,
        s.h,
        s.w,
        s.t,
        s.h,
        s.w,
        (3, 3, 3),
        (0, 1, 1),
    ))?;
    let wb = scope.import_copy(w.weight);
    let bb = scope.import_copy(w.bias);
    scope.conv3d(
        &pl.conv3d_pipeline,
        &pl.conv3d,
        padded,
        wb,
        bb,
        u,
        out,
        cout,
        out_shape.thw(),
        1,
    )?;
    Ok((out, out_shape))
}

/// Per-frame `(1,3,3)` conv (the `upsampler.0` Conv2d run as a no-time-mix
/// Conv3d): `[s.c,T,H,W]` -> `[cout,T,H,W]`. No temporal pad (kt=1).
fn conv3d_k133<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &LtxUpsamplerPipelines,
    x: BatchBuf<'w>,
    s: Shape,
    w: &ConvBufs,
    cout: u32,
) -> Result<(BatchBuf<'w>, Shape), WgpuError> {
    let out_shape = Shape { c: cout, ..s };
    let out = scope.alloc(out_shape.bytes())?;
    let u = scope.write_uniform(&conv3d_uniform_bytes(
        s.c,
        cout,
        s.t,
        s.h,
        s.w,
        s.t,
        s.h,
        s.w,
        (1, 3, 3),
        (0, 1, 1),
    ))?;
    let wb = scope.import_copy(w.weight);
    let bb = scope.import_copy(w.bias);
    scope.conv3d(
        &pl.conv3d_pipeline,
        &pl.conv3d,
        x,
        wb,
        bb,
        u,
        out,
        cout,
        out_shape.thw(),
        1,
    )?;
    Ok((out, out_shape))
}

/// GroupNorm(32) over `[C,F,H,W]` (B=1): collapse F into the spatial reduction
/// so each group reduces over `(C/G)*F*H*W` (matches PyTorch GroupNorm on 5D).
fn group_norm<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &LtxUpsamplerPipelines,
    x: BatchBuf<'w>,
    s: Shape,
    w: &NormBufs,
) -> Result<BatchBuf<'w>, WgpuError> {
    let out = scope.alloc(s.bytes())?;
    let u = scope.write_uniform(&group_norm_uniform_bytes(
        s.c,
        GN_GROUPS,
        s.t * s.h,
        s.w,
        GN_EPS,
    ))?;
    let wb = scope.import_copy(w.weight);
    let bb = scope.import_copy(w.bias);
    scope.group_norm::<GroupNormF32>(&pl.group_norm, x, wb, bb, u, out, GN_GROUPS)?;
    Ok(out)
}

fn silu<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &LtxUpsamplerPipelines,
    x: BatchBuf<'w>,
    s: Shape,
) -> Result<BatchBuf<'w>, WgpuError> {
    let out = scope.alloc(s.bytes())?;
    scope.dispatch_op::<SiluF32>(&pl.silu, &[x], out)?;
    Ok(out)
}

fn add<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &LtxUpsamplerPipelines,
    a: BatchBuf<'w>,
    b: BatchBuf<'w>,
    s: Shape,
) -> Result<BatchBuf<'w>, WgpuError> {
    let out = scope.alloc(s.bytes())?;
    scope.dispatch_op::<AddF32>(&pl.add, &[a, b], out)?;
    Ok(out)
}

/// One ResBlock: `silu(norm2(conv2(silu(norm1(conv1(x))))) + x)`.
fn res_block<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &LtxUpsamplerPipelines,
    x: BatchBuf<'w>,
    s: Shape,
    w: &ResBlockBufs,
    zeros: BufRef,
) -> Result<BatchBuf<'w>, WgpuError> {
    let residual = x;
    let (h, _) = conv3d_k3(scope, pl, x, s, &w.conv1, s.c, zeros)?;
    let h = group_norm(scope, pl, h, s, &w.norm1)?;
    let h = silu(scope, pl, h, s)?;
    let (h, _) = conv3d_k3(scope, pl, h, s, &w.conv2, s.c, zeros)?;
    let h = group_norm(scope, pl, h, s, &w.norm2)?;
    let h = add(scope, pl, h, residual, s)?;
    silu(scope, pl, h, s)
}

/// `PixelShuffleND(2)`: `[4*c, F, H, W]` -> `[c, F, 2H, 2W]` via depth-to-space
/// `p=[1,2,2]` (no temporal factor; row-major `(c, h-factor, w-factor)` channel
/// order matches PyTorch).
fn pixel_shuffle2d<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &LtxUpsamplerPipelines,
    x: BatchBuf<'w>,
    s: Shape,
) -> Result<(BatchBuf<'w>, Shape), WgpuError> {
    let cout = s.c / 4;
    let out_shape = Shape {
        c: cout,
        t: s.t,
        h: s.h * 2,
        w: s.w * 2,
    };
    let out = scope.alloc(out_shape.bytes())?;
    let u = scope.write_uniform(&depth_to_space_uniform_bytes(
        s.c,
        s,
        [1, 2, 2],
        cout,
        out_shape,
    ))?;
    scope.depth_to_space3d::<DepthToSpace3dF32>(
        &pl.depth_to_space,
        x,
        u,
        out,
        out_shape.elems(),
    )?;
    Ok((out, out_shape))
}

// ============================================================================
// Weights / handles
// ============================================================================

fn conv_w(prefix: &str) -> (WeightId, WeightId) {
    (
        WeightId(format!("{prefix}.weight")),
        WeightId(format!("{prefix}.bias")),
    )
}

struct ResBlockW {
    conv1: (WeightId, WeightId),
    norm1: (WeightId, WeightId),
    conv2: (WeightId, WeightId),
    norm2: (WeightId, WeightId),
}

impl ResBlockW {
    fn new(prefix: &str) -> Self {
        Self {
            conv1: conv_w(&format!("{prefix}.conv1")),
            norm1: conv_w(&format!("{prefix}.norm1")),
            conv2: conv_w(&format!("{prefix}.conv2")),
            norm2: conv_w(&format!("{prefix}.norm2")),
        }
    }
}

struct UpsamplerW {
    initial_conv: (WeightId, WeightId),
    initial_norm: (WeightId, WeightId),
    res_blocks: Vec<ResBlockW>,
    upsampler_conv: (WeightId, WeightId),
    post_blocks: Vec<ResBlockW>,
    final_conv: (WeightId, WeightId),
}

impl UpsamplerW {
    fn new() -> Self {
        Self {
            initial_conv: conv_w("initial_conv"),
            initial_norm: conv_w("initial_norm"),
            res_blocks: (0..N_BLOCKS_PER_STAGE)
                .map(|i| ResBlockW::new(&format!("res_blocks.{i}")))
                .collect(),
            upsampler_conv: conv_w("upsampler.0"),
            post_blocks: (0..N_BLOCKS_PER_STAGE)
                .map(|i| ResBlockW::new(&format!("post_upsample_res_blocks.{i}")))
                .collect(),
            final_conv: conv_w("final_conv"),
        }
    }
}

#[derive(Clone, Copy)]
struct ConvH {
    weight: WeightHandle,
    bias: WeightHandle,
}

struct ResBlockH {
    conv1: ConvH,
    norm1: ConvH,
    conv2: ConvH,
    norm2: ConvH,
}

struct UpsamplerH {
    initial_conv: ConvH,
    initial_norm: ConvH,
    res_blocks: Vec<ResBlockH>,
    upsampler_conv: ConvH,
    post_blocks: Vec<ResBlockH>,
    final_conv: ConvH,
}

fn reg_conv<S: WeightSource>(
    res: &WeightResidency<S>,
    ids: &(WeightId, WeightId),
) -> Result<ConvH, LoadError> {
    Ok(ConvH {
        weight: register_passthrough(res, &ids.0)?,
        bias: register_passthrough(res, &ids.1)?,
    })
}

fn reg_block<S: WeightSource>(
    res: &WeightResidency<S>,
    w: &ResBlockW,
) -> Result<ResBlockH, LoadError> {
    Ok(ResBlockH {
        conv1: reg_conv(res, &w.conv1)?,
        norm1: reg_conv(res, &w.norm1)?,
        conv2: reg_conv(res, &w.conv2)?,
        norm2: reg_conv(res, &w.norm2)?,
    })
}

impl UpsamplerH {
    fn register<S: WeightSource>(
        res: &WeightResidency<S>,
        w: &UpsamplerW,
    ) -> Result<Self, LoadError> {
        Ok(Self {
            initial_conv: reg_conv(res, &w.initial_conv)?,
            initial_norm: reg_conv(res, &w.initial_norm)?,
            res_blocks: w
                .res_blocks
                .iter()
                .map(|b| reg_block(res, b))
                .collect::<Result<_, _>>()?,
            upsampler_conv: reg_conv(res, &w.upsampler_conv)?,
            post_blocks: w
                .post_blocks
                .iter()
                .map(|b| reg_block(res, b))
                .collect::<Result<_, _>>()?,
            final_conv: reg_conv(res, &w.final_conv)?,
        })
    }
}

// Acquired (pinned) GPU buffers ------------------------------------------------

struct ResBlockBufs {
    conv1: ConvBufs,
    norm1: NormBufs,
    conv2: ConvBufs,
    norm2: NormBufs,
}

struct UpsamplerBufs {
    initial_conv: ConvBufs,
    initial_norm: NormBufs,
    res_blocks: Vec<ResBlockBufs>,
    upsampler_conv: ConvBufs,
    post_blocks: Vec<ResBlockBufs>,
    final_conv: ConvBufs,
}

async fn acquire_conv<'r, S: WeightSource>(
    res: &'r WeightResidency<S>,
    backend: &WgpuBackend,
    h: ConvH,
    pins: &mut Vec<GpuView<'r>>,
) -> Result<ConvBufs, ResidencyError<S::Error, WgpuError>> {
    let wv = res.acquire(h.weight, backend).await?;
    let bv = res.acquire(h.bias, backend).await?;
    let bufs = ConvBufs {
        weight: wv.buf(),
        bias: bv.buf(),
    };
    pins.push(wv);
    pins.push(bv);
    Ok(bufs)
}

async fn acquire_norm<'r, S: WeightSource>(
    res: &'r WeightResidency<S>,
    backend: &WgpuBackend,
    h: ConvH,
    pins: &mut Vec<GpuView<'r>>,
) -> Result<NormBufs, ResidencyError<S::Error, WgpuError>> {
    let c = acquire_conv(res, backend, h, pins).await?;
    Ok(NormBufs {
        weight: c.weight,
        bias: c.bias,
    })
}

async fn acquire_block<'r, S: WeightSource>(
    res: &'r WeightResidency<S>,
    backend: &WgpuBackend,
    h: &ResBlockH,
    pins: &mut Vec<GpuView<'r>>,
) -> Result<ResBlockBufs, ResidencyError<S::Error, WgpuError>> {
    Ok(ResBlockBufs {
        conv1: acquire_conv(res, backend, h.conv1, pins).await?,
        norm1: acquire_norm(res, backend, h.norm1, pins).await?,
        conv2: acquire_conv(res, backend, h.conv2, pins).await?,
        norm2: acquire_norm(res, backend, h.norm2, pins).await?,
    })
}

// ============================================================================
// Upsampler
// ============================================================================

pub struct LtxUpsampler {
    pub pipelines: LtxUpsamplerPipelines,
    handles: UpsamplerH,
}

#[derive(Debug)]
pub enum LtxUpsamplerError<SE: core::fmt::Debug> {
    Wgpu(WgpuError),
    Load(LoadError),
    Residency(ResidencyError<SE, WgpuError>),
}
impl<SE: core::fmt::Debug> From<WgpuError> for LtxUpsamplerError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}
impl<SE: core::fmt::Debug> From<LoadError> for LtxUpsamplerError<SE> {
    fn from(e: LoadError) -> Self {
        Self::Load(e)
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for LtxUpsamplerError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

impl LtxUpsampler {
    /// Register every conv/norm weight against `residency`.
    pub fn new<S: WeightSource>(
        pipelines: LtxUpsamplerPipelines,
        residency: &WeightResidency<S>,
    ) -> Result<Self, LoadError> {
        let handles = UpsamplerH::register(residency, &UpsamplerW::new())?;
        Ok(Self { pipelines, handles })
    }

    /// Upsample an UN-normalized video latent `[128, F, H, W]` (CTHW row-major)
    /// x2 spatially -> `[128, F, 2H, 2W]`. The caller un-normalizes before and
    /// re-normalizes after via the video VAE per-channel stats (`upsample_video`).
    pub async fn forward<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        latent: &[f32],
        f: usize,
        h: usize,
        w: usize,
    ) -> Result<Vec<f32>, LtxUpsamplerError<S::Error>> {
        use thinfer_core::backend::Backend;
        assert_eq!(
            latent.len(),
            IN_CHANNELS * f * h * w,
            "upsampler latent size"
        );

        let mut pins: Vec<GpuView> = Vec::new();
        let b = self.acquire(residency, backend, &mut pins).await?;

        let in_bytes: Vec<u8> = latent.iter().flat_map(|v| v.to_le_bytes()).collect();
        let in_buf = workspace.alloc(in_bytes.len() as u64)?;
        backend.write_buffer(in_buf.id(), 0, &in_bytes)?;

        // Single zero frame, sized to the largest [C,1,H,W] (post-upsample stage:
        // MID channels at 2H x 2W). Its leading C*H*W elements feed every time pad.
        let zeros_elems = MID_CHANNELS * (2 * h) * (2 * w);
        let zeros_buf = workspace.alloc((zeros_elems * 4) as u64)?;
        backend.write_buffer(zeros_buf.id(), 0, &vec![0u8; zeros_elems * 4])?;

        let out_shape = Shape {
            c: IN_CHANNELS as u32,
            t: f as u32,
            h: (2 * h) as u32,
            w: (2 * w) as u32,
        };
        let out_buf = workspace.alloc(out_shape.bytes())?;
        {
            let scope = workspace.batch();
            let pl = &self.pipelines;
            let zeros = zeros_buf.as_buf_ref();
            let x = scope.import_copy(in_buf.as_buf_ref());
            let s = Shape {
                c: IN_CHANNELS as u32,
                t: f as u32,
                h: h as u32,
                w: w as u32,
            };

            // initial_conv -> initial_norm -> silu.
            let (x, s) = conv3d_k3(
                &scope,
                pl,
                x,
                s,
                &b.initial_conv,
                MID_CHANNELS as u32,
                zeros,
            )?;
            let x = group_norm(&scope, pl, x, s, &b.initial_norm)?;
            let mut x = silu(&scope, pl, x, s)?;

            // pre-upsample res blocks.
            for blk in &b.res_blocks {
                x = res_block(&scope, pl, x, s, blk, zeros)?;
            }

            // upsampler.0 (per-frame k3 conv) -> pixel shuffle x2.
            let (x, s) = conv3d_k133(&scope, pl, x, s, &b.upsampler_conv, 4 * MID_CHANNELS as u32)?;
            let (mut x, s) = pixel_shuffle2d(&scope, pl, x, s)?;

            // post-upsample res blocks.
            for blk in &b.post_blocks {
                x = res_block(&scope, pl, x, s, blk, zeros)?;
            }

            // final_conv -> [128, F, 2H, 2W].
            let (x, fs) = conv3d_k3(&scope, pl, x, s, &b.final_conv, IN_CHANNELS as u32, zeros)?;
            debug_assert_eq!(fs, out_shape);
            let dst = scope.import_copy(out_buf.as_buf_ref());
            scope.copy_buffer_to_buffer(x, 0, dst, 0, out_shape.bytes())?;
            scope.submit_void().await?;
        }
        drop(pins);

        let bytes = backend
            .read_buffer(out_buf.id(), 0, out_shape.bytes())
            .await?;
        Ok(crate::common::seq::act_readback_to_f32(
            ActDtype::F32,
            &bytes,
            out_shape.elems() as usize,
        ))
    }

    async fn acquire<'r, S: WeightSource>(
        &self,
        res: &'r WeightResidency<S>,
        backend: &WgpuBackend,
        pins: &mut Vec<GpuView<'r>>,
    ) -> Result<UpsamplerBufs, ResidencyError<S::Error, WgpuError>> {
        let initial_conv = acquire_conv(res, backend, self.handles.initial_conv, pins).await?;
        let initial_norm = acquire_norm(res, backend, self.handles.initial_norm, pins).await?;
        let mut res_blocks = Vec::with_capacity(self.handles.res_blocks.len());
        for blk in &self.handles.res_blocks {
            res_blocks.push(acquire_block(res, backend, blk, pins).await?);
        }
        let upsampler_conv = acquire_conv(res, backend, self.handles.upsampler_conv, pins).await?;
        let mut post_blocks = Vec::with_capacity(self.handles.post_blocks.len());
        for blk in &self.handles.post_blocks {
            post_blocks.push(acquire_block(res, backend, blk, pins).await?);
        }
        let final_conv = acquire_conv(res, backend, self.handles.final_conv, pins).await?;
        Ok(UpsamplerBufs {
            initial_conv,
            initial_norm,
            res_blocks,
            upsampler_conv,
            post_blocks,
            final_conv,
        })
    }
}
