//! LightTAE (`lighttaew2_2`) tiny decoder for Wan2.2-TI2V-5B. An opt-in tiny
//! decoder (lightx2v TAEHV family, `model_type="wan22"`) that trades the real
//! `AutoencoderKLWan` VAE's conv3d bandwidth (the run's dominant cost) for a
//! ~0.4GB per-frame 2D-conv stack. The real VAE stays the parity default; this
//! is selected with `VaeChoice::Tiny`.
//!
//! Source: `third-party/lightx2v/.../hf/{tae.py,wan/vae_tiny.py}`. Full arch +
//! bias/dtype map in `scratch/notes/lighttae_spec.md`.
//!
//! Decode (batched-over-frames): a latent-frame batch `[T, C, H, W]` (N=1)
//! pushed through a flat block sequence:
//!   conv 48->256 -> ReLU -> 3x MemBlock@256
//!   -> [Upsample2x -> TGrow -> conv]x3 (256->128->64->64, 3 MemBlock each)
//!   -> ReLU -> conv 64->12.
//! Every op is a per-frame 2D conv / relu / memcat / nearest-upsample over the
//! batch. The MemBlock "past" is a one-frame causal shift (`memcat`), the ONLY
//! temporal coupling, so the decode tiles over latent-frame chunks (sized from
//! the VRAM budget, `plan_chunk`): each chunk carries every MemBlock's trailing
//! input frame into the next so boundaries see real history (exact, no halo).
//! A clip that fits the budget runs as one chunk, bit-identical to untiled.
//! `TGrow` is a 1x1 conv whose output reshape `[T, sC, H, W] == [sT, C, H, W]`
//! is a pure buffer reinterpret (byte-identical, contiguous), so the two
//! stride-2 TGrows grow T 1->2->4 with no data movement (and the chunk
//! boundary, being a latent-frame split, stays contiguous through it). Three
//! spatial Upsample2x (8x) + the host pixel_shuffle(2) = 16x spatial; two
//! stride-2 TGrows = 4x temporal.
//!
//! Host-side (off the GPU graph): denorm `z*std+mean` + Clamp `tanh(x/3)*3`
//! before upload; clamp[0,1] + pixel_shuffle(2) + trim first 3 frames + `*2-1`
//! after readback. So only conv2d/relu/memcat/upsample dispatch on-device.

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError};
use thinfer_core::cache::KernelKey;
use thinfer_core::ops::{
    ActDtype, AddF32, Conv2dConfig, Conv2dF32, Conv2dOp, MemCatF32, MemCatOp, Op, ReluF32,
    Upsample2dNearestF32, Upsample2dNearestOp, WgslConfig,
};
use thinfer_core::residency::{GpuView, ResidencyError, WeightHandle, WeightResidency};
use thinfer_core::trace::PHASE;
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, ScopePacker, Workspace};
use tracing::Instrument;

use crate::common::loader::{LoadError, register_passthrough};
use crate::wan::vae::{WanVaeConfig, act_bytes_to_f32_vec, f32s_to_act_bytes};

// ============================================================================
// Weight ids / handles / views / bufs
// ============================================================================

/// One conv with an optional bias (TGrow + inter-stage convs are bias-less; the
/// kernel binds a shared zero buffer for them, per conv2d's "bias always
/// required" contract).
#[derive(Clone, Debug)]
pub struct TinyConv {
    pub weight: WeightId,
    pub bias: Option<WeightId>,
}

/// `MemBlock.conv` (`nn.Sequential` idx 0/2/4): three 3x3 convs (all +bias);
/// the channel-cat with the previous frame is the fused `memcat` op, and the
/// skip is identity (decoder MemBlocks are all `C -> C`).
#[derive(Clone, Debug)]
pub struct TinyMemBlock {
    pub conv0: TinyConv, // 2C -> C
    pub conv2: TinyConv, // C -> C
    pub conv4: TinyConv, // C -> C
}

/// LightTAE decoder weights (`decoder.{1..22}` keys). `mb[k]` are the three
/// MemBlocks of stage k (channels 256/128/64); `tgrow[k]`/`inter[k]` are the
/// upsample-stage 1x1 TGrow conv + the channel-reducing 3x3 conv after it.
#[derive(Clone, Debug)]
pub struct TinyDecoderWeights {
    pub conv_in: TinyConv, // idx1  48 -> 256 (+bias)
    pub mb: [[TinyMemBlock; 3]; 3],
    pub tgrow: [TinyConv; 3], // idx7/13/19 .conv (no bias)
    pub inter: [TinyConv; 3], // idx8/14/20 (no bias)
    pub conv_out: TinyConv,   // idx22 64 -> 12 (+bias)
}

fn conv_id(idx: usize, bias: bool) -> TinyConv {
    TinyConv {
        weight: WeightId(format!("decoder.{idx}.weight")),
        bias: bias.then(|| WeightId(format!("decoder.{idx}.bias"))),
    }
}

fn memblock_ids(idx: usize) -> TinyMemBlock {
    let c = |sub: usize| TinyConv {
        weight: WeightId(format!("decoder.{idx}.conv.{sub}.weight")),
        bias: Some(WeightId(format!("decoder.{idx}.conv.{sub}.bias"))),
    };
    TinyMemBlock {
        conv0: c(0),
        conv2: c(2),
        conv4: c(4),
    }
}

/// TGrow `.conv` (1x1, no bias) at decoder indices 7/13/19.
fn tgrow_id(idx: usize) -> TinyConv {
    TinyConv {
        weight: WeightId(format!("decoder.{idx}.conv.weight")),
        bias: None,
    }
}

impl Default for TinyDecoderWeights {
    fn default() -> Self {
        Self::new()
    }
}

impl TinyDecoderWeights {
    pub fn new() -> Self {
        Self {
            conv_in: conv_id(1, true),
            mb: [
                [memblock_ids(3), memblock_ids(4), memblock_ids(5)],
                [memblock_ids(9), memblock_ids(10), memblock_ids(11)],
                [memblock_ids(15), memblock_ids(16), memblock_ids(17)],
            ],
            tgrow: [tgrow_id(7), tgrow_id(13), tgrow_id(19)],
            inter: [conv_id(8, false), conv_id(14, false), conv_id(20, false)],
            conv_out: conv_id(22, true),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TinyConvHandles {
    pub weight: WeightHandle,
    pub bias: Option<WeightHandle>,
}

#[derive(Clone, Copy, Debug)]
pub struct TinyMemBlockHandles {
    pub conv0: TinyConvHandles,
    pub conv2: TinyConvHandles,
    pub conv4: TinyConvHandles,
}

#[derive(Clone, Debug)]
pub struct TinyDecoderHandles {
    pub conv_in: TinyConvHandles,
    pub mb: [[TinyMemBlockHandles; 3]; 3],
    pub tgrow: [TinyConvHandles; 3],
    pub inter: [TinyConvHandles; 3],
    pub conv_out: TinyConvHandles,
}

fn reg_conv<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &TinyConv,
) -> Result<TinyConvHandles, LoadError> {
    Ok(TinyConvHandles {
        weight: register_passthrough(residency, &w.weight)?,
        bias: match &w.bias {
            Some(b) => Some(register_passthrough(residency, b)?),
            None => None,
        },
    })
}

fn reg_memblock<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &TinyMemBlock,
) -> Result<TinyMemBlockHandles, LoadError> {
    Ok(TinyMemBlockHandles {
        conv0: reg_conv(residency, &w.conv0)?,
        conv2: reg_conv(residency, &w.conv2)?,
        conv4: reg_conv(residency, &w.conv4)?,
    })
}

/// Register every LightTAE decoder weight with residency (no GPU upload yet).
pub fn register_decoder_tiny<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &TinyDecoderWeights,
) -> Result<TinyDecoderHandles, LoadError> {
    let mb = |s: usize| -> Result<[TinyMemBlockHandles; 3], LoadError> {
        Ok([
            reg_memblock(residency, &w.mb[s][0])?,
            reg_memblock(residency, &w.mb[s][1])?,
            reg_memblock(residency, &w.mb[s][2])?,
        ])
    };
    let conv = |c: &TinyConv| reg_conv(residency, c);
    Ok(TinyDecoderHandles {
        conv_in: reg_conv(residency, &w.conv_in)?,
        mb: [mb(0)?, mb(1)?, mb(2)?],
        tgrow: [conv(&w.tgrow[0])?, conv(&w.tgrow[1])?, conv(&w.tgrow[2])?],
        inter: [conv(&w.inter[0])?, conv(&w.inter[1])?, conv(&w.inter[2])?],
        conv_out: reg_conv(residency, &w.conv_out)?,
    })
}

// --- views (pin guards) + bufs (post-acquire BufRefs) ---

pub struct TinyConvViews<'a> {
    pub weight: GpuView<'a>,
    pub bias: Option<GpuView<'a>>,
}
pub struct TinyMemBlockViews<'a> {
    pub conv0: TinyConvViews<'a>,
    pub conv2: TinyConvViews<'a>,
    pub conv4: TinyConvViews<'a>,
}
pub struct TinyDecoderViews<'a> {
    pub conv_in: TinyConvViews<'a>,
    pub mb: [[TinyMemBlockViews<'a>; 3]; 3],
    pub tgrow: [TinyConvViews<'a>; 3],
    pub inter: [TinyConvViews<'a>; 3],
    pub conv_out: TinyConvViews<'a>,
}

/// A conv's weight + (optional) bias BufRefs. `bias` is `None` for the
/// bias-less convs; the forward substitutes a shared zero buffer.
#[derive(Clone, Copy, Debug)]
pub struct TinyConvBufs {
    pub weight: BufRef,
    pub bias: Option<BufRef>,
}
#[derive(Clone, Copy, Debug)]
pub struct TinyMemBlockBufs {
    pub conv0: TinyConvBufs,
    pub conv2: TinyConvBufs,
    pub conv4: TinyConvBufs,
}
#[derive(Clone, Debug)]
pub struct TinyDecoderBufs {
    pub conv_in: TinyConvBufs,
    pub mb: [[TinyMemBlockBufs; 3]; 3],
    pub tgrow: [TinyConvBufs; 3],
    pub inter: [TinyConvBufs; 3],
    pub conv_out: TinyConvBufs,
}

impl TinyConvViews<'_> {
    fn bufs(&self) -> TinyConvBufs {
        TinyConvBufs {
            weight: self.weight.buf(),
            bias: self.bias.as_ref().map(|b| b.buf()),
        }
    }
}
impl TinyMemBlockViews<'_> {
    fn bufs(&self) -> TinyMemBlockBufs {
        TinyMemBlockBufs {
            conv0: self.conv0.bufs(),
            conv2: self.conv2.bufs(),
            conv4: self.conv4.bufs(),
        }
    }
}
impl TinyDecoderViews<'_> {
    fn bufs(&self) -> TinyDecoderBufs {
        let mb_stage = |s: &[TinyMemBlockViews; 3]| [s[0].bufs(), s[1].bufs(), s[2].bufs()];
        TinyDecoderBufs {
            conv_in: self.conv_in.bufs(),
            mb: [
                mb_stage(&self.mb[0]),
                mb_stage(&self.mb[1]),
                mb_stage(&self.mb[2]),
            ],
            tgrow: [
                self.tgrow[0].bufs(),
                self.tgrow[1].bufs(),
                self.tgrow[2].bufs(),
            ],
            inter: [
                self.inter[0].bufs(),
                self.inter[1].bufs(),
                self.inter[2].bufs(),
            ],
            conv_out: self.conv_out.bufs(),
        }
    }
}

impl TinyConvHandles {
    async fn acquire<'r, S: WeightSource>(
        &self,
        residency: &'r WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<TinyConvViews<'r>, ResidencyError<S::Error, WgpuError>> {
        Ok(TinyConvViews {
            weight: residency.acquire(self.weight, backend).await?,
            bias: match self.bias {
                Some(b) => Some(residency.acquire(b, backend).await?),
                None => None,
            },
        })
    }
}
impl TinyMemBlockHandles {
    async fn acquire<'r, S: WeightSource>(
        &self,
        residency: &'r WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<TinyMemBlockViews<'r>, ResidencyError<S::Error, WgpuError>> {
        Ok(TinyMemBlockViews {
            conv0: self.conv0.acquire(residency, backend).await?,
            conv2: self.conv2.acquire(residency, backend).await?,
            conv4: self.conv4.acquire(residency, backend).await?,
        })
    }
}
async fn acquire_stage<'r, S: WeightSource>(
    s: &[TinyMemBlockHandles; 3],
    residency: &'r WeightResidency<S>,
    backend: &WgpuBackend,
) -> Result<[TinyMemBlockViews<'r>; 3], ResidencyError<S::Error, WgpuError>> {
    Ok([
        s[0].acquire(residency, backend).await?,
        s[1].acquire(residency, backend).await?,
        s[2].acquire(residency, backend).await?,
    ])
}

async fn acquire_trio<'r, S: WeightSource>(
    s: &[TinyConvHandles; 3],
    residency: &'r WeightResidency<S>,
    backend: &WgpuBackend,
) -> Result<[TinyConvViews<'r>; 3], ResidencyError<S::Error, WgpuError>> {
    Ok([
        s[0].acquire(residency, backend).await?,
        s[1].acquire(residency, backend).await?,
        s[2].acquire(residency, backend).await?,
    ])
}

impl TinyDecoderHandles {
    async fn acquire<'r, S: WeightSource>(
        &self,
        residency: &'r WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<TinyDecoderViews<'r>, ResidencyError<S::Error, WgpuError>> {
        Ok(TinyDecoderViews {
            conv_in: self.conv_in.acquire(residency, backend).await?,
            mb: [
                acquire_stage(&self.mb[0], residency, backend).await?,
                acquire_stage(&self.mb[1], residency, backend).await?,
                acquire_stage(&self.mb[2], residency, backend).await?,
            ],
            tgrow: acquire_trio(&self.tgrow, residency, backend).await?,
            inter: acquire_trio(&self.inter, residency, backend).await?,
            conv_out: self.conv_out.acquire(residency, backend).await?,
        })
    }
}

// ============================================================================
// Compiled pipelines
// ============================================================================

/// Decoder conv tile. The same pipeline serves both 3x3 and 1x1 convs (kernel
/// geometry is uniform-driven, not compile-time). `DEFAULT` (64x64) is a fine
/// first cut; the tiny decoder is not the bottleneck the real VAE conv tuning
/// targeted.
const TINY_CONV_TILE: Conv2dConfig = Conv2dConfig::DEFAULT;

pub struct WanVaeTinyPipelines {
    pub act_dtype: ActDtype,
    pub act_size: u64,
    pub conv2d: thinfer_core::backend::WgpuPipeline,
    pub conv2d_op: Conv2dF32,
    pub relu: thinfer_core::backend::WgpuPipeline,
    pub memcat: thinfer_core::backend::WgpuPipeline,
    pub upsample: thinfer_core::backend::WgpuPipeline,
    pub add: thinfer_core::backend::WgpuPipeline,
}

impl WanVaeTinyPipelines {
    pub async fn compile(backend: &WgpuBackend) -> Result<Self, WgpuError> {
        let act_dtype = if backend.supports_shader_f16() {
            ActDtype::F16
        } else {
            ActDtype::F32
        };
        let cfg = &WgslConfig {
            bf16_quant_writes: false,
            act_dtype,
            weight_dtype: thinfer_core::ops::WeightDtype::Bf16,
        };
        let conv2d_op = Conv2dF32::new(TINY_CONV_TILE);
        Ok(Self {
            act_dtype,
            act_size: act_dtype.bytes_per_elem(),
            conv2d: backend
                .create_pipeline(
                    "wan_vae_tiny_conv2d",
                    &conv2d_op.wgsl(cfg),
                    "main",
                    <Conv2dF32 as Conv2dOp>::layout(),
                )
                .await?,
            conv2d_op,
            relu: backend
                .create_pipeline(
                    "wan_vae_tiny_relu",
                    ReluF32::wgsl(cfg),
                    "main",
                    ReluF32::layout(),
                )
                .await?,
            memcat: backend
                .create_pipeline(
                    "wan_vae_tiny_memcat",
                    &<MemCatF32 as MemCatOp>::wgsl(cfg),
                    "main",
                    <MemCatF32 as MemCatOp>::layout(),
                )
                .await?,
            upsample: backend
                .create_pipeline(
                    "wan_vae_tiny_upsample",
                    <Upsample2dNearestF32 as Upsample2dNearestOp>::wgsl(cfg),
                    "main",
                    <Upsample2dNearestF32 as Upsample2dNearestOp>::layout(),
                )
                .await?,
            add: backend
                .create_pipeline(
                    "wan_vae_tiny_add",
                    AddF32::wgsl(cfg),
                    "main",
                    AddF32::layout(),
                )
                .await?,
        })
    }

    pub fn kernel_keys() -> [KernelKey; 5] {
        let kk = |id: &'static str| KernelKey {
            kernel_id: id,
            hint: String::new(),
        };
        [
            kk(<Conv2dF32 as Conv2dOp>::KERNEL_ID),
            kk(<ReluF32 as Op>::KERNEL_ID),
            kk(<MemCatF32 as MemCatOp>::KERNEL_ID),
            kk(<Upsample2dNearestF32 as Upsample2dNearestOp>::KERNEL_ID),
            kk(<AddF32 as Op>::KERNEL_ID),
        ]
    }
}

// ============================================================================
// Forward driver
// ============================================================================

/// NTCHW (`N=1`) activation shape carried through the decoder.
#[derive(Clone, Copy, Debug)]
struct Shape4 {
    t: u32,
    c: u32,
    h: u32,
    w: u32,
}

impl Shape4 {
    fn elems(&self) -> u32 {
        self.t * self.c * self.h * self.w
    }
    fn bytes(&self, asz: u64) -> u64 {
        self.elems() as u64 * asz
    }
}

#[derive(Debug)]
pub enum WanVaeTinyError {
    Wgpu(WgpuError),
}

impl From<WgpuError> for WanVaeTinyError {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}

fn conv2d_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    b: u32,
    cin: u32,
    cout: u32,
    h_in: u32,
    w_in: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    pad: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    // conv2d struct U: b, cin, cout, h_in, w_in, h_out, w_out, kh, kw, pad_h,
    // pad_w, stride_h, stride_w, _pad0..2.
    let fields: [u32; 16] = [
        b, cin, cout, h_in, w_in, h_out, w_out, kh, kw, pad, pad, 1, 1, 0, 0, 0,
    ];
    let mut bytes = [0u8; 64];
    for (i, v) in fields.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    scope.write_uniform(&bytes)
}

/// `conv2d(x, w) + bias` (stride 1, symmetric pad `(kh-1)/2`), batched over the
/// `T` frames. Bias-less convs pass `zero_bias`.
#[allow(clippy::too_many_arguments)]
fn conv2d_run<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pl: &WanVaeTinyPipelines,
    x: BatchBuf<'wsp>,
    shape: Shape4,
    w: &'wsp TinyConvBufs,
    zero_bias: &'wsp BufRef,
    cout: u32,
    k: u32,
) -> Result<(BatchBuf<'wsp>, Shape4), WgpuError> {
    let pad = (k - 1) / 2;
    let out_shape = Shape4 {
        t: shape.t,
        c: cout,
        h: shape.h,
        w: shape.w,
    };
    let out = scope.alloc(out_shape.bytes(pl.act_size))?;
    let u = conv2d_uniform(
        scope, shape.t, shape.c, cout, shape.h, shape.w, shape.h, shape.w, k, k, pad,
    )?;
    let wb = scope.import(&w.weight);
    let bb = scope.import(w.bias.as_ref().unwrap_or(zero_bias));
    scope.conv2d(
        &pl.conv2d,
        &pl.conv2d_op,
        x,
        wb,
        bb,
        u,
        out,
        cout,
        shape.h * shape.w,
        shape.t,
    )?;
    Ok((out, out_shape))
}

fn relu_run<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pl: &WanVaeTinyPipelines,
    x: BatchBuf<'wsp>,
    shape: Shape4,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let out = scope.alloc(shape.bytes(pl.act_size))?;
    scope.dispatch_op::<ReluF32>(&pl.relu, &[x], out)?;
    Ok(out)
}

fn add_run<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pl: &WanVaeTinyPipelines,
    a: BatchBuf<'wsp>,
    b: BatchBuf<'wsp>,
    shape: Shape4,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let out = scope.alloc(shape.bytes(pl.act_size))?;
    scope.dispatch_op::<AddF32>(&pl.add, &[a, b], out)?;
    Ok(out)
}

/// Per-MemBlock temporal carry for chunked decode. `prev` is the `t == 0`
/// "past" frame `[C, H, W]` (the previous chunk's trailing input frame) bound
/// into `memcat` iff `has_prev`; pass `x` itself with `has_prev = false` for
/// the untiled / first-chunk case. `out`, when set, receives this chunk's
/// trailing input frame for the next chunk (skipped on the last chunk).
struct MemBlockCarry<'wsp> {
    prev: BatchBuf<'wsp>,
    has_prev: bool,
    out: Option<BatchBuf<'wsp>>,
}

/// MemBlock: `ReLU(conv4(ReLU(conv2(ReLU(conv0(cat[x, past]))))) + x)`. `past`
/// is the previous frame (the carry frame at the chunk edge, else in-batch
/// `x[t-1]`, zero when neither), fused into `memcat`. The skip is identity (all
/// decoder MemBlocks are `C -> C`).
fn memblock_forward<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pl: &WanVaeTinyPipelines,
    x: BatchBuf<'wsp>,
    shape: Shape4,
    zero_bias: &'wsp BufRef,
    w: &'wsp TinyMemBlockBufs,
    carry: MemBlockCarry<'wsp>,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let c = shape.c;
    let asz = pl.act_size;
    // Stash this chunk's trailing input frame as the next chunk's causal carry,
    // before memcat consumes x (carry-out reads the same x, no hazard).
    if let Some(out) = carry.out {
        let frame_bytes = c as u64 * shape.h as u64 * shape.w as u64 * asz;
        let src_off = (shape.t as u64 - 1) * frame_bytes;
        scope.copy_buffer_to_buffer(x, src_off, out, 0, frame_bytes)?;
    }
    // memcat -> [T, 2C, H, W].
    let cat_shape = Shape4 {
        t: shape.t,
        c: 2 * c,
        h: shape.h,
        w: shape.w,
    };
    let cat = scope.alloc(cat_shape.bytes(asz))?;
    let mu = {
        // U = { t, c, h, w, has_prev }; padded to the 32-byte uniform min size.
        let fields: [u32; 5] = [shape.t, c, shape.h, shape.w, carry.has_prev as u32];
        let mut bytes = [0u8; 32];
        for (i, v) in fields.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        scope.write_uniform(&bytes)?
    };
    scope.memcat::<MemCatF32>(&pl.memcat, x, mu, cat, carry.prev, cat_shape.elems())?;

    let (h0, s0) = conv2d_run(scope, pl, cat, cat_shape, &w.conv0, zero_bias, c, 3)?;
    let h1 = relu_run(scope, pl, h0, s0)?;
    let (h2, s2) = conv2d_run(scope, pl, h1, s0, &w.conv2, zero_bias, c, 3)?;
    let h3 = relu_run(scope, pl, h2, s2)?;
    let (h4, s4) = conv2d_run(scope, pl, h3, s2, &w.conv4, zero_bias, c, 3)?;
    let summed = add_run(scope, pl, h4, x, s4)?;
    relu_run(scope, pl, summed, s4)
}

/// Nearest 2x spatial upsample over the `T` batch.
fn upsample_run<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pl: &WanVaeTinyPipelines,
    x: BatchBuf<'wsp>,
    shape: Shape4,
) -> Result<(BatchBuf<'wsp>, Shape4), WgpuError> {
    let out_shape = Shape4 {
        t: shape.t,
        c: shape.c,
        h: shape.h * 2,
        w: shape.w * 2,
    };
    let out = scope.alloc(out_shape.bytes(pl.act_size))?;
    let u = {
        let fields: [u32; 4] = [shape.t, shape.c, shape.h, shape.w];
        let mut bytes = [0u8; 16];
        for (i, v) in fields.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        scope.write_uniform(&bytes)?
    };
    scope.upsample2d_nearest::<Upsample2dNearestF32>(&pl.upsample, x, u, out, out_shape.elems())?;
    Ok((out, out_shape))
}

/// Per-submit work cap for the decoder's `ScopePacker` (NOT the VRAM ceiling):
/// keeps each submit under the Windows 2s GPU watchdog at native res while
/// letting small (parity) clips collapse to a single submit. Same rationale as
/// the real VAE's `VAE_SUBMIT_BUDGET_BYTES`.
const TINY_SUBMIT_BUDGET_BYTES: u64 = 128 * 1024 * 1024;

/// Channel count at each upsample stage (LightTAE `n_f = [256, 128, 64, 64]`,
/// decoder-order: conv_in -> 256, then 256->128->64->64).
const STAGE_C: [u32; 3] = [256, 128, 64];
/// TGrow temporal stride per upsample stage: 1, 2, 2 (two stride-2 = 4x time).
const TGROW_STRIDE: [u32; 3] = [1, 2, 2];

/// Min storage-buffer binding offset alignment (WebGPU `minStorageBuffer
/// OffsetAlignment`, 256 on all targets). Each carry region starts here-aligned
/// so a sub-range `BufRef::view` binds at a legal offset.
const STORAGE_ALIGN: u64 = 256;

/// Geometry of the 9-MemBlock causal carry (3 stages x 3 blocks), one trailing
/// input frame per block. Sizes are chunk-independent (a single frame), so the
/// ping-pong carry buffers allocate once for the whole clip. Block `i = 3*s + j`
/// at stage `s` runs at `C = STAGE_C[s]`, spatial `(h_in, w_in) << s` (one 2x
/// upsample per prior stage).
struct CarryGeom {
    off: [u64; 9],
    bytes: [u64; 9],
    total: u64,
}

impl CarryGeom {
    fn new(h_in: usize, w_in: usize, asz: u64) -> Self {
        let mut off = [0u64; 9];
        let mut bytes = [0u64; 9];
        let mut acc = 0u64;
        for (s, &stage_c) in STAGE_C.iter().enumerate() {
            for j in 0..3 {
                let i = s * 3 + j;
                let c = stage_c as u64;
                let (h, w) = ((h_in as u64) << s, (w_in as u64) << s);
                bytes[i] = c * h * w * asz;
                off[i] = acc;
                acc += bytes[i].div_ceil(STORAGE_ALIGN) * STORAGE_ALIGN;
            }
        }
        Self {
            off,
            bytes,
            total: acc,
        }
    }
}

/// Calibrated peak transient working set per latent frame in a chunk, in
/// elements scaled by spatial area: `peak ~ K * chunk_f * h_in * w_in * asz`.
/// The stage-2 MemBlock (`T = 2*chunk_f`, the 4x-upsampled `cat[T, 2C, H, W]`
/// plus its conv intermediates, held two-deep by `ScopePacker::MAX_INFLIGHT`)
/// dominates. Anchored to the measured 576x576x97 untiled peak and verified by
/// the `video_e2e` tiny-VAE budget assertion; `THINFER_VAE_TINY_CHUNK` forces a
/// chunk size for recalibration.
const TINY_PEAK_K: u64 = 81920;
/// Resident LightTAE weights (~0.4GB) + readback/upload staging, reserved out of
/// the budget before sizing the chunk so weights stay resident.
const TINY_WEIGHT_RESERVE: u64 = 512 * 1024 * 1024;

/// Largest latent-frame chunk whose decode working set fits the VRAM budget.
/// Returns `f` (single chunk, bit-identical to the untiled path) when the whole
/// clip fits. `override_chunk` (the e2e equivalence test) wins, then
/// `THINFER_VAE_TINY_CHUNK` (CLI calibration), else the budget-derived size.
fn plan_chunk(
    budget: u64,
    f: usize,
    h_in: usize,
    w_in: usize,
    asz: u64,
    override_chunk: Option<usize>,
) -> usize {
    if let Some(n) = override_chunk {
        return n.clamp(1, f.max(1));
    }
    if let Some(n) = std::env::var_os("THINFER_VAE_TINY_CHUNK")
        .and_then(|v| v.to_str().and_then(|s| s.trim().parse::<usize>().ok()))
    {
        return n.clamp(1, f.max(1));
    }
    let eff_budget = if budget == u64::MAX {
        2 * 1024 * 1024 * 1024
    } else {
        budget
    };
    let workspace_budget = eff_budget.saturating_sub(TINY_WEIGHT_RESERVE);
    let per_lat_frame = (TINY_PEAK_K * h_in as u64 * w_in as u64 * asz).max(1);
    let cf = (workspace_budget / per_lat_frame).max(1);
    cf.min(f.max(1) as u64) as usize
}

pub struct WanVaeTinyDecoder {
    pub pipelines: WanVaeTinyPipelines,
    pub handles: TinyDecoderHandles,
    pub cfg: WanVaeConfig,
}

#[derive(Debug)]
pub enum WanVaeTinyDecodeError<SE: core::fmt::Debug> {
    Forward(WanVaeTinyError),
    Residency(ResidencyError<SE, WgpuError>),
}

impl<SE: core::fmt::Debug> From<WanVaeTinyError> for WanVaeTinyDecodeError<SE> {
    fn from(e: WanVaeTinyError) -> Self {
        Self::Forward(e)
    }
}
impl<SE: core::fmt::Debug> From<WgpuError> for WanVaeTinyDecodeError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Forward(WanVaeTinyError::Wgpu(e))
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for WanVaeTinyDecodeError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

impl WanVaeTinyDecoder {
    /// Decode `latents` (CTHW row-major, `z_dim * f * h_in * w_in` f32,
    /// normalized like the DiT output) into a host video tensor CTHW
    /// `[3, 4*f-3, h_in*16, w_in*16]` f32 in `[-1, 1]`. Mirrors
    /// [`crate::wan::vae::WanVaeDecoder::decode`] so the pipeline picks either
    /// decoder behind one signature.
    pub async fn decode<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &mut Workspace<WgpuBackend>,
        latents: &[f32],
        f: usize,
        h_in: usize,
        w_in: usize,
        chunk_override: Option<usize>,
    ) -> Result<Vec<f32>, WanVaeTinyDecodeError<S::Error>> {
        let cfg = &self.cfg;
        let pl = &self.pipelines;
        let asz = pl.act_size;
        let z_dim = cfg.z_dim;
        let frame_elems = z_dim * h_in * w_in;
        assert_eq!(
            latents.len(),
            frame_elems * f,
            "decode: expected {} latents, got {}",
            frame_elems * f,
            latents.len()
        );
        let pix_c = 3usize;
        let patch = cfg.patch_size; // 2 (trailing pixel-shuffle)
        let spat = cfg.spatial_compression; // 16
        let (h_out, w_out) = (h_in * spat, w_in * spat);
        let t_total = if f == 0 { 0 } else { 4 * f - 3 };
        if f == 0 {
            return Ok(Vec::new());
        }

        let views = self
            .handles
            .acquire(residency, backend)
            .instrument(tracing::debug_span!(target: PHASE, "wan_vae_tiny.acquire"))
            .await?;
        let bufs = views.bufs();

        // Persistent across chunks: shared zero bias for the bias-less convs
        // (cout <= 256; a sub-range [0, cout) is bound per conv).
        let zero_bias_buf = workspace.alloc(256 * asz)?;
        backend.write_buffer(zero_bias_buf.id(), 0, &vec![0u8; (256 * asz) as usize])?;
        let zero_bias = zero_bias_buf.as_buf_ref();

        let packed_c = pix_c * patch * patch; // 12
        let dh = h_in * (spat / patch); // decoder conv output H (8x)
        let dw = w_in * (spat / patch);
        let hw_out = h_out * w_out;
        let mut video = vec![0.0_f32; pix_c * t_total * hw_out];

        // Temporal-chunk plan over latent frames so each chunk's decode working
        // set holds the VRAM budget. memcat is the only temporal coupling
        // (causal depth 1), so chunk boundaries carry each MemBlock's trailing
        // input frame; output frames concatenate exactly (no halo, no seam).
        // A single chunk (whole clip fits) is bit-identical to the untiled path.
        let budget = residency.arbiter().budget_bytes();
        let chunk_f = plan_chunk(budget, f, h_in, w_in, asz, chunk_override);
        let n_chunks = f.div_ceil(chunk_f);
        let geom = CarryGeom::new(h_in, w_in, asz);
        let (carry_a, carry_b) = if n_chunks > 1 {
            (
                Some(workspace.alloc(geom.total)?),
                Some(workspace.alloc(geom.total)?),
            )
        } else {
            (None, None)
        };
        // Backstop (mirrors the Full VAE / DiT step loop): hold all-but-weights
        // free so the arbiter caps weight residency and the chunk workspace
        // cannot push the true peak past the budget.
        if budget != u64::MAX {
            residency.set_transient_reserve(budget.saturating_sub(TINY_WEIGHT_RESERVE));
        }
        if std::env::var_os("THINFER_VAE_TINY_MEM").is_some() {
            eprintln!(
                "[vae_tiny_mem] budget={}MiB chunk_f={chunk_f} n_chunks={n_chunks} carry={}KiB (f={f} h_in={h_in} w_in={w_in} asz={asz})",
                budget / (1024 * 1024),
                geom.total / 1024,
            );
        }

        let mut flip = false;
        for chunk_idx in 0..n_chunks {
            let a = chunk_idx * chunk_f; // first latent frame of this chunk
            let cf = chunk_f.min(f - a);
            let has_prev = chunk_idx > 0;
            let is_last = chunk_idx + 1 == n_chunks;
            // Ping-pong: read the buffer the previous chunk wrote, write the
            // other for the next chunk.
            let (read_carry, write_carry) = if flip {
                (carry_b.as_ref(), carry_a.as_ref())
            } else {
                (carry_a.as_ref(), carry_b.as_ref())
            };

            // Host pre: denorm `z*std+mean` then Clamp `tanh(x/3)*3`, laid out
            // NTCHW `[cf, z_dim, h, w]` (decoder batch order; latents are CTHW).
            let mut clamped = vec![0.0_f32; cf * z_dim * h_in * w_in];
            for c in 0..z_dim {
                let std = cfg.latents_std[c];
                let mean = cfg.latents_mean[c];
                for t in 0..cf {
                    for hw in 0..(h_in * w_in) {
                        let src = (c * f + a + t) * h_in * w_in + hw;
                        let z = latents[src] * std + mean;
                        let v = (z / 3.0).tanh() * 3.0;
                        clamped[(t * z_dim + c) * h_in * w_in + hw] = v;
                    }
                }
            }
            let in_bytes = f32s_to_act_bytes(asz, &clamped);
            let in_buf = workspace.alloc(in_bytes.len() as u64)?;
            backend.write_buffer(in_buf.id(), 0, &in_bytes)?;

            // This chunk's decoder activation `[4cf, 12, h*8, w*8]`.
            let out_frames = 4 * cf; // before the global first-3 temporal trim
            let out_buf = workspace.alloc((out_frames * packed_c * dh * dw) as u64 * asz)?;
            let out_ref = out_buf.as_buf_ref();

            {
                let mut packer = ScopePacker::new(workspace, TINY_SUBMIT_BUDGET_BYTES);
                let mut shape = Shape4 {
                    t: cf as u32,
                    c: z_dim as u32,
                    h: h_in as u32,
                    w: w_in as u32,
                };
                let mut x = packer.scope().import_copy(in_buf.as_buf_ref());

                // conv_in 48->256, ReLU.
                x = packer
                    .advance(&[x], shape_peak(&shape, 256, asz))
                    .await?
                    .pop()
                    .unwrap();
                let (y, s) = conv2d_run(
                    packer.scope(),
                    pl,
                    x,
                    shape,
                    &bufs.conv_in,
                    &zero_bias,
                    256,
                    3,
                )?;
                shape = s;
                x = relu_run(packer.scope(), pl, y, shape)?;

                for stage in 0..3 {
                    let cur_c = STAGE_C[stage];
                    // 3 MemBlocks at this stage's channel count, each carrying
                    // its trailing input frame across chunk boundaries.
                    for (j, mb) in bufs.mb[stage].iter().enumerate() {
                        let i = stage * 3 + j;
                        x = packer.advance(&[x], shape.bytes(asz)).await?.pop().unwrap();
                        let scope = packer.scope();
                        let prev = match read_carry {
                            Some(rc) if has_prev => {
                                scope.import_copy(BufRef::view(rc.id(), geom.off[i], geom.bytes[i]))
                            }
                            // No carry-in: bind x (unread when has_prev = false).
                            _ => x,
                        };
                        let out = write_carry.filter(|_| !is_last).map(|wc| {
                            scope.import_copy(BufRef::view(wc.id(), geom.off[i], geom.bytes[i]))
                        });
                        let carry = MemBlockCarry {
                            prev,
                            has_prev,
                            out,
                        };
                        x = memblock_forward(scope, pl, x, shape, &zero_bias, mb, carry)?;
                    }

                    // Upsample 2x spatial.
                    x = packer
                        .advance(&[x], shape_peak(&shape, cur_c, asz) * 4)
                        .await?
                        .pop()
                        .unwrap();
                    let (y, s) = upsample_run(packer.scope(), pl, x, shape)?;
                    shape = s;
                    x = y;

                    // TGrow: 1x1 conv (cur_c -> cur_c*stride, no bias) then the
                    // reshape `[T, sC, H, W] == [sT, C, H, W]` (pure reinterpret).
                    let stride = TGROW_STRIDE[stage];
                    let tg_cout = cur_c * stride;
                    x = packer
                        .advance(&[x], shape_peak(&shape, tg_cout, asz))
                        .await?
                        .pop()
                        .unwrap();
                    let (y, _) = conv2d_run(
                        packer.scope(),
                        pl,
                        x,
                        shape,
                        &bufs.tgrow[stage],
                        &zero_bias,
                        tg_cout,
                        1,
                    )?;
                    x = y;
                    // Reinterpret the channel growth as temporal growth.
                    shape = Shape4 {
                        t: shape.t * stride,
                        c: cur_c,
                        h: shape.h,
                        w: shape.w,
                    };

                    // Inter-stage conv: cur_c -> next stage C (or 64 for last).
                    let next_c = if stage + 1 < 3 {
                        STAGE_C[stage + 1]
                    } else {
                        64
                    };
                    x = packer
                        .advance(&[x], shape_peak(&shape, next_c, asz))
                        .await?
                        .pop()
                        .unwrap();
                    let (y, s) = conv2d_run(
                        packer.scope(),
                        pl,
                        x,
                        shape,
                        &bufs.inter[stage],
                        &zero_bias,
                        next_c,
                        3,
                    )?;
                    shape = s;
                    x = y;
                }

                // final ReLU -> conv_out 64->12.
                x = packer
                    .advance(&[x], shape_peak(&shape, packed_c as u32, asz))
                    .await?
                    .pop()
                    .unwrap();
                x = relu_run(packer.scope(), pl, x, shape)?;
                let (cout, cout_shape) = conv2d_run(
                    packer.scope(),
                    pl,
                    x,
                    shape,
                    &bufs.conv_out,
                    &zero_bias,
                    packed_c as u32,
                    3,
                )?;

                let dst = packer.scope().import_copy(out_ref);
                packer
                    .scope()
                    .copy_buffer_to_buffer(cout, 0, dst, 0, cout_shape.bytes(asz))?;
                packer
                    .finish_void()
                    .instrument(tracing::debug_span!(target: PHASE, "wan_vae_tiny.decode"))
                    .await
                    .map_err(WanVaeTinyError::from)?;
            }

            // Readback `[4cf, 12, dh, dw]` and run host post into the global
            // video. clamp[0,1] -> pixel_shuffle(2) -> `*2-1`, placing each
            // chunk frame at its global output index (the first 3 global frames
            // are the dropped temporal warmup).
            let host = backend
                .read_buffer(
                    out_ref.id,
                    out_ref.offset,
                    (out_frames * packed_c * dh * dw) as u64 * asz,
                )
                .instrument(tracing::debug_span!(target: PHASE, "wan_vae_tiny.readback"))
                .await?;
            let vals = act_bytes_to_f32_vec(asz, &host);
            for fo in 0..out_frames {
                let global_out = 4 * a + fo;
                if global_out < 3 {
                    continue; // dropped temporal warmup (chunk 0 only)
                }
                let t_out = global_out - 3;
                for cp in 0..packed_c {
                    // pixel_shuffle(2): packed channel cp -> (ch, sh, sw).
                    let ch = cp / (patch * patch);
                    let idx = cp % (patch * patch);
                    let sh = idx / patch;
                    let sw = idx % patch;
                    for hh in 0..dh {
                        for ww in 0..dw {
                            let src = ((fo * packed_c + cp) * dh + hh) * dw + ww;
                            let v = vals[src].clamp(0.0, 1.0) * 2.0 - 1.0;
                            let oh = hh * patch + sh;
                            let ow = ww * patch + sw;
                            let dst = (ch * t_total + t_out) * hw_out + oh * w_out + ow;
                            video[dst] = v;
                        }
                    }
                }
            }

            // Return this chunk's idle buffers to the pool before the next grows
            // it, so the live set stays bounded to one chunk.
            if n_chunks > 1 {
                workspace.drain_pool();
            }
            flip = !flip;
        }
        Ok(video)
    }
}

/// Activation bytes of a `[t, c, h, w]` tensor with a substituted channel count
/// (the `ScopePacker` phase peak before an op that changes C).
fn shape_peak(shape: &Shape4, c: u32, asz: u64) -> u64 {
    shape.t as u64 * c as u64 * shape.h as u64 * shape.w as u64 * asz
}
