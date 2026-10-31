//! MiniMax-H3 video VAE ENCODER, single-frame path: a keyframe (FL2VA) or an
//! image reference (Ref2VA). Ground truth: `AutoencoderKLMiniMaxH3._encode_clip`
//! (one frame, no temporal chunking) over `MiniMaxH3VideoEncoder3d` and the
//! `quant_conv`.
//!
//! The encoder is a causal 3D CNN, but with ONE frame every causal conv sees two
//! zero frames ahead of it, so each `k=3` temporal kernel contributes only its
//! last tap and the temporal strides are no-ops. [`EncoderSource`] therefore
//! serves every conv weight sliced to `[O, I, 1, kh, kw]` (exact), and the
//! graph runs as a 2D CNN on the conv3d op with `T = 1`:
//!
//! - `conv_in` (3 -> 128), then six levels of two resnets (GroupNorm 32 per
//!   frame, silu, 3x3 conv with reflect padding, a 1x1 shortcut where the
//!   channels change) and, on the first four, a stride-2 conv whose input is
//!   reflect-padded by one on the bottom/right only (no conv padding);
//! - `norm_out`, silu, `conv_out` (-> 48), `quant_conv` (1x1);
//! - the posterior MEAN (first 24 channels), normalized by `latents_mean` /
//!   `latents_std` into the DiT's latent space.
//!
//! The reference samples the posterior with a fixed generator; the mean is the
//! deterministic choice (the logvar of a VAE posterior is small).
//!
//! Every tensor is served in bf16, the dtype the reference runs the whole
//! model in. One GPU scope per resnet / downsampler keeps the activation peak
//! to a few full-resolution buffers.

use std::collections::HashMap;

use half::{bf16, f16};

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError, WgpuPipeline};
use thinfer_core::ops::{
    ActDtype, AddF32, Conv3dConfig, Conv3dF32, Conv3dOp, GroupNormF32, GroupNormOp, Op, SiluF32,
    WeightDtype, WgslConfig,
};
use thinfer_core::residency::{GpuView, ResidencyError, WeightHandle, WeightResidency};
use thinfer_core::tensor::{Shape as TensorShape, StorageEncoding};
use thinfer_core::weight::{WeightCatalog, WeightEntry, WeightId, WeightReader, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace, WsBuf};

use crate::common::loader::{LoadError, register_passthrough};
use crate::common::lora::VecReader;
use crate::common::seq::{act_readback_to_f32, act_upload_bytes};

/// Channels per level, the per-level spatial downsampling (the temporal factors
/// are no-ops for one frame), resnets per level, GroupNorm groups / eps.
const CHANNELS: [u32; 6] = [128, 256, 256, 512, 512, 1024];
const SPATIAL_DOWN: [bool; 6] = [true, true, true, true, false, false];
const LAYERS: usize = 2;
const GROUPS: u32 = 32;
const EPS: f32 = 1e-6;
/// `conv_out` / `quant_conv` width: posterior mean then logvar.
const MOMENTS: u32 = 48;
/// Pixel normalization (ImageNet mean / std over [0, 1]).
const PIXEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const PIXEL_STD: [f32; 3] = [0.229, 0.224, 0.225];
/// Reflect padding in the conv3d op (`pad_mode`).
const REFLECT: u32 = 2;

// ============================================================================
// Weights: bf16, conv kernels sliced to their last temporal tap
// ============================================================================

#[derive(Debug)]
pub enum EncoderSourceError<E: core::fmt::Debug> {
    Source(E),
    Read(String),
    Missing(String),
    Encoding(String),
}

/// The VAE file's encoder half (`encoder.*`, `quant_conv.*`) as bf16, every
/// 5-D conv weight `[O, I, kt, kh, kw]` served as `[O, I, 1, kh, kw]` (tap
/// `kt - 1`, the only one a single causally-padded frame reaches).
pub struct EncoderSource<S: WeightSource> {
    inner: S,
    catalog: WeightCatalog,
    /// Served id -> the source tensor's shape.
    shapes: HashMap<WeightId, Vec<usize>>,
}

impl<S: WeightSource> EncoderSource<S> {
    pub fn new(inner: S) -> Self {
        let mut catalog = WeightCatalog::new();
        let mut shapes = HashMap::new();
        for (id, e) in &inner.catalog().entries {
            if !(id.0.starts_with("encoder.") || id.0.starts_with("quant_conv.")) {
                continue;
            }
            let src = e.shape.0.clone();
            let shape = match src[..] {
                [o, i, _, kh, kw] => vec![o, i, 1, kh, kw],
                _ => src.clone(),
            };
            let elems: usize = shape.iter().product();
            catalog.entries.insert(
                id.clone(),
                WeightEntry {
                    offset: 0,
                    size: (elems * 2) as u64,
                    encoding: Some(StorageEncoding::Bf16),
                    encoding_label: "BF16".to_string(),
                    shape: TensorShape(shape),
                },
            );
            shapes.insert(id.clone(), src);
        }
        Self {
            inner,
            catalog,
            shapes,
        }
    }
}

impl<S: WeightSource> WeightSource for EncoderSource<S> {
    type Reader = VecReader;
    type Error = EncoderSourceError<S::Error>;

    fn catalog(&self) -> &WeightCatalog {
        &self.catalog
    }

    async fn open(&self, id: &WeightId) -> Result<Self::Reader, Self::Error> {
        let src = self
            .shapes
            .get(id)
            .ok_or_else(|| EncoderSourceError::Missing(id.0.clone()))?;
        let e = self
            .inner
            .catalog()
            .get(id)
            .ok_or_else(|| EncoderSourceError::Missing(id.0.clone()))?;
        let mut bytes = vec![0u8; e.size as usize];
        let mut reader = self
            .inner
            .open(id)
            .await
            .map_err(EncoderSourceError::Source)?;
        reader
            .read_at(0, &mut bytes)
            .await
            .map_err(|err| EncoderSourceError::Read(format!("{}: {err:?}", id.0)))?;
        let vals: Vec<f32> = match e.encoding {
            Some(StorageEncoding::F16) => bytes
                .chunks_exact(2)
                .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            Some(StorageEncoding::Bf16) => bytes
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            Some(StorageEncoding::F32) => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            other => {
                return Err(EncoderSourceError::Encoding(format!("{}: {other:?}", id.0)));
            }
        };
        let vals = match src[..] {
            // [O, I, kt, kh, kw] -> tap kt - 1 of every (o, i) kernel.
            [o, i, kt, kh, kw] if kt > 1 => {
                let plane = kh * kw;
                let mut out = Vec::with_capacity(o * i * plane);
                for oi in 0..o * i {
                    let base = (oi * kt + kt - 1) * plane;
                    out.extend_from_slice(&vals[base..base + plane]);
                }
                out
            }
            _ => vals,
        };
        let out: Vec<u8> = vals
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect();
        Ok(VecReader::from_vec(out))
    }
}

// ============================================================================
// Handles
// ============================================================================

#[derive(Clone, Copy)]
struct Conv {
    w: WeightHandle,
    b: WeightHandle,
}

#[derive(Clone, Copy)]
struct Norm {
    w: WeightHandle,
    b: WeightHandle,
}

#[derive(Clone, Copy)]
struct Resnet {
    norm1: Norm,
    conv1: Conv,
    norm2: Norm,
    conv2: Conv,
    shortcut: Option<Conv>,
}

struct Level {
    resnets: Vec<Resnet>,
    down: Option<Conv>,
}

/// The encoder's registered weights.
pub struct H3VaeEncoder {
    conv_in: Conv,
    levels: Vec<Level>,
    norm_out: Norm,
    conv_out: Conv,
    quant_conv: Conv,
}

fn pair<S: WeightSource>(
    res: &WeightResidency<S>,
    prefix: &str,
) -> Result<(WeightHandle, WeightHandle), LoadError> {
    Ok((
        register_passthrough(res, &WeightId(format!("{prefix}.weight")))?,
        register_passthrough(res, &WeightId(format!("{prefix}.bias")))?,
    ))
}

fn conv<S: WeightSource>(res: &WeightResidency<S>, prefix: &str) -> Result<Conv, LoadError> {
    let (w, b) = pair(res, prefix)?;
    Ok(Conv { w, b })
}

fn norm<S: WeightSource>(res: &WeightResidency<S>, prefix: &str) -> Result<Norm, LoadError> {
    let (w, b) = pair(res, prefix)?;
    Ok(Norm { w, b })
}

impl H3VaeEncoder {
    /// Register over an [`EncoderSource`]-wrapped residency.
    pub fn new<S: WeightSource>(res: &WeightResidency<S>) -> Result<Self, LoadError> {
        let has = |id: String| res.source().catalog().get(&WeightId(id)).is_some();
        let mut levels = Vec::with_capacity(CHANNELS.len());
        for (i, down) in SPATIAL_DOWN.iter().enumerate() {
            let resnets = (0..LAYERS)
                .map(|b| {
                    let p = format!("encoder.down.{i}.block.{b}");
                    let shortcut = if has(format!("{p}.nin_shortcut.weight")) {
                        Some(conv(res, &format!("{p}.nin_shortcut"))?)
                    } else {
                        None
                    };
                    Ok(Resnet {
                        norm1: norm(res, &format!("{p}.norm1"))?,
                        conv1: conv(res, &format!("{p}.conv1"))?,
                        norm2: norm(res, &format!("{p}.norm2"))?,
                        conv2: conv(res, &format!("{p}.conv2"))?,
                        shortcut,
                    })
                })
                .collect::<Result<Vec<_>, LoadError>>()?;
            let down = if *down {
                Some(conv(res, &format!("encoder.down.{i}.downsample.conv"))?)
            } else {
                None
            };
            levels.push(Level { resnets, down });
        }
        Ok(Self {
            conv_in: conv(res, "encoder.conv_in")?,
            levels,
            norm_out: norm(res, "encoder.norm_out")?,
            conv_out: conv(res, "encoder.conv_out")?,
            quant_conv: conv(res, "quant_conv")?,
        })
    }
}

// ============================================================================
// Pipelines + ops
// ============================================================================

/// Activations are f32 (the encoder runs once per keyframe / reference).
pub struct EncoderPipelines {
    conv3d: Conv3dF32,
    conv3d_pipeline: WgpuPipeline,
    group_norm: WgpuPipeline,
    silu: WgpuPipeline,
    add: WgpuPipeline,
}

impl EncoderPipelines {
    pub async fn compile(backend: &WgpuBackend) -> Result<Self, WgpuError> {
        let cfg = &WgslConfig {
            bf16_quant_writes: false,
            act_dtype: ActDtype::F32,
            weight_dtype: WeightDtype::Bf16,
        };
        let conv3d = Conv3dF32::new(Conv3dConfig::DEFAULT);
        Ok(Self {
            conv3d_pipeline: backend
                .create_pipeline(
                    "h3_vae_enc_conv3d",
                    &conv3d.wgsl(cfg),
                    "main",
                    <Conv3dF32 as Conv3dOp>::layout(),
                )
                .await?,
            conv3d,
            group_norm: backend
                .create_pipeline(
                    "h3_vae_enc_group_norm",
                    <GroupNormF32 as GroupNormOp>::wgsl(cfg),
                    "main",
                    <GroupNormF32 as GroupNormOp>::layout(),
                )
                .await?,
            silu: backend
                .create_pipeline(
                    "h3_vae_enc_silu",
                    SiluF32::wgsl(cfg),
                    "main",
                    SiluF32::layout(),
                )
                .await?,
            add: backend
                .create_pipeline(
                    "h3_vae_enc_add",
                    AddF32::wgsl(cfg),
                    "main",
                    AddF32::layout(),
                )
                .await?,
        })
    }
}

/// `[c, h, w]` of a single-frame activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Dims {
    c: u32,
    h: u32,
    w: u32,
}

impl Dims {
    fn elems(self) -> u32 {
        self.c * self.h * self.w
    }
    fn bytes(self) -> u64 {
        self.elems() as u64 * 4
    }
}

#[derive(Clone, Copy)]
struct ConvBufs {
    w: BufRef,
    b: BufRef,
}

#[derive(Clone, Copy)]
struct NormBufs {
    w: BufRef,
    b: BufRef,
}

/// One single-frame conv: `k x k`, `stride`, reflect padding `pad` on both
/// sides (`pad = 0` with stride 2 is the downsampler: the kernel's reflect of
/// the one-past-the-end row / column is exactly the reference's bottom/right
/// reflect pad of 1). Output `(h + 2 pad + (stride - 1) - k) / stride + 1`
/// rows, i.e. `ceil(h / 2)` for the downsampler.
#[allow(clippy::too_many_arguments)]
fn conv2d<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &EncoderPipelines,
    x: BatchBuf<'w>,
    d: Dims,
    c: ConvBufs,
    cout: u32,
    k: u32,
    stride: u32,
    pad: u32,
) -> Result<(BatchBuf<'w>, Dims), WgpuError> {
    let extra = stride - 1;
    let out_d = Dims {
        c: cout,
        h: (d.h + 2 * pad + extra - k) / stride + 1,
        w: (d.w + 2 * pad + extra - k) / stride + 1,
    };
    let fields: [u32; 20] = [
        1, d.c, cout, 1, d.h, d.w, 1, out_d.h, out_d.w, 1, k, k, 0, pad, pad, 1, stride, stride,
        REFLECT, 0,
    ];
    let bytes: Vec<u8> = fields.iter().flat_map(|v| v.to_le_bytes()).collect();
    let u = scope.write_uniform(&bytes)?;
    let out = scope.alloc(out_d.bytes())?;
    scope.conv3d(
        &pl.conv3d_pipeline,
        &pl.conv3d,
        x,
        scope.import_copy(c.w),
        scope.import_copy(c.b),
        u,
        out,
        cout,
        out_d.h * out_d.w,
        1,
    )?;
    Ok((out, out_d))
}

fn group_norm_silu<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &EncoderPipelines,
    x: BatchBuf<'w>,
    d: Dims,
    n: NormBufs,
) -> Result<BatchBuf<'w>, WgpuError> {
    let mut bytes = Vec::with_capacity(32);
    for v in [1, d.c, GROUPS, d.h, d.w] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes.extend_from_slice(&EPS.to_le_bytes());
    bytes.extend_from_slice(&[0u8; 8]);
    let u = scope.write_uniform(&bytes)?;
    let normed = scope.alloc(d.bytes())?;
    scope.group_norm::<GroupNormF32>(
        &pl.group_norm,
        x,
        scope.import_copy(n.w),
        scope.import_copy(n.b),
        u,
        normed,
        GROUPS,
    )?;
    let out = scope.alloc(d.bytes())?;
    scope.dispatch_op::<SiluF32>(&pl.silu, &[normed], out)?;
    Ok(out)
}

// ============================================================================
// Forward
// ============================================================================

#[derive(Debug)]
pub enum EncodeError<SE: core::fmt::Debug> {
    Wgpu(WgpuError),
    Residency(ResidencyError<SE, WgpuError>),
    Shape(String),
}

impl<SE: core::fmt::Debug> From<WgpuError> for EncodeError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}

impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for EncodeError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

/// Stage outputs a parity test compares, `[C, H', W']` each.
#[derive(Default)]
pub struct EncodeTaps {
    pub conv_in: Vec<f32>,
    pub levels: Vec<Vec<f32>>,
    pub moments: Vec<f32>,
}

/// The posterior mean of one frame in DiT latent space, `[24, H/16, W/16]`.
pub struct EncodedFrame {
    pub latent: Vec<f32>,
    pub height: usize,
    pub width: usize,
}

async fn acquire2<'r, S: WeightSource>(
    res: &'r WeightResidency<S>,
    backend: &WgpuBackend,
    w: WeightHandle,
    b: WeightHandle,
    pins: &mut Vec<GpuView<'r>>,
) -> Result<(BufRef, BufRef), ResidencyError<S::Error, WgpuError>> {
    let wv = res.acquire(w, backend).await?;
    let bv = res.acquire(b, backend).await?;
    let out = (wv.buf(), bv.buf());
    pins.push(wv);
    pins.push(bv);
    Ok(out)
}

async fn acq_conv<'r, S: WeightSource>(
    res: &'r WeightResidency<S>,
    backend: &WgpuBackend,
    c: Conv,
    pins: &mut Vec<GpuView<'r>>,
) -> Result<ConvBufs, ResidencyError<S::Error, WgpuError>> {
    let (w, b) = acquire2(res, backend, c.w, c.b, pins).await?;
    Ok(ConvBufs { w, b })
}

async fn acq_norm<'r, S: WeightSource>(
    res: &'r WeightResidency<S>,
    backend: &WgpuBackend,
    n: Norm,
    pins: &mut Vec<GpuView<'r>>,
) -> Result<NormBufs, ResidencyError<S::Error, WgpuError>> {
    let (w, b) = acquire2(res, backend, n.w, n.b, pins).await?;
    Ok(NormBufs { w, b })
}

/// Copy a scope activation into a workspace buffer that outlives the submit.
fn persist<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    ws: &Workspace<WgpuBackend>,
    x: BatchBuf<'w>,
    d: Dims,
) -> Result<WsBuf<WgpuBackend>, WgpuError> {
    let out = ws.alloc(d.bytes())?;
    let dst = scope.import_copy(out.as_buf_ref());
    scope.copy_buffer_to_buffer(x, 0, dst, 0, d.bytes())?;
    Ok(out)
}

async fn read(
    backend: &WgpuBackend,
    b: &WsBuf<WgpuBackend>,
    d: Dims,
) -> Result<Vec<f32>, WgpuError> {
    let bytes = backend.read_buffer(b.id(), 0, d.bytes()).await?;
    Ok(act_readback_to_f32(
        ActDtype::F32,
        &bytes,
        d.elems() as usize,
    ))
}

/// How [`H3VaeEncoder::encode_frame`] turns a frame into a latent.
#[derive(Clone, Copy, Debug)]
pub struct EncodeOptions {
    /// Spatially tile the encode (the reference default: 256 px tiles, at
    /// least 64 px overlap, blended on the moments). Off = one pass.
    pub tiled: bool,
    pub posterior: Posterior,
}

#[derive(Clone, Copy, Debug)]
pub enum Posterior {
    /// The posterior mean.
    Mean,
    /// `mean + std * noise(seed)`, rounded through f16: what the reference does
    /// for keyframes and image references (seed 42, fresh per image).
    Sample { seed: u64 },
}

impl EncodeOptions {
    /// The reference's keyframe / image-reference recipe.
    pub const REFERENCE: Self = Self {
        tiled: true,
        posterior: Posterior::Sample { seed: 42 },
    };
}

/// The reference encoder's tile size and minimum overlap, in pixels.
const TILE_PIXELS: usize = 256;
const TILE_OVERLAP: usize = 64;
const SPATIAL_RATIO: usize = 16;

impl H3VaeEncoder {
    /// Encode one RGB frame `[3, H, W]` in [0, 1] (H, W multiples of 16) into
    /// DiT latent space, `[24, H/16, W/16]`, normalized by the VAE's
    /// `latents_mean` / `latents_std`. `taps` (untiled only) collects the stage
    /// outputs for a parity test.
    #[allow(clippy::too_many_arguments)]
    pub async fn encode_frame<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        res: &WeightResidency<S>,
        ws: &Workspace<WgpuBackend>,
        pl: &EncoderPipelines,
        rgb: &[f32],
        height: usize,
        width: usize,
        (latents_mean, latents_std): (&[f32], &[f32]),
        opts: EncodeOptions,
        mut taps: Option<&mut EncodeTaps>,
    ) -> Result<EncodedFrame, EncodeError<S::Error>> {
        if rgb.len() != 3 * height * width
            || !height.is_multiple_of(16)
            || !width.is_multiple_of(16)
        {
            return Err(EncodeError::Shape(format!(
                "frame must be [3, H, W] with H, W multiples of 16 (got {} values, {height}x{width})",
                rgb.len()
            )));
        }
        let channels = (MOMENTS / 2) as usize;
        if latents_mean.len() != channels || latents_std.len() != channels {
            return Err(EncodeError::Shape(format!(
                "latent stats must have {channels} channels"
            )));
        }
        let (moments, h, w) = if opts.tiled {
            self.tiled_moments(backend, res, ws, pl, rgb, height, width)
                .await?
        } else {
            self.moments(
                backend,
                res,
                ws,
                pl,
                rgb,
                height,
                width,
                taps.as_deref_mut(),
            )
            .await?
        };
        let hw = h * w;
        let (mean, logvar) = moments.split_at(channels * hw);
        let latent: Vec<f32> = match opts.posterior {
            Posterior::Mean => mean.to_vec(),
            Posterior::Sample { seed } => {
                let noise = crate::ltx::sampler::gaussian_noise(mean.len(), seed);
                mean.iter()
                    .zip(logvar)
                    .zip(noise)
                    .map(|((m, lv), n)| {
                        let x = m + (0.5 * lv.clamp(-30.0, 20.0)).exp() * n;
                        f16::from_f32(x).to_f32()
                    })
                    .collect()
            }
        }
        .into_iter()
        .enumerate()
        .map(|(i, v)| (v - latents_mean[i / hw]) / latents_std[i / hw])
        .collect();
        if let Some(t) = taps {
            t.moments = moments;
        }
        Ok(EncodedFrame {
            latent,
            height: h,
            width: w,
        })
    }

    /// The reference's `_encode_clip` with tiling on: 256 px tiles encoded
    /// independently, their 48-channel moments blended across the overlaps
    /// (raw neighbours, `w = i / extent`) and trimmed. `[48, H/16, W/16]`.
    #[allow(clippy::too_many_arguments)]
    async fn tiled_moments<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        res: &WeightResidency<S>,
        ws: &Workspace<WgpuBackend>,
        pl: &EncoderPipelines,
        rgb: &[f32],
        height: usize,
        width: usize,
    ) -> Result<(Vec<f32>, usize, usize), EncodeError<S::Error>> {
        use crate::minimax::video_vae::{PixelBlock, split_tiles};
        let (ys, ths, y_ov) = split_tiles(height, TILE_PIXELS, TILE_OVERLAP, SPATIAL_RATIO);
        let (xs, tws, x_ov) = split_tiles(width, TILE_PIXELS, TILE_OVERLAP, SPATIAL_RATIO);
        let plane = height * width;
        let mut grid = Vec::with_capacity(ys.len());
        for (&y0, &th) in ys.iter().zip(&ths) {
            let mut row = Vec::with_capacity(xs.len());
            for (&x0, &tw) in xs.iter().zip(&tws) {
                let mut tile = Vec::with_capacity(3 * th * tw);
                for c in 0..3 {
                    for y in y0..y0 + th {
                        let base = c * plane + y * width + x0;
                        tile.extend_from_slice(&rgb[base..base + tw]);
                    }
                }
                let (m, h, w) = self
                    .moments(backend, res, ws, pl, &tile, th, tw, None)
                    .await?;
                // Channels-last, one frame: the decoder's stitch block.
                let mut data = vec![0.0f32; m.len()];
                let c = MOMENTS as usize;
                for (i, v) in m.iter().enumerate() {
                    data[(i % (h * w)) * c + i / (h * w)] = *v;
                }
                row.push(PixelBlock {
                    data,
                    frames: 1,
                    height: h,
                    width: w,
                    channels: c,
                });
            }
            grid.push(row);
        }
        let lat =
            |px: &Vec<usize>| -> Vec<usize> { px.iter().map(|o| o / SPATIAL_RATIO).collect() };
        let (y_ov, x_ov) = (lat(&y_ov), lat(&x_ov));
        let (n_y, n_x) = (grid.len(), grid[0].len());
        let mut rows = Vec::with_capacity(n_y);
        for i in 0..n_y {
            let mut parts = Vec::with_capacity(n_x);
            for j in 0..n_x {
                let mut tile = grid[i][j].clone();
                if i > 0 {
                    tile.blend_head_from(&grid[i - 1][j], y_ov[i - 1], 1);
                }
                if j > 0 {
                    tile.blend_head_from(&grid[i][j - 1], x_ov[j - 1], 2);
                }
                if i + 1 < n_y {
                    tile.truncate_axis(tile.height - y_ov[i], 1);
                }
                if j + 1 < n_x {
                    tile.truncate_axis(tile.width - x_ov[j], 2);
                }
                parts.push(tile);
            }
            rows.push(PixelBlock::concat(parts, 2));
        }
        let out = PixelBlock::concat(rows, 1);
        let (h, w, c) = (out.height, out.width, out.channels);
        let mut moments = vec![0.0f32; out.data.len()];
        for (i, v) in out.data.iter().enumerate() {
            moments[(i % c) * h * w + i / c] = *v;
        }
        Ok((moments, h, w))
    }

    /// One untiled encode: `[48, H/16, W/16]` moments (mean then logvar).
    #[allow(clippy::too_many_arguments)]
    async fn moments<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        res: &WeightResidency<S>,
        ws: &Workspace<WgpuBackend>,
        pl: &EncoderPipelines,
        rgb: &[f32],
        height: usize,
        width: usize,
        mut taps: Option<&mut EncodeTaps>,
    ) -> Result<(Vec<f32>, usize, usize), EncodeError<S::Error>> {
        let plane = height * width;
        let normalized: Vec<f32> = rgb
            .iter()
            .enumerate()
            .map(|(i, v)| (v - PIXEL_MEAN[i / plane]) / PIXEL_STD[i / plane])
            .collect();
        let mut d = Dims {
            c: 3,
            h: height as u32,
            w: width as u32,
        };
        let up = act_upload_bytes(ActDtype::F32, &normalized);
        let mut cur = ws.alloc(up.len() as u64)?;
        backend.write_buffer(cur.id(), 0, &up)?;

        // conv_in.
        {
            let mut pins = Vec::new();
            let c = acq_conv(res, backend, self.conv_in, &mut pins).await?;
            let scope = ws.batch();
            let (y, nd) = conv2d(
                &scope,
                pl,
                scope.import_copy(cur.as_buf_ref()),
                d,
                c,
                CHANNELS[0],
                3,
                1,
                1,
            )?;
            let next = persist(&scope, ws, y, nd)?;
            scope.submit_void().await?;
            cur = next;
            d = nd;
            if let Some(t) = taps.as_deref_mut() {
                t.conv_in = read(backend, &cur, d).await?;
            }
        }

        // Levels: resnets, then the stride-2 downsampler.
        for (li, level) in self.levels.iter().enumerate() {
            for r in &level.resnets {
                let mut pins = Vec::new();
                let norm1 = acq_norm(res, backend, r.norm1, &mut pins).await?;
                let conv1 = acq_conv(res, backend, r.conv1, &mut pins).await?;
                let norm2 = acq_norm(res, backend, r.norm2, &mut pins).await?;
                let conv2 = acq_conv(res, backend, r.conv2, &mut pins).await?;
                let shortcut = match r.shortcut {
                    Some(s) => Some(acq_conv(res, backend, s, &mut pins).await?),
                    None => None,
                };
                let cout = CHANNELS[li];
                let scope = ws.batch();
                let x = scope.import_copy(cur.as_buf_ref());
                let h = group_norm_silu(&scope, pl, x, d, norm1)?;
                let (h, hd) = conv2d(&scope, pl, h, d, conv1, cout, 3, 1, 1)?;
                let h = group_norm_silu(&scope, pl, h, hd, norm2)?;
                let (h, hd) = conv2d(&scope, pl, h, hd, conv2, cout, 3, 1, 1)?;
                let skip = match shortcut {
                    Some(s) => conv2d(&scope, pl, x, d, s, cout, 1, 1, 0)?.0,
                    None => x,
                };
                let sum = scope.alloc(hd.bytes())?;
                scope.dispatch_op::<AddF32>(&pl.add, &[skip, h], sum)?;
                let next = persist(&scope, ws, sum, hd)?;
                scope.submit_void().await?;
                cur = next;
                d = hd;
            }
            if let Some(down) = level.down {
                let mut pins = Vec::new();
                let c = acq_conv(res, backend, down, &mut pins).await?;
                let scope = ws.batch();
                let (y, nd) = conv2d(
                    &scope,
                    pl,
                    scope.import_copy(cur.as_buf_ref()),
                    d,
                    c,
                    d.c,
                    3,
                    2,
                    0,
                )?;
                let next = persist(&scope, ws, y, nd)?;
                scope.submit_void().await?;
                cur = next;
                d = nd;
            }
            if let Some(t) = taps.as_deref_mut() {
                t.levels.push(read(backend, &cur, d).await?);
            }
        }

        // norm_out + silu + conv_out + quant_conv.
        let moments = {
            let mut pins = Vec::new();
            let n = acq_norm(res, backend, self.norm_out, &mut pins).await?;
            let co = acq_conv(res, backend, self.conv_out, &mut pins).await?;
            let qc = acq_conv(res, backend, self.quant_conv, &mut pins).await?;
            let scope = ws.batch();
            let x = scope.import_copy(cur.as_buf_ref());
            let h = group_norm_silu(&scope, pl, x, d, n)?;
            let (h, hd) = conv2d(&scope, pl, h, d, co, MOMENTS, 3, 1, 1)?;
            let (m, md) = conv2d(&scope, pl, h, hd, qc, MOMENTS, 1, 1, 0)?;
            let out = persist(&scope, ws, m, md)?;
            scope.submit_void().await?;
            d = md;
            read(backend, &out, md).await?
        };
        Ok((moments, d.h as usize, d.w as usize))
    }
}
