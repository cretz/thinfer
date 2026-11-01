//! MiniMax-H3 video VAE, DECODER only.
//!
//! t2va never encodes: keyframe conditioning is what needs the CNN encoder, and
//! that is deferred with the rest of FL2VA. What we need is the decoder, and it
//! is not a CNN at all: it is a non-causal **36-layer ViT at dim 2048**. Every
//! latent voxel becomes one token and a single output projection expands each
//! token straight into a `patch_size_t x patch_size x patch_size` pixel block,
//! doing all 16x spatial and 4x temporal upsampling at once.
//!
//! That makes this module a block stack, not a conv tower, and it shares all
//! three of the DiT's checkpoint quirks:
//!
//! - `attn.to_qkv` is per-head interleaved -> `QkvSplitHeadwiseF32`
//! - `ff.w1` is a fused `[gate; up]` SwiGLU -> `SiluMulFusedF32`
//! - rope rotates `rope_dim_ratio` (0.75) of each head -> `RopeF32HalfRotPartial`
//!
//! Differences from the DiT block: every projection carries a BIAS, the Q/K
//! RMSNorm is UNAFFINE (no weight in the checkpoint), and each sublayer's output
//! is scaled by a learned per-channel vector (`scale1` / `scale2`) before the
//! residual add, which is a plain `bcast_fma`.
//!
//! **Sequence length is bounded by tiling, not by the clip.** Decode runs over
//! 256x256 pixel tiles with 64-pixel overlap, so one ViT pass sees
//! `16 x 16 x tokens_chunk_size` latent voxels plus 5 suffix tokens (~1285),
//! never the whole 149k-voxel clip.

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError, WgpuPipeline};
use thinfer_core::ops::{
    BcastAddF32, BcastFmaF32, BcastModulateF32, LayerNormF32, QkvSplitHeadwiseF32, QkvSplitOp,
    RopeF32HalfRotPartial, RopeOp, SiluMulFusedF32, SiluMulFusedOp,
};
use thinfer_core::residency::{
    GpuView, ResidencyError, TransposePolicy, WeightHandle, WeightMeta, WeightResidency,
};
use thinfer_core::tensor::Shape;
use thinfer_core::trace::{self, PHASE};
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, GemmEpi, Nvfp4Epi, Workspace, WsBuf};
use tracing::Instrument;

use crate::common::block::{
    ActBuf, Block, BlockPipelines, BlockWgslConfigs, alloc_act, alloc_matmul_out_buf, op_rmsnorm,
    op_sdpa_f16,
};
use crate::common::seq;
use crate::z_image::text_encoder::{LoadError, register_one};

/// Video VAE config, from the upstream `AutoencoderKLLegacy` config. Verified
/// against the shipped fp16 safetensors (2026-08-07).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VideoVaeConfig {
    pub out_channels: usize,
    pub latent_channels: usize,
    pub decoder_layers: usize,
    pub decoder_heads: usize,
    pub decoder_head_dim: usize,
    pub decoder_register_tokens: usize,
    pub decoder_ffn_mult: usize,
    pub decoder_rope_theta: f32,
    /// Fraction of each head's channels that rotate (0.75 -> 48 of 64).
    pub decoder_rope_dim_ratio: f32,
    pub decoder_norm_eps: f32,
    /// Pixel frames the encoder consumed per clip.
    pub clip_length: usize,
    /// Trailing latent frames the encoder dropped from every clip.
    pub token_drop: usize,
    pub spatial_compression: usize,
    pub temporal_compression: usize,
}

impl VideoVaeConfig {
    pub const fn h3() -> Self {
        Self {
            out_channels: 3,
            latent_channels: 24,
            decoder_layers: 36,
            decoder_heads: 32,
            decoder_head_dim: 64,
            decoder_register_tokens: 4,
            decoder_ffn_mult: 4,
            decoder_rope_theta: 100.0,
            decoder_rope_dim_ratio: 0.75,
            decoder_norm_eps: 1e-5,
            clip_length: 17,
            token_drop: 3,
            spatial_compression: 16,
            temporal_compression: 4,
        }
    }

    pub const fn dim(&self) -> usize {
        self.decoder_heads * self.decoder_head_dim
    }

    /// SwiGLU intermediate width. `w1` emits twice this (fused `[gate; up]`).
    pub const fn ffn_inner(&self) -> usize {
        self.dim() * self.decoder_ffn_mult
    }

    /// Rotated channels per head.
    pub fn rotary_dim(&self) -> usize {
        (self.decoder_head_dim as f32 * self.decoder_rope_dim_ratio) as usize
    }

    /// Values one token's output projection expands into.
    pub const fn proj_out_dim(&self) -> usize {
        self.out_channels
            * self.temporal_compression
            * self.spatial_compression
            * self.spatial_compression
    }

    /// Latent frames per decoded clip.
    pub fn tokens_chunk_size(&self) -> usize {
        self.clip_length.div_ceil(self.temporal_compression)
    }

    /// Suffix tokens appended before the stack: learned registers plus one
    /// all-zero `mask_token`. All sit at position 0 and are dropped after.
    pub const fn suffix_tokens(&self) -> usize {
        self.decoder_register_tokens + 1
    }
}

/// Decode tiling. The reference tiles at 256 pixels with 64 overlap; both are
/// multiples of the 16x spatial compression so tile boundaries land on latent
/// voxels exactly.
#[derive(Clone, Copy, Debug)]
pub struct TilePolicy {
    pub tile_pixels: usize,
    pub overlap_pixels: usize,
}

impl Default for TilePolicy {
    fn default() -> Self {
        Self {
            tile_pixels: 256,
            overlap_pixels: 64,
        }
    }
}

/// Lay `tile_size`-wide tiles over `length`, keeping every overlap at least
/// `min_overlap`.
///
/// The tile count is the smallest whose union covers `length`; the leftover
/// slack is distributed round-robin over the overlaps in whole `ratio`
/// (spatial-compression) steps so every tile boundary stays latent-aligned.
/// This is `AutoencoderKLLegacy._split_tiles` verbatim, including the round-
/// robin: an "evenly spread the starts" variant lands on different boundaries
/// and silently changes the stitched output.
///
/// Returns `(starts, lengths, overlaps)` where `overlaps[i]` is the overlap
/// between tile `i` and tile `i + 1`.
pub fn split_tiles(
    length: usize,
    tile_size: usize,
    min_overlap: usize,
    ratio: usize,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    if tile_size >= length {
        return (vec![0], vec![length], vec![]);
    }
    let mut n = length.div_ceil(tile_size);
    while tile_size * n < length + min_overlap * (n - 1) {
        n += 1;
    }
    let mut overlaps = vec![min_overlap; n - 1];
    let remaining = tile_size * n - min_overlap * (n - 1) - length;
    for i in 0..remaining / ratio {
        overlaps[i % (n - 1)] += ratio;
    }
    let mut starts = vec![0usize];
    for i in 0..n - 1 {
        starts.push(starts[i] + tile_size - overlaps[i]);
    }
    (starts, vec![tile_size; n], overlaps)
}

/// 3-axis rotary frequencies for the ViT decoder.
///
/// Coordinates are length-normalized to `[-1, 1)` per axis and scaled by `2*pi`.
/// The `(t, h, w)` angle blocks are concatenated and then duplicated, so pair
/// `k` draws axis `k / freqs_per_axis`. Emits only the distinct
/// `(cos, sin)` entries, matching `RopeF32HalfRotPartial`'s `k % rot_pairs`.
pub fn build_freqs(cfg: &VideoVaeConfig, position_ids: &[[f32; 3]]) -> Vec<f32> {
    let rotary = cfg.rotary_dim();
    let per_axis = rotary / (2 * 3);
    let step = 2.0 * 3.0 / rotary as f64;
    let theta = cfg.decoder_rope_theta as f64;
    let inv: Vec<f64> = (0..per_axis)
        .map(|i| 1.0 / theta.powf(i as f64 * step))
        .collect();
    let two_pi = 2.0 * std::f64::consts::PI;

    let mut out = vec![0.0f32; position_ids.len() * rotary];
    for (row, pos) in position_ids.iter().enumerate() {
        for (axis, &coord) in pos.iter().enumerate() {
            for (j, &f) in inv.iter().enumerate() {
                let angle = two_pi * coord as f64 * f;
                let k = axis * per_axis + j;
                out[row * rotary + 2 * k] = angle.cos() as f32;
                out[row * rotary + 2 * k + 1] = angle.sin() as f32;
            }
        }
    }
    out
}

/// `(t, h, w)` positions of every token of one tile, followed by the suffix
/// tokens (registers + mask), which all sit at the origin.
///
/// Each axis is the cell-centre grid `2 * ((i + 0.5) / size) - 1`.
pub fn build_positions(
    cfg: &VideoVaeConfig,
    frames: usize,
    height: usize,
    width: usize,
) -> Vec<[f32; 3]> {
    let grid = |i: usize, size: usize| 2.0 * ((i as f32 + 0.5) / size as f32) - 1.0;
    let mut out = Vec::with_capacity(frames * height * width + cfg.suffix_tokens());
    for t in 0..frames {
        for h in 0..height {
            for w in 0..width {
                out.push([grid(t, frames), grid(h, height), grid(w, width)]);
            }
        }
    }
    out.extend(std::iter::repeat_n([0.0; 3], cfg.suffix_tokens()));
    out
}

/// Tensor names of one decoder block.
struct BlockWeights {
    norm1: WeightId,
    norm2: WeightId,
    scale1: WeightId,
    scale2: WeightId,
    to_qkv: WeightId,
    to_qkv_bias: WeightId,
    to_out: WeightId,
    to_out_bias: WeightId,
    w1: WeightId,
    w1_bias: WeightId,
    w2: WeightId,
    w2_bias: WeightId,
}

impl BlockWeights {
    fn new(idx: usize) -> Self {
        let p = format!("decoder.transformer_blocks.{idx}");
        let id = |s: &str| WeightId(format!("{p}.{s}"));
        Self {
            norm1: id("norm1.weight"),
            norm2: id("norm2.weight"),
            scale1: id("scale1"),
            scale2: id("scale2"),
            to_qkv: id("attn.to_qkv.weight"),
            to_qkv_bias: id("attn.to_qkv.bias"),
            to_out: id("attn.to_out.weight"),
            to_out_bias: id("attn.to_out.bias"),
            w1: id("ff.w1.weight"),
            w1_bias: id("ff.w1.bias"),
            w2: id("ff.w2.weight"),
            w2_bias: id("ff.w2.bias"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BlockHandles {
    norm1: WeightHandle,
    norm2: WeightHandle,
    scale1: WeightHandle,
    scale2: WeightHandle,
    to_qkv: WeightHandle,
    to_qkv_bias: WeightHandle,
    to_out: WeightHandle,
    to_out_bias: WeightHandle,
    w1: WeightHandle,
    w1_bias: WeightHandle,
    w2: WeightHandle,
    w2_bias: WeightHandle,
    /// The six linears are NVFP4 composites (FastVideo's NVFP4 decoders, see
    /// `fasth3::FastH3Source::vae_decoder`): accelerator-only.
    nvfp4: bool,
}

pub struct BlockViews<'a> {
    norm1: GpuView<'a>,
    norm2: GpuView<'a>,
    to_qkv: GpuView<'a>,
    to_qkv_bias: GpuView<'a>,
    to_out: GpuView<'a>,
    to_out_bias: GpuView<'a>,
    w1: GpuView<'a>,
    w1_bias: GpuView<'a>,
    w2: GpuView<'a>,
    w2_bias: GpuView<'a>,
    nvfp4: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct BlockBufs {
    pub norm1: BufRef,
    pub norm2: BufRef,
    pub to_qkv: BufRef,
    pub to_qkv_bias: BufRef,
    pub to_out: BufRef,
    pub to_out_bias: BufRef,
    pub w1: BufRef,
    pub w1_bias: BufRef,
    pub w2: BufRef,
    pub w2_bias: BufRef,
    pub nvfp4: bool,
}

impl BlockHandles {
    /// OS readahead for this block's matmul weights (see
    /// [`WeightResidency::hint_read`]), issued ahead of its acquire.
    pub async fn hint_read<S: WeightSource>(
        &self,
        residency: &WeightResidency<S>,
    ) -> Result<(), S::Error> {
        for h in [self.to_qkv, self.to_out, self.w1, self.w2] {
            residency.hint_read(h).await?;
        }
        Ok(())
    }

    pub async fn acquire<'a, S: WeightSource>(
        &self,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<BlockViews<'a>, ResidencyError<S::Error, WgpuError>> {
        Ok(BlockViews {
            norm1: residency.acquire(self.norm1, backend).await?,
            norm2: residency.acquire(self.norm2, backend).await?,
            to_qkv: residency.acquire(self.to_qkv, backend).await?,
            to_qkv_bias: residency.acquire(self.to_qkv_bias, backend).await?,
            to_out: residency.acquire(self.to_out, backend).await?,
            to_out_bias: residency.acquire(self.to_out_bias, backend).await?,
            w1: residency.acquire(self.w1, backend).await?,
            w1_bias: residency.acquire(self.w1_bias, backend).await?,
            w2: residency.acquire(self.w2, backend).await?,
            w2_bias: residency.acquire(self.w2_bias, backend).await?,
            nvfp4: self.nvfp4,
        })
    }
}

impl BlockViews<'_> {
    pub fn bufs(&self) -> BlockBufs {
        BlockBufs {
            norm1: self.norm1.buf(),
            norm2: self.norm2.buf(),
            to_qkv: self.to_qkv.buf(),
            to_qkv_bias: self.to_qkv_bias.buf(),
            to_out: self.to_out.buf(),
            to_out_bias: self.to_out_bias.buf(),
            w1: self.w1.buf(),
            w1_bias: self.w1_bias.buf(),
            w2: self.w2.buf(),
            w2_bias: self.w2_bias.buf(),
            nvfp4: self.nvfp4,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DecoderHandles {
    pub blocks: Vec<BlockHandles>,
    pub x_embedder: WeightHandle,
    pub x_embedder_bias: WeightHandle,
    pub register_tokens: WeightHandle,
    /// All-zero in the shipped checkpoint (verified), but read from the file
    /// rather than assumed: it is a trained parameter slot.
    pub mask_token: WeightHandle,
    pub norm_out: WeightHandle,
    pub norm_out_bias: WeightHandle,
    pub proj_out: WeightHandle,
    pub proj_out_bias: WeightHandle,
    pub post_quant_conv: WeightHandle,
    pub post_quant_conv_bias: WeightHandle,
    pub latents_mean: WeightHandle,
    pub latents_std: WeightHandle,
}

/// Register the decoder weights. The CNN encoder half of the file is skipped:
/// t2va never encodes.
pub fn register_handles<S: WeightSource>(
    cfg: &VideoVaeConfig,
    residency: &WeightResidency<S>,
) -> Result<DecoderHandles, LoadError> {
    let lin = |id: &WeightId| register_one(residency, id, TransposePolicy::Linear2D, None);
    let dense = |id: &WeightId| register_one(residency, id, TransposePolicy::None, None);

    let catalog = residency.source().catalog();
    let blocks = (0..cfg.decoder_layers)
        .map(|i| {
            let b = BlockWeights::new(i);
            let nvfp4 = catalog.get(&b.to_qkv).and_then(|e| e.encoding)
                == Some(thinfer_core::tensor::StorageEncoding::Nvfp4);
            Ok(BlockHandles {
                nvfp4,
                norm1: dense(&b.norm1)?,
                norm2: dense(&b.norm2)?,
                scale1: dense(&b.scale1)?,
                scale2: dense(&b.scale2)?,
                to_qkv: lin(&b.to_qkv)?,
                to_qkv_bias: dense(&b.to_qkv_bias)?,
                to_out: lin(&b.to_out)?,
                to_out_bias: dense(&b.to_out_bias)?,
                w1: lin(&b.w1)?,
                w1_bias: dense(&b.w1_bias)?,
                w2: lin(&b.w2)?,
                w2_bias: dense(&b.w2_bias)?,
            })
        })
        .collect::<Result<Vec<_>, LoadError>>()?;

    let w = |s: &str| WeightId(s.to_string());
    Ok(DecoderHandles {
        blocks,
        x_embedder: lin(&w("decoder.x_embedder.weight"))?,
        x_embedder_bias: dense(&w("decoder.x_embedder.bias"))?,
        register_tokens: dense(&w("decoder.register_tokens"))?,
        mask_token: dense(&w("decoder.mask_token"))?,
        norm_out: dense(&w("decoder.norm_out.weight"))?,
        norm_out_bias: dense(&w("decoder.norm_out.bias"))?,
        proj_out: lin(&w("decoder.proj_out.weight"))?,
        proj_out_bias: dense(&w("decoder.proj_out.bias"))?,
        // 1x1x1 conv == a plain linear, but it ships as a rank-5
        // `[C, C, 1, 1, 1]`, which the 2-D transpose would reject.
        post_quant_conv: register_squeezed(residency, &w("post_quant_conv.weight"))?,
        post_quant_conv_bias: dense(&w("post_quant_conv.bias"))?,
        latents_mean: dense(&w("latents_mean"))?,
        latents_std: dense(&w("latents_std"))?,
    })
}

/// Register a weight whose trailing unit axes are decorative, as a 2-D linear.
///
/// `post_quant_conv` is a `Conv3d(C, C, 1)`, so the checkpoint stores
/// `[C, C, 1, 1, 1]`. It is a plain linear and wants the `Linear2D` transpose,
/// but that path is rank-2 only, so squeeze the unit axes at registration.
fn register_squeezed<S: WeightSource>(
    residency: &WeightResidency<S>,
    id: &WeightId,
) -> Result<WeightHandle, LoadError> {
    let entry = residency
        .source()
        .catalog()
        .get(id)
        .ok_or_else(|| LoadError::UnknownWeight(id.clone()))?;
    let encoding = entry.encoding.ok_or_else(|| LoadError::Undecodable {
        id: id.clone(),
        encoding: None,
        label: entry.encoding_label.clone(),
    })?;
    let dims: Vec<usize> = entry.shape.0.iter().copied().filter(|&d| d != 1).collect();
    assert_eq!(
        dims.len(),
        2,
        "{id:?}: expected a 2-D linear once unit axes are squeezed, got {:?}",
        entry.shape.0
    );
    Ok(residency.register(WeightMeta {
        id: id.clone(),
        shape: Shape(dims),
        encoding,
        on_disk_bytes: entry.size,
        transpose: TransposePolicy::Linear2D,
        transcode: None,
    }))
}

/// The decoder's compiled pipelines. Shares the DiT's three H3-specific
/// kernels; see [`crate::minimax::dit::DitPipelines`] for why each exists.
pub struct DecoderPipelines {
    pub block: BlockPipelines,
    pub qkv_split_headwise: WgpuPipeline,
    pub rope_partial: WgpuPipeline,
    pub silu_mul_fused: WgpuPipeline,
}

/// Pipeline cfgs for the decoder: bf16 acts against bf16 weights (the fp16
/// file uploads as bf16 like every other float source), the mixed-precision
/// SDPA (normed Q/K/V cast to f16). FastVideo itself decodes under fp16
/// autocast; f32 acts kept every matmul and the attention off the CUDA fast
/// paths (~40x slower than the FLOP count warrants), with no parity gain.
pub fn decoder_cfgs() -> BlockWgslConfigs {
    use thinfer_core::ops::{ActDtype, WeightDtype, WgslConfig};
    let ops = WgslConfig {
        bf16_quant_writes: false,
        act_dtype: ActDtype::Bf16,
        weight_dtype: WeightDtype::Bf16,
    };
    BlockWgslConfigs {
        matmul_qkv: ops,
        matmul_qkv_self: ops,
        matmul_proj: ops,
        matmul_ffn_up: ops,
        matmul_ffn_down: ops,
        matmul_adaln: ops,
        ops,
        i8_sdpa: false,
        dense_acts: crate::common::block::DenseActSites {
            qkv: true,
            qkv_self: true,
            proj: true,
            ffn_up: true,
            ffn_down: true,
        },
        coopmat_acts: crate::common::block::CoopmatSites::default(),
        large_d_sdpa: false,
        fast_sdpa: true,
        decode_sdpa: false,
    }
}

impl DecoderPipelines {
    pub async fn compile(backend: &WgpuBackend) -> Result<Self, WgpuError> {
        let cfgs = decoder_cfgs();
        let block = BlockPipelines::compile(backend, &cfgs).await?;
        let qkv_split_headwise = backend
            .create_pipeline(
                "minimax_vae_qkv_split_headwise",
                <QkvSplitHeadwiseF32 as QkvSplitOp>::wgsl(&cfgs.ops),
                "main",
                <QkvSplitHeadwiseF32 as QkvSplitOp>::layout(),
            )
            .await?;
        let rope_partial = backend
            .create_pipeline(
                "minimax_vae_rope_halfrot_partial",
                <RopeF32HalfRotPartial as RopeOp>::wgsl(&cfgs.ops),
                "main",
                <RopeF32HalfRotPartial as RopeOp>::layout(),
            )
            .await?;
        let silu_mul_fused = backend
            .create_pipeline(
                "minimax_vae_silu_mul_fused",
                <SiluMulFusedF32 as SiluMulFusedOp>::wgsl(&cfgs.ops),
                "main",
                <SiluMulFusedF32 as SiluMulFusedOp>::layout(),
            )
            .await?;
        Ok(Self {
            block,
            qkv_split_headwise,
            rope_partial,
            silu_mul_fused,
        })
    }
}

#[derive(Debug)]
pub enum DecodeError<SE: core::fmt::Debug> {
    Wgpu(WgpuError),
    Residency(ResidencyError<SE, WgpuError>),
    /// Fewer latent frames than one chunk plus the encoder's dropped lead-in.
    TooFewLatentFrames {
        got: usize,
        need: usize,
    },
    /// An NVFP4 decoder block the accelerator declined: there is no backend
    /// path for its weights.
    Nvfp4Declined,
}

impl<SE: core::fmt::Debug> From<WgpuError> for DecodeError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}

impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for DecodeError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

/// Decoder parameters the kernels read as ACTIVATIONS, held on the host in f32
/// and uploaded per pass in the act dtype.
///
/// `bcast_fma` / `bcast_modulate` read their broadcast vectors in the pipeline's
/// act dtype, not as weights. The decoder runs f32 acts over bf16 weights, so
/// binding the residency's bf16 `scale1`/`scale2`/`norm_out` buffers directly
/// would reinterpret bf16 pairs as f32 (that bug turned every decode to noise).
pub struct DecoderConsts {
    /// Register tokens then `mask_token`, `suffix_tokens() * dim`. They ride in
    /// the uploaded token buffer rather than costing a per-pass GPU scatter.
    /// `mask_token` is all-zero in the shipped file but is read rather than
    /// assumed: it is a trained parameter slot.
    pub suffix: Vec<f32>,
    /// Per block: the attention and feed-forward residual scales.
    scales: Vec<[Vec<f32>; 2]>,
    /// The output LayerNorm's gain and bias.
    norm_out: [Vec<f32>; 2],
}

/// Read every [`DecoderConsts`] parameter back to the host once.
pub async fn load_consts<S: WeightSource>(
    cfg: &VideoVaeConfig,
    backend: &WgpuBackend,
    residency: &WeightResidency<S>,
    handles: &DecoderHandles,
) -> Result<DecoderConsts, DecodeError<S::Error>> {
    let dim = cfg.dim();
    let read =
        async |handle: WeightHandle, len: usize| -> Result<Vec<f32>, DecodeError<S::Error>> {
            let view = residency.acquire(handle, backend).await?;
            let buf = view.buf();
            let bytes = backend
                .read_buffer(buf.id, buf.offset, (len * 2) as u64)
                .await?;
            Ok(bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect())
        };
    let mut suffix = read(handles.register_tokens, cfg.decoder_register_tokens * dim).await?;
    suffix.extend(read(handles.mask_token, dim).await?);
    debug_assert_eq!(suffix.len(), cfg.suffix_tokens() * dim);
    let mut scales = Vec::with_capacity(handles.blocks.len());
    for b in &handles.blocks {
        scales.push([read(b.scale1, dim).await?, read(b.scale2, dim).await?]);
    }
    let norm_out = [
        read(handles.norm_out, dim).await?,
        read(handles.norm_out_bias, dim).await?,
    ];
    Ok(DecoderConsts {
        suffix,
        scales,
        norm_out,
    })
}

/// Matmul then broadcast the layer's bias over the output rows.
///
/// Every projection in this decoder carries a bias, unlike the DiT's, so the
/// bias add is part of the site rather than an occasional extra.
#[allow(clippy::too_many_arguments)]
fn mm_bias<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    p: &BlockPipelines,
    src: ActBuf<'wsp>,
    w: BufRef,
    bias: BufRef,
    rows: u32,
    out_w: u32,
    in_w: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let out = alloc_matmul_out_buf(scope, p, rows * out_w)?;
    let dims = scope.u32x4_uniform(rows, out_w, in_w, 0)?;
    Block::dispatch_matmul_site(
        scope,
        p,
        src,
        scope.import_copy(w),
        out,
        dims,
        p.matmul_i8_qkv.as_ref(),
        p.dequant_i8_qkv.as_ref(),
        None,
        &p.matmul_qkv,
        &p.matmuls.qkv,
        rows,
        out_w,
        in_w,
    )?;
    // A separate destination: binding one buffer as both the read and the
    // read_write operand of a single dispatch is a wgpu usage conflict.
    let biased = alloc_matmul_out_buf(scope, p, rows * out_w)?;
    let bu = scope.u32x4_uniform(out_w, 0, 0, 0)?;
    scope.bcast_add::<BcastAddF32>(
        &p.bcast_add,
        out,
        scope.import_copy(bias),
        bu,
        biased,
        rows * out_w,
    )?;
    Ok(biased)
}

/// One decoder block: pre-norm attention with a learned per-channel output
/// scale, then pre-norm SwiGLU with its own scale.
///
/// Deltas from the DiT block, all of them checkpoint facts rather than choices:
/// every projection carries a bias, the Q/K RMSNorm is UNAFFINE (there is no
/// weight in the file, hence the ones vector), and the sublayer output is
/// scaled by `scale1` / `scale2` before the residual instead of by an AdaLN
/// gate.
#[allow(clippy::too_many_arguments)]
fn block_forward<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    cfg: &VideoVaeConfig,
    pipelines: &DecoderPipelines,
    x_in: BatchBuf<'wsp>,
    freqs: BatchBuf<'wsp>,
    ones: BatchBuf<'wsp>,
    y_out: BatchBuf<'wsp>,
    bufs: &BlockBufs,
    [scale1_w, scale2_w]: [BatchBuf<'wsp>; 2],
    batch: u32,
    tokens: u32,
) -> Result<(), WgpuError> {
    let p = &pipelines.block;
    let rows = batch * tokens;
    let dim = cfg.dim() as u32;
    let hd = cfg.decoder_head_dim as u32;
    let heads = cfg.decoder_heads as u32;
    let ffn = cfg.ffn_inner() as u32;
    let eps = cfg.decoder_norm_eps;

    let x_in = ActBuf::dense(x_in);
    let norm1_w = scope.import_copy(bufs.norm1);
    let norm2_w = scope.import_copy(bufs.norm2);

    // `x + scale * y`, the learned per-channel residual weighting.
    let scaled_residual = |x: BatchBuf<'wsp>,
                           s: BatchBuf<'wsp>,
                           y: BatchBuf<'wsp>,
                           dst: BatchBuf<'wsp>|
     -> Result<(), WgpuError> {
        let u = scope.u32x4_uniform(dim, 0, 0, 0)?;
        scope.bcast_fma::<BcastFmaF32>(&p.bcast_fma, x, s, y, u, dst, rows * dim)
    };

    // --- attention ---
    let n1 = alloc_act(scope, p, rows, dim)?;
    op_rmsnorm(scope, p, x_in, norm1_w, n1, rows, dim, eps)?;

    let qkv = mm_bias(
        scope,
        p,
        n1,
        bufs.to_qkv,
        bufs.to_qkv_bias,
        rows,
        3 * dim,
        dim,
    )?;
    let q = alloc_act(scope, p, rows, dim)?;
    let k = alloc_act(scope, p, rows, dim)?;
    let v = alloc_act(scope, p, rows, dim)?;
    {
        let u = scope.u32x4_uniform(rows, dim, hd, 0)?;
        scope.qkv_split::<QkvSplitHeadwiseF32>(
            &pipelines.qkv_split_headwise,
            qkv,
            q.data,
            k.data,
            v.data,
            u,
            rows * dim,
        )?;
    }

    // Unaffine per-head RMSNorm: the standard kernel with a ones gain.
    let qn = alloc_act(scope, p, rows * heads, hd)?;
    op_rmsnorm(scope, p, q, ones, qn, rows * heads, hd, eps)?;
    let kn = alloc_act(scope, p, rows * heads, hd)?;
    op_rmsnorm(scope, p, k, ones, kn, rows * heads, hd, eps)?;

    let rot_pairs = (cfg.rotary_dim() / 2) as u32;
    let qr = alloc_act(scope, p, rows, dim)?;
    let u = scope.u32x4_uniform(rows, heads, hd / 2, rot_pairs)?;
    scope.rope::<RopeF32HalfRotPartial>(
        &pipelines.rope_partial,
        qn.data,
        freqs,
        u,
        qr.data,
        rows,
        heads,
        hd / 2,
    )?;
    let kr = alloc_act(scope, p, rows, dim)?;
    let u2 = scope.u32x4_uniform(rows, heads, hd / 2, rot_pairs)?;
    scope.rope::<RopeF32HalfRotPartial>(
        &pipelines.rope_partial,
        kn.data,
        freqs,
        u2,
        kr.data,
        rows,
        heads,
        hd / 2,
    )?;

    let sa = alloc_act(scope, p, rows, dim)?;
    // Every tile is the same length and carries no padding, so no mask.
    let dummy_mask = scope.alloc(4)?;
    op_sdpa_f16(
        scope,
        p,
        qr,
        kr,
        v,
        dummy_mask,
        sa,
        batch,
        tokens,
        tokens,
        heads,
        heads,
        hd,
        1.0 / (cfg.decoder_head_dim as f32).sqrt(),
        0,
    )?;

    let proj = mm_bias(scope, p, sa, bufs.to_out, bufs.to_out_bias, rows, dim, dim)?;
    let after_attn = alloc_act(scope, p, rows, dim)?;
    scaled_residual(x_in.data, scale1_w, proj, after_attn.data)?;

    // --- feed-forward ---
    let n2 = alloc_act(scope, p, rows, dim)?;
    op_rmsnorm(scope, p, after_attn, norm2_w, n2, rows, dim, eps)?;
    let fused = mm_bias(scope, p, n2, bufs.w1, bufs.w1_bias, rows, 2 * ffn, dim)?;
    let gu = alloc_act(scope, p, rows, ffn)?;
    {
        let u = scope.u32x4_uniform(rows, ffn, 0, 0)?;
        scope.silu_mul_fused::<SiluMulFusedF32>(
            &pipelines.silu_mul_fused,
            fused,
            u,
            gu.data,
            rows * ffn,
        )?;
    }
    let down = mm_bias(scope, p, gu, bufs.w2, bufs.w2_bias, rows, dim, ffn)?;
    scaled_residual(after_attn.data, scale2_w, down, y_out)
}

/// Batched within-tile attention as block-sparse 64-row tiles: every tile
/// of a pass is chunked into its own 64-row tiles, and each query tile lists
/// exactly its own tile's key tiles (block-diagonal), which is the batched
/// SDPA's contract (attention never crosses tiles, no mask).
pub(crate) struct TilePlan {
    pub(crate) geometry: WsBuf<WgpuBackend>,
    pub(crate) lists: WsBuf<WgpuBackend>,
    pub(crate) n_tiles: u32,
}

impl TilePlan {
    pub(crate) fn new(
        backend: &WgpuBackend,
        scratch: &Workspace<WgpuBackend>,
        batch: usize,
        tokens: usize,
        heads: usize,
    ) -> Result<Self, WgpuError> {
        let items: Vec<(usize, usize)> = (0..batch).map(|b| (b * tokens, tokens)).collect();
        let g = super::vsa::VsaGeometry::new(super::vsa::VsaTiling::T64, &items, 0, [0, 0, 0]);
        let (t, per) = (g.n_tiles(), tokens.div_ceil(g.tiling.rows));
        let mut lists = vec![0u32; heads * t * t + heads * t];
        for h in 0..heads {
            for i in 0..t {
                let first = i / per * per;
                for j in 0..per {
                    lists[(h * t + i) * t + j] = (first + j) as u32;
                }
                lists[heads * t * t + h * t + i] = per as u32;
            }
        }
        let upload = |bytes: &[u8]| -> Result<WsBuf<WgpuBackend>, WgpuError> {
            let buf = scratch.alloc(bytes.len() as u64)?;
            backend.write_buffer(buf.id(), 0, bytes)?;
            Ok(buf)
        };
        let lists: Vec<u8> = lists.iter().flat_map(|v| v.to_le_bytes()).collect();
        Ok(Self {
            geometry: upload(&g.bytes())?,
            lists: upload(&lists)?,
            n_tiles: t as u32,
        })
    }
}

/// [`block_forward`] as ONE deferred accelerator sequence: no backend round
/// trips between ops, and the caller streams the next block's weights before
/// the scope's submit waits. Same ops and bf16 rounding points (RMSNorm is
/// `NormModulate` with a zero scale and shift, the learned residual scales are
/// `GateAdd`); attention runs as [`TilePlan`]'s block-diagonal tiles. Buffers
/// enter by value so the scope can outlive this pass's locals. `false` =
/// nothing enqueued: run [`block_forward`].
#[allow(clippy::too_many_arguments)]
fn block_forward_accel<'s>(
    scope: &BatchScope<'s, WgpuBackend>,
    cfg: &VideoVaeConfig,
    pipelines: &DecoderPipelines,
    [x_in, freqs, zeros, y_out, scale1, scale2]: [BufRef; 6],
    bufs: &BlockBufs,
    plan: &TilePlan,
    batch: u32,
    tokens: u32,
) -> Result<bool, WgpuError> {
    use thinfer_core::backend::{ElementwiseOp, RowSegments, TileAttnShape};
    let p = &pipelines.block;
    let rows = batch * tokens;
    let dim = cfg.dim() as u32;
    let hd = cfg.decoder_head_dim as u32;
    let heads = cfg.decoder_heads as u32;
    let ffn = cfg.ffn_inner() as u32;
    let eps = cfg.decoder_norm_eps;
    let mm = &p.matmul_qkv;
    let Some(acc) = scope.accelerator() else {
        return Ok(false);
    };
    // `(n, k)` of to_qkv, to_out, w1, w2.
    let shapes = [(3 * dim, dim), (dim, dim), (2 * ffn, dim), (dim, ffn)];
    let gemms_ok = if bufs.nvfp4 {
        shapes
            .iter()
            .all(|&(n, k)| acc.accepts_gemm_nvfp4(rows, n, k).is_some())
    } else {
        shapes.iter().all(|&(n, _)| scope.accepts_gemm(mm, rows, n))
    };
    if p.act_dtype != thinfer_core::ops::ActDtype::Bf16
        || !acc.accepts_elementwise()
        || !acc.accepts_tile_attention(hd)
        || !gemms_ok
        || !scope.accel_begin()?
    {
        return Ok(false);
    }

    // Past this point work is enqueued, so a decline fails the run (no
    // fallback); the scope waits for the enqueued work as it drops.
    let ran = |ok: bool, what: &str| match ok {
        true => Ok(()),
        false => Err(WgpuError::AcceleratorDeclined { op: what.into() }),
    };
    let addr = |b: BatchBuf<'s>| {
        scope
            .accel_addr(b)
            .expect("workspace and weights are accelerator-addressable")
    };
    let at = |r: BufRef| addr(scope.import_copy(r));
    let act = |elems: u32| scope.alloc(elems as u64 * p.act_bytes(1));
    let one = |a: BufRef, b: Option<BufRef>| {
        let mut s = RowSegments {
            n: 1,
            ..Default::default()
        };
        (s.end[0], s.a[0], s.b[0]) = (rows, at(a), b.map_or(0, at));
        s
    };
    // `out = residual + gate * (src @ w + bias)` into `out`, all in the GEMM's
    // epilogue (no separate passes over `rows x n`), dense or NVFP4.
    let mm_into = |src: BatchBuf<'s>,
                   w: BufRef,
                   bias: BufRef,
                   residual: Option<(BatchBuf<'s>, BufRef)>,
                   out: BatchBuf<'s>,
                   n: u32,
                   k: u32| {
        if !bufs.nvfp4 {
            let epi = GemmEpi {
                bias: Some(scope.import_copy(bias)),
                residual: residual.map(|(r, _)| r),
                gate: residual.map(|(_, g)| scope.import_copy(g)),
            };
            ran(
                scope.try_gemm_fused(mm, src, scope.import_copy(w), out, rows, n, k, epi)?,
                "gemm",
            )?;
            return Ok::<_, WgpuError>(());
        }
        let epi = match residual {
            Some((r, g)) => Nvfp4Epi::GateAdd(r, one(g, None)),
            None => Nvfp4Epi::Store,
        };
        let w = scope.import_copy(w);
        let bias = Some(scope.import_copy(bias));
        ran(
            scope.try_gemm_nvfp4_epi(
                src,
                w,
                out,
                rows,
                n,
                k,
                None,
                NVFP4_FORMAT,
                None,
                bias,
                epi,
            )?,
            "nvfp4 gemm",
        )?;
        Ok(())
    };
    let mm_fused = |src: BatchBuf<'s>,
                    w: BufRef,
                    bias: BufRef,
                    residual: Option<(BatchBuf<'s>, BufRef)>,
                    n: u32,
                    k: u32| {
        let out = act(rows * n)?;
        mm_into(src, w, bias, residual, out, n, k)?;
        Ok::<_, WgpuError>(out)
    };

    // --- attention ---
    let n1 = act(rows * dim)?;
    ran(
        scope.accel_elementwise(&ElementwiseOp::NormModulate {
            x: at(x_in),
            w: at(bufs.norm1),
            out: addr(n1),
            rows,
            dim,
            eps,
            seg: one(zeros, Some(zeros)),
        }),
        "norm1",
    )?;
    let qkv = mm_fused(n1, bufs.to_qkv, bufs.to_qkv_bias, None, 3 * dim, dim)?;
    let (q, k, v) = (act(rows * dim)?, act(rows * dim)?, act(rows * dim)?);
    ran(
        scope.accel_elementwise(&ElementwiseOp::QkNormRope {
            // The dense file interleaves q/k/v per head; the NVFP4 composite is
            // `[q; k; v]` slabs (see `FastH3Source::vae_decoder`).
            headwise: !bufs.nvfp4,
            q_only: false,
            qkv: addr(qkv),
            q_norm: 0,
            k_norm: 0,
            freqs: at(freqs),
            q: addr(q),
            k: addr(k),
            v: addr(v),
            rows,
            heads,
            head_dim: hd,
            rot_pairs: (cfg.rotary_dim() / 2) as u32,
            eps,
        }),
        "qk norm + rope",
    )?;
    let sa = act(rows * dim)?;
    ran(
        scope.try_tile_attention(
            &TileAttnShape {
                rows,
                heads,
                head_dim: hd,
                n_tiles: plan.n_tiles,
                scale: 1.0 / (cfg.decoder_head_dim as f32).sqrt(),
            },
            q,
            k,
            v,
            sa,
            scope.import_copy(plan.geometry.as_buf_ref()),
            scope.import_copy(plan.lists.as_buf_ref()),
        )?,
        "tile attention",
    )?;
    let after_attn = mm_fused(
        sa,
        bufs.to_out,
        bufs.to_out_bias,
        Some((scope.import_copy(x_in), scale1)),
        dim,
        dim,
    )?;

    // --- feed-forward ---
    let n2 = act(rows * dim)?;
    ran(
        scope.accel_elementwise(&ElementwiseOp::NormModulate {
            x: addr(after_attn),
            w: at(bufs.norm2),
            out: addr(n2),
            rows,
            dim,
            eps,
            seg: one(zeros, Some(zeros)),
        }),
        "norm2",
    )?;
    let gu = act(rows * ffn)?;
    if bufs.nvfp4 {
        let (w1, bias) = (scope.import_copy(bufs.w1), scope.import_copy(bufs.w1_bias));
        let epi = Nvfp4Epi::SwiGlu;
        ran(
            scope.try_gemm_nvfp4_epi(
                n2,
                w1,
                gu,
                rows,
                ffn,
                dim,
                None,
                NVFP4_FORMAT,
                None,
                Some(bias),
                epi,
            )?,
            "w1 + swiglu",
        )?;
    } else {
        let fused = mm_fused(n2, bufs.w1, bufs.w1_bias, None, 2 * ffn, dim)?;
        ran(
            scope.accel_elementwise(&ElementwiseOp::SiluMulFused {
                x: addr(fused),
                out: addr(gu),
                rows,
                h: ffn,
            }),
            "swiglu",
        )?;
    }
    mm_into(
        gu,
        bufs.w2,
        bufs.w2_bias,
        Some((after_attn, scale2)),
        scope.import_copy(y_out),
        dim,
        ffn,
    )?;
    Ok(true)
}

/// Decode `batch` equally-shaped latent tiles in one batched ViT pass.
///
/// Batching is what makes this affordable: the 36-layer stack is ~4.8 GiB of
/// weights, so a pass per tile would re-stream that for every one of the ~150
/// tiles a 5-second clip needs. Every tile in a pass shares one grid, hence one
/// set of rotary frequencies, and full attention stays WITHIN a tile because
/// sdpa runs at `batch` > 1.
///
/// `latents` is `[batch, frames * height * width, latent_channels]` already
/// denormalized by `latents_mean` / `latents_std`; the result is
/// `[batch, frames * height * width, proj_out_dim]`, ready for [`unpatchify`].
#[allow(clippy::too_many_arguments)]
pub async fn decode_tiles<S: WeightSource>(
    cfg: &VideoVaeConfig,
    backend: &WgpuBackend,
    pipelines: &DecoderPipelines,
    residency: &WeightResidency<S>,
    scratch: &Workspace<WgpuBackend>,
    handles: &DecoderHandles,
    consts: &DecoderConsts,
    latents: &[f32],
    batch: usize,
    frames: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, DecodeError<S::Error>> {
    let passes = [TilePass { latents, batch }];
    decode_passes(
        cfg, backend, pipelines, residency, scratch, handles, consts, &passes, frames, height,
        width,
    )
    .await
}

/// One batched ViT pass: `batch` tiles' latents, `[batch, patches, channels]`.
pub struct TilePass<'a> {
    pub latents: &'a [f32],
    pub batch: usize,
}

/// [`decode_tiles`] over several passes of equally-shaped tiles, LAYER-MAJOR:
/// each block's weights stream once and every pass runs through it before the
/// next block, instead of re-streaming the ~4.8 GiB stack per pass. Passes are
/// independent (attention stays within a tile), so this is exactly the
/// pass-by-pass result; it only costs keeping every pass's activations live.
/// Returns every pass's tile tokens, concatenated in pass order.
#[allow(clippy::too_many_arguments)]
pub async fn decode_passes<S: WeightSource>(
    cfg: &VideoVaeConfig,
    backend: &WgpuBackend,
    pipelines: &DecoderPipelines,
    residency: &WeightResidency<S>,
    scratch: &Workspace<WgpuBackend>,
    handles: &DecoderHandles,
    consts: &DecoderConsts,
    passes: &[TilePass<'_>],
    frames: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, DecodeError<S::Error>> {
    let p = &pipelines.block;
    let act = p.act_dtype;
    let dim = cfg.dim();
    let lat_c = cfg.latent_channels;
    let n_patches = frames * height * width;
    let tokens = n_patches + cfg.suffix_tokens();

    // Rotary frequencies are per-token and identical across tiles, so a
    // pass's buffer is the one grid repeated `batch` times.
    let positions = build_positions(cfg, frames, height, width);
    let one_tile = build_freqs(cfg, &positions);

    // The unaffine Q/K norm's gain. `rmsnorm` reads its gain as a WEIGHT (the
    // decoder's bf16 weight dtype), not as an act: f32 bytes here would read
    // back as alternating 0 / 1 and zero half of every head.
    let ones_bytes: Vec<u8> = (0..cfg.decoder_head_dim)
        .flat_map(|_| half::bf16::ONE.to_le_bytes())
        .collect();
    let ones_buf = scratch.alloc(ones_bytes.len() as u64)?;
    backend.write_buffer(ones_buf.id(), 0, &ones_bytes)?;

    // Zero vector: `bcast_add` against it is an aligned row-block copy, which
    // is how the compact embedder output lands in each tile's token slice.
    let zeros_host = vec![0.0f32; dim];
    let zeros_buf = scratch.alloc(seq::act_upload_bytes(act, &zeros_host).len() as u64)?;
    backend.write_buffer(zeros_buf.id(), 0, &seq::act_upload_bytes(act, &zeros_host))?;

    // --- per pass: upload, then embed (post_quant_conv -> x_embedder ->
    // scatter into token slices). The suffix rows are written once and never
    // touched by the embedder, which only fills each tile's leading rows. ---
    let x_bytes_of = |batch: usize| p.act_bytes((batch * tokens * dim) as u32);
    let mut freqs_bufs = Vec::with_capacity(passes.len());
    let mut plans = Vec::with_capacity(passes.len());
    let mut curs = Vec::with_capacity(passes.len());
    for pass in passes {
        let batch = pass.batch;
        debug_assert_eq!(pass.latents.len(), batch * n_patches * lat_c);
        let mut freqs = Vec::with_capacity(batch * one_tile.len());
        for _ in 0..batch {
            freqs.extend_from_slice(&one_tile);
        }
        let freqs_bytes = seq::freqs_upload_bytes(act, &freqs);
        let freqs_buf = scratch.alloc(freqs_bytes.len() as u64)?;
        backend.write_buffer(freqs_buf.id(), 0, &freqs_bytes)?;
        freqs_bufs.push(freqs_buf);
        plans.push(TilePlan::new(
            backend,
            scratch,
            batch,
            tokens,
            cfg.decoder_heads,
        )?);

        let latent_buf = scratch.alloc(p.act_bytes((batch * n_patches * lat_c) as u32))?;
        backend.write_buffer(
            latent_buf.id(),
            0,
            &seq::act_upload_bytes(act, pass.latents),
        )?;
        let x_buf = scratch.alloc(x_bytes_of(batch))?;
        let suffix_bytes = seq::act_upload_bytes(act, &consts.suffix);
        for i in 0..batch {
            let off = ((i * tokens + n_patches) * dim) as u64 * p.act_bytes(1);
            backend.write_buffer(x_buf.id(), off, &suffix_bytes)?;
        }

        let pq = residency.acquire(handles.post_quant_conv, backend).await?;
        let pq_b = residency
            .acquire(handles.post_quant_conv_bias, backend)
            .await?;
        let xe = residency.acquire(handles.x_embedder, backend).await?;
        let xe_b = residency.acquire(handles.x_embedder_bias, backend).await?;

        let latent_ref = latent_buf.as_buf_ref();
        let zeros_ref = zeros_buf.as_buf_ref();
        let x_ref = x_buf.as_buf_ref();
        let scope = scratch.batch();
        let emb_rows = (batch * n_patches) as u32;

        let z = mm_bias(
            &scope,
            p,
            ActBuf::dense(scope.import(&latent_ref)),
            pq.buf(),
            pq_b.buf(),
            emb_rows,
            lat_c as u32,
            lat_c as u32,
        )?;
        let embedded = mm_bias(
            &scope,
            p,
            ActBuf::dense(z),
            xe.buf(),
            xe_b.buf(),
            emb_rows,
            dim as u32,
            lat_c as u32,
        )?;

        // Scatter: tile `i`'s compact rows -> its leading rows in `x`. Both
        // offsets are whole `dim`-wide row blocks, so both stay 256B-aligned.
        let x_all = scope.import(&x_ref);
        let zeros = scope.import(&zeros_ref);
        let row_bytes = dim as u64 * p.act_bytes(1);
        let bu = scope.u32x4_uniform(dim as u32, 0, 0, 0)?;
        for i in 0..batch {
            let src = scope.subview(
                &embedded,
                i as u64 * n_patches as u64 * row_bytes,
                n_patches as u64 * row_bytes,
            );
            let dst = scope.subview(
                &x_all,
                i as u64 * tokens as u64 * row_bytes,
                n_patches as u64 * row_bytes,
            );
            scope.bcast_add::<BcastAddF32>(
                &p.bcast_add,
                src,
                zeros,
                bu,
                dst,
                (n_patches * dim) as u32,
            )?;
        }
        scope.submit_void().await?;
        curs.push(x_buf);
    }

    // --- 36 ViT blocks, weights streamed ONCE with one block of lookahead,
    // every pass through each block before the next ---
    let ones_ref = ones_buf.as_buf_ref();
    let mut pending = match handles.blocks.first() {
        Some(h) => Some(h.acquire(residency, backend).await?),
        None => None,
    };
    for idx in 0..handles.blocks.len() {
        let _guard = trace::scope!(format!("minimax.vae.block.{idx}")).entered();
        let views = pending.take().expect("pending acquire missing");
        // Block idx+1 is acquired during this block's passes; start the OS
        // reading idx+2 now so cold reads keep the disk queue full.
        if let Some(h) = handles.blocks.get(idx + 2) {
            h.hint_read(residency)
                .await
                .map_err(ResidencyError::<_, WgpuError>::Source)?;
        }
        let bufs = views.bufs();
        let scale_bufs = upload_acts(backend, scratch, act, &consts.scales[idx])?;
        let scale_refs = scale_bufs.each_ref().map(|b| b.as_buf_ref());
        // Passes run as deferred accelerator sequences (one wait per pass, no
        // per-op drains); the last pass's wait comes after the next block's
        // weights have streamed, overlapping the two.
        let run = async {
            let mut nxts = Vec::with_capacity(passes.len());
            let mut deferred = Vec::new();
            for (((pass, cur), freqs_buf), plan) in
                passes.iter().zip(&curs).zip(&freqs_bufs).zip(&plans)
            {
                let nxt = scratch.alloc(x_bytes_of(pass.batch))?;
                let scope = scratch.batch();
                let refs = [
                    cur.as_buf_ref(),
                    freqs_buf.as_buf_ref(),
                    zeros_buf.as_buf_ref(),
                    nxt.as_buf_ref(),
                    scale_refs[0],
                    scale_refs[1],
                ];
                let accel = block_forward_accel(
                    &scope,
                    cfg,
                    pipelines,
                    refs,
                    &bufs,
                    plan,
                    pass.batch as u32,
                    tokens as u32,
                )?;
                // Only the block's last pass stays in flight across the next
                // acquire: holding every pass's intermediates at once does not
                // fit (~0.9 GB each at 8192 tokens).
                if accel && nxts.len() + 1 == passes.len() {
                    deferred.push(scope);
                } else if accel {
                    scope.submit_void().await?;
                } else if bufs.nvfp4 {
                    return Err(DecodeError::Nvfp4Declined);
                } else {
                    block_forward(
                        &scope,
                        cfg,
                        pipelines,
                        scope.import_copy(refs[0]),
                        scope.import_copy(refs[1]),
                        scope.import_copy(ones_ref),
                        scope.import_copy(refs[3]),
                        &bufs,
                        scale_refs.map(|r| scope.import_copy(r)),
                        pass.batch as u32,
                        tokens as u32,
                    )?;
                    scope
                        .submit_void()
                        .instrument(tracing::debug_span!(target: PHASE, "minimax.vae.submit", idx))
                        .await?;
                }
                nxts.push(nxt);
            }
            Ok::<_, DecodeError<S::Error>>((nxts, deferred))
        };
        let next_acquire = async {
            match handles.blocks.get(idx + 1) {
                Some(h) => Ok::<_, ResidencyError<S::Error, WgpuError>>(Some(
                    h.acquire(residency, backend).await?,
                )),
                None => Ok(None),
            }
        };
        let (run_res, next_res) = futures::join!(run, next_acquire);
        let (nxts, deferred) = run_res?;
        for scope in deferred {
            scope.submit_void().await?;
        }
        curs = nxts;
        pending = next_res?;
        drop(views);
    }

    // --- per pass: norm_out (affine LayerNorm) -> proj_out, then read back ---
    tracing::info!(target: thinfer_core::trace::DIAG, "vae blocks done");
    let out_w = cfg.proj_out_dim();
    let affine_bufs = upload_acts(backend, scratch, act, &consts.norm_out)?;
    let [no, no_b] = affine_bufs.each_ref().map(|b| b.as_buf_ref());
    let po = residency.acquire(handles.proj_out, backend).await?;
    let po_b = residency.acquire(handles.proj_out_bias, backend).await?;
    // Every pass's kept rows, in pass order.
    let per = n_patches * out_w;
    let mut out = vec![0f32; passes.iter().map(|ps| ps.batch).sum::<usize>() * per];
    let mut filled = 0;
    for (pass, cur) in passes.iter().zip(&curs) {
        let batch = pass.batch;
        let rows = batch * tokens;
        let out_buf = scratch.alloc(p.act_bytes((rows * out_w) as u32))?;
        {
            let cur_ref = cur.as_buf_ref();
            let out_ref = out_buf.as_buf_ref();
            let scope = scratch.batch();
            let x = scope.import(&cur_ref);
            let normed = alloc_act(&scope, p, rows as u32, dim as u32)?;
            let lu = {
                let mut bytes = [0u8; 16];
                bytes[0..4].copy_from_slice(&(rows as u32).to_le_bytes());
                bytes[4..8].copy_from_slice(&(dim as u32).to_le_bytes());
                bytes[8..12].copy_from_slice(&cfg.decoder_norm_eps.to_le_bytes());
                scope.write_uniform(&bytes)?
            };
            scope.layernorm::<LayerNormF32>(&p.layernorm, x, lu, normed.data, rows as u32)?;

            // The file's LayerNorm is affine, and the engine's is not: apply the
            // gain and bias as the standard broadcast modulate (bias term 0, so
            // this is a plain `x * w + b`, not AdaLN's `1 + w`).
            let affine = alloc_act(&scope, p, rows as u32, dim as u32)?;
            let mu = scope.u32x4_uniform(dim as u32, 0, 0, 0)?;
            scope.bcast_modulate::<BcastModulateF32>(
                &p.bcast_modulate,
                normed.data,
                scope.import(&no),
                scope.import(&no_b),
                mu,
                affine.data,
                (rows * dim) as u32,
            )?;

            // Matmul into scratch, then bias into the readback buffer: one buffer
            // cannot be both operands of a single dispatch.
            let mm_out = alloc_matmul_out_buf(&scope, p, (rows * out_w) as u32)?;
            let dims = scope.u32x4_uniform(rows as u32, out_w as u32, dim as u32, 0)?;
            Block::dispatch_matmul_site(
                &scope,
                p,
                affine,
                scope.import_copy(po.buf()),
                mm_out,
                dims,
                p.matmul_i8_qkv.as_ref(),
                p.dequant_i8_qkv.as_ref(),
                None,
                &p.matmul_qkv,
                &p.matmuls.qkv,
                rows as u32,
                out_w as u32,
                dim as u32,
            )?;
            let bu = scope.u32x4_uniform(out_w as u32, 0, 0, 0)?;
            scope.bcast_add::<BcastAddF32>(
                &p.bcast_add,
                mm_out,
                scope.import_copy(po_b.buf()),
                bu,
                scope.import(&out_ref),
                (rows * out_w) as u32,
            )?;
            scope.submit_void().await?;
        }

        // Drop the suffix rows: they exist only to be attended to. The kept
        // rows decode on worker threads, in about READBACK_PARTS pieces.
        let bytes = backend
            .read_buffer(out_buf.id(), 0, p.act_bytes((rows * out_w) as u32))
            .await?;
        let elem = p.act_bytes(1) as usize;
        let piece = per.div_ceil(READBACK_PARTS.div_ceil(batch));
        let dst = &mut out[filled..filled + batch * per];
        filled += batch * per;
        let mut chunks = Vec::new();
        for (i, tile) in dst.chunks_mut(per).enumerate() {
            for (j, c) in tile.chunks_mut(piece).enumerate() {
                chunks.push((i * tokens * out_w + j * piece, c));
            }
        }
        par_fill(chunks, |src, c| {
            seq::act_readback_into_f32(act, &bytes[src * elem..(src + c.len()) * elem], c);
        });
    }
    Ok(out)
}

/// Upload host vectors as act-dtype scratch buffers (see [`DecoderConsts`]).
fn upload_acts<const N: usize>(
    backend: &WgpuBackend,
    scratch: &Workspace<WgpuBackend>,
    act: thinfer_core::ops::ActDtype,
    host: &[Vec<f32>; N],
) -> Result<[WsBuf<WgpuBackend>; N], WgpuError> {
    let mut out = Vec::with_capacity(N);
    for v in host {
        let bytes = seq::act_upload_bytes(act, v);
        let buf = scratch.alloc(bytes.len() as u64)?;
        backend.write_buffer(buf.id(), 0, &bytes)?;
        out.push(buf);
    }
    Ok(out.try_into().unwrap_or_else(|_| unreachable!()))
}

/// Token budget for one batched ViT pass. Weights stream once per decode
/// whatever the pass size ([`decode_passes`] is layer-major), so this only
/// bounds the `rows x 2 * ffn_inner` intermediate every pass holds. Measured:
/// 16384 decodes ~3s SLOWER than 8192 at 640x352x124 (fewer, bigger passes
/// do not beat the per-pass overhead they save).
pub const MAX_TOKENS_PER_PASS: usize = 8192;

/// Worker pieces a pass's token readback decodes in (see `decode_tokens`).
const READBACK_PARTS: usize = 8;

/// Worker pieces of the host pixel passes over a decoded clip.
const PIXEL_PARTS: usize = 8;

/// The NVFP4 decoder's activations: unit static scale, FlashInfer nibble order.
const NVFP4_FORMAT: thinfer_core::backend::Nvfp4Format = thinfer_core::backend::Nvfp4Format {
    dynamic_act_scale: false,
    hi_first: false,
};

/// Tiles per pass at this tile shape, at least one.
fn tiles_per_pass(tokens_per_tile: usize) -> usize {
    (MAX_TOKENS_PER_PASS / tokens_per_tile.max(1)).max(1)
}

/// A temporal chunk's spatial tiling: tile origins, lengths' shared latent
/// extent and the blend overlaps. Every tile in the grid shares one latent
/// extent, which is what lets a pass batch them against one set of rotary
/// frequencies, and every chunk of one decode shares the grid.
struct ClipTiles {
    ys: Vec<usize>,
    xs: Vec<usize>,
    y_ov: Vec<usize>,
    x_ov: Vec<usize>,
    th: usize,
    tw: usize,
}

impl ClipTiles {
    fn new(cfg: &VideoVaeConfig, policy: TilePolicy, lat_height: usize, lat_width: usize) -> Self {
        let ratio = cfg.spatial_compression;
        let (ys, y_lens, y_ov) = split_tiles(
            lat_height * ratio,
            policy.tile_pixels,
            policy.overlap_pixels,
            ratio,
        );
        let (xs, x_lens, x_ov) = split_tiles(
            lat_width * ratio,
            policy.tile_pixels,
            policy.overlap_pixels,
            ratio,
        );
        Self {
            th: y_lens[0] / ratio,
            tw: x_lens[0] / ratio,
            ys,
            xs,
            y_ov,
            x_ov,
        }
    }

    fn count(&self) -> usize {
        self.ys.len() * self.xs.len()
    }

    /// Every tile's latents of `z` (`[frames, lat_height, lat_width, C]`),
    /// tile-major.
    fn gather(
        &self,
        cfg: &VideoVaeConfig,
        z: &[f32],
        frames: usize,
        lat_height: usize,
        lat_width: usize,
    ) -> Vec<f32> {
        let ratio = cfg.spatial_compression;
        let lat_c = cfg.latent_channels;
        let mut gathered = Vec::with_capacity(self.count() * frames * self.th * self.tw * lat_c);
        for &y0 in &self.ys {
            for &x0 in &self.xs {
                let (ly, lx) = (y0 / ratio, x0 / ratio);
                for t in 0..frames {
                    for h in ly..ly + self.th {
                        let base = ((t * lat_height + h) * lat_width + lx) * lat_c;
                        gathered.extend_from_slice(&z[base..base + self.tw * lat_c]);
                    }
                }
            }
        }
        gathered
    }

    /// Un-patchify each tile's decoded tokens, then blend-and-trim into one
    /// chunk. The blend reads the ORIGINAL neighbour rather than an
    /// already-stitched one, as the reference does.
    fn stitch(&self, cfg: &VideoVaeConfig, tokens: &[f32], frames: usize) -> PixelBlock {
        let ratio = cfg.spatial_compression;
        let tile_tokens = frames * self.th * self.tw * cfg.proj_out_dim();
        let (ny, nx) = (self.ys.len(), self.xs.len());
        // Blend reads the unblended neighbours, so the grid lives until every
        // tile is cut.
        let tiles = {
            let grid = par_map(ny * nx, |idx| PixelBlock {
                data: unpatchify(
                    cfg,
                    &tokens[idx * tile_tokens..(idx + 1) * tile_tokens],
                    frames,
                    self.th,
                    self.tw,
                ),
                frames: frames * cfg.temporal_compression,
                height: self.th * ratio,
                width: self.tw * ratio,
                channels: cfg.out_channels,
            });

            par_map(ny * nx, |idx| {
                let (i, j) = (idx / nx, idx % nx);
                let mut tile = grid[idx].clone();
                if i > 0 {
                    tile.blend_head_from(&grid[idx - nx], self.y_ov[i - 1], 1);
                }
                if j > 0 {
                    tile.blend_head_from(&grid[idx - 1], self.x_ov[j - 1], 2);
                }
                if i + 1 < ny {
                    tile.truncate_axis(tile.height - self.y_ov[i], 1);
                }
                if j + 1 < nx {
                    tile.truncate_axis(tile.width - self.x_ov[j], 2);
                }
                tile
            })
        };
        let mut parts = tiles.into_iter();
        let rows: Vec<PixelBlock> = (0..ny)
            .map(|_| PixelBlock::concat(parts.by_ref().take(nx).collect(), 2))
            .collect();
        PixelBlock::concat(rows, 1)
    }
}

/// Decode `n_tiles` equally-shaped tiles' latents (`gathered`, tile-major) in
/// batched passes of [`MAX_TOKENS_PER_PASS`], layer-major across all of them
/// ([`decode_passes`]); returns the tiles' tokens, tile-major.
#[allow(clippy::too_many_arguments)]
async fn decode_gathered<S: WeightSource>(
    cfg: &VideoVaeConfig,
    backend: &WgpuBackend,
    pipelines: &DecoderPipelines,
    residency: &WeightResidency<S>,
    scratch: &Workspace<WgpuBackend>,
    handles: &DecoderHandles,
    consts: &DecoderConsts,
    gathered: &[f32],
    n_tiles: usize,
    frames: usize,
    th: usize,
    tw: usize,
) -> Result<Vec<f32>, DecodeError<S::Error>> {
    let tile_lat = frames * th * tw * cfg.latent_channels;
    let per_pass = tiles_per_pass(frames * th * tw + cfg.suffix_tokens());
    let passes: Vec<TilePass<'_>> = (0..n_tiles)
        .step_by(per_pass)
        .map(|start| {
            let batch = per_pass.min(n_tiles - start);
            TilePass {
                latents: &gathered[start * tile_lat..(start + batch) * tile_lat],
                batch,
            }
        })
        .collect();
    decode_passes(
        cfg, backend, pipelines, residency, scratch, handles, consts, &passes, frames, th, tw,
    )
    .await
}

/// Decode one temporal chunk of latents, tiled over height and width.
///
/// `z` is `[frames, lat_height, lat_width, latent_channels]`, channels-last and
/// already denormalized. Tiles are decoded in batched passes and stitched with
/// the reference's blend-then-trim.
#[allow(clippy::too_many_arguments)]
pub async fn decode_clip<S: WeightSource>(
    cfg: &VideoVaeConfig,
    backend: &WgpuBackend,
    pipelines: &DecoderPipelines,
    residency: &WeightResidency<S>,
    scratch: &Workspace<WgpuBackend>,
    handles: &DecoderHandles,
    consts: &DecoderConsts,
    policy: TilePolicy,
    z: &[f32],
    frames: usize,
    lat_height: usize,
    lat_width: usize,
) -> Result<PixelBlock, DecodeError<S::Error>> {
    let tiles = ClipTiles::new(cfg, policy, lat_height, lat_width);
    let gathered = tiles.gather(cfg, z, frames, lat_height, lat_width);
    let tokens = decode_gathered(
        cfg,
        backend,
        pipelines,
        residency,
        scratch,
        handles,
        consts,
        &gathered,
        tiles.count(),
        frames,
        tiles.th,
        tiles.tw,
    )
    .await?;
    Ok(tiles.stitch(cfg, &tokens, frames))
}

/// Decode a whole latent video into pixels.
///
/// `z` is `[latent_frames, lat_height, lat_width, latent_channels]`,
/// channels-last and already scaled by `latents_std` / `latents_mean`.
///
/// The chunking mirrors what the encoder did: it dropped `token_drop` trailing
/// latent frames from every clip, so consecutive decoded chunks overlap by
/// `frame_overlap` pixel frames and are cross-faded. Latent frames are repeated
/// at the tail when the length is not a whole number of chunks and the extra
/// pixel frames are cut off again at the end.
///
/// Every chunk's tiles share one shape, so they decode together: the ViT
/// stack streams once for the whole video, not once per chunk.
#[allow(clippy::too_many_arguments)]
pub async fn decode_tokens<S: WeightSource>(
    cfg: &VideoVaeConfig,
    backend: &WgpuBackend,
    pipelines: &DecoderPipelines,
    residency: &WeightResidency<S>,
    scratch: &Workspace<WgpuBackend>,
    handles: &DecoderHandles,
    consts: &DecoderConsts,
    policy: TilePolicy,
    z: &[f32],
    latent_frames: usize,
    lat_height: usize,
    lat_width: usize,
) -> Result<DecodedTokens, DecodeError<S::Error>> {
    let plan = ChunkPlan::new(cfg, latent_frames)?;
    let lat_c = cfg.latent_channels;
    let frame_stride = lat_height * lat_width * lat_c;

    // Repeat the last latent frame to fill the final chunk.
    let mut padded = z.to_vec();
    for _ in 0..plan.pad_tokens {
        let last = padded[padded.len() - frame_stride..].to_vec();
        padded.extend_from_slice(&last);
    }

    let tiles = ClipTiles::new(cfg, policy, lat_height, lat_width);
    let len = plan.chunk_tokens + plan.token_overlap;
    let mut gathered = Vec::new();
    for i in 0..plan.num_chunks {
        let start = i * plan.chunk_tokens;
        gathered.extend(tiles.gather(
            cfg,
            &padded[start * frame_stride..(start + len) * frame_stride],
            len,
            lat_height,
            lat_width,
        ));
    }
    let tokens = decode_gathered(
        cfg,
        backend,
        pipelines,
        residency,
        scratch,
        handles,
        consts,
        &gathered,
        plan.num_chunks * tiles.count(),
        len,
        tiles.th,
        tiles.tw,
    )
    .await?;
    tracing::info!(target: thinfer_core::trace::DIAG, "vae passes read back");
    Ok(DecodedTokens {
        cfg: *cfg,
        plan,
        tiles,
        tokens,
        latent_frames,
    })
}

/// [`decode_tokens`] then [`DecodedTokens::stitch`].
#[allow(clippy::too_many_arguments)]
pub async fn decode<S: WeightSource>(
    cfg: &VideoVaeConfig,
    backend: &WgpuBackend,
    pipelines: &DecoderPipelines,
    residency: &WeightResidency<S>,
    scratch: &Workspace<WgpuBackend>,
    handles: &DecoderHandles,
    consts: &DecoderConsts,
    policy: TilePolicy,
    z: &[f32],
    latent_frames: usize,
    lat_height: usize,
    lat_width: usize,
) -> Result<PixelBlock, DecodeError<S::Error>> {
    let tokens = decode_tokens(
        cfg,
        backend,
        pipelines,
        residency,
        scratch,
        handles,
        consts,
        policy,
        z,
        latent_frames,
        lat_height,
        lat_width,
    )
    .await?;
    Ok(tokens.stitch())
}

/// A decode's tile tokens, read back: all GPU work is done, and
/// [`Self::stitch`] is host-only (free to run on another thread).
pub struct DecodedTokens {
    cfg: VideoVaeConfig,
    plan: ChunkPlan,
    tiles: ClipTiles,
    tokens: Vec<f32>,
    latent_frames: usize,
}

impl DecodedTokens {
    /// Un-patchify, blend and trim every chunk into the clip's pixels.
    pub fn stitch(&self) -> PixelBlock {
        let (cfg, plan, tiles, tokens) = (&self.cfg, &self.plan, &self.tiles, &self.tokens);
        let len = plan.chunk_tokens + plan.token_overlap;
        let chunk_tokens = tokens.len() / plan.num_chunks;

        let mut decoded: Vec<PixelBlock> = Vec::new();
        let mut overlap: Option<PixelBlock> = None;
        for i in 0..plan.num_chunks {
            let clip = tiles.stitch(cfg, &tokens[i * chunk_tokens..(i + 1) * chunk_tokens], len);

            // Pass 0 is the chunk itself; pass 1 (present only when the encoder
            // dropped a tail) is the lead-in that the NEXT chunk blends over.
            for j in 0..plan.passes_per_chunk() {
                // Pass j's `chunk_num_frames` window, minus its pre-padding.
                let s0 = (j * plan.chunk_num_frames).min(clip.frames);
                let end = s0 + plan.chunk_num_frames.min(clip.frames - s0);
                let mut chunk = clip.slice_frames((s0 + plan.frame_pre_padding).min(end), end);
                if j == 0 {
                    if let Some(prev) = &overlap {
                        chunk.blend_head_from(prev, plan.frame_overlap, 0);
                    }
                    decoded.push(chunk);
                } else {
                    overlap = Some(chunk);
                }
            }
        }
        if let Some(tail) = overlap {
            decoded.push(tail);
        }

        tracing::info!(target: thinfer_core::trace::DIAG, "vae stitched");
        let mut out = PixelBlock::concat(decoded, 0);
        let pad_frames = plan.pad_frames(cfg, self.latent_frames);
        if pad_frames > 0 {
            out.truncate_axis(out.frames - pad_frames, 0);
        }
        out
    }
}

/// Temporal chunking of a decode, derived from the encoder's clip geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChunkPlan {
    pub chunk_tokens: usize,
    pub token_overlap: usize,
    pub token_drop: usize,
    pub chunk_num_frames: usize,
    pub frame_pre_padding: usize,
    pub frame_overlap: usize,
    pub pad_tokens: usize,
    pub num_chunks: usize,
}

impl ChunkPlan {
    pub fn new<SE: core::fmt::Debug>(
        cfg: &VideoVaeConfig,
        latent_frames: usize,
    ) -> Result<Self, DecodeError<SE>> {
        let ratio_t = cfg.temporal_compression;
        let chunk_tokens = cfg.tokens_chunk_size();
        let frame_pre_padding = (ratio_t - cfg.clip_length % ratio_t) % ratio_t;
        let token_overlap = (chunk_tokens - cfg.token_drop % chunk_tokens) % chunk_tokens;
        let frame_overlap = (token_overlap * ratio_t).saturating_sub(frame_pre_padding);

        let num_tokens = latent_frames + cfg.token_drop;
        let pad_tokens = (chunk_tokens - num_tokens % chunk_tokens) % chunk_tokens;
        let chunks = (num_tokens + pad_tokens) / chunk_tokens;
        let num_chunks = chunks.saturating_sub(usize::from(cfg.token_drop > 0));
        if num_chunks < 1 {
            // `token_drop` eats a whole chunk of lead-in, so the shortest
            // decodable latent is one full chunk beyond it.
            return Err(DecodeError::TooFewLatentFrames {
                got: latent_frames,
                need: 2 * chunk_tokens - cfg.token_drop,
            });
        }
        Ok(Self {
            chunk_tokens,
            token_overlap,
            token_drop: cfg.token_drop,
            chunk_num_frames: chunk_tokens * ratio_t,
            frame_pre_padding,
            frame_overlap,
            pad_tokens,
            num_chunks,
        })
    }

    /// Decoded passes per chunk: the chunk, plus the lead-in for the next one
    /// when the encoder dropped a tail.
    pub fn passes_per_chunk(&self) -> usize {
        1 + usize::from(self.token_drop > 0)
    }

    /// Trailing pixel frames produced only by the repeated latent frames. A
    /// chunk's LAST latent frame covers `clip_length % ratio_t` pixel frames
    /// where the others cover `ratio_t`.
    pub fn pad_frames(&self, cfg: &VideoVaeConfig, latent_frames: usize) -> usize {
        let ratio_t = cfg.temporal_compression;
        let intra_tail = cfg.clip_length % ratio_t;
        (0..self.pad_tokens)
            .map(|k| {
                if intra_tail != 0 && (latent_frames + k).is_multiple_of(self.chunk_tokens) {
                    intra_tail
                } else {
                    ratio_t
                }
            })
            .sum()
    }
}

/// `(0..n).map(f)` with one thread per item on native, in order; serial on
/// wasm (single-threaded). Host-side decode work only: identical results.
/// `f(src, chunk)` over `(src, chunk)` pairs on scoped worker threads (in
/// order on wasm).
fn par_fill(chunks: Vec<(usize, &mut [f32])>, f: impl Fn(usize, &mut [f32]) + Sync) {
    #[cfg(not(target_arch = "wasm32"))]
    if chunks.len() > 1 {
        std::thread::scope(|s| {
            let f = &f;
            for (src, c) in chunks {
                s.spawn(move || f(src, c));
            }
        });
        return;
    }
    for (src, c) in chunks {
        f(src, c);
    }
}

fn par_map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    #[cfg(not(target_arch = "wasm32"))]
    if n > 1 {
        return std::thread::scope(|s| {
            let f = &f;
            let workers: Vec<_> = (0..n).map(|i| s.spawn(move || f(i))).collect();
            workers
                .into_iter()
                .map(|w| w.join().expect("vae stitch worker panicked"))
                .collect()
        });
    }
    (0..n).map(f).collect()
}

/// Un-patchify one tile's tokens into pixels.
///
/// `tokens` is `[frames * height * width, out_channels * pt * p * p]` and the
/// result is `(frames * pt, height * p, width * p, out_channels)`, matching the
/// reference's `reshape -> transpose(0, 1, 5, 2, 6, 3, 7, 4) -> reshape`.
pub fn unpatchify(
    cfg: &VideoVaeConfig,
    tokens: &[f32],
    frames: usize,
    height: usize,
    width: usize,
) -> Vec<f32> {
    let (pt, p, c) = (
        cfg.temporal_compression,
        cfg.spatial_compression,
        cfg.out_channels,
    );
    let (of, oh, ow) = (frames * pt, height * p, width * p);
    let mut out = vec![0.0f32; of * oh * ow * c];
    let row_w = c * pt * p * p;
    for t in 0..frames {
        for h in 0..height {
            for w in 0..width {
                let base = ((t * height + h) * width + w) * row_w;
                for ch in 0..c {
                    for dt in 0..pt {
                        for dh in 0..p {
                            for dw in 0..p {
                                // Token payload is (out_channels, pt, p, p).
                                let src = base + ((ch * pt + dt) * p + dh) * p + dw;
                                let dst =
                                    (((t * pt + dt) * oh + h * p + dh) * ow + w * p + dw) * c + ch;
                                out[dst] = tokens[src];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// A dense `(frames, height, width, channels)` pixel block.
///
/// Decode produces these per tile and per chunk and stitches them; carrying the
/// extents alongside the data is what keeps the axis arithmetic honest once
/// tiles start getting trimmed to different widths.
#[derive(Clone, Debug, PartialEq)]
pub struct PixelBlock {
    pub data: Vec<f32>,
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub channels: usize,
}

/// The decoder emits ImageNet-normalized RGB over a `[0, 1]` base, so a decode
/// lands at roughly `[-2.12, 2.64]` rather than in any display range.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

impl PixelBlock {
    /// Undo the ImageNet normalization and re-lay the block as the encoder's
    /// `[channels, frames, height, width]` in `[-1, 1]`.
    ///
    /// Two conversions ride together because both are trivially wrong in a way
    /// no range check would catch: the block is channels-LAST, and its values
    /// are normalized against a `[0, 1]` base rather than a signed one.
    pub fn to_signed_planar(&self) -> Vec<f32> {
        self.to_signed_planar_frames(self.frames)
    }

    /// [`Self::to_signed_planar`] of the leading `frames` frames only (each
    /// channel plane `frames * height * width` long), on worker threads.
    pub fn to_signed_planar_frames(&self, frames: usize) -> Vec<f32> {
        let keep = frames.min(self.frames) * self.height * self.width;
        let mut out = vec![0.0f32; self.channels * keep];
        if keep == 0 {
            return out;
        }
        let piece = keep.div_ceil(PIXEL_PARTS);
        let mut chunks = Vec::new();
        for (c, plane) in out.chunks_mut(keep).enumerate() {
            for (j, d) in plane.chunks_mut(piece).enumerate() {
                chunks.push((c * keep + j * piece, d));
            }
        }
        par_fill(chunks, |at, d| {
            let (c, i0) = (at / keep, at % keep);
            let (mu, sd) = (IMAGENET_MEAN[c], IMAGENET_STD[c]);
            for (k, o) in d.iter_mut().enumerate() {
                let unit = self.data[(i0 + k) * self.channels + c] * sd + mu;
                *o = (unit * 2.0 - 1.0).clamp(-1.0, 1.0);
            }
        });
        out
    }

    /// Whether every sample is finite, checked on worker threads.
    pub fn all_finite(&self) -> bool {
        let piece = self.data.len().div_ceil(PIXEL_PARTS).max(1);
        let parts = self.data.len().div_ceil(piece);
        par_map(parts, |j| {
            self.data[j * piece..((j + 1) * piece).min(self.data.len())]
                .iter()
                .all(|v| v.is_finite())
        })
        .into_iter()
        .all(|ok| ok)
    }

    /// Element count of one step along `axis`, and how many steps it has.
    /// Axis 0 is time, 1 is height, 2 is width.
    fn axis_geometry(&self, axis: usize) -> (usize, usize, usize) {
        match axis {
            0 => (1, self.frames, self.height * self.width * self.channels),
            1 => (self.frames, self.height, self.width * self.channels),
            2 => (self.frames * self.height, self.width, self.channels),
            _ => unreachable!("axis {axis} out of range"),
        }
    }

    fn axis_len(&self, axis: usize) -> usize {
        self.axis_geometry(axis).1
    }

    fn set_axis_len(&mut self, axis: usize, len: usize) {
        match axis {
            0 => self.frames = len,
            1 => self.height = len,
            2 => self.width = len,
            _ => unreachable!("axis {axis} out of range"),
        }
    }

    /// Cross-fade the TAIL of `a` into the HEAD of `self` along `axis`.
    ///
    /// The ramp is the reference's `w = i / extent`, so the first blended step
    /// is pure `a` and the last never quite reaches `self`. (An
    /// `(i + 1) / (extent + 1)` ramp is the symmetric-looking choice and is NOT
    /// what the reference does; it shifts every seam by half a step.)
    pub fn blend_head_from(&mut self, a: &PixelBlock, extent: usize, axis: usize) {
        let extent = extent.min(self.axis_len(axis)).min(a.axis_len(axis));
        if extent == 0 {
            return;
        }
        let (outer, len_b, inner) = self.axis_geometry(axis);
        let len_a = a.axis_len(axis);
        for o in 0..outer {
            for i in 0..extent {
                let w = i as f32 / extent as f32;
                let dst = (o * len_b + i) * inner;
                let src = (o * len_a + (len_a - extent) + i) * inner;
                for k in 0..inner {
                    self.data[dst + k] = a.data[src + k] * (1.0 - w) + self.data[dst + k] * w;
                }
            }
        }
    }

    /// Keep the leading `keep` steps along `axis`, dropping the tail.
    pub fn truncate_axis(&mut self, keep: usize, axis: usize) {
        let (outer, len, inner) = self.axis_geometry(axis);
        if keep >= len {
            return;
        }
        let mut out = Vec::with_capacity(outer * keep * inner);
        for o in 0..outer {
            let base = o * len * inner;
            out.extend_from_slice(&self.data[base..base + keep * inner]);
        }
        self.data = out;
        self.set_axis_len(axis, keep);
    }

    /// Keep the trailing steps from `start` along `axis`.
    pub fn drop_head_axis(&mut self, start: usize, axis: usize) {
        let (outer, len, inner) = self.axis_geometry(axis);
        if start == 0 {
            return;
        }
        let keep = len.saturating_sub(start);
        let mut out = Vec::with_capacity(outer * keep * inner);
        for o in 0..outer {
            let base = (o * len + start) * inner;
            out.extend_from_slice(&self.data[base..base + keep * inner]);
        }
        self.data = out;
        self.set_axis_len(axis, keep);
    }

    /// Frames `start..end` (axis 0 is outermost, so this is one contiguous copy).
    pub fn slice_frames(&self, start: usize, end: usize) -> PixelBlock {
        let step = self.height * self.width * self.channels;
        PixelBlock {
            data: self.data[start * step..end * step].to_vec(),
            frames: end - start,
            height: self.height,
            width: self.width,
            channels: self.channels,
        }
    }

    /// Concatenate along `axis` in one pass. Every input must agree on the
    /// other extents.
    pub fn concat(mut parts: Vec<PixelBlock>, axis: usize) -> PixelBlock {
        if parts.len() == 1 {
            return parts.pop().expect("one part");
        }
        let first = parts.first().expect("concat needs at least one block");
        let (outer, _, inner) = first.axis_geometry(axis);
        let total: usize = parts.iter().map(|p| p.axis_len(axis)).sum();
        let mut data = Vec::with_capacity(outer * total * inner);
        for o in 0..outer {
            for part in &parts {
                let len = part.axis_len(axis);
                data.extend_from_slice(&part.data[o * len * inner..(o + 1) * len * inner]);
            }
        }
        let mut out = PixelBlock {
            data,
            frames: first.frames,
            height: first.height,
            width: first.width,
            channels: first.channels,
        };
        out.set_axis_len(axis, total);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> VideoVaeConfig {
        VideoVaeConfig::h3()
    }

    /// Channels-last to channels-first WITH the ImageNet undo. A mid-grey pixel
    /// (the normalized value of the per-channel mean) must land at exactly 0 in
    /// the signed range, and each channel must pick up its OWN mean/std.
    #[test]
    fn signed_planar_undoes_imagenet_and_transposes() {
        // One 1x1x2 frame, channels-last. Pixel 0 is normalized zero, so it
        // decodes to the per-channel mean; pixel 1 is one std above it.
        let b = PixelBlock {
            data: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            frames: 1,
            height: 1,
            width: 2,
            channels: 3,
        };
        let out = b.to_signed_planar();
        assert_eq!(out.len(), b.channels * b.frames * b.height * b.width);
        for c in 0..3 {
            // Planar: channel c owns out[c * 2 .. c * 2 + 2].
            let px0 = (IMAGENET_MEAN[c] * 2.0 - 1.0).clamp(-1.0, 1.0);
            let px1 = ((IMAGENET_MEAN[c] + IMAGENET_STD[c]) * 2.0 - 1.0).clamp(-1.0, 1.0);
            assert!((out[c * 2] - px0).abs() < 1e-6, "channel {c} pixel 0");
            assert!((out[c * 2 + 1] - px1).abs() < 1e-6, "channel {c} pixel 1");
        }
    }

    /// Values outside the representable range must clamp, not wrap.
    #[test]
    fn signed_planar_clamps() {
        let b = PixelBlock {
            data: vec![100.0, -100.0, 0.0],
            frames: 1,
            height: 1,
            width: 1,
            channels: 3,
        };
        let out = b.to_signed_planar();
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], -1.0);
    }

    #[test]
    fn shapes_match_the_checkpoint() {
        let c = cfg();
        // to_qkv [6144, 2048], to_out [2048, 2048], w1 [16384, 2048], w2 [2048, 8192].
        assert_eq!(c.dim(), 2048);
        assert_eq!(3 * c.dim(), 6144);
        assert_eq!(c.ffn_inner(), 8192);
        assert_eq!(2 * c.ffn_inner(), 16384);
        // proj_out [3072, 2048] = 3 * 4 * 16 * 16.
        assert_eq!(c.proj_out_dim(), 3072);
        // x_embedder [2048, 24].
        assert_eq!(c.latent_channels, 24);
    }

    #[test]
    fn rope_rotates_three_quarters_of_each_head() {
        let c = cfg();
        assert_eq!(c.rotary_dim(), 48);
        assert!(c.rotary_dim() < c.decoder_head_dim);
        // Word alignment for the packed rope kernels.
        assert_eq!(c.rotary_dim() % 4, 0);
    }

    #[test]
    fn origin_is_the_identity_rotation() {
        let c = cfg();
        let f = build_freqs(&c, &[[0.0, 0.0, 0.0]]);
        assert_eq!(f.len(), c.rotary_dim());
        for k in 0..c.rotary_dim() / 2 {
            assert_eq!(f[2 * k], 1.0);
            assert_eq!(f[2 * k + 1], 0.0);
        }
    }

    /// Each axis must drive its own 8-frequency block.
    #[test]
    fn axes_drive_disjoint_frequency_blocks() {
        let c = cfg();
        let per_axis = c.rotary_dim() / 6;
        assert_eq!(per_axis, 8);
        let base = build_freqs(&c, &[[0.0, 0.0, 0.0]]);
        for (axis, moved) in [
            (0usize, [0.5f32, 0.0, 0.0]),
            (1, [0.0, 0.5, 0.0]),
            (2, [0.0, 0.0, 0.5]),
        ] {
            let f = build_freqs(&c, &[moved]);
            for k in 0..c.rotary_dim() / 2 {
                let touched = f[2 * k] != base[2 * k] || f[2 * k + 1] != base[2 * k + 1];
                assert_eq!(touched, k / per_axis == axis, "axis {axis} pair {k}");
            }
        }
    }

    /// Positions are cell centres on `[-1, 1)`, and the suffix tokens sit at the
    /// origin so registers carry no positional signal.
    #[test]
    fn positions_are_cell_centres_with_origin_suffix() {
        let c = cfg();
        let p = build_positions(&c, 2, 2, 2);
        assert_eq!(p.len(), 8 + c.suffix_tokens());
        assert_eq!(p[0], [-0.5, -0.5, -0.5]);
        assert_eq!(p[7], [0.5, 0.5, 0.5]);
        for s in &p[8..] {
            assert_eq!(*s, [0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn suffix_is_registers_plus_one_mask_token() {
        assert_eq!(cfg().suffix_tokens(), 5);
    }

    /// A 256x256 tile is 16x16 latent voxels, so one ViT pass stays ~1285
    /// tokens however big the clip is. This is the bound that makes a 36-layer
    /// full-attention decoder tractable at all.
    #[test]
    fn tile_bounds_the_sequence_length() {
        let c = cfg();
        let t = TilePolicy::default();
        let latent = t.tile_pixels / c.spatial_compression;
        assert_eq!(latent, 16);
        let tokens = latent * latent * c.tokens_chunk_size() + c.suffix_tokens();
        assert_eq!(c.tokens_chunk_size(), 5);
        assert_eq!(tokens, 1285);
    }

    /// Tile boundaries must land on whole latent voxels.
    #[test]
    fn tile_and_overlap_align_to_latent_voxels() {
        let c = cfg();
        let t = TilePolicy::default();
        assert_eq!(t.tile_pixels % c.spatial_compression, 0);
        assert_eq!(t.overlap_pixels % c.spatial_compression, 0);
    }

    #[test]
    fn single_tile_when_shorter_than_the_tile() {
        let (s, l, o) = split_tiles(192, 256, 64, 16);
        assert_eq!(s, vec![0]);
        assert_eq!(l, vec![192]);
        assert!(o.is_empty());
    }

    /// Tiles must cover the whole extent, start at 0, end exactly at `length`,
    /// and keep every overlap at or above the minimum.
    #[test]
    fn tiles_cover_the_extent_with_sufficient_overlap() {
        let c = cfg();
        for length in [256usize, 320, 512, 768, 1344] {
            let (starts, lens, overlaps) = split_tiles(length, 256, 64, c.spatial_compression);
            assert_eq!(starts[0], 0, "len {length}");
            let last = starts.last().unwrap() + lens.last().unwrap();
            assert_eq!(last, length, "len {length} ends at {last}");
            for (i, ov) in overlaps.iter().enumerate() {
                assert!(*ov >= 64, "len {length} overlap {i} = {ov}");
                assert_eq!(ov % c.spatial_compression, 0, "len {length} overlap {i}");
            }
            for w in starts.windows(2) {
                assert!(w[1] > w[0], "len {length} starts must advance");
            }
            for s in &starts {
                assert_eq!(s % c.spatial_compression, 0, "len {length} start {s}");
            }
        }
    }

    /// The 1344-wide canvas is the shipping case: 7 tiles, not the 6 that
    /// `ceil(1344 / 256)` suggests, because 6 cannot hold a 64 overlap.
    #[test]
    fn shipping_width_needs_a_seventh_tile() {
        let (starts, _, overlaps) = split_tiles(1344, 256, 64, 16);
        assert_eq!(starts.len(), 7);
        assert_eq!(overlaps.len(), 6);
    }

    #[test]
    fn unpatchify_expands_by_the_compression_ratios() {
        let c = cfg();
        let (f, h, w) = (1usize, 1, 1);
        let tokens = vec![0.0f32; f * h * w * c.proj_out_dim()];
        let out = unpatchify(&c, &tokens, f, h, w);
        assert_eq!(out.len(), 4 * 16 * 16 * 3);
    }

    /// One token must scatter to its own pixel block, channel-last.
    #[test]
    fn unpatchify_places_channels_last() {
        let c = VideoVaeConfig {
            out_channels: 2,
            temporal_compression: 1,
            spatial_compression: 2,
            ..cfg()
        };
        // Payload order is (channel, pt, ph, pw) = 2 * 1 * 2 * 2 = 8 values.
        let tokens: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let out = unpatchify(&c, &tokens, 1, 1, 1);
        // Output is (1, 2, 2, 2) as (h, w, c) row-major.
        // channel 0 block = 0,1,2,3 ; channel 1 block = 4,5,6,7.
        assert_eq!(out, vec![0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]);
    }

    fn block(data: Vec<f32>, frames: usize, height: usize, width: usize) -> PixelBlock {
        PixelBlock {
            data,
            frames,
            height,
            width,
            channels: 1,
        }
    }

    /// The ramp starts at pure `a` and never reaches pure `b`, matching the
    /// reference's `i / extent` weighting.
    #[test]
    fn blend_ramps_from_a_to_b() {
        let a = block(vec![0.0; 4], 1, 1, 4);
        let mut b = block(vec![1.0; 4], 1, 1, 4);
        b.blend_head_from(&a, 4, 2);
        assert_eq!(b.data, vec![0.0, 0.25, 0.5, 0.75]);
    }

    /// A blend along height must pair rows within the same frame, not run off
    /// the end of frame 0 into frame 1.
    #[test]
    fn blend_along_height_stays_within_each_frame() {
        // 2 frames, 2 rows, 1 col. `a`'s tail row is its row 1.
        let a = block(vec![10.0, 20.0, 30.0, 40.0], 2, 2, 1);
        let mut b = block(vec![0.0, 0.0, 0.0, 0.0], 2, 2, 1);
        b.blend_head_from(&a, 1, 1);
        // extent 1 -> w = 0 -> pure `a` tail: frame 0 takes 20, frame 1 takes 40.
        assert_eq!(b.data, vec![20.0, 0.0, 40.0, 0.0]);
    }

    #[test]
    fn truncate_and_concat_round_trip_along_width() {
        let mut a = block(vec![1.0, 2.0, 3.0, 4.0], 1, 2, 2);
        a.truncate_axis(1, 2);
        assert_eq!(a.data, vec![1.0, 3.0]);
        assert_eq!(a.width, 1);
        let b = block(vec![9.0, 9.0], 1, 2, 1);
        let j = PixelBlock::concat(vec![a, b], 2);
        assert_eq!(j.width, 2);
        assert_eq!(j.data, vec![1.0, 9.0, 3.0, 9.0]);
    }

    fn plan(latent_frames: usize) -> ChunkPlan {
        ChunkPlan::new::<std::convert::Infallible>(&cfg(), latent_frames).unwrap()
    }

    /// The encoder's clip geometry fixes every one of these; they are the
    /// numbers the chunk loop indexes with, so pin them literally.
    #[test]
    fn chunk_geometry_matches_the_encoder() {
        let p = plan(37);
        assert_eq!(p.chunk_tokens, 5);
        assert_eq!(p.chunk_num_frames, 20);
        // clip_length 17 is not a multiple of 4, so 3 pixel frames lead in.
        assert_eq!(p.frame_pre_padding, 3);
        // token_drop 3 of a 5-token chunk leaves a 2-token overlap.
        assert_eq!(p.token_overlap, 2);
        assert_eq!(p.frame_overlap, 5);
        assert_eq!(p.passes_per_chunk(), 2);
    }

    /// A 5-second clip is 124 frames -> 37 latent frames, and must decode back
    /// to exactly 124 with no padding anywhere.
    #[test]
    fn five_second_clip_round_trips_frame_exact() {
        let c = cfg();
        let p = plan(37);
        assert_eq!(p.pad_tokens, 0);
        assert_eq!(p.num_chunks, 7);
        assert_eq!(p.pad_frames(&c, 37), 0);
        // Each chunk emits `chunk_num_frames - frame_pre_padding`, plus the
        // final lead-in tail.
        assert_eq!(p.num_chunks * 17 + p.frame_overlap, 124);
    }

    /// Below one chunk beyond the dropped lead-in there is nothing to decode,
    /// and the reference fails with an empty concatenate instead of saying so.
    #[test]
    fn too_few_latent_frames_is_a_named_error() {
        let c = cfg();
        let err = ChunkPlan::new::<std::convert::Infallible>(&c, 2).unwrap_err();
        assert!(matches!(
            err,
            DecodeError::TooFewLatentFrames { got: 2, need: 7 }
        ));
        assert!(ChunkPlan::new::<std::convert::Infallible>(&c, 7).is_ok());
    }

    /// A latent length that is not a whole number of chunks pads with repeated
    /// frames, and every padded frame must be cut back off.
    #[test]
    fn padded_tail_frames_are_all_removed() {
        let c = cfg();
        for latent_frames in 7..40 {
            let p = plan(latent_frames);
            let produced = p.num_chunks * (p.chunk_num_frames - p.frame_pre_padding)
                + p.frame_overlap
                - p.pad_frames(&c, latent_frames);
            // Every latent frame covers ratio_t pixel frames except each
            // chunk's last, which covers `clip_length % ratio_t`.
            assert!(produced > 0, "{latent_frames} produced nothing");
            assert!(
                produced <= latent_frames * c.temporal_compression,
                "{latent_frames} produced {produced}"
            );
        }
    }

    #[test]
    fn drop_head_trims_leading_frames() {
        let mut a = block(vec![1.0, 2.0, 3.0], 3, 1, 1);
        a.drop_head_axis(1, 0);
        assert_eq!(a.frames, 2);
        assert_eq!(a.data, vec![2.0, 3.0]);
    }
}
