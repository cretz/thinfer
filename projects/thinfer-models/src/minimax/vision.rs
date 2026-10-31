//! Qwen3-VL-32B vision tower (mmproj), the image half of MiniMax-H3's
//! conditioner for reference images (Ref2VA) and keyframes (FL2VA). Ground
//! truth: `transformers/models/qwen3_vl::Qwen3VLVisionModel`.
//!
//! patch embed (Conv3d k = s = (2,16,16) with bias; both temporal slabs see the
//! same frame, so it is two 768-wide Linears summed) + a learned 48x48 position
//! table bilinearly resampled to the patch grid -> 27 pre-LayerNorm blocks (fused
//! `[q | k | v]` with bias, 2D half-rot rope, full attention over the image,
//! gelu-tanh MLP) -> merger (per-patch LayerNorm, 2x2 concat, fc1, GELU, fc2 ->
//! 5120). After blocks 8, 16 and 24 a deepstack merger (concat FIRST, then a
//! 4608-wide LayerNorm) emits a second 5120-d stream that the LLM adds into the
//! image rows after its decoder layers 0, 1 and 2.
//!
//! Patches are merge-unit-major (each 2x2 unit's four patches contiguous, units
//! row-major), the HF processor layout, so merged tokens come out row-major over
//! the `(gh/2, gw/2)` grid, the order the LLM's image rows take. The mergers
//! use the tanh GELU (`GeluF32`) for HF's exact one, as the Qwen2.5-VL tower
//! does. bf16 acts throughout.

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError};
use thinfer_core::ops::{ActDtype, BcastModulateF32, GeluF32, LayerNormF32};
use thinfer_core::residency::{
    GpuView, ResidencyError, TransposePolicy, WeightHandle, WeightMeta, WeightResidency,
};
use thinfer_core::tensor::Shape;
use thinfer_core::trace::{self, PHASE};
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace, WsBuf};
use tracing::Instrument;

use crate::common::block::{
    ActBuf, Block, BlockPipelines, BlockWgslConfigs, alloc_act, alloc_matmul_out_buf, op_add,
    op_qkv_split, op_rope_halfrot, op_sdpa,
};
use crate::common::embedders::{
    LinearBiasBufs, LinearBiasHandles, LinearBiasViews, bcast_add_uniform,
};
use crate::common::loader::{LoadError, register_linear, register_passthrough};
use crate::common::seq;
use crate::minimax::video_vae::TilePlan;
use thinfer_core::backend::TileAttnShape;

/// From the mmproj GGUF KV (`clip`, `qwen3vl_merger`) and the upstream config.
pub mod config {
    pub const HIDDEN: usize = 1152;
    pub const DEPTH: usize = 27;
    pub const N_HEADS: usize = 16;
    pub const HEAD_DIM: usize = HIDDEN / N_HEADS; // 72
    pub const FFN_HIDDEN: usize = 4304;
    pub const PATCH: usize = 16;
    pub const IN_CHANNELS: usize = 3;
    /// One temporal slab of the patch conv: `C * P * P`.
    pub const PATCH_ELEMS: usize = IN_CHANNELS * PATCH * PATCH; // 768
    pub const SPATIAL_MERGE: usize = 2;
    pub const MERGE_UNIT: usize = SPATIAL_MERGE * SPATIAL_MERGE;
    /// Merger width: four patches concatenated.
    pub const MERGER_HIDDEN: usize = HIDDEN * MERGE_UNIT; // 4608
    /// The LLM width the mergers project to.
    pub const PROJECTION_DIM: usize = 5120;
    /// Side of the learned position table (`num_position_embeddings` 2304).
    pub const POS_GRID: usize = 48;
    /// Blocks whose outputs feed the deepstack mergers (0-based).
    pub const DEEPSTACK: [usize; 3] = [8, 16, 24];
    pub const NORM_EPS: f32 = 1e-6;
    pub const ROPE_THETA: f64 = 10_000.0;
    /// Pixels per side of one merged token (the processor's resize factor).
    pub const MERGED_PIXELS: usize = PATCH * SPATIAL_MERGE; // 32
}

#[derive(Clone, Debug)]
struct Norm {
    weight: WeightHandle,
    bias: WeightHandle,
}

#[derive(Clone, Debug)]
struct BlockHandles {
    ln1: Norm,
    ln2: Norm,
    qkv: LinearBiasHandles,
    out: LinearBiasHandles,
    fc1: LinearBiasHandles,
    fc2: LinearBiasHandles,
}

#[derive(Clone, Debug)]
struct MergerHandles {
    norm: Norm,
    fc1: LinearBiasHandles,
    fc2: LinearBiasHandles,
}

#[derive(Clone, Debug)]
pub struct VisionHandles {
    /// The patch conv's two temporal slabs, each a `[1152, 768]` Linear.
    patch: [WeightHandle; 2],
    patch_bias: WeightHandle,
    pos_embed: WeightHandle,
    blocks: Vec<BlockHandles>,
    merger: MergerHandles,
    deepstack: [MergerHandles; 3],
}

fn lin_bias<S: WeightSource>(
    residency: &WeightResidency<S>,
    prefix: &str,
) -> Result<LinearBiasHandles, LoadError> {
    Ok(LinearBiasHandles {
        weight: register_linear(residency, &WeightId(format!("{prefix}.weight")))?,
        bias: register_passthrough(residency, &WeightId(format!("{prefix}.bias")))?,
    })
}

fn norm<S: WeightSource>(residency: &WeightResidency<S>, prefix: &str) -> Result<Norm, LoadError> {
    Ok(Norm {
        weight: register_passthrough(residency, &WeightId(format!("{prefix}.weight")))?,
        bias: register_passthrough(residency, &WeightId(format!("{prefix}.bias")))?,
    })
}

/// A 4-D patch slab (`[1152, 3, 16, 16]` in file order) as a `[1152, 768]`
/// Linear: the bytes are contiguous `[out, (c, py, px)]`.
fn register_patch<S: WeightSource>(
    residency: &WeightResidency<S>,
    name: &str,
) -> Result<WeightHandle, LoadError> {
    let id = WeightId(name.into());
    let entry = residency
        .source()
        .catalog()
        .get(&id)
        .ok_or_else(|| LoadError::UnknownWeight(id.clone()))?;
    let encoding = entry.encoding.ok_or_else(|| LoadError::Undecodable {
        id: id.clone(),
        encoding: None,
        label: entry.encoding_label.clone(),
    })?;
    assert_eq!(
        entry.shape.elements(),
        config::HIDDEN * config::PATCH_ELEMS,
        "patch slab element count"
    );
    Ok(residency.register(WeightMeta {
        id,
        shape: Shape(vec![config::HIDDEN, config::PATCH_ELEMS]),
        encoding,
        on_disk_bytes: entry.size,
        transpose: TransposePolicy::Linear2D,
        transcode: None,
    }))
}

/// Register every vision-tower weight from the mmproj GGUF (native keys).
pub fn register_handles<S: WeightSource>(
    residency: &WeightResidency<S>,
) -> Result<VisionHandles, LoadError> {
    let blocks = (0..config::DEPTH)
        .map(|i| {
            let p = format!("v.blk.{i}");
            Ok(BlockHandles {
                ln1: norm(residency, &format!("{p}.ln1"))?,
                ln2: norm(residency, &format!("{p}.ln2"))?,
                qkv: lin_bias(residency, &format!("{p}.attn_qkv"))?,
                out: lin_bias(residency, &format!("{p}.attn_out"))?,
                fc1: lin_bias(residency, &format!("{p}.ffn_up"))?,
                fc2: lin_bias(residency, &format!("{p}.ffn_down"))?,
            })
        })
        .collect::<Result<Vec<_>, LoadError>>()?;
    let deepstack = |i: usize| -> Result<MergerHandles, LoadError> {
        let p = format!("v.deepstack.{}", config::DEEPSTACK[i]);
        Ok(MergerHandles {
            norm: norm(residency, &format!("{p}.norm"))?,
            fc1: lin_bias(residency, &format!("{p}.fc1"))?,
            fc2: lin_bias(residency, &format!("{p}.fc2"))?,
        })
    };
    Ok(VisionHandles {
        patch: [
            register_patch(residency, "v.patch_embd.weight")?,
            register_patch(residency, "v.patch_embd.weight.1")?,
        ],
        patch_bias: register_passthrough(residency, &WeightId("v.patch_embd.bias".into()))?,
        pos_embed: register_passthrough(residency, &WeightId("v.position_embd.weight".into()))?,
        blocks,
        merger: MergerHandles {
            norm: norm(residency, "v.post_ln")?,
            fc1: lin_bias(residency, "mm.0")?,
            fc2: lin_bias(residency, "mm.2")?,
        },
        deepstack: [deepstack(0)?, deepstack(1)?, deepstack(2)?],
    })
}

struct NormBufs {
    weight: BufRef,
    bias: BufRef,
}

struct NormViews<'a> {
    weight: GpuView<'a>,
    bias: GpuView<'a>,
}

impl Norm {
    async fn acquire<'a, S: WeightSource>(
        &self,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<NormViews<'a>, ResidencyError<S::Error, WgpuError>> {
        Ok(NormViews {
            weight: residency.acquire(self.weight, backend).await?,
            bias: residency.acquire(self.bias, backend).await?,
        })
    }
}

impl NormViews<'_> {
    fn bufs(&self) -> NormBufs {
        NormBufs {
            weight: self.weight.buf(),
            bias: self.bias.buf(),
        }
    }
}

struct BlockViews<'a> {
    ln1: NormViews<'a>,
    ln2: NormViews<'a>,
    qkv: LinearBiasViews<'a>,
    out: LinearBiasViews<'a>,
    fc1: LinearBiasViews<'a>,
    fc2: LinearBiasViews<'a>,
}

struct BlockBufs {
    ln1: NormBufs,
    ln2: NormBufs,
    qkv: LinearBiasBufs,
    out: LinearBiasBufs,
    fc1: LinearBiasBufs,
    fc2: LinearBiasBufs,
}

impl BlockHandles {
    async fn acquire<'a, S: WeightSource>(
        &self,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<BlockViews<'a>, ResidencyError<S::Error, WgpuError>> {
        Ok(BlockViews {
            ln1: self.ln1.acquire(residency, backend).await?,
            ln2: self.ln2.acquire(residency, backend).await?,
            qkv: self.qkv.acquire(residency, backend).await?,
            out: self.out.acquire(residency, backend).await?,
            fc1: self.fc1.acquire(residency, backend).await?,
            fc2: self.fc2.acquire(residency, backend).await?,
        })
    }
}

impl BlockViews<'_> {
    fn bufs(&self) -> BlockBufs {
        BlockBufs {
            ln1: self.ln1.bufs(),
            ln2: self.ln2.bufs(),
            qkv: self.qkv.bufs(),
            out: self.out.bufs(),
            fc1: self.fc1.bufs(),
            fc2: self.fc2.bufs(),
        }
    }
}

/// `x @ wᵀ + bias` -> dense act `[rows, n]` (bf16 weights through the qkv site).
fn biased<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    w: &LinearBiasBufs,
    rows: u32,
    n: u32,
    k: u32,
) -> Result<ActBuf<'wsp>, WgpuError> {
    let pre = matmul(scope, bp, x, scope.import_copy(w.weight), rows, n, k)?;
    let out = alloc_act(scope, bp, rows, n)?;
    let u = bcast_add_uniform(scope, n)?;
    scope.bcast_add::<thinfer_core::ops::BcastAddF32>(
        &bp.bcast_add,
        pre,
        scope.import_copy(w.bias),
        u,
        out.data,
        rows * n,
    )?;
    Ok(out)
}

fn matmul<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    w: BatchBuf<'wsp>,
    rows: u32,
    n: u32,
    k: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let out = alloc_matmul_out_buf(scope, bp, rows * n)?;
    let dims = scope.u32x4_uniform(rows, n, k, 0)?;
    Block::dispatch_matmul_site(
        scope,
        bp,
        x,
        w,
        out,
        dims,
        bp.matmul_i8_qkv.as_ref(),
        bp.dequant_i8_qkv.as_ref(),
        bp.dequant_qkv.as_ref(),
        &bp.matmul_qkv,
        &bp.matmuls.qkv,
        rows,
        n,
        k,
    )?;
    Ok(out)
}

/// Affine LayerNorm over `dim`-wide rows.
fn layernorm<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    norm: &NormBufs,
    rows: u32,
    dim: u32,
) -> Result<ActBuf<'wsp>, WgpuError> {
    let normed = alloc_act(scope, bp, rows, dim)?;
    let mut lu = [0u8; 16];
    lu[0..4].copy_from_slice(&rows.to_le_bytes());
    lu[4..8].copy_from_slice(&dim.to_le_bytes());
    lu[8..12].copy_from_slice(&config::NORM_EPS.to_le_bytes());
    let lu = scope.write_uniform(&lu)?;
    scope.layernorm::<LayerNormF32>(&bp.layernorm, x.data, lu, normed.data, rows)?;
    // Plain `x * w + b` (modulate with its `1 +` term off).
    let out = alloc_act(scope, bp, rows, dim)?;
    let mu = scope.u32x4_uniform(dim, 0, 0, 0)?;
    scope.bcast_modulate::<BcastModulateF32>(
        &bp.bcast_modulate,
        normed.data,
        scope.import_copy(norm.weight),
        scope.import_copy(norm.bias),
        mu,
        out.data,
        rows * dim,
    )?;
    Ok(out)
}

fn gelu<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    rows: u32,
    n: u32,
) -> Result<ActBuf<'wsp>, WgpuError> {
    let out = alloc_act(scope, bp, rows, n)?;
    scope.dispatch_op::<GeluF32>(&bp.gelu, &[x.data], out.data)?;
    Ok(out)
}

/// One pre-LN block: `x += proj(attn(rope(qkv(ln1 x))))`; `x += fc2(gelu(fc1(ln2 x)))`.
#[allow(clippy::too_many_arguments)]
fn block_forward<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    freqs: BatchBuf<'wsp>,
    no_mask: BatchBuf<'wsp>,
    out: ActBuf<'wsp>,
    w: &BlockBufs,
    tiles: &TilePlan,
    rows: u32,
) -> Result<(), WgpuError> {
    let n = config::HIDDEN as u32;
    let (heads, hd) = (config::N_HEADS as u32, config::HEAD_DIM as u32);
    let ffn = config::FFN_HIDDEN as u32;

    let h = layernorm(scope, bp, x, &w.ln1, rows, n)?;
    let qkv = biased(scope, bp, h, &w.qkv, rows, 3 * n, n)?;
    let (q, k, v) = (
        alloc_act(scope, bp, rows, n)?,
        alloc_act(scope, bp, rows, n)?,
        alloc_act(scope, bp, rows, n)?,
    );
    op_qkv_split(scope, bp, qkv, q, k, v, rows, n)?;
    let qr = alloc_act(scope, bp, rows, n)?;
    op_rope_halfrot(scope, bp, q, freqs, qr, rows, heads, hd)?;
    let kr = alloc_act(scope, bp, rows, n)?;
    op_rope_halfrot(scope, bp, k, freqs, kr, rows, heads, hd)?;
    let sa = alloc_act(scope, bp, rows, n)?;
    let scale = 1.0 / (hd as f32).sqrt();
    // Full attention over one image: the accelerator's tile attention with
    // every key tile listed (thousands of patches; the backend SDPA crawls
    // there and can trip the GPU watchdog), else the backend op.
    let shape = TileAttnShape {
        rows,
        heads,
        head_dim: hd,
        n_tiles: tiles.n_tiles,
        scale,
    };
    if !scope.try_tile_attention(
        &shape,
        qr.data,
        kr.data,
        v.data,
        sa.data,
        scope.import_copy(tiles.geometry.as_buf_ref()),
        scope.import_copy(tiles.lists.as_buf_ref()),
    )? {
        op_sdpa(
            scope, bp, qr, kr, v, no_mask, sa, 1, rows, rows, heads, heads, hd, scale, 0,
        )?;
    }
    let attn = biased(scope, bp, sa, &w.out, rows, n, n)?;
    let mid = alloc_act(scope, bp, rows, n)?;
    op_add(scope, bp, x, attn, mid)?;

    let h = layernorm(scope, bp, mid, &w.ln2, rows, n)?;
    let up = biased(scope, bp, h, &w.fc1, rows, ffn, n)?;
    let act = gelu(scope, bp, up, rows, ffn)?;
    let down = biased(scope, bp, act, &w.fc2, rows, n, ffn)?;
    op_add(scope, bp, mid, down, out)
}

#[derive(Debug)]
pub enum VisionError<SE: core::fmt::Debug> {
    Wgpu(WgpuError),
    Residency(ResidencyError<SE, WgpuError>),
}

impl<SE: core::fmt::Debug> From<WgpuError> for VisionError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}

impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for VisionError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

/// One image's LLM-side features, merged tokens row-major over `(gh/2, gw/2)`.
#[derive(Clone, Debug)]
pub struct VisionOutput {
    /// `[tokens, 5120]`: what replaces the `<|image_pad|>` embeddings.
    pub embeds: Vec<f32>,
    /// `[tokens, 5120]` each: added to the image rows after LLM layers 0, 1, 2.
    pub deepstack: [Vec<f32>; 3],
    pub tokens: usize,
    /// Merged grid `(gh/2, gw/2)`.
    pub merged_grid: (usize, usize),
}

/// The pipeline configs the tower runs on (bf16 acts and weights).
pub fn wgsl_configs() -> BlockWgslConfigs {
    use thinfer_core::ops::{WeightDtype, WgslConfig};
    BlockWgslConfigs::uniform(WgslConfig {
        bf16_quant_writes: false,
        act_dtype: ActDtype::Bf16,
        weight_dtype: WeightDtype::Bf16,
    })
}

/// Patch rows `[gh * gw, 768]` in merge-unit-major order, each `(c, py, px)`,
/// normalized `(u8 - 127.5) / 127.5`. `rgb` is `[height, width, 3]`.
pub fn patchify(rgb: &[u8], height: usize, width: usize) -> Vec<f32> {
    let p = config::PATCH;
    let (gh, gw) = (height / p, width / p);
    let mut out = Vec::with_capacity(gh * gw * config::PATCH_ELEMS);
    for (ph, pw) in patch_order(gh, gw) {
        for c in 0..config::IN_CHANNELS {
            for py in 0..p {
                let row = ((ph * p + py) * width + pw * p) * 3;
                for px in 0..p {
                    out.push((rgb[row + px * 3 + c] as f32 - 127.5) / 127.5);
                }
            }
        }
    }
    out
}

/// Each patch's `(row, col)` on the full patch grid, merge-unit-major.
fn patch_order(gh: usize, gw: usize) -> Vec<(usize, usize)> {
    let m = config::SPATIAL_MERGE;
    let mut out = Vec::with_capacity(gh * gw);
    for uh in 0..gh / m {
        for uw in 0..gw / m {
            for ih in 0..m {
                for iw in 0..m {
                    out.push((uh * m + ih, uw * m + iw));
                }
            }
        }
    }
    out
}

/// HF `fast_pos_embed_interpolate`: the 48x48 table resampled to `(gh, gw)`
/// with `linspace(0, 47, n)` (corner-aligned) bilinear weights, weights and
/// result rounded to bf16 as the bf16 model does; rows in patch order.
fn interpolate_pos(table: &[f32], gh: usize, gw: usize) -> Vec<f32> {
    let n = config::POS_GRID;
    let d = config::HIDDEN;
    let bf = |x: f32| half::bf16::from_f32(x).to_f32();
    // torch.linspace in f32: start + i * step, the upper half mirrored from the end.
    let linspace = |len: usize| -> Vec<f32> {
        let end = (n - 1) as f32;
        if len == 1 {
            return vec![0.0];
        }
        let step = end / (len - 1) as f32;
        let half = len / 2;
        (0..len)
            .map(|i| {
                if i < half {
                    i as f32 * step
                } else {
                    end - (len - 1 - i) as f32 * step
                }
            })
            .collect()
    };
    let (hs, ws) = (linspace(gh), linspace(gw));
    let mut out = Vec::with_capacity(gh * gw * d);
    for (r, c) in patch_order(gh, gw) {
        let (h, w) = (hs[r], ws[c]);
        let (hf, wf) = (h as usize, w as usize);
        let (hc, wc) = ((hf + 1).min(n - 1), (wf + 1).min(n - 1));
        let (dh, dw) = (h - hf as f32, w - wf as f32);
        let taps = [
            (hf * n + wf, bf((1.0 - dh) * (1.0 - dw))),
            (hf * n + wc, bf((1.0 - dh) * dw)),
            (hc * n + wf, bf(dh * (1.0 - dw))),
            (hc * n + wc, bf(dh * dw)),
        ];
        // bf16 products summed tap by tap (`p[0] + p[1] + p[2] + p[3]`, each
        // add rounded), as the bf16 tensors are.
        for ch in 0..d {
            let sum = taps
                .iter()
                .map(|&(i, wt)| bf(table[i * d + ch] * wt))
                .reduce(|a, b| bf(a + b))
                .expect("four taps");
            out.push(sum);
        }
    }
    out
}

/// Half-rot rope freqs `[gh * gw, 72]` (`(cos, sin)` per pair): pairs 0..18
/// rotate by the patch row, 18..36 by the column, `inv = 1e4^(-2i/36)`.
fn rope_freqs(gh: usize, gw: usize) -> Vec<f32> {
    let pairs = config::HEAD_DIM / 2;
    let per_axis = pairs / 2;
    let inv: Vec<f64> = (0..per_axis)
        .map(|i| 1.0 / config::ROPE_THETA.powf(2.0 * i as f64 / pairs as f64))
        .collect();
    let mut out = Vec::with_capacity(gh * gw * config::HEAD_DIM);
    for (r, c) in patch_order(gh, gw) {
        for j in 0..pairs {
            let (pos, f) = if j < per_axis {
                (r, inv[j])
            } else {
                (c, inv[j - per_axis])
            };
            // f32 like HF: the angle is formed in f32 from f32 inv_freq.
            let arg = (pos as f32) * (f as f32);
            out.push(arg.cos());
            out.push(arg.sin());
        }
    }
    out
}

fn upload(
    scratch: &Workspace<WgpuBackend>,
    backend: &WgpuBackend,
    bytes: &[u8],
) -> Result<WsBuf<WgpuBackend>, WgpuError> {
    let buf = scratch.alloc(bytes.len() as u64)?;
    backend.write_buffer(buf.id(), 0, bytes)?;
    Ok(buf)
}

/// A merger over `rows` patch rows -> `[rows / 4, 5120]` read back to the host.
/// `post_shuffle` = deepstack form (LayerNorm over the 4608-wide concat).
#[allow(clippy::too_many_arguments)]
async fn merge<S: WeightSource>(
    backend: &WgpuBackend,
    bp: &BlockPipelines,
    residency: &WeightResidency<S>,
    scratch: &Workspace<WgpuBackend>,
    h: &MergerHandles,
    x: BufRef,
    rows: u32,
    post_shuffle: bool,
) -> Result<Vec<f32>, VisionError<S::Error>> {
    let (hid, mh) = (config::HIDDEN as u32, config::MERGER_HIDDEN as u32);
    let proj = config::PROJECTION_DIM as u32;
    let merged = rows / config::MERGE_UNIT as u32;
    let norm = h.norm.acquire(residency, backend).await?;
    let fc1 = h.fc1.acquire(residency, backend).await?;
    let fc2 = h.fc2.acquire(residency, backend).await?;
    let (nb, f1, f2) = (norm.bufs(), fc1.bufs(), fc2.bufs());
    let out_buf = scratch.alloc(bp.act_bytes(merged * proj))?;
    {
        let scope = scratch.batch();
        let x = ActBuf::dense(scope.import_copy(x));
        // The [rows, 1152] -> [rows/4, 4608] view is a reshape of contiguous rows.
        let normed = if post_shuffle {
            layernorm(&scope, bp, x, &nb, merged, mh)?
        } else {
            layernorm(&scope, bp, x, &nb, rows, hid)?
        };
        let up = biased(&scope, bp, normed, &f1, merged, mh, mh)?;
        let act = gelu(&scope, bp, up, merged, mh)?;
        let out = biased(&scope, bp, act, &f2, merged, proj, mh)?;
        let dst = scope.import_copy(out_buf.as_buf_ref());
        scope.copy_buffer_to_buffer(out.data, 0, dst, 0, bp.act_bytes(merged * proj))?;
        scope.submit_void().await?;
    }
    let bytes = backend
        .read_buffer(out_buf.id(), 0, bp.act_bytes(merged * proj))
        .await?;
    Ok(seq::act_readback_to_f32(
        bp.act_dtype,
        &bytes,
        (merged * proj) as usize,
    ))
}

/// Run the tower on one RGB image (`[height, width, 3]`, both multiples of 32).
#[allow(clippy::too_many_arguments)]
pub async fn forward<S: WeightSource>(
    backend: &WgpuBackend,
    bp: &BlockPipelines,
    residency: &WeightResidency<S>,
    scratch: &Workspace<WgpuBackend>,
    handles: &VisionHandles,
    rgb: &[u8],
    height: usize,
    width: usize,
) -> Result<VisionOutput, VisionError<S::Error>> {
    assert_eq!(bp.act_dtype, ActDtype::Bf16, "vision tower runs bf16 acts");
    let m = config::MERGED_PIXELS;
    assert!(
        height.is_multiple_of(m) && width.is_multiple_of(m),
        "image sides must be multiples of {m}"
    );
    assert_eq!(rgb.len(), height * width * 3);
    let (gh, gw) = (height / config::PATCH, width / config::PATCH);
    let rows = (gh * gw) as u32;
    let hid = config::HIDDEN as u32;
    let pe = config::PATCH_ELEMS as u32;
    let act = bp.act_dtype;

    // Host prep: patches, interpolated position table, rope table.
    let patches = upload(
        scratch,
        backend,
        &seq::act_upload_bytes(act, &patchify(rgb, height, width)),
    )?;
    let table = {
        let view = residency.acquire(handles.pos_embed, backend).await?;
        let len = config::POS_GRID * config::POS_GRID * config::HIDDEN;
        let bytes = backend
            .read_buffer(view.buf().id, view.buf().offset, (len * 2) as u64)
            .await?;
        seq::act_readback_to_f32(ActDtype::Bf16, &bytes, len)
    };
    let pos = upload(
        scratch,
        backend,
        &seq::act_upload_bytes(act, &interpolate_pos(&table, gh, gw)),
    )?;
    let freqs = upload(
        scratch,
        backend,
        &seq::freqs_upload_bytes(act, &rope_freqs(gh, gw)),
    )?;
    // op_sdpa binds a mask even when told it has none.
    let no_mask = upload(scratch, backend, &[0u8; 16])?;
    let tiles = TilePlan::new(backend, scratch, 1, rows as usize, config::N_HEADS)?;

    // Patch embed: both slabs over the same patch rows, + bias + positions.
    let mut cur = scratch.alloc(bp.act_bytes(rows * hid))?;
    {
        let w0 = residency.acquire(handles.patch[0], backend).await?;
        let w1 = residency.acquire(handles.patch[1], backend).await?;
        let b = residency.acquire(handles.patch_bias, backend).await?;
        let scope = scratch.batch();
        let x = ActBuf::dense(scope.import_copy(patches.as_buf_ref()));
        let o0 = matmul(&scope, bp, x, scope.import_copy(w0.buf()), rows, hid, pe)?;
        let o1 = matmul(&scope, bp, x, scope.import_copy(w1.buf()), rows, hid, pe)?;
        let sum = alloc_act(&scope, bp, rows, hid)?;
        op_add(&scope, bp, ActBuf::dense(o0), ActBuf::dense(o1), sum)?;
        let biased_sum = alloc_act(&scope, bp, rows, hid)?;
        let u = bcast_add_uniform(&scope, hid)?;
        scope.bcast_add::<thinfer_core::ops::BcastAddF32>(
            &bp.bcast_add,
            sum.data,
            scope.import_copy(b.buf()),
            u,
            biased_sum.data,
            rows * hid,
        )?;
        let dst = ActBuf::dense(scope.import_copy(cur.as_buf_ref()));
        op_add(
            &scope,
            bp,
            biased_sum,
            ActBuf::dense(scope.import_copy(pos.as_buf_ref())),
            dst,
        )?;
        scope.submit_void().await?;
    }

    let mut deepstack = Vec::with_capacity(config::DEEPSTACK.len());
    let mut pending = Some(handles.blocks[0].acquire(residency, backend).await?);
    for idx in 0..config::DEPTH {
        let _g = trace::scope!(format!("minimax.vision.block.{idx}")).entered();
        let views = pending.take().expect("pending acquire missing");
        let bufs = views.bufs();
        let nxt = scratch.alloc(bp.act_bytes(rows * hid))?;
        {
            let scope = scratch.batch();
            block_forward(
                &scope,
                bp,
                ActBuf::dense(scope.import_copy(cur.as_buf_ref())),
                scope.import_copy(freqs.as_buf_ref()),
                scope.import_copy(no_mask.as_buf_ref()),
                ActBuf::dense(scope.import_copy(nxt.as_buf_ref())),
                &bufs,
                &tiles,
                rows,
            )?;
            let next_acquire = async {
                match handles.blocks.get(idx + 1) {
                    Some(h) => Ok::<_, ResidencyError<S::Error, WgpuError>>(Some(
                        h.acquire(residency, backend).await?,
                    )),
                    None => Ok(None),
                }
            };
            let submit = scope
                .submit_void()
                .instrument(tracing::debug_span!(target: PHASE, "minimax.vision.submit", idx));
            let (submit_res, next_res) = futures::join!(submit, next_acquire);
            submit_res?;
            pending = next_res?;
        }
        drop(views);
        cur = nxt;
        if let Some(k) = config::DEEPSTACK.iter().position(|&d| d == idx) {
            let h = &handles.deepstack[k];
            deepstack.push(
                merge(
                    backend,
                    bp,
                    residency,
                    scratch,
                    h,
                    cur.as_buf_ref(),
                    rows,
                    true,
                )
                .await?,
            );
        }
    }
    let embeds = merge(
        backend,
        bp,
        residency,
        scratch,
        &handles.merger,
        cur.as_buf_ref(),
        rows,
        false,
    )
    .await?;
    let deepstack: [Vec<f32>; 3] = deepstack
        .try_into()
        .unwrap_or_else(|_| unreachable!("three deepstack taps"));
    Ok(VisionOutput {
        embeds,
        deepstack,
        tokens: (rows as usize) / config::MERGE_UNIT,
        merged_grid: (gh / 2, gw / 2),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_rows_are_merge_unit_major_and_normalized() {
        // 32x32 image = 2x2 patches = one merge unit; pixel value = patch index.
        let (h, w) = (32, 32);
        let mut rgb = vec![0u8; h * w * 3];
        for y in 0..h {
            for x in 0..w {
                let patch = (y / 16) * 2 + x / 16;
                rgb[(y * w + x) * 3] = patch as u8 * 50;
            }
        }
        let rows = patchify(&rgb, h, w);
        assert_eq!(rows.len(), 4 * config::PATCH_ELEMS);
        for p in 0..4 {
            let want = (p as f32 * 50.0 - 127.5) / 127.5;
            assert_eq!(rows[p * config::PATCH_ELEMS], want);
        }
    }

    #[test]
    fn interpolation_hits_the_table_corners() {
        let n = config::POS_GRID;
        let d = config::HIDDEN;
        let table: Vec<f32> = (0..n * n * d).map(|i| ((i / d) % 7) as f32).collect();
        let pe = interpolate_pos(&table, 4, 6);
        // First patch (0,0) = table row 0; patch (0,1) in merge order is col 1.
        assert_eq!(pe[0], table[0]);
        // Last merge unit's last patch = (3, 5) = table corner (47, 47).
        let last = pe.len() - d;
        assert_eq!(pe[last], table[(n * n - 1) * d]);
    }

    #[test]
    fn rope_pairs_split_rows_then_columns() {
        let f = rope_freqs(2, 4);
        // Patch order: unit (0,0) = (0,0),(0,1),(1,0),(1,1); row 1 is patch 2.
        let hd = config::HEAD_DIM;
        // Patch (0,1): row pairs identity, col pair 18 rotates by 1 rad.
        assert_eq!(f[hd], 1.0);
        assert!((f[hd + 36] - 1f32.cos()).abs() < 1e-6);
        // Patch (1,0): row pair 0 rotates by 1, col pairs identity.
        assert!((f[2 * hd] - 1f32.cos()).abs() < 1e-6);
        assert_eq!(f[2 * hd + 36], 1.0);
    }
}
