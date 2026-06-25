//! Qwen2.5-VL vision tower (mmproj). Ground truth:
//! `transformers/models/qwen2_5_vl::Qwen2_5_VisionTransformerPretrainedModel`.
//!
//! A ViT that turns a patchified reference image into LM-hidden (3584-d) tokens
//! for the edit path's `<|image_pad|>` slots. Pipeline:
//!   patch_embed (Linear 1176->1280, weight = patch_embd.weight + .weight.1)
//!   -> 32 pre-norm blocks (RMSNorm ln1/ln2; separate q/k/v with bias;
//!      2D half-rot RoPE on q/k; segmented SDPA; out proj with bias; SwiGLU FFN
//!      with bias) -> merger (post_ln RMSNorm -> view N/4 x 5120 -> Linear 5120
//!      -> GELU -> Linear 3584).
//!
//! Attention is windowed except on the `fullatt_block_indexes` {7,15,23,31}
//! (n % 8 == 7). Tokens are reordered by `window_index` (a permutation over the
//! N/4 merge units) so each window's 4 merge units (16 raw patches) are
//! contiguous; the merger runs on the reordered stream and the output is
//! un-permuted with `argsort(window_index)`. Windowing is realized as a
//! block-diagonal additive SDPA mask over the standard `op_sdpa` (O(N^2);
//! noted for a later perf pass). RoPE cos/sin rows are reordered the same way.
//!
//! GGUF keys are native (`v.blk.{i}.*`, `v.post_ln`, `mm.{0,2}`,
//! `v.patch_embd.weight(.1)`); F16 in-file, narrowed to bf16 on upload. bf16
//! acts throughout; block matmuls bf16 (the merger GELU uses the tanh-approx
//! `GeluF32`, within tolerance vs HF's erf GELU).

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError};
use thinfer_core::ops::{ActDtype, GeluF32};
use thinfer_core::residency::{
    GpuView, ResidencyError, TransposePolicy, WeightHandle, WeightMeta, WeightResidency,
};
use thinfer_core::tensor::Shape;
use thinfer_core::trace::{self, PHASE};
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace};
use tracing::Instrument;

use crate::common::block::{
    ActBuf, Block, BlockPipelines, BlockWgslConfigs, alloc_act, alloc_matmul_out_buf, op_add,
    op_rmsnorm, op_rope_halfrot, op_sdpa, op_silu_mul,
};
use crate::common::embedders::{
    LinearBiasBufs, LinearBiasHandles, LinearBiasViews, bcast_add_uniform,
};
use crate::common::loader::{LoadError, register_linear, register_passthrough};
use crate::common::rope_embedder::RopeEmbedder;
use crate::common::seq;
use crate::qwen_image::dit::QwenImageDitPipelines;

/// Audited against the mmproj GGUF KV (`clip` arch, `qwen2.5vl_merger`).
pub mod config {
    pub const HIDDEN: usize = 1280;
    pub const DEPTH: usize = 32;
    pub const N_HEADS: usize = 16;
    pub const HEAD_DIM: usize = HIDDEN / N_HEADS; // 80
    pub const FFN_HIDDEN: usize = 3420;
    pub const PATCH: usize = 14;
    pub const TEMPORAL_PATCH: usize = 2;
    pub const IN_CHANNELS: usize = 3;
    /// patchified row width = in_ch * temporal * patch^2 = 3*2*14*14.
    pub const PATCH_ELEMS: usize = IN_CHANNELS * TEMPORAL_PATCH * PATCH * PATCH; // 1176
    /// patch-embed reads ONE temporal slot worth (`in_ch * patch^2`); both
    /// temporal slabs hold identical image data so we sum two Linears over it.
    pub const PATCH_SLOT_ELEMS: usize = IN_CHANNELS * PATCH * PATCH; // 588
    pub const SPATIAL_MERGE: usize = 2;
    /// 2x2 merge unit = 4 contiguous patch rows.
    pub const MERGE_UNIT: usize = SPATIAL_MERGE * SPATIAL_MERGE; // 4
    /// 112px window / patch(14) / merge(2) = 4 merge units per window side.
    pub const WINDOW_MERGE: usize = 4;
    pub const PROJECTION_DIM: usize = 3584;
    /// merger MLP hidden = HIDDEN * MERGE_UNIT = 5120.
    pub const MERGER_HIDDEN: usize = HIDDEN * MERGE_UNIT; // 5120
    pub const NORM_EPS: f32 = 1e-6;
    pub const ROPE_THETA: f32 = 10_000.0;
    /// every 8th block is full-attention (n % 8 == 7).
    pub const N_WA_PATTERN: usize = 8;
}

fn is_fullatt(block: usize) -> bool {
    block % config::N_WA_PATTERN == config::N_WA_PATTERN - 1
}

// ---------------------------------------------------------------------------
// Weight registration
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct BlockHandles {
    ln1: WeightHandle,
    ln2: WeightHandle,
    attn_q: LinearBiasHandles,
    attn_k: LinearBiasHandles,
    attn_v: LinearBiasHandles,
    attn_out: LinearBiasHandles,
    ffn_gate: LinearBiasHandles,
    ffn_up: LinearBiasHandles,
    ffn_down: LinearBiasHandles,
}

#[derive(Clone, Debug)]
pub struct VisionHandles {
    /// Two patch-embed temporal slabs, each registered as a `[1280, 588]`
    /// Linear (4D `[1280,3,14,14]` flattened); the forward sums their outputs.
    patch_embd: WeightHandle,
    patch_embd_1: WeightHandle,
    blocks: Vec<BlockHandles>,
    post_ln: WeightHandle,
    mm0: LinearBiasHandles,
    mm2: LinearBiasHandles,
}

fn lin_bias<S: WeightSource>(
    residency: &WeightResidency<S>,
    weight: &str,
    bias: &str,
) -> Result<LinearBiasHandles, LoadError> {
    Ok(LinearBiasHandles {
        weight: register_linear(residency, &WeightId(weight.into()))?,
        bias: register_passthrough(residency, &WeightId(bias.into()))?,
    })
}

fn norm<S: WeightSource>(
    residency: &WeightResidency<S>,
    name: &str,
) -> Result<WeightHandle, LoadError> {
    register_passthrough(residency, &WeightId(name.into()))
}

/// Register a 4D `v.patch_embd.weight(.1)` (`[1280,3,14,14]`) as a 2D Linear
/// `[1280, 588]`. The on-disk bytes are contiguous, so the [out, in] row-major
/// reinterpretation is correct. `register_linear` would pass the 4D shape, so
/// we build the `WeightMeta` directly (F16 -> bf16 on upload, `Linear2D`
/// transpose to `[588, 1280]` for the matmul).
fn register_patch_embd<S: WeightSource>(
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
    debug_assert_eq!(
        entry.shape.elements(),
        config::HIDDEN * config::PATCH_SLOT_ELEMS,
        "patch_embd element count mismatch"
    );
    Ok(residency.register(WeightMeta {
        id,
        shape: Shape(vec![config::HIDDEN, config::PATCH_SLOT_ELEMS]),
        encoding,
        on_disk_bytes: entry.size,
        transpose: TransposePolicy::Linear2D,
        transcode: None,
    }))
}

/// Register all vision-tower weights from the mmproj GGUF (native keys).
pub fn register_handles<S: WeightSource>(
    residency: &WeightResidency<S>,
) -> Result<VisionHandles, LoadError> {
    let mut blocks = Vec::with_capacity(config::DEPTH);
    for i in 0..config::DEPTH {
        let p = format!("v.blk.{i}");
        let lb = |w: &str, b: &str| lin_bias(residency, &format!("{p}.{w}"), &format!("{p}.{b}"));
        blocks.push(BlockHandles {
            ln1: norm(residency, &format!("{p}.ln1.weight"))?,
            ln2: norm(residency, &format!("{p}.ln2.weight"))?,
            attn_q: lb("attn_q.weight", "attn_q.bias")?,
            attn_k: lb("attn_k.weight", "attn_k.bias")?,
            attn_v: lb("attn_v.weight", "attn_v.bias")?,
            attn_out: lb("attn_out.weight", "attn_out.bias")?,
            ffn_gate: lb("ffn_gate.weight", "ffn_gate.bias")?,
            ffn_up: lb("ffn_up.weight", "ffn_up.bias")?,
            ffn_down: lb("ffn_down.weight", "ffn_down.bias")?,
        });
    }
    Ok(VisionHandles {
        patch_embd: register_patch_embd(residency, "v.patch_embd.weight")?,
        patch_embd_1: register_patch_embd(residency, "v.patch_embd.weight.1")?,
        blocks,
        post_ln: norm(residency, "v.post_ln.weight")?,
        mm0: lin_bias(residency, "mm.0.weight", "mm.0.bias")?,
        mm2: lin_bias(residency, "mm.2.weight", "mm.2.bias")?,
    })
}

// ---------------------------------------------------------------------------
// Per-block weight views
// ---------------------------------------------------------------------------

struct BlockViews<'a> {
    ln1: GpuView<'a>,
    ln2: GpuView<'a>,
    attn_q: LinearBiasViews<'a>,
    attn_k: LinearBiasViews<'a>,
    attn_v: LinearBiasViews<'a>,
    attn_out: LinearBiasViews<'a>,
    ffn_gate: LinearBiasViews<'a>,
    ffn_up: LinearBiasViews<'a>,
    ffn_down: LinearBiasViews<'a>,
}

struct BlockBufs {
    ln1: BufRef,
    ln2: BufRef,
    attn_q: LinearBiasBufs,
    attn_k: LinearBiasBufs,
    attn_v: LinearBiasBufs,
    attn_out: LinearBiasBufs,
    ffn_gate: LinearBiasBufs,
    ffn_up: LinearBiasBufs,
    ffn_down: LinearBiasBufs,
}

impl<'a> BlockViews<'a> {
    async fn acquire<S: WeightSource>(
        h: &BlockHandles,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<BlockViews<'a>, ResidencyError<S::Error, WgpuError>> {
        Ok(BlockViews {
            ln1: residency.acquire(h.ln1, backend).await?,
            ln2: residency.acquire(h.ln2, backend).await?,
            attn_q: h.attn_q.acquire(residency, backend).await?,
            attn_k: h.attn_k.acquire(residency, backend).await?,
            attn_v: h.attn_v.acquire(residency, backend).await?,
            attn_out: h.attn_out.acquire(residency, backend).await?,
            ffn_gate: h.ffn_gate.acquire(residency, backend).await?,
            ffn_up: h.ffn_up.acquire(residency, backend).await?,
            ffn_down: h.ffn_down.acquire(residency, backend).await?,
        })
    }

    fn bufs(&self) -> BlockBufs {
        BlockBufs {
            ln1: self.ln1.buf(),
            ln2: self.ln2.buf(),
            attn_q: self.attn_q.bufs(),
            attn_k: self.attn_k.bufs(),
            attn_v: self.attn_v.bufs(),
            attn_out: self.attn_out.bufs(),
            ffn_gate: self.ffn_gate.bufs(),
            ffn_up: self.ffn_up.bufs(),
            ffn_down: self.ffn_down.bufs(),
        }
    }
}

// ---------------------------------------------------------------------------
// Forward helpers
// ---------------------------------------------------------------------------

/// `out = x @ wᵀ + bias` -> dense act `[rows, n]`, bf16 matmul through the qkv
/// site (dequant-once-bf16 under bf16 acts; weights are dense bf16 in-pool).
#[allow(clippy::too_many_arguments)]
fn biased<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    w: &LinearBiasBufs,
    rows: u32,
    n: u32,
    k: u32,
) -> Result<ActBuf<'wsp>, WgpuError> {
    let pre = alloc_matmul_out_buf(scope, bp, rows * n)?;
    let dims = scope.u32x4_uniform(rows, n, k, 0)?;
    let wv = scope.import_copy(w.weight);
    Block::dispatch_matmul_site(
        scope,
        bp,
        x,
        wv,
        pre,
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
    let bv = scope.import_copy(w.bias);
    let out = alloc_act(scope, bp, rows, n)?;
    let ba_u = bcast_add_uniform(scope, n)?;
    scope.bcast_add::<thinfer_core::ops::BcastAddF32>(
        &bp.bcast_add,
        pre,
        bv,
        ba_u,
        out.data,
        rows * n,
    )?;
    Ok(out)
}

/// One pre-norm vision block: `h += attn(ln1(h))`; `h += ffn(ln2(h))`.
/// `mask` is the segment (block-diagonal) additive SDPA mask for this block's
/// attention regime (full-attn or windowed). `rows` = sequence length (= N).
#[allow(clippy::too_many_arguments)]
fn block_forward_rows<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &QwenImageDitPipelines,
    h_in: BatchBuf<'wsp>,
    freqs: BatchBuf<'wsp>,
    mask: BatchBuf<'wsp>,
    h_out: BatchBuf<'wsp>,
    bufs: &'wsp BlockBufs,
    rows: u32,
) -> Result<(), WgpuError> {
    let bp = &pipelines.block;
    let n = config::HIDDEN as u32;
    let hd = config::HEAD_DIM as u32;
    let heads = config::N_HEADS as u32;
    let ffn = config::FFN_HIDDEN as u32;
    let eps = config::NORM_EPS;
    let scale = 1.0 / (hd as f32).sqrt();
    let h_in = ActBuf::dense(h_in);

    // --- attention ---
    let n1 = alloc_act(scope, bp, rows, n)?;
    let ln1 = scope.import_copy(bufs.ln1);
    op_rmsnorm(scope, bp, h_in, ln1, n1, rows, n, eps)?;

    let q = biased(scope, bp, n1, &bufs.attn_q, rows, n, n)?;
    let k = biased(scope, bp, n1, &bufs.attn_k, rows, n, n)?;
    let v = biased(scope, bp, n1, &bufs.attn_v, rows, n, n)?;

    let qr = alloc_act(scope, bp, rows, n)?;
    op_rope_halfrot(scope, bp, q, freqs, qr, rows, heads, hd)?;
    let kr = alloc_act(scope, bp, rows, n)?;
    op_rope_halfrot(scope, bp, k, freqs, kr, rows, heads, hd)?;

    let sa = alloc_act(scope, bp, rows, n)?;
    op_sdpa(
        scope, bp, qr, kr, v, mask, sa, 1, rows, rows, heads, heads, hd, scale, 1,
    )?;
    let attn = biased(scope, bp, sa, &bufs.attn_out, rows, n, n)?;
    let mid = alloc_act(scope, bp, rows, n)?;
    op_add(scope, bp, h_in, attn, mid)?;

    // --- SwiGLU FFN: down(silu(gate(x)) * up(x)) ---
    let n2 = alloc_act(scope, bp, rows, n)?;
    let ln2 = scope.import_copy(bufs.ln2);
    op_rmsnorm(scope, bp, mid, ln2, n2, rows, n, eps)?;
    let g = biased(scope, bp, n2, &bufs.ffn_gate, rows, ffn, n)?;
    let up = biased(scope, bp, n2, &bufs.ffn_up, rows, ffn, n)?;
    let gu = alloc_act(scope, bp, rows, ffn)?;
    op_silu_mul(scope, bp, g, up, gu)?;
    let down = biased(scope, bp, gu, &bufs.ffn_down, rows, n, ffn)?;
    op_add(scope, bp, mid, down, ActBuf::dense(h_out))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Host-side layout: patchify order, window index, segment mask
// ---------------------------------------------------------------------------

/// Merge-unit-major patch enumeration. For a `gh x gw` patch grid, yields each
/// raw patch's `(h, w)` in the row order the HF processor emits (`permute
/// (0,2,5,3,6,1,4,7)`): consecutive 4 rows = one 2x2 merge unit, units in
/// row-major merged-grid order. Returns `[(h, w); gh*gw]`.
fn patch_order(gh: usize, gw: usize) -> Vec<(usize, usize)> {
    let m = config::SPATIAL_MERGE;
    let mh = gh / m;
    let mw = gw / m;
    let mut out = Vec::with_capacity(gh * gw);
    for uh in 0..mh {
        for uw in 0..mw {
            for ih in 0..m {
                for iw in 0..m {
                    out.push((uh * m + ih, uw * m + iw));
                }
            }
        }
    }
    out
}

/// HF `get_window_index` for a single `(1, gh, gw)` image. Returns the
/// permutation over the `gh*gw/4` merge units (`window_index`) and the
/// cumulative window boundaries in RAW-PATCH units (`cu_window_seqlens`).
fn window_index(gh: usize, gw: usize) -> (Vec<usize>, Vec<usize>) {
    let m = config::SPATIAL_MERGE;
    let win = config::WINDOW_MERGE; // merge units per window side
    let lh = gh / m; // merged grid height
    let lw = gw / m;
    let pad_h = (win - lh % win) % win;
    let pad_w = (win - lw % win) % win;
    let nwh = (lh + pad_h) / win;
    let nww = (lw + pad_w) / win;

    let mut window_index = Vec::with_capacity(lh * lw);
    let mut cu = vec![0usize];
    // index[i, j] = merged-unit id at merged-grid (i, j), -100 for pad.
    for wh in 0..nwh {
        for ww in 0..nww {
            let mut count = 0usize;
            for ih in 0..win {
                for iw in 0..win {
                    let gi = wh * win + ih;
                    let gj = ww * win + iw;
                    if gi < lh && gj < lw {
                        window_index.push(gi * lw + gj);
                        count += 1;
                    }
                }
            }
            let prev = *cu.last().expect("cu non-empty");
            cu.push(prev + count * config::MERGE_UNIT);
        }
    }
    (window_index, cu)
}

/// Build the additive block-diagonal SDPA mask `[seq, seq]` (act dtype) for a
/// list of segment boundaries in raw-patch units: `0` within a segment, `-inf`
/// across. `seg_bounds` is `cu`-style (`[0, b1, b2, ..., seq]`).
fn segment_mask_bytes(seq: usize, seg_bounds: &[usize], act: ActDtype) -> Vec<u8> {
    let bpe = match act {
        ActDtype::F32 => 4,
        _ => 2,
    };
    let mut out = vec![0u8; seq * seq * bpe];
    let neg: &[u8] = match act {
        ActDtype::F32 => &f32::NEG_INFINITY.to_le_bytes(),
        ActDtype::Bf16 => &0xff80u16.to_le_bytes(),
        _ => &0xfc00u16.to_le_bytes(),
    };
    // seg_of[token] = segment id.
    let mut seg_of = vec![0usize; seq];
    for (s, win) in seg_bounds.windows(2).enumerate() {
        seg_of[win[0]..win[1]].fill(s);
    }
    for q in 0..seq {
        for kk in 0..seq {
            if seg_of[q] != seg_of[kk] {
                let off = (q * seq + kk) * bpe;
                out[off..off + bpe].copy_from_slice(neg);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum VisionError<SE: core::fmt::Debug> {
    Wgpu(WgpuError),
    Residency(ResidencyError<SE, WgpuError>),
    Load(LoadError),
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

#[derive(Clone, Debug)]
pub struct VisionOutput {
    /// `[N/4, PROJECTION_DIM=3584]` row-major, in RASTER (un-permuted) merged
    /// order: row `u` = merged token at merged-grid `(u / (gw/2), u % (gw/2))`.
    pub embeds: Vec<f32>,
    /// number of merged tokens (= gh*gw/4).
    pub tokens: usize,
}

pub struct VisionTower {
    rope: RopeEmbedder,
}

impl Default for VisionTower {
    fn default() -> Self {
        Self::new(64)
    }
}

impl VisionTower {
    /// `max_grid` sizes the 2D rope table (>= max patch grid side).
    pub fn new(max_grid: usize) -> Self {
        let g = max_grid.max(1);
        Self {
            rope: RopeEmbedder::new(
                config::ROPE_THETA,
                [config::HEAD_DIM / 2, config::HEAD_DIM / 2, 0],
                [g, g, 1],
            ),
        }
    }

    /// The DiT pipeline set this module shares (bf16 acts; block matmuls bf16).
    pub fn wgsl_configs() -> BlockWgslConfigs {
        use thinfer_core::ops::{WeightDtype, WgslConfig};
        let cfg = WgslConfig {
            bf16_quant_writes: false,
            act_dtype: ActDtype::Bf16,
            weight_dtype: WeightDtype::Bf16,
        };
        BlockWgslConfigs::uniform(cfg)
    }

    /// Run the vision tower on a patchified image.
    ///
    /// `pixel_values` is `[N, PATCH_ELEMS=1176]` row-major in merge-unit-major
    /// order (the HF processor layout, `N = gh*gw`). `(gh, gw)` is the raw
    /// patch grid (mult of `SPATIAL_MERGE`). Returns `[N/4, 3584]` LM embeds in
    /// raster merged order.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipelines: &QwenImageDitPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        handles: &VisionHandles,
        pixel_values: &[f32],
        gh: usize,
        gw: usize,
    ) -> Result<VisionOutput, VisionError<S::Error>> {
        let bp = &pipelines.block;
        assert_eq!(
            bp.act_dtype,
            ActDtype::Bf16,
            "vision tower expects bf16 acts"
        );
        let n = gh * gw;
        assert_eq!(
            pixel_values.len(),
            n * config::PATCH_ELEMS,
            "pixel_values must be [gh*gw, 1176]"
        );
        assert!(
            gh.is_multiple_of(config::SPATIAL_MERGE) && gw.is_multiple_of(config::SPATIAL_MERGE),
            "grid must be a multiple of the spatial merge size"
        );
        let hidden = config::HIDDEN as u32;
        let rows = n as u32;

        // --- patch embed = HF Conv3d(stride=kernel) = a per-patch dot product.
        // The Conv3d weight `[1280, C, T, P, P]` is stored as two temporal slabs
        // (`v.patch_embd.weight` = t=0, `.weight.1` = t=1), each `[1280, C, P, P]`
        // = `[1280, 588]`. The `[N, 1176]` row is laid out `[C, T, P, P]`, so the
        // t-th slab's 588 inputs are gathered at row index `c*(T*P*P) + t*(P*P) +
        // ph*P + pw`. out = W0 @ slot0 + W1 @ slot1.
        let pe = config::PATCH_SLOT_ELEMS; // 588 = C*P*P
        let pp = config::PATCH * config::PATCH; // 196
        let gather_slot = |t: usize| -> Vec<f32> {
            let mut slot = vec![0.0_f32; n * pe];
            for r in 0..n {
                let row = &pixel_values[r * config::PATCH_ELEMS..(r + 1) * config::PATCH_ELEMS];
                for c in 0..config::IN_CHANNELS {
                    for j in 0..pp {
                        let src = c * (config::TEMPORAL_PATCH * pp) + t * pp + j;
                        slot[r * pe + c * pp + j] = row[src];
                    }
                }
            }
            slot
        };
        let slot0 = gather_slot(0);
        let slot1 = gather_slot(1);
        let slot0_buf = upload_act(scratch, backend, bp, &slot0, rows * pe as u32)?;
        let slot1_buf = upload_act(scratch, backend, bp, &slot1, rows * pe as u32)?;
        let embed = scratch.alloc(bp.act_bytes(rows * hidden))?;
        {
            let v0 = residency.acquire(handles.patch_embd, backend).await?;
            let v1 = residency.acquire(handles.patch_embd_1, backend).await?;
            let scope = scratch.batch();
            let x0 = ActBuf::dense(scope.import_copy(slot0_buf.as_buf_ref()));
            let x1 = ActBuf::dense(scope.import_copy(slot1_buf.as_buf_ref()));
            let w0 = scope.import_copy(v0.buf());
            let w1 = scope.import_copy(v1.buf());
            let o0 = matmul_bf16(&scope, bp, x0, w0, rows, hidden, pe as u32)?;
            let o1 = matmul_bf16(&scope, bp, x1, w1, rows, hidden, pe as u32)?;
            let summed = alloc_act(&scope, bp, rows, hidden)?;
            op_add(&scope, bp, ActBuf::dense(o0), ActBuf::dense(o1), summed)?;
            let dst = scope.import_copy(embed.as_buf_ref());
            scope.copy_buffer_to_buffer(summed.data, 0, dst, 0, bp.act_bytes(rows * hidden))?;
            scope.submit_void().await?;
        }

        // --- host layout: patch order, window permutation, segment bounds ---
        let order = patch_order(gh, gw);
        assert_eq!(order.len(), n);
        let (win_idx, cu_window) = window_index(gh, gw);
        let merge_units = n / config::MERGE_UNIT;
        assert_eq!(win_idx.len(), merge_units);

        // --- 2D rope freqs in merge-unit-major (patch) order, then reordered by
        //     the window permutation (same as hidden) ---
        let mut pos_ids = Vec::with_capacity(n * 3);
        for &(h, w) in &order {
            pos_ids.push(h as i32);
            pos_ids.push(w as i32);
            pos_ids.push(0);
        }
        let freqs_raw = self.rope.lookup(&pos_ids); // [n, head_dim]
        let freqs = reorder_by_window(&freqs_raw, &win_idx, config::HEAD_DIM);
        let freqs_buf = upload_freqs(scratch, backend, bp, &freqs)?;

        // --- reorder the patch-embed hidden by the window permutation ---
        let embed_host = {
            let bytes = backend
                .read_buffer(embed.id(), 0, bp.act_bytes(rows * hidden))
                .await?;
            seq::act_readback_to_f32(bp.act_dtype, &bytes, n * config::HIDDEN)
        };
        let reordered = reorder_by_window(&embed_host, &win_idx, config::HIDDEN);
        let mut cur = upload_act(scratch, backend, bp, &reordered, rows * hidden)?;

        // --- segment masks: full = [0, N]; windowed = cu_window (raw units) ---
        let full_bounds = [0usize, n];
        let full_mask = segment_mask_bytes(n, &full_bounds, bp.act_dtype);
        let win_mask = segment_mask_bytes(n, &cu_window, bp.act_dtype);
        let full_mask_buf = scratch.alloc(full_mask.len() as u64)?;
        backend.write_buffer(full_mask_buf.id, 0, &full_mask)?;
        let win_mask_buf = scratch.alloc(win_mask.len() as u64)?;
        backend.write_buffer(win_mask_buf.id, 0, &win_mask)?;
        let freqs_ref = freqs_buf.as_buf_ref();

        // --- 32 blocks (prefetch next block's weights) ---
        let mut pending = Some(BlockViews::acquire(&handles.blocks[0], residency, backend).await?);
        for idx in 0..config::DEPTH {
            let _g = trace::scope!(format!("qwen_image.vision.block.{idx}")).entered();
            let views = pending.take().expect("pending acquire missing");
            let bufs = views.bufs();
            let mask_ref = if is_fullatt(idx) {
                full_mask_buf.as_buf_ref()
            } else {
                win_mask_buf.as_buf_ref()
            };
            let nxt = scratch.alloc(bp.act_bytes(rows * hidden))?;
            {
                let scope = scratch.batch();
                block_forward_rows(
                    &scope,
                    pipelines,
                    scope.import_copy(cur.as_buf_ref()),
                    scope.import_copy(freqs_ref),
                    scope.import_copy(mask_ref),
                    scope.import_copy(nxt.as_buf_ref()),
                    &bufs,
                    rows,
                )?;

                let next_idx = idx + 1;
                let next_acquire = async {
                    match handles.blocks.get(next_idx) {
                        Some(h) => Ok::<_, ResidencyError<S::Error, WgpuError>>(Some(
                            BlockViews::acquire(h, residency, backend).await?,
                        )),
                        None => Ok(None),
                    }
                };
                let submit_fut = scope.submit_void().instrument(
                    tracing::debug_span!(target: PHASE, "qwen_image.vision.submit", idx),
                );
                let (submit_res, next_res) = futures::join!(submit_fut, next_acquire);
                submit_res?;
                pending = next_res?;
            }
            drop(views);
            cur = nxt;
        }

        // --- merger: post_ln (RMSNorm) -> view [N/4, 5120] -> mm.0 -> GELU ->
        //     mm.2 -> [N/4, 3584] ---
        let mh = config::MERGER_HIDDEN as u32;
        let proj = config::PROJECTION_DIM as u32;
        let merged_rows = merge_units as u32;
        let merged_buf = scratch.alloc(bp.act_bytes(merged_rows * proj))?;
        {
            let pln = residency.acquire(handles.post_ln, backend).await?;
            let mm0 = handles.mm0.acquire(residency, backend).await?;
            let mm2 = handles.mm2.acquire(residency, backend).await?;
            let (mm0b, mm2b) = (mm0.bufs(), mm2.bufs());
            let scope = scratch.batch();
            let x = ActBuf::dense(scope.import_copy(cur.as_buf_ref()));
            let pw = scope.import_copy(pln.buf());
            // RMSNorm per token over HIDDEN; the [N/4, 5120] view is just a
            // reshape of the contiguous [N, 1280] normed buffer.
            let normed = alloc_act(&scope, bp, rows, hidden)?;
            op_rmsnorm(&scope, bp, x, pw, normed, rows, hidden, config::NORM_EPS)?;
            let up = biased(&scope, bp, normed, &mm0b, merged_rows, mh, mh)?;
            let g = alloc_act(&scope, bp, merged_rows, mh)?;
            scope.dispatch_op::<GeluF32>(&pipelines.gelu, &[up.data], g.data)?;
            let out = biased(&scope, bp, g, &mm2b, merged_rows, proj, mh)?;
            let dst = scope.import_copy(merged_buf.as_buf_ref());
            scope.copy_buffer_to_buffer(out.data, 0, dst, 0, bp.act_bytes(merged_rows * proj))?;
            scope.submit_void().await?;
        }

        let bytes = backend
            .read_buffer(merged_buf.id(), 0, bp.act_bytes(merged_rows * proj))
            .await?;
        let merged =
            seq::act_readback_to_f32(bp.act_dtype, &bytes, merge_units * config::PROJECTION_DIM);
        // un-permute merged tokens: argsort(window_index).
        let embeds = unsort_merged(&merged, &win_idx, config::PROJECTION_DIM);
        Ok(VisionOutput {
            embeds,
            tokens: merge_units,
        })
    }
}

/// Reorder a `[n, dim]` row-major buffer by `window_index` over the `n/4` merge
/// units: `out[unit u][i] = in[ win_idx[u] ][i]` for the 4 rows of each unit.
fn reorder_by_window(src: &[f32], win_idx: &[usize], dim: usize) -> Vec<f32> {
    let mu = config::MERGE_UNIT;
    let units = win_idx.len();
    let mut out = vec![0.0_f32; units * mu * dim];
    for (dst_u, &src_u) in win_idx.iter().enumerate() {
        for r in 0..mu {
            let src_row = (src_u * mu + r) * dim;
            let dst_row = (dst_u * mu + r) * dim;
            out[dst_row..dst_row + dim].copy_from_slice(&src[src_row..src_row + dim]);
        }
    }
    out
}

/// Inverse of the merge-unit permutation for the merger output (one row per
/// merged token): `out[ win_idx[u] ] = merged[u]`.
fn unsort_merged(merged: &[f32], win_idx: &[usize], dim: usize) -> Vec<f32> {
    let units = win_idx.len();
    let mut out = vec![0.0_f32; units * dim];
    for (u, &dst_u) in win_idx.iter().enumerate() {
        out[dst_u * dim..(dst_u + 1) * dim].copy_from_slice(&merged[u * dim..(u + 1) * dim]);
    }
    out
}

/// bf16-weight matmul through the qkv site (dequant-once-bf16 under bf16 acts).
fn matmul_bf16<'wsp>(
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

fn upload_act(
    scratch: &Workspace<WgpuBackend>,
    backend: &WgpuBackend,
    bp: &BlockPipelines,
    host: &[f32],
    n: u32,
) -> Result<thinfer_core::workspace::WsBuf<WgpuBackend>, WgpuError> {
    let buf = scratch.alloc(bp.act_bytes(n))?;
    let bytes = seq::act_upload_bytes(bp.act_dtype, host);
    backend.write_buffer(buf.id, 0, &bytes)?;
    Ok(buf)
}

fn upload_freqs(
    scratch: &Workspace<WgpuBackend>,
    backend: &WgpuBackend,
    bp: &BlockPipelines,
    freqs: &[f32],
) -> Result<thinfer_core::workspace::WsBuf<WgpuBackend>, WgpuError> {
    let bytes = seq::freqs_upload_bytes(bp.act_dtype, freqs);
    let buf = scratch.alloc(bytes.len() as u64)?;
    backend.write_buffer(buf.id, 0, &bytes)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fullatt_blocks_are_7_15_23_31() {
        let full: Vec<usize> = (0..config::DEPTH).filter(|&i| is_fullatt(i)).collect();
        assert_eq!(full, vec![7, 15, 23, 31]);
    }

    #[test]
    fn patch_order_is_merge_unit_major() {
        // 4x4 grid -> 4 merge units; first unit = (0,0),(0,1),(1,0),(1,1).
        let o = patch_order(4, 4);
        assert_eq!(o.len(), 16);
        assert_eq!(&o[0..4], &[(0, 0), (0, 1), (1, 0), (1, 1)]);
        // second unit (merged col 1) = (0,2),(0,3),(1,2),(1,3).
        assert_eq!(&o[4..8], &[(0, 2), (0, 3), (1, 2), (1, 3)]);
    }

    #[test]
    fn window_index_single_window_is_identity() {
        // 8x8 patch grid -> 4x4 merged -> exactly one 4x4 window -> identity.
        let (wi, cu) = window_index(8, 8);
        assert_eq!(wi.len(), 16);
        assert_eq!(wi, (0..16).collect::<Vec<_>>());
        // one window, all 16 merge units = 64 raw patches.
        assert_eq!(cu, vec![0, 64]);
    }

    #[test]
    fn window_index_multi_window_partitions() {
        // 16x16 patch grid -> 8x8 merged -> 2x2 windows of 4x4 merge units.
        let (wi, cu) = window_index(16, 16);
        assert_eq!(wi.len(), 64);
        // permutation property.
        let mut sorted = wi.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..64).collect::<Vec<_>>());
        // 4 windows, each 16 merge units * 4 = 64 raw patches.
        assert_eq!(cu, vec![0, 64, 128, 192, 256]);
        // first window covers merged-grid rows 0..4, cols 0..4.
        assert_eq!(&wi[0..4], &[0, 1, 2, 3]);
        assert_eq!(&wi[4..8], &[8, 9, 10, 11]);
    }

    #[test]
    fn reorder_then_unsort_roundtrips() {
        let (wi, _) = window_index(16, 16);
        let dim = 2;
        let units = wi.len();
        // distinct per-row data.
        let src: Vec<f32> = (0..units * config::MERGE_UNIT * dim)
            .map(|x| x as f32)
            .collect();
        let re = reorder_by_window(&src, &wi, dim);
        // merged: take one representative row per unit (row 0 of each).
        let merged: Vec<f32> = (0..units)
            .flat_map(|u| {
                re[u * config::MERGE_UNIT * dim..u * config::MERGE_UNIT * dim + dim].to_vec()
            })
            .collect();
        let un = unsort_merged(&merged, &wi, dim);
        // un[orig_unit] must equal src row0 of that orig unit.
        for u in 0..units {
            let want = &src[u * config::MERGE_UNIT * dim..u * config::MERGE_UNIT * dim + dim];
            assert_eq!(&un[u * dim..(u + 1) * dim], want);
        }
    }
}
