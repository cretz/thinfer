//! Qwen-Image-2.1 single-stream DiT. Ground truth: diffusers
//! `transformer_qwenimage21.py`.
//!
//! Every block: `LayerNorm` (no affine) -> `x * (1 + scale)` -> q / k / v
//! (per-head RMSNorm on q, k; interleaved 3-axis RoPE) -> block-causal
//! attention -> `out` -> `x += tanh(gate) * a`, then the same around a SwiGLU
//! MLP. ONE shared modulation feeds every block; the text / condition rows
//! (the prefix) read it at `t = 0`, the target rows at the step's `t`.
//!
//! Because the prefix is modulated from `t = 0` and is causal with respect to
//! the target (no prefix row attends a target row), its activations never
//! change across steps. [`Qwen21Dit::prefill`] runs it ONCE and keeps each
//! block's prefix K / V; [`Qwen21Dit::step`] then runs only the target rows,
//! attending `[cached prefix K/V ; target K/V]` with no mask (the reference's
//! `use_kv_cache` path, which is exact).
//!
//! bf16 activations; matmul weights are dequantized per use by their own GGUF
//! kind (the checkpoint mixes kinds per tensor) and run on the accelerator's
//! bf16 GEMM when present.

use std::collections::HashMap;

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError, WgpuPipeline};
use thinfer_core::ops::dequant::{DequantTarget, build_wgsl, layout};
use thinfer_core::ops::{
    ActDtype, BcastFmaF32, BcastModulateF32, DequantScheme, GeluF32, LayerNormF32, Op,
    RmsNormGemmaF32, RmsNormOp, SiluF32, SiluMulFusedF32, SiluMulFusedOp, WeightDtype, WgslConfig,
};
use thinfer_core::quant::QuantKind;
use thinfer_core::residency::{GpuView, ResidencyError, WeightResidency};
use thinfer_core::trace::{self, PHASE};
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace, WsBuf};
use tracing::Instrument;

use crate::common::block::{
    ActBuf, Block, BlockPipelines, BlockWgslConfigs, CoopmatSites, DenseActSites, DequantStep,
    alloc_act, alloc_matmul_out_buf, op_rmsnorm, op_rope, op_sdpa_f16,
};
use crate::common::embedders::rmsnorm_uniform;
use crate::common::seq;

use super::config;
use super::loader::{BlockHandles, Linear, Qwen21DitHandles};
use super::rope::{Qwen21Rope, Segment};

/// The block pipeline set plus a dequant step per GGUF kind and the two
/// elementwise kernels the common set lacks.
pub struct Qwen21DitPipelines {
    pub block: BlockPipelines,
    dequant: HashMap<QuantKind, DequantStep>,
    silu_mul_fused: WgpuPipeline,
    rmsnorm_zc: WgpuPipeline,
}

/// bf16 acts throughout. The matmul sites declare a `Quant` weight, which
/// selects the "pre-dequantized N-major workspace" convention the per-kind
/// dequant below produces (the kind itself is irrelevant to the WGSL).
/// `fast_sdpa` builds the f16 subgroup attention (Q / K / V are normed and
/// roped, O(1)) and its bf16 <-> f16 casts.
fn block_cfgs() -> BlockWgslConfigs {
    let ops = WgslConfig {
        bf16_quant_writes: false,
        act_dtype: ActDtype::Bf16,
        weight_dtype: WeightDtype::Bf16,
    };
    let mm = WgslConfig {
        weight_dtype: WeightDtype::Quant(QuantKind::Q4_K),
        ..ops
    };
    BlockWgslConfigs {
        matmul_qkv: mm,
        matmul_qkv_self: mm,
        matmul_proj: mm,
        matmul_ffn_up: mm,
        matmul_ffn_down: mm,
        matmul_adaln: mm,
        ops,
        i8_sdpa: false,
        dense_acts: DenseActSites::default(),
        coopmat_acts: CoopmatSites::default(),
        large_d_sdpa: false,
        fast_sdpa: true,
        decode_sdpa: false,
    }
}

impl Qwen21DitPipelines {
    pub async fn compile(backend: &WgpuBackend) -> Result<Self, WgpuError> {
        let cfgs = block_cfgs();
        let block = BlockPipelines::compile(backend, &cfgs).await?;
        let mut dequant = HashMap::new();
        for kind in [
            QuantKind::Q8_0,
            QuantKind::Q4_K,
            QuantKind::Q5_K,
            QuantKind::Q6_K,
        ] {
            let pipeline = backend
                .create_pipeline(
                    &format!("qwen21_dequant_{kind:?}"),
                    &build_wgsl(kind, DequantTarget::Bf16),
                    "main",
                    layout(),
                )
                .await?;
            dequant.insert(
                kind,
                DequantStep {
                    pipeline,
                    scheme: DequantScheme::Block(kind),
                },
            );
        }
        let silu_mul_fused = backend
            .create_pipeline(
                "qwen21_silu_mul_fused",
                <SiluMulFusedF32 as SiluMulFusedOp>::wgsl(&cfgs.ops),
                "main",
                <SiluMulFusedF32 as SiluMulFusedOp>::layout(),
            )
            .await?;
        let rmsnorm_zc = backend
            .create_pipeline(
                "qwen21_rmsnorm_zero_centred",
                <RmsNormGemmaF32 as RmsNormOp>::wgsl(&cfgs.ops),
                "main",
                <RmsNormGemmaF32 as RmsNormOp>::layout(),
            )
            .await?;
        Ok(Self {
            block,
            dequant,
            silu_mul_fused,
            rmsnorm_zc,
        })
    }
}

// --- ops -----------------------------------------------------------------------

/// `[rows, n] = x[rows, k] @ W^T`, routed by the weight's own storage.
fn linear<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pipes: &Qwen21DitPipelines,
    x: ActBuf<'w>,
    (w, kind): (BufRef, Option<QuantKind>),
    rows: u32,
    n: u32,
    k: u32,
) -> Result<ActBuf<'w>, WgpuError> {
    let bp = &pipes.block;
    let out = alloc_matmul_out_buf(scope, bp, rows * n)?;
    let dims = scope.u32x4_uniform(rows, n, k, 0)?;
    let w = scope.import_copy(w);
    let (dq, pipe, op) = match kind {
        Some(q) => (
            Some(pipes.dequant.get(&q).unwrap_or_else(|| {
                panic!("qwen-image-2.1: no dequant for {q:?} (unexpected GGUF kind)")
            })),
            &bp.matmul_qkv,
            &bp.matmuls.qkv,
        ),
        None => (None, &bp.matmul_module, &bp.matmuls.module),
    };
    Block::dispatch_matmul_site(
        scope, bp, x, w, out, dims, None, None, dq, pipe, op, rows, n, k,
    )?;
    Ok(ActBuf::dense(out))
}

fn layer_norm<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'w>,
    rows: u32,
    dim: u32,
) -> Result<ActBuf<'w>, WgpuError> {
    let out = alloc_act(scope, bp, rows, dim)?;
    let u = scope.u32x4_uniform(rows, dim, config::EPS.to_bits(), 0)?;
    scope.layernorm::<LayerNormF32>(&bp.layernorm, x.data, u, out.data, rows)?;
    Ok(out)
}

/// `x * (1 + scale)` (`shift` = a zero vector).
fn modulate<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'w>,
    scale: BatchBuf<'w>,
    zero: BatchBuf<'w>,
    rows: u32,
    dim: u32,
) -> Result<ActBuf<'w>, WgpuError> {
    let out = alloc_act(scope, bp, rows, dim)?;
    let u = scope.u32x4_uniform(dim, 1.0_f32.to_bits(), 0, 0)?;
    scope.bcast_modulate::<BcastModulateF32>(
        &bp.bcast_modulate,
        x.data,
        scale,
        zero,
        u,
        out.data,
        rows * dim,
    )?;
    Ok(out)
}

/// `x + gate * y`, `gate` broadcast over rows.
#[allow(clippy::too_many_arguments)]
fn gated_add<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'w>,
    gate: BatchBuf<'w>,
    y: ActBuf<'w>,
    out: BatchBuf<'w>,
    rows: u32,
    dim: u32,
) -> Result<(), WgpuError> {
    let u = scope.u32x4_uniform(dim, 0, 0, 0)?;
    scope.bcast_fma::<BcastFmaF32>(&bp.bcast_fma, x.data, gate, y.data, u, out, rows * dim)
}

fn act_op<'w, O: Op>(
    scope: &BatchScope<'w, WgpuBackend>,
    bp: &BlockPipelines,
    pipeline: &WgpuPipeline,
    x: ActBuf<'w>,
    elems: u32,
) -> Result<ActBuf<'w>, WgpuError> {
    let out = ActBuf::dense(scope.alloc(bp.act_bytes(elems))?);
    scope.dispatch_op::<O>(pipeline, &[x.data], out.data)?;
    Ok(out)
}

/// Rows `[r0, r0 + n)` of a `[rows, width]` activation.
fn rows_of<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    bp: &BlockPipelines,
    b: BatchBuf<'w>,
    r0: u32,
    n: u32,
    width: u32,
) -> BatchBuf<'w> {
    scope.subview(&b, bp.act_bytes(r0 * width), bp.act_bytes(n * width))
}

// --- weights per block -----------------------------------------------------------

struct BlockViews<'a> {
    views: Vec<GpuView<'a>>,
}

struct BlockBufs {
    norm_q: BufRef,
    norm_k: BufRef,
    q: (BufRef, Option<QuantKind>),
    k: (BufRef, Option<QuantKind>),
    v: (BufRef, Option<QuantKind>),
    out: (BufRef, Option<QuantKind>),
    gate_up: (BufRef, Option<QuantKind>),
    down: (BufRef, Option<QuantKind>),
}

impl<'a> BlockViews<'a> {
    async fn acquire<S: WeightSource>(
        h: &BlockHandles,
        r: &'a WeightResidency<S>,
        b: &WgpuBackend,
    ) -> Result<Self, ResidencyError<S::Error, WgpuError>> {
        let mut views = Vec::with_capacity(8);
        for handle in [
            h.norm_q,
            h.norm_k,
            h.to_q.handle,
            h.to_k.handle,
            h.to_v.handle,
            h.to_out.handle,
            h.gate_up.handle,
            h.down.handle,
        ] {
            views.push(r.acquire(handle, b).await?);
        }
        Ok(Self { views })
    }

    fn bufs(&self, h: &BlockHandles) -> BlockBufs {
        let v = |i: usize| self.views[i].buf();
        let l = |i: usize, w: &Linear| (v(i), w.kind);
        BlockBufs {
            norm_q: v(0),
            norm_k: v(1),
            q: l(2, &h.to_q),
            k: l(3, &h.to_k),
            v: l(4, &h.to_v),
            out: l(5, &h.to_out),
            gate_up: l(6, &h.gate_up),
            down: l(7, &h.down),
        }
    }
}

/// How a block's attention runs.
enum Attention<'a, 'w> {
    /// The prefix pass: block-causal over `segments` (all prefix runs, in
    /// order); stores K / V into `cache` (rows `[0, rows)`).
    Prefix {
        segments: &'a [Segment],
        masks: &'a [Option<BatchBuf<'w>>],
        cache: (BatchBuf<'w>, BatchBuf<'w>),
    },
    /// A step: the target rows attend `[prefix ; target]`.
    Target {
        cache: (BatchBuf<'w>, BatchBuf<'w>),
        prefix_rows: u32,
    },
}

/// One block over `rows` rows of `hidden_in` into `hidden_out`. `modv` holds
/// `[scale1, tanh(gate1), scale2, tanh(gate2)]` (each `[DIM]`).
#[allow(clippy::too_many_arguments)]
fn block_forward<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pipes: &Qwen21DitPipelines,
    hidden_in: BatchBuf<'w>,
    hidden_out: BatchBuf<'w>,
    modv: BatchBuf<'w>,
    zero: BatchBuf<'w>,
    freqs: BatchBuf<'w>,
    rows: u32,
    attn: Attention<'_, 'w>,
    w: &BlockBufs,
) -> Result<(), WgpuError> {
    let bp = &pipes.block;
    let dim = config::DIM as u32;
    let (heads, hd) = (config::N_HEADS as u32, config::HEAD_DIM as u32);
    let inner = config::MLP_INNER as u32;
    let x = ActBuf::dense(hidden_in);
    let mod_part = |i: u32| scope.subview(&modv, bp.act_bytes(i * dim), bp.act_bytes(dim));
    let (scale1, gate1, scale2, gate2) = (mod_part(0), mod_part(1), mod_part(2), mod_part(3));

    // --- attention ---
    let n1 = layer_norm(scope, bp, x, rows, dim)?;
    let a = modulate(scope, bp, n1, scale1, zero, rows, dim)?;
    let q = linear(scope, pipes, a, w.q, rows, dim, dim)?;
    let k = linear(scope, pipes, a, w.k, rows, dim, dim)?;
    let v = linear(scope, pipes, a, w.v, rows, dim, dim)?;
    let qk_norm_rope = |t: ActBuf<'w>, norm: BufRef| -> Result<ActBuf<'w>, WgpuError> {
        let normed = alloc_act(scope, bp, rows * heads, hd)?;
        op_rmsnorm(
            scope,
            bp,
            t,
            scope.import_copy(norm),
            normed,
            rows * heads,
            hd,
            config::EPS,
        )?;
        let roped = alloc_act(scope, bp, rows, dim)?;
        op_rope(scope, bp, normed, freqs, roped, rows, heads, hd)?;
        Ok(roped)
    };
    let q = qk_norm_rope(q, w.norm_q)?;
    let k = qk_norm_rope(k, w.norm_k)?;
    let scale = 1.0 / (hd as f32).sqrt();
    let sa = alloc_act(scope, bp, rows, dim)?;
    match attn {
        Attention::Prefix {
            segments,
            masks,
            cache: (ck, cv),
        } => {
            let bytes = bp.act_bytes(rows * dim);
            scope.copy_buffer_to_buffer(k.data, 0, ck, 0, bytes)?;
            scope.copy_buffer_to_buffer(v.data, 0, cv, 0, bytes)?;
            let mut start = 0u32;
            for (seg, mask) in segments.iter().zip(masks) {
                let len = seg.len() as u32;
                let end = start + len;
                // A text run masks its own causal triangle over an even-padded
                // key range (the f16 mask is read in pairs); an image block
                // sees every key before it plus itself, unmasked.
                let keys = if mask.is_some() {
                    end.next_multiple_of(2)
                } else {
                    end
                };
                let (m, has_mask) = match mask {
                    Some(m) => (*m, 1),
                    None => (k.data, 0),
                };
                op_sdpa_f16(
                    scope,
                    bp,
                    ActBuf::dense(rows_of(scope, bp, q.data, start, len, dim)),
                    k,
                    v,
                    m,
                    ActBuf::dense(rows_of(scope, bp, sa.data, start, len, dim)),
                    1,
                    len,
                    keys,
                    heads,
                    heads,
                    hd,
                    scale,
                    has_mask,
                )?;
                start = end;
            }
        }
        Attention::Target {
            cache: (ck, cv),
            prefix_rows,
        } => {
            // [prefix ; target] K and V, the target rows written after the
            // cached prefix rows.
            let total = prefix_rows + rows;
            let kk = scope.alloc(bp.act_bytes(total * dim))?;
            let vv = scope.alloc(bp.act_bytes(total * dim))?;
            let pb = bp.act_bytes(prefix_rows * dim);
            let tb = bp.act_bytes(rows * dim);
            scope.copy_buffer_to_buffer(ck, 0, kk, 0, pb)?;
            scope.copy_buffer_to_buffer(cv, 0, vv, 0, pb)?;
            scope.copy_buffer_to_buffer(k.data, 0, kk, pb, tb)?;
            scope.copy_buffer_to_buffer(v.data, 0, vv, pb, tb)?;
            op_sdpa_f16(
                scope,
                bp,
                q,
                ActBuf::dense(kk),
                ActBuf::dense(vv),
                kk,
                sa,
                1,
                rows,
                total,
                heads,
                heads,
                hd,
                scale,
                0,
            )?;
        }
    }
    let o = linear(scope, pipes, sa, w.out, rows, dim, dim)?;
    let h1 = alloc_act(scope, bp, rows, dim)?;
    gated_add(scope, bp, x, gate1, o, h1.data, rows, dim)?;

    // --- SwiGLU MLP ---
    let n2 = layer_norm(scope, bp, h1, rows, dim)?;
    let m = modulate(scope, bp, n2, scale2, zero, rows, dim)?;
    let gu = linear(scope, pipes, m, w.gate_up, rows, 2 * inner, dim)?;
    let act = alloc_act(scope, bp, rows, inner)?;
    let u = scope.u32x4_uniform(rows, inner, 0, 0)?;
    scope.silu_mul_fused::<SiluMulFusedF32>(
        &pipes.silu_mul_fused,
        gu.data,
        u,
        act.data,
        rows * inner,
    )?;
    let mlp = linear(scope, pipes, act, w.down, rows, dim, inner)?;
    gated_add(scope, bp, h1, gate2, mlp, hidden_out, rows, dim)
}

// --- driver --------------------------------------------------------------------

#[derive(Debug)]
pub enum DitError<SE: core::fmt::Debug> {
    Wgpu(WgpuError),
    Residency(ResidencyError<SE, WgpuError>),
}
impl<SE: core::fmt::Debug> From<WgpuError> for DitError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for DitError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

/// One run of the prefix as the encoder laid it out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixPart {
    /// `len` encoder rows starting at encoder row `start` (through `txt_in`).
    Text { start: usize, len: usize },
    /// A condition image: `h x w` latent tokens starting at token `start` of
    /// the condition latents (through `img_in`).
    Image { start: usize, h: usize, w: usize },
}

/// The prefix pass's output, reused by every step of the request.
pub struct Prefix {
    /// Prefix rows (text + condition tokens).
    rows: u32,
    /// Per block `(K, V)` `[rows, DIM]` (post norm + rope).
    cache: Vec<(WsBuf<WgpuBackend>, WsBuf<WgpuBackend>)>,
    /// RoPE freqs over `[prefix ; target]` (the target rows' slice is per step).
    freqs: WsBuf<WgpuBackend>,
    zero: WsBuf<WgpuBackend>,
}

impl Prefix {
    pub fn rows(&self) -> usize {
        self.rows as usize
    }
}

/// The step-invariant modulation inputs of one `t`, on the host.
struct ModulationRow {
    /// `[scale1, tanh(gate1), scale2, tanh(gate2)]`, `4 * DIM`.
    blocks: Vec<f32>,
    /// The final norm's scale, `DIM`.
    out_scale: Vec<f32>,
}

pub struct Qwen21Dit {
    rope: Qwen21Rope,
}

impl Default for Qwen21Dit {
    fn default() -> Self {
        Self::new()
    }
}

fn upload(
    ws: &Workspace<WgpuBackend>,
    backend: &WgpuBackend,
    bp: &BlockPipelines,
    host: &[f32],
) -> Result<WsBuf<WgpuBackend>, WgpuError> {
    let bytes = seq::act_upload_bytes(bp.act_dtype, host);
    let buf = ws.alloc(bytes.len() as u64)?;
    backend.write_buffer(buf.id(), 0, &bytes)?;
    Ok(buf)
}

async fn download(
    backend: &WgpuBackend,
    bp: &BlockPipelines,
    buf: &WsBuf<WgpuBackend>,
    elems: u32,
) -> Result<Vec<f32>, WgpuError> {
    let bytes = backend
        .read_buffer(buf.id(), 0, bp.act_bytes(elems))
        .await?;
    Ok(seq::act_readback_to_f32(
        bp.act_dtype,
        &bytes,
        elems as usize,
    ))
}

/// `[cos(a) ++ sin(a)]`, `a_j = 1000 t exp(-ln(10000) j / 128)`.
fn timestep_sinusoid(t: f32) -> Vec<f32> {
    let half = config::TIMESTEP_DIM / 2;
    let mut out = vec![0.0f32; 2 * half];
    for j in 0..half {
        let f = (-(config::TIMESTEP_MAX_PERIOD.ln()) * j as f32 / half as f32).exp();
        let a = config::TIMESTEP_SCALE * t * f;
        out[j] = a.cos();
        out[half + j] = a.sin();
    }
    out
}

/// The additive f16 causal mask of a text run `[start, start + len)` over the
/// even-padded keys `[0, keys)`: a query sees every key up to itself.
fn text_mask_bytes(start: usize, len: usize, keys: usize) -> Vec<u8> {
    const NEG_INF_F16: u16 = 0xfc00;
    let mut out = vec![0u8; len * keys * 2];
    for q in 0..len {
        for k in start + q + 1..keys {
            let off = (q * keys + k) * 2;
            out[off..off + 2].copy_from_slice(&NEG_INF_F16.to_le_bytes());
        }
    }
    out
}

impl Qwen21Dit {
    pub fn new() -> Self {
        Self {
            rope: Qwen21Rope::new(),
        }
    }

    /// Time embedding -> shared modulation + final-norm scale for each `t`.
    async fn modulation<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipes: &Qwen21DitPipelines,
        res: &WeightResidency<S>,
        ws: &Workspace<WgpuBackend>,
        h: &Qwen21DitHandles,
        ts: &[f32],
    ) -> Result<Vec<ModulationRow>, DitError<S::Error>> {
        let bp = &pipes.block;
        let dim = config::DIM as u32;
        let rows = ts.len() as u32;
        let sin: Vec<f32> = ts.iter().flat_map(|&t| timestep_sinusoid(t)).collect();
        let sin_buf = upload(ws, backend, bp, &sin)?;
        let mods = ws.alloc(bp.act_bytes(rows * 4 * dim))?;
        let outs = ws.alloc(bp.act_bytes(rows * dim))?;
        {
            let t = &h.top;
            let views = [
                res.acquire(t.time_1.handle, backend).await?,
                res.acquire(t.time_2.handle, backend).await?,
                res.acquire(t.modulation.handle, backend).await?,
                res.acquire(t.norm_out.handle, backend).await?,
            ];
            let scope = ws.batch();
            let x = ActBuf::dense(scope.import_copy(sin_buf.as_buf_ref()));
            let ts_dim = config::TIMESTEP_DIM as u32;
            let h1 = linear(
                &scope,
                pipes,
                x,
                (views[0].buf(), t.time_1.kind),
                rows,
                dim,
                ts_dim,
            )?;
            let a1 = act_op::<SiluF32>(&scope, bp, &bp.silu, h1, rows * dim)?;
            let temb = linear(
                &scope,
                pipes,
                a1,
                (views[1].buf(), t.time_2.kind),
                rows,
                dim,
                dim,
            )?;
            let st = act_op::<SiluF32>(&scope, bp, &bp.silu, temb, rows * dim)?;
            let m = linear(
                &scope,
                pipes,
                st,
                (views[2].buf(), t.modulation.kind),
                rows,
                4 * dim,
                dim,
            )?;
            let o = linear(
                &scope,
                pipes,
                st,
                (views[3].buf(), t.norm_out.kind),
                rows,
                dim,
                dim,
            )?;
            let md = scope.import_copy(mods.as_buf_ref());
            let od = scope.import_copy(outs.as_buf_ref());
            scope.copy_buffer_to_buffer(m.data, 0, md, 0, bp.act_bytes(rows * 4 * dim))?;
            scope.copy_buffer_to_buffer(o.data, 0, od, 0, bp.act_bytes(rows * dim))?;
            scope.submit_void().await?;
        }
        let m = download(backend, bp, &mods, rows * 4 * dim).await?;
        let o = download(backend, bp, &outs, rows * dim).await?;
        let d = config::DIM;
        Ok((0..ts.len())
            .map(|r| {
                let mut blocks = m[r * 4 * d..(r + 1) * 4 * d].to_vec();
                // Gates enter the residual through tanh.
                for g in [1, 3] {
                    for v in &mut blocks[g * d..(g + 1) * d] {
                        *v = v.tanh();
                    }
                }
                ModulationRow {
                    blocks,
                    out_scale: o[r * d..(r + 1) * d].to_vec(),
                }
            })
            .collect())
    }

    /// Run the prefix (text + condition images) through every block ONCE and
    /// keep each block's K / V. `encoder` = the conditioner rows `[n, TEXT_DIM]`
    /// (system rows already dropped), `cond` = the condition images' latent
    /// tokens `[m, LATENT_CH]`, `parts` = the prefix in sequence order,
    /// `target` = the target's latent grid.
    #[allow(clippy::too_many_arguments)]
    pub async fn prefill<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipes: &Qwen21DitPipelines,
        res: &WeightResidency<S>,
        ws: &Workspace<WgpuBackend>,
        h: &Qwen21DitHandles,
        encoder: &[f32],
        cond: &[f32],
        parts: &[PrefixPart],
        target: (usize, usize),
    ) -> Result<Prefix, DitError<S::Error>> {
        let bp = &pipes.block;
        let dim = config::DIM as u32;
        let segments: Vec<Segment> = parts
            .iter()
            .map(|p| match *p {
                PrefixPart::Text { len, .. } => Segment::Text(len),
                PrefixPart::Image { h, w, .. } => Segment::Image { h, w },
            })
            .collect();
        let rows: u32 = segments.iter().map(|s| s.len() as u32).sum();
        // One even-pad row past the prefix: a text run's mask reads its keys
        // in pairs. Its K / V must be finite (masked, never attended), so the
        // row repeats the last prefix row.
        let padded = rows.next_multiple_of(2);

        // RoPE over [prefix ; target] (+ the pad row's copy of the last).
        let mut all = segments.clone();
        all.push(Segment::Image {
            h: target.0,
            w: target.1,
        });
        let mut freqs = self.rope.freqs(&all);
        let hd = config::HEAD_DIM;
        if padded > rows {
            let last = freqs[(rows as usize - 1) * hd..rows as usize * hd].to_vec();
            freqs.splice(rows as usize * hd..rows as usize * hd, last);
        }
        let freqs = upload(ws, backend, bp, &freqs)?;
        let zero = upload(ws, backend, bp, &vec![0.0; config::DIM])?;

        // --- embed: txt_in over the text rows, img_in over the condition tokens.
        let enc_rows = (encoder.len() / config::TEXT_DIM) as u32;
        let cond_rows = (cond.len() / config::LATENT_CH) as u32;
        let enc_buf = upload(ws, backend, bp, encoder)?;
        let cond_buf = (cond_rows > 0)
            .then(|| upload(ws, backend, bp, cond))
            .transpose()?;
        let mut hidden = ws.alloc(bp.act_bytes(padded * dim))?;
        {
            let t = &h.top;
            let mut views = vec![
                res.acquire(t.txt_norm, backend).await?,
                res.acquire(t.txt_in.handle, backend).await?,
                res.acquire(t.txt_out.handle, backend).await?,
            ];
            if cond_buf.is_some() {
                views.push(res.acquire(t.img_in.handle, backend).await?);
            }
            let scope = ws.batch();
            let td = config::TEXT_DIM as u32;
            let x = ActBuf::dense(scope.import_copy(enc_buf.as_buf_ref()));
            let normed = alloc_act(&scope, bp, enc_rows, td)?;
            let u = rmsnorm_uniform(&scope, enc_rows, td, config::EPS)?;
            scope.rmsnorm::<RmsNormGemmaF32>(
                &pipes.rmsnorm_zc,
                x.data,
                scope.import_copy(views[0].buf()),
                u,
                normed.data,
                enc_rows,
            )?;
            let h1 = linear(
                &scope,
                pipes,
                normed,
                (views[1].buf(), t.txt_in.kind),
                enc_rows,
                dim,
                td,
            )?;
            let g = act_op::<GeluF32>(&scope, bp, &bp.gelu, h1, enc_rows * dim)?;
            let txt = linear(
                &scope,
                pipes,
                g,
                (views[2].buf(), t.txt_out.kind),
                enc_rows,
                dim,
                dim,
            )?;
            let img = match &cond_buf {
                Some(c) => Some(linear(
                    &scope,
                    pipes,
                    ActBuf::dense(scope.import_copy(c.as_buf_ref())),
                    (views[3].buf(), t.img_in.kind),
                    cond_rows,
                    dim,
                    config::LATENT_CH as u32,
                )?),
                None => None,
            };
            let dst = scope.import_copy(hidden.as_buf_ref());
            let row = bp.act_bytes(dim);
            let mut at = 0u64;
            let mut last = None;
            for part in parts {
                let (src, from, n) = match *part {
                    PrefixPart::Text { start, len } => (txt.data, start as u64, len as u64),
                    PrefixPart::Image { start, h, w } => (
                        img.expect("condition tokens for an image part").data,
                        start as u64,
                        (h * w) as u64,
                    ),
                };
                scope.copy_buffer_to_buffer(src, from * row, dst, at * row, n * row)?;
                at += n;
                last = Some((src, from + n - 1));
            }
            // The pad row repeats the last prefix row (copied from its source:
            // a buffer cannot be both ends of one copy).
            if padded > rows
                && let Some((src, from)) = last
            {
                scope.copy_buffer_to_buffer(src, from * row, dst, at * row, row)?;
            }
            scope.submit_void().await?;
        }

        // --- t = 0 modulation for every prefix row ---
        let m0 = self.modulation(backend, pipes, res, ws, h, &[0.0]).await?;
        let modv = upload(ws, backend, bp, &m0[0].blocks)?;

        // Text-run masks, built once (shared by every block).
        let mut mask_bufs = Vec::with_capacity(segments.len());
        let mut start = 0usize;
        for seg in &segments {
            mask_bufs.push(match seg {
                Segment::Text(len) => {
                    let keys = (start + len).next_multiple_of(2);
                    let bytes = text_mask_bytes(start, *len, keys);
                    let b = ws.alloc(bytes.len() as u64)?;
                    backend.write_buffer(b.id(), 0, &bytes)?;
                    Some(b)
                }
                Segment::Image { .. } => None,
            });
            start += seg.len();
        }

        let mut cache = Vec::with_capacity(h.blocks.len());
        let mut pending = match h.blocks.first() {
            Some(b) => Some(BlockViews::acquire(b, res, backend).await?),
            None => None,
        };
        for (i, bh) in h.blocks.iter().enumerate() {
            let _s = trace::scope!("qwen21.prefill.block", i = i).entered();
            let views = pending.take().expect("pending block acquire");
            let bufs = views.bufs(bh);
            let (ck, cv) = (
                ws.alloc(bp.act_bytes(padded * dim))?,
                ws.alloc(bp.act_bytes(padded * dim))?,
            );
            let out = ws.alloc(bp.act_bytes(padded * dim))?;
            let scope = ws.batch();
            let masks: Vec<Option<BatchBuf<'_>>> = mask_bufs
                .iter()
                .map(|m| m.as_ref().map(|b| scope.import_copy(b.as_buf_ref())))
                .collect();
            // The pad row runs as a one-row tail so its K / V land in the cache.
            let mut segs = segments.clone();
            let mut masks = masks;
            if padded > rows {
                segs.push(Segment::Image { h: 1, w: 1 });
                masks.push(None);
            }
            block_forward(
                &scope,
                pipes,
                scope.import_copy(hidden.as_buf_ref()),
                scope.import_copy(out.as_buf_ref()),
                scope.import_copy(modv.as_buf_ref()),
                scope.import_copy(zero.as_buf_ref()),
                scope.import_copy(freqs.as_buf_ref()),
                padded,
                Attention::Prefix {
                    segments: &segs,
                    masks: &masks,
                    cache: (
                        scope.import_copy(ck.as_buf_ref()),
                        scope.import_copy(cv.as_buf_ref()),
                    ),
                },
                &bufs,
            )?;
            let next = async {
                match h.blocks.get(i + 1) {
                    Some(b) => Ok::<_, ResidencyError<S::Error, WgpuError>>(Some(
                        BlockViews::acquire(b, res, backend).await?,
                    )),
                    None => Ok(None),
                }
            };
            let submit = scope
                .submit_void()
                .instrument(tracing::debug_span!(target: PHASE, "qwen21.prefill.submit", i));
            let (s, n) = futures::join!(submit, next);
            s?;
            pending = n?;
            drop(views);
            cache.push((ck, cv));
            hidden = out;
        }
        Ok(Prefix {
            rows,
            cache,
            freqs,
            zero,
        })
    }

    /// One denoise step over the target's packed latent tokens `[T, LATENT_CH]`
    /// at timestep `t`. Returns the velocity `[T, LATENT_CH]`.
    #[allow(clippy::too_many_arguments)]
    pub async fn step<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipes: &Qwen21DitPipelines,
        res: &WeightResidency<S>,
        ws: &Workspace<WgpuBackend>,
        h: &Qwen21DitHandles,
        prefix: &Prefix,
        latents: &[f32],
        t: f32,
    ) -> Result<Vec<f32>, DitError<S::Error>> {
        let bp = &pipes.block;
        let dim = config::DIM as u32;
        let lc = config::LATENT_CH as u32;
        let rows = (latents.len() / config::LATENT_CH) as u32;
        let modrow = self
            .modulation(backend, pipes, res, ws, h, &[t])
            .await?
            .pop()
            .expect("one modulation row");
        let modv = upload(ws, backend, bp, &modrow.blocks)?;
        let out_scale = upload(ws, backend, bp, &modrow.out_scale)?;
        let lat = upload(ws, backend, bp, latents)?;
        // The target rows' RoPE slice: after the prefix (and its pad row).
        let freq_row = bp.act_bytes(config::HEAD_DIM as u32);
        let freq_off = prefix.rows.next_multiple_of(2) as u64 * freq_row;
        let freq_len = rows as u64 * freq_row;

        let mut hidden = ws.alloc(bp.act_bytes(rows * dim))?;
        {
            let v = res.acquire(h.top.img_in.handle, backend).await?;
            let scope = ws.batch();
            let x = ActBuf::dense(scope.import_copy(lat.as_buf_ref()));
            let e = linear(
                &scope,
                pipes,
                x,
                (v.buf(), h.top.img_in.kind),
                rows,
                dim,
                lc,
            )?;
            let dst = scope.import_copy(hidden.as_buf_ref());
            scope.copy_buffer_to_buffer(e.data, 0, dst, 0, bp.act_bytes(rows * dim))?;
            scope.submit_void().await?;
        }

        let mut pending = match h.blocks.first() {
            Some(b) => Some(BlockViews::acquire(b, res, backend).await?),
            None => None,
        };
        for (i, bh) in h.blocks.iter().enumerate() {
            let _s = trace::scope!("qwen21.block", i = i).entered();
            let views = pending.take().expect("pending block acquire");
            let bufs = views.bufs(bh);
            let out = ws.alloc(bp.act_bytes(rows * dim))?;
            let (ck, cv) = &prefix.cache[i];
            let scope = ws.batch();
            let freqs = scope.subview(
                &scope.import_copy(prefix.freqs.as_buf_ref()),
                freq_off,
                freq_len,
            );
            block_forward(
                &scope,
                pipes,
                scope.import_copy(hidden.as_buf_ref()),
                scope.import_copy(out.as_buf_ref()),
                scope.import_copy(modv.as_buf_ref()),
                scope.import_copy(prefix.zero.as_buf_ref()),
                freqs,
                rows,
                Attention::Target {
                    cache: (
                        scope.import_copy(ck.as_buf_ref()),
                        scope.import_copy(cv.as_buf_ref()),
                    ),
                    prefix_rows: prefix.rows,
                },
                &bufs,
            )?;
            let next = async {
                match h.blocks.get(i + 1) {
                    Some(b) => Ok::<_, ResidencyError<S::Error, WgpuError>>(Some(
                        BlockViews::acquire(b, res, backend).await?,
                    )),
                    None => Ok(None),
                }
            };
            let submit = scope
                .submit_void()
                .instrument(tracing::debug_span!(target: PHASE, "qwen21.submit", i));
            let (s, n) = futures::join!(submit, next);
            s?;
            pending = n?;
            drop(views);
            hidden = out;
        }

        // Final norm (scale only) + proj_out.
        let vel = ws.alloc(bp.act_bytes(rows * lc))?;
        {
            let v = res.acquire(h.top.proj_out.handle, backend).await?;
            let scope = ws.batch();
            let x = ActBuf::dense(scope.import_copy(hidden.as_buf_ref()));
            let n = layer_norm(&scope, bp, x, rows, dim)?;
            let m = modulate(
                &scope,
                bp,
                n,
                scope.import_copy(out_scale.as_buf_ref()),
                scope.import_copy(prefix.zero.as_buf_ref()),
                rows,
                dim,
            )?;
            let o = linear(
                &scope,
                pipes,
                m,
                (v.buf(), h.top.proj_out.kind),
                rows,
                lc,
                dim,
            )?;
            let dst = scope.import_copy(vel.as_buf_ref());
            scope.copy_buffer_to_buffer(o.data, 0, dst, 0, bp.act_bytes(rows * lc))?;
            scope.submit_void().await?;
        }
        Ok(download(backend, bp, &vel, rows * lc).await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_mask_is_causal_from_the_run_start() {
        // A 3-row text run starting at key 2, keys padded to 6.
        let m = text_mask_bytes(2, 3, 6);
        let at =
            |q: usize, k: usize| u16::from_le_bytes([m[(q * 6 + k) * 2], m[(q * 6 + k) * 2 + 1]]);
        // Row 0 (key 2) sees keys 0..=2.
        assert_eq!(
            (0..6).map(|k| at(0, k) == 0).collect::<Vec<_>>(),
            [true, true, true, false, false, false]
        );
        // Row 2 (key 4) sees 0..=4, not the pad.
        assert_eq!(at(2, 4), 0);
        assert_eq!(at(2, 5), 0xfc00);
    }
}
