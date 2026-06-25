//! Qwen2.5-VL-7B text encoder (text-only LM path). Qwen-Image conditions the
//! DiT on `text_encoder(...).hidden_states[-1]` -- the last decoder layer output
//! AFTER the final RMSNorm (`output_norm`), `[seq, 3584]`. The vision tower
//! (`<|image_pad|>` slots) is the edit-path channel and lives elsewhere; this
//! module is the pure-text decoder stack.
//!
//! Ground truth: HF `transformers/models/qwen2_5_vl` (the LM is Qwen2-style).
//! Deltas vs the Qwen3 encoder ([`crate::z_image::text_encoder`], reused for the
//! kernels + embed gather + `register_one`): Qwen2.5 has **QKV bias** and **no
//! q/k-norm** (Qwen3 is the opposite). GQA 28 Q / 4 KV heads, head_dim 128,
//! SwiGLU FFN, 1-axis RoPE (text MRoPE collapses to the token position),
//! theta 1e6. Runs bf16 acts (residual-overflow guard, same lesson as umT5).
//!
//! The encoder GGUF (`qwen2vl` arch) ships native `blk.{i}.attn_*` /
//! `token_embd` / `output_norm` names; [`qwen2vl_gguf_renames`] re-keys them to
//! the HF `model.*` names this loader + the shared embed gather expect.

use std::collections::HashMap;

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError, WgpuPipeline};
use thinfer_core::ops::{BcastAddF32, RopeF32Mrope, RopeOp};
use thinfer_core::residency::{
    GpuView, ResidencyError, TransposePolicy, WeightHandle, WeightResidency,
};
use thinfer_core::trace::{self, PHASE};
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace};
use tracing::Instrument;

use crate::common::block::{
    ActBuf, Block, BlockPipelines, BlockWgslConfigs, alloc_act, alloc_matmul_out_buf, op_add,
    op_rmsnorm, op_rope_halfrot, op_sdpa, op_silu_mul,
};
use crate::common::embedders::bcast_add_uniform;
use crate::common::rope_embedder::RopeEmbedder;
use crate::common::seq;
use crate::z_image::text_encoder::{LoadError, embed_lookup_hidden, register_one};

/// Audited against the encoder GGUF metadata (`qwen2vl` arch) + HF
/// `Qwen2_5_VLConfig` text config.
pub mod config {
    pub const HIDDEN: usize = 3584;
    pub const N_LAYERS: usize = 28;
    pub const N_HEADS: usize = 28;
    pub const N_KV_HEADS: usize = 4;
    pub const HEAD_DIM: usize = 128;
    pub const FFN_HIDDEN: usize = 18944;
    pub const VOCAB: usize = 152064;
    pub const RMS_NORM_EPS: f32 = 1e-6;
    pub const ROPE_THETA: f32 = 1_000_000.0;
    /// Qwen-Image takes `hidden_states[-1]` = last layer AFTER `output_norm`, so
    /// every layer runs and the final norm is applied (unlike Z-Image's `[-2]`).
    pub const N_RUN_LAYERS: usize = N_LAYERS;
}

/// GGUF (`qwen2vl` native) -> HF (`model.*`) tensor-name map. Re-keys the
/// encoder source so [`register_one`] + [`embed_lookup_hidden`] (HF names) work
/// unchanged. Built programmatically over the 28 layers.
pub fn qwen2vl_gguf_renames() -> HashMap<WeightId, WeightId> {
    let mut m = HashMap::new();
    let mut put = |g: String, h: String| {
        m.insert(WeightId(g), WeightId(h));
    };
    put(
        "token_embd.weight".into(),
        "model.embed_tokens.weight".into(),
    );
    put("output_norm.weight".into(), "model.norm.weight".into());
    for i in 0..config::N_LAYERS {
        let g = |s: &str| format!("blk.{i}.{s}");
        let h = |s: &str| format!("model.layers.{i}.{s}");
        put(g("attn_norm.weight"), h("input_layernorm.weight"));
        put(g("ffn_norm.weight"), h("post_attention_layernorm.weight"));
        put(g("attn_q.weight"), h("self_attn.q_proj.weight"));
        put(g("attn_q.bias"), h("self_attn.q_proj.bias"));
        put(g("attn_k.weight"), h("self_attn.k_proj.weight"));
        put(g("attn_k.bias"), h("self_attn.k_proj.bias"));
        put(g("attn_v.weight"), h("self_attn.v_proj.weight"));
        put(g("attn_v.bias"), h("self_attn.v_proj.bias"));
        put(g("attn_output.weight"), h("self_attn.o_proj.weight"));
        put(g("ffn_gate.weight"), h("mlp.gate_proj.weight"));
        put(g("ffn_up.weight"), h("mlp.up_proj.weight"));
        put(g("ffn_down.weight"), h("mlp.down_proj.weight"));
    }
    m
}

// ---------------------------------------------------------------------------
// Weights / handles
// ---------------------------------------------------------------------------

/// Resolved HF weight names for one Qwen2.5 decoder layer (post-rename).
#[derive(Clone, Debug)]
struct BlockWeights {
    input_layernorm: WeightId,
    post_attention_layernorm: WeightId,
    q_proj: WeightId,
    q_bias: WeightId,
    k_proj: WeightId,
    k_bias: WeightId,
    v_proj: WeightId,
    v_bias: WeightId,
    o_proj: WeightId,
    mlp_gate: WeightId,
    mlp_up: WeightId,
    mlp_down: WeightId,
}

impl BlockWeights {
    fn new(idx: usize) -> Self {
        let p = format!("model.layers.{idx}");
        let id = |s: &str| WeightId(format!("{p}.{s}"));
        Self {
            input_layernorm: id("input_layernorm.weight"),
            post_attention_layernorm: id("post_attention_layernorm.weight"),
            q_proj: id("self_attn.q_proj.weight"),
            q_bias: id("self_attn.q_proj.bias"),
            k_proj: id("self_attn.k_proj.weight"),
            k_bias: id("self_attn.k_proj.bias"),
            v_proj: id("self_attn.v_proj.weight"),
            v_bias: id("self_attn.v_proj.bias"),
            o_proj: id("self_attn.o_proj.weight"),
            mlp_gate: id("mlp.gate_proj.weight"),
            mlp_up: id("mlp.up_proj.weight"),
            mlp_down: id("mlp.down_proj.weight"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BlockHandles {
    input_layernorm: WeightHandle,
    post_attention_layernorm: WeightHandle,
    q_proj: WeightHandle,
    q_bias: WeightHandle,
    k_proj: WeightHandle,
    k_bias: WeightHandle,
    v_proj: WeightHandle,
    v_bias: WeightHandle,
    o_proj: WeightHandle,
    mlp_gate: WeightHandle,
    mlp_up: WeightHandle,
    mlp_down: WeightHandle,
}

#[derive(Clone, Debug)]
pub struct EncoderHandles {
    layers: Vec<BlockHandles>,
    output_norm: WeightHandle,
}

/// Register the encoder weights (HF names; wrap the source in
/// [`qwen2vl_gguf_renames`] first). `transcode` requantizes the bf16 matmul
/// weights to Q8_0 at load; for the in-file-Q8_0 GGUF it is moot (pass `None`).
/// `mlp_down` is never transcoded (massive-activation precision, same lesson as
/// the Qwen3 path). Norms + biases are dense passthrough.
pub fn register_handles<S: WeightSource>(
    residency: &WeightResidency<S>,
    transcode: Option<thinfer_core::quant::QuantKind>,
) -> Result<EncoderHandles, LoadError> {
    let lin = |id: &WeightId| register_one(residency, id, TransposePolicy::Linear2D, transcode);
    let dense = |id: &WeightId| register_one(residency, id, TransposePolicy::None, None);
    let mut layers = Vec::with_capacity(config::N_RUN_LAYERS);
    for b in (0..config::N_RUN_LAYERS).map(BlockWeights::new) {
        layers.push(BlockHandles {
            input_layernorm: dense(&b.input_layernorm)?,
            post_attention_layernorm: dense(&b.post_attention_layernorm)?,
            q_proj: lin(&b.q_proj)?,
            q_bias: dense(&b.q_bias)?,
            k_proj: lin(&b.k_proj)?,
            k_bias: dense(&b.k_bias)?,
            v_proj: lin(&b.v_proj)?,
            v_bias: dense(&b.v_bias)?,
            o_proj: lin(&b.o_proj)?,
            mlp_gate: lin(&b.mlp_gate)?,
            mlp_up: lin(&b.mlp_up)?,
            // Never transcoded (massive activations corrupt per-block quant).
            mlp_down: register_one(residency, &b.mlp_down, TransposePolicy::Linear2D, None)?,
        });
    }
    let output_norm = dense(&WeightId("model.norm.weight".into()))?;
    Ok(EncoderHandles {
        layers,
        output_norm,
    })
}

// ---------------------------------------------------------------------------
// Per-layer forward
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct BlockBufs {
    input_layernorm: BufRef,
    post_attention_layernorm: BufRef,
    q_proj: BufRef,
    q_bias: BufRef,
    k_proj: BufRef,
    k_bias: BufRef,
    v_proj: BufRef,
    v_bias: BufRef,
    o_proj: BufRef,
    mlp_gate: BufRef,
    mlp_up: BufRef,
    mlp_down: BufRef,
}

struct BlockViews<'a> {
    input_layernorm: GpuView<'a>,
    post_attention_layernorm: GpuView<'a>,
    q_proj: GpuView<'a>,
    q_bias: GpuView<'a>,
    k_proj: GpuView<'a>,
    k_bias: GpuView<'a>,
    v_proj: GpuView<'a>,
    v_bias: GpuView<'a>,
    o_proj: GpuView<'a>,
    mlp_gate: GpuView<'a>,
    mlp_up: GpuView<'a>,
    mlp_down: GpuView<'a>,
}

impl BlockHandles {
    async fn acquire<'a, S: WeightSource>(
        &self,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<BlockViews<'a>, ResidencyError<S::Error, WgpuError>> {
        Ok(BlockViews {
            input_layernorm: residency.acquire(self.input_layernorm, backend).await?,
            post_attention_layernorm: residency
                .acquire(self.post_attention_layernorm, backend)
                .await?,
            q_proj: residency.acquire(self.q_proj, backend).await?,
            q_bias: residency.acquire(self.q_bias, backend).await?,
            k_proj: residency.acquire(self.k_proj, backend).await?,
            k_bias: residency.acquire(self.k_bias, backend).await?,
            v_proj: residency.acquire(self.v_proj, backend).await?,
            v_bias: residency.acquire(self.v_bias, backend).await?,
            o_proj: residency.acquire(self.o_proj, backend).await?,
            mlp_gate: residency.acquire(self.mlp_gate, backend).await?,
            mlp_up: residency.acquire(self.mlp_up, backend).await?,
            mlp_down: residency.acquire(self.mlp_down, backend).await?,
        })
    }
}

impl BlockViews<'_> {
    fn bufs(&self) -> BlockBufs {
        BlockBufs {
            input_layernorm: self.input_layernorm.buf(),
            post_attention_layernorm: self.post_attention_layernorm.buf(),
            q_proj: self.q_proj.buf(),
            q_bias: self.q_bias.buf(),
            k_proj: self.k_proj.buf(),
            k_bias: self.k_bias.buf(),
            v_proj: self.v_proj.buf(),
            v_bias: self.v_bias.buf(),
            o_proj: self.o_proj.buf(),
            mlp_gate: self.mlp_gate.buf(),
            mlp_up: self.mlp_up.buf(),
            mlp_down: self.mlp_down.buf(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BlockShape {
    dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ffn_hidden: usize,
    seq: usize,
    norm_eps: f32,
}

impl BlockShape {
    fn sdpa_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

/// 3-axis MRoPE on `src [rows, heads, head_dim]` via the standalone `mrope`
/// pipeline. `freqs` carries 4 values per pair (`[cos_lo, sin_lo, cos_hi,
/// sin_hi]`), so its row stride is `head_dim * 2`; the kernel derives that
/// internally from `pairs`. Mirrors `op_rope_halfrot` otherwise.
#[allow(clippy::too_many_arguments)]
fn op_rope_mrope<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    mrope: &WgpuPipeline,
    src: ActBuf<'wsp>,
    freqs: BatchBuf<'wsp>,
    dst: ActBuf<'wsp>,
    rows: u32,
    heads: u32,
    head_dim: u32,
) -> Result<(), WgpuError> {
    let pairs = head_dim / 2;
    let u = scope.u32x4_uniform(rows, heads, pairs, 0)?;
    scope.rope::<RopeF32Mrope>(mrope, src.data, freqs, u, dst.data, rows, heads, pairs)
}

/// One Qwen2.5 decoder layer: pre-norm GQA self-attn (QKV bias, NO q/k-norm,
/// 1-axis RoPE, causal SDPA) -> residual -> pre-norm SwiGLU FFN -> residual.
struct DecoderBlock {
    shape: BlockShape,
}

impl DecoderBlock {
    /// Bias-add after a projection matmul: `out = matmul_out + bias` broadcast
    /// over rows. Mirrors `embedders::linear_bias`'s `bcast_add` tail.
    fn add_bias<'wsp>(
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        pre: BatchBuf<'wsp>,
        bias: BufRef,
        out: BatchBuf<'wsp>,
        rows: u32,
        out_dim: u32,
    ) -> Result<(), WgpuError> {
        let bias_h = scope.import_copy(bias);
        let ba_u = bcast_add_uniform(scope, out_dim)?;
        scope.bcast_add::<BcastAddF32>(&pipelines.bcast_add, pre, bias_h, ba_u, out, rows * out_dim)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        x_in: BatchBuf<'wsp>,
        freqs_in: BatchBuf<'wsp>,
        mask_in: BatchBuf<'wsp>,
        y_out: BatchBuf<'wsp>,
        bufs: &'wsp BlockBufs,
        // `Some` => 3-axis MRoPE (edit path); `None` => 1-axis half-rot (t2i).
        mrope: Option<&WgpuPipeline>,
    ) -> Result<(), WgpuError> {
        let s = self.shape;
        let rows = s.seq as u32;
        let dim = s.dim as u32;
        let hd = s.head_dim as u32;
        let hq = s.n_heads as u32;
        let hkv = s.n_kv_heads as u32;
        let hid = s.ffn_hidden as u32;
        let eps = s.norm_eps;
        let scale = s.sdpa_scale();
        let x_in = ActBuf::dense(x_in);
        let y_out = ActBuf::dense(y_out);

        let in_ln = scope.import(&bufs.input_layernorm);
        let q_w = scope.import(&bufs.q_proj);
        let k_w = scope.import(&bufs.k_proj);
        let v_w = scope.import(&bufs.v_proj);
        let o_w = scope.import(&bufs.o_proj);
        let pa_ln = scope.import(&bufs.post_attention_layernorm);
        let g_w = scope.import(&bufs.mlp_gate);
        let up_w = scope.import(&bufs.mlp_up);
        let down_w = scope.import(&bufs.mlp_down);

        // --- pre-attn norm ---
        let n1 = alloc_act(scope, pipelines, rows, dim)?;
        op_rmsnorm(scope, pipelines, x_in, in_ln, n1, rows, dim, eps)?;

        // --- q/k/v projections (separate GQA weights) + bias ---
        let qkv = |w, out_w: u32| -> Result<BatchBuf<'wsp>, WgpuError> {
            let scratch = alloc_matmul_out_buf(scope, pipelines, rows * out_w)?;
            let dims = scope.u32x4_uniform(rows, out_w, dim, 0)?;
            Block::dispatch_matmul_site(
                scope,
                pipelines,
                n1,
                w,
                scratch,
                dims,
                pipelines.matmul_i8_qkv.as_ref(),
                pipelines.dequant_i8_qkv.as_ref(),
                pipelines.dequant_qkv.as_ref(),
                &pipelines.matmul_qkv,
                &pipelines.matmuls.qkv,
                rows,
                out_w,
                dim,
            )?;
            Ok(scratch)
        };
        let q_pre = qkv(q_w, hq * hd)?;
        let q = alloc_act(scope, pipelines, rows, hq * hd)?;
        Self::add_bias(scope, pipelines, q_pre, bufs.q_bias, q.data, rows, hq * hd)?;
        let k_pre = qkv(k_w, hkv * hd)?;
        let k = alloc_act(scope, pipelines, rows, hkv * hd)?;
        Self::add_bias(scope, pipelines, k_pre, bufs.k_bias, k.data, rows, hkv * hd)?;
        let v_pre = qkv(v_w, hkv * hd)?;
        let v = alloc_act(scope, pipelines, rows, hkv * hd)?;
        Self::add_bias(scope, pipelines, v_pre, bufs.v_bias, v.data, rows, hkv * hd)?;

        // --- rope on Q, K (NO q/k-norm, unlike Qwen3) ---
        let qr = alloc_act(scope, pipelines, rows, hq * hd)?;
        let kr = alloc_act(scope, pipelines, rows, hkv * hd)?;
        match mrope {
            Some(mp) => {
                op_rope_mrope(scope, mp, q, freqs_in, qr, rows, hq, hd)?;
                op_rope_mrope(scope, mp, k, freqs_in, kr, rows, hkv, hd)?;
            }
            None => {
                op_rope_halfrot(scope, pipelines, q, freqs_in, qr, rows, hq, hd)?;
                op_rope_halfrot(scope, pipelines, k, freqs_in, kr, rows, hkv, hd)?;
            }
        }

        // --- causal sdpa ---
        let sa = alloc_act(scope, pipelines, rows, hq * hd)?;
        op_sdpa(
            scope, pipelines, qr, kr, v, mask_in, sa, 1, rows, rows, hq, hkv, hd, scale, 1,
        )?;

        // --- o_proj (no bias) + residual ---
        let proj = alloc_matmul_out_buf(scope, pipelines, rows * dim)?;
        let dims_proj = scope.u32x4_uniform(rows, dim, hq * hd, 0)?;
        Block::dispatch_matmul_site(
            scope,
            pipelines,
            sa,
            o_w,
            proj,
            dims_proj,
            pipelines.matmul_i8_proj.as_ref(),
            pipelines.dequant_i8_proj.as_ref(),
            pipelines.dequant_proj.as_ref(),
            &pipelines.matmul_proj,
            &pipelines.matmuls.proj,
            rows,
            dim,
            hq * hd,
        )?;
        let after_attn = alloc_act(scope, pipelines, rows, dim)?;
        op_add(scope, pipelines, x_in, ActBuf::dense(proj), after_attn)?;

        // --- pre-ffn norm ---
        let n2 = alloc_act(scope, pipelines, rows, dim)?;
        op_rmsnorm(scope, pipelines, after_attn, pa_ln, n2, rows, dim, eps)?;

        // --- SwiGLU FFN: down(silu_mul(gate(x), up(x))) ---
        let ffn_up = |w| -> Result<BatchBuf<'wsp>, WgpuError> {
            let scratch = alloc_matmul_out_buf(scope, pipelines, rows * hid)?;
            let dims = scope.u32x4_uniform(rows, hid, dim, 0)?;
            Block::dispatch_matmul_site(
                scope,
                pipelines,
                n2,
                w,
                scratch,
                dims,
                pipelines.matmul_i8_ffn_up.as_ref(),
                pipelines.dequant_i8_ffn_up.as_ref(),
                pipelines.dequant_ffn_up.as_ref(),
                &pipelines.matmul_ffn_up,
                &pipelines.matmuls.ffn_up,
                rows,
                hid,
                dim,
            )?;
            Ok(scratch)
        };
        let g = ffn_up(g_w)?;
        let up = ffn_up(up_w)?;
        let gu = alloc_act(scope, pipelines, rows, hid)?;
        op_silu_mul(scope, pipelines, ActBuf::dense(g), ActBuf::dense(up), gu)?;
        let down = alloc_matmul_out_buf(scope, pipelines, rows * dim)?;
        let dims_down = scope.u32x4_uniform(rows, dim, hid, 0)?;
        Block::dispatch_matmul_site(
            scope,
            pipelines,
            gu,
            down_w,
            down,
            dims_down,
            pipelines.matmul_i8_ffn_down.as_ref(),
            pipelines.dequant_i8_ffn_down.as_ref(),
            pipelines.dequant_ffn_down.as_ref(),
            &pipelines.matmul_ffn_down,
            &pipelines.matmuls.ffn_down,
            rows,
            dim,
            hid,
        )?;
        op_add(scope, pipelines, after_attn, ActBuf::dense(down), y_out)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Edit-path MRoPE
// ---------------------------------------------------------------------------

/// `BlockPipelines` + the standalone 3-axis MRoPE rope pipeline. Used only by
/// [`TextEncoder::forward_edit`]; the t2i `forward` never touches `mrope`.
/// Mirrors how `QwenImageDitPipelines` carries an extra `gelu` next to the
/// shared block set.
pub struct EditEncoderPipelines {
    pub block: BlockPipelines,
    pub mrope: WgpuPipeline,
}

impl EditEncoderPipelines {
    pub async fn compile(
        backend: &WgpuBackend,
        cfgs: &BlockWgslConfigs,
    ) -> Result<Self, WgpuError> {
        let block = BlockPipelines::compile(backend, cfgs).await?;
        // MRoPE is an act-dtype kernel (no weights); compile from the block's
        // `ops` cfg so its act_dtype matches the rest of the encoder.
        let mrope = backend
            .create_pipeline(
                "rope_mrope",
                <RopeF32Mrope as RopeOp>::wgsl(&cfgs.ops),
                "main",
                <RopeF32Mrope as RopeOp>::layout(),
            )
            .await?;
        Ok(Self { block, mrope })
    }
}

/// One global Qwen2.5-VL inv_freq: `invf[j] = theta^(-2j/head_dim)`,
/// `j = 0..head_dim/2`. theta = 1e6 (`config::ROPE_THETA`).
fn mrope_inv_freq() -> Vec<f64> {
    let theta = config::ROPE_THETA as f64;
    let hd = config::HEAD_DIM as f64;
    (0..config::HEAD_DIM / 2)
        .map(|j| theta.powf(-(2.0 * j as f64) / hd))
        .collect()
}

/// Per-pair MRoPE axis. Verified against HF `apply_multimodal_rotary_pos_emb`:
/// `mrope_section = [16, 24, 24]` is the LIST, doubled by `* 2` to
/// `[16, 24, 24, 16, 24, 24]` and split over the 128 `cat(freqs, freqs)` dims,
/// picking axis `i % 3` per chunk. Boundaries land at 16, 40, 64, 80, 104, 128.
/// With rotate_half pairing `(k, k+64)` the imag dim `k+64` falls in the MIRROR
/// chunk of the SAME axis as the real dim `k`, so a whole pair `k` uses ONE
/// axis: `k<16 -> 0`, `16<=k<40 -> 1`, `40<=k<64 -> 2`.
fn mrope_pair_axis(k: usize) -> usize {
    if k < 16 {
        0
    } else if k < 40 {
        1
    } else {
        2
    }
}

/// Build the per-row, per-pair MRoPE freqs in the `[cos_lo, sin_lo, cos_hi,
/// sin_hi]` x pairs layout the `rope.f32.mrope` kernel reads (row stride =
/// `pairs * 4 = head_dim * 2`).
///
/// `position_ids[seq * 3]` row-major `(t, h, w)`. Each pair `k` rotates by
/// `pos[axis(k)] * invf[k]` (see [`mrope_pair_axis`]); both halves of the pair
/// (real x[k], imag x[k+pairs]) use that one axis, so `cos_lo == cos_hi` here.
/// The 4-per-pair freq layout is kept general (the kernel reads lo/hi
/// independently). On all-equal positions (text-only) every axis collapses to
/// the token position, reducing EXACTLY to `op_rope_halfrot`.
fn build_mrope_freqs(position_ids: &[i32], invf: &[f64]) -> Vec<f32> {
    let pairs = config::HEAD_DIM / 2; // 64
    assert!(position_ids.len().is_multiple_of(3));
    let seq = position_ids.len() / 3;
    let mut out = vec![0.0_f32; seq * pairs * 4];
    for row in 0..seq {
        let pos = &position_ids[row * 3..row * 3 + 3];
        for (k, &f) in invf.iter().enumerate() {
            let arg = pos[mrope_pair_axis(k)] as f64 * f;
            let (c, s) = (arg.cos() as f32, arg.sin() as f32);
            let o = (row * pairs + k) * 4;
            out[o] = c;
            out[o + 1] = s;
            out[o + 2] = c;
            out[o + 3] = s;
        }
    }
    out
}

/// Build the 3-axis MRoPE `position_ids[seq * 3]` `(t, h, w)` for the single-
/// image edit prompt (per the edit SPEC + HF `get_rope_index`). Text before the
/// image block runs `0..image_pad_start` on all 3 axes; the `n_img = mgh*mgw`
/// image tokens (row-major h-then-w over the merged grid) get `t = cur`,
/// `h = cur + row`, `w = cur + col`; then `cur += max(mgh, mgw)` and trailing
/// text resumes sequential on all axes.
fn build_edit_position_ids(
    seq: usize,
    image_pad_start: usize,
    merged_grid: (usize, usize),
) -> Vec<i32> {
    let (mgh, mgw) = merged_grid;
    let n_img = mgh * mgw;
    let mut pos = vec![0_i32; seq * 3];
    let set = |pos: &mut [i32], i: usize, t: i32, h: i32, w: i32| {
        pos[i * 3] = t;
        pos[i * 3 + 1] = h;
        pos[i * 3 + 2] = w;
    };
    // Leading text: 0..image_pad_start on all axes.
    for i in 0..image_pad_start {
        let p = i as i32;
        set(&mut pos, i, p, p, p);
    }
    let cur = image_pad_start as i32;
    // Image block: t const, h/w over the merged grid.
    for r in 0..mgh {
        for c in 0..mgw {
            let i = image_pad_start + r * mgw + c;
            set(&mut pos, i, cur, cur + r as i32, cur + c as i32);
        }
    }
    // Trailing text: sequential from `cur + max(mgh, mgw)` on all axes.
    let base = cur + mgh.max(mgw) as i32;
    for (step, i) in ((image_pad_start + n_img)..seq).enumerate() {
        let p = base + step as i32;
        set(&mut pos, i, p, p, p);
    }
    pos
}

// ---------------------------------------------------------------------------
// Encoder driver
// ---------------------------------------------------------------------------

/// Qwen2.5-VL text encoder: CPU embed lookup + 28 decoder layers + final
/// `output_norm`, returning `hidden_states[-1]` `[seq, HIDDEN]`.
pub struct TextEncoder {
    rope: RopeEmbedder,
}

impl TextEncoder {
    /// `max_seq` sizes the rope table (largest padded prompt length).
    pub fn new(max_seq: usize) -> Self {
        let seq_len = max_seq.max(1);
        Self {
            rope: RopeEmbedder::new(
                config::ROPE_THETA,
                [config::HEAD_DIM, 0, 0],
                [seq_len, 1, 1],
            ),
        }
    }

    pub fn rope(&self) -> &RopeEmbedder {
        &self.rope
    }

    /// Run all 28 layers + final norm; return `[token_ids.len(), HIDDEN]` f32.
    /// `want_layer_outputs` captures each post-layer residual (parity bisection).
    pub async fn forward<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipelines: &BlockPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        handles: &EncoderHandles,
        source: &S,
        token_ids: &[u32],
        want_layer_outputs: bool,
    ) -> Result<EncoderOutput, ForwardError<S::Error>> {
        let seq = token_ids.len();
        assert!(seq > 0, "TextEncoder::forward: empty token list");
        // Even-pad (F16/bf16 mask layouts want even s_k); pad row repeats the
        // last token, is causally invisible, sliced off after readback.
        let mut ids = token_ids.to_vec();
        if !ids.len().is_multiple_of(2) {
            ids.push(*ids.last().expect("non-empty"));
        }
        let seq_pad = ids.len();
        assert!(
            seq_pad <= self.rope.axes_lens[0],
            "prompt length {seq_pad} exceeds rope max_seq {}",
            self.rope.axes_lens[0]
        );

        let embeds = embed_lookup_hidden(source, &ids, config::HIDDEN)
            .instrument(tracing::debug_span!(target: PHASE, "qwen25.embed_lookup", seq))
            .await
            .map_err(ForwardError::Embed)?;

        let block = DecoderBlock {
            shape: BlockShape {
                dim: config::HIDDEN,
                n_heads: config::N_HEADS,
                n_kv_heads: config::N_KV_HEADS,
                head_dim: config::HEAD_DIM,
                ffn_hidden: config::FFN_HIDDEN,
                seq: seq_pad,
                norm_eps: config::RMS_NORM_EPS,
            },
        };
        let act_bytes = pipelines.act_bytes((seq_pad * config::HIDDEN) as u32);

        let x_buf = scratch.alloc(act_bytes)?;
        let embed_bytes = seq::act_upload_bytes(pipelines.act_dtype, &embeds);
        backend.write_buffer(x_buf.id, 0, &embed_bytes)?;

        // --- rope freqs: token positions 0..seq_pad on axis 0 ---
        let mut pos_ids = vec![0_i32; seq_pad * 3];
        for (i, p) in pos_ids.chunks_exact_mut(3).enumerate() {
            p[0] = i as i32;
        }
        let freqs_bytes = seq::freqs_upload_bytes(pipelines.act_dtype, &self.rope.lookup(&pos_ids));
        let freqs_buf = scratch.alloc(freqs_bytes.len() as u64)?;
        backend.write_buffer(freqs_buf.id, 0, &freqs_bytes)?;

        // --- causal mask [1, seq_pad, seq_pad] ---
        let mask_bytes = seq::causal_mask_bytes_act(seq_pad, pipelines.act_dtype);
        let mask_buf = scratch.alloc(mask_bytes.len() as u64)?;
        backend.write_buffer(mask_buf.id, 0, &mask_bytes)?;

        // --- run all layers, ping-pong acts, prefetch next layer's weights ---
        let mut layer_outputs = Vec::new();
        let mut cur = x_buf;
        let mut pending: Option<BlockViews<'_>> = if handles.layers.is_empty() {
            None
        } else {
            Some(handles.layers[0].acquire(residency, backend).await?)
        };
        let freqs_ref = freqs_buf.as_buf_ref();
        let mask_ref = mask_buf.as_buf_ref();
        for idx in 0..handles.layers.len() {
            let _guard = trace::scope!(format!("qwen25.layer.{idx}")).entered();
            let views = pending.take().expect("pending acquire missing");
            let bufs = views.bufs();
            let cur_ref = cur.as_buf_ref();
            let nxt = scratch.alloc(act_bytes)?;
            let nxt_ref = nxt.as_buf_ref();
            let scope = scratch.batch();
            let cur_h = scope.import(&cur_ref);
            let freqs_h = scope.import(&freqs_ref);
            let mask_h = scope.import(&mask_ref);
            let nxt_h = scope.import(&nxt_ref);
            block.forward(
                &scope, pipelines, cur_h, freqs_h, mask_h, nxt_h, &bufs, None,
            )?;

            let next_idx = idx + 1;
            let next_acquire = async {
                match handles.layers.get(next_idx) {
                    Some(h) => Ok::<_, ResidencyError<S::Error, WgpuError>>(Some(
                        h.acquire(residency, backend).await?,
                    )),
                    None => Ok(None),
                }
            };
            let submit_fut = scope
                .submit_void()
                .instrument(tracing::debug_span!(target: PHASE, "qwen25.submit", idx));
            let (submit_res, next_res) = futures::join!(submit_fut, next_acquire);
            submit_res?;
            pending = next_res?;

            if want_layer_outputs {
                let bytes = backend.read_buffer(nxt.id(), 0, act_bytes).await?;
                layer_outputs.push(seq::act_readback_to_f32(
                    pipelines.act_dtype,
                    &bytes,
                    seq_pad * config::HIDDEN,
                ));
            }

            drop(views);
            cur = nxt;
        }

        // --- final output_norm (model.norm), then readback ---
        let normed = scratch.alloc(act_bytes)?;
        {
            let on_view = residency.acquire(handles.output_norm, backend).await?;
            let on_buf = on_view.buf();
            let cur_ref = cur.as_buf_ref();
            let normed_ref = normed.as_buf_ref();
            let scope = scratch.batch();
            let cur_h = ActBuf::dense(scope.import(&cur_ref));
            let on_h = scope.import(&on_buf);
            let normed_h = ActBuf::dense(scope.import(&normed_ref));
            op_rmsnorm(
                &scope,
                pipelines,
                cur_h,
                on_h,
                normed_h,
                seq_pad as u32,
                config::HIDDEN as u32,
                config::RMS_NORM_EPS,
            )?;
            scope.submit_void().await?;
        }

        let bytes = backend.read_buffer(normed.id(), 0, act_bytes).await?;
        let mut hidden =
            seq::act_readback_to_f32(pipelines.act_dtype, &bytes, seq_pad * config::HIDDEN);
        hidden.truncate(seq * config::HIDDEN);
        Ok(EncoderOutput {
            hidden,
            seq,
            layer_outputs,
        })
    }

    /// Edit-path forward. Identical to [`Self::forward`] (28 layers, output_norm,
    /// even-pad, prefetch) except: (a) the contiguous image-pad block
    /// `[image_pad_start, image_pad_start + n_img)` of `inputs_embeds` is
    /// overwritten with the caller's `vision_embeds` `[n_img, HIDDEN]` (raster
    /// merged order, from the vision tower); (b) 3-axis MRoPE position_ids per
    /// the edit SPEC; (c) the new `rope.f32.mrope` op instead of half-rot.
    /// `n_img = merged_grid.0 * merged_grid.1` must equal the `<|image_pad|>`
    /// span width.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward_edit<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipelines: &EditEncoderPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        handles: &EncoderHandles,
        source: &S,
        token_ids: &[u32],
        image_pad_start: usize,
        vision_embeds: &[f32],
        merged_grid: (usize, usize),
        want_layer_outputs: bool,
    ) -> Result<EncoderOutput, ForwardError<S::Error>> {
        let block_pipelines = &pipelines.block;
        let seq = token_ids.len();
        assert!(seq > 0, "TextEncoder::forward_edit: empty token list");
        let n_img = merged_grid.0 * merged_grid.1;
        assert_eq!(
            vision_embeds.len(),
            n_img * config::HIDDEN,
            "vision_embeds must be [n_img={n_img}, HIDDEN]"
        );
        assert!(
            image_pad_start + n_img <= seq,
            "image-pad block [{image_pad_start}, {}) exceeds seq {seq}",
            image_pad_start + n_img
        );

        // Even-pad (mask layout wants even s_k); pad row repeats the last token,
        // is causally invisible, sliced off after readback.
        let mut ids = token_ids.to_vec();
        if !ids.len().is_multiple_of(2) {
            ids.push(*ids.last().expect("non-empty"));
        }
        let seq_pad = ids.len();
        assert!(
            seq_pad <= self.rope.axes_lens[0],
            "prompt length {seq_pad} exceeds rope max_seq {}",
            self.rope.axes_lens[0]
        );

        // inputs_embeds = embed table lookup, then scatter the vision rows into
        // the image-pad block (overwrite, raster order).
        let mut embeds = embed_lookup_hidden(source, &ids, config::HIDDEN)
            .instrument(tracing::debug_span!(target: PHASE, "qwen25.embed_lookup", seq))
            .await
            .map_err(ForwardError::Embed)?;
        let dst = image_pad_start * config::HIDDEN;
        embeds[dst..dst + n_img * config::HIDDEN].copy_from_slice(vision_embeds);

        let block = DecoderBlock {
            shape: BlockShape {
                dim: config::HIDDEN,
                n_heads: config::N_HEADS,
                n_kv_heads: config::N_KV_HEADS,
                head_dim: config::HEAD_DIM,
                ffn_hidden: config::FFN_HIDDEN,
                seq: seq_pad,
                norm_eps: config::RMS_NORM_EPS,
            },
        };
        let act_bytes = block_pipelines.act_bytes((seq_pad * config::HIDDEN) as u32);

        let x_buf = scratch.alloc(act_bytes)?;
        let embed_bytes = seq::act_upload_bytes(block_pipelines.act_dtype, &embeds);
        backend.write_buffer(x_buf.id, 0, &embed_bytes)?;

        // --- 3-axis MRoPE freqs (the [cos_lo, sin_lo, cos_hi, sin_hi] layout) ---
        // The pad row reuses the last real position (causally invisible).
        let mut pos_ids = build_edit_position_ids(seq, image_pad_start, merged_grid);
        if seq_pad > seq {
            let last = pos_ids[(seq - 1) * 3..seq * 3].to_vec();
            pos_ids.extend_from_slice(&last);
        }
        let invf = mrope_inv_freq();
        let freqs = build_mrope_freqs(&pos_ids, &invf);
        let freqs_bytes = seq::freqs_upload_bytes(block_pipelines.act_dtype, &freqs);
        let freqs_buf = scratch.alloc(freqs_bytes.len() as u64)?;
        backend.write_buffer(freqs_buf.id, 0, &freqs_bytes)?;

        // --- causal mask [1, seq_pad, seq_pad] ---
        let mask_bytes = seq::causal_mask_bytes_act(seq_pad, block_pipelines.act_dtype);
        let mask_buf = scratch.alloc(mask_bytes.len() as u64)?;
        backend.write_buffer(mask_buf.id, 0, &mask_bytes)?;

        // --- run all layers, ping-pong acts, prefetch next layer's weights ---
        let mut layer_outputs = Vec::new();
        let mut cur = x_buf;
        let mut pending: Option<BlockViews<'_>> = if handles.layers.is_empty() {
            None
        } else {
            Some(handles.layers[0].acquire(residency, backend).await?)
        };
        let freqs_ref = freqs_buf.as_buf_ref();
        let mask_ref = mask_buf.as_buf_ref();
        for idx in 0..handles.layers.len() {
            let _guard = trace::scope!(format!("qwen25.edit.layer.{idx}")).entered();
            let views = pending.take().expect("pending acquire missing");
            let bufs = views.bufs();
            let cur_ref = cur.as_buf_ref();
            let nxt = scratch.alloc(act_bytes)?;
            let nxt_ref = nxt.as_buf_ref();
            let scope = scratch.batch();
            let cur_h = scope.import(&cur_ref);
            let freqs_h = scope.import(&freqs_ref);
            let mask_h = scope.import(&mask_ref);
            let nxt_h = scope.import(&nxt_ref);
            block.forward(
                &scope,
                block_pipelines,
                cur_h,
                freqs_h,
                mask_h,
                nxt_h,
                &bufs,
                Some(&pipelines.mrope),
            )?;

            let next_idx = idx + 1;
            let next_acquire = async {
                match handles.layers.get(next_idx) {
                    Some(h) => Ok::<_, ResidencyError<S::Error, WgpuError>>(Some(
                        h.acquire(residency, backend).await?,
                    )),
                    None => Ok(None),
                }
            };
            let submit_fut = scope
                .submit_void()
                .instrument(tracing::debug_span!(target: PHASE, "qwen25.edit.submit", idx));
            let (submit_res, next_res) = futures::join!(submit_fut, next_acquire);
            submit_res?;
            pending = next_res?;

            if want_layer_outputs {
                let bytes = backend.read_buffer(nxt.id(), 0, act_bytes).await?;
                layer_outputs.push(seq::act_readback_to_f32(
                    block_pipelines.act_dtype,
                    &bytes,
                    seq_pad * config::HIDDEN,
                ));
            }

            drop(views);
            cur = nxt;
        }

        // --- final output_norm (model.norm), then readback ---
        let normed = scratch.alloc(act_bytes)?;
        {
            let on_view = residency.acquire(handles.output_norm, backend).await?;
            let on_buf = on_view.buf();
            let cur_ref = cur.as_buf_ref();
            let normed_ref = normed.as_buf_ref();
            let scope = scratch.batch();
            let cur_h = ActBuf::dense(scope.import(&cur_ref));
            let on_h = scope.import(&on_buf);
            let normed_h = ActBuf::dense(scope.import(&normed_ref));
            op_rmsnorm(
                &scope,
                block_pipelines,
                cur_h,
                on_h,
                normed_h,
                seq_pad as u32,
                config::HIDDEN as u32,
                config::RMS_NORM_EPS,
            )?;
            scope.submit_void().await?;
        }

        let bytes = backend.read_buffer(normed.id(), 0, act_bytes).await?;
        let mut hidden =
            seq::act_readback_to_f32(block_pipelines.act_dtype, &bytes, seq_pad * config::HIDDEN);
        hidden.truncate(seq * config::HIDDEN);
        Ok(EncoderOutput {
            hidden,
            seq,
            layer_outputs,
        })
    }
}

#[derive(Clone, Debug)]
pub struct EncoderOutput {
    /// `[seq, HIDDEN]` row-major f32 = `hidden_states[-1]` (post final norm).
    pub hidden: Vec<f32>,
    pub seq: usize,
    /// Per-layer post-residual `[seq_pad, HIDDEN]`, only if requested.
    pub layer_outputs: Vec<Vec<f32>>,
}

#[derive(Debug)]
pub enum ForwardError<SE: core::fmt::Debug> {
    Embed(crate::z_image::text_encoder::EmbedLookupError),
    Wgpu(WgpuError),
    Residency(ResidencyError<SE, WgpuError>),
}

impl<SE: core::fmt::Debug> From<WgpuError> for ForwardError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}

impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for ForwardError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rename_map_covers_layer0_and_top_level() {
        let m = qwen2vl_gguf_renames();
        assert_eq!(
            m.get(&WeightId("token_embd.weight".into())).unwrap().0,
            "model.embed_tokens.weight"
        );
        assert_eq!(
            m.get(&WeightId("output_norm.weight".into())).unwrap().0,
            "model.norm.weight"
        );
        assert_eq!(
            m.get(&WeightId("blk.0.attn_q.bias".into())).unwrap().0,
            "model.layers.0.self_attn.q_proj.bias"
        );
        assert_eq!(
            m.get(&WeightId("blk.27.ffn_down.weight".into())).unwrap().0,
            "model.layers.27.mlp.down_proj.weight"
        );
        // 2 top-level + 28 layers * 12 per-layer tensors.
        assert_eq!(m.len(), 2 + config::N_LAYERS * 12);
    }

    /// On all-equal positions (text-only) MRoPE must reduce EXACTLY to half-rot:
    /// `pos_lo == pos_hi` for every pair, so each pair is `[c, s, c, s]` where
    /// `(c, s)` is the half-rot freq for that position (RopeEmbedder, same theta
    /// + inv_freq). This is the sanity invariant from the spec.
    #[test]
    fn mrope_equals_halfrot_on_equal_positions() {
        let invf = mrope_inv_freq();
        let pairs = config::HEAD_DIM / 2;
        assert_eq!(invf.len(), pairs);

        // A text-only token at position p: all 3 axes equal.
        for p in [0_i32, 1, 7, 123] {
            let mrope = build_mrope_freqs(&[p, p, p], &invf);
            assert_eq!(mrope.len(), pairs * 4);

            // Half-rot reference via the same embedder the t2i path uses.
            let rope = RopeEmbedder::new(config::ROPE_THETA, [config::HEAD_DIM, 0, 0], [256, 1, 1]);
            let half = rope.lookup(&[p, 0, 0]); // [cos_0, sin_0, cos_1, sin_1, ...]
            assert_eq!(half.len(), config::HEAD_DIM); // pairs * 2

            for k in 0..pairs {
                let (c, s) = (half[2 * k], half[2 * k + 1]);
                let o = k * 4;
                let tol = 1e-5_f32;
                assert!((mrope[o] - c).abs() < tol, "cos_lo pair {k}");
                assert!((mrope[o + 1] - s).abs() < tol, "sin_lo pair {k}");
                assert!((mrope[o + 2] - c).abs() < tol, "cos_hi pair {k}");
                assert!((mrope[o + 3] - s).abs() < tol, "sin_hi pair {k}");
            }
        }
    }

    /// Image tokens get distinct per-axis positions; pad-handling and the merged
    /// grid layout must match the SPEC (row-major h-then-w, t const, cur bump).
    #[test]
    fn edit_position_ids_layout() {
        // 4 leading text, 2x3 merged image grid (6 tokens), 2 trailing text.
        let (mgh, mgw) = (2, 3);
        let pad_start = 4;
        let seq = pad_start + mgh * mgw + 2;
        let pos = build_edit_position_ids(seq, pad_start, (mgh, mgw));
        // Leading text: i on all axes.
        for i in 0..pad_start {
            assert_eq!(&pos[i * 3..i * 3 + 3], &[i as i32; 3]);
        }
        // Image token (r=1, c=2) -> seq index pad_start + 1*mgw + 2.
        let cur = pad_start as i32;
        let (r, c) = (1, 2);
        let i = pad_start + r * mgw + c;
        assert_eq!(
            &pos[i * 3..i * 3 + 3],
            &[cur, cur + r as i32, cur + c as i32]
        );
        // Trailing text resumes at cur + max(mgh, mgw) = 4 + 3 = 7.
        let t0 = pad_start + mgh * mgw;
        assert_eq!(&pos[t0 * 3..t0 * 3 + 3], &[7, 7, 7]);
        assert_eq!(&pos[(t0 + 1) * 3..(t0 + 1) * 3 + 3], &[8, 8, 8]);
    }
}
