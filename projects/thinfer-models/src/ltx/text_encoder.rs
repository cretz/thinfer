//! Gemma-3-12B text encoder (LTX-2.3 conditioning tower). Produces ALL 49 hidden
//! states (embedding layer + 48 decoder layers), each `[seq, 3840]`, which
//! FeatureExtractor V2 then stacks + projects (see `feature_extractor.rs`).
//!
//! Ground truth: HF `transformers/models/gemma3/modeling_gemma3.py` + the GGUF
//! `gemma3.*` config KV (snapshot in [`super::gemma`]). Reuses the z_image Qwen3
//! block's kernels (per-head QK-norm, embed gather, prefetch loop). Gemma deltas:
//!
//! - **4 norms/block**: the attn AND ffn sublayers are each wrapped pre-norm +
//!   post-norm (`h = resid + post_norm(sublayer(pre_norm(h)))`).
//! - **Gemma RMSNorm `x*(1+w)`**: handled at LOAD by [`super::loader::UnitOffset
//!   Source`] (bakes +1 into the norm weights), so the standard `op_rmsnorm` is
//!   reused unchanged here.
//! - **gated-GELU FFN** `down(gelu(gate(x)) * up(x))` (`GeluMulF32`).
//! - **NO qkv bias**; **embed scale** x sqrt(hidden); attn scale = query_pre_attn
//!   _scalar^-0.5 = 256^-0.5 == head_dim^-0.5.
//! - **dual rope**: full-attn layers (idx%6==5) use theta 1e6 + linear-x8 scaling
//!   (inv_freq/=8); sliding layers use theta 1e4 (no scaling). Half-rot.
//!
//! Real-tokens-only: LTX left-pads to 1024 with mask-cumsum positions, so encoding
//! just the n real tokens at positions 0..n-1 under a plain causal mask is
//! bit-identical (n << 1024 = sliding_window so sliding==full; only rope differs
//! per layer). The 1024 framing + registers happen in the connector stage.

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError, WgpuPipeline};
use thinfer_core::ops::{GeluMulF32, Op};
use thinfer_core::residency::{
    GpuView, ResidencyError, TransposePolicy, WeightHandle, WeightResidency,
};
use thinfer_core::trace::{self, PHASE};
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace};
use tracing::Instrument;

use std::collections::HashMap;

use thinfer_core::quant::QuantKind;
use thinfer_core::tensor::StorageEncoding;

use super::gemma;
use crate::common::block::{
    ActBuf, Block, BlockPipelines, BlockWgslConfigs, DequantStep, alloc_act, alloc_matmul_out_buf,
    op_add, op_rmsnorm, op_rope_halfrot, op_sdpa,
};
use crate::common::seq;
use crate::z_image::text_encoder::{LoadError, embed_lookup_hidden, register_one};

/// Number of hidden states FE V2 consumes: embedding layer + 48 decoder layers.
const N_STATES: usize = gemma::N_LAYERS + 1;

/// Layer type: every `SLIDING_PATTERN`-th layer (idx%6==5) is full attention
/// (global rope); the rest are sliding (local rope). Returns true for full.
pub fn is_full_attention(layer_idx: usize) -> bool {
    (layer_idx + 1).is_multiple_of(gemma::SLIDING_PATTERN)
}

/// Half-rot rope freqs `[seq, head_dim]` interleaved (cos, sin) per freq index,
/// matching `RopeEmbedder`'s layout but with Gemma's `inv_freq /= factor` linear
/// scaling. `inv_freq[j] = (1/theta^(2j/head_dim)) / factor`; `arg = pos * inv`.
/// Computed in f64 then narrowed (parity with HF's float32 rope path).
fn build_gemma_freqs(theta: f64, factor: f64, head_dim: usize, n: usize) -> Vec<f32> {
    let half = head_dim / 2;
    let inv: Vec<f64> = (0..half)
        .map(|j| (1.0 / theta.powf((2 * j) as f64 / head_dim as f64)) / factor)
        .collect();
    let mut out = vec![0.0_f32; n * head_dim];
    for p in 0..n {
        for (j, &f) in inv.iter().enumerate() {
            let arg = p as f64 * f;
            out[p * head_dim + 2 * j] = arg.cos() as f32;
            out[p * head_dim + 2 * j + 1] = arg.sin() as f32;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Weights / handles
// ---------------------------------------------------------------------------

/// Resolved HF weight names for one Gemma decoder layer (post-rename via
/// [`super::loader::gemma_gguf_renames`]).
#[derive(Clone, Debug)]
struct BlockWeights {
    input_layernorm: WeightId,
    post_attention_layernorm: WeightId,
    pre_feedforward_layernorm: WeightId,
    post_feedforward_layernorm: WeightId,
    q_proj: WeightId,
    k_proj: WeightId,
    v_proj: WeightId,
    o_proj: WeightId,
    q_norm: WeightId,
    k_norm: WeightId,
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
            pre_feedforward_layernorm: id("pre_feedforward_layernorm.weight"),
            post_feedforward_layernorm: id("post_feedforward_layernorm.weight"),
            q_proj: id("self_attn.q_proj.weight"),
            k_proj: id("self_attn.k_proj.weight"),
            v_proj: id("self_attn.v_proj.weight"),
            o_proj: id("self_attn.o_proj.weight"),
            q_norm: id("self_attn.q_norm.weight"),
            k_norm: id("self_attn.k_norm.weight"),
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
    pre_feedforward_layernorm: WeightHandle,
    post_feedforward_layernorm: WeightHandle,
    q_proj: WeightHandle,
    k_proj: WeightHandle,
    v_proj: WeightHandle,
    o_proj: WeightHandle,
    q_norm: WeightHandle,
    k_norm: WeightHandle,
    mlp_gate: WeightHandle,
    mlp_up: WeightHandle,
    mlp_down: WeightHandle,
    /// On-disk quant kind of each matmul weight, probed at register. The pure
    /// Q8_0 GGUF reports Q8_0 everywhere; the Q4_K_M variant ships them MIXED
    /// (Q4_K / Q6_K per tensor), so the dequant-once step is selected per site
    /// from [`GemmaEncoderPipelines::dense_dequant`]. `None` = a non-quant
    /// (bf16) weight read directly. Same pattern as the DiT's `BlockQuantKinds`.
    kinds: MatmulKinds,
}

/// Per-site dequant kind for one layer's seven matmul weights (q/k/v/o + gate/
/// up/down). `None` for a non-quant weight (the matmul reads it raw).
#[derive(Clone, Copy, Debug, Default)]
struct MatmulKinds {
    q: Option<QuantKind>,
    k: Option<QuantKind>,
    v: Option<QuantKind>,
    o: Option<QuantKind>,
    gate: Option<QuantKind>,
    up: Option<QuantKind>,
    down: Option<QuantKind>,
}

#[derive(Clone, Debug)]
pub struct EncoderHandles {
    layers: Vec<BlockHandles>,
    output_norm: WeightHandle,
}

/// Register all 48 Gemma layers + the final `model.norm`. Wrap the source in
/// [`super::loader::gemma_gguf_renames`] then [`super::loader::UnitOffsetSource`]
/// (the +1 norm bake) BEFORE calling. `transcode` requantizes the matmul weights
/// (the QAT GGUF already ships them Q4_K/Q6_K, so it is moot -> pass `None`).
pub fn register_handles<S: WeightSource>(
    residency: &WeightResidency<S>,
    transcode: Option<thinfer_core::quant::QuantKind>,
) -> Result<EncoderHandles, LoadError> {
    let lin = |id: &WeightId| register_one(residency, id, TransposePolicy::Linear2D, transcode);
    let dense = |id: &WeightId| register_one(residency, id, TransposePolicy::None, None);
    // Probe a matmul weight's resident quant kind. With `transcode = Some(k)` the
    // weight is requantized to `k` at load, so the resident kind is `k`; with
    // `None` it stays the on-disk kind (uniform Q8_0, or the Q4_K_M mix).
    let kind = |id: &WeightId, requant: Option<QuantKind>| -> Option<QuantKind> {
        requant.or_else(|| match residency.source().catalog().get(id).and_then(|e| e.encoding) {
            Some(StorageEncoding::Quant(k)) => Some(k),
            _ => None,
        })
    };
    let mut layers = Vec::with_capacity(gemma::N_LAYERS);
    for b in (0..gemma::N_LAYERS).map(BlockWeights::new) {
        layers.push(BlockHandles {
            input_layernorm: dense(&b.input_layernorm)?,
            post_attention_layernorm: dense(&b.post_attention_layernorm)?,
            pre_feedforward_layernorm: dense(&b.pre_feedforward_layernorm)?,
            post_feedforward_layernorm: dense(&b.post_feedforward_layernorm)?,
            q_proj: lin(&b.q_proj)?,
            k_proj: lin(&b.k_proj)?,
            v_proj: lin(&b.v_proj)?,
            o_proj: lin(&b.o_proj)?,
            q_norm: dense(&b.q_norm)?,
            k_norm: dense(&b.k_norm)?,
            mlp_gate: lin(&b.mlp_gate)?,
            mlp_up: lin(&b.mlp_up)?,
            // ffn_down never transcoded (massive-activation precision lesson).
            mlp_down: register_one(residency, &b.mlp_down, TransposePolicy::Linear2D, None)?,
            kinds: MatmulKinds {
                q: kind(&b.q_proj, transcode),
                k: kind(&b.k_proj, transcode),
                v: kind(&b.v_proj, transcode),
                o: kind(&b.o_proj, transcode),
                gate: kind(&b.mlp_gate, transcode),
                up: kind(&b.mlp_up, transcode),
                down: kind(&b.mlp_down, None),
            },
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
    pre_feedforward_layernorm: BufRef,
    post_feedforward_layernorm: BufRef,
    q_proj: BufRef,
    k_proj: BufRef,
    v_proj: BufRef,
    o_proj: BufRef,
    q_norm: BufRef,
    k_norm: BufRef,
    mlp_gate: BufRef,
    mlp_up: BufRef,
    mlp_down: BufRef,
}

struct BlockViews<'a> {
    input_layernorm: GpuView<'a>,
    post_attention_layernorm: GpuView<'a>,
    pre_feedforward_layernorm: GpuView<'a>,
    post_feedforward_layernorm: GpuView<'a>,
    q_proj: GpuView<'a>,
    k_proj: GpuView<'a>,
    v_proj: GpuView<'a>,
    o_proj: GpuView<'a>,
    q_norm: GpuView<'a>,
    k_norm: GpuView<'a>,
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
            pre_feedforward_layernorm: residency
                .acquire(self.pre_feedforward_layernorm, backend)
                .await?,
            post_feedforward_layernorm: residency
                .acquire(self.post_feedforward_layernorm, backend)
                .await?,
            q_proj: residency.acquire(self.q_proj, backend).await?,
            k_proj: residency.acquire(self.k_proj, backend).await?,
            v_proj: residency.acquire(self.v_proj, backend).await?,
            o_proj: residency.acquire(self.o_proj, backend).await?,
            q_norm: residency.acquire(self.q_norm, backend).await?,
            k_norm: residency.acquire(self.k_norm, backend).await?,
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
            pre_feedforward_layernorm: self.pre_feedforward_layernorm.buf(),
            post_feedforward_layernorm: self.post_feedforward_layernorm.buf(),
            q_proj: self.q_proj.buf(),
            k_proj: self.k_proj.buf(),
            v_proj: self.v_proj.buf(),
            o_proj: self.o_proj.buf(),
            q_norm: self.q_norm.buf(),
            k_norm: self.k_norm.buf(),
            mlp_gate: self.mlp_gate.buf(),
            mlp_up: self.mlp_up.buf(),
            mlp_down: self.mlp_down.buf(),
        }
    }
}

/// Canonical block config for the Gemma-3 encoder. **F32 acts** are mandatory,
/// not just the bf16-residual rule of the other encoders: Gemma's `head_dim=256`
/// exceeds the `SdpaF32` `MAX_D=128` tile, so attention needs `large_d_sdpa`,
/// which has no bf16-packed variant (F16/F32 only) -- and the residual grows past
/// f16's 65504 (~1.9e5), so F16 is out. F32 is the only act dtype that satisfies
/// both, and the encoder runs once per request so the cost is irrelevant. Every
/// matmul site is uniformly `weight_dtype` (the gemma GGUF is pure Q8_0); norms
/// stay dense via `matmul_adaln`/`ops`.
///
/// **All sites are `dense_acts` (no i8 DP4A act-quant).** Gemma's residual has
/// massive-activation outlier channels (max-abs ~1e5); per-32 i8 act-quant of the
/// proj/ffn_down A-sides crushes those outliers' block neighbours and corrupts the
/// residual stream. The encoder is once-per-request so the DP4A speedup is not
/// worth the quality hit -- keep every A-side dense (dequant-once bf16 matmul).
pub fn gemma_encoder_cfgs(weight_dtype: thinfer_core::ops::WeightDtype) -> BlockWgslConfigs {
    use thinfer_core::ops::{ActDtype, WgslConfig};
    let ops = WgslConfig {
        bf16_quant_writes: false,
        act_dtype: ActDtype::F32,
        weight_dtype: thinfer_core::ops::WeightDtype::Bf16,
    };
    let mm = WgslConfig {
        weight_dtype,
        ..ops
    };
    BlockWgslConfigs {
        matmul_qkv: mm,
        matmul_qkv_self: mm,
        matmul_proj: mm,
        matmul_ffn_up: mm,
        matmul_ffn_down: mm,
        matmul_adaln: ops,
        ops,
        i8_sdpa: false,
        dense_acts: crate::common::block::DenseActSites {
            qkv: true,
            proj: true,
            ffn_up: true,
            ffn_down: true,
        },
        large_d_sdpa: true,
    }
}

/// `BlockPipelines` + the standalone gated-GELU pipeline (the common block set
/// has SwiGLU but no gelu-mul; mirrors how `QwenImageDitPipelines` carries an
/// extra `gelu` next to the shared set).
pub struct GemmaEncoderPipelines {
    pub block: BlockPipelines,
    pub gelu_mul: WgpuPipeline,
    /// Dense (dequant-once) matmul dequant step per quant kind. The Q8_0 GGUF is
    /// uniform (every site Q8_0), but the Q4_K_M variant ships each matmul weight
    /// MIXED (Q4_K / Q6_K per tensor). The block matmul reads the bf16 dequant
    /// workspace regardless of source kind, so only this step varies by site --
    /// the same design as `ltx::dit`'s `dense_dequant`. F32 acts -> bf16 target.
    dense_dequant: HashMap<QuantKind, DequantStep>,
}

impl GemmaEncoderPipelines {
    pub async fn compile(
        backend: &WgpuBackend,
        cfgs: &BlockWgslConfigs,
    ) -> Result<Self, WgpuError> {
        let block = BlockPipelines::compile(backend, cfgs).await?;
        let gelu_mul = backend
            .create_pipeline(
                "gemma_gelu_mul",
                <GeluMulF32 as Op>::wgsl(&cfgs.ops),
                "main",
                <GeluMulF32 as Op>::layout(),
            )
            .await?;
        // The encoder runs F32 acts -> bf16 dequant workspace (see
        // `BlockPipelines::compile`'s `dequant_target`). Build a step per kind the
        // Gemma GGUFs use: pure Q8_0, or Q4_K_M's Q4_K + Q6_K mix.
        use thinfer_core::ops::dequant::{DequantTarget, build_wgsl, layout};
        let dq_layout = layout();
        let mut dense_dequant = HashMap::new();
        for scheme in [QuantKind::Q8_0, QuantKind::Q4_K, QuantKind::Q5_K, QuantKind::Q6_K] {
            let wgsl = build_wgsl(scheme, DequantTarget::Bf16);
            let pipeline = backend
                .create_pipeline(
                    &format!("gemma_dequant_{}", scheme.hint()),
                    &wgsl,
                    "main",
                    dq_layout,
                )
                .await?;
            dense_dequant.insert(scheme, DequantStep { pipeline, scheme });
        }
        Ok(Self {
            block,
            gelu_mul,
            dense_dequant,
        })
    }

    /// Dense dequant step for a site's resident quant kind, or `None` for a
    /// non-quant (bf16) weight (read raw by the matmul).
    fn dequant_for(&self, kind: Option<QuantKind>) -> Option<&DequantStep> {
        kind.and_then(|k| self.dense_dequant.get(&k))
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
        // query_pre_attn_scalar^-0.5; == head_dim^-0.5 since scalar == head_dim.
        1.0 / (gemma::QUERY_PRE_ATTN_SCALAR).sqrt()
    }
}

/// One Gemma-3 decoder layer: pre-norm GQA self-attn (head-dim QK-norm, half-rot
/// rope, causal SDPA) wrapped post-norm -> residual -> pre-norm gated-GELU FFN
/// wrapped post-norm -> residual. All norms are Gemma `(1+w)` (baked at load).
struct DecoderBlock {
    shape: BlockShape,
}

impl DecoderBlock {
    #[allow(clippy::too_many_arguments)]
    fn forward<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &GemmaEncoderPipelines,
        x_in: BatchBuf<'wsp>,
        freqs_in: BatchBuf<'wsp>,
        mask_in: BatchBuf<'wsp>,
        y_out: BatchBuf<'wsp>,
        bufs: &'wsp BlockBufs,
        kinds: MatmulKinds,
    ) -> Result<(), WgpuError> {
        let p = &pipelines.block;
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
        let pa_ln = scope.import(&bufs.post_attention_layernorm);
        let pre_ff_ln = scope.import(&bufs.pre_feedforward_layernorm);
        let post_ff_ln = scope.import(&bufs.post_feedforward_layernorm);
        let q_w = scope.import(&bufs.q_proj);
        let k_w = scope.import(&bufs.k_proj);
        let v_w = scope.import(&bufs.v_proj);
        let qn_w = scope.import(&bufs.q_norm);
        let kn_w = scope.import(&bufs.k_norm);
        let o_w = scope.import(&bufs.o_proj);
        let g_w = scope.import(&bufs.mlp_gate);
        let up_w = scope.import(&bufs.mlp_up);
        let down_w = scope.import(&bufs.mlp_down);

        // matmul helper (no bias for Gemma). The per-site dequant step is chosen
        // by the weight's resident quant kind (uniform Q8_0, or the Q4_K_M mix).
        let mm = |src: ActBuf<'wsp>,
                  w,
                  out_w: u32,
                  in_w: u32,
                  dq: Option<&DequantStep>|
         -> Result<BatchBuf<'wsp>, WgpuError> {
            let scratch = alloc_matmul_out_buf(scope, p, rows * out_w)?;
            let dims = scope.u32x4_uniform(rows, out_w, in_w, 0)?;
            Block::dispatch_matmul_site(
                scope,
                p,
                src,
                w,
                scratch,
                dims,
                p.matmul_i8_qkv.as_ref(),
                p.dequant_i8_qkv.as_ref(),
                dq,
                &p.matmul_qkv,
                &p.matmuls.qkv,
                rows,
                out_w,
                in_w,
            )?;
            Ok(scratch)
        };

        // --- attn: pre-norm, qkv, qk-norm, rope, sdpa, o_proj, post-norm, resid ---
        let n1 = alloc_act(scope, p, rows, dim)?;
        op_rmsnorm(scope, p, x_in, in_ln, n1, rows, dim, eps)?;

        let q = mm(n1, q_w, hq * hd, dim, pipelines.dequant_for(kinds.q))?;
        let k = mm(n1, k_w, hkv * hd, dim, pipelines.dequant_for(kinds.k))?;
        let v = mm(n1, v_w, hkv * hd, dim, pipelines.dequant_for(kinds.v))?;

        // per-head Q/K RMSNorm over head_dim (Gemma (1+w) baked).
        let qn = alloc_act(scope, p, rows * hq, hd)?;
        op_rmsnorm(scope, p, ActBuf::dense(q), qn_w, qn, rows * hq, hd, eps)?;
        let kn = alloc_act(scope, p, rows * hkv, hd)?;
        op_rmsnorm(scope, p, ActBuf::dense(k), kn_w, kn, rows * hkv, hd, eps)?;

        let qr = alloc_act(scope, p, rows, hq * hd)?;
        op_rope_halfrot(scope, p, qn, freqs_in, qr, rows, hq, hd)?;
        let kr = alloc_act(scope, p, rows, hkv * hd)?;
        op_rope_halfrot(scope, p, kn, freqs_in, kr, rows, hkv, hd)?;

        let sa = alloc_act(scope, p, rows, hq * hd)?;
        op_sdpa(
            scope,
            p,
            qr,
            kr,
            ActBuf::dense(v),
            mask_in,
            sa,
            1,
            rows,
            rows,
            hq,
            hkv,
            hd,
            scale,
            1,
        )?;

        let proj = mm(sa, o_w, dim, hq * hd, pipelines.dequant_for(kinds.o))?;
        let proj_n = alloc_act(scope, p, rows, dim)?;
        op_rmsnorm(scope, p, ActBuf::dense(proj), pa_ln, proj_n, rows, dim, eps)?;
        let after_attn = alloc_act(scope, p, rows, dim)?;
        op_add(scope, p, x_in, ActBuf::dense(proj_n.data), after_attn)?;

        // --- ffn: pre-norm, gated-gelu, down, post-norm, resid ---
        let n2 = alloc_act(scope, p, rows, dim)?;
        op_rmsnorm(scope, p, after_attn, pre_ff_ln, n2, rows, dim, eps)?;
        let g = mm(n2, g_w, hid, dim, pipelines.dequant_for(kinds.gate))?;
        let up = mm(n2, up_w, hid, dim, pipelines.dequant_for(kinds.up))?;
        let gu = alloc_act(scope, p, rows, hid)?;
        scope.dispatch_op::<GeluMulF32>(&pipelines.gelu_mul, &[g, up], gu.data)?;
        let down = mm(gu, down_w, dim, hid, pipelines.dequant_for(kinds.down))?;
        let down_n = alloc_act(scope, p, rows, dim)?;
        op_rmsnorm(
            scope,
            p,
            ActBuf::dense(down),
            post_ff_ln,
            down_n,
            rows,
            dim,
            eps,
        )?;
        op_add(scope, p, after_attn, ActBuf::dense(down_n.data), y_out)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Encoder driver
// ---------------------------------------------------------------------------

/// Gemma-3 text encoder. Returns ALL 49 hidden states for FE V2.
pub struct GemmaEncoder;

impl GemmaEncoder {
    /// Run the embedding + 48 layers; capture every hidden state (embedding
    /// output, then each layer output) as `[seq, HIDDEN]` f32 row-major. Real
    /// tokens only (positions 0..n-1, causal mask). Returns `N_STATES` vecs.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipelines: &GemmaEncoderPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        handles: &EncoderHandles,
        source: &S,
        token_ids: &[u32],
    ) -> Result<EncoderOutput, ForwardError<S::Error>> {
        let seq = token_ids.len();
        assert!(seq > 0, "GemmaEncoder::forward: empty token list");
        // Even-pad for the f16/bf16 mask word layout; the pad row repeats the
        // last token, is causally invisible, and is sliced off after readback.
        let mut ids = token_ids.to_vec();
        if !ids.len().is_multiple_of(2) {
            ids.push(*ids.last().expect("non-empty"));
        }
        let seq_pad = ids.len();
        let bp = &pipelines.block;

        // embed lookup (row-gather) then x sqrt(hidden) (Gemma embed scale).
        let mut embeds = embed_lookup_hidden(source, &ids, gemma::HIDDEN)
            .instrument(tracing::debug_span!(target: PHASE, "gemma.embed_lookup", seq))
            .await
            .map_err(ForwardError::Embed)?;
        let escale = (gemma::HIDDEN as f32).sqrt();
        for v in embeds.iter_mut() {
            *v *= escale;
        }

        let shape = BlockShape {
            dim: gemma::HIDDEN,
            n_heads: gemma::N_HEADS,
            n_kv_heads: gemma::N_KV_HEADS,
            head_dim: gemma::HEAD_DIM,
            ffn_hidden: gemma::FFN,
            seq: seq_pad,
            norm_eps: gemma::EPS,
        };
        let block = DecoderBlock { shape };
        let act_bytes = bp.act_bytes((seq_pad * gemma::HIDDEN) as u32);

        let x_buf = scratch.alloc(act_bytes)?;
        let embed_bytes = seq::act_upload_bytes(bp.act_dtype, &embeds);
        backend.write_buffer(x_buf.id, 0, &embed_bytes)?;

        // dual rope freq buffers (global = full layers, local = sliding layers).
        let global_freqs = build_gemma_freqs(
            gemma::GLOBAL_THETA,
            gemma::ROPE_LINEAR_FACTOR,
            gemma::HEAD_DIM,
            seq_pad,
        );
        let local_freqs = build_gemma_freqs(gemma::LOCAL_THETA, 1.0, gemma::HEAD_DIM, seq_pad);
        let global_bytes = seq::freqs_upload_bytes(bp.act_dtype, &global_freqs);
        let local_bytes = seq::freqs_upload_bytes(bp.act_dtype, &local_freqs);
        let global_buf = scratch.alloc(global_bytes.len() as u64)?;
        let local_buf = scratch.alloc(local_bytes.len() as u64)?;
        backend.write_buffer(global_buf.id, 0, &global_bytes)?;
        backend.write_buffer(local_buf.id, 0, &local_bytes)?;

        // causal mask [1, seq_pad, seq_pad] (n << sliding_window so sliding==full).
        let mask_bytes = seq::causal_mask_bytes_act(seq_pad, bp.act_dtype);
        let mask_buf = scratch.alloc(mask_bytes.len() as u64)?;
        backend.write_buffer(mask_buf.id, 0, &mask_bytes)?;

        // hidden states: [embedding] + per-layer outputs.
        let mut states: Vec<Vec<f32>> = Vec::with_capacity(N_STATES);
        states.push({
            let mut e = embeds.clone();
            e.truncate(seq * gemma::HIDDEN);
            e
        });

        let mut cur = x_buf;
        let mut pending: Option<BlockViews<'_>> = if handles.layers.is_empty() {
            None
        } else {
            Some(handles.layers[0].acquire(residency, backend).await?)
        };
        let global_ref = global_buf.as_buf_ref();
        let local_ref = local_buf.as_buf_ref();
        let mask_ref = mask_buf.as_buf_ref();
        for idx in 0..handles.layers.len() {
            let _guard = trace::scope!(format!("gemma.layer.{idx}")).entered();
            let views = pending.take().expect("pending acquire missing");
            let bufs = views.bufs();
            let cur_ref = cur.as_buf_ref();
            let nxt = scratch.alloc(act_bytes)?;
            let nxt_ref = nxt.as_buf_ref();
            let scope = scratch.batch();
            let cur_h = scope.import(&cur_ref);
            let freqs_ref = if is_full_attention(idx) {
                &global_ref
            } else {
                &local_ref
            };
            let freqs_h = scope.import(freqs_ref);
            let mask_h = scope.import(&mask_ref);
            let nxt_h = scope.import(&nxt_ref);
            let kinds = handles.layers[idx].kinds;
            block.forward(&scope, pipelines, cur_h, freqs_h, mask_h, nxt_h, &bufs, kinds)?;

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
                .instrument(tracing::debug_span!(target: PHASE, "gemma.submit", idx));
            let (submit_res, next_res) = futures::join!(submit_fut, next_acquire);
            submit_res?;
            pending = next_res?;

            let bytes = backend.read_buffer(nxt.id(), 0, act_bytes).await?;
            let mut h = seq::act_readback_to_f32(bp.act_dtype, &bytes, seq_pad * gemma::HIDDEN);
            h.truncate(seq * gemma::HIDDEN);
            states.push(h);

            drop(views);
            cur = nxt;
        }

        // final model.norm applied to the LAST layer output (HF last_hidden_state).
        // FE V2 consumes the 49 pre-final-norm states; the normed last is exposed
        // for diagnostics only.
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
                bp,
                cur_h,
                on_h,
                normed_h,
                seq_pad as u32,
                gemma::HIDDEN as u32,
                gemma::EPS,
            )?;
            scope.submit_void().await?;
        }
        let bytes = backend.read_buffer(normed.id(), 0, act_bytes).await?;
        let mut last_normed =
            seq::act_readback_to_f32(bp.act_dtype, &bytes, seq_pad * gemma::HIDDEN);
        last_normed.truncate(seq * gemma::HIDDEN);

        Ok(EncoderOutput {
            states,
            last_normed,
            seq,
        })
    }
}

#[derive(Clone, Debug)]
pub struct EncoderOutput {
    /// `N_STATES` (49) hidden states, each `[seq, HIDDEN]` row-major f32:
    /// `states[0]` = scaled embeddings, `states[i]` = layer i-1 output.
    pub states: Vec<Vec<f32>>,
    /// `model.norm(last layer)` `[seq, HIDDEN]` (diagnostics).
    pub last_normed: Vec<f32>,
    pub seq: usize,
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
    fn layer_types_pattern_6() {
        // idx%6==5 is full attention; all others sliding. 8 full layers in 48.
        let full: Vec<usize> = (0..gemma::N_LAYERS)
            .filter(|&i| is_full_attention(i))
            .collect();
        assert_eq!(full, vec![5, 11, 17, 23, 29, 35, 41, 47]);
    }

    #[test]
    fn freqs_layout_and_linear_scaling() {
        // coord 0 -> all (cos,sin) = (1,0).
        let f = build_gemma_freqs(10_000.0, 1.0, 8, 3);
        for j in 0..4 {
            assert!((f[2 * j] - 1.0).abs() < 1e-6);
            assert!(f[2 * j + 1].abs() < 1e-6);
        }
        // linear x8: global inv_freq is 1/8 of the unscaled -> pos*inv smaller.
        // j=0: inv = (1/theta^0)/factor = 1/8; at pos=1 -> arg=0.125.
        let g = build_gemma_freqs(1_000_000.0, 8.0, 8, 2);
        let arg = 1.0_f64 * (1.0 / 8.0);
        assert!((g[8] - (arg.cos() as f32)).abs() < 1e-6); // pos=1, j=0, cos
        assert!((g[9] - (arg.sin() as f32)).abs() < 1e-6); // pos=1, j=0, sin
    }
}
