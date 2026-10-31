//! MiniMax-H3's Qwen3-VL-32B conditioner.
//!
//! H3 does NOT use Qwen3-VL as a language model. It reads the **unnormalized**
//! hidden state after the 50th of its 64 decoder layers (`hidden_states[50]`,
//! where `hidden_states[0]` is the embedding output) and feeds that straight
//! into the DiT's `condition_proj`. The final norm, `lm_head` and layers 50..63
//! are never evaluated, so the manifest ships a checkpoint already truncated to
//! the 50 layers we run (asserted by [`check_depth`]).
//!
//! Only the LAST state is returned. Unlike the LTX Gemma tower (which hands all
//! 49 states to a feature extractor), H3 conditions on one state, so we never
//! materialize the per-layer stack: at 50 layers x [seq, 5120] f32 that would be
//! pure waste.
//!
//! **RoPE.** Qwen3-VL uses interleaved M-RoPE with `mrope_section [24, 20, 20]`.
//! For a TEXT-ONLY request `get_rope_index` gives every axis the same sequential
//! position, and M-RoPE with three equal position components is exactly 1-axis
//! rope. So the t2va path drives the plain half-rot kernel and needs no mrope
//! machinery. A keyframe (vision) request does NOT have this property and must
//! grow the mrope path before it can work.
//!
//! **Prompt presentation** (from the reference): no chat template and no special
//! tokens anywhere; the prompt is tokenized verbatim with
//! `add_special_tokens = false` and carries NO leading BOS. This is the same
//! shape as the LTX finding; do not "helpfully" add a template here.

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError};
use thinfer_core::ops::ActDtype;
use thinfer_core::quant::QuantKind;
use thinfer_core::residency::{ResidencyError, TransposePolicy, WeightHandle, WeightResidency};
use thinfer_core::tensor::StorageEncoding;
use thinfer_core::trace::{self, PHASE};
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace};
use tracing::Instrument;

use std::collections::HashMap;

use crate::common::block::{
    ActBuf, Block, BlockPipelines, BlockWgslConfigs, DequantStep, alloc_act, alloc_matmul_out_buf,
    op_add, op_rmsnorm, op_rope_halfrot, op_sdpa_f16, op_silu_mul,
};
use crate::common::seq;
use crate::z_image::text_encoder::{
    LoadError, Qwen3BlockWeights, embed_lookup_hidden, register_one,
};

/// Qwen3-VL-32B, from the upstream `config.json`. Verified against the shipped
/// GGUF tensor shapes (2026-08-07): `q_proj [5120, 8192]` = 64 heads x 128,
/// `k/v_proj [5120, 1024]` = 8 kv heads x 128, `mlp [5120, 25600]`.
pub mod config {
    /// Decoder layers H3 actually evaluates, of the full stack's 64.
    pub const N_LAYERS: usize = 50;
    pub const FULL_LAYERS: usize = 64;
    pub const HIDDEN: usize = 5120;
    pub const N_HEADS: usize = 64;
    pub const N_KV_HEADS: usize = 8;
    pub const HEAD_DIM: usize = 128;
    pub const FFN_HIDDEN: usize = 25600;
    pub const VOCAB: usize = 151_936;
    pub const EPS: f32 = 1e-6;
    pub const ROPE_THETA: f32 = 5_000_000.0;
    /// Interleaved M-RoPE frequency slots per `(t, h, w)` axis.
    pub const MROPE_SECTION: [usize; 3] = [24, 20, 20];
}

/// One image's `<|image_pad|>` block in the token sequence: rows
/// `start..start + vision.tokens` take its features.
pub struct ImageRows<'a> {
    pub start: usize,
    pub vision: &'a crate::minimax::vision::VisionOutput,
}

/// HF `get_rope_index` `(t, h, w)` per row for a sequence with image blocks
/// (`images` in sequence order). Text advances one position per token on all
/// three axes; an image holds `t` at the current position, steps `h` / `w`
/// over its merged grid from there, and advances the clock by the larger
/// merged side (not by its token count).
pub fn mrope_positions(seq: usize, images: &[ImageRows<'_>]) -> Vec<i64> {
    fn text_until(end: usize, row: &mut usize, clock: &mut i64, pos: &mut Vec<i64>) {
        while *row < end {
            pos.extend([*clock; 3]);
            *clock += 1;
            *row += 1;
        }
    }
    let mut pos = Vec::with_capacity(seq * 3);
    let (mut row, mut clock) = (0usize, 0i64);
    for img in images {
        text_until(img.start, &mut row, &mut clock, &mut pos);
        let (gh, gw) = img.vision.merged_grid;
        for r in 0..gh {
            for c in 0..gw {
                pos.extend([clock, clock + r as i64, clock + c as i64]);
            }
        }
        row += gh * gw;
        clock += gh.max(gw) as i64;
    }
    text_until(seq, &mut row, &mut clock, &mut pos);
    pos
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MatmulKinds {
    pub q: Option<QuantKind>,
    pub k: Option<QuantKind>,
    pub v: Option<QuantKind>,
    pub o: Option<QuantKind>,
    pub gate: Option<QuantKind>,
    pub up: Option<QuantKind>,
    pub down: Option<QuantKind>,
}

#[derive(Clone, Copy, Debug)]
pub struct BlockHandles {
    input_layernorm: WeightHandle,
    post_attention_layernorm: WeightHandle,
    q_proj: WeightHandle,
    k_proj: WeightHandle,
    v_proj: WeightHandle,
    o_proj: WeightHandle,
    q_norm: WeightHandle,
    k_norm: WeightHandle,
    mlp_gate: WeightHandle,
    mlp_up: WeightHandle,
    mlp_down: WeightHandle,
    kinds: MatmulKinds,
}

#[derive(Clone, Copy, Debug)]
struct BlockBufs {
    input_layernorm: BufRef,
    post_attention_layernorm: BufRef,
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

/// Weight views pinned for one layer's submit.
pub struct BlockViews<'a> {
    input_layernorm: thinfer_core::residency::GpuView<'a>,
    post_attention_layernorm: thinfer_core::residency::GpuView<'a>,
    q_proj: thinfer_core::residency::GpuView<'a>,
    k_proj: thinfer_core::residency::GpuView<'a>,
    v_proj: thinfer_core::residency::GpuView<'a>,
    o_proj: thinfer_core::residency::GpuView<'a>,
    q_norm: thinfer_core::residency::GpuView<'a>,
    k_norm: thinfer_core::residency::GpuView<'a>,
    mlp_gate: thinfer_core::residency::GpuView<'a>,
    mlp_up: thinfer_core::residency::GpuView<'a>,
    mlp_down: thinfer_core::residency::GpuView<'a>,
}

impl BlockHandles {
    /// OS readahead for this layer's matmul weights (see
    /// [`WeightResidency::hint_read`]), issued ahead of its acquire.
    async fn hint_read<S: WeightSource>(
        &self,
        residency: &WeightResidency<S>,
    ) -> Result<(), S::Error> {
        for h in [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.mlp_gate,
            self.mlp_up,
            self.mlp_down,
        ] {
            residency.hint_read(h).await?;
        }
        Ok(())
    }

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

#[derive(Clone, Debug)]
pub struct EncoderHandles {
    layers: Vec<BlockHandles>,
}

impl EncoderHandles {
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Keep only the first `n` layers, so `forward` returns `hidden_states[n]`
    /// (parity bisection by depth).
    pub fn truncate(&mut self, n: usize) {
        self.layers.truncate(n);
    }
}

/// Fail loudly if the checkpoint is not the truncated build.
///
/// A full 64-layer stack would still run, but the state after layer 50 is what
/// H3 conditions on and the extra 14 layers are pure download and streaming
/// cost. More dangerously, a checkpoint with FEWER than 50 layers would silently
/// condition on the wrong depth, which produces plausible-but-wrong video with
/// no error anywhere.
pub fn check_depth(available: usize) -> Result<(), LoadError> {
    if available < config::N_LAYERS {
        return Err(LoadError::UnknownWeight(WeightId(format!(
            "model.layers.{}.self_attn.q_proj.weight (checkpoint has {available} layers; \
             MiniMax-H3 conditions on hidden_states[{}] and needs at least that many)",
            config::N_LAYERS - 1,
            config::N_LAYERS
        ))));
    }
    Ok(())
}

/// Register the 50 decoder layers H3 evaluates.
///
/// `model.embed_tokens.weight` is deliberately NOT registered: a prompt indexes
/// a few hundred of its 151936 rows, so rows are gathered straight from the
/// source rather than paging a 780 MB table through residency.
pub fn register_handles<S: WeightSource>(
    residency: &WeightResidency<S>,
    transcode: Option<QuantKind>,
) -> Result<EncoderHandles, LoadError> {
    let lin = |id: &WeightId| register_one(residency, id, TransposePolicy::Linear2D, transcode);
    let dense = |id: &WeightId| register_one(residency, id, TransposePolicy::None, None);
    let kind = |id: &WeightId, requant: Option<QuantKind>| -> Option<QuantKind> {
        requant.or_else(|| {
            match residency
                .source()
                .catalog()
                .get(id)
                .and_then(|e| e.encoding)
            {
                Some(StorageEncoding::Quant(k)) => Some(k),
                _ => None,
            }
        })
    };

    let mut layers = Vec::with_capacity(config::N_LAYERS);
    for b in (0..config::N_LAYERS).map(Qwen3BlockWeights::new) {
        layers.push(BlockHandles {
            input_layernorm: dense(&b.input_layernorm)?,
            post_attention_layernorm: dense(&b.post_attention_layernorm)?,
            q_proj: lin(&b.q_proj)?,
            k_proj: lin(&b.k_proj)?,
            v_proj: lin(&b.v_proj)?,
            o_proj: lin(&b.o_proj)?,
            q_norm: dense(&b.q_norm)?,
            k_norm: dense(&b.k_norm)?,
            mlp_gate: lin(&b.mlp_gate)?,
            mlp_up: lin(&b.mlp_up)?,
            // Never transcoded: the massive-activation precision lesson.
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
    Ok(EncoderHandles { layers })
}

/// Encoder pipeline cfgs. F32 acts: this is a conditioning tower and the
/// residual carries the outlier channels that make f16 unsafe (the umT5 /
/// large-residual encoder lesson).
///
/// The matmul `weight_dtype` selects the B-operand CONVENTION, not what the
/// kernel reads: every site pre-dequants its own kind (Q4_K, Q6_K v_proj) into a
/// dense N-major workspace, which is what a `Quant` config declares. `Bf16`
/// would select K-major and silently transpose every weight (the H3 DiT bug,
/// which also turned this tower's output to noise).
pub fn encoder_cfgs() -> BlockWgslConfigs {
    use thinfer_core::ops::{WeightDtype, WgslConfig};
    let ops = WgslConfig {
        bf16_quant_writes: false,
        act_dtype: ActDtype::F32,
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
        large_d_sdpa: true,
        fast_sdpa: false,
        decode_sdpa: false,
    }
}

/// [`encoder_cfgs`] for long sequences (a reference image puts thousands of
/// vision tokens in the prompt): bf16 activations, the reference's own
/// encoder dtype, so the matmuls and the causal attention run on the
/// accelerator's tensor cores (f32 acts have neither path, and a long f32
/// sequence trips the GPU watchdog). Q and K are RMS-normed before rope, so
/// the f16 attention inputs stay in range.
pub fn encoder_cfgs_long() -> BlockWgslConfigs {
    use thinfer_core::ops::{WeightDtype, WgslConfig};
    let base = encoder_cfgs();
    let ops = WgslConfig {
        act_dtype: ActDtype::Bf16,
        ..base.ops
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
        matmul_adaln: ops,
        ops,
        fast_sdpa: true,
        // Head dim 128: no large-D kernel (and it has no bf16 variant).
        large_d_sdpa: false,
        ..base
    }
}

/// `BlockPipelines` plus a dense dequant step per quant kind.
///
/// The shipped Q4_K_M conditioner is MIXED per tensor (Q4_K for most sites,
/// Q6_K for `v_proj`), so a single compile-time scheme is wrong: the block
/// matmul reads a bf16 dequant workspace regardless of source kind, and only
/// this step varies by site.
pub struct EncoderPipelines {
    pub block: BlockPipelines,
    dense_dequant: HashMap<QuantKind, DequantStep>,
}

impl EncoderPipelines {
    pub async fn compile(
        backend: &WgpuBackend,
        cfgs: &BlockWgslConfigs,
    ) -> Result<Self, WgpuError> {
        let block = BlockPipelines::compile(backend, cfgs).await?;
        use thinfer_core::ops::dequant::{DequantTarget, build_wgsl, layout};
        let dq_layout = layout();
        let mut dense_dequant = HashMap::new();
        for scheme in [
            QuantKind::Q8_0,
            QuantKind::Q4_K,
            QuantKind::Q5_K,
            QuantKind::Q6_K,
        ] {
            let wgsl = build_wgsl(scheme, DequantTarget::Bf16);
            let pipeline = backend
                .create_pipeline(
                    &format!("minimax_encoder_dequant_{scheme:?}"),
                    &wgsl,
                    "main",
                    dq_layout,
                )
                .await?;
            dense_dequant.insert(
                scheme,
                DequantStep {
                    pipeline,
                    scheme: thinfer_core::ops::DequantScheme::Block(scheme),
                },
            );
        }
        Ok(Self {
            block,
            dense_dequant,
        })
    }

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
        1.0 / (self.head_dim as f32).sqrt()
    }
}

/// One Qwen3 decoder layer: pre-norm GQA self-attn (per-head Q/K RMSNorm,
/// half-rot rope, causal SDPA) -> residual -> pre-norm SwiGLU FFN -> residual.
/// Two norms per block, no post-norms and no `(1+w)` bake (that is Gemma).
struct DecoderBlock {
    shape: BlockShape,
}

impl DecoderBlock {
    #[allow(clippy::too_many_arguments)]
    fn forward<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &EncoderPipelines,
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
        let q_w = scope.import(&bufs.q_proj);
        let k_w = scope.import(&bufs.k_proj);
        let v_w = scope.import(&bufs.v_proj);
        let qn_w = scope.import(&bufs.q_norm);
        let kn_w = scope.import(&bufs.k_norm);
        let o_w = scope.import(&bufs.o_proj);
        let g_w = scope.import(&bufs.mlp_gate);
        let up_w = scope.import(&bufs.mlp_up);
        let down_w = scope.import(&bufs.mlp_down);

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

        // --- attn: pre-norm, qkv, qk-norm, rope, sdpa, o_proj, residual ---
        let n1 = alloc_act(scope, p, rows, dim)?;
        op_rmsnorm(scope, p, x_in, in_ln, n1, rows, dim, eps)?;

        let q = mm(n1, q_w, hq * hd, dim, pipelines.dequant_for(kinds.q))?;
        let k = mm(n1, k_w, hkv * hd, dim, pipelines.dequant_for(kinds.k))?;
        let v = mm(n1, v_w, hkv * hd, dim, pipelines.dequant_for(kinds.v))?;

        let qn = alloc_act(scope, p, rows * hq, hd)?;
        op_rmsnorm(scope, p, ActBuf::dense(q), qn_w, qn, rows * hq, hd, eps)?;
        let kn = alloc_act(scope, p, rows * hkv, hd)?;
        op_rmsnorm(scope, p, ActBuf::dense(k), kn_w, kn, rows * hkv, hd, eps)?;

        let qr = alloc_act(scope, p, rows, hq * hd)?;
        op_rope_halfrot(scope, p, qn, freqs_in, qr, rows, hq, hd)?;
        let kr = alloc_act(scope, p, rows, hkv * hd)?;
        op_rope_halfrot(scope, p, kn, freqs_in, kr, rows, hkv, hd)?;

        let sa = alloc_act(scope, p, rows, hq * hd)?;
        // f16 tensor-core attention under the bf16 config (see
        // `encoder_cfgs_long`); the f32 config compiles no f16 kernel and this
        // is the plain backend SDPA.
        op_sdpa_f16(
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
        let after_attn = alloc_act(scope, p, rows, dim)?;
        op_add(scope, p, x_in, ActBuf::dense(proj), after_attn)?;

        // --- ffn: pre-norm, SwiGLU, down, residual ---
        let n2 = alloc_act(scope, p, rows, dim)?;
        op_rmsnorm(scope, p, after_attn, pa_ln, n2, rows, dim, eps)?;
        let g = mm(n2, g_w, hid, dim, pipelines.dequant_for(kinds.gate))?;
        let up = mm(n2, up_w, hid, dim, pipelines.dequant_for(kinds.up))?;
        let gu = alloc_act(scope, p, rows, hid)?;
        op_silu_mul(scope, p, ActBuf::dense(g), ActBuf::dense(up), gu)?;
        let down = mm(gu, down_w, dim, hid, pipelines.dequant_for(kinds.down))?;
        op_add(scope, p, after_attn, ActBuf::dense(down), y_out)?;
        Ok(())
    }
}

/// The conditioning hidden state, `[seq, HIDDEN]` row-major f32.
#[derive(Clone, Debug)]
pub struct EncoderOutput {
    pub hidden: Vec<f32>,
    pub seq: usize,
}

/// Qwen3-VL-32B conditioner truncated to the layers H3 reads.
pub struct MiniMaxTextEncoder;

impl MiniMaxTextEncoder {
    /// Run the embedding + 50 decoder layers and return the LAST layer's output
    /// WITHOUT the final norm. Real tokens only, plain causal mask. `images`
    /// (sequence order) replace their `<|image_pad|>` rows' embeddings, move
    /// those rows onto 3-D M-RoPE positions, and add their deepstack streams
    /// into the same rows after layers 0, 1 and 2.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipelines: &EncoderPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        handles: &EncoderHandles,
        source: &S,
        token_ids: &[u32],
        images: &[ImageRows<'_>],
    ) -> Result<EncoderOutput, ForwardError<S::Error>> {
        let seq = token_ids.len();
        assert!(seq > 0, "MiniMaxTextEncoder::forward: empty token list");
        // Even-pad for the f16/bf16 mask word layout; the pad row repeats the
        // last token, is causally invisible, and is sliced off after readback.
        let mut ids = token_ids.to_vec();
        if !ids.len().is_multiple_of(2) {
            ids.push(*ids.last().expect("non-empty"));
        }
        let seq_pad = ids.len();
        let bp = &pipelines.block;

        // No embed scale: that is a Gemma detail, not a Qwen3 one.
        let mut embeds = embed_lookup_hidden(source, &ids, config::HIDDEN)
            .instrument(tracing::debug_span!(target: PHASE, "minimax.embed_lookup", seq))
            .await
            .map_err(ForwardError::Embed)?;
        let d = config::HIDDEN;
        let rows_of = |img: &ImageRows<'_>| img.start * d..(img.start + img.vision.tokens) * d;
        for img in images {
            assert!(
                img.start + img.vision.tokens <= seq,
                "image rows past the prompt"
            );
            embeds[rows_of(img)].copy_from_slice(&img.vision.embeds);
        }

        let shape = BlockShape {
            dim: config::HIDDEN,
            n_heads: config::N_HEADS,
            n_kv_heads: config::N_KV_HEADS,
            head_dim: config::HEAD_DIM,
            ffn_hidden: config::FFN_HIDDEN,
            seq: seq_pad,
            norm_eps: config::EPS,
        };
        let block = DecoderBlock { shape };
        let act_bytes = bp.act_bytes((seq_pad * config::HIDDEN) as u32);

        let x_buf = scratch.alloc(act_bytes)?;
        let embed_bytes = seq::act_upload_bytes(bp.act_dtype, &embeds);
        backend.write_buffer(x_buf.id, 0, &embed_bytes)?;

        // Interleaved M-RoPE through the half-rot kernel. A text-only request
        // gives every axis the same position, i.e. plain 1-axis rope.
        let freqs = crate::ideogram4::mrope::build_freqs(
            &mrope_positions(seq_pad, images),
            config::HEAD_DIM,
            config::ROPE_THETA,
            config::MROPE_SECTION,
        );
        let freqs_bytes = seq::freqs_upload_bytes(bp.act_dtype, &freqs);
        let freqs_buf = scratch.alloc(freqs_bytes.len() as u64)?;
        backend.write_buffer(freqs_buf.id, 0, &freqs_bytes)?;

        // The f16 attention reads its additive mask as f16 (a bf16 -inf is an
        // f16 NaN), whatever the activations are.
        let mask_dtype = if bp.sdpa_sg.is_some() && bp.act_dtype == ActDtype::Bf16 {
            ActDtype::F16
        } else {
            bp.act_dtype
        };
        let mask_bytes = seq::causal_mask_bytes_act(seq_pad, mask_dtype);
        let mask_buf = scratch.alloc(mask_bytes.len() as u64)?;
        backend.write_buffer(mask_buf.id, 0, &mask_bytes)?;

        // Deepstack stream k, laid over the whole sequence (zero off the image
        // rows), added after layer k.
        let mut deepstack = Vec::new();
        if !images.is_empty() {
            for k in 0..3 {
                let mut full = vec![0.0f32; seq_pad * d];
                for img in images {
                    full[rows_of(img)].copy_from_slice(&img.vision.deepstack[k]);
                }
                let bytes = seq::act_upload_bytes(bp.act_dtype, &full);
                let buf = scratch.alloc(bytes.len() as u64)?;
                backend.write_buffer(buf.id, 0, &bytes)?;
                deepstack.push(buf);
            }
        }

        let mut cur = x_buf;
        let mut pending: Option<BlockViews<'_>> = if handles.layers.is_empty() {
            None
        } else {
            Some(handles.layers[0].acquire(residency, backend).await?)
        };
        let freqs_ref = freqs_buf.as_buf_ref();
        let mask_ref = mask_buf.as_buf_ref();

        for idx in 0..handles.layers.len() {
            let _guard = trace::scope!(format!("minimax.encoder.layer.{idx}")).entered();
            let views = pending.take().expect("pending acquire missing");
            // Layer idx+1 is acquired during this layer's submit; start the
            // OS reading idx+2 now so cold reads keep the disk queue full.
            if let Some(h) = handles.layers.get(idx + 2) {
                h.hint_read(residency)
                    .await
                    .map_err(ResidencyError::<_, WgpuError>::Source)?;
            }
            let bufs = views.bufs();
            let cur_ref = cur.as_buf_ref();
            let nxt = scratch.alloc(act_bytes)?;
            let nxt_ref = nxt.as_buf_ref();
            let scope = scratch.batch();
            let cur_h = scope.import(&cur_ref);
            let freqs_h = scope.import(&freqs_ref);
            let mask_h = scope.import(&mask_ref);
            let nxt_h = scope.import(&nxt_ref);
            match deepstack.get(idx) {
                None => block.forward(
                    &scope,
                    pipelines,
                    cur_h,
                    freqs_h,
                    mask_h,
                    nxt_h,
                    &bufs,
                    handles.layers[idx].kinds,
                )?,
                Some(ds) => {
                    let out = alloc_act(&scope, bp, seq_pad as u32, config::HIDDEN as u32)?;
                    block.forward(
                        &scope,
                        pipelines,
                        cur_h,
                        freqs_h,
                        mask_h,
                        out.data,
                        &bufs,
                        handles.layers[idx].kinds,
                    )?;
                    let ds_ref = ds.as_buf_ref();
                    op_add(
                        &scope,
                        bp,
                        out,
                        ActBuf::dense(scope.import(&ds_ref)),
                        ActBuf::dense(nxt_h),
                    )?;
                }
            }

            // Overlap the next layer's weight fetch with this layer's submit.
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
                .instrument(tracing::debug_span!(target: PHASE, "minimax.encoder.submit", idx));
            let (submit_res, next_res) = futures::join!(submit_fut, next_acquire);
            submit_res?;
            pending = next_res?;

            drop(views);
            cur = nxt;
        }

        // Read back ONLY the final state. No `model.norm`: H3 conditions on the
        // unnormalized hidden state.
        let bytes = backend.read_buffer(cur.id(), 0, act_bytes).await?;
        let mut hidden = seq::act_readback_to_f32(bp.act_dtype, &bytes, seq_pad * config::HIDDEN);
        hidden.truncate(seq * config::HIDDEN);
        Ok(EncoderOutput { hidden, seq })
    }
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

    /// The conditioning depth is the whole point of the truncated checkpoint.
    #[test]
    fn conditioning_depth_matches_the_packing_constant() {
        assert_eq!(
            config::N_LAYERS,
            crate::minimax::packing::TEXT_ENCODER_LAYER
        );
        const { assert!(config::N_LAYERS < config::FULL_LAYERS) };
    }

    #[test]
    fn depth_check_rejects_a_short_checkpoint() {
        assert!(check_depth(config::N_LAYERS).is_ok());
        assert!(check_depth(config::FULL_LAYERS).is_ok());
        assert!(check_depth(config::N_LAYERS - 1).is_err());
    }

    /// Shapes must agree with the DiT's expected conditioning width, or
    /// `condition_proj` silently reads the wrong stride.
    #[test]
    fn hidden_matches_the_dit_text_dim() {
        assert_eq!(config::HIDDEN, crate::minimax::DitConfig::h3_33b().text_dim);
    }

    #[test]
    fn gqa_widths_match_the_checkpoint_shapes() {
        // q_proj [5120, 8192], k/v_proj [5120, 1024], mlp [5120, 25600].
        assert_eq!(config::N_HEADS * config::HEAD_DIM, 8192);
        assert_eq!(config::N_KV_HEADS * config::HEAD_DIM, 1024);
        assert!(config::N_HEADS.is_multiple_of(config::N_KV_HEADS));
    }

    #[test]
    fn text_only_positions_are_sequential_on_every_axis() {
        let p = mrope_positions(4, &[]);
        assert_eq!(p, vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]);
    }

    /// The spec's worked example: 7 label/start rows, then a (96 x 64) merged
    /// image at t = 7, then vision_end at 7 + max(96, 64) = 103.
    #[test]
    fn image_holds_t_steps_hw_and_advances_by_its_longer_side() {
        let vision = crate::minimax::vision::VisionOutput {
            embeds: Vec::new(),
            deepstack: [Vec::new(), Vec::new(), Vec::new()],
            tokens: 96 * 64,
            merged_grid: (96, 64),
        };
        let img = ImageRows {
            start: 7,
            vision: &vision,
        };
        let seq = 7 + 96 * 64 + 2;
        let p = mrope_positions(seq, &[img]);
        assert_eq!(p.len(), seq * 3);
        assert_eq!(&p[6 * 3..7 * 3], &[6, 6, 6]);
        assert_eq!(&p[7 * 3..8 * 3], &[7, 7, 7]);
        // Second image row: (t, h, w) = (7, 7, 8); first of row 1: (7, 8, 7).
        assert_eq!(&p[8 * 3..9 * 3], &[7, 7, 8]);
        assert_eq!(&p[(7 + 64) * 3..(8 + 64) * 3], &[7, 8, 7]);
        let after = 7 + 96 * 64;
        assert_eq!(&p[after * 3..(after + 1) * 3], &[103, 103, 103]);
    }
}
