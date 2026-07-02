//! Full (non-KV-cached) causal forward for the Qwen3-VL-8B-Instruct rewriter LM.
//!
//! One dense pass over ALL 36 decoder layers, the final `model.norm`, and the
//! UNTIED `lm_head`; returns the VOCAB next-token logits at the LAST prompt
//! position. No incremental decode, no sampling loop, no detokenize (Phase 3).
//!
//! The per-layer math is identical to the Z-Image Qwen3 block
//! ([`crate::z_image::text_encoder::Qwen3Block`]) EXCEPT the v projection, whose
//! GGUF encoding is Q6_K (q/k/o/gate/up are Q5_K); it dispatches through the
//! repurposed `matmul_qkv_self` site so the two weight encodings coexist in one
//! block. See [`wgsl_configs`] for the site->encoding map.
//!
//! Activation dtype is bf16 (load-bearing): Qwen3's residual stream carries
//! massive-activation outliers that f16 acts would clip, corrupting the block
//! output (same reason the Z-Image encoder runs bf16 acts).

use thinfer_core::Backend;
use thinfer_core::backend::{WgpuBackend, WgpuError};
use thinfer_core::ops::{ActDtype, WeightDtype, WgslConfig};
use thinfer_core::quant::QuantKind;
use thinfer_core::residency::{ResidencyError, WeightResidency};
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace, WsBuf};

use thinfer_core::backend::{BufRef, WgpuPipeline};
use thinfer_core::ops::{MatMulConfig, MatMulF32, MatmulOp};

use crate::common::block::{
    ActBuf, Block, BlockPipelines, BlockWgslConfigs, CoopmatSites, DenseActSites, DequantStep,
    alloc_act, alloc_matmul_out_buf, copy_tap, op_add, op_rmsnorm, op_rope_halfrot, op_sdpa,
    op_sdpa_decode, op_silu_mul,
};
use crate::common::rope_embedder::RopeEmbedder;
use crate::common::seq;
use crate::qwen3_lm::Qwen3LmConfig;
use crate::qwen3_lm::generate::Qwen3LmHandles;
use crate::z_image::text_encoder::{
    Qwen3BlockBufs, Qwen3BlockShape, Qwen3ForwardError, embed_lookup_hidden,
};

/// Per-site WGSL config set for the rewriter block.
///
/// The GGUF ships a mixed K-quant: q/k/o/gate/up = Q5_K, v/ffn_down/lm_head =
/// Q6_K, all norms F32. `matmul_qkv_self` is repurposed to drive the Q6_K v
/// projection (the shared `Block::forward` never uses that site; here it does).
/// `matmul_ffn_down` (Q6_K) also drives the Q6_K `lm_head`. Acts + weights for
/// the elementwise `ops` stay bf16.
pub fn wgsl_configs() -> BlockWgslConfigs {
    let ops = WgslConfig {
        act_dtype: ActDtype::Bf16,
        weight_dtype: WeightDtype::Bf16,
        bf16_quant_writes: false,
    };
    let q5 = WgslConfig {
        weight_dtype: WeightDtype::Quant(QuantKind::Q5_K),
        ..ops
    };
    let q6 = WgslConfig {
        weight_dtype: WeightDtype::Quant(QuantKind::Q6_K),
        ..ops
    };
    // The Q5_K_M GGUF mixes quant PER LAYER: q/k/o/gate/up are always Q5_K, but
    // attn_v and ffn_down are Q6_K in ~half the layers and Q5_K in the rest
    // (lm_head is Q6_K, token_embd Q5_K). The forward routes each matmul by the
    // weight's ACTUAL catalog quant, and by M: DECODE (M=1) uses the inline-Quant
    // decode pipelines built in `compile_pipelines`, PREFILL (M>1) uses these
    // per-site bf16 dequant-once workspace pipelines (`matmul_qkv` = Q5_K,
    // `matmul_qkv_self` = Q6_K, both K-agnostic). See `Qwen3LmPipelines::matmul_for`.
    // The other block matmul sites (proj/ffn_*) and the elementwise/norm/rope/sdpa
    // ops come from this same `BlockPipelines`.
    BlockWgslConfigs {
        matmul_qkv: q5,      // general Q5_K dense matmul
        matmul_qkv_self: q6, // general Q6_K dense matmul
        matmul_proj: q5,     // unused
        matmul_ffn_up: q5,   // unused
        matmul_ffn_down: q6, // unused
        matmul_adaln: ops,   // unused (bf16)
        ops,
        i8_sdpa: false,
        // PARITY BASELINE: all sites dense (no i8 act quant anywhere), so the
        // logits track llama.cpp's f32/f16-act path and the Qwen3 massive-
        // activation attention-sink outliers are never crushed by i8-per-32-block.
        // Once parity is green, re-add i8 on safe post-norm sites for prefill
        // speed (a separate, quality-checked config).
        dense_acts: DenseActSites {
            qkv: true,
            qkv_self: true,
            proj: true,
            ffn_up: true,
            ffn_down: true,
        },
        coopmat_acts: CoopmatSites::default(),
        large_d_sdpa: false,
        fast_sdpa: false,
        // Single-token decode attends the whole (long) KV cache; the cooperative
        // `SdpaDecode` kernel replaces `SdpaF32`'s one-thread-per-head softmax
        // (the measured long-ctx decode bottleneck). Prefill (rows>1) keeps the
        // chunked dense path. Bit-equivalent either way.
        decode_sdpa: true,
    }
}

/// The rewriter pipeline set: the shared block pipelines (`bp`) PLUS two
/// DECODE-tuned INLINE-Quant matmul pipelines (Q5_K, Q6_K) that read the compact
/// GGUF Q weight directly and dequant in-register (f32 accumulate).
///
/// The decode (M=1) matmuls use these inline pipelines: no per-matmul bf16
/// dequant workspace to retain, so decode layers batch into one submit, and the
/// thin `DECODE_MM_CFG` tile avoids the square tile's ~64x M-waste. PREFILL /
/// full-forward matmuls (M>1) go through `bp`'s bf16 dequant-once workspace
/// pipelines instead (see `matmul_for`), which is faster for a wide-M GEMM.
/// One Q5_K pipeline serves every Q5_K weight and one Q6_K every Q6_K weight
/// (K-agnostic; dims via uniforms); a single `mm_op_dec` drives both.
pub struct Qwen3LmPipelines {
    pub bp: BlockPipelines,
    pub mm_q5_dec: WgpuPipeline,
    pub mm_q6_dec: WgpuPipeline,
    pub mm_op_dec: MatMulF32,
}

impl Qwen3LmPipelines {
    /// The matmul `(pipeline, dequant_dense, op)` for a weight of the given quant
    /// at row count `m`, splitting DECODE from PREFILL:
    ///
    /// * `m == 1` (decode / lm_head): the INLINE decode tile (`dequant_dense =
    ///   None`) reads the compact weight and dequants in-register. No workspace,
    ///   so decode layers batch into one submit, and the thin `bm` avoids the
    ///   square tile's ~64x M-waste.
    /// * `m > 1` (prefill / full forward): the bf16-DEQUANT-ONCE workspace path
    ///   (`bp.matmul_*` + `bp.dequant_*`, `dequant_dense = Some`). For a wide-M
    ///   GEMM the inline kernel re-dequants each B tile ~M/bm times; dequant-once
    ///   into a dense bf16 workspace is measured faster (24 -> ~38 tok/s prefill).
    ///   The retained per-layer workspace forces one layer per submit (see
    ///   `run_layers_kv`), which is also the TDR-safe granularity for wide M.
    fn matmul_for(
        &self,
        quant: QuantKind,
        m: u32,
    ) -> (&WgpuPipeline, Option<&DequantStep>, &MatMulF32) {
        if m == 1 {
            let (p, op) = match quant {
                QuantKind::Q5_K => (&self.mm_q5_dec, &self.mm_op_dec),
                QuantKind::Q6_K => (&self.mm_q6_dec, &self.mm_op_dec),
                other => panic!("qwen3_lm: unsupported matmul weight quant {other:?}"),
            };
            (p, None, op)
        } else {
            let bp = &self.bp;
            match quant {
                QuantKind::Q5_K => (&bp.matmul_qkv, bp.dequant_qkv.as_ref(), &bp.matmuls.qkv),
                QuantKind::Q6_K => (
                    &bp.matmul_qkv_self,
                    bp.dequant_qkv_self.as_ref(),
                    &bp.matmuls.qkv_self,
                ),
                other => panic!("qwen3_lm: unsupported matmul weight quant {other:?}"),
            }
        }
    }
}

/// Decode (M=1) inline-Quant tile. A thin bm (4) keeps the workgroup busy on the
/// B-dequant + N reduction without the ~64x wasted M-accumulation the square
/// prefill tile would incur at M=1 (measured 6.2x faster decode than a 64x64
/// tile on an RTX 5070). bk=64 for the K-family (block_size 256) sub-block
/// cooperative dequant; `b_nmajor = false` (the Quant arm has its own N-major
/// block indexing and forbids `b_nmajor`). Override for tuning via
/// `THINFER_QWEN3LM_DECTILE=bm,bn,bk,tm,tn`.
const DECODE_MM_CFG: MatMulConfig = MatMulConfig {
    bm: 4,
    bn: 64,
    bk: 64,
    tm: 1,
    tn: 2,
    b_nmajor: false,
};

fn decode_mm_cfg() -> MatMulConfig {
    match std::env::var("THINFER_QWEN3LM_DECTILE") {
        Ok(s) => {
            let v: Vec<u32> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            assert_eq!(v.len(), 5, "THINFER_QWEN3LM_DECTILE=bm,bn,bk,tm,tn");
            MatMulConfig {
                bm: v[0],
                bn: v[1],
                bk: v[2],
                tm: v[3],
                tn: v[4],
                b_nmajor: false,
            }
        }
        Err(_) => DECODE_MM_CFG,
    }
}

/// Compile the rewriter pipeline set: the shared block pipelines (for the
/// elementwise / norm / rope / sdpa ops) plus the inline-Quant matmul pipelines
/// (Q5_K, Q6_K x prefill/decode tiles) that read the compact weight directly.
pub async fn compile_pipelines(backend: &WgpuBackend) -> Result<Qwen3LmPipelines, WgpuError> {
    let cfgs = wgsl_configs();
    let bp = BlockPipelines::compile(backend, &cfgs).await?;
    let mm_layout = <MatMulF32 as MatmulOp>::layout();
    // bf16 acts, INLINE Quant weight (NOT the bf16-workspace override that
    // `BlockPipelines::compile` applies): the kernel reads the raw Q bytes.
    let q5_cfg = WgslConfig {
        weight_dtype: WeightDtype::Quant(QuantKind::Q5_K),
        ..cfgs.ops
    };
    let q6_cfg = WgslConfig {
        weight_dtype: WeightDtype::Quant(QuantKind::Q6_K),
        ..cfgs.ops
    };
    let mm_op_dec = MatMulF32::new(decode_mm_cfg());
    let mm_q5_dec = backend
        .create_pipeline(
            "qwen3lm_mm_q5k_dec",
            &mm_op_dec.wgsl(&q5_cfg),
            "main",
            mm_layout,
        )
        .await?;
    let mm_q6_dec = backend
        .create_pipeline(
            "qwen3lm_mm_q6k_dec",
            &mm_op_dec.wgsl(&q6_cfg),
            "main",
            mm_layout,
        )
        .await?;
    Ok(Qwen3LmPipelines {
        bp,
        mm_q5_dec,
        mm_q6_dec,
        mm_op_dec,
    })
}

/// Build a block shape for the rewriter LM at padded length `seq_pad`.
fn block_shape(cfg: &Qwen3LmConfig, seq_pad: usize) -> Qwen3BlockShape {
    Qwen3BlockShape {
        dim: cfg.hidden,
        n_heads: cfg.n_heads,
        n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim,
        ffn_hidden: cfg.ffn_hidden,
        seq: seq_pad,
        norm_eps: cfg.rms_norm_eps,
    }
}

/// One rewriter decoder layer: pre-norm GQA self-attn (per-head Q/K RMSNorm,
/// half-rot RoPE, causal SDPA) -> residual -> pre-norm SwiGLU FFN -> residual.
///
/// Mirrors [`crate::z_image::text_encoder::Qwen3Block::forward_taps`] EXACTLY
/// except the v projection routes through the `qkv_self` (Q6_K) site instead of
/// `qkv` (Q5_K), and there are no diagnostic taps.
#[allow(clippy::too_many_arguments)]
fn lm_block_forward<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &Qwen3LmPipelines,
    shape: Qwen3BlockShape,
    x_in: BatchBuf<'wsp>,
    freqs_in: BatchBuf<'wsp>,
    mask_in: BatchBuf<'wsp>,
    y_out: BatchBuf<'wsp>,
    bufs: &'wsp Qwen3BlockBufs,
    // Per-layer quant of attn_v and ffn_down (Q5_K or Q6_K; the only weights that
    // vary across layers). All other matmul weights are Q5_K.
    v_quant: QuantKind,
    down_quant: QuantKind,
    // Diagnostic taps [n1, q, k, v], each a BufRef sized to the op output; `None`
    // in production. Localizes a per-layer NaN to a specific projection.
    taps: Option<&[BufRef; 4]>,
) -> Result<(), WgpuError> {
    let bp = &pipelines.bp;
    let rows = shape.seq as u32;
    let dim = shape.dim as u32;
    let hd = shape.head_dim as u32;
    let hq = shape.n_heads as u32;
    let hkv = shape.n_kv_heads as u32;
    let hid = shape.ffn_hidden as u32;
    let eps = shape.norm_eps;
    let scale = 1.0 / (shape.head_dim as f32).sqrt();
    let x_in = ActBuf::dense(x_in);
    let y_out = ActBuf::dense(y_out);

    // Route every matmul by its weight's actual quant (q/k/o/gate/up are Q5_K;
    // v and down vary per layer) and by M (decode=inline, prefill=workspace).
    let (q5_mm, q5_dq, q5_op) = pipelines.matmul_for(QuantKind::Q5_K, rows);
    let (v_mm, v_dq, v_op) = pipelines.matmul_for(v_quant, rows);
    let (d_mm, d_dq, d_op) = pipelines.matmul_for(down_quant, rows);

    let in_ln = scope.import(&bufs.input_layernorm);
    let q_w = scope.import(&bufs.q_proj);
    let k_w = scope.import(&bufs.k_proj);
    let v_w = scope.import(&bufs.v_proj);
    let qn_w = scope.import(&bufs.q_norm);
    let kn_w = scope.import(&bufs.k_norm);
    let o_w = scope.import(&bufs.o_proj);
    let pa_ln = scope.import(&bufs.post_attention_layernorm);
    let g_w = scope.import(&bufs.mlp_gate);
    let up_w = scope.import(&bufs.mlp_up);
    let down_w = scope.import(&bufs.mlp_down);

    // --- pre-attn norm ---
    let n1 = alloc_act(scope, bp, rows, dim)?;
    op_rmsnorm(scope, bp, x_in, in_ln, n1, rows, dim, eps)?;
    copy_tap(
        scope,
        n1.data,
        taps.map(|t| &t[0]),
        bp.act_bytes(rows * dim),
    )?;

    // --- q/k projections (Q5_K, inline) ---
    let q_scratch = alloc_matmul_out_buf(scope, bp, rows * hq * hd)?;
    let dims_q = scope.u32x4_uniform(rows, hq * hd, dim, 0)?;
    Block::dispatch_matmul_site(
        scope,
        bp,
        n1,
        q_w,
        q_scratch,
        dims_q,
        None,
        None,
        q5_dq,
        q5_mm,
        q5_op,
        rows,
        hq * hd,
        dim,
    )?;
    let k_scratch = alloc_matmul_out_buf(scope, bp, rows * hkv * hd)?;
    let dims_k = scope.u32x4_uniform(rows, hkv * hd, dim, 0)?;
    Block::dispatch_matmul_site(
        scope,
        bp,
        n1,
        k_w,
        k_scratch,
        dims_k,
        None,
        None,
        q5_dq,
        q5_mm,
        q5_op,
        rows,
        hkv * hd,
        dim,
    )?;
    // --- v projection (per-layer quant, inline) ---
    let v_scratch = alloc_matmul_out_buf(scope, bp, rows * hkv * hd)?;
    let dims_v = scope.u32x4_uniform(rows, hkv * hd, dim, 0)?;
    Block::dispatch_matmul_site(
        scope,
        bp,
        n1,
        v_w,
        v_scratch,
        dims_v,
        None,
        None,
        v_dq,
        v_mm,
        v_op,
        rows,
        hkv * hd,
        dim,
    )?;
    copy_tap(
        scope,
        q_scratch,
        taps.map(|t| &t[1]),
        bp.act_bytes(rows * hq * hd),
    )?;
    copy_tap(
        scope,
        k_scratch,
        taps.map(|t| &t[2]),
        bp.act_bytes(rows * hkv * hd),
    )?;
    copy_tap(
        scope,
        v_scratch,
        taps.map(|t| &t[3]),
        bp.act_bytes(rows * hkv * hd),
    )?;
    let q = ActBuf::dense(q_scratch);
    let k = ActBuf::dense(k_scratch);
    let v = ActBuf::dense(v_scratch);

    // --- per-head Q/K RMSNorm over head_dim ---
    let qn = alloc_act(scope, bp, rows * hq, hd)?;
    op_rmsnorm(scope, bp, q, qn_w, qn, rows * hq, hd, eps)?;
    let kn = alloc_act(scope, bp, rows * hkv, hd)?;
    op_rmsnorm(scope, bp, k, kn_w, kn, rows * hkv, hd, eps)?;

    // --- rope on Q, K (1-axis token-position freqs, half-rot) ---
    let qr = alloc_act(scope, bp, rows, hq * hd)?;
    op_rope_halfrot(scope, bp, qn, freqs_in, qr, rows, hq, hd)?;
    let kr = alloc_act(scope, bp, rows, hkv * hd)?;
    op_rope_halfrot(scope, bp, kn, freqs_in, kr, rows, hkv, hd)?;

    // --- causal sdpa ---
    let sa = alloc_act(scope, bp, rows, hq * hd)?;
    op_sdpa(
        scope, bp, qr, kr, v, mask_in, sa, 1, rows, rows, hq, hkv, hd, scale, 1,
    )?;

    // --- o_proj + residual ---
    let proj_scratch = alloc_matmul_out_buf(scope, bp, rows * dim)?;
    let dims_proj = scope.u32x4_uniform(rows, dim, hq * hd, 0)?;
    Block::dispatch_matmul_site(
        scope,
        bp,
        sa,
        o_w,
        proj_scratch,
        dims_proj,
        None,
        None,
        q5_dq,
        q5_mm,
        q5_op,
        rows,
        dim,
        hq * hd,
    )?;
    let after_attn = alloc_act(scope, bp, rows, dim)?;
    op_add(scope, bp, x_in, ActBuf::dense(proj_scratch), after_attn)?;

    // --- pre-ffn norm ---
    let n2 = alloc_act(scope, bp, rows, dim)?;
    op_rmsnorm(scope, bp, after_attn, pa_ln, n2, rows, dim, eps)?;

    // --- SwiGLU FFN: down(silu_mul(gate(x), up(x))) ---
    let g_scratch = alloc_matmul_out_buf(scope, bp, rows * hid)?;
    let dims_g = scope.u32x4_uniform(rows, hid, dim, 0)?;
    Block::dispatch_matmul_site(
        scope, bp, n2, g_w, g_scratch, dims_g, None, None, q5_dq, q5_mm, q5_op, rows, hid, dim,
    )?;
    let up_scratch = alloc_matmul_out_buf(scope, bp, rows * hid)?;
    let dims_up = scope.u32x4_uniform(rows, hid, dim, 0)?;
    Block::dispatch_matmul_site(
        scope, bp, n2, up_w, up_scratch, dims_up, None, None, q5_dq, q5_mm, q5_op, rows, hid, dim,
    )?;
    let gu = alloc_act(scope, bp, rows, hid)?;
    op_silu_mul(
        scope,
        bp,
        ActBuf::dense(g_scratch),
        ActBuf::dense(up_scratch),
        gu,
    )?;
    let down_scratch = alloc_matmul_out_buf(scope, bp, rows * dim)?;
    let dims_down = scope.u32x4_uniform(rows, dim, hid, 0)?;
    Block::dispatch_matmul_site(
        scope,
        bp,
        gu,
        down_w,
        down_scratch,
        dims_down,
        None,
        None,
        d_dq,
        d_mm,
        d_op,
        rows,
        dim,
        hid,
    )?;
    op_add(scope, bp, after_attn, ActBuf::dense(down_scratch), y_out)?;

    Ok(())
}

/// One rewriter decoder layer with a persistent KV cache (incremental decode).
///
/// Same math as [`lm_block_forward`] but for `rows` NEW tokens at global
/// positions `[past_len, past_len + rows)`: it projects q/k/v for the new tokens
/// only, ropes q and the new k, WRITES the new post-rope k and the new v straight
/// into `k_cache[past_len..]` / `v_cache[past_len..]` (op dst is a cache subview,
/// no copy), then runs SDPA with the new q against the FULL cached K/V range
/// `[0, s_k)` (`s_k = past_len + rows`). `has_mask == 1` drives a caller-built
/// rectangular causal mask `[rows, s_k]` (prefill chunks); `has_mask == 0` skips
/// the mask entirely (single-token decode attends every cached key).
#[allow(clippy::too_many_arguments)]
fn lm_block_forward_kv<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &Qwen3LmPipelines,
    shape: Qwen3BlockShape,
    x_in: BatchBuf<'wsp>,
    freqs_in: BatchBuf<'wsp>,
    mask_in: BatchBuf<'wsp>,
    y_out: BatchBuf<'wsp>,
    bufs: &'wsp Qwen3BlockBufs,
    k_cache: BatchBuf<'wsp>,
    v_cache: BatchBuf<'wsp>,
    past_len: u32,
    rows: u32,
    s_k: u32,
    has_mask: u32,
    v_quant: QuantKind,
    down_quant: QuantKind,
) -> Result<(), WgpuError> {
    let bp = &pipelines.bp;
    let dim = shape.dim as u32;
    let hd = shape.head_dim as u32;
    let hq = shape.n_heads as u32;
    let hkv = shape.n_kv_heads as u32;
    let hid = shape.ffn_hidden as u32;
    let eps = shape.norm_eps;
    let scale = 1.0 / (shape.head_dim as f32).sqrt();
    let kv_width = hkv * hd;
    let x_in = ActBuf::dense(x_in);
    let y_out = ActBuf::dense(y_out);

    // DECODE (rows==1): inline decode tile (dequant_dense None). PREFILL chunk
    // (rows>1): bf16 dequant-once workspace path. See `matmul_for`.
    let (q5_mm, q5_dq, q5_op) = pipelines.matmul_for(QuantKind::Q5_K, rows);
    let (v_mm, v_dq, v_op) = pipelines.matmul_for(v_quant, rows);
    let (d_mm, d_dq, d_op) = pipelines.matmul_for(down_quant, rows);

    let in_ln = scope.import(&bufs.input_layernorm);
    let q_w = scope.import(&bufs.q_proj);
    let k_w = scope.import(&bufs.k_proj);
    let v_w = scope.import(&bufs.v_proj);
    let qn_w = scope.import(&bufs.q_norm);
    let kn_w = scope.import(&bufs.k_norm);
    let o_w = scope.import(&bufs.o_proj);
    let pa_ln = scope.import(&bufs.post_attention_layernorm);
    let g_w = scope.import(&bufs.mlp_gate);
    let up_w = scope.import(&bufs.mlp_up);
    let down_w = scope.import(&bufs.mlp_down);

    // --- pre-attn norm over the NEW rows ---
    let n1 = alloc_act(scope, bp, rows, dim)?;
    op_rmsnorm(scope, bp, x_in, in_ln, n1, rows, dim, eps)?;

    // --- q/k projections (Q5_K) ---
    let q_scratch = alloc_matmul_out_buf(scope, bp, rows * hq * hd)?;
    let dims_q = scope.u32x4_uniform(rows, hq * hd, dim, 0)?;
    Block::dispatch_matmul_site(
        scope,
        bp,
        n1,
        q_w,
        q_scratch,
        dims_q,
        None,
        None,
        q5_dq,
        q5_mm,
        q5_op,
        rows,
        hq * hd,
        dim,
    )?;
    let k_scratch = alloc_matmul_out_buf(scope, bp, rows * kv_width)?;
    let dims_k = scope.u32x4_uniform(rows, kv_width, dim, 0)?;
    Block::dispatch_matmul_site(
        scope, bp, n1, k_w, k_scratch, dims_k, None, None, q5_dq, q5_mm, q5_op, rows, kv_width, dim,
    )?;
    // --- v projection (per-layer quant): write straight into v_cache[past_len..] ---
    let v_off = bp.act_bytes(past_len * kv_width);
    let v_len = bp.act_bytes(rows * kv_width);
    let v_dst = scope.subview(&v_cache, v_off, v_len);
    let dims_v = scope.u32x4_uniform(rows, kv_width, dim, 0)?;
    Block::dispatch_matmul_site(
        scope, bp, n1, v_w, v_dst, dims_v, None, None, v_dq, v_mm, v_op, rows, kv_width, dim,
    )?;
    let q = ActBuf::dense(q_scratch);
    let k = ActBuf::dense(k_scratch);

    // --- per-head Q/K RMSNorm over head_dim ---
    let qn = alloc_act(scope, bp, rows * hq, hd)?;
    op_rmsnorm(scope, bp, q, qn_w, qn, rows * hq, hd, eps)?;
    let kn = alloc_act(scope, bp, rows * hkv, hd)?;
    op_rmsnorm(scope, bp, k, kn_w, kn, rows * hkv, hd, eps)?;

    // --- rope q (scratch) and the new k (straight into k_cache[past_len..]) ---
    let qr = alloc_act(scope, bp, rows, hq * hd)?;
    op_rope_halfrot(scope, bp, qn, freqs_in, qr, rows, hq, hd)?;
    let k_dst = ActBuf::dense(scope.subview(&k_cache, v_off, v_len));
    op_rope_halfrot(scope, bp, kn, freqs_in, k_dst, rows, hkv, hd)?;

    // --- SDPA: new q rows vs the FULL cached K/V range [0, s_k) ---
    let k_all = ActBuf::dense(scope.subview(&k_cache, 0, bp.act_bytes(s_k * kv_width)));
    let v_all = ActBuf::dense(scope.subview(&v_cache, 0, bp.act_bytes(s_k * kv_width)));
    let sa = alloc_act(scope, bp, rows, hq * hd)?;
    // DECODE (rows==1, no mask) attends the whole long KV cache -> the cooperative
    // `SdpaDecode` kernel (full workgroup over the KV), the measured decode win.
    // PREFILL (rows>1, causal mask) keeps the chunked dense path (TDR-safe at M).
    if rows == 1 {
        op_sdpa_decode(
            scope, bp, qr, k_all, v_all, mask_in, sa, 1, rows, s_k, hq, hkv, hd, scale, has_mask,
        )?;
    } else {
        op_sdpa(
            scope, bp, qr, k_all, v_all, mask_in, sa, 1, rows, s_k, hq, hkv, hd, scale, has_mask,
        )?;
    }

    // --- o_proj + residual ---
    let proj_scratch = alloc_matmul_out_buf(scope, bp, rows * dim)?;
    let dims_proj = scope.u32x4_uniform(rows, dim, hq * hd, 0)?;
    Block::dispatch_matmul_site(
        scope,
        bp,
        sa,
        o_w,
        proj_scratch,
        dims_proj,
        None,
        None,
        q5_dq,
        q5_mm,
        q5_op,
        rows,
        dim,
        hq * hd,
    )?;
    let after_attn = alloc_act(scope, bp, rows, dim)?;
    op_add(scope, bp, x_in, ActBuf::dense(proj_scratch), after_attn)?;

    // --- pre-ffn norm + SwiGLU FFN ---
    let n2 = alloc_act(scope, bp, rows, dim)?;
    op_rmsnorm(scope, bp, after_attn, pa_ln, n2, rows, dim, eps)?;
    let g_scratch = alloc_matmul_out_buf(scope, bp, rows * hid)?;
    let dims_g = scope.u32x4_uniform(rows, hid, dim, 0)?;
    Block::dispatch_matmul_site(
        scope, bp, n2, g_w, g_scratch, dims_g, None, None, q5_dq, q5_mm, q5_op, rows, hid, dim,
    )?;
    let up_scratch = alloc_matmul_out_buf(scope, bp, rows * hid)?;
    let dims_up = scope.u32x4_uniform(rows, hid, dim, 0)?;
    Block::dispatch_matmul_site(
        scope, bp, n2, up_w, up_scratch, dims_up, None, None, q5_dq, q5_mm, q5_op, rows, hid, dim,
    )?;
    let gu = alloc_act(scope, bp, rows, hid)?;
    op_silu_mul(
        scope,
        bp,
        ActBuf::dense(g_scratch),
        ActBuf::dense(up_scratch),
        gu,
    )?;
    let down_scratch = alloc_matmul_out_buf(scope, bp, rows * dim)?;
    let dims_down = scope.u32x4_uniform(rows, dim, hid, 0)?;
    Block::dispatch_matmul_site(
        scope,
        bp,
        gu,
        down_w,
        down_scratch,
        dims_down,
        None,
        None,
        d_dq,
        d_mm,
        d_op,
        rows,
        dim,
        hid,
    )?;
    op_add(scope, bp, after_attn, ActBuf::dense(down_scratch), y_out)?;
    Ok(())
}

/// Build a `[q_len, s_k]` additive causal mask (act dtype) for a KV-cached
/// prefill chunk: query row `i` is at global position `past_len + i`, and key
/// `j` is visible iff `j <= past_len + i` (0.0), else `-inf`. `s_k` must be even
/// so each row's packed-bf16/f16 pair stream aligns to `array<u32>` words (the
/// SDPA mask read derives the per-row word base as `(row * s_k) >> 1`).
fn causal_rect_mask_bytes(
    past_len: usize,
    q_len: usize,
    s_k: usize,
    act: thinfer_core::ops::ActDtype,
) -> Vec<u8> {
    assert!(
        s_k.is_multiple_of(2),
        "causal_rect_mask_bytes: s_k must be even (got {s_k})"
    );
    let mut vals = vec![0.0f32; q_len * s_k];
    for i in 0..q_len {
        let vis = past_len + i;
        for j in (vis + 1)..s_k {
            vals[i * s_k + j] = f32::NEG_INFINITY;
        }
    }
    seq::act_upload_bytes(act, &vals)
}

/// Persistent per-layer KV cache: two `[max_ctx, n_kv_heads*head_dim]` bf16
/// buffers per layer, allocated once from the `Workspace` so they survive across
/// the many prefill/decode submits. `k` stores POST-ROPE k, `v` stores raw v.
/// The `*_refs` mirror the `WsBuf` views for `scope.import` across scopes.
struct KvCache {
    _k: Vec<WsBuf<WgpuBackend>>,
    _v: Vec<WsBuf<WgpuBackend>>,
    k_refs: Vec<BufRef>,
    v_refs: Vec<BufRef>,
}

impl KvCache {
    fn alloc(
        scratch: &Workspace<WgpuBackend>,
        pipelines: &BlockPipelines,
        cfg: &Qwen3LmConfig,
        max_ctx: usize,
    ) -> Result<Self, WgpuError> {
        let kv_width = cfg.kv_width() as u32;
        let bytes = pipelines.act_bytes(max_ctx as u32 * kv_width);
        let mut k = Vec::with_capacity(cfg.n_layers);
        let mut v = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            k.push(scratch.alloc(bytes)?);
            v.push(scratch.alloc(bytes)?);
        }
        let k_refs = k.iter().map(|b| b.as_buf_ref()).collect();
        let v_refs = v.iter().map(|b| b.as_buf_ref()).collect();
        Ok(Self {
            _k: k,
            _v: v,
            k_refs,
            v_refs,
        })
    }
}

/// Full causal generator forward for the rewriter LM.
pub struct Qwen3LmGenerator {
    cfg: Qwen3LmConfig,
    rope: RopeEmbedder,
}

impl Qwen3LmGenerator {
    /// `max_seq` is the largest padded prompt length ever fed; the rope table
    /// sizes against it.
    pub fn new(cfg: Qwen3LmConfig, max_seq: usize) -> Self {
        let seq_len = max_seq.max(1);
        Self {
            // 1-axis rope (token position only); other axes are no-op (`d=0`).
            rope: RopeEmbedder::new(cfg.rope_theta, [cfg.head_dim, 0, 0], [seq_len, 1, 1]),
            cfg,
        }
    }

    pub fn rope(&self) -> &RopeEmbedder {
        &self.rope
    }

    /// Run all 36 decoder layers + final norm + lm_head over `token_ids` and
    /// return the VOCAB next-token logits at the LAST (real) token position as a
    /// host `Vec<f32>`. Argmax / sampling is the caller's job.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipelines: &Qwen3LmPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        handles: &Qwen3LmHandles,
        source: &S,
        token_ids: &[u32],
    ) -> Result<Vec<f32>, Qwen3ForwardError<S::Error>> {
        let bp = &pipelines.bp;
        let cfg = self.cfg;
        debug_assert_eq!(handles.layers.len(), cfg.n_layers);
        let seq = token_ids.len();
        assert!(seq > 0, "Qwen3LmGenerator::forward: empty token list");

        // Even-pad the sequence: the bf16 act/mask layouts pack two elems per
        // u32 word, so the mask/freqs paths require an even row count. The pad
        // row repeats the last token and is causally invisible to every real
        // row (it sits after the last real query); we only ever read the row
        // (seq-1) logits, so the pad row's own output is never used.
        let mut ids = token_ids.to_vec();
        if !ids.len().is_multiple_of(2) {
            ids.push(*ids.last().expect("non-empty token list"));
        }
        let seq_pad = ids.len();
        assert!(
            seq_pad <= self.rope.axes_lens[0],
            "prompt length {seq_pad} exceeds Qwen3LmGenerator rope max_seq {}",
            self.rope.axes_lens[0]
        );

        // --- CPU embedding lookup (GGUF Q-quant token_embd -> fp32 per row) ---
        let embeds = embed_lookup_hidden(source, &ids, cfg.hidden)
            .await
            .map_err(Qwen3ForwardError::Embed)?;

        let diag = std::env::var("THINFER_QWEN3LM_DIAG").is_ok();
        if diag {
            let nf = embeds.iter().filter(|x| !x.is_finite()).count();
            eprintln!(
                "[qwen3lm-diag] embeds: {nf}/{} non-finite, row0[..6]={:?}",
                embeds.len(),
                &embeds[..6.min(embeds.len())]
            );
        }

        let shape = block_shape(&cfg, seq_pad);
        let dim = cfg.hidden as u32;
        let act_bytes = bp.act_bytes(seq_pad as u32 * dim);

        // --- upload embeddings ---
        let x_buf = scratch.alloc(act_bytes)?;
        let embed_bytes = seq::act_upload_bytes(bp.act_dtype, &embeds);
        backend.write_buffer(x_buf.id, 0, &embed_bytes)?;

        // --- rope freqs: positions 0..seq_pad on axis 0 ---
        let mut pos_ids = vec![0_i32; seq_pad * 3];
        for (i, chunk) in pos_ids.chunks_exact_mut(3).enumerate() {
            chunk[0] = i as i32;
        }
        let freqs_bytes = seq::freqs_upload_bytes(bp.act_dtype, &self.rope.lookup(&pos_ids));
        let freqs_buf = scratch.alloc(freqs_bytes.len() as u64)?;
        backend.write_buffer(freqs_buf.id, 0, &freqs_bytes)?;

        // --- causal mask [1, seq_pad, seq_pad] ---
        let mask_bytes = seq::causal_mask_bytes_act(seq_pad, bp.act_dtype);
        let mask_buf = scratch.alloc(mask_bytes.len() as u64)?;
        backend.write_buffer(mask_buf.id, 0, &mask_bytes)?;

        // --- run all 36 layers, ping-pong acts, prefetch next layer's weights
        //     concurrently with the current submit ---
        let freqs_ref = freqs_buf.as_buf_ref();
        let mask_ref = mask_buf.as_buf_ref();
        let tap_layer = std::env::var("THINFER_QWEN3LM_TAP_LAYER")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        let hid = cfg.ffn_hidden as u32;
        let mut cur = x_buf;
        let mut pending = Some(handles.layers[0].acquire(residency, backend).await?);
        for idx in 0..handles.layers.len() {
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
            // Diagnostic per-op taps [n1, q, k, v] for one layer (k/v are the GQA
            // width n_kv_heads*head_dim, not the full hidden).
            let _ = hid;
            let seq_dim = seq_pad as u32 * dim;
            let seq_kv = seq_pad as u32 * cfg.kv_width() as u32;
            let tap_bufs = if tap_layer == Some(idx) {
                Some([
                    scratch.alloc(bp.act_bytes(seq_dim))?,
                    scratch.alloc(bp.act_bytes(seq_dim))?,
                    scratch.alloc(bp.act_bytes(seq_kv))?,
                    scratch.alloc(bp.act_bytes(seq_kv))?,
                ])
            } else {
                None
            };
            let tap_refs = tap_bufs.as_ref().map(|b| {
                [
                    b[0].as_buf_ref(),
                    b[1].as_buf_ref(),
                    b[2].as_buf_ref(),
                    b[3].as_buf_ref(),
                ]
            });
            lm_block_forward(
                &scope,
                pipelines,
                shape,
                cur_h,
                freqs_h,
                mask_h,
                nxt_h,
                &bufs,
                handles.v_quant[idx],
                handles.down_quant[idx],
                tap_refs.as_ref(),
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
            let (submit_res, next_res) = futures::join!(scope.submit_void(), next_acquire);
            submit_res?;
            pending = next_res?;

            if diag {
                let bytes = backend.read_buffer(nxt.id(), 0, act_bytes).await?;
                let v = seq::act_readback_to_f32(bp.act_dtype, &bytes, seq_pad * cfg.hidden);
                let nf = v.iter().filter(|x| !x.is_finite()).count();
                let amax = v
                    .iter()
                    .filter(|x| x.is_finite())
                    .fold(0.0f32, |a, &x| a.max(x.abs()));
                eprintln!(
                    "[qwen3lm-diag] layer {idx:02}: {nf}/{} non-finite, max|finite|={amax:.3e}, row0[..4]={:?}",
                    v.len(),
                    &v[..4]
                );
            }
            if let Some(tb) = &tap_bufs {
                let names = ["n1", "q", "k", "v"];
                let kv = cfg.kv_width();
                let counts = [
                    seq_pad * cfg.hidden,
                    seq_pad * cfg.hidden,
                    seq_pad * kv,
                    seq_pad * kv,
                ];
                for (j, b) in tb.iter().enumerate() {
                    let bytes = backend
                        .read_buffer(b.id(), 0, bp.act_bytes(counts[j] as u32))
                        .await?;
                    let v = seq::act_readback_to_f32(bp.act_dtype, &bytes, counts[j]);
                    let nf = v.iter().filter(|x| !x.is_finite()).count();
                    let amax = v
                        .iter()
                        .filter(|x| x.is_finite())
                        .fold(0.0f32, |a, &x| a.max(x.abs()));
                    eprintln!(
                        "[qwen3lm-tap] layer {idx} {}: {nf}/{} non-finite, max|finite|={amax:.3e}",
                        names[j],
                        v.len(),
                    );
                }
            }

            drop(views);
            cur = nxt;
        }

        // --- final norm + lm_head on the LAST real token only (row seq-1) ---
        // Slice that row out of the post-layer residual, RMSNorm it, then run the
        // Q6_K lm_head matmul (m=1) for a [1, VOCAB] logits row. The lm_head N is
        // VOCAB=151936, which overflows wgpu's 65535-per-dispatch-dim limit, so
        // the matmul is TILED over VOCAB (each tile a contiguous row-range subview
        // of the N-major quant weight). Inline Quant (no per-tile dequant
        // workspace), so all tiles + the final RMSNorm run in ONE submit.
        let cur_ref = cur.as_buf_ref();
        let logits = self
            .norm_and_lmhead(
                backend,
                pipelines,
                residency,
                scratch,
                handles,
                &cur_ref,
                seq as u32 - 1,
            )
            .await?;
        Ok(logits)
    }

    /// KV-cached autoregressive greedy generation. Prefills `prompt_ids` in
    /// chunks (each writing post-rope k / raw v into a persistent per-layer KV
    /// cache), picks the first token from the last prompt position, then decodes
    /// one token per step against the growing cache. Returns the GENERATED token
    /// ids only (prompt excluded); stops at `eos_id` or after `max_new` tokens.
    #[allow(clippy::too_many_arguments)]
    pub async fn generate<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipelines: &Qwen3LmPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        handles: &Qwen3LmHandles,
        source: &S,
        prompt_ids: &[u32],
        max_new: usize,
        eos_id: u32,
    ) -> Result<Vec<u32>, Qwen3ForwardError<S::Error>> {
        let bp = &pipelines.bp;
        let cfg = self.cfg;
        debug_assert_eq!(handles.layers.len(), cfg.n_layers);
        let plen = prompt_ids.len();
        assert!(plen > 0, "Qwen3LmGenerator::generate: empty prompt");
        assert!(max_new > 0, "Qwen3LmGenerator::generate: max_new == 0");

        let max_ctx = plen + max_new + 2; // +2 pad (odd-prompt prefill pad row)
        assert!(
            max_ctx <= self.rope.axes_lens[0],
            "generate ctx {max_ctx} exceeds rope max_seq {}",
            self.rope.axes_lens[0]
        );

        let dim = cfg.hidden as u32;
        let shape = block_shape(&cfg, 0);
        let cache = KvCache::alloc(scratch, bp, &cfg, max_ctx)?;
        let diag = std::env::var("THINFER_QWEN3LM_DIAG").is_ok();
        let t_prefill = diag.then(std::time::Instant::now);

        // Dummy 1-word mask for the has_mask==0 decode SDPA (never read).
        let dummy_mask = scratch.alloc(bp.act_bytes(2))?;
        let dummy_ref = dummy_mask.as_buf_ref();

        // --- PREFILL: chunk the prompt, keep the last chunk's residual so we can
        //     read the last real prompt row for the first generated token. ---
        const PREFILL_CHUNK: usize = 256; // even
        let mut last_resid: Option<(WsBuf<WgpuBackend>, u32)> = None; // (residual, last real row)
        let mut chunk_start = 0usize;
        while chunk_start < plen {
            let real = (plen - chunk_start).min(PREFILL_CHUNK);
            // Keep q_len even so the rectangular causal mask (s_k even) aligns;
            // only the final chunk can be odd (pad by repeating the last token,
            // whose KV slot is later overwritten by the first decode token).
            let padded = if real.is_multiple_of(2) {
                real
            } else {
                real + 1
            };
            let mut ids: Vec<u32> = prompt_ids[chunk_start..chunk_start + real].to_vec();
            if padded > real {
                ids.push(*ids.last().expect("non-empty chunk"));
            }
            let s_k = chunk_start + padded;

            let embeds = embed_lookup_hidden(source, &ids, cfg.hidden)
                .await
                .map_err(Qwen3ForwardError::Embed)?;
            let x_buf = scratch.alloc(bp.act_bytes(padded as u32 * dim))?;
            backend.write_buffer(x_buf.id(), 0, &seq::act_upload_bytes(bp.act_dtype, &embeds))?;

            let freqs_buf = self.freqs_for(backend, bp, scratch, chunk_start, padded)?;
            let mask_bytes = causal_rect_mask_bytes(chunk_start, padded, s_k, bp.act_dtype);
            let mask_buf = scratch.alloc(mask_bytes.len() as u64)?;
            backend.write_buffer(mask_buf.id(), 0, &mask_bytes)?;

            let resid = self
                .run_layers_kv(
                    backend,
                    pipelines,
                    residency,
                    scratch,
                    handles,
                    shape,
                    x_buf,
                    &freqs_buf.as_buf_ref(),
                    &mask_buf.as_buf_ref(),
                    1,
                    chunk_start as u32,
                    padded as u32,
                    s_k as u32,
                    &cache,
                )
                .await?;
            last_resid = Some((resid, real as u32 - 1));
            chunk_start += real;
        }

        if let Some(t) = t_prefill {
            let dt = t.elapsed().as_secs_f64();
            eprintln!(
                "[qwen3lm-diag] prefill {plen} tokens in {dt:.2}s ({:.0} tok/s)",
                plen as f64 / dt
            );
        }
        let t_decode = diag.then(std::time::Instant::now);
        let (resid, last_row) = last_resid.expect("prefill produced a residual");
        let mut current_len = plen; // real cached positions
        let mut generated: Vec<u32> = Vec::with_capacity(max_new);

        // --- first token from the last real prompt row ---
        let logits = self
            .norm_and_lmhead(
                backend,
                pipelines,
                residency,
                scratch,
                handles,
                &resid.as_buf_ref(),
                last_row,
            )
            .await?;
        let mut next = argmax(&logits);
        drop(resid);
        generated.push(next);
        if next == eos_id {
            generated.pop();
            return Ok(generated);
        }

        // --- DECODE loop: one token per step against the growing cache ---
        while generated.len() < max_new {
            let pos = current_len; // the token we are about to embed sits here
            let embeds = embed_lookup_hidden(source, &[next], cfg.hidden)
                .await
                .map_err(Qwen3ForwardError::Embed)?;
            let x_buf = scratch.alloc(bp.act_bytes(dim))?;
            backend.write_buffer(x_buf.id(), 0, &seq::act_upload_bytes(bp.act_dtype, &embeds))?;
            let freqs_buf = self.freqs_for(backend, bp, scratch, pos, 1)?;
            let s_k = pos + 1;
            let resid = self
                .run_layers_kv(
                    backend,
                    pipelines,
                    residency,
                    scratch,
                    handles,
                    shape,
                    x_buf,
                    &freqs_buf.as_buf_ref(),
                    &dummy_ref,
                    0,
                    pos as u32,
                    1,
                    s_k as u32,
                    &cache,
                )
                .await?;
            current_len += 1;
            let logits = self
                .norm_and_lmhead(
                    backend,
                    pipelines,
                    residency,
                    scratch,
                    handles,
                    &resid.as_buf_ref(),
                    0,
                )
                .await?;
            next = argmax(&logits);
            drop(resid);
            if next == eos_id {
                break;
            }
            generated.push(next);
        }
        if let Some(t) = t_decode {
            let dt = t.elapsed().as_secs_f64();
            let n = generated.len().max(1);
            eprintln!(
                "[qwen3lm-diag] decode {n} tokens in {dt:.2}s ({:.1} tok/s)",
                n as f64 / dt
            );
        }
        Ok(generated)
    }

    /// Upload RoPE freqs (act dtype) for `rows` positions on axis 0 starting at
    /// `start` (`start..start+rows`); axes 1/2 are no-op (`d=0`).
    fn freqs_for(
        &self,
        backend: &WgpuBackend,
        pipelines: &BlockPipelines,
        scratch: &Workspace<WgpuBackend>,
        start: usize,
        rows: usize,
    ) -> Result<WsBuf<WgpuBackend>, WgpuError> {
        let mut pos_ids = vec![0_i32; rows * 3];
        for (i, chunk) in pos_ids.chunks_exact_mut(3).enumerate() {
            chunk[0] = (start + i) as i32;
        }
        let bytes = seq::freqs_upload_bytes(pipelines.act_dtype, &self.rope.lookup(&pos_ids));
        let buf = scratch.alloc(bytes.len() as u64)?;
        backend.write_buffer(buf.id(), 0, &bytes)?;
        Ok(buf)
    }

    /// Run all 36 KV-aware decoder layers over `rows` new tokens at global
    /// positions `[past_len, past_len+rows)`, writing kv into `cache`. Returns
    /// the post-layer residual `[rows, HIDDEN]`.
    ///
    /// PERF: with the inline-Quant matmul there is no per-layer bf16 dequant
    /// workspace, so multiple layers batch into ONE submit (ONE CPU<->GPU sync
    /// each instead of ~42). Decode (`rows == 1`) runs all 36 layers in a single
    /// submit; prefill (`rows` large) sub-batches `LAYER_BATCH_ROWS / rows`
    /// layers per submit to bound the in-scope activation working set. Weights are
    /// resident (acquired up front, all cache hits) and stay pinned for the run.
    /// The residual ping-pongs across two persistent buffers (A/B); wgpu inserts
    /// the intra- and inter-scope barriers for the KV write->read and cross-layer
    /// residual dependencies (validated by the parity gate).
    #[allow(clippy::too_many_arguments)]
    async fn run_layers_kv<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipelines: &Qwen3LmPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        handles: &Qwen3LmHandles,
        shape: Qwen3BlockShape,
        x_buf: WsBuf<WgpuBackend>,
        freqs_ref: &BufRef,
        mask_ref: &BufRef,
        has_mask: u32,
        past_len: u32,
        rows: u32,
        s_k: u32,
        cache: &KvCache,
    ) -> Result<WsBuf<WgpuBackend>, Qwen3ForwardError<S::Error>> {
        let bp = &pipelines.bp;
        let dim = self.cfg.hidden as u32;
        let act_bytes = bp.act_bytes(rows * dim);
        let n_layers = handles.layers.len();

        // Layers per submit. Two hard constraints on this 8GB card:
        //  * VRAM: the 36 layer weights (~5.85GB) + wgpu/context overhead do NOT
        //    all fit resident, so we must NOT pin all 36 at once (that blocks the
        //    residency arbiter from evicting -> OOM). We pin ONE GROUP at a time
        //    and drop it after its submit so the arbiter streams weights.
        //  * TDR: a submit that keeps the GPU busy > ~2s is killed (device lost).
        //    A PREFILL layer (rows=256) is heavy, so prefill runs ONE layer per
        //    submit (matches the original, proven TDR-safe). DECODE layers
        //    (rows=1) are tiny, so several batch per submit.
        // NB: batching decode layers was NOT the decode win (measured identical at
        // group 1/6/12/18 - decode is per-op GPU-bound, see DECODE_MM_CFG); the
        // grouping only bounds submit count and stays well clear of the OOM/TDR
        // edges. Override the decode group via THINFER_QWEN3LM_GROUP.
        let group = if rows == 1 {
            std::env::var("THINFER_QWEN3LM_GROUP")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(6)
                .clamp(1, n_layers)
        } else {
            1
        };

        // Persistent A/B residual ping-pong (input x -> A -> B -> A ...).
        let buf_a = scratch.alloc(act_bytes)?;
        let buf_b = scratch.alloc(act_bytes)?;
        let a_ref = buf_a.as_buf_ref();
        let b_ref = buf_b.as_buf_ref();
        let x_ref = x_buf.as_buf_ref();

        let mut idx = 0usize;
        while idx < n_layers {
            let end = (idx + group).min(n_layers);
            // Pin ONLY this group's weights; dropped after the submit below.
            let mut views = Vec::with_capacity(end - idx);
            for h in &handles.layers[idx..end] {
                views.push(h.acquire(residency, backend).await?);
            }
            let group_bufs: Vec<Qwen3BlockBufs> = views.iter().map(|v| v.bufs()).collect();

            let scope = scratch.batch();
            for (off, layer_bufs) in group_bufs.iter().enumerate() {
                let l = idx + off;
                // Layer l reads {x (l==0), A (l odd), B (l>=2 even)}, writes A
                // (l even) / B (l odd).
                let cur_ref = if l == 0 {
                    &x_ref
                } else if l.is_multiple_of(2) {
                    &b_ref
                } else {
                    &a_ref
                };
                let out_ref = if l.is_multiple_of(2) { &a_ref } else { &b_ref };
                let cur_h = scope.import(cur_ref);
                let out_h = scope.import(out_ref);
                let freqs_h = scope.import(freqs_ref);
                let mask_h = scope.import(mask_ref);
                let k_h = scope.import(&cache.k_refs[l]);
                let v_h = scope.import(&cache.v_refs[l]);
                lm_block_forward_kv(
                    &scope,
                    pipelines,
                    shape,
                    cur_h,
                    freqs_h,
                    mask_h,
                    out_h,
                    layer_bufs,
                    k_h,
                    v_h,
                    past_len,
                    rows,
                    s_k,
                    has_mask,
                    handles.v_quant[l],
                    handles.down_quant[l],
                )?;
            }
            scope.submit_void().await?;
            drop(views);
            idx = end;
        }

        // Final residual sits in A (last layer index even) or B (odd).
        Ok(if (n_layers - 1).is_multiple_of(2) {
            buf_a
        } else {
            buf_b
        })
    }

    /// Final RMSNorm of `resid` row `row` + tiled inline-Quant lm_head -> VOCAB
    /// logits (host `Vec<f32>`). The lm_head N = VOCAB overflows the 65535
    /// dispatch-dim limit, so the matmul is tiled over VOCAB; inline Quant means
    /// no per-tile dequant workspace, so the RMSNorm + all tiles run in ONE submit.
    #[allow(clippy::too_many_arguments)]
    async fn norm_and_lmhead<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipelines: &Qwen3LmPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        handles: &Qwen3LmHandles,
        resid_ref: &BufRef,
        row: u32,
    ) -> Result<Vec<f32>, Qwen3ForwardError<S::Error>> {
        let bp = &pipelines.bp;
        let cfg = self.cfg;
        let dim = cfg.hidden as u32;
        let vocab = cfg.vocab as u32;
        let fnorm_view = residency.acquire(handles.final_norm, backend).await?;
        let lmhead_view = residency.acquire(handles.lm_head, backend).await?;
        let fnorm_ref = fnorm_view.buf();
        let lmhead_ref = lmhead_view.buf();
        let logits_buf = scratch.alloc(bp.act_bytes(vocab))?;
        let logits_ref = logits_buf.as_buf_ref();
        let normed = scratch.alloc(bp.act_bytes(dim))?;
        let normed_ref = normed.as_buf_ref();

        // Q6_K row = HIDDEN elems = a whole number of blocks, so a row-range is a
        // contiguous byte range. Tile is a multiple of 32768 so byte + logits
        // offsets stay 256-aligned.
        let row_bytes = handles.lm_head_quant.bytes_for_elements(cfg.hidden as u64);
        // lm_head is always M=1 (single scored row) -> inline decode tile.
        let (lm_mm, lm_dq, lm_op) = pipelines.matmul_for(handles.lm_head_quant, 1);
        const LM_TILE: u32 = 32768; // <= 65535 dispatch-dim limit, offset-aligned

        {
            let scope = scratch.batch();
            // Final RMSNorm over the target row.
            let cur_h = scope.import(resid_ref);
            let fnorm_h = scope.import(&fnorm_ref);
            let normed_h = scope.import(&normed_ref);
            let row_off = bp.act_bytes(row * dim);
            let last_h = scope.subview(&cur_h, row_off, bp.act_bytes(dim));
            op_rmsnorm(
                &scope,
                bp,
                ActBuf::dense(last_h),
                fnorm_h,
                ActBuf::dense(normed_h),
                1,
                dim,
                cfg.rms_norm_eps,
            )?;
            // All VOCAB tiles, same scope.
            let mut c0 = 0u32;
            while c0 < vocab {
                let cn = (vocab - c0).min(LM_TILE);
                let normed_in = scope.import(&normed_ref);
                let lmhead_h = scope.import(&lmhead_ref);
                let logits_h = scope.import(&logits_ref);
                let w_sub = scope.subview(&lmhead_h, c0 as u64 * row_bytes, cn as u64 * row_bytes);
                let out_sub = scope.subview(&logits_h, bp.act_bytes(c0), bp.act_bytes(cn));
                let dims_lm = scope.u32x4_uniform(1, cn, dim, 0)?;
                Block::dispatch_matmul_site(
                    &scope,
                    bp,
                    ActBuf::dense(normed_in),
                    w_sub,
                    out_sub,
                    dims_lm,
                    None,
                    None,
                    lm_dq,
                    lm_mm,
                    lm_op,
                    1,
                    cn,
                    dim,
                )?;
                c0 += cn;
            }
            scope.submit_void().await?;
        }

        let bytes = backend
            .read_buffer(logits_buf.id(), 0, bp.act_bytes(vocab))
            .await?;
        Ok(seq::act_readback_to_f32(bp.act_dtype, &bytes, cfg.vocab))
    }
}

/// Greedy argmax over a logits row.
fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .expect("non-empty logits")
}
