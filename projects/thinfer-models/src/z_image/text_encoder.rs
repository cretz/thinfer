//! Qwen3-4B text encoder. Z-Image-Turbo's prompt conditioner: the upstream
//! pipeline calls a `Qwen3ForCausalLM` with `output_hidden_states=True` and
//! takes `hidden_states[-2]` (the layer-N-1 output, before the final RMSNorm).
//! No LM head is needed; tied word embeddings are similarly irrelevant.
//!
//! Source: HuggingFace `transformers/models/qwen3/modeling_qwen3.py`, config
//! at `Tongyi-MAI/Z-Image-Turbo/text_encoder/config.json` (snapshotted into the
//! `config` module below).
//!
//! Weight names follow upstream HF Qwen3 convention (`model.layers.{i}.…`).
//! Upstream `Tongyi-MAI/Z-Image-Turbo` ships the checkpoint as a 3-shard
//! safetensors split (pinned in `manifest.rs`); loading from those shards
//! requires multi-file `SafetensorsSource` support, which is the next infra
//! prerequisite to actually invoking `register_qwen3_handles`.
//!
//! Forward path (next chunk): standard pre-norm transformer with GQA (32 Q / 8
//! KV heads, head_dim=128), per-head Q/K RMSNorm before RoPE, SwiGLU FFN. RoPE
//! is 1-axis (token positions), theta=1e6. Reuses the same residency manager,
//! RMSNorm / RoPE / SDPA / Linear / SwiGLU kernels as the DiT block - the only
//! new ingredients are token embedding lookup and causal mask construction.

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError};
use thinfer_core::residency::{
    GpuView, ResidencyError, TransposePolicy, WeightHandle, WeightMeta, WeightResidency,
};
use thinfer_core::tensor::StorageEncoding;
use thinfer_core::trace::{self, PHASE};
use thinfer_core::weight::{Decoder, WeightCatalog, WeightId, WeightReader, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace};
use tracing::Instrument;

use crate::z_image::block::{
    ActBuf, Block, BlockPipelines, alloc_act, alloc_matmul_out_buf, copy_tap, op_add, op_rmsnorm,
    op_rope_halfrot, op_sdpa, op_silu_mul,
};
use crate::z_image::rope_embedder::RopeEmbedder;
use crate::z_image::seq;

/// Audited against `Tongyi-MAI/Z-Image-Turbo/text_encoder/config.json`
/// (snapshot `f332072aa78be7aecdf3ee76d5c247082da564a6`, 2026-05-12).
pub mod config {
    pub const HIDDEN: usize = 2560;
    pub const N_LAYERS: usize = 36;
    pub const N_HEADS: usize = 32;
    pub const N_KV_HEADS: usize = 8;
    pub const HEAD_DIM: usize = 128;
    pub const FFN_HIDDEN: usize = 9728;
    pub const VOCAB: usize = 151936;
    pub const RMS_NORM_EPS: f32 = 1e-6;
    pub const ROPE_THETA: f32 = 1_000_000.0;
    pub const MAX_POSITION: usize = 40960;
    /// Upstream pipeline reads `hidden_states[-2]`. HF tuple layout when
    /// `output_hidden_states=True` is `(embed, layer_0_out, …, layer_{N-1}_out,
    /// final_norm_out)`; `[-2]` is the layer-{N-1} output before the final
    /// RMSNorm. We never run the final norm or the LM head.
    pub const HIDDEN_STATES_LAYER: usize = N_LAYERS - 1;
}

/// Resolved weight names for one Qwen3 decoder layer.
#[derive(Clone, Debug)]
pub struct Qwen3BlockWeights {
    pub input_layernorm: WeightId,
    pub post_attention_layernorm: WeightId,
    pub q_proj: WeightId,
    pub k_proj: WeightId,
    pub v_proj: WeightId,
    pub o_proj: WeightId,
    /// Qwen3-specific: per-head RMSNorm over `HEAD_DIM` applied to Q after
    /// projection, before RoPE.
    pub q_norm: WeightId,
    /// Per-head RMSNorm over `HEAD_DIM` on K, before RoPE.
    pub k_norm: WeightId,
    pub mlp_gate: WeightId,
    pub mlp_up: WeightId,
    pub mlp_down: WeightId,
}

impl Qwen3BlockWeights {
    pub fn new(idx: usize) -> Self {
        let p = format!("model.layers.{idx}");
        let id = |s: &str| WeightId(format!("{p}.{s}"));
        Self {
            input_layernorm: id("input_layernorm.weight"),
            post_attention_layernorm: id("post_attention_layernorm.weight"),
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

/// Module-level (non-block) weights. We stop at `hidden_states[-2]` so the
/// final `model.norm` and `lm_head` are intentionally absent.
#[derive(Clone, Debug)]
pub struct Qwen3Weights {
    pub embed_tokens: WeightId,
    pub layers: Vec<Qwen3BlockWeights>,
}

impl Qwen3Weights {
    pub fn new() -> Self {
        Self {
            embed_tokens: WeightId("model.embed_tokens.weight".into()),
            layers: (0..config::N_LAYERS).map(Qwen3BlockWeights::new).collect(),
        }
    }
}

impl Default for Qwen3Weights {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Qwen3BlockHandles {
    pub input_layernorm: WeightHandle,
    pub post_attention_layernorm: WeightHandle,
    pub q_proj: WeightHandle,
    pub k_proj: WeightHandle,
    pub v_proj: WeightHandle,
    pub o_proj: WeightHandle,
    pub q_norm: WeightHandle,
    pub k_norm: WeightHandle,
    pub mlp_gate: WeightHandle,
    pub mlp_up: WeightHandle,
    pub mlp_down: WeightHandle,
}

#[derive(Clone, Debug)]
pub struct Qwen3Handles {
    pub layers: Vec<Qwen3BlockHandles>,
}

#[derive(Debug)]
pub enum EmbedLookupError {
    MissingTable,
    BadShape(Vec<usize>),
    Undecodable(StorageEncoding),
    TokenOutOfRange { id: u32, vocab: usize },
    Source(String),
    Reader(String),
    Decode(thinfer_core::weight::DecodeError),
}

/// CPU-side gather over `model.embed_tokens.weight`. Reads only the rows the
/// prompt needs (a few hundred at most), bf16 -> fp32 decoded per row. The
/// full table is `[VOCAB, HIDDEN]` ~ 777 MB at bf16 - never paged into RAM as a
/// whole, never uploaded to GPU. Output is row-major `[token_ids.len(), HIDDEN]`
/// fp32.
pub async fn embed_lookup<S: WeightSource>(
    source: &S,
    token_ids: &[u32],
) -> Result<Vec<f32>, EmbedLookupError> {
    let id = WeightId("model.embed_tokens.weight".into());
    let entry = source
        .catalog()
        .get(&id)
        .ok_or(EmbedLookupError::MissingTable)?;
    let shape = &entry.shape.0;
    if shape.len() != 2 || shape[1] != config::HIDDEN {
        return Err(EmbedLookupError::BadShape(shape.clone()));
    }
    let vocab = shape[0];
    let encoding = entry
        .encoding
        .ok_or(EmbedLookupError::Undecodable(StorageEncoding::F32))?;
    let bytes_per_elt: u64 = match encoding {
        StorageEncoding::Bf16 => 2,
        StorageEncoding::F32 => 4,
        enc => return Err(EmbedLookupError::Undecodable(enc)),
    };
    let row_src_bytes = (config::HIDDEN as u64) * bytes_per_elt;

    let mut reader = source
        .open(&id)
        .await
        .map_err(|e| EmbedLookupError::Source(format!("{e:?}")))?;
    let mut out = vec![0f32; token_ids.len() * config::HIDDEN];
    let mut row_src = vec![0u8; row_src_bytes as usize];
    let row_dst_bytes = config::HIDDEN * 4;
    for (i, &tok) in token_ids.iter().enumerate() {
        if (tok as usize) >= vocab {
            return Err(EmbedLookupError::TokenOutOfRange { id: tok, vocab });
        }
        let off = (tok as u64) * row_src_bytes;
        reader
            .read_at(off, &mut row_src)
            .await
            .map_err(|e| EmbedLookupError::Reader(format!("{e:?}")))?;
        let dst_byte_off = i * row_dst_bytes;
        // Safety: cast &mut [f32] -> &mut [u8] for byte-level decode write.
        let out_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut out[..]);
        let mut decoder = Decoder::new(encoding).map_err(EmbedLookupError::Decode)?;
        let n = decoder
            .feed(
                &row_src,
                &mut out_bytes[dst_byte_off..dst_byte_off + row_dst_bytes],
            )
            .map_err(EmbedLookupError::Decode)?;
        decoder.finish().map_err(EmbedLookupError::Decode)?;
        debug_assert_eq!(n, row_dst_bytes);
    }
    Ok(out)
}

#[derive(Debug)]
pub enum LoadError {
    UnknownWeight(WeightId),
    /// On-disk encoding can't be decoded to fp32 (quantized, or fp16). Qwen3
    /// from `Tongyi-MAI` ships bf16, so this should never fire in practice.
    Undecodable {
        id: WeightId,
        encoding: Option<StorageEncoding>,
        label: String,
    },
}

/// Register the encoder weights. Only the first `N_LAYERS - 1` layers are
/// registered: the output is `hidden_states[-2]` so layer `N-1` never runs
/// (no point paying its residency footprint or last-iteration prefetch).
///
/// `transcode`: load-time requantize target for 6 of the 7 matmul weights
/// per layer (q/k/v/o + gate/up). The Tongyi checkpoint ships them bf16;
/// `Some(Q8_0)` rides them onto the same quant matmul path the DiT uses
/// (DP4A `matmul_i8` or the dequant workspace fallback) at ~half the upload
/// bytes. Norm gains stay dense.
///
/// `mlp_down` is NEVER transcoded: Qwen3 massive activations (the
/// attention-sink token's gate*up row, max-abs ~8k from layer 6 on, with
/// heavy cancellation in the dot products) amplify per-block weight-quant
/// noise into ~1.7 absolute output error on that row, corrupting the
/// conditioning (qwen3_parity layer-6 forensics, 2026-06-05). llama.cpp's
/// K-quant mixes bump ffn_down precision for the same reason. Callers'
/// pipeline cfgs must compile `matmul_ffn_down` against Bf16 to match
/// (`ZImageModel::load` and qwen3_parity mirror this).
pub fn register_qwen3_handles<S: WeightSource>(
    residency: &WeightResidency<S>,
    transcode: Option<thinfer_core::quant::QuantKind>,
) -> Result<Qwen3Handles, LoadError> {
    let weights = Qwen3Weights::new();
    // `model.embed_tokens.weight` is the 777 MB [vocab, hidden] lookup table.
    // Intentionally NOT registered with residency: a prompt only ever indexes
    // a few hundred rows, so we gather rows directly from disk via
    // `embed_lookup` and never page or upload the full table.
    let n_run = config::HIDDEN_STATES_LAYER;
    let mut layers = Vec::with_capacity(n_run);
    for b in weights.layers.iter().take(n_run) {
        layers.push(Qwen3BlockHandles {
            input_layernorm: register_one(
                residency,
                &b.input_layernorm,
                TransposePolicy::None,
                None,
            )?,
            post_attention_layernorm: register_one(
                residency,
                &b.post_attention_layernorm,
                TransposePolicy::None,
                None,
            )?,
            q_proj: register_one(residency, &b.q_proj, TransposePolicy::Linear2D, transcode)?,
            k_proj: register_one(residency, &b.k_proj, TransposePolicy::Linear2D, transcode)?,
            v_proj: register_one(residency, &b.v_proj, TransposePolicy::Linear2D, transcode)?,
            o_proj: register_one(residency, &b.o_proj, TransposePolicy::Linear2D, transcode)?,
            q_norm: register_one(residency, &b.q_norm, TransposePolicy::None, None)?,
            k_norm: register_one(residency, &b.k_norm, TransposePolicy::None, None)?,
            mlp_gate: register_one(residency, &b.mlp_gate, TransposePolicy::Linear2D, transcode)?,
            mlp_up: register_one(residency, &b.mlp_up, TransposePolicy::Linear2D, transcode)?,
            // Never transcoded: see the massive-activation note on this fn.
            mlp_down: register_one(residency, &b.mlp_down, TransposePolicy::Linear2D, None)?,
        });
    }
    Ok(Qwen3Handles { layers })
}

fn register_one<S: WeightSource>(
    residency: &WeightResidency<S>,
    id: &WeightId,
    transpose: TransposePolicy,
    transcode: Option<thinfer_core::quant::QuantKind>,
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
    if thinfer_core::weight::Decoder::new(encoding).is_err() {
        return Err(LoadError::Undecodable {
            id: id.clone(),
            encoding: Some(encoding),
            label: entry.encoding_label.clone(),
        });
    }
    // Transcode targets keep the file's [N, K] row order (GGUF block layout
    // is N-major, no transpose); requires bf16 source and whole 32-blocks
    // along K. Mirrors `loader::register_linear_transcode`.
    let (transpose, transcode) = match (encoding, transcode) {
        (StorageEncoding::Bf16, Some(k)) => {
            assert_eq!(entry.shape.0.len(), 2, "transcode target must be 2-D");
            assert_eq!(
                entry.shape.0[1] % 32,
                0,
                "transcode requires K % 32 == 0 ({id:?})"
            );
            (TransposePolicy::None, Some(k))
        }
        (_, Some(_)) => {
            return Err(LoadError::Undecodable {
                id: id.clone(),
                encoding: Some(encoding),
                label: format!("transcode requires bf16 source ({})", entry.encoding_label),
            });
        }
        (_, None) => (transpose, None),
    };
    Ok(residency.register(WeightMeta {
        id: id.clone(),
        shape: entry.shape.clone(),
        encoding,
        on_disk_bytes: entry.size,
        transpose,
        transcode,
    }))
}

/// One expected weight entry: name + on-disk shape (PyTorch `[out, in]` for
/// Linear weights, `[N]` for RMSNorm gains, `[vocab, hidden]` for embedding).
#[derive(Clone, Debug)]
pub struct Expected {
    pub id: WeightId,
    pub shape: Vec<usize>,
}

/// Build the expected weight list for the Qwen3-4B text encoder. Shape source:
/// HF `Qwen3ForCausalLM` modeling code + the snapshotted `config.json` above.
pub fn expected_weights() -> Vec<Expected> {
    use config::*;
    let mut out = Vec::with_capacity(1 + N_LAYERS * 11);
    let push = |out: &mut Vec<Expected>, name: String, shape: Vec<usize>| {
        out.push(Expected {
            id: WeightId(name),
            shape,
        });
    };
    push(
        &mut out,
        "model.embed_tokens.weight".into(),
        vec![VOCAB, HIDDEN],
    );
    for i in 0..N_LAYERS {
        let p = format!("model.layers.{i}");
        push(
            &mut out,
            format!("{p}.input_layernorm.weight"),
            vec![HIDDEN],
        );
        push(
            &mut out,
            format!("{p}.post_attention_layernorm.weight"),
            vec![HIDDEN],
        );
        push(
            &mut out,
            format!("{p}.self_attn.q_proj.weight"),
            vec![N_HEADS * HEAD_DIM, HIDDEN],
        );
        push(
            &mut out,
            format!("{p}.self_attn.k_proj.weight"),
            vec![N_KV_HEADS * HEAD_DIM, HIDDEN],
        );
        push(
            &mut out,
            format!("{p}.self_attn.v_proj.weight"),
            vec![N_KV_HEADS * HEAD_DIM, HIDDEN],
        );
        push(
            &mut out,
            format!("{p}.self_attn.o_proj.weight"),
            vec![HIDDEN, N_HEADS * HEAD_DIM],
        );
        push(
            &mut out,
            format!("{p}.self_attn.q_norm.weight"),
            vec![HEAD_DIM],
        );
        push(
            &mut out,
            format!("{p}.self_attn.k_norm.weight"),
            vec![HEAD_DIM],
        );
        push(
            &mut out,
            format!("{p}.mlp.gate_proj.weight"),
            vec![FFN_HIDDEN, HIDDEN],
        );
        push(
            &mut out,
            format!("{p}.mlp.up_proj.weight"),
            vec![FFN_HIDDEN, HIDDEN],
        );
        push(
            &mut out,
            format!("{p}.mlp.down_proj.weight"),
            vec![HIDDEN, FFN_HIDDEN],
        );
    }
    out
}

/// Lightweight catalog audit. Returns missing names + shape mismatches against
/// the on-disk layout. Extras are ignored: the upstream Qwen3 checkpoint also
/// carries `model.norm.weight` and (untied) `lm_head.weight`, which we do not
/// use for the text encoder path.
pub fn audit(catalog: &WeightCatalog) -> AuditReport {
    let expected = expected_weights();
    let mut report = AuditReport {
        expected: expected.len(),
        ..Default::default()
    };
    for e in &expected {
        match catalog.get(&e.id) {
            None => report.missing.push(e.id.clone()),
            Some(entry) if entry.shape.0 != e.shape => {
                report.shape_mismatches.push(ShapeMismatch {
                    id: e.id.clone(),
                    expected: e.shape.clone(),
                    got: entry.shape.0.clone(),
                });
            }
            _ => {}
        }
    }
    report
}

#[derive(Clone, Debug, Default)]
pub struct AuditReport {
    pub expected: usize,
    pub missing: Vec<WeightId>,
    pub shape_mismatches: Vec<ShapeMismatch>,
}

impl AuditReport {
    pub fn ok(&self) -> bool {
        self.missing.is_empty() && self.shape_mismatches.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct ShapeMismatch {
    pub id: WeightId,
    pub expected: Vec<usize>,
    pub got: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Per-layer forward
// ---------------------------------------------------------------------------

/// Per-block buffer refs (materialized from `Qwen3BlockViews::bufs()`).
#[derive(Clone, Copy, Debug)]
pub struct Qwen3BlockBufs {
    pub input_layernorm: BufRef,
    pub post_attention_layernorm: BufRef,
    pub q_proj: BufRef,
    pub k_proj: BufRef,
    pub v_proj: BufRef,
    pub o_proj: BufRef,
    pub q_norm: BufRef,
    pub k_norm: BufRef,
    pub mlp_gate: BufRef,
    pub mlp_up: BufRef,
    pub mlp_down: BufRef,
}

pub struct Qwen3BlockViews<'a> {
    pub input_layernorm: GpuView<'a>,
    pub post_attention_layernorm: GpuView<'a>,
    pub q_proj: GpuView<'a>,
    pub k_proj: GpuView<'a>,
    pub v_proj: GpuView<'a>,
    pub o_proj: GpuView<'a>,
    pub q_norm: GpuView<'a>,
    pub k_norm: GpuView<'a>,
    pub mlp_gate: GpuView<'a>,
    pub mlp_up: GpuView<'a>,
    pub mlp_down: GpuView<'a>,
}

impl Qwen3BlockHandles {
    pub async fn acquire<'a, S: WeightSource>(
        &self,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<Qwen3BlockViews<'a>, ResidencyError<S::Error, WgpuError>> {
        Ok(Qwen3BlockViews {
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

impl Qwen3BlockViews<'_> {
    pub fn bufs(&self) -> Qwen3BlockBufs {
        Qwen3BlockBufs {
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

#[derive(Clone, Copy, Debug)]
pub struct Qwen3BlockShape {
    pub dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_hidden: usize,
    /// Padded prompt length (B=1).
    pub seq: usize,
    pub norm_eps: f32,
}

impl Qwen3BlockShape {
    pub fn default_from_config(seq: usize) -> Self {
        Self {
            dim: config::HIDDEN,
            n_heads: config::N_HEADS,
            n_kv_heads: config::N_KV_HEADS,
            head_dim: config::HEAD_DIM,
            ffn_hidden: config::FFN_HIDDEN,
            seq,
            norm_eps: config::RMS_NORM_EPS,
        }
    }

    fn sdpa_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

/// GPU tap destinations for one Qwen3 layer (parity diagnostics). Each
/// `Some` BufRef receives an in-scope copy of the corresponding dense
/// intermediate (act-dtype bytes, caller sizes and reads back post-submit).
#[derive(Default, Clone)]
pub struct Qwen3BlockTaps {
    /// Pre-attn rmsnorm out `[rows, dim]`.
    pub n1: Option<BufRef>,
    /// q/k/v projection outputs `[rows, hq*hd]` / `[rows, hkv*hd]` x2.
    pub q: Option<BufRef>,
    pub k: Option<BufRef>,
    pub v: Option<BufRef>,
    /// Per-head q/k rmsnorm outputs (same shapes as q/k).
    pub qn: Option<BufRef>,
    pub kn: Option<BufRef>,
    /// Post-rope q/k (same shapes).
    pub qr: Option<BufRef>,
    pub kr: Option<BufRef>,
    /// Sdpa out `[rows, hq*hd]`.
    pub sa: Option<BufRef>,
    /// o_proj out `[rows, dim]`.
    pub proj: Option<BufRef>,
    /// Post-attn residual `[rows, dim]`.
    pub after_attn: Option<BufRef>,
    /// Pre-ffn rmsnorm out `[rows, dim]`.
    pub n2: Option<BufRef>,
    /// gate/up matmul outs `[rows, hid]` and their silu_mul `[rows, hid]`.
    pub gate: Option<BufRef>,
    pub up: Option<BufRef>,
    pub gu: Option<BufRef>,
    /// down matmul out `[rows, dim]`.
    pub down: Option<BufRef>,
}

/// One Qwen3 decoder layer: pre-norm GQA self-attn (with per-head Q/K
/// RMSNorm, 1-axis RoPE, causal SDPA) -> residual -> pre-norm SwiGLU FFN ->
/// residual. No adaLN, no double-norm. Reuses the same kernels as the DiT
/// `Block` via `BlockPipelines`.
pub struct Qwen3Block {
    pub shape: Qwen3BlockShape,
}

impl Qwen3Block {
    pub fn new(shape: Qwen3BlockShape) -> Self {
        // Qwen3 decouples head_dim from hidden: n_heads*head_dim (4096) may
        // exceed hidden (2560). Q/O projections cross between the two; K/V
        // stay in n_kv_heads*head_dim. The forward path uses `hq*hd` / `hkv*hd`
        // / `dim` independently so they don't need to match.
        Self { shape }
    }

    /// Append one decoder layer's dispatches to the scope's encoder. Caller
    /// submits the scope. Runs at the pipeline set's act dtype; every matmul
    /// routes through `Block::dispatch_matmul_site` (DP4A `matmul_i8` on the
    /// quant path, dequant-workspace or dense fallback otherwise) and sdpa
    /// picks the subgroup flash kernel when available (`op_sdpa`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        x_in: BatchBuf<'wsp>,
        freqs_in: BatchBuf<'wsp>,
        mask_in: BatchBuf<'wsp>,
        y_out: BatchBuf<'wsp>,
        bufs: &'wsp Qwen3BlockBufs,
    ) -> Result<(), WgpuError> {
        self.forward_taps(
            scope,
            pipelines,
            x_in,
            freqs_in,
            mask_in,
            y_out,
            bufs,
            &Qwen3BlockTaps::default(),
        )
    }

    /// `forward` with per-op tap copies (see [`Qwen3BlockTaps`]).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_taps<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        x_in: BatchBuf<'wsp>,
        freqs_in: BatchBuf<'wsp>,
        mask_in: BatchBuf<'wsp>,
        y_out: BatchBuf<'wsp>,
        bufs: &'wsp Qwen3BlockBufs,
        taps: &Qwen3BlockTaps,
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
        let qn_w = scope.import(&bufs.q_norm);
        let kn_w = scope.import(&bufs.k_norm);
        let o_w = scope.import(&bufs.o_proj);
        let pa_ln = scope.import(&bufs.post_attention_layernorm);
        let g_w = scope.import(&bufs.mlp_gate);
        let up_w = scope.import(&bufs.mlp_up);
        let down_w = scope.import(&bufs.mlp_down);

        // --- pre-attn norm ---
        let n1 = alloc_act(scope, pipelines, rows, dim)?;
        op_rmsnorm(scope, pipelines, x_in, in_ln, n1, rows, dim, eps)?;
        copy_tap(
            scope,
            n1.data,
            taps.n1.as_ref(),
            pipelines.act_bytes(rows * dim),
        )?;

        // --- q/k/v projections (separate weights; GQA shapes) ---
        let q_scratch = alloc_matmul_out_buf(scope, pipelines, rows * hq * hd)?;
        let dims_q = scope.u32x4_uniform(rows, hq * hd, dim, 0)?;
        Block::dispatch_matmul_site(
            scope,
            pipelines,
            n1,
            q_w,
            q_scratch,
            dims_q,
            pipelines.matmul_i8_qkv.as_ref(),
            pipelines.dequant_i8_qkv.as_ref(),
            pipelines.dequant_qkv.as_ref(),
            &pipelines.matmul_qkv,
            &pipelines.matmuls.qkv,
            rows,
            hq * hd,
            dim,
        )?;
        let k_scratch = alloc_matmul_out_buf(scope, pipelines, rows * hkv * hd)?;
        let dims_k = scope.u32x4_uniform(rows, hkv * hd, dim, 0)?;
        Block::dispatch_matmul_site(
            scope,
            pipelines,
            n1,
            k_w,
            k_scratch,
            dims_k,
            pipelines.matmul_i8_qkv.as_ref(),
            pipelines.dequant_i8_qkv.as_ref(),
            pipelines.dequant_qkv.as_ref(),
            &pipelines.matmul_qkv,
            &pipelines.matmuls.qkv,
            rows,
            hkv * hd,
            dim,
        )?;
        let v_scratch = alloc_matmul_out_buf(scope, pipelines, rows * hkv * hd)?;
        let dims_v = scope.u32x4_uniform(rows, hkv * hd, dim, 0)?;
        Block::dispatch_matmul_site(
            scope,
            pipelines,
            n1,
            v_w,
            v_scratch,
            dims_v,
            pipelines.matmul_i8_qkv.as_ref(),
            pipelines.dequant_i8_qkv.as_ref(),
            pipelines.dequant_qkv.as_ref(),
            &pipelines.matmul_qkv,
            &pipelines.matmuls.qkv,
            rows,
            hkv * hd,
            dim,
        )?;
        let q = ActBuf::dense(q_scratch);
        let k = ActBuf::dense(k_scratch);
        let v = ActBuf::dense(v_scratch);
        copy_tap(
            scope,
            q.data,
            taps.q.as_ref(),
            pipelines.act_bytes(rows * hq * hd),
        )?;
        copy_tap(
            scope,
            k.data,
            taps.k.as_ref(),
            pipelines.act_bytes(rows * hkv * hd),
        )?;
        copy_tap(
            scope,
            v.data,
            taps.v.as_ref(),
            pipelines.act_bytes(rows * hkv * hd),
        )?;

        // --- per-head Q/K RMSNorm over head_dim ---
        let qn = alloc_act(scope, pipelines, rows * hq, hd)?;
        op_rmsnorm(scope, pipelines, q, qn_w, qn, rows * hq, hd, eps)?;
        let kn = alloc_act(scope, pipelines, rows * hkv, hd)?;
        op_rmsnorm(scope, pipelines, k, kn_w, kn, rows * hkv, hd, eps)?;
        copy_tap(
            scope,
            qn.data,
            taps.qn.as_ref(),
            pipelines.act_bytes(rows * hq * hd),
        )?;
        copy_tap(
            scope,
            kn.data,
            taps.kn.as_ref(),
            pipelines.act_bytes(rows * hkv * hd),
        )?;

        // --- rope on Q, K (1-axis token-position freqs broadcast across heads) ---
        let qr = alloc_act(scope, pipelines, rows, hq * hd)?;
        op_rope_halfrot(scope, pipelines, qn, freqs_in, qr, rows, hq, hd)?;
        let kr = alloc_act(scope, pipelines, rows, hkv * hd)?;
        op_rope_halfrot(scope, pipelines, kn, freqs_in, kr, rows, hkv, hd)?;
        copy_tap(
            scope,
            qr.data,
            taps.qr.as_ref(),
            pipelines.act_bytes(rows * hq * hd),
        )?;
        copy_tap(
            scope,
            kr.data,
            taps.kr.as_ref(),
            pipelines.act_bytes(rows * hkv * hd),
        )?;

        // --- causal sdpa ---
        let sa = alloc_act(scope, pipelines, rows, hq * hd)?;
        op_sdpa(
            scope, pipelines, qr, kr, v, mask_in, sa, 1, rows, rows, hq, hkv, hd, scale, 1,
        )?;
        copy_tap(
            scope,
            sa.data,
            taps.sa.as_ref(),
            pipelines.act_bytes(rows * hq * hd),
        )?;

        // --- o_proj + residual ---
        let proj_scratch = alloc_matmul_out_buf(scope, pipelines, rows * dim)?;
        let dims_proj = scope.u32x4_uniform(rows, dim, hq * hd, 0)?;
        Block::dispatch_matmul_site(
            scope,
            pipelines,
            sa,
            o_w,
            proj_scratch,
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
        copy_tap(
            scope,
            proj_scratch,
            taps.proj.as_ref(),
            pipelines.act_bytes(rows * dim),
        )?;
        let after_attn = alloc_act(scope, pipelines, rows, dim)?;
        op_add(
            scope,
            pipelines,
            x_in,
            ActBuf::dense(proj_scratch),
            after_attn,
        )?;
        copy_tap(
            scope,
            after_attn.data,
            taps.after_attn.as_ref(),
            pipelines.act_bytes(rows * dim),
        )?;

        // --- pre-ffn norm ---
        let n2 = alloc_act(scope, pipelines, rows, dim)?;
        op_rmsnorm(scope, pipelines, after_attn, pa_ln, n2, rows, dim, eps)?;
        copy_tap(
            scope,
            n2.data,
            taps.n2.as_ref(),
            pipelines.act_bytes(rows * dim),
        )?;

        // --- SwiGLU FFN: down(silu_mul(gate(x), up(x))) ---
        let g_scratch = alloc_matmul_out_buf(scope, pipelines, rows * hid)?;
        let dims_g = scope.u32x4_uniform(rows, hid, dim, 0)?;
        Block::dispatch_matmul_site(
            scope,
            pipelines,
            n2,
            g_w,
            g_scratch,
            dims_g,
            pipelines.matmul_i8_ffn_up.as_ref(),
            pipelines.dequant_i8_ffn_up.as_ref(),
            pipelines.dequant_ffn_up.as_ref(),
            &pipelines.matmul_ffn_up,
            &pipelines.matmuls.ffn_up,
            rows,
            hid,
            dim,
        )?;
        let up_scratch = alloc_matmul_out_buf(scope, pipelines, rows * hid)?;
        let dims_up = scope.u32x4_uniform(rows, hid, dim, 0)?;
        Block::dispatch_matmul_site(
            scope,
            pipelines,
            n2,
            up_w,
            up_scratch,
            dims_up,
            pipelines.matmul_i8_ffn_up.as_ref(),
            pipelines.dequant_i8_ffn_up.as_ref(),
            pipelines.dequant_ffn_up.as_ref(),
            &pipelines.matmul_ffn_up,
            &pipelines.matmuls.ffn_up,
            rows,
            hid,
            dim,
        )?;
        copy_tap(
            scope,
            g_scratch,
            taps.gate.as_ref(),
            pipelines.act_bytes(rows * hid),
        )?;
        copy_tap(
            scope,
            up_scratch,
            taps.up.as_ref(),
            pipelines.act_bytes(rows * hid),
        )?;
        let gu = alloc_act(scope, pipelines, rows, hid)?;
        op_silu_mul(
            scope,
            pipelines,
            ActBuf::dense(g_scratch),
            ActBuf::dense(up_scratch),
            gu,
        )?;
        copy_tap(
            scope,
            gu.data,
            taps.gu.as_ref(),
            pipelines.act_bytes(rows * hid),
        )?;
        let down_scratch = alloc_matmul_out_buf(scope, pipelines, rows * dim)?;
        let dims_down = scope.u32x4_uniform(rows, dim, hid, 0)?;
        Block::dispatch_matmul_site(
            scope,
            pipelines,
            gu,
            down_w,
            down_scratch,
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
        copy_tap(
            scope,
            down_scratch,
            taps.down.as_ref(),
            pipelines.act_bytes(rows * dim),
        )?;
        op_add(
            scope,
            pipelines,
            after_attn,
            ActBuf::dense(down_scratch),
            y_out,
        )?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Encoder driver
// ---------------------------------------------------------------------------

/// Qwen3-4B text encoder: CPU embed lookup + N decoder layers, returns the
/// `hidden_states[-2]` activations (layer N-1 output, before the unused
/// `model.norm`).
pub struct Qwen3Encoder {
    rope: RopeEmbedder,
}

impl Qwen3Encoder {
    /// `max_seq` is the largest padded prompt length we'll ever encode; the
    /// rope table sizes against it.
    pub fn new(max_seq: usize) -> Self {
        let seq_len = max_seq.max(1);
        Self {
            // 1-axis rope (token position only); other axes are no-op (`d=0`).
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

    /// Run `N_LAYERS-1` decoder layers and return the (`hidden_states[-2]`)
    /// output as a host `Vec<f32>` of shape `[token_ids.len(), HIDDEN]`.
    pub async fn forward<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipelines: &BlockPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        handles: &Qwen3Handles,
        source: &S,
        token_ids: &[u32],
    ) -> Result<Qwen3Output, Qwen3ForwardError<S::Error>> {
        self.forward_taps(
            backend, pipelines, residency, scratch, handles, source, token_ids, None,
        )
        .await
    }

    /// `forward` with parity diagnostics (see [`Qwen3Taps`]). Tap readbacks
    /// happen per layer; production callers pass `None` and pay nothing.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward_taps<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipelines: &BlockPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        handles: &Qwen3Handles,
        source: &S,
        token_ids: &[u32],
        mut taps: Option<&mut Qwen3Taps>,
    ) -> Result<Qwen3Output, Qwen3ForwardError<S::Error>> {
        debug_assert_eq!(handles.layers.len(), config::HIDDEN_STATES_LAYER);
        let seq = token_ids.len();
        assert!(seq > 0, "Qwen3Encoder::forward: empty token list");
        // Pad to an even row count: the F16 act/mask layouts pack two elems
        // per u32 word (mask word-offsets require `s_k` even). The pad row
        // repeats the last token and is causally invisible to every real
        // row; its own output row is sliced off after readback.
        let mut ids = token_ids.to_vec();
        if !ids.len().is_multiple_of(2) {
            ids.push(*ids.last().expect("non-empty token list"));
        }
        let seq_pad = ids.len();
        assert!(
            seq_pad <= self.rope.axes_lens[0],
            "prompt length {seq_pad} exceeds Qwen3Encoder rope max_seq {}",
            self.rope.axes_lens[0]
        );

        // --- CPU embedding lookup (bf16 -> fp32 per row) ---
        let embeds = embed_lookup(source, &ids)
            .instrument(tracing::debug_span!(target: PHASE, "qwen3.embed_lookup", seq))
            .await
            .map_err(Qwen3ForwardError::Embed)?;

        if let Some(t) = taps.as_deref_mut() {
            t.embeds = embeds.clone();
        }

        let shape = Qwen3BlockShape::default_from_config(seq_pad);
        let block = Qwen3Block::new(shape);
        let act_bytes = pipelines.act_bytes((seq_pad * config::HIDDEN) as u32);

        // Per-op tap buffers for the `tap_block` layer (parity diagnostics).
        // Allocated only when requested; read back after the layer loop.
        let rows = seq_pad as u32;
        let tap_block = taps.as_deref().and_then(|t| t.tap_block);
        let dim = config::HIDDEN as u32;
        let hq = config::N_HEADS as u32;
        let hkv = config::N_KV_HEADS as u32;
        let hd = config::HEAD_DIM as u32;
        let hid = config::FFN_HIDDEN as u32;
        // (elem counts; order mirrors Qwen3BlockTaps fields)
        let tap_sizes: [u32; 16] = [
            rows * dim,      // n1
            rows * hq * hd,  // q
            rows * hkv * hd, // k
            rows * hkv * hd, // v
            rows * hq * hd,  // qn
            rows * hkv * hd, // kn
            rows * hq * hd,  // qr
            rows * hkv * hd, // kr
            rows * hq * hd,  // sa
            rows * dim,      // proj
            rows * dim,      // after_attn
            rows * dim,      // n2
            rows * hid,      // gate
            rows * hid,      // up
            rows * hid,      // gu
            rows * dim,      // down
        ];
        let mut tap_wsbufs = Vec::with_capacity(if tap_block.is_some() { 16 } else { 0 });
        let block_gpu = if tap_block.is_some() {
            for n in tap_sizes {
                tap_wsbufs.push(scratch.alloc(pipelines.act_bytes(n))?);
            }
            let r = |i: usize| Some(tap_wsbufs[i].as_buf_ref());
            Some(Qwen3BlockTaps {
                n1: r(0),
                q: r(1),
                k: r(2),
                v: r(3),
                qn: r(4),
                kn: r(5),
                qr: r(6),
                kr: r(7),
                sa: r(8),
                proj: r(9),
                after_attn: r(10),
                n2: r(11),
                gate: r(12),
                up: r(13),
                gu: r(14),
                down: r(15),
            })
        } else {
            None
        };

        let x_buf = scratch.alloc(act_bytes)?;
        let embed_bytes = seq::act_upload_bytes(pipelines.act_dtype, &embeds);
        backend.write_buffer(x_buf.id, 0, &embed_bytes)?;

        // --- rope freqs: positions 0..seq_pad on axis 0 ---
        let mut pos_ids = vec![0_i32; seq_pad * 3];
        for i in 0..seq_pad {
            pos_ids[i * 3] = i as i32;
        }
        let freqs_bytes = seq::freqs_upload_bytes(pipelines.act_dtype, &self.rope.lookup(&pos_ids));
        let freqs_buf = scratch.alloc(freqs_bytes.len() as u64)?;
        backend.write_buffer(freqs_buf.id, 0, &freqs_bytes)?;

        // --- causal mask [1, seq_pad, seq_pad] ---
        let mask_bytes = seq::causal_mask_bytes_act(seq_pad, pipelines.act_dtype);
        let mask_buf = scratch.alloc(mask_bytes.len() as u64)?;
        backend.write_buffer(mask_buf.id, 0, &mask_bytes)?;

        // --- run the registered N_LAYERS-1 (== 35) layers, ping-pong acts ---
        //
        // The output we want is upstream `hidden_states[-2]` (= post-layer-(N-2)
        // = post-layer-34 for N=36), NOT the last-layer output. Upstream
        // `hidden_states[-1]` is post-`model.norm` of the last layer; we drop
        // both the final layer's compute AND that norm. `register_qwen3_handles`
        // registers exactly the layers this loop runs.
        //
        // Prefetch overlap: each iteration runs `submit(enc).await` concurrently
        // with the *next* layer's `acquire(...)`. The first layer's weights are
        // acquired up front.
        let mut cur = x_buf;
        let mut pending: Option<Qwen3BlockViews<'_>> = if handles.layers.is_empty() {
            None
        } else {
            Some(
                handles.layers[0]
                    .acquire(residency, backend)
                    .instrument(tracing::debug_span!(target: PHASE, "qwen3.acquire", idx = 0_usize))
                    .await?,
            )
        };
        let freqs_ref = freqs_buf.as_buf_ref();
        let mask_ref = mask_buf.as_buf_ref();
        for (idx, _h) in handles.layers.iter().enumerate() {
            let _layer_guard = trace::scope!(format!("qwen3.layer.{idx}")).entered();
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
            match block_gpu.as_ref() {
                Some(t) if tap_block == Some(idx) => block
                    .forward_taps(&scope, pipelines, cur_h, freqs_h, mask_h, nxt_h, &bufs, t)?,
                _ => block.forward(&scope, pipelines, cur_h, freqs_h, mask_h, nxt_h, &bufs)?,
            }

            let next_idx = idx + 1;
            let next_acquire = async {
                match handles.layers.get(next_idx) {
                    Some(h) => {
                        let span =
                            tracing::debug_span!(target: PHASE, "qwen3.acquire", idx = next_idx);
                        Ok::<_, ResidencyError<S::Error, WgpuError>>(Some(
                            h.acquire(residency, backend).instrument(span).await?,
                        ))
                    }
                    None => Ok(None),
                }
            };
            let submit_fut = scope
                .submit_void()
                .instrument(tracing::debug_span!(target: PHASE, "qwen3.submit", idx));
            let (submit_res, next_res) = futures::join!(submit_fut, next_acquire);
            submit_res?;
            pending = next_res?;

            if let Some(t) = taps.as_deref_mut()
                && t.want_layer_outputs
            {
                let bytes = backend.read_buffer(nxt.id(), 0, act_bytes).await?;
                t.layer_outputs.push(seq::act_readback_to_f32(
                    pipelines.act_dtype,
                    &bytes,
                    seq_pad * config::HIDDEN,
                ));
            }

            drop(views);
            cur = nxt;
        }

        // --- tapped-layer per-op readback (parity diagnostics) ---
        if tap_block.is_some() {
            let mut decoded: Vec<Vec<f32>> = Vec::with_capacity(tap_wsbufs.len());
            for (b, n) in tap_wsbufs.iter().zip(tap_sizes) {
                let bytes = backend
                    .read_buffer(b.id(), 0, pipelines.act_bytes(n))
                    .await?;
                decoded.push(seq::act_readback_to_f32(
                    pipelines.act_dtype,
                    &bytes,
                    n as usize,
                ));
            }
            let t = taps.take().expect("tap_block implies taps");
            let mut it = decoded.into_iter();
            let mut next = || it.next().expect("16 tap fields");
            t.block_ops = Qwen3BlockOpsHost {
                n1: next(),
                q: next(),
                k: next(),
                v: next(),
                qn: next(),
                kn: next(),
                qr: next(),
                kr: next(),
                sa: next(),
                proj: next(),
                after_attn: next(),
                n2: next(),
                gate: next(),
                up: next(),
                gu: next(),
                down: next(),
            };
        }

        // --- readback final-layer output (drop the even-pad row, if any) ---
        let bytes = backend.read_buffer(cur.id(), 0, act_bytes).await?;
        let mut hidden =
            seq::act_readback_to_f32(pipelines.act_dtype, &bytes, seq_pad * config::HIDDEN);
        hidden.truncate(seq * config::HIDDEN);
        Ok(Qwen3Output { hidden, seq })
    }
}

/// Host-side parity tap sinks for [`Qwen3Encoder::forward_taps`]. All
/// tensors decode to f32 at the PADDED seq (callers compare against a
/// reference fed the same padded ids).
#[derive(Default)]
pub struct Qwen3Taps {
    /// Capture the full post-layer residual stream per layer
    /// (`[seq_pad, HIDDEN]` each) into `layer_outputs`.
    pub want_layer_outputs: bool,
    pub layer_outputs: Vec<Vec<f32>>,
    /// Capture per-op intermediates of layer `tap_block` into `block_ops`.
    pub tap_block: Option<usize>,
    pub block_ops: Qwen3BlockOpsHost,
    /// f32 embeds as gathered (pre act-dtype narrowing), `[seq_pad, HIDDEN]`.
    pub embeds: Vec<f32>,
}

/// Decoded per-op intermediates for the tapped layer (see
/// [`Qwen3BlockTaps`] for shapes).
#[derive(Default)]
pub struct Qwen3BlockOpsHost {
    pub n1: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub qn: Vec<f32>,
    pub kn: Vec<f32>,
    pub qr: Vec<f32>,
    pub kr: Vec<f32>,
    pub sa: Vec<f32>,
    pub proj: Vec<f32>,
    pub after_attn: Vec<f32>,
    pub n2: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub gu: Vec<f32>,
    pub down: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Qwen3Output {
    /// `[seq, HIDDEN]` row-major fp32. Equivalent to upstream
    /// `hidden_states[-2]` (layer N-1 output, before the unused final norm).
    pub hidden: Vec<f32>,
    pub seq: usize,
}

#[derive(Debug)]
pub enum Qwen3ForwardError<SE: core::fmt::Debug> {
    Embed(EmbedLookupError),
    Wgpu(WgpuError),
    Residency(ResidencyError<SE, WgpuError>),
}

impl<SE: core::fmt::Debug> From<WgpuError> for Qwen3ForwardError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}

impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for Qwen3ForwardError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_count_matches_config() {
        let expected = expected_weights();
        // 1 (embed) + N_LAYERS * 11 (2 norms + 4 attn proj + 2 q/k norm + 3 mlp)
        assert_eq!(expected.len(), 1 + config::N_LAYERS * 11);
    }

    #[test]
    fn handles_field_names_match_weight_ids() {
        let w = Qwen3Weights::new();
        assert_eq!(w.embed_tokens.0, "model.embed_tokens.weight");
        assert_eq!(w.layers.len(), config::N_LAYERS);
        let b0 = &w.layers[0];
        assert_eq!(b0.q_proj.0, "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(b0.q_norm.0, "model.layers.0.self_attn.q_norm.weight");
        assert_eq!(b0.mlp_gate.0, "model.layers.0.mlp.gate_proj.weight");
    }
}
