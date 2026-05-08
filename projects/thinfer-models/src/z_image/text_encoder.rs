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
use thinfer_core::ops::{AddF32, MulF32, RmsNormF32, RopeF32HalfRot, SdpaF32, SiluF32};
use thinfer_core::residency::{
    GpuView, ResidencyError, TransposePolicy, WeightHandle, WeightMeta, WeightResidency,
};
use thinfer_core::tensor::StorageEncoding;
use thinfer_core::trace::{self, PHASE};
use thinfer_core::weight::{Decoder, WeightCatalog, WeightId, WeightReader, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace};
use tracing::Instrument;

use crate::z_image::block::BlockPipelines;
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

pub fn register_qwen3_handles<S: WeightSource>(
    residency: &WeightResidency<S>,
) -> Result<Qwen3Handles, LoadError> {
    let weights = Qwen3Weights::new();
    // `model.embed_tokens.weight` is the 777 MB [vocab, hidden] lookup table.
    // Intentionally NOT registered with residency: a prompt only ever indexes
    // a few hundred rows, so we gather rows directly from disk via
    // `embed_lookup` and never page or upload the full table.
    let mut layers = Vec::with_capacity(weights.layers.len());
    for b in &weights.layers {
        layers.push(Qwen3BlockHandles {
            input_layernorm: register_one(residency, &b.input_layernorm, TransposePolicy::None)?,
            post_attention_layernorm: register_one(
                residency,
                &b.post_attention_layernorm,
                TransposePolicy::None,
            )?,
            q_proj: register_one(residency, &b.q_proj, TransposePolicy::Linear2D)?,
            k_proj: register_one(residency, &b.k_proj, TransposePolicy::Linear2D)?,
            v_proj: register_one(residency, &b.v_proj, TransposePolicy::Linear2D)?,
            o_proj: register_one(residency, &b.o_proj, TransposePolicy::Linear2D)?,
            q_norm: register_one(residency, &b.q_norm, TransposePolicy::None)?,
            k_norm: register_one(residency, &b.k_norm, TransposePolicy::None)?,
            mlp_gate: register_one(residency, &b.mlp_gate, TransposePolicy::Linear2D)?,
            mlp_up: register_one(residency, &b.mlp_up, TransposePolicy::Linear2D)?,
            mlp_down: register_one(residency, &b.mlp_down, TransposePolicy::Linear2D)?,
        });
    }
    Ok(Qwen3Handles { layers })
}

fn register_one<S: WeightSource>(
    residency: &WeightResidency<S>,
    id: &WeightId,
    transpose: TransposePolicy,
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
    Ok(residency.register(WeightMeta {
        id: id.clone(),
        shape: entry.shape.clone(),
        encoding,
        on_disk_bytes: entry.size,
        transpose,
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
    /// submits the scope.
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
        let s = self.shape;
        let rows = s.seq as u32;
        let dim = s.dim as u32;
        let hd = s.head_dim as u32;
        let hq = s.n_heads as u32;
        let hkv = s.n_kv_heads as u32;
        let hid = s.ffn_hidden as u32;
        let eps = s.norm_eps;
        let scale = s.sdpa_scale();

        let act_bytes = (rows * dim) as u64 * 4;
        let q_bytes = (rows * hq * hd) as u64 * 4;
        let kv_bytes = (rows * hkv * hd) as u64 * 4;
        let hid_bytes = (rows * hid) as u64 * 4;
        let pairs = hd / 2;

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
        let n1 = scope.alloc(act_bytes)?;
        let u_n1 = scope_rmsnorm_uniform(scope, rows, dim, eps)?;
        scope.rmsnorm::<RmsNormF32>(&pipelines.rmsnorm, x_in, in_ln, u_n1, n1, rows)?;

        // --- q/k/v projections ---
        let q = scope.alloc(q_bytes)?;
        let k = scope.alloc(kv_bytes)?;
        let v = scope.alloc(kv_bytes)?;
        let u_q = scope.u32x4_uniform(rows, hq * hd, dim, 0)?;
        scope.matmul(
            &pipelines.matmul_qkv,
            &pipelines.matmuls.qkv,
            n1,
            q_w,
            u_q,
            q,
            rows,
            hq * hd,
        )?;
        let u_k = scope.u32x4_uniform(rows, hkv * hd, dim, 0)?;
        scope.matmul(
            &pipelines.matmul_qkv,
            &pipelines.matmuls.qkv,
            n1,
            k_w,
            u_k,
            k,
            rows,
            hkv * hd,
        )?;
        let u_v = scope.u32x4_uniform(rows, hkv * hd, dim, 0)?;
        scope.matmul(
            &pipelines.matmul_qkv,
            &pipelines.matmuls.qkv,
            n1,
            v_w,
            u_v,
            v,
            rows,
            hkv * hd,
        )?;

        // --- per-head Q/K RMSNorm over head_dim ---
        let qn = scope.alloc(q_bytes)?;
        let kn = scope.alloc(kv_bytes)?;
        let u_qn = scope_rmsnorm_uniform(scope, rows * hq, hd, eps)?;
        scope.rmsnorm::<RmsNormF32>(&pipelines.rmsnorm, q, qn_w, u_qn, qn, rows * hq)?;
        let u_kn = scope_rmsnorm_uniform(scope, rows * hkv, hd, eps)?;
        scope.rmsnorm::<RmsNormF32>(&pipelines.rmsnorm, k, kn_w, u_kn, kn, rows * hkv)?;

        // --- rope on Q, K (1-axis token-position freqs broadcast across heads) ---
        let qr = scope.alloc(q_bytes)?;
        let kr = scope.alloc(kv_bytes)?;
        let u_qr = scope.u32x4_uniform(rows, hq, pairs, 0)?;
        scope.rope::<RopeF32HalfRot>(
            &pipelines.rope_halfrot,
            qn,
            freqs_in,
            u_qr,
            qr,
            rows,
            hq,
            pairs,
        )?;
        let u_kr = scope.u32x4_uniform(rows, hkv, pairs, 0)?;
        scope.rope::<RopeF32HalfRot>(
            &pipelines.rope_halfrot,
            kn,
            freqs_in,
            u_kr,
            kr,
            rows,
            hkv,
            pairs,
        )?;

        // --- causal sdpa ---
        let sa = scope.alloc(q_bytes)?;
        let u_sa = scope_sdpa_uniform(scope, 1, hq, hkv, rows, rows, hd, scale, 1)?;
        scope.sdpa::<SdpaF32>(&pipelines.sdpa, qr, kr, v, mask_in, u_sa, sa, 1, rows, hq)?;

        // --- o_proj + residual ---
        let proj = scope.alloc(act_bytes)?;
        let u_proj = scope.u32x4_uniform(rows, dim, hq * hd, 0)?;
        scope.matmul(
            &pipelines.matmul_proj,
            &pipelines.matmuls.proj,
            sa,
            o_w,
            u_proj,
            proj,
            rows,
            dim,
        )?;
        let after_attn = scope.alloc(act_bytes)?;
        scope.dispatch_op::<AddF32>(&pipelines.add, &[x_in, proj], after_attn)?;

        // --- pre-ffn norm ---
        let n2 = scope.alloc(act_bytes)?;
        let u_n2 = scope_rmsnorm_uniform(scope, rows, dim, eps)?;
        scope.rmsnorm::<RmsNormF32>(&pipelines.rmsnorm, after_attn, pa_ln, u_n2, n2, rows)?;

        // --- SwiGLU FFN: down(silu(gate(x)) * up(x)) ---
        let g = scope.alloc(hid_bytes)?;
        let u_mlp_in = scope.alloc(hid_bytes)?;
        let u_g = scope.u32x4_uniform(rows, hid, dim, 0)?;
        scope.matmul(
            &pipelines.matmul_ffn_up,
            &pipelines.matmuls.ffn_up,
            n2,
            g_w,
            u_g,
            g,
            rows,
            hid,
        )?;
        let u_u = scope.u32x4_uniform(rows, hid, dim, 0)?;
        scope.matmul(
            &pipelines.matmul_ffn_up,
            &pipelines.matmuls.ffn_up,
            n2,
            up_w,
            u_u,
            u_mlp_in,
            rows,
            hid,
        )?;
        let gs = scope.alloc(hid_bytes)?;
        scope.dispatch_op::<SiluF32>(&pipelines.silu, &[g], gs)?;
        let gu = scope.alloc(hid_bytes)?;
        scope.dispatch_op::<MulF32>(&pipelines.mul, &[gs, u_mlp_in], gu)?;
        let down = scope.alloc(act_bytes)?;
        let u_down = scope.u32x4_uniform(rows, dim, hid, 0)?;
        scope.matmul(
            &pipelines.matmul_ffn_down,
            &pipelines.matmuls.ffn_down,
            gu,
            down_w,
            u_down,
            down,
            rows,
            dim,
        )?;
        scope.dispatch_op::<AddF32>(&pipelines.add, &[after_attn, down], y_out)?;

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
        debug_assert_eq!(handles.layers.len(), config::N_LAYERS);
        let seq = token_ids.len();
        assert!(seq > 0, "Qwen3Encoder::forward: empty token list");
        assert!(
            seq <= self.rope.axes_lens[0],
            "prompt length {seq} exceeds Qwen3Encoder rope max_seq {}",
            self.rope.axes_lens[0]
        );

        // --- CPU embedding lookup (bf16 -> fp32 per row) ---
        let embeds = embed_lookup(source, token_ids)
            .instrument(tracing::debug_span!(target: PHASE, "qwen3.embed_lookup", seq))
            .await
            .map_err(Qwen3ForwardError::Embed)?;

        let shape = Qwen3BlockShape::default_from_config(seq);
        let block = Qwen3Block::new(shape);
        let act_bytes = (seq * config::HIDDEN * 4) as u64;

        let x_buf = scratch.alloc(act_bytes)?;
        backend.write_buffer(x_buf.id, 0, bytes_of_f32(&embeds).as_slice())?;

        // --- rope freqs: positions 0..seq on axis 0 ---
        let mut pos_ids = vec![0_i32; seq * 3];
        for i in 0..seq {
            pos_ids[i * 3] = i as i32;
        }
        let freqs_bytes = self.rope.lookup_bytes(&pos_ids);
        let freqs_buf = scratch.alloc(freqs_bytes.len() as u64)?;
        backend.write_buffer(freqs_buf.id, 0, &freqs_bytes)?;

        // --- causal mask [1, seq, seq] ---
        let mask_bytes = seq::causal_mask_bytes(seq);
        let mask_buf = scratch.alloc(mask_bytes.len() as u64)?;
        backend.write_buffer(mask_buf.id, 0, &mask_bytes)?;

        // --- run N_LAYERS-1 (== 35) layers, ping-pong activations ---
        //
        // The output we want is upstream `hidden_states[-2]` (= post-layer-(N-2)
        // = post-layer-34 for N=36), NOT the last-layer output. Upstream
        // `hidden_states[-1]` is post-`model.norm` of the last layer; we drop
        // both the final layer's compute AND that norm. So this loop runs only
        // the first N_LAYERS-1 layers. Weights for the unused layer are still
        // registered today (TODO: drop layer-(N-1) from `register_qwen3_handles`
        // to free its residency footprint).
        //
        // Prefetch overlap: each iteration runs `submit(enc).await` concurrently
        // with the *next* layer's `acquire(...)`. The first layer's weights are
        // acquired up front; the last iteration's prefetch acquires the unused
        // final layer (wasted, see TODO above).
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
        let n_run = handles.layers.len().saturating_sub(1);
        let freqs_ref = freqs_buf.as_buf_ref();
        let mask_ref = mask_buf.as_buf_ref();
        for (idx, _h) in handles.layers.iter().take(n_run).enumerate() {
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
            block.forward(&scope, pipelines, cur_h, freqs_h, mask_h, nxt_h, &bufs)?;

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

            drop(views);
            cur = nxt;
        }

        // --- readback final-layer output ---
        let bytes = backend.read_buffer(cur.id(), 0, act_bytes).await?;
        let mut hidden = vec![0_f32; seq * config::HIDDEN];
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            hidden[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Ok(Qwen3Output { hidden, seq })
    }
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

fn bytes_of_f32(slice: &[f32]) -> Vec<u8> {
    let mut bytes = vec![0u8; slice.len() * 4];
    for (i, v) in slice.iter().enumerate() {
        bytes[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn scope_rmsnorm_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    n_rows: u32,
    d: u32,
    eps: f32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&n_rows.to_le_bytes());
    bytes[4..8].copy_from_slice(&d.to_le_bytes());
    bytes[8..12].copy_from_slice(&eps.to_le_bytes());
    scope.write_uniform(&bytes)
}

#[allow(clippy::too_many_arguments)]
fn scope_sdpa_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    b: u32,
    h_q: u32,
    h_kv: u32,
    s_q: u32,
    s_k: u32,
    d: u32,
    scale: f32,
    has_mask: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&b.to_le_bytes());
    bytes[4..8].copy_from_slice(&h_q.to_le_bytes());
    bytes[8..12].copy_from_slice(&h_kv.to_le_bytes());
    bytes[12..16].copy_from_slice(&s_q.to_le_bytes());
    bytes[16..20].copy_from_slice(&s_k.to_le_bytes());
    bytes[20..24].copy_from_slice(&d.to_le_bytes());
    bytes[24..28].copy_from_slice(&scale.to_le_bytes());
    bytes[28..32].copy_from_slice(&has_mask.to_le_bytes());
    scope.write_uniform(&bytes)
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
