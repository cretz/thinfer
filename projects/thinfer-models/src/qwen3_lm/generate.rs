//! Load + register + audit for the Qwen3-VL-Instruct rewriter LM (8B or 4B,
//! chosen by [`Qwen3LmConfig`]).
//!
//! The GGUF ships native `qwen3vl` tensor names (`token_embd`, `output`,
//! `output_norm`, `blk.{i}.*`); [`qwen3vl_gguf_renames`] re-keys them to the HF
//! `model.*` / `lm_head` names the shared Z-Image Qwen3 block machinery expects.
//! Wrap the `GgufSource` with
//! `RenamedSource::with_passthrough(gguf, qwen3vl_gguf_renames(&cfg))` so the
//! rest of the code sees HF names.
//!
//! Unlike the Z-Image text encoder (which stops at `hidden_states[-2]` and runs
//! `n_layers - 1` layers, no final norm, no lm_head), the rewriter is a full
//! generator: ALL layers register, plus the final `model.norm` and the `lm_head`.
//! When the head is UNTIED (8B) it is a distinct `output.weight`; when TIED (4B)
//! it reuses the `token_embd` matrix. The token embedding is registered ONLY in
//! the tied case (it doubles as the head); decode always gathers embedding rows
//! from disk (see [`crate::z_image::text_encoder::embed_lookup_hidden`]).

use std::collections::HashMap;

use thinfer_core::quant::QuantKind;
use thinfer_core::residency::{TransposePolicy, WeightHandle, WeightResidency};
use thinfer_core::tensor::StorageEncoding;
use thinfer_core::weight::{WeightCatalog, WeightId, WeightSource};

use crate::qwen3_lm::Qwen3LmConfig;
use crate::z_image::text_encoder::{
    AuditReport, Expected, LoadError, Qwen3BlockHandles, Qwen3BlockWeights, ShapeMismatch,
    register_one,
};

/// GGUF (`qwen3vl` native) -> HF (`model.*` / `lm_head`) tensor-name map.
///
/// Covers the embedding, the final norm (`output_norm.weight` ->
/// `model.norm.weight`), and every decoder layer. The UNTIED lm_head
/// (`output.weight` -> `lm_head.weight`) is only mapped when the model actually
/// ships it (`cfg.tied_embeddings == false`); a tied model has no `output.weight`
/// and reuses `token_embd` as the head. Re-keys the source so [`register_one`]
/// and the shared embed gather (HF names) work unchanged.
pub fn qwen3vl_gguf_renames(cfg: &Qwen3LmConfig) -> HashMap<WeightId, WeightId> {
    let mut m = HashMap::new();
    let mut put = |g: String, h: String| {
        m.insert(WeightId(g), WeightId(h));
    };
    put(
        "token_embd.weight".into(),
        "model.embed_tokens.weight".into(),
    );
    if !cfg.tied_embeddings {
        put("output.weight".into(), "lm_head.weight".into());
    }
    put("output_norm.weight".into(), "model.norm.weight".into());
    for i in 0..cfg.n_layers {
        let g = |s: &str| format!("blk.{i}.{s}");
        let h = |s: &str| format!("model.layers.{i}.{s}");
        put(g("attn_norm.weight"), h("input_layernorm.weight"));
        put(g("ffn_norm.weight"), h("post_attention_layernorm.weight"));
        put(g("attn_q.weight"), h("self_attn.q_proj.weight"));
        put(g("attn_k.weight"), h("self_attn.k_proj.weight"));
        put(g("attn_v.weight"), h("self_attn.v_proj.weight"));
        put(g("attn_output.weight"), h("self_attn.o_proj.weight"));
        put(g("attn_q_norm.weight"), h("self_attn.q_norm.weight"));
        put(g("attn_k_norm.weight"), h("self_attn.k_norm.weight"));
        put(g("ffn_gate.weight"), h("mlp.gate_proj.weight"));
        put(g("ffn_up.weight"), h("mlp.up_proj.weight"));
        put(g("ffn_down.weight"), h("mlp.down_proj.weight"));
    }
    m
}

/// HF weight names for the whole rewriter LM. The embedding is a handle ONLY
/// when it doubles as the tied lm_head; otherwise it is a CPU gather and appears
/// here only for the audit.
#[derive(Clone, Debug)]
pub struct Qwen3LmWeights {
    pub embed_tokens: WeightId,
    pub final_norm: WeightId,
    /// The lm_head weight id: a distinct `lm_head.weight` (untied) or the shared
    /// `model.embed_tokens.weight` (tied).
    pub lm_head: WeightId,
    pub layers: Vec<Qwen3BlockWeights>,
}

impl Qwen3LmWeights {
    pub fn new(cfg: &Qwen3LmConfig) -> Self {
        let embed = WeightId("model.embed_tokens.weight".into());
        let lm_head = if cfg.tied_embeddings {
            embed.clone()
        } else {
            WeightId("lm_head.weight".into())
        };
        Self {
            embed_tokens: embed,
            final_norm: WeightId("model.norm.weight".into()),
            lm_head,
            layers: (0..cfg.n_layers).map(Qwen3BlockWeights::new).collect(),
        }
    }
}

/// Registered residency handles for the whole rewriter LM. The token embedding
/// is intentionally absent (CPU gather from disk, never paged whole).
#[derive(Clone, Debug)]
pub struct Qwen3LmHandles {
    pub layers: Vec<Qwen3BlockHandles>,
    /// Per-layer quant of `attn_v` / `ffn_down`. The Q5_K_M GGUF bumps these two
    /// weights to Q6_K in roughly half the layers and keeps them Q5_K in the
    /// rest; every other matmul weight is Q5_K. The forward routes each matmul by
    /// its actual quant, so these must travel with the handles.
    pub v_quant: Vec<QuantKind>,
    pub down_quant: Vec<QuantKind>,
    pub final_norm: WeightHandle,
    pub lm_head: WeightHandle,
    pub lm_head_quant: QuantKind,
}

/// Register every rewriter weight into `residency`: all decoder layers, the
/// final `model.norm`, and the `lm_head` (a distinct `output.weight` when untied,
/// or the shared `token_embd` when tied -- so the embedding IS registered in the
/// tied case; the CPU gather reads it from the source either way).
///
/// Wrap the source in [`qwen3vl_gguf_renames`] before calling. The GGUF ships
/// every matmul weight already block-major Quant, so [`register_one`] registers
/// them as-is; the F32 norms register dense. Both cases pass
/// `TransposePolicy::None, None`.
pub fn register_qwen3_lm<S: WeightSource>(
    residency: &WeightResidency<S>,
    cfg: &Qwen3LmConfig,
) -> Result<Qwen3LmHandles, LoadError> {
    let weights = Qwen3LmWeights::new(cfg);
    let reg = |id: &WeightId| register_one(residency, id, TransposePolicy::None, None);
    let mut layers = Vec::with_capacity(weights.layers.len());
    let mut v_quant = Vec::with_capacity(weights.layers.len());
    let mut down_quant = Vec::with_capacity(weights.layers.len());
    for b in &weights.layers {
        layers.push(Qwen3BlockHandles {
            input_layernorm: reg(&b.input_layernorm)?,
            post_attention_layernorm: reg(&b.post_attention_layernorm)?,
            q_proj: reg(&b.q_proj)?,
            k_proj: reg(&b.k_proj)?,
            v_proj: reg(&b.v_proj)?,
            o_proj: reg(&b.o_proj)?,
            q_norm: reg(&b.q_norm)?,
            k_norm: reg(&b.k_norm)?,
            mlp_gate: reg(&b.mlp_gate)?,
            mlp_up: reg(&b.mlp_up)?,
            mlp_down: reg(&b.mlp_down)?,
        });
        v_quant.push(matmul_quant(residency, &b.v_proj)?);
        down_quant.push(matmul_quant(residency, &b.mlp_down)?);
    }
    let final_norm = reg(&weights.final_norm)?;
    let lm_head = reg(&weights.lm_head)?;
    let lm_head_quant = matmul_quant(residency, &weights.lm_head)?;
    Ok(Qwen3LmHandles {
        layers,
        v_quant,
        down_quant,
        final_norm,
        lm_head,
        lm_head_quant,
    })
}

/// The GGUF quant kind of a matmul weight, read from the residency catalog. Used
/// to route each matmul through the pipeline that matches its actual encoding
/// (the Q5_K_M mix varies attn_v / ffn_down per layer).
fn matmul_quant<S: WeightSource>(
    residency: &WeightResidency<S>,
    id: &WeightId,
) -> Result<QuantKind, LoadError> {
    let entry = residency
        .source()
        .catalog()
        .get(id)
        .ok_or_else(|| LoadError::UnknownWeight(id.clone()))?;
    match entry.encoding {
        Some(StorageEncoding::Quant(k)) => Ok(k),
        enc => Err(LoadError::Undecodable {
            id: id.clone(),
            encoding: enc,
            label: entry.encoding_label.clone(),
        }),
    }
}

/// Expected HF tensor names + on-disk shapes `[N, K]` (outer-first, matching the
/// engine's reversed GGUF dims). Embedding + final norm + lm_head + 36 layers x
/// 11 per-layer tensors.
pub fn expected_weights(cfg: &Qwen3LmConfig) -> Vec<Expected> {
    let hidden = cfg.hidden;
    let head = cfg.head_dim;
    let q_dim = cfg.n_heads * head;
    let kv_dim = cfg.n_kv_heads * head;
    let ffn = cfg.ffn_hidden;
    let mut out = Vec::with_capacity(3 + cfg.n_layers * 11);
    let push = |out: &mut Vec<Expected>, name: String, shape: Vec<usize>| {
        out.push(Expected {
            id: WeightId(name),
            shape,
        });
    };
    push(
        &mut out,
        "model.embed_tokens.weight".into(),
        vec![cfg.vocab, hidden],
    );
    push(&mut out, "model.norm.weight".into(), vec![hidden]);
    // A tied model has no separate `output.weight`; the head reuses `token_embd`.
    if !cfg.tied_embeddings {
        push(&mut out, "lm_head.weight".into(), vec![cfg.vocab, hidden]);
    }
    for i in 0..cfg.n_layers {
        let p = format!("model.layers.{i}");
        push(
            &mut out,
            format!("{p}.input_layernorm.weight"),
            vec![hidden],
        );
        push(
            &mut out,
            format!("{p}.post_attention_layernorm.weight"),
            vec![hidden],
        );
        push(
            &mut out,
            format!("{p}.self_attn.q_proj.weight"),
            vec![q_dim, hidden],
        );
        push(
            &mut out,
            format!("{p}.self_attn.k_proj.weight"),
            vec![kv_dim, hidden],
        );
        push(
            &mut out,
            format!("{p}.self_attn.v_proj.weight"),
            vec![kv_dim, hidden],
        );
        push(
            &mut out,
            format!("{p}.self_attn.o_proj.weight"),
            vec![hidden, q_dim],
        );
        push(&mut out, format!("{p}.self_attn.q_norm.weight"), vec![head]);
        push(&mut out, format!("{p}.self_attn.k_norm.weight"), vec![head]);
        push(
            &mut out,
            format!("{p}.mlp.gate_proj.weight"),
            vec![ffn, hidden],
        );
        push(
            &mut out,
            format!("{p}.mlp.up_proj.weight"),
            vec![ffn, hidden],
        );
        push(
            &mut out,
            format!("{p}.mlp.down_proj.weight"),
            vec![hidden, ffn],
        );
    }
    out
}

/// Catalog audit: missing names + shape mismatches against [`expected_weights`].
/// Unlike the Z-Image text-encoder audit (which ignores the final norm + lm_head),
/// the rewriter REQUIRES them, so they are part of the expected set here.
pub fn audit(catalog: &WeightCatalog, cfg: &Qwen3LmConfig) -> AuditReport {
    let expected = expected_weights(cfg);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rename_map_covers_top_level_and_layers() {
        let cfg = Qwen3LmConfig::qwen3_vl_8b();
        let m = qwen3vl_gguf_renames(&cfg);
        assert_eq!(
            m.get(&WeightId("token_embd.weight".into())).unwrap().0,
            "model.embed_tokens.weight"
        );
        assert_eq!(
            m.get(&WeightId("output.weight".into())).unwrap().0,
            "lm_head.weight"
        );
        assert_eq!(
            m.get(&WeightId("output_norm.weight".into())).unwrap().0,
            "model.norm.weight"
        );
        assert_eq!(
            m.get(&WeightId("blk.35.ffn_down.weight".into())).unwrap().0,
            "model.layers.35.mlp.down_proj.weight"
        );
        assert_eq!(
            m.get(&WeightId("blk.0.attn_q_norm.weight".into()))
                .unwrap()
                .0,
            "model.layers.0.self_attn.q_norm.weight"
        );
        // 3 top-level (embed, lm_head, norm) + 36 layers * 11 per-layer.
        assert_eq!(m.len(), 3 + cfg.n_layers * 11);
    }

    #[test]
    fn expected_count_matches_config() {
        // Untied 8B: 3 module-level (embed, norm, lm_head) + N_LAYERS * 11.
        let cfg8 = Qwen3LmConfig::qwen3_vl_8b();
        assert_eq!(expected_weights(&cfg8).len(), 3 + cfg8.n_layers * 11);
        // Tied 4B: no separate lm_head.weight, so 2 module-level + per-layer.
        let cfg4 = Qwen3LmConfig::qwen3_vl_4b();
        assert_eq!(expected_weights(&cfg4).len(), 2 + cfg4.n_layers * 11);
    }

    #[test]
    fn tied_model_omits_lm_head_rename() {
        let cfg = Qwen3LmConfig::qwen3_vl_4b();
        let m = qwen3vl_gguf_renames(&cfg);
        assert!(!m.contains_key(&WeightId("output.weight".into())));
        // 2 top-level (embed, norm) + n_layers * 11 per-layer.
        assert_eq!(m.len(), 2 + cfg.n_layers * 11);
    }
}
