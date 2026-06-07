//! Structural pre-check for the Z-Image-Turbo checkpoint.
//!
//! Runs against a parsed `WeightCatalog` (header-only, no tensor data needed).
//! Confirms every weight `forward()` would resolve actually exists in the file
//! and has the shape we'll dispatch against. Cheap (~600 entries) and runs at
//! load time before the first forward.
//!
//! Shape source: `third-party/Z-Image/src/zimage/transformer.py`
//! `ZImageTransformer2DModel` defaults, audited 2026-05-10. Per-PyTorch Linear
//! convention `weight` is `[out, in]`; we check that on-disk shape, not the
//! transposed shape we upload to GPU.

use thinfer_core::weight::{WeightCatalog, WeightId};

use crate::z_image::config;

/// One expected entry: name + on-disk shape (PyTorch convention).
#[derive(Clone, Debug)]
pub struct Expected {
    pub id: WeightId,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct AuditReport {
    pub expected: usize,
    pub missing: Vec<WeightId>,
    pub shape_mismatches: Vec<ShapeMismatch>,
    /// Names present in the catalog but not in the expected set. Informational
    /// only: ema-only checkpoints can drop optimizer state, full checkpoints
    /// can carry it; either way unexpected extras don't fail the audit.
    pub extra: Vec<WeightId>,
}

#[derive(Clone, Debug)]
pub struct ShapeMismatch {
    pub id: WeightId,
    pub expected: Vec<usize>,
    pub got: Vec<usize>,
}

impl AuditReport {
    pub fn ok(&self) -> bool {
        self.missing.is_empty() && self.shape_mismatches.is_empty()
    }
}

/// Build the expected weight list for a Z-Image-Turbo checkpoint with the
/// `2-1` patch variant.
pub fn expected_weights() -> Vec<Expected> {
    let mut out =
        Vec::with_capacity(2 + 7 + 6 + (config::N_LAYERS + 2 * config::N_REFINER_LAYERS) * 16);

    // module-level
    let dim = config::DIM;
    let head_dim = config::HEAD_DIM;
    let cap_feat_dim = config::CAP_FEAT_DIM;
    let adaln_embed = config::ADALN_EMBED_DIM;
    let adaln_out = config::ADALN_OUT;
    let mid = config::T_EMBEDDER_MID;
    let freq = config::FREQUENCY_EMBEDDING_SIZE;
    let in_channels: usize = 16; // VAE latent channels; Z-Image-Turbo defaults
    let out_channels = in_channels;
    let patch = 2usize;
    let f_patch = 1usize;
    let patch_in = f_patch * patch * patch * in_channels;
    let patch_out = patch * patch * f_patch * out_channels;
    let patch_key = config::PATCH_KEY;

    let push = |out: &mut Vec<Expected>, name: &str, shape: Vec<usize>| {
        out.push(Expected {
            id: WeightId(name.to_string()),
            shape,
        });
    };

    push(&mut out, "x_pad_token", vec![1, dim]);
    push(&mut out, "cap_pad_token", vec![1, dim]);
    push(&mut out, "t_embedder.mlp.0.weight", vec![mid, freq]);
    push(&mut out, "t_embedder.mlp.0.bias", vec![mid]);
    push(&mut out, "t_embedder.mlp.2.weight", vec![adaln_embed, mid]);
    push(&mut out, "t_embedder.mlp.2.bias", vec![adaln_embed]);
    push(&mut out, "cap_embedder.0.weight", vec![cap_feat_dim]);
    push(&mut out, "cap_embedder.1.weight", vec![dim, cap_feat_dim]);
    push(&mut out, "cap_embedder.1.bias", vec![dim]);
    push(
        &mut out,
        &format!("all_x_embedder.{patch_key}.weight"),
        vec![dim, patch_in],
    );
    push(
        &mut out,
        &format!("all_x_embedder.{patch_key}.bias"),
        vec![dim],
    );
    push(
        &mut out,
        &format!("all_final_layer.{patch_key}.linear.weight"),
        vec![patch_out, dim],
    );
    push(
        &mut out,
        &format!("all_final_layer.{patch_key}.linear.bias"),
        vec![patch_out],
    );
    push(
        &mut out,
        &format!("all_final_layer.{patch_key}.adaLN_modulation.1.weight"),
        vec![dim, adaln_embed],
    );
    push(
        &mut out,
        &format!("all_final_layer.{patch_key}.adaLN_modulation.1.bias"),
        vec![dim],
    );

    // block stacks
    for (prefix, n, modulated) in [
        ("layers", config::N_LAYERS, true),
        ("noise_refiner", config::N_REFINER_LAYERS, true),
        ("context_refiner", config::N_REFINER_LAYERS, false),
    ] {
        for i in 0..n {
            let p = format!("{prefix}.{i}");
            push(&mut out, &format!("{p}.attention_norm1.weight"), vec![dim]);
            push(&mut out, &format!("{p}.attention_norm2.weight"), vec![dim]);
            push(&mut out, &format!("{p}.ffn_norm1.weight"), vec![dim]);
            push(&mut out, &format!("{p}.ffn_norm2.weight"), vec![dim]);
            push(
                &mut out,
                &format!("{p}.attention.qkv.weight"),
                vec![(config::N_HEADS + 2 * config::N_KV_HEADS) * head_dim, dim],
            );
            push(
                &mut out,
                &format!("{p}.attention.out.weight"),
                vec![dim, config::N_HEADS * head_dim],
            );
            push(
                &mut out,
                &format!("{p}.attention.norm_q.weight"),
                vec![head_dim],
            );
            push(
                &mut out,
                &format!("{p}.attention.norm_k.weight"),
                vec![head_dim],
            );
            push(
                &mut out,
                &format!("{p}.feed_forward.w1.weight"),
                vec![config::FFN_HIDDEN, dim],
            );
            push(
                &mut out,
                &format!("{p}.feed_forward.w2.weight"),
                vec![dim, config::FFN_HIDDEN],
            );
            push(
                &mut out,
                &format!("{p}.feed_forward.w3.weight"),
                vec![config::FFN_HIDDEN, dim],
            );
            if modulated {
                push(
                    &mut out,
                    &format!("{p}.adaLN_modulation.0.weight"),
                    vec![adaln_out, adaln_embed],
                );
                push(
                    &mut out,
                    &format!("{p}.adaLN_modulation.0.bias"),
                    vec![adaln_out],
                );
            }
        }
    }

    out
}

pub fn audit(catalog: &WeightCatalog) -> AuditReport {
    let expected = expected_weights();
    let mut report = AuditReport {
        expected: expected.len(),
        ..Default::default()
    };

    let mut expected_names = std::collections::HashSet::with_capacity(expected.len());
    for e in &expected {
        expected_names.insert(e.id.clone());
        match catalog.get(&e.id) {
            None => report.missing.push(e.id.clone()),
            Some(entry) => {
                if entry.shape.0 != e.shape {
                    report.shape_mismatches.push(ShapeMismatch {
                        id: e.id.clone(),
                        expected: e.shape.clone(),
                        got: entry.shape.0.clone(),
                    });
                }
            }
        }
    }
    for id in catalog.entries.keys() {
        if !expected_names.contains(id) {
            report.extra.push(id.clone());
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use thinfer_core::tensor::{Shape, StorageEncoding};
    use thinfer_core::weight::WeightEntry;

    fn catalog_from(expected: &[Expected]) -> WeightCatalog {
        let mut entries = HashMap::new();
        for e in expected {
            entries.insert(
                e.id.clone(),
                WeightEntry {
                    offset: 0,
                    size: 0,
                    encoding: Some(StorageEncoding::F16),
                    encoding_label: "F16".into(),
                    shape: Shape(e.shape.clone()),
                },
            );
        }
        WeightCatalog { entries }
    }

    #[test]
    fn expected_list_self_consistent() {
        let e = expected_weights();
        let cat = catalog_from(&e);
        let r = audit(&cat);
        assert!(r.ok(), "self-audit failed: {r:?}");
        assert_eq!(r.expected, e.len());
        assert!(r.extra.is_empty());
    }

    #[test]
    fn block_count_matches_config() {
        let names: Vec<_> = expected_weights().into_iter().map(|e| e.id.0).collect();
        let main_norms = names
            .iter()
            .filter(|n| n.starts_with("layers.") && n.ends_with(".attention_norm1.weight"))
            .count();
        assert_eq!(main_norms, config::N_LAYERS);
        let nr = names
            .iter()
            .filter(|n| n.starts_with("noise_refiner.") && n.ends_with(".attention_norm1.weight"))
            .count();
        assert_eq!(nr, config::N_REFINER_LAYERS);
        let cr = names
            .iter()
            .filter(|n| n.starts_with("context_refiner.") && n.ends_with(".attention_norm1.weight"))
            .count();
        assert_eq!(cr, config::N_REFINER_LAYERS);
        // context_refiner is not modulated -> no adaLN.
        let cr_adaln = names
            .iter()
            .filter(|n| n.starts_with("context_refiner.") && n.contains("adaLN_modulation"))
            .count();
        assert_eq!(cr_adaln, 0);
        let main_adaln = names
            .iter()
            .filter(|n| n.starts_with("layers.") && n.ends_with(".adaLN_modulation.0.weight"))
            .count();
        assert_eq!(main_adaln, config::N_LAYERS);
    }

    #[test]
    fn missing_keys_reported() {
        let mut e = expected_weights();
        let dropped = e.pop().unwrap();
        let cat = catalog_from(&e);
        let r = audit(&cat);
        assert!(!r.ok());
        assert_eq!(r.missing.len(), 1);
        assert_eq!(r.missing[0].0, dropped.id.0);
    }

    #[test]
    fn shape_mismatch_reported() {
        let mut e = expected_weights();
        e[0].shape = vec![999];
        let cat = catalog_from(&e);
        let r = audit(&cat);
        assert_eq!(r.shape_mismatches.len(), 1);
        assert_eq!(r.shape_mismatches[0].got, vec![999]);
    }

    #[test]
    fn extra_keys_reported_but_not_failing() {
        let mut e = expected_weights();
        e.push(Expected {
            id: WeightId("optimizer.state.foo".into()),
            shape: vec![1],
        });
        let cat = catalog_from(&e);
        let r = audit(&cat);
        assert!(r.ok());
        assert_eq!(r.extra.len(), 1);
    }
}
