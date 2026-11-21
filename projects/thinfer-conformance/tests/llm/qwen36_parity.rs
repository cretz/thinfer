//! Qwen3.6-35B-A3B parity vs the transformers `Qwen3_5Moe*` modules
//! (`qwen35moe_ref.py`) over the same GGUF; see [`crate::parity`].

#![cfg(feature = "llm-e2e")]

use thinfer_models::llm::manifest::QWEN36_MANIFEST;
use thinfer_models::llm::qwen35::{Qwen35Config, Qwen35Model};

use crate::parity::{self, Tol};

/// Measured 2026-11-19 (12-token prompt, 40 layers): layer rel 0.4-3%
/// growing with depth, with single-row jumps to ~7-8% where a near-tied 8th
/// expert flips (layer 7 last row: expert 254 vs 108 at weight ~0.074,
/// decided by the reference's bf16 rounding); logits cos 0.99968 and the same
/// top-1 for prefill and prefill+decode.
const TOL: Tol = Tol {
    layer_rel: 0.10,
    logit_cos: 0.999,
};

#[tokio::test(flavor = "current_thread")]
async fn qwen36_parity() {
    parity::run(
        "qwen36_parity",
        &QWEN36_MANIFEST,
        "qwen35moe_ref",
        TOL,
        |backend, source, budget, max_ctx| async move {
            let config = thinfer_native::cache::resolve(
                QWEN36_MANIFEST
                    .get(thinfer_models::llm::manifest::role::CONFIG)
                    .expect("role"),
            )
            .expect("config cached");
            let cfg = Qwen35Config::from_json(
                &serde_json::from_str(&std::fs::read_to_string(config).unwrap()).unwrap(),
            )
            .expect("config");
            Qwen35Model::load(&backend, source, cfg, budget, max_ctx).expect("load")
        },
    )
    .await;
}
