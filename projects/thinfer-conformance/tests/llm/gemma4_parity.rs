//! Gemma 4 26B-A4B parity vs the transformers `Gemma4TextDecoderLayer`
//! modules (`gemma4_ref.py`) over the same GGUF; see [`crate::parity`].

#![cfg(feature = "llm-e2e")]

use thinfer_models::llm::gemma4::{Gemma4Config, Gemma4Model};
use thinfer_models::llm::manifest::{GEMMA4_MANIFEST, role};

use crate::parity::{self, Tol};

/// Measured 2026-11-19 (13-token prompt with BOS, 30 layers): rows without a
/// routing flip stay at 0.4-1.3% through every layer; flipped rows step up
/// at the flip (verified: layer 4 row 2 expert 122 vs 8 at 8th weight
/// 0.0879 / 0.0886, layer 8 row 5, layer 10 rows 6 and 12, all within
/// ~0.002) and reach 30-40% by layer 27; median row <= 7%. Logits cos
/// 0.99985, same top-1 (" Berlin") for prefill and prefill+decode.
const TOL: Tol = Tol {
    layer_rel: 0.10,
    logit_cos: 0.999,
};

#[tokio::test(flavor = "current_thread")]
async fn gemma4_parity() {
    parity::run(
        "gemma4_parity",
        &GEMMA4_MANIFEST,
        "gemma4_ref",
        TOL,
        |backend, source, budget, max_ctx| async move {
            let config =
                thinfer_native::cache::resolve(GEMMA4_MANIFEST.get(role::CONFIG).expect("role"))
                    .expect("config cached");
            let cfg = Gemma4Config::from_json(
                &serde_json::from_str(&std::fs::read_to_string(config).unwrap()).unwrap(),
            )
            .expect("config");
            Gemma4Model::load(&backend, source, cfg, budget, max_ctx)
                .await
                .expect("load")
        },
    )
    .await;
}
