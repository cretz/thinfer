//! Gemma 4 26B-A4B end-to-end perf bench; see [`crate::bench`].

#![cfg(feature = "llm-e2e")]

use thinfer_models::llm::gemma4::{Gemma4Config, Gemma4Model};
use thinfer_models::llm::manifest::{GEMMA4_MANIFEST, role};

#[tokio::test(flavor = "current_thread")]
async fn gemma4_bench() {
    crate::bench::run(
        "gemma4_bench",
        &GEMMA4_MANIFEST,
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
