//! Qwen3.6-35B-A3B end-to-end perf bench; see [`crate::bench`].

#![cfg(feature = "llm-e2e")]

use thinfer_models::llm::manifest::{QWEN36_MANIFEST, role};
use thinfer_models::llm::qwen35::{Qwen35Config, Qwen35Model};

#[tokio::test(flavor = "current_thread")]
async fn qwen36_bench() {
    crate::bench::run(
        "qwen36_bench",
        &QWEN36_MANIFEST,
        |backend, source, budget, max_ctx| async move {
            let config =
                thinfer_native::cache::resolve(QWEN36_MANIFEST.get(role::CONFIG).expect("role"))
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
