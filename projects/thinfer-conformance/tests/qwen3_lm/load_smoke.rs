//! Load-smoke for the native prompt-rewriter LM (Qwen3-VL-8B-Instruct GGUF).
//!
//! GPU-free and forward-free: resolve the cached Q5_K_M GGUF, open it via
//! `GgufSource`, re-key to HF names with `qwen3vl_gguf_renames`, audit the
//! catalog (0 missing / 0 shape mismatches), then register every weight into a
//! `WeightResidency` and assert 36 layer handles come back. Registration is
//! catalog-only (no pipeline compile, no backend), so this needs no GPU.
//!
//! Run:
//!   `cargo test -p thinfer-conformance --features qwen3-lm --release \
//!    --test qwen3_lm load_smoke -- --nocapture --test-threads=1`

#![cfg(feature = "qwen3-lm")]

use thinfer_core::format::gguf::GgufSource;
use thinfer_core::format::union::RenamedSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::weight::WeightSource;
use thinfer_models::hunyuan::manifest::{MANIFEST, role};
use thinfer_models::qwen3_lm::Qwen3LmConfig;
use thinfer_models::qwen3_lm::generate::{audit, qwen3vl_gguf_renames, register_qwen3_lm};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

#[test]
fn load_smoke() {
    let cfg = Qwen3LmConfig::qwen3_vl_8b();
    let fr = MANIFEST
        .get(role::REWRITER_GGUF_8B_Q5_K_M)
        .expect("rewriter GGUF role in manifest");
    let Some(path) = cache::resolve(fr) else {
        eprintln!(
            "skipped[load_smoke]: {}/{} not in HF cache",
            fr.repo, fr.path
        );
        return;
    };

    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    rt.block_on(async {
        let opener = MmapFileOpener::new(&path)
            .await
            .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
        let gguf = GgufSource::open(opener).await.expect("parse rewriter gguf");
        let renamed = RenamedSource::with_passthrough(gguf, qwen3vl_gguf_renames(&cfg));

        // --- audit: every expected HF name present with the right shape ---
        let report = audit(renamed.catalog(), &cfg);
        assert!(
            report.ok(),
            "audit failed: {} missing, {} shape mismatches\n  missing: {:?}\n  mismatches: {:?}",
            report.missing.len(),
            report.shape_mismatches.len(),
            report.missing,
            report.shape_mismatches,
        );

        // --- register every weight (36 layers + final norm + lm_head) ---
        let budget = ResidencyBudget {
            ram_bytes: 4 << 30,
            vram_bytes: 8 << 30,
        };
        let residency = WeightResidency::new(renamed, budget);
        let handles = register_qwen3_lm(&residency, &cfg).expect("register qwen3 lm");
        assert_eq!(
            handles.layers.len(),
            cfg.n_layers,
            "expected {} layer handles",
            cfg.n_layers
        );

        let tensor_count = residency.source().catalog().entries.len();
        eprintln!(
            "load_smoke: PASS - {} gguf tensors, audit {} expected (0 missing / 0 mismatch), \
             registered {} layers + final_norm + lm_head",
            tensor_count,
            report.expected,
            handles.layers.len(),
        );
    });
}
