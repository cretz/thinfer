//! Teacher-forced logits parity for the native Qwen3-VL-8B rewriter LM.
//!
//! Oracle (`gen_rewriter_ref.py`, llama-cpp-python, CPU) greedy-decodes N tokens
//! from a ChatML prompt and dumps `prompt_ids.bin` + `greedy_ids.bin`. This test
//! then teacher-forces the engine's full (non-KV-cached) causal `forward`: for
//! step i it runs the engine over `prompt_ids ++ greedy_ids[0..i]` and checks
//! that the engine's argmax logit equals `greedy_ids[i]`. Each step is
//! independent (no compounding), so a mismatch localizes to that step's logits.
//!
//! A SHORT system prompt keeps the full-forward SDPA well under the ~2s TDR: the
//! un-chunked causal `SdpaF32` is O(seq^2); a few-hundred-token total sequence
//! is plenty to validate numerics. Do NOT feed the full 5760-token production
//! system prompt here.
//!
//! Run (GPU; the parity judgment is run separately):
//!   `cargo test -p thinfer-conformance --features qwen3-lm --release \
//!    --test qwen3_lm logits_parity -- --nocapture --test-threads=1`

#![cfg(feature = "qwen3-lm")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::format::union::RenamedSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::hunyuan::manifest::{MANIFEST, role};
use thinfer_models::qwen3_lm::Qwen3LmConfig;
use thinfer_models::qwen3_lm::forward::{Qwen3LmGenerator, compile_pipelines};
use thinfer_models::qwen3_lm::generate::{qwen3vl_gguf_renames, register_qwen3_lm};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

/// Short system prompt for the full-forward test (NOT the production rewriter
/// system prompt, which is ~5760 tokens and would risk a TDR here).
const SYSTEM: &str = "You rewrite short prompts into vivid one-sentence video descriptions.";
const USER: &str = "a cat sleeps";
/// Greedy steps to teacher-force.
const N: usize = 16;

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn run_oracle(gguf: &Path, system_file: &Path, out_dir: &Path, n: usize) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let status = Command::new("uv")
        .args([
            "run",
            "--directory",
            py_dir.to_str().unwrap(),
            "--with",
            "llama-cpp-python",
            "python",
            "-m",
            "thinfer_pytorch_ref.hunyuan.gen_rewriter_ref",
            "--gguf",
            gguf.to_str().unwrap(),
            "--system-file",
            system_file.to_str().unwrap(),
            "--user",
            USER,
            "--out",
            out_dir.to_str().unwrap(),
            "--n",
            &n.to_string(),
        ])
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "rewriter oracle failed");
}

#[tokio::test(flavor = "current_thread")]
async fn logits_parity() {
    // --- resolve the Q5_K_M GGUF from the HF cache (skip cleanly if absent) ---
    let lm_cfg = Qwen3LmConfig::qwen3_vl_8b();
    let fr = MANIFEST
        .get(role::REWRITER_GGUF_8B_Q5_K_M)
        .expect("rewriter GGUF role in manifest");
    let Some(gguf_path) = cache::resolve(fr) else {
        eprintln!(
            "skipped[logits_parity]: {}/{} not in HF cache",
            fr.repo, fr.path
        );
        return;
    };

    // --- oracle: write the short system prompt, greedy-decode N tokens ---
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("qwen3_lm_logits");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let system_file = tmp.join("system.txt");
    std::fs::write(&system_file, SYSTEM).expect("write system prompt");
    run_oracle(&gguf_path, &system_file, &tmp, N);

    let prompt_ids = read_u32(&tmp.join("prompt_ids.bin"));
    let greedy_ids = read_u32(&tmp.join("greedy_ids.bin"));
    assert!(!prompt_ids.is_empty(), "empty prompt_ids from oracle");
    assert!(!greedy_ids.is_empty(), "empty greedy_ids from oracle");
    let n = greedy_ids.len().min(N);
    eprintln!(
        "logits_parity: prompt={} tokens, greedy={} tokens, teacher-forcing {n} steps",
        prompt_ids.len(),
        greedy_ids.len(),
    );

    // --- engine: GGUF -> HF names -> residency, register all weights ---
    let opener = MmapFileOpener::new(&gguf_path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", gguf_path.display()));
    let gguf = GgufSource::open(opener).await.expect("parse rewriter gguf");
    let renamed = RenamedSource::with_passthrough(gguf, qwen3vl_gguf_renames(&lm_cfg));
    let budget = ResidencyBudget {
        ram_bytes: 4 << 30,
        vram_bytes: 8 << 30,
    };
    let residency = WeightResidency::new(renamed, budget);
    let handles = register_qwen3_lm(&residency, &lm_cfg).expect("register qwen3 lm");

    // --- backend + pipelines ---
    let cfg = WgpuConfig {
        power_preference: match std::env::var("THINFER_POWER_PREF")
            .ok()
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("low" | "lowpower" | "integrated") => PowerPreference::LowPower,
            Some("none") => PowerPreference::None,
            _ => PowerPreference::HighPerformance,
        },
        timestamps: std::env::var("THINFER_TRACE").is_ok(),
        disable_coopmat: std::env::var("THINFER_NO_COOPMAT").is_ok(),
    };
    let backend = Arc::new(
        WgpuBackend::new_with_config(cfg)
            .await
            .expect("wgpu adapter unavailable for tests"),
    );
    let pipelines = compile_pipelines(&backend)
        .await
        .expect("compile rewriter pipelines");
    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    let max_seq = prompt_ids.len() + n + 2;
    let generator = Qwen3LmGenerator::new(lm_cfg, max_seq);

    // --- teacher-forced per-step parity ---
    let mut tokens = prompt_ids.clone();
    let mut matches = 0usize;
    let mut first_mismatch: Option<usize> = None;
    for (i, &oracle_tok) in greedy_ids.iter().take(n).enumerate() {
        let logits = generator
            .forward(
                &backend,
                &pipelines,
                &residency,
                &workspace,
                &handles,
                residency.source(),
                &tokens,
            )
            .await
            .expect("engine forward");
        assert_eq!(logits.len(), lm_cfg.vocab, "logits width");

        let (argmax, &argmax_logit) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("non-empty logits");
        let oracle_logit = logits[oracle_tok as usize];
        let hit = argmax as u32 == oracle_tok;
        if hit {
            matches += 1;
        } else if first_mismatch.is_none() {
            first_mismatch = Some(i);
        }
        eprintln!(
            "step {i:02}: engine_argmax={argmax} (logit {argmax_logit:+.4}) \
             oracle_tok={oracle_tok} (engine logit {oracle_logit:+.4}) {}",
            if hit { "OK" } else { "MISMATCH" }
        );
        if !hit {
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            let top5: Vec<String> = idx
                .iter()
                .take(5)
                .map(|&t| format!("{t}={:+.4}", logits[t]))
                .collect();
            eprintln!("         top5: {}", top5.join(" "));
        }

        // Teacher forcing: advance with the ORACLE token, not the engine argmax.
        tokens.push(oracle_tok);
    }

    eprintln!(
        "logits_parity: {matches}/{n} argmax matches, first mismatch {:?}",
        first_mismatch
    );

    // Provisional threshold pending the first GPU run. Q5_K/Q6_K weight quant +
    // bf16 acts vs the llama.cpp oracle should agree on argmax at nearly every
    // step; require a high match rate but tolerate a rare boundary flip. Do NOT
    // loosen numeric tolerances to force a pass; re-measure and re-document if
    // the first GPU run lands lower.
    let threshold = n * 3 / 4;
    assert!(
        matches >= threshold,
        "logits parity too low: {matches}/{n} (need >= {threshold}); first mismatch {:?}",
        first_mismatch
    );
}
