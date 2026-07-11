//! KV-cached autoregressive generation parity for the native Qwen3-VL-8B
//! rewriter LM.
//!
//! Oracle (`gen_rewriter_ref.py`, llama-cpp-python, CPU) greedy-decodes N tokens
//! from a ChatML prompt and dumps `prompt_ids.bin` + `greedy_ids.bin`. These
//! tests then run the engine's KV-cached `generate` (prefill chunks -> per-token
//! decode against a persistent per-layer KV cache) and check:
//!
//!   A `kv_self_consistency`: engine greedy ids EQUAL the oracle greedy ids
//!     (exact match expected for greedy on the same GGUF). This is the KV /
//!     rope-position / masking correctness gate.
//!   B `kv_matches_full`: the KV-decode first token equals the non-KV `forward`
//!     argmax on the same prompt (self-consistency between the two code paths).
//!   C `kv_generate_detok`: the generated ids detokenize to a non-empty String
//!     that is prefix-consistent with the oracle's decoded caption.
//!
//! A SHORT system prompt keeps the prefill SDPA well under the ~2s TDR. Do NOT
//! feed the full 5760-token production system prompt here.
//!
//! Run (GPU):
//!   `cargo test -p thinfer-conformance --features qwen3-lm --release \
//!    --test qwen3_lm kv_ -- --nocapture --test-threads=1`

#![cfg(feature = "qwen3-lm")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::format::union::RenamedSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::tokenizer::Tokenizer;
use thinfer_core::workspace::Workspace;
use thinfer_models::hunyuan::manifest::{MANIFEST, role};
use thinfer_models::qwen3_lm::Qwen3LmConfig;
use thinfer_models::qwen3_lm::forward::{Qwen3LmGenerator, compile_pipelines};
use thinfer_models::qwen3_lm::generate::{Qwen3LmHandles, qwen3vl_gguf_renames, register_qwen3_lm};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;
use thinfer_native::tokenizer::HfTokenizer;

/// Concrete weight source for these tests: the rewriter GGUF re-keyed to HF names.
type RewriterSource = RenamedSource<GgufSource<MmapFileOpener>>;

/// Short system prompt (NOT the production rewriter system prompt).
const SYSTEM: &str = "You rewrite short prompts into vivid one-sentence video descriptions.";
const USER: &str = "a cat sleeps";
/// Greedy steps to generate/compare.
const N: usize = 20;
/// Qwen3 ChatML end token (`<|im_end|>`), the rewriter EOS.
const EOS_ID: u32 = 151645;

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn run_oracle_user(gguf: &Path, system_file: &Path, user: &str, out_dir: &Path, n: usize) {
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
            user,
            "--out",
            out_dir.to_str().unwrap(),
            "--n",
            &n.to_string(),
        ])
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "rewriter oracle failed");
}

/// Everything the tests need after resolving the GGUF, running the oracle, and
/// building the engine. Returns `None` when the GGUF is not in the HF cache
/// (tests skip cleanly).
struct Setup {
    backend: Arc<WgpuBackend>,
    pipelines: thinfer_models::qwen3_lm::forward::Qwen3LmPipelines,
    residency: WeightResidency<RewriterSource>,
    handles: Qwen3LmHandles,
    workspace: Workspace<WgpuBackend>,
    generator: Qwen3LmGenerator,
    prompt_ids: Vec<u32>,
    greedy_ids: Vec<u32>,
}

async fn setup(tag: &str) -> Option<Setup> {
    setup_with(tag, SYSTEM, USER, N).await
}

async fn setup_with(tag: &str, system: &str, user: &str, n: usize) -> Option<Setup> {
    let fr = MANIFEST
        .get(role::REWRITER_GGUF_8B_Q5_K_M)
        .expect("rewriter GGUF role in manifest");
    let Some(gguf_path) = cache::resolve(fr) else {
        eprintln!("skipped[{tag}]: {}/{} not in HF cache", fr.repo, fr.path);
        return None;
    };

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("qwen3_lm_{tag}"));
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let system_file = tmp.join("system.txt");
    std::fs::write(&system_file, system).expect("write system prompt");
    run_oracle_user(&gguf_path, &system_file, user, &tmp, n);

    let prompt_ids = read_u32(&tmp.join("prompt_ids.bin"));
    let greedy_ids = read_u32(&tmp.join("greedy_ids.bin"));
    assert!(!prompt_ids.is_empty(), "empty prompt_ids from oracle");
    assert!(!greedy_ids.is_empty(), "empty greedy_ids from oracle");

    Some(
        build_engine(
            &gguf_path,
            Qwen3LmConfig::qwen3_vl_8b(),
            prompt_ids,
            greedy_ids,
            n,
        )
        .await,
    )
}

/// Build the engine (residency + pipelines + generator) for a given GGUF and
/// pre-tokenized prompt. Split out of `setup_with` so a perf probe can supply
/// prompt ids WITHOUT the (slow CPU) oracle. `greedy_ids` may be empty (perf).
async fn build_engine(
    gguf_path: &Path,
    lm_cfg: Qwen3LmConfig,
    prompt_ids: Vec<u32>,
    greedy_ids: Vec<u32>,
    n: usize,
) -> Setup {
    let opener = MmapFileOpener::new(gguf_path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", gguf_path.display()));
    let gguf = GgufSource::open(opener).await.expect("parse rewriter gguf");
    let renamed = RenamedSource::with_passthrough(gguf, qwen3vl_gguf_renames(&lm_cfg));
    // VRAM weight budget (GiB), env-overridable to probe the resident-vs-stream
    // boundary at long context. Default 8. A budget below the device size leaves
    // device slack for the KV cache + activations so the arbiter keeps weights
    // resident instead of evicting them under KV growth.
    let vram_gib: u64 = std::env::var("THINFER_QWEN3LM_VRAM_GB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let budget = ResidencyBudget {
        ram_bytes: 4 << 30,
        vram_bytes: vram_gib << 30,
    };
    let residency = WeightResidency::new(renamed, budget);
    let handles = register_qwen3_lm(&residency, &lm_cfg).expect("register qwen3 lm");

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

    let max_seq = prompt_ids.len() + n + 4;
    let generator = Qwen3LmGenerator::new(lm_cfg, max_seq);

    Setup {
        backend,
        pipelines,
        residency,
        handles,
        workspace,
        generator,
        prompt_ids,
        greedy_ids,
    }
}

/// A: engine KV-cached greedy ids must equal the oracle greedy ids exactly.
#[tokio::test(flavor = "current_thread")]
async fn kv_self_consistency() {
    let Some(s) = setup("kv_self").await else {
        return;
    };
    let n = s.greedy_ids.len().min(N);
    eprintln!(
        "kv_self_consistency: prompt={} tokens, oracle greedy={} tokens, comparing {n}",
        s.prompt_ids.len(),
        s.greedy_ids.len(),
    );

    let engine_ids = s
        .generator
        .generate(
            &s.backend,
            &s.pipelines,
            &s.residency,
            &s.workspace,
            &s.handles,
            s.residency.source(),
            &s.prompt_ids,
            n,
            EOS_ID,
        )
        .await
        .expect("engine generate");

    let m = engine_ids.len().min(n);
    let mut matches = 0usize;
    let mut first_div: Option<usize> = None;
    for (i, (&eng, &orc)) in engine_ids
        .iter()
        .zip(s.greedy_ids.iter())
        .take(m)
        .enumerate()
    {
        let hit = eng == orc;
        if hit {
            matches += 1;
        } else if first_div.is_none() {
            first_div = Some(i);
        }
        eprintln!(
            "  step {i:02}: engine={eng} oracle={orc} {}",
            if hit { "OK" } else { "MISMATCH" }
        );
    }
    eprintln!(
        "kv_self_consistency: {matches}/{n} exact greedy matches, first divergence {first_div:?} \
         (engine produced {} ids)",
        engine_ids.len(),
    );

    // Greedy on the same GGUF must reproduce the oracle exactly. A KV / rope /
    // masking bug shows up as an early divergence. Do NOT loosen this: require
    // ALL compared tokens to match.
    assert_eq!(
        engine_ids.len(),
        n,
        "engine stopped early ({} < {n}); first divergence {first_div:?}",
        engine_ids.len(),
    );
    assert_eq!(
        matches, n,
        "KV greedy diverged from oracle: {matches}/{n}, first at {first_div:?}",
    );
}

/// B: KV-decode first token == non-KV `forward` argmax on the same prompt.
#[tokio::test(flavor = "current_thread")]
async fn kv_matches_full() {
    let Some(s) = setup("kv_full").await else {
        return;
    };

    let logits = s
        .generator
        .forward(
            &s.backend,
            &s.pipelines,
            &s.residency,
            &s.workspace,
            &s.handles,
            s.residency.source(),
            &s.prompt_ids,
        )
        .await
        .expect("engine forward");
    let full_argmax = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .expect("non-empty logits");

    let engine_ids = s
        .generator
        .generate(
            &s.backend,
            &s.pipelines,
            &s.residency,
            &s.workspace,
            &s.handles,
            s.residency.source(),
            &s.prompt_ids,
            1,
            EOS_ID,
        )
        .await
        .expect("engine generate");

    eprintln!(
        "kv_matches_full: forward argmax={full_argmax}, kv first token={:?}",
        engine_ids.first()
    );
    assert_eq!(
        engine_ids.first().copied(),
        Some(full_argmax),
        "KV first token disagrees with full-forward argmax",
    );
}

/// C: generated ids detokenize to a non-empty, prefix-consistent caption.
#[tokio::test(flavor = "current_thread")]
async fn kv_generate_detok() {
    let Some(s) = setup("kv_detok").await else {
        return;
    };

    // Resolve the rewriter tokenizer (skip cleanly if uncached).
    let tok_fr = MANIFEST
        .get(role::REWRITER_TOKENIZER)
        .expect("rewriter tokenizer role in manifest");
    let Some(tok_path) = cache::resolve(tok_fr) else {
        eprintln!(
            "skipped[kv_detok]: tokenizer {}/{} not in HF cache",
            tok_fr.repo, tok_fr.path
        );
        return;
    };
    let tokenizer = HfTokenizer::from_path(&tok_path)
        .await
        .expect("load rewriter tokenizer");

    let n = s.greedy_ids.len().min(N);
    let engine_ids = s
        .generator
        .generate(
            &s.backend,
            &s.pipelines,
            &s.residency,
            &s.workspace,
            &s.handles,
            s.residency.source(),
            &s.prompt_ids,
            n,
            EOS_ID,
        )
        .await
        .expect("engine generate");

    let engine_text = tokenizer
        .decode(&engine_ids, true)
        .expect("detok engine ids");
    let oracle_text = tokenizer
        .decode(&s.greedy_ids[..n], true)
        .expect("detok oracle ids");
    eprintln!(
        "kv_generate_detok: engine={engine_text:?}\n                   oracle={oracle_text:?}"
    );

    assert!(
        !engine_text.trim().is_empty(),
        "engine detok produced empty string",
    );
    // Greedy-exact means the strings should match; assert prefix-consistency to
    // tolerate a trailing partial multi-byte token if the two id lists differ in
    // length by the final token.
    let short = engine_text.len().min(oracle_text.len());
    assert!(
        engine_text.is_char_boundary(short)
            && oracle_text.is_char_boundary(short)
            && engine_text[..short] == oracle_text[..short],
        "engine caption not prefix-consistent with oracle:\n  engine={engine_text:?}\n  oracle={oracle_text:?}",
    );
}

/// Diagnostic (`--ignored`): KV-cached generate must agree with the non-KV
/// `forward` (teacher-forced) across a MULTI-CHUNK prefill. Uses a medium system
/// prompt (~300 tokens > PREFILL_CHUNK=256) so the `past_len>0` chunk-boundary
/// masked-SDPA path is exercised. Both paths are ours, so agreement must be
/// EXACT; a mismatch localizes a chunked-prefill bug (vs llama.cpp precision).
#[tokio::test(flavor = "current_thread")]
#[ignore = "diagnostic: KV-vs-full over a multi-chunk prefill"]
async fn kv_vs_full_multichunk() {
    // >256 tokens: repeat the short system prompt to cross the 256 chunk line
    // (so prefill spans multiple chunks and the past_len>0 path runs).
    let system = SYSTEM.repeat(40);
    let steps = 8usize;
    let Some(s) = setup_with("kv_multichunk", &system, USER, steps).await else {
        return;
    };
    eprintln!(
        "kv_vs_full_multichunk: prompt={} tokens (chunks of 256)",
        s.prompt_ids.len(),
    );

    let kv_ids = s
        .generator
        .generate(
            &s.backend,
            &s.pipelines,
            &s.residency,
            &s.workspace,
            &s.handles,
            s.residency.source(),
            &s.prompt_ids,
            steps,
            EOS_ID,
        )
        .await
        .expect("engine generate");

    // Teacher-force the non-KV forward with the KV tokens; per-step argmax must
    // equal the KV token.
    let mut tokens = s.prompt_ids.clone();
    for (i, &kv_tok) in kv_ids.iter().enumerate() {
        let logits = s
            .generator
            .forward(
                &s.backend,
                &s.pipelines,
                &s.residency,
                &s.workspace,
                &s.handles,
                s.residency.source(),
                &tokens,
            )
            .await
            .expect("engine forward");
        let full = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .expect("non-empty logits");
        eprintln!(
            "  step {i:02}: kv={kv_tok} full={full} {}",
            if kv_tok == full { "OK" } else { "MISMATCH" }
        );
        assert_eq!(
            kv_tok, full,
            "KV-vs-full mismatch at step {i} (chunked prefill bug)",
        );
        tokens.push(kv_tok);
    }
    eprintln!("kv_vs_full_multichunk: {} steps agree", kv_ids.len());
}

/// Opt-in (`--ignored`) manual sanity run against the REAL ~5760-token
/// production rewriter system prompt (chunked prefill, no TDR). Confirms the
/// engine reproduces the oracle's caption for the canonical "A dog dancing on a
/// broadway stage" prompt. Ignored by default (heavy: full prefill + 40-token
/// decode + a slow CPU oracle); run explicitly:
///   `cargo test -p thinfer-conformance --features qwen3-lm --release \
///    --test qwen3_lm kv_real_system_prompt -- --ignored --nocapture --test-threads=1`
#[tokio::test(flavor = "current_thread")]
#[ignore = "manual: loads the full production system prompt (heavy)"]
async fn kv_real_system_prompt() {
    let real_n: usize = std::env::var("THINFER_QWEN3LM_MAXNEW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let system = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../thinfer-app/assets/t2v_rewrite_system_prompt.txt"),
    )
    .expect("read production system prompt");
    let user = "A dog dancing on a broadway stage";

    let Some(s) = setup_with("kv_real", &system, user, real_n).await else {
        return;
    };
    let tok_fr = MANIFEST
        .get(role::REWRITER_TOKENIZER)
        .expect("rewriter tokenizer role in manifest");
    let Some(tok_path) = cache::resolve(tok_fr) else {
        eprintln!("skipped[kv_real]: tokenizer not in HF cache");
        return;
    };
    let tokenizer = HfTokenizer::from_path(&tok_path)
        .await
        .expect("load rewriter tokenizer");

    let n = s.greedy_ids.len().min(real_n);
    eprintln!(
        "kv_real_system_prompt: prompt={} tokens, oracle greedy={} tokens",
        s.prompt_ids.len(),
        s.greedy_ids.len(),
    );

    let engine_ids = s
        .generator
        .generate(
            &s.backend,
            &s.pipelines,
            &s.residency,
            &s.workspace,
            &s.handles,
            s.residency.source(),
            &s.prompt_ids,
            n,
            EOS_ID,
        )
        .await
        .expect("engine generate");

    let m = engine_ids.len().min(n);
    let matches = engine_ids
        .iter()
        .zip(s.greedy_ids.iter())
        .take(m)
        .filter(|(a, b)| a == b)
        .count();
    let engine_text = tokenizer.decode(&engine_ids, true).expect("detok engine");
    let oracle_text = tokenizer
        .decode(&s.greedy_ids[..n], true)
        .expect("detok oracle");
    eprintln!(
        "kv_real_system_prompt: {matches}/{m} token matches\n  engine20={:?}\n  oracle20={:?}\n  engine={engine_text:?}\n  oracle={oracle_text:?}",
        &engine_ids[..engine_ids.len().min(20)],
        &s.greedy_ids[..s.greedy_ids.len().min(20)],
    );

    // NOTE (report, not a hard gate): exact greedy agreement with the llama.cpp
    // CPU oracle over a ~5760-token context is precision-bound, NOT a KV-cache
    // correctness question. The bf16 activation path (load-bearing for Qwen3's
    // massive-activation outliers) diverges from llama.cpp's f32 CPU KV cache by
    // sub-ulp amounts that a chaotic greedy argmax amplifies into an early token
    // flip at long context; the engine still emits a coherent, on-topic caption.
    // The KV-cache path itself is proven bit-exact against the non-KV `forward`
    // by `kv_vs_full_multichunk` (chunked prefill) and `kv_matches_full`. So this
    // manual run only sanity-checks that generation runs to completion and yields
    // a plausible caption; it does not assert oracle-exactness.
    assert!(
        !engine_text.trim().is_empty() && engine_ids.first() == s.greedy_ids.first(),
        "engine produced no caption / wrong first token on the real prompt",
    );
}

/// Engine-only perf probe (`--ignored`, NO oracle): loads the REAL ~5760-token
/// production system prompt, builds the ChatML prompt, tokenizes it locally, and
/// times prefill + decode. Because there is no slow CPU oracle in the loop, a
/// VRAM-budget sweep is cheap. Set `THINFER_QWEN3LM_DIAG=1` for the prefill /
/// decode tok/s prints, `THINFER_QWEN3LM_VRAM_GB` to probe the resident-vs-stream
/// boundary (a budget below the 8GB device leaves slack for the KV cache so the
/// 5.85GB weights stay resident), and `THINFER_QWEN3LM_MAXNEW` for the decode
/// length (default 24 = enough for steady-state decode tok/s). Run:
///   `THINFER_QWEN3LM_DIAG=1 THINFER_QWEN3LM_VRAM_GB=6 cargo test -p \
///    thinfer-conformance --features qwen3-lm --release --test qwen3_lm \
///    kv_real_perf -- --ignored --nocapture --test-threads=1`
#[tokio::test(flavor = "current_thread")]
#[ignore = "manual perf: real system prompt, engine-only (no oracle)"]
async fn kv_real_perf() {
    let max_new: usize = std::env::var("THINFER_QWEN3LM_MAXNEW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let fr = MANIFEST
        .get(role::REWRITER_GGUF_8B_Q5_K_M)
        .expect("rewriter GGUF role in manifest");
    let Some(gguf_path) = cache::resolve(fr) else {
        eprintln!(
            "skipped[kv_real_perf]: {}/{} not in HF cache",
            fr.repo, fr.path
        );
        return;
    };
    let tok_fr = MANIFEST
        .get(role::REWRITER_TOKENIZER)
        .expect("rewriter tokenizer role in manifest");
    let Some(tok_path) = cache::resolve(tok_fr) else {
        eprintln!("skipped[kv_real_perf]: tokenizer not in HF cache");
        return;
    };
    let tokenizer = HfTokenizer::from_path(&tok_path)
        .await
        .expect("load rewriter tokenizer");

    let system = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../thinfer-app/assets/t2v_rewrite_system_prompt.txt"),
    )
    .expect("read production system prompt");
    let user = "A dog dancing on a broadway stage";
    // Qwen3 ChatML with the generation prompt (matches the oracle template that
    // `kv_self_consistency` validated: system / user / assistant-open).
    let chat = format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    );
    let prompt_ids = tokenizer
        .encode(&chat, false)
        .expect("tokenize chat prompt");
    eprintln!(
        "kv_real_perf: prompt={} tokens, max_new={max_new}",
        prompt_ids.len()
    );

    // Per-pipeline `gpu_ms` rollup (THINFER_TRACE=1): localizes the prefill /
    // decode cost by op (matmul_* vs sdpa). Prefill dominates the aggregate, so
    // run with a small MAXNEW to read the prefill breakdown.
    let trace = thinfer_core::trace::init_from_env();
    let s = build_engine(
        &gguf_path,
        Qwen3LmConfig::qwen3_vl_8b(),
        prompt_ids,
        Vec::new(),
        max_new,
    )
    .await;
    let engine_ids = s
        .generator
        .generate(
            &s.backend,
            &s.pipelines,
            &s.residency,
            &s.workspace,
            &s.handles,
            s.residency.source(),
            &s.prompt_ids,
            max_new,
            EOS_ID,
        )
        .await
        .expect("engine generate");
    let text = tokenizer
        .decode(&engine_ids, true)
        .expect("detok engine ids");
    eprintln!(
        "kv_real_perf: generated {} tokens\n  caption={text:?}",
        engine_ids.len()
    );
    assert!(!text.trim().is_empty(), "engine produced empty caption");
    if let Some(h) = trace.as_ref() {
        let mut buf = Vec::new();
        h.dump(&mut buf).ok();
        eprintln!("{}", String::from_utf8_lossy(&buf));
    }
}

/// Manual (`--ignored`): run BOTH rewriter sizes (4B Fast + 8B Full) on the real
/// production system prompt + a canonical short user prompt, print each caption
/// and its wall-clock timing, and assert both are non-empty + structured. This is
/// the quality/perf gate for the Fast (4B, TIED-lm_head) default vs the Full (8B,
/// untied) option -- it also exercises the tied-embedding forward path on GPU
/// (the automated parity tests only cover the 8B). Honors
/// `THINFER_QWEN3LM_VRAM_GB` (the deployer's budget), `THINFER_QWEN3LM_MAXNEW`
/// (default 200, a realistic caption length), `THINFER_QWEN3LM_USER` (probe an
/// arbitrary prompt), and `THINFER_QWEN3LM_SIZE=4b|8b` (run one size). Run:
///   `THINFER_QWEN3LM_VRAM_GB=5 cargo test -p thinfer-conformance --features \
///    qwen3-lm --release --test qwen3_lm rewriter_caption_compare -- --ignored \
///    --nocapture --test-threads=1`
#[tokio::test(flavor = "current_thread")]
#[ignore = "manual quality/perf gate: 4B vs 8B caption + timing"]
async fn rewriter_caption_compare() {
    let max_new: usize = std::env::var("THINFER_QWEN3LM_MAXNEW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let tok_fr = MANIFEST
        .get(role::REWRITER_TOKENIZER)
        .expect("rewriter tokenizer role in manifest");
    let Some(tok_path) = cache::resolve(tok_fr) else {
        eprintln!("skipped[caption_compare]: tokenizer not in HF cache");
        return;
    };
    let tokenizer = HfTokenizer::from_path(&tok_path)
        .await
        .expect("load rewriter tokenizer");
    let system = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../thinfer-app/assets/t2v_rewrite_system_prompt.txt"),
    )
    .expect("read production system prompt");
    // Override the user prompt (`THINFER_QWEN3LM_USER`) to probe an arbitrary
    // prompt's caption (e.g. a multi-subject one that surfaced a coherence bug).
    let user = std::env::var("THINFER_QWEN3LM_USER")
        .unwrap_or_else(|_| "A dog dancing on a broadway stage".to_string());
    eprintln!("caption_compare: user={user:?}");
    let chat = format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    );
    let prompt_ids = tokenizer
        .encode(&chat, false)
        .expect("tokenize chat prompt");

    // (label, size-tag, manifest role, arch config) per rewriter size.
    // `THINFER_QWEN3LM_SIZE=4b|8b` runs only that size (default: both).
    let size = std::env::var("THINFER_QWEN3LM_SIZE").unwrap_or_default();
    let variants = [
        (
            "Fast(4B)",
            "4b",
            role::REWRITER_GGUF_4B_Q5_K_M,
            Qwen3LmConfig::qwen3_vl_4b(),
        ),
        (
            "Full(8B)",
            "8b",
            role::REWRITER_GGUF_8B_Q5_K_M,
            Qwen3LmConfig::qwen3_vl_8b(),
        ),
    ];
    for (label, tag, gguf_role, lm_cfg) in variants {
        if !size.is_empty() && size != tag {
            continue;
        }
        let fr = MANIFEST.get(gguf_role).expect("rewriter GGUF role");
        let Some(gguf_path) = cache::resolve(fr) else {
            eprintln!("skipped[{label}]: {}/{} not in HF cache", fr.repo, fr.path);
            continue;
        };
        let s = build_engine(&gguf_path, lm_cfg, prompt_ids.clone(), Vec::new(), max_new).await;
        let t0 = std::time::Instant::now();
        let ids = s
            .generator
            .generate(
                &s.backend,
                &s.pipelines,
                &s.residency,
                &s.workspace,
                &s.handles,
                s.residency.source(),
                &s.prompt_ids,
                max_new,
                EOS_ID,
            )
            .await
            .expect("engine generate");
        let dt = t0.elapsed().as_secs_f64();
        let text = tokenizer.decode(&ids, true).expect("detok");
        eprintln!(
            "\n=== {label}: {} prompt tok -> {} gen tok in {dt:.1}s ({:.2} tok/s) ===\n{text}\n",
            prompt_ids.len(),
            ids.len(),
            ids.len() as f64 / dt,
        );
        assert!(!text.trim().is_empty(), "{label} produced empty caption");
        // A real structured caption is many words, not a single-token echo.
        assert!(
            text.split_whitespace().count() >= 12,
            "{label} caption suspiciously short: {text:?}"
        );
    }
}
