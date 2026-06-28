//! Qwen-Image text-encoder parity: the engine Qwen2.5-VL-7B encoder (Q8_0
//! weights from the GGUF + bf16 acts + DP4A/flash-sdpa) vs a bf16 reference
//! dequantized from the SAME GGUF (`gen_encoder_ref.py`), fed the SAME tokens.
//!
//! Both sides start from identical Q8_0 bytes, so the band is dominated by the
//! engine's bf16-act rounding (plus the QKV-bias add + no-qk-norm deltas vs the
//! Qwen3 path this block was adapted from). Compares the final
//! `hidden_states[-1]` (post `output_norm`) -- the DiT conditioning surface.

#![cfg(feature = "qwen-image-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::format::union::RenamedSource;
use thinfer_core::ops::{ActDtype, WeightDtype, WgslConfig};
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::quant::QuantKind;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::common::block::{BlockPipelines, BlockWgslConfigs, DenseActSites};
use thinfer_models::qwen_image::manifest::{self, role};
use thinfer_models::qwen_image::text_encoder::{
    TextEncoder, qwen2vl_gguf_renames, register_handles,
};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, read_u32, report};

const PROMPT: &str = "a red apple on a wooden table";

// bf16-act engine vs bf16 reference, same Q8_0 weights. Re-measure + re-document
// on a config change; do not loosen to go green.
const SLOPE_TOL: f64 = 0.03;
const REL_TOL: f64 = 0.06;

#[tokio::test(flavor = "current_thread")]
async fn encoder_parity() {
    let _trace = thinfer_core::trace::init_from_env();

    let fr = manifest::MANIFEST
        .get(role::ENCODER_GGUF_Q8_0)
        .expect("encoder gguf role");
    let tok_fr = manifest::MANIFEST
        .get(role::TOKENIZER)
        .expect("tokenizer role");
    let (Some(gguf_path), Some(tok_path)) = (cache::resolve(fr), cache::resolve(tok_fr)) else {
        eprintln!("skipped[encoder_parity]: encoder gguf or tokenizer not in HF cache");
        return;
    };
    let tok_dir = tok_path.parent().expect("tokenizer dir");

    // --- python reference (owns tokenization; writes token_ids + hidden) ---
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("qwen_encoder_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("prompt.txt");
    let hidden_path = tmp.join("hidden.bin");
    let cached =
        hidden_path.exists() && std::fs::read_to_string(&marker).is_ok_and(|p| p == PROMPT);
    if cached {
        eprintln!(
            "encoder-parity: reusing cached pyref dumps ({})",
            tmp.display()
        );
    } else {
        for ent in std::fs::read_dir(&tmp).into_iter().flatten().flatten() {
            let _ = std::fs::remove_file(ent.path());
        }
        run_python_ref(&gguf_path, tok_dir, &tmp);
        std::fs::write(&marker, PROMPT).expect("write marker");
    }

    let ids = read_u32(&tmp.join("token_ids.bin"));
    let seq = ids.len();
    assert!(seq > 0, "pyref produced no tokens");
    let meta = std::fs::read_to_string(tmp.join("meta.txt")).expect("meta.txt");
    let m: Vec<usize> = meta
        .split_whitespace()
        .take(2)
        .map(|s| s.parse().expect("meta int"))
        .collect();
    assert_eq!(m[0], seq, "pyref seq mismatch");
    let hidden_dim = m[1];
    eprintln!("encoder-parity: {seq} tokens, hidden {hidden_dim}");

    // --- engine encoder: GGUF Q8_0 (no transcode) + bf16 acts ---
    let opener = MmapFileOpener::new(&gguf_path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", gguf_path.display()));
    let gguf = GgufSource::open(opener).await.expect("parse encoder gguf");
    let source = RenamedSource::with_passthrough(gguf, qwen2vl_gguf_renames());
    let budget = ResidencyBudget {
        ram_bytes: 12 << 30,
        vram_bytes: 6 << 30,
    };
    let residency = WeightResidency::new(source, budget);
    let handles = register_handles(&residency, None).expect("register encoder handles");

    let backend = Arc::new(
        WgpuBackend::new_with_config(WgpuConfig {
            power_preference: match std::env::var("THINFER_POWER_PREF").as_deref() {
                Ok("low" | "lowpower" | "integrated") => PowerPreference::LowPower,
                Ok("none") => PowerPreference::None,
                _ => PowerPreference::HighPerformance,
            },
            timestamps: std::env::var("THINFER_TRACE").is_ok(),
            disable_coopmat: std::env::var("THINFER_NO_COOPMAT").is_ok(),
        })
        .await
        .expect("wgpu adapter unavailable for tests"),
    );

    // bf16 acts (residual-overflow guard; same lesson as umT5/Qwen3).
    let ops = WgslConfig {
        bf16_quant_writes: false,
        act_dtype: ActDtype::Bf16,
        weight_dtype: WeightDtype::Bf16,
    };
    // Every encoder matmul site is Q8_0 in the GGUF (ffn_down included).
    let q8 = WgslConfig {
        weight_dtype: WeightDtype::Quant(QuantKind::Q8_0),
        ..ops
    };
    let cfgs = BlockWgslConfigs {
        matmul_qkv: q8,
        matmul_qkv_self: q8,
        matmul_proj: q8,
        matmul_ffn_up: q8,
        matmul_ffn_down: q8,
        matmul_adaln: ops,
        ops,
        i8_sdpa: false,
        dense_acts: DenseActSites::default(),
        large_d_sdpa: false,
    };
    let pipelines = BlockPipelines::compile(&backend, &cfgs)
        .await
        .expect("compile encoder pipelines");

    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));
    let encoder = TextEncoder::new(seq.next_multiple_of(2));
    let out = encoder
        .forward(
            &backend,
            &pipelines,
            &residency,
            &workspace,
            &handles,
            residency.source(),
            &ids,
            false,
        )
        .await
        .expect("engine encoder forward");
    assert_eq!(out.seq, seq);
    assert_eq!(out.hidden.len(), seq * hidden_dim);

    eprintln!("---- qwen-image encoder parity ----");
    let exp = read_f32(&hidden_path);
    let (slope, rel) = report("hidden", &exp, &out.hidden);

    let mut failures = Vec::new();
    if (slope - 1.0).abs() > SLOPE_TOL || rel > REL_TOL {
        failures.push(format!(
            "hidden diverges: slope={slope:.6} rel={:.3}% (tol slope 1+-{SLOPE_TOL}, rel {REL_TOL})",
            rel * 100.0
        ));
    }
    assert!(
        failures.is_empty(),
        "encoder parity:\n{}",
        failures.join("\n")
    );
}

fn run_python_ref(gguf: &Path, tok_dir: &Path, out_dir: &Path) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let status = Command::new("uv")
        .args([
            "run",
            "--directory",
            py_dir.to_str().unwrap(),
            "--with",
            "gguf",
            "python",
            "-m",
            "thinfer_pytorch_ref.qwen_image.gen_encoder_ref",
            "--gguf",
            gguf.to_str().unwrap(),
            "--tokenizer-dir",
            tok_dir.to_str().unwrap(),
            "--prompt",
            PROMPT,
            "--out",
            out_dir.to_str().unwrap(),
        ])
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "qwen-image encoder pyref failed");
}
