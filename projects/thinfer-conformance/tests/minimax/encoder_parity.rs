//! MiniMax-H3 conditioner parity: the engine's truncated Qwen3-VL-32B
//! (`MiniMaxTextEncoder`, production config: Q4_K/Q6_K weights, f32 acts) vs the
//! torch reference `minimax/gen_encoder_ref.py` over the SAME GGUF.
//!
//! H3 conditions on `hidden_states[50]` before the final norm, where Qwen3 carries
//! massive-activation channels (absmax ~1e4 by layer 7). Stages: token ids
//! (engine tokenizer vs the reference's), the embedding gather, then the residual
//! stream at a ladder of depths (each a truncated engine forward), then the final state. The shallowest
//! failing depth localizes the bug.

#![cfg(feature = "minimax-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::tokenizer::Tokenizer;
use thinfer_core::workspace::Workspace;
use thinfer_models::minimax::manifest::{self, role};
use thinfer_models::minimax::text_encoder::{
    EncoderPipelines, MiniMaxTextEncoder, config, encoder_cfgs, register_handles,
};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;
use thinfer_native::tokenizer::HfTokenizer;

use crate::parity_util::{read_f32, read_i32, report};

/// f32 acts against an f32 reference over identical dequantized weights; the
/// GPU and torch differ only in accumulation order.
const SLOPE_TOL: f64 = 0.01;
const REL_TOL: f64 = 0.01;
/// Depths the reference dumps (`layer{i}.bin` = the state after layer i).
const PROBES: [usize; 8] = [0, 1, 2, 5, 10, 20, 30, 40];

#[tokio::test(flavor = "current_thread")]
async fn encoder_parity() {
    let _trace = thinfer_core::trace::init_from_env();
    let resolve = |r| cache::resolve(manifest::MANIFEST.get(r).expect("encoder role"));
    let (Some(gguf), Some(tok)) = (resolve(role::ENCODER_GGUF), resolve(role::TOKENIZER)) else {
        eprintln!("skipped[minimax encoder_parity]: encoder GGUF / tokenizer not in HF cache");
        return;
    };
    let dir = ensure_pyref(&gguf, &tok);
    let mut failures = Vec::new();

    let prompt = std::fs::read_to_string(dir.join("prompt.txt")).expect("prompt.txt");
    let exp_ids: Vec<u32> = read_i32(&dir.join("ids.bin"))
        .into_iter()
        .map(|i| i as u32)
        .collect();
    let ids = HfTokenizer::from_path(&tok)
        .await
        .expect("load tokenizer")
        .encode(&prompt, false)
        .expect("tokenize");
    eprintln!("ids: engine {} / reference {}", ids.len(), exp_ids.len());
    if ids != exp_ids {
        failures.push(format!("token ids differ: {ids:?} vs {exp_ids:?}"));
    }

    let backend = Arc::new(
        WgpuBackend::new_with_config(WgpuConfig {
            power_preference: PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("wgpu backend"),
    );
    let opener = MmapFileOpener::new(&gguf).await.expect("open encoder gguf");
    let src = GgufSource::open(opener).await.expect("parse encoder gguf");
    // Production-like: the 19 GB tower streams through a small VRAM budget.
    let residency = WeightResidency::new(
        src,
        ResidencyBudget {
            ram_bytes: 6 << 30,
            vram_bytes: 4 << 30,
        },
    );
    let pipelines = EncoderPipelines::compile(&backend, &encoder_cfgs())
        .await
        .expect("compile encoder pipelines");
    let ws = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    // Depth 0 = the embedding gather alone.
    let stages = std::iter::once((0, "embed".to_string()))
        .chain(PROBES.iter().map(|&i| (i + 1, format!("layer{i}"))))
        .chain([(config::N_LAYERS, "hidden".to_string())]);
    for (depth, name) in stages {
        let mut handles = register_handles(&residency, None).expect("register encoder");
        handles.truncate(depth);
        let got = MiniMaxTextEncoder
            .forward(
                &backend,
                &pipelines,
                &residency,
                &ws,
                &handles,
                residency.source(),
                &exp_ids,
                &[],
            )
            .await
            .expect("encoder forward")
            .hidden;
        let exp = read_f32(&dir.join(format!("{name}.bin")));
        let absmax = |v: &[f32]| v.iter().fold(0f32, |m, x| m.max(x.abs()));
        eprintln!(
            "[{name:>8}] absmax engine {:.1} / reference {:.1}",
            absmax(&got),
            absmax(&exp)
        );
        let (slope, rel) = report(&name, &exp, &got);
        if !slope.is_finite() || (slope - 1.0).abs() > SLOPE_TOL || rel.is_nan() || rel > REL_TOL {
            failures.push(format!("{name}: slope {slope:.5} rel {:.3}%", rel * 100.0));
        }
    }
    assert!(
        failures.is_empty(),
        "minimax encoder parity failed:\n{}",
        failures.join("\n")
    );
}

fn ensure_pyref(gguf: &Path, tokenizer: &Path) -> PathBuf {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("minimax_encoder_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("done.txt");
    if marker.exists() {
        eprintln!(
            "minimax encoder-parity: reusing cached pyref dumps ({})",
            tmp.display()
        );
        return tmp;
    }
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let status = Command::new("uv")
        .args(["run", "--directory", py_dir.to_str().unwrap()])
        .args(["--with", "gguf", "--with", "tokenizers", "python", "-m"])
        .arg("thinfer_pytorch_ref.minimax.gen_encoder_ref")
        .arg("--gguf")
        .arg(gguf)
        .arg("--tokenizer")
        .arg(tokenizer)
        .arg("--out")
        .arg(&tmp)
        // 50 CPU layers, one resident at a time; keep it off the user's other cores.
        .env("OMP_NUM_THREADS", "4")
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "minimax encoder pyref failed");
    std::fs::write(&marker, "ok").expect("write marker");
    tmp
}
