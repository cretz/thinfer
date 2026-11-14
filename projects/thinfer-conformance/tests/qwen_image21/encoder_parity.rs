//! Qwen-Image-2.1 conditioner parity: the engine's Qwen3-VL-8B
//! (`MiniMaxTextEncoder` at `LmDims::QWEN3_VL_8B`, the shipped f32-act text
//! config, Q8_0 weights) vs transformers' Qwen3-VL text decoder layers over the
//! SAME GGUF (`gen_encoder_ref.py`). The prompt is the diffusers 2.1 template;
//! the engine tokenizer must reproduce the processor's ids. Stages: embedding,
//! a few depths (truncated engine forwards), the full un-normed last layer.

#![cfg(feature = "qwen-image-e2e")]

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
use thinfer_models::minimax::text_encoder::{
    EncoderPipelines, LmDims, MiniMaxTextEncoder, encoder_cfgs, register_handles,
};
use thinfer_models::qwen_image21::config;
use thinfer_models::qwen_image21::manifest::{MANIFEST, role};
use thinfer_models::z_image::qwen3_gguf_renames;
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;
use thinfer_native::tokenizer::HfTokenizer;

use crate::parity_util::{read_f32, read_u32, report};

/// f32 acts vs an f32 reference over identical dequantized weights.
const SLOPE_TOL: f64 = 0.01;
const REL_TOL: f64 = 0.01;
/// Depths the reference dumps (`layer{i}.bin` = the state after layer i).
const PROBES: [usize; 3] = [0, 1, 10];
/// The Qwen3-VL-8B config the reference builds its layers from.
const CONFIG_REPO: &str = "Qwen/Qwen3-VL-8B-Instruct";

#[tokio::test(flavor = "current_thread")]
async fn encoder_parity() {
    let _trace = thinfer_core::trace::init_from_env();
    let resolve = |r| cache::resolve(MANIFEST.get(r).expect("encoder role"));
    let conf = cache::resolve(&thinfer_core::manifest::FileRef::new(
        CONFIG_REPO,
        "config.json",
    ));
    let (Some(gguf), Some(tok), Some(conf)) =
        (resolve(role::ENCODER_GGUF), resolve(role::TOKENIZER), conf)
    else {
        eprintln!(
            "skipped[qwen21 encoder_parity]: encoder GGUF / tokenizer / config not in HF cache"
        );
        return;
    };
    let dir = ensure_pyref(&gguf, &tok, &conf);
    let mut failures = Vec::new();

    let meta = std::fs::read_to_string(dir.join("meta.txt")).expect("meta.txt");
    let drop: usize = meta
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("meta drop");
    let ids = read_u32(&dir.join("ids.bin"));
    let text = std::fs::read_to_string(dir.join("text.txt")).expect("text.txt");
    let engine_ids = HfTokenizer::from_path(&tok)
        .await
        .expect("load tokenizer")
        .encode(&text, false)
        .expect("tokenize");
    if engine_ids != ids {
        failures.push(format!("token ids differ: {engine_ids:?} vs {ids:?}"));
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
    // The app's view: llama.cpp names renamed to the HF keys the engine registers.
    let residency = WeightResidency::new(
        RenamedSource::with_passthrough(
            GgufSource::open(opener).await.expect("parse encoder gguf"),
            qwen3_gguf_renames(),
        ),
        ResidencyBudget {
            ram_bytes: 4 << 30,
            vram_bytes: 4 << 30,
        },
    );
    let pipelines = EncoderPipelines::compile(&backend, &encoder_cfgs())
        .await
        .expect("compile encoder pipelines");
    let ws = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    let dims = LmDims::QWEN3_VL_8B;
    let stages = std::iter::once((0, "embed".to_string()))
        .chain(PROBES.iter().map(|&i| (i + 1, format!("layer{i}"))))
        .chain([(dims.n_layers, "hidden".to_string())]);
    for (depth, name) in stages {
        let mut handles = register_handles(&residency, None, dims).expect("register encoder");
        handles.truncate(depth);
        let got = MiniMaxTextEncoder
            .forward(
                &backend,
                &pipelines,
                &residency,
                &ws,
                &handles,
                residency.source(),
                &ids,
                &[],
            )
            .await
            .expect("encoder forward")
            .hidden;
        // The DiT reads the rows after the system prompt.
        let exp = read_f32(&dir.join(format!("{name}.bin")));
        let rows = drop * config::TEXT_DIM;
        let (slope, rel) = report(&name, &exp[rows..], &got[rows..]);
        if !slope.is_finite() || (slope - 1.0).abs() > SLOPE_TOL || rel.is_nan() || rel > REL_TOL {
            failures.push(format!("{name}: slope {slope:.5} rel {:.3}%", rel * 100.0));
        }
    }
    assert!(
        failures.is_empty(),
        "qwen21 encoder parity failed:\n{}",
        failures.join("\n")
    );
}

fn ensure_pyref(gguf: &Path, tokenizer: &Path, conf: &Path) -> PathBuf {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("qwen21_encoder_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("done.txt");
    if marker.exists() {
        eprintln!(
            "encoder-parity: reusing cached pyref dumps ({})",
            tmp.display()
        );
        return tmp;
    }
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let mut cmd = Command::new("uv");
    cmd.args(["run", "--directory", py_dir.to_str().unwrap()]);
    for dep in [
        "gguf",
        "tokenizers",
        "huggingface_hub>=2",
        "transformers>5.8",
    ] {
        cmd.args(["--with", dep]);
    }
    let status = cmd
        .args([
            "python",
            "-m",
            "thinfer_pytorch_ref.qwen_image21.gen_encoder_ref",
        ])
        .arg("--gguf")
        .arg(gguf)
        .arg("--tokenizer")
        .arg(tokenizer)
        .arg("--config")
        .arg(conf)
        .arg("--out")
        .arg(&tmp)
        // 36 CPU layers, one resident at a time; keep it off the user's other cores.
        .env("OMP_NUM_THREADS", "4")
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "qwen21 encoder pyref failed");
    std::fs::write(&marker, "ok").expect("write marker");
    tmp
}
