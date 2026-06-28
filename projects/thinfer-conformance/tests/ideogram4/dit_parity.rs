//! Ideogram-4 DiT parity: one velocity forward of the engine single-stream DiT
//! (Q8_0 weights from the DiT GGUF, bf16 acts -> dequant-once bf16 matmul path)
//! vs a bf16 reference dequantized from the SAME GGUF (`gen_dit_ref.py`), fed
//! IDENTICAL `llm_features`/noise (the pyref dumps them).
//!
//! Both sides start from identical Q8_0 bytes dequantized to bf16, so the band
//! is dominated by matmul accumulation rounding compounded across 34 blocks.
//! Localizes via `adaln_input`, block-0 output, and the last-block output so a
//! regression names where it diverges before the velocity gate.

#![cfg(feature = "ideogram4-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::ops::{ActDtype, WeightDtype, WgslConfig};
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::quant::QuantKind;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::common::block::{BlockPipelines, BlockWgslConfigs, DenseActSites};
use thinfer_models::ideogram4::dit::{DitInputs, DitTaps, Ideogram4Dit};
use thinfer_models::ideogram4::loader::register_handles;
use thinfer_models::ideogram4::manifest::{self, role};
use thinfer_models::ideogram4::packing::ImageGrid;
use thinfer_models::ideogram4::text_encoder::config as enc_config;
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, report};

// Tiny parity dims: 64x64 -> 4x4 grid (16 image tokens), 8 text tokens.
const WIDTH: usize = 64;
const HEIGHT: usize = 64;
const NUM_TEXT: usize = 8;
const TIMESTEP: f64 = 0.5;

// Velocity gate. 34-block bf16-vs-bf16 (same Q8_0 weights) accumulation noise;
// re-measure + re-document on a config change, do not loosen to go green.
const VEL_SLOPE_TOL: f64 = 0.06;
const VEL_REL_TOL: f64 = 0.15;
// Early-stage sanity (tighter: fewer blocks compounded).
const ADALN_REL_TOL: f64 = 0.03;
const BLOCK0_REL_TOL: f64 = 0.06;

#[tokio::test(flavor = "current_thread")]
async fn dit_parity() {
    let _trace = thinfer_core::trace::init_from_env();

    let fr = manifest::MANIFEST
        .get(role::DIT_GGUF_Q8_0)
        .expect("dit gguf role");
    let Some(gguf_path) = cache::resolve(fr) else {
        eprintln!(
            "skipped[dit_parity]: {}/{} not in HF cache",
            fr.repo, fr.path
        );
        return;
    };

    // --- python reference (deterministic; dumps inputs + velocity + taps) ---
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ideo_dit_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("config.txt");
    let cfg_tag = format!("{WIDTH}x{HEIGHT}t{NUM_TEXT}@{TIMESTEP}");
    let cached = tmp.join("velocity.bin").exists()
        && std::fs::read_to_string(&marker).is_ok_and(|c| c == cfg_tag);
    if cached {
        eprintln!("dit-parity: reusing cached pyref dumps ({})", tmp.display());
    } else {
        for ent in std::fs::read_dir(&tmp).into_iter().flatten().flatten() {
            let _ = std::fs::remove_file(ent.path());
        }
        run_python_ref(&gguf_path, &tmp);
        std::fs::write(&marker, &cfg_tag).expect("write marker");
    }

    let meta = std::fs::read_to_string(tmp.join("meta.txt")).expect("meta.txt");
    let m: Vec<usize> = meta
        .split_whitespace()
        .take(4)
        .map(|s| s.parse().expect("meta int"))
        .collect();
    let (num_text, num_image, grid_h, grid_w) = (m[0], m[1], m[2], m[3]);
    let seq = num_text + num_image;
    eprintln!(
        "dit-parity: seq={seq} num_text={num_text} num_image={num_image} grid={grid_h}x{grid_w}"
    );

    let llm_features = read_f32(&tmp.join("llm_features.bin"));
    let noise = read_f32(&tmp.join("noise.bin"));
    assert_eq!(llm_features.len(), num_text * enc_config::LLM_FEATURES_DIM);
    assert_eq!(noise.len(), num_image * 128);

    // --- engine DiT: open the DiT GGUF directly (keys are 1:1, no rename) ---
    let opener = MmapFileOpener::new(&gguf_path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", gguf_path.display()));
    let gguf = GgufSource::open(opener).await.expect("parse dit gguf");
    let budget = ResidencyBudget {
        ram_bytes: 12 << 30,
        vram_bytes: 6 << 30,
    };
    let residency = WeightResidency::new(gguf, budget);
    let handles = register_handles(&residency).expect("register dit handles");

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

    // F16 acts (the well-trodden z_image DiT path), with DP4A disabled per-site
    // via dense_acts so the Q8_0 matmuls take the dequant-once -> f16 path (no
    // i8-act quantization, which is numerically unsound for the residual
    // stream). head_dim 256 routes to SdpaF32LargeD (F16 variant). The pyref's
    // bf16 rounding is the noise floor; f16 has a finer mantissa. f16+DP4A is a
    // later perf pass.
    let act = ActDtype::F16;
    let ops = WgslConfig {
        bf16_quant_writes: false,
        act_dtype: act,
        weight_dtype: WeightDtype::Bf16,
    };
    // Block projection matmuls are Q8_0 in the file; module-level linears + the
    // adaln matmul are bf16.
    let q8 = WgslConfig {
        weight_dtype: WeightDtype::Quant(QuantKind::Q8_0),
        ..ops
    };
    let main_cfgs = BlockWgslConfigs {
        matmul_qkv: q8,
        matmul_qkv_self: q8,
        matmul_proj: q8,
        matmul_ffn_up: q8,
        matmul_ffn_down: q8,
        matmul_adaln: ops,
        ops,
        i8_sdpa: false,
        dense_acts: DenseActSites {
            qkv: true,
            proj: true,
            ffn_up: true,
            ffn_down: true,
        },
        large_d_sdpa: true, // head_dim 256 -> SdpaF32LargeD
    };
    let dense_cfgs = BlockWgslConfigs::uniform(ops);
    let main_pipelines = BlockPipelines::compile(&backend, &main_cfgs)
        .await
        .expect("compile main pipelines");
    let dense_pipelines = BlockPipelines::compile(&backend, &dense_cfgs)
        .await
        .expect("compile dense pipelines");

    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));
    let dit = Ideogram4Dit::assemble(handles, seq);

    let inputs = DitInputs {
        llm_features: &llm_features,
        num_text,
        noise: &noise,
        grid: ImageGrid { grid_h, grid_w },
        timestep: TIMESTEP as f32,
    };
    let (mut adaln_got, mut block0_got, mut block_last_got) = (Vec::new(), Vec::new(), Vec::new());
    let taps = DitTaps {
        adaln_input: Some(&mut adaln_got),
        h_assembled: None,
        block0_out: Some(&mut block0_got),
        block_last_out: Some(&mut block_last_got),
    };
    let velocity = dit
        .forward_with_taps(
            &backend,
            &main_pipelines,
            &dense_pipelines,
            &residency,
            &workspace,
            &inputs,
            taps,
        )
        .await
        .expect("engine dit forward");

    // --- compare (localize, then gate velocity) ---
    eprintln!("---- ideogram dit parity ----");
    let (_, adaln_rel) = report("adaln", &read_f32(&tmp.join("adaln_input.bin")), &adaln_got);
    let (_, block0_rel) = report(
        "block0",
        &read_f32(&tmp.join("block0_out.bin")),
        &block0_got,
    );
    let _ = report(
        "block_last",
        &read_f32(&tmp.join("block_last_out.bin")),
        &block_last_got,
    );
    let (vel_slope, vel_rel) = report("velocity", &read_f32(&tmp.join("velocity.bin")), &velocity);

    let mut failures = Vec::new();
    if adaln_rel > ADALN_REL_TOL {
        failures.push(format!(
            "adaln_input diverges: rel={:.3}% (tol {ADALN_REL_TOL})",
            adaln_rel * 100.0
        ));
    }
    if block0_rel > BLOCK0_REL_TOL {
        failures.push(format!(
            "block0 diverges: rel={:.3}% (tol {BLOCK0_REL_TOL})",
            block0_rel * 100.0
        ));
    }
    if (vel_slope - 1.0).abs() > VEL_SLOPE_TOL || vel_rel > VEL_REL_TOL {
        failures.push(format!(
            "velocity diverges: slope={vel_slope:.6} rel={:.3}% (tol slope 1+-{VEL_SLOPE_TOL}, rel {VEL_REL_TOL})",
            vel_rel * 100.0
        ));
    }
    assert!(failures.is_empty(), "dit parity:\n{}", failures.join("\n"));
}

fn run_python_ref(gguf: &Path, out_dir: &Path) {
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
            "thinfer_pytorch_ref.ideogram4.gen_dit_ref",
            "--gguf",
            gguf.to_str().unwrap(),
            "--out",
            out_dir.to_str().unwrap(),
            "--width",
            &WIDTH.to_string(),
            "--height",
            &HEIGHT.to_string(),
            "--num-text",
            &NUM_TEXT.to_string(),
            "--timestep",
            &TIMESTEP.to_string(),
        ])
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "ideogram dit pyref failed");
}
