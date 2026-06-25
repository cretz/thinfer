//! Qwen2.5-VL vision tower (mmproj) parity: the engine vision tower (bf16 acts,
//! bf16 block matmuls, 2D half-rot RoPE, windowed/full segmented attention,
//! SwiGLU FFN, GELU merger) vs the HF `Qwen2_5_VisionTransformerPretrainedModel`
//! dequantized from the SAME mmproj GGUF (`gen_vision_ref.py`), fed the IDENTICAL
//! patchified `pixel_values [N, 1176]`. Both sides consume the same patch input
//! and same (dequantized) weights, so the band is just the engine's bf16
//! rounding vs the fp32 reference.
//!
//! 112x112 -> grid 8x8 -> N=64 patches -> 16 merged tokens: exactly one 4x4
//! window (exercises the full-window path + the full-attn blocks). The
//! `vision_parity_multi` case uses 224x224 -> 16x16 -> 4 windows (exercises the
//! windowed segmented attention + the window permutation/unsort).
#![cfg(feature = "qwen-image-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::qwen_image::dit::QwenImageDitPipelines;
use thinfer_models::qwen_image::manifest::{self, role};
use thinfer_models::qwen_image::vision::{VisionTower, register_handles};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, report};

const SLOPE_TOL: f64 = 0.05;
const REL_TOL: f64 = 0.08;

#[tokio::test(flavor = "current_thread")]
async fn vision_parity() {
    run_case(8, 8, "qwen_vision_parity").await;
}

/// Multi-window case (224x224 -> 16x16 grid -> 2x2 windows). Exercises the
/// windowed segmented-attention mask + the window permutation/unsort.
#[tokio::test(flavor = "current_thread")]
async fn vision_parity_multi() {
    run_case(16, 16, "qwen_vision_parity_multi").await;
}

async fn run_case(gh: usize, gw: usize, tag: &str) {
    let _trace = thinfer_core::trace::init_from_env();

    let fr = manifest::MANIFEST
        .get(role::MMPROJ_F16)
        .expect("mmproj role");
    let Some(mmproj_path) = cache::resolve(fr) else {
        eprintln!(
            "skipped[vision_parity {gh}x{gw}]: {}/{} not in HF cache",
            fr.repo, fr.path
        );
        return;
    };

    // --- python reference (deterministic; dumps pixel_values + embeds) ---
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(tag);
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("config.txt");
    let cfg_tag = format!("{gh}x{gw}");
    let cached = tmp.join("vision_embeds.bin").exists()
        && std::fs::read_to_string(&marker).is_ok_and(|c| c == cfg_tag);
    if cached {
        eprintln!("vision-parity: reusing cached pyref ({})", tmp.display());
    } else {
        for ent in std::fs::read_dir(&tmp).into_iter().flatten().flatten() {
            let _ = std::fs::remove_file(ent.path());
        }
        run_python_ref(&mmproj_path, &tmp, gh, gw);
        std::fs::write(&marker, &cfg_tag).expect("write marker");
    }

    let meta = std::fs::read_to_string(tmp.join("meta.txt")).expect("meta.txt");
    let m: Vec<usize> = meta
        .split_whitespace()
        .take(4)
        .map(|s| s.parse().expect("meta int"))
        .collect();
    let (mgh, mgw, n, tokens) = (m[0], m[1], m[2], m[3]);
    assert_eq!((mgh, mgw), (gh, gw));
    eprintln!("vision-parity: grid={gh}x{gw} N={n} tokens={tokens}");

    let pixel_values = read_f32(&tmp.join("pixel_values.bin"));
    assert_eq!(pixel_values.len(), n * 1176);
    let embeds_ref = read_f32(&tmp.join("vision_embeds.bin"));
    assert_eq!(embeds_ref.len(), tokens * 3584);

    // --- engine: open the mmproj GGUF (native keys), run the tower ---
    let opener = MmapFileOpener::new(&mmproj_path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", mmproj_path.display()));
    let gguf = GgufSource::open(opener).await.expect("parse mmproj gguf");
    let budget = ResidencyBudget {
        ram_bytes: 12 << 30,
        vram_bytes: 6 << 30,
    };
    let residency = WeightResidency::new(gguf, budget);
    let handles = register_handles(&residency).expect("register vision handles");

    let backend = Arc::new(
        WgpuBackend::new_with_config(WgpuConfig {
            power_preference: match std::env::var("THINFER_POWER_PREF").as_deref() {
                Ok("low" | "lowpower" | "integrated") => PowerPreference::LowPower,
                Ok("none") => PowerPreference::None,
                _ => PowerPreference::HighPerformance,
            },
            timestamps: std::env::var("THINFER_TRACE").is_ok(),
        })
        .await
        .expect("wgpu adapter unavailable for tests"),
    );

    let cfgs = VisionTower::wgsl_configs();
    // Vision uses the common-block masked op_sdpa, not the f16 fast path.
    let pipelines = QwenImageDitPipelines::compile(&backend, &cfgs, false)
        .await
        .expect("compile vision pipelines");

    let scratch = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));
    let tower = VisionTower::new(gh.max(gw));
    let out = tower
        .forward(
            &backend,
            &pipelines,
            &residency,
            &scratch,
            &handles,
            &pixel_values,
            gh,
            gw,
        )
        .await
        .expect("engine vision forward");
    assert_eq!(out.tokens, tokens);
    assert_eq!(out.embeds.len(), embeds_ref.len());

    eprintln!("---- qwen-image vision parity {gh}x{gw} ----");
    let (slope, rel) = report("vision_embeds", &embeds_ref, &out.embeds);

    let mut failures = Vec::new();
    if (slope - 1.0).abs() > SLOPE_TOL || rel > REL_TOL {
        failures.push(format!(
            "vision_embeds diverges: slope={slope:.6} rel={:.3}% (tol slope 1+-{SLOPE_TOL}, rel {REL_TOL})",
            rel * 100.0
        ));
    }
    assert!(
        failures.is_empty(),
        "vision parity {gh}x{gw}:\n{}",
        failures.join("\n")
    );
}

fn run_python_ref(mmproj: &Path, out_dir: &Path, gh: usize, gw: usize) {
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
            "thinfer_pytorch_ref.qwen_image.gen_vision_ref",
            "--mmproj",
            mmproj.to_str().unwrap(),
            "--out",
            out_dir.to_str().unwrap(),
            "--gh",
            &gh.to_string(),
            "--gw",
            &gw.to_string(),
        ])
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "qwen-image vision pyref failed");
}
