//! Qwen3-VL-32B vision tower parity (MiniMax-H3's image conditioner): the
//! engine's `minimax::vision` vs HF `Qwen3VLVisionModel` in bf16 on CPU over
//! the SAME mmproj GGUF (`minimax/gen_vision_ref.py`), on a synthetic image.
//! Checks the merged embeds and all three deepstack streams.

#![cfg(feature = "minimax-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::common::block::BlockPipelines;
use thinfer_models::minimax::manifest::{self, role};
use thinfer_models::minimax::vision;
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, report};

/// bf16 torch on CPU vs the engine's bf16 tower (tanh GELU in the mergers).
/// Rounding drift accumulates with depth (measured: 2.1% after block 8, 3.0%
/// after 16, 5.0% after 24, 5.7% at the merger; slopes within 0.006), so the
/// gate matches the Qwen2.5-VL tower's (`qwen_image/vision_parity`).
const SLOPE_TOL: f64 = 0.05;
const REL_TOL: f64 = 0.08;

#[tokio::test(flavor = "current_thread")]
async fn vision_parity() {
    let _trace = thinfer_core::trace::init_from_env();
    let fr = manifest::MANIFEST.get(role::VISION).expect("vision role");
    let Some(path) = cache::resolve(fr) else {
        eprintln!("skipped[minimax vision_parity]: vision mmproj not in HF cache");
        return;
    };
    let dir = ensure_pyref(&path);
    let meta: Vec<usize> = std::fs::read_to_string(dir.join("meta.txt"))
        .expect("meta.txt")
        .split_whitespace()
        .map(|t| t.parse().expect("meta int"))
        .collect();
    let [h, w] = meta[..] else {
        panic!("meta.txt needs 2 fields")
    };
    let rgb = std::fs::read(dir.join("image.bin")).expect("image.bin");

    let backend = Arc::new(
        WgpuBackend::new_with_config(WgpuConfig {
            power_preference: PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("wgpu backend"),
    );
    // The production path: attention on the CUDA tile kernel where present.
    match thinfer_cuda::install(&backend) {
        Ok((name, _)) => eprintln!("vision parity: cuda on {name}"),
        Err(e) => eprintln!("vision parity: no cuda ({e}), backend attention"),
    }
    let gguf = GgufSource::open(MmapFileOpener::new(&path).await.expect("open mmproj"))
        .await
        .expect("parse mmproj");
    let residency = WeightResidency::new(
        gguf,
        ResidencyBudget {
            ram_bytes: 4 << 30,
            vram_bytes: 4 << 30,
        },
    );
    let handles = vision::register_handles(&residency).expect("register vision");
    let bp = BlockPipelines::compile(&backend, &vision::wgsl_configs())
        .await
        .expect("pipelines");
    let ws = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));
    let out = vision::forward(&backend, &bp, &residency, &ws, &handles, &rgb, h, w)
        .await
        .expect("vision forward");
    assert_eq!(out.tokens, (h / 32) * (w / 32));

    let mut worst = Vec::new();
    let mut check = |label: &str, exp: &[f32], got: &[f32]| {
        assert_eq!(exp.len(), got.len(), "{label}: length");
        let (slope, rel) = report(label, exp, got);
        worst.push((label.to_string(), slope, rel));
    };
    check("embeds", &read_f32(&dir.join("embeds.bin")), &out.embeds);
    for (k, ds) in out.deepstack.iter().enumerate() {
        check(
            &format!("deepstack{k}"),
            &read_f32(&dir.join(format!("deepstack{k}.bin"))),
            ds,
        );
    }
    for (label, slope, rel) in &worst {
        assert!(
            (slope - 1.0).abs() < SLOPE_TOL && *rel < REL_TOL,
            "{label}: slope {slope:.4} rel {rel:.4}"
        );
    }
}

fn ensure_pyref(mmproj: &Path) -> PathBuf {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("minimax_vision_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("done.txt");
    if marker.exists() {
        eprintln!(
            "minimax vision parity: reusing cached pyref dumps ({})",
            tmp.display()
        );
        return tmp;
    }
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let status = Command::new("uv")
        .args([
            "run",
            "--directory",
            py_dir.to_str().unwrap(),
            "python",
            "-m",
        ])
        .arg("thinfer_pytorch_ref.minimax.gen_vision_ref")
        .arg("--mmproj")
        .arg(mmproj)
        .arg("--out")
        .arg(&tmp)
        .env("OMP_NUM_THREADS", "4")
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "minimax vision pyref failed");
    std::fs::write(&marker, "ok").expect("write marker");
    tmp
}
