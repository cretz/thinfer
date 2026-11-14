//! Qwen-Image-2.1 VAE parity: the shared Wan-family residual VAE engine in its
//! 2D RGBA config (`qwen_image21_vae`) vs the fp32 diffusers
//! `AutoencoderKLQwenImage21` from the SAME safetensors (`gen_vae_ref.py`).
//! Encode: a seeded smooth RGBA image -> posterior mean. Decode: the normalized
//! mean back to pixels (a realistic latent; random ones saturate the clamp).

#![cfg(feature = "qwen-image-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::qwen_image21::config;
use thinfer_models::qwen_image21::manifest::{MANIFEST, role};
use thinfer_models::qwen_image21::vae::{PIXEL_CHANNELS, qwen_image21_vae};
use thinfer_models::wan::vae::{
    VaeDecoderWeights, VaeEncoderWeights, WanVaeDecoder, WanVaeEncoder, WanVaePipelines,
    register_decoder, register_encoder,
};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, report};

const SIZE: usize = 256;

// Engine (bf16 weights, f16 acts, f32 conv accumulate) vs the fp32 reference.
// Measured mean rel 0.22%, decoded 3.2% (2026-11-10); re-measure on a config
// change.
const SLOPE_TOL: f64 = 0.01;
const MEAN_REL_TOL: f64 = 0.01;
const DECODED_REL_TOL: f64 = 0.05;

#[tokio::test(flavor = "current_thread")]
async fn vae_parity() {
    let _trace = thinfer_core::trace::init_from_env();

    let fr = MANIFEST.get(role::VAE).expect("vae role");
    let Some(vae_path) = cache::resolve(fr) else {
        eprintln!(
            "skipped[vae_parity]: {}/{} not in HF cache",
            fr.repo, fr.path
        );
        return;
    };

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("qwen21_vae_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("config.txt");
    let cfg_tag = SIZE.to_string();
    if std::fs::read_to_string(&marker).is_ok_and(|c| c == cfg_tag) {
        eprintln!("vae-parity: reusing cached pyref dumps ({})", tmp.display());
    } else {
        run_python_ref(vae_path.parent().expect("vae dir"), &tmp);
        std::fs::write(&marker, &cfg_tag).expect("write marker");
    }

    let lat = SIZE / config::PIXELS_PER_TOKEN;
    let image = read_f32(&tmp.join("image.bin"));
    let exp_mean = read_f32(&tmp.join("mean.bin"));
    let latent = read_f32(&tmp.join("latent_norm.bin"));
    let exp_decoded = read_f32(&tmp.join("decoded.bin"));
    assert_eq!(image.len(), PIXEL_CHANNELS * SIZE * SIZE);
    assert_eq!(exp_mean.len(), config::LATENT_CH * lat * lat);

    let opener = MmapFileOpener::new(&vae_path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", vae_path.display()));
    let residency = WeightResidency::new(
        SafetensorsSource::open(opener)
            .await
            .expect("parse vae safetensors"),
        ResidencyBudget {
            ram_bytes: 4 << 30,
            vram_bytes: 4 << 30,
        },
    );
    let cfg = qwen_image21_vae();
    let before = residency.total_registered_bytes();
    let enc_handles =
        register_encoder(&residency, &VaeEncoderWeights::new(&cfg)).expect("register encoder");
    let enc_footprint = residency.total_registered_bytes() - before;
    let before = residency.total_registered_bytes();
    let dec_handles =
        register_decoder(&residency, &VaeDecoderWeights::new(&cfg)).expect("register decoder");
    let dec_footprint = residency.total_registered_bytes() - before;

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
    let encoder = WanVaeEncoder {
        pipelines: WanVaePipelines::compile(&backend)
            .await
            .expect("compile encoder"),
        handles: enc_handles,
        cfg: cfg.clone(),
        weight_footprint: enc_footprint,
    };
    let decoder = WanVaeDecoder {
        pipelines: WanVaePipelines::compile(&backend)
            .await
            .expect("compile decoder"),
        handles: dec_handles,
        cfg,
        weight_footprint: dec_footprint,
    };
    let mut ws = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    let moments = encoder
        .encode(&backend, &residency, &mut ws, &image, 1, SIZE, SIZE)
        .await
        .expect("engine vae encode");
    let got_mean = &moments[..exp_mean.len()];
    let decoded = decoder
        .decode(&backend, &residency, &mut ws, &latent, 1, lat, lat)
        .await
        .expect("engine vae decode");

    eprintln!("---- qwen-image-2.1 vae parity ({SIZE}x{SIZE}) ----");
    let mut failures = Vec::new();
    for (label, exp, got, rel_tol) in [
        ("mean", &exp_mean, got_mean, MEAN_REL_TOL),
        ("decoded", &exp_decoded, &decoded[..], DECODED_REL_TOL),
    ] {
        let (slope, rel) = report(label, exp, got);
        if (slope - 1.0).abs() > SLOPE_TOL || rel > rel_tol {
            failures.push(format!(
                "{label}: slope={slope:.6} rel={:.3}% (tol slope 1+-{SLOPE_TOL}, rel {rel_tol})",
                rel * 100.0
            ));
        }
    }
    assert!(failures.is_empty(), "vae parity:\n{}", failures.join("\n"));
}

fn run_python_ref(vae_dir: &Path, out_dir: &Path) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let mut cmd = Command::new("uv");
    cmd.args(["run", "--directory", py_dir.to_str().unwrap()]);
    for dep in ["huggingface_hub>=2", "transformers>5.8"] {
        cmd.args(["--with", dep]);
    }
    let status = cmd
        .args([
            "python",
            "-m",
            "thinfer_pytorch_ref.qwen_image21.gen_vae_ref",
            "--vae-dir",
            vae_dir.to_str().unwrap(),
            "--out",
            out_dir.to_str().unwrap(),
            "--size",
            &SIZE.to_string(),
        ])
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "qwen-image-2.1 vae pyref failed");
}
