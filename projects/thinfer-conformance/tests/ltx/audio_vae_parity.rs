//! LTX-2.3 audio VAE decoder parity: the engine `AudioVaeDecoder` (bf16
//! safetensors weights, F32 acts) vs the upstream `AudioDecoder`
//! (`gen_audio_vae_ref.py`, same bf16 bytes, f32 compute). Decodes a fixed-seed
//! normalized latent `[8,frames,16]` to a mel `[2,4*frames-3,64]` and compares.
//! Both sides consume the same normalized latent and apply the same per-channel
//! un-normalize internally. Same weights -> bit-tight; the only drift is GPU f32
//! conv accumulation order vs torch over the decoder's ~16-conv causal stack.

#![cfg(feature = "ltx-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::ltx::audio_vae::{
    AudioVaeDecoder, AudioVaePipelines, LATENT_CHANNELS, LATENT_MEL_BINS, MEL_BINS, OUT_CHANNELS,
    load_latent_stats,
};
use thinfer_models::ltx::manifest::{self, role};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, report};

const FRAMES: usize = 2;
const SLOPE_TOL: f64 = 0.01;
const REL_TOL: f64 = 0.002;

#[tokio::test(flavor = "current_thread")]
async fn audio_vae_parity() {
    let _trace = thinfer_core::trace::init_from_env();

    let fr = manifest::MANIFEST
        .get(role::AUDIO_VAE)
        .expect("audio vae role");
    let Some(path) = cache::resolve(fr) else {
        eprintln!("skipped[ltx audio_vae_parity]: audio VAE not in HF cache");
        return;
    };

    // --- python reference (cached by frames) ---
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ltx_audio_vae_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("frames.txt");
    let cached = tmp.join("mel.bin").exists()
        && std::fs::read_to_string(&marker).is_ok_and(|d| d == FRAMES.to_string());
    if !cached {
        for ent in std::fs::read_dir(&tmp).into_iter().flatten().flatten() {
            let _ = std::fs::remove_file(ent.path());
        }
        run_python_ref(&path, &tmp);
        std::fs::write(&marker, FRAMES.to_string()).expect("write marker");
    }

    let meta = std::fs::read_to_string(tmp.join("meta.txt")).expect("meta.txt");
    let m: Vec<usize> = meta
        .split_whitespace()
        .map(|s| s.parse().expect("meta int"))
        .collect();
    let (mf, t_mel, mel_bins) = (m[0], m[1], m[2]);
    assert_eq!(mf, FRAMES, "pyref frames");
    assert_eq!((t_mel, mel_bins), (4 * FRAMES - 3, MEL_BINS), "mel shape");

    let latent = read_f32(&tmp.join("latent.bin"));
    assert_eq!(
        latent.len(),
        LATENT_CHANNELS * FRAMES * LATENT_MEL_BINS,
        "latent size"
    );

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

    let pipelines = AudioVaePipelines::compile(&backend)
        .await
        .expect("compile audio vae pipelines");

    let opener = MmapFileOpener::new(&path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let src = SafetensorsSource::open(opener)
        .await
        .expect("parse audio vae safetensors");
    let residency = WeightResidency::new(
        src,
        ResidencyBudget {
            ram_bytes: 16 << 30,
            vram_bytes: 5 << 30,
        },
    );
    let (mean, std) = load_latent_stats(&residency, &backend)
        .await
        .expect("load audio latent stats");
    let decoder =
        AudioVaeDecoder::new(pipelines, &residency, mean, std).expect("build audio vae decoder");
    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    let mel = decoder
        .decode(&backend, &residency, &workspace, &latent, FRAMES)
        .await
        .expect("audio vae decode");

    let exp = read_f32(&tmp.join("mel.bin"));
    assert_eq!(mel.len(), OUT_CHANNELS * t_mel * mel_bins, "mel size");
    assert_eq!(
        mel.len(),
        exp.len(),
        "mel len exp={} got={}",
        exp.len(),
        mel.len()
    );

    eprintln!("---- ltx audio vae parity ----");
    let (slope, rel) = report("mel", &exp, &mel);
    let mut failures = Vec::new();
    if !slope.is_finite() || (slope - 1.0).abs() > SLOPE_TOL {
        failures.push(format!("slope off: {slope:.6}"));
    }
    if rel > REL_TOL {
        failures.push(format!("rel={:.3}% > {:.3}%", rel * 100.0, REL_TOL * 100.0));
    }
    assert!(
        failures.is_empty(),
        "ltx audio vae parity (slope 1+-{SLOPE_TOL}):\n{}",
        failures.join("\n")
    );
}

fn run_python_ref(audio_vae: &Path, out_dir: &Path) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let status = Command::new("uv")
        .args([
            "run",
            "--directory",
            py_dir.to_str().unwrap(),
            "--with",
            "einops",
            "--with",
            "torchaudio",
            "python",
            "-m",
            "thinfer_pytorch_ref.ltx.gen_audio_vae_ref",
            "--audio-vae",
            audio_vae.to_str().unwrap(),
            "--out",
            out_dir.to_str().unwrap(),
            "--frames",
            &FRAMES.to_string(),
        ])
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "ltx audio vae pyref failed");
}
