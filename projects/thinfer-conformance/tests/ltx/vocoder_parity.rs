//! LTX-2.3 vocoder parity: the engine `Vocoder` (BigVGAN main + BWE on GPU via
//! the net-new f32 1D ops; STFT-mel + Hann resampler on host) vs the upstream
//! `VocoderWithBWE` (`gen_vocoder_ref.py`, same bf16 weights, f32 autocast). Runs
//! a fixed-seed log-mel `[2,frames,64]` through the full chain and compares the
//! 48kHz stereo waveform. Both sides upcast the bf16 weights to f32; drift is the
//! GPU-vs-torch f32 accumulation order over ~108 sequential convolutions, so the
//! band is looser than the shallow VAE gates.

#![cfg(feature = "ltx-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::ltx::manifest::{self, role};
use thinfer_models::ltx::vocoder::{MEL_BINS, OUT_CHANNELS, Vocoder, VocoderPipelines};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, report};

const FRAMES: usize = 8;
const SLOPE_TOL: f64 = 0.02;
const REL_TOL: f64 = 0.05;

#[tokio::test(flavor = "current_thread")]
async fn vocoder_parity() {
    let _trace = thinfer_core::trace::init_from_env();

    let fr = manifest::MANIFEST
        .get(role::AUDIO_VAE)
        .expect("audio vae role");
    let Some(path) = cache::resolve(fr) else {
        eprintln!("skipped[ltx vocoder_parity]: audio VAE not in HF cache");
        return;
    };

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ltx_vocoder_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("frames.txt");
    let cached = tmp.join("wav.bin").exists()
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
    let (mf, t_wav, _out_sr) = (m[0], m[1], m[2]);
    assert_eq!(mf, FRAMES, "pyref frames");

    let mel = read_f32(&tmp.join("mel.bin"));
    assert_eq!(mel.len(), OUT_CHANNELS * FRAMES * MEL_BINS, "mel size");

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

    let pipelines = VocoderPipelines::compile(&backend)
        .await
        .expect("compile vocoder pipelines");

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
            vram_bytes: 6 << 30,
        },
    );
    let vocoder = Vocoder::new(pipelines, &residency, &backend)
        .await
        .expect("build vocoder");
    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    let wav = vocoder
        .decode(&backend, &residency, &workspace, &mel, FRAMES)
        .await
        .expect("vocoder decode");

    let exp = read_f32(&tmp.join("wav.bin"));
    assert_eq!(wav.len(), OUT_CHANNELS * t_wav, "wav size");
    assert_eq!(
        wav.len(),
        exp.len(),
        "wav len exp={} got={}",
        exp.len(),
        wav.len()
    );

    eprintln!("---- ltx vocoder parity ----");
    let (slope, rel) = report("wav", &exp, &wav);
    let mut failures = Vec::new();
    if !slope.is_finite() || (slope - 1.0).abs() > SLOPE_TOL {
        failures.push(format!("slope off: {slope:.6}"));
    }
    if rel > REL_TOL {
        failures.push(format!("rel={:.3}% > {:.3}%", rel * 100.0, REL_TOL * 100.0));
    }
    assert!(
        failures.is_empty(),
        "ltx vocoder parity (slope 1+-{SLOPE_TOL}):\n{}",
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
            "thinfer_pytorch_ref.ltx.gen_vocoder_ref",
            "--audio-vae",
            audio_vae.to_str().unwrap(),
            "--out",
            out_dir.to_str().unwrap(),
            "--frames",
            &FRAMES.to_string(),
        ])
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "ltx vocoder pyref failed");
}
