//! MiniMax-H3 audio VAE ENCODER parity (Ref2VA audio references): the
//! engine's `audio_encoder` vs ComfyUI's torch `MiniMaxH3AudioVAE.encode`
//! over the SAME Comfy-Org fp32 file (`minimax/gen_audio_enc_ref.py`, fp32 on
//! CPU, a 2.01 s stereo test signal), first the DAC tower (channel 0) so a
//! mismatch localizes, then the normalized latents of both channels.

#![cfg(feature = "minimax-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::minimax::audio_encoder::AudioEncoder;
use thinfer_models::minimax::audio_vae::{AudioVae, AudioVaeConfig};
use thinfer_models::minimax::manifest::{self, role};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, report};

/// bf16-weight / f32-act tower and f32 host head vs fp32 torch (measured
/// 2026-11-23: tower 0.09%, latents 0.29%).
const SLOPE_TOL: f64 = 0.005;
const REL_TOL: f64 = 0.01;

#[tokio::test(flavor = "current_thread")]
async fn audio_encoder_parity() {
    let _trace = thinfer_core::trace::init_from_env();
    let fr = manifest::MANIFEST
        .get(role::AUDIO_VAE)
        .expect("audio vae role");
    let Some(path) = cache::resolve(fr) else {
        eprintln!("skipped[minimax audio_encoder_parity]: audio VAE not in HF cache");
        return;
    };
    let dir = ensure_pyref(&path);
    let meta: Vec<usize> = std::fs::read_to_string(dir.join("meta.txt"))
        .expect("meta.txt")
        .split_whitespace()
        .map(|t| t.parse().expect("meta int"))
        .collect();
    let [samples, t] = meta[..] else {
        panic!("meta.txt needs 2 fields")
    };

    let backend = Arc::new(
        WgpuBackend::new_with_config(WgpuConfig {
            power_preference: PowerPreference::HighPerformance,
            timestamps: std::env::var("THINFER_TRACE").is_ok(),
            ..Default::default()
        })
        .await
        .expect("wgpu backend"),
    );
    let opener = MmapFileOpener::new(&path).await.expect("open audio vae");
    let src = SafetensorsSource::open(opener)
        .await
        .expect("parse audio vae");
    let residency = WeightResidency::new(
        src,
        ResidencyBudget {
            ram_bytes: 4 << 30,
            vram_bytes: 4 << 30,
        },
    );
    let cfg = AudioVaeConfig::h3();
    let enc = AudioEncoder::new(&residency, cfg.clone())
        .await
        .expect("build audio encoder");
    let pl = AudioVae::compile(&backend).await.expect("compile");
    let ws = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    let wave = read_f32(&dir.join("input.bin"));
    assert_eq!(wave.len(), 2 * samples);
    assert_eq!(enc.num_latents(samples), t, "latent count");

    let mut ch0 = wave[..samples].to_vec();
    ch0.resize(t * cfg.hop_length(), 0.0);
    let started = std::time::Instant::now();
    let tower = enc
        .tower(&backend, &residency, &ws, &pl, &ch0)
        .await
        .expect("tower");
    eprintln!("tower (1 ch, {samples} samples): {:?}", started.elapsed());
    let (ts, tr) = report("tower ch0", &read_f32(&dir.join("tower.bin")), &tower);

    let started = std::time::Instant::now();
    let (latents, got_t) = enc
        .encode(&backend, &residency, &ws, &pl, &wave, samples)
        .await
        .expect("encode");
    eprintln!("encode (2 ch): {:?}", started.elapsed());
    assert_eq!(got_t, t);
    let (ls, lr) = report("latents", &read_f32(&dir.join("latents.bin")), &latents);

    assert!(
        (ts - 1.0).abs() < SLOPE_TOL && tr < REL_TOL,
        "tower slope {ts} rel {tr}"
    );
    assert!(
        (ls - 1.0).abs() < SLOPE_TOL && lr < REL_TOL,
        "latents slope {ls} rel {lr}"
    );
}

fn ensure_pyref(vae: &Path) -> PathBuf {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("minimax_audio_enc_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("done.txt");
    if marker.exists() {
        eprintln!(
            "minimax audio-encoder parity: reusing cached pyref dumps ({})",
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
        .arg("thinfer_pytorch_ref.minimax.gen_audio_enc_ref")
        .arg("--vae")
        .arg(vae)
        .arg("--out")
        .arg(&tmp)
        .env("OMP_NUM_THREADS", "4")
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "minimax audio encoder pyref failed");
    std::fs::write(&marker, "ok").expect("write marker");
    tmp
}
