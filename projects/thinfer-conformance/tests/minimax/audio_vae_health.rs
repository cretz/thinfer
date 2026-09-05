//! MiniMax-H3 audio VAE decoder health check.
//!
//! H3's audio decoder is a BigVGAN v2 tower, so this also exercises the
//! `common::bigvgan` extraction against a SECOND checkpoint with different
//! hyperparameters and tensor naming (nested `ups.N.0`, interleaved
//! `activations.N`, a `dec_in_proj` the LTX vocoder has no equivalent of).
//!
//! Health, not parity: it proves the tower runs at H3's rates and emits the
//! right sample count in a plausible waveform range.

#![cfg(feature = "minimax-e2e")]

use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::minimax::audio_vae::{AudioVae, AudioVaeConfig};
use thinfer_models::minimax::manifest::{self, role};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

/// Half a second at 40 latents/s.
const FRAMES: usize = 20;

#[tokio::test(flavor = "current_thread")]
async fn audio_vae_health() {
    let _trace = thinfer_core::trace::init_from_env();

    let fr = manifest::MANIFEST
        .get(role::AUDIO_VAE)
        .expect("audio vae role");
    let Some(path) = cache::resolve(fr) else {
        eprintln!("skipped[minimax audio_vae_health]: audio VAE not in HF cache");
        return;
    };

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

    let cfg = AudioVaeConfig::h3();
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
    let pipelines = AudioVae::compile(&backend)
        .await
        .expect("compile audio vae pipelines");
    let vae = AudioVae::new(&backend, &residency, cfg.clone())
        .await
        .expect("build audio vae");
    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    // Deterministic stand-in latents, `[channels, latent_channels, frames]`.
    let n = cfg.channels * cfg.latent_channels * FRAMES;
    let z: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.6180339887).fract() * 2.0 - 1.0)
        .collect();

    let wav = vae
        .decode(&backend, &residency, &workspace, &pipelines, &z, FRAMES)
        .await
        .expect("audio decode");

    let want = cfg.channels * FRAMES * cfg.hop_length();
    assert_eq!(wav.len(), want, "sample count");

    let nonfinite = wav.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(nonfinite, 0, "{nonfinite} non-finite samples");

    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for &v in &wav {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    eprintln!(
        "---- minimax audio vae health ----\n\
         {} samples ({} ch x {} frames x {} hop), range [{lo:.4}, {hi:.4}]",
        wav.len(),
        cfg.channels,
        FRAMES,
        cfg.hop_length()
    );

    // The decoder clamps to [-1, 1]; a tower that is silent or railed is the
    // failure this catches.
    assert!(lo >= -1.0 && hi <= 1.0, "outside the clamp: [{lo}, {hi}]");
    assert!(hi > lo, "waveform is constant");
}
