//! MiniMax-H3 video VAE decoder health check.
//!
//! The decoder is a 36-layer ViT, not a conv tower, and it is the first H3
//! component whose weights are fully on disk, so it is the first thing that can
//! touch a GPU. This drives the real checkpoint through the real decode path
//! (embed -> 36 blocks -> norm_out -> proj_out -> unpatchify -> tile stitch) at
//! one chunk of latents and asserts the output is finite and in range.
//!
//! It is a HEALTH check, not parity: it proves the pipeline runs, the three new
//! H3 kernels dispatch against real weights, and nothing produces NaN. Numerical
//! parity against the reference decoder is a separate gate and is NOT implied by
//! this passing.

#![cfg(feature = "minimax-e2e")]

use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::minimax::manifest::{self, role};
use thinfer_models::minimax::video_vae::{
    DecoderPipelines, TilePolicy, VideoVaeConfig, decode_clip, load_suffix_tokens, register_handles,
};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

/// One decode chunk: `tokens_chunk_size + token_overlap` latent frames, which is
/// what `decode` feeds `decode_clip`.
const LATENT_FRAMES: usize = 7;
/// A single 256px tile's worth of latents, so this stays one batched pass.
const LATENT_H: usize = 16;
const LATENT_W: usize = 16;

#[tokio::test(flavor = "current_thread")]
async fn video_vae_health() {
    let _trace = thinfer_core::trace::init_from_env();

    let fr = manifest::MANIFEST
        .get(role::VIDEO_VAE)
        .expect("video vae role");
    let Some(path) = cache::resolve(fr) else {
        eprintln!("skipped[minimax video_vae_health]: video VAE not in HF cache");
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

    let cfg = VideoVaeConfig::h3();
    let opener = MmapFileOpener::new(&path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let src = SafetensorsSource::open(opener)
        .await
        .expect("parse video vae safetensors");
    let residency = WeightResidency::new(
        src,
        ResidencyBudget {
            ram_bytes: 16 << 30,
            vram_bytes: 6 << 30,
        },
    );
    let handles = register_handles(&cfg, &residency).expect("register decoder handles");
    let pipelines = DecoderPipelines::compile(&backend)
        .await
        .expect("compile decoder pipelines");
    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    let suffix = load_suffix_tokens(&cfg, &backend, &residency, &handles)
        .await
        .expect("read register + mask tokens");
    assert_eq!(
        suffix.len(),
        cfg.suffix_tokens() * cfg.dim(),
        "suffix token count"
    );

    // A deterministic latent standing in for a denoised one. Unit-scale normal
    // is the right order of magnitude for a normalized latent.
    let n = LATENT_FRAMES * LATENT_H * LATENT_W * cfg.latent_channels;
    let z: Vec<f32> = (0..n)
        .map(|i| {
            let x = (i as f32 * 0.6180339887).fract() * 2.0 - 1.0;
            x * 1.5
        })
        .collect();

    let out = decode_clip(
        &cfg,
        &backend,
        &pipelines,
        &residency,
        &workspace,
        &handles,
        &suffix,
        TilePolicy::default(),
        &z,
        LATENT_FRAMES,
        LATENT_H,
        LATENT_W,
    )
    .await
    .expect("decode_clip");

    assert_eq!(out.frames, LATENT_FRAMES * cfg.temporal_compression);
    assert_eq!(out.height, LATENT_H * cfg.spatial_compression);
    assert_eq!(out.width, LATENT_W * cfg.spatial_compression);
    assert_eq!(out.channels, cfg.out_channels);
    assert_eq!(
        out.data.len(),
        out.frames * out.height * out.width * out.channels
    );

    let nonfinite = out.data.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(nonfinite, 0, "{nonfinite} non-finite samples in the decode");

    let (mut lo, mut hi, mut sum) = (f32::MAX, f32::MIN, 0.0f64);
    for &v in &out.data {
        lo = lo.min(v);
        hi = hi.max(v);
        sum += v as f64;
    }
    let mean = sum / out.data.len() as f64;
    eprintln!(
        "---- minimax video vae health ----\n\
         {}x{} x {} frames, range [{lo:.3}, {hi:.3}], mean {mean:.4}",
        out.width, out.height, out.frames
    );

    // The decoder emits ImageNet-normalized RGB over a [0, 1] base, so pixels
    // land roughly in [-2.2, 2.7]. A decode stuck at zero or blown up by orders
    // of magnitude is the failure this catches.
    assert!(hi > lo, "decode is constant");
    assert!(
        hi.abs() < 100.0 && lo.abs() < 100.0,
        "decode out of any plausible range: [{lo}, {hi}]"
    );
}
