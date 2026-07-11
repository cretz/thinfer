//! LTX-2 rapid VAE encode->decode round-trip: a fast integration check that the
//! new `LtxVaeEncoder` composes with the shipped `LtxVaeDecoder` in the SAME
//! normalized latent space (channel order, patchify/unpatchify, and the
//! normalize/un-normalize pair all consistent). This is what the encoder-vs-
//! reference parity gate can NOT catch on its own -- it only checks the encoder
//! against upstream, not that encode(x) then decode() returns to pixel space.
//!
//! A VAE round-trip is lossy (32x spatial compression), so this asserts the
//! decoded frame is finite, non-degenerate, and strongly positively correlated
//! with a smooth input gradient (a channel-swap / normalize-direction bug would
//! destroy the correlation or flip the slope). No pyref, no DiT -- both VAE
//! halves only, single frame, so it runs in a couple of seconds.

#![cfg(feature = "ltx-e2e")]

use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::ops::ActDtype;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::ltx::manifest::{self, role};
use thinfer_models::ltx::video_vae::{
    LtxVaeConfig, LtxVaeDecoder, LtxVaeEncoder, LtxVaeEncoderConfig, LtxVaePipelines,
    load_latent_stats,
};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::report;

// Single frame, 256x256 -> latent [128,1,8,8] -> decode back to [3,1,256,256].
const H: usize = 256;
const W: usize = 256;

#[tokio::test(flavor = "current_thread")]
async fn rapid_vae_roundtrip() {
    let _trace = thinfer_core::trace::init_from_env();

    let vae_fr = manifest::LTX2_RAPID_MANIFEST
        .get(role::VIDEO_VAE)
        .expect("rapid vae role");
    let Some(vae_path) = cache::resolve(vae_fr) else {
        eprintln!("skipped[ltx rapid_vae_roundtrip]: LTX-2 video VAE not in HF cache");
        return;
    };

    // Smooth input in [-1,1]: a diagonal gradient, distinct per channel so a
    // channel swap would show. `[3, 1, H, W]` CTHW.
    let mut frame = vec![0.0f32; 3 * H * W];
    for c in 0..3 {
        for y in 0..H {
            for x in 0..W {
                let g = (x + y) as f32 / (H + W) as f32; // 0..1 diagonal
                let v = (g + 0.15 * c as f32).min(1.0) * 2.0 - 1.0;
                frame[c * H * W + y * W + x] = v;
            }
        }
    }

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

    let opener = MmapFileOpener::new(&vae_path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", vae_path.display()));
    let src = SafetensorsSource::open(opener)
        .await
        .expect("parse vae safetensors");
    let residency = WeightResidency::new(
        src,
        ResidencyBudget {
            ram_bytes: 16 << 30,
            vram_bytes: 6 << 30,
        },
    );
    let (mean, std) = load_latent_stats(&residency, &backend)
        .await
        .expect("load latent stats");
    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    // Encode (f32 acts for a clean reference).
    let enc_pl = LtxVaePipelines::compile_with(&backend, ActDtype::F32)
        .await
        .expect("compile enc pipelines");
    let encoder = LtxVaeEncoder::new(
        enc_pl,
        &residency,
        mean.clone(),
        std.clone(),
        LtxVaeEncoderConfig::ltx2_rapid(),
    )
    .expect("build encoder");
    let latent = encoder
        .encode(&backend, &residency, &workspace, &frame, 1, H, W)
        .await
        .expect("encode");
    let (lh, lw) = (H / 32, W / 32);
    assert_eq!(latent.len(), 128 * lh * lw, "latent size");
    assert!(latent.iter().all(|v| v.is_finite()), "non-finite latent");
    let lat_var = variance(&latent);
    assert!(lat_var > 1e-3, "degenerate latent (var {lat_var})");

    // Decode the same latent back to pixels.
    let dec_pl = LtxVaePipelines::compile_with(&backend, ActDtype::F32)
        .await
        .expect("compile dec pipelines");
    let decoder =
        LtxVaeDecoder::new_with_config(dec_pl, &residency, mean, std, LtxVaeConfig::ltx2_rapid())
            .expect("build decoder");
    let video = decoder
        .decode(&backend, &residency, &workspace, &latent, 1, lh, lw)
        .await
        .expect("decode");
    assert_eq!(video.len(), 3 * H * W, "decoded size");
    assert!(video.iter().all(|v| v.is_finite()), "non-finite decode");

    // The decoded frame should track the input gradient: strong positive slope,
    // and each channel individually correlated (guards channel order). Round-trip
    // is lossy at 32x, so the band is generous -- we only assert it did NOT scramble.
    let (slope, rel) = report("roundtrip", &frame, &video);
    eprintln!("ltx rapid vae roundtrip: latent var {lat_var:.3}, slope {slope:.3}, rel {rel:.3}");
    assert!(
        slope.is_finite() && slope > 0.5,
        "round-trip slope {slope:.3} not positive (channel/normalize bug?)"
    );
    for c in 0..3 {
        let fi = &frame[c * H * W..(c + 1) * H * W];
        let vi = &video[c * H * W..(c + 1) * H * W];
        let r = pearson(fi, vi);
        eprintln!("  channel {c}: pearson {r:.3}");
        assert!(r > 0.85, "channel {c} correlation {r:.3} too low");
    }
}

fn variance(x: &[f32]) -> f32 {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n
}

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let mut num = 0.0;
    let (mut da, mut db) = (0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        let (dx, dy) = (x - ma, y - mb);
        num += dx * dy;
        da += dx * dx;
        db += dy * dy;
    }
    num / (da.sqrt() * db.sqrt() + 1e-12)
}
