//! HunyuanVideo 1.5 VAE up-stage TILING parity: the tiled up-stage decode
//! (`decode_with_tiles`, explicit multi-tile plan) vs the whole-tensor decode
//! (`decode_with_taps`). Both share conv_in+mid (whole tensor); only the
//! up-stages differ (tiled overlap-blend vs single submit). A correct blend is
//! close-but-not-bit-exact (causal-conv replicate pad differs at tile seams);
//! an indexing bug in gather/blend / the causal temporal shift decorrelates the
//! output. Engine-vs-engine, so no pyref / HF download is needed beyond the VAE
//! weights. Forces a MULTI-TILE TEMPORAL case (the non-trivial seam): `tf<f` and
//! `tile<h,w`.

#![cfg(feature = "hunyuan-e2e")]

use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::ops::ActDtype;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::hunyuan::vae::{HunyuanVaeDecoder, HunyuanVaePipelines, LATENT_CHANNELS};
use thinfer_native::MmapFileOpener;

use crate::parity_util::{report, resolve_hf};

// Dims kept small enough that the WHOLE reference up-stage decode still fits
// (that single-submit path is what OOMs at 480p; tiling is the fix being
// verified). Multi-tile = tile 3 < 6 (both spatial axes) + tf 2 < 3 (temporal).
const F: usize = 3;
const H: usize = 6;
const W: usize = 6;
const TF: usize = 2;
const TILE: u32 = 3;

// Single-tile (tf>=F, tile>=H,W) must reduce to the whole decode bit-for-bit
// (modulo f32 reorder) -- this is the strong catch for the per-tile path
// (gather / run_upstages / blend / normalize).
const SINGLE_REL_TOL: f64 = 0.001;

// Multi-tile: a correct overlap-blend stays correlated with the whole decode
// (slope ~1); a gather/blend/temporal-shift indexing bug decorrelates (slope
// far from 1, rel -> tens of percent of garbage). The rel FLOOR here is the
// tiny-tile blend approximation, NOT a bug: overlap is clamped to 1 latent cell
// (< the up-stack conv receptive field), so seams are under-reconstructed;
// production tiles are far larger with proportionally more overlap. The job of
// this band is to catch decorrelation, gated tight on slope.
const MULTI_SLOPE_TOL: f64 = 0.08;
const MULTI_REL_TOL: f64 = 0.18;

/// Deterministic standard-normal latent (splitmix64 + Box-Muller), CTHW.
fn seeded_latent(n: usize) -> Vec<f32> {
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let u1 = next().max(1e-12);
            let u2 = next();
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        })
        .collect()
}

#[tokio::test(flavor = "current_thread")]
async fn vae_tiling_parity() {
    let _trace = thinfer_core::trace::init_from_env();

    let Some(vae_path) = resolve_hf(
        "THINFER_HUNYUAN_VAE",
        "models--Comfy-Org--HunyuanVideo_1.5_repackaged",
        "split_files/vae/hunyuanvideo15_vae_fp16.safetensors",
    ) else {
        eprintln!("skipped[hunyuan vae_tiling_parity]: VAE not in HF cache");
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

    let latent = seeded_latent(LATENT_CHANNELS * F * H * W);

    // Whole-tensor reference (single up-stage submit).
    let pipelines = HunyuanVaePipelines::compile_with(&backend, ActDtype::F32)
        .await
        .expect("compile vae pipelines");
    let decoder = HunyuanVaeDecoder::new(pipelines, &residency).expect("build decoder");
    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));
    let whole = decoder
        .decode_with_taps(&backend, &residency, &workspace, &latent, F, H, W, None)
        .await
        .expect("whole vae decode");

    eprintln!("---- hunyuan vae up-stage tiling parity ----");

    // Single tile (tf=F, tile>=H,W) must be bit-identical to the whole decode.
    let single = decoder
        .decode_with_tiles(
            &backend,
            &residency,
            &workspace,
            &latent,
            F,
            H,
            W,
            F,
            H.max(W) as u32,
        )
        .await
        .expect("single-tile vae decode");
    assert_eq!(whole.len(), single.len(), "video len whole vs single tile");
    let (s_slope, s_rel) = report("single_tile", &whole, &single);
    assert!(
        s_slope.is_finite() && (s_slope - 1.0).abs() <= SINGLE_REL_TOL && s_rel <= SINGLE_REL_TOL,
        "single-tile NOT bit-exact vs whole: slope={s_slope:.6} rel={:.4}% (per-tile path bug)",
        s_rel * 100.0,
    );

    // Multi-tile (explicit tf<F, tile<H,W) -- must stay correlated (no seam bug).
    let tiled = decoder
        .decode_with_tiles(&backend, &residency, &workspace, &latent, F, H, W, TF, TILE)
        .await
        .expect("multi-tile vae decode");
    assert_eq!(whole.len(), tiled.len(), "video len whole vs multi tile");
    let (m_slope, m_rel) = report("multi_tile", &whole, &tiled);
    assert!(
        m_slope.is_finite() && (m_slope - 1.0).abs() <= MULTI_SLOPE_TOL && m_rel <= MULTI_REL_TOL,
        "multi-tile drift: slope={m_slope:.5} rel={:.3}% (tol slope 1+-{MULTI_SLOPE_TOL}, rel {:.3}%) \
         -- decorrelation indicates a gather/blend/temporal-shift bug",
        m_rel * 100.0,
        MULTI_REL_TOL * 100.0,
    );

    // F16 up-stages (the production perf path: F32 mid + F16 conv-heavy up-stages)
    // vs the F32 whole reference. f16 storage is well within range (acts |x|<~140)
    // so this is a clean precision band (silu computes in f32; only storage is
    // f16). Single tile to isolate the precision delta from the blend floor.
    let mid_f32 = HunyuanVaePipelines::compile_with(&backend, ActDtype::F32)
        .await
        .expect("compile vae mid f32");
    let up_f16 = HunyuanVaePipelines::compile_with(&backend, ActDtype::F16)
        .await
        .expect("compile vae up f16");
    let decoder_f16 =
        HunyuanVaeDecoder::new_mixed(mid_f32, Some(up_f16), &residency).expect("build f16 decoder");
    let f16_video = decoder_f16
        .decode_with_tiles(
            &backend,
            &residency,
            &workspace,
            &latent,
            F,
            H,
            W,
            F,
            H.max(W) as u32,
        )
        .await
        .expect("f16 up-stage vae decode");
    assert_eq!(whole.len(), f16_video.len(), "video len whole vs f16");
    assert!(
        f16_video.iter().all(|v| v.is_finite()),
        "f16 up-stage decode produced non-finite output"
    );
    let (f_slope, f_rel) = report("f16_upstage", &whole, &f16_video);
    // f16 precision over the conv-heavy up-stack; comfortably tighter than the
    // multi-tile blend floor. Catches an overflow/regression in the f16 path.
    const F16_SLOPE_TOL: f64 = 0.02;
    const F16_REL_TOL: f64 = 0.02;
    assert!(
        f_slope.is_finite() && (f_slope - 1.0).abs() <= F16_SLOPE_TOL && f_rel <= F16_REL_TOL,
        "f16 up-stage drift: slope={f_slope:.5} rel={:.3}% (tol slope 1+-{F16_SLOPE_TOL}, rel {:.3}%)",
        f_rel * 100.0,
        F16_REL_TOL * 100.0,
    );
}
