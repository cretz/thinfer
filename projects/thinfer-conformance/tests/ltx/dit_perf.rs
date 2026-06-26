//! LTX-2.3 DiT-only perf harness (NOT a parity/health gate -- a measurement
//! vehicle for the denoise-loop hot path). Loads the real Q8_0 DiT GGUF, builds
//! correctly-shaped (random) connector KV + latents at env-configurable dims, and
//! runs `denoise_loop` for a few steps, dumping the `THINFER_TRACE` rollup so the
//! `gpu_ms by pipeline` + per-scope streaming/submit costs are visible.
//!
//! The full e2e (`t2v_e2e_health`) runs at tiny dims (2-8 video tokens) so it is
//! streaming/sync-bound and never exercises the matmul/sdpa ceiling. This harness
//! lets us pick dims that put real compute through the blocks (i8 DP4A + f16 SDPA
//! wins show here) while skipping the encoder/VAE/pyref phases entirely.
//!
//! Env knobs (all optional):
//!   LTX_PERF_FRAMES  (default 25)   pixel frames (8k+1)
//!   LTX_PERF_HEIGHT  (default 256)  pixel height (/32 -> latent)
//!   LTX_PERF_WIDTH   (default 256)  pixel width
//!   LTX_PERF_STEPS   (default 4)    denoise steps (<= 8)
//!   LTX_PERF_LAYERS  (default 48)   DiT blocks (lower = faster iteration)
//!   THINFER_E2E_BUDGET_GB (default 6) VRAM budget GB
//!
//! Run: THINFER_TRACE=1 THINFER_POWER_PREF=high cargo test -p thinfer-conformance
//!   --features ltx-e2e --release dit_perf -- --nocapture --test-threads=1

#![cfg(feature = "ltx-e2e")]

use std::sync::Arc;
use std::time::Instant;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::ltx::config as dit;
use thinfer_models::ltx::connector::CONN_SEQ;
use thinfer_models::ltx::dit::{DitModel, DitPipelines};
use thinfer_models::ltx::manifest::{self, role};
use thinfer_models::ltx::pipeline::{build_dit_freqs, denoise_loop, streams_for};
use thinfer_models::ltx::sampler::{self, AudioLatentDims, STAGE1_SIGMAS, VideoLatentDims};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[tokio::test(flavor = "current_thread")]
async fn dit_perf() {
    let trace = thinfer_core::trace::init_from_env();

    // DiT quant select: q8_0 (default baseline) or q4_k_m (mixed Q4_K/Q6_K, the
    // footprint variant). Validates the per-quant-kind dense dequant routing.
    let dit_role = match std::env::var("LTX_DIT_QUANT").as_deref() {
        Ok("q4_k_m" | "q4") => role::DIT_GGUF_Q4_K_M,
        _ => role::DIT_GGUF_Q8_0,
    };
    let dit_fr = manifest::MANIFEST.get(dit_role).expect("dit role");
    let Some(dit_path) = cache::resolve(dit_fr) else {
        eprintln!("skipped[ltx dit_perf]: DiT GGUF ({dit_role}) not in HF cache");
        return;
    };
    eprintln!("ltx dit_perf: DiT role = {dit_role}");

    let frames = env_usize("LTX_PERF_FRAMES", 25);
    let height = env_usize("LTX_PERF_HEIGHT", 256);
    let width = env_usize("LTX_PERF_WIDTH", 256);
    let steps = env_usize("LTX_PERF_STEPS", 4).min(STAGE1_SIGMAS.len() - 1);
    let layers = env_usize("LTX_PERF_LAYERS", dit::NUM_LAYERS).min(dit::NUM_LAYERS);
    let budget_gb = env_usize("THINFER_E2E_BUDGET_GB", 6);
    let fps = 24.0;

    let vd = VideoLatentDims::from_pixels(frames, height, width);
    let ad = AudioLatentDims::from_video(frames, fps);
    let s = streams_for(vd, ad);
    eprintln!(
        "ltx dit_perf: video latent {}x{}x{} ({} tok), audio {} tok, {steps} steps, {layers} layers, budget {budget_gb}G",
        vd.frames,
        vd.height,
        vd.width,
        vd.tokens(),
        ad.tokens(),
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
    let pipelines = DitPipelines::compile(&backend)
        .await
        .expect("compile dit pipelines");
    let opener = MmapFileOpener::new(&dit_path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", dit_path.display()));
    let src = GgufSource::open(opener).await.expect("parse dit gguf");
    let residency = WeightResidency::new(
        src,
        ResidencyBudget {
            ram_bytes: 16 << 30,
            vram_bytes: (budget_gb as u64) << 30,
        },
    );
    let model = DitModel::register(&backend, &residency, layers)
        .await
        .expect("register dit model");
    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    // Random-but-shaped connector KV + latents (perf only; values irrelevant).
    let vtext = sampler::gaussian_noise(CONN_SEQ * dit::DIM, 1);
    let atext = sampler::gaussian_noise(CONN_SEQ * dit::AUDIO_DIM, 2);
    let lat_v = sampler::gaussian_noise(vd.elems(), 3);
    let lat_a = sampler::gaussian_noise(ad.elems(), 4);
    let freqs = build_dit_freqs(vd, ad, fps);
    let sigmas = &STAGE1_SIGMAS[..=steps];

    // Warm-up step (compiles/streams) excluded from the timed window so we
    // measure the steady-state per-step cost.
    let (lat_v, lat_a) = denoise_loop(
        &backend,
        &pipelines,
        &residency,
        &workspace,
        &model,
        s,
        &STAGE1_SIGMAS[..=1],
        lat_v,
        lat_a,
        &vtext,
        &atext,
        &freqs,
        None,
    )
    .await
    .expect("warmup denoise");

    let t0 = Instant::now();
    let (lat_v, _lat_a) = denoise_loop(
        &backend, &pipelines, &residency, &workspace, &model, s, sigmas, lat_v, lat_a, &vtext,
        &atext, &freqs, None,
    )
    .await
    .expect("timed denoise");
    let dt = t0.elapsed();
    assert!(lat_v.iter().all(|v| v.is_finite()), "non-finite latent");

    let per_step = dt.as_secs_f64() / (steps as f64);
    eprintln!(
        "ltx dit_perf: {steps} steps in {:.2}s = {:.3}s/step ({layers} layers, {} video tok)",
        dt.as_secs_f64(),
        per_step,
        vd.tokens(),
    );

    if let Some(h) = &trace {
        let _ = h.dump(&mut std::io::stderr());
    }
}
