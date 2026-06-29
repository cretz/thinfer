//! LongLive-2.0-5B multi-shot AR end-to-end test on the real weights. Drives the
//! scene-cut path that single-prompt `longlive_e2e` never exercises: a per-shot
//! prompt list, the in-band `SCENE_CUT_PREFIX` on each shot boundary, and the
//! cache's shot-boundary bookkeeping (RoPE temporal-phase advance before the
//! boundary chunk + chunk pinning after its clean commit). Asserts the decoded
//! video is finite, clamped, and non-trivial across a >=2-shot run, so the
//! boundary chunk's prefixed cross-attn + advanced RoPE + pinned sink all wire up
//! without NaNs or a dead cache.
//!
//! This is a health gate, not a pyref byte-compare: the single-forward parity is
//! covered by `longlive_parity`, and the AR-depth precision compounding is
//! tolerated by design (see worklog). Multi-shot adds no new ops, only cache
//! routing, so finite + variance across a scene cut is the right bar.
//!
//! Gated on `THINFER_LONGLIVE_PT` (skips when unset) AND the FastWan base bundle
//! (umT5 + VAE + tokenizer) being in the HF cache (skips, never downloads). Run:
//!   `THINFER_LONGLIVE_PT=<path> cargo test -p thinfer-conformance --features \
//!    wan-e2e --release longlive_multishot -- --nocapture --test-threads=1`
//!
//! Knobs: `THINFER_LL_{WIDTH,HEIGHT}` (default 128x128); `THINFER_LL_BUDGET_GB`.

#![cfg(feature = "wan-e2e")]

use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_models::wan::dit_block::WanDitConfig;
use thinfer_models::wan::manifest::{self, role};
use thinfer_models::wan::pipeline::{GenerationParams, Shot, VaeChoice, VideoSampler, WanModel};
use thinfer_models::wan::source::open_longlive_source;
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;
use thinfer_native::tokenizer::HfTokenizer;

/// Two shots, one chunk each -> 2 chunks = f_lat 16 = 61 output frames. The shot
/// boundary at chunk 1 fires the scene-cut path; distinct prompts make the cut
/// semantically real (cross-attn switches text mid-video).
const SHOT_A: &str = "a red balloon drifting over a green field, smooth camera pan";
const SHOT_B: &str = "a blue kite diving through grey storm clouds, fast motion";
const SEED: u64 = 42;
const TEMPORAL_SCALE: usize = 4;
const CHUNK_FRAMES: usize = 8;

#[tokio::test(flavor = "current_thread")]
async fn longlive_multishot_e2e_ar() {
    let _ = thinfer_core::trace::init_from_env();

    let Some(pt_path) = std::env::var_os("THINFER_LONGLIVE_PT") else {
        eprintln!("longlive_multishot_e2e: THINFER_LONGLIVE_PT unset; skipping");
        return;
    };

    let env_u32 = |k: &str, d: u32| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let width = env_u32("THINFER_LL_WIDTH", 128);
    let height = env_u32("THINFER_LL_HEIGHT", 128);
    let budget_gb: u64 = env_u32("THINFER_LL_BUDGET_GB", 8) as u64;

    // Two single-chunk shots. num_frames must match the engine's chunk math:
    // f_lat = sum(shot.chunks) * CHUNK_FRAMES, frames = 4 * f_lat - 3.
    let shots = vec![
        Shot {
            prompt: SHOT_A.to_string(),
            chunks: 1,
        },
        Shot {
            prompt: SHOT_B.to_string(),
            chunks: 1,
        },
    ];
    let total_chunks: usize = shots.iter().map(|s| s.chunks).sum();
    let f_lat = total_chunks * CHUNK_FRAMES;
    let num_frames = (TEMPORAL_SCALE * f_lat - 3) as u32;
    eprintln!(
        "longlive_multishot_e2e: {width}x{height} frames={num_frames} \
         (f_lat={f_lat}, {} shots, chunks={total_chunks})",
        shots.len()
    );

    let base_roles = [
        role::TEXT_ENCODER_SHARD_1,
        role::TEXT_ENCODER_SHARD_2,
        role::TEXT_ENCODER_SHARD_3,
        role::VAE,
    ];
    let mut base_openers: Vec<MmapFileOpener> = Vec::with_capacity(base_roles.len());
    for r in base_roles {
        let fr = manifest::MANIFEST.get(r).expect("role in manifest");
        let Some(p) = cache::resolve(fr) else {
            eprintln!(
                "skipped: {}/{} not in HF cache ({})",
                fr.repo,
                fr.path,
                cache::cache_root().display()
            );
            return;
        };
        base_openers.push(
            MmapFileOpener::new(&p)
                .await
                .unwrap_or_else(|e| panic!("open {}: {e}", p.display())),
        );
    }
    let tok_fr = manifest::MANIFEST
        .get(role::TOKENIZER_JSON)
        .expect("tok role");
    let Some(tok_path) = cache::resolve(tok_fr) else {
        eprintln!("skipped: tokenizer not in HF cache");
        return;
    };
    let tokenizer = HfTokenizer::from_path(&tok_path).await.expect("tokenizer");

    let dit_opener = MmapFileOpener::new(&pt_path)
        .await
        .unwrap_or_else(|e| panic!("open .pt {pt_path:?}: {e}"));
    let num_layers = WanDitConfig::longlive_2_0_5b().num_layers;
    let source = open_longlive_source(dit_opener, base_openers, num_layers)
        .await
        .unwrap_or_else(|e| panic!("open LongLive source: {e:?}"));

    let cfg = WgpuConfig {
        power_preference: match std::env::var("THINFER_POWER_PREF").as_deref() {
            Ok("low") => PowerPreference::LowPower,
            _ => PowerPreference::HighPerformance,
        },
        timestamps: std::env::var("THINFER_TRACE").is_ok(),
        disable_coopmat: std::env::var("THINFER_NO_COOPMAT").is_ok(),
    };
    let backend = Arc::new(
        WgpuBackend::new_with_config(cfg)
            .await
            .expect("wgpu adapter unavailable"),
    );
    let budget = ResidencyBudget {
        ram_bytes: budget_gb << 30,
        vram_bytes: budget_gb << 30,
    };
    let residency = WeightResidency::new(source, budget);
    let model = WanModel::load(
        Arc::clone(&backend),
        residency,
        tokenizer,
        VaeChoice::Full,
        std::env::var_os("THINFER_WAN_I8_FFN").is_some(),
    )
    .await
    .expect("WanModel::load (LongLive)");

    let params = GenerationParams {
        prompt: SHOT_A.to_string(), // ignored on the multi-shot path; shots drive it
        height,
        width,
        num_frames,
        seed: SEED,
        sampler: VideoSampler::default(), // AR path; ignored
        attn_window: None,
    };
    let progress = |_ev: thinfer_models::wan::pipeline::ProgressEvent| {};
    let video = model
        .generate_ar(&params, &shots, VaeChoice::Full, Some(&progress))
        .await
        .expect("generate_ar (multi-shot)");

    // --- health: finite, clamped, non-trivial ---
    let out_frames = TEMPORAL_SCALE * f_lat - 3;
    assert_eq!(video.num_frames, out_frames, "decoded frame count");
    assert_eq!(video.height, height as usize);
    assert_eq!(video.width, width as usize);
    assert_eq!(
        video.frames.len(),
        3 * out_frames * height as usize * width as usize,
        "frame buffer size"
    );

    let nonfinite = video.frames.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(
        nonfinite, 0,
        "multi-shot video has {nonfinite} non-finite samples"
    );
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &x in &video.frames {
        lo = lo.min(x);
        hi = hi.max(x);
        assert!(
            (-1.0..=1.0).contains(&x),
            "decoded sample {x} out of [-1, 1]"
        );
    }
    let n = video.frames.len() as f64;
    let mean = video.frames.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = video
        .frames
        .iter()
        .map(|&x| (x as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    eprintln!("longlive_multishot_e2e: range=[{lo:.4}, {hi:.4}] mean={mean:.4} var={var:.6}");
    assert!(
        var > 1e-5,
        "decoded video is ~constant (var={var:.3e}); scene-cut path likely dead"
    );

    // Diagnostic (not asserted: VAE temporal mixing blends across the boundary,
    // so a strict cross-shot delta would be flaky at one chunk/shot): print the
    // first- vs last-frame mean so a stuck-cache regression (shot B == shot A) is
    // visible in the log. CTHW channel-planar, so gather each frame across C.
    let hw = height as usize * width as usize;
    let per_frame = 3 * hw;
    let frame_mean = |t: usize| -> f64 {
        let plane = out_frames * hw;
        let base = t * hw;
        let mut s = 0.0f64;
        for c in 0..3 {
            for px in 0..hw {
                s += video.frames[c * plane + base + px] as f64;
            }
        }
        s / per_frame as f64
    };
    eprintln!(
        "longlive_multishot_e2e: first-frame mean={:.4} last-frame mean={:.4}",
        frame_mean(0),
        frame_mean(out_frames - 1)
    );

    eprintln!(
        "longlive_multishot_e2e: OK ({} shots, {out_frames} frames, finite + clamped)",
        shots.len()
    );
}
