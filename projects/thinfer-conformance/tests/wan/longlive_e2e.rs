//! LongLive-2.0-5B AR (causal/streaming) end-to-end engine test on the real
//! weights. Drives the full AR path - tokenize -> umT5 encode -> per-chunk
//! 4-step FlowUniPC denoise over the windowed KV cache (with the timestep-0 clean
//! recache between chunks) -> TI2V VAE decode -> CTHW frames - and asserts the
//! output is finite, clamped, and non-trivial across a MULTI-CHUNK run (so the
//! window/commit/causal path is actually exercised; a single-chunk run validates
//! none of the new AR code).
//!
//! This is the engine gate for the AR backbone, NOT the pyref byte-compare (that
//! is the next step: the reference `inference.py` on the same `.pt` + pinned
//! noise, ending in a user pause). Health plus multi-chunk coverage here proves
//! the GPU AR self-attn, KV cache, and chunk loop wire up correctly.
//!
//! Gated on `THINFER_LONGLIVE_PT` pointing at `model_bf16.pt` (skips when unset)
//! AND the FastWan base bundle (umT5 + VAE + tokenizer) being in the HF cache
//! (skips, never downloads). Run:
//!   `THINFER_LONGLIVE_PT=<path> cargo test -p thinfer-conformance --features \
//!    wan-e2e --release longlive_e2e -- --nocapture --test-threads=1`
//!
//! Knobs: `THINFER_LL_{WIDTH,HEIGHT,FRAMES}` (defaults 128x128, 61 frames = 16
//! latent frames = 2 chunks of 8); `THINFER_LL_BUDGET_GB` residency budget.

#![cfg(feature = "wan-e2e")]

use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_models::wan::dit_block::WanDitConfig;
use thinfer_models::wan::manifest::{self, role};
use thinfer_models::wan::pipeline::{GenerationParams, VaeChoice, VideoSampler, WanModel};
use thinfer_models::wan::source::open_longlive_source;
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;
use thinfer_native::tokenizer::HfTokenizer;

const PROMPT: &str = "a red balloon drifting over a green field, smooth camera pan";
const SEED: u64 = 42;
const TEMPORAL_SCALE: usize = 4;
/// LongLive chunk size in latent frames (`num_frame_per_block`).
const CHUNK_FRAMES: usize = 8;

#[tokio::test(flavor = "current_thread")]
async fn longlive_e2e_ar() {
    let _ = thinfer_core::trace::init_from_env();

    let Some(pt_path) = std::env::var_os("THINFER_LONGLIVE_PT") else {
        eprintln!("longlive_e2e: THINFER_LONGLIVE_PT unset; skipping");
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
    // 61 frames -> f_lat = 16 = 2 chunks of 8. The multi-chunk run is the whole
    // point: chunk 1 attends chunk 0's committed (clean-pass) window.
    let num_frames = env_u32("THINFER_LL_FRAMES", 61);
    let budget_gb: u64 = env_u32("THINFER_LL_BUDGET_GB", 8) as u64;

    let f_lat = (num_frames as usize - 1) / TEMPORAL_SCALE + 1;
    let num_chunks = f_lat / CHUNK_FRAMES;
    assert!(
        f_lat.is_multiple_of(CHUNK_FRAMES) && num_chunks >= 2,
        "longlive_e2e needs a >=2-chunk run: f_lat={f_lat} must be a multiple of {CHUNK_FRAMES} \
         and >= {}",
        2 * CHUNK_FRAMES
    );
    eprintln!(
        "longlive_e2e: {width}x{height} frames={num_frames} (f_lat={f_lat}, chunks={num_chunks})"
    );

    // Base bundle (umT5 + VAE + tokenizer) from the FastWan repo; the `.pt` DiT is
    // the LongLive-specific file. Skip cleanly if the base misses the cache.
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
            Ok("high") => PowerPreference::HighPerformance,
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
        prompt: PROMPT.to_string(),
        height,
        width,
        num_frames,
        seed: SEED,
        // AR path drives its own UniPC; this field is ignored here.
        sampler: VideoSampler::default(),
    };
    let progress = |_ev: thinfer_models::wan::pipeline::ProgressEvent| {};
    let video = model
        .generate_ar(&params, &[], VaeChoice::Full, Some(&progress))
        .await
        .expect("generate_ar");

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
        "decoded video has {nonfinite} non-finite samples"
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
    // Non-trivial: a NaN-free-but-constant frame (e.g. a dead cache) would pass
    // the range check. Require real spatial variance.
    let n = video.frames.len() as f64;
    let mean = video.frames.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = video
        .frames
        .iter()
        .map(|&x| (x as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    eprintln!("longlive_e2e: range=[{lo:.4}, {hi:.4}] mean={mean:.4} var={var:.6}");
    assert!(
        var > 1e-5,
        "decoded video is ~constant (var={var:.3e}); AR path likely dead"
    );

    eprintln!("longlive_e2e: OK ({num_chunks} chunks, {out_frames} frames, finite + clamped)");
}
