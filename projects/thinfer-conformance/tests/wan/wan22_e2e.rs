//! Wan2.2-T2V-A14B (two-expert MoE) e2e HEALTH gate. No pyref: the Wan2.2 path
//! runs the fixed 4-step LightX2V distill (`Wan22DistillSampler`), a different
//! sampler than AnyFlow's any-step flow-map, and a full 14B CPU pyref is
//! infeasible (two ~13GB Q8 expert folds). So this is engine-only health:
//! finiteness, latent variance, decoded-video variance + temporal motion.
//!
//! Load recipe mirrors the executor's `Wan22T2vA14b` arm: the safetensors tail
//! (umT5 shards reused from the FastWan bundle + the Wan2.1 VAE) plus the two
//! folded GGUF experts (each = fold(GGUF, LightX2V distill LoRA) -> Q8_0 blocks),
//! built via `open_wan22_source`, loaded with `WanVariant::wan22_t2v_a14b()` and
//! i8 matmul ON (the production default). The full Wan2.1 VAE is forced (no tiny
//! decoder path exists for A14B). Optional PNG staging via `THINFER_E2E_PNG_DIR`.

use std::path::PathBuf;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::trace;
use thinfer_core::workspace::Workspace;
use thinfer_models::wan::dit_block::WanDitConfig;
use thinfer_models::wan::manifest::{self, role};
use thinfer_models::wan::pipeline::{
    GenerationParams, VaeChoice, VideoSampler, WanModel, WanStepDiag, WanVariant,
};
use thinfer_models::wan::source::open_wan22_source;
use thinfer_native::MmapFileOpener;
use thinfer_native::tokenizer::HfTokenizer;

use thinfer_native::cache;

const PROMPT: &str = "A bright red vintage car drives slowly down a sunlit \
coastal road, ocean waves in the background, realistic style, steady camera.";
const SEED: u64 = 42;
/// Wan2.1 VAE temporal downsample (4x): decoded `out_frames = 4 * f_lat - 3`.
const TEMPORAL_SCALE: usize = 4;
/// The LightX2V distill schedule is fixed at 4 steps
/// (`Wan22DistillConfig::wan22_t2v_a14b().denoising_step_list` = [1000,750,500,
/// 250], boundary 2 -> 2 high-noise + 2 low-noise expert steps). `params.sampler`
/// and `params.steps` are ignored on the MoE path (pipeline.rs step-distill arm),
/// so the emitted step count is always this, independent of `THINFER_E2E_STEPS`.
const WAN22_DISTILL_STEPS: usize = 4;

fn env_u32(k: &str, d: u32) -> u32 {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn std_of(x: &[f32]) -> f64 {
    let n = x.len().max(1) as f64;
    let mean = x.iter().map(|v| *v as f64).sum::<f64>() / n;
    (x.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / n).sqrt()
}

/// Dumps the trace rollup (gpu_ms by pipeline + scope timings) on drop, pass
/// or panic -- the perf-localization table (same guard as `anyflow_e2e` /
/// `video_e2e`).
struct RollupDumpOnDrop(Option<trace::RollupHandle>);
impl Drop for RollupDumpOnDrop {
    fn drop(&mut self) {
        if let Some(h) = self.0.take() {
            let _ = h.dump(&mut std::io::stderr());
        }
    }
}

/// Resolve a manifest role to a cached path, or print the skip line + return
/// `None` (mirrors `anyflow_e2e`'s `skipped[...]` pattern).
fn resolve_or_skip(r: &str) -> Option<PathBuf> {
    let fr = manifest::MANIFEST.get(r).expect("role in manifest");
    match cache::resolve(fr) {
        Some(p) => Some(p),
        None => {
            eprintln!(
                "skipped[wan22_e2e]: {}/{} not in HF cache",
                fr.repo, fr.path
            );
            None
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn wan22_e2e() {
    let _rollup = RollupDumpOnDrop(trace::init_from_env());

    // Tiny health grid (NOT the product visual-quality regime). Wan2.1 VAE:
    // width/height divisible by 16, num_frames = 4k+1.
    let width = env_u32("THINFER_E2E_WIDTH", 512) as usize;
    let height = env_u32("THINFER_E2E_HEIGHT", 320) as usize;
    let num_frames = env_u32("THINFER_E2E_FRAMES", 49);
    let vram_gb = env_u32("THINFER_E2E_VRAM_GB", 6) as u64;

    // Roles: the safetensors TAIL (umT5 3 shards + Wan2.1 VAE, in
    // `WAN22_TAIL_ROLES` order) + the two expert GGUFs + their LightX2V distill
    // LoRAs + the tokenizer. All resolved from the FastWan/Wan2.2 manifest.
    let (
        Some(te1),
        Some(te2),
        Some(te3),
        Some(vae_path),
        Some(hi_gguf_path),
        Some(lo_gguf_path),
        Some(hi_lora_path),
        Some(lo_lora_path),
        Some(tok_path),
    ) = (
        resolve_or_skip(role::TEXT_ENCODER_SHARD_1),
        resolve_or_skip(role::TEXT_ENCODER_SHARD_2),
        resolve_or_skip(role::TEXT_ENCODER_SHARD_3),
        resolve_or_skip(role::VAE_WAN21),
        resolve_or_skip(role::DIT_HIGH_NOISE),
        resolve_or_skip(role::DIT_LOW_NOISE),
        resolve_or_skip(role::LORA_HIGH_NOISE),
        resolve_or_skip(role::LORA_LOW_NOISE),
        resolve_or_skip(role::TOKENIZER_JSON),
    )
    else {
        return;
    };
    eprintln!(
        "wan22-e2e: {width}x{height} f{num_frames} steps={WAN22_DISTILL_STEPS} (roles resolved)"
    );

    // Safetensors tail openers, in `WAN22_TAIL_ROLES` order (umT5 shard1..3, then
    // the Wan2.1 VAE); passed as `tail_openers` to `open_wan22_source`.
    let mut tail_openers: Vec<MmapFileOpener> = Vec::with_capacity(4);
    for p in [&te1, &te2, &te3, &vae_path] {
        tail_openers.push(
            MmapFileOpener::new(p)
                .await
                .unwrap_or_else(|e| panic!("open {}: {e}", p.display())),
        );
    }
    let open = |p: &PathBuf| {
        let p = p.clone();
        async move {
            MmapFileOpener::new(&p)
                .await
                .unwrap_or_else(|e| panic!("open {}: {e}", p.display()))
        }
    };
    let hi_gguf = open(&hi_gguf_path).await;
    let lo_gguf = open(&lo_gguf_path).await;
    let hi_lora = open(&hi_lora_path).await;
    let lo_lora = open(&lo_lora_path).await;

    // Two folded experts (each -> Q8_0 blocks) unioned over the safetensors tail.
    // Arg order matches the executor's `open_wan22_source(hi_gguf, hi_lora,
    // lo_gguf, lo_lora, weight_openers, num_layers)`.
    let num_layers = WanDitConfig::wan22_14b().num_layers;
    let source = open_wan22_source(hi_gguf, hi_lora, lo_gguf, lo_lora, tail_openers, num_layers)
        .await
        .unwrap_or_else(|e| panic!("open_wan22_source: {e:?}"));

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
    let tokenizer = HfTokenizer::from_path(&tok_path).await.expect("tokenizer");

    // 28GB RAM: two ~13GB Q8_0 expert folds coexist in host RAM (per wan-plan);
    // AnyFlow used 20GB for its single expert. VRAM stays at the thin default.
    let residency = WeightResidency::new(
        source,
        ResidencyBudget {
            ram_bytes: 28 << 30,
            vram_bytes: vram_gb << 30,
        },
    );
    // Wan2.2-A14B has ONLY the full Wan2.1 VAE (no tiny decoder), so the executor
    // forces `VaeChoice::Full`; do the same. i8 matmul ON = the production default
    // (`req.i8_matmul`): DP4A-safe qkv_self + ffn_up sites take Q8_0.
    let vae_choice = VaeChoice::Full;
    let model = WanModel::load_variant(
        Arc::clone(&backend),
        residency,
        tokenizer,
        vae_choice,
        WanVariant::wan22_t2v_a14b(),
        None,
        true,
    )
    .await
    .expect("WanModel::load_variant(wan22)");

    // `THINFER_E2E_ATTN_WINDOW=W` opts into the temporal window (0/unset = full
    // attention). Wan2.2 ships all-steps windowing (its `default_from` is 0);
    // `THINFER_WAN_WINDOW_FROM_STEP` (read in the pipeline) drives the hybrid A/B.
    let attn_window = Some(env_u32("THINFER_E2E_ATTN_WINDOW", 0)).filter(|w| *w > 0);
    let prompt = std::env::var("THINFER_E2E_PROMPT").unwrap_or_else(|_| PROMPT.to_string());
    let params = GenerationParams {
        prompt,
        height: height as u32,
        width: width as u32,
        num_frames,
        seed: SEED,
        // Ignored on the MoE step-distill path (the schedule is fixed); default
        // suffices. Kept for the shared `GenerationParams` shape.
        sampler: VideoSampler::default(),
        attn_window,
        steps: None,
    };

    // --- denoise (fixed 4-step distill; per-step diag taps) ---
    let mut ws = Workspace::new(Arc::clone(&backend), Arc::clone(model.arbiter()));
    let mut step_diag: Vec<WanStepDiag> = Vec::new();
    let t0 = std::time::Instant::now();
    let (latent, f_lat, h_lat, w_lat) = model
        .denoise_with(&params, None, &mut ws, Some(&mut step_diag), None, None)
        .await
        .expect("denoise");
    eprintln!(
        "wan22 denoise: {} steps, latent std={:.4}, {:.1}s",
        step_diag.len(),
        std_of(&latent),
        t0.elapsed().as_secs_f32()
    );
    assert_eq!(step_diag.len(), WAN22_DISTILL_STEPS, "distill step count");
    assert!(latent.iter().all(|v| v.is_finite()), "latent finite");
    assert!(std_of(&latent) > 0.05, "latent degenerate");

    // --- VAE decode (full Wan2.1 VAE) + health ---
    let t0 = std::time::Instant::now();
    let video = model
        .decode_latent_to_video(&latent, f_lat, h_lat, w_lat, vae_choice, &mut ws)
        .await
        .expect("vae decode");
    eprintln!("wan22 vae decode: {:.1}s", t0.elapsed().as_secs_f32());
    let out_frames = TEMPORAL_SCALE * f_lat - 3;
    assert_eq!(video.len(), 3 * out_frames * height * width, "video size");
    assert!(video.iter().all(|v| v.is_finite()), "video finite");
    let vstd = std_of(&video);
    assert!(vstd > 0.05, "video degenerate (std {vstd})");
    // Temporal motion: last frame differs from first.
    let frame_len = height * width;
    let per_c = out_frames * frame_len;
    let mut diff = 0.0f64;
    for c in 0..3 {
        for p in 0..frame_len {
            let a = video[c * per_c + p];
            let b = video[c * per_c + (out_frames - 1) * frame_len + p];
            diff += ((a - b) as f64).abs();
        }
    }
    let mad = diff / (3 * frame_len) as f64;
    eprintln!("wan22 health: video std={vstd:.3} first-vs-last MAD={mad:.4}");
    assert!(mad > 1e-3, "no temporal motion (MAD {mad})");

    if let Some(dir) = std::env::var_os("THINFER_E2E_PNG_DIR").map(PathBuf::from) {
        std::fs::create_dir_all(&dir).expect("png dir");
        for f in 0..out_frames {
            let mut chw = vec![0.0f32; frame_len * 3];
            for p in 0..frame_len {
                for c in 0..3 {
                    chw[c * frame_len + p] = video[c * per_c + f * frame_len + p];
                }
            }
            let png =
                thinfer_models::z_image::pipeline::encode_png(&chw, width as u32, height as u32)
                    .expect("encode png");
            std::fs::write(dir.join(format!("wan22_{f:03}.png")), png).expect("write png");
        }
        eprintln!("staged {out_frames} frames");
    }
}
