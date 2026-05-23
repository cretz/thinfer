//! Forever end-to-end Z-Image parity test. Drives the engine through the
//! full path - tokenize -> Qwen3 encode -> N denoising steps -> VAE decode
//! -> RGB CHW fp32 - against a pinned PyTorch reference for the same
//! prompt, dims, seed, and initial noise. Both sides byte-load the same
//! noise so divergence is attributable to engine math (Qwen3 / DiT /
//! scheduler / VAE), never to RNG drift.
//!
//! Two variants share this file:
//!   - `e2e_parity_for_safetensors`: bf16 DiT shards (M1 baseline).
//!   - `e2e_parity_for_gguf_q8_0`: DiT matmuls unioned in from
//!     `unsloth/Z-Image-Turbo-GGUF` (Q8_0). Tolerances bumped to absorb
//!     Q8_0 quantization error vs the bf16 PyTorch reference.
//!
//! The only path difference is the source: the GGUF variant wraps the
//! same safetensors source in `UnionSource::new(RenamedSource(GgufSource),
//! sts)`. Tokenizer, TE, VAE, py reference, budget assertions, per-stage
//! VAE diag, PNG dump, and divergence-checking logic are all shared via
//! `run_pipeline_and_compare`.
//!
//! If everything matches, end-to-end byte parity is established. If a
//! stage diverges, the narrower bisection tests (qwen3_parity /
//! dit_parity) localize it.
//!
//! Run with:
//!   `cargo test -p thinfer-conformance --features zimage-e2e --release \
//!    e2e_parity_for -- --nocapture`
//! (the `e2e_parity_for` filter catches both variants).

#![cfg(feature = "zimage-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::Backend;
use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::format::safetensors::ShardedSafetensorsSource;
use thinfer_core::format::union::{
    QuantOnlySource, RenamedSource, SplitToFusedQkvSource, UnionSource,
};
use thinfer_core::ops::WeightDtype;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::quant::QuantKind;
use thinfer_core::residency::WeightResidency;
use thinfer_core::trace::{self, DIAG};
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::Workspace;
use thinfer_models::z_image::dit_qkv_triples;
use thinfer_models::z_image::manifest::{self, role};
use thinfer_models::z_image::pipeline::{GenerationParams, ZImageModel, encode_png};
use thinfer_models::z_image::vae::VaeStageSample;
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;
use thinfer_native::tokenizer::HfTokenizer;

/// Pinned config. Same prompt/noise/seed across both variants so the same
/// PyTorch reference applies. Small enough to keep the test under a few
/// minutes; large enough to exercise the full engine including VAE at
/// production dims.
const PROMPT: &str = "a red apple on a wooden table";
const HEIGHT: u32 = 256;
const WIDTH: u32 = 256;
const STEPS: u32 = 2;
const SEED: u64 = 42;

const LATENT_CHANNELS: usize = 16;
const VAE_SCALE: usize = 8;
/// Matches `thinfer_models::z_image::vae::config::{SCALING,SHIFT}_FACTOR`.
/// Used to scale our pre-VAE latent dump into the same space py captures
/// (py's `vae.decode` hook fires AFTER diffusers does `z/SCALING + SHIFT`;
/// our `vae.decode` does that same transform internally, so our dump is
/// pre-scaling). Comparing the two without this shift always shows a
/// spurious 1/SCALING_FACTOR ratio drift.
const VAE_SCALING_FACTOR: f32 = 0.3611;
const VAE_SHIFT_FACTOR: f32 = 0.1159;

/// Per-tap tolerance for the divergence checker. The cap on cells over
/// tol is the only per-variant slack we offer; the per-tap tolerance
/// stays shared so a Q8_0 cell within Q8_0's expected error window still
/// has to be within 3% of |expected| (i.e. one sloppy fused op doesn't
/// get a free pass; quantization noise has to express as a bounded
/// *count* of over-tol cells, not as one giant outlier).
const TOL_MULT: f32 = 0.03;
const TOL_FLOOR: f32 = 1e-3;

/// Per-stage cap on cells over per-tap tol, per variant.
#[derive(Clone, Copy)]
struct Tolerances {
    /// `step{i}.prev_sample`: post-scheduler-step latent, one per step.
    step_prev_sample: usize,
    /// `pre_vae_latent`: last step's prev_sample (post `z/SCALING + SHIFT`).
    pre_vae: usize,
    /// `vae_rgb`: VAE-decoded CHW fp32 RGB.
    vae_rgb: usize,
    /// `vae.<stage_label>`: per-VAE-stage head sample.
    vae_diag: usize,
}

// Variant names mirror the upstream GGUF filename suffixes
// (`z-image-turbo-Q8_0.gguf`, `Q4_K_M`, etc.) so the slug-to-file
// mapping stays obvious. Underscores are load-bearing for the next
// variant (`GgufQ4_K_M`).
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
enum Variant {
    Safetensors,
    GgufQ8_0,
    GgufQ4_K_M,
}

impl Variant {
    fn slug(self) -> &'static str {
        match self {
            Variant::Safetensors => "safetensors",
            Variant::GgufQ8_0 => "gguf_q8_0",
            Variant::GgufQ4_K_M => "gguf_q4_k_m",
        }
    }

    /// Optional extra role to resolve from the HF cache. `None` = pure
    /// safetensors path; `Some(role)` = union GGUF over safetensors.
    fn gguf_role(self) -> Option<&'static str> {
        match self {
            Variant::Safetensors => None,
            Variant::GgufQ8_0 => Some(role::DIT_GGUF_Q8_0),
            Variant::GgufQ4_K_M => Some(role::DIT_GGUF_Q4_K_M),
        }
    }

    /// Dtype the DiT main matmul kernels must compile against for this
    /// variant. Asserted right after `ZImageModel::load` so a misnamed
    /// rename map (GGUF source returns the file but its catalog keys
    /// never match the safetensors names) can't silently fall through
    /// to the bf16 safetensors side of the union and produce a passing
    /// run that's actually identical to the safetensors variant.
    fn expected_dit_matmul_weight(self) -> WeightDtype {
        match self {
            Variant::Safetensors => WeightDtype::Bf16,
            Variant::GgufQ8_0 => WeightDtype::Quant(QuantKind::Q8_0),
            // Q4_K_M is per-(layer, slot) mixed Q4_K + Q5_K + Q6_K.
            // The probe tensor (`layers.0.attention.qkv.weight`) is Q6_K
            // per llama.cpp's "2 special layers" convention (first+last
            // layer's qkv promoted from Q5_K to Q6_K).
            Variant::GgufQ4_K_M => WeightDtype::Quant(QuantKind::Q6_K),
        }
    }

    fn tolerances(self) -> Tolerances {
        // Safetensors values are the historical caps that established
        // bf16-clean parity (iGPU ~600, discrete ~4600). GGUF starts
        // generous and tightens after a first run lands real numbers -
        // see `[Tighten e2e parity tol]` in the worklog backlog. The DiT
        // stages (step.prev_sample, pre_vae) are expected to absorb
        // most of Q8_0's per-matmul quant error; budget is sized for
        // "broken-vs-noisy", not "tight":
        //   - n_lat = 16 * (256/8)^2 = 16384 cells.
        //   - vae_rgb = 3 * 256^2 = 196608 cells.
        // Bumping vae_rgb to ~50% leaves room for Q8_0 drift
        // accumulating through VAE up_blocks.
        match self {
            Variant::Safetensors => Tolerances {
                step_prev_sample: 0,
                pre_vae: 0,
                vae_rgb: 6000,
                vae_diag: 96,
            },
            Variant::GgufQ8_0 => Tolerances {
                step_prev_sample: 8192,
                pre_vae: 8192,
                vae_rgb: 98_304,
                vae_diag: 256,
            },
            // Q4_K_M is 4-bit but uses 8 sub-block scales per super-block
            // (vs Q4_0's one) so per-element error stays much closer to Q8_0
            // than to Q4_0. First-run baseline; tighten after a clean run.
            Variant::GgufQ4_K_M => Tolerances {
                step_prev_sample: 12_288,
                pre_vae: 12_288,
                vae_rgb: 147_456,
                vae_diag: 512,
            },
        }
    }
}

/// Dumps `RollupHandle` to stderr on test exit (success or panic).
/// Without this the rollup table is lost when divergence triggers
/// `panic!` before any tail logging runs.
struct RollupDumpOnDrop(Option<thinfer_core::trace::RollupHandle>);
impl Drop for RollupDumpOnDrop {
    fn drop(&mut self) {
        if let Some(h) = self.0.take() {
            let _ = h.dump(&mut std::io::stderr());
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn e2e_parity_for_safetensors() {
    run(Variant::Safetensors).await;
}

#[tokio::test(flavor = "current_thread")]
async fn e2e_parity_for_gguf_q8_0() {
    run(Variant::GgufQ8_0).await;
}

#[tokio::test(flavor = "current_thread")]
async fn e2e_parity_for_gguf_q4_k_m() {
    run(Variant::GgufQ4_K_M).await;
}

async fn run(variant: Variant) {
    let _rollup = RollupDumpOnDrop(trace::init_from_env());
    eprintln!("e2e-parity[{}]: starting", variant.slug());
    let h_lat = (HEIGHT as usize) / VAE_SCALE;
    let w_lat = (WIDTH as usize) / VAE_SCALE;
    let n_lat = LATENT_CHANNELS * h_lat * w_lat;
    let img_h = HEIGHT as usize;
    let img_w = WIDTH as usize;
    let rgb_elems = 3 * img_h * img_w;

    // Resolve every weight role the engine needs (variant-aware). Skip
    // cleanly if any role misses cache - matches dit_parity's discipline
    // so the test never spuriously fails on a machine without the HF
    // cache populated.
    let mut needed: Vec<&str> = vec![
        role::DIT_SHARD_1,
        role::DIT_SHARD_2,
        role::TEXT_ENCODER_SHARD_1,
        role::TEXT_ENCODER_SHARD_2,
        role::TEXT_ENCODER_SHARD_3,
        role::VAE,
        role::TOKENIZER_JSON,
    ];
    if let Some(r) = variant.gguf_role() {
        needed.push(r);
    }
    let mut resolved: Vec<(&str, PathBuf)> = Vec::with_capacity(needed.len());
    for r in needed {
        let fr = manifest::MANIFEST.get(r).expect("role in manifest");
        match cache::resolve(fr) {
            Some(p) => resolved.push((r, p)),
            None => {
                eprintln!(
                    "skipped[{}]: {}/{} not in HF cache ({})",
                    variant.slug(),
                    fr.repo,
                    fr.path,
                    cache::cache_root().display()
                );
                return;
            }
        }
    }
    let path_of =
        |role_name: &str| -> &Path { &resolved.iter().find(|(r, _)| *r == role_name).unwrap().1 };
    eprintln!(
        "e2e-parity[{}]: all roles resolved from HF cache",
        variant.slug()
    );

    // Variant-scoped tmpdir under CARGO_TARGET_TMPDIR so the two
    // variants don't clobber py dumps if they run in parallel
    // (`cargo test -- --test-threads=N>1`).
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(variant.slug());
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let noise_path = tmp.join("e2e_parity_noise.bin");

    let py_starting_path = tmp.join("py_starting_latents.bin");
    let py_pre_vae_path = tmp.join("py_pre_vae_latent.bin");
    let py_vae_rgb_path = tmp.join("py_vae_rgb.bin");
    let py_step_paths: Vec<PathBuf> = (0..STEPS as usize)
        .map(|i| tmp.join(format!("py_step{i}_post.bin")))
        .collect();

    // Clear stale dumps so a stale file can't mask a hook that never fires.
    let mut clear_paths = vec![
        py_starting_path.clone(),
        py_pre_vae_path.clone(),
        py_vae_rgb_path.clone(),
    ];
    clear_paths.extend(py_step_paths.iter().cloned());
    for p in &clear_paths {
        let _ = std::fs::remove_file(p);
    }

    // Deterministic pinned noise. Identical to dit_parity's RNG
    // (Box-Muller over SplitMix64) so both tests exercise the same
    // starting tensor.
    let noise = make_pinned_noise(n_lat);
    summarize("noise (pinned)", &noise);
    std::fs::write(&noise_path, bytemuck_cast(&noise)).expect("write noise");

    // Opt-in PNG dumps. Stamped with variant slug so both variants can
    // share the same THINFER_E2E_PNG_DIR without clobbering.
    let png_dir = std::env::var_os("THINFER_E2E_PNG_DIR").map(PathBuf::from);
    let png_filename_ours = format!("ours_{}.png", variant.slug());
    let png_filename_py = format!("py_{}.png", variant.slug());
    if let Some(d) = png_dir.as_ref() {
        std::fs::create_dir_all(d).expect("create THINFER_E2E_PNG_DIR");
        eprintln!(
            "png dump enabled[{}]: {} ({}, {})",
            variant.slug(),
            d.display(),
            png_filename_ours,
            png_filename_py
        );
    }

    let vae_diag_dir = tmp.join("vae_diag");
    std::fs::create_dir_all(&vae_diag_dir).expect("create vae_diag dir");
    if let Ok(rd) = std::fs::read_dir(&vae_diag_dir) {
        for ent in rd.flatten() {
            let _ = std::fs::remove_file(ent.path());
        }
    }
    eprintln!("vae diag[{}]: {}", variant.slug(), vae_diag_dir.display());

    // Drive the PyTorch reference. bf16 regardless of engine variant -
    // GGUF parity is "engine Q8_0 vs pytorch bf16", with looser
    // tolerances baked into `Variant::tolerances()`.
    let dit_shards = [
        path_of(role::DIT_SHARD_1).to_owned(),
        path_of(role::DIT_SHARD_2).to_owned(),
    ];
    run_python_ref(
        &noise_path,
        &dit_shards,
        &tmp,
        png_dir.as_deref(),
        &png_filename_py,
        &vae_diag_dir,
    );

    // Build the safetensors base source (always needed; GGUF variant
    // unions over the top).
    let weight_roles = [
        role::DIT_SHARD_1,
        role::DIT_SHARD_2,
        role::TEXT_ENCODER_SHARD_1,
        role::TEXT_ENCODER_SHARD_2,
        role::TEXT_ENCODER_SHARD_3,
        role::VAE,
    ];
    let mut openers: Vec<MmapFileOpener> = Vec::with_capacity(weight_roles.len());
    for r in weight_roles {
        let path = path_of(r);
        openers.push(
            MmapFileOpener::new(path)
                .await
                .unwrap_or_else(|e| panic!("open {}: {e}", path.display())),
        );
    }
    let base_source = ShardedSafetensorsSource::open(openers)
        .await
        .expect("parse sharded safetensors");
    // Engine consumes canonical fused QKV; split safetensors checkpoints
    // (dimitribarbot) flow through this adapter, fused-fused checkpoints
    // see it as a passthrough.
    let base_source = SplitToFusedQkvSource::new(base_source, dit_qkv_triples());
    let base_source =
        RenamedSource::with_passthrough(base_source, thinfer_models::z_image::dit_to_out_renames());

    // Aggressive 2 GiB / 2 GiB budgets - intentionally well under model
    // size to exercise the eviction + rolling-residency paths. Anything
    // that assumes "the whole DiT fits in VRAM" must be re-thought.
    let budget = ResidencyBudget {
        ram_bytes: 2 << 30,
        vram_bytes: 2 << 30,
        workspace_reserve: 1 << 30,
    };

    // Default `HighPerformance` to mirror CLI. Unset Vulkan power-pref hints
    // are interpreted as background-priority by the driver on Intel iGPUs
    // (~2.5x slower DiT, narrower subgroup_size range). Tests that need to
    // exercise the LowPower path set `THINFER_POWER_PREF=low`.
    let cfg = WgpuConfig {
        power_preference: match std::env::var("THINFER_POWER_PREF")
            .ok()
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("high" | "highperformance" | "discrete") => PowerPreference::HighPerformance,
            Some("low" | "lowpower" | "integrated") => PowerPreference::LowPower,
            Some("none") => PowerPreference::None,
            _ => PowerPreference::HighPerformance,
        },
        timestamps: std::env::var("THINFER_TRACE").is_ok(),
    };
    let backend = Arc::new(
        WgpuBackend::new_with_config(cfg)
            .await
            .expect("wgpu adapter unavailable for tests"),
    );
    let tokenizer = HfTokenizer::from_path(path_of(role::TOKENIZER_JSON))
        .await
        .expect("tokenizer load");

    let ctx = CompareCtx {
        variant,
        noise,
        n_lat,
        rgb_elems,
        h_lat,
        w_lat,
        png_dir,
        png_filename_ours,
        vae_diag_dir,
        py_starting_path,
        py_pre_vae_path,
        py_vae_rgb_path,
        py_step_paths,
        budget,
    };

    // Dispatch by variant. Source type changes between arms so this is
    // where the union happens; everything below
    // `run_pipeline_and_compare` is source-agnostic.
    match variant.gguf_role() {
        None => {
            let residency = WeightResidency::new(base_source, budget);
            run_pipeline_and_compare(backend, residency, tokenizer, ctx).await;
        }
        Some(gguf_role) => {
            let gguf_path = path_of(gguf_role);
            eprintln!(
                "e2e-parity[{}]: union GGUF over safetensors: {}",
                variant.slug(),
                gguf_path.display()
            );
            let gguf_opener = MmapFileOpener::new(gguf_path)
                .await
                .unwrap_or_else(|e| panic!("open gguf {}: {e}", gguf_path.display()));
            let gguf = GgufSource::open(gguf_opener)
                .await
                .unwrap_or_else(|e| panic!("parse gguf {}: {e:?}", gguf_path.display()));
            // GGUF (unsloth Z-Image-Turbo) already ships canonical upstream
            // names including fused `attention.qkv.weight`; no rename. The
            // safetensors fallback feeds AdaLN/biases/norms under matching
            // names (and supplies the fused QKV via `SplitToFusedQkvSource`
            // for any quant-eligible tensor the GGUF doesn't carry).
            //
            // `QuantOnlySource` hides GGUF's F32 norms/AdaLN from the union
            // so the bf16 safetensors entries (same canonical names) aren't
            // shadowed by F32 ones the engine residency can't decode.

            // unsloth Z-Image-Turbo-GGUF quantizes more tensors than the
            // engine treats as quantized (e.g. main-layer AdaLN, refiners,
            // possibly biases). Engine quantizes only the five main-DiT
            // matmul weights per `layers.<i>.` block. Allowlist those.
            let unioned = UnionSource::new(
                QuantOnlySource::with_allowed_substrings(
                    gguf,
                    &[
                        ".attention.qkv.weight",
                        ".attention.out.weight",
                        ".feed_forward.w1.weight",
                        ".feed_forward.w2.weight",
                        ".feed_forward.w3.weight",
                    ],
                ),
                base_source,
            );
            let residency = WeightResidency::new(unioned, budget);
            run_pipeline_and_compare(backend, residency, tokenizer, ctx).await;
        }
    }
}

/// Source-agnostic context passed to `run_pipeline_and_compare`. Owns
/// every input the post-load body needs so the function can be generic
/// over the `WeightSource` impl.
struct CompareCtx {
    variant: Variant,
    noise: Vec<f32>,
    n_lat: usize,
    rgb_elems: usize,
    h_lat: usize,
    w_lat: usize,
    png_dir: Option<PathBuf>,
    png_filename_ours: String,
    vae_diag_dir: PathBuf,
    py_starting_path: PathBuf,
    py_pre_vae_path: PathBuf,
    py_vae_rgb_path: PathBuf,
    py_step_paths: Vec<PathBuf>,
    budget: ResidencyBudget,
}

async fn run_pipeline_and_compare<S: WeightSource>(
    backend: Arc<WgpuBackend>,
    residency: WeightResidency<S>,
    tokenizer: HfTokenizer,
    ctx: CompareCtx,
) {
    let model = ZImageModel::load(Arc::clone(&backend), residency, tokenizer)
        .await
        .expect("ZImageModel::load");

    let actual_dtype = model.dit_matmul_weight();
    let expected_dtype = ctx.variant.expected_dit_matmul_weight();
    eprintln!(
        "[{}] DiT matmul weight dtype: {:?} (expected {:?})",
        ctx.variant.slug(),
        actual_dtype,
        expected_dtype
    );
    assert_eq!(
        actual_dtype,
        expected_dtype,
        "variant {} expected DiT matmul dtype {:?} but loader picked {:?}; \
         GGUF source likely missing `attention.qkv.weight`, falling through \
         to the safetensors fallback in the UnionSource. Cross-check the \
         GGUF tensor catalog against the canonical upstream Z-Image schema.",
        ctx.variant.slug(),
        expected_dtype,
        actual_dtype,
    );

    let t_full = std::time::Instant::now();
    tracing::info!(target: DIAG, t_ms = 0_u64, "milestone: starting denoise_with");
    let params = GenerationParams {
        prompt: PROMPT.to_string(),
        height: HEIGHT,
        width: WIDTH,
        steps: STEPS,
        seed: SEED,
    };
    let mut ws = Workspace::new(Arc::clone(&backend));
    let mut our_step_dumps: Vec<Vec<f32>> = Vec::with_capacity(STEPS as usize);
    let (our_pre_vae, our_h_lat, our_w_lat) = model
        .denoise_with(
            &params,
            Some(&ctx.noise),
            &mut ws,
            Some(&mut our_step_dumps),
        )
        .await
        .expect("denoise_with");
    tracing::info!(
        target: DIAG,
        t_ms = t_full.elapsed().as_millis() as u64,
        pre_vae_len = our_pre_vae.len(),
        h_lat = our_h_lat,
        w_lat = our_w_lat,
        step_dumps = our_step_dumps.len(),
        "milestone: denoise_with done",
    );
    assert_eq!(our_h_lat, ctx.h_lat);
    assert_eq!(our_w_lat, ctx.w_lat);
    assert_eq!(our_step_dumps.len(), STEPS as usize);

    // VAE decode -> CHW fp32 RGB in [-1, 1], capturing per-stage head
    // samples for compare against py's per-submodule hooks.
    let mut our_vae_diag: Vec<VaeStageSample> = Vec::new();
    let our_rgb = model
        .decode_latents_to_rgb_with_diag(
            &our_pre_vae,
            ctx.h_lat,
            ctx.w_lat,
            &mut ws,
            &mut our_vae_diag,
        )
        .await
        .expect("decode_latents_to_rgb_with_diag");
    for st in &our_vae_diag {
        let p = ctx.vae_diag_dir.join(format!("our_{}.bin", st.label));
        std::fs::write(&p, bytemuck_cast(&st.head)).expect("write our vae diag stage");
    }
    eprintln!(
        "[vae-diag] captured {} stages from our VAE; dumped to {}",
        our_vae_diag.len(),
        ctx.vae_diag_dir.display()
    );
    tracing::info!(
        target: DIAG,
        t_ms = t_full.elapsed().as_millis() as u64,
        rgb_len = our_rgb.len(),
        rgb_expected = ctx.rgb_elems,
        "milestone: vae decode done",
    );
    assert_eq!(our_rgb.len(), ctx.rgb_elems);

    // Write PNG before the budget assertion so a budget failure doesn't
    // suppress the visual output we want to inspect.
    if let Some(d) = ctx.png_dir.as_ref() {
        match encode_png(&our_rgb, WIDTH, HEIGHT) {
            Ok(png) => {
                let p = d.join(&ctx.png_filename_ours);
                std::fs::write(&p, &png).expect("write ours.png");
                eprintln!("wrote {}", p.display());
            }
            Err(e) => eprintln!("encode_png failed: {e}"),
        }
    }

    // Budget snapshot: emit unconditionally so a parity-failing run also
    // shows where we landed on memory. Actual ceiling assertions move to
    // the end of the test so parity numbers always print on failure.
    let snap = backend.mem_account().snapshot();
    eprintln!(
        "[mem] vram TRUE_PEAK={} / budget {} | per-cat peaks: weights={} workspace={} staging={} (sum {})",
        fmt_mib(snap.vram_total_peak),
        fmt_mib(ctx.budget.vram_bytes),
        fmt_mib(snap.vram_weights.1),
        fmt_mib(snap.vram_workspace.1),
        fmt_mib(snap.vram_staging.1),
        fmt_mib(snap.vram_per_cat_peak_sum()),
    );
    eprintln!(
        "[mem] ram  TRUE_PEAK={} / budget {} | per-cat peaks: upload={} readback={} other={} (sum {})",
        fmt_mib(snap.ram_total_peak),
        fmt_mib(ctx.budget.ram_bytes),
        fmt_mib(snap.ram_upload.1),
        fmt_mib(snap.ram_readback.1),
        fmt_mib(snap.ram_other.1),
        fmt_mib(snap.ram_per_cat_peak_sum()),
    );

    let py_starting = read_f32(&ctx.py_starting_path);
    let py_pre_vae = read_f32(&ctx.py_pre_vae_path);
    let py_vae_rgb = read_f32(&ctx.py_vae_rgb_path);
    let py_steps: Vec<Vec<f32>> = ctx.py_step_paths.iter().map(|p| read_f32(p)).collect();

    summarize("py_starting_latents", &py_starting);
    for (i, s) in py_steps.iter().enumerate() {
        summarize(&format!("py_step{i}_prev_sample"), s);
    }
    summarize("py_pre_vae_latent", &py_pre_vae);
    summarize("py_vae_rgb", &py_vae_rgb);

    for (i, s) in our_step_dumps.iter().enumerate() {
        summarize(&format!("our_step{i}_prev_sample"), s);
    }
    summarize("our_pre_vae_latent", &our_pre_vae);
    summarize("our_vae_rgb", &our_rgb);

    let tols = ctx.variant.tolerances();
    let mut diverged: Option<String> = None;
    let mut check = |label: &str, got: &[f32], expected: &[f32], max_n_over: usize| {
        let n = got.len().min(expected.len());
        let max_ref = expected[..n]
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0f32, f32::max);
        let tol = (max_ref * TOL_MULT).max(TOL_FLOOR);
        let (max_abs, n_over) = diff_stats(&got[..n], &expected[..n], tol);
        eprintln!(
            "[{label}] max_abs={max_abs:.4e} tol={tol:.4e} above_tol={n_over}/{n} \
             ref_max_abs={max_ref:.4e}"
        );
        if n_over > max_n_over && diverged.is_none() {
            diverged = Some(format!(
                "{label}: {n_over}/{n} cells over tol={tol:.4e} \
                 (max_n_over={max_n_over}, max_abs={max_abs:.4e}, ref_max_abs={max_ref:.4e})"
            ));
        }
    };

    assert_eq!(py_starting.len(), ctx.n_lat, "py_starting_latents length");
    assert_eq!(py_pre_vae.len(), ctx.n_lat, "py_pre_vae_latent length");
    assert_eq!(py_vae_rgb.len(), ctx.rgb_elems, "py_vae_rgb length");
    for (i, s) in py_steps.iter().enumerate() {
        assert_eq!(s.len(), ctx.n_lat, "py_step{i}_prev_sample length");
    }

    // starting_latents is the injected noise both sides byte-load - it
    // has to match exactly regardless of variant.
    check(
        "starting_latents (== injected noise)",
        &ctx.noise,
        &py_starting,
        0,
    );
    for i in 0..STEPS as usize {
        check(
            &format!("step{i}.prev_sample"),
            &our_step_dumps[i],
            &py_steps[i],
            tols.step_prev_sample,
        );
    }
    // Apply diffusers' `z/SCALING + SHIFT` transform so our pre-VAE dump
    // is in the same space as py's vae.decode-hook capture. Without
    // this, both sides differ by a constant 1/SCALING ~2.77x ratio -
    // a wiring artifact, not a bug.
    let our_pre_vae_scaled: Vec<f32> = our_pre_vae
        .iter()
        .map(|z| z / VAE_SCALING_FACTOR + VAE_SHIFT_FACTOR)
        .collect();
    summarize(
        "our_pre_vae_latent (post-scale, vae.decode-input space)",
        &our_pre_vae_scaled,
    );
    check(
        "pre_vae_latent (post-scale, ==what VAE math sees)",
        &our_pre_vae_scaled,
        &py_pre_vae,
        tols.pre_vae,
    );
    check(
        "vae_rgb (CHW fp32 [-1, 1])",
        &our_rgb,
        &py_vae_rgb,
        tols.vae_rgb,
    );

    // VAE per-stage diag compare. Stage names ours produces but py
    // doesn't hook (e.g. `up{i}.upsample`, `silu_out`) are
    // summarized-only.
    eprintln!("---- vae per-stage diag ----");
    for st in &our_vae_diag {
        summarize(&format!("our_vae.{}", st.label), &st.head);
    }
    for st in &our_vae_diag {
        let py_path = ctx.vae_diag_dir.join(format!("py_{}.bin", st.label));
        if !py_path.exists() {
            eprintln!(
                "[vae.{}] (py side did not dump - ours-only stage, see summary above)",
                st.label
            );
            continue;
        }
        let py_head = read_f32(&py_path);
        summarize(&format!("py_vae.{}", st.label), &py_head);
        check(
            &format!("vae.{}", st.label),
            &st.head,
            &py_head,
            tols.vae_diag,
        );
    }

    // Now the budget ceilings. Run after parity so the parity diff
    // numbers above are always in the log, regardless of which check
    // tripped first.
    let vram_over = snap.vram_total_peak > ctx.budget.vram_bytes;
    let ram_over = snap.ram_total_peak > ctx.budget.ram_bytes;

    if let Some(msg) = diverged {
        panic!("FIRST DIVERGENCE: {msg}");
    }
    assert!(
        !vram_over,
        "vram true peak {} > budget {} (per-cat peaks: weights {}, workspace {}, staging {})",
        fmt_mib(snap.vram_total_peak),
        fmt_mib(ctx.budget.vram_bytes),
        fmt_mib(snap.vram_weights.1),
        fmt_mib(snap.vram_workspace.1),
        fmt_mib(snap.vram_staging.1),
    );
    assert!(
        !ram_over,
        "ram true peak {} > budget {} (per-cat peaks: upload {}, readback {}, other {})",
        fmt_mib(snap.ram_total_peak),
        fmt_mib(ctx.budget.ram_bytes),
        fmt_mib(snap.ram_upload.1),
        fmt_mib(snap.ram_readback.1),
        fmt_mib(snap.ram_other.1),
    );
}

fn run_python_ref(
    noise_path: &Path,
    dit_shards: &[PathBuf],
    out_dir: &Path,
    png_dir: Option<&Path>,
    png_filename: &str,
    vae_diag_dir: &Path,
) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let mut cmd = Command::new("uv");
    cmd.args([
        "run",
        "--directory",
        py_dir.to_str().unwrap(),
        "python",
        "-m",
        "thinfer_pytorch_ref.zimage.gen_e2e_parity_ref",
        "--initial-noise",
        noise_path.to_str().unwrap(),
        "--out",
        out_dir.to_str().unwrap(),
        "--prompt",
        PROMPT,
        "--height",
        &HEIGHT.to_string(),
        "--width",
        &WIDTH.to_string(),
        "--steps",
        &STEPS.to_string(),
        "--seed",
        &SEED.to_string(),
        "--dtype",
        "bf16",
    ]);
    for shard in dit_shards {
        cmd.args(["--transformer-shard", shard.to_str().unwrap()]);
    }
    if let Some(d) = png_dir {
        cmd.args(["--png-dir", d.to_str().unwrap()]);
        cmd.args(["--png-filename", png_filename]);
    }
    cmd.args(["--vae-diag-dir", vae_diag_dir.to_str().unwrap()]);
    let status = cmd
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "pytorch e2e-parity ref failed");
}

/// SplitMix64 + Box-Muller. Identical to dit_parity::make_pinned_noise
/// so both tests load the same starting tensor.
fn make_pinned_noise(n: usize) -> Vec<f32> {
    let mut state: u64 = 0xFEED_F00D_C0DE_BEEFu64;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let uniform = |x: u64| -> f64 { ((x >> 11) as f64 / (1u64 << 53) as f64).max(1e-12) };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = uniform(next());
        let u2 = uniform(next());
        let mag = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push((mag * theta.cos()) as f32);
        if out.len() < n {
            out.push((mag * theta.sin()) as f32);
        }
    }
    out
}

fn bytemuck_cast(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn fmt_mib(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.2}GiB", bytes as f64 / (1u64 << 30) as f64)
    } else {
        format!("{:.1}MiB", bytes as f64 / (1u64 << 20) as f64)
    }
}

fn summarize(label: &str, v: &[f32]) {
    let (mut max_abs, mut min, mut max, mut sum, mut nan) =
        (0f32, f32::INFINITY, f32::NEG_INFINITY, 0.0f64, 0usize);
    for &x in v {
        if x.is_nan() {
            nan += 1;
            continue;
        }
        let a = x.abs();
        if a > max_abs {
            max_abs = a;
        }
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
        sum += x as f64;
    }
    let denom = (v.len() - nan).max(1) as f64;
    let mean = sum / denom;
    eprintln!(
        "[{label}] len={} nan={} min={:.4e} max={:.4e} max_abs={:.4e} mean={:.4e}",
        v.len(),
        nan,
        min,
        max,
        max_abs,
        mean
    );
}

fn read_f32(p: &Path) -> Vec<f32> {
    let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn diff_stats(got: &[f32], expected: &[f32], tol: f32) -> (f32, usize) {
    let mut max_abs = 0f32;
    let mut over = 0usize;
    for (g, e) in got.iter().zip(expected.iter()) {
        // NaN / inf in either side counts as "over tol" — silently passing
        // an all-NaN run is the bug behind `[e2e_parity NaN-loose
        // assertion]` in the worklog backlog. NaN.abs() = NaN, and any
        // comparison with NaN is false, so the naive `d > tol` misses it.
        let nan_or_inf = !g.is_finite() || !e.is_finite();
        let d = (g - e).abs();
        if d.is_finite() && d > max_abs {
            max_abs = d;
        }
        if nan_or_inf || d > tol {
            over += 1;
        }
    }
    (max_abs, over)
}
