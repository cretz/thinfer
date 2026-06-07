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
//! The only path difference is the source: the GGUF variant passes a GGUF
//! opener to `ZImageSource::open`, which unions the DiT matmuls over the
//! safetensors side. Tokenizer, TE, VAE, py reference, budget assertions, per-stage
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
use thinfer_core::ops::WeightDtype;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::quant::QuantKind;
use thinfer_core::residency::WeightResidency;
use thinfer_core::trace::{self, DIAG};
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::Workspace;
use thinfer_models::z_image::manifest::{self, RecipeOverrideGuard, ZImageRecipe, role};
use thinfer_models::z_image::pipeline::{
    Block0LocalTaps, GenerationParams, Step0LocalizationTaps, ZImageModel, encode_png,
};
use thinfer_models::z_image::source::{GgufOpeners, ZImageSource};
use thinfer_models::z_image::vae::VaeStageSample;
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;
use thinfer_native::tokenizer::HfTokenizer;

/// Pinned config. Same prompt/noise/seed across both variants so the same
/// PyTorch reference applies. Small enough to keep the test under a few
/// minutes; large enough to exercise the full engine including VAE at
/// production dims.
const PROMPT: &str = "a red apple on a wooden table";
const DEFAULT_HEIGHT: u32 = 256;
const DEFAULT_WIDTH: u32 = 256;
const STEPS: u32 = 2;
const SEED: u64 = 42;

/// `THINFER_E2E_DIMS=WxH` (e.g. `512x512`) overrides the default
/// 256x256. Custom dims require `THINFER_E2E_SKIP_PYREF` because the
/// per-variant tolerance caps and 2 GiB budget are tuned for 256x256.
fn resolve_dims() -> (u32, u32) {
    match std::env::var("THINFER_E2E_DIMS") {
        Ok(s) => {
            let (w, h) = s
                .split_once('x')
                .unwrap_or_else(|| panic!("THINFER_E2E_DIMS must be WxH, got {s:?}"));
            let w: u32 = w.parse().unwrap_or_else(|_| panic!("bad width in {s:?}"));
            let h: u32 = h.parse().unwrap_or_else(|_| panic!("bad height in {s:?}"));
            (w, h)
        }
        Err(_) => (DEFAULT_WIDTH, DEFAULT_HEIGHT),
    }
}

/// `THINFER_E2E_SKIP_PYREF=1` skips the pytorch reference invocation
/// and py-vs-engine divergence checks. VRAM/RAM ceiling assertions
/// still run: the workspace peak must not depend on whether py-ref
/// was executed.
fn skip_pyref() -> bool {
    std::env::var("THINFER_E2E_SKIP_PYREF").is_ok()
}

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
    GgufQ8_0I8Sdpa,
    GgufQ4_K_M,
}

impl Variant {
    fn slug(self) -> &'static str {
        match self {
            Variant::Safetensors => "safetensors",
            Variant::GgufQ8_0 => "gguf_q8_0",
            Variant::GgufQ8_0I8Sdpa => "gguf_q8_0_i8_sdpa",
            Variant::GgufQ4_K_M => "gguf_q4_k_m",
        }
    }

    /// Optional extra role to resolve from the HF cache. `None` = pure
    /// safetensors path; `Some(role)` = union GGUF over safetensors.
    fn gguf_role(self) -> Option<&'static str> {
        match self {
            Variant::Safetensors => None,
            Variant::GgufQ8_0 | Variant::GgufQ8_0I8Sdpa => Some(role::DIT_GGUF_Q8_0),
            Variant::GgufQ4_K_M => Some(role::DIT_GGUF_Q4_K_M),
        }
    }

    /// Text-encoder GGUF role, paired with `gguf_role` (every quant
    /// variant ships the Q8_0 TE; see `manifest::VARIANTS`).
    fn te_gguf_role(self) -> Option<&'static str> {
        self.gguf_role().map(|_| role::TE_GGUF_Q8_0)
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
            Variant::GgufQ8_0 | Variant::GgufQ8_0I8Sdpa => WeightDtype::Quant(QuantKind::Q8_0),
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
            // Same DiT weights + activation path as Q8_0 plus the opt-in i8
            // attention (q/k/v quantized once post-rope, sdpa_i8, paired
            // output into proj). Budget sits between Q8_0 and a broken run;
            // tighten after a clean run lands real numbers.
            Variant::GgufQ8_0I8Sdpa => Tolerances {
                step_prev_sample: 12_288,
                pre_vae: 12_288,
                vae_rgb: 147_456,
                vae_diag: 512,
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
async fn e2e_parity_for_gguf_q8_0_i8_sdpa() {
    run(Variant::GgufQ8_0I8Sdpa).await;
}

#[tokio::test(flavor = "current_thread")]
async fn e2e_parity_for_gguf_q4_k_m() {
    run(Variant::GgufQ4_K_M).await;
}

async fn run(variant: Variant) {
    let _rollup = RollupDumpOnDrop(trace::init_from_env());
    // Test-only recipe override: flip `i8_sdpa = true` for the i8-attention
    // variant. Guard drops at end of `run`, restoring the production recipe.
    let _recipe = match variant {
        Variant::GgufQ8_0I8Sdpa => Some(RecipeOverrideGuard::install(ZImageRecipe {
            bf16_quant_writes: manifest::RECIPE.bf16_quant_writes,
            i8_sdpa: true,
        })),
        _ => None,
    };
    let (width, height) = resolve_dims();
    let skip_pyref = skip_pyref();
    if (width != DEFAULT_WIDTH || height != DEFAULT_HEIGHT) && !skip_pyref {
        panic!(
            "THINFER_E2E_DIMS={width}x{height} requires THINFER_E2E_SKIP_PYREF=1 \
             (tolerances + 2 GiB budget are tuned for {DEFAULT_WIDTH}x{DEFAULT_HEIGHT})"
        );
    }
    eprintln!(
        "e2e-parity[{}]: starting dims={}x{} skip_pyref={}",
        variant.slug(),
        width,
        height,
        skip_pyref
    );
    let h_lat = (height as usize) / VAE_SCALE;
    let w_lat = (width as usize) / VAE_SCALE;
    let n_lat = LATENT_CHANNELS * h_lat * w_lat;
    let img_h = height as usize;
    let img_w = width as usize;
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
    if let Some(r) = variant.te_gguf_role() {
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
    // DIAG (slope-0.938 localization): pyref full residual after block 29
    // and full final_layer_out, dumped via `_mk_once_post_dump`. Named
    // `py_<label>.bin` to match the python side.
    let py_block29_full_path = tmp.join("py_10_main_block29_full.bin");
    let py_final_layer_full_path = tmp.join("py_11_final_layer_full.bin");

    // Clear stale dumps so a stale file can't mask a hook that never fires.
    let mut clear_paths = vec![
        py_starting_path.clone(),
        py_pre_vae_path.clone(),
        py_vae_rgb_path.clone(),
        py_block29_full_path.clone(),
        py_final_layer_full_path.clone(),
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
    if skip_pyref {
        eprintln!(
            "e2e-parity[{}]: SKIP_PYREF set; pytorch reference not invoked",
            variant.slug()
        );
    } else {
        run_python_ref(
            &noise_path,
            &dit_shards,
            &tmp,
            png_dir.as_deref(),
            &png_filename_py,
            &vae_diag_dir,
            width,
            height,
        );
    }

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
    let gguf_openers = match (variant.gguf_role(), variant.te_gguf_role()) {
        (Some(dit_r), Some(te_r)) => {
            let dit_path = path_of(dit_r);
            let te_path = path_of(te_r);
            eprintln!(
                "e2e-parity[{}]: union GGUFs over safetensors: {} + {}",
                variant.slug(),
                dit_path.display(),
                te_path.display()
            );
            Some(GgufOpeners {
                dit: MmapFileOpener::new(dit_path)
                    .await
                    .unwrap_or_else(|e| panic!("open gguf {}: {e}", dit_path.display())),
                te: MmapFileOpener::new(te_path)
                    .await
                    .unwrap_or_else(|e| panic!("open gguf {}: {e}", te_path.display())),
            })
        }
        _ => None,
    };
    // Schema adapters + optional GGUF union live in `ZImageSource::open`,
    // shared with the CLI and web.
    let source = ZImageSource::open(openers, gguf_openers)
        .await
        .expect("parse weight files");
    // The base source still carries the bf16 TE shards, so a misnamed
    // `qwen3_gguf_renames` map would silently fall back to safetensors and
    // pass a run identical to the bf16 TE. Assert the union actually
    // serves the TE from the GGUF (same trap-guard as
    // `expected_dit_matmul_weight` for the DiT side).
    if variant.te_gguf_role().is_some() {
        use thinfer_core::tensor::StorageEncoding;
        use thinfer_core::weight::{WeightId, WeightSource as _};
        let enc = source
            .catalog()
            .get(&WeightId("model.layers.0.self_attn.q_proj.weight".into()))
            .and_then(|e| e.encoding);
        assert_eq!(
            enc,
            Some(StorageEncoding::Quant(QuantKind::Q8_0)),
            "TE q_proj must come from the Q8_0 GGUF, not the safetensors fallback"
        );
    }

    // Aggressive 2 GiB / 2 GiB budgets - intentionally well under model
    // size to exercise the eviction + rolling-residency paths. Anything
    // that assumes "the whole DiT fits in VRAM" must be re-thought.
    // `THINFER_E2E_BUDGET_GB` overrides both (perf A/B: streaming vs
    // fully-resident weights).
    let budget_gb: u64 = std::env::var("THINFER_E2E_BUDGET_GB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let budget = ResidencyBudget {
        ram_bytes: budget_gb << 30,
        vram_bytes: budget_gb << 30,
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
        width,
        height,
        skip_pyref,
        png_dir,
        png_filename_ours,
        vae_diag_dir,
        py_starting_path,
        py_pre_vae_path,
        py_vae_rgb_path,
        py_step_paths,
        py_block29_full_path,
        py_final_layer_full_path,
        budget,
    };

    let residency = WeightResidency::new(source, budget);
    run_pipeline_and_compare(backend, residency, tokenizer, ctx).await;
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
    width: u32,
    height: u32,
    skip_pyref: bool,
    png_dir: Option<PathBuf>,
    png_filename_ours: String,
    vae_diag_dir: PathBuf,
    py_starting_path: PathBuf,
    py_pre_vae_path: PathBuf,
    py_vae_rgb_path: PathBuf,
    py_step_paths: Vec<PathBuf>,
    py_block29_full_path: PathBuf,
    py_final_layer_full_path: PathBuf,
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
        height: ctx.height,
        width: ctx.width,
        steps: STEPS,
        seed: SEED,
    };
    let mut ws = Workspace::new(Arc::clone(&backend), Arc::clone(model.arbiter()));
    let mut our_step_dumps: Vec<Vec<f32>> = Vec::with_capacity(STEPS as usize);
    // DIAG sink for step 0: residual after block 29, final_layer_out,
    // post-unpatchify+negation velocity, step-0 dt, and the damage-zone
    // per-op taps + block-26 byte-level matmul audit. The tap buffers pin
    // hundreds of MiB of workspace, which busts the 2 GiB VRAM budget on
    // default runs - opt in via THINFER_E2E_STEP0_DIAG=1 when localizing
    // a divergence. All downstream consumers skip on empty sinks.
    let step0_requested = std::env::var("THINFER_E2E_STEP0_DIAG").as_deref() == Ok("1");
    let mut step0: Step0LocalizationTaps = Step0LocalizationTaps::default();
    let (our_pre_vae, our_h_lat, our_w_lat) = model
        .denoise_with(
            &params,
            Some(&ctx.noise),
            &mut ws,
            Some(&mut our_step_dumps),
            if step0_requested {
                Some(&mut step0)
            } else {
                None
            },
            None,
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
        match encode_png(&our_rgb, ctx.width, ctx.height) {
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

    for (i, s) in our_step_dumps.iter().enumerate() {
        summarize(&format!("our_step{i}_prev_sample"), s);
    }
    summarize("our_pre_vae_latent", &our_pre_vae);
    summarize("our_vae_rgb", &our_rgb);

    // Budget asserts: same ceiling regardless of skip_pyref. Workspace
    // peak must not depend on whether py-ref ran. Defined once here so
    // both branches gate identically.
    let assert_budgets = || {
        let vram_over = snap.vram_total_peak > ctx.budget.vram_bytes;
        let ram_over = snap.ram_total_peak > ctx.budget.ram_bytes;
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
    };

    if ctx.skip_pyref {
        eprintln!(
            "[{}] SKIP_PYREF: skipping py-vs-engine divergence checks (budget asserts still enforced)",
            ctx.variant.slug()
        );
        eprintln!("---- vae per-stage diag (ours-only) ----");
        for st in &our_vae_diag {
            summarize(&format!("our_vae.{}", st.label), &st.head);
        }
        assert_budgets();
        return;
    }

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

    let tols = ctx.variant.tolerances();
    let mut diverged: Option<String> = None;
    let trace_dump = std::env::var("THINFER_TRACE").is_ok();
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
        // Trace-gated: cell-level dump that lets us tell apart (a) accumulated
        // noise (cells off-by-a-little everywhere), (b) layout / token-order
        // bugs (cells correct but shifted/permuted), (c) localized blow-up
        // (a few cells very wrong, most fine). Head/mid/tail give the
        // structural view; first-K divergent indices give the failure modes.
        if trace_dump && n > 0 {
            let dump_n = 8.min(n);
            let mid_start = (n / 2).saturating_sub(dump_n / 2);
            let mid_end = (mid_start + dump_n).min(n);
            let tail_start = n.saturating_sub(dump_n);
            eprintln!("[{label}] head        got={:?}", &got[..dump_n]);
            eprintln!("[{label}] head        exp={:?}", &expected[..dump_n]);
            eprintln!(
                "[{label}] mid@{mid_start}     got={:?}",
                &got[mid_start..mid_end]
            );
            eprintln!(
                "[{label}] mid@{mid_start}     exp={:?}",
                &expected[mid_start..mid_end]
            );
            eprintln!(
                "[{label}] tail@{tail_start}    got={:?}",
                &got[tail_start..n]
            );
            eprintln!(
                "[{label}] tail@{tail_start}    exp={:?}",
                &expected[tail_start..n]
            );
            // Sample the first 16 divergent indices: tells transpose
            // (regular index pattern) apart from quant drift (random).
            let mut sampled = 0usize;
            for i in 0..n {
                if sampled >= 16 {
                    break;
                }
                let g = got[i];
                let e = expected[i];
                let nan_or_inf = !g.is_finite() || !e.is_finite();
                let d = (g - e).abs();
                if nan_or_inf || d > tol {
                    eprintln!(
                        "[{label}] diverge[{sampled}] idx={i} got={g:+.6e} exp={e:+.6e} diff={d:.4e}"
                    );
                    sampled += 1;
                }
            }
            // Scale/shift fit over the FULL tensor. (The previous head-4096
            // fit covered a third of row 0 and hid both row-misalignment and
            // the true noise floor for weeks.) `rel` = rmse / std(exp) is the
            // scale-free residual: comparable across ops of different
            // magnitudes, so the per-op chain shows exactly where error
            // jumps. rel ~ 0 clean, rel ~ 1 means got is uncorrelated noise.
            let (a, b, rmse, cnt) = linfit(expected, got);
            if cnt > 2 {
                let mut sx = 0.0f64;
                let mut sxx = 0.0f64;
                let mut nf = 0usize;
                for &x in expected {
                    if x.is_finite() {
                        sx += x as f64;
                        sxx += (x as f64) * (x as f64);
                        nf += 1;
                    }
                }
                let mean = sx / nf as f64;
                let std = (sxx / nf as f64 - mean * mean).max(0.0).sqrt();
                let rel = if std > 0.0 { rmse / std } else { f64::NAN };
                eprintln!(
                    "[{label}] linfit  got ~= {a:.6}*exp + {b:+.6}  rmse={rmse:.4e}  \
                     rel={rel:.4}  std_exp={std:.4e}  (n={cnt} full)"
                );
            }
        }
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

    // DIAG (slope-0.938 localization): four-point comparison across the
    // chain block29 -> final_layer -> model_output_post_neg ->
    // prev_sample. The linfit slope printed for each tells us where the
    // ~0.94x shrink first appears. The 0/0/0 max_n_over makes these
    // pure-diagnostic (won't gate the test) -- failure was already
    // recorded at step0.prev_sample above.
    if !step0.last_main.is_empty() && ctx.py_block29_full_path.exists() {
        let py_block29 = read_f32(&ctx.py_block29_full_path);
        summarize("py_10_main_block29_full", &py_block29);
        summarize("our_step0_last_main_block29", &step0.last_main);
        check(
            "DIAG step0.block29_full (last_main residual)",
            &step0.last_main,
            &py_block29,
            usize::MAX,
        );
    } else {
        eprintln!(
            "[DIAG] block29_full skipped: our_len={} py_exists={}",
            step0.last_main.len(),
            ctx.py_block29_full_path.exists()
        );
    }
    if !step0.final_layer.is_empty() && ctx.py_final_layer_full_path.exists() {
        let py_final = read_f32(&ctx.py_final_layer_full_path);
        summarize("py_11_final_layer_full", &py_final);
        summarize("our_step0_final_layer", &step0.final_layer);
        check(
            "DIAG step0.final_layer_full",
            &step0.final_layer,
            &py_final,
            usize::MAX,
        );
    } else {
        eprintln!(
            "[DIAG] final_layer_full skipped: our_len={} py_exists={}",
            step0.final_layer.len(),
            ctx.py_final_layer_full_path.exists()
        );
    }
    // Post-unpatchify + post-negation velocity = scheduler input at step
    // 0. Pyref doesn't dump this directly; derive it from prev_sample
    // and starting_latents using the engine's dt_step0 (matches pyref's
    // because both run FlowMatchEulerScheduler with identical sigmas).
    if !step0.model_output_post_neg.is_empty() && !py_steps.is_empty() {
        let dt = step0.dt_step0;
        if dt.abs() > 1e-9 {
            let py_vel: Vec<f32> = py_steps[0]
                .iter()
                .zip(py_starting.iter())
                .map(|(p, s)| (p - s) / dt)
                .collect();
            summarize("py_step0_velocity (= (prev_sample - starting)/dt)", &py_vel);
            summarize(
                "our_step0_model_output_post_neg",
                &step0.model_output_post_neg,
            );
            eprintln!("[DIAG] step0 dt = {dt:.6e}");
            check(
                "DIAG step0.model_output_post_neg (== scheduler input)",
                &step0.model_output_post_neg,
                &py_vel,
                usize::MAX,
            );
        }
    }

    // Per-block residual stream linfit across all 30 main blocks. The
    // first block where slope deviates from 1.0 localizes the bug to
    // that block's I8 ops. Pyref dumps `py_09_main_block{n}_out.bin`
    // for every n; engine fills `per_block_residual[n]` with f32 post
    // -block-n residual.
    let tmp_dir = ctx
        .py_starting_path
        .parent()
        .expect("py_starting_path has parent")
        .to_path_buf();
    eprintln!("---- DIAG per-block residual linfit ----");
    for b in 0..step0.per_block_residual.len() {
        let py_path = tmp_dir.join(format!("py_09_main_block{b}_out.bin"));
        if step0.per_block_residual[b].is_empty() || !py_path.exists() {
            eprintln!(
                "[DIAG block{b}] skipped: our_len={} py_exists={}",
                step0.per_block_residual[b].len(),
                py_path.exists()
            );
            continue;
        }
        let py = read_f32(&py_path);
        check(
            &format!("DIAG main_block{b}_out"),
            &step0.per_block_residual[b],
            &py,
            usize::MAX,
        );
    }

    // Per-op linfit inside main block 0 and main block N-1. Mapping
    // from engine field name to pyref `_mk_once_post` label suffix
    // (gen_e2e_parity_ref.py). Compares only where both sides
    // populate; engine fields without a pyref counterpart (e.g.
    // attn_norm1_out, attn_q_norm, attn_q_rope) print "skipped".
    type BlockOpPair = (
        &'static str,
        &'static str,
        fn(&Block0LocalTaps) -> &Vec<f32>,
    );
    // Dataflow order within a block. The full-tensor linfit `rel` column
    // across consecutive entries shows exactly which step injects error:
    // modulate -> qkv matmul -> sdpa -> o_proj -> norm2 -> RESIDUAL ADD
    // (x_mid) -> ffn_norm1 -> modulate -> ffn (3 matmuls + silu_mul) ->
    // norm2 -> residual add (= main_blockN_out, in the trajectory probe).
    let block_op_pairs: &[BlockOpPair] = &[
        ("modulated_attn_in", "modulated_attn_in", |t| {
            &t.modulated_attn_in
        }),
        ("attn_q", "to_q", |t| &t.attn_q),
        ("attn_k", "to_k", |t| &t.attn_k),
        ("attn_v", "to_v", |t| &t.attn_v),
        ("attn_sdpa", "to_out0_in", |t| &t.attn_sdpa),
        ("attn_out", "to_out0", |t| &t.attn_out),
        ("attn_norm2_out", "attn_norm2_out", |t| &t.attn_norm2_out),
        ("x_mid", "x_mid", |t| &t.x_mid),
        ("ffn_norm1_out", "ffn_norm1_out", |t| &t.ffn_norm1_out),
        ("modulated_ffn_in", "modulated_ffn_in", |t| {
            &t.modulated_ffn_in
        }),
        ("ffn_raw", "ffn_out", |t| &t.ffn_raw),
        ("ffn_norm2_out", "ffn_norm2_out", |t| &t.ffn_norm2_out),
    ];
    let mut block_op_iter: Vec<(usize, &Block0LocalTaps)> =
        vec![(0usize, &step0.block0), (29usize, &step0.block_last)];
    for (b, t) in &step0.damage_zone {
        block_op_iter.push((*b, t));
    }
    block_op_iter.sort_by_key(|(b, _)| *b);
    for (block_n, taps) in block_op_iter {
        eprintln!("---- DIAG block{block_n} per-op linfit ----");
        for (eng_name, py_suffix, getter) in block_op_pairs {
            let our = getter(taps);
            let py_path = tmp_dir.join(format!("py_09_main_block{block_n}_{py_suffix}.bin"));
            if our.is_empty() || !py_path.exists() {
                eprintln!(
                    "[DIAG block{block_n}.{eng_name}] skipped: our_len={} py_path={} (exists={})",
                    our.len(),
                    py_path.display(),
                    py_path.exists()
                );
                continue;
            }
            let py = read_f32(&py_path);
            check(
                &format!("DIAG block{block_n}.{eng_name}"),
                our,
                &py,
                usize::MAX,
            );
        }
    }
    // Block-26 matmul_i8 byte-level audit. CPU-recomputes one f16 output
    // element from the captured byte heads (a_i8, a_params, b_i8, b_scale,
    // b_qsum) and compares against the GPU's `attn_qkv_f16_pre_quant`.
    // Formula audited: `acc = Σ_K (i8_dot * sa * sb) + Σ_K (za * sb * b_qsum)`
    // where K = number of 32-element blocks, sa/za from (s,z) per a_params
    // K-block, sb from f16 b_scale per K-block, b_qsum from f32 b_qsum per
    // K-block. Bit-clean match → matmul kernel + dequant_i8 + act_quant
    // arithmetic verified. Mismatch → identifies which factor is wrong.
    audit_block26_matmul_i8(&step0.damage_zone);
    // Phase A: per-(Q/K/V)-segment linfit of the fused matmul output
    // `attn_qkv_f16_pre_quant` against pyref `to_q.bin` / `to_k.bin` /
    // `to_v.bin`. Isolates "matmul V-columns already bad" from "qkv_split
    // mangles V" — V-segment slope ≈ pyref engine.attn_v slope means the
    // matmul output is the bug; V-segment slope ≈ Q/K (≈0.99) means the
    // split or downstream requant is the bug. Also reports magnitude
    // stats per segment so we can see whether V columns are
    // systematically heavier-tailed than Q/K (relevant to act_quant
    // error per block).
    audit_qkv_segment_slopes(&step0.damage_zone, &tmp_dir);
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

    // Parity first so the divergence message lands in the log before
    // any budget failure. Budget gate runs after.
    if let Some(msg) = diverged {
        panic!("FIRST DIVERGENCE: {msg}");
    }
    assert_budgets();
}

fn run_python_ref(
    noise_path: &Path,
    dit_shards: &[PathBuf],
    out_dir: &Path,
    png_dir: Option<&Path>,
    png_filename: &str,
    vae_diag_dir: &Path,
    width: u32,
    height: u32,
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
        &height.to_string(),
        "--width",
        &width.to_string(),
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

pub(crate) fn summarize(label: &str, v: &[f32]) {
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

pub(crate) fn read_f32(p: &Path) -> Vec<f32> {
    let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Block-26 matmul_i8 byte-level audit. CPU-recomputes one f16 output
/// element from the captured byte heads. Bit-clean match against the
/// GPU's `attn_qkv_f16_pre_quant` proves matmul_i8 + dequant_i8 +
/// act_quant arithmetic is correct on real heavy-tail data. Drift
/// localizes the bug to a specific factor (main term, correction term,
/// or one of sa/sb/za/b_qsum).
/// Phase A: linfit each (Q, K, V) column-segment of the captured fused
/// matmul output against pyref `to_q.bin` / `to_k.bin` / `to_v.bin` per
/// block. Also linfits each segment against the engine's downstream
/// post-split tap (`attn_q` / `attn_k` / `attn_v`) so we can see how
/// much slope drift is added by `act_quant_matmul_out` + `qkv_split_i8`
/// between the fused f16 output and the per-Q/K/V paired-i8 buffers.
fn audit_qkv_segment_slopes(damage_zone: &[(usize, Block0LocalTaps)], tmp_dir: &Path) {
    const N_QKV: usize = 11520;
    const H: usize = 3840;
    eprintln!("---- Phase A: per-segment matmul-output linfit ----");
    for (b, taps) in damage_zone {
        let fused = &taps.attn_qkv_f16_pre_quant;
        if fused.is_empty() || !fused.len().is_multiple_of(N_QKV) {
            eprintln!(
                "[Phase A block{b}] skipped: fused.len()={} (expected multiple of {N_QKV})",
                fused.len()
            );
            continue;
        }
        let rows = fused.len() / N_QKV;
        let segs: [(&str, &str, usize, &Vec<f32>); 3] = [
            ("Q", "to_q", 0, &taps.attn_q),
            ("K", "to_k", H, &taps.attn_k),
            ("V", "to_v", 2 * H, &taps.attn_v),
        ];
        for (seg_name, py_suffix, col_off, eng_split) in segs {
            // Gather seg = fused[:, col_off..col_off+H] as flat Vec<f32>.
            let mut seg = Vec::with_capacity(rows * H);
            for r in 0..rows {
                let row_off = r * N_QKV + col_off;
                seg.extend_from_slice(&fused[row_off..row_off + H]);
            }
            // Per-segment magnitude stats (max_abs, mean, std).
            let mut sum = 0.0f64;
            let mut sumsq = 0.0f64;
            let mut max_abs = 0.0f32;
            for &x in &seg {
                if !x.is_finite() {
                    continue;
                }
                sum += x as f64;
                sumsq += (x as f64) * (x as f64);
                let a = x.abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
            let n = seg.len() as f64;
            let mean = sum / n;
            let var = (sumsq / n - mean * mean).max(0.0);
            let std = var.sqrt();
            eprintln!(
                "[Phase A block{b}.{seg_name}] segment stats: max_abs={max_abs:.4e} \
                 mean={mean:.4e} std={std:.4e} (rows={rows}, h={H})"
            );
            // Locate the top-5 outlier cells (by |seg[i] - pyref[i]| if pyref
            // available, else by |seg[i]|) and report their (row, col-within-
            // segment) and the matching pyref value. Tells us whether the
            // outliers are concentrated at specific N-columns (e.g. a single
            // mis-strided weight column) vs scattered randomly.
            let py_path = tmp_dir.join(format!("py_09_main_block{b}_{py_suffix}.bin"));
            let py_opt: Option<Vec<f32>> = if py_path.exists() {
                let v = read_f32(&py_path);
                if v.len() == seg.len() {
                    Some(v)
                } else {
                    eprintln!(
                        "[Phase A block{b}.{seg_name}] py len mismatch: \
                         seg={} py={}",
                        seg.len(),
                        v.len()
                    );
                    None
                }
            } else {
                eprintln!(
                    "[Phase A block{b}.{seg_name}] pyref missing: {}",
                    py_path.display()
                );
                None
            };
            if let Some(ref py) = py_opt {
                // Full-tensor linfit (1.1M cells, includes outliers).
                let (a_full, _, rmse_full, n_full) = linfit(&seg, py);
                // First-4096-cell linfit (matches existing per-op DIAG).
                let head_n = 4096usize.min(seg.len());
                let (a_head, _, rmse_head, n_head) = linfit(&seg[..head_n], &py[..head_n]);
                // Outlier-trimmed linfit: drop any cell where |seg[i]| or
                // |py[i]| exceeds 5 * std (relative to the segment's own
                // std). Should match the head linfit if outliers are the
                // story; if it's still ≈ 0.13, the bulk is also off.
                let cutoff = 5.0 * std as f32;
                let mut x_trim = Vec::with_capacity(seg.len());
                let mut y_trim = Vec::with_capacity(seg.len());
                let mut n_outliers = 0usize;
                for i in 0..seg.len() {
                    if seg[i].abs() > cutoff || py[i].abs() > cutoff {
                        n_outliers += 1;
                    } else {
                        x_trim.push(seg[i]);
                        y_trim.push(py[i]);
                    }
                }
                let (a_trim, _, rmse_trim, n_trim) = linfit(&x_trim, &y_trim);
                eprintln!(
                    "[Phase A block{b}.{seg_name}] fused-vs-pyref({py_suffix}): \
                     full slope={a_full:.6} rmse={rmse_full:.4e} n={n_full} | \
                     head4096 slope={a_head:.6} rmse={rmse_head:.4e} n={n_head} | \
                     trimmed(>{cutoff:.2e}) slope={a_trim:.6} rmse={rmse_trim:.4e} \
                     n={n_trim} outliers={n_outliers}"
                );
                // Top-5 outliers by |seg - py| with their (row, col-within-seg).
                let mut diffs: Vec<(usize, f32, f32, f32)> = seg
                    .iter()
                    .zip(py.iter())
                    .enumerate()
                    .map(|(i, (s, p))| (i, *s, *p, (s - p).abs()))
                    .collect();
                let pivot = 5.min(diffs.len() - 1);
                diffs.select_nth_unstable_by(pivot, |a, b| {
                    b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal)
                });
                let top: Vec<_> = diffs[..5.min(diffs.len())].to_vec();
                let mut top_sorted = top.clone();
                top_sorted
                    .sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
                for (i, s_val, p_val, d) in top_sorted {
                    let row = i / H;
                    let col_in_seg = i % H;
                    eprintln!(
                        "[Phase A block{b}.{seg_name}] outlier row={row} \
                         col_in_seg={col_in_seg} (abs_col={}) seg={s_val:+.4e} \
                         py={p_val:+.4e} diff={d:.4e}",
                        col_in_seg + col_off
                    );
                }
            }
            // Per-column max_abs scan within this segment. If outliers
            // are concentrated at specific N-columns (i.e. a specific
            // weight row produces giant outputs), this prints the top-5
            // columns by max_abs. If random/sparse, the max_abs will be
            // similar across columns. Also reports column-wise mean(|x|)
            // for the worst columns vs the segment-wide mean.
            let mut col_max_abs = vec![0.0f32; H];
            let mut col_sum_abs = vec![0.0f64; H];
            for r in 0..rows {
                let row_off = r * N_QKV + col_off;
                for c in 0..H {
                    let v = fused[row_off + c];
                    let a = v.abs();
                    if a > col_max_abs[c] {
                        col_max_abs[c] = a;
                    }
                    col_sum_abs[c] += a as f64;
                }
            }
            let seg_mean_abs = (col_sum_abs.iter().sum::<f64>() / (rows as f64 * H as f64)) as f32;
            let mut col_idx: Vec<usize> = (0..H).collect();
            col_idx.sort_by(|&a, &b| {
                col_max_abs[b]
                    .partial_cmp(&col_max_abs[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            eprint!(
                "[Phase A block{b}.{seg_name}] top-5 columns by max_abs (seg_mean_abs={seg_mean_abs:.3e}):"
            );
            for &c in col_idx.iter().take(5) {
                let abs_col = c + col_off;
                let cma = col_max_abs[c];
                let cm = (col_sum_abs[c] / rows as f64) as f32;
                eprint!(" col_in_seg={c}(abs={abs_col}, max_abs={cma:.3e}, mean_abs={cm:.3e})");
            }
            eprintln!();
            // Linfit engine-attn_{q,k,v} (post-requant+split) against the
            // engine-fused-segment. Slope ≈ 1.0 here means requant+split
            // preserved the segment; slope < 1.0 means the i8 round-trip
            // shrunk the segment additionally on top of any matmul drift.
            if !eng_split.is_empty() && eng_split.len() == seg.len() {
                let head_n = 4096usize.min(seg.len());
                let (a_full, _, rmse_full, n_full) = linfit(eng_split, &seg);
                let (a_head, _, rmse_head, n_head) = linfit(&eng_split[..head_n], &seg[..head_n]);
                eprintln!(
                    "[Phase A block{b}.{seg_name}] split-vs-fused (i8 round-trip): \
                     full slope={a_full:.6} rmse={rmse_full:.4e} n={n_full} | \
                     head4096 slope={a_head:.6} rmse={rmse_head:.4e} n={n_head}"
                );
            }
            // CPU per-cell recompute of the WORST outlier in this segment.
            // Compares GPU output, per-block formula CPU, and naive element-
            // wise CPU. Three-way agreement isolates the bug:
            //   GPU == naive == per_block != pyref → bytes are wrong (upstream)
            //   GPU != naive == per_block          → matmul kernel is wrong
            //   per_block != naive                 → formula off (precision)
            // Requires the qkv_* byte heads populated. Currently only block 26
            // populates them (see pipeline.rs damage_zone setup).
            if !taps.qkv_attn_in_data_head.is_empty()
                && !taps.qkv_attn_in_params_head.is_empty()
                && !taps.qkv_b_i8_head.is_empty()
                && !taps.qkv_b_scale_head.is_empty()
                && !taps.qkv_b_qsum_head.is_empty()
                && let Some(ref py) = py_opt
            {
                // Use the very top outlier (by |seg - py| diff) we identified
                // earlier. Reconstruct (m, abs_col) from the linear index.
                // Find it again here so we don't have to thread it out of
                // the previous block.
                let mut worst_idx = 0usize;
                let mut worst_diff = 0.0f32;
                for i in 0..seg.len() {
                    let d = (seg[i] - py[i]).abs();
                    if d > worst_diff {
                        worst_diff = d;
                        worst_idx = i;
                    }
                }
                let m = worst_idx / H;
                let n = (worst_idx % H) + col_off;
                let gpu_val = fused[m * N_QKV + n];
                let py_val = py[worst_idx];
                recompute_cell_at(
                    *b,
                    seg_name,
                    m,
                    n,
                    gpu_val,
                    Some(py_val),
                    &taps.qkv_attn_in_data_head,
                    &taps.qkv_attn_in_params_head,
                    &taps.qkv_b_i8_head,
                    &taps.qkv_b_scale_head,
                    &taps.qkv_b_qsum_head,
                    rows,
                );
                // Same recompute at a control row (m=100, interior) and at
                // every tail row 318..324, all at the SAME outlier column.
                // If only tail rows diverge, the bug is partial-tile-related.
                // If the interior row also disagrees with GPU, it's a more
                // fundamental kernel/recompute mismatch.
                for &m_extra in &[100usize, 200, 282, 283, 284, 285, 286, 287] {
                    if m_extra == m {
                        continue;
                    }
                    if m_extra >= rows {
                        continue;
                    }
                    let gpu_v = fused[m_extra * N_QKV + n];
                    let py_v = py[m_extra * H + (n - col_off)];
                    recompute_cell_at(
                        *b,
                        seg_name,
                        m_extra,
                        n,
                        gpu_v,
                        Some(py_v),
                        &taps.qkv_attn_in_data_head,
                        &taps.qkv_attn_in_params_head,
                        &taps.qkv_b_i8_head,
                        &taps.qkv_b_scale_head,
                        &taps.qkv_b_qsum_head,
                        rows,
                    );
                }
                // Full-row CPU oracle at an interior row and the worst tail
                // rows: validates the captured B bytes + formula against
                // pyref truth across all H columns at once.
                for &m_row in &[0usize, 100, 286, 287] {
                    if m_row >= rows {
                        continue;
                    }
                    let gpu_row: Vec<f32> =
                        (0..H).map(|c| fused[m_row * N_QKV + col_off + c]).collect();
                    let py_row = &py[m_row * H..(m_row + 1) * H];
                    recompute_full_row(
                        *b,
                        seg_name,
                        m_row,
                        col_off,
                        &gpu_row,
                        py_row,
                        &taps.qkv_attn_in_data_head,
                        &taps.qkv_attn_in_params_head,
                        &taps.qkv_b_i8_head,
                        &taps.qkv_b_scale_head,
                        &taps.qkv_b_qsum_head,
                    );
                }
                // Per-row scan at the outlier column: row signature of GPU
                // badness vs the row's max |za| (heavy-tail correlation).
                scan_rows_at_col(
                    *b,
                    seg_name,
                    n,
                    col_off,
                    rows,
                    fused,
                    py,
                    &taps.qkv_attn_in_data_head,
                    &taps.qkv_attn_in_params_head,
                    &taps.qkv_b_i8_head,
                    &taps.qkv_b_scale_head,
                    &taps.qkv_b_qsum_head,
                );
                // GPU in-kernel per-K-block trace at the target hardcoded in
                // block.rs (m=287, n=255 in the QKV-fused output, which is Q
                // col 255). Decode 120 K-blocks * 8 f32s and print side-by-
                // side with the CPU recompute's per-block view, so we see
                // exactly which K-block first diverges between GPU and CPU.
                if seg_name == "Q" && !taps.qkv_dbg_trace_head.is_empty() {
                    decode_and_diff_gpu_trace(
                        *b,
                        287,
                        255,
                        &taps.qkv_dbg_trace_head,
                        &taps.qkv_attn_in_data_head,
                        &taps.qkv_attn_in_params_head,
                        &taps.qkv_b_i8_head,
                        &taps.qkv_b_scale_head,
                        &taps.qkv_b_qsum_head,
                    );
                }
                // Also dump the per-K-block (s, z) for THIS outlier row m
                // so we can see whether the row has anomalously large z
                // anywhere (i.e. heavy-tail input block driving the
                // correction).
                dump_row_act_params(*b, seg_name, m, &taps.qkv_attn_in_params_head);
                // Compare engine's PRE-QUANT input row m against pyref's
                // modulated_attn_in row m. If the matmul-bytes recompute
                // matches GPU but disagrees with pyref, this tells us
                // whether the input itself was already wrong, or whether
                // i8 quant of a correct input lost the heavy-tail info.
                compare_input_row_against_pyref(
                    *b,
                    seg_name,
                    m,
                    rows,
                    tmp_dir,
                    &taps.qkv_attn_in_data_head,
                    &taps.qkv_attn_in_params_head,
                );
            }
        }
    }
}

/// CPU recompute one matmul cell (m, n) two ways: per-block asymmetric
/// formula `acc = Σ dot*sa*sb + Σ za*sb*b_qsum`, and naive element-wise
/// `Σ (q_a*sa+z_a) * (q_b*sb)`. Equal under perfect arithmetic; divergence
/// means a precision / formula issue. Both should match GPU; if they
/// don't, the kernel is wrong.
fn recompute_cell_at(
    block: usize,
    seg_name: &str,
    m: usize,
    n: usize,
    gpu_val: f32,
    py_val: Option<f32>,
    a_i8: &[u8],
    a_params: &[u8],
    b_i8: &[u8],
    b_scale: &[u8],
    b_qsum: &[u8],
    rows: usize,
) {
    const K: usize = 3840;
    const K_BLOCKS: usize = 120;
    let n_a_rows_have = a_i8.len() / K;
    let n_b_rows_have = b_i8.len() / K;
    if m >= n_a_rows_have || n >= n_b_rows_have {
        eprintln!(
            "[recompute block{block}.{seg_name}] SKIP m={m} n={n}: \
             byte-heads cover only a_rows={n_a_rows_have} b_rows={n_b_rows_have}"
        );
        return;
    }
    let _ = rows;
    let read_f16 = |bytes: &[u8], i: usize| -> f32 {
        let b = i * 2;
        half::f16::from_le_bytes([bytes[b], bytes[b + 1]]).to_f32()
    };
    let read_f32 = |bytes: &[u8], i: usize| -> f32 {
        let b = i * 4;
        f32::from_le_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]])
    };
    let mut acc_main = 0.0f64;
    let mut acc_corr = 0.0f64;
    let mut acc_naive = 0.0f64;
    let mut max_per_block_main = 0.0f64;
    let mut max_per_block_corr = 0.0f64;
    for kb in 0..K_BLOCKS {
        let sa = read_f16(a_params, (m * K_BLOCKS + kb) * 2) as f64;
        let za = read_f16(a_params, (m * K_BLOCKS + kb) * 2 + 1) as f64;
        let sb = read_f32(b_scale, n * K_BLOCKS + kb) as f64;
        let bq = read_f32(b_qsum, n * K_BLOCKS + kb) as f64;
        let mut dot: i64 = 0;
        let mut naive_blk = 0.0f64;
        for i in 0..32 {
            let ai = a_i8[m * K + kb * 32 + i] as i8 as i64;
            let bi = b_i8[n * K + kb * 32 + i] as i8 as i64;
            dot += ai * bi;
            let xa = (ai as f64) * sa + za;
            let xb = (bi as f64) * sb;
            naive_blk += xa * xb;
        }
        let main = (dot as f64) * sa * sb;
        let corr = za * sb * bq;
        acc_main += main;
        acc_corr += corr;
        acc_naive += naive_blk;
        if main.abs() > max_per_block_main.abs() {
            max_per_block_main = main;
        }
        if corr.abs() > max_per_block_corr.abs() {
            max_per_block_corr = corr;
        }
    }
    let cpu_pb = (acc_main + acc_corr) as f32;
    let cpu_naive = acc_naive as f32;
    let py_str = match py_val {
        Some(v) => format!("py={v:+.4e}"),
        None => "py=N/A".into(),
    };
    eprintln!(
        "[recompute block{block}.{seg_name}] m={m} n={n} \
         gpu={gpu_val:+.4e} cpu_pb={cpu_pb:+.4e} cpu_naive={cpu_naive:+.4e} \
         {py_str} | acc_main={:+.4e} acc_corr={:+.4e} max_blk_main={:+.4e} \
         max_blk_corr={:+.4e}",
        acc_main as f32, acc_corr as f32, max_per_block_main as f32, max_per_block_corr as f32,
    );
}

/// CPU recompute of one FULL output row (all H columns of one segment) from
/// the captured byte heads, linfit against both the pyref row and the GPU
/// row. This is the decisive validity check on the captured B-side bytes:
/// if cpu-vs-py slope ≈ 1 the captured (b_i8, b_scale, b_qsum) really are
/// this site's weights and the asymmetric formula is right; if cpu-vs-py is
/// uncorrelated, the captured B bytes are NOT what the kernel consumed.
#[allow(clippy::too_many_arguments)]
fn recompute_full_row(
    block: usize,
    seg_name: &str,
    m: usize,
    col_off: usize,
    gpu_row: &[f32],
    py_row: &[f32],
    a_i8: &[u8],
    a_params: &[u8],
    b_i8: &[u8],
    b_scale: &[u8],
    b_qsum: &[u8],
) {
    const K: usize = 3840;
    const K_BLOCKS: usize = 120;
    const H: usize = 3840;
    let n_a_rows_have = a_i8.len() / K;
    let n_b_rows_have = b_i8.len() / K;
    if m >= n_a_rows_have || col_off + H > n_b_rows_have {
        eprintln!(
            "[cpu_row block{block}.{seg_name}] SKIP m={m}: heads cover \
             a_rows={n_a_rows_have} b_rows={n_b_rows_have}"
        );
        return;
    }
    let read_f16 = |bytes: &[u8], i: usize| -> f32 {
        let b = i * 2;
        half::f16::from_le_bytes([bytes[b], bytes[b + 1]]).to_f32()
    };
    let read_f32 = |bytes: &[u8], i: usize| -> f32 {
        let b = i * 4;
        f32::from_le_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]])
    };
    // Hoist the A row decode: per-block (sa, za) + i8 values.
    let mut cpu = vec![0.0f32; H];
    for (c, cell) in cpu.iter_mut().enumerate() {
        let n = col_off + c;
        let mut acc = 0.0f64;
        for kb in 0..K_BLOCKS {
            let sa = read_f16(a_params, (m * K_BLOCKS + kb) * 2) as f64;
            let za = read_f16(a_params, (m * K_BLOCKS + kb) * 2 + 1) as f64;
            let sb = read_f32(b_scale, n * K_BLOCKS + kb) as f64;
            let bq = read_f32(b_qsum, n * K_BLOCKS + kb) as f64;
            let mut dot: i64 = 0;
            for i in 0..32 {
                let ai = a_i8[m * K + kb * 32 + i] as i8 as i64;
                let bi = b_i8[n * K + kb * 32 + i] as i8 as i64;
                dot += ai * bi;
            }
            acc += (dot as f64) * sa * sb + za * sb * bq;
        }
        *cell = acc as f32;
    }
    let (s_py, _, rmse_py, _) = linfit(&cpu, py_row);
    let (s_gpu, _, rmse_gpu, _) = linfit(&cpu, gpu_row);
    // Worst-3 cells by |cpu - py|.
    let mut worst: Vec<(usize, f32)> = cpu
        .iter()
        .zip(py_row.iter())
        .map(|(c, p)| (c - p).abs())
        .enumerate()
        .collect();
    worst.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!(
        "[cpu_row block{block}.{seg_name}] m={m}: cpu-vs-py slope={s_py:.6} \
         rmse={rmse_py:.4e} | cpu-vs-gpu slope={s_gpu:.6} rmse={rmse_gpu:.4e} | \
         worst3 vs py: n={}(cpu={:+.3e} py={:+.3e}) n={}(cpu={:+.3e} py={:+.3e}) \
         n={}(cpu={:+.3e} py={:+.3e})",
        worst[0].0,
        cpu[worst[0].0],
        py_row[worst[0].0],
        worst[1].0,
        cpu[worst[1].0],
        py_row[worst[1].0],
        worst[2].0,
        cpu[worst[2].0],
        py_row[worst[2].0],
    );
}

/// Per-row scan at ONE output column: for every row m, CPU-recompute the
/// cell, read the GPU and pyref values, and the row's max |za|. Prints a
/// compact line per row where |gpu - py| exceeds a threshold, plus summary
/// counts. Correlates row badness with heavy-tail act params.
#[allow(clippy::too_many_arguments)]
fn scan_rows_at_col(
    block: usize,
    seg_name: &str,
    n: usize,
    col_off: usize,
    rows: usize,
    fused: &[f32],
    py: &[f32],
    a_i8: &[u8],
    a_params: &[u8],
    b_i8: &[u8],
    b_scale: &[u8],
    b_qsum: &[u8],
) {
    const K: usize = 3840;
    const K_BLOCKS: usize = 120;
    const N_QKV: usize = 11520;
    const H: usize = 3840;
    let n_a_rows_have = a_i8.len() / K;
    if n * K + K > b_i8.len() {
        return;
    }
    let read_f16 = |bytes: &[u8], i: usize| -> f32 {
        let b = i * 2;
        half::f16::from_le_bytes([bytes[b], bytes[b + 1]]).to_f32()
    };
    let read_f32 = |bytes: &[u8], i: usize| -> f32 {
        let b = i * 4;
        f32::from_le_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]])
    };
    let mut n_bad_gpu = 0usize;
    let mut n_bad_cpu = 0usize;
    let mut printed = 0usize;
    eprintln!(
        "[row_scan block{block}.{seg_name}] n={n}: rows where |gpu-py| > 10 \
         (gpu | cpu_pb | py | row max|za| max|sa|):"
    );
    for m in 0..rows.min(n_a_rows_have) {
        let gpu_v = fused[m * N_QKV + n];
        let py_v = py[m * H + (n - col_off)];
        let mut acc = 0.0f64;
        let mut max_za = 0.0f32;
        let mut max_sa = 0.0f32;
        for kb in 0..K_BLOCKS {
            let sa = read_f16(a_params, (m * K_BLOCKS + kb) * 2);
            let za = read_f16(a_params, (m * K_BLOCKS + kb) * 2 + 1);
            if za.abs() > max_za {
                max_za = za.abs();
            }
            if sa.abs() > max_sa {
                max_sa = sa.abs();
            }
            let sb = read_f32(b_scale, n * K_BLOCKS + kb) as f64;
            let bq = read_f32(b_qsum, n * K_BLOCKS + kb) as f64;
            let mut dot: i64 = 0;
            for i in 0..32 {
                let ai = a_i8[m * K + kb * 32 + i] as i8 as i64;
                let bi = b_i8[n * K + kb * 32 + i] as i8 as i64;
                dot += ai * bi;
            }
            acc += (dot as f64) * (sa as f64) * sb + (za as f64) * sb * bq;
        }
        let cpu_v = acc as f32;
        let bad_gpu = (gpu_v - py_v).abs() > 10.0;
        let bad_cpu = (cpu_v - py_v).abs() > 10.0;
        if bad_gpu {
            n_bad_gpu += 1;
        }
        if bad_cpu {
            n_bad_cpu += 1;
        }
        if bad_gpu && printed < 40 {
            printed += 1;
            eprintln!(
                "  m={m:3} gpu={gpu_v:+9.2} cpu={cpu_v:+9.2} py={py_v:+9.2} \
                 max|za|={max_za:7.3} max|sa|={max_sa:.4}"
            );
        }
    }
    eprintln!(
        "[row_scan block{block}.{seg_name}] n={n}: bad_gpu={n_bad_gpu} \
         bad_cpu={n_bad_cpu} of {rows} rows (threshold |diff|>10)"
    );
}

/// Decode the GPU in-kernel per-K-block trace (96 blocks * 8 f32) and print
/// it side-by-side with a CPU recompute from the SAME captured operand bytes.
/// The first K-block where any column diverges between GPU and CPU is the
/// smoking gun: a mismatch in sa/za/sb/qsum means the tile-load read different
/// bytes than the CPU is reading (race or layout); a mismatch in dot/main/corr
/// with matching s/z/sb/qsum means the arithmetic itself diverged.
#[allow(clippy::too_many_arguments)]
fn decode_and_diff_gpu_trace(
    block: usize,
    m: usize,
    n: usize,
    dbg_trace: &[u8],
    a_i8: &[u8],
    a_params: &[u8],
    b_i8: &[u8],
    b_scale: &[u8],
    b_qsum: &[u8],
) {
    const K: usize = 3840;
    const K_BLOCKS: usize = 120;
    // Probe area at f32 indices probe0..probe0+16 (probe0 = K_BLOCKS*8):
    // kernel writes (wid.x, wid.y, enabled, dbg.m, dbg.n) at probe0..+5 and
    // marker 1234.0 at probe0+15. The head buffer is pre-filled with sentinel
    // -777.0 before the dispatch, so -777.0 here means "kernel never wrote
    // this index" while garbage means the readback hit a different buffer
    // than the kernel wrote.
    const PROBE0: usize = K_BLOCKS * 8;
    if dbg_trace.len() >= (PROBE0 + 16) * 4 {
        let r = |i: usize| -> f32 {
            let b = i * 4;
            f32::from_le_bytes([
                dbg_trace[b],
                dbg_trace[b + 1],
                dbg_trace[b + 2],
                dbg_trace[b + 3],
            ])
        };
        eprintln!(
            "[gpu_trace probe] wid=({:.0},{:.0}) dbg.enabled={:.0} dbg.m={:.0} dbg.n={:.0} marker={}",
            r(PROBE0),
            r(PROBE0 + 1),
            r(PROBE0 + 2),
            r(PROBE0 + 3),
            r(PROBE0 + 4),
            r(PROBE0 + 15)
        );
        eprint!("[gpu_trace probe raw]:");
        for i in PROBE0..PROBE0 + 16 {
            eprint!(" {}", r(i));
        }
        eprintln!();
    }
    if dbg_trace.len() < K_BLOCKS * 8 * 4 {
        eprintln!(
            "[gpu_trace block{block}] SKIP: dbg_trace.len={} (expected >= {})",
            dbg_trace.len(),
            K_BLOCKS * 8 * 4
        );
        return;
    }
    let n_a_rows_have = a_i8.len() / K;
    let n_b_rows_have = b_i8.len() / K;
    if m >= n_a_rows_have || n >= n_b_rows_have {
        eprintln!(
            "[gpu_trace block{block}] SKIP m={m} n={n}: byte-heads cover \
             a_rows={n_a_rows_have} b_rows={n_b_rows_have}"
        );
        return;
    }
    let read_f16 = |bytes: &[u8], i: usize| -> f32 {
        let b = i * 2;
        half::f16::from_le_bytes([bytes[b], bytes[b + 1]]).to_f32()
    };
    let read_f32 = |bytes: &[u8], i: usize| -> f32 {
        let b = i * 4;
        f32::from_le_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]])
    };
    eprintln!("[gpu_trace block{block}] m={m} n={n} (target hardcoded in block.rs):");
    eprintln!(
        "  kb  | sa(gpu)   sa(cpu) | za(gpu)   za(cpu) | sb(gpu)   sb(cpu) | \
         qs(gpu)   qs(cpu) | dot(gpu) dot(cpu) | main(gpu) main(cpu) | \
         corr(gpu) corr(cpu) | acc(gpu)  acc(cpu)"
    );
    let mut cpu_acc = 0.0f64;
    let mut first_div_kb: Option<usize> = None;
    for kb in 0..K_BLOCKS {
        // GPU side: read trace
        let g_sa = read_f32(dbg_trace, kb * 8);
        let g_za = read_f32(dbg_trace, kb * 8 + 1);
        let g_sb = read_f32(dbg_trace, kb * 8 + 2);
        let g_qs = read_f32(dbg_trace, kb * 8 + 3);
        let g_dot = read_f32(dbg_trace, kb * 8 + 4);
        let g_main = read_f32(dbg_trace, kb * 8 + 5);
        let g_corr = read_f32(dbg_trace, kb * 8 + 6);
        let g_acc = read_f32(dbg_trace, kb * 8 + 7);
        // CPU side: recompute from captured bytes
        let c_sa = read_f16(a_params, (m * K_BLOCKS + kb) * 2);
        let c_za = read_f16(a_params, (m * K_BLOCKS + kb) * 2 + 1);
        let c_sb = read_f32(b_scale, n * K_BLOCKS + kb);
        let c_qs = read_f32(b_qsum, n * K_BLOCKS + kb);
        let mut c_dot: i64 = 0;
        for i in 0..32 {
            let ai = a_i8[m * K + kb * 32 + i] as i8 as i64;
            let bi = b_i8[n * K + kb * 32 + i] as i8 as i64;
            c_dot += ai * bi;
        }
        let c_dot_f = c_dot as f32;
        let c_main = c_dot_f * c_sa * c_sb;
        let c_corr = c_za * c_sb * c_qs;
        cpu_acc += (c_main + c_corr) as f64;
        let div_marker = if (g_sa - c_sa).abs() > 1e-4 * (c_sa.abs() + 1e-6).max(1e-6)
            || (g_za - c_za).abs() > 1e-4 * (c_za.abs() + 1e-6).max(1e-6)
            || (g_sb - c_sb).abs() > 1e-4 * (c_sb.abs() + 1e-6).max(1e-6)
            || (g_qs - c_qs).abs() > 1e-4 * (c_qs.abs() + 1e-6).max(1e-6)
            || (g_dot - c_dot_f).abs() > 0.5
            || (g_acc as f64 - cpu_acc).abs() > 0.01 * cpu_acc.abs().max(1.0)
        {
            if first_div_kb.is_none() {
                first_div_kb = Some(kb);
            }
            " <<<<<"
        } else {
            ""
        };
        eprintln!(
            "  {kb:3} | {g_sa:+.3e} {c_sa:+.3e} | {g_za:+.3e} {c_za:+.3e} | \
             {g_sb:+.3e} {c_sb:+.3e} | {g_qs:+.3e} {c_qs:+.3e} | \
             {g_dot:+8.0} {c_dot_f:+8.0} | {g_main:+.3e} {c_main:+.3e} | \
             {g_corr:+.3e} {c_corr:+.3e} | {g_acc:+.3e} {:+.3e}{div_marker}",
            cpu_acc as f32
        );
    }
    match first_div_kb {
        Some(kb) => eprintln!(
            "[gpu_trace block{block}] FIRST DIVERGENCE at K-block {kb} \
             (final cpu_acc={:+.4e})",
            cpu_acc as f32
        ),
        None => eprintln!(
            "[gpu_trace block{block}] GPU and CPU agree across all K-blocks \
             (final cpu_acc={:+.4e}). If gpu_val still differs, the bug is \
             OUTSIDE matmul_i8 (f16 store, readback, indexing, etc).",
            cpu_acc as f32
        ),
    }
}

/// Print the per-K-block (s, z) pairs for one row of `qkv_attn_in_params_head`
/// so we can see whether any block has an anomalously large z (the
/// asymmetric correction's per-block multiplier).
fn dump_row_act_params(block: usize, seg_name: &str, m: usize, a_params: &[u8]) {
    const K_BLOCKS: usize = 120;
    let n_rows_have = a_params.len() / (K_BLOCKS * 4);
    if m >= n_rows_have {
        eprintln!("[dump_params block{block}.{seg_name}] SKIP m={m}: rows_have={n_rows_have}");
        return;
    }
    let read_f16 = |bytes: &[u8], i: usize| -> f32 {
        let b = i * 2;
        half::f16::from_le_bytes([bytes[b], bytes[b + 1]]).to_f32()
    };
    let mut max_abs_s = 0.0f32;
    let mut max_abs_z = 0.0f32;
    let mut argmax_kb_z = 0usize;
    let mut argmax_kb_s = 0usize;
    for kb in 0..K_BLOCKS {
        let s = read_f16(a_params, (m * K_BLOCKS + kb) * 2);
        let z = read_f16(a_params, (m * K_BLOCKS + kb) * 2 + 1);
        if s.abs() > max_abs_s {
            max_abs_s = s.abs();
            argmax_kb_s = kb;
        }
        if z.abs() > max_abs_z {
            max_abs_z = z.abs();
            argmax_kb_z = kb;
        }
    }
    let s_at_kbz = read_f16(a_params, (m * K_BLOCKS + argmax_kb_z) * 2);
    let z_at_kbs = read_f16(a_params, (m * K_BLOCKS + argmax_kb_s) * 2 + 1);
    eprintln!(
        "[dump_params block{block}.{seg_name}] row m={m}: \
         max|s|={max_abs_s:.4e}@kb={argmax_kb_s}(z_there={z_at_kbs:+.4e}) \
         max|z|={max_abs_z:.4e}@kb={argmax_kb_z}(s_there={s_at_kbz:+.4e})"
    );
}

/// Compare engine's pre-quant QKV-matmul input at row `m` against pyref's
/// `modulated_attn_in` row `m`. Also reconstructs the dequantized i8 row
/// (data * sa + za per K-block) and compares THAT against pyref, to
/// distinguish "input itself wrong" vs "i8 quantization lost info".
///
/// Three outcomes:
///   - engine_pre ≈ py AND dequant_i8 ≈ py → input + quant both fine; bug
///     is in the matmul reconstruction (cancellation, or formula).
///   - engine_pre ≈ py AND dequant_i8 ≠ py → i8 quantization of row m lost
///     enough info that asymmetric Q8_1 cannot recover it at this row.
///   - engine_pre ≠ py → bug is UPSTREAM of the QKV matmul; walk back to
///     mod_attn / attn_norm1 / resid_in.
#[allow(clippy::too_many_arguments)]
fn compare_input_row_against_pyref(
    block: usize,
    seg_name: &str,
    m: usize,
    rows: usize,
    tmp_dir: &Path,
    a_i8: &[u8],
    a_params: &[u8],
) {
    const K: usize = 3840;
    const K_BLOCKS: usize = 120;
    let py_path = tmp_dir.join(format!("py_09_main_block{block}_modulated_attn_in.bin"));
    if !py_path.exists() {
        eprintln!(
            "[input_cmp block{block}.{seg_name}] py missing: {}",
            py_path.display()
        );
        return;
    }
    let py = read_f32(&py_path);
    if py.len() != rows * K {
        eprintln!(
            "[input_cmp block{block}.{seg_name}] py len mismatch: py={} expected rows*K={}",
            py.len(),
            rows * K
        );
        return;
    }
    let py_row = &py[m * K..(m + 1) * K];
    let n_a_rows_have = a_i8.len() / K;
    if m >= n_a_rows_have || a_params.len() < (m + 1) * K_BLOCKS * 4 {
        eprintln!(
            "[input_cmp block{block}.{seg_name}] SKIP m={m}: a_rows_have={n_a_rows_have} \
             a_params_len={}",
            a_params.len()
        );
        return;
    }
    // Reconstruct row m from captured (a_i8, a_params) — exactly the values
    // the matmul kernel ostensibly consumes for row m.
    let read_f16 = |bytes: &[u8], i: usize| -> f32 {
        let b = i * 2;
        half::f16::from_le_bytes([bytes[b], bytes[b + 1]]).to_f32()
    };
    let mut dq_row = vec![0.0f32; K];
    for kb in 0..K_BLOCKS {
        let sa = read_f16(a_params, (m * K_BLOCKS + kb) * 2);
        let za = read_f16(a_params, (m * K_BLOCKS + kb) * 2 + 1);
        for i in 0..32 {
            let ai = a_i8[m * K + kb * 32 + i] as i8 as f32;
            dq_row[kb * 32 + i] = ai * sa + za;
        }
    }
    let (a_dq, _, rmse_dq, _) = linfit(&dq_row, py_row);
    let mut max_diff = 0.0f32;
    let mut max_at = 0usize;
    for i in 0..K {
        let d = (dq_row[i] - py_row[i]).abs();
        if d > max_diff {
            max_diff = d;
            max_at = i;
        }
    }
    // Row-wide magnitude stats so we can compare dq vs py "shape".
    let (dq_mean_abs, dq_max_abs) = row_stats(&dq_row);
    let (py_mean_abs, py_max_abs) = row_stats(py_row);
    eprintln!(
        "[input_cmp block{block}.{seg_name}] row m={m}: \
         dq-vs-py slope={a_dq:.6} rmse={rmse_dq:.4e} max_diff={max_diff:.4e}@k={max_at} | \
         dq(mean_abs={dq_mean_abs:.3e}, max_abs={dq_max_abs:.3e}) \
         py(mean_abs={py_mean_abs:.3e}, max_abs={py_max_abs:.3e})"
    );
    // Top-3 |dq - py| cells with side-by-side values.
    let mut diffs: Vec<(usize, f32)> = (0..K).map(|i| (i, (dq_row[i] - py_row[i]).abs())).collect();
    diffs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (k, _) in diffs.iter().take(3) {
        eprintln!(
            "[input_cmp block{block}.{seg_name}] row m={m} k={k}: \
             dq={dq:+.4e} py={py:+.4e}",
            dq = dq_row[*k],
            py = py_row[*k],
        );
    }
}

fn row_stats(row: &[f32]) -> (f32, f32) {
    let mut sum_abs = 0.0f64;
    let mut max_abs = 0.0f32;
    for &x in row {
        if !x.is_finite() {
            continue;
        }
        let a = x.abs();
        sum_abs += a as f64;
        if a > max_abs {
            max_abs = a;
        }
    }
    ((sum_abs / row.len() as f64) as f32, max_abs)
}

/// Ordinary least-squares fit `y ≈ slope * x + bias` returning
/// `(slope, bias, rmse, n_finite)`. Skips non-finite cells.
pub(crate) fn linfit(x: &[f32], y: &[f32]) -> (f64, f64, f64, usize) {
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut sxx = 0.0f64;
    let mut sxy = 0.0f64;
    let mut cnt = 0usize;
    let n = x.len().min(y.len());
    for i in 0..n {
        let xi = x[i] as f64;
        let yi = y[i] as f64;
        if xi.is_finite() && yi.is_finite() {
            sx += xi;
            sy += yi;
            sxx += xi * xi;
            sxy += xi * yi;
            cnt += 1;
        }
    }
    if cnt < 3 {
        return (0.0, 0.0, 0.0, cnt);
    }
    let nf = cnt as f64;
    let denom = nf * sxx - sx * sx;
    if denom.abs() < 1e-18 {
        return (0.0, 0.0, 0.0, cnt);
    }
    let slope = (nf * sxy - sx * sy) / denom;
    let bias = (sy - slope * sx) / nf;
    let mut resid_sq = 0.0f64;
    for i in 0..n {
        let xi = x[i] as f64;
        let yi = y[i] as f64;
        if xi.is_finite() && yi.is_finite() {
            let r = yi - (slope * xi + bias);
            resid_sq += r * r;
        }
    }
    let rmse = (resid_sq / nf).sqrt();
    (slope, bias, rmse, cnt)
}

fn audit_block26_matmul_i8(damage_zone: &[(usize, Block0LocalTaps)]) {
    let Some(taps) = damage_zone
        .iter()
        .find_map(|(b, t)| (*b == 26).then_some(t))
    else {
        eprintln!("[B26-AUDIT] skipped: block 26 not in damage_zone");
        return;
    };
    if taps.qkv_attn_in_data_head.is_empty()
        || taps.qkv_attn_in_params_head.is_empty()
        || taps.qkv_b_i8_head.is_empty()
        || taps.qkv_b_scale_head.is_empty()
        || taps.qkv_b_qsum_head.is_empty()
        || taps.attn_qkv_f16_pre_quant.is_empty()
    {
        eprintln!(
            "[B26-AUDIT] skipped: byte heads missing \
             (a_i8={} a_p={} b_i8={} b_sc={} b_qs={} out={})",
            taps.qkv_attn_in_data_head.len(),
            taps.qkv_attn_in_params_head.len(),
            taps.qkv_b_i8_head.len(),
            taps.qkv_b_scale_head.len(),
            taps.qkv_b_qsum_head.len(),
            taps.attn_qkv_f16_pre_quant.len(),
        );
        return;
    }

    // Z-Image config: dim=3840 (K), n_qkv=11520 (N), 120 K-blocks of 32.
    const K: usize = 3840;
    const K_BLOCKS: usize = K / 32;
    const N_QKV: usize = 11520;

    let a_i8 = &taps.qkv_attn_in_data_head;
    let a_params = &taps.qkv_attn_in_params_head;
    let b_i8 = &taps.qkv_b_i8_head;
    let b_scale = &taps.qkv_b_scale_head;
    let b_qsum = &taps.qkv_b_qsum_head;
    let out_f32 = &taps.attn_qkv_f16_pre_quant;

    let read_f16 = |bytes: &[u8], i: usize| -> f32 {
        let b = i * 2;
        half::f16::from_le_bytes([bytes[b], bytes[b + 1]]).to_f32()
    };
    let read_f32 = |bytes: &[u8], i: usize| -> f32 {
        let b = i * 4;
        f32::from_le_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]])
    };

    eprintln!("---- B26 matmul_i8 byte-level audit ----");
    eprintln!(
        "  bytes: a_i8={} a_params={} b_i8={} b_scale={} b_qsum={} out_f32_len={}",
        a_i8.len(),
        a_params.len(),
        b_i8.len(),
        b_scale.len(),
        b_qsum.len(),
        out_f32.len()
    );
    // Head value sanity print: confirms byte heads aren't all-zero / stale.
    eprint!("  a_params[m=0,kb=0..4] (s,z) pairs:");
    for kb in 0..4 {
        let s = read_f16(a_params, kb * 2);
        let z = read_f16(a_params, kb * 2 + 1);
        eprint!(" ({s:+.4e},{z:+.4e})");
    }
    eprintln!();
    eprint!("  b_scale[n=0,kb=0..4]:");
    for kb in 0..4 {
        eprint!(" {:+.4e}", read_f32(b_scale, kb));
    }
    eprintln!();
    eprint!("  b_qsum[n=0,kb=0..4]:");
    for kb in 0..4 {
        eprint!(" {:+.4e}", read_f32(b_qsum, kb));
    }
    eprintln!();
    // CPU-recompute b_qsum from b_i8 to verify dequant_i8 is writing what
    // we expect. If these disagree with the GPU's b_qsum dump, the bug is
    // in dequant_i8's b_qsum tree reduce.
    eprint!("  CPU b_qsum recompute[n=0,kb=0..4]:");
    for kb in 0..4 {
        let mut s: i32 = 0;
        for &b in &b_i8[kb * 32..kb * 32 + 32] {
            s += b as i8 as i32;
        }
        eprint!(" {:+.4e}", s as f32);
    }
    eprintln!();
    // CPU-recompute b_qsum for all 4 captured rows at kb=0.
    eprint!("  CPU b_qsum recompute[n=0..3, kb=0]:");
    for n in 0..4 {
        let mut s: i32 = 0;
        for &b in &b_i8[n * K..n * K + 32] {
            s += b as i8 as i32;
        }
        eprint!(" {:+.4e}", s as f32);
    }
    eprintln!();
    // Raw i8 head dumps for the first row of acts and weights.
    eprint!("  a_i8[m=0, k=0..16]:");
    for &a in &a_i8[..16] {
        eprint!(" {:+4}", a as i8);
    }
    eprintln!();
    eprint!("  b_i8[n=0, k=0..16]:");
    for &b in &b_i8[..16] {
        eprint!(" {:+4}", b as i8);
    }
    eprintln!();
    // Sanity: print last 16 of a_i8[m=0] in case the row is short.
    eprint!("  a_i8[m=0, k=K-16..K]:");
    for i in 0..16 {
        eprint!(" {:+4}", a_i8[K - 16 + i] as i8);
    }
    eprintln!();

    for &(m, n) in &[(0usize, 0usize), (0, 1), (1, 0), (1, 1), (2, 2)] {
        if (m + 1) * K > a_i8.len() || (n + 1) * K > b_i8.len() {
            eprintln!("[B26-AUDIT cell({m},{n})] skipped: byte heads too short");
            continue;
        }
        let mut acc_main = 0.0f64;
        let mut acc_corr = 0.0f64;
        for kb in 0..K_BLOCKS {
            let mut dot: i32 = 0;
            for i in 0..32 {
                let ai = a_i8[m * K + kb * 32 + i] as i8 as i32;
                let bi = b_i8[n * K + kb * 32 + i] as i8 as i32;
                dot += ai * bi;
            }
            let sa = read_f16(a_params, (m * K_BLOCKS + kb) * 2);
            let za = read_f16(a_params, (m * K_BLOCKS + kb) * 2 + 1);
            let sb = read_f32(b_scale, n * K_BLOCKS + kb);
            let bq = read_f32(b_qsum, n * K_BLOCKS + kb);
            acc_main += (dot as f64) * (sa as f64) * (sb as f64);
            acc_corr += (za as f64) * (sb as f64) * (bq as f64);
        }
        let cpu_acc = (acc_main + acc_corr) as f32;
        let gpu = out_f32[m * N_QKV + n];
        let abs_diff = cpu_acc - gpu;
        let rel_diff = if gpu.abs() > 1e-6 {
            abs_diff / gpu
        } else {
            0.0
        };
        eprintln!(
            "[B26-AUDIT cell({m},{n})] cpu={cpu_acc:+.6e} gpu={gpu:+.6e} \
             abs_diff={abs_diff:+.4e} rel_diff={rel_diff:+.4e} \
             main={:+.6e} corr={:+.6e} corr/main={:+.4e}",
            acc_main as f32,
            acc_corr as f32,
            if acc_main.abs() > 1e-12 {
                (acc_corr / acc_main) as f32
            } else {
                0.0
            },
        );
    }
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
