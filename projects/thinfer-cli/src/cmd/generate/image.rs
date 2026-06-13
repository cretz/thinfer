//! `thinfer generate image` (Z-Image-Turbo t2i).

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Args, ValueEnum};
use thinfer_core::backend::WgpuBackend;
use thinfer_core::manifest::{FileRef, ModelManifest};
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::tokenizer::Tokenizer;
use thinfer_core::trace::DIAG;
use thinfer_core::weight::WeightSource;
use thinfer_models::z_image::manifest::{VariantFiles, role as zrole};
use thinfer_models::z_image::pipeline::{GenerationParams, ProgressEvent, ProgressFn, ZImageModel};
use thinfer_models::z_image::source::{GgufOpeners, ZImageSource};
use thinfer_native::tokenizer::HfTokenizer;
use thinfer_native::{MmapFileOpener, cache};

use super::{
    PercentLogger, backend_for_stats, confirm_download, init_backend, parse_budget, random_seed,
    report_mem, resolve_output_format, validate_dim,
};

/// Output container/codec. PNG-only today (the only encoder we ship; every
/// parity baseline is PNG). Adding a second format is a one-arm change here
/// plus the matching encoder call in `run_image`.
#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum ImageFormat {
    Png,
}

impl ImageFormat {
    /// Lower-cased file extension -> format. `None` is "unknown extension", a
    /// hard error in `resolve_output_format` (never a silent default).
    fn from_ext(ext: &str) -> Option<Self> {
        match ext {
            "png" => Some(Self::Png),
            _ => None,
        }
    }
    const KNOWN: &'static str = "png";
}

/// Defaults follow upstream Z-Image (`Tongyi-MAI/Z-Image:src/config/inference.py`):
/// 8 inference steps, guidance_scale=0 (Turbo is a no-CFG model). Dims
/// intentionally diverge: upstream defaults to 1024x1024 assuming datacenter
/// GPUs; we default to 768x768 as the thin-hardware sweet spot (every parity
/// and perf baseline is at 768). Seed defaults to randomized; HF Space uses
/// 42, upstream pipeline takes a generator and we randomize when `--seed` is
/// omitted.
///
/// TODO: these defaults are Z-Image-specific. Once we add a second image model,
/// move height/width/steps/guidance defaults into the model registry
/// (`ModelId::defaults() -> ImageGenDefaults`) and have clap pull from there
/// instead of hardcoded constants on the args struct.
#[derive(Args)]
pub struct GenerateImage {
    /// Model identifier. Defaults to `zimage-turbo-q4` (Q4_K_M DiT: ~half
    /// the VRAM/bandwidth of Q8_0 at visually-confirmed-acceptable quality).
    #[arg(long, default_value_t = ModelId::ZImageTurboQ4, value_enum)]
    pub model: ModelId,
    #[arg(long)]
    pub prompt: String,
    #[arg(long)]
    pub output: PathBuf,
    /// Output format. Defaults to inferring from the `--output` extension;
    /// errors if the extension is missing or unrecognized.
    #[arg(long, value_enum)]
    pub output_format: Option<ImageFormat>,
    /// Image height in pixels. Must be divisible by VAE_SCALE (16).
    #[arg(long, default_value_t = 768)]
    pub height: u32,
    /// Image width in pixels. Must be divisible by VAE_SCALE (16).
    #[arg(long, default_value_t = 768)]
    pub width: u32,
    /// Inference steps. Upstream default is 8 (Turbo).
    #[arg(long, default_value_t = 8)]
    pub steps: u32,
    /// Seed. Omit for a randomized seed.
    #[arg(long)]
    pub seed: Option<u64>,
    /// Host RAM budget for the weight residency manager. e.g. `8G`, `512M`,
    /// raw bytes.
    #[arg(long)]
    pub ram_budget: Option<String>,
    /// GPU VRAM budget for the weight residency manager.
    #[arg(long)]
    pub vram_budget: Option<String>,
    /// Skip the TTY consent prompt and download missing weight files.
    #[arg(long, default_value_t = false)]
    pub download_as_needed: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum ModelId {
    /// Z-Image-Turbo with Q8_0 DiT matmul weights (unsloth GGUF). Rest of
    /// the model stays bf16 safetensors.
    #[value(name = "zimage-turbo-q8")]
    ZImageTurboQ8,
    /// Z-Image-Turbo with Q4_K_M DiT matmul weights (unsloth GGUF). Halves
    /// DiT VRAM/bandwidth vs Q8_0 at production-grade quality. Rest of the
    /// model stays bf16 safetensors. (Q4_K_M is Q4_K + Q6_K mixed; engine
    /// support in progress.)
    #[value(name = "zimage-turbo-q4")]
    ZImageTurboQ4,
    /// Z-Image-Turbo with bf16 DiT weights (dimitribarbot safetensors).
    #[value(name = "zimage-turbo-bf16")]
    ZImageTurboBf16,
}

impl ModelId {
    fn manifest(self) -> &'static ModelManifest {
        match self {
            ModelId::ZImageTurboQ8 | ModelId::ZImageTurboQ4 | ModelId::ZImageTurboBf16 => {
                &thinfer_models::z_image::manifest::MANIFEST
            }
        }
    }

    /// File set from the shared variant registry (single source of truth
    /// with web; keyed by the same id strings clap displays).
    fn variant(self) -> &'static VariantFiles {
        thinfer_models::z_image::manifest::variant(&self.to_string())
            .expect("CLI ModelId missing from VARIANTS registry")
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelId::ZImageTurboQ8 => f.write_str("zimage-turbo-q8"),
            ModelId::ZImageTurboQ4 => f.write_str("zimage-turbo-q4"),
            ModelId::ZImageTurboBf16 => f.write_str("zimage-turbo-bf16"),
        }
    }
}

pub async fn run_image(args: GenerateImage) -> Result<(), String> {
    validate_dim("height", args.height)?;
    validate_dim("width", args.width)?;
    if args.steps == 0 {
        return Err("--steps must be > 0".into());
    }
    // Resolve up front so a bad extension fails before any download / GPU work.
    let ImageFormat::Png = resolve_output_format(
        args.output_format,
        &args.output,
        ImageFormat::from_ext,
        ImageFormat::KNOWN,
    )?;

    let ram_bytes = parse_budget("--ram-budget", args.ram_budget.as_deref())?;
    let vram_bytes = parse_budget("--vram-budget", args.vram_budget.as_deref())?;
    let budget = ResidencyBudget {
        ram_bytes,
        vram_bytes,
    };

    let manifest = args.model.manifest();
    let variant = args.model.variant();
    let dit_gguf_role = variant.dit_gguf_role;
    let all_files: Vec<FileRef> = variant.files().map(|(_, f)| *f).collect();
    let (_resolved, missing) = cache::resolve_all(all_files.iter());

    if !missing.is_empty()
        && !confirm_download(&missing, args.download_as_needed).map_err(|e| e.to_string())?
    {
        return Err("declined download; rerun with --download-as-needed or `hf download …`".into());
    }
    for f in &missing {
        let progress = std::sync::Arc::new(PercentLogger::new(format!("{}/{}", f.repo, f.path)));
        cache::download_with_progress(f, Some(progress))
            .await
            .map_err(|e| format!("{e:?}"))?;
    }

    let resolve_role = |role: &str| -> Result<std::path::PathBuf, String> {
        let r = manifest
            .get(role)
            .ok_or_else(|| format!("manifest missing role {role}"))?;
        cache::resolve(r)
            .ok_or_else(|| format!("{}/{} not in cache after download", r.repo, r.path))
    };

    let mut weight_openers: Vec<MmapFileOpener> = Vec::with_capacity(variant.weight_roles.len());
    for role in variant.weight_roles {
        let path = resolve_role(role)?;
        weight_openers.push(
            MmapFileOpener::new(&path)
                .await
                .map_err(|e| format!("open {}: {e}", path.display()))?,
        );
    }
    let mut open_gguf = Vec::with_capacity(2);
    for role in [dit_gguf_role, variant.te_gguf_role].into_iter().flatten() {
        let path = resolve_role(role)?;
        open_gguf.push(
            MmapFileOpener::new(&path)
                .await
                .map_err(|e| format!("open gguf {}: {e}", path.display()))?,
        );
    }
    let gguf_openers = match (open_gguf.pop(), open_gguf.pop()) {
        (Some(te), Some(dit)) => Some(GgufOpeners { dit, te }),
        (None, None) => None,
        _ => return Err("variant must set both gguf roles or neither".into()),
    };
    // Schema adapters + optional GGUF union live in `ZImageSource::open`,
    // shared with web and the e2e tests.
    let source = ZImageSource::open(weight_openers, gguf_openers)
        .await
        .map_err(|e| format!("parse weight files: {e:?}"))?;

    let tokenizer_path = resolve_role(zrole::TOKENIZER_JSON)?;
    let tokenizer = HfTokenizer::from_path(&tokenizer_path)
        .await
        .map_err(|e| format!("tokenizer {}: {e:?}", tokenizer_path.display()))?;

    tracing::info!(
        target: DIAG,
        model = %args.model,
        prompt = %args.prompt,
        width = args.width,
        height = args.height,
        steps = args.steps,
        seed = ?args.seed,
        ram_budget = ram_bytes,
        vram_budget = vram_bytes,
        "generate start",
    );

    let backend = init_backend().await?;
    let stats = backend_for_stats(&backend);

    let seed = args.seed.unwrap_or_else(random_seed);
    // User-facing progress: capitalized one-liners to stderr, each prefixed
    // with elapsed-from-start so per-stage durations read off directly.
    // Timer starts here (post-download, pre-GPU-init) so it reflects
    // generation work, not network.
    let t_run = std::time::Instant::now();
    let stamp = move || format!("[{:6.1}s]", t_run.elapsed().as_secs_f64());
    eprintln!(
        "{} Generating {}x{} image, {} steps, seed {} ({})",
        stamp(),
        args.width,
        args.height,
        args.steps,
        seed,
        args.model,
    );
    let progress = move |ev: ProgressEvent| match ev {
        ProgressEvent::TextEncode => eprintln!("{} Encoding prompt", stamp()),
        ProgressEvent::Step { i, n } => eprintln!("{} Denoising step {i}/{n}", stamp()),
        ProgressEvent::VaeDecode => eprintln!("{} Decoding latents (VAE)", stamp()),
    };
    let gen_params = GenerationParams {
        prompt: args.prompt,
        height: args.height,
        width: args.width,
        steps: args.steps,
        seed,
    };
    let residency = WeightResidency::new(source, budget);
    let png =
        load_and_generate(backend, residency, tokenizer, &gen_params, Some(&progress)).await?;
    tokio::fs::write(&args.output, &png)
        .await
        .map_err(|e| format!("write {}: {e}", args.output.display()))?;
    eprintln!(
        "{} Wrote {} ({}x{}, seed {}) in {:.1}s",
        stamp(),
        args.output.display(),
        args.width,
        args.height,
        seed,
        t_run.elapsed().as_secs_f64(),
    );
    tracing::info!(target: DIAG, path = %args.output.display(), bytes = png.len(), "wrote output");
    if let Some(b) = stats {
        report_mem(&b, ram_bytes, vram_bytes);
    }
    Ok(())
}

async fn load_and_generate<S: WeightSource, T: Tokenizer>(
    backend: Arc<WgpuBackend>,
    residency: WeightResidency<S>,
    tokenizer: T,
    params: &GenerationParams,
    progress: ProgressFn<'_>,
) -> Result<Vec<u8>, String> {
    let model = {
        let _s = tracing::info_span!("model_load").entered();
        ZImageModel::load(backend, residency, tokenizer)
            .await
            .map_err(|e| format!("model load: {e:?}"))?
    };
    model
        .generate(params, progress)
        .await
        .map_err(|e| format!("generate: {e:?}"))
}
