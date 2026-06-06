use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::{Args, Subcommand, ValueEnum};
use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::format::safetensors::ShardedSafetensorsSource;
use thinfer_core::format::union::{
    QuantOnlySource, RenamedSource, SplitToFusedQkvSource, UnionSource,
};
use thinfer_core::manifest::{FileRef, ModelManifest};
use thinfer_core::policy::{ResidencyBudget, parse_bytes};
use thinfer_core::residency::WeightResidency;
use thinfer_core::tokenizer::Tokenizer;
use thinfer_core::trace::DIAG;
use thinfer_core::weight::WeightSource;
use thinfer_models::z_image::manifest::role as zrole;
use thinfer_models::z_image::pipeline::{GenerationParams, ProgressEvent, ProgressFn, ZImageModel};
use thinfer_native::tokenizer::HfTokenizer;
use thinfer_native::{MmapFileOpener, cache};

/// 2 GiB default for both RAM and VRAM. Chosen so a low-spec laptop can run
/// at all; larger budgets help, but the residency manager pages weights so a
/// small budget just means more disk traffic, not failure. Override with
/// `--ram-budget` / `--vram-budget`.
const DEFAULT_BUDGET_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Subcommand)]
pub enum GenerateCmd {
    /// Generate an image from a prompt.
    Image(GenerateImage),
}

/// Defaults follow upstream Z-Image (`Tongyi-MAI/Z-Image:src/config/inference.py`):
/// 8 inference steps, guidance_scale=0 (Turbo is a no-CFG model). Dims
/// intentionally diverge: upstream defaults to 1024x1024 assuming datacenter
/// GPUs; we default to 768x768 as the thin-hardware sweet spot (every parity
/// and perf baseline is at 768). Seed defaults to randomized; HF Space uses
/// 42, upstream pipeline takes a generator and we randomize when `--seed` is
/// omitted.
///
/// TODO: these defaults are Z-Image-specific. Once we add a second model
/// (LTX-Video per M3), move height/width/steps/guidance defaults into the
/// model registry (`ModelId::defaults() -> ImageGenDefaults`) and have clap
/// pull from there instead of hardcoded constants on the args struct.
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

    /// Role of the GGUF file to union over the safetensors source, if any.
    fn dit_gguf_role(self) -> Option<&'static str> {
        match self {
            ModelId::ZImageTurboQ8 => Some(zrole::DIT_GGUF_Q8_0),
            ModelId::ZImageTurboQ4 => Some(zrole::DIT_GGUF_Q4_K_M),
            ModelId::ZImageTurboBf16 => None,
        }
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

pub async fn run(cmd: GenerateCmd) -> ExitCode {
    match cmd {
        GenerateCmd::Image(args) => match run_image(args).await {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(1)
            }
        },
    }
}

async fn run_image(args: GenerateImage) -> Result<(), String> {
    validate_dim("height", args.height)?;
    validate_dim("width", args.width)?;
    if args.steps == 0 {
        return Err("--steps must be > 0".into());
    }

    let ram_bytes = parse_budget("--ram-budget", args.ram_budget.as_deref())?;
    let vram_bytes = parse_budget("--vram-budget", args.vram_budget.as_deref())?;
    let budget = ResidencyBudget {
        ram_bytes,
        vram_bytes,
    };

    let manifest = args.model.manifest();
    let dit_gguf_role = args.model.dit_gguf_role();
    // Weight shards (safetensors, merged into one ShardedSafetensorsSource)
    // and tokenizer JSON live in the manifest. As future loaders land they
    // join the relevant list here; nothing else in CLI needs to change.
    let weight_roles: &[&str] = &[
        zrole::DIT_SHARD_1,
        zrole::DIT_SHARD_2,
        zrole::TEXT_ENCODER_SHARD_1,
        zrole::TEXT_ENCODER_SHARD_2,
        zrole::TEXT_ENCODER_SHARD_3,
        zrole::VAE,
    ];
    let aux_roles: &[&str] = &[zrole::TOKENIZER_JSON];
    let all_files: Vec<FileRef> = weight_roles
        .iter()
        .chain(aux_roles.iter())
        .chain(dit_gguf_role.iter())
        .map(|r| {
            *manifest
                .get(r)
                .expect("required role missing from model manifest")
        })
        .collect();
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

    let mut weight_openers: Vec<MmapFileOpener> = Vec::with_capacity(weight_roles.len());
    for role in weight_roles {
        let path = resolve_role(role)?;
        weight_openers.push(
            MmapFileOpener::new(&path)
                .await
                .map_err(|e| format!("open {}: {e}", path.display()))?,
        );
    }
    let source = ShardedSafetensorsSource::open(weight_openers)
        .await
        .map_err(|e| format!("parse sharded safetensors: {e:?}"))?;
    // Z-Image canonical schema is fused `attention.qkv.weight`. Checkpoints
    // that ship split `to_q`/`to_k`/`to_v` (dimitribarbot) flow through this
    // adapter; checkpoints with a fused entry already see the adapter as a
    // passthrough.
    let source = SplitToFusedQkvSource::new(source, thinfer_models::z_image::dit_qkv_triples());
    // dimitribarbot publishes `attention.to_out.0.weight`; engine asks for
    // canonical `attention.out.weight` (matches unsloth GGUF schema).
    let source =
        RenamedSource::with_passthrough(source, thinfer_models::z_image::dit_to_out_renames());

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

    let backend = {
        let _s = tracing::info_span!("wgpu_init").entered();
        // Default `HighPerformance` (not `None`) because Vulkan drivers treat
        // an unset preference as a background-priority hint: on Intel Arc
        // iGPU this clamps clocks / shrinks the subgroup_size range, slowing
        // DiT by ~2.5x. Users who explicitly want thin-hardware-friendly
        // scheduling can set `THINFER_POWER_PREF=low`.
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
        Arc::new(
            WgpuBackend::new_with_config(cfg)
                .await
                .map_err(|e| format!("wgpu init: {e:?}"))?,
        )
    };
    // Clone for end-of-run stats reporting; ZImageModel::load takes
    // ownership of one Arc, we keep the other to read mem snapshots when
    // THINFER_TRACE is set (same gate that enables the rollup table in
    // main.rs, so a single env var turns on the full report).
    let backend_for_stats = std::env::var_os("THINFER_TRACE").map(|_| Arc::clone(&backend));

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
    let png = match dit_gguf_role {
        None => {
            let residency = WeightResidency::new(source, budget);
            load_and_generate(backend, residency, tokenizer, &gen_params, Some(&progress)).await?
        }
        Some(role) => {
            let path = resolve_role(role)?;
            let opener = MmapFileOpener::new(&path)
                .await
                .map_err(|e| format!("open gguf {}: {e}", path.display()))?;
            let gguf = GgufSource::open(opener)
                .await
                .map_err(|e| format!("parse gguf {}: {e:?}", path.display()))?;
            // GGUF (unsloth Z-Image-Turbo) ships upstream canonical names
            // including fused `attention.qkv.weight`; no rename needed.
            // safetensors fallback supplies AdaLN/biases/norms under the
            // same canonical names. unsloth's file Q8-quantizes the
            // main-layer AdaLN modulation weights too, but the engine
            // keeps AdaLN as bf16 (see pipeline.rs::dit_adaln_cfg), so
            // we hide AdaLN ids from the GGUF side to fall through.
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
                source,
            );
            let residency = WeightResidency::new(unioned, budget);
            load_and_generate(backend, residency, tokenizer, &gen_params, Some(&progress)).await?
        }
    };
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
    if let Some(b) = backend_for_stats {
        let snap = thinfer_core::backend::Backend::mem_account(&*b).snapshot();
        eprintln!(
            "[mem] vram TRUE_PEAK={} / budget {} | per-cat peaks: weights={} workspace={} staging={}",
            fmt_mib(snap.vram_total_peak),
            fmt_mib(vram_bytes),
            fmt_mib(snap.vram_weights.1),
            fmt_mib(snap.vram_workspace.1),
            fmt_mib(snap.vram_staging.1),
        );
        eprintln!(
            "[mem] ram  TRUE_PEAK={} / budget {} | per-cat peaks: upload={} readback={} other={}",
            fmt_mib(snap.ram_total_peak),
            fmt_mib(ram_bytes),
            fmt_mib(snap.ram_upload.1),
            fmt_mib(snap.ram_readback.1),
            fmt_mib(snap.ram_other.1),
        );
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

fn fmt_mib(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.2}GiB", bytes as f64 / (1u64 << 30) as f64)
    } else {
        format!("{:.1}MiB", bytes as f64 / (1u64 << 20) as f64)
    }
}

/// Non-cryptographic seed for `--seed`-omitted runs. Mixes nanos and pid so
/// rapid successive invocations don't collide.
fn random_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ (std::process::id() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn parse_budget(flag: &str, raw: Option<&str>) -> Result<u64, String> {
    match raw {
        Some(s) => parse_bytes(s).map_err(|e| format!("{flag}={s:?}: {e}")),
        None => Ok(DEFAULT_BUDGET_BYTES),
    }
}

fn validate_dim(name: &str, v: u32) -> Result<(), String> {
    // Upstream `pipeline.py` requires divisibility by vae_scale = vae_factor*2
    // (16 for the default VAE). No min/max bound upstream; the HF Space caps
    // at 512..=2048 for UX only.
    const VAE_SCALE: u32 = 16;
    if v == 0 {
        return Err(format!("--{name} must be > 0"));
    }
    if !v.is_multiple_of(VAE_SCALE) {
        return Err(format!(
            "--{name} must be a multiple of {VAE_SCALE} (got {v})"
        ));
    }
    Ok(())
}

/// Emits a stderr line at each 10% boundary. hf-hub fans chunks across tasks
/// and the adapter clones the `Arc` per chunk, so `update` calls are racy -
/// state lives in atomics.
struct PercentLogger {
    name: String,
    size: std::sync::atomic::AtomicU64,
    downloaded: std::sync::atomic::AtomicU64,
    last_decile: std::sync::atomic::AtomicU8,
}

impl PercentLogger {
    fn new(name: String) -> Self {
        Self {
            name,
            size: std::sync::atomic::AtomicU64::new(0),
            downloaded: std::sync::atomic::AtomicU64::new(0),
            last_decile: std::sync::atomic::AtomicU8::new(0),
        }
    }
}

impl cache::DownloadProgress for PercentLogger {
    fn init(&self, size: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.size.store(size, Relaxed);
        self.downloaded.store(0, Relaxed);
        self.last_decile.store(0, Relaxed);
        tracing::info!(
            target: DIAG,
            name = %self.name,
            gib = size as f64 / (1024.0 * 1024.0 * 1024.0),
            "downloading",
        );
    }
    fn update(&self, delta: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let size = self.size.load(Relaxed);
        if size == 0 {
            return;
        }
        let new = self.downloaded.fetch_add(delta, Relaxed) + delta;
        let pct = ((new.min(size) * 10) / size) as u8;
        let prev = self.last_decile.fetch_max(pct, Relaxed);
        if pct > prev {
            // Cover gaps when a single chunk crosses several deciles, and when
            // hf-hub's resume update jumps from 0 to N0% in one call.
            for p in (prev + 1)..=pct {
                tracing::info!(target: DIAG, name = %self.name, pct = p * 10, "download progress");
            }
        }
    }
    fn finish(&self) {
        tracing::info!(target: DIAG, name = %self.name, "download done");
    }
}

fn confirm_download(missing: &[FileRef], download_as_needed: bool) -> io::Result<bool> {
    eprintln!("{} file(s) not in HF cache:", missing.len());
    for f in missing {
        eprintln!("  {}/{}", f.repo, f.path);
    }
    if download_as_needed {
        return Ok(true);
    }
    let stdin = io::stdin();
    if !stdin.is_terminal() {
        eprintln!("non-interactive: use --download-as-needed to proceed");
        return Ok(false);
    }
    eprint!("download now? [y/N] ");
    io::stderr().flush()?;
    let mut line = String::new();
    stdin.lock().read_line(&mut line)?;
    Ok(matches!(line.trim(), "y" | "Y" | "yes" | "Yes"))
}
