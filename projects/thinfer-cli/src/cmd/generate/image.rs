//! `thinfer generate image` (Z-Image-Turbo t2i). Thin clap adapter over
//! `thinfer_app::ImageRequest`.

use std::path::PathBuf;

use clap::Args;
use thinfer_app::config::ResidencyBudget;
use thinfer_app::model::{
    IMAGE_DEFAULT_HEIGHT, IMAGE_DEFAULT_STEPS, IMAGE_DEFAULT_WIDTH, ImageModelId,
};
use thinfer_app::request::{ImageFormat, ImageRequest, LoraRef, Secret};
use thinfer_app::vault::{self, Vault};
use thinfer_app::wire::{ImageSpec, JobSpec};
use thinfer_app::{JobRequest, parse_budget, resolve_output_format};

#[derive(Args)]
pub struct GenerateImage {
    /// Model identifier. Defaults to `zimage-turbo-q4` (Q4_K_M DiT: ~half the
    /// VRAM/bandwidth of Q8_0 at visually-confirmed-acceptable quality).
    #[arg(long, default_value_t = ImageModelId::DEFAULT, value_enum)]
    pub model: ImageModelId,
    #[arg(long)]
    pub prompt: String,
    /// Reference image to edit (REQUIRED for `qwen-image-edit-rapid`; rejected
    /// for the t2i models). PNG/JPEG.
    #[arg(long)]
    pub input_image: Option<PathBuf>,
    #[arg(long)]
    pub output: PathBuf,
    /// Output format. Defaults to inferring from the `--output` extension;
    /// errors if the extension is missing or unrecognized.
    #[arg(long, value_enum)]
    pub output_format: Option<ImageFormat>,
    /// Image height in pixels. Must be divisible by VAE_SCALE (16).
    #[arg(long, default_value_t = IMAGE_DEFAULT_HEIGHT)]
    pub height: u32,
    /// Image width in pixels. Must be divisible by VAE_SCALE (16).
    #[arg(long, default_value_t = IMAGE_DEFAULT_WIDTH)]
    pub width: u32,
    /// Inference steps. Upstream default is 8 (Turbo).
    #[arg(long, default_value_t = IMAGE_DEFAULT_STEPS)]
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
    /// Disable the DP4A i8 matmul on the DP4A-safe DiT sites (Ideogram-4 only;
    /// forces the bf16 reference path). No effect on Z-Image.
    #[arg(long)]
    pub no_i8_matmul: bool,
    /// Fold a stored adapter into the DiT, as `NAME_OR_ID[:WEIGHT]` (repeatable,
    /// applied in order). Resolved against the vault for `--model`; needs the
    /// vault password (hidden prompt, or `THINFER_VAULT_PASSWORD`). Local runs
    /// only (not `--remote`); only models that support adapters accept it.
    #[arg(long = "lora", value_name = "NAME[:WEIGHT]")]
    pub lora: Vec<String>,
    /// Vault directory for `--lora`. Defaults to the shared location
    /// (`THINFER_VAULT_DIR`, else `<hf-cache>/vault`).
    #[arg(long)]
    pub vault_dir: Option<PathBuf>,
    /// Skip the TTY consent prompt and download missing weight files.
    #[arg(long, default_value_t = false)]
    pub download_as_needed: bool,
    #[command(flatten)]
    pub remote: super::RemoteArgs,
}

pub async fn run_image(args: GenerateImage) -> Result<(), String> {
    // Resolve up front so a bad extension fails before any download / GPU work.
    let format = resolve_output_format(
        args.output_format,
        &args.output,
        ImageFormat::from_ext,
        ImageFormat::KNOWN,
    )?;

    if args.remote.remote.is_some() {
        if args.input_image.is_some() {
            return Err("--input-image (image edit) is not supported over --remote yet".into());
        }
        if !args.lora.is_empty() {
            return Err(
                "--lora (adapters) is not supported over --remote; run locally or use the web UI"
                    .into(),
            );
        }
        let spec = JobSpec::Image(ImageSpec {
            model: Some(args.model),
            prompt: args.prompt,
            width: Some(args.width),
            height: Some(args.height),
            steps: Some(args.steps),
            seed: args.seed,
            // Edit-over-remote is guarded above (returns early), so no image here.
            input_image: None,
            i8_matmul: Some(!args.no_i8_matmul),
            public_key: None,
            // Remote path defers coopmat to the server default; local runs read
            // THINFER_NO_COOPMAT via BackendConfig.
            disable_coopmat: None,
            // Adapters over --remote are guarded above.
            lora: Vec::new(),
            password: None,
        });
        return super::run_remote(&args.remote, spec, args.output).await;
    }
    let ram = parse_budget("--ram-budget", args.ram_budget.as_deref())?;
    let vram = parse_budget("--vram-budget", args.vram_budget.as_deref())?;

    // Resolve any --lora names/ids against the vault (needs the password).
    let (lora, vault_password) = if args.lora.is_empty() {
        (Vec::new(), None)
    } else {
        let (refs, password) = resolve_loras(args.model, args.vault_dir.as_deref(), &args.lora)?;
        (refs, Some(password))
    };

    let req = ImageRequest {
        model: args.model,
        prompt: args.prompt,
        width: args.width,
        height: args.height,
        steps: args.steps,
        seed: args.seed,
        i8_matmul: !args.no_i8_matmul,
        input_image: args.input_image,
        lora,
        vault_password,
        vault_dir: vault::resolve_dir(args.vault_dir.as_deref()),
        budget: ResidencyBudget {
            ram_bytes: ram,
            vram_bytes: vram,
        },
        output: args.output,
        format,
    };
    req.validate()?;
    let files = req.required_files();
    super::run_job(
        JobRequest::Image(req),
        &files,
        args.download_as_needed,
        ram,
        vram,
    )
    .await
}

/// Parse a `--lora` value: `NAME_OR_ID` optionally suffixed with `:WEIGHT`. The
/// suffix is only a weight if it parses as a float (ids/names never contain a
/// trailing `:float`), so a bare name is unambiguous.
fn parse_lora_arg(s: &str) -> (&str, Option<f32>) {
    if let Some((key, w)) = s.rsplit_once(':')
        && let Ok(weight) = w.parse::<f32>()
    {
        return (key, Some(weight));
    }
    (s, None)
}

/// Resolve `--lora` values to vault entry ids by matching each against the
/// model's stored adapters (by id or decrypted name). Reads the vault password
/// (prompt / env). The weight is the explicit `:WEIGHT`, else the adapter's
/// stored suggestion, else 1.0.
fn resolve_loras(
    model: ImageModelId,
    vault_dir: Option<&std::path::Path>,
    specs: &[String],
) -> Result<(Vec<LoraRef>, Secret), String> {
    let password = crate::cmd::vault::read_password(false)?;
    let vault = Vault::new(vault::resolve_dir(vault_dir));
    let entries = vault
        .list(password.expose(), &model.to_string())
        .map_err(|e| e.to_string())?;
    let mut out = Vec::with_capacity(specs.len());
    for spec in specs {
        let (key, weight) = parse_lora_arg(spec);
        let entry = entries
            .iter()
            .find(|e| e.id == key || e.name == key)
            .ok_or_else(|| {
                format!(
                    "no adapter '{key}' for {model} in the vault \
                     (see `thinfer vault list --model {model}`)"
                )
            })?;
        let weight = weight
            .or_else(|| entry.extra.get("weight").and_then(|w| w.parse().ok()))
            .unwrap_or(1.0);
        out.push(LoraRef {
            id: entry.id.clone(),
            weight,
        });
    }
    Ok((out, password))
}
