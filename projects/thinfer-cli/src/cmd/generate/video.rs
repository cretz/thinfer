//! `thinfer generate video` (FastWan / LongLive t2v). Thin clap adapter over
//! `thinfer_app::VideoRequest`; the shot-plan resolution, denoise, and MP4
//! encode all live in `thinfer-app`.

use std::path::PathBuf;

use clap::Args;
use thinfer_app::config::ResidencyBudget;
use thinfer_app::model::{EncoderQuant, RewriteQuality, VaeChoice, VideoModelId, VideoSampler};
use thinfer_app::request::{ImageBytes, VideoFormat, VideoInput, VideoRequest};
use thinfer_app::wire::{JobSpec, VideoSpec};
use thinfer_app::{JobRequest, parse_budget, resolve_output_format};

/// FastWan is CFG-free (no `--guidance-scale`/`--negative-prompt`). The denoise
/// sampler is selectable: `--sampler unipc` (default, `--steps` configurable,
/// matches the public FastWan Spaces) or `--sampler dmd` (the fixed 3-step
/// byte-parity reference; `--steps` is ignored). Dims must be /32; frames must be
/// 4k+1 (the causal-VAE temporal grid). For a fast first run, pass `--frames 5`.
#[derive(Args)]
pub struct GenerateVideo {
    /// Model identifier. Defaults to `fastwan-ti2v-5b` (a fast 5B DMD distill,
    /// 960x544, no prompt rewrite; the e2e-validated path).
    #[arg(long, default_value_t = VideoModelId::DEFAULT, value_enum)]
    pub model: VideoModelId,
    /// Text prompt. Repeat `--prompt` for a multi-shot video (LongLive only):
    /// each prompt is one shot, with a scene cut between shots. Not used by
    /// `dreamid-v` (the context is baked; pass `--input-video` + `--source-image`
    /// instead). Required for every other video model.
    #[arg(long)]
    pub prompt: Vec<String>,
    /// DreamID-V only: the target VIDEO (mp4) to swap a face into.
    #[arg(long)]
    pub input_video: Option<PathBuf>,
    /// DreamID-V only: the source FACE image (PNG/JPEG) to swap in.
    #[arg(long)]
    pub source_image: Option<PathBuf>,
    /// DreamID-V only: image-CFG guidance scale on the source-face reference
    /// (default 4.0). Ignored by the other (CFG-free) video models.
    #[arg(long)]
    pub guide_scale: Option<f32>,
    /// Output video file (e.g. `out.mp4`).
    #[arg(long)]
    pub output: PathBuf,
    /// Output format. Defaults to inferring from the `--output` extension.
    #[arg(long, value_enum)]
    pub output_format: Option<VideoFormat>,
    /// Frame width in pixels. Defaults per model (FastWan/LongLive 960, LTX
    /// 256). FastWan/LongLive need a multiple of 32; LTX a multiple of 64.
    #[arg(long)]
    pub width: Option<u32>,
    /// Frame height in pixels. Defaults per model (FastWan/LongLive 544, LTX
    /// 256). FastWan/LongLive need a multiple of 32; LTX a multiple of 64.
    #[arg(long)]
    pub height: Option<u32>,
    /// Output frame count. Must be `4 * k + 1`; LongLive additionally needs
    /// `(frames-1)/4+1` divisible by 8 (29, 61, 93, ...). Mutually exclusive
    /// with `--duration`. One value splits a multi-shot clip evenly; one per
    /// `--prompt` sizes each shot.
    #[arg(long, conflicts_with = "duration")]
    pub frames: Vec<u32>,
    /// Target clip length in seconds (mutually exclusive with `--frames`):
    /// frames = round(duration * fps), snapped to the model's legal grid.
    #[arg(long, conflicts_with = "frames")]
    pub duration: Vec<f32>,
    /// Playback fps written into the MP4 (and used to convert `--duration`).
    /// Defaults to the model's preferred fps (Wan TI2V = 24).
    #[arg(long)]
    pub fps: Option<u32>,
    /// Seed. Omit for a randomized seed.
    #[arg(long)]
    pub seed: Option<u64>,
    /// Denoise sampler: `unipc` (default, multistep, `--steps` configurable) or
    /// `dmd` (fixed 3-step byte-parity path). Ignored on the AR (LongLive) path.
    #[arg(long, value_enum, default_value_t = VideoSampler::default())]
    pub sampler: VideoSampler,
    /// UniPC denoise steps (1..=8; dreamid-v defaults to 16). Ignored when
    /// `--sampler dmd`. Unset -> the model default.
    #[arg(long)]
    pub steps: Option<u32>,
    /// Temporal self-attention window radius in LATENT frames (Wan2.2 14B). Each
    /// query attends only to keys within `±N` latent frames, breaking the
    /// O(frames^2) self-attention cost on long clips at the price of long-range
    /// temporal coherence. Unset = the model default (3 for Wan2.2 14B, long
    /// clips only); `0` forces full attention. Honored only on the activation-
    /// tiled long-clip path.
    #[arg(long)]
    pub attn_window: Option<u32>,
    /// First-frame conditioning image (PNG/JPEG). Optional, on
    /// hunyuan-video-1.5-ti2v and ltx2-rapid: with it the run animates the image
    /// (I2V); without it the model generates from the prompt alone.
    #[arg(long)]
    pub input_image: Option<PathBuf>,
    /// LTX native-I2V frame-0 conditioning strength (0.0..=1.0, default 1.0):
    /// 1.0 locks the input frame through the denoise; lower lets it drift toward
    /// the model's own first frame. Used only with --input-image on ltx2-rapid.
    #[arg(long, default_value_t = 1.0)]
    pub strength: f32,
    /// Host RAM budget for the weight residency manager. e.g. `8G`, `512M`.
    #[arg(long)]
    pub ram_budget: Option<String>,
    /// GPU VRAM budget for the weight residency manager.
    #[arg(long)]
    pub vram_budget: Option<String>,
    /// Skip the TTY consent prompt and download missing weight files.
    #[arg(long, default_value_t = false)]
    pub download_as_needed: bool,
    /// VAE decoder. Default `tiny` (LightTAE): ~0.4GB, ~50x faster decode at
    /// near-identical quality. `full` is the bit-clean parity path.
    #[arg(long, value_enum, default_value_t = VaeChoice::Tiny)]
    pub vae: VaeChoice,
    /// LTX-2.3 text encoder (all LTX/Sulphur models). `q8` (default) is the
    /// conditioning-quality baseline; `q4` is Q4_K_M -- ~2.8x faster encode
    /// (~16s -> ~6s) for lower-precision conditioning. Ignored by Wan.
    #[arg(long, value_enum, default_value_t = EncoderQuant::Q8)]
    pub encoder: EncoderQuant,
    /// Disable the default DP4A int8 matmul on the quantization-safe DiT sites
    /// (forces the bf16 reference path throughout).
    #[arg(long)]
    pub no_i8_matmul: bool,
    /// Skip the audio track (LTX joint-AV only): decode video only, for a faster
    /// silent MP4. Ignored by the silent Wan models.
    #[arg(long)]
    pub no_audio: bool,
    /// LTX-2.3 distilled only: opt in to the 2x spatial-upscale refine path
    /// (half-res denoise -> latent upscale -> refine), the cheaper route to high
    /// res. Default off = single-stage denoise at the target res. Ignored by Wan.
    #[arg(long)]
    pub upscale: bool,
    /// HunyuanVideo 1.5 only: skip prompt rewriting and send the raw prompt. By
    /// default a short prompt is expanded into a detailed, structured caption by
    /// the on-device Qwen3-VL rewriter (the model is trained on long captions;
    /// raw short prompts are out-of-distribution and yield incoherent video).
    #[arg(long)]
    pub no_rewrite: bool,
    /// HunyuanVideo 1.5 only: which rewriter model to use. `fast` (default) runs
    /// the ~2.5GB Qwen3-VL-4B (quick even under a tight VRAM budget); `full` runs
    /// the ~5.85GB Qwen3-VL-8B (slightly richer captions, much slower). Ignored
    /// with `--no-rewrite` or on non-Hunyuan models.
    #[arg(long, value_enum, default_value_t = RewriteQuality::Fast)]
    pub rewrite_quality: RewriteQuality,
    /// Fold a stored adapter into the DiT, as `NAME_OR_ID[:WEIGHT]` (repeatable,
    /// applied in order). Resolved against the vault for `--model`; needs the
    /// vault password (hidden prompt, or `THINFER_VAULT_PASSWORD`). Local runs
    /// only (not `--remote`); only AnyFlow accepts adapters today.
    #[arg(long = "lora", value_name = "NAME[:WEIGHT]")]
    pub lora: Vec<String>,
    /// Vault directory for `--lora`. Defaults to the shared location
    /// (`THINFER_VAULT_DIR`, else `<hf-cache>/vault`).
    #[arg(long)]
    pub vault_dir: Option<PathBuf>,
    #[command(flatten)]
    pub remote: super::RemoteArgs,
}

/// Read a file and base64-encode it for a remote job spec, or `None` when the
/// path is absent.
fn base64_of(path: Option<&std::path::Path>) -> Result<Option<String>, String> {
    use base64::Engine;
    match path {
        Some(p) => {
            let bytes = std::fs::read(p).map_err(|e| format!("read {}: {e}", p.display()))?;
            Ok(Some(
                base64::engine::general_purpose::STANDARD.encode(bytes),
            ))
        }
        None => Ok(None),
    }
}

pub async fn run_video(args: GenerateVideo) -> Result<(), String> {
    let format = resolve_output_format(
        args.output_format,
        &args.output,
        VideoFormat::from_ext,
        VideoFormat::KNOWN,
    )?;

    if args.remote.remote.is_some() {
        if format != VideoFormat::Mp4 {
            return Err("--remote only produces MP4 (png-frames is a local debug format)".into());
        }
        if !args.lora.is_empty() {
            return Err(
                "--lora (adapters) is not supported over --remote; run locally or use the web UI"
                    .into(),
            );
        }
        let spec = JobSpec::Video(VideoSpec {
            model: Some(args.model),
            prompts: args.prompt,
            width: args.width,
            height: args.height,
            frames: (!args.frames.is_empty()).then_some(args.frames),
            durations: (!args.duration.is_empty()).then_some(args.duration),
            fps: args.fps,
            seed: args.seed,
            sampler: Some(args.sampler),
            steps: args.steps,
            attn_window: args.attn_window,
            vae: Some(args.vae),
            input_image: base64_of(args.input_image.as_deref())?,
            strength: Some(args.strength),
            source_image: base64_of(args.source_image.as_deref())?,
            input_video: base64_of(args.input_video.as_deref())?,
            // The CLI reads local files inline; the raw-upload endpoint is a
            // serve-only path, so there is never an upload id here.
            input_video_upload: None,
            guide_scale: args.guide_scale,
            encoder: Some(args.encoder),
            i8_matmul: Some(!args.no_i8_matmul),
            audio: Some(!args.no_audio),
            upscale: Some(args.upscale || args.model.two_stage_default()),
            rewrite: Some(!args.no_rewrite),
            rewrite_quality: Some(args.rewrite_quality),
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
        let (refs, password) =
            super::image::resolve_loras(args.model, args.vault_dir.as_deref(), &args.lora)?;
        (refs, Some(password))
    };

    let (def_w, def_h) = args.model.video_defaults();
    let req = VideoRequest {
        model: args.model,
        prompts: args.prompt,
        width: args.width.unwrap_or(def_w),
        height: args.height.unwrap_or(def_h),
        frames: args.frames,
        durations: args.duration,
        fps: args.fps,
        seed: args.seed,
        input_image: match &args.input_image {
            Some(p) => Some(ImageBytes(
                std::fs::read(p).map_err(|e| format!("read {}: {e}", p.display()))?,
            )),
            None => None,
        },
        strength: args.strength,
        source_image: match &args.source_image {
            Some(p) => Some(ImageBytes(
                std::fs::read(p).map_err(|e| format!("read {}: {e}", p.display()))?,
            )),
            None => None,
        },
        input_video: match &args.input_video {
            Some(p) => Some(VideoInput::Ram(
                std::fs::read(p).map_err(|e| format!("read {}: {e}", p.display()))?,
            )),
            None => None,
        },
        guide_scale: args.guide_scale,
        sampler: args.sampler,
        steps: args.steps.unwrap_or(args.model.default_steps()),
        attn_window: args.attn_window,
        vae: args.vae,
        encoder: args.encoder,
        i8_matmul: !args.no_i8_matmul,
        audio: !args.no_audio,
        // LTX defaults to two-stage (single-stage at the widescreen default OOMs
        // 8GB and low-res single-stage is OOD); `--upscale` is store-true, so OR
        // with the model default. Wan ignores upscale.
        upscale: args.upscale || args.model.two_stage_default(),
        rewrite: !args.no_rewrite,
        rewrite_quality: args.rewrite_quality,
        lora,
        vault_password,
        vault_dir: thinfer_app::vault::resolve_dir(args.vault_dir.as_deref()),
        budget: ResidencyBudget {
            ram_bytes: ram,
            vram_bytes: vram,
        },
        output: args.output,
        format,
    };
    // Fail fast on dims / shot-plan errors before any download or GPU work.
    req.resolve()?;
    let files = req.required_files()?;
    super::run_job(
        JobRequest::Video(req),
        &files,
        args.download_as_needed,
        ram,
        vram,
    )
    .await
}
