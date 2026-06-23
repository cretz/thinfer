//! `thinfer generate video` (FastWan / LongLive t2v). Thin clap adapter over
//! `thinfer_app::VideoRequest`; the shot-plan resolution, denoise, and MP4
//! encode all live in `thinfer-app`.

use std::path::PathBuf;

use clap::Args;
use thinfer_app::config::ResidencyBudget;
use thinfer_app::model::{
    VIDEO_DEFAULT_HEIGHT, VIDEO_DEFAULT_STEPS, VIDEO_DEFAULT_WIDTH, VaeChoice, VideoModelId,
    VideoSampler,
};
use thinfer_app::request::{VideoFormat, VideoRequest};
use thinfer_app::wire::{JobSpec, VideoSpec};
use thinfer_app::{JobRequest, parse_budget, resolve_output_format};

/// FastWan is CFG-free (no `--guidance-scale`/`--negative-prompt`). The denoise
/// sampler is selectable: `--sampler unipc` (default, `--steps` configurable,
/// matches the public FastWan Spaces) or `--sampler dmd` (the fixed 3-step
/// byte-parity reference; `--steps` is ignored). Dims must be /32; frames must be
/// 4k+1 (the causal-VAE temporal grid). For a fast first run, pass `--frames 5`.
#[derive(Args)]
pub struct GenerateVideo {
    /// Model identifier. Defaults to `fastwan-ti2v-5b` (safetensors, the
    /// e2e-validated path).
    #[arg(long, default_value_t = VideoModelId::DEFAULT, value_enum)]
    pub model: VideoModelId,
    /// Text prompt. Repeat `--prompt` for a multi-shot video (LongLive only):
    /// each prompt is one shot, with a scene cut between shots.
    #[arg(long, required = true)]
    pub prompt: Vec<String>,
    /// Output video file (e.g. `out.mp4`).
    #[arg(long)]
    pub output: PathBuf,
    /// Output format. Defaults to inferring from the `--output` extension.
    #[arg(long, value_enum)]
    pub output_format: Option<VideoFormat>,
    /// Frame width in pixels. Must be divisible by 32.
    #[arg(long, default_value_t = VIDEO_DEFAULT_WIDTH)]
    pub width: u32,
    /// Frame height in pixels. Must be divisible by 32.
    #[arg(long, default_value_t = VIDEO_DEFAULT_HEIGHT)]
    pub height: u32,
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
    /// UniPC denoise steps (1..=8). Ignored when `--sampler dmd`.
    #[arg(long, default_value_t = VIDEO_DEFAULT_STEPS)]
    pub steps: u32,
    /// img2vid conditioning image. Not yet wired (engine path is t2v-only).
    #[arg(long)]
    pub input_image: Option<PathBuf>,
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
    /// Disable the default DP4A int8 matmul on the quantization-safe DiT sites
    /// (forces the bf16 reference path throughout).
    #[arg(long)]
    pub no_i8_matmul: bool,
    #[command(flatten)]
    pub remote: super::RemoteArgs,
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
        let spec = JobSpec::Video(VideoSpec {
            model: Some(args.model),
            prompts: args.prompt,
            width: Some(args.width),
            height: Some(args.height),
            frames: (!args.frames.is_empty()).then_some(args.frames),
            durations: (!args.duration.is_empty()).then_some(args.duration),
            fps: args.fps,
            seed: args.seed,
            sampler: Some(args.sampler),
            steps: Some(args.steps),
            vae: Some(args.vae),
            i8_matmul: Some(!args.no_i8_matmul),
            public_key: None,
        });
        return super::run_remote(&args.remote, spec, args.output).await;
    }
    let ram = parse_budget("--ram-budget", args.ram_budget.as_deref())?;
    let vram = parse_budget("--vram-budget", args.vram_budget.as_deref())?;

    let req = VideoRequest {
        model: args.model,
        prompts: args.prompt,
        width: args.width,
        height: args.height,
        frames: args.frames,
        durations: args.duration,
        fps: args.fps,
        seed: args.seed,
        input_image: args.input_image,
        sampler: args.sampler,
        steps: args.steps,
        vae: args.vae,
        i8_matmul: !args.no_i8_matmul,
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
