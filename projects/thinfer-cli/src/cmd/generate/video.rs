//! `thinfer generate video` (FastWan / LongLive t2v). Thin clap adapter over
//! `thinfer_app::VideoRequest`; the shot-plan resolution, denoise, and MP4
//! encode all live in `thinfer-app`.

use std::path::PathBuf;

use clap::Args;
use thinfer_app::config::ResidencyBudget;
use thinfer_app::model::{
    EncoderQuant, VIDEO_DEFAULT_STEPS, VaeChoice, VideoModelId, VideoSampler,
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
    /// UniPC denoise steps (1..=8). Ignored when `--sampler dmd`.
    #[arg(long, default_value_t = VIDEO_DEFAULT_STEPS)]
    pub steps: u32,
    /// Temporal self-attention window radius in LATENT frames (Wan2.2 14B). Each
    /// query attends only to keys within `±N` latent frames, breaking the
    /// O(frames^2) self-attention cost on long clips at the price of long-range
    /// temporal coherence. Unset = the model default (3 for Wan2.2 14B, long
    /// clips only); `0` forces full attention. Honored only on the activation-
    /// tiled long-clip path.
    #[arg(long)]
    pub attn_window: Option<u32>,
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
            width: args.width,
            height: args.height,
            frames: (!args.frames.is_empty()).then_some(args.frames),
            durations: (!args.duration.is_empty()).then_some(args.duration),
            fps: args.fps,
            seed: args.seed,
            sampler: Some(args.sampler),
            steps: Some(args.steps),
            attn_window: args.attn_window,
            vae: Some(args.vae),
            encoder: Some(args.encoder),
            i8_matmul: Some(!args.no_i8_matmul),
            audio: Some(!args.no_audio),
            upscale: Some(args.upscale || args.model.two_stage_default()),
            public_key: None,
            // Remote path defers coopmat to the server default; local runs read
            // THINFER_NO_COOPMAT via BackendConfig.
            disable_coopmat: None,
        });
        return super::run_remote(&args.remote, spec, args.output).await;
    }
    let ram = parse_budget("--ram-budget", args.ram_budget.as_deref())?;
    let vram = parse_budget("--vram-budget", args.vram_budget.as_deref())?;

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
        input_image: args.input_image,
        sampler: args.sampler,
        steps: args.steps,
        attn_window: args.attn_window,
        vae: args.vae,
        encoder: args.encoder,
        i8_matmul: !args.no_i8_matmul,
        audio: !args.no_audio,
        // LTX defaults to two-stage (single-stage at the widescreen default OOMs
        // 8GB and low-res single-stage is OOD); `--upscale` is store-true, so OR
        // with the model default. Wan ignores upscale.
        upscale: args.upscale || args.model.two_stage_default(),
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
