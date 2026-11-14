//! `thinfer generate face-swap`. Thin clap adapter over
//! `thinfer_app::FaceSwapRequest`; decode/swap/encode lives in `thinfer-app`.

use std::path::PathBuf;

use clap::Args;
use thinfer_app::config::ResidencyBudget;
use thinfer_app::model::SwapModel;
use thinfer_app::request::{
    DiffusionSwapOptions, FaceSwapOptions, FaceSwapRequest, ImageBytes, VideoInput,
};
use thinfer_app::{JobRequest, parse_budget};

#[derive(Args)]
pub struct GenerateFaceSwap {
    /// Input video (mp4 / H.264). Every frame is processed.
    #[arg(long)]
    pub input_video: PathBuf,
    /// Source image (PNG/JPEG): the face to paste into the video.
    #[arg(long)]
    pub source_image: PathBuf,
    /// Output video file (.mp4).
    #[arg(long)]
    pub output: PathBuf,
    /// Swap model: a HyperSwap checkpoint (fast, per-frame paste) or
    /// `dreamid-v` (diffusion; slow, holds identity through hard poses).
    #[arg(long, value_enum, default_value_t = SwapModel::DEFAULT)]
    pub model: SwapModel,
    /// dreamid-v: denoise steps (the distill is authored for 16).
    #[arg(long)]
    pub steps: Option<u32>,
    /// dreamid-v: image-CFG guidance scale on the source face (default 4).
    #[arg(long)]
    pub guide_scale: Option<f32>,
    /// dreamid-v: seed (default random).
    #[arg(long)]
    pub seed: Option<u64>,
    /// dreamid-v: working width; the clip is resized toward width x height
    /// pixels, its own aspect kept (default 832x480, multiples of 16).
    #[arg(long)]
    pub width: Option<u32>,
    /// dreamid-v: working height (see `--width`).
    #[arg(long)]
    pub height: Option<u32>,
    /// dreamid-v opt-in: run the unguided forward only every N steps and
    /// reuse its guidance in between (1 = every step, the reference; 2 cuts
    /// denoise ~25%).
    #[arg(long)]
    pub cfg_every: Option<u32>,
    /// dreamid-v opt-in: run on the face region only (the face masks' union
    /// with context, at up to its native size) and paste it back into the
    /// full-resolution frames. Far faster; the output keeps the source size.
    #[arg(long)]
    pub face_crop: bool,
    /// Intersect the paste mask with an XSeg occlusion mask so occluders (hands,
    /// hair, glasses, mics) crossing the face show the original frame through
    /// instead of swapped pixels. Adds one ONNX forward per face.
    #[arg(long, default_value_t = false)]
    pub occlusion_mask: bool,
    /// Intersect the paste mask with HyperSwap's own confidence mask output
    /// (free; tightens the blend to the synthesized region).
    #[arg(long, default_value_t = false)]
    pub hyperswap_mask: bool,
    /// Run the GFPGAN face enhancer on each swapped face (restores skin/eye/teeth
    /// detail). Adds one ONNX forward per face.
    #[arg(long, default_value_t = false)]
    pub enhance: bool,
    /// Run face detection every Nth frame, reusing the previous detection in
    /// between (the per-frame swap still runs). 1 = every frame (default). Higher
    /// is faster but tracks fast head motion less tightly.
    #[arg(long, default_value_t = 1)]
    pub detect_stride: u32,
    /// Output H.264 bitrate as a multiple of the source video bitrate. 1.15
    /// (default) matches the input's size/quality with a small cushion; higher =
    /// larger/cleaner, lower = smaller/softer. Ignored if the source bitrate can't
    /// be read (a bits-per-pixel target is used instead).
    #[arg(long, default_value_t = 1.15)]
    pub bitrate_scale: f32,
    /// Only swap+output the source window starting at this many seconds (rebased
    /// to 0 in the output). Default: clip start.
    #[arg(long)]
    pub start: Option<f32>,
    /// Only swap+output the source window ending at this many seconds. Default:
    /// clip end.
    #[arg(long)]
    pub end: Option<f32>,
    /// Host RAM budget. e.g. `8G`, `512M`, raw bytes. Face-swap streams the clip
    /// frame-by-frame, so peak RAM is a few frames regardless; the flag exists
    /// for parity with the other commands and feeds the `[mem]` rollup.
    #[arg(long)]
    pub ram_budget: Option<String>,
    /// GPU VRAM budget (parity flag; see `--ram-budget`).
    #[arg(long)]
    pub vram_budget: Option<String>,
    /// Skip the TTY consent prompt and download missing model files.
    #[arg(long, default_value_t = false)]
    pub download_as_needed: bool,
}

pub async fn run_faceswap(args: GenerateFaceSwap) -> Result<(), String> {
    let ram = parse_budget("--ram-budget", args.ram_budget.as_deref())?;
    let vram = parse_budget("--vram-budget", args.vram_budget.as_deref())?;

    // The target video rides RAM-first (the request holds the mp4 bytes; a serve
    // upload may instead be an encrypted spill). The CLI reads the local file.
    let video_bytes = std::fs::read(&args.input_video)
        .map_err(|e| format!("read {}: {e}", args.input_video.display()))?;
    let source_bytes = std::fs::read(&args.source_image)
        .map_err(|e| format!("read {}: {e}", args.source_image.display()))?;
    let req = FaceSwapRequest {
        model: args.model,
        input_video: VideoInput::Ram(video_bytes),
        source_image: ImageBytes(source_bytes),
        output: args.output,
        budget: ResidencyBudget {
            ram_bytes: ram,
            vram_bytes: vram,
        },
        options: FaceSwapOptions {
            hyperswap_mask: args.hyperswap_mask,
            occlusion: args.occlusion_mask,
            enhance: args.enhance,
            detect_stride: args.detect_stride,
            bitrate_scale: args.bitrate_scale,
            start_secs: args.start,
            end_secs: args.end,
            diffusion: {
                let d = DiffusionSwapOptions::default();
                DiffusionSwapOptions {
                    steps: args.steps.unwrap_or(d.steps),
                    guide_scale: args.guide_scale.unwrap_or(d.guide_scale),
                    seed: args.seed,
                    width: args.width.unwrap_or(d.width),
                    height: args.height.unwrap_or(d.height),
                    cfg_every: args.cfg_every.unwrap_or(d.cfg_every),
                    face_crop: args.face_crop,
                }
            },
        },
    };
    req.validate()?;
    let files = req.required_files();
    super::run_job(
        JobRequest::FaceSwap(req),
        &files,
        args.download_as_needed,
        ram,
        vram,
    )
    .await
}
