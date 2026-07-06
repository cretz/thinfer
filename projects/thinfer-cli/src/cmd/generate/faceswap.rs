//! `thinfer generate face-swap`. Thin clap adapter over
//! `thinfer_app::FaceSwapRequest`; decode/swap/encode lives in `thinfer-app`.

use std::path::PathBuf;

use clap::Args;
use thinfer_app::config::ResidencyBudget;
use thinfer_app::model::SwapModel;
use thinfer_app::request::{FaceImage, FaceSwapRequest, VideoInput};
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
    /// Swap model checkpoint.
    #[arg(long, value_enum, default_value_t = SwapModel::DEFAULT)]
    pub model: SwapModel,
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
        source_image: FaceImage(source_bytes),
        output: args.output,
        budget: ResidencyBudget {
            ram_bytes: ram,
            vram_bytes: vram,
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
