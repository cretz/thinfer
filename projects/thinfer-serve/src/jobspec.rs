//! Decrypted job spec -> executable request. Runs in the worker only after the
//! job key arrived (see `crate::job`); before that the spec is ciphertext.

use std::path::Path;

use thinfer_app::JobRequest;
use thinfer_app::jobkey::JobKey;
use thinfer_app::model::{ImageModelId, SwapModel, VaeChoice, VideoModelId};
use thinfer_app::request::{
    FaceSwapRequest, ImageBytes, ImageFormat, ImageRequest, LoraRef, Secret, VideoFormat,
    VideoInput, VideoRequest,
};
use thinfer_app::wire::JobSpec;

use crate::config::ServeConfig;
use crate::uploads::UploadStore;

/// Base64-decode a required upload field, tagging errors with the field name.
fn b64_decode(field: &str, b64: &str) -> Result<Vec<u8>, String> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(b64.as_bytes())
        .map_err(|e| format!("{field} is not valid base64: {e}"))
}

/// Turn decoded video bytes into a [`VideoInput`]: held in RAM under the spill
/// threshold, else sealed to `<dir>/input_video.enc` under a fresh ephemeral key
/// (held in the returned value, RAM only). The raw plaintext never touches disk.
/// Shared with the streaming-upload path ([`crate::uploads`]).
fn video_input(bytes: Vec<u8>, dir: &std::path::Path) -> Result<VideoInput, String> {
    crate::uploads::seal_video(bytes, dir, crate::uploads::VIDEO_SPILL_THRESHOLD)
}

/// Build the executable request for a decrypted `spec` in the job dir `dir`.
/// Runs in the worker, AFTER the job key arrived (the spec is sealed until
/// then), so validation errors surface as job errors. This is where the server
/// fills in what a client does not send (artifact path, budgets, output
/// format). Returns the request + the per-request coopmat opt-out.
pub fn spec_into_request(
    spec: JobSpec,
    dir: &Path,
    key: &JobKey,
    config: &ServeConfig,
    uploads: &UploadStore,
) -> Result<(JobRequest, Option<bool>), String> {
    let budget = config.budget()?;
    let mp4 = || dir.join("output.mp4");
    let make_dir =
        || std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()));
    let disable_coopmat = spec.disable_coopmat();
    match spec {
        JobSpec::Image(s) => {
            let model = s.model.unwrap_or(ImageModelId::DEFAULT);
            let d = model.defaults();
            // Image-edit reference image: base64-decode and hold in RAM
            // ([`ImageBytes`]); a decrypted image never touches disk.
            // `ImageRequest::validate` enforces the present/required-by-kind
            // rules (400 on mismatch).
            let input_image = match s.input_image {
                Some(b64) => Some(ImageBytes(b64_decode("input_image", &b64)?)),
                None => None,
            };
            let req = ImageRequest {
                model,
                prompt: s.prompt,
                width: s.width.unwrap_or(d.width),
                height: s.height.unwrap_or(d.height),
                steps: s.steps.unwrap_or(d.steps),
                seed: s.seed,
                i8_matmul: s.i8_matmul.unwrap_or(true),
                full_precision: s.full_precision.unwrap_or(false),
                input_image,
                lora: s
                    .lora
                    .into_iter()
                    .map(|l| LoraRef {
                        id: l.id,
                        weight: l.weight.unwrap_or(1.0),
                    })
                    .collect(),
                vault_password: s.password.map(Secret::new),
                vault_dir: config.resolved_vault_dir(),
                budget,
                output: dir.join("output.png"),
                format: ImageFormat::Png,
            };
            req.validate()?;
            make_dir()?;
            Ok((JobRequest::Image(req), disable_coopmat))
        }
        JobSpec::Video(s) => {
            let model = s.model.unwrap_or(VideoModelId::DEFAULT);
            let (def_w, def_h) = model.video_defaults();
            // First-frame conditioning image (hunyuan-video-1.5-i2v): decode the
            // base64 payload and hold in RAM ([`ImageBytes`]); a decrypted image
            // never touches disk. `VideoRequest::resolve` enforces
            // required/rejected per model.
            let input_image = match s.input_image {
                Some(b64) => Some(ImageBytes(b64_decode("input_image", &b64)?)),
                None => None,
            };
            // DreamID-V source FACE image: small, held in RAM only (never on
            // disk), matching the target video's RAM-first privacy.
            let source_image = match s.source_image {
                Some(b64) => Some(ImageBytes(b64_decode("source_image", &b64)?)),
                None => None,
            };
            // DreamID-V target VIDEO: a streamed upload id (preferred; the
            // client sealed it under the job key, so it stays ciphertext until
            // the frames are decoded) is consumed into the job dir; else a
            // base64 payload is decoded RAM-first with an encrypted spill for
            // large clips. Either way, raw plaintext never lands on disk.
            let input_video = match (s.input_video_upload, s.input_video) {
                (Some(uid), _) => Some(uploads.consume(&uid, dir, key)?),
                (None, Some(b64)) => Some(video_input(b64_decode("input_video", &b64)?, dir)?),
                (None, None) => None,
            };
            let req = VideoRequest {
                model,
                prompts: s.prompts,
                width: s.width.unwrap_or(def_w),
                height: s.height.unwrap_or(def_h),
                frames: s.frames.unwrap_or_default(),
                durations: s.durations.unwrap_or_default(),
                fps: s.fps,
                seed: s.seed,
                input_image,
                strength: s.strength.unwrap_or(1.0),
                source_image,
                input_video,
                guide_scale: s.guide_scale,
                sampler: s.sampler.unwrap_or(model.default_sampler()),
                steps: s.steps.unwrap_or(model.default_steps()),
                attn_window: s.attn_window,
                vae: s.vae.unwrap_or(VaeChoice::Tiny),
                encoder: s.encoder.unwrap_or_default(),
                i8_matmul: s.i8_matmul.unwrap_or(true),
                audio: s.audio.unwrap_or(true),
                upscale: s.upscale.unwrap_or(model.two_stage_default()),
                audio_hq: s.audio_hq.unwrap_or(false),
                rewrite: s.rewrite.unwrap_or(true),
                rewrite_quality: s.rewrite_quality.unwrap_or_default(),
                lora: s
                    .lora
                    .into_iter()
                    .map(|l| LoraRef {
                        id: l.id,
                        weight: l.weight.unwrap_or(1.0),
                    })
                    .collect(),
                vault_password: s.password.map(Secret::new),
                vault_dir: config.resolved_vault_dir(),
                budget,
                // Server emits MP4 only (PNG-frames is a CLI debug format).
                output: mp4(),
                format: VideoFormat::Mp4,
            };
            req.resolve()?;
            make_dir()?;
            Ok((JobRequest::Video(req), disable_coopmat))
        }
        JobSpec::FaceSwap(s) => {
            // Target video: a streamed upload id (sealed under the job key,
            // consumed into the job dir) or base64-in-JSON bytes (RAM-first /
            // encrypted spill). Server-side paths are refused: a client must not
            // be able to feed arbitrary server files into a job.
            if s.input_video.is_some() || s.source_image.is_some() {
                return Err("server-side input paths are not accepted; upload the media".into());
            }
            let input_video = match (s.input_video_upload, s.input_video_b64) {
                (Some(uid), _) => uploads.consume(&uid, dir, key)?,
                (None, Some(b64)) => video_input(b64_decode("input_video_b64", &b64)?, dir)?,
                (None, None) => return Err("face-swap requires an input video".into()),
            };
            // Source face image: uploaded bytes, held in RAM only.
            let source_image = match s.source_image_b64 {
                Some(b64) => ImageBytes(b64_decode("source_image_b64", &b64)?),
                None => return Err("face-swap requires a source image".into()),
            };
            let req = FaceSwapRequest {
                model: s.model.unwrap_or(SwapModel::DEFAULT),
                input_video,
                source_image,
                output: mp4(),
                budget,
                options: thinfer_app::request::FaceSwapOptions {
                    hyperswap_mask: s.hyperswap_mask.unwrap_or(false),
                    occlusion: s.occlusion_mask.unwrap_or(false),
                    enhance: s.enhance.unwrap_or(false),
                    detect_stride: s.detect_stride.unwrap_or(1),
                    bitrate_scale: s.bitrate_scale.unwrap_or(1.15),
                    start_secs: s.start_secs,
                    end_secs: s.end_secs,
                },
            };
            req.validate()?;
            make_dir()?;
            Ok((JobRequest::FaceSwap(req), disable_coopmat))
        }
    }
}
