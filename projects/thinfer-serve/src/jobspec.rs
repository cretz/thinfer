//! Decrypted job spec -> executable request. Runs in the worker only after the
//! job key arrived (see `crate::job`); before that the spec is ciphertext.

use std::path::Path;

use thinfer_app::JobRequest;
use thinfer_app::jobkey::{JobKey, OpenLocked};
use thinfer_app::model::{
    AudioKind, AudioModelId, ImageModelId, SwapModel, TextModelId, VideoModelId,
};
use thinfer_app::request::{
    AudioBytes, AudioRequest, FaceSwapRequest, ImageBytes, ImageFormat, ImageRequest, LoraRef,
    Secret, SoundParams, TextRequest, VideoFormat, VideoInput, VideoRequest, resolve_keep_size,
};
use thinfer_app::wire::{JobSpec, TextSessionRef};

use crate::config::ServeConfig;
use crate::sessions::SessionStore;
use crate::uploads::UploadStore;

/// Where a session job's new node goes once the reply is done: the session,
/// the node it continued (`None` = a new root), and the session key that
/// seals the node (dropped, zeroed, with the job).
pub struct SessionCommit {
    pub id: String,
    pub parent: Option<String>,
    pub key: JobKey,
}

/// A decrypted job, ready to run.
pub struct Built {
    pub request: JobRequest,
    /// Per-request coopmat opt-out.
    pub disable_coopmat: Option<bool>,
    pub session: Option<SessionCommit>,
}

impl Built {
    fn new(request: JobRequest, disable_coopmat: Option<bool>) -> Self {
        Self {
            request,
            disable_coopmat,
            session: None,
        }
    }
}

/// Open a text job's session: check the key against the session, decrypt the
/// root-to-`from` chain into page-locked state chunks.
fn open_session(
    mut r: TextSessionRef,
    model: TextModelId,
    sessions: &SessionStore,
) -> Result<(thinfer_app::request::TextSession, SessionCommit), String> {
    let key = JobKey::from_b64(&r.key).map_err(|e| format!("session key: {e}"))?;
    zeroize::Zeroize::zeroize(&mut r.key);
    let chain = sessions
        .chain(&r.id, &key.proof(), r.from.as_deref())
        .map_err(|e| format!("session: {e}"))?;
    if chain.model != model {
        return Err(format!("session is for {}, not {model}", chain.model));
    }
    let mut path = Vec::with_capacity(chain.links.len());
    for (start, end, sealed) in &chain.links {
        path.push(thinfer_app::request::StateChunk {
            start: *start,
            end: *end,
            bytes: key
                .open_locked(sealed)
                .map_err(|e| format!("session node: {e}"))?,
        });
    }
    let session = thinfer_app::request::TextSession {
        path,
        weights: chain.weights,
        tail: chain.tail,
    };
    let commit = SessionCommit {
        id: r.id,
        parent: r.from,
        key,
    };
    Ok((session, commit))
}

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
    job_id: &str,
    dir: &Path,
    key: &JobKey,
    config: &ServeConfig,
    uploads: &UploadStore,
    sessions: &SessionStore,
) -> Result<Built, String> {
    let budget = config.budget()?;
    let mp4 = || dir.join("output.mp4");
    let disable_coopmat = spec.disable_coopmat();
    match spec {
        JobSpec::Text(s) => {
            let model = s.model.unwrap_or(TextModelId::DEFAULT);
            let (session, commit) = match s.session {
                Some(r) => {
                    let (session, commit) = open_session(r, model, sessions)?;
                    (Some(std::sync::Arc::new(session)), Some(commit))
                }
                None => (None, None),
            };
            let req = TextRequest {
                model,
                messages: s
                    .messages
                    .into_iter()
                    .map(thinfer_app::llm::chat_message)
                    .collect(),
                max_tokens: s.max_tokens.unwrap_or(2048),
                think: s.think.unwrap_or(false),
                temperature: s.temperature,
                top_p: s.top_p,
                top_k: s.top_k,
                min_p: s.min_p,
                presence_penalty: s.presence_penalty,
                repetition_penalty: s.repetition_penalty,
                seed: s.seed,
                budget: config.text_budget()?,
                output: Some(dir.join("reply.txt")),
                format: s.format,
                session,
                tools: s.tools,
                tool_choice: s.tool_choice.unwrap_or_default(),
            };
            req.validate()?;
            Ok(Built {
                session: commit,
                ..Built::new(JobRequest::Text(req), disable_coopmat)
            })
        }
        JobSpec::Audio(s) => {
            let mut clips = s
                .ref_audios
                .unwrap_or_default()
                .iter()
                .map(|b64| b64_decode("ref_audios", b64).map(AudioBytes))
                .collect::<Result<Vec<_>, String>>()?;
            if clips.len() > 1 {
                return Err("one reference clip at most".into());
            }
            let model = s.model.unwrap_or(AudioModelId::DEFAULT);
            // Sound settings ride only on sound models; on a speech model
            // they are an error (validate), not silently dropped.
            let sound_fields = s.seconds.is_some()
                || s.looped.is_some()
                || s.steps.is_some()
                || s.guidance.is_some();
            let sound = (model.kind() == AudioKind::Sound || sound_fields)
                .then(|| SoundParams::resolve(s.seconds, s.looped, s.steps, s.guidance));
            let req = AudioRequest {
                model,
                sound,
                text: s.text,
                instruct: s.instruct.filter(|i| !i.trim().is_empty()),
                ref_audio: clips.pop(),
                ref_text: s.ref_text.filter(|t| !t.trim().is_empty()),
                language: s.language.unwrap_or_else(|| "auto".into()),
                seed: s.seed,
                temperature: s.temperature,
                top_k: s.top_k,
                top_p: s.top_p,
                max_seconds: s.max_seconds.unwrap_or(60.0),
                budget,
                output: dir.join("output.wav"),
            };
            req.validate()?;
            Ok(Built::new(JobRequest::Audio(req), disable_coopmat))
        }
        JobSpec::Image(s) => {
            let model = s.model.unwrap_or(ImageModelId::DEFAULT);
            let d = model.defaults();
            // Input images: base64-decode and hold in RAM ([`ImageBytes`]); a
            // decrypted image never touches disk. `ImageRequest::validate`
            // enforces the per-kind counts (400 on mismatch).
            let input_images = s
                .input_images
                .iter()
                .map(|b64| b64_decode("input_images", b64).map(ImageBytes))
                .collect::<Result<Vec<_>, String>>()?;
            let req = ImageRequest {
                model,
                prompt: s.prompt,
                width: s.width.unwrap_or(d.width),
                height: s.height.unwrap_or(d.height),
                steps: s.steps.unwrap_or(d.steps),
                seed: s.seed,
                i8_matmul: s.i8_matmul.unwrap_or(true),
                full_precision: s.full_precision.unwrap_or(false),
                input_images,
                keep_size: resolve_keep_size(s.keep_size, s.width, s.height),
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
            Ok(Built::new(JobRequest::Image(req), disable_coopmat))
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
            // FastH3-Omni reference images: RAM only, never on disk.
            let ref_images = s
                .ref_images
                .unwrap_or_default()
                .iter()
                .map(|b64| b64_decode("ref_images", b64).map(ImageBytes))
                .collect::<Result<Vec<_>, String>>()?;
            let ref_audios = s
                .ref_audios
                .unwrap_or_default()
                .iter()
                .map(|b64| b64_decode("ref_audios", b64).map(AudioBytes))
                .collect::<Result<Vec<_>, String>>()?;
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
                ref_images,
                ref_audios,
                ref_image_size: s.ref_image_size.unwrap_or_default(),
                ref_crop: s.ref_crop.unwrap_or_default(),
                sampler: s.sampler.unwrap_or(model.default_sampler()),
                steps: s.steps.unwrap_or(model.default_steps()),
                attn_window: s.attn_window,
                vae: s.vae.unwrap_or(model.default_vae()),
                encoder: s.encoder.unwrap_or_default(),
                i8_matmul: s.i8_matmul.unwrap_or(true),
                attn: s.attn.unwrap_or(model.default_attn()),
                audio: s.audio.unwrap_or(true),
                upscale: s.upscale.unwrap_or(model.two_stage_default()),
                super_res: s.super_res,
                audio_hq: s.audio_hq.unwrap_or(false),
                vae_encode_f32: s.vae_encode_f32.unwrap_or(false),
                bf16_gemm: s.bf16_gemm.unwrap_or(false),
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
            Ok(Built::new(
                JobRequest::Video(Box::new(req)),
                disable_coopmat,
            ))
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
                (Some(uid), _) => uploads.consume(&uid, job_id, dir, key)?,
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
                    diffusion: {
                        let d = thinfer_app::request::DiffusionSwapOptions::default();
                        thinfer_app::request::DiffusionSwapOptions {
                            steps: s.steps.unwrap_or(d.steps),
                            guide_scale: s.guide_scale.unwrap_or(d.guide_scale),
                            seed: s.seed,
                            width: s.width.unwrap_or(d.width),
                            height: s.height.unwrap_or(d.height),
                            cfg_every: s.cfg_every.unwrap_or(d.cfg_every),
                            face_crop: s.face_crop.unwrap_or(d.face_crop),
                        }
                    },
                },
            };
            req.validate()?;
            Ok(Built::new(JobRequest::FaceSwap(req), disable_coopmat))
        }
    }
}
