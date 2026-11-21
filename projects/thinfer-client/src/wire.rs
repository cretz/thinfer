//! The client/server wire types: the JSON job specs a client sends, the
//! responses it gets back, and the SSE event payloads. Both ends share this
//! one definition: `thinfer-serve` deserializes specs + serializes events,
//! and [`crate::Client`] does the mirror. The server keeps the
//! parts a client never sees -- artifact paths, budgets, the in-memory job
//! store -- next to its handlers.
//!
//! The OpenAPI `ToSchema` derives ride the `utoipa` feature (the server's).

use serde::{Deserialize, Serialize};

use crate::model::{
    AttnPrecision, EncoderQuant, ImageModelId, RefCrop, RefImageSize, RewriteQuality, SuperRes,
    SwapModel, TextModelId, VaeChoice, VideoModelId, VideoSampler,
};
use crate::progress::Stage;

/// What a client submits to `POST /jobs`: a [`JobSpec`] JSON sealed under a
/// per-job key the server does not have yet (`crate::jobkey` layout, base64).
/// The server queues only this ciphertext; when a worker is ready it emits
/// [`JobEvent::KeyRequired`] and the client answers with [`JobKeyDelivery`].
#[derive(Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct SealedJob {
    pub sealed: String,
    /// `POST /uploads` ids the sealed spec references, in the clear so the
    /// server can hold them for this job while it waits in the queue (exempt
    /// from the upload TTL). Opaque ids the server issued; they reveal nothing
    /// about the content. See [`JobSpec::upload_ids`].
    #[serde(default)]
    pub uploads: Vec<String>,
    /// Queue priority: higher runs first (default 0). Waiting raises it by
    /// one level per `queue_aging_secs`, so low-priority work never starves.
    #[serde(default)]
    pub priority: i32,
    /// Base64 SHA-256 of the job key (reveals nothing about it). When set,
    /// managing the job (cancel, `PATCH`, `DELETE`) requires it in the
    /// `x-job-key-sha256` header, and the key delivered later must match it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_proof: Option<String>,
    /// Run next, ahead of the queue: when no worker is free, the
    /// lowest-priority running job that can pause pauses this way and goes
    /// back in the queue, resuming after this job. Any job may preempt: a
    /// pause loses no work (like a higher `priority`, it only reorders).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preempt: Option<PauseMode>,
}

/// How a running job pauses. Encode, each denoise step, and the VAE decode
/// are steps; a paused job keeps its state from its last finished step
/// (sealed under its key, on disk), gives up the GPU and its memory, and
/// resumes from there: the result is the same as an uninterrupted run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub enum PauseMode {
    /// When the step in flight finishes.
    #[default]
    StepEnd,
    /// As soon as the job can stop: the step in flight is abandoned and runs
    /// again on resume.
    Now,
}

/// `POST /jobs/{id}/pause` body.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct JobPause {
    #[serde(default)]
    pub mode: PauseMode,
    /// Hold the job until `POST /jobs/{id}/resume` (default: back in the
    /// queue as soon as it has paused).
    #[serde(default)]
    pub hold: bool,
    /// Its queue priority from here on (default: unchanged).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<i32>,
}

/// `POST /jobs/{id}/resume` body.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct JobResume {
    /// Its queue priority from here on (default: unchanged).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<i32>,
}

/// `PATCH /jobs/{id}` body.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct JobPatch {
    /// New queue priority (queued jobs only).
    pub priority: i32,
}

/// `POST /jobs/{id}/key` body: the base64 job key, sent only in answer to
/// [`JobEvent::KeyRequired`]. Never logged (no `Debug`).
#[derive(Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct JobKeyDelivery {
    pub key: String,
}

/// A job request from a client, as sealed inside [`SealedJob`]. Internally tagged by `kind`. Carries only what a
/// caller chooses; the server assigns the artifact path and pulls budgets from
/// its config.
#[derive(Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum JobSpec {
    Image(ImageSpec),
    Text(TextSpec),
    Video(VideoSpec),
    FaceSwap(FaceSwapSpec),
}

impl JobSpec {
    /// The `POST /uploads` ids this spec references (sent in the clear beside
    /// the sealed spec; see [`SealedJob::uploads`]).
    pub fn upload_ids(&self) -> Vec<String> {
        let id = match self {
            JobSpec::Image(_) | JobSpec::Text(_) | JobSpec::Video(_) => None,
            JobSpec::FaceSwap(s) => s.input_video_upload.as_ref(),
        };
        id.into_iter().cloned().collect()
    }

    /// Per-request opt-out of the cooperative-matrix (tensor-core) path. `None`
    /// = use the server default. `Some(true)` forces the dense/i8 kernels even
    /// on a coopmat-capable GPU (A/B, debugging). A no-op when the GPU lacks
    /// coopmat support (it falls back regardless).
    pub fn disable_coopmat(&self) -> Option<bool> {
        match self {
            JobSpec::Image(s) => s.disable_coopmat,
            JobSpec::Text(_) => None,
            JobSpec::Video(s) => s.disable_coopmat,
            JobSpec::FaceSwap(s) => s.disable_coopmat,
        }
    }
}

/// One chat turn of a [`TextSpec`]. No `Debug`: content is never logged.
#[derive(Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct TextMessage {
    /// `system`, `user`, `assistant` or `tool` (a tool call's result).
    pub role: String,
    /// The text (a `tool` turn: the call's result, as the model should read
    /// it).
    #[serde(default)]
    pub content: String,
    /// An `assistant` turn's tool calls (as a reply returned them).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<crate::tools::ToolCall>,
    /// A `tool` turn: the id of the call it answers...
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// ...and that call's tool name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl TextMessage {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }

    /// The assistant turn of a reply that made `calls` (as the reply
    /// returned them), for resending the conversation statelessly.
    pub fn assistant_calls(calls: Vec<crate::tools::ToolCall>) -> Self {
        let mut m = Self::new("assistant", "");
        m.tool_calls = calls;
        m
    }

    /// The result of running `call`, as the model should read it.
    pub fn tool_result(call: &crate::tools::ToolCall, content: impl Into<String>) -> Self {
        let mut m = Self::new("tool", content);
        m.tool_call_id = Some(call.id.clone());
        m.name = Some(call.name.clone());
        m
    }
}

/// A chat reply request (LLM text generation). Unset sampling fields take the
/// model card's defaults for the chosen mode.
#[derive(Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct TextSpec {
    pub model: Option<TextModelId>,
    /// The conversation, oldest first.
    pub messages: Vec<TextMessage>,
    /// Reply token budget. `0` with a `session`: a prefix-only request that
    /// processes `messages` (system / user turns) into a node and generates
    /// nothing; requests continuing from that node send the turns that
    /// follow (same `tools` and `think`).
    pub max_tokens: Option<u32>,
    /// Let the model reason before it answers (thinking mode).
    pub think: Option<bool>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub seed: Option<u64>,
    /// Constrain the reply (after any thinking) to a JSON Schema or a Lark
    /// grammar. Forced runs (fixed keys, punctuation) are inserted in one pass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<crate::text_format::TextFormat>,
    /// Continue a server-held session (`POST /sessions`) instead of
    /// re-sending the conversation: `messages` then holds only the new turns.
    /// The reply's state becomes a new node (`TextResult.sessionNode`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<TextSessionRef>,
    /// Tools the model may call. Every call names one of them and its
    /// arguments match the tool's schema (constrained decoding in the
    /// model's own call syntax). A reply of calls ends the job with
    /// `finishReason: toolCalls` and the calls in `TextResult.toolCalls`;
    /// run them, then continue with one `tool` turn per call (send the tools
    /// again). With a session, continue from the reply's node with just the
    /// `tool` turns.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<crate::tools::ToolDef>,
    /// Default `auto`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<crate::tools::ToolChoice>,
}

impl TextSpec {
    /// A reply to `messages` (oldest first) with the model's defaults.
    pub fn new(messages: Vec<TextMessage>) -> Self {
        Self {
            messages,
            ..Self::default()
        }
    }
}

/// A text job's session reference. Rides inside the sealed spec, so the
/// session key reaches the server only when the job runs.
#[derive(Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct TextSessionRef {
    pub id: String,
    /// Node to continue from; `None` = a new root (send the whole
    /// conversation, system prompt included).
    #[serde(default)]
    pub from: Option<String>,
    /// Base64 32-byte AES-256-GCM session key (client-held; the server's
    /// copy of the session is sealed under it).
    pub key: String,
}

/// `POST /sessions` body.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct CreateSession {
    /// The text model every node is built with (default: the text default).
    #[serde(default)]
    pub model: Option<TextModelId>,
    /// Idle time before the server forgets the session (default
    /// `session_ttl_secs`).
    #[serde(default)]
    pub ttl_secs: Option<u64>,
    /// Base64 SHA-256 of the session key: managing the session requires it
    /// (`x-session-key-sha256`), and jobs must use the matching key.
    pub key_proof: String,
}

/// `PATCH /sessions/{id}` body.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct SessionPatch {
    pub ttl_secs: u64,
}

/// A session's structure (no content: every node's state is sealed).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct SessionInfo {
    pub id: String,
    pub model: TextModelId,
    pub ttl_secs: u64,
    /// Seconds until the server forgets it if unused.
    pub expires_in_secs: u64,
    /// Sealed bytes held across all nodes.
    pub bytes: u64,
    pub nodes: Vec<SessionNodeInfo>,
}

/// One node: the state one text request added (positions `start..end`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct SessionNodeInfo {
    pub id: String,
    pub parent: Option<String>,
    pub start: u32,
    pub end: u32,
    pub bytes: u64,
    pub age_secs: u64,
    /// Ends without a reply: a prefix node (`maxTokens: 0`), continued with
    /// new turns.
    #[serde(default)]
    pub open: bool,
}

#[derive(Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct ImageSpec {
    pub model: Option<ImageModelId>,
    pub prompt: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub steps: Option<u32>,
    pub seed: Option<u64>,
    /// DP4A i8 matmul on the DP4A-safe DiT sites (Ideogram-4 only). Default true.
    pub i8_matmul: Option<bool>,
    /// Run a bf16-source Z-Image DiT (`zimage-turbo-bf16`) at full bf16
    /// precision instead of the default load-time Q8 transcode (~6x slower, max
    /// precision). Default false. No effect on GGUF variants / other models.
    #[serde(default)]
    pub full_precision: Option<bool>,
    /// Base64-encoded input images (PNG/JPEG), in prompt order: 1 to 3 for
    /// qwen-image-edit-rapid (the prompt names them "Picture 1", ...), up to
    /// 10 condition images for qwen-image-2.1-turbo, rejected otherwise. Held
    /// in RAM server-side.
    #[serde(default)]
    pub input_images: Vec<String>,
    /// With input images (qwen-image-2.1-turbo; other models always render
    /// the requested size): true = exactly the requested width x height,
    /// false = the last input's aspect ratio at that pixel area. Unset =
    /// true when `width` or `height` is set, else false.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_size: Option<bool>,
    /// Opt out of the cooperative-matrix (tensor-core) path; see
    /// [`JobSpec::disable_coopmat`].
    #[serde(default)]
    pub disable_coopmat: Option<bool>,
    /// User adapters (vault LoRAs) to fold into the DiT for this generation, in
    /// fold order. Each `id` is a `vault list` entry for THIS model. Empty/absent
    /// = the plain base model. Only models that support adapters accept a
    /// non-empty list; requires `password`.
    #[serde(default)]
    pub lora: Vec<LoraSpec>,
    /// Vault password, required when `lora` is non-empty. Transient: used to
    /// decrypt the adapters for this request only, never stored or logged.
    #[serde(default)]
    pub password: Option<String>,
}

/// One adapter reference in a generate request: a vault entry id + optional blend
/// weight (defaults to 1.0).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct LoraSpec {
    pub id: String,
    #[serde(default)]
    pub weight: Option<f32>,
}

impl ImageSpec {
    /// A `prompt` image with the model's defaults.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            ..Self::default()
        }
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct VideoSpec {
    pub model: Option<VideoModelId>,
    /// One prompt, or several for chained FastH3 / MiniMax-H3 clips.
    pub prompts: Vec<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    /// Frame counts (one for the whole clip, or one per chained-clip prompt).
    /// Mutually exclusive with `durations`.
    pub frames: Option<Vec<u32>>,
    pub durations: Option<Vec<f32>>,
    pub fps: Option<u32>,
    pub seed: Option<u64>,
    /// FastWan denoise sampler. Unset -> the model default (UniPC, matching the
    /// public Spaces; DMD for FastWan2.1-1.3B, its card's 3-step regime). DMD is
    /// also the byte-parity reference path.
    pub sampler: Option<VideoSampler>,
    /// UniPC denoise steps (1..=8, default 4). DMD ignores it.
    pub steps: Option<u32>,
    /// Temporal self-attention window radius in LATENT frames (Wan2.2 14B and
    /// Hunyuan 1.5). When set, DiT self-attention is restricted to keys within
    /// `±N` latent frames, breaking the O(frames^2) cost on long clips at the
    /// price of long-range temporal coherence. `None`/omitted = the model
    /// default (Wan2.2-14B: 3; Hunyuan and others: full attention). `0` = full
    /// attention. Ignored by models without a windowed path.
    #[serde(default)]
    pub attn_window: Option<u32>,
    pub vae: Option<VaeChoice>,
    /// Base64-encoded first-frame image bytes (PNG/JPEG). OPTIONAL on
    /// hunyuan-video-1.5-ti2v: with it the run is image-conditioned (I2V),
    /// without it the model generates text-only. Rejected by every other model.
    /// The server decodes it to a temp file under the job dir like the
    /// image-edit path.
    #[serde(default)]
    pub input_image: Option<String>,
    /// LTX native-I2V frame-0 conditioning strength (`0.0..=1.0`; default 1.0).
    /// `1.0` locks the encoded input frame through the denoise; lower lets it
    /// drift. Used only with `input_image` on ltx2-rapid; ignored otherwise.
    #[serde(default)]
    pub strength: Option<f32>,
    /// Base64-encoded reference images (PNG/JPEG). FastH3-Omni only (at most
    /// 9): the prompt names them `<Picture 1>`, `<Picture 2>`, ... in order.
    /// Held in RAM by the server, never written to disk.
    #[serde(default)]
    pub ref_images: Option<Vec<String>>,
    /// FastH3-Omni: how each reference image is sized. `match` (default, as in
    /// ComfyUI: the canvas's pixel area, never upscaled), `max` (short edge
    /// capped at 2048) or `diffusers` (short edge exactly 2048, the reference;
    /// ~2.4x slower with one reference).
    #[serde(default)]
    pub ref_image_size: Option<RefImageSize>,
    /// FastH3-Omni: `auto` (default) crops each reference in on its faces
    /// when they would come out small (a person photographed from afar);
    /// `off` always keeps the whole image, as the reference does.
    #[serde(default)]
    pub ref_crop: Option<RefCrop>,
    /// LTX-2.3 text-encoder quantization: `q8` (default, conditioning-quality
    /// baseline) or `q4` (Q4_K_M, ~2.8x faster encode, lower precision). Applies
    /// to all LTX models; ignored by Wan.
    pub encoder: Option<EncoderQuant>,
    pub i8_matmul: Option<bool>,
    /// MiniMax-H3 family attention precision: `int8-fp8` (default), `int8`
    /// or `bf16` (the reference: slower, same quality, different sample).
    pub attn: Option<AttnPrecision>,
    /// Decode + mux an audio track. Joint-AV models only (LTX and MiniMax-H3;
    /// the silent Wan models ignore it). Defaults to `true`; `false` skips the
    /// audio tail for a video-only, faster MP4.
    pub audio: Option<bool>,
    /// LTX-2.3 distilled only: opt in to the 2x spatial-upscale refine path
    /// (half-res denoise -> latent upscale -> refine). Defaults to `false` =
    /// single-stage denoise at the target res. Ignored by the Wan models.
    pub upscale: Option<bool>,
    /// MiniMax-H3 / FastH3: RelayVSR super-resolution of the finished video,
    /// `2x` or `4x`. Absent = native resolution.
    pub super_res: Option<SuperRes>,
    /// LTX joint-AV: opt in to a finer denoise schedule for better audio
    /// (subdivides the coarse distilled sigma jumps into more Euler substeps).
    /// Honored on every LTX path: the single-stage schedule (ltx2-rapid always,
    /// the 22B line with `upscale` off) and both stage schedules on the two-stage
    /// 22B line (8+3 -> 16+11 steps, ~2.5x denoise). Wan ignores it. Defaults to
    /// `false`.
    #[serde(default)]
    pub audio_hq: Option<bool>,
    /// MiniMax-H3 / FastH3: the reference / first-frame VAE encode in f32
    /// (the exact path) instead of bf16 on tensor cores (the reference's
    /// precision, ~3x faster). Defaults to `false`.
    #[serde(default)]
    pub vae_encode_f32: Option<bool>,
    /// MiniMax-H3 GGUF models: every block GEMM in bf16 with unfolded adapters
    /// (the exact path) instead of the default 8-bit GEMMs and folded
    /// adapters. Defaults to `false`.
    #[serde(default)]
    pub bf16_gemm: Option<bool>,
    /// HunyuanVideo 1.5 only: expand a short prompt into a detailed caption via
    /// the rewrite endpoint before encoding. Defaults to `true` (the model needs
    /// detailed captions); `false` sends the raw prompt. Other models ignore it.
    #[serde(default)]
    pub rewrite: Option<bool>,
    /// HunyuanVideo 1.5 only: which rewriter model to run when `rewrite` is set.
    /// `fast` (default) = the ~2.5GB Qwen3-VL-4B (budget-honest), `full` = the
    /// ~5.85GB Qwen3-VL-8B. Ignored when `rewrite` is false or by other models.
    #[serde(default)]
    pub rewrite_quality: Option<RewriteQuality>,
    /// Opt out of the cooperative-matrix (tensor-core) path; see
    /// [`JobSpec::disable_coopmat`].
    #[serde(default)]
    pub disable_coopmat: Option<bool>,
    /// User adapters (vault LoRAs) to fold into the DiT for this generation, in
    /// fold order. Each `id` is a `vault list` entry for THIS model. Empty/absent
    /// = the plain base model. Only models that support adapters (AnyFlow,
    /// FastWan2.1-1.3B) accept a non-empty list; requires `password`.
    #[serde(default)]
    pub lora: Vec<LoraSpec>,
    /// Vault password, required when `lora` is non-empty. Transient: used to
    /// decrypt the adapters for this request only, never stored or logged.
    #[serde(default)]
    pub password: Option<String>,
}

impl VideoSpec {
    /// A one-prompt video with the model's defaults.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompts: vec![prompt.into()],
            ..Self::default()
        }
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct FaceSwapSpec {
    pub model: Option<SwapModel>,
    /// Local path to the input video (the server reads it directly; localhost
    /// deployments only). Mutually exclusive with `input_video_b64`.
    #[serde(default)]
    pub input_video: Option<String>,
    /// Local path to the source face image. Mutually exclusive with
    /// `source_image_b64`.
    #[serde(default)]
    pub source_image: Option<String>,
    /// Base64-encoded input VIDEO bytes (mp4), for browser uploads. Held
    /// RAM-first; a large upload spills to an encrypted on-disk blob under a
    /// per-request ephemeral key (never plaintext on disk). Takes precedence over
    /// `input_video` when both are set. Prefer `input_video_upload` for large
    /// clips: base64-in-JSON inflates the body ~33% and hits the JSON size cap.
    #[serde(default)]
    pub input_video_b64: Option<String>,
    /// Id of a video previously streamed to `POST /uploads`. The raw mp4 rides
    /// its own request body (no base64, no JSON size cap); the server holds it
    /// RAM-first / encrypted-spill under a TTL until this job consumes it. Takes
    /// precedence over `input_video_b64` and `input_video`.
    #[serde(default)]
    pub input_video_upload: Option<String>,
    /// Base64-encoded source FACE image bytes (PNG/JPEG), for browser uploads.
    /// Takes precedence over `source_image` when both are set.
    #[serde(default)]
    pub source_image_b64: Option<String>,
    /// Opt out of the cooperative-matrix (tensor-core) path; see
    /// [`JobSpec::disable_coopmat`].
    #[serde(default)]
    pub disable_coopmat: Option<bool>,
    /// Intersect the paste mask with an XSeg occlusion mask so occluders
    /// (hands/hair/glasses) crossing the face show the original frame through.
    /// Adds one ONNX forward per face.
    #[serde(default)]
    pub occlusion_mask: Option<bool>,
    /// Intersect the paste mask with HyperSwap's own confidence mask (free).
    #[serde(default)]
    pub hyperswap_mask: Option<bool>,
    /// Run the GFPGAN face enhancer on each swapped face.
    #[serde(default)]
    pub enhance: Option<bool>,
    /// Detect faces every Nth frame, reusing the previous detection in between.
    /// 1 = every frame (default). Higher = faster, looser tracking.
    #[serde(default)]
    pub detect_stride: Option<u32>,
    /// Output H.264 bitrate as a multiple of the source video bitrate (default
    /// 1.15). Falls back to a bits-per-pixel target if the source bitrate is
    /// unknown.
    #[serde(default)]
    pub bitrate_scale: Option<f32>,
    /// Only swap+output the source window starting at this many seconds (rebased
    /// to 0). Default: clip start.
    #[serde(default)]
    pub start_secs: Option<f32>,
    /// Only swap+output the source window ending at this many seconds. Default:
    /// clip end.
    #[serde(default)]
    pub end_secs: Option<f32>,
    /// dreamid-v: denoise steps (default 16).
    #[serde(default)]
    pub steps: Option<u32>,
    /// dreamid-v: image-CFG guidance scale on the source face (default 4).
    #[serde(default)]
    pub guide_scale: Option<f32>,
    /// dreamid-v: seed (default random).
    #[serde(default)]
    pub seed: Option<u64>,
    /// dreamid-v: working size; the clip is resized toward this pixel area,
    /// its aspect kept (default 832x480, multiples of 16).
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    /// dreamid-v opt-in: the unguided forward every N steps (default 1).
    #[serde(default)]
    pub cfg_every: Option<u32>,
    /// dreamid-v opt-in: run on the face region only, pasted back at the
    /// source resolution.
    #[serde(default)]
    pub face_crop: Option<bool>,
}

/// `GET /capabilities`: what the server's GPU supports.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct Capabilities {
    /// GPU supports the coopmat (tensor-core) matmul path.
    pub coopmat: bool,
}

/// The `POST /jobs` response.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct CreateResponse {
    pub id: String,
    pub queue_position: usize,
}

/// The `POST /uploads` response: an opaque id a job spec references via
/// `input_video_upload`, plus the received byte count.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct UploadResponse {
    pub id: String,
    pub size: usize,
}

/// Coarse job lifecycle state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "lowercase")]
pub enum JobStateKind {
    Queued,
    /// Taken by a worker: waiting for the key, then opening the spec.
    Starting,
    Running,
    /// Paused and held: out of the queue until resumed.
    Paused,
    Done,
    Failed,
    Cancelled,
}

/// Serializable mirror of [`Stage`] (the wire shape of a progress event).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(tag = "stage", rename_all = "camelCase")]
pub enum ProgressStage {
    /// The job has a worker and is loading its model (see [`Stage::Loading`]).
    Loading,
    TextEncode,
    Step {
        i: u32,
        n: u32,
    },
    // Enum-level `rename_all` renames the variant tag, not struct-variant
    // fields, so camelCase the fields explicitly (clients read `numChunks` /
    // `numSteps`).
    #[serde(rename_all = "camelCase")]
    ChunkStep {
        chunk: u32,
        num_chunks: u32,
        step: u32,
        num_steps: u32,
    },
    VaeDecode,
    FrameSwapped {
        done: u32,
        total: u32,
    },
}

impl From<Stage> for ProgressStage {
    fn from(s: Stage) -> Self {
        match s {
            Stage::Loading => ProgressStage::Loading,
            Stage::TextEncode => ProgressStage::TextEncode,
            Stage::Step { i, n } => ProgressStage::Step { i, n },
            Stage::ChunkStep {
                chunk,
                num_chunks,
                step,
                num_steps,
            } => ProgressStage::ChunkStep {
                chunk,
                num_chunks,
                step,
                num_steps,
            },
            Stage::VaeDecode => ProgressStage::VaeDecode,
            Stage::FrameSwapped { done, total } => ProgressStage::FrameSwapped { done, total },
        }
    }
}

impl From<ProgressStage> for Stage {
    fn from(s: ProgressStage) -> Self {
        match s {
            ProgressStage::Loading => Stage::Loading,
            ProgressStage::TextEncode => Stage::TextEncode,
            ProgressStage::Step { i, n } => Stage::Step { i, n },
            ProgressStage::ChunkStep {
                chunk,
                num_chunks,
                step,
                num_steps,
            } => Stage::ChunkStep {
                chunk,
                num_chunks,
                step,
                num_steps,
            },
            ProgressStage::VaeDecode => Stage::VaeDecode,
            ProgressStage::FrameSwapped { done, total } => Stage::FrameSwapped { done, total },
        }
    }
}

/// What a finished job produced (the SSE `done` payload + status `result`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct JobResult {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: Option<u32>,
    pub seed: Option<u64>,
    /// Relative URL to fetch the artifact bytes.
    pub result_url: String,
    /// Text jobs: token counts and speed (the reply itself is the artifact).
    #[serde(default)]
    pub text: Option<TextResult>,
}

/// Token counts and timings of a finished text job.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct TextResult {
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    /// The session node this reply created (session jobs only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_node: Option<String>,
    #[serde(default)]
    pub finish_reason: crate::tools::FinishReason,
    /// With `finishReason: toolCalls`: the calls, a JSON array of
    /// `ToolCall`, sealed under the job key (`crate::jobkey` blob, base64).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<String>,
}

/// Snapshot of a job for `GET /jobs/{id}`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct JobStatus {
    pub id: String,
    pub state: JobStateKind,
    pub priority: i32,
    pub queue_position: Option<usize>,
    pub progress: Option<ProgressStage>,
    pub result: Option<JobResult>,
    pub error: Option<String>,
    /// Whether the job can pause now: it runs a model that can, and is not
    /// already pausing.
    #[serde(default)]
    pub pausable: bool,
}

/// One `GET /jobs` row: structural fields only (the spec is sealed; no result
/// or error text).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct JobListEntry {
    pub id: String,
    pub state: JobStateKind,
    pub priority: i32,
    /// Run-order index while queued.
    pub queue_position: Option<usize>,
    pub progress: Option<ProgressStage>,
    /// Seconds since submit.
    pub age_secs: u64,
    /// See [`JobStatus::pausable`].
    #[serde(default)]
    pub pausable: bool,
}

/// `GET /jobs` body: pending jobs in run order, then the rest newest first.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct JobList {
    pub jobs: Vec<JobListEntry>,
}

/// One SSE event payload. `type` tags the variant; the SSE `event:` name is
/// [`JobEvent::kind`]. The server serializes these; a `RemoteExecutor` parses
/// them back.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum JobEvent {
    Queued {
        position: usize,
    },
    /// A worker is ready: deliver the job key now (`POST /jobs/{id}/key`).
    /// The job decrypts and runs only after it arrives.
    KeyRequired,
    Started,
    Progress {
        stage: ProgressStage,
    },
    Log {
        message: String,
    },
    /// A streamed piece of a text job's reply, sealed under the job key
    /// (`crate::jobkey` blob, base64): reply text never crosses the server in
    /// the clear.
    Text {
        sealed: String,
    },
    Done {
        result: JobResult,
    },
    Error {
        message: String,
    },
    Cancelled,
    /// The job paused (see [`PauseMode`]) and gave up its worker. It resumes
    /// from its last finished step, asking for its key again
    /// ([`JobEvent::KeyRequired`]). `held`: it waits for `POST
    /// /jobs/{id}/resume`; else it is back in the queue (a `queued` event
    /// follows).
    Paused {
        held: bool,
    },
}

impl JobEvent {
    pub fn kind(&self) -> &'static str {
        match self {
            JobEvent::Queued { .. } => "queued",
            JobEvent::KeyRequired => "keyRequired",
            JobEvent::Started => "started",
            JobEvent::Progress { .. } => "progress",
            JobEvent::Log { .. } => "log",
            JobEvent::Text { .. } => "text",
            JobEvent::Done { .. } => "done",
            JobEvent::Error { .. } => "error",
            JobEvent::Cancelled => "cancelled",
            JobEvent::Paused { .. } => "paused",
        }
    }

    /// Terminal events end the SSE stream.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            JobEvent::Done { .. } | JobEvent::Error { .. } | JobEvent::Cancelled
        )
    }
}

/// `GET /models` body: every model this server runs.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct ModelList {
    pub models: Vec<ModelInfo>,
}

/// What a model runs, takes and makes.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    /// The job spec's `model` value.
    pub id: String,
    pub kind: ModelKind,
    /// One-line description.
    pub title: String,
    /// The model a spec without `model` runs.
    pub default: bool,
    /// Every weight file its default job needs is on the server already (else
    /// the first job downloads them, when the server allows it).
    pub downloaded: bool,
    /// Its jobs can pause (`POST /jobs/{id}/pause`).
    pub pausable: bool,
    /// The images it takes; empty: none.
    pub inputs: Vec<InputSlot>,
    /// Image / video models: what sets the output size, and its rules.
    pub size: Option<SizeRules>,
    /// Video models.
    pub video: Option<VideoInfo>,
}

/// A model family, as a job spec's `kind` names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub enum ModelKind {
    Text,
    Image,
    Video,
    FaceSwap,
}

/// One kind of image input a model takes.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct InputSlot {
    /// The spec field that carries it (`inputImages`, `inputImage`,
    /// `refImages`, `sourceImage`).
    pub field: String,
    pub used_as: InputUse,
    pub min: u32,
    pub max: u32,
    /// How a prompt names the n-th image (`{n}` from 1), when it can.
    pub prompt_tag: Option<String>,
}

/// What a model does with an input image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub enum InputUse {
    /// Conditioning for an edit or a composition (several, in prompt order).
    Condition,
    /// The one image an edit changes.
    Edit,
    /// The video's first frame.
    FirstFrame,
    /// Named references (people, objects, styles) the video draws on.
    Reference,
    /// The face to swap in.
    Face,
}

/// What decides the output size, and the sizes allowed.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct SizeRules {
    /// Width and height must be multiples of this.
    pub multiple: u32,
    pub default_width: u32,
    pub default_height: u32,
    /// With input images, the output can take the last input's aspect ratio
    /// at the requested pixel area: when the spec sets no size, or sets
    /// `keepSize: false`.
    pub follows_last_input: bool,
}

/// Video limits that matter to callers.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct VideoInfo {
    /// Playback frame rate (fixed when `fixedFps`).
    pub fps: u32,
    pub fixed_fps: bool,
    /// At the default size, the most frames one clip can have on this
    /// server's GPU budget (larger sizes allow fewer); `None`: set by the
    /// prompt's length (a longer one leaves fewer).
    pub max_frames: Option<u32>,
    /// Legal clip lengths: `frameStep * n + frameOffset` frames (`n >= 0`).
    /// Other requests are moved onto this grid (see `framesRoundUp`).
    pub frame_step: u32,
    pub frame_offset: u32,
    /// Off-grid lengths round UP to the next legal one (else to the nearest).
    pub frames_round_up: bool,
    /// Several prompts render as chained clips (each starts on the previous
    /// one's last frame).
    pub chained_clips: bool,
    /// It renders an audio track.
    pub audio: bool,
}
