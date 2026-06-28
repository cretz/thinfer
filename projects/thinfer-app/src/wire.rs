//! The client/server wire types: the JSON job specs a client sends, the
//! responses it gets back, and the SSE event payloads. They live here (not in
//! `thinfer-serve`) so both ends share one definition: `thinfer-serve`
//! deserializes specs + serializes events, and the `RemoteExecutor` (the HTTP
//! client, behind the `remote` feature) does the mirror. The server keeps the
//! parts a client never sees -- artifact paths, budgets, the in-memory job
//! store -- next to its handlers.
//!
//! Gated behind `serde` (no consumer wants these without it). The OpenAPI
//! `ToSchema` derives ride the `serve` feature so a plain `serde` client (the
//! CLI's `RemoteExecutor`) does not pull `utoipa`.

use serde::{Deserialize, Serialize};

use crate::model::{EncoderQuant, ImageModelId, SwapModel, VaeChoice, VideoModelId, VideoSampler};
use crate::progress::Stage;

/// A job request from a client. Internally tagged by `kind`. Carries only what a
/// caller chooses; the server assigns the artifact path and pulls budgets from
/// its config.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum JobSpec {
    Image(ImageSpec),
    Video(VideoSpec),
    FaceSwap(FaceSwapSpec),
}

impl JobSpec {
    /// True if this reads a large local input (face-swap video): a server runs
    /// it now-or-rejects rather than queuing.
    pub fn is_large_input(&self) -> bool {
        matches!(self, JobSpec::FaceSwap(_))
    }

    /// The client's result-encryption public key, if it asked for encryption.
    pub fn public_key(&self) -> Option<&str> {
        match self {
            JobSpec::Image(s) => s.public_key.as_deref(),
            JobSpec::Video(s) => s.public_key.as_deref(),
            JobSpec::FaceSwap(s) => s.public_key.as_deref(),
        }
    }

    /// Per-request opt-out of the cooperative-matrix (tensor-core) path. `None`
    /// = use the server default. `Some(true)` forces the dense/i8 kernels even
    /// on a coopmat-capable GPU (A/B, debugging). A no-op when the GPU lacks
    /// coopmat support (it falls back regardless).
    pub fn disable_coopmat(&self) -> Option<bool> {
        match self {
            JobSpec::Image(s) => s.disable_coopmat,
            JobSpec::Video(s) => s.disable_coopmat,
            JobSpec::FaceSwap(s) => s.disable_coopmat,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
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
    /// Base64-encoded reference image bytes (PNG/JPEG). REQUIRED for
    /// qwen-image-edit, ignored otherwise. The server decodes it to a temp file
    /// under the job dir and feeds it to the image-edit path.
    #[serde(default)]
    pub input_image: Option<String>,
    /// Base64 SPKI RSA-OAEP public key. When present, the server encrypts the
    /// result so only the holder of the matching private key can read it; absent
    /// = plaintext. See [`crate::wire`] / `thinfer-serve` crypto.
    #[serde(default)]
    pub public_key: Option<String>,
    /// Opt out of the cooperative-matrix (tensor-core) path; see
    /// [`JobSpec::disable_coopmat`].
    #[serde(default)]
    pub disable_coopmat: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct VideoSpec {
    pub model: Option<VideoModelId>,
    /// One prompt, or several for a multi-shot clip (LongLive only).
    pub prompts: Vec<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    /// Frame counts (one to split evenly, or one per prompt). Mutually exclusive
    /// with `durations`.
    pub frames: Option<Vec<u32>>,
    pub durations: Option<Vec<f32>>,
    pub fps: Option<u32>,
    pub seed: Option<u64>,
    /// FastWan denoise sampler. Defaults to UniPC (matches the public Spaces);
    /// DMD is the byte-parity reference path. Ignored on the AR (LongLive) path.
    pub sampler: Option<VideoSampler>,
    /// UniPC denoise steps (1..=8, default 4). DMD ignores it.
    pub steps: Option<u32>,
    pub vae: Option<VaeChoice>,
    /// LTX-2.3 text-encoder quantization: `q8` (default, conditioning-quality
    /// baseline) or `q4` (Q4_K_M, ~2.8x faster encode, lower precision). Applies
    /// to all LTX/Sulphur models; ignored by Wan.
    pub encoder: Option<EncoderQuant>,
    pub i8_matmul: Option<bool>,
    /// Decode + mux an audio track. LTX joint-AV only (silent Wan models ignore
    /// it). Defaults to `true`; `false` skips the audio tail for a video-only,
    /// faster MP4.
    pub audio: Option<bool>,
    /// LTX-2.3 distilled only: opt in to the 2x spatial-upscale refine path
    /// (half-res denoise -> latent upscale -> refine). Defaults to `false` =
    /// single-stage denoise at the target res. Ignored by the Wan models.
    pub upscale: Option<bool>,
    /// Base64 SPKI RSA-OAEP public key for result encryption (see [`ImageSpec`]).
    #[serde(default)]
    pub public_key: Option<String>,
    /// Opt out of the cooperative-matrix (tensor-core) path; see
    /// [`JobSpec::disable_coopmat`].
    #[serde(default)]
    pub disable_coopmat: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct FaceSwapSpec {
    pub model: Option<SwapModel>,
    /// Local path to the input video (the server reads it directly; localhost
    /// deployments only).
    pub input_video: String,
    /// Local path to the source face image.
    pub source_image: String,
    /// Base64 SPKI RSA-OAEP public key for result encryption (see [`ImageSpec`]).
    #[serde(default)]
    pub public_key: Option<String>,
    /// Opt out of the cooperative-matrix (tensor-core) path; see
    /// [`JobSpec::disable_coopmat`].
    #[serde(default)]
    pub disable_coopmat: Option<bool>,
}

/// The `POST /jobs` response.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct CreateResponse {
    pub id: String,
    pub queue_position: usize,
}

/// Coarse job lifecycle state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
#[serde(rename_all = "lowercase")]
pub enum JobStateKind {
    Queued,
    Running,
    Done,
    Failed,
    Cancelled,
}

/// Serializable mirror of [`Stage`] (the wire shape of a progress event).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
#[serde(tag = "stage", rename_all = "camelCase")]
pub enum ProgressStage {
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
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct JobResult {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: Option<u32>,
    pub seed: Option<u64>,
    /// Relative URL to fetch the artifact bytes.
    pub result_url: String,
}

/// Snapshot of a job for `GET /jobs/{id}`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct JobStatus {
    pub id: String,
    pub state: JobStateKind,
    pub queue_position: Option<usize>,
    pub progress: Option<ProgressStage>,
    pub result: Option<JobResult>,
    pub error: Option<String>,
}

/// One SSE event payload. `type` tags the variant; the SSE `event:` name is
/// [`JobEvent::kind`]. The server serializes these; a `RemoteExecutor` parses
/// them back.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum JobEvent {
    Queued { position: usize },
    Started,
    Progress { stage: ProgressStage },
    Log { message: String },
    Done { result: JobResult },
    Error { message: String },
    Cancelled,
}

impl JobEvent {
    pub fn kind(&self) -> &'static str {
        match self {
            JobEvent::Queued { .. } => "queued",
            JobEvent::Started => "started",
            JobEvent::Progress { .. } => "progress",
            JobEvent::Log { .. } => "log",
            JobEvent::Done { .. } => "done",
            JobEvent::Error { .. } => "error",
            JobEvent::Cancelled => "cancelled",
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
