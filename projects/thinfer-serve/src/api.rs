//! HTTP surface: the conversion of client-facing job specs into `thinfer_app`
//! requests and the axum handlers. The wire types (specs, responses, events)
//! live in `thinfer_app::wire` so the `RemoteExecutor` client shares them; the
//! server keeps what a client never sees -- artifact paths, budgets pulled from
//! config, and the in-memory job store.

use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path as AxPath, Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use thinfer_app::JobRequest;
use thinfer_app::model::{ImageModelId, SwapModel, VaeChoice, VideoModelId};
use thinfer_app::request::{
    FaceSwapRequest, ImageBytes, ImageFormat, ImageRequest, LoraRef, Secret, VideoFormat,
    VideoInput, VideoRequest,
};
use thinfer_app::wire::{CreateResponse, JobSpec, JobStateKind, JobStatus, UploadResponse};
use tokio::sync::broadcast::error::RecvError;

use crate::config::ServeConfig;
use crate::job::{JobStore, SeqEvent};
use crate::uploads::UploadStore;

#[derive(Clone)]
pub struct AppState {
    pub store: Arc<JobStore>,
    pub config: Arc<ServeConfig>,
    /// Whether this server's GPU exposes the cooperative-matrix (tensor-core)
    /// path. Probed once at startup; surfaced via `GET /capabilities` so the web
    /// UI can grey the coopmat toggle on hardware that can't accelerate.
    pub coopmat_supported: bool,
    /// The encrypted adapter (LoRA) vault, rooted at the configured dir. Shared
    /// with the CLI (same on-disk vault); every op re-derives the key from the
    /// request password.
    pub vault: Arc<thinfer_app::vault::Vault>,
    /// Pending streamed video uploads (RAM-first / encrypted-spill), keyed by id
    /// and reaped by TTL. A job spec references one via `input_video_upload`.
    pub uploads: Arc<UploadStore>,
}

/// `GET /capabilities` body: static server/GPU capabilities the UI adapts to.
#[derive(serde::Serialize)]
struct CapabilitiesResponse {
    /// GPU supports the coopmat (tensor-core) matmul path.
    coopmat: bool,
}

async fn capabilities(State(state): State<AppState>) -> Json<CapabilitiesResponse> {
    Json(CapabilitiesResponse {
        coopmat: state.coopmat_supported,
    })
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

/// Build the executable request for `spec`, placing the artifact under
/// `artifact_dir/<id>/`. Validates eagerly so a bad spec fails the POST. The
/// spec is the client wire shape (`thinfer_app::wire`); this is where the server
/// fills in what a client does not send (artifact path, budgets, output format).
#[allow(clippy::type_complexity)]
fn spec_into_request(
    spec: JobSpec,
    id: &str,
    config: &ServeConfig,
    uploads: &UploadStore,
) -> Result<(JobRequest, PathBuf, Option<String>, Option<bool>), String> {
    let budget = config.budget()?;
    let dir = config.artifact_dir.join(id);
    let mp4 = || dir.join("output.mp4");
    let make_dir =
        || std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()));
    let public_key = spec.public_key().map(str::to_string);
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
            let out = req.output.clone();
            Ok((JobRequest::Image(req), out, public_key, disable_coopmat))
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
            // DreamID-V target VIDEO: a streamed upload id (preferred) is
            // consumed from the store into the job dir; else a base64 payload is
            // decoded RAM-first with an encrypted spill for large clips. Either
            // way, raw plaintext never lands on disk.
            let input_video = match (s.input_video_upload, s.input_video) {
                (Some(uid), _) => Some(uploads.consume(&uid, &dir)?),
                (None, Some(b64)) => Some(video_input(b64_decode("input_video", &b64)?, &dir)?),
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
                sampler: s.sampler.unwrap_or_default(),
                steps: s.steps.unwrap_or(model.default_steps()),
                attn_window: s.attn_window,
                vae: s.vae.unwrap_or(VaeChoice::Tiny),
                encoder: s.encoder.unwrap_or_default(),
                i8_matmul: s.i8_matmul.unwrap_or(true),
                audio: s.audio.unwrap_or(true),
                upscale: s.upscale.unwrap_or(model.two_stage_default()),
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
            let out = req.output.clone();
            Ok((JobRequest::Video(req), out, public_key, disable_coopmat))
        }
        JobSpec::FaceSwap(s) => {
            // Target video, in precedence order: a streamed upload id (consumed
            // from the store into the job dir), then base64-in-JSON bytes
            // (RAM-first / encrypted spill), then a server-side path (localhost
            // deployments) read into RAM. Exactly one must be present.
            let input_video = match (s.input_video_upload, s.input_video_b64, s.input_video) {
                (Some(uid), _, _) => uploads.consume(&uid, &dir)?,
                (None, Some(b64), _) => video_input(b64_decode("input_video_b64", &b64)?, &dir)?,
                (None, None, Some(path)) => {
                    let bytes = std::fs::read(&path).map_err(|e| format!("read {path}: {e}"))?;
                    video_input(bytes, &dir)?
                }
                (None, None, None) => return Err("face-swap requires an input video".into()),
            };
            // Source face image: uploaded bytes, or a server-side path read into
            // RAM. Held in RAM only (never written to the job dir).
            let source_image = match (s.source_image_b64, s.source_image) {
                (Some(b64), _) => ImageBytes(b64_decode("source_image_b64", &b64)?),
                (None, Some(path)) => {
                    ImageBytes(std::fs::read(&path).map_err(|e| format!("read {path}: {e}"))?)
                }
                (None, None) => return Err("face-swap requires a source image".into()),
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
            let out = req.output.clone();
            Ok((JobRequest::FaceSwap(req), out, public_key, disable_coopmat))
        }
    }
}

/// The job API routes (everything under `/jobs` + the OpenAPI doc). When an
/// `auth_token` is configured these sit behind the Bearer check; the static web
/// UI is mounted separately so it can load and prompt for the token.
pub fn router(state: AppState) -> Router {
    let auth = state.config.auth_token.clone();
    let max_json = state.config.max_json_bytes;
    let max_upload = state.config.max_upload_bytes;
    Router::new()
        // Raw video upload: its own large body limit (route-level, so it wins
        // over the modest router-wide JSON limit below).
        .route(
            "/uploads",
            post(upload_video).layer(DefaultBodyLimit::max(max_upload)),
        )
        .route("/jobs", post(create_job))
        .route("/jobs/{id}", get(get_status))
        .route("/jobs/{id}/events", get(events))
        .route("/jobs/{id}/result", get(get_result))
        .route("/jobs/{id}/cancel", post(cancel))
        .route("/vault/adapters/list", post(crate::vault::list_adapters))
        .route("/vault/adapters/add", post(crate::vault::add_adapter))
        .route("/vault/adapters/remove", post(crate::vault::remove_adapter))
        .route("/capabilities", get(capabilities))
        .route("/openapi.json", get(openapi))
        // Router-wide cap for the JSON job spec (small base64 image only; big
        // video takes the /uploads path). `/uploads` overrides this per-route.
        .layer(DefaultBodyLimit::max(max_json))
        .layer(middleware::from_fn(move |req, next| {
            require_auth(auth.clone(), req, next)
        }))
        .with_state(state)
}

/// Bearer-token gate. A no-op when no token is configured; otherwise every
/// request must carry `Authorization: Bearer <token>`. `/openapi.json` is gated
/// too -- a deployment that wants it public can run without a token.
async fn require_auth(token: Option<String>, req: Request, next: Next) -> Response {
    let Some(token) = token else {
        return next.run(req).await;
    };
    let ok = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|got| got == token);
    if ok {
        next.run(req).await
    } else {
        ApiError::new(StatusCode::UNAUTHORIZED, "missing or invalid bearer token").into_response()
    }
}

/// Submit a job. Large-input jobs (face-swap) are rejected with 409 when a
/// worker is busy rather than queued.
#[utoipa::path(
    post, path = "/jobs",
    request_body = JobSpec,
    responses(
        (status = 202, description = "Accepted", body = CreateResponse),
        (status = 400, description = "Invalid request"),
        (status = 409, description = "Busy (large-input job, worker not idle)"),
    )
)]
async fn create_job(
    State(state): State<AppState>,
    Json(spec): Json<JobSpec>,
) -> Result<(StatusCode, Json<CreateResponse>), ApiError> {
    if spec.is_large_input() && state.store.is_busy() {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "worker busy; large-input jobs are not queued, retry when idle",
        ));
    }
    let (handle, position) = state
        .store
        .submit(|id| spec_into_request(spec, id, &state.config, &state.uploads))
        .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, e))?;
    Ok((
        StatusCode::ACCEPTED,
        Json(CreateResponse {
            id: handle.id.clone(),
            queue_position: position,
        }),
    ))
}

/// Stream a raw video (mp4) body into the upload store, returning an id a job
/// spec then references via `input_video_upload`. The body is the raw bytes (no
/// base64, no JSON) so a large clip does not inflate or hit the JSON size cap;
/// the store holds it RAM-first or seals it to an encrypted spill. Uploads are
/// reaped after `upload_ttl_secs` if never consumed by a job.
#[utoipa::path(
    post, path = "/uploads",
    request_body(content = Vec<u8>, description = "Raw mp4 bytes", content_type = "application/octet-stream"),
    responses(
        (status = 200, description = "Stored", body = UploadResponse),
        (status = 400, description = "Empty body"),
        (status = 413, description = "Body exceeds max_upload_bytes"),
    )
)]
async fn upload_video(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Json<UploadResponse>, ApiError> {
    if body.is_empty() {
        return Err(ApiError::new(StatusCode::BAD_REQUEST, "empty upload body"));
    }
    let (id, size) = state
        .uploads
        .put(body.to_vec())
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(UploadResponse { id, size }))
}

/// Job status snapshot (polling fallback for the SSE stream).
#[utoipa::path(
    get, path = "/jobs/{id}",
    params(("id" = String, Path, description = "Job id")),
    responses((status = 200, body = JobStatus), (status = 404, description = "Unknown job"))
)]
async fn get_status(
    State(state): State<AppState>,
    AxPath(id): AxPath<String>,
) -> Result<Json<JobStatus>, ApiError> {
    let job = state.store.get(&id).ok_or_else(ApiError::not_found)?;
    Ok(Json(job.status(state.store.position(&id))))
}

/// Live progress as Server-Sent Events. Honors `Last-Event-ID` to replay missed
/// events after a reconnect. The stream ends on a terminal event.
#[utoipa::path(
    get, path = "/jobs/{id}/events",
    params(("id" = String, Path, description = "Job id")),
    responses((status = 200, description = "SSE stream"), (status = 404, description = "Unknown job"))
)]
async fn events(
    State(state): State<AppState>,
    AxPath(id): AxPath<String>,
    headers: HeaderMap,
) -> Result<Sse<impl futures_core::Stream<Item = Result<Event, std::convert::Infallible>>>, ApiError>
{
    let job = state.store.get(&id).ok_or_else(ApiError::not_found)?;
    let after = headers
        .get("last-event-id")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    let (replay, mut rx) = job.subscribe(after);

    let stream = async_stream::stream! {
        for se in replay {
            let terminal = se.event.is_terminal();
            yield Ok(to_event(&se));
            if terminal {
                return;
            }
        }
        loop {
            match rx.recv().await {
                Ok(se) => {
                    let terminal = se.event.is_terminal();
                    yield Ok(to_event(&se));
                    if terminal {
                        break;
                    }
                }
                // Lagged: skip; the client can refetch status. Closed: done.
                Err(RecvError::Lagged(_)) => continue,
                Err(RecvError::Closed) => break,
            }
        }
    };
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

fn to_event(se: &SeqEvent) -> Event {
    let data = serde_json::to_string(&se.event).unwrap_or_else(|_| "{}".into());
    Event::default()
        .id(se.seq.to_string())
        .event(se.event.kind())
        .data(data)
}

/// Return the finished artifact bytes, then DELETE them. The artifact lives
/// exactly until the client fetches it once (the browser holds the only lasting
/// copy, in memory). When the job carried a public key the bytes are the
/// encrypted blob (see [`crate::crypto`]); the client decrypts.
#[utoipa::path(
    get, path = "/jobs/{id}/result",
    params(("id" = String, Path, description = "Job id")),
    responses(
        (status = 200, description = "Artifact bytes (deleted after this fetch)"),
        (status = 404, description = "Unknown job or already fetched"),
        (status = 409, description = "Job not finished"),
    )
)]
async fn get_result(
    State(state): State<AppState>,
    AxPath(id): AxPath<String>,
) -> Result<Response, ApiError> {
    let job = state.store.get(&id).ok_or_else(ApiError::not_found)?;
    if job.state_kind() != JobStateKind::Done {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "job has not finished successfully",
        ));
    }
    let path = job.output_path.clone();
    let bytes = tokio::fs::read(&path).await.map_err(|e| {
        ApiError::new(
            StatusCode::NOT_FOUND,
            format!("artifact unavailable (already fetched?): {e}"),
        )
    })?;
    // Delete-on-fetch: drop the whole job dir so nothing lingers on disk.
    if let Some(dir) = path.parent() {
        let _ = tokio::fs::remove_dir_all(dir).await;
    }
    // Encrypted results are opaque bytes; the client knows the real media type.
    let content_type = if job.public_key.is_some() {
        "application/octet-stream"
    } else {
        match path.extension().and_then(|e| e.to_str()) {
            Some("mp4") => "video/mp4",
            Some("png") => "image/png",
            _ => "application/octet-stream",
        }
    };
    Ok(([(header::CONTENT_TYPE, content_type)], bytes).into_response())
}

/// Request cancellation. A queued job is dropped; a running job is flagged but
/// (v1) runs to completion -- mid-generate interruption is not yet wired.
#[utoipa::path(
    post, path = "/jobs/{id}/cancel",
    params(("id" = String, Path, description = "Job id")),
    responses((status = 200, body = JobStatus), (status = 404, description = "Unknown job"))
)]
async fn cancel(
    State(state): State<AppState>,
    AxPath(id): AxPath<String>,
) -> Result<Json<JobStatus>, ApiError> {
    let job = state.store.get(&id).ok_or_else(ApiError::not_found)?;
    job.request_cancel();
    Ok(Json(job.status(state.store.position(&id))))
}

async fn openapi() -> Json<utoipa::openapi::OpenApi> {
    Json(openapi_doc())
}

/// The OpenAPI 3.1 document for the job API.
#[derive(utoipa::OpenApi)]
#[openapi(
    info(
        title = "thinfer",
        description = "Async job API for image/video/face-swap generation."
    ),
    paths(
        create_job,
        upload_video,
        get_status,
        events,
        get_result,
        cancel,
        crate::vault::list_adapters,
        crate::vault::add_adapter,
        crate::vault::remove_adapter,
    ),
    components(schemas(
        JobSpec,
        thinfer_app::wire::ImageSpec,
        thinfer_app::wire::VideoSpec,
        thinfer_app::wire::FaceSwapSpec,
        thinfer_app::wire::LoraSpec,
        CreateResponse,
        UploadResponse,
        JobStatus,
        JobStateKind,
        thinfer_app::wire::ProgressStage,
        thinfer_app::wire::JobResult,
        ImageModelId,
        VideoModelId,
        SwapModel,
        VaeChoice,
        crate::vault::AddAdapterRequest,
        crate::vault::ListAdaptersRequest,
        crate::vault::RemoveAdapterRequest,
        crate::vault::AdaptersResponse,
        thinfer_app::vault::VaultEntryInfo,
    ))
)]
struct ApiDoc;

pub fn openapi_doc() -> utoipa::openapi::OpenApi {
    <ApiDoc as utoipa::OpenApi>::openapi()
}

/// A JSON error response.
pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    pub(crate) fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
    fn not_found() -> Self {
        Self::new(StatusCode::NOT_FOUND, "unknown job")
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({ "error": self.message })),
        )
            .into_response()
    }
}
