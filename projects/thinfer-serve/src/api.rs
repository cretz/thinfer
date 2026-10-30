//! HTTP surface: the axum handlers. Jobs arrive SEALED under a per-job key the
//! server does not have (`thinfer_app::jobkey`); the key is delivered only when
//! a worker is ready (`POST /jobs/{id}/key`), and results are served as
//! ciphertext to the client that proves ownership of that key. The wire types
//! live in `thinfer_app::wire` so the `RemoteExecutor` client shares them.

use std::sync::Arc;

use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path as AxPath, Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use base64::Engine;
use thinfer_app::jobkey::JobKey;
use thinfer_app::wire::{
    CreateResponse, JobKeyDelivery, JobStateKind, JobStatus, SealedJob, UploadResponse,
};
use tokio::sync::broadcast::error::RecvError;

use crate::config::ServeConfig;
use crate::job::{JobStore, SeqEvent};
use crate::uploads::UploadStore;

/// Header carrying the result-fetch proof: base64 SHA-256 of the job key.
const KEY_PROOF_HEADER: &str = "x-job-key-sha256";

/// Never let a browser or proxy cache job data.
const NO_STORE: (header::HeaderName, &str) = (header::CACHE_CONTROL, "no-store");

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
        .route("/jobs/{id}/key", post(deliver_key))
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
        .is_some_and(|got| constant_time_eq(got.as_bytes(), token.as_bytes()));
    if ok {
        next.run(req).await
    } else {
        ApiError::new(StatusCode::UNAUTHORIZED, "missing or invalid bearer token").into_response()
    }
}

/// Constant-time byte compare (length is not secret).
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Submit a sealed job. The server cannot read it; it queues the ciphertext.
/// Large-input jobs (face-swap) are rejected with 409 when a worker is busy
/// rather than queued.
#[utoipa::path(
    post, path = "/jobs",
    request_body = SealedJob,
    responses(
        (status = 202, description = "Accepted", body = CreateResponse),
        (status = 400, description = "Invalid request"),
        (status = 409, description = "Busy (large-input job, worker not idle)"),
    )
)]
async fn create_job(
    State(state): State<AppState>,
    Json(job): Json<SealedJob>,
) -> Result<(StatusCode, Json<CreateResponse>), ApiError> {
    if job.large_input && state.store.is_busy() {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "worker busy; large-input jobs are not queued, retry when idle",
        ));
    }
    let sealed = base64::engine::general_purpose::STANDARD
        .decode(job.sealed.as_bytes())
        .map_err(|e| {
            ApiError::new(
                StatusCode::BAD_REQUEST,
                format!("sealed is not base64: {e}"),
            )
        })?;
    let (handle, position) = state.store.submit(&state.config.artifact_dir, sealed);
    Ok((
        StatusCode::ACCEPTED,
        Json(CreateResponse {
            id: handle.id.clone(),
            queue_position: position,
        }),
    ))
}

/// Deliver the job key, only while a worker is waiting for it (the
/// `keyRequired` event). The job decrypts and runs right after.
#[utoipa::path(
    post, path = "/jobs/{id}/key",
    params(("id" = String, Path, description = "Job id")),
    request_body = JobKeyDelivery,
    responses(
        (status = 204, description = "Key accepted"),
        (status = 404, description = "Unknown job"),
        (status = 409, description = "Job is not waiting for its key"),
    )
)]
async fn deliver_key(
    State(state): State<AppState>,
    AxPath(id): AxPath<String>,
    Json(body): Json<JobKeyDelivery>,
) -> Result<StatusCode, ApiError> {
    let job = state.store.get(&id).ok_or_else(ApiError::not_found)?;
    let key = JobKey::from_b64(&body.key).map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, e))?;
    job.deliver_key(key)
        .map_err(|e| ApiError::new(StatusCode::CONFLICT, e))?;
    Ok(StatusCode::NO_CONTENT)
}

/// Stream a client-sealed video (`[nonce][AES-GCM ciphertext]` under the job
/// key) into the upload store, returning an id the sealed spec references via
/// `input_video_upload`. Stored as ciphertext; the server has no key for it
/// until the job runs. Reaped after `upload_ttl_secs` if never consumed.
#[utoipa::path(
    post, path = "/uploads",
    request_body(content = Vec<u8>, description = "Sealed mp4 bytes", content_type = "application/octet-stream"),
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
        .put(&body)
        .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, e))?;
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

/// Return the finished artifact (sealed under the job key), then forget the job
/// and DELETE its dir. Requires the `x-job-key-sha256` proof so only the key
/// holder can fetch (and thereby delete) it.
#[utoipa::path(
    get, path = "/jobs/{id}/result",
    params(("id" = String, Path, description = "Job id")),
    responses(
        (status = 200, description = "Artifact bytes (deleted after this fetch)"),
        (status = 403, description = "Missing or wrong key proof"),
        (status = 404, description = "Unknown job or already fetched"),
        (status = 409, description = "Job not finished"),
    )
)]
async fn get_result(
    State(state): State<AppState>,
    AxPath(id): AxPath<String>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let job = state.store.get(&id).ok_or_else(ApiError::not_found)?;
    let proof = headers
        .get(KEY_PROOF_HEADER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| base64::engine::general_purpose::STANDARD.decode(v).ok())
        .unwrap_or_default();
    if !job.owns(&proof) {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "missing or wrong job key proof",
        ));
    }
    if job.state_kind() != JobStateKind::Done {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "job has not finished successfully",
        ));
    }
    let path = job.dir.join(crate::worker::RESULT_FILE);
    let bytes = tokio::fs::read(&path).await.map_err(|e| {
        ApiError::new(
            StatusCode::NOT_FOUND,
            format!("artifact unavailable (already fetched?): {e}"),
        )
    })?;
    // Delete-on-fetch: forget the job and drop its dir so nothing lingers.
    state.store.remove(&id);
    Ok((
        [(header::CONTENT_TYPE, "application/octet-stream"), NO_STORE],
        bytes,
    )
        .into_response())
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
        deliver_key,
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
        SealedJob,
        JobKeyDelivery,
        thinfer_app::wire::JobSpec,
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
        thinfer_app::model::ImageModelId,
        thinfer_app::model::VideoModelId,
        thinfer_app::model::SwapModel,
        thinfer_app::model::VaeChoice,
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
