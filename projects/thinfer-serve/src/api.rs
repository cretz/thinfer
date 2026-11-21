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
    Capabilities, CreateResponse, CreateSession, JobKeyDelivery, JobList, JobPatch, JobStateKind,
    JobStatus, SealedJob, SessionInfo, SessionPatch, UploadResponse,
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
    /// Opt-in text sessions (sealed state trees, RAM only).
    pub sessions: Arc<crate::sessions::SessionStore>,
}

/// `GET /capabilities` body: static server/GPU capabilities the UI adapts to.
#[utoipa::path(
    get, path = "/capabilities",
    responses((status = 200, body = Capabilities))
)]
async fn capabilities(State(state): State<AppState>) -> Json<Capabilities> {
    Json(Capabilities {
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
        .route("/jobs", post(create_job).get(list_jobs))
        .route("/jobs/{id}/key", post(deliver_key))
        .route(
            "/jobs/{id}",
            get(get_status).patch(patch_job).delete(delete_job),
        )
        .route("/jobs/{id}/events", get(events))
        .route("/jobs/{id}/result", get(get_result))
        .route("/jobs/{id}/cancel", post(cancel))
        .route("/sessions", post(create_session))
        .route(
            "/sessions/import",
            post(import_session).layer(DefaultBodyLimit::max(max_upload)),
        )
        .route(
            "/sessions/{id}",
            get(get_session).patch(patch_session).delete(delete_session),
        )
        .route(
            "/sessions/{id}/nodes/{node}",
            axum::routing::delete(delete_session_node),
        )
        .route("/sessions/{id}/export", get(export_session))
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
/// The uploads it names are held for it (exempt from the upload TTL) while it
/// waits.
#[utoipa::path(
    post, path = "/jobs",
    request_body = SealedJob,
    responses(
        (status = 202, description = "Accepted", body = CreateResponse),
        (status = 400, description = "Invalid request or unknown upload"),
    )
)]
async fn create_job(
    State(state): State<AppState>,
    Json(job): Json<SealedJob>,
) -> Result<(StatusCode, Json<CreateResponse>), ApiError> {
    let sealed = base64::engine::general_purpose::STANDARD
        .decode(job.sealed.as_bytes())
        .map_err(|e| {
            ApiError::new(
                StatusCode::BAD_REQUEST,
                format!("sealed is not base64: {e}"),
            )
        })?;
    let owner = match &job.key_proof {
        Some(p) => Some(decode_proof(p).ok_or_else(|| {
            ApiError::new(
                StatusCode::BAD_REQUEST,
                "keyProof must be base64 of a 32-byte SHA-256",
            )
        })?),
        None => None,
    };
    let handle = state
        .store
        .create(&state.config.artifact_dir, sealed, job.priority, owner);
    state
        .uploads
        .claim(&job.uploads, &handle)
        .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, e))?;
    let position = state.store.enqueue(&handle);
    Ok((
        StatusCode::ACCEPTED,
        Json(CreateResponse {
            id: handle.id.clone(),
            queue_position: position,
        }),
    ))
}

/// A base64 32-byte key proof.
fn decode_proof(b64: &str) -> Option<[u8; 32]> {
    base64::engine::general_purpose::STANDARD
        .decode(b64)
        .ok()?
        .try_into()
        .ok()
}

/// The `x-job-key-sha256` header's bytes (empty when absent or malformed).
fn header_proof(headers: &HeaderMap) -> Vec<u8> {
    headers
        .get(KEY_PROOF_HEADER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| base64::engine::general_purpose::STANDARD.decode(v).ok())
        .unwrap_or_default()
}

/// The job, if the caller may manage it (the declared key proof, when the job
/// has one, in `x-job-key-sha256`).
fn managed_job(
    state: &AppState,
    id: &str,
    headers: &HeaderMap,
) -> Result<Arc<crate::job::JobHandle>, ApiError> {
    let job = state.store.get(id).ok_or_else(ApiError::not_found)?;
    if !job.manageable_by(&header_proof(headers)) {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "missing or wrong job key proof",
        ));
    }
    Ok(job)
}

/// Every job the server holds: pending ones in run order, then running and
/// finished ones newest first. Structural fields only (specs are sealed).
#[utoipa::path(
    get, path = "/jobs",
    responses((status = 200, body = JobList))
)]
async fn list_jobs(State(state): State<AppState>) -> Json<JobList> {
    Json(JobList {
        jobs: state.store.list(),
    })
}

/// Change a queued job's priority.
#[utoipa::path(
    patch, path = "/jobs/{id}",
    params(("id" = String, Path, description = "Job id")),
    request_body = JobPatch,
    responses(
        (status = 200, body = JobStatus),
        (status = 403, description = "Missing or wrong key proof"),
        (status = 404, description = "Unknown job"),
        (status = 409, description = "Job is no longer queued"),
    )
)]
async fn patch_job(
    State(state): State<AppState>,
    AxPath(id): AxPath<String>,
    headers: HeaderMap,
    Json(body): Json<JobPatch>,
) -> Result<Json<JobStatus>, ApiError> {
    let job = managed_job(&state, &id, &headers)?;
    if !state.store.set_priority(&id, body.priority) {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "job is no longer queued",
        ));
    }
    Ok(Json(job.status(state.store.position(&id))))
}

/// Remove a job: a queued one is cancelled and forgotten, a running one is
/// cancelled, a finished one is forgotten with its dir (an unfetched result
/// included).
#[utoipa::path(
    delete, path = "/jobs/{id}",
    params(("id" = String, Path, description = "Job id")),
    responses(
        (status = 204, description = "Removed (or cancelling, if running)"),
        (status = 403, description = "Missing or wrong key proof"),
        (status = 404, description = "Unknown job"),
    )
)]
async fn delete_job(
    State(state): State<AppState>,
    AxPath(id): AxPath<String>,
    headers: HeaderMap,
) -> Result<StatusCode, ApiError> {
    managed_job(&state, &id, &headers)?;
    state.store.delete(&id);
    Ok(StatusCode::NO_CONTENT)
}

/// Header proving a session's ownership: base64 SHA-256 of the session key.
const SESSION_PROOF_HEADER: &str = "x-session-key-sha256";

fn session_proof(headers: &HeaderMap) -> Vec<u8> {
    headers
        .get(SESSION_PROOF_HEADER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| base64::engine::general_purpose::STANDARD.decode(v).ok())
        .unwrap_or_default()
}

impl From<crate::sessions::SessionError> for ApiError {
    fn from(e: crate::sessions::SessionError) -> Self {
        use crate::sessions::SessionError as E;
        let status = match &e {
            E::NotFound(_) => StatusCode::NOT_FOUND,
            E::Forbidden => StatusCode::FORBIDDEN,
            E::Conflict(_) => StatusCode::CONFLICT,
            E::Invalid(_) => StatusCode::BAD_REQUEST,
        };
        ApiError::new(status, e.to_string())
    }
}

/// Open an opt-in text session: a tree of sealed model states that text
/// jobs continue from (`TextSpec.session`) instead of re-sending the whole
/// conversation. RAM only; forgotten after its TTL.
#[utoipa::path(
    post, path = "/sessions",
    request_body = CreateSession,
    responses(
        (status = 201, body = SessionInfo),
        (status = 400, description = "Bad key proof"),
    )
)]
async fn create_session(
    State(state): State<AppState>,
    Json(body): Json<CreateSession>,
) -> Result<(StatusCode, Json<SessionInfo>), ApiError> {
    let owner = decode_proof(&body.key_proof).ok_or_else(|| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "keyProof must be base64 of a 32-byte SHA-256",
        )
    })?;
    let model = body
        .model
        .unwrap_or(thinfer_app::model::TextModelId::DEFAULT);
    Ok((
        StatusCode::CREATED,
        Json(state.sessions.create(model, body.ttl_secs, owner)),
    ))
}

/// A session's structure (node tree, sizes, expiry). No content.
#[utoipa::path(
    get, path = "/sessions/{id}",
    params(("id" = String, Path, description = "Session id")),
    responses(
        (status = 200, body = SessionInfo),
        (status = 403, description = "Missing or wrong session key proof"),
        (status = 404, description = "Unknown session"),
    )
)]
async fn get_session(
    State(state): State<AppState>,
    AxPath(id): AxPath<String>,
    headers: HeaderMap,
) -> Result<Json<SessionInfo>, ApiError> {
    Ok(Json(state.sessions.info(&id, &session_proof(&headers))?))
}

/// Change a session's idle TTL (also refreshes it).
#[utoipa::path(
    patch, path = "/sessions/{id}",
    params(("id" = String, Path, description = "Session id")),
    request_body = SessionPatch,
    responses(
        (status = 200, body = SessionInfo),
        (status = 403, description = "Missing or wrong session key proof"),
        (status = 404, description = "Unknown session"),
    )
)]
async fn patch_session(
    State(state): State<AppState>,
    AxPath(id): AxPath<String>,
    headers: HeaderMap,
    Json(body): Json<SessionPatch>,
) -> Result<Json<SessionInfo>, ApiError> {
    Ok(Json(state.sessions.set_ttl(
        &id,
        &session_proof(&headers),
        body.ttl_secs,
    )?))
}

/// Forget a session and all its state now.
#[utoipa::path(
    delete, path = "/sessions/{id}",
    params(("id" = String, Path, description = "Session id")),
    responses(
        (status = 204, description = "Deleted"),
        (status = 403, description = "Missing or wrong session key proof"),
        (status = 404, description = "Unknown session"),
    )
)]
async fn delete_session(
    State(state): State<AppState>,
    AxPath(id): AxPath<String>,
    headers: HeaderMap,
) -> Result<StatusCode, ApiError> {
    state.sessions.delete(&id, &session_proof(&headers))?;
    Ok(StatusCode::NO_CONTENT)
}

/// Prune a node and its whole subtree (e.g. an abandoned branch).
#[utoipa::path(
    delete, path = "/sessions/{id}/nodes/{node}",
    params(
        ("id" = String, Path, description = "Session id"),
        ("node" = String, Path, description = "Node id"),
    ),
    responses(
        (status = 200, body = SessionInfo),
        (status = 403, description = "Missing or wrong session key proof"),
        (status = 404, description = "Unknown session or node"),
    )
)]
async fn delete_session_node(
    State(state): State<AppState>,
    AxPath((id, node)): AxPath<(String, String)>,
    headers: HeaderMap,
) -> Result<Json<SessionInfo>, ApiError> {
    Ok(Json(state.sessions.delete_node(
        &id,
        &session_proof(&headers),
        &node,
    )?))
}

/// The whole session as one blob: its structure plus every node's state,
/// still sealed under the session key. `POST /sessions/import` restores it.
#[utoipa::path(
    get, path = "/sessions/{id}/export",
    params(("id" = String, Path, description = "Session id")),
    responses(
        (status = 200, description = "Session export bytes"),
        (status = 403, description = "Missing or wrong session key proof"),
        (status = 404, description = "Unknown session"),
    )
)]
async fn export_session(
    State(state): State<AppState>,
    AxPath(id): AxPath<String>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let blob = state.sessions.export(&id, &session_proof(&headers))?;
    Ok((
        [(header::CONTENT_TYPE, "application/octet-stream"), NO_STORE],
        blob,
    )
        .into_response())
}

/// Restore an exported session (under a new id; same key, nodes and TTL).
#[utoipa::path(
    post, path = "/sessions/import",
    request_body(content = Vec<u8>, description = "A session export", content_type = "application/octet-stream"),
    responses(
        (status = 201, body = SessionInfo),
        (status = 400, description = "Not a valid export"),
        (status = 409, description = "Exceeds session_ram_budget"),
    )
)]
async fn import_session(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<(StatusCode, Json<SessionInfo>), ApiError> {
    Ok((StatusCode::CREATED, Json(state.sessions.import(&body)?)))
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
        let mut last = after;
        let mut pending = replay;
        loop {
            for se in pending.drain(..) {
                if se.seq <= last {
                    continue;
                }
                last = se.seq;
                let terminal = se.event.is_terminal();
                yield Ok(to_event(&se));
                if terminal {
                    return;
                }
            }
            match rx.recv().await {
                Ok(se) => pending.push(se),
                // Lagged: catch up from the job log (streamed text must not
                // drop pieces). Closed: done.
                Err(RecvError::Lagged(_)) => {
                    let (replay, fresh) = job.subscribe(last);
                    pending = replay;
                    rx = fresh;
                }
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
    if !job.owns(&header_proof(&headers)) {
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
    responses(
        (status = 200, body = JobStatus),
        (status = 403, description = "Missing or wrong key proof"),
        (status = 404, description = "Unknown job"),
    )
)]
async fn cancel(
    State(state): State<AppState>,
    AxPath(id): AxPath<String>,
    headers: HeaderMap,
) -> Result<Json<JobStatus>, ApiError> {
    let job = managed_job(&state, &id, &headers)?;
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
        description = "Async job API for image, video, face-swap and text (LLM) generation.

Every job is end-to-end sealed. The client generates a 32-byte AES-256-GCM key per job and submits `SealedJob.sealed` = base64(`[12-byte nonce][ciphertext + 16-byte tag]`) of the `JobSpec` JSON (tagged by `kind`). It then follows `GET /jobs/{id}/events` (SSE; events are `JobEvent` JSON tagged by `type`: queued, keyRequired, started, progress, log, text, done, error, cancelled). On `keyRequired` it posts the key to `/jobs/{id}/key`. `text` events carry reply pieces sealed under the same key. After `done`, `GET /jobs/{id}/result` with header `x-job-key-sha256: base64(SHA-256(key))` returns the artifact sealed under the key, then deletes it. Declaring `keyProof` at submit makes cancel / PATCH / DELETE require that header.

Tool calling: a text job with `tools` may end with `TextResult.finishReason` = `toolCalls` and `TextResult.toolCalls` = a JSON array of `ToolCall` sealed under the job key. Every call names a defined tool and its arguments match that tool's schema. Run the calls, then submit the next text job with the assistant turn (`toolCalls` as returned) and one `tool` turn per call (`toolCallId`, `name`, `content`), the tools sent again; with a session, continue from the reply's `sessionNode` with just the `tool` turns."
    ),
    paths(
        create_job,
        list_jobs,
        patch_job,
        delete_job,
        capabilities,
        create_session,
        get_session,
        patch_session,
        delete_session,
        delete_session_node,
        export_session,
        import_session,
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
        JobPatch,
        JobList,
        thinfer_app::wire::JobListEntry,
        thinfer_app::wire::JobEvent,
        Capabilities,
        thinfer_app::wire::JobSpec,
        thinfer_app::wire::TextSpec,
        thinfer_app::wire::TextMessage,
        thinfer_app::wire::TextResult,
        thinfer_app::wire::TextSessionRef,
        CreateSession,
        SessionPatch,
        SessionInfo,
        thinfer_app::wire::SessionNodeInfo,
        thinfer_app::text_format::TextFormat,
        thinfer_app::tools::ToolDef,
        thinfer_app::tools::ToolChoice,
        thinfer_app::tools::ToolCall,
        thinfer_app::tools::FinishReason,
        thinfer_app::model::TextModelId,
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

#[cfg(test)]
mod tests {
    #[test]
    fn openapi_lists_every_route() {
        let doc = serde_json::to_value(super::openapi_doc()).unwrap();
        let paths = doc["paths"].as_object().unwrap();
        for p in [
            "/jobs",
            "/jobs/{id}",
            "/jobs/{id}/key",
            "/jobs/{id}/events",
            "/jobs/{id}/result",
            "/jobs/{id}/cancel",
            "/uploads",
            "/capabilities",
            "/sessions",
            "/sessions/{id}",
            "/sessions/{id}/nodes/{node}",
            "/sessions/{id}/export",
            "/sessions/import",
        ] {
            assert!(paths.contains_key(p), "{p} missing from the OpenAPI doc");
        }
        let jobs = &paths["/jobs"];
        assert!(jobs.get("get").is_some() && jobs.get("post").is_some());
        let job = &paths["/jobs/{id}"];
        assert!(job.get("patch").is_some() && job.get("delete").is_some());
        let schemas = doc["components"]["schemas"].as_object().unwrap();
        for s in [
            "TextSpec",
            "TextFormat",
            "TextSessionRef",
            "ToolDef",
            "ToolCall",
            "ToolChoice",
            "FinishReason",
            "JobEvent",
            "SessionInfo",
            "JobList",
        ] {
            assert!(schemas.contains_key(s), "{s} schema missing");
        }
    }
}
