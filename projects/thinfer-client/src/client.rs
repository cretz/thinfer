//! The async HTTP client.

use std::time::Duration;

use futures_util::StreamExt;
use zeroize::Zeroizing;

use crate::jobkey::{self, JobKey};
use crate::model::TextModelId;
use crate::progress::Stage;
use crate::tools::{FinishReason, ToolCall};
use crate::wire::{
    Capabilities, CreateResponse, CreateSession, JobEvent, JobKeyDelivery, JobList, JobPatch,
    JobResult, JobSpec, JobStatus, SealedJob, SessionInfo, SessionPatch, TextSessionRef,
    UploadResponse,
};
use crate::{Error, sse};

/// Header carrying a job's key proof.
const JOB_PROOF: &str = "x-job-key-sha256";
/// Header carrying a session's key proof.
const SESSION_PROOF: &str = "x-session-key-sha256";
/// Event-stream reconnects before [`Client::follow`] gives up.
const RECONNECTS: u32 = 5;

/// A transport error with its cause chain (reqwest's own message omits the
/// cause, e.g. an untrusted certificate).
fn transport(e: reqwest::Error) -> Error {
    let mut msg = e.to_string();
    let mut src = std::error::Error::source(&e);
    while let Some(cause) = src {
        msg.push_str(": ");
        msg.push_str(&cause.to_string());
        src = cause.source();
    }
    Error::Transport(msg)
}

/// A connection to one `thinfer-serve`. Cheap to clone; reuse it across jobs.
#[derive(Clone)]
pub struct Client {
    /// Base URL, no trailing slash.
    base: String,
    http: reqwest::Client,
    token: Option<Zeroizing<String>>,
}

/// Configures a [`Client`].
pub struct ClientBuilder {
    base: String,
    token: Option<Zeroizing<String>>,
    roots: Vec<Vec<u8>>,
    accept_invalid_certs: bool,
    connect_timeout: Duration,
}

impl ClientBuilder {
    /// The server's bearer token (its `auth_token`).
    pub fn token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(Zeroizing::new(token.into()));
        self
    }

    /// Trust this PEM certificate (e.g. the server's self-signed one) in
    /// addition to the system roots.
    pub fn root_certificate_pem(mut self, pem: &[u8]) -> Self {
        self.roots.push(pem.to_vec());
        self
    }

    /// Skip TLS certificate checks. Anyone on the path can then read and
    /// change the traffic outside the sealed parts (and the bearer token):
    /// prefer [`Self::root_certificate_pem`].
    pub fn danger_accept_invalid_certs(mut self, accept: bool) -> Self {
        self.accept_invalid_certs = accept;
        self
    }

    /// Connection timeout (default 30 s). Requests themselves have none: an
    /// event stream lasts as long as its job.
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    pub fn build(self) -> Result<Client, Error> {
        let base = self.base.trim_end_matches('/').to_string();
        if !(base.starts_with("http://") || base.starts_with("https://")) {
            return Err(Error::InvalidUrl(self.base));
        }
        let mut http = reqwest::Client::builder()
            .connect_timeout(self.connect_timeout)
            .danger_accept_invalid_certs(self.accept_invalid_certs);
        for pem in &self.roots {
            let cert = reqwest::Certificate::from_pem(pem)
                .map_err(|e| Error::Transport(format!("root certificate: {e}")))?;
            http = http.add_root_certificate(cert);
        }
        let http = http.build().map_err(transport)?;
        Ok(Client {
            base,
            http,
            token: self.token,
        })
    }
}

/// A submitted job: its id and key. Manage it with [`Client::follow`],
/// [`Client::fetch_result`], [`Client::cancel`] and friends.
pub struct Job {
    pub id: String,
    /// Queue position at submit.
    pub queue_position: usize,
    key: JobKey,
}

impl Job {
    /// The job's key (it opens the result and the streamed text).
    pub fn key(&self) -> &JobKey {
        &self.key
    }
}

/// Queue options for [`Client::submit`].
#[derive(Clone, Debug, Default)]
pub struct SubmitOptions {
    /// Higher runs first (default 0).
    pub priority: i32,
}

/// A live update while a job runs. Text is decrypted; it borrows a buffer
/// that is zeroed right after the callback returns.
#[derive(Debug)]
pub enum Update<'a> {
    Queued {
        position: usize,
    },
    /// A worker took the job (its key was just delivered).
    Started,
    Progress(Stage),
    /// Server narration (no job data).
    Log(&'a str),
    /// A piece of a text job's reply.
    Text(&'a str),
}

/// How a job ended: the server's result, plus a text reply's tool calls
/// (opened with the job key).
pub struct Finished {
    pub result: JobResult,
    pub tool_calls: Vec<ToolCall>,
}

/// A finished job's output, decrypted.
pub struct Output {
    pub result: JobResult,
    /// The artifact (PNG, MP4; a text job's reply as UTF-8). Zeroed on drop.
    pub artifact: Zeroizing<Vec<u8>>,
    /// A text reply's tool calls (empty unless `finish_reason` is
    /// `ToolCalls`).
    pub tool_calls: Vec<ToolCall>,
}

impl Output {
    /// A text job's reply (`None` for other jobs, or non-UTF-8 output).
    pub fn text(&self) -> Option<&str> {
        self.result.text.as_ref()?;
        std::str::from_utf8(&self.artifact).ok()
    }

    /// How a text reply ended (`Stop` for other jobs).
    pub fn finish_reason(&self) -> FinishReason {
        self.result
            .text
            .as_ref()
            .map_or(FinishReason::Stop, |t| t.finish_reason)
    }

    /// The session node a session reply created: continue from it.
    pub fn session_node(&self) -> Option<&str> {
        self.result.text.as_ref()?.session_node.as_deref()
    }
}

/// A server-held text session: its id and key. The server's copy of every
/// node is sealed under the key; keep it (and [`Client::export_session`]'s
/// blob) to resume later, even on another server.
pub struct Session {
    pub id: String,
    key: JobKey,
}

impl Session {
    /// A session you already hold (e.g. after a restart).
    pub fn from_parts(id: impl Into<String>, key: JobKey) -> Self {
        Self { id: id.into(), key }
    }

    pub fn key(&self) -> &JobKey {
        &self.key
    }

    /// The reference a [`crate::wire::TextSpec`] continues this session with:
    /// from `node` (a reply's [`Output::session_node`]), or `None` for a new
    /// root (send the whole conversation then).
    pub fn at(&self, node: Option<&str>) -> TextSessionRef {
        TextSessionRef {
            id: self.id.clone(),
            from: node.map(str::to_string),
            key: self.key.to_b64().to_string(),
        }
    }
}

impl Client {
    /// A client for `base_url` (e.g. `https://gpu-box:8080`) with defaults.
    pub fn new(base_url: &str) -> Result<Self, Error> {
        Self::builder(base_url).build()
    }

    pub fn builder(base_url: &str) -> ClientBuilder {
        ClientBuilder {
            base: base_url.to_string(),
            token: None,
            roots: Vec::new(),
            accept_invalid_certs: false,
            connect_timeout: Duration::from_secs(30),
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    fn req(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let rb = self.http.request(method, self.url(path));
        match &self.token {
            Some(t) => rb.bearer_auth(t.as_str()),
            None => rb,
        }
    }

    async fn send(rb: reqwest::RequestBuilder) -> Result<reqwest::Response, Error> {
        let resp = rb.send().await.map_err(transport)?;
        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }
        let body = resp.text().await.unwrap_or_default();
        let message = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v.get("error")?.as_str().map(str::to_string))
            .unwrap_or(if body.is_empty() {
                status.canonical_reason().unwrap_or("").to_string()
            } else {
                body
            });
        Err(Error::Server {
            status: status.as_u16(),
            message,
        })
    }

    async fn json<R: serde::de::DeserializeOwned>(rb: reqwest::RequestBuilder) -> Result<R, Error> {
        Self::send(rb)
            .await?
            .json()
            .await
            .map_err(|e| Error::Protocol(format!("response: {e}")))
    }

    async fn bytes(rb: reqwest::RequestBuilder) -> Result<Vec<u8>, Error> {
        Ok(Self::send(rb)
            .await?
            .bytes()
            .await
            .map_err(transport)?
            .to_vec())
    }

    /// What the server's GPU supports.
    pub async fn capabilities(&self) -> Result<Capabilities, Error> {
        Self::json(self.req(reqwest::Method::GET, "/capabilities")).await
    }

    // ---- jobs ----

    /// Submit `spec`, follow it, and fetch its artifact: the whole job in one
    /// call, under a fresh key. `on` sees every [`Update`].
    pub async fn run(&self, spec: &JobSpec, on: impl FnMut(Update<'_>)) -> Result<Output, Error> {
        self.run_with(spec, JobKey::generate(), &SubmitOptions::default(), on)
            .await
    }

    /// [`Self::run`] under `key` (the key media were [`Self::upload`]ed
    /// under) and `opts`.
    pub async fn run_with(
        &self,
        spec: &JobSpec,
        key: JobKey,
        opts: &SubmitOptions,
        on: impl FnMut(Update<'_>),
    ) -> Result<Output, Error> {
        let job = self.submit(spec, key, opts).await?;
        let finished = self.follow(&job, on).await?;
        let artifact = self.fetch_result(&job).await?;
        Ok(Output {
            result: finished.result,
            artifact,
            tool_calls: finished.tool_calls,
        })
    }

    /// Upload media a spec references by id (`FaceSwapSpec.inputVideoUpload`),
    /// sealed under the job's `key` first: the server stores ciphertext.
    pub async fn upload(&self, key: &JobKey, plaintext: &[u8]) -> Result<UploadResponse, Error> {
        Self::json(
            self.req(reqwest::Method::POST, "/uploads")
                .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
                .body(key.seal(plaintext)),
        )
        .await
    }

    /// Seal `spec` under `key` and queue it. The key stays here until the
    /// worker asks for it ([`Self::follow`] delivers it).
    pub async fn submit(
        &self,
        spec: &JobSpec,
        key: JobKey,
        opts: &SubmitOptions,
    ) -> Result<Job, Error> {
        let json = Zeroizing::new(
            serde_json::to_vec(spec).map_err(|e| Error::Protocol(format!("encode spec: {e}")))?,
        );
        let sealed = SealedJob {
            sealed: jobkey::seal_b64(&key, &json),
            uploads: spec.upload_ids(),
            priority: opts.priority,
            key_proof: Some(key.proof_b64()),
        };
        let created: CreateResponse =
            Self::json(self.req(reqwest::Method::POST, "/jobs").json(&sealed)).await?;
        Ok(Job {
            id: created.id,
            queue_position: created.queue_position,
            key,
        })
    }

    /// Follow `job`'s events to the end: deliver its key when the worker asks,
    /// decrypt streamed text into `on`, reconnect (resuming where it left off)
    /// if the stream drops. Returns the result, or the job's failure.
    pub async fn follow(
        &self,
        job: &Job,
        mut on: impl FnMut(Update<'_>),
    ) -> Result<Finished, Error> {
        let mut last: Option<u64> = None;
        let mut retries = 0;
        loop {
            match self.follow_once(job, &mut last, &mut on).await {
                Ok(Some(result)) => {
                    let tool_calls = match result.text.as_ref().and_then(|t| t.tool_calls.as_ref())
                    {
                        Some(sealed) => {
                            let plain = jobkey::open_b64(&job.key, sealed)?;
                            serde_json::from_slice(&plain)
                                .map_err(|e| Error::Protocol(format!("tool calls: {e}")))?
                        }
                        None => Vec::new(),
                    };
                    return Ok(Finished { result, tool_calls });
                }
                // The stream ended or broke before a terminal event.
                Ok(None) | Err(Error::Transport(_)) if retries < RECONNECTS => {
                    retries += 1;
                    tokio::time::sleep(Duration::from_millis(250 << retries)).await;
                }
                Ok(None) => {
                    return Err(Error::Protocol(
                        "event stream ended before the job finished".into(),
                    ));
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// One pass over the event stream from `last`. `Ok(None)`: it ended
    /// without a terminal event.
    async fn follow_once(
        &self,
        job: &Job,
        last: &mut Option<u64>,
        on: &mut impl FnMut(Update<'_>),
    ) -> Result<Option<JobResult>, Error> {
        let mut rb = self.req(reqwest::Method::GET, &format!("/jobs/{}/events", job.id));
        if let Some(id) = last {
            rb = rb.header("last-event-id", id.to_string());
        }
        let mut stream = Self::send(rb).await?.bytes_stream();
        let mut frames = sse::Decoder::default();
        while let Some(chunk) = stream.next().await {
            frames.push(&chunk.map_err(transport)?);
            while let Some(frame) = frames.next()? {
                if frame.id.is_some() {
                    *last = frame.id;
                }
                match frame.event {
                    JobEvent::Queued { position } => on(Update::Queued { position }),
                    JobEvent::KeyRequired => self.deliver_key(job).await?,
                    JobEvent::Started => on(Update::Started),
                    JobEvent::Progress { stage } => on(Update::Progress(stage.into())),
                    JobEvent::Log { message } => on(Update::Log(&message)),
                    JobEvent::Text { sealed } => {
                        let plain = jobkey::open_b64(&job.key, &sealed)?;
                        let text = String::from_utf8_lossy(&plain);
                        on(Update::Text(&text));
                        if let std::borrow::Cow::Owned(mut s) = text {
                            zeroize::Zeroize::zeroize(&mut s);
                        }
                    }
                    JobEvent::Done { result } => return Ok(Some(result)),
                    JobEvent::Error { message } => return Err(Error::JobFailed(message)),
                    JobEvent::Cancelled => return Err(Error::Cancelled),
                }
            }
        }
        Ok(None)
    }

    async fn deliver_key(&self, job: &Job) -> Result<(), Error> {
        let body = JobKeyDelivery {
            key: job.key.to_b64().to_string(),
        };
        let sent = Self::send(
            self.req(reqwest::Method::POST, &format!("/jobs/{}/key", job.id))
                .json(&body),
        )
        .await;
        let mut key = body.key;
        zeroize::Zeroize::zeroize(&mut key);
        sent.map(drop)
    }

    /// Fetch and open a finished job's artifact. The server deletes the job
    /// once it is fetched.
    pub async fn fetch_result(&self, job: &Job) -> Result<Zeroizing<Vec<u8>>, Error> {
        let sealed = Self::bytes(
            self.req(reqwest::Method::GET, &format!("/jobs/{}/result", job.id))
                .header(JOB_PROOF, job.key.proof_b64()),
        )
        .await?;
        job.key.open(&sealed)
    }

    /// A job's status snapshot.
    pub async fn status(&self, id: &str) -> Result<JobStatus, Error> {
        Self::json(self.req(reqwest::Method::GET, &format!("/jobs/{id}"))).await
    }

    /// Every job the server holds (structural fields only).
    pub async fn list_jobs(&self) -> Result<JobList, Error> {
        Self::json(self.req(reqwest::Method::GET, "/jobs")).await
    }

    /// Ask a job to stop (a queued one is dropped).
    pub async fn cancel(&self, job: &Job) -> Result<JobStatus, Error> {
        Self::json(
            self.req(reqwest::Method::POST, &format!("/jobs/{}/cancel", job.id))
                .header(JOB_PROOF, job.key.proof_b64()),
        )
        .await
    }

    /// Change a queued job's priority.
    pub async fn set_priority(&self, job: &Job, priority: i32) -> Result<JobStatus, Error> {
        Self::json(
            self.req(reqwest::Method::PATCH, &format!("/jobs/{}", job.id))
                .header(JOB_PROOF, job.key.proof_b64())
                .json(&JobPatch { priority }),
        )
        .await
    }

    /// Forget a job (cancelling it if it runs), its result included.
    pub async fn delete(&self, job: &Job) -> Result<(), Error> {
        Self::send(
            self.req(reqwest::Method::DELETE, &format!("/jobs/{}", job.id))
                .header(JOB_PROOF, job.key.proof_b64()),
        )
        .await
        .map(drop)
    }

    // ---- sessions ----

    /// Open a text session under a fresh key. `model` defaults to the
    /// server's text default, `ttl_secs` (idle time before the server forgets
    /// it) to its `session_ttl_secs`.
    pub async fn create_session(
        &self,
        model: Option<TextModelId>,
        ttl_secs: Option<u64>,
    ) -> Result<Session, Error> {
        let key = JobKey::generate();
        let info: SessionInfo = Self::json(self.req(reqwest::Method::POST, "/sessions").json(
            &CreateSession {
                model,
                ttl_secs,
                key_proof: key.proof_b64(),
            },
        ))
        .await?;
        Ok(Session { id: info.id, key })
    }

    fn session_req(
        &self,
        method: reqwest::Method,
        s: &Session,
        rest: &str,
    ) -> reqwest::RequestBuilder {
        self.req(method, &format!("/sessions/{}{rest}", s.id))
            .header(SESSION_PROOF, s.key.proof_b64())
    }

    /// A session's structure: nodes, sizes, expiry (no content).
    pub async fn session_info(&self, s: &Session) -> Result<SessionInfo, Error> {
        Self::json(self.session_req(reqwest::Method::GET, s, "")).await
    }

    /// Change a session's idle TTL (also refreshes it).
    pub async fn set_session_ttl(&self, s: &Session, ttl_secs: u64) -> Result<SessionInfo, Error> {
        Self::json(
            self.session_req(reqwest::Method::PATCH, s, "")
                .json(&SessionPatch { ttl_secs }),
        )
        .await
    }

    /// Forget a session and all its state now.
    pub async fn delete_session(&self, s: &Session) -> Result<(), Error> {
        Self::send(self.session_req(reqwest::Method::DELETE, s, ""))
            .await
            .map(drop)
    }

    /// Prune a node and its subtree (e.g. an abandoned branch).
    pub async fn delete_session_node(&self, s: &Session, node: &str) -> Result<SessionInfo, Error> {
        Self::json(self.session_req(reqwest::Method::DELETE, s, &format!("/nodes/{node}"))).await
    }

    /// The whole session as one blob, every node still sealed under the
    /// session key: safe to store; [`Self::import_session`] restores it.
    pub async fn export_session(&self, s: &Session) -> Result<Vec<u8>, Error> {
        Self::bytes(self.session_req(reqwest::Method::GET, s, "/export")).await
    }

    /// Restore an exported session (under a new id; the same key, nodes and
    /// TTL). `key` is the session's key.
    pub async fn import_session(&self, export: &[u8], key: JobKey) -> Result<Session, Error> {
        let info: SessionInfo = Self::json(
            self.req(reqwest::Method::POST, "/sessions/import")
                .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
                .body(export.to_vec()),
        )
        .await?;
        Ok(Session { id: info.id, key })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_http_base_url() {
        assert!(Client::new("box:8080").is_err());
        assert_eq!(
            Client::new("http://box:8080/").unwrap().base_url(),
            "http://box:8080"
        );
    }
}
