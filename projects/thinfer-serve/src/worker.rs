//! GPU worker. Each worker runs on its own OS thread with a current-thread
//! tokio runtime so the (potentially `!Send`) model generate futures run
//! directly via `block_on` -- no `Send` bound, no cross-thread GPU handoff. The
//! worker pulls jobs from the shared [`JobStore`] and reports progress into the
//! job's event log.
//!
//! ## A warm text / audio model between jobs (no job data, no poison)
//!
//! A text job's fixed costs (tokenizer, kernels: the CUDA LLM set takes ~3 s
//! to build per device, weights, the expert cache) dwarf a short reply, so a
//! worker keeps its [`LocalExecutor`], with the last text model on it, until
//! `warm_text_secs` after a text job last used it; likewise the last speech
//! model with `warm_audio_secs` (its load is most of a short line's time).
//! Opt-in (default 0: every job's device goes when it ends, so an idle server
//! holds no GPU memory). Image and video models are never kept: a device
//! holding no warm text / audio model goes after its job. What
//! a long-lived device used to risk is excluded by rule:
//! - Poison: a device-OOM / device-lost poisons a wgpu device, which then
//!   holds its VRAM for good. Any job that does not finish cleanly discards
//!   the device; the next job gets a fresh one.
//! - Job data: only pristine model data stays. A job must leave no buffer
//!   behind ([`WgpuBackend::live_buffers`] back to zero, the warm model's
//!   weights aside), or the device is discarded.
//! - Other jobs: an image/video job frees the text model first (all of it,
//!   or with `keep_text_host` its VRAM only), so it gets the whole device.
//!
//! ## Paused jobs
//!
//! A run asked to pause hands its state to the sink at a step boundary
//! ([`thinfer_app::checkpoint`]); the worker seals it under the job key into
//! the job dir, gives the sealed spec back to the job, and requeues (or holds)
//! it. The resumed run asks for the key again, opens the state into
//! page-locked memory, deletes the file, and continues from it.
//!
//! [`WgpuBackend::live_buffers`]: thinfer_core::backend::WgpuBackend::live_buffers

use std::sync::Arc;
use std::time::Duration;

use thinfer_app::JobRequest;
use thinfer_app::config::BackendConfig;
use thinfer_app::download::{self, DownloadReporter, NullReporter};
use thinfer_app::jobkey::{JobKey, OpenLocked};
use thinfer_app::progress::{ProgressSink, Stage};
use thinfer_app::wire::{JobEvent, JobResult, JobSpec};
use thinfer_app::{JobSummary, LocalExecutor};
use thinfer_core::mem::VramCategory;

use crate::config::ServeConfig;
use crate::job::{JobHandle, JobStore};
use crate::jobspec::{Built, SessionCommit};
use crate::sessions::SessionStore;
use crate::uploads::UploadStore;

/// Worker thread stack. The default spawned-thread stack (2 MiB) is NOT enough:
/// model load builds a pipeline set inside one deep async fn, so the whole
/// generate future (every nested await's locals, live at once) sits in this
/// thread's frame. Overflowing it kills the PROCESS with no panic message and no
/// log line, which reads as a silent server crash mid-load. The CLI never hit
/// this because it generates on the main thread (8 MiB from the exe header).
/// Sized well past the current high-water mark so adding pipelines/awaits to a
/// model does not resurrect the crash.
const WORKER_STACK_BYTES: usize = 64 * 1024 * 1024;

/// What a worker needs besides the queue: the config (budgets, key wait) and
/// the upload store (sealed uploads a job consumes once decrypted).
#[derive(Clone)]
pub struct WorkerCtx {
    pub store: Arc<JobStore>,
    pub config: Arc<ServeConfig>,
    pub uploads: Arc<UploadStore>,
    pub sessions: Arc<SessionStore>,
}

/// Spawn a worker on a dedicated OS thread. It builds its own backend +
/// executor inside a current-thread runtime, then loops over the queue forever.
pub fn spawn_worker(index: usize, ctx: WorkerCtx) {
    std::thread::Builder::new()
        .name(format!("thinfer-worker-{index}"))
        .stack_size(WORKER_STACK_BYTES)
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build worker runtime");
            rt.block_on(async move {
                tracing::info!("worker {index} ready");
                worker_loop(ctx).await;
            });
        })
        .expect("spawn worker thread");
}

/// A worker's device kept between jobs (holding a warm text model), and the
/// per-job setting it was built with.
struct Warm {
    executor: LocalExecutor,
    disable_coopmat: bool,
}

/// The warm-model TTLs (text, audio).
fn ttls(config: &ServeConfig) -> (Duration, Duration) {
    (
        Duration::from_secs(config.warm_text_secs),
        Duration::from_secs(config.warm_audio_secs),
    )
}

async fn worker_loop(ctx: WorkerCtx) {
    let (text_ttl, audio_ttl) = ttls(&ctx.config);
    let mut warm: Option<Warm> = None;
    loop {
        match ctx.store.take_next() {
            Some(job) => {
                // A panicking job fails alone: the worker lives on, and the
                // device (whatever state the panic left it in) goes.
                if CatchPanic(Box::pin(run_job(&ctx, &job, &mut warm)))
                    .await
                    .is_err()
                {
                    tracing::error!(id = %job.id, "job panicked; device discarded");
                    warm = None;
                    job.push(JobEvent::Error {
                        message: "internal error: the job panicked".into(),
                    });
                }
                ctx.store.finish();
                // A freed worker may unblock another queued job.
                ctx.store.notify.notify_one();
            }
            None => {
                // Idle with a warm model: each goes when its time since last
                // use reaches its TTL, the device with the last one.
                let left = warm
                    .as_ref()
                    .and_then(|w| w.executor.warm_time_left(text_ttl, audio_ttl));
                match left {
                    Some(left) => {
                        if tokio::time::timeout(left, ctx.store.notify.notified())
                            .await
                            .is_err()
                            && let Some(w) = &warm
                        {
                            w.executor.expire_warm(text_ttl, audio_ttl);
                            if !w.executor.holds_warm_model() {
                                warm = None;
                                tracing::info!("warm models and device released (idle)");
                            }
                        }
                    }
                    None => ctx.store.notify.notified().await,
                }
            }
        }
    }
}

/// Polls a job future, turning a panic inside it into `Err`.
struct CatchPanic<F>(std::pin::Pin<Box<F>>);

impl<F: std::future::Future> std::future::Future for CatchPanic<F> {
    type Output = Result<F::Output, ()>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let inner = self.0.as_mut();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| inner.poll(cx))) {
            Ok(std::task::Poll::Ready(v)) => std::task::Poll::Ready(Ok(v)),
            Ok(std::task::Poll::Pending) => std::task::Poll::Pending,
            Err(_) => std::task::Poll::Ready(Err(())),
        }
    }
}

/// One job, end to end. The spec is ciphertext until the key arrives; the
/// decrypted request, the key, and every plaintext buffer this owns are dropped
/// (keys/specs zeroed) when this returns.
async fn run_job(ctx: &WorkerCtx, job: &Arc<JobHandle>, warm: &mut Option<Warm>) {
    let Some(sealed) = job.take_sealed() else {
        return job.push(JobEvent::Error {
            message: "job already taken".into(),
        });
    };
    let wait = Duration::from_secs(ctx.config.key_wait_secs);
    let Some(key) = job.request_key(wait).await else {
        return job.push(if job.is_cancel_requested() {
            JobEvent::Cancelled
        } else {
            JobEvent::Error {
                message: format!("no job key within {}s; the job was not run", wait.as_secs()),
            }
        });
    };
    // Job plaintext is page-locked from here on; a lock that fails (no
    // lockable memory left) stops the job rather than let it reach swap.
    let lock_mark = thinfer_core::secmem::failure_mark();
    let Built {
        request,
        disable_coopmat,
        session,
    } = match decrypt_request(ctx, job, &key, &sealed) {
        Ok(r) => r,
        Err(message) => return job.push(JobEvent::Error { message }),
    };
    // Its outputs (and a pause's state) live in the job dir.
    if let Err(e) = std::fs::create_dir_all(&job.dir) {
        return job.push(JobEvent::Error {
            message: format!("create {}: {e}", job.dir.display()),
        });
    }
    // Paused before it ran (taken while waiting for its key): given back.
    if let Some(pause) = job.take_pause() {
        job.restore_sealed(sealed);
        ctx.store.requeue(job, pause);
        return;
    }
    job.set_pausable(request.pausable());
    // A resumed job: its state, opened into page-locked memory (the sealed
    // file goes; a later pause writes a new one).
    let resume = match read_checkpoint(job, &key) {
        Ok(r) => r,
        Err(message) => {
            job.set_pausable(false);
            return job.push(JobEvent::Error { message });
        }
    };
    // The warm device when it matches this job's setup, else a fresh one (see
    // module docs). The per-request coopmat opt-out layers over the server
    // default.
    let mut cfg: BackendConfig = ctx.config.backend_config();
    if let Some(disable) = disable_coopmat {
        cfg.disable_coopmat = disable;
    }
    let executor = match warm.take() {
        Some(w) if w.disable_coopmat == cfg.disable_coopmat => w.executor,
        stale => {
            // Released before the new device is built.
            drop(stale);
            let disable = cfg.disable_coopmat;
            match LocalExecutor::new(cfg).await {
                Ok(executor) => {
                    tracing::info!(disable_coopmat = disable, "device created");
                    executor.keep_text_host(ctx.config.keep_text_host)
                }
                Err(e) => {
                    job.set_pausable(false);
                    return job.push(JobEvent::Error {
                        message: format!("backend init: {e}"),
                    });
                }
            }
        }
    };
    let (text_ttl, audio_ttl) = ttls(&ctx.config);
    executor.expire_warm(text_ttl, audio_ttl);
    let ran = process(
        &executor,
        job,
        &request,
        &key,
        ctx.config.download_as_needed,
        lock_mark,
        resume,
        session.map(|c| (c, ctx.sessions.as_ref())),
    )
    .await;
    job.set_pausable(false);
    let clean = ran != Ran::Failed;
    if ran == Ran::Paused {
        job.restore_sealed(sealed);
        ctx.store.requeue(job, job.take_pause().unwrap_or_default());
    }
    // A model whose TTL is 0 goes now, before the buffer check (its
    // weights must not count as kept).
    executor.expire_warm(text_ttl, audio_ttl);
    // Keep the device only after a clean job that left no buffer behind
    // (a kept model's weights aside).
    let except: &[VramCategory] = if executor.holds_warm_weights() {
        &[VramCategory::Weights]
    } else {
        &[]
    };
    let (left, bytes) = executor.backend().live_buffers(except);
    // Kept for a warm text / audio model only (image and video never).
    if clean && left == 0 && executor.holds_warm_model() {
        *warm = Some(Warm {
            executor,
            disable_coopmat: cfg.disable_coopmat,
        });
    } else if clean && left > 0 {
        tracing::warn!(
            buffers = left,
            bytes,
            "job left buffers behind; device discarded"
        );
    }
}

/// A paused job's state, if it has one: opened with the job key into
/// page-locked memory, the sealed file deleted.
fn read_checkpoint(
    job: &JobHandle,
    key: &JobKey,
) -> Result<Option<thinfer_core::secmem::Locked<u8>>, String> {
    let path = job.checkpoint_path();
    let sealed = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("read the paused job's state: {e}")),
    };
    let _ = std::fs::remove_file(&path);
    key.open_locked(&sealed)
        .map(Some)
        .map_err(|e| format!("open the paused job's state: {e}"))
}

/// How a job's run went.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Ran {
    Done,
    Failed,
    /// It stopped at a checkpoint and gives up its worker.
    Paused,
    /// The owner cancelled it (a clean stop).
    Cancelled,
}

/// Open the sealed spec with the job key (plaintext zeroed on drop) and build
/// the executable request.
fn decrypt_request(
    ctx: &WorkerCtx,
    job: &JobHandle,
    key: &JobKey,
    sealed: &[u8],
) -> Result<Built, String> {
    let spec_json = key.open_locked(sealed)?;
    let spec: JobSpec =
        serde_json::from_slice(&spec_json).map_err(|e| format!("invalid job spec: {e}"))?;
    crate::jobspec::spec_into_request(
        spec,
        &job.id,
        &job.dir,
        key,
        &ctx.config,
        &ctx.uploads,
        &ctx.sessions,
    )
}

async fn process(
    executor: &LocalExecutor,
    job: &Arc<JobHandle>,
    request: &JobRequest,
    key: &JobKey,
    download_as_needed: bool,
    lock_mark: u64,
    resume: Option<thinfer_core::secmem::Locked<u8>>,
    session: Option<(SessionCommit, &SessionStore)>,
) -> Ran {
    job.push(JobEvent::Started);
    // Until the model's first stage: loading it (weights, kernels).
    job.push(JobEvent::Progress {
        stage: thinfer_app::wire::ProgressStage::Loading,
    });
    // Durable job record (structural fields ONLY -- prompts are never logged).
    tracing::info!(id = %job.id, "job started");
    let t0 = std::time::Instant::now();

    let files = match request.required_files() {
        Ok(f) => f,
        Err(e) => return fail(job, e),
    };
    let missing = match download::missing(&files) {
        Ok(m) => m,
        Err(e) => return fail(job, e),
    };
    if !missing.is_empty() {
        if !download_as_needed {
            return fail(
                job,
                format!(
                    "{} weight file(s) missing and download_as_needed=false",
                    missing.len()
                ),
            );
        }
        job.push(JobEvent::Log {
            message: format!("Downloading {} missing file(s)", missing.len()),
        });
        if let Err(e) = download::ensure(&files, &NullReporter as &dyn DownloadReporter).await {
            return fail(job, e);
        }
    }

    let sink = ServeSink {
        job: job.clone(),
        key,
        lock_mark,
        resume: std::sync::Mutex::new(resume),
        suspended: std::sync::atomic::AtomicBool::new(false),
    };
    let summary = match executor.run(request, &sink).await {
        Ok(summary) => summary,
        // A cancel-requested job that errored out was interrupted by the user's
        // cancel (the denoise loop bails with GenerateError::Cancelled); report
        // it as Cancelled, not a failure. Any other error is a real failure.
        Err(message) => {
            if thinfer_core::secmem::failed_since(lock_mark) {
                tracing::warn!(id = %job.id, "job stopped: job data could not be locked in RAM");
                job.push(JobEvent::Error {
                    message: LOCK_FAILED.into(),
                })
            } else if sink.suspended.load(std::sync::atomic::Ordering::SeqCst) {
                tracing::info!(id = %job.id, "job paused");
                return Ran::Paused;
            } else if job.is_cancel_requested() {
                tracing::info!(id = %job.id, "job cancelled");
                job.push(JobEvent::Cancelled);
                // A cancel is no device fault: the device may stay (the
                // no-leftover-buffers check still decides).
                return Ran::Cancelled;
            } else {
                tracing::warn!(id = %job.id, %message, "job failed");
                job.push(JobEvent::Error { message })
            }
            return Ran::Failed;
        }
    };
    // A lock that failed after the last cancel poll: the result was sealed,
    // but some plaintext was swappable, so the job does not count as done.
    if thinfer_core::secmem::failed_since(lock_mark) {
        tracing::warn!(id = %job.id, "job stopped: job data could not be locked in RAM");
        job.push(JobEvent::Error {
            message: LOCK_FAILED.into(),
        });
        return Ran::Failed;
    }
    tracing::info!(
        id = %job.id,
        width = summary.width,
        height = summary.height,
        frames = summary.frames,
        fps = summary.fps,
        seed = summary.seed,
        elapsed_s = t0.elapsed().as_secs(),
        "job done"
    );

    let node = match (session, &summary.session) {
        (Some((commit, store)), Some(save)) => save_node(job, store, commit, save),
        _ => None,
    };
    job.push(JobEvent::Done {
        result: to_result(&job.id, &summary, node, key),
    });
    Ran::Done
}

/// End a job with `message` (before it ran): not a clean finish.
fn fail(job: &JobHandle, message: String) -> Ran {
    job.push(JobEvent::Error { message });
    Ran::Failed
}

/// Seal a session reply's state under the session key and add it as a node.
/// A node that cannot be kept (budget, evicted parent) leaves the reply
/// intact; the client sees no `sessionNode` and a log line.
fn save_node(
    job: &JobHandle,
    store: &SessionStore,
    commit: SessionCommit,
    save: &thinfer_app::request::SessionSave,
) -> Option<String> {
    let sealed = commit.key.seal(&save.chunk.bytes);
    match store.insert(
        &commit.id,
        commit.parent.as_deref(),
        save.chunk.start,
        save.chunk.end,
        sealed,
        save.weights.clone(),
        save.tail.clone(),
    ) {
        Ok(node) => Some(node),
        Err(e) => {
            job.push(JobEvent::Log {
                message: format!("session node not kept: {e}"),
            });
            None
        }
    }
}

/// The job's result as the wire reports it; a reply's tool calls are sealed
/// under the job key (job data, like the reply text).
fn to_result(id: &str, s: &JobSummary, session_node: Option<String>, key: &JobKey) -> JobResult {
    use base64::Engine;
    let tool_calls = (!s.tool_calls.is_empty()).then(|| {
        let json = zeroize::Zeroizing::new(
            serde_json::to_vec(&s.tool_calls).expect("tool calls serialize"),
        );
        base64::engine::general_purpose::STANDARD.encode(key.seal(&json))
    });
    JobResult {
        width: s.width,
        height: s.height,
        frames: s.frames,
        fps: s.fps,
        seed: s.seed,
        result_url: format!("/jobs/{id}/result"),
        text: s.text.map(|t| thinfer_app::wire::TextResult {
            prompt_tokens: t.prompt_tokens,
            generated_tokens: t.generated_tokens,
            prefill_secs: t.prefill_secs,
            decode_secs: t.decode_secs,
            session_node,
            finish_reason: t.finish_reason,
            tool_calls,
        }),
        audio: s.audio.map(|a| thinfer_app::wire::AudioResult {
            seconds: a.seconds,
            sample_rate: a.sample_rate,
        }),
    }
}

/// The error a job ends with when its plaintext could not be page-locked.
const LOCK_FAILED: &str = "not enough lockable memory to keep this job's data out of swap; \
    the job was stopped (free some RAM, or raise the memory-lock limit)";

/// The sealed artifact's file name in the job dir.
pub const RESULT_FILE: &str = "result.sealed";

/// Routes engine progress into the job's event log, and seals the artifact
/// under the job key in RAM so only ciphertext reaches disk.
struct ServeSink<'k> {
    job: Arc<JobHandle>,
    key: &'k JobKey,
    /// [`thinfer_core::secmem::failure_mark`] at the job's start.
    lock_mark: u64,
    /// A resumed run's state, until the run takes it.
    resume: std::sync::Mutex<Option<thinfer_core::secmem::Locked<u8>>>,
    /// The run handed over its state to pause.
    suspended: std::sync::atomic::AtomicBool,
}

impl ProgressSink for ServeSink<'_> {
    fn artifact(&self, _path: &std::path::Path, bytes: &[u8]) -> Result<bool, String> {
        if thinfer_core::secmem::failed_since(self.lock_mark) {
            return Err(LOCK_FAILED.into());
        }
        let out = self.job.dir.join(RESULT_FILE);
        std::fs::write(&out, self.key.seal(bytes))
            .map_err(|e| format!("write {}: {e}", out.display()))?;
        Ok(true)
    }
    fn stage(&self, stage: Stage) {
        self.job.push(JobEvent::Progress {
            stage: stage.into(),
        });
    }
    fn note(&self, msg: &str) {
        self.job.push(JobEvent::Log {
            message: msg.to_string(),
        });
    }
    fn text(&self, delta: &str) {
        use base64::Engine;
        if delta.is_empty() {
            return;
        }
        self.job.push(JobEvent::Text {
            sealed: base64::engine::general_purpose::STANDARD
                .encode(self.key.seal(delta.as_bytes())),
        });
    }
    fn cancelled(&self) -> bool {
        // A failed page lock stops the job at its next poll.
        self.job.is_cancel_requested() || thinfer_core::secmem::failed_since(self.lock_mark)
    }
    fn pause(&self) -> Option<thinfer_app::checkpoint::PauseMode> {
        self.job.pause_mode()
    }
    fn suspend(&self, state: &[u8]) -> Result<(), String> {
        if thinfer_core::secmem::failed_since(self.lock_mark) {
            return Err(LOCK_FAILED.into());
        }
        // Sealed in RAM; written whole, then moved into place.
        let path = self.job.checkpoint_path();
        let tmp = path.with_extension("tmp");
        std::fs::create_dir_all(&self.job.dir)
            .and_then(|()| std::fs::write(&tmp, self.key.seal(state)))
            .and_then(|()| std::fs::rename(&tmp, &path))
            .map_err(|e| {
                let _ = std::fs::remove_file(&tmp);
                format!("keep the paused job's state: {e}")
            })?;
        self.suspended
            .store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
    fn take_resume(&self) -> Option<thinfer_core::secmem::Locked<u8>> {
        self.resume.lock().unwrap().take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What a pausing run hands over is ciphertext on disk, and the resumed
    /// run (the same key) gets it back exactly, once.
    #[test]
    fn a_suspended_state_is_sealed_at_rest_and_comes_back() {
        let tmp = std::env::temp_dir().join(format!("thinfer-ckpt-{}", std::process::id()));
        let store = crate::job::JobStore::default();
        let job = store.create(&tmp, vec![1], 0, None, None);
        let key = JobKey::generate();
        let sink = ServeSink {
            job: job.clone(),
            key: &key,
            lock_mark: thinfer_core::secmem::failure_mark(),
            resume: std::sync::Mutex::new(None),
            suspended: std::sync::atomic::AtomicBool::new(false),
        };
        let state = b"latents entering step 3".repeat(100);
        sink.suspend(&state).unwrap();
        assert!(sink.suspended.load(std::sync::atomic::Ordering::SeqCst));
        let at_rest = std::fs::read(job.checkpoint_path()).unwrap();
        assert!(
            !at_rest.windows(7).any(|w| w == b"latents"),
            "the state reached disk in the clear"
        );
        assert!(
            read_checkpoint(&job, &JobKey::generate()).is_err(),
            "another key opens it"
        );
        sink.suspend(&state).unwrap();
        let back = read_checkpoint(&job, &key).unwrap().unwrap();
        assert_eq!(&*back, &state[..]);
        assert!(!job.checkpoint_path().exists(), "the file goes once read");
        assert!(read_checkpoint(&job, &key).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
