//! GPU worker. Each worker runs on its own OS thread with a current-thread
//! tokio runtime so the (potentially `!Send`) model generate futures run
//! directly via `block_on` -- no `Send` bound, no cross-thread GPU handoff. The
//! worker pulls jobs from the shared [`JobStore`] and reports progress into the
//! job's event log.
//!
//! ## A fresh GPU device per job (crash isolation, no VRAM leak)
//!
//! The worker builds a NEW [`LocalExecutor`] -- and therefore a new wgpu device
//! -- for EACH job, and drops it when the job finishes. Nothing is kept resident
//! across jobs: the executor holds only the device handle, and every model's
//! weights are streamed in per request anyway (scoped residency), so a warm
//! cache would buy nothing. What a long-lived device DID buy was a failure mode:
//! a device-OOM / device-lost poisons the wgpu device, which then holds its VRAM
//! allocations for the life of the process -- every later job fails until the
//! server is killed. Recreating per job means that poison is reclaimed the
//! instant the job ends; the next job gets a clean device. Device + pipeline
//! creation is ~sub-second (the driver shader cache survives device recreation),
//! negligible against a multi-minute generate.

use std::sync::Arc;
use std::time::Duration;

use thinfer_app::JobRequest;
use thinfer_app::config::BackendConfig;
use thinfer_app::download::{self, DownloadReporter, NullReporter};
use thinfer_app::jobkey::JobKey;
use thinfer_app::progress::{ProgressSink, Stage};
use thinfer_app::wire::{JobEvent, JobResult, JobSpec};
use thinfer_app::{JobSummary, LocalExecutor};

use crate::config::ServeConfig;
use crate::job::{JobHandle, JobStore};
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

async fn worker_loop(ctx: WorkerCtx) {
    loop {
        match ctx.store.take_next() {
            Some(job) => {
                run_job(&ctx, &job).await;
                ctx.store.finish();
                // A freed worker may unblock another queued job.
                ctx.store.notify.notify_one();
            }
            None => ctx.store.notify.notified().await,
        }
    }
}

/// One job, end to end. The spec is ciphertext until the key arrives; the
/// decrypted request, the key, and every plaintext buffer this owns are dropped
/// (keys/specs zeroed) when this returns.
async fn run_job(ctx: &WorkerCtx, job: &Arc<JobHandle>) {
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
    let (request, disable_coopmat) = match decrypt_request(ctx, job, &key, &sealed) {
        Ok(r) => r,
        Err(message) => return job.push(JobEvent::Error { message }),
    };
    drop(sealed);
    // Fresh device per job (see module docs); dropped at the end of this
    // function, releasing all VRAM even if the job poisoned it. The
    // per-request coopmat opt-out layers over the server default.
    let mut cfg: BackendConfig = ctx.config.backend_config();
    if let Some(disable) = disable_coopmat {
        cfg.disable_coopmat = disable;
    }
    match LocalExecutor::new(cfg).await {
        Ok(executor) => {
            process(
                &executor,
                job,
                &request,
                &key,
                ctx.config.download_as_needed,
            )
            .await
        }
        Err(e) => job.push(JobEvent::Error {
            message: format!("backend init: {e}"),
        }),
    }
}

/// Open the sealed spec with the job key (plaintext zeroed on drop) and build
/// the executable request.
fn decrypt_request(
    ctx: &WorkerCtx,
    job: &JobHandle,
    key: &JobKey,
    sealed: &[u8],
) -> Result<(JobRequest, Option<bool>), String> {
    let spec_json = key.open(sealed)?;
    let spec: JobSpec =
        serde_json::from_slice(&spec_json).map_err(|e| format!("invalid job spec: {e}"))?;
    crate::jobspec::spec_into_request(spec, &job.dir, key, &ctx.config, &ctx.uploads)
}

async fn process(
    executor: &LocalExecutor,
    job: &Arc<JobHandle>,
    request: &JobRequest,
    key: &JobKey,
    download_as_needed: bool,
) {
    job.push(JobEvent::Started);
    // Durable job record (structural fields ONLY -- prompts are never logged).
    tracing::info!(id = %job.id, "job started");
    let t0 = std::time::Instant::now();

    let files = match request.required_files() {
        Ok(f) => f,
        Err(e) => return job.push(JobEvent::Error { message: e }),
    };
    let missing = match download::missing(&files) {
        Ok(m) => m,
        Err(e) => return job.push(JobEvent::Error { message: e }),
    };
    if !missing.is_empty() {
        if !download_as_needed {
            return job.push(JobEvent::Error {
                message: format!(
                    "{} weight file(s) missing and download_as_needed=false",
                    missing.len()
                ),
            });
        }
        job.push(JobEvent::Log {
            message: format!("Downloading {} missing file(s)", missing.len()),
        });
        if let Err(e) = download::ensure(&files, &NullReporter as &dyn DownloadReporter).await {
            return job.push(JobEvent::Error { message: e });
        }
    }

    let sink = ServeSink {
        job: job.clone(),
        key,
    };
    let summary = match executor.run(request, &sink).await {
        Ok(summary) => summary,
        // A cancel-requested job that errored out was interrupted by the user's
        // cancel (the denoise loop bails with GenerateError::Cancelled); report
        // it as Cancelled, not a failure. Any other error is a real failure.
        Err(message) => {
            return if job.is_cancel_requested() {
                tracing::info!(id = %job.id, "job cancelled");
                job.push(JobEvent::Cancelled)
            } else {
                tracing::warn!(id = %job.id, %message, "job failed");
                job.push(JobEvent::Error { message })
            };
        }
    };
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

    job.push(JobEvent::Done {
        result: to_result(&job.id, &summary),
    });
}

fn to_result(id: &str, s: &JobSummary) -> JobResult {
    JobResult {
        width: s.width,
        height: s.height,
        frames: s.frames,
        fps: s.fps,
        seed: s.seed,
        result_url: format!("/jobs/{id}/result"),
    }
}

/// The sealed artifact's file name in the job dir.
pub const RESULT_FILE: &str = "result.sealed";

/// Routes engine progress into the job's event log, and seals the artifact
/// under the job key in RAM so only ciphertext reaches disk.
struct ServeSink<'k> {
    job: Arc<JobHandle>,
    key: &'k JobKey,
}

impl ProgressSink for ServeSink<'_> {
    fn artifact(&self, _path: &std::path::Path, bytes: &[u8]) -> Result<bool, String> {
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
    fn cancelled(&self) -> bool {
        self.job.is_cancel_requested()
    }
}
