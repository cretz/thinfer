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

use thinfer_app::config::BackendConfig;
use thinfer_app::download::{self, DownloadReporter, NullReporter};
use thinfer_app::progress::{ProgressSink, Stage};
use thinfer_app::wire::{JobEvent, JobResult};
use thinfer_app::{JobSummary, LocalExecutor};

use crate::job::{JobHandle, JobStore};

/// Spawn a worker on a dedicated OS thread. It builds its own backend +
/// executor inside a current-thread runtime, then loops over the queue forever.
pub fn spawn_worker(
    index: usize,
    store: Arc<JobStore>,
    backend_cfg: BackendConfig,
    download_as_needed: bool,
) {
    std::thread::Builder::new()
        .name(format!("thinfer-worker-{index}"))
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build worker runtime");
            rt.block_on(async move {
                tracing::info!("worker {index} ready");
                worker_loop(store, backend_cfg, download_as_needed).await;
            });
        })
        .expect("spawn worker thread");
}

async fn worker_loop(store: Arc<JobStore>, backend_cfg: BackendConfig, download_as_needed: bool) {
    loop {
        match store.take_next() {
            Some(job) => {
                // Fresh device per job (see module docs); dropped at the end of
                // this block, releasing all VRAM even if the job poisoned it.
                // Layer the per-request coopmat opt-out over the server default.
                let mut cfg = backend_cfg;
                if let Some(disable) = job.disable_coopmat {
                    cfg.disable_coopmat = disable;
                }
                match LocalExecutor::new(cfg).await {
                    Ok(executor) => process(&executor, &job, download_as_needed).await,
                    Err(e) => job.push(JobEvent::Error {
                        message: format!("backend init: {e}"),
                    }),
                }
                store.finish();
                // A freed worker may unblock another queued job.
                store.notify.notify_one();
            }
            None => store.notify.notified().await,
        }
    }
}

async fn process(executor: &LocalExecutor, job: &Arc<JobHandle>, download_as_needed: bool) {
    job.push(JobEvent::Started);
    // Durable job record (structural fields ONLY -- prompts are never logged).
    // The engine's own generate-start line rides the muted `thinfer::diag`
    // target, so without this line a lost client (closed tab + delete-on-fetch
    // artifact) has no way to recover the run's seed.
    tracing::info!(id = %job.id, "job started");
    let t0 = std::time::Instant::now();

    let files = match job.request.required_files() {
        Ok(f) => f,
        Err(e) => return job.push(JobEvent::Error { message: e }),
    };
    let missing = download::missing(&files);
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

    let sink = ServeSink { job: job.clone() };
    let summary = match executor.run(&job.request, &sink).await {
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

    // Encrypt the artifact at rest when the client supplied a public key, so the
    // on-disk copy is unreadable by the server and only the client can decrypt.
    if let Some(public_key) = &job.public_key {
        sink.note("Encrypting result");
        if let Err(e) = encrypt_artifact(&job.output_path, public_key).await {
            return job.push(JobEvent::Error {
                message: format!("encrypt result: {e}"),
            });
        }
    }
    job.push(JobEvent::Done {
        result: to_result(&job.id, &summary),
    });
}

/// Replace the plaintext artifact at `path` with its encrypted blob.
async fn encrypt_artifact(path: &std::path::Path, public_key: &str) -> Result<(), String> {
    let plaintext = tokio::fs::read(path)
        .await
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    let blob = crate::crypto::encrypt_for(public_key, &plaintext)?;
    tokio::fs::write(path, &blob)
        .await
        .map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(())
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

/// Routes engine progress into the job's event log.
struct ServeSink {
    job: Arc<JobHandle>,
}

impl ProgressSink for ServeSink {
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
