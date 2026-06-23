//! GPU worker. Each worker runs on its own OS thread with a current-thread
//! tokio runtime so the (potentially `!Send`) model generate futures run
//! directly via `block_on` -- no `Send` bound, no cross-thread GPU handoff. The
//! worker owns one [`LocalExecutor`] (one resident model), pulls jobs from the
//! shared [`JobStore`], and reports progress into the job's event log.

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
                let executor = match LocalExecutor::new(backend_cfg).await {
                    Ok(e) => e,
                    Err(e) => {
                        tracing::error!("worker {index} backend init failed: {e}");
                        return;
                    }
                };
                tracing::info!("worker {index} ready");
                worker_loop(store, executor, download_as_needed).await;
            });
        })
        .expect("spawn worker thread");
}

async fn worker_loop(store: Arc<JobStore>, executor: LocalExecutor, download_as_needed: bool) {
    loop {
        match store.take_next() {
            Some(job) => {
                process(&executor, &job, download_as_needed).await;
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
        Err(message) => return job.push(JobEvent::Error { message }),
    };

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
}
