//! The remote execution path: a `thinfer-serve` box through
//! [`thinfer_client::Client`]. It is the mirror of
//! [`crate::executor::LocalExecutor`] -- same job, same progress vocabulary --
//! but the work runs on the server: the client's updates go into the caller's
//! [`ProgressSink`] (so the CLI renders the exact same stderr lines as a local
//! run), and the finished artifact is written where a local run would put it.
//!
//! Every job is end-to-end encrypted (see [`thinfer_client::jobkey`]).

use std::path::Path;

use thinfer_client::wire::JobSpec;
use thinfer_client::{Client, JobKey, SubmitOptions, Update};

use crate::progress::ProgressSink;
use crate::request::JobSummary;

/// A `thinfer-serve` deployment. Construct once with the base URL (and an
/// optional bearer token); reuse across jobs.
pub struct RemoteExecutor {
    client: Client,
    /// Queue priority of submitted jobs (higher first; default 0).
    priority: i32,
}

impl RemoteExecutor {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            priority: 0,
        }
    }

    /// Submit jobs at this queue priority (higher runs first).
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Submit `spec`, stream progress into `sink`, and write the result to
    /// `output` on success (a text job's reply only when `output` is set: it
    /// already streamed through the sink). Returns the summary the server
    /// reported.
    pub async fn run(
        &self,
        spec: &JobSpec,
        output: &Path,
        sink: &dyn ProgressSink,
    ) -> Result<JobSummary, String> {
        let opts = SubmitOptions {
            priority: self.priority,
        };
        let job = self.client.submit(spec, JobKey::generate(), &opts).await?;
        sink.note(&format!("Submitted job {}", job.id));
        let finished = self
            .client
            .follow(&job, |u| match u {
                Update::Queued { position } => sink.note(&format!("Queued at position {position}")),
                Update::Started => sink.note("Started"),
                Update::Progress(stage) => sink.stage(stage),
                Update::Log(message) => sink.note(message),
                Update::Text(delta) => sink.text(delta),
            })
            .await?;
        // Fetched either way: the fetch is what deletes the job server-side.
        let artifact = self.client.fetch_result(&job).await?;
        let result = finished.result;
        if result.text.is_none() || !output.as_os_str().is_empty() {
            if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|e| format!("create {}: {e}", parent.display()))?;
            }
            tokio::fs::write(output, &*artifact)
                .await
                .map_err(|e| format!("write {}: {e}", output.display()))?;
        }
        Ok(JobSummary {
            text: result.text.map(|t| crate::request::TextStats {
                prompt_tokens: t.prompt_tokens,
                generated_tokens: t.generated_tokens,
                prefill_secs: t.prefill_secs,
                decode_secs: t.decode_secs,
                finish_reason: t.finish_reason,
            }),
            tool_calls: finished.tool_calls,
            output: output.to_path_buf(),
            width: result.width,
            height: result.height,
            frames: result.frames,
            fps: result.fps,
            seed: result.seed,
            session: None,
        })
    }
}
