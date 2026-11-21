//! Unified progress vocabulary. The per-model pipelines emit their own
//! `ProgressEvent` enums (`z_image::pipeline`, `wan::pipeline`); the executor
//! maps those plus the face-swap stream into one [`Stage`] set that every front
//! end consumes: the CLI stamps each to stderr, and (later) `thinfer-serve`
//! turns each into one SSE event. Free-form narration that is not a discrete
//! step (audio passthrough notes, "downscaling output", ...) goes through
//! [`ProgressSink::note`] so nothing the CLI used to print is lost.

pub use thinfer_client::progress::Stage;

/// Where progress goes. Implementors must be cheap and non-panicking: the
/// generate loop calls `stage`/`note` inline. A CLI sink stamps + prints to
/// stderr; a server sink pushes onto a job's event buffer.
pub trait ProgressSink {
    /// A structured milestone.
    fn stage(&self, stage: Stage);
    /// Free-form narration (kept for CLI fidelity; a server maps it to a log
    /// event). Default no-op so sinks that only care about structured stages
    /// can ignore it.
    fn note(&self, _msg: &str) {}
    /// A streamed piece of generated text (text jobs). Default no-op.
    fn text(&self, _delta: &str) {}
    /// Cooperative cancellation: the generate loop polls this at step
    /// boundaries and aborts when it returns `true`. Default `false` (the CLI
    /// runs to completion); the server sink returns the job's cancel flag.
    fn cancelled(&self) -> bool {
        false
    }
    /// A pause the job's owner asked for (the server; see
    /// [`crate::checkpoint`]). Default none: the run goes to the end.
    fn pause(&self) -> Option<crate::checkpoint::PauseMode> {
        None
    }
    /// Keep a pausing run's state (the server seals it under the job key
    /// before it reaches disk); the run returns
    /// [`crate::checkpoint::SUSPENDED`] right after. Default: refused.
    fn suspend(&self, _state: &[u8]) -> Result<(), String> {
        Err("this run cannot pause".into())
    }
    /// The state a paused run left, page-locked: taken by the run that
    /// resumes it. Default none (a fresh run).
    fn take_resume(&self) -> Option<thinfer_core::secmem::Locked<u8>> {
        None
    }
    /// Take ownership of the finished artifact's bytes (encoded PNG / MP4)
    /// before anything is written to `path`. `Ok(true)` = the sink stored it
    /// (a server seals it under the job key in RAM and writes only
    /// ciphertext); `Ok(false)` (the default, the CLI) = the caller writes the
    /// plaintext file itself.
    fn artifact(&self, _path: &std::path::Path, _bytes: &[u8]) -> Result<bool, String> {
        Ok(false)
    }
}

/// Deliver a finished artifact: through the sink when it takes it, else as a
/// plain file at `path`.
pub fn write_artifact(
    sink: &dyn ProgressSink,
    path: &std::path::Path,
    bytes: Vec<u8>,
) -> Result<(), String> {
    // The finished output is job plaintext until the sink seals it: locked
    // while it is sealed, zeroized after.
    let bytes = thinfer_core::secmem::Locked::new(bytes);
    if sink.artifact(path, &bytes)? {
        return Ok(());
    }
    std::fs::write(path, &*bytes).map_err(|e| format!("write {}: {e}", path.display()))
}

/// A sink that drops everything. Useful for non-interactive callers and tests.
pub struct NullSink;

impl ProgressSink for NullSink {
    fn stage(&self, _stage: Stage) {}
}
