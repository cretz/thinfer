//! Unified progress vocabulary. The per-model pipelines emit their own
//! `ProgressEvent` enums (`z_image::pipeline`, `wan::pipeline`); the executor
//! maps those plus the face-swap stream into one [`Stage`] set that every front
//! end consumes: the CLI stamps each to stderr, and (later) `thinfer-serve`
//! turns each into one SSE event. Free-form narration that is not a discrete
//! step (audio passthrough notes, "downscaling output", ...) goes through
//! [`ProgressSink::note`] so nothing the CLI used to print is lost.

/// A discrete, structured progress milestone. Stable enough to serialize as the
/// SSE `progress` payload; map every front end off this, never off stdout text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Text encoder pass (umT5 / Qwen) over the prompt.
    TextEncode,
    /// Single-shot denoise step `i` of `n` (image, FastWan video).
    Step { i: u32, n: u32 },
    /// AR/chunked denoise: `step` of `num_steps` within `chunk` of `num_chunks`
    /// (LongLive).
    ChunkStep {
        chunk: u32,
        num_chunks: u32,
        step: u32,
        num_steps: u32,
    },
    /// VAE decode of the final latents to pixels.
    VaeDecode,
    /// Face-swap streamed `done` of `total` frames swapped.
    FrameSwapped { done: u32, total: u32 },
}

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
    /// Cooperative cancellation: the generate loop polls this at step
    /// boundaries and aborts when it returns `true`. Default `false` (the CLI
    /// runs to completion); the server sink returns the job's cancel flag.
    fn cancelled(&self) -> bool {
        false
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
    bytes: &[u8],
) -> Result<(), String> {
    if sink.artifact(path, bytes)? {
        return Ok(());
    }
    std::fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}

/// A sink that drops everything. Useful for non-interactive callers and tests.
pub struct NullSink;

impl ProgressSink for NullSink {
    fn stage(&self, _stage: Stage) {}
}
