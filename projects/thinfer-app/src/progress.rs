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
}

/// A sink that drops everything. Useful for non-interactive callers and tests.
pub struct NullSink;

impl ProgressSink for NullSink {
    fn stage(&self, _stage: Stage) {}
}
