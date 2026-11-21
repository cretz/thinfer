//! The progress vocabulary every job reports in.

/// A discrete, structured progress milestone. Stable enough to serialize as the
/// SSE `progress` payload; map every front end off this, never off stdout text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Text encoder pass (umT5 / Qwen) over the prompt.
    TextEncode,
    /// Single-shot denoise step `i` of `n` (image, FastWan video).
    Step { i: u32, n: u32 },
    /// Chunked denoise: `step` of `num_steps` within `chunk` of `num_chunks`
    /// (HunyuanVideo causal I2V).
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
