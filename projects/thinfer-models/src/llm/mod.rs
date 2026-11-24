//! Autoregressive LLM runtime: VRAM/RAM-budgeted decode + chunked prefill
//! over GGUF weights (dense weights pinned, routed experts paged through a
//! VRAM slot pool).

pub mod chat;
pub mod decoder;
pub mod dense;
pub mod experts;
pub mod gemma4;
pub mod kernels;
pub mod manifest;
pub mod qwen35;
pub mod sampler;
