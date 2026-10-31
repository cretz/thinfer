//! Cross-model primitives shared by the per-model pipelines (z_image, wan, ...).
//!
//! These are the model-agnostic building blocks: the DiT transformer block and
//! its op/pipeline scaffolding (`block`), the bias-Linear helpers and their
//! residency views (`embedders`), the residency-aware weight registration
//! primitives (`loader`), the CPU RoPE table (`rope_embedder`), and the
//! activation dtype / mask / readback helpers (`seq`). Per-model glue
//! (patchify, weight-name maps, the model's own embedders) stays in the model
//! module.

pub mod autoencoder_kl;
pub mod bigvgan;
pub mod block;
pub mod embedders;
pub mod loader;
pub mod lora;
pub mod merge;
pub mod rope_embedder;
pub mod seq;
pub mod vae_taehv;
pub mod vae_tiling;

/// Cooperative cancellation hook. Returns `true` when the caller wants the
/// in-flight generate to stop; the denoise loops poll it at step (and, where a
/// step is long, block) boundaries and bail with their `Cancelled` error.
/// `None` (CLI default) never cancels. The server passes a closure backed by
/// the job's cancel flag.
pub type Cancel<'a> = Option<&'a dyn Fn() -> bool>;
