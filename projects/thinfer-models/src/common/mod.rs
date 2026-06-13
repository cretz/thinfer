//! Cross-model primitives shared by the per-model pipelines (z_image, wan, ...).
//!
//! These are the model-agnostic building blocks: the DiT transformer block and
//! its op/pipeline scaffolding (`block`), the bias-Linear helpers and their
//! residency views (`embedders`), the residency-aware weight registration
//! primitives (`loader`), the CPU RoPE table (`rope_embedder`), and the
//! activation dtype / mask / readback helpers (`seq`). Per-model glue
//! (patchify, weight-name maps, the model's own embedders) stays in the model
//! module.

pub mod block;
pub mod embedders;
pub mod loader;
pub mod rope_embedder;
pub mod seq;
