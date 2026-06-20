//! Wan-family video models. First target: SkyReels-V2-DF-1.3B-540P.
//!
//! Shared text encoder is umT5-XXL (`umt5`); the Wan DiT block and 3D causal
//! VAE land alongside it as the port proceeds. See
//! `projects/thinfer-working-area/wan-plan.md`.

pub mod condition_embedder;
pub mod dit;
pub mod dit_block;
pub mod kv_cache;
pub mod loader;
pub mod manifest;
pub mod patchify;
pub mod pipeline;
pub mod rope3d;
pub mod scheduler;
pub mod source;
pub mod umt5;
pub mod unipc;
pub mod vae;
pub mod vae_tiny;
