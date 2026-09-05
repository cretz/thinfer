//! Per-model integration suite for MiniMax-H3 (33B joint audio-video).
//! Bundled into one test binary (cargo auto-discovers `tests/<dir>/main.rs`).
//! Gated on `minimax-e2e`: touches the local HF cache, skips cleanly when the
//! weights are not there and never triggers a download.

mod adaln_precompute;
mod audio_vae_health;
mod dit_health;
mod video_vae_health;
