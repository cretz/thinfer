//! Per-model integration suite for FastWan2.2-TI2V-5B (Wan). Bundled into one
//! test binary because cargo only auto-discovers integration tests directly
//! under `tests/` (a `tests/wan/main.rs` dir-crate, mirroring `tests/zimage`).
//! Opt-in via the `wan-e2e` feature; run manually, never under default
//! `cargo test`.

mod anyflow_e2e;
mod dreamidv_e2e;
mod longlive_e2e;
mod longlive_load;
mod longlive_multishot_e2e;
mod longlive_parity;
mod video_e2e;
mod wan22_e2e;
