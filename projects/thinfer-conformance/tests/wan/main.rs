//! Per-model integration suite for SkyReels-V2-DF-1.3B (Wan). Bundled into one
//! test binary because cargo only auto-discovers integration tests directly
//! under `tests/` (a `tests/wan/main.rs` dir-crate, mirroring `tests/zimage`).
//! Opt-in via the `wan-e2e` feature; run manually, never under default
//! `cargo test`.

mod video_e2e_parity;
