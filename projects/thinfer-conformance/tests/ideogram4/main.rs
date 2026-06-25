//! Per-model integration suite for Ideogram-4. Bundled into one test binary
//! (cargo auto-discovers `tests/<dir>/main.rs`); submodules are the individual
//! parity tests.

mod dit_parity;
mod e2e_parity;
mod encoder_parity;
mod parity_util;
mod vae_parity;
