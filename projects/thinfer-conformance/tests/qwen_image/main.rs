//! Per-model integration suite for Qwen-Image-Edit-Rapid-AIO. Bundled into one
//! test binary (cargo auto-discovers `tests/<dir>/main.rs`); submodules are the
//! individual parity tests. Gated on the `qwen-image-e2e` feature (needs the HF
//! GGUF bundle in cache + `uv`).

mod dit_parity;
mod dit_perf;
mod e2e;
mod edit_e2e;
mod encoder_edit_parity;
mod encoder_parity;
mod parity_util;
mod vae_encode_parity;
mod vae_parity;
mod vision_parity;
