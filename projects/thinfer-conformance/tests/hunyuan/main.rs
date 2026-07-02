//! Per-model integration suite for HunyuanVideo 1.5 (480p T2V, lightx2v 4-step
//! distill). Bundled into one test binary (cargo auto-discovers
//! `tests/<dir>/main.rs`); submodules are the individual parity tests. Gated on
//! the `hunyuan-e2e` feature (needs the Comfy-Org VAE + lightx2v DiT in the HF
//! cache + `uv`). See `projects/thinfer-working-area/hunyuan-plan.md`.

mod dit_parity;
mod e2e;
mod i2v_e2e;
mod parity_util;
mod perf;
mod refiner_parity;
mod vae_parity;
mod vae_tiling_parity;
