//! Per-model integration suite for LTX-2.3 distilled-1.1 (22B joint audio-video).
//! Bundled into one test binary (cargo auto-discovers `tests/<dir>/main.rs`);
//! submodules are the individual parity tests. Gated on the `ltx-e2e` feature
//! (needs the unsloth LTX-2.3 + gemma Q8_0 GGUF bundle in cache + `uv`).

mod audio_vae_parity;
mod connector_parity;
mod dit_parity;
mod dit_perf;
mod e2e;
mod encoder_parity;
mod encoder_perf;
mod parity_util;
mod upsampler_parity;
mod vae_parity;
mod vocoder_parity;
