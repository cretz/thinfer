//! Qwen-Image-2.1-Turbo parity suite. Gated on `qwen-image-e2e` (needs the
//! unsloth GGUF in cache + `uv` + the third-party diffusers clone).

mod dit_parity;
mod encoder_parity;
#[path = "../qwen_image/parity_util.rs"]
mod parity_util;
mod vae_parity;
