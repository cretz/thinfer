//! Qwen-Image-2.1 (Turbo): a 7B single-stream DiT for text-to-image and
//! image editing, 8 CFG-free steps. Ground truth: diffusers
//! `QwenImage21Transformer2DModel` / `QwenImage21Pipeline` /
//! `AutoencoderKLQwenImage21`.
//!
//! - **Conditioning**: stock Qwen3-VL-8B-Instruct reads the prompt (and any
//!   condition images through its vision tower); the DiT takes the last decoder
//!   layer's output BEFORE the final norm, system-prompt rows dropped.
//! - **One sequence**: the encoder rows go through `txt_in`; each condition
//!   image's `<|image_pad|>` slot (a 2x2 group of latent tokens) is replaced by
//!   four of its VAE latent tokens, and the target's latent tokens are appended.
//! - **Block-causal attention + `causal_condition`**: text is causal, every image
//!   block is bidirectional inside, and the text / condition rows (the prefix)
//!   are modulated from `t = 0`. So the prefix never sees the target and does
//!   not depend on the step: it runs ONCE per request ([`dit::Qwen21Dit::prefill`])
//!   and every step processes only the target rows against its cached K / V.
//! - **VAE**: a Wan2.2-style residual VAE with 2D kernels, 16x spatial, 64
//!   latent channels, RGBA in and out ([`vae::qwen_image21_vae`]).

pub mod dit;
pub mod loader;
pub mod lora;
pub mod manifest;
pub mod pipeline;
pub mod rope;
pub mod scheduler;
pub mod vae;

pub mod config {
    /// `heads * head_dim`.
    pub const DIM: usize = 4096;
    pub const N_LAYERS: usize = 32;
    pub const N_HEADS: usize = 32;
    pub const HEAD_DIM: usize = 128;
    /// SwiGLU inner width (`mlp_ratio` 3).
    pub const MLP_INNER: usize = 12288;
    /// VAE latent channels = DiT in / out channels (patch size 1).
    pub const LATENT_CH: usize = 64;
    /// Pixels per latent token side (the VAE's 16x, no patchify).
    pub const PIXELS_PER_TOKEN: usize = 16;
    /// Latent tokens per vision-language image slot (a 2x2 group).
    pub const TOKENS_PER_SLOT: usize = 4;
    /// Output sides must be multiples of this (one slot = 32 px).
    pub const SIDE_MULTIPLE: u32 = 32;
    pub const TIMESTEP_DIM: usize = 256;
    pub const TIMESTEP_MAX_PERIOD: f32 = 10_000.0;
    /// `time_factor`: the sinusoid reads `1000 * t`.
    pub const TIMESTEP_SCALE: f32 = 1000.0;
    pub const EPS: f32 = 1e-6;
    pub const ROPE_THETA: f32 = 10_000.0;
    /// Rotary dims per (frame, height, width) axis; sum = `HEAD_DIM`.
    pub const AXES_DIMS_ROPE: [usize; 3] = [16, 56, 56];
    /// Qwen3-VL-8B hidden width (`context_in_dim`).
    pub const TEXT_DIM: usize = 4096;
    /// Condition images are resized to this pixel area (the pipeline's
    /// `output_resolution ** 2`) before both the vision tower and the VAE.
    pub const CONDITION_AREA: u64 = 1024 * 1024;
    /// At most this many condition images (the reference's limit).
    pub const MAX_CONDITION_IMAGES: usize = 10;
}
