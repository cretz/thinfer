//! Qwen-Image-Edit-Rapid-AIO: a 4-step distilled, CFG-free edit MMDiT. A
//! reference image conditions generation through TWO channels: the Qwen2.5-VL
//! vision tower (interleaved `<|image_pad|>` tokens in the text stream) and the
//! VAE (reference latents concatenated to the image stream). The backbone is a
//! 60-layer dual-stream (text+image joint-attention) MMDiT with 3-axis complex
//! RoPE.
//!
//! Ground truth = `third-party/diffusers` (`transformer_qwenimage.py`,
//! `pipeline_qwenimage_edit.py`, `autoencoder_kl_qwenimage.py`) +
//! `third-party/transformers/models/qwen2_5_vl`. Port plan +
//! audit-vs-CONFIRM list: `projects/thinfer-working-area/qwen-image-plan.md`.
//!
//! Template = `ideogram4` (GGUF-only image DiT, Qwen-VL encoder tap, KL VAE,
//! folded turbo recipe). Divergences from ideogram4: dual-stream (not single),
//! complex interleaved-pair RoPE (reuse `wan/rope3d` machinery, NOT the
//! half-rot mrope), Wan-family 3D-causal KL VAE (reuse `wan/vae`), a Qwen2.5
//! encoder block (qkv bias + GQA, no qk-norm), and a vision tower (new).

pub mod dit;
pub mod loader;
pub mod lora;
pub mod manifest;
pub mod packing;
pub mod pipeline;
pub mod rope;
pub mod text_encoder;
pub mod vae;
pub mod vision;

/// DiT backbone config. Audited against `QwenImageTransformer2DModel` defaults
/// in `transformer_qwenimage.py`. Flags/layer-count marked CONFIRM are verified
/// against the GGUF metadata header before the loader is written (the Rapid-AIO
/// finetune may override a flag); see the CONFIRM list in `qwen-image-plan.md`.
pub mod config {
    /// `num_attention_heads * attention_head_dim` (24 * 128).
    pub const DIM: usize = 3072;
    /// CONFIRM vs GGUF `block_count` (diffusers default 60).
    pub const N_LAYERS: usize = 60;
    pub const N_HEADS: usize = 24;
    pub const HEAD_DIM: usize = 128;
    /// SwiGLU-free GELU MLP: FeedForward(dim -> 4*dim -> dim), gelu-approximate.
    pub const FFN_HIDDEN: usize = 4 * DIM;
    /// Patchified latent channels: ae z_channels(16) * patch(2)^2.
    pub const IN_CHANNELS: usize = 64;
    /// Latent channels the head emits (per 2x2 patch: 2*2*16 = 64).
    pub const OUT_CHANNELS: usize = 16;
    pub const PATCH_SIZE: usize = 2;
    /// Text-stream conditioning width = Qwen2.5-VL hidden size.
    pub const JOINT_ATTENTION_DIM: usize = 3584;
    pub const NORM_EPS: f32 = 1e-6;
    /// RoPE base. Per-axis (frame, h, w) section widths over head_dim (sum 128).
    pub const ROPE_THETA: f32 = 10_000.0;
    pub const AXES_DIMS_ROPE: [usize; 3] = [16, 56, 56];
    /// vae spatial downscale * patch = 8 * 2: pixels per DiT token side.
    pub const PIXELS_PER_TOKEN: usize = 16;
}
