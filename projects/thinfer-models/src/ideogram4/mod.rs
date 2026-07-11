//! Ideogram-4 text-to-image. Single-stream flow-matching DiT conditioned on a
//! Qwen3-VL-8B language-model multi-layer tap, with the ostris turbotime LoRA
//! folded in to drop CFG (single conditional transformer, no unconditional
//! branch, no guidance).
//!
//! Ground truth is the official inference repo `third-party/ideogram4`
//! (`src/ideogram4/{modeling_ideogram4,pipeline_ideogram4,autoencoder,
//! scheduler,latent_norm,constants}.py`). Per-model porting notes live in
//! `projects/thinfer-working-area/ideogram-plan.md`.
//!
//! Reuse: the per-block DiT recipe is the model-agnostic `common::block::Block`
//! (sandwich-norm + tanh-gated AdaLN, QK-RMSNorm, SwiGLU) with half-rot RoPE;
//! the encoder reuses `z_image::text_encoder`'s per-layer `Qwen3Block`; the VAE
//! reuses the KL conv/resnet/mid-attn/upsample ops. New here: 3-axis interleaved
//! MRoPE, single-stream sequence packing, LayerNorm-no-affine final layer, the
//! logit-normal resolution-aware Euler sampler, and the LoRA fold.

pub mod dit;
pub mod loader;
pub mod lora;
pub mod manifest;
pub mod mrope;
pub mod packing;
pub mod pipeline;
pub mod sampler;
pub mod t_embedder;
pub mod text_encoder;
pub mod vae;

/// DiT backbone config, audited against `Ideogram4Config` defaults in
/// `modeling_ideogram4.py` and `docs/model_architecture.md`.
pub mod config {
    /// Transformer hidden size (`emb_dim`).
    pub const DIM: usize = 4608;
    pub const N_LAYERS: usize = 34;
    pub const N_HEADS: usize = 18;
    /// `emb_dim / num_heads`. NOTE 256 > 128, so SDPA uses the dense path, not
    /// the subgroup-flash kernel (which caps at head_dim 128).
    pub const HEAD_DIM: usize = DIM / N_HEADS; // 256
    /// SwiGLU intermediate (`intermediate_size`).
    pub const FFN_HIDDEN: usize = 12288;
    /// adaLN conditioning width (`adanln_dim`): the per-block modulation Linear
    /// maps `ADALN_DIM -> 4*DIM`.
    pub const ADALN_DIM: usize = 512;
    /// Patchified latent channels = ae z_channels(32) * patch(2)^2 = 128.
    pub const IN_CHANNELS: usize = 128;
    pub const NORM_EPS: f32 = 1e-5;
    /// MRoPE base. Same value as the Qwen3-VL encoder (5e6).
    pub const ROPE_THETA: f32 = 5_000_000.0;
    /// Per-axis (t, h, w) section widths over head_dim/2 = 128.
    pub const MROPE_SECTION: [usize; 3] = [24, 20, 20];

    /// Image grid coordinates start here so they never collide with text token
    /// positions (which start at 0 and stay below `max_text_tokens`).
    pub const IMAGE_POSITION_OFFSET: i64 = 65536;
    pub const OUTPUT_IMAGE_INDICATOR: i64 = 2;
    pub const LLM_TOKEN_INDICATOR: i64 = 3;

    /// patch_size (DiT token = 2x2 latent patch) and ae spatial downscale.
    pub const PATCH_SIZE: usize = 2;
    pub const AE_SCALE_FACTOR: usize = 8;
    pub const MAX_TEXT_TOKENS: usize = 2048;
}
