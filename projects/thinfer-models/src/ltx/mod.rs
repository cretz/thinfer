//! LTX-2.3 distilled-1.1: a 22B joint audio-video diffusion model (Lightricks
//! LTX-2, `AVTransformer3DModel`). The largest port in the tree: Gemma-3-12B text
//! encoder + 8-layer gated connector + dual-stream (video+audio) DiT with 5 attn
//! sublayers per block + causal video VAE + audio mel VAE + BigVGAN/BWE vocoder +
//! latent spatial upscaler, run as a full two-stage distilled pipeline.
//!
//! Ground truth = `third-party/LTX-2` (`ltx-core`, `ltx-pipelines`) +
//! `third-party/diffusers` (`transformer_ltx2`, `autoencoder_kl_ltx2{,_audio}`).
//! All architecture consts below are the GGUF `config` KV (authoritative; the DiT
//! GGUF embeds the full transformer/vae/scheduler/audio_vae/vocoder config JSON).
//! Port plan + non-negotiables: `projects/thinfer-working-area/ltx-plan.md`.
//!
//! RoPE is `split` = HALF-ROT (Qwen3-style, reuse `op_rope`), NOT Wan
//! interleaved-pair. The residual stream MUST stay bf16 (large-outlier channels
//! overflow f16; same dead-end as the other big DiTs).

pub mod audio_vae;
pub mod cond;
pub mod connector;
pub mod dit;
pub mod loader;
pub mod lora;
pub mod manifest;
pub mod patchify;
pub mod pipeline;
pub mod sampler;
pub mod text_encoder;
pub mod upsampler;
pub mod video_vae;
pub mod vocoder;

/// DiT backbone (`AVTransformer3DModel`) config, from the GGUF `config.transformer`
/// KV. Video and audio are separate streams with their own widths; they sync
/// through audio<->video cross-attention inside each block.
pub mod config {
    pub const NUM_LAYERS: usize = 48;
    pub const NORM_EPS: f32 = 1e-6;
    /// gelu-approximate per-modality FFN. `attention_bias=true` (qkv carry bias).
    /// `apply_gated_attention=true`: per-head sigmoid gate (`to_gate_logits`).
    pub const ATTENTION_BIAS: bool = true;

    // --- video stream ---
    pub const HEAD_DIM: usize = 128;
    pub const N_HEADS: usize = 32;
    /// `N_HEADS * HEAD_DIM`.
    pub const DIM: usize = 4096;
    /// Patchified latent channels in/out (`patch_size=1`: 1 latent cell = 1 token).
    pub const IN_CHANNELS: usize = 128;
    pub const OUT_CHANNELS: usize = 128;
    /// Video<->text cross-attention (`attn2`) key/value width = Gemma connector out.
    pub const CROSS_ATTENTION_DIM: usize = 4096;

    // --- audio stream ---
    pub const AUDIO_HEAD_DIM: usize = 64;
    pub const AUDIO_N_HEADS: usize = 32;
    /// `AUDIO_N_HEADS * AUDIO_HEAD_DIM`.
    pub const AUDIO_DIM: usize = 2048;
    pub const AUDIO_OUT_CHANNELS: usize = 128;
    /// Audio<->text cross-attention (`audio_attn2`) key/value width.
    pub const AUDIO_CROSS_ATTENTION_DIM: usize = 2048;

    // --- conditioning / adaln ---
    /// Gemma-3 hidden width (caption channels into the connector/cross-attn).
    pub const CAPTION_CHANNELS: usize = 3840;
    /// AdaLN sinusoidal timestep range + scale (`adaln_single`).
    pub const NUM_EMBEDS_ADA_NORM: usize = 1000;
    pub const TIMESTEP_SCALE_MULTIPLIER: f32 = 1000.0;
    pub const AV_CA_TIMESTEP_SCALE_MULTIPLIER: f32 = 1000.0;
    /// 9 modulation params per block (`scale_shift_table [DIM, 9]`); separate
    /// continuous-sigma prompt AdaLN (`prompt_scale_shift_table [DIM, 2]`); and
    /// av-cross ada-norm (`scale_shift_table_a2v_ca_* [*, 5]`).
    pub const N_ADALN_MOD: usize = 9;

    // --- RoPE (split = half-rot, physical-coord middle-index positions) ---
    pub const ROPE_THETA: f64 = 10_000.0;
    /// Per-axis max position: [temporal(frames), height, width] (video).
    pub const ROPE_MAX_POS: [usize; 3] = [20, 2048, 2048];
    pub const AUDIO_ROPE_MAX_POS: [usize; 1] = [20];
    /// Positions are [start,end) patch bounds -> use the middle of each patch.
    pub const USE_MIDDLE_INDICES_GRID: bool = true;
    /// Frame 0 gets a causal temporal offset of 1.
    pub const CAUSAL_TEMPORAL_POSITIONING: bool = true;

    // --- embeddings connector (8-layer gated transformer; config here, weights
    // in the connector safetensors) ---
    pub const CONNECTOR_NUM_LAYERS: usize = 8;
    pub const CONNECTOR_N_HEADS: usize = 32;
    pub const CONNECTOR_HEAD_DIM: usize = 128;
    pub const CONNECTOR_MAX_POS: usize = 4096;
    pub const CONNECTOR_NUM_LEARNABLE_REGISTERS: usize = 128;
    pub const AUDIO_CONNECTOR_N_HEADS: usize = 32;
    pub const AUDIO_CONNECTOR_HEAD_DIM: usize = 64;
}

/// Gemma-3-12B-it text encoder config (from the QAT GGUF `gemma3.*` KV). Standard
/// Gemma-3: pre+post RMSNorms around attn and ffn, head-dim QK-norm, dual rope
/// (global linear-x8 + local), sliding/full attention interleave (pattern 6). The
/// residual grows large (>1e5) -> bf16 acts mandatory (f16 overflows -> NaN).
pub mod gemma {
    pub const VOCAB: usize = 262_208;
    pub const HIDDEN: usize = 3840;
    pub const FFN: usize = 15360;
    pub const N_LAYERS: usize = 48;
    pub const N_HEADS: usize = 16;
    pub const N_KV_HEADS: usize = 8;
    pub const HEAD_DIM: usize = 256;
    pub const EPS: f32 = 1e-6;
    pub const SLIDING_WINDOW: usize = 1024;
    /// Every `SLIDING_PATTERN`-th layer is full attention; the rest are sliding.
    pub const SLIDING_PATTERN: usize = 6;
    pub const QUERY_PRE_ATTN_SCALAR: f32 = 256.0;
    pub const GLOBAL_THETA: f64 = 1_000_000.0;
    pub const LOCAL_THETA: f64 = 10_000.0;
    /// Linear rope scaling on the global (full-attention) layers.
    pub const ROPE_LINEAR_FACTOR: f64 = 8.0;
}
