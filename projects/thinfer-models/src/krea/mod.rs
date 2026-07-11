//! Krea 2 Turbo: an 8-step distilled, CFG-free single-stream MMDiT text-to-image
//! model. Text and image tokens are concatenated into one `[txt ++ img]`
//! sequence through all 28 blocks. Conditioning enters through a Qwen3-VL-4B
//! encoder whose 12 tapped hidden layers are fused by a small text-fusion
//! transformer (layerwise-attention -> `Linear(12->1)` projector -> refiner)
//! before being projected into the DiT feature width.
//!
//! Ground truth = `third-party/stable-diffusion.cpp/src/model/diffusion/
//! krea2.hpp` (the reference impl; block structure + norms) plus the diffusers
//! `Krea2Transformer2DModel` config. Per-model port notes:
//! `projects/thinfer-working-area/worklog.md` (Krea section).
//!
//! Template = `qwen_image` (GGUF image DiT, Qwen-VL encoder tap, Wan-family KL
//! VAE reuse, FlowMatchEuler CFG-free t2i). Divergences from qwen_image:
//! single-stream (not dual), a text-fusion transformer over 12 encoder taps
//! (not a single last-hidden tap), shared adaLN modulation from the timestep
//! plus a per-block learned offset, per-head `(1+w)` QK-RMSNorm with a sigmoid
//! output gate, and Flux-style joint 3-axis RoPE ids.

pub mod dit;
pub mod loader;
pub mod lora;
pub mod manifest;
pub mod packing;
pub mod pipeline;
pub mod rope;
pub mod scheduler;
pub mod text_encoder;
pub mod vae;

pub mod config {
    /// `heads * head_dim` (48 * 128). Krea `features`.
    pub const DIM: usize = 6144;
    pub const N_LAYERS: usize = 28;
    pub const N_HEADS: usize = 48;
    /// GQA key/value heads (48 q share 12 kv).
    pub const N_KV_HEADS: usize = 12;
    pub const HEAD_DIM: usize = 128;
    /// SwiGLU inner width: `ceil_to_multiple((2*DIM/3) * mlp_mult, 128)`.
    /// `(2*6144/3)*4 = 16384`, already a multiple of 128.
    pub const MLP_INNER: usize = 16384;
    pub const MLP_MULT: usize = 4;
    /// VAE latent channels; packed DiT in/out = `LATENT_CH * PATCH^2` = 64.
    pub const LATENT_CH: usize = 16;
    pub const PATCH_SIZE: usize = 2;
    pub const PACKED_CH: usize = LATENT_CH * PATCH_SIZE * PATCH_SIZE;
    /// Sinusoidal timestep embedding width.
    pub const TIMESTEP_DIM: usize = 256;
    /// `ggml_ext_timestep_embedding` params: period 10000, input scale 1000.
    pub const TIMESTEP_MAX_PERIOD: f32 = 10_000.0;
    pub const TIMESTEP_SCALE: f32 = 1000.0;
    pub const NORM_EPS: f32 = 1e-5;
    /// RoPE base and per-axis (t, h, w) section widths over head_dim (sum 128).
    pub const ROPE_THETA: f32 = 1000.0;
    pub const AXES_DIMS_ROPE: [usize; 3] = [32, 48, 48];
    /// vae spatial downscale (8) * patch (2): pixels per DiT token side.
    pub const PIXELS_PER_TOKEN: usize = 16;

    // --- text-fusion (txtfusion) over the encoder taps ---
    /// Qwen3-VL hidden width per tapped layer.
    pub const TEXT_DIM: usize = 2560;
    /// Number of encoder hidden layers stacked per token.
    pub const TEXT_LAYERS: usize = 12;
    /// Full MHA on the text side (no GQA).
    pub const TEXT_HEADS: usize = 20;
    pub const TEXT_KV_HEADS: usize = 20;
    pub const TEXT_HEAD_DIM: usize = TEXT_DIM / TEXT_HEADS; // 128
    /// Text SwiGLU inner width: `ceil_to_multiple((2*TEXT_DIM/3)*MLP_MULT, 128)`
    /// = `ceil(6826.67, 128) = 6912`.
    pub const TEXT_MLP_INNER: usize = 6912;
    /// Layerwise-attention blocks (attend across the 12-layer axis per token),
    /// then a `Linear(TEXT_LAYERS -> 1)` projector, then refiner blocks
    /// (attend across tokens).
    pub const N_LAYERWISE_BLOCKS: usize = 2;
    pub const N_REFINER_BLOCKS: usize = 2;
    /// Max prompt tokens fed to the encoder.
    pub const MAX_SEQ: usize = 512;

    /// Qwen3-VL-4B decoder-layer indices tapped for the text-fusion stack
    /// (`text_encoder_select_layers`; index 0 = embedding output, so these are
    /// every 3rd layer output). Order matters: the projector weights are keyed
    /// to this order.
    pub const ENCODER_SELECT_LAYERS: [usize; TEXT_LAYERS] =
        [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35];
}
