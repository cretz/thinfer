//! HunyuanVideo 1.5 config constants. First target: 480p T2V, lightx2v 4-step
//! distill (`lightx2v/Hy1.5-Distill-Models/hy1.5_t2v_480p_lightx2v_4step
//! .safetensors`, fp16, loaded direct as the WHOLE DiT). See
//! `projects/thinfer-working-area/hunyuan-plan.md`.
//!
//! Architecture = dual-stream MMDiT (Flux/HV1.0 lineage). All dims verified
//! against the base `480p_t2v/config.json` + the lightx2v checkpoint tensor
//! shapes (1793 bf16 tensors). The ONLY architectural delta vs Tencent's I2V
//! step-distill is `use_meanflow` (true there, FALSE here): the lightx2v T2V
//! distill is a plain single-timestep flow-match Euler distill.

/// Dual-stream MMDiT (`HunyuanVideo_1_5_DiffusionTransformer`).
pub mod dit {
    /// Hidden width.
    pub const HIDDEN: usize = 2048;
    /// Attention heads (head_dim = HIDDEN / HEADS = 128).
    pub const HEADS: usize = 16;
    pub const HEAD_DIM: usize = 128;
    /// All blocks are dual-stream (`mm_double_blocks_depth`); zero single-stream.
    pub const DOUBLE_BLOCKS: usize = 54;
    pub const SINGLE_BLOCKS: usize = 0;
    /// Latent (velocity field) channels in/out = 32. The DiT predicts a 32-ch
    /// velocity; the noisy latent is 32-ch.
    pub const LATENT_CHANNELS: usize = 32;
    /// `concat_condition=true` even for T2V: `img_in` is a 1x1x1 Conv3d over
    /// `32*2+1 = 65` channels. The conv input is `[noisy(32) | cond(32) | mask(1)]`;
    /// for T2V the cond block + mask are ALL ZERO (i2v fills cond on frame 0).
    /// Verified by `img_in.proj.weight = [2048, 65, 1, 1, 1]`.
    pub const CONCAT_CONDITION: bool = true;
    pub const CONV_IN_CHANNELS: usize = LATENT_CHANNELS * 2 + 1; // 65
    /// Patch size (t,h,w) = [1,1,1]: no patchify; the VAE does all compression.
    pub const PATCH: [usize; 3] = [1, 1, 1];
    /// 3D axial RoPE split over (t,h,w); sum = HEAD_DIM. Interleaved-pair
    /// (Wan/SkyReels convention, NOT Qwen3 half-rot). theta 256. Image tokens
    /// only (text tokens are not rotated).
    pub const ROPE_DIM: [usize; 3] = [16, 56, 56];
    pub const ROPE_THETA: f64 = 256.0;
    /// Per-head qk RMSNorm, eps 1e-6. qkv has bias.
    pub const QK_NORM_EPS: f32 = 1e-6;
    pub const QKV_BIAS: bool = true;
    /// FFN: gelu_tanh, width ratio 4.0 -> mlp hidden 8192.
    pub const MLP_RATIO: f32 = 4.0;
    pub const MLP_HIDDEN: usize = HIDDEN * 4;
    /// Text states width (Qwen2.5-VL-7B last hidden). Refined by a 2-layer
    /// SingleTokenRefiner (conditioned on the timestep) into joint tokens.
    pub const TEXT_STATES_DIM: usize = 3584;
    pub const TEXT_REFINER_DEPTH: usize = 2;
    /// `cond_type_embedding = nn.Embedding(3, 2048)`, ADDED by token type
    /// (0=text, 1=byt5, 2=vision). Shipped in the checkpoint; add type-0 to the
    /// Qwen text tokens (do not skip even though zero-init at construction).
    pub const COND_TYPE_COUNT: usize = 3;
    /// adaLN-Zero modulation: 6 chunks per stream in a double block
    /// (shift/scale/gate x2), produced by SiLU + Linear.
    pub const MOD_CHUNKS_DOUBLE: usize = 6;
    /// Single timestep embed (`time_in`): sinusoidal freq_dim 256, max_period
    /// 1e4, SiLU MLP. NO `time_r_in` (meanflow off), NO guidance embed, NO
    /// pooled-text vector (`text_states_dim_2=null`).
    pub const USE_MEANFLOW: bool = false;
    pub const TIME_FREQ_DIM: usize = 256;
    pub const TIME_MAX_PERIOD: f32 = 10_000.0;
    /// ByT5-glyph + SigLIP-vision streams are masked-zero for plain T2V and
    /// bit-faithfully omittable (masked-zero tokens don't affect attention).
    /// Wire ByT5 only for rendered in-frame text; vision only for I2V (phase 2).
    pub const T2V_SKIP_BYT5: bool = true;
    pub const T2V_SKIP_VISION: bool = true;
}

/// Causal AR I2V (minWM WorldPlay `HY15/TI2V/dmd`): the same MMDiT run
/// chunk-autoregressively. Text (+SigLIP vision) K/V are cached once at t=0;
/// each chunk of `CHUNK_LATENT_FRAMES` latent frames runs `STEPS` flow-match
/// Euler steps + one t=0 recache forward that commits the chunk's K/V.
pub mod ar {
    /// Latent frames denoised per AR chunk (`chunk_latent_frames`).
    pub const CHUNK_LATENT_FRAMES: usize = 4;
    /// 4 uniform steps: `set_timesteps(4)` = labels [1000,750,500,250].
    pub const DENOISING_STEP_LIST: [u32; 4] = [1000, 750, 500, 250];
    /// minWM inference shift (run_infer_causal.sh `--shift 5.0`).
    pub const FLOW_SHIFT: f32 = 5.0;
    /// Recache timestep = `stabilization_level - 1` with the default level 1.
    pub const RECACHE_TIMESTEP: f32 = 0.0;
    /// SigLIP so400m-patch14-384 token grid/width for the vision stream.
    pub const VISION_TOKENS: usize = 729;
    pub const VISION_DIM: usize = 1152;
    /// Default clip: 77 frames @ 16 fps -> 20 latent frames = 5 chunks.
    pub const DEFAULT_FRAMES: usize = 77;
    pub const DEFAULT_FPS: usize = 16;
}

/// VAE = `AutoencoderKLConv3D`, causal-conv3d (Wan VAE family).
pub mod vae {
    pub const LATENT_CHANNELS: usize = 32;
    /// Spatial downscale 16x (vs Wan 8x), temporal 4x.
    pub const FFACTOR_SPATIAL: usize = 16;
    pub const FFACTOR_TEMPORAL: usize = 4;
    pub const BLOCK_OUT_CHANNELS: [usize; 5] = [128, 256, 512, 1024, 1024];
    pub const LAYERS_PER_BLOCK: usize = 2;
    /// decode: z / SCALING (no shift), then *0.5+0.5.
    pub const SCALING_FACTOR: f32 = 1.03682;
}

/// Sampling: lightx2v 4-step flow-match Euler, CFG OFF (single forward/step, no
/// negative prompt). MeanFlow off -> single timestep per step.
pub mod sampling {
    pub const STEPS: usize = 4;
    /// `denoising_step_list` = INDICES into the 1000-point schedule (not literal
    /// timesteps). sigmas_full[k] = 1 - k/1000; idx = 1000 - step.
    pub const DENOISING_STEP_LIST: [u32; 4] = [1000, 750, 500, 250];
    /// SD3/flux shift: sigma' = S*sigma / (1 + (S-1)*sigma), S = FLOW_SHIFT.
    /// Model card (authoritative for these weights) = 9.0; LightX2V repo default
    /// = 5.0. Default 9.0, keep 5.0 as an A/B fallback.
    pub const FLOW_SHIFT: f32 = 9.0;
    pub const FLOW_SHIFT_FALLBACK: f32 = 5.0;
    /// Per-step Euler: x += (sigma_{i+1} - sigma_i) * v; terminal sigma 0.
    /// Default clip: 81 frames @ 16 fps, 16:9, 480p.
    pub const DEFAULT_FRAMES: usize = 81;
    pub const DEFAULT_FPS: usize = 16;
}
