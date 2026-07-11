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

/// Which LTX-2 line a checkpoint is: the 22B (LTX-2.3, our shipped default) vs the
/// 19B (`ltx2-rapid` merge). The DiT backbone consts (DIM, heads, layers, rope,
/// channels) are IDENTICAL; only the text-conditioning tail + a few block flags
/// differ. Threaded through register + forward so a single code path serves both.
///
/// 19B deltas (all confirmed vs the v62 GGUF + connector safetensors + LTX-2 src):
/// FeatureExtractor V1 (range-norm, single bias-free aggregate embed) + an
/// in-transformer caption projection (3840 -> 4096/2048); a 2-layer / 30-head /
/// 3840 UNGATED connector loaded from the connector safetensors (22B: 8-layer /
/// 4096 gated, from the DiT GGUF); 6-way block modulation (22B: 9-way); no gated
/// attention; no prompt-AdaLN / cross-attn AdaLN.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LtxVariant {
    /// Per-head sigmoid attention gate (`to_gate_logits`). 22B true, 19B false.
    pub gated_attn: bool,
    /// Text cross-attn q/kv AdaLN modulation. 22B true, 19B false (raw cross).
    pub cross_adaln: bool,
    /// Prompt-AdaLN (continuous-sigma KV modulation module). 22B true, 19B false.
    pub prompt_adaln: bool,
    /// Per-block `scale_shift_table` rows (`adaln_single.linear` = this * DIM).
    /// 22B 9, 19B 6.
    pub n_block_mod: usize,
    /// In-transformer `caption_projection` (PixArtAlpha 3840 -> inner). 19B only;
    /// 22B does the projection inside the FE V2 aggregate embeds instead.
    pub caption_proj: bool,
    /// FeatureExtractor V1 (19B) vs V2 (22B).
    pub fe_v1: bool,
    /// Embeddings-connector geometry + weight provenance.
    pub connector: ConnectorSpec,
}

/// Embeddings-connector geometry. Both modalities share `layers`/`head_dim`/
/// `gated`/`from_safetensors`; the per-modality inner width + head count differ
/// (22B: video 4096/32, audio 2048/32; 19B: both 3840/30).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConnectorSpec {
    pub layers: usize,
    pub video_inner: usize,
    pub video_heads: usize,
    pub audio_inner: usize,
    pub audio_heads: usize,
    /// Per-head sigmoid gate in the connector block. 22B true, 19B false.
    pub gated: bool,
    /// Connector blocks + registers live in the connector SAFETENSORS (19B) vs
    /// the DiT GGUF (22B). Toggles the register prefix + weight dtype (bf16 vs Q8).
    pub from_safetensors: bool,
}

impl LtxVariant {
    /// LTX-2.3 (22B): our shipped default. Reproduces the pre-variant behavior.
    pub const fn ltx_2_3_22b() -> Self {
        Self {
            gated_attn: true,
            cross_adaln: true,
            prompt_adaln: true,
            n_block_mod: 9,
            caption_proj: false,
            fe_v1: false,
            connector: ConnectorSpec {
                layers: config::CONNECTOR_NUM_LAYERS,
                video_inner: config::DIM,
                video_heads: config::CONNECTOR_N_HEADS,
                audio_inner: config::AUDIO_DIM,
                audio_heads: config::AUDIO_CONNECTOR_N_HEADS,
                gated: true,
                from_safetensors: false,
            },
        }
    }

    /// ltx2-rapid (LTX-2 19B merge): FE V1 + caption projection + 2-layer ungated
    /// 3840 connector + 6-way ungated blocks. Connector head_dim is 128 (3840/30).
    pub const fn ltx2_rapid_19b() -> Self {
        Self {
            gated_attn: false,
            cross_adaln: false,
            prompt_adaln: false,
            n_block_mod: 6,
            caption_proj: true,
            fe_v1: true,
            connector: ConnectorSpec {
                layers: 2,
                video_inner: 3840,
                video_heads: 30,
                audio_inner: 3840,
                audio_heads: 30,
                gated: false,
                from_safetensors: true,
            },
        }
    }
}

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

#[cfg(test)]
mod variant_tests {
    use super::*;

    #[test]
    fn variant_shapes_consistent() {
        let v22 = LtxVariant::ltx_2_3_22b();
        assert!(v22.gated_attn && v22.cross_adaln && v22.prompt_adaln);
        assert_eq!(v22.n_block_mod, 9);
        assert!(!v22.fe_v1 && !v22.caption_proj);
        // 22B connector head_dim: video 4096/32 = 128, audio 2048/32 = 64.
        assert_eq!(v22.connector.video_inner / v22.connector.video_heads, 128);
        assert_eq!(v22.connector.audio_inner / v22.connector.audio_heads, 64);

        let v19 = LtxVariant::ltx2_rapid_19b();
        assert!(!v19.gated_attn && !v19.cross_adaln && !v19.prompt_adaln);
        assert_eq!(v19.n_block_mod, 6);
        assert!(v19.fe_v1 && v19.caption_proj && v19.connector.from_safetensors);
        // 19B connector head_dim: both streams 3840/30 = 128.
        assert_eq!(v19.connector.video_inner / v19.connector.video_heads, 128);
        assert_eq!(v19.connector.audio_inner / v19.connector.audio_heads, 128);
        assert_eq!(v19.connector.layers, 2);
    }
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
