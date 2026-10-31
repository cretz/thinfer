//! MiniMax-H3 file manifest. A 33B dense single-stream joint audio-video
//! diffusion transformer (MiniMaxAI). One packed sequence carries text, video
//! and audio rows; full self-attention over all of it, no cross-attention.
//!
//! The DiT is GGUF (checkpoint-native tensor keys). The text encoder is
//! Qwen3-VL-32B, of which only decoder layers 0..49 of 64 are ever evaluated
//! (H3 conditions on the UNNORMALIZED `hidden_states[50]`), so `lm_head` and the
//! last 14 layers are dead weight in any full-stack file. Both VAEs are
//! safetensors from the Comfy-Org conversion.
//!
//! The base checkpoint is not step-distilled. The 4-step turbo LoRA is folded at
//! load and is a REQUIRED part of the fast variant, not a user adapter.

use thinfer_core::manifest::{FileRef, ModelManifest};

/// DiT + text-encoder GGUF conversions of the upstream FL2VA checkpoint.
/// These are the FULL (unpruned) DiT: they carry the `adaln_proj` projections,
/// which is what lets us precompute the modulation table exactly (see
/// `minimax::adaln`) rather than relying on the pruned "curve-form" checkpoint.
const REPO_GGUF: &str = "Abiray/MiniMax-H3-GGUF";
/// Qwen3-VL-32B conditioner pre-truncated to the 50 layers H3 reads, in smaller
/// quant tiers than the full-stack file. Same weights for layers 0..49.
const REPO_ENCODER_TRUNC: &str = "nif0/Qwen3-VL-32B-Instruct-MiniMax-H3-GGUF";
/// Both VAEs (Comfy-Org conversion of the upstream autoencoders).
const REPO_VAE: &str = "Comfy-Org/MiniMax-H3";
/// Upstream release: the PRODUCT tokenizer ships here. Use the checkpoint's own
/// `text_encoder/tokenizer.json`, not a generic Qwen3-VL mirror (LTX lesson: a
/// reconstructed or mismatched tokenizer is exactly where a silent adherence bug
/// hides, and the encode path is never covered by a component parity gate).
const REPO_UPSTREAM: &str = "MiniMaxAI/MiniMax-H3";
/// lightx2v (ModelTC) 4-step DMD distillation. Rank 128, diffusers/PEFT tensor
/// naming, attention + feed-forward sites only (no AdaLN pairs), so it folds
/// against the full checkpoint unchanged. v0.1 is a PREVIEW: lightx2v flags
/// soft image detail, and an improved release is in development.
const REPO_TURBO_LORA: &str = "lightx2v/Minimax-h3-Turbo";
/// FastVideo FastH3-Trim: 8-step DMD distillation of H3, 42 of 50 blocks kept,
/// rank-16 AdaLN, block linears in calibrated NVFP4 (served as composites by
/// `minimax::fasth3`). Two files: NVFP4 linears + the bf16/f32 rest.
const REPO_FASTH3_TRIM: &str = "FastVideo/FastVideo-FastH3-Trim-8-Step-NVFP4";
/// FastVideo FastH3-Omni: the 8-step distill of the Ref2VA (`transformer_ref`)
/// checkpoint, all 50 blocks, NVFP4 linears in the Comfy layout, AdaLN as a
/// `[1025, 8]` timestep table + rank-8 projections, PDD output heads.
const REPO_FASTH3_OMNI: &str = "FastVideo/FastH3-Omni-NVFP4-QAD-Comfy";
/// Qwen3-VL-32B vision tower (mmproj). H3's text encoder is byte-identical to
/// stock Qwen3-VL-32B-Instruct (shard hashes match), so the stock mmproj is
/// its vision half; BF16 is the checkpoint dtype, lossless.
const REPO_VISION: &str = "unsloth/Qwen3-VL-32B-Instruct-GGUF";
/// FastVideo's NVFP4 video decoders (QAD, opt-in): `quality-36L` and the
/// depth-cut `fast-8L`, packed `decoder-nvfp4.safetensors` each.
const REPO_VAE_NVFP4: &str = "FastVideo/FastH3-NVFP4-VAE-Decoder";
/// Fallback 4-step distillation if the lightx2v preview does not hold up.
/// Native `blocks.*` naming, 259 pairs (51 of them AdaLN, which we ignore since
/// the modulation table is precomputed). ckpt850-EMA is the author's pick.
const REPO_TURBO_LORA_ALT: &str = "larryvrh/MiniMax-H3-Turbo-Lora";

pub mod role {
    /// DiT GGUF, Q4_K_M. Default. 18.5G on disk, of which ~7G is `adaln_proj`
    /// that is read once at load and then never streamed again.
    pub const DIT_GGUF_Q4_K_M: &str = "dit/gguf-q4_k_m";
    /// DiT GGUF, Q8_0. Parity canary + quality baseline. 33.6G.
    pub const DIT_GGUF_Q8_0: &str = "dit/gguf-q8_0";
    /// Qwen3-VL-32B conditioner, Q4_K_M. 13.6G. Default: same repo as the DiT.
    /// ASSERTED AT LOAD: `block_count` must be 50, not the full stack's 64 (see
    /// `loader::check_encoder_depth`). 13.6G is consistent with a truncated
    /// build; a full stack at this quant would be ~18.5G.
    pub const ENCODER_GGUF: &str = "encoder/qwen3vl-gguf";
    /// Pre-truncated conditioner at IQ4_XS (12.5G). Footprint option.
    pub const ENCODER_GGUF_IQ4: &str = "encoder/qwen3vl-gguf-iq4";
    /// Pre-truncated conditioner at UD-IQ3_XXS (9.1G). Footprint floor;
    /// conditioning quality cost is unmeasured, so this stays opt-in.
    pub const ENCODER_GGUF_IQ3: &str = "encoder/qwen3vl-gguf-iq3";
    /// Product tokenizer (HF fast tokenizer json) from the upstream checkpoint.
    pub const TOKENIZER: &str = "encoder/tokenizer";
    /// Causal video autoencoder: 16x spatial, 4x temporal. Decoder emits
    /// ImageNet-normalized RGB over a [0,1] base range.
    pub const VIDEO_VAE: &str = "vae/video";
    /// Audio autoencoder: mono, 40 latents/s, 32 kHz. Run once per stereo
    /// channel.
    pub const AUDIO_VAE: &str = "vae/audio";
    /// lightx2v 4-step turbo LoRA (default distillation).
    pub const TURBO_LORA: &str = "dit/turbo-lora";
    /// larryvrh ckpt850-EMA 4-step turbo LoRA (fallback distillation).
    pub const TURBO_LORA_ALT: &str = "dit/turbo-lora-alt";
    /// FastH3-Trim block linears, NVFP4 (10.0G).
    pub const FASTH3_TRIM_NVFP4: &str = "dit/fasth3-trim-nvfp4";
    /// FastH3-Trim norms, embedders, refiner, rank-16 AdaLN (1.9G).
    pub const FASTH3_TRIM_REST: &str = "dit/fasth3-trim-rest";
    /// FastH3-Omni reference DiT, one Comfy NVFP4 file (12.8G).
    pub const FASTH3_OMNI: &str = "dit/fasth3-omni";
    /// Qwen3-VL-32B vision tower, BF16 mmproj (1.2G).
    pub const VISION: &str = "encoder/qwen3vl-vision";
    /// NVFP4 video decoder, 36 layers (1.4G). Opt-in, Blackwell.
    pub const VAE_DECODER_NVFP4: &str = "vae/decoder-nvfp4";
    /// NVFP4 video decoder, 8 layers (0.3G). Opt-in, Blackwell.
    pub const VAE_DECODER_NVFP4_FAST: &str = "vae/decoder-nvfp4-fast";
}

/// Roles a FastH3-Omni reference-image generate (Ref2VA) needs.
pub const RUNTIME_ROLES_REF2VA_FASTH3_OMNI: &[&str] = &[
    role::ENCODER_GGUF,
    role::VISION,
    role::TOKENIZER,
    role::FASTH3_OMNI,
    role::VIDEO_VAE,
    role::AUDIO_VAE,
];

/// FastH3's trained DMD rungs (`fastvideo_inference.json`
/// `dmd_denoising_steps`): 8 forwards, Euler on `shift(rung / 1000)`.
pub const FASTH3_RUNGS: [u32; 8] = [999, 874, 749, 624, 500, 375, 250, 125];
/// FastH3's per-modality sigma shifts (`video_/audio_scheduler_shift`).
pub const FASTH3_SHIFT_VIDEO: f32 = 10.0;
pub const FASTH3_SHIFT_AUDIO: f32 = 3.0;

/// Roles a FastH3-Trim t2va generate needs: the H3 conditioner, tokenizer and
/// VAEs, with the distilled DiT in place of the base GGUF + turbo LoRA.
pub const RUNTIME_ROLES_T2VA_FASTH3_TRIM: &[&str] = &[
    role::ENCODER_GGUF,
    role::TOKENIZER,
    role::FASTH3_TRIM_NVFP4,
    role::FASTH3_TRIM_REST,
    role::VIDEO_VAE,
    role::AUDIO_VAE,
];

/// Roles a t2va (text -> video + audio) generate needs at the Q4 default.
/// Everything here is required: the base checkpoint is not distilled, so the
/// turbo LoRA is part of the run path, not an option.
pub const RUNTIME_ROLES_T2VA_Q4: &[&str] = &[
    role::ENCODER_GGUF,
    role::TOKENIZER,
    role::DIT_GGUF_Q4_K_M,
    role::TURBO_LORA,
    role::VIDEO_VAE,
    role::AUDIO_VAE,
];

/// Same chain against the Q8_0 DiT (parity canary / quality baseline).
pub const RUNTIME_ROLES_T2VA_Q8: &[&str] = &[
    role::ENCODER_GGUF,
    role::TOKENIZER,
    role::DIT_GGUF_Q8_0,
    role::TURBO_LORA,
    role::VIDEO_VAE,
    role::AUDIO_VAE,
];

pub static MANIFEST: ModelManifest = ModelManifest {
    id: "minimax-h3",
    files: &[
        (
            role::DIT_GGUF_Q4_K_M,
            FileRef::new(REPO_GGUF, "unet/MiniMax-H3-FL2VA-Q4_K_M.gguf"),
        ),
        (
            role::DIT_GGUF_Q8_0,
            FileRef::new(REPO_GGUF, "unet/MiniMax-H3-FL2VA-Q8_0.gguf"),
        ),
        (
            role::ENCODER_GGUF,
            FileRef::new(
                REPO_GGUF,
                "text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
            ),
        ),
        (
            role::ENCODER_GGUF_IQ4,
            FileRef::new(
                REPO_ENCODER_TRUNC,
                "Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-IQ4_XS.gguf",
            ),
        ),
        (
            role::ENCODER_GGUF_IQ3,
            FileRef::new(
                REPO_ENCODER_TRUNC,
                "Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-UD-IQ3_XXS.gguf",
            ),
        ),
        (
            role::TOKENIZER,
            FileRef::new(REPO_UPSTREAM, "FL2VA/text_encoder/tokenizer.json"),
        ),
        (
            role::VIDEO_VAE,
            FileRef::new(REPO_VAE, "vae/minimax_h3_video_vae_fp16.safetensors"),
        ),
        (
            role::AUDIO_VAE,
            FileRef::new(REPO_VAE, "vae/minimax_h3_audio_vae_fp32.safetensors"),
        ),
        (
            role::TURBO_LORA,
            FileRef::new(
                REPO_TURBO_LORA,
                "minimax_h3_fl2v_turbo_4step_v0.1.safetensors",
            ),
        ),
        (
            role::FASTH3_TRIM_NVFP4,
            FileRef::new(REPO_FASTH3_TRIM, "transformer/nvfp4_weights.safetensors"),
        ),
        (
            role::FASTH3_TRIM_REST,
            FileRef::new(
                REPO_FASTH3_TRIM,
                "transformer/diffusion_pytorch_model-00001-of-00001.safetensors",
            ),
        ),
        (
            role::FASTH3_OMNI,
            FileRef::new(
                REPO_FASTH3_OMNI,
                "fasth3_omniref_qad_nvfp4_comfy.safetensors",
            ),
        ),
        (role::VISION, FileRef::new(REPO_VISION, "mmproj-BF16.gguf")),
        (
            role::VAE_DECODER_NVFP4,
            FileRef::new(REPO_VAE_NVFP4, "quality-36L/decoder-nvfp4.safetensors"),
        ),
        (
            role::VAE_DECODER_NVFP4_FAST,
            FileRef::new(REPO_VAE_NVFP4, "fast-8L/decoder-nvfp4.safetensors"),
        ),
        (
            role::TURBO_LORA_ALT,
            FileRef::new(
                REPO_TURBO_LORA_ALT,
                "minimax_h3_turbo_4step_ema_ckpt850.safetensors",
            ),
        ),
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_resolve() {
        for r in RUNTIME_ROLES_T2VA_Q4
            .iter()
            .chain(RUNTIME_ROLES_T2VA_Q8)
            .chain(RUNTIME_ROLES_T2VA_FASTH3_TRIM)
            .chain(RUNTIME_ROLES_REF2VA_FASTH3_OMNI)
        {
            assert!(MANIFEST.get(r).is_some(), "missing role {r}");
        }
        assert!(MANIFEST.get("no-such-role").is_none());
    }

    /// The turbo LoRA is not optional: the base checkpoint needs ~50 steps.
    #[test]
    fn turbo_lora_is_a_runtime_role() {
        assert!(RUNTIME_ROLES_T2VA_Q4.contains(&role::TURBO_LORA));
        assert!(RUNTIME_ROLES_T2VA_Q8.contains(&role::TURBO_LORA));
    }

    /// Every encoder build we offer must already be truncated to the 50 layers
    /// H3 reads; a full 64-layer stack would carry `lm_head` plus 14 dead
    /// layers. The default rides in the DiT repo (verified 50 blocks at load,
    /// see `loader::check_encoder_depth`); the footprint tiers are the
    /// explicitly-named `L0-49` builds.
    #[test]
    fn encoder_builds_are_truncated() {
        let f = MANIFEST.get(role::ENCODER_GGUF).unwrap();
        assert_eq!(f.repo, REPO_GGUF);
        assert!(f.path.contains("minimax_h3"));
        for r in [role::ENCODER_GGUF_IQ4, role::ENCODER_GGUF_IQ3] {
            let f = MANIFEST.get(r).unwrap();
            assert_eq!(f.repo, REPO_ENCODER_TRUNC);
            assert!(f.path.contains("L0-49"));
        }
    }

    /// The tokenizer is the checkpoint's own, not a generic Qwen3-VL mirror.
    #[test]
    fn tokenizer_is_the_product_one() {
        let f = MANIFEST.get(role::TOKENIZER).unwrap();
        assert_eq!(f.repo, REPO_UPSTREAM);
        assert!(f.path.starts_with("FL2VA/text_encoder/"));
    }
}
