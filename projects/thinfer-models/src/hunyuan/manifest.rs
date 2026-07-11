//! HunyuanVideo 1.5 T2V file manifest (480p lightx2v 4-step distill). Four
//! roles: the DiT (lightx2v fp16 safetensors, the whole transformer), the Comfy
//! repackaged fp16 VAE, and the text encoder + tokenizer REUSED verbatim from
//! `qwen_image` (the same cached abliterated Qwen2.5-VL-7B Q8_0 GGUF + clean
//! tokenizer.json -- zero new download; see the ENCODER DECISION note in the
//! worklog). The encoder is embedding-only, so abliteration drift on benign
//! prompt embeddings is moot; swap to a clean Instruct Q8_0 GGUF here (one-line)
//! if a serve eyeball ever shows adherence drift.
//!
//! Roles are open-set strings to keep `ModelManifest` model-agnostic; the
//! `role::*` consts are the typed accessors for callers.

use thinfer_core::manifest::{FileRef, ModelManifest};

/// lightx2v step-distill repo. The 480p T2V 4-step checkpoint is the WHOLE DiT
/// (fp16 safetensors, 1793 tensors), loaded direct (residency narrows to bf16).
const REPO_DIT: &str = "lightx2v/Hy1.5-Distill-Models";
/// Comfy-Org repackaged HunyuanVideo 1.5 bundle; we take only the fp16 VAE.
const REPO_VAE: &str = "Comfy-Org/HunyuanVideo_1.5_repackaged";
/// Qwen2.5-VL-7B encoder GGUF lives in the Qwen-Image AIO repo (same file the
/// qwen_image port already caches). Embedding-only use -> abliteration is moot.
const REPO_ENCODER: &str = "Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF";
/// Clean Qwen2.5-VL tokenizer (vocab + chat template). Already cached by
/// qwen_image; the Hunyuan host-side video prompt template wraps these tokens.
const REPO_TOKENIZER: &str = "Qwen/Qwen2.5-VL-7B-Instruct";
/// TAEHV tiny decoder (`taehv1_5.pth`, ~22MB fp16 state_dict, madebyollin TAEHV
/// family). Opt-in `VaeChoice::Tiny` fast path; downloaded only when selected.
const REPO_TAEHV: &str = "Overworld-Models/taehv1_5";
/// Hunyuan-1.5 FINE-TUNE of the SAME TAEHV decoder (`taehv1_5.pth`, identical
/// arch + key names, different repo + config.json). `VaeChoice::TinyFt`: the
/// tiny decoder's SECONDS-fast decode but tuned on Hunyuan-1.5 latents, so it
/// sits between the base tiny (draft) and the ~22min full conv3d VAE in quality.
/// Opt-in, downloaded only when selected; loads through the same `TaehvDecoder`
/// + `taehv_config` (the fine-tune does not change z_dim/compression/norm).
const REPO_TAEHV_FT: &str = "Overworld-Models/taehv-hy1_5-ft";
/// minWM (Causal-Forcing/WorldPlay) causal AR distill of the HY1.5 8B DiT:
/// `HY15/TI2V/dmd` is the final 4-step DMD student (F32 safetensors, same 1793
/// diffusers tensor names as the T2V DiT; `img_in` takes 65 ch = 32 latent + 32
/// cond-latent + 1 mask). I2V: first-frame image conditioning, chunked KV-cache
/// generation.
const REPO_DIT_I2V: &str = "MIN-Lab/minWM";
/// SigLIP so400m-patch14-384 vision tower (the FLUX.1-Redux image encoder,
/// ungated Comfy repackage). Encodes the I2V conditioning image into 729x1152
/// tokens for the DiT's `vision_in` stream.
const REPO_SIGLIP: &str = "Comfy-Org/sigclip_vision_384";
/// Native prompt-rewriter LM (FULL quality): Qwen3-VL-8B-Instruct as a text-only
/// GGUF. Rewrites terse user prompts into the richer phrasing the HunyuanVideo
/// DiT expects. Text-only (no vision tensors); loaded by the `qwen3_lm` module.
const REPO_REWRITER_GGUF_8B: &str = "unsloth/Qwen3-VL-8B-Instruct-GGUF";
/// Native prompt-rewriter LM (FAST, the default): Qwen3-VL-4B-Instruct text-only
/// GGUF (~2.5GB Q5_K_M). Same `qwen3vl` arch as the 8B (narrower + tied lm_head),
/// so it loads through the same runtime-parameterized `qwen3_lm` stack; small
/// enough to decode fast even under a tight VRAM budget where the 8B streams.
const REPO_REWRITER_GGUF_4B: &str = "unsloth/Qwen3-VL-4B-Instruct-GGUF";
/// Clean Qwen3-VL tokenizer (vocab + chat template) for the rewriter. Shared by
/// both sizes (the Qwen3 tokenizer is identical across the family).
const REPO_REWRITER_TOKENIZER: &str = "Qwen/Qwen3-VL-8B-Instruct";

pub mod role {
    /// lightx2v 480p T2V 4-step DiT (fp16 safetensors, whole transformer).
    pub const DIT: &str = "dit/safetensors";
    /// minWM causal AR I2V DiT (F32 safetensors, whole transformer).
    pub const DIT_I2V: &str = "dit/i2v-safetensors";
    /// SigLIP so400m vision tower (fp16 safetensors) for I2V image conditioning.
    pub const SIGLIP: &str = "vision/siglip";
    /// HunyuanVideo 1.5 VAE (fp16 safetensors, Comfy repackaged).
    pub const VAE: &str = "vae/safetensors";
    /// Qwen2.5-VL-7B encoder GGUF, Q8_0 (bf16 acts at runtime). Shared with the
    /// qwen_image port (identical cached file).
    pub const ENCODER_GGUF_Q8_0: &str = "encoder/gguf-q8_0";
    /// Qwen2.5-VL `tokenizer.json` (vocab + chat template).
    pub const TOKENIZER: &str = "tokenizer/json";
    /// TAEHV (`taehv1_5`) tiny decoder, `.pth` state_dict. Opt-in
    /// (`VaeChoice::Tiny`); NOT in [`RUNTIME_ROLES`] (selected per-request).
    pub const TINY_VAE: &str = "vae/tiny";
    /// Hunyuan-1.5 fine-tuned TAEHV decoder, `.pth` state_dict. Opt-in
    /// (`VaeChoice::TinyFt`); same arch as [`TINY_VAE`], selected per-request.
    pub const TINY_VAE_FT: &str = "vae/tiny-ft";
    /// Native prompt-rewriter Qwen3-VL-4B-Instruct GGUF, Q5_K_M: the FAST default
    /// (`RewriteQuality::Fast`). Opt-in per-request; NOT in [`RUNTIME_ROLES`].
    pub const REWRITER_GGUF_4B_Q5_K_M: &str = "rewriter/gguf-4b-q5_k_m";
    /// Native prompt-rewriter Qwen3-VL-8B-Instruct GGUF, Q5_K_M: the FULL-quality
    /// option (`RewriteQuality::Full`). Opt-in per-request; NOT in [`RUNTIME_ROLES`].
    pub const REWRITER_GGUF_8B_Q5_K_M: &str = "rewriter/gguf-8b-q5_k_m";
    /// Qwen3-VL `tokenizer.json` (vocab + chat template) for the rewriter. Opt-in
    /// per-request; NOT in [`RUNTIME_ROLES`].
    pub const REWRITER_TOKENIZER: &str = "rewriter/tokenizer";
}

/// Roles a runtime T2V generate needs, in download order (encoder first, then
/// DiT + VAE + tokenizer).
pub const RUNTIME_ROLES: &[&str] = &[
    role::ENCODER_GGUF_Q8_0,
    role::DIT,
    role::VAE,
    role::TOKENIZER,
];

/// Roles a runtime causal I2V generate needs, in download order. Everything but
/// the DiT + SigLIP is shared with (and usually already cached by) the T2V model.
pub const RUNTIME_ROLES_I2V: &[&str] = &[
    role::ENCODER_GGUF_Q8_0,
    role::DIT_I2V,
    role::SIGLIP,
    role::VAE,
    role::TOKENIZER,
];

pub static MANIFEST: ModelManifest = ModelManifest {
    id: "hunyuan-video-1.5-t2v",
    files: &[
        (
            role::DIT,
            FileRef::new(REPO_DIT, "hy1.5_t2v_480p_lightx2v_4step.safetensors"),
        ),
        (
            role::VAE,
            FileRef::new(
                REPO_VAE,
                "split_files/vae/hunyuanvideo15_vae_fp16.safetensors",
            ),
        ),
        (
            role::ENCODER_GGUF_Q8_0,
            FileRef::new(
                REPO_ENCODER,
                "Qwen2.5-VL-7B-Instruct-abliterated/Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf",
            ),
        ),
        (
            role::TOKENIZER,
            FileRef::new(REPO_TOKENIZER, "tokenizer.json"),
        ),
        (role::TINY_VAE, FileRef::new(REPO_TAEHV, "taehv1_5.pth")),
        (
            role::TINY_VAE_FT,
            FileRef::new(REPO_TAEHV_FT, "taehv1_5.pth"),
        ),
        (
            role::REWRITER_GGUF_4B_Q5_K_M,
            FileRef::new(REPO_REWRITER_GGUF_4B, "Qwen3-VL-4B-Instruct-Q5_K_M.gguf"),
        ),
        (
            role::REWRITER_GGUF_8B_Q5_K_M,
            FileRef::new(REPO_REWRITER_GGUF_8B, "Qwen3-VL-8B-Instruct-Q5_K_M.gguf"),
        ),
        (
            role::REWRITER_TOKENIZER,
            FileRef::new(REPO_REWRITER_TOKENIZER, "tokenizer.json"),
        ),
        (
            role::DIT_I2V,
            FileRef::new(
                REPO_DIT_I2V,
                "HY15/TI2V/dmd/diffusion_pytorch_model.safetensors",
            ),
        ),
        (
            role::SIGLIP,
            FileRef::new(REPO_SIGLIP, "sigclip_vision_patch14_384.safetensors"),
        ),
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_resolve() {
        for r in RUNTIME_ROLES.iter().chain(RUNTIME_ROLES_I2V) {
            assert!(MANIFEST.get(r).is_some(), "missing role {r}");
        }
        assert!(MANIFEST.get("no-such-role").is_none());
    }

    #[test]
    fn rewriter_roles_resolve() {
        // Opt-in rewriter roles are NOT in RUNTIME_ROLES but must still resolve.
        for r in [
            role::REWRITER_GGUF_4B_Q5_K_M,
            role::REWRITER_GGUF_8B_Q5_K_M,
            role::REWRITER_TOKENIZER,
        ] {
            assert!(MANIFEST.get(r).is_some(), "missing role {r}");
        }
    }
}
