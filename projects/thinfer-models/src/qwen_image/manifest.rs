//! Qwen-Image-Edit-Rapid-AIO file manifest. All weights are GGUF, sourced per
//! the Phil2Sat recipe (the DiT-only GGUF carries no TE/VAE, same scope as
//! ideogram/zimage GGUFs). Parity discipline: the pyref dequantizes the SAME
//! GGUF the engine loads, so kernel correctness is isolated from quant loss.
//! Canary = Q8_0; runtime default = Q4_K_M DiT (encoder/VAE stay Q8/f16).
//!
//! Roles are open-set strings to keep `ModelManifest` model-agnostic; the
//! `role::*` consts are the typed accessors for callers.

use thinfer_core::manifest::{FileRef, ModelManifest};

/// DiT + abliterated Qwen2.5-VL encoder + mmproj all live in one repo. The DiT
/// is versioned under `vNN/` subdirs; v9.0 is the latest at port time.
const REPO_AIO: &str = "Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF";
/// Full Wan-family Qwen-Image KL VAE (diffusers `AutoencoderKLQwenImage`,
/// encoder + decoder). The decode-only calcuis "pig" GGUF cannot serve the edit
/// path's reference-image VAE-encode, so we source the complete VAE here and use
/// it for both encode and decode (one source, native diffusers tensor keys).
const REPO_VAE: &str = "Qwen/Qwen-Image";
/// Qwen2.5-VL tokenizer + processor (vision preprocess config). The GGUF embeds
/// tokenizer metadata, but the engine consumes a standard `tokenizer.json`.
const REPO_TOKENIZER: &str = "Qwen/Qwen2.5-VL-7B-Instruct";

pub mod role {
    /// DiT GGUF, Q8_0 (parity canary).
    pub const DIT_GGUF_Q8_0: &str = "dit/gguf-q8_0";
    /// DiT GGUF, Q4_K_M (runtime default).
    pub const DIT_GGUF_Q4_K_M: &str = "dit/gguf-q4_k_m";
    /// Qwen2.5-VL-7B LM GGUF, Q8_0 (bf16 acts at runtime).
    pub const ENCODER_GGUF_Q8_0: &str = "encoder/gguf-q8_0";
    /// Qwen2.5-VL vision tower (mmproj), f16. Needed for the edit image channel.
    pub const MMPROJ_F16: &str = "encoder/mmproj-f16";
    /// Qwen-Image KL VAE (diffusers safetensors, encoder + decoder, bf16).
    pub const VAE: &str = "vae/safetensors";
    /// Qwen2.5-VL `tokenizer.json` (vocab + chat template).
    pub const TOKENIZER: &str = "tokenizer/json";
    /// Qwen2.5-VL `preprocessor_config.json` (vision resize/normalize params).
    pub const PREPROCESSOR: &str = "tokenizer/preprocessor";
}

/// Roles a runtime generate needs, canary (Q8_0 DiT) in download order.
pub const RUNTIME_ROLES_Q8: &[&str] = &[
    role::ENCODER_GGUF_Q8_0,
    role::MMPROJ_F16,
    role::DIT_GGUF_Q8_0,
    role::VAE,
    role::TOKENIZER,
    role::PREPROCESSOR,
];

/// Roles a TEXT-TO-IMAGE generate needs: no vision tower, so the mmproj +
/// preprocessor are omitted (the encoder runs text-only). Q8_0 DiT canary.
pub const RUNTIME_ROLES_T2I_Q8: &[&str] = &[
    role::ENCODER_GGUF_Q8_0,
    role::DIT_GGUF_Q8_0,
    role::VAE,
    role::TOKENIZER,
];

/// Runtime default: Q4_K_M DiT, everything else unchanged.
pub const RUNTIME_ROLES_Q4: &[&str] = &[
    role::ENCODER_GGUF_Q8_0,
    role::MMPROJ_F16,
    role::DIT_GGUF_Q4_K_M,
    role::VAE,
    role::TOKENIZER,
    role::PREPROCESSOR,
];

pub static MANIFEST: ModelManifest = ModelManifest {
    id: "qwen-image-edit-rapid",
    files: &[
        (
            role::DIT_GGUF_Q8_0,
            FileRef::new(REPO_AIO, "v90/qwen-rapid-nsfw-v9.0-Q8_0.gguf"),
        ),
        (
            role::DIT_GGUF_Q4_K_M,
            FileRef::new(REPO_AIO, "v90/qwen-rapid-nsfw-v9.0-Q4_K_M.gguf"),
        ),
        (
            role::ENCODER_GGUF_Q8_0,
            FileRef::new(
                REPO_AIO,
                "Qwen2.5-VL-7B-Instruct-abliterated/Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf",
            ),
        ),
        (
            role::MMPROJ_F16,
            FileRef::new(
                REPO_AIO,
                "Qwen2.5-VL-7B-Instruct-abliterated/Qwen2.5-VL-7B-Instruct-abliterated.mmproj-f16.gguf",
            ),
        ),
        (
            role::VAE,
            FileRef::new(REPO_VAE, "vae/diffusion_pytorch_model.safetensors"),
        ),
        (
            role::TOKENIZER,
            FileRef::new(REPO_TOKENIZER, "tokenizer.json"),
        ),
        (
            role::PREPROCESSOR,
            FileRef::new(REPO_TOKENIZER, "preprocessor_config.json"),
        ),
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_resolve() {
        for r in RUNTIME_ROLES_Q8.iter().chain(RUNTIME_ROLES_Q4) {
            assert!(MANIFEST.get(r).is_some(), "missing role {r}");
        }
        assert!(MANIFEST.get("no-such-role").is_none());
    }

    #[test]
    fn dit_canary_is_q8_default_is_q4() {
        assert!(
            MANIFEST
                .get(role::DIT_GGUF_Q8_0)
                .unwrap()
                .path
                .ends_with("Q8_0.gguf")
        );
        assert!(
            MANIFEST
                .get(role::DIT_GGUF_Q4_K_M)
                .unwrap()
                .path
                .ends_with("Q4_K_M.gguf")
        );
    }
}
