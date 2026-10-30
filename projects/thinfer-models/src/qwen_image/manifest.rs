//! Qwen-Image(-Edit) rapid (4-step distilled, CFG-free) file manifest. The DiT
//! is a user weight slot with no default: the project ships no 4-step
//! checkpoint, so the user's sources choose one (a DiT-only GGUF of the
//! Qwen-Image MMDiT; it carries no TE/VAE, same scope as ideogram/zimage
//! GGUFs). The encoder defaults to the upstream Qwen2.5-VL-7B-Instruct GGUF,
//! overridable per slot. Parity discipline: the pyref dequantizes the SAME GGUF
//! the engine loads, so kernel correctness is isolated from quant loss. Canary
//! = Q8_0; runtime default = Q4_K_M DiT (encoder/VAE stay Q8/f16).
//!
//! Roles are open-set strings to keep `ModelManifest` model-agnostic; the
//! `role::*` consts are the typed accessors for callers.

use thinfer_core::manifest::{FileRef, ModelManifest};

/// Upstream Qwen2.5-VL-7B-Instruct as GGUF (LM Q8_0 + vision mmproj f16).
const REPO_ENCODER: &str = "unsloth/Qwen2.5-VL-7B-Instruct-GGUF";

/// The Qwen2.5-VL-7B LM GGUF (Q8_0), in its user slot. Shared with
/// HunyuanVideo 1.5, which embeds with the same encoder.
pub const ENCODER_Q8_0: FileRef =
    FileRef::new(REPO_ENCODER, "Qwen2.5-VL-7B-Instruct-Q8_0.gguf").in_slot(slot::ENCODER_Q8_0);
/// Full Wan-family Qwen-Image KL VAE (diffusers `AutoencoderKLQwenImage`,
/// encoder + decoder). The decode-only calcuis "pig" GGUF cannot serve the edit
/// path's reference-image VAE-encode, so we source the complete VAE here and use
/// it for both encode and decode (one source, native diffusers tensor keys).
const REPO_VAE: &str = "Qwen/Qwen-Image";
/// Qwen2.5-VL tokenizer + processor (vision preprocess config). The GGUF embeds
/// tokenizer metadata, but the engine consumes a standard `tokenizer.json`.
const REPO_TOKENIZER: &str = "Qwen/Qwen2.5-VL-7B-Instruct";

/// User weight slots (see `thinfer_core::manifest::FileRef::slot`). The encoder
/// slots are shared with HunyuanVideo 1.5, which embeds with the same
/// Qwen2.5-VL-7B, so one sources entry serves both.
pub mod slot {
    /// 4-step distilled Qwen-Image(-Edit) MMDiT GGUF, Q8_0. No default.
    pub const DIT_Q8_0: &str = "qwen-image-rapid/dit-q8_0";
    /// 4-step distilled Qwen-Image(-Edit) MMDiT GGUF, Q4_K_M. No default.
    pub const DIT_Q4_K_M: &str = "qwen-image-rapid/dit-q4_k_m";
    /// Qwen2.5-VL-7B LM GGUF, Q8_0.
    pub const ENCODER_Q8_0: &str = "qwen2.5-vl-7b/encoder-q8_0";
    /// Qwen2.5-VL-7B vision tower (mmproj), f16.
    pub const MMPROJ_F16: &str = "qwen2.5-vl-7b/mmproj-f16";
}

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
        (role::DIT_GGUF_Q8_0, FileRef::user(slot::DIT_Q8_0)),
        (role::DIT_GGUF_Q4_K_M, FileRef::user(slot::DIT_Q4_K_M)),
        (role::ENCODER_GGUF_Q8_0, ENCODER_Q8_0),
        (
            role::MMPROJ_F16,
            FileRef::new(REPO_ENCODER, "mmproj-F16.gguf").in_slot(slot::MMPROJ_F16),
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
    fn dit_slots_are_user_filled_per_quant() {
        let q8 = MANIFEST.get(role::DIT_GGUF_Q8_0).unwrap();
        let q4 = MANIFEST.get(role::DIT_GGUF_Q4_K_M).unwrap();
        assert!(q8.is_unfilled() && q4.is_unfilled());
        assert_eq!(
            (q8.slot, q4.slot),
            (Some(slot::DIT_Q8_0), Some(slot::DIT_Q4_K_M))
        );
        // The encoder has an upstream default, still overridable.
        let enc = MANIFEST.get(role::ENCODER_GGUF_Q8_0).unwrap();
        assert!(!enc.is_unfilled() && enc.slot == Some(slot::ENCODER_Q8_0));
    }
}
