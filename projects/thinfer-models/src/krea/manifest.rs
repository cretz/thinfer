//! Krea 2 Turbo file manifest. The DiT GGUF (`realrebelai/KREA-2_GGUFs`) carries
//! krea2-native tensor keys consumed 1:1 by [`crate::krea::loader`]; the text
//! encoder is a Qwen3-VL-4B GGUF (llama.cpp `blk.N.*` names, remapped via
//! [`crate::z_image::qwen3_gguf_renames`]); the VAE is the Wan2.1 KL VAE in
//! diffusers naming (consumed by [`crate::wan::vae`]). Parity discipline: the
//! pyref dequantizes the SAME DiT GGUF the engine loads. Canary = Q8_0 DiT;
//! runtime footprint option = Q4_K_M DiT.
//!
//! Roles are open-set strings; the `role::*` consts are the typed accessors.

use thinfer_core::manifest::{FileRef, ModelManifest};

const REPO_DIT: &str = "realrebelai/KREA-2_GGUFs";
const REPO_ENCODER: &str = "unsloth/Qwen3-VL-4B-Instruct-GGUF";
/// Wan2.1 VAE in diffusers naming, from the Wan2.2-A14B diffusers bundle (the
/// same source the Wan port uses).
const REPO_VAE: &str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers";
/// Qwen3-VL-4B tokenizer (chat template + vocab).
const REPO_TOKENIZER: &str = "Qwen/Qwen3-VL-4B-Instruct";

pub mod role {
    /// DiT GGUF, Q8_0 (parity canary + quality tier).
    pub const DIT_GGUF_Q8_0: &str = "dit/gguf-q8_0";
    /// DiT GGUF, Q4_K_M (footprint option).
    pub const DIT_GGUF_Q4_K_M: &str = "dit/gguf-q4_k_m";
    /// Qwen3-VL-4B text-encoder GGUF (Q8_0; bf16 acts at runtime). Must be a
    /// UNIFORM-quant GGUF: the encoder shares one compiled `BlockPipelines`
    /// across all layers, so a `_K_M` mix (which bumps `attn_v`/`ffn_down` to
    /// Q6_K while q/k stay Q5_K) dequantizes those tensors with the wrong scheme
    /// -> NaN. Q8_0 (no `_K_*` suffix) is uniform per-tensor.
    pub const ENCODER_GGUF: &str = "encoder/gguf";
    /// Wan2.1 KL VAE (diffusers safetensors, decoder path).
    pub const VAE: &str = "vae/safetensors";
    /// Qwen3-VL-4B `tokenizer.json`.
    pub const TOKENIZER: &str = "tokenizer/json";
}

/// Roles a Q8_0-canary generate needs, in download order.
pub const RUNTIME_ROLES_Q8: &[&str] = &[
    role::ENCODER_GGUF,
    role::DIT_GGUF_Q8_0,
    role::VAE,
    role::TOKENIZER,
];

/// Roles a Q4_K_M-footprint generate needs.
pub const RUNTIME_ROLES_Q4: &[&str] = &[
    role::ENCODER_GGUF,
    role::DIT_GGUF_Q4_K_M,
    role::VAE,
    role::TOKENIZER,
];

pub static MANIFEST: ModelManifest = ModelManifest {
    id: "krea-2-turbo",
    files: &[
        (
            role::DIT_GGUF_Q8_0,
            FileRef::new(REPO_DIT, "TURBO/Krea-2-Turbo-Q8_0.gguf"),
        ),
        (
            role::DIT_GGUF_Q4_K_M,
            FileRef::new(REPO_DIT, "TURBO/Krea-2-Turbo-Q4_K_M.gguf"),
        ),
        (
            role::ENCODER_GGUF,
            FileRef::new(REPO_ENCODER, "Qwen3-VL-4B-Instruct-Q8_0.gguf"),
        ),
        (
            role::VAE,
            FileRef::new(REPO_VAE, "vae/diffusion_pytorch_model.safetensors"),
        ),
        (
            role::TOKENIZER,
            FileRef::new(REPO_TOKENIZER, "tokenizer.json"),
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
    fn dit_canary_is_q8() {
        assert!(
            MANIFEST
                .get(role::DIT_GGUF_Q8_0)
                .unwrap()
                .path
                .ends_with("Q8_0.gguf")
        );
    }
}
