//! Qwen-Image-2.1-Turbo file manifest. The DiT is unsloth's GGUF of the Turbo
//! checkpoint (ComfyUI keys under `model.diffusion_model.`, `img_mlp.gate_up`
//! fused `[gate_layer; proj]`, mixed `_K_M` quant per tensor). The text encoder
//! is byte-identical to stock Qwen3-VL-8B-Instruct, so the stock unsloth GGUF
//! and its mmproj serve it. The VAE and tokenizer come from the official repo.

use thinfer_core::manifest::{FileRef, ModelManifest};

const REPO_DIT: &str = "unsloth/Qwen-Image-2.1-Turbo-GGUF";
const REPO_ENCODER: &str = "unsloth/Qwen3-VL-8B-Instruct-GGUF";
const REPO_UPSTREAM: &str = "Qwen/Qwen-Image-2.1-Turbo";

pub mod role {
    /// DiT GGUF, Q8_0 (parity canary).
    pub const DIT_GGUF_Q8_0: &str = "dit/gguf-q8_0";
    /// DiT GGUF, Q4_K_M (default: fits an 8 GB card whole).
    pub const DIT_GGUF_Q4_K_M: &str = "dit/gguf-q4_k_m";
    /// Qwen3-VL-8B language model GGUF (llama.cpp names).
    pub const ENCODER_GGUF: &str = "encoder/gguf";
    /// Qwen3-VL-8B vision tower (mmproj).
    pub const VISION: &str = "encoder/mmproj";
    /// The 2D residual VAE (diffusers safetensors).
    pub const VAE: &str = "vae/safetensors";
    /// Qwen3-VL tokenizer.
    pub const TOKENIZER: &str = "tokenizer/json";
}

/// Everything a generate needs (the vision tower serves edits; it is small).
pub const RUNTIME_ROLES_Q4: &[&str] = &[
    role::ENCODER_GGUF,
    role::VISION,
    role::DIT_GGUF_Q4_K_M,
    role::VAE,
    role::TOKENIZER,
];

/// The same with the Q8_0 canary DiT.
pub const RUNTIME_ROLES_Q8: &[&str] = &[
    role::ENCODER_GGUF,
    role::VISION,
    role::DIT_GGUF_Q8_0,
    role::VAE,
    role::TOKENIZER,
];

pub static MANIFEST: ModelManifest = ModelManifest {
    id: "qwen-image-2.1-turbo",
    files: &[
        (
            role::DIT_GGUF_Q8_0,
            FileRef::new(REPO_DIT, "qwen-image-2.1-turbo-Q8_0.gguf"),
        ),
        (
            role::DIT_GGUF_Q4_K_M,
            FileRef::new(REPO_DIT, "qwen-image-2.1-turbo-Q4_K_M.gguf"),
        ),
        (
            role::ENCODER_GGUF,
            FileRef::new(REPO_ENCODER, "Qwen3-VL-8B-Instruct-Q8_0.gguf"),
        ),
        (role::VISION, FileRef::new(REPO_ENCODER, "mmproj-F16.gguf")),
        (
            role::VAE,
            FileRef::new(REPO_UPSTREAM, "vae/diffusion_pytorch_model.safetensors"),
        ),
        (
            role::TOKENIZER,
            FileRef::new(REPO_UPSTREAM, "processor/tokenizer.json"),
        ),
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_resolve() {
        for r in RUNTIME_ROLES_Q4.iter().chain(RUNTIME_ROLES_Q8) {
            assert!(MANIFEST.get(r).is_some(), "missing role {r}");
        }
    }
}
