//! Ideogram-4 file manifest. All weights are sourced UNGATED (the official
//! `ideogram-ai/ideogram-4-fp8` repo is gated): the DiT and text encoder come
//! from community GGUFs, the VAE from FLUX.2-VAE safetensors, and the
//! turbotime LoRA from ostris. See `ideogram-plan.md` "RESOLVED".
//!
//! Parity discipline: the pyref dequantizes the SAME GGUF the engine loads
//! (no bf16 download), so kernel correctness is isolated from quantization
//! loss. Canary = Q8_0 everywhere; the runtime defaults (encoder Q5_K_M, DiT
//! Q4_K) land once parity is GREEN on the canary.
//!
//! Roles are open-set strings to keep `ModelManifest` model-agnostic; the
//! `role::*` consts are the typed accessors for callers.

use thinfer_core::manifest::{FileRef, ModelManifest};

/// Qwen3-VL-8B-Instruct language-model GGUFs (LM only; the vision mmproj is
/// NOT needed for t2i). Tensor keys are the standard llama.cpp `blk.{i}.*`
/// form, so `z_image::qwen3_gguf_renames` maps them 1:1.
const REPO_ENCODER_GGUF: &str = "unsloth/Qwen3-VL-8B-Instruct-GGUF";
/// DiT-only GGUF (no TE/VAE). Keys are 1:1 with the modeling module paths,
/// so no rename map is needed.
const REPO_DIT_GGUF: &str = "Abiray/ideogram-4-GGUF";
/// Flux2 KL VAE; the diffusers-layout safetensors `autoencoder.py`'s
/// `convert_diffusers_state_dict` consumes.
const REPO_VAE: &str = "black-forest-labs/FLUX.2-VAE";
/// ostris turbotime LoRA (rank 128, no `.alpha` -> scale 1.0). Folded at load.
const REPO_LORA: &str = "ostris/ideogram_4_turbotime_lora";
/// Qwen3-VL tokenizer (the chat template + vocab). The unsloth GGUF embeds
/// tokenizer metadata, but the engine consumes a standard `tokenizer.json` via
/// `HfTokenizer`, so pull it from the base (ungated) repo.
const REPO_TOKENIZER: &str = "Qwen/Qwen3-VL-8B-Instruct";

pub mod role {
    /// Encoder LM GGUF, Q8_0 (near-lossless; bf16 acts at runtime).
    pub const ENCODER_GGUF_Q8_0: &str = "encoder/gguf-q8_0";
    /// DiT GGUF, Q8_0 (the runtime weights; the LoRA folds to Q8_0).
    pub const DIT_GGUF_Q8_0: &str = "dit/gguf-q8_0";
    /// Flux2 KL VAE decoder (safetensors, bf16).
    pub const VAE: &str = "vae/decoder";
    /// turbotime LoRA safetensors.
    pub const LORA: &str = "lora/turbotime";
    /// Qwen3-VL `tokenizer.json` (chat template + vocab).
    pub const TOKENIZER: &str = "tokenizer/json";
}

/// Roles a runtime generate needs, in download order: Q8_0 encoder + DiT, the
/// VAE, LoRA, and tokenizer. The CLI/serve `required_files` resolve these
/// against [`MANIFEST`]. (A Q4_K DiT default was tried and dropped -- the
/// per-request fold re-quantizes the whole DiT, so Q4_K was ~2x slower than
/// Q8_0 with worse quality; not worth the smaller download here.)
pub const RUNTIME_ROLES_Q8: &[&str] = &[
    role::ENCODER_GGUF_Q8_0,
    role::DIT_GGUF_Q8_0,
    role::VAE,
    role::LORA,
    role::TOKENIZER,
];

pub static MANIFEST: ModelManifest = ModelManifest {
    id: "ideogram-4",
    files: &[
        (
            role::ENCODER_GGUF_Q8_0,
            FileRef::new(REPO_ENCODER_GGUF, "Qwen3-VL-8B-Instruct-Q8_0.gguf"),
        ),
        (
            role::DIT_GGUF_Q8_0,
            FileRef::new(REPO_DIT_GGUF, "ideogram4-Q8_0.gguf"),
        ),
        (
            role::VAE,
            FileRef::new(REPO_VAE, "diffusion_pytorch_model.safetensors"),
        ),
        (
            role::LORA,
            FileRef::new(REPO_LORA, "ideogram_4_turbotime_v1.safetensors"),
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
        for r in [
            role::ENCODER_GGUF_Q8_0,
            role::DIT_GGUF_Q8_0,
            role::VAE,
            role::LORA,
            role::TOKENIZER,
        ] {
            assert!(MANIFEST.get(r).is_some(), "missing role {r}");
        }
        assert!(MANIFEST.get("no-such-role").is_none());
    }

    #[test]
    fn encoder_canary_points_at_q8() {
        let f = MANIFEST.get(role::ENCODER_GGUF_Q8_0).expect("q8 role");
        assert_eq!(f.repo, REPO_ENCODER_GGUF);
        assert!(f.path.ends_with("Q8_0.gguf"));
    }
}
