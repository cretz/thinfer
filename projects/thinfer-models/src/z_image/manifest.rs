//! Z-Image-Turbo file manifest. DiT comes from `dimitribarbot/Z-Image-Turbo-BF16`
//! (clean bf16 conversion of the upstream sharded transformer, 1063 keys, no
//! EMA extraction step that broke earlier third-party releases). Text encoder,
//! tokenizer, VAE come from upstream `Tongyi-MAI/Z-Image-Turbo`.
//!
//! Earlier the DiT role pointed at `tsqn/Z-Image-Turbo_fp32-fp16-bf16_full_and_ema-only`
//! `diffusion_pytorch_model-ema-only-bf16.safetensors`, which reddit/HF reports
//! describe as a buggy EMA-only extraction. Symptom: gray / black-screen output.
//! See worklog for the investigation that pinpointed weight provenance as the
//! cause (engine VAE was already bit-exact vs PyTorch).
//!
//! M1 dtype: bf16 storage everywhere, expand-to-fp32 at GPU upload, fp32
//! kernels. fp16 is intentionally avoided: Z-Image was trained in bf16 and
//! the magnitude clamp from a bf16->fp16 cast produces broken weights (range
//! exceeds fp16 max). See `plan-details.md`.
//!
//! Roles are open-set strings to keep `ModelManifest` model-agnostic; the
//! `role::*` consts below are the typed accessors for callers.

use thinfer_core::manifest::{FileRef, ModelManifest};

const REPO_DIT_BF16: &str = "dimitribarbot/Z-Image-Turbo-BF16";
const REPO_UPSTREAM: &str = "Tongyi-MAI/Z-Image-Turbo";
/// unsloth quantized GGUFs (DiT only). bf16 / Q8_0 / Q4_K_M / etc all live
/// here under filenames like `z-image-turbo-Q8_0.gguf`. The DiT matmuls
/// surface through the GGUF; norms, biases, AdaLN, and everything else
/// stay safetensors and union over the top via
/// `thinfer_core::format::union::UnionSource`.
const REPO_GGUF: &str = "unsloth/Z-Image-Turbo-GGUF";

pub mod role {
    /// DiT (transformer). bf16 storage, sharded across 2 safetensors files
    /// from `dimitribarbot/Z-Image-Turbo-BF16`. Expand to fp32 at GPU upload.
    pub const DIT_SHARD_1: &str = "dit/shard1";
    pub const DIT_SHARD_2: &str = "dit/shard2";
    pub const DIT_INDEX: &str = "dit/index";
    /// Qwen3-4B text encoder shards. Upstream `Tongyi-MAI` ships as a 3-shard
    /// safetensors split.
    pub const TEXT_ENCODER_SHARD_1: &str = "text_encoder/shard1";
    pub const TEXT_ENCODER_SHARD_2: &str = "text_encoder/shard2";
    pub const TEXT_ENCODER_SHARD_3: &str = "text_encoder/shard3";
    pub const TEXT_ENCODER_INDEX: &str = "text_encoder/index";
    pub const TOKENIZER_JSON: &str = "tokenizer/tokenizer.json";
    pub const TOKENIZER_CONFIG: &str = "tokenizer/tokenizer_config";
    pub const VAE: &str = "vae/decoder";
    pub const VAE_CONFIG: &str = "vae/config";
    pub const SCHEDULER_CONFIG: &str = "scheduler/config";
    /// DiT-only GGUF, Q8_0. Pulled from unsloth and unioned with the
    /// safetensors source so only the matmul tensors get quantized;
    /// everything else (AdaLN, norms, embedders, refiners' non-matmul
    /// weights) stays bf16 safetensors.
    pub const DIT_GGUF_Q8_0: &str = "dit/gguf-q8_0";
    /// DiT-only GGUF, Q4_0. Same union pattern as Q8_0; halves the
    /// weight footprint (4-bit nibbles vs 8-bit ints, same 32-elem
    /// block geometry) and halves load bandwidth.
    pub const DIT_GGUF_Q4_0: &str = "dit/gguf-q4_0";
    /// DiT-only GGUF, Q4_K_M. Same union pattern as Q8_0.
    pub const DIT_GGUF_Q4_K_M: &str = "dit/gguf-q4_k_m";
}

/// Compute recipe for Z-Image-Turbo. Z-Image was trained in bf16 and the
/// pytorch reference (`diffusers --dtype bf16`) RNE-quantizes every module
/// output to bf16; matching that at inference is required for parity.
/// `ZImageModel::load` reads this and compiles `BlockPipelines` with the
/// corresponding `WgslConfig`. Per-call user overrides are intentionally
/// absent: dtype semantics are a model property, not a runtime knob.
pub struct ZImageRecipe {
    /// RNE-quantize every activation-producing store to bf16 in-shader.
    pub bf16_quant_writes: bool,
}

pub static RECIPE: ZImageRecipe = ZImageRecipe {
    bf16_quant_writes: true,
};

/// M1 manifest. DiT is the bf16 sharded transformer from `dimitribarbot`. Text
/// encoder, tokenizer, VAE come from upstream `Tongyi-MAI`. Other roles get
/// added as their loaders land (see `worklog.md`). All callers should resolve
/// via `MANIFEST.get()` so adding entries doesn't churn call sites.
pub static MANIFEST: ModelManifest = ModelManifest {
    id: "zimage-turbo-m1",
    files: &[
        (
            role::DIT_INDEX,
            FileRef::new(
                REPO_DIT_BF16,
                "transformer/diffusion_pytorch_model.safetensors.index.json",
            ),
        ),
        (
            role::DIT_SHARD_1,
            FileRef::new(
                REPO_DIT_BF16,
                "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            ),
        ),
        (
            role::DIT_SHARD_2,
            FileRef::new(
                REPO_DIT_BF16,
                "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            ),
        ),
        (
            role::TEXT_ENCODER_INDEX,
            FileRef::new(REPO_UPSTREAM, "text_encoder/model.safetensors.index.json"),
        ),
        (
            role::TEXT_ENCODER_SHARD_1,
            FileRef::new(
                REPO_UPSTREAM,
                "text_encoder/model-00001-of-00003.safetensors",
            ),
        ),
        (
            role::TEXT_ENCODER_SHARD_2,
            FileRef::new(
                REPO_UPSTREAM,
                "text_encoder/model-00002-of-00003.safetensors",
            ),
        ),
        (
            role::TEXT_ENCODER_SHARD_3,
            FileRef::new(
                REPO_UPSTREAM,
                "text_encoder/model-00003-of-00003.safetensors",
            ),
        ),
        (
            role::TOKENIZER_JSON,
            FileRef::new(REPO_UPSTREAM, "tokenizer/tokenizer.json"),
        ),
        (
            role::TOKENIZER_CONFIG,
            FileRef::new(REPO_UPSTREAM, "tokenizer/tokenizer_config.json"),
        ),
        (
            role::VAE,
            FileRef::new(REPO_UPSTREAM, "vae/diffusion_pytorch_model.safetensors"),
        ),
        (
            role::DIT_GGUF_Q8_0,
            FileRef::new(REPO_GGUF, "z-image-turbo-Q8_0.gguf"),
        ),
        (
            role::DIT_GGUF_Q4_0,
            FileRef::new(REPO_GGUF, "z-image-turbo-Q4_0.gguf"),
        ),
        (
            role::DIT_GGUF_Q4_K_M,
            FileRef::new(REPO_GGUF, "z-image-turbo-Q4_K_M.gguf"),
        ),
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dit_shards_resolve() {
        for r in [role::DIT_SHARD_1, role::DIT_SHARD_2] {
            let f = MANIFEST.get(r).expect(r);
            assert_eq!(f.repo, REPO_DIT_BF16);
            assert!(f.path.ends_with(".safetensors"));
        }
        let idx = MANIFEST.get(role::DIT_INDEX).expect("dit index");
        assert_eq!(idx.repo, REPO_DIT_BF16);
        assert!(idx.path.ends_with(".index.json"));
    }

    #[test]
    fn text_encoder_shards_resolve() {
        for r in [
            role::TEXT_ENCODER_SHARD_1,
            role::TEXT_ENCODER_SHARD_2,
            role::TEXT_ENCODER_SHARD_3,
        ] {
            let f = MANIFEST.get(r).expect(r);
            assert_eq!(f.repo, REPO_UPSTREAM);
            assert!(f.path.ends_with(".safetensors"));
        }
    }

    #[test]
    fn tokenizer_resolves() {
        let t = MANIFEST.get(role::TOKENIZER_JSON).expect("tokenizer");
        assert_eq!(t.repo, REPO_UPSTREAM);
        assert_eq!(t.path, "tokenizer/tokenizer.json");
    }
}
