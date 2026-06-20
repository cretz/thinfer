//! FastWan2.2-TI2V-5B-FullAttn file manifest. The reference bundle is
//! `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` (Diffusers safetensors:
//! DMD-distilled Wan2.2-TI2V-5B transformer + umT5-XXL text encoder +
//! Wan2.2-TI2V high-compression VAE + tokenizer + scheduler). This is what the
//! e2e reference consumes, so the safetensors variant is the parity path; fp32/
//! bf16 narrows to bf16 at GPU upload (residency `read_for_gpu`), as Z-Image.
//!
//! Unlike the SkyReels-V2 bundle, the 5B DiT ships as a SINGLE safetensors file
//! (no shard index) and the umT5 text encoder as 3 shards.
//!
//! Quant (GGUF) variants are deferred: bringup is safetensors-only (the
//! bit-clean path). The GGUF-over-safetensors union in `wan/source.rs` is
//! retained for when a published FastWan/Wan2.2-TI2V GGUF is wired.
//!
//! Roles are open-set strings (model-agnostic `ModelManifest`); the `role::*`
//! consts are the typed accessors.

use thinfer_core::manifest::{FileRef, ModelManifest};

const REPO_DIFFUSERS: &str = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers";
/// LightTAE tiny decoder lives in lightx2v's standalone autoencoder repo (NOT
/// the FastVideo bundle); downloaded only when `VaeChoice::Tiny` is selected.
const REPO_LIGHTX2V_AE: &str = "lightx2v/Autoencoders";

pub mod role {
    /// DMD-distilled Wan2.2-TI2V-5B DiT (`WanTransformer3DModel`), single file.
    pub const DIT: &str = "dit/model";
    pub const DIT_CONFIG: &str = "dit/config";
    /// umT5-XXL text encoder, 3 fp32 shards + index.
    pub const TEXT_ENCODER_SHARD_1: &str = "text_encoder/shard1";
    pub const TEXT_ENCODER_SHARD_2: &str = "text_encoder/shard2";
    pub const TEXT_ENCODER_SHARD_3: &str = "text_encoder/shard3";
    pub const TEXT_ENCODER_INDEX: &str = "text_encoder/index";
    pub const TOKENIZER_JSON: &str = "tokenizer/tokenizer.json";
    pub const TOKENIZER_SPIECE: &str = "tokenizer/spiece.model";
    pub const TOKENIZER_CONFIG: &str = "tokenizer/tokenizer_config";
    /// Wan2.2-TI2V high-compression VAE (`AutoencoderKLWan`), single file.
    pub const VAE: &str = "vae/decoder";
    pub const VAE_CONFIG: &str = "vae/config";
    pub const SCHEDULER_CONFIG: &str = "scheduler/config";
    /// LightTAE (`lighttaew2_2`) opt-in tiny decoder, single file.
    pub const TINY_VAE: &str = "vae/tiny";
}

/// One loadable variant: the file set `WanModel::load` needs. Mirrors
/// `z_image::manifest::VariantFiles`.
pub struct VariantFiles {
    pub id: &'static str,
    /// Safetensors weight files, in `ShardedSafetensorsSource` order
    /// (DiT first, then the umT5 shards, then the VAE).
    pub weight_roles: &'static [&'static str],
    /// Non-weight files (tokenizer).
    pub aux_roles: &'static [&'static str],
}

const WEIGHT_ROLES: &[&str] = &[
    role::DIT,
    role::TEXT_ENCODER_SHARD_1,
    role::TEXT_ENCODER_SHARD_2,
    role::TEXT_ENCODER_SHARD_3,
    role::VAE,
];
const AUX_ROLES: &[&str] = &[role::TOKENIZER_JSON];

pub static VARIANTS: &[VariantFiles] = &[VariantFiles {
    id: "fastwan-ti2v-5b",
    weight_roles: WEIGHT_ROLES,
    aux_roles: AUX_ROLES,
}];

pub fn variant(id: &str) -> Option<&'static VariantFiles> {
    VARIANTS.iter().find(|v| v.id == id)
}

impl VariantFiles {
    pub fn files(&self) -> impl Iterator<Item = (&'static str, &'static FileRef)> + '_ {
        self.weight_roles
            .iter()
            .chain(self.aux_roles.iter())
            .map(|r| {
                (
                    *r,
                    MANIFEST.get(r).expect("variant role missing from MANIFEST"),
                )
            })
    }
}

/// Compute recipe. Wan2.2 was trained in bf16; the diffusers reference
/// RNE-quantizes module outputs to bf16, so matching that at inference is
/// required for parity (same rationale as Z-Image). Tune against the e2e.
pub struct WanRecipe {
    pub bf16_quant_writes: bool,
}

pub static RECIPE: WanRecipe = WanRecipe {
    bf16_quant_writes: true,
};

pub static MANIFEST: ModelManifest = ModelManifest {
    id: "fastwan-ti2v-5b-fullattn",
    files: &[
        (
            role::DIT,
            FileRef::new(
                REPO_DIFFUSERS,
                "transformer/diffusion_pytorch_model.safetensors",
            ),
        ),
        (
            role::DIT_CONFIG,
            FileRef::new(REPO_DIFFUSERS, "transformer/config.json"),
        ),
        (
            role::TEXT_ENCODER_INDEX,
            FileRef::new(REPO_DIFFUSERS, "text_encoder/model.safetensors.index.json"),
        ),
        (
            role::TEXT_ENCODER_SHARD_1,
            FileRef::new(
                REPO_DIFFUSERS,
                "text_encoder/model-00001-of-00003.safetensors",
            ),
        ),
        (
            role::TEXT_ENCODER_SHARD_2,
            FileRef::new(
                REPO_DIFFUSERS,
                "text_encoder/model-00002-of-00003.safetensors",
            ),
        ),
        (
            role::TEXT_ENCODER_SHARD_3,
            FileRef::new(
                REPO_DIFFUSERS,
                "text_encoder/model-00003-of-00003.safetensors",
            ),
        ),
        (
            role::TOKENIZER_JSON,
            FileRef::new(REPO_DIFFUSERS, "tokenizer/tokenizer.json"),
        ),
        (
            role::TOKENIZER_SPIECE,
            FileRef::new(REPO_DIFFUSERS, "tokenizer/spiece.model"),
        ),
        (
            role::TOKENIZER_CONFIG,
            FileRef::new(REPO_DIFFUSERS, "tokenizer/tokenizer_config.json"),
        ),
        (
            role::VAE,
            FileRef::new(REPO_DIFFUSERS, "vae/diffusion_pytorch_model.safetensors"),
        ),
        (
            role::VAE_CONFIG,
            FileRef::new(REPO_DIFFUSERS, "vae/config.json"),
        ),
        (
            role::SCHEDULER_CONFIG,
            FileRef::new(REPO_DIFFUSERS, "scheduler/scheduler_config.json"),
        ),
        (
            role::TINY_VAE,
            FileRef::new(REPO_LIGHTX2V_AE, "lighttaew2_2.safetensors"),
        ),
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variants_resolve() {
        for v in VARIANTS {
            assert_eq!(variant(v.id).map(|x| x.id), Some(v.id));
            assert_eq!(v.files().count(), v.weight_roles.len() + v.aux_roles.len());
        }
        assert!(variant("no-such-model").is_none());
    }

    #[test]
    fn safetensors_variant_loads_full_bundle() {
        let v = variant("fastwan-ti2v-5b").expect("safetensors variant");
        // 1 DiT + 3 TE + 1 VAE.
        assert_eq!(v.weight_roles.len(), 5);
    }
}
