//! SkyReels-V2-DF-1.3B-540P file manifest. The reference bundle is
//! `Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers` (full fp32 safetensors:
//! transformer + umT5-XXL text encoder + Wan2.1 VAE + tokenizer + scheduler).
//! This is what the e2e pyref consumes, so the safetensors variant is the
//! bit-clean parity path; fp32 narrows to bf16 at GPU upload (residency
//! `read_for_gpu`), same as Z-Image.
//!
//! Quant variants take the COMPLETE DiT and umT5 from DiT-only GGUFs and union
//! them over the safetensors source that still supplies the VAE (the GGUF repos
//! carry no VAE — see the Z-Image GGUF-scope lesson). DiT GGUF from
//! `wsbagnsv1/SkyReels-V2-DF-1.3B-540P-GGUF`, umT5 GGUF from
//! `city96/umt5-xxl-encoder-gguf`. Tensor-name divergence between the GGUFs and
//! the diffusers canonical names is mapped in `wan/source.rs` rename tables.
//!
//! Roles are open-set strings (model-agnostic `ModelManifest`); the `role::*`
//! consts are the typed accessors.

use thinfer_core::manifest::{FileRef, ModelManifest};

const REPO_DIFFUSERS: &str = "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers";
/// DiT-only GGUFs (BF16 / Q8_0 / Q4_K_M / ...), converted from the original-Wan
/// single-file checkpoint. Matmuls quantized; norms/biases F32.
const REPO_DIT_GGUF: &str = "wsbagnsv1/SkyReels-V2-DF-1.3B-540P-GGUF";
/// umT5-XXL encoder-only GGUFs. Shared across the whole Wan family.
const REPO_UMT5_GGUF: &str = "city96/umt5-xxl-encoder-gguf";

pub mod role {
    /// DiT (SkyReelsV2Transformer3DModel), 2 fp32 shards + index.
    pub const DIT_SHARD_1: &str = "dit/shard1";
    pub const DIT_SHARD_2: &str = "dit/shard2";
    pub const DIT_INDEX: &str = "dit/index";
    /// umT5-XXL text encoder, 5 fp32 shards + index.
    pub const TEXT_ENCODER_SHARD_1: &str = "text_encoder/shard1";
    pub const TEXT_ENCODER_SHARD_2: &str = "text_encoder/shard2";
    pub const TEXT_ENCODER_SHARD_3: &str = "text_encoder/shard3";
    pub const TEXT_ENCODER_SHARD_4: &str = "text_encoder/shard4";
    pub const TEXT_ENCODER_SHARD_5: &str = "text_encoder/shard5";
    pub const TEXT_ENCODER_INDEX: &str = "text_encoder/index";
    pub const TOKENIZER_JSON: &str = "tokenizer/tokenizer.json";
    pub const TOKENIZER_SPIECE: &str = "tokenizer/spiece.model";
    pub const TOKENIZER_CONFIG: &str = "tokenizer/tokenizer_config";
    /// Wan2.1 VAE, single fp32 file. Always loaded from safetensors (no GGUF
    /// variant ships it).
    pub const VAE: &str = "vae/decoder";
    pub const VAE_CONFIG: &str = "vae/config";
    pub const SCHEDULER_CONFIG: &str = "scheduler/config";
    /// DiT-only GGUF, Q8_0 (canary tier) and Q4_K_M (default).
    pub const DIT_GGUF_Q8_0: &str = "dit/gguf-q8_0";
    pub const DIT_GGUF_Q4_K_M: &str = "dit/gguf-q4_k_m";
    /// umT5-XXL encoder GGUF, Q8_0.
    pub const UMT5_GGUF_Q8_0: &str = "text_encoder/gguf-q8_0";
}

/// One loadable variant: the file set `WanModel::load` needs. Mirrors
/// `z_image::manifest::VariantFiles`.
pub struct VariantFiles {
    pub id: &'static str,
    /// Safetensors weight shards, in `ShardedSafetensorsSource` order.
    pub weight_roles: &'static [&'static str],
    /// Non-weight files (tokenizer).
    pub aux_roles: &'static [&'static str],
    /// DiT GGUF to union over the safetensors source, if the variant quantizes.
    pub dit_gguf_role: Option<&'static str>,
    /// umT5 GGUF, unioned the same way. `Some` exactly when `dit_gguf_role` is.
    pub umt5_gguf_role: Option<&'static str>,
}

const WEIGHT_ROLES: &[&str] = &[
    role::DIT_SHARD_1,
    role::DIT_SHARD_2,
    role::TEXT_ENCODER_SHARD_1,
    role::TEXT_ENCODER_SHARD_2,
    role::TEXT_ENCODER_SHARD_3,
    role::TEXT_ENCODER_SHARD_4,
    role::TEXT_ENCODER_SHARD_5,
    role::VAE,
];
/// Quant variants take DiT + umT5 from GGUF; only the VAE stays safetensors.
const QUANT_WEIGHT_ROLES: &[&str] = &[role::VAE];
const AUX_ROLES: &[&str] = &[role::TOKENIZER_JSON];

pub static VARIANTS: &[VariantFiles] = &[
    VariantFiles {
        id: "skyreels-v2-df-1.3b",
        weight_roles: WEIGHT_ROLES,
        aux_roles: AUX_ROLES,
        dit_gguf_role: None,
        umt5_gguf_role: None,
    },
    VariantFiles {
        id: "skyreels-v2-df-1.3b-q8",
        weight_roles: QUANT_WEIGHT_ROLES,
        aux_roles: AUX_ROLES,
        dit_gguf_role: Some(role::DIT_GGUF_Q8_0),
        umt5_gguf_role: Some(role::UMT5_GGUF_Q8_0),
    },
    VariantFiles {
        id: "skyreels-v2-df-1.3b-q4",
        weight_roles: QUANT_WEIGHT_ROLES,
        aux_roles: AUX_ROLES,
        dit_gguf_role: Some(role::DIT_GGUF_Q4_K_M),
        umt5_gguf_role: Some(role::UMT5_GGUF_Q8_0),
    },
];

pub fn variant(id: &str) -> Option<&'static VariantFiles> {
    VARIANTS.iter().find(|v| v.id == id)
}

impl VariantFiles {
    pub fn files(&self) -> impl Iterator<Item = (&'static str, &'static FileRef)> + '_ {
        self.weight_roles
            .iter()
            .chain(self.aux_roles.iter())
            .chain(self.dit_gguf_role.iter())
            .chain(self.umt5_gguf_role.iter())
            .map(|r| {
                (
                    *r,
                    MANIFEST.get(r).expect("variant role missing from MANIFEST"),
                )
            })
    }
}

/// Compute recipe. Wan2.1 was trained in bf16; the diffusers reference
/// RNE-quantizes module outputs to bf16, so matching that at inference is
/// required for parity (same rationale as Z-Image). Tune against the e2e.
pub struct WanRecipe {
    pub bf16_quant_writes: bool,
    /// Model's recommended classifier-free-guidance scale (the CLI default when
    /// `--guidance-scale` is omitted). SkyReels-V2-DF is NOT guidance-distilled:
    /// the diffusers `SkyReelsV2DiffusionForcingPipeline` defaults to 6.0 for
    /// T2V (the I2V sibling uses 5.0). `<= 1.0` would disable CFG.
    pub default_guidance_scale: f32,
}

pub static RECIPE: WanRecipe = WanRecipe {
    bf16_quant_writes: true,
    default_guidance_scale: 6.0,
};

pub static MANIFEST: ModelManifest = ModelManifest {
    id: "skyreels-v2-df-1.3b-540p",
    files: &[
        (
            role::DIT_INDEX,
            FileRef::new(
                REPO_DIFFUSERS,
                "transformer/diffusion_pytorch_model.safetensors.index.json",
            ),
        ),
        (
            role::DIT_SHARD_1,
            FileRef::new(
                REPO_DIFFUSERS,
                "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            ),
        ),
        (
            role::DIT_SHARD_2,
            FileRef::new(
                REPO_DIFFUSERS,
                "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            ),
        ),
        (
            role::TEXT_ENCODER_INDEX,
            FileRef::new(REPO_DIFFUSERS, "text_encoder/model.safetensors.index.json"),
        ),
        (
            role::TEXT_ENCODER_SHARD_1,
            FileRef::new(
                REPO_DIFFUSERS,
                "text_encoder/model-00001-of-00005.safetensors",
            ),
        ),
        (
            role::TEXT_ENCODER_SHARD_2,
            FileRef::new(
                REPO_DIFFUSERS,
                "text_encoder/model-00002-of-00005.safetensors",
            ),
        ),
        (
            role::TEXT_ENCODER_SHARD_3,
            FileRef::new(
                REPO_DIFFUSERS,
                "text_encoder/model-00003-of-00005.safetensors",
            ),
        ),
        (
            role::TEXT_ENCODER_SHARD_4,
            FileRef::new(
                REPO_DIFFUSERS,
                "text_encoder/model-00004-of-00005.safetensors",
            ),
        ),
        (
            role::TEXT_ENCODER_SHARD_5,
            FileRef::new(
                REPO_DIFFUSERS,
                "text_encoder/model-00005-of-00005.safetensors",
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
        // GGUF filenames verified against the published repo listings.
        (
            role::DIT_GGUF_Q8_0,
            FileRef::new(REPO_DIT_GGUF, "Skywork-SkyReels-V2-DF-1.3B-540P-Q8_0.gguf"),
        ),
        (
            role::DIT_GGUF_Q4_K_M,
            FileRef::new(
                REPO_DIT_GGUF,
                "Skywork-SkyReels-V2-DF-1.3B-540P-Q4_K_M.gguf",
            ),
        ),
        (
            role::UMT5_GGUF_Q8_0,
            FileRef::new(REPO_UMT5_GGUF, "umt5-xxl-encoder-Q8_0.gguf"),
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
            assert_eq!(
                v.files().count(),
                v.weight_roles.len()
                    + v.aux_roles.len()
                    + usize::from(v.dit_gguf_role.is_some())
                    + usize::from(v.umt5_gguf_role.is_some()),
            );
        }
        assert!(variant("no-such-model").is_none());
    }

    #[test]
    fn safetensors_variant_loads_full_bundle() {
        let v = variant("skyreels-v2-df-1.3b").expect("safetensors variant");
        assert!(v.dit_gguf_role.is_none());
        // 2 DiT + 5 TE + 1 VAE shards.
        assert_eq!(v.weight_roles.len(), 8);
    }
}
