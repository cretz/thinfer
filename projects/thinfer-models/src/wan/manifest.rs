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
/// LongLive-2.0-5B causal/AR DiT: a single 10GB `torch.save` `.pt` (the complete
/// merged generator). umT5 + VAE + tokenizer reuse the FastVideo base bundle, so
/// only this one file is LongLive-specific.
const REPO_LONGLIVE: &str = "Efficient-Large-Model/LongLive-2.0-5B";
/// Wan2.2-T2V-A14B GGUF bundle: both MoE experts (high/low noise) in every quant,
/// plus the Wan2.1 VAE. We consume Q5_K_M (the user-requested footprint tier;
/// blocks stream per-block so VRAM is bounded by activations, not quant size).
const REPO_WAN22_GGUF: &str = "QuantStack/Wan2.2-T2V-A14B-GGUF";
/// LightX2V step-distill LoRAs (one per expert, rank 64, 4-step CFG-free).
const REPO_WAN22_LORA: &str = "lightx2v/Wan2.2-Distill-Loras";
/// Wan2.2-A14B diffusers bundle: source of the Wan2.1 VAE in DIFFUSERS tensor
/// naming (`decoder.up_blocks.*`). The QuantStack GGUF repo also ships a
/// `Wan2.1_VAE.safetensors`, but in ORIGINAL-Wan naming (`decoder.middle.*`,
/// `decoder.conv1`), which the engine's diffusers-named VAE loader can't read.
const REPO_WAN22_DIFFUSERS: &str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers";

pub mod role {
    /// DMD-distilled Wan2.2-TI2V-5B DiT (`WanTransformer3DModel`), single file.
    pub const DIT: &str = "dit/model";
    /// LongLive-2.0-5B causal DiT, single `torch.save` `.pt` (runtime-read via
    /// `PytorchSource`; not safetensors).
    pub const LONGLIVE_DIT: &str = "dit/longlive_pt";
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
    /// Wan2.1 VAE (`AutoencoderKLWan`, z16 4x8x8), used by the Wan2.2-A14B model.
    pub const VAE_WAN21: &str = "vae/wan21";
    /// Wan2.2-A14B MoE experts (GGUF, original-Wan names) + LightX2V distill LoRAs.
    pub const DIT_HIGH_NOISE: &str = "dit/wan22_high";
    pub const DIT_LOW_NOISE: &str = "dit/wan22_low";
    pub const LORA_HIGH_NOISE: &str = "lora/wan22_high";
    pub const LORA_LOW_NOISE: &str = "lora/wan22_low";
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
    /// `Some` when the DiT is a standalone `.pt` (LongLive) rather than the
    /// leading safetensors shard: then `weight_roles` carries only the umT5 +
    /// VAE safetensors and this role is the `.pt` consumed by
    /// `open_longlive_source`. `None` for the all-safetensors FastWan bundle.
    pub dit_pt_role: Option<&'static str>,
    /// Wan2.2-A14B MoE: the two expert GGUFs + their LightX2V LoRAs (consumed by
    /// `open_wan22_source`). When set, `weight_roles` carries only the safetensors
    /// TAIL (umT5 shards + Wan2.1 VAE). `None` for single-DiT variants.
    pub gguf_high_role: Option<&'static str>,
    pub gguf_low_role: Option<&'static str>,
    pub lora_high_role: Option<&'static str>,
    pub lora_low_role: Option<&'static str>,
}

const WEIGHT_ROLES: &[&str] = &[
    role::DIT,
    role::TEXT_ENCODER_SHARD_1,
    role::TEXT_ENCODER_SHARD_2,
    role::TEXT_ENCODER_SHARD_3,
    role::VAE,
];
const AUX_ROLES: &[&str] = &[role::TOKENIZER_JSON];

/// LongLive shares the FastVideo umT5 + VAE (the Wan2.2 base components); only
/// the DiT differs (the causal `.pt`), carried in `dit_pt_role`.
const LONGLIVE_WEIGHT_ROLES: &[&str] = &[
    role::TEXT_ENCODER_SHARD_1,
    role::TEXT_ENCODER_SHARD_2,
    role::TEXT_ENCODER_SHARD_3,
    role::VAE,
];

/// Wan2.2-A14B safetensors tail: umT5 (reused from the FastWan bundle) + the
/// Wan2.1 VAE. The two expert DiTs + LoRAs come via the gguf/lora roles.
const WAN22_TAIL_ROLES: &[&str] = &[
    role::TEXT_ENCODER_SHARD_1,
    role::TEXT_ENCODER_SHARD_2,
    role::TEXT_ENCODER_SHARD_3,
    role::VAE_WAN21,
];

pub static VARIANTS: &[VariantFiles] = &[
    VariantFiles {
        id: "fastwan-ti2v-5b",
        weight_roles: WEIGHT_ROLES,
        aux_roles: AUX_ROLES,
        dit_pt_role: None,
        gguf_high_role: None,
        gguf_low_role: None,
        lora_high_role: None,
        lora_low_role: None,
    },
    VariantFiles {
        id: "longlive-2.0-5b",
        weight_roles: LONGLIVE_WEIGHT_ROLES,
        aux_roles: AUX_ROLES,
        dit_pt_role: Some(role::LONGLIVE_DIT),
        gguf_high_role: None,
        gguf_low_role: None,
        lora_high_role: None,
        lora_low_role: None,
    },
    VariantFiles {
        id: "wan2.2-t2v-a14b",
        weight_roles: WAN22_TAIL_ROLES,
        aux_roles: AUX_ROLES,
        dit_pt_role: None,
        gguf_high_role: Some(role::DIT_HIGH_NOISE),
        gguf_low_role: Some(role::DIT_LOW_NOISE),
        lora_high_role: Some(role::LORA_HIGH_NOISE),
        lora_low_role: Some(role::LORA_LOW_NOISE),
    },
];

pub fn variant(id: &str) -> Option<&'static VariantFiles> {
    VARIANTS.iter().find(|v| v.id == id)
}

impl VariantFiles {
    pub fn files(&self) -> impl Iterator<Item = (&'static str, &'static FileRef)> + '_ {
        // The standalone DiT `.pt` (when present) is part of the download set too,
        // so resolution/caching pulls it alongside the safetensors + aux files.
        self.dit_pt_role
            .iter()
            .chain(self.gguf_high_role.iter())
            .chain(self.gguf_low_role.iter())
            .chain(self.lora_high_role.iter())
            .chain(self.lora_low_role.iter())
            .chain(self.weight_roles.iter())
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
            role::LONGLIVE_DIT,
            FileRef::new(REPO_LONGLIVE, "model_bf16.pt"),
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
        // --- Wan2.2-T2V-A14B (MoE) ---
        (
            role::DIT_HIGH_NOISE,
            FileRef::new(
                REPO_WAN22_GGUF,
                "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q5_K_M.gguf",
            ),
        ),
        (
            role::DIT_LOW_NOISE,
            FileRef::new(
                REPO_WAN22_GGUF,
                "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q5_K_M.gguf",
            ),
        ),
        (
            role::VAE_WAN21,
            FileRef::new(
                REPO_WAN22_DIFFUSERS,
                "vae/diffusion_pytorch_model.safetensors",
            ),
        ),
        (
            role::LORA_HIGH_NOISE,
            FileRef::new(
                REPO_WAN22_LORA,
                "wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors",
            ),
        ),
        (
            role::LORA_LOW_NOISE,
            FileRef::new(
                REPO_WAN22_LORA,
                "wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors",
            ),
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
            let extra = usize::from(v.dit_pt_role.is_some())
                + usize::from(v.gguf_high_role.is_some())
                + usize::from(v.gguf_low_role.is_some())
                + usize::from(v.lora_high_role.is_some())
                + usize::from(v.lora_low_role.is_some());
            // `files()` also panics if any declared role is missing from MANIFEST.
            assert_eq!(
                v.files().count(),
                extra + v.weight_roles.len() + v.aux_roles.len()
            );
        }
        assert!(variant("no-such-model").is_none());
    }

    #[test]
    fn safetensors_variant_loads_full_bundle() {
        let v = variant("fastwan-ti2v-5b").expect("safetensors variant");
        // 1 DiT + 3 TE + 1 VAE.
        assert_eq!(v.weight_roles.len(), 5);
        assert!(v.dit_pt_role.is_none());
    }

    #[test]
    fn longlive_variant_separates_pt_dit_from_base() {
        let v = variant("longlive-2.0-5b").expect("longlive variant");
        // DiT is the standalone `.pt`; safetensors side is umT5 (3) + VAE (1).
        assert_eq!(v.dit_pt_role, Some(role::LONGLIVE_DIT));
        assert_eq!(v.weight_roles.len(), 4);
        assert!(!v.weight_roles.contains(&role::DIT));
        // The `.pt` is in the download set and points at the LongLive repo.
        let pt = v
            .files()
            .find(|(r, _)| *r == role::LONGLIVE_DIT)
            .expect("pt in file set");
        assert!(pt.1.path.ends_with("model_bf16.pt"));
    }
}
