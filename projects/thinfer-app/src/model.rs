//! The engine side of the model ids ([`thinfer_client::model`]): manifests,
//! file roles and pipeline configs, as one extension trait per id type.

pub use thinfer_client::model::*;
use thinfer_core::manifest::ModelManifest;
use thinfer_models::hunyuan::manifest as hunmf;
use thinfer_models::ideogram4::manifest as idmf;
use thinfer_models::krea::manifest as kreamf;
use thinfer_models::llm::manifest as llmmf;
use thinfer_models::ltx::manifest as ltxmf;
use thinfer_models::minimax::manifest as mmmf;
use thinfer_models::moss_sfx::manifest as sfxmf;
use thinfer_models::qwen_image::manifest as qimf;
use thinfer_models::qwen_image21::manifest as qi21mf;
use thinfer_models::qwen3_tts::manifest as ttsmf;
use thinfer_models::wan::manifest as wanmf;
use thinfer_models::z_image::manifest as zmf;

/// An image id's engine side: its manifest and files.
pub trait ImageModelEngine {
    fn manifest(self) -> &'static ModelManifest;
    /// File set from the shared Z-Image variant registry (keyed by `Display`).
    /// Z-Image only: Ideogram-4 sources its files by role (see
    /// [`Self::required_roles`]); call sites must branch on [`ImageModelId::kind`].
    fn variant(self) -> &'static zmf::VariantFiles;
    /// For non-registry models (Ideogram-4), the manifest roles a generate
    /// needs. Empty for Z-Image (it uses the variant registry instead).
    fn required_roles(self) -> &'static [&'static str];
}

impl ImageModelEngine for ImageModelId {
    fn manifest(self) -> &'static ModelManifest {
        match self.kind() {
            ImageKind::ZImage => &zmf::MANIFEST,
            ImageKind::Ideogram4 => &idmf::MANIFEST,
            ImageKind::QwenImageEdit | ImageKind::QwenImage => &qimf::MANIFEST,
            ImageKind::Krea2 => &kreamf::MANIFEST,
            ImageKind::QwenImage21 => &qi21mf::MANIFEST,
        }
    }

    fn variant(self) -> &'static zmf::VariantFiles {
        debug_assert_eq!(self.kind(), ImageKind::ZImage, "variant() is Z-Image only");
        zmf::variant(&self.to_string()).expect("ImageModelId missing from VARIANTS registry")
    }

    fn required_roles(self) -> &'static [&'static str] {
        match self {
            ImageModelId::Ideogram4Q8 => idmf::RUNTIME_ROLES_Q8,
            // Q8_0 DiT for now (Q4_K_M streaming is a later perf task).
            ImageModelId::QwenImageEditRapid => qimf::RUNTIME_ROLES_Q8,
            // t2i omits the vision tower (mmproj) + preprocessor.
            ImageModelId::QwenImageRapid => qimf::RUNTIME_ROLES_T2I_Q8,
            // Krea sources by role (Q8_0 canary DiT + Qwen3-VL encoder + VAE).
            ImageModelId::Krea2Turbo => kreamf::RUNTIME_ROLES_Q8,
            ImageModelId::QwenImage21Turbo => qi21mf::RUNTIME_ROLES_Q4,
            // Z-Image uses the variant registry, not roles.
            _ => &[],
        }
    }
}

/// A video id's engine side: manifests, file roles, DiT configs.
pub trait VideoModelEngine {
    fn manifest(self) -> &'static ModelManifest;
    /// LTX DiT GGUF role for this variant: Q8_0 (quality baseline) or Q4_K_M
    /// (footprint). Panics for non-LTX -- guard with [`VideoModelId::is_ltx`].
    fn ltx_dit_role(self) -> &'static str;
    /// LTX text-encoder GGUF role for an [`EncoderQuant`] choice: the Q8_0
    /// baseline (uniform per-site, the conditioning-quality default) or the Q4_K_M
    /// variant (7.3G vs 12.5G; ~2.8x faster encode, lower-precision conditioning).
    /// The encoder's per-site dequant reads the Q4_K/Q6_K mix directly. Shared by
    /// every LTX variant (the encoder is common to all).
    fn ltx_encoder_role(self, encoder: EncoderQuant) -> &'static str;
    /// The distill-LoRA stack to fold into the `dev` DiT, as `(role, strength)`
    /// pairs accumulated in order (see `ltx::lora`). Default = the single distill
    /// slot at 1.0. `THINFER_LTX_DISTILL=stack` -> the distilled ComfyUI workflow
    /// recipe: the distill slot @ 0.7 + official `distilled-lora-384 @ 0.5` (the
    /// distill applied at <1.0 may cut the illustrative skew of the 1.0
    /// single-fold). Dev-only.
    fn distill_stack(self) -> Vec<(&'static str, f32)>;
    /// LTX joint-AV runtime role list for this variant (Q8_0 vs Q4_K_M DiT).
    fn ltx_runtime_roles(self) -> &'static [&'static str];
    /// Wan variant-registry entry. Wan models only (LTX sources by role; see
    /// [`VideoModelId::is_ltx`]). Panics for LTX -- guard call sites with `is_ltx`.
    fn variant(self) -> &'static wanmf::VariantFiles;
    /// The roles a MiniMax-family t2va generate downloads and opens.
    fn minimax_roles(self) -> &'static [&'static str];
    /// The packed-row limit of a MiniMax-family run: FastH3's NVFP4 block or
    /// the GGUF block when `accelerated`, else the backend block's.
    fn minimax_row_limit(self, accelerated: bool) -> usize;
    /// The DiT a MiniMax-family id runs.
    fn minimax_dit_config(self) -> thinfer_models::minimax::DitConfig;
    /// The Wan-manifest tiny-decoder role a `VaeChoice::Tiny` request pulls for
    /// this model: the tiny decoder must pair with the variant's FULL VAE
    /// latent space (`lighttaew2_2` for the Wan2.2 z48 line, `taew2_1` for the
    /// Wan2.1 z16 AnyFlow). Wan models only; others resolve their VAE roles
    /// through their own manifests.
    fn wan_tiny_vae_role(self) -> &'static str;
}

impl VideoModelEngine for VideoModelId {
    fn manifest(self) -> &'static ModelManifest {
        match self {
            VideoModelId::Ltx23Distilled | VideoModelId::Ltx23DistilledQ4 => &ltxmf::MANIFEST,
            VideoModelId::Ltx23Dev | VideoModelId::Ltx23DevQ4 => &ltxmf::DEV_MANIFEST,
            VideoModelId::Ltx2Rapid => &ltxmf::LTX2_RAPID_MANIFEST,
            VideoModelId::Hunyuan15T2v | VideoModelId::Hunyuan15I2v => &hunmf::MANIFEST,
            VideoModelId::MinimaxH3
            | VideoModelId::MinimaxH3Turbo
            | VideoModelId::MinimaxH3TurboRef
            | VideoModelId::Fasth3Trim
            | VideoModelId::Fasth3V2
            | VideoModelId::Fasth3Omni
            | VideoModelId::HyperflowFlash
            | VideoModelId::HyperflowFlashRef => &mmmf::MANIFEST,
            _ => &wanmf::MANIFEST,
        }
    }

    fn ltx_dit_role(self) -> &'static str {
        match self {
            VideoModelId::Ltx23DistilledQ4 | VideoModelId::Ltx23DevQ4 => {
                ltxmf::role::DIT_GGUF_Q4_K_M
            }
            VideoModelId::Ltx23Distilled | VideoModelId::Ltx23Dev => ltxmf::role::DIT_GGUF_Q8_0,
            VideoModelId::Ltx2Rapid => ltxmf::role::DIT_GGUF_Q5_K_M,
            other => panic!("ltx_dit_role on non-LTX model {other}"),
        }
    }

    fn ltx_encoder_role(self, encoder: EncoderQuant) -> &'static str {
        debug_assert!(self.is_ltx());
        match encoder {
            EncoderQuant::Q4 => ltxmf::role::ENCODER_GGUF_Q4,
            EncoderQuant::Q8 => ltxmf::role::ENCODER_GGUF,
        }
    }

    fn distill_stack(self) -> Vec<(&'static str, f32)> {
        debug_assert!(self.folds_distill_lora());
        match std::env::var("THINFER_LTX_DISTILL").ok().as_deref() {
            Some("stack") => vec![
                (ltxmf::role::DISTILL_LORA, 0.7),
                (ltxmf::role::DISTILL_LORA_384, 0.5),
            ],
            _ => vec![(ltxmf::role::DISTILL_LORA, 1.0)],
        }
    }

    fn ltx_runtime_roles(self) -> &'static [&'static str] {
        match self {
            VideoModelId::Ltx23DistilledQ4 | VideoModelId::Ltx23DevQ4 => ltxmf::RUNTIME_ROLES_AV_Q4,
            VideoModelId::Ltx2Rapid => ltxmf::RUNTIME_ROLES_RAPID,
            _ => ltxmf::RUNTIME_ROLES_AV_Q8,
        }
    }

    fn variant(self) -> &'static wanmf::VariantFiles {
        debug_assert!(!self.is_ltx(), "variant() is Wan-only; LTX sources by role");
        wanmf::variant(&self.to_string()).expect("VideoModelId missing from VARIANTS registry")
    }

    fn minimax_roles(self) -> &'static [&'static str] {
        use thinfer_models::minimax::manifest as m;
        match self {
            VideoModelId::Fasth3Trim => m::RUNTIME_ROLES_T2VA_FASTH3_TRIM,
            VideoModelId::Fasth3V2 => m::RUNTIME_ROLES_T2VA_FASTH3_V2,
            VideoModelId::Fasth3Omni => m::RUNTIME_ROLES_REF2VA_FASTH3_OMNI,
            VideoModelId::HyperflowFlash => m::RUNTIME_ROLES_T2VA_HYPERFLOW,
            VideoModelId::HyperflowFlashRef => m::RUNTIME_ROLES_REF2VA_HYPERFLOW,
            VideoModelId::MinimaxH3Turbo => m::RUNTIME_ROLES_T2VA_TURBO_V4,
            VideoModelId::MinimaxH3TurboRef => m::RUNTIME_ROLES_REF2VA_TURBO_V4,
            _ => m::RUNTIME_ROLES_T2VA_Q4,
        }
    }

    fn minimax_row_limit(self, accelerated: bool) -> usize {
        use thinfer_models::minimax::pipeline::{BlockPath, max_packed_rows};
        max_packed_rows(match (accelerated, self.is_fasth3()) {
            (false, _) => BlockPath::Backend,
            (true, true) => BlockPath::Fasth3Accel,
            (true, false) => BlockPath::GenericAccel,
        })
    }

    fn minimax_dit_config(self) -> thinfer_models::minimax::DitConfig {
        use thinfer_models::minimax::DitConfig;
        match self {
            VideoModelId::Fasth3Trim => DitConfig::fasth3_trim(),
            VideoModelId::Fasth3V2 => DitConfig::fasth3_v2(),
            VideoModelId::Fasth3Omni => DitConfig::fasth3_omni(),
            VideoModelId::HyperflowFlash => DitConfig::hyperflow(47),
            VideoModelId::HyperflowFlashRef => DitConfig::hyperflow(49),
            _ => DitConfig::h3_33b(),
        }
    }

    fn wan_tiny_vae_role(self) -> &'static str {
        use thinfer_models::wan::manifest::role;
        match self {
            VideoModelId::AnyflowT2v14b
            | VideoModelId::Fastwan21T2v13b
            | VideoModelId::Wan21T2v13b => role::TINY_VAE_WAN21,
            _ => role::TINY_VAE,
        }
    }
}

/// A VAE choice's engine side.
pub trait VaeChoiceEngine {
    /// The MiniMax-H3 NVFP4 decoder this selects, as (manifest role, decoder
    /// layers). `None` = the H3 family's own fp16 decoder.
    fn minimax_nvfp4(self) -> Option<(&'static str, usize)>;
    /// The Wan pipeline's decoder choice.
    fn engine(self) -> thinfer_models::wan::pipeline::VaeChoice;
}

impl VaeChoiceEngine for VaeChoice {
    fn minimax_nvfp4(self) -> Option<(&'static str, usize)> {
        use thinfer_models::minimax::manifest::role;
        match self {
            VaeChoice::Nvfp4 => Some((role::VAE_DECODER_NVFP4, 36)),
            VaeChoice::Nvfp4Fast => Some((role::VAE_DECODER_NVFP4_FAST, 8)),
            VaeChoice::Full | VaeChoice::Tiny | VaeChoice::TinyFt => None,
        }
    }

    fn engine(self) -> thinfer_models::wan::pipeline::VaeChoice {
        use thinfer_models::wan::pipeline::VaeChoice as E;
        match self {
            // The NVFP4 decoders are MiniMax-H3 only and rejected for every
            // other model at request validation.
            VaeChoice::Full | VaeChoice::Nvfp4 | VaeChoice::Nvfp4Fast => E::Full,
            // Wan has no fine-tuned tiny decoder; a `tiny-ft` request on a Wan
            // model degrades to its own tiny (the ft is a Hunyuan-only option,
            // and the web only offers it for Hunyuan).
            VaeChoice::Tiny | VaeChoice::TinyFt => E::Tiny,
        }
    }
}

/// A rewrite quality's engine side: the rewriter LM it runs.
pub trait RewriteQualityEngine {
    /// The Hunyuan manifest role for this quality's rewriter GGUF.
    fn gguf_role(self) -> &'static str;
    /// The `qwen3_lm` architecture config for this quality's GGUF.
    fn lm_config(self) -> thinfer_models::qwen3_lm::Qwen3LmConfig;
}

impl RewriteQualityEngine for RewriteQuality {
    fn gguf_role(self) -> &'static str {
        use thinfer_models::hunyuan::manifest::role;
        match self {
            RewriteQuality::Fast => role::REWRITER_GGUF_4B_Q5_K_M,
            RewriteQuality::Full => role::REWRITER_GGUF_8B_Q5_K_M,
        }
    }

    fn lm_config(self) -> thinfer_models::qwen3_lm::Qwen3LmConfig {
        use thinfer_models::qwen3_lm::Qwen3LmConfig;
        match self {
            RewriteQuality::Fast => Qwen3LmConfig::qwen3_vl_4b(),
            RewriteQuality::Full => Qwen3LmConfig::qwen3_vl_8b(),
        }
    }
}

/// A sampler choice's engine side.
pub trait VideoSamplerEngine {
    /// Build the engine sampler, folding in the user step count (UniPC only; DMD
    /// bakes its own 3-step schedule and ignores `steps`).
    fn into_engine(self, steps: u32) -> thinfer_models::wan::pipeline::VideoSampler;
}

impl VideoSamplerEngine for VideoSampler {
    fn into_engine(self, steps: u32) -> thinfer_models::wan::pipeline::VideoSampler {
        use thinfer_models::wan::pipeline::VideoSampler as E;
        match self {
            VideoSampler::UniPc => E::UniPc { steps },
            VideoSampler::Dmd => E::Dmd,
        }
    }
}

/// A swap model's engine side: its swapper and files.
pub trait SwapModelEngine {
    /// The per-frame swapper network; `None` for DreamID-V.
    fn swapper(self) -> Option<thinfer_models::faceswap::SwapperKind>;
    /// The per-frame swapper's weight file; `None` for DreamID-V (its files
    /// come from [`Self::dreamidv_files`]).
    fn file(self) -> Option<thinfer_core::manifest::FileRef>;
    /// DreamID-V's whole fetch set: the DiT `.pth`, the Wan2.1 VAE and the two
    /// DWPose ONNX nets (the Wan manifest's `dreamid-v` variant).
    fn dreamidv_files() -> Vec<thinfer_core::manifest::FileRef>;
}

impl SwapModelEngine for SwapModel {
    fn swapper(self) -> Option<thinfer_models::faceswap::SwapperKind> {
        use thinfer_models::faceswap::SwapperKind;
        match self {
            SwapModel::Hyperswap1a | SwapModel::Hyperswap1b | SwapModel::Hyperswap1c => {
                Some(SwapperKind::HyperSwap)
            }
            SwapModel::AlphaFace => Some(SwapperKind::AlphaFace),
            SwapModel::DreamIdV => None,
        }
    }

    fn file(self) -> Option<thinfer_core::manifest::FileRef> {
        let (repo, path) = match self {
            SwapModel::Hyperswap1a => ("facefusion/models-3.3.0", "hyperswap_1a_256.onnx"),
            SwapModel::Hyperswap1b => ("facefusion/models-3.3.0", "hyperswap_1b_256.onnx"),
            SwapModel::Hyperswap1c => ("facefusion/models-3.3.0", "hyperswap_1c_256.onnx"),
            SwapModel::AlphaFace => ("facefusion/models-3.9.0", "alphaface_256.onnx"),
            SwapModel::DreamIdV => return None,
        };
        Some(thinfer_core::manifest::FileRef::new(repo, path))
    }

    fn dreamidv_files() -> Vec<thinfer_core::manifest::FileRef> {
        wanmf::variant("dreamid-v")
            .expect("dreamid-v missing from the Wan VARIANTS registry")
            .files()
            .map(|(_, f)| *f)
            .collect()
    }
}

/// A text id's engine side: its manifest and files.
pub trait TextModelEngine {
    fn manifest(self) -> &'static ModelManifest;
    fn required_roles(self) -> &'static [&'static str];
}

impl TextModelEngine for TextModelId {
    fn manifest(self) -> &'static ModelManifest {
        match self {
            Self::Qwen36_35bA3b => &llmmf::QWEN36_MANIFEST,
            Self::Gemma4_26bA4b => &llmmf::GEMMA4_MANIFEST,
        }
    }

    fn required_roles(self) -> &'static [&'static str] {
        llmmf::RUNTIME_ROLES
    }
}

/// An audio id's engine side: its manifest and files.
pub trait AudioModelEngine {
    fn manifest(self) -> &'static ModelManifest;
    fn required_roles(self) -> &'static [&'static str];
}

impl AudioModelEngine for AudioModelId {
    fn manifest(self) -> &'static ModelManifest {
        match self {
            Self::Qwen3Tts => &ttsmf::BASE_MANIFEST,
            Self::Qwen3TtsDesign => &ttsmf::DESIGN_MANIFEST,
            Self::MossSfx => &sfxmf::MANIFEST,
        }
    }

    fn required_roles(self) -> &'static [&'static str] {
        match self {
            Self::Qwen3Tts | Self::Qwen3TtsDesign => ttsmf::RUNTIME_ROLES,
            Self::MossSfx => sfxmf::RUNTIME_ROLES,
        }
    }
}

/// The SCRFD detector + ArcFace embedder shared by every face-swap run.
pub const SCRFD: thinfer_core::manifest::FileRef =
    thinfer_core::manifest::FileRef::new("deepghs/insightface", "buffalo_s/det_500m.onnx");
pub const ARCFACE: thinfer_core::manifest::FileRef =
    thinfer_core::manifest::FileRef::new("facefusion/models-3.0.0", "arcface_w600k_r50.onnx");
/// XSeg occlusion mask, pulled in only when `--occlusion-mask` is set.
pub const XSEG: thinfer_core::manifest::FileRef =
    thinfer_core::manifest::FileRef::new("facefusion/models-3.1.0", "xseg_1.onnx");
/// GFPGAN 1.4 face enhancer, pulled in only when `--enhance` is set.
pub const GFPGAN: thinfer_core::manifest::FileRef =
    thinfer_core::manifest::FileRef::new("facefusion/models-3.0.0", "gfpgan_1.4.onnx");

#[cfg(test)]
mod tests {
    #[test]
    fn hyperflow_steps_match_the_engine() {
        assert_eq!(
            super::HYPERFLOW_DEFAULT_STEPS as usize,
            thinfer_models::minimax::hyperflow::DEFAULT_STEPS
        );
    }
}
