//! Canonical model identifiers and their per-model defaults / grids. Single
//! source of truth shared by every front end: the CLI derives clap `ValueEnum`
//! off these (the `cli` feature), and the values feed the variant registry in
//! `thinfer-models`. `Display` is the canonical id string (matches the clap
//! value names and the registry keys); the registry lookups key off it.

use thinfer_core::manifest::ModelManifest;
use thinfer_models::ideogram4::manifest as idmf;
use thinfer_models::qwen_image::manifest as qimf;
use thinfer_models::wan::manifest as wanmf;
use thinfer_models::z_image::manifest as zmf;

// Image defaults follow upstream Z-Image (8 Turbo steps, CFG-free) except dims:
// upstream assumes datacenter GPUs at 1024x1024; we default to 768x768, the
// thin-hardware sweet spot every parity/perf baseline uses. Defined as consts so
// clap `default_value_t` and [`ImageModelId::defaults`] read the same numbers.
pub const IMAGE_DEFAULT_WIDTH: u32 = 768;
pub const IMAGE_DEFAULT_HEIGHT: u32 = 768;
pub const IMAGE_DEFAULT_STEPS: u32 = 8;

// Video defaults: 960x544 (the FastWan e2e baseline). Frame count + fps are
// model-derived (see [`VideoModelId`]).
pub const VIDEO_DEFAULT_WIDTH: u32 = 960;
pub const VIDEO_DEFAULT_HEIGHT: u32 = 544;

// Default UniPC denoise steps for FastWan (the served sampler). 4 matches the
// public HF Spaces (slider 1..=8). DMD ignores this (its schedule is fixed at 3).
pub const VIDEO_DEFAULT_STEPS: u32 = 4;
/// Upper bound on the UniPC step slider (the Spaces cap at 8).
pub const VIDEO_MAX_STEPS: u32 = 8;

/// Resolved per-model image defaults (the registry accessor `thinfer-serve` and
/// the CLI both read).
#[derive(Clone, Copy, Debug)]
pub struct ImageDefaults {
    pub width: u32,
    pub height: u32,
    pub steps: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
pub enum ImageModelId {
    /// Z-Image-Turbo, Q8_0 DiT matmul weights (unsloth GGUF); rest bf16.
    #[cfg_attr(feature = "cli", value(name = "zimage-turbo-q8"))]
    #[cfg_attr(feature = "serde", serde(rename = "zimage-turbo-q8"))]
    ZImageTurboQ8,
    /// Z-Image-Turbo, Q4_K_M DiT matmul weights; halves DiT VRAM/bandwidth vs
    /// Q8_0 at production quality. The default.
    #[cfg_attr(feature = "cli", value(name = "zimage-turbo-q4"))]
    #[cfg_attr(feature = "serde", serde(rename = "zimage-turbo-q4"))]
    ZImageTurboQ4,
    /// Z-Image-Turbo, bf16 DiT weights (dimitribarbot safetensors).
    #[cfg_attr(feature = "cli", value(name = "zimage-turbo-bf16"))]
    #[cfg_attr(feature = "serde", serde(rename = "zimage-turbo-bf16"))]
    ZImageTurboBf16,
    /// Ideogram-4 + the ostris turbotime LoRA (CFG-free): Q8_0 encoder + DiT,
    /// LoRA folded to Q8_0 at load (near-lossless); FLUX.2 KL VAE. (A Q4_K DiT
    /// default was tried and dropped: per-request re-fold to Q4_K was ~2x slower
    /// than Q8_0 with worse quality -- not worth it for this fold-per-request
    /// pipeline.)
    #[cfg_attr(feature = "cli", value(name = "ideogram4-q8"))]
    #[cfg_attr(feature = "serde", serde(rename = "ideogram4-q8"))]
    Ideogram4Q8,
    /// Qwen-Image-Edit-Rapid-AIO: a 4-step distilled, CFG-free image-EDIT MMDiT.
    /// Requires `--input-image` (the reference image to edit). Q8_0 DiT for now
    /// (Q4_K_M streaming is a later perf task).
    #[cfg_attr(feature = "cli", value(name = "qwen-image-edit-rapid"))]
    #[cfg_attr(feature = "serde", serde(rename = "qwen-image-edit-rapid"))]
    QwenImageEditRapid,
    /// Qwen-Image-Rapid: the same 4-step distilled, CFG-free MMDiT as the edit
    /// model, driven text-to-image (no reference image, no vision tower).
    #[cfg_attr(feature = "cli", value(name = "qwen-image-rapid"))]
    #[cfg_attr(feature = "serde", serde(rename = "qwen-image-rapid"))]
    QwenImageRapid,
}

/// Which engine pipeline an image id drives. The executor branches on this:
/// Z-Image and Ideogram-4 have different sources, tokenizers, and pipelines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageKind {
    ZImage,
    Ideogram4,
    /// Qwen-Image-Edit (image->image; requires a reference image).
    QwenImageEdit,
    /// Qwen-Image (text->image; same MMDiT, no reference image).
    QwenImage,
}

impl ImageModelId {
    pub const DEFAULT: ImageModelId = ImageModelId::ZImageTurboQ4;

    /// Which engine pipeline this id drives.
    pub fn kind(self) -> ImageKind {
        match self {
            ImageModelId::Ideogram4Q8 => ImageKind::Ideogram4,
            ImageModelId::QwenImageEditRapid => ImageKind::QwenImageEdit,
            ImageModelId::QwenImageRapid => ImageKind::QwenImage,
            _ => ImageKind::ZImage,
        }
    }

    pub fn manifest(self) -> &'static ModelManifest {
        match self.kind() {
            ImageKind::ZImage => &zmf::MANIFEST,
            ImageKind::Ideogram4 => &idmf::MANIFEST,
            ImageKind::QwenImageEdit | ImageKind::QwenImage => &qimf::MANIFEST,
        }
    }

    /// File set from the shared Z-Image variant registry (keyed by `Display`).
    /// Z-Image only: Ideogram-4 sources its files by role (see
    /// [`Self::required_roles`]); call sites must branch on [`Self::kind`].
    pub fn variant(self) -> &'static zmf::VariantFiles {
        debug_assert_eq!(self.kind(), ImageKind::ZImage, "variant() is Z-Image only");
        zmf::variant(&self.to_string()).expect("ImageModelId missing from VARIANTS registry")
    }

    /// For non-registry models (Ideogram-4), the manifest roles a generate
    /// needs. Empty for Z-Image (it uses the variant registry instead).
    pub fn required_roles(self) -> &'static [&'static str] {
        match self {
            ImageModelId::Ideogram4Q8 => idmf::RUNTIME_ROLES_Q8,
            // Q8_0 DiT for now (Q4_K_M streaming is a later perf task).
            ImageModelId::QwenImageEditRapid => qimf::RUNTIME_ROLES_Q8,
            // t2i omits the vision tower (mmproj) + preprocessor.
            ImageModelId::QwenImageRapid => qimf::RUNTIME_ROLES_T2I_Q8,
            // Z-Image uses the variant registry, not roles.
            _ => &[],
        }
    }

    pub fn defaults(self) -> ImageDefaults {
        match self.kind() {
            // Ideogram-4 trains at 1024 (resolution-aware schedule). The
            // turbotime LoRA is a few-step distill (2/4/8); 4 is the balance
            // default (8 = quality ceiling, 2 = fastest).
            ImageKind::Ideogram4 => ImageDefaults {
                width: 1024,
                height: 1024,
                steps: 4,
            },
            // Qwen-Image(-Edit)-Rapid: 4-step distill, authored at 1024.
            ImageKind::QwenImageEdit | ImageKind::QwenImage => ImageDefaults {
                width: 1024,
                height: 1024,
                steps: 4,
            },
            ImageKind::ZImage => ImageDefaults {
                width: IMAGE_DEFAULT_WIDTH,
                height: IMAGE_DEFAULT_HEIGHT,
                steps: IMAGE_DEFAULT_STEPS,
            },
        }
    }
}

impl std::fmt::Display for ImageModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ImageModelId::ZImageTurboQ8 => "zimage-turbo-q8",
            ImageModelId::ZImageTurboQ4 => "zimage-turbo-q4",
            ImageModelId::ZImageTurboBf16 => "zimage-turbo-bf16",
            ImageModelId::Ideogram4Q8 => "ideogram4-q8",
            ImageModelId::QwenImageEditRapid => "qwen-image-edit-rapid",
            ImageModelId::QwenImageRapid => "qwen-image-rapid",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
pub enum VideoModelId {
    /// FastWan2.2-TI2V-5B-FullAttn, DMD-distilled (3-step, CFG-free). The
    /// e2e-validated path.
    #[cfg_attr(feature = "cli", value(name = "fastwan-ti2v-5b"))]
    #[cfg_attr(feature = "serde", serde(rename = "fastwan-ti2v-5b"))]
    FastwanTi2v5b,
    /// LongLive-2.0-5B: causal/AR long-video finetune of the FastWan base.
    /// 4-step FlowUniPC per chunk over a windowed KV cache.
    #[cfg_attr(feature = "cli", value(name = "longlive-2.0-5b"))]
    #[cfg_attr(feature = "serde", serde(rename = "longlive-2.0-5b"))]
    Longlive205b,
}

impl VideoModelId {
    pub const DEFAULT: VideoModelId = VideoModelId::FastwanTi2v5b;

    pub fn manifest(self) -> &'static ModelManifest {
        &wanmf::MANIFEST
    }

    pub fn variant(self) -> &'static wanmf::VariantFiles {
        wanmf::variant(&self.to_string()).expect("VideoModelId missing from VARIANTS registry")
    }

    /// AR (LongLive) path: the `.pt` DiT + windowed-KV-cache chunk loop.
    pub fn is_ar(self) -> bool {
        matches!(self, VideoModelId::Longlive205b)
    }

    /// Model-preferred playback fps: default for fps and the `--duration`
    /// divisor. The Wan TI2V line is authored at 24.
    pub fn fps(self) -> u32 {
        24
    }

    /// Default clip length in seconds when neither frames nor duration is
    /// given. 5s is the Wan2.2-TI2V-5B design point; LongLive (same base) is
    /// happy at the same length, just on a coarser grid.
    pub const DEFAULT_DURATION_SECS: u32 = 5;

    /// Default clip length (frames) when neither frames nor duration is given:
    /// [`Self::DEFAULT_DURATION_SECS`] at the model fps, snapped to its legal
    /// grid. Same target seconds for both models; the snap lands FastWan at 121
    /// and LongLive at 125 (its chunk-of-8 grid is coarser).
    pub fn default_frames(self) -> u32 {
        self.snap_frames(Self::DEFAULT_DURATION_SECS * self.fps())
    }

    /// Snap a raw frame count to this model's legal temporal grid. FastWan needs
    /// `4k+1` (causal-VAE grid); LongLive additionally needs latent frame count
    /// `(frames-1)/4+1` a positive multiple of 8 -> frames in {29, 61, 93, ...}.
    pub fn snap_frames(self, raw: u32) -> u32 {
        let raw = raw.max(1);
        if self.is_ar() {
            let f_lat = (raw as f32 + 3.0) / 4.0;
            let f_lat8 = ((f_lat / 8.0).round().max(1.0) as u32) * 8;
            4 * f_lat8 - 3
        } else {
            let k = ((raw - 1) as f32 / 4.0).round() as u32;
            4 * k + 1
        }
    }

    /// Validate an explicit frame count against the model grid (see
    /// [`Self::snap_frames`]).
    pub fn validate_frames(self, frames: u32) -> Result<(), String> {
        if frames == 0 || frames % 4 != 1 {
            return Err(format!(
                "--frames must be 4*k + 1 (got {frames}); e.g. 1, 5, 9, ..., 97"
            ));
        }
        if self.is_ar() {
            let f_lat = (frames - 1) / 4 + 1;
            if !f_lat.is_multiple_of(8) {
                return Err(format!(
                    "--frames for {self} must have latent frame count (frames-1)/4+1 \
                     divisible by 8 (got {frames} -> {f_lat}); e.g. 29, 61, 93, 125"
                ));
            }
        }
        Ok(())
    }

    /// AR chunk size in latent frames (`num_frame_per_block`). Multi-shot
    /// lengths split in chunk units (a scene cut lands on a chunk boundary).
    pub const AR_CHUNK_FLAT: u32 = 8;

    /// AR-grid frame count -> whole AR chunks. Validates the grid first.
    pub fn frames_to_chunks(self, frames: u32) -> Result<usize, String> {
        self.validate_frames(frames)?;
        let f_lat = (frames - 1) / 4 + 1;
        Ok((f_lat / Self::AR_CHUNK_FLAT) as usize)
    }

    /// Whole AR chunks -> the continuous clip's frame count.
    pub fn chunks_to_frames(self, chunks: usize) -> u32 {
        4 * (chunks as u32 * Self::AR_CHUNK_FLAT) - 3
    }
}

impl std::fmt::Display for VideoModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            VideoModelId::FastwanTi2v5b => "fastwan-ti2v-5b",
            VideoModelId::Longlive205b => "longlive-2.0-5b",
        })
    }
}

/// VAE decoder choice. App-local mirror of `wan::pipeline::VaeChoice` so the
/// clap derive (and a future `ToSchema`) lives on a type we own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum VaeChoice {
    Full,
    Tiny,
}

impl From<VaeChoice> for thinfer_models::wan::pipeline::VaeChoice {
    fn from(v: VaeChoice) -> Self {
        match v {
            VaeChoice::Full => Self::Full,
            VaeChoice::Tiny => Self::Tiny,
        }
    }
}

impl std::fmt::Display for VaeChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            VaeChoice::Full => "full",
            VaeChoice::Tiny => "tiny",
        })
    }
}

/// FastWan denoise sampler. App-local mirror of `wan::pipeline::VideoSampler` (so
/// the clap/`ToSchema` derives live on a type we own). The step count travels
/// alongside on the request, not in the enum, so it stays a plain UI number.
/// `UniPc` is the serve/UI default (matches the public FastWan Spaces); `Dmd` is
/// the byte-parity reference path. Ignored on the AR (LongLive) path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum VideoSampler {
    #[default]
    UniPc,
    Dmd,
}

impl VideoSampler {
    /// Build the engine sampler, folding in the user step count (UniPC only; DMD
    /// bakes its own 3-step schedule and ignores `steps`).
    pub fn into_engine(self, steps: u32) -> thinfer_models::wan::pipeline::VideoSampler {
        use thinfer_models::wan::pipeline::VideoSampler as E;
        match self {
            VideoSampler::UniPc => E::UniPc { steps },
            VideoSampler::Dmd => E::Dmd,
        }
    }
}

impl std::fmt::Display for VideoSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            VideoSampler::UniPc => "unipc",
            VideoSampler::Dmd => "dmd",
        })
    }
}

/// HyperSwap checkpoint (FaceFusion `models-3.3.0`). 1a/1b/1c share architecture
/// and speed; different visual character.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
pub enum SwapModel {
    #[cfg_attr(feature = "cli", value(name = "hyperswap-1a"))]
    #[cfg_attr(feature = "serde", serde(rename = "hyperswap-1a"))]
    Hyperswap1a,
    #[cfg_attr(feature = "cli", value(name = "hyperswap-1b"))]
    #[cfg_attr(feature = "serde", serde(rename = "hyperswap-1b"))]
    Hyperswap1b,
    #[cfg_attr(feature = "cli", value(name = "hyperswap-1c"))]
    #[cfg_attr(feature = "serde", serde(rename = "hyperswap-1c"))]
    Hyperswap1c,
}

impl SwapModel {
    pub const DEFAULT: SwapModel = SwapModel::Hyperswap1a;

    /// The HyperSwap weight file for this checkpoint.
    pub fn file(self) -> thinfer_core::manifest::FileRef {
        let path = match self {
            SwapModel::Hyperswap1a => "hyperswap_1a_256.onnx",
            SwapModel::Hyperswap1b => "hyperswap_1b_256.onnx",
            SwapModel::Hyperswap1c => "hyperswap_1c_256.onnx",
        };
        thinfer_core::manifest::FileRef::new("facefusion/models-3.3.0", path)
    }
}

impl std::fmt::Display for SwapModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            SwapModel::Hyperswap1a => "hyperswap-1a",
            SwapModel::Hyperswap1b => "hyperswap-1b",
            SwapModel::Hyperswap1c => "hyperswap-1c",
        })
    }
}

/// The SCRFD detector + ArcFace embedder shared by every face-swap run.
pub const SCRFD: thinfer_core::manifest::FileRef =
    thinfer_core::manifest::FileRef::new("deepghs/insightface", "buffalo_s/det_500m.onnx");
pub const ARCFACE: thinfer_core::manifest::FileRef =
    thinfer_core::manifest::FileRef::new("facefusion/models-3.0.0", "arcface_w600k_r50.onnx");
