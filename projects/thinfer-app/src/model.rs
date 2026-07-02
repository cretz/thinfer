//! Canonical model identifiers and their per-model defaults / grids. Single
//! source of truth shared by every front end: the CLI derives clap `ValueEnum`
//! off these (the `cli` feature), and the values feed the variant registry in
//! `thinfer-models`. `Display` is the canonical id string (matches the clap
//! value names and the registry keys); the registry lookups key off it.

use thinfer_core::manifest::ModelManifest;
use thinfer_models::hunyuan::manifest as hunmf;
use thinfer_models::ideogram4::manifest as idmf;
use thinfer_models::ltx::manifest as ltxmf;
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
    /// LTX-2.3 distilled-1.1: a 22B joint audio-video DiT (two-stage distilled).
    /// Its own pipeline (Gemma-3 encoder, dual-stream DiT, two VAEs + vocoder);
    /// ignores the FastWan sampler/vae/shot knobs. Output MP4 carries an AAC
    /// audio track unless audio is disabled (`--no-audio`), which skips the audio
    /// tail for a faster video-only MP4. The video VAE decode tiles spatially to
    /// the residency budget, so larger dims fit (slower); keep clips short (the
    /// temporal dim is decoded whole).
    #[cfg_attr(feature = "cli", value(name = "ltx-2.3-distilled"))]
    #[cfg_attr(feature = "serde", serde(rename = "ltx-2.3-distilled"))]
    Ltx23Distilled,
    /// LTX-2.3 distilled-1.1 with the Q4_K_M DiT (footprint variant): same
    /// pipeline as [`Self::Ltx23Distilled`] but the 14.2GB Q4_K_M DiT GGUF in
    /// place of the 22.8GB Q8_0 (encoder/connector/VAEs/upscaler unchanged). The
    /// DiT runs the per-quant-kind dense dequant path. Q8_0 stays the quality
    /// baseline; this halves DiT VRAM/disk for tighter budgets. No per-step speed
    /// change at product scale (the DiT is compute-bound, weight streaming hidden
    /// by prefetch), so it is a footprint option, not a perf one.
    #[cfg_attr(feature = "cli", value(name = "ltx-2.3-distilled-q4"))]
    #[cfg_attr(feature = "serde", serde(rename = "ltx-2.3-distilled-q4"))]
    Ltx23DistilledQ4,
    /// Sulphur-2 (SulphurAI): an uncensored LTX-2.3 DiT finetune. Byte-identical
    /// DiT layout to [`Self::Ltx23Distilled`], so it runs the exact same pipeline
    /// (Gemma-3 encoder, dual-stream DiT, two VAEs + vocoder, two-stage distilled
    /// sampler); only the DiT weights differ (Q8_0 baseline). The
    /// encoder/connector/VAEs/upscaler are the unchanged LTX-2.3 components.
    #[cfg_attr(feature = "cli", value(name = "sulphur-2"))]
    #[cfg_attr(feature = "serde", serde(rename = "sulphur-2"))]
    Sulphur2,
    /// Sulphur-2 with the Q4_K_M DiT (footprint variant): same pipeline as
    /// [`Self::Sulphur2`] with the smaller DiT GGUF (per-quant-kind dense dequant
    /// path), mirroring the LTX Q8/Q4 pair. Q8_0 stays the quality baseline.
    #[cfg_attr(feature = "cli", value(name = "sulphur-2-q4"))]
    #[cfg_attr(feature = "serde", serde(rename = "sulphur-2-q4"))]
    Sulphur2Q4,
    /// Wan2.2-T2V-A14B (MoE): two 14B experts (high/low noise) + the LightX2V
    /// 4-step distill LoRA, GGUF Q5_K_M, on the Wan backbone. CFG-free distill
    /// denoise (high expert steps 0-1, low expert 2-3); Wan2.1 VAE. The state-of-
    /// the-art Wan-family quality tier; heavier than the 5B, so it defaults to
    /// 832x480 on the 8GB card.
    #[cfg_attr(feature = "cli", value(name = "wan2.2-t2v-a14b"))]
    #[cfg_attr(feature = "serde", serde(rename = "wan2.2-t2v-a14b"))]
    Wan22T2vA14b,
    /// HunyuanVideo 1.5 T2V, lightx2v 4-step flow-match distill (CFG-free). Its
    /// own pipeline (Qwen2.5-VL encoder + SingleTokenRefiner, 54-block dual-stream
    /// MMDiT, 16x-spatial/4x-temporal causal-conv VAE); shares none of the
    /// Wan/LTX machinery. Industry-norm 480p default (832x480, 81f @ 16fps, 4
    /// step). FastWan-class quality target with stronger faces.
    #[cfg_attr(feature = "cli", value(name = "hunyuan-video-1.5-t2v"))]
    #[cfg_attr(feature = "serde", serde(rename = "hunyuan-video-1.5-t2v"))]
    Hunyuan15T2v,
    /// HunyuanVideo 1.5 causal TI2V (minWM WorldPlay `HY15/TI2V/dmd`, 4-step
    /// DMD). Same 8B MMDiT family as the T2V, run chunk-autoregressively over a
    /// KV cache: 4 flow-match Euler steps + a recache pass per 4-latent-frame
    /// chunk. `--input-image` is OPTIONAL: with it the run is image-conditioned
    /// (SigLIP + VAE-encoded first frame); without it the model generates from
    /// the prompt alone (probe-validated coherent despite the i2v training).
    /// 832x480, 77f @ 16fps default (latent frames must chunk by 4 -> frames in
    /// {13, 29, 45, 61, 77, ...}).
    #[cfg_attr(feature = "cli", value(name = "hunyuan-video-1.5-ti2v"))]
    #[cfg_attr(feature = "serde", serde(rename = "hunyuan-video-1.5-ti2v"))]
    Hunyuan15I2v,
}

impl VideoModelId {
    pub const DEFAULT: VideoModelId = VideoModelId::FastwanTi2v5b;

    pub fn manifest(self) -> &'static ModelManifest {
        match self {
            VideoModelId::Ltx23Distilled | VideoModelId::Ltx23DistilledQ4 => &ltxmf::MANIFEST,
            VideoModelId::Sulphur2 | VideoModelId::Sulphur2Q4 => &ltxmf::SULPHUR_MANIFEST,
            VideoModelId::Hunyuan15T2v | VideoModelId::Hunyuan15I2v => &hunmf::MANIFEST,
            _ => &wanmf::MANIFEST,
        }
    }

    /// LTX DiT GGUF role for this variant: Q8_0 (quality baseline) or Q4_K_M
    /// (footprint). Panics for non-LTX -- guard with [`Self::is_ltx`].
    pub fn ltx_dit_role(self) -> &'static str {
        match self {
            VideoModelId::Ltx23DistilledQ4 | VideoModelId::Sulphur2Q4 => {
                ltxmf::role::DIT_GGUF_Q4_K_M
            }
            VideoModelId::Ltx23Distilled | VideoModelId::Sulphur2 => ltxmf::role::DIT_GGUF_Q8_0,
            other => panic!("ltx_dit_role on non-LTX model {other}"),
        }
    }

    /// LTX text-encoder GGUF role for an [`EncoderQuant`] choice: the Q8_0
    /// baseline (uniform per-site, the conditioning-quality default) or the Q4_K_M
    /// variant (7.3G vs 12.5G; ~2.8x faster encode, lower-precision conditioning).
    /// The encoder's per-site dequant reads the Q4_K/Q6_K mix directly. Shared by
    /// every LTX/Sulphur variant (the encoder is common to all).
    pub fn ltx_encoder_role(self, encoder: EncoderQuant) -> &'static str {
        debug_assert!(self.is_ltx());
        match encoder {
            EncoderQuant::Q4 => ltxmf::role::ENCODER_GGUF_Q4,
            EncoderQuant::Q8 => ltxmf::role::ENCODER_GGUF,
        }
    }

    /// The distill-LoRA stack to fold into the Sulphur `dev` DiT, as
    /// `(role, strength)` pairs accumulated in order (see `ltx::lora`). Default =
    /// the single rank-reduced `condsafe` distill at 1.0 (the shipped, eyeballed
    /// path). `THINFER_SULPHUR_DISTILL`:
    /// - `stack` -> the distilled ComfyUI workflow recipe: `condsafe @ 0.7` +
    ///   official `distilled-lora-384 @ 0.5` (the distill applied at <1.0 may cut
    ///   the illustrative skew of the 1.0 single-fold).
    /// - `rank768` -> the `sulphur_lora_rank_768` content LoRA standalone
    ///   (known-bad in the 8-step path: undercooked mush -- repro only).
    ///
    /// Sulphur-only.
    pub fn sulphur_distill_stack(self) -> Vec<(&'static str, f32)> {
        debug_assert!(self.is_sulphur());
        match std::env::var("THINFER_SULPHUR_DISTILL").ok().as_deref() {
            Some("stack") => vec![
                (ltxmf::role::DISTILL_LORA, 0.7),
                (ltxmf::role::DISTILL_LORA_384, 0.5),
            ],
            Some("rank768") | Some("r768") => vec![(ltxmf::role::DISTILL_LORA_R768, 1.0)],
            _ => vec![(ltxmf::role::DISTILL_LORA, 1.0)],
        }
    }

    /// LTX joint-AV runtime role list for this variant (Q8_0 vs Q4_K_M DiT).
    pub fn ltx_runtime_roles(self) -> &'static [&'static str] {
        match self {
            VideoModelId::Ltx23DistilledQ4 | VideoModelId::Sulphur2Q4 => ltxmf::RUNTIME_ROLES_AV_Q4,
            _ => ltxmf::RUNTIME_ROLES_AV_Q8,
        }
    }

    /// Wan variant-registry entry. Wan models only (LTX sources by role; see
    /// [`Self::is_ltx`]). Panics for LTX -- guard call sites with `is_ltx`.
    pub fn variant(self) -> &'static wanmf::VariantFiles {
        debug_assert!(!self.is_ltx(), "variant() is Wan-only; LTX sources by role");
        wanmf::variant(&self.to_string()).expect("VideoModelId missing from VARIANTS registry")
    }

    /// AR (LongLive) path: the `.pt` DiT + windowed-KV-cache chunk loop.
    pub fn is_ar(self) -> bool {
        matches!(self, VideoModelId::Longlive205b)
    }

    /// HunyuanVideo 1.5 path: its own Qwen2.5-VL-encoder + dual-stream-MMDiT +
    /// causal-conv-VAE pipeline (`crate::hunyuan`); none of the Wan/LTX variant,
    /// sampler, or VAE-choice machinery applies. Covers the T2V and causal I2V
    /// variants; dispatch checks [`Self::is_hunyuan_i2v`] first.
    pub fn is_hunyuan(self) -> bool {
        matches!(
            self,
            VideoModelId::Hunyuan15T2v | VideoModelId::Hunyuan15I2v
        )
    }

    /// Causal I2V variant: chunked AR denoise over a KV cache, first-frame
    /// image conditioning (`--input-image` required), SigLIP + VAE-encoder
    /// components on top of the shared Hunyuan stack.
    pub fn is_hunyuan_i2v(self) -> bool {
        matches!(self, VideoModelId::Hunyuan15I2v)
    }

    /// LTX-2.3 joint audio-video path: its own two-stage pipeline + audio tail,
    /// none of the FastWan/LongLive (Wan-base) machinery applies.
    pub fn is_ltx(self) -> bool {
        matches!(
            self,
            VideoModelId::Ltx23Distilled
                | VideoModelId::Ltx23DistilledQ4
                | VideoModelId::Sulphur2
                | VideoModelId::Sulphur2Q4
        )
    }

    /// Sulphur-2 variants: the published GGUF is the BASE (`dev`) checkpoint, so
    /// the distill LoRA is folded into the DiT at load (see `ltx::lora`). The LTX
    /// distilled models ship pre-distilled DiTs and need no fold.
    pub fn is_sulphur(self) -> bool {
        matches!(self, VideoModelId::Sulphur2 | VideoModelId::Sulphur2Q4)
    }

    /// Default frame dims for this model. FastWan/LongLive ship the 960x544
    /// e2e baseline; LTX ships 1280x704 (16:9 widescreen), the regime the
    /// distilled model is in-distribution for. Lower-res LTX (e.g. the old
    /// 768x512 / 512x320 defaults) is OUT of distribution: the denoise stays
    /// coherent but renders the wrong subject/action (eyeballed: a man instead
    /// of the requested woman, off-script motion), because the distilled few-step
    /// model was trained at widescreen high res. 1280x704 is reached via the
    /// two-stage path (`upscale`, default-on for LTX): stage 1 denoises at half
    /// res (640x352) then a 2x latent upscale + 3-step refine at full res, which
    /// keeps the per-step DiT activation peak inside the 8GB card when paired
    /// with the LTX vram-budget cap (see `ltx::LTX_VRAM_BUDGET_CAP`). The video
    /// VAE decode tiles (spatial + temporal) to the residency budget. All callers
    /// resolve unset `--width/--height` through this.
    pub fn video_defaults(self) -> (u32, u32) {
        if self.is_ltx() {
            (1280, 704)
        } else if self.is_hunyuan() {
            // 832x480: HunyuanVideo 1.5's native 480p T2V regime (the res the
            // lightx2v 4-step distill was trained for). 16:9, /16-divisible.
            (832, 480)
        } else if matches!(self, VideoModelId::Wan22T2vA14b) {
            // 832x480: the lightx2v 480p distill regime, the industry-norm res for
            // this model. The old <=512x288 default existed only to dodge a
            // device-loss that was MISDIAGNOSED as a >4096-cell shader fault; the
            // real cause is the 2s Windows GPU watchdog (TDR) tripping on a single
            // long self-attention dispatch, now fixed by per-dispatch query
            // chunking in `op_sdpa` (bit-exact, scales to any sequence length). So
            // 832x480 is the default again; clip length is bounded by wall time /
            // VRAM, not a fault (see WAN22_MAX_LATENT_CELLS).
            (832, 480)
        } else {
            (VIDEO_DEFAULT_WIDTH, VIDEO_DEFAULT_HEIGHT)
        }
    }

    /// Default temporal self-attention window (latent-frame radius) when the
    /// caller leaves `--attn-window` unset. `Some(3)` for Wan2.2-14B (~1.55x e2e,
    /// user-eyeballed). `None` (full attention) for HunyuanVideo 1.5: W=3 looked
    /// clean on a single-subject clip (~2x DiT) but breaks multi-subject
    /// coherence -- a second cat spawned at latent frame ~14 once the +-3 window
    /// could no longer see the opening frames (browser eyeball 2026-07-01) -- so
    /// windowing there is opt-in via `--attn-window`. An explicit `0` disables it.
    pub fn default_attn_window(self) -> Option<u32> {
        match self {
            VideoModelId::Wan22T2vA14b => Some(3),
            _ => None,
        }
    }

    /// Whether this model's default denoise path is the two-stage upscale-refine
    /// (stage 1 half-res -> 2x latent upscale -> 3-step refine). True for LTX: on
    /// the 8GB target card, single-stage at the in-distribution widescreen res
    /// OOMs, and low-res single-stage is out of distribution, so two-stage is the
    /// only good regime. Surfaces that leave `upscale` unset default to this.
    pub fn two_stage_default(self) -> bool {
        self.is_ltx()
    }

    /// Minimum LTX pixel dim. 256 (the "fastest" tier, stage 1 at 128) is the
    /// floor: below it the model is so far out of distribution that output is
    /// incoherent, so dims under this are rejected rather than silently bad.
    pub const LTX_MIN_DIM: u32 = 256;

    /// Pixel-dim divisor this model requires (`--width`/`--height` must be a
    /// multiple). Wan narrows by 16; LTX's two-stage halving + /32 latent grid
    /// needs a multiple of 64 (stage 1 runs at half res, /32-divisible).
    pub fn dim_multiple(self) -> u32 {
        if self.is_ltx() { 64 } else { 16 }
    }

    /// Model-preferred playback fps: default for fps and the `--duration`
    /// divisor. The Wan TI2V line is authored at 24; Wan2.2-A14B at 16 (upstream).
    pub fn fps(self) -> u32 {
        if matches!(self, VideoModelId::Wan22T2vA14b) || self.is_hunyuan() {
            16
        } else {
            24
        }
    }

    /// Default clip length in seconds when neither frames nor duration is
    /// given. 5s is the Wan2.2-TI2V-5B design point; LongLive (same base) is
    /// happy at the same length, just on a coarser grid.
    pub const DEFAULT_DURATION_SECS: u32 = 5;

    /// Default clip length (frames) when neither frames nor duration is given:
    /// [`Self::DEFAULT_DURATION_SECS`] at the model fps, snapped to its legal
    /// grid. Same target seconds for both Wan models; the snap lands FastWan at
    /// 121 and LongLive at 125 (its chunk-of-8 grid is coarser). LTX overrides
    /// this with a short default ([`Self::LTX_DEFAULT_FRAMES`]): the full chain
    /// (single-submit VAE decode) fits only modest lengths on thin hardware.
    pub fn default_frames(self) -> u32 {
        if self.is_ltx() {
            Self::LTX_DEFAULT_FRAMES
        } else if matches!(self, VideoModelId::Wan22T2vA14b) {
            Self::WAN22_DEFAULT_FRAMES
        } else {
            self.snap_frames(Self::DEFAULT_DURATION_SECS * self.fps())
        }
    }

    /// Wan2.2-A14B default clip length (frames) at the 832x480 default res. Length
    /// is a wall-time choice now, not a fault envelope: the TDR crash is fixed (see
    /// `op_sdpa` query chunking, which removes the watchdog trip at ANY row count).
    /// 33f = f_lat 9 = 14040 DiT rows (~2.1s @16fps), ~2.25x the 13f/6240-row point
    /// that is validated e2e (`fixed_full.mp4`); 33f itself is browser-confirmed,
    /// not yet eyeballed in a parity gate. Longer is allowed up to
    /// [`Self::max_frames`] (the model's 81f design envelope) via explicit
    /// `--frames`/`--duration`; at 832x480 those run progressively slower (the
    /// self-attention is O(rows^2), ~70min for 33f / hours for 81f on the 8GB card).
    pub const WAN22_DEFAULT_FRAMES: u32 = 33;

    /// LTX ideal clip length: 121 frames (latent frame count `(f-1)/8+1 = 16`,
    /// ~5s @ 24fps) -- the length the model is distilled for (upstream
    /// `num_frames=121`). This is the UPPER target only: the actual default is
    /// `min(this, ltx_max_frames(w, h))`, because at the widescreen default the
    /// 8GB card cannot hold 121 frames (see [`Self::ltx_max_frames`]). At low res
    /// the cap is above 121, so the full 5s is used; at 1280x704 it lands at 49.
    pub const LTX_DEFAULT_FRAMES: u32 = 121;

    /// Max full-res latent cells (`f_lat * h/32 * w/32`) the 8GB card sustains
    /// through the LTX two-stage stage-2 denoise (which runs the DiT at full res).
    /// Above this the stage-2 activation peak, plus the per-block streaming
    /// alloc/free churn's fragmentation, loses the device. Derived empirically on
    /// the RTX 5070 Laptop (8GB): 49 frames @ 1280x704 = 6160 cells holds with
    /// ~2.7GB margin (peak 5.4GB); 73 frames @ 1024x576 = 5760 holds (peak 5.2GB);
    /// 73 frames @ 1280x704 = 8800 device-loses. 6300 sits just above both proven
    /// points and well below the failure. Revisit for >8GB cards.
    pub const LTX_MAX_LATENT_CELLS: u32 = 6300;

    /// Max DiT latent cells (`f_lat * h/16 * w/16`, == the per-forward token/row
    /// count) allowed for a Wan2.2-A14B clip. This is the model's 81-frame design
    /// envelope (81f @ 832x480 = f_lat 21 * 30 * 52 = 32760 rows), NOT a fault
    /// guard: the old "device LOST above ~6240 rows" was a 2s Windows GPU watchdog
    /// (TDR) tripping on a single long self-attention dispatch (nvlddmkm Event 153
    /// = an engine reset on watchdog timeout, NOT the shader OOB it was first read
    /// as). That is fixed by per-dispatch query chunking in `op_sdpa` (each SDPA
    /// dispatch is bounded to ~10M q*k pairs, <1s, in its own submit), which is
    /// bit-exact and scales to any row count. So this cap now just bounds wall
    /// time / VRAM to the model's intended envelope; explicit over-cap `--frames`
    /// still errors (fail fast) and a default/`--duration` caps down with a
    /// warning. Wan2.1 VAE is 8x spatial + patch 2 -> /16 grid (4x denser than
    /// LTX's /32). NB the longest length actually validated e2e is tracked by
    /// [`Self::WAN22_DEFAULT_FRAMES`]; lengths between that and this cap run but
    /// are progressively slower (O(rows^2) attention) and not yet eyeballed.
    pub const WAN22_MAX_LATENT_CELLS: u32 = 32760;

    /// Per-spatial-axis latent downscale (VAE spatial factor * patch): the DiT
    /// token grid is `h/div * w/div` per latent frame. LTX /32; Wan2.1 VAE /16.
    fn latent_spatial_divisor(self) -> u32 {
        if self.is_ltx() { 32 } else { 16 }
    }

    /// VAE temporal downscale (latent frames = `(frames-1)/factor + 1`). LTX 8,
    /// Wan 4. Inverse: `frames = factor*(f_lat-1) + 1`.
    fn temporal_factor(self) -> u32 {
        if self.is_ltx() { 8 } else { 4 }
    }

    /// The per-resolution VRAM activation-envelope cap, in latent cells, or `None`
    /// for models that fit at every legal dim on the 8GB card (FastWan/LongLive
    /// ship low-res defaults well inside the envelope). LTX and Wan2.2-A14B run
    /// the DiT at full res over a large token grid and need an explicit cap.
    fn max_latent_cells(self) -> Option<u32> {
        if self.is_ltx() {
            Some(Self::LTX_MAX_LATENT_CELLS)
        } else if matches!(self, VideoModelId::Wan22T2vA14b) {
            Some(Self::WAN22_MAX_LATENT_CELLS)
        } else {
            None
        }
    }

    /// Largest legal frame count whose full-res latent-cell count stays within
    /// this model's activation envelope ([`Self::max_latent_cells`]) at the given
    /// pixel dims, or `None` for an uncapped model. Lower res -> more frames fit;
    /// higher res -> fewer. Used to size the default clip and to reject explicit
    /// over-budget requests before any GPU work (so a long duration at high res
    /// fails fast at submit instead of device-losing mid-denoise). The returned
    /// count is on the model's temporal grid (`factor*k + 1`).
    pub fn max_frames(self, width: u32, height: u32) -> Option<u32> {
        let max_cells = self.max_latent_cells()?;
        let div = self.latent_spatial_divisor();
        let cells_per_lat_frame = (height / div).max(1) * (width / div).max(1);
        let max_f_lat = (max_cells / cells_per_lat_frame).max(1);
        Some(self.temporal_factor() * (max_f_lat - 1) + 1)
    }

    /// LTX alias of [`Self::max_frames`] (always `Some` for LTX). Kept for the
    /// LTX resolve path + its regression tests. Panics for non-LTX.
    pub fn ltx_max_frames(self, width: u32, height: u32) -> u32 {
        debug_assert!(self.is_ltx());
        self.max_frames(width, height)
            .expect("LTX always has an envelope cap")
    }

    /// Snap a raw frame count to this model's legal temporal grid. FastWan needs
    /// `4k+1` (causal-VAE grid); LongLive additionally needs latent frame count
    /// `(frames-1)/4+1` a positive multiple of 8 -> frames in {29, 61, 93, ...}.
    /// LTX needs `8k+1` (its VAE has temporal factor 8).
    pub fn snap_frames(self, raw: u32) -> u32 {
        let raw = raw.max(1);
        if self.is_ltx() {
            let k = ((raw - 1) as f32 / 8.0).round() as u32;
            8 * k + 1
        } else if self.is_ar() {
            let f_lat = (raw as f32 + 3.0) / 4.0;
            let f_lat8 = ((f_lat / 8.0).round().max(1.0) as u32) * 8;
            4 * f_lat8 - 3
        } else if self.is_hunyuan_i2v() {
            // Causal I2V chunks 4 latent frames at a time: latent frame count
            // (frames-1)/4+1 must be a positive multiple of 4 -> frames in
            // {13, 29, 45, 61, 77, ...}.
            let f_lat = (raw as f32 + 3.0) / 4.0;
            let f_lat4 = ((f_lat / 4.0).round().max(1.0) as u32) * 4;
            4 * f_lat4 - 3
        } else {
            let k = ((raw - 1) as f32 / 4.0).round() as u32;
            4 * k + 1
        }
    }

    /// Validate an explicit frame count against the model grid (see
    /// [`Self::snap_frames`]).
    pub fn validate_frames(self, frames: u32) -> Result<(), String> {
        if self.is_ltx() {
            if frames == 0 || !(frames - 1).is_multiple_of(8) {
                return Err(format!(
                    "--frames for {self} must be 8*k + 1 (got {frames}); e.g. 1, 9, 17, 25, 33"
                ));
            }
            return Ok(());
        }
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
            VideoModelId::Ltx23Distilled => "ltx-2.3-distilled",
            VideoModelId::Ltx23DistilledQ4 => "ltx-2.3-distilled-q4",
            VideoModelId::Sulphur2 => "sulphur-2",
            VideoModelId::Sulphur2Q4 => "sulphur-2-q4",
            VideoModelId::Wan22T2vA14b => "wan2.2-t2v-a14b",
            VideoModelId::Hunyuan15T2v => "hunyuan-video-1.5-t2v",
            VideoModelId::Hunyuan15I2v => "hunyuan-video-1.5-ti2v",
        })
    }
}

/// VAE decoder choice. App-local mirror of `wan::pipeline::VaeChoice` so the
/// clap derive (and a future `ToSchema`) lives on a type we own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum VaeChoice {
    Full,
    Tiny,
    /// Hunyuan-1.5 fine-tuned TAEHV: tiny-VAE speed (~seconds), better fidelity
    /// than the base tiny. Hunyuan-only; other models fall back to `Tiny`.
    #[cfg_attr(feature = "cli", value(name = "tiny-ft"))]
    TinyFt,
}

impl From<VaeChoice> for thinfer_models::wan::pipeline::VaeChoice {
    fn from(v: VaeChoice) -> Self {
        match v {
            VaeChoice::Full => Self::Full,
            // Wan has no fine-tuned tiny decoder; a `tiny-ft` request on a Wan
            // model degrades to its own tiny (the ft is a Hunyuan-only option,
            // and the web only offers it for Hunyuan).
            VaeChoice::Tiny | VaeChoice::TinyFt => Self::Tiny,
        }
    }
}

impl std::fmt::Display for VaeChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            VaeChoice::Full => "full",
            VaeChoice::Tiny => "tiny",
            VaeChoice::TinyFt => "tiny-ft",
        })
    }
}

/// LTX-2.3 text-encoder (Gemma-3-12B) quantization. Applies to ALL LTX models
/// (the encoder is shared across `ltx-2.3-distilled{,-q4}` and `sulphur-2{,-q4}`).
/// `Q8` is the conditioning-quality baseline (uniform Q8_0, 12.5G); `Q4` is the
/// Q4_K_M variant (7.3G; mixed Q4_K/Q6_K) -- ~2.8x faster encode (~16s -> ~6s)
/// for lower-precision conditioning. Ignored by the Wan models.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum EncoderQuant {
    #[default]
    Q8,
    Q4,
}

impl std::fmt::Display for EncoderQuant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            EncoderQuant::Q8 => "q8",
            EncoderQuant::Q4 => "q4",
        })
    }
}

/// HunyuanVideo 1.5 prompt-rewriter model choice. The rewriter expands a terse
/// prompt into the long, structured caption the DiT was trained on. `Fast` (the
/// default) runs the Qwen3-VL-4B GGUF (~2.5GB): small enough to decode quickly
/// even under a tight VRAM budget, so it is the budget-honest default. `Full`
/// runs the Qwen3-VL-8B (~5.85GB) for slightly richer captions at a much higher
/// time cost (it streams weights under a normal budget). Both share the same
/// runtime-parameterized `qwen3_lm` stack; only the weights + dims differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum RewriteQuality {
    #[default]
    Fast,
    Full,
}

impl RewriteQuality {
    /// The Hunyuan manifest role for this quality's rewriter GGUF.
    pub fn gguf_role(self) -> &'static str {
        use thinfer_models::hunyuan::manifest::role;
        match self {
            RewriteQuality::Fast => role::REWRITER_GGUF_4B_Q5_K_M,
            RewriteQuality::Full => role::REWRITER_GGUF_8B_Q5_K_M,
        }
    }

    /// The `qwen3_lm` architecture config for this quality's GGUF.
    pub fn lm_config(self) -> thinfer_models::qwen3_lm::Qwen3LmConfig {
        use thinfer_models::qwen3_lm::Qwen3LmConfig;
        match self {
            RewriteQuality::Fast => Qwen3LmConfig::qwen3_vl_4b(),
            RewriteQuality::Full => Qwen3LmConfig::qwen3_vl_8b(),
        }
    }
}

impl std::fmt::Display for RewriteQuality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            RewriteQuality::Fast => "fast",
            RewriteQuality::Full => "full",
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
