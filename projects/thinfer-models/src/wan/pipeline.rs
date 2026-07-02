//! FastWan2.2-TI2V-5B pipeline orchestrator. Single entry point for CLI, web,
//! and the e2e: `WanModel::load(...)` builds the bundle once, `generate` runs
//! the whole stack (tokenize -> umT5 encode -> DMD few-step denoise loop with
//! the Wan DiT -> TI2V VAE decode -> video frames). Mirrors
//! `z_image::pipeline::ZImageModel`.
//!
//! Owns the compiled `Umt5Pipelines` + `WanDitPipelines`, the residency-backed
//! handle bundles, the `WanVaeDecoder`, the residency, backend, and tokenizer.
//! No model internals leak past `generate`'s `WanVideo` return.
//!
//! DMD distillation: a fixed handful of timesteps (3 for FastWan), CFG-free, one
//! DiT forward per step over the whole `[48, f_lat, h, w]` latent with a single
//! scalar timestep, renoised between steps. LongLive (4-step, autoregressive)
//! reuses this backbone with its own `DmdConfig` and an AR path (see
//! `wan-plan.md`).

use std::collections::HashMap;
use std::sync::Arc;

use thinfer_core::arbiter::RECLAIM_EVICTABLE_WEIGHTS;
use thinfer_core::backend::{WgpuBackend, WgpuError};
use thinfer_core::ops::{ActDtype, WeightDtype, WgslConfig};
use thinfer_core::residency::{ResidencyError, WeightResidency};
use thinfer_core::tensor::StorageEncoding;
use thinfer_core::tokenizer::{Tokenizer, TokenizerError};
use thinfer_core::trace;
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::Workspace;

use crate::common::block::{BlockWgslConfigs, CoopmatSites, DenseActSites};
use crate::wan::dit::{
    LoadedWanDitHandles, WanDit, WanDitError, WanDitInputs, WanDitShape, WanDitTaps, read_into_f32,
};
use crate::wan::dit_block::{WanDitBlockTaps, WanDitConfig, WanDitPipelines, config as dit_config};
use crate::wan::kv_cache::{KvCacheConfig, KvWindowCache, RamKvStore};
use crate::wan::loader::{WanI8Sites, register_wan_dit_handles};
use crate::wan::manifest::RECIPE;
use crate::wan::scheduler::{DmdConfig, DmdSampler, Wan22DistillConfig, Wan22DistillSampler};
use crate::wan::umt5::{
    Umt5BlockOpsHost, Umt5Encoder, Umt5ForwardError, Umt5Handles, Umt5Pipelines, Umt5Taps,
    register_umt5_handles,
};
use crate::wan::unipc::{FlowUniPc, UniPcConfig};
use crate::wan::vae::{
    VaeDecoderWeights, WanVaeConfig, WanVaeDecodeError, WanVaeDecoder, WanVaePipelines,
    register_decoder,
};
use crate::wan::vae_tiny::{
    TinyDecoderWeights, WanVaeTinyDecodeError, WanVaeTinyDecoder, WanVaeTinyPipelines,
    register_decoder_tiny,
};

/// umT5 rope-free context cap. The Wan DiT cross-attends to a fixed
/// `max_sequence_length`; SkyReels-V2 ships 512.
const TEXT_SEQ: usize = dit_config::TEXT_SEQ;
const MAX_PROMPT_TOKENS: usize = TEXT_SEQ;
/// VAE temporal compression: 4x for BOTH the Wan2.1 VAE (14B) and the
/// Wan2.2-TI2V high-compression VAE (5B). Spatial compression DIFFERS (8x Wan2.1
/// vs 16x TI2V) and is per-model (`WanModel.vae_scale`, from the loaded
/// `WanVaeConfig`); the latent grid the DiT sees derives from both.
const TEMPORAL_SCALE: usize = 4;

/// The DiT step loop prefetches two blocks ahead (next-acquire + prefetch via
/// the `join!` in `WanDit::forward`), so up to this many whole-tensor upload
/// stagings overlap in VRAM at once.
const PREFETCH_STAGING_DEPTH: u64 = 2;
/// VRAM headroom held for the DiT forward's activation workspace, on top of the
/// prefetch staging envelope (see `set_transient_reserve`). Calibrated at the
/// e2e gate dims (32x32x5, workspace ~46 MiB); larger resolutions grow the
/// workspace and rely on the budget-tier / activation-tiling path, not this
/// fixed pad.
const DIT_WORKSPACE_RESERVE: u64 = 64 * 1024 * 1024;

/// FastWan (non-AR) denoise sampler. Both run the SAME DiT forward CFG-free; only
/// the schedule around it differs. `Dmd` is the few-step renoise schedule the DMD
/// distillation bakes (the parity default); `UniPc` is the plain UniPC multistep
/// solver the public HF Spaces use (`flow_shift=8.0`), with a user step count.
/// Ignored on the AR path ([`WanModel::generate_ar`] drives its own UniPC).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VideoSampler {
    /// DMD 3-step renoise. The byte-parity reference path.
    #[default]
    Dmd,
    /// UniPC multistep, `flow_shift=8.0`, CFG-free. `steps` denoise steps
    /// (UI 1..=8, default 4). The served/UI default for FastWan.
    UniPc { steps: u32 },
}

/// Inputs to one `generate` call. CFG-free (no `guidance_scale`/`negative_prompt`
/// knob, as the DMD distillation and the UniPC Spaces both run guidance 0). The
/// step count is not here: DMD bakes its schedule, UniPC carries it in
/// [`VideoSampler::UniPc`].
pub struct GenerationParams {
    pub prompt: String,
    /// Frame height in pixels. Divisible by `vae_scale * PATCH_H` (32 for the
    /// TI2V 5B VAE, 16 for the Wan2.1 14B VAE).
    pub height: u32,
    /// Frame width in pixels. Divisible by `vae_scale * PATCH_W`.
    pub width: u32,
    /// Output frame count. Must be `4 * k + 1` (the causal-VAE temporal grid).
    pub num_frames: u32,
    /// Deterministic seed for the initial latent noise (and per-step renoise).
    pub seed: u64,
    /// Denoise sampler (FastWan path only; AR ignores it).
    pub sampler: VideoSampler,
    /// Temporal self-attention window radius in latent frames. `Some(W)`
    /// restricts DiT self-attention to keys within `±W` latent frames (breaks
    /// the O(frames^2) cost on long clips at the price of dropping long-range
    /// temporal links); `None` runs full self-attention. Honored only on the
    /// activation-tiled path (long clips); short clips run full attention.
    pub attn_window: Option<u32>,
}

/// One shot of a multi-shot LongLive AR generation: a prompt that holds for
/// `chunks` consecutive AR chunks (each chunk = `num_frame_per_block` latent
/// frames). A scene cut between shots is signalled to the model in-band by
/// prepending [`SCENE_CUT_PREFIX`] to the first chunk of every shot after the
/// first (mirrors upstream `MultiTextConcatDataset._apply_shot_durations`), and
/// in the cache by advancing the RoPE phase + pinning the new shot's sink. A
/// single-shot long video is the empty-shots path ([`WanModel::denoise_ar`]).
#[derive(Clone, Debug)]
pub struct Shot {
    pub prompt: String,
    /// Length of this shot in AR chunks (>= 1). The sum across shots must equal
    /// the run's total chunk count (`f_lat / num_frame_per_block`).
    pub chunks: usize,
}

/// In-band scene-cut marker prepended to the first chunk's prompt of every shot
/// after the first. The model was trained with this exact string as the cut
/// signal (upstream `DEFAULT_SCENE_CUT_PREFIX`), so it is part of the umT5 text,
/// not just a host-side flag.
pub const SCENE_CUT_PREFIX: &str = "The scene transitions. ";

/// Decoded video: CTHW f32 in `[-1, 1]` plus dims, ready for the caller's
/// staging (per-frame PNG sequence + contact sheet). No codec here.
pub struct WanVideo {
    /// `[3, num_frames, height, width]` row-major f32, clamped to `[-1, 1]`.
    pub frames: Vec<f32>,
    pub num_frames: usize,
    pub height: usize,
    pub width: usize,
}

/// Per-step denoise telemetry for parity bisection. Filled when `denoise_with`
/// gets a `Some(step_diag)` sink; every field is read back from the GPU only on
/// that path, so prod / `generate` (which passes `None`) pays nothing. Lets the
/// e2e split a per-step divergence into DiT-velocity vs scheduler vs a specific
/// DiT block, at EVERY step (not just step 0 like `diag_step0`).
pub struct WanStepDiag {
    /// The model timestep fed to the DiT this step.
    pub timestep: f32,
    /// Flow sigma `t / num_train_timesteps` used to convert velocity -> x0.
    pub sigma: f32,
    /// Raw DiT output (flow velocity) handed to the sampler == the exact
    /// tensor pyref dumps as `py_dit_out_step{i}`.
    pub velocity: Vec<f32>,
    /// Latent after `sampler.step` (the old per-step dump; == `py_step{i}_post`).
    pub post: Vec<f32>,
    /// Residual stream after each DiT block (`len == num_layers`). Localizes a
    /// velocity divergence to a block (vs `py_block{b}_out_step{i}`).
    pub per_block: Vec<Vec<f32>>,
    pub temb: Vec<f32>,
    pub timestep_proj: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub proj_out: Vec<f32>,
}

/// Stage notifications for user-facing progress. Distinct from tracing.
#[derive(Clone, Copy, Debug)]
pub enum ProgressEvent {
    TextEncode,
    Step {
        i: u32,
        n: u32,
    },
    /// AR (LongLive) per-chunk denoise step: `chunk`/`num_chunks` and the 1-based
    /// `step`/`num_steps` within that chunk (the FlowUniPC schedule). The clean
    /// recache forward is not reported (it is not a scheduler step).
    ChunkStep {
        chunk: u32,
        num_chunks: u32,
        step: u32,
        num_steps: u32,
    },
    VaeDecode,
}

pub type ProgressFn<'a> = Option<&'a dyn Fn(ProgressEvent)>;

/// Cooperative cancellation hook. Returns `true` when the caller wants the
/// in-flight generate to stop; the denoise loops poll it at each step boundary
/// and bail with [`GenerateError::Cancelled`]. `None` (CLI default) never
/// cancels. The server passes a closure backed by the job's cancel flag.
pub type Cancel<'a> = Option<&'a dyn Fn() -> bool>;

/// Which VAE decoder a generate/decode uses. No `Default`: callers state intent
/// explicitly (the real VAE is the parity path; the LightTAE tiny decoder is
/// opt-in and only loaded when selected). Requesting `Tiny` on a model loaded
/// `Full` is a [`GenerateError::TinyVaeNotLoaded`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VaeChoice {
    /// Real `AutoencoderKLWan` decoder (`wan/vae.rs`); bit-clean parity default.
    Full,
    /// LightTAE `lighttaew2_2` tiny decoder (`wan/vae_tiny.rs`); fast, opt-in.
    Tiny,
}

pub struct WanModel<S: WeightSource, T: Tokenizer> {
    backend: Arc<WgpuBackend>,
    residency: WeightResidency<S>,
    tokenizer: T,
    umt5: Umt5Encoder,
    umt5_pipelines: Umt5Pipelines,
    umt5_handles: Umt5Handles,
    dit: WanDitPipelines,
    /// The (high-noise, for MoE) DiT handle set, registered under the `high.`
    /// prefix on the Wan2.2 path or unprefixed otherwise.
    dit_handles: LoadedWanDitHandles,
    /// Wan2.2-A14B low-noise expert (`low.` prefix). `Some` switches `denoise`
    /// to the step-distill MoE path (high steps `0..boundary`, low after).
    dit_handles_low: Option<LoadedWanDitHandles>,
    /// VAE spatial compression (8 for the Wan2.1 14B VAE, 16 for the TI2V 5B).
    /// Latent <-> pixel dim math reads this (temporal is the const TEMPORAL_SCALE).
    vae_scale: usize,
    /// Per-variant DiT geometry. Drives latent channels, the block dims, and the
    /// loader layer count.
    cfg: WanDitConfig,
    vae: WanVaeDecoder,
    /// LightTAE tiny decoder, present only when the model was loaded with
    /// `VaeChoice::Tiny` (the tiny weights are downloaded + registered +
    /// compiled only then). `None` keeps the parity path free of tiny-VAE cost.
    vae_tiny: Option<WanVaeTinyDecoder>,
    /// Weight dtype the DiT matmul kernels were compiled against (`Bf16` for the
    /// safetensors path, `Quant(k)` when the GGUF surfaced the matmuls). Lets
    /// the e2e assert the GGUF union actually fed the DiT.
    dit_matmul_weight: WeightDtype,
}

#[derive(Debug)]
pub enum ModelLoadError {
    Dit(crate::wan::loader::LoadError),
    Umt5(crate::wan::umt5::LoadError),
    // The VAE loader uses the shared `common::loader` registration primitives.
    Vae(crate::common::loader::LoadError),
    Wgpu(WgpuError),
}

impl From<crate::wan::loader::LoadError> for ModelLoadError {
    fn from(e: crate::wan::loader::LoadError) -> Self {
        Self::Dit(e)
    }
}
impl From<crate::wan::umt5::LoadError> for ModelLoadError {
    fn from(e: crate::wan::umt5::LoadError) -> Self {
        Self::Umt5(e)
    }
}
impl From<crate::common::loader::LoadError> for ModelLoadError {
    fn from(e: crate::common::loader::LoadError) -> Self {
        Self::Vae(e)
    }
}
impl From<WgpuError> for ModelLoadError {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}

#[derive(Debug)]
pub enum GenerateError<SE: core::fmt::Debug> {
    Tokenizer(TokenizerError),
    Umt5(Umt5ForwardError<SE>),
    Dit(WanDitError<SE>),
    Vae(WanVaeDecodeError<SE>),
    TinyVae(WanVaeTinyDecodeError<SE>),
    /// `generate`/`decode` requested `VaeChoice::Tiny` but the model was loaded
    /// `Full` (tiny weights never registered). Reload with `Tiny`.
    TinyVaeNotLoaded,
    Wgpu(WgpuError),
    Residency(ResidencyError<SE, WgpuError>),
    InvalidDims {
        height: u32,
        width: u32,
    },
    InvalidFrames {
        num_frames: u32,
    },
    PromptTooLong {
        tokens: usize,
        max: usize,
    },
    /// Multi-shot plan invalid: the shot chunk counts do not sum to the run's
    /// total chunk count, or a shot has zero chunks. (AR path only.)
    InvalidShots {
        shot_chunks: usize,
        num_chunks: usize,
    },
    /// The caller's [`Cancel`] hook asked the in-flight generate to stop; the
    /// denoise loop bailed at a step boundary. Not an error condition -- the
    /// front end maps it to a "cancelled" job state, not a failure.
    Cancelled,
}

impl<SE: core::fmt::Debug> From<WgpuError> for GenerateError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for GenerateError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}
impl<SE: core::fmt::Debug> From<Umt5ForwardError<SE>> for GenerateError<SE> {
    fn from(e: Umt5ForwardError<SE>) -> Self {
        Self::Umt5(e)
    }
}
impl<SE: core::fmt::Debug> From<WanDitError<SE>> for GenerateError<SE> {
    fn from(e: WanDitError<SE>) -> Self {
        Self::Dit(e)
    }
}
impl<SE: core::fmt::Debug> From<WanVaeDecodeError<SE>> for GenerateError<SE> {
    fn from(e: WanVaeDecodeError<SE>) -> Self {
        Self::Vae(e)
    }
}
impl<SE: core::fmt::Debug> From<WanVaeTinyDecodeError<SE>> for GenerateError<SE> {
    fn from(e: WanVaeTinyDecodeError<SE>) -> Self {
        Self::TinyVae(e)
    }
}

/// Which Wan variant to build: DiT geometry, VAE config, and the quant/MoE
/// policy. Keeps `WanModel::load_variant` model-agnostic.
#[derive(Clone)]
pub struct WanVariant {
    pub dit: WanDitConfig,
    pub vae: WanVaeConfig,
    /// Two-expert MoE: register `high.`/`low.` handle sets + run the step-distill
    /// denoise with the per-step expert switch.
    pub moe: bool,
    /// Forced block activation dtype (overrides the f16-probe). The 14B residual
    /// stream has large-outlier channels that exceed f16's 65504 ceiling over its
    /// 40 layers (f16 saturates -> flat), so it runs bf16 acts (f32 exponent range,
    /// matches the pyref dtype). The 5B (3072 dim) fits f16. `None` = probe (F16
    /// when supported).
    pub act_pref: Option<ActDtype>,
}

impl WanVariant {
    /// FastWan2.2-TI2V-5B / LongLive-2.0-5B: single expert, bf16 safetensors,
    /// the TI2V high-compression VAE.
    pub fn fastwan_ti2v_5b() -> Self {
        Self {
            dit: WanDitConfig::fastwan_ti2v_5b(),
            vae: WanVaeConfig::fastwan_ti2v_5b(),
            moe: false,
            act_pref: None,
        }
    }

    /// Wan2.2-T2V-A14B: two 14B experts (GGUF, LightX2V distill folded -> Q8_0
    /// blocks; dense module-level weights stay bf16), the Wan2.1 VAE, distill MoE
    /// denoise. Bf16 acts (the 14B residual overflows f16).
    pub fn wan22_t2v_a14b() -> Self {
        Self {
            dit: WanDitConfig::wan22_14b(),
            vae: WanVaeConfig::wan2_1(),
            moe: true,
            act_pref: Some(ActDtype::Bf16),
        }
    }
}

impl<S: WeightSource, T: Tokenizer> WanModel<S, T> {
    /// Build the model: register every umT5 + DiT + VAE handle with residency
    /// and compile every WGSL kernel once. No bytes flow until `generate`.
    /// Single-expert FastWan/LongLive 5B; probe-driven act dtype.
    pub async fn load(
        backend: Arc<WgpuBackend>,
        residency: WeightResidency<S>,
        tokenizer: T,
        vae: VaeChoice,
        i8_matmul: bool,
    ) -> Result<Self, ModelLoadError> {
        Self::load_variant(
            backend,
            residency,
            tokenizer,
            vae,
            WanVariant::fastwan_ti2v_5b(),
            None,
            i8_matmul,
        )
        .await
    }

    /// Diagnostic [`load`] that forces the block activation dtype instead of
    /// probing device f16 support (the fp32-acts e2e, to separate amplified-bf16
    /// rounding from algorithmic error). FastWan/LongLive 5B variant.
    pub async fn load_with_act(
        backend: Arc<WgpuBackend>,
        residency: WeightResidency<S>,
        tokenizer: T,
        vae: VaeChoice,
        act_override: Option<ActDtype>,
        i8_matmul: bool,
    ) -> Result<Self, ModelLoadError> {
        Self::load_variant(
            backend,
            residency,
            tokenizer,
            vae,
            WanVariant::fastwan_ti2v_5b(),
            act_override,
            i8_matmul,
        )
        .await
    }

    /// Build a specific Wan variant (DiT geometry + VAE + MoE/quant policy).
    /// `WanVariant::fastwan_ti2v_5b` = the single-expert 5B path; `wan22_t2v_a14b`
    /// = the two-expert A14B (Q8_0 module transcode, distill denoise).
    pub async fn load_variant(
        backend: Arc<WgpuBackend>,
        residency: WeightResidency<S>,
        tokenizer: T,
        vae: VaeChoice,
        variant: WanVariant,
        act_override: Option<ActDtype>,
        i8_matmul: bool,
    ) -> Result<Self, ModelLoadError> {
        let timing = tracing::enabled!(tracing::Level::INFO);
        let t0 = timing.then(trace::Instant::now);

        let cfg = variant.dit;

        // --- handle registration (no upload) ---
        let vae_cfg = variant.vae.clone();
        let vae_scale = vae_cfg.spatial_compression;
        // Opt-in DP4A matmul: transcode the DP4A-safe site weights (qkv + ffn_up)
        // to Q8_0 at load and route them through the i8xi8 `matmul_i8`
        // (dot4I8Packed) path. Those A-sides are norm-conditioned (no massive
        // outliers), so it holds parity while cutting their matmul ~6x. o-proj /
        // ffn_down stay bf16 (outlier-prone A-side). Off => bf16 byte-identical.
        let q8 = i8_matmul.then_some(thinfer_core::quant::QuantKind::Q8_0);
        // qkv is split into two sites so only the SAFE half goes i8: self-attn
        // qkv has a normed A-side (DP4A-safe), so it takes Q8_0 + the i8 matmul;
        // cross-attn qkv stays bf16 because its K/V project from the un-normed
        // umT5 text and per-32 i8 act-quant there overflows f16 (latent saturates
        // to 65504). The split is the separate `matmul_qkv_self` pipeline +
        // `WanI8Sites.qkv_self` transcode. ffn_up's A-side is clean too.
        let i8_sites = WanI8Sites {
            qkv_self: q8,
            ffn_up: q8,
        };
        let umt5_handles = register_umt5_handles(&residency, None)?;
        // MoE registers two expert handle sets under `high.`/`low.` prefixes so
        // both coexist in one residency; single-expert uses no prefix. Block
        // matmuls carry no load transcode (bf16 path) or are already Q8_0 (the
        // Wan2.2 GGUF, folded); the dense module-level weights always stay bf16.
        let (dit_handles, dit_handles_low) = if variant.moe {
            let high = register_wan_dit_handles(&residency, &cfg, "high.", None, i8_sites)?;
            let low = register_wan_dit_handles(&residency, &cfg, "low.", None, i8_sites)?;
            (high, Some(low))
        } else {
            let h = register_wan_dit_handles(&residency, &cfg, "", None, i8_sites)?;
            (h, None)
        };
        let dit_probe_prefix = if variant.moe { "high." } else { "" };
        // VAE decoder weights all fit resident; diff the registered footprint
        // across registration so the decode can reserve exactly it (not a budget
        // fraction) when sizing its non-evictable tile workspace.
        let before_vae_bytes = residency.total_registered_bytes();
        let vae_handles = register_decoder(&residency, &VaeDecoderWeights::new(&vae_cfg))?;
        let vae_weight_footprint = residency.total_registered_bytes() - before_vae_bytes;
        // Tiny decoder registers only when selected (its weights are a separate
        // download in the source). Keeps the parity path free of tiny-VAE cost.
        let vae_tiny_handles = match vae {
            VaeChoice::Full => None,
            VaeChoice::Tiny => Some(register_decoder_tiny(
                &residency,
                &TinyDecoderWeights::new(),
            )?),
        };

        // Weights join the VRAM arbiter's reclaim chain so workspace/staging
        // growth evicts unpinned (LRU / prefetch-warmed) residents instead of
        // overshooting the budget. Without this the streamed weight set is
        // bounded only by same-size recycling and pins at the budget ceiling,
        // leaving no room for the in-flight transient envelope.
        residency.arbiter().register(
            RECLAIM_EVICTABLE_WEIGHTS,
            residency.reclaimer(Arc::clone(&backend)),
        );

        // --- dtype selection ---
        // Probe a representative matmul tensor per submodel: `Quant(k)` when the
        // GGUF surfaced it, `Bf16`/`F32` -> Bf16 dense. The Wan DiT uses ONE
        // pipeline set for every matmul site (patch, condition embedder, all 30
        // blocks, proj_out), so the dtype must be uniform; the ComfyUI GGUF
        // quantizes those big linears uniformly (norms/biases stay F32).
        let dit_w = probe_weight(
            &residency,
            &format!("{dit_probe_prefix}blocks.0.attn1.to_q.weight"),
        );
        let umt5_w = probe_weight(&residency, "encoder.block.0.layer.0.SelfAttention.q.weight");
        // `act_override` (the fp32 diagnostic) wins; else the variant's forced
        // dtype (14B -> Bf16, its residual overflows f16); else probe (F16 when
        // the device supports it).
        let act = act_override
            .or(variant.act_pref)
            .unwrap_or(if backend.supports_shader_f16() {
                ActDtype::F16
            } else {
                ActDtype::F32
            });
        // umT5 must NOT use f16 acts: T5-family residual streams are large and
        // ours grow monotonically through the 24 blocks, exceeding f16's 65504
        // ceiling around block ~20 (overflow -> inf -> NaN in final_layer_norm).
        // The magnitude is prompt-content-dependent, so the f16 path only blows
        // up on some prompts (the even-token gate prompt stayed in range; longer
        // prompts did not). The pyref text encoder runs bf16, so bf16 acts both
        // hold the range (f32 exponent) and match the reference dtype. Honor an
        // explicit override (the fp32 diagnostic), else force bf16 for umT5.
        let umt5_act = act_override.unwrap_or(ActDtype::Bf16);
        tracing::info!(?dit_w, ?umt5_w, ?act, ?umt5_act, "Wan dtype selection");

        let ffn_up_wd = i8_sites.ffn_up.map(WeightDtype::Quant);
        // On a Q8 DiT (Wan2.2 GGUF: every matmul weight is Quant), the i8 DP4A
        // path would be selected for ALL sites -- but the cross-attn qkv (un-normed
        // text K/V), proj, and ffn_down acts are outlier-heavy and overflow i8
        // act_quant's f16 scale (-> inf). Force those to the dense dequant-once
        // path; the normed A-sides (self-attn qkv + ffn_up) keep i8. FastWan
        // (bf16 DiT) leaves the non-i8 sites bf16 so this never triggers.
        let dense_acts = if matches!(dit_w, WeightDtype::Quant(_)) {
            DenseActSites {
                qkv: true,
                qkv_self: false,
                proj: true,
                ffn_up: false,
                ffn_down: true,
            }
        } else {
            DenseActSites::default()
        };
        let dit = WanDitPipelines::compile(
            &backend,
            &block_cfgs(
                dit_w,
                act,
                SiteOverride {
                    qkv_self: i8_sites.qkv_self.map(WeightDtype::Quant),
                    ffn_up: ffn_up_wd,
                    dense_acts,
                },
                // Mixed-precision f16 self-attention: only the bf16-act DiT (the
                // 14B, whose residual forces bf16) uses it; the F16-act FastWan
                // already runs the native f16 subgroup SDPA.
                true,
            ),
        )
        .await?;
        let umt5_pipelines = Umt5Pipelines::compile(
            &backend,
            // umT5 keeps bf16 SDPA: its residual is raw (un-normed) and the f16
            // path is only validated for normed DiT Q/K/V.
            &block_cfgs(umt5_w, umt5_act, SiteOverride::default(), false),
        )
        .await?;
        let vae_pipelines = WanVaePipelines::compile(&backend).await?;
        let vae_tiny = match vae_tiny_handles {
            None => None,
            Some(handles) => Some(WanVaeTinyDecoder {
                pipelines: WanVaeTinyPipelines::compile(&backend).await?,
                handles,
                cfg: (&vae_cfg).into(),
            }),
        };

        tracing::info!(
            elapsed_ms = t0.map_or(0, |t| t.elapsed().as_millis() as u64),
            "WanModel loaded"
        );
        Ok(Self {
            backend,
            residency,
            tokenizer,
            umt5: Umt5Encoder::new(MAX_PROMPT_TOKENS),
            umt5_pipelines,
            umt5_handles,
            dit,
            dit_handles,
            dit_handles_low,
            vae_scale,
            cfg,
            vae: WanVaeDecoder {
                pipelines: vae_pipelines,
                handles: vae_handles,
                cfg: vae_cfg,
                weight_footprint: vae_weight_footprint,
            },
            vae_tiny,
            dit_matmul_weight: dit_w,
        })
    }

    /// DiT matmul weight dtype picked up at load (`Bf16` safetensors,
    /// `Quant(k)` GGUF). Lets tests assert the GGUF union fed the DiT.
    pub fn dit_matmul_weight(&self) -> WeightDtype {
        self.dit_matmul_weight
    }

    /// Shared VRAM arbiter. Callers building their own `Workspace` (e2e) must
    /// use this so the budget has one owner.
    pub fn arbiter(&self) -> &Arc<thinfer_core::arbiter::MemArbiter> {
        self.residency.arbiter()
    }

    /// Dispatch a VAE decode to the chosen decoder. Both emit CTHW video
    /// `[3, 4*f-3, h*16, w*16]` in `[-1, 1]` at the same dims, so callers are
    /// decoder-agnostic. Tiny errors if the model was loaded `Full`.
    async fn decode_video(
        &self,
        vae: VaeChoice,
        workspace: &mut Workspace<WgpuBackend>,
        latent: &[f32],
        f_lat: usize,
        h_lat: usize,
        w_lat: usize,
        tiny_chunk: Option<usize>,
    ) -> Result<Vec<f32>, GenerateError<S::Error>> {
        let _s = trace::scope!("vae_decode", f_lat = f_lat).entered();
        Ok(match vae {
            VaeChoice::Full => {
                self.vae
                    .decode(
                        &self.backend,
                        &self.residency,
                        workspace,
                        latent,
                        f_lat,
                        h_lat,
                        w_lat,
                    )
                    .await?
            }
            VaeChoice::Tiny => {
                self.vae_tiny
                    .as_ref()
                    .ok_or(GenerateError::TinyVaeNotLoaded)?
                    .decode(
                        &self.backend,
                        &self.residency,
                        workspace,
                        latent,
                        f_lat,
                        h_lat,
                        w_lat,
                        tiny_chunk,
                    )
                    .await?
            }
        })
    }

    /// Run the full pipeline and return decoded video frames.
    pub async fn generate(
        &self,
        params: &GenerationParams,
        vae: VaeChoice,
        progress: ProgressFn<'_>,
        cancel: Cancel<'_>,
    ) -> Result<WanVideo, GenerateError<S::Error>> {
        let mut workspace = Workspace::new(
            Arc::clone(&self.backend),
            Arc::clone(self.residency.arbiter()),
        );
        let (latent, f_lat, h_lat, w_lat) = self
            .denoise_with(params, None, &mut workspace, None, progress, cancel)
            .await?;

        if let Some(p) = progress {
            p(ProgressEvent::VaeDecode);
        }
        let frames = self
            .decode_video(vae, &mut workspace, &latent, f_lat, h_lat, w_lat, None)
            .await?;
        // Phase boundary: nothing stays resident between generates.
        self.residency.evict_all_and_free(&*self.backend);

        let num_frames = if f_lat == 0 {
            0
        } else {
            TEMPORAL_SCALE * f_lat - 3
        };
        Ok(WanVideo {
            frames,
            num_frames,
            height: h_lat * self.vae_scale,
            width: w_lat * self.vae_scale,
        })
    }

    /// VAE-decode a pre-VAE latent (CTHW `[16, f_lat, h_lat, w_lat]`) to video
    /// frames. Mirrors the post-denoise half of `generate` without re-running
    /// the loop; the caller owns `workspace`.
    pub async fn decode_latent_to_video(
        &self,
        latent: &[f32],
        f_lat: usize,
        h_lat: usize,
        w_lat: usize,
        vae: VaeChoice,
        workspace: &mut Workspace<WgpuBackend>,
    ) -> Result<Vec<f32>, GenerateError<S::Error>> {
        self.decode_video(vae, workspace, latent, f_lat, h_lat, w_lat, None)
            .await
    }

    /// [`decode_latent_to_video`] with an explicit tiny-decoder chunk size (the
    /// `video_e2e` tiling-equivalence test forces single- vs multi-chunk to
    /// prove the carry path is exact). `None` is the prod budget-derived size.
    pub async fn decode_latent_to_video_chunked(
        &self,
        latent: &[f32],
        f_lat: usize,
        h_lat: usize,
        w_lat: usize,
        vae: VaeChoice,
        workspace: &mut Workspace<WgpuBackend>,
        tiny_chunk: Option<usize>,
    ) -> Result<Vec<f32>, GenerateError<S::Error>> {
        self.decode_video(vae, workspace, latent, f_lat, h_lat, w_lat, tiny_chunk)
            .await
    }

    /// LongLive-2.0-5B AR generate: the autoregressive/causal long-video path.
    /// Same backbone + umT5 + VAE as [`Self::generate`]; the denoise is the AR
    /// chunk loop ([`Self::denoise_ar`]) instead of the FastWan DMD loop. Load the
    /// model from a `LongLiveSource` (the `.pt` DiT renamed to canonical) for the
    /// causal weights; the geometry is identical to FastWan so nothing else here
    /// changes.
    ///
    /// `shots`: multi-shot scene plan. Empty = single-shot long video from
    /// `params.prompt` (the parity path). Non-empty = a prompt per shot with a
    /// scene cut at each shot boundary (see [`Shot`] / [`WanModel::denoise_ar`]).
    pub async fn generate_ar(
        &self,
        params: &GenerationParams,
        shots: &[Shot],
        vae: VaeChoice,
        progress: ProgressFn<'_>,
    ) -> Result<WanVideo, GenerateError<S::Error>> {
        let mut workspace = Workspace::new(
            Arc::clone(&self.backend),
            Arc::clone(self.residency.arbiter()),
        );
        let (latent, f_lat, h_lat, w_lat) = self
            .denoise_ar(
                params,
                shots,
                None,
                &mut workspace,
                None,
                None,
                None,
                progress,
            )
            .await?;

        if let Some(p) = progress {
            p(ProgressEvent::VaeDecode);
        }
        let frames = self
            .decode_video(vae, &mut workspace, &latent, f_lat, h_lat, w_lat, None)
            .await?;
        self.residency.evict_all_and_free(&*self.backend);

        let num_frames = if f_lat == 0 {
            0
        } else {
            TEMPORAL_SCALE * f_lat - 3
        };
        Ok(WanVideo {
            frames,
            num_frames,
            height: h_lat * self.vae_scale,
            width: w_lat * self.vae_scale,
        })
    }

    /// AR (LongLive) denoise: tokenize -> umT5 encode -> per-chunk FlowUniPC
    /// denoise over a windowed KV cache -> assembled pre-VAE latent (CTHW f32).
    /// Each chunk runs 4 UniPC steps over `[committed window ++ this chunk]`, then
    /// a clean-context timestep-0 pass commits the chunk's K/V into the cache for
    /// future chunks (the streaming core). `f_lat` must be a multiple of the chunk
    /// size (`num_frame_per_block`, 8). `initial_noise`: full `[C, f_lat, h, w]`
    /// noise used verbatim when `Some` (parity byte-load), else seed-derived.
    ///
    /// `shots`: multi-shot plan. Empty = one shot spanning all chunks with
    /// `params.prompt` (bit-identical to the single-prompt parity path). Non-empty
    /// = `Shot{prompt, chunks}` list whose chunk counts sum to the run's total;
    /// the first chunk of every shot after the first carries [`SCENE_CUT_PREFIX`]
    /// into umT5 and, in the cache, advances the multi-shot RoPE phase before the
    /// chunk and pins the chunk as the new shot's attention sink after its clean
    /// commit (release `shot_clean_recache=False`, so no cache zeroing). Mirrors
    /// upstream `causal_diffusion_inference.py` (shot-boundary branch) +
    /// `MultiTextConcatDataset._apply_shot_durations`.
    #[allow(clippy::too_many_arguments)]
    pub async fn denoise_ar(
        &self,
        params: &GenerationParams,
        shots: &[Shot],
        initial_noise: Option<&[f32]>,
        workspace: &mut Workspace<WgpuBackend>,
        mut chunk_diag: Option<&mut Vec<Vec<f32>>>,
        mut vel_diag: Option<&mut Vec<Vec<f32>>>,
        mut block_res_diag: Option<&mut Vec<Vec<f32>>>,
        progress: ProgressFn<'_>,
    ) -> Result<(Vec<f32>, usize, usize, usize), GenerateError<S::Error>> {
        if let Some(sink) = chunk_diag.as_deref_mut() {
            sink.clear();
        }
        if let Some(sink) = vel_diag.as_deref_mut() {
            sink.clear();
        }
        if let Some(sink) = block_res_diag.as_deref_mut() {
            sink.clear();
        }
        let div = (self.vae_scale * dit_config::PATCH_H) as u32;
        if params.height == 0
            || params.width == 0
            || !params.height.is_multiple_of(div)
            || !params.width.is_multiple_of(div)
        {
            return Err(GenerateError::InvalidDims {
                height: params.height,
                width: params.width,
            });
        }
        if params.num_frames == 0 || params.num_frames % TEMPORAL_SCALE as u32 != 1 {
            return Err(GenerateError::InvalidFrames {
                num_frames: params.num_frames,
            });
        }
        let h_lat = (params.height as usize) / self.vae_scale;
        let w_lat = (params.width as usize) / self.vae_scale;
        let f_lat = (params.num_frames as usize - 1) / TEMPORAL_SCALE + 1;

        let _denoise = trace::scope!("denoise_ar").entered();

        // --- AR geometry (needed before the shot plan: chunk count) ---
        let act_bytes = self.dit.block.act_dtype.bytes_per_elem() as usize;
        let frame_seq_len = (h_lat / dit_config::PATCH_H) * (w_lat / dit_config::PATCH_W);
        let kv_cfg = KvCacheConfig::longlive_runtime(
            self.cfg.num_layers,
            frame_seq_len,
            self.cfg.inner(),
            act_bytes,
        );
        let chunk_frames = kv_cfg.chunk_frames();
        if !f_lat.is_multiple_of(chunk_frames) {
            // The release T2V path requires the latent frame count to tile the
            // chunk size exactly (no independent-first-frame here).
            return Err(GenerateError::InvalidFrames {
                num_frames: params.num_frames,
            });
        }
        let num_chunks = f_lat / chunk_frames;

        // --- 1. shot plan: per-chunk prompt text + boundary flags ---
        // Build the `num_chunks`-long block-prompt list exactly as upstream
        // `MultiTextConcatDataset._apply_shot_durations`: each shot contributes
        // `chunks` consecutive entries; the first entry of every shot after the
        // first is `SCENE_CUT_PREFIX + caption`, the rest are the plain caption.
        // Empty `shots` collapses to one shot over all chunks => the single
        // prompt is reused unchanged (the parity path).
        let plan_shots: Vec<Shot> = if shots.is_empty() {
            vec![Shot {
                prompt: params.prompt.clone(),
                chunks: num_chunks,
            }]
        } else {
            shots.to_vec()
        };
        let total_shot_chunks: usize = plan_shots.iter().map(|s| s.chunks).sum();
        if total_shot_chunks != num_chunks || plan_shots.iter().any(|s| s.chunks == 0) {
            return Err(GenerateError::InvalidShots {
                shot_chunks: total_shot_chunks,
                num_chunks,
            });
        }
        let mut chunk_prompt: Vec<String> = Vec::with_capacity(num_chunks);
        let mut chunk_is_boundary: Vec<bool> = Vec::with_capacity(num_chunks);
        for (shot_idx, shot) in plan_shots.iter().enumerate() {
            for block_in_shot in 0..shot.chunks {
                let boundary = shot_idx > 0 && block_in_shot == 0;
                chunk_prompt.push(if boundary {
                    format!("{SCENE_CUT_PREFIX}{}", shot.prompt)
                } else {
                    shot.prompt.clone()
                });
                chunk_is_boundary.push(boundary);
            }
        }

        // --- 2. tokenize + umT5 encode each UNIQUE block prompt once ---
        if let Some(p) = progress {
            p(ProgressEvent::TextEncode);
        }
        let mut text_by_prompt: HashMap<&str, Arc<Vec<f32>>> = HashMap::new();
        for prompt in &chunk_prompt {
            if text_by_prompt.contains_key(prompt.as_str()) {
                continue;
            }
            let token_ids = self
                .tokenizer
                .encode(prompt, true)
                .map_err(GenerateError::Tokenizer)?;
            if token_ids.len() > MAX_PROMPT_TOKENS {
                return Err(GenerateError::PromptTooLong {
                    tokens: token_ids.len(),
                    max: MAX_PROMPT_TOKENS,
                });
            }
            let qout = self
                .umt5
                .forward(
                    &self.backend,
                    &self.umt5_pipelines,
                    &self.residency,
                    &*workspace,
                    &self.umt5_handles,
                    self.residency.source(),
                    &token_ids,
                )
                .await?;
            let padded = pad_text(&qout.hidden, qout.seq, dit_config::TEXT_DIM, TEXT_SEQ);
            text_by_prompt.insert(prompt.as_str(), Arc::new(padded));
        }
        // Per-chunk text handles (cheap Arc clones; one encoded tensor per unique
        // prompt, shared across the chunks that use it).
        let chunk_text: Vec<Arc<Vec<f32>>> = chunk_prompt
            .iter()
            .map(|p| Arc::clone(&text_by_prompt[p.as_str()]))
            .collect();
        self.residency.evict_all_and_free(&*self.backend);
        workspace.drain_pool();

        // --- 3. windowed KV cache (host-resident) ---
        let mut cache = KvWindowCache::new(kv_cfg);
        let mut store = RamKvStore::new(kv_cfg.num_layers, kv_cfg.bytes_per_layer());

        // --- 4. initial noise [C, f_lat, h, w] ---
        let n_lat = self.cfg.in_channels * f_lat * h_lat * w_lat;
        let full_noise: Vec<f32> = match initial_noise {
            Some(buf) => {
                assert_eq!(
                    buf.len(),
                    n_lat,
                    "initial_noise len {} != {n_lat}",
                    buf.len()
                );
                buf.to_vec()
            }
            None => gaussian_noise(n_lat, params.seed),
        };

        // --- 5. AR chunk loop ---
        let chunk_shape =
            WanDitShape::new(self.cfg.in_channels, chunk_frames, h_lat, w_lat, TEXT_SEQ);
        let dit = WanDit::assemble(self.dit_handles.clone(), chunk_shape, self.cfg);
        let mut unipc = FlowUniPc::new(&UniPcConfig::longlive());
        let n_steps = unipc.num_steps();

        self.residency.set_transient_reserve(
            self.residency.vram_staging_reserve_bytes() + DIT_WORKSPACE_RESERVE,
        );

        let c = self.cfg.in_channels;
        let hw = h_lat * w_lat;
        let chunk_len = c * chunk_frames * hw;
        let num_layers = self.cfg.num_layers;
        let mut output = vec![0.0_f32; n_lat];

        for chunk in 0..num_chunks {
            let _cs = trace::scope!("ar_chunk", chunk = chunk).entered();
            let f0 = chunk * chunk_frames;
            let current_start = chunk * chunk_frames * frame_seq_len;
            // Scene cut: advance the multi-shot RoPE temporal phase BEFORE the
            // window is planned, so this chunk's q/k rotate at the new shot's
            // offset (`begin_chunk` folds the cached `temporal_offset` into the
            // returned plan). Release order mirrors upstream: advance, denoise,
            // clean recache, pin.
            if chunk_is_boundary[chunk] {
                cache.advance_shot();
            }
            let plan = cache.begin_chunk(&mut store, current_start, chunk_frames);
            let text = chunk_text[chunk].as_slice();

            let mut latents = slice_chunk(&full_noise, c, f_lat, f0, chunk_frames, hw);

            unipc.reset();
            for step in 0..n_steps {
                if let Some(p) = progress {
                    p(ProgressEvent::ChunkStep {
                        chunk: chunk as u32 + 1,
                        num_chunks: num_chunks as u32,
                        step: step as u32 + 1,
                        num_steps: n_steps as u32,
                    });
                }
                let inputs = WanDitInputs {
                    image: &latents,
                    text,
                    timestep: unipc.timestep(step),
                    // AR self-attention uses its own windowed KV cache, not the
                    // dense self-SDPA window.
                    attn_window: None,
                };
                // Per-block residual taps only on the very first forward (chunk 0,
                // step 0): isolates where the AR forward first diverges from the
                // pyref without flooding readbacks across every step.
                let bd = if chunk == 0 && step == 0 {
                    block_res_diag.as_deref_mut()
                } else {
                    None
                };
                let out = dit
                    .forward_ar(
                        &self.backend,
                        &self.dit,
                        &self.residency,
                        &*workspace,
                        &inputs,
                        &store,
                        &plan,
                        None,
                        bd,
                    )
                    .await?;
                // Raw DiT velocity (CTHW [c, chunk_frames, hw]) before the
                // scheduler step; matches the pyref `py_c{c}_s{s}_vel` dump and
                // isolates the DiT forward from the UniPC sampler.
                if let Some(sink) = vel_diag.as_deref_mut() {
                    sink.push(out.image.clone());
                }
                latents = unipc.step(&out.image, &latents);
            }

            // Record the denoised chunk, then run the clean-context recache pass
            // (timestep 0) whose K/V are committed into the cache tail.
            scatter_chunk(&mut output, &latents, c, f_lat, f0, chunk_frames, hw);
            // Per-chunk post latent (CTHW [c, chunk_frames, hw]) for parity
            // localization; matches the pyref `py_chunk{c}_post` dump.
            if let Some(sink) = chunk_diag.as_deref_mut() {
                sink.push(latents.clone());
            }

            let mut k_commit: Vec<Vec<u8>> = vec![Vec::new(); num_layers];
            let mut v_commit: Vec<Vec<u8>> = vec![Vec::new(); num_layers];
            let clean_inputs = WanDitInputs {
                image: &latents,
                text,
                timestep: 0.0,
                attn_window: None,
            };
            dit.forward_ar(
                &self.backend,
                &self.dit,
                &self.residency,
                &*workspace,
                &clean_inputs,
                &store,
                &plan,
                Some((&mut k_commit, &mut v_commit)),
                None,
            )
            .await?;
            let k_refs: Vec<&[u8]> = k_commit.iter().map(|v| v.as_slice()).collect();
            let v_refs: Vec<&[u8]> = v_commit.iter().map(|v| v.as_slice()).collect();
            cache.commit_chunk(&mut store, &plan, &k_refs, &v_refs);

            // Scene cut: pin this just-committed chunk as the new shot's
            // attention sink (release `multi_shot_sink=true`, `sink_size>0`, so a
            // shot boundary is always a scene cut). The next roll keeps it and
            // relocates rolling data around it; no bytes move here.
            if chunk_is_boundary[chunk] {
                cache.pin_current_chunk(chunk_frames);
            }

            debug_assert_eq!(chunk_len, latents.len());
        }

        self.residency.evict_all_and_free(&*self.backend);
        workspace.drain_pool();
        Ok((output, f_lat, h_lat, w_lat))
    }

    /// Tokenize -> umT5 encode -> DMD few-step denoise loop. Returns the final
    /// pre-VAE latent (CTHW f32, `16 * f_lat * h_lat * w_lat`) plus the latent
    /// dims. Caller owns `workspace` so the GPU pool survives the DiT->VAE seam.
    ///
    /// `initial_noise`: used verbatim as the starting latent when `Some` (e2e
    /// pinned-noise byte-load), else derived from `seed`. `step_diag`: when
    /// `Some`, a [`WanStepDiag`] per step is pushed (cleared on entry) with the
    /// velocity, sigma, post-step latent, and per-block residual; the final
    /// entry's `post` equals the returned latent. `None` is the prod path (no
    /// GPU readbacks, plain `forward`/`step`).
    #[allow(clippy::too_many_arguments)]
    pub async fn denoise_with(
        &self,
        params: &GenerationParams,
        initial_noise: Option<&[f32]>,
        workspace: &mut Workspace<WgpuBackend>,
        mut step_diag: Option<&mut Vec<WanStepDiag>>,
        progress: ProgressFn<'_>,
        cancel: Cancel<'_>,
    ) -> Result<(Vec<f32>, usize, usize, usize), GenerateError<S::Error>> {
        if let Some(sink) = step_diag.as_deref_mut() {
            sink.clear();
        }
        let div = (self.vae_scale * dit_config::PATCH_H) as u32;
        if params.height == 0
            || params.width == 0
            || !params.height.is_multiple_of(div)
            || !params.width.is_multiple_of(div)
        {
            return Err(GenerateError::InvalidDims {
                height: params.height,
                width: params.width,
            });
        }
        // num_frames must be 4k+1 so the causal VAE encode/decode grid lines up.
        if params.num_frames == 0 || params.num_frames % TEMPORAL_SCALE as u32 != 1 {
            return Err(GenerateError::InvalidFrames {
                num_frames: params.num_frames,
            });
        }
        let h_lat = (params.height as usize) / self.vae_scale;
        let w_lat = (params.width as usize) / self.vae_scale;
        let f_lat = (params.num_frames as usize - 1) / TEMPORAL_SCALE + 1;

        let _denoise = trace::scope!("denoise").entered();

        // --- 1. tokenize ---
        let token_ids = {
            let _s = trace::scope!("tokenize").entered();
            // umT5 needs the trailing `</s>` EOS the diffusers reference appends
            // (add_special_tokens=True). Omitting it shifts every token's
            // bidirectional attention and compounds across layers.
            let ids = self
                .tokenizer
                .encode(&params.prompt, true)
                .map_err(GenerateError::Tokenizer)?;
            if ids.len() > MAX_PROMPT_TOKENS {
                return Err(GenerateError::PromptTooLong {
                    tokens: ids.len(),
                    max: MAX_PROMPT_TOKENS,
                });
            }
            ids
        };

        // --- 2. umT5 encode -> text states, padded to the DiT context ---
        // Encoded while umT5 is resident (one phase), then the weights are
        // evicted; the padded host tensor lives through the loop. CFG-free, so a
        // single conditional pass (no negative prompt).
        if let Some(p) = progress {
            p(ProgressEvent::TextEncode);
        }
        let text = {
            let qout = self
                .umt5
                .forward(
                    &self.backend,
                    &self.umt5_pipelines,
                    &self.residency,
                    &*workspace,
                    &self.umt5_handles,
                    self.residency.source(),
                    &token_ids,
                )
                .await?;
            // Diffusers pads `prompt_embeds` to `max_sequence_length` (512) and
            // the DiT cross-attends over the full padded context (no mask). Pad
            // with zeros / truncate to TEXT_SEQ rows. The e2e must feed pyref the
            // same padded context. (Cross-attn masking is deferred -- verify at
            // e2e.)
            pad_text(&qout.hidden, qout.seq, dit_config::TEXT_DIM, TEXT_SEQ)
        };

        // Phase boundary: umT5 weights are dead for the rest of the call.
        self.residency.evict_all_and_free(&*self.backend);
        workspace.drain_pool();

        // --- 3. initial noise [z_dim, f_lat, h_lat, w_lat] ---
        let n_lat = self.cfg.in_channels * f_lat * h_lat * w_lat;
        let mut sample: Vec<f32> = match initial_noise {
            Some(buf) => {
                assert_eq!(
                    buf.len(),
                    n_lat,
                    "initial_noise len {} != {n_lat}",
                    buf.len()
                );
                buf.to_vec()
            }
            None => gaussian_noise(n_lat, params.seed),
        };

        // --- 4. DiT denoise (CFG-free). The DiT forward is byte-identical for
        // both samplers; only the schedule around it differs. UniPC (the served
        // default) mirrors the public FastWan Spaces; DMD is the parity path. ---
        let shape = || WanDitShape::new(self.cfg.in_channels, f_lat, h_lat, w_lat, TEXT_SEQ);

        // Hold the DiT forward's full non-weight working set free on every weight
        // admission, so weight residency only ever fills `budget - working_set`
        // and the activation+dequant peak stays UNDER budget instead of spilling
        // into device slack. (The old fixed `DIT_WORKSPACE_RESERVE` pad was
        // calibrated at the tiny e2e dims -- ~64 MiB -- so at video dims the
        // ~1 GiB of pooled activation + Q8->bf16 dequant scratch overflowed the
        // budget into the physical card.)
        //
        // The reserve is MAX-of-phase, not sum-of-phases. The self-attention and
        // FFN phases never coexist (q/k/v are freed before the FFN runs), so
        // reserving `rows*(dff + 8*dim)` -- the OLD formula -- double-counted both
        // and, at a long clip (81f @ 832x480 = 32760 rows), held ~4GB free per
        // weight admission. That left <1GB for the 14B weight cache and the DiT
        // thrashed on constant re-streaming. Now the FFN is row-tiled
        // (`dit_block::FFN_TILE_ROWS`), so its `rows*ffn_dim` intermediate is
        // bounded to a tile; the residency-visible peak is whichever phase is
        // larger plus the weight dequant scratch:
        //   - self-attn / residual: ~6x the rows*dim residual (x + qr/kr/v/sa,
        //     the q/k/qn/kn intermediates reuse the pool), scales with the grid;
        //   - tiled FFN: TILE rows * (2*ffn_dim + 2*dim), constant in clip length;
        //   - weight dequant: ~4x the FFN weight (dim*ffn_dim), the largest site.
        // Generous within each phase on purpose (overflow PREVENTED up front; a
        // rare residual peak rides soft budget mode), but no longer quadratic in
        // practice. Set once for the whole step loop.
        let rows = shape().n_tok as u64;
        let dim = self.cfg.inner() as u64;
        let dff = self.cfg.ffn_dim as u64;
        let tile = (super::dit_block::FFN_TILE_ROWS as u64).min(rows.max(1));
        let bpe = self.dit.block.act_bytes(1);
        let attn_phase = rows * 6 * dim * bpe;
        let ffn_phase = rows * 2 * dim * bpe + tile * 2 * dff * bpe;
        let dit_workspace = attn_phase.max(ffn_phase) + 4 * dim * dff * bpe;
        self.residency.set_transient_reserve(
            PREFETCH_STAGING_DEPTH * self.residency.vram_staging_reserve_bytes() + dit_workspace,
        );

        // --- Wan2.2-A14B MoE step-distill path (two experts). 4-step flow-Euler,
        // CFG-free; the high-noise expert runs the early steps, the low-noise
        // expert the rest. At the switch, evict the high expert's streamed blocks
        // so the low expert streams into freed VRAM (the experts never overlap in
        // residency). `params.sampler` is ignored here (the schedule is fixed). ---
        if let Some(low_handles) = self.dit_handles_low.clone() {
            let high = WanDit::assemble(self.dit_handles.clone(), shape(), self.cfg);
            let low = WanDit::assemble(low_handles, shape(), self.cfg);
            let sampler = Wan22DistillSampler::new(&Wan22DistillConfig::wan22_t2v_a14b());
            let n_steps = sampler.num_steps();
            for i in 0..n_steps {
                let _step = trace::scope!("step", i = i).entered();
                if cancel.is_some_and(|c| c()) {
                    return Err(GenerateError::Cancelled);
                }
                if let Some(p) = progress {
                    p(ProgressEvent::Step {
                        i: i as u32 + 1,
                        n: n_steps as u32,
                    });
                }
                if i > 0 && sampler.is_high_noise(i - 1) && !sampler.is_high_noise(i) {
                    self.residency.evict_all_and_free(&*self.backend);
                    workspace.drain_pool();
                }
                let dit = if sampler.is_high_noise(i) {
                    &high
                } else {
                    &low
                };
                let inputs = WanDitInputs {
                    image: &sample,
                    text: &text,
                    timestep: sampler.timestep(i),
                    attn_window: params.attn_window,
                };
                let out = dit
                    .forward(
                        &self.backend,
                        &self.dit,
                        &self.residency,
                        &*workspace,
                        &inputs,
                    )
                    .await?;
                sample = sampler.step(i, &out.image, &sample);
            }
            self.residency.evict_all_and_free(&*self.backend);
            workspace.drain_pool();
            return Ok((sample, f_lat, h_lat, w_lat));
        }

        let dit = WanDit::assemble(self.dit_handles.clone(), shape(), self.cfg);
        match params.sampler {
            // --- DMD: one DiT forward per fixed timestep, renoise between. The
            // step-diag taps live here (the parity/bisection path is DMD-only). ---
            VideoSampler::Dmd => {
                let sampler = DmdSampler::new(&DmdConfig::fastwan_ti2v_5b());
                let n_steps = sampler.num_steps();
                for i in 0..n_steps {
                    let _step = trace::scope!("step", i = i).entered();
                    if cancel.is_some_and(|c| c()) {
                        return Err(GenerateError::Cancelled);
                    }
                    if let Some(p) = progress {
                        p(ProgressEvent::Step {
                            i: i as u32 + 1,
                            n: n_steps as u32,
                        });
                    }
                    let t = sampler.timestep(i);
                    let inputs = WanDitInputs {
                        image: &sample,
                        text: &text,
                        timestep: t,
                        attn_window: params.attn_window,
                    };
                    // Diag path captures per-block + final-stage taps via forward_with_taps;
                    // prod takes the plain forward.
                    let (velocity, per_block, temb, timestep_proj, final_norm, proj_out) =
                        if step_diag.is_some() {
                            let mut per_block = Vec::new();
                            let (mut temb, mut timestep_proj, mut final_norm, mut proj_out) =
                                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                            let taps = WanDitTaps {
                                per_block: Some(&mut per_block),
                                temb: Some(&mut temb),
                                timestep_proj: Some(&mut timestep_proj),
                                final_norm: Some(&mut final_norm),
                                proj_out: Some(&mut proj_out),
                                ..Default::default()
                            };
                            let out = dit
                                .forward_with_taps(
                                    &self.backend,
                                    &self.dit,
                                    &self.residency,
                                    &*workspace,
                                    &inputs,
                                    taps,
                                )
                                .await?;
                            (
                                out.image,
                                per_block,
                                temb,
                                timestep_proj,
                                final_norm,
                                proj_out,
                            )
                        } else {
                            let out = dit
                                .forward(
                                    &self.backend,
                                    &self.dit,
                                    &self.residency,
                                    &*workspace,
                                    &inputs,
                                )
                                .await?;
                            (
                                out.image,
                                Vec::new(),
                                Vec::new(),
                                Vec::new(),
                                Vec::new(),
                                Vec::new(),
                            )
                        };
                    // DMD: convert the predicted flow velocity to x0 and renoise to the
                    // next fixed timestep (the final step returns x0 unchanged). The
                    // renoise Gaussian is independent per step, seeded deterministically.
                    let noise = match sampler.noise_len(i, n_lat) {
                        0 => Vec::new(),
                        len => gaussian_noise(len, renoise_seed(params.seed, i)),
                    };
                    sample = sampler.step(i, &velocity, &sample, &noise);
                    if let Some(sink) = step_diag.as_deref_mut() {
                        sink.push(WanStepDiag {
                            timestep: t,
                            sigma: sampler.sigma(i),
                            velocity, // moved; the flow velocity fed to the sampler
                            post: sample.clone(),
                            per_block,
                            temb,
                            timestep_proj,
                            final_norm,
                            proj_out,
                        });
                    }
                }
            }
            // --- UniPC multistep (flow_shift=8.0, CFG-free): feed each step's DiT
            // velocity into the FlowUniPc solver; no renoise, no guidance. Same DiT
            // forward as DMD. Not a parity path, so no step-diag taps. ---
            VideoSampler::UniPc { steps } => {
                let mut unipc = FlowUniPc::new(&UniPcConfig::fastwan(steps));
                let n_steps = unipc.num_steps();
                for i in 0..n_steps {
                    let _step = trace::scope!("step", i = i).entered();
                    if cancel.is_some_and(|c| c()) {
                        return Err(GenerateError::Cancelled);
                    }
                    if let Some(p) = progress {
                        p(ProgressEvent::Step {
                            i: i as u32 + 1,
                            n: n_steps as u32,
                        });
                    }
                    let inputs = WanDitInputs {
                        image: &sample,
                        text: &text,
                        timestep: unipc.timestep(i),
                        attn_window: params.attn_window,
                    };
                    let out = dit
                        .forward(
                            &self.backend,
                            &self.dit,
                            &self.residency,
                            &*workspace,
                            &inputs,
                        )
                        .await?;
                    sample = unipc.step(&out.image, &sample);
                }
            }
        }

        // Phase boundary: DiT weights dead until VAE.
        self.residency.evict_all_and_free(&*self.backend);
        workspace.drain_pool();
        Ok((sample, f_lat, h_lat, w_lat))
    }

    /// Bringup diagnostic: tokenize -> umT5 -> ONE DiT forward (step 0) with all
    /// Vec-based per-stage taps, returning owned readbacks. Localizes a numerical
    /// blowup to a stage without a Python reference. Not a committed path.
    pub async fn diag_step0(
        &self,
        params: &GenerationParams,
        initial_noise: &[f32],
        workspace: &mut Workspace<WgpuBackend>,
    ) -> Result<WanStep0Diag, GenerateError<S::Error>> {
        self.diag_step_at(params, initial_noise, 0, 0, workspace)
            .await
    }

    /// Bringup diagnostic: like [`Self::diag_step0`] but at an arbitrary DMD
    /// `step_index` on an externally supplied `latent` (the reference's input to
    /// that step, drift-stripped). The per-stage taps + their pyref dumps are
    /// step-agnostic, so pointing this at the FIRST divergent step localizes a
    /// timestep-specific divergence (e.g. a t=757 modulation underscale) to a
    /// stage. `step_index == 0` + `initial_noise` reproduces `diag_step0`. Not a
    /// committed path.
    pub async fn diag_step_at(
        &self,
        params: &GenerationParams,
        latent: &[f32],
        step_index: usize,
        tap_block: usize,
        workspace: &mut Workspace<WgpuBackend>,
    ) -> Result<WanStep0Diag, GenerateError<S::Error>> {
        let h_lat = (params.height as usize) / self.vae_scale;
        let w_lat = (params.width as usize) / self.vae_scale;
        let f_lat = (params.num_frames as usize - 1) / TEMPORAL_SCALE + 1;

        let token_ids = self
            .tokenizer
            .encode(&params.prompt, true) // umT5: append `</s>` EOS (see denoise_with)
            .map_err(GenerateError::Tokenizer)?;
        // Full umT5 telemetry: gathered embeds, every block's post-residual
        // output, and per-op intermediates of block 0. One run traces the
        // magnitude through the whole encoder so a clean scale (e.g. the
        // ~0.365x text-output bug) is localized to embeds vs a specific layer
        // vs the final RMSNorm.
        let mut umt5_taps = Umt5Taps {
            want_layer_outputs: true,
            want_block_ops: true,
            ..Default::default()
        };
        let qout = self
            .umt5
            .forward_taps(
                &self.backend,
                &self.umt5_pipelines,
                &self.residency,
                &*workspace,
                &self.umt5_handles,
                self.residency.source(),
                &token_ids,
                Some(&mut umt5_taps),
            )
            .await?;
        let umt5_hidden = qout.hidden.clone();
        let umt5_seq = qout.seq;
        let umt5_embeds = std::mem::take(&mut umt5_taps.embeds);
        let umt5_layer_outputs = std::mem::take(&mut umt5_taps.layer_outputs);
        let umt5_block_ops = std::mem::take(&mut umt5_taps.block_ops);
        let text = pad_text(&qout.hidden, qout.seq, dit_config::TEXT_DIM, TEXT_SEQ);
        self.residency.evict_all_and_free(&*self.backend);
        workspace.drain_pool();

        let n_lat = self.cfg.in_channels * f_lat * h_lat * w_lat;
        assert_eq!(latent.len(), n_lat, "diag_step_at latent len");
        let sample = latent.to_vec();

        let shape = WanDitShape::new(self.cfg.in_channels, f_lat, h_lat, w_lat, TEXT_SEQ);
        let dit = WanDit::assemble(self.dit_handles.clone(), shape, self.cfg);
        let sampler = DmdSampler::new(&DmdConfig::fastwan_ti2v_5b());
        let t = sampler.timestep(step_index);
        let inputs = WanDitInputs {
            image: &sample,
            text: &text,
            timestep: t,
            // Parity/bisection diagnostic: always full attention.
            attn_window: None,
        };

        // Per-op sinks for block `tap_block`: the driver only fills these GPU
        // buffers; the caller allocates them (persistent) and reads them back
        // after forward. Sized rows*inner except ffn_gelu (rows*ffn_dim).
        let bp = &self.dit.block;
        let rows = shape.n_tok as u32;
        let inner = self.cfg.inner() as u32;
        let ffn = self.cfg.ffn_dim as u32;
        let inner_buf = || workspace.alloc(bp.act_bytes(rows * inner));
        let vec_buf = || workspace.alloc(bp.act_bytes(inner));
        let b_norm1_premod = inner_buf()?;
        let b_mod_scale = vec_buf()?;
        let b_mod_shift = vec_buf()?;
        let b_norm1 = inner_buf()?;
        let b_self_q = inner_buf()?;
        let b_self_k = inner_buf()?;
        let b_self_v = inner_buf()?;
        let b_self_sa = inner_buf()?;
        let b_after_self = inner_buf()?;
        let b_norm2 = inner_buf()?;
        let b_cross_sa = inner_buf()?;
        let b_after_cross = inner_buf()?;
        let b_norm3 = inner_buf()?;
        let b_ffn_gelu = workspace.alloc(bp.act_bytes(rows * ffn))?;
        let b_ffn_down = inner_buf()?;
        let block0 = WanDitBlockTaps {
            norm1_premod: Some(b_norm1_premod.as_buf_ref()),
            mod_scale: Some(b_mod_scale.as_buf_ref()),
            mod_shift: Some(b_mod_shift.as_buf_ref()),
            norm1: Some(b_norm1.as_buf_ref()),
            self_q: Some(b_self_q.as_buf_ref()),
            self_k: Some(b_self_k.as_buf_ref()),
            self_v: Some(b_self_v.as_buf_ref()),
            self_sa: Some(b_self_sa.as_buf_ref()),
            after_self: Some(b_after_self.as_buf_ref()),
            norm2: Some(b_norm2.as_buf_ref()),
            cross_sa: Some(b_cross_sa.as_buf_ref()),
            after_cross: Some(b_after_cross.as_buf_ref()),
            norm3: Some(b_norm3.as_buf_ref()),
            ffn_gelu: Some(b_ffn_gelu.as_buf_ref()),
            ffn_down: Some(b_ffn_down.as_buf_ref()),
        };

        let (mut patch_x, mut temb, mut timestep_proj, mut text_proj) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut per_block, mut final_norm, mut proj_out) = (Vec::new(), Vec::new(), Vec::new());
        let taps = WanDitTaps {
            patch_x: Some(&mut patch_x),
            temb: Some(&mut temb),
            timestep_proj: Some(&mut timestep_proj),
            text_proj: Some(&mut text_proj),
            per_block: Some(&mut per_block),
            block0: Some(block0),
            tap_block,
            final_norm: Some(&mut final_norm),
            proj_out: Some(&mut proj_out),
        };
        let out = dit
            .forward_with_taps(
                &self.backend,
                &self.dit,
                &self.residency,
                &*workspace,
                &inputs,
                taps,
            )
            .await?;
        // Renoise this step (the sampler returns plain x0 on the final step, and
        // `noise_len` is 0 there, so this is correct for any step_index).
        let noise = gaussian_noise(
            sampler.noise_len(step_index, n_lat),
            renoise_seed(params.seed, step_index),
        );
        let stepped = sampler.step(step_index, &out.image, &sample, &noise);

        // Read block-0 sinks back (in execution order) before draining the pool.
        let act = bp.act_dtype;
        let mut block0_stages: Vec<(String, Vec<f32>)> = Vec::new();
        for (name, buf, n) in [
            ("norm1_premod", &b_norm1_premod, rows * inner),
            ("mod_scale", &b_mod_scale, inner),
            ("mod_shift", &b_mod_shift, inner),
            ("norm1", &b_norm1, rows * inner),
            ("self_q", &b_self_q, rows * inner),
            ("self_k", &b_self_k, rows * inner),
            ("self_v", &b_self_v, rows * inner),
            ("self_sa", &b_self_sa, rows * inner),
            ("after_self", &b_after_self, rows * inner),
            ("norm2", &b_norm2, rows * inner),
            ("cross_sa", &b_cross_sa, rows * inner),
            ("after_cross", &b_after_cross, rows * inner),
            ("norm3", &b_norm3, rows * inner),
            ("ffn_gelu", &b_ffn_gelu, rows * ffn),
            ("ffn_down", &b_ffn_down, rows * inner),
        ] {
            let mut sink = Vec::new();
            read_into_f32(&self.backend, &buf.as_buf_ref(), n as usize, act, &mut sink).await?;
            block0_stages.push((name.to_string(), sink));
        }

        self.residency.evict_all_and_free(&*self.backend);
        workspace.drain_pool();
        Ok(WanStep0Diag {
            timestep: t,
            umt5_hidden,
            umt5_seq,
            umt5_embeds,
            umt5_layer_outputs,
            umt5_block_ops,
            patch_x,
            temb,
            timestep_proj,
            text_proj,
            block0_stages,
            per_block,
            final_norm,
            proj_out,
            dit_out: out.image,
            stepped,
        })
    }

    /// Bringup diagnostic: run ONE DiT forward at DMD `step_index`'s timestep on
    /// an externally supplied `latent`, returning the raw velocity. Lets the e2e
    /// feed the reference's per-step input back through our forward (identical
    /// input, no accumulated drift), isolating per-step forward error from the
    /// drift that compounds across the schedule. Not a committed path.
    pub async fn forward_velocity_at(
        &self,
        params: &GenerationParams,
        latent: &[f32],
        step_index: usize,
        workspace: &mut Workspace<WgpuBackend>,
    ) -> Result<Vec<f32>, GenerateError<S::Error>> {
        let h_lat = (params.height as usize) / self.vae_scale;
        let w_lat = (params.width as usize) / self.vae_scale;
        let f_lat = (params.num_frames as usize - 1) / TEMPORAL_SCALE + 1;
        let n_lat = self.cfg.in_channels * f_lat * h_lat * w_lat;
        assert_eq!(latent.len(), n_lat, "forward_velocity_at latent len");

        let token_ids = self
            .tokenizer
            .encode(&params.prompt, true)
            .map_err(GenerateError::Tokenizer)?;
        let qout = self
            .umt5
            .forward(
                &self.backend,
                &self.umt5_pipelines,
                &self.residency,
                &*workspace,
                &self.umt5_handles,
                self.residency.source(),
                &token_ids,
            )
            .await?;
        let text = pad_text(&qout.hidden, qout.seq, dit_config::TEXT_DIM, TEXT_SEQ);
        self.residency.evict_all_and_free(&*self.backend);
        workspace.drain_pool();

        let shape = WanDitShape::new(self.cfg.in_channels, f_lat, h_lat, w_lat, TEXT_SEQ);
        let dit = WanDit::assemble(self.dit_handles.clone(), shape, self.cfg);
        let sampler = DmdSampler::new(&DmdConfig::fastwan_ti2v_5b());
        let inputs = WanDitInputs {
            image: latent,
            text: &text,
            timestep: sampler.timestep(step_index),
            attn_window: None,
        };
        let out = dit
            .forward(
                &self.backend,
                &self.dit,
                &self.residency,
                &*workspace,
                &inputs,
            )
            .await?;
        self.residency.evict_all_and_free(&*self.backend);
        workspace.drain_pool();
        Ok(out.image)
    }
}

/// Owned per-stage readbacks from [`WanModel::diag_step0`] (bringup telemetry).
pub struct WanStep0Diag {
    pub timestep: f32,
    /// umT5 encoder output, post `final_layer_norm`, real tokens `[seq, D_MODEL]`.
    pub umt5_hidden: Vec<f32>,
    /// Real (unpadded) umT5 token count. The `umt5_embeds`/`umt5_layer_outputs`
    /// rows are even-padded to `seq_pad >= seq`; slice to `seq` rows to align
    /// with the pyref real-token rows.
    pub umt5_seq: usize,
    /// Gathered token embeddings fed into umT5 block 0 `[seq_pad, D_MODEL]`.
    pub umt5_embeds: Vec<f32>,
    /// Post-residual output of each umT5 block (`len == N_LAYERS`); the last
    /// entry is the input to `final_layer_norm`. Each `[seq_pad, D_MODEL]`.
    pub umt5_layer_outputs: Vec<Vec<f32>>,
    /// Per-op intermediates of EVERY umT5 block (`len == N_LAYERS`):
    /// norm/q/k/v/sa/proj/after_attn/n2/ffn stages, for localizing which op in
    /// which block first injects the divergent outlier channel.
    pub umt5_block_ops: Vec<Umt5BlockOpsHost>,
    pub patch_x: Vec<f32>,
    pub temb: Vec<f32>,
    pub timestep_proj: Vec<f32>,
    pub text_proj: Vec<f32>,
    /// Block-0 per-op readbacks, in execution order (norm1 .. ffn_down).
    pub block0_stages: Vec<(String, Vec<f32>)>,
    /// Residual stream after each block (`len == num_layers`).
    pub per_block: Vec<Vec<f32>>,
    pub final_norm: Vec<f32>,
    pub proj_out: Vec<f32>,
    /// Raw DiT model output (flow velocity) for step 0.
    pub dit_out: Vec<f32>,
    /// Scheduler output after consuming `dit_out` (= our_step0_prev_sample).
    pub stepped: Vec<f32>,
}

/// Probe a tensor's GPU matmul weight dtype from the source catalog.
fn probe_weight<S: WeightSource>(residency: &WeightResidency<S>, id: &str) -> WeightDtype {
    match residency
        .source()
        .catalog()
        .get(&WeightId(id.to_string()))
        .and_then(|e| e.encoding)
    {
        Some(StorageEncoding::Quant(k)) => WeightDtype::Quant(k),
        _ => WeightDtype::Bf16,
    }
}

/// Uniform `BlockWgslConfigs` for one submodel: every matmul slot takes
/// `weight_dtype`, the ops template carries the chosen act dtype + recipe.
/// Per-site weight-dtype override for the DP4A-safe matmul sites; `None` leaves a
/// site at the probed `weight_dtype`. Must agree with the loader's
/// [`WanI8Sites`] transcode so each matmul's pipeline encoding matches its
/// registered weight.
#[derive(Clone, Copy, Default)]
struct SiteOverride {
    /// Self-attention QKV (`matmul_qkv_self`). Cross-attention QKV
    /// (`matmul_qkv`) is never overridden: its K/V project from un-normed umT5
    /// text and overflow f16 under i8 acts, so it stays dense at `weight_dtype`.
    qkv_self: Option<WeightDtype>,
    ffn_up: Option<WeightDtype>,
    /// Sites that must use the dense (dequant-once -> bf16 matmul) path instead of
    /// i8 DP4A even when their weight is Quant. On a Q8 DiT (Wan2.2 GGUF, every
    /// site Quant) the cross-attn qkv, proj, and ffn_down acts are un-normed /
    /// outlier-heavy: i8 act_quant's f16 scale overflows -> inf. FastWan keeps
    /// these bf16 so the issue never arises; here we force the dense path.
    dense_acts: DenseActSites,
}

fn block_cfgs(
    weight_dtype: WeightDtype,
    act: ActDtype,
    ovr: SiteOverride,
    fast_sdpa: bool,
) -> BlockWgslConfigs {
    let ops = WgslConfig {
        bf16_quant_writes: RECIPE.bf16_quant_writes,
        act_dtype: act,
        weight_dtype: WeightDtype::Bf16,
    };
    let mm = WgslConfig {
        weight_dtype,
        ..ops
    };
    // Per-site overrides: self-attn qkv / ffn_up may run an i8/Q8_0 weight
    // encoding for the DP4A path while the rest of the block stays at
    // `weight_dtype`. Cross-attn qkv (`matmul_qkv`) is left at the base dtype.
    let mm_qkv_self = WgslConfig {
        weight_dtype: ovr.qkv_self.unwrap_or(weight_dtype),
        ..ops
    };
    let mm_ffn_up = WgslConfig {
        weight_dtype: ovr.ffn_up.unwrap_or(weight_dtype),
        ..ops
    };
    BlockWgslConfigs {
        matmul_qkv: mm,
        matmul_qkv_self: mm_qkv_self,
        matmul_proj: mm,
        matmul_ffn_up: mm_ffn_up,
        matmul_ffn_down: mm,
        matmul_adaln: ops,
        ops,
        i8_sdpa: false,
        dense_acts: ovr.dense_acts,
        // Coopmat (tensor-core) on the outlier-bound bf16 sites i8 can't take:
        // proj (attn-out) + ffn_down (gelu product). Only on the bf16 14B path
        // (fast_sdpa); honored only when the device exposes a coopmat config
        // (else the compile gate falls back). Cross-attn qkv stays OFF: its
        // un-normed umT5 text K/V overflow f16 (same reason it stays dense for
        // i8). Validate via wan22 5f e2e parity before widening.
        coopmat_acts: if fast_sdpa {
            CoopmatSites {
                proj: true,
                ffn_down: true,
                // Cross-attn qkv stays OFF coopmat: TESTED 2026-06-28 -> DEVICE
                // LOST at denoise step 1 (un-normed umT5 text K/V overflow f16,
                // as the i8 note predicted). It is the #1 remaining matmul (30.6
                // ms/disp), so it's the prize, but needs a CLAMPED bf16->f16
                // cast on the A-side + quality validation before it can ship.
                qkv: false,
            }
        } else {
            CoopmatSites::default()
        },
        large_d_sdpa: false,
        fast_sdpa,
        decode_sdpa: false,
    }
}

/// Extract chunk frames `[f0, f0 + cf)` from a CTHW latent `[c, f_lat, hw]` into
/// a contiguous `[c, cf, hw]` chunk. Per channel the frame block is contiguous,
/// so this is `c` slice copies.
fn slice_chunk(full: &[f32], c: usize, f_lat: usize, f0: usize, cf: usize, hw: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; c * cf * hw];
    for ch in 0..c {
        let src = ch * f_lat * hw + f0 * hw;
        let dst = ch * cf * hw;
        out[dst..dst + cf * hw].copy_from_slice(&full[src..src + cf * hw]);
    }
    out
}

/// Inverse of [`slice_chunk`]: write a `[c, cf, hw]` chunk back into frames
/// `[f0, f0 + cf)` of the CTHW latent `[c, f_lat, hw]`.
fn scatter_chunk(
    full: &mut [f32],
    chunk: &[f32],
    c: usize,
    f_lat: usize,
    f0: usize,
    cf: usize,
    hw: usize,
) {
    for ch in 0..c {
        let dst = ch * f_lat * hw + f0 * hw;
        let src = ch * cf * hw;
        full[dst..dst + cf * hw].copy_from_slice(&chunk[src..src + cf * hw]);
    }
}

/// Pad / truncate umT5 states `[seq, dim]` to `[rows, dim]` (zero pad).
fn pad_text(hidden: &[f32], seq: usize, dim: usize, rows: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * dim];
    let copy = seq.min(rows);
    out[..copy * dim].copy_from_slice(&hidden[..copy * dim]);
    out
}

/// Per-step renoise seed: an independent stream for step `i` derived from the
/// generation seed. DMD becomes byte-parity-friendly because this is
/// deterministic: the e2e dumps the exact per-step renoise tensor (via this same
/// fn + [`gaussian_noise`]) and the pyref byte-loads it, so both sides consume
/// identical renoise rather than each drawing from its own RNG.
pub fn renoise_seed(seed: u64, i: usize) -> u64 {
    seed.wrapping_add((i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// Deterministic standard-normal samples via SplitMix64 -> Box-Muller (avoids a
/// `rand` dep). Same generator as `z_image::pipeline::gaussian_noise`. `pub` so
/// the e2e parity test can reproduce the exact per-step renoise tensors the
/// denoise loop consumes and dump them for the pyref to byte-load.
pub fn gaussian_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next_u64 = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = ((next_u64() >> 11) as f64 + 1.0) * (1.0 / ((1u64 << 53) as f64 + 1.0));
        let u2 = (next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push((r * theta.cos()) as f32);
        if out.len() < n {
            out.push((r * theta.sin()) as f32);
        }
    }
    out
}
