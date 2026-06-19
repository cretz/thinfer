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

use crate::common::block::{BlockWgslConfigs, DenseActSites};
use crate::wan::dit::{
    LoadedWanDitHandles, WanDit, WanDitError, WanDitInputs, WanDitShape, WanDitTaps, read_into_f32,
};
use crate::wan::dit_block::{WanDitBlockTaps, WanDitConfig, WanDitPipelines, config as dit_config};
use crate::wan::loader::register_wan_dit_handles;
use crate::wan::manifest::RECIPE;
use crate::wan::scheduler::{DmdConfig, DmdSampler};
use crate::wan::umt5::{
    Umt5BlockOpsHost, Umt5Encoder, Umt5ForwardError, Umt5Handles, Umt5Pipelines, Umt5Taps,
    register_umt5_handles,
};
use crate::wan::vae::{
    VaeDecoderWeights, WanVaeConfig, WanVaeDecodeError, WanVaeDecoder, WanVaePipelines,
    register_decoder,
};

/// umT5 rope-free context cap. The Wan DiT cross-attends to a fixed
/// `max_sequence_length`; SkyReels-V2 ships 512.
const TEXT_SEQ: usize = dit_config::TEXT_SEQ;
const MAX_PROMPT_TOKENS: usize = TEXT_SEQ;
/// Wan2.2-TI2V high-compression VAE: 16x spatial, 4x temporal (the new module,
/// `wan/vae.rs`). The latent grid the DiT sees derives from these.
const VAE_SCALE: usize = 16;
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

/// Inputs to one `generate` call. The DMD distillation bakes the step schedule
/// and is CFG-free, so there is no `steps`, `guidance_scale`, or
/// `negative_prompt` knob (the abandoned SkyReels-V2-DF path had all three).
pub struct GenerationParams {
    pub prompt: String,
    /// Frame height in pixels. Divisible by `VAE_SCALE * PATCH_H` (32).
    pub height: u32,
    /// Frame width in pixels. Divisible by `VAE_SCALE * PATCH_W` (32).
    pub width: u32,
    /// Output frame count. Must be `4 * k + 1` (the causal-VAE temporal grid).
    pub num_frames: u32,
    /// Deterministic seed for the initial latent noise (and per-step renoise).
    pub seed: u64,
}

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
    Step { i: u32, n: u32 },
    VaeDecode,
}

pub type ProgressFn<'a> = Option<&'a dyn Fn(ProgressEvent)>;

pub struct WanModel<S: WeightSource, T: Tokenizer> {
    backend: Arc<WgpuBackend>,
    residency: WeightResidency<S>,
    tokenizer: T,
    umt5: Umt5Encoder,
    umt5_pipelines: Umt5Pipelines,
    umt5_handles: Umt5Handles,
    dit: WanDitPipelines,
    dit_handles: LoadedWanDitHandles,
    /// Per-variant DiT geometry (FastWan 5B today). Drives latent channels, the
    /// block dims, and the loader layer count.
    cfg: WanDitConfig,
    vae: WanVaeDecoder,
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
    Wgpu(WgpuError),
    Residency(ResidencyError<SE, WgpuError>),
    InvalidDims { height: u32, width: u32 },
    InvalidFrames { num_frames: u32 },
    PromptTooLong { tokens: usize, max: usize },
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

impl<S: WeightSource, T: Tokenizer> WanModel<S, T> {
    /// Build the model: register every umT5 + DiT + VAE handle with residency
    /// and compile every WGSL kernel once. No bytes flow until `generate`.
    pub async fn load(
        backend: Arc<WgpuBackend>,
        residency: WeightResidency<S>,
        tokenizer: T,
    ) -> Result<Self, ModelLoadError> {
        Self::load_with_act(backend, residency, tokenizer, None).await
    }

    /// Diagnostic variant of [`load`] that forces the block activation dtype
    /// instead of probing device f16 support. Lets the e2e run an fp32-acts
    /// forward to separate amplified-bf16 rounding from algorithmic (dtype-
    /// independent) error. Prod callers use [`load`] (probe-driven).
    pub async fn load_with_act(
        backend: Arc<WgpuBackend>,
        residency: WeightResidency<S>,
        tokenizer: T,
        act_override: Option<ActDtype>,
    ) -> Result<Self, ModelLoadError> {
        let timing = tracing::enabled!(tracing::Level::INFO);
        let t0 = timing.then(trace::Instant::now);

        let cfg = WanDitConfig::fastwan_ti2v_5b();

        // --- handle registration (no upload) ---
        // umT5 GGUF ships matmuls quantized in-file; bf16/f32 safetensors stay
        // dense (no transcode for v1 -- keep the parity path bit-clean).
        let vae_cfg = WanVaeConfig::fastwan_ti2v_5b();
        let umt5_handles = register_umt5_handles(&residency, None)?;
        let dit_handles = register_wan_dit_handles(&residency, &cfg, None)?;
        // VAE decoder weights all fit resident; diff the registered footprint
        // across registration so the decode can reserve exactly it (not a budget
        // fraction) when sizing its non-evictable tile workspace.
        let before_vae_bytes = residency.total_registered_bytes();
        let vae_handles = register_decoder(&residency, &VaeDecoderWeights::new(&vae_cfg))?;
        let vae_weight_footprint = residency.total_registered_bytes() - before_vae_bytes;

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
        let dit_w = probe_weight(&residency, "blocks.0.attn1.to_q.weight");
        let umt5_w = probe_weight(&residency, "encoder.block.0.layer.0.SelfAttention.q.weight");
        let act = act_override.unwrap_or(if backend.supports_shader_f16() {
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

        let dit = WanDitPipelines::compile(&backend, &block_cfgs(dit_w, act)).await?;
        let umt5_pipelines =
            Umt5Pipelines::compile(&backend, &block_cfgs(umt5_w, umt5_act)).await?;
        let vae_pipelines = WanVaePipelines::compile(&backend).await?;

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
            cfg,
            vae: WanVaeDecoder {
                pipelines: vae_pipelines,
                handles: vae_handles,
                cfg: vae_cfg,
                weight_footprint: vae_weight_footprint,
            },
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

    /// Run the full pipeline and return decoded video frames.
    pub async fn generate(
        &self,
        params: &GenerationParams,
        progress: ProgressFn<'_>,
    ) -> Result<WanVideo, GenerateError<S::Error>> {
        let mut workspace = Workspace::new(
            Arc::clone(&self.backend),
            Arc::clone(self.residency.arbiter()),
        );
        let (latent, f_lat, h_lat, w_lat) = self
            .denoise_with(params, None, &mut workspace, None, progress)
            .await?;

        if let Some(p) = progress {
            p(ProgressEvent::VaeDecode);
        }
        let frames = {
            let _s = trace::scope!("vae_decode", f_lat = f_lat).entered();
            self.vae
                .decode(
                    &self.backend,
                    &self.residency,
                    &mut workspace,
                    &latent,
                    f_lat,
                    h_lat,
                    w_lat,
                )
                .await?
        };
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
            height: h_lat * VAE_SCALE,
            width: w_lat * VAE_SCALE,
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
        workspace: &mut Workspace<WgpuBackend>,
    ) -> Result<Vec<f32>, GenerateError<S::Error>> {
        Ok(self
            .vae
            .decode(
                &self.backend,
                &self.residency,
                workspace,
                latent,
                f_lat,
                h_lat,
                w_lat,
            )
            .await?)
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
    ) -> Result<(Vec<f32>, usize, usize, usize), GenerateError<S::Error>> {
        if let Some(sink) = step_diag.as_deref_mut() {
            sink.clear();
        }
        let div = (VAE_SCALE * dit_config::PATCH_H) as u32;
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
        let h_lat = (params.height as usize) / VAE_SCALE;
        let w_lat = (params.width as usize) / VAE_SCALE;
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

        // --- 4. DiT + DMD sampler (fixed few-step schedule, CFG-free) ---
        let shape = WanDitShape::new(self.cfg.in_channels, f_lat, h_lat, w_lat, TEXT_SEQ);
        let dit = WanDit::assemble(self.dit_handles.clone(), shape, self.cfg);
        let sampler = DmdSampler::new(&DmdConfig::fastwan_ti2v_5b());
        let n_steps = sampler.num_steps();

        // Cap the streamed DiT weight set below budget by the in-flight transient
        // envelope (overlapping prefetch stagings + the forward workspace), so the
        // VRAM true peak holds under the (hard) budget ceiling even at the thin
        // 2 GB default. Budget-independent; set once for the whole step loop.
        self.residency.set_transient_reserve(
            PREFETCH_STAGING_DEPTH * self.residency.vram_staging_reserve_bytes()
                + DIT_WORKSPACE_RESERVE,
        );

        // --- 5. step loop: one DiT forward per fixed timestep, renoise between ---
        for i in 0..n_steps {
            let _step = trace::scope!("step", i = i).entered();
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
        self.diag_step_at(params, initial_noise, 0, workspace).await
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
        workspace: &mut Workspace<WgpuBackend>,
    ) -> Result<WanStep0Diag, GenerateError<S::Error>> {
        let h_lat = (params.height as usize) / VAE_SCALE;
        let w_lat = (params.width as usize) / VAE_SCALE;
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
        };

        // Block-0 per-op sinks: the driver only fills these GPU buffers; the
        // caller allocates them (persistent) and reads them back after forward.
        // Sized rows*inner except ffn_gelu (rows*ffn_dim).
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
        let h_lat = (params.height as usize) / VAE_SCALE;
        let w_lat = (params.width as usize) / VAE_SCALE;
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
fn block_cfgs(weight_dtype: WeightDtype, act: ActDtype) -> BlockWgslConfigs {
    let ops = WgslConfig {
        bf16_quant_writes: RECIPE.bf16_quant_writes,
        act_dtype: act,
        weight_dtype: WeightDtype::Bf16,
    };
    let mm = WgslConfig {
        weight_dtype,
        ..ops
    };
    BlockWgslConfigs {
        matmul_qkv: mm,
        matmul_proj: mm,
        matmul_ffn_up: mm,
        matmul_ffn_down: mm,
        matmul_adaln: ops,
        ops,
        i8_sdpa: false,
        dense_acts: DenseActSites::default(),
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
