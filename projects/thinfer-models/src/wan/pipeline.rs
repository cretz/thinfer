//! SkyReels-V2-DF-1.3B (Wan) pipeline orchestrator. Single entry point for CLI,
//! web, and the e2e: `WanModel::load(...)` builds the bundle once, `generate`
//! runs the whole stack (tokenize -> umT5 encode -> synchronous Diffusion-
//! Forcing denoise loop with the Wan DiT -> 3D causal VAE decode -> video
//! frames). Mirrors `z_image::pipeline::ZImageModel`.
//!
//! Owns the compiled `Umt5Pipelines` + `WanDitPipelines`, the residency-backed
//! handle bundles, the `WanVaeDecoder`, the residency, backend, and tokenizer.
//! No model internals leak past `generate`'s `WanVideo` return.
//!
//! Synchronous Diffusion Forcing only (the parity mode): every latent frame
//! shares `timesteps[i]` at step `i`, so the loop is a standard flow-match
//! denoise over the whole `[16, f_lat, h, w]` latent with per-frame timesteps
//! broadcast equal. Async/causal staggering + overlap_history stitching are
//! deferred to long-video (see `wan-plan.md`).

use std::sync::Arc;

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
use crate::wan::dit_block::{WanDitBlockTaps, WanDitPipelines, config as dit_config};
use crate::wan::loader::register_wan_dit_handles;
use crate::wan::manifest::RECIPE;
use crate::wan::scheduler::{SchedulerStepDiag, UniPCScheduler};
use crate::wan::umt5::{
    Umt5BlockOpsHost, Umt5Encoder, Umt5ForwardError, Umt5Handles, Umt5Pipelines, Umt5Taps,
    register_umt5_handles,
};
use crate::wan::vae::{
    VaeDecoderWeights, WanVaeDecodeError, WanVaeDecoder, WanVaePipelines, register_decoder,
};

/// umT5 rope-free context cap. The Wan DiT cross-attends to a fixed
/// `max_sequence_length`; SkyReels-V2 ships 512.
const TEXT_SEQ: usize = dit_config::TEXT_SEQ;
const MAX_PROMPT_TOKENS: usize = TEXT_SEQ;
const Z_DIM: usize = dit_config::IN_CHANNELS;
const VAE_SCALE: usize = 8;
const TEMPORAL_SCALE: usize = 4;

/// Inputs to one `generate` call.
pub struct GenerationParams {
    pub prompt: String,
    /// Classifier-free-guidance negative prompt. Encoded as a second umT5 pass
    /// and cross-attended by the unconditional DiT forward. Ignored when
    /// `guidance_scale <= 1.0` (no CFG). Upstream diffusers default is `""`.
    pub negative_prompt: String,
    /// CFG scale. `<= 1.0` disables CFG (one DiT forward per step, the bit-clean
    /// parity path). SkyReels-V2-DF is NOT guidance-distilled: the diffusers
    /// default is 6.0 (T2V) / 5.0 (I2V); CFG combines per step as
    /// `uncond + guidance_scale * (cond - uncond)`.
    pub guidance_scale: f32,
    /// Frame height in pixels. Divisible by `VAE_SCALE * PATCH_H` (16).
    pub height: u32,
    /// Frame width in pixels. Divisible by `VAE_SCALE * PATCH_W` (16).
    pub width: u32,
    /// Output frame count. Must be `4 * k + 1` (the causal-VAE temporal grid).
    pub num_frames: u32,
    /// Inference steps.
    pub steps: u32,
    /// Deterministic seed for the initial latent noise.
    pub seed: u64,
    /// Diffusion-Forcing `fps` bucket (`inject_sample_info`).
    pub fps: usize,
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
    /// The scheduler timestep fed to the DiT this step.
    pub timestep: f32,
    /// Raw DiT output (flow velocity) handed to the scheduler == the exact
    /// tensor pyref dumps as `py_dit_out_step{i}`.
    pub velocity: Vec<f32>,
    /// Latent after `scheduler.step` (the old per-step dump; == `py_step{i}_post`).
    pub post: Vec<f32>,
    /// Residual stream after each DiT block (`len == num_layers`). Localizes a
    /// velocity divergence to a block (vs `py_block{b}_out_step{i}`).
    pub per_block: Vec<Vec<f32>>,
    pub temb: Vec<f32>,
    pub timestep_proj: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub proj_out: Vec<f32>,
    /// Scheduler internals for this step (sigma/order/m_conv/corrected).
    pub sched: SchedulerStepDiag,
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
        let timing = tracing::enabled!(tracing::Level::INFO);
        let t0 = timing.then(trace::Instant::now);

        // --- handle registration (no upload) ---
        // umT5 GGUF ships matmuls quantized in-file; bf16/f32 safetensors stay
        // dense (no transcode for v1 -- keep the parity path bit-clean).
        let umt5_handles = register_umt5_handles(&residency, None)?;
        let dit_handles = register_wan_dit_handles(&residency, None)?;
        let vae_handles = register_decoder(&residency, &VaeDecoderWeights::new())?;

        // --- dtype selection ---
        // Probe a representative matmul tensor per submodel: `Quant(k)` when the
        // GGUF surfaced it, `Bf16`/`F32` -> Bf16 dense. The Wan DiT uses ONE
        // pipeline set for every matmul site (patch, condition embedder, all 30
        // blocks, proj_out), so the dtype must be uniform; the ComfyUI GGUF
        // quantizes those big linears uniformly (norms/biases stay F32).
        let dit_w = probe_weight(&residency, "blocks.0.attn1.to_q.weight");
        let umt5_w = probe_weight(&residency, "encoder.block.0.layer.0.SelfAttention.q.weight");
        let act = if backend.supports_shader_f16() {
            ActDtype::F16
        } else {
            ActDtype::F32
        };
        tracing::info!(?dit_w, ?umt5_w, ?act, "Wan dtype selection");

        let dit = WanDitPipelines::compile(&backend, &block_cfgs(dit_w, act)).await?;
        let umt5_pipelines = Umt5Pipelines::compile(&backend, &block_cfgs(umt5_w, act)).await?;
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
            vae: WanVaeDecoder {
                pipelines: vae_pipelines,
                handles: vae_handles,
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

    /// Tokenize -> umT5 encode -> synchronous DF denoise loop. Returns the final
    /// pre-VAE latent (CTHW f32, `16 * f_lat * h_lat * w_lat`) plus the latent
    /// dims. Caller owns `workspace` so the GPU pool survives the DiT->VAE seam.
    ///
    /// `initial_noise`: used verbatim as the starting latent when `Some` (e2e
    /// pinned-noise byte-load), else derived from `seed`. `step_diag`: when
    /// `Some`, a [`WanStepDiag`] per step is pushed (cleared on entry) with the
    /// velocity, post-step latent, per-block residual, and scheduler internals;
    /// the final entry's `post` equals the returned latent. `None` is the prod
    /// path (no GPU readbacks, plain `forward`/`step`).
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

        // CFG is on when the scale exceeds 1.0 (mirrors diffusers
        // `do_classifier_free_guidance = guidance_scale > 1.0`). When off, the
        // step loop runs exactly one DiT forward -- the bit-clean parity path
        // the e2e drives (it pins `guidance_scale = 1.0`).
        let do_cfg = params.guidance_scale > 1.0;

        let _denoise = trace::scope!("denoise").entered();

        // --- 1. tokenize (prompt + CFG negative prompt) ---
        let encode_ids = |prompt: &str| -> Result<Vec<u32>, GenerateError<S::Error>> {
            // umT5 needs the trailing `</s>` EOS the diffusers reference appends
            // (add_special_tokens=True). Omitting it shifts every token's
            // bidirectional attention and compounds across layers.
            let ids = self
                .tokenizer
                .encode(prompt, true)
                .map_err(GenerateError::Tokenizer)?;
            if ids.len() > MAX_PROMPT_TOKENS {
                return Err(GenerateError::PromptTooLong {
                    tokens: ids.len(),
                    max: MAX_PROMPT_TOKENS,
                });
            }
            Ok(ids)
        };
        let token_ids = {
            let _s = trace::scope!("tokenize").entered();
            encode_ids(&params.prompt)?
        };
        let neg_token_ids = if do_cfg {
            let _s = trace::scope!("tokenize_neg").entered();
            Some(encode_ids(&params.negative_prompt)?)
        } else {
            None
        };

        // --- 2. umT5 encode -> text states, padded to the DiT context ---
        // Both prompts are encoded while umT5 is resident (one phase), then the
        // weights are evicted; the padded host tensors live through the loop.
        if let Some(p) = progress {
            p(ProgressEvent::TextEncode);
        }
        // Shared reborrow so the encode closure captures a `Copy` `&Workspace`
        // (callable twice); the `&mut workspace` reverts after the encodes for
        // `drain_pool` below.
        let ws: &Workspace<WgpuBackend> = &*workspace;
        // `ids` is taken by value so the returned future owns it (a closure
        // can't tie a borrowed arg's lifetime into its async return type).
        let umt5_encode = |ids: Vec<u32>| async move {
            let qout = self
                .umt5
                .forward(
                    &self.backend,
                    &self.umt5_pipelines,
                    &self.residency,
                    ws,
                    &self.umt5_handles,
                    self.residency.source(),
                    &ids,
                )
                .await?;
            // Diffusers pads `prompt_embeds` to `max_sequence_length` (512) and
            // the DiT cross-attends over the full padded context (no mask). Pad
            // with zeros / truncate to TEXT_SEQ rows. The e2e must feed pyref the
            // same padded context. (Cross-attn masking is deferred -- verify at
            // e2e.)
            Ok::<Vec<f32>, GenerateError<S::Error>>(pad_text(
                &qout.hidden,
                qout.seq,
                dit_config::TEXT_DIM,
                TEXT_SEQ,
            ))
        };
        let text = umt5_encode(token_ids).await?;
        let neg_text = match neg_token_ids {
            Some(ids) => Some(umt5_encode(ids).await?),
            None => None,
        };

        // Phase boundary: umT5 weights are dead for the rest of the call.
        self.residency.evict_all_and_free(&*self.backend);
        workspace.drain_pool();

        // --- 3. initial noise [16, f_lat, h_lat, w_lat] ---
        let n_lat = Z_DIM * f_lat * h_lat * w_lat;
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

        // --- 4. DiT + scheduler ---
        let shape = WanDitShape::new(Z_DIM, f_lat, h_lat, w_lat, TEXT_SEQ);
        let dit = WanDit::assemble(self.dit_handles.clone(), shape);
        let mut scheduler = UniPCScheduler::new(params.steps as usize);
        let ppf = shape.grid.ppf;

        // --- 5. step loop (synchronous DF: every frame shares timesteps[i]) ---
        for i in 0..params.steps as usize {
            let _step = trace::scope!("step", i = i).entered();
            if let Some(p) = progress {
                p(ProgressEvent::Step {
                    i: i as u32 + 1,
                    n: params.steps,
                });
            }
            let t = scheduler.timesteps()[i];
            let timesteps = vec![t; ppf];
            let inputs = WanDitInputs {
                image: &sample,
                text: &text,
                timesteps: &timesteps,
                fps: params.fps,
            };
            // Diag path captures per-block + final-stage taps via forward_with_taps
            // and the scheduler internals; prod takes the plain forward/step.
            let (out, per_block, temb, timestep_proj, final_norm, proj_out) = if step_diag.is_some()
            {
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
                (out, per_block, temb, timestep_proj, final_norm, proj_out)
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
                    out,
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                )
            };
            // CFG: a second (unconditional) DiT forward over the negative prompt,
            // combined as `uncond + guidance_scale * (cond - uncond)` (diffusers
            // parity, line 909 of the DF pipeline). `neg_text` is `Some` exactly
            // when `do_cfg`, so when off the single cond forward is used verbatim
            // and the e2e (guidance_scale = 1.0) stays bit-identical.
            let velocity = match &neg_text {
                Some(neg) => {
                    let _cfg = trace::scope!("cfg_uncond").entered();
                    let neg_inputs = WanDitInputs {
                        image: &sample,
                        text: neg,
                        timesteps: &timesteps,
                        fps: params.fps,
                    };
                    let uncond = dit
                        .forward(
                            &self.backend,
                            &self.dit,
                            &self.residency,
                            &*workspace,
                            &neg_inputs,
                        )
                        .await?;
                    let gs = params.guidance_scale;
                    out.image
                        .iter()
                        .zip(uncond.image.iter())
                        .map(|(c, u)| u + gs * (c - u))
                        .collect::<Vec<f32>>()
                }
                None => out.image,
            };
            // Wan flow-prediction: the DiT predicts the flow velocity; the UniPC
            // scheduler converts to x0 and steps (no output negation, unlike
            // Z-Image's `-noise_pred`).
            let mut sched_diag = SchedulerStepDiag::default();
            sample = if step_diag.is_some() {
                scheduler.step_with_diag(&velocity, &sample, &mut sched_diag)
            } else {
                scheduler.step(&velocity, &sample)
            };
            if let Some(sink) = step_diag.as_deref_mut() {
                sink.push(WanStepDiag {
                    timestep: t,
                    velocity, // moved; the post-CFG velocity fed to the scheduler
                    post: sample.clone(),
                    per_block,
                    temb,
                    timestep_proj,
                    final_norm,
                    proj_out,
                    sched: sched_diag,
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

        let n_lat = Z_DIM * f_lat * h_lat * w_lat;
        assert_eq!(initial_noise.len(), n_lat, "diag_step0 noise len");
        let sample = initial_noise.to_vec();

        let shape = WanDitShape::new(Z_DIM, f_lat, h_lat, w_lat, TEXT_SEQ);
        let dit = WanDit::assemble(self.dit_handles.clone(), shape);
        let mut scheduler = UniPCScheduler::new(params.steps as usize);
        let ppf = shape.grid.ppf;
        let t = scheduler.timesteps()[0];
        let timesteps = vec![t; ppf];
        let inputs = WanDitInputs {
            image: &sample,
            text: &text,
            timesteps: &timesteps,
            fps: params.fps,
        };

        // Block-0 per-op sinks: the driver only fills these GPU buffers; the
        // caller allocates them (persistent) and reads them back after forward.
        // Sized rows*inner except ffn_gelu (rows*ffn_dim).
        let bp = &self.dit.block;
        let rows = shape.n_tok as u32;
        let inner = dit_config::INNER as u32;
        let ffn = dit_config::FFN_DIM as u32;
        let inner_buf = || workspace.alloc(bp.act_bytes(rows * inner));
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
        let stepped = scheduler.step(&out.image, &sample);

        // Read block-0 sinks back (in execution order) before draining the pool.
        let act = bp.act_dtype;
        let mut block0_stages: Vec<(String, Vec<f32>)> = Vec::new();
        for (name, buf, n) in [
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

/// Deterministic standard-normal samples via SplitMix64 -> Box-Muller (avoids a
/// `rand` dep). Same generator as `z_image::pipeline::gaussian_noise`.
fn gaussian_noise(n: usize, seed: u64) -> Vec<f32> {
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
