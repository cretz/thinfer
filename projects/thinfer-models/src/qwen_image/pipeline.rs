//! Qwen-Image t2i pipeline: Qwen2.5-VL encode -> dual-stream DiT FlowMatchEuler
//! denoise (CFG-free) -> Wan-family KL VAE decode -> RGB. Mirrors the ideogram4
//! pipeline's phase structure (`evict_all_and_free` between encode / denoise /
//! decode so peak VRAM holds the budget). Weights come from one residency over a
//! union of the DiT GGUF (1:1 keys) + the renamed encoder + renamed VAE sources.
//!
//! Edit conditioning (vision tower + VAE-latent concat) is a later step; this is
//! the text-only path. `generate` takes `token_ids` (the CLI supplies the
//! Qwen2.5-VL chat-template tokenizer).

use crate::common::{Generated, StepControl, StepState};
use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::residency::WeightResidency;
use thinfer_core::trace;
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::Workspace;

use crate::common::block::BlockPipelines;
use crate::qwen_image::config;
use crate::qwen_image::dit::{QwenImageDit, QwenImageDitPipelines};
use crate::qwen_image::loader::{DitHandles, register_handles as register_dit_handles};
use crate::qwen_image::packing::{pack_latents, unpack_latents};
use crate::qwen_image::text_encoder::{
    EditEncoderPipelines, EncoderHandles, TextEncoder, config as enc_config,
    register_handles as register_encoder_handles,
};
use crate::qwen_image::vae::{normalize_ref_latent, qwen_image_vae};
use crate::qwen_image::vision::{
    VisionHandles, VisionTower, register_handles as register_vision_handles,
};
use crate::wan::vae::{
    VaeDecoderWeights, VaeEncoderWeights, WanVaeDecoder, WanVaeEncoder, WanVaePipelines,
    register_decoder, register_encoder,
};
use crate::z_image::pipeline::encode_png;

/// Tokens dropped from the front of the encoder output: the t2i prompt template's
/// system preamble (`prompt_template_encode_start_idx`).
pub const DROP_IDX: usize = 34;

/// Progress notifications, mirroring `z_image`/`ideogram4::pipeline`.
#[derive(Clone, Copy, Debug)]
pub enum ProgressEvent {
    TextEncode,
    Step { i: u32, n: u32 },
    VaeDecode,
}

pub type ProgressFn<'a> = Option<&'a dyn Fn(ProgressEvent)>;

/// One FlowMatchEuler step: `t` is the DiT timestep (sigma in `[0,1]`), `delta =
/// sigma_next - sigma` (the Euler coefficient for `z += delta * velocity`).
#[derive(Clone, Copy, Debug)]
struct Step {
    t: f32,
    delta: f32,
}

/// FlowMatchEuler schedule with dynamic mu shift (`calculate_shift`). `sigmas =
/// linspace(1, 1/N, N) ++ 0`; `shifted = e^mu / (e^mu + (1/sigma - 1))`;
/// `t = shifted`, `delta = shifted_next - shifted`.
fn build_steps(num_steps: usize, img_seq: usize) -> Vec<Step> {
    // calculate_shift(img_seq, base_seq=256, max_seq=4096, base_shift=0.5, max_shift=1.15)
    let m = (1.15 - 0.5) / (4096.0 - 256.0);
    let b = 0.5 - m * 256.0;
    let mu: f64 = img_seq as f64 * m + b;
    let emu = mu.exp();
    let shift = |sigma: f64| -> f64 {
        if sigma <= 0.0 {
            0.0
        } else {
            emu / (emu + (1.0 / sigma - 1.0))
        }
    };
    let n = num_steps.max(1);
    // linspace(1.0, 1/n, n) then append 0.0.
    let mut shifted: Vec<f64> = (0..n)
        .map(|i| {
            let sigma = 1.0 + (1.0 / n as f64 - 1.0) * (i as f64 / (n - 1).max(1) as f64);
            shift(sigma)
        })
        .collect();
    shifted.push(0.0);
    (0..n)
        .map(|i| Step {
            t: shifted[i] as f32,
            delta: (shifted[i + 1] - shifted[i]) as f32,
        })
        .collect()
}

/// Deterministic standard-normal noise (SplitMix64 -> Box-Muller). Mirrors the
/// ideogram4 helper.
fn gaussian_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next = || -> f64 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z = z ^ (z >> 31);
        // 53-bit mantissa -> (0,1)
        ((z >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    };
    let mut out = vec![0.0_f32; n];
    let mut i = 0;
    while i < n {
        let u1 = next().max(1e-12);
        let u2 = next();
        let r = (-2.0 * u1.ln()).sqrt();
        let a = std::f64::consts::TAU * u2;
        out[i] = (r * a.cos()) as f32;
        if i + 1 < n {
            out[i + 1] = (r * a.sin()) as f32;
        }
        i += 2;
    }
    out
}

#[derive(Debug)]
pub enum LoadError {
    Encoder(crate::z_image::text_encoder::LoadError),
    Vae(crate::common::loader::LoadError),
    Wgpu(thinfer_core::backend::WgpuError),
}

impl From<crate::z_image::text_encoder::LoadError> for LoadError {
    fn from(e: crate::z_image::text_encoder::LoadError) -> Self {
        Self::Encoder(e)
    }
}
impl From<crate::common::loader::LoadError> for LoadError {
    fn from(e: crate::common::loader::LoadError) -> Self {
        Self::Vae(e)
    }
}
impl From<thinfer_core::backend::WgpuError> for LoadError {
    fn from(e: thinfer_core::backend::WgpuError) -> Self {
        Self::Wgpu(e)
    }
}

#[derive(Debug)]
pub enum GenerateError<SE: core::fmt::Debug> {
    EmptyPrompt,
    InvalidDims {
        height: u32,
        width: u32,
    },
    Encode(crate::qwen_image::text_encoder::ForwardError<SE>),
    Vision(crate::qwen_image::vision::VisionError<SE>),
    VaeEncode(crate::wan::vae::WanVaeEncodeError<SE>),
    Dit(crate::qwen_image::dit::DitError<SE>),
    Vae(crate::wan::vae::WanVaeDecodeError<SE>),
    Png(String),
    /// A resume state that does not fit this request.
    ResumeMismatch,
}

/// One input image of an edit, preprocessed (see `thinfer_app::preprocess`).
pub struct EditImage<'a> {
    /// Its `<|image_pad|>` run start in the edit tokens.
    pub pad_start: usize,
    /// ViT patches `[gh * gw, 1176]`.
    pub vit_pixels: &'a [f32],
    /// ViT grid `(gh, gw)`.
    pub vit_grid: (usize, usize),
    /// VAE image CTHW `[3, 1, Hv, Wv]` in `[-1, 1]`.
    pub vae_image: &'a [f32],
    /// `(Hv, Wv)`, multiples of 32.
    pub vae_dims: (usize, usize),
}

pub struct QwenImagePipeline<S: WeightSource> {
    backend: Arc<WgpuBackend>,
    residency: WeightResidency<S>,
    encoder: TextEncoder,
    encoder_handles: EncoderHandles,
    encoder_pipelines: BlockPipelines,
    dit: QwenImageDit,
    dit_handles: DitHandles,
    dit_pipelines: QwenImageDitPipelines,
    vae: WanVaeDecoder,
    // --- edit (image->image) path; `None` on a text-to-image load ---
    edit: Option<EditPath>,
}

/// The extra weights/pipelines only the image-edit path needs: vision tower,
/// the 3-axis-MRoPE edit encoder, and the VAE encoder. Skipped on a t2i load so
/// it neither downloads the mmproj nor registers those weights.
struct EditPath {
    vision: VisionTower,
    vision_handles: VisionHandles,
    vision_pipelines: QwenImageDitPipelines,
    edit_encoder_pipelines: EditEncoderPipelines,
    vae_encoder: WanVaeEncoder,
}

impl<S: WeightSource> QwenImagePipeline<S> {
    /// Build the pipeline over a residency that already unions the DiT GGUF +
    /// renamed encoder + renamed VAE sources. `max_seq` sizes the encoder rope.
    /// `edit` selects the image-edit path: when true, also load the vision
    /// tower, edit encoder, and VAE encoder (their weights must be in the
    /// residency). A text-to-image load passes `false` and omits them, so the
    /// residency need only union DiT + encoder + VAE (no mmproj).
    pub async fn load(
        backend: Arc<WgpuBackend>,
        residency: WeightResidency<S>,
        max_seq: usize,
        i8_matmul: bool,
        edit: bool,
    ) -> Result<Self, LoadError> {
        // encoder: bf16 acts, all matmul sites Q8_0 (shared config, see
        // `text_encoder::encoder_block_cfgs`).
        let encoder_handles = register_encoder_handles(&residency, None)?;
        let enc_cfgs = crate::qwen_image::text_encoder::encoder_block_cfgs();
        let encoder_pipelines = BlockPipelines::compile(&backend, &enc_cfgs).await?;

        // DiT: bf16 acts (the residual stream exceeds f16 range); Q8_0 block
        // matmuls dequant-once, bf16 (adaln) embedders. See `dit::block_cfgs`.
        // `i8_matmul` gates the Q/K/V-only f16 subgroup SDPA (fast-attention
        // path); the residual + all matmuls stay bf16/Q8_0 either way.
        let dit_handles = register_dit_handles(&residency, config::N_LAYERS)?;
        // Edit-2511 family: the checkpoint's marker tensor selects
        // reference-at-timestep-0 modulation (as ComfyUI detects it).
        let zero_cond_t = residency
            .source()
            .catalog()
            .get(&thinfer_core::weight::WeightId(
                crate::qwen_image::dit::ZERO_COND_T_MARKER.into(),
            ))
            .is_some();
        let dit_cfgs = crate::qwen_image::dit::block_cfgs();
        let dit_pipelines = QwenImageDitPipelines::compile(&backend, &dit_cfgs, i8_matmul).await?;

        // VAE (Wan-family decoder + encoder; one config, both directions).
        let vae_cfg = qwen_image_vae();
        let before = residency.total_registered_bytes();
        let vae_handles = register_decoder(&residency, &VaeDecoderWeights::new(&vae_cfg))?;
        let vae_footprint = residency.total_registered_bytes() - before;
        let vae_pipelines = WanVaePipelines::compile(&backend).await?;

        // --- edit path: vision tower + edit-encoder + VAE encoder ---
        // Only loaded for the image-edit path; a t2i load skips it entirely (no
        // mmproj download, no extra registered weights).
        let edit = if edit {
            // Vision shares the DiT block driver but with bf16 (not Q8) block
            // matmuls, so it needs a SEPARATE pipeline instance.
            let vision_handles = register_vision_handles(&residency)?;
            let vision_cfgs = VisionTower::wgsl_configs();
            // Vision windowed attention rides the common-block `op_sdpa` with a
            // real block-diagonal mask, so it stays off the f16 fast-attention
            // path.
            let vision_pipelines =
                QwenImageDitPipelines::compile(&backend, &vision_cfgs, false).await?;
            // Edit encoder reuses the same bf16-act / Q8-matmul block config as
            // t2i, plus the 3-axis MRoPE kernel.
            let edit_encoder_pipelines = EditEncoderPipelines::compile(&backend, &enc_cfgs).await?;
            // VAE encoder (Wan-family); same pipeline family covers encode +
            // decode.
            let before_enc = residency.total_registered_bytes();
            let vae_enc_handles = register_encoder(&residency, &VaeEncoderWeights::new(&vae_cfg))?;
            let vae_enc_footprint = residency.total_registered_bytes() - before_enc;
            let vae_encoder = WanVaeEncoder {
                pipelines: WanVaePipelines::compile(&backend).await?,
                handles: vae_enc_handles,
                cfg: vae_cfg.clone(),
                weight_footprint: vae_enc_footprint,
            };
            Some(EditPath {
                // `max_grid` sizes the 2D rope table; ViT smart_resize keeps
                // grids small (256 raw patches/side is far beyond any
                // test/runtime image).
                vision: VisionTower::new(256),
                vision_handles,
                vision_pipelines,
                edit_encoder_pipelines,
                vae_encoder,
            })
        } else {
            None
        };

        Ok(Self {
            backend,
            residency,
            encoder: TextEncoder::new(max_seq),
            encoder_handles,
            encoder_pipelines,
            dit: QwenImageDit::new(zero_cond_t),
            dit_handles,
            dit_pipelines,
            vae: WanVaeDecoder {
                pipelines: vae_pipelines,
                handles: vae_handles,
                cfg: vae_cfg,
                weight_footprint: vae_footprint,
            },
            edit,
        })
    }

    /// Encode -> denoise -> VAE decode. Returns RGB CHW f32 in `[-1, 1]`,
    /// `[3, height, width]`.
    #[allow(clippy::too_many_arguments)]
    pub async fn generate_rgb(
        &self,
        token_ids: &[u32],
        height: u32,
        width: u32,
        steps: u32,
        seed: u64,
        progress: ProgressFn<'_>,
    ) -> Result<Vec<f32>, GenerateError<S::Error>> {
        match self
            .generate_rgb_ctl(
                token_ids,
                height,
                width,
                steps,
                seed,
                progress,
                StepControl::default(),
            )
            .await?
        {
            Generated::Done(rgb) => Ok(rgb),
            Generated::Stopped(_) => unreachable!("no stop predicate was given"),
        }
    }

    /// [`Self::generate_rgb`] that can stop at a step boundary (`ctl`) and
    /// resume there; the encode is recomputed on resume (deterministic).
    #[allow(clippy::too_many_arguments)]
    pub async fn generate_rgb_ctl(
        &self,
        token_ids: &[u32],
        height: u32,
        width: u32,
        steps: u32,
        seed: u64,
        progress: ProgressFn<'_>,
        ctl: StepControl<'_>,
    ) -> Result<Generated<Vec<f32>>, GenerateError<S::Error>> {
        if token_ids.is_empty() {
            return Err(GenerateError::EmptyPrompt);
        }
        if !height.is_multiple_of(config::PIXELS_PER_TOKEN as u32)
            || !width.is_multiple_of(config::PIXELS_PER_TOKEN as u32)
            || height == 0
            || width == 0
        {
            return Err(GenerateError::InvalidDims { height, width });
        }
        let gh = height as usize / config::PIXELS_PER_TOKEN;
        let gw = width as usize / config::PIXELS_PER_TOKEN;
        let lat_h = gh * 2;
        let lat_w = gw * 2;
        let img_seq = gh * gw;

        let mut workspace = Workspace::new(
            Arc::clone(&self.backend),
            Arc::clone(self.residency.arbiter()),
        );

        // --- 1. encode (bf16) -> drop the template preamble ---
        let steps_plan = build_steps(steps as usize, img_seq);

        // A resumed run starts at its step, from its latent.
        let (first, mut tokens) = match ctl.resume {
            Some(st) => {
                if st.step > steps_plan.len()
                    || st.latent.len() != config::OUT_CHANNELS * lat_h * lat_w
                {
                    return Err(GenerateError::ResumeMismatch);
                }
                (st.step, st.latent)
            }
            None => (
                0,
                pack_latents(
                    &gaussian_noise(config::OUT_CHANNELS * lat_h * lat_w, seed),
                    lat_h,
                    lat_w,
                ),
            ),
        };
        // Denoised already: straight to the decode.
        if first < steps_plan.len() {
            if let Some(p) = progress {
                p(ProgressEvent::TextEncode);
            }
            let enc = {
                let _s = trace::scope!("qwen_image.encode", tokens = token_ids.len()).entered();
                self.encoder
                    .forward(
                        &self.backend,
                        &self.encoder_pipelines,
                        &self.residency,
                        &workspace,
                        &self.encoder_handles,
                        self.residency.source(),
                        token_ids,
                        false,
                    )
                    .await
                    .map_err(GenerateError::Encode)?
            };
            let hd = enc_config::HIDDEN;
            let drop = DROP_IDX.min(enc.seq.saturating_sub(1));
            let txt_embeds: Vec<f32> = enc.hidden[drop * hd..].to_vec();
            debug_assert_eq!(txt_embeds.len() % config::JOINT_ATTENTION_DIM, 0);
            self.residency.evict_all_and_free(&*self.backend);
            workspace.drain_pool();

            // --- 2. FlowMatchEuler denoise (CFG-free) ---
            {
                let _s = trace::scope!("qwen_image.denoise", steps = steps).entered();
                for (i, step) in steps_plan.iter().enumerate().skip(first) {
                    // Between steps: a step boundary.
                    if i > first && ctl.stop_here.is_some_and(|c| c()) {
                        self.residency.evict_all_and_free(&*self.backend);
                        return Ok(Generated::Stopped(StepState::at(i, tokens)));
                    }
                    if let Some(p) = progress {
                        p(ProgressEvent::Step {
                            i: i as u32 + 1,
                            n: steps,
                        });
                    }
                    let out = self
                        .dit
                        .forward(
                            &self.backend,
                            &self.dit_pipelines,
                            &self.residency,
                            &workspace,
                            &self.dit_handles,
                            &tokens,
                            &txt_embeds,
                            step.t,
                            1,
                            gh,
                            gw,
                            None,
                        )
                        .await
                        .map_err(GenerateError::Dit)?;
                    debug_assert_eq!(out.velocity.len(), tokens.len());
                    for (zi, &v) in tokens.iter_mut().zip(out.velocity.iter()) {
                        *zi += v * step.delta;
                    }
                }
            }
            self.residency.evict_all_and_free(&*self.backend);
            workspace.drain_pool();

            // --- 3. unpack -> VAE decode ---
            // Before the decode: a step boundary.
            if ctl.stop_here.is_some_and(|c| c()) {
                return Ok(Generated::Stopped(StepState::at(steps_plan.len(), tokens)));
            }
        }

        if let Some(p) = progress {
            p(ProgressEvent::VaeDecode);
        }
        let z = unpack_latents(&tokens, lat_h, lat_w);
        let rgb = self
            .vae
            .decode(
                &self.backend,
                &self.residency,
                &mut workspace,
                &z,
                1,
                lat_h,
                lat_w,
            )
            .await
            .map_err(GenerateError::Vae)?;
        self.residency.evict_all_and_free(&*self.backend);
        // decode returns CTHW [3, 1, lat_h*8, lat_w*8] = [3, height, width].
        Ok(Generated::Done(rgb))
    }

    /// Image-to-image edit: vision-tower + edit-encoder text conditioning, plus a
    /// VAE-encoded reference latent concatenated onto the DiT image stream.
    /// CFG-free FlowMatchEuler denoise -> VAE decode. Returns RGB CHW f32 in
    /// `[-1, 1]`, `[3, height, width]`.
    ///
    /// `token_ids` carry the edit chat template with `<|image_pad|>` already
    /// expanded to `n_img = (vit_gh/2)*(vit_gw/2)` slots; `image_pad_start` is the
    /// index of the first image-pad token; the encoder output drops the first
    /// `drop` rows (the template preamble; the image-pad hiddens survive it).
    /// `vit_pixels` is the patchified ViT
    /// input `[N, 1176]` (`N = vit_gh*vit_gw`). `vae_image` is the SAME source
    /// image preprocessed for the VAE channel: CTHW `[3, 1, Hv, Wv]` in `[-1, 1]`.
    #[allow(clippy::too_many_arguments)]
    pub async fn generate_edit_rgb(
        &self,
        token_ids: &[u32],
        image_pad_start: usize,
        drop: usize,
        vit_pixels: &[f32],
        vit_grid: (usize, usize),
        vae_image: &[f32],
        vae_dims: (usize, usize),
        height: u32,
        width: u32,
        steps: u32,
        seed: u64,
        progress: ProgressFn<'_>,
    ) -> Result<Vec<f32>, GenerateError<S::Error>> {
        let image = EditImage {
            pad_start: image_pad_start,
            vit_pixels,
            vit_grid,
            vae_image,
            vae_dims,
        };
        match self
            .generate_edit_rgb_ctl(
                token_ids,
                drop,
                std::slice::from_ref(&image),
                height,
                width,
                steps,
                seed,
                progress,
                StepControl::default(),
            )
            .await?
        {
            Generated::Done(rgb) => Ok(rgb),
            Generated::Stopped(_) => unreachable!("no stop predicate was given"),
        }
    }

    /// [`Self::generate_edit_rgb`] over one or more input images (in prompt
    /// order: each its own vision block and reference latent), that can stop
    /// at a step boundary (`ctl`) and resume there; the encodes are
    /// recomputed on resume (deterministic).
    #[allow(clippy::too_many_arguments)]
    pub async fn generate_edit_rgb_ctl(
        &self,
        token_ids: &[u32],
        drop: usize,
        images: &[EditImage<'_>],
        height: u32,
        width: u32,
        steps: u32,
        seed: u64,
        progress: ProgressFn<'_>,
        ctl: StepControl<'_>,
    ) -> Result<Generated<Vec<f32>>, GenerateError<S::Error>> {
        if token_ids.is_empty() {
            return Err(GenerateError::EmptyPrompt);
        }
        if !height.is_multiple_of(config::PIXELS_PER_TOKEN as u32)
            || !width.is_multiple_of(config::PIXELS_PER_TOKEN as u32)
            || height == 0
            || width == 0
        {
            return Err(GenerateError::InvalidDims { height, width });
        }
        let edit = self.edit.as_ref().expect("generate_edit_rgb on a t2i load");
        if images.is_empty() {
            return Err(GenerateError::EmptyPrompt);
        }
        // Noise (target) geometry: DiT patch grid = pixels / (8 * patch 2) = /16.
        let gh_noise = height as usize / config::PIXELS_PER_TOKEN;
        let gw_noise = width as usize / config::PIXELS_PER_TOKEN;
        let lat_h = gh_noise * 2;
        let lat_w = gw_noise * 2;
        let noise_seq = gh_noise * gw_noise;

        let mut workspace = Workspace::new(
            Arc::clone(&self.backend),
            Arc::clone(self.residency.arbiter()),
        );

        // --- 1. vision tower: patches -> LM embeds (bf16 block matmuls) ---
        let steps_plan = build_steps(steps as usize, noise_seq);

        // A resumed run starts at its step, from its latent.
        let (first, mut noise_tokens) = match ctl.resume {
            Some(st) => {
                if st.step > steps_plan.len()
                    || st.latent.len() != config::OUT_CHANNELS * lat_h * lat_w
                {
                    return Err(GenerateError::ResumeMismatch);
                }
                (st.step, st.latent)
            }
            None => (
                0,
                pack_latents(
                    &gaussian_noise(config::OUT_CHANNELS * lat_h * lat_w, seed),
                    lat_h,
                    lat_w,
                ),
            ),
        };
        // Denoised already: straight to the decode.
        if first < steps_plan.len() {
            if let Some(p) = progress {
                p(ProgressEvent::TextEncode);
            }
            // One vision-tower pass per image (its attention never crosses
            // images, so this is the batched pass).
            let mut visions = Vec::with_capacity(images.len());
            for img in images {
                let (gh_vit, gw_vit) = img.vit_grid;
                let _s = trace::scope!("qwen_image.vision", grid = gh_vit * gw_vit).entered();
                let v = edit
                    .vision
                    .forward(
                        &self.backend,
                        &edit.vision_pipelines,
                        &self.residency,
                        &workspace,
                        &edit.vision_handles,
                        img.vit_pixels,
                        gh_vit,
                        gw_vit,
                    )
                    .await
                    .map_err(GenerateError::Vision)?;
                visions.push((v, (gh_vit / 2, gw_vit / 2)));
            }
            self.residency.evict_all_and_free(&*self.backend);
            workspace.drain_pool();

            // --- 2. edit encode (LM + MRoPE + vision scatter) -> drop preamble ---
            let enc = {
                let _s =
                    trace::scope!("qwen_image.encode_edit", tokens = token_ids.len()).entered();
                self.encoder
                    .forward_edit_multi(
                        &self.backend,
                        &edit.edit_encoder_pipelines,
                        &self.residency,
                        &workspace,
                        &self.encoder_handles,
                        self.residency.source(),
                        token_ids,
                        &images
                            .iter()
                            .zip(&visions)
                            .map(
                                |(img, (v, grid))| crate::qwen_image::text_encoder::EditVision {
                                    pad_start: img.pad_start,
                                    embeds: &v.embeds,
                                    merged_grid: *grid,
                                },
                            )
                            .collect::<Vec<_>>(),
                        false,
                    )
                    .await
                    .map_err(GenerateError::Encode)?
            };
            let hd = enc_config::HIDDEN;
            let drop = drop.min(enc.seq.saturating_sub(1));
            let txt_embeds: Vec<f32> = enc.hidden[drop * hd..].to_vec();
            debug_assert_eq!(txt_embeds.len() % config::JOINT_ATTENTION_DIM, 0);
            self.residency.evict_all_and_free(&*self.backend);
            workspace.drain_pool();

            // --- 3. reference VAE latent: encode -> mode -> normalize -> pack ---
            // The image stream's grids: the target's, then each reference's
            // (latent /8, DiT grid /16), packed in that order.
            let mut grids = vec![(1, gh_noise, gw_noise)];
            let mut ref_tokens = Vec::new();
            for img in images {
                let (hv, wv) = img.vae_dims;
                let (lat_hv, lat_wv) = (hv / 8, wv / 8);
                let _s = trace::scope!("qwen_image.ref_latent", h = hv, w = wv).entered();
                let moments = edit
                    .vae_encoder
                    .encode(
                        &self.backend,
                        &self.residency,
                        &mut workspace,
                        img.vae_image,
                        1,
                        hv,
                        wv,
                    )
                    .await
                    .map_err(GenerateError::VaeEncode)?;
                let ref_latent =
                    normalize_ref_latent(&moments, &edit.vae_encoder.cfg, lat_hv, lat_wv);
                ref_tokens.extend(pack_latents(&ref_latent, lat_hv, lat_wv));
                grids.push((
                    1,
                    hv / config::PIXELS_PER_TOKEN,
                    wv / config::PIXELS_PER_TOKEN,
                ));
            }
            let ref_seq: usize = grids[1..].iter().map(|&(f, h, w)| f * h * w).sum();
            debug_assert_eq!(ref_tokens.len(), ref_seq * config::IN_CHANNELS);
            self.residency.evict_all_and_free(&*self.backend);
            workspace.drain_pool();

            // --- 4. FlowMatchEuler denoise (CFG-free), [noise ++ ref] image stream ---
            debug_assert_eq!(noise_tokens.len(), noise_seq * config::IN_CHANNELS);
            if std::env::var_os("THINFER_VAE_MEM").is_some() {
                use thinfer_core::backend::Backend;
                let mem = self.backend.mem_account();
                eprintln!(
                    "[vae_mem] DENOISE_EDIT noise_seq={} ref_seq={} total_img={} vram_total={}MiB",
                    noise_seq,
                    ref_seq,
                    noise_seq + ref_seq,
                    mem.vram_total_current() / (1024 * 1024),
                );
            }
            {
                let _s = trace::scope!("qwen_image.denoise_edit", steps = steps).entered();
                for (i, step) in steps_plan.iter().enumerate().skip(first) {
                    // Between steps: a step boundary.
                    if i > first && ctl.stop_here.is_some_and(|c| c()) {
                        self.residency.evict_all_and_free(&*self.backend);
                        return Ok(Generated::Stopped(StepState::at(i, noise_tokens)));
                    }
                    if let Some(p) = progress {
                        p(ProgressEvent::Step {
                            i: i as u32 + 1,
                            n: steps,
                        });
                    }
                    let mut img_tokens = Vec::with_capacity(noise_tokens.len() + ref_tokens.len());
                    img_tokens.extend_from_slice(&noise_tokens);
                    img_tokens.extend_from_slice(&ref_tokens);
                    let out = self
                        .dit
                        .forward_multi(
                            &self.backend,
                            &self.dit_pipelines,
                            &self.residency,
                            &workspace,
                            &self.dit_handles,
                            &img_tokens,
                            &txt_embeds,
                            step.t,
                            &grids,
                            noise_seq,
                            None,
                        )
                        .await
                        .map_err(GenerateError::Dit)?;
                    debug_assert_eq!(out.velocity.len(), noise_tokens.len());
                    for (zi, &v) in noise_tokens.iter_mut().zip(out.velocity.iter()) {
                        *zi += v * step.delta;
                    }
                }
            }
            self.residency.evict_all_and_free(&*self.backend);
            workspace.drain_pool();

            // --- 5. unpack -> VAE decode ---
            // Before the decode: a step boundary.
            if ctl.stop_here.is_some_and(|c| c()) {
                return Ok(Generated::Stopped(StepState::at(
                    steps_plan.len(),
                    noise_tokens,
                )));
            }
        }

        if let Some(p) = progress {
            p(ProgressEvent::VaeDecode);
        }
        let z = unpack_latents(&noise_tokens, lat_h, lat_w);
        let rgb = self
            .vae
            .decode(
                &self.backend,
                &self.residency,
                &mut workspace,
                &z,
                1,
                lat_h,
                lat_w,
            )
            .await
            .map_err(GenerateError::Vae)?;
        self.residency.evict_all_and_free(&*self.backend);
        Ok(Generated::Done(rgb))
    }

    /// `generate_rgb` + PNG encode.
    pub async fn generate(
        &self,
        token_ids: &[u32],
        height: u32,
        width: u32,
        steps: u32,
        seed: u64,
        progress: ProgressFn<'_>,
    ) -> Result<Vec<u8>, GenerateError<S::Error>> {
        let rgb = self
            .generate_rgb(token_ids, height, width, steps, seed, progress)
            .await?;
        encode_png(&rgb, width, height).map_err(GenerateError::Png)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_start_at_one_and_end_near_zero() {
        let s = build_steps(4, 16);
        assert_eq!(s.len(), 4);
        // first sigma is 1.0 (pure noise), shifted(1.0)=1.0.
        assert!((s[0].t - 1.0).abs() < 1e-5, "t0={}", s[0].t);
        // cumulative deltas drive sigma from 1 -> 0.
        let final_sigma = s[0].t + s.iter().map(|x| x.delta).sum::<f32>();
        assert!(final_sigma.abs() < 1e-5, "final sigma {final_sigma}");
        // monotonic descending t.
        for w in s.windows(2) {
            assert!(w[1].t < w[0].t);
        }
    }
}
