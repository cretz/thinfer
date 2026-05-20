//! Z-Image-Turbo pipeline orchestrator. Single entry point for CLI and web:
//! `ZImageModel::load(...)` builds the bundle once, `generate(...)` runs the
//! whole pipeline (tokenize -> Qwen3 encode -> noise -> scheduler step loop
//! with DiT -> VAE tiled decode -> PNG bytes).
//!
//! Owns: compiled `BlockPipelines` (shared by encoder, DiT, embedders),
//! `VaeDecoder` (handles + pipelines + tile cfg), DiT handle bundle, encoder
//! handle bundle, residency, backend, tokenizer. No model internals leak
//! through the public API past `generate`'s `Vec<u8>` PNG return.
//!
//! Turbo specifics baked in:
//! - guidance_scale = 0 (no CFG; positive prompt only).
//! - patch_size=2, f_patch_size=1, c_latent=16.
//! - VAE scale factor = 8 (height/width must be divisible by 16 because the
//!   DiT patch size is 2 on top of VAE's 8).

use std::sync::Arc;
use thinfer_core::backend::{Backend, WgpuBackend, WgpuError};
use thinfer_core::ops::{WeightDtype, WgslConfig};
use thinfer_core::residency::{ResidencyError, WeightResidency};
use thinfer_core::tokenizer::{Tokenizer, TokenizerError};
use thinfer_core::trace;
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::Workspace;

use crate::z_image::block::BlockPipelines;
use crate::z_image::dit::{DitInputs, DitShape, ZImageDit};
use crate::z_image::loader::{LoadError, register_dit_handles};
use crate::z_image::scheduler::FlowMatchEulerScheduler;
use crate::z_image::text_encoder::{
    EmbedLookupError, Qwen3Encoder, Qwen3ForwardError, Qwen3Handles, register_qwen3_handles,
};
use crate::z_image::tokenizer::format_qwen3_prompt;
use crate::z_image::vae::{
    VaeDecodeError, VaeDecoder, VaeDecoderPipelines, VaeTileConfig, register_vae_decoder_handles,
};

/// Qwen3 rope precomputed-table size. Hard-caps the post-chat-template prompt
/// length; Z-Image's `max_position_embeddings` is 1536 (same as upstream).
const MAX_PROMPT_TOKENS: usize = 1536;
const PATCH_SIZE: usize = 2;
const F_PATCH_SIZE: usize = 1;
const C_LATENT: usize = 16;
const VAE_SCALE: usize = 8;

/// Inputs to one `generate` call.
pub struct GenerationParams {
    pub prompt: String,
    /// Image height in pixels. Must be divisible by 16.
    pub height: u32,
    /// Image width in pixels. Must be divisible by 16.
    pub width: u32,
    /// Inference steps. Z-Image-Turbo default is 8.
    pub steps: u32,
    /// Deterministic seed for the initial latent noise.
    pub seed: u64,
}

pub struct ZImageModel<S: WeightSource, T: Tokenizer> {
    backend: Arc<WgpuBackend>,
    residency: WeightResidency<S>,
    tokenizer: T,
    /// Block pipelines compiled with `BF16_PACKED`. Used by every DiT-side
    /// consumer (XEmbedder, CapEmbedder, TimestepEmbedder, Block, FinalLayer).
    block_pipelines: BlockPipelines,
    /// Block pipelines compiled with `BF16_QUANT_WRITES` (fp32 activation
    /// storage + RNE writes for parity against bf16-PyTorch). Used only by the
    /// Qwen3 text encoder, which stays on the untuned matmul/fp32-storage path
    /// for now; bf16-packing the encoder is queued for a follow-up.
    encoder_block_pipelines: BlockPipelines,
    dit_handles: crate::z_image::loader::LoadedDitHandles,
    encoder: Qwen3Encoder,
    encoder_handles: Qwen3Handles,
    vae: VaeDecoder,
}

#[derive(Debug)]
pub enum GenerateError<SE: core::fmt::Debug> {
    Tokenizer(TokenizerError),
    Embed(EmbedLookupError),
    Encoder(Qwen3ForwardError<SE>),
    Dit(crate::z_image::dit::DitError<SE>),
    Vae(VaeDecodeError<SE>),
    Wgpu(WgpuError),
    Residency(ResidencyError<SE, WgpuError>),
    Png(String),
    InvalidDims { height: u32, width: u32 },
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

impl<SE: core::fmt::Debug> From<Qwen3ForwardError<SE>> for GenerateError<SE> {
    fn from(e: Qwen3ForwardError<SE>) -> Self {
        Self::Encoder(e)
    }
}

impl<SE: core::fmt::Debug> From<crate::z_image::dit::DitError<SE>> for GenerateError<SE> {
    fn from(e: crate::z_image::dit::DitError<SE>) -> Self {
        Self::Dit(e)
    }
}

impl<SE: core::fmt::Debug> From<VaeDecodeError<SE>> for GenerateError<SE> {
    fn from(e: VaeDecodeError<SE>) -> Self {
        Self::Vae(e)
    }
}

#[derive(Debug)]
pub enum ModelLoadError {
    Dit(LoadError),
    Encoder(crate::z_image::text_encoder::LoadError),
    Wgpu(WgpuError),
}

impl From<LoadError> for ModelLoadError {
    fn from(e: LoadError) -> Self {
        Self::Dit(e)
    }
}

impl From<crate::z_image::text_encoder::LoadError> for ModelLoadError {
    fn from(e: crate::z_image::text_encoder::LoadError) -> Self {
        Self::Encoder(e)
    }
}

impl From<WgpuError> for ModelLoadError {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}

impl<S: WeightSource, T: Tokenizer> ZImageModel<S, T> {
    /// Build the model. Registers all DiT + Qwen3 + VAE handles with the
    /// shared residency, compiles every WGSL kernel once. No bytes flow until
    /// `generate` triggers `acquire`.
    pub async fn load(
        backend: Arc<WgpuBackend>,
        residency: WeightResidency<S>,
        tokenizer: T,
    ) -> Result<Self, ModelLoadError> {
        let t0 = std::time::Instant::now();
        let dit_handles = register_dit_handles(&residency)?;
        let encoder_handles = register_qwen3_handles(&residency)?;
        let vae_handles = register_vae_decoder_handles(&residency)?;
        tracing::debug!(
            elapsed_ms = t0.elapsed().as_millis() as u64,
            "handles registered"
        );
        let t_compile = std::time::Instant::now();
        let wgsl_cfg = WgslConfig::BF16_PACKED;
        let block_pipelines = BlockPipelines::compile(&backend, &wgsl_cfg).await?;
        let encoder_cfg = WgslConfig {
            bf16_quant_writes: crate::z_image::manifest::RECIPE.bf16_quant_writes,
            act_dtype: thinfer_core::ops::ActDtype::F32,
            weight_dtype: WeightDtype::Bf16,
        };
        let encoder_block_pipelines = BlockPipelines::compile(&backend, &encoder_cfg).await?;
        let vae_pipelines = VaeDecoderPipelines::compile(&backend).await?;
        tracing::info!(
            compile_ms = t_compile.elapsed().as_millis() as u64,
            total_ms = t0.elapsed().as_millis() as u64,
            "ZImageModel loaded"
        );
        let encoder = Qwen3Encoder::new(MAX_PROMPT_TOKENS);
        let vae = VaeDecoder {
            pipelines: vae_pipelines,
            handles: vae_handles,
            tile_cfg: VaeTileConfig::default(),
        };
        Ok(Self {
            backend,
            residency,
            tokenizer,
            block_pipelines,
            encoder_block_pipelines,
            dit_handles,
            encoder,
            encoder_handles,
            vae,
        })
    }

    /// Run the full pipeline. Returns PNG bytes; the caller writes them to
    /// disk (CLI) or to a `Blob` (web) without touching model internals.
    pub async fn generate(
        &self,
        params: &GenerationParams,
    ) -> Result<Vec<u8>, GenerateError<S::Error>> {
        let t_gen = std::time::Instant::now();
        let mut workspace = Workspace::new(Arc::clone(&self.backend));
        let (sample, h_lat, w_lat) = self
            .denoise_with(params, None, &mut workspace, None)
            .await?;

        // VAE decode -> RGB CHW fp32 in [-1, 1]. Workspace carries over from
        // denoise so the DiT-phase buffer pool feeds VAE allocations (and
        // doesn't leak - Workspace has no Drop).
        let rgb = {
            let _s = tracing::info_span!("vae_decode", h_lat = h_lat, w_lat = w_lat).entered();
            let t = std::time::Instant::now();
            let out = self
                .vae
                .decode(
                    &self.backend,
                    &self.residency,
                    &mut workspace,
                    &sample,
                    h_lat,
                    w_lat,
                )
                .await?;
            tracing::info!(
                elapsed_ms = t.elapsed().as_millis() as u64,
                "vae decode done"
            );
            out
        };

        // Diag dump: gated on DIAG-target INFO enablement so the stat passes
        // (full sweep over `rgb`) don't fire when tracing is off. Zero-cost in
        // the disabled path: `tracing::enabled!` is a cached metadata lookup.
        if tracing::enabled!(target: trace::DIAG, tracing::Level::INFO) {
            let mut nan = 0usize;
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            let mut sum = 0f64;
            for &v in &rgb {
                if v.is_nan() {
                    nan += 1;
                } else {
                    if v < mn {
                        mn = v;
                    }
                    if v > mx {
                        mx = v;
                    }
                    sum += v as f64;
                }
            }
            let mean = sum / ((rgb.len() - nan) as f64);
            tracing::info!(
                target: trace::DIAG,
                "  [GEN-DUMP] vae_rgb len={} nan={} min={:.5e} max={:.5e} mean={:.5e} (gray if mean approx 0 and max-min much less than 2)",
                rgb.len(),
                nan,
                mn,
                mx,
                mean,
            );
            let plane = (params.width as usize) * (params.height as usize);
            for c in 0..3 {
                let first: Vec<f32> = rgb[c * plane..c * plane + 8].to_vec();
                tracing::info!(target: trace::DIAG, "  [GEN-DUMP] vae_rgb ch{c} first8: {:?}", first);
            }
        }

        // fp32 -> u8 RGB interleaved + PNG.
        let png = {
            let _s = tracing::debug_span!("png_encode").entered();
            encode_png(&rgb, params.width, params.height).map_err(GenerateError::Png)?
        };
        tracing::info!(
            elapsed_ms = t_gen.elapsed().as_millis() as u64,
            png_bytes = png.len(),
            "generate done"
        );
        Ok(png)
    }

    /// VAE decode a pre-VAE latent to CHW fp32 RGB in `[-1, 1]`. Mirrors
    /// the post-denoise half of `generate()` without the PNG encode. Caller
    /// owns `workspace` (same convention as `denoise_with`): the RAII pool
    /// returns buffers on `WsBuf` drop and frees physical memory when
    /// `Workspace` itself drops.
    pub async fn decode_latents_to_rgb(
        &self,
        latents: &[f32],
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
                latents,
                h_lat,
                w_lat,
            )
            .await?)
    }

    /// Like `decode_latents_to_rgb`, but also captures per-stage diag
    /// samples from inside the VAE decoder. See
    /// `VaeDecoder::decode_with_diag` for the sample format and the
    /// bounded-readback discipline.
    pub async fn decode_latents_to_rgb_with_diag(
        &self,
        latents: &[f32],
        h_lat: usize,
        w_lat: usize,
        workspace: &mut Workspace<WgpuBackend>,
        diag_sink: &mut Vec<crate::z_image::vae::VaeStageSample>,
    ) -> Result<Vec<f32>, GenerateError<S::Error>> {
        Ok(self
            .vae
            .decode_with_diag(
                &self.backend,
                &self.residency,
                workspace,
                latents,
                h_lat,
                w_lat,
                diag_sink,
            )
            .await?)
    }

    /// Tokenize -> Qwen3 encode -> scheduler step loop. Returns the final
    /// pre-VAE latent (CHW fp32, length `LATENT_CHANNELS * h_lat * w_lat`)
    /// plus the latent spatial dims so the caller can plumb them into VAE.
    /// Caller owns `workspace` so the GPU scratch pool is preserved across
    /// the DiT->VAE seam (and is freed in one place when the `Workspace`
    /// drops).
    ///
    /// `initial_noise`: when `Some`, used verbatim as the starting `sample`
    /// (must have length `LATENT_CHANNELS * h_lat * w_lat`). When `None`,
    /// derived deterministically from `params.seed` via Box-Muller. Used by
    /// the `dit_parity` integration test to drive byte-identical noise on
    /// both engine and reference sides.
    ///
    /// `step_dumps`: when `Some`, after each scheduler step the new sample
    /// (post-step prev_sample) is cloned into the vec. Cleared on entry. The
    /// final entry equals the returned pre-VAE latent. Used by the
    /// `e2e_parity` integration test.
    pub async fn denoise_with(
        &self,
        params: &GenerationParams,
        initial_noise: Option<&[f32]>,
        workspace: &mut Workspace<WgpuBackend>,
        mut step_dumps: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<(Vec<f32>, usize, usize), GenerateError<S::Error>> {
        if let Some(sink) = step_dumps.as_deref_mut() {
            sink.clear();
        }
        if !params
            .height
            .is_multiple_of(VAE_SCALE as u32 * PATCH_SIZE as u32)
            || !params
                .width
                .is_multiple_of(VAE_SCALE as u32 * PATCH_SIZE as u32)
            || params.height == 0
            || params.width == 0
        {
            return Err(GenerateError::InvalidDims {
                height: params.height,
                width: params.width,
            });
        }
        let h_lat = (params.height as usize) / VAE_SCALE;
        let w_lat = (params.width as usize) / VAE_SCALE;
        tracing::info!(
            height = params.height,
            width = params.width,
            steps = params.steps,
            seed = params.seed,
            injected_noise = initial_noise.is_some(),
            "denoise start"
        );

        let _denoise = trace::scope!("denoise").entered();

        // 1. Tokenize. Chat-template wrap then encode.
        let token_ids = {
            let _s = trace::scope!("tokenize").entered();
            let wrapped = format_qwen3_prompt(&params.prompt);
            let ids = self
                .tokenizer
                .encode(&wrapped)
                .map_err(GenerateError::Tokenizer)?;
            tracing::debug!(n_tokens = ids.len(), "tokenize done");
            if ids.len() > MAX_PROMPT_TOKENS {
                return Err(GenerateError::PromptTooLong {
                    tokens: ids.len(),
                    max: MAX_PROMPT_TOKENS,
                });
            }
            ids
        };

        // 2. Qwen3 encode -> cap features.
        let qout = {
            let _s = trace::scope!("text_encode", tokens = token_ids.len()).entered();
            let t = std::time::Instant::now();
            let out = self
                .encoder
                .forward(
                    &self.backend,
                    &self.encoder_block_pipelines,
                    &self.residency,
                    &*workspace,
                    &self.encoder_handles,
                    self.residency.source(),
                    &token_ids,
                )
                .await?;
            tracing::info!(
                elapsed_ms = t.elapsed().as_millis() as u64,
                seq = out.seq,
                "text encode done"
            );
            out
        };
        // Phase boundary: text encoder weights are dead for the rest of this
        // call. Evict to the residency pool so DiT acquires reuse the slots.
        self.residency.evict_all_and_free(&*self.backend);

        // 3. Initial noise: [16, 1, h_lat, w_lat] standard normal.
        let n_lat = C_LATENT * h_lat * w_lat;
        let mut sample: Vec<f32> = match initial_noise {
            Some(buf) => {
                assert_eq!(
                    buf.len(),
                    n_lat,
                    "initial_noise length {} != expected {}",
                    buf.len(),
                    n_lat
                );
                buf.to_vec()
            }
            None => gaussian_noise(n_lat, params.seed),
        };
        if tracing::enabled!(target: trace::DIAG, tracing::Level::INFO) {
            let s = &sample;
            let smin = s.iter().copied().fold(f32::INFINITY, f32::min);
            let smax = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let smean = s.iter().sum::<f32>() / s.len() as f32;
            tracing::info!(
                target: trace::DIAG,
                "  [OURS-DUMP] starting_latents: len={} min={:.5e} max={:.5e} max_abs={:.5e} mean={:.5e}",
                s.len(),
                smin,
                smax,
                smax.abs().max(smin.abs()),
                smean,
            );
        }

        // 4. Assemble DiT for this image shape + scheduler.
        let shape = DitShape::for_image(C_LATENT, h_lat, w_lat, qout.seq, PATCH_SIZE, F_PATCH_SIZE);
        let dit = ZImageDit::assemble(self.dit_handles.clone(), shape);
        let scheduler = FlowMatchEulerScheduler::new(params.steps as usize, shape.seq_x);
        tracing::info!(
            target: trace::DIAG,
            "  [OURS-DUMP] sigmas (n={}): {:?}",
            scheduler.sigmas().len(),
            scheduler.sigmas()
        );

        // 5. Step loop. Turbo is guidance_scale=0: one DiT call per step
        // (positive prompt only, no CFG concat).
        let _steps_span =
            trace::scope!("diffusion_steps", steps = params.steps, seq_x = shape.seq_x).entered();
        for i in 0..params.steps as usize {
            let _step = trace::scope!("step", i = i, t = scheduler.t_norm(i)).entered();
            let t_step = std::time::Instant::now();
            let inputs = DitInputs {
                image: &sample,
                size: (C_LATENT, 1, h_lat, w_lat),
                cap_feats: &qout.hidden,
                cap_len: qout.seq,
                timestep: scheduler.t_norm(i),
                patch_size: PATCH_SIZE,
                f_patch_size: F_PATCH_SIZE,
            };
            let layout = {
                let _f = trace::scope!("dit_forward").entered();
                dit.forward(
                    &self.backend,
                    &self.block_pipelines,
                    &self.residency,
                    &*workspace,
                    &inputs,
                )
                .await?
            };
            let total_rows = (layout.seq_x_padded + layout.seq_cap_padded) as u64;
            let row_bytes = (layout.out_channels as u64) * layout.act_dtype.bytes_per_elem();
            let bytes = {
                let _r = trace::scope!("dit_readback", bytes = total_rows * row_bytes).entered();
                self.backend
                    .read_buffer(
                        layout.final_out.id,
                        layout.final_out.offset,
                        total_rows * row_bytes,
                    )
                    .await?
            };
            let mut out = dit.decode_image(&layout, &bytes);
            debug_assert_eq!(out.image.len(), n_lat);
            // Upstream Z-Image pipelines negate the transformer output before
            // the Euler step (`pipeline_z_image.py:559`,
            // `Z-Image/src/zimage/pipeline.py:274`: `noise_pred = -noise_pred`).
            // The model is trained to predict `-velocity`; the scheduler then
            // applies `x += dt * noise_pred` with positive direction.
            for v in out.image.iter_mut() {
                *v = -*v;
            }
            // [DUMP] pre-step state for multi-step parity diagnosis. Stat
            // sweeps gated on DIAG enablement (zero-cost when off).
            if tracing::enabled!(target: trace::DIAG, tracing::Level::INFO) {
                let s = &sample;
                let m = &out.image;
                let smin = s.iter().copied().fold(f32::INFINITY, f32::min);
                let smax = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let smean = s.iter().sum::<f32>() / s.len() as f32;
                let mmin = m.iter().copied().fold(f32::INFINITY, f32::min);
                let mmax = m.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mmean = m.iter().sum::<f32>() / m.len() as f32;
                let dt = scheduler.sigmas()[i + 1] - scheduler.sigmas()[i];
                tracing::info!(
                    target: trace::DIAG,
                    "  [OURS-DUMP] step{i} sigma={:.6} sigma_next={:.6} dt={:.6}",
                    scheduler.sigmas()[i],
                    scheduler.sigmas()[i + 1],
                    dt,
                );
                tracing::info!(
                    target: trace::DIAG,
                    "  [OURS-DUMP] step{i}.model_output (post-negation): min={smin_m:.5e} max={smax_m:.5e} max_abs={mab:.5e} mean={mmean:.5e}",
                    smin_m = mmin,
                    smax_m = mmax,
                    mab = mmax.abs().max(mmin.abs()),
                    mmean = mmean,
                );
                tracing::info!(
                    target: trace::DIAG,
                    "  [OURS-DUMP] step{i}.sample_in: min={smin:.5e} max={smax:.5e} max_abs={sab:.5e} mean={smean:.5e}",
                    sab = smax.abs().max(smin.abs()),
                );
            }
            scheduler.step(i, &out.image, &mut sample);
            if let Some(sink) = step_dumps.as_deref_mut() {
                sink.push(sample.clone());
            }
            if tracing::enabled!(target: trace::DIAG, tracing::Level::INFO) {
                let s = &sample;
                let smin = s.iter().copied().fold(f32::INFINITY, f32::min);
                let smax = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let smean = s.iter().sum::<f32>() / s.len() as f32;
                tracing::info!(
                    target: trace::DIAG,
                    "  [OURS-DUMP] step{i}.prev_sample: min={smin:.5e} max={smax:.5e} max_abs={sab:.5e} mean={smean:.5e}",
                    sab = smax.abs().max(smin.abs()),
                );
            }
            tracing::info!(
                elapsed_ms = t_step.elapsed().as_millis() as u64,
                "step done"
            );
        }

        // Phase boundary: DiT block weights are dead until next inference.
        // Evict so VAE decode's allocations reuse the pool's slots.
        self.residency.evict_all_and_free(&*self.backend);
        Ok((sample, h_lat, w_lat))
    }
}

/// CHW fp32 in `[-1, 1]` -> interleaved RGB u8 -> PNG bytes. Single allocation
/// for the interleaved buffer; png crate writes into a `Vec<u8>` writer.
pub fn encode_png(chw: &[f32], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let plane = (width as usize) * (height as usize);
    if chw.len() != 3 * plane {
        return Err(format!(
            "encode_png: expected {} fp32 values, got {}",
            3 * plane,
            chw.len()
        ));
    }
    let mut rgb = vec![0u8; 3 * plane];
    for i in 0..plane {
        for c in 0..3 {
            let v = chw[c * plane + i];
            let scaled = ((v.clamp(-1.0, 1.0) + 1.0) * 127.5).round();
            rgb[i * 3 + c] = scaled as u8;
        }
    }
    let mut out = Vec::with_capacity(rgb.len() / 4);
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|e| format!("png header: {e}"))?;
        writer
            .write_image_data(&rgb)
            .map_err(|e| format!("png write: {e}"))?;
    }
    Ok(out)
}

/// Deterministic standard-normal samples via SplitMix64 -> Box-Muller. Avoids
/// pulling in `rand` for one consumer. Output is `Vec<f32>` of length `n`.
fn gaussian_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next_u64 = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    // Convert two uniforms in (0,1] to two N(0,1) via Box-Muller.
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
