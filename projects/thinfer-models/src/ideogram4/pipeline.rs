//! Ideogram-4 end-to-end pipeline: Qwen3-VL-8B encode (13-tap) -> single-stream
//! DiT no-CFG Euler denoise loop -> Flux2 KL VAE decode -> PNG.
//!
//! The turbotime LoRA removes CFG, so this is one conditional DiT forward per
//! step (`v = pos_v`, `z += v*delta`); there is no unconditional transformer.
//! Mirrors `Ideogram4Pipeline.__call__` (`pipeline_ideogram4.py`) with the
//! no-CFG simplification.
//!
//! Weights come from one residency over a union of the encoder GGUF (Qwen3-VL
//! Q8_0), the DiT (the LoRA-folded GGUF -> bf16 matmul sites), and the FLUX.2
//! VAE safetensors. The three phases are run with phase-aware eviction
//! (`evict_all_and_free` between encode / denoise / decode) so peak VRAM is
//! ~one model, not the sum.
//!
//! Tokenization is the CALLER's job (the e2e parity gate reads pyref-dumped
//! token ids; the CLI wiring will supply a Qwen3-VL chat-template tokenizer):
//! `generate` takes `token_ids` directly.

use std::sync::Arc;

use thinfer_core::arbiter::RECLAIM_EVICTABLE_WEIGHTS;
use thinfer_core::backend::{WgpuBackend, WgpuError};
use thinfer_core::ops::{ActDtype, WeightDtype, WgslConfig};
use thinfer_core::residency::{ResidencyError, WeightResidency};
use thinfer_core::tensor::StorageEncoding;
use thinfer_core::trace;
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::Workspace;

use crate::common::block::{BlockPipelines, BlockWgslConfigs, DenseActSites};
use crate::common::loader::LoadError;
use crate::z_image::pipeline::encode_png;

use super::dit::{DitInputs, Ideogram4Dit};
use super::loader::{DitHandles, register_handles as register_dit_handles};
use super::packing::{ImageGrid, image_grid};
use super::sampler::{DEFAULT_MU, DEFAULT_STD, build_steps};
use super::text_encoder::{
    self, Qwen3VlEncoder, config as enc_config, register_handles as register_encoder_handles,
};
use super::vae::{
    IDEOGRAM4_KL_VAE, VaeDecodeError, VaeDecoder, VaeDecoderPipelines, VaeTileConfig,
    register_vae_decoder_handles, unpatch_denorm,
};
use crate::z_image::text_encoder::Qwen3Handles;

const PATCH: usize = 16; // patch_size(2) * ae_scale_factor(8)

/// Inputs to one `generate` call.
pub struct GenerationParams {
    /// Prompt token ids (Qwen3-VL chat-templated; supplied by the caller).
    pub token_ids: Vec<u32>,
    /// Image height in pixels. Must be divisible by 16.
    pub height: u32,
    /// Image width in pixels. Must be divisible by 16.
    pub width: u32,
    /// Inference steps (turbotime LoRA: 2 / 4 / 8).
    pub steps: u32,
    /// Deterministic seed for the initial latent noise (ignored if
    /// `initial_noise` is supplied).
    pub seed: u64,
    /// Schedule `mu` (resolution-aware mean offset).
    pub mu: f64,
    /// Schedule `std`.
    pub std: f64,
}

impl GenerationParams {
    /// Defaults for a prompt: 1024x1024, 8 steps, TURBO_12 schedule constants.
    pub fn new(token_ids: Vec<u32>, height: u32, width: u32, steps: u32, seed: u64) -> Self {
        Self {
            token_ids,
            height,
            width,
            steps,
            seed,
            mu: DEFAULT_MU,
            std: DEFAULT_STD,
        }
    }
}

/// Progress notifications, mirroring `z_image::pipeline::ProgressEvent`.
#[derive(Clone, Copy, Debug)]
pub enum ProgressEvent {
    TextEncode,
    Step { i: u32, n: u32 },
    VaeDecode,
}

pub type ProgressFn<'a> = Option<&'a dyn Fn(ProgressEvent)>;

pub struct Ideogram4Pipeline<S: WeightSource> {
    backend: Arc<WgpuBackend>,
    residency: WeightResidency<S>,
    encoder: Qwen3VlEncoder,
    encoder_handles: Qwen3Handles,
    encoder_pipelines: BlockPipelines,
    dit_handles: DitHandles,
    /// Main DiT block pipelines: probed weight dtype (Q8_0 raw, or bf16 folded),
    /// F16 acts, `large_d_sdpa` (head_dim 256). DP4A is on for the safe sites
    /// (qkv + ffn_up) when `i8_matmul`, off (dense_acts all) otherwise.
    dit_main_pipelines: BlockPipelines,
    /// Dense pipelines for the embedders + final layer (bf16 weights, acts
    /// matching the main set).
    dit_dense_pipelines: BlockPipelines,
    vae: VaeDecoder,
}

#[derive(Debug)]
pub enum ModelLoadError {
    Load(LoadError),
    EncoderLoad(crate::z_image::text_encoder::LoadError),
    Wgpu(WgpuError),
}

impl From<LoadError> for ModelLoadError {
    fn from(e: LoadError) -> Self {
        Self::Load(e)
    }
}
impl From<crate::z_image::text_encoder::LoadError> for ModelLoadError {
    fn from(e: crate::z_image::text_encoder::LoadError) -> Self {
        Self::EncoderLoad(e)
    }
}
impl From<WgpuError> for ModelLoadError {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}

#[derive(Debug)]
pub enum GenerateError<SE: core::fmt::Debug> {
    Encoder(text_encoder::EncodeError<SE>),
    Dit(super::dit::DitError<SE>),
    Vae(VaeDecodeError<SE>),
    Wgpu(WgpuError),
    Residency(ResidencyError<SE, WgpuError>),
    Png(String),
    InvalidDims { height: u32, width: u32 },
    EmptyPrompt,
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
impl<SE: core::fmt::Debug> From<text_encoder::EncodeError<SE>> for GenerateError<SE> {
    fn from(e: text_encoder::EncodeError<SE>) -> Self {
        Self::Encoder(e)
    }
}
impl<SE: core::fmt::Debug> From<super::dit::DitError<SE>> for GenerateError<SE> {
    fn from(e: super::dit::DitError<SE>) -> Self {
        Self::Dit(e)
    }
}
impl<SE: core::fmt::Debug> From<VaeDecodeError<SE>> for GenerateError<SE> {
    fn from(e: VaeDecodeError<SE>) -> Self {
        Self::Vae(e)
    }
}

impl<S: WeightSource> Ideogram4Pipeline<S> {
    /// Register all handles + compile every kernel once. No bytes flow until
    /// `generate` triggers `acquire`.
    pub async fn load(
        backend: Arc<WgpuBackend>,
        residency: WeightResidency<S>,
        i8_matmul: bool,
    ) -> Result<Self, ModelLoadError> {
        residency.arbiter().register(
            RECLAIM_EVICTABLE_WEIGHTS,
            residency.reclaimer(Arc::clone(&backend)),
        );

        // Encoder: Q8_0 in the GGUF (no transcode), bf16 acts (Qwen3 massive
        // activations crush under f16; same rationale as the encoder gate).
        let encoder_handles = register_encoder_handles(&residency, None)?;
        let dit_handles = register_dit_handles(&residency)?;
        let vae_handles = register_vae_decoder_handles(&residency)?;

        // --- encoder pipelines (bf16 acts; probe the GGUF weight dtype so the
        // canary Q8_0 and the runtime-default Q5_K_M both work). The Qwen3
        // encoder GGUF quantizes its big linears uniformly, so one probed dtype
        // drives every matmul site; norms/embeds stay dense (passthrough). ---
        let enc_ops = WgslConfig {
            bf16_quant_writes: false,
            act_dtype: ActDtype::Bf16,
            weight_dtype: WeightDtype::Bf16,
        };
        let enc_weight = match residency
            .source()
            .catalog()
            .get(&WeightId(
                "model.layers.0.self_attn.q_proj.weight".to_string(),
            ))
            .and_then(|e| e.encoding)
        {
            Some(StorageEncoding::Quant(k)) => WeightDtype::Quant(k),
            _ => WeightDtype::Bf16,
        };
        let enc_mm = WgslConfig {
            weight_dtype: enc_weight,
            ..enc_ops
        };
        tracing::info!(?enc_weight, "ideogram4 encoder matmul dtype");
        let enc_cfgs = BlockWgslConfigs {
            matmul_qkv: enc_mm,
            matmul_qkv_self: enc_mm,
            matmul_proj: enc_mm,
            matmul_ffn_up: enc_mm,
            matmul_ffn_down: enc_mm,
            matmul_adaln: enc_ops,
            ops: enc_ops,
            i8_sdpa: false,
            dense_acts: DenseActSites::default(),
            coopmat_acts: crate::common::block::CoopmatSites::default(),
            large_d_sdpa: false,
            fast_sdpa: false,
        };
        let encoder_pipelines = BlockPipelines::compile(&backend, &enc_cfgs).await?;

        // --- DiT pipelines: probe layer-0 qkv weight dtype (Q8_0 raw, or bf16
        // when the LoRA fold republished it). F16 acts, head_dim 256 ->
        // SdpaF32LargeD, DP4A off (dense_acts) so Q8_0 sites take the
        // dequant-once path. ---
        let act = if backend.supports_shader_f16() {
            ActDtype::F16
        } else {
            ActDtype::F32
        };
        let dit_weight = match residency
            .source()
            .catalog()
            .get(&WeightId("layers.0.attention.qkv.weight".to_string()))
            .and_then(|e| e.encoding)
        {
            Some(StorageEncoding::Quant(k)) => WeightDtype::Quant(k),
            _ => WeightDtype::Bf16,
        };
        tracing::info!(?dit_weight, ?act, i8_matmul, "ideogram4 DiT matmul dtype");
        let dit_ops = WgslConfig {
            bf16_quant_writes: false,
            act_dtype: act,
            weight_dtype: WeightDtype::Bf16,
        };
        let dit_main = WgslConfig {
            weight_dtype: dit_weight,
            ..dit_ops
        };
        // The block matmul weights are already Quant (Q8_0 from the fold/GGUF),
        // so the DP4A path needs no transcode: opting a site OUT of `dense_acts`
        // routes its (normed, outlier-free) A-side through `act_quant` ->
        // `matmul_i8` (dot4I8Packed). qkv + ffn_up A-sides are sandwich-normed
        // (DP4A-safe); proj (attn-out) + ffn_down (SwiGLU product) carry outlier
        // channels that per-32 i8 act-quant would crush, so they stay dense. The
        // i8 matmul still emits paired vec2<f16>, so head_dim-256 large-D SDPA
        // (F16-only) is unaffected. `i8_matmul=false` keeps every site dense
        // (the bf16-reference path), mirroring the video `--no-i8-matmul`.
        let dense_acts = if i8_matmul {
            DenseActSites {
                qkv: false,
                qkv_self: false,
                proj: true,
                ffn_up: false,
                ffn_down: true,
            }
        } else {
            DenseActSites {
                qkv: true,
                qkv_self: true,
                proj: true,
                ffn_up: true,
                ffn_down: true,
            }
        };
        let main_cfgs = BlockWgslConfigs {
            matmul_qkv: dit_main,
            matmul_qkv_self: dit_main,
            matmul_proj: dit_main,
            matmul_ffn_up: dit_main,
            matmul_ffn_down: dit_main,
            matmul_adaln: dit_ops,
            ops: dit_ops,
            i8_sdpa: false,
            dense_acts,
            coopmat_acts: crate::common::block::CoopmatSites::default(),
            large_d_sdpa: true,
            fast_sdpa: false,
        };
        let dit_main_pipelines = BlockPipelines::compile(&backend, &main_cfgs).await?;
        let dit_dense_pipelines =
            BlockPipelines::compile(&backend, &BlockWgslConfigs::uniform(dit_ops)).await?;

        let vae_pipelines = VaeDecoderPipelines::compile(&backend).await?;
        let vae = VaeDecoder {
            pipelines: vae_pipelines,
            handles: vae_handles,
            tile_cfg: VaeTileConfig::default(),
            arch: IDEOGRAM4_KL_VAE,
        };

        // Rope table spans the max prompt length (1-axis; text positions only).
        let encoder = Qwen3VlEncoder::new(enc_config::N_TAPS.max(super::config::MAX_TEXT_TOKENS));

        Ok(Self {
            backend,
            residency,
            encoder,
            encoder_handles,
            encoder_pipelines,
            dit_handles,
            dit_main_pipelines,
            dit_dense_pipelines,
            vae,
        })
    }

    pub fn arbiter(&self) -> &Arc<thinfer_core::arbiter::MemArbiter> {
        self.residency.arbiter()
    }

    /// Run the whole pipeline, returning PNG bytes. `initial_noise`, when
    /// `Some`, is the starting latent `[num_image, 128]` (the e2e gate injects
    /// pyref's `torch.randn`); otherwise it is derived from `params.seed`.
    pub async fn generate(
        &self,
        params: &GenerationParams,
        initial_noise: Option<&[f32]>,
        progress: ProgressFn<'_>,
    ) -> Result<Vec<u8>, GenerateError<S::Error>> {
        let (rgb, _z, _grid) = self.generate_rgb(params, initial_noise, progress).await?;
        encode_png(&rgb, params.width, params.height).map_err(GenerateError::Png)
    }

    /// The pipeline body: encode -> denoise -> VAE, returning the raw VAE RGB
    /// (CHW f32 in `[-1, 1]`), the final latent (`[num_image, 128]`), and the
    /// image grid. The e2e parity gate compares the RGB against the staged
    /// pyref; `generate` wraps this with the PNG encode.
    pub async fn generate_rgb(
        &self,
        params: &GenerationParams,
        initial_noise: Option<&[f32]>,
        progress: ProgressFn<'_>,
    ) -> Result<(Vec<f32>, Vec<f32>, ImageGrid), GenerateError<S::Error>> {
        if params.token_ids.is_empty() {
            return Err(GenerateError::EmptyPrompt);
        }
        if !params.height.is_multiple_of(PATCH as u32)
            || !params.width.is_multiple_of(PATCH as u32)
            || params.height == 0
            || params.width == 0
        {
            return Err(GenerateError::InvalidDims {
                height: params.height,
                width: params.width,
            });
        }
        let grid = image_grid(params.height as usize, params.width as usize).map_err(|_| {
            GenerateError::InvalidDims {
                height: params.height,
                width: params.width,
            }
        })?;
        let num_image = grid.num_image_tokens();

        let mut workspace = Workspace::new(
            Arc::clone(&self.backend),
            Arc::clone(self.residency.arbiter()),
        );

        // --- 1. encode ---
        if let Some(p) = progress {
            p(ProgressEvent::TextEncode);
        }
        let qout = {
            let _s = trace::scope!("ideogram.encode", tokens = params.token_ids.len()).entered();
            self.encoder
                .encode(
                    &self.backend,
                    &self.encoder_pipelines,
                    &self.residency,
                    &workspace,
                    &self.encoder_handles,
                    self.residency.source(),
                    &params.token_ids,
                    false,
                )
                .await?
        };
        let num_text = qout.seq;
        // Phase boundary: encoder weights dead; free for the DiT phase.
        self.residency.evict_all_and_free(&*self.backend);
        workspace.drain_pool();

        // --- 2. DiT no-CFG Euler denoise loop ---
        let seq = num_text + num_image;
        let dit = Ideogram4Dit::assemble(self.dit_handles.clone(), seq);
        // Fill VRAM to budget: pin a deterministic prefix of the DiT weights for
        // the denoise phase so steps 2..N skip their re-upload (the ring streams
        // only the remainder). Pin-on-first-touch -- step 1 uploads as usual,
        // pinned entries just aren't evicted afterwards. The headroom reserve
        // keeps the rings, the packer's workspace, and one staging buffer out of
        // the pinned span. This is per-request (mirrors `ZImageDit`): the
        // `evict_all_and_free` after the loop drops the plan so the VAE phase
        // (and any following request) reclaims the VRAM -- no cross-request
        // residue.
        {
            let vram = self.residency.budget().vram_bytes;
            let reserve = self
                .residency
                .ring_reserved_bytes()
                .saturating_add(dit.workspace_reserve_estimate(&self.dit_main_pipelines))
                .saturating_add(self.residency.staging_reserve_bytes());
            let pin_budget = vram.saturating_sub(reserve);
            let (pin_bytes, pin_count) =
                self.residency.set_pin_plan(&dit.pin_priority(), pin_budget);
            tracing::info!(
                pinned_mib = pin_bytes / (1024 * 1024),
                pinned_count = pin_count,
                pin_budget_mib = pin_budget / (1024 * 1024),
                reserve_mib = reserve / (1024 * 1024),
                "ideogram4 dit pin plan"
            );
        }
        let steps = build_steps(
            params.steps as usize,
            params.height as usize,
            params.width as usize,
            params.mu,
            params.std,
        );
        let mut z: Vec<f32> = match initial_noise {
            Some(buf) => {
                assert_eq!(
                    buf.len(),
                    num_image * super::config::IN_CHANNELS,
                    "noise len"
                );
                buf.to_vec()
            }
            None => gaussian_noise(num_image * super::config::IN_CHANNELS, params.seed),
        };
        {
            let _s = trace::scope!("ideogram.denoise", steps = params.steps, seq).entered();
            for (i, step) in steps.iter().enumerate() {
                if let Some(p) = progress {
                    p(ProgressEvent::Step {
                        i: i as u32 + 1,
                        n: params.steps,
                    });
                }
                let inputs = DitInputs {
                    llm_features: &qout.features,
                    num_text,
                    noise: &z,
                    grid,
                    timestep: step.t_val as f32,
                };
                let velocity = dit
                    .forward(
                        &self.backend,
                        &self.dit_main_pipelines,
                        &self.dit_dense_pipelines,
                        &self.residency,
                        &workspace,
                        &inputs,
                    )
                    .await?;
                debug_assert_eq!(velocity.len(), z.len());
                let delta = step.delta as f32;
                for (zi, &v) in z.iter_mut().zip(velocity.iter()) {
                    *zi += v * delta;
                }
            }
        }
        // Phase boundary: DiT weights dead; free for VAE.
        self.residency.evict_all_and_free(&*self.backend);
        workspace.drain_pool();

        // --- 3. VAE decode -> RGB CHW [-1,1] ---
        if let Some(p) = progress {
            p(ProgressEvent::VaeDecode);
        }
        let spatial = unpatch_denorm(&z, grid.grid_h, grid.grid_w);
        let rgb = self
            .vae
            .decode(
                &self.backend,
                &self.residency,
                &mut workspace,
                &spatial,
                grid.grid_h * 2,
                grid.grid_w * 2,
            )
            .await?;
        self.residency.evict_all_and_free(&*self.backend);
        Ok((rgb, z, grid))
    }
}

/// Deterministic standard-normal noise (SplitMix64 -> Box-Muller). Mirrors
/// `z_image::pipeline`'s private helper; kept local to avoid widening that API.
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
