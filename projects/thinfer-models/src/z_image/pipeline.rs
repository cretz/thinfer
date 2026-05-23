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
use thinfer_core::tensor::StorageEncoding;
use thinfer_core::tokenizer::{Tokenizer, TokenizerError};
use thinfer_core::trace;
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::Workspace;

use crate::z_image::block::{BlockPipelines, BlockWgslConfigs};
use crate::z_image::dit::{Block0Taps, DitInputs, DitShape, DitTaps, ZImageDit};
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
    /// Per-main-layer block pipelines. Each entry holds the five matmul
    /// kernels (qkv, proj, ffn_up, ffn_down, adaln) compiled with the
    /// encoding probed from that layer's weights. For a uniform-encoding
    /// file (Q8_0, bf16) all entries are identical and the WgpuPipeline
    /// cache deduplicates the compiled WGSL behind them; for mixed
    /// Q4_K_M files the per-(layer, slot) encoding selects the right
    /// kernel at dispatch time.
    block_pipelines: Vec<BlockPipelines>,
    /// Block pipelines compiled with `BF16_QUANT_WRITES` (fp32 activation
    /// storage + RNE writes for parity against bf16-PyTorch). Used only by the
    /// Qwen3 text encoder, which stays on the untuned matmul/fp32-storage path
    /// for now; bf16-packing the encoder is queued for a follow-up.
    encoder_block_pipelines: BlockPipelines,
    /// Block pipelines for the DiT-side encoder ops (x/t/cap embedders, noise
    /// and context refiners, final_layer). Shares `act_dtype` with
    /// `block_pipelines` (their outputs feed directly into the main loop) but
    /// keeps `weight_dtype = Bf16` because refiners/embedders aren't quantized
    /// even in the GGUF path.
    dit_encoder_block_pipelines: BlockPipelines,
    dit_handles: crate::z_image::loader::LoadedDitHandles,
    encoder: Qwen3Encoder,
    encoder_handles: Qwen3Handles,
    vae: VaeDecoder,
    /// Dtype the four main DiT matmul kernels were compiled against -
    /// `Bf16` when the source returned bf16 (or any non-quant encoding),
    /// `Quant(k)` when the source returned a GGUF-style quant scheme. Set
    /// once at `load()` from the `layers.0.attention.qkv.weight` probe;
    /// exposed via `dit_matmul_weight()` so callers (notably the
    /// e2e_parity GGUF variant) can assert the GGUF source actually
    /// supplied the matmul tensors instead of silently falling through
    /// to the safetensors side of a `UnionSource`.
    dit_matmul_weight: WeightDtype,
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
        // Detect whether the DiT-side matmul tensors arrived as GGUF
        // quant (e.g. Q8_0) or stayed bf16. Peek the canonical fused QKV
        // tensor (`layers.0.attention.qkv.weight`) in the source catalog. If
        // it's quant, the 4 main matmul kernels are compiled with that
        // scheme; AdaLN stays bf16 (its weights are too small to amortize
        // dequant and we keep them in safetensors regardless).
        // Per-(layer, slot) encoding probe. Z-Image-Turbo Q4_K_M files
        // ship mixed quant per layer: the first/last two layers use one
        // scheme (e.g. Q6_K for qkv) while the other 28 use another
        // (e.g. Q5_K). For uniform-quant files (Q8_0, bf16) all probes
        // return the same encoding and the WgpuPipeline cache collapses
        // the compiled kernels.
        let probe_slot = |id: &thinfer_core::weight::WeightId| -> WeightDtype {
            match residency
                .source()
                .catalog()
                .get(id)
                .and_then(|e| e.encoding)
            {
                Some(StorageEncoding::Quant(k)) => WeightDtype::Quant(k),
                _ => WeightDtype::Bf16,
            }
        };
        let mut per_layer_weights: Vec<[WeightDtype; 4]> =
            Vec::with_capacity(crate::z_image::config::N_LAYERS);
        for li in 0..crate::z_image::config::N_LAYERS {
            let bw = crate::z_image::BlockWeights::new(crate::z_image::BlockKind::Main, li);
            per_layer_weights.push([
                probe_slot(&bw.attn_qkv),
                probe_slot(&bw.attn_to_out),
                probe_slot(&bw.ffn_w1),
                probe_slot(&bw.ffn_w2),
            ]);
        }
        // Layer-0 qkv encoding is the headline matmul weight reported via
        // `dit_matmul_weight()`. Used by tests to assert the GGUF source
        // actually supplied the matmul tensors and (for Q4_K_M) to pick
        // the right tolerance band.
        let dit_matmul_weight = per_layer_weights[0][0];
        // Activation dtype selection for the Quant matmul path:
        // - When the adapter exposes `SHADER_F16`, use `ActDtype::F16`:
        //   native `vec2<f16>` storage halves activation VRAM and bandwidth
        //   vs F32, pointwise kernels run as native f16 SIMD, and the
        //   matmul dequant narrows f32 accumulators to f16 on write.
        // - Otherwise fall back to `ActDtype::F32` — the safe baseline.
        // Bf16 acts on the Quant path are intentionally NOT chosen: bf16's
        // 7-bit mantissa loses precision at every kernel boundary in a way
        // f16 (10 bits) does not, and the bf16-packed workaround was tied
        // to bit-clean parity with bf16 weights.
        //
        // AdaLN matmul keeps Bf16 weights but follows the block's chosen
        // act dtype (the `BlockWgslConfigs` invariant requires all matmuls
        // and ops to agree on act_dtype + bf16_quant_writes).
        // Whether the file is "quant-flavored" overall — true as long as
        // any main-layer matmul slot probed as Quant. Activation dtype is
        // chosen once for the whole DiT: F16 when shader-f16 is available
        // and any quant kernel is present (bf16 storage at f16 boundaries
        // would lose precision relative to a native f16 path); F32
        // baseline otherwise. The choice is uniform across all layers so
        // activation buffers feeding one layer into the next stay
        // type-consistent.
        let any_quant = per_layer_weights
            .iter()
            .any(|slots| slots.iter().any(|s| matches!(s, WeightDtype::Quant(_))));
        let (quant_act_dtype, ops_template_bf16w): (thinfer_core::ops::ActDtype, WgslConfig) =
            if any_quant {
                let a = if backend.supports_shader_f16() {
                    thinfer_core::ops::ActDtype::F16
                } else {
                    thinfer_core::ops::ActDtype::F32
                };
                tracing::info!(
                    ?a,
                    shader_f16 = backend.supports_shader_f16(),
                    "DiT quant matmul activation dtype",
                );
                let bf16w = WgslConfig {
                    bf16_quant_writes: crate::z_image::manifest::RECIPE.bf16_quant_writes,
                    act_dtype: a,
                    weight_dtype: WeightDtype::Bf16,
                };
                (a, bf16w)
            } else {
                (thinfer_core::ops::ActDtype::Bf16, WgslConfig::BF16_PACKED)
            };
        // Build per-layer configs. Within each layer the four main slots
        // (qkv, proj, ffn_up, ffn_down) carry their probed weight dtype;
        // adaln stays bf16; ops template is uniform.
        let mut block_pipelines: Vec<BlockPipelines> =
            Vec::with_capacity(crate::z_image::config::N_LAYERS);
        let mk_main = |wd: WeightDtype| -> WgslConfig {
            if any_quant {
                WgslConfig {
                    bf16_quant_writes: crate::z_image::manifest::RECIPE.bf16_quant_writes,
                    act_dtype: quant_act_dtype,
                    weight_dtype: wd,
                }
            } else {
                WgslConfig {
                    weight_dtype: wd,
                    ..WgslConfig::BF16_PACKED
                }
            }
        };
        for slots in &per_layer_weights {
            let cfgs = BlockWgslConfigs {
                matmul_qkv: mk_main(slots[0]),
                matmul_proj: mk_main(slots[1]),
                matmul_ffn_up: mk_main(slots[2]),
                matmul_ffn_down: mk_main(slots[3]),
                matmul_adaln: ops_template_bf16w,
                ops: ops_template_bf16w,
            };
            block_pipelines.push(BlockPipelines::compile(&backend, &cfgs).await?);
        }
        // Headline "main matmul cfg" for downstream encoder/ops sizing.
        // act_dtype + bf16_quant_writes are uniform across all layers so
        // any entry works; use layer 0's qkv slot.
        let dit_main_matmul_cfg = mk_main(per_layer_weights[0][0]);
        tracing::info!(
            ?dit_matmul_weight,
            "DiT block matmul weight dtype (layer 0 qkv probe)"
        );
        // Qwen3 text encoder pipelines: fp32 acts + bf16 weights, untuned
        // matmul path. Independent of the DiT path's dtype choice.
        let encoder_cfgs = BlockWgslConfigs::uniform(WgslConfig {
            bf16_quant_writes: crate::z_image::manifest::RECIPE.bf16_quant_writes,
            act_dtype: thinfer_core::ops::ActDtype::F32,
            weight_dtype: WeightDtype::Bf16,
        });
        let encoder_block_pipelines = BlockPipelines::compile(&backend, &encoder_cfgs).await?;
        // DiT-side encoder ops (x/t/cap embedders + refiners + final_layer):
        // must share `act_dtype` with the main DiT loop because their outputs
        // feed directly into the main layers' activation buffers. Weights
        // stay bf16 — refiners/embedders are never quantized in the GGUF
        // path that quantizes the main matmuls.
        let dit_encoder_cfgs = BlockWgslConfigs::uniform(WgslConfig {
            bf16_quant_writes: dit_main_matmul_cfg.bf16_quant_writes,
            act_dtype: dit_main_matmul_cfg.act_dtype,
            weight_dtype: WeightDtype::Bf16,
        });
        let dit_encoder_block_pipelines =
            BlockPipelines::compile(&backend, &dit_encoder_cfgs).await?;
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
            dit_encoder_block_pipelines,
            dit_handles,
            encoder,
            encoder_handles,
            vae,
            dit_matmul_weight,
        })
    }

    /// Dtype the four main DiT matmul kernels picked up at load time -
    /// `Bf16` for the safetensors path, `Quant(k)` when a GGUF source
    /// surfaced the matmul tensors. Lets tests assert the GGUF union
    /// actually fed the DiT rather than silently falling through to
    /// the safetensors side.
    pub fn dit_matmul_weight(&self) -> WeightDtype {
        self.dit_matmul_weight
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
            let vae_ms = t.elapsed().as_millis() as u64;
            tracing::info!(elapsed_ms = vae_ms, "vae decode done");
            eprintln!("[thinfer] vae_decode: {:.2}s", vae_ms as f64 / 1000.0);
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
        let gen_ms = t_gen.elapsed().as_millis() as u64;
        tracing::info!(elapsed_ms = gen_ms, png_bytes = png.len(), "generate done");
        eprintln!("[thinfer] total: {:.2}s", gen_ms as f64 / 1000.0);
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
            let text_ms = t.elapsed().as_millis() as u64;
            tracing::info!(elapsed_ms = text_ms, seq = out.seq, "text encode done");
            eprintln!("[thinfer] text_encode: {:.2}s", text_ms as f64 / 1000.0);
            out
        };
        // Phase boundary: text encoder weights are dead for the rest of this
        // call. Evict to the residency pool so DiT acquires reuse the slots,
        // and drain the workspace pool so size classes from text_encode
        // (cap-shaped) don't sit live in VRAM while DiT allocates its own
        // (image-token-shaped) working set.
        self.residency.evict_all_and_free(&*self.backend);
        workspace.drain_pool();

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
            // DiT taps: when DIAG tracing is enabled, capture intermediate
            // f32 readbacks at well-defined points inside `dit.forward` and
            // print per-tap NaN/finite/min/max/mean stats. Sinks are allocated
            // only when DIAG fires, so the non-diag path stays zero-cost.
            // Useful for narrowing the first NaN-producing op when a new
            // weight encoding (Q8_0, future Q4_K) breaks parity.
            let diag = tracing::enabled!(target: trace::DIAG, tracing::Level::INFO);
            let mut tap_main0: Vec<f32> = Vec::new();
            let mut tap_main14: Vec<f32> = Vec::new();
            let mut tap_unified: Vec<f32> = Vec::new();
            let mut tap_last_main: Vec<f32> = Vec::new();
            let mut tap_final: Vec<f32> = Vec::new();
            let mut tap_ctx0: Vec<f32> = Vec::new();
            let mut tap_last_ctx: Vec<f32> = Vec::new();
            let mut tap_cap_emb: Vec<f32> = Vec::new();
            // Block-0 per-op taps: request all of them so we can pinpoint
            // the first NaN-producing op inside the first main DiT block.
            // Setting Some(Vec::new()) flags the field; the engine fills it.
            let mut block0_taps = Block0Taps {
                adaln_input: Some(Vec::new()),
                adaln_pre: Some(Vec::new()),
                adaln_full: Some(Vec::new()),
                scale_msa: Some(Vec::new()),
                gate_msa: Some(Vec::new()),
                scale_mlp: Some(Vec::new()),
                gate_mlp: Some(Vec::new()),
                attn_norm1_out: Some(Vec::new()),
                modulated_attn_in: Some(Vec::new()),
                attn_q: Some(Vec::new()),
                attn_k: Some(Vec::new()),
                attn_v: Some(Vec::new()),
                attn_q_norm: Some(Vec::new()),
                attn_k_norm: Some(Vec::new()),
                attn_q_rope: Some(Vec::new()),
                attn_k_rope: Some(Vec::new()),
                attn_sdpa: Some(Vec::new()),
                attn_out: Some(Vec::new()),
                attn_norm2_out: Some(Vec::new()),
                x_mid: Some(Vec::new()),
                ffn_norm1_out: Some(Vec::new()),
                modulated_ffn_in: Some(Vec::new()),
                ffn_raw: Some(Vec::new()),
                ffn_norm2_out: Some(Vec::new()),
            };
            // ctx_refiner block 0 per-op taps. modulation=false so the
            // adaln_* / scale_* / gate_* fields stay empty after the run;
            // request only the ops that actually fire on this block.
            let mut ctx_block0_taps = Block0Taps {
                attn_norm1_out: Some(Vec::new()),
                modulated_attn_in: Some(Vec::new()),
                attn_q: Some(Vec::new()),
                attn_k: Some(Vec::new()),
                attn_v: Some(Vec::new()),
                attn_q_norm: Some(Vec::new()),
                attn_k_norm: Some(Vec::new()),
                attn_q_rope: Some(Vec::new()),
                attn_k_rope: Some(Vec::new()),
                attn_sdpa: Some(Vec::new()),
                attn_out: Some(Vec::new()),
                attn_norm2_out: Some(Vec::new()),
                x_mid: Some(Vec::new()),
                ffn_norm1_out: Some(Vec::new()),
                modulated_ffn_in: Some(Vec::new()),
                ffn_raw: Some(Vec::new()),
                ffn_norm2_out: Some(Vec::new()),
                ..Block0Taps::default()
            };
            let layout = {
                let _f = trace::scope!("dit_forward").entered();
                if diag {
                    let taps = DitTaps {
                        cap_embedded: Some(&mut tap_cap_emb),
                        ctx_refiner_0_out: Some(&mut tap_ctx0),
                        last_ctx_refiner_out: Some(&mut tap_last_ctx),
                        unified_in: Some(&mut tap_unified),
                        main_layer_0_out: Some(&mut tap_main0),
                        main_layer_14_out: Some(&mut tap_main14),
                        last_main_layer_out: Some(&mut tap_last_main),
                        final_layer_out: Some(&mut tap_final),
                        block0: Some(&mut block0_taps),
                        ctx_block0: Some(&mut ctx_block0_taps),
                        ..DitTaps::default()
                    };
                    dit.forward_with_taps(
                        &self.backend,
                        &self.block_pipelines,
                        &self.dit_encoder_block_pipelines,
                        &self.residency,
                        &*workspace,
                        &inputs,
                        taps,
                    )
                    .await?
                } else {
                    dit.forward(
                        &self.backend,
                        &self.block_pipelines,
                        &self.dit_encoder_block_pipelines,
                        &self.residency,
                        &*workspace,
                        &inputs,
                    )
                    .await?
                }
            };
            if diag {
                let print = |label: &str, v: &[f32]| {
                    if v.is_empty() {
                        return;
                    }
                    // Whole-buffer stats. Split nan / +inf / -inf / exact-zero
                    // so "all zero" vs "bf16 0x0000" vs "stale NaN" vs "+inf
                    // saturation" are distinguishable in the log; the previous
                    // single `nan/inf` counter hid the bf16-readback bug.
                    let mut nan = 0usize;
                    let mut pinf = 0usize;
                    let mut ninf = 0usize;
                    let mut zeros = 0usize;
                    let mut min = f32::INFINITY;
                    let mut max = f32::NEG_INFINITY;
                    let mut sum_abs = 0.0f64;
                    let mut n_fin = 0usize;
                    for &x in v {
                        if x.is_nan() {
                            nan += 1;
                            continue;
                        }
                        if x == f32::INFINITY {
                            pinf += 1;
                            continue;
                        }
                        if x == f32::NEG_INFINITY {
                            ninf += 1;
                            continue;
                        }
                        if x == 0.0 {
                            zeros += 1;
                        }
                        n_fin += 1;
                        if x < min {
                            min = x;
                        }
                        if x > max {
                            max = x;
                        }
                        sum_abs += x.abs() as f64;
                    }
                    let mean_abs = if n_fin > 0 {
                        sum_abs / n_fin as f64
                    } else {
                        0.0
                    };
                    // 4-bucket mean_abs along the buffer axis. Exposes
                    // first-half / second-half asymmetry (under-dispatch, byte
                    // miscount, slab-order swap) that a whole-buffer mean
                    // averages away.
                    let n = v.len();
                    let mut b_means = [0.0f64; 4];
                    for (bi, b) in b_means.iter_mut().enumerate() {
                        let lo = bi * n / 4;
                        let hi = (bi + 1) * n / 4;
                        let mut s = 0.0f64;
                        let mut c = 0usize;
                        for &x in &v[lo..hi] {
                            if x.is_finite() {
                                s += x.abs() as f64;
                                c += 1;
                            }
                        }
                        *b = if c > 0 { s / c as f64 } else { 0.0 };
                    }
                    // Head / tail samples expose structural patterns: e.g.
                    // bf16 0x0000 reads exactly 0.0; bf16 0xffff reads NaN;
                    // an upload-byte-order bug usually shows mirrored values.
                    let head_n = 8.min(n);
                    let tail_lo = n.saturating_sub(8);
                    tracing::info!(
                        target: trace::DIAG,
                        "  [DIT-TAP] step{i} {label}: len={n} nan={nan} +inf={pinf} -inf={ninf} \
                         zeros={zeros} min={:.4e} max={:.4e} mean_abs={:.4e} \
                         buckets=[{:.4e},{:.4e},{:.4e},{:.4e}] head={:?} tail={:?}",
                        min, max, mean_abs,
                        b_means[0], b_means[1], b_means[2], b_means[3],
                        &v[..head_n], &v[tail_lo..],
                    );
                };
                // Per-row bucketed mean_abs for matmul-output taps. The
                // whole-buffer mean averages a magnitude blow-up across all
                // rows; bucketing by row exposes whether the blow-up is
                // uniform (weight-scale / dequant issue) vs concentrated in
                // a contiguous row range (under-dispatch / stride bug) vs
                // alternating (interleaved-write bug).
                let print_rows = |label: &str, v: &[f32], rows: usize, cols: usize| {
                    if v.is_empty() || rows == 0 || cols == 0 || v.len() != rows * cols {
                        return;
                    }
                    let nb = 8usize.min(rows);
                    let mut bm = vec![0.0f64; nb];
                    for (bi, b) in bm.iter_mut().enumerate() {
                        let r_lo = bi * rows / nb;
                        let r_hi = (bi + 1) * rows / nb;
                        let mut s = 0.0f64;
                        let mut c = 0usize;
                        for r in r_lo..r_hi {
                            for x in &v[r * cols..(r + 1) * cols] {
                                if x.is_finite() {
                                    s += x.abs() as f64;
                                    c += 1;
                                }
                            }
                        }
                        *b = if c > 0 { s / c as f64 } else { 0.0 };
                    }
                    tracing::info!(
                        target: trace::DIAG,
                        "  [DIT-TAP-ROWS] step{i} {label}: rows={rows} cols={cols} row_buckets={bm:?}",
                    );
                };
                print("cap_embedded", &tap_cap_emb);
                print("ctx_refiner_0_out", &tap_ctx0);
                print("last_ctx_refiner_out", &tap_last_ctx);
                print("unified_in (pre main layer 0)", &tap_unified);
                print("main_layer_0_out", &tap_main0);
                print("main_layer_14_out", &tap_main14);
                print("last_main_layer_out", &tap_last_main);
                print("final_layer_out", &tap_final);
                // Per-op narrowing within main layer block 0.
                let b0 = &block0_taps;
                if let Some(v) = &b0.adaln_input {
                    print("block0.adaln_input", v);
                }
                if let Some(v) = &b0.adaln_pre {
                    print("block0.adaln_pre (matmul out)", v);
                }
                if let Some(v) = &b0.adaln_full {
                    print("block0.adaln_full (post bias)", v);
                }
                if let Some(v) = &b0.scale_msa {
                    print("block0.scale_msa", v);
                }
                if let Some(v) = &b0.gate_msa {
                    print("block0.gate_msa", v);
                }
                if let Some(v) = &b0.scale_mlp {
                    print("block0.scale_mlp", v);
                }
                if let Some(v) = &b0.gate_mlp {
                    print("block0.gate_mlp", v);
                }
                if let Some(v) = &b0.attn_norm1_out {
                    print("block0.attn_norm1_out", v);
                }
                if let Some(v) = &b0.modulated_attn_in {
                    print("block0.modulated_attn_in", v);
                }
                // Row/col dims for matmul-output taps within block 0.
                // `dim` is the unified-stream channel count, `head_dim` is
                // per-head, `n_heads` is the attention head count. These
                // are config constants (z_image::config) — not runtime-only
                // — so importing them here for per-row bucketing is cheap.
                let dim = crate::z_image::config::DIM;
                let head_dim = crate::z_image::config::HEAD_DIM;
                let n_heads = crate::z_image::config::N_HEADS;
                let ffn_hidden = crate::z_image::config::FFN_HIDDEN;
                let seq_u = shape.seq_x + shape.seq_cap;
                if let Some(v) = &b0.attn_q {
                    print("block0.attn_q", v);
                    print_rows("block0.attn_q", v, seq_u * n_heads, head_dim);
                }
                if let Some(v) = &b0.attn_k {
                    print("block0.attn_k", v);
                }
                if let Some(v) = &b0.attn_v {
                    print("block0.attn_v", v);
                }
                if let Some(v) = &b0.attn_q_norm {
                    print("block0.attn_q_norm", v);
                }
                if let Some(v) = &b0.attn_k_norm {
                    print("block0.attn_k_norm", v);
                }
                if let Some(v) = &b0.attn_q_rope {
                    print("block0.attn_q_rope", v);
                }
                if let Some(v) = &b0.attn_k_rope {
                    print("block0.attn_k_rope", v);
                }
                if let Some(v) = &b0.attn_sdpa {
                    print("block0.attn_sdpa", v);
                }
                if let Some(v) = &b0.attn_out {
                    print("block0.attn_out", v);
                    print_rows("block0.attn_out", v, seq_u, dim);
                }
                if let Some(v) = &b0.attn_norm2_out {
                    print("block0.attn_norm2_out", v);
                }
                if let Some(v) = &b0.x_mid {
                    print("block0.x_mid", v);
                }
                if let Some(v) = &b0.ffn_norm1_out {
                    print("block0.ffn_norm1_out", v);
                }
                if let Some(v) = &b0.modulated_ffn_in {
                    print("block0.modulated_ffn_in", v);
                }
                if let Some(v) = &b0.ffn_raw {
                    print("block0.ffn_raw", v);
                    print_rows("block0.ffn_raw", v, seq_u, dim);
                }
                let _ = ffn_hidden;
                if let Some(v) = &b0.ffn_norm2_out {
                    print("block0.ffn_norm2_out", v);
                }
                // ctx_refiner block 0 per-op narrowing. modulation=false so
                // adaln_*/scale_*/gate_* slots are empty by design; only the
                // path ops are printed. Row-bucketing uses seq_cap so the
                // padding-row tail (text padding) splits out from real rows.
                let cb0 = &ctx_block0_taps;
                let seq_cap = shape.seq_cap;
                if let Some(v) = &cb0.attn_norm1_out {
                    print("ctx_block0.attn_norm1_out", v);
                    print_rows("ctx_block0.attn_norm1_out", v, seq_cap, dim);
                }
                if let Some(v) = &cb0.modulated_attn_in {
                    print("ctx_block0.modulated_attn_in", v);
                }
                if let Some(v) = &cb0.attn_q {
                    print("ctx_block0.attn_q", v);
                    print_rows("ctx_block0.attn_q", v, seq_cap * n_heads, head_dim);
                }
                if let Some(v) = &cb0.attn_k {
                    print("ctx_block0.attn_k", v);
                }
                if let Some(v) = &cb0.attn_v {
                    print("ctx_block0.attn_v", v);
                }
                if let Some(v) = &cb0.attn_q_norm {
                    print("ctx_block0.attn_q_norm", v);
                    print_rows("ctx_block0.attn_q_norm", v, seq_cap * n_heads, head_dim);
                }
                if let Some(v) = &cb0.attn_k_norm {
                    print("ctx_block0.attn_k_norm", v);
                }
                if let Some(v) = &cb0.attn_q_rope {
                    print("ctx_block0.attn_q_rope", v);
                    print_rows("ctx_block0.attn_q_rope", v, seq_cap * n_heads, head_dim);
                }
                if let Some(v) = &cb0.attn_k_rope {
                    print("ctx_block0.attn_k_rope", v);
                }
                if let Some(v) = &cb0.attn_sdpa {
                    print("ctx_block0.attn_sdpa", v);
                    print_rows("ctx_block0.attn_sdpa", v, seq_cap * n_heads, head_dim);
                }
                if let Some(v) = &cb0.attn_out {
                    print("ctx_block0.attn_out", v);
                    print_rows("ctx_block0.attn_out", v, seq_cap, dim);
                }
                if let Some(v) = &cb0.attn_norm2_out {
                    print("ctx_block0.attn_norm2_out", v);
                    print_rows("ctx_block0.attn_norm2_out", v, seq_cap, dim);
                }
                if let Some(v) = &cb0.x_mid {
                    print("ctx_block0.x_mid", v);
                    print_rows("ctx_block0.x_mid", v, seq_cap, dim);
                }
                if let Some(v) = &cb0.ffn_norm1_out {
                    print("ctx_block0.ffn_norm1_out", v);
                    print_rows("ctx_block0.ffn_norm1_out", v, seq_cap, dim);
                }
                if let Some(v) = &cb0.modulated_ffn_in {
                    print("ctx_block0.modulated_ffn_in", v);
                }
                if let Some(v) = &cb0.ffn_raw {
                    print("ctx_block0.ffn_raw", v);
                    print_rows("ctx_block0.ffn_raw", v, seq_cap, dim);
                }
                if let Some(v) = &cb0.ffn_norm2_out {
                    print("ctx_block0.ffn_norm2_out", v);
                    print_rows("ctx_block0.ffn_norm2_out", v, seq_cap, dim);
                }
            }
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
            let step_ms = t_step.elapsed().as_millis() as u64;
            tracing::info!(elapsed_ms = step_ms, "step done");
            eprintln!(
                "[thinfer] step {}/{}: dit={:.2}s",
                i + 1,
                params.steps,
                step_ms as f64 / 1000.0,
            );
        }

        // Phase boundary: DiT block weights are dead until next inference.
        // Evict so VAE decode's allocations reuse the pool's slots, and drain
        // workspace size classes so DiT's bigger activation buffers aren't
        // held idle while VAE allocates its own.
        self.residency.evict_all_and_free(&*self.backend);
        workspace.drain_pool();
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
