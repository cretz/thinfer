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
use thinfer_core::arbiter::{MemArbiter, RECLAIM_EVICTABLE_WEIGHTS};
use thinfer_core::backend::{Backend, WgpuBackend, WgpuError};
use thinfer_core::mem::VramCategory;
use thinfer_core::ops::{WeightDtype, WgslConfig};
use thinfer_core::residency::{ResidencyError, WeightResidency};
use thinfer_core::tensor::StorageEncoding;
use thinfer_core::tokenizer::{Tokenizer, TokenizerError};
use thinfer_core::trace;
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::Workspace;

use crate::common::block::{BlockPipelines, BlockWgslConfigs, DenseActSites};
use crate::common::loader::LoadError;
use crate::z_image::dit::{Block0Taps, DitInputs, DitShape, DitTaps, ZImageDit};
use crate::z_image::loader::register_dit_handles;
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

/// Stage notifications emitted during `generate` / `denoise_with` for
/// user-facing progress (CLI lines, web form updates). Distinct from
/// tracing: tracing is engine telemetry, this is product surface. The
/// callback is a plain `Fn` ref (no Send/Sync) so it works on
/// single-threaded wasm.
#[derive(Clone, Copy, Debug)]
pub enum ProgressEvent {
    /// Qwen3 prompt encode is starting.
    TextEncode,
    /// Diffusion step `i` of `n` is starting (1-based).
    Step { i: u32, n: u32 },
    /// VAE decode is starting.
    VaeDecode,
}

/// Optional progress sink. `None` is zero-cost.
pub type ProgressFn<'a> = Option<&'a dyn Fn(ProgressEvent)>;

/// DIAG sinks for step-0 only: captures the chain of intermediate
/// tensors between the DiT residual stream out of block N-1 and the
/// scheduler's `prev_sample`. Used by the parity test to linfit each
/// stage against pyref and localize where the slope-shrink appears.
#[derive(Default)]
pub struct Step0LocalizationTaps {
    /// Residual stream after block N-1, full `seq_u * dim` f32.
    pub last_main: Vec<f32>,
    /// `final_layer` output, full `seq_u * oc` f32.
    pub final_layer: Vec<f32>,
    /// Scheduler input at step 0: post-unpatchify, post-negation
    /// `[c_latent * h_lat * w_lat]` f32. Equivalent to the model's
    /// velocity estimate that the Euler step consumes.
    pub model_output_post_neg: Vec<f32>,
    /// `sigmas[1] - sigmas[0]` -- needed to back-derive pyref's
    /// velocity at step 0 from `prev_sample - starting_latents`.
    pub dt_step0: f32,
    /// Full residual stream after every main block (length 30, one
    /// `seq_u * dim` f32 vec per block). Used to plot per-block slope
    /// vs pyref and localize the first block where slope deviates.
    pub per_block_residual: Vec<Vec<f32>>,
    /// Per-op intermediates at main block 0 and block N-1. Same fields
    /// as the engine's `Block0Taps`; populated only at step 0.
    pub block0: Block0LocalTaps,
    pub block_last: Block0LocalTaps,
    /// Per-op intermediates at "damage zone" main blocks. Engine-side
    /// `extra_blocks` populates each `(idx, Block0LocalTaps)`. The
    /// parity test pre-fills the block indices it wants instrumented
    /// (typically 24..29) before the denoise call.
    pub damage_zone: Vec<(usize, Block0LocalTaps)>,
}

/// Per-op intermediates inside a DiT main block, decoded to f32. Each
/// `Vec<f32>` is full-tensor; empty when the engine didn't populate the
/// matching `Block0Taps` field (kernel doesn't fire in this branch).
#[derive(Default)]
pub struct Block0LocalTaps {
    pub adaln_input: Vec<f32>,
    pub adaln_pre: Vec<f32>,
    pub adaln_full: Vec<f32>,
    pub scale_msa: Vec<f32>,
    pub gate_msa: Vec<f32>,
    pub scale_mlp: Vec<f32>,
    pub gate_mlp: Vec<f32>,
    pub attn_norm1_out: Vec<f32>,
    pub modulated_attn_in: Vec<f32>,
    pub attn_q: Vec<f32>,
    pub attn_k: Vec<f32>,
    pub attn_v: Vec<f32>,
    pub attn_q_norm: Vec<f32>,
    pub attn_k_norm: Vec<f32>,
    pub attn_q_rope: Vec<f32>,
    pub attn_k_rope: Vec<f32>,
    pub attn_sdpa: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub attn_norm2_out: Vec<f32>,
    pub x_mid: Vec<f32>,
    pub ffn_norm1_out: Vec<f32>,
    pub modulated_ffn_in: Vec<f32>,
    pub ffn_raw: Vec<f32>,
    pub ffn_norm2_out: Vec<f32>,
    /// QKV-site byte snapshots for the matmul_i8 audit. Empty unless the
    /// corresponding `Block0Taps` byte-head fields were requested.
    pub qkv_attn_in_data_head: Vec<u8>,
    pub qkv_attn_in_params_head: Vec<u8>,
    pub qkv_b_i8_head: Vec<u8>,
    pub qkv_b_scale_head: Vec<u8>,
    pub qkv_b_qsum_head: Vec<u8>,
    pub qkv_dbg_trace_head: Vec<u8>,
    /// QKV-site f16 matmul output, decoded to f32. Mirrors `attn_qkv_f16_pre_quant`
    /// already used at block0 / block_last, exposed here for damage-zone audit.
    pub attn_qkv_f16_pre_quant: Vec<f32>,
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
    /// Qwen3 text-encoder pipeline set: matmul slots compiled against the
    /// load-time Q8_0 weight transcode (DP4A matmul_i8 / dequant fallback),
    /// F16 acts when the adapter has SHADER_F16 (F32 fallback). Chosen
    /// independently of the DiT path's dtype.
    encoder_block_pipelines: BlockPipelines,
    /// Block pipelines for the DiT-side encoder ops (x/t/cap embedders, noise
    /// and context refiners, final_layer). Shares `act_dtype` with
    /// `block_pipelines` (their outputs feed directly into the main loop) but
    /// keeps `weight_dtype = Bf16` because refiners/embedders aren't quantized
    /// even in the GGUF path.
    dit_encoder_block_pipelines: BlockPipelines,
    /// Pipeline set for the dense-input front-door ops: `XEmbedder` /
    /// `CapEmbedder` (their inputs are dense F32 uploaded patches / cap
    /// features) and `FinalLayer`. Mirrors `dit_encoder_block_pipelines`.
    dit_embedder_block_pipelines: BlockPipelines,
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
        // Stage-timing clocks are gated on subscriber interest: the values
        // only surface via the info/debug events, and on wasm each enabled
        // read is a JS roundtrip (`trace::Instant`). An INFO gate covers the
        // debug event too (a filter admitting DEBUG necessarily admits INFO).
        let timing = tracing::enabled!(tracing::Level::INFO);
        let t0 = timing.then(trace::Instant::now);
        // Weights join the VRAM arbiter's reclaim chain: workspace growth
        // can evict prefetch-warmed (unpinned) residents instead of
        // overshooting the budget. The inverse direction (weights evicting
        // idle workspace) is registered by each `Workspace::new`.
        residency.arbiter().register(
            RECLAIM_EVICTABLE_WEIGHTS,
            residency.reclaimer(Arc::clone(&backend)),
        );
        // Refiner matmuls ship bf16 in the unsloth GGUFs; when the main
        // path is quant-flavored, requantize them to Q8_0 at upload so
        // they ride the quant matmul path (see `refiner_transcode_target`).
        let refiner_transcode = crate::z_image::loader::refiner_transcode_target(&residency);
        let dit_handles = register_dit_handles(&residency, refiner_transcode)?;
        // Qwen3 encoder matmuls: GGUF TE checkpoints ship them quantized
        // in-file (registered natively); bf16 safetensors sources are
        // requantized to Q8_0 at upload (lossless-tier, ~half the upload
        // bytes, rides the quant matmul path). `encoder_cfgs` below probes
        // the catalog so the compiled slots match either way.
        let encoder_transcode = thinfer_core::quant::QuantKind::Q8_0;
        let encoder_handles = register_qwen3_handles(&residency, Some(encoder_transcode))?;
        let vae_handles = register_vae_decoder_handles(&residency)?;
        tracing::debug!(
            elapsed_ms = t0.map_or(0, |t| t.elapsed().as_millis() as u64),
            "handles registered"
        );
        let t_compile = timing.then(trace::Instant::now);
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
                // SHADER_F16 -> F16 acts (the production path); no
                // SHADER_F16 -> F32 fallback. I8 is never a block-wide act
                // dtype: it exists only inside matmul sites (DP4A act_quant
                // transcode) and the opt-in i8 attention below.
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
                    bf16_quant_writes: crate::z_image::manifest::current_recipe().bf16_quant_writes,
                    act_dtype: a,
                    weight_dtype: WeightDtype::Bf16,
                };
                (a, bf16w)
            } else {
                (thinfer_core::ops::ActDtype::Bf16, WgslConfig::BF16_PACKED)
            };
        // i8 attention: opt-in via `RECIPE.i8_sdpa`, requires SHADER_F16 and
        // the Quant path. Main DiT blocks only — the refiners/encoder run
        // short context sequences where i8 attention buys nothing.
        let i8_sdpa = crate::z_image::manifest::current_recipe().i8_sdpa
            && any_quant
            && backend.supports_shader_f16();
        tracing::info!(i8_sdpa, "DiT i8 attention opt-in");
        // Build per-layer configs. Within each layer the four main slots
        // (qkv, proj, ffn_up, ffn_down) carry their probed weight dtype;
        // adaln stays bf16; ops template is uniform.
        let mut block_pipelines: Vec<BlockPipelines> =
            Vec::with_capacity(crate::z_image::config::N_LAYERS);
        let mk_main = |wd: WeightDtype| -> WgslConfig {
            if any_quant {
                WgslConfig {
                    bf16_quant_writes: crate::z_image::manifest::current_recipe().bf16_quant_writes,
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
        let matmul_adaln_cfg = ops_template_bf16w;
        for slots in &per_layer_weights {
            let cfgs = BlockWgslConfigs {
                matmul_qkv: mk_main(slots[0]),
                matmul_qkv_self: mk_main(slots[0]),
                matmul_proj: mk_main(slots[1]),
                matmul_ffn_up: mk_main(slots[2]),
                matmul_ffn_down: mk_main(slots[3]),
                matmul_adaln: matmul_adaln_cfg,
                ops: ops_template_bf16w,
                i8_sdpa,
                dense_acts: DenseActSites::default(),
                coopmat_acts: crate::common::block::CoopmatSites::default(),
                large_d_sdpa: false,
                fast_sdpa: false,
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
        // Qwen3 text encoder pipelines: matmul slots compile against the
        // quant scheme (DP4A matmul_i8 when the adapter has it,
        // dequant-workspace fallback otherwise). Per-site catalog probe:
        // GGUF TE checkpoints answer Quant(k) (every site, ffn_down
        // included); bf16 safetensors answer Bf16 and the slot compiles
        // against the upload-time transcode target instead — except
        // ffn_down, which is never transcoded (massive-activation
        // amplification; see `register_qwen3_handles`) and stays bf16.
        // Acts are F16 with SHADER_F16, F32 fallback — same rule as the DiT
        // quant path, but chosen independently of the DiT's dtype.
        let encoder_act = if backend.supports_shader_f16() {
            thinfer_core::ops::ActDtype::F16
        } else {
            thinfer_core::ops::ActDtype::F32
        };
        let encoder_ops = WgslConfig {
            bf16_quant_writes: crate::z_image::manifest::current_recipe().bf16_quant_writes,
            act_dtype: encoder_act,
            weight_dtype: WeightDtype::Bf16,
        };
        let qwen3_l0 = crate::z_image::text_encoder::Qwen3Weights::new();
        let encoder_slot = |id, on_bf16| WgslConfig {
            weight_dtype: match probe_slot(id) {
                WeightDtype::Quant(k) => WeightDtype::Quant(k),
                _ => on_bf16,
            },
            ..encoder_ops
        };
        let transcoded = WeightDtype::Quant(encoder_transcode);
        let encoder_cfgs = BlockWgslConfigs {
            matmul_qkv: encoder_slot(&qwen3_l0.layers[0].q_proj, transcoded),
            matmul_qkv_self: encoder_slot(&qwen3_l0.layers[0].q_proj, transcoded),
            matmul_proj: encoder_slot(&qwen3_l0.layers[0].o_proj, transcoded),
            matmul_ffn_up: encoder_slot(&qwen3_l0.layers[0].mlp_gate, transcoded),
            matmul_ffn_down: encoder_slot(&qwen3_l0.layers[0].mlp_down, WeightDtype::Bf16),
            matmul_adaln: encoder_ops,
            ops: encoder_ops,
            i8_sdpa: false,
            dense_acts: DenseActSites::default(),
            coopmat_acts: crate::common::block::CoopmatSites::default(),
            large_d_sdpa: false,
            fast_sdpa: false,
        };
        let encoder_block_pipelines = BlockPipelines::compile(&backend, &encoder_cfgs).await?;
        // DiT-side encoder ops (x/t/cap embedders + refiners + final_layer):
        // must share `act_dtype` with the main DiT loop because their outputs
        // feed directly into the main layers' activation buffers. Refiner
        // matmuls follow the loader's transcode decision (bf16-in-file
        // requantized to Q8_0 on the quant path); everything else stays bf16.
        let dit_encoder_ops = WgslConfig {
            bf16_quant_writes: dit_main_matmul_cfg.bf16_quant_writes,
            act_dtype: dit_main_matmul_cfg.act_dtype,
            weight_dtype: WeightDtype::Bf16,
        };
        // Refiner blocks consume this set's qkv/proj/ffn slots; t_embedder
        // only touches the adaln slot, which stays bf16.
        let refiner_matmul_cfg = match refiner_transcode {
            Some(k) => WgslConfig {
                weight_dtype: WeightDtype::Quant(k),
                ..dit_encoder_ops
            },
            None => dit_encoder_ops,
        };
        let dit_encoder_cfgs = BlockWgslConfigs {
            matmul_qkv: refiner_matmul_cfg,
            matmul_qkv_self: refiner_matmul_cfg,
            matmul_proj: refiner_matmul_cfg,
            matmul_ffn_up: refiner_matmul_cfg,
            matmul_ffn_down: refiner_matmul_cfg,
            matmul_adaln: matmul_adaln_cfg,
            ops: dit_encoder_ops,
            i8_sdpa: false,
            dense_acts: DenseActSites::default(),
            coopmat_acts: crate::common::block::CoopmatSites::default(),
            large_d_sdpa: false,
            fast_sdpa: false,
        };
        let dit_encoder_block_pipelines =
            BlockPipelines::compile(&backend, &dit_encoder_cfgs).await?;
        // Embedder/final_layer front-door set: same ops/act dtype as the
        // dit-encoder set but all matmul slots stay bf16 (embedder and
        // final-layer linears are never transcoded).
        let dit_embedder_cfgs = BlockWgslConfigs {
            matmul_qkv: dit_encoder_ops,
            matmul_qkv_self: dit_encoder_ops,
            matmul_proj: dit_encoder_ops,
            matmul_ffn_up: dit_encoder_ops,
            matmul_ffn_down: dit_encoder_ops,
            matmul_adaln: matmul_adaln_cfg,
            ops: dit_encoder_ops,
            i8_sdpa: false,
            dense_acts: DenseActSites::default(),
            coopmat_acts: crate::common::block::CoopmatSites::default(),
            large_d_sdpa: false,
            fast_sdpa: false,
        };
        let dit_embedder_block_pipelines =
            BlockPipelines::compile(&backend, &dit_embedder_cfgs).await?;
        let vae_pipelines = VaeDecoderPipelines::compile(&backend).await?;
        tracing::info!(
            compile_ms = t_compile.map_or(0, |t| t.elapsed().as_millis() as u64),
            total_ms = t0.map_or(0, |t| t.elapsed().as_millis() as u64),
            "ZImageModel loaded"
        );
        let encoder = Qwen3Encoder::new(MAX_PROMPT_TOKENS);
        let vae = VaeDecoder {
            pipelines: vae_pipelines,
            handles: vae_handles,
            tile_cfg: VaeTileConfig::default(),
            arch: crate::z_image::vae::Z_IMAGE_KL_VAE,
        };
        Ok(Self {
            backend,
            residency,
            tokenizer,
            block_pipelines,
            encoder_block_pipelines,
            dit_encoder_block_pipelines,
            dit_embedder_block_pipelines,
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

    /// The VRAM budget arbiter shared by every client. Callers that build
    /// their own `Workspace` (e2e tests) must construct it with this so the
    /// budget has a single owner.
    pub fn arbiter(&self) -> &Arc<MemArbiter> {
        self.residency.arbiter()
    }

    /// Run the full pipeline. Returns PNG bytes; the caller writes them to
    /// disk (CLI) or to a `Blob` (web) without touching model internals.
    pub async fn generate(
        &self,
        params: &GenerationParams,
        progress: ProgressFn<'_>,
    ) -> Result<Vec<u8>, GenerateError<S::Error>> {
        // INFO-gated clocks: see `load` for the wasm rationale.
        let timing = tracing::enabled!(tracing::Level::INFO);
        let t_gen = timing.then(trace::Instant::now);
        let mut workspace = Workspace::new(
            Arc::clone(&self.backend),
            Arc::clone(self.residency.arbiter()),
        );
        let (sample, h_lat, w_lat) = self
            .denoise_with(params, None, &mut workspace, None, None, progress)
            .await?;

        // VAE decode -> RGB CHW fp32 in [-1, 1]. Workspace carries over from
        // denoise so the DiT-phase buffer pool feeds VAE allocations (and
        // doesn't leak - Workspace has no Drop).
        let rgb = {
            let _s = tracing::info_span!("vae_decode", h_lat = h_lat, w_lat = w_lat).entered();
            if let Some(p) = progress {
                p(ProgressEvent::VaeDecode);
            }
            let t = timing.then(trace::Instant::now);
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
            let vae_ms = t.map_or(0, |t| t.elapsed().as_millis() as u64);
            tracing::info!(elapsed_ms = vae_ms, "vae decode done");
            out
        };
        // Phase boundary: generation is done; nothing stays resident between
        // generates. Evicts the VAE weights (and any other unpinned residents)
        // so an idle model holds no VRAM; the workspace pool frees via RAII
        // when `workspace` drops at return. The next generate re-acquires from
        // the source on demand, same as a first run.
        self.residency.evict_all_and_free(&*self.backend);

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
        let gen_ms = t_gen.map_or(0, |t| t.elapsed().as_millis() as u64);
        tracing::info!(elapsed_ms = gen_ms, png_bytes = png.len(), "generate done");
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
        mut step0_taps: Option<&mut Step0LocalizationTaps>,
        progress: ProgressFn<'_>,
    ) -> Result<(Vec<f32>, usize, usize), GenerateError<S::Error>> {
        if let Some(sink) = step_dumps.as_deref_mut() {
            sink.clear();
        }
        // INFO-gated clocks: see `load` for the wasm rationale.
        let timing = tracing::enabled!(tracing::Level::INFO);
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
            // Chat-template specials are already literal text in `wrapped`; do
            // not let the tokenizer insert its own (would add a stray BOS/EOS).
            let ids = self
                .tokenizer
                .encode(&wrapped, false)
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
            if let Some(p) = progress {
                p(ProgressEvent::TextEncode);
            }
            let t = timing.then(trace::Instant::now);
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
            let text_ms = t.map_or(0, |t| t.elapsed().as_millis() as u64);
            tracing::info!(elapsed_ms = text_ms, seq = out.seq, "text encode done");
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
        // Fill VRAM to budget: pin a deterministic prefix of the DiT weights
        // for the denoise phase so steps 2..N skip their re-upload (the ring
        // streams only the remainder). Pin-on-first-touch - step 1 uploads as
        // usual, pinned entries just aren't evicted afterwards. The headroom
        // reserve keeps the rings, the packer's workspace, and one prep
        // staging buffer out of the pinned span; `evict_all_and_free` at the
        // end of denoise drops the plan so VAE reclaims the VRAM.
        {
            let vram = self.residency.budget().vram_bytes;
            let reserve = self
                .residency
                .ring_reserved_bytes()
                .saturating_add(dit.workspace_reserve_estimate(&self.block_pipelines))
                .saturating_add(self.residency.staging_reserve_bytes());
            let pin_budget = vram.saturating_sub(reserve);
            let (pin_bytes, pin_count) =
                self.residency.set_pin_plan(&dit.pin_priority(), pin_budget);
            tracing::info!(
                pinned_mib = pin_bytes / (1024 * 1024),
                pinned_count = pin_count,
                pin_budget_mib = pin_budget / (1024 * 1024),
                reserve_mib = reserve / (1024 * 1024),
                "dit pin plan"
            );
        }
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
            if let Some(p) = progress {
                p(ProgressEvent::Step {
                    i: i as u32 + 1,
                    n: params.steps,
                });
            }
            let t_step = timing.then(trace::Instant::now);
            let step_src0 = self.backend.mem_account().source_bytes_total();
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
            // Force diag for step 0 when the localization sink was requested,
            // even with TRACE off, so the test captures last_main / final
            // without needing THINFER_TRACE.
            let want_step0 = i == 0 && step0_taps.is_some();
            let diag = want_step0 || tracing::enabled!(target: trace::DIAG, tracing::Level::INFO);
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
                attn_qkv_f16_pre_quant: Some(Vec::new()),
                attn_proj_f16_pre_quant: Some(Vec::new()),
                ffn_h1_f16_pre_quant: Some(Vec::new()),
                ffn_h3_f16_pre_quant: Some(Vec::new()),
                ffn_h2_f16_pre_quant: Some(Vec::new()),
                proj_sa_data_head: Some(Vec::new()),
                proj_sa_scale_head: Some(Vec::new()),
                proj_wo_b_i8_head: Some(Vec::new()),
                proj_wo_b_scale_head: Some(Vec::new()),
                ..Block0Taps::default()
            };
            // Last-main-block per-op taps: same five pre-quant captures so
            // we can compare per-quant-event slope at block 0 vs block N-1.
            // If loss-per-event grows from block 0 to block N-1 then the
            // bug is data-dependent (heavy tails amplify f16 narrowing);
            // if loss-per-event is constant then it's uniform compounding.
            // Mirror block 0's full intra-op request so we get the same
            // per-op slope info at block 29. Required to tell apart (a)
            // per-op shrink that grows with block index (data-dependent)
            // vs (b) uniform per-op shrink (compounding bug).
            let mut block_last_taps = Block0Taps {
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
                attn_qkv_f16_pre_quant: Some(Vec::new()),
                attn_proj_f16_pre_quant: Some(Vec::new()),
                ffn_h1_f16_pre_quant: Some(Vec::new()),
                ffn_h3_f16_pre_quant: Some(Vec::new()),
                ffn_h2_f16_pre_quant: Some(Vec::new()),
                ..Block0Taps::default()
            };
            // Per-block full-residual sink. Engine fills index `b` with
            // the post-block-`b` residual stream (f32, seq_u*dim).
            let mut per_block_residual: Vec<Vec<f32>> = Vec::new();
            // Damage-zone per-op taps. Only fires at step 0 when the
            // localization sink is requested. Block indices are chosen
            // around the slope-shrink zone (blocks 24-28); block 26 also
            // gets QKV-site byte-head taps for the matmul_i8 audit.
            let damage_zone_indices: &[usize] = if want_step0 {
                &[24, 25, 26, 27, 28]
            } else {
                &[]
            };
            let mut damage_zone_taps: Vec<(usize, Block0Taps)> = damage_zone_indices
                .iter()
                .map(|&b| {
                    let mut taps = Block0Taps {
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
                        attn_qkv_f16_pre_quant: Some(Vec::new()),
                        ..Block0Taps::default()
                    };
                    if b == 26 {
                        // Block-26 only: byte-level matmul_i8 audit at the
                        // QKV site. e2e_parity CPU-recomputes one output
                        // element from these bytes.
                        taps.qkv_attn_in_data_head = Some(Vec::new());
                        taps.qkv_attn_in_params_head = Some(Vec::new());
                        taps.qkv_b_i8_head = Some(Vec::new());
                        taps.qkv_b_scale_head = Some(Vec::new());
                        taps.qkv_b_qsum_head = Some(Vec::new());
                        taps.qkv_dbg_trace_head = Some(Vec::new());
                    }
                    (b, taps)
                })
                .collect();
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
                        block_last: Some(&mut block_last_taps),
                        extra_blocks: if want_step0 && !damage_zone_taps.is_empty() {
                            Some(damage_zone_taps.as_mut_slice())
                        } else {
                            None
                        },
                        per_main_block_residual: if want_step0 {
                            Some(&mut per_block_residual)
                        } else {
                            None
                        },
                        ..DitTaps::default()
                    };
                    dit.forward_with_taps(
                        &self.backend,
                        &self.block_pipelines,
                        &self.dit_encoder_block_pipelines,
                        &self.dit_embedder_block_pipelines,
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
                        &self.dit_embedder_block_pipelines,
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
                // Per-quant-event loss probe on pre-act_quant f16 matmul
                // outputs. For each event, print stats AND run a host
                // simulation of the act_quant -> dequant round-trip using
                // both f16 params (current GPU path) and f32 params
                // (hypothesis: param narrowing is the slope source). The
                // slope-per-event lets us check whether 30-block
                // compounding explains the ~0.938 latent slope.
                let roundtrip = |label: &str, v: &[f32]| {
                    if v.is_empty() || (seq_u as usize) == 0 {
                        return;
                    }
                    let inner = v.len() / (seq_u as usize);
                    if inner == 0 || !inner.is_multiple_of(32) {
                        return;
                    }
                    crate::common::seq::diag_quant_roundtrip_loss(label, v, seq_u as usize, inner);
                };
                if let Some(v) = &b0.attn_qkv_f16_pre_quant {
                    print("block0.attn_qkv_f16_pre_quant", v);
                    roundtrip("DIAG_QUANT_LOSS block0.qkv_pre_quant", v);
                }
                if let Some(v) = &b0.attn_proj_f16_pre_quant {
                    print("block0.attn_proj_f16_pre_quant", v);
                    roundtrip("DIAG_QUANT_LOSS block0.proj_pre_quant", v);
                }
                if let Some(v) = &b0.ffn_h1_f16_pre_quant {
                    print("block0.ffn_h1_f16_pre_quant", v);
                    roundtrip("DIAG_QUANT_LOSS block0.ffn_h1_pre_quant", v);
                }
                if let Some(v) = &b0.ffn_h3_f16_pre_quant {
                    print("block0.ffn_h3_f16_pre_quant", v);
                    roundtrip("DIAG_QUANT_LOSS block0.ffn_h3_pre_quant", v);
                }
                if let Some(v) = &b0.ffn_h2_f16_pre_quant {
                    print("block0.ffn_h2_f16_pre_quant", v);
                    roundtrip("DIAG_QUANT_LOSS block0.ffn_h2_pre_quant", v);
                }
                // Same five events at the LAST main block. If per-event
                // slope_f16 is uniformly e.g. ~0.998 here, compounding
                // alone explains the observed final-latent slope. If
                // block_last is meaningfully worse than block0, the
                // bug is data-dependent (heavy-tail amplification).
                let bn = &block_last_taps;
                if let Some(v) = &bn.attn_qkv_f16_pre_quant {
                    print("block_last.attn_qkv_f16_pre_quant", v);
                    roundtrip("DIAG_QUANT_LOSS block_last.qkv_pre_quant", v);
                }
                if let Some(v) = &bn.attn_proj_f16_pre_quant {
                    print("block_last.attn_proj_f16_pre_quant", v);
                    roundtrip("DIAG_QUANT_LOSS block_last.proj_pre_quant", v);
                }
                if let Some(v) = &bn.ffn_h1_f16_pre_quant {
                    print("block_last.ffn_h1_f16_pre_quant", v);
                    roundtrip("DIAG_QUANT_LOSS block_last.ffn_h1_pre_quant", v);
                }
                if let Some(v) = &bn.ffn_h3_f16_pre_quant {
                    print("block_last.ffn_h3_f16_pre_quant", v);
                    roundtrip("DIAG_QUANT_LOSS block_last.ffn_h3_pre_quant", v);
                }
                if let Some(v) = &bn.ffn_h2_f16_pre_quant {
                    print("block_last.ffn_h2_f16_pre_quant", v);
                    roundtrip("DIAG_QUANT_LOSS block_last.ffn_h2_pre_quant", v);
                }
                // DIAG raw byte heads. Print as hex chunks + decoded views.
                fn print_i8_head(label: &str, bytes: &[u8]) {
                    let n = bytes.len().min(64);
                    let i8s: Vec<i8> = bytes[..n].iter().map(|&b| b as i8).collect();
                    let hex: String = bytes[..n.min(32)]
                        .iter()
                        .map(|b| format!("{:02x}", b))
                        .collect::<Vec<_>>()
                        .join("");
                    tracing::info!(target: "thinfer::diag", "[DIAG-RAW] {label}: n_bytes={} head_i8={:?} hex32={}", bytes.len(), i8s, hex);
                }
                fn print_f32_head(label: &str, bytes: &[u8]) {
                    let n_f32 = bytes.len() / 4;
                    let take = n_f32.min(16);
                    let mut vals = Vec::with_capacity(take);
                    for i in 0..take {
                        let off = i * 4;
                        let arr = [bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]];
                        vals.push(f32::from_le_bytes(arr));
                    }
                    tracing::info!(target: "thinfer::diag", "[DIAG-RAW] {label}: n_bytes={} head_f32={:?}", bytes.len(), vals);
                }
                if let Some(v) = &b0.proj_sa_data_head {
                    print_i8_head("block0.proj_sa_data_head", v);
                }
                if let Some(v) = &b0.proj_sa_scale_head {
                    print_f32_head("block0.proj_sa_scale_head", v);
                }
                if let Some(v) = &b0.proj_wo_b_i8_head {
                    print_i8_head("block0.proj_wo_b_i8_head", v);
                }
                if let Some(v) = &b0.proj_wo_b_scale_head {
                    print_f32_head("block0.proj_wo_b_scale_head", v);
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
            if i == 0
                && let Some(sink) = step0_taps.as_deref_mut()
            {
                sink.last_main = std::mem::take(&mut tap_last_main);
                sink.final_layer = std::mem::take(&mut tap_final);
                sink.model_output_post_neg = out.image.clone();
                sink.dt_step0 = scheduler.sigmas()[i + 1] - scheduler.sigmas()[i];
                sink.per_block_residual = std::mem::take(&mut per_block_residual);
                copy_block_taps(&block0_taps, &mut sink.block0);
                copy_block_taps(&block_last_taps, &mut sink.block_last);
                sink.damage_zone.clear();
                for (b, t) in &damage_zone_taps {
                    let mut local = Block0LocalTaps::default();
                    copy_block_taps(t, &mut local);
                    sink.damage_zone.push((*b, local));
                }
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
            let step_ms = t_step.map_or(0, |t| t.elapsed().as_millis() as u64);
            // Per-step weight streaming volume + effective read throughput.
            // When streaming overlaps GPU compute fully, step time tracks
            // compute; when it doesn't, this throughput sags and step time
            // tracks IO instead (the GPU-idle signal we're chasing).
            let mem = self.backend.mem_account();
            let streamed = mem.source_bytes_total().saturating_sub(step_src0);
            let mbps = if step_ms > 0 {
                (streamed as f64) / 1.0e6 / (step_ms as f64 / 1000.0)
            } else {
                0.0
            };
            tracing::info!(
                elapsed_ms = step_ms,
                step = i + 1,
                steps = params.steps,
                streamed_mib = streamed / (1024 * 1024),
                read_mbps = mbps as u64,
                weights_mib = mem.vram_current(VramCategory::Weights) / (1024 * 1024),
                vram_peak_mib = mem.vram_total_peak() / (1024 * 1024),
                "step done"
            );
        }
        // Denoise-phase snapshot: budget vs realized peak + total streamed,
        // so a run shows at a glance whether weights were re-streamed (total
        // streamed past model size = paging under a tight budget).
        {
            let mem = self.backend.mem_account();
            tracing::info!(
                budget_mib = self.residency.budget().vram_bytes / (1024 * 1024),
                vram_peak_mib = mem.vram_total_peak() / (1024 * 1024),
                weights_peak_mib = mem.vram_peak(VramCategory::Weights) / (1024 * 1024),
                workspace_peak_mib = mem.vram_peak(VramCategory::Workspace) / (1024 * 1024),
                staging_peak_mib = mem.vram_peak(VramCategory::Staging) / (1024 * 1024),
                "denoise done"
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

/// Copy populated `Block0Taps` fields (engine-side, `Option<Vec<f32>>`)
/// into the test-side `Block0LocalTaps` (plain `Vec<f32>`). Used to
/// hand block-0 / block-29 per-op intermediates back to the test for
/// per-op linfit comparison.
fn copy_block_taps(src: &Block0Taps, dst: &mut Block0LocalTaps) {
    let take = |o: &Option<Vec<f32>>| o.clone().unwrap_or_default();
    dst.adaln_input = take(&src.adaln_input);
    dst.adaln_pre = take(&src.adaln_pre);
    dst.adaln_full = take(&src.adaln_full);
    dst.scale_msa = take(&src.scale_msa);
    dst.gate_msa = take(&src.gate_msa);
    dst.scale_mlp = take(&src.scale_mlp);
    dst.gate_mlp = take(&src.gate_mlp);
    dst.attn_norm1_out = take(&src.attn_norm1_out);
    dst.modulated_attn_in = take(&src.modulated_attn_in);
    dst.attn_q = take(&src.attn_q);
    dst.attn_k = take(&src.attn_k);
    dst.attn_v = take(&src.attn_v);
    dst.attn_q_norm = take(&src.attn_q_norm);
    dst.attn_k_norm = take(&src.attn_k_norm);
    dst.attn_q_rope = take(&src.attn_q_rope);
    dst.attn_k_rope = take(&src.attn_k_rope);
    dst.attn_sdpa = take(&src.attn_sdpa);
    dst.attn_out = take(&src.attn_out);
    dst.attn_norm2_out = take(&src.attn_norm2_out);
    dst.x_mid = take(&src.x_mid);
    dst.ffn_norm1_out = take(&src.ffn_norm1_out);
    dst.modulated_ffn_in = take(&src.modulated_ffn_in);
    dst.ffn_raw = take(&src.ffn_raw);
    dst.ffn_norm2_out = take(&src.ffn_norm2_out);
    dst.attn_qkv_f16_pre_quant = take(&src.attn_qkv_f16_pre_quant);
    let take_bytes = |o: &Option<Vec<u8>>| o.clone().unwrap_or_default();
    dst.qkv_attn_in_data_head = take_bytes(&src.qkv_attn_in_data_head);
    dst.qkv_attn_in_params_head = take_bytes(&src.qkv_attn_in_params_head);
    dst.qkv_b_i8_head = take_bytes(&src.qkv_b_i8_head);
    dst.qkv_b_scale_head = take_bytes(&src.qkv_b_scale_head);
    dst.qkv_b_qsum_head = take_bytes(&src.qkv_b_qsum_head);
    dst.qkv_dbg_trace_head = take_bytes(&src.qkv_dbg_trace_head);
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
