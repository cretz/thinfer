//! Krea 2 Turbo t2i pipeline: Qwen3-VL-4B encode (12 taps) -> `txtfusion`
//! (once) -> single-stream DiT FlowMatchEuler denoise (CFG-free, 8-step turbo)
//! -> Wan2.1 KL VAE decode -> RGB. Mirrors the qwen_image t2i phase structure
//! (`evict_all_and_free` between encode / denoise / decode so peak VRAM holds
//! the budget). Weights come from one residency over the DiT GGUF (krea2 keys) +
//! the renamed Qwen3-VL encoder GGUF + the Wan2.1 VAE.

use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::ops::{ActDtype, WeightDtype, WgslConfig};
use thinfer_core::quant::QuantKind;
use thinfer_core::residency::WeightResidency;
use thinfer_core::tensor::StorageEncoding;
use thinfer_core::trace;
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::Workspace;

use crate::common::block::{BlockPipelines, BlockWgslConfigs, CoopmatSites, DenseActSites};
use crate::krea::config;
use crate::krea::dit::{DitError, KreaDit, KreaDitPipelines, block_cfgs};
use crate::krea::loader::{KreaDitHandles, register_handles as register_dit_handles};
use crate::krea::packing::{pack_latents, unpack_latents};
use crate::krea::scheduler::{TURBO_MU, build_steps};
use crate::krea::text_encoder::KreaTextEncoder;
use crate::krea::vae::krea_vae;
use crate::wan::vae::{VaeDecoderWeights, WanVaeDecoder, WanVaePipelines, register_decoder};
use crate::z_image::pipeline::encode_png;
use crate::z_image::text_encoder::{Qwen3Handles, Qwen3Weights, register_qwen3_handles};

#[derive(Clone, Copy, Debug)]
pub enum ProgressEvent {
    TextEncode,
    Step { i: u32, n: u32 },
    VaeDecode,
}
pub type ProgressFn<'a> = Option<&'a dyn Fn(ProgressEvent)>;

/// Tokens dropped from the front of the encoder taps: the prompt template's
/// system preamble (`prompt_template_encode_start_idx`; Krea shares the
/// Qwen-Image t2i template, so 34). The system tokens still condition the causal
/// encode; only their DiT-side taps are dropped.
pub const DROP_IDX: usize = 34;

/// Diag: latent stats over packed tokens `[img_seq, ch]`. Returns
/// `(overall_std, spatial_std)` where `spatial_std` is the std ACROSS image-token
/// positions of each token's channel-mean. A near-zero `spatial_std` means every
/// spatial position is identical -> the DiT produced no spatial structure (the
/// flat-blob signature); a healthy latent has `spatial_std` comparable to
/// `overall_std`.
fn latent_stats(tokens: &[f32], img_seq: usize) -> (f32, f32) {
    if img_seq == 0 || tokens.is_empty() {
        return (0.0, 0.0);
    }
    let ch = tokens.len() / img_seq;
    let n = tokens.len() as f64;
    let mean = tokens.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = tokens
        .iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    let mut per_tok = vec![0.0f64; img_seq];
    for (t, pm) in per_tok.iter_mut().enumerate() {
        let s: f64 = tokens[t * ch..(t + 1) * ch].iter().map(|&v| v as f64).sum();
        *pm = s / ch as f64;
    }
    let gmean = per_tok.iter().sum::<f64>() / img_seq as f64;
    let svar = per_tok.iter().map(|&v| (v - gmean).powi(2)).sum::<f64>() / img_seq as f64;
    (var.sqrt() as f32, svar.sqrt() as f32)
}

/// Deterministic standard-normal noise (SplitMix64 -> Box-Muller).
fn gaussian_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next = || -> f64 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
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
    InvalidDims { height: u32, width: u32 },
    Encode(crate::z_image::text_encoder::Qwen3ForwardError<SE>),
    Dit(DitError<SE>),
    Vae(crate::wan::vae::WanVaeDecodeError<SE>),
    Png(String),
}

pub struct KreaPipeline<S: WeightSource> {
    backend: Arc<WgpuBackend>,
    residency: WeightResidency<S>,
    encoder: KreaTextEncoder,
    encoder_handles: Qwen3Handles,
    encoder_pipelines: BlockPipelines,
    dit: KreaDit,
    dit_handles: KreaDitHandles,
    dit_pipelines: KreaDitPipelines,
    vae: WanVaeDecoder,
}

/// bf16-act encoder config with per-site quant probed from the catalog (the
/// Qwen3-VL GGUF answers `Quant(k)` at every matmul site, mixed K-quants
/// included). Qwen3 massive activations mandate bf16 acts (f16 overflows).
fn encoder_cfgs<S: WeightSource>(residency: &WeightResidency<S>) -> BlockWgslConfigs {
    let ops = WgslConfig {
        bf16_quant_writes: false,
        act_dtype: ActDtype::Bf16,
        weight_dtype: WeightDtype::Bf16,
    };
    let probe = |id: &WeightId| -> WgslConfig {
        let wd = match residency
            .source()
            .catalog()
            .get(id)
            .and_then(|e| e.encoding)
        {
            Some(StorageEncoding::Quant(k)) => WeightDtype::Quant(k),
            _ => WeightDtype::Bf16,
        };
        WgslConfig {
            weight_dtype: wd,
            ..ops
        }
    };
    let l0 = Qwen3Weights::new();
    let b = &l0.layers[0];
    BlockWgslConfigs {
        matmul_qkv: probe(&b.q_proj),
        matmul_qkv_self: probe(&b.q_proj),
        matmul_proj: probe(&b.o_proj),
        matmul_ffn_up: probe(&b.mlp_gate),
        matmul_ffn_down: probe(&b.mlp_down),
        matmul_adaln: ops,
        ops,
        i8_sdpa: false,
        dense_acts: DenseActSites::default(),
        coopmat_acts: CoopmatSites::default(),
        large_d_sdpa: false,
        fast_sdpa: false,
        decode_sdpa: false,
    }
}

impl<S: WeightSource> KreaPipeline<S> {
    /// Build the pipeline over a residency unioning the DiT GGUF + renamed
    /// Qwen3-VL encoder GGUF + Wan2.1 VAE. `max_seq` sizes the encoder rope;
    /// `quant` is the DiT GGUF quant family (Q8_0 canary / Q4_K_M footprint).
    pub async fn load(
        backend: Arc<WgpuBackend>,
        residency: WeightResidency<S>,
        max_seq: usize,
        quant: QuantKind,
    ) -> Result<Self, LoadError> {
        // encoder: native GGUF quant (no transcode), bf16 acts.
        let encoder_handles = register_qwen3_handles(&residency, None)?;
        let enc_cfgs = encoder_cfgs(&residency);
        let encoder_pipelines = BlockPipelines::compile(&backend, &enc_cfgs).await?;

        // DiT: bf16 acts, `quant` block matmuls dequant-once.
        let dit_handles = register_dit_handles(&residency, config::N_LAYERS)?;
        let dit_cfgs = block_cfgs(quant);
        let dit_pipelines = KreaDitPipelines::compile(&backend, &dit_cfgs).await?;

        // VAE (Wan2.1 decoder).
        let vae_cfg = krea_vae();
        let before = residency.total_registered_bytes();
        let vae_handles = register_decoder(&residency, &VaeDecoderWeights::new(&vae_cfg))?;
        let vae_footprint = residency.total_registered_bytes() - before;
        let vae_pipelines = WanVaePipelines::compile(&backend).await?;

        Ok(Self {
            backend,
            residency,
            encoder: KreaTextEncoder::new(max_seq),
            encoder_handles,
            encoder_pipelines,
            dit: KreaDit::new(),
            dit_handles,
            dit_pipelines,
            vae: WanVaeDecoder {
                pipelines: vae_pipelines,
                handles: vae_handles,
                cfg: vae_cfg,
                weight_footprint: vae_footprint,
            },
        })
    }

    /// Encode -> denoise -> VAE decode. Returns RGB CHW f32 in `[-1, 1]`,
    /// `[3, height, width]`.
    pub async fn generate_rgb(
        &self,
        token_ids: &[u32],
        height: u32,
        width: u32,
        steps: u32,
        seed: u64,
        progress: ProgressFn<'_>,
    ) -> Result<Vec<f32>, GenerateError<S::Error>> {
        if token_ids.is_empty() {
            return Err(GenerateError::EmptyPrompt);
        }
        let ppt = config::PIXELS_PER_TOKEN as u32;
        if !height.is_multiple_of(ppt) || !width.is_multiple_of(ppt) || height == 0 || width == 0 {
            return Err(GenerateError::InvalidDims { height, width });
        }
        let gh = height as usize / config::PIXELS_PER_TOKEN;
        let gw = width as usize / config::PIXELS_PER_TOKEN;
        let lat_h = gh * 2;
        let lat_w = gw * 2;

        let mut workspace = Workspace::new(
            Arc::clone(&self.backend),
            Arc::clone(self.residency.arbiter()),
        );

        // --- 1. encode (12 taps) + txtfusion (once) -> txt_features [txt_tok, DIM]
        if let Some(p) = progress {
            p(ProgressEvent::TextEncode);
        }
        let txt_features = {
            let _s = trace::scope!("krea.encode", tokens = token_ids.len()).entered();
            let taps = self
                .encoder
                .forward(
                    &self.backend,
                    &self.encoder_pipelines,
                    &self.residency,
                    &workspace,
                    &self.encoder_handles,
                    self.residency.source(),
                    token_ids,
                )
                .await
                .map_err(GenerateError::Encode)?;
            // Drop the template preamble taps (system prefix) before txtfusion;
            // each token carries TEXT_LAYERS*TEXT_DIM tap values.
            let per_tok = config::TEXT_LAYERS * config::TEXT_DIM;
            let txt_tok = taps.len() / per_tok;
            let drop = DROP_IDX.min(txt_tok.saturating_sub(1));
            if tracing::enabled!(target: "thinfer::diag", tracing::Level::DEBUG) {
                let bad = taps.iter().filter(|v| !v.is_finite()).count();
                let (s, _) = latent_stats(&taps, txt_tok);
                tracing::event!(
                    target: "thinfer::diag", tracing::Level::DEBUG,
                    tokens = txt_tok, nonfinite = bad, std = s,
                    "krea encoder taps stats"
                );
            }
            let feats = self
                .dit
                .prepare_txt(
                    &self.backend,
                    &self.dit_pipelines,
                    &self.residency,
                    &workspace,
                    &self.dit_handles,
                    &taps[drop * per_tok..],
                )
                .await
                .map_err(GenerateError::Dit)?;
            if tracing::enabled!(target: "thinfer::diag", tracing::Level::DEBUG) {
                let bad = feats.iter().filter(|v| !v.is_finite()).count();
                let toks = feats.len() / config::DIM;
                let (s, _) = latent_stats(&feats, toks);
                tracing::event!(
                    target: "thinfer::diag", tracing::Level::DEBUG,
                    tokens = toks, nonfinite = bad, std = s,
                    "krea txt_features (post prepare_txt) stats"
                );
            }
            feats
        };
        self.residency.evict_all_and_free(&*self.backend);
        workspace.drain_pool();

        // --- 2. FlowMatchEuler denoise (CFG-free, turbo mu) ---
        let steps_plan = build_steps(steps as usize, TURBO_MU);
        let z = gaussian_noise(config::LATENT_CH * lat_h * lat_w, seed);
        let mut tokens = pack_latents(&z, lat_h, lat_w);
        {
            let _s = trace::scope!("krea.denoise", steps = steps).entered();
            for (i, step) in steps_plan.iter().enumerate() {
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
                        &txt_features,
                        step.t,
                        gh,
                        gw,
                    )
                    .await
                    .map_err(GenerateError::Dit)?;
                debug_assert_eq!(out.velocity.len(), tokens.len());
                if tracing::enabled!(target: "thinfer::diag", tracing::Level::DEBUG) {
                    let (vstd, vspat) = latent_stats(&out.velocity, out.img_seq);
                    for (zi, &v) in tokens.iter_mut().zip(out.velocity.iter()) {
                        *zi += v * step.delta;
                    }
                    let (zstd, zspat) = latent_stats(&tokens, out.img_seq);
                    tracing::event!(
                        target: "thinfer::diag",
                        tracing::Level::DEBUG,
                        step = i as u32 + 1,
                        t = step.t,
                        delta = step.delta,
                        vel_std = vstd,
                        vel_spatial_std = vspat,
                        lat_std = zstd,
                        lat_spatial_std = zspat,
                        "krea denoise step latent stats"
                    );
                } else {
                    for (zi, &v) in tokens.iter_mut().zip(out.velocity.iter()) {
                        *zi += v * step.delta;
                    }
                }
            }
        }
        self.residency.evict_all_and_free(&*self.backend);
        workspace.drain_pool();

        // --- 3. unpack -> VAE decode ---
        if let Some(p) = progress {
            p(ProgressEvent::VaeDecode);
        }
        if tracing::enabled!(target: "thinfer::diag", tracing::Level::DEBUG) {
            let (zstd, zspat) = latent_stats(&tokens, gh * gw);
            tracing::event!(
                target: "thinfer::diag",
                tracing::Level::DEBUG,
                lat_std = zstd,
                lat_spatial_std = zspat,
                "krea pre-VAE final latent stats"
            );
        }
        let z = unpack_latents(&tokens, lat_h, lat_w);
        // Diag: dump the pre-VAE latent [LATENT_CH, lat_h, lat_w] f32-le to
        // localize artifacts (weave/darkness) as DiT-side vs VAE-side.
        if let Ok(p) = std::env::var("THINFER_KREA_DUMP_LATENT") {
            let bytes: Vec<u8> = z.iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = std::fs::write(&p, &bytes);
        }
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
        Ok(rgb)
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
