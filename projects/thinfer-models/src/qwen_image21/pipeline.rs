//! Qwen-Image-2.1-Turbo generate: Qwen3-VL-8B conditioner (vision features
//! for condition images come in precomputed) -> VAE-encode the condition
//! images -> DiT prefix pass ONCE -> 8 FlowMatchEuler steps over the target
//! rows only -> VAE decode to RGBA. One residency unions the DiT GGUF, the
//! renamed encoder GGUF and the VAE; `evict_all_and_free` between phases keeps
//! each phase's peak inside the budget.

use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::residency::WeightResidency;
use thinfer_core::trace;
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::Workspace;

use crate::minimax::text_encoder::{
    EncoderHandles, EncoderPipelines, ImageRows, LmDims, MiniMaxTextEncoder, encoder_cfgs,
    encoder_cfgs_long, register_handles as register_encoder,
};
use crate::minimax::vision::VisionOutput;
use crate::wan::vae::{
    VaeDecoderWeights, VaeEncoderWeights, WanVaeConfig, WanVaeDecoder, WanVaeEncoder,
    WanVaePipelines, register_decoder, register_encoder as register_vae_encoder,
};

use super::config;
use super::dit::{DitError, KvPlacement, PrefixPart, Qwen21Dit, Qwen21DitPipelines};
use super::loader::{Qwen21DitHandles, register_handles as register_dit};
use super::scheduler::turbo_steps;
use super::vae::{PIXEL_CHANNELS, qwen_image21_vae};

#[derive(Clone, Copy, Debug)]
pub enum ProgressEvent {
    TextEncode,
    ImageEncode,
    Step { i: u32, n: u32 },
    VaeDecode,
}
pub type ProgressFn<'a> = Option<&'a dyn Fn(ProgressEvent)>;

/// A condition image, prepared by the caller: RGBA `[4, height, width]` in
/// `[-1, 1]` (planar) for the VAE, its vision-tower features, and where its
/// `<|image_pad|>` block starts in the conditioner's token sequence.
pub struct ConditionImage {
    pub rgba: Vec<f32>,
    pub height: usize,
    pub width: usize,
    pub vision: VisionOutput,
    pub pad_start: usize,
}

/// One request's conditioning input.
pub struct Request<'a> {
    /// The full chat-template token sequence (image pads expanded).
    pub token_ids: &'a [u32],
    /// Leading system-prompt rows the DiT does not see.
    pub drop: usize,
    pub images: &'a [ConditionImage],
    pub height: u32,
    pub width: u32,
    pub seed: u64,
}

#[derive(Debug)]
pub enum LoadError {
    Encoder(crate::z_image::text_encoder::LoadError),
    Weights(crate::common::loader::LoadError),
    Wgpu(thinfer_core::backend::WgpuError),
}
impl From<crate::z_image::text_encoder::LoadError> for LoadError {
    fn from(e: crate::z_image::text_encoder::LoadError) -> Self {
        Self::Encoder(e)
    }
}
impl From<crate::common::loader::LoadError> for LoadError {
    fn from(e: crate::common::loader::LoadError) -> Self {
        Self::Weights(e)
    }
}
impl From<thinfer_core::backend::WgpuError> for LoadError {
    fn from(e: thinfer_core::backend::WgpuError) -> Self {
        Self::Wgpu(e)
    }
}

#[derive(Debug)]
pub enum GenerateError<SE: core::fmt::Debug> {
    InvalidDims { height: u32, width: u32 },
    Layout(String),
    Encode(crate::minimax::text_encoder::ForwardError<SE>),
    VaeEncode(crate::wan::vae::WanVaeEncodeError<SE>),
    Dit(DitError<SE>),
    VaeDecode(crate::wan::vae::WanVaeDecodeError<SE>),
    Wgpu(thinfer_core::backend::WgpuError),
}

pub struct Qwen21Pipeline<S: WeightSource> {
    backend: Arc<WgpuBackend>,
    residency: WeightResidency<S>,
    encoder_handles: EncoderHandles,
    dit: Qwen21Dit,
    dit_handles: Qwen21DitHandles,
    dit_pipelines: Qwen21DitPipelines,
    vae_cfg: WanVaeConfig,
    vae: WanVaeDecoder,
    vae_encoder: WanVaeEncoder,
}

/// Deterministic standard-normal noise (SplitMix64 -> Box-Muller), the
/// engine's image-model noise source.
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

/// `[C, h, w]` planar -> `[h*w, C]` tokens (the reference's spatial flatten).
pub fn pack(planar: &[f32], c: usize, hw: usize) -> Vec<f32> {
    let mut out = vec![0.0; c * hw];
    for ch in 0..c {
        for p in 0..hw {
            out[p * c + ch] = planar[ch * hw + p];
        }
    }
    out
}

/// Inverse of [`pack`].
pub fn unpack(tokens: &[f32], c: usize, hw: usize) -> Vec<f32> {
    let mut out = vec![0.0; c * hw];
    for p in 0..hw {
        for ch in 0..c {
            out[ch * hw + p] = tokens[p * c + ch];
        }
    }
    out
}

/// The prefix runs after the dropped system rows: text between image blocks,
/// each image block replaced by its condition latents.
pub fn prefix_parts(
    seq: usize,
    drop: usize,
    images: &[ConditionImage],
) -> Result<Vec<PrefixPart>, String> {
    let s = config::PIXELS_PER_TOKEN;
    let mut parts = Vec::new();
    let (mut row, mut token) = (drop, 0usize);
    for img in images {
        if img.pad_start < row || img.pad_start + img.vision.tokens > seq {
            return Err(format!(
                "image block at {} (+{}) is outside the conditioned rows {row}..{seq}",
                img.pad_start, img.vision.tokens
            ));
        }
        let (h, w) = (img.height / s, img.width / s);
        if (h / 2) * (w / 2) != img.vision.tokens {
            return Err(format!(
                "image {}x{}: {} vision tokens, expected {}",
                img.width,
                img.height,
                img.vision.tokens,
                (h / 2) * (w / 2)
            ));
        }
        if img.pad_start > row {
            parts.push(PrefixPart::Text {
                start: row - drop,
                len: img.pad_start - row,
            });
        }
        parts.push(PrefixPart::Image { start: token, h, w });
        token += h * w;
        row = img.pad_start + img.vision.tokens;
    }
    if seq > row {
        parts.push(PrefixPart::Text {
            start: row - drop,
            len: seq - row,
        });
    }
    Ok(parts)
}

impl<S: WeightSource> Qwen21Pipeline<S> {
    pub async fn load(
        backend: Arc<WgpuBackend>,
        residency: WeightResidency<S>,
    ) -> Result<Self, LoadError> {
        let encoder_handles = register_encoder(&residency, None, LmDims::QWEN3_VL_8B)?;
        let dit_handles = register_dit(&residency, config::N_LAYERS)?;
        let dit_pipelines = Qwen21DitPipelines::compile(&backend).await?;
        let vae_cfg = qwen_image21_vae();
        let before = residency.total_registered_bytes();
        let dec_handles = register_decoder(&residency, &VaeDecoderWeights::new(&vae_cfg))?;
        let dec_footprint = residency.total_registered_bytes() - before;
        let before = residency.total_registered_bytes();
        let enc_handles = register_vae_encoder(&residency, &VaeEncoderWeights::new(&vae_cfg))?;
        let enc_footprint = residency.total_registered_bytes() - before;
        let vae_pipelines = WanVaePipelines::compile(&backend).await?;
        let enc_pipelines = WanVaePipelines::compile(&backend).await?;
        Ok(Self {
            backend,
            residency,
            encoder_handles,
            dit: Qwen21Dit::new(),
            dit_handles,
            dit_pipelines,
            vae: WanVaeDecoder {
                pipelines: vae_pipelines,
                handles: dec_handles,
                cfg: vae_cfg.clone(),
                weight_footprint: dec_footprint,
            },
            vae_encoder: WanVaeEncoder {
                pipelines: enc_pipelines,
                handles: enc_handles,
                cfg: vae_cfg.clone(),
                weight_footprint: enc_footprint,
            },
            vae_cfg,
        })
    }

    /// The conditioner's last-layer rows (system rows dropped), `[n, 4096]`.
    async fn encode(
        &self,
        req: &Request<'_>,
        workspace: &Workspace<WgpuBackend>,
    ) -> Result<Vec<f32>, GenerateError<S::Error>> {
        let images: Vec<ImageRows<'_>> = req
            .images
            .iter()
            .map(|i| ImageRows {
                start: i.pad_start,
                vision: &i.vision,
            })
            .collect();
        // Condition images make the prompt thousands of tokens long: bf16 acts
        // (tensor-core matmuls and attention). Text alone keeps f32.
        let cfgs = if images.is_empty() {
            encoder_cfgs()
        } else {
            encoder_cfgs_long()
        };
        let pipelines = EncoderPipelines::compile(&self.backend, &cfgs)
            .await
            .map_err(GenerateError::Wgpu)?;
        let out = MiniMaxTextEncoder
            .forward(
                &self.backend,
                &pipelines,
                &self.residency,
                workspace,
                &self.encoder_handles,
                self.residency.source(),
                req.token_ids,
                &images,
            )
            .await
            .map_err(GenerateError::Encode)?;
        Ok(out.hidden[req.drop * config::TEXT_DIM..].to_vec())
    }

    /// The condition images' normalized latent tokens, `[sum h*w, 64]`.
    async fn encode_images(
        &self,
        images: &[ConditionImage],
        workspace: &mut Workspace<WgpuBackend>,
    ) -> Result<Vec<f32>, GenerateError<S::Error>> {
        let z = self.vae_cfg.z_dim;
        let s = config::PIXELS_PER_TOKEN;
        let mut tokens = Vec::new();
        for img in images {
            let moments = self
                .vae_encoder
                .encode(
                    &self.backend,
                    &self.residency,
                    workspace,
                    &img.rgba,
                    1,
                    img.height,
                    img.width,
                )
                .await
                .map_err(GenerateError::VaeEncode)?;
            let hw = (img.height / s) * (img.width / s);
            // The posterior mode (mean channels), then `(z - mean) / std`.
            let mean: Vec<f32> = (0..z * hw)
                .map(|i| {
                    let c = i / hw;
                    (moments[i] - self.vae_cfg.latents_mean[c]) / self.vae_cfg.latents_std[c]
                })
                .collect();
            tokens.extend(pack(&mean, z, hw));
        }
        Ok(tokens)
    }

    /// Generate one image: RGBA `[4, height, width]` planar in `[-1, 1]`.
    pub async fn generate(
        &self,
        req: &Request<'_>,
        progress: ProgressFn<'_>,
    ) -> Result<Vec<f32>, GenerateError<S::Error>> {
        let (height, width) = (req.height, req.width);
        let m = config::SIDE_MULTIPLE;
        if height == 0 || width == 0 || !height.is_multiple_of(m) || !width.is_multiple_of(m) {
            return Err(GenerateError::InvalidDims { height, width });
        }
        let parts = prefix_parts(req.token_ids.len(), req.drop, req.images)
            .map_err(GenerateError::Layout)?;
        let s = config::PIXELS_PER_TOKEN;
        let (lat_h, lat_w) = (height as usize / s, width as usize / s);
        let hw = lat_h * lat_w;
        let mut workspace = Workspace::new(
            Arc::clone(&self.backend),
            Arc::clone(self.residency.arbiter()),
        );
        let notify = |e| {
            if let Some(p) = progress {
                p(e);
            }
        };

        notify(ProgressEvent::TextEncode);
        let encoder = {
            let _s = trace::scope!("qwen21.encode", tokens = req.token_ids.len()).entered();
            self.encode(req, &workspace).await?
        };
        self.residency.evict_all_and_free(&*self.backend);
        workspace.drain_pool();

        let cond = if req.images.is_empty() {
            Vec::new()
        } else {
            notify(ProgressEvent::ImageEncode);
            let _s = trace::scope!("qwen21.image_encode", images = req.images.len()).entered();
            let c = self.encode_images(req.images, &mut workspace).await?;
            self.residency.evict_all_and_free(&*self.backend);
            workspace.drain_pool();
            c
        };

        let lc = config::LATENT_CH;
        let mut tokens = pack(&gaussian_noise(lc * hw, req.seed), lc, hw);
        {
            let kv = KvPlacement::for_budget(
                &self.dit_pipelines,
                parts.iter().map(|p| p.rows()).sum(),
                self.dit_handles.blocks.len(),
                self.residency.budget().vram_bytes,
            );
            let prefix = {
                let _s = trace::scope!("qwen21.prefill").entered();
                self.dit
                    .prefill(
                        &self.backend,
                        &self.dit_pipelines,
                        &self.residency,
                        &workspace,
                        &self.dit_handles,
                        &encoder,
                        &cond,
                        &parts,
                        (lat_h, lat_w),
                        kv,
                    )
                    .await
                    .map_err(GenerateError::Dit)?
            };
            tracing::info!(
                target: trace::DIAG,
                prefix_rows = prefix.rows(),
                target_rows = hw,
                spilled = prefix.spilled(),
                "qwen-image-2.1 prefix cached",
            );
            let steps = turbo_steps();
            let n = steps.len() as u32;
            let _s = trace::scope!("qwen21.denoise", steps = n).entered();
            for (i, step) in steps.iter().enumerate() {
                notify(ProgressEvent::Step { i: i as u32 + 1, n });
                let v = self
                    .dit
                    .step(
                        &self.backend,
                        &self.dit_pipelines,
                        &self.residency,
                        &workspace,
                        &self.dit_handles,
                        &prefix,
                        &tokens,
                        step.t,
                    )
                    .await
                    .map_err(GenerateError::Dit)?;
                for (z, dv) in tokens.iter_mut().zip(&v) {
                    *z += dv * step.delta;
                }
            }
        }
        self.residency.evict_all_and_free(&*self.backend);
        workspace.drain_pool();

        notify(ProgressEvent::VaeDecode);
        let latent = unpack(&tokens, lc, hw);
        let rgba = self
            .vae
            .decode(
                &self.backend,
                &self.residency,
                &mut workspace,
                &latent,
                1,
                lat_h,
                lat_w,
            )
            .await
            .map_err(GenerateError::VaeDecode)?;
        self.residency.evict_all_and_free(&*self.backend);
        debug_assert_eq!(
            rgba.len(),
            PIXEL_CHANNELS * height as usize * width as usize
        );
        Ok(rgba)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(pad_start: usize, h: usize, w: usize) -> ConditionImage {
        ConditionImage {
            rgba: Vec::new(),
            height: h * 16,
            width: w * 16,
            vision: VisionOutput {
                embeds: Vec::new(),
                deepstack: [Vec::new(), Vec::new(), Vec::new()],
                tokens: (h / 2) * (w / 2),
                merged_grid: (h / 2, w / 2),
            },
            pad_start,
        }
    }

    #[test]
    fn parts_interleave_text_and_condition_tokens() {
        // 10 system rows, 3 text rows, an 4x6 image (6 slots), 5 text rows.
        let img = image(13, 4, 6);
        let parts = prefix_parts(24, 10, std::slice::from_ref(&img)).unwrap();
        assert_eq!(
            parts,
            vec![
                PrefixPart::Text { start: 0, len: 3 },
                PrefixPart::Image {
                    start: 0,
                    h: 4,
                    w: 6
                },
                PrefixPart::Text { start: 9, len: 5 },
            ]
        );
    }

    #[test]
    fn pack_round_trips() {
        let x: Vec<f32> = (0..24).map(|v| v as f32).collect();
        assert_eq!(unpack(&pack(&x, 4, 6), 4, 6), x);
    }
}
