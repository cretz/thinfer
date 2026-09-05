//! MiniMax-H3 audio VAE, DECODER only.
//!
//! This is a BigVGAN v2 generator and we already ship one: it is the same
//! family as the LTX-2.3 vocoder, with IDENTICAL `resblock_kernel_sizes
//! [3, 7, 11]` and `resblock_dilation_sizes [1, 3, 5]`. So the whole conv tower
//! comes from [`crate::common::bigvgan`] and what lives here is the H3
//! hyperparameters, its tensor names, and the stereo/denorm glue.
//!
//! Two H3-specific structural facts:
//!
//! - A `dec_in_proj` k=1 convolution lifts the 32 latent channels to 2048
//!   BEFORE `conv_pre`, which the LTX vocoder has no equivalent of (it feeds
//!   `conv_pre` a mel directly).
//! - **The model is MONO.** The two stereo channels are decoded as two
//!   independent runs, which is also how the reference does it.
//!
//! The upsample rates `5,5,2,2,2,2,2` multiply to 800, which is the encoder hop
//! at 40 latents/s and 32 kHz. That product is load-bearing: mistagging the
//! output sample rate is exactly the bug that stretched LTX rapid playback 1.5x.
//!
//! Weight-norm is already folded in the Comfy-Org conversion (plain `weight`
//! tensors, no `weight_g` / `weight_v` pair), so nothing is refolded at load.

use thinfer_core::backend::{WgpuBackend, WgpuError};
use thinfer_core::residency::WeightResidency;
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::Workspace;

use crate::common::bigvgan::{BigVgan, BigVganCfg, BigVganError, BigVganPipelines, InProj};

/// Audio VAE config, from the upstream `AudioVAEConfig`. Verified against the
/// shipped fp32 safetensors (2026-08-07).
#[derive(Clone, Debug, PartialEq)]
pub struct AudioVaeConfig {
    pub latent_channels: usize,
    /// Width `dec_in_proj` lifts the latents to, and `conv_pre` consumes.
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub decoder_kernel_sizes: Vec<usize>,
    pub sample_rate: u32,
    /// Stereo channels the pipeline carries; the MODEL itself is mono.
    pub channels: usize,
}

impl AudioVaeConfig {
    pub fn h3() -> Self {
        Self {
            latent_channels: 32,
            latent_dim: 2048,
            decoder_dim: 1024,
            decoder_rates: vec![5, 5, 2, 2, 2, 2, 2],
            decoder_kernel_sizes: vec![9, 9, 4, 4, 4, 4, 4],
            sample_rate: 32_000,
            channels: 2,
        }
    }

    /// Waveform samples one latent frame expands into.
    pub fn hop_length(&self) -> usize {
        self.decoder_rates.iter().product()
    }

    /// Latent frames per second of audio.
    pub fn latents_per_second(&self) -> usize {
        self.sample_rate as usize / self.hop_length()
    }

    fn bigvgan(&self) -> BigVganCfg {
        BigVganCfg {
            prefix: "decoder".into(),
            in_channels: self.latent_channels,
            out_channels: 1,
            init_ch: self.decoder_dim,
            up_rates: self.decoder_rates.clone(),
            up_kernels: self.decoder_kernel_sizes.clone(),
            ups_nested: true,
            acts_interleaved: true,
            act_post: "activation_post".into(),
            in_proj: Some(InProj {
                name: "dec_in_proj".into(),
                out_channels: self.latent_dim,
            }),
            up_filter: "decoder.resblocks.0.activations.0.upsample.filter".into(),
            down_filter: "decoder.resblocks.0.activations.0.downsample.lowpass.filter".into(),
        }
    }
}

pub type AudioVaeError<SE> = BigVganError<SE>;
pub type AudioVaePipelines = BigVganPipelines;

pub struct AudioVae {
    cfg: AudioVaeConfig,
    generator: BigVgan,
    /// Per-channel latent normalization, read from the checkpoint.
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
}

impl AudioVae {
    pub async fn new<S: WeightSource>(
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        cfg: AudioVaeConfig,
    ) -> Result<Self, AudioVaeError<S::Error>> {
        let generator = BigVgan::new(backend, residency, cfg.bigvgan()).await?;
        let latents_mean = crate::common::bigvgan::read_weight_f32(
            backend,
            residency,
            "latents_mean",
            cfg.latent_channels,
        )
        .await?;
        let latents_std = crate::common::bigvgan::read_weight_f32(
            backend,
            residency,
            "latents_std",
            cfg.latent_channels,
        )
        .await?;
        Ok(Self {
            cfg,
            generator,
            latents_mean,
            latents_std,
        })
    }

    pub fn cfg(&self) -> &AudioVaeConfig {
        &self.cfg
    }

    pub async fn compile(backend: &WgpuBackend) -> Result<AudioVaePipelines, WgpuError> {
        BigVganPipelines::compile(backend, "minimax_audio_vae").await
    }

    /// Decode `[channels, latent_channels, frames]` latents (host f32, NOT yet
    /// denormalized) into `[channels, frames * hop_length]` waveform samples.
    ///
    /// The model is mono, so each stereo channel is a separate run.
    pub async fn decode<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        pipelines: &AudioVaePipelines,
        latents: &[f32],
        frames: usize,
    ) -> Result<Vec<f32>, AudioVaeError<S::Error>> {
        let c = self.cfg.latent_channels;
        let ch = self.cfg.channels;
        assert_eq!(latents.len(), ch * c * frames, "audio latent size");

        let mut out = Vec::with_capacity(ch * frames * self.cfg.hop_length());
        for s in 0..ch {
            // `z * std + mean`, per latent channel.
            let mut x = vec![0.0f32; c * frames];
            for k in 0..c {
                for t in 0..frames {
                    let v = latents[(s * c + k) * frames + t];
                    x[k * frames + t] = v * self.latents_std[k] + self.latents_mean[k];
                }
            }
            let wav = self
                .generator
                .run(backend, residency, workspace, pipelines, &x, frames)
                .await?;
            // The reference clamps rather than applying a final tanh.
            out.extend(wav.into_iter().map(|v| v.clamp(-1.0, 1.0)));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 40 latents/s at 32 kHz means each latent frame must expand into exactly
    /// 800 samples. Getting this wrong mistags the sample rate and stretches
    /// playback, which has already cost us once on LTX rapid.
    #[test]
    fn hop_length_ties_the_latent_rate_to_the_sample_rate() {
        let c = AudioVaeConfig::h3();
        assert_eq!(c.hop_length(), 800);
        assert_eq!(c.latents_per_second(), 40);
        assert_eq!(
            c.latents_per_second(),
            crate::minimax::packing::AUDIO_LATENTS_PER_SECOND
        );
        assert_eq!(c.sample_rate, crate::minimax::packing::AUDIO_SAMPLE_RATE);
    }

    /// The DiT emits 32 audio latent channels; the VAE must read exactly that.
    #[test]
    fn latent_width_matches_the_dit() {
        assert_eq!(
            AudioVaeConfig::h3().latent_channels,
            crate::minimax::DitConfig::h3_33b().audio_latents_dim
        );
    }

    /// Stereo is carried as two mono runs, so the tower itself emits one
    /// channel; a 2-channel tower would silently halve the output length.
    #[test]
    fn the_tower_is_mono() {
        let c = AudioVaeConfig::h3();
        assert_eq!(c.bigvgan().out_channels, 1);
        assert_eq!(c.channels, 2);
        assert_eq!(c.bigvgan().in_channels, c.latent_channels);
        assert_eq!(
            c.bigvgan().in_proj.expect("dec_in_proj").out_channels,
            c.latent_dim
        );
    }
}
