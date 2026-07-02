//! HunyuanVideo 1.5 TAEHV (`taehv1_5`, madebyollin) tiny decoder config -- the
//! `VaeChoice::Tiny` fast path (decodes in seconds vs the full conv3d VAE's
//! minutes; draft quality). The decoder graph + forward are the shared
//! [`crate::common::vae_taehv`]; Hunyuan differs only in:
//!   - `z_dim = 32` (vs Wan's 48),
//!   - RAW latents: NO scale/shift. Confirmed against madebyollin `taehv.py` --
//!     the decoder's only pre-op is the `tanh(x/3)*3` clamp; it does NOT divide
//!     by the full VAE's `SCALING_FACTOR` (1.03682). The shared decoder maps the
//!     `[0,1]` output to `[-1,1]`, matching the full VAE's range so the driver
//!     stays VAE-choice agnostic.
//!
//! TUNING KNOB (serve eyeball): if the tiny clip reads scale-shifted vs the full
//! VAE, set `latents_std = 1.0 / vae::SCALING_FACTOR` (i.e. feed `z / SCALING`).
//! Default is identity per `taehv.py`.
//!
//! Weights load from `madebyollin/taehv` `taehv1_5.pth` (a flat fp16 state_dict;
//! keys `decoder.{1..22}...` match [`crate::common::vae_taehv::TaehvDecoderWeights`]
//! verbatim, so it reads via `PytorchSource` with no rename, narrowing f16->bf16
//! on upload).

use crate::common::vae_taehv::TaehvConfig;
use crate::hunyuan::config::vae;

/// `taehv1_5` decode config: z32, 16x spatial (8x conv stack + patch-2 shuffle),
/// identity latent norm (raw latents).
pub fn taehv_config() -> TaehvConfig {
    TaehvConfig {
        z_dim: vae::LATENT_CHANNELS,
        // image_channels(3) * patch_size^2 == conv_out channels (12) -> patch 2.
        patch_size: 2,
        spatial_compression: vae::FFACTOR_SPATIAL,
        latents_mean: vec![0.0; vae::LATENT_CHANNELS],
        latents_std: vec![1.0; vae::LATENT_CHANNELS],
    }
}
