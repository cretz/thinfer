//! Wan adapter over the shared TAEHV tiny decoder ([`crate::common::vae_taehv`]).
//! Wan selects `lighttaew2_2` (z48, per-channel latent denorm `z*std+mean`).
//! The decoder graph + forward live in `common`; this module only maps
//! [`WanVaeConfig`] to [`TaehvConfig`] and re-exports the type names the Wan
//! pipeline already imports, so the shared move is source-compatible.

pub use crate::common::vae_taehv::{
    TaehvConfig, TaehvDecodeError as WanVaeTinyDecodeError, TaehvDecoder as WanVaeTinyDecoder,
    TaehvDecoderWeights as TinyDecoderWeights, TaehvPipelines as WanVaeTinyPipelines,
    register_decoder_taehv as register_decoder_tiny,
};

use crate::wan::vae::WanVaeConfig;

/// Wan's tiny decode pre-scales the latent by the VAE's baked per-channel
/// `z*std+mean`; the rest of the geometry is the family default.
impl From<&WanVaeConfig> for TaehvConfig {
    fn from(c: &WanVaeConfig) -> Self {
        Self {
            z_dim: c.z_dim,
            patch_size: c.patch_size,
            spatial_compression: c.spatial_compression,
            latents_mean: c.latents_mean.clone(),
            latents_std: c.latents_std.clone(),
        }
    }
}
