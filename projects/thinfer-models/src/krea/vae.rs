//! Krea 2 uses the Wan2.1 KL VAE (z16, 8x spatial / 4x temporal), identical to
//! the one already ported in [`crate::wan::vae`]. This module is a THIN ADAPTER:
//! it returns the shared [`WanVaeConfig::wan2_1`] with its baked Wan2.1 latent
//! normalization, and (like the qwen_image adapter) converts VAE encoder
//! `moments` to a normalized latent for any future reference-image path.
//!
//! Source = `Wan-AI/Wan2.2-T2V-A14B-Diffusers` `vae/` safetensors (diffusers
//! `AutoencoderKLWan` tensor keys), consumed directly by the
//! [`crate::wan::vae`] loader with no rename shim. Image model => a single
//! temporal frame (no halo), same single-chunk path as qwen_image.

use crate::wan::vae::WanVaeConfig;

/// Krea 2 VAE config = the shared Wan2.1 VAE.
pub fn krea_vae() -> WanVaeConfig {
    WanVaeConfig::wan2_1()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_wan21_z16() {
        let c = krea_vae();
        assert_eq!(c.z_dim, 16);
        assert_eq!(c.latents_mean.len(), 16);
        assert_eq!(c.latents_std.len(), 16);
        assert_eq!(c.spatial_compression, 8);
        assert_eq!(c.pixel_channels(), 3);
    }
}
