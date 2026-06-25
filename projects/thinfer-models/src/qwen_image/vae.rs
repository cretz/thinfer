//! Qwen-Image KL VAE config. The diffusers `AutoencoderKLQwenImage`
//! (`third-party/diffusers/.../autoencoder_kl_qwenimage.py`) is built from the
//! EXACT Wan-family 3D causal-conv KL primitives already ported in
//! [`crate::wan::vae`] (`QwenImageCausalConv3d / ResidualBlock / Resample /
//! MidBlock / RMS_norm / AttentionBlock`), so this module is a THIN ADAPTER: a
//! [`WanVaeConfig`] constructor with Qwen's dims + latent norm, not a new VAE.
//!
//! Shape = Wan2.1 (8x spatial / 4x temporal, z16, non-residual, no pixel
//! shuffle). Image model => frame = 1 (a single temporal chunk, no halo, same
//! as the tiny-VAE single-chunk path).
//!
//! Source = the full `Qwen/Qwen-Image` `vae/` safetensors (encoder + decoder,
//! bf16). Its tensor keys are native diffusers
//! (`encoder.*/decoder.*/quant_conv/post_quant_conv`), which is exactly what the
//! [`crate::wan::vae`] loader builds, so the source is consumed directly with no
//! rename shim. We use this one source for both decode (t2i + edit output) and
//! encode (the edit path's reference-image latent channel). The values below are
//! verbatim from the safetensors `config.json` (confirmed equal to the diffusers
//! `AutoencoderKLQwenImage` defaults).

use crate::wan::vae::WanVaeConfig;

/// Turn the VAE encoder output `moments` (CTHW `[2*z_dim, 1, h, w]`, mean ++
/// logvar) for a reference image into the normalized latent `[z_dim, h, w]` the
/// edit path packs into image-conditioning tokens. Mirrors the edit pipeline:
/// take the distribution MODE (= mean, channels `0..z_dim`), then per-channel
/// `(z - latents_mean) / latents_std`. Returns row-major `[z_dim, h, w]` ready
/// for [`crate::qwen_image::packing::pack_latents`].
pub fn normalize_ref_latent(moments: &[f32], cfg: &WanVaeConfig, h: usize, w: usize) -> Vec<f32> {
    let z = cfg.z_dim;
    let plane = h * w;
    assert_eq!(
        moments.len(),
        2 * z * plane,
        "moments len != 2*z_dim*h*w (got {}, want {})",
        moments.len(),
        2 * z * plane
    );
    let mut out = vec![0.0_f32; z * plane];
    for c in 0..z {
        let mean = cfg.latents_mean[c];
        let std = cfg.latents_std[c];
        let src = &moments[c * plane..c * plane + plane]; // mode = mean channels
        let dst = &mut out[c * plane..c * plane + plane];
        for (o, &m) in dst.iter_mut().zip(src) {
            *o = (m - mean) / std;
        }
    }
    out
}

/// Qwen-Image KL VAE config (Wan2.1 shape). All values verbatim from the
/// `Qwen/Qwen-Image` `vae/config.json`.
pub fn qwen_image_vae() -> WanVaeConfig {
    WanVaeConfig {
        // Wan2.1 has a single base dim (no Wan2.2 decoder split).
        base_dim: 96,
        decoder_base_dim: 96,
        z_dim: 16,
        // RGB in/out; Wan2.1 does no boundary pixel (un)shuffle (patch_size 1),
        // so conv_in/out channels are the raw pixel channels.
        in_channels: 3,
        out_channels: 3,
        dim_mult: vec![1, 2, 4, 4],
        num_res_blocks: 2,
        temporal_downsample: vec![false, true, true],
        is_residual: false,
        patch_size: 1,
        spatial_compression: 8,
        temporal_compression: 4,
        // Wan `WanRMS_norm` / `F.normalize` default eps.
        norm_eps: 1e-12,
        latents_mean: vec![
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715,
            0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
        ],
        latents_std: vec![
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652,
            1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_matches_wan21_qwen_image_shape() {
        let c = qwen_image_vae();
        assert_eq!(c.z_dim, 16);
        assert_eq!(c.latents_mean.len(), c.z_dim);
        assert_eq!(c.latents_std.len(), c.z_dim);
        // Non-residual, single base dim, no pixel shuffle (Wan2.1).
        assert!(!c.is_residual);
        assert_eq!(c.base_dim, c.decoder_base_dim);
        assert_eq!(c.patch_size, 1);
        // 4 stages, 3 spatial downsamples => 8x; 2 temporal Trues => 4x.
        assert_eq!(c.spatial_compression, 8);
        assert_eq!(c.temporal_compression, 4);
        assert_eq!(c.temporal_downsample.len(), c.dim_mult.len() - 1);
        assert_eq!(c.pixel_channels(), 3);
    }
}
