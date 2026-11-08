//! Qwen-Image-2.1 VAE (`AutoencoderKLQwenImage21`): the Wan2.2 residual VAE
//! (AvgDown / DupUp shortcuts, split decoder base) specialized to images: 2D
//! kernels (one temporal tap), RGBA in and out, 64 latent channels, 16x
//! spatial with no patchify. Values verbatim from the checkpoint's
//! `vae/config.json`; consumed by the shared [`crate::wan::vae`] engine.

use crate::wan::vae::WanVaeConfig;

/// Pixel channels in and out (RGBA).
pub const PIXEL_CHANNELS: usize = 4;

// The latent statistics are checkpoint data; one happens to sit near pi / 6.
#[allow(clippy::approx_constant)]
pub fn qwen_image21_vae() -> WanVaeConfig {
    WanVaeConfig {
        base_dim: 96,
        decoder_base_dim: 144,
        z_dim: 64,
        in_channels: PIXEL_CHANNELS,
        out_channels: PIXEL_CHANNELS,
        dim_mult: vec![1, 2, 4, 8, 8],
        num_res_blocks: 2,
        temporal_downsample: vec![false, true, true, true],
        is_residual: true,
        patch_size: 1,
        temporal_kernel: 1,
        spatial_compression: 16,
        temporal_compression: 8,
        norm_eps: 1e-12,
        latents_mean: vec![
            0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177,
            -0.6510, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.0210, 0.0472, 0.9320, 2.0087,
            2.4954, -0.1391, -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518,
            -1.9036, -1.9955, -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376,
            -1.0914, -1.9435, -0.2784, -1.3680, 0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.0960,
            -1.3260, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026, 1.9373, 1.4937, 0.2632,
            0.5547, -1.7121, -0.1562, 0.0304,
        ],
        latents_std: vec![
            3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910, 3.0785, 3.6500, 3.9308,
            3.0904, 2.8778, 3.7675, 3.7320, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454,
            3.0988, 4.2489, 3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.2830, 3.5694, 4.2467, 3.9037,
            3.2947, 5.0770, 3.5075, 3.2700, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819,
            3.8527, 3.8312, 3.5605, 4.3875, 3.9624, 4.0168, 3.5643, 4.0550, 5.5614, 4.2963, 4.4080,
            3.4959, 3.8747, 3.7608, 3.5735, 3.1490, 3.7662, 3.6746, 3.4563, 3.8161,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_matches_the_checkpoint() {
        let c = qwen_image21_vae();
        assert_eq!(c.latents_mean.len(), c.z_dim);
        assert_eq!(c.latents_std.len(), c.z_dim);
        // dims = 96 * [1, 1, 2, 4, 8, 8]; decoder 144 * [8, 8, 8, 4, 2, 1].
        assert_eq!(c.enc_mid_channels(), 768);
        assert_eq!(c.mid_channels(), 1152);
        assert_eq!(c.up_out_channels(c.n_up_blocks() - 1), 144);
        assert_eq!(c.temporal_downsample.len(), c.dim_mult.len() - 1);
        assert_eq!(c.pixel_channels(), PIXEL_CHANNELS);
    }
}
