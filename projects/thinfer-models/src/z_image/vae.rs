//! Z-Image-Turbo VAE decoder = the Flux-style `AutoencoderKL` decoder shared
//! across the model tree. The generic, config-parameterized implementation
//! (ops, residency handles, pipelines, single-shot + tiled forward) lives in
//! [`crate::common::autoencoder_kl`]; this module pins the Z-Image arch
//! (`latent_channels=16`, `block_out=[128,256,512,512]`, no post-quant-conv,
//! scalar latent pre-transform `(z / scaling) + shift`) and re-exports the
//! decoder types so existing `crate::z_image::vae::*` paths keep resolving.
//!
//! Config from `vae/config.json` (Tongyi-MAI/Z-Image-Turbo):
//! `scaling_factor = 0.3611`, `shift_factor = 0.1159`, `use_quant_conv =
//! false`, `use_post_quant_conv = false`.

use thinfer_core::residency::WeightResidency;
use thinfer_core::weight::WeightSource;

use crate::common::autoencoder_kl::{KlVaeConfig, LatentPretransform, VaeDecoderHandles};
use crate::common::loader::LoadError;

pub use crate::common::autoencoder_kl::{
    ActShape, VaeDecodeError, VaeDecoder, VaeDecoderPipelines, VaeForwardConfig, VaeForwardError,
    VaeStageSample, VaeTileConfig,
};

/// Z-Image-Turbo KL-VAE decoder arch (16-channel latent, scalar pre-transform).
pub const Z_IMAGE_KL_VAE: KlVaeConfig = KlVaeConfig {
    block_out_channels: &[128, 256, 512, 512],
    latent_channels: 16,
    has_post_quant_conv: false,
    pretransform: LatentPretransform::Scalar {
        scaling: 0.3611,
        shift: 0.1159,
    },
};

/// Register the Z-Image VAE decoder handles (diffusers-layout weight ids).
pub fn register_vae_decoder_handles<S: WeightSource>(
    residency: &WeightResidency<S>,
) -> Result<VaeDecoderHandles, LoadError> {
    crate::common::autoencoder_kl::register_vae_decoder_handles(residency, &Z_IMAGE_KL_VAE)
}
