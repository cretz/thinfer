//! Ideogram-4 VAE glue: the Flux2 KL `AutoencoderKL` decoder (shared impl in
//! [`crate::common::autoencoder_kl`]) pinned to the Ideogram-4 arch, plus the
//! host-side latent denorm + unpatch that turns the DiT's packed
//! `[num_image, 128]` velocity-integrated latent into the spatial
//! `[32, h_in, w_in]` tensor the decoder consumes.
//!
//! Ground truth: `third-party/ideogram4/src/ideogram4/{autoencoder,
//! latent_norm,pipeline_ideogram4}.py` (`Ideogram4Pipeline._decode`).
//!
//! Arch (from the FLUX.2-VAE checkpoint, `black-forest-labs/FLUX.2-VAE`): the
//! decoder is `ch=96` -> `block_out_channels=[96,192,384,384]` (mid 384, final
//! 96), `z_channels=32`, with a 1x1 `post_quant_conv`. NOTE the autoencoder is
//! ASYMMETRIC (its encoder is ch=128); we only decode, so only the decoder
//! widths matter. This differs from the `AutoEncoderParams` defaults in
//! `autoencoder.py` (ch=128) -- those are not what this checkpoint ships.
//!
//! Decode (`_decode`): `z = z * LATENT_SCALE + LATENT_SHIFT` (per-channel over
//! the 128-dim patch), then unpatch `(gh, gw, 2, 2, 32) -> (32, gh*2, gw*2)`,
//! then the KL decoder. Output `(decoded + 1) * 127.5` is the pipeline's job.

use thinfer_core::residency::WeightResidency;
use thinfer_core::weight::WeightSource;

use crate::common::autoencoder_kl::{KlVaeConfig, LatentPretransform, VaeDecoderHandles};
use crate::common::loader::LoadError;

use super::config;

pub use crate::common::autoencoder_kl::{
    VaeDecodeError, VaeDecoder, VaeDecoderPipelines, VaeStageSample, VaeTileConfig,
};

/// Ideogram-4 Flux2 KL-VAE decoder arch. 32-channel latent + 1x1 post-quant
/// conv; the per-channel denorm + unpatch happen here (host side), so the
/// decoder takes already-spatial latents (`LatentPretransform::None`).
pub const IDEOGRAM4_KL_VAE: KlVaeConfig = KlVaeConfig {
    block_out_channels: &[96, 192, 384, 384],
    latent_channels: 32,
    has_post_quant_conv: true,
    pretransform: LatentPretransform::None,
};

/// AE channels (`z_channels`) the 128-dim patch unpacks into.
pub const AE_CHANNELS: usize = 32;

/// Register the Ideogram-4 VAE decoder handles from a diffusers-layout VAE
/// source (FLUX.2-VAE safetensors). Weight ids are the common KL schema plus
/// the top-level `post_quant_conv`.
pub fn register_vae_decoder_handles<S: WeightSource>(
    residency: &WeightResidency<S>,
) -> Result<VaeDecoderHandles, LoadError> {
    crate::common::autoencoder_kl::register_vae_decoder_handles(residency, &IDEOGRAM4_KL_VAE)
}

/// Denorm + unpatch the packed latent into spatial CHW for the KL decoder.
///
/// Input: `latent_tokens` is `[num_image, IN_CHANNELS]` row-major
/// (`num_image = grid_h * grid_w`, `IN_CHANNELS = patch^2 * AE_CHANNELS =
/// 128`). Output: `[AE_CHANNELS, grid_h*2, grid_w*2]` row-major (CHW).
///
/// Mirrors `Ideogram4Pipeline._decode`:
/// ```text
/// z = z * LATENT_SCALE + LATENT_SHIFT                       # per 128-channel
/// z = z.view(gh, gw, p, p, c).permute(5,0,2,1,3)           # -> (c, gh, p, gw, p)
///     .reshape(c, gh*p, gw*p)
/// ```
/// so `out[c, gh_i*2+ph, gw_i*2+pw] = z[token=gh_i*gw+gw_i, ph*2*c + pw*c + c]`.
pub fn unpatch_denorm(latent_tokens: &[f32], grid_h: usize, grid_w: usize) -> Vec<f32> {
    const P: usize = config::PATCH_SIZE; // 2
    let c = AE_CHANNELS;
    let in_ch = P * P * c; // 128
    assert_eq!(in_ch, config::IN_CHANNELS, "patch^2 * ae != IN_CHANNELS");
    let num_image = grid_h * grid_w;
    assert_eq!(
        latent_tokens.len(),
        num_image * in_ch,
        "unpatch_denorm: expected {} latents, got {}",
        num_image * in_ch,
        latent_tokens.len()
    );

    let out_h = grid_h * P;
    let out_w = grid_w * P;
    let plane = out_h * out_w;
    let mut out = vec![0.0_f32; c * plane];

    for gh_i in 0..grid_h {
        for gw_i in 0..grid_w {
            let token = gh_i * grid_w + gw_i;
            let tok_base = token * in_ch;
            for ph in 0..P {
                for pw in 0..P {
                    // patch element layout within the 128-vec is [ph, pw, c].
                    let patch_base = (ph * P + pw) * c;
                    let oy = gh_i * P + ph;
                    let ox = gw_i * P + pw;
                    let spatial = oy * out_w + ox;
                    for ci in 0..c {
                        // Norm is over the full 128-dim patch channel
                        // (`ph*64 + pw*32 + ci`), applied before unpatch.
                        let nidx = patch_base + ci;
                        let v = latent_tokens[tok_base + nidx];
                        out[ci * plane + spatial] = v * LATENT_SCALE[nidx] + LATENT_SHIFT[nidx];
                    }
                }
            }
        }
    }
    out
}

/// Latent denorm shift, the full 128-dim patch vec (`latent_norm.py::
/// LATENT_SHIFT`). Applied before unpatch, so the norm channel for output
/// `(ci, ph, pw)` is `ph*64 + pw*32 + ci`. The four (ph,pw) blocks are
/// near-identical but NOT bit-equal, so keep all 128 and index exactly.
///
/// Verbatim from `latent_norm.py` (f32-parsed; the extra digits round to the
/// nearest f32, identical to the pyref's `torch.tensor(..., float32)`).
#[allow(clippy::excessive_precision)]
const LATENT_SHIFT_128: [f32; 128] = [
    0.01984364,
    0.10149707,
    0.29689495,
    0.27188619,
    -0.21445648,
    -0.15979549,
    0.05021099,
    -0.15083604,
    -0.15360136,
    -0.20131799,
    0.01922352,
    0.0622626,
    0.10140969,
    -0.06739428,
    0.3758261,
    -0.233712,
    0.35164491,
    -0.02590912,
    -0.0271935,
    -0.10833897,
    -0.1476848,
    -0.01130957,
    -0.2298372,
    0.23526423,
    -0.10893522,
    0.11957631,
    0.04047799,
    0.3134589,
    -0.17225064,
    -0.18646109,
    -0.34691978,
    -0.03571246,
    0.02583857,
    0.10190072,
    0.28402294,
    0.26952152,
    -0.21634675,
    -0.17938656,
    0.04358909,
    -0.15007621,
    -0.1548502,
    -0.18971131,
    0.02710861,
    0.05609494,
    0.10697846,
    -0.06854968,
    0.38167698,
    -0.24269937,
    0.35705471,
    -0.03063305,
    -0.02946109,
    -0.11244286,
    -0.14336038,
    -0.01362137,
    -0.21863696,
    0.23228983,
    -0.11739769,
    0.11693044,
    0.02563311,
    0.31356594,
    -0.17420591,
    -0.19006285,
    -0.34905377,
    -0.04025005,
    0.01924137,
    0.07652984,
    0.2995608,
    0.2628057,
    -0.22011674,
    -0.12715361,
    0.04879879,
    -0.14075719,
    -0.15935895,
    -0.2123584,
    0.01974813,
    0.05523547,
    0.10011992,
    -0.06428964,
    0.37781868,
    -0.21491644,
    0.34254215,
    -0.03153528,
    -0.0310082,
    -0.10761415,
    -0.14730405,
    -0.02475182,
    -0.2285588,
    0.2515081,
    -0.10445128,
    0.12446,
    0.07062869,
    0.30880162,
    -0.18016875,
    -0.18869164,
    -0.34533499,
    -0.0129177,
    0.02578168,
    0.07993659,
    0.28642181,
    0.26038408,
    -0.22459419,
    -0.14820155,
    0.04059549,
    -0.14043529,
    -0.16111187,
    -0.2020305,
    0.02602069,
    0.04852717,
    0.10432153,
    -0.06309942,
    0.38402443,
    -0.22397003,
    0.34814481,
    -0.03774432,
    -0.03381438,
    -0.11245691,
    -0.14128767,
    -0.02853208,
    -0.21752016,
    0.24872463,
    -0.11399775,
    0.1222687,
    0.05620835,
    0.309178,
    -0.18065738,
    -0.19401479,
    -0.34495114,
    -0.01760592,
];

#[allow(clippy::excessive_precision)]
const LATENT_SCALE_128: [f32; 128] = [
    1.63933691, 1.70204478, 1.73642566, 1.90004803, 1.6675316, 1.69059584, 1.56853198, 1.62314944,
    1.89106626, 1.58086668, 1.60822129, 1.60962993, 1.63322129, 1.56074359, 1.73419528, 1.7919265,
    1.64040632, 1.66802808, 1.60390303, 1.75480492, 1.63187587, 1.64334594, 1.61722884, 1.60146046,
    1.63459219, 1.55291476, 1.68771497, 1.68415657, 1.78966054, 1.66631641, 1.65626686, 1.65976433,
    1.63487607, 1.69513249, 1.72933756, 1.91310663, 1.67035057, 1.72286863, 1.56719251, 1.61934825,
    1.88628859, 1.56911539, 1.59455129, 1.60829869, 1.62470611, 1.56052853, 1.73677003, 1.77563606,
    1.63732541, 1.66370527, 1.59508952, 1.75153949, 1.63029275, 1.64517667, 1.61659342, 1.59722044,
    1.64103121, 1.5408531, 1.68610394, 1.67772755, 1.78998563, 1.66621713, 1.65458955, 1.66041308,
    1.64710857, 1.68163503, 1.74000294, 1.92784786, 1.67411194, 1.67395548, 1.57406532, 1.62199356,
    1.87618195, 1.5584375, 1.57438785, 1.61711053, 1.63094305, 1.55644029, 1.73124302, 1.80666627,
    1.6463621, 1.65932006, 1.60816188, 1.75682671, 1.64695873, 1.63121722, 1.61380832, 1.60478651,
    1.63396035, 1.53505068, 1.65534289, 1.67132281, 1.80317197, 1.6767314, 1.65700938, 1.68426259,
    1.65339716, 1.67540638, 1.73298504, 1.94067348, 1.67893609, 1.70635117, 1.5730906, 1.61928553,
    1.87148809, 1.56244866, 1.56697152, 1.61584394, 1.62759496, 1.55480378, 1.73484107, 1.79055143,
    1.64688773, 1.66121492, 1.60135887, 1.75254572, 1.64798332, 1.62989921, 1.61381592, 1.60792883,
    1.63939668, 1.53075757, 1.65371318, 1.66801185, 1.80029087, 1.67591476, 1.65655173, 1.68533454,
];

// `unpatch_denorm` indexes the full 128-vec via `ph*64 + pw*32 + ci`. Bind the
// arrays to the names used in that loop.
const LATENT_SHIFT: &[f32; 128] = &LATENT_SHIFT_128;
const LATENT_SCALE: &[f32; 128] = &LATENT_SCALE_128;
