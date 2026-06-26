//! LTX-2.3 distilled two-stage orchestration glue. These are stateless per-phase
//! helpers (no file IO, no residency creation) so the driver -- the conformance
//! e2e or the app layer, both of which may open files -- composes the full chain
//! while owning the per-phase residency lifecycle (each weight file gets its own
//! scoped `WeightResidency`, built -> used -> dropped, matching the upstream
//! `DiffusionStage` "build model, use, free" pattern and the parity tests).
//!
//! The denoise loop is X0-prediction velocity Euler (`sampler::euler_step`): the
//! DiT predicts velocity per step, we form the X0 estimate and step the latent.
//! Both streams are token-major `[tokens, IN_CHANNELS]` throughout the loop; the
//! caller transposes to/from CTHW (`sampler::video_*`) at the VAE/upsampler
//! boundaries.
//!
//! Chain (driver-composed): tokenize -> Gemma encoder -> FE V2 -> connector ->
//! [stage1: denoise 8 steps @ half-res] -> upsample x2 -> [stage2: renoise +
//! denoise 3 steps @ full-res] -> video VAE decode -> frames.

use super::config as dit;
use super::dit::{DitError, DitModel, DitPipelines, HostFreqs, Streams, build_split_freqs};
use super::patchify;
use super::sampler::{self, AudioLatentDims, VideoLatentDims};

use thinfer_core::backend::WgpuBackend;
use thinfer_core::residency::WeightResidency;
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::Workspace;

/// Token counts for a video+audio latent grid (DiT patch_size 1; B=1). Text KV is
/// the connector output length (`connector::CONN_SEQ`, all 1024 valid).
pub fn streams_for(vd: VideoLatentDims, ad: AudioLatentDims) -> Streams {
    Streams {
        video_tokens: vd.tokens(),
        audio_tokens: ad.tokens(),
        video_text: super::connector::CONN_SEQ,
        audio_text: super::connector::CONN_SEQ,
    }
}

/// Build the split/half-rot rope freq tables for a `(video, audio)` latent grid,
/// mirroring `precompute_freqs_cis` for `rope_type=split`. Self-attn freqs span
/// the full 3-axis (video) / 1-axis (audio) grid; the av-cross `cross_pe` uses
/// only the temporal axis (axis 0), `cross_pe_max_pos = max(20,20) = 20`. This is
/// the production analog of `dit_parity::build_freqs` (positions built from the
/// grid here instead of dumped).
pub fn build_dit_freqs(vd: VideoLatentDims, ad: AudioLatentDims, fps: f64) -> HostFreqs {
    let v_pos = patchify::build_video_positions(vd.frames, vd.height, vd.width, fps);
    let a_pos = patchify::build_audio_positions(ad.frames);
    let video_tokens = vd.tokens();
    let audio_tokens = ad.tokens();
    let theta = dit::ROPE_THETA;
    let vmax: Vec<f64> = dit::ROPE_MAX_POS.iter().map(|&x| x as f64).collect();
    let amax = [dit::AUDIO_ROPE_MAX_POS[0] as f64];
    let cross_max = [dit::AUDIO_ROPE_MAX_POS[0] as f64];
    let v_temporal = &v_pos[0..video_tokens * 2]; // axis-0 slice for the video cross_pe.
    HostFreqs {
        video_self: build_split_freqs(
            &v_pos,
            3,
            video_tokens,
            &vmax,
            dit::DIM,
            dit::N_HEADS,
            dit::HEAD_DIM,
            theta,
        ),
        audio_self: build_split_freqs(
            &a_pos,
            1,
            audio_tokens,
            &amax,
            dit::AUDIO_DIM,
            dit::AUDIO_N_HEADS,
            dit::AUDIO_HEAD_DIM,
            theta,
        ),
        video_cross: build_split_freqs(
            v_temporal,
            1,
            video_tokens,
            &cross_max,
            dit::AUDIO_CROSS_ATTENTION_DIM,
            dit::N_HEADS,
            dit::AUDIO_HEAD_DIM,
            theta,
        ),
        audio_cross: build_split_freqs(
            &a_pos,
            1,
            audio_tokens,
            &cross_max,
            dit::AUDIO_CROSS_ATTENTION_DIM,
            dit::AUDIO_N_HEADS,
            dit::AUDIO_HEAD_DIM,
            theta,
        ),
    }
}

/// Un-normalize a CTHW latent in place: `z*std + mean` per channel. The upsampler
/// operates on un-normalized latents (`upsample_video`); the DiT/VAE use the
/// normalized space. `thw = F*H*W` (channel stride, B=1).
pub fn un_normalize_cthw(latent: &mut [f32], mean: &[f32], std: &[f32], thw: usize) {
    debug_assert_eq!(latent.len(), mean.len() * thw);
    for c in 0..mean.len() {
        let (m, s) = (mean[c], std[c]);
        for v in &mut latent[c * thw..(c + 1) * thw] {
            *v = *v * s + m;
        }
    }
}

/// Inverse of [`un_normalize_cthw`]: `(z - mean)/std` per channel.
pub fn normalize_cthw(latent: &mut [f32], mean: &[f32], std: &[f32], thw: usize) {
    debug_assert_eq!(latent.len(), mean.len() * thw);
    for c in 0..mean.len() {
        let (m, s) = (mean[c], std[c]);
        for v in &mut latent[c * thw..(c + 1) * thw] {
            *v = (*v - m) / s;
        }
    }
}

/// Run an X0-prediction velocity Euler denoising loop over `sigmas` (iterating
/// `sigmas[..len-1]`; the tail 0 is the final `sigma_next`). The DiT runs once per
/// step with the resident block weights; `latent_v`/`latent_a` are token-major
/// `[tokens, IN_CHANNELS]` and are advanced in place. `vtext`/`atext` are the
/// connector cross-attn KV. Returns the final `(latent_v, latent_a)`.
///
/// No image/video conditioning here (`post_process_latent` is identity for the
/// all-ones denoise mask), so the X0 estimate is `latent - v*sigma` and the step
/// recovers the velocity exactly; the explicit X0 round-trip keeps the shape that
/// future conditioning (blend X0 against a clean latent) will slot into.
#[allow(clippy::too_many_arguments)]
pub async fn denoise_loop<S: WeightSource>(
    backend: &WgpuBackend,
    pipes: &DitPipelines,
    residency: &WeightResidency<S>,
    workspace: &Workspace<WgpuBackend>,
    model: &DitModel,
    s: Streams,
    sigmas: &[f32],
    mut latent_v: Vec<f32>,
    mut latent_a: Vec<f32>,
    vtext: &[f32],
    atext: &[f32],
    freqs: &HostFreqs,
    progress: Option<&dyn Fn(usize)>,
) -> Result<(Vec<f32>, Vec<f32>), DitError<S::Error>> {
    for (step, w) in sigmas.windows(2).enumerate() {
        let (sigma, sigma_next) = (w[0], w[1]);
        let (vel_v, vel_a) = model
            .forward(
                backend, pipes, residency, workspace, s, &latent_v, &latent_a, vtext, atext, sigma,
                freqs,
            )
            .await?;
        let x0_v = sampler::to_denoised(&latent_v, &vel_v, sigma);
        let x0_a = sampler::to_denoised(&latent_a, &vel_a, sigma);
        latent_v = sampler::euler_step(&latent_v, &x0_v, sigma, sigma_next);
        latent_a = sampler::euler_step(&latent_a, &x0_a, sigma, sigma_next);
        if let Some(p) = progress {
            p(step);
        }
    }
    Ok((latent_v, latent_a))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freqs_have_expected_shapes() {
        let vd = VideoLatentDims {
            frames: 2,
            height: 2,
            width: 2,
        };
        let ad = AudioLatentDims { frames: 3 };
        let f = build_dit_freqs(vd, ad, 24.0);
        // self freqs: [tokens*heads, head_dim] interleaved.
        assert_eq!(
            f.video_self.len(),
            vd.tokens() * dit::N_HEADS * dit::HEAD_DIM
        );
        assert_eq!(
            f.audio_self.len(),
            ad.tokens() * dit::AUDIO_N_HEADS * dit::AUDIO_HEAD_DIM
        );
        // cross freqs use the audio head_dim for the av-cross sync attention.
        assert_eq!(
            f.video_cross.len(),
            vd.tokens() * dit::N_HEADS * dit::AUDIO_HEAD_DIM
        );
        assert_eq!(
            f.audio_cross.len(),
            ad.tokens() * dit::AUDIO_N_HEADS * dit::AUDIO_HEAD_DIM
        );
    }

    #[test]
    fn streams_match_dims() {
        let vd = VideoLatentDims {
            frames: 2,
            height: 4,
            width: 4,
        };
        let ad = AudioLatentDims { frames: 5 };
        let s = streams_for(vd, ad);
        assert_eq!(s.video_tokens, 32);
        assert_eq!(s.audio_tokens, 5);
        assert_eq!(s.video_text, super::super::connector::CONN_SEQ);
    }
}
