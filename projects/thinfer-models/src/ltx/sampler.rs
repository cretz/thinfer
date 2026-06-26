//! LTX-2.3 distilled two-stage sampler: X0-prediction velocity Euler steps,
//! seeded Gaussian noise + flow re-noising, latent-grid shape derivation, and the
//! token-major <-> CTHW layout transpose the DiT and VAE/upsampler disagree on.
//!
//! Ground truth:
//! - Euler step: `ltx_core/components/diffusion_steps.py` `EulerDiffusionStep` +
//!   `ltx_core/utils.py` `to_denoised`/`to_velocity`. The DiT predicts velocity
//!   `v`; the denoised estimate is `x0 = latent - v*sigma`; the step advances
//!   `latent += v_eff*(sigma_next - sigma)` where `v_eff = (latent - x0)/sigma`
//!   (== `v` for unconditioned t2v; the x0 round-trip is kept so image/video
//!   conditioning can blend `x0` via `post_process_latent` before re-deriving v).
//! - Re-noise: `ltx_core/components/noisers.py` `GaussianNoiser` -> `latent =
//!   lerp(initial, noise, scale)` then `lerp(clean, latent, mask)`. For
//!   unconditioned t2v (`mask==1`) this is `(1-scale)*initial + scale*noise`;
//!   stage 1 uses `initial=0, scale=1` (pure noise), stage 2 re-noises the
//!   upscaled latent at `scale = STAGE2_SIGMAS[0]`.
//! - Shapes: `ltx_core/types.py` `VideoLatentShape::from_pixel_shape` (video) and
//!   `AudioLatentShape::from_duration` (audio).
//!
//! The sigma tables live in [`super::sampler`] consts; this module is pure host
//! arithmetic (no GPU, no weights) so it is unit-tested directly.

use super::config as dit;

// --- distilled schedule constants (`ltx_pipelines/utils/constants.py`) ---

/// Stage 1: 8 steps at half resolution.
pub const STAGE1_SIGMAS: [f32; 9] = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0,
];
/// Stage 2: 3-step refine (tail subset) after the latent is upscaled x2 and
/// re-noised to sigma 0.909375.
pub const STAGE2_SIGMAS: [f32; 4] = [0.909375, 0.725, 0.421875, 0.0];
/// Two-stage: H,W divisible by 64; frames = 8k+1. Default render target.
pub const DEFAULT_WIDTH: usize = 1536;
pub const DEFAULT_HEIGHT: usize = 1024;
pub const DEFAULT_FRAMES: usize = 121;
pub const DEFAULT_FPS: usize = 24;

/// Video VAE downsample (time, height, width); see `SpatioTemporalScaleFactors`.
const VAE_T: usize = 8;
const VAE_HW: usize = 32;
/// Audio latents per second: `sample_rate / hop / downsample = 16000/160/4`.
const AUDIO_LATENTS_PER_SEC: f64 = 16000.0 / 160.0 / 4.0;
/// Audio latent feature width fed to the DiT patchifier: `channels * mel_bins`
/// (`8 * 16`), matching the video stream's `IN_CHANNELS` so one patchify path
/// serves both.
pub const AUDIO_LATENT_CHANNELS: usize = 8;
pub const AUDIO_MEL_BINS: usize = 16;

/// Video latent grid `[channels, frames, height, width]` for a pixel target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoLatentDims {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
}

impl VideoLatentDims {
    /// `from_pixel_shape`: `frames = (F-1)/8 + 1`, `height = H/32`, `width = W/32`.
    pub fn from_pixels(num_frames: usize, height: usize, width: usize) -> Self {
        Self {
            frames: (num_frames - 1) / VAE_T + 1,
            height: height / VAE_HW,
            width: width / VAE_HW,
        }
    }
    /// DiT token count (patch_size 1: one token per latent cell).
    pub fn tokens(&self) -> usize {
        self.frames * self.height * self.width
    }
    /// Latent element count (`IN_CHANNELS` per cell).
    pub fn elems(&self) -> usize {
        dit::IN_CHANNELS * self.tokens()
    }
}

/// Audio latent grid `[channels, frames, mel_bins]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioLatentDims {
    pub frames: usize,
}

impl AudioLatentDims {
    /// `from_duration`: `frames = round(duration * latents_per_second)`,
    /// `duration = num_frames / fps`.
    pub fn from_video(num_frames: usize, fps: f64) -> Self {
        let duration = num_frames as f64 / fps;
        Self {
            frames: (duration * AUDIO_LATENTS_PER_SEC).round() as usize,
        }
    }
    /// DiT token count (one token per audio latent frame).
    pub fn tokens(&self) -> usize {
        self.frames
    }
    /// Latent element count: `channels * mel_bins` features per frame (==
    /// `IN_CHANNELS`).
    pub fn elems(&self) -> usize {
        AUDIO_LATENT_CHANNELS * AUDIO_MEL_BINS * self.frames
    }
}

/// Deterministic per-stage / per-step seed derivation (SplitMix64 mixing constant
/// folded in), so stage 1, the stage-2 video re-noise, and the audio re-noise all
/// draw independent but reproducible noise from one user seed.
pub fn substream_seed(seed: u64, stream: u64) -> u64 {
    seed.wrapping_add((stream + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// Deterministic standard-normal samples via SplitMix64 -> Box-Muller (no `rand`
/// dep). Same generator family as the other model pipelines so behavior is
/// uniform across the tree.
pub fn gaussian_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next_u64 = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = ((next_u64() >> 11) as f64 + 1.0) * (1.0 / ((1u64 << 53) as f64 + 1.0));
        let u2 = (next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push((r * theta.cos()) as f32);
        if out.len() < n {
            out.push((r * theta.sin()) as f32);
        }
    }
    out
}

/// Flow re-noise (unconditioned, `mask==1`): `(1-scale)*initial + scale*noise`.
/// `initial` and `noise` must match in length. Stage 1 passes `initial=zeros`,
/// `scale=1.0` (pure noise); stage 2 passes the upscaled latent + `scale=σ0`.
pub fn renoise(initial: &[f32], noise: &[f32], scale: f32) -> Vec<f32> {
    assert_eq!(initial.len(), noise.len(), "renoise length mismatch");
    initial
        .iter()
        .zip(noise)
        .map(|(i, n)| i + (n - i) * scale)
        .collect()
}

/// X0 estimate from the predicted velocity: `x0 = latent - v*sigma`.
pub fn to_denoised(latent: &[f32], velocity: &[f32], sigma: f32) -> Vec<f32> {
    assert_eq!(latent.len(), velocity.len(), "to_denoised length mismatch");
    latent
        .iter()
        .zip(velocity)
        .map(|(x, v)| x - v * sigma)
        .collect()
}

/// One Euler step from `sigma` to `sigma_next` given the predicted velocity.
/// Advances `latent + v_eff*(sigma_next - sigma)`, re-deriving `v_eff` from the
/// (possibly conditioning-blended) `denoised` so the step matches upstream even
/// when `denoised != latent - v*sigma`. `sigma` must be > 0 (true for every
/// step in the distilled schedules, whose tails end at 0 only as `sigma_next`).
pub fn euler_step(latent: &[f32], denoised: &[f32], sigma: f32, sigma_next: f32) -> Vec<f32> {
    assert!(sigma > 0.0, "euler_step: sigma must be > 0");
    assert_eq!(latent.len(), denoised.len(), "euler_step length mismatch");
    let dt = sigma_next - sigma;
    latent
        .iter()
        .zip(denoised)
        .map(|(x, d)| {
            let v_eff = (x - d) / sigma;
            x + v_eff * dt
        })
        .collect()
}

/// Convert the DiT's token-major video latent `[tokens, C]` (token `t = f*H*W +
/// h*W + w`, channel-last) to the channel-first CTHW `[C, F, H, W]` the video
/// VAE and latent upsampler consume. `C = IN_CHANNELS`.
pub fn video_tokens_to_cthw(tokens_major: &[f32], d: VideoLatentDims) -> Vec<f32> {
    let c = dit::IN_CHANNELS;
    let thw = d.tokens();
    assert_eq!(tokens_major.len(), c * thw, "video latent size");
    let mut out = vec![0.0f32; c * thw];
    for t in 0..thw {
        for ch in 0..c {
            out[ch * thw + t] = tokens_major[t * c + ch];
        }
    }
    out
}

/// Inverse of [`video_tokens_to_cthw`]: CTHW `[C, F, H, W]` -> token-major
/// `[tokens, C]` for the DiT patchifier.
pub fn video_cthw_to_tokens(cthw: &[f32], d: VideoLatentDims) -> Vec<f32> {
    let c = dit::IN_CHANNELS;
    let thw = d.tokens();
    assert_eq!(cthw.len(), c * thw, "video latent size");
    let mut out = vec![0.0f32; c * thw];
    for t in 0..thw {
        for ch in 0..c {
            out[t * c + ch] = cthw[ch * thw + t];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_dims_default_target() {
        // 121 frames, 1024x1536 -> latent 16 x 32 x 48.
        let d = VideoLatentDims::from_pixels(121, 1024, 1536);
        assert_eq!(d.frames, 16);
        assert_eq!(d.height, 32);
        assert_eq!(d.width, 48);
        assert_eq!(d.tokens(), 16 * 32 * 48);
        // Stage 1 half-res 512x768 -> 16 x 16 x 24.
        let s1 = VideoLatentDims::from_pixels(121, 512, 768);
        assert_eq!((s1.frames, s1.height, s1.width), (16, 16, 24));
    }

    #[test]
    fn audio_dims_default_target() {
        // 121 frames @ 24fps -> 5.041667s * 25 = 126.04 -> round 126.
        let a = AudioLatentDims::from_video(121, 24.0);
        assert_eq!(a.frames, 126);
        assert_eq!(a.elems(), 8 * 16 * 126);
    }

    #[test]
    fn renoise_endpoints() {
        let init = vec![2.0, -4.0];
        let noise = vec![1.0, 1.0];
        // scale 0 -> initial unchanged.
        assert_eq!(renoise(&init, &noise, 0.0), init);
        // scale 1 -> pure noise.
        assert_eq!(renoise(&init, &noise, 1.0), noise);
        // scale 0.5 -> midpoint.
        assert_eq!(renoise(&init, &noise, 0.5), vec![1.5, -1.5]);
    }

    #[test]
    fn euler_last_step_lands_on_x0() {
        // sigma_next == 0: latent + v_eff*(0 - sigma) = latent - v_eff*sigma = x0.
        let latent = vec![1.0, 2.0, 3.0];
        let v = vec![0.5, 0.5, 0.5];
        let sigma = 0.4;
        let x0 = to_denoised(&latent, &v, sigma);
        let stepped = euler_step(&latent, &x0, sigma, 0.0);
        for (a, b) in stepped.iter().zip(&x0) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn euler_unconditioned_recovers_velocity() {
        // denoised = x0 -> v_eff == v, so step == latent + v*(dt).
        let latent = vec![1.0, -2.0];
        let v = vec![0.25, -0.75];
        let (sigma, sigma_next) = (0.9, 0.7);
        let x0 = to_denoised(&latent, &v, sigma);
        let got = euler_step(&latent, &x0, sigma, sigma_next);
        let exp: Vec<f32> = latent
            .iter()
            .zip(&v)
            .map(|(x, vv)| x + vv * (sigma_next - sigma))
            .collect();
        for (a, b) in got.iter().zip(&exp) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn cthw_roundtrip() {
        let d = VideoLatentDims {
            frames: 2,
            height: 1,
            width: 2,
        };
        // build token-major with a distinct value per (token, channel).
        let c = dit::IN_CHANNELS;
        let thw = d.tokens();
        let mut tm = vec![0.0f32; c * thw];
        for t in 0..thw {
            for ch in 0..c {
                tm[t * c + ch] = (t * 1000 + ch) as f32;
            }
        }
        let cthw = video_tokens_to_cthw(&tm, d);
        // spot-check: token 1 channel 3 lands at [3*thw + 1] with value 1*1000+3.
        assert_eq!(cthw[3 * thw + 1], (1000 + 3) as f32);
        assert_eq!(video_cthw_to_tokens(&cthw, d), tm);
    }

    #[test]
    fn substream_seeds_distinct() {
        let s = 42;
        assert_ne!(substream_seed(s, 0), substream_seed(s, 1));
        assert_ne!(substream_seed(s, 1), substream_seed(s, 2));
    }
}
