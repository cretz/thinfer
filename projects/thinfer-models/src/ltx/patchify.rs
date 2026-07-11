//! LTX-2.3 patchifier position construction: maps a latent grid to the
//! physical-coordinate `[start, end)` rope position bounds the DiT consumes
//! (`Modality.positions`). Host-side, pure arithmetic.
//!
//! Ground truth: `ltx_core/components/patchifiers.py` (`VideoLatentPatchifier.
//! get_patch_grid_bounds` + `get_pixel_coords`, `AudioPatchifier`) and
//! `ltx_core/tools.py` `VideoLatentTools` (the `positions[:,0] /= fps` step).
//!
//! Video (DiT patch_size=1 -> one token per latent cell, frame-major then h, w):
//! per token `(f, h, w)` the bounds are
//! ```text
//! t_start = (f*8     + 1 - 8).clamp(0) / fps    // causal_fix + /fps, seconds
//! t_end   = ((f+1)*8 + 1 - 8).clamp(0) / fps
//! h_start = h*32,  h_end = (h+1)*32             // pixels
//! w_start = w*32,  w_end = (w+1)*32
//! ```
//! laid out `[3, T, 2]` (axis-major: all temporal bounds, then height, width).
//!
//! Audio: per latent frame `f`, the bounds are timestamps in seconds
//! ```text
//! mel(f)  = (f*4 + 1 - 4).clamp(0)              // is_causal, downsample 4
//! t_start = mel(f)   * hop / sr                 // hop 160, sr 16000
//! t_end   = mel(f+1) * hop / sr
//! ```
//! laid out `[1, T, 2]`. The DiT cross-attn temporal `cross_pe` uses axis 0 of
//! each (video / audio), so both layouts put the temporal axis first.

use super::config as dit;

/// Video VAE downsample factors `(time, height, width)` -- the latent->pixel
/// scale applied before the causal-fix + fps division.
pub const VAE_SCALE: (i64, i64, i64) = (8, 32, 32);
/// Audio: spectrogram frames per latent frame, hop, and sample rate.
pub const AUDIO_DOWNSAMPLE: i64 = 4;
pub const AUDIO_HOP: f64 = 160.0;
pub const AUDIO_SR: f64 = 16000.0;

/// Build the video rope position bounds `[3, T, 2]` (f32) for an `(f, h, w)`
/// latent grid. Token order is frame-major then height then width (matching the
/// `b (f h w) c` patchify rearrange).
pub fn build_video_positions(frames: usize, height: usize, width: usize, fps: f64) -> Vec<f32> {
    let (st, _sh, _sw) = VAE_SCALE;
    let t = frames * height * width;
    let mut out = vec![0.0f32; 3 * t * 2];
    let causal = |pix: i64| -> f64 { ((pix + 1 - st).max(0)) as f64 / fps };
    let mut idx = 0;
    for f in 0..frames {
        for h in 0..height {
            for w in 0..width {
                // axis 0 = temporal (seconds), 1 = height (px), 2 = width (px).
                let t_base = idx * 2;
                out[t_base] = causal((f as i64) * st) as f32;
                out[t_base + 1] = causal(((f + 1) as i64) * st) as f32;
                let h_base = (t + idx) * 2;
                out[h_base] = (h as i64 * VAE_SCALE.1) as f32;
                out[h_base + 1] = ((h + 1) as i64 * VAE_SCALE.1) as f32;
                let w_base = (2 * t + idx) * 2;
                out[w_base] = (w as i64 * VAE_SCALE.2) as f32;
                out[w_base + 1] = ((w + 1) as i64 * VAE_SCALE.2) as f32;
                idx += 1;
            }
        }
    }
    out
}

/// Build the audio rope position bounds `[1, T, 2]` (f32, timestamps in seconds)
/// for `T` audio latent frames.
pub fn build_audio_positions(frames: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; frames * 2];
    let sec = |latent: i64| -> f64 {
        let mel = (latent * AUDIO_DOWNSAMPLE + 1 - AUDIO_DOWNSAMPLE).max(0);
        mel as f64 * AUDIO_HOP / AUDIO_SR
    };
    for f in 0..frames {
        out[f * 2] = sec(f as i64) as f32;
        out[f * 2 + 1] = sec((f + 1) as i64) as f32;
    }
    out
}

/// Token counts for a `(frames, height, width)` video grid + `audio_frames`
/// audio latent frames (B=1; DiT patch_size=1).
pub fn token_counts(
    frames: usize,
    height: usize,
    width: usize,
    audio_frames: usize,
) -> (usize, usize) {
    (frames * height * width, audio_frames)
}

/// Default rope thetas / max_pos for the video + audio self-attn freqs (from the
/// DiT config; re-exported here so the pipeline builds freqs without reaching
/// into two modules).
pub fn video_max_pos() -> [f64; 3] {
    [
        dit::ROPE_MAX_POS[0] as f64,
        dit::ROPE_MAX_POS[1] as f64,
        dit::ROPE_MAX_POS[2] as f64,
    ]
}
pub fn audio_max_pos() -> [f64; 1] {
    [dit::AUDIO_ROPE_MAX_POS[0] as f64]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_positions_causal_first_frame() {
        // 2 frames, 1x1 spatial, fps 24. Frame 0 temporal: causal((0)*8) =
        // (0+1-8).clamp(0)/24 = 0; end causal(8) = (8+1-8)/24 = 1/24.
        let p = build_video_positions(2, 1, 1, 24.0);
        let t = 2; // tokens
        // token 0 temporal bounds (axis 0):
        assert!((p[0] - 0.0).abs() < 1e-6, "t0 start");
        assert!((p[1] - (1.0 / 24.0)).abs() < 1e-6, "t0 end");
        // token 1 (frame 1): start causal(8)=1/24, end causal(16)=(16+1-8)/24=9/24.
        assert!((p[2] - (1.0 / 24.0)).abs() < 1e-6, "t1 start");
        assert!((p[3] - (9.0 / 24.0)).abs() < 1e-6, "t1 end");
        // height axis (axis 1) for token 0: [0, 32].
        let h0 = t * 2;
        assert_eq!(p[h0], 0.0);
        assert_eq!(p[h0 + 1], 32.0);
        // width axis (axis 2) for token 0: [0, 32].
        let w0 = (2 * t) * 2;
        assert_eq!(p[w0], 0.0);
        assert_eq!(p[w0 + 1], 32.0);
    }

    #[test]
    fn video_token_order_frame_major() {
        // 1 frame, 2x2 spatial -> 4 tokens ordered (h,w): (0,0)(0,1)(1,0)(1,1).
        let p = build_video_positions(1, 2, 2, 24.0);
        let t = 4;
        // token 1 = (h=0, w=1): width bounds [32, 64].
        let w1 = (2 * t + 1) * 2;
        assert_eq!(p[w1], 32.0);
        assert_eq!(p[w1 + 1], 64.0);
        // token 2 = (h=1, w=0): height bounds [32, 64], width [0,32].
        let h2 = (t + 2) * 2;
        assert_eq!(p[h2], 32.0);
        let w2 = (2 * t + 2) * 2;
        assert_eq!(p[w2], 0.0);
    }

    #[test]
    fn audio_positions_causal() {
        // frame 0: mel = (0+1-4).clamp(0)=0 -> 0s; end mel=(4+1-4)=1 -> 160/16000.
        let p = build_audio_positions(3);
        assert!((p[0] - 0.0).abs() < 1e-9, "f0 start");
        assert!((p[1] - (160.0 / 16000.0)).abs() < 1e-9, "f0 end");
        // frame 1: start mel=(4+1-4)=1 -> 160/16000; end mel=(8+1-4)=5 -> 800/16000.
        assert!((p[2] - (160.0 / 16000.0)).abs() < 1e-9, "f1 start");
        assert!((p[3] - (800.0 / 16000.0)).abs() < 1e-9, "f1 end");
    }
}
