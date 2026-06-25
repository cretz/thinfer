//! Qwen-Image 3-axis complex RoPE (`QwenEmbedRope`, `scale_rope=True`).
//!
//! Ground truth: `transformer_qwenimage.py::QwenEmbedRope` +
//! `_compute_video_freqs`. Same interleaved-pair complex family as Wan RoPE3D
//! (NOT Qwen3 half-rot; see [[project_qwen3_rope_halfrot]]), so the output is
//! the `[seq, head_dim]` `(re, im)`-interleaved f32 layout the shared rope
//! kernel consumes, produced via [`RopeEmbedder`].
//!
//! Axes `(frame, height, width)` with sub-dims `[16, 56, 56]` (sum = head_dim
//! 128), `theta = 10000`. `scale_rope` centers the two SPATIAL axes around 0
//! (`_compute_video_freqs`: `cat([neg[-(H-H//2):], pos[:H//2]])`), i.e. axis
//! positions run `-ceil(H/2) ..= floor(H/2)-1`; the frame axis is not centered
//! (positions `0..frame`). Negative positions are the complex conjugate of
//! their magnitude, handled by [`RopeEmbedder::lookup_signed`].
//!
//! Text tokens take a single shared position on all three axes, offset past the
//! image's max spatial half-extent: `p = max(H//2, W//2) + t` (the
//! `txt_freqs = pos_freqs[max_vid_index : max_vid_index + txt_len]` slice).
//!
//! Image model => `frame = 1` (one temporal slot, position 0).

use crate::common::rope_embedder::RopeEmbedder;

use super::config;

/// Max absolute RoPE position (matches the diffusers `arange(4096)` buffers);
/// covers the largest centered spatial extent + text offset we generate.
const ROPE_MAX_POS: usize = 4096;

/// Qwen-Image RoPE freq generator. Holds the shared per-axis cis tables; the
/// per-call work is just id construction + a table lookup.
#[derive(Clone, Debug)]
pub struct QwenImageRope {
    inner: RopeEmbedder,
}

impl QwenImageRope {
    pub fn new() -> Self {
        Self {
            inner: RopeEmbedder::new(
                config::ROPE_THETA,
                config::AXES_DIMS_ROPE,
                [ROPE_MAX_POS; 3],
            ),
        }
    }

    /// `head_dim` = sum of the axis sub-dims (= 128).
    pub fn head_dim(&self) -> usize {
        self.inner.head_dim()
    }

    /// Centered start position for a `scale_rope` spatial axis of length `n`:
    /// `-(n - n/2) = -ceil(n/2)` (positions then run `start ..= start+n-1`).
    fn centered_start(n: usize) -> i32 {
        -((n - n / 2) as i32)
    }

    /// Image (video) RoPE freqs, `[frame*height*width, head_dim]` interleaved
    /// `(re, im)`. Token order is row-major `(frame, height, width)`, matching
    /// the diffusers `reshape(seq_lens, -1)`.
    pub fn vid_freqs(&self, frame: usize, height: usize, width: usize) -> Vec<f32> {
        self.vid_freqs_multi(&[(frame, height, width)])
    }

    /// Multi-grid image RoPE freqs for the EDIT path: the DiT image stream is
    /// `[noise_tokens ++ ref_tokens (++ ...)]`, and `QwenEmbedRope.forward`
    /// concatenates each grid's freqs in that order. Per `_compute_video_freqs`:
    /// the FRAME axis of grid `i` is offset by `i` (`freqs_pos[0][i:i+frame]`),
    /// while the two spatial axes are centered (`scale_rope`) INDEPENDENTLY per
    /// grid. Token order within a grid is row-major `(frame, height, width)`.
    pub fn vid_freqs_multi(&self, grids: &[(usize, usize, usize)]) -> Vec<f32> {
        let n: usize = grids.iter().map(|&(f, h, w)| f * h * w).sum();
        let mut ids = Vec::with_capacity(n * 3);
        for (idx, &(frame, height, width)) in grids.iter().enumerate() {
            let h_start = Self::centered_start(height);
            let w_start = Self::centered_start(width);
            for f in 0..frame {
                for h in 0..height {
                    for w in 0..width {
                        ids.push((idx + f) as i32); // frame axis: offset by grid index
                        ids.push(h_start + h as i32);
                        ids.push(w_start + w as i32);
                    }
                }
            }
        }
        self.inner.lookup_signed(&ids)
    }

    /// Text RoPE freqs, `[txt_len, head_dim]`. Each text token shares one
    /// position on all three axes, offset past the image's max spatial
    /// half-extent: `p = max(height/2, width/2) + t`.
    pub fn txt_freqs(&self, height: usize, width: usize, txt_len: usize) -> Vec<f32> {
        self.txt_freqs_multi(&[(1, height, width)], txt_len)
    }

    /// Text RoPE freqs for the multi-grid (edit) path: the offset past the
    /// images is `max_vid_index = max over grids of max(h/2, w/2)` (the
    /// `scale_rope` branch of `QwenEmbedRope.forward`).
    pub fn txt_freqs_multi(&self, grids: &[(usize, usize, usize)], txt_len: usize) -> Vec<f32> {
        let max_vid_index = grids
            .iter()
            .map(|&(_, h, w)| (h / 2).max(w / 2))
            .max()
            .unwrap_or(0) as i32;
        let mut ids = Vec::with_capacity(txt_len * 3);
        for t in 0..txt_len {
            let p = max_vid_index + t as i32;
            ids.extend_from_slice(&[p, p, p]);
        }
        self.inner.lookup_signed(&ids)
    }
}

impl Default for QwenImageRope {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_dim_is_128() {
        assert_eq!(QwenImageRope::new().head_dim(), config::HEAD_DIM);
        assert_eq!(
            config::AXES_DIMS_ROPE.iter().sum::<usize>(),
            config::HEAD_DIM
        );
    }

    #[test]
    fn vid_freqs_shape_and_frame_zero_identity() {
        let rope = QwenImageRope::new();
        let (frame, h, w) = (1, 4, 6);
        let f = rope.vid_freqs(frame, h, w);
        assert_eq!(f.len(), frame * h * w * config::HEAD_DIM);
        // Frame axis at position 0 => its 16 lanes are (1,0) pairs (identity).
        // First token: frame=0, but h/w are centered (non-zero), so only the
        // first AXES_DIMS_ROPE[0] lanes are guaranteed identity.
        let frame_lanes = config::AXES_DIMS_ROPE[0];
        for (j, &fj) in f.iter().take(frame_lanes).enumerate() {
            let want = if j % 2 == 0 { 1.0 } else { 0.0 };
            assert!((fj - want).abs() < 1e-6, "lane {j} = {fj}");
        }
    }

    #[test]
    fn centered_start_matches_diffusers_split() {
        // cat([neg[-(n-n/2):], pos[:n/2]]) => positions -(n-n/2) ..= n/2-1.
        assert_eq!(QwenImageRope::centered_start(4), -2); // [-2,-1,0,1]
        assert_eq!(QwenImageRope::centered_start(2), -1); // [-1,0]
        assert_eq!(QwenImageRope::centered_start(5), -3); // [-3,-2,-1,0,1]
        assert_eq!(QwenImageRope::centered_start(1), -1); // [-1]
    }

    #[test]
    fn txt_freqs_shape_and_offset() {
        let rope = QwenImageRope::new();
        let (h, w, txt) = (8, 4, 3);
        let f = rope.txt_freqs(h, w, txt);
        assert_eq!(f.len(), txt * config::HEAD_DIM);
        // max_vid_index = max(4, 2) = 4; first text token sits at position 4 on
        // every axis, so it equals a centered lookup of (4,4,4) read positively.
        let direct = RopeEmbedder::new(
            config::ROPE_THETA,
            config::AXES_DIMS_ROPE,
            [ROPE_MAX_POS; 3],
        )
        .lookup_signed(&[4, 4, 4]);
        for (a, b) in f[..config::HEAD_DIM].iter().zip(direct.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
