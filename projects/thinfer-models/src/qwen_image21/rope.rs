//! Qwen-Image-2.1 3-axis RoPE over the joint text / image sequence
//! (`QwenImage21Rope`). Interleaved-pair complex rotation (`use_real=False`),
//! so the shared [`RopeEmbedder`] and `op_rope` consume it directly.
//!
//! Text tokens advance one shared position on all three axes. An image block
//! freezes the frame axis at the position its preceding text reached, lays its
//! tokens on a height / width grid CENTRED on zero (`-(h - h/2) .. h/2`), and
//! advances the position by its longer side, so a block's spatial positions do
//! not depend on where it sits in the sequence.

use crate::common::rope_embedder::RopeEmbedder;

use super::config;

/// Largest absolute position on any axis (text + images stay far below).
const ROPE_MAX_POS: usize = 16_384;

/// One run of the joint sequence, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Segment {
    /// `n` text rows (causal among themselves).
    Text(usize),
    /// An image block of `h x w` latent tokens, row-major (bidirectional).
    Image { h: usize, w: usize },
}

impl Segment {
    pub fn len(self) -> usize {
        match self {
            Segment::Text(n) => n,
            Segment::Image { h, w } => h * w,
        }
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}

/// `(frame, height, width)` position ids for every row of `segments`.
pub fn position_ids(segments: &[Segment]) -> Vec<i32> {
    let n: usize = segments.iter().map(|s| s.len()).sum();
    let mut ids = Vec::with_capacity(n * 3);
    let mut pos = 0i32;
    for seg in segments {
        match *seg {
            Segment::Text(len) => {
                for _ in 0..len {
                    ids.extend_from_slice(&[pos, pos, pos]);
                    pos += 1;
                }
            }
            Segment::Image { h, w } => {
                let (h0, w0) = (-((h - h / 2) as i32), -((w - w / 2) as i32));
                for r in 0..h as i32 {
                    for c in 0..w as i32 {
                        ids.extend_from_slice(&[pos, h0 + r, w0 + c]);
                    }
                }
                pos += h.max(w) as i32;
            }
        }
    }
    ids
}

#[derive(Clone, Debug)]
pub struct Qwen21Rope {
    inner: RopeEmbedder,
}

impl Qwen21Rope {
    pub fn new() -> Self {
        Self {
            inner: RopeEmbedder::new(
                config::ROPE_THETA,
                config::AXES_DIMS_ROPE,
                [ROPE_MAX_POS; 3],
            ),
        }
    }

    /// Interleaved `(re, im)` freqs `[rows, HEAD_DIM]` for `segments`.
    pub fn freqs(&self, segments: &[Segment]) -> Vec<f32> {
        self.inner.lookup_signed(&position_ids(segments))
    }
}

impl Default for Qwen21Rope {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference's worked layout: text, a 4x6 condition image, text, then
    /// a 4x4 target.
    #[test]
    fn image_blocks_centre_and_advance_by_longer_side() {
        let segs = [
            Segment::Text(3),
            Segment::Image { h: 4, w: 6 },
            Segment::Text(2),
            Segment::Image { h: 4, w: 4 },
        ];
        let ids = position_ids(&segs);
        assert_eq!(ids.len(), (3 + 24 + 2 + 16) * 3);
        assert_eq!(&ids[6..9], &[2, 2, 2]);
        // First condition token: frame 3, h -2, w -3; last: h 1, w 2.
        assert_eq!(&ids[9..12], &[3, -2, -3]);
        assert_eq!(&ids[(3 + 23) * 3..(3 + 24) * 3], &[3, 1, 2]);
        // Text after the image resumes at 3 + max(4, 6) = 9.
        assert_eq!(&ids[(3 + 24) * 3..(3 + 25) * 3], &[9, 9, 9]);
        // Target frame = 11 (after two more text tokens).
        assert_eq!(&ids[(3 + 24 + 2) * 3..(3 + 24 + 3) * 3], &[11, -2, -2]);
    }
}
