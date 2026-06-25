//! Ideogram-4 3-axis interleaved MRoPE freq table.
//!
//! Mirrors `Ideogram4MRoPE` in `modeling_ideogram4.py`. Per token there is a
//! position triple `(t, h, w)`. The base angles are the t-axis freqs; the h and
//! w axes overwrite interleaved index slots within `section*3` spans (offset 1
//! for h, 2 for w). The result is `head_dim/2` angles, emitted as interleaved
//! `(cos, sin)` words exactly like [`RopeEmbedder::lookup`], so it feeds the
//! existing half-rot kernel (`op_rope_halfrot` / `RopeF32HalfRot`): that kernel
//! applies angle `i` to element pair `(i, i + head_dim/2)` and reads freqs as
//! per-token `head_dim/2` `(cos, sin)` words. (Half-rot, NOT interleaved-pair
//! like Wan; consistent with our Qwen3 half-rot.)
//!
//! Upstream computes `inv_freq` and the outer product, then duplicates
//! `cat(freqs, freqs)` before cos/sin. The duplication is implicit here: the
//! half-rot kernel reuses each angle for both halves, so we only store the
//! `head_dim/2` distinct angles. f64 internally (mirror upstream), f32 out.

use super::config;

/// Axis (0=t, 1=h, 2=w) that supplies the angle at frequency index `j`.
/// Base is t; h overwrites `j in {1,4,..}` and w `j in {2,5,..}`, each within
/// `section[axis]*3`.
#[inline]
fn axis_for_index(j: usize, section: [usize; 3]) -> usize {
    if j % 3 == 1 && j < section[1] * 3 {
        1
    } else if j % 3 == 2 && j < section[2] * 3 {
        2
    } else {
        0
    }
}

/// Build the freqs buffer for `position_ids` (`[seq, 3]` row-major `(t,h,w)`),
/// returning `[seq, head_dim]` f32 interleaved `(cos, sin)` (head_dim/2 words
/// per token), ready for `seq::freqs_upload_bytes` + `op_rope_halfrot`.
pub fn build_freqs(
    position_ids: &[i64],
    head_dim: usize,
    theta: f32,
    section: [usize; 3],
) -> Vec<f32> {
    assert!(
        position_ids.len().is_multiple_of(3),
        "position_ids must be [seq,3]"
    );
    assert!(head_dim.is_multiple_of(2), "head_dim must be even");
    let seq = position_ids.len() / 3;
    let half = head_dim / 2;
    let theta = theta as f64;
    // inv_freq[j] = 1 / theta^(2j / head_dim), j in 0..half.
    let inv_freq: Vec<f64> = (0..half)
        .map(|j| 1.0 / theta.powf((2 * j) as f64 / head_dim as f64))
        .collect();
    let axis: Vec<usize> = (0..half).map(|j| axis_for_index(j, section)).collect();

    let mut out = vec![0.0_f32; seq * head_dim];
    for row in 0..seq {
        let pos = [
            position_ids[row * 3] as f64,
            position_ids[row * 3 + 1] as f64,
            position_ids[row * 3 + 2] as f64,
        ];
        let dst = &mut out[row * head_dim..(row + 1) * head_dim];
        for j in 0..half {
            let arg = pos[axis[j]] * inv_freq[j];
            dst[2 * j] = arg.cos() as f32;
            dst[2 * j + 1] = arg.sin() as f32;
        }
    }
    out
}

/// Convenience for the DiT config.
pub fn build_freqs_dit(position_ids: &[i64]) -> Vec<f32> {
    build_freqs(
        position_ids,
        config::HEAD_DIM,
        config::ROPE_THETA,
        config::MROPE_SECTION,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::rope_embedder::RopeEmbedder;

    #[test]
    fn pos_zero_is_identity() {
        let f = build_freqs(&[0, 0, 0], 8, 5_000_000.0, [1, 1, 1]);
        for j in 0..4 {
            assert!((f[2 * j] - 1.0).abs() < 1e-6); // cos 0
            assert!(f[2 * j + 1].abs() < 1e-6); // sin 0
        }
    }

    #[test]
    fn axis_selection_matches_upstream() {
        let s = [24, 20, 20];
        // h slots: 1,4,..,58 ; w slots: 2,5,..,59 ; everything else t.
        for j in 0..128 {
            let a = axis_for_index(j, s);
            let expect = if j % 3 == 1 && j < 60 {
                1
            } else if j % 3 == 2 && j < 60 {
                2
            } else {
                0
            };
            assert_eq!(a, expect, "index {j}");
        }
        // Index 60+ is always t (h/w spans end at 60).
        assert_eq!(axis_for_index(61, s), 0);
        assert_eq!(axis_for_index(62, s), 0);
    }

    #[test]
    fn text_positions_reduce_to_1d_rope() {
        // When (t,h,w) are all equal (text tokens), MRoPE collapses to plain 1D
        // RoPE over head_dim -> must equal RopeEmbedder([head_dim,0,0]).
        let head_dim = 256;
        let theta = 5_000_000.0;
        let rope = RopeEmbedder::new(theta, [head_dim, 0, 0], [64, 1, 1]);
        for p in [0_i64, 1, 7, 33] {
            let mine = build_freqs(&[p, p, p], head_dim, theta, config::MROPE_SECTION);
            let theirs = rope.lookup(&[p as i32, 0, 0]);
            assert_eq!(mine.len(), theirs.len());
            for (a, b) in mine.iter().zip(theirs.iter()) {
                assert!((a - b).abs() < 1e-5, "p={p}: {a} vs {b}");
            }
        }
    }
}
