//! 3-axis rotary embedding over the `(t, h, w)` coordinates of the packed
//! sequence.
//!
//! One `inv_freq` buffer of `rope_inv_freq_len` frequencies is shared by the
//! three axes. Each axis contributes that many angles; the three blocks are
//! concatenated to `3 * rope_inv_freq_len` (48) and then concatenated with
//! themselves, so the rotate-half convention rotates `2 * 3 * inv_freq_len` (96)
//! of each 128-channel head and passes the remaining 32 through.
//!
//! Because the angle block is duplicated, `cos`/`sin` for pair `k` and pair
//! `k + 48` are equal, which is exactly what
//! [`RopeF32HalfRotPartial`](thinfer_core::ops::RopeF32HalfRotPartial) assumes:
//! it indexes `k % rot_pairs`. So this builder emits only the 48 distinct
//! `(cos, sin)` entries per row, not all 96.
//!
//! NB the positions are FLOAT, not integer token indices: the spatial axes are
//! aspect-normalized onto `[0, 32)` and the temporal axis runs on the shared
//! 40-units-per-second audio/video clock. See [`super::packing`].

use super::DitConfig;

/// `(cos, sin)` entries per row: one per distinct rotary pair.
pub fn pairs_per_row(cfg: &DitConfig) -> usize {
    3 * cfg.rope_inv_freq_len
}

/// Width of one freqs row in f32 elements (`pairs * 2`). This is the value the
/// partial half-rot kernel reads as its rotary width.
pub fn freqs_row_width(cfg: &DitConfig) -> usize {
    2 * pairs_per_row(cfg)
}

/// `inv_freq[j] = 1 / theta^(2j / (2n))` for `j` in `0..n`.
///
/// Computed in f64: the exponent is a ratio of small integers and the f32 path
/// loses low bits that shift every angle at large positions.
fn inv_freq(cfg: &DitConfig) -> Vec<f64> {
    let n = cfg.rope_inv_freq_len;
    let theta = cfg.rope_theta as f64;
    (0..n)
        .map(|j| 1.0 / theta.powf(2.0 * j as f64 / (2 * n) as f64))
        .collect()
}

/// Build `[seq, 2 * 3 * inv_freq_len]` freqs, `(cos, sin)` interleaved per pair.
///
/// Pair `k` draws its angle from axis `k / inv_freq_len` at frequency
/// `k % inv_freq_len`, matching the reference's per-axis concatenation order
/// `[t block | h block | w block]`.
pub fn build_freqs(cfg: &DitConfig, position_ids: &[[f32; 3]]) -> Vec<f32> {
    let inv = inv_freq(cfg);
    let n = cfg.rope_inv_freq_len;
    let width = freqs_row_width(cfg);
    let mut out = vec![0.0f32; position_ids.len() * width];
    for (row, pos) in position_ids.iter().enumerate() {
        for (axis, &coord) in pos.iter().enumerate() {
            let p = coord as f64;
            for (j, &f) in inv.iter().enumerate() {
                let angle = p * f;
                let k = axis * n + j;
                out[row * width + 2 * k] = angle.cos() as f32;
                out[row * width + 2 * k + 1] = angle.sin() as f32;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> DitConfig {
        DitConfig::h3_33b()
    }

    #[test]
    fn row_width_matches_the_rotary_dim() {
        let c = cfg();
        assert_eq!(pairs_per_row(&c), 48);
        // The kernel's rotary width is the freqs row width; it must equal the
        // config's rotary_dim or the rotated channel count silently disagrees.
        assert_eq!(freqs_row_width(&c), c.rotary_dim());
        assert_eq!(freqs_row_width(&c), 96);
    }

    /// The packed kernels require a word-aligned rotary block.
    #[test]
    fn rotary_width_is_word_aligned() {
        assert_eq!(freqs_row_width(&cfg()) % 4, 0);
        assert_eq!(pairs_per_row(&cfg()) % 2, 0);
    }

    #[test]
    fn origin_is_the_identity_rotation() {
        let c = cfg();
        let f = build_freqs(&c, &[[0.0, 0.0, 0.0]]);
        assert_eq!(f.len(), freqs_row_width(&c));
        for k in 0..pairs_per_row(&c) {
            assert_eq!(f[2 * k], 1.0, "cos at pair {k}");
            assert_eq!(f[2 * k + 1], 0.0, "sin at pair {k}");
        }
    }

    #[test]
    fn first_frequency_is_unscaled() {
        let c = cfg();
        // j = 0 -> inv_freq = 1, so the angle is the coordinate itself.
        let f = build_freqs(&c, &[[1.0, 0.0, 0.0]]);
        assert!((f[0] as f64 - 1.0f64.cos()).abs() < 1e-6);
        assert!((f[1] as f64 - 1.0f64.sin()).abs() < 1e-6);
    }

    /// Each axis must drive only its own 16-frequency block. Crossing them is
    /// silent and would misalign audio against video.
    #[test]
    fn axes_drive_disjoint_frequency_blocks() {
        let c = cfg();
        let n = c.rope_inv_freq_len;
        let base = build_freqs(&c, &[[0.0, 0.0, 0.0]]);
        for (axis, moved) in [
            (0usize, [1.0f32, 0.0, 0.0]),
            (1, [0.0, 1.0, 0.0]),
            (2, [0.0, 0.0, 1.0]),
        ] {
            let f = build_freqs(&c, &[moved]);
            for k in 0..pairs_per_row(&c) {
                let touched = f[2 * k] != base[2 * k] || f[2 * k + 1] != base[2 * k + 1];
                let in_block = k / n == axis;
                assert_eq!(
                    touched,
                    in_block,
                    "axis {axis} moved pair {k} (block {})",
                    k / n
                );
            }
        }
    }

    /// Frequencies must decay along each block, or the high-frequency end is
    /// wired backwards.
    #[test]
    fn frequencies_decay_within_a_block() {
        let inv = inv_freq(&cfg());
        assert_eq!(inv.len(), 16);
        assert_eq!(inv[0], 1.0);
        assert!(inv.windows(2).all(|w| w[1] < w[0]));
        assert!(*inv.last().unwrap() > 0.0);
    }

    #[test]
    fn rows_are_independent() {
        let c = cfg();
        let pos = [[1.0f32, 2.0, 3.0], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0]];
        let all = build_freqs(&c, &pos);
        let w = freqs_row_width(&c);
        for (i, p) in pos.iter().enumerate() {
            let one = build_freqs(&c, std::slice::from_ref(p));
            assert_eq!(&all[i * w..(i + 1) * w], &one[..], "row {i}");
        }
    }

    /// cos^2 + sin^2 = 1 for every entry, at the large coordinates the video
    /// clock actually reaches (a 15s clip runs past t = 600).
    #[test]
    fn entries_stay_on_the_unit_circle_at_large_positions() {
        let c = cfg();
        let f = build_freqs(&c, &[[900.0, 31.5, 31.5]]);
        for k in 0..pairs_per_row(&c) {
            let (co, si) = (f[2 * k] as f64, f[2 * k + 1] as f64);
            assert!((co * co + si * si - 1.0).abs() < 1e-5, "pair {k}");
        }
    }
}
