//! CPU 3-axis RoPE freq table + per-token lookup.
//!
//! Mirrors `RopeEmbedder` in `src/zimage/transformer.py`. Precompute is
//! `precompute_freqs_cis(axes_dims, axes_lens, theta)` — one table per axis,
//! shape `[axes_lens[i], axes_dims[i]/2]` complex. `__call__(ids[seq, 3])`
//! looks up `freqs_cis[i][ids[:, i]]` per axis and concats along the last
//! dim, producing `[seq, head_dim/2]` complex.
//!
//! Output layout for our rope kernel: `[seq, head_dim]` interleaved
//! `(re, im, re, im, ...)` floats. Total bytes: `seq * head_dim * 4`.
//!
//! Precompute math (per axis `d`, length `e`):
//!   `freqs = 1 / theta.powf(arange(0, d, 2) / d)`   // [d/2]
//!   `args  = arange(e).outer(freqs)`                 // [e, d/2]
//!   `cis   = (cos(args), sin(args))`
//! Upstream does the computation in f64 then casts to f32, which we mirror to
//! keep numerics aligned for differential tests.

#[derive(Clone, Debug)]
pub struct RopeEmbedder {
    pub theta: f32,
    pub axes_dims: [usize; 3],
    pub axes_lens: [usize; 3],
    /// Per-axis cis tables. `tables[i]` is `axes_lens[i]` rows of
    /// `axes_dims[i]/2` (re, im) pairs flattened: `len = axes_lens[i] * axes_dims[i]` f32s.
    tables: [Vec<f32>; 3],
}

impl RopeEmbedder {
    pub fn new(theta: f32, axes_dims: [usize; 3], axes_lens: [usize; 3]) -> Self {
        let theta_f64 = theta as f64;
        let tables = std::array::from_fn(|i| {
            let d = axes_dims[i];
            let e = axes_lens[i];
            assert!(d.is_multiple_of(2), "rope axis dim must be even");
            let half = d / 2;
            // freqs[j] = 1 / theta^(2j / d)
            let freqs: Vec<f64> = (0..half)
                .map(|j| 1.0 / theta_f64.powf((2 * j) as f64 / d as f64))
                .collect();
            let mut buf = vec![0.0_f32; e * d];
            for t in 0..e {
                for (j, &f) in freqs.iter().enumerate() {
                    let arg = (t as f64) * f;
                    buf[t * d + 2 * j] = arg.cos() as f32;
                    buf[t * d + 2 * j + 1] = arg.sin() as f32;
                }
            }
            buf
        });
        Self {
            theta,
            axes_dims,
            axes_lens,
            tables,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.axes_dims.iter().sum()
    }

    /// `ids[seq * 3]` row-major; returns `[seq, head_dim]` interleaved (re, im).
    pub fn lookup(&self, ids: &[i32]) -> Vec<f32> {
        assert!(ids.len().is_multiple_of(3), "ids must be [seq, 3]");
        let seq = ids.len() / 3;
        let hd = self.head_dim();
        let mut out = vec![0.0_f32; seq * hd];
        for row in 0..seq {
            let mut dst_off = row * hd;
            for axis in 0..3 {
                let coord = ids[row * 3 + axis];
                let d = self.axes_dims[axis];
                debug_assert!(
                    (coord as usize) < self.axes_lens[axis],
                    "rope coord {} >= axes_lens[{}]={} (row {row})",
                    coord,
                    axis,
                    self.axes_lens[axis]
                );
                let src_off = (coord as usize) * d;
                out[dst_off..dst_off + d].copy_from_slice(&self.tables[axis][src_off..src_off + d]);
                dst_off += d;
            }
        }
        out
    }

    #[cfg(test)]
    fn table(&self, axis: usize) -> &[f32] {
        &self.tables[axis]
    }

    /// Same as `lookup` but returns raw bytes ready for `writeBuffer`.
    pub fn lookup_bytes(&self, ids: &[i32]) -> Vec<u8> {
        let f = self.lookup(ids);
        let mut bytes = vec![0u8; f.len() * 4];
        for (i, v) in f.iter().enumerate() {
            bytes[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() <= tol, "want {b}, got {a} (tol {tol})");
    }

    #[test]
    fn coord_zero_is_identity() {
        // At coord=0, args=0, so every (re, im) is (1, 0).
        let r = RopeEmbedder::new(10_000.0, [4, 0, 0], [8, 1, 1]);
        let t = r.table(0);
        for j in 0..2 {
            approx(t[2 * j], 1.0, 1e-6);
            approx(t[2 * j + 1], 0.0, 1e-6);
        }
    }

    #[test]
    fn first_axis_known_value() {
        // axes_dims=[2], so half=1, freqs=[1/theta^0]=[1]. At coord=t, the
        // single (re, im) pair is (cos(t), sin(t)).
        let r = RopeEmbedder::new(10_000.0, [2, 0, 0], [4, 1, 1]);
        let t = r.table(0);
        // Each coord occupies 2 floats (re, im); coord=k starts at index 2*k.
        approx(t[2], 1.0_f32.cos(), 1e-6); // coord=1
        approx(t[3], 1.0_f32.sin(), 1e-6);
        approx(t[4], 2.0_f32.cos(), 1e-6); // coord=2
        approx(t[5], 2.0_f32.sin(), 1e-6);
    }

    #[test]
    fn lookup_concats_axes() {
        // 3 axes, each axes_dims=2, axes_lens=4 -> head_dim=6 per token.
        let r = RopeEmbedder::new(10_000.0, [2, 2, 2], [4, 4, 4]);
        // Single token at coord (1, 2, 3).
        let ids = vec![1_i32, 2, 3];
        let out = r.lookup(&ids);
        assert_eq!(out.len(), 6);
        // First 2 floats from axis-0 coord 1, next 2 from axis-1 coord 2, last
        // 2 from axis-2 coord 3. Each axis has 1 freq=1, so (cos(c), sin(c)).
        approx(out[0], 1.0_f32.cos(), 1e-6);
        approx(out[1], 1.0_f32.sin(), 1e-6);
        approx(out[2], 2.0_f32.cos(), 1e-6);
        approx(out[3], 2.0_f32.sin(), 1e-6);
        approx(out[4], 3.0_f32.cos(), 1e-6);
        approx(out[5], 3.0_f32.sin(), 1e-6);
    }

    #[test]
    fn lookup_bytes_matches_lookup() {
        let r = RopeEmbedder::new(10_000.0, [4, 2, 2], [4, 4, 4]);
        let ids = vec![0_i32, 1, 2, 1, 0, 3];
        let f = r.lookup(&ids);
        let b = r.lookup_bytes(&ids);
        assert_eq!(b.len(), f.len() * 4);
        for (i, v) in f.iter().enumerate() {
            let got = f32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]);
            approx(got, *v, 0.0);
        }
    }
}
