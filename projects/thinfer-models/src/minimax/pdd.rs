//! PDD output heads (FastH3-Omni): the final layer's `video_out` / `audio_out`
//! stack `heads` heads trained on a fine sigma grid, and each of the `steps`
//! sampling steps runs ONE head fused from its stretch of that grid
//! (FastVideo `fastvideo/layers/pdd.py`, Comfy `MiniMaxH3` fused head).
//!
//! The Comfy file stores `head_0` whole and every other head as an offset
//! from it (`head_j = rows_0 + rows_j`), so a fused head is `rows_0 + sum_j
//! w_j rows_j` with `sum w_j = 1` over the step's block.
//!
//! The sampler's sigmas are the shifted fine grid at the block boundaries,
//! scaled by FastVideo's `max_t` (the trained grid starts at 0.999, not 1).

/// Heads in the file and the steps a run takes (the contract's 32 / 8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pdd {
    pub heads: usize,
    pub steps: usize,
}

/// FastVideo's `max_t`: the fine grid runs `0.999 -> 0`.
pub const MAX_SIGMA: f64 = 0.999;

/// The flow shift `u -> s u / (1 + (s - 1) u)` on the unit grid.
fn shifted(u: f64, shift: f64) -> f64 {
    shift * u / (1.0 + (shift - 1.0) * u)
}

impl Pdd {
    /// Heads step `step` fuses.
    pub fn block(&self, step: usize) -> std::ops::Range<usize> {
        let per = self.heads / self.steps;
        step * per..(step + 1) * per
    }

    /// `(head, weight)` for step `step`, weights normalized to 1: each head's
    /// share of the shifted sigma drop across the block (independent of
    /// `max_t`, which scales every drop alike).
    pub fn mix(&self, step: usize, shift: f64) -> Vec<(usize, f64)> {
        let n = self.heads as f64;
        let drop = |j: usize| {
            shifted(1.0 - j as f64 / n, shift) - shifted(1.0 - (j + 1) as f64 / n, shift)
        };
        let block = self.block(step);
        let total: f64 = block.clone().map(drop).sum();
        block.map(|j| (j, drop(j) / total)).collect()
    }

    /// The `steps + 1` sampler sigmas, ending at 0.
    pub fn sigmas(&self, shift: f64) -> Vec<f32> {
        (0..self.steps)
            .map(|k| (MAX_SIGMA * shifted(1.0 - k as f64 / self.steps as f64, shift)) as f32)
            .chain([0.0])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OMNI: Pdd = Pdd {
        heads: 32,
        steps: 8,
    };

    fn close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 2e-6, "{a:?} vs {b:?}");
        }
    }

    /// The normalized block weights the reference computes (format notes).
    #[test]
    fn mixes_match_the_reference() {
        let w = |step, shift| {
            OMNI.mix(step, shift)
                .iter()
                .map(|&(_, w)| w)
                .collect::<Vec<_>>()
        };
        close(&w(0, 12.0), &[0.227882, 0.241731, 0.256882, 0.273504]);
        close(&w(7, 12.0), &[0.123077, 0.173219, 0.261843, 0.441860]);
        close(&w(0, 3.0), &[0.234043, 0.244218, 0.255072, 0.266667]);
        close(&w(7, 3.0), &[0.210526, 0.233918, 0.261438, 0.294118]);
        assert_eq!(OMNI.mix(7, 12.0)[0].0, 28);
    }

    #[test]
    fn sigmas_match_fastvideo() {
        let v: Vec<f64> = OMNI.sigmas(12.0).iter().map(|&s| s as f64).collect();
        close(
            &v,
            &[
                0.999, 0.987247, 0.972, 0.951429, 0.922154, 0.877171, 0.7992, 0.630947, 0.0,
            ],
        );
        let a: Vec<f64> = OMNI.sigmas(3.0).iter().map(|&s| s as f64).collect();
        close(
            &a,
            &[
                0.999, 0.953591, 0.8991, 0.8325, 0.74925, 0.642214, 0.4995, 0.2997, 0.0,
            ],
        );
    }
}
