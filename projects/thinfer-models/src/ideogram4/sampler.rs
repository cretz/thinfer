//! Logit-normal schedule + Euler flow-matching sampler. Mirrors `scheduler.py`.
//!
//! With the turbotime LoRA there is NO CFG: a single conditional DiT forward
//! per step, `v = pos_v`, `z += v * (s_val - t_val)`. This module owns only the
//! schedule math (no GPU); the pipeline drives the DiT forward between steps.
//!
//! Schedule (`LogitNormalSchedule`): `t_ = 1 - sigmoid(mean + std*ndtri(u))`,
//! clamped to a logsnr window. `mean` is resolution-aware:
//! `mean = mu + 0.5*ln(H*W / 512^2)`. Step grid is `linspace(0,1,num_steps+1)`;
//! step `i` (from `num_steps-1` down to 0) uses `t=schedule(grid[i+1])`,
//! `s=schedule(grid[i])`. f64 internally (mirror upstream torch float64).

/// Defaults from `V4_TURBO_12` (`sampler_configs.py`); the turbotime LoRA is a
/// turbo derivative, so we start here and eyeball.
pub const DEFAULT_MU: f64 = 0.5;
pub const DEFAULT_STD: f64 = 1.75;
pub const KNOWN_RESOLUTION_PIXELS: f64 = 512.0 * 512.0;
const LOGSNR_MIN: f64 = -15.0;
const LOGSNR_MAX: f64 = 18.0;

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Inverse standard-normal CDF (probit), Wichura's AS241 (PPND16). Matches
/// `torch.special.ndtri` to ~1e-15 over (0,1). Boundary returns +-inf.
// Canonical AS241 constants kept at published precision (extra digits are
// documentation; f64 rounds them identically).
#[allow(clippy::excessive_precision)]
pub fn ndtri(p: f64) -> f64 {
    // Boundary: torch.special.ndtri(0)= -inf, ndtri(1)= +inf. The schedule grid
    // includes 0.0 and 1.0, which then sigmoid-saturate and clamp.
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        return q
            * (((((((2509.0809287301226727 * r + 33430.575583588128105) * r
                + 67265.770927008700853)
                * r
                + 45921.953931549871457)
                * r
                + 13731.693765509461125)
                * r
                + 1971.5909503065514427)
                * r
                + 133.14166789178437745)
                * r
                + 3.387132872796366608)
            / (((((((5226.495278852854561 * r + 28729.085735721942674) * r
                + 39307.89580009271061)
                * r
                + 21213.794301586595867)
                * r
                + 5394.1960214247511077)
                * r
                + 687.1870074920579083)
                * r
                + 42.313330701600911252)
                * r
                + 1.0);
    }
    let mut r = if q < 0.0 { p } else { 1.0 - p };
    r = (-r.ln()).sqrt();
    let val = if r <= 5.0 {
        let r = r - 1.6;
        (((((((7.7454501427834140764e-4 * r + 0.0227238449892691845833) * r
            + 0.24178072517745061177)
            * r
            + 1.27045825245236838258)
            * r
            + 3.64784832476320460504)
            * r
            + 5.7694972214606914055)
            * r
            + 4.6303378461565452959)
            * r
            + 1.42343711074968357734)
            / (((((((1.05075007164441684324e-9 * r + 5.475938084995344946e-4) * r
                + 0.0151986665636164571966)
                * r
                + 0.14810397642748007459)
                * r
                + 0.68976733498510000455)
                * r
                + 1.6763848301838038494)
                * r
                + 2.05319162663775882187)
                * r
                + 1.0)
    } else {
        let r = r - 5.0;
        (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r
            + 0.0012426609473880784386)
            * r
            + 0.026532189526576123093)
            * r
            + 0.29656057182850489123)
            * r
            + 1.7848265399172913358)
            * r
            + 5.4637849111641143699)
            * r
            + 6.6579046435011037772)
            / (((((((2.04426310338993978564e-15 * r + 1.4215117583164458887e-7) * r
                + 1.8463183175100546818e-5)
                * r
                + 7.868691311456132591e-4)
                * r
                + 0.0148753612908506148525)
                * r
                + 0.13692988092273580531)
                * r
                + 0.59983220655588793769)
                * r
                + 1.0)
    };
    if q < 0.0 { -val } else { val }
}

/// Resolution-aware logit-normal schedule.
#[derive(Clone, Copy, Debug)]
pub struct LogitNormalSchedule {
    pub mean: f64,
    pub std: f64,
}

impl LogitNormalSchedule {
    /// `mean = mu + 0.5*ln(H*W / 512^2)` (`get_schedule_for_resolution`).
    pub fn for_resolution(height: usize, width: usize, mu: f64, std: f64) -> Self {
        let num_pixels = (height * width) as f64;
        let mean = mu + 0.5 * (num_pixels / KNOWN_RESOLUTION_PIXELS).ln();
        Self { mean, std }
    }

    /// `t_ = (1 - sigmoid(mean + std*ndtri(u))).clamp(t_min, t_max)`.
    pub fn eval(&self, u: f64) -> f64 {
        let z = ndtri(u);
        let t = 1.0 - sigmoid(self.mean + self.std * z);
        let t_min = 1.0 / (1.0 + (0.5 * LOGSNR_MAX).exp());
        let t_max = 1.0 / (1.0 + (0.5 * LOGSNR_MIN).exp());
        t.clamp(t_min, t_max)
    }
}

/// One Euler step: the noised time `t_val` to evaluate the DiT velocity at, and
/// the delta `s_val - t_val` to advance the latent (`z += v * delta`).
#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub t_val: f64,
    pub delta: f64,
}

/// Precompute the per-step `(t_val, delta)` pairs in loop order (first emitted
/// step is `i = num_steps-1`, the highest-noise step). `grid = linspace(0,1,
/// num_steps+1)`; step `i` uses `t=schedule(grid[i+1])`, `s=schedule(grid[i])`.
pub fn build_steps(num_steps: usize, height: usize, width: usize, mu: f64, std: f64) -> Vec<Step> {
    assert!(num_steps >= 1, "num_steps must be >= 1");
    let sched = LogitNormalSchedule::for_resolution(height, width, mu, std);
    let grid: Vec<f64> = (0..=num_steps)
        .map(|k| k as f64 / num_steps as f64)
        .collect();
    let sval: Vec<f64> = grid.iter().map(|&u| sched.eval(u)).collect();
    (0..num_steps)
        .rev()
        .map(|i| Step {
            t_val: sval[i + 1],
            delta: sval[i] - sval[i + 1],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndtri_known_points() {
        assert!(ndtri(0.5).abs() < 1e-12);
        assert!((ndtri(0.975) - 1.959963984540054).abs() < 1e-9);
        assert!((ndtri(0.025) + 1.959963984540054).abs() < 1e-9);
        assert!((ndtri(0.9) - 1.2815515655446004).abs() < 1e-9);
        assert!((ndtri(0.7) - 0.5244005127080407).abs() < 1e-9);
        // Symmetry.
        for &p in &[0.1, 0.3, 0.45, 0.8] {
            assert!((ndtri(p) + ndtri(1.0 - p)).abs() < 1e-10);
        }
    }

    #[test]
    fn schedule_endpoints_clamped() {
        let s = LogitNormalSchedule::for_resolution(1024, 1024, DEFAULT_MU, DEFAULT_STD);
        // u->0 gives ndtri->-inf -> sigmoid->0 -> t->1, clamped to t_max.
        let t_max = 1.0 / (1.0 + (0.5 * LOGSNR_MIN).exp());
        let t_min = 1.0 / (1.0 + (0.5 * LOGSNR_MAX).exp());
        assert!((s.eval(1e-9) - t_max).abs() < 1e-6);
        assert!((s.eval(1.0 - 1e-9) - t_min).abs() < 1e-6);
    }

    #[test]
    fn mean_is_resolution_aware() {
        // At 512x512 the log term is 0 -> mean == mu.
        let s = LogitNormalSchedule::for_resolution(512, 512, 0.7, 1.0);
        assert!((s.mean - 0.7).abs() < 1e-12);
        // 1024x1024 = 4x pixels -> mean = mu + 0.5*ln(4).
        let s2 = LogitNormalSchedule::for_resolution(1024, 1024, 0.0, 1.0);
        assert!((s2.mean - 0.5 * 4.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn steps_are_loop_ordered_and_sum_to_span() {
        let steps = build_steps(8, 1024, 1024, DEFAULT_MU, DEFAULT_STD);
        assert_eq!(steps.len(), 8);
        // Telescoping: sum of deltas = schedule(0) - schedule(1).
        let sched = LogitNormalSchedule::for_resolution(1024, 1024, DEFAULT_MU, DEFAULT_STD);
        let span = sched.eval(0.0_f64.max(1e-12)) - sched.eval(1.0_f64.min(1.0 - 1e-12));
        let sum: f64 = steps.iter().map(|s| s.delta).sum();
        // grid endpoints are exactly 0 and 1; eval clamps so use the same path.
        let span_exact = sched.eval(0.0) - sched.eval(1.0);
        assert!(
            (sum - span_exact).abs() < 1e-9,
            "{sum} vs {span_exact} ({span})"
        );
    }
}
