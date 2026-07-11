//! Krea 2 Turbo scheduler: `FlowMatchEulerDiscreteScheduler`, CFG-free.
//!
//! `sigmas = linspace(1, 1/N, N) ++ 0`; each sigma is time-shifted by
//! `shifted = e^mu / (e^mu + (1/sigma - 1))`. The Turbo (distilled) checkpoint
//! uses a FIXED `mu = 1.15` and `N = 8` steps (the Raw/Base checkpoint instead
//! computes `mu` from the image sequence length via `calculate_shift`). One
//! Euler step is `z += (sigma_next - sigma) * velocity`.
//!
//! FLAG (worklog): confirm `mu`/step-count against the shipped
//! `scheduler_config.json` when the safetensors bundle is available; the value
//! here matches the stable-diffusion.cpp reference for Turbo.

/// Fixed time-shift for the distilled Turbo checkpoint.
pub const TURBO_MU: f64 = 1.15;
/// Default inference steps for Turbo.
pub const TURBO_STEPS: usize = 8;

/// One FlowMatchEuler step: `t` is the DiT timestep (sigma in `[0,1]`),
/// `delta = sigma_next - sigma` (the Euler coefficient for `z += delta * v`).
#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub t: f32,
    pub delta: f32,
}

/// Build the `num_steps` FlowMatchEuler schedule with a fixed shift `mu`.
pub fn build_steps(num_steps: usize, mu: f64) -> Vec<Step> {
    let emu = mu.exp();
    let shift = |sigma: f64| -> f64 {
        if sigma <= 0.0 {
            0.0
        } else {
            emu / (emu + (1.0 / sigma - 1.0))
        }
    };
    let n = num_steps.max(1);
    // linspace(1.0, 1/n, n) then append 0.0.
    let mut shifted: Vec<f64> = (0..n)
        .map(|i| {
            let sigma = 1.0 + (1.0 / n as f64 - 1.0) * (i as f64 / (n - 1).max(1) as f64);
            shift(sigma)
        })
        .collect();
    shifted.push(0.0);
    (0..n)
        .map(|i| Step {
            t: shifted[i] as f32,
            delta: (shifted[i + 1] - shifted[i]) as f32,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turbo_schedule_descends_to_zero() {
        let s = build_steps(TURBO_STEPS, TURBO_MU);
        assert_eq!(s.len(), TURBO_STEPS);
        // First sigma is 1.0 (pure noise); shift(1.0) == 1.0.
        assert!((s[0].t - 1.0).abs() < 1e-5, "t0={}", s[0].t);
        // Cumulative deltas drive sigma from 1 -> 0.
        let final_sigma = s[0].t + s.iter().map(|x| x.delta).sum::<f32>();
        assert!(final_sigma.abs() < 1e-5, "final sigma {final_sigma}");
        // Monotonic descending t.
        for w in s.windows(2) {
            assert!(w[1].t < w[0].t, "not descending: {:?}", w);
        }
    }
}
