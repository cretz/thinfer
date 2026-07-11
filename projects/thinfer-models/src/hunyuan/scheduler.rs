//! HunyuanVideo 1.5 flow-match Euler scheduler (lightx2v 4-step T2V distill).
//!
//! Rectified-flow / flow-match: the DiT predicts a velocity `v`, integrated by
//! plain Euler `x += (sigma_{i+1} - sigma_i) * v` with a terminal sigma of 0.
//! CFG is OFF (single forward per step, no negative prompt). The lightx2v distill
//! is single-timestep (NO MeanFlow `timestep_r`).
//!
//! Schedule (from `lightx2v/.../step_distill/scheduler.py`):
//!   sigmas_full[k] = 1 - k/1000,  k in 0..1000            (linspace(1,0,1001)[:-1])
//!   sigmas_full = S*sigmas_full / (1 + (S-1)*sigmas_full) (SD3/flux shift, S=shift)
//!   idx[i]      = 1000 - denoising_step_list[i]
//!   sigmas[i]   = sigmas_full[idx[i]]                     (+ terminal 0 appended)
//!   timesteps[i]= sigmas[i] * 1000                        (fed to the DiT time embed)
//!
//! `denoising_step_list` values are step labels, NOT literal timesteps; the index
//! into the 1000-point grid is `1000 - label`.

use crate::hunyuan::config::sampling;

/// A resolved denoise schedule: `timesteps[i]` is the model-time input for step
/// `i`; `sigmas` has `steps + 1` entries (terminal sigma 0 appended) so each
/// step's Euler delta is `dt(i) = sigmas[i+1] - sigmas[i]` (negative).
#[derive(Clone, Debug)]
pub struct FlowMatchSchedule {
    /// Per-step model timestep (sigma_shifted * 1000), length = steps.
    pub timesteps: Vec<f32>,
    /// Shifted sigmas, length = steps + 1 (last entry is the terminal 0).
    pub sigmas: Vec<f32>,
}

impl FlowMatchSchedule {
    /// Build from explicit step labels + shift. `train_steps` is the sigma-grid
    /// resolution (1000). Panics if a label is out of `1..=train_steps`.
    pub fn build(denoising_step_list: &[u32], shift: f32, train_steps: u32) -> Self {
        let shifted = |sigma: f32| -> f32 {
            // sigma' = S*sigma / (1 + (S-1)*sigma)
            shift * sigma / (1.0 + (shift - 1.0) * sigma)
        };
        let sigma_full = |k: u32| -> f32 {
            // sigmas_full[k] = 1 - k/train_steps, then shifted.
            shifted(1.0 - (k as f32) / (train_steps as f32))
        };

        let mut timesteps = Vec::with_capacity(denoising_step_list.len());
        let mut sigmas = Vec::with_capacity(denoising_step_list.len() + 1);
        for &label in denoising_step_list {
            assert!(
                label >= 1 && label <= train_steps,
                "denoising step label {label} out of 1..={train_steps}"
            );
            let idx = train_steps - label; // index into the 1000-point grid
            let s = sigma_full(idx);
            sigmas.push(s);
            timesteps.push(s * 1000.0);
        }
        sigmas.push(0.0); // terminal sigma for the last Euler step
        Self { timesteps, sigmas }
    }

    /// The lightx2v 4-step 480p T2V distill schedule (shift 9.0 default; 5.0 is
    /// the A/B fallback per `sampling::FLOW_SHIFT_FALLBACK`).
    pub fn lightx2v_t2v_480p() -> Self {
        Self::build(&sampling::DENOISING_STEP_LIST, sampling::FLOW_SHIFT, 1000)
    }

    pub fn steps(&self) -> usize {
        self.timesteps.len()
    }

    /// Euler delta for step `i`: `sigmas[i+1] - sigmas[i]` (negative). Apply as
    /// `x += dt * v` on the predicted velocity.
    pub fn dt(&self, i: usize) -> f32 {
        self.sigmas[i + 1] - self.sigmas[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        // Relative tolerance: timesteps are ~1000-scale, sigmas ~1-scale.
        let tol = 1e-4 * b.abs().max(1.0);
        assert!((a - b).abs() <= tol, "expected {b}, got {a}");
    }

    #[test]
    fn lightx2v_4step_shift9_schedule() {
        let s = FlowMatchSchedule::lightx2v_t2v_480p();
        assert_eq!(s.steps(), 4);
        // pre-shift sigmas at idx [0,250,500,750] = [1.0, 0.75, 0.5, 0.25];
        // shift 9 -> [1.0, 0.96429, 0.9, 0.75].
        approx(s.sigmas[0], 1.0);
        approx(s.sigmas[1], 0.96429);
        approx(s.sigmas[2], 0.9);
        approx(s.sigmas[3], 0.75);
        approx(s.sigmas[4], 0.0); // terminal
        // timesteps = sigma * 1000.
        approx(s.timesteps[0], 1000.0);
        approx(s.timesteps[1], 964.29);
        approx(s.timesteps[2], 900.0);
        approx(s.timesteps[3], 750.0);
        // Euler dt (negative), terminal step jumps the remaining 0.75 -> 0.
        approx(s.dt(0), -0.03571);
        approx(s.dt(1), -0.06429);
        approx(s.dt(2), -0.15);
        approx(s.dt(3), -0.75);
    }

    #[test]
    fn shift5_fallback_matches_reference() {
        let s = FlowMatchSchedule::build(&sampling::DENOISING_STEP_LIST, 5.0, 1000);
        // shift 5 -> sigmas [1.0, 0.9375, 0.83333, 0.625], timesteps *1000.
        approx(s.sigmas[1], 0.9375);
        approx(s.sigmas[2], 0.83333);
        approx(s.sigmas[3], 0.625);
        approx(s.timesteps[1], 937.5);
    }
}
