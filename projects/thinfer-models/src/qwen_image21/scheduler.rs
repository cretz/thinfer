//! Qwen-Image-2.1-Turbo sampling: FlowMatchEuler over the checkpoint's own
//! 8-sigma grid (`model_index.json` `sample_sigmas`). The scheduler config has
//! `shift = 1` and no dynamic shifting, so the grid is used verbatim; a
//! terminal 0 is appended. The DiT reads `t = sigma` (in `[0, 1]`).
//!
//! Qwen ships no other schedule for this checkpoint ("other schedules have not
//! been evaluated"), so the step count is fixed.

/// `sample_sigmas` from the Turbo checkpoint, excluding the terminal 0.
pub const TURBO_SIGMAS: [f32; 8] = [
    1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568,
];

/// One Euler step: the DiT timestep and `sigma_next - sigma`.
#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub t: f32,
    pub delta: f32,
}

pub fn turbo_steps() -> Vec<Step> {
    let n = TURBO_SIGMAS.len();
    (0..n)
        .map(|i| {
            let next = if i + 1 < n { TURBO_SIGMAS[i + 1] } else { 0.0 };
            Step {
                t: TURBO_SIGMAS[i],
                delta: next - TURBO_SIGMAS[i],
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_integrate_to_zero() {
        let s = turbo_steps();
        assert_eq!(s.len(), 8);
        let end = s[0].t + s.iter().map(|x| x.delta).sum::<f32>();
        assert!(end.abs() < 1e-6, "{end}");
    }
}
