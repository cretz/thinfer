//! FlowMatchEulerDiscreteScheduler, Z-Image-Turbo config.
//!
//! Mirrors `diffusers.FlowMatchEulerDiscreteScheduler` with the config baked
//! by `Z-Image/src/zimage/pipeline.py`:
//! - `num_train_timesteps = 1000`, `use_dynamic_shifting = True`
//! - `base_image_seq_len = 256`, `max_image_seq_len = 4096`
//! - `base_shift = 0.5`, `max_shift = 1.15`, `time_shift_type = "exponential"`
//! - pipeline override: `scheduler.sigma_min = 0.0`
//!
//! No karras/exponential/beta sigma paths, no `shift_terminal`, no
//! `invert_sigmas`, no `stochastic_sampling`, no `per_token_timesteps`.
//! Stdlib-only; CPU; runs once per step between DiT submits.

const BASE_SEQ_LEN: f32 = 256.0;
const MAX_SEQ_LEN: f32 = 4096.0;
const BASE_SHIFT: f32 = 0.5;
const MAX_SHIFT: f32 = 1.15;

/// `calculate_shift` (`pipeline.py`): linear interp of base/max shift over the
/// image sequence length.
pub fn calculate_shift(image_seq_len: usize) -> f32 {
    let m = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ_LEN - BASE_SEQ_LEN);
    let b = BASE_SHIFT - m * BASE_SEQ_LEN;
    image_seq_len as f32 * m + b
}

/// Exponential time shift used by the scheduler when `use_dynamic_shifting`.
/// `sigma` is the shifting exponent (always 1.0 for our path).
fn time_shift_exponential(mu: f32, sigma_exp: f32, t: f32) -> f32 {
    let e_mu = mu.exp();
    e_mu / (e_mu + (1.0 / t - 1.0).powf(sigma_exp))
}

pub struct FlowMatchEulerScheduler {
    /// Length `num_inference_steps + 1`. `sigmas[N]` is the appended `0.0`
    /// terminal sentinel; Euler step uses `sigmas[i+1] - sigmas[i]`.
    sigmas: Vec<f32>,
}

impl FlowMatchEulerScheduler {
    pub fn new(num_inference_steps: usize, image_seq_len: usize) -> Self {
        assert!(num_inference_steps >= 2, "need at least 2 inference steps");
        let mu = calculate_shift(image_seq_len);

        // `set_timesteps`: timesteps = linspace(sigma_to_t(sigma_max=1.0)=1000,
        // sigma_to_t(sigma_min=0.0)=0, N); sigmas = timesteps / 1000.
        // Endpoints inclusive: sigma[0]=1.0, sigma[N-1]=0.0.
        let n = num_inference_steps;
        let mut sigmas = Vec::with_capacity(n + 1);
        let denom = (n - 1) as f32;
        for i in 0..n {
            let frac = i as f32 / denom;
            sigmas.push(1.0 - frac);
        }
        // Dynamic shift: time_shift(mu, 1.0, sigma). t=0 is a singularity; the
        // appended terminal 0.0 below is what the Euler step actually reads.
        // Last linspace point is 0.0 which would `1/0`; replace its post-shift
        // value with 0.0 explicitly.
        for s in sigmas.iter_mut() {
            if *s > 0.0 {
                *s = time_shift_exponential(mu, 1.0, *s);
            } else {
                *s = 0.0;
            }
        }
        sigmas.push(0.0); // terminal sentinel (appended after time shift)
        Self { sigmas }
    }

    pub fn num_inference_steps(&self) -> usize {
        self.sigmas.len() - 1
    }

    /// `sigmas[i]` is the current step; `sigmas[i+1]` is the next step.
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// Normalized timestep fed to the model at step `i`, matching
    /// `pipeline.py: timestep = (1000 - t) / 1000` with `t = sigma * 1000`,
    /// i.e. `1 - sigma[i]` in `[0, 1]`.
    pub fn t_norm(&self, i: usize) -> f32 {
        1.0 - self.sigmas[i]
    }

    /// Euler update in place: `sample += (sigma[i+1] - sigma[i]) * model_out`.
    /// `model_out` and `sample` must have the same length.
    pub fn step(&self, i: usize, model_out: &[f32], sample: &mut [f32]) {
        assert_eq!(model_out.len(), sample.len(), "shape mismatch");
        let dt = self.sigmas[i + 1] - self.sigmas[i];
        for (s, m) in sample.iter_mut().zip(model_out.iter()) {
            *s += dt * *m;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_endpoints_match_python() {
        // Hand-computed: m = (1.15-0.5)/(4096-256) = 0.65/3840.
        // At seq=256: mu = 0.5. At seq=4096: mu = 1.15.
        assert!((calculate_shift(256) - 0.5).abs() < 1e-6);
        assert!((calculate_shift(4096) - 1.15).abs() < 1e-6);
    }

    #[test]
    fn sigmas_are_monotone_with_endpoints() {
        let sch = FlowMatchEulerScheduler::new(8, 1024);
        let s = sch.sigmas();
        assert_eq!(s.len(), 9);
        assert!((s[0] - 1.0).abs() < 1e-6);
        assert_eq!(s[8], 0.0);
        for w in s.windows(2) {
            assert!(w[0] >= w[1], "sigmas must be monotonically decreasing");
        }
    }

    #[test]
    fn euler_step_zero_pred_is_identity() {
        let sch = FlowMatchEulerScheduler::new(4, 1024);
        let mut x = vec![0.3_f32, -0.7, 1.5, 0.0];
        let pred = vec![0.0_f32; 4];
        let before = x.clone();
        sch.step(0, &pred, &mut x);
        assert_eq!(x, before);
    }
}
