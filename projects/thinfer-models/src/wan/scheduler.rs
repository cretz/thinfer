//! DMD (Distribution Matching Distillation) few-step sampler for the distilled
//! Wan2.2 line (FastWan, LongLive). Replaces the SkyReels-DF UniPC multistep
//! solver: a DMD-distilled generator predicts the flow velocity at a small set
//! of fixed timesteps and is re-noised between them, so there is no multistep
//! corrector/predictor state and no CFG.
//!
//! Inference (per FastVideo `DmdDenoisingStage` + `FlowMatchEulerDiscreteScheduler`):
//! the schedule is a list of integer model timesteps (e.g. `[1000, 757, 522]`),
//! already in the shifted flow space. For each step `i` at timestep `t`:
//!
//! ```text
//! sigma_i  = t / num_train_timesteps                  // flow sigma at t
//! x0       = x_t - sigma_i * velocity                 // velocity -> clean latent
//! x_{i+1}  = (1 - sigma_{i+1}) * x0 + sigma_{i+1} * noise   // re-noise (i < last)
//! result   = x0                                        // last step
//! ```
//!
//! `sigma = t / num_train_timesteps` because FastVideo looks the sigma up by
//! nearest-timestep against a schedule whose `timesteps == sigma * 1000`, so the
//! lookup returns exactly `t / 1000`; the `shift` only reshapes which integer
//! timesteps the distillation chose (baked into the schedule), not the per-step
//! sigma. The model is fed the scalar `t` (uniform over all latent tokens) as
//! its time embedding input.
//!
//! The re-noise Gaussian is supplied by the caller (the pipeline owns seeding),
//! keeping this module pure schedule math. Stdlib-only; CPU; runs between DiT
//! submits.

/// Per-model DMD schedule. Lives on the model side (`manifest`) so each distilled
/// variant (FastWan 3-step, LongLive 4-step, ...) supplies its own steps without
/// touching the sampler.
#[derive(Clone, Debug)]
pub struct DmdConfig {
    /// Integer model timesteps, high noise -> low, in shifted flow space. Length
    /// is the number of inference steps. FastWan2.2-TI2V-5B: `[1000, 757, 522]`.
    pub denoising_steps: Vec<f32>,
    /// Flow `num_train_timesteps` (1000 for the Wan family); divides a timestep
    /// to its flow sigma.
    pub num_train_timesteps: f32,
}

impl DmdConfig {
    /// FastWan2.2-TI2V-5B-FullAttn: DMD 3-step (`--dmd-denoising-steps
    /// "1000,757,522"`), `num_train_timesteps = 1000`.
    pub fn fastwan_ti2v_5b() -> Self {
        Self {
            denoising_steps: vec![1000.0, 757.0, 522.0],
            num_train_timesteps: 1000.0,
        }
    }
}

/// Stateless DMD sampler over a [`DmdConfig`]. Holds only the precomputed
/// per-step sigmas.
pub struct DmdSampler {
    /// Model timesteps fed to the DiT, one per step.
    timesteps: Vec<f32>,
    /// `sigmas[i] = timesteps[i] / num_train_timesteps`.
    sigmas: Vec<f32>,
}

impl DmdSampler {
    pub fn new(cfg: &DmdConfig) -> Self {
        assert!(
            !cfg.denoising_steps.is_empty(),
            "DMD needs at least one denoising step"
        );
        let sigmas = cfg
            .denoising_steps
            .iter()
            .map(|&t| t / cfg.num_train_timesteps)
            .collect();
        Self {
            timesteps: cfg.denoising_steps.clone(),
            sigmas,
        }
    }

    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    /// Model timestep fed to the DiT at step `i` (scalar, broadcast over all
    /// latent tokens).
    pub fn timestep(&self, i: usize) -> f32 {
        self.timesteps[i]
    }

    /// Flow sigma at step `i`.
    pub fn sigma(&self, i: usize) -> f32 {
        self.sigmas[i]
    }

    /// Number of i.i.d. standard-normal samples the caller must supply to
    /// [`DmdSampler::step`] at step `i`: the full latent size when `i` is not the
    /// last step (re-noise), else 0.
    pub fn noise_len(&self, i: usize, latent_len: usize) -> usize {
        if i + 1 < self.num_steps() {
            latent_len
        } else {
            0
        }
    }

    /// Advance one step: convert the DiT `velocity` at the current latent
    /// `sample` to the clean latent `x0 = sample - sigma_i * velocity`, then
    /// re-noise to the next step (`x0` survives unchanged on the final step).
    /// `noise` must have length [`DmdSampler::noise_len`] for this step.
    pub fn step(&self, i: usize, velocity: &[f32], sample: &[f32], noise: &[f32]) -> Vec<f32> {
        assert_eq!(
            velocity.len(),
            sample.len(),
            "velocity/sample length mismatch"
        );
        let sigma = self.sigmas[i];
        let x0 = sample
            .iter()
            .zip(velocity)
            .map(|(&x, &v)| x - sigma * v)
            .collect::<Vec<f32>>();
        if i + 1 == self.num_steps() {
            return x0;
        }
        let sigma_next = self.sigmas[i + 1];
        assert_eq!(
            noise.len(),
            x0.len(),
            "re-noise needs one Gaussian per latent"
        );
        x0.iter()
            .zip(noise)
            .map(|(&z, &n)| (1.0 - sigma_next) * z + sigma_next * n)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Wan 2.2 14B MoE step-distill sampler (LightX2V distilled LoRA path)
// ---------------------------------------------------------------------------

/// Step-distillation sampler for Wan2.2-A14B + LightX2V (lightx2v
/// `Wan22StepDistillScheduler`). 4 steps, CFG-free, flow-match Euler over a
/// SHIFTED sigma schedule. For `denoising_step_list = [1000,750,500,250]` the raw
/// sigma at step `i` is `list[i] / 1000`; the flow shift reshapes it as
/// `s' = shift*s / (1 + (shift-1)*s)`; the DiT is fed `timestep = s' * 1000`. The
/// Euler update is `x_{i+1} = x_i - (s'_i - s'_{i+1}) * velocity` (`s'_{last+1} =
/// 0`). No re-noise, no corrector state. The two-expert switch (high steps
/// `0..boundary`, low steps `boundary..`) lives in the pipeline, not here.
#[derive(Clone, Debug)]
pub struct Wan22DistillConfig {
    /// Nominal step labels, high noise -> low. Wan2.2 distill: `[1000,750,500,250]`.
    pub denoising_step_list: Vec<f32>,
    /// Flow shift (5.0 for the Wan2.2 distill T2V/I2V configs).
    pub shift: f32,
    pub num_train_timesteps: f32,
    /// Step index at which the MoE switches high-noise -> low-noise expert
    /// (`boundary_step_index = 2` for the 4-step distill: 2 high / 2 low).
    pub boundary_step: usize,
}

impl Wan22DistillConfig {
    /// Wan2.2-T2V-A14B + LightX2V 4-step distill.
    pub fn wan22_t2v_a14b() -> Self {
        Self {
            denoising_step_list: vec![1000.0, 750.0, 500.0, 250.0],
            shift: 5.0,
            num_train_timesteps: 1000.0,
            boundary_step: 2,
        }
    }
}

/// Stateless flow-Euler sampler over the shifted distill schedule.
pub struct Wan22DistillSampler {
    /// Shifted sigmas, one per step.
    sigmas: Vec<f32>,
    /// Model timesteps fed to the DiT (`sigma * num_train_timesteps`).
    timesteps: Vec<f32>,
    boundary_step: usize,
}

impl Wan22DistillSampler {
    pub fn new(cfg: &Wan22DistillConfig) -> Self {
        assert!(
            !cfg.denoising_step_list.is_empty(),
            "distill needs at least one step"
        );
        let shift = cfg.shift;
        let sigmas: Vec<f32> = cfg
            .denoising_step_list
            .iter()
            .map(|&x| {
                let s = x / cfg.num_train_timesteps;
                shift * s / (1.0 + (shift - 1.0) * s)
            })
            .collect();
        let timesteps = sigmas
            .iter()
            .map(|&s| s * cfg.num_train_timesteps)
            .collect();
        Self {
            sigmas,
            timesteps,
            boundary_step: cfg.boundary_step,
        }
    }

    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    /// Model timestep fed to the DiT at step `i`.
    pub fn timestep(&self, i: usize) -> f32 {
        self.timesteps[i]
    }

    /// Which MoE expert runs at step `i`: `true` = high-noise (expert 0, early
    /// steps), `false` = low-noise (expert 1).
    pub fn is_high_noise(&self, i: usize) -> bool {
        i < self.boundary_step
    }

    /// Flow-Euler step: `x_{i+1} = x_i - (sigma_i - sigma_{i+1}) * velocity`
    /// (`sigma_{i+1} = 0` past the last step).
    pub fn step(&self, i: usize, velocity: &[f32], sample: &[f32]) -> Vec<f32> {
        assert_eq!(
            velocity.len(),
            sample.len(),
            "velocity/sample length mismatch"
        );
        let sigma_next = if i + 1 < self.num_steps() {
            self.sigmas[i + 1]
        } else {
            0.0
        };
        let d = self.sigmas[i] - sigma_next;
        sample
            .iter()
            .zip(velocity)
            .map(|(&x, &v)| x - d * v)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wan22_distill_shifted_schedule() {
        let s = Wan22DistillSampler::new(&Wan22DistillConfig::wan22_t2v_a14b());
        assert_eq!(s.num_steps(), 4);
        // s'=shift*s/(1+(shift-1)*s), shift=5: s=1->1.0, .75->.9375, .5->.8333, .25->.625
        let want = [1000.0, 937.5, 833.3333, 625.0];
        for (i, w) in want.iter().enumerate() {
            assert!(
                (s.timestep(i) - w).abs() < 0.1,
                "step {i}: {} vs {w}",
                s.timestep(i)
            );
        }
        assert!(s.is_high_noise(0) && s.is_high_noise(1));
        assert!(!s.is_high_noise(2) && !s.is_high_noise(3));
    }

    #[test]
    fn wan22_distill_euler_last_step_is_x0() {
        let s = Wan22DistillSampler::new(&Wan22DistillConfig::wan22_t2v_a14b());
        let sample = vec![0.5_f32, -0.2, 1.0];
        let vel = vec![0.1_f32, 0.3, -0.4];
        let last = s.num_steps() - 1;
        let out = s.step(last, &vel, &sample);
        // sigma_next=0 -> x - sigma_last*v
        let sig = (1000.0_f32 * 0.625) / 1000.0; // s'(0.25)=0.625
        for k in 0..3 {
            assert!((out[k] - (sample[k] - sig * vel[k])).abs() < 1e-5);
        }
    }

    #[test]
    fn sigmas_are_timestep_over_train() {
        let s = DmdSampler::new(&DmdConfig::fastwan_ti2v_5b());
        assert_eq!(s.num_steps(), 3);
        assert_eq!(s.timestep(0), 1000.0);
        assert!((s.sigma(0) - 1.0).abs() < 1e-6);
        assert!((s.sigma(1) - 0.757).abs() < 1e-6);
        assert!((s.sigma(2) - 0.522).abs() < 1e-6);
    }

    #[test]
    fn final_step_returns_x0_and_ignores_noise() {
        let s = DmdSampler::new(&DmdConfig::fastwan_ti2v_5b());
        let sample = vec![0.5_f32, -0.2, 1.0];
        let vel = vec![0.1_f32, 0.3, -0.4];
        let last = s.num_steps() - 1;
        assert_eq!(s.noise_len(last, sample.len()), 0);
        let out = s.step(last, &vel, &sample, &[]);
        let sigma = s.sigma(last);
        for k in 0..3 {
            assert!((out[k] - (sample[k] - sigma * vel[k])).abs() < 1e-6);
        }
    }

    #[test]
    fn nonfinal_step_renoises_to_next_sigma() {
        let s = DmdSampler::new(&DmdConfig::fastwan_ti2v_5b());
        let sample = vec![0.3_f32, -0.7, 1.5, 0.2];
        let vel = vec![0.05_f32; 4];
        assert_eq!(s.noise_len(0, sample.len()), 4);
        let noise = vec![1.0_f32, -1.0, 0.5, 0.0];
        let out = s.step(0, &vel, &sample, &noise);
        let s1 = s.sigma(1);
        for k in 0..4 {
            let x0 = sample[k] - s.sigma(0) * vel[k];
            let want = (1.0 - s1) * x0 + s1 * noise[k];
            assert!((out[k] - want).abs() < 1e-6, "k={k} {} vs {want}", out[k]);
        }
    }
}
