//! UniPCMultistepScheduler, SkyReels-V2 / Wan2.1 config (the shipped
//! `scheduler_config.json`). Mirrors
//! `diffusers.UniPCMultistepScheduler` with:
//! - `prediction_type = "flow_prediction"`, `use_flow_sigmas = True`,
//!   `flow_shift = 1.0`, `use_dynamic_shifting = False`
//! - `solver_order = 2`, `solver_type = "bh2"`, `predict_x0 = True`
//! - `lower_order_final = True`, `disable_corrector = []`, no `solver_p`
//! - `final_sigmas_type = "zero"`, `timestep_spacing = "linspace"`,
//!   `num_train_timesteps = 1000`
//!
//! Diffusion-Forcing SYNCHRONOUS mode only (parity): `ar_step = 0`,
//! `causal_block_size = 1`, so `generate_timestep_matrix` collapses to "all
//! frames share `timesteps[i]` at step `i`" and the per-frame `sample_schedulers`
//! all advance in lockstep. Every scalar coefficient (h, B_h, rhos) depends only
//! on the (frame-shared) sigmas, and every tensor op is elementwise, so one
//! scheduler over the whole flattened latent `[z, f, h, w]` is bit-identical to
//! the pyref's per-frame schedulers. Async/causal-block staggering is deferred
//! to long-video.
//!
//! Stdlib-only; CPU; runs once per step between DiT submits. Only solver orders
//! 1 and 2 are realized (this family pins `solver_order = 2`); a higher order
//! would need the general `R x = b` solve and is a deliberate `unimplemented!`.

const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;
const SOLVER_ORDER: usize = 2;

/// Per-step scheduler internals for parity bisection (filled by
/// [`UniPCScheduler::step_with_diag`]). No effect on the returned sample.
#[derive(Default, Clone)]
pub struct SchedulerStepDiag {
    /// `sigmas[step_index]` at this step.
    pub sigma: f32,
    /// Order the corrector ran at this step (1 during warmup and the final
    /// `lower_order_final` step; 0 on step 0 where no corrector runs).
    pub this_order: usize,
    /// Whether the corrector ran (false on step 0).
    pub used_corrector: bool,
    /// `convert_model_output` result (the x0 prediction): `sample - sigma*model`.
    pub m_conv: Vec<f32>,
    /// The post-corrector sample (the corrector output when `used_corrector`,
    /// else the unchanged input sample). This is what becomes the next step's
    /// `last_sample`.
    pub corrected: Vec<f32>,
}

/// Flow-sigma `_sigma_to_alpha_sigma_t`: `alpha_t = 1 - sigma`, `sigma_t = sigma`.
fn alpha_sigma(sigma: f32) -> (f32, f32) {
    (1.0 - sigma, sigma)
}

/// `lambda = log(alpha_t) - log(sigma_t)` for a flow sigma.
fn lambda(sigma: f32) -> f32 {
    let (a, s) = alpha_sigma(sigma);
    a.ln() - s.ln()
}

pub struct UniPCScheduler {
    /// Length `N + 1`; `sigmas[N] = 0` (the `final_sigmas_type = "zero"`
    /// terminal). `sigmas[0]` carries the `-1e-6` nudge that keeps the first
    /// `log(alpha)` finite.
    sigmas: Vec<f32>,
    /// Per-step model timesteps fed to the DiT: `trunc(sigmas[i] * 1000)` (the
    /// pyref int64 cast), length `N`.
    timesteps: Vec<f32>,

    // --- multistep state (advances across `step`) ---
    /// Converted (x0-prediction) outputs, newest last; `None` until warm.
    model_outputs: Vec<Option<Vec<f32>>>,
    /// The corrector's `last_sample` (its `x`) = diffusers' `self.last_sample`,
    /// which it sets to the POST-corrector `sample` just before the predictor.
    /// NOT, in general, the prior step's predictor output: the SkyReels-DF
    /// pipeline writes each step result back in place (`latents[:, :, idx] =
    /// step(...)`), so the only step where `self.last_sample` ALIASES the
    /// predictor output is step 0 (no corrector ran, so `sample` is still the
    /// raw latent view that the in-place write clobbers). Once the corrector
    /// runs (step 1+), `sample` is its fresh output tensor and is unaffected by
    /// that write. Verified bit-exact vs the pyref `py_sched_corrlast` dumps:
    /// step1 last == step0 predictor out; step2 last == step1 CORRECTOR out.
    last_sample: Option<Vec<f32>>,
    /// Order used by the *next* corrector (set at the end of each step).
    this_order: usize,
    lower_order_nums: usize,
    step_index: usize,
}

impl UniPCScheduler {
    /// `set_timesteps(num_inference_steps)` for the pinned flow config.
    pub fn new(num_inference_steps: usize) -> Self {
        assert!(num_inference_steps >= 1, "need at least 1 inference step");
        let n = num_inference_steps;

        // Flow sigmas: linspace(1, 1/1000, N+1)[:-1]; flow_shift 1.0 + no dynamic
        // shifting leaves them untouched. Then nudge sigmas[0] off 1.0 and append
        // the zero terminal.
        let mut sigmas = Vec::with_capacity(n + 1);
        let step = (1.0 / NUM_TRAIN_TIMESTEPS - 1.0) / n as f32;
        for i in 0..n {
            sigmas.push(1.0 + i as f32 * step);
        }
        if (sigmas[0] - 1.0).abs() < 1e-6 {
            sigmas[0] -= 1e-6;
        }
        let timesteps = sigmas
            .iter()
            .map(|s| (s * NUM_TRAIN_TIMESTEPS).trunc())
            .collect();
        sigmas.push(0.0);

        Self {
            sigmas,
            timesteps,
            model_outputs: vec![None; SOLVER_ORDER],
            last_sample: None,
            this_order: 0,
            lower_order_nums: 0,
            step_index: 0,
        }
    }

    pub fn num_inference_steps(&self) -> usize {
        self.timesteps.len()
    }

    /// Per-step DiT timestep values (broadcast equal over all latent frames in
    /// synchronous DF mode), length `num_inference_steps`.
    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// `convert_model_output` (flow_prediction, predict_x0): `x0 = sample -
    /// sigma[i] * model_output`.
    fn convert(&self, model_output: &[f32], sample: &[f32]) -> Vec<f32> {
        let sigma = self.sigmas[self.step_index];
        sample
            .iter()
            .zip(model_output)
            .map(|(&x, &m)| x - sigma * m)
            .collect()
    }

    /// `b_i = h_phi_k * factorial_i / B_h` for `i in 1..=order` (the shared
    /// predictor/corrector coefficient recurrence; `bh2` so `B_h = expm1(hh)`).
    fn b_coeffs(order: usize, hh: f32, b_h: f32) -> Vec<f32> {
        let h_phi_1 = hh.exp_m1();
        let mut h_phi_k = h_phi_1 / hh - 1.0;
        let mut factorial_i = 1.0_f32;
        let mut b = Vec::with_capacity(order);
        for i in 1..=order {
            b.push(h_phi_k * factorial_i / b_h);
            factorial_i *= (i + 1) as f32;
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i;
        }
        b
    }

    /// `multistep_uni_p_bh_update` (predict_x0, bh2). `m0` is the newest
    /// converted output; the predictor steps `sigmas[step] -> sigmas[step+1]`.
    fn predictor(&self, sample: &[f32], order: usize) -> Vec<f32> {
        let m0 = self.model_outputs[SOLVER_ORDER - 1]
            .as_ref()
            .expect("predictor needs a converted output");
        let sigma_t = self.sigmas[self.step_index + 1];
        let sigma_s0 = self.sigmas[self.step_index];
        let (alpha_t, sigma_t) = alpha_sigma(sigma_t);
        let h = lambda(sigma_t) - lambda(sigma_s0);
        let hh = -h; // predict_x0
        let h_phi_1 = hh.exp_m1();
        let b_h = hh.exp_m1(); // bh2

        let ratio = sigma_t / sigma_s0;
        // x_t_ = ratio * x - alpha_t * h_phi_1 * m0.
        let mut x: Vec<f32> = sample
            .iter()
            .zip(m0)
            .map(|(&xi, &mi)| ratio * xi - alpha_t * h_phi_1 * mi)
            .collect();

        if order >= 2 {
            // Order-2 uses the simplified rho_p = 0.5 over a single D1 term.
            let m1 = self.model_outputs[SOLVER_ORDER - 2]
                .as_ref()
                .expect("order-2 predictor needs two outputs");
            let rk = (lambda(self.sigmas[self.step_index - 1]) - lambda(sigma_s0)) / h;
            let coeff = alpha_t * b_h * 0.5;
            for ((xt, &a), &c) in x.iter_mut().zip(m1).zip(m0) {
                // D1 = (m1 - m0) / rk; pred_res = 0.5 * D1.
                *xt -= coeff * ((a - c) / rk);
            }
            assert!(order <= 2, "solver_order > 2 not implemented");
        }
        x
    }

    /// `multistep_uni_c_bh_update` (predict_x0, bh2). Corrects `this_sample` at
    /// `sigmas[step]` using `last_sample` at `sigmas[step-1]`. `model_t` is the
    /// freshly converted output at the current step.
    fn corrector(&self, model_t: &[f32], last_sample: &[f32], order: usize) -> Vec<f32> {
        let m0 = self.model_outputs[SOLVER_ORDER - 1]
            .as_ref()
            .expect("corrector needs a converted output");
        let sigma_t = self.sigmas[self.step_index];
        let sigma_s0 = self.sigmas[self.step_index - 1];
        let (alpha_t, sigma_t) = alpha_sigma(sigma_t);
        let h = lambda(sigma_t) - lambda(sigma_s0);
        let hh = -h; // predict_x0
        let h_phi_1 = hh.exp_m1();
        let b_h = hh.exp_m1(); // bh2

        let ratio = sigma_t / sigma_s0;
        let mut x: Vec<f32> = last_sample
            .iter()
            .zip(m0)
            .map(|(&xi, &mi)| ratio * xi - alpha_t * h_phi_1 * mi)
            .collect();

        let coeff = alpha_t * b_h;
        if order == 1 {
            // rho_c = 0.5; corr_res = 0; x -= alpha_t*B_h*(0.5 * D1_t).
            for ((xt, &mt), &c) in x.iter_mut().zip(model_t).zip(m0) {
                *xt -= coeff * (0.5 * (mt - c));
            }
        } else {
            assert_eq!(order, 2, "solver_order > 2 not implemented");
            let m1 = self.model_outputs[SOLVER_ORDER - 2]
                .as_ref()
                .expect("order-2 corrector needs two outputs");
            let rk = (lambda(self.sigmas[self.step_index - 2]) - lambda(sigma_s0)) / h;
            // Solve R x = b with R = [[1, 1], [rk, 1]], b = b_coeffs(2).
            let b = Self::b_coeffs(2, hh, b_h);
            let det = 1.0 - rk;
            let rho0 = (b[0] - b[1]) / det;
            let rho1 = (b[1] - rk * b[0]) / det;
            for ((xt, &a), (&c, &mt)) in x.iter_mut().zip(m1).zip(m0.iter().zip(model_t)) {
                let d1 = (a - c) / rk; // D1 = (m1 - m0)/rk
                let d1_t = mt - c; // D1_t = model_t - m0
                *xt -= coeff * (rho0 * d1 + rho1 * d1_t);
            }
        }
        x
    }

    /// One UniPC step: `prev_sample` from `model_output` (raw DiT velocity) and
    /// the current `sample`. Mirrors `UniPCMultistepScheduler.step`.
    pub fn step(&mut self, model_output: &[f32], sample: &[f32]) -> Vec<f32> {
        self.step_inner(model_output, sample, None)
    }

    /// Like [`UniPCScheduler::step`] but fills `diag` with this step's internals
    /// (sigma, order, converted output, post-corrector sample) for parity
    /// bisection. Identical returned sample; only used off the prod path.
    pub fn step_with_diag(
        &mut self,
        model_output: &[f32],
        sample: &[f32],
        diag: &mut SchedulerStepDiag,
    ) -> Vec<f32> {
        self.step_inner(model_output, sample, Some(diag))
    }

    fn step_inner(
        &mut self,
        model_output: &[f32],
        sample: &[f32],
        diag: Option<&mut SchedulerStepDiag>,
    ) -> Vec<f32> {
        assert_eq!(model_output.len(), sample.len(), "shape mismatch");
        let use_corrector = self.step_index > 0 && self.last_sample.is_some();
        let sigma = self.sigmas[self.step_index];

        let m_conv = self.convert(model_output, sample);
        let sample: Vec<f32> = if use_corrector {
            let last = self
                .last_sample
                .take()
                .expect("corrector needs last_sample");
            self.corrector(&m_conv, &last, self.this_order)
        } else {
            sample.to_vec()
        };

        if let Some(d) = diag {
            // `this_order` here is the order the corrector just ran at (set at the
            // end of the previous step); the predictor's order is recomputed
            // below. `corrected` == the input sample when no corrector ran.
            d.sigma = sigma;
            d.this_order = self.this_order;
            d.used_corrector = use_corrector;
            d.m_conv = m_conv.clone();
            d.corrected = sample.clone();
        }

        // Shift the converted-output history; newest goes last.
        for i in 0..SOLVER_ORDER - 1 {
            self.model_outputs[i] = self.model_outputs[i + 1].take();
        }
        self.model_outputs[SOLVER_ORDER - 1] = Some(m_conv);

        // lower_order_final warmup/cooldown.
        let n = self.timesteps.len();
        let this_order = SOLVER_ORDER.min(n - self.step_index); // lower_order_final
        self.this_order = this_order.min(self.lower_order_nums + 1);
        assert!(self.this_order > 0);

        let prev = self.predictor(&sample, self.this_order);
        // last_sample for the next corrector = diffusers' `self.last_sample =
        // sample`: the POST-corrector sample, captured before the predictor (see
        // the field doc). The DF pipeline's in-place latent write only aliases it
        // to the predictor OUTPUT on the no-corrector step (step 0), where
        // `sample` is the raw latent VIEW that `latents[:, :, idx] = prev`
        // clobbers. Once the corrector runs (step 1+), `sample` is the
        // corrector's fresh output tensor, untouched by that write, so the next
        // corrector reads the CORRECTED sample, not `prev`.
        self.last_sample = Some(if use_corrector {
            sample.clone()
        } else {
            prev.clone()
        });

        if self.lower_order_nums < SOLVER_ORDER {
            self.lower_order_nums += 1;
        }
        self.step_index += 1;
        prev
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmas_and_timesteps_match_flow_config() {
        // N=2: linspace(1, 0.001, 3)[:-1] = [1, 0.5005]; sigmas[0] nudged; append 0.
        let s = UniPCScheduler::new(2);
        assert_eq!(s.sigmas().len(), 3);
        assert!((s.sigmas()[0] - (1.0 - 1e-6)).abs() < 1e-7);
        assert!((s.sigmas()[1] - 0.5005).abs() < 1e-4);
        assert_eq!(s.sigmas()[2], 0.0);
        // timesteps = trunc(sigma*1000) = [999, 500].
        assert_eq!(s.timesteps(), &[999.0, 500.0]);
    }

    #[test]
    fn final_step_returns_x0_prediction() {
        // At the last step sigma_t = 0 (order collapses to 1), so prev_sample is
        // exactly the x0 prediction `sample - sigma*model_output`.
        let mut s = UniPCScheduler::new(1);
        let sample = vec![0.5_f32, -0.2, 1.0];
        let model = vec![0.1_f32, 0.3, -0.4];
        let sigma0 = s.sigmas()[0];
        let prev = s.step(&model, &sample);
        for i in 0..3 {
            let x0 = sample[i] - sigma0 * model[i];
            assert!((prev[i] - x0).abs() < 1e-5, "i={i} {} vs {}", prev[i], x0);
        }
    }

    #[test]
    fn last_sample_matches_diffusers_post_corrector_semantics() {
        // diffusers sets `self.last_sample = sample` (the POST-corrector sample)
        // right before the predictor. The SkyReels-DF in-place latent write
        // aliases that to the predictor OUTPUT only on the no-corrector step
        // (step 0, where `sample` is still the raw latent view); once the
        // corrector runs (step 1+), last_sample is the corrected sample, not the
        // predictor output. Lock both arms (verified bit-exact vs the pyref
        // py_sched_corrlast dumps).
        let mut s = UniPCScheduler::new(4);
        let mut x = vec![0.3_f32, -0.7, 1.5, 0.2];
        for i in 0..s.num_inference_steps() {
            let model = vec![0.05_f32 * (i as f32 + 1.0); 4];
            let mut d = SchedulerStepDiag::default();
            let prev = s.step_with_diag(&model, &x, &mut d);
            if d.used_corrector {
                // Stored last_sample is this step's corrector output, and is
                // distinct from the predictor output just returned.
                assert_eq!(s.last_sample.as_deref(), Some(d.corrected.as_slice()));
                assert_ne!(s.last_sample.as_deref(), Some(prev.as_slice()));
            } else {
                // Step 0: aliases the predictor output (no corrector ran).
                assert_eq!(s.last_sample.as_deref(), Some(prev.as_slice()));
            }
            x = prev;
        }
    }

    #[test]
    fn two_step_runs_predictor_then_corrector() {
        // Smoke test: a 2-step denoise produces finite output and consumes both
        // the predictor-only (step 0) and corrector+predictor (step 1) paths.
        let mut s = UniPCScheduler::new(2);
        let mut x = vec![0.3_f32, -0.7, 1.5, 0.0, 0.9];
        for i in 0..s.num_inference_steps() {
            let model = vec![0.05_f32 * (i as f32 + 1.0); 5];
            x = s.step(&model, &x);
            assert!(x.iter().all(|v| v.is_finite()));
        }
    }
}
