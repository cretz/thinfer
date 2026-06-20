//! FlowUniPC multistep sampler for LongLive-2.0-5B (the AR causal variant).
//!
//! LongLive does NOT use FastWan's stateless Euler-renoise [`super::scheduler::
//! DmdSampler`]; per chunk it runs upstream's `FlowUniPCMultistepScheduler`
//! (`wan_5b/utils/fm_solvers_unipc.py`) with `solver_order=2`, `predict_x0=True`,
//! `solver_type="bh2"`, `lower_order_final=True`, `final_sigmas_type="zero"`, CFG
//! off. This is a genuine predictor+corrector multistep solver, so it carries a
//! short history of converted-x0 model outputs across steps. Ported line-by-line
//! from the source; orders never exceed 2, so the only linear solve (the order-2
//! corrector's 2x2 system) is inlined in closed form.
//!
//! Per AR chunk: [`FlowUniPc::reset`], then for each of the 4 steps feed the DiT
//! velocity at [`FlowUniPc::timestep`] into [`FlowUniPc::step`]; the returned
//! latent is the next step's input. After the last step the result is the clean
//! latent (final sigma 0). Scalar coefficients are computed in f32 to mirror the
//! float32 `sigmas` table torch builds; the latent vectors stay f32.

/// Per-variant FlowUniPC schedule knobs. LongLive: 4 steps, shift 5.0.
#[derive(Clone, Debug)]
pub struct UniPcConfig {
    pub sampling_steps: usize,
    pub shift: f32,
    pub num_train_timesteps: f32,
    /// `sigma_min` of the training table (`1 - alphas_cumprod[-1]`); for the Wan
    /// flow schedule `alphas = linspace(1, 1/T, T)[::-1]` so this is `1/T`.
    pub sigma_min: f32,
}

impl UniPcConfig {
    /// LongLive-2.0-5B (`configs/inference.yaml`): `sampling_steps=4`,
    /// `timestep_shift=5.0`, `num_train_timesteps=1000`.
    pub fn longlive() -> Self {
        Self {
            sampling_steps: 4,
            shift: 5.0,
            num_train_timesteps: 1000.0,
            sigma_min: 0.001,
        }
    }
}

const SOLVER_ORDER: usize = 2;

/// Stateful FlowUniPC sampler. Hold one per AR denoise; [`Self::reset`] between
/// chunks (the upstream `set_timesteps` re-init).
pub struct FlowUniPc {
    /// Flow sigmas, length `sampling_steps + 1`, final entry 0.0.
    sigmas: Vec<f32>,
    /// Integer model timesteps fed to the DiT, length `sampling_steps`.
    timesteps: Vec<i64>,
    /// Converted-x0 history, FIFO of length `SOLVER_ORDER` (oldest first).
    model_outputs: Vec<Option<Vec<f32>>>,
    /// Sample before the previous predictor (the corrector's `x`).
    last_sample: Option<Vec<f32>>,
    step_index: usize,
    lower_order_nums: usize,
    /// Order selected by the previous step's predictor; the next corrector uses it.
    this_order: usize,
}

impl FlowUniPc {
    pub fn new(cfg: &UniPcConfig) -> Self {
        let n = cfg.sampling_steps;
        // sigmas = linspace(1, sigma_min, n+1)[:-1]; shift; append 0.0.
        let mut sigmas = Vec::with_capacity(n + 1);
        let mut timesteps = Vec::with_capacity(n);
        for i in 0..n {
            let lin = 1.0 + (cfg.sigma_min - 1.0) * (i as f32) / (n as f32);
            let s = cfg.shift * lin / (1.0 + (cfg.shift - 1.0) * lin);
            sigmas.push(s);
            // timesteps = sigma * num_train_timesteps, truncated to int64.
            timesteps.push((s * cfg.num_train_timesteps) as i64);
        }
        sigmas.push(0.0); // final_sigmas_type = "zero"
        let mut s = Self {
            sigmas,
            timesteps,
            model_outputs: vec![None; SOLVER_ORDER],
            last_sample: None,
            step_index: 0,
            lower_order_nums: 0,
            this_order: 0,
        };
        s.reset();
        s
    }

    /// Re-init the multistep state for a new chunk (history, counters cleared;
    /// the sigma/timestep tables are fixed).
    pub fn reset(&mut self) {
        self.model_outputs = vec![None; SOLVER_ORDER];
        self.last_sample = None;
        self.step_index = 0;
        self.lower_order_nums = 0;
        self.this_order = 0;
    }

    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    /// Integer model timestep fed to the DiT at step `i`.
    pub fn timestep(&self, i: usize) -> f32 {
        self.timesteps[i] as f32
    }

    /// `(alpha, sigma)` for a flow sigma: `alpha = 1 - sigma`.
    fn alpha_sigma(sigma: f32) -> (f32, f32) {
        (1.0 - sigma, sigma)
    }

    /// `lambda = log(alpha) - log(sigma)` (half-log-SNR). `sigma = 0` yields
    /// `+inf`, which flows through `expm1(-inf) = -1` correctly at the final step.
    fn lambda(sigma: f32) -> f32 {
        let (alpha, sig) = Self::alpha_sigma(sigma);
        alpha.ln() - sig.ln()
    }

    /// `convert_model_output` (flow -> x0, predict_x0): `x0 = sample - sigma*v`.
    fn convert(&self, flow: &[f32], sample: &[f32]) -> Vec<f32> {
        let sigma = self.sigmas[self.step_index];
        sample
            .iter()
            .zip(flow)
            .map(|(&x, &v)| x - sigma * v)
            .collect()
    }

    /// One scheduler step: `flow` is the DiT velocity at the current latent
    /// `sample`; returns the latent for the next step. Mutates the multistep
    /// state. Mirrors `FlowUniPCMultistepScheduler.step`.
    pub fn step(&mut self, flow: &[f32], sample: &[f32]) -> Vec<f32> {
        let use_corrector = self.step_index > 0 && self.last_sample.is_some();
        let m_conv = self.convert(flow, sample);

        // Corrector runs BEFORE the history shift, against the previous step's
        // model output (`model_outputs[-1]`) and `last_sample`.
        let mut sample: Vec<f32> = if use_corrector {
            self.uni_c(&m_conv)
        } else {
            sample.to_vec()
        };

        // History FIFO shift: push the freshly converted output.
        self.model_outputs.rotate_left(1);
        *self.model_outputs.last_mut().unwrap() = Some(m_conv);

        // Predictor order: lower_order_final + multistep warmup.
        let this_order = SOLVER_ORDER.min(self.timesteps.len() - self.step_index);
        self.this_order = this_order.min(self.lower_order_nums + 1);

        self.last_sample = Some(sample.clone());
        let prev = self.uni_p(&mut sample);

        if self.lower_order_nums < SOLVER_ORDER {
            self.lower_order_nums += 1;
        }
        self.step_index += 1;
        prev
    }

    /// `bh2` coefficients for a `(sigma_t, sigma_s0)` pair (predict_x0): returns
    /// `(coef_x = sigma_t/sigma_s0, alpha_t, h_phi_1, b_h, hh)` plus `lambda_s0`,
    /// `h` for the rk terms. `h_phi_1 == b_h` for bh2.
    fn coeffs(sigma_t: f32, sigma_s0: f32) -> Bh {
        let (alpha_t, _) = Self::alpha_sigma(sigma_t);
        let lambda_t = Self::lambda(sigma_t);
        let lambda_s0 = Self::lambda(sigma_s0);
        let h = lambda_t - lambda_s0;
        let hh = -h; // predict_x0
        let h_phi_1 = hh.exp_m1();
        Bh {
            coef_x: sigma_t / sigma_s0,
            alpha_t,
            h_phi_1,
            b_h: h_phi_1, // bh2: B_h = expm1(hh)
            hh,
            lambda_s0,
            h,
        }
    }

    /// UniP predictor (`multistep_uni_p_bh_update`, predict_x0). Uses sigmas at
    /// `[step_index+1, step_index]`. Order 1 or 2 only.
    fn uni_p(&self, x: &mut [f32]) -> Vec<f32> {
        let i = self.step_index;
        let c = Self::coeffs(self.sigmas[i + 1], self.sigmas[i]);
        let m0 = self.model_outputs[SOLVER_ORDER - 1].as_ref().unwrap();
        let mut out: Vec<f32> = x
            .iter()
            .zip(m0)
            .map(|(&xv, &m)| c.coef_x * xv - c.alpha_t * c.h_phi_1 * m)
            .collect();
        if self.this_order == 2 {
            // si = step_index - 1; rk = (lambda_si - lambda_s0)/h; rho = 0.5.
            let rk = (Self::lambda(self.sigmas[i - 1]) - c.lambda_s0) / c.h;
            let m_prev = self.model_outputs[SOLVER_ORDER - 2].as_ref().unwrap();
            let scale = c.alpha_t * c.b_h * 0.5 / rk;
            for ((o, &m), &mp) in out.iter_mut().zip(m0).zip(m_prev) {
                *o -= scale * (mp - m);
            }
        }
        out
    }

    /// UniC corrector (`multistep_uni_c_bh_update`, predict_x0). Uses sigmas at
    /// `[step_index, step_index-1]`; corrects and returns the sample. `model_t`
    /// is the current step's converted x0. Order 1 or 2 only. `this_sample`
    /// (upstream `x_t`) is never read - it is overwritten by the formula - so it
    /// is not a parameter here.
    fn uni_c(&self, model_t: &[f32]) -> Vec<f32> {
        let i = self.step_index;
        let order = self.this_order;
        let c = Self::coeffs(self.sigmas[i], self.sigmas[i - 1]);
        let m0 = self.model_outputs[SOLVER_ORDER - 1].as_ref().unwrap();
        let x = self.last_sample.as_ref().unwrap();

        // x_t_ = coef_x * x - alpha_t * h_phi_1 * m0.
        let base: Vec<f32> = x
            .iter()
            .zip(m0)
            .map(|(&xv, &m)| c.coef_x * xv - c.alpha_t * c.h_phi_1 * m)
            .collect();

        // rho weights: order 1 -> [0.5]; order 2 -> the 2x2 UniPC system
        // R*rho = b with R = [[1,1],[rk,1]]. `corr_res = rho0 * D1s` where
        // D1s = (m_prev - m0)/rk, and the D1_t term uses rho1.
        let (rho_d1s, rho_dt) = if order == 1 {
            (0.0, 0.5)
        } else {
            // si = step_index - 2; rk = (lambda_si - lambda_s0)/h. At step 2 this
            // references sigma=1.0 (alpha=0 -> lambda=-inf -> rk=-inf); the
            // pivoted solve keeps it finite (rho0->0, rho1->b0), matching LAPACK.
            let rk = (Self::lambda(self.sigmas[i - 2]) - c.lambda_s0) / c.h;
            let (b0, b1) = Self::bh_b_vector(&c);
            let (rho0, rho1) = solve_unipc_2x2(rk, b0, b1);
            (rho0 / rk, rho1)
        };

        let m_prev = if order == 2 {
            self.model_outputs[SOLVER_ORDER - 2].as_ref()
        } else {
            None
        };
        let scale = c.alpha_t * c.b_h;
        base.iter()
            .enumerate()
            .map(|(e, &bv)| {
                let m0e = m0[e];
                let d1_t = model_t[e] - m0e;
                let corr_res = match m_prev {
                    Some(mp) => rho_d1s * (mp[e] - m0e),
                    None => 0.0,
                };
                bv - scale * (corr_res + rho_dt * d1_t)
            })
            .collect()
    }

    /// The `b` vector (length 2) of the UniPC linear system for order 2, bh2.
    fn bh_b_vector(c: &Bh) -> (f32, f32) {
        let mut h_phi_k = c.h_phi_1 / c.hh - 1.0;
        let mut factorial = 1.0f32;
        let b0 = h_phi_k * factorial / c.b_h;
        factorial *= 2.0;
        h_phi_k = h_phi_k / c.hh - 1.0 / factorial;
        let b1 = h_phi_k * factorial / c.b_h;
        (b0, b1)
    }
}

/// Solve `R*rho = b` for the order-2 UniPC matrix `R = [[1, 1], [rk, 1]]` with
/// partial pivoting, mirroring `torch.linalg.solve` (LAPACK). Pivoting on the
/// larger-magnitude column-0 entry is what keeps the result finite when
/// `rk = -inf` (the sigma=1.0 edge): without it the naive `(b1 - rk*b0)/(1-rk)`
/// is `inf/inf = NaN`.
fn solve_unipc_2x2(rk: f32, b0: f32, b1: f32) -> (f32, f32) {
    if rk.abs() > 1.0 {
        // Pivot rows -> [[rk,1],[1,1]] with rhs [b1,b0]; eliminate, back-sub.
        let f = 1.0 / rk;
        let rho1 = (b0 - f * b1) / (1.0 - f);
        let rho0 = (b1 - rho1) / rk;
        (rho0, rho1)
    } else {
        let det = 1.0 - rk;
        ((b0 - b1) / det, (b1 - rk * b0) / det)
    }
}

/// bh2 per-pair coefficients.
struct Bh {
    coef_x: f32,
    alpha_t: f32,
    h_phi_1: f32,
    b_h: f32,
    hh: f32,
    lambda_s0: f32,
    h: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_matches_upstream() {
        let s = FlowUniPc::new(&UniPcConfig::longlive());
        assert_eq!(s.num_steps(), 4);
        assert_eq!(s.timesteps, vec![1000, 937, 833, 625]);
        let want = [1.0, 0.937578, 0.833611, 0.625936, 0.0];
        for (i, &w) in want.iter().enumerate() {
            assert!(
                (s.sigmas[i] - w).abs() < 2e-4,
                "sigma[{i}] = {} vs {w}",
                s.sigmas[i]
            );
        }
    }

    /// Fixed-point invariant: if the model predicts a CONSTANT clean latent x0=c
    /// at every step (velocity = (sample - c)/sigma), the UniPC trajectory must
    /// converge to exactly c. This exercises the full predictor+corrector path,
    /// the order [1,2,2,1] progression, and the final sigma=0 / expm1(-inf) edge.
    #[test]
    fn constant_x0_is_a_fixed_point() {
        let mut s = FlowUniPc::new(&UniPcConfig::longlive());
        s.reset();
        let c: Vec<f32> = vec![0.3, -1.2, 0.75, 2.0, -0.5];
        // Start from arbitrary noise.
        let mut sample: Vec<f32> = vec![1.7, 0.4, -2.1, 0.9, 3.3];
        for i in 0..s.num_steps() {
            let sigma = s.sigmas[i];
            // velocity such that x0 = sample - sigma*v = c.
            let flow: Vec<f32> = sample
                .iter()
                .zip(&c)
                .map(|(&x, &cc)| (x - cc) / sigma)
                .collect();
            sample = s.step(&flow, &sample);
        }
        for (g, &cc) in sample.iter().zip(&c) {
            assert!((g - cc).abs() < 1e-3, "got {g} want {cc}");
        }
    }

    /// The order progression is [1,2,2,1] with the corrector engaged on steps
    /// 1..=3. Drive trivial (zero-velocity) steps and inspect the recorded order.
    #[test]
    fn order_progression_is_1_2_2_1() {
        let mut s = FlowUniPc::new(&UniPcConfig::longlive());
        let mut orders = Vec::new();
        let mut sample = vec![0.5f32; 3];
        for _ in 0..s.num_steps() {
            let flow = vec![0.0f32; 3];
            sample = s.step(&flow, &sample);
            orders.push(s.this_order);
        }
        assert_eq!(orders, vec![1, 2, 2, 1]);
    }
}
