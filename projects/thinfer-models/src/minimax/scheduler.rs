//! `MiniMaxH3Scheduler`: rectified-flow Euler (`eta = 0`) with an exponential
//! sigma shift.
//!
//! A run drives TWO independent schedules over one shared transformer forward:
//! the released checkpoints use shift 12.0 for video latents and 3.0 for audio.
//!
//! Two details are load-bearing and easy to get wrong:
//!
//! - The model output is a **data-ward** velocity, so the denoised estimate is
//!   `x0 = x_t + (1 - t) * v`. Note the `+`, the OPPOSITE of the usual
//!   flow-match sign.
//! - The sigma used for `x0` is recovered from the **timestep** the transformer
//!   was conditioned on, while the Euler ratio uses the **sigma grid**. For
//!   `sigma < 0.5` the f32 round trip `1 - (1 - sigma)` is not exact, and the
//!   reference deliberately keeps the two sources apart.

/// Coefficients for one Euler step, in the reference's evaluation order.
///
/// The step is `prev = ratio * x_t + (1 - ratio) * (x_t + sigma_from_timestep * v)`.
/// This is algebraically `x_t + (1 - ratio) * sigma_from_timestep * v`, but the
/// two forms are NOT bit-identical in f32; the pipeline evaluates the form above
/// so the trajectory matches the reference step for step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepCoeffs {
    pub ratio: f32,
    pub one_minus_ratio: f32,
    pub sigma_from_timestep: f32,
}

#[derive(Debug, PartialEq)]
pub enum SchedulerError {
    /// Sigma shift must be positive.
    NonPositiveShift(f32),
    /// A schedule needs at least two sigmas to drive one evaluation.
    TooFewSteps(usize),
    /// `(requested, available)`.
    StepOutOfRange(usize, usize),
}

impl core::fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonPositiveShift(s) => write!(f, "sigma shift must be positive, got {s}"),
            Self::TooFewSteps(n) => {
                write!(f, "a schedule needs at least 2 sigmas, got {n}")
            }
            Self::StepOutOfRange(i, n) => {
                write!(f, "step {i} is past the end of a {n}-step schedule")
            }
        }
    }
}

impl core::error::Error for SchedulerError {}

/// `linspace(1, 0, n)` in f32, bit-identical to `torch.linspace`.
///
/// The rectified-flow sigma grid is collapsed with a consecutive-duplicate
/// check, so a one-ulp difference can change how many sigmas survive and
/// therefore the number of model evaluations.
///
/// ATen's kernel takes an f32 step, splits the range at the halfway point to
/// keep the two ends symmetric, and evaluates `start + step * i` with a fused
/// multiply-add (a single rounding). Computing in f64 from the f32 step and
/// narrowing once reproduces that FMA.
fn linspace_1_to_0(n: usize) -> Vec<f32> {
    let (start, end) = (1.0f64, 0.0f64);
    let step = ((end - start) as f32 / (n - 1) as f32) as f64;
    let half = n / 2;
    (0..n)
        .map(|i| {
            if i < half {
                (start + step * i as f64) as f32
            } else {
                (end - step * (n - 1 - i) as f64) as f32
            }
        })
        .collect()
}

/// Rectified-flow Euler scheduler with exponential sigma shift.
///
/// `sigma' = s*sigma / (1 + (s-1)*sigma)` over a `linspace(1, 0, N)` grid. The
/// terminal `0` is part of the grid (the shift maps 0 to exactly 0), so `N`
/// sigmas drive `N - 1` model evaluations and `timesteps = 1 - sigmas[:-1]`.
#[derive(Clone, Debug)]
pub struct Scheduler {
    shift: f32,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
}

impl Scheduler {
    pub fn new(shift: f32, num_inference_steps: usize) -> Result<Self, SchedulerError> {
        if shift <= 0.0 {
            return Err(SchedulerError::NonPositiveShift(shift));
        }
        if num_inference_steps < 2 {
            return Err(SchedulerError::TooFewSteps(num_inference_steps));
        }

        let base = linspace_1_to_0(num_inference_steps);
        let shift_m1 = shift - 1.0;
        let mut sigmas: Vec<f32> = Vec::with_capacity(base.len());
        for b in base {
            let v = (shift * b) / (1.0 + shift_m1 * b);
            // The shift compresses the grid near sigma = 1; collapse the f32
            // collisions it creates. This CHANGES the effective step count.
            if sigmas.last() != Some(&v) {
                sigmas.push(v);
            }
        }
        let timesteps = sigmas[..sigmas.len() - 1].iter().map(|s| 1.0 - s).collect();
        Ok(Self {
            shift,
            sigmas,
            timesteps,
        })
    }

    /// Build from an explicit strictly-decreasing sigma grid ending at 0.
    pub fn from_sigmas(sigmas: Vec<f32>) -> Result<Self, SchedulerError> {
        if sigmas.len() < 2
            || !sigmas.windows(2).all(|w| w[1] < w[0])
            || sigmas.last() != Some(&0.0)
        {
            return Err(SchedulerError::TooFewSteps(sigmas.len()));
        }
        let timesteps = sigmas[..sigmas.len() - 1].iter().map(|s| 1.0 - s).collect();
        Ok(Self {
            shift: f32::NAN,
            sigmas,
            timesteps,
        })
    }

    pub fn shift(&self) -> f32 {
        self.shift
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// The timestep the transformer is conditioned on at each step.
    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    /// Model evaluations this schedule drives.
    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    /// Coefficients for the Euler update at `step_index`.
    pub fn step_coeffs(&self, step_index: usize) -> Result<StepCoeffs, SchedulerError> {
        if step_index >= self.timesteps.len() {
            return Err(SchedulerError::StepOutOfRange(
                step_index,
                self.timesteps.len(),
            ));
        }
        let ratio = self.sigmas[step_index + 1] / self.sigmas[step_index];
        Ok(StepCoeffs {
            ratio,
            one_minus_ratio: 1.0 - ratio,
            // Recovered from the timestep, NOT from the sigma grid: see the
            // module note. `1 - (1 - sigma)` is not exact below 0.5.
            sigma_from_timestep: 1.0 - self.timesteps[step_index],
        })
    }

    /// Rectified-flow forward process in H3's `t` convention:
    /// `x_t = t*x0 + (1 - t)*noise`.
    ///
    /// Used to noise conditioning anchors, where `t` is a noise-augmentation
    /// level rather than a schedule entry, so it is taken at face value and
    /// never looked up in the schedule.
    pub fn scale_noise_coeffs(timestep: f32) -> (f32, f32) {
        (timestep, 1.0 - timestep)
    }
}

/// The video and audio schedules of one run. Both are stepped in lockstep: they
/// have the same number of model evaluations, driven by one shared forward.
#[derive(Clone, Debug)]
pub struct JointSchedule {
    pub video: Scheduler,
    pub audio: Scheduler,
}

impl JointSchedule {
    pub fn new(num_inference_steps: usize) -> Result<Self, SchedulerError> {
        Ok(Self {
            video: Scheduler::new(super::SIGMA_SHIFT_VIDEO, num_inference_steps)?,
            audio: Scheduler::new(super::SIGMA_SHIFT_AUDIO, num_inference_steps)?,
        })
    }

    /// Model evaluations. The two schedules can collapse different numbers of
    /// duplicate sigmas, so the run is bounded by the shorter one.
    pub fn num_steps(&self) -> usize {
        self.video.num_steps().min(self.audio.num_steps())
    }

    /// Every distinct noise level the run will present to the transformer,
    /// sorted. This is exactly what the AdaLN modulation table is built over.
    ///
    /// `conditioning_levels` are the fixed levels conditioning rows sit at (the
    /// keyframe noise-aug level, and 1.0 for reference audio); pass an empty
    /// slice for t2va.
    pub fn distinct_timesteps(&self, conditioning_levels: &[f32]) -> Vec<f32> {
        let n = self.num_steps();
        let mut all: Vec<f32> = self.video.timesteps()[..n]
            .iter()
            .chain(&self.audio.timesteps()[..n])
            .chain(conditioning_levels)
            .copied()
            .collect();
        all.sort_by(|a, b| a.partial_cmp(b).expect("timesteps are finite"));
        all.dedup();
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minimax::{SIGMA_SHIFT_AUDIO, SIGMA_SHIFT_VIDEO};

    #[test]
    fn linspace_endpoints_are_exact() {
        for n in [2usize, 3, 4, 16, 50] {
            let g = linspace_1_to_0(n);
            assert_eq!(g.len(), n);
            assert_eq!(g[0], 1.0);
            assert_eq!(g[n - 1], 0.0);
            assert!(g.windows(2).all(|w| w[1] < w[0]), "n={n} must decrease");
        }
    }

    #[test]
    fn shift_maps_endpoints_to_themselves() {
        let s = Scheduler::new(SIGMA_SHIFT_VIDEO, 5).unwrap();
        assert_eq!(*s.sigmas().first().unwrap(), 1.0);
        assert_eq!(*s.sigmas().last().unwrap(), 0.0);
    }

    #[test]
    fn n_sigmas_drive_n_minus_1_evaluations() {
        let s = Scheduler::new(SIGMA_SHIFT_VIDEO, 5).unwrap();
        assert_eq!(s.num_steps(), s.sigmas().len() - 1);
        assert_eq!(s.timesteps().len(), s.num_steps());
    }

    #[test]
    fn timesteps_are_one_minus_sigma() {
        let s = Scheduler::new(SIGMA_SHIFT_AUDIO, 8).unwrap();
        for (i, t) in s.timesteps().iter().enumerate() {
            assert_eq!(*t, 1.0 - s.sigmas()[i]);
        }
    }

    /// A larger shift compresses the grid toward sigma = 1, so the video
    /// schedule holds high noise for longer than the audio one.
    #[test]
    fn larger_shift_holds_high_noise_longer() {
        let v = Scheduler::new(SIGMA_SHIFT_VIDEO, 8).unwrap();
        let a = Scheduler::new(SIGMA_SHIFT_AUDIO, 8).unwrap();
        assert!(
            v.sigmas()[1] > a.sigmas()[1],
            "video shift 12 > audio shift 3"
        );
    }

    #[test]
    fn rejects_bad_construction() {
        assert!(Scheduler::new(0.0, 4).is_err());
        assert!(Scheduler::new(-1.0, 4).is_err());
        assert!(Scheduler::new(SIGMA_SHIFT_VIDEO, 1).is_err());
    }

    /// The Euler blend must be a convex combination that lands on the sigma grid.
    #[test]
    fn step_coeffs_are_a_convex_blend() {
        let s = Scheduler::new(SIGMA_SHIFT_VIDEO, 4).unwrap();
        for i in 0..s.num_steps() {
            let c = s.step_coeffs(i).unwrap();
            assert!((0.0..=1.0).contains(&c.ratio), "step {i} ratio {}", c.ratio);
            assert!((c.ratio + c.one_minus_ratio - 1.0).abs() < 1e-6);
        }
        // The last step lands exactly on sigma = 0, so it fully takes x0.
        let last = s.step_coeffs(s.num_steps() - 1).unwrap();
        assert_eq!(last.ratio, 0.0);
        assert_eq!(last.one_minus_ratio, 1.0);
    }

    #[test]
    fn step_index_is_bounds_checked() {
        let s = Scheduler::new(SIGMA_SHIFT_VIDEO, 4).unwrap();
        assert!(s.step_coeffs(s.num_steps()).is_err());
    }

    /// The velocity is data-ward: x0 = x_t + (1 - t) * v, and at the final step
    /// the update takes x0 outright.
    #[test]
    fn final_step_takes_the_denoised_estimate() {
        let s = Scheduler::new(SIGMA_SHIFT_VIDEO, 4).unwrap();
        let c = s.step_coeffs(s.num_steps() - 1).unwrap();
        let (x_t, v) = (0.3f32, 2.0f32);
        let x0 = x_t + c.sigma_from_timestep * v;
        let prev = c.ratio * x_t + c.one_minus_ratio * x0;
        assert_eq!(prev, x0);
    }

    #[test]
    fn joint_schedule_runs_both_streams_in_lockstep() {
        let j = JointSchedule::new(4).unwrap();
        assert_eq!(j.video.shift(), SIGMA_SHIFT_VIDEO);
        assert_eq!(j.audio.shift(), SIGMA_SHIFT_AUDIO);
        assert!(j.num_steps() >= 1);
        assert!(j.num_steps() <= j.video.num_steps());
        assert!(j.num_steps() <= j.audio.num_steps());
    }

    /// The modulation table is built over this set, so it must cover every level
    /// either stream presents, sorted and deduplicated.
    #[test]
    fn distinct_timesteps_cover_both_streams() {
        let j = JointSchedule::new(4).unwrap();
        let d = j.distinct_timesteps(&[]);
        assert!(d.windows(2).all(|w| w[0] < w[1]), "sorted and deduped");
        for t in j.video.timesteps().iter().chain(j.audio.timesteps()) {
            assert!(d.contains(t), "missing {t}");
        }
    }

    #[test]
    fn distinct_timesteps_include_conditioning_levels() {
        let j = JointSchedule::new(4).unwrap();
        let d = j.distinct_timesteps(&[0.999, 1.0]);
        assert!(d.contains(&0.999));
        assert!(d.contains(&1.0));
    }

    /// A 4-step turbo run must actually produce 4 evaluations: the duplicate
    /// collapse near sigma = 1 is exactly where a step silently goes missing.
    #[test]
    fn four_step_turbo_gives_four_evaluations() {
        let j = JointSchedule::new(5).unwrap();
        assert_eq!(j.num_steps(), 4, "5 sigmas -> 4 model evaluations");
    }

    #[test]
    fn scale_noise_is_a_convex_blend() {
        let (a, b) = Scheduler::scale_noise_coeffs(0.999);
        assert_eq!(a, 0.999);
        assert!((a + b - 1.0).abs() < 1e-6);
    }
}
