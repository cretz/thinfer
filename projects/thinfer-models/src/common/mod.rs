//! Cross-model primitives shared by the per-model pipelines (z_image, wan, ...).
//!
//! These are the model-agnostic building blocks: the DiT transformer block and
//! its op/pipeline scaffolding (`block`), the bias-Linear helpers and their
//! residency views (`embedders`), the residency-aware weight registration
//! primitives (`loader`), the CPU RoPE table (`rope_embedder`), and the
//! activation dtype / mask / readback helpers (`seq`). Per-model glue
//! (patchify, weight-name maps, the model's own embedders) stays in the model
//! module.

pub mod autoencoder_kl;
pub mod bigvgan;
pub mod block;
pub mod convrot;
pub mod embedders;
pub mod loader;
pub mod lora;
pub mod merge;
pub mod rope_embedder;
pub mod seq;
pub mod vae_taehv;
pub mod vae_tiling;

/// Cooperative cancellation hook. Returns `true` when the caller wants the
/// in-flight generate to stop; the denoise loops poll it at step (and, where a
/// step is long, block) boundaries and bail with their `Cancelled` error.
/// `None` (CLI default) never cancels. The server passes a closure backed by
/// the job's cancel flag.
pub type Cancel<'a> = Option<&'a dyn Fn() -> bool>;

/// A denoise between steps: the latent entering `step` (`step` equal to the
/// step count: denoised, before the decode). What a pipeline that recomputes
/// its conditioning on resume needs to continue.
pub struct StepState {
    pub step: usize,
    pub latent: Vec<f32>,
    /// A multistep solver's history, when the sampler keeps one (empty
    /// otherwise).
    pub solver: SolverState,
}

impl StepState {
    /// The latent entering `step`, for a sampler with no history.
    pub fn at(step: usize, latent: Vec<f32>) -> Self {
        Self {
            step,
            latent,
            solver: SolverState::default(),
        }
    }
}

/// A multistep solver's state between steps: its latent-sized history and
/// its counters, in the solver's own order.
#[derive(Default)]
pub struct SolverState {
    pub vecs: Vec<Option<Vec<f32>>>,
    pub ints: Vec<u64>,
}

/// Stopping a generate at a step boundary and resuming it there.
#[derive(Default)]
pub struct StepControl<'a> {
    /// Where a stopped run stood.
    pub resume: Option<StepState>,
    /// Polled at each step boundary (between steps, and before the decode):
    /// stop there.
    pub stop_here: Cancel<'a>,
}

/// How a stoppable generate ended.
pub enum Generated<T> {
    Done(T),
    Stopped(StepState),
}
