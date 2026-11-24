//! The audio kind: speech ([`crate::tts`]) and sound effects
//! ([`crate::sfx`]). One loaded audio model may stay between audio jobs
//! (opt-in, `warm_audio_secs`); another audio model replaces it, and any
//! other job frees it.

use std::sync::Arc;
use std::time::Duration;

use thinfer_core::backend::WgpuBackend;

use crate::model::{AudioKind, AudioModelId};
use crate::progress::{ProgressSink, Stage};
use crate::request::{AudioRequest, JobSummary};
use crate::sfx::WarmSfx;
use crate::tts::WarmTts;

/// A loaded audio model.
pub enum WarmAudio {
    Speech(Box<WarmTts>),
    Sound(Box<WarmSfx>),
}

impl WarmAudio {
    pub fn model(&self) -> AudioModelId {
        match self {
            Self::Speech(w) => w.model(),
            Self::Sound(w) => w.model(),
        }
    }

    /// Time since a job last used it.
    pub fn idle(&self) -> Duration {
        match self {
            Self::Speech(w) => w.idle(),
            Self::Sound(w) => w.idle(),
        }
    }
}

/// Run `req` on the model in `warm` when it is the same one, else load it
/// there (freeing any other first). The caller decides whether `warm`
/// outlives the job.
pub async fn run(
    backend: &Arc<WgpuBackend>,
    req: &AudioRequest,
    sink: &dyn ProgressSink,
    warm: &mut Option<WarmAudio>,
) -> Result<JobSummary, String> {
    req.validate()?;
    sink.stage(Stage::Loading);
    if warm.as_ref().is_some_and(|w| w.model() != req.model) {
        *warm = None;
    }
    match req.model.kind() {
        AudioKind::Speech => {
            let mut m = match warm.take() {
                Some(WarmAudio::Speech(w)) => w,
                _ => Box::new(WarmTts::load(backend, req.model, req.budget).await?),
            };
            let out = crate::tts::generate(backend, &m, req, sink).await;
            m.touch();
            *warm = Some(WarmAudio::Speech(m));
            out
        }
        AudioKind::Sound => {
            let mut m = match warm.take() {
                Some(WarmAudio::Sound(w)) => w,
                _ => Box::new(WarmSfx::load(backend, req.model, req.budget).await?),
            };
            let out = m.generate(req, sink).await;
            m.touch();
            *warm = Some(WarmAudio::Sound(m));
            out
        }
    }
}
