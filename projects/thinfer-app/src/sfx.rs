//! Sound effects (MOSS-SoundEffect v2.0): a description to a 48 kHz mono
//! WAV, optionally a seamless loop. The description is job data: never
//! logged, wiped after use. The loaded model ([`WarmSfx`]) is pristine model
//! data only, so a server may keep it between audio jobs (opt-in).

use std::sync::Arc;
use std::time::{Duration, Instant};

use thinfer_core::backend::WgpuBackend;
use thinfer_core::format::pytorch::PytorchSource;
use thinfer_core::format::safetensors::{SafetensorsSource, ShardedSafetensorsSource};
use thinfer_core::format::union::UnionSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::tokenizer::Tokenizer;
use thinfer_models::common::weight_norm::WeightNormSource;
use thinfer_models::moss_sfx::manifest::role;
use thinfer_models::moss_sfx::pipeline::{SfxError, SfxModel, SfxParams};
use thinfer_models::moss_sfx::{SAMPLE_RATE, prompt_text};
use thinfer_native::MmapFileOpener;
use thinfer_native::tokenizer::HfTokenizer;
use zeroize::Zeroizing;

use crate::config::random_seed;
use crate::download::resolve_role;
use crate::minimax::open_mmap;
use crate::model::{AudioModelEngine, AudioModelId};
use crate::progress::{ProgressSink, Stage};
use crate::request::{AudioRequest, AudioStats, JobSummary};
use crate::tts::wav_bytes;

type Src = UnionSource<
    UnionSource<SafetensorsSource<MmapFileOpener>, ShardedSafetensorsSource<MmapFileOpener>>,
    WeightNormSource<PytorchSource<MmapFileOpener>>,
>;

fn serr(e: SfxError) -> String {
    match e {
        SfxError::Cancelled => "cancelled".into(),
        e => format!("{e:?}"),
    }
}

/// A loaded sound model: tokenizer, kernels and every weight registered.
pub struct WarmSfx {
    model: AudioModelId,
    tok: HfTokenizer,
    sfx: SfxModel<Src>,
    used: Instant,
}

impl WarmSfx {
    pub fn model(&self) -> AudioModelId {
        self.model
    }

    /// Time since a job last used it.
    pub fn idle(&self) -> Duration {
        self.used.elapsed()
    }

    pub fn touch(&mut self) {
        self.used = Instant::now();
    }

    pub async fn load(
        backend: &Arc<WgpuBackend>,
        model: AudioModelId,
        budget: ResidencyBudget,
    ) -> Result<Self, String> {
        let m = model.manifest();
        let path = |r: &str| resolve_role(m, r);
        let tok = HfTokenizer::from_bpe_files(
            path(role::VOCAB)?,
            path(role::MERGES)?,
            path(role::TOKENIZER_CONFIG)?,
        )
        .await
        .map_err(|e| format!("tokenizer: {e:?}"))?;
        let dit = SafetensorsSource::open(open_mmap(&path(role::DIT)?).await?)
            .await
            .map_err(|e| format!("parse dit: {e:?}"))?;
        let text = ShardedSafetensorsSource::open(vec![
            open_mmap(&path(role::TEXT_SHARD_1)?).await?,
            open_mmap(&path(role::TEXT_SHARD_2)?).await?,
        ])
        .await
        .map_err(|e| format!("parse text encoder: {e:?}"))?;
        let vae = PytorchSource::open(open_mmap(&path(role::VAE)?).await?)
            .await
            .map_err(|e| format!("parse vae: {e:?}"))?;
        let src = UnionSource::new(UnionSource::new(dit, text), WeightNormSource::new(vae));
        let residency = WeightResidency::new(src, budget);
        let sfx = SfxModel::load(backend.clone(), residency)
            .await
            .map_err(serr)?;
        Ok(Self {
            model,
            tok,
            sfx,
            used: Instant::now(),
        })
    }

    pub async fn generate(
        &self,
        req: &AudioRequest,
        sink: &dyn ProgressSink,
    ) -> Result<JobSummary, String> {
        let s = req.sound.ok_or("a sound job needs its sound settings")?;
        let text = Zeroizing::new(prompt_text(&req.text, s.seconds));
        let ids = Zeroizing::new(
            self.tok
                .encode(&text, false)
                .map_err(|e| format!("tokenize: {e:?}"))?,
        );
        let seed = req.seed.unwrap_or_else(random_seed);
        tracing::info!(
            target: thinfer_core::trace::DIAG,
            model = %req.model, seconds = s.seconds, looped = s.looped, steps = s.steps,
            guidance = s.guidance, seed, prompt_tokens = ids.len(), "generate start",
        );
        sink.stage(Stage::TextEncode);
        let t0 = Instant::now();
        let mut on_step = |i: u32, n: u32| {
            sink.stage(Stage::Step { i, n });
            !sink.cancelled()
        };
        let params = SfxParams {
            text_ids: &ids,
            seconds: s.seconds,
            steps: s.steps,
            guidance: s.guidance,
            seed,
            looped: s.looped,
            noise: None,
        };
        let x = Zeroizing::new(
            self.sfx
                .denoise(&params, &mut on_step)
                .await
                .map_err(serr)?,
        );
        let denoise_secs = t0.elapsed().as_secs_f64();
        sink.stage(Stage::VaeDecode);
        let wav = Zeroizing::new(
            self.sfx
                .decode_clip(&x, s.seconds, s.looped)
                .await
                .map_err(serr)?,
        );
        let seconds = wav.len() as f32 / SAMPLE_RATE as f32;
        sink.note(&format!(
            "{seconds:.1} s of sound in {:.1} s (denoise {denoise_secs:.1} s, {} steps)",
            t0.elapsed().as_secs_f64(),
            s.steps,
        ));
        crate::progress::write_artifact(sink, &req.output, wav_bytes(&wav, SAMPLE_RATE))?;
        Ok(JobSummary {
            output: req.output.clone(),
            seed: Some(seed),
            audio: Some(AudioStats {
                seconds,
                sample_rate: SAMPLE_RATE,
            }),
            ..Default::default()
        })
    }
}
