//! Speech generation (Qwen3-TTS): text (+ a voice description, or a voice
//! clip and its transcript) to a 24 kHz mono WAV. The text, description,
//! transcript and clip are job data: never logged, wiped after use. The
//! loaded model ([`WarmTts`]) is pristine model data only, so a server may
//! keep it between audio jobs (opt-in).

use std::sync::Arc;
use std::time::{Duration, Instant};

use thinfer_core::backend::WgpuBackend;
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::tokenizer::Tokenizer;
use thinfer_core::workspace::Workspace;
use thinfer_models::llm::decoder::LlmError;
use thinfer_models::llm::kernels::LlmPipelines;
use thinfer_models::qwen3_tts::codec::{CodecDecoder, CodecEncoder};
use thinfer_models::qwen3_tts::manifest::role;
use thinfer_models::qwen3_tts::prompt::{self, IclRef, PromptSpec};
use thinfer_models::qwen3_tts::speaker::{self, SpeakerEncoder};
use thinfer_models::qwen3_tts::talker::{FrameSink, GenSpec, Talker, TalkerWeights};
use thinfer_models::qwen3_tts::{CODE_GROUPS, FRAME_RATE, SAMPLE_RATE, TtsSampling, language_id};
use thinfer_native::MmapFileOpener;
use thinfer_native::tokenizer::HfTokenizer;
use zeroize::Zeroizing;

use crate::config::random_seed;
use crate::download::resolve_role;
use crate::minimax::open_mmap;
use crate::model::{AudioModelEngine, AudioModelId};
use crate::progress::{ProgressSink, Stage};
use crate::request::{AudioRequest, AudioStats, JobSummary};

/// The longest reference clip used (seconds): the reference clones from a
/// few seconds; longer clips only lengthen the prompt.
const MAX_REF_SECONDS: f64 = 30.0;

type Src = SafetensorsSource<MmapFileOpener>;

struct Progress<'a> {
    sink: &'a dyn ProgressSink,
}

impl FrameSink for Progress<'_> {
    fn frame(&mut self, frames: u32) -> bool {
        self.sink.stage(Stage::Audio {
            ms: frames * (1000.0 / FRAME_RATE) as u32,
        });
        !self.sink.cancelled()
    }
}

/// 16-bit PCM WAV bytes of mono `samples` (clamped to [-1, 1]).
pub fn wav_bytes(samples: &[f32], rate: u32) -> Vec<u8> {
    let data = (samples.len() * 2) as u32;
    let mut b = Vec::with_capacity(44 + data as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes()); // PCM
    b.extend_from_slice(&1u16.to_le_bytes()); // mono
    b.extend_from_slice(&rate.to_le_bytes());
    b.extend_from_slice(&(rate * 2).to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data.to_le_bytes());
    for &s in samples {
        b.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    b
}

/// The reference clip as 24 kHz mono (channels averaged).
fn reference_samples(clip: &[u8]) -> Result<Zeroizing<Vec<f32>>, String> {
    let decoded = crate::audio_in::decode(clip)?;
    let (stereo, n) = crate::audio_in::prepare(decoded, speaker::SAMPLE_RATE, MAX_REF_SECONDS);
    let stereo = Zeroizing::new(stereo);
    if n == 0 {
        return Err("the reference clip has no audio".into());
    }
    Ok(Zeroizing::new(
        (0..n).map(|i| 0.5 * (stereo[i] + stereo[n + i])).collect(),
    ))
}

fn lerr(e: LlmError) -> String {
    format!("{e:?}")
}

/// A loaded speech model: tokenizer, kernels and every weight.
pub struct WarmTts {
    model: AudioModelId,
    tok: HfTokenizer,
    model_res: WeightResidency<Src>,
    codec_res: WeightResidency<Src>,
    p: LlmPipelines,
    weights: TalkerWeights,
    dec: CodecDecoder,
    /// Voice-clone models: the clip's codes and its x-vector.
    enc: Option<CodecEncoder>,
    spk: Option<SpeakerEncoder>,
    used: Instant,
}

impl WarmTts {
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
        let open = |r: &'static str| async move {
            SafetensorsSource::open(open_mmap(&path(r)?).await?)
                .await
                .map_err(|e| format!("parse {r}: {e:?}"))
        };
        let model_res = WeightResidency::new(open(role::MODEL).await?, budget);
        let codec_res = WeightResidency::new(open(role::CODEC).await?, budget);
        let p = LlmPipelines::compile(backend)
            .await
            .map_err(|e| format!("pipelines: {e:?}"))?;
        let weights = TalkerWeights::load(backend, &model_res)
            .await
            .map_err(lerr)?;
        let dec = CodecDecoder::load(backend, &codec_res)
            .await
            .map_err(lerr)?;
        let (enc, spk) = if model.takes_voice() {
            (
                Some(
                    CodecEncoder::load(backend, &codec_res)
                        .await
                        .map_err(lerr)?,
                ),
                Some(SpeakerEncoder::load(model_res.source()).await?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            model,
            tok,
            model_res,
            codec_res,
            p,
            weights,
            dec,
            enc,
            spk,
            used: Instant::now(),
        })
    }
}

pub(crate) async fn generate(
    backend: &Arc<WgpuBackend>,
    m: &WarmTts,
    req: &AudioRequest,
    sink: &dyn ProgressSink,
) -> Result<JobSummary, String> {
    let ids = |s: String| -> Result<Zeroizing<Vec<u32>>, String> {
        let s = Zeroizing::new(s);
        Ok(Zeroizing::new(
            m.tok
                .encode(&s, false)
                .map_err(|e| format!("tokenize: {e:?}"))?,
        ))
    };
    let text_ids = ids(format!(
        "<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n",
        req.text
    ))?;
    let instruct_ids = req
        .instruct
        .as_ref()
        .map(|i| ids(format!("<|im_start|>user\n{i}<|im_end|>\n")))
        .transpose()?;
    let ref_ids = req
        .ref_text
        .as_ref()
        .map(|t| ids(format!("<|im_start|>assistant\n{t}<|im_end|>\n")))
        .transpose()?;
    // This job's buffers: none outlive it.
    let ws = Workspace::new(backend.clone(), Arc::clone(m.model_res.arbiter()));

    // The voice: the clip's x-vector, and its codes for in-context cloning.
    let mut speaker_vec: Option<Zeroizing<Vec<f32>>> = None;
    let mut ref_codes: Zeroizing<Vec<[u32; CODE_GROUPS]>> = Zeroizing::new(Vec::new());
    if let Some(clip) = &req.ref_audio {
        let (Some(spk), Some(enc)) = (&m.spk, &m.enc) else {
            return Err(format!("{} takes no reference clip", req.model));
        };
        let wav = reference_samples(&clip.0)?;
        speaker_vec = Some(Zeroizing::new(spk.embed(&wav)));
        if ref_ids.is_some() {
            ref_codes = Zeroizing::new(
                enc.encode(backend, &m.codec_res, &m.p, &ws, &wav, None)
                    .await
                    .map_err(lerr)?,
            );
        }
    }

    let language = language_id(&req.language).ok_or("unknown language")?;
    let pr = prompt::build(&PromptSpec {
        text_ids: &text_ids,
        instruct_ids: instruct_ids.as_ref().map(|v| v.as_slice()),
        language,
        speaker: speaker_vec.is_some(),
        icl: ref_ids.as_ref().map(|r| IclRef {
            text_ids: r,
            frames: ref_codes.len(),
        }),
        // The reference wrappers' defaults: a described voice reads the text
        // up front, a cloned one streams it.
        non_streaming: req.model == AudioModelId::Qwen3TtsDesign,
    })?;

    let seed = req.seed.unwrap_or_else(random_seed);
    let d = TtsSampling::default();
    let sampling = TtsSampling {
        temperature: req.temperature.unwrap_or(d.temperature),
        top_k: req.top_k.unwrap_or(d.top_k),
        top_p: req.top_p.unwrap_or(d.top_p),
        ..d
    };
    tracing::info!(
        target: thinfer_core::trace::DIAG,
        model = %req.model, language = %req.language, seed,
        clone = req.ref_audio.is_some(), icl = !ref_codes.is_empty(),
        max_seconds = req.max_seconds, "generate start",
    );
    let talker = Talker {
        w: &m.weights,
        residency: &m.model_res,
        backend,
        p: &m.p,
        ws: &ws,
    };
    let mut progress = Progress { sink };
    let t0 = Instant::now();
    let out = talker
        .generate(
            &GenSpec {
                prompt: &pr,
                speaker: speaker_vec.as_ref().map(|v| v.as_slice()),
                ref_codes: &ref_codes,
                sampling,
                seed,
                max_frames: (req.max_seconds * FRAME_RATE).ceil() as u32,
                tap_logits: 0,
                force: None,
            },
            &mut progress,
        )
        .await
        .map_err(lerr)?;
    let gen_secs = t0.elapsed().as_secs_f64();
    if out.codes.is_empty() {
        return Err("the model produced no speech".into());
    }

    // Decode the reference frames with the new ones (the codec is causal:
    // the reference is the new frames' left context), then cut them off.
    sink.stage(Stage::VaeDecode);
    let mut all = Zeroizing::new(ref_codes.to_vec());
    all.extend_from_slice(&out.codes);
    let wav = Zeroizing::new(
        m.dec
            .decode(backend, &m.codec_res, &m.p, &ws, &all)
            .await
            .map_err(lerr)?,
    );
    let cut = ref_codes.len() * wav.len() / all.len();
    let speech = &wav[cut..];
    let seconds = speech.len() as f32 / SAMPLE_RATE as f32;
    sink.note(&format!(
        "{seconds:.1} s of speech in {:.1} s ({} frames, {:.0} ms/frame)",
        t0.elapsed().as_secs_f64(),
        out.codes.len(),
        gen_secs * 1e3 / out.codes.len() as f64,
    ));
    crate::progress::write_artifact(sink, &req.output, wav_bytes(speech, SAMPLE_RATE))?;
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
