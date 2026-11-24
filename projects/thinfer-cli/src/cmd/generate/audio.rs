//! `thinfer generate audio`: speech in a described voice or a reference
//! clip's voice, or a sound effect from a description, to a WAV. Thin clap
//! adapter over `thinfer_app::AudioRequest`.

use std::path::PathBuf;

use clap::Args;
use thinfer_app::config::ResidencyBudget;
use thinfer_app::model::{AUDIO_DEFAULT_RAM, AUDIO_DEFAULT_VRAM, AudioKind, AudioModelId};
use thinfer_app::request::{AudioBytes, AudioRequest, SoundParams};
use thinfer_app::wire::{AudioSpec, JobSpec};
use thinfer_app::{JobRequest, parse_budget};

#[derive(Args)]
pub struct GenerateAudio {
    /// `qwen3-tts-design` (a voice from `--instruct`), `qwen3-tts` (the
    /// voice of `--ref-audio`) or `moss-sfx` (the sound `--text` describes).
    #[arg(long, default_value_t = AudioModelId::DEFAULT, value_enum)]
    pub model: AudioModelId,
    /// What to say (speech) or the sound to make (sound effects).
    #[arg(long)]
    pub text: String,
    /// The voice and delivery in words ("a warm young woman; whispering").
    /// Voice design models.
    #[arg(long)]
    pub instruct: Option<String>,
    /// The voice to speak in (WAV, FLAC, MP3, M4A, Ogg). Voice clone models.
    #[arg(long)]
    pub ref_audio: Option<PathBuf>,
    /// The reference clip's transcript: its delivery carries over too.
    #[arg(long, requires = "ref_audio")]
    pub ref_text: Option<String>,
    /// `auto` or a language name (`english`, `chinese`, `japanese`, ...).
    #[arg(long, default_value = "auto")]
    pub language: String,
    /// Longest output in seconds.
    #[arg(long, default_value_t = 60.0)]
    pub max_seconds: f32,
    #[arg(long)]
    pub seed: Option<u64>,
    #[arg(long)]
    pub temperature: Option<f32>,
    #[arg(long)]
    pub top_k: Option<u32>,
    #[arg(long)]
    pub top_p: Option<f32>,
    /// Sound: the clip's length in seconds (default 10, up to 30).
    #[arg(long)]
    pub seconds: Option<f32>,
    /// Sound: make a seamless loop (its end runs into its start).
    #[arg(long = "loop")]
    pub looped: bool,
    /// Sound: denoising steps (default 100; fewer is faster and rougher).
    #[arg(long)]
    pub steps: Option<u32>,
    /// Sound: classifier-free guidance (default 4).
    #[arg(long)]
    pub guidance: Option<f32>,
    /// Output WAV path.
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long)]
    pub ram_budget: Option<String>,
    #[arg(long)]
    pub vram_budget: Option<String>,
    #[arg(long, default_value_t = false)]
    pub download_as_needed: bool,
    #[command(flatten)]
    pub remote: super::RemoteArgs,
}

pub async fn run_audio(args: GenerateAudio) -> Result<(), String> {
    if args.remote.remote.is_some() {
        let spec = JobSpec::Audio(AudioSpec {
            model: Some(args.model),
            text: args.text,
            instruct: args.instruct,
            ref_audios: args
                .ref_audio
                .as_ref()
                .map(|p| super::video::base64_file(p).map(|b| vec![b]))
                .transpose()?,
            ref_text: args.ref_text,
            language: Some(args.language),
            max_seconds: Some(args.max_seconds),
            seed: args.seed,
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
            seconds: args.seconds,
            looped: args.looped.then_some(true),
            steps: args.steps,
            guidance: args.guidance,
        });
        return super::run_remote(&args.remote, spec, args.output).await;
    }
    let ram = parse_budget(
        "--ram-budget",
        Some(args.ram_budget.as_deref().unwrap_or(AUDIO_DEFAULT_RAM)),
    )?;
    let vram = parse_budget(
        "--vram-budget",
        Some(args.vram_budget.as_deref().unwrap_or(AUDIO_DEFAULT_VRAM)),
    )?;
    let ref_audio = args
        .ref_audio
        .as_ref()
        .map(|p| {
            std::fs::read(p)
                .map(AudioBytes)
                .map_err(|e| format!("read {}: {e}", p.display()))
        })
        .transpose()?;
    let sound_fields =
        args.seconds.is_some() || args.looped || args.steps.is_some() || args.guidance.is_some();
    let sound = (args.model.kind() == AudioKind::Sound || sound_fields)
        .then(|| SoundParams::resolve(args.seconds, Some(args.looped), args.steps, args.guidance));
    let req = AudioRequest {
        model: args.model,
        sound,
        text: args.text,
        instruct: args.instruct,
        ref_audio,
        ref_text: args.ref_text,
        language: args.language,
        seed: args.seed,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        max_seconds: args.max_seconds,
        budget: ResidencyBudget {
            ram_bytes: ram,
            vram_bytes: vram,
        },
        output: args.output,
    };
    req.validate()?;
    let files = req.required_files();
    super::run_job(
        JobRequest::Audio(req),
        &files,
        args.download_as_needed,
        ram,
        vram,
    )
    .await
}
