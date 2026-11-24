//! The model catalog `GET /models` serves: every model this build runs, what
//! it takes (images, audio clips, a video; how a prompt names them), what it
//! makes, what decides its output size, and its video limits. Built from the
//! same per-model rules the request validation enforces, so the two cannot
//! drift; the server adds whether each one's weights are already local (from
//! [`Entry::spec`]).

use thinfer_client::wire::{
    AudioInfo, AudioKindInfo, AudioSpec, ImageSpec, InputSlot, InputUse, JobSpec, Media, ModelInfo,
    ModelKind, SizeRules, TextMessage, TextSpec, VideoInfo, VideoSpec,
};

use crate::model::{
    AudioKind, AudioModelId, ImageKind, ImageModelEngine, ImageModelId, SwapModel, SwapModelEngine,
    TextModelId, VideoModelId,
};

/// One model: its description, and the default job whose weight files say
/// whether it is downloaded (`None`: see [`Entry::files`]).
pub struct Entry {
    pub info: ModelInfo,
    pub spec: Option<JobSpec>,
    /// The files a job needs, for a model whose default job cannot be built
    /// without uploads (face swap).
    pub files: Vec<thinfer_core::manifest::FileRef>,
}

/// The image decoders this build carries.
const IMAGE_FORMATS: &[&str] = &["png", "jpg", "jpeg"];

fn slot(
    field: &str,
    media: Media,
    used_as: InputUse,
    (min, max): (u32, u32),
    tag: Option<&str>,
    notes: &str,
) -> InputSlot {
    let formats: &[&str] = match media {
        Media::Image => IMAGE_FORMATS,
        Media::Audio => crate::audio_in::FORMATS,
        Media::Video => &["mp4"],
        Media::Text => &[],
    };
    InputSlot {
        field: field.into(),
        media,
        formats: formats.iter().map(|f| (*f).into()).collect(),
        used_as,
        min,
        max,
        prompt_tag: tag.map(Into::into),
        notes: notes.into(),
    }
}

fn image(m: ImageModelId) -> Entry {
    let d = m.defaults();
    let (inputs, follows) = match m.kind() {
        ImageKind::QwenImageEdit => (
            vec![slot(
                "inputImages",
                Media::Image,
                InputUse::Edit,
                (1, crate::preprocess::MAX_EDIT_IMAGES as u32),
                Some("Picture {n}"),
                "The image(s) the prompt edits or combines, in order: \"Picture 1\" is the first.",
            )],
            false,
        ),
        ImageKind::QwenImage21 => (
            vec![slot(
                "inputImages",
                Media::Image,
                InputUse::Condition,
                (
                    0,
                    thinfer_models::qwen_image21::config::MAX_CONDITION_IMAGES as u32,
                ),
                Some("<image{n}>"),
                "People, objects or styles to compose into the image, in order: \"<image1>\" is \
                 the first (e.g. \"the woman in <image1> holds the cup from <image2>\").",
            )],
            true,
        ),
        _ => (Vec::new(), false),
    };
    let multiple = match m.kind() {
        ImageKind::QwenImage21 => thinfer_models::qwen_image21::config::SIDE_MULTIPLE,
        _ => crate::request::VAE_SCALE,
    };
    let spec = JobSpec::Image(ImageSpec {
        model: Some(m),
        prompt: "x".into(),
        ..Default::default()
    });
    Entry {
        info: ModelInfo {
            id: m.to_string(),
            kind: ModelKind::Image,
            title: m.title().into(),
            default: m == ImageModelId::DEFAULT,
            downloaded: false,
            pausable: true,
            inputs,
            outputs: vec![Media::Image],
            size: Some(SizeRules {
                multiple,
                default_width: d.width,
                default_height: d.height,
                follows_last_input: follows,
            }),
            video: None,
            audio: None,
        },
        spec: (m.kind() != ImageKind::QwenImageEdit).then_some(spec),
        files: if m.kind() == ImageKind::QwenImageEdit {
            let man = m.manifest();
            m.required_roles()
                .iter()
                .filter_map(|r| man.get(r).copied())
                .collect()
        } else {
            Vec::new()
        },
    }
}

fn video(m: VideoModelId) -> Entry {
    let (w, h) = m.video_defaults();
    let mut inputs = Vec::new();
    if m.is_hunyuan_i2v() || m.is_ltx_i2v() || m.minimax_fl2va() {
        inputs.push(slot(
            "inputImage",
            Media::Image,
            InputUse::FirstFrame,
            (0, 1),
            None,
            "The video's first frame, stretched to the output size.",
        ));
    }
    if m.takes_refs() {
        inputs.push(slot(
            "refImages",
            Media::Image,
            InputUse::Reference,
            (0, crate::minimax_ref::MAX_REFERENCES as u32),
            Some("<Picture {n}>"),
            "People, objects, places or styles the video shows, in order: \"<Picture 1>\" is \
             the first. Define what each one provides, then use the subject label in the \
             scene (e.g. \"<Subject 1> is the woman in <Picture 1>. <Subject 1> waves.\").",
        ));
        inputs.push(slot(
            "refAudios",
            Media::Audio,
            InputUse::Reference,
            (0, crate::minimax_ref::MAX_REFERENCE_AUDIOS as u32),
            Some("<Audio {n}>"),
            "A voice to speak in, or music or a sound to reuse, cut to the clip's length; \
             needs at least one refImages entry. Numbered apart from the pictures: \
             \"<Audio 1>\" is the first clip. Say what each one is for, e.g. \"<Audio 1> is \
             the voice-timbre reference for <Subject 1> (S1).\", and write the speech as \
             \"<Subject 1> (S1) says, <d>[English] Hello there.</d>\".",
        ));
    }
    let audio = m.is_ltx() || m.is_minimax();
    let outputs = if audio {
        vec![Media::Video, Media::Audio]
    } else {
        vec![Media::Video]
    };
    // Each family's clip-length grid (what `snap_frames` / the H3 aligner
    // produce).
    let (frame_step, frame_offset) = if m.is_minimax() {
        (
            thinfer_models::minimax::packing::FRAMES_PER_CHUNK as u32,
            thinfer_models::minimax::packing::LATENTS_PER_CHUNK as u32,
        )
    } else if m.is_ltx() {
        (8, 1)
    } else if m.is_hunyuan_i2v() {
        // Chunks of 4 latent frames: 13, 29, 45, ...
        (16, 13)
    } else {
        (4, 1)
    };
    Entry {
        info: ModelInfo {
            id: m.to_string(),
            kind: ModelKind::Video,
            title: m.title().into(),
            default: m == VideoModelId::DEFAULT,
            downloaded: false,
            pausable: true,
            inputs,
            outputs,
            size: Some(SizeRules {
                multiple: m.dim_multiple(),
                default_width: w,
                default_height: h,
                follows_last_input: false,
            }),
            video: Some(VideoInfo {
                fps: if m.is_minimax() {
                    crate::minimax::FPS
                } else {
                    m.fps()
                },
                fixed_fps: m.is_minimax(),
                max_frames: (!m.is_minimax()).then(|| m.max_frames(w, h)).flatten(),
                frame_step,
                frame_offset,
                frames_round_up: m.is_minimax(),
                chained_clips: m.minimax_fl2va(),
            }),
            audio: None,
        },
        spec: Some(JobSpec::Video(VideoSpec {
            model: Some(m),
            prompts: vec!["x".into()],
            ..Default::default()
        })),
        files: Vec::new(),
    }
}

fn swap(m: SwapModel) -> Entry {
    let files = match m.file() {
        Some(f) => vec![crate::model::SCRFD, crate::model::ARCFACE, f],
        None => SwapModel::dreamidv_files(),
    };
    Entry {
        info: ModelInfo {
            id: m.to_string(),
            kind: ModelKind::FaceSwap,
            title: m.title().into(),
            default: m == SwapModel::DEFAULT,
            downloaded: false,
            pausable: m.is_dreamidv(),
            inputs: vec![
                slot(
                    "sourceImageB64",
                    Media::Image,
                    InputUse::Face,
                    (1, 1),
                    None,
                    "The face to swap in: one clearly visible face.",
                ),
                slot(
                    "inputVideoUpload",
                    Media::Video,
                    InputUse::Target,
                    (1, 1),
                    None,
                    "The video whose faces are replaced: an id from POST /uploads \
                     (inputVideoB64 also works for small clips). Its audio is copied to the \
                     result.",
                ),
            ],
            outputs: vec![Media::Video, Media::Audio],
            size: None,
            video: None,
            audio: None,
        },
        spec: None,
        files,
    }
}

fn text(m: TextModelId) -> Entry {
    Entry {
        info: ModelInfo {
            id: m.to_string(),
            kind: ModelKind::Text,
            title: m.title().into(),
            default: m == TextModelId::DEFAULT,
            downloaded: false,
            pausable: false,
            inputs: Vec::new(),
            outputs: vec![Media::Text],
            size: None,
            video: None,
            audio: None,
        },
        spec: Some(JobSpec::Text(TextSpec {
            model: Some(m),
            messages: vec![TextMessage::user("x")],
            ..Default::default()
        })),
        files: Vec::new(),
    }
}

fn audio(m: AudioModelId) -> Entry {
    let inputs = if m.takes_voice() {
        vec![slot(
            "refAudios",
            Media::Audio,
            InputUse::Voice,
            (1, 1),
            None,
            "The voice to speak in: a clean clip of one speaker, 3 to 15 s. Send its transcript as refText and the clip's delivery (pace, tone, emotion) carries over too: keep a clip per mood (calm, whispering, shouting) of the same voice and pick the one each line needs.",
        )]
    } else {
        Vec::new()
    };
    let notes = match m {
        AudioModelId::Qwen3TtsDesign => {
            "Describe the voice and how it speaks in instruct (\"a young woman with a warm, slightly husky voice; exhausted, out of breath\"). The same description gives a similar but not identical voice each time: for one consistent voice, make a clip once and speak with qwen3-tts from it."
        }
        AudioModelId::Qwen3Tts => {
            "Speaks in the reference clip's voice. Emotion follows the clip (with refText), not the words: an angry line from a calm clip stays calm."
        }
        AudioModelId::MossSfx => {
            "Describe the sound in text (\"heavy rain on a tin roof, distant thunder\"), English or Chinese; seconds sets the length (up to 30). loop makes a seamless loop for ambience. Not for speech or music with lyrics."
        }
    };
    let languages = match m.kind() {
        AudioKind::Speech => thinfer_models::qwen3_tts::LANGUAGES
            .iter()
            .map(|l| (*l).into())
            .collect(),
        AudioKind::Sound => Vec::new(),
    };
    let spec = (!m.takes_voice()).then(|| {
        JobSpec::Audio(AudioSpec {
            model: Some(m),
            text: "x".into(),
            ..Default::default()
        })
    });
    let files = if m.takes_voice() {
        crate::request::AudioRequest::files_of(m)
    } else {
        Vec::new()
    };
    Entry {
        info: ModelInfo {
            id: m.to_string(),
            kind: ModelKind::Audio,
            title: m.title().into(),
            default: m == AudioModelId::DEFAULT,
            downloaded: false,
            pausable: false,
            inputs,
            outputs: vec![Media::Audio],
            size: None,
            video: None,
            audio: Some(AudioInfo {
                kind: match m.kind() {
                    AudioKind::Speech => AudioKindInfo::Speech,
                    AudioKind::Sound => AudioKindInfo::Sound,
                },
                sample_rate: m.sample_rate(),
                max_seconds: m.max_seconds(),
                instruct: m.takes_instruct(),
                languages,
                notes: notes.into(),
            }),
        },
        spec,
        files,
    }
}

/// Every model this build runs: text, image, audio, video, face swap.
pub fn entries() -> Vec<Entry> {
    TextModelId::ALL
        .into_iter()
        .map(text)
        .chain(ImageModelId::ALL.into_iter().map(image))
        .chain(AudioModelId::ALL.into_iter().map(audio))
        .chain(VideoModelId::ALL.into_iter().map(video))
        .chain(SwapModel::ALL.into_iter().map(swap))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_model_is_listed_once_with_its_input_rules() {
        let e = entries();
        let ids: Vec<&str> = e.iter().map(|e| e.info.id.as_str()).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "duplicate ids");
        let get = |id: &str| &e.iter().find(|e| e.info.id == id).expect(id).info;
        let q21 = get("qwen-image-2.1-turbo");
        assert_eq!(q21.inputs[0].max, 10);
        assert_eq!(q21.inputs[0].prompt_tag.as_deref(), Some("<image{n}>"));
        assert!(q21.size.as_ref().unwrap().follows_last_input);
        let omni = get("fasth3-omni");
        assert!(
            omni.inputs
                .iter()
                .any(|s| s.used_as == InputUse::Reference && s.max == 9)
        );
        assert_eq!(omni.outputs, [Media::Video, Media::Audio]);
        let audio = omni
            .inputs
            .iter()
            .find(|s| s.media == Media::Audio)
            .expect("omni takes reference audio");
        assert_eq!(
            (audio.field.as_str(), audio.max, audio.prompt_tag.as_deref()),
            ("refAudios", 3, Some("<Audio {n}>"))
        );
        assert!(audio.formats.iter().any(|f| f == "mp3"));
        assert_eq!(get("fastwan-ti2v-5b").outputs, [Media::Video]);
        assert_eq!(get("zimage-turbo-q8").outputs, [Media::Image]);
        assert!(e.iter().all(|e| !e.info.outputs.is_empty()));
        assert!(
            e.iter()
                .flat_map(|e| &e.info.inputs)
                .all(|s| !s.notes.is_empty() && !s.formats.is_empty())
        );
        let trim = get("fasth3-trim");
        assert!(
            trim.inputs
                .iter()
                .any(|s| s.used_as == InputUse::FirstFrame)
        );
        let v = trim.video.as_ref().unwrap();
        assert!(v.chained_clips);
        // 120 frames round up to 124 on the 17n + 5 grid.
        assert_eq!(
            (v.frame_step, v.frame_offset, v.frames_round_up),
            (17, 5, true)
        );
        for id in ["fastwan-ti2v-5b", "ltx2-rapid", "hunyuan-video-1.5-ti2v"] {
            let v = get(id).video.as_ref().unwrap();
            let m: VideoModelId = serde_json::from_value(serde_json::json!(id)).unwrap();
            for raw in 1..200u32 {
                let f = m.snap_frames(raw);
                assert!(
                    f >= v.frame_offset && (f - v.frame_offset).is_multiple_of(v.frame_step),
                    "{id}: {f} is off its listed grid"
                );
            }
        }
        let edit = &get("qwen-image-edit-rapid").inputs[0];
        assert_eq!((edit.min, edit.max), (1, 3));
        assert!(get("zimage-turbo-q8").inputs.is_empty());
        let clone = get("qwen3-tts");
        assert_eq!(clone.kind, ModelKind::Audio);
        assert_eq!(clone.inputs[0].used_as, InputUse::Voice);
        assert!(!clone.audio.as_ref().unwrap().instruct);
        assert!(get("qwen3-tts-design").audio.as_ref().unwrap().instruct);
        assert_eq!(
            e.iter().filter(|e| e.info.default).count(),
            5,
            "one default per kind"
        );
    }
}
