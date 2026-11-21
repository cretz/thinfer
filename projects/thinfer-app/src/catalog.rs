//! The model catalog `GET /models` serves: every model this build runs, what
//! it takes (input images, how a prompt names them), what decides its output
//! size, and its video limits. Built from the same per-model rules the
//! request validation enforces, so the two cannot drift; the server adds
//! whether each one's weights are already local (from [`Entry::spec`]).

use thinfer_client::wire::{
    ImageSpec, InputSlot, InputUse, JobSpec, ModelInfo, ModelKind, SizeRules, TextMessage,
    TextSpec, VideoInfo, VideoSpec,
};

use crate::model::{
    ImageKind, ImageModelEngine, ImageModelId, SwapModel, SwapModelEngine, TextModelId,
    VideoModelId,
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

fn slot(field: &str, used_as: InputUse, min: u32, max: u32, tag: Option<&str>) -> InputSlot {
    InputSlot {
        field: field.into(),
        used_as,
        min,
        max,
        prompt_tag: tag.map(Into::into),
    }
}

fn image(m: ImageModelId) -> Entry {
    let d = m.defaults();
    let (inputs, follows) = match m.kind() {
        ImageKind::QwenImageEdit => (
            vec![slot(
                "inputImages",
                InputUse::Edit,
                1,
                crate::preprocess::MAX_EDIT_IMAGES as u32,
                Some("Picture {n}"),
            )],
            false,
        ),
        ImageKind::QwenImage21 => (
            vec![slot(
                "inputImages",
                InputUse::Condition,
                0,
                thinfer_models::qwen_image21::config::MAX_CONDITION_IMAGES as u32,
                Some("<image{n}>"),
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
            size: Some(SizeRules {
                multiple,
                default_width: d.width,
                default_height: d.height,
                follows_last_input: follows,
            }),
            video: None,
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
        inputs.push(slot("inputImage", InputUse::FirstFrame, 0, 1, None));
    }
    if m.takes_refs() {
        inputs.push(slot(
            "refImages",
            InputUse::Reference,
            0,
            crate::minimax_ref::MAX_REFERENCES as u32,
            Some("<Picture {n}>"),
        ));
    }
    let audio = m.is_ltx() || m.is_minimax();
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
                audio,
            }),
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
            inputs: vec![slot("sourceImage", InputUse::Face, 1, 1, None)],
            size: None,
            video: None,
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
            size: None,
            video: None,
        },
        spec: Some(JobSpec::Text(TextSpec {
            model: Some(m),
            messages: vec![TextMessage::user("x")],
            ..Default::default()
        })),
        files: Vec::new(),
    }
}

/// Every model this build runs: text, image, video, face swap.
pub fn entries() -> Vec<Entry> {
    TextModelId::ALL
        .into_iter()
        .map(text)
        .chain(ImageModelId::ALL.into_iter().map(image))
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
        assert!(omni.video.as_ref().unwrap().audio);
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
        assert_eq!(
            e.iter().filter(|e| e.info.default).count(),
            4,
            "one default per kind"
        );
    }
}
