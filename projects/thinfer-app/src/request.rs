//! Job request types and their validation/resolution. One `JobRequest` per
//! modality; every front end (CLI flags, serve JSON) builds these and hands
//! them to a [`crate::executor::LocalExecutor`]. Resolution that can fail
//! (output format, frame grid, shot plan) is exposed standalone so a server can
//! reject a bad request at submit time rather than mid-job.

use std::path::PathBuf;

use thinfer_core::manifest::FileRef;
use thinfer_core::policy::ResidencyBudget;
pub use thinfer_models::wan::pipeline::Shot;

use crate::model::{ImageModelId, SwapModel, VaeChoice, VideoModelId, VideoSampler};

/// Both image and video models narrow dims by `vae_scale = vae_factor*2 = 16`.
const VAE_SCALE: u32 = 16;

/// Validate a pixel dimension: positive and a multiple of the VAE scale.
pub fn validate_dim(name: &str, v: u32) -> Result<(), String> {
    if v == 0 {
        return Err(format!("--{name} must be > 0"));
    }
    if !v.is_multiple_of(VAE_SCALE) {
        return Err(format!(
            "--{name} must be a multiple of {VAE_SCALE} (got {v})"
        ));
    }
    Ok(())
}

/// Resolve an output format: an explicit choice wins, otherwise infer from the
/// path extension. A missing/unrecognized extension is a hard error (we never
/// silently write the wrong container). `from_ext` gets the lower-cased
/// extension; `known` lists recognized extensions for the failure message.
pub fn resolve_output_format<T: Copy>(
    explicit: Option<T>,
    output: &std::path::Path,
    from_ext: impl Fn(&str) -> Option<T>,
    known: &str,
) -> Result<T, String> {
    if let Some(f) = explicit {
        return Ok(f);
    }
    let ext = output.extension().and_then(|e| e.to_str()).ok_or_else(|| {
        format!(
            "cannot infer output format: {} has no file extension. Pass an explicit format or use a known extension ({known}).",
            output.display(),
        )
    })?;
    from_ext(&ext.to_ascii_lowercase()).ok_or_else(|| {
        format!("cannot infer output format from extension {ext:?}; known: {known}. Pass an explicit format.")
    })
}

/// Image output container. PNG-only today (the only encoder shipped; every
/// parity baseline is PNG).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum ImageFormat {
    Png,
}

impl ImageFormat {
    pub fn from_ext(ext: &str) -> Option<Self> {
        match ext {
            "png" => Some(Self::Png),
            _ => None,
        }
    }
    pub const KNOWN: &'static str = "png";
}

/// Video output container.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum VideoFormat {
    Mp4,
    /// Raw per-frame PNG sequence: the output path is a directory, frames land
    /// as `frame{n:03}.png`. Not inferable from an extension, so it must be set
    /// explicitly. Bypasses the H.264 encode entirely (the codec-free view of
    /// the VAE decode).
    PngFrames,
}

impl VideoFormat {
    pub fn from_ext(ext: &str) -> Option<Self> {
        match ext {
            "mp4" => Some(Self::Mp4),
            _ => None,
        }
    }
    pub const KNOWN: &'static str = "mp4";
}

/// Generate an image from a prompt (Z-Image t2i).
#[derive(Clone, Debug)]
pub struct ImageRequest {
    pub model: ImageModelId,
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    /// `None` -> randomized at run time; the resolved value is reported back.
    pub seed: Option<u64>,
    /// DP4A i8 matmul on the DP4A-safe DiT sites (qkv + ffn_up). Default true;
    /// `false` forces the bf16/dequant-once reference path. Ideogram-4 only
    /// (Z-Image ignores it).
    pub i8_matmul: bool,
    /// Reference image for the image-EDIT path (Qwen-Image-Edit). REQUIRED for
    /// the `QwenImageEdit` kind; rejected for the t2i kinds (Z-Image, Ideogram).
    pub input_image: Option<PathBuf>,
    pub budget: ResidencyBudget,
    pub output: PathBuf,
    pub format: ImageFormat,
}

impl ImageRequest {
    /// Files this request needs in the HF cache.
    pub fn required_files(&self) -> Vec<FileRef> {
        match self.model.kind() {
            crate::model::ImageKind::ZImage => {
                self.model.variant().files().map(|(_, f)| *f).collect()
            }
            // Ideogram-4 + Qwen-Image(-Edit) source by role (encoder + DiT GGUFs,
            // VAE, mmproj/LoRA, tokenizer), not via the Z-Image variant registry.
            crate::model::ImageKind::Ideogram4
            | crate::model::ImageKind::QwenImageEdit
            | crate::model::ImageKind::QwenImage => {
                let m = self.model.manifest();
                self.model
                    .required_roles()
                    .iter()
                    .map(|r| *m.get(r).expect("image role in manifest"))
                    .collect()
            }
        }
    }

    /// Validate everything that can fail before any GPU/network work.
    pub fn validate(&self) -> Result<(), String> {
        validate_dim("height", self.height)?;
        validate_dim("width", self.width)?;
        if self.steps == 0 {
            return Err("--steps must be > 0".into());
        }
        match self.model.kind() {
            crate::model::ImageKind::QwenImageEdit => {
                if self.input_image.is_none() {
                    return Err(format!("--input-image is required for {}", self.model));
                }
            }
            _ => {
                if self.input_image.is_some() {
                    return Err(format!("--input-image is not supported by {}", self.model));
                }
            }
        }
        Ok(())
    }
}

/// Generate a video from one or more prompts (t2v; multi-prompt = multi-shot,
/// LongLive only).
#[derive(Clone, Debug)]
pub struct VideoRequest {
    pub model: VideoModelId,
    pub prompts: Vec<String>,
    pub width: u32,
    pub height: u32,
    /// Verbatim frame counts. Mutually exclusive with `durations` (the caller
    /// enforces, as clap does); at most one of the two is non-empty.
    pub frames: Vec<u32>,
    pub durations: Vec<f32>,
    /// `None` -> the model's preferred fps.
    pub fps: Option<u32>,
    pub seed: Option<u64>,
    /// img2vid conditioning (not yet wired; rejected in [`Self::resolve`]).
    pub input_image: Option<PathBuf>,
    /// FastWan denoise sampler (ignored on the AR path).
    pub sampler: VideoSampler,
    /// UniPC denoise step count (1..=`VIDEO_MAX_STEPS`); DMD ignores it.
    pub steps: u32,
    pub vae: VaeChoice,
    pub i8_matmul: bool,
    pub budget: ResidencyBudget,
    pub output: PathBuf,
    pub format: VideoFormat,
}

/// The resolved video plan: total frame count, per-shot list (empty = the
/// single-shot path), and the effective fps.
#[derive(Clone, Debug)]
pub struct VideoPlan {
    pub frames: u32,
    pub shots: Vec<Shot>,
    pub fps: u32,
}

impl VideoRequest {
    pub fn required_files(&self) -> Result<Vec<FileRef>, String> {
        let mut files: Vec<FileRef> = self.model.variant().files().map(|(_, f)| *f).collect();
        if self.vae == VaeChoice::Tiny {
            files.push(
                *self
                    .model
                    .manifest()
                    .get(thinfer_models::wan::manifest::role::TINY_VAE)
                    .ok_or("manifest missing tiny VAE role")?,
            );
        }
        Ok(files)
    }

    /// Validate dims + resolve fps and the shot plan. Fails fast on every user
    /// error so a server can 400 at submit.
    pub fn resolve(&self) -> Result<VideoPlan, String> {
        validate_dim("height", self.height)?;
        validate_dim("width", self.width)?;
        if self.input_image.is_some() {
            return Err(
                "--input-image (img2vid) not yet wired; the engine path is t2v-only".into(),
            );
        }
        if self.prompts.is_empty() {
            return Err("at least one prompt is required".into());
        }
        let fps = self.fps.unwrap_or_else(|| self.model.fps());
        if fps == 0 {
            return Err("--fps must be > 0".into());
        }
        if self.steps == 0 {
            return Err("--steps must be > 0".into());
        }
        let (frames, shots) = resolve_shot_plan(
            self.model,
            &self.prompts,
            &self.frames,
            &self.durations,
            fps,
        )?;
        Ok(VideoPlan { frames, shots, fps })
    }
}

/// Swap a face from a source image into every frame of an input video.
#[derive(Clone, Debug)]
pub struct FaceSwapRequest {
    pub model: SwapModel,
    pub input_video: PathBuf,
    pub source_image: PathBuf,
    pub output: PathBuf,
    pub budget: ResidencyBudget,
}

impl FaceSwapRequest {
    pub fn required_files(&self) -> Vec<FileRef> {
        vec![
            crate::model::SCRFD,
            crate::model::ARCFACE,
            self.model.file(),
        ]
    }

    pub fn validate(&self) -> Result<(), String> {
        let ok = self
            .output
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("mp4"))
            .unwrap_or(false);
        if !ok {
            return Err("--output must be a .mp4 file".into());
        }
        Ok(())
    }
}

/// A unit of work. Large-input variants (face-swap) are flagged so a server can
/// reject them when busy instead of queuing.
#[derive(Clone, Debug)]
pub enum JobRequest {
    Image(ImageRequest),
    Video(VideoRequest),
    FaceSwap(FaceSwapRequest),
}

impl JobRequest {
    /// True if this request reads a large local input (face-swap video), which
    /// a server must run now-or-reject rather than queue.
    pub fn is_large_input(&self) -> bool {
        matches!(self, JobRequest::FaceSwap(_))
    }

    /// Files this request needs in the HF cache.
    pub fn required_files(&self) -> Result<Vec<FileRef>, String> {
        match self {
            JobRequest::Image(r) => Ok(r.required_files()),
            JobRequest::Video(r) => r.required_files(),
            JobRequest::FaceSwap(r) => Ok(r.required_files()),
        }
    }
}

/// What a finished job produced.
#[derive(Clone, Debug)]
pub struct JobSummary {
    pub output: PathBuf,
    pub width: u32,
    pub height: u32,
    /// Frame count (1 for an image).
    pub frames: u32,
    /// Playback fps (video / face-swap only).
    pub fps: Option<u32>,
    /// The seed actually used (image/video; `None` for face-swap).
    pub seed: Option<u64>,
}

/// Resolve prompts + frames/durations into the total frame count and per-shot
/// plan. A single frames/duration value splits the clip evenly across shots (in
/// chunk units, mirroring upstream `_even_durations`); one value per prompt
/// sizes each shot independently; any other count is an error. A single prompt
/// yields an empty shot list (the parity single-shot path stays untouched).
/// Multi-shot is LongLive-only.
pub fn resolve_shot_plan(
    model: VideoModelId,
    prompts: &[String],
    frames: &[u32],
    durations: &[f32],
    fps: u32,
) -> Result<(u32, Vec<Shot>), String> {
    let num_shots = prompts.len();
    let values: Vec<u32> = if !durations.is_empty() {
        durations
            .iter()
            .map(|&d| {
                if !(d.is_finite() && d > 0.0) {
                    return Err(format!(
                        "--duration must be a positive number of seconds (got {d})"
                    ));
                }
                Ok(model.snap_frames((d * fps as f32).round() as u32))
            })
            .collect::<Result<_, _>>()?
    } else {
        frames.to_vec()
    };

    if !model.is_ar() {
        if num_shots != 1 {
            return Err(format!(
                "multiple --prompt (multi-shot) is only supported by longlive-2.0-5b, not {model}"
            ));
        }
        if values.len() > 1 {
            return Err(format!(
                "multiple --frames/--duration is only for multi-shot longlive-2.0-5b, not {model}"
            ));
        }
        let frames = match values.first() {
            Some(&f) if !frames.is_empty() => {
                model.validate_frames(f)?;
                f
            }
            Some(&f) => f, // from --duration, already snapped legal
            None => model.default_frames(),
        };
        return Ok((frames, Vec::new()));
    }

    let shot_chunks: Vec<usize> = match values.len() {
        0 => even_chunk_split(model.frames_to_chunks(model.default_frames())?, num_shots)?,
        1 => even_chunk_split(model.frames_to_chunks(values[0])?, num_shots)?,
        n if n == num_shots => values
            .iter()
            .map(|&f| model.frames_to_chunks(f))
            .collect::<Result<_, _>>()?,
        n => {
            return Err(format!(
                "expected 1 --{} value (split evenly) or {num_shots} (one per --prompt), got {n}",
                if durations.is_empty() {
                    "frames"
                } else {
                    "duration"
                }
            ));
        }
    };

    let total_chunks: usize = shot_chunks.iter().sum();
    let frames = model.chunks_to_frames(total_chunks);
    if num_shots == 1 {
        return Ok((frames, Vec::new()));
    }
    let shots = prompts
        .iter()
        .zip(shot_chunks)
        .map(|(p, chunks)| Shot {
            prompt: p.clone(),
            chunks,
        })
        .collect();
    Ok((frames, shots))
}

/// Split `total` chunks across `num_shots` shots as evenly as possible (the
/// first `total % num_shots` shots get one extra), mirroring upstream
/// `_even_durations`. Errors if there are fewer chunks than shots.
pub fn even_chunk_split(total: usize, num_shots: usize) -> Result<Vec<usize>, String> {
    if total < num_shots {
        return Err(format!(
            "clip is {total} chunk(s) but there are {num_shots} shots; give a longer \
             --frames/--duration or fewer --prompt"
        ));
    }
    let base = total / num_shots;
    let extra = total % num_shots;
    Ok((0..num_shots)
        .map(|i| base + usize::from(i < extra))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{even_chunk_split, resolve_shot_plan};
    use crate::model::VideoModelId::{FastwanTi2v5b as Fw, Longlive205b as Ll};

    const FPS: u32 = 24;

    fn prompts(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("shot {i}")).collect()
    }

    #[test]
    fn even_split_distributes_remainder_to_front() {
        assert_eq!(even_chunk_split(6, 3).unwrap(), vec![2, 2, 2]);
        assert_eq!(even_chunk_split(7, 3).unwrap(), vec![3, 2, 2]);
        assert_eq!(even_chunk_split(8, 3).unwrap(), vec![3, 3, 2]);
        assert_eq!(even_chunk_split(3, 3).unwrap(), vec![1, 1, 1]);
        assert!(even_chunk_split(2, 3).is_err());
    }

    #[test]
    fn single_prompt_yields_no_shots() {
        let (frames, shots) = resolve_shot_plan(Ll, &prompts(1), &[61], &[], FPS).unwrap();
        assert_eq!(frames, 61);
        assert!(shots.is_empty());
        let (frames, shots) = resolve_shot_plan(Fw, &prompts(1), &[], &[], FPS).unwrap();
        assert_eq!(frames, Fw.default_frames());
        assert!(shots.is_empty());
    }

    #[test]
    fn one_frames_value_splits_evenly_across_shots() {
        let (frames, shots) = resolve_shot_plan(Ll, &prompts(2), &[125], &[], FPS).unwrap();
        assert_eq!(frames, 125);
        assert_eq!(shots.len(), 2);
        assert_eq!((shots[0].chunks, shots[1].chunks), (2, 2));
        assert_eq!(shots[0].prompt, "shot 0");
        assert_eq!(shots[1].prompt, "shot 1");
    }

    #[test]
    fn frames_per_shot_sizes_each_independently() {
        let (frames, shots) = resolve_shot_plan(Ll, &prompts(2), &[29, 61], &[], FPS).unwrap();
        assert_eq!(shots.len(), 2);
        assert_eq!((shots[0].chunks, shots[1].chunks), (1, 2));
        assert_eq!(frames, Ll.chunks_to_frames(3));
        assert_eq!(frames, 93);
    }

    #[test]
    fn wrong_value_count_is_an_error() {
        assert!(resolve_shot_plan(Ll, &prompts(3), &[29, 61], &[], FPS).is_err());
        assert!(resolve_shot_plan(Fw, &prompts(2), &[], &[], FPS).is_err());
        assert!(resolve_shot_plan(Fw, &prompts(1), &[61, 61], &[], FPS).is_err());
    }

    #[test]
    fn duration_per_shot_converts_and_snaps() {
        let (frames, shots) = resolve_shot_plan(Ll, &prompts(2), &[], &[1.2, 2.5], FPS).unwrap();
        assert_eq!(shots.len(), 2);
        for s in &shots {
            assert!(s.chunks >= 1);
        }
        assert_eq!(
            frames,
            Ll.chunks_to_frames(shots.iter().map(|s| s.chunks).sum())
        );
    }

    #[test]
    fn fastwan_snaps_to_4k_plus_1() {
        assert_eq!(Fw.snap_frames(97), 97);
        assert_eq!(Fw.snap_frames(60), 61);
        assert_eq!(Fw.snap_frames(50), 49);
        assert_eq!(Fw.snap_frames(1), 1);
        assert_eq!(Fw.snap_frames(0), 1);
        // Default is 5s @ 24fps = 120 -> 121 (nearest 4k+1).
        assert_eq!(Fw.default_frames(), 121);
    }

    #[test]
    fn longlive_snaps_to_latent_multiple_of_8() {
        assert_eq!(Ll.snap_frames(61), 61);
        assert_eq!(Ll.snap_frames(50), 61);
        assert_eq!(Ll.snap_frames(40), 29);
        assert_eq!(Ll.snap_frames(97), 93);
        assert_eq!(Ll.snap_frames(1), 29);
        // Default is 5s @ 24fps = 120 -> 125 (nearest on the chunk-of-8 grid).
        assert_eq!(Ll.default_frames(), 125);
    }

    #[test]
    fn snapped_frames_always_validate() {
        for raw in 1u32..400 {
            for m in [Fw, Ll] {
                m.validate_frames(m.snap_frames(raw))
                    .unwrap_or_else(|e| panic!("{m} snap({raw}) not legal: {e}"));
            }
        }
    }

    #[test]
    fn validate_rejects_off_grid() {
        assert!(Fw.validate_frames(96).is_err());
        assert!(Fw.validate_frames(97).is_ok());
        assert!(Ll.validate_frames(97).is_err());
        assert!(Ll.validate_frames(61).is_ok());
        assert!(Ll.validate_frames(29).is_ok());
    }
}
