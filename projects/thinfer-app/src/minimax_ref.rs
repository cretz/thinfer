//! FastH3-Omni reference images (Ref2VA): every image the prompt can name as
//! `<Picture i>` reaches the model twice, as the reference does it
//! (`modular/packing_ref2va.py`, `encoders.py`):
//!
//! - through the Qwen3-VL vision tower into the conditioner: the prompt gains
//!   `"<Picture i>: " <|vision_start|> N x <|image_pad|> <|vision_end|>` per
//!   image, the pads take the tower's features (deepstack included), and the
//!   whole vision block is tagged video for the DiT's modulation;
//! - through the video VAE encoder (tiled, posterior sampled at seed 42,
//!   f16-rounded) as clean conditioning rows the DiT attends to.
//!
//! Both see the same prepared image: EXIF-oriented RGB, Lanczos-resized per
//! [`RefImageSize`] (the reference: short edge 2048) with both sides multiples
//! of 32 (the vision processor's own resize is then the identity).
//!
//! An FL2VA keyframe (fasth3-trim / minimax-h3 first frame, or a chained
//! clip's handoff) takes the same two paths, prepared at the target canvas
//! instead (`prepare_keyframe_image`, stretched as the reference does its
//! first keyframe) and pinned at the clip start rather than given a time slot.

use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::residency::WeightResidency;
use thinfer_core::tokenizer::Tokenizer;
use thinfer_core::weight::{WeightId, WeightReader, WeightSource, decode_into};
use thinfer_core::workspace::Workspace;
use thinfer_models::common::block::BlockPipelines;
use thinfer_models::minimax::audio_encoder::AudioEncoder;
use thinfer_models::minimax::audio_vae::{AudioVae, AudioVaeConfig};
use thinfer_models::minimax::manifest::role;
use thinfer_models::minimax::packing::{self, Condition, KeyframeAnchor};
use thinfer_models::minimax::vae_encoder::{
    EncodeOptions, EncoderPipelines, EncoderSource, H3VaeEncoder,
};
use thinfer_models::minimax::vision::{self, VisionOutput};
use thinfer_models::minimax::{DitConfig, TAG_TEXT, TAG_VIDEO};

use crate::config::ResidencyBudget;
use crate::minimax::{open_mmap, open_st};
use crate::model::RefImageSize;

/// The side multiple every reference is resized to.
const SIDE_MULTIPLE: usize = 32;
/// The reference rejects anything flatter than 4:1 either way.
const MAX_ASPECT: usize = 4;
/// At most this many images per request (the reference's limit).
pub const MAX_REFERENCES: usize = 9;
/// At most this many audio clips per request (the reference's limit).
pub const MAX_REFERENCE_AUDIOS: usize = 3;

/// Qwen2 tokenizer special tokens (verified against the tokenizer at run time).
const VISION_START: (&str, u32) = ("<|vision_start|>", 151_652);
const VISION_END: (&str, u32) = ("<|vision_end|>", 151_653);
const IMAGE_PAD: (&str, u32) = ("<|image_pad|>", 151_655);

/// One reference image, prepared: RGB8 `[height, width, 3]`. User media, so
/// page-locked on a server and zeroed on drop.
pub struct Prepared {
    pub rgb: thinfer_core::secmem::Locked<u8>,
    pub height: usize,
    pub width: usize,
}

/// Python's `round` (half to even), as the reference sizes with it.
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// Decode one user image, EXIF-oriented.
fn decode_oriented(bytes: &[u8], what: &str) -> Result<image::DynamicImage, String> {
    use image::ImageDecoder;
    let mut decoder = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("{what}: {e}"))?
        .into_decoder()
        .map_err(|e| format!("{what}: {e}"))?;
    let orientation = decoder
        .orientation()
        .map_err(|e| format!("{what} orientation: {e}"))?;
    let mut img = image::DynamicImage::from_decoder(decoder).map_err(|e| format!("{what}: {e}"))?;
    img.apply_orientation(orientation);
    Ok(img)
}

/// Decode, orient and stretch an FL2VA first frame to the `width x height`
/// canvas (Lanczos), as the reference prepares its first keyframe.
pub fn prepare_keyframe(bytes: &[u8], height: usize, width: usize) -> Result<Prepared, String> {
    let mut img = decode_oriented(bytes, "input image")?;
    if (img.width() as usize, img.height() as usize) != (width, height) {
        img = img.resize_exact(
            width as u32,
            height as u32,
            image::imageops::FilterType::Lanczos3,
        );
    }
    Ok(Prepared {
        rgb: img.to_rgb8().into_raw().into(),
        height,
        width,
    })
}

/// Decode and orient one reference image.
pub fn decode(bytes: &[u8]) -> Result<image::DynamicImage, String> {
    decode_oriented(bytes, "reference image")
}

/// [`RefCrop::Auto`]: crop in when the largest face would come out shorter
/// than this many pixels once the image is sized...
const CROP_BELOW_FACE: f64 = 96.0;
/// ...aiming for this prepared face height...
const CROP_TARGET_FACE: f64 = 144.0;
/// ...while always keeping this many face heights of the person (head to
/// chest), plus a face height of margin around everyone kept.
const CROP_MIN_FACES: f64 = 3.0;
/// Faces at least this fraction of the largest one's height stay in the crop
/// (a couple photographed together); smaller ones are background.
const CROP_KEEP_FACE: f32 = 0.6;
/// Not worth a crop that keeps more than this fraction of the image height.
const CROP_MAX_KEEP: f64 = 0.9;
/// Detection runs on a copy with at most this long edge, whole and in 640
/// tiles (one whole-image pass shrinks a 2400 px photo 4x, below where SCRFD
/// sees the small faces this exists for); bounds the tile count to 12.
const DETECT_EDGE: u32 = 1920;

/// A source-pixel rectangle: x, y, width, height.
pub type Rect = [u32; 4];

/// The SCRFD face detector, loaded once per request.
pub struct FaceFinder(thinfer_core::onnx::OnnxModel);

impl FaceFinder {
    pub async fn load(backend: &Arc<WgpuBackend>) -> Result<Self, String> {
        let path = crate::download::resolve_file(&crate::model::SCRFD)?;
        let onnx = std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        thinfer_models::faceswap::detect::load(Arc::clone(backend), &onnx)
            .await
            .map(Self)
            .map_err(|e| format!("load face detector: {e:?}"))
    }

    /// Where to crop `img` for a `width x height` canvas sized per `size`,
    /// or `None` to keep it whole.
    pub async fn crop(
        &self,
        img: &image::DynamicImage,
        size: RefImageSize,
        width: usize,
        height: usize,
    ) -> Result<Option<Rect>, String> {
        let (w, h) = (img.width(), img.height());
        let small = if w.max(h) > DETECT_EDGE {
            img.resize(
                DETECT_EDGE,
                DETECT_EDGE,
                image::imageops::FilterType::Triangle,
            )
        } else {
            img.clone()
        };
        let ratio = w as f32 / small.width() as f32;
        let rgb = thinfer_core::secmem::Locked::from(small.to_rgb8().into_raw());
        let det = thinfer_models::faceswap::image::Image::from_rgb8(
            small.width() as usize,
            small.height() as usize,
            &rgb,
        );
        let faces = thinfer_models::faceswap::detect::detect_tiled(&self.0, &det)
            .await
            .map_err(|e| format!("face detection: {e:?}"))?;
        let boxes: Vec<[f32; 4]> = faces.iter().map(|f| f.bbox.map(|v| v * ratio)).collect();
        let rect = face_crop(&boxes, w as usize, h as usize, size, width, height);
        tracing::debug!(
            target: "thinfer::diag",
            faces = boxes.len(),
            largest = boxes.iter().map(|b| b[3] - b[1]).fold(0.0, f32::max),
            source = ?(w, h),
            crop = ?rect,
            "omni reference framing"
        );
        Ok(rect)
    }
}

/// [`RefCrop::Auto`]'s rectangle for face `boxes` (`[x1, y1, x2, y2]`,
/// source pixels) in a `w x h` image sized per `size` for the canvas: `None`
/// when no face is found, the largest is already big enough, or the crop
/// would keep nearly everything. Keeps the source aspect ratio; the face sits
/// near the top so the crop runs down the body.
fn face_crop(
    boxes: &[[f32; 4]],
    w: usize,
    h: usize,
    size: RefImageSize,
    canvas_w: usize,
    canvas_h: usize,
) -> Option<Rect> {
    let height = |b: &[f32; 4]| b[3] - b[1];
    let largest = boxes.iter().map(height).fold(0.0, f32::max);
    if largest <= 0.0 {
        return None;
    }
    let face = largest as f64;
    if face * size.scale(w, h, canvas_w, canvas_h) >= CROP_BELOW_FACE {
        return None;
    }
    let kept = boxes
        .iter()
        .filter(|b| height(b) >= CROP_KEEP_FACE * largest);
    let u = kept.fold([f32::MAX, f32::MAX, f32::MIN, f32::MIN], |u, b| {
        [
            u[0].min(b[0]),
            u[1].min(b[1]),
            u[2].max(b[2]),
            u[3].max(b[3]),
        ]
    });
    let (wf, hf) = (w as f64, h as f64);
    let aspect = wf / hf;
    let min_h = (CROP_MIN_FACES * face)
        .max((u[3] - u[1]) as f64 + 2.0 * face)
        .max(((u[2] - u[0]) as f64 + 2.0 * face) / aspect);
    // Shrink until the face reaches its target size or the crop is no
    // longer downscaled (sizing never upscales, so a tighter crop would only
    // lose context); never below the person. The sizing modes differ, so
    // search rather than solve.
    let mut ch = hf;
    while ch > min_h {
        let s = size.scale(
            (aspect * ch).round() as usize,
            ch.round() as usize,
            canvas_w,
            canvas_h,
        );
        if face * s >= CROP_TARGET_FACE || s >= 1.0 {
            break;
        }
        ch *= 0.95;
    }
    let ch = ch.max(min_h);
    if ch >= CROP_MAX_KEEP * hf {
        return None;
    }
    let cw = (aspect * ch).min(wf);
    let cx = (u[0] + u[2]) as f64 / 2.0;
    let x = (cx - cw / 2.0).clamp(0.0, wf - cw);
    // Hair above the box, then the body.
    let y = (u[1] as f64 - 0.6 * face).clamp(0.0, hf - ch);
    Some([
        x.round() as u32,
        y.round() as u32,
        (cw.round() as u32).min(w as u32 - x.round() as u32),
        (ch.round() as u32).min(h as u32 - y.round() as u32),
    ])
}

/// Crop (optionally), then resize one decoded reference image for a
/// `width x height` canvas.
pub fn prepare(
    mut img: image::DynamicImage,
    crop: Option<Rect>,
    size: RefImageSize,
    width: usize,
    height: usize,
) -> Result<Prepared, String> {
    if let Some([x, y, cw, ch]) = crop {
        img = img.crop_imm(x, y, cw, ch);
    }
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w > MAX_ASPECT * h || h > MAX_ASPECT * w {
        return Err(format!(
            "reference image {w}x{h} is flatter than {MAX_ASPECT}:1"
        ));
    }
    let scale = size.scale(w, h, width, height);
    let m = SIDE_MULTIPLE as f64;
    let side = |x: usize| (round_half_even(x as f64 * scale / m) * m).max(m) as usize;
    let (tw, th) = (side(w), side(h));
    if (tw, th) != (w, h) {
        img = img.resize_exact(tw as u32, th as u32, image::imageops::FilterType::Lanczos3);
    }
    Ok(Prepared {
        rgb: img.to_rgb8().into_raw().into(),
        height: th,
        width: tw,
    })
}

/// The conditioner's input for a reference request.
pub struct RefPrompt {
    pub ids: Vec<u32>,
    /// Per row: text, except each vision block (start, pads, end) as video.
    pub tags: Vec<i32>,
    /// Each image's first `<|image_pad|>` row.
    pub image_starts: Vec<usize>,
    /// Rows the prompt itself tokenized to (the labels and vision blocks
    /// excluded).
    pub prompt_tokens: usize,
}

/// `"<Picture i>: " <|vision_start|> pads <|vision_end|>` per image, then
/// `"<Audio j>: "` per audio clip (a label only: the clip reaches the DiT as
/// audio rows), then the prompt, each segment tokenized on its own with no
/// special tokens. Images then audio is the reference's request order.
pub fn assemble(
    tok: &impl Tokenizer,
    prompt: &str,
    tokens_per_image: &[usize],
    audios: usize,
) -> Result<RefPrompt, String> {
    for (text, id) in [VISION_START, VISION_END, IMAGE_PAD] {
        let got = tok
            .encode(text, false)
            .map_err(|e| format!("tokenize {text}: {e:?}"))?;
        if got != [id] {
            return Err(format!("tokenizer maps {text} to {got:?}, expected [{id}]"));
        }
    }
    let encode = |s: &str| tok.encode(s, false).map_err(|e| format!("tokenize: {e:?}"));
    let (mut ids, mut tags, mut image_starts) = (Vec::new(), Vec::new(), Vec::new());
    for (i, &n) in tokens_per_image.iter().enumerate() {
        let label = encode(&format!("<Picture {}>: ", i + 1))?;
        tags.extend(std::iter::repeat_n(TAG_TEXT, label.len()));
        ids.extend(label);
        ids.push(VISION_START.1);
        image_starts.push(ids.len());
        ids.extend(std::iter::repeat_n(IMAGE_PAD.1, n));
        ids.push(VISION_END.1);
        tags.extend(std::iter::repeat_n(TAG_VIDEO, n + 2));
    }
    for j in 0..audios {
        let label = encode(&format!("<Audio {}>: ", j + 1))?;
        tags.extend(std::iter::repeat_n(TAG_TEXT, label.len()));
        ids.extend(label);
    }
    let text = encode(prompt)?;
    let prompt_tokens = text.len();
    tags.extend(std::iter::repeat_n(TAG_TEXT, text.len()));
    ids.extend(text);
    Ok(RefPrompt {
        ids,
        tags,
        image_starts,
        prompt_tokens,
    })
}

/// Run the vision tower over every prepared reference.
pub async fn vision_features(
    backend: &Arc<WgpuBackend>,
    manifest: &thinfer_core::manifest::ModelManifest,
    budget: ResidencyBudget,
    refs: &[Prepared],
) -> Result<Vec<VisionOutput>, String> {
    let path = crate::download::resolve_role(manifest, role::VISION)?;
    let src = GgufSource::open(open_mmap(&path).await?)
        .await
        .map_err(|e| format!("parse vision mmproj: {e:?}"))?;
    let residency = WeightResidency::new(src, budget);
    let handles =
        vision::register_handles(&residency).map_err(|e| format!("register vision: {e:?}"))?;
    let bp = BlockPipelines::compile(backend, &vision::wgsl_configs())
        .await
        .map_err(|e| format!("compile vision pipelines: {e:?}"))?;
    let ws = Workspace::new(Arc::clone(backend), Arc::clone(residency.arbiter()));
    let mut out = Vec::with_capacity(refs.len());
    for r in refs {
        out.push(
            vision::forward(
                backend, &bp, &residency, &ws, &handles, &r.rgb, r.height, r.width,
            )
            .await
            .map_err(|e| format!("vision forward: {e:?}"))?,
        );
    }
    residency.evict_all_and_free(&**backend);
    Ok(out)
}

/// VAE-encode every image (the reference's recipe) into DiT conditioning
/// blocks and their patchified rows, in order: references (`anchor` `None`)
/// or keyframes pinned at `anchor`.
pub async fn latent_conditions(
    backend: &Arc<WgpuBackend>,
    cfg: &DitConfig,
    vae_path: &std::path::Path,
    budget: ResidencyBudget,
    refs: &[Prepared],
    anchor: Option<KeyframeAnchor>,
    bf16_conv: bool,
) -> Result<(Vec<Condition>, Vec<f32>), String> {
    let raw = open_st(vae_path).await?;
    let channels = cfg.latents_dim;
    let mean = read_vec(&raw, "latents_mean", channels).await?;
    let std = read_vec(&raw, "latents_std", channels).await?;
    let residency = WeightResidency::new(EncoderSource::new(raw), budget);
    let enc = H3VaeEncoder::new(&residency).map_err(|e| format!("register vae encoder: {e:?}"))?;
    let pl = EncoderPipelines::compile(backend)
        .await
        .map_err(|e| format!("compile vae encoder: {e:?}"))?;
    let ws = Workspace::new(Arc::clone(backend), Arc::clone(residency.arbiter()));
    let (mut conditions, mut rows) = (Vec::new(), Vec::new());
    for r in refs {
        let plane = r.height * r.width;
        let mut planar = vec![0f32; 3 * plane];
        for (i, px) in r.rgb.chunks_exact(3).enumerate() {
            for c in 0..3 {
                planar[c * plane + i] = px[c] as f32 / 255.0;
            }
        }
        let frame = enc
            .encode_frame(
                backend,
                &residency,
                &ws,
                &pl,
                &planar,
                r.height,
                r.width,
                (&mean, &std),
                EncodeOptions {
                    bf16_conv,
                    ..EncodeOptions::REFERENCE
                },
                None,
            )
            .await
            .map_err(|e| format!("vae encode reference: {e:?}"))?;
        rows.extend(
            packing::patchify_video_latents(
                &frame.latent,
                channels,
                1,
                frame.height,
                frame.width,
                cfg.patch,
            )
            .map_err(|e| format!("patchify reference: {e:?}"))?,
        );
        conditions.push(match anchor {
            Some(a) => Condition::Keyframe(a),
            None => Condition::Reference {
                latent_height: frame.height,
                latent_width: frame.width,
            },
        });
    }
    residency.evict_all_and_free(&**backend);
    Ok((conditions, rows))
}

/// One reference clip, prepared: `[2, samples]` planar at the audio VAE's
/// rate. User media, so page-locked on a server and zeroed on drop.
pub struct PreparedAudio {
    pub wave: thinfer_core::secmem::Locked<f32>,
    pub samples: usize,
}

impl PreparedAudio {
    /// Audio-VAE latent frames per channel (the hop-padded length).
    pub fn latents(&self) -> usize {
        self.samples.div_ceil(AudioVaeConfig::h3().hop_length())
    }
}

/// Decode every reference clip and cut it to `seconds` (the clip's length,
/// as the reference does), stereo at the audio VAE's rate.
pub fn prepare_audio(
    audios: &[crate::request::AudioBytes],
    seconds: f64,
) -> Result<Vec<PreparedAudio>, String> {
    let rate = AudioVaeConfig::h3().sample_rate;
    audios
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let decoded =
                crate::audio_in::decode(&a.0).map_err(|e| format!("{e} (audio {})", i + 1))?;
            let (wave, samples) = crate::audio_in::prepare(decoded, rate, seconds);
            Ok(PreparedAudio {
                wave: wave.into(),
                samples,
            })
        })
        .collect()
}

/// Audio-VAE-encode every prepared clip into clean DiT conditioning: one
/// [`Condition::AudioReference`] per clip and its channel-major latent rows.
pub async fn audio_conditions(
    backend: &Arc<WgpuBackend>,
    audio_vae_path: &std::path::Path,
    budget: ResidencyBudget,
    audios: &[PreparedAudio],
) -> Result<(Vec<Condition>, Vec<f32>), String> {
    let residency = WeightResidency::new(open_st(audio_vae_path).await?, budget);
    let enc = AudioEncoder::new(&residency, AudioVaeConfig::h3())
        .await
        .map_err(|e| format!("build audio encoder: {e:?}"))?;
    let pl = AudioVae::compile(backend)
        .await
        .map_err(|e| format!("compile audio encoder: {e:?}"))?;
    let ws = Workspace::new(Arc::clone(backend), Arc::clone(residency.arbiter()));
    let (mut conditions, mut rows) = (Vec::new(), Vec::new());
    for (i, a) in audios.iter().enumerate() {
        let (latents, t) = enc
            .encode(backend, &residency, &ws, &pl, &a.wave, a.samples)
            .await
            .map_err(|e| format!("encode reference audio {}: {e:?}", i + 1))?;
        debug_assert_eq!(t, a.latents());
        if !latents.iter().all(|v| v.is_finite()) {
            return Err(format!(
                "reference audio {} encoded to non-finite latents",
                i + 1
            ));
        }
        rows.extend(latents);
        conditions.push(Condition::AudioReference { latents: t });
    }
    residency.evict_all_and_free(&**backend);
    Ok((conditions, rows))
}

async fn read_vec<S: WeightSource>(src: &S, name: &str, n: usize) -> Result<Vec<f32>, String> {
    let id = WeightId(name.into());
    let enc = src
        .catalog()
        .get(&id)
        .and_then(|e| e.encoding)
        .ok_or_else(|| format!("{name}: missing or undecodable"))?;
    let mut r = src.open(&id).await.map_err(|e| format!("{name}: {e:?}"))?;
    let mut bytes = vec![0u8; r.len() as usize];
    r.read_at(0, &mut bytes)
        .await
        .map_err(|e| format!("{name}: {e:?}"))?;
    let mut out = vec![0u8; n * 4];
    decode_into(enc, &bytes, &mut out).map_err(|e| format!("{name}: {e:?}"))?;
    Ok(out
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_round_is_half_to_even() {
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(3.5), 4.0);
        assert_eq!(round_half_even(2.4), 2.0);
        assert_eq!(round_half_even(-2.5), -2.0);
    }

    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(image::RgbImage::new(width, height))
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        png
    }

    /// The spec's worked size: 1024x1536 -> 2048x3072 (6144 vision tokens).
    #[test]
    fn diffusers_scales_the_short_edge_to_2048() {
        let p = prepare(
            decode(&png(1024, 1536)).unwrap(),
            None,
            RefImageSize::Diffusers,
            832,
            480,
        )
        .unwrap();
        assert_eq!((p.width, p.height), (2048, 3072));
        assert_eq!(p.rgb.len(), 2048 * 3072 * 3);
    }

    /// `match` takes the canvas area (sqrt(832*480 / (1024*1536)) = 0.504:
    /// 516x774 -> 512x768); `max` never upscales.
    #[test]
    fn match_and_max_never_upscale() {
        let p = prepare(
            decode(&png(1024, 1536)).unwrap(),
            None,
            RefImageSize::Match,
            832,
            480,
        )
        .unwrap();
        assert_eq!((p.width, p.height), (512, 768));
        let p = prepare(
            decode(&png(320, 480)).unwrap(),
            None,
            RefImageSize::Match,
            832,
            480,
        )
        .unwrap();
        assert_eq!((p.width, p.height), (320, 480));
        let p = prepare(
            decode(&png(1024, 1536)).unwrap(),
            None,
            RefImageSize::Max,
            832,
            480,
        )
        .unwrap();
        assert_eq!((p.width, p.height), (1024, 1536));
    }

    /// A far-away person (face 60 px of a 3000x2000 photo, ~15 px once sized
    /// to an 832x480 canvas) is cropped in on: the face keeps its source
    /// pixels (sizing never upscales), inside the crop, near its top, with
    /// the source aspect kept and as much body as that allows.
    #[test]
    fn auto_crop_zooms_in_on_a_small_face() {
        let face = [1400.0, 600.0, 1450.0, 660.0];
        let r = face_crop(&[face], 3000, 2000, RefImageSize::Match, 832, 480).unwrap();
        let [x, y, w, h] = r.map(|v| v as f32);
        assert!(x <= face[0] && x + w >= face[2] && y <= face[1] && y + h >= face[3]);
        assert!((w / h - 1.5).abs() < 0.01, "{r:?}");
        assert!(face[1] - y < 0.25 * h, "{r:?}");
        let s = RefImageSize::Match.scale(w as usize, h as usize, 832, 480);
        assert!(60.0 * s >= 57.0, "{s} {r:?}");
        assert!(h >= 6.0 * 60.0, "{r:?}");
    }

    /// Big faces, no faces and crops that would keep nearly everything pass
    /// the image whole; a second similar face stays in the crop.
    #[test]
    fn auto_crop_keeps_large_faces_and_couples() {
        let big = [400.0, 300.0, 700.0, 700.0];
        assert_eq!(
            face_crop(&[big], 1024, 1536, RefImageSize::Match, 832, 480),
            None
        );
        assert_eq!(
            face_crop(&[], 1024, 1536, RefImageSize::Match, 832, 480),
            None
        );
        let a = [1000.0, 600.0, 1050.0, 660.0];
        let b = [1800.0, 620.0, 1848.0, 676.0];
        let tiny = [100.0, 100.0, 110.0, 110.0];
        let [x, _, w, _] = face_crop(&[a, b, tiny], 3000, 2000, RefImageSize::Match, 832, 480)
            .unwrap()
            .map(|v| v as f32);
        assert!(x <= a[0] && x + w >= b[2] && x > tiny[2]);
    }
}
