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
//! Both see the same prepared image: EXIF-oriented RGB, Lanczos-resized so the
//! short edge is 2048 and both sides are multiples of 32 (no area cap; the
//! vision processor's own resize is then the identity).

use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::residency::WeightResidency;
use thinfer_core::tokenizer::Tokenizer;
use thinfer_core::weight::{WeightId, WeightReader, WeightSource, decode_into};
use thinfer_core::workspace::Workspace;
use thinfer_models::common::block::BlockPipelines;
use thinfer_models::minimax::manifest::role;
use thinfer_models::minimax::packing::{self, Condition};
use thinfer_models::minimax::vae_encoder::{
    EncodeOptions, EncoderPipelines, EncoderSource, H3VaeEncoder,
};
use thinfer_models::minimax::vision::{self, VisionOutput};
use thinfer_models::minimax::{DitConfig, TAG_TEXT, TAG_VIDEO};

use crate::config::ResidencyBudget;
use crate::minimax::{open_mmap, open_st};

/// Short edge every reference is resized to, and the side multiple.
const SHORT_EDGE: f64 = 2048.0;
const SIDE_MULTIPLE: usize = 32;
/// The reference rejects anything flatter than 4:1 either way.
const MAX_ASPECT: usize = 4;
/// At most this many images per request (the reference's limit).
pub const MAX_REFERENCES: usize = 9;

/// Qwen2 tokenizer special tokens (verified against the tokenizer at run time).
const VISION_START: (&str, u32) = ("<|vision_start|>", 151_652);
const VISION_END: (&str, u32) = ("<|vision_end|>", 151_653);
const IMAGE_PAD: (&str, u32) = ("<|image_pad|>", 151_655);

/// One reference image, prepared: RGB8 `[height, width, 3]`. User media, so
/// zeroed on drop like the request's [`crate::request::ImageBytes`].
pub struct Prepared {
    pub rgb: Vec<u8>,
    pub height: usize,
    pub width: usize,
}

impl Drop for Prepared {
    fn drop(&mut self) {
        zeroize::Zeroize::zeroize(&mut self.rgb);
    }
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

/// Decode, orient and resize one reference image.
pub fn prepare(bytes: &[u8]) -> Result<Prepared, String> {
    use image::ImageDecoder;
    let mut decoder = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("reference image: {e}"))?
        .into_decoder()
        .map_err(|e| format!("reference image: {e}"))?;
    let orientation = decoder
        .orientation()
        .map_err(|e| format!("reference image orientation: {e}"))?;
    let mut img =
        image::DynamicImage::from_decoder(decoder).map_err(|e| format!("reference image: {e}"))?;
    img.apply_orientation(orientation);
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w > MAX_ASPECT * h || h > MAX_ASPECT * w {
        return Err(format!(
            "reference image {w}x{h} is flatter than {MAX_ASPECT}:1"
        ));
    }
    let scale = SHORT_EDGE / w.min(h) as f64;
    let m = SIDE_MULTIPLE as f64;
    let side = |x: usize| (round_half_even(x as f64 * scale / m) * m).max(m) as usize;
    let (tw, th) = (side(w), side(h));
    if (tw, th) != (w, h) {
        img = img.resize_exact(tw as u32, th as u32, image::imageops::FilterType::Lanczos3);
    }
    Ok(Prepared {
        rgb: img.to_rgb8().into_raw(),
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
}

/// `"<Picture i>: " <|vision_start|> pads <|vision_end|>` per image, then the
/// prompt, each segment tokenized on its own with no special tokens.
pub fn assemble(
    tok: &impl Tokenizer,
    prompt: &str,
    tokens_per_image: &[usize],
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
    let text = encode(prompt)?;
    tags.extend(std::iter::repeat_n(TAG_TEXT, text.len()));
    ids.extend(text);
    Ok(RefPrompt {
        ids,
        tags,
        image_starts,
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

/// VAE-encode every reference (the reference's recipe) into DiT conditioning
/// blocks and their patchified rows, in order.
pub async fn latent_conditions(
    backend: &Arc<WgpuBackend>,
    cfg: &DitConfig,
    vae_path: &std::path::Path,
    budget: ResidencyBudget,
    refs: &[Prepared],
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
                EncodeOptions::REFERENCE,
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
        conditions.push(Condition::Reference {
            latent_height: frame.height,
            latent_width: frame.width,
        });
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

    /// The spec's worked size: 1024x1536 -> 2048x3072 (6144 vision tokens).
    #[test]
    fn prepare_scales_the_short_edge_to_2048() {
        let img = image::RgbImage::new(1024, 1536);
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let p = prepare(&png).unwrap();
        assert_eq!((p.width, p.height), (2048, 3072));
        assert_eq!(p.rgb.len(), 2048 * 3072 * 3);
    }
}
