//! Host-side input preparation for the Qwen-Image-Edit path. The engine's
//! `generate_edit_rgb` consumes preprocessed tensors; this module derives them
//! from a single source image + the edit prompt, mirroring `diffusers`
//! `pipeline_qwenimage_edit.py` (VAE-side `calculate_dimensions`, 32-grid) and
//! `transformers` `Qwen2VLImageProcessor` (ViT-side `smart_resize`, 28-grid).
//!
//! Two conditioning channels share the SAME source image at DIFFERENT
//! resolutions:
//!   * ViT channel: `[N, 1176]` patchified pixels (merge-unit-major layout, the
//!     HF processor's `permute(0,2,5,3,6,1,4,7)`), grid `gh x gw`, plus the
//!     edit-templated tokens with `<|image_pad|>` expanded to `(gh/2)*(gw/2)`.
//!   * VAE channel: `[3, 1, Hv, Wv]` CTHW image in `[-1, 1]` at a 32-multiple.
//!
//! The reference (parity-checked) is `gen_edit_inputs.py` / `gen_vision_ref.py`
//! / `gen_encoder_edit_ref.py` in `thinfer-conformance`.

use image::imageops::FilterType;
use image::{DynamicImage, RgbImage};

use thinfer_core::tokenizer::Tokenizer;

/// `<|image_pad|>` token id (Qwen2.5-VL vocab). The single template placeholder
/// expands to this id `n_img` times; we locate the contiguous run to recover
/// `image_pad_start`.
const IMAGE_PAD_TOKEN_ID: u32 = 151655;

/// The single literal `<|image_pad|>` placeholder string (expanded `n_img`x
/// before tokenization, exactly as the HF processor does).
const IMAGE_PAD: &str = "<|image_pad|>";

/// ViT patch side (raw patch in pixels).
const PATCH: usize = 14;
/// Spatial merge factor (2x2 merge unit).
const MERGE: usize = 2;
/// smart_resize factor = patch * merge.
const VIT_FACTOR: usize = PATCH * MERGE; // 28
/// Temporal patch size (a still image duplicates the single frame).
const TEMPORAL: usize = 2;
/// Patchified row width = C * T * P * P.
const PATCH_ELEMS: usize = 3 * TEMPORAL * PATCH * PATCH; // 1176

const VIT_MIN_PIXELS: usize = 56 * 56; // 3136
const VIT_MAX_PIXELS: usize = 28 * 28 * 1280; // 1003520

/// CLIP normalization (OpenAI defaults; the Qwen2-VL processor uses these).
const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// The text-to-image chat template (`pipeline_qwenimage.py`, drop_idx 34). No
/// image channel, so the prompt slots straight into the user turn. Mirrors
/// `tokenize_prompt.py::PROMPT_TEMPLATE`.
const T2I_TEMPLATE_PREFIX: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";
const T2I_TEMPLATE_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";

/// The edit chat template (`pipeline_qwenimage_edit.py`, drop_idx 64). The
/// single `<|image_pad|>` expands to `n_img` placeholders before tokenization.
const EDIT_TEMPLATE_PREFIX: &str = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|>";
const EDIT_TEMPLATE_MID: &str = "<|vision_end|>";
const EDIT_TEMPLATE_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";

/// Everything the edit pipeline needs from the host. Layouts match
/// `generate_edit_rgb`'s parameters exactly.
#[derive(Clone, Debug)]
pub struct EditInputs {
    /// Edit-templated tokens, `<|image_pad|>` expanded to `(gh/2)*(gw/2)`.
    pub token_ids: Vec<u32>,
    /// Index of the first `<|image_pad|>` token (the contiguous run start).
    pub image_pad_start: usize,
    /// ViT patches `[N=gh*gw, 1176]`, merge-unit-major, CLIP-normalized.
    pub vit_pixels: Vec<f32>,
    /// ViT grid `(gh, gw)` (raw patches: H/14, W/14).
    pub vit_grid: (usize, usize),
    /// VAE image CTHW `[3, 1, Hv, Wv]` in `[-1, 1]`.
    pub vae_image: Vec<f32>,
    /// VAE dims `(Hv, Wv)`, both multiples of 32.
    pub vae_dims: (usize, usize),
}

/// VAE-side target dimensions for a `in_w x in_h` source: `diffusers`
/// `calculate_dimensions(1024*1024, in_w/in_h)`, each rounded to a multiple of
/// 32. Returns `(Hv, Wv)`.
pub fn calc_vae_dims(in_w: u32, in_h: u32) -> (usize, usize) {
    const TARGET_AREA: f64 = 1024.0 * 1024.0;
    let ratio = in_w as f64 / in_h as f64;
    let width = (TARGET_AREA * ratio).sqrt();
    let height = width / ratio;
    let wv = (((width / 32.0).round() as usize) * 32).max(32);
    let hv = (((height / 32.0).round() as usize) * 32).max(32);
    (hv, wv)
}

/// ViT-side target dimensions (`smart_resize`, factor 28, CLIP min/max px).
/// Returns `(Hp, Wp)`, both multiples of 28. Mirrors
/// `image_processing_qwen2_vl.smart_resize`.
pub fn calc_vit_dims(in_w: u32, in_h: u32) -> (usize, usize) {
    let (h, w) = (in_h as f64, in_w as f64);
    let f = VIT_FACTOR as f64;
    let mut h_bar = (h / f).round() * f;
    let mut w_bar = (w / f).round() * f;
    if h_bar * w_bar > VIT_MAX_PIXELS as f64 {
        let beta = ((h * w) / VIT_MAX_PIXELS as f64).sqrt();
        h_bar = ((h / beta / f).floor() * f).max(f);
        w_bar = ((w / beta / f).floor() * f).max(f);
    } else if h_bar * w_bar < VIT_MIN_PIXELS as f64 {
        let beta = (VIT_MIN_PIXELS as f64 / (h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    (h_bar as usize, w_bar as usize)
}

/// Build all edit-path inputs from a decoded RGB source image + prompt.
pub fn prepare_edit_inputs(
    rgb: &RgbImage,
    prompt: &str,
    tokenizer: &impl Tokenizer,
) -> Result<EditInputs, String> {
    let (in_w, in_h) = (rgb.width(), rgb.height());

    // --- ViT channel ---
    let (hp, wp) = calc_vit_dims(in_w, in_h);
    let gh = hp / PATCH;
    let gw = wp / PATCH;
    if !gh.is_multiple_of(MERGE) || !gw.is_multiple_of(MERGE) {
        return Err(format!(
            "ViT grid {gh}x{gw} not divisible by merge {MERGE} (smart_resize invariant broken)"
        ));
    }
    // PIL default resample is BICUBIC; `CatmullRom` is image-rs's bicubic.
    let vit_img = DynamicImage::ImageRgb8(rgb.clone())
        .resize_exact(wp as u32, hp as u32, FilterType::CatmullRom)
        .to_rgb8();
    let vit_pixels = patchify(&vit_img, gh, gw);

    // --- VAE channel ---
    let (hv, wv) = calc_vae_dims(in_w, in_h);
    // Diffusers `VaeImageProcessor` default resample is lanczos; `Lanczos3` is
    // image-rs's closest match (resampling-filter differences are within the
    // parity band).
    let vae_img = DynamicImage::ImageRgb8(rgb.clone())
        .resize_exact(wv as u32, hv as u32, FilterType::Lanczos3)
        .to_rgb8();
    let vae_image = to_vae_cthw(&vae_img, hv, wv);

    // --- tokens ---
    let n_img = (gh / MERGE) * (gw / MERGE);
    let (token_ids, image_pad_start) = tokenize_edit(prompt, n_img, tokenizer)?;

    Ok(EditInputs {
        token_ids,
        image_pad_start,
        vit_pixels,
        vit_grid: (gh, gw),
        vae_image,
        vae_dims: (hv, wv),
    })
}

/// Patchify `[3, H, W]` -> `[N=gh*gw, 1176]` merge-unit-major (HF processor
/// `permute(0,2,5,3,6,1,4,7)` then temporal-expand). Normalizes
/// `(px/255 - mean)/std` per channel first. Each consecutive 4 rows = one 2x2
/// merge unit. Row layout is `[C, T, P, P]` with T a pure repeat of the frame.
fn patchify(img: &RgbImage, gh: usize, gw: usize) -> Vec<f32> {
    let w = gw * PATCH;
    let n = gh * gw;
    let mut out = vec![0.0f32; n * PATCH_ELEMS];
    let mgh = gh / MERGE;
    let mgw = gw / MERGE;
    // Iterate output rows in merge-unit-major order: for each merge cell
    // (mh, mw), the 2x2 sub-patches in (sh, sw) order.
    let mut row = 0usize;
    for mh in 0..mgh {
        for mw in 0..mgw {
            for sh in 0..MERGE {
                for sw in 0..MERGE {
                    let ph = mh * MERGE + sh; // patch row index in [0, gh)
                    let pw = mw * MERGE + sw; // patch col index in [0, gw)
                    let base = row * PATCH_ELEMS;
                    // Row layout: [C, T, P, P]. Element order within a row is
                    // c-major, then t, then py, then px.
                    let mut e = base;
                    for c in 0..3usize {
                        for _t in 0..TEMPORAL {
                            for py in 0..PATCH {
                                for px in 0..PATCH {
                                    let y = ph * PATCH + py;
                                    let x = pw * PATCH + px;
                                    let p = img.get_pixel(x as u32, y as u32);
                                    let v = p[c] as f32 / 255.0;
                                    out[e] = (v - CLIP_MEAN[c]) / CLIP_STD[c];
                                    e += 1;
                                }
                            }
                        }
                    }
                    let _ = w; // silence: w only documents the layout
                    row += 1;
                }
            }
        }
    }
    out
}

/// Resize-result RGB `[H, W]` -> CTHW `[3, 1, H, W]` in `[-1, 1]` via
/// `px/127.5 - 1`.
fn to_vae_cthw(img: &RgbImage, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; 3 * h * w];
    for c in 0..3usize {
        let plane = c * h * w;
        for y in 0..h {
            for x in 0..w {
                let p = img.get_pixel(x as u32, y as u32);
                out[plane + y * w + x] = p[c] as f32 / 127.5 - 1.0;
            }
        }
    }
    out
}

/// Tokenize the text-to-image template (no image channel). Mirrors
/// `tokenize_prompt.py`: wrap the prompt in the t2i chat template and encode
/// with `add_special_tokens=false`. The pipeline drops the leading
/// `DROP_IDX = 34` template tokens from the encoder output.
pub fn tokenize_t2i(prompt: &str, tokenizer: &impl Tokenizer) -> Result<Vec<u32>, String> {
    let text = format!("{T2I_TEMPLATE_PREFIX}{prompt}{T2I_TEMPLATE_SUFFIX}");
    let ids = tokenizer
        .encode(&text, false)
        .map_err(|e| format!("tokenize t2i prompt: {e:?}"))?;
    if ids.is_empty() {
        return Err("t2i template tokenized to nothing".into());
    }
    Ok(ids)
}

/// Build + tokenize the edit template (image_pad expanded to `n_img`); return
/// `(token_ids, image_pad_start)`. Mirrors `gen_edit_inputs.py`: encode with
/// `add_special_tokens=false`, then find the contiguous `<|image_pad|>` run.
fn tokenize_edit(
    prompt: &str,
    n_img: usize,
    tokenizer: &impl Tokenizer,
) -> Result<(Vec<u32>, usize), String> {
    let pads = IMAGE_PAD.repeat(n_img);
    let text =
        format!("{EDIT_TEMPLATE_PREFIX}{pads}{EDIT_TEMPLATE_MID}{prompt}{EDIT_TEMPLATE_SUFFIX}");
    let ids = tokenizer
        .encode(&text, false)
        .map_err(|e| format!("tokenize edit prompt: {e:?}"))?;
    if ids.is_empty() {
        return Err("edit template tokenized to nothing".into());
    }
    let pad_positions: Vec<usize> = ids
        .iter()
        .enumerate()
        .filter(|&(_, &t)| t == IMAGE_PAD_TOKEN_ID)
        .map(|(i, _)| i)
        .collect();
    if pad_positions.len() != n_img {
        return Err(format!(
            "expected {n_img} <|image_pad|> tokens, got {} (special tokens not single-id?)",
            pad_positions.len()
        ));
    }
    let start = pad_positions[0];
    if pad_positions != (start..start + n_img).collect::<Vec<_>>() {
        return Err("image-pad tokens are not contiguous".into());
    }
    Ok((ids, start))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vae_dims_square_1024() {
        // 512x512 -> ratio 1 -> width=height=1024, rounded /32 = 1024.
        assert_eq!(calc_vae_dims(512, 512), (1024, 1024));
    }

    #[test]
    fn vae_dims_preserve_aspect_and_mult32() {
        let (hv, wv) = calc_vae_dims(1024, 512); // ratio 2 (landscape)
        assert!(hv.is_multiple_of(32) && wv.is_multiple_of(32));
        assert!(wv > hv, "landscape stays wider");
        // area near 1024*1024.
        let area = hv * wv;
        assert!((area as f64 - 1024.0 * 1024.0).abs() / (1024.0 * 1024.0) < 0.05);
    }

    #[test]
    fn vit_dims_mult_28_and_in_px_range() {
        for &(w, h) in &[
            (64u32, 64u32),
            (256, 256),
            (1024, 768),
            (37, 200),
            (4000, 30),
        ] {
            let (hp, wp) = calc_vit_dims(w, h);
            assert!(
                hp.is_multiple_of(28) && wp.is_multiple_of(28),
                "{w}x{h} -> {hp}x{wp}"
            );
            let px = hp * wp;
            assert!(px >= VIT_MIN_PIXELS, "{w}x{h}: {px} < min");
            assert!(px <= VIT_MAX_PIXELS, "{w}x{h}: {px} > max");
        }
    }

    #[test]
    fn vit_small_image_upscales_to_min() {
        // 14x14 source: round(14/28)*28 = 0 -> clamps up via min_pixels branch.
        let (hp, wp) = calc_vit_dims(14, 14);
        assert!(hp * wp >= VIT_MIN_PIXELS);
        assert!(hp.is_multiple_of(28) && wp.is_multiple_of(28));
    }

    #[test]
    fn patchify_shape_and_merge_order() {
        let (gh, gw) = (4usize, 4usize);
        let img = RgbImage::from_fn((gw * PATCH) as u32, (gh * PATCH) as u32, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 0])
        });
        let px = patchify(&img, gh, gw);
        assert_eq!(px.len(), gh * gw * PATCH_ELEMS);
        // Merge-unit-major: rows 0..4 are the four sub-patches of merge cell
        // (0,0): patch coords (0,0),(0,1),(1,0),(1,1). Row 0's first R-channel
        // value is normalized px[0,0]=0; verify it differs from row 1 (px col 14).
        let r0 = px[0];
        let r1 = px[PATCH_ELEMS]; // row 1, channel 0, first element
        // row1 = sub-patch (0,1) -> x starts at 14 -> R=(14/255-mean)/std.
        let expect_r1 = (14.0 / 255.0 - CLIP_MEAN[0]) / CLIP_STD[0];
        assert!((r1 - expect_r1).abs() < 1e-5, "{r1} vs {expect_r1}");
        let expect_r0 = (0.0 / 255.0 - CLIP_MEAN[0]) / CLIP_STD[0];
        assert!((r0 - expect_r0).abs() < 1e-5);
    }

    #[test]
    fn vae_cthw_range_and_layout() {
        let img = RgbImage::from_fn(8, 8, |x, _| image::Rgb([(x * 32) as u8, 0, 255]));
        let v = to_vae_cthw(&img, 8, 8);
        assert_eq!(v.len(), 3 * 8 * 8);
        // channel 2 is all 255 -> 255/127.5 - 1 = 1.0.
        assert!((v[2 * 64] - 1.0).abs() < 1e-6);
        // channel 0 at x=0 is 0 -> -1.0.
        assert!((v[0] + 1.0).abs() < 1e-6);
        for &x in &v {
            assert!((-1.0..=1.0).contains(&x));
        }
    }
}
