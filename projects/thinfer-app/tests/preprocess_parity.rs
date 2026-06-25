//! Qwen-Image-Edit host PREPROCESS parity: `thinfer_app::preprocess` (Rust) on a
//! real PNG vs the authoritative HF `Qwen2VLImageProcessor` + diffusers
//! `VaeImageProcessor` + tokenizer (`gen_preprocess_ref.py`). The engine edit
//! path is already health-green; this is the durable proof the HOST feeds it
//! correct tensors.
//!
//! token_ids must match EXACTLY (tokenization is deterministic). vit_pixels and
//! vae_image use a relative-rmse band (resampling-filter differences between
//! image-rs and PIL are allowed; PIL bicubic/lanczos vs CatmullRom/Lanczos3).
//!
//! Gated on `preprocess-parity` (needs `uv` + the cached tokenizer/processor).
#![cfg(feature = "preprocess-parity")]

use std::path::{Path, PathBuf};
use std::process::Command;

use thinfer_app::preprocess::{prepare_edit_inputs, tokenize_t2i};
use thinfer_models::qwen_image::manifest::{self, role};
use thinfer_native::cache;
use thinfer_native::tokenizer::HfTokenizer;

const VIT_REL_TOL: f64 = 0.02; // 2% (resampling-filter differences allowed)
const VAE_REL_TOL: f64 = 0.02;

fn read_f32(p: &Path) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_u32(p: &Path) -> Vec<u32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Relative rmse: rmse(exp, got) / mean(|exp|).
fn rel_rmse(exp: &[f32], got: &[f32]) -> f64 {
    assert_eq!(exp.len(), got.len(), "length mismatch");
    let n = exp.len().max(1) as f64;
    let mse = exp
        .iter()
        .zip(got)
        .map(|(&a, &b)| ((a - b) as f64).powi(2))
        .sum::<f64>()
        / n;
    let mean_abs = exp.iter().map(|x| x.abs() as f64).sum::<f64>() / n;
    if mean_abs > 0.0 {
        mse.sqrt() / mean_abs
    } else {
        0.0
    }
}

/// A deterministic RGB PNG (a smooth gradient + a couple of blocks so resampling
/// has real high-frequency content but stays filter-robust within 2%).
fn write_test_png(path: &Path, w: u32, h: u32) {
    let img = image::RgbImage::from_fn(w, h, |x, y| {
        let r = (x * 255 / w.max(1)) as u8;
        let g = (y * 255 / h.max(1)) as u8;
        let b = (((x + y) % 64) * 4) as u8;
        image::Rgb([r, g, b])
    });
    img.save(path).expect("write test png");
}

#[tokio::test(flavor = "current_thread")]
async fn preprocess_parity() {
    // Resolve the cached tokenizer.json; its parent snapshot dir also holds
    // preprocessor_config.json (same repo), so it serves as processor-dir too.
    let tok_fr = manifest::MANIFEST
        .get(role::TOKENIZER)
        .expect("tokenizer role");
    let prep_fr = manifest::MANIFEST
        .get(role::PREPROCESSOR)
        .expect("preprocessor role");
    let (Some(tok_path), Some(_prep_path)) = (cache::resolve(tok_fr), cache::resolve(prep_fr))
    else {
        eprintln!("skipped[preprocess_parity]: tokenizer/preprocessor not in HF cache");
        return;
    };
    let snapshot_dir = tok_path.parent().expect("tokenizer parent dir");

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("qwen_preprocess_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let png = tmp.join("input.png");
    // Non-square so VAE/ViT aspect handling is exercised; mid-size keeps the run
    // fast while still landing on real 28/32 grids.
    write_test_png(&png, 200, 140);
    let prompt = "make the sky a vivid sunset orange";

    run_python_ref(&png, snapshot_dir, snapshot_dir, prompt, &tmp);

    let meta = std::fs::read_to_string(tmp.join("meta.txt")).expect("meta.txt");
    let m: Vec<usize> = meta
        .split_whitespace()
        .map(|s| s.parse().expect("meta int"))
        .collect();
    let (gh, gw, n_img, pad_start, hv, wv) = (m[0], m[1], m[2], m[3], m[4], m[5]);
    eprintln!("preprocess-parity: ViT={gh}x{gw} n_img={n_img} pad_start={pad_start} VAE={hv}x{wv}");

    let ref_pixels = read_f32(&tmp.join("pixel_values.bin"));
    let ref_vae = read_f32(&tmp.join("vae_image.bin"));
    let ref_ids = read_u32(&tmp.join("token_ids.bin"));

    // --- Rust preprocessing on the SAME PNG ---
    let tokenizer = HfTokenizer::from_path(&tok_path)
        .await
        .expect("load tokenizer");
    let rgb = image::open(&png).expect("decode png").to_rgb8();
    let got = prepare_edit_inputs(&rgb, prompt, &tokenizer).expect("rust preprocess");

    assert_eq!(got.vit_grid, (gh, gw), "ViT grid mismatch");
    assert_eq!(got.vae_dims, (hv, wv), "VAE dims mismatch");
    assert_eq!(got.image_pad_start, pad_start, "image_pad_start mismatch");

    // token_ids EXACT.
    assert_eq!(
        got.token_ids.len(),
        ref_ids.len(),
        "token count: rust {} vs py {}",
        got.token_ids.len(),
        ref_ids.len()
    );
    let mismatches = got
        .token_ids
        .iter()
        .zip(&ref_ids)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        mismatches, 0,
        "token_ids must match exactly ({mismatches} differ)"
    );

    // vit_pixels + vae_image within band.
    let vit_rel = rel_rmse(&ref_pixels, &got.vit_pixels);
    let vae_rel = rel_rmse(&ref_vae, &got.vae_image);
    eprintln!(
        "preprocess-parity: token_ids EXACT ({} tokens); vit rel={:.3}% vae rel={:.3}%",
        ref_ids.len(),
        vit_rel * 100.0,
        vae_rel * 100.0
    );
    assert!(
        vit_rel < VIT_REL_TOL,
        "vit_pixels rel {:.3}% > {:.1}%",
        vit_rel * 100.0,
        VIT_REL_TOL * 100.0
    );
    assert!(
        vae_rel < VAE_REL_TOL,
        "vae_image rel {:.3}% > {:.1}%",
        vae_rel * 100.0,
        VAE_REL_TOL * 100.0
    );
}

/// The text-to-image path has no image channel, so its only host step is the
/// t2i-template tokenize. Prove `tokenize_t2i` matches `tokenize_prompt.py`
/// (the same template the diffusers t2i pipeline uses) byte-for-byte.
#[tokio::test(flavor = "current_thread")]
async fn t2i_tokenize_parity() {
    let tok_fr = manifest::MANIFEST
        .get(role::TOKENIZER)
        .expect("tokenizer role");
    let Some(tok_path) = cache::resolve(tok_fr) else {
        eprintln!("skipped[t2i_tokenize_parity]: tokenizer not in HF cache");
        return;
    };
    let tok_dir = tok_path.parent().expect("tokenizer parent dir");

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("qwen_t2i_tokenize_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let prompt = "a red apple on a wooden table";
    let ids_path = tmp.join("token_ids.bin");
    run_t2i_tokenizer_ref(tok_dir, prompt, &ids_path);
    let ref_ids = read_u32(&ids_path);

    let tokenizer = HfTokenizer::from_path(&tok_path)
        .await
        .expect("load tokenizer");
    let got = tokenize_t2i(prompt, &tokenizer).expect("rust t2i tokenize");

    assert_eq!(
        got.len(),
        ref_ids.len(),
        "token count: rust {} vs py {}",
        got.len(),
        ref_ids.len()
    );
    assert_eq!(got, ref_ids, "t2i token_ids must match exactly");
    eprintln!(
        "t2i-tokenize-parity: token_ids EXACT ({} tokens)",
        got.len()
    );
}

fn run_t2i_tokenizer_ref(tokenizer_dir: &Path, prompt: &str, out: &Path) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../thinfer-conformance/python");
    let status = Command::new("uv")
        .args([
            "run",
            "--directory",
            py_dir.to_str().unwrap(),
            "python",
            "-m",
            "thinfer_pytorch_ref.qwen_image.tokenize_prompt",
            "--tokenizer-dir",
            tokenizer_dir.to_str().unwrap(),
            "--prompt",
            prompt,
            "--out",
            out.to_str().unwrap(),
        ])
        .status()
        .expect("spawn `uv run` (is uv installed?)");
    assert!(status.success(), "qwen-image t2i tokenize pyref failed");
}

fn run_python_ref(
    image: &Path,
    processor_dir: &Path,
    tokenizer_dir: &Path,
    prompt: &str,
    out: &Path,
) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../thinfer-conformance/python");
    let status = Command::new("uv")
        .args([
            "run",
            "--directory",
            py_dir.to_str().unwrap(),
            "python",
            "-m",
            "thinfer_pytorch_ref.qwen_image.gen_preprocess_ref",
            "--image",
            image.to_str().unwrap(),
            "--processor-dir",
            processor_dir.to_str().unwrap(),
            "--tokenizer-dir",
            tokenizer_dir.to_str().unwrap(),
            "--prompt",
            prompt,
            "--out",
            out.to_str().unwrap(),
        ])
        .status()
        .expect("spawn `uv run` (is uv installed?)");
    assert!(status.success(), "qwen-image preprocess pyref failed");
}
