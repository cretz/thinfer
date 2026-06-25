//! Qwen-Image-Edit (image->image) end-to-end HEALTH gate: vision tower ->
//! Qwen2.5-VL edit-encode (LM + 3-axis MRoPE + vision scatter) -> VAE-encode the
//! reference image -> dual-stream DiT (full 60 layers, streamed) FlowMatchEuler
//! denoise over `[noise ++ ref]` tokens (CFG-free) -> Wan KL VAE decode -> RGB.
//!
//! The full-DiT byte-parity pyref is RAM-blocked (20B), so per-component parity
//! (vision/encoder_edit/vae_encode/dit) is the kernel proof; this gate asserts
//! the ASSEMBLED edit pipeline produces a sane, finite, non-degenerate image.
//! The pyref (`gen_edit_inputs`) produces the ENGINE INPUTS (token_ids,
//! image_pad_start, ViT pixels + grid, VAE image + dims) from one tiny
//! deterministic image, isolating assembly from preprocessing. Set
//! `THINFER_E2E_PNG_DIR` to dump the PNG for a visual eyeball.

#![cfg(feature = "qwen-image-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::format::union::{RenamedSource, UnionSource};
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_models::qwen_image::manifest::{self, role};
use thinfer_models::qwen_image::pipeline::QwenImagePipeline;
use thinfer_models::qwen_image::text_encoder::qwen2vl_gguf_renames;
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, read_u32};

const PROMPT: &str = "make the apple green";
// ViT patch grid (8x8 -> N=64 -> n_img=16 merged tokens).
const GH: usize = 8;
const GW: usize = 8;
// VAE reference resolution (32-multiple) and DiT output target.
const VAE_H: usize = 64;
const VAE_W: usize = 64;
const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;
const STEPS: u32 = 4;

#[tokio::test(flavor = "current_thread")]
async fn edit_e2e_health() {
    let _trace = thinfer_core::trace::init_from_env();

    let dit_fr = manifest::MANIFEST
        .get(role::DIT_GGUF_Q8_0)
        .expect("dit role");
    let enc_fr = manifest::MANIFEST
        .get(role::ENCODER_GGUF_Q8_0)
        .expect("enc role");
    let mm_fr = manifest::MANIFEST
        .get(role::MMPROJ_F16)
        .expect("mmproj role");
    let vae_fr = manifest::MANIFEST.get(role::VAE).expect("vae role");
    let tok_fr = manifest::MANIFEST.get(role::TOKENIZER).expect("tok role");
    let (Some(dit_path), Some(enc_path), Some(mmproj_path), Some(vae_path), Some(tok_path)) = (
        cache::resolve(dit_fr),
        cache::resolve(enc_fr),
        cache::resolve(mm_fr),
        cache::resolve(vae_fr),
        cache::resolve(tok_fr),
    ) else {
        eprintln!("skipped[edit_e2e_health]: qwen-image edit bundle not fully in HF cache");
        return;
    };

    // --- engine inputs from one deterministic image (tokenizer-only python) ---
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("qwen_edit_e2e");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    run_inputs(tok_path.parent().expect("tok dir"), &tmp);

    let token_ids = read_u32(&tmp.join("token_ids.bin"));
    assert!(token_ids.len() > 64, "need > edit drop_idx tokens");
    let meta = std::fs::read_to_string(tmp.join("meta.txt")).expect("meta.txt");
    let m: Vec<usize> = meta
        .split_whitespace()
        .take(6)
        .map(|s| s.parse().expect("meta int"))
        .collect();
    let (gh, gw, n_img, pad_start, vae_h, vae_w) = (m[0], m[1], m[2], m[3], m[4], m[5]);
    assert_eq!((gh, gw), (GH, GW));
    assert_eq!((vae_h, vae_w), (VAE_H, VAE_W));
    let pixel_values = read_f32(&tmp.join("pixel_values.bin"));
    assert_eq!(pixel_values.len(), gh * gw * 1176);
    let vae_image = read_f32(&tmp.join("vae_image.bin"));
    assert_eq!(vae_image.len(), 3 * vae_h * vae_w);
    eprintln!(
        "edit-e2e: {} tokens, image_pad_start={pad_start}, n_img={n_img}, \
         ViT {gh}x{gw}, VAE {vae_h}x{vae_w}",
        token_ids.len()
    );

    // --- union: DiT (1:1) + renamed encoder + mmproj (native) + VAE safetensors ---
    let open = |p: PathBuf| async move { MmapFileOpener::new(&p).await.expect("mmap open") };
    let dit_src = GgufSource::open(open(dit_path).await)
        .await
        .expect("dit gguf");
    let enc_src = RenamedSource::with_passthrough(
        GgufSource::open(open(enc_path).await)
            .await
            .expect("enc gguf"),
        qwen2vl_gguf_renames(),
    );
    let mmproj_src = GgufSource::open(open(mmproj_path).await)
        .await
        .expect("mmproj gguf");
    let vae_src = SafetensorsSource::open(open(vae_path).await)
        .await
        .expect("vae safetensors");
    // (((DiT + enc) + mmproj) + VAE): vision tower uses NATIVE mmproj keys.
    let source = UnionSource::new(
        UnionSource::new(UnionSource::new(dit_src, enc_src), mmproj_src),
        vae_src,
    );
    let budget = ResidencyBudget {
        ram_bytes: 48 << 30,
        vram_bytes: 6 << 30,
    };
    let residency = WeightResidency::new(source, budget);

    let backend = Arc::new(
        WgpuBackend::new_with_config(WgpuConfig {
            power_preference: match std::env::var("THINFER_POWER_PREF").as_deref() {
                Ok("low" | "lowpower" | "integrated") => PowerPreference::LowPower,
                Ok("none") => PowerPreference::None,
                _ => PowerPreference::HighPerformance,
            },
            timestamps: std::env::var("THINFER_TRACE").is_ok(),
        })
        .await
        .expect("wgpu adapter unavailable for tests"),
    );

    // i8 DP4A matmul on the DiT by default (the shipped path); QWEN_NO_I8=1 opts out.
    let i8_matmul = std::env::var_os("QWEN_NO_I8").is_none();
    let pipeline = QwenImagePipeline::load(
        Arc::clone(&backend),
        residency,
        token_ids.len() + 2,
        i8_matmul,
        true,
    )
    .await
    .expect("load qwen-image edit pipeline");

    let rgb = pipeline
        .generate_edit_rgb(
            &token_ids,
            pad_start,
            &pixel_values,
            (gh, gw),
            &vae_image,
            (vae_h, vae_w),
            HEIGHT,
            WIDTH,
            STEPS,
            42,
            None,
        )
        .await
        .expect("generate_edit_rgb");

    assert_eq!(rgb.len(), 3 * (HEIGHT as usize) * (WIDTH as usize));
    assert!(rgb.iter().all(|v| v.is_finite()), "non-finite pixels");
    let (mut mn, mut mx, mut sum, mut sq) = (f32::MAX, f32::MIN, 0.0f64, 0.0f64);
    for &v in &rgb {
        mn = mn.min(v);
        mx = mx.max(v);
        sum += v as f64;
        sq += (v as f64) * (v as f64);
    }
    let n = rgb.len() as f64;
    let mean = sum / n;
    let std = (sq / n - mean * mean).max(0.0).sqrt();
    eprintln!("edit-e2e rgb: min={mn:.3} max={mx:.3} mean={mean:.3} std={std:.4}");
    assert!(mn >= -1.05 && mx <= 1.05, "rgb out of [-1,1]: [{mn},{mx}]");
    assert!(std > 0.01, "degenerate (near-constant) image, std={std}");

    if let Some(dir) = std::env::var_os("THINFER_E2E_PNG_DIR") {
        let png =
            thinfer_models::z_image::pipeline::encode_png(&rgb, WIDTH, HEIGHT).expect("png encode");
        let out = Path::new(&dir).join("qwen_image_edit_e2e.png");
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(&out, &png).expect("write png");
        eprintln!("edit-e2e: wrote {}", out.display());
    }
}

fn run_inputs(tok_dir: &Path, out: &Path) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let status = Command::new("uv")
        .args([
            "run",
            "--directory",
            py_dir.to_str().unwrap(),
            "python",
            "-m",
            "thinfer_pytorch_ref.qwen_image.gen_edit_inputs",
            "--tokenizer-dir",
            tok_dir.to_str().unwrap(),
            "--prompt",
            PROMPT,
            "--out",
            out.to_str().unwrap(),
            "--gh",
            &GH.to_string(),
            "--gw",
            &GW.to_string(),
            "--vae-h",
            &VAE_H.to_string(),
            "--vae-w",
            &VAE_W.to_string(),
        ])
        .status()
        .expect("failed to spawn `uv run`");
    assert!(status.success(), "edit inputs pyref failed");
}
