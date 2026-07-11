//! Qwen-Image t2i end-to-end health gate: tokenize -> Qwen2.5-VL encode ->
//! dual-stream DiT (full 60 layers, streamed) FlowMatchEuler denoise -> Wan KL
//! VAE decode -> RGB. The full-DiT byte-parity pyref is RAM-blocked (20B bf16 ~
//! 41GB), so per-component parity (vae/encoder/dit) is the kernel proof; this
//! gate asserts the assembled pipeline produces a sane, finite, non-degenerate
//! image. Set `THINFER_E2E_PNG_DIR` to dump the PNG for a visual eyeball.

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

use crate::parity_util::read_u32;

const PROMPT: &str = "a red apple on a wooden table";
const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;
const STEPS: u32 = 4;

#[tokio::test(flavor = "current_thread")]
async fn e2e_health() {
    let _trace = thinfer_core::trace::init_from_env();

    let dit_fr = manifest::MANIFEST
        .get(role::DIT_GGUF_Q8_0)
        .expect("dit role");
    let enc_fr = manifest::MANIFEST
        .get(role::ENCODER_GGUF_Q8_0)
        .expect("enc role");
    let vae_fr = manifest::MANIFEST.get(role::VAE).expect("vae role");
    let tok_fr = manifest::MANIFEST.get(role::TOKENIZER).expect("tok role");
    let (Some(dit_path), Some(enc_path), Some(vae_path), Some(tok_path)) = (
        cache::resolve(dit_fr),
        cache::resolve(enc_fr),
        cache::resolve(vae_fr),
        cache::resolve(tok_fr),
    ) else {
        eprintln!("skipped[e2e_health]: qwen-image bundle not fully in HF cache");
        return;
    };

    // --- tokenize (fast, tokenizer-only python) ---
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("qwen_e2e");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let ids_path = tmp.join("token_ids.bin");
    run_tokenizer(tok_path.parent().expect("tok dir"), &ids_path);
    let token_ids = read_u32(&ids_path);
    assert!(token_ids.len() > 34, "need > drop_idx tokens");
    eprintln!("e2e: {} tokens", token_ids.len());

    // --- union residency: DiT (1:1) + renamed encoder + renamed VAE ---
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
    let vae_src = SafetensorsSource::open(open(vae_path).await)
        .await
        .expect("vae safetensors");
    // t2i load (`edit=false`) skips the vision tower + VAE-encoder, so the
    // residency unions just DiT + encoder + VAE (no mmproj).
    let source = UnionSource::new(UnionSource::new(dit_src, enc_src), vae_src);
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
            disable_coopmat: std::env::var("THINFER_NO_COOPMAT").is_ok(),
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
        false,
    )
    .await
    .expect("load qwen-image pipeline");

    let rgb = pipeline
        .generate_rgb(&token_ids, HEIGHT, WIDTH, STEPS, 42, None)
        .await
        .expect("generate");

    assert_eq!(rgb.len(), 3 * (HEIGHT as usize) * (WIDTH as usize));
    // sanity: finite, in-range, non-degenerate.
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
    eprintln!("e2e rgb: min={mn:.3} max={mx:.3} mean={mean:.3} std={std:.4}");
    assert!(mn >= -1.05 && mx <= 1.05, "rgb out of [-1,1]: [{mn},{mx}]");
    assert!(std > 0.01, "degenerate (near-constant) image, std={std}");

    if let Some(dir) = std::env::var_os("THINFER_E2E_PNG_DIR") {
        let png =
            thinfer_models::z_image::pipeline::encode_png(&rgb, WIDTH, HEIGHT).expect("png encode");
        let out = Path::new(&dir).join("qwen_image_e2e.png");
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(&out, &png).expect("write png");
        eprintln!("e2e: wrote {}", out.display());
    }
}

fn run_tokenizer(tok_dir: &Path, out: &Path) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let status = Command::new("uv")
        .args([
            "run",
            "--directory",
            py_dir.to_str().unwrap(),
            "python",
            "-m",
            "thinfer_pytorch_ref.qwen_image.tokenize_prompt",
            "--tokenizer-dir",
            tok_dir.to_str().unwrap(),
            "--prompt",
            PROMPT,
            "--out",
            out.to_str().unwrap(),
        ])
        .status()
        .expect("failed to spawn `uv run`");
    assert!(status.success(), "tokenizer pyref failed");
}
