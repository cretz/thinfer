//! Krea 2 Turbo t2i end-to-end health + perf gate: tokenize -> Qwen3-VL-4B encode
//! (12 taps) -> txtfusion -> single-stream DiT (28 layers, streamed) FlowMatch
//! Euler denoise -> Wan2.1 KL VAE decode -> RGB. The full-DiT byte-parity pyref
//! is RAM-blocked (12.9B), so this gate asserts the assembled pipeline produces a
//! sane, finite, non-degenerate image; per-component parity follows. Set
//! `THINFER_E2E_PNG_DIR` to dump the PNG, `THINFER_TRACE=1` for the perf rollup,
//! and `THINFER_E2E_{WIDTH,HEIGHT,STEPS}` to scale (default 64x64x8).

#![cfg(feature = "krea-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::format::union::{RenamedSource, UnionSource};
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::quant::QuantKind;
use thinfer_core::residency::WeightResidency;
use thinfer_models::krea::manifest::{self, role};
use thinfer_models::krea::pipeline::KreaPipeline;
use thinfer_models::z_image::qwen3_gguf_renames;
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

const PROMPT: &str = "a red apple on a wooden table";

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn read_u32(p: &Path) -> Vec<u32> {
    let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[tokio::test(flavor = "current_thread")]
async fn e2e_health() {
    let trace = thinfer_core::trace::init_from_env();
    let width = env_u32("THINFER_E2E_WIDTH", 64);
    let height = env_u32("THINFER_E2E_HEIGHT", 64);
    let steps = env_u32("THINFER_E2E_STEPS", 8);

    let dit_fr = manifest::MANIFEST
        .get(role::DIT_GGUF_Q8_0)
        .expect("dit role");
    let enc_fr = manifest::MANIFEST
        .get(role::ENCODER_GGUF)
        .expect("enc role");
    let vae_fr = manifest::MANIFEST.get(role::VAE).expect("vae role");
    let tok_fr = manifest::MANIFEST.get(role::TOKENIZER).expect("tok role");
    let (Some(dit_path), Some(enc_path), Some(vae_path), Some(tok_path)) = (
        cache::resolve(dit_fr),
        cache::resolve(enc_fr),
        cache::resolve(vae_fr),
        cache::resolve(tok_fr),
    ) else {
        eprintln!("skipped[e2e_health]: krea bundle not fully in HF cache");
        return;
    };

    // --- tokenize (Krea shares the Qwen-Image t2i template) ---
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("krea_e2e");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let ids_path = tmp.join("token_ids.bin");
    run_tokenizer(tok_path.parent().expect("tok dir"), &ids_path);
    let token_ids = read_u32(&ids_path);
    assert!(token_ids.len() > 34, "need > drop_idx tokens");
    eprintln!("e2e: {} tokens, {height}x{width}x{steps}", token_ids.len());

    // --- union residency: DiT (1:1 krea2 keys) + renamed encoder + VAE ---
    let open = |p: PathBuf| async move { MmapFileOpener::new(&p).await.expect("mmap open") };
    let dit_src = GgufSource::open(open(dit_path).await)
        .await
        .expect("dit gguf");
    let enc_src = RenamedSource::with_passthrough(
        GgufSource::open(open(enc_path).await)
            .await
            .expect("enc gguf"),
        qwen3_gguf_renames(),
    );
    let vae_src = SafetensorsSource::open(open(vae_path).await)
        .await
        .expect("vae safetensors");
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

    let pipeline = KreaPipeline::load(
        Arc::clone(&backend),
        residency,
        token_ids.len() + 2,
        QuantKind::Q8_0,
    )
    .await
    .expect("load krea pipeline");

    let rgb = pipeline
        .generate_rgb(&token_ids, height, width, steps, 42, None)
        .await
        .expect("generate");

    assert_eq!(rgb.len(), 3 * (height as usize) * (width as usize));
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
            thinfer_models::z_image::pipeline::encode_png(&rgb, width, height).expect("png encode");
        let out = Path::new(&dir).join("krea_e2e.png");
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(&out, &png).expect("write png");
        eprintln!("e2e: wrote {}", out.display());
    }

    if let Some(h) = trace.as_ref() {
        let mut buf = Vec::new();
        h.dump(&mut buf).ok();
        eprintln!("{}", String::from_utf8_lossy(&buf));
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
