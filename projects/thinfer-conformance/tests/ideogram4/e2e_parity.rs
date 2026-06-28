//! Ideogram-4 full end-to-end parity: the engine pipeline (Qwen3-VL-8B encode
//! -> LoRA-folded DiT no-CFG Euler loop -> Flux2 KL VAE decode) vs a STAGED
//! pyref (`gen_e2e_ref.py`, one process per stage so peak host RAM is ~one
//! model). Both sides consume identical Q8_0 GGUF bytes + the same turbotime
//! LoRA fold (`W += B@A`); the engine injects the pyref-dumped seeded noise and
//! token ids so the only band is the engine's f16-act vs the bf16/fp32 staged
//! reference, compounded across the whole pipeline (loose).
//!
//! Run (needs the 4 HF files + `uv`):
//!   THINFER_POWER_PREF=high cargo test -p thinfer-conformance \
//!     --features ideogram4-e2e --release e2e_parity -- --nocapture --test-threads=1

#![cfg(feature = "ideogram4-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::format::union::{RenamedSource, UnionSource};
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_models::ideogram4::lora::LoraFoldSource;
use thinfer_models::ideogram4::manifest::{self, role};
use thinfer_models::ideogram4::pipeline::{GenerationParams, Ideogram4Pipeline};
use thinfer_models::z_image::qwen3_gguf_renames;
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, read_u32, report};

// Tiny parity dims: 64x64 -> 4x4 image grid. 4 steps (turbotime).
const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;
const STEPS: u32 = 4;
const SEED: u64 = 0;
const MU: f64 = 0.5;
const STD: f64 = 1.75;
const PROMPT: &str = "a red apple on a wooden table";

// Whole-pipeline f16-act engine vs bf16/fp32 staged pyref. Loose band: the DiT
// f16-vs-bf16 residual drift compounds over 34 blocks x 4 steps before the VAE.
// Re-measure + re-document on a config change; do not loosen to go green.
const SLOPE_TOL: f64 = 0.10;
const REL_TOL: f64 = 0.20;

#[tokio::test(flavor = "current_thread")]
async fn e2e_parity() {
    let _trace = thinfer_core::trace::init_from_env();

    let (Some(enc), Some(dit), Some(vae), Some(lora)) = (
        manifest::MANIFEST
            .get(role::ENCODER_GGUF_Q8_0)
            .and_then(cache::resolve),
        manifest::MANIFEST
            .get(role::DIT_GGUF_Q8_0)
            .and_then(cache::resolve),
        manifest::MANIFEST.get(role::VAE).and_then(cache::resolve),
        manifest::MANIFEST.get(role::LORA).and_then(cache::resolve),
    ) else {
        eprintln!("skipped[e2e_parity]: one of encoder/dit/vae/lora not in HF cache");
        return;
    };

    // --- staged pyref (one process per stage; cache by config tag) ---
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ideo_e2e_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("config.txt");
    let cfg_tag = format!("{WIDTH}x{HEIGHT}s{STEPS}seed{SEED}mu{MU}std{STD}::{PROMPT}");
    let cached = tmp.join("decoded.bin").exists()
        && std::fs::read_to_string(&marker).is_ok_and(|c| c == cfg_tag);
    if cached {
        eprintln!("e2e-parity: reusing cached pyref dumps ({})", tmp.display());
    } else {
        for ent in std::fs::read_dir(&tmp).into_iter().flatten().flatten() {
            let _ = std::fs::remove_file(ent.path());
        }
        run_stage(
            &["--stage", "encode", "--enc-gguf", enc.to_str().unwrap()],
            &tmp,
            "gguf",
        );
        run_stage(
            &[
                "--stage",
                "dit",
                "--dit-gguf",
                dit.to_str().unwrap(),
                "--lora",
                lora.to_str().unwrap(),
            ],
            &tmp,
            "gguf",
        );
        run_stage(
            &["--stage", "vae", "--vae", vae.to_str().unwrap()],
            &tmp,
            "einops",
        );
        std::fs::write(&marker, &cfg_tag).expect("write marker");
    }

    let token_ids = read_u32(&tmp.join("token_ids.bin"));
    let noise = read_f32(&tmp.join("noise.bin"));
    let decoded_ref = read_f32(&tmp.join("decoded.bin"));
    let meta: Vec<usize> = std::fs::read_to_string(tmp.join("meta_dit.txt"))
        .expect("meta_dit")
        .split_whitespace()
        .take(3)
        .map(|s| s.parse().unwrap())
        .collect();
    let (num_image, grid_h, grid_w) = (meta[0], meta[1], meta[2]);
    assert_eq!(noise.len(), num_image * 128);
    assert_eq!(decoded_ref.len(), 3 * (grid_h * 16) * (grid_w * 16));
    eprintln!(
        "e2e-parity: {} tokens, grid {grid_h}x{grid_w}, {num_image} image tokens",
        token_ids.len()
    );

    // --- engine: union(encoder-renamed-gguf, lora-folded-dit-gguf, vae-st) ---
    let enc_src = RenamedSource::with_passthrough(
        GgufSource::open(open(&enc).await).await.expect("enc gguf"),
        qwen3_gguf_renames(),
    );
    let dit_src = LoraFoldSource::new(
        GgufSource::open(open(&dit).await).await.expect("dit gguf"),
        SafetensorsSource::open(open(&lora).await)
            .await
            .expect("lora st"),
        thinfer_core::quant::QuantKind::Q8_0,
    );
    eprintln!("e2e-parity: lora fold sites = {}", dit_src.fold_count());
    let vae_src = SafetensorsSource::open(open(&vae).await)
        .await
        .expect("vae st");
    let source = UnionSource::new(UnionSource::new(dit_src, enc_src), vae_src);

    // Budget: the LoRA fold materializes the full DiT as Q8_0 in host RAM
    // (~9GB compute-once cache); give the residency RAM headroom + a <8GB VRAM
    // ceiling (the 5070 OOMs at 8GB). Phase-aware eviction keeps peak ~one model.
    let budget = ResidencyBudget {
        ram_bytes: 20 << 30,
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

    // Parity runs the bf16/dequant-once reference path (i8_matmul = false) so the
    // gate stays byte-comparable to the staged pyref; the DP4A path is exercised
    // separately (and is quality-neutral by the same A-side-normed argument as
    // the video i8 lesson).
    let pipeline = Ideogram4Pipeline::load(Arc::clone(&backend), residency, false)
        .await
        .expect("load ideogram4 pipeline");

    let params = GenerationParams {
        token_ids,
        height: HEIGHT,
        width: WIDTH,
        steps: STEPS,
        seed: SEED,
        mu: MU,
        std: STD,
    };
    let (rgb, _z, grid) = pipeline
        .generate_rgb(&params, Some(&noise), None)
        .await
        .expect("engine generate");
    assert_eq!(grid.grid_h, grid_h);
    assert_eq!(rgb.len(), decoded_ref.len());

    eprintln!("---- ideogram e2e parity ----");
    let (slope, rel) = report("decoded", &decoded_ref, &rgb);
    let mut failures = Vec::new();
    if (slope - 1.0).abs() > SLOPE_TOL || rel > REL_TOL {
        failures.push(format!(
            "decoded diverges: slope={slope:.6} rel={:.3}% (tol slope 1+-{SLOPE_TOL}, rel {REL_TOL})",
            rel * 100.0
        ));
    }
    assert!(failures.is_empty(), "e2e parity:\n{}", failures.join("\n"));
}

async fn open(path: &Path) -> MmapFileOpener {
    MmapFileOpener::new(path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()))
}

fn run_stage(stage_args: &[&str], out_dir: &Path, with: &str) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let mut args: Vec<String> = vec![
        "run".into(),
        "--directory".into(),
        py_dir.to_str().unwrap().into(),
        "--with".into(),
        with.into(),
        "python".into(),
        "-m".into(),
        "thinfer_pytorch_ref.ideogram4.gen_e2e_ref".into(),
        "--out".into(),
        out_dir.to_str().unwrap().into(),
        "--width".into(),
        WIDTH.to_string(),
        "--height".into(),
        HEIGHT.to_string(),
        "--steps".into(),
        STEPS.to_string(),
        "--seed".into(),
        SEED.to_string(),
        "--mu".into(),
        MU.to_string(),
        "--std".into(),
        STD.to_string(),
        "--prompt".into(),
        PROMPT.into(),
    ];
    args.extend(stage_args.iter().map(|s| s.to_string()));
    let status = Command::new("uv")
        .args(&args)
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(
        status.success(),
        "ideogram e2e pyref stage failed: {stage_args:?}"
    );
}
