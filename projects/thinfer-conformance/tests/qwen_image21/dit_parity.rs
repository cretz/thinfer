//! Qwen-Image-2.1 DiT parity at truncated depth: the engine's prefill (text +
//! condition image, K / V cached once) + per-step target forward vs the
//! uncached block-causal diffusers `QwenImage21Transformer2DModel` dequantized
//! from the SAME GGUF (`gen_dit_ref.py`), fed identical seeded inputs. Two
//! timesteps over one prefix exercise the KV-cache reuse.

#![cfg(feature = "qwen-image-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::qwen_image21::config;
use thinfer_models::qwen_image21::dit::{KvPlacement, PrefixPart, Qwen21Dit, Qwen21DitPipelines};
use thinfer_models::qwen_image21::loader::register_handles;
use thinfer_models::qwen_image21::manifest::{MANIFEST, role};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, report};

const LAYERS: usize = 2;
const TEXT_A: usize = 6;
const COND: (usize, usize) = (4, 4);
const TEXT_B: usize = 5;
const TARGET: (usize, usize) = (4, 6);
const TIMESTEPS: [f32; 2] = [0.95418, 0.414568];

// Engine acts vs the bf16 reference, same Q8_0 weights. Measured rel 0.87 /
// 0.89%, slope 0.998 (2026-11-10); re-measure on a config change.
const VEL_SLOPE_TOL: f64 = 0.01;
const VEL_REL_TOL: f64 = 0.02;

#[tokio::test(flavor = "current_thread")]
async fn dit_parity() {
    let _trace = thinfer_core::trace::init_from_env();

    let fr = MANIFEST.get(role::DIT_GGUF_Q8_0).expect("dit gguf role");
    let Some(gguf_path) = cache::resolve(fr) else {
        eprintln!(
            "skipped[dit_parity]: {}/{} not in HF cache",
            fr.repo, fr.path
        );
        return;
    };

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("qwen21_dit_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("config.txt");
    let cfg_tag = format!("{LAYERS} {TEXT_A} {COND:?} {TEXT_B} {TARGET:?} {TIMESTEPS:?}");
    if std::fs::read_to_string(&marker).is_ok_and(|c| c == cfg_tag) {
        eprintln!("dit-parity: reusing cached pyref dumps ({})", tmp.display());
    } else {
        run_python_ref(&gguf_path, &tmp);
        std::fs::write(&marker, &cfg_tag).expect("write marker");
    }

    let cond_slots = COND.0 * COND.1 / config::TOKENS_PER_SLOT;
    let encoder = read_f32(&tmp.join("encoder.bin"));
    let cond = read_f32(&tmp.join("cond.bin"));
    let target = read_f32(&tmp.join("target.bin"));
    assert_eq!(
        encoder.len(),
        (TEXT_A + cond_slots + TEXT_B) * config::TEXT_DIM
    );
    assert_eq!(cond.len(), COND.0 * COND.1 * config::LATENT_CH);
    assert_eq!(target.len(), TARGET.0 * TARGET.1 * config::LATENT_CH);

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
    let pipes = Qwen21DitPipelines::compile(&backend)
        .await
        .expect("compile dit pipelines");
    let dit = Qwen21Dit::new();

    let parts = [
        PrefixPart::Text {
            start: 0,
            len: TEXT_A,
        },
        PrefixPart::Image {
            start: 0,
            h: COND.0,
            w: COND.1,
        },
        PrefixPart::Text {
            start: TEXT_A + cond_slots,
            len: TEXT_B,
        },
    ];
    let opener = MmapFileOpener::new(&gguf_path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", gguf_path.display()));
    let residency = WeightResidency::new(
        GgufSource::open(opener).await.expect("parse dit gguf"),
        ResidencyBudget {
            ram_bytes: 4 << 30,
            vram_bytes: 4 << 30,
        },
    );
    let handles = register_handles(&residency, LAYERS).expect("register dit handles");
    let ws = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    // The prefix K / V on the device, then spilled to the host.
    let mut failures = Vec::new();
    for kv in [KvPlacement::Device, KvPlacement::Host] {
        let prefix = dit
            .prefill(
                &backend, &pipes, &residency, &ws, &handles, &encoder, &cond, &parts, TARGET, kv,
            )
            .await
            .expect("engine prefill");
        assert_eq!(prefix.spilled(), kv == KvPlacement::Host);

        eprintln!("---- qwen-image-2.1 dit parity ({LAYERS} layers, kv {kv:?}) ----");
        for (i, &t) in TIMESTEPS.iter().enumerate() {
            let got = dit
                .step(
                    &backend, &pipes, &residency, &ws, &handles, &prefix, &target, t,
                )
                .await
                .expect("engine step");
            let exp = read_f32(&tmp.join(format!("vel_{i}.bin")));
            let (slope, rel) = report(&format!("velocity t={t}"), &exp, &got);
            if (slope - 1.0).abs() > VEL_SLOPE_TOL || rel > VEL_REL_TOL {
                failures.push(format!(
                    "kv {kv:?} t={t}: slope={slope:.6} rel={:.3}% (tol slope 1+-{VEL_SLOPE_TOL}, rel {VEL_REL_TOL})",
                    rel * 100.0
                ));
            }
        }
    }
    assert!(failures.is_empty(), "dit parity:\n{}", failures.join("\n"));
}

fn run_python_ref(gguf: &Path, out_dir: &Path) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let mut cmd = Command::new("uv");
    cmd.args(["run", "--directory", py_dir.to_str().unwrap()]);
    // The third-party diffusers (Qwen-Image-2.1) needs newer hub/transformers
    // than the project venv pins.
    for dep in ["gguf", "huggingface_hub>=2", "transformers>5.8"] {
        cmd.args(["--with", dep]);
    }
    cmd.args([
        "python",
        "-m",
        "thinfer_pytorch_ref.qwen_image21.gen_dit_ref",
        "--gguf",
        gguf.to_str().unwrap(),
        "--out",
        out_dir.to_str().unwrap(),
        "--layers",
        &LAYERS.to_string(),
        "--text-a",
        &TEXT_A.to_string(),
        "--cond",
        &COND.0.to_string(),
        &COND.1.to_string(),
        "--text-b",
        &TEXT_B.to_string(),
        "--target",
        &TARGET.0.to_string(),
        &TARGET.1.to_string(),
        "--timesteps",
    ]);
    cmd.args(TIMESTEPS.map(|t| t.to_string()));
    let status = cmd
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "qwen-image-2.1 dit pyref failed");
}
