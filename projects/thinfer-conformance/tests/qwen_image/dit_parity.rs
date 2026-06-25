//! Qwen-Image DiT single-block parity: the engine dual-stream block + embedders
//! (Q8_0 block matmuls, F16->bf16 embedders, bf16 acts) vs a 1-layer diffusers
//! `QwenImageTransformer2DModel` dequantized from the SAME GGUF
//! (`gen_dit_ref.py`), fed identical seeded inputs. The 60-layer DiT is just
//! repetition of this block, so a green single block validates every kernel
//! (modulation, joint attention, complex RoPE, GELU MLP, norm_out/proj_out).

#![cfg(feature = "qwen-image-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::qwen_image::dit::{DitTaps, QwenImageDit, QwenImageDitPipelines};
use thinfer_models::qwen_image::loader::register_handles;
use thinfer_models::qwen_image::manifest::{self, role};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, report};

const WIDTH: usize = 64;
const HEIGHT: usize = 64;
const TXT_SEQ: usize = 8;
const TIMESTEP: f32 = 500.0;

// bf16-act engine vs bf16 reference, same Q8_0 weights. Single block; re-measure
// + re-document on a config change, do not loosen to go green.
const TEMB_REL_TOL: f64 = 0.03;
const BLOCK0_REL_TOL: f64 = 0.06;
const VEL_SLOPE_TOL: f64 = 0.06;
const VEL_REL_TOL: f64 = 0.15;

#[tokio::test(flavor = "current_thread")]
async fn dit_parity() {
    let _trace = thinfer_core::trace::init_from_env();

    let fr = manifest::MANIFEST
        .get(role::DIT_GGUF_Q8_0)
        .expect("dit gguf role");
    let Some(gguf_path) = cache::resolve(fr) else {
        eprintln!(
            "skipped[dit_parity]: {}/{} not in HF cache",
            fr.repo, fr.path
        );
        return;
    };

    // --- python reference (deterministic single-block) ---
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("qwen_dit_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("config.txt");
    let cfg_tag = format!("{WIDTH}x{HEIGHT}x{TXT_SEQ}x{TIMESTEP}");
    let cached = tmp.join("velocity.bin").exists()
        && std::fs::read_to_string(&marker).is_ok_and(|c| c == cfg_tag);
    if cached {
        eprintln!("dit-parity: reusing cached pyref dumps ({})", tmp.display());
    } else {
        for ent in std::fs::read_dir(&tmp).into_iter().flatten().flatten() {
            let _ = std::fs::remove_file(ent.path());
        }
        run_python_ref(&gguf_path, &tmp);
        std::fs::write(&marker, &cfg_tag).expect("write marker");
    }

    let meta = std::fs::read_to_string(tmp.join("meta.txt")).expect("meta.txt");
    let m: Vec<usize> = meta
        .split_whitespace()
        .take(4)
        .map(|s| s.parse().expect("meta int"))
        .collect();
    let (img_seq, txt_seq, gh, gw) = (m[0], m[1], m[2], m[3]);
    eprintln!("dit-parity: img_seq={img_seq} txt_seq={txt_seq} grid={gh}x{gw}");

    let img_tokens = read_f32(&tmp.join("img_tokens.bin"));
    let txt_embeds = read_f32(&tmp.join("txt_embeds.bin"));
    assert_eq!(img_tokens.len(), img_seq * 64);
    assert_eq!(txt_embeds.len(), txt_seq * 3584);
    let exp_temb = read_f32(&tmp.join("temb.bin"));
    let exp_hs = read_f32(&tmp.join("block0_hs.bin"));
    let exp_eh = read_f32(&tmp.join("block0_eh.bin"));
    let exp_vel = read_f32(&tmp.join("velocity.bin"));

    // --- engine: open the DiT GGUF (1:1 keys), single block ---
    let opener = MmapFileOpener::new(&gguf_path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", gguf_path.display()));
    let gguf = GgufSource::open(opener).await.expect("parse dit gguf");
    let budget = ResidencyBudget {
        ram_bytes: 24 << 30,
        vram_bytes: 6 << 30,
    };
    let residency = WeightResidency::new(gguf, budget);
    let handles = register_handles(&residency, 1).expect("register dit handles (1 block)");

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

    // The shipped DiT config (bf16 acts, per-site Q8_0 matmuls), validated vs the
    // bf16 pyref.
    let cfgs = thinfer_models::qwen_image::dit::block_cfgs();
    // Fast-attention (f16 subgroup SDPA) default-on; QWEN_NO_I8 forces bf16.
    let i8_matmul = std::env::var_os("QWEN_NO_I8").is_none();
    let pipelines = QwenImageDitPipelines::compile(&backend, &cfgs, i8_matmul)
        .await
        .expect("compile dit pipelines");

    let scratch = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));
    let dit = QwenImageDit::new();
    let mut taps = DitTaps::default();
    let out = dit
        .forward(
            &backend,
            &pipelines,
            &residency,
            &scratch,
            &handles,
            &img_tokens,
            &txt_embeds,
            TIMESTEP,
            1,
            gh,
            gw,
            Some(&mut taps),
        )
        .await
        .expect("engine dit forward");
    assert_eq!(out.velocity.len(), img_seq * 64);

    eprintln!("---- qwen-image dit parity ----");
    let (_, temb_rel) = report("temb", &exp_temb, taps.temb.as_ref().expect("temb tap"));
    let (_, hs_rel) = report(
        "block0_hs",
        &exp_hs,
        taps.block0_img.as_ref().expect("hs tap"),
    );
    let (_, eh_rel) = report(
        "block0_eh",
        &exp_eh,
        taps.block0_txt.as_ref().expect("eh tap"),
    );
    let (vel_slope, vel_rel) = report("velocity", &exp_vel, &out.velocity);

    let mut failures = Vec::new();
    if temb_rel > TEMB_REL_TOL {
        failures.push(format!(
            "temb rel {:.3}% > {TEMB_REL_TOL}",
            temb_rel * 100.0
        ));
    }
    if hs_rel > BLOCK0_REL_TOL {
        failures.push(format!(
            "block0_hs rel {:.3}% > {BLOCK0_REL_TOL}",
            hs_rel * 100.0
        ));
    }
    if eh_rel > BLOCK0_REL_TOL {
        failures.push(format!(
            "block0_eh rel {:.3}% > {BLOCK0_REL_TOL}",
            eh_rel * 100.0
        ));
    }
    if (vel_slope - 1.0).abs() > VEL_SLOPE_TOL || vel_rel > VEL_REL_TOL {
        failures.push(format!(
            "velocity diverges: slope={vel_slope:.6} rel={:.3}% (tol slope 1+-{VEL_SLOPE_TOL}, rel {VEL_REL_TOL})",
            vel_rel * 100.0
        ));
    }
    assert!(failures.is_empty(), "dit parity:\n{}", failures.join("\n"));
}

fn run_python_ref(gguf: &Path, out_dir: &Path) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let status = Command::new("uv")
        .args([
            "run",
            "--directory",
            py_dir.to_str().unwrap(),
            "--with",
            "gguf",
            "python",
            "-m",
            "thinfer_pytorch_ref.qwen_image.gen_dit_ref",
            "--gguf",
            gguf.to_str().unwrap(),
            "--out",
            out_dir.to_str().unwrap(),
            "--width",
            &WIDTH.to_string(),
            "--height",
            &HEIGHT.to_string(),
            "--txt-seq",
            &TXT_SEQ.to_string(),
            "--timestep",
            &TIMESTEP.to_string(),
        ])
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "qwen-image dit pyref failed");
}
