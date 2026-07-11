//! HunyuanVideo 1.5 T2V DiT parity: the engine `HunyuanDit` (full transformer,
//! lightx2v weights fp16->bf16, f32 acts) vs the upstream forward
//! (`gen_dit_ref.py`, same fp16 bytes as f32) at tiny dims. Compares per-stage
//! taps (vec, img_in, txt_in, block0) + the final velocity. The band absorbs the
//! fp16->bf16 weight narrowing compounded over 54 blocks (wider than the refiner
//! gate, like the qwen_image DiT parity).

#![cfg(feature = "hunyuan-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::ops::ActDtype;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::hunyuan::dit::{HunyuanDit, HunyuanDitPipelines, HunyuanDitTaps};
use thinfer_models::hunyuan::refiner::{HunyuanRefiner, HunyuanRefinerPipelines};
use thinfer_native::MmapFileOpener;

use crate::parity_util::{read_f32, report, resolve_hf};

const SEQ: usize = 16;
const T: usize = 3;
const H: usize = 4;
const W: usize = 4;

const SLOPE_TOL: f64 = 0.05;
// fp16->bf16 weight narrowing compounded over 54 dual-stream blocks.
const REL_TOL: f64 = 0.05;

#[tokio::test(flavor = "current_thread")]
async fn dit_parity() {
    let _trace = thinfer_core::trace::init_from_env();

    let Some(dit_path) = resolve_hf(
        "THINFER_HUNYUAN_DIT",
        "models--lightx2v--Hy1.5-Distill-Models",
        "hy1.5_t2v_480p_lightx2v_4step.safetensors",
    ) else {
        eprintln!("skipped[hunyuan dit_parity]: lightx2v DiT not in HF cache");
        return;
    };

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("hunyuan_dit_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("dims.txt");
    let dims = format!("{SEQ} {T} {H} {W}");
    let cached = tmp.join("velocity.bin").exists()
        && std::fs::read_to_string(&marker).is_ok_and(|d| d == dims);
    if cached {
        eprintln!("hunyuan dit-parity: reusing cached pyref dumps");
    } else {
        for ent in std::fs::read_dir(&tmp).into_iter().flatten().flatten() {
            let _ = std::fs::remove_file(ent.path());
        }
        run_python_ref(&dit_path, &tmp);
        std::fs::write(&marker, &dims).expect("write marker");
    }

    let text = read_f32(&tmp.join("text_in.bin"));
    let img_tokens = read_f32(&tmp.join("img_tokens.bin"));
    assert_eq!(text.len(), SEQ * 3584, "text size");
    assert_eq!(img_tokens.len(), T * H * W * 65, "img tokens size");

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

    let opener = MmapFileOpener::new(&dit_path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", dit_path.display()));
    let src = SafetensorsSource::open(opener)
        .await
        .expect("parse DiT safetensors");
    let residency = WeightResidency::new(
        src,
        ResidencyBudget {
            ram_bytes: 24 << 30,
            vram_bytes: 5 << 30,
        },
    );

    let refiner_pl = HunyuanRefinerPipelines::compile_with(&backend, ActDtype::F32)
        .await
        .expect("compile refiner pipelines");
    let refiner = HunyuanRefiner::new(refiner_pl, &residency).expect("build refiner");
    // DiT runs bf16 acts (+ mixed-precision f16 SDPA); the refiner stays F32
    // and the DiT<->refiner handoff is host f32, so the two are independent.
    // i8 DP4A on (self-attn q/k/v + ffn-up) is the SHIPPING default; the band
    // absorbs it. `THINFER_HY_I8=0` checks the pure bf16 reference path.
    let i8 = std::env::var("THINFER_HY_I8")
        .map(|v| v != "0")
        .unwrap_or(true);
    let dit_pl = HunyuanDitPipelines::compile_with(&backend, ActDtype::Bf16, i8)
        .await
        .expect("compile dit pipelines");
    let dit = HunyuanDit::new(dit_pl, refiner, &residency, i8).expect("build dit");
    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    let mut vec = Vec::new();
    let mut img_in = Vec::new();
    let mut txt_in = Vec::new();
    let mut block0_img = Vec::new();
    let mut block0_txt = Vec::new();
    let mut taps = HunyuanDitTaps {
        vec: Some(&mut vec),
        img_in: Some(&mut img_in),
        txt_in: Some(&mut txt_in),
        block0_img: Some(&mut block0_img),
        block0_txt: Some(&mut block0_txt),
    };
    let velocity = dit
        .forward(
            &backend,
            &residency,
            &workspace,
            &text,
            SEQ,
            &img_tokens,
            (T, H, W),
            500.0,
            0, // full attention (parity)
            Some(&mut taps),
        )
        .await
        .expect("dit forward");

    eprintln!("---- hunyuan dit parity ----");
    let mut failures = Vec::new();
    let mut check = |label: &str, exp: &[f32], got: &[f32], rel_tol: f64| {
        assert_eq!(
            exp.len(),
            got.len(),
            "[{label}] len exp={} got={}",
            exp.len(),
            got.len()
        );
        let (slope, rel) = report(label, exp, got);
        if !slope.is_finite() || (slope - 1.0).abs() > SLOPE_TOL {
            failures.push(format!("{label} slope off: {slope:.6}"));
        }
        if rel > rel_tol {
            failures.push(format!(
                "{label} rel={:.3}% > {:.3}%",
                rel * 100.0,
                rel_tol * 100.0
            ));
        }
    };

    // Per-stage taps validate i8 faithfulness at the tight bf16 band (the i8
    // sites are in every block; block0 already exercises both). The final
    // velocity at these tiny near-zero-mean dims amplifies relative error
    // (bf16 ~4.9%, i8 ~8.8%); its band widens under i8 accordingly. The real
    // i8 quality signal is e2e (larger dims) + the serve eyeball.
    let vel_tol = if i8 { 0.10 } else { REL_TOL };
    check("vec", &read_f32(&tmp.join("vec.bin")), &vec, REL_TOL);
    check(
        "img_in",
        &read_f32(&tmp.join("img_in.bin")),
        &img_in,
        REL_TOL,
    );
    check(
        "txt_in",
        &read_f32(&tmp.join("txt_in.bin")),
        &txt_in,
        REL_TOL,
    );
    check(
        "block0_img",
        &read_f32(&tmp.join("block0_img.bin")),
        &block0_img,
        REL_TOL,
    );
    check(
        "block0_txt",
        &read_f32(&tmp.join("block0_txt.bin")),
        &block0_txt,
        REL_TOL,
    );
    check(
        "velocity",
        &read_f32(&tmp.join("velocity.bin")),
        &velocity,
        vel_tol,
    );
    assert_eq!(velocity.len(), T * H * W * 32, "velocity size");

    assert!(
        failures.is_empty(),
        "hunyuan dit parity:\n{}",
        failures.join("\n")
    );
}

fn run_python_ref(dit: &Path, out_dir: &Path) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let status = Command::new("uv")
        .args([
            "run",
            "--directory",
            py_dir.to_str().unwrap(),
            "--with",
            "einops",
            "--with",
            "loguru",
            "python",
            "-m",
            "thinfer_pytorch_ref.hunyuan.gen_dit_ref",
            "--dit",
            dit.to_str().unwrap(),
            "--out",
            out_dir.to_str().unwrap(),
            "--seq",
            &SEQ.to_string(),
            "--t",
            &T.to_string(),
            "--h",
            &H.to_string(),
            "--w",
            &W.to_string(),
        ])
        .status()
        .expect("failed to spawn `uv run`");
    assert!(status.success(), "hunyuan dit pyref failed");
}
