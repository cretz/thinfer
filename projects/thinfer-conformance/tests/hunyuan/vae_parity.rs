//! HunyuanVideo 1.5 VAE decode parity (MILESTONE 1a): the engine
//! `HunyuanVaeDecoder` (F16 safetensors weights narrowed to bf16, F32 acts) vs
//! the upstream `Decoder` (`gen_vae_decode_ref.py`, same F16 bytes, f32 compute)
//! through conv_in + mid. Decodes a fixed-seed normalized latent at f=1 (causal
//! mid-attn trivial) and compares the conv_in (+repeat_interleave residual) and
//! mid bands. Same bytes both sides -> bit-tight; the only drift is GPU f32 conv
//! accumulation order vs torch. 1b extends to the full video.

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
use thinfer_models::hunyuan::vae::{
    HunyuanVaeDecoder, HunyuanVaePipelines, HunyuanVaeTaps, LATENT_CHANNELS,
};
use thinfer_native::MmapFileOpener;

use crate::parity_util::{read_f32, report, resolve_hf};

// f=2 exercises the temporal upsample multi-frame path + causal mid-attn mask
// (f=1 left only the first-frame/spatial paths).
const F: usize = 2;
const H: usize = 8;
const W: usize = 8;

const SLOPE_TOL: f64 = 0.01;
// Same F16->bf16 weights both sides; bit-tight (GPU f32 conv accumulation order
// vs torch only). Generous headroom over the measured drift.
const REL_TOL: f64 = 0.005;

#[tokio::test(flavor = "current_thread")]
async fn vae_parity() {
    let _trace = thinfer_core::trace::init_from_env();

    let Some(vae_path) = resolve_hf(
        "THINFER_HUNYUAN_VAE",
        "models--Comfy-Org--HunyuanVideo_1.5_repackaged",
        "split_files/vae/hunyuanvideo15_vae_fp16.safetensors",
    ) else {
        eprintln!("skipped[hunyuan vae_parity]: VAE not in HF cache");
        return;
    };

    // --- python reference (cached by dims) ---
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("hunyuan_vae_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("dims.txt");
    let dims = format!("{F} {H} {W}");
    let cached =
        tmp.join("mid.bin").exists() && std::fs::read_to_string(&marker).is_ok_and(|d| d == dims);
    if cached {
        eprintln!(
            "hunyuan vae-parity: reusing cached pyref dumps ({})",
            tmp.display()
        );
    } else {
        for ent in std::fs::read_dir(&tmp).into_iter().flatten().flatten() {
            let _ = std::fs::remove_file(ent.path());
        }
        run_python_ref(&vae_path, &tmp);
        std::fs::write(&marker, &dims).expect("write marker");
    }

    let latent = read_f32(&tmp.join("latent.bin"));
    assert_eq!(latent.len(), LATENT_CHANNELS * F * H * W, "latent size");

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

    // Parity pins f32 acts (bit-exact reference path).
    let pipelines = HunyuanVaePipelines::compile_with(&backend, ActDtype::F32)
        .await
        .expect("compile vae pipelines");

    let opener = MmapFileOpener::new(&vae_path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", vae_path.display()));
    let src = SafetensorsSource::open(opener)
        .await
        .expect("parse vae safetensors");
    let residency = WeightResidency::new(
        src,
        ResidencyBudget {
            ram_bytes: 16 << 30,
            vram_bytes: 5 << 30,
        },
    );
    let decoder = HunyuanVaeDecoder::new(pipelines, &residency).expect("build decoder");
    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    let mut conv_in = Vec::new();
    let mut mid_block1 = Vec::new();
    let mut mid_attn = Vec::new();
    let mut mid = Vec::new();
    let mut up: Vec<Vec<f32>> = Vec::new();
    let mut conv_out = Vec::new();
    let mut taps = HunyuanVaeTaps {
        conv_in: Some(&mut conv_in),
        mid_block1: Some(&mut mid_block1),
        mid_attn: Some(&mut mid_attn),
        mid: Some(&mut mid),
        up: Some(&mut up),
        conv_out: Some(&mut conv_out),
    };
    let video = decoder
        .decode_with_taps(
            &backend,
            &residency,
            &workspace,
            &latent,
            F,
            H,
            W,
            Some(&mut taps),
        )
        .await
        .expect("vae decode");

    eprintln!("---- hunyuan vae parity (1a) ----");
    let mut failures = Vec::new();
    let mut check = |label: &str, exp: &[f32], got: &[f32]| {
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
        if rel > REL_TOL {
            failures.push(format!(
                "{label} rel={:.3}% > {:.3}%",
                rel * 100.0,
                REL_TOL * 100.0
            ));
        }
    };

    check("conv_in", &read_f32(&tmp.join("conv_in.bin")), &conv_in);
    check(
        "mid_block1",
        &read_f32(&tmp.join("mid_block1.bin")),
        &mid_block1,
    );
    check("mid_attn", &read_f32(&tmp.join("mid_attn.bin")), &mid_attn);
    check("mid", &read_f32(&tmp.join("mid.bin")), &mid);
    for (i, got) in up.iter().enumerate() {
        check(
            &format!("up_{i:02}"),
            &read_f32(&tmp.join(format!("up_{i:02}.bin"))),
            got,
        );
    }
    check("conv_out", &read_f32(&tmp.join("conv_out.bin")), &conv_out);
    check("video", &read_f32(&tmp.join("video.bin")), &video);

    assert!(
        failures.is_empty(),
        "hunyuan vae parity (slope 1+-{SLOPE_TOL}):\n{}",
        failures.join("\n")
    );
}

fn run_python_ref(vae: &Path, out_dir: &Path) {
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
            "thinfer_pytorch_ref.hunyuan.gen_vae_decode_ref",
            "--vae",
            vae.to_str().unwrap(),
            "--out",
            out_dir.to_str().unwrap(),
            "--frames",
            &F.to_string(),
            "--height",
            &H.to_string(),
            "--width",
            &W.to_string(),
        ])
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "hunyuan vae pyref failed");
}
