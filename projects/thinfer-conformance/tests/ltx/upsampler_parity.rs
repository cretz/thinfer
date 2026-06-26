//! LTX-2.3 latent spatial upscaler parity: the engine `LtxUpsampler` (bf16
//! safetensors weights, F32 acts) vs the upstream `LatentUpsampler`
//! (`gen_upsampler_ref.py`, same bf16 bytes, f32 compute). Upsamples a fixed-seed
//! latent x2 spatially and compares the final output. Same weights both sides ->
//! bit-tight; the only drift is the GPU f32 conv accumulation order vs torch over
//! ~20 conv layers (initial + 8 ResBlock convs * 2 + upsampler + final).

#![cfg(feature = "ltx-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::ltx::manifest::{self, role};
use thinfer_models::ltx::upsampler::{IN_CHANNELS, LtxUpsampler, LtxUpsamplerPipelines};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, report};

// Tiny latent (exercises every conv + the pixel-shuffle upsample).
const F: usize = 2;
const H: usize = 4;
const W: usize = 4;

const SLOPE_TOL: f64 = 0.01;
// Same bf16 weights both sides -> bit-tight; band keeps headroom over GPU-f32
// vs torch conv accumulation drift across the ~20-conv stack.
const REL_TOL: f64 = 0.002;

#[tokio::test(flavor = "current_thread")]
async fn upsampler_parity() {
    let _trace = thinfer_core::trace::init_from_env();

    let fr = manifest::MANIFEST
        .get(role::UPSCALER)
        .expect("upscaler role");
    let Some(path) = cache::resolve(fr) else {
        eprintln!("skipped[ltx upsampler_parity]: spatial upscaler not in HF cache");
        return;
    };

    // --- python reference (cached by dims) ---
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ltx_upsampler_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("dims.txt");
    let dims = format!("{F} {H} {W}");
    let cached =
        tmp.join("out.bin").exists() && std::fs::read_to_string(&marker).is_ok_and(|d| d == dims);
    if cached {
        eprintln!(
            "ltx upsampler-parity: reusing cached pyref dumps ({})",
            tmp.display()
        );
    } else {
        for ent in std::fs::read_dir(&tmp).into_iter().flatten().flatten() {
            let _ = std::fs::remove_file(ent.path());
        }
        run_python_ref(&path, &tmp);
        std::fs::write(&marker, &dims).expect("write marker");
    }

    let meta = std::fs::read_to_string(tmp.join("meta.txt")).expect("meta.txt");
    let m: Vec<usize> = meta
        .split_whitespace()
        .map(|s| s.parse().expect("meta int"))
        .collect();
    assert_eq!((m[0], m[1], m[2]), (F, H, W), "pyref dims");

    let latent = read_f32(&tmp.join("latent.bin"));
    assert_eq!(latent.len(), IN_CHANNELS * F * H * W, "latent size");

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

    let pipelines = LtxUpsamplerPipelines::compile(&backend)
        .await
        .expect("compile upsampler pipelines");

    let opener = MmapFileOpener::new(&path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let src = SafetensorsSource::open(opener)
        .await
        .expect("parse upsampler safetensors");
    let residency = WeightResidency::new(
        src,
        ResidencyBudget {
            ram_bytes: 16 << 30,
            vram_bytes: 5 << 30,
        },
    );
    let upsampler = LtxUpsampler::new(pipelines, &residency).expect("build upsampler");
    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    let out = upsampler
        .forward(&backend, &residency, &workspace, &latent, F, H, W)
        .await
        .expect("upsample");

    let exp = read_f32(&tmp.join("out.bin"));
    assert_eq!(
        out.len(),
        exp.len(),
        "out size exp={} got={}",
        exp.len(),
        out.len()
    );

    eprintln!("---- ltx upsampler parity ----");
    let (slope, rel) = report("upsampled", &exp, &out);
    let mut failures = Vec::new();
    if !slope.is_finite() || (slope - 1.0).abs() > SLOPE_TOL {
        failures.push(format!("slope off: {slope:.6}"));
    }
    if rel > REL_TOL {
        failures.push(format!("rel={:.3}% > {:.3}%", rel * 100.0, REL_TOL * 100.0));
    }
    assert!(
        failures.is_empty(),
        "ltx upsampler parity (slope 1+-{SLOPE_TOL}):\n{}",
        failures.join("\n")
    );
}

fn run_python_ref(upsampler: &Path, out_dir: &Path) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let status = Command::new("uv")
        .args([
            "run",
            "--directory",
            py_dir.to_str().unwrap(),
            "--with",
            "einops",
            "python",
            "-m",
            "thinfer_pytorch_ref.ltx.gen_upsampler_ref",
            "--upsampler",
            upsampler.to_str().unwrap(),
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
    assert!(status.success(), "ltx upsampler pyref failed");
}
