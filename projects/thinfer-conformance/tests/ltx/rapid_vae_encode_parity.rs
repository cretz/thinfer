//! LTX-2 (non-.3) video VAE ENCODE parity for ltx2-rapid: the engine
//! `LtxVaeEncoder` with `LtxVaeEncoderConfig::ltx2_rapid()` (bf16 safetensors,
//! F32 acts) vs the upstream `VideoEncoder` (`gen_vae_ref.py --encode`, SAME
//! bf16 bytes, f32 compute). The encoder is the image->latent half needed for
//! native I2V frame-0 conditioning: host patchify + `SpaceToDepthDownsample`
//! (space-to-depth + grouped-mean skip) around causal k3 convs, 32x spatial /
//! 8x temporal. `gen_vae_ref` builds the encoder from the on-disk
//! `__metadata__.config`, so this gate exercises the exact reference schedule.

#![cfg(feature = "ltx-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::ops::ActDtype;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::ltx::manifest::{self, role};
use thinfer_models::ltx::video_vae::{
    LATENT_CHANNELS, LtxVaeEncoder, LtxVaeEncoderConfig, LtxVaePipelines, load_latent_stats,
};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, report};

// Tiny video that still exercises every block: temporal compress (T 9->5->3->2)
// and spatial compress (64px -> latent 2). Latent = [128, 2, 2, 2].
const FPIX: usize = 9;
const HPIX: usize = 64;
const WPIX: usize = 64;

const SLOPE_TOL: f64 = 0.01;
// Same bf16 weights both sides -> bit-tight; only drift is GPU f32 conv
// accumulation order vs torch over the deep (2048-ch) encoder stack.
const REL_TOL: f64 = 0.005;
// f16 acts (production default): track the f32/torch reference within a loose
// band (the encoder is deeper/wider than the decoder, so a touch more headroom).
const F16_LATENT_REL_TOL: f64 = 0.02;

#[tokio::test(flavor = "current_thread")]
async fn rapid_vae_encode_parity() {
    let _trace = thinfer_core::trace::init_from_env();

    let vae_fr = manifest::LTX2_RAPID_MANIFEST
        .get(role::VIDEO_VAE)
        .expect("rapid vae role");
    let Some(vae_path) = cache::resolve(vae_fr) else {
        eprintln!("skipped[ltx rapid_vae_encode_parity]: LTX-2 video VAE not in HF cache");
        return;
    };

    // --- python reference (cached by dims) ---
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ltx_rapid_vae_encode_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("dims.txt");
    let dims = format!("{FPIX} {HPIX} {WPIX}");
    let cached = tmp.join("latent.bin").exists()
        && std::fs::read_to_string(&marker).is_ok_and(|d| d == dims);
    if cached {
        eprintln!(
            "ltx rapid vae-encode-parity: reusing cached pyref dumps ({})",
            tmp.display()
        );
    } else {
        for ent in std::fs::read_dir(&tmp).into_iter().flatten().flatten() {
            let _ = std::fs::remove_file(ent.path());
        }
        run_python_ref(&vae_path, &tmp);
        std::fs::write(&marker, &dims).expect("write marker");
    }

    let meta = std::fs::read_to_string(tmp.join("meta.txt")).expect("meta.txt");
    let m: Vec<usize> = meta
        .split_whitespace()
        .map(|s| s.parse().expect("meta int"))
        .collect();
    let (mf, mh, mw, n_down) = (m[0], m[1], m[2], m[3]);
    let (tlat, hlat, wlat) = (m[4], m[5], m[6]);
    assert_eq!((mf, mh, mw), (FPIX, HPIX, WPIX), "pyref dims");

    let frame = read_f32(&tmp.join("frame.bin"));
    assert_eq!(frame.len(), 3 * FPIX * HPIX * WPIX, "frame size");

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
            vram_bytes: 6 << 30,
        },
    );
    // The engine loads the SAME baked stats the pyref normalizes with.
    let (mean, std) = load_latent_stats(&residency, &backend)
        .await
        .expect("load latent stats");
    let exp_mean = read_f32(&tmp.join("mean.bin"));
    let exp_std = read_f32(&tmp.join("std.bin"));
    let (ms, mr) = report("mean", &exp_mean, &mean);
    let (ss, sr) = report("std", &exp_std, &std);
    assert!(
        mr < 1e-4 && sr < 1e-4 && ms.is_finite() && ss.is_finite(),
        "latent stats mismatch"
    );

    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    // Parity is bit-exact only at f32; pin the reference encode to f32 acts.
    let pipelines = LtxVaePipelines::compile_with(&backend, ActDtype::F32)
        .await
        .expect("compile vae pipelines");
    let encoder = LtxVaeEncoder::new(
        pipelines,
        &residency,
        mean.clone(),
        std.clone(),
        LtxVaeEncoderConfig::ltx2_rapid(),
    )
    .expect("build rapid encoder");

    let mut down_blocks: Vec<Vec<f32>> = Vec::new();
    let latent = encoder
        .encode_with_taps(
            &backend,
            &residency,
            &workspace,
            &frame,
            FPIX,
            HPIX,
            WPIX,
            Some(&mut down_blocks),
        )
        .await
        .expect("vae encode");

    eprintln!("---- ltx rapid vae encode parity ----");
    let mut failures = Vec::new();
    let mut check = |label: &str, exp: &[f32], got: &[f32]| {
        assert_eq!(
            exp.len(),
            got.len(),
            "[{label}] length exp={} got={}",
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

    assert_eq!(down_blocks.len(), n_down, "down_block count");
    for (i, got) in down_blocks.iter().enumerate() {
        let exp = read_f32(&tmp.join(format!("down_{i:02}.bin")));
        check(&format!("down_{i:02}"), &exp, got);
    }
    let exp_latent = read_f32(&tmp.join("latent.bin"));
    assert_eq!(
        exp_latent.len(),
        LATENT_CHANNELS * tlat * hlat * wlat,
        "latent size"
    );
    check("latent", &exp_latent, &latent);

    // f16 acts (production default): encode again, assert the latent tracks the
    // f32/torch reference within a loose band.
    let f16_pipelines = LtxVaePipelines::compile_with(&backend, ActDtype::F16)
        .await
        .expect("compile f16 vae pipelines");
    let f16_encoder = LtxVaeEncoder::new(
        f16_pipelines,
        &residency,
        mean,
        std,
        LtxVaeEncoderConfig::ltx2_rapid(),
    )
    .expect("build f16 rapid encoder");
    let f16_latent = f16_encoder
        .encode(&backend, &residency, &workspace, &frame, FPIX, HPIX, WPIX)
        .await
        .expect("f16 vae encode");
    let (f16_slope, f16_rel) = report("latent_f16", &exp_latent, &f16_latent);
    if !f16_slope.is_finite() || (f16_slope - 1.0).abs() > SLOPE_TOL {
        failures.push(format!("latent_f16 slope off: {f16_slope:.6}"));
    }
    if f16_rel > F16_LATENT_REL_TOL {
        failures.push(format!(
            "latent_f16 rel={:.3}% > {:.3}%",
            f16_rel * 100.0,
            F16_LATENT_REL_TOL * 100.0
        ));
    }

    assert!(
        failures.is_empty(),
        "ltx rapid vae encode parity (slope 1+-{SLOPE_TOL}):\n{}",
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
            "python",
            "-m",
            "thinfer_pytorch_ref.ltx.gen_vae_ref",
            "--encode",
            "--vae",
            vae.to_str().unwrap(),
            "--out",
            out_dir.to_str().unwrap(),
            "--frames",
            &FPIX.to_string(),
            "--height",
            &HPIX.to_string(),
            "--width",
            &WPIX.to_string(),
        ])
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "ltx rapid vae encode pyref failed");
}
