//! LTX-2 (non-.3) video VAE decode parity for ltx2-rapid: the engine
//! `LtxVaeDecoder` with `LtxVaeConfig::ltx2_rapid()` (bf16 safetensors, F32 acts)
//! vs the upstream `VideoDecoder` (`gen_vae_ref.py`, SAME bf16 bytes, f32
//! compute). This decoder differs from the distilled one in two ways the config
//! carries: an all-`compress_all` block schedule AND `residual: True` on every
//! `DepthToSpaceUpsample` (the up-shortcut that adds the input's own
//! shuffle-and-tile). `gen_vae_ref` builds the decoder from the on-disk
//! `__metadata__.config`, so it exercises the residual with no python change;
//! this test is the gate that the engine's residual path matches it.

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
    LATENT_CHANNELS, LtxVaeConfig, LtxVaeDecoder, LtxVaePipelines, LtxVaeTaps,
};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, report};

// Tiny latent (exercises every block incl. all 3 residual temporal+spatial
// upsamplers).
const F: usize = 2;
const H: usize = 4;
const W: usize = 4;

const SLOPE_TOL: f64 = 0.01;
// Same bf16 weights both sides -> bit-tight. The residual add is exact; the only
// drift is GPU f32 conv accumulation order vs torch over the deeper (4x res_x(5))
// rapid stack. Band keeps generous headroom.
const REL_TOL: f64 = 0.005;
// f16 acts (the production default) track the f32 reference closely but are not
// bit-exact; guards that the fast default does not degrade the decoded video.
const F16_VIDEO_REL_TOL: f64 = 0.01;

#[tokio::test(flavor = "current_thread")]
async fn rapid_vae_parity() {
    let _trace = thinfer_core::trace::init_from_env();

    let vae_fr = manifest::LTX2_RAPID_MANIFEST
        .get(role::VIDEO_VAE)
        .expect("rapid vae role");
    let Some(vae_path) = cache::resolve(vae_fr) else {
        eprintln!("skipped[ltx rapid_vae_parity]: LTX-2 video VAE not in HF cache");
        return;
    };

    // --- python reference (cached by dims) ---
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ltx_rapid_vae_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("dims.txt");
    let dims = format!("{F} {H} {W}");
    let cached =
        tmp.join("video.bin").exists() && std::fs::read_to_string(&marker).is_ok_and(|d| d == dims);
    if cached {
        eprintln!(
            "ltx rapid vae-parity: reusing cached pyref dumps ({})",
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
    let (mf, mh, mw, n_up) = (m[0], m[1], m[2], m[3]);
    assert_eq!((mf, mh, mw), (F, H, W), "pyref dims");

    let latent = read_f32(&tmp.join("latent.bin"));
    assert_eq!(latent.len(), LATENT_CHANNELS * F * H * W, "latent size");
    let mean = read_f32(&tmp.join("mean.bin"));
    let std = read_f32(&tmp.join("std.bin"));

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

    // Parity is bit-exact only at f32; pin the reference decode to f32 acts.
    let pipelines = LtxVaePipelines::compile_with(&backend, ActDtype::F32)
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
    let decoder = LtxVaeDecoder::new_with_config(
        pipelines,
        &residency,
        mean.clone(),
        std.clone(),
        LtxVaeConfig::ltx2_rapid(),
    )
    .expect("build rapid decoder");
    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    let mut conv_in = Vec::new();
    let mut up_blocks: Vec<Vec<f32>> = Vec::new();
    let mut conv_out = Vec::new();
    let mut taps = LtxVaeTaps {
        conv_in: Some(&mut conv_in),
        up_blocks: Some(&mut up_blocks),
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

    eprintln!("---- ltx rapid vae parity ----");
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

    check("conv_in", &read_f32(&tmp.join("conv_in.bin")), &conv_in);
    assert_eq!(up_blocks.len(), n_up, "up_block count");
    for (i, got) in up_blocks.iter().enumerate() {
        let exp = read_f32(&tmp.join(format!("up_{i:02}.bin")));
        check(&format!("up_{i:02}"), &exp, got);
    }
    check("conv_out", &read_f32(&tmp.join("conv_out.bin")), &conv_out);
    let exp_video = read_f32(&tmp.join("video.bin"));
    check("video", &exp_video, &video);

    // f16 acts (the production default): decode again and assert the video tracks
    // the f32/torch reference within a loose band (visually identical).
    let f16_pipelines = LtxVaePipelines::compile_with(&backend, ActDtype::F16)
        .await
        .expect("compile f16 vae pipelines");
    let f16_decoder = LtxVaeDecoder::new_with_config(
        f16_pipelines,
        &residency,
        mean,
        std,
        LtxVaeConfig::ltx2_rapid(),
    )
    .expect("build f16 rapid decoder");
    let f16_video = f16_decoder
        .decode_with_taps(&backend, &residency, &workspace, &latent, F, H, W, None)
        .await
        .expect("f16 vae decode");
    let (f16_slope, f16_rel) = report("video_f16", &exp_video, &f16_video);
    if !f16_slope.is_finite() || (f16_slope - 1.0).abs() > SLOPE_TOL {
        failures.push(format!("video_f16 slope off: {f16_slope:.6}"));
    }
    if f16_rel > F16_VIDEO_REL_TOL {
        failures.push(format!(
            "video_f16 rel={:.3}% > {:.3}%",
            f16_rel * 100.0,
            F16_VIDEO_REL_TOL * 100.0
        ));
    }

    assert!(
        failures.is_empty(),
        "ltx rapid vae parity (slope 1+-{SLOPE_TOL}):\n{}",
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
    assert!(status.success(), "ltx rapid vae pyref failed");
}
