//! MiniMax-H3 video VAE ENCODER parity, single frame (keyframes / image
//! references): the engine's `vae_encoder` vs FastVideo's torch
//! `_encode_clip` (tiling off) over the SAME Comfy-Org fp16 file
//! (`minimax/gen_vae_enc_ref.py`, bf16 on CPU), stage by stage (`conv_in`, each
//! level, the moments) so a mismatch localizes, then the normalized latent.

#![cfg(feature = "minimax-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::weight::{WeightId, WeightReader, WeightSource, decode_into};
use thinfer_core::workspace::Workspace;
use thinfer_models::minimax::manifest::{self, role};
use thinfer_models::minimax::vae_encoder::{
    EncodeOptions, EncodeTaps, EncoderPipelines, EncoderSource, H3VaeEncoder, Posterior,
};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, report};

/// bf16 torch on CPU vs the engine's f32-act / bf16-weight encoder.
const SLOPE_TOL: f64 = 0.03;
const REL_TOL: f64 = 0.05;

#[tokio::test(flavor = "current_thread")]
async fn vae_encoder_parity() {
    let _trace = thinfer_core::trace::init_from_env();
    let fr = manifest::MANIFEST
        .get(role::VIDEO_VAE)
        .expect("video vae role");
    let Some(path) = cache::resolve(fr) else {
        eprintln!("skipped[minimax vae_encoder_parity]: video VAE not in HF cache");
        return;
    };
    let dir = ensure_pyref(&path);
    let meta: Vec<usize> = std::fs::read_to_string(dir.join("meta.txt"))
        .expect("meta.txt")
        .split_whitespace()
        .map(|t| t.parse().expect("meta int"))
        .collect();
    let [hh, ww, h, w] = meta[..] else {
        panic!("meta.txt needs 4 fields")
    };

    let backend = Arc::new(
        WgpuBackend::new_with_config(WgpuConfig {
            power_preference: PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("wgpu backend"),
    );
    let opener = MmapFileOpener::new(&path).await.expect("open video vae");
    let raw = SafetensorsSource::open(opener)
        .await
        .expect("parse video vae");
    let mean = read_vec(&raw, "latents_mean", 24).await;
    let std = read_vec(&raw, "latents_std", 24).await;
    let residency = WeightResidency::new(
        EncoderSource::new(raw),
        ResidencyBudget {
            ram_bytes: 4 << 30,
            vram_bytes: 4 << 30,
        },
    );
    let enc = H3VaeEncoder::new(&residency).expect("register encoder");
    let pl = EncoderPipelines::compile(&backend)
        .await
        .expect("pipelines");
    let ws = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    let input = read_f32(&dir.join("input.bin"));
    assert_eq!(input.len(), 3 * hh * ww);
    let mut taps = EncodeTaps::default();
    let out = enc
        .encode_frame(
            &backend,
            &residency,
            &ws,
            &pl,
            &input,
            hh,
            ww,
            (&mean, &std),
            // The pyref runs `_encode_clip` with tiling off and dumps the mean.
            EncodeOptions {
                tiled: false,
                posterior: Posterior::Mean,
            },
            Some(&mut taps),
        )
        .await
        .expect("encode");
    assert_eq!((out.height, out.width), (h, w));

    let mut worst = Vec::new();
    let mut check = |label: &str, exp: &[f32], got: &[f32]| {
        assert_eq!(exp.len(), got.len(), "{label}: length");
        let (slope, rel) = report(label, exp, got);
        worst.push((label.to_string(), slope, rel));
    };
    check(
        "conv_in",
        &read_f32(&dir.join("conv_in.bin")),
        &taps.conv_in,
    );
    for (i, lv) in taps.levels.iter().enumerate() {
        check(
            &format!("down{i}"),
            &read_f32(&dir.join(format!("down{i}.bin"))),
            lv,
        );
    }
    check(
        "moments",
        &read_f32(&dir.join("moments.bin")),
        &taps.moments,
    );
    check("latents", &read_f32(&dir.join("latents.bin")), &out.latent);
    for (label, slope, rel) in &worst {
        assert!(
            (slope - 1.0).abs() < SLOPE_TOL && *rel < REL_TOL,
            "{label}: slope {slope:.4} rel {rel:.4}"
        );
    }
}

async fn read_vec<S: WeightSource>(src: &S, name: &str, n: usize) -> Vec<f32> {
    let id = WeightId(name.into());
    let enc = src
        .catalog()
        .get(&id)
        .and_then(|e| e.encoding)
        .expect("encoding");
    let mut r = src.open(&id).await.expect("open");
    let mut bytes = vec![0u8; r.len() as usize];
    r.read_at(0, &mut bytes).await.expect("read");
    let mut out = vec![0u8; n * 4];
    decode_into(enc, &bytes, &mut out).expect("decode");
    out.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn ensure_pyref(vae: &Path) -> PathBuf {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("minimax_vae_enc_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("done.txt");
    if marker.exists() {
        eprintln!(
            "minimax vae-encoder parity: reusing cached pyref dumps ({})",
            tmp.display()
        );
        return tmp;
    }
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let status = Command::new("uv")
        .args([
            "run",
            "--directory",
            py_dir.to_str().unwrap(),
            "python",
            "-m",
        ])
        .arg("thinfer_pytorch_ref.minimax.gen_vae_enc_ref")
        .arg("--vae")
        .arg(vae)
        .arg("--out")
        .arg(&tmp)
        .env("OMP_NUM_THREADS", "4")
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "minimax vae encoder pyref failed");
    std::fs::write(&marker, "ok").expect("write marker");
    tmp
}
