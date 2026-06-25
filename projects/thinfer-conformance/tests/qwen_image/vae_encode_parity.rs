//! Qwen-Image VAE-encode parity (the edit path's reference-image latent
//! channel): the shared Wan-family 3D-causal KL encoder (`wan/vae.rs`), loaded
//! from the full `Qwen/Qwen-Image` `vae/` safetensors (native diffusers keys),
//! vs an fp32 reference encoder built from the SAME safetensors
//! (`gen_vae_encode_ref.py`), fed an identical `[3, H, W]` image in `[-1, 1]`.
//!
//! Compares the raw encoder distribution params `[2*z_dim, h, w]` (mean ++
//! logvar, pre-normalization); the edit pipeline consumes the mode (= mean,
//! channels 0..z_dim). Band is the engine's bf16-weight / f16-act rounding vs
//! the fp32 truth (convs accumulate f32 in-kernel).
//!
//! Two cases share one driver:
//!   - `vae_encode_parity` (64x64): one full tile == the untiled encode path,
//!     so this also guards the single-tile bit-identical fast path.
//!   - `vae_encode_parity_large` (768x512 under a 6 GiB budget): forces the
//!     budget-derived SPATIAL TILING (the untiled encode OOMs at this res on an
//!     8 GiB card); proves the tiled encode still tracks the fp32 reference.
#![cfg(feature = "qwen-image-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::qwen_image::manifest::{self, role};
use thinfer_models::qwen_image::vae::qwen_image_vae;
use thinfer_models::wan::vae::{
    VaeEncoderWeights, WanVaeEncoder, WanVaePipelines, register_encoder,
};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, report};

const SLOPE_TOL: f64 = 0.05;
const REL_TOL: f64 = 0.08;

/// 64x64: a single full tile (`plan_tiles` returns one tile), so the engine
/// path is byte-identical to the pre-tiling untiled encode.
#[tokio::test(flavor = "current_thread")]
async fn vae_encode_parity() {
    run_case("small", 64, 64, 6 << 30).await;
}

/// 1216x832 (~1 MP, the diffusers edit ref res) under a thin 6 GiB VRAM
/// budget: the untiled encode (full-res 96ch activations over ~1 MP) OOMs on an
/// 8 GiB card; the budget-derived spatial tiling splits it into a multi-tile
/// grid that fits and stays within the reference band.
#[tokio::test(flavor = "current_thread")]
async fn vae_encode_parity_large() {
    run_case("large", 1216, 832, 6 << 30).await;
}

async fn run_case(tag: &str, width: usize, height: usize, vram_bytes: u64) {
    let _trace = thinfer_core::trace::init_from_env();

    let fr = manifest::MANIFEST.get(role::VAE).expect("vae role");
    let Some(vae_path) = cache::resolve(fr) else {
        eprintln!(
            "skipped[vae_encode_parity:{tag}]: {}/{} not in HF cache",
            fr.repo, fr.path
        );
        return;
    };

    // --- python reference (deterministic; dumps image + encoder moments) ---
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("qwen_vae_encode_{tag}"));
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("config.txt");
    let cfg_tag = format!("{width}x{height}");
    let cached = tmp.join("moments.bin").exists()
        && std::fs::read_to_string(&marker).is_ok_and(|c| c == cfg_tag);
    if cached {
        eprintln!(
            "vae-encode-parity[{tag}]: reusing cached pyref ({})",
            tmp.display()
        );
    } else {
        for ent in std::fs::read_dir(&tmp).into_iter().flatten().flatten() {
            let _ = std::fs::remove_file(ent.path());
        }
        run_python_ref(&vae_path, &tmp, width, height);
        std::fs::write(&marker, &cfg_tag).expect("write marker");
    }

    let meta = std::fs::read_to_string(tmp.join("meta.txt")).expect("meta.txt");
    let m: Vec<usize> = meta
        .split_whitespace()
        .take(3)
        .map(|s| s.parse().expect("meta int"))
        .collect();
    let (z_dim, h_out, w_out) = (m[0], m[1], m[2]);
    eprintln!("vae-encode-parity[{tag}]: z_dim={z_dim} latent={h_out}x{w_out}");

    let image = read_f32(&tmp.join("image.bin"));
    assert_eq!(image.len(), 3 * height * width);
    let moments_ref = read_f32(&tmp.join("moments.bin"));
    assert_eq!(moments_ref.len(), 2 * z_dim * h_out * w_out);

    // --- engine: open the full VAE safetensors, encode the same image ---
    let opener = MmapFileOpener::new(&vae_path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", vae_path.display()));
    let vae_src = SafetensorsSource::open(opener)
        .await
        .expect("parse Qwen-Image VAE safetensors");
    let budget = ResidencyBudget {
        ram_bytes: 12 << 30,
        vram_bytes,
    };
    let residency = WeightResidency::new(vae_src, budget);

    let cfg = qwen_image_vae();
    let before = residency.total_registered_bytes();
    let handles =
        register_encoder(&residency, &VaeEncoderWeights::new(&cfg)).expect("register vae encoder");
    let weight_footprint = residency.total_registered_bytes() - before;

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

    let pipelines = WanVaePipelines::compile(&backend)
        .await
        .expect("compile vae pipelines");
    let encoder = WanVaeEncoder {
        pipelines,
        handles,
        cfg,
        weight_footprint,
    };

    let mut workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));
    // f=1 (image): CTHW input [3, 1, H, W]; output [2*z_dim, 1, h_out, w_out].
    // THINFER_VAE_MEM=1 logs the budget-derived tile/halo/grid this run used.
    let moments = encoder
        .encode(
            &backend,
            &residency,
            &mut workspace,
            &image,
            1,
            height,
            width,
        )
        .await
        .expect("engine vae encode");
    assert_eq!(moments.len(), moments_ref.len());

    eprintln!("---- qwen-image vae encode parity [{tag}] ----");
    let (slope, rel) = report("moments", &moments_ref, &moments);
    // Mode (mean) is what the edit path consumes; report it separately.
    let zc = z_dim * h_out * w_out;
    let (slope_m, rel_m) = report("mode", &moments_ref[..zc], &moments[..zc]);

    let mut failures = Vec::new();
    for (name, slope, rel) in [("moments", slope, rel), ("mode", slope_m, rel_m)] {
        if (slope - 1.0).abs() > SLOPE_TOL || rel > REL_TOL {
            failures.push(format!(
                "{name} diverges: slope={slope:.6} rel={:.3}% (tol slope 1+-{SLOPE_TOL}, rel {REL_TOL})",
                rel * 100.0
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "vae encode parity [{tag}]:\n{}",
        failures.join("\n")
    );
}

fn run_python_ref(vae: &Path, out_dir: &Path, width: usize, height: usize) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let status = Command::new("uv")
        .args([
            "run",
            "--directory",
            py_dir.to_str().unwrap(),
            "--with",
            "safetensors",
            "python",
            "-m",
            "thinfer_pytorch_ref.qwen_image.gen_vae_encode_ref",
            "--vae",
            vae.to_str().unwrap(),
            "--out",
            out_dir.to_str().unwrap(),
            "--width",
            &width.to_string(),
            "--height",
            &height.to_string(),
        ])
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "qwen-image vae encode pyref failed");
}
