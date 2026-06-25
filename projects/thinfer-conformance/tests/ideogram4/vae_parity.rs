//! Ideogram-4 VAE-decode parity: the shared Flux2 KL-VAE decoder (FLUX.2-VAE
//! weights, f16 acts / bf16 weights) vs an fp32 reference decoder built from
//! the SAME safetensors (`gen_vae_ref.py`), fed an identical packed latent.
//!
//! The pyref dumps a deterministic `[num_image, 128]` latent; both sides run
//! the same host denorm + unpatch (`ideogram4::vae::unpatch_denorm`) to a
//! spatial `[32, h_in, w_in]` tensor, then decode. The band is the engine's
//! f16/bf16 rounding vs the fp32 truth (convs accumulate f32 in-kernel).

#![cfg(feature = "ideogram4-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::ideogram4::manifest::{self, role};
use thinfer_models::ideogram4::vae::{
    IDEOGRAM4_KL_VAE, VaeDecoder, VaeDecoderPipelines, VaeTileConfig, register_vae_decoder_handles,
    unpatch_denorm,
};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, report};

// Tiny parity dims: 64x64 image -> 4x4 grid (16 image tokens), latent 8x8.
const WIDTH: usize = 64;
const HEIGHT: usize = 64;

// f16-act / bf16-weight engine vs fp32 reference. Re-measure + re-document on
// a config change; do not loosen to go green.
const SLOPE_TOL: f64 = 0.05;
const REL_TOL: f64 = 0.08;

#[tokio::test(flavor = "current_thread")]
async fn vae_parity() {
    let _trace = thinfer_core::trace::init_from_env();

    let fr = manifest::MANIFEST.get(role::VAE).expect("vae role");
    let Some(vae_path) = cache::resolve(fr) else {
        eprintln!(
            "skipped[vae_parity]: {}/{} not in HF cache",
            fr.repo, fr.path
        );
        return;
    };

    // --- python reference (deterministic; dumps latent + decoded image) ---
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ideo_vae_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("config.txt");
    let cfg_tag = format!("{WIDTH}x{HEIGHT}");
    let cached = tmp.join("decoded.bin").exists()
        && std::fs::read_to_string(&marker).is_ok_and(|c| c == cfg_tag);
    if cached {
        eprintln!("vae-parity: reusing cached pyref dumps ({})", tmp.display());
    } else {
        for ent in std::fs::read_dir(&tmp).into_iter().flatten().flatten() {
            let _ = std::fs::remove_file(ent.path());
        }
        run_python_ref(&vae_path, &tmp);
        std::fs::write(&marker, &cfg_tag).expect("write marker");
    }

    let meta = std::fs::read_to_string(tmp.join("meta.txt")).expect("meta.txt");
    let m: Vec<usize> = meta
        .split_whitespace()
        .take(3)
        .map(|s| s.parse().expect("meta int"))
        .collect();
    let (num_image, grid_h, grid_w) = (m[0], m[1], m[2]);
    let h_in = grid_h * 2;
    let w_in = grid_w * 2;
    eprintln!("vae-parity: num_image={num_image} grid={grid_h}x{grid_w} latent={h_in}x{w_in}");

    let latent_tokens = read_f32(&tmp.join("latent_tokens.bin"));
    assert_eq!(latent_tokens.len(), num_image * 128);
    let decoded_ref = read_f32(&tmp.join("decoded.bin"));
    assert_eq!(decoded_ref.len(), 3 * (h_in * 8) * (w_in * 8));

    // --- engine: open FLUX.2-VAE safetensors, decode the same latent ---
    let opener = MmapFileOpener::new(&vae_path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", vae_path.display()));
    let vae_src = SafetensorsSource::open(opener)
        .await
        .expect("parse FLUX.2-VAE safetensors");
    let budget = ResidencyBudget {
        ram_bytes: 12 << 30,
        vram_bytes: 6 << 30,
    };
    let residency = WeightResidency::new(vae_src, budget);
    let handles = register_vae_decoder_handles(&residency).expect("register vae handles");

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

    let pipelines = VaeDecoderPipelines::compile(&backend)
        .await
        .expect("compile vae pipelines");
    let vae = VaeDecoder {
        pipelines,
        handles,
        tile_cfg: VaeTileConfig::default(),
        arch: IDEOGRAM4_KL_VAE,
    };

    // Host denorm + unpatch -> spatial [32, h_in, w_in].
    let spatial = unpatch_denorm(&latent_tokens, grid_h, grid_w);
    assert_eq!(spatial.len(), 32 * h_in * w_in);

    let mut workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));
    let rgb = vae
        .decode(&backend, &residency, &mut workspace, &spatial, h_in, w_in)
        .await
        .expect("engine vae decode");
    assert_eq!(rgb.len(), decoded_ref.len());

    eprintln!("---- ideogram vae parity ----");
    let (slope, rel) = report("decoded", &decoded_ref, &rgb);

    let mut failures = Vec::new();
    if (slope - 1.0).abs() > SLOPE_TOL || rel > REL_TOL {
        failures.push(format!(
            "decoded diverges: slope={slope:.6} rel={:.3}% (tol slope 1+-{SLOPE_TOL}, rel {REL_TOL})",
            rel * 100.0
        ));
    }
    assert!(failures.is_empty(), "vae parity:\n{}", failures.join("\n"));
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
            "thinfer_pytorch_ref.ideogram4.gen_vae_ref",
            "--vae",
            vae.to_str().unwrap(),
            "--out",
            out_dir.to_str().unwrap(),
            "--width",
            &WIDTH.to_string(),
            "--height",
            &HEIGHT.to_string(),
        ])
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "ideogram vae pyref failed");
}
