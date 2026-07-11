//! HunyuanVideo 1.5 T2V end-to-end parity: pinned noise + fixed text-hidden ->
//! engine `HunyuanDit::denoise` (4-step flow-match Euler, lightx2v shift 9) ->
//! `HunyuanVaeDecoder::decode` vs the upstream reference (`gen_e2e_ref.py`, which
//! reuses the per-component reference forwards). Validates the integration not
//! covered by the component gates: noise init, the 65-channel DiT packing, the
//! Euler stepping, and the latent -> VAE scaling. The band absorbs the
//! fp16->bf16 weight narrowing compounded over 4 steps x 54 blocks + the VAE.

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
use thinfer_models::hunyuan::dit::{HunyuanDit, HunyuanDitPipelines};
use thinfer_models::hunyuan::refiner::{HunyuanRefiner, HunyuanRefinerPipelines};
use thinfer_models::hunyuan::scheduler::FlowMatchSchedule;
use thinfer_models::hunyuan::vae::{HunyuanVaeDecoder, HunyuanVaePipelines};
use thinfer_native::MmapFileOpener;

use crate::parity_util::{read_f32, report, resolve_hf};

const SEQ: usize = 16;
const T: usize = 2;
const H: usize = 4;
const W: usize = 4;

const SLOPE_TOL: f64 = 0.05;
const REL_TOL: f64 = 0.06;

#[tokio::test(flavor = "current_thread")]
async fn t2v_e2e() {
    let _trace = thinfer_core::trace::init_from_env();

    let Some(dit_path) = resolve_hf(
        "THINFER_HUNYUAN_DIT",
        "models--lightx2v--Hy1.5-Distill-Models",
        "hy1.5_t2v_480p_lightx2v_4step.safetensors",
    ) else {
        eprintln!("skipped[hunyuan t2v_e2e]: lightx2v DiT not in HF cache");
        return;
    };
    let Some(vae_path) = resolve_hf(
        "THINFER_HUNYUAN_VAE",
        "models--Comfy-Org--HunyuanVideo_1.5_repackaged",
        "split_files/vae/hunyuanvideo15_vae_fp16.safetensors",
    ) else {
        eprintln!("skipped[hunyuan t2v_e2e]: Comfy VAE not in HF cache");
        return;
    };

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("hunyuan_t2v_e2e");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("dims.txt");
    let dims = format!("{SEQ} {T} {H} {W}");
    let cached =
        tmp.join("video.bin").exists() && std::fs::read_to_string(&marker).is_ok_and(|d| d == dims);
    if cached {
        eprintln!("hunyuan t2v-e2e: reusing cached pyref dumps");
    } else {
        for ent in std::fs::read_dir(&tmp).into_iter().flatten().flatten() {
            let _ = std::fs::remove_file(ent.path());
        }
        run_python_ref(&dit_path, &vae_path, &tmp);
        std::fs::write(&marker, &dims).expect("write marker");
    }

    let text = read_f32(&tmp.join("text_in.bin"));
    let latent_init = read_f32(&tmp.join("latent_init.bin"));
    assert_eq!(text.len(), SEQ * 3584, "text size");
    assert_eq!(latent_init.len(), 32 * T * H * W, "latent init size");

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

    // i8 DP4A (self-attn q/k/v + ffn-up) is the SHIPPING default; `THINFER_HY_I8=0`
    // selects the pure bf16 reference. The band widens under i8: at these tiny
    // near-zero-mean dims (n_img=32) i8 quant amplifies relative error (bf16
    // latent ~3.3%, i8 ~6.8%); the per-stage dit_parity taps stay tight and the
    // real i8 quality signal is the serve eyeball at 480p.
    let i8 = std::env::var("THINFER_HY_I8")
        .map(|v| v != "0")
        .unwrap_or(true);

    // --- DiT denoise (scoped: free all DiT VRAM before the VAE decode; the 8GB
    // card can't hold both the DiT and VAE residencies at once) ---
    let latent = {
        let dit_src = SafetensorsSource::open(
            MmapFileOpener::new(&dit_path)
                .await
                .unwrap_or_else(|e| panic!("open {}: {e}", dit_path.display())),
        )
        .await
        .expect("parse DiT safetensors");
        let dit_res = WeightResidency::new(
            dit_src,
            ResidencyBudget {
                ram_bytes: 24 << 30,
                vram_bytes: 5 << 30,
            },
        );
        let refiner = HunyuanRefiner::new(
            HunyuanRefinerPipelines::compile_with(&backend, ActDtype::F32)
                .await
                .expect("refiner pl"),
            &dit_res,
        )
        .expect("refiner");
        let dit = HunyuanDit::new(
            HunyuanDitPipelines::compile_with(&backend, ActDtype::Bf16, i8)
                .await
                .expect("dit pl"),
            refiner,
            &dit_res,
            i8,
        )
        .expect("dit");
        let dit_ws = Workspace::new(Arc::clone(&backend), Arc::clone(dit_res.arbiter()));
        let schedule = FlowMatchSchedule::lightx2v_t2v_480p();
        let latent = dit
            .denoise(
                &backend,
                &dit_res,
                &dit_ws,
                &text,
                SEQ,
                &latent_init,
                (T, H, W),
                &schedule,
                0,    // full attention (parity)
                None, // no progress reporting in tests
                None, // no cancellation in tests
            )
            .await
            .expect("denoise");
        drop(dit_ws);
        dit_res.evict_all_and_free(&*backend); // return all DiT VRAM before the VAE phase
        latent
    };

    // --- VAE decode ---
    let vae_src = SafetensorsSource::open(
        MmapFileOpener::new(&vae_path)
            .await
            .unwrap_or_else(|e| panic!("open {}: {e}", vae_path.display())),
    )
    .await
    .expect("parse VAE safetensors");
    let vae_res = WeightResidency::new(
        vae_src,
        ResidencyBudget {
            ram_bytes: 24 << 30,
            vram_bytes: 5 << 30,
        },
    );
    let vae = HunyuanVaeDecoder::new(
        HunyuanVaePipelines::compile_with(&backend, ActDtype::F32)
            .await
            .expect("vae pl"),
        &vae_res,
    )
    .expect("vae");
    let vae_ws = Workspace::new(Arc::clone(&backend), Arc::clone(vae_res.arbiter()));
    let video = vae
        .decode_with_taps(&backend, &vae_res, &vae_ws, &latent, T, H, W, None)
        .await
        .expect("decode");

    eprintln!("---- hunyuan t2v e2e ----");
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
        let rel_tol = if i8 { 0.10 } else { REL_TOL };
        if rel > rel_tol {
            failures.push(format!(
                "{label} rel={:.3}% > {:.3}%",
                rel * 100.0,
                rel_tol * 100.0
            ));
        }
    };
    check("latent", &read_f32(&tmp.join("latent.bin")), &latent);
    check("video", &read_f32(&tmp.join("video.bin")), &video);

    assert!(
        failures.is_empty(),
        "hunyuan t2v e2e:\n{}",
        failures.join("\n")
    );
}

fn run_python_ref(dit: &Path, vae: &Path, out_dir: &Path) {
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
            "thinfer_pytorch_ref.hunyuan.gen_e2e_ref",
            "--dit",
            dit.to_str().unwrap(),
            "--vae",
            vae.to_str().unwrap(),
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
    assert!(status.success(), "hunyuan e2e pyref failed");
}
