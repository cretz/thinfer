//! HunyuanVideo 1.5 PERF turnaround harness (NOT a parity gate). Drives the real
//! `HunyuanDit::denoise` + `HunyuanVaeDecoder::decode` on the actual checkpoint
//! weights with RANDOM inputs (no encoder, no pyref) at an env-configurable latent
//! grid, so a perf change can be measured in ~1-2 min instead of the ~48 min full
//! 480p run. The kernels exercised (bf16 / f16-SDPA / i8 matmul / tiling / causal
//! VAE) are bit-for-bit the production ones; only the dims + step count shrink.
//!
//! Reports wall time per phase; pair with `THINFER_TRACE=1` for the per-pipeline
//! `gpu_ms` rollup. No correctness assertion beyond a finite-output check
//! (parity is owned by `e2e` / the component gates).
//!
//! Env knobs (latent grid, NOT pixels):
//!   THINFER_HY_GT / GH / GW  latent grid extent  (default 5 / 16 / 16)
//!   THINFER_HY_SEQ           refined-text token count (default 32)
//!   THINFER_HY_STEPS         denoise steps (default 2; product = 4)
//!   THINFER_HY_VAE=1         also run the VAE decode (off by default; DiT-only)
//!   THINFER_HY_VRAM_GB       residency budget GiB (default 5)
//! Plus the production knobs (THINFER_DIT_TILE_ROWS, THINFER_HY_I8, ...).

#![cfg(feature = "hunyuan-e2e")]

use std::sync::Arc;
use std::time::Instant;

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

use crate::parity_util::resolve_hf;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

/// Deterministic pseudo-random f32 in [-1, 1] from a counter (splitmix64). Keeps
/// inputs reproducible run-to-run so wall-time comparisons aren't noise from the
/// data distribution.
fn fill_rand(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

#[tokio::test(flavor = "current_thread")]
async fn t2v_perf() {
    let trace = thinfer_core::trace::init_from_env();

    let Some(dit_path) = resolve_hf(
        "THINFER_HUNYUAN_DIT",
        "models--lightx2v--Hy1.5-Distill-Models",
        "hy1.5_t2v_480p_lightx2v_4step.safetensors",
    ) else {
        eprintln!("skipped[hunyuan t2v_perf]: lightx2v DiT not in HF cache");
        return;
    };

    let gt = env_usize("THINFER_HY_GT", 5);
    let gh = env_usize("THINFER_HY_GH", 16);
    let gw = env_usize("THINFER_HY_GW", 16);
    let seq = env_usize("THINFER_HY_SEQ", 32);
    let steps = env_usize("THINFER_HY_STEPS", 2);
    let run_vae = env_usize("THINFER_HY_VAE", 0) != 0;
    let vram_gb = env_usize("THINFER_HY_VRAM_GB", 5) as u64;
    let i8 = env_usize("THINFER_HY_I8", 1) != 0;
    // Temporal attention window radius in latent frames (0 = full attention).
    let window = env_usize("THINFER_HY_WINDOW", 0) as u32;
    let n_img = gt * gh * gw;

    eprintln!(
        "---- hunyuan t2v PERF: grid {gt}x{gh}x{gw} (n_img={n_img}), seq={seq}, \
         steps={steps}, vae={run_vae}, i8={i8}, window={window}, vram={vram_gb}G ----"
    );

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

    let text = fill_rand(seq * 3584, 1);
    let latent_init = fill_rand(32 * n_img, 2);
    let budget = ResidencyBudget {
        ram_bytes: 24 << 30,
        vram_bytes: vram_gb << 30,
    };
    // Phase-aware DiT weight budget, mirroring the production driver
    // (`thinfer-app/src/hunyuan.rs` Phase B): the activation-tiled forward needs a
    // large workspace that shares the device with resident weights but is NOT
    // counted against the residency budget, so reserve the tiled-scope peak out of
    // `vram_gb` -- otherwise a full weight set plus the workspace overruns the 8GB
    // device and the GPU resets (device lost) at high token counts. Makes vram_gb a
    // true TOTAL device ceiling, so real-res grids are measurable here.
    let dit_budget = {
        const HIDDEN: u64 = 2048; // hunyuan::config::dit::HIDDEN
        let n_img_u = n_img as u64;
        let joint = n_img_u + seq as u64;
        let ws_peak = 2 * HIDDEN * (6 * joint + 2 * n_img_u);
        let reserve = ws_peak + ws_peak / 3 + (128 << 20);
        ResidencyBudget {
            ram_bytes: budget.ram_bytes,
            vram_bytes: budget.vram_bytes.saturating_sub(reserve).max(1 << 30),
        }
    };

    // --- DiT denoise (scoped; free DiT VRAM before any VAE phase) ---
    let t0 = Instant::now();
    let latent = {
        let dit_src = SafetensorsSource::open(
            MmapFileOpener::new(&dit_path)
                .await
                .unwrap_or_else(|e| panic!("open {}: {e}", dit_path.display())),
        )
        .await
        .expect("parse DiT safetensors");
        let dit_res = WeightResidency::new(dit_src, dit_budget);
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
        // Perf only: the first `steps` labels of the production 4-step schedule
        // (denoising is meaningless on random inputs; we just want N real forwards).
        const LABELS: [u32; 4] = [1000, 750, 500, 250];
        let schedule = FlowMatchSchedule::build(&LABELS[..steps.clamp(1, 4)], 9.0, 1000);
        let latent = dit
            .denoise(
                &backend,
                &dit_res,
                &dit_ws,
                &text,
                seq,
                &latent_init,
                (gt, gh, gw),
                &schedule,
                window,
                None,
                None,
            )
            .await
            .expect("denoise");
        drop(dit_ws);
        dit_res.evict_all_and_free(&*backend);
        latent
    };
    let dit_s = t0.elapsed().as_secs_f64();
    let per_step = dit_s / steps as f64;
    eprintln!("[perf] DiT denoise: {dit_s:.1}s total, {per_step:.1}s/step ({steps} steps)");
    assert!(
        latent.iter().all(|v| v.is_finite()),
        "DiT produced non-finite latent"
    );

    if run_vae {
        let Some(vae_path) = resolve_hf(
            "THINFER_HUNYUAN_VAE",
            "models--Comfy-Org--HunyuanVideo_1.5_repackaged",
            "split_files/vae/hunyuanvideo15_vae_fp16.safetensors",
        ) else {
            eprintln!("skipped[hunyuan t2v_perf VAE]: Comfy VAE not in HF cache");
            return;
        };
        let t1 = Instant::now();
        let vae_src = SafetensorsSource::open(
            MmapFileOpener::new(&vae_path)
                .await
                .unwrap_or_else(|e| panic!("open {}: {e}", vae_path.display())),
        )
        .await
        .expect("parse VAE safetensors");
        let vae_res = WeightResidency::new(vae_src, budget);
        // Mid set is always F32 (causal attn). When VAE_F16, the up-stages run a
        // separate F16 set (the production perf config).
        let vae_mid = HunyuanVaePipelines::compile_with(&backend, ActDtype::F32)
            .await
            .expect("vae mid pl");
        let vae_up = if env_usize("THINFER_HY_VAE_F16", 0) != 0 {
            Some(
                HunyuanVaePipelines::compile_with(&backend, ActDtype::F16)
                    .await
                    .expect("vae up pl"),
            )
        } else {
            None
        };
        let vae = HunyuanVaeDecoder::new_mixed(vae_mid, vae_up, &vae_res).expect("vae");
        let vae_ws = Workspace::new(Arc::clone(&backend), Arc::clone(vae_res.arbiter()));
        let video = vae
            .decode(&backend, &vae_res, &vae_ws, &latent, gt, gh, gw)
            .await
            .expect("decode");
        vae_res.evict_all_and_free(&*backend);
        let vae_s = t1.elapsed().as_secs_f64();
        eprintln!("[perf] VAE decode: {vae_s:.1}s");
        assert!(
            video.iter().all(|v| v.is_finite()),
            "VAE produced non-finite video"
        );
    }

    if let Some(h) = trace.as_ref() {
        let mut buf = Vec::new();
        h.dump(&mut buf).ok();
        eprintln!("{}", String::from_utf8_lossy(&buf));
    }
}
