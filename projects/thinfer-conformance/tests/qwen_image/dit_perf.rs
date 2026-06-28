//! Qwen-Image DiT perf harness (engine-only, NO pyref): loads a handful of real
//! DiT blocks from the GGUF (like `dit_parity`, but `QWEN_PERF_BLOCKS` of them)
//! and runs `forward_multi` at a configurable joint sequence for a few steps,
//! reporting per-step wall + average per-block wall. This is the cheap perf
//! turnaround instrument -- it skips the 7B encoder / VAE / vision-tower / full
//! 60-layer load that makes `edit_e2e` heavy, while exercising the exact DiT hot
//! loop (the perf bottleneck). Pair with `THINFER_TRACE=1` to get the per-pipeline
//! `gpu_disp_ms` rollup at exit; (wall - gpu_disp) per block is the submit-bubble.
//!
//! Inputs are deterministic synthetic activations (perf is data-independent;
//! correctness is `dit_parity`'s job). Knobs (all optional):
//!   QWEN_PERF_BLOCKS  resident block count (default 6)
//!   QWEN_PERF_STEPS   denoise steps to time (default 3; first is warmup)
//!   QWEN_PERF_NOISE   noise grid "GHxGW" (default 16x16 = 256 tokens)
//!   QWEN_PERF_REF     ref grid "GHxGW" ("" = t2i, no ref; default 48x40 = 1920)
//!   QWEN_PERF_TXT     text token count (default 140)
//!   QWEN_PERF_BUDGET  vram budget GB (default 6)
//! Not a pass/fail gate (no assertions on time); it prints and returns.

#![cfg(feature = "qwen-image-e2e")]

use std::sync::Arc;
use std::time::Instant;

use thinfer_core::backend::{Backend, PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::qwen_image::dit::{QwenImageDit, QwenImageDitPipelines};
use thinfer_models::qwen_image::loader::register_handles;
use thinfer_models::qwen_image::manifest::{self, role};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

const IN_CHANNELS: usize = 64;
const JOINT_DIM: usize = 3584;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Parse a "GHxGW" grid; empty/unset -> None.
fn env_grid(key: &str, default: Option<(usize, usize)>) -> Option<(usize, usize)> {
    match std::env::var(key) {
        Ok(s) if s.trim().is_empty() => None,
        Ok(s) => {
            let (a, b) = s.split_once('x').expect("grid must be GHxGW");
            Some((a.parse().expect("grid gh"), b.parse().expect("grid gw")))
        }
        Err(_) => default,
    }
}

/// Deterministic ~N(0,1) fill (SplitMix64 -> one Box-Muller half). Magnitude is
/// irrelevant to timing; just keep it finite and non-degenerate.
fn synthetic(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next = || -> f64 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z = z ^ (z >> 31);
        ((z >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    };
    (0..n)
        .map(|_| {
            let u1 = next().max(1e-12);
            let u2 = next();
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        })
        .collect()
}

#[tokio::test(flavor = "current_thread")]
async fn dit_perf() {
    let _trace = thinfer_core::trace::init_from_env();

    let fr = manifest::MANIFEST
        .get(role::DIT_GGUF_Q8_0)
        .expect("dit gguf role");
    let Some(gguf_path) = cache::resolve(fr) else {
        eprintln!("skipped[dit_perf]: {}/{} not in HF cache", fr.repo, fr.path);
        return;
    };

    let blocks = env_usize("QWEN_PERF_BLOCKS", 6).max(1);
    let steps = env_usize("QWEN_PERF_STEPS", 3).max(1);
    let noise = env_grid("QWEN_PERF_NOISE", Some((16, 16))).expect("noise grid required");
    let refg = env_grid("QWEN_PERF_REF", Some((48, 40)));
    let txt_seq = env_usize("QWEN_PERF_TXT", 140).max(1);
    let budget_gb = env_usize("QWEN_PERF_BUDGET", 6).max(1);

    let mut grids = vec![(1, noise.0, noise.1)];
    if let Some(r) = refg {
        grids.push((1, r.0, r.1));
    }
    let noise_seq = noise.0 * noise.1;
    let img_seq: usize = grids.iter().map(|&(f, h, w)| f * h * w).sum();
    let joint = txt_seq + img_seq;
    eprintln!(
        "dit-perf: {blocks} blocks, {steps} steps, noise={}x{} ref={:?} txt={txt_seq} \
         -> img_seq={img_seq} joint={joint} (budget {budget_gb}G)",
        noise.0, noise.1, refg,
    );

    let img_tokens = synthetic(img_seq * IN_CHANNELS, 1);
    let txt_embeds = synthetic(txt_seq * JOINT_DIM, 2);

    let opener = MmapFileOpener::new(&gguf_path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", gguf_path.display()));
    let gguf = GgufSource::open(opener).await.expect("parse dit gguf");
    let budget = ResidencyBudget {
        ram_bytes: 48 << 30,
        vram_bytes: (budget_gb as u64) << 30,
    };
    let residency = WeightResidency::new(gguf, budget);
    let handles = register_handles(&residency, blocks).expect("register dit handles");

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

    // The shipped DiT config (bf16 acts, per-site Q8_0 matmuls).
    let cfgs = thinfer_models::qwen_image::dit::block_cfgs();
    // Fast-attention (f16 subgroup SDPA) default-on; QWEN_NO_I8 measures bf16.
    let i8_matmul = std::env::var_os("QWEN_NO_I8").is_none();
    let pipelines = QwenImageDitPipelines::compile(&backend, &cfgs, i8_matmul)
        .await
        .expect("compile dit pipelines");

    let scratch = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));
    let dit = QwenImageDit::new();

    let mut step_ms = Vec::with_capacity(steps);
    for s in 0..steps {
        let t0 = Instant::now();
        let out = dit
            .forward_multi(
                &backend,
                &pipelines,
                &residency,
                &scratch,
                &handles,
                &img_tokens,
                &txt_embeds,
                0.5,
                &grids,
                noise_seq,
                None,
            )
            .await
            .expect("dit forward_multi");
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(out.velocity.len(), noise_seq * IN_CHANNELS);
        let mem = backend.mem_account();
        eprintln!(
            "  step {s}: {ms:.0}ms total ({:.1}ms/block) vram={}MiB W={}MiB Ws={}MiB{}",
            ms / blocks as f64,
            mem.vram_total_current() / (1024 * 1024),
            mem.vram_current(thinfer_core::mem::VramCategory::Weights) / (1024 * 1024),
            mem.vram_current(thinfer_core::mem::VramCategory::Workspace) / (1024 * 1024),
            if s == 0 { " [warmup]" } else { "" },
        );
        if s > 0 {
            step_ms.push(ms);
        }
    }
    if !step_ms.is_empty() {
        let avg = step_ms.iter().sum::<f64>() / step_ms.len() as f64;
        eprintln!(
            "dit-perf: steady-state {:.0}ms/step ({:.1}ms/block over {blocks} blocks); \
             extrapolated 60-block step = {:.1}s",
            avg,
            avg / blocks as f64,
            avg / blocks as f64 * 60.0 / 1000.0,
        );
    }
}
