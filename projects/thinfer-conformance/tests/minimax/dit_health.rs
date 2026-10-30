//! MiniMax-H3 DiT health check: the first time the transformer itself runs.
//!
//! Drives the real Q4_K checkpoint through one full forward over a packed t2va
//! sequence: patch projections -> token refiner -> 50 blocks -> final modulated
//! norm -> both output heads. The conditioner is NOT loaded; the text rows are
//! synthetic, which keeps this a DiT test rather than an encoder test and keeps
//! the 13.6 GiB encoder out of the binary.
//!
//! HEALTH, not parity: it proves the stack dispatches against real weights, the
//! precomputed modulation table lines up with what the blocks index, and the
//! predicted velocity is finite and at a plausible scale. Numerical parity is a
//! separate gate and is NOT implied by this passing.

#![cfg(feature = "minimax-e2e")]

use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf;
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::ops::{ActDtype, WeightDtype};
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::quant::QuantKind;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::common::block::CoopmatSites;
use thinfer_models::common::merge::MergedSource;
use thinfer_models::minimax::DitConfig;
use thinfer_models::minimax::dit::{COOPMAT_SITES, DitPipelines, dit_cfgs, register_handles};
use thinfer_models::minimax::loader::{build_modulation_tables, register_video_patch_proj};
use thinfer_models::minimax::lora::LoraSource;
use thinfer_models::minimax::manifest::{self, role};
use thinfer_models::minimax::pipeline::{
    DenoisePlan, forward, initial_noise, upload_modulation_tables,
};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

/// A deliberately reduced canvas. The native 1344x768 does not fit 8 GiB until
/// the row-independent sites are chunked over rows, and this is a bring-up
/// check, not a quality run: 512x320 keeps the packed sequence near 1.3k rows.
const LATENT_FRAMES: usize = 7;
const LATENT_H: usize = 12;
const LATENT_W: usize = 20;
/// 40 latents/s over the ~0.9 s this many frames covers, stereo.
const AUDIO_LATENTS: usize = 37;
const TEXT_ROWS: usize = 64;
const STEPS: usize = 4;

/// Deterministic stand-in for the conditioner's hidden state. Real Qwen3-VL
/// hidden states are unnormalized and land at roughly unit scale.
fn synthetic_rows(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.618_034 + seed).fract() * 2.0 - 1.0) * 1.2)
        .collect()
}

#[tokio::test(flavor = "current_thread")]
async fn dit_health() {
    let _trace = thinfer_core::trace::init_from_env();

    let fr = manifest::MANIFEST
        .get(role::DIT_GGUF_Q4_K_M)
        .expect("dit role");
    let Some(path) = cache::resolve(fr) else {
        eprintln!("skipped[minimax dit_health]: DiT GGUF not in HF cache");
        return;
    };

    let cfg = DitConfig::h3_33b();
    let plan = DenoisePlan::build::<std::convert::Infallible>(
        &cfg,
        TEXT_ROWS,
        LATENT_FRAMES,
        LATENT_H,
        LATENT_W,
        AUDIO_LATENTS,
        STEPS,
    )
    .expect("build denoise plan");
    eprintln!(
        "---- minimax dit health ----\n\
         packed {} rows = {} text + {} audio + {} video, {} distinct timesteps",
        plan.packed.seq_len,
        plan.num_text_rows,
        plan.num_audio_rows,
        plan.num_video_rows,
        plan.modulation.timesteps.len()
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

    let opener = MmapFileOpener::new(&path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let base = gguf::GgufSource::open(opener)
        .await
        .expect("parse dit gguf");

    // The turbo distillation rides as a RUNTIME adapter in the same residency,
    // not folded into the base tensors: see `dit::BlockLoraHandles`.
    let lora_fr = manifest::MANIFEST
        .get(role::TURBO_LORA)
        .expect("turbo lora role");
    let lora_path = cache::resolve(lora_fr).expect("turbo LoRA not in HF cache");
    let lora_opener = MmapFileOpener::new(&lora_path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", lora_path.display()));
    let lora = SafetensorsSource::open(lora_opener)
        .await
        .expect("parse turbo lora safetensors");
    let src = MergedSource::new(
        base,
        LoraSource::turbo(&cfg, lora)
            .await
            .expect("stack turbo lora"),
    );
    eprintln!("turbo lora: {} synthetic tensors", src.overlay_len());

    // The table is built over exactly the plan's distinct timesteps, in the
    // plan's order: the block stack indexes it by that position.
    let started = std::time::Instant::now();
    let tables = build_modulation_tables(&cfg, &src, &plan.modulation.timesteps)
        .await
        .expect("build modulation tables");
    eprintln!(
        "adaln precompute {:.1}s, {:.1} MB table",
        started.elapsed().as_secs_f64(),
        tables.bytes(4) as f64 / 1.0e6
    );

    let residency = WeightResidency::new(
        src,
        ResidencyBudget {
            ram_bytes: 6 << 30,
            vram_bytes: 4 << 30,
        },
    );
    let mut handles = register_handles(&cfg, &residency, None).expect("register dit handles");
    // The converter's superblock reshape hides this tensor's true shape; left
    // alone the upload transpose scrambles every video row with no NaN.
    handles.video_patch_proj =
        register_video_patch_proj(&cfg, &residency).expect("register video_patch_proj");

    // Production config, plus a dense twin so the tensor-core path can be held
    // against the reference matmul below.
    let cfgs = dit_cfgs(
        WeightDtype::Quant(QuantKind::Q4_K),
        ActDtype::Bf16,
        COOPMAT_SITES,
    );
    let pipelines = DitPipelines::compile(&backend, &cfgs)
        .await
        .expect("compile dit pipelines");
    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    let (table, _held) = upload_modulation_tables(&backend, &workspace, &tables, ActDtype::Bf16)
        .expect("upload table");

    let (video, audio) = initial_noise(&cfg, plan.num_video_rows, plan.num_audio_rows, 42);
    let text = synthetic_rows(plan.num_text_rows * cfg.text_dim, 0.25);

    let started = std::time::Instant::now();
    let v = forward(
        &cfg, &backend, &pipelines, &residency, &workspace, &handles, &table, &plan, 0, &video,
        &audio, &text,
    )
    .await
    .expect("dit forward");
    let elapsed = started.elapsed().as_secs_f64();

    assert_eq!(v.video.len(), plan.num_video_rows * cfg.video_patch_dim());
    assert_eq!(v.audio.len(), plan.num_audio_rows * cfg.audio_latents_dim);

    for (name, out) in [("video", &v.video), ("audio", &v.audio)] {
        let bad = out.iter().filter(|x| !x.is_finite()).count();
        assert_eq!(bad, 0, "{bad} non-finite values in the {name} velocity");
        let (mut lo, mut hi, mut sum) = (f32::MAX, f32::MIN, 0.0f64);
        for &x in out.iter() {
            lo = lo.min(x);
            hi = hi.max(x);
            sum += (x as f64) * (x as f64);
        }
        let rms = (sum / out.len() as f64).sqrt();
        eprintln!("{name} velocity: range [{lo:.3}, {hi:.3}], rms {rms:.4}");
        assert!(hi > lo, "{name} velocity is constant");
        // Rectified-flow velocity against unit-variance noise sits at order 1.
        // Orders of magnitude off means a misread projection or a scrambled
        // row layout, neither of which produces a NaN.
        assert!(
            rms > 1.0e-3 && rms < 1.0e2,
            "{name} velocity rms {rms} is implausible"
        );
    }
    eprintln!("forward {elapsed:.1}s for {} rows", plan.packed.seq_len);

    // The adapter must actually reach the output. A wrong per-head scatter in
    // the fused-qkv synthesis produces a plausible-but-wrong model with no NaN
    // and no range anomaly, so compare against the base stack directly.
    assert_eq!(
        handles.lora_block_count(),
        cfg.n_layers + cfg.token_refiner_layers,
        "every block must carry the turbo adapter"
    );
    let base_only = handles.without_lora();
    let v0 = forward(
        &cfg, &backend, &pipelines, &residency, &workspace, &base_only, &table, &plan, 0, &video,
        &audio, &text,
    )
    .await
    .expect("base-only dit forward");

    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (a, b) in v.video.iter().zip(&v0.video) {
        num += ((a - b) as f64).powi(2);
        den += (*b as f64).powi(2);
    }
    let rel = (num / den).sqrt();
    eprintln!(
        "turbo adapter changes the video velocity by {:.2}% rel",
        rel * 100.0
    );
    assert!(
        rel > 0.01,
        "adapter had no effect ({rel}); the synthesized qkv pair is not reaching the model"
    );
    assert!(
        rel < 2.0,
        "adapter changed the velocity by {rel} rel; suspect a wrong per-head scatter"
    );

    // Every tensor-core site must agree with the reference dense matmul. The
    // kernel computes in f16, so this is a CLOSENESS gate, not equality -- but
    // a mis-sited coopmat (an A-side or an output that leaves the f16 range)
    // shows up as a large divergence, and the final velocity can still look
    // finite and correctly-scaled while being wrong. Sites are checked ONE AT A
    // TIME so a failure names the culprit instead of just the bundle.
    let none = CoopmatSites::default();
    let dense_pipes = DitPipelines::compile(
        &backend,
        &dit_cfgs(WeightDtype::Quant(QuantKind::Q4_K), ActDtype::Bf16, none),
    )
    .await
    .expect("compile dense dit pipelines");
    let reference = forward(
        &cfg,
        &backend,
        &dense_pipes,
        &residency,
        &workspace,
        &handles,
        &table,
        &plan,
        0,
        &video,
        &audio,
        &text,
    )
    .await
    .expect("dense dit forward");
    assert!(
        reference.video.iter().all(|x| x.is_finite()),
        "the dense reference itself went non-finite"
    );

    let rel_to_reference = |out: &[f32]| -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (a, b) in out.iter().zip(&reference.video) {
            num += ((a - b) as f64).powi(2);
            den += (*b as f64).powi(2);
        }
        (num / den).sqrt()
    };

    // Every site is measured even when production has it disabled: the whole
    // point is to know what the tensor-core path WOULD cost before turning it
    // on. The assertion below only fires for sites production actually enables.
    let mut worst: Option<(&str, f64)> = None;
    for (name, sites) in [
        ("qkv", CoopmatSites { qkv: true, ..none }),
        ("proj", CoopmatSites { proj: true, ..none }),
        (
            "ffn_up",
            CoopmatSites {
                ffn_up: true,
                ..none
            },
        ),
    ] {
        let pipes = DitPipelines::compile(
            &backend,
            &dit_cfgs(WeightDtype::Quant(QuantKind::Q4_K), ActDtype::Bf16, sites),
        )
        .await
        .unwrap_or_else(|e| panic!("compile {name} coopmat pipelines: {e:?}"));
        let out = forward(
            &cfg, &backend, &pipes, &residency, &workspace, &handles, &table, &plan, 0, &video,
            &audio, &text,
        )
        .await
        .unwrap_or_else(|e| panic!("{name} coopmat forward: {e:?}"));
        let finite = out.video.iter().all(|x| x.is_finite());
        let rel = rel_to_reference(&out.video);
        eprintln!(
            "coopmat {name}: {:.3}% rel vs dense, finite={finite}",
            rel * 100.0
        );
        assert!(finite, "coopmat site {name} produced non-finite velocity");
        if coopmat_site_enabled(name) && worst.is_none_or(|(_, w)| rel > w) {
            worst = Some((name, rel));
        }
    }
    let Some((name, rel)) = worst else {
        eprintln!("no coopmat site is enabled in production; divergences above are advisory");
        return;
    };
    assert!(
        rel < 0.05,
        "coopmat site {name} diverges from dense by {rel} rel; its A-side or its          output leaves the f16 range"
    );
}

/// Whether the production config runs this site on the tensor cores.
fn coopmat_site_enabled(name: &str) -> bool {
    match name {
        "qkv" => COOPMAT_SITES.qkv,
        "proj" => COOPMAT_SITES.proj,
        "ffn_up" => COOPMAT_SITES.ffn_up,
        _ => false,
    }
}
