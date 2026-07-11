//! Forever end-to-end FastWan2.2-TI2V-5B video parity test. Drives the full
//! engine path - tokenize -> umT5 encode -> DMD 3-step denoise -> TI2V
//! high-compression VAE decode -> CTHW fp32 frames - against a pinned PyTorch
//! reference (`gen_fastwan_video_e2e_ref.py`) for the same prompt / dims / seed /
//! initial noise / per-step renoise. DMD re-noises between steps, so we make it
//! byte-parity-friendly by DUMPING the exact initial latent + each per-step
//! renoise tensor (via the engine's own `gaussian_noise` / `renoise_seed`) and
//! handing them to the reference; both sides then consume identical noise rather
//! than each drawing from its own RNG.
//!
//! Compare set (per `WanStepDiag` + the pre-VAE / decode taps):
//!   - `dit_out_step{i}` : raw DiT velocity, localizes a per-step blowup.
//!   - `step{i}_post`    : latent after each DMD step.
//!   - `pre_vae_latent`  : final pre-VAE latent (== last step's post).
//!   - `vae_rgb`         : decoded CTHW video, exercises DupUp3D / unpatchify e2e.
//!
//! Health is also asserted on both paths: every stage finite + non-trivial, the
//! decode clamped to [-1, 1], and the VRAM/RAM true peak within the residency
//! budget (the arbiter's phase-aware streaming is what fits 5B in a thin budget).
//!
//! `THINFER_E2E_SKIP_PYREF=1` is the perf/trace escape: it skips the reference
//! and the divergence checks (budget + health still enforced) so a TRACE run can
//! profile without `uv`. Parity-ON is the norm.
//!
//! Run (parity; needs the HF bundle in cache + `uv`, no download):
//!   `cargo test -p thinfer-conformance --features wan-e2e --release \
//!    video_e2e -- --nocapture --test-threads=1`
//!
//! Knobs: `THINFER_E2E_{WIDTH,HEIGHT,FRAMES}` resize (defaults 32x32, 5 frames);
//! `THINFER_E2E_BUDGET_GB` residency budget; `THINFER_E2E_PNG_DIR` stages the
//! per-frame PNG sequence + contact sheet for both sides (`ours_*` / `theirs_*`);
//! `THINFER_WAN_DIAG` dumps step-0 stage magnitudes and returns early;
//! `THINFER_TRACE` emits the per-scope + gpu_ms rollup and cell-level diff dumps.

#![cfg(feature = "wan-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::Backend;
use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::ops::WeightDtype;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::trace::{self, DIAG};
use thinfer_core::workspace::Workspace;
use thinfer_models::wan::manifest::{self, role};
use thinfer_models::wan::pipeline::{
    GenerationParams, VaeChoice, VideoSampler, WanModel, WanStepDiag, gaussian_noise, renoise_seed,
};
use thinfer_models::wan::source::WanSource;
use thinfer_models::z_image::pipeline::encode_png;
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;
use thinfer_native::tokenizer::HfTokenizer;

const PROMPT: &str = "a red balloon over a green field";
const SEED: u64 = 42;

/// TI2V high-compression VAE geometry + DiT latent channels (z_dim). The latent
/// grid the DiT denoises derives from these; they are fixed by the model, so a
/// drift here means the test and the pipeline disagree about the tensor shape.
const Z_DIM: usize = 48;
const VAE_SCALE: usize = 16;
const TEMPORAL_SCALE: usize = 4;
/// DiT text context length (umT5 rows cross-attended), == `config::TEXT_SEQ`.
const TEXT_SEQ: usize = 512;
/// FastWan DMD schedule length (`DmdConfig::fastwan_ti2v_5b` = `[1000,757,522]`).
const DMD_STEPS: usize = 3;

/// Printed-only per-cell tol: a cell is counted "over" when `|got - exp| >
/// max(ref_max_abs * TOL_MULT, TOL_FLOOR)`. Informational (f16 scatter trips it
/// even with slope == 1), NOT the gate; the gate is [`CmpTol`] (slope + rmse).
const TOL_MULT: f32 = 0.03;
const TOL_FLOOR: f32 = 1e-3;

/// Acceptance band for a precision-dominated parity stage: the linfit slope must
/// sit within `±slope_dev` of 1, and the relative rmse (`rmse / ref_max_abs`)
/// under `rel_rmse`. Slope catches systematic scale/sign regressions (every Wan
/// DiT bug -- #1 explosion, #4 norm2 -- showed as a slope blowup); rel_rmse
/// bounds f16 scatter. Cell-count-over-a-fixed-tol can't see a systematic slope
/// offset, so it is the wrong gate for these stages.
#[derive(Clone, Copy)]
struct CmpTol {
    slope_dev: f64,
    rel_rmse: f64,
}
impl CmpTol {
    /// Never trips: for env-gated diagnostics (ident / fwddiff) that print but
    /// must not fail the run.
    const INERT: CmpTol = CmpTol {
        slope_dev: f64::INFINITY,
        rel_rmse: f64::INFINITY,
    };
}

/// Bands from the measured f16 floor (256x256x5, 2026-06-19) + margin. High-t
/// steps are bit-clean (slope 1.000, rel_rmse ~0.003). The final low-t step and
/// the VAE carry f16 act precision amplified by the stiff low-sigma velocity
/// field (slope 0.99 -> 0.96, rel_rmse up to ~0.053), proven to be precision not
/// a bug (the f16-vs-fp32 act-swing at t=522 exceeds the deviation from the fp32
/// reference). Prod runs f16 acts, so this floor is the real target.
const TOL_EARLY: CmpTol = CmpTol {
    slope_dev: 0.02,
    rel_rmse: 0.012,
};
const TOL_FINAL_LATENT: CmpTol = CmpTol {
    slope_dev: 0.03,
    rel_rmse: 0.055,
};
const TOL_VAE: CmpTol = CmpTol {
    slope_dev: 0.06,
    rel_rmse: 0.075,
};

/// Dumps the `RollupHandle` to stderr on drop (success or panic) so the
/// per-scope + gpu_ms timing table survives a failing assertion.
struct RollupDumpOnDrop(Option<thinfer_core::trace::RollupHandle>);
impl Drop for RollupDumpOnDrop {
    fn drop(&mut self) {
        if let Some(h) = self.0.take() {
            let _ = h.dump(&mut std::io::stderr());
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn video_e2e_safetensors() {
    let _rollup = RollupDumpOnDrop(trace::init_from_env());

    let env_u32 = |k: &str, d: u32| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    // Parity default is 256x256x5: the dims the [`CmpTol`] bands are calibrated
    // at. Smaller grids are NOT valid parity targets -- at e.g. 32x32x5 the DiT
    // is only 2 tokens, so a single high-magnitude outlier channel dominates the
    // whole-tensor slope/rmse and trips the bands even though the per-stage
    // forward is bit-clean (confirmed via the THINFER_WAN_DIAG_STEP bisect: the
    // modulation path is exact at t=757; the deviation is one outlier channel and
    // it averages out over the 256x256 token grid -> step1 rel_rmse 3.1e-3). Use
    // a small grid only for fast SKIP_PYREF perf/health smoke, never for parity.
    let width = env_u32("THINFER_E2E_WIDTH", 256);
    let height = env_u32("THINFER_E2E_HEIGHT", 256);
    let num_frames = env_u32("THINFER_E2E_FRAMES", 5);
    let skip_pyref = std::env::var("THINFER_E2E_SKIP_PYREF").is_ok();
    // THINFER_E2E_TINY=1 also exercises the LightTAE tiny decoder: loads its
    // weights alongside the real VAE (the Full parity gate is untouched) and,
    // after the gate, decodes the same latent single- vs multi-chunk to prove
    // the temporal-tiling carry is exact, plus a NaN-free + clamp-range check.
    // No pyref (no TAEHV reference); the whole-run TRUE_PEAK assert covers VRAM.
    let want_tiny = std::env::var("THINFER_E2E_TINY").is_ok();
    // Prompt override (threaded to BOTH the engine and the pyref so a parity run
    // can localize a prompt-specific divergence, e.g. umT5 odd-token EOS pad).
    let prompt = std::env::var("THINFER_E2E_PROMPT").unwrap_or_else(|_| PROMPT.to_string());

    let f_lat = (num_frames as usize - 1) / TEMPORAL_SCALE + 1;
    let h_lat = height as usize / VAE_SCALE;
    let w_lat = width as usize / VAE_SCALE;
    let n_lat = Z_DIM * f_lat * h_lat * w_lat;
    let out_frames = TEMPORAL_SCALE * f_lat - 3;
    let rgb_elems = 3 * out_frames * height as usize * width as usize;

    eprintln!(
        "video-e2e[safetensors]: {width}x{height} frames={num_frames} skip_pyref={skip_pyref} \
         (f_lat={f_lat}, h_lat={h_lat}, w_lat={w_lat}, n_lat={n_lat}, out_frames={out_frames})"
    );

    // Resolve every safetensors weight role (1 DiT + 3 umT5 + 1 VAE) plus the
    // tokenizer. Skip cleanly if any role misses the HF cache - never download.
    let weight_roles = [
        role::DIT,
        role::TEXT_ENCODER_SHARD_1,
        role::TEXT_ENCODER_SHARD_2,
        role::TEXT_ENCODER_SHARD_3,
        role::VAE,
    ];
    let mut needed: Vec<&str> = weight_roles.to_vec();
    needed.push(role::TOKENIZER_JSON);
    // The LightTAE decoder weight is a separate lightx2v download; resolve it
    // only when the tiny path is requested (else the gate never needs it).
    if want_tiny {
        needed.push(role::TINY_VAE);
    }
    let mut resolved: Vec<(&str, PathBuf)> = Vec::with_capacity(needed.len());
    for r in needed {
        let fr = manifest::MANIFEST.get(r).expect("role in manifest");
        match cache::resolve(fr) {
            Some(p) => resolved.push((r, p)),
            None => {
                eprintln!(
                    "skipped: {}/{} not in HF cache ({})",
                    fr.repo,
                    fr.path,
                    cache::cache_root().display()
                );
                return;
            }
        }
    }
    let path_of = |name: &str| -> &Path { &resolved.iter().find(|(r, _)| *r == name).unwrap().1 };
    eprintln!("video-e2e[safetensors]: all roles resolved from HF cache");

    // Deterministic pinned noise (same generator as the zimage e2e) so reruns
    // start from an identical latent and the PNG output is comparable run-to-run.
    let noise = make_pinned_noise(n_lat);
    summarize("noise (pinned)", &noise);
    // Per-step renoise tensors the DMD loop will consume internally, reproduced
    // here via the engine's own fns so the dumped bytes are identical to what
    // `denoise_with` draws. The reference byte-loads these.
    let renoise: Vec<Vec<f32>> = (0..DMD_STEPS - 1)
        .map(|i| gaussian_noise(n_lat, renoise_seed(SEED, i)))
        .collect();

    // Tmpdir for the pinned-noise inputs + the py-dumped stage tensors.
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("wan_video_e2e");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let noise_path = tmp.join("initial_noise.bin");
    let renoise_paths: Vec<PathBuf> = (0..DMD_STEPS - 1)
        .map(|i| tmp.join(format!("renoise_step{i}.bin")))
        .collect();
    let py_step_paths: Vec<PathBuf> = (0..DMD_STEPS)
        .map(|i| tmp.join(format!("py_step{i}_post.bin")))
        .collect();
    let py_dit_paths: Vec<PathBuf> = (0..DMD_STEPS)
        .map(|i| tmp.join(format!("py_dit_out_step{i}.bin")))
        .collect();
    let py_pre_vae_path = tmp.join("py_pre_vae_latent.bin");
    let py_vae_rgb_path = tmp.join("py_vae_rgb.bin");
    // DiT-internal step-0 dumps (compared in the THINFER_WAN_DIAG bisect branch).
    let py_in = |name: &str| tmp.join(format!("py_in_{name}.bin"));
    let py_in_block = |i: usize| tmp.join(format!("py_in_block{i:02}.bin"));

    if !skip_pyref {
        // Clear stale py dumps so a stale file can't mask a step that never ran.
        let mut clear = vec![py_pre_vae_path.clone(), py_vae_rgb_path.clone()];
        clear.extend(py_step_paths.iter().cloned());
        clear.extend(py_dit_paths.iter().cloned());
        for n in [
            "umt5_hidden",
            "patch_x",
            "temb",
            "timestep_proj",
            "text_proj",
            "final_norm",
            "proj_out",
        ] {
            clear.push(py_in(n));
        }
        clear.extend((0..64).map(py_in_block)); // generous upper bound on blocks
        for n in [
            "norm1_premod",
            "norm1",
            "self_q",
            "self_k",
            "self_v",
            "self_sa",
            "after_self",
            "norm2",
            "after_cross",
            "norm3",
            "ffn_gelu",
            "ffn_down",
        ] {
            clear.push(tmp.join(format!("py_b0_{n}.bin")));
        }
        for p in &clear {
            let _ = std::fs::remove_file(p);
        }
        std::fs::write(&noise_path, bytemuck_cast(&noise)).expect("write noise");
        for (p, r) in renoise_paths.iter().zip(&renoise) {
            std::fs::write(p, bytemuck_cast(r)).expect("write renoise");
        }
        run_python_ref(
            &noise_path,
            &renoise_paths,
            path_of(role::DIT),
            &tmp,
            &prompt,
            width,
            height,
            num_frames,
            SEED,
        );
    }

    // Opt-in frame staging: per-frame PNG sequence + tiled contact sheet.
    let png_dir = std::env::var_os("THINFER_E2E_PNG_DIR").map(PathBuf::from);
    if let Some(d) = png_dir.as_ref() {
        std::fs::create_dir_all(d).expect("create THINFER_E2E_PNG_DIR");
        eprintln!("png staging enabled: {}", d.display());
    }

    // Build the safetensors source (openers in WEIGHT_ROLES order). No GGUF
    // union for the bringup path.
    let mut openers: Vec<MmapFileOpener> = Vec::with_capacity(weight_roles.len());
    for r in weight_roles {
        let path = path_of(r);
        openers.push(
            MmapFileOpener::new(path)
                .await
                .unwrap_or_else(|e| panic!("open {}: {e}", path.display())),
        );
    }
    // Append the tiny decoder as an extra shard (its `decoder.{N}` keys are
    // disjoint from the real VAE's), so both decoders load from one catalog.
    if want_tiny {
        let path = path_of(role::TINY_VAE);
        openers.push(
            MmapFileOpener::new(path)
                .await
                .unwrap_or_else(|e| panic!("open {}: {e}", path.display())),
        );
    }
    let source = WanSource::open(openers, None)
        .await
        .expect("parse weight files");

    // 6 GB thin-hardware default for the 256x256x5 parity grid (measured TRUE_PEAK
    // ~5.98 GiB): still well under the ~10 GB bf16 DiT + ~11 GB fp32 umT5 footprint,
    // so it forces per-phase eviction at each stage boundary and exercises the
    // arbiter's hard ceiling (TRUE_PEAK assert). Higher budgets relax via the same
    // adaptive arbiter, never a flag. Drop to 2 with a small grid (e.g. SKIP_PYREF
    // perf smoke). THINFER_E2E_BUDGET_GB overrides.
    let budget_gb: u64 = std::env::var("THINFER_E2E_BUDGET_GB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);
    let budget = ResidencyBudget {
        ram_bytes: budget_gb << 30,
        vram_bytes: budget_gb << 30,
    };

    let cfg = WgpuConfig {
        power_preference: match std::env::var("THINFER_POWER_PREF")
            .ok()
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("low" | "lowpower" | "integrated") => PowerPreference::LowPower,
            Some("none") => PowerPreference::None,
            _ => PowerPreference::HighPerformance,
        },
        timestamps: std::env::var("THINFER_TRACE").is_ok(),
        disable_coopmat: std::env::var("THINFER_NO_COOPMAT").is_ok(),
    };
    let backend = Arc::new(
        WgpuBackend::new_with_config(cfg)
            .await
            .expect("wgpu adapter unavailable for tests"),
    );
    let tokenizer = HfTokenizer::from_path(path_of(role::TOKENIZER_JSON))
        .await
        .expect("tokenizer load");

    let residency = WeightResidency::new(source, budget);
    // THINFER_E2E_ACT=fp32 forces fp32 block acts (default: probe device f16).
    // Used with THINFER_E2E_IDENT_CHECK to classify the end-step residual as
    // amplified-bf16 (collapses under fp32) vs algorithmic (dtype-independent).
    let act_override = match std::env::var("THINFER_E2E_ACT").as_deref() {
        Ok("fp32") => Some(thinfer_core::ops::ActDtype::F32),
        Ok("f16") => Some(thinfer_core::ops::ActDtype::F16),
        _ => None,
    };
    // The parity gate is the real VAE (`Full`). Loading `Tiny` additionally
    // registers the LightTAE decoder (both are then available); the Full path
    // is unchanged, so the gate is identical with or without the tiny add-on.
    let load_vae = if want_tiny {
        VaeChoice::Tiny
    } else {
        VaeChoice::Full
    };
    let model = WanModel::load_with_act(
        Arc::clone(&backend),
        residency,
        tokenizer,
        load_vae,
        act_override,
        std::env::var_os("THINFER_WAN_I8_FFN").is_some(),
    )
    .await
    .expect("WanModel::load");

    // The FastWan variant transcodes bf16 block matmuls to Q8_0 at load. A
    // misnamed rename map / wrong probe would silently fall to some other dtype.
    let dit_w = model.dit_matmul_weight();
    eprintln!("[safetensors] DiT matmul weight dtype: {dit_w:?} (expected Quant(Q8_0))");
    assert_eq!(
        dit_w,
        WeightDtype::Quant(thinfer_core::quant::QuantKind::Q8_0),
        "safetensors DiT matmul dtype"
    );

    // Temporal self-attention window (latent frames) for perf/quality A/B. Set
    // THINFER_E2E_ATTN_WINDOW=W to restrict self-attention to +-W latent frames;
    // unset = full attention (the parity reference). Windowing CHANGES the output
    // (it is a different op), so a windowed run is a rust-vs-rust quality/perf
    // comparison, not a pyref-parity run.
    let attn_window = std::env::var("THINFER_E2E_ATTN_WINDOW")
        .ok()
        .and_then(|s| s.parse::<u32>().ok());
    let params = GenerationParams {
        prompt: prompt.clone(),
        height,
        width,
        num_frames,
        seed: SEED,
        // Parity gate: the DMD reference schedule (step-diag taps are DMD-only).
        sampler: VideoSampler::Dmd,
        attn_window,
        steps: None,
    };

    // Step-0 stage telemetry (magnitudes only; no reference). Localizes a
    // numerical blowup to a stage during bringup, then returns early.
    if std::env::var("THINFER_WAN_DIAG").is_ok() {
        // Which DMD step the per-stage bisect captures (default 0 == t=1000).
        // Point it at the first divergent step (e.g. 1 == t=757) to localize a
        // timestep-specific divergence to a stage. The pyref dumps its internal
        // taps at the SAME step (it reads THINFER_WAN_DIAG_STEP), and for step>0
        // we feed the reference's input to that step (py_step{N-1}_post,
        // drift-stripped) so engine and pyref share an identical forward input.
        let diag_step: usize = std::env::var("THINFER_WAN_DIAG_STEP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let diag_latent: Vec<f32> = if diag_step == 0 {
            noise.clone()
        } else {
            assert!(
                !skip_pyref,
                "THINFER_WAN_DIAG_STEP>0 needs the pyref step input (py_step{}_post); \
                 cannot run with SKIP_PYREF",
                diag_step - 1
            );
            read_f32(&py_step_paths[diag_step - 1])
        };
        let mut ws = Workspace::new(Arc::clone(&backend), Arc::clone(model.arbiter()));
        let d = model
            .diag_step_at(&params, &diag_latent, diag_step, 0, &mut ws)
            .await
            .expect("diag_step_at");
        eprintln!(
            "=== WAN STEP-{diag_step} STAGE TELEMETRY (timestep={}) ===",
            d.timestep
        );
        summarize("umt5_hidden", &d.umt5_hidden);
        // umT5 first-NaN localization (the odd-token-prompt blocker). Walks
        // embeds -> per-block post-residual outputs -> per-op intermediates of
        // the first NaN block, so one SKIP_PYREF run names the exact op. seq is
        // the REAL token count; the even-pad row is sliced from layer/op taps'
        // logical view but the readbacks are seq_pad-wide, so just report raw.
        {
            // Count NON-FINITE (NaN OR inf): f16 act overflow -> inf is the
            // suspected failure mode, and inf is_nan()==false so a nan-only
            // check misses it. Report max finite |.| too, to catch a blowup
            // approaching the f16 ceiling (65504) before it tips to inf.
            let bad_of = |v: &[f32]| v.iter().filter(|x| !x.is_finite()).count();
            let maxabs_of = |v: &[f32]| {
                v.iter()
                    .filter(|x| x.is_finite())
                    .fold(0f32, |m, x| m.max(x.abs()))
            };
            eprintln!(
                "=== umT5 first-nonfinite localization (umt5_seq={}) ===",
                d.umt5_seq
            );
            summarize("umt5.embeds", &d.umt5_embeds);
            let first_bad_layer = d.umt5_layer_outputs.iter().position(|v| bad_of(v) > 0);
            for (i, v) in d.umt5_layer_outputs.iter().enumerate() {
                eprintln!(
                    "[umt5.layer[{i:02}]] len={} nonfinite={} max_abs={:.4e}",
                    v.len(),
                    bad_of(v),
                    maxabs_of(v),
                );
            }
            // Per-op trace of the FIRST block that produced a non-finite value
            // (block_ops holds every block; index matches layer_outputs).
            if let Some(li) = first_bad_layer {
                eprintln!("--- first nonfinite at layer {li}; per-op trace of block {li} ---");
                if let Some(ops) = d.umt5_block_ops.get(li) {
                    for (name, v) in [
                        ("n1", &ops.n1),
                        ("q", &ops.q),
                        ("k", &ops.k),
                        ("v", &ops.v),
                        ("sa", &ops.sa),
                        ("proj", &ops.proj),
                        ("after_attn", &ops.after_attn),
                        ("n2", &ops.n2),
                        ("wi0", &ops.wi0),
                        ("wi1", &ops.wi1),
                        ("gu", &ops.gu),
                        ("wo", &ops.wo),
                    ] {
                        summarize(&format!("umt5.b{li}.{name}"), v);
                    }
                }
            } else {
                eprintln!("[umt5] NO nonfinite in any layer output");
            }
        }
        summarize("patch_x (post-patchify input)", &d.patch_x);
        summarize("temb", &d.temb);
        summarize("timestep_proj", &d.timestep_proj);
        summarize("text_proj", &d.text_proj);
        for (name, v) in &d.block0_stages {
            summarize(&format!("block0.{name}"), v);
        }
        for (i, b) in d.per_block.iter().enumerate() {
            if !b.is_empty() {
                summarize(&format!("per_block[{i:02}] residual"), b);
            }
        }
        summarize("final_norm", &d.final_norm);
        summarize("proj_out", &d.proj_out);
        summarize("dit_out (flow velocity)", &d.dit_out);
        summarize("stepped (= our_step0_prev_sample)", &d.stepped);

        // --- stage-by-stage bisect vs the pyref step-0 internal dumps. The
        // engine `diag_step0` reproduces the FIRST DMD forward (t=1000), which
        // the pyref also dumps; compare each stage and report the FIRST that
        // diverges. `check` prints every stage's stats + linfit (slope ~1 ==
        // clean, ~0 == uncorrelated), so one run localizes the bug. ---
        if !skip_pyref {
            let n_tok = f_lat * (h_lat / 2) * (w_lat / 2);
            eprintln!("=== STEP-{diag_step} BISECT vs pyref (n_tok={n_tok}) ===");
            let mut diverged: Option<String> = None;
            let mut cmp = |label: &str, got: &[f32], path: &Path, rows: usize| {
                if !path.exists() {
                    eprintln!("[{label}] MISSING pyref dump {}", path.display());
                    return;
                }
                let py = read_f32(path);
                if rows > 0 {
                    cross_token_var(&format!("{label} ours"), got, rows);
                    cross_token_var(&format!("{label} pyref"), &py, rows);
                }
                check(label, got, &py, TOL_EARLY, true, &mut diverged);
            };
            cmp(
                "umt5_hidden",
                &d.umt5_hidden,
                &py_in("umt5_hidden"),
                d.umt5_seq,
            );
            cmp("patch_x", &d.patch_x, &py_in("patch_x"), n_tok);
            cmp("temb", &d.temb, &py_in("temb"), 0);
            cmp(
                "timestep_proj",
                &d.timestep_proj,
                &py_in("timestep_proj"),
                0,
            );
            cmp("text_proj", &d.text_proj, &py_in("text_proj"), TEXT_SEQ);
            // Block-0 internal ops (engine `block0_stages`, in execution order) vs
            // the pyref instrumented block-0 forward. Only the module-boundary
            // stages have a pyref dump; q/k/v/sa are engine-only for now.
            for (name, v) in &d.block0_stages {
                let path = tmp.join(format!("py_b0_{name}.bin"));
                if path.exists() {
                    cmp(&format!("b0.{name}"), v, &path, n_tok);
                }
            }
            for (i, b) in d.per_block.iter().enumerate() {
                if !b.is_empty() {
                    cmp(&format!("block{i:02}"), b, &py_in_block(i), n_tok);
                }
            }
            cmp("final_norm", &d.final_norm, &py_in("final_norm"), n_tok);
            cmp("proj_out", &d.proj_out, &py_in("proj_out"), n_tok);
            cross_token_var("dit_out ours", &d.dit_out, n_tok);
            // Dump engine stages to scratch for offline analysis (modulation solve
            // etc.) so the slow pyref need not rerun. Scratch only, never committed.
            if let Ok(dir) = std::env::var("THINFER_WAN_ENG_DUMP") {
                let dir = PathBuf::from(dir);
                std::fs::create_dir_all(&dir).ok();
                let put = |name: &str, v: &[f32]| {
                    std::fs::write(dir.join(format!("eng_{name}.bin")), bytemuck_cast(v)).ok();
                };
                put("patch_x", &d.patch_x);
                put("temb", &d.temb);
                put("timestep_proj", &d.timestep_proj);
                put("text_proj", &d.text_proj);
                for (name, v) in &d.block0_stages {
                    put(&format!("b0_{name}"), v);
                }
                put("final_norm", &d.final_norm);
                put("proj_out", &d.proj_out);
                put("dit_out", &d.dit_out);
                eprintln!("[eng-dump] wrote engine stages to {}", dir.display());
            }
            match diverged {
                Some(msg) => eprintln!("\n>>> FIRST STAGE DIVERGENCE: {msg}"),
                None => eprintln!("\n>>> all step-{diag_step} stages within tol"),
            }
        }
        return;
    }

    let t_full = std::time::Instant::now();
    tracing::info!(target: DIAG, t_ms = 0_u64, "milestone: starting denoise_with");
    let mut ws = Workspace::new(Arc::clone(&backend), Arc::clone(model.arbiter()));
    // Capture per-step velocity + post for the parity compare (cleared on entry).
    let mut step_diag: Vec<WanStepDiag> = Vec::with_capacity(DMD_STEPS);
    let (pre_vae, got_f_lat, got_h_lat, got_w_lat) = model
        .denoise_with(
            &params,
            Some(&noise),
            &mut ws,
            Some(&mut step_diag),
            None,
            None,
        )
        .await
        .expect("denoise_with");
    tracing::info!(
        target: DIAG,
        t_ms = t_full.elapsed().as_millis() as u64,
        pre_vae_len = pre_vae.len(),
        f_lat = got_f_lat,
        steps = step_diag.len(),
        "milestone: denoise_with done",
    );
    assert_eq!(got_f_lat, f_lat, "f_lat");
    assert_eq!(got_h_lat, h_lat, "h_lat");
    assert_eq!(got_w_lat, w_lat, "w_lat");
    assert_eq!(pre_vae.len(), n_lat, "pre-VAE latent length");
    assert_eq!(step_diag.len(), DMD_STEPS, "step_diag count");
    for (i, s) in step_diag.iter().enumerate() {
        summarize(&format!("our_dit_out_step{i} (velocity)"), &s.velocity);
        summarize(&format!("our_step{i}_post"), &s.post);
    }
    summarize("pre_vae_latent", &pre_vae);
    assert_healthy("pre_vae_latent", &pre_vae);

    // Diagnostic: dump the pre-VAE latent and return before the VAE decode.
    // Lets a DiT-only run (no VAE) capture the latent for offline analysis
    // (budget-to-budget identity, left-right mirror symmetry). Scratch only.
    if let Ok(p) = std::env::var("THINFER_E2E_LATENT_DUMP") {
        std::fs::write(&p, bytemuck_cast(&pre_vae)).expect("write latent dump");
        eprintln!(
            "[latent-dump] wrote pre_vae latent ({} f32, f_lat={f_lat} h_lat={h_lat} w_lat={w_lat}) to {p}",
            pre_vae.len()
        );
        return;
    }

    // VAE decode -> CTHW fp32 frames [3, out_frames, H, W] (clamped to [-1, 1]).
    let frames = model
        .decode_latent_to_video(&pre_vae, f_lat, h_lat, w_lat, VaeChoice::Full, &mut ws)
        .await
        .expect("decode_latent_to_video");
    tracing::info!(
        target: DIAG,
        t_ms = t_full.elapsed().as_millis() as u64,
        frames_len = frames.len(),
        frames_expected = rgb_elems,
        "milestone: vae decode done",
    );
    assert_eq!(frames.len(), rgb_elems, "decoded frame element count");
    summarize("frames (CTHW [-1,1])", &frames);
    assert_healthy("frames", &frames);
    // Decode clamps to [-1, 1]; a value far outside means the unpatchify /
    // unnormalize math is wrong even though the values are finite.
    let max_abs = frames.iter().copied().map(f32::abs).fold(0f32, f32::max);
    assert!(
        max_abs <= 1.5,
        "decoded frames exceed the clamp range: max_abs={max_abs:.4e}"
    );

    // Tiny (LightTAE) decoder: NaN-free + clamp-range health, and the temporal
    // tiling carry is EXACT -- decoding the same latent as one chunk vs one
    // chunk-per-latent-frame must be bit-identical (memcat is the only temporal
    // coupling, carried losslessly across boundaries). No pyref (no TAEHV
    // reference); VRAM is covered by the whole-run TRUE_PEAK assert below, which
    // now includes these tiny decodes. Runs the multi-chunk carry path even at
    // the small parity grid, so it is a real (not single-chunk) tiling check.
    if want_tiny {
        let single = model
            .decode_latent_to_video_chunked(
                &pre_vae,
                f_lat,
                h_lat,
                w_lat,
                VaeChoice::Tiny,
                &mut ws,
                Some(f_lat.max(1)), // whole clip in one chunk (== untiled)
            )
            .await
            .expect("tiny decode (single chunk)");
        assert_eq!(single.len(), rgb_elems, "tiny decoded element count");
        summarize("tiny frames single-chunk (CTHW [-1,1])", &single);
        assert_healthy("tiny frames", &single);
        let tiny_max = single.iter().copied().map(f32::abs).fold(0f32, f32::max);
        assert!(
            tiny_max <= 1.5,
            "tiny decoded frames exceed clamp range: max_abs={tiny_max:.4e}"
        );

        let multi = model
            .decode_latent_to_video_chunked(
                &pre_vae,
                f_lat,
                h_lat,
                w_lat,
                VaeChoice::Tiny,
                &mut ws,
                Some(1), // one latent frame per chunk -> exercises every carry
            )
            .await
            .expect("tiny decode (multi chunk)");
        assert_eq!(multi.len(), single.len(), "tiny chunk lengths differ");
        let chunk_diff = single
            .iter()
            .zip(&multi)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "[tiny] single-vs-multi-chunk max_abs_diff={chunk_diff:.3e} (f_lat={f_lat}, chunks={f_lat})"
        );
        assert_eq!(
            chunk_diff, 0.0,
            "tiny temporal tiling not exact: single vs {f_lat}-chunk decode differ by {chunk_diff:.3e}"
        );
    }

    // Stage frames before the budget/parity assertions so a later failure still
    // leaves the visual output (ours, and theirs when the reference ran).
    if let Some(d) = png_dir.as_ref() {
        stage_frames(&frames, out_frames, height, width, d, "ours");
        if !skip_pyref && py_vae_rgb_path.exists() {
            let py_rgb = read_f32(&py_vae_rgb_path);
            if py_rgb.len() == rgb_elems {
                stage_frames(&py_rgb, out_frames, height, width, d, "theirs");
            } else {
                eprintln!(
                    "[png] theirs skipped: py_vae_rgb len {} != {rgb_elems}",
                    py_rgb.len()
                );
            }
        }
    }

    // Engine-only f16-vs-fp32 self-diff (no pyref). Holding dims constant and
    // varying ONLY the act dtype isolates how much of the low-t forward residual
    // is f16-act rounding (amplified by the stiffer low-sigma Jacobian) vs
    // algorithmic. THINFER_E2E_FWD_DUMP writes each step's input + velocity;
    // a second run at THINFER_E2E_ACT=fp32 reads them via THINFER_E2E_FWD_REF,
    // reruns OUR forward on the SAME inputs, and diffs. Small fp32-vs-f16 diff at
    // step0 (t=1000) but large at the end step => the residual is precision.
    if let Ok(dir) = std::env::var("THINFER_E2E_FWD_DUMP") {
        let d = PathBuf::from(dir);
        std::fs::create_dir_all(&d).expect("fwd-dump dir");
        for i in 0..DMD_STEPS {
            let input: &[f32] = if i == 0 {
                &noise
            } else {
                &step_diag[i - 1].post
            };
            std::fs::write(d.join(format!("fwd_in{i}.bin")), bytemuck_cast(input))
                .expect("write fwd input");
            std::fs::write(
                d.join(format!("fwd_vel{i}.bin")),
                bytemuck_cast(&step_diag[i].velocity),
            )
            .expect("write fwd velocity");
        }
        eprintln!(
            "[fwd-dump] wrote {DMD_STEPS} step inputs+velocities to {}",
            d.display()
        );
    }
    if let Ok(dir) = std::env::var("THINFER_E2E_FWD_REF") {
        let d = PathBuf::from(dir);
        let mut div = None;
        for i in 0..DMD_STEPS {
            let input = read_f32(&d.join(format!("fwd_in{i}.bin")));
            let ref_vel = read_f32(&d.join(format!("fwd_vel{i}.bin")));
            let our = model
                .forward_velocity_at(&params, &input, i, &mut ws)
                .await
                .expect("forward_velocity_at");
            check(
                &format!("fwddiff_step{i} (this-act vs dumped, t=step{i})"),
                &our,
                &ref_vel,
                CmpTol::INERT, // diagnostic: never trip the gate
                true,
                &mut div,
            );
        }
    }

    let snap = backend.mem_account().snapshot();
    eprintln!(
        "[mem] vram TRUE_PEAK={} / budget {}",
        fmt_mib(snap.vram_total_peak),
        fmt_mib(budget.vram_bytes),
    );
    eprintln!(
        "[mem]   peaks: weights={} workspace={} staging={}",
        fmt_mib(snap.vram_weights.1),
        fmt_mib(snap.vram_workspace.1),
        fmt_mib(snap.vram_staging.1),
    );
    eprintln!(
        "[mem] ram  TRUE_PEAK={} / budget {}",
        fmt_mib(snap.ram_total_peak),
        fmt_mib(budget.ram_bytes),
    );

    // --- parity vs the pyref (the gate; skipped only by THINFER_E2E_SKIP_PYREF) ---
    if !skip_pyref {
        let mut diverged: Option<String> = None;
        let trace_dump = std::env::var("THINFER_TRACE").is_ok();

        for i in 0..DMD_STEPS {
            // The final low-t step carries the amplified f16 precision floor; the
            // earlier high-t steps are bit-clean (see CmpTol bands).
            let band = if i == DMD_STEPS - 1 {
                TOL_FINAL_LATENT
            } else {
                TOL_EARLY
            };
            let py_v = read_f32(&py_dit_paths[i]);
            let py_p = read_f32(&py_step_paths[i]);
            assert_eq!(py_v.len(), n_lat, "py_dit_out_step{i} length");
            assert_eq!(py_p.len(), n_lat, "py_step{i}_post length");
            check(
                &format!("dit_out_step{i} (velocity)"),
                &step_diag[i].velocity,
                &py_v,
                band,
                trace_dump,
                &mut diverged,
            );
            check(
                &format!("step{i}_post"),
                &step_diag[i].post,
                &py_p,
                band,
                trace_dump,
                &mut diverged,
            );
        }

        // Identical-input forward check (THINFER_E2E_IDENT_CHECK): for each step
        // i>0, run OUR forward on the REFERENCE's step input (py_step{i-1}_post)
        // at step i's timestep and compare to the reference velocity. This strips
        // accumulated input drift, leaving only per-step forward error -- the
        // discriminator for "is step-i divergence a forward bug or drift the
        // schedule amplifies". A clean result (~step-0's identical-input floor)
        // means the wider end-step tol is precision, not a bug.
        if std::env::var("THINFER_E2E_IDENT_CHECK").is_ok() {
            let mut ident_diverged: Option<String> = None;
            for i in 1..DMD_STEPS {
                let ref_input = read_f32(&py_step_paths[i - 1]);
                let our_v = model
                    .forward_velocity_at(&params, &ref_input, i, &mut ws)
                    .await
                    .expect("forward_velocity_at");
                check(
                    &format!("ident_step{i} (our forward, ref input, t=step{i})"),
                    &our_v,
                    &read_f32(&py_dit_paths[i]),
                    CmpTol::INERT, // diagnostic only: never trip the gate
                    true,          // always print linfit
                    &mut ident_diverged,
                );
            }
        }
        let py_pre_vae = read_f32(&py_pre_vae_path);
        let py_vae_rgb = read_f32(&py_vae_rgb_path);
        assert_eq!(py_pre_vae.len(), n_lat, "py_pre_vae_latent length");
        assert_eq!(py_vae_rgb.len(), rgb_elems, "py_vae_rgb length");
        check(
            "pre_vae_latent",
            &pre_vae,
            &py_pre_vae,
            TOL_FINAL_LATENT,
            trace_dump,
            &mut diverged,
        );
        check(
            "vae_rgb (CTHW [-1,1])",
            &frames,
            &py_vae_rgb,
            TOL_VAE,
            trace_dump,
            &mut diverged,
        );

        if let Some(msg) = diverged {
            panic!("FIRST DIVERGENCE: {msg}");
        }
        eprintln!("[safetensors] parity OK vs pyref");
    } else {
        eprintln!("[safetensors] SKIP_PYREF: divergence checks skipped (health + budget enforced)");
    }

    assert!(
        snap.vram_total_peak <= budget.vram_bytes,
        "vram true peak {} > budget {}",
        fmt_mib(snap.vram_total_peak),
        fmt_mib(budget.vram_bytes),
    );
    assert!(
        snap.ram_total_peak <= budget.ram_bytes,
        "ram true peak {} > budget {}",
        fmt_mib(snap.ram_total_peak),
        fmt_mib(budget.ram_bytes),
    );

    eprintln!(
        "[safetensors] FastWan e2e OK: {DMD_STEPS}-step DMD -> {out_frames} frames in {:?}",
        t_full.elapsed()
    );
}

/// Invoke the PyTorch reference over `uv run`. Dumps the compare-set `.bin`
/// tensors into `out_dir`. Panics if `uv` is missing or the ref fails.
#[allow(clippy::too_many_arguments)]
fn run_python_ref(
    noise_path: &Path,
    renoise_paths: &[PathBuf],
    dit_shard: &Path,
    out_dir: &Path,
    prompt: &str,
    width: u32,
    height: u32,
    num_frames: u32,
    seed: u64,
) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    // Reference precision class. Default bf16 (the model's trained dtype); set
    // `THINFER_E2E_REF_DTYPE=fp32` to compare the engine against the fp32 truth
    // rather than another 16-bit rounding (two distinct 16-bit paths can deviate
    // from each other more than either does from fp32).
    let ref_dtype = std::env::var("THINFER_E2E_REF_DTYPE").unwrap_or_else(|_| "bf16".into());
    let mut cmd = Command::new("uv");
    cmd.args([
        "run",
        "--directory",
        py_dir.to_str().unwrap(),
        "python",
        "-m",
        "thinfer_pytorch_ref.wan.gen_fastwan_video_e2e_ref",
        "--initial-noise",
        noise_path.to_str().unwrap(),
        "--transformer-shard",
        dit_shard.to_str().unwrap(),
        "--out",
        out_dir.to_str().unwrap(),
        "--prompt",
        prompt,
        "--height",
        &height.to_string(),
        "--width",
        &width.to_string(),
        "--frames",
        &num_frames.to_string(),
        "--seed",
        &seed.to_string(),
        "--dtype",
        &ref_dtype,
    ]);
    for r in renoise_paths {
        cmd.args(["--renoise", r.to_str().unwrap()]);
    }
    let status = cmd
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "FastWan e2e pyref failed");
}

/// Compare `got` vs `expected` against a [`CmpTol`] band: gate on linfit slope
/// (systematic scale/sign) + relative rmse (scatter). The per-cell over-tol
/// count is printed for context only. Records the first stage that breaches the
/// band in `diverged`.
fn check(
    label: &str,
    got: &[f32],
    expected: &[f32],
    tol: CmpTol,
    trace_dump: bool,
    diverged: &mut Option<String>,
) {
    let n = got.len().min(expected.len());
    let max_ref = expected[..n]
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0f32, f32::max);
    let abs_tol = (max_ref * TOL_MULT).max(TOL_FLOOR);
    let mut max_abs = 0f32;
    let mut n_over = 0usize;
    for i in 0..n {
        let d = (got[i] - expected[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
        if !got[i].is_finite() || !expected[i].is_finite() || d > abs_tol {
            n_over += 1;
        }
    }
    let (a, b, rmse, cnt) = linfit(expected, got);
    let rel_rmse = if max_ref > 0.0 {
        rmse / max_ref as f64
    } else {
        rmse
    };
    let slope_dev = (a - 1.0).abs();
    eprintln!(
        "[{label}] slope={a:.5} rel_rmse={rel_rmse:.4e} (band dev<={:.3}, rel<={:.3}) \
         rmse={rmse:.4e} max_abs={max_abs:.4e} cells_over_abstol={n_over}/{n} ref_max_abs={max_ref:.4e}",
        tol.slope_dev, tol.rel_rmse
    );
    if trace_dump && n > 0 {
        eprintln!("[{label}] linfit got ~= {a:.6}*exp + {b:+.6} (n={cnt})");
        let dump_n = 8.min(n);
        eprintln!("[{label}] head got={:?}", &got[..dump_n]);
        eprintln!("[{label}] head exp={:?}", &expected[..dump_n]);
    }
    if diverged.is_none() && cnt > 2 && (slope_dev > tol.slope_dev || rel_rmse > tol.rel_rmse) {
        *diverged = Some(format!(
            "{label}: slope={a:.4} (dev {slope_dev:.4} > {:.3}) OR rel_rmse={rel_rmse:.4e} \
             (> {:.3}); rmse={rmse:.4e}, max_abs={max_abs:.4e}, ref_max_abs={max_ref:.4e}",
            tol.slope_dev, tol.rel_rmse
        ));
    }
}

/// Least-squares fit `got ~= a*exp + b` over finite pairs; returns `(a, b, rmse,
/// count)`. `a` near 1 + low rmse == clean; `a` off == a scale bug.
fn linfit(exp: &[f32], got: &[f32]) -> (f64, f64, f64, usize) {
    let (mut sx, mut sy, mut sxx, mut sxy, mut nf) = (0f64, 0f64, 0f64, 0f64, 0usize);
    for (&x, &y) in exp.iter().zip(got) {
        if x.is_finite() && y.is_finite() {
            let (x, y) = (x as f64, y as f64);
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
            nf += 1;
        }
    }
    if nf < 2 {
        return (f64::NAN, f64::NAN, f64::NAN, nf);
    }
    let n = nf as f64;
    let denom = n * sxx - sx * sx;
    let a = if denom.abs() > 0.0 {
        (n * sxy - sx * sy) / denom
    } else {
        f64::NAN
    };
    let b = (sy - a * sx) / n;
    let mut se = 0f64;
    for (&x, &y) in exp.iter().zip(got) {
        if x.is_finite() && y.is_finite() {
            let r = y as f64 - (a * x as f64 + b);
            se += r * r;
        }
    }
    (a, b, (se / n).sqrt(), nf)
}

/// Assert a stage is finite and non-trivial: NaN/inf is a hard fail, and an
/// all-(near)-zero tensor means a stage silently produced nothing.
fn assert_healthy(label: &str, v: &[f32]) {
    let nonfinite = v.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(nonfinite, 0, "{label}: {nonfinite} non-finite cells");
    let max_abs = v.iter().copied().map(f32::abs).fold(0f32, f32::max);
    assert!(
        max_abs > 1e-6,
        "{label}: all-zero output (max_abs={max_abs:.3e})"
    );
}

/// Write a per-frame PNG sequence (`<prefix>_frame{n}.png`) plus a tiled
/// contact-sheet PNG (`<prefix>_contact.png`) from CTHW frames in `[-1, 1]`.
fn stage_frames(frames: &[f32], n: usize, h: u32, w: u32, dir: &Path, prefix: &str) {
    let (hh, ww) = (h as usize, w as usize);
    let per = hh * ww;
    let frame_chw = |f: usize| -> Vec<f32> {
        let mut chw = vec![0.0f32; 3 * per];
        for c in 0..3 {
            let src = (c * n + f) * per;
            chw[c * per..(c + 1) * per].copy_from_slice(&frames[src..src + per]);
        }
        chw
    };
    for f in 0..n {
        match encode_png(&frame_chw(f), w, h) {
            Ok(png) => {
                let p = dir.join(format!("{prefix}_frame{f}.png"));
                std::fs::write(&p, &png).expect("write frame png");
                eprintln!("wrote {}", p.display());
            }
            Err(e) => eprintln!("encode_png frame{f} failed: {e}"),
        }
    }

    let cols = (n as f64).sqrt().ceil() as usize;
    let rows = n.div_ceil(cols);
    let (sheet_h, sheet_w) = (rows * hh, cols * ww);
    let mut sheet = vec![-1.0f32; 3 * sheet_h * sheet_w];
    for f in 0..n {
        let (gr, gc) = (f / cols, f % cols);
        let chw = frame_chw(f);
        for c in 0..3 {
            for y in 0..hh {
                let dst_y = gr * hh + y;
                let dst = (c * sheet_h + dst_y) * sheet_w + gc * ww;
                let src = (c * per) + y * ww;
                sheet[dst..dst + ww].copy_from_slice(&chw[src..src + ww]);
            }
        }
    }
    match encode_png(&sheet, sheet_w as u32, sheet_h as u32) {
        Ok(png) => {
            let p = dir.join(format!("{prefix}_contact.png"));
            std::fs::write(&p, &png).expect("write contact sheet");
            eprintln!(
                "wrote {} ({}x{} grid {}x{})",
                p.display(),
                cols,
                rows,
                sheet_w,
                sheet_h
            );
        }
        Err(e) => eprintln!("encode_png contact sheet failed: {e}"),
    }
}

/// SplitMix64 + Box-Muller. Identical to the zimage e2e so reruns start from an
/// identical latent.
fn make_pinned_noise(n: usize) -> Vec<f32> {
    let mut state: u64 = 0xFEED_F00D_C0DE_BEEFu64;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let uniform = |x: u64| -> f64 { ((x >> 11) as f64 / (1u64 << 53) as f64).max(1e-12) };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = uniform(next());
        let u2 = uniform(next());
        let mag = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push((mag * theta.cos()) as f32);
        if out.len() < n {
            out.push((mag * theta.sin()) as f32);
        }
    }
    out
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "{} not f32-aligned", path.display());
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bytemuck_cast(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn fmt_mib(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.2}GiB", bytes as f64 / (1u64 << 30) as f64)
    } else {
        format!("{:.1}MiB", bytes as f64 / (1u64 << 20) as f64)
    }
}

/// For a `[rows, dim]` row-major tensor, report how much each channel varies
/// ACROSS tokens (mean over channels of the cross-token variance). Near-zero ==
/// the rows are token-uniform, which is the signature the worklog flagged for
/// the bad DiT velocity. Also prints the mean per-channel magnitude for scale.
fn cross_token_var(label: &str, v: &[f32], rows: usize) {
    if rows == 0 || !v.len().is_multiple_of(rows) {
        return;
    }
    let dim = v.len() / rows;
    let (mut var_acc, mut mag_acc) = (0f64, 0f64);
    for c in 0..dim {
        let (mut sum, mut sumsq) = (0f64, 0f64);
        for r in 0..rows {
            let x = v[r * dim + c] as f64;
            sum += x;
            sumsq += x * x;
        }
        let mean = sum / rows as f64;
        var_acc += sumsq / rows as f64 - mean * mean;
        mag_acc += mean.abs();
    }
    eprintln!(
        "[{label}] cross_token_var={:.4e} mean_chan_mag={:.4e} (rows={rows} dim={dim})",
        var_acc / dim as f64,
        mag_acc / dim as f64,
    );
}

fn summarize(label: &str, v: &[f32]) {
    let (mut max_abs, mut min, mut max, mut sum, mut nan) =
        (0f32, f32::INFINITY, f32::NEG_INFINITY, 0.0f64, 0usize);
    for &x in v {
        if x.is_nan() {
            nan += 1;
            continue;
        }
        let a = x.abs();
        if a > max_abs {
            max_abs = a;
        }
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
        sum += x as f64;
    }
    let denom = (v.len() - nan).max(1) as f64;
    eprintln!(
        "[{label}] len={} nan={} min={:.4e} max={:.4e} max_abs={:.4e} mean={:.4e}",
        v.len(),
        nan,
        min,
        max,
        max_abs,
        sum / denom,
    );
}
