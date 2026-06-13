//! Forever end-to-end SkyReels-V2-DF (Wan) video parity test. Drives the engine
//! through the full path - tokenize -> umT5 encode -> N synchronous-DF denoise
//! steps -> 3D causal VAE decode -> CTHW fp32 frames - against a pinned PyTorch
//! reference for the same prompt, dims, frames, seed, and initial noise. Both
//! sides byte-load the same noise so divergence is attributable to engine math
//! (umT5 / DiT / scheduler / VAE), never to RNG drift. Mirrors
//! `tests/zimage/e2e_parity.rs`.
//!
//! Drives the `gen_video_e2e_parity_ref.py` reference via `uv run` and
//! per-stage-compares `step{i}_post`, `pre_vae_latent`, and `vae_rgb` (broken-
//! vs-noisy tolerances; tighten after the first clean run). `umt5_out` and the
//! VAE-encode tap are not wired yet (umT5 breakage still surfaces at step0).
//! `THINFER_E2E_SKIP_PYREF=1` skips the reference + checks but keeps the
//! VRAM/RAM budget asserts and PNG staging.
//!
//! Run with (pyref needs `uv` + the HF bundle in cache):
//!   `cargo test -p thinfer-conformance --features wan-e2e --release \
//!    video_e2e_parity -- --nocapture`

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
use thinfer_models::wan::pipeline::{GenerationParams, WanModel, WanStepDiag};
use thinfer_models::wan::scheduler::UniPCScheduler;
use thinfer_models::wan::source::WanSource;
use thinfer_models::z_image::pipeline::encode_png;
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;
use thinfer_native::tokenizer::HfTokenizer;

/// Pinned config (wan-plan "e2e parity"). Lowest pyref that still exercises both
/// causal-VAE branches each side: 64x64, F=5 -> f_lat=2, h_lat=w_lat=8; DiT
/// tokens = 2*4*4 = 32; 2 steps; no CFG (one DiT forward/step). The pinned noise
/// is `[16, 2, 8, 8]` = 2048 f32, byte-loaded both sides. The dims/steps live as
/// literal defaults in the `env_u32` calls below (overridable via THINFER_E2E_*);
/// there are deliberately NO dim/step consts, so a stray const use can't silently
/// desync the resolved value from what the loop actually runs.
const PROMPT: &str = "a red balloon over a green field";
const SEED: u64 = 42;
/// DF `fps_embedding` bucket (binary onehot into the `[2, inner]` table). The
/// pipeline maps real fps -> bucket via `0 if fps == 16 else 1`, so `REAL_FPS`
/// (any value != 16) yields this bucket; both sides land on 1.
const FPS_BUCKET: usize = 1;
/// Real fps handed to the pyref; `!= 16` so its bucket == `FPS_BUCKET`.
const REAL_FPS: u32 = 24;

const Z_DIM: usize = 16;
const VAE_SCALE: usize = 8;
const TEMPORAL_SCALE: usize = 4;
/// umT5-XXL hidden dim (`D_MODEL`). Used to slice the rust seq_pad-row taps to
/// the real `seq` rows for the per-layer pyref linfit.
const UMT5_DIM: usize = 4096;

/// Per-tap tolerance: a cell is "over" if it differs by more than
/// `max(TOL_MULT * ref_max_abs, TOL_FLOOR)`. Mirrors the zimage e2e.
const TOL_MULT: f32 = 0.03;
const TOL_FLOOR: f32 = 1e-3;

/// Broken-vs-noisy first-pass caps (cells allowed over per-tap tol per stage).
/// Generous on purpose: catches NaN/inf and gross structural breakage while a
/// numerically-healthy bf16-vs-bf16 run passes. TIGHTEN after the first clean
/// run lands real numbers (worklog 4b). Cell counts at the pinned config:
/// n_lat = 16*2*8*8 = 2048; vae_rgb = 3*5*64*64 = 61440.
const CAP_STEP: usize = 1024;
const CAP_PRE_VAE: usize = 1024;
const CAP_VAE_RGB: usize = 30_720;

/// `THINFER_E2E_SKIP_PYREF=1` skips the pytorch reference + divergence checks.
/// The budget assertions still run: the workspace peak must not depend on
/// whether the reference was executed.
fn skip_pyref() -> bool {
    std::env::var("THINFER_E2E_SKIP_PYREF").is_ok()
}

/// Dumps the `RollupHandle` to stderr on drop (test success or panic) so the
/// per-scope timing table survives a `panic!` from a divergence check.
struct RollupDumpOnDrop(Option<thinfer_core::trace::RollupHandle>);
impl Drop for RollupDumpOnDrop {
    fn drop(&mut self) {
        if let Some(h) = self.0.take() {
            let _ = h.dump(&mut std::io::stderr());
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn video_e2e_parity_safetensors() {
    // Dump the TRACE rollup (per-scope CPU + GPU-timestamp table) on test exit,
    // success or panic, so a parity panic doesn't swallow the perf breakdown.
    let _rollup = RollupDumpOnDrop(trace::init_from_env());

    // The committed canary is the fast 2-step / 64px config (the consts); these
    // env knobs let a deep accuracy run (more steps -> exercises the order-2
    // corrector + shows whether the per-step divergence compounds) or a large-dim
    // perf run (use with THINFER_E2E_SKIP_PYREF; the CPU pyref can't scale) reuse
    // the same harness without editing the gate.
    let env_u32 = |k: &str, d: u32| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let width = env_u32("THINFER_E2E_WIDTH", 64);
    let height = env_u32("THINFER_E2E_HEIGHT", 64);
    let num_frames = env_u32("THINFER_E2E_FRAMES", 5);
    let steps_n = env_u32("THINFER_E2E_STEPS", 2);

    let f_lat = (num_frames as usize - 1) / TEMPORAL_SCALE + 1;
    let h_lat = height as usize / VAE_SCALE;
    let w_lat = width as usize / VAE_SCALE;
    let n_lat = Z_DIM * f_lat * h_lat * w_lat;
    let out_frames = TEMPORAL_SCALE * f_lat - 3;
    let rgb_elems = 3 * out_frames * height as usize * width as usize;

    let skip_pyref = skip_pyref();
    eprintln!(
        "video-e2e[safetensors]: {width}x{height} frames={num_frames} steps={steps_n} \
         (f_lat={f_lat}, h_lat={h_lat}, w_lat={w_lat}, n_lat={n_lat}, out_frames={out_frames})"
    );

    // Resolve every safetensors weight role (8: 2 DiT + 5 umT5 + 1 VAE) plus the
    // tokenizer. Skip cleanly if any role misses the HF cache - never download.
    let weight_roles = [
        role::DIT_SHARD_1,
        role::DIT_SHARD_2,
        role::TEXT_ENCODER_SHARD_1,
        role::TEXT_ENCODER_SHARD_2,
        role::TEXT_ENCODER_SHARD_3,
        role::TEXT_ENCODER_SHARD_4,
        role::TEXT_ENCODER_SHARD_5,
        role::VAE,
    ];
    let mut needed: Vec<&str> = weight_roles.to_vec();
    needed.push(role::TOKENIZER_JSON);
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

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("wan_safetensors");
    std::fs::create_dir_all(&tmp).expect("tmpdir");

    // Deterministic pinned noise, same generator as the zimage e2e so both
    // tests exercise the same starting tensor and the pyref can byte-load it.
    let noise = make_pinned_noise(n_lat);
    summarize("noise (pinned)", &noise);
    let noise_path = tmp.join("video_e2e_noise.bin");
    std::fs::write(&noise_path, bytemuck_cast(&noise)).expect("write noise");

    // Opt-in frame staging. THINFER_E2E_PNG_DIR gets a per-frame PNG sequence
    // plus a tiled contact sheet for at-a-glance inspection (no codec).
    let png_dir = std::env::var_os("THINFER_E2E_PNG_DIR").map(PathBuf::from);
    if let Some(d) = png_dir.as_ref() {
        std::fs::create_dir_all(d).expect("create THINFER_E2E_PNG_DIR");
        eprintln!("png staging enabled: {}", d.display());
    }

    // Per-stage pyref dump paths (siblings of the noise file). Clear stale
    // dumps first so a stale file can't mask a hook that never fired.
    let py_pre_vae_path = tmp.join("py_pre_vae_latent.bin");
    let py_vae_rgb_path = tmp.join("py_vae_rgb.bin");
    let py_step_paths: Vec<PathBuf> = (0..steps_n as usize)
        .map(|i| tmp.join(format!("py_step{i}_post.bin")))
        .collect();
    // Per-stage diag dumps (umT5 + raw DiT velocity + within-DiT taps). Cleared
    // alongside the rest so a stale dump can't mask a hook that never fired.
    let py_diag_paths: Vec<PathBuf> = [
        "py_umt5_out.bin",
        "py_temb.bin",
        "py_timestep_proj.bin",
        "py_proj_out.bin",
        "py_patch_in.bin",
        "py_text_in.bin",
    ]
    .iter()
    .map(|n| tmp.join(n))
    .chain((0..steps_n as usize).map(|i| tmp.join(format!("py_dit_out_step{i}.bin"))))
    .collect();
    if !skip_pyref {
        for p in std::iter::once(&py_pre_vae_path)
            .chain(std::iter::once(&py_vae_rgb_path))
            .chain(py_step_paths.iter())
            .chain(py_diag_paths.iter())
        {
            let _ = std::fs::remove_file(p);
        }
        // Sweep stale per-block + per-umT5-layer dumps (counts are the models'
        // layer counts, not test consts) so a dropped hook can't be masked by an
        // old file.
        if let Ok(rd) = std::fs::read_dir(&tmp) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if name.starts_with("py_block") || name.starts_with("py_umt5_") {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        }
        // Drive the PyTorch reference (bf16 DiT/umT5, fp32 VAE) before building
        // the engine model. Writes py_*.bin into `tmp` and, if staging, py_*
        // PNGs beside ours_*.
        run_python_ref(
            &noise_path,
            &tmp,
            png_dir.as_deref(),
            height,
            width,
            num_frames,
            steps_n,
        );
    } else {
        eprintln!("video-e2e[safetensors]: SKIP_PYREF set; pytorch reference not invoked");
    }

    // Build the safetensors source (openers in WEIGHT_ROLES order). No GGUF
    // union for the parity path.
    let mut openers: Vec<MmapFileOpener> = Vec::with_capacity(weight_roles.len());
    for r in weight_roles {
        let path = path_of(r);
        openers.push(
            MmapFileOpener::new(path)
                .await
                .unwrap_or_else(|e| panic!("open {}: {e}", path.display())),
        );
    }
    let source = WanSource::open(openers, None)
        .await
        .expect("parse weight files");

    // Aggressive budget to exercise eviction + rolling residency: well under the
    // ~6 GB DiT + ~12 GB umT5 fp32 footprint, forcing the phase-aware eviction
    // the engine does at each stage boundary. THINFER_E2E_BUDGET_GB overrides.
    let budget_gb: u64 = std::env::var("THINFER_E2E_BUDGET_GB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
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
    let model = WanModel::load(Arc::clone(&backend), residency, tokenizer)
        .await
        .expect("WanModel::load");

    // Safetensors path must compile the DiT matmuls against bf16. A misnamed
    // rename map / wrong probe would silently fall to some other dtype.
    let dit_w = model.dit_matmul_weight();
    eprintln!("[safetensors] DiT matmul weight dtype: {dit_w:?} (expected Bf16)");
    assert_eq!(dit_w, WeightDtype::Bf16, "safetensors DiT matmul dtype");

    // Bringup telemetry: THINFER_WAN_DIAG dumps every step-0 DiT stage (umT5 ->
    // patchify -> condition embedder -> per-block residual -> final norm ->
    // proj_out -> scheduler) and returns early. Localizes a numerical blowup to a
    // stage without the Python reference.
    if std::env::var("THINFER_WAN_DIAG").is_ok() {
        let params = GenerationParams {
            prompt: PROMPT.to_string(),
            // Parity pins guidance_scale = 1.0 -> no CFG -> one DiT forward per
            // step (matches the pyref `guidance_scale=1.0`).
            negative_prompt: String::new(),
            guidance_scale: 1.0,
            height,
            width,
            num_frames,
            steps: steps_n,
            seed: SEED,
            fps: FPS_BUCKET,
        };
        let mut ws = Workspace::new(Arc::clone(&backend), Arc::clone(model.arbiter()));
        let d = model
            .diag_step0(&params, &noise, &mut ws)
            .await
            .expect("diag_step0");
        eprintln!(
            "=== WAN STEP-0 STAGE TELEMETRY (timestep={}) ===",
            d.timestep
        );
        summarize("umt5_hidden", &d.umt5_hidden);
        // Full umT5 magnitude trace: embeds -> every block output -> hidden.
        // A clean scale that appears at one stage and persists localizes the
        // ~0.365x text-output bug (embeds vs a block vs the final RMSNorm).
        eprintln!(
            "--- umT5 per-layer magnitude trace (real seq={}) ---",
            d.umt5_seq
        );
        summarize("umt5.embeds", &d.umt5_embeds);
        for (i, lo) in d.umt5_layer_outputs.iter().enumerate() {
            summarize(&format!("umt5.block{i:02}_out"), lo);
        }
        // Block-0 per-op intermediates (within-block magnitude detail).
        if let Some(bo) = d.umt5_block_ops.first() {
            for (name, v) in [
                ("n1", &bo.n1),
                ("q", &bo.q),
                ("k", &bo.k),
                ("v", &bo.v),
                ("sa", &bo.sa),
                ("proj", &bo.proj),
                ("after_attn", &bo.after_attn),
                ("n2", &bo.n2),
                ("wi0", &bo.wi0),
                ("wi1", &bo.wi1),
                ("gu", &bo.gu),
                ("wo", &bo.wo),
            ] {
                summarize(&format!("umt5.block0.{name}"), v);
            }
        }
        // Per-channel (over D_MODEL) RMS of each block output: the residual
        // stream carries an outlier dimension that inflates the final RMSNorm
        // denominator (the 0.33x bulk scale). Find the dims that dominate the
        // LAST block and watch them grow layer by layer. (ours only here; the
        // ours-vs-py split is in the pyref section.)
        if let Some(last) = d.umt5_layer_outputs.last() {
            let rms = per_channel_rms(last, d.umt5_seq, UMT5_DIM);
            let mut idx: Vec<usize> = (0..UMT5_DIM).collect();
            idx.sort_by(|&a, &b| rms[b].total_cmp(&rms[a]));
            let top: Vec<usize> = idx.into_iter().take(6).collect();
            eprintln!("--- umT5 outlier channels (top-6 by final-block per-channel RMS) ---");
            eprintln!("top dims = {top:?}");
            for (i, lo) in d.umt5_layer_outputs.iter().enumerate() {
                let r = per_channel_rms(lo, d.umt5_seq, UMT5_DIM);
                let vals: Vec<String> = top.iter().map(|&c| format!("{:.1}", r[c])).collect();
                eprintln!("  block{i:02} ours RMS[top] = [{}]", vals.join(", "));
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

        // Three-way localization against the pyref step-0 dumps (when the
        // reference ran): umT5 output, raw DiT velocity, and post-scheduler
        // latent. A clean slope in `dit_out` with matching `umt5_hidden` puts the
        // bug inside the DiT body; a clean `dit_out` with a sloped `stepped`
        // puts it in the scheduler. linfit reports `ours ~= a*py + b`.
        if !skip_pyref {
            let cmp = |label: &str, got: &[f32], exp: &[f32]| {
                let n = got.len().min(exp.len());
                if n < 3 {
                    eprintln!("[diag-cmp {label}] SKIP (n={n})");
                    return;
                }
                let (a, b, rmse, cnt) = linfit(&exp[..n], &got[..n]);
                let (max_abs, _) = diff_stats(&got[..n], &exp[..n], f32::INFINITY);
                eprintln!(
                    "[diag-cmp {label}] ours ~= {a:.5}*py {b:+.5}  rmse={rmse:.3e} \
                     max_abs={max_abs:.3e} n={cnt} (got_len={} py_len={})",
                    got.len(),
                    exp.len(),
                );
            };
            let py_umt5 = read_f32(&tmp.join("py_umt5_out.bin"));
            cmp("umt5_hidden(vs padded out)", &d.umt5_hidden, &py_umt5);

            // Full per-umT5-layer linfit. Rust embeds/layer-outputs carry
            // seq_pad (even) rows; slice to the real `seq` rows so they align
            // with the pyref real-token rows (pyref runs the 512-padded
            // sequence, real tokens first). The first stage whose slope `a`
            // departs from ~1.0 is where the ~0.365x text scale is born.
            let seq_rows = d.umt5_seq * UMT5_DIM;
            let rust_rows = |v: &[f32]| v[..seq_rows.min(v.len())].to_vec();
            let cmp_umt5 = |label: &str, got: &[f32], file: &str| {
                let p = tmp.join(file);
                if p.exists() {
                    cmp(label, &rust_rows(got), &read_f32(&p));
                } else {
                    eprintln!("[diag-cmp {label}] SKIP (no {file})");
                }
            };
            cmp_umt5("umt5.embeds", &d.umt5_embeds, "py_umt5_embeds.bin");
            for (i, lo) in d.umt5_layer_outputs.iter().enumerate() {
                cmp_umt5(
                    &format!("umt5.block{i:02}_out"),
                    lo,
                    &format!("py_umt5_block{i}_out.bin"),
                );
            }
            // final_in == last block output (cross-check the readback); hidden
            // == post final_layer_norm. If every block sits at a~1.0 and the
            // slope only drops at `hidden`, the bug is the final RMSNorm.
            if let Some(last) = d.umt5_layer_outputs.last() {
                cmp_umt5("umt5.final_in", last, "py_umt5_final_in.bin");
            }
            cmp_umt5("umt5.hidden", &d.umt5_hidden, "py_umt5_hidden.bin");

            // --- outlier-channel ours-vs-py trace. For the dims that dominate
            // the final block (union of ours + py top), print per-channel RMS on
            // both sides for every block. The block where ours/py first diverges
            // in the dominant dim is exactly where (and in which channel) the
            // outlier is mis-computed. ---
            let py_block_out: Vec<Option<Vec<f32>>> = (0..d.umt5_layer_outputs.len())
                .map(|i| {
                    let p = tmp.join(format!("py_umt5_block{i}_out.bin"));
                    p.exists().then(|| read_f32(&p))
                })
                .collect();
            if let (Some(our_last), Some(Some(py_last))) =
                (d.umt5_layer_outputs.last(), py_block_out.last())
            {
                let top_of = |flat: &[f32]| {
                    let rms = per_channel_rms(flat, d.umt5_seq, UMT5_DIM);
                    let mut idx: Vec<usize> = (0..UMT5_DIM).collect();
                    idx.sort_by(|&a, &b| rms[b].total_cmp(&rms[a]));
                    idx.into_iter().take(4).collect::<Vec<_>>()
                };
                let mut top = top_of(our_last);
                for c in top_of(py_last) {
                    if !top.contains(&c) {
                        top.push(c);
                    }
                }
                eprintln!("--- umT5 outlier-channel ours-vs-py (dims {top:?}) ---");
                eprintln!("  (each cell ours_rms|py_rms for that channel)");
                for (i, lo) in d.umt5_layer_outputs.iter().enumerate() {
                    let Some(py) = py_block_out[i].as_ref() else {
                        continue;
                    };
                    let ro = per_channel_rms(&rust_rows(lo), d.umt5_seq, UMT5_DIM);
                    let rp = per_channel_rms(py, d.umt5_seq, UMT5_DIM);
                    let cells: Vec<String> = top
                        .iter()
                        .map(|&c| format!("{:.0}|{:.0}", ro[c], rp[c]))
                        .collect();
                    eprintln!("  block{i:02}: {}", cells.join("  "));
                }
            }

            // --- per-op slope table: for every block, linfit each shared op
            // (rust vs py) so the op that injects the divergence is named. ops:
            // n1 q k v o(=proj) n2 wi0 wi1 wo. The rust taps carry seq_pad rows
            // (op dim varies: 4096 for most, 10240 for wi0/wi1); slice to real
            // seq. py dumps are already seq-sliced. ---
            let seq_pad = d.umt5_seq + (d.umt5_seq & 1);
            let rust_op_rows = |got: &[f32]| -> Vec<f32> {
                if seq_pad == 0 {
                    return got.to_vec();
                }
                let op_dim = got.len() / seq_pad;
                got[..(d.umt5_seq * op_dim).min(got.len())].to_vec()
            };
            eprintln!("--- umT5 per-op slope table (ours ~= a*py, per block) ---");
            for (i, ops) in d.umt5_block_ops.iter().enumerate() {
                let fields: [(&str, &Vec<f32>); 9] = [
                    ("n1", &ops.n1),
                    ("q", &ops.q),
                    ("k", &ops.k),
                    ("v", &ops.v),
                    ("o", &ops.proj),
                    ("n2", &ops.n2),
                    ("wi0", &ops.wi0),
                    ("wi1", &ops.wi1),
                    ("wo", &ops.wo),
                ];
                let mut cols: Vec<String> = Vec::with_capacity(fields.len());
                for (op, got) in fields {
                    let p = tmp.join(format!("py_umt5_block{i}_{op}.bin"));
                    if !p.exists() {
                        cols.push(format!("{op}=NA"));
                        continue;
                    }
                    let py = read_f32(&p);
                    let g = rust_op_rows(got);
                    let n = g.len().min(py.len());
                    if n < 3 {
                        cols.push(format!("{op}=n{n}"));
                        continue;
                    }
                    let (a, _, _, _) = linfit(&py[..n], &g[..n]);
                    cols.push(format!("{op}={a:.3}"));
                }
                eprintln!("  block{i:02}: {}", cols.join(" "));
            }

            let py_dit0 = tmp.join("py_dit_out_step0.bin");
            if py_dit0.exists() {
                let py_v0 = read_f32(&py_dit0);
                cmp("dit_out(velocity)", &d.dit_out, &py_v0);
                // Scheduler isolation: run OUR UniPC step on the PY velocity (the
                // exact tensor diffusers handed its scheduler) starting from the
                // shared noise, then compare to py_step0_post. Same input both
                // sides, so any divergence is purely our scheduler math -- not the
                // DiT (whose velocity already matched py at slope ~1.0). A clean
                // slope here localizes the parity bug to UniPCScheduler::step.
                let mut sched = UniPCScheduler::new(steps_n as usize);
                let our_on_py = sched.step(&py_v0, &noise);
                cmp(
                    "sched(ours)@py_v0",
                    &our_on_py,
                    &read_f32(&tmp.join("py_step0_post.bin")),
                );
            } else {
                eprintln!("[diag-cmp dit_out(velocity)] SKIP (no py_dit_out_step0.bin)");
            }
            // DiT-body localization: post-patch token stream + projected text
            // (the two block-0 inputs), then each block's output residual. The
            // first stage whose slope departs from ~1.0 is where the velocity
            // divergence is born.
            let py_patch = tmp.join("py_patch_in.bin");
            if py_patch.exists() {
                cmp("patch_x", &d.patch_x, &read_f32(&py_patch));
            }
            let py_text = tmp.join("py_text_in.bin");
            if py_text.exists() {
                cmp("text_proj", &d.text_proj, &read_f32(&py_text));
            }
            for (i, b) in d.per_block.iter().enumerate() {
                let pb = tmp.join(format!("py_block{i}_out_step0.bin"));
                if !b.is_empty() && pb.exists() {
                    cmp(&format!("block{i:02}_out"), b, &read_f32(&pb));
                }
            }
            let py_temb = tmp.join("py_temb.bin");
            if py_temb.exists() {
                cmp("temb", &d.temb, &read_f32(&py_temb));
            }
            let py_tsp = tmp.join("py_timestep_proj.bin");
            if py_tsp.exists() {
                cmp("timestep_proj", &d.timestep_proj, &read_f32(&py_tsp));
            }
            let py_proj = tmp.join("py_proj_out.bin");
            if py_proj.exists() {
                cmp("proj_out", &d.proj_out, &read_f32(&py_proj));
            }
            cmp(
                "stepped(step0)",
                &d.stepped,
                &read_f32(&tmp.join("py_step0_post.bin")),
            );
        }
        return;
    }

    let t_full = std::time::Instant::now();
    tracing::info!(target: DIAG, t_ms = 0_u64, "milestone: starting denoise_with");
    let params = GenerationParams {
        prompt: PROMPT.to_string(),
        // Parity pins guidance_scale = 1.0 -> no CFG -> one DiT forward per step
        // (matches the pyref `guidance_scale=1.0`); keeps `velocity` == the raw
        // cond forward for the `py_dit_out_step{i}` compare.
        negative_prompt: String::new(),
        guidance_scale: 1.0,
        height,
        width,
        num_frames,
        steps: steps_n,
        seed: SEED,
        fps: FPS_BUCKET,
    };
    let mut ws = Workspace::new(Arc::clone(&backend), Arc::clone(model.arbiter()));
    let mut our_step_diag: Vec<WanStepDiag> = Vec::with_capacity(steps_n as usize);
    let (our_pre_vae, got_f_lat, got_h_lat, got_w_lat) = model
        .denoise_with(
            &params,
            Some(&noise),
            &mut ws,
            Some(&mut our_step_diag),
            None,
        )
        .await
        .expect("denoise_with");
    tracing::info!(
        target: DIAG,
        t_ms = t_full.elapsed().as_millis() as u64,
        pre_vae_len = our_pre_vae.len(),
        f_lat = got_f_lat,
        "milestone: denoise_with done",
    );
    assert_eq!(got_f_lat, f_lat);
    assert_eq!(got_h_lat, h_lat);
    assert_eq!(got_w_lat, w_lat);
    assert_eq!(our_step_diag.len(), steps_n as usize);
    assert_eq!(our_pre_vae.len(), n_lat);

    // VAE decode -> CTHW fp32 frames [3, out_frames, H, W] in [-1, 1].
    let our_frames = model
        .decode_latent_to_video(&our_pre_vae, f_lat, h_lat, w_lat, &mut ws)
        .await
        .expect("decode_latent_to_video");
    tracing::info!(
        target: DIAG,
        t_ms = t_full.elapsed().as_millis() as u64,
        frames_len = our_frames.len(),
        frames_expected = rgb_elems,
        "milestone: vae decode done",
    );
    assert_eq!(our_frames.len(), rgb_elems, "decoded frame element count");

    for (i, s) in our_step_diag.iter().enumerate() {
        summarize(&format!("our_step{i}_prev_sample"), &s.post);
    }
    summarize("our_pre_vae_latent", &our_pre_vae);
    summarize("our_frames (CTHW [-1,1])", &our_frames);

    // Stage frames before the budget assertion so a budget failure doesn't
    // suppress the visual output.
    if let Some(d) = png_dir.as_ref() {
        stage_frames(&our_frames, out_frames, height, width, d, "ours");
    }

    // Budget snapshot: emit unconditionally so a failing run still shows where
    // memory landed.
    let snap = backend.mem_account().snapshot();
    eprintln!(
        "[mem] vram TRUE_PEAK={} / budget {}",
        fmt_mib(snap.vram_total_peak),
        fmt_mib(budget.vram_bytes),
    );
    eprintln!(
        "[mem] ram  TRUE_PEAK={} / budget {}",
        fmt_mib(snap.ram_total_peak),
        fmt_mib(budget.ram_bytes),
    );

    // Budget asserts deferred until after the parity checks so a divergence
    // message always lands in the log before any budget failure.
    let assert_budgets = || {
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
    };

    if skip_pyref {
        eprintln!(
            "[safetensors] SKIP_PYREF: pipeline ran end-to-end; budget asserts enforced \
             (py-vs-engine divergence checks skipped)"
        );
        assert_budgets();
        return;
    }

    // --- py-vs-engine per-stage divergence checks ---
    let py_pre_vae = read_f32(&py_pre_vae_path);
    let py_vae_rgb = read_f32(&py_vae_rgb_path);
    let py_steps: Vec<Vec<f32>> = py_step_paths.iter().map(|p| read_f32(p)).collect();
    for (i, s) in py_steps.iter().enumerate() {
        summarize(&format!("py_step{i}_post"), s);
    }
    summarize("py_pre_vae_latent", &py_pre_vae);
    summarize("py_vae_rgb", &py_vae_rgb);

    assert_eq!(py_pre_vae.len(), n_lat, "py_pre_vae_latent length");
    assert_eq!(py_vae_rgb.len(), rgb_elems, "py_vae_rgb length");
    for (i, s) in py_steps.iter().enumerate() {
        assert_eq!(s.len(), n_lat, "py_step{i}_post length");
    }

    let trace_dump = std::env::var("THINFER_TRACE").is_ok();
    let mut diverged: Option<String> = None;
    let mut check = |label: &str, got: &[f32], expected: &[f32], max_n_over: usize| {
        let n = got.len().min(expected.len());
        let max_ref = expected[..n]
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0f32, f32::max);
        let tol = (max_ref * TOL_MULT).max(TOL_FLOOR);
        let (max_abs, n_over) = diff_stats(&got[..n], &expected[..n], tol);
        eprintln!(
            "[{label}] max_abs={max_abs:.4e} tol={tol:.4e} above_tol={n_over}/{n} \
             ref_max_abs={max_ref:.4e}"
        );
        if trace_dump {
            let (a, b, rmse, cnt) = linfit(expected, got);
            if cnt > 2 {
                eprintln!(
                    "[{label}] linfit  got ~= {a:.6}*exp + {b:+.6}  rmse={rmse:.4e} (n={cnt})"
                );
            }
        }
        if n_over > max_n_over && diverged.is_none() {
            diverged = Some(format!(
                "{label}: {n_over}/{n} cells over tol={tol:.4e} \
                 (max_n_over={max_n_over}, max_abs={max_abs:.4e}, ref_max_abs={max_ref:.4e})"
            ));
        }
    };

    // --- per-step bisection diagnostics (always printed; not gated). These split
    // a per-step divergence into DiT-velocity vs scheduler vs a specific DiT
    // block at EVERY step. linfit reports `ours ~= a*py + b`; slope `a` far from
    // 1.0 names the culprit stage. ---
    let cmp = |label: &str, got: &[f32], exp: &[f32]| {
        let n = got.len().min(exp.len());
        if n < 3 {
            eprintln!("[cmp {label}] SKIP (n={n})");
            return;
        }
        let (a, b, rmse, cnt) = linfit(&exp[..n], &got[..n]);
        let (max_abs, _) = diff_stats(&got[..n], &exp[..n], f32::INFINITY);
        eprintln!(
            "[cmp {label}] ours ~= {a:.5}*py {b:+.5}  rmse={rmse:.3e} max_abs={max_abs:.3e} n={cnt}"
        );
    };

    // 1. Raw DiT velocity per step vs the exact tensor diffusers handed its
    //    scheduler. A clean slope here puts the divergence past the DiT.
    let py_vels: Vec<Vec<f32>> = (0..steps_n as usize)
        .map(|i| read_f32(&tmp.join(format!("py_dit_out_step{i}.bin"))))
        .collect();
    eprintln!("--- DiT velocity per step (ours vs py_dit_out_step{{i}}) ---");
    for i in 0..steps_n as usize {
        cmp(
            &format!("velocity step{i}"),
            &our_step_diag[i].velocity,
            &py_vels[i],
        );
    }

    // 2. Scheduler isolation: feed the PY velocities through OUR UniPC scheduler
    //    from the shared noise. The DiT is fully factored out, so any divergence
    //    from py_step{i}_post is purely our scheduler math (multistep state +
    //    corrector). Stateful, so it must replay every step in order.
    eprintln!("--- scheduler isolation (our UniPC on py velocities vs py_step{{i}}_post) ---");
    {
        use thinfer_models::wan::scheduler::SchedulerStepDiag;
        let mut sched = UniPCScheduler::new(steps_n as usize);
        let mut s_sample = noise.clone();
        for i in 0..steps_n as usize {
            let mut d = SchedulerStepDiag::default();
            s_sample = sched.step_with_diag(&py_vels[i], &s_sample, &mut d);
            // Sub-stage split (py_sched_*_step{i}, present only when the pyref
            // scheduler tap ran): convert (m_conv) and corrector precede the
            // predictor, so a clean conv + clean corr with a divergent post pins
            // the bug on the predictor. The corr dump is absent on step 0.
            let conv_p = tmp.join(format!("py_sched_conv_step{i}.bin"));
            if conv_p.exists() {
                cmp(&format!("  conv step{i}"), &d.m_conv, &read_f32(&conv_p));
            }
            let corr_p = tmp.join(format!("py_sched_corr_step{i}.bin"));
            if corr_p.exists() {
                cmp(&format!("  corr step{i}"), &d.corrected, &read_f32(&corr_p));
            } else if d.used_corrector {
                eprintln!("[cmp   corr step{i}] py ran no corrector but ours did (order desync)");
            }
            cmp(
                &format!("sched(ours)@py_v[0..={i}]"),
                &s_sample,
                &py_steps[i],
            );
        }
    }

    // 3. Our scheduler internals per step (no py counterpart; informative): sigma,
    //    corrector order, and whether the corrector ran. The divergence first
    //    appears at the first step the corrector runs (step 1 at the pinned N=2).
    for (i, d) in our_step_diag.iter().enumerate() {
        eprintln!(
            "[sched step{i}] t={:.1} sigma={:.5} corrector_order={} used_corrector={}",
            d.timestep, d.sched.sigma, d.sched.this_order, d.sched.used_corrector
        );
    }

    // 4. Per-block residual slope per step: localize a velocity divergence to a
    //    block (the worklog suspects blocks 20-23 under-build at high residual
    //    magnitude in bf16). The first block whose slope departs from ~1.0 is
    //    where the step's divergence is born.
    for (i, sd) in our_step_diag.iter().enumerate() {
        let mut cols: Vec<String> = Vec::new();
        for (b, blk) in sd.per_block.iter().enumerate() {
            let p = tmp.join(format!("py_block{b}_out_step{i}.bin"));
            if blk.is_empty() || !p.exists() {
                continue;
            }
            let py = read_f32(&p);
            let n = blk.len().min(py.len());
            if n < 3 {
                continue;
            }
            let (a, _, _, _) = linfit(&py[..n], &blk[..n]);
            cols.push(format!("b{b:02}={a:.3}"));
        }
        eprintln!("--- DiT per-block slope step{i} (ours ~= a*py) ---");
        eprintln!("  {}", cols.join(" "));
    }

    for i in 0..steps_n as usize {
        check(
            &format!("step{i}_post"),
            &our_step_diag[i].post,
            &py_steps[i],
            CAP_STEP,
        );
    }
    // pre_vae_latent is the final denoised latent (pre prescale) on both sides;
    // it equals step{last}_post but is checked explicitly as the documented
    // VAE-input stage.
    check("pre_vae_latent", &our_pre_vae, &py_pre_vae, CAP_PRE_VAE);
    check(
        "vae_rgb (CTHW [-1,1])",
        &our_frames,
        &py_vae_rgb,
        CAP_VAE_RGB,
    );

    // Parity first so the divergence message precedes any budget failure.
    if let Some(msg) = diverged {
        panic!("FIRST DIVERGENCE: {msg}");
    }
    assert_budgets();
}

/// Drive `gen_video_e2e_parity_ref.py` via `uv run`. Writes `py_*.bin` stage
/// dumps into `out_dir` and, when `png_dir` is set, `py_frame{n}.png` +
/// `py_contact.png` beside the engine's `ours_*`.
#[allow(clippy::too_many_arguments)]
fn run_python_ref(
    noise_path: &Path,
    out_dir: &Path,
    png_dir: Option<&Path>,
    height: u32,
    width: u32,
    num_frames: u32,
    steps: u32,
) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    // THINFER_E2E_PYREF_DTYPE overrides the reference precision (bf16 default;
    // fp32 to establish the bf16-vs-fp32 floor when judging a bf16-vs-bf16 gap).
    let py_dtype = std::env::var("THINFER_E2E_PYREF_DTYPE").unwrap_or_else(|_| "bf16".to_string());
    let mut cmd = Command::new("uv");
    cmd.args([
        "run",
        "--directory",
        py_dir.to_str().unwrap(),
        "python",
        "-m",
        "thinfer_pytorch_ref.wan.gen_video_e2e_parity_ref",
        "--initial-noise",
        noise_path.to_str().unwrap(),
        "--out",
        out_dir.to_str().unwrap(),
        "--prompt",
        PROMPT,
        "--height",
        &height.to_string(),
        "--width",
        &width.to_string(),
        "--num-frames",
        &num_frames.to_string(),
        "--steps",
        &steps.to_string(),
        "--seed",
        &SEED.to_string(),
        "--fps",
        &REAL_FPS.to_string(),
        "--dtype",
        &py_dtype,
    ]);
    if let Some(d) = png_dir {
        cmd.args(["--png-dir", d.to_str().unwrap(), "--png-prefix", "py"]);
    }
    let status = cmd
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "pytorch video e2e-parity ref failed");
}

/// Read a little-endian f32 dump.
fn read_f32(p: &Path) -> Vec<f32> {
    let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Max abs diff + count of cells over `tol`. NaN/inf in either side always
/// counts as over (a silently-NaN run must fail, not pass).
fn diff_stats(got: &[f32], expected: &[f32], tol: f32) -> (f32, usize) {
    let mut max_abs = 0f32;
    let mut over = 0usize;
    for (g, e) in got.iter().zip(expected.iter()) {
        let nan_or_inf = !g.is_finite() || !e.is_finite();
        let d = (g - e).abs();
        if d.is_finite() && d > max_abs {
            max_abs = d;
        }
        if nan_or_inf || d > tol {
            over += 1;
        }
    }
    (max_abs, over)
}

/// Per-channel RMS of a row-major `[rows, dim]` tensor over its first `rows`
/// rows: `out[c] = sqrt(mean_r x[r,c]^2)`, length `dim`. Used to surface the
/// outlier hidden dimension(s) that dominate the residual stream (and thus the
/// final RMSNorm denominator). `rows` is derived from `flat.len()/dim` if the
/// slice is shorter than `rows*dim` (defensive against odd-pad slices).
fn per_channel_rms(flat: &[f32], rows: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; dim];
    if dim == 0 {
        return out;
    }
    let r = rows.min(flat.len() / dim);
    if r == 0 {
        return out;
    }
    for row in 0..r {
        let base = row * dim;
        for c in 0..dim {
            let v = flat[base + c];
            out[c] += v * v;
        }
    }
    for v in out.iter_mut() {
        *v = (*v / r as f32).sqrt();
    }
    out
}

/// Least-squares fit `y ~= a*x + b` over finite pairs; returns (a, b, rmse, n).
fn linfit(x: &[f32], y: &[f32]) -> (f64, f64, f64, usize) {
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut cnt = 0usize;
    let n = x.len().min(y.len());
    for i in 0..n {
        let (xi, yi) = (x[i] as f64, y[i] as f64);
        if xi.is_finite() && yi.is_finite() {
            sx += xi;
            sy += yi;
            sxx += xi * xi;
            sxy += xi * yi;
            cnt += 1;
        }
    }
    if cnt < 3 {
        return (0.0, 0.0, 0.0, cnt);
    }
    let nf = cnt as f64;
    let denom = nf * sxx - sx * sx;
    if denom.abs() < 1e-18 {
        return (0.0, 0.0, 0.0, cnt);
    }
    let slope = (nf * sxy - sx * sy) / denom;
    let bias = (sy - slope * sx) / nf;
    let mut resid_sq = 0.0f64;
    for i in 0..n {
        let (xi, yi) = (x[i] as f64, y[i] as f64);
        if xi.is_finite() && yi.is_finite() {
            let r = yi - (slope * xi + bias);
            resid_sq += r * r;
        }
    }
    (slope, bias, (resid_sq / nf).sqrt(), cnt)
}

/// Write a per-frame PNG sequence (`<prefix>_frame{n}.png`) plus a tiled
/// contact-sheet PNG (`<prefix>_contact.png`) from CTHW frames in `[-1, 1]`.
/// The contact sheet packs frames into a near-square grid so all frames are
/// inspectable in one image without a player/codec.
fn stage_frames(frames: &[f32], n: usize, h: u32, w: u32, dir: &Path, prefix: &str) {
    let (hh, ww) = (h as usize, w as usize);
    let per = hh * ww;
    // Per-frame: gather channel-planar [3, H, W] for frame f out of [3, N, H, W].
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

    // Contact sheet: near-square grid, black (-1.0) background.
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

/// SplitMix64 + Box-Muller. Identical to the zimage e2e so both tests load the
/// same starting tensor and the pyref byte-loads it verbatim.
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
