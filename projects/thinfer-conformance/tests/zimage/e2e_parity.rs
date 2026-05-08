//! Forever end-to-end Z-Image parity test. Drives the engine through the
//! full path - tokenize -> Qwen3 encode -> N denoising steps -> VAE decode
//! -> RGB CHW fp32 - against a pinned PyTorch reference for the same
//! prompt, dims, seed, and initial noise. Both sides byte-load the same
//! noise so divergence is attributable to engine math (Qwen3 / DiT /
//! scheduler / VAE), never to RNG drift.
//!
//! Unlike `dit_parity` (which has internal DiT/Qwen3 taps and is intended
//! to be RETIRED once the gray-PNG bug is fixed), this test only compares
//! externally-observable stages:
//!
//!   1. starting_latents (sanity: byte-identical input)
//!   2. per-step prev_sample (post-scheduler-step latent, one per step)
//!   3. pre_vae_latent (== last step's prev_sample, fed to VAE)
//!   4. vae_rgb (CHW fp32 RGB in [-1, 1], output of vae.decode)
//!
//! If everything matches, end-to-end byte parity is established. If a
//! stage diverges, the narrower bisection tests (qwen3_parity / dit_parity)
//! localize it. This test stays around forever; the others are temporary.
//!
//! Run with: `cargo test -p thinfer-conformance --features zimage-e2e
//!           --release e2e_parity_matches_pytorch -- --nocapture`

#![cfg(feature = "zimage-e2e")]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::safetensors::ShardedSafetensorsSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::trace::{self, DIAG};
use thinfer_core::workspace::Workspace;
use thinfer_models::z_image::manifest::{self, role};
use thinfer_models::z_image::pipeline::{GenerationParams, ZImageModel, encode_png};
use thinfer_models::z_image::vae::VaeStageSample;
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;
use thinfer_native::tokenizer::HfTokenizer;

/// Pinned config. Matches dit_parity so the same prompt/noise/seed flows
/// through both. Small enough to keep the test under a few minutes; large
/// enough to exercise the full engine including VAE at production dims.
const PROMPT: &str = "a red apple on a wooden table";
const HEIGHT: u32 = 256;
const WIDTH: u32 = 256;
const STEPS: u32 = 2;
const SEED: u64 = 42;

const LATENT_CHANNELS: usize = 16;
const VAE_SCALE: usize = 8;
/// Matches `thinfer_models::z_image::vae::config::{SCALING,SHIFT}_FACTOR`.
/// Used to scale our pre-VAE latent dump into the same space py captures
/// (py's vae.decode hook fires AFTER diffusers does `z/SCALING + SHIFT`;
/// our `vae.decode` does that same transform internally, so our dump is
/// pre-scaling). Comparing the two without this shift always shows a
/// spurious 1/SCALING_FACTOR ratio drift.
const VAE_SCALING_FACTOR: f32 = 0.3611;
const VAE_SHIFT_FACTOR: f32 = 0.1159;

// Per-tap tolerance = max(TOL_MULT * |expected|, TOL_FLOOR). Same 3%/1e-3
// applied everywhere. DiT/pre-VAE stages stay at 0 cells over (absorbs bf16
// accumulator-order noise on step0 prev_sample). vae_rgb keeps the same
// per-tap tol but allows up to VAE_RGB_MAX_OVER cells past it: bf16 drift
// through VAE up_blocks produces a small outlier population that is
// sub-visible (<0.3% of pixels); tightening this should follow bf16
// activations work for the whole DiT, not vae_rgb-specific tuning.
const TOL_MULT: f32 = 0.03;
const TOL_FLOOR: f32 = 1e-3;
const VAE_RGB_MAX_OVER: usize = 500;
const VAE_DIAG_MAX_OVER: usize = 64;

/// Drops `RollupHandle::dump` to stderr on test exit (success or panic).
/// Without this the rollup table is lost when divergence triggers `panic!`
/// before any tail logging runs.
struct RollupDumpOnDrop(Option<thinfer_core::trace::RollupHandle>);
impl Drop for RollupDumpOnDrop {
    fn drop(&mut self) {
        if let Some(h) = self.0.take() {
            let _ = h.dump(&mut std::io::stderr());
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn e2e_parity_matches_pytorch() {
    let _rollup = RollupDumpOnDrop(trace::init_from_env());
    let h_lat = (HEIGHT as usize) / VAE_SCALE;
    let w_lat = (WIDTH as usize) / VAE_SCALE;
    let n_lat = LATENT_CHANNELS * h_lat * w_lat;
    let img_h = HEIGHT as usize;
    let img_w = WIDTH as usize;
    let rgb_elems = 3 * img_h * img_w;

    // Resolve every weight role the engine needs. Skip cleanly if any
    // miss - matches dit_parity's discipline so this test never spuriously
    // fails on a machine without the HF cache populated.
    let needed = [
        role::DIT_SHARD_1,
        role::DIT_SHARD_2,
        role::TEXT_ENCODER_SHARD_1,
        role::TEXT_ENCODER_SHARD_2,
        role::TEXT_ENCODER_SHARD_3,
        role::VAE,
        role::TOKENIZER_JSON,
    ];
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
    let path_of = |role_name: &str| -> &std::path::Path {
        &resolved.iter().find(|(r, _)| *r == role_name).unwrap().1
    };
    eprintln!("e2e-parity: all roles resolved from HF cache");

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let noise_path = tmp.join("e2e_parity_noise.bin");

    // Per-stage py dump paths. Sibling of `--out` dir, named to avoid
    // collision with dit_parity's dumps in the same tmpdir.
    let py_starting_path = tmp.join("py_starting_latents.bin");
    let py_pre_vae_path = tmp.join("py_pre_vae_latent.bin");
    let py_vae_rgb_path = tmp.join("py_vae_rgb.bin");
    let py_step_path = |i: usize| tmp.join(format!("py_step{i}_post.bin"));

    // Clear stale dumps so a stale file can't mask a hook that never fires.
    let mut clear_paths = vec![
        py_starting_path.clone(),
        py_pre_vae_path.clone(),
        py_vae_rgb_path.clone(),
    ];
    for i in 0..STEPS as usize {
        clear_paths.push(py_step_path(i));
    }
    for p in &clear_paths {
        let _ = std::fs::remove_file(p);
    }

    // Deterministic pinned noise. Identical to dit_parity's RNG (Box-Muller
    // over SplitMix64) so both tests exercise the same starting tensor.
    let noise = make_pinned_noise(n_lat);
    summarize("noise (pinned)", &noise);
    std::fs::write(&noise_path, bytemuck_cast(&noise)).expect("write noise");

    // Opt-in PNG dumps. Set THINFER_E2E_PNG_DIR=/some/path to get
    // <dir>/ours.png and <dir>/py.png (both from raw VAE output with the
    // same (v+1)*127.5 transform) for eyeballing.
    let png_dir = std::env::var_os("THINFER_E2E_PNG_DIR").map(PathBuf::from);
    if let Some(d) = png_dir.as_ref() {
        std::fs::create_dir_all(d).expect("create THINFER_E2E_PNG_DIR");
        eprintln!("png dump enabled: {}", d.display());
    }

    // VAE per-stage diag is always on: ours-side dumps a bounded head
    // sample (<=256 fp32) per `decoder_back` stage, py-side hooks every
    // `pipe.vae.decoder` submodule and dumps the matching sample. Total
    // payload is ~20 KB per side - small enough to be permanent. Bounded
    // per `[[feedback-vae-diag-hazard]]`.
    let vae_diag_dir = tmp.join("vae_diag");
    std::fs::create_dir_all(&vae_diag_dir).expect("create vae_diag dir");
    if let Ok(rd) = std::fs::read_dir(&vae_diag_dir) {
        for ent in rd.flatten() {
            let _ = std::fs::remove_file(ent.path());
        }
    }
    eprintln!("vae diag: {}", vae_diag_dir.display());

    // ---- Drive PyTorch reference. Captures all py-side dumps under tmp/. ----
    let dit_shards = [
        path_of(role::DIT_SHARD_1).to_owned(),
        path_of(role::DIT_SHARD_2).to_owned(),
    ];
    run_python_ref(
        &noise_path,
        &dit_shards,
        &tmp,
        png_dir.as_deref(),
        &vae_diag_dir,
    );

    // ---- Build our side. Same loader path as thinfer-cli::cmd::generate. ----
    let weight_roles = [
        role::DIT_SHARD_1,
        role::DIT_SHARD_2,
        role::TEXT_ENCODER_SHARD_1,
        role::TEXT_ENCODER_SHARD_2,
        role::TEXT_ENCODER_SHARD_3,
        role::VAE,
    ];
    let mut openers: Vec<MmapFileOpener> = Vec::with_capacity(weight_roles.len());
    for r in weight_roles {
        let path = path_of(r);
        openers.push(
            MmapFileOpener::new(path)
                .await
                .unwrap_or_else(|e| panic!("open {}: {e}", path.display())),
        );
    }
    let source = ShardedSafetensorsSource::open(openers)
        .await
        .expect("parse sharded safetensors");
    // VRAM sized to hold the full DiT working set in residency so the perf
    // trace measures kernel/host overhead rather than per-block weight
    // re-upload. Pair with `THINFER_POWER_PREF=low` to target the iGPU
    // (shared system RAM) on hybrid laptops; on a discrete 8 GiB card this
    // budget will trip the OOM noted in the worklog Backlog re: workspace
    // + slabs not counting against ResidencyBudget.
    let budget = ResidencyBudget {
        ram_bytes: 10 << 30,
        vram_bytes: 16 << 30,
    };
    let residency = WeightResidency::new(source, budget);
    let cfg = WgpuConfig {
        power_preference: match std::env::var("THINFER_POWER_PREF")
            .ok()
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("high" | "highperformance" | "discrete") => PowerPreference::HighPerformance,
            Some("low" | "lowpower" | "integrated") => PowerPreference::LowPower,
            _ => PowerPreference::None,
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
    let model = ZImageModel::load(Arc::clone(&backend), residency, tokenizer)
        .await
        .expect("ZImageModel::load");

    // ---- Drive our side. denoise_with(step_dumps=Some) captures every
    //      post-scheduler-step latent; final return is the pre-VAE latent. ----
    let t_full = std::time::Instant::now();
    tracing::info!(target: DIAG, t_ms = 0_u64, "milestone: starting denoise_with");
    let params = GenerationParams {
        prompt: PROMPT.to_string(),
        height: HEIGHT,
        width: WIDTH,
        steps: STEPS,
        seed: SEED,
    };
    let mut ws = Workspace::new(Arc::clone(&backend));
    let mut our_step_dumps: Vec<Vec<f32>> = Vec::with_capacity(STEPS as usize);
    let (our_pre_vae, our_h_lat, our_w_lat) = model
        .denoise_with(&params, Some(&noise), &mut ws, Some(&mut our_step_dumps))
        .await
        .expect("denoise_with");
    tracing::info!(
        target: DIAG,
        t_ms = t_full.elapsed().as_millis() as u64,
        pre_vae_len = our_pre_vae.len(),
        h_lat = our_h_lat,
        w_lat = our_w_lat,
        step_dumps = our_step_dumps.len(),
        "milestone: denoise_with done",
    );
    assert_eq!(our_h_lat, h_lat);
    assert_eq!(our_w_lat, w_lat);
    assert_eq!(our_step_dumps.len(), STEPS as usize);

    // VAE decode -> CHW fp32 RGB in [-1, 1], capturing per-stage head
    // samples for compare against py's per-submodule hooks.
    let mut our_vae_diag: Vec<VaeStageSample> = Vec::new();
    let our_rgb = model
        .decode_latents_to_rgb_with_diag(&our_pre_vae, h_lat, w_lat, &mut ws, &mut our_vae_diag)
        .await
        .expect("decode_latents_to_rgb_with_diag");
    for st in &our_vae_diag {
        let p = vae_diag_dir.join(format!("our_{}.bin", st.label));
        std::fs::write(&p, bytemuck_cast(&st.head)).expect("write our vae diag stage");
    }
    eprintln!(
        "[vae-diag] captured {} stages from our VAE; dumped to {}",
        our_vae_diag.len(),
        vae_diag_dir.display()
    );
    tracing::info!(
        target: DIAG,
        t_ms = t_full.elapsed().as_millis() as u64,
        rgb_len = our_rgb.len(),
        rgb_expected = rgb_elems,
        "milestone: vae decode done",
    );
    assert_eq!(our_rgb.len(), rgb_elems);

    if let Some(d) = png_dir.as_ref() {
        match encode_png(&our_rgb, WIDTH, HEIGHT) {
            Ok(png) => {
                let p = d.join("ours.png");
                std::fs::write(&p, &png).expect("write ours.png");
                eprintln!("wrote {}", p.display());
            }
            Err(e) => eprintln!("encode_png failed: {e}"),
        }
    }

    // ---- Read py dumps. ----
    let py_starting = read_f32(&py_starting_path);
    let py_pre_vae = read_f32(&py_pre_vae_path);
    let py_vae_rgb = read_f32(&py_vae_rgb_path);
    let py_steps: Vec<Vec<f32>> = (0..STEPS as usize)
        .map(|i| read_f32(&py_step_path(i)))
        .collect();

    summarize("py_starting_latents", &py_starting);
    for (i, s) in py_steps.iter().enumerate() {
        summarize(&format!("py_step{i}_prev_sample"), s);
    }
    summarize("py_pre_vae_latent", &py_pre_vae);
    summarize("py_vae_rgb", &py_vae_rgb);

    for (i, s) in our_step_dumps.iter().enumerate() {
        summarize(&format!("our_step{i}_prev_sample"), s);
    }
    summarize("our_pre_vae_latent", &our_pre_vae);
    summarize("our_vae_rgb", &our_rgb);

    // ---- Compare stages. Per-tap tolerance = max(|expected|)*2% with a
    //      1e-3 floor: matches dit_parity. Anything above is a real engine
    //      divergence, not bf16/group_norm reduction noise. Records first
    //      divergence and panics at the end so we see every stage every run. ----
    let mut diverged: Option<String> = None;
    let mut check = |label: &str,
                     got: &[f32],
                     expected: &[f32],
                     tol_mult: f32,
                     tol_floor: f32,
                     max_n_over: usize| {
        let n = got.len().min(expected.len());
        let max_ref = expected[..n]
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0f32, f32::max);
        let tol = (max_ref * tol_mult).max(tol_floor);
        let (max_abs, n_over) = diff_stats(&got[..n], &expected[..n], tol);
        eprintln!(
            "[{label}] max_abs={max_abs:.4e} tol={tol:.4e} above_tol={n_over}/{n} \
             ref_max_abs={max_ref:.4e}"
        );
        if n_over > max_n_over && diverged.is_none() {
            diverged = Some(format!(
                "{label}: {n_over}/{n} cells over tol={tol:.4e} \
                 (max_n_over={max_n_over}, max_abs={max_abs:.4e}, ref_max_abs={max_ref:.4e})"
            ));
        }
    };

    // Data-flow order: starting noise -> per-step latents -> pre-VAE -> RGB.
    // Length-mismatched stages would mask real divergence, so assert shape
    // separately before content.
    assert_eq!(py_starting.len(), n_lat, "py_starting_latents length");
    assert_eq!(py_pre_vae.len(), n_lat, "py_pre_vae_latent length");
    assert_eq!(py_vae_rgb.len(), rgb_elems, "py_vae_rgb length");
    for (i, s) in py_steps.iter().enumerate() {
        assert_eq!(s.len(), n_lat, "py_step{i}_prev_sample length");
    }

    check(
        "starting_latents (== injected noise)",
        &noise,
        &py_starting,
        TOL_MULT,
        TOL_FLOOR,
        0,
    );
    for i in 0..STEPS as usize {
        check(
            &format!("step{i}.prev_sample"),
            &our_step_dumps[i],
            &py_steps[i],
            TOL_MULT,
            TOL_FLOOR,
            0,
        );
    }
    // Apply diffusers' `z/SCALING + SHIFT` transform so our pre-VAE dump
    // is in the same space as py's vae.decode-hook capture (see
    // VAE_{SCALING,SHIFT}_FACTOR comment). Without this both sides differ
    // by a constant 1/SCALING ~2.77x ratio - a wiring artifact, not a bug.
    let our_pre_vae_scaled: Vec<f32> = our_pre_vae
        .iter()
        .map(|z| z / VAE_SCALING_FACTOR + VAE_SHIFT_FACTOR)
        .collect();
    summarize(
        "our_pre_vae_latent (post-scale, vae.decode-input space)",
        &our_pre_vae_scaled,
    );
    check(
        "pre_vae_latent (post-scale, ==what VAE math sees)",
        &our_pre_vae_scaled,
        &py_pre_vae,
        TOL_MULT,
        TOL_FLOOR,
        0,
    );
    check(
        "vae_rgb (CHW fp32 [-1, 1])",
        &our_rgb,
        &py_vae_rgb,
        TOL_MULT,
        TOL_FLOOR,
        VAE_RGB_MAX_OVER,
    );

    // ---- VAE per-stage diag compare. Stage names that our side produces
    //      but py doesn't hook (e.g. "up{i}.upsample", "silu_out") are
    //      summarized-only - the ours head sample is still printed so we
    //      can see "this stage went to zero". ----
    eprintln!("---- vae per-stage diag ----");
    for st in &our_vae_diag {
        summarize(&format!("our_vae.{}", st.label), &st.head);
    }
    for st in &our_vae_diag {
        let py_path = vae_diag_dir.join(format!("py_{}.bin", st.label));
        if !py_path.exists() {
            eprintln!(
                "[vae.{}] (py side did not dump - ours-only stage, see summary above)",
                st.label
            );
            continue;
        }
        let py_head = read_f32(&py_path);
        summarize(&format!("py_vae.{}", st.label), &py_head);
        check(
            &format!("vae.{}", st.label),
            &st.head,
            &py_head,
            TOL_MULT,
            TOL_FLOOR,
            VAE_DIAG_MAX_OVER,
        );
    }

    if let Some(msg) = diverged {
        panic!("FIRST DIVERGENCE: {msg}");
    }
}

fn run_python_ref(
    noise_path: &std::path::Path,
    dit_shards: &[PathBuf],
    out_dir: &std::path::Path,
    png_dir: Option<&std::path::Path>,
    vae_diag_dir: &std::path::Path,
) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let mut cmd = Command::new("uv");
    cmd.args([
        "run",
        "--directory",
        py_dir.to_str().unwrap(),
        "python",
        "-m",
        "thinfer_pytorch_ref.zimage.gen_e2e_parity_ref",
        "--initial-noise",
        noise_path.to_str().unwrap(),
        "--out",
        out_dir.to_str().unwrap(),
        "--prompt",
        PROMPT,
        "--height",
        &HEIGHT.to_string(),
        "--width",
        &WIDTH.to_string(),
        "--steps",
        &STEPS.to_string(),
        "--seed",
        &SEED.to_string(),
        "--dtype",
        "bf16",
    ]);
    for shard in dit_shards {
        cmd.args(["--transformer-shard", shard.to_str().unwrap()]);
    }
    if let Some(d) = png_dir {
        cmd.args(["--png-dir", d.to_str().unwrap()]);
    }
    cmd.args(["--vae-diag-dir", vae_diag_dir.to_str().unwrap()]);
    let status = cmd
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "pytorch e2e-parity ref failed");
}

/// SplitMix64 + Box-Muller. Identical to dit_parity::make_pinned_noise so
/// both tests load the same starting tensor; not factored out because each
/// test owns its pin and we don't want a shared crate just for this.
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
    let mean = sum / denom;
    eprintln!(
        "[{label}] len={} nan={} min={:.4e} max={:.4e} max_abs={:.4e} mean={:.4e}",
        v.len(),
        nan,
        min,
        max,
        max_abs,
        mean
    );
}

fn read_f32(p: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn diff_stats(got: &[f32], expected: &[f32], tol: f32) -> (f32, usize) {
    let mut max_abs = 0f32;
    let mut over = 0usize;
    for (g, e) in got.iter().zip(expected.iter()) {
        let d = (g - e).abs();
        if d > max_abs {
            max_abs = d;
        }
        if d > tol {
            over += 1;
        }
    }
    (max_abs, over)
}
