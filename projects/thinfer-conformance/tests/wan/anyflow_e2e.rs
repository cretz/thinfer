//! AnyFlow-Wan2.1-T2V-14B e2e gate: dual-timestep EMBEDDER PARITY (the port's
//! only new math vs the FastWan-parity-gated Wan machinery) + an engine-only
//! health run at tiny dims with the user-facing 2-step schedule.
//!
//! Parity scope rationale (low-RAM pyref policy): a full 14B CPU pyref is
//! infeasible (28GB bf16 resident; fp32 doubles it), so the pyref
//! (`gen_anyflow_embedder_ref.py`) recomputes ONLY the AnyFlow delta -- the
//! blended `rt_emb = 0.75*time_mlp(t) + 0.25*delta_mlp(r)` and its
//! `timestep_proj` -- in fp32 from the checkpoint's own `condition_embedder.*`
//! tensors (a few hundred MB). Blocks, umT5, and the Wan2.1 VAE are the same
//! code paths the FastWan/Wan22 gates cover. The engine taps `temb` /
//! `timestep_proj` at step 0 via `WanStepDiag`.
//!
//! Health: finiteness, latent variance, decoded-video variance + temporal
//! motion; optional PNG staging via `THINFER_E2E_PNG_DIR`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::trace;
use thinfer_core::workspace::Workspace;
use thinfer_models::wan::manifest::{self, role};
use thinfer_models::wan::pipeline::{
    GenerationParams, VaeChoice, VideoSampler, WanModel, WanStepDiag, WanVariant,
};
use thinfer_models::wan::scheduler::AnyFlowSampler;
use thinfer_models::wan::source::WanSource;
use thinfer_native::MmapFileOpener;
use thinfer_native::tokenizer::HfTokenizer;

use thinfer_native::cache;

const PROMPT: &str = "A bright red vintage car drives slowly down a sunlit \
coastal road, ocean waves in the background, realistic style, steady camera.";
const SEED: u64 = 42;
/// Wan2.1 VAE: 8x spatial, patch 2 -> /16 pixel grid; temporal 4x.
const VAE_SCALE: usize = 8;
const TEMPORAL_SCALE: usize = 4;
const Z_DIM: usize = 16;

fn env_u32(k: &str, d: u32) -> u32 {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn std_of(x: &[f32]) -> f64 {
    let n = x.len().max(1) as f64;
    let mean = x.iter().map(|v| *v as f64).sum::<f64>() / n;
    (x.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / n).sqrt()
}

/// Least-squares slope of got vs want + relative rmse (rmse / want max-abs).
fn slope_rel_rmse(got: &[f32], want: &[f32]) -> (f64, f64) {
    assert_eq!(got.len(), want.len());
    let n = got.len() as f64;
    let (mut sxy, mut sxx, mut serr2, mut wmax) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&g, &w) in got.iter().zip(want) {
        sxy += g as f64 * w as f64;
        sxx += (w as f64) * (w as f64);
        serr2 += ((g - w) as f64).powi(2);
        wmax = wmax.max((w as f64).abs());
    }
    (sxy / sxx.max(1e-30), (serr2 / n).sqrt() / wmax.max(1e-30))
}

/// Dumps the trace rollup (gpu_ms by pipeline + scope timings) on drop, pass
/// or panic -- the perf-localization table (same guard as `video_e2e`).
struct RollupDumpOnDrop(Option<trace::RollupHandle>);
impl Drop for RollupDumpOnDrop {
    fn drop(&mut self) {
        if let Some(h) = self.0.take() {
            let _ = h.dump(&mut std::io::stderr());
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn anyflow_e2e() {
    let _rollup = RollupDumpOnDrop(trace::init_from_env());

    // Tiny health grid (NOT a visual-quality regime; 832x480x81 is the product
    // one). 2 steps = the fast play this model exists for.
    let width = env_u32("THINFER_E2E_WIDTH", 256) as usize;
    let height = env_u32("THINFER_E2E_HEIGHT", 256) as usize;
    let num_frames = env_u32("THINFER_E2E_FRAMES", 9);
    let steps = env_u32("THINFER_E2E_STEPS", 2);
    let vram_gb = env_u32("THINFER_E2E_VRAM_GB", 5) as u64;
    let skip_pyref = std::env::var("THINFER_E2E_SKIP_PYREF").is_ok();
    // Opt-in tiny-decoder arm: load + decode with the taew2_1 tiny VAE
    // (Wan2.1 z16) instead of the full Wan2.1 VAE. Health-gates the AnyFlow
    // tiny path end to end; the full VAE stays the parity default.
    let tiny_vae = std::env::var("THINFER_E2E_TINY_VAE").is_ok();

    let mut roles = vec![
        role::DIT_ANYFLOW_1,
        role::DIT_ANYFLOW_2,
        role::DIT_ANYFLOW_3,
        role::TEXT_ENCODER_SHARD_1,
        role::TEXT_ENCODER_SHARD_2,
        role::TEXT_ENCODER_SHARD_3,
        role::VAE_WAN21,
    ];
    if tiny_vae {
        roles.push(role::TINY_VAE_WAN21);
    }
    let mut resolved: Vec<PathBuf> = Vec::new();
    for r in roles.iter().chain([role::TOKENIZER_JSON].iter()) {
        let fr = manifest::MANIFEST.get(r).expect("role in manifest");
        match cache::resolve(fr) {
            Some(p) => resolved.push(p),
            None => {
                eprintln!(
                    "skipped[anyflow_e2e]: {}/{} not in HF cache",
                    fr.repo, fr.path
                );
                return;
            }
        }
    }
    let tok_path = resolved.pop().unwrap();
    eprintln!("anyflow-e2e: {width}x{height} f{num_frames} steps={steps} (roles resolved)");

    let mut openers: Vec<MmapFileOpener> = Vec::with_capacity(resolved.len());
    for p in &resolved {
        openers.push(
            MmapFileOpener::new(p)
                .await
                .unwrap_or_else(|e| panic!("open {}: {e}", p.display())),
        );
    }
    let source = WanSource::open(openers, None).await.expect("parse weights");

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
    let tokenizer = HfTokenizer::from_path(&tok_path).await.expect("tokenizer");

    // 20GB RAM: the Q8_0 DiT transcode (~15GB host) + streamed umT5. VRAM stays
    // at the thin-hardware default.
    let residency = WeightResidency::new(
        source,
        ResidencyBudget {
            ram_bytes: 20 << 30,
            vram_bytes: vram_gb << 30,
        },
    );
    let vae_choice = if tiny_vae {
        VaeChoice::Tiny
    } else {
        VaeChoice::Full
    };
    let model = WanModel::load_variant(
        Arc::clone(&backend),
        residency,
        tokenizer,
        vae_choice,
        WanVariant::anyflow_t2v_14b(),
        None,
        false,
    )
    .await
    .expect("WanModel::load_variant(anyflow)");

    // `THINFER_E2E_ATTN_WINDOW=W` opts into the temporal window (0/unset =
    // full attention); combine with `THINFER_WAN_WINDOW_FROM_STEP` for the
    // hybrid step-windowing eyeball.
    let attn_window = Some(env_u32("THINFER_E2E_ATTN_WINDOW", 0)).filter(|w| *w > 0);
    let params = GenerationParams {
        prompt: PROMPT.to_string(),
        height: height as u32,
        width: width as u32,
        num_frames,
        seed: SEED,
        sampler: VideoSampler::default(),
        attn_window,
        steps: Some(steps),
    };

    // --- denoise with step taps ---
    let mut ws = Workspace::new(Arc::clone(&backend), Arc::clone(model.arbiter()));
    let mut step_diag: Vec<WanStepDiag> = Vec::new();
    let t0 = std::time::Instant::now();
    let (latent, f_lat, h_lat, w_lat) = model
        .denoise_with(&params, None, &mut ws, Some(&mut step_diag), None, None)
        .await
        .expect("denoise");
    eprintln!(
        "anyflow denoise: {} steps, latent std={:.4}, {:.1}s",
        step_diag.len(),
        std_of(&latent),
        t0.elapsed().as_secs_f32()
    );
    assert_eq!(step_diag.len(), steps as usize, "step count");
    assert!(latent.iter().all(|v| v.is_finite()), "latent finite");
    assert!(std_of(&latent) > 0.05, "latent degenerate");

    // --- embedder parity at step 0 (t = 1000, r = shifted next sigma) ---
    if !skip_pyref {
        let sampler = AnyFlowSampler::new(steps as usize);
        let (t, r) = (sampler.timestep(0), sampler.r_timestep(0));
        let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("anyflow_e2e");
        std::fs::create_dir_all(&tmp).expect("tmpdir");
        run_embedder_ref(&resolved[0..3], t, r, &tmp);
        let want_temb = read_f32(&tmp.join("temb.bin"));
        let want_proj = read_f32(&tmp.join("timestep_proj.bin"));
        let d = &step_diag[0];
        for (label, got, want) in [
            ("temb (rt_emb blend)", &d.temb, &want_temb),
            ("timestep_proj", &d.timestep_proj, &want_proj),
        ] {
            assert_eq!(got.len(), want.len(), "{label} length");
            let (slope, rel_rmse) = slope_rel_rmse(got, want);
            eprintln!("parity {label}: slope={slope:.5} rel_rmse={rel_rmse:.5}");
            // bf16 act floor: the engine runs the embedder MLPs in bf16 acts
            // against the fp32 ref (rel ~4e-3/step); the band leaves margin.
            assert!((slope - 1.0).abs() < 0.02, "{label} slope {slope}");
            assert!(rel_rmse < 0.02, "{label} rel_rmse {rel_rmse}");
        }
    }

    // --- VAE decode + health ---
    let t0 = std::time::Instant::now();
    let video = model
        .decode_latent_to_video(&latent, f_lat, h_lat, w_lat, vae_choice, &mut ws)
        .await
        .expect("vae decode");
    eprintln!("anyflow vae decode: {:.1}s", t0.elapsed().as_secs_f32());
    let out_frames = TEMPORAL_SCALE * f_lat - 3;
    assert_eq!(video.len(), 3 * out_frames * height * width, "video size");
    assert!(video.iter().all(|v| v.is_finite()), "video finite");
    let vstd = std_of(&video);
    assert!(vstd > 0.05, "video degenerate (std {vstd})");
    // Temporal motion: last frame differs from first.
    let frame_len = height * width;
    let per_c = out_frames * frame_len;
    let mut diff = 0.0f64;
    for c in 0..3 {
        for p in 0..frame_len {
            let a = video[c * per_c + p];
            let b = video[c * per_c + (out_frames - 1) * frame_len + p];
            diff += ((a - b) as f64).abs();
        }
    }
    let mad = diff / (3 * frame_len) as f64;
    eprintln!("anyflow health: video std={vstd:.3} first-vs-last MAD={mad:.4}");
    assert!(mad > 1e-3, "no temporal motion (MAD {mad})");

    if let Some(dir) = std::env::var_os("THINFER_E2E_PNG_DIR").map(PathBuf::from) {
        std::fs::create_dir_all(&dir).expect("png dir");
        for f in 0..out_frames {
            let mut chw = vec![0.0f32; frame_len * 3];
            for p in 0..frame_len {
                for c in 0..3 {
                    chw[c * frame_len + p] = video[c * per_c + f * frame_len + p];
                }
            }
            let png =
                thinfer_models::z_image::pipeline::encode_png(&chw, width as u32, height as u32)
                    .expect("encode png");
            std::fs::write(dir.join(format!("anyflow_{f:03}.png")), png).expect("write png");
        }
        eprintln!("staged {out_frames} frames");
    }
    let _ = (Z_DIM, VAE_SCALE);
}

fn run_embedder_ref(shards: &[PathBuf], t: f32, r: f32, out: &Path) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let mut cmd = Command::new("uv");
    cmd.args([
        "run",
        "--directory",
        py_dir.to_str().unwrap(),
        "python",
        "-m",
        "thinfer_pytorch_ref.wan.gen_anyflow_embedder_ref",
    ]);
    for s in shards {
        cmd.args(["--shard", s.to_str().unwrap()]);
    }
    let status = cmd
        .args([
            "--t",
            &t.to_string(),
            "--r",
            &r.to_string(),
            "--out",
            out.to_str().unwrap(),
        ])
        .status()
        .expect("run embedder pyref (uv)");
    assert!(status.success(), "embedder pyref failed");
}

fn read_f32(p: &Path) -> Vec<f32> {
    let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
