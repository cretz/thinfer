//! DreamID-V-Wan-1.3B-Faster health e2e: the whole video face-swap pipeline on
//! the REAL converted DiT weights + the Wan2.1 VAE + the DreamID-V repo test-case
//! assets, at TINY dims. It is NOT the torch parity gate (the user drives that
//! separately); it proves the pipeline runs end to end and produces STRUCTURED
//! output - finite AND spatially non-constant (std > 0) at every checkpoint (the
//! three VAE-encode latents, the pre-VAE denoised latent, the final RGB), so the
//! all-NaN-laundered-to-finite / dead-constant false green cannot pass.
//!
//! Inputs (skips cleanly, never downloads, when any is absent):
//!   - DiT `dreamidv_faster.pth` + baked `context.pth`: the `DIT_DREAMIDV` /
//!     `CONTEXT_DREAMIDV` manifest roles, read directly from the HF cache via
//!     `PytorchSource` (same path as the production executor).
//!   - Wan2.1 VAE: the `VAE_WAN21` manifest role, from the HF cache.
//!   - Assets: the DreamID-V clone's `assets/test_case` (ref image `an_1.jpg`,
//!     target video `a_girl.mp4`, precomputed mask `a_girl_mask.mp4`).
//!     `THINFER_DREAMIDV_ASSETS` overrides the default third-party path.
//!   - `ffmpeg` on PATH (decodes the jpg/mp4 to PNG frames the test reads).
//!
//! Run:
//!   `cargo test -p thinfer-conformance --features dreamidv-e2e --release \
//!    dreamidv_health -- --nocapture --test-threads=1`
//!
//! Knobs: `THINFER_DREAMIDV_{FRAMES,AREA,STEPS}` (defaults 5, 128*128, 2);
//! `THINFER_DREAMIDV_BUDGET_GB` (default 8). Output PNGs -> `THINFER_PNG_DIR` (or
//! the scratch default).

#![cfg(feature = "dreamidv-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_models::wan::dreamidv::{DreamIdvDiag, DreamIdvInputs, DreamIdvPipeline, RgbFrames};
use thinfer_models::wan::manifest::{self, role};
use thinfer_models::wan::source::open_dreamidv_source;
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

const NUM_LAYERS: usize = 30;
const SEED: u64 = 42;

/// Dumps the `RollupHandle` (per-scope + gpu_ms timing table) to stderr on drop,
/// so a `THINFER_TRACE=1` run surfaces where the pipeline spends its time.
struct RollupDumpOnDrop(Option<thinfer_core::trace::RollupHandle>);
impl Drop for RollupDumpOnDrop {
    fn drop(&mut self) {
        if let Some(h) = self.0.take() {
            let _ = h.dump(&mut std::io::stderr());
        }
    }
}

fn env_str(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}
fn env_usize(key: &str, d: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

/// Decode `n` frames of `input` (video or image) into `dir/f%03d.png` via the
/// ffmpeg CLI. Returns false when ffmpeg is unavailable or fails.
fn ffmpeg_decode(input: &Path, n: usize, dir: &Path) -> bool {
    let pat = dir.join("f%03d.png");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(input)
        .args(["-frames:v", &n.to_string()])
        .arg(&pat)
        .status();
    matches!(status, Ok(s) if s.success())
}

/// Read `n` PNG frames `dir/f001.png..` into a flat interleaved RGB `u8` buffer
/// `[n, h, w, 3]`, asserting all frames share dims. Returns `(data, h, w)`.
fn read_frames(dir: &Path, n: usize) -> (Vec<u8>, usize, usize) {
    let mut data = Vec::new();
    let (mut fh, mut fw) = (0usize, 0usize);
    for i in 1..=n {
        let path = dir.join(format!("f{i:03}.png"));
        let decoder = png::Decoder::new(std::fs::File::open(&path).expect("open frame png"));
        let mut reader = decoder.read_info().expect("png header");
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).expect("png frame");
        let (w, h) = (info.width as usize, info.height as usize);
        let rgb: Vec<u8> = match info.color_type {
            png::ColorType::Rgb => buf[..w * h * 3].to_vec(),
            png::ColorType::Rgba => buf[..w * h * 4]
                .chunks_exact(4)
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect(),
            other => panic!("unsupported png color type {other:?}"),
        };
        if i == 1 {
            (fh, fw) = (h, w);
        } else {
            assert_eq!((h, w), (fh, fw), "frame {i} dims differ");
        }
        data.extend_from_slice(&rgb);
    }
    (data, fh, fw)
}

/// Resolve the DreamID-V DiT `.pth` from the HF cache (manifest role), matching
/// the production executor. `None` (skip) when not cached. The baked context is an
/// in-tree constant (`wan::dreamidv::baked_context`), not a download.
fn resolve_dreamidv_dit() -> Option<PathBuf> {
    cache::resolve(manifest::MANIFEST.get(role::DIT_DREAMIDV)?)
}

fn stats(v: &[f32]) -> (usize, f64, f64, f32, f32) {
    let nonfinite = v.iter().filter(|x| !x.is_finite()).count();
    let n = v.len() as f64;
    let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
    let lo = v.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (nonfinite, mean, var.sqrt(), lo, hi)
}

fn write_png(path: &Path, rgb: &[f32], t: usize, num_t: usize, h: usize, w: usize) {
    // rgb is CTHW [3, num_t, h, w] in [-1, 1]; emit frame `t`.
    let mut buf = vec![0u8; h * w * 3];
    let plane = h * w;
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = rgb[(c * num_t + t) * plane + y * w + x];
                buf[(y * w + x) * 3 + c] =
                    (((v + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    let file = std::fs::File::create(path).expect("create png");
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .expect("png header")
        .write_image_data(&buf)
        .expect("png data");
}

// ---------------------------------------------------------------------------
// Stage-A DiT single-forward parity vs the torch reference
// ---------------------------------------------------------------------------

/// Read a named f32 tensor from an already-parsed safetensors bundle.
fn st_f32(st: &safetensors::SafeTensors, name: &str) -> (Vec<f32>, Vec<usize>) {
    let t = st.tensor(name).unwrap_or_else(|_| panic!("tensor {name}"));
    let shape = t.shape().to_vec();
    let flat: Vec<f32> = t
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (flat, shape)
}

/// Least-squares `got ~= a*exp + b`; returns `(slope, rmse)` over finite pairs.
fn linfit(exp: &[f32], got: &[f32]) -> (f64, f64) {
    let (mut sx, mut sy, mut sxx, mut sxy, mut n) = (0f64, 0f64, 0f64, 0f64, 0usize);
    for (&x, &y) in exp.iter().zip(got) {
        if x.is_finite() && y.is_finite() {
            let (x, y) = (x as f64, y as f64);
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
            n += 1;
        }
    }
    if n < 2 {
        return (f64::NAN, f64::NAN);
    }
    let nn = n as f64;
    let denom = nn * sxx - sx * sx;
    let a = if denom.abs() > 0.0 {
        (nn * sxy - sx * sy) / denom
    } else {
        f64::NAN
    };
    let b = (sy - a * sx) / nn;
    let mut se = 0f64;
    for (&x, &y) in exp.iter().zip(got) {
        if x.is_finite() && y.is_finite() {
            let r = y as f64 - (a * x as f64 + b);
            se += r * r;
        }
    }
    (a, (se / nn).sqrt())
}

/// `(rel_rmse vs max|exp|, slope, max_abs_diff, cosine)`.
fn metrics(exp: &[f32], got: &[f32]) -> (f64, f64, f64, f64) {
    let n = exp.len().min(got.len());
    let (exp, got) = (&exp[..n], &got[..n]);
    let (slope, rmse) = linfit(exp, got);
    let max_ref = exp.iter().copied().map(f32::abs).fold(0f32, f32::max) as f64;
    let rel = if max_ref > 0.0 { rmse / max_ref } else { rmse };
    let mut max_abs = 0f64;
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..n {
        max_abs = max_abs.max((got[i] - exp[i]).abs() as f64);
        dot += exp[i] as f64 * got[i] as f64;
        na += (exp[i] as f64).powi(2);
        nb += (got[i] as f64).powi(2);
    }
    let cos = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        f64::NAN
    };
    (rel, slope, max_abs, cos)
}

/// STAGE A: DiT single-forward parity. Loads the torch dump
/// (`<dir>/../dreamidv_parity/stageA.safetensors`, produced by
/// `scratch/dreamidv_parity_ref.py`), feeds the identical 48-ch input +
/// source-face `img_ref` + baked context through the real DiT weights, and
/// diffs the `[16,2,16,16]` velocity + per-block residuals. Raw metrics only
/// (no verdict). Gated on `THINFER_DREAMIDV_PARITY`; skips cleanly when the dump
/// or weights are absent.
///
/// Run:
///   `THINFER_DREAMIDV_PARITY=1 cargo test -p thinfer-conformance \
///    --features dreamidv-e2e --release dreamidv_dit_parity -- --nocapture \
///    --test-threads=1`
#[tokio::test(flavor = "current_thread")]
async fn dreamidv_dit_parity() {
    let _ = thinfer_core::trace::init_from_env();
    if std::env::var("THINFER_DREAMIDV_PARITY").is_err() {
        eprintln!("dreamidv_dit_parity: THINFER_DREAMIDV_PARITY unset; skipping");
        return;
    }

    let Some(dit_path) = resolve_dreamidv_dit() else {
        eprintln!("dreamidv_dit_parity: DiT not in HF cache; skipping");
        return;
    };
    let dump_path = env_str(
        "THINFER_DREAMIDV_PARITY_DUMP",
        "C:/work/personal/thinfer/scratch/dreamidv_parity/stageA.safetensors",
    );
    if !PathBuf::from(&dump_path).exists() {
        eprintln!("dreamidv_dit_parity: missing {dump_path}; skipping");
        return;
    }

    // --- torch dump ---
    let bytes = std::fs::read(&dump_path).expect("read stageA dump");
    let st = safetensors::SafeTensors::deserialize(&bytes).expect("parse stageA dump");
    let (x, xs) = st_f32(&st, "x"); // [16,F,h,w]
    let (y, ys) = st_f32(&st, "y"); // [32,F,h,w]
    let (img_ref, irs) = st_f32(&st, "img_ref"); // [16,1,h,w]
    let (ref_out, os) = st_f32(&st, "out"); // [16,F,h,w]
    let (f_lat, h_lat, w_lat) = (xs[1], xs[2], xs[3]);
    eprintln!(
        "dreamidv_dit_parity: dims F={f_lat} h={h_lat} w={w_lat}; x{xs:?} y{ys:?} img_ref{irs:?} out{os:?}"
    );
    assert_eq!(xs[0], 16, "x must be 16-ch noise");
    assert_eq!(ys[0], 32, "y must be 32-ch (video++mask)");

    // 48-ch input, EXACTLY as DreamIdvPipeline::generate lays out `x_in`:
    // channels 0..16 = noise (x), 16..48 = y (video ++ mask), all CTHW row-major.
    let mut image = Vec::with_capacity(x.len() + y.len());
    image.extend_from_slice(&x);
    image.extend_from_slice(&y);
    assert_eq!(image.len(), 48 * f_lat * h_lat * w_lat);
    assert_eq!(img_ref.len(), 16 * h_lat * w_lat);

    let (context, ctx_rows) = thinfer_models::wan::dreamidv::baked_context();

    // --- source + residency + backend + pipeline (same setup as the health test) ---
    let vae_fr = manifest::MANIFEST
        .get(role::VAE_WAN21)
        .expect("VAE_WAN21 role");
    let Some(vae_path) = cache::resolve(vae_fr) else {
        eprintln!("dreamidv_dit_parity: Wan2.1 VAE not in HF cache; skipping");
        return;
    };
    let dit_opener = MmapFileOpener::new(&dit_path).await.expect("open dit");
    let vae_opener = MmapFileOpener::new(&vae_path).await.expect("open vae");
    let source = open_dreamidv_source(dit_opener, vec![vae_opener], NUM_LAYERS)
        .await
        .unwrap_or_else(|e| panic!("open dreamidv source: {e:?}"));

    let cfg = WgpuConfig {
        power_preference: match std::env::var("THINFER_POWER_PREF").as_deref() {
            Ok("low") => PowerPreference::LowPower,
            _ => PowerPreference::HighPerformance,
        },
        timestamps: std::env::var("THINFER_TRACE").is_ok(),
        disable_coopmat: std::env::var("THINFER_NO_COOPMAT").is_ok(),
    };
    let backend = Arc::new(
        WgpuBackend::new_with_config(cfg)
            .await
            .expect("wgpu adapter unavailable"),
    );
    let budget_gb = env_usize("THINFER_DREAMIDV_BUDGET_GB", 8) as u64;
    let budget = ResidencyBudget {
        ram_bytes: budget_gb << 30,
        vram_bytes: budget_gb << 30,
    };
    let residency = WeightResidency::new(source, budget);
    // i8 DP4A on the normed matmul sites is the production default; opt out via
    // THINFER_DREAMIDV_NO_I8 for a bf16 A/B.
    let i8_matmul = std::env::var_os("THINFER_DREAMIDV_NO_I8").is_none();
    let pipeline = DreamIdvPipeline::load(
        Arc::clone(&backend),
        residency,
        &context,
        ctx_rows,
        i8_matmul,
    )
    .await
    .unwrap_or_else(|e| panic!("DreamIdvPipeline::load: {e:?}"));

    // --- the single DiT forward (velocity + per-block residuals) ---
    let (vel, per_block) = pipeline
        .dit_forward_parity(f_lat, h_lat, w_lat, &image, &img_ref, 500.0)
        .await
        .unwrap_or_else(|e| panic!("dit_forward_parity: {e:?}"));
    assert_eq!(vel.len(), ref_out.len(), "velocity length vs ref out");

    // --- per-block localization (token-space [rows, inner], ref prefix first) ---
    eprintln!("dreamidv_dit_parity: per-block residual rel_rmse (localizes divergence onset):");
    for (i, eng_b) in per_block.iter().enumerate() {
        let name = format!("block{i:02}");
        if st.tensor(&name).is_err() {
            continue;
        }
        let (ref_b, rbs) = st_f32(&st, &name);
        if ref_b.len() != eng_b.len() {
            eprintln!(
                "  {name}: len mismatch eng={} ref={} {rbs:?}",
                eng_b.len(),
                ref_b.len()
            );
            continue;
        }
        let (rel, slope, maxd, cos) = metrics(&ref_b, eng_b);
        eprintln!(
            "  {name}: rel_rmse={rel:.4e} slope={slope:.5} max_abs_diff={maxd:.4e} cosine={cos:.6}"
        );
    }

    // --- output velocity metrics (THE Stage-A number) ---
    let (rel, slope, maxd, cos) = metrics(&ref_out, &vel);
    let (rn, rmean, rstd, rlo, rhi) = stats(&ref_out);
    let (en, emean, estd, elo, ehi) = stats(&vel);
    eprintln!(
        "\ndreamidv_dit_parity: === STAGE A: DiT output velocity [16,{f_lat},{h_lat},{w_lat}] ==="
    );
    eprintln!("  rel_rmse={rel:.6e}  slope={slope:.6}  max_abs_diff={maxd:.6e}  cosine={cos:.8}");
    eprintln!("  ref: n={rn} mean={rmean:.6} std={rstd:.6} min={rlo:.6} max={rhi:.6}");
    eprintln!("  our: n={en} mean={emean:.6} std={estd:.6} min={elo:.6} max={ehi:.6}");
    eprintln!("  side-by-side (idx: ref vs our):");
    for &idx in &[0usize, 1, 100, 4096, 8191] {
        if idx < ref_out.len() {
            eprintln!("    [{idx}] {:.6} vs {:.6}", ref_out[idx], vel[idx]);
        }
    }
}

/// STAGE B: Wan2.1 VAE-encode parity. Feeds the identical fixed-seed pixel clip
/// from `stageB.safetensors` (produced by `scratch/dreamidv_vae_parity_ref.py`)
/// through `WanVaeEncoder::encode` + the same `(mu - mean)/std` normalization,
/// and diffs the z16 latent vs the reference. Raw metrics only. Gated on
/// `THINFER_DREAMIDV_PARITY`; skips cleanly when the dump or weights are absent.
#[tokio::test(flavor = "current_thread")]
async fn dreamidv_vae_parity() {
    let _ = thinfer_core::trace::init_from_env();
    if std::env::var("THINFER_DREAMIDV_PARITY").is_err() {
        eprintln!("dreamidv_vae_parity: THINFER_DREAMIDV_PARITY unset; skipping");
        return;
    }
    let Some(dit_path) = resolve_dreamidv_dit() else {
        eprintln!("dreamidv_vae_parity: DiT not in HF cache; skipping");
        return;
    };
    let dump_path = env_str(
        "THINFER_DREAMIDV_PARITY_DUMP_B",
        "C:/work/personal/thinfer/scratch/dreamidv_parity/stageB.safetensors",
    );
    if !PathBuf::from(&dump_path).exists() {
        eprintln!("dreamidv_vae_parity: missing {dump_path}; skipping");
        return;
    }
    let bytes = std::fs::read(&dump_path).expect("read stageB dump");
    let st = safetensors::SafeTensors::deserialize(&bytes).expect("parse stageB dump");
    let (pixels, ps) = st_f32(&st, "pixels"); // [3,1,h,w]
    let (ref_lat, ls) = st_f32(&st, "latent"); // [16,1,h/8,w/8]
    let (h_in, w_in) = (ps[2], ps[3]);
    eprintln!("dreamidv_vae_parity: pixels{ps:?} ref_latent{ls:?}");
    assert_eq!(ps[0], 3, "pixels must be 3-ch");

    let (context, ctx_rows) = thinfer_models::wan::dreamidv::baked_context();
    let vae_fr = manifest::MANIFEST
        .get(role::VAE_WAN21)
        .expect("VAE_WAN21 role");
    let Some(vae_path) = cache::resolve(vae_fr) else {
        eprintln!("dreamidv_vae_parity: Wan2.1 VAE not in HF cache; skipping");
        return;
    };
    let dit_opener = MmapFileOpener::new(&dit_path).await.expect("open dit");
    let vae_opener = MmapFileOpener::new(&vae_path).await.expect("open vae");
    let source = open_dreamidv_source(dit_opener, vec![vae_opener], NUM_LAYERS)
        .await
        .unwrap_or_else(|e| panic!("open dreamidv source: {e:?}"));
    let cfg = WgpuConfig {
        power_preference: match std::env::var("THINFER_POWER_PREF").as_deref() {
            Ok("low") => PowerPreference::LowPower,
            _ => PowerPreference::HighPerformance,
        },
        timestamps: std::env::var("THINFER_TRACE").is_ok(),
        disable_coopmat: std::env::var("THINFER_NO_COOPMAT").is_ok(),
    };
    let backend = Arc::new(
        WgpuBackend::new_with_config(cfg)
            .await
            .expect("wgpu adapter unavailable"),
    );
    let budget_gb = env_usize("THINFER_DREAMIDV_BUDGET_GB", 8) as u64;
    let budget = ResidencyBudget {
        ram_bytes: budget_gb << 30,
        vram_bytes: budget_gb << 30,
    };
    let residency = WeightResidency::new(source, budget);
    // i8 DP4A on the normed matmul sites is the production default; opt out via
    // THINFER_DREAMIDV_NO_I8 for a bf16 A/B.
    let i8_matmul = std::env::var_os("THINFER_DREAMIDV_NO_I8").is_none();
    let pipeline = DreamIdvPipeline::load(
        Arc::clone(&backend),
        residency,
        &context,
        ctx_rows,
        i8_matmul,
    )
    .await
    .unwrap_or_else(|e| panic!("DreamIdvPipeline::load: {e:?}"));

    let lat = pipeline
        .vae_encode_parity(&pixels, 1, h_in, w_in)
        .await
        .unwrap_or_else(|e| panic!("vae_encode_parity: {e:?}"));
    assert_eq!(lat.len(), ref_lat.len(), "latent length vs ref");

    let (rel, slope, maxd, cos) = metrics(&ref_lat, &lat);
    let (rn, rmean, rstd, rlo, rhi) = stats(&ref_lat);
    let (en, emean, estd, elo, ehi) = stats(&lat);
    eprintln!("\ndreamidv_vae_parity: === STAGE B: VAE-encode z16 latent {ls:?} ===");
    eprintln!("  rel_rmse={rel:.6e}  slope={slope:.6}  max_abs_diff={maxd:.6e}  cosine={cos:.8}");
    eprintln!("  ref: n={rn} mean={rmean:.6} std={rstd:.6} min={rlo:.6} max={rhi:.6}");
    eprintln!("  our: n={en} mean={emean:.6} std={estd:.6} min={elo:.6} max={ehi:.6}");
    eprintln!("  side-by-side (idx: ref vs our):");
    for &idx in &[0usize, 1, 100, 2048, 4095] {
        if idx < ref_lat.len() {
            eprintln!("    [{idx}] {:.6} vs {:.6}", ref_lat[idx], lat[idx]);
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn dreamidv_health() {
    let _rollup = RollupDumpOnDrop(thinfer_core::trace::init_from_env());

    let Some(dit_path) = resolve_dreamidv_dit() else {
        eprintln!("dreamidv_health: DiT not in HF cache; skipping");
        return;
    };
    let assets = env_str(
        "THINFER_DREAMIDV_ASSETS",
        "C:/work/personal/thinfer/third-party/DreamID-V/assets/test_case",
    );
    let ref_image = PathBuf::from(&assets).join("ref_image/an_1.jpg");
    let ref_video = PathBuf::from(&assets).join("ref_video/a_girl.mp4");
    let ref_mask = PathBuf::from(&assets).join("ref_video/temp_generated/a_girl_mask.mp4");
    for p in [&ref_image, &ref_video, &ref_mask] {
        if !p.exists() {
            eprintln!("dreamidv_health: asset {} missing; skipping", p.display());
            return;
        }
    }

    // Wan2.1 VAE from the HF cache.
    let vae_fr = manifest::MANIFEST
        .get(role::VAE_WAN21)
        .expect("VAE_WAN21 role");
    let Some(vae_path) = cache::resolve(vae_fr) else {
        eprintln!(
            "dreamidv_health: {}/{} not in HF cache ({}); skipping",
            vae_fr.repo,
            vae_fr.path,
            cache::cache_root().display()
        );
        return;
    };

    let frames = env_usize("THINFER_DREAMIDV_FRAMES", 5); // 4k+1
    let area = env_usize("THINFER_DREAMIDV_AREA", 128 * 128) as f64;
    let steps = env_usize("THINFER_DREAMIDV_STEPS", 2) as u32;
    assert!((frames - 1).is_multiple_of(4), "frames must be 4k+1");

    // Decode assets via ffmpeg into temp dirs.
    let tmp = tempfile::tempdir().expect("tempdir");
    let (vdir, mdir, idir) = (
        tmp.path().join("v"),
        tmp.path().join("m"),
        tmp.path().join("i"),
    );
    for d in [&vdir, &mdir, &idir] {
        std::fs::create_dir_all(d).unwrap();
    }
    if !ffmpeg_decode(&ref_video, frames, &vdir)
        || !ffmpeg_decode(&ref_mask, frames, &mdir)
        || !ffmpeg_decode(&ref_image, 1, &idir)
    {
        eprintln!("dreamidv_health: ffmpeg unavailable or failed; skipping");
        return;
    }
    let (video, vh, vw) = read_frames(&vdir, frames);
    let (mask, mh, mw) = read_frames(&mdir, frames);
    let (image, ih, iw) = read_frames(&idir, 1);
    eprintln!("dreamidv_health: decoded video {vw}x{vh}x{frames}, mask {mw}x{mh}, image {iw}x{ih}");

    let (context, ctx_rows) = thinfer_models::wan::dreamidv::baked_context();

    // --- source + residency + backend ---
    let dit_opener = MmapFileOpener::new(&dit_path).await.expect("open dit");
    let vae_opener = MmapFileOpener::new(&vae_path).await.expect("open vae");
    let source = open_dreamidv_source(dit_opener, vec![vae_opener], NUM_LAYERS)
        .await
        .unwrap_or_else(|e| panic!("open dreamidv source: {e:?}"));

    let cfg = WgpuConfig {
        power_preference: match std::env::var("THINFER_POWER_PREF").as_deref() {
            Ok("low") => PowerPreference::LowPower,
            _ => PowerPreference::HighPerformance,
        },
        timestamps: std::env::var("THINFER_TRACE").is_ok(),
        disable_coopmat: std::env::var("THINFER_NO_COOPMAT").is_ok(),
    };
    let backend = Arc::new(
        WgpuBackend::new_with_config(cfg)
            .await
            .expect("wgpu adapter unavailable"),
    );
    let budget_gb = env_usize("THINFER_DREAMIDV_BUDGET_GB", 8) as u64;
    let budget = ResidencyBudget {
        ram_bytes: budget_gb << 30,
        vram_bytes: budget_gb << 30,
    };
    let residency = WeightResidency::new(source, budget);
    // i8 DP4A on the normed matmul sites is the production default; opt out via
    // THINFER_DREAMIDV_NO_I8 for a bf16 A/B.
    let i8_matmul = std::env::var_os("THINFER_DREAMIDV_NO_I8").is_none();
    let pipeline = DreamIdvPipeline::load(
        Arc::clone(&backend),
        residency,
        &context,
        ctx_rows,
        i8_matmul,
    )
    .await
    .unwrap_or_else(|e| panic!("DreamIdvPipeline::load: {e:?}"));

    let inputs = DreamIdvInputs {
        video: RgbFrames {
            data: &video,
            frames,
            h: vh,
            w: vw,
        },
        mask: RgbFrames {
            data: &mask,
            frames,
            h: mh,
            w: mw,
        },
        image: RgbFrames {
            data: &image,
            frames: 1,
            h: ih,
            w: iw,
        },
        target_area: area,
        steps,
        guide_scale: thinfer_models::wan::dreamidv::DEFAULT_GUIDE_SCALE,
        seed: SEED,
    };
    let mut diag = DreamIdvDiag::default();
    let video_out = pipeline
        .generate(&inputs, Some(&mut diag))
        .await
        .unwrap_or_else(|e| panic!("generate: {e:?}"));

    // --- health assertions: finite + non-constant at every checkpoint ---
    let check = |name: &str, v: &[f32], want_var: bool| {
        assert!(!v.is_empty(), "{name} empty");
        let (nonfinite, mean, std, lo, hi) = stats(v);
        eprintln!(
            "dreamidv_health: {name}: n={} nonfinite={nonfinite} mean={mean:.4} std={std:.6} range=[{lo:.4},{hi:.4}]",
            v.len()
        );
        assert_eq!(nonfinite, 0, "{name} has {nonfinite} non-finite values");
        if want_var {
            assert!(std > 1e-5, "{name} is ~constant (std={std:.3e})");
        }
    };
    check("video_latent", &diag.video_latent, true);
    check("mask_latent", &diag.mask_latent, false); // a mask CAN be near-flat
    check("image_latent", &diag.image_latent, true);
    check("denoised_latent", &diag.denoised_latent, true);
    check("rgb", &video_out.frames, true);
    for &x in &video_out.frames {
        assert!(
            (-1.001..=1.001).contains(&x),
            "rgb sample {x} out of [-1, 1]"
        );
    }
    assert_eq!(
        video_out.frames.len(),
        3 * video_out.num_frames * video_out.height * video_out.width,
        "rgb buffer size"
    );

    // --- eyeball: dump the output frames ---
    let png_dir = env_str(
        "THINFER_PNG_DIR",
        "C:/work/personal/thinfer/scratch/dreamidv/out",
    );
    std::fs::create_dir_all(&png_dir).ok();
    for t in 0..video_out.num_frames {
        let p = PathBuf::from(&png_dir).join(format!("dreamidv_{t:03}.png"));
        write_png(
            &p,
            &video_out.frames,
            t,
            video_out.num_frames,
            video_out.height,
            video_out.width,
        );
    }
    eprintln!(
        "dreamidv_health: OK - {} frames {}x{}, PNGs -> {png_dir}",
        video_out.num_frames, video_out.width, video_out.height
    );
}
