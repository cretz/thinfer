//! Shared parity helpers (linfit + binary readers + HF-cache resolution) for the
//! HunyuanVideo 1.5 suite. Mirrors the LTX/qwen_image suites so the hunyuan tests
//! don't reach across test binaries.
#![cfg(feature = "hunyuan-e2e")]

use std::path::{Path, PathBuf};

/// Least-squares fit `y ~ slope*x + bias`; returns `(slope, bias, rmse, n)` over
/// finite pairs. Slope near 1 + low rmse == a faithful match.
pub fn linfit(x: &[f32], y: &[f32]) -> (f64, f64, f64, usize) {
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

/// Relative rmse `rmse / mean(|exp|)` plus the linfit slope. Prints a one-line
/// report and returns `(slope, rel)`.
pub fn report(label: &str, exp: &[f32], got: &[f32]) -> (f64, f64) {
    assert_eq!(
        exp.len(),
        got.len(),
        "[{label}] length mismatch exp={} got={}",
        exp.len(),
        got.len()
    );
    let (slope, bias, rmse, cnt) = linfit(exp, got);
    let mean_abs = exp.iter().map(|x| x.abs() as f64).sum::<f64>() / (exp.len().max(1) as f64);
    let rel = if mean_abs > 0.0 { rmse / mean_abs } else { 0.0 };
    eprintln!(
        "[{label}] slope={slope:.6} bias={bias:+.4e} rmse={rmse:.4e} rel={:.3}% n={cnt}",
        rel * 100.0
    );
    (slope, rel)
}

pub fn read_f32(p: &Path) -> Vec<f32> {
    let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Resolve a file under an HF-hub snapshot dir: `THINFER_<env>` override, else the
/// newest match of `<HF_HOME or ~/.cache/huggingface>/hub/<repo>/snapshots/*/<rel>`.
/// Returns `None` (test skips) when uncached.
pub fn resolve_hf(env: &str, repo: &str, rel: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(env) {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let hub = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("USERPROFILE")
                .or_else(|_| std::env::var("HOME"))
                .expect("home dir");
            PathBuf::from(home).join(".cache").join("huggingface")
        })
        .join("hub")
        .join(repo)
        .join("snapshots");
    let snaps = std::fs::read_dir(&hub).ok()?;
    let mut best: Option<PathBuf> = None;
    let mut best_t = std::time::SystemTime::UNIX_EPOCH;
    for ent in snaps.flatten() {
        let cand = ent.path().join(rel);
        if cand.exists() {
            let t = ent
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            if best.is_none() || t >= best_t {
                best = Some(cand);
                best_t = t;
            }
        }
    }
    best
}
