//! Small shared parity helpers (linfit + binary readers) for the LTX-2.3 suite,
//! mirrored from the qwen_image/ideogram4 suites so the LTX tests don't reach
//! across test binaries.
//!
//! Gated on the same feature as the parity tests that consume it (without it
//! every test body is empty and these helpers would be dead code).
#![cfg(feature = "ltx-e2e")]

use std::path::Path;

/// Least-squares fit of `y ~ slope*x + bias`; returns `(slope, bias, rmse, n)`
/// over finite pairs. Slope near 1 + low rmse == a faithful match.
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

/// Relative rmse: `rmse(exp,got) / mean(|exp|)`, plus the linfit slope.
/// Prints a one-line report and returns `(slope, rel)`.
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

pub fn read_u32(p: &Path) -> Vec<u32> {
    let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
