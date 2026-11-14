//! Small shared parity helpers (linfit + binary readers), mirrored from the
//! ideogram4 suite so the Qwen-Image tests don't reach across test binaries.
//!
//! Gated on the same feature as the parity tests that consume it (without it
//! every test body is empty and these helpers would be dead code).
#![cfg(feature = "qwen-image-e2e")]

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

/// Relative rmse of the DIRECT elementwise error, `rmse(got - exp) /
/// mean(|exp|)`, plus the linfit slope (the fit's bias and residual are
/// printed as diagnostics only, so a constant offset still fails). Panics on
/// any non-finite value in either side. Prints a one-line report and returns
/// `(slope, rel)`.
pub fn report(label: &str, exp: &[f32], got: &[f32]) -> (f64, f64) {
    assert_eq!(
        exp.len(),
        got.len(),
        "[{label}] length mismatch exp={} got={}",
        exp.len(),
        got.len()
    );
    let bad = |v: &[f32]| v.iter().filter(|x| !x.is_finite()).count();
    let (bad_exp, bad_got) = (bad(exp), bad(got));
    assert!(
        bad_exp == 0 && bad_got == 0,
        "[{label}] non-finite values: exp {bad_exp}, got {bad_got} of {}",
        exp.len()
    );
    let (slope, bias, fit_rmse, cnt) = linfit(exp, got);
    let n = exp.len().max(1) as f64;
    let rmse = (exp
        .iter()
        .zip(got)
        .map(|(&e, &g)| (g as f64 - e as f64).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    let mean_abs = exp.iter().map(|x| x.abs() as f64).sum::<f64>() / n;
    let rel = if mean_abs > 0.0 { rmse / mean_abs } else { 0.0 };
    eprintln!(
        "[{label}] slope={slope:.6} bias={bias:+.4e} fit_rmse={fit_rmse:.4e} rmse={rmse:.4e} rel={:.3}% n={cnt}",
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
