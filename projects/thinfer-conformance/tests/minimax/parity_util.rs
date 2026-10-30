//! Small parity helpers (linfit + binary readers) for the MiniMax-H3 suite,
//! mirrored from the LTX suite so the tests don't reach across test binaries.
#![cfg(feature = "minimax-e2e")]

use std::path::Path;

/// Least-squares fit of `y ~ slope*x + bias` over finite pairs; returns
/// `(slope, bias, rmse)`. Slope near 1 + low rmse == a faithful match.
pub fn linfit(x: &[f32], y: &[f32]) -> (f64, f64, f64) {
    let (mut sx, mut sy, mut sxx, mut sxy, mut cnt) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
    for (&xi, &yi) in x.iter().zip(y) {
        let (xi, yi) = (xi as f64, yi as f64);
        if xi.is_finite() && yi.is_finite() {
            sx += xi;
            sy += yi;
            sxx += xi * xi;
            sxy += xi * yi;
            cnt += 1;
        }
    }
    let nf = cnt as f64;
    let denom = nf * sxx - sx * sx;
    if cnt < 3 || denom.abs() < 1e-18 {
        return (f64::NAN, 0.0, f64::NAN);
    }
    let slope = (nf * sxy - sx * sy) / denom;
    let bias = (sy - slope * sx) / nf;
    let mut resid_sq = 0.0f64;
    for (&xi, &yi) in x.iter().zip(y) {
        let (xi, yi) = (xi as f64, yi as f64);
        if xi.is_finite() && yi.is_finite() {
            let r = yi - (slope * xi + bias);
            resid_sq += r * r;
        }
    }
    (slope, bias, (resid_sq / nf).sqrt())
}

/// One stage's agreement: linfit slope plus `rms(got - exp) / rms(exp)`.
/// The relative error is against the raw difference, not the fit residual, so
/// a slope or bias error counts against it too.
pub fn report(label: &str, exp: &[f32], got: &[f32]) -> (f64, f64) {
    assert_eq!(
        exp.len(),
        got.len(),
        "[{label}] length mismatch exp={} got={}",
        exp.len(),
        got.len()
    );
    let (slope, bias, _) = linfit(exp, got);
    let (mut num, mut den, mut bad) = (0.0f64, 0.0f64, 0usize);
    for (&e, &g) in exp.iter().zip(got) {
        if !g.is_finite() {
            bad += 1;
            continue;
        }
        num += ((g - e) as f64).powi(2);
        den += (e as f64).powi(2);
    }
    let rel = if den > 0.0 { (num / den).sqrt() } else { 0.0 };
    let rel = if bad > 0 { f64::INFINITY } else { rel };
    eprintln!(
        "[{label:>12}] slope={slope:.5} bias={bias:+.3e} rel={:.3}% n={} nonfinite={bad}",
        rel * 100.0,
        exp.len()
    );
    (slope, rel)
}

pub fn read_f32(p: &Path) -> Vec<f32> {
    read_words(p).into_iter().map(f32::from_le_bytes).collect()
}

pub fn read_u32(p: &Path) -> Vec<u32> {
    read_words(p).into_iter().map(u32::from_le_bytes).collect()
}

pub fn read_i32(p: &Path) -> Vec<i32> {
    read_words(p).into_iter().map(i32::from_le_bytes).collect()
}

fn read_words(p: &Path) -> Vec<[u8; 4]> {
    let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    bytes
        .chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect()
}
