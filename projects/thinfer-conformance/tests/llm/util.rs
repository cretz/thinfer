//! Compare helpers for the LLM parity suite.
#![cfg(feature = "llm-e2e")]

use std::path::Path;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};

pub fn read_f32(p: &Path) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    bytemuck::cast_slice(&b).to_vec()
}

pub fn read_i32(p: &Path) -> Vec<i32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    bytemuck::cast_slice(&b).to_vec()
}

/// `(rel, cos)`: `rmse(got - exp) / rms(exp)` and cosine similarity. Panics
/// on non-finite values. Prints one line.
pub fn compare(label: &str, exp: &[f32], got: &[f32]) -> (f64, f64) {
    assert_eq!(exp.len(), got.len(), "[{label}] length mismatch");
    assert!(
        got.iter().all(|x| x.is_finite()),
        "[{label}] engine output has non-finite values"
    );
    let (mut se, mut ee, mut gg, mut eg) = (0f64, 0f64, 0f64, 0f64);
    for (&e, &g) in exp.iter().zip(got) {
        let (e, g) = (e as f64, g as f64);
        se += (g - e) * (g - e);
        ee += e * e;
        gg += g * g;
        eg += e * g;
    }
    let rel = (se / ee.max(1e-30)).sqrt();
    let cos = eg / (ee.sqrt() * gg.sqrt()).max(1e-30);
    eprintln!("[{label}] rel={:.4}% cos={cos:.6}", rel * 100.0);
    (rel, cos)
}

/// The test backend, CUDA-accelerated unless `THINFER_CUDA=off`.
pub async fn backend() -> Arc<WgpuBackend> {
    let b = backend_wgpu().await;
    if std::env::var("THINFER_CUDA").as_deref() != Ok("off") {
        match thinfer_cuda::install(&b) {
            Ok((name, sm)) => eprintln!("cuda: {name} sm {sm:?}"),
            Err(e) => eprintln!("cuda unavailable: {e}"),
        }
    }
    b
}

/// RAM budget for the LLM tests (`THINFER_LLM_RAM_GB`, default 16).
pub fn ram_budget() -> u64 {
    std::env::var("THINFER_LLM_RAM_GB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(16)
        << 30
}

async fn backend_wgpu() -> Arc<WgpuBackend> {
    Arc::new(
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
    )
}
