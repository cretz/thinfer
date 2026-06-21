//! ONNX executor parity vs onnxruntime goldens for the face-swap models.
//!
//! Validates `thinfer_core::onnx` op-for-op against onnxruntime by running each
//! model on the same fixed input and comparing to a saved golden. Opt-in
//! (`faceswap-e2e`); skips cleanly unless the model + golden paths are set:
//!
//! ```text
//! THINFER_FS_SCRFD=<det_500m.onnx> THINFER_FS_ARCFACE=<arcface.onnx>
//! THINFER_FS_HYPERSWAP=<hyperswap_1a_256.onnx> THINFER_FS_GOLDEN=<scratch/faceswap/golden>
//! cargo test -p thinfer-conformance --features faceswap-e2e faceswap_onnx -- --nocapture
//! ```
//!
//! Goldens are produced by `scratch/faceswap/gen_golden.py` (numpy `.npy`).

#![cfg(feature = "faceswap-e2e")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use thinfer_core::backend::WgpuBackend;
use thinfer_core::onnx::OnnxModel;

/// Minimal numpy `.npy` reader for C-order little-endian float32 arrays.
fn read_npy_f32(path: &Path) -> (Vec<i64>, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(&bytes[0..6], b"\x93NUMPY", "not an npy file: {}", path.display());
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
    assert!(
        header.contains("<f4") || header.contains("|f4"),
        "npy not float32: {header}"
    );
    // Parse the shape tuple from "'shape': (a, b, ...)".
    let sh_start = header.find("'shape':").expect("shape key") + "'shape':".len();
    let open = header[sh_start..].find('(').unwrap() + sh_start + 1;
    let close = header[open..].find(')').unwrap() + open;
    let shape: Vec<i64> = header[open..close]
        .split(',')
        .filter_map(|s| s.trim().parse::<i64>().ok())
        .collect();
    let data_off = 10 + header_len;
    let data: Vec<f32> = bytes[data_off..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (shape, data)
}

struct Stats {
    max_abs: f32,
    rel: f32,
    nan: usize,
}

fn compare(got: &[f32], exp: &[f32]) -> Stats {
    assert_eq!(got.len(), exp.len(), "length mismatch {} vs {}", got.len(), exp.len());
    let mut max_abs = 0f32;
    let mut max_ref = 0f32;
    let mut nan = 0;
    for (&g, &e) in got.iter().zip(exp) {
        if !g.is_finite() {
            nan += 1;
            continue;
        }
        max_abs = max_abs.max((g - e).abs());
        max_ref = max_ref.max(e.abs());
    }
    Stats { max_abs, rel: max_abs / (max_ref + 1e-6), nan }
}

/// Run one model: load goldens by `tag`, build the executor at the golden input
/// shapes, run, and compare every output. Returns false if the model/goldens
/// are missing (skip).
async fn run_model(backend: std::sync::Arc<WgpuBackend>, env: &str, tag: &str, tol: f32) -> bool {
    let Ok(model_path) = std::env::var(env) else {
        eprintln!("[skip] {env} unset");
        return false;
    };
    let Ok(golden_dir) = std::env::var("THINFER_FS_GOLDEN") else {
        eprintln!("[skip] THINFER_FS_GOLDEN unset");
        return false;
    };
    let golden = PathBuf::from(golden_dir);
    let onnx_bytes = std::fs::read(&model_path).expect("read onnx");

    // Discover inputs/outputs from the golden filenames `{tag}_in_*` / `_out_*`.
    let mut inputs: HashMap<String, Vec<f32>> = HashMap::new();
    let mut input_shapes: HashMap<String, Vec<i64>> = HashMap::new();
    let mut outputs: HashMap<String, (Vec<i64>, Vec<f32>)> = HashMap::new();
    for entry in std::fs::read_dir(&golden).unwrap() {
        let p = entry.unwrap().path();
        let fname = p.file_name().unwrap().to_string_lossy().to_string();
        if let Some(rest) = fname.strip_prefix(&format!("{tag}_in_")) {
            let name = rest.trim_end_matches(".npy").to_string();
            let (shape, data) = read_npy_f32(&p);
            input_shapes.insert(name.clone(), shape);
            inputs.insert(name, data);
        } else if let Some(rest) = fname.strip_prefix(&format!("{tag}_out_")) {
            let name = rest.trim_end_matches(".npy").to_string();
            outputs.insert(name.clone(), read_npy_f32(&p));
        }
    }
    assert!(!inputs.is_empty(), "no golden inputs for {tag}");

    let model = OnnxModel::load(backend, &onnx_bytes, &input_shapes)
        .await
        .expect("load onnx model");
    let got = model.run(&inputs).await.expect("run onnx model");

    let mut worst = 0f32;
    for (name, (exp_shape, exp)) in &outputs {
        let (got_shape, got_data) = got.get(name).unwrap_or_else(|| panic!("missing output {name}"));
        assert_eq!(got_shape, exp_shape, "output {name} shape");
        let s = compare(got_data, exp);
        eprintln!(
            "[{tag}] output {name}: max_abs={:.5e} rel={:.5e} nan={} shape={:?}",
            s.max_abs, s.rel, s.nan, got_shape
        );
        assert_eq!(s.nan, 0, "output {name} has non-finite values");
        worst = worst.max(s.rel);
    }
    assert!(worst <= tol, "[{tag}] worst rel {worst:.5e} exceeds tol {tol:.1e}");
    true
}

#[test]
fn scrfd_parity() {
    let ran = pollster::block_on(async {
        let backend = std::sync::Arc::new(WgpuBackend::new().await.expect("wgpu"));
        run_model(backend, "THINFER_FS_SCRFD", "scrfd", 2e-2).await
    });
    if !ran {
        eprintln!("scrfd_parity skipped");
    }
}

#[test]
fn arcface_parity() {
    let ran = pollster::block_on(async {
        let backend = std::sync::Arc::new(WgpuBackend::new().await.expect("wgpu"));
        run_model(backend, "THINFER_FS_ARCFACE", "arcface", 2e-2).await
    });
    if !ran {
        eprintln!("arcface_parity skipped");
    }
}

#[test]
fn hyperswap_parity() {
    let ran = pollster::block_on(async {
        let backend = std::sync::Arc::new(WgpuBackend::new().await.expect("wgpu"));
        // fp16 weights upcast to f32; onnxruntime CPU may compute fp16, so a
        // looser tolerance.
        run_model(backend, "THINFER_FS_HYPERSWAP", "hyperswap", 6e-2).await
    });
    if !ran {
        eprintln!("hyperswap_parity skipped");
    }
}
