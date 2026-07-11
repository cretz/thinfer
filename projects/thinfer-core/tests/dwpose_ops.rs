//! ONNX-executor parity for the DreamID-V DWPose models (yolox_l + RTMPose
//! dw-ll_ucoco_384) vs onnxruntime goldens. Validates the ops added for these
//! nets: Slice/Split/MatMul/GlobalAveragePool/ReduceSum/Clip/Sqrt/HardSigmoid.
//!
//! Independent of thinfer-models (runs the executor directly), so it validates
//! the ops even while other crates are mid-refactor. Opt-in; skips unless the
//! model + golden paths are set:
//! ```text
//! THINFER_DWPOSE_YOLOX=<yolox_l.onnx> THINFER_DWPOSE_POSE=<dw-ll_ucoco_384.onnx>
//! THINFER_DWPOSE_GOLDEN=<scratch/golden>
//! cargo test -p thinfer-core --test dwpose_ops -- --nocapture --test-threads=1
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::onnx::OnnxModel;

fn read_npy_f32(path: &Path) -> (Vec<i64>, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(&bytes[0..6], b"\x93NUMPY", "not npy: {}", path.display());
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
    assert!(header.contains("<f4") || header.contains("|f4"), "not f32");
    let sh_start = header.find("'shape':").unwrap() + "'shape':".len();
    let open = header[sh_start..].find('(').unwrap() + sh_start + 1;
    let close = header[open..].find(')').unwrap() + open;
    let shape: Vec<i64> = header[open..close]
        .split(',')
        .filter_map(|s| s.trim().parse::<i64>().ok())
        .collect();
    let data: Vec<f32> = bytes[10 + header_len..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (shape, data)
}

fn compare(got: &[f32], exp: &[f32]) -> (f32, f32, usize) {
    assert_eq!(got.len(), exp.len(), "len {} vs {}", got.len(), exp.len());
    let (mut max_abs, mut max_ref, mut nan) = (0f32, 0f32, 0usize);
    for (&g, &e) in got.iter().zip(exp) {
        if !g.is_finite() {
            nan += 1;
            continue;
        }
        max_abs = max_abs.max((g - e).abs());
        max_ref = max_ref.max(e.abs());
    }
    (max_abs, max_abs / (max_ref + 1e-6), nan)
}

fn simcc_argmax(t: &[f32], k: usize, w: usize) -> Vec<usize> {
    (0..k)
        .map(|j| {
            t[j * w..j * w + w]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        })
        .collect()
}

fn golden() -> Option<PathBuf> {
    std::env::var("THINFER_DWPOSE_GOLDEN")
        .ok()
        .map(PathBuf::from)
}

#[test]
fn yolox_parity() {
    let (Some(model), Some(g)) = (std::env::var("THINFER_DWPOSE_YOLOX").ok(), golden()) else {
        eprintln!("yolox_parity skipped (set THINFER_DWPOSE_YOLOX + THINFER_DWPOSE_GOLDEN)");
        return;
    };
    pollster::block_on(async {
        let backend = Arc::new(WgpuBackend::new().await.expect("wgpu"));
        let (in_shape, in_data) = read_npy_f32(&g.join("yolox_in_images.npy"));
        let (out_shape, exp) = read_npy_f32(&g.join("yolox_out_output.npy"));
        let onnx = std::fs::read(&model).unwrap();
        let mut ins = HashMap::new();
        ins.insert("images".to_string(), in_shape);
        let m = OnnxModel::load(backend, &onnx, &ins).await.expect("load");
        let mut feeds = HashMap::new();
        feeds.insert("images".to_string(), in_data);
        let out = m.run(&feeds).await.expect("run");
        let (got_shape, got) = &out[&m.output_names[0]];
        assert_eq!(got_shape, &out_shape, "yolox output shape");
        let (max_abs, rel, nan) = compare(got, &exp);
        eprintln!("[yolox] max_abs={max_abs:.4e} rel={rel:.4e} nan={nan} shape={got_shape:?}");
        assert_eq!(nan, 0);
        assert!(rel <= 2e-2, "yolox rel {rel:.4e} exceeds 2e-2");
    });
}

#[test]
fn dwpose_parity() {
    let (Some(model), Some(g)) = (std::env::var("THINFER_DWPOSE_POSE").ok(), golden()) else {
        eprintln!("dwpose_parity skipped (set THINFER_DWPOSE_POSE + THINFER_DWPOSE_GOLDEN)");
        return;
    };
    pollster::block_on(async {
        let backend = Arc::new(WgpuBackend::new().await.expect("wgpu"));
        let (in_shape, in_data) = read_npy_f32(&g.join("dwpose_in_input.npy"));
        let (sx_shape, exp_x) = read_npy_f32(&g.join("dwpose_out_simcc_x.npy"));
        let (_sy_shape, exp_y) = read_npy_f32(&g.join("dwpose_out_simcc_y.npy"));
        let onnx = std::fs::read(&model).unwrap();
        let mut ins = HashMap::new();
        ins.insert("input".to_string(), in_shape);
        let m = OnnxModel::load(backend, &onnx, &ins).await.expect("load");
        let mut feeds = HashMap::new();
        feeds.insert("input".to_string(), in_data);
        let out = m.run(&feeds).await.expect("run");
        let (_gx, got_x) = &out["simcc_x"];
        let (_gy, got_y) = &out["simcc_y"];
        let (ax, rx, nx) = compare(got_x, &exp_x);
        let (ay, ry, ny) = compare(got_y, &exp_y);
        eprintln!("[dwpose] simcc_x max_abs={ax:.4e} rel={rx:.4e} nan={nx}");
        eprintln!("[dwpose] simcc_y max_abs={ay:.4e} rel={ry:.4e} nan={ny}");
        assert_eq!(nx + ny, 0);
        let k = sx_shape[1] as usize;
        let wx = sx_shape[2] as usize;
        let wy = exp_y.len() / k;
        let (glx, elx) = (simcc_argmax(got_x, k, wx), simcc_argmax(&exp_x, k, wx));
        let (gly, ely) = (simcc_argmax(got_y, k, wy), simcc_argmax(&exp_y, k, wy));
        let dx = glx
            .iter()
            .zip(&elx)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        let dy = gly
            .iter()
            .zip(&ely)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        let agx = glx.iter().zip(&elx).filter(|(a, b)| a == b).count();
        let agy = gly.iter().zip(&ely).filter(|(a, b)| a == b).count();
        eprintln!("[dwpose] argmax agree x={agx}/{k} y={agy}/{k}  max bin drift x={dx} y={dy}");
        assert!(rx <= 2e-2 && ry <= 2e-2, "dwpose rel x={rx:.4e} y={ry:.4e}");
        assert!(
            dx <= 1 && dy <= 1,
            "keypoint argmax drifted >1 bin (x={dx} y={dy})"
        );
    });
}
