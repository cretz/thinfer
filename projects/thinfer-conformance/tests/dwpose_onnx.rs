//! DWPose (DreamID-V face-mask) parity vs onnxruntime goldens + end-to-end mask.
//!
//! Two levels:
//!   1. Op parity: run yolox_l and dw-ll_ucoco_384 through the thinfer ONNX
//!      executor on the exact onnxruntime preprocessed inputs, compare the raw
//!      output tensors (this validates the newly added Slice/Split/MatMul/
//!      GlobalAveragePool/ReduceSum/Clip/Sqrt/HardSigmoid ops).
//!   2. Mask e2e: run the full `dwpose::face_mask` on a real frame and compare
//!      the produced face mask to the reference (`extract.py`) mask by IoU.
//!
//! Opt-in (`faceswap-e2e`); skips unless the models + goldens are provided:
//! ```text
//! THINFER_DWPOSE_YOLOX=<yolox_l.onnx> THINFER_DWPOSE_POSE=<dw-ll_ucoco_384.onnx>
//! THINFER_DWPOSE_GOLDEN=<scratch/golden>
//! cargo test -p thinfer-conformance --features faceswap-e2e --test dwpose_onnx -- --nocapture
//! ```
//! Goldens are produced by `scratch/gen_golden.py` + `scratch/rename_golden.py`.

#![cfg(feature = "faceswap-e2e")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::onnx::OnnxModel;
use thinfer_models::faceswap::dwpose::DwPose;
use thinfer_models::faceswap::image::Image;

/// Minimal numpy `.npy` reader for C-order little-endian float32 arrays.
fn read_npy_f32(path: &Path) -> (Vec<i64>, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(&bytes[0..6], b"\x93NUMPY", "not npy: {}", path.display());
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
    assert!(
        header.contains("<f4") || header.contains("|f4"),
        "npy not float32: {header}"
    );
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

/// max abs diff, relative-to-peak, and NaN count between two tensors.
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

fn golden_dir() -> Option<PathBuf> {
    std::env::var("THINFER_DWPOSE_GOLDEN")
        .ok()
        .map(PathBuf::from)
}

/// Per-keypoint argmax over the last dim of a `[1, K, W]` SimCC tensor.
fn simcc_argmax(t: &[f32], k: usize, w: usize) -> Vec<usize> {
    (0..k)
        .map(|j| {
            let s = &t[j * w..j * w + w];
            s.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        })
        .collect()
}

#[test]
fn yolox_parity() {
    let (Some(model), Some(g)) = (std::env::var("THINFER_DWPOSE_YOLOX").ok(), golden_dir()) else {
        eprintln!("yolox_parity skipped");
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
    let (Some(model), Some(g)) = (std::env::var("THINFER_DWPOSE_POSE").ok(), golden_dir()) else {
        eprintln!("dwpose_parity skipped");
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
        let (_gx_shape, got_x) = &out["simcc_x"];
        let (_gy_shape, got_y) = &out["simcc_y"];
        let (ax, rx, nx) = compare(got_x, &exp_x);
        let (ay, ry, ny) = compare(got_y, &exp_y);
        eprintln!("[dwpose] simcc_x max_abs={ax:.4e} rel={rx:.4e} nan={nx}");
        eprintln!("[dwpose] simcc_y max_abs={ay:.4e} rel={ry:.4e} nan={ny}");
        assert_eq!(nx + ny, 0);
        // The mask only depends on the keypoint locations (per-keypoint argmax),
        // so verify those agree exactly even if raw logits differ slightly.
        let (k, wx, wy) = (
            sx_shape[1] as usize,
            sx_shape[2] as usize,
            exp_y.len() / sx_shape[1] as usize,
        );
        let got_lx = simcc_argmax(got_x, k, wx);
        let exp_lx = simcc_argmax(&exp_x, k, wx);
        let got_ly = simcc_argmax(got_y, k, wy);
        let exp_ly = simcc_argmax(&exp_y, k, wy);
        let agree_x = got_lx.iter().zip(&exp_lx).filter(|(a, b)| a == b).count();
        let agree_y = got_ly.iter().zip(&exp_ly).filter(|(a, b)| a == b).count();
        let maxbin_x = got_lx
            .iter()
            .zip(&exp_lx)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        let maxbin_y = got_ly
            .iter()
            .zip(&exp_ly)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        eprintln!(
            "[dwpose] argmax agree x={agree_x}/{k} y={agree_y}/{k}  max bin drift x={maxbin_x} y={maxbin_y}"
        );
        assert!(rx <= 2e-2 && ry <= 2e-2, "dwpose rel x={rx:.4e} y={ry:.4e}");
        assert!(
            maxbin_x <= 1 && maxbin_y <= 1,
            "keypoint argmax drifted >1 bin"
        );
    });
}

#[test]
fn dwpose_mask_e2e() {
    let (Some(yolox), Some(pose), Some(g)) = (
        std::env::var("THINFER_DWPOSE_YOLOX").ok(),
        std::env::var("THINFER_DWPOSE_POSE").ok(),
        golden_dir(),
    ) else {
        eprintln!("dwpose_mask_e2e skipped");
        return;
    };
    pollster::block_on(async {
        let backend = Arc::new(WgpuBackend::new().await.expect("wgpu"));
        let dw = DwPose::load(
            backend,
            &std::fs::read(&yolox).unwrap(),
            &std::fs::read(&pose).unwrap(),
        )
        .await
        .expect("load DwPose");

        let (fshape, fdata) = read_npy_f32(&g.join("frame0_f32.npy")); // [H, W, 3]
        let (h, w) = (fshape[0] as usize, fshape[1] as usize);
        let frame = Image { w, h, data: fdata };
        let (mshape, mref) = read_npy_f32(&g.join("mask0_f32.npy")); // [H, W]
        assert_eq!((mshape[0] as usize, mshape[1] as usize), (h, w));

        let mask = dw.face_mask(&frame).await.expect("face_mask");
        assert_eq!((mask.w, mask.h), (w, h));

        // IoU of the white region (threshold 127) + reference white-pixel recall.
        let (mut inter, mut uni, mut got_wh, mut ref_wh) = (0u64, 0u64, 0u64, 0u64);
        for (m_ref, px) in mref.iter().zip(mask.data.chunks_exact(3)) {
            let a = px[0] > 127.0;
            let b = *m_ref > 127.0;
            got_wh += a as u64;
            ref_wh += b as u64;
            inter += (a && b) as u64;
            uni += (a || b) as u64;
        }
        let iou = inter as f64 / uni.max(1) as f64;
        let recall = inter as f64 / ref_wh.max(1) as f64;
        eprintln!(
            "[dwpose-mask] IoU={iou:.3} recall={recall:.3} white got={got_wh} ref={ref_wh} inter={inter}"
        );
        assert!(ref_wh > 0, "reference mask empty");
        assert!(iou > 0.80, "mask IoU {iou:.3} too low");
    });
}
