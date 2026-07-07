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
    assert_eq!(
        &bytes[0..6],
        b"\x93NUMPY",
        "not an npy file: {}",
        path.display()
    );
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
    assert_eq!(
        got.len(),
        exp.len(),
        "length mismatch {} vs {}",
        got.len(),
        exp.len()
    );
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
    Stats {
        max_abs,
        rel: max_abs / (max_ref + 1e-6),
        nan,
    }
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
        let (got_shape, got_data) = got
            .get(name)
            .unwrap_or_else(|| panic!("missing output {name}"));
        assert_eq!(got_shape, exp_shape, "output {name} shape");
        let s = compare(got_data, exp);
        eprintln!(
            "[{tag}] output {name}: max_abs={:.5e} rel={:.5e} nan={} shape={:?}",
            s.max_abs, s.rel, s.nan, got_shape
        );
        assert_eq!(s.nan, 0, "output {name} has non-finite values");
        worst = worst.max(s.rel);
    }
    assert!(
        worst <= tol,
        "[{tag}] worst rel {worst:.5e} exceeds tol {tol:.1e}"
    );
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

/// XSeg occlusion mask parity. Exercises the ops added for XSeg (Pad, Max/Min,
/// Reciprocal) + the NHWC input path. Looser tol than the pure conv models: the
/// GlobalAveragePool -> Sqrt -> Reciprocal norm chain compounds fp order.
#[test]
fn xseg_parity() {
    let ran = pollster::block_on(async {
        let backend = std::sync::Arc::new(WgpuBackend::new().await.expect("wgpu"));
        run_model(backend, "THINFER_FS_XSEG", "xseg", 3e-2).await
    });
    if !ran {
        eprintln!("xseg_parity skipped");
    }
}

/// GFPGAN face-enhancer parity. Exercises the ops added for GFPGAN (Pow,
/// ConstantOfShape, Tile) + the StyleGAN weight-demod chain. Deep generator, so
/// a looser tol (the Pow/ReduceSum/Sqrt/Div demod compounds fp over 95 convs).
#[test]
fn gfpgan_parity() {
    let ran = pollster::block_on(async {
        let backend = std::sync::Arc::new(WgpuBackend::new().await.expect("wgpu"));
        run_model(backend, "THINFER_FS_GFPGAN", "gfpgan", 5e-2).await
    });
    if !ran {
        eprintln!("gfpgan_parity skipped");
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

/// Elementwise-chain fusion parity: the fused path must match the per-op path to
/// f32 noise. Arithmetic ops are identical (f32 buffer round-trips are lossless);
/// the residual ~1e-5 is transcendental functions (exp/sigmoid/tanh) compiling
/// slightly differently in the fused vs standalone shader - not a broadcast bug
/// (which would perturb many elements by more). `THINFER_ONNX_NO_FUSION` toggles
/// fusion off for the reference run. Needs only THINFER_FS_HYPERSWAP.
#[test]
fn hyperswap_fusion_parity() {
    let ran = pollster::block_on(async {
        let Ok(model_path) = std::env::var("THINFER_FS_HYPERSWAP") else {
            return false;
        };
        let onnx = std::fs::read(&model_path).unwrap();
        let backend = std::sync::Arc::new(WgpuBackend::new().await.expect("wgpu"));

        // Identify inputs by rank (4D target image, 2D source embedding).
        let g = thinfer_core::onnx::proto::parse_model(&onnx).unwrap();
        let ins: Vec<(String, usize)> = g
            .inputs
            .iter()
            .filter(|vi| !g.initializers.iter().any(|t| t.name == vi.name))
            .map(|vi| (vi.name.clone(), vi.dims.len()))
            .collect();
        let tname = ins.iter().find(|(_, r)| *r == 4).unwrap().0.clone();
        let sname = ins.iter().find(|(_, r)| *r == 2).unwrap().0.clone();
        let shapes = {
            let mut m = HashMap::new();
            m.insert(tname.clone(), vec![1i64, 3, 256, 256]);
            m.insert(sname.clone(), vec![1i64, 512]);
            m
        };
        let mut feeds = HashMap::new();
        feeds.insert(
            tname.clone(),
            (0..3 * 256 * 256)
                .map(|i| ((i % 509) as f32 / 509.0) * 2.0 - 1.0)
                .collect::<Vec<f32>>(),
        );
        feeds.insert(sname.clone(), vec![1.0 / (512f32).sqrt(); 512]);
        let out4 = |o: &HashMap<String, (Vec<i64>, Vec<f32>)>| {
            o.values()
                .find(|(sh, _)| sh.len() == 4 && sh[1] == 3)
                .unwrap()
                .1
                .clone()
        };

        // Reference: fusion off.
        unsafe { std::env::set_var("THINFER_ONNX_NO_FUSION", "1") };
        let m_ref = OnnxModel::load(std::sync::Arc::clone(&backend), &onnx, &shapes)
            .await
            .unwrap();
        unsafe { std::env::remove_var("THINFER_ONNX_NO_FUSION") };
        let m_fused = OnnxModel::load(std::sync::Arc::clone(&backend), &onnx, &shapes)
            .await
            .unwrap();
        let o_ref = m_ref.run(&feeds).await.unwrap();
        let o_fused = m_fused.run(&feeds).await.unwrap();

        let s = compare(&out4(&o_fused), &out4(&o_ref));
        eprintln!(
            "[fusion] rel_vs_unfused={:.3e} max_abs={:.3e} nan={}",
            s.rel, s.max_abs, s.nan
        );
        assert_eq!(s.nan, 0, "fused output has NaNs");
        assert!(s.rel <= 1e-4, "fused output diverged: {:.3e}", s.rel);
        true
    });
    if !ran {
        eprintln!("hyperswap_fusion_parity skipped (set THINFER_FS_HYPERSWAP)");
    }
}

/// Clean full-forward wall time of one HyperSwap forward (synthetic inputs, no
/// golden). Warms up, then times N forwards single-submit. The A/B harness for
/// conv-path work. Run: THINFER_FS_HYPERSWAP=<...> cargo test --features
/// faceswap-e2e --release hyperswap_forward_timing -- --ignored --nocapture.
#[test]
#[ignore]
fn hyperswap_forward_timing() {
    pollster::block_on(async {
        let model_path = std::env::var("THINFER_FS_HYPERSWAP").expect("THINFER_FS_HYPERSWAP");
        let onnx = std::fs::read(&model_path).unwrap();
        // Time on the discrete GPU (the production device): plain
        // WgpuBackend::new() defaults to PowerPreference::None -> the iGPU.
        let cfg = thinfer_core::backend::WgpuConfig {
            power_preference: thinfer_core::backend::PowerPreference::HighPerformance,
            ..Default::default()
        };
        let backend = std::sync::Arc::new(WgpuBackend::new_with_config(cfg).await.expect("wgpu"));
        let g = thinfer_core::onnx::proto::parse_model(&onnx).unwrap();
        let ins: Vec<(String, usize)> = g
            .inputs
            .iter()
            .filter(|vi| !g.initializers.iter().any(|t| t.name == vi.name))
            .map(|vi| (vi.name.clone(), vi.dims.len()))
            .collect();
        let tname = ins.iter().find(|(_, r)| *r == 4).unwrap().0.clone();
        let sname = ins.iter().find(|(_, r)| *r == 2).unwrap().0.clone();
        // Batch sweep: does stacking N frames into one forward amortize the
        // per-dispatch GPU underutilization (sublinear) or is it saturated
        // (linear)? THINFER_FS_BATCH=1,2,4,8.
        let batch: i64 = std::env::var("THINFER_FS_BATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let mut shapes = HashMap::new();
        shapes.insert(tname.clone(), vec![batch, 3, 256, 256]);
        shapes.insert(sname.clone(), vec![batch, 512]);
        let mut feeds = HashMap::new();
        feeds.insert(
            tname,
            (0..batch as usize * 3 * 256 * 256)
                .map(|i| ((i % 509) as f32 / 509.0) * 2.0 - 1.0)
                .collect::<Vec<f32>>(),
        );
        feeds.insert(sname, vec![1.0 / (512f32).sqrt(); batch as usize * 512]);
        let m = OnnxModel::load(std::sync::Arc::clone(&backend), &onnx, &shapes)
            .await
            .unwrap();
        for _ in 0..5 {
            m.run(&feeds).await.unwrap();
        }
        let iters = 30;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            std::hint::black_box(m.run(&feeds).await.unwrap());
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        eprintln!(
            "[hyperswap-timing] batch={batch} mean_forward={ms:.2}ms \
             ({:.2}ms/frame) over {iters} iters",
            ms / batch as f64
        );
    });
}

/// i8 DP4A conv A/B: load HyperSwap f32 and i8 (THINFER_ONNX_I8_CONV), run both
/// on the same input, report the output rel error + both timings. Numeric sanity
/// for the opt-in i8 path (the real gate is a visual eyeball on a face). Run
/// --ignored --nocapture with THINFER_FS_HYPERSWAP set.
#[test]
#[ignore]
fn hyperswap_i8_ab() {
    pollster::block_on(async {
        let model_path = std::env::var("THINFER_FS_HYPERSWAP").expect("THINFER_FS_HYPERSWAP");
        let onnx = std::fs::read(&model_path).unwrap();
        let cfg = thinfer_core::backend::WgpuConfig {
            power_preference: thinfer_core::backend::PowerPreference::HighPerformance,
            ..Default::default()
        };
        let backend = std::sync::Arc::new(WgpuBackend::new_with_config(cfg).await.expect("wgpu"));
        let g = thinfer_core::onnx::proto::parse_model(&onnx).unwrap();
        let ins: Vec<(String, usize)> = g
            .inputs
            .iter()
            .filter(|vi| !g.initializers.iter().any(|t| t.name == vi.name))
            .map(|vi| (vi.name.clone(), vi.dims.len()))
            .collect();
        let tname = ins.iter().find(|(_, r)| *r == 4).unwrap().0.clone();
        let sname = ins.iter().find(|(_, r)| *r == 2).unwrap().0.clone();
        let mut shapes = HashMap::new();
        shapes.insert(tname.clone(), vec![1i64, 3, 256, 256]);
        shapes.insert(sname.clone(), vec![1i64, 512]);
        let mut feeds = HashMap::new();
        // A smoothly varying target (closer to a face than white noise).
        feeds.insert(
            tname.clone(),
            (0..3 * 256 * 256)
                .map(|i| {
                    let (c, p) = (i / (256 * 256), i % (256 * 256));
                    let (y, x) = ((p / 256) as f32, (p % 256) as f32);
                    ((x / 256.0 * 6.0).sin() * (y / 256.0 * 6.0).cos() * 0.5 + 0.1 * c as f32)
                        .clamp(-1.0, 1.0)
                })
                .collect::<Vec<f32>>(),
        );
        feeds.insert(sname.clone(), vec![1.0 / (512f32).sqrt(); 512]);
        let out4 = |o: &HashMap<String, (Vec<i64>, Vec<f32>)>| {
            o.values()
                .find(|(sh, _)| sh.len() == 4 && sh[1] == 3)
                .unwrap()
                .1
                .clone()
        };

        let m_f32 = OnnxModel::load(std::sync::Arc::clone(&backend), &onnx, &shapes)
            .await
            .unwrap();
        let o_f32 = m_f32.run(&feeds).await.unwrap();
        for _ in 0..3 {
            m_f32.run(&feeds).await.unwrap();
        }
        let t = std::time::Instant::now();
        for _ in 0..20 {
            std::hint::black_box(m_f32.run(&feeds).await.unwrap());
        }
        let t_f32 = t.elapsed().as_secs_f64() * 1e3 / 20.0;

        let m_i8 = OnnxModel::load_i8(std::sync::Arc::clone(&backend), &onnx, &shapes, true)
            .await
            .unwrap();
        let o_i8 = m_i8.run(&feeds).await.unwrap();
        for _ in 0..3 {
            m_i8.run(&feeds).await.unwrap();
        }
        let t = std::time::Instant::now();
        for _ in 0..20 {
            std::hint::black_box(m_i8.run(&feeds).await.unwrap());
        }
        let t_i8 = t.elapsed().as_secs_f64() * 1e3 / 20.0;

        let (a, b) = (out4(&o_f32), out4(&o_i8));
        let mut sq_err = 0f64;
        let mut sq_ref = 0f64;
        let mut max_abs = 0f32;
        let mut nan = 0;
        for (&x, &y) in a.iter().zip(&b) {
            if !y.is_finite() {
                nan += 1;
                continue;
            }
            sq_err += ((x - y) as f64).powi(2);
            sq_ref += (x as f64).powi(2);
            max_abs = max_abs.max((x - y).abs());
        }
        let rel = (sq_err / sq_ref).sqrt();
        eprintln!(
            "[i8-ab] rel_rmse={rel:.3e} max_abs={max_abs:.3e} (out in [-1,1]) nan={nan} | \
             f32={t_f32:.1}ms i8={t_i8:.1}ms speedup={:.2}x",
            t_f32 / t_i8
        );
        assert_eq!(nan, 0, "i8 output has non-finite values");
    });
}

/// Batch spike: run HyperSwap with target `[B,3,256,256]` (B copies of the
/// golden target) + source `[1,512]`, assert every output slice matches the
/// batch-1 golden, and report per-crop wall time vs batch 1. Validates that
/// batching crops is numerically identical and measures the occupancy win.
#[test]
fn hyperswap_batch_spike() {
    let ran = pollster::block_on(async {
        let (Ok(model_path), Ok(golden_dir)) = (
            std::env::var("THINFER_FS_HYPERSWAP"),
            std::env::var("THINFER_FS_GOLDEN"),
        ) else {
            return false;
        };
        let g = PathBuf::from(golden_dir);
        let (t_shape, t_data) = read_npy_f32(&g.join("hyperswap_in_target.npy"));
        let (s_shape, s_data) = read_npy_f32(&g.join("hyperswap_in_source.npy"));
        let (_o_shape, o_data) = read_npy_f32(&g.join("hyperswap_out_output.npy"));
        let onnx = std::fs::read(&model_path).unwrap();
        let backend = std::sync::Arc::new(WgpuBackend::new().await.expect("wgpu"));

        const B: i64 = 8;
        let per = t_data.len();
        // Names from a batch-1 load.
        let m1 = OnnxModel::load(std::sync::Arc::clone(&backend), &onnx, &{
            let mut m = HashMap::new();
            m.insert("target".into(), t_shape.clone());
            m.insert("source".into(), s_shape.clone());
            m
        })
        .await
        .expect("load b1");
        let tname = m1
            .input_names
            .iter()
            .find(|n| n.contains("target"))
            .cloned()
            .unwrap_or_else(|| "target".into());
        let sname = m1
            .input_names
            .iter()
            .find(|n| n.contains("source"))
            .cloned()
            .unwrap_or_else(|| "source".into());

        // Time batch-1 (5 runs).
        let mut f1 = HashMap::new();
        f1.insert(tname.clone(), t_data.clone());
        f1.insert(sname.clone(), s_data.clone());
        let _ = m1.run(&f1).await.unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..5 {
            let _ = m1.run(&f1).await.unwrap();
        }
        let ms1 = t0.elapsed().as_secs_f64() * 1e3 / 5.0;

        // Batch B.
        let mut bt = t_shape.clone();
        bt[0] = B;
        let mb = OnnxModel::load(std::sync::Arc::clone(&backend), &onnx, &{
            let mut m = HashMap::new();
            m.insert("target".into(), bt.clone());
            m.insert("source".into(), s_shape.clone());
            m
        })
        .await
        .expect("load bB");
        let mut tiled = Vec::with_capacity(per * B as usize);
        for _ in 0..B {
            tiled.extend_from_slice(&t_data);
        }
        let mut fb = HashMap::new();
        fb.insert(tname, tiled);
        fb.insert(sname, s_data.clone());
        let outb = mb.run(&fb).await.unwrap();
        // HyperSwap has two 4D outputs (the 3-ch image and the 1-ch mask); the
        // batch check is on the image output.
        let (osh, od) = outb
            .values()
            .find(|(sh, _)| sh.len() == 4 && sh[1] == 3)
            .unwrap();
        assert_eq!(osh[0], B);
        // Each batch slice equals the golden.
        let mut worst = 0f32;
        for b in 0..B as usize {
            let s = compare(&od[b * per..(b + 1) * per], &o_data);
            worst = worst.max(s.rel);
            assert_eq!(s.nan, 0);
        }
        let t0 = std::time::Instant::now();
        for _ in 0..3 {
            let _ = mb.run(&fb).await.unwrap();
        }
        let msb = t0.elapsed().as_secs_f64() * 1e3 / 3.0;
        eprintln!(
            "[batch] b1={ms1:.1}ms/crop  b{B}={:.1}ms/crop ({:.1}ms total)  speedup={:.2}x  worst_rel={worst:.2e}",
            msb / B as f64,
            msb,
            ms1 / (msb / B as f64),
        );
        assert!(worst <= 6e-2, "batch output diverged: {worst:.2e}");
        true
    });
    if !ran {
        eprintln!("hyperswap_batch_spike skipped");
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
