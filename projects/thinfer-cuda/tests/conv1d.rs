//! The CUDA f32-act conv1d (plain and transposed, f32 and bf16 weights)
//! against a CPU f64 reference of `torch.nn.functional.conv1d` /
//! `conv_transpose1d`: strides, dilations, odd strides with
//! `output_padding`, shapes that are not tile multiples. Skips when no CUDA
//! device / NVRTC is present.
#![cfg(not(target_os = "macos"))]

use std::sync::Arc;

use cudarc::driver::{CudaContext, sys};
use half::bf16;
use thinfer_core::backend::{Conv1dAddrs, Conv1dShape};
use thinfer_cuda::gemm::GemmKernels;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
}

fn upload(ctx: &Arc<CudaContext>, bytes: &[u8]) -> sys::CUdeviceptr {
    ctx.bind_to_thread().unwrap();
    let mut p = 0;
    unsafe {
        sys::cuMemAlloc_v2(&mut p, bytes.len().max(16))
            .result()
            .unwrap();
        sys::cuMemcpyHtoD_v2(p, bytes.as_ptr().cast(), bytes.len())
            .result()
            .unwrap();
    }
    p
}

fn bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

/// CPU reference (f64 accumulate). `w` `[cout, cin, k]` plain, `[cin, cout,
/// k]` transposed.
fn reference(s: &Conv1dShape, x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    let (cin, cout, lin, lout, k) = (
        s.cin as usize,
        s.cout as usize,
        s.lin as i64,
        s.lout as usize,
        s.k as usize,
    );
    let (st, dil, pad) = (s.stride as i64, s.dilation as i64, s.pad as i64);
    let mut out = vec![0f32; cout * lout];
    for co in 0..cout {
        for lo in 0..lout {
            let mut acc = b[co] as f64;
            for ci in 0..cin {
                for t in 0..k {
                    let (li, wv) = if s.transposed {
                        let num = lo as i64 + pad - t as i64;
                        if num < 0 || num % st != 0 {
                            continue;
                        }
                        (num / st, w[(ci * cout + co) * k + t])
                    } else {
                        (
                            lo as i64 * st + t as i64 * dil - pad,
                            w[(co * cin + ci) * k + t],
                        )
                    };
                    if li >= 0 && li < lin {
                        acc += wv as f64 * x[ci * lin as usize + li as usize] as f64;
                    }
                }
            }
            out[co * lout + lo] = acc as f32;
        }
    }
    out
}

#[test]
fn cuda_conv1d_matches_reference() {
    if !thinfer_cuda::driver_present() {
        eprintln!("skipped[cuda conv1d]: no CUDA driver");
        return;
    }
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skipped[cuda conv1d]: no device");
        return;
    };
    let sm = ctx.compute_capability().unwrap();
    let Ok(kernels) = GemmKernels::new(ctx.clone(), sm) else {
        eprintln!("skipped[cuda conv1d]: no kernels");
        return;
    };
    let mut rng = Rng(5);
    // (cin, cout, lin, k, stride, dilation, pad, transposed, output_padding)
    let cases = [
        (128, 256, 50, 7, 1, 1, 3, false, 0),
        (64, 64, 333, 7, 1, 9, 27, false, 0),
        (96, 48, 200, 1, 1, 1, 0, false, 0),
        (64, 1, 900, 7, 1, 1, 3, false, 0),
        (32, 70, 101, 4, 2, 1, 1, false, 0),
        // DAC up blocks: k = 2s, pad = ceil(s/2), output_padding = s % 2.
        (256, 128, 37, 16, 8, 1, 4, true, 0),
        (128, 64, 41, 10, 5, 1, 3, true, 1),
        (64, 32, 29, 6, 3, 1, 2, true, 1),
        (48, 24, 53, 4, 2, 1, 1, true, 0),
        // The TTS codec's 2x upsamplers (k = stride, no pad).
        (40, 40, 31, 2, 2, 1, 0, true, 0),
    ];
    for (cin, cout, lin, k, stride, dil, pad, transposed, op) in cases {
        let lout = if transposed {
            (lin - 1) * stride + k - 2 * pad + op
        } else {
            (lin + 2 * pad - dil * (k - 1) - 1) / stride + 1
        };
        let x: Vec<f32> = (0..cin * lin).map(|_| rng.next()).collect();
        let w: Vec<f32> = (0..cin * cout * k).map(|_| rng.next() * 0.2).collect();
        let b: Vec<f32> = (0..cout).map(|_| rng.next()).collect();
        for w_bf16 in [false, true] {
            let shape = Conv1dShape {
                cin: cin as u32,
                cout: cout as u32,
                lin: lin as u32,
                lout: lout as u32,
                k: k as u32,
                stride: stride as u32,
                dilation: dil as u32,
                pad: pad as u32,
                transposed,
                w_bf16,
            };
            let (wq, bq): (Vec<f32>, Vec<f32>) = if w_bf16 {
                (
                    w.iter().map(|&v| bf16::from_f32(v).to_f32()).collect(),
                    b.iter().map(|&v| bf16::from_f32(v).to_f32()).collect(),
                )
            } else {
                (w.clone(), b.clone())
            };
            let want = reference(&shape, &x, &wq, &bq);
            let (pw, pb) = if w_bf16 {
                let w16: Vec<bf16> = w.iter().map(|&v| bf16::from_f32(v)).collect();
                let b16: Vec<bf16> = b.iter().map(|&v| bf16::from_f32(v)).collect();
                (upload(&ctx, bytes(&w16)), upload(&ctx, bytes(&b16)))
            } else {
                (upload(&ctx, bytes(&w)), upload(&ctx, bytes(&b)))
            };
            let px = upload(&ctx, bytes(&x));
            let po = upload(&ctx, &vec![0xffu8; want.len() * 4]);
            let ran = kernels
                .run_conv1d(&Conv1dAddrs {
                    x: px,
                    w: pw,
                    bias: pb,
                    out: po,
                    shape,
                })
                .unwrap();
            assert!(ran, "declined {shape:?}");
            let mut got = vec![0f32; want.len()];
            unsafe { sys::cuMemcpyDtoH_v2(got.as_mut_ptr().cast(), po, got.len() * 4) }
                .result()
                .unwrap();
            for p in [px, pw, pb, po] {
                unsafe { sys::cuMemFree_v2(p) }.result().unwrap();
            }
            let (mut e2, mut r2, mut worst) = (0f64, 0f64, 0f64);
            for (g, w) in got.iter().zip(&want) {
                let d = (g - w) as f64;
                e2 += d * d;
                r2 += (*w as f64).powi(2);
                worst = worst.max(d.abs() / (w.abs() as f64 + 1e-3));
            }
            let rel = (e2 / r2).sqrt();
            eprintln!("cuda conv1d {shape:?}: rel_rms {rel:.3e} worst {worst:.3e}");
            assert!(
                got.iter().all(|v| v.is_finite()),
                "unwritten output {shape:?}"
            );
            assert!(rel < 1e-5, "{shape:?}: rel_rms {rel}");
        }
    }
}
