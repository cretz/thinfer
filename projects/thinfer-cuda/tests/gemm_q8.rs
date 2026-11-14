//! The 8-bit tensor-core GEMM (`gemm_i8` / `gemm_f8`) and its per-row
//! quantizers: INT8 against a CPU evaluation of the same formula on the
//! GPU-quantized operands (exact int32 dot, same f32 epilogue order), FP8
//! against an f64 dot of the decoded e4m3 operands (f32 accumulation order
//! differs), and both whole quantized products against the f32 one. Skips
//! without CUDA (FP8 below sm_89). `THINFER_GEMM_BENCH=1` adds timings at
//! MiniMax-H3 block shapes.
#![cfg(not(target_os = "macos"))]

use std::sync::Arc;

use cudarc::driver::{CudaContext, sys};
use half::bf16;
use thinfer_cuda::gemm::{GemmKernels, GemmQ8Addrs, Q8};

fn rand(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn alloc(ctx: &Arc<CudaContext>, bytes: usize) -> sys::CUdeviceptr {
    ctx.bind_to_thread().unwrap();
    let mut p = 0;
    unsafe { sys::cuMemAlloc_v2(&mut p, bytes.max(16)).result().unwrap() };
    p
}

fn upload<T>(ctx: &Arc<CudaContext>, v: &[T]) -> sys::CUdeviceptr {
    let n = std::mem::size_of_val(v);
    let p = alloc(ctx, n);
    unsafe {
        sys::cuMemcpyHtoD_v2(p, v.as_ptr().cast(), n)
            .result()
            .unwrap()
    };
    p
}

fn download<T: Clone + Default>(p: sys::CUdeviceptr, n: usize) -> Vec<T> {
    let mut v = vec![T::default(); n];
    unsafe {
        sys::cuCtxSynchronize().result().unwrap();
        sys::cuMemcpyDtoH_v2(v.as_mut_ptr().cast(), p, n * std::mem::size_of::<T>())
            .result()
            .unwrap();
    }
    v
}

fn setup() -> Option<(Arc<CudaContext>, GemmKernels, (i32, i32))> {
    if !thinfer_cuda::driver_present() {
        eprintln!("gemm_q8: no CUDA driver, skipped");
        return None;
    }
    let ctx = CudaContext::new(0).ok()?;
    let sm = ctx.compute_capability().ok()?;
    let k = GemmKernels::new(ctx.clone(), sm).ok()?;
    Some((ctx, k, sm))
}

/// An e4m3 byte (bias 7, no infinities, 0x7f / 0xff NaN) as f32.
fn e4m3(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let (e, m) = ((b >> 3) & 0xf, (b & 7) as f32);
    sign * match e {
        0 => m / 8.0 * 2f32.powi(-6),
        _ => (1.0 + m / 8.0) * 2f32.powi(e as i32 - 7),
    }
}

/// The format's code as f32, and its max.
fn decode(format: Q8, b: u8) -> f32 {
    match format {
        Q8::I8 => b as i8 as f32,
        Q8::E4m3 => e4m3(b),
    }
}

fn q8_max(format: Q8) -> f32 {
    match format {
        Q8::I8 => 127.0,
        Q8::E4m3 => 16.0,
    }
}

fn check(format: Q8) {
    let Some((ctx, kernels, sm)) = setup() else {
        return;
    };
    if format == Q8::E4m3 && sm < (8, 9) {
        eprintln!("gemm_q8: FP8 needs sm_89+, skipped");
        return;
    }
    // Not tile multiples in m / n; k a multiple of 16.
    let (m, n, k) = (200usize, 136usize, 336usize);
    // Outlier channels in the activations (the attention / SwiGLU outputs
    // FP8 serves) and a few large weights per row, like real projections.
    let a: Vec<bf16> = rand(m * k, 1)
        .into_iter()
        .enumerate()
        .map(|(i, x)| bf16::from_f32(if i % k < 3 { x * 30.0 } else { x }))
        .collect();
    let w: Vec<bf16> = rand(n * k, 2)
        .into_iter()
        .enumerate()
        .map(|(i, x)| bf16::from_f32(if i % 97 == 0 { x * 8.0 } else { x }))
        .collect();
    let bias: Vec<bf16> = rand(n, 3).into_iter().map(bf16::from_f32).collect();
    let (pa, pw, pbias) = (upload(&ctx, &a), upload(&ctx, &w), upload(&ctx, &bias));
    let (qa, sa) = (alloc(&ctx, m * k), alloc(&ctx, m * 4));
    let (qw, sw) = (alloc(&ctx, n * k), alloc(&ctx, n * 4));
    let out = alloc(&ctx, m * n * 2);
    assert!(
        kernels
            .quant_rows_q8(format, pa, qa, sa, m as u32, k as u32)
            .unwrap()
    );
    assert!(
        kernels
            .quant_rows_q8(format, pw, qw, sw, n as u32, k as u32)
            .unwrap()
    );
    let addrs = GemmQ8Addrs {
        format,
        a: qa,
        sa,
        b: qw,
        sb: sw,
        out,
        bias: pbias,
    };
    assert!(
        kernels
            .run_q8(&addrs, m as u32, n as u32, k as u32)
            .unwrap()
    );
    let got: Vec<u16> = download(out, m * n);
    let (qa_h, sa_h): (Vec<u8>, Vec<f32>) = (download(qa, m * k), download(sa, m));
    let (qw_h, sw_h): (Vec<u8>, Vec<f32>) = (download(qw, n * k), download(sw, n));
    // The quantizer: every row's max maps to the format's max.
    for r in 0..m {
        let row = &a[r * k..][..k];
        let mx = row.iter().fold(0f32, |x, v| x.max(v.to_f32().abs()));
        assert_eq!(sa_h[r], mx / q8_max(format), "row {r} scale");
        for (i, v) in row.iter().enumerate() {
            let back = decode(format, qa_h[r * k + i]) * sa_h[r];
            let tol = match format {
                Q8::I8 => sa_h[r] * 0.5,
                // 3 mantissa bits, or the subnormal step.
                Q8::E4m3 => (v.to_f32().abs() / 16.0).max(sa_h[r] * 2f32.powi(-10)),
            };
            assert!(
                (back - v.to_f32()).abs() <= tol * 1.0001,
                "row {r} col {i}: {} -> {back}",
                v.to_f32()
            );
        }
    }
    let (mut err2, mut ref2, mut kern2) = (0f64, 0f64, 0f64);
    for r in 0..m {
        for c in 0..n {
            let dot: f64 = (0..k)
                .map(|i| {
                    decode(format, qa_h[r * k + i]) as f64 * decode(format, qw_h[c * k + i]) as f64
                })
                .sum();
            let got_f = bf16::from_bits(got[r * n + c]).to_f32();
            match format {
                Q8::I8 => {
                    let want = bf16::from_f32(dot as f32 * sa_h[r] * sw_h[c] + bias[c].to_f32());
                    assert_eq!(got[r * n + c], want.to_bits(), "r {r} c {c}");
                }
                Q8::E4m3 => {
                    let want = dot * sa_h[r] as f64 * sw_h[c] as f64 + bias[c].to_f32() as f64;
                    kern2 += (got_f as f64 - want).powi(2);
                }
            }
            let exact: f32 = (0..k)
                .map(|i| a[r * k + i].to_f32() * w[c * k + i].to_f32())
                .sum::<f32>()
                + bias[c].to_f32();
            let d = (got_f - exact) as f64;
            err2 += d * d;
            ref2 += (exact as f64) * (exact as f64);
        }
    }
    let rel = (err2 / ref2).sqrt();
    eprintln!("gemm_q8 {format:?} {m}x{n}x{k}: rel_rms vs f32 {rel:.2e}");
    // FP8 against its own operands: only the bf16 output rounding and the
    // f32 accumulation order differ.
    let kern = (kern2 / ref2).sqrt();
    assert!(kern < 4e-3, "FP8 kernel vs its operands rel_rms {kern}");
    assert!(rel < 4e-2, "quantized product rel_rms {rel}");
    for p in [pa, pw, pbias, qa, sa, qw, sw, out] {
        unsafe { sys::cuMemFree_v2(p) }.result().unwrap();
    }
}

#[test]
fn cuda_gemm_i8_matches_reference() {
    check(Q8::I8);
}

#[test]
fn cuda_gemm_f8_matches_reference() {
    check(Q8::E4m3);
}

#[test]
fn cuda_gemm_q8_bench() {
    if std::env::var_os("THINFER_GEMM_BENCH").is_none() {
        return;
    }
    let Some((ctx, kernels, sm)) = setup() else {
        return;
    };
    let (max_m, max_n, max_k) = (14912usize, 28672usize, 14336usize);
    // Codes valid in both formats (int8 3 / -2, e4m3 0.0273 / -0.0234).
    let qa = upload(&ctx, &vec![3u8; max_m * max_k]);
    let qb = upload(&ctx, &vec![0x8au8; max_n * max_k]);
    let sa = upload(&ctx, &vec![0.01f32; max_m]);
    let sb = upload(&ctx, &vec![0.02f32; max_n]);
    let out = alloc(&ctx, max_m * max_n * 2);
    let xa = upload(&ctx, &vec![0x3c00u16; max_m * max_k]);
    let formats: &[Q8] = if sm >= (8, 9) {
        &[Q8::I8, Q8::E4m3]
    } else {
        &[Q8::I8]
    };
    // MiniMax-H3 block sites at 832x480 5s (14891 rows): qkv, gate, fc1, out, fc2.
    for &format in formats {
        for (n, k) in [
            (21504usize, 5376usize),
            (7168, 5376),
            (28672, 5376),
            (5376, 7168),
            (5376, 14336),
        ] {
            let m = 14891usize;
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t = std::time::Instant::now();
                assert!(
                    kernels
                        .run_q8(
                            &GemmQ8Addrs {
                                format,
                                a: qa,
                                sa,
                                b: qb,
                                sb,
                                out,
                                bias: 0,
                            },
                            m as u32,
                            n as u32,
                            k as u32,
                        )
                        .unwrap()
                );
                unsafe { sys::cuCtxSynchronize() }.result().unwrap();
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            let t = std::time::Instant::now();
            kernels
                .quant_rows_q8(format, xa, qa, sa, m as u32, k as u32)
                .unwrap();
            unsafe { sys::cuCtxSynchronize() }.result().unwrap();
            let q_ms = t.elapsed().as_secs_f64() * 1e3;
            let tops = 2.0 * (m * n * k) as f64 / best / 1e9;
            eprintln!(
                "gemm_q8 {format:?} m={m} n={n} k={k}: {best:.2} ms ({tops:.1} TOP/s), act quant {q_ms:.2} ms"
            );
        }
    }
    for p in [qa, qb, sa, sb, out, xa] {
        unsafe { sys::cuMemFree_v2(p) }.result().unwrap();
    }
}
