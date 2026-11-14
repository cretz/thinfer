//! The CUDA dense matmul against a CPU f32 reference with the wgpu matmul
//! contract: every supported (A type, B type, B layout) variant, shapes that are not
//! tile multiples, and the bf16 -> f16 overflow decline. Skips when no CUDA
//! device / NVRTC is present.
#![cfg(not(target_os = "macos"))]

use std::sync::Arc;

use cudarc::driver::{CudaContext, sys};
use half::{bf16, f16};
use thinfer_core::backend::GemmAddrs;
use thinfer_core::ops::{GemmDesc, GemmType};
use thinfer_cuda::gemm::{GemmDims, GemmKernels};

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

/// Round-trip `x` through `ty`; returns (value as stored, raw bits).
fn store(ty: GemmType, x: f32) -> (f32, u16) {
    match ty {
        GemmType::F16 => {
            let h = f16::from_f32(x);
            (h.to_f32(), h.to_bits())
        }
        GemmType::Bf16 => {
            let h = bf16::from_f32(x);
            (h.to_f32(), h.to_bits())
        }
        GemmType::F32 => unreachable!("16-bit operands only"),
    }
}

fn load(ty: GemmType, bits: u16) -> f32 {
    match ty {
        GemmType::F16 => f16::from_bits(bits).to_f32(),
        GemmType::Bf16 => bf16::from_bits(bits).to_f32(),
        GemmType::F32 => unreachable!("16-bit operands only"),
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

struct Case {
    desc: GemmDesc,
    m: usize,
    n: usize,
    k: usize,
    /// One B value forced to this (overflow probe).
    poke: Option<f32>,
}

/// Run one case; `None` when the kernel declined.
fn run(ctx: &Arc<CudaContext>, kernels: &GemmKernels, c: &Case, rng: &mut Rng) -> Option<f32> {
    let (m, n, k) = (c.m, c.n, c.k);
    let (a_val, a_bits): (Vec<f32>, Vec<u16>) =
        (0..m * k).map(|_| store(c.desc.a, rng.next())).unzip();
    // Logical B[k][n].
    let mut b_log: Vec<f32> = (0..k * n).map(|_| rng.next() * 0.1).collect();
    if let Some(p) = c.poke {
        b_log[k * n / 2] = p;
    }
    let b_val: Vec<f32> = b_log.iter().map(|&x| store(c.desc.b, x).0).collect();
    let b_bits: Vec<u16> = if c.desc.b_nmajor {
        (0..n * k)
            .map(|i| store(c.desc.b, b_log[(i % k) * n + i / k]).1)
            .collect()
    } else {
        b_log.iter().map(|&x| store(c.desc.b, x).1).collect()
    };
    let dims = GemmDims {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        pad: 0,
    };
    let (pa, pb, pd) = (
        upload(ctx, bytes(&a_bits)),
        upload(ctx, bytes(&b_bits)),
        upload(ctx, bytes(std::slice::from_ref(&dims))),
    );
    let out_bytes = if c.desc.out_f32 { 4 } else { 2 };
    let po = upload(ctx, &vec![0u8; m * n * out_bytes]);
    // k = 0: read K from the dims buffer (the uniform path).
    let ran = kernels
        .run(
            &c.desc,
            &GemmAddrs {
                a: pa,
                b: pb,
                out: po,
                dims: pd,
                m: m as u32,
                n: n as u32,
                k: 0,
                epilogue: Default::default(),
                checked: true,
            },
        )
        .unwrap();
    let mut raw = vec![0u8; m * n * out_bytes];
    unsafe { sys::cuMemcpyDtoH_v2(raw.as_mut_ptr().cast(), po, raw.len()) }
        .result()
        .unwrap();
    let got: Vec<f32> = if c.desc.out_f32 {
        raw.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    } else {
        raw.chunks_exact(2)
            .map(|b| load(c.desc.a, u16::from_le_bytes([b[0], b[1]])))
            .collect()
    };
    for p in [pa, pb, pd, po] {
        unsafe { sys::cuMemFree_v2(p) }.result().unwrap();
    }
    if !ran {
        return None;
    }
    let (mut num, mut den) = (0f64, 0f64);
    for i in 0..m {
        for j in 0..n {
            let w: f32 = (0..k).map(|kk| a_val[i * k + kk] * b_val[kk * n + j]).sum();
            let e = got[i * n + j] - w;
            num += (e * e) as f64;
            den += (w * w) as f64;
        }
    }
    Some((num / den).sqrt() as f32)
}

#[test]
fn cuda_gemm_matches_reference() {
    if !thinfer_cuda::driver_present() {
        eprintln!("skipped[cuda gemm]: no CUDA driver");
        return;
    }
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipped[cuda gemm]: {e:?}");
            return;
        }
    };
    let sm = ctx.compute_capability().unwrap();
    let kernels = match GemmKernels::new(ctx.clone(), sm) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("skipped[cuda gemm]: {e}");
            return;
        }
    };
    let mut rng = Rng(11);
    let types = [GemmType::F16, GemmType::Bf16];
    for a in types {
        for b in types {
            for (b_nmajor, out_f32) in [(false, false), (true, false), (false, true), (true, true)]
            {
                let desc = GemmDesc {
                    a,
                    b,
                    b_nmajor,
                    out_f32,
                };
                if thinfer_cuda::gemm::index(&desc).is_none() {
                    continue;
                }
                // Ragged M/N/K (not tile multiples) and an exact-tile shape.
                // 1100 x 520: 9 x 5 tiles, past one 8-row-tile raster group, ragged.
                for (m, n, k) in [(200, 136, 72), (256, 256, 256), (1100, 520, 264)] {
                    let c = Case {
                        desc,
                        m,
                        n,
                        k,
                        poke: None,
                    };
                    let rel = run(&ctx, &kernels, &c, &mut rng)
                        .unwrap_or_else(|| panic!("declined {desc:?} {m}x{n}x{k}"));
                    eprintln!("cuda gemm {desc:?} {m}x{n}x{k}: rel_l2 {rel:.2e}");
                    // Output rounding in A's dtype dominates: bf16 ~4e-3, f16 ~5e-4.
                    let tol = if a == GemmType::Bf16 { 8e-3 } else { 2e-3 };
                    assert!(rel < tol, "{desc:?} {m}x{n}x{k}: rel_l2 {rel:.2e}");
                }
            }
        }
    }
    // A bf16 weight past the f16 range on an f16-act call must decline.
    let c = Case {
        desc: GemmDesc {
            a: GemmType::F16,
            b: GemmType::Bf16,
            b_nmajor: false,
            out_f32: false,
        },
        m: 128,
        n: 64,
        k: 64,
        poke: Some(1.0e5),
    };
    assert!(
        run(&ctx, &kernels, &c, &mut rng).is_none(),
        "overflowing bf16 weight did not decline"
    );
}

/// The f32-act form (bf16 n-major B, f32 out, M <= 64) against an f64 CPU
/// reference: ragged rows / columns / K strips, one row, and the full block;
/// past 64 rows (or an epilogue) it declines.
#[test]
fn cuda_gemm_f32_acts() {
    if !thinfer_cuda::driver_present() {
        eprintln!("skipped[cuda gemm f32]: no CUDA driver");
        return;
    }
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skipped[cuda gemm f32]: no device");
        return;
    };
    let sm = ctx.compute_capability().unwrap();
    let Ok(kernels) = GemmKernels::new(ctx.clone(), sm) else {
        eprintln!("skipped[cuda gemm f32]: no kernels");
        return;
    };
    let desc = GemmDesc {
        a: GemmType::F32,
        b: GemmType::Bf16,
        b_nmajor: true,
        out_f32: false,
    };
    let mut rng = Rng(5);
    for (m, n, k) in [(47, 200, 136), (1, 64, 64), (64, 1032, 5128), (65, 64, 64)] {
        let a: Vec<f32> = (0..m * k).map(|_| rng.next() * 3.0).collect();
        let b: Vec<bf16> = (0..n * k)
            .map(|_| bf16::from_f32(rng.next() * 0.1))
            .collect();
        let dims = GemmDims {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            pad: 0,
        };
        let (pa, pb, pd) = (
            upload(&ctx, bytes(&a)),
            upload(&ctx, bytes(&b)),
            upload(&ctx, bytes(std::slice::from_ref(&dims))),
        );
        let po = upload(&ctx, &vec![0u8; m * n * 4]);
        let ran = kernels
            .run(
                &desc,
                &GemmAddrs {
                    a: pa,
                    b: pb,
                    out: po,
                    dims: pd,
                    m: m as u32,
                    n: n as u32,
                    k: 0,
                    epilogue: Default::default(),
                    checked: true,
                },
            )
            .unwrap();
        assert_eq!(ran, thinfer_cuda::gemm::accepts(&desc, m as u32, n as u32));
        let mut got = vec![0f32; m * n];
        unsafe { sys::cuMemcpyDtoH_v2(got.as_mut_ptr().cast(), po, m * n * 4) }
            .result()
            .unwrap();
        for p in [pa, pb, pd, po] {
            unsafe { sys::cuMemFree_v2(p) }.result().unwrap();
        }
        if m > 64 {
            assert!(!ran, "{m} rows must decline");
            continue;
        }
        assert!(ran, "declined {m}x{n}x{k}");
        let mut worst = 0f64;
        for i in 0..m {
            for j in 0..n {
                let w: f64 = (0..k)
                    .map(|kk| a[i * k + kk] as f64 * b[j * k + kk].to_f64())
                    .sum();
                let scale: f64 = (0..k)
                    .map(|kk| (a[i * k + kk] as f64 * b[j * k + kk].to_f64()).abs())
                    .sum();
                worst = worst.max((got[i * n + j] as f64 - w).abs() / scale.max(1e-30));
            }
        }
        eprintln!("cuda gemm f32 {m}x{n}x{k}: worst err / sum|ab| {worst:.2e}");
        // f32 accumulation over K terms: a few ulps of the absolute sum.
        assert!(worst < 1e-6, "{m}x{n}x{k}: {worst:.2e}");
    }
}

/// The fused epilogue `out = residual + gate * (acc + bias)` against the CPU
/// reference, with the residual in its own buffer and aliasing the output
/// (the in-place accumulate), on a multi-group ragged shape; a non-bf16 output
/// declines it.
#[test]
fn cuda_gemm_fused_epilogue() {
    if !thinfer_cuda::driver_present() {
        eprintln!("skipped[cuda gemm epilogue]: no CUDA driver");
        return;
    }
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skipped[cuda gemm epilogue]: no device");
        return;
    };
    let sm = ctx.compute_capability().unwrap();
    let Ok(kernels) = GemmKernels::new(ctx.clone(), sm) else {
        eprintln!("skipped[cuda gemm epilogue]: no kernels");
        return;
    };
    let mut rng = Rng(23);
    let (m, n, k) = (1100usize, 520usize, 264usize);
    let desc = GemmDesc {
        a: GemmType::Bf16,
        b: GemmType::Bf16,
        b_nmajor: false,
        out_f32: false,
    };
    let bf = |rng: &mut Rng, len: usize, s: f32| -> (Vec<f32>, Vec<u16>) {
        (0..len)
            .map(|_| store(GemmType::Bf16, rng.next() * s))
            .unzip()
    };
    let (a_val, a_bits) = bf(&mut rng, m * k, 1.0);
    let (b_val, b_bits) = bf(&mut rng, k * n, 0.1);
    let (bias_val, bias_bits) = bf(&mut rng, n, 0.5);
    let (gate_val, gate_bits) = bf(&mut rng, n, 1.5);
    let (res_val, res_bits) = bf(&mut rng, m * n, 2.0);
    let want: Vec<f32> = (0..m * n)
        .map(|i| {
            let (r, c) = (i / n, i % n);
            let acc: f32 = (0..k).map(|kk| a_val[r * k + kk] * b_val[kk * n + c]).sum();
            res_val[i] + gate_val[c] * (acc + bias_val[c])
        })
        .collect();
    for in_place in [false, true] {
        let (pa, pb) = (upload(&ctx, bytes(&a_bits)), upload(&ctx, bytes(&b_bits)));
        let (pbias, pgate) = (
            upload(&ctx, bytes(&bias_bits)),
            upload(&ctx, bytes(&gate_bits)),
        );
        let pres = upload(&ctx, bytes(&res_bits));
        let po = if in_place {
            pres
        } else {
            upload(&ctx, &vec![0u8; m * n * 2])
        };
        let epi = thinfer_core::backend::GemmEpilogue {
            bias: pbias,
            residual: pres,
            gate: pgate,
        };
        let ran = kernels
            .run(
                &desc,
                &GemmAddrs {
                    a: pa,
                    b: pb,
                    out: po,
                    dims: 0,
                    m: m as u32,
                    n: n as u32,
                    k: k as u32,
                    epilogue: epi,
                    checked: true,
                },
            )
            .unwrap();
        assert!(ran, "fused epilogue declined (in_place {in_place})");
        let mut raw = vec![0u8; m * n * 2];
        unsafe { sys::cuMemcpyDtoH_v2(raw.as_mut_ptr().cast(), po, raw.len()) }
            .result()
            .unwrap();
        let (mut num, mut den) = (0f64, 0f64);
        for (i, b) in raw.chunks_exact(2).enumerate() {
            let e = load(GemmType::Bf16, u16::from_le_bytes([b[0], b[1]])) - want[i];
            num += (e * e) as f64;
            den += (want[i] * want[i]) as f64;
        }
        let rel = (num / den).sqrt();
        eprintln!("cuda gemm epilogue in_place {in_place}: rel_l2 {rel:.2e}");
        assert!(
            rel < 8e-3,
            "epilogue rel_l2 {rel:.2e} (in_place {in_place})"
        );
        let mut ptrs = vec![pa, pb, pbias, pgate, pres];
        if !in_place {
            ptrs.push(po);
        }
        for p in ptrs {
            unsafe { sys::cuMemFree_v2(p) }.result().unwrap();
        }
    }
    // f16 output takes a bias-only epilogue; a residual or gate is declined
    // (before any launch: the addresses are fake), never silently dropped.
    let f16_desc = GemmDesc {
        a: GemmType::F16,
        ..desc
    };
    let epi = thinfer_core::backend::GemmEpilogue {
        bias: 16,
        residual: 16,
        gate: 0,
    };
    assert!(
        !kernels
            .run(
                &f16_desc,
                &GemmAddrs {
                    a: 16,
                    b: 16,
                    out: 16,
                    dims: 0,
                    m: 128,
                    n: 64,
                    k: 64,
                    epilogue: epi,
                    checked: true,
                },
            )
            .unwrap(),
        "f16 output took a residual epilogue"
    );
}

/// The one-image conv against a CPU reference summing each output with
/// `mul_add` in the kernel's (and the WGSL op's) order: bit-exact, over the
/// encoder's forms (3x3 reflect, the stride-2 downsampler, a 1x1 shortcut),
/// ragged shapes, and both tile kernels (few tiles -> small; many -> big).
#[test]
fn cuda_conv2d_is_bit_exact() {
    use thinfer_core::backend::{Conv2dAddrs, Conv2dShape};
    if !thinfer_cuda::driver_present() {
        eprintln!("skipped[cuda conv2d]: no CUDA driver");
        return;
    }
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skipped[cuda conv2d]: no device");
        return;
    };
    let sm = ctx.compute_capability().unwrap();
    let Ok(kernels) = GemmKernels::new(ctx.clone(), sm) else {
        eprintln!("skipped[cuda conv2d]: no kernels");
        return;
    };
    let mut rng = Rng(9);
    // (cin, cout, h, w, k, stride, pad, reflect)
    let cases = [
        (3, 128, 20, 18, 3, 1, 1, true),
        (24, 48, 9, 7, 3, 2, 0, true),
        (32, 70, 5, 5, 1, 1, 0, true),
        (8, 1024, 64, 32, 3, 1, 1, true),
        (5, 16, 6, 6, 3, 1, 1, false),
    ];
    for (cin, cout, h, w, k, stride, pad, reflect) in cases {
        let h_out = (h + 2 * pad + stride - 1 - k) / stride + 1;
        let w_out = (w + 2 * pad + stride - 1 - k) / stride + 1;
        let x: Vec<f32> = (0..cin * h * w).map(|_| rng.next()).collect();
        let wt: Vec<bf16> = (0..cout * cin * k * k)
            .map(|_| bf16::from_f32(rng.next() * 0.2))
            .collect();
        let bias: Vec<bf16> = (0..cout).map(|_| bf16::from_f32(rng.next())).collect();
        let at = |i: i64, n: usize| -> Option<usize> {
            let n = n as i64;
            match (reflect, i) {
                (true, i) if i < 0 => Some(-i as usize),
                (true, i) if i >= n => Some((2 * n - 2 - i) as usize),
                (false, i) if i < 0 || i >= n => None,
                (_, i) => Some(i as usize),
            }
        };
        let mut want = vec![0f32; cout * h_out * w_out];
        for co in 0..cout {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut acc = 0f32;
                    for ci in 0..cin {
                        for dh in 0..k {
                            for dw in 0..k {
                                let hi = at((ho * stride + dh) as i64 - pad as i64, h);
                                let wi = at((wo * stride + dw) as i64 - pad as i64, w);
                                let xv = match (hi, wi) {
                                    (Some(hi), Some(wi)) => x[(ci * h + hi) * w + wi],
                                    _ => 0.0,
                                };
                                let wv = wt[((co * cin + ci) * k + dh) * k + dw].to_f32();
                                acc = wv.mul_add(xv, acc);
                            }
                        }
                    }
                    want[(co * h_out + ho) * w_out + wo] = acc + bias[co].to_f32();
                }
            }
        }
        let (px, pw, pb) = (
            upload(&ctx, bytes(&x)),
            upload(&ctx, bytes(&wt)),
            upload(&ctx, bytes(&bias)),
        );
        let po = upload(&ctx, &vec![0u8; want.len() * 4]);
        let shape = Conv2dShape {
            cin: cin as u32,
            cout: cout as u32,
            h: h as u32,
            w: w as u32,
            h_out: h_out as u32,
            w_out: w_out as u32,
            k: k as u32,
            stride: stride as u32,
            pad: pad as u32,
            reflect,
            bf16: false,
        };
        let ran = kernels
            .run_conv2d(&Conv2dAddrs {
                x: px,
                w: pw,
                bias: pb,
                out: po,
                shape,
            })
            .unwrap();
        assert!(ran, "declined {shape:?}");
        let read = || {
            let mut got = vec![0f32; want.len()];
            unsafe { sys::cuMemcpyDtoH_v2(got.as_mut_ptr().cast(), po, got.len() * 4) }
                .result()
                .unwrap();
            got
        };
        let got = read();
        // The opt-in bf16 tensor-core form: inputs rounded to bf16 (kdim
        // multiples of 8; the rest stay on the exact kernel).
        let bf16_shape = Conv2dShape {
            bf16: true,
            ..shape
        };
        let ran = kernels
            .run_conv2d(&Conv2dAddrs {
                x: px,
                w: pw,
                bias: pb,
                out: po,
                shape: bf16_shape,
            })
            .unwrap();
        assert!(ran, "declined {bf16_shape:?}");
        let got16 = read();
        let (mut e2, mut r2) = (0f64, 0f64);
        for (g, w) in got16.iter().zip(&want) {
            e2 += ((g - w) as f64).powi(2);
            r2 += (*w as f64).powi(2);
        }
        let rel = (e2 / r2).sqrt();
        eprintln!("cuda conv2d bf16 {shape:?}: rel_rms {rel:.3e}");
        assert!(rel < 5e-3, "bf16 conv rel_rms {rel}");
        for p in [px, pw, pb, po] {
            unsafe { sys::cuMemFree_v2(p) }.result().unwrap();
        }
        let bad = got
            .iter()
            .zip(&want)
            .filter(|(g, w)| g.to_bits() != w.to_bits())
            .count();
        eprintln!("cuda conv2d {shape:?}: {bad} of {} differ", want.len());
        assert_eq!(bad, 0, "{shape:?}");
    }
}

/// Throughput at the MiniMax-H3 generic-block site shapes (bf16 x bf16
/// N-major), per row-chunk size. Opt-in (`THINFER_GEMM_BENCH=1`); garbage
/// operands, timing only.
#[test]
fn cuda_gemm_bench() {
    if std::env::var_os("THINFER_GEMM_BENCH").is_none() || !thinfer_cuda::driver_present() {
        return;
    }
    let ctx = CudaContext::new(0).unwrap();
    let sm = ctx.compute_capability().unwrap();
    let kernels = GemmKernels::new(ctx.clone(), sm).unwrap();
    let desc = GemmDesc {
        a: GemmType::Bf16,
        b: GemmType::Bf16,
        b_nmajor: true,
        out_f32: false,
    };
    let (max_m, max_n, max_k) = (8192usize, 28672usize, 14336usize);
    let pa = upload(&ctx, &vec![0x3cu8; max_m * max_k * 2]);
    let pb = upload(&ctx, &vec![0x3cu8; max_n * max_k * 2]);
    let po = upload(&ctx, &vec![0u8; max_m * max_n * 2]);
    for (n, k) in [
        (21504usize, 5376usize),
        (5376, 5376),
        (28672, 5376),
        (5376, 14336),
        (512, 5376),
        (5376, 512),
    ] {
        for m in [1024usize, 2048, 4096, 8192] {
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t = std::time::Instant::now();
                assert!(
                    kernels
                        .run(
                            &desc,
                            &GemmAddrs {
                                a: pa,
                                b: pb,
                                out: po,
                                dims: 0,
                                m: m as u32,
                                n: n as u32,
                                k: k as u32,
                                epilogue: Default::default(),
                                checked: true,
                            },
                        )
                        .unwrap()
                );
                unsafe { sys::cuCtxSynchronize() }.result().unwrap();
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            let tf = 2.0 * (m * n * k) as f64 / best / 1e9;
            eprintln!("gemm m={m} n={n} k={k}: {best:.2} ms ({tf:.1} TFLOP/s)");
        }
    }
    for p in [pa, pb, po] {
        unsafe { sys::cuMemFree_v2(p) }.result().unwrap();
    }
}
