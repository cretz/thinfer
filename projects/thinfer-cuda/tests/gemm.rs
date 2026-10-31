//! The CUDA dense matmul against a CPU f32 reference with the wgpu matmul
//! contract: every supported (A type, B type, B layout) variant, shapes that are not
//! tile multiples, and the bf16 -> f16 overflow decline. Skips when no CUDA
//! device / NVRTC is present.

use std::sync::Arc;

use cudarc::driver::{CudaContext, sys};
use half::{bf16, f16};
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
    }
}

fn load(ty: GemmType, bits: u16) -> f32 {
    match ty {
        GemmType::F16 => f16::from_bits(bits).to_f32(),
        GemmType::Bf16 => bf16::from_bits(bits).to_f32(),
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
            pa,
            pb,
            po,
            pd,
            m as u32,
            n as u32,
            0,
            Default::default(),
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

/// The fused epilogue `out = residual + gate * (acc + bias)` against the CPU
/// reference, with the residual in its own buffer and aliasing the output
/// (the in-place accumulate), on a multi-group ragged shape; a non-bf16 output
/// declines it.
#[test]
fn cuda_gemm_fused_epilogue() {
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
            .run(&desc, pa, pb, po, 0, m as u32, n as u32, k as u32, epi)
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
    // f16 output: a non-empty epilogue is declined, never silently dropped.
    let f16_desc = GemmDesc {
        a: GemmType::F16,
        ..desc
    };
    let epi = thinfer_core::backend::GemmEpilogue {
        bias: 16,
        residual: 0,
        gate: 0,
    };
    assert!(
        !kernels
            .run(&f16_desc, 16, 16, 16, 0, 128, 64, 64, epi)
            .unwrap(),
        "f16 output took an epilogue"
    );
}
