//! The CUDA attention kernel against a CPU f32 reference with the `sdpa_sg`
//! contract: both head dims, GQA, all three mask modes, and sequence lengths
//! that are not tile multiples. Skips when no CUDA device / NVRTC is present.
#![cfg(not(target_os = "macos"))]

use std::sync::Arc;

use cudarc::driver::{CudaContext, sys};
use half::f16;
use thinfer_cuda::sdpa::{SdpaKernels, SdpaUniform};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f16> {
        (0..n).map(|_| f16::from_f32(self.next() * scale)).collect()
    }
}

#[allow(clippy::too_many_arguments)]
fn reference(u: &SdpaUniform, q: &[f16], k: &[f16], v: &[f16], mask: &[f16]) -> Vec<f32> {
    let (b, sq, sk, hq, hkv, d) = (
        u.b as usize,
        u.s_q as usize,
        u.s_k as usize,
        u.h_q as usize,
        u.h_kv as usize,
        u.d as usize,
    );
    let mut out = vec![0f32; b * sq * hq * d];
    for bb in 0..b {
        for h in 0..hq {
            let hk = h * hkv / hq;
            for i in 0..sq {
                let qi = &q[((bb * sq + i) * hq + h) * d..][..d];
                let s: Vec<f32> = (0..sk)
                    .map(|j| {
                        let kj = &k[((bb * sk + j) * hkv + hk) * d..][..d];
                        let dot: f32 = qi
                            .iter()
                            .zip(kj)
                            .map(|(a, b)| a.to_f32() * b.to_f32())
                            .sum();
                        let bias = match u.has_mask {
                            0 => 0.0,
                            1 => mask[(bb * sq + i) * sk + j].to_f32(),
                            _ => mask[((bb * hq + h) * sq + i) * sk + j].to_f32(),
                        };
                        dot * u.scale + bias
                    })
                    .collect();
                let m = s.iter().cloned().fold(f32::MIN, f32::max);
                let p: Vec<f32> = s.iter().map(|x| (x - m).exp()).collect();
                let l: f32 = p.iter().sum();
                let o = &mut out[((bb * sq + i) * hq + h) * d..][..d];
                for (j, pj) in p.iter().enumerate() {
                    let vj = &v[((bb * sk + j) * hkv + hk) * d..][..d];
                    for (oc, vc) in o.iter_mut().zip(vj) {
                        *oc += pj * vc.to_f32() / l;
                    }
                }
            }
        }
    }
    out
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

#[test]
fn cuda_sdpa_matches_reference() {
    if !thinfer_cuda::driver_present() {
        eprintln!("skipped[cuda sdpa]: no CUDA driver");
        return;
    }
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipped[cuda sdpa]: {e:?}");
            return;
        }
    };
    let sm = ctx.compute_capability().unwrap();
    let kernels = match SdpaKernels::new(ctx.clone(), sm) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("skipped[cuda sdpa]: {e}");
            return;
        }
    };
    let mut rng = Rng(7);
    let mut worst = 0f32;
    for (d, has_mask) in [(64u32, 0u32), (128, 0), (128, 1), (64, 2)] {
        let u = SdpaUniform {
            b: 1,
            h_q: 4,
            h_kv: 2,
            s_q: 100,
            s_k: 77,
            d,
            scale: 1.0 / (d as f32).sqrt(),
            has_mask,
        };
        let (sq, sk, hq, hkv, du) = (100usize, 77usize, 4usize, 2usize, d as usize);
        let q = rng.vec(sq * hq * du, 1.0);
        let k = rng.vec(sk * hkv * du, 1.0);
        let v = rng.vec(sk * hkv * du, 1.0);
        let mask = match has_mask {
            0 => vec![f16::ZERO; 1],
            1 => rng.vec(sq * sk, 2.0),
            _ => rng.vec(hq * sq * sk, 2.0),
        };
        let want = reference(&u, &q, &k, &v, &mask);
        let (pq, pk, pv, pm) = (
            upload(&ctx, bytes(&q)),
            upload(&ctx, bytes(&k)),
            upload(&ctx, bytes(&v)),
            upload(&ctx, bytes(&mask)),
        );
        let pu = upload(&ctx, bytes(std::slice::from_ref(&u)));
        let po = upload(&ctx, &vec![0u8; want.len() * 2]);
        assert!(
            kernels.run(&u, pq, pk, pv, pm, po, pu).unwrap(),
            "kernel declined d={d}"
        );
        let mut got = vec![f16::ZERO; want.len()];
        unsafe { sys::cuMemcpyDtoH_v2(got.as_mut_ptr().cast(), po, want.len() * 2) }
            .result()
            .unwrap();
        let (mut num, mut den, mut max_abs) = (0f64, 0f64, 0f32);
        for (g, w) in got.iter().zip(&want) {
            let e = g.to_f32() - w;
            num += (e * e) as f64;
            den += (w * w) as f64;
            max_abs = max_abs.max(e.abs());
        }
        let rel = (num / den).sqrt() as f32;
        eprintln!("cuda sdpa d={d} mask={has_mask}: rel_l2 {rel:.2e} max_abs {max_abs:.2e}");
        worst = worst.max(rel);
        for p in [pq, pk, pv, pm, pu, po] {
            unsafe { sys::cuMemFree_v2(p) }.result().unwrap();
        }
    }
    // f16 inputs/outputs and f16 P: ~1e-3 relative is the expected floor.
    assert!(worst < 5e-3, "worst rel_l2 {worst:.2e}");
}
