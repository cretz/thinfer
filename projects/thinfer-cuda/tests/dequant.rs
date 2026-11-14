//! The CUDA GGUF dequant (bf16 N-major image) bit-for-bit against the CPU
//! reference in `thinfer_core::quant` rounded to bf16, every kind the H3
//! generic block routes, plus the bf16 <-> f16 casts. Skips without CUDA.
#![cfg(not(target_os = "macos"))]

use cudarc::driver::{CudaContext, sys};
use half::{bf16, f16};
use thinfer_core::backend::ElementwiseOp;
use thinfer_core::quant::{QuantKind, dequantize_row};
use thinfer_cuda::elementwise::ElementwiseKernels;

fn upload(ctx: &std::sync::Arc<CudaContext>, bytes: &[u8]) -> sys::CUdeviceptr {
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

fn download(p: sys::CUdeviceptr, n: usize) -> Vec<u16> {
    let mut out = vec![0u16; n];
    unsafe { sys::cuCtxSynchronize() }.result().unwrap();
    unsafe { sys::cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), p, n * 2) }
        .result()
        .unwrap();
    out
}

/// Random block bytes with finite, modest f16 scales at the scheme's offsets.
fn blocks(kind: QuantKind, count: usize, seed: &mut u64) -> Vec<u8> {
    let mut next = || {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 33) as u8
    };
    let bpb = kind.bytes_per_block() as usize;
    let mut b: Vec<u8> = (0..count * bpb).map(|_| next()).collect();
    let scale_at: &[usize] = match kind {
        QuantKind::Q8_0 => &[0],
        QuantKind::Q4_K | QuantKind::Q5_K => &[0, 2],
        QuantKind::Q6_K => &[208],
        QuantKind::Q4_0 => &[0],
    };
    for blk in b.chunks_mut(bpb) {
        for &at in scale_at {
            let v = f16::from_f32((next() as f32 / 255.0 + 0.01) * 0.02);
            blk[at..at + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    b
}

#[test]
fn cuda_dequant_matches_cpu_reference() {
    if !thinfer_cuda::driver_present() {
        eprintln!("skipped[cuda dequant]: no CUDA driver");
        return;
    }
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skipped[cuda dequant]: no device");
        return;
    };
    let sm = ctx.compute_capability().unwrap();
    let k = match ElementwiseKernels::new(ctx.clone(), sm) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("skipped[cuda dequant]: {e}");
            return;
        }
    };
    let mut seed = 11u64;
    let (n, kk) = (24u32, 512u32);
    for kind in [
        QuantKind::Q4_K,
        QuantKind::Q5_K,
        QuantKind::Q6_K,
        QuantKind::Q8_0,
    ] {
        let count = (n * kk / kind.block_size()) as usize;
        let src = blocks(kind, count, &mut seed);
        let mut want = vec![0f32; (n * kk) as usize];
        dequantize_row(kind, &src, &mut want);
        let (ps, po) = (upload(&ctx, &src), upload(&ctx, &vec![0u8; want.len() * 2]));
        let op = ElementwiseOp::Dequant {
            kind,
            src: ps,
            out: po,
            n,
            k: kk,
        };
        assert!(k.launch(&op).unwrap(), "{kind:?} declined");
        let got = download(po, want.len());
        let bad = got
            .iter()
            .zip(&want)
            .filter(|(g, w)| **g != bf16::from_f32(**w).to_bits())
            .count();
        eprintln!("cuda dequant {kind:?}: {bad} of {} differ", want.len());
        assert_eq!(bad, 0, "{kind:?}");
        for p in [ps, po] {
            unsafe { sys::cuMemFree_v2(p) }.result().unwrap();
        }
    }
    // bf16 -> f16 -> bf16 in place.
    let vals: Vec<u16> = (0..1000)
        .map(|i| bf16::from_f32((i as f32 - 500.0) * 0.37).to_bits())
        .collect();
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let p = upload(&ctx, &bytes);
    let n = vals.len() as u32;
    assert!(
        k.launch(&ElementwiseOp::Bf16ToF16 { x: p, out: p, n })
            .unwrap()
    );
    let h = download(p, vals.len());
    for (v, h) in vals.iter().zip(&h) {
        assert_eq!(*h, f16::from_f32(bf16::from_bits(*v).to_f32()).to_bits());
    }
    assert!(
        k.launch(&ElementwiseOp::F16ToBf16 { x: p, out: p, n })
            .unwrap()
    );
    let back = download(p, vals.len());
    for (h, b) in h.iter().zip(&back) {
        assert_eq!(*b, bf16::from_f32(f16::from_bits(*h).to_f32()).to_bits());
    }
    unsafe { sys::cuMemFree_v2(p) }.result().unwrap();
}
