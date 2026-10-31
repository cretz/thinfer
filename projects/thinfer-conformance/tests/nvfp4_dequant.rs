//! NVFP4 dequant kernel (`ops::nvfp4`) vs its CPU reference, bf16 and f16
//! targets, including shapes that are NOT whole 128-row / 4-block scale tiles
//! (the padded swizzle path), every E2M1 code, a wide E4M3 scale range and a
//! distinct alpha per 128-row group.

#![cfg(feature = "conformance")]

use thinfer_core::backend::{Backend, BufRef, WgpuBackend};
use thinfer_core::ops::dequant::{DequantTarget, layout};
use thinfer_core::ops::nvfp4::{self, Nvfp4Layout};
use thinfer_core::ops::{DequantBufs, DequantScheme, dispatch_dequant};

/// A composite with pseudo-random codes, finite E4M3 scales (exponents 1..14)
/// and a non-trivial alpha.
fn composite(l: Nvfp4Layout, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut next = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 33) as u32
    };
    let mut bytes = vec![0u8; l.total_len()];
    for b in &mut bytes[..l.packed_len()] {
        *b = next() as u8;
    }
    let bp = l.blocks_padded();
    for r in 0..l.n {
        for blk in 0..l.k / nvfp4::BLOCK {
            let e = 1 + next() % 14;
            let byte = ((next() & 1) << 7) | (e << 3) | (next() & 7);
            bytes[l.scales_offset() + nvfp4::scale_offset(r, blk, bp)] = byte as u8;
        }
    }
    // A distinct alpha per 128-row group (assembled composites mix sources).
    for g in 0..l.groups() {
        let a = l.alpha_offset() + 4 * g;
        bytes[a..a + 4].copy_from_slice(&(0.037f32 * (g + 1) as f32).to_le_bytes());
    }
    bytes
}

async fn gpu_dequant(bytes: &[u8], l: Nvfp4Layout, target: DequantTarget) -> Vec<f32> {
    let backend = WgpuBackend::new().await.expect("wgpu adapter");
    let alloc_with = |b: &[u8]| -> BufRef {
        let id = backend.allocate(b.len() as u64).expect("allocate");
        backend.write_buffer(id, 0, b).expect("write");
        BufRef::new(id, b.len() as u64)
    };
    let pipe = backend
        .create_pipeline("nvfp4", &nvfp4::build_wgsl(target), "main", layout())
        .await
        .expect("nvfp4 pipe");
    let (n, k) = (l.n as u32, l.k as u32);
    let src = alloc_with(bytes);
    let dense_len = n as u64 * k as u64 * 2;
    let dense_id = backend.allocate(dense_len).expect("alloc dense");
    let dense = BufRef::new(dense_id, dense_len);
    let mut dims = [0u8; 16];
    dims[0..4].copy_from_slice(&n.to_le_bytes());
    dims[4..8].copy_from_slice(&k.to_le_bytes());
    let dims = alloc_with(&dims);
    let mut enc = backend.create_command_encoder();
    let bufs = DequantBufs {
        b_quant: &src,
        b_dense: &dense,
        dims: &dims,
    };
    dispatch_dequant(&backend, &mut enc, &pipe, DequantScheme::Nvfp4, &bufs, n, k)
        .expect("dispatch");
    backend.submit(enc).await.expect("submit");
    let raw = backend
        .read_buffer(dense_id, 0, dense_len)
        .await
        .expect("read");
    raw.chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            match target {
                DequantTarget::Bf16 => half::bf16::from_bits(bits).to_f32(),
                DequantTarget::F16 => half::f16::from_bits(bits).to_f32(),
            }
        })
        .collect()
}

#[test]
fn nvfp4_dequant_matches_cpu_reference() {
    for (n, k) in [(256usize, 128usize), (200, 48), (128, 2688)] {
        let l = Nvfp4Layout::new(n, k);
        let bytes = composite(l, (n * 31 + k) as u64);
        let want = nvfp4::dequantize(&bytes, l);
        for target in [DequantTarget::Bf16, DequantTarget::F16] {
            let got = pollster::block_on(gpu_dequant(&bytes, l, target));
            assert_eq!(got.len(), want.len());
            // Exact up to the target's rounding of the f32 product.
            let tol = match target {
                DequantTarget::Bf16 => 1.0 / 128.0,
                DequantTarget::F16 => 1.0 / 1024.0,
            };
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                let w16 = w.clamp(-65504.0, 65504.0);
                assert!(
                    (g - w16).abs() <= tol * w16.abs().max(1e-30) || g == &w16,
                    "{n}x{k} {target:?} elem {i} (r {}, c {}): gpu {g} vs cpu {w16}",
                    i / k,
                    i % k
                );
            }
        }
    }
}
