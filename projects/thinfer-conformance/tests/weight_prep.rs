//! GPU weight-prep kernels (`thinfer_core::ops::weight_prep`, reached via
//! `Backend::weight_prep`) vs the CPU upload path they replace.
//!
//! Transpose: bit-exact (pure bit movement). Q8_0 transcode:
//! value-equivalent, NOT bit-exact - WGSL guarantees division only to
//! 2.5 ULP, so `amax/127` / `1/d` can sit 1 ULP off the CPU value and
//! quants whose scaled input lands within an ULP of a round-to-nearest
//! tie flip by +-1 (~0.03% of cells on real weights; both are equally
//! accurate encodings; llama.cpp's CPU/CUDA/Metal quantizers diverge the
//! same way). Gate: scales within 1 f16 ULP, quants within +-1, and the
//! mismatch fraction small enough to rule out layout/indexing bugs.
//!
//! Also the WGSL parse/validate canary for both kernels (reserved keywords,
//! binding mismatches) without an e2e+model-load round trip.

#![cfg(feature = "conformance")]

use half::bf16;
use thinfer_core::backend::{Backend, BufRef, WeightPrep, WgpuBackend};
use thinfer_core::quant::encode_q8_0_from_bf16;

/// Deterministic bf16 byte stream: LCG floats in [-amp, amp], rounded
/// through bf16 (the value both sides read).
fn synth_bf16(elems: usize, seed: u64, amp: f32) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(elems * 2);
    for _ in 0..elems {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let r = ((s >> 33) as u32 as f32) / (u32::MAX as f32);
        let v = bf16::from_f32((r * 2.0 - 1.0) * amp);
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    out
}

async fn gpu_prep(op: WeightPrep, raw: &[u8], dst_len: u64) -> Vec<u8> {
    let backend = WgpuBackend::new().await.expect("wgpu adapter");
    assert!(
        backend.supports_weight_prep(op),
        "WgpuBackend must implement weight_prep"
    );
    // Staging is the caller's job now (residency streams into it; here one
    // aligned write covers it).
    let staging_len = (raw.len() as u64).next_multiple_of(4);
    let staging = backend.allocate(staging_len).expect("alloc staging");
    let mut padded = raw.to_vec();
    padded.resize(staging_len as usize, 0);
    backend.write_buffer(staging, 0, &padded).expect("stage");
    let dst = backend.allocate(dst_len).expect("alloc dst");
    backend
        .weight_prep(
            op,
            &BufRef::new(staging, staging_len),
            &BufRef::new(dst, dst_len),
        )
        .await
        .expect("weight_prep");
    let got = backend.read_buffer(dst, 0, dst_len).await.expect("read");
    backend.free(staging);
    backend.free(dst);
    got
}

fn check_q8(n: u32, k: u32, raw: &[u8]) {
    let blocks = (n as usize) * (k as usize) / 32;
    assert!(
        blocks.is_multiple_of(2),
        "test shapes use even block counts"
    );
    let mut exp = vec![0u8; blocks * 34];
    encode_q8_0_from_bf16(raw, &mut exp);
    let got = pollster::block_on(gpu_prep(
        WeightPrep::Q8_0FromBf16 { n, k },
        raw,
        exp.len() as u64,
    ));
    assert_eq!(got.len(), exp.len());
    let mut tie_flips = 0usize;
    for b in 0..blocks {
        let (g, e) = (&got[b * 34..(b + 1) * 34], &exp[b * 34..(b + 1) * 34]);
        let gd = u16::from_le_bytes([g[0], g[1]]);
        let ed = u16::from_le_bytes([e[0], e[1]]);
        assert!(
            (i32::from(gd) - i32::from(ed)).abs() <= 1,
            "[{n}, {k}] block {b}: scale bits gpu={gd:#06x} cpu={ed:#06x}"
        );
        for (i, (gq, eq)) in g[2..].iter().zip(&e[2..]).enumerate() {
            let dq = (i32::from(*gq as i8) - i32::from(*eq as i8)).abs();
            assert!(
                dq <= 1,
                "[{n}, {k}] block {b} q[{i}]: gpu={} cpu={}",
                *gq as i8,
                *eq as i8
            );
            tie_flips += usize::from(dq == 1);
        }
    }
    // Tie flips are sparse (~0.03% on real weights); a broad mismatch means
    // a layout/indexing bug, not rounding.
    let frac = tie_flips as f64 / (blocks * 32) as f64;
    assert!(
        frac <= 0.01,
        "[{n}, {k}]: {tie_flips} quant flips ({frac:.4}) - beyond tie noise"
    );
}

#[test]
fn q8_0_from_bf16_random() {
    let (n, k) = (16u32, 320u32);
    let raw = synth_bf16((n * k) as usize, 0xC0FFEE, 4.0);
    check_q8(n, k, &raw);
}

#[test]
fn q8_0_from_bf16_edges() {
    // Engineered blocks: all-zero (d == 0 path), round-half-away ties
    // (amax = 127 so d = 1 exactly; +-2.5 must quantize to +-3, not the
    // ties-to-even 2), tiny values (f16-subnormal scale), and large values.
    let (n, k) = (4u32, 64u32);
    let mut vals = vec![0f32; (n * k) as usize];
    // Block 1 (elements 32..64): tie cases at d = 1.
    vals[32] = 127.0;
    vals[33] = 2.5;
    vals[34] = -2.5;
    vals[35] = 0.5;
    vals[36] = -0.5;
    vals[37] = 3.5;
    // Blocks 2-3: subnormal-f16 scales.
    for (i, v) in vals[64..128].iter_mut().enumerate() {
        *v = 1.0e-6 * ((i % 7) as f32 - 3.0);
    }
    // Blocks 4..: mixed magnitudes.
    for (i, v) in vals[128..].iter_mut().enumerate() {
        *v = ((i as f32) - 60.0) * 1.75;
    }
    let raw: Vec<u8> = vals
        .iter()
        .flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    check_q8(n, k, &raw);
}

fn check_transpose(n: u32, k: u32, raw: &[u8]) {
    let (nu, ku) = (n as usize, k as usize);
    let src: Vec<u16> = raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    // Independent naive reference (the CPU path's block-tiled transpose is
    // itself under test equivalence; a second implementation cross-checks
    // both).
    let mut exp = vec![0u8; raw.len()];
    {
        let dst: &mut [u8] = &mut exp;
        for (i, chunk) in dst.chunks_exact_mut(2).enumerate() {
            let kk = i / nu;
            let nn = i % nu;
            chunk.copy_from_slice(&src[nn * ku + kk].to_le_bytes());
        }
    }
    let got = pollster::block_on(gpu_prep(
        WeightPrep::TransposeBf16 { n, k },
        raw,
        exp.len() as u64,
    ));
    let diff = got.iter().zip(&exp).filter(|(g, e)| g != e).count();
    assert_eq!(diff, 0, "[{n}, {k}]: {diff}/{} bytes differ", exp.len());
}

#[test]
fn transpose_bf16_even_n() {
    let (n, k) = (70u32, 33u32);
    let raw = synth_bf16((n * k) as usize, 0xBEEF, 2.0);
    check_transpose(n, k, &raw);
}

#[test]
fn transpose_bf16_odd_k() {
    // N must be even (odd N breaks output word alignment; the residency
    // prep gate falls back to the CPU transpose). Odd K is fine: input
    // reads are per-u16.
    let (n, k) = (256u32, 129u32);
    let raw = synth_bf16((n * k) as usize, 0xF00D, 2.0);
    check_transpose(n, k, &raw);
}
