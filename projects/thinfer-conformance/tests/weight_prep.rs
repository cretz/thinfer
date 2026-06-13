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

/// Compare `narrow_transpose_f32` (fused f32->bf16 RNE + `[N,K]`->`[K,N]`
/// transpose) against the production CPU narrow (`half::bf16::from_f32`)
/// placed transposed. Bit-exact: the WGSL RNE is pure integer bit movement.
fn check_narrow_transpose(n: u32, k: u32, vals: &[f32]) {
    let (nu, ku) = (n as usize, k as usize);
    assert_eq!(vals.len(), nu * ku);
    let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    // Expected bf16 `[K, N]`: output word i holds elements (kk, nn) low and
    // (kk, nn+1) high, each narrowed by `half` (what the CPU upload uses).
    let mut exp = vec![0u8; nu * ku * 2];
    for (i, chunk) in exp.chunks_exact_mut(2).enumerate() {
        let kk = i / nu;
        let nn = i % nu;
        let v = bf16::from_f32(vals[nn * ku + kk]).to_bits();
        chunk.copy_from_slice(&v.to_le_bytes());
    }
    // Single whole-tensor band (n0 = 0, band_n = n): the residency prep loop
    // exercises multi-band striping separately via `narrow_transpose_f32_banded`.
    let got = pollster::block_on(gpu_prep(
        WeightPrep::NarrowTransposeF32 {
            n,
            k,
            n0: 0,
            band_n: n,
        },
        &raw,
        exp.len() as u64,
    ));
    let diff = got.iter().zip(&exp).filter(|(g, e)| g != e).count();
    assert_eq!(diff, 0, "[{n}, {k}]: {diff}/{} bytes differ", exp.len());
}

#[test]
fn narrow_transpose_f32_random() {
    // Full-precision f32 LCG values (NOT pre-rounded), so every element
    // exercises the RNE rounding path, not just a copy.
    let (n, k) = (70u32, 33u32);
    let mut s = 0x1234_5678_9abc_def0u64;
    let vals: Vec<f32> = (0..n * k)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            // Wide exponent spread (mantissa from high bits, exponent from
            // low) to hit normals across magnitudes.
            let mant = ((s >> 40) as u32) & 0x007F_FFFF;
            let exp = (((s >> 8) as u32) % 60 + 100) << 23; // exp in [100,160)
            let sign = ((s >> 4) as u32 & 1) << 31;
            f32::from_bits(sign | exp | mant)
        })
        .collect();
    check_narrow_transpose(n, k, &vals);
}

#[test]
fn narrow_transpose_f32_banded() {
    // Stripe a tensor through several offset bands (as the residency prep loop
    // does) into one [K, N] buffer; the result must equal the whole-tensor
    // narrow+transpose. Exercises n0 offsets, a non-divisor band size (last
    // band is smaller), and the global-stride output. K odd (=5) too.
    let (n, k, band_rows) = (16u32, 5u32, 6u32);
    let (nu, ku) = (n as usize, k as usize);
    let mut s = 0xDEAD_BEEF_0000_0001u64;
    let vals: Vec<f32> = (0..n * k)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let mant = ((s >> 40) as u32) & 0x007F_FFFF;
            let exp = (((s >> 8) as u32) % 50 + 110) << 23;
            f32::from_bits(exp | mant)
        })
        .collect();
    let mut exp = vec![0u8; nu * ku * 2];
    for (i, chunk) in exp.chunks_exact_mut(2).enumerate() {
        let (kk, nn) = (i / nu, i % nu);
        chunk.copy_from_slice(&bf16::from_f32(vals[nn * ku + kk]).to_bits().to_le_bytes());
    }
    let got = pollster::block_on(async {
        let backend = WgpuBackend::new().await.expect("wgpu adapter");
        let row_bytes = (k as u64) * 4;
        let staging = backend
            .allocate(band_rows as u64 * row_bytes)
            .expect("stage");
        let dst_len = (nu * ku * 2) as u64;
        let dst = backend.allocate(dst_len).expect("dst");
        let mut n0 = 0u32;
        while n0 < n {
            let band_n = band_rows.min(n - n0);
            let band_bytes = band_n as u64 * row_bytes;
            let mut buf = Vec::with_capacity(band_bytes as usize);
            for r in 0..band_n {
                let row = (n0 + r) as usize;
                for &v in &vals[row * ku..row * ku + ku] {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            backend.write_buffer(staging, 0, &buf).expect("write band");
            backend
                .weight_prep(
                    WeightPrep::NarrowTransposeF32 { n, k, n0, band_n },
                    &BufRef::new(staging, band_bytes),
                    &BufRef::new(dst, dst_len),
                )
                .await
                .expect("weight_prep band");
            n0 += band_n;
        }
        let out = backend.read_buffer(dst, 0, dst_len).await.expect("read");
        backend.free(staging);
        backend.free(dst);
        out
    });
    let diff = got.iter().zip(&exp).filter(|(g, e)| g != e).count();
    assert_eq!(
        diff,
        0,
        "banded [{n}, {k}]: {diff}/{} bytes differ",
        exp.len()
    );
}

#[test]
fn narrow_transpose_f32_rounding_edges() {
    // Hand-built f32 bit patterns that pin the RNE corners: exact ties
    // (low 16 == 0x8000) with even vs odd result LSB, sticky-set round-up,
    // round-down, plus NaN / +-inf / max-finite. N even, K=1 keeps the map
    // a straight narrow so a mismatch is rounding, not layout.
    let f = f32::from_bits;
    let vals: Vec<f32> = vec![
        f(0x3F80_8000), // tie, result LSB 0 (0x3F80) -> stays even (round down)
        f(0x3F81_8000), // tie, result LSB 1 (0x3F81) -> round up to 0x3F82
        f(0x3F80_8001), // round bit + sticky -> round up
        f(0x3F80_7FFF), // below half -> round down
        f(0x7F80_0000), // +inf -> 0x7F80
        f(0xFF80_0000), // -inf -> 0xFF80
        f(0x7FC0_0000), // NaN -> high mantissa | quiet bit
        f(0x7F7F_FFFF), // max finite, round bit set -> rounds up to 0x7F80
    ];
    check_narrow_transpose(vals.len() as u32, 1, &vals);
}
