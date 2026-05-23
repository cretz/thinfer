//! GPU block-quant -> packed int8 + per-K=32 scale (`thinfer_core::ops::dequant_i8`)
//! vs scalar Rust reference.
//!
//! Two coverage levels:
//! - `dequant_i8_pipeline_builds_<scheme>`: build the WGSL for each
//!   `QuantKind` (Q8_0, Q4_0, Q4_K, Q5_K, Q6_K). Builds catch shader
//!   validation regressions (reserved keywords, unknown enable directives,
//!   binding-kind mismatches) without needing a quantize helper for each
//!   scheme — most useful for the K-family which has no public
//!   `quantize_row_*`.
//! - `dequant_i8_q8_0_numeric` / `dequant_i8_q4_0_numeric`: round-trip
//!   numerical check against a CPU dequant-then-requantize-to-i8 reference
//!   for the two schemes that expose a public quantizer in thinfer-core.

#![cfg(feature = "conformance")]

use thinfer_core::backend::{Backend, BufRef, WgpuBackend};
use thinfer_core::ops::dequant_i8::{DequantI8Bufs, build_wgsl, dispatch_dequant_i8, hint, layout};
use thinfer_core::quant::{
    QuantKind, dequantize_row_q4_0, dequantize_row_q8_0, quantize_row_q4_0, quantize_row_q8_0,
};

fn pack_dims_u32x4(n: u32, k: u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&n.to_le_bytes());
    out[4..8].copy_from_slice(&k.to_le_bytes());
    out
}

/// CPU reference matching the kernel's per-sub-block i8 quantization rule:
/// for each (n, k/32) sub-block of `b_deq[N, K]`, find absmax, scale = absmax/127,
/// then round each element to i8 (clamped to [-127, 127]).
fn cpu_dequant_i8_ref(b_deq: &[f32], n: u32, k: u32) -> (Vec<i8>, Vec<f32>) {
    assert!(k.is_multiple_of(32));
    let n = n as usize;
    let k = k as usize;
    let blocks = k / 32;
    let mut i8_out = vec![0i8; n * k];
    let mut scale_out = vec![0f32; n * blocks];
    for ni in 0..n {
        for sb in 0..blocks {
            let off = ni * k + sb * 32;
            let block = &b_deq[off..off + 32];
            let absmax = block.iter().copied().fold(0f32, |a, x| a.max(x.abs()));
            let scale = absmax / 127.0;
            scale_out[ni * blocks + sb] = scale;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            for (i, &v) in block.iter().enumerate() {
                let q = (v * inv).round().clamp(-127.0, 127.0) as i32;
                i8_out[off + i] = q as i8;
            }
        }
    }
    (i8_out, scale_out)
}

async fn run_one(
    scheme: QuantKind,
    n: u32,
    k: u32,
    b_q: &[u8],
    b_deq: &[f32],
) -> (Vec<i8>, Vec<f32>, Vec<i8>, Vec<f32>) {
    let bs = scheme.block_size();
    assert!(k.is_multiple_of(bs), "K must be multiple of block_size");
    let (exp_i8, exp_scale) = cpu_dequant_i8_ref(b_deq, n, k);

    let backend = WgpuBackend::new().await.expect("wgpu adapter");
    let wgsl = build_wgsl(scheme);
    let pipeline = backend
        .create_pipeline(&wgsl, "main", layout())
        .await
        .expect("dequant_i8 pipeline");

    let alloc_with = |bytes: &[u8]| -> BufRef {
        let id = backend.allocate(bytes.len() as u64).expect("allocate");
        backend.write_buffer(id, 0, bytes).expect("write");
        BufRef::new(id, bytes.len() as u64)
    };
    let b_buf = alloc_with(b_q);
    let dims_buf = alloc_with(&pack_dims_u32x4(n, k));
    let i8_len = (n as u64) * (k as u64); // 1 byte per i8 elem
    let scale_len = (n as u64) * (k as u64 / 32) * 4;
    let i8_id = backend.allocate(i8_len).expect("alloc i8");
    let scale_id = backend.allocate(scale_len).expect("alloc scale");
    let i8_buf = BufRef::new(i8_id, i8_len);
    let scale_buf = BufRef::new(scale_id, scale_len);

    let mut enc = backend.create_command_encoder();
    dispatch_dequant_i8(
        &backend,
        &mut enc,
        &pipeline,
        scheme,
        &DequantI8Bufs {
            b_quant: &b_buf,
            b_i8: &i8_buf,
            b_scale: &scale_buf,
            dims: &dims_buf,
        },
        n,
        k,
    )
    .expect("dispatch");
    backend.submit(enc).await.expect("submit");

    let i8_bytes = backend
        .read_buffer(i8_id, 0, i8_len)
        .await
        .expect("read i8");
    let got_i8: Vec<i8> = i8_bytes.iter().map(|b| *b as i8).collect();
    let scale_bytes = backend
        .read_buffer(scale_id, 0, scale_len)
        .await
        .expect("read scale");
    let got_scale: Vec<f32> = scale_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    backend.free(b_buf.id);
    backend.free(dims_buf.id);
    backend.free(i8_id);
    backend.free(scale_id);

    (got_i8, got_scale, exp_i8, exp_scale)
}

fn pipeline_build_check(scheme: QuantKind) {
    let src = build_wgsl(scheme);
    // Defensive sanity checks on the emitted source — these would not catch
    // every regression, but they're cheap and surface obvious template
    // breakage before the WGSL parser ever runs.
    assert!(src.contains("pack4xI8"), "missing pack4xI8 for {scheme:?}");
    assert!(
        src.contains("@compute @workgroup_size(64"),
        "missing 64-thread workgroup for {scheme:?}"
    );
    // No reserved keywords. WGSL reserves `active`, `target`, etc.
    assert!(
        !src.contains(" active:") && !src.contains("(active)"),
        "WGSL reserved-keyword `active` appears in dequant_i8 for {scheme:?}"
    );
    assert_eq!(hint(scheme), format!("dequant_i8-{}", scheme.hint()));
}

#[test]
fn dequant_i8_pipeline_builds_q8_0() {
    pipeline_build_check(QuantKind::Q8_0);
}
#[test]
fn dequant_i8_pipeline_builds_q4_0() {
    pipeline_build_check(QuantKind::Q4_0);
}
#[test]
fn dequant_i8_pipeline_builds_q4_k() {
    pipeline_build_check(QuantKind::Q4_K);
}
#[test]
fn dequant_i8_pipeline_builds_q5_k() {
    pipeline_build_check(QuantKind::Q5_K);
}
#[test]
fn dequant_i8_pipeline_builds_q6_k() {
    pipeline_build_check(QuantKind::Q6_K);
}

/// Q8_0 round-trip: synthesize f32, quantize -> Q8_0 bytes, dequant ->
/// requantize-to-i8 on CPU. Compare to GPU output.
#[test]
fn dequant_i8_q8_0_numeric() {
    let n = 2u32;
    let k = 64u32;
    let mut s = 0xC0FFEE_u64;
    let mut rand = || -> f32 {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let r = ((s >> 33) as u32 as f32) / (u32::MAX as f32);
        r * 2.0 - 1.0
    };
    let b_f32: Vec<f32> = (0..(n * k)).map(|_| rand() * 0.5).collect();
    let mut b_q: Vec<u8> = Vec::new();
    let mut row_q = Vec::new();
    for ni in 0..n as usize {
        let row = &b_f32[ni * k as usize..(ni + 1) * k as usize];
        quantize_row_q8_0(row, &mut row_q);
        b_q.extend_from_slice(&row_q);
    }
    // Reference: dequant the same bytes back to f32, then i8-quantize.
    let mut b_deq = vec![0f32; (n * k) as usize];
    dequantize_row_q8_0(&b_q, &mut b_deq);
    let (got_i8, got_scale, exp_i8, exp_scale) =
        pollster::block_on(run_one(QuantKind::Q8_0, n, k, &b_q, &b_deq));
    assert_eq!(got_scale.len(), exp_scale.len());
    for (i, (g, e)) in got_scale.iter().zip(&exp_scale).enumerate() {
        assert!(
            (g - e).abs() <= 1e-5 * e.abs().max(1e-6),
            "scale[{i}] gpu={g} cpu={e}"
        );
    }
    let mismatches: usize = got_i8
        .iter()
        .zip(&exp_i8)
        .filter(|(g, e)| (i32::from(**g) - i32::from(**e)).abs() > 1)
        .count();
    assert_eq!(mismatches, 0);
}

/// Q4_0 round-trip: same shape as Q8_0 but the underlying dequant loses
/// nibbles of precision. The CPU reference uses the same dequant, so the
/// final i8 result should still agree within ±1 ULP.
#[test]
fn dequant_i8_q4_0_numeric() {
    let n = 2u32;
    let k = 64u32;
    let mut s = 0xBEEF_u64;
    let mut rand = || -> f32 {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let r = ((s >> 33) as u32 as f32) / (u32::MAX as f32);
        r * 2.0 - 1.0
    };
    let b_f32: Vec<f32> = (0..(n * k)).map(|_| rand() * 0.5).collect();
    let mut b_q: Vec<u8> = Vec::new();
    let mut row_q = Vec::new();
    for ni in 0..n as usize {
        let row = &b_f32[ni * k as usize..(ni + 1) * k as usize];
        quantize_row_q4_0(row, &mut row_q);
        b_q.extend_from_slice(&row_q);
    }
    let mut b_deq = vec![0f32; (n * k) as usize];
    dequantize_row_q4_0(&b_q, &mut b_deq);
    let (got_i8, got_scale, exp_i8, exp_scale) =
        pollster::block_on(run_one(QuantKind::Q4_0, n, k, &b_q, &b_deq));
    for (i, (g, e)) in got_scale.iter().zip(&exp_scale).enumerate() {
        assert!(
            (g - e).abs() <= 1e-5 * e.abs().max(1e-6),
            "scale[{i}] gpu={g} cpu={e}"
        );
    }
    let mismatches: usize = got_i8
        .iter()
        .zip(&exp_i8)
        .filter(|(g, e)| (i32::from(**g) - i32::from(**e)).abs() > 1)
        .count();
    assert_eq!(mismatches, 0);
}
