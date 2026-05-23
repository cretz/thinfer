//! GPU activation int8 quantizer (`thinfer_core::ops::act_quant`) vs scalar
//! Rust reference. Self-contained: synthesizes f16 inputs, runs the WGSL
//! shader once, compares the packed-i8 buffer and the per-(M, K/32) scale
//! buffer against a transcription of the kernel's quantization rule.
//!
//! Primary value: building the pipeline catches WGSL parse/validate errors
//! before any e2e+model-load round trip (reserved-keyword regressions,
//! unknown enable directives, binding-kind mismatches).

#![cfg(feature = "conformance")]

use thinfer_core::backend::{Backend, BufRef, WgpuBackend};
use thinfer_core::ops::act_quant::{ActQuantBufs, build_wgsl, dispatch_act_quant, hint, layout};

fn pack_dims_u32x4(m: u32, k: u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&m.to_le_bytes());
    out[4..8].copy_from_slice(&k.to_le_bytes());
    out
}

/// f32 -> IEEE-754 binary16 (round-to-nearest-even), returning the 16-bit
/// pattern. Standard ARM/x86 vcvtps2ph algorithm; finite values only — the
/// test synthesizes inputs in [-0.5, 0.5] so subnormals/inf/NaN don't occur.
fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 31) & 1) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let mant = b & 0x7FFFFF;
    if exp == 0 {
        return sign << 15; // ±0
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1F {
        // overflow -> ±inf
        return (sign << 15) | (0x1F << 10);
    }
    if new_exp <= 0 {
        // subnormal or underflow -> 0 (test inputs avoid this regime)
        return sign << 15;
    }
    // Round-to-nearest-even on the 13 dropped mantissa bits.
    let m10 = mant >> 13;
    let rem = mant & 0x1FFF;
    let half_bit = 0x1000;
    let round_up = rem > half_bit || (rem == half_bit && (m10 & 1) == 1);
    let mut out_mant = m10 + if round_up { 1 } else { 0 };
    let mut out_exp = new_exp as u16;
    if out_mant == 0x400 {
        out_mant = 0;
        out_exp += 1;
    }
    (sign << 15) | (out_exp << 10) | (out_mant as u16 & 0x3FF)
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        return f32::from_bits(sign);
    }
    if exp == 0x1F {
        return f32::from_bits(sign | 0x7F80_0000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
}

/// Round f32 to f16 then back to f32 (the value the shader actually sees
/// when reading the `array<vec2<f16>>` input).
fn round_f32_to_f16_f32(x: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(x))
}

/// CPU reference. For each (m, sub_k) where sub_k = K/32, find absmax over
/// the 32 elements (in the rounded-to-f16 domain), scale = absmax/127,
/// quantize each elem to i8 via round-half-to-even (we use `round()` which
/// is round-half-away-from-zero on Rust f32; the kernel uses WGSL `round()`
/// which is the same), clamp to [-127, 127].
fn cpu_act_quant_ref(a_f32: &[f32], m: u32, k: u32) -> (Vec<i8>, Vec<f32>) {
    assert!(k.is_multiple_of(32));
    let m = m as usize;
    let k = k as usize;
    let blocks = k / 32;
    let mut i8_out = vec![0i8; m * k];
    let mut scale_out = vec![0f32; m * blocks];
    for mi in 0..m {
        for sb in 0..blocks {
            let off = mi * k + sb * 32;
            let block = &a_f32[off..off + 32];
            let absmax = block.iter().copied().fold(0f32, |a, x| a.max(x.abs()));
            let scale = absmax / 127.0;
            scale_out[mi * blocks + sb] = scale;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            for (i, &v) in block.iter().enumerate() {
                let q = (v * inv).round().clamp(-127.0, 127.0) as i32;
                i8_out[off + i] = q as i8;
            }
        }
    }
    (i8_out, scale_out)
}

async fn run_one(m: u32, k: u32, seed: u64) -> (Vec<i8>, Vec<f32>, Vec<i8>, Vec<f32>) {
    let backend = WgpuBackend::new().await.expect("wgpu adapter");
    assert!(
        backend.supports_shader_f16(),
        "act_quant requires SHADER_F16; this adapter does not expose it"
    );

    // Deterministic LCG inputs in the f16-safe range.
    let mut s = seed;
    let mut rand = || -> f32 {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let r = ((s >> 33) as u32 as f32) / (u32::MAX as f32);
        r * 2.0 - 1.0
    };
    let a_raw: Vec<f32> = (0..(m * k)).map(|_| rand() * 0.5).collect();
    // Round to f16 so the CPU reference sees the same values the GPU does.
    let a_f16: Vec<f32> = a_raw.iter().copied().map(round_f32_to_f16_f32).collect();

    let (exp_i8, exp_scale) = cpu_act_quant_ref(&a_f16, m, k);

    let wgsl = build_wgsl();
    let pipeline = backend
        .create_pipeline(&wgsl, "main", layout())
        .await
        .expect("act_quant pipeline");

    // Pack input as vec2<f16>: two K-adjacent values per u32.
    assert!(k.is_multiple_of(2));
    let mut a_bytes: Vec<u8> = Vec::with_capacity((m * k) as usize * 2);
    for mi in 0..m as usize {
        for ki in (0..k as usize).step_by(2) {
            let lo = f32_to_f16_bits(a_raw[mi * k as usize + ki]);
            let hi = f32_to_f16_bits(a_raw[mi * k as usize + ki + 1]);
            a_bytes.extend_from_slice(&lo.to_le_bytes());
            a_bytes.extend_from_slice(&hi.to_le_bytes());
        }
    }

    let alloc_with = |bytes: &[u8]| -> BufRef {
        let id = backend.allocate(bytes.len() as u64).expect("allocate");
        backend.write_buffer(id, 0, bytes).expect("write");
        BufRef::new(id, bytes.len() as u64)
    };
    let a_buf = alloc_with(&a_bytes);
    let dims_buf = alloc_with(&pack_dims_u32x4(m, k));
    let i8_len = (m as u64) * (k as u64); // 1 byte per i8 elem
    let scale_len = (m as u64) * (k as u64 / 32) * 4;
    let i8_id = backend.allocate(i8_len).expect("alloc i8");
    let scale_id = backend.allocate(scale_len).expect("alloc scale");
    let i8_buf = BufRef::new(i8_id, i8_len);
    let scale_buf = BufRef::new(scale_id, scale_len);

    let mut enc = backend.create_command_encoder();
    dispatch_act_quant(
        &backend,
        &mut enc,
        &pipeline,
        &ActQuantBufs {
            a: &a_buf,
            out_i8: &i8_buf,
            out_scale: &scale_buf,
            dims: &dims_buf,
        },
        m,
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

    backend.free(a_buf.id);
    backend.free(dims_buf.id);
    backend.free(i8_id);
    backend.free(scale_id);

    (got_i8, got_scale, exp_i8, exp_scale)
}

#[test]
fn act_quant_pipeline_builds() {
    // Even if no GPU is available, the WGSL hint must be deterministic and
    // the layout slot count must match the four bindings.
    assert_eq!(hint(), "act_quant_i8-f16");
    assert_eq!(layout().len(), 4);
    // Build the source string itself — exercises the format!() but not naga.
    let src = build_wgsl();
    assert!(src.contains("enable f16"));
    assert!(src.contains("pack4xI8"));
}

#[test]
fn act_quant_small() {
    let (got_i8, got_scale, exp_i8, exp_scale) = pollster::block_on(run_one(2, 64, 0xC0FFEE));
    assert_eq!(got_scale.len(), exp_scale.len());
    for (i, (g, e)) in got_scale.iter().zip(&exp_scale).enumerate() {
        let err = (g - e).abs();
        assert!(
            err <= 1e-5 * e.abs().max(1e-6),
            "scale[{i}] gpu={g} cpu={e}"
        );
    }
    assert_eq!(got_i8.len(), exp_i8.len());
    // ±1 i8 ULP tolerance: GPU and CPU rounding modes for round() agree on
    // ties-away-from-zero, but f16-quantized inputs at sub-block boundaries
    // can hit rounding ambiguity. Most cells exact; a handful may differ by 1.
    let mismatches: usize = got_i8
        .iter()
        .zip(&exp_i8)
        .filter(|(g, e)| (i32::from(**g) - i32::from(**e)).abs() > 1)
        .count();
    assert_eq!(
        mismatches,
        0,
        "{} cells differ by >1 ULP (out of {})",
        mismatches,
        got_i8.len()
    );
}

#[test]
fn act_quant_multi_block() {
    // K=128 = 4 sub-blocks per row, M=3 (non-power-of-two row count).
    let (got_i8, got_scale, exp_i8, exp_scale) = pollster::block_on(run_one(3, 128, 0xBEEF));
    assert_eq!(got_scale, exp_scale, "scale mismatch (multi-block)");
    let mismatches: usize = got_i8
        .iter()
        .zip(&exp_i8)
        .filter(|(g, e)| (i32::from(**g) - i32::from(**e)).abs() > 1)
        .count();
    assert_eq!(mismatches, 0);
}
