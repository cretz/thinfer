//! Mixed-precision matmul (`thinfer_core::ops::matmul_i8_bf16`) vs scalar
//! Rust reference. A is paired packed-i8 with **asymmetric** per-K=32 params
//! `(scale, zero)` in f16; B is dense bf16 weights stored K-major (`array<u32>`,
//! 2 bf16 per word) with sibling per-K-block sum (`b_sum`, f32). Output is
//! paired `vec2<f16>`.
//!
//! Skips cleanly when the adapter lacks `SHADER_F16`. The kernel does NOT
//! depend on `Packed4x8IntegerDotProduct` (B is bf16, not int8).

#![cfg(feature = "conformance")]

use thinfer_core::backend::{Backend, BufRef, WgpuBackend};
use thinfer_core::ops::matmul_i8_bf16::{
    MatMulI8Bf16Bufs, MatMulI8Bf16Config, build_wgsl, dispatch_matmul_i8_bf16, layout,
};

fn pack_dims_u32x4(m: u32, n: u32, k: u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&m.to_le_bytes());
    out[4..8].copy_from_slice(&n.to_le_bytes());
    out[8..12].copy_from_slice(&k.to_le_bytes());
    out
}

fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 31) & 1) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let mant = b & 0x7FFFFF;
    if exp == 0 {
        return sign << 15;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1F {
        return (sign << 15) | (0x1F << 10);
    }
    if new_exp <= 0 {
        return sign << 15;
    }
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

fn f16_round(x: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(x))
}

fn f32_to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if (b & 0x7F80_0000) == 0x7F80_0000 {
        return ((b >> 16) & 0xFFFF) as u16;
    }
    let l = (b >> 16) & 1;
    (((b + 0x7FFF + l) >> 16) & 0xFFFF) as u16
}

fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

fn pack4_i8(q: [i8; 4]) -> u32 {
    (q[0] as u8 as u32)
        | ((q[1] as u8 as u32) << 8)
        | ((q[2] as u8 as u32) << 16)
        | ((q[3] as u8 as u32) << 24)
}

/// CPU reference, asymmetric A.
///   `O = Σ_t [ sa * Σ_k(qa*b)  +  za * Σ_k(b) ]`
#[allow(clippy::too_many_arguments)]
fn cpu_matmul_i8_bf16_ref(
    a_i8: &[i8],
    a_scale: &[f32],
    a_zero: &[f32],
    b: &[f32], // bf16-rounded values, K-major [K, N]
    m: u32,
    n: u32,
    k: u32,
) -> Vec<f32> {
    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let blocks = k / 32;
    let mut out = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f32;
            for t in 0..blocks {
                let sa = a_scale[mi * blocks + t];
                let za = a_zero[mi * blocks + t];
                let mut partial = 0f32;
                let mut bsum = 0f32;
                for kj in 0..32 {
                    let av = a_i8[mi * k + t * 32 + kj] as f32;
                    let bv = b[(t * 32 + kj) * n + ni];
                    partial += av * bv;
                    bsum += bv;
                }
                acc += partial * sa + za * bsum;
            }
            out[mi * n + ni] = acc.clamp(-65504.0, 65504.0);
        }
    }
    out
}

async fn try_run(
    m: u32,
    n: u32,
    k: u32,
    cfg: MatMulI8Bf16Config,
    seed: u64,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let backend = WgpuBackend::new().await.expect("wgpu adapter");
    if !backend.supports_shader_f16() {
        eprintln!("skip: no SHADER_F16");
        return None;
    }
    assert!(k.is_multiple_of(32));
    assert!(n.is_multiple_of(2));

    let blocks = k / 32;
    let mut s = seed;
    let mut rand_i8 = || -> i8 {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((s >> 56) as i32 - 128).clamp(-127, 127) as i8
    };
    let a_i8: Vec<i8> = (0..(m as usize * k as usize)).map(|_| rand_i8()).collect();
    let mut s2 = seed ^ 0xABCD_EF01;
    let mut rand_scale = || -> f32 {
        s2 = s2.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let r = ((s2 >> 40) as u32 as f32) / ((1u32 << 24) as f32);
        1e-3 + r * 9e-3
    };
    let mut s_z = seed ^ 0x1234_5678;
    let mut rand_zero = || -> f32 {
        s_z = s_z.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let r = ((s_z >> 40) as u32 as f32) / ((1u32 << 24) as f32);
        r * 0.6 - 0.3
    };
    let a_scale: Vec<f32> = (0..(m as usize * blocks as usize))
        .map(|_| f16_round(rand_scale()))
        .collect();
    let a_zero: Vec<f32> = (0..(m as usize * blocks as usize))
        .map(|_| f16_round(rand_zero()))
        .collect();
    let mut s3 = seed ^ 0x5A5A_C3C3;
    let mut rand_bf16 = || -> f32 {
        s3 = s3.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let r = ((s3 >> 40) as u32 as f32) / ((1u32 << 24) as f32);
        let v = (r - 0.5) * 0.1;
        bf16_bits_to_f32(f32_to_bf16_bits(v))
    };
    let b_f32: Vec<f32> = (0..(k as usize * n as usize))
        .map(|_| rand_bf16())
        .collect();

    // Precompute b_sum: per (n, k_block) sum of B over its 32-elem K-block.
    let b_sum: Vec<f32> = {
        let mut out = vec![0f32; n as usize * blocks as usize];
        for ni in 0..n as usize {
            for t in 0..blocks as usize {
                let mut s = 0f32;
                for kj in 0..32 {
                    s += b_f32[(t * 32 + kj) * n as usize + ni];
                }
                out[ni * blocks as usize + t] = s;
            }
        }
        out
    };

    let expected = cpu_matmul_i8_bf16_ref(&a_i8, &a_scale, &a_zero, &b_f32, m, n, k);

    let wgsl = build_wgsl(&cfg);
    let pipeline = backend
        .create_pipeline(&wgsl, "main", layout())
        .await
        .expect("matmul_i8_bf16 pipeline");

    let words_per_row = k as usize / 4;
    let pack_a = |src: &[i8], rows: usize| -> Vec<u8> {
        let mut out: Vec<u8> = Vec::with_capacity(rows * words_per_row * 4);
        for r in 0..rows {
            for w in 0..words_per_row {
                let base = r * k as usize + w * 4;
                let word = pack4_i8([src[base], src[base + 1], src[base + 2], src[base + 3]]);
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        out
    };
    let a_bytes = pack_a(&a_i8, m as usize);
    let b_bytes: Vec<u8> = {
        let mut out: Vec<u8> = Vec::with_capacity(k as usize * n as usize * 2);
        for kk in 0..(k as usize) {
            let mut nn = 0;
            while nn < n as usize {
                let lo = f32_to_bf16_bits(b_f32[kk * n as usize + nn]);
                let hi = f32_to_bf16_bits(b_f32[kk * n as usize + nn + 1]);
                let word: u32 = (lo as u32) | ((hi as u32) << 16);
                out.extend_from_slice(&word.to_le_bytes());
                nn += 2;
            }
        }
        out
    };
    let mut a_params_bytes: Vec<u8> = Vec::with_capacity(a_scale.len() * 4);
    for i in 0..a_scale.len() {
        a_params_bytes.extend_from_slice(&f32_to_f16_bits(a_scale[i]).to_le_bytes());
        a_params_bytes.extend_from_slice(&f32_to_f16_bits(a_zero[i]).to_le_bytes());
    }
    let b_sum_bytes: Vec<u8> = b_sum.iter().flat_map(|x| x.to_le_bytes()).collect();

    let alloc_with = |bytes: &[u8]| -> BufRef {
        let id = backend.allocate(bytes.len() as u64).expect("allocate");
        backend.write_buffer(id, 0, bytes).expect("write");
        BufRef::new(id, bytes.len() as u64)
    };
    let a_buf = alloc_with(&a_bytes);
    let a_params_buf = alloc_with(&a_params_bytes);
    let b_buf = alloc_with(&b_bytes);
    let b_sum_buf = alloc_with(&b_sum_bytes);
    let dims_buf = alloc_with(&pack_dims_u32x4(m, n, k));
    let out_len = (m as u64) * (n as u64) * 2;
    let out_id = backend.allocate(out_len).expect("alloc out");
    let out_buf = BufRef::new(out_id, out_len);

    let mut enc = backend.create_command_encoder();
    dispatch_matmul_i8_bf16(
        &backend,
        &mut enc,
        &pipeline,
        &cfg,
        &MatMulI8Bf16Bufs {
            a: &a_buf,
            a_params: &a_params_buf,
            b: &b_buf,
            b_sum: &b_sum_buf,
            out: &out_buf,
            dims: &dims_buf,
        },
        m,
        n,
    )
    .expect("dispatch");
    backend.submit(enc).await.expect("submit");

    let out_bytes = backend.read_buffer(out_id, 0, out_len).await.expect("read");
    let got: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();

    backend.free(a_buf.id);
    backend.free(a_params_buf.id);
    backend.free(b_buf.id);
    backend.free(b_sum_buf.id);
    backend.free(dims_buf.id);
    backend.free(out_id);

    Some((got, expected))
}

fn check(got: &[f32], exp: &[f32], scale: f32) {
    let max_abs_ref = exp.iter().copied().map(f32::abs).fold(0f32, f32::max);
    let tol = (scale * max_abs_ref).max(1e-4);
    let nan_count = got.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(nan_count, 0, "GPU output has {nan_count} NaN/inf cells");
    let (max_err, idx) = got
        .iter()
        .zip(exp)
        .enumerate()
        .map(|(i, (g, e))| ((g - e).abs(), i))
        .fold((0f32, 0usize), |a, b| if b.0 > a.0 { b } else { a });
    assert!(
        max_err <= tol,
        "max abs diff {max_err:.3e} > tol {tol:.3e} at idx {idx} (got={} exp={}) ref_max={}",
        got[idx],
        exp[idx],
        max_abs_ref
    );
}

#[test]
fn matmul_i8_bf16_wgsl_sanity() {
    let cfg = MatMulI8Bf16Config::DEFAULT;
    let src = build_wgsl(&cfg);
    assert!(src.contains("enable f16"));
    assert!(src.contains("unpack_bf16"));
    assert!(
        !src.contains("dot4I8Packed"),
        "matmul_i8_bf16 must not use DP4A (B is bf16)"
    );
    assert_eq!(layout().len(), 6);
}

#[test]
fn matmul_i8_bf16_small() {
    let cfg = MatMulI8Bf16Config {
        bm: 8,
        bn: 8,
        tm: 2,
        tn: 2,
    };
    let pair = pollster::block_on(try_run(8, 8, 32, cfg, 0xC0DE_F00D));
    let Some((got, exp)) = pair else { return };
    check(&got, &exp, 5e-3);
}

#[test]
fn matmul_i8_bf16_multi_block_k() {
    let cfg = MatMulI8Bf16Config {
        bm: 8,
        bn: 8,
        tm: 2,
        tn: 2,
    };
    let pair = pollster::block_on(try_run(8, 8, 64, cfg, 0xDEAD_BEEF));
    let Some((got, exp)) = pair else { return };
    check(&got, &exp, 5e-3);
}

#[test]
fn matmul_i8_bf16_default_tile() {
    let cfg = MatMulI8Bf16Config::DEFAULT;
    let pair = pollster::block_on(try_run(64, 64, 128, cfg, 0xFEED_FACE));
    let Some((got, exp)) = pair else { return };
    check(&got, &exp, 5e-3);
}
