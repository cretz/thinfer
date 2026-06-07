//! GPU DP4A int8 matmul (`thinfer_core::ops::matmul_i8`) vs scalar Rust
//! reference. Skips cleanly when the adapter exposes neither
//! `Packed4x8IntegerDotProduct` nor `SHADER_F16`.
//!
//! Asymmetric A (llama.cpp Q8_1-style): per-block params `(scale, zero)` in
//! f16. Symmetric B: per-block scale only. Correction term `za * sb * Σ qb`
//! exercised by including a non-zero `za` in the test inputs.

#![cfg(feature = "conformance")]

use thinfer_core::backend::{Backend, BufRef, WgpuBackend};
use thinfer_core::ops::matmul_i8::{
    MatMulI8Bufs, MatMulI8Config, build_wgsl, dispatch_matmul_i8, hint, layout,
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

fn pack4_i8(q: [i8; 4]) -> u32 {
    (q[0] as u8 as u32)
        | ((q[1] as u8 as u32) << 8)
        | ((q[2] as u8 as u32) << 16)
        | ((q[3] as u8 as u32) << 24)
}

/// CPU reference, asymmetric A. Decomposes as:
///   `O = Σ_t [ sa * sb * Σ_k(qa*qb)  +  za * sb * Σ_k(qb) ]`
/// Same arithmetic the kernel does internally.
#[allow(clippy::too_many_arguments)]
fn cpu_matmul_i8_ref(
    a_i8: &[i8],
    a_scale: &[f32], // f16-rounded
    a_zero: &[f32],  // f16-rounded
    b_i8: &[i8],
    b_scale: &[f32],
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
                let mut dot_i: i32 = 0;
                let mut qsum_b: i32 = 0;
                for kj in 0..32 {
                    let av = a_i8[mi * k + t * 32 + kj] as i32;
                    let bv = b_i8[ni * k + t * 32 + kj] as i32;
                    dot_i += av * bv;
                    qsum_b += bv;
                }
                let sa = a_scale[mi * blocks + t];
                let za = a_zero[mi * blocks + t];
                let sb = b_scale[ni * blocks + t];
                acc += (dot_i as f32) * sa * sb + za * sb * (qsum_b as f32);
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
    cfg: MatMulI8Config,
    seed: u64,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let backend = WgpuBackend::new().await.expect("wgpu adapter");
    if !backend.supports_packed_int_dot() {
        eprintln!("skip: no Packed4x8IntegerDotProduct");
        return None;
    }
    if !backend.supports_shader_f16() {
        eprintln!("skip: no SHADER_F16");
        return None;
    }
    if cfg.use_subgroup && !backend.supports_subgroups() {
        eprintln!("skip: no Features::SUBGROUP");
        return None;
    }

    let blocks = k / 32;
    let mut s = seed;
    let mut rand_i8 = || -> i8 {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((s >> 56) as i32 - 128).clamp(-127, 127) as i8
    };
    let a_i8: Vec<i8> = (0..(m as usize * k as usize)).map(|_| rand_i8()).collect();
    let b_i8: Vec<i8> = (0..(n as usize * k as usize)).map(|_| rand_i8()).collect();
    let mut s2 = seed ^ 0xABCD_EF01;
    let mut rand_scale = || -> f32 {
        s2 = s2.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let r = ((s2 >> 40) as u32 as f32) / ((1u32 << 24) as f32);
        1e-3 + r * 9e-3
    };
    let mut s3 = seed ^ 0x1234_5678;
    let mut rand_zero = || -> f32 {
        s3 = s3.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let r = ((s3 >> 40) as u32 as f32) / ((1u32 << 24) as f32);
        // Range [-0.3, 0.3] — small but non-trivial; exercises the correction.
        r * 0.6 - 0.3
    };
    // a_scale and a_zero are f16 in the on-GPU params buffer; reference
    // must use the f16-rounded values to match what the kernel sees.
    let a_scale: Vec<f32> = (0..(m as usize * blocks as usize))
        .map(|_| f16_round(rand_scale()))
        .collect();
    let a_zero: Vec<f32> = (0..(m as usize * blocks as usize))
        .map(|_| f16_round(rand_zero()))
        .collect();
    let b_scale: Vec<f32> = (0..(n as usize * blocks as usize))
        .map(|_| rand_scale())
        .collect();
    // Precompute b_qsum (matches what dequant_i8 produces).
    let b_qsum: Vec<f32> = {
        let mut out = vec![0f32; n as usize * blocks as usize];
        for ni in 0..n as usize {
            for t in 0..blocks as usize {
                let mut s: i32 = 0;
                for kj in 0..32 {
                    s += b_i8[ni * k as usize + t * 32 + kj] as i32;
                }
                out[ni * blocks as usize + t] = s as f32;
            }
        }
        out
    };

    let expected = cpu_matmul_i8_ref(&a_i8, &a_scale, &a_zero, &b_i8, &b_scale, m, n, k);

    let wgsl = build_wgsl(&cfg);
    let pipeline = backend
        .create_pipeline(&wgsl, "main", layout())
        .await
        .expect("matmul_i8 pipeline");

    assert!(k.is_multiple_of(4));
    let words_per_row = k as usize / 4;
    let pack_rows = |src: &[i8], rows: usize| -> Vec<u8> {
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
    let a_bytes = pack_rows(&a_i8, m as usize);
    let b_bytes = pack_rows(&b_i8, n as usize);
    // a_params: vec2<f16> per block per row, (scale, zero).
    let mut a_params_bytes: Vec<u8> = Vec::with_capacity(a_scale.len() * 4);
    for i in 0..a_scale.len() {
        a_params_bytes.extend_from_slice(&f32_to_f16_bits(a_scale[i]).to_le_bytes());
        a_params_bytes.extend_from_slice(&f32_to_f16_bits(a_zero[i]).to_le_bytes());
    }
    let b_scale_bytes: Vec<u8> = b_scale.iter().flat_map(|x| x.to_le_bytes()).collect();
    let b_qsum_bytes: Vec<u8> = b_qsum.iter().flat_map(|x| x.to_le_bytes()).collect();

    let alloc_with = |bytes: &[u8]| -> BufRef {
        let id = backend.allocate(bytes.len() as u64).expect("allocate");
        backend.write_buffer(id, 0, bytes).expect("write");
        BufRef::new(id, bytes.len() as u64)
    };
    let a_buf = alloc_with(&a_bytes);
    let a_params_buf = alloc_with(&a_params_bytes);
    let b_buf = alloc_with(&b_bytes);
    let b_sc_buf = alloc_with(&b_scale_bytes);
    let b_qsum_buf = alloc_with(&b_qsum_bytes);
    let dims_buf = alloc_with(&pack_dims_u32x4(m, n, k));
    // Per-K-block debug trace slots: disabled (dbg.enable = 0), minimal
    // dbg_out placeholder (the kernel writes nothing when disabled).
    let dbg_out_buf = alloc_with(&[0u8; 4]);
    let dbg_buf = alloc_with(&pack_dims_u32x4(0, 0, 0));
    let out_len = (m as u64) * (n as u64) * 2;
    let out_id = backend.allocate(out_len).expect("alloc out");
    let out_buf = BufRef::new(out_id, out_len);

    let mut enc = backend.create_command_encoder();
    dispatch_matmul_i8(
        &backend,
        &mut enc,
        &pipeline,
        &cfg,
        &MatMulI8Bufs {
            a: &a_buf,
            a_params: &a_params_buf,
            b: &b_buf,
            b_scale: &b_sc_buf,
            b_qsum: &b_qsum_buf,
            out: &out_buf,
            dims: &dims_buf,
            dbg_out: &dbg_out_buf,
            dbg: &dbg_buf,
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
    backend.free(b_sc_buf.id);
    backend.free(b_qsum_buf.id);
    backend.free(dims_buf.id);
    backend.free(dbg_out_buf.id);
    backend.free(dbg_buf.id);
    backend.free(out_id);

    Some((got, expected))
}

#[test]
fn matmul_i8_wgsl_sanity() {
    let cfg = MatMulI8Config::DEFAULT;
    let src = build_wgsl(&cfg);
    assert!(src.contains("enable f16"));
    assert!(src.contains("dot4I8Packed"));
    assert!(
        !src.contains("enable packed_4x8_integer_dot_product"),
        "matmul_i8 must not emit unknown `enable packed_4x8_integer_dot_product` directive"
    );
    // 9 slots: a, a_params, b, b_scale, b_qsum, out, dims + the per-K-block
    // debug trace pair (dbg_out, dbg).
    assert_eq!(layout().len(), 9);
    assert!(hint(&cfg).starts_with("matmul_i8-"));
}

fn check(got: &[f32], exp: &[f32]) {
    let max_abs_ref = exp.iter().copied().map(f32::abs).fold(0f32, f32::max);
    let tol = (5e-3 * max_abs_ref).max(1e-4);
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
        "max abs diff {max_err:.3e} > tol {tol:.3e} at idx {idx} \
         (got={} exp={}) ref_max={}",
        got[idx],
        exp[idx],
        max_abs_ref
    );
}

#[test]
fn matmul_i8_small() {
    let cfg = MatMulI8Config {
        bm: 8,
        bn: 8,
        tm: 2,
        tn: 2,
        use_subgroup: false,
    };
    let pair = pollster::block_on(try_run(8, 8, 32, cfg, 0xC0DE_F00D));
    let Some((got, exp)) = pair else { return };
    check(&got, &exp);
}

#[test]
fn matmul_i8_multi_block_k() {
    let cfg = MatMulI8Config {
        bm: 8,
        bn: 8,
        tm: 2,
        tn: 2,
        use_subgroup: false,
    };
    let pair = pollster::block_on(try_run(8, 8, 64, cfg, 0xDEAD_BEEF));
    let Some((got, exp)) = pair else { return };
    check(&got, &exp);
}

#[test]
fn matmul_i8_small_subgroup() {
    let cfg = MatMulI8Config {
        bm: 8,
        bn: 8,
        tm: 2,
        tn: 2,
        use_subgroup: true,
    };
    let pair = pollster::block_on(try_run(8, 8, 32, cfg, 0xC0DE_F00D));
    let Some((got, exp)) = pair else { return };
    check(&got, &exp);
}

#[test]
fn matmul_i8_multi_block_k_subgroup() {
    let cfg = MatMulI8Config {
        bm: 8,
        bn: 8,
        tm: 2,
        tn: 2,
        use_subgroup: true,
    };
    let pair = pollster::block_on(try_run(8, 8, 64, cfg, 0xDEAD_BEEF));
    let Some((got, exp)) = pair else { return };
    check(&got, &exp);
}

#[test]
fn matmul_i8_default_tile_subgroup() {
    let cfg = MatMulI8Config {
        use_subgroup: true,
        ..MatMulI8Config::DEFAULT
    };
    let pair = pollster::block_on(try_run(64, 64, 128, cfg, 0xFEED_FACE));
    let Some((got, exp)) = pair else { return };
    check(&got, &exp);
}
