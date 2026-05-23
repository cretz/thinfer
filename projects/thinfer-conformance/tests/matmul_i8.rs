//! GPU DP4A int8 matmul (`thinfer_core::ops::matmul_i8`) vs scalar Rust
//! reference. Skips cleanly when the adapter exposes neither
//! `Packed4x8IntegerDotProduct` nor `SHADER_F16`.
//!
//! Primary value: builds the WGSL (catches enable-directive / reserved-
//! keyword regressions that would otherwise only surface at e2e model load)
//! AND verifies numerical correctness on a small case.

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

/// Pack 4 i8 values into a u32 the same way `pack4xI8` does: byte i (low to
/// high) holds lane i.
fn pack4_i8(q: [i8; 4]) -> u32 {
    (q[0] as u8 as u32)
        | ((q[1] as u8 as u32) << 8)
        | ((q[2] as u8 as u32) << 16)
        | ((q[3] as u8 as u32) << 24)
}

/// CPU reference. Inputs are raw i8 + per-(K/32) scale.
fn cpu_matmul_i8_ref(
    a_i8: &[i8],
    a_scale: &[f32],
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
                for kj in 0..32 {
                    let av = a_i8[mi * k + t * 32 + kj] as i32;
                    let bv = b_i8[ni * k + t * 32 + kj] as i32;
                    dot_i += av * bv;
                }
                acc += (dot_i as f32) * a_scale[mi * blocks + t] * b_scale[ni * blocks + t];
            }
            // Saturated narrow at ±65504 (matches the kernel output path).
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
    let (sg_min, sg_max) = backend.subgroup_size_range();
    eprintln!(
        "adapter caps: packed_int_dot={} shader_f16={} subgroups={} subgroup_size=[{}, {}]",
        backend.supports_packed_int_dot(),
        backend.supports_shader_f16(),
        backend.supports_subgroups(),
        sg_min,
        sg_max,
    );
    if !backend.supports_packed_int_dot() {
        eprintln!("skip: adapter does not expose Packed4x8IntegerDotProduct");
        return None;
    }
    if !backend.supports_shader_f16() {
        eprintln!("skip: adapter does not expose SHADER_F16");
        return None;
    }
    if cfg.use_subgroup && !backend.supports_subgroups() {
        eprintln!("skip: cfg.use_subgroup=true but adapter lacks Features::SUBGROUP");
        return None;
    }
    eprintln!("running matmul_i8: m={} n={} k={} cfg={:?}", m, n, k, cfg);

    let blocks = k / 32;
    let mut s = seed;
    let mut rand_i8 = || -> i8 {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        // Range [-127, 127], avoid -128 to match the kernel's invariant.
        ((s >> 56) as i32 - 128).clamp(-127, 127) as i8
    };
    let a_i8: Vec<i8> = (0..(m as usize * k as usize)).map(|_| rand_i8()).collect();
    let b_i8: Vec<i8> = (0..(n as usize * k as usize)).map(|_| rand_i8()).collect();
    // Scales: small positive f32 (one per sub-block per row).
    let mut s2 = seed ^ 0xABCD_EF01;
    let mut rand_scale = || -> f32 {
        s2 = s2.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let r = ((s2 >> 40) as u32 as f32) / ((1u32 << 24) as f32);
        // Avoid zero scales; range roughly [1e-3, 0.01].
        1e-3 + r * 9e-3
    };
    let a_scale: Vec<f32> = (0..(m as usize * blocks as usize))
        .map(|_| rand_scale())
        .collect();
    let b_scale: Vec<f32> = (0..(n as usize * blocks as usize))
        .map(|_| rand_scale())
        .collect();

    let expected = cpu_matmul_i8_ref(&a_i8, &a_scale, &b_i8, &b_scale, m, n, k);

    let wgsl = build_wgsl(&cfg);
    let pipeline = backend
        .create_pipeline(&wgsl, "main", layout())
        .await
        .expect("matmul_i8 pipeline");

    // Pack a_i8 / b_i8 into u32 words (4 i8 per word, K-major).
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
    let a_scale_bytes: Vec<u8> = a_scale.iter().flat_map(|x| x.to_le_bytes()).collect();
    let b_scale_bytes: Vec<u8> = b_scale.iter().flat_map(|x| x.to_le_bytes()).collect();

    let alloc_with = |bytes: &[u8]| -> BufRef {
        let id = backend.allocate(bytes.len() as u64).expect("allocate");
        backend.write_buffer(id, 0, bytes).expect("write");
        BufRef::new(id, bytes.len() as u64)
    };
    let a_buf = alloc_with(&a_bytes);
    let a_sc_buf = alloc_with(&a_scale_bytes);
    let b_buf = alloc_with(&b_bytes);
    let b_sc_buf = alloc_with(&b_scale_bytes);
    let dims_buf = alloc_with(&pack_dims_u32x4(m, n, k));
    let out_len = (m as u64) * (n as u64) * 2; // f16 per cell
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
            a_scale: &a_sc_buf,
            b: &b_buf,
            b_scale: &b_sc_buf,
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
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            f16_bits_to_f32(bits)
        })
        .collect();

    backend.free(a_buf.id);
    backend.free(a_sc_buf.id);
    backend.free(b_buf.id);
    backend.free(b_sc_buf.id);
    backend.free(dims_buf.id);
    backend.free(out_id);

    Some((got, expected))
}

#[test]
fn matmul_i8_wgsl_sanity() {
    let cfg = MatMulI8Config::DEFAULT;
    let src = build_wgsl(&cfg);
    assert!(src.contains("enable f16"));
    assert!(src.contains("dot4I8Packed"));
    // Reject the unknown enable-extension that triggered the previous naga
    // failure — `dot4I8Packed` is gated at the API level via
    // WgslLanguageFeatures::Packed4x8IntegerDotProduct, not via an `enable`
    // directive.
    assert!(
        !src.contains("enable packed_4x8_integer_dot_product"),
        "matmul_i8 must not emit unknown `enable packed_4x8_integer_dot_product` directive"
    );
    assert_eq!(layout().len(), 6);
    assert!(hint(&cfg).starts_with("matmul_i8-"));
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
    // Output is f16-quantized. Reference magnitudes are scales*scales*dot
    // ≈ (5e-3)*(5e-3)*(127*127*32) ≈ 0.013 typical, |max| ~ 0.05. f16 ULP
    // at that magnitude is ~4.9e-5; tolerance 5e-3 generously covers it.
    let max_abs_ref = exp.iter().copied().map(f32::abs).fold(0f32, f32::max);
    let tol = (5e-3 * max_abs_ref).max(1e-4);
    let (max_err, idx) = got
        .iter()
        .zip(&exp)
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
fn matmul_i8_multi_block_k() {
    // K=64 = 2 sub-blocks → exercises the t-loop in the kernel.
    let cfg = MatMulI8Config {
        bm: 8,
        bn: 8,
        tm: 2,
        tn: 2,
        use_subgroup: false,
    };
    let pair = pollster::block_on(try_run(8, 8, 64, cfg, 0xDEAD_BEEF));
    let Some((got, exp)) = pair else { return };
    let max_abs_ref = exp.iter().copied().map(f32::abs).fold(0f32, f32::max);
    let tol = (5e-3 * max_abs_ref).max(1e-4);
    let nan_count = got.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(nan_count, 0, "GPU output has {nan_count} NaN/inf cells");
    let max_err = got
        .iter()
        .zip(&exp)
        .map(|(g, e)| (g - e).abs())
        .fold(0f32, f32::max);
    assert!(max_err <= tol, "max abs diff {max_err:.3e} > tol {tol:.3e}");
}

/// Subgroup-enabled variant of `matmul_i8_small`. Skips cleanly when the
/// adapter lacks `Features::SUBGROUP` (via `try_run`'s gate). When it does
/// run, the eprintln in `try_run` documents the adapter's subgroup range
/// so you can confirm the subgroup-broadcast/shuffle path actually executed.
/// Same numerical tolerance as the non-subgroup variant — the subgroup ops
/// only change WHERE values are loaded from, not what they are.
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
    let max_abs_ref = exp.iter().copied().map(f32::abs).fold(0f32, f32::max);
    let tol = (5e-3 * max_abs_ref).max(1e-4);
    let (max_err, idx) = got
        .iter()
        .zip(&exp)
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

/// Subgroup-enabled multi-K-block variant. Exercises the inner t-loop
/// under the subgroup branch so the shuffle/broadcast select runs against
/// multiple sub-block accumulations. Skips cleanly without `Features::SUBGROUP`.
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
    let max_abs_ref = exp.iter().copied().map(f32::abs).fold(0f32, f32::max);
    let tol = (5e-3 * max_abs_ref).max(1e-4);
    let nan_count = got.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(nan_count, 0, "GPU output has {nan_count} NaN/inf cells");
    let max_err = got
        .iter()
        .zip(&exp)
        .map(|(g, e)| (g - e).abs())
        .fold(0f32, f32::max);
    assert!(max_err <= tol, "max abs diff {max_err:.3e} > tol {tol:.3e}");
}

/// Subgroup variant at the production-default tile shape (matches what
/// `block.rs` actually builds when `Features::SUBGROUP` is exposed).
/// Catches any divisibility/shared-mem-budget edge cases that the small
/// cases miss.
#[test]
fn matmul_i8_default_tile_subgroup() {
    let cfg = MatMulI8Config {
        use_subgroup: true,
        ..MatMulI8Config::DEFAULT
    };
    // M, N multiples of bm/bn; K multiple of 32. Use 64x64 to exactly
    // fill one WG and 128 K=4 sub-blocks of t-loop work.
    let pair = pollster::block_on(try_run(64, 64, 128, cfg, 0xFEED_FACE));
    let Some((got, exp)) = pair else { return };
    let max_abs_ref = exp.iter().copied().map(f32::abs).fold(0f32, f32::max);
    let tol = (5e-3 * max_abs_ref).max(1e-4);
    let nan_count = got.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(nan_count, 0, "GPU output has {nan_count} NaN/inf cells");
    let max_err = got
        .iter()
        .zip(&exp)
        .map(|(g, e)| (g - e).abs())
        .fold(0f32, f32::max);
    assert!(max_err <= tol, "max abs diff {max_err:.3e} > tol {tol:.3e}");
}
