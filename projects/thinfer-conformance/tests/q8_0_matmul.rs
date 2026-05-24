//! GPU matmul Q8_0 vs CPU dequant-then-matmul. Self-contained: no Python
//! reference, no safetensors fixture. The CPU side uses the same Rust
//! port of `dequantize_row_q8_0` as the ground truth, so this test
//! validates the WGSL kernel's dequant + matmul against an identical
//! formula expressed in scalar Rust.
//!
//! Layout reminder: GGUF Q8_0 stores weights as `[N, K]` blocks
//! (N-major). The matmul kernel views B in the same orientation; no
//! transpose at upload. Conventional matmul output is `Out = A @ B^T`
//! when B is `[N, K]`.

#![cfg(feature = "conformance")]

use thinfer_core::backend::{Backend, BufRef, WgpuBackend};
use thinfer_core::ops::{
    ActDtype, MatMulConfig, MatMulF32, MatmulBufs, MatmulOp, WeightDtype, WgslConfig,
    dispatch_matmul,
};
use thinfer_core::quant::{QuantKind, quantize_row_q8_0};

fn pack_dims_u32x4(m: u32, n: u32, k: u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&m.to_le_bytes());
    out[4..8].copy_from_slice(&n.to_le_bytes());
    out[8..12].copy_from_slice(&k.to_le_bytes());
    out
}

/// CPU reference: dequant Q8_0 B (N rows of K elems) and compute
/// `Out[m, n] = sum_k A[m, k] * B_deq[n, k]`. Matches the kernel's
/// access pattern.
fn cpu_matmul_q8_0_ref(a: &[f32], m: u32, k: u32, b_q: &[u8], n: u32) -> Vec<f32> {
    assert_eq!(a.len(), (m * k) as usize);
    assert_eq!(
        b_q.len(),
        (n as usize) * (k as usize / 32) * 34,
        "B must be N rows of K/32 Q8_0 blocks"
    );
    let mut b_deq = vec![0f32; (n as usize) * (k as usize)];
    thinfer_core::quant::dequantize_row_q8_0(b_q, &mut b_deq);
    let mut out = vec![0f32; (m as usize) * (n as usize)];
    for mi in 0..m as usize {
        for ni in 0..n as usize {
            let mut acc = 0f32;
            for ki in 0..k as usize {
                acc += a[mi * k as usize + ki] * b_deq[ni * k as usize + ki];
            }
            out[mi * n as usize + ni] = acc;
        }
    }
    out
}

async fn run_one(m: u32, n: u32, k: u32, seed: u64) -> (Vec<f32>, Vec<f32>) {
    run_one_with_cfg(
        m,
        n,
        k,
        seed,
        MatMulConfig {
            bm: 16,
            bn: 16,
            bk: 32,
            tm: 1,
            tn: 1,
            b_nmajor: false,
        },
    )
    .await
}

async fn run_one_with_cfg(
    m: u32,
    n: u32,
    k: u32,
    seed: u64,
    mm_cfg: MatMulConfig,
) -> (Vec<f32>, Vec<f32>) {
    assert!(
        k.is_multiple_of(32),
        "K must be multiple of Q8_0 block size"
    );

    // Deterministic input synthesis (LCG; conformance fixtures don't need
    // crypto-grade randomness).
    let mut s = seed;
    let mut rand = || -> f32 {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let r = ((s >> 33) as u32 as f32) / (u32::MAX as f32);
        r * 2.0 - 1.0
    };

    let a: Vec<f32> = (0..(m * k)).map(|_| rand() * 0.5).collect();
    let b_f32: Vec<f32> = (0..(n * k)).map(|_| rand() * 0.5).collect();

    // Quantize B row-by-row (each row is K elements; K is a multiple of
    // 32 so per-row quantization aligns with block boundaries).
    let mut b_q: Vec<u8> = Vec::with_capacity((n as usize) * (k as usize / 32) * 34);
    let mut row_q = Vec::with_capacity((k as usize / 32) * 34);
    for ni in 0..n as usize {
        let row = &b_f32[ni * k as usize..(ni + 1) * k as usize];
        quantize_row_q8_0(row, &mut row_q);
        b_q.extend_from_slice(&row_q);
    }

    let expected = cpu_matmul_q8_0_ref(&a, m, k, &b_q, n);

    let backend = WgpuBackend::new().await.expect("wgpu adapter");
    let cfg = WgslConfig {
        bf16_quant_writes: false,
        act_dtype: ActDtype::F32,
        weight_dtype: WeightDtype::Quant(QuantKind::Q8_0),
    };
    let op = MatMulF32::new(mm_cfg);
    let wgsl = op.wgsl(&cfg);
    let pipeline = backend
        .create_pipeline(&wgsl, "main", MatMulF32::layout())
        .await
        .expect("pipeline");

    let alloc_with = |bytes: &[u8]| -> BufRef {
        let id = backend.allocate(bytes.len() as u64).expect("allocate");
        backend.write_buffer(id, 0, bytes).expect("write");
        BufRef::new(id, bytes.len() as u64)
    };
    let a_bytes: Vec<u8> = a.iter().flat_map(|x| x.to_le_bytes()).collect();
    let a_buf = alloc_with(&a_bytes);
    let b_buf = alloc_with(&b_q);
    let dims_buf = alloc_with(&pack_dims_u32x4(m, n, k));
    let out_len = (m as u64) * (n as u64) * 4;
    let out_id = backend.allocate(out_len).expect("alloc out");
    let out_buf = BufRef::new(out_id, out_len);

    let mut enc = backend.create_command_encoder();
    dispatch_matmul::<MatMulF32, _>(
        &backend,
        &mut enc,
        &pipeline,
        &op,
        &MatmulBufs {
            a: &a_buf,
            b: &b_buf,
            dims: &dims_buf,
            out: &out_buf,
        },
        m,
        n,
    )
    .expect("dispatch");
    backend.submit(enc).await.expect("submit");

    let got_bytes = backend.read_buffer(out_id, 0, out_len).await.expect("read");
    let got: Vec<f32> = got_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    backend.free(a_buf.id);
    backend.free(b_buf.id);
    backend.free(dims_buf.id);
    backend.free(out_buf.id);

    (got, expected)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    a.iter()
        .zip(b)
        .enumerate()
        .map(|(i, (x, y))| ((x - y).abs(), i))
        .fold(
            (0f32, 0usize),
            |acc, (e, i)| {
                if e > acc.0 { (e, i) } else { acc }
            },
        )
}

/// Tight tolerance vs. the same scalar Rust formula: the only divergence
/// sources are accumulator ordering (workgroup-level reductions vs.
/// serial CPU sum) and fp32 non-associativity. Empirical tolerance for
/// K up to a few hundred is ~1e-5 * K.
fn assert_close(got: &[f32], expected: &[f32], k: u32) {
    assert_eq!(got.len(), expected.len());
    let (max_err, idx) = max_abs_diff(got, expected);
    let tol = 1e-5 * (k as f32);
    assert!(
        max_err <= tol,
        "max abs diff {max_err:.3e} > tol {tol:.3e} at idx {idx} (got={} exp={})",
        got[idx],
        expected[idx]
    );
}

#[test]
fn q8_0_matmul_small() {
    let (got, exp) = pollster::block_on(run_one(16, 16, 32, 0xC0FFEE));
    assert_close(&got, &exp, 32);
}

#[test]
fn q8_0_matmul_non_tile_aligned() {
    // M=10, N=10 force partial output tiles (bm=bn=16 → bounds checks fire).
    let (got, exp) = pollster::block_on(run_one(10, 10, 32, 0xBEEF));
    assert_close(&got, &exp, 32);
}

#[test]
fn q8_0_matmul_multi_block_k() {
    // K=128 = 4 Q8_0 blocks per row; exercises the t-loop in the kernel.
    let (got, exp) = pollster::block_on(run_one(8, 8, 128, 0xDEAD));
    assert_close(&got, &exp, 128);
}

#[test]
fn q8_0_matmul_rectangular() {
    // M != N, K large enough to span several K-tiles.
    let (got, exp) = pollster::block_on(run_one(24, 48, 96, 0x1234_5678));
    assert_close(&got, &exp, 96);
}

/// Production-N/K geometry for the fused QKV matmul in Z-Image DiT
/// (hidden=3840, 3*hidden=11520) at the production register-blocked tile
/// shape (`bm=bn=64, tm=tn=4, bk=32`) with F32 acts. M is held at 64 (one
/// bm tile) since prod seqlen (~1024 tokens) scales only the CPU reference
/// without exercising new dispatch paths. If unit-test scale passes but
/// this NaNs, the bug is in the K-tile loop, the block-cooperative loader,
/// or large-N dispatch path.
#[test]
fn q8_0_matmul_prod_geom_qkv_f32_acts() {
    let cfg = MatMulConfig {
        bm: 64,
        bn: 64,
        bk: 32,
        tm: 4,
        tn: 4,
        b_nmajor: false,
    };
    let (got, exp) = pollster::block_on(run_one_with_cfg(64, 11520, 3840, 0xC0DE_BABE, cfg));
    let nan_count = got.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(
        nan_count,
        0,
        "GPU output has {nan_count}/{} NaN/inf cells",
        got.len()
    );
    assert_close(&got, &exp, 3840);
}

/// Production-N/K geometry for the FFN-up matmul (`[hidden, 4*hidden]` =
/// `[3840, 15360]`). Larger N stresses output workgroup tiling more
/// than QKV. M=64 (see qkv variant for why).
#[test]
fn q8_0_matmul_prod_geom_ffn_up_f32_acts() {
    let cfg = MatMulConfig {
        bm: 64,
        bn: 64,
        bk: 32,
        tm: 4,
        tn: 4,
        b_nmajor: false,
    };
    let (got, exp) = pollster::block_on(run_one_with_cfg(64, 15360, 3840, 0xFFEE_BEEF, cfg));
    let nan_count = got.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(
        nan_count,
        0,
        "GPU output has {nan_count}/{} NaN/inf cells",
        got.len()
    );
    assert_close(&got, &exp, 3840);
}

/// Production-N/K geometry for the FFN-down matmul (`[4*hidden, hidden]` =
/// `[15360, 3840]`). Larger K stresses the per-K-step block load. M=64.
#[test]
fn q8_0_matmul_prod_geom_ffn_down_f32_acts() {
    let cfg = MatMulConfig {
        bm: 64,
        bn: 64,
        bk: 32,
        tm: 4,
        tn: 4,
        b_nmajor: false,
    };
    let (got, exp) = pollster::block_on(run_one_with_cfg(64, 3840, 15360, 0xFAB0_BEEF, cfg));
    let nan_count = got.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(
        nan_count,
        0,
        "GPU output has {nan_count}/{} NaN/inf cells",
        got.len()
    );
    assert_close(&got, &exp, 15360);
}

/// Q8_0 weights × `ActDtype::Bf16` activations × packed-bf16 output.
/// This is the production combo for Z-Image DiT main matmuls and was
/// previously untested — `run_one` above only exercises `ActDtype::F32`.
/// Activations are produced from f32 samples via round-to-bf16, the
/// reference computes in fp32 from the rounded values, and the GPU
/// readback unpacks bf16x2-per-u32. Tolerance is widened from `1e-5*K`
/// to `5e-3*max_abs_ref` to accommodate bf16-packed output (one ULP of
/// bf16 is ~3.9e-3 of magnitude).
#[test]
fn q8_0_matmul_bf16_acts() {
    let (got, exp) = pollster::block_on(run_one_bf16_acts(16, 16, 64, 0x00AB_CDEF));
    let max_abs_ref = exp.iter().copied().map(f32::abs).fold(0f32, f32::max);
    let tol = (5e-3 * max_abs_ref).max(1e-4);
    let (max_err, idx) = max_abs_diff(&got, &exp);
    assert!(
        max_err <= tol,
        "max abs diff {max_err:.3e} > tol {tol:.3e} at idx {idx} \
         (got={} exp={}) ref_max={}",
        got[idx],
        exp[idx],
        max_abs_ref
    );
}

/// Round f32 to bf16 with RNE then re-expand to f32 (the value the GPU
/// load_a path will see). Mirrors the kernel's `pack_bf16x2` round.
fn round_f32_to_bf16_f32(x: f32) -> f32 {
    let b = x.to_bits();
    let bf16_bits: u16 = if (b & 0x7F80_0000) == 0x7F80_0000 {
        (b >> 16) as u16 // inf/NaN passthrough of upper 16 bits
    } else {
        let l = (b >> 16) & 1;
        (((b + 0x7FFF + l) >> 16) & 0xFFFF) as u16
    };
    f32::from_bits((bf16_bits as u32) << 16)
}

fn pack_bf16_pair(lo: f32, hi: f32) -> u32 {
    let lo_bits = (round_f32_to_bf16_f32(lo).to_bits() >> 16) as u16 as u32;
    let hi_bits = (round_f32_to_bf16_f32(hi).to_bits() >> 16) as u16 as u32;
    lo_bits | (hi_bits << 16)
}

async fn run_one_bf16_acts(m: u32, n: u32, k: u32, seed: u64) -> (Vec<f32>, Vec<f32>) {
    assert!(k.is_multiple_of(32));
    assert!(n.is_multiple_of(2), "packed bf16 output requires even N");
    let mut s = seed;
    let mut rand = || -> f32 {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let r = ((s >> 33) as u32 as f32) / (u32::MAX as f32);
        r * 2.0 - 1.0
    };
    // Generate f32 inputs, then quantize A to bf16 (the GPU only ever
    // sees bf16 values for activations). Reference computes from the
    // rounded f32 form to match.
    let a_f32: Vec<f32> = (0..(m * k)).map(|_| rand() * 0.5).collect();
    let a_bf16_f32: Vec<f32> = a_f32.iter().copied().map(round_f32_to_bf16_f32).collect();
    let b_f32: Vec<f32> = (0..(n * k)).map(|_| rand() * 0.5).collect();
    let mut b_q: Vec<u8> = Vec::with_capacity((n as usize) * (k as usize / 32) * 34);
    let mut row_q = Vec::with_capacity((k as usize / 32) * 34);
    for ni in 0..n as usize {
        let row = &b_f32[ni * k as usize..(ni + 1) * k as usize];
        quantize_row_q8_0(row, &mut row_q);
        b_q.extend_from_slice(&row_q);
    }
    let expected_f32 = cpu_matmul_q8_0_ref(&a_bf16_f32, m, k, &b_q, n);
    // Reference: same matmul, then round each cell to bf16 (the kernel
    // writes bf16-packed output, so the readback observes rounded values).
    let expected: Vec<f32> = expected_f32
        .iter()
        .copied()
        .map(round_f32_to_bf16_f32)
        .collect();

    let backend = WgpuBackend::new().await.expect("wgpu adapter");
    let cfg = WgslConfig {
        bf16_quant_writes: false,
        act_dtype: ActDtype::Bf16,
        weight_dtype: WeightDtype::Quant(QuantKind::Q8_0),
    };
    let op = MatMulF32::new(MatMulConfig {
        bm: 16,
        bn: 16,
        bk: 32,
        tm: 1,
        tn: 2,
        b_nmajor: false,
    });
    let wgsl = op.wgsl(&cfg);
    let pipeline = backend
        .create_pipeline(&wgsl, "main", MatMulF32::layout())
        .await
        .expect("pipeline");

    // Pack A as bf16x2 per u32: row-major, columns paired.
    assert!(k.is_multiple_of(2));
    let mut a_packed: Vec<u32> = Vec::with_capacity((m * k / 2) as usize);
    for mi in 0..m as usize {
        for ki in (0..k as usize).step_by(2) {
            let lo = a_f32[mi * k as usize + ki];
            let hi = a_f32[mi * k as usize + ki + 1];
            a_packed.push(pack_bf16_pair(lo, hi));
        }
    }
    let a_bytes: Vec<u8> = a_packed.iter().flat_map(|w| w.to_le_bytes()).collect();

    let alloc_with = |bytes: &[u8]| -> BufRef {
        let id = backend.allocate(bytes.len() as u64).expect("allocate");
        backend.write_buffer(id, 0, bytes).expect("write");
        BufRef::new(id, bytes.len() as u64)
    };
    let a_buf = alloc_with(&a_bytes);
    let b_buf = alloc_with(&b_q);
    let dims_buf = alloc_with(&pack_dims_u32x4(m, n, k));
    // Packed bf16 output: 2 bf16 per u32, so byte count = m*n*2.
    let out_len = (m as u64) * (n as u64) * 2;
    let out_id = backend.allocate(out_len).expect("alloc out");
    let out_buf = BufRef::new(out_id, out_len);

    let mut enc = backend.create_command_encoder();
    dispatch_matmul::<MatMulF32, _>(
        &backend,
        &mut enc,
        &pipeline,
        &op,
        &MatmulBufs {
            a: &a_buf,
            b: &b_buf,
            dims: &dims_buf,
            out: &out_buf,
        },
        m,
        n,
    )
    .expect("dispatch");
    backend.submit(enc).await.expect("submit");

    let got_bytes = backend.read_buffer(out_id, 0, out_len).await.expect("read");
    let got: Vec<f32> = got_bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]) as u32;
            f32::from_bits(bits << 16)
        })
        .collect();

    backend.free(a_buf.id);
    backend.free(b_buf.id);
    backend.free(dims_buf.id);
    backend.free(out_buf.id);

    (got, expected)
}
