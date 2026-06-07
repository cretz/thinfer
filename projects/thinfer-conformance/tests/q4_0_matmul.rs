//! GPU matmul Q4_0 vs CPU dequant-then-matmul. Mirror of `q8_0_matmul.rs`
//! against the Q4_0 scheme (18-byte blocks, 4-bit nibbles, `(nib - 8) * d`
//! dequant). Same self-contained pattern: scalar Rust port of
//! `dequantize_row_q4_0` is the ground truth.

#![cfg(feature = "conformance")]

use thinfer_core::backend::{Backend, BufRef, WgpuBackend};
use thinfer_core::ops::{
    ActDtype, MatMulConfig, MatMulF32, MatmulBufs, MatmulOp, WeightDtype, WgslConfig,
    dispatch_matmul,
};
use thinfer_core::quant::{QuantKind, quantize_row_q4_0};

fn pack_dims_u32x4(m: u32, n: u32, k: u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&m.to_le_bytes());
    out[4..8].copy_from_slice(&n.to_le_bytes());
    out[8..12].copy_from_slice(&k.to_le_bytes());
    out
}

fn cpu_matmul_q4_0_ref(a: &[f32], m: u32, k: u32, b_q: &[u8], n: u32) -> Vec<f32> {
    assert_eq!(a.len(), (m * k) as usize);
    assert_eq!(
        b_q.len(),
        (n as usize) * (k as usize / 32) * 18,
        "B must be N rows of K/32 Q4_0 blocks"
    );
    let mut b_deq = vec![0f32; (n as usize) * (k as usize)];
    thinfer_core::quant::dequantize_row_q4_0(b_q, &mut b_deq);
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

async fn run_one_with_cfg(
    m: u32,
    n: u32,
    k: u32,
    seed: u64,
    mm_cfg: MatMulConfig,
) -> (Vec<f32>, Vec<f32>) {
    assert!(
        k.is_multiple_of(32),
        "K must be multiple of Q4_0 block size"
    );

    let mut s = seed;
    let mut rand = || -> f32 {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let r = ((s >> 33) as u32 as f32) / (u32::MAX as f32);
        r * 2.0 - 1.0
    };

    let a: Vec<f32> = (0..(m * k)).map(|_| rand() * 0.5).collect();
    let b_f32: Vec<f32> = (0..(n * k)).map(|_| rand() * 0.5).collect();

    let mut b_q: Vec<u8> = Vec::with_capacity((n as usize) * (k as usize / 32) * 18);
    let mut row_q = Vec::with_capacity((k as usize / 32) * 18);
    for ni in 0..n as usize {
        let row = &b_f32[ni * k as usize..(ni + 1) * k as usize];
        quantize_row_q4_0(row, &mut row_q);
        b_q.extend_from_slice(&row_q);
    }

    let expected = cpu_matmul_q4_0_ref(&a, m, k, &b_q, n);

    let backend = WgpuBackend::new().await.expect("wgpu adapter");
    let cfg = WgslConfig {
        bf16_quant_writes: false,
        act_dtype: ActDtype::F32,
        weight_dtype: WeightDtype::Quant(QuantKind::Q4_0),
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
fn q4_0_matmul_small() {
    let (got, exp) = pollster::block_on(run_one(16, 16, 32, 0xC0FFEE));
    assert_close(&got, &exp, 32);
}

#[test]
fn q4_0_matmul_non_tile_aligned() {
    let (got, exp) = pollster::block_on(run_one(10, 10, 32, 0xBEEF));
    assert_close(&got, &exp, 32);
}

#[test]
fn q4_0_matmul_multi_block_k() {
    let (got, exp) = pollster::block_on(run_one(8, 8, 128, 0xDEAD));
    assert_close(&got, &exp, 128);
}

#[test]
fn q4_0_matmul_rectangular() {
    let (got, exp) = pollster::block_on(run_one(24, 48, 96, 0x1234_5678));
    assert_close(&got, &exp, 96);
}

/// Production geometry for the fused QKV matmul. Mirrors the Q8_0 test
/// so any K-tile-loop / block-cooperative-loader regression hits here too.
#[test]
fn q4_0_matmul_prod_geom_qkv_f32_acts() {
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
