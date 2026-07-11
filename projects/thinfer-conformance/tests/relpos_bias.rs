//! `thinfer_core::ops::relpos_bias::RelposBiasF32` (compact T5/umT5 relative-
//! position bias -> dense `[H,S,S]` per-head mask) vs a scalar Rust reference,
//! plus an integration check that the dense output drives `SdpaF32`'s per-head
//! mask mode (`has_mask == 2`) matching a CPU sdpa-with-bias reference.

#![cfg(feature = "conformance")]

mod i8_common;

use i8_common::*;
use thinfer_core::backend::{Backend, WgpuBackend};
use thinfer_core::ops::{
    RelposBiasF32, RelposBiasOp, SdpaF32, SdpaOp, WgslConfig, relpos_bucket_map,
};

// out[h, i, j] = table[bucket_map[i*S + j] * H + h]
fn cpu_expand(table: &[f32], bucket_map: &[u32], h: usize, s: usize) -> Vec<f32> {
    let mut out = vec![0f32; h * s * s];
    for hh in 0..h {
        for i in 0..s {
            for j in 0..s {
                let bucket = bucket_map[i * s + j] as usize;
                out[(hh * s + i) * s + j] = table[bucket * h + hh];
            }
        }
    }
    out
}

#[test]
fn relpos_bias_f32_matches_reference() {
    pollster::block_on(async {
        let backend = WgpuBackend::new().await.expect("wgpu adapter");
        let (num_buckets, h, s) = (32u32, 4usize, 8usize); // S even (packed-mask compatible)
        let bucket_map = relpos_bucket_map(s, true, num_buckets, 128);

        let mut rng = Rng::new(0xA17_u64);
        let table: Vec<f32> = (0..num_buckets as usize * h)
            .map(|_| rng.f32_signed())
            .collect();
        let exp = cpu_expand(&table, &bucket_map, h, s);

        let table_bytes: Vec<u8> = table.iter().flat_map(|x| x.to_le_bytes()).collect();
        let bm_bytes: Vec<u8> = bucket_map.iter().flat_map(|x| x.to_le_bytes()).collect();
        let table_buf = alloc_with(&backend, &table_bytes);
        let bm_buf = alloc_with(&backend, &bm_bytes);
        let n = (h * s * s) as u32;
        let out_buf = alloc_zero(&backend, (n as u64) * 4);

        let mut u = [0u8; 16];
        u[0..4].copy_from_slice(&(h as u32).to_le_bytes());
        u[4..8].copy_from_slice(&(s as u32).to_le_bytes());
        let u_buf = alloc_with(&backend, &u);

        let pipeline = backend
            .create_pipeline(
                "relpos_bias_f32",
                <RelposBiasF32 as RelposBiasOp>::wgsl(&WgslConfig::FP32),
                "main",
                <RelposBiasF32 as RelposBiasOp>::layout(),
            )
            .await
            .expect("pipeline");
        let mut enc = backend.create_command_encoder();
        let bindings = [
            table_buf.binding(0),
            bm_buf.binding(1),
            out_buf.binding(2),
            u_buf.binding(3),
        ];
        backend
            .dispatch(
                &mut enc,
                &pipeline,
                &bindings,
                <RelposBiasF32 as RelposBiasOp>::workgroups(n),
            )
            .expect("dispatch");
        backend.submit(enc).await.expect("submit");

        let out_bytes = backend
            .read_buffer(out_buf.id, out_buf.offset, out_buf.len)
            .await
            .expect("read");
        let got: Vec<f32> = out_bytes
            .chunks_exact(4)
            .map(|p| f32::from_le_bytes([p[0], p[1], p[2], p[3]]))
            .collect();

        assert_eq!(got.len(), exp.len());
        let max_err = got
            .iter()
            .zip(&exp)
            .fold(0f32, |a, (g, e)| a.max((g - e).abs()));
        assert!(
            max_err == 0.0,
            "relpos_bias f32 exact mismatch: {max_err:e}"
        );
    });
}

#[allow(clippy::too_many_arguments)]
fn cpu_sdpa_perhead(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: &[f32], // [H, S_q, S_k]
    s_q: usize,
    s_k: usize,
    h: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    // BSHD layout, B=1, H_q == H_kv == h.
    let mut out = vec![0f32; s_q * h * d];
    for hh in 0..h {
        for i in 0..s_q {
            let mut scores = vec![0f32; s_k];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..s_k {
                let mut dot = 0f32;
                for dd in 0..d {
                    dot += q[(i * h + hh) * d + dd] * k[(j * h + hh) * d + dd];
                }
                let sj = dot * scale + mask[(hh * s_q + i) * s_k + j];
                scores[j] = sj;
                mx = mx.max(sj);
            }
            let mut den = 0f32;
            for sj in scores.iter_mut() {
                *sj = (*sj - mx).exp();
                den += *sj;
            }
            for dd in 0..d {
                let mut acc = 0f32;
                for j in 0..s_k {
                    acc += scores[j] * v[(j * h + hh) * d + dd];
                }
                out[(i * h + hh) * d + dd] = acc / den;
            }
        }
    }
    out
}

#[test]
fn sdpa_per_head_mask_matches_reference() {
    pollster::block_on(async {
        let backend = WgpuBackend::new().await.expect("wgpu adapter");
        let (h, s, d) = (2usize, 6usize, 8usize);
        let mut rng = Rng::new(0x5D00_u64);
        let qkv_len = s * h * d;
        let q: Vec<f32> = (0..qkv_len).map(|_| rng.f32_signed()).collect();
        let k: Vec<f32> = (0..qkv_len).map(|_| rng.f32_signed()).collect();
        let v: Vec<f32> = (0..qkv_len).map(|_| rng.f32_signed()).collect();
        // Per-head additive mask [H, S, S].
        let mask: Vec<f32> = (0..h * s * s).map(|_| rng.f32_signed() * 0.5).collect();
        let scale = 1.0 / (d as f32).sqrt();
        let exp = cpu_sdpa_perhead(&q, &k, &v, &mask, s, s, h, d, scale);

        let f32b = |xs: &[f32]| -> Vec<u8> { xs.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let q_buf = alloc_with(&backend, &f32b(&q));
        let k_buf = alloc_with(&backend, &f32b(&k));
        let v_buf = alloc_with(&backend, &f32b(&v));
        let mask_buf = alloc_with(&backend, &f32b(&mask));
        let out_buf = alloc_zero(&backend, (qkv_len as u64) * 4);

        // sdpa uniform: b, h_q, h_kv, s_q, s_k, d, scale, has_mask=2.
        let mut u = [0u8; 32];
        u[0..4].copy_from_slice(&1u32.to_le_bytes());
        u[4..8].copy_from_slice(&(h as u32).to_le_bytes());
        u[8..12].copy_from_slice(&(h as u32).to_le_bytes());
        u[12..16].copy_from_slice(&(s as u32).to_le_bytes());
        u[16..20].copy_from_slice(&(s as u32).to_le_bytes());
        u[20..24].copy_from_slice(&(d as u32).to_le_bytes());
        u[24..28].copy_from_slice(&scale.to_le_bytes());
        u[28..32].copy_from_slice(&2u32.to_le_bytes());
        let u_buf = alloc_with(&backend, &u);

        let pipeline = backend
            .create_pipeline(
                "sdpa_f32_perhead",
                <SdpaF32 as SdpaOp>::wgsl(&WgslConfig::FP32),
                "main",
                <SdpaF32 as SdpaOp>::layout(),
            )
            .await
            .expect("pipeline");
        let mut enc = backend.create_command_encoder();
        let bindings = [
            q_buf.binding(0),
            k_buf.binding(1),
            v_buf.binding(2),
            mask_buf.binding(3),
            out_buf.binding(4),
            u_buf.binding(5),
        ];
        backend
            .dispatch(
                &mut enc,
                &pipeline,
                &bindings,
                <SdpaF32 as SdpaOp>::workgroups(1, s as u32, h as u32),
            )
            .expect("dispatch");
        backend.submit(enc).await.expect("submit");

        let out_bytes = backend
            .read_buffer(out_buf.id, out_buf.offset, out_buf.len)
            .await
            .expect("read");
        let got: Vec<f32> = out_bytes
            .chunks_exact(4)
            .map(|p| f32::from_le_bytes([p[0], p[1], p[2], p[3]]))
            .collect();

        assert_eq!(got.len(), exp.len());
        let max_abs = exp.iter().fold(0f32, |a, &x| a.max(x.abs()));
        let max_err = got
            .iter()
            .zip(&exp)
            .fold(0f32, |a, (g, e)| a.max((g - e).abs()));
        let tol = (1e-5 * max_abs).max(1e-5);
        assert!(
            max_err <= tol,
            "sdpa per-head mask: max abs diff {max_err:e} > tol {tol:e}"
        );
    });
}
