//! `thinfer_core::ops::sdpa_i8` flash-attention vs scalar Rust reference.
//! Compares dequantized GPU output against a naive `softmax(Q K^T * scale +
//! mask) V` reference on dequantized Q, K, V, then re-quantized per-(D/32).

#![cfg(feature = "conformance")]

mod i8_common;

use i8_common::*;
use thinfer_core::backend::{Backend, BufRef};
use thinfer_core::ops::sdpa_i8::{SdpaI8Bufs, build_wgsl, dispatch_sdpa_i8, hint, layout};

#[allow(clippy::too_many_arguments)]
fn cpu_ref(
    q_i8: &[i8],
    q_scale: &[f32],
    k_i8: &[i8],
    k_scale: &[f32],
    v_i8: &[i8],
    v_scale: &[f32],
    mask: Option<&[f32]>,
    b: usize,
    h_q: usize,
    h_kv: usize,
    s_q: usize,
    s_k: usize,
    d: usize,
    sm_scale: f32,
) -> (Vec<i8>, Vec<f32>) {
    // Dequantize Q, K, V row-by-row. Row index = (b * S_* + s_*) * H + h.
    let q_rows = b * s_q * h_q;
    let kv_rows = b * s_k * h_kv;
    let q_dense = dequant_paired(q_i8, q_scale, q_rows, d);
    let k_dense = dequant_paired(k_i8, k_scale, kv_rows, d);
    let v_dense = dequant_paired(v_i8, v_scale, kv_rows, d);

    let mut o_dense = vec![0f32; q_rows * d];
    for bi in 0..b {
        for sq in 0..s_q {
            for hq in 0..h_q {
                let hkv = (hq * h_kv) / h_q;
                let q_row = ((bi * s_q + sq) * h_q + hq) * d;
                let mut scores = vec![0f32; s_k];
                for sk in 0..s_k {
                    let k_row = ((bi * s_k + sk) * h_kv + hkv) * d;
                    let dot: f32 = (0..d)
                        .map(|j| q_dense[q_row + j] * k_dense[k_row + j])
                        .sum();
                    let m = mask.map(|mm| mm[(bi * s_q + sq) * s_k + sk]).unwrap_or(0.0);
                    scores[sk] = dot * sm_scale + m;
                }
                let smax = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - smax).exp()).collect();
                let sum_e: f32 = exps.iter().sum();
                for (sk, &e) in exps.iter().enumerate() {
                    let p = e / sum_e;
                    let v_row = ((bi * s_k + sk) * h_kv + hkv) * d;
                    for j in 0..d {
                        o_dense[q_row + j] += p * v_dense[v_row + j];
                    }
                }
            }
        }
    }
    requant_rows(&o_dense, q_rows, d)
}

#[allow(clippy::too_many_arguments)]
async fn try_run(
    b: u32,
    h_q: u32,
    h_kv: u32,
    s_q: u32,
    s_k: u32,
    d: u32,
    has_mask: bool,
    seed: u64,
) -> Option<(Vec<i8>, Vec<f32>, Vec<i8>, Vec<f32>)> {
    let backend = make_backend_with_f16().await?;

    let q_rows = (b * s_q * h_q) as usize;
    let kv_rows = (b * s_k * h_kv) as usize;
    let (q_i8, q_scale) = gen_paired_i8(q_rows, d as usize, seed);
    let (k_i8, k_scale) = gen_paired_i8(kv_rows, d as usize, seed ^ 0x1010);
    let (v_i8, v_scale) = gen_paired_i8(kv_rows, d as usize, seed ^ 0x2020);

    let mut rng = Rng::new(seed ^ 0x3030);
    let mask_f32: Vec<f32> = if has_mask {
        (0..b * s_q * s_k).map(|_| rng.f32_signed() * 0.5).collect()
    } else {
        vec![0f32; (b * s_q * s_k) as usize]
    };
    // mask buffer is array<vec2<f16>>: pack pairs along the S_k axis (kernel
    // indexes as `mask[mask_w_base + (key_global >> 1u)]`). Requires S_k % 2 = 0.
    assert!(s_k.is_multiple_of(2));
    let mask_bytes = pack_f16_vec(&mask_f32);

    let sm_scale = 1.0 / (d as f32).sqrt();
    let mask_ref = if has_mask {
        Some(mask_f32.as_slice())
    } else {
        None
    };
    let (exp_d, exp_s) = cpu_ref(
        &q_i8,
        &q_scale,
        &k_i8,
        &k_scale,
        &v_i8,
        &v_scale,
        mask_ref,
        b as usize,
        h_q as usize,
        h_kv as usize,
        s_q as usize,
        s_k as usize,
        d as usize,
        sm_scale,
    );

    // Fused per-role buffers: [data || scale] contiguous in one allocation.
    // Sdpa kernel binds the full span; producers/readers index sub-views.
    fn alloc_fused<B: Backend>(backend: &B, data: &[u8], scale: &[u8]) -> (BufRef, BufRef, BufRef) {
        let total = (data.len() + scale.len()) as u64;
        let id = backend.allocate(total).expect("allocate");
        backend.write_buffer(id, 0, data).expect("write data");
        backend
            .write_buffer(id, data.len() as u64, scale)
            .expect("write scale");
        let fused = BufRef::new(id, total);
        let data_view = BufRef::view(id, 0, data.len() as u64);
        let scale_view = BufRef::view(id, data.len() as u64, scale.len() as u64);
        (fused, data_view, scale_view)
    }
    let q_data_bytes = pack_i8_rows(&q_i8, q_rows, d as usize);
    let q_scale_bytes = params_to_bytes(&q_scale);
    let k_data_bytes = pack_i8_rows(&k_i8, kv_rows, d as usize);
    let k_scale_bytes = params_to_bytes(&k_scale);
    let v_data_bytes = pack_i8_rows(&v_i8, kv_rows, d as usize);
    let v_scale_bytes = params_to_bytes(&v_scale);
    let (q_fused, _q_dv, _q_sv) = alloc_fused(&backend, &q_data_bytes, &q_scale_bytes);
    let (k_fused, _k_dv, _k_sv) = alloc_fused(&backend, &k_data_bytes, &k_scale_bytes);
    let (v_fused, _v_dv, _v_sv) = alloc_fused(&backend, &v_data_bytes, &v_scale_bytes);
    let mask_buf = alloc_with(&backend, &mask_bytes);

    // Uniform: b, h_q, h_kv, s_q, s_k, d, scale, has_mask (32 bytes).
    let mut u_bytes = [0u8; 32];
    u_bytes[0..4].copy_from_slice(&b.to_le_bytes());
    u_bytes[4..8].copy_from_slice(&h_q.to_le_bytes());
    u_bytes[8..12].copy_from_slice(&h_kv.to_le_bytes());
    u_bytes[12..16].copy_from_slice(&s_q.to_le_bytes());
    u_bytes[16..20].copy_from_slice(&s_k.to_le_bytes());
    u_bytes[20..24].copy_from_slice(&d.to_le_bytes());
    u_bytes[24..28].copy_from_slice(&sm_scale.to_le_bytes());
    u_bytes[28..32].copy_from_slice(&(has_mask as u32).to_le_bytes());
    let u_buf = alloc_with(&backend, &u_bytes);

    let out_data_len = (q_rows as u64) * (d as u64);
    let out_scale_len = (q_rows as u64) * (d as u64 / 32) * 4;
    let out_total = out_data_len + out_scale_len;
    let out_id = backend.allocate(out_total).expect("allocate out");
    // Zero-init via a one-time write so the data segment doesn't carry GC bits.
    let zero = vec![0u8; out_total as usize];
    backend.write_buffer(out_id, 0, &zero).expect("zero out");
    let out_fused = BufRef::new(out_id, out_total);
    let out_data_view = BufRef::view(out_id, 0, out_data_len);
    let out_scale_view = BufRef::view(out_id, out_data_len, out_scale_len);

    let (g_d, g_s) = run_and_read_paired(
        &backend,
        &build_wgsl(),
        layout(),
        &out_data_view,
        &out_scale_view,
        |bk, enc, pipe| {
            dispatch_sdpa_i8(
                bk,
                enc,
                pipe,
                &SdpaI8Bufs {
                    q: &q_fused,
                    k: &k_fused,
                    v: &v_fused,
                    mask: &mask_buf,
                    out: &out_fused,
                    uniform: &u_buf,
                },
                b,
                s_q,
                h_q,
                d,
            )
            .expect("dispatch");
        },
    )
    .await;

    for buf in [q_fused, k_fused, v_fused, mask_buf, u_buf, out_fused] {
        backend.free(buf.id);
    }
    Some((g_d, g_s, exp_d, exp_s))
}

#[test]
fn sdpa_i8_wgsl_sanity() {
    let src = build_wgsl();
    assert!(src.contains("enable f16"));
    assert!(src.contains("inv_l"));
    assert_eq!(layout().len(), 6);
    assert_eq!(hint(), "ai8-perblock-flashattn-br64-bc64-fused");
}

#[test]
fn sdpa_i8_small_no_mask() {
    // B=1, H_q=H_kv=1, S_q=S_k=64 (exactly one BR=BC=64 tile), D=64.
    let Some((g_d, g_s, e_d, e_s)) =
        pollster::block_on(try_run(1, 1, 1, 64, 64, 64, false, 0xDB5D_A5DA))
    else {
        return;
    };
    assert_paired_close(&g_d, &g_s, &e_d, &e_s, 64, 64, 5e-2, 1e-3, "sdpa_i8_small");
}

#[test]
fn sdpa_i8_with_mask() {
    let Some((g_d, g_s, e_d, e_s)) =
        pollster::block_on(try_run(1, 1, 1, 64, 64, 64, true, 0xD_B5DA_BEEF))
    else {
        return;
    };
    assert_paired_close(
        &g_d,
        &g_s,
        &e_d,
        &e_s,
        64,
        64,
        5e-2,
        1e-3,
        "sdpa_i8_with_mask",
    );
}

#[test]
fn sdpa_i8_gqa() {
    // H_q=2, H_kv=1 -> grouped-query attention; both heads share K/V.
    let Some((g_d, g_s, e_d, e_s)) =
        pollster::block_on(try_run(1, 2, 1, 64, 64, 64, false, 0xDB6A_BCDE))
    else {
        return;
    };
    // 64 q-rows per head * 2 heads = 128 rows total in the paired comparison.
    assert_paired_close(&g_d, &g_s, &e_d, &e_s, 128, 64, 5e-2, 1e-3, "sdpa_i8_gqa");
}
