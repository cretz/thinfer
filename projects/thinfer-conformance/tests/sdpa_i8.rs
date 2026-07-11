//! `thinfer_core::ops::sdpa_i8` flash-attention vs scalar Rust reference.
//! Compares dequantized GPU output against a naive `softmax(Q K^T * scale +
//! mask) V` reference on dequantized Q, K, V, then re-quantized per-(D/32).

#![cfg(feature = "conformance")]

mod i8_common;

use i8_common::*;
use thinfer_core::backend::{Backend, BufRef};
use thinfer_core::ops::sdpa_i8::{SdpaI8Bufs, build_wgsl, dispatch_sdpa_i8, hint, layout};

/// Dense (pre-output-quant) reference O `[B*S_q*H_q, D]`. Paired-out tests
/// requant via `requant_rows`; dense-out tests compare directly.
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
) -> Vec<f32> {
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
    o_dense
}

/// 48-byte uniform matching the kernel's `struct U` (row0/out_mode chunking
/// and output-form fields after the 8 base fields).
#[allow(clippy::too_many_arguments)]
fn uniform_bytes(
    b: u32,
    h_q: u32,
    h_kv: u32,
    s_q: u32,
    s_k: u32,
    d: u32,
    sm_scale: f32,
    has_mask: bool,
    row0: u32,
    out_mode: u32,
) -> [u8; 48] {
    let mut u = [0u8; 48];
    u[0..4].copy_from_slice(&b.to_le_bytes());
    u[4..8].copy_from_slice(&h_q.to_le_bytes());
    u[8..12].copy_from_slice(&h_kv.to_le_bytes());
    u[12..16].copy_from_slice(&s_q.to_le_bytes());
    u[16..20].copy_from_slice(&s_k.to_le_bytes());
    u[20..24].copy_from_slice(&d.to_le_bytes());
    u[24..28].copy_from_slice(&sm_scale.to_le_bytes());
    u[28..32].copy_from_slice(&(has_mask as u32).to_le_bytes());
    u[32..36].copy_from_slice(&row0.to_le_bytes());
    u[36..40].copy_from_slice(&out_mode.to_le_bytes());
    u
}

/// `chunk_rows = 0` dispatches whole; otherwise the Q range is split into
/// `chunk_rows`-row dispatches (one uniform each, `row0` carrying the global
/// offset) inside a single submit -- the bit-exactness contract the engine's
/// TDR chunking relies on. `out_mode` 0 returns the paired output; 1/2 return
/// the dense f16/bf16 output decoded to f32 (paired result vecs empty).
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
    chunk_rows: u32,
    out_mode: u32,
) -> Option<(Vec<i8>, Vec<f32>, Vec<f32>, Vec<f32>)> {
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
    let exp_dense = cpu_ref(
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

    let out_data_len = (q_rows as u64) * (d as u64);
    let out_scale_len = (q_rows as u64) * (d as u64 / 32) * 4;
    let out_total = if out_mode == 0 {
        out_data_len + out_scale_len
    } else {
        // Dense packed f16/bf16: 2 bytes per element.
        out_data_len * 2
    };
    let out_id = backend.allocate(out_total).expect("allocate out");
    // Zero-init via a one-time write so the data segment doesn't carry GC bits.
    let zero = vec![0u8; out_total as usize];
    backend.write_buffer(out_id, 0, &zero).expect("zero out");
    let out_fused = BufRef::new(out_id, out_total);

    // Per-chunk uniforms (row0 walks the global Q range); one submit total.
    let step = if chunk_rows == 0 { s_q } else { chunk_rows };
    let mut u_bufs = Vec::new();
    let mut r0 = 0u32;
    while r0 < s_q {
        let rows = (s_q - r0).min(step);
        let u = uniform_bytes(b, h_q, h_kv, s_q, s_k, d, sm_scale, has_mask, r0, out_mode);
        u_bufs.push((alloc_with(&backend, &u), rows));
        r0 += rows;
    }

    let pipeline = backend
        .create_pipeline("sdpa_i8", &build_wgsl(), "main", layout())
        .await
        .expect("pipeline");
    let mut enc = backend.create_command_encoder();
    for (u_buf, rows) in &u_bufs {
        dispatch_sdpa_i8(
            &backend,
            &mut enc,
            &pipeline,
            &SdpaI8Bufs {
                q: &q_fused,
                k: &k_fused,
                v: &v_fused,
                mask: &mask_buf,
                out: &out_fused,
                uniform: u_buf,
            },
            b,
            *rows,
            h_q,
            d,
        )
        .expect("dispatch");
    }
    backend.submit(enc).await.expect("submit");

    let (g_d, g_s, g_dense) = if out_mode == 0 {
        let data_bytes = backend
            .read_buffer(out_id, 0, out_data_len)
            .await
            .expect("read data");
        let scale_bytes = backend
            .read_buffer(out_id, out_data_len, out_scale_len)
            .await
            .expect("read scale");
        let data_i8: Vec<i8> = data_bytes.iter().map(|by| *by as i8).collect();
        (data_i8, bytes_to_params(&scale_bytes), Vec::new())
    } else {
        let bytes = backend
            .read_buffer(out_id, 0, out_total)
            .await
            .expect("read dense");
        let dense: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                if out_mode == 1 {
                    f16_bits_to_f32(bits)
                } else {
                    bf16_bits_to_f32(bits)
                }
            })
            .collect();
        (Vec::new(), Vec::new(), dense)
    };

    for buf in [q_fused, k_fused, v_fused, mask_buf, out_fused] {
        backend.free(buf.id);
    }
    for (u_buf, _) in u_bufs {
        backend.free(u_buf.id);
    }
    Some((g_d, g_s, g_dense, exp_dense))
}

#[test]
fn sdpa_i8_wgsl_sanity() {
    let src = build_wgsl();
    assert!(src.contains("enable f16"));
    assert!(src.contains("inv_l"));
    assert_eq!(layout().len(), 6);
    assert_eq!(hint(), "ai8-perblock-flashattn-br64-bc64-fused");
}

/// Paired-out comparison against the requantized CPU reference.
fn check_paired(got: (Vec<i8>, Vec<f32>, Vec<f32>, Vec<f32>), rows: usize, d: usize, label: &str) {
    let (g_d, g_s, _dense, exp_dense) = got;
    let (e_d, e_s) = requant_rows(&exp_dense, rows, d);
    assert_paired_close(&g_d, &g_s, &e_d, &e_s, rows, d, 5e-2, 1e-3, label);
}

/// Dense-out comparison: GPU f16/bf16 output vs the CPU dense reference.
fn check_dense(got: (Vec<i8>, Vec<f32>, Vec<f32>, Vec<f32>), rel_tol: f32, label: &str) {
    let (_d, _s, g_dense, exp_dense) = got;
    assert_eq!(g_dense.len(), exp_dense.len(), "{label}: length");
    let max_abs = exp_dense.iter().fold(0f32, |a, &x| a.max(x.abs()));
    let tol = (rel_tol * max_abs).max(1e-3);
    let (max_err, idx) = g_dense
        .iter()
        .zip(&exp_dense)
        .enumerate()
        .map(|(i, (g, e))| ((g - e).abs(), i))
        .fold((0f32, 0usize), |a, b| if b.0 > a.0 { b } else { a });
    assert!(
        max_err <= tol,
        "{label}: max abs diff {max_err:.3e} > tol {tol:.3e} at idx {idx} \
         (got={} exp={}) exp_max={max_abs}",
        g_dense[idx],
        exp_dense[idx],
    );
}

#[test]
fn sdpa_i8_small_no_mask() {
    // B=1, H_q=H_kv=1, S_q=S_k=64 (exactly one BR=BC=64 tile), D=64.
    let Some(got) = pollster::block_on(try_run(1, 1, 1, 64, 64, 64, false, 0xDB5D_A5DA, 0, 0))
    else {
        return;
    };
    check_paired(got, 64, 64, "sdpa_i8_small");
}

#[test]
fn sdpa_i8_with_mask() {
    let Some(got) = pollster::block_on(try_run(1, 1, 1, 64, 64, 64, true, 0xD_B5DA_BEEF, 0, 0))
    else {
        return;
    };
    check_paired(got, 64, 64, "sdpa_i8_with_mask");
}

#[test]
fn sdpa_i8_gqa() {
    // H_q=2, H_kv=1 -> grouped-query attention; both heads share K/V.
    let Some(got) = pollster::block_on(try_run(1, 2, 1, 64, 64, 64, false, 0xDB6A_BCDE, 0, 0))
    else {
        return;
    };
    // 64 q-rows per head * 2 heads = 128 rows total in the paired comparison.
    check_paired(got, 128, 64, "sdpa_i8_gqa");
}

#[test]
fn sdpa_i8_chunked_matches_whole() {
    // 3 x 64-row chunks over S_q=192 (row0 = 0/64/128) must be bit-identical
    // to the whole-tensor dispatch: this is the TDR-chunking contract.
    let seed = 0xC0FF_EE00;
    let Some(whole) = pollster::block_on(try_run(1, 1, 1, 192, 128, 64, false, seed, 0, 0)) else {
        return;
    };
    let Some(chunked) = pollster::block_on(try_run(1, 1, 1, 192, 128, 64, false, seed, 64, 0))
    else {
        return;
    };
    assert_eq!(whole.0, chunked.0, "chunked i8 data differs from whole");
    assert_eq!(whole.1, chunked.1, "chunked scales differ from whole");
    check_paired(whole, 192, 64, "sdpa_i8_chunked_whole_ref");
}

#[test]
fn sdpa_i8_dense_f16_out() {
    // out_mode=1: dense packed-f16 output, no output quantize. Chunked to
    // also cover row0 with a dense destination.
    let Some(got) = pollster::block_on(try_run(1, 2, 2, 128, 64, 64, false, 0xDEAD_00F1, 64, 1))
    else {
        return;
    };
    check_dense(got, 2e-3, "sdpa_i8_dense_f16");
}

#[test]
fn sdpa_i8_dense_bf16_out() {
    // out_mode=2: dense packed-bf16 output (the Wan residual dtype).
    let Some(got) = pollster::block_on(try_run(1, 1, 1, 64, 64, 128, false, 0xDEAD_00F2, 0, 2))
    else {
        return;
    };
    // bf16 has ~3 decimal digits; wider rel tol than f16.
    check_dense(got, 1e-2, "sdpa_i8_dense_bf16");
}
