//! `thinfer_core::ops::sdpa::build_f16_sg_wgsl` (subgroup flash sdpa) vs a
//! scalar Rust reference: `softmax(Q K^T * scale + mask) V` on f16-rounded
//! dense inputs. Skips when the adapter lacks SHADER_F16 or SUBGROUP, or
//! reports a subgroup floor below the requested CL.

#![cfg(feature = "conformance")]

mod i8_common;

use i8_common::*;
use thinfer_core::backend::Backend;
use thinfer_core::ops::{
    SdpaF32, SdpaOp, build_f16_sg_wgsl, build_f16_sg_windowed_wgsl, f16_sg_workgroups,
};

fn gen_f16_rows(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| f16_round(rng.f32_signed())).collect()
}

#[allow(clippy::too_many_arguments)]
fn cpu_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    b: usize,
    h_q: usize,
    h_kv: usize,
    s_q: usize,
    s_k: usize,
    d: usize,
    sm_scale: f32,
) -> Vec<f32> {
    let mut o = vec![0f32; b * s_q * h_q * d];
    for bi in 0..b {
        for sq in 0..s_q {
            for hq in 0..h_q {
                let hkv = (hq * h_kv) / h_q;
                let q_row = ((bi * s_q + sq) * h_q + hq) * d;
                let mut scores = vec![0f32; s_k];
                for sk in 0..s_k {
                    let k_row = ((bi * s_k + sk) * h_kv + hkv) * d;
                    let dot: f32 = (0..d).map(|j| q[q_row + j] * k[k_row + j]).sum();
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
                        o[q_row + j] += p * v[v_row + j];
                    }
                }
            }
        }
    }
    o
}

#[allow(clippy::too_many_arguments)]
async fn try_run(
    cl: u32,
    b: u32,
    h_q: u32,
    h_kv: u32,
    s_q: u32,
    s_k: u32,
    d: u32,
    has_mask: bool,
    seed: u64,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let backend = make_backend_with_f16().await?;
    if !backend.supports_subgroups() {
        eprintln!("skip: adapter does not expose SUBGROUP");
        return None;
    }
    let (sg_min, _) = backend.subgroup_size_range();
    if sg_min < cl {
        eprintln!("skip: subgroup floor {sg_min} < CL {cl}");
        return None;
    }

    let q_rows = (b * s_q * h_q) as usize;
    let kv_rows = (b * s_k * h_kv) as usize;
    let q = gen_f16_rows(q_rows * d as usize, seed);
    let k = gen_f16_rows(kv_rows * d as usize, seed ^ 0x1010);
    let v = gen_f16_rows(kv_rows * d as usize, seed ^ 0x2020);

    let mut rng = Rng::new(seed ^ 0x3030);
    let mask_f32: Vec<f32> = if has_mask {
        (0..b * s_q * s_k)
            .map(|_| f16_round(rng.f32_signed() * 0.5))
            .collect()
    } else {
        vec![0f32; (b * s_q * s_k) as usize]
    };
    // mask buffer is array<vec2<f16>>: pairs along S_k. Requires S_k % 2 == 0.
    assert!(s_k.is_multiple_of(2));

    let sm_scale = 1.0 / (d as f32).sqrt();
    let mask_ref = if has_mask {
        Some(mask_f32.as_slice())
    } else {
        None
    };
    let exp = cpu_ref(
        &q,
        &k,
        &v,
        mask_ref,
        b as usize,
        h_q as usize,
        h_kv as usize,
        s_q as usize,
        s_k as usize,
        d as usize,
        sm_scale,
    );

    let q_buf = alloc_with(&backend, &pack_f16_vec(&q));
    let k_buf = alloc_with(&backend, &pack_f16_vec(&k));
    let v_buf = alloc_with(&backend, &pack_f16_vec(&v));
    let mask_buf = alloc_with(&backend, &pack_f16_vec(&mask_f32));
    let out_buf = alloc_zero(&backend, (q_rows * d as usize * 2) as u64);

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

    let pipeline = backend
        .create_pipeline(
            "sdpa_sg_conf",
            &build_f16_sg_wgsl(cl),
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
            f16_sg_workgroups(cl, b, s_q, h_q),
        )
        .expect("dispatch");
    backend.submit(enc).await.expect("submit");

    let out_bytes = backend
        .read_buffer(out_buf.id, out_buf.offset, out_buf.len)
        .await
        .expect("read out");
    let got: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|p| f16_bits_to_f32(u16::from_le_bytes([p[0], p[1]])))
        .collect();

    for buf in [q_buf, k_buf, v_buf, mask_buf, out_buf, u_buf] {
        backend.free(buf.id);
    }
    Some((got, exp))
}

/// Temporal sliding-window subgroup SDPA vs reference. The reference is the
/// dense `cpu_ref` fed a synthesized additive mask that is 0 inside the window
/// (`|fq - fk| <= window`, frame = token / period) and a large negative outside,
/// so out-of-window keys get p_j = 0 -- exactly what the windowed kernel's
/// in-kernel fold computes (with `has_mask == 0`). `row_off` exercises the
/// chunked-Q path: each query's GLOBAL frame is `(row_off + sq) / period`.
#[allow(clippy::too_many_arguments)]
async fn try_run_windowed(
    cl: u32,
    s_q: u32,
    s_k: u32,
    d: u32,
    period: u32,
    window: u32,
    row_off: u32,
    seed: u64,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let backend = make_backend_with_f16().await?;
    if !backend.supports_subgroups() {
        eprintln!("skip: adapter does not expose SUBGROUP");
        return None;
    }
    let (sg_min, _) = backend.subgroup_size_range();
    if sg_min < cl {
        eprintln!("skip: subgroup floor {sg_min} < CL {cl}");
        return None;
    }
    assert!(s_k.is_multiple_of(period) && s_k.is_multiple_of(2));

    let (b, h_q, h_kv) = (1u32, 1u32, 1u32);
    let q = gen_f16_rows((s_q * d) as usize, seed);
    let k = gen_f16_rows((s_k * d) as usize, seed ^ 0x1010);
    let v = gen_f16_rows((s_k * d) as usize, seed ^ 0x2020);

    // Window as an additive mask for the dense reference: 0 in-window, -3e38 out.
    let mut win_mask = vec![0f32; (s_q * s_k) as usize];
    for sq in 0..s_q {
        let fq = (row_off + sq) / period;
        for sk in 0..s_k {
            let fk = sk / period;
            let dfr = fq.abs_diff(fk);
            win_mask[(sq * s_k + sk) as usize] = if dfr <= window { 0.0 } else { -3.0e38 };
        }
    }
    let sm_scale = 1.0 / (d as f32).sqrt();
    let exp = cpu_ref(
        &q,
        &k,
        &v,
        Some(&win_mask),
        b as usize,
        h_q as usize,
        h_kv as usize,
        s_q as usize,
        s_k as usize,
        d as usize,
        sm_scale,
    );

    let q_buf = alloc_with(&backend, &pack_f16_vec(&q));
    let k_buf = alloc_with(&backend, &pack_f16_vec(&k));
    let v_buf = alloc_with(&backend, &pack_f16_vec(&v));
    // has_mask == 0: the mask binding is unread, but slot 3 still needs a buffer.
    let mask_buf = alloc_zero(&backend, 16);
    let out_buf = alloc_zero(&backend, (s_q * d * 2) as u64);

    // 48-byte uniform: 8 base fields + period, window, row_off (+ 4 pad).
    let mut u = [0u8; 48];
    u[0..4].copy_from_slice(&b.to_le_bytes());
    u[4..8].copy_from_slice(&h_q.to_le_bytes());
    u[8..12].copy_from_slice(&h_kv.to_le_bytes());
    u[12..16].copy_from_slice(&s_q.to_le_bytes());
    u[16..20].copy_from_slice(&s_k.to_le_bytes());
    u[20..24].copy_from_slice(&d.to_le_bytes());
    u[24..28].copy_from_slice(&sm_scale.to_le_bytes());
    u[28..32].copy_from_slice(&0u32.to_le_bytes()); // has_mask
    u[32..36].copy_from_slice(&period.to_le_bytes());
    u[36..40].copy_from_slice(&window.to_le_bytes());
    u[40..44].copy_from_slice(&row_off.to_le_bytes());
    let u_buf = alloc_with(&backend, &u);

    let pipeline = backend
        .create_pipeline(
            "sdpa_sg_win_conf",
            &build_f16_sg_windowed_wgsl(cl),
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
            f16_sg_workgroups(cl, b, s_q, h_q),
        )
        .expect("dispatch");
    backend.submit(enc).await.expect("submit");

    let out_bytes = backend
        .read_buffer(out_buf.id, out_buf.offset, out_buf.len)
        .await
        .expect("read out");
    let got: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|p| f16_bits_to_f32(u16::from_le_bytes([p[0], p[1]])))
        .collect();

    for buf in [q_buf, k_buf, v_buf, mask_buf, out_buf, u_buf] {
        backend.free(buf.id);
    }
    Some((got, exp))
}

fn assert_dense_close(got: &[f32], exp: &[f32], rel_tol: f32, abs_tol: f32, label: &str) {
    assert_eq!(got.len(), exp.len(), "{label}: length mismatch");
    let max_abs_exp = exp.iter().fold(0f32, |a, &x| a.max(x.abs()));
    let tol = (rel_tol * max_abs_exp).max(abs_tol);
    let (max_err, idx) = got
        .iter()
        .zip(exp)
        .enumerate()
        .map(|(i, (g, e))| ((g - e).abs(), i))
        .fold((0f32, 0usize), |a, b| if b.0 > a.0 { b } else { a });
    assert!(
        max_err <= tol,
        "{label}: max abs diff {max_err:.3e} > tol {tol:.3e} at idx {idx} \
         (got={} exp={}) exp_max={}",
        got[idx],
        exp[idx],
        max_abs_exp,
    );
}

#[test]
fn sdpa_sg_wgsl_sanity() {
    for cl in [4u32, 8] {
        let src = build_f16_sg_wgsl(cl);
        assert!(src.contains("enable f16"));
        // Build sites prepend the web-only `enable subgroups;` directive.
        assert!(!src.contains("enable subgroups"));
        assert!(src.contains(&format!("const CL: u32 = {cl}u")));
        // Cluster dot-reduce; present in any form of this kernel.
        assert!(src.contains("subgroupShuffleXor"));
        // The dense kernel has no window machinery.
        assert!(!src.contains("u.window"));
        let win = build_f16_sg_windowed_wgsl(cl);
        assert!(win.contains("period: u32, window: u32, row_off: u32"));
        assert!(win.contains("u.window"));
        assert!(win.contains("subgroupShuffleXor"));
    }
    assert_eq!(<SdpaF32 as SdpaOp>::layout().len(), 6);
}

#[test]
fn sdpa_sg_windowed() {
    // 6 frames of 16 tokens (period=16), window=±1: each query frame attends to
    // 3 frames (or 2 at the ends). Two BC=32 tiles span the kept range; the
    // tile-skip + per-key fold must reproduce the masked reference.
    let Some((got, exp)) =
        pollster::block_on(try_run_windowed(8, 96, 96, 128, 16, 1, 0, 0x5D9A_3F00))
    else {
        return;
    };
    assert_dense_close(&got, &exp, 1e-2, 2e-3, "sdpa_sg_windowed");
}

#[test]
fn sdpa_sg_windowed_chunked() {
    // Chunked-Q path: this dispatch covers global rows [32, 64) (row_off=32),
    // so query frames are (32 + sq)/16. Validates the global-frame recovery the
    // chunked f16 SDPA relies on. window=0 -> strictly intra-frame attention.
    let Some((got, exp)) =
        pollster::block_on(try_run_windowed(8, 32, 96, 128, 16, 0, 32, 0x5D9A_44C0))
    else {
        return;
    };
    assert_dense_close(&got, &exp, 1e-2, 2e-3, "sdpa_sg_windowed_chunked");
}

#[test]
fn sdpa_sg_small_no_mask() {
    // B=1, H=1, S_q=S_k=64 (two BC=32 tiles), D=128 (full n_l).
    let Some((got, exp)) = pollster::block_on(try_run(8, 1, 1, 1, 64, 64, 128, false, 0x5D9A_51D0))
    else {
        return;
    };
    assert_dense_close(&got, &exp, 1e-2, 2e-3, "sdpa_sg_small");
}

#[test]
fn sdpa_sg_with_mask() {
    let Some((got, exp)) = pollster::block_on(try_run(8, 1, 1, 1, 64, 64, 128, true, 0x5D9A_BEEF))
    else {
        return;
    };
    assert_dense_close(&got, &exp, 1e-2, 2e-3, "sdpa_sg_with_mask");
}

#[test]
fn sdpa_sg_gqa() {
    // H_q=2, H_kv=1 -> both heads share K/V.
    let Some((got, exp)) = pollster::block_on(try_run(8, 1, 2, 1, 64, 64, 128, false, 0x5D9A_ABCD))
    else {
        return;
    };
    assert_dense_close(&got, &exp, 1e-2, 2e-3, "sdpa_sg_gqa");
}

#[test]
fn sdpa_sg_tails() {
    // S_q=33 (row-tail: 15 invalid rows in the last BR=16 group), S_k=40
    // (key-tail: 24 folded keys in tile 2), D=64 (partial n_l guards).
    let Some((got, exp)) = pollster::block_on(try_run(8, 1, 1, 1, 33, 40, 64, true, 0x5D9A_7A11))
    else {
        return;
    };
    assert_dense_close(&got, &exp, 1e-2, 2e-3, "sdpa_sg_tails");
}

#[test]
fn sdpa_sg_cl4() {
    // CL=4 codegen (web/mobile shape): 8 score regs, 2 xor hops, n_l=8.
    let Some((got, exp)) = pollster::block_on(try_run(4, 1, 1, 1, 64, 64, 128, true, 0x5D9A_C14A))
    else {
        return;
    };
    assert_dense_close(&got, &exp, 1e-2, 2e-3, "sdpa_sg_cl4");
}
