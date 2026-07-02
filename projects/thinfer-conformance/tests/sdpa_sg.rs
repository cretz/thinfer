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
///
/// `txt_start` is the first joint-token index that is text (always-in-window):
/// image queries (`global row < txt_start`) attend their frame window PLUS all
/// text keys; text queries (`global row >= txt_start`) attend everything; text
/// keys (`sk >= txt_start`) are always in-window. Pure-video callers pass
/// `txt_start == s_k` (the text branches go dead, recovering plain windowing).
#[allow(clippy::too_many_arguments)]
async fn try_run_windowed(
    cl: u32,
    s_q: u32,
    s_k: u32,
    d: u32,
    period: u32,
    window: u32,
    row_off: u32,
    txt_start: u32,
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
    // The image region (frames) must be period-aligned; the text tail need not be.
    assert!(txt_start.is_multiple_of(period) && s_k.is_multiple_of(2));

    let (b, h_q, h_kv) = (1u32, 1u32, 1u32);
    let q = gen_f16_rows((s_q * d) as usize, seed);
    let k = gen_f16_rows((s_k * d) as usize, seed ^ 0x1010);
    let v = gen_f16_rows((s_k * d) as usize, seed ^ 0x2020);

    // Window as an additive mask for the dense reference: 0 in-window, -3e38 out.
    // Joint semantics: text queries and text keys bypass the frame window.
    let mut win_mask = vec![0f32; (s_q * s_k) as usize];
    for sq in 0..s_q {
        let gq = row_off + sq;
        let fq = gq / period;
        let is_txt_q = gq >= txt_start;
        for sk in 0..s_k {
            let is_txt_k = sk >= txt_start;
            let fk = sk / period;
            let in_win = is_txt_q || is_txt_k || fq.abs_diff(fk) <= window;
            win_mask[(sq * s_k + sk) as usize] = if in_win { 0.0 } else { -3.0e38 };
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

    // 48-byte uniform: 8 base fields + period, window, row_off, txt_start.
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
    u[44..48].copy_from_slice(&txt_start.to_le_bytes());
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

/// Queries-per-dispatch cap for the f16 subgroup SDPA. Mirror of
/// `thinfer_models::common::block::sdpa_chunk_rows_f16` so the test chunks the
/// query range exactly like the real dispatch (op_sdpa_impl f16 path).
fn sdpa_chunk_rows_f16(s_k: u32) -> u32 {
    const MAX_QK: u64 = 20_000_000;
    let rows = (MAX_QK / s_k.max(1) as u64) as u32;
    (rows / 64 * 64).max(64)
}

/// Like [`try_run_windowed`] but drives the REAL chunked dispatch: it splits the
/// query range into `sdpa_chunk_rows_f16(s_k)`-row chunks and issues one windowed
/// dispatch per chunk with `row_off = r0` (each chunk in its own submit), exactly
/// as `op_sdpa_impl` does for the mixed-precision f16 path. K/V stay whole; only
/// Q is chunked. This is the path the single-dispatch `try_run_windowed` tests
/// never hit: a second chunk with a non-zero `row_off` whose workgroups span high
/// image frames and the image/text straddle. The CPU reference is the full joint
/// windowed attention over all `s_q` rows.
#[allow(clippy::too_many_arguments)]
async fn try_run_windowed_chunked(
    cl: u32,
    s_q: u32,
    s_k: u32,
    d: u32,
    period: u32,
    window: u32,
    txt_start: u32,
    seed: u64,
) -> Option<(Vec<f32>, Vec<f32>, u32)> {
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
    // The image region (frames) must be period-aligned. s_k may be odd here:
    // has_mask==0 so the vec2<f16> mask (which needs s_k % 2 == 0) is never read.
    assert!(txt_start.is_multiple_of(period));

    let (b, h_q, h_kv) = (1u32, 1u32, 1u32);
    let q = gen_f16_rows((s_q * d) as usize, seed);
    let k = gen_f16_rows((s_k * d) as usize, seed ^ 0x1010);
    let v = gen_f16_rows((s_k * d) as usize, seed ^ 0x2020);

    // Full joint windowed reference over every query row (mask 0 in-window, else
    // -3e38). Image query at frame fq attends image keys in frames [fq-W, fq+W]
    // PLUS all text keys; text query (global row >= txt_start) attends all keys.
    let mut win_mask = vec![0f32; (s_q * s_k) as usize];
    for sq in 0..s_q {
        let fq = sq / period;
        let is_txt_q = sq >= txt_start;
        for sk in 0..s_k {
            let is_txt_k = sk >= txt_start;
            let fk = sk / period;
            let in_win = is_txt_q || is_txt_k || fq.abs_diff(fk) <= window;
            win_mask[(sq * s_k + sk) as usize] = if in_win { 0.0 } else { -3.0e38 };
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

    // K/V uploaded whole (indexed globally by the kernel); Q chunked per dispatch.
    let k_buf = alloc_with(&backend, &pack_f16_vec(&k));
    let v_buf = alloc_with(&backend, &pack_f16_vec(&v));
    let mask_buf = alloc_zero(&backend, 16);

    let pipeline = backend
        .create_pipeline(
            "sdpa_sg_win_chunked_conf",
            &build_f16_sg_windowed_wgsl(cl),
            "main",
            <SdpaF32 as SdpaOp>::layout(),
        )
        .await
        .expect("pipeline");

    let max_rows = sdpa_chunk_rows_f16(s_k);
    let mut got = vec![0f32; (s_q * d) as usize];
    let mut n_chunks = 0u32;
    let mut r0 = 0u32;
    while r0 < s_q {
        let rows = (s_q - r0).min(max_rows);
        n_chunks += 1;
        let q_slice = &q[(r0 * d) as usize..((r0 + rows) * d) as usize];
        let q_buf = alloc_with(&backend, &pack_f16_vec(q_slice));
        let out_buf = alloc_zero(&backend, (rows * d * 2) as u64);

        // 48-byte uniform: chunk s_q is `rows`, row_off is the global r0.
        let mut u = [0u8; 48];
        u[0..4].copy_from_slice(&b.to_le_bytes());
        u[4..8].copy_from_slice(&h_q.to_le_bytes());
        u[8..12].copy_from_slice(&h_kv.to_le_bytes());
        u[12..16].copy_from_slice(&rows.to_le_bytes());
        u[16..20].copy_from_slice(&s_k.to_le_bytes());
        u[20..24].copy_from_slice(&d.to_le_bytes());
        u[24..28].copy_from_slice(&sm_scale.to_le_bytes());
        u[28..32].copy_from_slice(&0u32.to_le_bytes()); // has_mask
        u[32..36].copy_from_slice(&period.to_le_bytes());
        u[36..40].copy_from_slice(&window.to_le_bytes());
        u[40..44].copy_from_slice(&r0.to_le_bytes());
        u[44..48].copy_from_slice(&txt_start.to_le_bytes());
        let u_buf = alloc_with(&backend, &u);

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
                f16_sg_workgroups(cl, b, rows, h_q),
            )
            .expect("dispatch");
        backend.submit(enc).await.expect("submit");

        let out_bytes = backend
            .read_buffer(out_buf.id, out_buf.offset, out_buf.len)
            .await
            .expect("read out");
        let chunk: Vec<f32> = out_bytes
            .chunks_exact(2)
            .map(|p| f16_bits_to_f32(u16::from_le_bytes([p[0], p[1]])))
            .collect();
        got[(r0 * d) as usize..((r0 + rows) * d) as usize].copy_from_slice(&chunk);

        backend.free(q_buf.id);
        backend.free(out_buf.id);
        backend.free(u_buf.id);
        r0 += rows;
    }

    for buf in [k_buf, v_buf, mask_buf] {
        backend.free(buf.id);
    }
    Some((got, exp, n_chunks))
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
        assert!(win.contains("period: u32, window: u32, row_off: u32, txt_start: u32"));
        assert!(win.contains("u.window"));
        assert!(win.contains("u.txt_start"));
        assert!(win.contains("subgroupShuffleXor"));
    }
    assert_eq!(<SdpaF32 as SdpaOp>::layout().len(), 6);
}

#[test]
fn sdpa_sg_windowed() {
    // 6 frames of 16 tokens (period=16), window=±1: each query frame attends to
    // 3 frames (or 2 at the ends). Two BC=32 tiles span the kept range; the
    // tile-skip + per-key fold must reproduce the masked reference. Pure video
    // (txt_start == s_k): the joint branches stay dead.
    let Some((got, exp)) =
        pollster::block_on(try_run_windowed(8, 96, 96, 128, 16, 1, 0, 96, 0x5D9A_3F00))
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
        pollster::block_on(try_run_windowed(8, 32, 96, 128, 16, 0, 32, 96, 0x5D9A_44C0))
    else {
        return;
    };
    assert_dense_close(&got, &exp, 1e-2, 2e-3, "sdpa_sg_windowed_chunked");
}

#[test]
fn sdpa_sg_windowed_joint() {
    // Joint [img ; txt]: 4 image frames x 16 tokens (txt_start=64, BC-aligned) +
    // 32 text tokens -> s_k=96. window=1: image queries attend their ±1 frame
    // window AND all 32 text keys; the 32 text queries (rows 64..96) attend
    // everything. Exercises the always-in-window text tiles + the full-attention
    // text workgroups against the joint-masked reference.
    let Some((got, exp)) =
        pollster::block_on(try_run_windowed(8, 96, 96, 128, 16, 1, 0, 64, 0x5D9A_70A1))
    else {
        return;
    };
    assert_dense_close(&got, &exp, 1e-2, 2e-3, "sdpa_sg_windowed_joint");
}

#[test]
fn sdpa_sg_windowed_joint_unaligned() {
    // Text boundary NOT on a BC=32 tile edge: 3 image frames x 16 (txt_start=48)
    // + 16 text tokens -> s_k=64. The straddling tile [32,64) holds image keys
    // 32..47 (frame 2) and text keys 48..63; the per-key in-window fold must keep
    // the text keys and window the image keys correctly.
    let Some((got, exp)) =
        pollster::block_on(try_run_windowed(8, 64, 64, 128, 16, 1, 0, 48, 0x5D9A_88B2))
    else {
        return;
    };
    assert_dense_close(&got, &exp, 1e-2, 2e-3, "sdpa_sg_windowed_joint_unaligned");
}

#[test]
fn sdpa_sg_windowed_joint_chunked() {
    // Chunked-Q crossing the image/text boundary: global rows [48, 80) (row_off=
    // 48) over a 4-image-frame (txt_start=64) + 32-text sequence. Rows 48..63 are
    // image queries (frame 3, window=1), rows 64..79 are text queries (full); a
    // single workgroup may straddle the boundary and must fall back to full.
    let Some((got, exp)) =
        pollster::block_on(try_run_windowed(8, 32, 96, 128, 16, 1, 48, 64, 0x5D9A_99C3))
    else {
        return;
    };
    assert_dense_close(&got, &exp, 1e-2, 2e-3, "sdpa_sg_windowed_joint_chunked");
}

#[test]
fn sdpa_sg_windowed_joint_realshape_chunked() {
    // Real-dims-SHAPED joint windowed case that drives the REAL two-chunk f16
    // dispatch AND the image/text straddle workgroup, but small enough for a fast
    // CPU reference. period=200, 25 image frames => n_img=txt_start=5000 (note
    // 5000 % BR(=16 on NVIDIA CL=8) == 8, so a workgroup straddles the img/txt
    // boundary), + seq_txt=53 (odd, non-tile-aligned) => s_q=s_k=5053, d=128,
    // window=3. s_q=5053 > max_rows(=floor(20M/5053)/64*64=3904) so the f16 path
    // CHUNKS into 2 dispatches; the 2nd chunk's row_off=3904 is non-zero and its
    // workgroups span high image frames + the img/txt boundary -- the path the
    // tiny single-dispatch tests never hit. Must not device-lose and must match
    // the full joint windowed reference within the f16 tolerance.
    let Some((got, exp, n_chunks)) = pollster::block_on(try_run_windowed_chunked(
        8,
        5053,
        5053,
        128,
        200,
        3,
        5000,
        0x5D9A_C0DE,
    )) else {
        return;
    };
    assert_eq!(
        n_chunks, 2,
        "expected the f16 path to chunk into 2 dispatches"
    );
    assert_dense_close(
        &got,
        &exp,
        1e-2,
        2e-3,
        "sdpa_sg_windowed_joint_realshape_chunked",
    );
}

#[test]
fn sdpa_sg_windowed_joint_realshape_chunked_w1() {
    // Same chunked + straddle stress as `_realshape_chunked` but window=1 and a
    // different image/text split: period=192, 26 image frames => n_img=txt_start=
    // 4992 (BR-aligned boundary) but still lands inside the 2nd chunk since 4992 >
    // max_rows(=floor(20M/5063)/64*64=3968); seq_txt=71 (odd) => s_q=s_k=5063.
    // Varies the window arithmetic (kf_lo/kf_hi clamp) and the straddle position
    // relative to the chunk boundary. Fresh data draw via a different seed.
    let Some((got, exp, n_chunks)) = pollster::block_on(try_run_windowed_chunked(
        8,
        5063,
        5063,
        128,
        192,
        1,
        4992,
        0x5D9A_1CE0,
    )) else {
        return;
    };
    assert_eq!(
        n_chunks, 2,
        "expected the f16 path to chunk into 2 dispatches"
    );
    assert_dense_close(
        &got,
        &exp,
        1e-2,
        2e-3,
        "sdpa_sg_windowed_joint_realshape_chunked_w1",
    );
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
