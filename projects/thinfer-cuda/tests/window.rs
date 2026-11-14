//! The CUDA window attention against the f32 host reference of the
//! `ops::window_attn` contract (`window_attn_host`): clamped and unclamped,
//! per-plane windows, GQA, both head dims, tiles that overhang the frame.
//! `THINFER_WINDOW_BENCH=1` adds a timing run at the RelayVSR shapes, bounded
//! to ONE query plane per launch. Skips without a CUDA device / NVRTC.
#![cfg(not(target_os = "macos"))]

use std::sync::Arc;

use cudarc::driver::{CudaContext, sys};
use half::f16;
use thinfer_core::ops::window_attn::{WindowAttnUniform, window_attn_host};
use thinfer_cuda::window::WindowKernels;

fn rand(n: usize, seed: u64) -> Vec<f16> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            f16::from_f32(((s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0)
        })
        .collect()
}

fn upload(ctx: &Arc<CudaContext>, bytes: &[u8]) -> sys::CUdeviceptr {
    ctx.bind_to_thread().unwrap();
    let mut p = 0;
    unsafe {
        sys::cuMemAlloc_v2(&mut p, bytes.len().max(16))
            .result()
            .unwrap();
        sys::cuMemcpyHtoD_v2(p, bytes.as_ptr().cast(), bytes.len())
            .result()
            .unwrap();
    }
    p
}

fn bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

fn setup() -> Option<(Arc<CudaContext>, WindowKernels)> {
    if !thinfer_cuda::driver_present() {
        eprintln!("skipped[cuda window]: no CUDA driver");
        return None;
    }
    let ctx = CudaContext::new(0).ok()?;
    let sm = ctx.compute_capability().ok()?;
    match WindowKernels::new(ctx.clone(), sm) {
        Ok(k) => Some((ctx, k)),
        Err(e) => {
            eprintln!("skipped[cuda window]: {e}");
            None
        }
    }
}

/// Returns (cuda output f32, run ms).
fn run(
    ctx: &Arc<CudaContext>,
    k: &WindowKernels,
    u: &WindowAttnUniform,
    q: &[f16],
    kk: &[f16],
    v: &[f16],
) -> (Vec<f32>, f64) {
    let (pq, pk, pv) = (
        upload(ctx, bytes(q)),
        upload(ctx, bytes(kk)),
        upload(ctx, bytes(v)),
    );
    let po = upload(ctx, &vec![0u8; q.len() * 2]);
    let t0 = std::time::Instant::now();
    assert!(k.run(u, pq, pk, pv, po).unwrap(), "kernel declined");
    // `run` only enqueues.
    unsafe { sys::cuCtxSynchronize().result().unwrap() };
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut out = vec![f16::ZERO; q.len()];
    unsafe {
        sys::cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), po, q.len() * 2)
            .result()
            .unwrap();
        for p in [pq, pk, pv, po] {
            sys::cuMemFree_v2(p).result().unwrap();
        }
    }
    (out.iter().map(|x| x.to_f32()).collect(), ms)
}

fn shape(u: &WindowAttnUniform) -> (usize, usize) {
    (
        (u.b * u.tq * u.h * u.w * u.nh * u.hd) as usize,
        (u.b * u.tk * u.h * u.w * u.kvh * u.hd) as usize,
    )
}

#[test]
fn cuda_window_matches_reference() {
    let Some((ctx, k)) = setup() else { return };
    let mut flash = WindowAttnUniform::same_window(5, 5);
    flash[0] = (9, 9);
    flash[1] = (9, 9);
    let cases = [
        // keyframe-like: clamped, one window, 5 planes > temporal 4, a row
        // wider than one tile.
        WindowAttnUniform {
            b: 1,
            tq: 5,
            tk: 5,
            h: 11,
            w: 70,
            nh: 2,
            kvh: 2,
            hd: 128,
            temporal_window: 4,
            clamp: true,
            scale: 128f32.powf(-0.5),
            windows: WindowAttnUniform::same_window(7, 13),
        },
        // Flash-like: one query plane over [cond, cond, past, current], cut at
        // the border, GQA 8/2.
        WindowAttnUniform {
            b: 1,
            tq: 1,
            tk: 4,
            h: 13,
            w: 21,
            nh: 8,
            kvh: 2,
            hd: 64,
            temporal_window: 4,
            clamp: false,
            scale: 0.125,
            windows: flash,
        },
        // window bigger than the frame (clamp shrinks it), d64
        WindowAttnUniform {
            b: 1,
            tq: 2,
            tk: 3,
            h: 6,
            w: 9,
            nh: 4,
            kvh: 4,
            hd: 64,
            temporal_window: 2,
            clamp: true,
            scale: 0.125,
            windows: WindowAttnUniform::same_window(22, 40),
        },
    ];
    for u in cases {
        let (qn, kvn) = shape(&u);
        let (q, kk, v) = (rand(qn, 1), rand(kvn, 2), rand(kvn, 3));
        let f = |x: &[f16]| x.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
        let want = window_attn_host(&f(&q), &f(&kk), &f(&v), &u);
        let (got, _) = run(&ctx, &k, &u, &q, &kk, &v);
        let max_ref = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let err = got
            .iter()
            .zip(&want)
            .fold(0f32, |m, (g, w)| m.max((g - w).abs()))
            / max_ref;
        eprintln!(
            "cuda window tq={} tk={} {}x{} d{} clamp={} rel={err:.2e}",
            u.tq, u.tk, u.h, u.w, u.hd, u.clamp
        );
        assert!(err < 1e-2, "cuda window rel {err}");
    }
}

#[test]
fn cuda_window_bench() {
    if std::env::var_os("THINFER_WINDOW_BENCH").is_none() {
        return;
    }
    let Some((ctx, k)) = setup() else { return };
    let mut flash = WindowAttnUniform::same_window(24, 24);
    flash[0] = (48, 48);
    flash[1] = (48, 48);
    let cases = [
        (
            "keyframe plane x2 480p",
            WindowAttnUniform {
                b: 1,
                tq: 1,
                tk: 4,
                h: 30,
                w: 52,
                nh: 24,
                kvh: 24,
                hd: 128,
                temporal_window: 4,
                clamp: true,
                scale: 128f32.powf(-0.5),
                windows: WindowAttnUniform::same_window(22, 40),
            },
            4.0 * 22.0 * 40.0,
        ),
        (
            "flash frame x2 480p",
            WindowAttnUniform {
                b: 1,
                tq: 1,
                tk: 4,
                h: 60,
                w: 104,
                nh: 8,
                kvh: 2,
                hd: 64,
                temporal_window: 4,
                clamp: false,
                scale: 0.125,
                windows: flash,
            },
            2.0 * 48.0 * 48.0 + 2.0 * 24.0 * 24.0,
        ),
    ];
    for (name, u, keys) in cases {
        let (qn, kvn) = shape(&u);
        let (q, kk, v) = (rand(qn, 1), rand(kvn, 2), rand(kvn, 3));
        for it in 0..3 {
            let (_, ms) = run(&ctx, &k, &u, &q, &kk, &v);
            let flops = (u.h * u.w * u.nh) as f64 * keys * u.hd as f64 * 4.0;
            eprintln!(
                "{name} it{it}: {ms:.2} ms ({:.1} useful TFLOP/s)",
                flops / ms / 1e9
            );
        }
    }
}
