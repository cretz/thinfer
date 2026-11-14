//! GPU vs host reference for `ops::window_attn` and `ops::hadamard` (the
//! RelayVSR kernels). Skips without an adapter.
//! `cargo test --release -p thinfer-core --test window_attn -- --nocapture`

use std::sync::Arc;

use half::f16;
use thinfer_core::arbiter::{MemArbiter, MemTier};
use thinfer_core::backend::{Backend, BufRef, WgpuBackend};
use thinfer_core::ops::window_attn::{self, WindowAttnUniform};
use thinfer_core::ops::{ActDtype, hadamard};
use thinfer_core::workspace::Workspace;

fn rand(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn encode(act: ActDtype, x: &[f32]) -> Vec<u8> {
    match act {
        ActDtype::F32 => x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ActDtype::F16 => x
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect(),
        _ => unreachable!(),
    }
}

fn decode(act: ActDtype, b: &[u8]) -> Vec<f32> {
    match act {
        ActDtype::F32 => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        ActDtype::F16 => b
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        _ => unreachable!(),
    }
}

fn upload(be: &WgpuBackend, bytes: &[u8]) -> BufRef {
    let id = be.allocate(bytes.len() as u64).unwrap();
    be.write_buffer(id, 0, bytes).unwrap();
    BufRef::view(id, 0, bytes.len() as u64)
}

fn max_rel(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())) + 1e-6;
    got.iter()
        .zip(want)
        .fold(0f32, |m, (g, w)| m.max((g - w).abs()))
        / scale
}

async fn run_window(be: &Arc<WgpuBackend>, act: ActDtype, u: &WindowAttnUniform, tol: f32) {
    let q_n = (u.b * u.tq * u.h * u.w * u.nh * u.hd) as usize;
    let kv_n = (u.b * u.tk * u.h * u.w * u.kvh * u.hd) as usize;
    let (q, k, v) = (rand(q_n, 1), rand(kv_n, 2), rand(kv_n, 3));
    let want = window_attn::window_attn_host(&q, &k, &v, u);
    let pipe = be
        .create_pipeline(
            "window_attn",
            window_attn::wgsl(act),
            "main",
            window_attn::layout(),
        )
        .await
        .unwrap();
    let (qb, kb, vb) = (
        upload(be, &encode(act, &q)),
        upload(be, &encode(act, &k)),
        upload(be, &encode(act, &v)),
    );
    let ob = upload(be, &vec![0u8; encode(act, &q).len()]);
    let ws = Workspace::new(be.clone(), MemArbiter::new(MemTier::Vram, u64::MAX));
    let scope = ws.batch();
    let un = scope.write_uniform(&u.to_bytes()).unwrap();
    let bufs = [qb, kb, vb, ob].map(|b| scope.import_copy(b));
    scope
        .dispatch(
            &pipe,
            &[bufs[0], bufs[1], bufs[2], bufs[3], un],
            window_attn::workgroups(u),
        )
        .unwrap();
    scope.submit_void().await.unwrap();
    let got = decode(act, &be.read_buffer(ob.id, 0, ob.len).await.unwrap());
    let r = max_rel(&got, &want);
    eprintln!(
        "window_attn {act:?} tq={} tk={} {}x{} rel={r:.2e}",
        u.tq, u.tk, u.h, u.w
    );
    assert!(r < tol, "window_attn {act:?} rel {r}");
}

async fn run_hadamard(be: &Arc<WgpuBackend>, act: ActDtype, tol: f32) {
    let (rows, k) = (5u32, 768u32);
    let x = rand((rows * k) as usize, 9);
    let mut want = x.clone();
    hadamard::rotate_host(&mut want, k as usize);
    let pipe = be
        .create_pipeline("hadamard", hadamard::wgsl(act), "main", hadamard::layout())
        .await
        .unwrap();
    let xb = upload(be, &encode(act, &x));
    let ws = Workspace::new(be.clone(), MemArbiter::new(MemTier::Vram, u64::MAX));
    let scope = ws.batch();
    let un = scope
        .write_uniform(&[rows, k, 0, 0].map(u32::to_le_bytes).concat())
        .unwrap();
    let xh = scope.import_copy(xb);
    scope
        .dispatch(&pipe, &[xh, un], hadamard::workgroups(rows, k))
        .unwrap();
    scope.submit_void().await.unwrap();
    let got = decode(act, &be.read_buffer(xb.id, 0, xb.len).await.unwrap());
    let r = max_rel(&got, &want);
    eprintln!("hadamard {act:?} rel={r:.2e}");
    assert!(r < tol, "hadamard {act:?} rel {r}");
}

#[test]
fn window_attn_and_hadamard_match_host() {
    pollster::block_on(async {
        let Ok(be) = WgpuBackend::new().await else {
            eprintln!("skipped: no adapter");
            return;
        };
        // CI's Windows runners only have WARP, which did not finish this test
        // in 30+ minutes (the kernel's 128-float per-thread arrays).
        if be.is_software() {
            eprintln!("skipped: software adapter");
            return;
        }
        let be = Arc::new(be);
        let mut acts = vec![(ActDtype::F32, 1e-4f32)];
        if be.supports_shader_f16() {
            acts.push((ActDtype::F16, 2e-2));
        }
        // Keyframe-Wan-like: clamped, one window, causal temporal window,
        // a row wider than one 64-query segment.
        let wan = WindowAttnUniform {
            b: 1,
            tq: 3,
            tk: 3,
            h: 9,
            w: 70,
            nh: 2,
            kvh: 2,
            hd: 128,
            temporal_window: 2,
            clamp: true,
            scale: 128f32.powf(-0.5),
            windows: WindowAttnUniform::same_window(5, 11),
        };
        // Flash-like: one query plane over [cond, cond, past, current] with
        // per-plane windows, unclamped, grouped-query 4/2.
        let mut windows = WindowAttnUniform::same_window(3, 3);
        windows[0] = (7, 7);
        windows[1] = (7, 7);
        let flash = WindowAttnUniform {
            b: 1,
            tq: 1,
            tk: 4,
            h: 10,
            w: 12,
            nh: 4,
            kvh: 2,
            hd: 64,
            temporal_window: 4,
            clamp: false,
            scale: 64f32.powf(-0.5),
            windows,
        };
        for (act, tol) in acts {
            run_window(&be, act, &wan, tol).await;
            run_window(&be, act, &flash, tol).await;
            run_hadamard(&be, act, tol).await;
        }
    });
}

/// Timing at the RelayVSR shapes on ONE query plane (bounded so a slow kernel
/// cannot reach the TDR): `THINFER_WINDOW_BENCH=1`.
#[test]
fn window_attn_bench() {
    if std::env::var_os("THINFER_WINDOW_BENCH").is_none() {
        return;
    }
    pollster::block_on(async {
        let be = Arc::new(WgpuBackend::new().await.expect("wgpu"));
        let act = ActDtype::F16;
        let cases = [
            (
                "wan 1 plane",
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
            ),
            ("flash frame", {
                let mut windows = WindowAttnUniform::same_window(24, 24);
                windows[0] = (48, 48);
                windows[1] = (48, 48);
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
                    windows,
                }
            }),
        ];
        for (name, u) in cases {
            let acts: [ActDtype; 2] = [act, ActDtype::F32];
            for a in acts {
                let q_n = (u.tq * u.h * u.w * u.nh * u.hd) as usize;
                let kv_n = (u.tk * u.h * u.w * u.kvh * u.hd) as usize;
                let pipe = be
                    .create_pipeline("wa", window_attn::wgsl(a), "main", window_attn::layout())
                    .await
                    .unwrap();
                let (qb, kb, vb) = (
                    upload(&be, &encode(a, &rand(q_n, 1))),
                    upload(&be, &encode(a, &rand(kv_n, 2))),
                    upload(&be, &encode(a, &rand(kv_n, 3))),
                );
                let ob = upload(&be, &vec![0u8; encode(a, &rand(q_n, 1)).len()]);
                let ws = Workspace::new(be.clone(), MemArbiter::new(MemTier::Vram, u64::MAX));
                for it in 0..3 {
                    let t0 = std::time::Instant::now();
                    let scope = ws.batch();
                    let un = scope.write_uniform(&u.to_bytes()).unwrap();
                    let b = [qb, kb, vb, ob].map(|x| scope.import_copy(x));
                    scope
                        .dispatch(
                            &pipe,
                            &[b[0], b[1], b[2], b[3], un],
                            window_attn::workgroups(&u),
                        )
                        .unwrap();
                    scope.submit_void().await.unwrap();
                    let _ = be.read_buffer(ob.id, 0, 16).await.unwrap();
                    let ms = t0.elapsed().as_secs_f64() * 1e3;
                    let keys: f64 = if u.clamp {
                        4.0 * 22.0 * 40.0
                    } else {
                        2.0 * 48.0 * 48.0 + 2.0 * 24.0 * 24.0
                    };
                    let flops = (u.h * u.w * u.nh) as f64 * keys * u.hd as f64 * 4.0;
                    eprintln!(
                        "{name} {a:?} it{it}: {ms:.1} ms ({:.2} TFLOP/s)",
                        flops / ms / 1e9
                    );
                }
            }
        }
    });
}
