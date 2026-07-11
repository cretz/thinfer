//! Cooperative-matrix (tensor-core) matmul (`thinfer_core::ops::matmul_coopmat`)
//! vs an f32 CPU reference.
//!
//! This is also the end-to-end PROBE for the coopmat stack: it confirms that
//! (1) a Vulkan adapter on this machine exposes `VK_KHR_cooperative_matrix`
//! with a usable f16/f32 square config, (2) naga compiles WGSL `coop_matNxN`
//! load/store/multiply-add into a runnable SPIR-V pipeline, and (3) the result
//! matches a scalar matmul within f16 tolerance. When the device has no
//! coopmat support the test SKIPS (prints + returns) rather than failing, so
//! it stays green on non-Vulkan / non-tensor-core hardware.

#![cfg(feature = "conformance")]

use half::f16;
use thinfer_core::backend::{Backend, BufRef, PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::ops::matmul_coopmat::{
    CoopmatBufs, CoopmatMatmulConfig, CoopmatOut, build_wgsl, dispatch_coopmat, layout,
};

/// Coopmat lives on the discrete GPU's tensor cores; an integrated adapter may
/// expose only a non-square config we can't use. Steer to the high-performance
/// adapter, matching the production default (`BackendConfig`).
async fn discrete_backend() -> WgpuBackend {
    WgpuBackend::new_with_config(WgpuConfig {
        power_preference: PowerPreference::HighPerformance,
        timestamps: false,
        disable_coopmat: false,
    })
    .await
    .expect("wgpu adapter")
}

fn pack_dims(m: u32, n: u32, k: u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&m.to_le_bytes());
    out[4..8].copy_from_slice(&n.to_le_bytes());
    out[8..12].copy_from_slice(&k.to_le_bytes());
    out
}

fn to_f16_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for &x in v {
        out.extend_from_slice(&f16::from_f32(x).to_bits().to_le_bytes());
    }
    out
}

/// Scalar matmul reference, but A and B first round-tripped through f16 so the
/// reference shares the kernel's input precision (the only remaining
/// difference is f32-accumulate order, well within tolerance).
fn cpu_ref(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let af: Vec<f32> = a.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let bf: Vec<f32> = b.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += af[i * k + kk] * bf[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

async fn run_f32_out(m: u32, n: u32, k: u32) -> Option<(Vec<f32>, Vec<f32>)> {
    run_f32_out_layout(m, n, k, false).await
}

async fn run_f32_out_layout(
    m: u32,
    n: u32,
    k: u32,
    b_col_major: bool,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let backend = discrete_backend().await;
    let Some(hw) = backend.coopmat() else {
        eprintln!("[coopmat] device reports no cooperative-matrix support; SKIP");
        return None;
    };
    let (smin, smax) = backend.subgroup_size_range();
    if smin != smax {
        eprintln!("[coopmat] non-uniform subgroup size {smin}..{smax}; SKIP");
        return None;
    }
    let t = hw.tile;
    assert!(
        m.is_multiple_of(t) && n.is_multiple_of(t) && k.is_multiple_of(t),
        "test dims must be multiples of the tile {t}"
    );
    eprintln!("[coopmat] tile={t} subgroup={smin} -> running {m}x{n}x{k} bcm={b_col_major}");

    let mut cfg = CoopmatMatmulConfig::new(t, smin, CoopmatOut::F32);
    cfg.b_col_major = b_col_major;
    let wgsl = build_wgsl(&cfg);
    let pipeline = backend
        .create_pipeline("matmul_coopmat", &wgsl, "main", layout())
        .await
        .expect("coopmat pipeline (naga should compile coop_mat WGSL)");

    let mut seed = 0x1234_5678u64;
    let mut rand = || -> f32 {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((seed >> 33) as u32 as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };
    let a: Vec<f32> = (0..(m * k)).map(|_| rand()).collect();
    let b: Vec<f32> = (0..(k * n)).map(|_| rand()).collect(); // logical [K,N]
    let expect = cpu_ref(&a, &b, m as usize, n as usize, k as usize);

    // Storage: [K,N] row-major for coopLoadT, or [N,K] (transposed) for the
    // column-major n-major path.
    let b_store: Vec<f32> = if b_col_major {
        let (ku, nu) = (k as usize, n as usize);
        let mut t = vec![0f32; ku * nu];
        for kk in 0..ku {
            for nn in 0..nu {
                t[nn * ku + kk] = b[kk * nu + nn];
            }
        }
        t
    } else {
        b.clone()
    };

    let alloc_with = |bytes: &[u8]| -> BufRef {
        let id = backend.allocate(bytes.len() as u64).expect("allocate");
        backend.write_buffer(id, 0, bytes).expect("write");
        BufRef::new(id, bytes.len() as u64)
    };
    let a_buf = alloc_with(&to_f16_bytes(&a));
    let b_buf = alloc_with(&to_f16_bytes(&b_store));
    let dims_buf = alloc_with(&pack_dims(m, n, k));
    let out_len = (m as u64) * (n as u64) * 4;
    let out_id = backend.allocate(out_len).expect("alloc out");
    let out_buf = BufRef::new(out_id, out_len);

    let mut enc = backend.create_command_encoder();
    dispatch_coopmat(
        &backend,
        &mut enc,
        &pipeline,
        &cfg,
        &CoopmatBufs {
            a: &a_buf,
            b: &b_buf,
            out: &out_buf,
            dims: &dims_buf,
        },
        m,
        n,
    )
    .expect("dispatch");
    backend.submit(enc).await.expect("submit");

    let bytes = backend.read_buffer(out_id, 0, out_len).await.expect("read");
    let got: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    backend.free(a_buf.id);
    backend.free(b_buf.id);
    backend.free(dims_buf.id);
    backend.free(out_id);
    Some((got, expect))
}

/// f16-input matmul accumulates in f32; per-cell error is bounded by the
/// k-length sum of f16 rounding. A relative band scaled by sqrt(k) covers it
/// comfortably without being so loose it would miss a real bug (a transposed
/// load, wrong stride, or zeroed accumulator would blow this by orders).
fn assert_close(got: &[f32], expect: &[f32], k: u32) {
    assert_eq!(got.len(), expect.len());
    let tol = 2e-3 * (k as f32).sqrt();
    let mut worst = 0f32;
    for (i, (g, e)) in got.iter().zip(expect).enumerate() {
        let d = (g - e).abs();
        worst = worst.max(d);
        assert!(
            d <= tol + 1e-3 * e.abs(),
            "cell {i}: gpu={g} cpu={e} |d|={d} > tol={tol}"
        );
    }
    eprintln!("[coopmat] OK worst abs err={worst:.5} (tol={tol:.5})");
}

/// Throughput check: confirm the coopmat path actually engages the tensor
/// cores (target ~tens of TFLOPS, vs the ~3 TFLOPS scalar matmul floor). Not a
/// correctness test; gated behind `--ignored` so it only runs on request.
/// `cargo test ... coopmat_perf -- --ignored --nocapture`.
#[test]
#[ignore]
fn coopmat_perf() {
    pollster::block_on(async {
        let backend = discrete_backend().await;
        let Some(hw) = backend.coopmat() else {
            eprintln!("[coopmat] no support; SKIP");
            return;
        };
        let (sg, _) = backend.subgroup_size_range();
        let t = hw.tile;
        // Shapes: a square compute-bound case and two DiT-realistic ffn-ish
        // matmuls (tokens x dim from dff).
        let shapes: [(u32, u32, u32); 3] =
            [(4096, 4096, 4096), (2048, 5120, 14336), (2048, 14336, 5120)];
        // Register-tile configs to sweep: (tm, tn) accumulators per subgroup.
        let configs: [(u32, u32); 8] = [
            (1, 1), // naive single tile
            (2, 2),
            (4, 4),
            (2, 4),
            (4, 2),
            (4, 8),
            (8, 4),
            (8, 8),
        ];
        for (m, n, k) in shapes {
            let a = vec![0u8; (m * k * 2) as usize];
            let b = vec![0u8; (k * n * 2) as usize];
            let a_id = backend.allocate(a.len() as u64).unwrap();
            backend.write_buffer(a_id, 0, &a).unwrap();
            let b_id = backend.allocate(b.len() as u64).unwrap();
            backend.write_buffer(b_id, 0, &b).unwrap();
            let out_id = backend.allocate((m * n * 4) as u64).unwrap();
            let dims_id = backend.allocate(16).unwrap();
            backend
                .write_buffer(dims_id, 0, &pack_dims(m, n, k))
                .unwrap();
            let a_buf = BufRef::new(a_id, a.len() as u64);
            let b_buf = BufRef::new(b_id, b.len() as u64);
            let out_buf = BufRef::new(out_id, (m * n * 4) as u64);
            let dims_buf = BufRef::new(dims_id, 16);
            eprintln!("[coopmat-perf] shape {m}x{n}x{k}:");
            for (tm, tn) in configs {
                let mut cfg = CoopmatMatmulConfig::new(t, sg, CoopmatOut::F32);
                cfg.tm = tm;
                cfg.tn = tn;
                let wgsl = build_wgsl(&cfg);
                let pipeline = match backend
                    .create_pipeline("matmul_coopmat", &wgsl, "main", layout())
                    .await
                {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("    tm{tm}_tn{tn}: pipeline err {e:?}");
                        continue;
                    }
                };
                // Submit each iter separately so no single submit can approach
                // the 2s TDR watchdog regardless of config speed.
                let iters = 20u32;
                for _ in 0..3 {
                    let mut enc = backend.create_command_encoder();
                    dispatch_coopmat(
                        &backend,
                        &mut enc,
                        &pipeline,
                        &cfg,
                        &CoopmatBufs {
                            a: &a_buf,
                            b: &b_buf,
                            out: &out_buf,
                            dims: &dims_buf,
                        },
                        m,
                        n,
                    )
                    .unwrap();
                    backend.submit(enc).await.unwrap();
                }
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    let mut enc = backend.create_command_encoder();
                    dispatch_coopmat(
                        &backend,
                        &mut enc,
                        &pipeline,
                        &cfg,
                        &CoopmatBufs {
                            a: &a_buf,
                            b: &b_buf,
                            out: &out_buf,
                            dims: &dims_buf,
                        },
                        m,
                        n,
                    )
                    .unwrap();
                    backend.submit(enc).await.unwrap();
                }
                let secs = t0.elapsed().as_secs_f64();
                let flop = 2.0 * (m as f64) * (n as f64) * (k as f64) * (iters as f64);
                eprintln!(
                    "    tm{tm}_tn{tn}: {:.2} ms/iter, {:.1} GFLOP/s",
                    secs * 1e3 / iters as f64,
                    flop / secs / 1e9
                );
            }
            backend.free(a_id);
            backend.free(b_id);
            backend.free(out_id);
            backend.free(dims_id);
        }
    });
}

#[test]
fn coopmat_matmul_square() {
    let Some((got, expect)) = pollster::block_on(run_f32_out(64, 64, 64)) else {
        return;
    };
    assert_close(&got, &expect, 64);
}

#[test]
fn coopmat_matmul_oblong() {
    // Non-square M/N/K and a larger K to exercise the accumulate loop.
    let Some((got, expect)) = pollster::block_on(run_f32_out(48, 32, 256)) else {
        return;
    };
    assert_close(&got, &expect, 256);
}

#[test]
fn coopmat_matmul_ncol() {
    // n-major B (natural Linear weight layout) via the column-major load.
    let Some((got, expect)) = pollster::block_on(run_f32_out_layout(96, 64, 256, true)) else {
        return;
    };
    assert_close(&got, &expect, 256);
}
