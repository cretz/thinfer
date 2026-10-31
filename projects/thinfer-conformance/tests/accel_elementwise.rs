//! The accelerator's MiniMax-H3 block ops (`thinfer_cuda::elementwise`) vs CPU
//! references of the wgpu ops they replace, including the bf16 rounding
//! between RMSNorm and modulation / RoPE: RMSNorm + AdaLN over three row
//! segments, qkv split + per-head q/k RMSNorm + partial half-rot RoPE, the
//! gated residual add, and the fused SwiGLU. Skips without CUDA.

use thinfer_core::backend::{
    Backend, BufRef, ElementwiseOp, PowerPreference, RowSegments, WgpuBackend, WgpuConfig,
};

const ROWS: usize = 37;
const DIM: usize = 96;
const HEADS: usize = 3;
const HD: usize = 128;
const ROT_PAIRS: usize = 48;
const EPS: f32 = 1e-5;
/// Row ends of three segments (text | audio | video).
const ENDS: [usize; 3] = [5, 14, ROWS];

fn bf(v: f32) -> f32 {
    half::bf16::from_f32(v).to_f32()
}

struct Rng(u64);
impl Rng {
    fn vec(&mut self, n: usize, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                bf((((self.0 >> 33) as u32) as f32 / u32::MAX as f32 * 2.0 - 1.0) * amp)
            })
            .collect()
    }
}

fn seg_of(row: usize) -> usize {
    ENDS.iter().position(|&e| row < e).unwrap()
}

fn rmsnorm(x: &[f32], w: &[f32]) -> Vec<f32> {
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + EPS).sqrt();
    x.iter().zip(w).map(|(v, g)| bf(v * inv * g)).collect()
}

fn assert_close(what: &str, got: &[f32], want: &[f32]) {
    let (mut e2, mut r2) = (0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        e2 += ((g - w) as f64).powi(2);
        r2 += (*w as f64).powi(2);
    }
    let rel = (e2 / r2).sqrt();
    eprintln!("{what}: rel_rms {rel:.3e}");
    // One bf16 ulp at the store, plus rsqrt/exp ulps upstream of it.
    assert!(rel < 4e-3, "{what}: rel_rms {rel}");
}

#[test]
fn block_elementwise_ops_match_cpu_reference() {
    pollster::block_on(async {
        let backend = WgpuBackend::new_with_config(WgpuConfig {
            power_preference: PowerPreference::HighPerformance,
            timestamps: false,
            disable_coopmat: false,
        })
        .await
        .expect("wgpu adapter");
        if let Err(e) = thinfer_cuda::install(&backend) {
            eprintln!("SKIP: no CUDA ({e})");
            return;
        }
        let acc = backend.accelerator().expect("accelerator installed");
        assert!(
            acc.accepts_elementwise(),
            "elementwise kernels failed to build"
        );

        let upload = |x: &[f32]| -> BufRef {
            let b: Vec<u8> = x
                .iter()
                .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
                .collect();
            let id = backend.allocate(b.len() as u64).expect("allocate");
            backend.write_buffer(id, 0, &b).expect("write");
            BufRef::new(id, b.len() as u64)
        };
        let out =
            |n: usize| BufRef::new(backend.allocate(n as u64 * 2).expect("alloc"), n as u64 * 2);
        let addr = |b: &BufRef| backend.device_address(b).expect("cuda-addressable");
        let segs = |a: &[BufRef], b: Option<&[BufRef]>| {
            let mut s = RowSegments {
                n: 3,
                ..Default::default()
            };
            for i in 0..3 {
                s.end[i] = ENDS[i] as u32;
                s.a[i] = addr(&a[i]);
                if let Some(b) = b {
                    s.b[i] = addr(&b[i]);
                }
            }
            s
        };
        let mut rng = Rng(11);

        // Inputs.
        let x = rng.vec(ROWS * DIM, 3.0);
        let y = rng.vec(ROWS * DIM, 2.0);
        let w = rng.vec(DIM, 1.5);
        let (sc, sh, ga): (Vec<_>, Vec<_>, Vec<_>) = (0..3)
            .map(|_| (rng.vec(DIM, 0.5), rng.vec(DIM, 0.5), rng.vec(DIM, 0.5)))
            .fold((vec![], vec![], vec![]), |mut acc, (a, b, c)| {
                acc.0.push(a);
                acc.1.push(b);
                acc.2.push(c);
                acc
            });
        let inner = HEADS * HD;
        let qkv = rng.vec(ROWS * 3 * inner, 2.0);
        let (qw, kw) = (rng.vec(HD, 1.5), rng.vec(HD, 1.5));
        let freqs: Vec<f32> = (0..ROWS * ROT_PAIRS)
            .flat_map(|i| {
                let a = (i % 97) as f32 * 0.37;
                [bf(a.cos()), bf(a.sin())]
            })
            .collect();
        let ffn = 80;
        let fused = rng.vec(ROWS * 2 * ffn, 4.0);

        let (xb, yb, wb) = (upload(&x), upload(&y), upload(&w));
        let scb: Vec<_> = sc.iter().map(|v| upload(v)).collect();
        let shb: Vec<_> = sh.iter().map(|v| upload(v)).collect();
        let gab: Vec<_> = ga.iter().map(|v| upload(v)).collect();
        let (qkvb, qwb, kwb, fb, fub) = (
            upload(&qkv),
            upload(&qw),
            upload(&kw),
            upload(&freqs),
            upload(&fused),
        );
        let (nm, gadd, q, k, v, sw) = (
            out(ROWS * DIM),
            out(ROWS * DIM),
            out(ROWS * inner),
            out(ROWS * inner),
            out(ROWS * inner),
            out(ROWS * ffn),
        );
        // The q projection alone (q-only mode reads `[rows, inner]`).
        let q_part: Vec<f32> = (0..ROWS)
            .flat_map(|r| qkv[r * 3 * inner..r * 3 * inner + inner].to_vec())
            .collect();
        let (q_partb, q_only) = (upload(&q_part), out(ROWS * inner));
        backend
            .submit(backend.create_command_encoder())
            .await
            .expect("flush writes");
        backend.wait_idle();

        let ops = [
            ElementwiseOp::NormModulate {
                x: addr(&xb),
                w: addr(&wb),
                out: addr(&nm),
                rows: ROWS as u32,
                dim: DIM as u32,
                eps: EPS,
                seg: segs(&scb, Some(&shb)),
            },
            ElementwiseOp::GateAdd {
                x: addr(&xb),
                y: addr(&yb),
                out: addr(&gadd),
                rows: ROWS as u32,
                dim: DIM as u32,
                seg: segs(&gab, None),
            },
            ElementwiseOp::QkNormRope {
                headwise: false,
                q_only: false,
                qkv: addr(&qkvb),
                q_norm: addr(&qwb),
                k_norm: addr(&kwb),
                freqs: addr(&fb),
                q: addr(&q),
                k: addr(&k),
                v: addr(&v),
                rows: ROWS as u32,
                heads: HEADS as u32,
                head_dim: HD as u32,
                rot_pairs: ROT_PAIRS as u32,
                eps: EPS,
            },
            ElementwiseOp::QkNormRope {
                headwise: false,
                q_only: true,
                qkv: addr(&q_partb),
                q_norm: addr(&qwb),
                k_norm: 0,
                freqs: addr(&fb),
                q: addr(&q_only),
                k: 0,
                v: 0,
                rows: ROWS as u32,
                heads: HEADS as u32,
                head_dim: HD as u32,
                rot_pairs: ROT_PAIRS as u32,
                eps: EPS,
            },
            ElementwiseOp::SiluMulFused {
                x: addr(&fub),
                out: addr(&sw),
                rows: ROWS as u32,
                h: ffn as u32,
            },
        ];
        for op in &ops {
            assert!(acc.elementwise(op).expect("launch"), "declined {op:?}");
        }
        acc.sync().expect("sync");
        let backend = &backend;
        let read = |b: BufRef| async move {
            let raw = backend.read_buffer(b.id, 0, b.len).await.expect("read");
            raw.chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect::<Vec<f32>>()
        };

        // NormModulate.
        let want: Vec<f32> = (0..ROWS)
            .flat_map(|r| {
                let g = seg_of(r);
                let n = rmsnorm(&x[r * DIM..(r + 1) * DIM], &w);
                (0..DIM)
                    .map(|c| bf(n[c] * (sc[g][c] + 1.0) + sh[g][c]))
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_close("norm_modulate", &read(nm).await, &want);

        // GateAdd.
        let want: Vec<f32> = (0..ROWS * DIM)
            .map(|i| bf(x[i] + ga[seg_of(i / DIM)][i % DIM] * y[i]))
            .collect();
        assert_close("gate_add", &read(gadd).await, &want);

        // QkNormRope: q and k, then v as a plain copy.
        for (which, (got, gain)) in [(read(q).await, &qw), (read(k).await, &kw)]
            .into_iter()
            .enumerate()
        {
            let mut want = vec![0f32; ROWS * inner];
            for r in 0..ROWS {
                for h in 0..HEADS {
                    let src = r * 3 * inner + which * inner + h * HD;
                    let n = rmsnorm(&qkv[src..src + HD], gain);
                    for c in 0..HD {
                        let o = if c < 2 * ROT_PAIRS {
                            let kk = c % ROT_PAIRS;
                            let (cr, ci) = (
                                freqs[(r * ROT_PAIRS + kk) * 2],
                                freqs[(r * ROT_PAIRS + kk) * 2 + 1],
                            );
                            if c < ROT_PAIRS {
                                n[c] * cr - n[c + ROT_PAIRS] * ci
                            } else {
                                n[c - ROT_PAIRS] * ci + n[c] * cr
                            }
                        } else {
                            n[c]
                        };
                        want[r * inner + h * HD + c] = bf(o);
                    }
                }
            }
            assert_close(
                if which == 0 {
                    "q norm+rope"
                } else {
                    "k norm+rope"
                },
                &got,
                &want,
            );
        }
        let want: Vec<f32> = (0..ROWS)
            .flat_map(|r| qkv[r * 3 * inner + 2 * inner..r * 3 * inner + 3 * inner].to_vec())
            .collect();
        assert_eq!(read(v).await, want, "v must be an exact copy");
        assert_eq!(
            read(q_only).await,
            read(q).await,
            "q-only norm+rope must match the fused qkv's q bit for bit"
        );

        // SiluMulFused.
        let want: Vec<f32> = (0..ROWS * ffn)
            .map(|i| {
                let (r, c) = (i / ffn, i % ffn);
                let g = fused[r * 2 * ffn + c];
                bf(g / (1.0 + (-g).exp()) * fused[r * 2 * ffn + ffn + c])
            })
            .collect();
        assert_close("silu_mul_fused", &read(sw).await, &want);
    });
}
