//! MiniMax-H3 video sparse attention (`thinfer_cuda::vsa`) vs a CPU port of
//! the reference definition (FastVideo `minimax_h3_vsa.h3_vsa_attention`):
//! tile means, pooled scores, the per-video-query top-k key tiles (prefix
//! tiles always kept, prefix queries dense) and the gated compression branch.
//! Ragged prefix and video tiles, with and without the gate, sparse and
//! dense. Skips when no CUDA device is present.

use thinfer_core::backend::{
    Backend, BufRef, PowerPreference, VsaAddrs, VsaShape, WgpuBackend, WgpuConfig,
};
use thinfer_models::minimax::vsa::{TILE, VsaGeometry};

const H: usize = 4;
const D: usize = 128;

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as u32) as f32 / u32::MAX as f32 * 2.0 - 1.0
    }
}

fn bf(v: f32) -> f32 {
    half::bf16::from_f32(v).to_f32()
}

/// `[rows, H, D]` bf16-rounded values with a per-tile bias so pooled scores
/// are well separated (no near-ties in the top-k).
fn tensor(rows: usize, g: &VsaGeometry, rng: &mut Rng, amp: f32) -> Vec<f32> {
    let mut x: Vec<f32> = (0..rows * H * D).map(|_| rng.unit() * amp).collect();
    for (t, &n) in g.tile_sizes.iter().enumerate() {
        let bias: Vec<f32> = (0..H * D).map(|_| rng.unit() * amp).collect();
        for &r in &g.tile_rows[t * TILE..t * TILE + n as usize] {
            for (c, b) in bias.iter().enumerate() {
                x[r as usize * H * D + c] += b;
            }
        }
    }
    x.into_iter().map(bf).collect()
}

/// The reference VSA forward, f64 accumulate.
#[allow(clippy::too_many_arguments)]
fn reference(
    g: &VsaGeometry,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: Option<&[f32]>,
    rows: usize,
    topk: usize,
    scale: f64,
) -> Vec<f32> {
    let t_n = g.n_tiles();
    let p = g.n_prefix;
    let rows_of = |t: usize| &g.tile_rows[t * TILE..t * TILE + g.tile_sizes[t] as usize];
    let at = |x: &[f32], r: u32, h: usize, d: usize| x[(r as usize * H + h) * D + d] as f64;
    let mut out = vec![0f32; rows * H * D];
    for h in 0..H {
        let pool = |x: &[f32]| -> Vec<Vec<f64>> {
            (0..t_n)
                .map(|t| {
                    let rs = rows_of(t);
                    (0..D)
                        .map(|d| rs.iter().map(|&r| at(x, r, h, d)).sum::<f64>() / rs.len() as f64)
                        .collect()
                })
                .collect()
        };
        let (qp, kp, vp) = (pool(q), pool(k), pool(v));
        for i in 0..t_n {
            let s: Vec<f64> = (0..t_n)
                .map(|j| (0..D).map(|d| qp[i][d] * kp[j][d]).sum::<f64>() * scale)
                .collect();
            let mx = s.iter().cloned().fold(f64::MIN, f64::max);
            let e: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
            let sum: f64 = e.iter().sum();
            let coarse: Vec<f64> = (0..D)
                .map(|d| (0..t_n).map(|j| e[j] * vp[j][d]).sum::<f64>() / sum)
                .collect();
            let keys: Vec<usize> = if i < p || topk >= t_n - p {
                (0..t_n).collect()
            } else {
                let mut vid: Vec<usize> = (p..t_n).collect();
                vid.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap());
                (0..p).chain(vid.into_iter().take(topk)).collect()
            };
            let key_rows: Vec<u32> = keys.iter().flat_map(|&j| rows_of(j).to_vec()).collect();
            for &r in rows_of(i) {
                let sc: Vec<f64> = key_rows
                    .iter()
                    .map(|&kr| {
                        (0..D)
                            .map(|d| at(q, r, h, d) * at(k, kr, h, d))
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let m = sc.iter().cloned().fold(f64::MIN, f64::max);
                let w: Vec<f64> = sc.iter().map(|x| (x - m).exp()).collect();
                let z: f64 = w.iter().sum();
                for d in 0..D {
                    let mut o = key_rows
                        .iter()
                        .zip(&w)
                        .map(|(&kr, wi)| wi * at(v, kr, h, d))
                        .sum::<f64>()
                        / z;
                    if let Some(gt) = gate {
                        o += at(gt, r, h, d) * coarse[d];
                    }
                    out[(r as usize * H + h) * D + d] = o as f32;
                }
            }
        }
    }
    out
}

#[test]
fn vsa_matches_cpu_reference() {
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

        // 40 text + 70 audio rows (tiles 40 | 64, 6), then a 6x5x9 grid whose
        // cubes are ragged on every axis: 12 video tiles.
        let g = VsaGeometry::new(&[(0, 40), (40, 70)], 110, [6, 5, 9]);
        let rows = 110 + 6 * 5 * 9;
        assert_eq!((g.n_prefix, g.n_video()), (3, 12));
        let mut rng = Rng(7);
        let q = tensor(rows, &g, &mut rng, 1.0);
        let k = tensor(rows, &g, &mut rng, 1.0);
        let v = tensor(rows, &g, &mut rng, 4.0);
        let gate = tensor(rows, &g, &mut rng, 0.5);
        let scale = 1.0 / (D as f32).sqrt();

        let upload = |x: &[u8]| -> BufRef {
            let id = backend.allocate(x.len() as u64).expect("allocate");
            backend.write_buffer(id, 0, x).expect("write");
            BufRef::new(id, x.len() as u64)
        };
        let to_bytes = |x: &[f32]| -> Vec<u8> {
            x.iter()
                .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
                .collect()
        };
        let (qb, kb, vb, gb) = (
            upload(&to_bytes(&q)),
            upload(&to_bytes(&k)),
            upload(&to_bytes(&v)),
            upload(&to_bytes(&gate)),
        );
        let geom = upload(&g.bytes());
        let out_len = (rows * H * D * 2) as u64;
        let out = BufRef::new(backend.allocate(out_len).expect("out"), out_len);

        for (topk, with_gate) in [
            (g.topk(0.8), true),
            (g.topk(0.8), false),
            (g.n_video(), true),
        ] {
            let shape = VsaShape {
                rows: rows as u32,
                heads: H as u32,
                head_dim: D as u32,
                n_tiles: g.n_tiles() as u32,
                n_prefix: g.n_prefix as u32,
                topk: topk as u32,
                scale,
            };
            let Some(scratch_len) = acc.accepts_vsa(&shape) else {
                // Surface why: a kernel compile failure is otherwise only a log.
                let dev = thinfer_cuda::probe(backend.wgpu_device()).expect("probe");
                let why = thinfer_cuda::vsa::VsaKernels::new(dev.ctx.clone(), dev.sm).err();
                panic!("vsa declined {shape:?}: {why:?}");
            };
            let scratch = BufRef::new(backend.allocate(scratch_len).expect("scratch"), scratch_len);
            backend
                .submit(backend.create_command_encoder())
                .await
                .expect("flush writes");
            backend.wait_idle();
            let addr = |b: &BufRef| backend.device_address(b).expect("cuda-addressable");
            let ran = acc
                .vsa(
                    &shape,
                    &VsaAddrs {
                        q: addr(&qb),
                        k: addr(&kb),
                        v: addr(&vb),
                        gate: if with_gate { addr(&gb) } else { 0 },
                        out: addr(&out),
                        geometry: addr(&geom),
                        scratch: addr(&scratch),
                    },
                )
                .expect("vsa");
            assert!(ran, "vsa declined");
            acc.sync().expect("sync");
            let got = backend.read_buffer(out.id, 0, out_len).await.expect("read");
            let want = reference(
                &g,
                &q,
                &k,
                &v,
                with_gate.then_some(gate.as_slice()),
                rows,
                topk,
                scale as f64,
            );
            let (mut e2, mut r2) = (0f64, 0f64);
            for (i, w) in want.iter().enumerate() {
                let gv = half::bf16::from_bits(u16::from_le_bytes([got[2 * i], got[2 * i + 1]]))
                    .to_f32();
                e2 += ((gv - w) as f64).powi(2);
                r2 += (*w as f64).powi(2);
            }
            let rel = (e2 / r2).sqrt();
            eprintln!(
                "topk {topk}/{} gate {with_gate}: rel_rms {rel:.3e}",
                g.n_video()
            );
            // f16 probabilities + bf16 output rounding.
            assert!(rel < 5e-3, "vsa rel_rms {rel}");
        }
    });
}
