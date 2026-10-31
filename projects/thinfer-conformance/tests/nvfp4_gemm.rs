//! Native NVFP4 W4A4 matmul (`thinfer_cuda::gemm_fp4`) vs a CPU reference.
//!
//! Two checks per shape: (1) the activation quantizer's codes and E4M3 block
//! scales against a CPU port of FlashInfer's `nvfp4_quantize` arithmetic, and
//! (2) the block-scaled MMA against the exact W4A4 product computed on the CPU
//! from the GPU's OWN quantized activations, so the GEMM (fragment layouts,
//! scale swizzle, per-group alpha, input scale) is checked independently of
//! quantizer rounding. Skips when no CUDA sm_120 device is present.
#![cfg(not(target_os = "macos"))]

use thinfer_core::backend::{
    Backend, BufRef, GemmNvfp4Addrs, Nvfp4Format, Nvfp4GateAdd, Nvfp4Rows, PowerPreference,
    RowSegments, WgpuBackend, WgpuConfig,
};
use thinfer_core::ops::nvfp4::{self, E2M1, Nvfp4Layout};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        self.next() as f32 / u32::MAX as f32 * 2.0 - 1.0
    }
}

/// Composite with random codes, POSITIVE E4M3 scales (the MMA reads them as
/// ue4m3), a distinct alpha per 128-row group and a shared input scale.
fn composite(l: Nvfp4Layout, input_sf: f32, rng: &mut Rng) -> Vec<u8> {
    let mut bytes = vec![0u8; l.total_len()];
    for b in &mut bytes[..l.packed_len()] {
        *b = rng.next() as u8;
    }
    let bp = l.blocks_padded();
    for r in 0..l.n {
        for blk in 0..l.k / nvfp4::BLOCK {
            let byte = ((4 + rng.next() % 9) << 3) | (rng.next() & 7);
            bytes[l.scales_offset() + nvfp4::scale_offset(r, blk, bp)] = byte as u8;
        }
    }
    for g in 0..l.groups() {
        let a = l.alpha_offset() + 4 * g;
        bytes[a..a + 4].copy_from_slice(&(0.0011f32 * (g + 1) as f32).to_le_bytes());
        let s = l.input_sf_offset() + 4 * g;
        bytes[s..s + 4].copy_from_slice(&input_sf.to_le_bytes());
    }
    bytes
}

/// Nearest finite E4M3 (ties to the even code), saturating at 448.
fn to_e4m3(v: f32) -> u8 {
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for code in 0u8..0x7f {
        let err = (nvfp4::e4m3(code) - v).abs();
        if err < best_err || (err == best_err && code % 2 == 0) {
            best = code;
            best_err = err;
        }
    }
    best
}

fn to_e2m1(v: f32) -> u8 {
    let a = v.abs();
    let c = if a <= 0.25 {
        0
    } else if a < 0.75 {
        1
    } else if a <= 1.25 {
        2
    } else if a < 1.75 {
        3
    } else if a <= 2.5 {
        4
    } else if a < 3.5 {
        5
    } else if a <= 5.0 {
        6
    } else {
        7
    };
    c | if v.is_sign_negative() { 8 } else { 0 }
}

/// FlashInfer `nvfp4_quantize` on the CPU: (codes `[M,K/2]`, scales `[M,K/16]`).
fn quantize(x: &[f32], m: usize, k: usize, gsf: f32) -> (Vec<u8>, Vec<u8>) {
    let mut q = vec![0u8; m * k / 2];
    let mut s = vec![0u8; m * k / 16];
    for r in 0..m {
        for b in 0..k / 16 {
            let blk = &x[r * k + b * 16..r * k + b * 16 + 16];
            let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
            let s8 = to_e4m3(gsf * (amax * (1.0 / 6.0)));
            let sq = nvfp4::e4m3(s8);
            let scale = if sq != 0.0 {
                1.0 / (sq * (1.0 / gsf))
            } else {
                0.0
            };
            for (j, v) in blk.iter().enumerate() {
                let c = to_e2m1(v * scale);
                q[r * k / 2 + b * 8 + j / 2] |= c << (4 * (j % 2));
            }
            s[r * k / 16 + b] = s8;
        }
    }
    (q, s)
}

fn code(packed: &[u8], i: usize) -> f32 {
    E2M1[((packed[i / 2] >> (4 * (i % 2))) & 0xf) as usize]
}

fn bf16_bits(v: f32) -> u16 {
    half::bf16::from_f32(v).to_bits()
}

#[test]
fn nvfp4_w4a4_gemm_matches_cpu_reference() {
    pollster::block_on(async {
        // The discrete adapter: the CUDA device must be the one wgpu drives.
        let Ok(backend) = WgpuBackend::new_with_config(WgpuConfig {
            power_preference: PowerPreference::HighPerformance,
            timestamps: false,
            disable_coopmat: false,
        })
        .await
        else {
            eprintln!("SKIP: no wgpu adapter");
            return;
        };
        match thinfer_cuda::install(&backend) {
            Ok((name, sm)) => eprintln!("cuda: {name} sm_{}{}", sm.0, sm.1),
            Err(e) => {
                eprintln!("SKIP: no CUDA ({e})");
                return;
            }
        }
        let acc = backend.accelerator().expect("accelerator installed");
        // 1100 rows: 9 row tiles, past one 8-row-tile raster group.
        for (m, n, k) in [
            (200usize, 256usize, 256usize),
            (300, 384, 5376),
            (1100, 512, 256),
        ] {
            if acc
                .accepts_gemm_nvfp4(m as u32, n as u32, k as u32)
                .is_none()
            {
                eprintln!("SKIP: device declines nvfp4 gemm");
                return;
            }
            let mut rng = Rng((m * 7 + n * 13 + k) as u64);
            let gsf = 90.0f32;
            let l = Nvfp4Layout::new(n, k);
            let w = composite(l, gsf, &mut rng);
            // Activations with a few outlier channels, like the real stream.
            let x: Vec<f32> = (0..m * k)
                .map(|i| {
                    let v = rng.unit() * 4.0;
                    if i % k % 97 == 0 { v * 8.0 } else { v }
                })
                .map(|v| half::bf16::from_f32(v).to_f32())
                .collect();
            let xb: Vec<u8> = x.iter().flat_map(|v| bf16_bits(*v).to_le_bytes()).collect();

            let upload = |b: &[u8]| -> BufRef {
                let id = backend.allocate(b.len() as u64).expect("allocate");
                backend.write_buffer(id, 0, b).expect("write");
                BufRef::new(id, b.len() as u64)
            };
            let a_buf = upload(&xb);
            let w_buf = upload(&w);
            let out_len = (m * n * 2) as u64;
            let out_buf = BufRef::new(backend.allocate(out_len).expect("out"), out_len);
            let scratch_len = acc
                .accepts_gemm_nvfp4(m as u32, n as u32, k as u32)
                .unwrap();
            let scratch = BufRef::new(backend.allocate(scratch_len).expect("scratch"), scratch_len);
            backend
                .submit(backend.create_command_encoder())
                .await
                .expect("flush writes");
            backend.wait_idle();
            let addr = |b: &BufRef| backend.device_address(b).expect("cuda-addressable");
            let full = GemmNvfp4Addrs {
                a: addr(&a_buf),
                w: addr(&w_buf),
                out: addr(&out_buf),
                scratch: addr(&scratch),
                m: m as u32,
                n: n as u32,
                k: k as u32,
                lora: Default::default(),
                format: Default::default(),
                rows: None,
                gate_add: None,
            };
            let ran = acc.gemm_nvfp4(&full).expect("gemm");
            assert!(ran, "gemm declined {m}x{n}x{k}");
            acc.sync().expect("sync");

            // (1b) Weight rows [128, 256) alone = those output columns of the
            // full product, bit for bit (a fused qkv's q-only GEMM).
            if n >= 256 {
                let sub_len = (m * 128 * 2) as u64;
                let sub = BufRef::new(backend.allocate(sub_len).expect("sub"), sub_len);
                let ran = acc
                    .gemm_nvfp4(&GemmNvfp4Addrs {
                        out: addr(&sub),
                        n: 128,
                        rows: Some(Nvfp4Rows {
                            base: 128,
                            total: n as u32,
                        }),
                        ..full
                    })
                    .expect("gemm rows");
                assert!(ran, "row-range gemm declined");
                acc.sync().expect("sync");
                let whole = backend
                    .read_buffer(out_buf.id, 0, out_len)
                    .await
                    .expect("read");
                let part = backend.read_buffer(sub.id, 0, sub_len).await.expect("read");
                for r in 0..m {
                    let at = (r * n + 128) * 2;
                    assert_eq!(
                        &whole[at..at + 256],
                        &part[r * 256..(r + 1) * 256],
                        "{m}x{n}x{k}: row-range gemm differs at output row {r}"
                    );
                }
                eprintln!("{m}x{n}x{k}: row range [128, 256) bit-identical");
            }

            // (1c) The gated-residual epilogue = the separate gate-add pass
            // over the plain product (`rn(r + g * p)`, contracted to one FMA
            // as the kernels compile it), two row segments.
            {
                let bf_bytes = |v: &[f32]| -> Vec<u8> {
                    v.iter().flat_map(|v| bf16_bits(*v).to_le_bytes()).collect()
                };
                let resid: Vec<f32> = (0..m * n).map(|_| rng.unit() * 3.0).collect();
                let gates: Vec<Vec<f32>> = (0..2)
                    .map(|_| (0..n).map(|_| rng.unit()).collect())
                    .collect();
                let r_buf = upload(&bf_bytes(&resid));
                let g_bufs: Vec<BufRef> = gates.iter().map(|g| upload(&bf_bytes(g))).collect();
                let gated = BufRef::new(backend.allocate(out_len).expect("gated"), out_len);
                backend
                    .submit(backend.create_command_encoder())
                    .await
                    .expect("flush writes");
                backend.wait_idle();
                let split = (m / 2) as u32;
                let mut seg = RowSegments {
                    n: 2,
                    ..Default::default()
                };
                seg.end[..2].copy_from_slice(&[split, m as u32]);
                seg.a[0] = addr(&g_bufs[0]);
                seg.a[1] = addr(&g_bufs[1]);
                let ran = acc
                    .gemm_nvfp4(&GemmNvfp4Addrs {
                        out: addr(&gated),
                        gate_add: Some(Nvfp4GateAdd {
                            residual: addr(&r_buf),
                            seg,
                        }),
                        ..full
                    })
                    .expect("gemm gate add");
                assert!(ran, "gate-add gemm declined");
                acc.sync().expect("sync");
                let plain = backend
                    .read_buffer(out_buf.id, 0, out_len)
                    .await
                    .expect("read");
                let got = backend
                    .read_buffer(gated.id, 0, out_len)
                    .await
                    .expect("read");
                let bf = |b: &[u8], i: usize| {
                    half::bf16::from_bits(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])).to_f32()
                };
                for i in 0..m * n {
                    let (r, c) = (i / n, i % n);
                    let g = gates[(r as u32 >= split) as usize][c];
                    let rb = half::bf16::from_f32(resid[i]).to_f32();
                    let gb = half::bf16::from_f32(g).to_f32();
                    let want = bf16_bits(gb.mul_add(bf(&plain, i), rb));
                    let have = u16::from_le_bytes([got[2 * i], got[2 * i + 1]]);
                    assert_eq!(have, want, "{m}x{n}x{k}: gate add differs at ({r}, {c})");
                }
                eprintln!("{m}x{n}x{k}: gate-add epilogue bit-identical");
            }

            // (1) quantizer vs CPU.
            let raw = backend
                .read_buffer(scratch.id, 0, scratch_len)
                .await
                .expect("read scratch");
            let (gq, gs) = raw.split_at(m * k / 2);
            let (cq, cs) = quantize(&x, m, k, gsf);
            let code_diff = (0..m * k).filter(|&i| code(gq, i) != code(&cq, i)).count();
            let scale_diff = gs.iter().zip(&cs).filter(|(a, b)| a != b).count();
            eprintln!(
                "{m}x{n}x{k}: quant code diffs {code_diff}/{} scale diffs {scale_diff}/{}",
                m * k,
                cs.len()
            );
            assert!(scale_diff == 0, "E4M3 block scales disagree");
            assert!(
                code_diff * 1000 <= m * k,
                "E2M1 codes disagree beyond rounding ties"
            );

            // (2) GEMM vs the exact W4A4 product of the GPU's own codes.
            let bp = l.blocks_padded();
            let alpha = |g: usize| {
                let o = l.alpha_offset() + 4 * g;
                f32::from_le_bytes(w[o..o + 4].try_into().unwrap())
            };
            let got = backend
                .read_buffer(out_buf.id, 0, out_len)
                .await
                .expect("read out");
            let mut err2 = 0f64;
            let mut ref2 = 0f64;
            let mut worst = 0f64;
            let mut wants = vec![0f64; m * n];
            for r in 0..m {
                for c in 0..n {
                    let mut sum = 0f64;
                    for kk in 0..k {
                        let sa = nvfp4::e4m3(gs[r * k / 16 + kk / 16]) as f64;
                        let sw =
                            nvfp4::e4m3(w[l.scales_offset() + nvfp4::scale_offset(c, kk / 16, bp)])
                                as f64;
                        sum += code(gq, r * k + kk) as f64
                            * sa
                            * code(&w[..l.packed_len()], c * k + kk) as f64
                            * sw;
                    }
                    let want = sum * alpha(c / 128) as f64 / gsf as f64;
                    wants[r * n + c] = want;
                    let o = (r * n + c) * 2;
                    let g = half::bf16::from_bits(u16::from_le_bytes([got[o], got[o + 1]])).to_f32()
                        as f64;
                    err2 += (g - want).powi(2);
                    ref2 += want.powi(2);
                    worst = worst.max((g - want).abs() / want.abs().max(1e-3));
                }
            }
            let rel = (err2 / ref2).sqrt();
            eprintln!("{m}x{n}x{k}: gemm rel_rms {rel:.3e} worst_rel {worst:.3e}");
            // bf16 output rounding (2^-9) bounds the rms error; f32 accumulate
            // order adds far less.
            assert!(rel < 4e-3, "gemm rel_rms {rel}");

            // (2b) Comfy format: the same weights packed high-nibble-first and
            // a dynamic activation scale (`1 / bf16(amax / 2688)`), against the
            // exact product of the GPU's own codes at the scale it computed.
            let w_hi: Vec<u8> = w
                .iter()
                .enumerate()
                .map(|(i, &b)| {
                    if i < l.packed_len() {
                        b.rotate_left(4)
                    } else {
                        b
                    }
                })
                .collect();
            let w_hi_buf = upload(&w_hi);
            let out_hi = BufRef::new(backend.allocate(out_len).expect("out"), out_len);
            backend
                .submit(backend.create_command_encoder())
                .await
                .expect("flush writes");
            backend.wait_idle();
            let ran = acc
                .gemm_nvfp4(&GemmNvfp4Addrs {
                    a: addr(&a_buf),
                    w: addr(&w_hi_buf),
                    out: addr(&out_hi),
                    scratch: addr(&scratch),
                    m: m as u32,
                    n: n as u32,
                    k: k as u32,
                    lora: Default::default(),
                    format: Nvfp4Format {
                        dynamic_act_scale: true,
                        hi_first: true,
                    },
                    rows: None,
                    gate_add: None,
                })
                .expect("gemm comfy");
            assert!(ran, "comfy gemm declined {m}x{n}x{k}");
            acc.sync().expect("sync");
            let raw = backend
                .read_buffer(scratch.id, 0, scratch_len)
                .await
                .expect("read scratch");
            let slot = (m * k / 2 + m * k / 16).next_multiple_of(16);
            let dyn_gsf = f32::from_le_bytes(raw[slot + 4..slot + 8].try_into().unwrap());
            let amax = x.iter().fold(0f32, |a, v| a.max(v.abs()));
            let want_gsf = 1.0 / half::bf16::from_f32(amax / 2688.0).to_f32();
            assert_eq!(dyn_gsf, want_gsf, "dynamic activation scale");
            let (hq, hs) = raw.split_at(m * k / 2);
            let hi = |packed: &[u8], i: usize| code(packed, i ^ 1);
            let got = backend
                .read_buffer(out_hi.id, 0, out_len)
                .await
                .expect("read out");
            let (mut err2, mut ref2) = (0f64, 0f64);
            let mut comfy_wants = Vec::with_capacity(m * n);
            for r in 0..m {
                for c in 0..n {
                    let mut sum = 0f64;
                    for kk in 0..k {
                        let sa = nvfp4::e4m3(hs[r * k / 16 + kk / 16]) as f64;
                        let sw =
                            nvfp4::e4m3(w[l.scales_offset() + nvfp4::scale_offset(c, kk / 16, bp)])
                                as f64;
                        sum += hi(hq, r * k + kk) as f64
                            * sa
                            * hi(&w_hi[..l.packed_len()], c * k + kk) as f64
                            * sw;
                    }
                    let want = sum * alpha(c / 128) as f64 / dyn_gsf as f64;
                    comfy_wants.push(want);
                    let o = (r * n + c) * 2;
                    let g = half::bf16::from_bits(u16::from_le_bytes([got[o], got[o + 1]])).to_f32()
                        as f64;
                    err2 += (g - want).powi(2);
                    ref2 += want.powi(2);
                }
            }
            let rel = (err2 / ref2).sqrt();
            eprintln!("{m}x{n}x{k}: comfy gemm rel_rms {rel:.3e}");
            assert!(rel < 4e-3, "comfy gemm rel_rms {rel}");
            // The hi-first codes are the low-first ones nibble-swapped, so the
            // weights decode the same: the product must also track run (2).
            let (mut d2, mut b2) = (0f64, 0f64);
            for (i, want) in wants.iter().enumerate() {
                let g = half::bf16::from_bits(u16::from_le_bytes([got[2 * i], got[2 * i + 1]]))
                    .to_f32() as f64;
                d2 += (g - want).powi(2);
                b2 += want.powi(2);
            }
            let drift = (d2 / b2).sqrt();
            eprintln!("{m}x{n}x{k}: comfy vs static-scale product {drift:.3e}");
            assert!(drift < 0.1, "comfy weights decode differently: {drift}");

            // (3) The fused low-rank epilogue: out = W4A4 + mid @ bt (a
            // rank-16 adapter), against the same product plus the CPU term.
            let rank = 16usize;
            let bf = |v: f32| half::bf16::from_f32(v).to_f32();
            let mid: Vec<f32> = (0..m * rank).map(|_| bf(rng.unit() * 2.0)).collect();
            let bt: Vec<f32> = (0..rank * n).map(|_| bf(rng.unit())).collect();
            let to_bytes = |v: &[f32]| -> Vec<u8> {
                v.iter().flat_map(|x| bf16_bits(*x).to_le_bytes()).collect()
            };
            let (mid_buf, bt_buf) = (upload(&to_bytes(&mid)), upload(&to_bytes(&bt)));
            let out2 = BufRef::new(backend.allocate(out_len).expect("out2"), out_len);
            backend
                .submit(backend.create_command_encoder())
                .await
                .expect("flush writes");
            backend.wait_idle();
            let ran = acc
                .gemm_nvfp4(&GemmNvfp4Addrs {
                    a: addr(&a_buf),
                    w: addr(&w_buf),
                    out: addr(&out2),
                    scratch: addr(&scratch),
                    m: m as u32,
                    n: n as u32,
                    k: k as u32,
                    lora: thinfer_core::backend::Nvfp4Lora {
                        mid: addr(&mid_buf),
                        bt: addr(&bt_buf),
                        rank: rank as u32,
                    },
                    format: Default::default(),
                    rows: None,
                    gate_add: None,
                })
                .expect("gemm lora");
            assert!(ran, "fused-lora gemm declined {m}x{n}x{k}");
            acc.sync().expect("sync");
            let got2 = backend
                .read_buffer(out2.id, 0, out_len)
                .await
                .expect("read out2");
            let (mut e2, mut r2) = (0f64, 0f64);
            for r in 0..m {
                for c in 0..n {
                    let term: f64 = (0..rank)
                        .map(|j| mid[r * rank + j] as f64 * bt[j * n + c] as f64)
                        .sum();
                    let want = wants[r * n + c] + term;
                    let o = (r * n + c) * 2;
                    let g = half::bf16::from_bits(u16::from_le_bytes([got2[o], got2[o + 1]]))
                        .to_f32() as f64;
                    e2 += (g - want).powi(2);
                    r2 += want.powi(2);
                }
            }
            let rel2 = (e2 / r2).sqrt();
            eprintln!("{m}x{n}x{k}: fused lora rel_rms {rel2:.3e}");
            assert!(rel2 < 4e-3, "fused lora rel_rms {rel2}");

            // (3b) The same adapter on the Comfy format (FastH3-Omni): the
            // dynamic scale must not touch the bf16 low-rank seed.
            let out3 = BufRef::new(backend.allocate(out_len).expect("out3"), out_len);
            backend
                .submit(backend.create_command_encoder())
                .await
                .expect("flush writes");
            backend.wait_idle();
            let ran = acc
                .gemm_nvfp4(&GemmNvfp4Addrs {
                    a: addr(&a_buf),
                    w: addr(&w_hi_buf),
                    out: addr(&out3),
                    scratch: addr(&scratch),
                    m: m as u32,
                    n: n as u32,
                    k: k as u32,
                    lora: thinfer_core::backend::Nvfp4Lora {
                        mid: addr(&mid_buf),
                        bt: addr(&bt_buf),
                        rank: rank as u32,
                    },
                    format: Nvfp4Format {
                        dynamic_act_scale: true,
                        hi_first: true,
                    },
                    rows: None,
                    gate_add: None,
                })
                .expect("gemm comfy lora");
            assert!(ran, "comfy fused-lora gemm declined {m}x{n}x{k}");
            acc.sync().expect("sync");
            let got3 = backend
                .read_buffer(out3.id, 0, out_len)
                .await
                .expect("read out3");
            let (mut e3, mut r3) = (0f64, 0f64);
            for r in 0..m {
                for c in 0..n {
                    let term: f64 = (0..rank)
                        .map(|j| mid[r * rank + j] as f64 * bt[j * n + c] as f64)
                        .sum();
                    let want = comfy_wants[r * n + c] + term;
                    let o = (r * n + c) * 2;
                    let g = half::bf16::from_bits(u16::from_le_bytes([got3[o], got3[o + 1]]))
                        .to_f32() as f64;
                    e3 += (g - want).powi(2);
                    r3 += want.powi(2);
                }
            }
            let rel3 = (e3 / r3).sqrt();
            eprintln!("{m}x{n}x{k}: comfy fused lora rel_rms {rel3:.3e}");
            assert!(rel3 < 4e-3, "comfy fused lora rel_rms {rel3}");
        }
    });
}
