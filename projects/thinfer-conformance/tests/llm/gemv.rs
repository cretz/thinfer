//! `ops::llm::gemv_wgsl` vs a CPU reference (`quant::dequantize_row` dot x)
//! for every weight format, direct and indirect (expert-slot) modes, at
//! reduction lengths whose quant rows are not 4-byte aligned (Q6_K / Q8_0 at
//! K = 512 give 420 / 544-byte rows).

#![cfg(feature = "llm-e2e")]

use thinfer_core::Backend;
use thinfer_core::arbiter::{MemArbiter, MemTier};
use thinfer_core::backend::BufRef;
use thinfer_core::ops::llm::LlmWeight;
use thinfer_core::quant::{QuantKind, dequantize_row};
use thinfer_core::workspace::Workspace;
use thinfer_models::llm::kernels::{Act, LlmPipelines, Mat, act_bytes};

use crate::util::{backend, compare};

/// Deterministic xorshift.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// Random but well-formed rows: random payload bytes, small finite f16 scales.
fn random_rows(fmt: LlmWeight, n: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let rb = fmt.row_bytes(k as u64) as usize;
    let mut v: Vec<u8> = (0..n * rb).map(|_| rng.next() as u8).collect();
    let f16 = |x: f32| half::f16::from_f32(x).to_le_bytes();
    match fmt {
        LlmWeight::F32 => {
            for c in v.chunks_mut(4) {
                c.copy_from_slice(&rng.f().to_le_bytes());
            }
        }
        LlmWeight::Quant(q) => {
            let bb = q.bytes_per_block() as usize;
            for blk in v.chunks_mut(bb) {
                let d = f16(0.01 + 0.01 * rng.f().abs());
                match q {
                    QuantKind::Q8_0 | QuantKind::Q4_0 | QuantKind::Q5_0 => {
                        blk[..2].copy_from_slice(&d)
                    }
                    QuantKind::Q4_K | QuantKind::Q5_K | QuantKind::Q5_1 => {
                        blk[..2].copy_from_slice(&d);
                        blk[2..4].copy_from_slice(&f16(0.005 * rng.f().abs()));
                    }
                    QuantKind::Q6_K => blk[208..210].copy_from_slice(&d),
                }
            }
        }
    }
    v
}

fn dequant(fmt: LlmWeight, row: &[u8], k: usize) -> Vec<f32> {
    match fmt {
        LlmWeight::F32 => bytemuck::cast_slice::<u8, f32>(row).to_vec(),
        LlmWeight::Quant(q) => {
            let mut out = vec![0f32; k];
            dequantize_row(q, row, &mut out);
            out
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn gemv_formats() {
    let b = backend().await;
    let pipes = LlmPipelines::compile(&b).await.expect("pipelines");
    let ws = Workspace::new(b.clone(), MemArbiter::new(MemTier::Vram, u64::MAX));
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let mut fails = Vec::new();
    for fmt in thinfer_models::llm::kernels::FORMATS {
        // k = 704 is not a multiple of 256: the partial-chunk tail (32-block
        // formats only; K-quant rows are whole 256 super-blocks).
        let block = match fmt {
            LlmWeight::Quant(q) => q.block_size() as usize,
            LlmWeight::F32 => 32,
        };
        for (n, k) in [(37usize, 512usize), (64, 2048), (19, 704)] {
            if k % block != 0 {
                continue;
            }
            let (rows, slots) = (3usize, 4usize);
            let rb = fmt.row_bytes(k as u64) as usize;
            let slot_bytes = (n * rb).next_multiple_of(256);
            let mut wbytes = vec![0u8; slots * slot_bytes];
            for s in 0..slots {
                let r = random_rows(fmt, n, k, &mut rng);
                wbytes[s * slot_bytes..s * slot_bytes + r.len()].copy_from_slice(&r);
            }
            let x: Vec<f32> = (0..rows * k).map(|_| rng.f()).collect();
            // Pairs: (slot, x_row, y_row).
            let table: Vec<u32> = vec![2, 0, 0, 0, 2, 1, 3, 1, 2, 1, 0, 3];
            let pairs = table.len() / 3;
            let mut exp_direct = vec![0f32; rows * n];
            let mut exp_ind = vec![0f32; pairs * n];
            for nn in 0..n {
                for r in 0..rows {
                    let w = dequant(fmt, &wbytes[nn * rb..(nn + 1) * rb], k);
                    exp_direct[r * n + nn] = (0..k).map(|i| w[i] * x[r * k + i]).sum();
                }
                for p in 0..pairs {
                    let (s, xr, yr) = (
                        table[3 * p] as usize,
                        table[3 * p + 1] as usize,
                        table[3 * p + 2] as usize,
                    );
                    let o = s * slot_bytes + nn * rb;
                    let w = dequant(fmt, &wbytes[o..o + rb], k);
                    exp_ind[yr * n + nn] = (0..k).map(|i| w[i] * x[xr * k + i]).sum();
                }
            }
            let wb = ws.alloc(wbytes.len() as u64).unwrap();
            b.write_buffer(wb.id(), 0, &wbytes).unwrap();
            let xb = ws.alloc(act_bytes(rows as u32, k as u32)).unwrap();
            b.write_buffer(xb.id(), 0, bytemuck::cast_slice(&x))
                .unwrap();
            let s = ws.batch();
            let w_dir = s.import_copy(BufRef::new(wb.id(), (n * rb) as u64));
            let w_all = s.import_copy(wb.as_buf_ref());
            let xi = s.import_copy(xb.as_buf_ref());
            let yd = s.alloc(act_bytes(rows as u32, n as u32)).unwrap();
            let yi = s.alloc(act_bytes(pairs as u32, n as u32)).unwrap();
            let tb = s.write_uniform(bytemuck::cast_slice(&table)).unwrap();
            let mat = |buf, slot_bytes| Mat {
                buf,
                fmt,
                slot_bytes,
            };
            pipes
                .gemv(
                    &s,
                    mat(w_dir, 0),
                    Act::dense(xi, k as u32),
                    Act::dense(yd, n as u32),
                    None,
                    rows as u32,
                    n as u32,
                    k as u32,
                )
                .unwrap();
            pipes
                .gemv_gather(
                    &s,
                    mat(w_all, slot_bytes as u32),
                    Act::dense(xi, k as u32),
                    Act::dense(yi, n as u32),
                    tb,
                    pairs as u32,
                    n as u32,
                    k as u32,
                )
                .unwrap();
            let outs = s.submit_many(&[yd, yi]).await.unwrap();
            let gd = b.read_buffer(outs[0].id(), 0, outs[0].len()).await.unwrap();
            let gi = b.read_buffer(outs[1].id(), 0, outs[1].len()).await.unwrap();
            for (label, exp, got) in [
                ("direct", &exp_direct, bytemuck::cast_slice::<u8, f32>(&gd)),
                ("indirect", &exp_ind, bytemuck::cast_slice::<u8, f32>(&gi)),
            ] {
                let (rel, _) = compare(&format!("{fmt:?} n={n} k={k} {label}"), exp, got);
                if rel > 1e-4 {
                    fails.push(format!("{fmt:?} n={n} k={k} {label}: rel {rel:.2e}"));
                }
            }
        }
    }
    assert!(fails.is_empty(), "gemv:\n{}", fails.join("\n"));
}

/// The CUDA twins on one accelerator stream: dense gemv at 1 row (plain
/// kernel), 3 and 13 rows (multi-row prefill kernel, one and two row tiles),
/// 70 rows (tensor-core prefill, a full and a partial tile), and routed
/// experts through the address table at 6 pairs (plain), 40 pairs (grouped
/// by expert into tiles) and 300 pairs (tensor-core tiles), every format,
/// k = 512 / 2048 / 704. Tensor-core results (f16 operands, f32 sums) are
/// held to 2e-3, the rest to 1e-4. Skips without an accelerator.
#[tokio::test(flavor = "current_thread")]
async fn gemv_formats_cuda() {
    let b = backend().await;
    if b.accelerator().is_none() {
        eprintln!("no accelerator: skipped");
        return;
    }
    let pipes = LlmPipelines::compile(&b).await.expect("pipelines");
    let ws = Workspace::new(b.clone(), MemArbiter::new(MemTier::Vram, u64::MAX));
    let mut rng = Rng(0x2545f4914f6cdd1d);
    let mut fails = Vec::new();
    for fmt in thinfer_models::llm::kernels::FORMATS {
        let block = match fmt {
            LlmWeight::Quant(q) => q.block_size() as usize,
            LlmWeight::F32 => 32,
        };
        for (n, k) in [(37usize, 512usize), (64, 2048), (19, 704)] {
            if k % block != 0 {
                continue;
            }
            let (slots, xrows) = (5usize, 75usize);
            let rb = fmt.row_bytes(k as u64) as usize;
            let slot_bytes = (n * rb).next_multiple_of(256);
            let mut wbytes = vec![0u8; slots * slot_bytes];
            for s in 0..slots {
                let r = random_rows(fmt, n, k, &mut rng);
                wbytes[s * slot_bytes..s * slot_bytes + r.len()].copy_from_slice(&r);
            }
            let x: Vec<f32> = (0..xrows * k).map(|_| rng.f()).collect();
            let deq: Vec<Vec<f32>> = (0..slots * n)
                .map(|i| {
                    let (s, nn) = (i / n, i % n);
                    let o = s * slot_bytes + nn * rb;
                    dequant(fmt, &wbytes[o..o + rb], k)
                })
                .collect();
            let dot = |s: usize, nn: usize, xr: usize| -> f32 {
                (0..k).map(|i| deq[s * n + nn][i] * x[xr * k + i]).sum()
            };
            let wb = ws.alloc(wbytes.len() as u64).unwrap();
            b.write_buffer(wb.id(), 0, &wbytes).unwrap();
            let xb = ws.alloc(act_bytes(xrows as u32, k as u32)).unwrap();
            b.write_buffer(xb.id(), 0, bytemuck::cast_slice(&x))
                .unwrap();
            let base = b
                .device_address(&BufRef::new(wb.id(), wbytes.len() as u64))
                .expect("weights device-addressable");
            let addrs: Vec<u64> = (0..slots).map(|s| base + (s * slot_bytes) as u64).collect();
            let tbl = ws.alloc((slots * 8) as u64).unwrap();
            b.write_buffer(tbl.id(), 0, bytemuck::cast_slice(&addrs))
                .unwrap();
            // Experts: (pairs, x_div); ids cycle through the slots unevenly.
            let cases = [(6usize, 1usize), (40, 4), (300, 4)];
            let ids: Vec<Vec<u32>> = cases
                .iter()
                .map(|&(p, _)| (0..p).map(|i| ((i * 7 + i / 3) % slots) as u32).collect())
                .collect();
            let idb: Vec<_> = ids
                .iter()
                .map(|v| {
                    let buf = ws.alloc((v.len() * 4) as u64).unwrap();
                    b.write_buffer(buf.id(), 0, bytemuck::cast_slice(v))
                        .unwrap();
                    buf
                })
                .collect();
            let s = ws.batch();
            assert!(s.accel_begin().unwrap(), "accelerator scope");
            let w = s.import_copy(BufRef::new(wb.id(), (n * rb) as u64));
            let xi = s.import_copy(xb.as_buf_ref());
            let mut outs = Vec::new();
            for rows in [1u32, 3, 13, 70] {
                let y = s.alloc(act_bytes(rows, n as u32)).unwrap();
                let m = Mat {
                    buf: w,
                    fmt,
                    slot_bytes: 0,
                };
                let (xa, ya) = (Act::dense(xi, k as u32), Act::dense(y, n as u32));
                pipes
                    .gemv(&s, m, xa, ya, None, rows, n as u32, k as u32)
                    .unwrap();
                outs.push(y);
            }
            let ti = s.import_copy(tbl.as_buf_ref());
            for (c, &(pairs, x_div)) in cases.iter().enumerate() {
                let y = s.alloc(act_bytes(pairs as u32, n as u32)).unwrap();
                let idv = s.import_copy(idb[c].as_buf_ref());
                pipes
                    .gemv_experts(
                        &s,
                        fmt,
                        ti,
                        idv,
                        Act::dense(xi, k as u32),
                        Act::dense(y, n as u32),
                        pairs as u32,
                        x_div as u32,
                        n as u32,
                        k as u32,
                    )
                    .unwrap();
                outs.push(y);
            }
            let outs = s.submit_many(&outs).await.unwrap();
            let read = |i: usize| {
                let o = &outs[i];
                let b = b.clone();
                let (id, len) = (o.id(), o.len());
                async move { b.read_buffer(id, 0, len).await.unwrap() }
            };
            // The tensor-core kernels take quantized weights above 16 rows.
            let tol = |rows: usize| {
                if rows > 16 && fmt != LlmWeight::F32 {
                    2e-3
                } else {
                    1e-4
                }
            };
            for (i, rows) in [1usize, 3, 13, 70].into_iter().enumerate() {
                let got = read(i).await;
                let exp: Vec<f32> = (0..rows * n).map(|j| dot(0, j % n, j / n)).collect();
                let label = format!("{fmt:?} n={n} k={k} rows={rows}");
                let (rel, _) = compare(&label, &exp, bytemuck::cast_slice(&got));
                if rel > tol(rows) {
                    fails.push(format!("{label}: rel {rel:.2e}"));
                }
            }
            for (c, &(pairs, x_div)) in cases.iter().enumerate() {
                let got = read(4 + c).await;
                let exp: Vec<f32> = (0..pairs * n)
                    .map(|j| dot(ids[c][j / n] as usize, j % n, j / n / x_div))
                    .collect();
                let label = format!("{fmt:?} n={n} k={k} experts pairs={pairs}");
                let (rel, _) = compare(&label, &exp, bytemuck::cast_slice(&got));
                if rel > tol(pairs) {
                    fails.push(format!("{label}: rel {rel:.2e}"));
                }
            }
        }
    }
    assert!(fails.is_empty(), "gemv cuda:\n{}", fails.join("\n"));
}
