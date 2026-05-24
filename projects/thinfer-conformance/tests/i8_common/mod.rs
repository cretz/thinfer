//! Shared helpers for the per-op `*_i8.rs` conformance tests. Each test
//! binary in `tests/` includes this via `mod i8_common;`.
//!
//! Contract on packed-int8 activations: `(data: array<u32>, params:
//! array<vec2<f16>>)` with params length `rows * (dim/32)` blocks, each
//! block carrying `(scale, zero)` as two f16s (4 bytes/block, identical
//! footprint to the old f32-scale layout). Data packs 4 i8 per u32 along
//! the K axis. Producer kernels emit asymmetric llama.cpp Q8_1-style quant:
//! `scale = (max-min)/254`, `zero = min + 127*scale`,
//! `q = round((x - zero) / scale).clamp(-127, 127)`. Helpers here mirror
//! that exactly so per-op CPU references stay aligned. CPU-side params are
//! interleaved `Vec<f32>` of length `2 * rows * blocks` with the f16-rounded
//! values the kernel actually stores.

// Cross-binary helpers: each integration test binary in `tests/` compiles
// this module independently and uses a different subset.
#![allow(dead_code)]

use thinfer_core::backend::{Backend, BufRef, WgpuBackend};

pub const BK: usize = 32;

// ---------------- f16 / bf16 helpers -----------------------------------

pub fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 31) & 1) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let mant = b & 0x7FFFFF;
    if exp == 0 {
        return sign << 15;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1F {
        return (sign << 15) | (0x1F << 10);
    }
    if new_exp <= 0 {
        return sign << 15;
    }
    let m10 = mant >> 13;
    let rem = mant & 0x1FFF;
    let half_bit = 0x1000;
    let round_up = rem > half_bit || (rem == half_bit && (m10 & 1) == 1);
    let mut out_mant = m10 + if round_up { 1 } else { 0 };
    let mut out_exp = new_exp as u16;
    if out_mant == 0x400 {
        out_mant = 0;
        out_exp += 1;
    }
    (sign << 15) | (out_exp << 10) | (out_mant as u16 & 0x3FF)
}

pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        return f32::from_bits(sign);
    }
    if exp == 0x1F {
        return f32::from_bits(sign | 0x7F80_0000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
}

/// bf16 = top 16 bits of f32 (truncate-toward-zero on mantissa; matches the
/// kernel's `bitcast<f32>(w << 16u)` decode for both halves).
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

pub fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Pack `vals` as bf16 stored as `array<u32>`: two adjacent bf16s per u32
/// (lo=even index, hi=odd index), matching the kernel's `load_*1` decode.
pub fn pack_bf16_vec(vals: &[f32]) -> Vec<u8> {
    assert!(vals.len().is_multiple_of(2));
    let mut out = Vec::with_capacity(vals.len() * 2);
    for pair in vals.chunks_exact(2) {
        let lo = f32_to_bf16_bits(pair[0]) as u32;
        let hi = f32_to_bf16_bits(pair[1]) as u32;
        let w = lo | (hi << 16);
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

/// Pack `vals` as f16x2 stored as `array<vec2<f16>>` (two adjacent f16s per
/// 4-byte word, lo at lower address).
pub fn pack_f16_vec(vals: &[f32]) -> Vec<u8> {
    assert!(vals.len().is_multiple_of(2));
    let mut out = Vec::with_capacity(vals.len() * 2);
    for pair in vals.chunks_exact(2) {
        let lo = f32_to_f16_bits(pair[0]);
        let hi = f32_to_f16_bits(pair[1]);
        out.extend_from_slice(&lo.to_le_bytes());
        out.extend_from_slice(&hi.to_le_bytes());
    }
    out
}

// ---------------- i8 pack / unpack -------------------------------------

pub fn pack4_i8(q: [i8; 4]) -> u32 {
    (q[0] as u8 as u32)
        | ((q[1] as u8 as u32) << 8)
        | ((q[2] as u8 as u32) << 16)
        | ((q[3] as u8 as u32) << 24)
}

/// Pack `[rows, dim]` of i8 row-major into the kernel's u32 layout (4 i8
/// per word along K).
pub fn pack_i8_rows(src: &[i8], rows: usize, dim: usize) -> Vec<u8> {
    assert_eq!(src.len(), rows * dim);
    assert!(dim.is_multiple_of(4));
    let words_per_row = dim / 4;
    let mut out = Vec::with_capacity(rows * words_per_row * 4);
    for r in 0..rows {
        for w in 0..words_per_row {
            let base = r * dim + w * 4;
            let word = pack4_i8([src[base], src[base + 1], src[base + 2], src[base + 3]]);
            out.extend_from_slice(&word.to_le_bytes());
        }
    }
    out
}

pub fn unpack_i8_rows(bytes: &[u8], rows: usize, dim: usize) -> Vec<i8> {
    assert_eq!(bytes.len(), rows * dim);
    bytes.iter().map(|b| *b as i8).collect()
}

/// Pack interleaved `(scale, zero)` f32 pairs as `array<vec2<f16>>`: each
/// block emits 4 bytes (2x f16). `params.len()` must be `2 * blocks_total`.
pub fn params_to_bytes(params: &[f32]) -> Vec<u8> {
    assert!(params.len().is_multiple_of(2));
    let mut out = Vec::with_capacity(params.len() * 2);
    for pair in params.chunks_exact(2) {
        let s = f32_to_f16_bits(pair[0]);
        let z = f32_to_f16_bits(pair[1]);
        out.extend_from_slice(&s.to_le_bytes());
        out.extend_from_slice(&z.to_le_bytes());
    }
    out
}

/// Decode `array<vec2<f16>>` bytes back to interleaved `Vec<f32>` of length
/// `2 * blocks` (i.e. `[s0, z0, s1, z1, ...]`).
pub fn bytes_to_params(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len().is_multiple_of(2));
    bytes
        .chunks_exact(2)
        .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

// ---------------- Random + fixture generators --------------------------

/// Deterministic LCG f32 in `(-1, 1)`.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0xA5A5_5A5A_DEAD_BEEF))
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.0
    }
    pub fn f32_signed(&mut self) -> f32 {
        let r = ((self.next_u64() >> 40) as u32 as f32) / ((1u32 << 24) as f32);
        r * 2.0 - 1.0
    }
    /// i8 in `[-127, 127]` (avoids -128 to match the kernel's invariant).
    pub fn i8(&mut self) -> i8 {
        ((self.next_u64() >> 56) as i32 - 128).clamp(-127, 127) as i8
    }
    /// Small positive f32 in `[1e-3, 1e-2]` (typical block scale magnitude).
    pub fn scale(&mut self) -> f32 {
        let r = ((self.next_u64() >> 40) as u32 as f32) / ((1u32 << 24) as f32);
        1e-3 + r * 9e-3
    }
}

/// Round f32 through f16 and back. Producer kernels narrow `(scale, zero)`
/// to f16 before storing, so CPU references must match that rounding to
/// stay bit-aligned.
pub fn f16_round(x: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(x))
}

/// Generate paired `(i8 data [rows*dim], params [2*rows*blocks])` from a
/// fully asymmetric source: per-block dense f32 with a per-block offset so
/// `zero` is non-zero and varies block to block. This is critical - a z=0
/// fixture lets kernels that ignore the input zero-point pass trivially,
/// then fail at runtime where upstream emits z != 0. Params are interleaved
/// `[s0, z0, s1, z1, ...]` and already f16-rounded.
pub fn gen_paired_i8(rows: usize, dim: usize, seed: u64) -> (Vec<i8>, Vec<f32>) {
    assert!(dim.is_multiple_of(BK));
    let blocks = dim / BK;
    let mut rng = Rng::new(seed);
    let mut offset_rng = Rng::new(seed ^ 0xDEAD_BEEF);
    let mut dense = vec![0f32; rows * dim];
    for r in 0..rows {
        for kb in 0..blocks {
            // Per-block bias in roughly [-1.0, 1.0] forces a non-zero zero
            // point after asymmetric quant.
            let bias = offset_rng.f32_signed();
            let amp = 0.25 + 0.5 * (offset_rng.f32_signed() + 1.0) * 0.5;
            for j in 0..BK {
                let r_unit = rng.f32_signed();
                dense[r * dim + kb * BK + j] = bias + amp * r_unit;
            }
        }
    }
    requant_rows(&dense, rows, dim)
}

/// Dequantize the paired `(i8, params)` representation to dense f32
/// `[rows, dim]` row-major using `value = i8 * scale[r, kb] + zero[r, kb]`.
/// `params` is interleaved `[s, z]` of length `2 * rows * blocks`.
pub fn dequant_paired(data: &[i8], params: &[f32], rows: usize, dim: usize) -> Vec<f32> {
    assert!(dim.is_multiple_of(BK));
    let blocks = dim / BK;
    assert_eq!(data.len(), rows * dim);
    assert_eq!(params.len(), 2 * rows * blocks);
    let mut out = vec![0f32; rows * dim];
    for r in 0..rows {
        for kb in 0..blocks {
            let s = params[2 * (r * blocks + kb)];
            let z = params[2 * (r * blocks + kb) + 1];
            for j in 0..BK {
                let i = r * dim + kb * BK + j;
                out[i] = data[i] as f32 * s + z;
            }
        }
    }
    out
}

/// Mirror the kernel's per-block asymmetric quantize (llama.cpp Q8_1-style):
/// `scale = (max-min)/254` (or 1e-30 if range<=0), `zero = min + 127*scale`,
/// then `q = round((x - zero) / scale).clamp(-127, 127)`. Both scale and
/// zero are f16-rounded to match what the kernel stores.
pub fn quant_block_to_i8(vals: &[f32]) -> ([i8; BK], f32, f32) {
    assert_eq!(vals.len(), BK);
    let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in vals {
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    let range = mx - mn;
    let s_raw = if range <= 0.0 { 1.0e-30 } else { range / 254.0 };
    let z_raw = mn + 127.0 * s_raw;
    let s = f16_round(s_raw);
    let z = f16_round(z_raw);
    let inv = 1.0 / s;
    let mut q = [0i8; BK];
    for (i, &v) in vals.iter().enumerate() {
        let r = ((v - z) * inv).round().clamp(-127.0, 127.0);
        q[i] = r as i32 as i8;
    }
    (q, s, z)
}

/// Apply `quant_block_to_i8` row-by-row across `[rows, dim]` f32 to produce
/// paired `(i8 [rows*dim], params [2*rows*blocks])`.
pub fn requant_rows(dense: &[f32], rows: usize, dim: usize) -> (Vec<i8>, Vec<f32>) {
    assert!(dim.is_multiple_of(BK));
    assert_eq!(dense.len(), rows * dim);
    let blocks = dim / BK;
    let mut data = vec![0i8; rows * dim];
    let mut params = vec![0f32; 2 * rows * blocks];
    for r in 0..rows {
        for kb in 0..blocks {
            let off = r * dim + kb * BK;
            let block = &dense[off..off + BK];
            let (q, s, z) = quant_block_to_i8(block);
            data[off..off + BK].copy_from_slice(&q);
            params[2 * (r * blocks + kb)] = s;
            params[2 * (r * blocks + kb) + 1] = z;
        }
    }
    (data, params)
}

// ---------------- Backend helpers --------------------------------------

pub async fn make_backend_with_f16() -> Option<WgpuBackend> {
    let backend = WgpuBackend::new().await.expect("wgpu adapter");
    if !backend.supports_shader_f16() {
        eprintln!("skip: adapter does not expose SHADER_F16");
        return None;
    }
    Some(backend)
}

pub fn alloc_with<B: Backend>(backend: &B, bytes: &[u8]) -> BufRef {
    let id = backend.allocate(bytes.len() as u64).expect("allocate");
    backend.write_buffer(id, 0, bytes).expect("write");
    BufRef::new(id, bytes.len() as u64)
}

pub fn alloc_zero<B: Backend>(backend: &B, len: u64) -> BufRef {
    let id = backend.allocate(len).expect("allocate");
    BufRef::new(id, len)
}

/// 16-byte uniform: `[rows, dim, pad0, pad1]` (kernels' canonical `struct U`).
pub fn pack_rows_dim_uniform(rows: u32, dim: u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&rows.to_le_bytes());
    out[4..8].copy_from_slice(&dim.to_le_bytes());
    out
}

/// 16-byte uniform: `[rows, dim, f32_field, pad]`.
pub fn pack_rows_dim_f32(rows: u32, dim: u32, f: f32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&rows.to_le_bytes());
    out[4..8].copy_from_slice(&dim.to_le_bytes());
    out[8..12].copy_from_slice(&f.to_le_bytes());
    out
}

// ---------------- Tolerance comparison ---------------------------------

/// Compare paired (got_data, got_params) vs (exp_data, exp_params) in the
/// dequantized domain. Tolerance: per-block i8 quant step + a relative slop
/// for arithmetic precision. Returns max abs error in the dequant domain.
pub fn assert_paired_close(
    got_data: &[i8],
    got_params: &[f32],
    exp_data: &[i8],
    exp_params: &[f32],
    rows: usize,
    dim: usize,
    rel_tol: f32,
    abs_tol: f32,
    label: &str,
) {
    let got = dequant_paired(got_data, got_params, rows, dim);
    let exp = dequant_paired(exp_data, exp_params, rows, dim);
    let max_abs_exp = exp.iter().fold(0f32, |a, &x| a.max(x.abs()));
    let tol = (rel_tol * max_abs_exp).max(abs_tol);
    let (max_err, idx) = got
        .iter()
        .zip(&exp)
        .enumerate()
        .map(|(i, (g, e))| ((g - e).abs(), i))
        .fold((0f32, 0usize), |a, b| if b.0 > a.0 { b } else { a });
    assert!(
        max_err <= tol,
        "{label}: max dequant abs diff {max_err:.3e} > tol {tol:.3e} at idx {idx} \
         (got={} exp={}) exp_max={}",
        got[idx],
        exp[idx],
        max_abs_exp,
    );
}

/// Run a single dispatch: build pipeline, encode, submit, then read both
/// output buffers (paired data+params). Returns dequantized comparison-ready
/// `(got_i8, got_params)` where params is interleaved `[s, z]`. Caller owns
/// binding & dispatch via the closure.
pub async fn run_and_read_paired<B, F>(
    backend: &B,
    wgsl: &str,
    layout: &'static [thinfer_core::backend::BindingLayout],
    out_data: &BufRef,
    out_params: &BufRef,
    dispatch: F,
) -> (Vec<i8>, Vec<f32>)
where
    B: Backend,
    F: FnOnce(&B, &mut B::CommandEncoder, &B::Pipeline),
{
    let pipeline = backend
        .create_pipeline(wgsl, "main", layout)
        .await
        .expect("pipeline");
    let mut enc = backend.create_command_encoder();
    dispatch(backend, &mut enc, &pipeline);
    backend.submit(enc).await.expect("submit");
    let data_bytes = backend
        .read_buffer(out_data.id, out_data.offset, out_data.len)
        .await
        .expect("read data");
    let params_bytes = backend
        .read_buffer(out_params.id, out_params.offset, out_params.len)
        .await
        .expect("read params");
    let data_i8: Vec<i8> = data_bytes.iter().map(|b| *b as i8).collect();
    let params: Vec<f32> = bytes_to_params(&params_bytes);
    (data_i8, params)
}
