//! Activation-storage dtype helpers shared across models: upload/readback
//! encoders for the on-GPU activation dtypes, additive attention-mask byte
//! builders, and the f32->bf16 rounding helper. Model-specific sequence layout
//! (patchify, pos-id grids, pad lengths) lives in the model's own `seq` module.

use thinfer_core::ops::ActDtype;

/// Build the additive `[B=1, seq, seq]` attention mask for the bsz=1
/// full-attention path: all zeros.
///
/// The mask is 0 (-inf additive) *only* for batch-pad regions added by
/// `pad_sequence` when bsz > 1. Inner pad rows (the ones a model substitutes
/// pad tokens into) are treated as fully-attending tokens.
///
/// For bsz=1 there is no batch padding, so every position is attending and
/// the additive mask is all zeros. The SDPA kernel still requires a mask
/// binding, so we write a zero buffer rather than skipping it.
pub fn attn_mask_zero_bytes(seq: usize) -> Vec<u8> {
    vec![0u8; seq * seq * 4]
}

/// Mask bytes per elem for sdpa's additive mask. F32 path = 4; F16/Bf16/I8
/// all use the F16 vec2 mask layout (matmul_i8 emits f16, sdpa_i8 reads
/// vec2<f16> mask per the I8 acts contract).
fn mask_bytes_per_elem(act: ActDtype) -> usize {
    match act {
        ActDtype::F32 => 4,
        ActDtype::Bf16 | ActDtype::F16 | ActDtype::I8 => 2,
    }
}

/// Bytes for an all-attending additive mask in the sdpa mask layout.
/// F32: `seq*seq*4` zero bytes. F16/Bf16/I8: `seq*seq*2` zero bytes.
/// Caller passes `seq` that is even on packed paths so each row's pair
/// stream fits whole `array<u32>` words.
pub fn attn_mask_zero_bytes_act(seq: usize, act: ActDtype) -> Vec<u8> {
    vec![0u8; seq * seq * mask_bytes_per_elem(act)]
}

/// Build a `[1, seq, seq]` additive causal mask: `0.0` on/below diagonal,
/// `-inf` strictly above. Used by causal LM stacks (e.g. Qwen3 text encoder).
pub fn causal_mask_bytes(seq: usize) -> Vec<u8> {
    let mut out = vec![0u8; seq * seq * 4];
    let neg_inf = f32::NEG_INFINITY.to_le_bytes();
    for q in 0..seq {
        for k in (q + 1)..seq {
            let off = (q * seq + k) * 4;
            out[off..off + 4].copy_from_slice(&neg_inf);
        }
    }
    out
}

/// Causal mask in the activation storage layout. Bf16: each elem encoded as
/// 2-byte bf16 (`0x0000` for 0.0, `0xff80` for -inf), little-endian in
/// `array<u32>` packed pairs along `s_k`. `seq` must be even for the Bf16
/// path so each row's pair stream fits a whole word.
pub fn causal_mask_bytes_act(seq: usize, act: ActDtype) -> Vec<u8> {
    match act {
        ActDtype::F32 => causal_mask_bytes(seq),
        ActDtype::Bf16 => {
            assert!(
                seq.is_multiple_of(2),
                "causal_mask_bytes_act: bf16 path requires seq even (got {seq})"
            );
            let mut out = vec![0u8; seq * seq * 2];
            let neg_inf_bf16: u16 = 0xff80;
            for q in 0..seq {
                for k in (q + 1)..seq {
                    let off = (q * seq + k) * 2;
                    out[off..off + 2].copy_from_slice(&neg_inf_bf16.to_le_bytes());
                }
            }
            out
        }
        ActDtype::F16 | ActDtype::I8 => {
            // I8 acts reuse the F16 vec2 mask layout: matmul_i8 emits f16
            // and sdpa_i8 reads `vec2<f16>` mask (per I8 acts contract).
            assert!(
                seq.is_multiple_of(2),
                "causal_mask_bytes_act: f16/i8 path requires seq even (got {seq})"
            );
            let mut out = vec![0u8; seq * seq * 2];
            // IEEE binary16 -inf is `0xfc00`.
            let neg_inf_f16: u16 = 0xfc00;
            for q in 0..seq {
                for k in (q + 1)..seq {
                    let off = (q * seq + k) * 2;
                    out[off..off + 2].copy_from_slice(&neg_inf_f16.to_le_bytes());
                }
            }
            out
        }
    }
}

/// Encode `slice` for upload into a RoPE freqs buffer (act-dtype layout).
pub fn freqs_upload_bytes(act: ActDtype, slice: &[f32]) -> Vec<u8> {
    act_upload_bytes(act, slice)
}

/// Encode `slice` for upload into an activation-storage buffer.
/// - `F32`: 4 bytes per elem little-endian.
/// - `Bf16`: 2 bytes per elem RNE-rounded bf16; consecutive even/odd pairs
///   land in the same `array<u32>` word (low half = even index, high half =
///   odd index), matching `pack_bf16x2(lo, hi)` in WGSL.
pub fn act_upload_bytes(act: ActDtype, slice: &[f32]) -> Vec<u8> {
    match act {
        ActDtype::F32 => {
            let mut bytes = vec![0u8; slice.len() * 4];
            for (i, v) in slice.iter().enumerate() {
                bytes[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
            }
            bytes
        }
        ActDtype::Bf16 => {
            let mut bytes = vec![0u8; slice.len() * 2];
            for (i, v) in slice.iter().enumerate() {
                let h = round_f32_to_bf16(*v);
                bytes[i * 2..(i + 1) * 2].copy_from_slice(&h.to_le_bytes());
            }
            bytes
        }
        ActDtype::F16 => {
            // IEEE binary16. `half::f16::from_f32` uses RNE; matches the
            // WGSL `f16(...)` constructor's rounding mode.
            let mut bytes = vec![0u8; slice.len() * 2];
            for (i, v) in slice.iter().enumerate() {
                let h = half::f16::from_f32(*v).to_bits();
                bytes[i * 2..(i + 1) * 2].copy_from_slice(&h.to_le_bytes());
            }
            bytes
        }
        ActDtype::I8 => unreachable!("I8 is never a block act_dtype"),
    }
}

/// Decode `bytes` produced by an activation-storage readback into a `Vec<f32>`
/// of `n_elems`. Bf16 path zero-extends the 2-byte half to f32 via `(h << 16)`.
pub fn act_readback_to_f32(act: ActDtype, bytes: &[u8], n_elems: usize) -> Vec<f32> {
    let mut out = vec![0f32; n_elems];
    match act {
        ActDtype::F32 => {
            debug_assert_eq!(bytes.len(), n_elems * 4);
            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        ActDtype::Bf16 => {
            debug_assert_eq!(bytes.len(), n_elems * 2);
            for (i, chunk) in bytes.chunks_exact(2).enumerate() {
                let half = u16::from_le_bytes([chunk[0], chunk[1]]);
                out[i] = f32::from_bits((half as u32) << 16);
            }
        }
        ActDtype::F16 => {
            debug_assert_eq!(bytes.len(), n_elems * 2);
            for (i, chunk) in bytes.chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out[i] = half::f16::from_bits(bits).to_f32();
            }
        }
        ActDtype::I8 => {
            unreachable!(
                "I8 acts use the paired (data, scale) ABI: call `act_readback_i8_to_f32` instead"
            )
        }
    }
    out
}

/// Decode paired I8 bytes (packed i8 data + per-32-block vec2<f16> params)
/// back to `rows * dim` f32 elements. Used for sdpa_i8 I/O tap decode.
pub fn act_readback_i8_to_f32(data: &[u8], scale: &[u8], rows: usize, dim: usize) -> Vec<f32> {
    assert!(
        dim.is_multiple_of(32),
        "act_readback_i8_to_f32: dim={dim} must be a multiple of 32"
    );
    assert_eq!(
        data.len(),
        rows * dim,
        "act_readback_i8_to_f32: data len mismatch"
    );
    let blocks = dim / 32;
    assert_eq!(
        scale.len(),
        rows * blocks * 4,
        "act_readback_i8_to_f32: scale len mismatch"
    );
    let mut out = vec![0f32; rows * dim];
    for r in 0..rows {
        let row_base = r * dim;
        for k in 0..blocks {
            let scale_off = (r * blocks + k) * 4;
            let sc =
                half::f16::from_bits(u16::from_le_bytes([scale[scale_off], scale[scale_off + 1]]))
                    .to_f32();
            let z = half::f16::from_bits(u16::from_le_bytes([
                scale[scale_off + 2],
                scale[scale_off + 3],
            ]))
            .to_f32();
            let data_off = row_base + k * 32;
            for j in 0..32 {
                let q = data[data_off + j] as i8;
                out[data_off + j] = (q as f32) * sc + z;
            }
        }
    }
    out
}

/// Diagnostic: simulate the quant -> dequant round-trip of a dense f32 tensor
/// using BOTH the current scheme (vec2<f16> params) and a hypothetical
/// vec2<f32> params layout. Emit linfit (slope, offset, rmse) for each
/// round-tripped tensor vs the original.
///
/// Purpose: tell us whether f16 narrowing of `(s, z)` is responsible for the
/// 0.938 slope observed in `step0.prev_sample`. If f16 round-trip slope is
/// ~0.998 then 30-block compounding gives 0.998^30 ~= 0.94 (matches);
/// if it's ~0.9999 then 30-block compounding gives ~0.997 (rules out).
pub fn diag_quant_roundtrip_loss(label: &str, dense: &[f32], rows: usize, dim: usize) {
    assert_eq!(dense.len(), rows * dim);
    assert!(dim.is_multiple_of(32));
    let blocks = dim / 32;
    let mut got_f16 = vec![0f32; dense.len()];
    let mut got_f32 = vec![0f32; dense.len()];
    let mut max_param_drift_f16: f32 = 0.0;
    let mut sum_param_drift_f16: f64 = 0.0;
    let mut n_blocks_total = 0usize;
    for r in 0..rows {
        for k in 0..blocks {
            let base = r * dim + k * 32;
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for j in 0..32 {
                let v = dense[base + j];
                if v < mn {
                    mn = v;
                }
                if v > mx {
                    mx = v;
                }
            }
            let range = mx - mn;
            let s_exact = if range <= 0.0 { 1.0e-30 } else { range / 254.0 };
            let z_exact = mn + 127.0 * s_exact;
            // Current scheme: narrow (s, z) to f16, dequant via f16 values.
            let s_f16 = half::f16::from_f32(s_exact).to_f32();
            let z_f16 = half::f16::from_f32(z_exact).to_f32();
            // Quantize using the SAME params we'll dequant with (matches the
            // GPU kernel which reads back the narrowed shared_params).
            for j in 0..32 {
                let v = dense[base + j];
                // f16 path
                let q_f16 = (((v - z_f16) / s_f16).round()).clamp(-127.0, 127.0) as i32;
                got_f16[base + j] = (q_f16 as f32) * s_f16 + z_f16;
                // f32 path
                let q_f32 = (((v - z_exact) / s_exact).round()).clamp(-127.0, 127.0) as i32;
                got_f32[base + j] = (q_f32 as f32) * s_exact + z_exact;
            }
            let drift = ((s_f16 - s_exact).abs() + (z_f16 - z_exact).abs())
                / (s_exact.abs() + z_exact.abs() + 1e-12);
            if drift > max_param_drift_f16 {
                max_param_drift_f16 = drift;
            }
            sum_param_drift_f16 += drift as f64;
            n_blocks_total += 1;
        }
    }
    let fit = |got: &[f32], exp: &[f32]| -> (f64, f64, f64) {
        let n = got.len().min(exp.len()).min(4096);
        let mut sx = 0.0f64;
        let mut sy = 0.0f64;
        let mut sxx = 0.0f64;
        let mut sxy = 0.0f64;
        for i in 0..n {
            let x = exp[i] as f64;
            let y = got[i] as f64;
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        let nf = n as f64;
        let denom = nf * sxx - sx * sx;
        if denom.abs() < 1e-12 {
            return (1.0, 0.0, 0.0);
        }
        let a = (nf * sxy - sx * sy) / denom;
        let b = (sy - a * sx) / nf;
        let mut resid_sq = 0.0f64;
        for i in 0..n {
            let x = exp[i] as f64;
            let y = got[i] as f64;
            let r = y - (a * x + b);
            resid_sq += r * r;
        }
        (a, b, (resid_sq / nf).sqrt())
    };
    let (af16, bf16, rf16) = fit(&got_f16, dense);
    let (af32, bf32, rf32) = fit(&got_f32, dense);
    let mean_drift = sum_param_drift_f16 / (n_blocks_total.max(1) as f64);
    tracing::debug!(
        target: thinfer_core::trace::DIAG,
        "[{label}] roundtrip(f16-params) slope={af16:.6} off={bf16:+.6e} rmse={rf16:.4e} \
         |  roundtrip(f32-params) slope={af32:.6} off={bf32:+.6e} rmse={rf32:.4e} \
         |  param_rel_drift_f16 mean={mean_drift:.4e} max={max_param_drift_f16:.4e} \
         (n_blocks={n_blocks_total}, dim={dim}, rows={rows})"
    );
}

/// f32 -> bf16 round-to-nearest-even with NaN canonicalization. Bit-identical
/// to the WGSL `round_bf16` helper.
pub fn round_f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let exp = (bits >> 23) & 0xff;
    if exp == 0xff {
        let mant = bits & 0x7f_ffff;
        let top = (bits >> 16) as u16;
        if mant == 0 { top } else { top | 0x0040 }
    } else {
        let rounding = 0x7fff + ((bits >> 16) & 1);
        ((bits.wrapping_add(rounding)) >> 16) as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attn_mask_all_zero_for_bsz1() {
        // bsz=1: no batch padding -> additive mask is all zeros. Inner pad
        // rows are fully attending.
        let bytes = attn_mask_zero_bytes(5);
        let m: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        assert_eq!(m, vec![0.0; 25]);
    }

    #[test]
    fn causal_mask_is_triangular() {
        let bytes = causal_mask_bytes(4);
        let m: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        assert_eq!(m.len(), 16);
        // Row 0: only (0,0) attending; row 1: (1,0), (1,1); etc.
        for q in 0..4 {
            for k in 0..4 {
                let v = m[q * 4 + k];
                if k <= q {
                    assert_eq!(v, 0.0);
                } else {
                    assert!(v.is_infinite() && v.is_sign_negative());
                }
            }
        }
    }
}
