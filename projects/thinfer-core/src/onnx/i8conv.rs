//! Opt-in i8 DP4A path for the ONNX executor's plain (group=1, dilation=1)
//! convs -- the HyperSwap interior bulk. Per-Cout symmetric i8 weights + a
//! per-tensor dynamic ASYMMETRIC i8 activation quant (zero-point, llama.cpp
//! Q8_1 style -- post-activation tensors are non-negative/skewed, so a symmetric
//! range wastes half the codes). `dot4I8Packed` over the same implicit-GEMM
//! tiling as the f32 `ops::Conv2dF32`. Dequant per the asymmetric expansion
//! `Σ(qa*sa+za)*(qw*sw) = sa*sw*Σqa*qw + za*sw*Σqw`, so:
//!   `out = f32(acc_i32) * (a_scale * w_scale[co]) + a_zero * w_scale[co]
//!          * w_qsum[co] + bias[co]`.
//!
//! Gated OFF by default (`THINFER_ONNX_I8_CONV`, HyperSwap-scoped by the
//! faceswap loader): i8 rounds the conv and leaves a faint grain on the swapped
//! face (per-tensor quant), so it needs an A/B eyeball. Input (cin<=4) + output
//! (cout<=4) convs stay f32.
//!
//! Pipeline per eligible conv (built in `exec.rs`): 1. `CLEAR_U32` (zero the
//! min/max accumulator), 2. `act_minmax_wgsl` (atomic min+max reduction),
//! 3. `act_quant_wgsl` (pack f32 -> i8, write `(scale, zero)`), 4. `conv_i8`.
//! Weights are quantized + packed once at load.
//!
//! Packing: i8 values are 4-per-u32, little-endian byte order, matching
//! `dot4I8Packed` (signed). Weights are `[Cout, ceil(K/4)]` u32 with K =
//! Cin*kH*kW in (ci, dh, dw) order, tail-padded with zeros. Activations are
//! packed in flat NCHW order; the conv's im2col gather extracts individual
//! bytes and re-packs 4 K-consecutive taps per word, substituting the quantized
//! value of 0.0 at out-of-bounds (padding) taps so the correction stays exact.

use crate::backend::{BindingKind, BindingLayout};

/// Binding layout for `act_minmax_wgsl`: x(read), mm(rw), uniform.
pub fn act_maxabs_layout() -> Vec<BindingLayout> {
    vec![
        BindingLayout {
            slot: 0,
            kind: BindingKind::StorageRead,
        },
        BindingLayout {
            slot: 1,
            kind: BindingKind::StorageReadWrite,
        },
        BindingLayout {
            slot: 2,
            kind: BindingKind::Uniform,
        },
    ]
}

/// Binding layout for `act_quant_wgsl`: x(read), mm(read), x_i8(rw), params(rw), uniform.
pub fn act_quant_layout() -> Vec<BindingLayout> {
    vec![
        BindingLayout {
            slot: 0,
            kind: BindingKind::StorageRead,
        },
        BindingLayout {
            slot: 1,
            kind: BindingKind::StorageRead,
        },
        BindingLayout {
            slot: 2,
            kind: BindingKind::StorageReadWrite,
        },
        BindingLayout {
            slot: 3,
            kind: BindingKind::StorageReadWrite,
        },
        BindingLayout {
            slot: 4,
            kind: BindingKind::Uniform,
        },
    ]
}

/// Binding layout for `conv_i8`: x_i8, a_params(scale,zero), wgt, w_scale,
/// w_qsum, bias (read), out(rw), uniform.
pub fn conv_i8_layout() -> Vec<BindingLayout> {
    let mut l: Vec<BindingLayout> = (0..6)
        .map(|i| BindingLayout {
            slot: i,
            kind: BindingKind::StorageRead,
        })
        .collect();
    l.push(BindingLayout {
        slot: 6,
        kind: BindingKind::StorageReadWrite,
    });
    l.push(BindingLayout {
        slot: 7,
        kind: BindingKind::Uniform,
    });
    l
}

/// Pack a signed-i8 slice (values already clamped to [-127, 127]) 4-per-u32 in
/// little-endian byte order. Tail short of a multiple of 4 is zero-padded.
pub fn pack_i8(vals: &[i8]) -> Vec<u32> {
    let mut out = vec![0u32; vals.len().div_ceil(4)];
    for (i, &v) in vals.iter().enumerate() {
        out[i / 4] |= ((v as u8) as u32) << ((i % 4) * 8);
    }
    out
}

/// Quantize a conv weight `[Cout, Cin, kH, kW]` (row-major) to symmetric i8 with
/// one scale per Cout row, packed `[Cout, ceil(K/4)]` u32 (K = Cin*kH*kW, tail
/// zero-padded per row). Returns (packed_words, per_cout_scale, per_cout_qsum),
/// where `qsum[co] = Σ_k q[co,k]` feeds the asymmetric-activation correction
/// term. Scale is `maxabs/127` (or a tiny epsilon for an all-zero row).
pub fn quantize_weight(w: &[f32], cout: usize, k: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let kw4 = k.div_ceil(4);
    let mut packed = vec![0u32; cout * kw4];
    let mut scales = vec![0f32; cout];
    let mut qsum = vec![0f32; cout];
    for co in 0..cout {
        let row = &w[co * k..co * k + k];
        let maxabs = row.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let scale = if maxabs > 0.0 { maxabs / 127.0 } else { 1e-30 };
        scales[co] = scale;
        let mut sum = 0i32;
        for (kk, &x) in row.iter().enumerate() {
            let q = (x / scale).round().clamp(-127.0, 127.0) as i32;
            sum += q;
            packed[co * kw4 + kk / 4] |= ((q as i8 as u8) as u32) << ((kk % 4) * 8);
        }
        qsum[co] = sum as f32;
    }
    (packed, scales, qsum)
}

/// Zero the min/max accumulator (`mm[0]`=max init 0, `mm[1]`=min init 0xFFFF_FFFF)
/// inside the encoder. A host `write_buffer` can't be used: all queue writes are
/// ordered before the single executor submit, so per-conv clears would all land
/// before any kernel runs. Bindings: 0=mm (rw, 2 u32). One 1-thread workgroup.
pub const CLEAR_U32: &str = r#"
@group(0) @binding(0) var<storage, read_write> mm: array<u32>;
@compute @workgroup_size(1)
fn main() { mm[0] = 0u; mm[1] = 0xFFFFFFFFu; }
"#;

/// Binding layout for `CLEAR_U32`: one rw buffer.
pub fn clear_layout() -> Vec<BindingLayout> {
    vec![BindingLayout {
        slot: 0,
        kind: BindingKind::StorageReadWrite,
    }]
}

// Order-preserving f32<->u32 map so integer atomics compute float min/max:
// negatives (sign bit set) -> ~bits (descending), non-negatives -> bits|0x8000_0000.
const ORDER_FNS: &str = r#"
fn order(f: f32) -> u32 {
  let b = bitcast<u32>(f);
  if ((b >> 31u) != 0u) { return ~b; }
  return b | 0x80000000u;
}
fn unorder(uu: u32) -> f32 {
  if ((uu >> 31u) != 0u) { return bitcast<f32>(uu & 0x7FFFFFFFu); }
  return bitcast<f32>(~uu);
}
"#;

/// Per-workgroup min+max reduction over an f32 activation tensor, folded into
/// atomic `mm` (`[max, min]` as order-preserving u32; host clears to [0, MAX]).
/// One workgroup per 256 elements. Bindings: 0=x (f32), 1=mm (atomic<u32> x2),
/// 2=uniform{n}.
pub fn act_minmax_wgsl() -> String {
    format!(
        r#"
struct U {{ n: u32, _a: u32, _b: u32, _c: u32 }};
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> mm: array<atomic<u32>, 2>;
@group(0) @binding(2) var<uniform> u: U;
{ORDER_FNS}
var<workgroup> rmax: array<f32, 256>;
var<workgroup> rmin: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>) {{
  let i = gid.y * (ng.x * 256u) + gid.x;
  var v: f32 = 0.0;
  if (i < u.n) {{ v = x[i]; }}
  rmax[lid.x] = v;
  rmin[lid.x] = v;
  workgroupBarrier();
  var s: u32 = 128u;
  loop {{
    if (s == 0u) {{ break; }}
    if (lid.x < s) {{
      rmax[lid.x] = max(rmax[lid.x], rmax[lid.x + s]);
      rmin[lid.x] = min(rmin[lid.x], rmin[lid.x + s]);
    }}
    workgroupBarrier();
    s = s >> 1u;
  }}
  if (lid.x == 0u) {{
    atomicMax(&mm[0], order(rmax[0]));
    atomicMin(&mm[1], order(rmin[0]));
  }}
}}
"#
    )
}

/// Asymmetric quantize+pack an f32 activation to i8 (4-per-u32 flat order) and
/// publish `(scale, zero)`. Reads `mm=[max,min]`, computes `scale=(max-min)/254`,
/// `zero=min+127*scale`, writes `params[0]=scale params[1]=zero`, and packs
/// `q = round((x-zero)/scale)` clamped to [-127,127] (Q8_1 style). One thread per
/// output word. Bindings: 0=x (f32), 1=mm (u32 read), 2=x_i8 (u32 write),
/// 3=params (f32 write, 2), 4=uniform{n, nwords}.
pub fn act_quant_wgsl() -> String {
    format!(
        r#"
struct U {{ n: u32, nwords: u32, _b: u32, _c: u32 }};
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> mm: array<u32>;
@group(0) @binding(2) var<storage, read_write> x_i8: array<u32>;
@group(0) @binding(3) var<storage, read_write> params: array<f32>;
@group(0) @binding(4) var<uniform> u: U;
{ORDER_FNS}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>) {{
  let w = gid.y * (ng.x * 64u) + gid.x;
  if (w >= u.nwords) {{ return; }}
  let hi = unorder(mm[0]);
  let lo = unorder(mm[1]);
  let range = hi - lo;
  let scale = select(1e-30, range / 254.0, range > 0.0);
  let zero = lo + 127.0 * scale;
  if (w == 0u) {{ params[0] = scale; params[1] = zero; }}
  let inv = 1.0 / scale;
  var packed: u32 = 0u;
  for (var e: u32 = 0u; e < 4u; e = e + 1u) {{
    let i = w * 4u + e;
    if (i < u.n) {{
      let q = clamp(round((x[i] - zero) * inv), -127.0, 127.0);
      packed = packed | ((bitcast<u32>(i32(q)) & 0xFFu) << (e * 8u));
    }}
  }}
  x_i8[w] = packed;
}}
"#
    )
}

/// Build the tiled i8 implicit-GEMM conv WGSL. Mirrors `ops::Conv2dF32`'s
/// two-level tiling (workgroup owns a `bm x bn` output tile over (Cout,
/// spatial); each thread a `tm x tn` register block) but with packed-i8 shared
/// tiles and a `dot4I8Packed` K-reduction. `bk` must be a multiple of 4.
///
/// Bindings: 0=x_i8 (packed acts, flat NCHW), 1=a_scale[1], 2=wgt_i8 (packed
/// `[Cout, ceil(K/4)]`), 3=w_scale[Cout], 4=bias[Cout], 5=out (f32), 6=uniform.
pub fn build_conv_i8_wgsl(bm: u32, bn: u32, bk: u32, tm: u32, tn: u32) -> String {
    assert!(
        bk.is_multiple_of(4),
        "bk must be a multiple of 4 for dot4I8Packed"
    );
    assert!(bm.is_multiple_of(tm) && bn.is_multiple_of(tn));
    let bk4 = bk / 4;
    let wg_x = bn / tn;
    let wg_y = bm / tm;
    let threads = wg_x * wg_y;
    let tile_a = bm * bk4; // u32 words
    let tile_b = bk4 * bn;
    let a_loads = tile_a.div_ceil(threads);
    let b_loads = tile_b.div_ceil(threads);

    let mut acc_decls = String::new();
    for i in 0..tm {
        for j in 0..tn {
            acc_decls.push_str(&format!("    var acc_{i}_{j}: i32 = 0;\n"));
        }
    }
    let mut inner = String::new();
    for i in 0..tm {
        inner.push_str(&format!(
            "            let a_{i}: u32 = tile_a[(lid.y * TM + {i}u) * BK4 + kg];\n"
        ));
    }
    for j in 0..tn {
        inner.push_str(&format!(
            "            let b_{j}: u32 = tile_b[kg * BN + lid.x * TN + {j}u];\n"
        ));
    }
    for i in 0..tm {
        for j in 0..tn {
            inner.push_str(&format!(
                "            acc_{i}_{j} = acc_{i}_{j} + dot4I8Packed(a_{i}, b_{j});\n"
            ));
        }
    }
    let mut store = String::new();
    for i in 0..tm {
        store.push_str(&format!(
            r#"    {{
        let row: u32 = bm0 + lid.y * TM + {i}u;
        if (row < u.cout) {{
            let sw: f32 = w_scale[row];
            let deq: f32 = sw * a_sc;
            // Asymmetric correction: Σ(qa*sa+za)*(qw*sw) = sa*sw*Σqa*qw + za*sw*Σqw.
            let corr: f32 = a_ze * sw * w_qsum[row] + bias[row];
            let row_base: u32 = out_base + row * m_total;
"#
        ));
        for j in 0..tn {
            store.push_str(&format!(
                r#"            {{
                let col: u32 = bn0 + lid.x * TN + {j}u;
                if (col < m_total) {{
                    out[row_base + col] = f32(acc_{i}_{j}) * deq + corr;
                }}
            }}
"#
            ));
        }
        store.push_str("        }\n    }\n");
    }

    format!(
        r#"
struct U {{
    b: u32, cin: u32, cout: u32, h_in: u32,
    w_in: u32, h_out: u32, w_out: u32, kh: u32,
    kw: u32, pad_h: u32, pad_w: u32, stride_h: u32,
    stride_w: u32, kdim: u32, kw4: u32, _p: u32,
}};

@group(0) @binding(0) var<storage, read> x_i8: array<u32>;
@group(0) @binding(1) var<storage, read> a_params: array<f32>;
@group(0) @binding(2) var<storage, read> wgt: array<u32>;
@group(0) @binding(3) var<storage, read> w_scale: array<f32>;
@group(0) @binding(4) var<storage, read> w_qsum: array<f32>;
@group(0) @binding(5) var<storage, read> bias: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<uniform> u: U;

const BM: u32 = {bm}u;
const BN: u32 = {bn}u;
const BK4: u32 = {bk4}u;
const TM: u32 = {tm}u;
const TN: u32 = {tn}u;
const THREADS: u32 = {threads}u;

var<workgroup> tile_a: array<u32, {tile_a}u>;
var<workgroup> tile_b: array<u32, {tile_b}u>;

// Signed i8 tap from the packed flat-NCHW activation buffer.
fn gather(flat: u32) -> i32 {{
    let word = x_i8[flat >> 2u];
    return extractBits(bitcast<i32>(word), (flat & 3u) * 8u, 8u);
}}

@compute @workgroup_size({wg_x}, {wg_y}, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let tid: u32 = lid.y * {wg_x}u + lid.x;
    let bm0: u32 = wid.y * BM;
    let bn0: u32 = wid.x * BN;
    let khkw: u32 = u.kh * u.kw;
    let m_total: u32 = u.h_out * u.w_out;
    let hw_in: u32 = u.h_in * u.w_in;
    let x_base: u32 = wid.z * u.cin * hw_in;
    let out_base: u32 = wid.z * u.cout * m_total;
    let a_sc: f32 = a_params[0];
    let a_ze: f32 = a_params[1];
    // Quantized value of 0.0: packed at spatially out-of-bounds taps so the
    // asymmetric correction (za*sw*Σqw over ALL taps) stays exact at padding.
    let q0: u32 = bitcast<u32>(i32(clamp(round(-a_ze / a_sc), -127.0, 127.0))) & 0xFFu;

{acc_decls}
    // One K-group (4 taps) per BK4 step; a strip of BK4 groups per outer tile.
    let num_groups: u32 = (u.kdim + 3u) / 4u;
    let num_tiles: u32 = (num_groups + BK4 - 1u) / BK4;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {{
        let g0: u32 = t * BK4; // first global K-group of this strip

        // A-tile: packed weight words [Cout, kw4]. gr = Cout row, gword = group.
        for (var s: u32 = 0u; s < {a_loads}u; s = s + 1u) {{
            let idx: u32 = s * THREADS + tid;
            if (idx < BM * BK4) {{
                let ar: u32 = idx / BK4;
                let gc: u32 = idx % BK4;
                let gr: u32 = bm0 + ar;
                let gword: u32 = g0 + gc;
                var v: u32 = 0u;
                if (gr < u.cout && gword < u.kw4) {{
                    v = wgt[gr * u.kw4 + gword];
                }}
                tile_a[idx] = v;
            }}
        }}
        // B-tile: gather 4 taps per (group, spatial col) and pack. gr = group,
        // gc = spatial -> (ho, wo). Zero at padding / beyond kdim.
        for (var s: u32 = 0u; s < {b_loads}u; s = s + 1u) {{
            let idx: u32 = s * THREADS + tid;
            if (idx < BK4 * BN) {{
                let br: u32 = idx / BN;
                let bc: u32 = idx % BN;
                let g: u32 = g0 + br;
                let col: u32 = bn0 + bc;
                var word: u32 = 0u;
                if (g < num_groups && col < m_total) {{
                    let ho: u32 = col / u.w_out;
                    let wo: u32 = col - ho * u.w_out;
                    for (var e: u32 = 0u; e < 4u; e = e + 1u) {{
                        let k: u32 = g * 4u + e;
                        if (k < u.kdim) {{
                            let ci: u32 = k / khkw;
                            let r: u32 = k - ci * khkw;
                            let dh: u32 = r / u.kw;
                            let dw: u32 = r - dh * u.kw;
                            let hi: i32 = i32(ho * u.stride_h + dh) - i32(u.pad_h);
                            let wi: i32 = i32(wo * u.stride_w + dw) - i32(u.pad_w);
                            var q: u32 = q0; // out-of-bounds tap == quantized 0.0
                            if (hi >= 0 && hi < i32(u.h_in) && wi >= 0 && wi < i32(u.w_in)) {{
                                let flat: u32 = x_base + ci * hw_in + u32(hi) * u.w_in + u32(wi);
                                q = bitcast<u32>(gather(flat)) & 0xFFu;
                            }}
                            word = word | (q << (e * 8u));
                        }}
                    }}
                }}
                tile_b[idx] = word;
            }}
        }}
        workgroupBarrier();

        for (var kg: u32 = 0u; kg < BK4; kg = kg + 1u) {{
{inner}        }}
        workgroupBarrier();
    }}

{store}}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Backend, Binding, PowerPreference, WgpuBackend, WgpuConfig};
    use crate::mem::VramCategory;

    fn uni(vals: &[u32]) -> Vec<u8> {
        let mut b: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        while !b.len().is_multiple_of(16) {
            b.push(0);
        }
        b
    }

    /// GPU i8 conv (maxabs -> quant -> conv_i8) vs a host reference doing the
    /// identical symmetric-quant math, plus the rel error vs the true f32 conv
    /// (quantization noise, not a bug). Covers a 3x3-pad1 multi-tile case and a
    /// 1x1 case.
    // Skipped on macOS: the GitHub macOS runner's virtualized Metal rejects the
    // conv_i8 pipeline at submit ("ComputePipeline is invalid"). It is not a
    // thinfer bug: naga emits valid MSL for this shader at every lang version
    // (1.2 through 3.1), and `dot4I8Packed` itself compiles and runs on that
    // Metal. The runner's paravirtual GPU chokes on the larger shader; real
    // Apple Silicon does not. Runs normally on every other backend.
    #[cfg_attr(
        target_os = "macos",
        ignore = "GH macOS runner's virtualized Metal rejects conv_i8 (CI-runner limitation, not real Apple Silicon)"
    )]
    #[test]
    fn conv_i8_matches_host_quant() {
        pollster::block_on(async {
            let cfg = WgpuConfig {
                power_preference: PowerPreference::HighPerformance,
                ..Default::default()
            };
            let Ok(backend) = WgpuBackend::new_with_config(cfg).await else {
                eprintln!("skip conv_i8_matches_host_quant: no GPU adapter");
                return;
            };
            let backend = std::sync::Arc::new(backend);
            for &(cin, cout, h, w, kh, kw, pad, stride) in &[
                (
                    20usize, 70usize, 10usize, 10usize, 3usize, 3usize, 1usize, 1usize,
                ),
                (64, 32, 8, 8, 1, 1, 0, 1),
                (16, 48, 12, 12, 4, 4, 1, 2),
            ] {
                let ho = (h + 2 * pad - kh) / stride + 1;
                let wo = (w + 2 * pad - kw) / stride + 1;
                let k = cin * kh * kw;
                let n = cin * h * w;
                // Deterministic pseudo-random inputs.
                let rf = |i: usize, s: usize| {
                    (((i * 2654435761 + s * 40503) % 2039) as f32 / 2039.0) * 2.0 - 1.0
                };
                let x: Vec<f32> = (0..n).map(|i| rf(i, 1)).collect();
                let wt: Vec<f32> = (0..cout * k).map(|i| rf(i, 2) * 0.3).collect();
                let bias: Vec<f32> = (0..cout).map(|i| rf(i, 3) * 0.1).collect();

                // Host reference: asymmetric quantize, integer GEMM, dequant
                // (matches the GPU: out = acc*sa*sw + za*sw*qsum_w + bias).
                let amax = x.iter().fold(f32::MIN, |m, &v| m.max(v));
                let amin = x.iter().fold(f32::MAX, |m, &v| m.min(v));
                let range = amax - amin;
                let a_scale = if range > 0.0 { range / 254.0 } else { 1e-30 };
                let a_zero = amin + 127.0 * a_scale;
                let q0 = (-a_zero / a_scale).round().clamp(-127.0, 127.0) as i64;
                let qx: Vec<i32> = x
                    .iter()
                    .map(|&v| ((v - a_zero) / a_scale).round().clamp(-127.0, 127.0) as i32)
                    .collect();
                let (packed_w, w_scale, w_qsum) = quantize_weight(&wt, cout, k);
                let mut ref_q = vec![0f32; cout * ho * wo];
                let mut ref_f32 = vec![0f32; cout * ho * wo];
                for co in 0..cout {
                    for oy in 0..ho {
                        for ox in 0..wo {
                            let mut acc_i: i64 = 0;
                            let mut acc_f: f32 = 0.0;
                            for ci in 0..cin {
                                for r in 0..kh {
                                    for s in 0..kw {
                                        let kk = (ci * kh + r) * kw + s;
                                        let wq = (wt[co * k + kk] / w_scale[co])
                                            .round()
                                            .clamp(-127.0, 127.0)
                                            as i64;
                                        let iy = oy * stride + r;
                                        let ix = ox * stride + s;
                                        // Out-of-bounds tap == quantized 0.0 (q0),
                                        // matching the GPU B-loader.
                                        let (xq, xf) = if iy < pad
                                            || ix < pad
                                            || iy - pad >= h
                                            || ix - pad >= w
                                        {
                                            (q0, 0.0)
                                        } else {
                                            let p = (ci * h + (iy - pad)) * w + (ix - pad);
                                            (qx[p] as i64, x[p])
                                        };
                                        acc_i += wq * xq;
                                        acc_f += wt[co * k + kk] * xf;
                                    }
                                }
                            }
                            let idx = (co * ho + oy) * wo + ox;
                            ref_q[idx] = acc_i as f32 * w_scale[co] * a_scale
                                + a_zero * w_scale[co] * w_qsum[co]
                                + bias[co];
                            ref_f32[idx] = acc_f + bias[co];
                        }
                    }
                }

                // GPU pipeline.
                let p_max = backend
                    .create_pipeline("t_minmax", &act_minmax_wgsl(), "main", &act_maxabs_layout())
                    .await
                    .unwrap();
                let p_q = backend
                    .create_pipeline("t_quant", &act_quant_wgsl(), "main", &act_quant_layout())
                    .await
                    .unwrap();
                let conv_src = build_conv_i8_wgsl(64, 64, 32, 4, 4);
                let p_conv = backend
                    .create_pipeline("t_conv_i8", &conv_src, "main", &conv_i8_layout())
                    .await
                    .unwrap();

                let nwords = n.div_ceil(4);
                let mk = |bytes: usize| {
                    backend
                        .allocate_in(bytes.max(4) as u64, VramCategory::Workspace)
                        .unwrap()
                };
                let b_x = mk(n * 4);
                let b_mm = mk(8); // [max, min] order-preserving u32
                let b_xi8 = mk(nwords * 4);
                let b_params = mk(8); // [scale, zero]
                let b_w = mk(packed_w.len() * 4);
                let b_wsc = mk(cout * 4);
                let b_wqsum = mk(cout * 4);
                let b_bias = mk(cout * 4);
                let b_out = mk(cout * ho * wo * 4);
                let b_umax = mk(16);
                let b_uq = mk(16);
                let b_uc = mk(64);
                backend
                    .write_buffer(b_x, 0, bytemuck::cast_slice(&x))
                    .unwrap();
                backend
                    .write_buffer(b_mm, 0, bytemuck::cast_slice(&[0u32, 0xFFFF_FFFFu32]))
                    .unwrap();
                backend
                    .write_buffer(b_w, 0, bytemuck::cast_slice(&packed_w))
                    .unwrap();
                backend
                    .write_buffer(b_wsc, 0, bytemuck::cast_slice(&w_scale))
                    .unwrap();
                backend
                    .write_buffer(b_wqsum, 0, bytemuck::cast_slice(&w_qsum))
                    .unwrap();
                backend
                    .write_buffer(b_bias, 0, bytemuck::cast_slice(&bias))
                    .unwrap();
                backend.write_buffer(b_umax, 0, &uni(&[n as u32])).unwrap();
                backend
                    .write_buffer(b_uq, 0, &uni(&[n as u32, nwords as u32]))
                    .unwrap();
                backend
                    .write_buffer(
                        b_uc,
                        0,
                        &uni(&[
                            1,
                            cin as u32,
                            cout as u32,
                            h as u32,
                            w as u32,
                            ho as u32,
                            wo as u32,
                            kh as u32,
                            kw as u32,
                            pad as u32,
                            pad as u32,
                            stride as u32,
                            stride as u32,
                            k as u32,
                            k.div_ceil(4) as u32,
                            0,
                        ]),
                    )
                    .unwrap();

                let mut enc = backend.create_command_encoder();
                let bind = |slot: u32, id, len: usize| Binding {
                    slot,
                    buffer: id,
                    offset: 0,
                    size: (len.max(4)) as u64,
                };
                backend
                    .dispatch(
                        &mut enc,
                        &p_max,
                        &[bind(0, b_x, n * 4), bind(1, b_mm, 8), bind(2, b_umax, 16)],
                        crate::ops::linear_workgroups(n as u32, 256),
                    )
                    .unwrap();
                backend
                    .dispatch(
                        &mut enc,
                        &p_q,
                        &[
                            bind(0, b_x, n * 4),
                            bind(1, b_mm, 8),
                            bind(2, b_xi8, nwords * 4),
                            bind(3, b_params, 8),
                            bind(4, b_uq, 16),
                        ],
                        crate::ops::linear_workgroups(nwords as u32, 64),
                    )
                    .unwrap();
                backend
                    .dispatch(
                        &mut enc,
                        &p_conv,
                        &[
                            bind(0, b_xi8, nwords * 4),
                            bind(1, b_params, 8),
                            bind(2, b_w, packed_w.len() * 4),
                            bind(3, b_wsc, cout * 4),
                            bind(4, b_wqsum, cout * 4),
                            bind(5, b_bias, cout * 4),
                            bind(6, b_out, cout * ho * wo * 4),
                            bind(7, b_uc, 64),
                        ],
                        [(wo * ho).div_ceil(64) as u32, cout.div_ceil(64) as u32, 1],
                    )
                    .unwrap();
                backend.submit(enc).await.unwrap();
                let raw = backend
                    .read_buffer(b_out, 0, (cout * ho * wo * 4) as u64)
                    .await
                    .unwrap();
                let got: Vec<f32> = bytemuck::cast_slice(&raw).to_vec();

                // GPU must match the host quantized reference to f32 precision.
                let mut max_q = 0f32;
                let mut ref_range = 0f32;
                let mut sq_err = 0f64;
                let mut sq_ref = 0f64;
                for i in 0..got.len() {
                    max_q = max_q.max((got[i] - ref_q[i]).abs());
                    ref_range = ref_range.max(ref_q[i].abs());
                    sq_err += (got[i] - ref_f32[i]).powi(2) as f64;
                    sq_ref += ref_f32[i].powi(2) as f64;
                }
                let rel_vs_f32 = (sq_err / sq_ref).sqrt();
                eprintln!(
                    "[conv_i8] {cin}x{cout} {h}x{w} k{kh} s{stride}: max|gpu-hostq|={max_q:.3e} \
                     (range {ref_range:.2}) rel_rmse_vs_f32={rel_vs_f32:.3e}"
                );
                assert!(
                    max_q <= 1e-3 * (ref_range + 1e-6),
                    "GPU i8 conv diverged from host quant ref: {max_q:.3e}"
                );
                assert!(
                    rel_vs_f32 < 0.1,
                    "quant error implausibly large: {rel_vs_f32:.3e}"
                );
                for id in [
                    b_x, b_mm, b_xi8, b_params, b_w, b_wsc, b_wqsum, b_bias, b_out, b_umax, b_uq,
                    b_uc,
                ] {
                    backend.free(id);
                }
            }
        });
    }
}
