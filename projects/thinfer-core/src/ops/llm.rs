//! Portable (WGSL) kernels for autoregressive LLM decode and chunked prefill.
//!
//! Activations are f32, row-major `[rows, cols]`; weights stay in their GGUF
//! encoding and are dequantized in-register. Every kernel takes its geometry
//! from a uniform, so one pipeline per (kernel, weight format) serves every
//! layer and model. Bindings are listed per kernel in its `*_LAYOUT`.
//!
//! The decode hot path is weight-bandwidth bound: [`gemv_wgsl`] streams each
//! weight row once per pair and reduces in-workgroup. The same kernel serves
//! MoE experts by reading weight slots through an indirection table (one
//! `(slot, x_row)` pair per output row), so routed experts never round-trip the
//! host per token.

use crate::backend::{BindingKind, BindingLayout};
use crate::quant::QuantKind;

/// A weight matrix encoding the LLM kernels read in place.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LlmWeight {
    F32,
    /// Raw bf16 (safetensors checkpoints), widened in-register.
    Bf16,
    Quant(QuantKind),
}

impl LlmWeight {
    /// Bytes of one weight row of `k` elements.
    pub fn row_bytes(self, k: u64) -> u64 {
        match self {
            Self::F32 => k * 4,
            Self::Bf16 => k * 2,
            Self::Quant(q) => q.bytes_for_elements(k),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
            Self::Quant(QuantKind::Q8_0) => "q8_0",
            Self::Quant(QuantKind::Q4_0) => "q4_0",
            Self::Quant(QuantKind::Q5_0) => "q5_0",
            Self::Quant(QuantKind::Q5_1) => "q5_1",
            Self::Quant(QuantKind::Q4_K) => "q4_k",
            Self::Quant(QuantKind::Q5_K) => "q5_k",
            Self::Quant(QuantKind::Q6_K) => "q6_k",
        }
    }
}

const fn layout<const N: usize>(kinds: [BindingKind; N]) -> [BindingLayout; N] {
    let mut out = [BindingLayout {
        slot: 0,
        kind: BindingKind::Uniform,
    }; N];
    let mut i = 0;
    while i < N {
        out[i] = BindingLayout {
            slot: i as u32,
            kind: kinds[i],
        };
        i += 1;
    }
    out
}

use BindingKind::{StorageRead as R, StorageReadWrite as W, Uniform as U};

/// Shared byte/half loaders over a `array<u32>` weight binding `w`.
const LOADERS: &str = r#"
fn ld_u32(off: u32) -> u32 {
    let i = off >> 2u;
    let sh = (off & 3u) * 8u;
    if (sh == 0u) { return w[i]; }
    return (w[i] >> sh) | (w[i + 1u] << (32u - sh));
}
fn ld_u8(off: u32) -> u32 {
    return (w[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu;
}
fn ld_f16(off: u32) -> f32 {
    let word = w[off >> 2u];
    let h = select(word & 0xFFFFu, word >> 16u, (off & 2u) != 0u);
    return unpack2x16float(h).x;
}
fn i8_of(b: u32) -> f32 {
    return f32(i32(b << 24u) >> 24u);
}
"#;

/// Per-format `dot8(base, c, l)`: the dot of the 8 weight elements
/// `[c * 256 + l * 8, +8)` of the row starting at byte `base` with `xa`/`xb`
/// (x elements in the same order). `c` = 256-element chunk, `l` = lane 0..31.
fn dot8_fn(fmt: LlmWeight) -> &'static str {
    match fmt {
        LlmWeight::F32 => {
            r#"
fn dot8(base: u32, c: u32, l: u32, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
    let o = (base >> 2u) + c * 256u + l * 8u;
    let wa = vec4<f32>(bitcast<f32>(w[o]), bitcast<f32>(w[o + 1u]), bitcast<f32>(w[o + 2u]), bitcast<f32>(w[o + 3u]));
    let wb = vec4<f32>(bitcast<f32>(w[o + 4u]), bitcast<f32>(w[o + 5u]), bitcast<f32>(w[o + 6u]), bitcast<f32>(w[o + 7u]));
    return dot(wa, xa) + dot(wb, xb);
}
"#
        }
        LlmWeight::Bf16 => {
            r#"
fn bf2(v: u32) -> vec2<f32> {
    return vec2<f32>(bitcast<f32>(v << 16u), bitcast<f32>(v & 0xFFFF0000u));
}
fn dot8(base: u32, c: u32, l: u32, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
    let o = (base >> 2u) + c * 128u + l * 4u;
    let wa = vec4<f32>(bf2(w[o]), bf2(w[o + 1u]));
    let wb = vec4<f32>(bf2(w[o + 2u]), bf2(w[o + 3u]));
    return dot(wa, xa) + dot(wb, xb);
}
"#
        }
        LlmWeight::Quant(QuantKind::Q8_0) => {
            r#"
fn dot8(base: u32, c: u32, l: u32, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
    let blk = base + (c * 8u + (l >> 2u)) * 34u;
    let d = ld_f16(blk);
    let q0 = ld_u32(blk + 2u + (l & 3u) * 8u);
    let q1 = ld_u32(blk + 6u + (l & 3u) * 8u);
    let wa = vec4<f32>(i8_of(q0 & 0xFFu), i8_of((q0 >> 8u) & 0xFFu), i8_of((q0 >> 16u) & 0xFFu), i8_of(q0 >> 24u));
    let wb = vec4<f32>(i8_of(q1 & 0xFFu), i8_of((q1 >> 8u) & 0xFFu), i8_of((q1 >> 16u) & 0xFFu), i8_of(q1 >> 24u));
    return d * (dot(wa, xa) + dot(wb, xb));
}
"#
        }
        LlmWeight::Quant(QuantKind::Q4_0) => {
            r#"
fn dot8(base: u32, c: u32, l: u32, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
    // 8 lanes per 32-elem block would split nibble pairs; Q4_0 stores elem i
    // in byte i%16 (low nibble i<16, high i>=16).
    let blk = base + (c * 8u + (l >> 2u)) * 18u;
    let d = ld_f16(blk);
    let o = (l & 3u) * 8u;
    let byte0 = select(o, o - 16u, o >= 16u);
    let sh = select(0u, 4u, o >= 16u);
    let q0 = ld_u32(blk + 2u + byte0);
    let q1 = ld_u32(blk + 6u + byte0);
    let wa = vec4<f32>(f32((q0 >> sh) & 15u), f32((q0 >> (8u + sh)) & 15u), f32((q0 >> (16u + sh)) & 15u), f32((q0 >> (24u + sh)) & 15u)) - 8.0;
    let wb = vec4<f32>(f32((q1 >> sh) & 15u), f32((q1 >> (8u + sh)) & 15u), f32((q1 >> (16u + sh)) & 15u), f32((q1 >> (24u + sh)) & 15u)) - 8.0;
    return d * (dot(wa, xa) + dot(wb, xb));
}
"#
        }
        LlmWeight::Quant(QuantKind::Q5_0) => {
            r#"
fn dot8(base: u32, c: u32, l: u32, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
    // Q5_1's element order without the min; f = (q - 16) * d.
    let blk = base + (c * 8u + (l >> 2u)) * 22u;
    let d = ld_f16(blk);
    let o = (l & 3u) * 8u;
    let qh = ld_u32(blk + 2u) >> o;
    let byte0 = select(o, o - 16u, o >= 16u);
    let sh = select(0u, 4u, o >= 16u);
    let q0 = ld_u32(blk + 6u + byte0);
    let q1 = ld_u32(blk + 10u + byte0);
    let wa = vec4<f32>(
        f32(((q0 >> sh) & 15u) | ((qh & 1u) << 4u)),
        f32(((q0 >> (8u + sh)) & 15u) | (((qh >> 1u) & 1u) << 4u)),
        f32(((q0 >> (16u + sh)) & 15u) | (((qh >> 2u) & 1u) << 4u)),
        f32(((q0 >> (24u + sh)) & 15u) | (((qh >> 3u) & 1u) << 4u))) - 16.0;
    let wb = vec4<f32>(
        f32(((q1 >> sh) & 15u) | (((qh >> 4u) & 1u) << 4u)),
        f32(((q1 >> (8u + sh)) & 15u) | (((qh >> 5u) & 1u) << 4u)),
        f32(((q1 >> (16u + sh)) & 15u) | (((qh >> 6u) & 1u) << 4u)),
        f32(((q1 >> (24u + sh)) & 15u) | (((qh >> 7u) & 1u) << 4u))) - 16.0;
    return d * (dot(wa, xa) + dot(wb, xb));
}
"#
        }
        LlmWeight::Quant(QuantKind::Q5_1) => {
            r#"
fn dot8(base: u32, c: u32, l: u32, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
    // Element i of a block: nibble of byte i%16 (low i<16, high i>=16), 5th
    // bit = qh bit i; f = q * d + m.
    let blk = base + (c * 8u + (l >> 2u)) * 24u;
    let d = ld_f16(blk);
    let m = ld_f16(blk + 2u);
    let o = (l & 3u) * 8u;
    let qh = ld_u32(blk + 4u) >> o;
    let byte0 = select(o, o - 16u, o >= 16u);
    let sh = select(0u, 4u, o >= 16u);
    let q0 = ld_u32(blk + 8u + byte0);
    let q1 = ld_u32(blk + 12u + byte0);
    let wa = vec4<f32>(
        f32(((q0 >> sh) & 15u) | ((qh & 1u) << 4u)),
        f32(((q0 >> (8u + sh)) & 15u) | (((qh >> 1u) & 1u) << 4u)),
        f32(((q0 >> (16u + sh)) & 15u) | (((qh >> 2u) & 1u) << 4u)),
        f32(((q0 >> (24u + sh)) & 15u) | (((qh >> 3u) & 1u) << 4u)));
    let wb = vec4<f32>(
        f32(((q1 >> sh) & 15u) | (((qh >> 4u) & 1u) << 4u)),
        f32(((q1 >> (8u + sh)) & 15u) | (((qh >> 5u) & 1u) << 4u)),
        f32(((q1 >> (16u + sh)) & 15u) | (((qh >> 6u) & 1u) << 4u)),
        f32(((q1 >> (24u + sh)) & 15u) | (((qh >> 7u) & 1u) << 4u)));
    let one = vec4<f32>(1.0);
    return d * (dot(wa, xa) + dot(wb, xb)) + m * (dot(one, xa) + dot(one, xb));
}
"#
        }
        LlmWeight::Quant(QuantKind::Q4_K) => {
            r#"
fn scale_min_k4(blk: u32, j: u32) -> vec2<f32> {
    if (j < 4u) {
        return vec2<f32>(f32(ld_u8(blk + 4u + j) & 63u), f32(ld_u8(blk + 8u + j) & 63u));
    }
    let a = ld_u8(blk + 8u + j);
    let lo = ld_u8(blk + j);
    let hi = ld_u8(blk + 4u + j);
    return vec2<f32>(f32((a & 0xFu) | ((lo >> 6u) << 4u)), f32((a >> 4u) | ((hi >> 6u) << 4u)));
}
fn dot8(base: u32, c: u32, l: u32, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
    let blk = base + c * 144u;
    let d = ld_f16(blk);
    let dmin = ld_f16(blk + 2u);
    let j = l >> 2u;
    let sm = scale_min_k4(blk, j);
    let sh = (j & 1u) * 4u;
    let qo = blk + 16u + (j >> 1u) * 32u + (l & 3u) * 8u;
    let q0 = w[qo >> 2u];
    let q1 = w[(qo >> 2u) + 1u];
    let wa = vec4<f32>(f32((q0 >> sh) & 15u), f32((q0 >> (8u + sh)) & 15u), f32((q0 >> (16u + sh)) & 15u), f32((q0 >> (24u + sh)) & 15u));
    let wb = vec4<f32>(f32((q1 >> sh) & 15u), f32((q1 >> (8u + sh)) & 15u), f32((q1 >> (16u + sh)) & 15u), f32((q1 >> (24u + sh)) & 15u));
    let xs = dot(xa, vec4<f32>(1.0)) + dot(xb, vec4<f32>(1.0));
    return d * sm.x * (dot(wa, xa) + dot(wb, xb)) - dmin * sm.y * xs;
}
"#
        }
        LlmWeight::Quant(QuantKind::Q5_K) => {
            r#"
fn scale_min_k4(blk: u32, j: u32) -> vec2<f32> {
    if (j < 4u) {
        return vec2<f32>(f32(ld_u8(blk + 4u + j) & 63u), f32(ld_u8(blk + 8u + j) & 63u));
    }
    let a = ld_u8(blk + 8u + j);
    let lo = ld_u8(blk + j);
    let hi = ld_u8(blk + 4u + j);
    return vec2<f32>(f32((a & 0xFu) | ((lo >> 6u) << 4u)), f32((a >> 4u) | ((hi >> 6u) << 4u)));
}
fn dot8(base: u32, c: u32, l: u32, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
    let blk = base + c * 176u;
    let d = ld_f16(blk);
    let dmin = ld_f16(blk + 2u);
    let j = l >> 2u;
    let sm = scale_min_k4(blk, j);
    let sh = (j & 1u) * 4u;
    let o = (l & 3u) * 8u;
    let qo = blk + 48u + (j >> 1u) * 32u + o;
    let q0 = w[qo >> 2u];
    let q1 = w[(qo >> 2u) + 1u];
    let h0 = w[(blk + 16u + o) >> 2u] >> j;
    let h1 = w[((blk + 16u + o) >> 2u) + 1u] >> j;
    let wa = vec4<f32>(
        f32(((q0 >> sh) & 15u) | ((h0 & 1u) << 4u)),
        f32(((q0 >> (8u + sh)) & 15u) | (((h0 >> 8u) & 1u) << 4u)),
        f32(((q0 >> (16u + sh)) & 15u) | (((h0 >> 16u) & 1u) << 4u)),
        f32(((q0 >> (24u + sh)) & 15u) | (((h0 >> 24u) & 1u) << 4u)));
    let wb = vec4<f32>(
        f32(((q1 >> sh) & 15u) | ((h1 & 1u) << 4u)),
        f32(((q1 >> (8u + sh)) & 15u) | (((h1 >> 8u) & 1u) << 4u)),
        f32(((q1 >> (16u + sh)) & 15u) | (((h1 >> 16u) & 1u) << 4u)),
        f32(((q1 >> (24u + sh)) & 15u) | (((h1 >> 24u) & 1u) << 4u)));
    let xs = dot(xa, vec4<f32>(1.0)) + dot(xb, vec4<f32>(1.0));
    return d * sm.x * (dot(wa, xa) + dot(wb, xb)) - dmin * sm.y * xs;
}
"#
        }
        LlmWeight::Quant(QuantKind::Q6_K) => {
            r#"
fn dot8(base: u32, c: u32, l: u32, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
    let blk = base + c * 210u;
    let half = l >> 4u;
    let bk = (l >> 2u) & 3u;
    let jj = (l & 3u) * 8u;
    let ql = blk + half * 64u + (bk & 1u) * 32u + jj;
    let qh = blk + 128u + half * 32u + jj;
    let nsh = select(4u, 0u, bk < 2u);
    let hsh = bk * 2u;
    let l0 = ld_u32(ql);
    let l1 = ld_u32(ql + 4u);
    let g0 = ld_u32(qh);
    let g1 = ld_u32(qh + 4u);
    let is = half * 8u + bk * 2u + (jj >> 4u);
    let sc = i8_of(ld_u8(blk + 192u + is)) * ld_f16(blk + 208u);
    let wa = vec4<f32>(
        f32(i32(((l0 >> nsh) & 15u) | (((g0 >> hsh) & 3u) << 4u)) - 32),
        f32(i32(((l0 >> (8u + nsh)) & 15u) | (((g0 >> (8u + hsh)) & 3u) << 4u)) - 32),
        f32(i32(((l0 >> (16u + nsh)) & 15u) | (((g0 >> (16u + hsh)) & 3u) << 4u)) - 32),
        f32(i32(((l0 >> (24u + nsh)) & 15u) | (((g0 >> (24u + hsh)) & 3u) << 4u)) - 32));
    let wb = vec4<f32>(
        f32(i32(((l1 >> nsh) & 15u) | (((g1 >> hsh) & 3u) << 4u)) - 32),
        f32(i32(((l1 >> (8u + nsh)) & 15u) | (((g1 >> (8u + hsh)) & 3u) << 4u)) - 32),
        f32(i32(((l1 >> (16u + nsh)) & 15u) | (((g1 >> (16u + hsh)) & 3u) << 4u)) - 32),
        f32(i32(((l1 >> (24u + nsh)) & 15u) | (((g1 >> (24u + hsh)) & 3u) << 4u)) - 32));
    return sc * (dot(wa, xa) + dot(wb, xb));
}
"#
        }
    }
}

/// Uniform of [`gemv_wgsl`] (8 x u32).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GemvUniform {
    /// Output columns (weight rows).
    pub n: u32,
    /// Reduction length (multiple of 256).
    pub k: u32,
    /// x row stride and element offset.
    pub x_stride: u32,
    pub x_off: u32,
    /// y row stride and element offset.
    pub y_stride: u32,
    pub y_off: u32,
    /// Bytes between consecutive weight slots (indirect mode).
    pub slot_bytes: u32,
    /// Bit 0: indirect triples (`table[3p..3p+3]` = slot, x row, y row);
    /// else slot 0 and x / y row `p`. Bit 1: add `res` at the output index.
    pub flags: u32,
}

pub const GEMV_INDIRECT: u32 = 1;
pub const GEMV_RESIDUAL: u32 = 2;
/// Output rows per workgroup (one 32-lane group each).
pub const GEMV_ROWS_PER_WG: u32 = 8;

/// `w, x, y, uniform, table, res`.
pub const GEMV_LAYOUT: [BindingLayout; 6] = layout([R, R, W, U, R, R]);

/// `y[yrow(p), n] = W_slot(p)[n, :] . x[xrow(p), :]` (+ residual). Dispatch
/// `[ceil(n / 8), pairs, 1]`. `k % 32 == 0` (K-quant rows are whole 256
/// super-blocks); a trailing partial 256-chunk is summed by the lanes it
/// covers. The weight row stride is `row_bytes(k)` (rows packed back to
/// back, slot after slot).
pub fn gemv_wgsl(fmt: LlmWeight) -> String {
    let rb = match fmt {
        LlmWeight::F32 => "u.k * 4u".to_string(),
        LlmWeight::Bf16 => "u.k * 2u".to_string(),
        LlmWeight::Quant(q) => format!("(u.k / {}u) * {}u", q.block_size(), q.bytes_per_block()),
    };
    format!(
        r#"// gemv {label}
struct U {{ n: u32, k: u32, x_stride: u32, x_off: u32, y_stride: u32, y_off: u32, slot_bytes: u32, flags: u32 }};
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
@group(0) @binding(4) var<storage, read> table: array<u32>;
@group(0) @binding(5) var<storage, read> res: array<f32>;
var<workgroup> red: array<f32, 256>;
{LOADERS}
{dot8}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {{
    let grp = lid >> 5u;
    let l = lid & 31u;
    let n = wg.x * 8u + grp;
    let p = wg.y;
    var slot = 0u;
    var xrow = p;
    var yrow = p;
    if ((u.flags & 1u) != 0u) {{
        slot = table[3u * p];
        xrow = table[3u * p + 1u];
        yrow = table[3u * p + 2u];
    }}
    var acc = 0.0;
    if (n < u.n) {{
        let base = slot * u.slot_bytes + n * ({rb});
        let xb0 = xrow * u.x_stride + u.x_off;
        let chunks = u.k / 256u;
        for (var c = 0u; c < chunks; c = c + 1u) {{
            let xo = xb0 + c * 256u + l * 8u;
            let xa = vec4<f32>(x[xo], x[xo + 1u], x[xo + 2u], x[xo + 3u]);
            let xb = vec4<f32>(x[xo + 4u], x[xo + 5u], x[xo + 6u], x[xo + 7u]);
            acc = acc + dot8(base, c, l, xa, xb);
        }}
        if (l * 8u < u.k - chunks * 256u) {{
            let xo = xb0 + chunks * 256u + l * 8u;
            let xa = vec4<f32>(x[xo], x[xo + 1u], x[xo + 2u], x[xo + 3u]);
            let xb = vec4<f32>(x[xo + 4u], x[xo + 5u], x[xo + 6u], x[xo + 7u]);
            acc = acc + dot8(base, chunks, l, xa, xb);
        }}
    }}
    red[lid] = acc;
    workgroupBarrier();
    for (var s = 16u; s > 0u; s = s >> 1u) {{
        if (l < s) {{ red[lid] = red[lid] + red[lid + s]; }}
        workgroupBarrier();
    }}
    if (l == 0u && n < u.n) {{
        let o = yrow * u.y_stride + u.y_off + n;
        var v = red[lid];
        if ((u.flags & 2u) != 0u) {{ v = v + res[o]; }}
        y[o] = v;
    }}
}}
"#,
        label = fmt.label(),
        dot8 = dot8_fn(fmt),
    )
}

/// Uniform of [`RMSNORM_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RmsNormUniform {
    pub rows: u32,
    pub dim: u32,
    pub x_stride: u32,
    pub x_off: u32,
    pub y_stride: u32,
    pub y_off: u32,
    pub eps: f32,
    /// Bit 0: no weight (plain normalize). Bit 1: add `res[row]` after
    /// scaling (`y = res + norm(x) * w`). Bit 2: then multiply by `w[dim]`
    /// (a per-layer output scale stored after the weights).
    pub flags: u32,
}

pub const RMSNORM_NO_WEIGHT: u32 = 1;
pub const RMSNORM_RESIDUAL: u32 = 2;
pub const RMSNORM_OUT_SCALE: u32 = 4;

/// `x, w, y, uniform, res`. One workgroup per row; dispatch `[rows, 1, 1]`.
pub const RMSNORM_LAYOUT: [BindingLayout; 5] = layout([R, R, W, U, R]);

pub const RMSNORM_WGSL: &str = r#"// rmsnorm f32
struct U { rows: u32, dim: u32, x_stride: u32, x_off: u32, y_stride: u32, y_off: u32, eps: f32, flags: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
@group(0) @binding(4) var<storage, read> res: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let r = wg.x + wg.y * 65535u;
    if (r >= u.rows) { return; }
    let xb = r * u.x_stride + u.x_off;
    var ss = 0.0;
    for (var i = lid; i < u.dim; i = i + 256u) { let v = x[xb + i]; ss = ss + v * v; }
    red[lid] = ss;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) { red[lid] = red[lid] + red[lid + s]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(red[0] / f32(u.dim) + u.eps);
    let yb = r * u.y_stride + u.y_off;
    for (var i = lid; i < u.dim; i = i + 256u) {
        var v = x[xb + i] * inv;
        if ((u.flags & 1u) == 0u) { v = v * wt[i]; }
        if ((u.flags & 2u) != 0u) { v = v + res[yb + i]; }
        if ((u.flags & 4u) != 0u) { v = v * wt[u.dim]; }
        y[yb + i] = v;
    }
}
"#;

/// Uniform of [`ROPE_WGSL`]: NeoX (rotate-half) rope over the first
/// `2 * half` dims of each head, in place.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RopeUniform {
    pub rows: u32,
    pub heads: u32,
    pub row_stride: u32,
    pub head_stride: u32,
    pub off: u32,
    pub half: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// `x (rw), cos_sin [rows, half, 2], uniform`. Dispatch `[ceil(rows*heads*half/256), 1, 1]`.
pub const ROPE_LAYOUT: [BindingLayout; 3] = layout([W, R, U]);

pub const ROPE_WGSL: &str = r#"// rope neox partial
struct U { rows: u32, heads: u32, row_stride: u32, head_stride: u32, off: u32, half: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> cs: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = gid.x + gid.y * nwg.x * 256u;
    let total = u.rows * u.heads * u.half;
    if (id >= total) { return; }
    let i = id % u.half;
    let rh = id / u.half;
    let h = rh % u.heads;
    let r = rh / u.heads;
    let b = r * u.row_stride + h * u.head_stride + u.off;
    let c = cs[(r * u.half + i) * 2u];
    let s = cs[(r * u.half + i) * 2u + 1u];
    let x0 = x[b + i];
    let x1 = x[b + i + u.half];
    x[b + i] = x0 * c - x1 * s;
    x[b + i + u.half] = x1 * c + x0 * s;
}
"#;

/// Uniform of [`KV_STORE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct KvStoreUniform {
    pub rows: u32,
    /// `kv_heads * head_dim` (even).
    pub width: u32,
    pub src_stride: u32,
    pub src_off: u32,
    /// Destination row in the cache of source row 0 (absolute position).
    pub pos0: u32,
    /// Ring capacity in rows (position `p` lives in row `p % ring`); 0 =
    /// linear (row = position).
    pub ring: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// `src f32, cache u32 (f16 pairs) rw, uniform`. Writes rows
/// `[pos0, pos0 + rows)`. Dispatch `[ceil(rows * width / 2 / 256), 1, 1]`.
pub const KV_STORE_LAYOUT: [BindingLayout; 3] = layout([R, W, U]);

pub const KV_STORE_WGSL: &str = r#"// kv store f16
struct U { rows: u32, width: u32, src_stride: u32, src_off: u32, pos0: u32, ring: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> cache: array<u32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = gid.x + gid.y * nwg.x * 256u;
    let hw = u.width / 2u;
    if (id >= u.rows * hw) { return; }
    let r = id / hw;
    let i = id % hw;
    let s = r * u.src_stride + u.src_off + 2u * i;
    var row = u.pos0 + r;
    if (u.ring > 0u) { row = row % u.ring; }
    cache[row * hw + i] = pack2x16float(vec2<f32>(src[s], src[s + 1u]));
}
"#;

/// Uniform of [`attention_wgsl`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AttnUniform {
    pub rows: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// Absolute position of query row 0.
    pub pos0: u32,
    /// Sliding window (keys `> qpos - window`); 0 = full causal.
    pub window: u32,
    pub scale: f32,
    /// Bit 0: multiply the output by `sigmoid(gate)`.
    pub flags: u32,
    pub q_stride: u32,
    pub q_head_stride: u32,
    pub q_off: u32,
    pub out_stride: u32,
    pub gate_stride: u32,
    pub gate_head_stride: u32,
    pub gate_off: u32,
    /// KV ring capacity in rows (key `p` in row `p % ring`); 0 = linear.
    /// With a window, `ring` must exceed it by the forward's row count.
    pub ring: u32,
}

pub const ATTN_GATE: u32 = 1;

/// `q f32, kcache u32, vcache u32, gate f32, out f32 rw, uniform`. One
/// workgroup per (row, head); dispatch `[heads, rows, 1]`.
pub const ATTN_LAYOUT: [BindingLayout; 6] = layout([R, R, R, R, W, U]);

/// Causal (optionally sliding-window) GQA attention over an f16 KV cache,
/// online softmax in f32, 256 keys per block. `head_dim <= 256 * DPT`.
pub fn attention_wgsl(dims_per_thread: u32) -> String {
    format!(
        r#"// attention dpt={dpt}
const DPT: u32 = {dpt}u;
struct U {{ rows: u32, heads: u32, kv_heads: u32, head_dim: u32, pos0: u32, window: u32, scale: f32, flags: u32,
           q_stride: u32, q_head_stride: u32, q_off: u32, out_stride: u32, gate_stride: u32, gate_head_stride: u32, gate_off: u32,
           ring: u32 }};
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<u32>;
@group(0) @binding(2) var<storage, read> vc: array<u32>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> u: U;
var<workgroup> qs: array<f32, {qmax}>;
var<workgroup> pr: array<f32, 256>;
var<workgroup> red: array<f32, 256>;
fn kv_f16(buf_is_k: bool, idx: u32) -> f32 {{
    var word: u32;
    if (buf_is_k) {{ word = kc[idx >> 1u]; }} else {{ word = vc[idx >> 1u]; }}
    let v = unpack2x16float(word);
    return select(v.x, v.y, (idx & 1u) != 0u);
}}
fn slot(p: u32) -> u32 {{
    if (u.ring > 0u) {{ return p % u.ring; }}
    return p;
}}
fn reduce_max(v: f32, lid: u32) -> f32 {{
    red[lid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {{
        if (lid < s) {{ red[lid] = max(red[lid], red[lid + s]); }}
        workgroupBarrier();
    }}
    let r = red[0];
    workgroupBarrier();
    return r;
}}
fn reduce_sum(v: f32, lid: u32) -> f32 {{
    red[lid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {{
        if (lid < s) {{ red[lid] = red[lid] + red[lid + s]; }}
        workgroupBarrier();
    }}
    let r = red[0];
    workgroupBarrier();
    return r;
}}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {{
    let h = wg.x;
    let t = wg.y;
    let d = u.head_dim;
    let kvh = h / (u.heads / u.kv_heads);
    let qb = t * u.q_stride + h * u.q_head_stride + u.q_off;
    for (var i = lid; i < d; i = i + 256u) {{ qs[i] = q[qb + i] * u.scale; }}
    workgroupBarrier();
    let qpos = u.pos0 + t;
    var k0 = 0u;
    if (u.window > 0u && qpos + 1u > u.window) {{ k0 = qpos + 1u - u.window; }}
    let kw = u.kv_heads * d;
    var m = -3.0e38;
    var lsum = 0.0;
    var acc: array<f32, DPT>;
    for (var i = 0u; i < DPT; i = i + 1u) {{ acc[i] = 0.0; }}
    for (var kb = k0; kb <= qpos; kb = kb + 256u) {{
        let key = kb + lid;
        var s = -3.0e38;
        if (key <= qpos) {{
            let kbase = slot(key) * kw + kvh * d;
            var dotv = 0.0;
            for (var i = 0u; i < d; i = i + 2u) {{
                let kk = unpack2x16float(kc[(kbase + i) >> 1u]);
                dotv = dotv + qs[i] * kk.x + qs[i + 1u] * kk.y;
            }}
            s = dotv;
        }}
        let bm = reduce_max(s, lid);
        let nm = max(m, bm);
        var p = 0.0;
        if (key <= qpos) {{ p = exp(s - nm); }}
        pr[lid] = p;
        let bs = reduce_sum(p, lid);
        let corr = exp(m - nm);
        lsum = lsum * corr + bs;
        m = nm;
        let nk = min(256u, qpos + 1u - kb);
        for (var i = 0u; i < DPT; i = i + 1u) {{
            let dd = lid + i * 256u;
            var a = acc[i] * corr;
            if (dd < d) {{
                for (var j = 0u; j < nk; j = j + 1u) {{
                    a = a + pr[j] * kv_f16(false, slot(kb + j) * kw + kvh * d + dd);
                }}
            }}
            acc[i] = a;
        }}
        workgroupBarrier();
    }}
    let ob = t * u.out_stride + h * d;
    for (var i = 0u; i < DPT; i = i + 1u) {{
        let dd = lid + i * 256u;
        if (dd < d) {{
            var o = acc[i] / lsum;
            if ((u.flags & 1u) != 0u) {{
                let g = gate[t * u.gate_stride + h * u.gate_head_stride + u.gate_off + dd];
                o = o / (1.0 + exp(-g));
            }}
            out[ob + dd] = o;
        }}
    }}
}}
"#,
        dpt = dims_per_thread,
        qmax = 256 * dims_per_thread,
    )
}

/// Uniform of [`CONV1D_SILU_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Conv1dUniform {
    pub rows: u32,
    pub channels: u32,
    /// Kernel width (state holds `kernel - 1` past inputs per channel).
    pub kernel: u32,
    pub _pad: u32,
}

/// `x [rows, C], w [C, kernel] f32, state [C, kernel-1] rw, y [rows, C] rw,
/// uniform`. Depthwise causal conv + SiLU, state carried across calls.
/// Dispatch `[ceil(C / 256), 1, 1]`.
pub const CONV1D_SILU_LAYOUT: [BindingLayout; 5] = layout([R, R, W, W, U]);

pub const CONV1D_SILU_WGSL: &str = r#"// causal conv1d + silu
struct U { rows: u32, channels: u32, kernel: u32, p0: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read_write> st: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= u.channels) { return; }
    let km = u.kernel - 1u;
    var hist: array<f32, 8>;
    for (var i = 0u; i < km; i = i + 1u) { hist[i] = st[c * km + i]; }
    for (var t = 0u; t < u.rows; t = t + 1u) {
        let xv = x[t * u.channels + c];
        var acc = wt[c * u.kernel + km] * xv;
        for (var i = 0u; i < km; i = i + 1u) { acc = acc + wt[c * u.kernel + i] * hist[i]; }
        for (var i = 0u; i + 1u < km; i = i + 1u) { hist[i] = hist[i + 1u]; }
        if (km > 0u) { hist[km - 1u] = xv; }
        y[t * u.channels + c] = acc / (1.0 + exp(-acc));
    }
    for (var i = 0u; i < km; i = i + 1u) { st[c * km + i] = hist[i]; }
}
"#;

/// Uniform of [`DELTA_GATES_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DeltaGatesUniform {
    pub rows: u32,
    pub heads: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// `ab [rows, 2*heads] (alpha | beta), a_neg [heads] (= -exp(A_log)),
/// dt_bias [heads], gb [rows, 2*heads] rw (g | beta), uniform`.
/// `g = a_neg * softplus(alpha + dt_bias)`, `beta = sigmoid(beta)`.
pub const DELTA_GATES_LAYOUT: [BindingLayout; 5] = layout([R, R, R, W, U]);

pub const DELTA_GATES_WGSL: &str = r#"// gated delta gates
struct U { rows: u32, heads: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> ab: array<f32>;
@group(0) @binding(1) var<storage, read> an: array<f32>;
@group(0) @binding(2) var<storage, read> dt: array<f32>;
@group(0) @binding(3) var<storage, read_write> gb: array<f32>;
@group(0) @binding(4) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= u.rows * u.heads) { return; }
    let t = id / u.heads;
    let h = id % u.heads;
    let o = t * 2u * u.heads;
    let z = ab[o + h] + dt[h];
    let sp = select(log(1.0 + exp(z)), z, z > 20.0);
    gb[o + h] = an[h] * sp;
    gb[o + u.heads + h] = 1.0 / (1.0 + exp(-ab[o + u.heads + h]));
}
"#;

/// Uniform of [`DELTA_NET_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DeltaNetUniform {
    pub rows: u32,
    /// Value heads (one workgroup each); key head of value head `h` is
    /// `h % k_heads` (llama.cpp tiled V order).
    pub v_heads: u32,
    pub k_heads: u32,
    /// Conv-output row width: `2 * k_heads * 128 + v_heads * 128`.
    pub conv_width: u32,
    pub eps: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// `qkv [rows, conv_width], gb [rows, 2*v_heads], z [rows, v_heads*128],
/// norm_w [128], state [v_heads, 128 (v), 128 (k)] rw, out [rows, v_heads*128] rw,
/// uniform`. Gated delta rule (head dims 128), sequential over rows, then the
/// gated RMSNorm `norm(o) * w * silu(z)`. Dispatch `[v_heads, 1, 1]`.
pub const DELTA_NET_LAYOUT: [BindingLayout; 7] = layout([R, R, R, R, W, W, U]);

pub const DELTA_NET_WGSL: &str = r#"// gated delta net (recurrent)
struct U { rows: u32, v_heads: u32, k_heads: u32, conv_width: u32, eps: f32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> gb: array<f32>;
@group(0) @binding(2) var<storage, read> z: array<f32>;
@group(0) @binding(3) var<storage, read> nw: array<f32>;
@group(0) @binding(4) var<storage, read_write> st: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform> u: U;
var<workgroup> qs: array<f32, 128>;
var<workgroup> ks: array<f32, 128>;
var<workgroup> red: array<f32, 128>;
var<workgroup> red2: array<f32, 128>;
fn sum2(a: f32, b: f32, j: u32) -> vec2<f32> {
    red[j] = a;
    red2[j] = b;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (j < s) { red[j] = red[j] + red[j + s]; red2[j] = red2[j] + red2[j + s]; }
        workgroupBarrier();
    }
    let r = vec2<f32>(red[0], red2[0]);
    workgroupBarrier();
    return r;
}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) j: u32) {
    let h = wg.x;
    let kh = h % u.k_heads;
    let kdim = u.k_heads * 128u;
    let sb = (h * 128u + j) * 128u;
    var s: array<f32, 128>;
    for (var i = 0u; i < 128u; i = i + 1u) { s[i] = st[sb + i]; }
    for (var t = 0u; t < u.rows; t = t + 1u) {
        let rb = t * u.conv_width;
        let qv = qkv[rb + kh * 128u + j];
        let kv = qkv[rb + kdim + kh * 128u + j];
        let n2 = sum2(qv * qv, kv * kv, j);
        qs[j] = qv * inverseSqrt(n2.x + 1e-6) * inverseSqrt(128.0);
        ks[j] = kv * inverseSqrt(n2.y + 1e-6);
        workgroupBarrier();
        let vj = qkv[rb + 2u * kdim + h * 128u + j];
        let e = exp(gb[t * 2u * u.v_heads + h]);
        let beta = gb[t * 2u * u.v_heads + u.v_heads + h];
        var sk = 0.0;
        var sq = 0.0;
        var kq = 0.0;
        for (var i = 0u; i < 128u; i = i + 1u) {
            sk = sk + s[i] * ks[i];
            sq = sq + s[i] * qs[i];
            kq = kq + ks[i] * qs[i];
        }
        let delta = (vj - e * sk) * beta;
        let o = e * sq + delta * kq;
        for (var i = 0u; i < 128u; i = i + 1u) { s[i] = e * s[i] + ks[i] * delta; }
        let ms = sum2(o * o, 0.0, j);
        let zz = z[t * u.v_heads * 128u + h * 128u + j];
        let on = o * inverseSqrt(ms.x / 128.0 + u.eps) * nw[j] * (zz / (1.0 + exp(-zz)));
        out[t * u.v_heads * 128u + h * 128u + j] = on;
    }
    for (var i = 0u; i < 128u; i = i + 1u) { st[sb + i] = s[i]; }
}
"#;

/// Uniform of [`ROUTER_TOPK_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RouterUniform {
    pub rows: u32,
    pub experts: u32,
    pub top_k: u32,
    /// Bit 0: renormalize the selected probabilities to sum 1. Bit 1:
    /// multiply weight k by `scale[id_k]`.
    pub flags: u32,
}

pub const ROUTER_RENORM: u32 = 1;
pub const ROUTER_EXPERT_SCALE: u32 = 2;

/// `logits [rows, E], scale [E], ids u32 [rows, k] rw, weights [rows, k] rw,
/// uniform`. Softmax over experts, top-k by probability. One workgroup per
/// row; dispatch `[rows, 1, 1]`.
pub const ROUTER_TOPK_LAYOUT: [BindingLayout; 5] = layout([R, R, W, W, U]);

pub const ROUTER_TOPK_WGSL: &str = r#"// router softmax top-k
struct U { rows: u32, experts: u32, top_k: u32, flags: u32 };
@group(0) @binding(0) var<storage, read> lg: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read_write> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> wts: array<f32>;
@group(0) @binding(4) var<uniform> u: U;
var<workgroup> pv: array<f32, 1024>;
var<workgroup> rv: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let r = wg.x;
    let b = r * u.experts;
    var mx = -3.0e38;
    for (var e = lid; e < u.experts; e = e + 256u) { mx = max(mx, lg[b + e]); }
    rv[lid] = mx;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) { rv[lid] = max(rv[lid], rv[lid + s]); }
        workgroupBarrier();
    }
    let m = rv[0];
    workgroupBarrier();
    var sm = 0.0;
    for (var e = lid; e < u.experts; e = e + 256u) { let p = exp(lg[b + e] - m); pv[e] = p; sm = sm + p; }
    rv[lid] = sm;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) { rv[lid] = rv[lid] + rv[lid + s]; }
        workgroupBarrier();
    }
    let total = rv[0];
    workgroupBarrier();
    var picked_sum = 0.0;
    for (var k = 0u; k < u.top_k; k = k + 1u) {
        var bv = -1.0;
        var bi = 0xFFFFFFFFu;
        for (var e = lid; e < u.experts; e = e + 256u) {
            let p = pv[e];
            if (p > bv) { bv = p; bi = e; }
        }
        rv[lid] = bv;
        ri[lid] = bi;
        workgroupBarrier();
        for (var s = 128u; s > 0u; s = s >> 1u) {
            if (lid < s) {
                let ov = rv[lid + s];
                let oi = ri[lid + s];
                if (ov > rv[lid] || (ov == rv[lid] && oi < ri[lid])) { rv[lid] = ov; ri[lid] = oi; }
            }
            workgroupBarrier();
        }
        let wi = ri[0];
        let wv = rv[0] / total;
        workgroupBarrier();
        if (lid == 0u) {
            ids[r * u.top_k + k] = wi;
            wts[r * u.top_k + k] = wv;
            pv[wi] = -2.0;
        }
        picked_sum = picked_sum + wv;
        workgroupBarrier();
    }
    if (lid < u.top_k) {
        var v = wts[r * u.top_k + lid];
        if ((u.flags & 1u) != 0u) { v = v / picked_sum; }
        if ((u.flags & 2u) != 0u) { v = v * scale[ids[r * u.top_k + lid]]; }
        wts[r * u.top_k + lid] = v;
    }
}
"#;

/// `a, b, y rw, uniform {n}`: `y = silu(a) * b` (or gelu-tanh with flag 1).
pub const GLU_LAYOUT: [BindingLayout; 4] = layout([R, R, W, U]);

pub const GLU_WGSL: &str = r#"// glu
struct U { n: u32, flags: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= u.n) { return; }
    let g = a[i];
    var act: f32;
    if ((u.flags & 1u) != 0u) {
        act = 0.5 * g * (1.0 + tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)));
    } else {
        act = g / (1.0 + exp(-g));
    }
    y[i] = act * b[i];
}
"#;

pub const GLU_GELU_TANH: u32 = 1;

/// Uniform of [`MOE_COMBINE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MoeCombineUniform {
    pub rows: u32,
    pub dim: u32,
    pub top_k: u32,
    /// Bit 0: add `sigmoid(sg[row]) * shared[row]`. Bit 1: add `res`.
    pub flags: u32,
}

pub const MOE_SHARED: u32 = 1;
pub const MOE_RESIDUAL: u32 = 2;

/// `experts [rows*k, dim], weights [rows, k], shared [rows, dim], sg [rows],
/// res [rows, dim], y rw, uniform`.
pub const MOE_COMBINE_LAYOUT: [BindingLayout; 7] = layout([R, R, R, R, R, W, U]);

pub const MOE_COMBINE_WGSL: &str = r#"// moe combine
struct U { rows: u32, dim: u32, top_k: u32, flags: u32 };
@group(0) @binding(0) var<storage, read> ex: array<f32>;
@group(0) @binding(1) var<storage, read> wts: array<f32>;
@group(0) @binding(2) var<storage, read> sh: array<f32>;
@group(0) @binding(3) var<storage, read> sg: array<f32>;
@group(0) @binding(4) var<storage, read> res: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
@group(0) @binding(6) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= u.rows * u.dim) { return; }
    let r = i / u.dim;
    let d = i % u.dim;
    var v = 0.0;
    for (var k = 0u; k < u.top_k; k = k + 1u) {
        v = v + wts[r * u.top_k + k] * ex[(r * u.top_k + k) * u.dim + d];
    }
    if ((u.flags & 1u) != 0u) { v = v + sh[i] / (1.0 + exp(-sg[r])); }
    if ((u.flags & 2u) != 0u) { v = v + res[i]; }
    y[i] = v;
}
"#;

/// Bytes of a uniform struct (all are `#[repr(C)]` u32/f32 fields).
pub fn uniform_bytes<T: Copy>(u: &T) -> Vec<u8> {
    let n = core::mem::size_of::<T>();
    let mut v = vec![0u8; n];
    // SAFETY: `T` is a `#[repr(C)]` plain-old-data uniform of u32/f32 fields.
    unsafe { core::ptr::copy_nonoverlapping(u as *const T as *const u8, v.as_mut_ptr(), n) };
    v
}

/// Workgroups for a 1-D dispatch of `n` invocations at 256 per group, folded
/// into `y` past the 65535 limit.
pub fn grid_1d(n: u32) -> [u32; 3] {
    let g = n.div_ceil(256).max(1);
    if g <= 65535 {
        [g, 1, 1]
    } else {
        [65535, g.div_ceil(65535), 1]
    }
}

/// One LLM kernel on device addresses, for an [`crate::backend::Accelerator`]
/// (same math and uniforms as the WGSL kernels above). Enqueued in stream
/// order; complete after the accelerator's sync / fence.
#[derive(Clone, Copy, Debug)]
pub enum LlmOp {
    /// [`gemv_wgsl`] direct mode over `rows` rows.
    Gemv {
        u: GemvUniform,
        fmt: LlmWeight,
        w: u64,
        x: u64,
        y: u64,
        res: u64,
        rows: u32,
    },
    /// Routed-expert gemv: pair `p` uses expert `ids[p]`, whose matrix is at
    /// `table[ids[p]]` (a u64 device address: VRAM slot or mapped host
    /// memory), reads x row `p / x_div` and writes y row `p`.
    GemvExperts {
        u: GemvUniform,
        fmt: LlmWeight,
        table: u64,
        ids: u64,
        x: u64,
        y: u64,
        pairs: u32,
        x_div: u32,
    },
    RmsNorm {
        u: RmsNormUniform,
        x: u64,
        w: u64,
        y: u64,
        res: u64,
    },
    Rope {
        u: RopeUniform,
        x: u64,
        cos_sin: u64,
    },
    KvStore {
        u: KvStoreUniform,
        src: u64,
        cache: u64,
    },
    Attention {
        u: AttnUniform,
        q: u64,
        k_cache: u64,
        v_cache: u64,
        gate: u64,
        out: u64,
    },
    Conv1dSilu {
        u: Conv1dUniform,
        x: u64,
        w: u64,
        state: u64,
        y: u64,
    },
    DeltaGates {
        u: DeltaGatesUniform,
        ab: u64,
        a_neg: u64,
        dt_bias: u64,
        gb: u64,
    },
    DeltaNet {
        u: DeltaNetUniform,
        qkv: u64,
        gb: u64,
        z: u64,
        norm_w: u64,
        state: u64,
        out: u64,
    },
    Router {
        u: RouterUniform,
        logits: u64,
        scale: u64,
        ids: u64,
        weights: u64,
    },
    Glu {
        n: u32,
        gelu: bool,
        a: u64,
        b: u64,
        y: u64,
    },
    /// Sample one id from f32 `logits[..n]` as the host `Sampler` does with
    /// top-p 1 and no penalties: the `k` (<= 64) largest by logit, `p_i =
    /// exp(l_i / t - l_max / t)` normalized and summed in rank order, the
    /// first rank where `uniform[0] * total` runs out. `t <= 0`: argmax
    /// (ties: the highest id, like the host). Writes the id (u32) to `out`.
    SampleTopK {
        logits: u64,
        n: u32,
        k: u32,
        temperature: f32,
        uniform: u64,
        out: u64,
    },
    /// `out[..dim] = f32(table[idx[0] * dim ..])`, a bf16 `[rows, dim]` table
    /// and a u32 index read on the device.
    GatherRowBf16 {
        table: u64,
        idx: u64,
        dim: u32,
        out: u64,
    },
    MoeCombine {
        u: MoeCombineUniform,
        experts: u64,
        weights: u64,
        shared: u64,
        sg: u64,
        res: u64,
        y: u64,
    },
}
