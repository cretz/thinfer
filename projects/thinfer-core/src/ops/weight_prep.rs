//! GPU-side weight preparation at upload time.
//!
//! Two load-time kernels that replace CPU weight munging in the residency
//! miss path (the CPU equivalents run ~1 GB/s and dominate cold text-encode
//! wall time; see `WeightResidency::acquire`):
//!
//! - `q8_0_from_bf16`: raw bf16 `[N, K]` row-major (`K % 32 == 0`) into the
//!   GGUF-native Q8_0 block stream, mirroring `quant::encode_q8_0_from_bf16`
//!   (amax/127 f16-RNE scale, round-half-away quants). Value-equivalent but
//!   NOT bit-exact vs the CPU encoder: WGSL division is only 2.5-ULP
//!   accurate, so quants whose scaled input lands within an ULP of a
//!   round-to-nearest tie flip by +-1 (~0.03% of cells on real weights;
//!   either choice reconstructs with equal error). Same class of divergence
//!   llama.cpp accepts between its CPU/CUDA/Metal quantizers.
//! - `transpose_bf16_2d`: raw bf16 `[N, K]` row-major into `[K, N]`
//!   (nn.Linear upload transpose), bit-exact vs the CPU `Linear2D` path.
//! - `narrow_transpose_f32`: raw f32 `[N, K]` row-major into bf16 `[K, N]`,
//!   fusing the RNE narrow with the Linear2D transpose. Bit-exact vs the CPU
//!   `narrow_f32_to_bf16` + `transpose_bf16_cpu` pair (the f32/safetensors
//!   weight path, ~1.4s/umT5-layer single-threaded while the GPU idles).
//!
//! Both kernels are shape-independent (dims via uniform), so one pipeline
//! per op serves every weight.
//!
//! Q8_0 blocks are 34 bytes, so a single block is not u32-aligned; threads
//! therefore process PAIRS of consecutive blocks (68 bytes = 17 u32 words).
//! The block stream has no row structure (`K % 32 == 0` keeps blocks inside
//! rows, and the stream is contiguous across rows), so pairing needs only an
//! even total block count - enforced by the caller (`prep_op` gate).

use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};

const LAYOUT: &[BindingLayout] = &[
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
];

pub fn layout() -> &'static [BindingLayout] {
    LAYOUT
}

/// f32 -> f16 bits with round-to-nearest-even, mirroring
/// `half::f16::from_f32` for non-negative finite inputs (the only values the
/// Q8_0 scale can take). Carry out of the mantissa propagates into the
/// exponent by construction; overflow saturates to +inf like the CPU side.
const F32_TO_F16_RNE: &str = r#"
fn f16_bits_rne(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    if (b == 0u) { return 0u; }
    let e = i32((b >> 23u) & 0xFFu) - 127;
    if (e < -24) { return 0u; }
    let mant = b & 0x7FFFFFu;
    if (e < -14) {
        // Subnormal f16: include the implicit bit, shift by the extra amount.
        let shift = u32(-14 - e) + 13u;
        let m = mant | 0x800000u;
        let q = m >> shift;
        let rem = m & ((1u << shift) - 1u);
        let half_ = 1u << (shift - 1u);
        if (rem > half_ || (rem == half_ && (q & 1u) == 1u)) { return q + 1u; }
        return q;
    }
    let q = mant >> 13u;
    let rem = mant & 0x1FFFu;
    var r = (u32(e + 15) << 10u) | q;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u) == 1u)) { r = r + 1u; }
    if (r >= 0x7C00u) { return 0x7C00u; }
    return r;
}
"#;

/// Build the bf16 -> Q8_0 transcode WGSL. One thread per block PAIR; caller
/// dispatches `(ceil(pairs / 64), 1, 1)` workgroups with `pairs` in the
/// uniform. Input is the raw bf16 stream as `array<u32>` (two elements per
/// word, little-endian); output is the packed Q8_0 block stream.
pub fn build_q8_0_from_bf16_wgsl() -> String {
    // encode_block body: unrolled 16-word read, vec4 abs-max tree, then 8
    // packed quant words. No dynamically-indexed locals (kernel register
    // rule); everything is explicit scalars/vec4s via codegen.
    let mut body = String::new();
    for j in 0..8 {
        let w0 = 2 * j;
        let w1 = 2 * j + 1;
        body.push_str(&format!(
            "    let u{w0} = src[wb + {w0}u];\n    let u{w1} = src[wb + {w1}u];\n    let v{j} = vec4<f32>(bf_lo(u{w0}), bf_hi(u{w0}), bf_lo(u{w1}), bf_hi(u{w1}));\n",
        ));
    }
    body.push_str(
        "    var am = max(abs(v0), abs(v1));\n    am = max(am, max(abs(v2), abs(v3)));\n    am = max(am, max(abs(v4), abs(v5)));\n    am = max(am, max(abs(v6), abs(v7)));\n    let amax = max(max(am.x, am.y), max(am.z, am.w));\n    let d = amax / 127.0;\n    var id = 0.0;\n    if (d != 0.0) { id = 1.0 / d; }\n",
    );
    for j in 0..8 {
        body.push_str(&format!(
            "    let s{j} = v{j} * id;\n    let q{j} = vec4<i32>(sign(s{j}) * floor(abs(s{j}) + 0.5));\n    out.w{j} = pack_q(q{j});\n",
        ));
    }
    format!(
        r#"struct Dims {{ pairs: u32, _p0: u32, _p1: u32, _p2: u32 }};

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> dims: Dims;

fn bf_lo(u: u32) -> f32 {{ return bitcast<f32>(u << 16u); }}
fn bf_hi(u: u32) -> f32 {{ return bitcast<f32>(u & 0xFFFF0000u); }}
fn pack_q(q: vec4<i32>) -> u32 {{
    return (u32(q.x) & 0xFFu) | ((u32(q.y) & 0xFFu) << 8u)
        | ((u32(q.z) & 0xFFu) << 16u) | ((u32(q.w) & 0xFFu) << 24u);
}}
{F32_TO_F16_RNE}
struct Q8Block {{
    d: u32,
    w0: u32, w1: u32, w2: u32, w3: u32, w4: u32, w5: u32, w6: u32, w7: u32,
}};

fn encode_block(wb: u32) -> Q8Block {{
    var out: Q8Block;
{body}    out.d = f16_bits_rne(d);
    return out;
}}

@compute @workgroup_size(64, 1, 1)
fn q8_0_from_bf16(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let pair = wid.x * 64u + lid.x;
    if (pair >= dims.pairs) {{ return; }}
    // 64 elements per pair = 32 input words; 68 output bytes = 17 words.
    let b0 = encode_block(pair * 32u);
    let b1 = encode_block(pair * 32u + 16u);
    let ow = pair * 17u;
    // Stream layout per block: [d: f16][qs: i8 x 32], packed back to back.
    dst[ow] = b0.d | ((b0.w0 & 0xFFFFu) << 16u);
    dst[ow + 1u] = (b0.w0 >> 16u) | ((b0.w1 & 0xFFFFu) << 16u);
    dst[ow + 2u] = (b0.w1 >> 16u) | ((b0.w2 & 0xFFFFu) << 16u);
    dst[ow + 3u] = (b0.w2 >> 16u) | ((b0.w3 & 0xFFFFu) << 16u);
    dst[ow + 4u] = (b0.w3 >> 16u) | ((b0.w4 & 0xFFFFu) << 16u);
    dst[ow + 5u] = (b0.w4 >> 16u) | ((b0.w5 & 0xFFFFu) << 16u);
    dst[ow + 6u] = (b0.w5 >> 16u) | ((b0.w6 & 0xFFFFu) << 16u);
    dst[ow + 7u] = (b0.w6 >> 16u) | ((b0.w7 & 0xFFFFu) << 16u);
    dst[ow + 8u] = (b0.w7 >> 16u) | (b1.d << 16u);
    dst[ow + 9u] = b1.w0;
    dst[ow + 10u] = b1.w1;
    dst[ow + 11u] = b1.w2;
    dst[ow + 12u] = b1.w3;
    dst[ow + 13u] = b1.w4;
    dst[ow + 14u] = b1.w5;
    dst[ow + 15u] = b1.w6;
    dst[ow + 16u] = b1.w7;
}}
"#
    )
}

/// Build the bf16 `[N, K]` -> `[K, N]` transpose WGSL. One thread per output
/// word (two consecutive output-row elements, i.e. two source rows at one
/// column); caller dispatches `(ceil(N/2 / 64), K, 1)`. N must be even:
/// odd N would put one output u32 across two output rows, racing adjacent
/// threads (the residency prep gate routes odd-N weights to the CPU path).
pub fn build_transpose_bf16_wgsl() -> String {
    r#"struct Dims { n: u32, k: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> dims: Dims;

fn elem_u16(i: u32) -> u32 {
    return (src[i >> 1u] >> ((i & 1u) * 16u)) & 0xFFFFu;
}

@compute @workgroup_size(64, 1, 1)
fn transpose_bf16_2d(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let kk = wid.y;
    let nn = (wid.x * 64u + lid.x) * 2u;
    if (nn >= dims.n) { return; }
    let i0 = nn * dims.k + kk;
    dst[(kk * dims.n + nn) >> 1u] = elem_u16(i0) | (elem_u16(i0 + dims.k) << 16u);
}
"#
    .to_string()
}

/// f32 -> bf16 bits with round-to-nearest-even, bit-exact vs
/// `half::bf16::from_f32` (truncate the low 16 bits, round on the discarded
/// MSB with ties-to-even; NaN keeps its high mantissa plus the quiet bit).
/// `3 * 0x8000 - 1 == 0x17FFF == 0x7FFF | 0x10000`: the sticky bits OR the
/// result LSB, excluding the round bit, so an exact half rounds to even.
const F32_TO_BF16_RNE: &str = r#"
fn bf16_bits_rne(b: u32) -> u32 {
    if ((b & 0x7FFFFFFFu) > 0x7F800000u) { return (b >> 16u) | 0x40u; }
    if ((b & 0x8000u) != 0u && (b & 0x17FFFu) != 0u) { return ((b >> 16u) + 1u) & 0xFFFFu; }
    return b >> 16u;
}
"#;

/// Build the f32 `[N, K]` -> bf16 `[K, N]` fused narrow+transpose WGSL. One
/// thread per output word (two consecutive output-row elements = two source
/// rows at one column); caller dispatches `(ceil(band_n/2 / 64), K, 1)`.
///
/// Banded: `src` holds only rows `[n0, n0 + band_n)` of the `[N, K]` tensor
/// (band-local indexing), while output uses the global `[K, N]` layout (stride
/// `n`, column offset `n0`), so the f32 staging stays bounded to one band. `n`
/// and `n0`, `band_n` must be even (odd would put one output u32 across two
/// output rows, racing adjacent threads; the residency prep gate routes odd-N
/// weights to the CPU path). Source is the raw f32 stream as `array<u32>` (one
/// word per element); each element narrows through `bf16_bits_rne`.
pub fn build_narrow_transpose_f32_wgsl() -> String {
    format!(
        r#"struct Dims {{ n: u32, k: u32, n0: u32, band_n: u32 }};

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> dims: Dims;
{F32_TO_BF16_RNE}
@compute @workgroup_size(64, 1, 1)
fn narrow_transpose_f32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let kk = wid.y;
    let r = (wid.x * 64u + lid.x) * 2u; // band-local row
    if (r >= dims.band_n) {{ return; }}
    let i0 = r * dims.k + kk; // band-local source element
    let lo = bf16_bits_rne(src[i0]);
    let hi = bf16_bits_rne(src[i0 + dims.k]);
    let nn = dims.n0 + r; // global output column
    dst[(kk * dims.n + nn) >> 1u] = lo | (hi << 16u);
}}
"#
    )
}

pub struct WeightPrepBufs<'a> {
    pub src: &'a BufRef,
    pub dst: &'a BufRef,
    pub dims: &'a BufRef,
}

/// Dispatch one Q8_0 transcode. `pairs` = total 32-element blocks / 2; the
/// caller guarantees an even block count and writes `pairs` into the dims
/// uniform.
pub fn dispatch_q8_0_from_bf16<B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &WeightPrepBufs<'_>,
    pairs: u32,
) -> Result<(), B::Error> {
    let wgx = pairs.div_ceil(64);
    assert!(
        wgx <= 65535,
        "q8_0_from_bf16: {pairs} pairs exceeds dispatch"
    );
    let bindings = [
        bufs.src.binding(0),
        bufs.dst.binding(1),
        bufs.dims.binding(2),
    ];
    backend.dispatch(encoder, pipeline, &bindings, [wgx, 1, 1])
}

/// Dispatch one bf16 2-D transpose of `[n, k]` into `[k, n]`.
pub fn dispatch_transpose_bf16<B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &WeightPrepBufs<'_>,
    n: u32,
    k: u32,
) -> Result<(), B::Error> {
    assert!(n.is_multiple_of(2), "transpose_bf16_2d: N={n} must be even");
    let wgx = (n / 2).div_ceil(64);
    assert!(
        wgx <= 65535 && k <= 65535,
        "transpose_bf16_2d: [{n}, {k}] exceeds dispatch"
    );
    let bindings = [
        bufs.src.binding(0),
        bufs.dst.binding(1),
        bufs.dims.binding(2),
    ];
    backend.dispatch(encoder, pipeline, &bindings, [wgx, k, 1])
}

/// Dispatch one f32 -> bf16 narrow+transpose band: source rows
/// `[n0, n0 + band_n)` of the global `[n, k]` tensor into the `[k, n]` output.
/// One thread per output word over the band, `(ceil(band_n/2 / 64), k, 1)`.
pub fn dispatch_narrow_transpose_f32<B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &WeightPrepBufs<'_>,
    n: u32,
    k: u32,
    n0: u32,
    band_n: u32,
) -> Result<(), B::Error> {
    assert!(
        n.is_multiple_of(2) && n0.is_multiple_of(2) && band_n.is_multiple_of(2),
        "narrow_transpose_f32: N={n} n0={n0} band_n={band_n} must be even"
    );
    let wgx = (band_n / 2).div_ceil(64);
    assert!(
        wgx <= 65535 && k <= 65535,
        "narrow_transpose_f32: band [{band_n}, {k}] exceeds dispatch"
    );
    let bindings = [
        bufs.src.binding(0),
        bufs.dst.binding(1),
        bufs.dims.binding(2),
    ];
    backend.dispatch(encoder, pipeline, &bindings, [wgx, k, 1])
}
