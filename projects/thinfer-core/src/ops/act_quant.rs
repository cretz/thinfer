//! Activation quantizer: f16 acts -> packed int8 + per-(M, K/32)
//! `(scale_f16, zero_f16)` parameters (llama.cpp Q8_1-style asymmetric).
//!
//! Runs immediately before a DP4A matmul site. Reads `[M, K]` f16 activations,
//! emits `[M, K]` packed int8 (4 per u32, K-major) plus `[M, K/32]` params
//! (vec2<f16> per K-block per row). The matmul then consumes the i8 buffer
//! with `dot4I8Packed`, multiplies the dot accumulator by per-block
//! `(scale_a * scale_b)`, and adds a correction `zero_a * scale_b * Σ qb`
//! that comes from the asymmetric expansion of `(qa*sa + za)*(qb*sb)`.
//!
//! Quantization: per (m, K/32) sub-block, find (min, max) of the 32 f16
//! values. scale = (max-min)/254 (signed-i8 [-127, 127] keeps `dot4I8Packed`
//! overflow-safe). zero = min + 127*scale. q = round((x - zero) / scale)
//! clamped to [-127, 127]. Degenerate range emits scale=1e-30 / zero=min so
//! dequant returns min for every element (no NaN from zero scale).
//!
//! Dispatch shape: one workgroup per (m, K/32) sub-block, 32 threads/WG, each
//! thread handles one i8 output. Thread 0 computes the params; threads 0..7
//! pack 4 i8 each into 8 u32 outputs via `pack_f32x4_aff_to_i8`.

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
        kind: BindingKind::StorageReadWrite,
    },
    BindingLayout {
        slot: 3,
        kind: BindingKind::Uniform,
    },
];

pub fn layout() -> &'static [BindingLayout] {
    LAYOUT
}

/// Pipeline-cache hint. Single configuration today (f16 input, asymmetric).
pub fn hint() -> &'static str {
    "act_quant_i8-f16-aff"
}

/// Build the WGSL for the activation int8 quantizer. Input is `array<vec2<f16>>`
/// `[M, K]` (K-major, two K-adjacent elements per word). Outputs:
/// - `out_i8`: `array<u32>` `[M, K/4]` (4 packed i8 per word).
/// - `out_params`: `array<vec2<f16>>` `[M, K/32]` (one `(scale, zero)` pair
///   per K=32 sub-block).
///
/// Caller dispatches `(K/32, M, 1)` workgroups.
pub fn build_wgsl() -> String {
    // 32-thread workgroup: one thread per i8 output element in the 32-element
    // sub-block. Reduction across the 32 elements is done in workgroup memory.
    r#"enable f16;

struct Dims { m: u32, k: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> a: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read_write> out_i8: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_params: array<vec2<f16>>;
@group(0) @binding(3) var<uniform> d: Dims;

const BLOCK: u32 = 32u;

// Per-block scratch.
var<workgroup> vals: array<f32, BLOCK>;
var<workgroup> mn_buf: array<f32, BLOCK>;
var<workgroup> mx_buf: array<f32, BLOCK>;
var<workgroup> shared_params: vec2<f16>;

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let m: u32 = wid.y;
    let block_k: u32 = wid.x;
    let t: u32 = lid.x;
    if (m >= d.m) { return; }
    let blocks_per_row: u32 = d.k / BLOCK;
    let k_in_block: u32 = t;
    let g_k: u32 = block_k * BLOCK + k_in_block;
    // a is array<vec2<f16>>: two f16 per word, K-major.
    var v: f32 = 0.0;
    if (g_k < d.k) {
        let pair: vec2<f32> = vec2<f32>(a[(m * d.k + g_k) >> 1u]);
        v = select(pair.x, pair.y, (g_k & 1u) == 1u);
    }
    vals[t] = v;
    mn_buf[t] = v;
    mx_buf[t] = v;
    workgroupBarrier();
    // Tree reduction: 32 -> 16 -> 8 -> 4 -> 2 -> 1.
    if (t < 16u) {
        mn_buf[t] = min(mn_buf[t], mn_buf[t + 16u]);
        mx_buf[t] = max(mx_buf[t], mx_buf[t + 16u]);
    }
    workgroupBarrier();
    if (t < 8u) {
        mn_buf[t] = min(mn_buf[t], mn_buf[t + 8u]);
        mx_buf[t] = max(mx_buf[t], mx_buf[t + 8u]);
    }
    workgroupBarrier();
    if (t < 4u) {
        mn_buf[t] = min(mn_buf[t], mn_buf[t + 4u]);
        mx_buf[t] = max(mx_buf[t], mx_buf[t + 4u]);
    }
    workgroupBarrier();
    if (t < 2u) {
        mn_buf[t] = min(mn_buf[t], mn_buf[t + 2u]);
        mx_buf[t] = max(mx_buf[t], mx_buf[t + 2u]);
    }
    workgroupBarrier();
    if (t == 0u) {
        let mn = min(mn_buf[0], mn_buf[1]);
        let mx = max(mx_buf[0], mx_buf[1]);
        let range = mx - mn;
        let s = select(range / 254.0, 1.0e-30, range <= 0.0);
        let z = mn + 127.0 * s;
        let p = vec2<f16>(f16(s), f16(z));
        shared_params = p;
        out_params[m * blocks_per_row + block_k] = p;
    }
    workgroupBarrier();
    // Pack 4 i8 per u32. Threads 0..7 each handle 4 elements of the block.
    if (t < 8u) {
        let p = shared_params;
        let s = f32(p.x);
        let z = f32(p.y);
        let inv = 1.0 / s;
        let base: u32 = t * 4u;
        let v0: f32 = vals[base + 0u];
        let v1: f32 = vals[base + 1u];
        let v2: f32 = vals[base + 2u];
        let v3: f32 = vals[base + 3u];
        let q = clamp(
            round((vec4<f32>(v0, v1, v2, v3) - vec4<f32>(z)) * inv),
            vec4<f32>(-127.0),
            vec4<f32>(127.0),
        );
        let b0 = u32(i32(q.x)) & 0xFFu;
        let b1 = u32(i32(q.y)) & 0xFFu;
        let b2 = u32(i32(q.z)) & 0xFFu;
        let b3 = u32(i32(q.w)) & 0xFFu;
        let word: u32 = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
        let words_per_row: u32 = d.k / 4u;
        let word_idx: u32 = m * words_per_row + block_k * 8u + t;
        out_i8[word_idx] = word;
    }
}
"#
    .to_string()
}

pub struct ActQuantBufs<'a> {
    pub a: &'a BufRef,
    pub out_i8: &'a BufRef,
    pub out_params: &'a BufRef,
    pub dims: &'a BufRef,
}

/// Dispatch one activation-quantize pass. `m` rows, `k` columns; `k` must be a
/// multiple of 32 (one K-sub-block per workgroup).
pub fn dispatch_act_quant<B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &ActQuantBufs<'_>,
    m: u32,
    k: u32,
) -> Result<(), B::Error> {
    assert!(
        k.is_multiple_of(32),
        "act_quant: k={k} must be multiple of 32"
    );
    let blocks_per_row = k / 32;
    let bindings = [
        bufs.a.binding(0),
        bufs.out_i8.binding(1),
        bufs.out_params.binding(2),
        bufs.dims.binding(3),
    ];
    backend.dispatch(encoder, pipeline, &bindings, [blocks_per_row, m, 1])
}
