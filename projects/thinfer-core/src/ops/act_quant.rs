//! Activation quantizer: f16 acts -> packed int8 + per-(M, K/32) f32 scale.
//!
//! Runs immediately before a DP4A matmul site. Reads `[M, K]` f16 activations,
//! emits `[M, K]` packed int8 (4 per u32, K-major) plus `[M, K/32]` f32 scales,
//! one scale per 32-element K-block per row. The matmul then consumes the i8
//! buffer with `dot4I8Packed` and multiplies the dot accumulator by the
//! per-block (a_scale * b_scale) at the K-block boundary.
//!
//! Quantization: per (m, K/32) sub-block, find max-abs of the 32 f16 values,
//! divide by 127, round-to-nearest-even, clamp to [-127, 127]. -128 is
//! excluded so `dot4I8Packed` (SDot lowering) cannot overflow the i32
//! accumulator on any (i8 * i8) -> i16 step. Zero-scale sub-blocks emit all
//! zeros and a scale of 0.0 so the matmul produces a clean zero contribution.
//!
//! Dispatch shape: one workgroup per (m, K/32) sub-block, 32 threads/WG, each
//! thread handles one i8 output. Thread 0 computes the scale; threads 0..7
//! pack 4 i8 each into 8 u32 outputs via `pack4xI8Clamp`.

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

/// Pipeline-cache hint. Single configuration today (f16 input).
pub fn hint() -> &'static str {
    "act_quant_i8-f16"
}

/// Build the WGSL for the activation int8 quantizer. Input is `array<vec2<f16>>`
/// `[M, K]` (K-major, two K-adjacent elements per word). Outputs:
/// - `out_i8`: `array<u32>` `[M, K/4]` (4 packed i8 per word).
/// - `out_scale`: `array<f32>` `[M, K/32]` (one scale per K=32 sub-block).
///
/// Caller dispatches `(K/32, M, 1)` workgroups.
pub fn build_wgsl() -> String {
    // 32-thread workgroup: one thread per i8 output element in the 32-element
    // sub-block. Reduction across the 32 elements is done in workgroup memory.
    r#"enable f16;

struct Dims { m: u32, k: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> a: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read_write> out_i8: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_scale: array<f32>;
@group(0) @binding(3) var<uniform> d: Dims;

const BLOCK: u32 = 32u;

// Per-block scratch: 32 f32 values, then reduce to max-abs via shared scan.
var<workgroup> vals: array<f32, BLOCK>;
var<workgroup> absmax: array<f32, BLOCK>;

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
    // a is array<vec2<f16>>: two f16 per word, K-major. Word index = (m*K + g_k) / 2.
    var v: f32 = 0.0;
    if (g_k < d.k) {
        let pair: vec2<f32> = vec2<f32>(a[(m * d.k + g_k) >> 1u]);
        v = select(pair.x, pair.y, (g_k & 1u) == 1u);
    }
    vals[t] = v;
    absmax[t] = abs(v);
    workgroupBarrier();
    // Tree reduction to absmax[0]. 32 -> 16 -> 8 -> 4 -> 2 -> 1.
    if (t < 16u) { absmax[t] = max(absmax[t], absmax[t + 16u]); }
    workgroupBarrier();
    if (t < 8u)  { absmax[t] = max(absmax[t], absmax[t + 8u]);  }
    workgroupBarrier();
    if (t < 4u)  { absmax[t] = max(absmax[t], absmax[t + 4u]);  }
    workgroupBarrier();
    if (t < 2u)  { absmax[t] = max(absmax[t], absmax[t + 2u]);  }
    workgroupBarrier();
    if (t == 0u) { absmax[0] = max(absmax[0], absmax[1]); }
    workgroupBarrier();
    let amax: f32 = absmax[0];
    // Scale: amax / 127. Zero-scale sub-block emits all zeros (no NaN risk).
    let scale: f32 = amax / 127.0;
    let inv_scale: f32 = select(0.0, 1.0 / scale, scale > 0.0);
    // Thread 0 writes the scale.
    if (t == 0u) {
        out_scale[m * blocks_per_row + block_k] = scale;
    }
    // Pack 4 i8 per u32. Threads 0..7 each handle 4 elements of the block:
    // thread `q` handles elements [4q .. 4q+4). pack4xI8Clamp clamps to
    // [-128, 127] but we cap inputs at +-127 so -128 never appears.
    if (t < 8u) {
        let base: u32 = t * 4u;
        let q0: f32 = round(vals[base + 0u] * inv_scale);
        let q1: f32 = round(vals[base + 1u] * inv_scale);
        let q2: f32 = round(vals[base + 2u] * inv_scale);
        let q3: f32 = round(vals[base + 3u] * inv_scale);
        let v4: vec4<f32> = clamp(
            vec4<f32>(q0, q1, q2, q3),
            vec4<f32>(-127.0),
            vec4<f32>(127.0),
        );
        let word: u32 = pack4xI8(vec4<i32>(v4));
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
    pub out_scale: &'a BufRef,
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
        bufs.out_scale.binding(2),
        bufs.dims.binding(3),
    ];
    backend.dispatch(encoder, pipeline, &bindings, [blocks_per_row, m, 1])
}
