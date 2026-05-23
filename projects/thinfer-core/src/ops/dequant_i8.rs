//! Block-quantized weight -> packed int8 + per-K=32 scale dequant pass.
//!
//! Materializes a quant weight tensor (Q4_K/Q5_K/Q6_K/Q8_0/Q4_0) into:
//! - `b_i8`: `array<u32>` `[N, K/4]` (4 packed i8 per word, N-major).
//! - `b_scale`: `array<f32>` `[N, K/32]` (one scale per K=32 sub-block).
//!
//! Pairs with `act_quant_i8` (same int8 + scale shape on the A side) and the
//! DP4A matmul (`matmul_i8`) which reads both via `dot4I8Packed`.
//!
//! For K-family schemes (Q4_K/Q5_K/Q6_K, block_size=256): one Q-block contains
//! 8 K=32 sub-blocks. The 64-thread WG dequants 256 f32 values cooperatively
//! (4 per thread), reduces absmax within each 8-thread sub-block group, and
//! writes 8 scales + 64 u32 i8-packed words per Q-block.
//!
//! For Q8_0 / Q4_0 (block_size=32): one Q-block IS one K=32 sub-block. Only
//! the first 8 threads of the WG do work (4 elements each); 1 scale + 8 u32
//! per Q-block.
//!
//! The secondary i8 quantization on top of the Q-family dequant is a tiny
//! precision loss (~0.3 LSB on already-Q4-quantized values); Q4 was the
//! dominant lossy step. For Q8_0 the round trip is mathematically a
//! near-identity (original scale recovered, original bytes recovered).

use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
use crate::quant::QuantKind;

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

pub fn hint(scheme: QuantKind) -> String {
    format!("dequant_i8-{}", scheme.hint())
}

/// Build the I8 dequant WGSL for one quant scheme.
///
/// `block_size` (BS) must be a multiple of 32 (the K=32 sub-block size).
/// All supported schemes satisfy this: K-family is 256, Q8_0/Q4_0 is 32.
pub fn build_wgsl(scheme: QuantKind) -> String {
    let scheme_wgsl = scheme.wgsl();
    let init_call = scheme.block_state_call();
    let elem4_call = scheme.block_elem4_call();
    let bs = scheme.block_size();
    let bpb = scheme.bytes_per_block();
    assert!(
        bs.is_multiple_of(32),
        "dequant_i8 requires block_size % 32 == 0 (got {bs})"
    );
    let subblocks_per_qblock = bs / 32;
    // 64 threads × 4 elements = 256 = max Q-block elements. For bs=32 only
    // threads 0..7 do work (8 threads × 4 elems = 32).
    let active_threads = bs / 4;
    format!(
        r#"struct Dims {{ n: u32, k: u32, _pad0: u32, _pad1: u32 }};

@group(0) @binding(0) var<storage, read> b: array<u32>;
@group(0) @binding(1) var<storage, read_write> b_i8: array<u32>;
@group(0) @binding(2) var<storage, read_write> b_scale: array<f32>;
@group(0) @binding(3) var<uniform> d: Dims;

{scheme_wgsl}

const BS: u32 = {bs}u;
const BPB: u32 = {bpb}u;
const ACTIVE_THREADS: u32 = {active_threads}u;
const SUBBLOCKS_PER_QBLOCK: u32 = {subblocks_per_qblock}u;

// Per-thread absmax scratch, indexed by lid.x. Reduced within each 8-thread
// sub-block group (in_sb = lid.x % 8).
var<workgroup> absmax: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let n: u32 = wid.y;
    let block_k_outer: u32 = wid.x;
    if (n >= d.n) {{ return; }}
    let blocks_per_row: u32 = d.k / BS;
    let block_idx: u32 = n * blocks_per_row + block_k_outer;
    let byte0: u32 = block_idx * BPB;
    let is_active: bool = lid.x < ACTIVE_THREADS;
    let st = {init_call}(byte0);
    let elem_start: u32 = lid.x * 4u;
    var v: vec4<f32> = vec4<f32>(0.0);
    if (is_active) {{
        v = {elem4_call}(byte0, st, elem_start);
    }}
    // Thread-local absmax over the 4 elements.
    let tabs: f32 = max(max(abs(v.x), abs(v.y)), max(abs(v.z), abs(v.w)));
    absmax[lid.x] = tabs;
    workgroupBarrier();
    // Tree reduce within each 8-thread sub-block group: 8 -> 4 -> 2 -> 1.
    let sb: u32 = lid.x / 8u;
    let in_sb: u32 = lid.x % 8u;
    if (is_active && in_sb < 4u) {{
        absmax[lid.x] = max(absmax[lid.x], absmax[lid.x + 4u]);
    }}
    workgroupBarrier();
    if (is_active && in_sb < 2u) {{
        absmax[lid.x] = max(absmax[lid.x], absmax[lid.x + 2u]);
    }}
    workgroupBarrier();
    if (is_active && in_sb == 0u) {{
        absmax[lid.x] = max(absmax[lid.x], absmax[lid.x + 1u]);
    }}
    workgroupBarrier();
    let amax: f32 = absmax[sb * 8u];
    let scale: f32 = amax / 127.0;
    let inv_scale: f32 = select(0.0, 1.0 / scale, scale > 0.0);
    // Scale write: one per sub-block. Layout [N, K/32], indexed
    // n * (K/32) + (block_k_outer * SUBBLOCKS_PER_QBLOCK + sb).
    if (is_active && in_sb == 0u) {{
        let subblocks_per_row: u32 = d.k / 32u;
        let sub_idx: u32 = n * subblocks_per_row + block_k_outer * SUBBLOCKS_PER_QBLOCK + sb;
        b_scale[sub_idx] = scale;
    }}
    // Pack 4 i8 to one u32. Layout [N, K/4], indexed
    // n * (K/4) + block_k_outer * (BS/4) + lid.x.
    if (is_active) {{
        let q4: vec4<f32> = clamp(
            round(v * inv_scale),
            vec4<f32>(-127.0),
            vec4<f32>(127.0),
        );
        let word: u32 = pack4xI8(vec4<i32>(q4));
        let words_per_row: u32 = d.k / 4u;
        let word_idx: u32 = n * words_per_row + block_k_outer * (BS / 4u) + lid.x;
        b_i8[word_idx] = word;
    }}
}}
"#
    )
}

pub struct DequantI8Bufs<'a> {
    pub b_quant: &'a BufRef,
    pub b_i8: &'a BufRef,
    pub b_scale: &'a BufRef,
    pub dims: &'a BufRef,
}

/// Dispatch one I8 dequant pass. `n` and `k` are the dense matrix dimensions
/// of B (N rows of K elements each in the dequanted output).
pub fn dispatch_dequant_i8<B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    scheme: QuantKind,
    bufs: &DequantI8Bufs<'_>,
    n: u32,
    k: u32,
) -> Result<(), B::Error> {
    let bs = scheme.block_size();
    assert!(
        k.is_multiple_of(bs),
        "dequant_i8: K={k} must be multiple of block_size={bs}",
    );
    let blocks_per_row = k / bs;
    let bindings = [
        bufs.b_quant.binding(0),
        bufs.b_i8.binding(1),
        bufs.b_scale.binding(2),
        bufs.dims.binding(3),
    ];
    backend.dispatch(encoder, pipeline, &bindings, [blocks_per_row, n, 1])
}
