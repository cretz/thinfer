//! Block-quantized weight -> dense bf16-packed dequant pass.
//!
//! Materializes a quant weight tensor (Q4_K/Q5_K/Q6_K/Q8_0/Q4_0) into a
//! bf16-packed `[N, K]` dense buffer, ready for a plain bf16 matmul. Run
//! once per matmul site to eliminate the in-matmul re-dequant of the same
//! B columns across `M/BM` output tiles (4x redundancy at bm=64, M=256).
//!
//! Output layout matches `MatMulConfig.b_nmajor = true`: row n contains
//! that column's K elements contiguously (n*K + k), so dequant writes are
//! coalesced and matmul reads stay N-major.
//!
//! One workgroup per quant block, 64 threads/WG, each thread vec4-dequants
//! 4 K-adjacent elements via the existing `block_elem4_<k>` helpers in
//! `quant.rs`, then writes 2 u32 (4 packed bf16) at coalesced positions.
//! 64 threads × 4 elems = 256 = max block_size, so one WG fills one block.

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
        kind: BindingKind::Uniform,
    },
];

pub fn layout() -> &'static [BindingLayout] {
    LAYOUT
}

/// Output encoding for the dense `[N, K]` workspace the dequant kernel emits.
/// Both variants stride 4 bytes per pair (32-bit-aligned storage). `Bf16`
/// emits `array<u32>` with RNE-rounded bf16-packed pairs; `F16` emits
/// `array<vec2<f16>>` natively when the adapter exposes `SHADER_F16`. The
/// F16 path skips the bf16-round helper on the producer side AND the
/// bf16-unpack helper on the consumer (matmul) side, so the writeback +
/// readback round-trip costs zero conversions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DequantTarget {
    Bf16,
    F16,
}

/// Pipeline-cache discriminator. One pipeline per (quant scheme, target).
pub fn hint(scheme: QuantKind, target: DequantTarget) -> String {
    let t = match target {
        DequantTarget::Bf16 => "bf16",
        DequantTarget::F16 => "f16",
    };
    format!("dequant-{}-{t}", scheme.hint())
}

/// Build the dequant WGSL for one quant scheme + output target. Output is a
/// dense `[N, K]` workspace in either bf16-packed `array<u32>` or native
/// `array<vec2<f16>>` (4-byte stride either way). Caller dispatches
/// `(K/block_size, N, 1)` workgroups.
pub fn build_wgsl(scheme: QuantKind, target: DequantTarget) -> String {
    let scheme_wgsl = scheme.wgsl();
    let init_call = scheme.block_state_call();
    let elem4_call = scheme.block_elem4_call();
    let bs = scheme.block_size();
    let bpb = scheme.bytes_per_block();
    // 64 threads × 4 elements/thread = 256 = max block_size. For Q8_0 / Q4_0
    // (bs=32), only the first 8 threads do work; the rest early-exit. Keeps
    // workgroup_size uniform across schemes.
    assert!(
        bs <= 256,
        "dequant kernel assumes block_size <= 256 (got {bs})"
    );
    let elems_per_thread = 4u32;
    let active_threads = bs / elems_per_thread;
    let (prelude, dense_decl, pack_helpers, store_lines) = match target {
        DequantTarget::Bf16 => (
            "",
            "@group(0) @binding(1) var<storage, read_write> b_dense: array<u32>;",
            // bf16-pack helpers identical to the original kernel.
            "fn round_bf16(x: f32) -> u32 {\n    let b32 = bitcast<u32>(x);\n    if ((b32 & 0x7F800000u) == 0x7F800000u) { return (b32 >> 16u) & 0xFFFFu; }\n    let l = (b32 >> 16u) & 1u;\n    return ((b32 + 0x7FFFu + l) >> 16u) & 0xFFFFu;\n}\nfn pack_bf16x2(lo: f32, hi: f32) -> u32 {\n    return round_bf16(lo) | (round_bf16(hi) << 16u);\n}\n",
            // Each 4-element vec4 writes two packed bf16 words.
            "    b_dense[dense_off >> 1u] = pack_bf16x2(v.x, v.y);\n    b_dense[(dense_off + 2u) >> 1u] = pack_bf16x2(v.z, v.w);",
        ),
        DequantTarget::F16 => (
            "enable f16;\n",
            "@group(0) @binding(1) var<storage, read_write> b_dense: array<vec2<f16>>;",
            // Q-block scales bound dequant outputs well inside the f16 finite
            // range, but clamp defensively on the saturated narrow path to
            // avoid +-inf propagating into the matmul accumulator on any
            // pathological scale.
            "",
            "    let lo01 = clamp(vec2<f32>(v.x, v.y), vec2<f32>(-65504.0), vec2<f32>(65504.0));\n    let lo23 = clamp(vec2<f32>(v.z, v.w), vec2<f32>(-65504.0), vec2<f32>(65504.0));\n    b_dense[dense_off >> 1u] = vec2<f16>(lo01);\n    b_dense[(dense_off + 2u) >> 1u] = vec2<f16>(lo23);",
        ),
    };
    format!(
        r#"{prelude}struct Dims {{ n: u32, k: u32, _pad0: u32, _pad1: u32 }};

@group(0) @binding(0) var<storage, read> b: array<u32>;
{dense_decl}
@group(0) @binding(2) var<uniform> d: Dims;

{scheme_wgsl}

{pack_helpers}
const BS: u32 = {bs}u;
const BPB: u32 = {bpb}u;
const ACTIVE_THREADS: u32 = {active_threads}u;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    if (lid.x >= ACTIVE_THREADS) {{ return; }}
    let block_k_outer: u32 = wid.x;
    let n: u32 = wid.y;
    if (n >= d.n) {{ return; }}
    let blocks_per_row: u32 = d.k / BS;
    let block_idx: u32 = n * blocks_per_row + block_k_outer;
    let byte0: u32 = block_idx * BPB;
    let st = {init_call}(byte0);
    let elem_start: u32 = lid.x * 4u;
    let v: vec4<f32> = {elem4_call}(byte0, st, elem_start);
    let dense_off: u32 = n * d.k + block_k_outer * BS + elem_start;
{store_lines}
}}
"#
    )
}

pub struct DequantBufs<'a> {
    pub b_quant: &'a BufRef,
    pub b_dense: &'a BufRef,
    pub dims: &'a BufRef,
}

/// Dispatch one dequant pass. `n` and `k` are the dense matrix dimensions
/// of B (N rows of K elements each in the dequanted output).
pub fn dispatch_dequant<B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    scheme: QuantKind,
    bufs: &DequantBufs<'_>,
    n: u32,
    k: u32,
) -> Result<(), B::Error> {
    let bs = scheme.block_size();
    assert!(
        k.is_multiple_of(bs),
        "dequant: K={k} must be multiple of block_size={bs}",
    );
    let blocks_per_row = k / bs;
    let bindings = [
        bufs.b_quant.binding(0),
        bufs.b_dense.binding(1),
        bufs.dims.binding(2),
    ];
    backend.dispatch(encoder, pipeline, &bindings, [blocks_per_row, n, 1])
}
