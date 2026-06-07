//! Per-K=32-block sum of a bf16 weight tensor.
//!
//! For matmul_i8_bf16's asymmetric-acts correction term:
//!     `O += za[m,t] * Σ_{k in block t} b[n, k]`
//!
//! This kernel materializes `b_sum[n, t] = Σ_{k in block t} b[n, k]` as f32
//! from a bf16 K-major weight `b[N, K]` (stored as `array<u32>`, 2 bf16
//! elements per word, element (k, n) at word index `(k*N + n) >> 1`).
//!
//! Mirrors the architectural pattern of `dequant_i8`: a small per-dispatch
//! GPU pass that produces matmul-ready side data into `scope.alloc`, not a
//! weight-load-time precompute. Keeps the bf16-weight path consistent with
//! the Quant-weight path (dequant_i8 also runs per dispatch into scope).

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

pub fn hint() -> &'static str {
    "bf16_block_sum"
}

/// Build the bf16-block-sum WGSL. One workgroup processes 64 (n, t) pairs;
/// each thread sums its 32 bf16 elements serially. Pure ALU + load, no
/// shared memory.
pub fn build_wgsl() -> String {
    r#"struct Dims { n: u32, k: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> b: array<u32>;
@group(0) @binding(1) var<storage, read_write> b_sum: array<f32>;
@group(0) @binding(2) var<uniform> d: Dims;

fn unpack_bf16(w: u32, lane: u32) -> f32 {
    let half: u32 = (w >> (lane * 16u)) & 0xFFFFu;
    return bitcast<f32>(half << 16u);
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ng: vec3<u32>,
) {
    let blocks_per_row: u32 = d.k / 32u;
    let total: u32 = d.n * blocks_per_row;
    let i: u32 = gid.y * (ng.x * 64u) + gid.x;
    if (i >= total) { return; }
    let n: u32 = i / blocks_per_row;
    let t: u32 = i % blocks_per_row;
    var s: f32 = 0.0;
    for (var kk: u32 = 0u; kk < 32u; kk = kk + 1u) {
        let k: u32 = t * 32u + kk;
        let elem_idx: u32 = k * d.n + n;
        let lane: u32 = elem_idx & 1u;
        let w: u32 = b[elem_idx >> 1u];
        s = s + unpack_bf16(w, lane);
    }
    b_sum[i] = s;
}
"#
    .to_string()
}

pub struct Bf16BlockSumBufs<'a> {
    pub b: &'a BufRef,
    pub b_sum: &'a BufRef,
    pub dims: &'a BufRef,
}

pub fn workgroups(n: u32, k: u32) -> [u32; 3] {
    let total = n * (k / 32);
    crate::ops::linear_workgroups(total, 64)
}

pub fn dispatch_bf16_block_sum<B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &Bf16BlockSumBufs<'_>,
    n: u32,
    k: u32,
) -> Result<(), B::Error> {
    assert!(
        k.is_multiple_of(32),
        "bf16_block_sum: K={k} must be multiple of 32",
    );
    let bindings = [
        bufs.b.binding(0),
        bufs.b_sum.binding(1),
        bufs.dims.binding(2),
    ];
    backend.dispatch(encoder, pipeline, &bindings, workgroups(n, k))
}
