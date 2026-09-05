//! Split a fused QKV matmul output into three contiguous slabs.
//!
//! Shape:
//! - `input:  [rows, 3*h]` activation dtype (q slab at cols [0, h), k slab at
//!   [h, 2h), v slab at [2h, 3h)). Matches canonical upstream fused QKV
//!   layout: matmul with weight `[3h, k]` (N-major) produces this directly.
//! - `q:      [rows, h]`, `k: [rows, h]`, `v: [rows, h]` activation dtype.
//!
//! Sibling to the `Op` trait: multi-output, so the standard `dispatch_op` path
//! does not fit. One dispatch with 4 storage bindings (input + three outputs)
//! plus one uniform.
//!
//! Bf16-packed path: storage is `array<u32>`; `h` must be even so each row
//! packs into `h/2` words per slab. Z-Image has h = n_heads * head_dim = 3840.

use super::{ActDtype, WgslConfig};
use crate::act_f16_prelude;
use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
#[cfg(feature = "conformance")]
use crate::conformance::{Dtype, OpSpec, OpTest, OpTestContext, TestCase, linspace, t};
use crate::tensor::{ComputeDtype, F32};

pub trait QkvSplitOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const INPUT: &'static str;
    const OUTPUT_Q: &'static str;
    const OUTPUT_K: &'static str;
    const OUTPUT_V: &'static str;
    const DIMS: &'static str;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n_words: u32) -> [u32; 3] {
        super::linear_workgroups(n_words, 64)
    }
}

pub struct QkvSplitBufs<'a> {
    pub input: &'a BufRef,
    pub q: &'a BufRef,
    pub k: &'a BufRef,
    pub v: &'a BufRef,
    pub uniform: &'a BufRef,
}

pub(crate) fn dispatch_qkv_split<O: QkvSplitOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &QkvSplitBufs<'_>,
    n_words: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.input.binding(0),
        bufs.q.binding(1),
        bufs.k.binding(2),
        bufs.v.binding(3),
        bufs.uniform.binding(4),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_words))
}

// One thread per output element (per slab). Reads three values per row from
// the fused input, writes one into each of q/k/v. Dispatch count is
// `rows * h` (per-slab element count); each thread writes one slab element.
const WGSL_F32: &str = r#"
struct U { rows: u32, h: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
@group(0) @binding(2) var<storage, read_write> k: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    let total = u.rows * u.h;
    if (i >= total) { return; }
    let row = i / u.h;
    let col = i - row * u.h;
    let fused_base = row * (3u * u.h);
    q[i] = x[fused_base + col];
    k[i] = x[fused_base + u.h + col];
    v[i] = x[fused_base + 2u * u.h + col];
}
"#;

// Bf16-packed path. h must be even; words per slab row = h/2, fused row =
// 3*h/2 words. Workgroups dispatched over `rows * (h/2)` words per slab.
const WGSL_BF16_PACKED: &str = r#"
struct U { rows: u32, h: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> q: array<u32>;
@group(0) @binding(2) var<storage, read_write> k: array<u32>;
@group(0) @binding(3) var<storage, read_write> v: array<u32>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    let slab_words = u.h >> 1u;
    let total_words = u.rows * slab_words;
    if (i >= total_words) { return; }
    let row = i / slab_words;
    let col_word = i - row * slab_words;
    let fused_row_words = 3u * slab_words;
    let base = row * fused_row_words;
    q[i] = x[base + col_word];
    k[i] = x[base + slab_words + col_word];
    v[i] = x[base + 2u * slab_words + col_word];
}
"#;

// Native f16 path. Same shape as bf16-packed (4-byte stride, h must be
// even) but typed as `array<vec2<f16>>`. No arithmetic — pure copy.
const WGSL_F16_PACKED: &str = concat!(
    act_f16_prelude!(),
    r#"
struct U { rows: u32, h: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> x: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read_write> q: array<vec2<f16>>;
@group(0) @binding(2) var<storage, read_write> k: array<vec2<f16>>;
@group(0) @binding(3) var<storage, read_write> v: array<vec2<f16>>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    let slab_words = u.h >> 1u;
    let total_words = u.rows * slab_words;
    if (i >= total_words) { return; }
    let row = i / slab_words;
    let col_word = i - row * slab_words;
    let fused_row_words = 3u * slab_words;
    let base = row * fused_row_words;
    q[i] = x[base + col_word];
    k[i] = x[base + slab_words + col_word];
    v[i] = x[base + 2u * slab_words + col_word];
}
"#
);

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
        kind: BindingKind::StorageReadWrite,
    },
    BindingLayout {
        slot: 4,
        kind: BindingKind::Uniform,
    },
];

pub struct QkvSplitF32;

impl QkvSplitOp for QkvSplitF32 {
    const KERNEL_ID: &'static str = "qkv_split.f32";
    type Dtype = F32;
    const INPUT: &'static str = "qkv_split/x";
    const OUTPUT_Q: &'static str = "qkv_split/q";
    const OUTPUT_K: &'static str = "qkv_split/k";
    const OUTPUT_V: &'static str = "qkv_split/v";
    const DIMS: &'static str = "qkv_split/dims";
    fn wgsl(cfg: &WgslConfig) -> &'static str {
        match cfg.act_dtype {
            ActDtype::F32 => WGSL_F32,
            ActDtype::Bf16 => WGSL_BF16_PACKED,
            ActDtype::F16 => WGSL_F16_PACKED,
            ActDtype::I8 => unreachable!("ActDtype::I8 is never a block-level act dtype"),
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

// ---------------------------------------------------------------------------
// PER-HEAD INTERLEAVED split (MiniMax-H3).
//
// H3's raw checkpoint packs `qkv_proj`'s output axis as
// `[h0: q,k,v][h1: q,k,v]...`, i.e. `(rows, heads, 3, head_dim)`, NOT the three
// contiguous slabs `QkvSplitF32` assumes. Reading it with the contiguous kernel
// silently interleaves heads across q/k/v and produces structured garbage with
// no NaN anywhere.
//
// The reference handles this by reshaping in the forward pass rather than
// rewriting weights, and so do we: permuting the projection's output rows at
// load would have to re-block the quantized tensor, which is exactly the kind of
// silent weight surgery that bites.
//
// `head_dim` rides in the uniform's third slot. It must be EVEN so the packed
// paths stay word-aligned (H3's is 128).

const WGSL_HEADWISE_F32: &str = r#"
struct U { rows: u32, h: u32, head_dim: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
@group(0) @binding(2) var<storage, read_write> k: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    let total = u.rows * u.h;
    if (i >= total) { return; }
    let row = i / u.h;
    let col = i - row * u.h;
    let head = col / u.head_dim;
    let d = col - head * u.head_dim;
    // Row stride is 3*h; within a row each head owns 3*head_dim contiguous.
    let base = row * (3u * u.h) + head * (3u * u.head_dim) + d;
    q[i] = x[base];
    k[i] = x[base + u.head_dim];
    v[i] = x[base + 2u * u.head_dim];
}
"#;

const WGSL_HEADWISE_BF16_PACKED: &str = r#"
struct U { rows: u32, h: u32, head_dim: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> q: array<u32>;
@group(0) @binding(2) var<storage, read_write> k: array<u32>;
@group(0) @binding(3) var<storage, read_write> v: array<u32>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    let slab_words = u.h >> 1u;
    let total_words = u.rows * slab_words;
    if (i >= total_words) { return; }
    let hd_words = u.head_dim >> 1u;
    let row = i / slab_words;
    let w = i - row * slab_words;
    let head = w / hd_words;
    let dw = w - head * hd_words;
    let base = row * (3u * slab_words) + head * (3u * hd_words) + dw;
    q[i] = x[base];
    k[i] = x[base + hd_words];
    v[i] = x[base + 2u * hd_words];
}
"#;

const WGSL_HEADWISE_F16_PACKED: &str = concat!(
    act_f16_prelude!(),
    r#"
struct U { rows: u32, h: u32, head_dim: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read> x: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read_write> q: array<vec2<f16>>;
@group(0) @binding(2) var<storage, read_write> k: array<vec2<f16>>;
@group(0) @binding(3) var<storage, read_write> v: array<vec2<f16>>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    let slab_words = u.h >> 1u;
    let total_words = u.rows * slab_words;
    if (i >= total_words) { return; }
    let hd_words = u.head_dim >> 1u;
    let row = i / slab_words;
    let w = i - row * slab_words;
    let head = w / hd_words;
    let dw = w - head * hd_words;
    let base = row * (3u * slab_words) + head * (3u * hd_words) + dw;
    q[i] = x[base];
    k[i] = x[base + hd_words];
    v[i] = x[base + 2u * hd_words];
}
"#
);

/// Split a fused QKV output whose columns are per-head interleaved
/// `(rows, heads, 3, head_dim)`. See the section comment above.
pub struct QkvSplitHeadwiseF32;

impl QkvSplitOp for QkvSplitHeadwiseF32 {
    const KERNEL_ID: &'static str = "qkv_split.headwise.f32";
    type Dtype = F32;
    const INPUT: &'static str = "qkv_split_headwise/x";
    const OUTPUT_Q: &'static str = "qkv_split_headwise/q";
    const OUTPUT_K: &'static str = "qkv_split_headwise/k";
    const OUTPUT_V: &'static str = "qkv_split_headwise/v";
    const DIMS: &'static str = "qkv_split_headwise/dims";
    fn wgsl(cfg: &WgslConfig) -> &'static str {
        match cfg.act_dtype {
            ActDtype::F32 => WGSL_HEADWISE_F32,
            ActDtype::Bf16 => WGSL_HEADWISE_BF16_PACKED,
            ActDtype::F16 => WGSL_HEADWISE_F16_PACKED,
            ActDtype::I8 => unreachable!("ActDtype::I8 is never a block-level act dtype"),
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl OpTest for QkvSplitHeadwiseF32 {
    /// No `bf16w`: this is a pure data-movement op with no quant-writes kernel
    /// variant, so under that dtype it copies exact fp32 while the reference is
    /// rounded, and the delta is just bf16 epsilon rather than a real defect.
    /// Both dtypes a block actually runs it at (`F32`, packed `Bf16`) are here.
    fn dtypes(&self) -> &'static [Dtype] {
        &[Dtype::Fp32, Dtype::Bf16Packed]
    }
    fn test_cases(&self) -> Vec<TestCase> {
        vec![
            // Shapes are chosen so each q/k/v third of the packed output lands
            // on a 256-byte storage-binding boundary: rows * heads * head_dim
            // must be a multiple of 64 fp32 elements.
            //
            // 4 heads x head_dim 16 over 4 rows. The interleave is the whole
            // point: head h's q/k/v sit at h*3*head_dim, NOT in three
            // contiguous blocks, so a block-contiguous kernel mixes heads
            // across q/k/v and still produces finite, plausible values.
            TestCase {
                name: "qkv_split_headwise_basic",
                op: OpSpec::QkvSplitHeadwise,
                inputs: vec![t("x", [4, 4, 3, 16], linspace(-1.0, 1.0, false))],
            },
            // The H3 head_dim (128) at a single head: catches word-alignment
            // mistakes in the packed bf16/f16 kernels.
            TestCase {
                name: "qkv_split_headwise_h3_head_dim",
                op: OpSpec::QkvSplitHeadwise,
                inputs: vec![t("x", [2, 1, 3, 128], linspace(-1.5, 1.5, false))],
            },
        ]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_qkv_split_headwise::<QkvSplitHeadwiseF32>())
    }
}
