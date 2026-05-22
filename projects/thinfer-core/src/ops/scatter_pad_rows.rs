//! Broadcast a bf16-packed pad row into mask-selected rows of an fp32 buffer.
//!
//! Used by Z-Image DiT to substitute `x_pad_token` / `cap_pad_token` into the
//! padded positions of the embedded activation tensor. Unifies the load path
//! with every other weight in the model: bf16 on GPU, decoded to fp32 inside
//! the kernel via the standard `(half << 16u)` bitcast. The previous
//! implementation byte-copied the weight directly, which broke the moment
//! weights moved to bf16 storage.
//!
//! Shapes:
//! - `pad:  [dim]` bf16-packed (`dim/2` u32 words; tail half is zero when dim is odd)
//! - `mask: [n_rows]` u32 (0 = leave, nonzero = overwrite)
//! - `dst:  [n_rows, dim]` fp32 row-major; mask-selected rows are overwritten,
//!   others are left untouched
//!
//! Layout: 0=Pad, 1=Mask, 2=Dst, 3=Uniform.

use super::{ActDtype, WgslConfig};
use crate::act_f16_prelude;
use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
use crate::tensor::{ComputeDtype, F32};

pub trait ScatterPadRowsOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const PAD: &'static str;
    const MASK: &'static str;
    const DIMS: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n_elems: u32) -> [u32; 3] {
        super::linear_workgroups(n_elems, 64)
    }
}

pub struct ScatterPadRowsBufs<'a> {
    pub pad: &'a BufRef,
    pub mask: &'a BufRef,
    pub uniform: &'a BufRef,
    pub dst: &'a BufRef,
}

pub(crate) fn dispatch_scatter_pad_rows<O: ScatterPadRowsOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &ScatterPadRowsBufs<'_>,
    n_elems: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.pad.binding(0),
        bufs.mask.binding(1),
        bufs.dst.binding(2),
        bufs.uniform.binding(3),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_elems))
}

const WGSL_F32: &str = r#"
struct U { n_rows: u32, dim: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> pad: array<u32>;
@group(0) @binding(1) var<storage, read> mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    let total = u.n_rows * u.dim;
    if (i >= total) { return; }
    let row = i / u.dim;
    if (mask[row] == 0u) { return; }
    let col = i - row * u.dim;
    let pair = pad[col >> 1u];
    let half = (pair >> ((col & 1u) * 16u)) & 0xFFFFu;
    dst[i] = bitcast<f32>(half << 16u);
}
"#;

// Packed-bf16 path: pad and dst share the same bf16-packed layout, so the
// kernel is a word-wise copy gated by the per-row mask. `n_elems` from the
// caller is still the elem count; words per row = dim / 2.
const WGSL_BF16_PACKED: &str = r#"
struct U { n_rows: u32, dim: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> pad: array<u32>;
@group(0) @binding(1) var<storage, read> mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    let words_per_row = u.dim >> 1u;
    let total_words = u.n_rows * words_per_row;
    if (i >= total_words) { return; }
    let row = i / words_per_row;
    if (mask[row] == 0u) { return; }
    let col_word = i - row * words_per_row;
    dst[i] = pad[col_word];
}
"#;

// Native f16 dst, bf16-packed pad weight source. Convert each bf16 pair to
// vec2<f16> on the fly (bf16's 7-bit mantissa fits losslessly in f16's
// 10-bit mantissa for non-overflowing values; the embedding pad token
// magnitudes are O(1)). Pure mask-gated copy otherwise.
const WGSL_F16_PACKED: &str = concat!(
    act_f16_prelude!(),
    r#"
struct U { n_rows: u32, dim: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> pad: array<u32>;
@group(0) @binding(1) var<storage, read> mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f16>>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    let words_per_row = u.dim >> 1u;
    let total_words = u.n_rows * words_per_row;
    if (i >= total_words) { return; }
    let row = i / words_per_row;
    if (mask[row] == 0u) { return; }
    let col_word = i - row * words_per_row;
    let p = pad[col_word];
    let lo = bitcast<f32>((p & 0xFFFFu) << 16u);
    let hi = bitcast<f32>(p & 0xFFFF0000u);
    dst[i] = vec2<f16>(vec2<f32>(lo, hi));
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
        kind: BindingKind::StorageRead,
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

pub struct ScatterPadRowsF32;

impl ScatterPadRowsOp for ScatterPadRowsF32 {
    const KERNEL_ID: &'static str = "scatter_pad_rows.f32";
    type Dtype = F32;
    const PAD: &'static str = "scatter_pad_rows/pad";
    const MASK: &'static str = "scatter_pad_rows/mask";
    const DIMS: &'static str = "scatter_pad_rows/dims";
    const OUTPUT: &'static str = "scatter_pad_rows/dst";
    fn wgsl(cfg: &WgslConfig) -> &'static str {
        match cfg.act_dtype {
            ActDtype::F32 => WGSL_F32,
            ActDtype::Bf16 => WGSL_BF16_PACKED,
            ActDtype::F16 => WGSL_F16_PACKED,
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}
