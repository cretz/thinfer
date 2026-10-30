use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
#[cfg(feature = "conformance")]
use crate::conformance::{
    DTYPES_ACT_BF16, Dtype, OpSpec, OpTest, OpTestContext, TestCase, linspace, t,
};
use crate::ops::{ActDtype, WgslConfig};
use crate::tensor::{ComputeDtype, F32};
use crate::{act_bf16_prelude, act_f16_prelude, wgsl_with_bf16_variant};

/// Rotary embedding via complex-pair multiply, broadcasting freqs across heads.
///
/// `x: [rows, heads, dim]`, `freqs: [rows, dim]`, `out: [rows, heads, dim]`.
/// `dim` is even; the last axis is interleaved (real, imag) pairs.
/// Per pair: `(xr+i*xi) * (cr+i*ci) = (xr*cr - xi*ci) + i*(xr*ci + xi*cr)`.
///
/// Z-Image's 3-axis RoPE is encoded by the caller into `freqs` (concatenated
/// per-axis frequencies along the last dim); the kernel itself is axis-agnostic.
///
/// Layout: 0=X, 1=Freqs, 2=Out, 3=Uniform `{rows, heads, pairs, _pad}` (pairs = dim/2).
pub trait RopeOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const FREQS: &'static str;
    const DIMS: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(rows: u32, heads: u32, pairs: u32) -> [u32; 3] {
        // Kernels read `gid.y * (ng.x * 64u) + gid.x` so workgroup counts
        // > 65535 spill to Y. At 1024x1024 default-resolution, seq_x = 4096
        // and n_heads = 30 give rows*heads*pairs/64 = 30720 per dispatch on
        // the DiT main path; the noise-refiner's per-axis rope can push
        // that past the 65535 cap, hence the spill.
        super::linear_workgroups(rows * heads * pairs, 64)
    }
}

pub struct RopeBufs<'a> {
    pub x: &'a BufRef,
    pub freqs: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_rope<O: RopeOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &RopeBufs<'_>,
    rows: u32,
    heads: u32,
    pairs: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.freqs.binding(1),
        bufs.out.binding(2),
        bufs.uniform.binding(3),
    ];
    backend.dispatch(
        encoder,
        pipeline,
        &bindings,
        O::workgroups(rows, heads, pairs),
    )
}

wgsl_with_bf16_variant!(
    WGSL_F32,
    WGSL_F32_BF16 = r#"
struct U { rows: u32, heads: u32, pairs: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> freqs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let total = u.rows * u.heads * u.pairs;
    let idx = gid.y * (ng.x * 64u) + gid.x;
    if (idx >= total) { return; }
    let pair = idx % u.pairs;
    let rh   = idx / u.pairs;
    let row  = rh / u.heads;
    let dim  = u.pairs * 2u;
    let x_off = rh  * dim + pair * 2u;
    let f_off = row * dim + pair * 2u;
    let xr = x[x_off];
    let xi = x[x_off + 1u];
    let cr = freqs[f_off];
    let ci = freqs[f_off + 1u];
    out[x_off]      = act_store(xr * cr - xi * ci);
    out[x_off + 1u] = act_store(xr * ci + xi * cr);
}
"#
);

/// Packed-bf16 interleaved rope. One thread = one rotary pair = one packed
/// word (since adjacent (re, im) elements share a word). Freqs same layout
/// as x: per-pair (cr, ci) lives in one word at `row*pairs + pair`.
const WGSL_BF16_PACKED: &str = concat!(
    act_bf16_prelude!(),
    r#"
struct U { rows: u32, heads: u32, pairs: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> freqs: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let total = u.rows * u.heads * u.pairs;
    let idx = gid.y * (ng.x * 64u) + gid.x;
    if (idx >= total) { return; }
    let pair = idx % u.pairs;
    let rh   = idx / u.pairs;
    let row  = rh / u.heads;
    let xw_idx = rh  * u.pairs + pair;
    let fw_idx = row * u.pairs + pair;
    let xv = unpack_bf16x2(x[xw_idx]);
    let fv = unpack_bf16x2(freqs[fw_idx]);
    let or_ = xv.x * fv.x - xv.y * fv.y;
    let oi  = xv.x * fv.y + xv.y * fv.x;
    out[xw_idx] = pack_bf16x2(or_, oi);
}
"#
);

// Native f16 interleaved rope. Freqs follow the act dtype: in the F16 path
// the RopeEmbedder uploads f16-packed freqs. Pair-rotation is two muls and
// two add/subs in vec2<f16> — well within f16 dynamic range (|cos|<=1).
const WGSL_F16_PACKED: &str = concat!(
    act_f16_prelude!(),
    r#"
struct U { rows: u32, heads: u32, pairs: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read> x: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read> freqs: array<vec2<f16>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec2<f16>>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let total = u.rows * u.heads * u.pairs;
    let idx = gid.y * (ng.x * 64u) + gid.x;
    if (idx >= total) { return; }
    let pair = idx % u.pairs;
    let rh   = idx / u.pairs;
    let row  = rh / u.heads;
    let xw_idx = rh  * u.pairs + pair;
    let fw_idx = row * u.pairs + pair;
    let xv: vec2<f16> = x[xw_idx];
    let fv: vec2<f16> = freqs[fw_idx];
    out[xw_idx] = vec2<f16>(xv.x * fv.x - xv.y * fv.y, xv.x * fv.y + xv.y * fv.x);
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

pub struct RopeF32;

impl RopeOp for RopeF32 {
    const KERNEL_ID: &'static str = "rope.f32";
    type Dtype = F32;
    const X: &'static str = "rope/x";
    const FREQS: &'static str = "rope/freqs";
    const DIMS: &'static str = "rope/dims";
    const OUTPUT: &'static str = "rope/out";
    fn wgsl(cfg: &WgslConfig) -> &'static str {
        match (cfg.act_dtype, cfg.bf16_quant_writes) {
            (ActDtype::F32, false) => WGSL_F32,
            (ActDtype::F32, true) => WGSL_F32_BF16,
            (ActDtype::Bf16, _) => WGSL_BF16_PACKED,
            (ActDtype::F16, _) => WGSL_F16_PACKED,
            (ActDtype::I8, _) => unreachable!("ActDtype::I8 is never a block-level act dtype"),
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

// ---------------------------------------------------------------------------
// Half-rotation (NEOX / HuggingFace) variant: pair k = (x[k], x[k + D/2]).
// ---------------------------------------------------------------------------
//
// HF-style LLMs (LLaMA, Qwen, Mistral, ...) place the real/imag halves of
// each rotary pair at indices `k` and `k + D/2` rather than `2k` and `2k+1`.
// The rotation math is identical; only the x-index pattern changes. Freqs are
// kept in the same pair-interleaved layout `[cos_0, sin_0, cos_1, sin_1, ...]`
// as the standard variant so callers reuse `RopeEmbedder::lookup_bytes`.
//
// Pick this op (over `RopeF32`) when the upstream model trains with HF's
// `rotate_half`. Z-Image's DiT continues to use the interleaved `RopeF32`.
wgsl_with_bf16_variant!(
    WGSL_F32_HALFROT,
    WGSL_F32_HALFROT_BF16 = r#"
struct U { rows: u32, heads: u32, pairs: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> freqs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let total = u.rows * u.heads * u.pairs;
    let idx = gid.y * (ng.x * 64u) + gid.x;
    if (idx >= total) { return; }
    let pair = idx % u.pairs;
    let rh   = idx / u.pairs;
    let row  = rh / u.heads;
    let dim  = u.pairs * 2u;
    let x_re_off = rh  * dim + pair;
    let x_im_off = x_re_off + u.pairs;
    let f_off    = row * dim + pair * 2u;
    let xr = x[x_re_off];
    let xi = x[x_im_off];
    let cr = freqs[f_off];
    let ci = freqs[f_off + 1u];
    out[x_re_off] = act_store(xr * cr - xi * ci);
    out[x_im_off] = act_store(xr * ci + xi * cr);
}
"#
);

/// Packed-bf16 half-rot rope. Real (x[k]) and imag (x[k+pairs]) halves are
/// non-adjacent, so per-thread covers 2 consecutive pairs (k=2j, k=2j+1)
/// whose real elements share one packed word and whose imag elements share
/// the next-half-row packed word. Requires `pairs % 2 == 0`.
const WGSL_BF16_PACKED_HALFROT: &str = concat!(
    act_bf16_prelude!(),
    r#"
struct U { rows: u32, heads: u32, pairs: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> freqs: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let pair_words = u.pairs >> 1u;
    let total = u.rows * u.heads * pair_words;
    let idx = gid.y * (ng.x * 64u) + gid.x;
    if (idx >= total) { return; }
    let j  = idx % pair_words;
    let rh = idx / pair_words;
    let row = rh / u.heads;
    let row_w_base = rh * u.pairs;
    let xr_w = row_w_base + j;
    let xi_w = row_w_base + pair_words + j;
    let frow_w_base = row * u.pairs;
    let f0 = unpack_bf16x2(freqs[frow_w_base + 2u * j]);
    let f1 = unpack_bf16x2(freqs[frow_w_base + 2u * j + 1u]);
    let xr = unpack_bf16x2(x[xr_w]);
    let xi = unpack_bf16x2(x[xi_w]);
    let or0 = xr.x * f0.x - xi.x * f0.y;
    let or1 = xr.y * f1.x - xi.y * f1.y;
    let oi0 = xr.x * f0.y + xi.x * f0.x;
    let oi1 = xr.y * f1.y + xi.y * f1.x;
    out[xr_w] = pack_bf16x2(or0, or1);
    out[xi_w] = pack_bf16x2(oi0, oi1);
}
"#
);

// Native f16 half-rot. Real/imag halves live `pair_words` apart; per-thread
// covers two consecutive pairs whose real elements share one f16 pair word
// and imag elements share another. Requires `pairs % 2 == 0`.
const WGSL_F16_PACKED_HALFROT: &str = concat!(
    act_f16_prelude!(),
    r#"
struct U { rows: u32, heads: u32, pairs: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read> x: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read> freqs: array<vec2<f16>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec2<f16>>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let pair_words = u.pairs >> 1u;
    let total = u.rows * u.heads * pair_words;
    let idx = gid.y * (ng.x * 64u) + gid.x;
    if (idx >= total) { return; }
    let j  = idx % pair_words;
    let rh = idx / pair_words;
    let row = rh / u.heads;
    let row_w_base = rh * u.pairs;
    let xr_w = row_w_base + j;
    let xi_w = row_w_base + pair_words + j;
    let frow_w_base = row * u.pairs;
    let f0: vec2<f16> = freqs[frow_w_base + 2u * j];
    let f1: vec2<f16> = freqs[frow_w_base + 2u * j + 1u];
    let xr: vec2<f16> = x[xr_w];
    let xi: vec2<f16> = x[xi_w];
    out[xr_w] = vec2<f16>(xr.x * f0.x - xi.x * f0.y, xr.y * f1.x - xi.y * f1.y);
    out[xi_w] = vec2<f16>(xr.x * f0.y + xi.x * f0.x, xr.y * f1.y + xi.y * f1.x);
}
"#
);

pub struct RopeF32HalfRot;

impl RopeOp for RopeF32HalfRot {
    const KERNEL_ID: &'static str = "rope.f32.halfrot";
    type Dtype = F32;
    const X: &'static str = "rope_halfrot/x";
    const FREQS: &'static str = "rope_halfrot/freqs";
    const DIMS: &'static str = "rope_halfrot/dims";
    const OUTPUT: &'static str = "rope_halfrot/out";
    fn wgsl(cfg: &WgslConfig) -> &'static str {
        match (cfg.act_dtype, cfg.bf16_quant_writes) {
            (ActDtype::F32, false) => WGSL_F32_HALFROT,
            (ActDtype::F32, true) => WGSL_F32_HALFROT_BF16,
            (ActDtype::Bf16, _) => WGSL_BF16_PACKED_HALFROT,
            (ActDtype::F16, _) => WGSL_F16_PACKED_HALFROT,
            (ActDtype::I8, _) => unreachable!(
                "ActDtype::I8 halfrot is not implemented - Qwen3 (the only halfrot consumer) is not an I8 target"
            ),
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

// ---------------------------------------------------------------------------
// PARTIAL half-rot (MiniMax-H3): rotate only the leading `rot_pairs * 2` head
// channels and pass the rest through unchanged.
//
// H3 shares one 16-frequency `inv_freq` across three position axes, concatenates
// them to 48 and then concatenates that with itself, so 96 of each 128-dim head
// rotates and 32 channels pass through. The pairing is `(k, k + rot_pairs)`
// WITHIN the rotary block, i.e. `(k, k + 48)`, NOT `(k, k + head_dim/2)` --
// `RopeF32HalfRot` would pair `(k, k + 64)` and silently mix a rotated channel
// with a pass-through one.
//
// The alternative, permuting the qkv projection rows at load so the standard
// half-rot pairing lines up, is deliberately NOT taken: weight surgery of that
// shape scrambles silently when it is wrong (no NaN, just structured artifacts).
//
// Uniform reuses the `_pad` slot for `rot_pairs`, so `pairs` stays `head_dim/2`
// and `freqs` carries `rot_pairs` (cos, sin) entries per row.
wgsl_with_bf16_variant!(
    WGSL_F32_HALFROT_PARTIAL,
    WGSL_F32_HALFROT_PARTIAL_BF16 = r#"
struct U { rows: u32, heads: u32, pairs: u32, rot_pairs: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> freqs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let dim = u.pairs * 2u;
    let total = u.rows * u.heads * dim;
    let idx = gid.y * (ng.x * 64u) + gid.x;
    if (idx >= total) { return; }
    let c   = idx % dim;
    let rh  = idx / dim;
    let row = rh / u.heads;
    let base = rh * dim;
    let rot = u.rot_pairs * 2u;
    if (c >= rot) {
        out[base + c] = act_store(x[base + c]);
        return;
    }
    // cos/sin repeat across the two halves of the rotary block.
    let k = c % u.rot_pairs;
    let f_off = row * rot + k * 2u;
    let cr = freqs[f_off];
    let ci = freqs[f_off + 1u];
    if (c < u.rot_pairs) {
        out[base + c] = act_store(x[base + c] * cr - x[base + c + u.rot_pairs] * ci);
    } else {
        out[base + c] = act_store(x[base + c] * cr + x[base + c - u.rot_pairs] * ci);
    }
}
"#
);

// Packed variants: one thread per vec2 WORD. `rot_pairs` is even for H3 (48),
// so the rotary block ends on a word boundary and the halves stay word-aligned:
// word j covers elements (2j, 2j+1) and pairs with word `j + rot_pairs/2`.
// Threads in `[rot_pairs/2, rot_pairs)` are covered by their partner and return.
const WGSL_BF16_PACKED_HALFROT_PARTIAL: &str = concat!(
    act_bf16_prelude!(),
    r#"
struct U { rows: u32, heads: u32, pairs: u32, rot_pairs: u32 };

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> freqs: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let total = u.rows * u.heads * u.pairs;
    let idx = gid.y * (ng.x * 64u) + gid.x;
    if (idx >= total) { return; }
    let j  = idx % u.pairs;
    let rh = idx / u.pairs;
    let row = rh / u.heads;
    let base = rh * u.pairs;
    if (j >= u.rot_pairs) {
        out[base + j] = x[base + j];
        return;
    }
    let half_w = u.rot_pairs >> 1u;
    if (j >= half_w) { return; }
    let frow = row * u.rot_pairs;
    let f0 = unpack_bf16x2(freqs[frow + 2u * j]);
    let f1 = unpack_bf16x2(freqs[frow + 2u * j + 1u]);
    let xr = unpack_bf16x2(x[base + j]);
    let xi = unpack_bf16x2(x[base + j + half_w]);
    out[base + j] = pack_bf16x2(
        xr.x * f0.x - xi.x * f0.y,
        xr.y * f1.x - xi.y * f1.y);
    out[base + j + half_w] = pack_bf16x2(
        xr.x * f0.y + xi.x * f0.x,
        xr.y * f1.y + xi.y * f1.x);
}
"#
);

const WGSL_F16_PACKED_HALFROT_PARTIAL: &str = concat!(
    act_f16_prelude!(),
    r#"
struct U { rows: u32, heads: u32, pairs: u32, rot_pairs: u32 };

@group(0) @binding(0) var<storage, read> x: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read> freqs: array<vec2<f16>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec2<f16>>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let total = u.rows * u.heads * u.pairs;
    let idx = gid.y * (ng.x * 64u) + gid.x;
    if (idx >= total) { return; }
    let j  = idx % u.pairs;
    let rh = idx / u.pairs;
    let row = rh / u.heads;
    let base = rh * u.pairs;
    if (j >= u.rot_pairs) {
        out[base + j] = x[base + j];
        return;
    }
    let half_w = u.rot_pairs >> 1u;
    if (j >= half_w) { return; }
    let frow = row * u.rot_pairs;
    let f0: vec2<f16> = freqs[frow + 2u * j];
    let f1: vec2<f16> = freqs[frow + 2u * j + 1u];
    let xr: vec2<f16> = x[base + j];
    let xi: vec2<f16> = x[base + j + half_w];
    out[base + j] = vec2<f16>(
        xr.x * f0.x - xi.x * f0.y,
        xr.y * f1.x - xi.y * f1.y);
    out[base + j + half_w] = vec2<f16>(
        xr.x * f0.y + xi.x * f0.x,
        xr.y * f1.y + xi.y * f1.x);
}
"#
);

/// Half-rot rope over the leading `rot_pairs * 2` channels of each head, with
/// the remaining channels copied through. See the section comment above.
pub struct RopeF32HalfRotPartial;

impl RopeOp for RopeF32HalfRotPartial {
    const KERNEL_ID: &'static str = "rope.f32.halfrot.partial";
    type Dtype = F32;
    const X: &'static str = "rope_halfrot_partial/x";
    const FREQS: &'static str = "rope_halfrot_partial/freqs";
    const DIMS: &'static str = "rope_halfrot_partial/dims";
    const OUTPUT: &'static str = "rope_halfrot_partial/out";

    fn wgsl(cfg: &WgslConfig) -> &'static str {
        match (cfg.act_dtype, cfg.bf16_quant_writes) {
            (ActDtype::F32, false) => WGSL_F32_HALFROT_PARTIAL,
            (ActDtype::F32, true) => WGSL_F32_HALFROT_PARTIAL_BF16,
            (ActDtype::Bf16, _) => WGSL_BF16_PACKED_HALFROT_PARTIAL,
            (ActDtype::F16, _) => WGSL_F16_PACKED_HALFROT_PARTIAL,
            (ActDtype::I8, _) => unreachable!(
                "ActDtype::I8 partial halfrot is not implemented - MiniMax-H3 (the only consumer) runs f16/bf16 acts"
            ),
        }
    }

    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }

    /// The F32 variant runs one thread per ELEMENT (it must also copy the
    /// pass-through tail), so it needs `dim = pairs * 2` threads per head. The
    /// packed variants are one thread per word and over-dispatch 2x, returning
    /// early; that is cheaper than branching the launch on act dtype.
    fn workgroups(rows: u32, heads: u32, pairs: u32) -> [u32; 3] {
        super::linear_workgroups(rows * heads * pairs * 2, 64)
    }
}

// ---------------------------------------------------------------------------
// 3-axis MRoPE (Qwen2.5-VL LM): half-rot pairing, but each pair k draws its
// cos/sin for the real element (x[k]) and the imag element (x[k+pairs]) from
// DIFFERENT position axes (the [32,48,48] mrope_section split over the 128 dims
// after `cat(freqs, freqs)`). The caller bakes the per-pair, per-axis cos/sin
// into `freqs` as 4 values per pair `[cos_lo, sin_lo, cos_hi, sin_hi]`, so the
// freqs row stride is `pairs * 4 = dim * 2` (twice the half-rot layout).
//
// Per pair k (xr = x[k], xi = x[k+pairs]):
//   out[k]        = xr*cos_lo - xi*sin_lo
//   out[k+pairs]  = xi*cos_hi + xr*sin_hi
// When cos_lo==cos_hi && sin_lo==sin_hi (text-only, all axes equal) this is
// EXACTLY `RopeF32HalfRot`. f16 / i8 are unreachable (encoder is bf16/f32).
wgsl_with_bf16_variant!(
    WGSL_F32_MROPE,
    WGSL_F32_MROPE_BF16 = r#"
struct U { rows: u32, heads: u32, pairs: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> freqs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let total = u.rows * u.heads * u.pairs;
    let idx = gid.y * (ng.x * 64u) + gid.x;
    if (idx >= total) { return; }
    let pair = idx % u.pairs;
    let rh   = idx / u.pairs;
    let row  = rh / u.heads;
    let dim  = u.pairs * 2u;
    let x_re_off = rh  * dim + pair;
    let x_im_off = x_re_off + u.pairs;
    let f_off    = row * (u.pairs * 4u) + pair * 4u;
    let xr = x[x_re_off];
    let xi = x[x_im_off];
    let cos_lo = freqs[f_off];
    let sin_lo = freqs[f_off + 1u];
    let cos_hi = freqs[f_off + 2u];
    let sin_hi = freqs[f_off + 3u];
    out[x_re_off] = act_store(xr * cos_lo - xi * sin_lo);
    out[x_im_off] = act_store(xi * cos_hi + xr * sin_hi);
}
"#
);

// Packed-bf16 MRoPE. Same 2-pairs/thread word scheme as the packed half-rot:
// the real word holds x[2j], x[2j+1]; the imag word holds x[2j+pairs],
// x[2j+1+pairs]. Each of the two pairs reads its own 4 freqs. `pairs % 2 == 0`.
const WGSL_BF16_PACKED_MROPE: &str = concat!(
    act_bf16_prelude!(),
    r#"
struct U { rows: u32, heads: u32, pairs: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> freqs: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let pair_words = u.pairs >> 1u;
    let total = u.rows * u.heads * pair_words;
    let idx = gid.y * (ng.x * 64u) + gid.x;
    if (idx >= total) { return; }
    let j  = idx % pair_words;
    let rh = idx / pair_words;
    let row = rh / u.heads;
    let row_w_base = rh * u.pairs;
    let xr_w = row_w_base + j;
    let xi_w = row_w_base + pair_words + j;
    // 4 freqs per pair => 2 words per pair => 4 words cover pairs 2j, 2j+1.
    let frow_w_base = row * (u.pairs * 2u);
    let g0 = unpack_bf16x2(freqs[frow_w_base + 4u * j]);       // pair 2j:   cos_lo, sin_lo
    let g1 = unpack_bf16x2(freqs[frow_w_base + 4u * j + 1u]);  // pair 2j:   cos_hi, sin_hi
    let g2 = unpack_bf16x2(freqs[frow_w_base + 4u * j + 2u]);  // pair 2j+1: cos_lo, sin_lo
    let g3 = unpack_bf16x2(freqs[frow_w_base + 4u * j + 3u]);  // pair 2j+1: cos_hi, sin_hi
    let xr = unpack_bf16x2(x[xr_w]);
    let xi = unpack_bf16x2(x[xi_w]);
    let or0 = xr.x * g0.x - xi.x * g0.y;
    let or1 = xr.y * g2.x - xi.y * g2.y;
    let oi0 = xi.x * g1.x + xr.x * g1.y;
    let oi1 = xi.y * g3.x + xr.y * g3.y;
    out[xr_w] = pack_bf16x2(or0, or1);
    out[xi_w] = pack_bf16x2(oi0, oi1);
}
"#
);

pub struct RopeF32Mrope;

impl RopeOp for RopeF32Mrope {
    const KERNEL_ID: &'static str = "rope.f32.mrope";
    type Dtype = F32;
    const X: &'static str = "rope_mrope/x";
    const FREQS: &'static str = "rope_mrope/freqs";
    const DIMS: &'static str = "rope_mrope/dims";
    const OUTPUT: &'static str = "rope_mrope/out";
    fn wgsl(cfg: &WgslConfig) -> &'static str {
        match (cfg.act_dtype, cfg.bf16_quant_writes) {
            (ActDtype::F32, false) => WGSL_F32_MROPE,
            (ActDtype::F32, true) => WGSL_F32_MROPE_BF16,
            (ActDtype::Bf16, _) => WGSL_BF16_PACKED_MROPE,
            (ActDtype::F16, _) => {
                unreachable!("MRoPE (Qwen2.5-VL edit encoder) runs bf16/f32 acts, never f16")
            }
            (ActDtype::I8, _) => {
                unreachable!("MRoPE (Qwen2.5-VL edit encoder) runs bf16/f32 acts, never i8")
            }
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl OpTest for RopeF32 {
    fn dtypes(&self) -> &'static [Dtype] {
        DTYPES_ACT_BF16
    }
    fn test_cases(&self) -> Vec<TestCase> {
        vec![TestCase {
            name: "rope_basic",
            op: OpSpec::Rope,
            inputs: vec![
                t("x", [3, 2, 8], linspace(-1.0, 1.0, false)),
                t("freqs", [3, 8], linspace(-0.75, 0.75, false)),
            ],
        }]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_rope::<RopeF32>())
    }
}

#[cfg(feature = "conformance")]
impl OpTest for RopeF32HalfRot {
    fn dtypes(&self) -> &'static [Dtype] {
        DTYPES_ACT_BF16
    }
    fn test_cases(&self) -> Vec<TestCase> {
        vec![TestCase {
            name: "rope_halfrot_basic",
            op: OpSpec::RopeHalfRot,
            inputs: vec![
                t("x", [3, 2, 8], linspace(-1.0, 1.0, false)),
                t("freqs", [3, 8], linspace(-0.75, 0.75, false)),
            ],
        }]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_rope::<RopeF32HalfRot>())
    }
}

#[cfg(feature = "conformance")]
impl OpTest for RopeF32HalfRotPartial {
    fn dtypes(&self) -> &'static [Dtype] {
        DTYPES_ACT_BF16
    }
    fn test_cases(&self) -> Vec<TestCase> {
        vec![
            // head_dim 16, rotary 8: half the channels rotate, half pass through.
            TestCase {
                name: "rope_halfrot_partial_basic",
                op: OpSpec::RopeHalfRotPartial,
                inputs: vec![
                    t("x", [3, 2, 16], linspace(-1.0, 1.0, false)),
                    t("freqs", [3, 8], linspace(-0.75, 0.75, false)),
                ],
            },
            // The H3 ratio: 96 of 128 rotate. Catches word-alignment mistakes in
            // the packed kernels that the tiny case is too small to expose.
            TestCase {
                name: "rope_halfrot_partial_h3_ratio",
                op: OpSpec::RopeHalfRotPartial,
                inputs: vec![
                    t("x", [2, 2, 128], linspace(-1.0, 1.0, false)),
                    t("freqs", [2, 96], linspace(-0.75, 0.75, false)),
                ],
            },
            // Fully rotated: must agree with the plain half-rot kernel.
            TestCase {
                name: "rope_halfrot_partial_full_width",
                op: OpSpec::RopeHalfRotPartial,
                inputs: vec![
                    t("x", [3, 2, 8], linspace(-1.0, 1.0, false)),
                    t("freqs", [3, 8], linspace(-0.75, 0.75, false)),
                ],
            },
        ]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_rope_partial::<RopeF32HalfRotPartial>())
    }
}
