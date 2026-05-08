use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
#[cfg(feature = "conformance")]
use crate::conformance::{OpSpec, OpTest, OpTestContext, TestCase, linspace, t};
use crate::ops::WgslConfig;
use crate::tensor::{ComputeDtype, F32};
use crate::wgsl_with_bf16_variant;

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
        [(rows * heads * pairs).div_ceil(64), 1, 1]
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = u.rows * u.heads * u.pairs;
    let idx = gid.x;
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
        if cfg.bf16_quant_writes {
            WGSL_F32_BF16
        } else {
            WGSL_F32
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = u.rows * u.heads * u.pairs;
    let idx = gid.x;
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

pub struct RopeF32HalfRot;

impl RopeOp for RopeF32HalfRot {
    const KERNEL_ID: &'static str = "rope.f32.halfrot";
    type Dtype = F32;
    const X: &'static str = "rope_halfrot/x";
    const FREQS: &'static str = "rope_halfrot/freqs";
    const DIMS: &'static str = "rope_halfrot/dims";
    const OUTPUT: &'static str = "rope_halfrot/out";
    fn wgsl(cfg: &WgslConfig) -> &'static str {
        if cfg.bf16_quant_writes {
            WGSL_F32_HALFROT_BF16
        } else {
            WGSL_F32_HALFROT
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl OpTest for RopeF32 {
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
