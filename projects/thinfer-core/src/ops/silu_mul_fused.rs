//! SwiGLU over a FUSED `[gate; value]` projection: `out[r, c] = silu(x[r, c]) *
//! x[r, h + c]` for `x: [rows, 2h]`, `out: [rows, h]`.
//!
//! MiniMax-H3's `mlp.fc1` is a single `[hidden, 2 * ffn]` projection whose
//! output row is `[gate | value]`, not the separate `gate_proj` / `up_proj`
//! pair the Wan and Qwen3 families use. The two halves are a STRIDED per-row
//! slice, not contiguous slabs, so they cannot be handed to
//! [`SiluMulF32`](super::silu_mul::SiluMulF32) as two buffer views.
//!
//! Splitting first would cost an extra full-width buffer and pass over the
//! widest activation in the block (`rows x 28672`). Reading the fused buffer
//! directly is one dispatch and no intermediate.
//!
//! Sibling to the `Op` trait rather than an impl of it: `Op` is uniform-less and
//! this kernel needs `h` to recover the row stride.

use super::{ActDtype, WgslConfig};
use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
#[cfg(feature = "conformance")]
use crate::conformance::{
    DTYPES_ACT_BF16, Dtype, OpSpec, OpTest, OpTestContext, TestCase, linspace, t,
};
use crate::tensor::{ComputeDtype, F32};
use crate::{act_bf16_prelude, act_f16_prelude, wgsl_with_bf16_variant};

pub trait SiluMulFusedOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const INPUT: &'static str;
    const OUTPUT: &'static str;
    const DIMS: &'static str;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n_elems: u32) -> [u32; 3] {
        super::linear_workgroups(n_elems, 64)
    }
}

pub struct SiluMulFusedBufs<'a> {
    pub input: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_silu_mul_fused<O: SiluMulFusedOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &SiluMulFusedBufs<'_>,
    n_elems: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.input.binding(0),
        bufs.out.binding(1),
        bufs.uniform.binding(2),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_elems))
}

wgsl_with_bf16_variant!(
    WGSL_F32,
    WGSL_F32_BF16 = r#"
struct U { rows: u32, h: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    let total = u.rows * u.h;
    if (i >= total) { return; }
    let row = i / u.h;
    let c = i - row * u.h;
    let base = row * (2u * u.h) + c;
    let g = x[base];
    let s = g / (1.0 + exp(-g));
    out[i] = act_store(s * x[base + u.h]);
}
"#
);

// Packed paths: one thread per WORD (2 elements). `h` must be even so a word
// never straddles the gate/value boundary; H3's ffn is 14336.
const WGSL_BF16_PACKED: &str = concat!(
    act_bf16_prelude!(),
    r#"
struct U { rows: u32, h: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    let hw = u.h >> 1u;
    let total = u.rows * hw;
    if (i >= total) { return; }
    let row = i / hw;
    let cw = i - row * hw;
    let base = row * (2u * hw) + cw;
    let g = unpack_bf16x2(x[base]);
    let v = unpack_bf16x2(x[base + hw]);
    let s0 = g.x / (1.0 + exp(-g.x));
    let s1 = g.y / (1.0 + exp(-g.y));
    out[i] = pack_bf16x2(s0 * v.x, s1 * v.y);
}
"#
);

const WGSL_F16_PACKED: &str = concat!(
    act_f16_prelude!(),
    r#"
struct U { rows: u32, h: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> x: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec2<f16>>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    let hw = u.h >> 1u;
    let total = u.rows * hw;
    if (i >= total) { return; }
    let row = i / hw;
    let cw = i - row * hw;
    let base = row * (2u * hw) + cw;
    let g: vec2<f16> = x[base];
    let v: vec2<f16> = x[base + hw];
    // silu in f32: exp() on f16 saturates early and the gate spans a wide range.
    let gx = f32(g.x);
    let gy = f32(g.y);
    let s0 = gx / (1.0 + exp(-gx));
    let s1 = gy / (1.0 + exp(-gy));
    out[i] = vec2<f16>(f16(s0) * v.x, f16(s1) * v.y);
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
        kind: BindingKind::Uniform,
    },
];

pub struct SiluMulFusedF32;

impl SiluMulFusedOp for SiluMulFusedF32 {
    const KERNEL_ID: &'static str = "silu_mul_fused.f32";
    type Dtype = F32;
    const INPUT: &'static str = "silu_mul_fused/x";
    const OUTPUT: &'static str = "silu_mul_fused/out";
    const DIMS: &'static str = "silu_mul_fused/dims";
    fn wgsl(cfg: &WgslConfig) -> &'static str {
        match (cfg.act_dtype, cfg.bf16_quant_writes) {
            (ActDtype::F32, false) => WGSL_F32,
            (ActDtype::F32, true) => WGSL_F32_BF16,
            (ActDtype::Bf16, _) => WGSL_BF16_PACKED,
            (ActDtype::F16, _) => WGSL_F16_PACKED,
            (ActDtype::I8, _) => {
                unreachable!("ActDtype::I8 is never a block-level act dtype")
            }
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl OpTest for SiluMulFusedF32 {
    fn dtypes(&self) -> &'static [Dtype] {
        DTYPES_ACT_BF16
    }
    fn test_cases(&self) -> Vec<TestCase> {
        vec![
            // The halves are a STRIDED per-row slice of ONE buffer, not two
            // buffer views: gate is x[r][..h] and value is x[r][h..]. A kernel
            // that treats the input as [all gates; all values] reads the wrong
            // partner for every row past the first.
            TestCase {
                name: "silu_mul_fused_basic",
                op: OpSpec::SiluMulFused,
                inputs: vec![t("x", [4, 16], linspace(-3.0, 3.0, false))],
            },
            // Wide row (H3 fc1 is 2 x 14336) and an odd row count, so the
            // strided partner lookup is exercised well past one workgroup.
            TestCase {
                name: "silu_mul_fused_wide",
                op: OpSpec::SiluMulFused,
                inputs: vec![t("x", [3, 512], linspace(-6.0, 6.0, false))],
            },
        ]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_silu_mul_fused::<SiluMulFusedF32>())
    }
}
