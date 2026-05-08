//! Channel-broadcast affine scale: `out[i] = x[i] * (s[i % C] + bias)`.
//!
//! Used in Z-Image modulation=True blocks: `norm(x) * (1 + scale)` (bias=1)
//! and gate/scale multiplies (bias=0). The broadcast unit is a single
//! per-channel vector `[C]` repeated across all rows of `x [N, C]`.
//!
//! v1 single-batch only (B=1). With B>1, scale/gate are `[B, 1, C]` per
//! upstream and require batch-aware indexing (`(i / (S*C)) * C + i%C`);
//! extend the uniform when batched generation lands.

use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
#[cfg(feature = "conformance")]
use crate::conformance::{OpSpec, OpTest, OpTestContext, TestCase, linspace, t};
use crate::ops::WgslConfig;
use crate::tensor::{ComputeDtype, F32};
use crate::wgsl_with_bf16_variant;

pub trait BcastAffineOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const S: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n_elems: u32) -> [u32; 3] {
        super::linear_workgroups(n_elems, 64)
    }
}

pub struct BcastAffineBufs<'a> {
    pub x: &'a BufRef,
    pub s: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_bcast_affine<O: BcastAffineOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &BcastAffineBufs<'_>,
    n_elems: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.s.binding(1),
        bufs.out.binding(2),
        bufs.uniform.binding(3),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_elems))
}

wgsl_with_bf16_variant!(
    WGSL_F32,
    WGSL_F32_BF16 = r#"
struct U { c: u32, bias: f32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    if (i >= arrayLength(&out)) { return; }
    out[i] = act_store(x[i] * (s[i % u.c] + u.bias));
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

pub struct BcastAffineF32;

impl BcastAffineOp for BcastAffineF32 {
    const KERNEL_ID: &'static str = "bcast_affine.f32";
    type Dtype = F32;
    const X: &'static str = "bcast_affine/x";
    const S: &'static str = "bcast_affine/s";
    const OUTPUT: &'static str = "bcast_affine/out";
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

#[cfg(feature = "conformance")]
impl OpTest for BcastAffineF32 {
    fn test_cases(&self) -> Vec<TestCase> {
        vec![
            TestCase {
                name: "bcast_affine_scale",
                op: OpSpec::BcastAffine { bias: 0.0 },
                inputs: vec![
                    t("x", [4, 16], linspace(-1.0, 1.0, false)),
                    t("s", [16], linspace(-0.5, 0.5, false)),
                ],
            },
            TestCase {
                name: "bcast_affine_one_plus_scale",
                op: OpSpec::BcastAffine { bias: 1.0 },
                inputs: vec![
                    t("x", [4, 16], linspace(-1.0, 1.0, false)),
                    t("s", [16], linspace(-0.5, 0.5, false)),
                ],
            },
        ]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_bcast_affine::<BcastAffineF32>())
    }
}
