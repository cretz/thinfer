//! Channel-broadcast fused multiply-add: `out[i] = x[i] + s[i % C] * y[i]`.
//!
//! Z-Image modulation=True residual: `x = x + gate * norm(...)`. Single
//! dispatch and no extra activation buffer vs `mul` + `add`. v1 single-batch
//! only (see `bcast_affine` for the multi-batch caveat).

use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
#[cfg(feature = "conformance")]
use crate::conformance::{
    DTYPES_ACT_BF16, Dtype, OpSpec, OpTest, OpTestContext, TestCase, linspace, t,
};
use crate::ops::{ActDtype, WgslConfig};
use crate::tensor::{ComputeDtype, F32};
use crate::{act_bf16_prelude, wgsl_with_bf16_variant};

pub trait BcastFmaOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const S: &'static str;
    const Y: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n_elems: u32) -> [u32; 3] {
        super::linear_workgroups(n_elems, 64)
    }
}

pub struct BcastFmaBufs<'a> {
    pub x: &'a BufRef,
    pub s: &'a BufRef,
    pub y: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_bcast_fma<O: BcastFmaOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &BcastFmaBufs<'_>,
    n_elems: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.s.binding(1),
        bufs.y.binding(2),
        bufs.out.binding(3),
        bufs.uniform.binding(4),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_elems))
}

wgsl_with_bf16_variant!(
    WGSL_F32,
    WGSL_F32_BF16 = r#"
struct U { c: u32, _pad0: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    if (i >= arrayLength(&out)) { return; }
    out[i] = act_store(x[i] + s[i % u.c] * y[i]);
}
"#
);

const WGSL_BF16_PACKED: &str = concat!(
    act_bf16_prelude!(),
    r#"
struct U { c: u32, _pad0: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> s: array<u32>;
@group(0) @binding(2) var<storage, read> y: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let w = gid.y * (ng.x * 64u) + gid.x;
    if (w >= arrayLength(&out)) { return; }
    let xv = unpack_bf16x2(x[w]);
    let yv = unpack_bf16x2(y[w]);
    let c0 = (w * 2u) % u.c;
    let sv = unpack_bf16x2(s[c0 >> 1u]);
    out[w] = pack_bf16x2(xv.x + sv.x * yv.x, xv.y + sv.y * yv.y);
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
        kind: BindingKind::StorageRead,
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

pub struct BcastFmaF32;

impl BcastFmaOp for BcastFmaF32 {
    const KERNEL_ID: &'static str = "bcast_fma.f32";
    type Dtype = F32;
    const X: &'static str = "bcast_fma/x";
    const S: &'static str = "bcast_fma/s";
    const Y: &'static str = "bcast_fma/y";
    const OUTPUT: &'static str = "bcast_fma/out";
    fn wgsl(cfg: &WgslConfig) -> &'static str {
        match (cfg.act_dtype, cfg.bf16_quant_writes) {
            (ActDtype::F32, false) => WGSL_F32,
            (ActDtype::F32, true) => WGSL_F32_BF16,
            (ActDtype::Bf16, _) => WGSL_BF16_PACKED,
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl OpTest for BcastFmaF32 {
    fn dtypes(&self) -> &'static [Dtype] {
        DTYPES_ACT_BF16
    }
    fn test_cases(&self) -> Vec<TestCase> {
        vec![TestCase {
            name: "bcast_fma_basic",
            op: OpSpec::BcastFma,
            inputs: vec![
                t("x", [4, 16], linspace(-1.0, 1.0, false)),
                t("s", [16], linspace(-0.5, 0.5, false)),
                t("y", [4, 16], linspace(-2.0, 2.0, true)),
            ],
        }]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_bcast_fma::<BcastFmaF32>())
    }
}
