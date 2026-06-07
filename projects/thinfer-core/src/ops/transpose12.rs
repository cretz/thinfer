use super::{ActDtype, WgslConfig};
use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
#[cfg(feature = "conformance")]
use crate::conformance::{
    DTYPES_FP32_ONLY, Dtype, OpSpec, OpTest, OpTestContext, TestCase, linspace, t,
};
use crate::tensor::{ComputeDtype, F32};

/// Swap axes 1 and 2 of a 4D tensor: `[d0, d1, d2, d3] -> [d0, d2, d1, d3]`.
/// Used to bridge `[B, S, H, D]` (post-rope) ↔ `[B, H, S, D]` (sdpa input/output).
///
/// Layout: 0=In, 1=Out, 2=Uniform `{d0, d1, d2, d3}` (u32x4).
pub trait Transpose12Op {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const IN: &'static str;
    const DIMS: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(total: u32) -> [u32; 3] {
        super::linear_workgroups(total, 64)
    }
}

pub struct Transpose12Bufs<'a> {
    pub input: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_transpose12<O: Transpose12Op, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &Transpose12Bufs<'_>,
    total: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.input.binding(0),
        bufs.out.binding(1),
        bufs.uniform.binding(2),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(total))
}

macro_rules! transpose12_body {
    () => {
        r#"
struct U { d0: u32, d1: u32, d2: u32, d3: u32 };

@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let total = u.d0 * u.d1 * u.d2 * u.d3;
    let idx = gid.y * (ng.x * 64u) + gid.x;
    if (idx >= total) { return; }
    // decode output index over [d0, d2, d1, d3]
    let d3 = idx % u.d3;
    let r0 = idx / u.d3;
    let b1 = r0 % u.d1;          // axis-1 of output (= old axis-2)
    let r1 = r0 / u.d1;
    let c2 = r1 % u.d2;          // axis-2 of output (= old axis-1)
    let a0 = r1 / u.d2;
    let in_idx = ((a0 * u.d1 + b1) * u.d2 + c2) * u.d3 + d3;
    out[idx] = x[in_idx];
}
"#
    };
}

const WGSL_F32: &str = concat!(
    r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
"#,
    transpose12_body!()
);

// Native-f16 acts: pure permutation, scalar f16 element access.
const WGSL_F16: &str = concat!(
    r#"enable f16;
@group(0) @binding(0) var<storage, read> x: array<f16>;
@group(0) @binding(1) var<storage, read_write> out: array<f16>;
"#,
    transpose12_body!()
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

pub struct Transpose12F32;

impl Transpose12Op for Transpose12F32 {
    const KERNEL_ID: &'static str = "transpose12.f32";
    type Dtype = F32;
    const IN: &'static str = "transpose12/in";
    const DIMS: &'static str = "transpose12/dims";
    const OUTPUT: &'static str = "transpose12/out";
    fn wgsl(cfg: &WgslConfig) -> &'static str {
        assert!(!cfg.bf16_quant_writes);
        match cfg.act_dtype {
            ActDtype::F16 => WGSL_F16,
            _ => WGSL_F32,
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl OpTest for Transpose12F32 {
    fn dtypes(&self) -> &'static [Dtype] {
        DTYPES_FP32_ONLY
    }
    fn test_cases(&self) -> Vec<TestCase> {
        vec![TestCase {
            name: "transpose12_basic",
            op: OpSpec::Transpose12,
            inputs: vec![t("x", [2, 3, 4, 5], linspace(-1.0, 1.0, false))],
        }]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_transpose12::<Transpose12F32>())
    }
}
