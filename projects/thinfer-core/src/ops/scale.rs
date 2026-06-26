use super::{ActDtype, WgslConfig};
use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
use crate::tensor::{ComputeDtype, F32};

/// Pointwise scalar multiply: `out[i] = x[i] * scale`. The scalar comes from the
/// uniform so a single compiled pipeline serves any factor. Used for BigVGAN's
/// `mean(dim=0)` over the `num_kernels` resblock outputs (sum via `AddF32`, then
/// scale by `1/num_kernels`).
///
/// Layout: 0=X, 1=Out, 2=Uniform `{n, scale_bits, _, _}`.
pub trait ScaleOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const DIMS: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> String;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n: u32) -> [u32; 3] {
        crate::ops::linear_workgroups(n, 64)
    }
}

pub struct ScaleBufs<'a> {
    pub x: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

/// Uniform `[n, scale_bits, 0, 0]` (16 bytes).
pub fn scale_uniform_bytes(n: u32, scale: f32) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&n.to_le_bytes());
    bytes[4..8].copy_from_slice(&scale.to_bits().to_le_bytes());
    bytes
}

pub(crate) fn dispatch_scale<O: ScaleOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &ScaleBufs<'_>,
    n: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.out.binding(1),
        bufs.uniform.binding(2),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n))
}

fn build_wgsl(cfg: &WgslConfig) -> String {
    assert!(!cfg.bf16_quant_writes, "scale has no bf16-quant-write mode");
    let (f16_prelude, x_decl, out_decl, load_x, store_expr) = match cfg.act_dtype {
        ActDtype::F32 => (
            "",
            "@group(0) @binding(0) var<storage, read> x: array<f32>;",
            "@group(0) @binding(1) var<storage, read_write> out: array<f32>;",
            "fn load_x(i: u32) -> f32 { return x[i]; }\n",
            "v",
        ),
        other => panic!("scale does not support act dtype {other:?}"),
    };

    format!(
        r#"{f16_prelude}
struct U {{ n: u32, scale_bits: u32, _p0: u32, _p1: u32 }};

{x_decl}
{out_decl}
@group(0) @binding(2) var<uniform> u: U;

{load_x}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {{
    let i: u32 = gid.y * (ng.x * 64u) + gid.x;
    if (i >= u.n) {{ return; }}
    let s: f32 = bitcast<f32>(u.scale_bits);
    let v: f32 = load_x(i) * s;
    out[i] = {store_expr};
}}
"#
    )
}

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

pub struct ScaleF32;

impl ScaleOp for ScaleF32 {
    const KERNEL_ID: &'static str = "scale.f32";
    type Dtype = F32;
    const X: &'static str = "scale/x";
    const DIMS: &'static str = "scale/dims";
    const OUTPUT: &'static str = "scale/out";
    fn wgsl(cfg: &WgslConfig) -> String {
        build_wgsl(cfg)
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl crate::conformance::OpTest for ScaleF32 {
    fn dtypes(&self) -> &'static [crate::conformance::Dtype] {
        crate::conformance::DTYPES_FP32_ONLY
    }
    fn test_cases(&self) -> Vec<crate::conformance::TestCase> {
        use crate::conformance::{OpSpec, TestCase, linspace, t};
        vec![TestCase {
            name: "scale_third",
            op: OpSpec::Scale { scale: 1.0 / 3.0 },
            inputs: vec![t("x", [24], linspace(-2.0, 2.0, false))],
        }]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a crate::conformance::OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_scale::<ScaleF32>())
    }
}
