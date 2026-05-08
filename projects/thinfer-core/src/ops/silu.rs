use super::{Op, WgslConfig};
use crate::backend::{BindingKind, BindingLayout};
#[cfg(feature = "conformance")]
use crate::conformance::{OpSpec, OpTest, OpTestContext, TestCase, linspace, t};
use crate::tensor::F32;
use crate::wgsl_with_bf16_variant;

wgsl_with_bf16_variant!(
    WGSL_F32,
    WGSL_F32_BF16 = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    if (i >= arrayLength(&out)) { return; }
    let v = x[i];
    out[i] = act_store(v / (1.0 + exp(-v)));
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
];

pub struct SiluF32;

impl Op for SiluF32 {
    const KERNEL_ID: &'static str = "silu.f32";
    type Dtype = F32;
    const INPUTS: &'static [&'static str] = &["silu/x"];
    const OUTPUT: &'static str = "silu/out";
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
impl OpTest for SiluF32 {
    fn test_cases(&self) -> Vec<TestCase> {
        vec![TestCase {
            name: "silu_basic",
            op: OpSpec::Silu,
            inputs: vec![t("x", [64], linspace(-4.0, 4.0, false))],
        }]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_elementwise::<SiluF32>())
    }
}
