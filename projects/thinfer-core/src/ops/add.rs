use super::{ActDtype, Op, WgslConfig};
use crate::backend::{BindingKind, BindingLayout};
#[cfg(feature = "conformance")]
use crate::conformance::{
    DTYPES_ACT_BF16, Dtype, OpSpec, OpTest, OpTestContext, TestCase, linspace, t,
};
use crate::tensor::F32;
use crate::{act_bf16_prelude, act_f16_prelude, wgsl_with_bf16_variant};

wgsl_with_bf16_variant!(
    WGSL_F32,
    WGSL_F32_BF16 = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    if (i >= arrayLength(&out)) { return; }
    out[i] = act_store(a[i] + b[i]);
}
"#
);

const WGSL_BF16_PACKED: &str = concat!(
    act_bf16_prelude!(),
    r#"
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let w = gid.y * (ng.x * 64u) + gid.x;
    if (w >= arrayLength(&out)) { return; }
    let av = unpack_bf16x2(a[w]);
    let bv = unpack_bf16x2(b[w]);
    out[w] = pack_bf16x2(av.x + bv.x, av.y + bv.y);
}
"#
);

const WGSL_F16_PACKED: &str = concat!(
    act_f16_prelude!(),
    r#"
@group(0) @binding(0) var<storage, read> a: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read> b: array<vec2<f16>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec2<f16>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let w = gid.y * (ng.x * 64u) + gid.x;
    if (w >= arrayLength(&out)) { return; }
    out[w] = a[w] + b[w];
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
];

pub struct AddF32;

impl Op for AddF32 {
    const KERNEL_ID: &'static str = "add.f32";
    type Dtype = F32;
    const INPUTS: &'static [&'static str] = &["add/a", "add/b"];
    const OUTPUT: &'static str = "add/out";
    fn wgsl(cfg: &WgslConfig) -> &'static str {
        match (cfg.act_dtype, cfg.bf16_quant_writes) {
            (ActDtype::F32, false) => WGSL_F32,
            (ActDtype::F32, true) => WGSL_F32_BF16,
            (ActDtype::Bf16, _) => WGSL_BF16_PACKED,
            (ActDtype::F16, _) => WGSL_F16_PACKED,
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl OpTest for AddF32 {
    fn dtypes(&self) -> &'static [Dtype] {
        DTYPES_ACT_BF16
    }
    fn test_cases(&self) -> Vec<TestCase> {
        vec![TestCase {
            name: "add_basic",
            op: OpSpec::Add,
            inputs: vec![
                t("a", [64], linspace(-2.0, 2.0, false)),
                t("b", [64], linspace(-1.5, 1.75, true)),
            ],
        }]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_elementwise::<AddF32>())
    }
}
