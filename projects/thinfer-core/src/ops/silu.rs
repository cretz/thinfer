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

const WGSL_BF16_PACKED: &str = concat!(
    act_bf16_prelude!(),
    r#"
@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let w = gid.y * (ng.x * 64u) + gid.x;
    if (w >= arrayLength(&out)) { return; }
    let xv = unpack_bf16x2(x[w]);
    let r0 = xv.x / (1.0 + exp(-xv.x));
    let r1 = xv.y / (1.0 + exp(-xv.y));
    out[w] = pack_bf16x2(r0, r1);
}
"#
);

// Native f16 path. Storage is `array<vec2<f16>>` — 2 f16 elements per word,
// same dispatch shape as bf16-packed. The COMPUTE upcasts to f32: silu's
// `exp(-x)` saturates f16 at |x| > ~11 (`exp(11) > 65504`), which the Hunyuan
// VAE's pre-norm conv activations (|x| up to ~140) hit -> inf -> downstream NaN.
// f32 compute is also strictly closer to the f32 reference; the f16 store clamps
// to the representable range. (The values themselves are well within f16 range;
// only the intermediate `exp` overflowed.)
const WGSL_F16_PACKED: &str = concat!(
    act_f16_prelude!(),
    r#"
@group(0) @binding(0) var<storage, read> x: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec2<f16>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let w = gid.y * (ng.x * 64u) + gid.x;
    if (w >= arrayLength(&out)) { return; }
    let xv: vec2<f32> = vec2<f32>(x[w]);
    let r: vec2<f32> = xv / (vec2<f32>(1.0, 1.0) + exp(-xv));
    out[w] = vec2<f16>(clamp(r, vec2<f32>(-65504.0, -65504.0), vec2<f32>(65504.0, 65504.0)));
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

#[cfg(feature = "conformance")]
impl OpTest for SiluF32 {
    fn dtypes(&self) -> &'static [Dtype] {
        DTYPES_ACT_BF16
    }
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
