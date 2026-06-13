use super::{ActDtype, Op, WgslConfig};
use crate::backend::{BindingKind, BindingLayout};
#[cfg(feature = "conformance")]
use crate::conformance::{
    DTYPES_ACT_BF16, Dtype, OpSpec, OpTest, OpTestContext, TestCase, linspace, t,
};
use crate::tensor::F32;
use crate::{act_bf16_prelude, act_f16_prelude, act_store_bf16q, act_store_f32};

// Fused gated-GELU half: `out[i] = gelu_new(a[i]) * b[i]`. The umT5 / T5
// `DenseReluDense` feed-forward is `wo(gelu_new(wi_0(x)) * wi_1(x))`; this is
// the gate*up half, mirroring `SiluMulF32` (the SwiGLU half) one-for-one. Only
// the gating nonlinearity differs.
//
// `gelu_new` is the tanh approximation (HF `NewGELUActivation`, torch
// `F.gelu(.., approximate="tanh")`):
//   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// The cube term amplifies error, so every variant promotes to f32 for the gate
// math and narrows only on the final store (the f16-IO / f32-accum pattern).
// `SiluMulF32` can stay in f16 because its sigmoid has no such amplification;
// gelu cannot.
//
// Storage variants over `WgslConfig` (same set as `silu_mul`):
// - `ActDtype::F32, !bf16_quant_writes`: f32 in/out, no rounding.
// - `ActDtype::F32,  bf16_quant_writes`: f32 layout, RNE-round writes to bf16.
// - `ActDtype::Bf16`: packed `array<u32>` storage (2 elem/word).
// - `ActDtype::F16`:  native `vec2<f16>` storage, f32 gate compute.

// Shared gate: gelu_new via the tanh approximation, evaluated in f32. A local
// literal-producing macro so it composes inside `concat!` (which rejects const
// idents), mirroring how `act_store_f32!` is threaded.
macro_rules! gelu_fn {
    () => {
        r#"
fn gelu_new(x: f32) -> f32 {
    let c = 0.7978845608028654; // sqrt(2/pi)
    let inner = c * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}
"#
    };
}

// f32 / bf16-writes share one body (both store through `act_store`); the two
// consts differ only in the `act_store` prelude. Built directly rather than
// via `wgsl_with_bf16_variant!` so the gate fn lands between prelude and body.
macro_rules! f32_body {
    () => {
        r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    if (i >= arrayLength(&out)) { return; }
    out[i] = act_store(gelu_new(a[i]) * b[i]);
}
"#
    };
}

const WGSL_F32: &str = concat!(act_store_f32!(), gelu_fn!(), f32_body!());
const WGSL_F32_BF16: &str = concat!(act_store_bf16q!(), gelu_fn!(), f32_body!());

const WGSL_BF16_PACKED: &str = concat!(
    act_bf16_prelude!(),
    gelu_fn!(),
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
    out[w] = pack_bf16x2(gelu_new(av.x) * bv.x, gelu_new(av.y) * bv.y);
}
"#
);

// Native f16 storage, f32 gate compute (promote inputs, narrow only on store).
const WGSL_F16_PACKED: &str = concat!(
    act_f16_prelude!(),
    gelu_fn!(),
    r#"
@group(0) @binding(0) var<storage, read> a: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read> b: array<vec2<f16>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec2<f16>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let w = gid.y * (ng.x * 64u) + gid.x;
    if (w >= arrayLength(&out)) { return; }
    let av = unpack_f16x2_f32(a[w]);
    let bv = unpack_f16x2_f32(b[w]);
    out[w] = pack_f32_to_f16x2(gelu_new(av.x) * bv.x, gelu_new(av.y) * bv.y);
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

pub struct GeluMulF32;

impl Op for GeluMulF32 {
    const KERNEL_ID: &'static str = "gelu_mul.f32";
    type Dtype = F32;
    const INPUTS: &'static [&'static str] = &["gelu_mul/a", "gelu_mul/b"];
    const OUTPUT: &'static str = "gelu_mul/out";
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
impl OpTest for GeluMulF32 {
    fn dtypes(&self) -> &'static [Dtype] {
        DTYPES_ACT_BF16
    }
    fn test_cases(&self) -> Vec<TestCase> {
        vec![TestCase {
            name: "gelu_mul_basic",
            op: OpSpec::GeluMul,
            inputs: vec![
                t("a", [64], linspace(-4.0, 4.0, false)),
                t("b", [64], linspace(-1.5, 1.75, true)),
            ],
        }]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_elementwise::<GeluMulF32>())
    }
}
