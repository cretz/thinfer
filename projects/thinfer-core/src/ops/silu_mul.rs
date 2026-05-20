use super::{ActDtype, Op, WgslConfig};
use crate::backend::{BindingKind, BindingLayout};
#[cfg(feature = "conformance")]
use crate::conformance::{
    DTYPES_ACT_BF16, Dtype, OpSpec, OpTest, OpTestContext, TestCase, linspace, t,
};
use crate::tensor::F32;
use crate::{act_bf16_prelude, wgsl_with_bf16_variant};

// Fused SwiGLU half: `out[i] = silu(a[i]) * b[i]`. Replaces the silu+mul pair
// in FFN. Halves the elementwise pass over hidden-size buffers (one combined
// dispatch reads a and b once, writes out once; the unfused path reads a,
// writes h1s, then reads h1s and b, writes out).
//
// Three storage variants over `WgslConfig`:
// - `ActDtype::F32, !bf16_quant_writes`: f32 in/out, no rounding.
// - `ActDtype::F32,  bf16_quant_writes`: f32 layout, RNE-round writes to bf16.
// - `ActDtype::Bf16`: packed `array<u32>` storage (2 elem/word); read via
//   `unpack_bf16x2`, written via `pack_bf16x2`. Dispatch counts words: each
//   thread emits one word covering 2 consecutive elements.
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
    let v = a[i];
    let s = v / (1.0 + exp(-v));
    out[i] = act_store(s * b[i]);
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
    let s0 = av.x / (1.0 + exp(-av.x));
    let s1 = av.y / (1.0 + exp(-av.y));
    out[w] = pack_bf16x2(s0 * bv.x, s1 * bv.y);
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

pub struct SiluMulF32;

impl Op for SiluMulF32 {
    const KERNEL_ID: &'static str = "silu_mul.f32";
    type Dtype = F32;
    const INPUTS: &'static [&'static str] = &["silu_mul/a", "silu_mul/b"];
    const OUTPUT: &'static str = "silu_mul/out";
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
impl OpTest for SiluMulF32 {
    fn dtypes(&self) -> &'static [Dtype] {
        DTYPES_ACT_BF16
    }
    fn test_cases(&self) -> Vec<TestCase> {
        vec![TestCase {
            name: "silu_mul_basic",
            op: OpSpec::SiluMul,
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
        Box::pin(ctx.run_elementwise::<SiluMulF32>())
    }
}
