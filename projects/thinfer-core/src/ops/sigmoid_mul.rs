use super::{ActDtype, Op, WgslConfig};
use crate::backend::{BindingKind, BindingLayout};
use crate::tensor::F32;
use crate::{act_bf16_prelude, act_f16_prelude, wgsl_with_bf16_variant};

// Fused output gate: `out[i] = a[i] * sigmoid(b[i])`, `sigmoid(x) = 1/(1+e^-x)`.
// Krea 2's attention applies a learned full-width sigmoid gate to the attention
// output (`out = attn_out * sigmoid(gate(x))`, `krea2.hpp::KreaAttention`); `a`
// is the attention output and `b` the gate projection, both `[rows, features]`.
// One combined dispatch reads a and b once and writes out once (vs a standalone
// sigmoid pass + a mul). Distinct from `gated_head_mul` (per-HEAD gate + 2x) and
// `silu_mul` (which gates on `a` itself, not a second tensor).
//
// Three storage variants over `WgslConfig`, mirroring `silu_mul`:
// - `ActDtype::F32, !bf16_quant_writes`: f32 in/out.
// - `ActDtype::F32,  bf16_quant_writes`: f32 layout, RNE-round writes to bf16.
// - `ActDtype::Bf16`: packed `array<u32>` (2 elem/word). The Krea DiT runs bf16
//   acts (the residual exceeds f16 range); the gated output magnitude <= |a|.
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
    out[i] = act_store(a[i] / (1.0 + exp(-b[i])));
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
    let o0 = av.x / (1.0 + exp(-bv.x));
    let o1 = av.y / (1.0 + exp(-bv.y));
    out[w] = pack_bf16x2(o0, o1);
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
    let av: vec2<f16> = a[w];
    let bv: vec2<f16> = b[w];
    let one: vec2<f16> = vec2<f16>(f16(1.0), f16(1.0));
    out[w] = av / (one + exp(-bv));
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

pub struct SigmoidMulF32;

impl Op for SigmoidMulF32 {
    const KERNEL_ID: &'static str = "sigmoid_mul.f32";
    type Dtype = F32;
    const INPUTS: &'static [&'static str] = &["sigmoid_mul/a", "sigmoid_mul/b"];
    const OUTPUT: &'static str = "sigmoid_mul/out";
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
