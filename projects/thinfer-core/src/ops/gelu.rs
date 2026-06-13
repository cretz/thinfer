use super::{ActDtype, Op, WgslConfig};
use crate::backend::{BindingKind, BindingLayout};
use crate::tensor::F32;
use crate::{act_bf16_prelude, act_f16_prelude, act_store_bf16q, act_store_f32};

// Plain (non-gated) GELU: `out[i] = gelu_new(x[i])`. The Wan / SkyReels-V2 DiT
// feed-forward is `proj_out(gelu_new(proj_in(x)))` (diffusers `FeedForward`
// `activation_fn="gelu-approximate"`), i.e. the SAME `gelu_new` nonlinearity as
// the umT5 gated FFN but WITHOUT the `* up` gate. This is the single-input
// sibling of [`super::gelu_mul::GeluMulF32`].
//
// `gelu_new` is the tanh approximation (HF `NewGELUActivation`, torch
// `F.gelu(.., approximate="tanh")`):
//   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// The cube term amplifies error, so every variant promotes to f32 for the gate
// math and narrows only on the final store (the f16-IO / f32-accum pattern),
// matching `gelu_mul`.

// Shared gate, evaluated in f32. A local literal-producing macro so it composes
// inside `concat!` (which rejects const idents), mirroring `gelu_mul`.
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
// consts differ only in the `act_store` prelude.
macro_rules! f32_body {
    () => {
        r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    if (i >= arrayLength(&out)) { return; }
    out[i] = act_store(gelu_new(x[i]));
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
@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let w = gid.y * (ng.x * 64u) + gid.x;
    if (w >= arrayLength(&out)) { return; }
    let xv = unpack_bf16x2(x[w]);
    out[w] = pack_bf16x2(gelu_new(xv.x), gelu_new(xv.y));
}
"#
);

// Native f16 storage, f32 gate compute (promote inputs, narrow only on store).
const WGSL_F16_PACKED: &str = concat!(
    act_f16_prelude!(),
    gelu_fn!(),
    r#"
@group(0) @binding(0) var<storage, read> x: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec2<f16>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let w = gid.y * (ng.x * 64u) + gid.x;
    if (w >= arrayLength(&out)) { return; }
    let xv = unpack_f16x2_f32(x[w]);
    out[w] = pack_f32_to_f16x2(gelu_new(xv.x), gelu_new(xv.y));
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

pub struct GeluF32;

impl Op for GeluF32 {
    const KERNEL_ID: &'static str = "gelu.f32";
    type Dtype = F32;
    const INPUTS: &'static [&'static str] = &["gelu/x"];
    const OUTPUT: &'static str = "gelu/out";
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
