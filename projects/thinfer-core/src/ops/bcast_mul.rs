//! Channel-broadcast multiply by a stored weight: `out[i] = x[i] * s[i % C]`.
//!
//! The multiply twin of [`super::bcast_add`]: same binding layout, dispatch
//! (`dispatch_bcast_add`), and generic scope method (`Workspace::bcast_add`),
//! reused via the shared [`super::BcastAddOp`] contract. The distinction is the
//! WGSL body (`*` not `+`) and a separate compiled pipeline.
//!
//! Exists because affine norms apply a per-channel *weight* scale (e.g. the Wan
//! cross-attn `norm2` `LayerNorm(x) * w + b`). `bcast_affine` reads its scale as
//! an ACT (the modulation path), so feeding it a bf16 stored weight reinterprets
//! the bits; `bcast_mul` decodes the weight by `weight_dtype` exactly like
//! `bcast_add` decodes a bias.

use crate::backend::{BindingKind, BindingLayout};
#[cfg(feature = "conformance")]
use crate::conformance::{
    DTYPES_ACT_BF16, Dtype, OpSpec, OpTest, OpTestContext, TestCase, linspace, t,
};
use crate::ops::{ActDtype, BcastAddOp, WeightDtype, WgslConfig};
use crate::tensor::F32;
use crate::{act_bf16_prelude, act_f16_prelude};

macro_rules! bcast_mul_body {
    () => {
        r#"
struct U { c: u32, _pad0: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    if (i >= arrayLength(&out)) { return; }
    out[i] = act_store(x[i] * load_s(i % u.c));
}
"#
    };
}

macro_rules! bcast_mul_f32_bindings {
    () => {
        r#"
@group(0) @binding(1) var<storage, read> s: array<f32>;
fn load_s(i: u32) -> f32 { return s[i]; }
"#
    };
}

macro_rules! bcast_mul_bf16_bindings {
    () => {
        r#"
@group(0) @binding(1) var<storage, read> s: array<u32>;
fn load_s(i: u32) -> f32 {
    let pair = s[i >> 1u];
    let half = (pair >> ((i & 1u) * 16u)) & 0xFFFFu;
    return bitcast<f32>(half << 16u);
}
"#
    };
}

crate::weight_op_wgsl! {
    (WGSL_F32, WGSL_F32_BF16Q, WGSL_F32_WBF16, WGSL_F32_BF16Q_WBF16);
    body = bcast_mul_body!();
    f32_bindings = bcast_mul_f32_bindings!();
    bf16_bindings = bcast_mul_bf16_bindings!();
}

// Packed-bf16 activations. `x`/`out` are array<u32> (2 elems/word); `s` (scale
// weight) can be fp32 or bf16-packed independently.
const WGSL_BF16_PACKED_WF32: &str = concat!(
    act_bf16_prelude!(),
    r#"
struct U { c: u32, _pad0: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let w = gid.y * (ng.x * 64u) + gid.x;
    if (w >= arrayLength(&out)) { return; }
    let xv = unpack_bf16x2(x[w]);
    let c0 = (w * 2u) % u.c;
    out[w] = pack_bf16x2(xv.x * s[c0], xv.y * s[c0 + 1u]);
}
"#
);

const WGSL_BF16_PACKED_WBF16: &str = concat!(
    act_bf16_prelude!(),
    r#"
struct U { c: u32, _pad0: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> s: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let w = gid.y * (ng.x * 64u) + gid.x;
    if (w >= arrayLength(&out)) { return; }
    let xv = unpack_bf16x2(x[w]);
    let c0 = (w * 2u) % u.c;
    let sv = unpack_bf16x2(s[c0 >> 1u]);
    out[w] = pack_bf16x2(xv.x * sv.x, xv.y * sv.y);
}
"#
);

// Native f16 path, f32-weight scale.
const WGSL_F16_PACKED_WF32: &str = concat!(
    act_f16_prelude!(),
    r#"
struct U { c: u32, _pad0: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read> x: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<vec2<f16>>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let w = gid.y * (ng.x * 64u) + gid.x;
    if (w >= arrayLength(&out)) { return; }
    let c0 = (w * 2u) % u.c;
    let sv: vec2<f16> = vec2<f16>(f16(s[c0]), f16(s[c0 + 1u]));
    out[w] = x[w] * sv;
}
"#
);

// Native f16 path, bf16-weight scale: unpack to f32 pair, narrow to vec2<f16>.
const WGSL_F16_PACKED_WBF16: &str = concat!(
    act_f16_prelude!(),
    r#"
struct U { c: u32, _pad0: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read> x: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read> s: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<vec2<f16>>;
@group(0) @binding(3) var<uniform> u: U;

fn unpack_bf16x2_f32(w: u32) -> vec2<f32> {
    let lo = bitcast<f32>((w & 0xFFFFu) << 16u);
    let hi = bitcast<f32>(w & 0xFFFF0000u);
    return vec2<f32>(lo, hi);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let w = gid.y * (ng.x * 64u) + gid.x;
    if (w >= arrayLength(&out)) { return; }
    let c0 = (w * 2u) % u.c;
    let sv_f32 = unpack_bf16x2_f32(s[c0 >> 1u]);
    let sv: vec2<f16> = vec2<f16>(sv_f32);
    out[w] = x[w] * sv;
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
    BindingLayout {
        slot: 3,
        kind: BindingKind::Uniform,
    },
];

pub struct BcastMulF32;

impl BcastAddOp for BcastMulF32 {
    const KERNEL_ID: &'static str = "bcast_mul.f32";
    type Dtype = F32;
    const X: &'static str = "bcast_mul/x";
    const S: &'static str = "bcast_mul/s";
    const OUTPUT: &'static str = "bcast_mul/out";
    fn wgsl(cfg: &WgslConfig) -> &'static str {
        match (cfg.act_dtype, cfg.bf16_quant_writes, cfg.weight_dtype) {
            (ActDtype::F32, false, WeightDtype::F32) => WGSL_F32,
            (ActDtype::F32, true, WeightDtype::F32) => WGSL_F32_BF16Q,
            (ActDtype::F32, false, WeightDtype::Bf16) => WGSL_F32_WBF16,
            (ActDtype::F32, true, WeightDtype::Bf16) => WGSL_F32_BF16Q_WBF16,
            (ActDtype::Bf16, _, WeightDtype::F32) => WGSL_BF16_PACKED_WF32,
            (ActDtype::Bf16, _, WeightDtype::Bf16) => WGSL_BF16_PACKED_WBF16,
            (ActDtype::F16, _, WeightDtype::F32) => WGSL_F16_PACKED_WF32,
            (ActDtype::F16, _, WeightDtype::Bf16) => WGSL_F16_PACKED_WBF16,
            (ActDtype::I8, _, WeightDtype::F32) | (ActDtype::I8, _, WeightDtype::Bf16) => {
                unreachable!("ActDtype::I8 is never a block-level act dtype")
            }
            // Same scope as bcast_add: affine norm scales stay full-precision;
            // quant flows only through matmul.
            (_, _, WeightDtype::F16) => {
                unreachable!("bcast_mul does not consume f16 weights (workspace-only dtype)")
            }
            (_, _, WeightDtype::Quant(_)) => {
                unreachable!("bcast_mul does not consume quant weights")
            }
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl OpTest for BcastMulF32 {
    fn dtypes(&self) -> &'static [Dtype] {
        DTYPES_ACT_BF16
    }
    fn test_cases(&self) -> Vec<TestCase> {
        vec![TestCase {
            name: "bcast_mul",
            op: OpSpec::BcastMul,
            inputs: vec![
                t("x", [4, 16], linspace(-1.0, 1.0, false)),
                t("s", [16], linspace(-0.5, 0.5, false)),
            ],
        }]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_bcast_add::<BcastMulF32>())
    }
}
