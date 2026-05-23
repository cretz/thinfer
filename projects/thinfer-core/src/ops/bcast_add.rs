//! Channel-broadcast additive bias: `out[i] = x[i] + s[i % C]`.
//!
//! Used after Linear layers whose bias `[C]` broadcasts over rows of
//! `x [N, C]`. Plain `add` only works when `len(x) == len(s)` (single-row
//! escape hatch); this op covers `N > 1`.

use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
#[cfg(feature = "conformance")]
use crate::conformance::{
    DTYPES_ACT_BF16, Dtype, OpSpec, OpTest, OpTestContext, TestCase, linspace, t,
};
use crate::ops::{ActDtype, WeightDtype, WgslConfig};
use crate::tensor::{ComputeDtype, F32};
use crate::{act_bf16_prelude, act_f16_prelude};

pub trait BcastAddOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const S: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n_elems: u32) -> [u32; 3] {
        super::linear_workgroups(n_elems, 64)
    }
}

pub struct BcastAddBufs<'a> {
    pub x: &'a BufRef,
    pub s: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_bcast_add<O: BcastAddOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &BcastAddBufs<'_>,
    n_elems: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.s.binding(1),
        bufs.out.binding(2),
        bufs.uniform.binding(3),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_elems))
}

macro_rules! bcast_add_body {
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
    out[i] = act_store(x[i] + load_s(i % u.c));
}
"#
    };
}

macro_rules! bcast_add_f32_bindings {
    () => {
        r#"
@group(0) @binding(1) var<storage, read> s: array<f32>;
fn load_s(i: u32) -> f32 { return s[i]; }
"#
    };
}

macro_rules! bcast_add_bf16_bindings {
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
    body = bcast_add_body!();
    f32_bindings = bcast_add_f32_bindings!();
    bf16_bindings = bcast_add_bf16_bindings!();
}

// Packed-bf16 activations. `x` and `out` are array<u32> (2 elems/word); `s`
// (bias weight) can be fp32 or bf16-packed independently. Each thread emits
// one output word covering 2 consecutive channels.
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
    out[w] = pack_bf16x2(xv.x + s[c0], xv.y + s[c0 + 1u]);
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
    out[w] = pack_bf16x2(xv.x + sv.x, xv.y + sv.y);
}
"#
);

// Native f16 path, f32-weight bias. `s` is fp32 bias storage; narrow the
// two f32 reads to f16 and do the add in vec2<f16>.
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
    out[w] = x[w] + sv;
}
"#
);

// Native f16 path, bf16-weight bias. `s` is bf16-packed (the DiT AdaLN
// path); unpack to f32 pair, narrow to vec2<f16>, do the add native.
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
    out[w] = x[w] + sv;
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

pub struct BcastAddF32;

impl BcastAddOp for BcastAddF32 {
    const KERNEL_ID: &'static str = "bcast_add.f32";
    type Dtype = F32;
    const X: &'static str = "bcast_add/x";
    const S: &'static str = "bcast_add/s";
    const OUTPUT: &'static str = "bcast_add/out";
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
            // bcast_add operates on bias/affine vectors; norms+biases stay
            // full-precision under any quant scheme. Quant weights flow
            // only through matmul.
            (_, _, WeightDtype::F16) => {
                unreachable!("bcast_add does not consume f16 weights (workspace-only dtype)")
            }
            (_, _, WeightDtype::Quant(_)) => {
                unreachable!("bcast_add does not consume quant weights")
            }
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl OpTest for BcastAddF32 {
    fn dtypes(&self) -> &'static [Dtype] {
        DTYPES_ACT_BF16
    }
    fn test_cases(&self) -> Vec<TestCase> {
        vec![TestCase {
            name: "bcast_add",
            op: OpSpec::BcastAdd,
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
        Box::pin(ctx.run_bcast_add::<BcastAddF32>())
    }
}
