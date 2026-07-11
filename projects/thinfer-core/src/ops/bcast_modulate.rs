//! Channel-broadcast affine modulation: `out[i] = x[i] * (s[i % C] + bias) + t[i % C]`.
//!
//! AdaLN modulation `norm(x) * (1 + scale) + shift` in one dispatch, with the
//! `1 +` folded into `bias`. Unlike [`bcast_add`](super::bcast_add), BOTH
//! broadcast vectors (`s` = scale, `t` = shift) are read as ACTIVATIONS in the
//! pipeline's act dtype, not weights: the modulation signals are computed at
//! runtime (`scale_shift_table + timestep_proj`), so they live in act buffers,
//! not the weight catalog. (Z-Image AdaLN is scale+gate only and never needed a
//! broadcast shift add; the Wan video family is the first to.)

use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
#[cfg(feature = "conformance")]
use crate::conformance::{
    DTYPES_ACT_BF16, Dtype, OpSpec, OpTest, OpTestContext, TestCase, linspace, t,
};
use crate::ops::{ActDtype, WgslConfig};
use crate::tensor::{ComputeDtype, F32};
use crate::{act_bf16_prelude, act_f16_prelude, wgsl_with_bf16_variant};

pub trait BcastModulateOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const S: &'static str;
    const T: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n_elems: u32) -> [u32; 3] {
        super::linear_workgroups(n_elems, 64)
    }
}

pub struct BcastModulateBufs<'a> {
    pub x: &'a BufRef,
    pub s: &'a BufRef,
    pub t: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_bcast_modulate<O: BcastModulateOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &BcastModulateBufs<'_>,
    n_elems: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.s.binding(1),
        bufs.t.binding(2),
        bufs.out.binding(3),
        bufs.uniform.binding(4),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_elems))
}

wgsl_with_bf16_variant!(
    WGSL_F32,
    WGSL_F32_BF16 = r#"
struct U { c: u32, bias: f32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read> t: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    if (i >= arrayLength(&out)) { return; }
    let c = i % u.c;
    out[i] = act_store(x[i] * (s[c] + u.bias) + t[c]);
}
"#
);

const WGSL_BF16_PACKED: &str = concat!(
    act_bf16_prelude!(),
    r#"
struct U { c: u32, bias: f32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> s: array<u32>;
@group(0) @binding(2) var<storage, read> t: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let w = gid.y * (ng.x * 64u) + gid.x;
    if (w >= arrayLength(&out)) { return; }
    let xv = unpack_bf16x2(x[w]);
    let c0 = (w * 2u) % u.c;
    let sv = unpack_bf16x2(s[c0 >> 1u]);
    let tv = unpack_bf16x2(t[c0 >> 1u]);
    out[w] = pack_bf16x2(xv.x * (sv.x + u.bias) + tv.x, xv.y * (sv.y + u.bias) + tv.y);
}
"#
);

// Native f16 path. `s` (scale) and `t` (shift) are f16-packed alongside x/out;
// in the F16 act pipeline every activation buffer is f16. `bias` is an f32
// uniform narrowed to f16 once per thread. C even per the bf16-packed invariant.
const WGSL_F16_PACKED: &str = concat!(
    act_f16_prelude!(),
    r#"
struct U { c: u32, bias: f32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> x: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read> s: array<vec2<f16>>;
@group(0) @binding(2) var<storage, read> t: array<vec2<f16>>;
@group(0) @binding(3) var<storage, read_write> out: array<vec2<f16>>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let w = gid.y * (ng.x * 64u) + gid.x;
    if (w >= arrayLength(&out)) { return; }
    let c0 = (w * 2u) % u.c;
    let sv: vec2<f16> = s[c0 >> 1u];
    let tv: vec2<f16> = t[c0 >> 1u];
    let bias_h: f16 = f16(u.bias);
    let bias_v: vec2<f16> = vec2<f16>(bias_h, bias_h);
    out[w] = x[w] * (sv + bias_v) + tv;
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
        kind: BindingKind::StorageRead,
    },
    BindingLayout {
        slot: 3,
        kind: BindingKind::StorageReadWrite,
    },
    BindingLayout {
        slot: 4,
        kind: BindingKind::Uniform,
    },
];

pub struct BcastModulateF32;

impl BcastModulateOp for BcastModulateF32 {
    const KERNEL_ID: &'static str = "bcast_modulate.f32";
    type Dtype = F32;
    const X: &'static str = "bcast_modulate/x";
    const S: &'static str = "bcast_modulate/s";
    const T: &'static str = "bcast_modulate/t";
    const OUTPUT: &'static str = "bcast_modulate/out";
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
impl OpTest for BcastModulateF32 {
    fn dtypes(&self) -> &'static [Dtype] {
        DTYPES_ACT_BF16
    }
    fn test_cases(&self) -> Vec<TestCase> {
        vec![TestCase {
            name: "bcast_modulate",
            op: OpSpec::BcastModulate { bias: 1.0 },
            inputs: vec![
                t("x", [4, 16], linspace(-1.0, 1.0, false)),
                t("s", [16], linspace(-0.5, 0.5, false)),
                t("t", [16], linspace(-0.25, 0.25, true)),
            ],
        }]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_bcast_modulate::<BcastModulateF32>())
    }
}
