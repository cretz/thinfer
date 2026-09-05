use super::{ActDtype, WgslConfig};
use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
use crate::tensor::{ComputeDtype, F32};

/// Pointwise leaky ReLU: `out[i] = x[i] >= 0 ? x[i] : slope * x[i]`. The negative
/// slope comes from the uniform so one compiled pipeline serves any slope (the
/// plain-HiFiGAN vocoder uses 0.1 in the up-stages / resblocks and 0.01 for the
/// pre-`conv_post` activation). f32-only (the whole vocoder tail runs f32).
///
/// Layout: 0=X, 1=Out, 2=Uniform `{n, slope_bits, _, _}`.
pub trait LeakyReluOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> String;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n: u32) -> [u32; 3] {
        crate::ops::linear_workgroups(n, 64)
    }
}

pub struct LeakyReluBufs<'a> {
    pub x: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

/// Uniform `[n, slope_bits, 0, 0]` (16 bytes).
pub fn leaky_relu_uniform_bytes(n: u32, slope: f32) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&n.to_le_bytes());
    bytes[4..8].copy_from_slice(&slope.to_bits().to_le_bytes());
    bytes
}

pub(crate) fn dispatch_leaky_relu<O: LeakyReluOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &LeakyReluBufs<'_>,
    n: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.out.binding(1),
        bufs.uniform.binding(2),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n))
}

fn build_wgsl(cfg: &WgslConfig) -> String {
    assert!(
        !cfg.bf16_quant_writes,
        "leaky_relu has no bf16-quant-write mode"
    );
    match cfg.act_dtype {
        ActDtype::F32 => {}
        other => panic!("leaky_relu does not support act dtype {other:?}"),
    }
    r#"
struct U { n: u32, slope_bits: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i: u32 = gid.y * (ng.x * 64u) + gid.x;
    if (i >= u.n) { return; }
    let s: f32 = bitcast<f32>(u.slope_bits);
    let v: f32 = x[i];
    out[i] = select(s * v, v, v >= 0.0);
}
"#
    .to_string()
}

const LAYOUT: &[BindingLayout] = &[
    BindingLayout {
        slot: 0,
        kind: BindingKind::StorageRead,
    },
    BindingLayout {
        slot: 1,
        kind: BindingKind::StorageReadWrite,
    },
    BindingLayout {
        slot: 2,
        kind: BindingKind::Uniform,
    },
];

pub struct LeakyReluF32;

impl LeakyReluOp for LeakyReluF32 {
    const KERNEL_ID: &'static str = "leaky_relu.f32";
    type Dtype = F32;
    const X: &'static str = "leaky_relu/x";
    const OUTPUT: &'static str = "leaky_relu/out";
    fn wgsl(cfg: &WgslConfig) -> String {
        build_wgsl(cfg)
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}
