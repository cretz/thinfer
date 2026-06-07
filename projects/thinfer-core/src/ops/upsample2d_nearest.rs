use super::{ActDtype, WgslConfig};
use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
use crate::tensor::{ComputeDtype, F32};

/// 2x nearest-neighbor upsampling on NCHW: `out[b,c,h,w] = x[b,c,h/2,w/2]`.
///
/// Matches `torch.nn.functional.interpolate(scale_factor=2, mode="nearest")`.
/// The scale factor is fixed at 2 since that's all the VAE decoder uses.
///
/// Shapes:
/// - `x:   [B, C, H, W]`
/// - `out: [B, C, 2H, 2W]`
///
/// Layout: 0=X, 1=Out, 2=Uniform.
pub trait Upsample2dNearestOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const DIMS: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n_out_elems: u32) -> [u32; 3] {
        super::linear_workgroups(n_out_elems, 64)
    }
}

pub struct Upsample2dNearestBufs<'a> {
    pub x: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_upsample2d_nearest<O: Upsample2dNearestOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &Upsample2dNearestBufs<'_>,
    n_out_elems: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.out.binding(1),
        bufs.uniform.binding(2),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_out_elems))
}

macro_rules! upsample_body {
    () => {
        r#"
struct U { b: u32, c: u32, h_in: u32, w_in: u32 };

@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    let h_out = u.h_in * 2u;
    let w_out = u.w_in * 2u;
    let total = u.b * u.c * h_out * w_out;
    if (i >= total) { return; }

    let ho_wo = h_out * w_out;
    let c_stride = u.c * ho_wo;
    let bi = i / c_stride;
    let r1 = i - bi * c_stride;
    let ci = r1 / ho_wo;
    let r2 = r1 - ci * ho_wo;
    let ho = r2 / w_out;
    let wo = r2 - ho * w_out;

    let hi = ho / 2u;
    let wi = wo / 2u;
    let in_idx = bi * (u.c * u.h_in * u.w_in) + ci * (u.h_in * u.w_in) + hi * u.w_in + wi;
    out[i] = x[in_idx];
}
"#
    };
}

const WGSL_F32: &str = concat!(
    r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
"#,
    upsample_body!()
);

// Native-f16 acts: pure gather/copy, scalar f16 element access (the 2x
// duplication makes paired access awkward; bandwidth still halves).
const WGSL_F16: &str = concat!(
    r#"enable f16;
@group(0) @binding(0) var<storage, read> x: array<f16>;
@group(0) @binding(1) var<storage, read_write> out: array<f16>;
"#,
    upsample_body!()
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
    BindingLayout {
        slot: 2,
        kind: BindingKind::Uniform,
    },
];

pub struct Upsample2dNearestF32;

impl Upsample2dNearestOp for Upsample2dNearestF32 {
    const KERNEL_ID: &'static str = "upsample2d_nearest.f32";
    type Dtype = F32;
    const X: &'static str = "upsample2d_nearest/x";
    const DIMS: &'static str = "upsample2d_nearest/dims";
    const OUTPUT: &'static str = "upsample2d_nearest/out";
    fn wgsl(cfg: &WgslConfig) -> &'static str {
        assert!(!cfg.bf16_quant_writes);
        match cfg.act_dtype {
            ActDtype::F16 => WGSL_F16,
            _ => WGSL_F32,
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}
