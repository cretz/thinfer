use super::{ActDtype, WgslConfig};
use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
use crate::tensor::{ComputeDtype, F32};

/// `DupUp3D`: the parameter-free residual up-shortcut of the Wan2.2 (residual)
/// VAE decoder (`autoencoder_kl_wan.py::DupUp3D`). Duplicate-upsamples NCTHW by
/// `factor_t` in time and `factor_s` in each spatial axis while regrouping the
/// channel axis: a channel `repeat_interleave(repeats)` followed by a
/// `view/permute/view` that folds the duplicated channels into the t/h/w grid.
///
/// For output element `(oc, T, H, W)`:
/// - `t = T / ft, a = T % ft`
/// - `h = H / fs, b = H % fs`
/// - `w = W / fs, c = W % fs`
/// - `e = ((oc*ft + a)*fs + b)*fs + c`  (the expanded channel index)
/// - `ic = e / repeats`                  (repeat_interleave inverse)
/// - `out[oc, T, H, W] = x[ic, t, h, w]`
///
/// `t_drop` implements the `first_chunk` temporal trim (drop the leading
/// `ft - 1` output frames): output frame `T` reads from full frame `T + t_drop`,
/// and the output buffer holds `t_in*ft - t_drop` frames. B is fixed at 1 (the
/// VAE decodes one video).
///
/// Shapes:
/// - `x:   [1, in_c, t_in, h_in, w_in]`
/// - `out: [1, out_c, t_in*ft - t_drop, h_in*fs, w_in*fs]`
///
/// Layout: 0=X, 1=Out, 2=Uniform.
pub trait DupUp3dOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n_out_elems: u32) -> [u32; 3] {
        super::linear_workgroups(n_out_elems, 64)
    }
}

pub struct DupUp3dBufs<'a> {
    pub x: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_dupup3d<O: DupUp3dOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &DupUp3dBufs<'_>,
    n_out_elems: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.out.binding(1),
        bufs.uniform.binding(2),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_out_elems))
}

macro_rules! dupup3d_body {
    () => {
        r#"
struct U {
    in_c: u32, out_c: u32, ft: u32, fs: u32,
    t_in: u32, h_in: u32, w_in: u32, repeats: u32,
    t_drop: u32, pad0: u32, pad1: u32, pad2: u32,
};

@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    let t_out = u.t_in * u.ft - u.t_drop;
    let h_out = u.h_in * u.fs;
    let w_out = u.w_in * u.fs;
    let total = u.out_c * t_out * h_out * w_out;
    if (i >= total) { return; }

    let hw_out = h_out * w_out;
    let thw_out = t_out * hw_out;
    let oc = i / thw_out;
    let r1 = i - oc * thw_out;
    let to = r1 / hw_out;
    let r2 = r1 - to * hw_out;
    let ho = r2 / w_out;
    let wo = r2 - ho * w_out;

    let tf = to + u.t_drop; // full (pre-trim) output time index
    let t = tf / u.ft;
    let a = tf - t * u.ft;
    let h = ho / u.fs;
    let b = ho - h * u.fs;
    let w = wo / u.fs;
    let c = wo - w * u.fs;

    let e = ((oc * u.ft + a) * u.fs + b) * u.fs + c;
    let ic = e / u.repeats;
    let in_idx = ic * (u.t_in * u.h_in * u.w_in) + t * (u.h_in * u.w_in) + h * u.w_in + w;
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
    dupup3d_body!()
);

// Native-f16 acts: pure gather/copy, scalar f16 element access.
const WGSL_F16: &str = concat!(
    r#"enable f16;
@group(0) @binding(0) var<storage, read> x: array<f16>;
@group(0) @binding(1) var<storage, read_write> out: array<f16>;
"#,
    dupup3d_body!()
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

pub struct DupUp3dF32;

impl DupUp3dOp for DupUp3dF32 {
    const KERNEL_ID: &'static str = "dupup3d.f32";
    type Dtype = F32;
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
