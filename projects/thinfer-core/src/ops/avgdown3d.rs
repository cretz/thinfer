use super::{ActDtype, WgslConfig};
use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
use crate::tensor::{ComputeDtype, F32};

/// `AvgDown3D`: the parameter-free residual down-shortcut of the Wan2.2-style
/// (residual) VAE encoder (`autoencoder_kl_wan.py::AvgDown3D`). Front-pads time
/// with zero frames to a multiple of `factor_t`, folds the `ft x fs x fs`
/// neighbourhood into the channel axis, then averages groups of `group`
/// consecutive folded channels into each output channel.
///
/// For output element `(oc, t, h, w)`, with `g = in_c * ft * fs * fs / out_c`:
/// - folded channel `e = oc * g + j` for `j in 0..g`
/// - `c = e / (ft*fs*fs)`, `a = (e / (fs*fs)) % ft`, `b = (e / fs) % fs`,
///   `d = e % fs`
/// - padded frame `tf = t*ft + a`, real frame `tf - t_pad` (zero when < 0)
/// - `out[oc, t, h, w] = mean_j x[c, tf - t_pad, h*fs + b, w*fs + d]`
///
/// B is fixed at 1.
///
/// Shapes:
/// - `x:   [1, in_c, t_in, h_in, w_in]`
/// - `out: [1, out_c, (t_in + t_pad) / ft, h_in / fs, w_in / fs]`
///
/// Layout: 0=X, 1=Out, 2=Uniform.
pub trait AvgDown3dOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n_out_elems: u32) -> [u32; 3] {
        super::linear_workgroups(n_out_elems, 64)
    }
}

pub struct AvgDown3dBufs<'a> {
    pub x: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_avgdown3d<O: AvgDown3dOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &AvgDown3dBufs<'_>,
    n_out_elems: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.out.binding(1),
        bufs.uniform.binding(2),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_out_elems))
}

macro_rules! avgdown3d_body {
    () => {
        r#"
struct U {
    in_c: u32, out_c: u32, ft: u32, fs: u32,
    t_in: u32, h_in: u32, w_in: u32, group: u32,
    t_pad: u32, pad0: u32, pad1: u32, pad2: u32,
};

@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    let t_out = (u.t_in + u.t_pad) / u.ft;
    let h_out = u.h_in / u.fs;
    let w_out = u.w_in / u.fs;
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

    let fold = u.ft * u.fs * u.fs;
    let plane = u.h_in * u.w_in;
    var acc = 0.0;
    for (var j = 0u; j < u.group; j = j + 1u) {
        let e = oc * u.group + j;
        let c = e / fold;
        let a = (e / (u.fs * u.fs)) % u.ft;
        let b = (e / u.fs) % u.fs;
        let d = e % u.fs;
        let tf = to * u.ft + a;
        if (tf >= u.t_pad) {
            let t = tf - u.t_pad;
            let idx = c * (u.t_in * plane) + t * plane + (ho * u.fs + b) * u.w_in + wo * u.fs + d;
            acc = acc + f32(x[idx]);
        }
    }
    out[i] = OUT_T(acc / f32(u.group));
}
"#
    };
}

const WGSL_F32: &str = concat!(
    r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
alias OUT_T = f32;
"#,
    avgdown3d_body!()
);

// Native-f16 acts: f16 storage, f32 accumulate.
const WGSL_F16: &str = concat!(
    r#"enable f16;
@group(0) @binding(0) var<storage, read> x: array<f16>;
@group(0) @binding(1) var<storage, read_write> out: array<f16>;
alias OUT_T = f16;
"#,
    avgdown3d_body!()
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

pub struct AvgDown3dF32;

impl AvgDown3dOp for AvgDown3dF32 {
    const KERNEL_ID: &'static str = "avgdown3d.f32";
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
