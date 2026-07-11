use super::{ActDtype, WgslConfig};
use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
use crate::tensor::{ComputeDtype, F32};

/// `HunyuanUpsample3d`: the spatial/temporal upsampler of the HunyuanVideo 1.5
/// VAE decoder (`hunyuanvideo_15_vae.py::Upsample.forward`). Fuses the conv-output
/// pixelshuffle and the `repeat_interleave` residual shortcut into one gather+add.
///
/// Inputs (both at the upsampler's input grid `[*, f_in, h_in, w_in]`, B=1):
/// - `h`: the causal-conv output `[out_c * factor, f_in, h_in, w_in]`, factor =
///   8 (temporal) or 4 (spatial-only).
/// - `x`: the pre-conv input `[in_c, f_in, h_in, w_in]` (the residual source).
///
/// Output `[out_c, F, 2*h_in, 2*w_in]`, `F = 2*f_in - 1` (temporal) or `f_in`
/// (spatial-only). `r = 4*out_c/in_c` is the spatial / first-frame
/// `repeat_interleave`; the temporal "next" frames use `2*r`.
///
/// Per output `(co, ofr, oh, ow)` with `r2=oh&1, hi=oh>>1, r3=ow&1, wi=ow>>1`:
/// - spatial: `hp = (r2*2+r3)*out_c + co`; `out = h[hp, ofr, hi, wi] + x[hp/r,
///   ofr, hi, wi]`.
/// - temporal, `ofr==0`: `hp = (r2*2+r3)*(2*out_c) + co` (first-frame half-channel
///   split); residual channel `hpr = (r2*2+r3)*out_c + co`; `out = h[hp, 0, hi,
///   wi] + x[hpr/r, 0, hi, wi]`.
/// - temporal, `ofr>=1`: `r1=(ofr-1)&1, fi=(ofr-1)>>1, sf=1+fi`; `hp =
///   ((r1*2+r2)*2+r3)*out_c + co`; `out = h[hp, sf, hi, wi] + x[hp/(2*r), sf, hi,
///   wi]`.
///
/// `out_c*factor` divides cleanly so every channel quotient is exact (in_c | 4,
/// 8 by construction). Layout: 0=H, 1=X, 2=Out, 3=Uniform.
pub trait HunyuanUpsample3dOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n_out_elems: u32) -> [u32; 3] {
        super::linear_workgroups(n_out_elems, 64)
    }
}

pub struct HunyuanUpsample3dBufs<'a> {
    pub h: &'a BufRef,
    pub x: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_hunyuan_upsample3d<O: HunyuanUpsample3dOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &HunyuanUpsample3dBufs<'_>,
    n_out_elems: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.h.binding(0),
        bufs.x.binding(1),
        bufs.out.binding(2),
        bufs.uniform.binding(3),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_out_elems))
}

macro_rules! hunyuan_upsample3d_body {
    () => {
        r#"
struct U {
    in_c: u32, out_c: u32, temporal: u32, f_in: u32,
    h_in: u32, w_in: u32, r: u32, pad0: u32,
};

@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    let f_out = select(u.f_in, 2u * u.f_in - 1u, u.temporal == 1u);
    let h_out = u.h_in * 2u;
    let w_out = u.w_in * 2u;
    let total = u.out_c * f_out * h_out * w_out;
    if (i >= total) { return; }

    let hw_out = h_out * w_out;
    let thw_out = f_out * hw_out;
    let co = i / thw_out;
    let r1o = i - co * thw_out;
    let ofr = r1o / hw_out;
    let r2o = r1o - ofr * hw_out;
    let oh = r2o / w_out;
    let ow = r2o - oh * w_out;

    let r2 = oh & 1u;
    let hi = oh >> 1u;
    let r3 = ow & 1u;
    let wi = ow >> 1u;

    let hw_in = u.h_in * u.w_in;
    let h_chan = u.f_in * hw_in; // h/x channel stride (f_in frames)

    var hp: u32;        // packed channel into `h`
    var sf: u32;        // source frame in h/x
    var xc: u32;        // source channel into `x`
    if (u.temporal == 0u) {
        hp = (r2 * 2u + r3) * u.out_c + co;
        sf = ofr;
        xc = hp / u.r;
    } else if (ofr == 0u) {
        hp = (r2 * 2u + r3) * (2u * u.out_c) + co;
        sf = 0u;
        let hpr = (r2 * 2u + r3) * u.out_c + co;
        xc = hpr / u.r;
    } else {
        let ofp = ofr - 1u;
        let r1 = ofp & 1u;
        sf = 1u + (ofp >> 1u);
        hp = ((r1 * 2u + r2) * 2u + r3) * u.out_c + co;
        xc = hp / (2u * u.r);
    }

    let h_idx = hp * h_chan + sf * hw_in + hi * u.w_in + wi;
    let x_idx = xc * h_chan + sf * hw_in + hi * u.w_in + wi;
    out[i] = h[h_idx] + x[x_idx];
}
"#
    };
}

const WGSL_F32: &str = concat!(
    r#"
@group(0) @binding(0) var<storage, read> h: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
"#,
    hunyuan_upsample3d_body!()
);

const WGSL_F16: &str = concat!(
    r#"enable f16;
@group(0) @binding(0) var<storage, read> h: array<f16>;
@group(0) @binding(1) var<storage, read> x: array<f16>;
@group(0) @binding(2) var<storage, read_write> out: array<f16>;
"#,
    hunyuan_upsample3d_body!()
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

pub struct HunyuanUpsample3dF32;

impl HunyuanUpsample3dOp for HunyuanUpsample3dF32 {
    const KERNEL_ID: &'static str = "hunyuan_upsample3d.f32";
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
