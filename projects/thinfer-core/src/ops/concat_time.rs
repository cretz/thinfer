use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
use crate::ops::{ActDtype, WgslConfig};
use crate::tensor::{ComputeDtype, F32};

/// Concatenate two NCTHW activations along the time axis, with per-input range
/// selection and an optional zero front-pad. This is the Wan VAE `feat_cache`
/// primitive: assembling cached frames in front of the current chunk before a
/// causal conv, and building the next cache from trailing frames, both reduce
/// to a single dispatch of this op.
///
/// Output is `[B, C, a_count + b_count, H, W]`:
/// - output time `to < a_count` reads `A[:, :, a_start + to, :, :]` (or zero
///   when `a_zero != 0`, which models the `"Rep"` zero-prefix case);
/// - output time `to >= a_count` reads `B[:, :, b_start + (to - a_count), :, :]`.
///
/// `A` and `B` share `B, C, H, W`; their time extents are `a_t` / `b_t`. Setting
/// `b_count = 0` makes this a pure tail-slice of `A`; `a_zero = 1` makes the
/// front `a_count` frames zero (then `A` is unread, but a buffer must still be
/// bound - pass any same-dtype buffer).
///
/// Layout: 0=A, 1=B, 2=Out, 3=Uniform.
pub trait ConcatTimeOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const A: &'static str;
    const B: &'static str;
    const DIMS: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> String;
    fn layout() -> &'static [BindingLayout];

    /// One thread per output element.
    fn workgroups(n_out: u32) -> [u32; 3] {
        super::linear_workgroups(n_out, 64)
    }
}

pub struct ConcatTimeBufs<'a> {
    pub a: &'a BufRef,
    pub b: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_concat_time<O: ConcatTimeOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &ConcatTimeBufs<'_>,
    n_out: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.a.binding(0),
        bufs.b.binding(1),
        bufs.out.binding(2),
        bufs.uniform.binding(3),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_out))
}

fn build_wgsl(cfg: &WgslConfig) -> String {
    assert!(
        !cfg.bf16_quant_writes,
        "concat_time has no bf16-quant-write mode"
    );
    let (f16_prelude, a_decl, b_decl, out_decl, load_a, load_b, store_expr) = match cfg.act_dtype {
        ActDtype::F32 => (
            "",
            "@group(0) @binding(0) var<storage, read> a: array<f32>;",
            "@group(0) @binding(1) var<storage, read> b: array<f32>;",
            "@group(0) @binding(2) var<storage, read_write> out: array<f32>;",
            "fn load_a(i: u32) -> f32 { return a[i]; }\n",
            "fn load_b(i: u32) -> f32 { return b[i]; }\n",
            "v",
        ),
        ActDtype::F16 => (
            "enable f16;\n",
            "@group(0) @binding(0) var<storage, read> a: array<f16>;",
            "@group(0) @binding(1) var<storage, read> b: array<f16>;",
            "@group(0) @binding(2) var<storage, read_write> out: array<f16>;",
            "fn load_a(i: u32) -> f32 { return f32(a[i]); }\n",
            "fn load_b(i: u32) -> f32 { return f32(b[i]); }\n",
            "f16(clamp(v, -65504.0, 65504.0))",
        ),
        other => panic!("concat_time does not support act dtype {other:?}"),
    };

    format!(
        r#"{f16_prelude}
struct U {{
    b: u32, c: u32, h: u32, w: u32,
    a_t: u32, b_t: u32, a_start: u32, a_count: u32,
    b_start: u32, b_count: u32, a_zero: u32, _pad: u32,
}};

{a_decl}
{b_decl}
{out_decl}
@group(0) @binding(3) var<uniform> u: U;

{load_a}
{load_b}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {{
    let o: u32 = gid.y * (ng.x * 64u) + gid.x;
    let hw: u32 = u.h * u.w;
    let t_out: u32 = u.a_count + u.b_count;
    let n_out: u32 = u.b * u.c * t_out * hw;
    if (o >= n_out) {{ return; }}

    // Decode flat output index into (bb, cc, to, spatial s = h*w + w).
    let s: u32 = o % hw;
    var rem: u32 = o / hw;
    let to: u32 = rem % t_out;
    rem = rem / t_out;
    let cc: u32 = rem % u.c;
    let bb: u32 = rem / u.c;
    let bc: u32 = bb * u.c + cc;

    var v: f32 = 0.0;
    if (to < u.a_count) {{
        if (u.a_zero == 0u) {{
            let ti: u32 = u.a_start + to;
            v = load_a((bc * u.a_t + ti) * hw + s);
        }}
    }} else {{
        let ti: u32 = u.b_start + (to - u.a_count);
        v = load_b((bc * u.b_t + ti) * hw + s);
    }}
    out[o] = {store_expr};
}}
"#
    )
}

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

pub struct ConcatTimeF32;

impl ConcatTimeOp for ConcatTimeF32 {
    const KERNEL_ID: &'static str = "concat_time.f32";
    type Dtype = F32;
    const A: &'static str = "concat_time/a";
    const B: &'static str = "concat_time/b";
    const DIMS: &'static str = "concat_time/dims";
    const OUTPUT: &'static str = "concat_time/out";
    fn wgsl(cfg: &WgslConfig) -> String {
        build_wgsl(cfg)
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}
