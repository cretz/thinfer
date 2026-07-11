use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
use crate::ops::{ActDtype, WgslConfig};
use crate::tensor::{ComputeDtype, F32};

/// Depth-to-space (pixel-shuffle) upsample for NCTHW activations (LTX-2 video
/// VAE `DepthToSpaceUpsample`).
///
/// Realizes the einops `b (c p1 p2 p3) t h w -> b c (t p1) (h p2) (w p3)`: the
/// input channel block of size `P1*P2*P3` is unrolled into the temporal/spatial
/// axes. Input `[B, Cin, Tin, Hin, Win]` -> output `[B, Cin/(P1*P2*P3),
/// Tin*P1 - t_drop, Hin*P2, Win*P3]`. `t_drop` (0 or 1) drops the leading output
/// frame after a temporal (P1==2) shuffle, matching the upstream
/// `x = x[:, :, 1:]` trim.
///
/// Channel decomposition is row-major over `(c, p1, p2, p3)`: the source channel
/// for output `(co, p1, p2, p3)` is `((co*P1 + p1)*P2 + p2)*P3 + p3`. One thread
/// per output element gathers its single source value (pure copy, no arithmetic).
///
/// `base_cout` supports the residual up-shortcut (`DepthToSpaceUpsample(residual=
/// True)`): the output channel index is wrapped by `base_cout` before decoding
/// the source channel, so a shuffle into `base_cout` channels is tiled (torch
/// `.repeat`) up to `cout = base_cout * repeat`. Set `base_cout == cout` for the
/// plain (non-residual) shuffle, which makes `co % cout == co` a no-op.
///
/// Layout: 0=X, 1=Out, 2=Uniform.
pub trait DepthToSpace3dOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const DIMS: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> String;
    fn layout() -> &'static [BindingLayout];

    /// One thread per output element.
    fn workgroups(n_out: u32) -> [u32; 3] {
        crate::ops::linear_workgroups(n_out, 64)
    }
}

pub struct DepthToSpace3dBufs<'a> {
    pub x: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_depth_to_space3d<O: DepthToSpace3dOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &DepthToSpace3dBufs<'_>,
    n_out: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.out.binding(1),
        bufs.uniform.binding(2),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_out))
}

fn build_wgsl(cfg: &WgslConfig) -> String {
    assert!(
        !cfg.bf16_quant_writes,
        "depth_to_space3d has no bf16-quant-write mode"
    );
    let (f16_prelude, x_decl, out_decl, load_x, store_expr) = match cfg.act_dtype {
        ActDtype::F32 => (
            "",
            "@group(0) @binding(0) var<storage, read> x: array<f32>;",
            "@group(0) @binding(1) var<storage, read_write> out: array<f32>;",
            "fn load_x(i: u32) -> f32 { return x[i]; }\n",
            "v",
        ),
        ActDtype::F16 => (
            "enable f16;\n",
            "@group(0) @binding(0) var<storage, read> x: array<f16>;",
            "@group(0) @binding(1) var<storage, read_write> out: array<f16>;",
            "fn load_x(i: u32) -> f32 { return f32(x[i]); }\n",
            "f16(clamp(v, -65504.0, 65504.0))",
        ),
        other => panic!("depth_to_space3d does not support act dtype {other:?}"),
    };

    format!(
        r#"{f16_prelude}
struct U {{
    cin: u32, t_in: u32, h_in: u32, w_in: u32,
    p1: u32, p2: u32, p3: u32, t_drop: u32,
    cout: u32, t_out: u32, h_out: u32, w_out: u32,
    base_cout: u32, pad0: u32, pad1: u32, pad2: u32,
}};

{x_decl}
{out_decl}
@group(0) @binding(2) var<uniform> u: U;

{load_x}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {{
    let o: u32 = gid.y * (ng.x * 64u) + gid.x;
    let hw_out: u32 = u.h_out * u.w_out;
    let n_out: u32 = u.cout * u.t_out * hw_out;
    if (o >= n_out) {{ return; }}

    // Decode flat output index into (co, to, ho, wo).
    let wo: u32 = o % u.w_out;
    var rem: u32 = o / u.w_out;
    let ho: u32 = rem % u.h_out;
    rem = rem / u.h_out;
    let to: u32 = rem % u.t_out;
    let co: u32 = rem / u.t_out;

    // Output time index after the leading-frame drop maps back to the full
    // pre-drop index `to + t_drop`.
    let tf: u32 = to + u.t_drop;
    let p1: u32 = tf % u.p1;
    let ti: u32 = tf / u.p1;
    let p2: u32 = ho % u.p2;
    let hi: u32 = ho / u.p2;
    let p3: u32 = wo % u.p3;
    let wi: u32 = wo / u.p3;

    // Wrap the output channel into the shuffle group so a `base_cout`-channel
    // shuffle is tiled up to `cout` (the residual up-shortcut); a no-op when
    // `base_cout == cout`.
    let cb: u32 = co % u.base_cout;
    let ci: u32 = ((cb * u.p1 + p1) * u.p2 + p2) * u.p3 + p3;
    let in_idx: u32 = ((ci * u.t_in + ti) * u.h_in + hi) * u.w_in + wi;
    let v: f32 = load_x(in_idx);
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
        kind: BindingKind::StorageReadWrite,
    },
    BindingLayout {
        slot: 2,
        kind: BindingKind::Uniform,
    },
];

pub struct DepthToSpace3dF32;

impl DepthToSpace3dOp for DepthToSpace3dF32 {
    const KERNEL_ID: &'static str = "depth_to_space3d.f32";
    type Dtype = F32;
    const X: &'static str = "depth_to_space3d/x";
    const DIMS: &'static str = "depth_to_space3d/dims";
    const OUTPUT: &'static str = "depth_to_space3d/out";
    fn wgsl(cfg: &WgslConfig) -> String {
        build_wgsl(cfg)
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl crate::conformance::OpTest for DepthToSpace3dF32 {
    fn dtypes(&self) -> &'static [crate::conformance::Dtype] {
        crate::conformance::DTYPES_FP32_ONLY
    }
    fn test_cases(&self) -> Vec<crate::conformance::TestCase> {
        use crate::conformance::{OpSpec, TestCase, linspace, t};
        // Input [cin, t_in, h_in, w_in]; cin must be cout*p1*p2*p3.
        vec![
            // Spatial-only upsample (p1=1, no frame drop): [8,2,2,2] -> cout=2.
            TestCase {
                name: "depth_to_space3d_spatial",
                op: OpSpec::DepthToSpace3d {
                    p1: 1,
                    p2: 2,
                    p3: 2,
                },
                inputs: vec![t("x", [8, 2, 2, 2], linspace(-2.0, 2.0, false))],
            },
            // Temporal x2 (p1=2 -> leading frame dropped): [16,2,2,2] -> cout=2,
            // t_out = 2*2 - 1 = 3.
            TestCase {
                name: "depth_to_space3d_temporal",
                op: OpSpec::DepthToSpace3d {
                    p1: 2,
                    p2: 2,
                    p3: 2,
                },
                inputs: vec![t("x", [16, 2, 2, 2], linspace(-1.0, 1.0, true))],
            },
        ]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a crate::conformance::OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_depth_to_space3d::<DepthToSpace3dF32>())
    }
}
