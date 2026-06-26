use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
use crate::ops::{ActDtype, WgslConfig};
use crate::tensor::{ComputeDtype, F32};

/// Per-location channel-RMS normalization for NCTHW activations (LTX-2 video VAE
/// `PixelNorm`).
///
/// For each spatio-temporal position `(b, t, h, w)`, normalize across the `C`
/// channels: `out[c] = x[c] / sqrt(mean_c(x[c]^2) + eps)`. Unlike
/// [`super::rmsnorm3d`] this carries NO per-channel gain (`PixelNorm` is
/// weightless) and floors with `eps` INSIDE the mean (matching torch's
/// `x / sqrt(mean(x^2) + eps)`, eps `1e-8`), not on the L2 norm.
///
/// The channel axis is dim 1 of NCTHW, so it is NOT contiguous: channel stride
/// is `stride = T*H*W`. One thread owns one `(b, position)` pair and loops over
/// `C` with scalar strided loads.
///
/// Layout: 0=X, 1=Out, 2=Uniform `{n_pos, channels, stride, eps_bits}`.
pub trait PixelNorm3dOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const DIMS: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> String;
    fn layout() -> &'static [BindingLayout];

    /// One thread per `(batch, spatio-temporal position)`.
    fn workgroups(n_pos: u32) -> [u32; 3] {
        crate::ops::linear_workgroups(n_pos, 64)
    }
}

pub struct PixelNorm3dBufs<'a> {
    pub x: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_pixel_norm3d<O: PixelNorm3dOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &PixelNorm3dBufs<'_>,
    n_pos: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.out.binding(1),
        bufs.uniform.binding(2),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_pos))
}

fn build_wgsl(cfg: &WgslConfig) -> String {
    assert!(
        !cfg.bf16_quant_writes,
        "pixel_norm3d has no bf16-quant-write mode"
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
        other => panic!("pixel_norm3d does not support act dtype {other:?}"),
    };

    format!(
        r#"{f16_prelude}
struct U {{ n_pos: u32, channels: u32, stride: u32, eps_bits: u32 }};

{x_decl}
{out_decl}
@group(0) @binding(2) var<uniform> u: U;

{load_x}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {{
    let p: u32 = gid.y * (ng.x * 64u) + gid.x;
    if (p >= u.n_pos) {{ return; }}
    // Decode the flat position p into (batch, spatial s). One batch block holds
    // channels*stride elements; the channel axis steps by `stride` within it.
    let b: u32 = p / u.stride;
    let s: u32 = p - b * u.stride;
    let base: u32 = b * u.channels * u.stride + s;

    var sum_sq: f32 = 0.0;
    for (var c: u32 = 0u; c < u.channels; c = c + 1u) {{
        let v = load_x(base + c * u.stride);
        sum_sq = sum_sq + v * v;
    }}
    let eps: f32 = bitcast<f32>(u.eps_bits);
    let mean_sq: f32 = sum_sq / f32(u.channels);
    let inv: f32 = inverseSqrt(mean_sq + eps);
    for (var c: u32 = 0u; c < u.channels; c = c + 1u) {{
        let idx = base + c * u.stride;
        let v: f32 = load_x(idx) * inv;
        out[idx] = {store_expr};
    }}
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

pub struct PixelNorm3dF32;

impl PixelNorm3dOp for PixelNorm3dF32 {
    const KERNEL_ID: &'static str = "pixel_norm3d.f32";
    type Dtype = F32;
    const X: &'static str = "pixel_norm3d/x";
    const DIMS: &'static str = "pixel_norm3d/dims";
    const OUTPUT: &'static str = "pixel_norm3d/out";
    fn wgsl(cfg: &WgslConfig) -> String {
        build_wgsl(cfg)
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl crate::conformance::OpTest for PixelNorm3dF32 {
    fn dtypes(&self) -> &'static [crate::conformance::Dtype] {
        crate::conformance::DTYPES_FP32_ONLY
    }
    fn test_cases(&self) -> Vec<crate::conformance::TestCase> {
        use crate::conformance::{OpSpec, TestCase, linspace, t};
        vec![
            // NCTHW, channel-RMS over C=8 per (t,h,w) position. eps inside the mean.
            TestCase {
                name: "pixel_norm3d_basic",
                op: OpSpec::PixelNorm3d { eps: 1e-8 },
                inputs: vec![t("x", [1, 8, 2, 3, 3], linspace(-2.0, 2.0, false))],
            },
            // Single-frame chunk (T=1).
            TestCase {
                name: "pixel_norm3d_t1",
                op: OpSpec::PixelNorm3d { eps: 1e-8 },
                inputs: vec![t("x", [1, 16, 1, 4, 4], linspace(-1.0, 1.0, true))],
            },
        ]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a crate::conformance::OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_pixel_norm3d::<PixelNorm3dF32>())
    }
}
