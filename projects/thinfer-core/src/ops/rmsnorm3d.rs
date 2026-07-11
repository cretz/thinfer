use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
use crate::ops::{ActDtype, WeightDtype, WgslConfig};
use crate::tensor::{ComputeDtype, F32};

/// Channel-dim RMS normalization for NCTHW activations (`WanRMS_norm`).
///
/// For each spatio-temporal position `(b, t, h, w)`, normalize across the `C`
/// channels: `out[c] = x[c] * rsqrt(sum_c x[c]^2) * sqrt(C) * gamma[c]`. This is
/// HF's `F.normalize(x, dim=1) * scale * gamma` with `scale = C**0.5` (the Wan
/// VAE uses `bias=False`), which is algebraically RMSNorm across the channel
/// axis with per-channel gain.
///
/// The channel axis is dim 1 of NCTHW, so it is NOT contiguous: channel stride
/// is `stride = T*H*W`. One thread owns one `(b, position)` pair and loops over
/// `C` with scalar strided loads (no packed-pair path). `gamma` is `[C]`.
///
/// Layout: 0=X, 1=W (gamma `[C]`), 2=Out, 3=Uniform `{n_pos, channels, stride, _pad}`.
pub trait RmsNorm3dOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const W: &'static str;
    const DIMS: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> String;
    fn layout() -> &'static [BindingLayout];

    /// One thread per `(batch, spatio-temporal position)`.
    fn workgroups(n_pos: u32) -> [u32; 3] {
        crate::ops::linear_workgroups(n_pos, 64)
    }
}

pub struct RmsNorm3dBufs<'a> {
    pub x: &'a BufRef,
    pub w: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_rmsnorm3d<O: RmsNorm3dOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &RmsNorm3dBufs<'_>,
    n_pos: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.w.binding(1),
        bufs.out.binding(2),
        bufs.uniform.binding(3),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_pos))
}

/// `F.normalize` floors the L2 denominator at this eps; carry it so all-zero
/// channels map to zero rather than NaN.
const NORM_EPS: &str = "1e-12";

fn build_wgsl(cfg: &WgslConfig) -> String {
    assert!(
        !cfg.bf16_quant_writes,
        "rmsnorm3d has no bf16-quant-write mode"
    );
    let (f16_prelude, x_decl, out_decl, load_x, store_expr) = match cfg.act_dtype {
        ActDtype::F32 => (
            "",
            "@group(0) @binding(0) var<storage, read> x: array<f32>;",
            "@group(0) @binding(2) var<storage, read_write> out: array<f32>;",
            "fn load_x(i: u32) -> f32 { return x[i]; }\n",
            "v",
        ),
        ActDtype::F16 => (
            "enable f16;\n",
            "@group(0) @binding(0) var<storage, read> x: array<f16>;",
            "@group(0) @binding(2) var<storage, read_write> out: array<f16>;",
            "fn load_x(i: u32) -> f32 { return f32(x[i]); }\n",
            "f16(clamp(v, -65504.0, 65504.0))",
        ),
        other => panic!("rmsnorm3d does not support act dtype {other:?}"),
    };
    let (w_decl, load_w) = match cfg.weight_dtype {
        WeightDtype::F32 => (
            "@group(0) @binding(1) var<storage, read> gamma: array<f32>;",
            "fn load_w(i: u32) -> f32 { return gamma[i]; }\n",
        ),
        WeightDtype::Bf16 => (
            "@group(0) @binding(1) var<storage, read> gamma: array<u32>;",
            concat!(
                "fn load_w(i: u32) -> f32 {\n",
                "  let pair = gamma[i >> 1u];\n",
                "  let half = (pair >> ((i & 1u) * 16u)) & 0xFFFFu;\n",
                "  return bitcast<f32>(half << 16u);\n",
                "}\n",
            ),
        ),
        WeightDtype::F16 => unreachable!("rmsnorm3d does not consume f16 weights"),
        WeightDtype::Quant(_) => unreachable!("rmsnorm3d does not consume quant weights"),
    };

    format!(
        r#"{f16_prelude}
struct U {{ n_pos: u32, channels: u32, stride: u32, _pad: u32 }};

{x_decl}
{w_decl}
{out_decl}
@group(0) @binding(3) var<uniform> u: U;

{load_x}
{load_w}

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
    let scale: f32 = sqrt(f32(u.channels));
    let inv: f32 = scale / max(sqrt(sum_sq), {NORM_EPS});
    for (var c: u32 = 0u; c < u.channels; c = c + 1u) {{
        let idx = base + c * u.stride;
        let v: f32 = load_x(idx) * inv * load_w(c);
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

pub struct RmsNorm3dF32;

impl RmsNorm3dOp for RmsNorm3dF32 {
    const KERNEL_ID: &'static str = "rmsnorm3d.f32";
    type Dtype = F32;
    const X: &'static str = "rmsnorm3d/x";
    const W: &'static str = "rmsnorm3d/w";
    const DIMS: &'static str = "rmsnorm3d/dims";
    const OUTPUT: &'static str = "rmsnorm3d/out";
    fn wgsl(cfg: &WgslConfig) -> String {
        build_wgsl(cfg)
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl crate::conformance::OpTest for RmsNorm3dF32 {
    fn dtypes(&self) -> &'static [crate::conformance::Dtype] {
        crate::conformance::DTYPES_FP32_ONLY
    }
    fn test_cases(&self) -> Vec<crate::conformance::TestCase> {
        use crate::conformance::{OpSpec, TestCase, linspace, t};
        vec![
            // NCTHW, C=8 across spatial positions. gamma per-channel.
            TestCase {
                name: "rmsnorm3d_basic",
                op: OpSpec::RmsNorm3d,
                inputs: vec![
                    t("x", [1, 8, 2, 3, 3], linspace(-2.0, 2.0, false)),
                    t("w", [8], linspace(0.5, 1.5, false)),
                ],
            },
            // Single frame (T=1, the per-latent-frame decode chunk shape).
            TestCase {
                name: "rmsnorm3d_t1",
                op: OpSpec::RmsNorm3d,
                inputs: vec![
                    t("x", [1, 16, 1, 4, 4], linspace(-1.0, 1.0, false)),
                    t("w", [16], linspace(0.8, 1.2, true)),
                ],
            },
        ]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a crate::conformance::OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_rmsnorm3d::<RmsNorm3dF32>())
    }
}
