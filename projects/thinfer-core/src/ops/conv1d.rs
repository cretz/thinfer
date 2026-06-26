use super::{ActDtype, WeightDtype, WgslConfig};
use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
use crate::tensor::{ComputeDtype, F32};

/// `out = conv1d(x, weight) + bias` over NCL (1D conv), with stride, dilation,
/// groups, and SYMMETRIC padding.
///
/// Shapes (row-major):
/// - `x:      [B, Cin, Lin]`
/// - `weight: [Cout, Cin/groups, K]` (PyTorch `nn.Conv1d` layout)
/// - `bias:   [Cout]`
/// - `out:    [B, Cout, Lout]`
///
/// Geometry: `Lout = (Lin + 2*pad - dilation*(K-1) - 1)/stride + 1`. Padding is
/// symmetric (`pad` zeros on BOTH sides), unlike the causal front-pad of
/// [`super::conv3d`]. Groups split channels: output channel `co` belongs to group
/// `g = co / (Cout/groups)` and convolves only the input channels in
/// `[g*Cin/groups, (g+1)*Cin/groups)`; depthwise is `groups == Cin == Cout`.
///
/// Direct (non-GEMM) kernel: one thread per `(b, co, lo)` output element, looping
/// over the group's input channels and the K taps. This is the BigVGAN vocoder
/// path (P5) which runs once per request in f32, so a simple correct kernel is
/// preferred over an im2col-GEMM; revisit if it ever becomes a bottleneck.
///
/// Layout: 0=X, 1=W, 2=Bias, 3=Out, 4=Uniform.
pub trait Conv1dOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const W: &'static str;
    const BIAS: &'static str;
    const DIMS: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> String;
    fn layout() -> &'static [BindingLayout];

    /// One thread per output element `(b, co, lo)`.
    fn workgroups(n_out: u32) -> [u32; 3] {
        crate::ops::linear_workgroups(n_out, 64)
    }
}

pub struct Conv1dBufs<'a> {
    pub x: &'a BufRef,
    pub w: &'a BufRef,
    pub bias: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

/// Uniform layout shared by [`conv1d_uniform_bytes`] and the WGSL `U` struct.
/// `[b, cin, cout, lin, lout, k, stride, dilation, pad, groups, cin_pg, cout_pg]`
/// padded to 64 bytes (16 u32).
#[allow(clippy::too_many_arguments)]
pub fn conv1d_uniform_bytes(
    b: u32,
    cin: u32,
    cout: u32,
    lin: u32,
    lout: u32,
    k: u32,
    stride: u32,
    dilation: u32,
    pad: u32,
    groups: u32,
) -> [u8; 64] {
    let fields: [u32; 16] = [
        b,
        cin,
        cout,
        lin,
        lout,
        k,
        stride,
        dilation,
        pad,
        groups,
        cin / groups,
        cout / groups,
        0,
        0,
        0,
        0,
    ];
    let mut bytes = [0u8; 64];
    for (i, v) in fields.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// `Lout` for the given geometry.
pub fn conv1d_lout(lin: u32, k: u32, stride: u32, dilation: u32, pad: u32) -> u32 {
    (lin + 2 * pad - dilation * (k - 1) - 1) / stride + 1
}

pub(crate) fn dispatch_conv1d<O: Conv1dOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &Conv1dBufs<'_>,
    n_out: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.w.binding(1),
        bufs.bias.binding(2),
        bufs.out.binding(3),
        bufs.uniform.binding(4),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_out))
}

fn build_wgsl(cfg: &WgslConfig) -> String {
    assert!(
        !cfg.bf16_quant_writes,
        "conv1d has no bf16-quant-write mode"
    );
    let (f16_prelude, x_decl, out_decl, load_x, store_expr) = match cfg.act_dtype {
        ActDtype::F32 => (
            "",
            "@group(0) @binding(0) var<storage, read> x: array<f32>;",
            "@group(0) @binding(3) var<storage, read_write> out: array<f32>;",
            "fn load_x(i: u32) -> f32 { return x[i]; }\n",
            "v",
        ),
        ActDtype::F16 => (
            "enable f16;\n",
            "@group(0) @binding(0) var<storage, read> x: array<f16>;",
            "@group(0) @binding(3) var<storage, read_write> out: array<f16>;",
            "fn load_x(i: u32) -> f32 { return f32(x[i]); }\n",
            "f16(clamp(v, -65504.0, 65504.0))",
        ),
        other => panic!("conv1d does not support act dtype {other:?}"),
    };
    let (w_decl, load_w) = match cfg.weight_dtype {
        WeightDtype::F32 => (
            concat!(
                "@group(0) @binding(1) var<storage, read> wgt: array<f32>;\n",
                "@group(0) @binding(2) var<storage, read> bias: array<f32>;",
            ),
            concat!(
                "fn load_w(i: u32) -> f32 { return wgt[i]; }\n",
                "fn load_bias(i: u32) -> f32 { return bias[i]; }\n",
            ),
        ),
        WeightDtype::Bf16 => (
            concat!(
                "@group(0) @binding(1) var<storage, read> wgt: array<u32>;\n",
                "@group(0) @binding(2) var<storage, read> bias: array<u32>;",
            ),
            concat!(
                "fn unpack_bf16(pair: u32, i: u32) -> f32 {\n",
                "  let half = (pair >> ((i & 1u) * 16u)) & 0xFFFFu;\n",
                "  return bitcast<f32>(half << 16u);\n",
                "}\n",
                "fn load_w(i: u32) -> f32 { return unpack_bf16(wgt[i >> 1u], i); }\n",
                "fn load_bias(i: u32) -> f32 { return unpack_bf16(bias[i >> 1u], i); }\n",
            ),
        ),
        other => panic!("conv1d does not consume weight dtype {other:?}"),
    };

    format!(
        r#"{f16_prelude}
struct U {{
    b: u32, cin: u32, cout: u32, lin: u32,
    lout: u32, k: u32, stride: u32, dilation: u32,
    pad: u32, groups: u32, cin_pg: u32, cout_pg: u32,
    _p0: u32, _p1: u32, _p2: u32, _p3: u32,
}};

{x_decl}
{w_decl}
{out_decl}
@group(0) @binding(4) var<uniform> u: U;

{load_x}
{load_w}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {{
    let idx: u32 = gid.y * (ng.x * 64u) + gid.x;
    let n_out: u32 = u.b * u.cout * u.lout;
    if (idx >= n_out) {{ return; }}
    let bnum: u32 = idx / (u.cout * u.lout);
    let rem: u32 = idx - bnum * (u.cout * u.lout);
    let co: u32 = rem / u.lout;
    let lo: u32 = rem - co * u.lout;

    let g: u32 = co / u.cout_pg;
    let ci_base: u32 = g * u.cin_pg;
    let x_b: u32 = bnum * u.cin * u.lin;
    let w_co: u32 = co * u.cin_pg * u.k; // weight [Cout, Cin/groups, K]

    var acc: f32 = load_bias(co);
    for (var kk: u32 = 0u; kk < u.k; kk = kk + 1u) {{
        let li_s: i32 = i32(lo * u.stride + kk * u.dilation) - i32(u.pad);
        if (li_s >= 0 && li_s < i32(u.lin)) {{
            let li: u32 = u32(li_s);
            for (var cl: u32 = 0u; cl < u.cin_pg; cl = cl + 1u) {{
                let ci: u32 = ci_base + cl;
                let xv: f32 = load_x(x_b + ci * u.lin + li);
                let wv: f32 = load_w(w_co + cl * u.k + kk);
                acc = fma(wv, xv, acc);
            }}
        }}
    }}
    let v: f32 = acc;
    out[idx] = {store_expr};
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

pub struct Conv1dF32;

impl Conv1dOp for Conv1dF32 {
    const KERNEL_ID: &'static str = "conv1d.f32";
    type Dtype = F32;
    const X: &'static str = "conv1d/x";
    const W: &'static str = "conv1d/w";
    const BIAS: &'static str = "conv1d/bias";
    const DIMS: &'static str = "conv1d/dims";
    const OUTPUT: &'static str = "conv1d/out";
    fn wgsl(cfg: &WgslConfig) -> String {
        build_wgsl(cfg)
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl crate::conformance::OpTest for Conv1dF32 {
    fn dtypes(&self) -> &'static [crate::conformance::Dtype] {
        crate::conformance::DTYPES_FP32_ONLY
    }
    fn test_cases(&self) -> Vec<crate::conformance::TestCase> {
        use crate::conformance::{OpSpec, TestCase, linspace, t};
        vec![
            // k7 pad3 stride1 groups1: the BigVGAN conv_pre/conv_post geometry.
            TestCase {
                name: "conv1d_k7_pad3",
                op: OpSpec::Conv1d {
                    k: 7,
                    stride: 1,
                    dilation: 1,
                    pad: 3,
                    groups: 1,
                },
                inputs: vec![
                    t("x", [1, 6, 20], linspace(-1.0, 1.0, false)),
                    t("w", [8, 6, 7], linspace(-0.3, 0.3, false)),
                    t("bias", [8], linspace(-0.2, 0.2, false)),
                ],
            },
            // Dilated k3 (resblock convs1: dilation 3), symmetric pad keeps Lout==Lin.
            TestCase {
                name: "conv1d_k3_dilation3",
                op: OpSpec::Conv1d {
                    k: 3,
                    stride: 1,
                    dilation: 3,
                    pad: 3,
                    groups: 1,
                },
                inputs: vec![
                    t("x", [1, 4, 16], linspace(-1.0, 1.0, false)),
                    t("w", [4, 4, 3], linspace(-0.25, 0.25, true)),
                    t("bias", [4], linspace(-0.1, 0.1, false)),
                ],
            },
            // Depthwise (groups == cin == cout): the anti-alias LowPassFilter1d.
            TestCase {
                name: "conv1d_depthwise_stride2",
                op: OpSpec::Conv1d {
                    k: 12,
                    stride: 2,
                    dilation: 1,
                    pad: 5,
                    groups: 5,
                },
                inputs: vec![
                    t("x", [1, 5, 24], linspace(-1.0, 1.0, false)),
                    t("w", [5, 1, 12], linspace(-0.2, 0.2, false)),
                    t("bias", [5], linspace(-0.05, 0.05, false)),
                ],
            },
        ]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a crate::conformance::OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_conv1d::<Conv1dF32>())
    }
}
