use super::{ActDtype, WeightDtype, WgslConfig};
use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
use crate::tensor::{ComputeDtype, F32};

/// `out = conv_transpose1d(x, weight) + bias` over NCL, with stride, dilation,
/// groups, and symmetric padding (output_padding is always 0 here).
///
/// Shapes (row-major):
/// - `x:      [B, Cin, Lin]`
/// - `weight: [Cin, Cout/groups, K]` (PyTorch `nn.ConvTranspose1d` layout: the
///   FIRST weight axis is the INPUT channel count, unlike forward conv)
/// - `bias:   [Cout]`
/// - `out:    [B, Cout, Lout]`
///
/// Geometry: `Lout = (Lin-1)*stride - 2*pad + dilation*(K-1) + 1`. The vocoder
/// (P5) uses transposed conv for the BigVGAN `ups` (stride = upsample rate,
/// groups=1) and for the anti-alias / Hann `UpSample1d` (stride = ratio, groups =
/// channels, a SHARED expanded sinc filter). dilation is 1 in all current sites
/// but supported for generality.
///
/// Direct (gather) kernel: one thread per `(b, co, lo)` output element. For each
/// tap `kk`, the contributing input position is `li = (lo + pad - kk*dilation)/
/// stride` when that quantity is non-negative and divisible by `stride`; the
/// thread accumulates over the group's input channels. Once-per-request f32, so
/// correctness over GEMM cleverness (see [`super::conv1d`]).
///
/// Layout: 0=X, 1=W, 2=Bias, 3=Out, 4=Uniform.
pub trait ConvTranspose1dOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const W: &'static str;
    const BIAS: &'static str;
    const DIMS: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> String;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n_out: u32) -> [u32; 3] {
        crate::ops::linear_workgroups(n_out, 64)
    }
}

pub struct ConvTranspose1dBufs<'a> {
    pub x: &'a BufRef,
    pub w: &'a BufRef,
    pub bias: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

/// `Lout` for a transposed conv (output_padding 0).
pub fn conv_transpose1d_lout(lin: u32, k: u32, stride: u32, dilation: u32, pad: u32) -> u32 {
    (lin - 1) * stride + dilation * (k - 1) + 1 - 2 * pad
}

pub(crate) fn dispatch_conv_transpose1d<O: ConvTranspose1dOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &ConvTranspose1dBufs<'_>,
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
        "conv_transpose1d has no bf16-quant-write mode"
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
        other => panic!("conv_transpose1d does not support act dtype {other:?}"),
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
        other => panic!("conv_transpose1d does not consume weight dtype {other:?}"),
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
    let co_local: u32 = co - g * u.cout_pg;
    let ci_base: u32 = g * u.cin_pg;
    let x_b: u32 = bnum * u.cin * u.lin;

    var acc: f32 = load_bias(co);
    for (var kk: u32 = 0u; kk < u.k; kk = kk + 1u) {{
        let pos: i32 = i32(lo + u.pad) - i32(kk * u.dilation);
        if (pos >= 0 && (u32(pos) % u.stride) == 0u) {{
            let li: u32 = u32(pos) / u.stride;
            if (li < u.lin) {{
                for (var cl: u32 = 0u; cl < u.cin_pg; cl = cl + 1u) {{
                    let ci: u32 = ci_base + cl;
                    // weight [Cin, Cout/groups, K]
                    let wv: f32 = load_w((ci * u.cout_pg + co_local) * u.k + kk);
                    let xv: f32 = load_x(x_b + ci * u.lin + li);
                    acc = fma(wv, xv, acc);
                }}
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

pub struct ConvTranspose1dF32;

impl ConvTranspose1dOp for ConvTranspose1dF32 {
    const KERNEL_ID: &'static str = "conv_transpose1d.f32";
    type Dtype = F32;
    const X: &'static str = "conv_transpose1d/x";
    const W: &'static str = "conv_transpose1d/w";
    const BIAS: &'static str = "conv_transpose1d/bias";
    const DIMS: &'static str = "conv_transpose1d/dims";
    const OUTPUT: &'static str = "conv_transpose1d/out";
    fn wgsl(cfg: &WgslConfig) -> String {
        build_wgsl(cfg)
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl crate::conformance::OpTest for ConvTranspose1dF32 {
    fn dtypes(&self) -> &'static [crate::conformance::Dtype] {
        crate::conformance::DTYPES_FP32_ONLY
    }
    fn test_cases(&self) -> Vec<crate::conformance::TestCase> {
        use crate::conformance::{OpSpec, TestCase, linspace, t};
        vec![
            // BigVGAN ups geometry: stride 4, k4, pad (k-stride)/2 = 0, groups 1.
            TestCase {
                name: "convt1d_stride4_k4",
                op: OpSpec::ConvTranspose1d {
                    k: 4,
                    stride: 4,
                    dilation: 1,
                    pad: 0,
                    groups: 1,
                },
                inputs: vec![
                    t("x", [1, 8, 6], linspace(-1.0, 1.0, false)),
                    t("w", [8, 4, 4], linspace(-0.3, 0.3, false)),
                    t("bias", [4], linspace(-0.1, 0.1, false)),
                ],
            },
            // Odd kernel, padded: stride 5, k11, pad (11-5)/2 = 3.
            TestCase {
                name: "convt1d_stride5_k11_pad3",
                op: OpSpec::ConvTranspose1d {
                    k: 11,
                    stride: 5,
                    dilation: 1,
                    pad: 3,
                    groups: 1,
                },
                inputs: vec![
                    t("x", [1, 6, 5], linspace(-1.0, 1.0, true)),
                    t("w", [6, 3, 11], linspace(-0.2, 0.2, false)),
                    t("bias", [3], linspace(-0.05, 0.05, false)),
                ],
            },
            // Depthwise transposed (Hann/anti-alias UpSample1d): groups == channels.
            TestCase {
                name: "convt1d_depthwise_stride3",
                op: OpSpec::ConvTranspose1d {
                    k: 6,
                    stride: 3,
                    dilation: 1,
                    pad: 0,
                    groups: 4,
                },
                inputs: vec![
                    t("x", [1, 4, 7], linspace(-1.0, 1.0, false)),
                    t("w", [4, 1, 6], linspace(-0.25, 0.25, false)),
                    t("bias", [4], linspace(-0.05, 0.05, false)),
                ],
            },
        ]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a crate::conformance::OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_conv_transpose1d::<ConvTranspose1dF32>())
    }
}
