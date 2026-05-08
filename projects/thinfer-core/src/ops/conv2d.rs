use super::{WeightDtype, WgslConfig};
use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
use crate::tensor::{ComputeDtype, F32};

/// `out = conv2d(x, weight) + bias` over NCHW.
///
/// Shapes (row-major):
/// - `x:      [B, Cin, Hin, Win]`
/// - `weight: [Cout, Cin, kH, kW]` (PyTorch `nn.Conv2d` layout)
/// - `bias:   [Cout]`
/// - `out:    [B, Cout, Hout, Wout]`
///
/// Geometry: `Hout = (Hin + 2*pad_h - kH) / stride_h + 1`, same for W.
///
/// Single direct-conv kernel; one thread per output element. fp32. Bias is
/// always required (callers pass a zero buffer if no bias). Dilation and
/// groups are NOT supported (no Z-Image-Turbo VAE conv uses them).
///
/// Layout: 0=X, 1=W, 2=Bias, 3=Out, 4=Uniform.
pub trait Conv2dOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const W: &'static str;
    const BIAS: &'static str;
    const DIMS: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n_out_elems: u32) -> [u32; 3] {
        super::linear_workgroups(n_out_elems, 64)
    }
}

pub struct Conv2dBufs<'a> {
    pub x: &'a BufRef,
    pub w: &'a BufRef,
    pub bias: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_conv2d<O: Conv2dOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &Conv2dBufs<'_>,
    n_out_elems: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.w.binding(1),
        bufs.bias.binding(2),
        bufs.out.binding(3),
        bufs.uniform.binding(4),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_out_elems))
}

macro_rules! conv2d_body {
    () => {
        r#"
struct U {
    b: u32, cin: u32, cout: u32, h_in: u32,
    w_in: u32, h_out: u32, w_out: u32, kh: u32,
    kw: u32, pad_h: u32, pad_w: u32, stride_h: u32,
    stride_w: u32, _pad0: u32, _pad1: u32, _pad2: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    let total = u.b * u.cout * u.h_out * u.w_out;
    if (i >= total) { return; }

    let ho_wo = u.h_out * u.w_out;
    let cout_stride = u.cout * ho_wo;
    let bi = i / cout_stride;
    let r1 = i - bi * cout_stride;
    let co = r1 / ho_wo;
    let r2 = r1 - co * ho_wo;
    let ho = r2 / u.w_out;
    let wo = r2 - ho * u.w_out;

    let h_base = i32(ho * u.stride_h) - i32(u.pad_h);
    let w_base = i32(wo * u.stride_w) - i32(u.pad_w);

    let in_chw = u.cin * u.h_in * u.w_in;
    let in_hw = u.h_in * u.w_in;
    let w_per_cout = u.cin * u.kh * u.kw;
    let w_per_cin = u.kh * u.kw;

    var acc: f32 = load_bias(co);
    for (var ci: u32 = 0u; ci < u.cin; ci = ci + 1u) {
        for (var kh: u32 = 0u; kh < u.kh; kh = kh + 1u) {
            let hi = h_base + i32(kh);
            if (hi < 0 || hi >= i32(u.h_in)) { continue; }
            for (var kw: u32 = 0u; kw < u.kw; kw = kw + 1u) {
                let wi = w_base + i32(kw);
                if (wi < 0 || wi >= i32(u.w_in)) { continue; }
                let x_idx = bi * in_chw + ci * in_hw + u32(hi) * u.w_in + u32(wi);
                let w_idx = co * w_per_cout + ci * w_per_cin + kh * u.kw + kw;
                acc = acc + x[x_idx] * load_w(w_idx);
            }
        }
    }
    out[i] = act_store(acc);
}
"#
    };
}

macro_rules! conv2d_f32_bindings {
    () => {
        r#"
@group(0) @binding(1) var<storage, read> wgt: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;

fn load_w(i: u32) -> f32 { return wgt[i]; }
fn load_bias(i: u32) -> f32 { return bias[i]; }
"#
    };
}

// bf16-weight variant: wgt and bias both packed bf16-in-u32. Element i lives
// in word i/2, low half if even, high half if odd; cast to f32 by shifting
// the bf16 bit pattern into the f32 upper-16.
macro_rules! conv2d_bf16_bindings {
    () => {
        r#"
@group(0) @binding(1) var<storage, read> wgt: array<u32>;
@group(0) @binding(2) var<storage, read> bias: array<u32>;

fn unpack_bf16(pair: u32, i: u32) -> f32 {
    let half = (pair >> ((i & 1u) * 16u)) & 0xFFFFu;
    return bitcast<f32>(half << 16u);
}
fn load_w(i: u32) -> f32 { return unpack_bf16(wgt[i >> 1u], i); }
fn load_bias(i: u32) -> f32 { return unpack_bf16(bias[i >> 1u], i); }
"#
    };
}

crate::weight_op_wgsl_no_bf16q! {
    (WGSL_F32, WGSL_F32_WBF16);
    body = conv2d_body!();
    f32_bindings = conv2d_f32_bindings!();
    bf16_bindings = conv2d_bf16_bindings!();
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

pub struct Conv2dF32;

impl Conv2dOp for Conv2dF32 {
    const KERNEL_ID: &'static str = "conv2d.f32";
    type Dtype = F32;
    const X: &'static str = "conv2d/x";
    const W: &'static str = "conv2d/w";
    const BIAS: &'static str = "conv2d/bias";
    const DIMS: &'static str = "conv2d/dims";
    const OUTPUT: &'static str = "conv2d/out";
    fn wgsl(cfg: &WgslConfig) -> &'static str {
        assert!(!cfg.bf16_quant_writes);
        match cfg.weight_dtype {
            WeightDtype::F32 => WGSL_F32,
            WeightDtype::Bf16 => WGSL_F32_WBF16,
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl crate::conformance::OpTest for Conv2dF32 {
    fn dtypes(&self) -> &'static [crate::conformance::Dtype] {
        crate::conformance::DTYPES_FP32_ONLY
    }
    fn test_cases(&self) -> Vec<crate::conformance::TestCase> {
        use crate::conformance::{OpSpec, TestCase, linspace, t};
        vec![
            // Tiny 3x3 stride=1 pad=1 - kernel correctness baseline.
            TestCase {
                name: "conv2d_3x3_pad1",
                op: OpSpec::Conv2d {
                    kh: 3,
                    kw: 3,
                    pad_h: 1,
                    pad_w: 1,
                    stride_h: 1,
                    stride_w: 1,
                },
                inputs: vec![
                    t("x", [1, 4, 6, 6], linspace(-1.0, 1.0, false)),
                    t("w", [3, 4, 3, 3], linspace(-0.5, 0.5, false)),
                    t("bias", [3], linspace(-0.25, 0.25, false)),
                ],
            },
            // 1x1 pad=0 - residual shortcut path in VAE.
            TestCase {
                name: "conv2d_1x1",
                op: OpSpec::Conv2d {
                    kh: 1,
                    kw: 1,
                    pad_h: 0,
                    pad_w: 0,
                    stride_h: 1,
                    stride_w: 1,
                },
                inputs: vec![
                    t("x", [1, 4, 4, 4], linspace(-1.0, 1.0, false)),
                    t("w", [3, 4, 1, 1], linspace(-0.5, 0.5, false)),
                    t("bias", [3], linspace(-0.25, 0.25, true)),
                ],
            },
            // Wider cin to exercise a longer accumulation loop. Kept at
            // cin=16 so 144-term fp32 summation stays under 1e-5 tol.
            TestCase {
                name: "conv2d_3x3_widec",
                op: OpSpec::Conv2d {
                    kh: 3,
                    kw: 3,
                    pad_h: 1,
                    pad_w: 1,
                    stride_h: 1,
                    stride_w: 1,
                },
                inputs: vec![
                    t("x", [1, 16, 4, 4], linspace(-1.0, 1.0, false)),
                    t("w", [8, 16, 3, 3], linspace(-0.5, 0.5, false)),
                    t("bias", [8], linspace(-0.25, 0.25, false)),
                ],
            },
        ]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a crate::conformance::OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_conv2d::<Conv2dF32>())
    }
}
