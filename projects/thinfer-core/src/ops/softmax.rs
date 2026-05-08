use super::WgslConfig;
use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
#[cfg(feature = "conformance")]
use crate::conformance::{
    DTYPES_FP32_ONLY, Dtype, OpSpec, OpTest, OpTestContext, TestCase, linspace, t,
};
use crate::tensor::{ComputeDtype, F32};

/// `y[r,i] = exp(x[r,i] - max_j x[r,j]) / sum_j exp(x[r,j] - max_j x[r,j])`.
///
/// Last-axis softmax. Logit scale (e.g. 1/sqrt(d_k) for attention) is applied
/// externally before this op.
///
/// Layout: 0=X, 1=Out, 2=Uniform `{n_rows, dim, _pad, _pad}`.
pub trait SoftmaxOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const DIMS: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n_rows: u32) -> [u32; 3] {
        [n_rows.div_ceil(64), 1, 1]
    }
}

pub struct SoftmaxBufs<'a> {
    pub x: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_softmax<O: SoftmaxOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &SoftmaxBufs<'_>,
    n_rows: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.out.binding(1),
        bufs.uniform.binding(2),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_rows))
}

const WGSL: &str = r#"
struct U { n_rows: u32, dim: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= u.n_rows) { return; }
    let base = row * u.dim;

    var m: f32 = x[base];
    for (var i: u32 = 1u; i < u.dim; i = i + 1u) {
        m = max(m, x[base + i]);
    }
    var s: f32 = 0.0;
    for (var i: u32 = 0u; i < u.dim; i = i + 1u) {
        let e = exp(x[base + i] - m);
        out[base + i] = e;
        s = s + e;
    }
    let inv = 1.0 / s;
    for (var i: u32 = 0u; i < u.dim; i = i + 1u) {
        out[base + i] = out[base + i] * inv;
    }
}
"#;

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

pub struct SoftmaxF32;

impl SoftmaxOp for SoftmaxF32 {
    const KERNEL_ID: &'static str = "softmax.f32";
    type Dtype = F32;
    const X: &'static str = "softmax/x";
    const DIMS: &'static str = "softmax/dims";
    const OUTPUT: &'static str = "softmax/out";
    fn wgsl(cfg: &WgslConfig) -> &'static str {
        assert!(!cfg.bf16_quant_writes);
        WGSL
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl OpTest for SoftmaxF32 {
    fn dtypes(&self) -> &'static [Dtype] {
        DTYPES_FP32_ONLY
    }
    fn test_cases(&self) -> Vec<TestCase> {
        vec![TestCase {
            name: "softmax_basic",
            op: OpSpec::Softmax,
            inputs: vec![t("x", [4, 16], linspace(-2.0, 2.0, false))],
        }]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_softmax::<SoftmaxF32>())
    }
}
