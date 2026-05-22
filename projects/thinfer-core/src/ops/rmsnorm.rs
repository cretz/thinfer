use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
#[cfg(feature = "conformance")]
use crate::conformance::{
    DTYPES_ACT_BF16, Dtype, OpSpec, OpTest, OpTestContext, TestCase, linspace, t,
};
use crate::ops::{ActDtype, WeightDtype, WgslConfig};
use crate::tensor::{ComputeDtype, F32};
use crate::{act_bf16_prelude, act_f16_prelude, wgsl_with_bf16_variant};

/// `y[r,i] = x[r,i] * rsqrt(mean(x[r,:]^2) + eps) * w[i]`.
///
/// Layout: 0=X, 1=W (per-feature gain, shape [dim]), 2=Out, 3=Uniform `{n_rows, dim, eps_bits, _pad}` (u32x4; eps is f32 bit-cast).
pub trait RmsNormOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const W: &'static str;
    const DIMS: &'static str;
    const OUTPUT: &'static str;
    const EPS: f32 = 1e-6;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n_rows: u32) -> [u32; 3] {
        [n_rows.div_ceil(64), 1, 1]
    }
}

pub struct RmsNormBufs<'a> {
    pub x: &'a BufRef,
    pub w: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_rmsnorm<O: RmsNormOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &RmsNormBufs<'_>,
    n_rows: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.w.binding(1),
        bufs.out.binding(2),
        bufs.uniform.binding(3),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_rows))
}

wgsl_with_bf16_variant!(
    WGSL_F32,
    WGSL_F32_BF16 = r#"
struct U { n_rows: u32, dim: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

fn load_w(i: u32) -> f32 { return w[i]; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= u.n_rows) { return; }
    var sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < u.dim; i = i + 1u) {
        let v = x[row * u.dim + i];
        sum_sq = sum_sq + v * v;
    }
    let inv = inverseSqrt(sum_sq / f32(u.dim) + u.eps);
    for (var i: u32 = 0u; i < u.dim; i = i + 1u) {
        out[row * u.dim + i] = act_store(x[row * u.dim + i] * inv * load_w(i));
    }
}
"#
);

wgsl_with_bf16_variant!(
    WGSL_F32_WBF16,
    WGSL_F32_BF16_WBF16 = r#"
struct U { n_rows: u32, dim: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

fn load_w(i: u32) -> f32 {
    let pair = w[i >> 1u];
    let half = (pair >> ((i & 1u) * 16u)) & 0xFFFFu;
    return bitcast<f32>(half << 16u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= u.n_rows) { return; }
    var sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < u.dim; i = i + 1u) {
        let v = x[row * u.dim + i];
        sum_sq = sum_sq + v * v;
    }
    let inv = inverseSqrt(sum_sq / f32(u.dim) + u.eps);
    for (var i: u32 = 0u; i < u.dim; i = i + 1u) {
        out[row * u.dim + i] = act_store(x[row * u.dim + i] * inv * load_w(i));
    }
}
"#
);

/// Packed-bf16 activations + packed-bf16 weights. Requires `dim % 2 == 0`.
/// One thread per row; the row-wise reduction loops over `dim/2` words,
/// unpacking 2 elements per iteration. Writes pack 2 elements per word.
const WGSL_BF16_PACKED: &str = concat!(
    act_bf16_prelude!(),
    r#"
struct U { n_rows: u32, dim: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= u.n_rows) { return; }
    let dim_w = u.dim >> 1u;
    let row_base = row * dim_w;
    var sum_sq: f32 = 0.0;
    for (var wi: u32 = 0u; wi < dim_w; wi = wi + 1u) {
        let v = unpack_bf16x2(x[row_base + wi]);
        sum_sq = sum_sq + v.x * v.x + v.y * v.y;
    }
    let inv = inverseSqrt(sum_sq / f32(u.dim) + u.eps);
    for (var wi: u32 = 0u; wi < dim_w; wi = wi + 1u) {
        let v = unpack_bf16x2(x[row_base + wi]);
        let wv = unpack_bf16x2(w[wi]);
        out[row_base + wi] = pack_bf16x2(v.x * inv * wv.x, v.y * inv * wv.y);
    }
}
"#
);

// F16 acts + bf16-packed weights. Storage halves vs f32-acts; the row
// reduction widens to f32 because squared sums over D=3840 elements
// overflow f16 and lose mantissa precision faster than the result
// tolerates. Compute `x * inv` and `* w` in f32, narrow to vec2<f16> on
// write — this is the standard f16-IO / f32-accum pattern.
const WGSL_F16_PACKED_WBF16: &str = concat!(
    act_f16_prelude!(),
    r#"
struct U { n_rows: u32, dim: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<storage, read> x: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<vec2<f16>>;
@group(0) @binding(3) var<uniform> u: U;

fn load_w_pair(wi: u32) -> vec2<f32> {
    let pair = w[wi];
    let lo = bitcast<f32>((pair & 0xFFFFu) << 16u);
    let hi = bitcast<f32>(pair & 0xFFFF0000u);
    return vec2<f32>(lo, hi);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= u.n_rows) { return; }
    let dim_w = u.dim >> 1u;
    let row_base = row * dim_w;
    var sum_sq: f32 = 0.0;
    for (var wi: u32 = 0u; wi < dim_w; wi = wi + 1u) {
        let v: vec2<f32> = vec2<f32>(x[row_base + wi]);
        sum_sq = sum_sq + v.x * v.x + v.y * v.y;
    }
    let inv = inverseSqrt(sum_sq / f32(u.dim) + u.eps);
    for (var wi: u32 = 0u; wi < dim_w; wi = wi + 1u) {
        let v: vec2<f32> = vec2<f32>(x[row_base + wi]);
        let wv = load_w_pair(wi);
        out[row_base + wi] = vec2<f16>(vec2<f32>(v.x * inv * wv.x, v.y * inv * wv.y));
    }
}
"#
);

// F16 acts + f32 weights. Only exercised when an f32 RMSNorm weight gets
// paired with the F16 path; less common than wbf16 but kept exhaustive.
const WGSL_F16_PACKED_WF32: &str = concat!(
    act_f16_prelude!(),
    r#"
struct U { n_rows: u32, dim: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<storage, read> x: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<vec2<f16>>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= u.n_rows) { return; }
    let dim_w = u.dim >> 1u;
    let row_base = row * dim_w;
    var sum_sq: f32 = 0.0;
    for (var wi: u32 = 0u; wi < dim_w; wi = wi + 1u) {
        let v: vec2<f32> = vec2<f32>(x[row_base + wi]);
        sum_sq = sum_sq + v.x * v.x + v.y * v.y;
    }
    let inv = inverseSqrt(sum_sq / f32(u.dim) + u.eps);
    for (var wi: u32 = 0u; wi < dim_w; wi = wi + 1u) {
        let v: vec2<f32> = vec2<f32>(x[row_base + wi]);
        let i0 = wi * 2u;
        out[row_base + wi] = vec2<f16>(vec2<f32>(v.x * inv * w[i0], v.y * inv * w[i0 + 1u]));
    }
}
"#
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

pub struct RmsNormF32;

impl RmsNormOp for RmsNormF32 {
    const KERNEL_ID: &'static str = "rmsnorm.f32";
    type Dtype = F32;
    const X: &'static str = "rmsnorm/x";
    const W: &'static str = "rmsnorm/w";
    const DIMS: &'static str = "rmsnorm/dims";
    const OUTPUT: &'static str = "rmsnorm/out";
    fn wgsl(cfg: &WgslConfig) -> &'static str {
        match (cfg.act_dtype, cfg.weight_dtype, cfg.bf16_quant_writes) {
            (ActDtype::F32, WeightDtype::F32, false) => WGSL_F32,
            (ActDtype::F32, WeightDtype::F32, true) => WGSL_F32_BF16,
            (ActDtype::F32, WeightDtype::Bf16, false) => WGSL_F32_WBF16,
            (ActDtype::F32, WeightDtype::Bf16, true) => WGSL_F32_BF16_WBF16,
            (ActDtype::Bf16, WeightDtype::Bf16, _) => WGSL_BF16_PACKED,
            (ActDtype::Bf16, WeightDtype::F32, _) => {
                panic!("rmsnorm: packed-bf16 acts with fp32 weights not implemented")
            }
            (ActDtype::F16, WeightDtype::Bf16, _) => WGSL_F16_PACKED_WBF16,
            (ActDtype::F16, WeightDtype::F32, _) => WGSL_F16_PACKED_WF32,
            (_, WeightDtype::Quant(_), _) => {
                unreachable!("rmsnorm does not consume quant weights")
            }
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl OpTest for RmsNormF32 {
    fn dtypes(&self) -> &'static [Dtype] {
        DTYPES_ACT_BF16
    }
    fn test_cases(&self) -> Vec<TestCase> {
        vec![TestCase {
            name: "rmsnorm_basic",
            op: OpSpec::Rmsnorm { eps: 1e-6 },
            inputs: vec![
                t("x", [4, 16], linspace(-2.0, 2.0, false)),
                t("w", [16], linspace(0.5, 1.5, false)),
            ],
        }]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_rmsnorm::<RmsNormF32>())
    }
}
