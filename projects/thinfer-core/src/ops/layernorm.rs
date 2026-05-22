use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
#[cfg(feature = "conformance")]
use crate::conformance::{
    DTYPES_ACT_BF16, Dtype, OpSpec, OpTest, OpTestContext, TestCase, linspace, t,
};
use crate::ops::{ActDtype, WgslConfig};
use crate::tensor::{ComputeDtype, F32};
use crate::{act_bf16_prelude, act_f16_prelude, wgsl_with_bf16_variant};

/// `y[r,i] = (x[r,i] - mean(x[r,:])) * rsqrt(var(x[r,:]) + eps)`. No affine
/// (FinalLayer uses `elementwise_affine=False`; scale is applied outside via
/// `bcast_affine`).
///
/// Layout: 0=X, 1=Out, 2=Uniform `{n_rows, dim, eps_bits, _pad}`.
pub trait LayerNormOp {
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

pub struct LayerNormBufs<'a> {
    pub x: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_layernorm<O: LayerNormOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &LayerNormBufs<'_>,
    n_rows: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.out.binding(1),
        bufs.uniform.binding(2),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_rows))
}

wgsl_with_bf16_variant!(
    WGSL_F32,
    WGSL_F32_BF16 = r#"
struct U { n_rows: u32, dim: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= u.n_rows) { return; }
    let base = row * u.dim;
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < u.dim; i = i + 1u) {
        sum = sum + x[base + i];
    }
    let mean = sum / f32(u.dim);
    var sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < u.dim; i = i + 1u) {
        let d = x[base + i] - mean;
        sum_sq = sum_sq + d * d;
    }
    let inv = inverseSqrt(sum_sq / f32(u.dim) + u.eps);
    for (var i: u32 = 0u; i < u.dim; i = i + 1u) {
        out[base + i] = act_store((x[base + i] - mean) * inv);
    }
}
"#
);

/// Packed-bf16 activations. Requires `dim % 2 == 0`. One thread per row;
/// row reductions loop over `dim/2` words, unpacking 2 elements per word.
const WGSL_BF16_PACKED: &str = concat!(
    act_bf16_prelude!(),
    r#"
struct U { n_rows: u32, dim: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= u.n_rows) { return; }
    let dim_w = u.dim >> 1u;
    let row_base = row * dim_w;
    var sum: f32 = 0.0;
    for (var wi: u32 = 0u; wi < dim_w; wi = wi + 1u) {
        let v = unpack_bf16x2(x[row_base + wi]);
        sum = sum + v.x + v.y;
    }
    let mean = sum / f32(u.dim);
    var sum_sq: f32 = 0.0;
    for (var wi: u32 = 0u; wi < dim_w; wi = wi + 1u) {
        let v = unpack_bf16x2(x[row_base + wi]);
        let dx = v.x - mean;
        let dy = v.y - mean;
        sum_sq = sum_sq + dx * dx + dy * dy;
    }
    let inv = inverseSqrt(sum_sq / f32(u.dim) + u.eps);
    for (var wi: u32 = 0u; wi < dim_w; wi = wi + 1u) {
        let v = unpack_bf16x2(x[row_base + wi]);
        out[row_base + wi] = pack_bf16x2((v.x - mean) * inv, (v.y - mean) * inv);
    }
}
"#
);

// F16 acts, no weight. Mean + variance reductions widen to f32 for the
// same reason rmsnorm does — squared accumulations over D=3840 exceed
// f16 dynamic range. Storage halves vs the f32-acts variant.
const WGSL_F16_PACKED: &str = concat!(
    act_f16_prelude!(),
    r#"
struct U { n_rows: u32, dim: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<storage, read> x: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec2<f16>>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= u.n_rows) { return; }
    let dim_w = u.dim >> 1u;
    let row_base = row * dim_w;
    var sum: f32 = 0.0;
    for (var wi: u32 = 0u; wi < dim_w; wi = wi + 1u) {
        let v: vec2<f32> = vec2<f32>(x[row_base + wi]);
        sum = sum + v.x + v.y;
    }
    let mean = sum / f32(u.dim);
    var sum_sq: f32 = 0.0;
    for (var wi: u32 = 0u; wi < dim_w; wi = wi + 1u) {
        let v: vec2<f32> = vec2<f32>(x[row_base + wi]);
        let dx = v.x - mean;
        let dy = v.y - mean;
        sum_sq = sum_sq + dx * dx + dy * dy;
    }
    let inv = inverseSqrt(sum_sq / f32(u.dim) + u.eps);
    for (var wi: u32 = 0u; wi < dim_w; wi = wi + 1u) {
        let v: vec2<f32> = vec2<f32>(x[row_base + wi]);
        out[row_base + wi] = vec2<f16>(vec2<f32>((v.x - mean) * inv, (v.y - mean) * inv));
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
        kind: BindingKind::StorageReadWrite,
    },
    BindingLayout {
        slot: 2,
        kind: BindingKind::Uniform,
    },
];

pub struct LayerNormF32;

impl LayerNormOp for LayerNormF32 {
    const KERNEL_ID: &'static str = "layernorm.f32";
    type Dtype = F32;
    const X: &'static str = "layernorm/x";
    const DIMS: &'static str = "layernorm/dims";
    const OUTPUT: &'static str = "layernorm/out";
    fn wgsl(cfg: &WgslConfig) -> &'static str {
        match (cfg.act_dtype, cfg.bf16_quant_writes) {
            (ActDtype::F32, false) => WGSL_F32,
            (ActDtype::F32, true) => WGSL_F32_BF16,
            (ActDtype::Bf16, _) => WGSL_BF16_PACKED,
            (ActDtype::F16, _) => WGSL_F16_PACKED,
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl OpTest for LayerNormF32 {
    fn dtypes(&self) -> &'static [Dtype] {
        DTYPES_ACT_BF16
    }
    fn test_cases(&self) -> Vec<TestCase> {
        vec![TestCase {
            name: "layernorm_basic",
            op: OpSpec::Layernorm { eps: 1e-6 },
            inputs: vec![t("x", [4, 16], linspace(-2.0, 2.0, false))],
        }]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_layernorm::<LayerNormF32>())
    }
}
