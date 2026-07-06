use crate::backend::{BindingKind, BindingLayout};
use crate::ops::rmsnorm::RmsNormOp;
use crate::ops::{ActDtype, WeightDtype, WgslConfig};
use crate::tensor::F32;
use crate::{act_bf16_prelude, wgsl_with_bf16_variant};

// Gemma-style RMSNorm: `y[r,i] = x[r,i] * rsqrt(mean(x^2)+eps) * (1 + w[i])`.
// Krea 2's `KreaRMSNorm` bakes `(scale + 1)` (`krea2.hpp`), and its scale
// tensors ship F32 while the DiT runs bf16 acts -- a combo the base `RmsNormF32`
// panics on. This variant folds both: it reads F32 weights under bf16-packed (or
// f32) acts and applies the `1 +`. Same binding layout / uniform as `RmsNormF32`,
// so it dispatches through `scope.rmsnorm` with a krea-owned pipeline.

wgsl_with_bf16_variant!(
    WGSL_F32,
    WGSL_F32_BF16 = r#"
struct U { n_rows: u32, dim: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

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
        out[row * u.dim + i] = act_store(x[row * u.dim + i] * inv * (1.0 + w[i]));
    }
}
"#
);

// Packed-bf16 acts + F32 weights (the Krea DiT path). Reduction + scale in f32,
// pack 2 elements per word on write. Requires `dim % 2 == 0`.
const WGSL_BF16_PACKED_WF32: &str = concat!(
    act_bf16_prelude!(),
    r#"
struct U { n_rows: u32, dim: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
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
        let i0 = wi * 2u;
        out[row_base + wi] = pack_bf16x2(
            v.x * inv * (1.0 + w[i0]),
            v.y * inv * (1.0 + w[i0 + 1u]),
        );
    }
}
"#
);

// Packed-bf16 acts + packed-bf16 weights. Passthrough F32 scales narrow to bf16
// on GPU upload (residency), so this is the actual Krea DiT path. Reduction +
// scale in f32; `(1 + w)` applied per element.
const WGSL_BF16_PACKED_WBF16: &str = concat!(
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
        out[row_base + wi] = pack_bf16x2(
            v.x * inv * (1.0 + wv.x),
            v.y * inv * (1.0 + wv.y),
        );
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

pub struct RmsNormGemmaF32;

impl RmsNormOp for RmsNormGemmaF32 {
    const KERNEL_ID: &'static str = "rmsnorm_gemma.f32";
    type Dtype = F32;
    const X: &'static str = "rmsnorm_gemma/x";
    const W: &'static str = "rmsnorm_gemma/w";
    const DIMS: &'static str = "rmsnorm_gemma/dims";
    const OUTPUT: &'static str = "rmsnorm_gemma/out";
    fn wgsl(cfg: &WgslConfig) -> &'static str {
        match (cfg.act_dtype, cfg.weight_dtype, cfg.bf16_quant_writes) {
            (ActDtype::F32, WeightDtype::F32, false) => WGSL_F32,
            (ActDtype::F32, WeightDtype::F32, true) => WGSL_F32_BF16,
            (ActDtype::Bf16, WeightDtype::F32, _) => WGSL_BF16_PACKED_WF32,
            (ActDtype::Bf16, WeightDtype::Bf16, _) => WGSL_BF16_PACKED_WBF16,
            other => panic!(
                "rmsnorm_gemma: unsupported dtype combo {other:?} (Krea uses F32 scales under f32/bf16 acts)"
            ),
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}
