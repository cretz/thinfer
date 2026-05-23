use super::{WeightDtype, WgslConfig};
use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
use crate::tensor::{ComputeDtype, F32};

/// `y[b,c,h,w] = ((x - mean(group)) * rsqrt(var(group) + eps)) * weight[c] + bias[c]`.
///
/// Channels `[0, C)` are split into `G` groups of `C/G` contiguous channels.
/// Mean and variance are reduced over `(C/G, H, W)` for each `(b, group)`.
///
/// Shapes:
/// - `x:      [B, C, H, W]`
/// - `weight: [C]` (affine gain, always present)
/// - `bias:   [C]` (affine offset, always present)
/// - `out:    [B, C, H, W]`
///
/// One thread per `(b, group)` row. Two passes (mean, variance) for numerical
/// stability vs `E[X^2] - E[X]^2`. Slow at full VAE resolution; planned
/// follow-up is workgroup-per-row parallel reduction.
///
/// Layout: 0=X, 1=Weight, 2=Bias, 3=Out, 4=Uniform.
pub trait GroupNormOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const W: &'static str;
    const BIAS: &'static str;
    const DIMS: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n_rows: u32) -> [u32; 3] {
        [n_rows.div_ceil(64), 1, 1]
    }
}

pub struct GroupNormBufs<'a> {
    pub x: &'a BufRef,
    pub w: &'a BufRef,
    pub bias: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_group_norm<O: GroupNormOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &GroupNormBufs<'_>,
    n_rows: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.w.binding(1),
        bufs.bias.binding(2),
        bufs.out.binding(3),
        bufs.uniform.binding(4),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_rows))
}

macro_rules! group_norm_body {
    () => {
        r#"
struct U { b: u32, c: u32, g: u32, h: u32, w: u32, eps: f32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= u.b * u.g) { return; }
    let bi = row / u.g;
    let gi = row - bi * u.g;
    let c_per_g = u.c / u.g;
    let hw = u.h * u.w;
    let n_elems = c_per_g * hw;
    let group_base = bi * (u.c * hw) + gi * (c_per_g * hw);

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < n_elems; i = i + 1u) {
        sum = sum + x[group_base + i];
    }
    let mean = sum / f32(n_elems);

    var sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < n_elems; i = i + 1u) {
        let d = x[group_base + i] - mean;
        sum_sq = sum_sq + d * d;
    }
    let inv = inverseSqrt(sum_sq / f32(n_elems) + u.eps);

    for (var ic: u32 = 0u; ic < c_per_g; ic = ic + 1u) {
        let ch = gi * c_per_g + ic;
        let scale = load_weight(ch) * inv;
        let offset = load_bias(ch) - mean * scale;
        let ch_base = group_base + ic * hw;
        for (var j: u32 = 0u; j < hw; j = j + 1u) {
            out[ch_base + j] = act_store(x[ch_base + j] * scale + offset);
        }
    }
}
"#
    };
}

macro_rules! group_norm_f32_bindings {
    () => {
        r#"
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
fn load_weight(i: u32) -> f32 { return weight[i]; }
fn load_bias(i: u32) -> f32 { return bias[i]; }
"#
    };
}

macro_rules! group_norm_bf16_bindings {
    () => {
        r#"
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read> bias: array<u32>;
fn unpack_bf16(pair: u32, i: u32) -> f32 {
    let half = (pair >> ((i & 1u) * 16u)) & 0xFFFFu;
    return bitcast<f32>(half << 16u);
}
fn load_weight(i: u32) -> f32 { return unpack_bf16(weight[i >> 1u], i); }
fn load_bias(i: u32) -> f32 { return unpack_bf16(bias[i >> 1u], i); }
"#
    };
}

crate::weight_op_wgsl_no_bf16q! {
    (WGSL_F32, WGSL_F32_WBF16);
    body = group_norm_body!();
    f32_bindings = group_norm_f32_bindings!();
    bf16_bindings = group_norm_bf16_bindings!();
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

pub struct GroupNormF32;

impl GroupNormOp for GroupNormF32 {
    const KERNEL_ID: &'static str = "group_norm.f32";
    type Dtype = F32;
    const X: &'static str = "group_norm/x";
    const W: &'static str = "group_norm/w";
    const BIAS: &'static str = "group_norm/bias";
    const DIMS: &'static str = "group_norm/dims";
    const OUTPUT: &'static str = "group_norm/out";
    fn wgsl(cfg: &WgslConfig) -> &'static str {
        assert!(!cfg.bf16_quant_writes);
        match cfg.weight_dtype {
            WeightDtype::F32 => WGSL_F32,
            WeightDtype::Bf16 => WGSL_F32_WBF16,
            WeightDtype::F16 => unreachable!("group_norm does not consume f16 weights"),
            WeightDtype::Quant(_) => unreachable!("group_norm does not consume quant weights"),
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}
