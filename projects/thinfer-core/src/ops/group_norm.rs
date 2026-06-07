use super::{ActDtype, WeightDtype, WgslConfig};
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
/// One WORKGROUP (256 threads) per `(b, group)` row: threads stride the
/// group's `(C/G)*H*W` elements, partial sums tree-reduce in shared memory.
/// Two passes (mean, variance) for numerical stability vs `E[X^2] - E[X]^2`.
/// Reductions and normalize math are f32 regardless of act dtype.
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
        [n_rows, 1, 1]
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

@group(0) @binding(4) var<uniform> u: U;

const WG: u32 = 256u;
var<workgroup> partial: array<f32, 256u>;

fn reduce_sum(tid: u32) -> f32 {
    workgroupBarrier();
    var stride: u32 = WG / 2u;
    while (stride > 0u) {
        if (tid < stride) {
            partial[tid] = partial[tid] + partial[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let total = partial[0];
    workgroupBarrier();
    return total;
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    if (row >= u.b * u.g) { return; }
    let tid = lid.x;
    let bi = row / u.g;
    let gi = row - bi * u.g;
    let c_per_g = u.c / u.g;
    let hw = u.h * u.w;
    let n_elems = c_per_g * hw;
    let group_base = bi * (u.c * hw) + gi * (c_per_g * hw);

    var sum: f32 = 0.0;
    for (var i: u32 = tid; i < n_elems; i = i + WG) {
        sum = sum + load_x(group_base + i);
    }
    partial[tid] = sum;
    let mean = reduce_sum(tid) / f32(n_elems);

    var sum_sq: f32 = 0.0;
    for (var i: u32 = tid; i < n_elems; i = i + WG) {
        let d = load_x(group_base + i) - mean;
        sum_sq = sum_sq + d * d;
    }
    partial[tid] = sum_sq;
    let inv = inverseSqrt(reduce_sum(tid) / f32(n_elems) + u.eps);

    for (var i: u32 = tid; i < n_elems; i = i + WG) {
        let ch = gi * c_per_g + i / hw;
        let scale = load_weight(ch) * inv;
        let offset = load_bias(ch) - mean * scale;
        store_out(group_base + i, load_x(group_base + i) * scale + offset);
    }
}
"#
    };
}

macro_rules! group_norm_acts_f32 {
    () => {
        r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
fn load_x(i: u32) -> f32 { return x[i]; }
fn store_out(i: u32, v: f32) { out[i] = v; }
"#
    };
}

// Native-f16 act storage; scalar element loads (group rows have arbitrary
// element alignment). Reductions stay f32; saturated narrow at the store.
macro_rules! group_norm_acts_f16 {
    () => {
        r#"enable f16;
@group(0) @binding(0) var<storage, read> x: array<f16>;
@group(0) @binding(3) var<storage, read_write> out: array<f16>;
fn load_x(i: u32) -> f32 { return f32(x[i]); }
fn store_out(i: u32, v: f32) { out[i] = f16(clamp(v, -65504.0, 65504.0)); }
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

const WGSL_F32: &str = concat!(
    group_norm_acts_f32!(),
    group_norm_f32_bindings!(),
    group_norm_body!()
);
const WGSL_F32_WBF16: &str = concat!(
    group_norm_acts_f32!(),
    group_norm_bf16_bindings!(),
    group_norm_body!()
);
const WGSL_F16_WBF16: &str = concat!(
    group_norm_acts_f16!(),
    group_norm_bf16_bindings!(),
    group_norm_body!()
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
        match (cfg.act_dtype, cfg.weight_dtype) {
            (ActDtype::F32, WeightDtype::F32) => WGSL_F32,
            (ActDtype::F32, WeightDtype::Bf16) => WGSL_F32_WBF16,
            (ActDtype::F16, WeightDtype::Bf16) => WGSL_F16_WBF16,
            (a, w) => unreachable!("group_norm unsupported (act={a:?}, weight={w:?})"),
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}
