use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
#[cfg(feature = "conformance")]
use crate::conformance::{
    DTYPES_ACT_BF16, Dtype, OpSpec, OpTest, OpTestContext, TestCase, linspace, t,
};
use crate::ops::{ActDtype, WgslConfig};
use crate::tensor::{ComputeDtype, F32};
use crate::{act_bf16_prelude, act_f16_prelude, wgsl_with_bf16_variant};

/// Fused scaled-dot-product attention with online softmax.
///
/// Two implementations live here:
/// - [`SdpaF32`]: one thread per output row, per-thread `array<f32, 128>`
///   accumulator. Fastest for `D <= 128` (Z-Image DiT, Qwen3). Streams keys
///   serially in a single thread.
/// - [`SdpaF32LargeD`]: one workgroup per output row, D split across threads,
///   tree-reduced dot product. Used for VAE mid-block self-attention (`D=512`)
///   where per-thread `array<f32, 512>` would spill out of registers and the
///   serial D-loop would dominate.
///
/// Shapes (all flat, row-major; BSHD so this composes with rope w/o an
/// axis swap):
/// - Q   `[B, S_q, H_q,  D]`
/// - K   `[B, S_k, H_kv, D]`     (GQA: H_kv | H_q)
/// - V   `[B, S_k, H_kv, D]`
/// - Mask `[B, S_q, S_k]`         additive bias per (query, key) pair. Causal,
///   sliding-window, prefix-LM, and pad-only are all degenerate cases.
/// - Out `[B, S_q, H_q,  D]`
///
/// `SdpaF32` requires `D <= 128`; `SdpaF32LargeD` requires `D` divisible by
/// `WG=128` and `D <= 512`. Caller provides a zero-filled mask when none.
///
/// Layout: 0=Q, 1=K, 2=V, 3=Mask, 4=Out, 5=Uniform
/// `{B, H_q, H_kv, S_q, S_k, D, scale: f32, has_mask: u32}`.
///
/// `has_mask` gates the mask read. When 0 the kernel uses 0.0 as the additive
/// bias and the `mask` binding need only be a 1-element zero scratch. This
/// avoids allocating the `[B, S_q, S_k]` zero-filled tensor for full-attention
/// callers (e.g. VAE mid-block: at S=16384 that mask is 1 GiB).
pub trait SdpaOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const Q: &'static str;
    const K: &'static str;
    const V: &'static str;
    const MASK: &'static str;
    const DIMS: &'static str;
    const OUTPUT: &'static str;
    const MAX_D: u32 = 128;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(b: u32, s_q: u32, h_q: u32) -> [u32; 3] {
        [(b * s_q * h_q).div_ceil(64), 1, 1]
    }
}

pub struct SdpaBufs<'a> {
    pub q: &'a BufRef,
    pub k: &'a BufRef,
    pub v: &'a BufRef,
    pub mask: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_sdpa<O: SdpaOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &SdpaBufs<'_>,
    b: u32,
    s_q: u32,
    h_q: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.q.binding(0),
        bufs.k.binding(1),
        bufs.v.binding(2),
        bufs.mask.binding(3),
        bufs.out.binding(4),
        bufs.uniform.binding(5),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(b, s_q, h_q))
}

wgsl_with_bf16_variant!(
    WGSL_F32,
    WGSL_F32_BF16 = r#"
struct U {
    b: u32, h_q: u32, h_kv: u32, s_q: u32,
    s_k: u32, d: u32, scale: f32, has_mask: u32,
};

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> u: U;

const MAX_D: u32 = 128u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = u.b * u.s_q * u.h_q;
    let idx = gid.x;
    if (idx >= total) { return; }
    let hq  = idx % u.h_q;
    let tmp = idx / u.h_q;
    let sq  = tmp % u.s_q;
    let bb  = tmp / u.s_q;
    let hkv = (hq * u.h_kv) / u.h_q;

    let q_off    = ((bb * u.s_q + sq) * u.h_q + hq) * u.d;
    let kv_b0    = (bb * u.s_k * u.h_kv + hkv) * u.d;
    let kv_step  = u.h_kv * u.d;
    let mask_base = (bb * u.s_q + sq) * u.s_k;

    var o: array<f32, 128>;
    for (var d = 0u; d < u.d; d = d + 1u) { o[d] = 0.0; }
    var m: f32 = bitcast<f32>(0xff800000u); // -inf
    var l: f32 = 0.0;

    for (var j = 0u; j < u.s_k; j = j + 1u) {
        let kj = kv_b0 + j * kv_step;
        var dot: f32 = 0.0;
        for (var d = 0u; d < u.d; d = d + 1u) {
            dot = dot + q[q_off + d] * k[kj + d];
        }
        let bias = select(0.0, mask[mask_base + j], u.has_mask != 0u);
        let s_j  = dot * u.scale + bias;
        let m_new = max(m, s_j);
        let alpha = exp(m - m_new);
        let p_j  = exp(s_j - m_new);
        for (var d = 0u; d < u.d; d = d + 1u) {
            o[d] = o[d] * alpha + p_j * v[kj + d];
        }
        l = l * alpha + p_j;
        m = m_new;
    }

    let inv_l = 1.0 / l;
    for (var d = 0u; d < u.d; d = d + 1u) {
        out[q_off + d] = act_store(o[d] * inv_l);
    }
}
"#
);

/// Packed-bf16 small-D sdpa. Q/K/V/Mask/Out all `array<u32>` (bf16 pairs).
/// Compute (dot products, softmax, accumulators) stays fp32. Requires `D`
/// even (always true: 8/128 in Z-Image+Qwen3) and `s_k` even when
/// `has_mask != 0` (so each row's bf16 mask aligns to packed words).
const WGSL_BF16_PACKED: &str = concat!(
    act_bf16_prelude!(),
    r#"
struct U {
    b: u32, h_q: u32, h_kv: u32, s_q: u32,
    s_k: u32, d: u32, scale: f32, has_mask: u32,
};

@group(0) @binding(0) var<storage, read> q: array<u32>;
@group(0) @binding(1) var<storage, read> k: array<u32>;
@group(0) @binding(2) var<storage, read> v: array<u32>;
@group(0) @binding(3) var<storage, read> mask: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<u32>;
@group(0) @binding(5) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = u.b * u.s_q * u.h_q;
    let idx = gid.x;
    if (idx >= total) { return; }
    let hq  = idx % u.h_q;
    let tmp = idx / u.h_q;
    let sq  = tmp % u.s_q;
    let bb  = tmp / u.s_q;
    let hkv = (hq * u.h_kv) / u.h_q;

    let d_w        = u.d >> 1u;
    let q_w_off    = (((bb * u.s_q + sq) * u.h_q + hq) * u.d) >> 1u;
    let kv_b0_w    = ((bb * u.s_k * u.h_kv + hkv) * u.d) >> 1u;
    let kv_step_w  = (u.h_kv * u.d) >> 1u;
    let mask_w_base = ((bb * u.s_q + sq) * u.s_k) >> 1u;

    var o: array<f32, 128>;
    for (var d = 0u; d < u.d; d = d + 1u) { o[d] = 0.0; }
    var m: f32 = bitcast<f32>(0xff800000u);
    var l: f32 = 0.0;

    for (var j = 0u; j < u.s_k; j = j + 1u) {
        let kj_w = kv_b0_w + j * kv_step_w;
        var dot: f32 = 0.0;
        for (var dw = 0u; dw < d_w; dw = dw + 1u) {
            let qv = unpack_bf16x2(q[q_w_off + dw]);
            let kv = unpack_bf16x2(k[kj_w + dw]);
            dot = dot + qv.x * kv.x + qv.y * kv.y;
        }
        var bias: f32 = 0.0;
        if (u.has_mask != 0u) {
            let mw = unpack_bf16x2(mask[mask_w_base + (j >> 1u)]);
            bias = select(mw.x, mw.y, (j & 1u) == 1u);
        }
        let s_j  = dot * u.scale + bias;
        let m_new = max(m, s_j);
        let alpha = exp(m - m_new);
        let p_j  = exp(s_j - m_new);
        for (var dw = 0u; dw < d_w; dw = dw + 1u) {
            let vv = unpack_bf16x2(v[kj_w + dw]);
            o[dw * 2u]      = o[dw * 2u]      * alpha + p_j * vv.x;
            o[dw * 2u + 1u] = o[dw * 2u + 1u] * alpha + p_j * vv.y;
        }
        l = l * alpha + p_j;
        m = m_new;
    }

    let inv_l = 1.0 / l;
    for (var dw = 0u; dw < d_w; dw = dw + 1u) {
        out[q_w_off + dw] = pack_bf16x2(o[dw * 2u] * inv_l, o[dw * 2u + 1u] * inv_l);
    }
}
"#
);

// F16 small-D sdpa. Q/K/V/Mask/Out all `array<vec2<f16>>`. Dot products,
// softmax (m/l/p/alpha), and per-key V-accumulation stay f32 — softmax
// `exp(s_j)` saturates f16 once scores exceed ~11.1, and the online
// renormalization mixes magnitudes across keys in a way that loses bits
// in f16. Same numerical contract as bf16-packed.
const WGSL_F16_PACKED: &str = concat!(
    act_f16_prelude!(),
    r#"
struct U {
    b: u32, h_q: u32, h_kv: u32, s_q: u32,
    s_k: u32, d: u32, scale: f32, has_mask: u32,
};

@group(0) @binding(0) var<storage, read> q: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read> k: array<vec2<f16>>;
@group(0) @binding(2) var<storage, read> v: array<vec2<f16>>;
@group(0) @binding(3) var<storage, read> mask: array<vec2<f16>>;
@group(0) @binding(4) var<storage, read_write> out: array<vec2<f16>>;
@group(0) @binding(5) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = u.b * u.s_q * u.h_q;
    let idx = gid.x;
    if (idx >= total) { return; }
    let hq  = idx % u.h_q;
    let tmp = idx / u.h_q;
    let sq  = tmp % u.s_q;
    let bb  = tmp / u.s_q;
    let hkv = (hq * u.h_kv) / u.h_q;

    let d_w        = u.d >> 1u;
    let q_w_off    = (((bb * u.s_q + sq) * u.h_q + hq) * u.d) >> 1u;
    let kv_b0_w    = ((bb * u.s_k * u.h_kv + hkv) * u.d) >> 1u;
    let kv_step_w  = (u.h_kv * u.d) >> 1u;
    let mask_w_base = ((bb * u.s_q + sq) * u.s_k) >> 1u;

    var o: array<f32, 128>;
    for (var d = 0u; d < u.d; d = d + 1u) { o[d] = 0.0; }
    var m: f32 = bitcast<f32>(0xff800000u);
    var l: f32 = 0.0;

    for (var j = 0u; j < u.s_k; j = j + 1u) {
        let kj_w = kv_b0_w + j * kv_step_w;
        var dot: f32 = 0.0;
        for (var dw = 0u; dw < d_w; dw = dw + 1u) {
            let qv: vec2<f32> = vec2<f32>(q[q_w_off + dw]);
            let kv: vec2<f32> = vec2<f32>(k[kj_w + dw]);
            dot = dot + qv.x * kv.x + qv.y * kv.y;
        }
        var bias: f32 = 0.0;
        if (u.has_mask != 0u) {
            let mw: vec2<f32> = vec2<f32>(mask[mask_w_base + (j >> 1u)]);
            bias = select(mw.x, mw.y, (j & 1u) == 1u);
        }
        let s_j  = dot * u.scale + bias;
        let m_new = max(m, s_j);
        let alpha = exp(m - m_new);
        let p_j  = exp(s_j - m_new);
        for (var dw = 0u; dw < d_w; dw = dw + 1u) {
            let vv: vec2<f32> = vec2<f32>(v[kj_w + dw]);
            o[dw * 2u]      = o[dw * 2u]      * alpha + p_j * vv.x;
            o[dw * 2u + 1u] = o[dw * 2u + 1u] * alpha + p_j * vv.y;
        }
        l = l * alpha + p_j;
        m = m_new;
    }

    let inv_l = 1.0 / l;
    for (var dw = 0u; dw < d_w; dw = dw + 1u) {
        out[q_w_off + dw] = vec2<f16>(vec2<f32>(o[dw * 2u] * inv_l, o[dw * 2u + 1u] * inv_l));
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
        kind: BindingKind::StorageRead,
    },
    BindingLayout {
        slot: 3,
        kind: BindingKind::StorageRead,
    },
    BindingLayout {
        slot: 4,
        kind: BindingKind::StorageReadWrite,
    },
    BindingLayout {
        slot: 5,
        kind: BindingKind::Uniform,
    },
];

pub struct SdpaF32;

impl SdpaOp for SdpaF32 {
    const KERNEL_ID: &'static str = "sdpa.f32";
    type Dtype = F32;
    const Q: &'static str = "sdpa/q";
    const K: &'static str = "sdpa/k";
    const V: &'static str = "sdpa/v";
    const MASK: &'static str = "sdpa/mask";
    const DIMS: &'static str = "sdpa/dims";
    const OUTPUT: &'static str = "sdpa/out";
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

// ---------------------------------------------------------------------------
// Large-D variant: workgroup-per-row with D tiled across threads.
// ---------------------------------------------------------------------------
//
// Layout is identical to `SdpaF32` (same bindings, same uniform). Differences:
// - dispatch grid is `[B*S_q*H_q, 1, 1]` workgroups (1 row per workgroup),
//   workgroup_size = `WG` (128).
// - Each thread owns `D/WG` elements of `o` in registers (max 4 for D=512).
// - Dot products parallel-reduce across the workgroup; m/l/p_j/alpha scalars
//   are broadcast via `var<workgroup>` after a single-thread softmax update.
//
// Constraints: `D % WG == 0`, `D <= WG * MAX_LOCAL_D` (== 512 for WG=128,
// MAX_LOCAL_D=4). Asserted at dispatch time.

wgsl_with_bf16_variant!(
    WGSL_LARGE_D_F32,
    WGSL_LARGE_D_F32_BF16 = r#"
struct U {
    b: u32, h_q: u32, h_kv: u32, s_q: u32,
    s_k: u32, d: u32, scale: f32, has_mask: u32,
};

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> u: U;

const WG: u32 = 128u;
const MAX_LOCAL_D: u32 = 4u; // supports D up to WG*MAX_LOCAL_D = 512.

var<workgroup> shared_red: array<f32, 128>;
// 0=m, 1=l, 2=p_j, 3=alpha. Written by thread 0, read by all threads.
var<workgroup> shared_scalar: array<f32, 4>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) ng: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let total = u.b * u.s_q * u.h_q;
    let row = wgid.y * ng.x + wgid.x;
    if (row >= total) { return; }
    let t = lid.x;

    let hq  = row % u.h_q;
    let tmp = row / u.h_q;
    let sq  = tmp % u.s_q;
    let bb  = tmp / u.s_q;
    let hkv = (hq * u.h_kv) / u.h_q;

    let q_off    = ((bb * u.s_q + sq) * u.h_q + hq) * u.d;
    let kv_b0    = (bb * u.s_k * u.h_kv + hkv) * u.d;
    let kv_step  = u.h_kv * u.d;
    let mask_base = (bb * u.s_q + sq) * u.s_k;

    // d_per is D/WG; caller guarantees divisibility.
    let d_per = u.d / WG;
    let d_off = t * d_per;

    // Load this thread's slice of q into registers.
    var q_local: array<f32, 4>;
    for (var i = 0u; i < d_per; i = i + 1u) {
        q_local[i] = q[q_off + d_off + i];
    }
    var o_local: array<f32, 4>;
    for (var i = 0u; i < d_per; i = i + 1u) {
        o_local[i] = 0.0;
    }

    if (t == 0u) {
        shared_scalar[0] = bitcast<f32>(0xff800000u); // m = -inf
        shared_scalar[1] = 0.0;                       // l
    }
    workgroupBarrier();

    for (var j = 0u; j < u.s_k; j = j + 1u) {
        let kj = kv_b0 + j * kv_step;

        // Partial dot over this thread's D slice.
        var partial: f32 = 0.0;
        for (var i = 0u; i < d_per; i = i + 1u) {
            partial = partial + q_local[i] * k[kj + d_off + i];
        }
        shared_red[t] = partial;
        workgroupBarrier();

        // Tree reduce. WG = 128 = 2^7, fully unrolled by the compiler.
        var stride: u32 = WG / 2u;
        loop {
            if (stride == 0u) { break; }
            if (t < stride) {
                shared_red[t] = shared_red[t] + shared_red[t + stride];
            }
            workgroupBarrier();
            stride = stride / 2u;
        }

        // Single-thread online-softmax update; broadcast p_j, alpha.
        if (t == 0u) {
            let dot   = shared_red[0];
            let bias  = select(0.0, mask[mask_base + j], u.has_mask != 0u);
            let s_j   = dot * u.scale + bias;
            let m_cur = shared_scalar[0];
            let l_cur = shared_scalar[1];
            let m_new = max(m_cur, s_j);
            let alpha = exp(m_cur - m_new);
            let p_j   = exp(s_j - m_new);
            shared_scalar[0] = m_new;
            shared_scalar[1] = l_cur * alpha + p_j;
            shared_scalar[2] = p_j;
            shared_scalar[3] = alpha;
        }
        workgroupBarrier();

        let alpha = shared_scalar[3];
        let p_j   = shared_scalar[2];
        for (var i = 0u; i < d_per; i = i + 1u) {
            o_local[i] = o_local[i] * alpha + p_j * v[kj + d_off + i];
        }
    }

    let inv_l = 1.0 / shared_scalar[1];
    for (var i = 0u; i < d_per; i = i + 1u) {
        out[q_off + d_off + i] = act_store(o_local[i] * inv_l);
    }
}
"#
);

// F16 large-D sdpa. Used only if a future F16-acts pipeline drives the
// VAE mid-block self-attention (currently VAE stays on the bf16/f32 path,
// so this is unexercised — but keeping the match arms exhaustive avoids
// silent storage-type mismatches if VAE ever opts in).
const WGSL_LARGE_D_F16: &str = concat!(
    act_f16_prelude!(),
    r#"
struct U {
    b: u32, h_q: u32, h_kv: u32, s_q: u32,
    s_k: u32, d: u32, scale: f32, has_mask: u32,
};

@group(0) @binding(0) var<storage, read> q: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read> k: array<vec2<f16>>;
@group(0) @binding(2) var<storage, read> v: array<vec2<f16>>;
@group(0) @binding(3) var<storage, read> mask: array<vec2<f16>>;
@group(0) @binding(4) var<storage, read_write> out: array<vec2<f16>>;
@group(0) @binding(5) var<uniform> u: U;

const WG: u32 = 128u;
const MAX_LOCAL_D: u32 = 4u;

var<workgroup> shared_red: array<f32, 128>;
var<workgroup> shared_scalar: array<f32, 4>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) ng: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let total = u.b * u.s_q * u.h_q;
    let row = wgid.y * ng.x + wgid.x;
    if (row >= total) { return; }
    let t = lid.x;

    let hq  = row % u.h_q;
    let tmp = row / u.h_q;
    let sq  = tmp % u.s_q;
    let bb  = tmp / u.s_q;
    let hkv = (hq * u.h_kv) / u.h_q;

    let d_w        = u.d >> 1u;
    let q_w_off    = (((bb * u.s_q + sq) * u.h_q + hq) * u.d) >> 1u;
    let kv_b0_w    = ((bb * u.s_k * u.h_kv + hkv) * u.d) >> 1u;
    let kv_step_w  = (u.h_kv * u.d) >> 1u;
    let mask_w_base = ((bb * u.s_q + sq) * u.s_k) >> 1u;

    // d_per (in elements) must be even since D is even and WG divides D.
    // d_per_w (in words) = d_per / 2.
    let d_per = u.d / WG;
    let d_per_w = d_per >> 1u;
    let d_off_w = t * d_per_w;

    var q_local: array<f32, 4>;
    for (var i = 0u; i < d_per_w; i = i + 1u) {
        let qv: vec2<f32> = vec2<f32>(q[q_w_off + d_off_w + i]);
        q_local[i * 2u]      = qv.x;
        q_local[i * 2u + 1u] = qv.y;
    }
    var o_local: array<f32, 4>;
    for (var i = 0u; i < d_per; i = i + 1u) {
        o_local[i] = 0.0;
    }

    if (t == 0u) {
        shared_scalar[0] = bitcast<f32>(0xff800000u);
        shared_scalar[1] = 0.0;
    }
    workgroupBarrier();

    for (var j = 0u; j < u.s_k; j = j + 1u) {
        let kj_w = kv_b0_w + j * kv_step_w;
        var partial: f32 = 0.0;
        for (var i = 0u; i < d_per_w; i = i + 1u) {
            let kv: vec2<f32> = vec2<f32>(k[kj_w + d_off_w + i]);
            partial = partial + q_local[i * 2u] * kv.x + q_local[i * 2u + 1u] * kv.y;
        }
        shared_red[t] = partial;
        workgroupBarrier();

        var stride: u32 = WG / 2u;
        loop {
            if (stride == 0u) { break; }
            if (t < stride) {
                shared_red[t] = shared_red[t] + shared_red[t + stride];
            }
            workgroupBarrier();
            stride = stride / 2u;
        }

        if (t == 0u) {
            let dot   = shared_red[0];
            var bias: f32 = 0.0;
            if (u.has_mask != 0u) {
                let mw: vec2<f32> = vec2<f32>(mask[mask_w_base + (j >> 1u)]);
                bias = select(mw.x, mw.y, (j & 1u) == 1u);
            }
            let s_j   = dot * u.scale + bias;
            let m_cur = shared_scalar[0];
            let l_cur = shared_scalar[1];
            let m_new = max(m_cur, s_j);
            let alpha = exp(m_cur - m_new);
            let p_j   = exp(s_j - m_new);
            shared_scalar[0] = m_new;
            shared_scalar[1] = l_cur * alpha + p_j;
            shared_scalar[2] = p_j;
            shared_scalar[3] = alpha;
        }
        workgroupBarrier();

        let alpha = shared_scalar[3];
        let p_j   = shared_scalar[2];
        for (var i = 0u; i < d_per_w; i = i + 1u) {
            let vv: vec2<f32> = vec2<f32>(v[kj_w + d_off_w + i]);
            o_local[i * 2u]      = o_local[i * 2u]      * alpha + p_j * vv.x;
            o_local[i * 2u + 1u] = o_local[i * 2u + 1u] * alpha + p_j * vv.y;
        }
    }

    let inv_l = 1.0 / shared_scalar[1];
    for (var i = 0u; i < d_per_w; i = i + 1u) {
        out[q_w_off + d_off_w + i] = vec2<f16>(vec2<f32>(
            o_local[i * 2u] * inv_l, o_local[i * 2u + 1u] * inv_l
        ));
    }
}
"#
);

pub struct SdpaF32LargeD;

impl SdpaOp for SdpaF32LargeD {
    const KERNEL_ID: &'static str = "sdpa_large_d.f32";
    type Dtype = F32;
    const Q: &'static str = "sdpa/q";
    const K: &'static str = "sdpa/k";
    const V: &'static str = "sdpa/v";
    const MASK: &'static str = "sdpa/mask";
    const DIMS: &'static str = "sdpa/dims";
    const OUTPUT: &'static str = "sdpa/out";
    const MAX_D: u32 = 512;

    fn wgsl(cfg: &WgslConfig) -> &'static str {
        match cfg.act_dtype {
            ActDtype::F32 if cfg.bf16_quant_writes => WGSL_LARGE_D_F32_BF16,
            ActDtype::F32 => WGSL_LARGE_D_F32,
            // VAE currently uses F32 acts; the F16 large-D path exists so
            // future opt-in (e.g. f16 VAE mid-block) gets a real kernel
            // instead of a silent storage-dtype mismatch.
            ActDtype::F16 => WGSL_LARGE_D_F16,
            ActDtype::Bf16 => {
                panic!("sdpa_large_d: bf16-packed acts variant not implemented (VAE is f32)")
            }
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
    fn workgroups(b: u32, s_q: u32, h_q: u32) -> [u32; 3] {
        // One workgroup per query row (kernel uses `wgid.x + wgid.y * ng.x`
        // to recover the linear row index, so values > 65535 spill to Y).
        super::linear_workgroups(b * s_q * h_q, 1)
    }
}

#[cfg(feature = "conformance")]
impl OpTest for SdpaF32LargeD {
    fn test_cases(&self) -> Vec<TestCase> {
        // B=1, S_q=4, S_k=4, H_q=1, H_kv=1, D=128 (smallest D matching the
        // WG=128 / d_per>=1 constraint). Reuses the same scale convention.
        let scale = 1.0_f32 / 128.0_f32.sqrt();
        vec![
            TestCase {
                name: "sdpa_large_d_d128",
                op: OpSpec::Sdpa { scale },
                inputs: vec![
                    t("q", [1, 4, 1, 128], linspace(-1.0, 1.0, false)),
                    t("k", [1, 4, 1, 128], linspace(-0.5, 1.5, false)),
                    t("v", [1, 4, 1, 128], linspace(-1.25, 0.75, true)),
                    t("mask", [1, 4, 4], linspace(0.0, 0.0, false)),
                ],
            },
            TestCase {
                name: "sdpa_large_d_d512",
                op: OpSpec::Sdpa {
                    scale: 1.0_f32 / 512.0_f32.sqrt(),
                },
                inputs: vec![
                    t("q", [1, 2, 1, 512], linspace(-1.0, 1.0, false)),
                    t("k", [1, 2, 1, 512], linspace(-0.5, 1.5, false)),
                    t("v", [1, 2, 1, 512], linspace(-1.25, 0.75, true)),
                    t("mask", [1, 2, 2], linspace(0.0, 0.0, false)),
                ],
            },
        ]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_sdpa::<SdpaF32LargeD>())
    }
}

#[cfg(feature = "conformance")]
impl OpTest for SdpaF32 {
    fn dtypes(&self) -> &'static [Dtype] {
        DTYPES_ACT_BF16
    }
    fn test_cases(&self) -> Vec<TestCase> {
        // B=1, S_q=4, S_k=4, H_q=2, H_kv=2, D=8. scale = 1/sqrt(D).
        let scale = 1.0_f32 / 8.0_f32.sqrt();
        vec![TestCase {
            name: "sdpa_basic",
            op: OpSpec::Sdpa { scale },
            inputs: vec![
                t("q", [1, 4, 2, 8], linspace(-1.0, 1.0, false)),
                t("k", [1, 4, 2, 8], linspace(-0.5, 1.5, false)),
                t("v", [1, 4, 2, 8], linspace(-1.25, 0.75, true)),
                t("mask", [1, 4, 4], linspace(0.0, 0.0, false)),
            ],
        }]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_sdpa::<SdpaF32>())
    }
}
