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
///   Per-head mode (`has_mask == 2`) reads `[B, H_q, S_q, S_k]` instead - one
///   bias plane per query head, e.g. T5/umT5 relative-position bias (built by
///   `relpos_bias`).
/// - Out `[B, S_q, H_q,  D]`
///
/// `SdpaF32` requires `D <= 128`; `SdpaF32LargeD` requires `D` divisible by
/// `WG=128` and `D <= 1024` (e.g. the Hunyuan VAE mid-attn single head, D=1024).
/// Caller provides a zero-filled mask when none.
///
/// Layout: 0=Q, 1=K, 2=V, 3=Mask, 4=Out, 5=Uniform
/// `{B, H_q, H_kv, S_q, S_k, D, scale: f32, has_mask: u32}`.
///
/// `has_mask` is a mode, not a bool: 0 = no mask (kernel uses 0.0 bias; the
/// `mask` binding need only be a 1-element scratch - avoids the `[B, S_q, S_k]`
/// zero tensor for full-attention callers like VAE mid-block, 1 GiB at
/// S=16384); 1 = shared `[B, S_q, S_k]` mask; 2 = per-head `[B, H_q, S_q, S_k]`
/// mask. Modes 1 and 2 differ only in the row index (`+ hq` for per-head); the
/// read itself is gated by `has_mask != 0u` in every variant.
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
        // FlashAttention tiling: one workgroup per (b, hq, q_tile),
        // BR=64 Q rows per workgroup. Grid [ceil(S_q/64), H_q, B].
        // Both H_q and B are small (≤ 16 / 1), so the 65535/dim cap is
        // hit only by the x-axis: 65535 * 64 = 4.2M Q rows.
        [s_q.div_ceil(64), h_q, b]
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

// FlashAttention-style tiled sdpa. One workgroup per (b, hq, q_tile) where
// q_tile groups BR=64 consecutive Q rows. Each thread owns one Q row in
// registers; K/V are streamed in BC=32-row tiles into workgroup-shared
// memory and reused by all BR threads. Online softmax keeps (m, l, O) in
// per-thread registers across K-tile iterations. One barrier per K-tile
// instead of zero-but-thread-serial — global K/V traffic drops by BR=64x.
//
// Workgroup grid: [ceil(S_q/64), H_q, B] (set by `workgroups()` below).
// Shared storage: BC*MAX_D * (f32 K + f32 V) = 32*128*4*2 = 32 KiB.
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

const BR: u32 = 64u;
const BC: u32 = 32u;
const WG: u32 = 64u;
const MAX_D: u32 = 128u;

var<workgroup> k_tile: array<f32, 4096>; // BC * MAX_D
var<workgroup> v_tile: array<f32, 4096>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let t  = lid.x;
    let qt = wgid.x;
    let hq = wgid.y;
    let bb = wgid.z;
    let sq = qt * BR + t;
    let valid = sq < u.s_q;

    let hkv = (hq * u.h_kv) / u.h_q;
    let q_off    = ((bb * u.s_q + sq) * u.h_q + hq) * u.d;
    let kv_b0    = (bb * u.s_k * u.h_kv + hkv) * u.d;
    let kv_step  = u.h_kv * u.d;
    // has_mask==2: per-head mask [B, Hq, Sq, Sk]; else shared [B, Sq, Sk].
    let mask_row = select(bb * u.s_q + sq, (bb * u.h_q + hq) * u.s_q + sq, u.has_mask == 2u);
    let mask_base = mask_row * u.s_k;

    var q_local: array<f32, 128>;
    for (var d = 0u; d < u.d; d = d + 1u) { q_local[d] = 0.0; }
    if (valid) {
        for (var d = 0u; d < u.d; d = d + 1u) {
            q_local[d] = q[q_off + d];
        }
    }

    var o_local: array<f32, 128>;
    for (var d = 0u; d < u.d; d = d + 1u) { o_local[d] = 0.0; }
    // Large-negative finite f32, not -inf: Tint (browser WebGPU) rejects
    // const-exprs evaluating to -inf (bitcast<f32>(0xff800000u)) and ALSO
    // out-of-range decimals like -3.4028235e38 (Rust's f32::MIN print is
    // a hair beyond max-finite as an exact decimal); naga accepts both.
    // Equivalent as a running-max init.
    var m: f32 = -3.4e38;
    var l: f32 = 0.0;

    let n_tiles = (u.s_k + BC - 1u) / BC;
    let elems_per_tile = BC * u.d;
    let per_thread = elems_per_tile / WG;

    for (var kt = 0u; kt < n_tiles; kt = kt + 1u) {
        let kc_base = kt * BC;

        for (var i = 0u; i < per_thread; i = i + 1u) {
            let idx = i * WG + t;
            let kc  = idx / u.d;
            let dd  = idx % u.d;
            let key_global = kc_base + kc;
            var kv: f32 = 0.0;
            var vv: f32 = 0.0;
            if (key_global < u.s_k) {
                let base = kv_b0 + key_global * kv_step + dd;
                kv = k[base];
                vv = v[base];
            }
            k_tile[idx] = kv;
            v_tile[idx] = vv;
        }
        workgroupBarrier();

        if (valid) {
            for (var kc = 0u; kc < BC; kc = kc + 1u) {
                let key_global = kc_base + kc;
                if (key_global < u.s_k) {
                    var dot: f32 = 0.0;
                    for (var dd = 0u; dd < u.d; dd = dd + 1u) {
                        dot = dot + q_local[dd] * k_tile[kc * u.d + dd];
                    }
                    let bias = select(0.0, mask[mask_base + key_global], u.has_mask != 0u);
                    let s_j  = dot * u.scale + bias;
                    let m_new = max(m, s_j);
                    let alpha = exp(m - m_new);
                    let p_j   = exp(s_j - m_new);
                    for (var dd = 0u; dd < u.d; dd = dd + 1u) {
                        o_local[dd] = o_local[dd] * alpha + p_j * v_tile[kc * u.d + dd];
                    }
                    l = l * alpha + p_j;
                    m = m_new;
                }
            }
        }
        workgroupBarrier();
    }

    if (valid) {
        let inv_l = 1.0 / l;
        for (var d = 0u; d < u.d; d = d + 1u) {
            out[q_off + d] = act_store(o_local[d] * inv_l);
        }
    }
}
"#
);

/// Packed-bf16 small-D sdpa. Q/K/V/Mask/Out all `array<u32>` (bf16 pairs).
/// Compute (dot products, softmax, accumulators) stays fp32. Requires `D`
/// even (always true: 8/128 in Z-Image+Qwen3) and `s_k` even when
/// `has_mask != 0` (so each row's bf16 mask aligns to packed words).
/// Tiled flash-attention layout — see WGSL_F32 above for the shared design.
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

const BR: u32 = 64u;
const BC: u32 = 32u;
const WG: u32 = 64u;
const MAX_DW: u32 = 64u;

// Shared at 16 KiB/tile to keep 2 workgroups/SM (32 KiB+ collapses to 1/SM
// on Ada and similar; BC=64 measured ~85% slower on a 5070 768x768).
var<workgroup> k_tile: array<u32, 2048>; // BC * MAX_DW
var<workgroup> v_tile: array<u32, 2048>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let t  = lid.x;
    let qt = wgid.x;
    let hq = wgid.y;
    let bb = wgid.z;
    let sq = qt * BR + t;
    let valid = sq < u.s_q;

    let hkv = (hq * u.h_kv) / u.h_q;
    let d_w        = u.d >> 1u;
    let q_w_off    = (((bb * u.s_q + sq) * u.h_q + hq) * u.d) >> 1u;
    let kv_b0_w    = ((bb * u.s_k * u.h_kv + hkv) * u.d) >> 1u;
    let kv_step_w  = (u.h_kv * u.d) >> 1u;
    // has_mask==2: per-head mask [B, Hq, Sq, Sk]; else shared [B, Sq, Sk].
    let mask_row = select(bb * u.s_q + sq, (bb * u.h_q + hq) * u.s_q + sq, u.has_mask == 2u);
    let mask_w_base = (mask_row * u.s_k) >> 1u;

    var q_local: array<f32, 128>;
    for (var d = 0u; d < u.d; d = d + 1u) { q_local[d] = 0.0; }
    if (valid) {
        for (var dw = 0u; dw < d_w; dw = dw + 1u) {
            let qv = unpack_bf16x2(q[q_w_off + dw]);
            q_local[dw * 2u]      = qv.x;
            q_local[dw * 2u + 1u] = qv.y;
        }
    }

    var o_local: array<f32, 128>;
    for (var d = 0u; d < u.d; d = d + 1u) { o_local[d] = 0.0; }
    // Large-negative finite f32, not -inf: Tint (browser WebGPU) rejects
    // const-exprs evaluating to -inf (bitcast<f32>(0xff800000u)) and ALSO
    // out-of-range decimals like -3.4028235e38 (Rust's f32::MIN print is
    // a hair beyond max-finite as an exact decimal); naga accepts both.
    // Equivalent as a running-max init.
    var m: f32 = -3.4e38;
    var l: f32 = 0.0;

    let n_tiles = (u.s_k + BC - 1u) / BC;
    let words_per_tile = BC * d_w;
    let per_thread = words_per_tile / WG;

    for (var kt = 0u; kt < n_tiles; kt = kt + 1u) {
        let kc_base = kt * BC;

        for (var i = 0u; i < per_thread; i = i + 1u) {
            let idx = i * WG + t;
            let kc  = idx / d_w;
            let dw  = idx % d_w;
            let key_global = kc_base + kc;
            var kw: u32 = 0u;
            var vw: u32 = 0u;
            if (key_global < u.s_k) {
                let base = kv_b0_w + key_global * kv_step_w + dw;
                kw = k[base];
                vw = v[base];
            }
            k_tile[idx] = kw;
            v_tile[idx] = vw;
        }
        workgroupBarrier();

        if (valid) {
            for (var kc = 0u; kc < BC; kc = kc + 1u) {
                let key_global = kc_base + kc;
                if (key_global < u.s_k) {
                    var dot: f32 = 0.0;
                    for (var dw = 0u; dw < d_w; dw = dw + 1u) {
                        let kv = unpack_bf16x2(k_tile[kc * d_w + dw]);
                        dot = dot + q_local[dw * 2u] * kv.x + q_local[dw * 2u + 1u] * kv.y;
                    }
                    var bias: f32 = 0.0;
                    if (u.has_mask != 0u) {
                        let mw = unpack_bf16x2(mask[mask_w_base + (key_global >> 1u)]);
                        bias = select(mw.x, mw.y, (key_global & 1u) == 1u);
                    }
                    let s_j  = dot * u.scale + bias;
                    let m_new = max(m, s_j);
                    let alpha = exp(m - m_new);
                    let p_j   = exp(s_j - m_new);
                    for (var dw = 0u; dw < d_w; dw = dw + 1u) {
                        let vv = unpack_bf16x2(v_tile[kc * d_w + dw]);
                        o_local[dw * 2u]      = o_local[dw * 2u]      * alpha + p_j * vv.x;
                        o_local[dw * 2u + 1u] = o_local[dw * 2u + 1u] * alpha + p_j * vv.y;
                    }
                    l = l * alpha + p_j;
                    m = m_new;
                }
            }
        }
        workgroupBarrier();
    }

    if (valid) {
        let inv_l = 1.0 / l;
        for (var dw = 0u; dw < d_w; dw = dw + 1u) {
            out[q_w_off + dw] = pack_bf16x2(o_local[dw * 2u] * inv_l, o_local[dw * 2u + 1u] * inv_l);
        }
    }
}
"#
);

// F16 small-D sdpa. Q/K/V/Mask/Out all `array<vec2<f16>>`. Dot products,
// softmax (m/l/p/alpha), and per-key V-accumulation stay f32 — softmax
// `exp(s_j)` saturates f16 once scores exceed ~11.1, and the online
// renormalization mixes magnitudes across keys in a way that loses bits
// in f16. Same numerical contract as bf16-packed.
// Tiled flash-attention layout — see WGSL_F32 above for the shared design.
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

const BR: u32 = 64u;
const BC: u32 = 32u;
const WG: u32 = 64u;
const MAX_DW: u32 = 64u;

// Shared at 16 KiB/tile to keep 2 workgroups/SM (32 KiB+ collapses to 1/SM
// on Ada and similar; BC=64 measured ~85% slower on a 5070 768x768).
var<workgroup> k_tile: array<vec2<f16>, 2048>; // BC * MAX_DW
var<workgroup> v_tile: array<vec2<f16>, 2048>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let t  = lid.x;
    let qt = wgid.x;
    let hq = wgid.y;
    let bb = wgid.z;
    let sq = qt * BR + t;
    let valid = sq < u.s_q;

    let hkv = (hq * u.h_kv) / u.h_q;
    let d_w        = u.d >> 1u;
    let q_w_off    = (((bb * u.s_q + sq) * u.h_q + hq) * u.d) >> 1u;
    let kv_b0_w    = ((bb * u.s_k * u.h_kv + hkv) * u.d) >> 1u;
    let kv_step_w  = (u.h_kv * u.d) >> 1u;
    // has_mask==2: per-head mask [B, Hq, Sq, Sk]; else shared [B, Sq, Sk].
    let mask_row = select(bb * u.s_q + sq, (bb * u.h_q + hq) * u.s_q + sq, u.has_mask == 2u);
    let mask_w_base = (mask_row * u.s_k) >> 1u;

    var q_local: array<f32, 128>;
    for (var d = 0u; d < u.d; d = d + 1u) { q_local[d] = 0.0; }
    if (valid) {
        for (var dw = 0u; dw < d_w; dw = dw + 1u) {
            let qv: vec2<f32> = vec2<f32>(q[q_w_off + dw]);
            q_local[dw * 2u]      = qv.x;
            q_local[dw * 2u + 1u] = qv.y;
        }
    }

    var o_local: array<f32, 128>;
    for (var d = 0u; d < u.d; d = d + 1u) { o_local[d] = 0.0; }
    // Large-negative finite f32, not -inf: Tint (browser WebGPU) rejects
    // const-exprs evaluating to -inf (bitcast<f32>(0xff800000u)) and ALSO
    // out-of-range decimals like -3.4028235e38 (Rust's f32::MIN print is
    // a hair beyond max-finite as an exact decimal); naga accepts both.
    // Equivalent as a running-max init.
    var m: f32 = -3.4e38;
    var l: f32 = 0.0;

    let n_tiles = (u.s_k + BC - 1u) / BC;
    let words_per_tile = BC * d_w;
    let per_thread = words_per_tile / WG;

    for (var kt = 0u; kt < n_tiles; kt = kt + 1u) {
        let kc_base = kt * BC;

        for (var i = 0u; i < per_thread; i = i + 1u) {
            let idx = i * WG + t;
            let kc  = idx / d_w;
            let dw  = idx % d_w;
            let key_global = kc_base + kc;
            var kw: vec2<f16> = vec2<f16>(f16(0.0), f16(0.0));
            var vw: vec2<f16> = vec2<f16>(f16(0.0), f16(0.0));
            if (key_global < u.s_k) {
                let base = kv_b0_w + key_global * kv_step_w + dw;
                kw = k[base];
                vw = v[base];
            }
            k_tile[idx] = kw;
            v_tile[idx] = vw;
        }
        workgroupBarrier();

        if (valid) {
            for (var kc = 0u; kc < BC; kc = kc + 1u) {
                let key_global = kc_base + kc;
                if (key_global < u.s_k) {
                    var dot: f32 = 0.0;
                    for (var dw = 0u; dw < d_w; dw = dw + 1u) {
                        let kv: vec2<f32> = vec2<f32>(k_tile[kc * d_w + dw]);
                        dot = dot + q_local[dw * 2u] * kv.x + q_local[dw * 2u + 1u] * kv.y;
                    }
                    var bias: f32 = 0.0;
                    if (u.has_mask != 0u) {
                        let mw: vec2<f32> = vec2<f32>(mask[mask_w_base + (key_global >> 1u)]);
                        bias = select(mw.x, mw.y, (key_global & 1u) == 1u);
                    }
                    let s_j  = dot * u.scale + bias;
                    let m_new = max(m, s_j);
                    let alpha = exp(m - m_new);
                    let p_j   = exp(s_j - m_new);
                    for (var dw = 0u; dw < d_w; dw = dw + 1u) {
                        let vv: vec2<f32> = vec2<f32>(v_tile[kc * d_w + dw]);
                        o_local[dw * 2u]      = o_local[dw * 2u]      * alpha + p_j * vv.x;
                        o_local[dw * 2u + 1u] = o_local[dw * 2u + 1u] * alpha + p_j * vv.y;
                    }
                    l = l * alpha + p_j;
                    m = m_new;
                }
            }
        }
        workgroupBarrier();
    }

    if (valid) {
        let inv_l = 1.0 / l;
        for (var dw = 0u; dw < d_w; dw = dw + 1u) {
            out[q_w_off + dw] = vec2<f16>(vec2<f32>(o_local[dw * 2u] * inv_l, o_local[dw * 2u + 1u] * inv_l));
        }
    }
}
"#
);

// F16 subgroup sdpa, CL-parameterized. Same flash layout as WGSL_F16_PACKED but
// each Q row is owned by a CL-lane cluster instead of one thread: D is split
// across the cluster (vec4<f16> slices, MAX_DV4/CL per lane at D=128), the
// per-key dot is cluster-reduced with log2(CL) subgroupShuffleXor hops, and
// every lane then replicates the online-softmax scalars (m, l, p_j, alpha)
// locally - no shared-memory round trip, no barrier in the inner loop. This
// removes the per-thread `array<f32, 128>` q/o state of the one-thread-one-row
// kernel, which spills to local memory at D=128 and made sdpa memory-latency-
// bound (~0.22 TFLOPS effective at 768x768; llama.cpp's Vulkan flash_attn.comp
// uses the same D_split + subgroupShuffleXor decomposition).
//
// CL is chosen at the build site = min(8, reported subgroup_min_size): native
// (NVIDIA min=32) uses CL=8; web/mobile, where the browser reports the spec
// floor of 4 (and wgpu-web hardcodes it - it can't read the real value), uses
// CL=4. The cluster reduce is correct as long as CL divides the ACTUAL runtime
// subgroup size, which holds for any power-of-2 size >= CL; picking CL <=
// reported_min guarantees it whatever size the driver picks. Math is identical
// across CL, so native CL=8 stays bit-for-bit unchanged.
//
// Tail keys (s_k not a multiple of BC) are folded branchlessly: their score
// is forced to -FLT_MAX so p_j = 0 and the shuffle stays in uniform control
// flow.
//
// Constraints: d_v4 % CL == 0 (i.e. D % (4*CL) == 0), D <= 128. Build sites pass
// D % 32 == 0, which satisfies both CL=4 and CL=8. Grid [ceil(S_q/BR), H_q, B]
// with BR = WG/CL. Shared: BC * MAX_DV4 * 8 bytes * 2 = 16 KiB.
// NOTE: no `enable subgroups;` here - naga (native) implements the subgroup
// builtins but rejects the enable directive itself (gfx-rs/wgpu#5555); matmul_i8
// relies on the same behavior. On the web (Tint) backend the directive IS
// required, so the model layer prepends `backend.subgroup_enable_directive()`
// to this source at the build site.
pub fn build_f16_sg_wgsl(cl: u32, r: u32) -> String {
    build_f16_sg(cl, r, false)
}

/// Temporal sliding-window variant of [`build_f16_sg_wgsl`]. Each query attends
/// only to keys whose latent frame is within `±window` of the query's frame,
/// where the token layout is frame-major (`(f, h, w)` row-major) so frame =
/// `token / period` and `period` = tokens per latent frame. The uniform carries
/// three extra fields (`period`, `window`, `row_off`); `row_off` is the global
/// row index of the first query in this (possibly chunked) dispatch, so the
/// kernel recovers each query's GLOBAL frame. Two savings vs the dense kernel:
/// per-workgroup the key-tile loop is clamped to the window's frame span (the
/// O(frames^2) -> O(frames * window) win), and per-key out-of-window scores are
/// folded to `-FLT_MAX` (p_j = 0) for the exact boundary. K/V are still the full
/// sequence (only the Q range is ever chunked), so `key_global` indexes globally.
/// This CHANGES the attention output (it is a different operation, not an
/// approximation of full attention); gate it behind the run-time flag.
pub fn build_f16_sg_windowed_wgsl(cl: u32, r: u32) -> String {
    build_f16_sg(cl, r, true)
}

/// Threads per workgroup for the subgroup sdpa. 256 (the WebGPU default
/// invocations-per-workgroup limit) rather than 128: BR = WG/CL query rows
/// share each streamed K/V tile, and at video sequence lengths the kernel is
/// DRAM-bound on K/V re-streaming (each workgroup reads the full K/V for its
/// head), so doubling BR halves the dominant traffic. Measured 2026-07-02 at
/// 832x480x81f (s = 32760): SDPA was ~9.0s of a 13.2s DiT block at WG=128.
pub const SG_WG: u32 = 256;

fn build_f16_sg(cl: u32, r: u32, windowed: bool) -> String {
    assert!(cl == 4 || cl == 8, "sdpa_sg: CL must be 4 or 8, got {cl}");
    assert!(
        matches!(r, 1 | 2 | 4),
        "sdpa_sg: R (Q rows per lane cluster) must be 1/2/4, got {r}"
    );
    let br = SG_WG / cl * r; // Q rows per workgroup
    let max_nl = 32 / cl; // MAX_DV4=32 vec4 (D=128) split across CL lanes
    // Q-register blocking: each CL-lane cluster owns R consecutive Q rows.
    // Per key the K/V shared-tile loads + f16->f32 converts happen ONCE per
    // cluster (in kw*/vw* registers) and are reused by all R rows; the dot
    // FMAs, shuffle reduce, and softmax state stay per-row, so each row's
    // f32 accumulation order is IDENTICAL across R values (bit-exact vs R=1).
    // Cost: R x (q + o) vec4 register sets; R=2 at D=128/CL=8 is ~24 live
    // vec4s per lane.
    let decls = |kw: &str| -> String {
        (0..r)
            .flat_map(|rr| {
                (0..max_nl).map(move |i| format!("    var {kw}{rr}_{i} = vec4<f32>();\n"))
            })
            .collect()
    };
    let q_decls = decls("q");
    let o_decls = decls("o");
    // Per-row scalar state + Q loads.
    let mut row_state = String::new();
    let mut q_loads = String::new();
    for rr in 0..r {
        row_state.push_str(&format!(
            "    let sq{rr} = row0 + {rr}u;\n    let valid{rr} = sq{rr} < u.s_q;\n    let sq_c{rr} = min(sq{rr}, u.s_q - 1u);\n    let q_off{rr} = (((bb * u.s_q + sq_c{rr}) * u.h_q + hq) * u.d) >> 2u;\n    let mask_row{rr} = select(bb * u.s_q + sq_c{rr}, (bb * u.h_q + hq) * u.s_q + sq_c{rr}, u.has_mask == 2u);\n    let mask_w_base{rr} = (mask_row{rr} * u.s_k) >> 1u;\n"
        ));
        q_loads.push_str(&format!("    q{rr}_0 = vec4<f32>(q[q_off{rr} + l_off]);\n"));
        for i in 1..max_nl {
            q_loads.push_str(&format!(
                "    if (n_l > {i}u) {{ q{rr}_{i} = vec4<f32>(q[q_off{rr} + l_off + {i}u]); }}\n"
            ));
        }
    }
    // K tile -> registers, once per cluster per key.
    let mut k_loads = String::from("            let kw0 = vec4<f32>(k_tile[tb]);\n");
    let mut v_loads = String::from("            let vw0 = vec4<f32>(v_tile[tb]);\n");
    for i in 1..max_nl {
        k_loads.push_str(&format!(
            "            var kw{i} = vec4<f32>();\n            if (n_l > {i}u) {{ kw{i} = vec4<f32>(k_tile[tb + {i}u]); }}\n"
        ));
        v_loads.push_str(&format!(
            "            var vw{i} = vec4<f32>();\n            if (n_l > {i}u) {{ vw{i} = vec4<f32>(v_tile[tb + {i}u]); }}\n"
        ));
    }
    // Per-row dot + cluster shuffle reduce (xor hops 1, 2, .., CL/2 so every
    // lane ends with the full dot).
    let mut dot = String::new();
    for rr in 0..r {
        dot.push_str(&format!("            var part{rr} = dot(q{rr}_0, kw0);\n"));
        for i in 1..max_nl {
            dot.push_str(&format!(
                "            if (n_l > {i}u) {{ part{rr} = part{rr} + dot(q{rr}_{i}, kw{i}); }}\n"
            ));
        }
        let mut off = 1u32;
        while off < cl {
            dot.push_str(&format!(
                "            part{rr} = part{rr} + subgroupShuffleXor(part{rr}, {off}u);\n"
            ));
            off <<= 1;
        }
    }
    // Per-row softmax step + O update (V registers shared across rows).
    let mut softmax_o = String::new();
    for rr in 0..r {
        softmax_o.push_str(&format!(
            "            var bias{rr}: f32 = 0.0;\n            if (u.has_mask != 0u) {{\n                let mw: vec2<f32> = vec2<f32>(mask[mask_w_base{rr} + (key_global >> 1u)]);\n                bias{rr} = select(mw.x, mw.y, (key_global & 1u) == 1u);\n            }}\n"
        ));
        if windowed {
            softmax_o.push_str(&format!(
                "            let is_txt_q{rr} = (u.row_off + sq_c{rr}) >= u.txt_start;\n            let in_win{rr} = is_txt_q{rr} || is_txt_k || ((max(fq{rr}, fk) - min(fq{rr}, fk)) <= u.window);\n            let s_j{rr} = select(NEG_MAX, part{rr} * u.scale + bias{rr}, key_global < u.s_k && in_win{rr});\n"
            ));
        } else {
            softmax_o.push_str(&format!(
                "            let s_j{rr} = select(NEG_MAX, part{rr} * u.scale + bias{rr}, key_global < u.s_k);\n"
            ));
        }
        softmax_o.push_str(&format!(
            "            let m_new{rr} = max(m{rr}, s_j{rr});\n            let alpha{rr} = exp(m{rr} - m_new{rr});\n            let p_j{rr}   = exp(s_j{rr} - m_new{rr});\n"
        ));
        softmax_o.push_str(&format!(
            "            o{rr}_0 = o{rr}_0 * alpha{rr} + p_j{rr} * vw0;\n"
        ));
        for i in 1..max_nl {
            softmax_o.push_str(&format!(
                "            if (n_l > {i}u) {{ o{rr}_{i} = o{rr}_{i} * alpha{rr} + p_j{rr} * vw{i}; }}\n"
            ));
        }
        softmax_o.push_str(&format!(
            "            l{rr} = l{rr} * alpha{rr} + p_j{rr};\n            m{rr} = m_new{rr};\n"
        ));
    }
    // Per-row output writes.
    let mut out_w = String::new();
    for rr in 0..r {
        out_w.push_str(&format!(
            "    if (valid{rr}) {{\n        let inv_l{rr} = select(0.0, 1.0 / l{rr}, l{rr} > 0.0);\n        out[q_off{rr} + l_off] = vec4<f16>(o{rr}_0 * inv_l{rr});\n"
        ));
        for i in 1..max_nl {
            out_w.push_str(&format!(
                "        if (n_l > {i}u) {{ out[q_off{rr} + l_off + {i}u] = vec4<f16>(o{rr}_{i} * inv_l{rr}); }}\n"
            ));
        }
        out_w.push_str("    }\n");
    }
    let mut m_l_decls = String::new();
    for rr in 0..r {
        m_l_decls.push_str(&format!(
            "    var m{rr}: f32 = NEG_MAX;\n    var l{rr}: f32 = 0.0;\n"
        ));
    }

    let u_window = if windowed {
        // `txt_start` = first joint-token index that is text (always-in-window,
        // never windowed). Pure-video callers pass `txt_start = s_k` so the text
        // branches are dead and the kernel matches the original windowed form.
        "\n    period: u32, window: u32, row_off: u32, txt_start: u32,"
    } else {
        ""
    };
    let mut s = format!(
        r#"enable f16;

struct U {{
    b: u32, h_q: u32, h_kv: u32, s_q: u32,
    s_k: u32, d: u32, scale: f32, has_mask: u32,{u_window}
}};

@group(0) @binding(0) var<storage, read> q: array<vec4<f16>>;"#
    );
    s.push_str(
        r#"
@group(0) @binding(1) var<storage, read> k: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> v: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> mask: array<vec2<f16>>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f16>>;
@group(0) @binding(5) var<uniform> u: U;

"#,
    );
    s.push_str(&format!(
        "const BR: u32 = {br}u;      // Q rows per workgroup (WG/CL*R)\nconst BC: u32 = 32u;      // keys per shared tile\nconst WG: u32 = {SG_WG}u;\nconst CL: u32 = {cl}u;       // lanes per Q row (D split)\nconst R: u32 = {r}u;       // Q rows per lane cluster\nconst MAX_DV4: u32 = 32u; // vec4s per row at D=128\nconst NEG_MAX: f32 = -3.402823e38;\n"
    ));
    s.push_str(&format!(
        r#"
var<workgroup> k_tile: array<vec4<f16>, 1024>; // BC * MAX_DV4
var<workgroup> v_tile: array<vec4<f16>, 1024>;

@compute @workgroup_size({SG_WG})
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{"#
    ));
    s.push_str(
        r#"
    let t    = lid.x;
    let lane = t % CL;
    let row0 = wgid.x * BR + (t / CL) * R; // first Q row of this lane cluster
    let hq    = wgid.y;
    let bb    = wgid.z;

    let hkv     = (hq * u.h_kv) / u.h_q;
    let d_v4    = u.d >> 2u;
    let n_l     = d_v4 / CL; // vec4s per lane
    let kv_b0   = ((bb * u.s_k * u.h_kv + hkv) * u.d) >> 2u;
    let kv_step = (u.h_kv * u.d) >> 2u;
    let l_off   = lane * n_l;
    // Per-row state (has_mask==2: per-head mask [B, Hq, Sq, Sk]; else
    // shared [B, Sq, Sk]).
"#,
    );
    s.push_str(&row_state);
    s.push('\n');
    if windowed {
        // Global query frame per row (frame-major layout); row_off shifts the
        // chunk.
        for rr in 0..r {
            s.push_str(&format!(
                "    let fq{rr} = (u.row_off + sq_c{rr}) / u.period;\n"
            ));
        }
        s.push('\n');
    }
    s.push_str(&q_decls);
    s.push_str(&q_loads);
    s.push('\n');
    s.push_str(&o_decls);
    s.push_str(&m_l_decls);
    s.push_str(
        r#"
    let v4_per_tile = BC * d_v4;
"#,
    );
    if windowed {
        // Clamp the key-tile loop to the frame window spanning this workgroup's
        // BR global query rows: image keys in frames [min_fq - W, max_fq + W],
        // PLUS the text key tiles (always in-window). A workgroup that holds any
        // text query (`wg_last >= txt_start`) -- the pure-text region and the one
        // straddling tile -- runs full attention (all tiles, no skip) so its text
        // queries see every key; pure-image workgroups iterate the windowed image
        // tiles and the text tiles, skipping the gap between. For pure-video
        // callers (`txt_start == s_k`) `full` is always false and the text range
        // is empty, so this reduces to the original windowed loop bit-identically.
        s.push_str(
            r#"    let wg_first = u.row_off + wgid.x * BR;
    let wg_last  = u.row_off + min(wgid.x * BR + BR - 1u, u.s_q - 1u);
    let n_tiles  = (u.s_k + BC - 1u) / BC;
    let n_f      = u.txt_start / u.period; // image frames only
    let full     = wg_last >= u.txt_start; // workgroup contains a text query
    let fq_lo    = wg_first / u.period;
    let kf_lo    = select(fq_lo - u.window, 0u, fq_lo < u.window);
    // Guard n_f == 0 (txt_start < period): `n_f - 1u` would wrap to 0xFFFFFFFF.
    let n_f_last = select(0u, n_f - 1u, n_f > 0u);
    let kf_hi    = min(n_f_last, wg_last / u.period + u.window);
    let kt_lo    = select((kf_lo * u.period) / BC, 0u, full);
    let kt_hi_img = ((kf_hi + 1u) * u.period + BC - 1u) / BC;
    let txt_tile_lo = u.txt_start / BC;

    for (var kt = kt_lo; kt < n_tiles; kt = kt + 1u) {
        if (!full && kt >= kt_hi_img && kt < txt_tile_lo) { continue; }
"#,
        );
    } else {
        s.push_str(
            r#"    let n_tiles = (u.s_k + BC - 1u) / BC;

    for (var kt = 0u; kt < n_tiles; kt = kt + 1u) {
"#,
        );
    }
    s.push_str(
        r#"        let kc_base = kt * BC;

        for (var idx = t; idx < v4_per_tile; idx = idx + WG) {
            let kc = idx / d_v4;
            let dv = idx % d_v4;
            let key_global = kc_base + kc;
            var kw = vec4<f16>();
            var vw = vec4<f16>();
            if (key_global < u.s_k) {
                let base = kv_b0 + key_global * kv_step + dv;
                kw = k[base];
                vw = v[base];
            }
            k_tile[idx] = kw;
            v_tile[idx] = vw;
        }
        workgroupBarrier();

        for (var kc = 0u; kc < BC; kc = kc + 1u) {
            let key_global = kc_base + kc;
            let tb = kc * d_v4 + l_off;

"#,
    );
    // K/V tile -> registers once per cluster per key; then per-row dot +
    // shuffle reduce + softmax step + O update. Out-of-window / tail keys
    // score -FLT_MAX -> p_j = 0 (exact fold). In the windowed variant text
    // keys and text queries bypass the frame window (joint attention: image
    // queries attend all text; text queries attend everything); pure-video
    // callers (`txt_start == s_k`) reduce to the plain `|fq - fk| <= window`
    // test.
    s.push_str(&k_loads);
    s.push_str(&dot);
    if windowed {
        s.push_str(
            r#"            let is_txt_k = key_global >= u.txt_start;
            let fk = key_global / u.period;
"#,
        );
    }
    s.push_str(&v_loads);
    s.push_str(&softmax_o);
    s.push_str(
        r#"        }
        workgroupBarrier();
    }

    // Guard an all-masked query (l == 0): write 0 rather than 1/0 = inf/NaN.
    // A NaN propagating into the residual stream can surface as a device loss.
    // When l > 0 (the normal case: every query has >= 1 in-window key) inv_l
    // is bit-identically 1.0 / l.
"#,
    );
    s.push_str(&out_w);
    s.push_str("}\n");
    s
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
            (ActDtype::I8, _) => unreachable!("ActDtype::I8 is never a block-level act dtype"),
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

/// Bindings for the subgroup small-D sdpa (same 6-slot layout as [`SdpaF32`]).
/// F16 acts only, D % 32 == 0, D <= 128; built via [`build_f16_sg_wgsl`] and
/// dispatched via [`dispatch_sdpa_f16_sg`]. Dispatch sites fall back to
/// [`SdpaF32`] when the D constraint doesn't hold.
pub fn sg_layout() -> &'static [BindingLayout] {
    LAYOUT
}

/// Workgroup grid for the CL/R-parameterized subgroup sdpa: BR = WG/CL*R Q
/// rows per workgroup, so the grid is [ceil(S_q/BR), H_q, B]. `cl` and `r`
/// must match the values passed to [`build_f16_sg_wgsl`] for the bound
/// pipeline.
pub fn f16_sg_workgroups(cl: u32, r: u32, b: u32, s_q: u32, h_q: u32) -> [u32; 3] {
    [s_q.div_ceil(SG_WG / cl * r), h_q, b]
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_sdpa_f16_sg<B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &SdpaBufs<'_>,
    cl: u32,
    r: u32,
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
    backend.dispatch(
        encoder,
        pipeline,
        &bindings,
        f16_sg_workgroups(cl, r, b, s_q, h_q),
    )
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
// Constraints: `D % WG == 0`, `D <= WG * MAX_LOCAL_D` (== 1024 for WG=128,
// MAX_LOCAL_D=8). Asserted at dispatch time.

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
const MAX_LOCAL_D: u32 = 8u; // supports D up to WG*MAX_LOCAL_D = 1024.

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
    // has_mask==2: per-head mask [B, Hq, Sq, Sk]; else shared [B, Sq, Sk].
    let mask_row = select(bb * u.s_q + sq, (bb * u.h_q + hq) * u.s_q + sq, u.has_mask == 2u);
    let mask_base = mask_row * u.s_k;

    // d_per is D/WG; caller guarantees divisibility.
    let d_per = u.d / WG;
    let d_off = t * d_per;

    // Load this thread's slice of q into registers.
    var q_local: array<f32, 8>;
    for (var i = 0u; i < d_per; i = i + 1u) {
        q_local[i] = q[q_off + d_off + i];
    }
    var o_local: array<f32, 8>;
    for (var i = 0u; i < d_per; i = i + 1u) {
        o_local[i] = 0.0;
    }

    if (t == 0u) {
        shared_scalar[0] = -3.4e38; // m running-max init (in-range finite; see note at sdpa var m)
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
const MAX_LOCAL_D: u32 = 8u;

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
    // has_mask==2: per-head mask [B, Hq, Sq, Sk]; else shared [B, Sq, Sk].
    let mask_row = select(bb * u.s_q + sq, (bb * u.h_q + hq) * u.s_q + sq, u.has_mask == 2u);
    let mask_w_base = (mask_row * u.s_k) >> 1u;

    // d_per (in elements) must be even since D is even and WG divides D.
    // d_per_w (in words) = d_per / 2.
    let d_per = u.d / WG;
    let d_per_w = d_per >> 1u;
    let d_off_w = t * d_per_w;

    var q_local: array<f32, 8>;
    for (var i = 0u; i < d_per_w; i = i + 1u) {
        let qv: vec2<f32> = vec2<f32>(q[q_w_off + d_off_w + i]);
        q_local[i * 2u]      = qv.x;
        q_local[i * 2u + 1u] = qv.y;
    }
    var o_local: array<f32, 8>;
    for (var i = 0u; i < d_per; i = i + 1u) {
        o_local[i] = 0.0;
    }

    if (t == 0u) {
        shared_scalar[0] = -3.4e38; // m running-max init (in-range finite; see note at sdpa var m)
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

// ---------------------------------------------------------------------------
// Causal large-D variant: on-the-fly per-frame causal prefix, no materialized
// mask. Used by the Hunyuan VAE mid-block self-attention at production frame
// counts where the additive `[N, N]` mask (~4.3 GiB at N=32760) does not fit.
// ---------------------------------------------------------------------------
//
// The mid-attn causal rule is a clean per-frame prefix: query token `sq`
// (frame `sq/period`) attends every key whose frame is `<= sq/period`, i.e.
// keys `[0, (sq/period + 1)*period)`. That cutoff lands on a frame boundary, so
// it is an exact loop bound (no per-key bias) -- the kernel just clamps the key
// loop, which also ~halves the work vs full attention. `period` (= tokens per
// latent frame, `h*w`) is uniform field 8; fields 9-11 pad the struct to the
// 48-byte (16-aligned) uniform size. The `mask` binding is unused (bind a
// 1-element scratch); identical bindings/layout to `SdpaF32LargeD` otherwise.
wgsl_with_bf16_variant!(
    WGSL_LARGE_D_CAUSAL_F32,
    WGSL_LARGE_D_CAUSAL_F32_BF16 = r#"
struct U {
    b: u32, h_q: u32, h_kv: u32, s_q: u32,
    s_k: u32, d: u32, scale: f32, has_mask: u32,
    period: u32, row_off: u32, _pad1: u32, _pad2: u32,
};

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> u: U;

const WG: u32 = 128u;
const MAX_LOCAL_D: u32 = 8u; // supports D up to WG*MAX_LOCAL_D = 1024.

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

    // Causal prefix: keys up to (and including) the query's frame. Cutoff lands
    // on a frame boundary, so it is an exact key-loop bound (no per-key mask).
    // `row_off` is the GLOBAL first-row of this dispatch's query range: q/out are
    // bound as subviews starting at `row_off` (so `q_off` stays local), but the
    // causal frame is computed from the global row `row_off + sq` -- this lets the
    // caller chunk the query range into per-submit pieces (TDR-safe) at real f.
    let s_k_eff = min(u.s_k, ((u.row_off + sq) / u.period + 1u) * u.period);

    let d_per = u.d / WG;
    let d_off = t * d_per;

    var q_local: array<f32, 8>;
    for (var i = 0u; i < d_per; i = i + 1u) {
        q_local[i] = q[q_off + d_off + i];
    }
    var o_local: array<f32, 8>;
    for (var i = 0u; i < d_per; i = i + 1u) {
        o_local[i] = 0.0;
    }

    if (t == 0u) {
        shared_scalar[0] = -3.4e38; // m running-max init (in-range finite)
        shared_scalar[1] = 0.0;                       // l
    }
    workgroupBarrier();

    for (var j = 0u; j < s_k_eff; j = j + 1u) {
        let kj = kv_b0 + j * kv_step;

        var partial: f32 = 0.0;
        for (var i = 0u; i < d_per; i = i + 1u) {
            partial = partial + q_local[i] * k[kj + d_off + i];
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
            let s_j   = dot * u.scale;
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

/// Causal per-frame large-D SDPA (no materialized mask). Same bindings/layout as
/// [`SdpaF32LargeD`]; the uniform carries an extra `period` (tokens per latent
/// frame) field. Hunyuan VAE mid-block only (f32 acts).
pub struct SdpaF32LargeDCausal;

impl SdpaOp for SdpaF32LargeDCausal {
    const KERNEL_ID: &'static str = "sdpa_large_d_causal.f32";
    type Dtype = F32;
    const Q: &'static str = "sdpa/q";
    const K: &'static str = "sdpa/k";
    const V: &'static str = "sdpa/v";
    const MASK: &'static str = "sdpa/mask";
    const DIMS: &'static str = "sdpa/dims";
    const OUTPUT: &'static str = "sdpa/out";
    const MAX_D: u32 = 1024;

    fn wgsl(cfg: &WgslConfig) -> &'static str {
        match cfg.act_dtype {
            ActDtype::F32 if cfg.bf16_quant_writes => WGSL_LARGE_D_CAUSAL_F32_BF16,
            ActDtype::F32 => WGSL_LARGE_D_CAUSAL_F32,
            other => unreachable!("sdpa_large_d_causal is f32-only (VAE mid-block), got {other:?}"),
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
    fn workgroups(b: u32, s_q: u32, h_q: u32) -> [u32; 3] {
        super::linear_workgroups(b * s_q * h_q, 1)
    }
}

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
    const MAX_D: u32 = 1024;

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
            ActDtype::I8 => {
                // VAE never gets I8 acts (per worklog: VAE I8 explicitly out of
                // scope - different shape regime, tile-bounded working set).
                unreachable!("sdpa_large_d: ActDtype::I8 not supported (VAE stays F32)")
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

// ---------------------------------------------------------------------------
// Decode variant: one workgroup per query row, the workgroup COOPERATES over
// the KV length. Same 6-slot layout + uniform as `SdpaF32`.
// ---------------------------------------------------------------------------
//
// `SdpaF32` assigns one thread per query row and streams keys serially in that
// thread. That is fine when S_q is large (each row is a busy thread), but it is
// catastrophic for single-token DECODE (S_q == 1): the whole attention over a
// long KV cache runs in ONE thread per head (~32 active threads on the device),
// while the 5.8k-key softmax dominates. This kernel keeps ONE workgroup per
// (b, s_q, h_q) row but splits the work across all WG=128 threads:
//   * cooperative K/V tile load (BC=32 keys/tile) into shared, like `SdpaF32`;
//   * phase A: threads [0, BC) each score one key of the tile (dot over D);
//   * a BC-wide online-softmax bookkeeping step (thread 0, cheap);
//   * phase B: threads [0, D) each accumulate one output dim over the tile.
// Barriers are per TILE (~S_k/32), not per key, so the long-KV decode SDPA goes
// from ~1 active thread to a full workgroup. Bit-equivalent to `SdpaF32` (same
// f32 online-softmax math); intended for S_q small + S_k large (decode, and the
// low-M attention sites). Grid: one workgroup per row (linear, Y-spill).
wgsl_with_bf16_variant!(
    WGSL_DECODE_F32,
    WGSL_DECODE_F32_BF16 = r#"
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
const BC: u32 = 32u;
const MAX_D: u32 = 128u;

var<workgroup> q_sh: array<f32, 128>;
var<workgroup> k_tile: array<f32, 4096>; // BC * MAX_D
var<workgroup> v_tile: array<f32, 4096>;
var<workgroup> sc: array<f32, 32>;       // BC: key scores, then probabilities
var<workgroup> o_sh: array<f32, 128>;    // running output accumulator
// 0 = running max m, 1 = running sum l, 2 = alpha (this tile's rescale).
var<workgroup> sm: array<f32, 4>;

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

    let q_off   = ((bb * u.s_q + sq) * u.h_q + hq) * u.d;
    let kv_b0   = (bb * u.s_k * u.h_kv + hkv) * u.d;
    let kv_step = u.h_kv * u.d;
    // has_mask==2: per-head mask [B, Hq, Sq, Sk]; else shared [B, Sq, Sk].
    let mask_row = select(bb * u.s_q + sq, (bb * u.h_q + hq) * u.s_q + sq, u.has_mask == 2u);
    let mask_base = mask_row * u.s_k;

    if (t < u.d) {
        q_sh[t] = q[q_off + t];
        o_sh[t] = 0.0;
    }
    if (t == 0u) {
        sm[0] = -3.4e38; // m init (in-range finite; see note at SdpaF32 var m)
        sm[1] = 0.0;
    }
    workgroupBarrier();

    let n_tiles = (u.s_k + BC - 1u) / BC;
    let tile_elems = BC * u.d;
    for (var kt = 0u; kt < n_tiles; kt = kt + 1u) {
        let kc_base = kt * BC;
        // Cooperative K/V tile load (all WG threads, strided).
        for (var idx = t; idx < tile_elems; idx = idx + WG) {
            let kc  = idx / u.d;
            let dd  = idx % u.d;
            let key = kc_base + kc;
            var kv: f32 = 0.0;
            var vv: f32 = 0.0;
            if (key < u.s_k) {
                let base = kv_b0 + key * kv_step + dd;
                kv = k[base];
                vv = v[base];
            }
            k_tile[idx] = kv;
            v_tile[idx] = vv;
        }
        workgroupBarrier();

        // Phase A: threads [0, BC) score one key each (tail keys -> -inf).
        if (t < BC) {
            let key = kc_base + t;
            var s_j: f32 = -3.4e38;
            if (key < u.s_k) {
                var dot: f32 = 0.0;
                for (var dd = 0u; dd < u.d; dd = dd + 1u) {
                    dot = dot + q_sh[dd] * k_tile[t * u.d + dd];
                }
                let bias = select(0.0, mask[mask_base + key], u.has_mask != 0u);
                s_j = dot * u.scale + bias;
            }
            sc[t] = s_j;
        }
        workgroupBarrier();

        // Online-softmax bookkeeping over the BC tile scores (thread 0). Rewrites
        // sc[] in place with the per-key probabilities for phase B.
        if (t == 0u) {
            var tmax: f32 = -3.4e38;
            for (var kc = 0u; kc < BC; kc = kc + 1u) {
                tmax = max(tmax, sc[kc]);
            }
            let m_cur = sm[0];
            let m_new = max(m_cur, tmax);
            let alpha = exp(m_cur - m_new);
            var l_new = sm[1] * alpha;
            for (var kc = 0u; kc < BC; kc = kc + 1u) {
                let p = exp(sc[kc] - m_new); // tail keys: exp(-inf) = 0
                sc[kc] = p;
                l_new = l_new + p;
            }
            sm[0] = m_new;
            sm[1] = l_new;
            sm[2] = alpha;
        }
        workgroupBarrier();

        // Phase B: threads [0, D) accumulate one output dim over the tile.
        if (t < u.d) {
            let alpha = sm[2];
            var acc = o_sh[t] * alpha;
            for (var kc = 0u; kc < BC; kc = kc + 1u) {
                acc = acc + sc[kc] * v_tile[kc * u.d + t];
            }
            o_sh[t] = acc;
        }
        workgroupBarrier();
    }

    if (t < u.d) {
        // Guard an all-masked query (l == 0): write 0 not 1/0 = inf/NaN.
        let l = sm[1];
        let inv_l = select(0.0, 1.0 / l, l > 0.0);
        out[q_off + t] = act_store(o_sh[t] * inv_l);
    }
}
"#
);

/// Packed-bf16 decode sdpa (see [`WGSL_DECODE_F32`] for the cooperative design).
/// Q/K/V/Mask/Out are `array<u32>` (bf16 pairs); dot products, softmax, and the
/// V accumulation stay f32. Shared K/V tiles hold the UNPACKED f32 values.
const WGSL_DECODE_BF16: &str = concat!(
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

const WG: u32 = 128u;
const BC: u32 = 32u;
const MAX_D: u32 = 128u;

var<workgroup> q_sh: array<f32, 128>;
var<workgroup> k_tile: array<f32, 4096>; // BC * MAX_D (unpacked)
var<workgroup> v_tile: array<f32, 4096>;
var<workgroup> sc: array<f32, 32>;
var<workgroup> o_sh: array<f32, 128>;
var<workgroup> sm: array<f32, 4>;

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
    let mask_row = select(bb * u.s_q + sq, (bb * u.h_q + hq) * u.s_q + sq, u.has_mask == 2u);
    let mask_w_base = (mask_row * u.s_k) >> 1u;

    if (t < d_w) {
        let qv = unpack_bf16x2(q[q_w_off + t]);
        q_sh[t * 2u]      = qv.x;
        q_sh[t * 2u + 1u] = qv.y;
    }
    if (t < u.d) { o_sh[t] = 0.0; }
    if (t == 0u) {
        sm[0] = -3.4e38;
        sm[1] = 0.0;
    }
    workgroupBarrier();

    let n_tiles = (u.s_k + BC - 1u) / BC;
    let tile_words = BC * d_w;
    for (var kt = 0u; kt < n_tiles; kt = kt + 1u) {
        let kc_base = kt * BC;
        for (var idxw = t; idxw < tile_words; idxw = idxw + WG) {
            let kc  = idxw / d_w;
            let dw  = idxw % d_w;
            let key = kc_base + kc;
            var kw: u32 = 0u;
            var vw: u32 = 0u;
            if (key < u.s_k) {
                let base = kv_b0_w + key * kv_step_w + dw;
                kw = k[base];
                vw = v[base];
            }
            let ku = unpack_bf16x2(kw);
            let vu = unpack_bf16x2(vw);
            k_tile[kc * u.d + dw * 2u]      = ku.x;
            k_tile[kc * u.d + dw * 2u + 1u] = ku.y;
            v_tile[kc * u.d + dw * 2u]      = vu.x;
            v_tile[kc * u.d + dw * 2u + 1u] = vu.y;
        }
        workgroupBarrier();

        if (t < BC) {
            let key = kc_base + t;
            var s_j: f32 = -3.4e38;
            if (key < u.s_k) {
                var dot: f32 = 0.0;
                for (var dd = 0u; dd < u.d; dd = dd + 1u) {
                    dot = dot + q_sh[dd] * k_tile[t * u.d + dd];
                }
                var bias: f32 = 0.0;
                if (u.has_mask != 0u) {
                    let mw = unpack_bf16x2(mask[mask_w_base + (key >> 1u)]);
                    bias = select(mw.x, mw.y, (key & 1u) == 1u);
                }
                s_j = dot * u.scale + bias;
            }
            sc[t] = s_j;
        }
        workgroupBarrier();

        if (t == 0u) {
            var tmax: f32 = -3.4e38;
            for (var kc = 0u; kc < BC; kc = kc + 1u) {
                tmax = max(tmax, sc[kc]);
            }
            let m_cur = sm[0];
            let m_new = max(m_cur, tmax);
            let alpha = exp(m_cur - m_new);
            var l_new = sm[1] * alpha;
            for (var kc = 0u; kc < BC; kc = kc + 1u) {
                let p = exp(sc[kc] - m_new);
                sc[kc] = p;
                l_new = l_new + p;
            }
            sm[0] = m_new;
            sm[1] = l_new;
            sm[2] = alpha;
        }
        workgroupBarrier();

        if (t < u.d) {
            let alpha = sm[2];
            var acc = o_sh[t] * alpha;
            for (var kc = 0u; kc < BC; kc = kc + 1u) {
                acc = acc + sc[kc] * v_tile[kc * u.d + t];
            }
            o_sh[t] = acc;
        }
        workgroupBarrier();
    }

    if (t < d_w) {
        let l = sm[1];
        let inv_l = select(0.0, 1.0 / l, l > 0.0);
        out[q_w_off + t] = pack_bf16x2(o_sh[t * 2u] * inv_l, o_sh[t * 2u + 1u] * inv_l);
    }
}
"#
);

pub struct SdpaDecode;

impl SdpaOp for SdpaDecode {
    const KERNEL_ID: &'static str = "sdpa_decode.f32";
    type Dtype = F32;
    const Q: &'static str = "sdpa/q";
    const K: &'static str = "sdpa/k";
    const V: &'static str = "sdpa/v";
    const MASK: &'static str = "sdpa/mask";
    const DIMS: &'static str = "sdpa/dims";
    const OUTPUT: &'static str = "sdpa/out";
    fn wgsl(cfg: &WgslConfig) -> &'static str {
        match (cfg.act_dtype, cfg.bf16_quant_writes) {
            (ActDtype::F32, false) => WGSL_DECODE_F32,
            (ActDtype::F32, true) => WGSL_DECODE_F32_BF16,
            (ActDtype::Bf16, _) => WGSL_DECODE_BF16,
            (ActDtype::F16, _) => {
                panic!("sdpa_decode: f16-packed acts variant not built (bf16/f32 only)")
            }
            (ActDtype::I8, _) => unreachable!("ActDtype::I8 is never a block-level act dtype"),
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
    fn workgroups(b: u32, s_q: u32, h_q: u32) -> [u32; 3] {
        // One workgroup per query row (kernel recovers the linear row from
        // `wgid.x + wgid.y * ng.x`, so counts > 65535 spill to Y).
        super::linear_workgroups(b * s_q * h_q, 1)
    }
}

#[cfg(feature = "conformance")]
impl OpTest for SdpaDecode {
    fn dtypes(&self) -> &'static [Dtype] {
        DTYPES_ACT_BF16
    }
    fn test_cases(&self) -> Vec<TestCase> {
        // Two shapes: a single-query DECODE row over a longer KV (the target
        // regime), and the same generic multi-row/GQA case `SdpaF32` uses. Both
        // compare to the CPU sdpa reference.
        let scale = 1.0_f32 / 8.0_f32.sqrt();
        vec![
            TestCase {
                name: "sdpa_decode_sq1",
                op: OpSpec::Sdpa { scale },
                inputs: vec![
                    t("q", [1, 1, 2, 8], linspace(-1.0, 1.0, false)),
                    t("k", [1, 40, 2, 8], linspace(-0.5, 1.5, false)),
                    t("v", [1, 40, 2, 8], linspace(-1.25, 0.75, true)),
                    t("mask", [1, 1, 40], linspace(0.0, 0.0, false)),
                ],
            },
            TestCase {
                name: "sdpa_decode_multirow",
                op: OpSpec::Sdpa { scale },
                inputs: vec![
                    t("q", [1, 4, 2, 8], linspace(-1.0, 1.0, false)),
                    t("k", [1, 4, 2, 8], linspace(-0.5, 1.5, false)),
                    t("v", [1, 4, 2, 8], linspace(-1.25, 0.75, true)),
                    t("mask", [1, 4, 4], linspace(0.0, 0.0, false)),
                ],
            },
        ]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_sdpa::<SdpaDecode>())
    }
}
