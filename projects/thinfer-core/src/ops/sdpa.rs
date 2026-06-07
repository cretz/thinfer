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
    let mask_base = (bb * u.s_q + sq) * u.s_k;

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
    let mask_w_base = ((bb * u.s_q + sq) * u.s_k) >> 1u;

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
    let mask_w_base = ((bb * u.s_q + sq) * u.s_k) >> 1u;

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
pub fn build_f16_sg_wgsl(cl: u32) -> String {
    assert!(cl == 4 || cl == 8, "sdpa_sg: CL must be 4 or 8, got {cl}");
    let br = 128 / cl; // WG=128 lanes / CL lanes-per-row
    let max_nl = 32 / cl; // MAX_DV4=32 vec4 (D=128) split across CL lanes
    let decls = |kw: &str| -> String {
        (0..max_nl)
            .map(|i| format!("    var {kw}{i} = vec4<f32>();\n"))
            .collect()
    };
    let q_decls = decls("q");
    let o_decls = decls("o");
    let mut q_loads = String::from("    q0 = vec4<f32>(q[q_off + l_off]);\n");
    let mut dot = String::from("            var part = dot(q0, vec4<f32>(k_tile[tb]));\n");
    let mut o_upd = String::from("            o0 = o0 * alpha + p_j * vec4<f32>(v_tile[tb]);\n");
    let mut out_w = String::from("        out[q_off + l_off] = vec4<f16>(o0 * inv_l);\n");
    for i in 1..max_nl {
        q_loads.push_str(&format!(
            "    if (n_l > {i}u) {{ q{i} = vec4<f32>(q[q_off + l_off + {i}u]); }}\n"
        ));
        dot.push_str(&format!(
            "            if (n_l > {i}u) {{ part = part + dot(q{i}, vec4<f32>(k_tile[tb + {i}u])); }}\n"
        ));
        o_upd.push_str(&format!(
            "            if (n_l > {i}u) {{ o{i} = o{i} * alpha + p_j * vec4<f32>(v_tile[tb + {i}u]); }}\n"
        ));
        out_w.push_str(&format!(
            "        if (n_l > {i}u) {{ out[q_off + l_off + {i}u] = vec4<f16>(o{i} * inv_l); }}\n"
        ));
    }
    // Cluster reduce: xor hops 1, 2, .., CL/2 so every lane ends with the full dot.
    let mut hops = String::new();
    let mut off = 1u32;
    while off < cl {
        hops.push_str(&format!(
            "            part = part + subgroupShuffleXor(part, {off}u);\n"
        ));
        off <<= 1;
    }

    let mut s = String::from(
        r#"enable f16;

struct U {
    b: u32, h_q: u32, h_kv: u32, s_q: u32,
    s_k: u32, d: u32, scale: f32, has_mask: u32,
};

@group(0) @binding(0) var<storage, read> q: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> k: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> v: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> mask: array<vec2<f16>>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f16>>;
@group(0) @binding(5) var<uniform> u: U;

"#,
    );
    s.push_str(&format!(
        "const BR: u32 = {br}u;      // Q rows per workgroup (WG/CL)\nconst BC: u32 = 32u;      // keys per shared tile\nconst WG: u32 = 128u;\nconst CL: u32 = {cl}u;       // lanes per Q row (D split)\nconst MAX_DV4: u32 = 32u; // vec4s per row at D=128\nconst NEG_MAX: f32 = -3.402823e38;\n"
    ));
    s.push_str(
        r#"
var<workgroup> k_tile: array<vec4<f16>, 1024>; // BC * MAX_DV4
var<workgroup> v_tile: array<vec4<f16>, 1024>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let t    = lid.x;
    let row  = t / CL;
    let lane = t % CL;
    let sq    = wgid.x * BR + row;
    let hq    = wgid.y;
    let bb    = wgid.z;
    let valid = sq < u.s_q;
    let sq_c  = min(sq, u.s_q - 1u);

    let hkv     = (hq * u.h_kv) / u.h_q;
    let d_v4    = u.d >> 2u;
    let n_l     = d_v4 / CL; // vec4s per lane
    let q_off   = (((bb * u.s_q + sq_c) * u.h_q + hq) * u.d) >> 2u;
    let kv_b0   = ((bb * u.s_k * u.h_kv + hkv) * u.d) >> 2u;
    let kv_step = (u.h_kv * u.d) >> 2u;
    let mask_w_base = ((bb * u.s_q + sq_c) * u.s_k) >> 1u;
    let l_off   = lane * n_l;

"#,
    );
    s.push_str(&q_decls);
    s.push_str(&q_loads);
    s.push('\n');
    s.push_str(&o_decls);
    s.push_str(
        r#"    var m: f32 = NEG_MAX;
    var l: f32 = 0.0;

    let n_tiles = (u.s_k + BC - 1u) / BC;
    let v4_per_tile = BC * d_v4;

    for (var kt = 0u; kt < n_tiles; kt = kt + 1u) {
        let kc_base = kt * BC;

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
    s.push_str(&dot);
    s.push_str(&hops);
    s.push_str(
        r#"
            var bias: f32 = 0.0;
            if (u.has_mask != 0u) {
                let mw: vec2<f32> = vec2<f32>(mask[mask_w_base + (key_global >> 1u)]);
                bias = select(mw.x, mw.y, (key_global & 1u) == 1u);
            }
            // Tail keys score -FLT_MAX -> p_j = 0, alpha = 1: no-op fold.
            let s_j = select(NEG_MAX, part * u.scale + bias, key_global < u.s_k);
            let m_new = max(m, s_j);
            let alpha = exp(m - m_new);
            let p_j   = exp(s_j - m_new);
"#,
    );
    s.push_str(&o_upd);
    s.push_str(
        r#"            l = l * alpha + p_j;
            m = m_new;
        }
        workgroupBarrier();
    }

    if (valid) {
        let inv_l = 1.0 / l;
"#,
    );
    s.push_str(&out_w);
    s.push_str("    }\n}\n");
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

/// Workgroup grid for the CL-parameterized subgroup sdpa: BR = WG/CL Q rows
/// per workgroup, so the grid is [ceil(S_q/BR), H_q, B]. `cl` must match the
/// value passed to [`build_f16_sg_wgsl`] for the bound pipeline.
pub fn f16_sg_workgroups(cl: u32, b: u32, s_q: u32, h_q: u32) -> [u32; 3] {
    [s_q.div_ceil(128 / cl), h_q, b]
}

pub(crate) fn dispatch_sdpa_f16_sg<B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &SdpaBufs<'_>,
    cl: u32,
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
        f16_sg_workgroups(cl, b, s_q, h_q),
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
