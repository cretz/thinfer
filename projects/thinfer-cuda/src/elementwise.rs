//! The small ops around a MiniMax-H3 block's matmuls and attention, as CUDA
//! kernels, so a whole block runs on one stream with no wgpu round trips
//! ([`thinfer_core::backend::ElementwiseOp`]). Each matches the wgpu op it
//! replaces, including where that path rounds to bf16 between steps: bf16
//! `[rows, C]` activations, f32 math, round-to-nearest-even stores. Plus the
//! H3 VAE encoder's f32 GroupNorm (+ SiLU) and add.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaContext, sys};
use thinfer_core::backend::{ElementwiseOp, GROUP_NORM_PART, MAX_SEGMENTS, RowSegments};

use crate::nvrtc::Module;

const SOURCE: &str = r#"
#include <cuda_bf16.h>

#define MAX_SEG 16

#include <cuda_fp16.h>
struct Seg { unsigned n; unsigned end[MAX_SEG]; unsigned long long a[MAX_SEG]; unsigned long long b[MAX_SEG]; };

__device__ __forceinline__ float bf(unsigned short v) { return __uint_as_float((unsigned)v << 16); }
__device__ __forceinline__ unsigned short rn(float x) { return __bfloat16_as_ushort(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float rnf(float x) { return bf(rn(x)); }

__device__ __forceinline__ unsigned seg_of(const Seg& s, unsigned row) {
    unsigned i = 0;
    while (i + 1 < s.n && row >= s.end[i]) ++i;
    return i;
}

__device__ float block_sum(float x, float* red) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = x;
    __syncthreads();
    float r = 0.f;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) r += red[w];
    __syncthreads();
    return r;
}

// out = rn(rn(x * inv * w) * (1 + scale[seg]) + shift[seg]); one block per row.
extern "C" __global__ void __launch_bounds__(256) norm_modulate(
    const unsigned short* __restrict__ x, const unsigned short* __restrict__ w,
    unsigned short* __restrict__ out, unsigned dim, float eps, Seg s)
{
    __shared__ float red[8];
    const unsigned row = blockIdx.x;
    const unsigned short* xr = x + (size_t)row * dim;
    float ss = 0.f;
    for (unsigned c = threadIdx.x; c < dim; c += blockDim.x) { const float v = bf(xr[c]); ss += v * v; }
    const float inv = rsqrtf(block_sum(ss, red) / (float)dim + eps);
    const unsigned g = seg_of(s, row);
    const unsigned short* sc = reinterpret_cast<const unsigned short*>(s.a[g]);
    const unsigned short* sh = reinterpret_cast<const unsigned short*>(s.b[g]);
    for (unsigned c = threadIdx.x; c < dim; c += blockDim.x) {
        const float n = rnf(bf(xr[c]) * inv * bf(w[c]));
        out[(size_t)row * dim + c] = rn(n * (bf(sc[c]) + 1.f) + bf(sh[c]));
    }
}

// Fused qkv [rows, 3*H*D] -> per-head rmsnorm(q|k) then partial half-rot rope
// on the leading 2*rot_pairs channels (pairs (c, c + rot_pairs)), v copied.
// Grid (rows, heads), D threads. freqs: bf16 (cos, sin) x rot_pairs per row.
extern "C" __global__ void __launch_bounds__(128) qk_norm_rope(
    const unsigned short* __restrict__ qkv, const unsigned short* __restrict__ qw,
    const unsigned short* __restrict__ kw, const unsigned short* __restrict__ freqs,
    unsigned short* __restrict__ q, unsigned short* __restrict__ k, unsigned short* __restrict__ v,
    unsigned heads, unsigned rot_pairs, float eps, unsigned headwise, unsigned q_only)
{
    __shared__ float y[128];
    __shared__ float red[4];
    const unsigned row = blockIdx.x, h = blockIdx.y, c = threadIdx.x, D = blockDim.x;
    const size_t inner = (size_t)heads * D;
    // Slab layout [q | k | v], or headwise [q_h | k_h | v_h] per head, or
    // (q_only) the q projection alone.
    const size_t part = headwise ? D : inner;
    const size_t src = (size_t)row * (q_only ? 1 : 3) * inner + (size_t)h * (headwise ? 3 * D : D) + c;
    const size_t dst = (size_t)row * inner + (size_t)h * D + c;
    if (!q_only) v[dst] = qkv[src + 2 * part];
    const unsigned rot = 2 * rot_pairs;
    for (int which = 0; which < (q_only ? 1 : 2); ++which) {
        const float x = bf(qkv[src + which * part]);
        const float inv = rsqrtf(block_sum(x * x, red) / (float)D + eps);
        const unsigned short* gw = which ? kw : qw;
        const float n = rnf(x * inv * (gw ? bf(gw[c]) : 1.f));
        y[c] = n;
        __syncthreads();
        float o = n;
        if (c < rot) {
            const unsigned kk = c % rot_pairs;
            const float cr = bf(freqs[((size_t)row * rot_pairs + kk) * 2]);
            const float ci = bf(freqs[((size_t)row * rot_pairs + kk) * 2 + 1]);
            o = c < rot_pairs ? n * cr - y[c + rot_pairs] * ci : y[c - rot_pairs] * ci + n * cr;
        }
        (which ? k : q)[dst] = rn(o);
        __syncthreads();
    }
}

// out = rn(x + gate[seg][c] * y), elementwise over [rows, dim].
extern "C" __global__ void __launch_bounds__(256) gate_add(
    const unsigned short* __restrict__ x, const unsigned short* __restrict__ y,
    unsigned short* __restrict__ out, unsigned rows, unsigned dim, Seg s)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * dim) return;
    const unsigned row = (unsigned)(i / dim), c = (unsigned)(i % dim);
    const unsigned short* g = reinterpret_cast<const unsigned short*>(s.a[seg_of(s, row)]);
    out[i] = rn(bf(x[i]) + bf(g[c]) * bf(y[i]));
}

// out = rn(x + bias[c]) over [rows, dim]. `out` may alias `x` (each thread
// reads then writes its own element), so neither is __restrict__.
extern "C" __global__ void __launch_bounds__(256) add_bias(
    const unsigned short* x, const unsigned short* __restrict__ bias,
    unsigned short* out, unsigned rows, unsigned dim)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * dim) return;
    out[i] = rn(bf(x[i]) + bf(bias[i % dim]));
}

// out[r, c] = silu(x[r, c]) * x[r, h + c] over x [rows, 2h].
extern "C" __global__ void __launch_bounds__(256) silu_mul_fused(
    const unsigned short* __restrict__ x, unsigned short* __restrict__ out, unsigned rows, unsigned h)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * h) return;
    const size_t r = i / h, c = i % h;
    const float g = bf(x[r * 2 * h + c]);
    out[i] = rn(g / (1.f + expf(-g)) * bf(x[r * 2 * h + h + c]));
}

// GroupNorm of f32 [C, HW] in three launches, deterministic: per (group,
// part) the part's mean and M2 (two passes over its <= GN_PART elements),
// a fixed-order Chan merge per group into (mean, 1 / sqrt(var + eps)), then
// the affine (+ SiLU). `part` = [groups][parts] (mean, M2) pairs, `stats`
// [groups] pairs after it.
#define GN_PART 4096
extern "C" __global__ void __launch_bounds__(256) gn_partial(
    const float* __restrict__ x, float* __restrict__ part, unsigned n_group, unsigned parts)
{
    __shared__ float red[8];
    const unsigned g = blockIdx.x, p = blockIdx.y;
    const unsigned lo = p * GN_PART, hi = min(lo + GN_PART, n_group);
    const float* xg = x + (size_t)g * n_group;
    float s = 0.f;
    for (unsigned i = lo + threadIdx.x; i < hi; i += blockDim.x) s += xg[i];
    const float mean = block_sum(s, red) / (float)(hi - lo);
    float q = 0.f;
    for (unsigned i = lo + threadIdx.x; i < hi; i += blockDim.x) { const float d = xg[i] - mean; q += d * d; }
    const float m2 = block_sum(q, red);
    if (threadIdx.x == 0) {
        part[((size_t)g * parts + p) * 2] = mean;
        part[((size_t)g * parts + p) * 2 + 1] = m2;
    }
}

extern "C" __global__ void __launch_bounds__(256) gn_stats(
    const float* __restrict__ part, float* __restrict__ stats, unsigned n_group, unsigned parts,
    unsigned groups, float eps)
{
    const unsigned g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    float n = 0.f, mean = 0.f, m2 = 0.f;
    for (unsigned p = 0; p < parts; ++p) {
        const float nb = (float)min(GN_PART, n_group - p * GN_PART);
        const float mb = part[((size_t)g * parts + p) * 2], m2b = part[((size_t)g * parts + p) * 2 + 1];
        const float nn = n + nb, d = mb - mean;
        mean += d * (nb / nn);
        m2 += m2b + d * d * (n * nb / nn);
        n = nn;
    }
    stats[2 * g] = mean;
    stats[2 * g + 1] = rsqrtf(m2 / n + eps);
}

extern "C" __global__ void __launch_bounds__(256) gn_apply(
    const float* __restrict__ x, const unsigned short* __restrict__ w,
    const unsigned short* __restrict__ b, const float* __restrict__ stats,
    float* __restrict__ out, unsigned c, unsigned hw, unsigned cpg, unsigned silu)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)c * hw) return;
    const unsigned ch = (unsigned)(i / hw), g = ch / cpg;
    const float scale = bf(w[ch]) * stats[2 * g + 1];
    const float offset = bf(b[ch]) - stats[2 * g] * scale;
    float y = x[i] * scale + offset;
    if (silu) y = y / (1.f + expf(-y));
    out[i] = y;
}

extern "C" __global__ void __launch_bounds__(256) add_f32(
    const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ out, unsigned n)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + y[i];
}

// The Wan DiT's f16 forms. h2f/f2h: f16 bits <-> f32 (round to nearest even,
// clamped to the f16 range like the WGSL stores).
__device__ __forceinline__ float h2f(unsigned short v) { return __half2float(__ushort_as_half(v)); }
__device__ __forceinline__ unsigned short f2h(float v) {
    return __half_as_ushort(__float2half_rn(fminf(fmaxf(v, -65504.f), 65504.f)));
}
// n = f16(layernorm(x)) (two passes: mean, then biased variance), then
// out = f16(n * (1 + scale[c]) + shift[c]); one block per row.
extern "C" __global__ void __launch_bounds__(256) ln_modulate_f16(
    const unsigned short* __restrict__ x, const unsigned short* __restrict__ scale,
    const unsigned short* __restrict__ shift, unsigned short* __restrict__ out,
    unsigned dim, float eps)
{
    __shared__ float red[8];
    const unsigned short* xr = x + (size_t)blockIdx.x * dim;
    float sum = 0.f;
    for (unsigned c = threadIdx.x; c < dim; c += blockDim.x) sum += h2f(xr[c]);
    const float mean = block_sum(sum, red) / (float)dim;
    float sq = 0.f;
    for (unsigned c = threadIdx.x; c < dim; c += blockDim.x) { const float d = h2f(xr[c]) - mean; sq += d * d; }
    const float inv = rsqrtf(block_sum(sq, red) / (float)dim + eps);
    for (unsigned c = threadIdx.x; c < dim; c += blockDim.x) {
        const float n = h2f(f2h((h2f(xr[c]) - mean) * inv));
        out[(size_t)blockIdx.x * dim + c] = f2h(n * (h2f(scale[c]) + 1.f) + h2f(shift[c]));
    }
}
// Interleaved-pair RoPE in f16 arithmetic: grid over rows*heads*pairs.
extern "C" __global__ void __launch_bounds__(256) rope_f16(
    const __half2* __restrict__ x, const __half2* __restrict__ freqs, __half2* __restrict__ out,
    unsigned rows, unsigned heads, unsigned pairs)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * heads * pairs) return;
    const unsigned p = (unsigned)(i % pairs);
    const size_t row = i / pairs / heads;
    const __half2 v = x[i], f = freqs[row * pairs + p];
    out[i] = __halves2half2(__hsub(__hmul(v.x, f.x), __hmul(v.y, f.y)),
                            __hadd(__hmul(v.x, f.y), __hmul(v.y, f.x)));
}
// out = f16(x * rsqrt(mean(x^2) + eps) * w); one block per row.
extern "C" __global__ void __launch_bounds__(256) rms_norm_f16(
    const unsigned short* __restrict__ x, const unsigned short* __restrict__ w,
    unsigned short* __restrict__ out, unsigned dim, float eps)
{
    __shared__ float red[8];
    const unsigned short* xr = x + (size_t)blockIdx.x * dim;
    float ss = 0.f;
    for (unsigned c = threadIdx.x; c < dim; c += blockDim.x) { const float v = h2f(xr[c]); ss += v * v; }
    const float inv = rsqrtf(block_sum(ss, red) / (float)dim + eps);
    for (unsigned c = threadIdx.x; c < dim; c += blockDim.x)
        out[(size_t)blockIdx.x * dim + c] = f2h(h2f(xr[c]) * inv * bf(w[c]));
}
// out = f16(layernorm(x) * w + b) (biased variance); one block per row.
extern "C" __global__ void __launch_bounds__(256) ln_affine_f16(
    const unsigned short* __restrict__ x, const unsigned short* __restrict__ w,
    const unsigned short* __restrict__ b, unsigned short* __restrict__ out, unsigned dim, float eps)
{
    __shared__ float red[8];
    const unsigned short* xr = x + (size_t)blockIdx.x * dim;
    float sum = 0.f;
    for (unsigned c = threadIdx.x; c < dim; c += blockDim.x) sum += h2f(xr[c]);
    const float mean = block_sum(sum, red) / (float)dim;
    float sq = 0.f;
    for (unsigned c = threadIdx.x; c < dim; c += blockDim.x) { const float d = h2f(xr[c]) - mean; sq += d * d; }
    const float inv = rsqrtf(block_sum(sq, red) / (float)dim + eps);
    for (unsigned c = threadIdx.x; c < dim; c += blockDim.x)
        out[(size_t)blockIdx.x * dim + c] = f2h((h2f(xr[c]) - mean) * inv * bf(w[c]) + bf(b[c]));
}
// out = f16(x + bias[c]) (bf16 bias); out may alias x.
extern "C" __global__ void __launch_bounds__(256) add_bias_f16(
    const unsigned short* x, const unsigned short* __restrict__ bias, unsigned short* out,
    unsigned rows, unsigned dim)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * dim) return;
    out[i] = f2h(h2f(x[i]) + bf(bias[i % dim]));
}
// out = f16(0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))).
extern "C" __global__ void __launch_bounds__(256) gelu_tanh_f16(
    const unsigned short* __restrict__ x, unsigned short* __restrict__ out, unsigned n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = h2f(x[i]);
    out[i] = f2h(0.5f * v * (1.f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v))));
}
// out = f16(layernorm(x)) (no affine); one block per row.
extern "C" __global__ void __launch_bounds__(256) ln_f16(
    const unsigned short* __restrict__ x, unsigned short* __restrict__ out, unsigned dim, float eps)
{
    __shared__ float red[8];
    const unsigned short* xr = x + (size_t)blockIdx.x * dim;
    float sum = 0.f;
    for (unsigned c = threadIdx.x; c < dim; c += blockDim.x) sum += h2f(xr[c]);
    const float mean = block_sum(sum, red) / (float)dim;
    float sq = 0.f;
    for (unsigned c = threadIdx.x; c < dim; c += blockDim.x) { const float d = h2f(xr[c]) - mean; sq += d * d; }
    const float inv = rsqrtf(block_sum(sq, red) / (float)dim + eps);
    for (unsigned c = threadIdx.x; c < dim; c += blockDim.x)
        out[(size_t)blockIdx.x * dim + c] = f2h((h2f(xr[c]) - mean) * inv);
}
// out = f16(x * (1 + scale[c]) + shift[c]) over [rows, dim].
extern "C" __global__ void __launch_bounds__(256) modulate_f16(
    const unsigned short* __restrict__ x, const unsigned short* __restrict__ scale,
    const unsigned short* __restrict__ shift, unsigned short* __restrict__ out, unsigned rows, unsigned dim)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * dim) return;
    const unsigned c = (unsigned)(i % dim);
    out[i] = f2h(fmaf(h2f(x[i]), 1.f + h2f(scale[c]), h2f(shift[c])));
}
// out = f16(x * w[c]) (bf16 w) over [rows, dim].
extern "C" __global__ void __launch_bounds__(256) mul_weight_f16(
    const unsigned short* __restrict__ x, const unsigned short* __restrict__ w,
    unsigned short* __restrict__ out, unsigned rows, unsigned dim)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)rows * dim) out[i] = f2h(h2f(x[i]) * bf(w[i % dim]));
}
// out = f16(x + y) over n.
extern "C" __global__ void __launch_bounds__(256) add_f16(
    const unsigned short* __restrict__ x, const unsigned short* __restrict__ y,
    unsigned short* __restrict__ out, unsigned n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f2h(h2f(x[i]) + h2f(y[i]));
}
// The Wan VAE's channel RMSNorm, one thread per position in the WGSL op's
// channel order: out = f16(x * (sqrt(C) / max(sqrt(sum x^2), 1e-12)) * gamma[c]).
extern "C" __global__ void __launch_bounds__(256) rms_norm3d_f16(
    const unsigned short* __restrict__ x, const unsigned short* __restrict__ gamma,
    unsigned short* __restrict__ out, unsigned n_pos, unsigned channels, unsigned stride)
{
    const unsigned p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pos) return;
    const unsigned b = p / stride, s = p - b * stride;
    const size_t base = (size_t)b * channels * stride + s;
    float ss = 0.f;
    for (unsigned c = 0; c < channels; ++c) { const float v = h2f(x[base + (size_t)c * stride]); ss = __fadd_rn(ss, __fmul_rn(v, v)); }
    const float inv = sqrtf((float)channels) / fmaxf(sqrtf(ss), 1e-12f);
    for (unsigned c = 0; c < channels; ++c) {
        const size_t i = base + (size_t)c * stride;
        out[i] = f2h(__fmul_rn(__fmul_rn(h2f(x[i]), inv), bf(gamma[c])));
    }
}
// out = f16(x / (1 + exp(-x))).
extern "C" __global__ void __launch_bounds__(256) silu_f16(
    const unsigned short* __restrict__ x, unsigned short* __restrict__ out, unsigned n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = h2f(x[i]);
    out[i] = f2h(v / (1.f + expf(-v)));
}
// Time concat over [bc, t, hw]: frame `to` from a (or zero) then b.
struct ConcatT { unsigned bc, hw, a_t, b_t, a_start, a_count, b_start, b_count, a_zero; };
extern "C" __global__ void __launch_bounds__(256) concat_time_f16(
    const unsigned short* __restrict__ a, const unsigned short* __restrict__ b,
    unsigned short* __restrict__ out, const ConcatT p)
{
    const unsigned t_out = p.a_count + p.b_count;
    const size_t o = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= (size_t)p.bc * t_out * p.hw) return;
    const unsigned s = (unsigned)(o % p.hw);
    const size_t rem = o / p.hw;
    const unsigned to = (unsigned)(rem % t_out), bc = (unsigned)(rem / t_out);
    unsigned short v = 0;
    if (to < p.a_count) {
        if (!p.a_zero) v = a[((size_t)bc * p.a_t + p.a_start + to) * p.hw + s];
    } else {
        v = b[((size_t)bc * p.b_t + p.b_start + (to - p.a_count)) * p.hw + s];
    }
    out[o] = v;
}
// out = f16(x + gate[c] * y) over [rows, dim].
extern "C" __global__ void __launch_bounds__(256) gate_add_f16(
    const unsigned short* __restrict__ x, const unsigned short* __restrict__ gate,
    const unsigned short* __restrict__ y, unsigned short* __restrict__ out, unsigned rows, unsigned dim)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * dim) return;
    out[i] = f2h(fmaf(h2f(gate[i % dim]), h2f(y[i]), h2f(x[i])));
}
"#;

/// `struct Seg` by value (`MAX_SEG` in the source mirrors `MAX_SEGMENTS`).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Seg {
    n: u32,
    end: [u32; MAX_SEGMENTS],
    a: [u64; MAX_SEGMENTS],
    b: [u64; MAX_SEGMENTS],
}

impl From<&RowSegments> for Seg {
    fn from(s: &RowSegments) -> Self {
        Self {
            n: s.n,
            end: s.end,
            a: s.a,
            b: s.b,
        }
    }
}

pub struct ElementwiseKernels {
    module: Module,
    norm_modulate: sys::CUfunction,
    qk_norm_rope: sys::CUfunction,
    gate_add: sys::CUfunction,
    ln_modulate_f16: sys::CUfunction,
    gate_add_f16: sys::CUfunction,
    rope_f16: sys::CUfunction,
    rms_norm_f16: sys::CUfunction,
    ln_affine_f16: sys::CUfunction,
    add_bias_f16: sys::CUfunction,
    gelu_tanh_f16: sys::CUfunction,
    add_f16: sys::CUfunction,
    rms_norm3d_f16: sys::CUfunction,
    silu_f16: sys::CUfunction,
    concat_time_f16: sys::CUfunction,
    ln_f16: sys::CUfunction,
    modulate_f16: sys::CUfunction,
    mul_weight_f16: sys::CUfunction,
    add_bias: sys::CUfunction,
    silu_mul_fused: sys::CUfunction,
    gn_partial: sys::CUfunction,
    gn_stats: sys::CUfunction,
    gn_apply: sys::CUfunction,
    add_f32: sys::CUfunction,
}

// The source's `GN_PART` mirrors the core constant.
const _: () = assert!(GROUP_NORM_PART == 4096);

// SAFETY: function handles are plain driver handles; every use binds the
// owning context first.
unsafe impl Send for ElementwiseKernels {}
unsafe impl Sync for ElementwiseKernels {}

impl ElementwiseKernels {
    pub fn new(ctx: Arc<CudaContext>, sm: (i32, i32)) -> Result<Self, String> {
        let module = Module::compile(ctx, SOURCE, "thinfer_elementwise.cu", sm)?;
        Ok(Self {
            norm_modulate: module.function(c"norm_modulate", 0)?,
            qk_norm_rope: module.function(c"qk_norm_rope", 0)?,
            gate_add: module.function(c"gate_add", 0)?,
            ln_modulate_f16: module.function(c"ln_modulate_f16", 0)?,
            gate_add_f16: module.function(c"gate_add_f16", 0)?,
            rope_f16: module.function(c"rope_f16", 0)?,
            rms_norm_f16: module.function(c"rms_norm_f16", 0)?,
            ln_affine_f16: module.function(c"ln_affine_f16", 0)?,
            add_bias_f16: module.function(c"add_bias_f16", 0)?,
            gelu_tanh_f16: module.function(c"gelu_tanh_f16", 0)?,
            add_f16: module.function(c"add_f16", 0)?,
            rms_norm3d_f16: module.function(c"rms_norm3d_f16", 0)?,
            silu_f16: module.function(c"silu_f16", 0)?,
            concat_time_f16: module.function(c"concat_time_f16", 0)?,
            ln_f16: module.function(c"ln_f16", 0)?,
            modulate_f16: module.function(c"modulate_f16", 0)?,
            mul_weight_f16: module.function(c"mul_weight_f16", 0)?,
            add_bias: module.function(c"add_bias", 0)?,
            silu_mul_fused: module.function(c"silu_mul_fused", 0)?,
            gn_partial: module.function(c"gn_partial", 0)?,
            gn_stats: module.function(c"gn_stats", 0)?,
            gn_apply: module.function(c"gn_apply", 0)?,
            add_f32: module.function(c"add_f32", 0)?,
            module,
        })
    }

    /// Enqueue `op` on the default stream (no wait). `Ok(false)` = a shape
    /// the kernels do not cover.
    pub fn launch(&self, op: &ElementwiseOp) -> Result<bool, String> {
        self.module.bind()?;
        let blocks = |n: u64| n.div_ceil(256) as u32;
        match *op {
            ElementwiseOp::NormModulate {
                x,
                w,
                out,
                rows,
                dim,
                eps,
                ref seg,
            } => {
                let (mut x, mut w, mut out, mut dim, mut eps, mut s) =
                    (x, w, out, dim, eps, Seg::from(seg));
                let mut a: [*mut c_void; 6] = [
                    (&raw mut x).cast(),
                    (&raw mut w).cast(),
                    (&raw mut out).cast(),
                    (&raw mut dim).cast(),
                    (&raw mut eps).cast(),
                    (&raw mut s).cast(),
                ];
                self.module.launch(
                    self.norm_modulate,
                    (rows, 1, 1),
                    256,
                    0,
                    &mut a,
                    "norm_modulate",
                )?;
            }
            ElementwiseOp::QkNormRope {
                headwise,
                q_only,
                qkv,
                q_norm,
                k_norm,
                freqs,
                q,
                k,
                v,
                rows,
                heads,
                head_dim,
                rot_pairs,
                eps,
            } => {
                if !matches!(head_dim, 64 | 128) || 2 * rot_pairs > head_dim || (headwise && q_only)
                {
                    return Ok(false);
                }
                let (mut qkv, mut qw, mut kw, mut freqs, mut q, mut k, mut v) =
                    (qkv, q_norm, k_norm, freqs, q, k, v);
                let (mut heads_, mut rot_pairs, mut eps, mut hw, mut qo) =
                    (heads, rot_pairs, eps, headwise as u32, q_only as u32);
                let mut a: [*mut c_void; 12] = [
                    (&raw mut qkv).cast(),
                    (&raw mut qw).cast(),
                    (&raw mut kw).cast(),
                    (&raw mut freqs).cast(),
                    (&raw mut q).cast(),
                    (&raw mut k).cast(),
                    (&raw mut v).cast(),
                    (&raw mut heads_).cast(),
                    (&raw mut rot_pairs).cast(),
                    (&raw mut eps).cast(),
                    (&raw mut hw).cast(),
                    (&raw mut qo).cast(),
                ];
                self.module.launch(
                    self.qk_norm_rope,
                    (rows, heads, 1),
                    head_dim,
                    0,
                    &mut a,
                    "qk_norm_rope",
                )?;
            }
            ElementwiseOp::GateAdd {
                x,
                y,
                out,
                rows,
                dim,
                ref seg,
            } => {
                let (mut x, mut y, mut out, mut rows_, mut dim, mut s) =
                    (x, y, out, rows, dim, Seg::from(seg));
                let mut a: [*mut c_void; 6] = [
                    (&raw mut x).cast(),
                    (&raw mut y).cast(),
                    (&raw mut out).cast(),
                    (&raw mut rows_).cast(),
                    (&raw mut dim).cast(),
                    (&raw mut s).cast(),
                ];
                self.module.launch(
                    self.gate_add,
                    (blocks(rows as u64 * dim as u64), 1, 1),
                    256,
                    0,
                    &mut a,
                    "gate_add",
                )?;
            }
            ElementwiseOp::AddBias {
                x,
                bias,
                out,
                rows,
                dim,
            } => {
                let (mut x, mut bias, mut out, mut rows_, mut dim_) = (x, bias, out, rows, dim);
                let mut a: [*mut c_void; 5] = [
                    (&raw mut x).cast(),
                    (&raw mut bias).cast(),
                    (&raw mut out).cast(),
                    (&raw mut rows_).cast(),
                    (&raw mut dim_).cast(),
                ];
                self.module.launch(
                    self.add_bias,
                    (blocks(rows as u64 * dim as u64), 1, 1),
                    256,
                    0,
                    &mut a,
                    "add_bias",
                )?;
            }
            ElementwiseOp::SiluMulFused { x, out, rows, h } => {
                let (mut x, mut out, mut rows_, mut h_) = (x, out, rows, h);
                let mut a: [*mut c_void; 4] = [
                    (&raw mut x).cast(),
                    (&raw mut out).cast(),
                    (&raw mut rows_).cast(),
                    (&raw mut h_).cast(),
                ];
                self.module.launch(
                    self.silu_mul_fused,
                    (blocks(rows as u64 * h as u64), 1, 1),
                    256,
                    0,
                    &mut a,
                    "silu_mul_fused",
                )?;
            }
            ElementwiseOp::GroupNormF32 {
                x,
                w,
                bias,
                out,
                scratch,
                c,
                hw,
                groups,
                eps,
                silu,
            } => {
                if groups == 0 || !c.is_multiple_of(groups) || hw == 0 {
                    return Ok(false);
                }
                let mut n_group = c / groups * hw;
                let mut parts = n_group.div_ceil(GROUP_NORM_PART);
                let (mut x, mut part) = (x, scratch);
                let mut stats = scratch + 8 * groups as u64 * parts as u64;
                let mut a: [*mut c_void; 4] = [
                    (&raw mut x).cast(),
                    (&raw mut part).cast(),
                    (&raw mut n_group).cast(),
                    (&raw mut parts).cast(),
                ];
                self.module.launch(
                    self.gn_partial,
                    (groups, parts, 1),
                    256,
                    0,
                    &mut a,
                    "gn_partial",
                )?;
                let (mut groups_, mut eps) = (groups, eps);
                let mut a: [*mut c_void; 6] = [
                    (&raw mut part).cast(),
                    (&raw mut stats).cast(),
                    (&raw mut n_group).cast(),
                    (&raw mut parts).cast(),
                    (&raw mut groups_).cast(),
                    (&raw mut eps).cast(),
                ];
                self.module.launch(
                    self.gn_stats,
                    (groups.div_ceil(256), 1, 1),
                    256,
                    0,
                    &mut a,
                    "gn_stats",
                )?;
                let (mut w, mut b, mut out, mut c_, mut hw_) = (w, bias, out, c, hw);
                let (mut cpg, mut silu) = (c / groups, silu as u32);
                let mut a: [*mut c_void; 9] = [
                    (&raw mut x).cast(),
                    (&raw mut w).cast(),
                    (&raw mut b).cast(),
                    (&raw mut stats).cast(),
                    (&raw mut out).cast(),
                    (&raw mut c_).cast(),
                    (&raw mut hw_).cast(),
                    (&raw mut cpg).cast(),
                    (&raw mut silu).cast(),
                ];
                self.module.launch(
                    self.gn_apply,
                    (blocks(c as u64 * hw as u64), 1, 1),
                    256,
                    0,
                    &mut a,
                    "gn_apply",
                )?;
            }
            ElementwiseOp::AddF32 { x, y, out, n } => {
                let (mut x, mut y, mut out, mut n_) = (x, y, out, n);
                let mut a: [*mut c_void; 4] = [
                    (&raw mut x).cast(),
                    (&raw mut y).cast(),
                    (&raw mut out).cast(),
                    (&raw mut n_).cast(),
                ];
                self.module.launch(
                    self.add_f32,
                    (blocks(n as u64), 1, 1),
                    256,
                    0,
                    &mut a,
                    "add_f32",
                )?;
            }
            ElementwiseOp::LayerNormModulateF16 {
                x,
                scale,
                shift,
                out,
                rows,
                dim,
                eps,
            } => {
                let (mut x, mut sc, mut sh, mut out, mut dim, mut eps) =
                    (x, scale, shift, out, dim, eps);
                let mut a: [*mut c_void; 6] = [
                    (&raw mut x).cast(),
                    (&raw mut sc).cast(),
                    (&raw mut sh).cast(),
                    (&raw mut out).cast(),
                    (&raw mut dim).cast(),
                    (&raw mut eps).cast(),
                ];
                self.module.launch(
                    self.ln_modulate_f16,
                    (rows, 1, 1),
                    256,
                    0,
                    &mut a,
                    "ln_modulate_f16",
                )?;
            }
            ElementwiseOp::RopeF16 {
                x,
                freqs,
                out,
                rows,
                heads,
                pairs,
            } => {
                let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5) =
                    (x, freqs, out, rows, heads, pairs);
                let mut a: [*mut c_void; 6] = [
                    (&raw mut a0).cast(),
                    (&raw mut a1).cast(),
                    (&raw mut a2).cast(),
                    (&raw mut a3).cast(),
                    (&raw mut a4).cast(),
                    (&raw mut a5).cast(),
                ];
                self.module.launch(
                    self.rope_f16,
                    (blocks(rows as u64 * heads as u64 * pairs as u64), 1, 1),
                    256,
                    0,
                    &mut a,
                    "rope_f16",
                )?;
            }
            ElementwiseOp::RmsNormF16 {
                x,
                w,
                out,
                rows,
                dim,
                eps,
            } => {
                let (mut a0, mut a1, mut a2, mut a3, mut a4) = (x, w, out, dim, eps);
                let mut a: [*mut c_void; 5] = [
                    (&raw mut a0).cast(),
                    (&raw mut a1).cast(),
                    (&raw mut a2).cast(),
                    (&raw mut a3).cast(),
                    (&raw mut a4).cast(),
                ];
                self.module.launch(
                    self.rms_norm_f16,
                    (rows, 1, 1),
                    256,
                    0,
                    &mut a,
                    "rms_norm_f16",
                )?;
            }
            ElementwiseOp::LayerNormAffineF16 {
                x,
                w,
                b,
                out,
                rows,
                dim,
                eps,
            } => {
                let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5) = (x, w, b, out, dim, eps);
                let mut a: [*mut c_void; 6] = [
                    (&raw mut a0).cast(),
                    (&raw mut a1).cast(),
                    (&raw mut a2).cast(),
                    (&raw mut a3).cast(),
                    (&raw mut a4).cast(),
                    (&raw mut a5).cast(),
                ];
                self.module.launch(
                    self.ln_affine_f16,
                    (rows, 1, 1),
                    256,
                    0,
                    &mut a,
                    "ln_affine_f16",
                )?;
            }
            ElementwiseOp::AddBiasF16 {
                x,
                bias,
                out,
                rows,
                dim,
            } => {
                let (mut a0, mut a1, mut a2, mut a3, mut a4) = (x, bias, out, rows, dim);
                let mut a: [*mut c_void; 5] = [
                    (&raw mut a0).cast(),
                    (&raw mut a1).cast(),
                    (&raw mut a2).cast(),
                    (&raw mut a3).cast(),
                    (&raw mut a4).cast(),
                ];
                self.module.launch(
                    self.add_bias_f16,
                    (blocks(rows as u64 * dim as u64), 1, 1),
                    256,
                    0,
                    &mut a,
                    "add_bias_f16",
                )?;
            }
            ElementwiseOp::GeluTanhF16 { x, out, n } => {
                let (mut a0, mut a1, mut a2) = (x, out, n);
                let mut a: [*mut c_void; 3] = [
                    (&raw mut a0).cast(),
                    (&raw mut a1).cast(),
                    (&raw mut a2).cast(),
                ];
                self.module.launch(
                    self.gelu_tanh_f16,
                    (blocks(n as u64), 1, 1),
                    256,
                    0,
                    &mut a,
                    "gelu_tanh_f16",
                )?;
            }
            ElementwiseOp::LayerNormF16 {
                x,
                out,
                rows,
                dim,
                eps,
            } => {
                let (mut a0, mut a1, mut a2, mut a3) = (x, out, dim, eps);
                let mut a: [*mut c_void; 4] = [
                    (&raw mut a0).cast(),
                    (&raw mut a1).cast(),
                    (&raw mut a2).cast(),
                    (&raw mut a3).cast(),
                ];
                self.module
                    .launch(self.ln_f16, (rows, 1, 1), 256, 0, &mut a, "ln_f16")?;
            }
            ElementwiseOp::ModulateF16 {
                x,
                scale,
                shift,
                out,
                rows,
                dim,
            } => {
                let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5) =
                    (x, scale, shift, out, rows, dim);
                let mut a: [*mut c_void; 6] = [
                    (&raw mut a0).cast(),
                    (&raw mut a1).cast(),
                    (&raw mut a2).cast(),
                    (&raw mut a3).cast(),
                    (&raw mut a4).cast(),
                    (&raw mut a5).cast(),
                ];
                let n = blocks(rows as u64 * dim as u64);
                self.module
                    .launch(self.modulate_f16, (n, 1, 1), 256, 0, &mut a, "modulate_f16")?;
            }
            ElementwiseOp::MulWeightF16 {
                x,
                w,
                out,
                rows,
                dim,
            } => {
                let (mut a0, mut a1, mut a2, mut a3, mut a4) = (x, w, out, rows, dim);
                let mut a: [*mut c_void; 5] = [
                    (&raw mut a0).cast(),
                    (&raw mut a1).cast(),
                    (&raw mut a2).cast(),
                    (&raw mut a3).cast(),
                    (&raw mut a4).cast(),
                ];
                let n = blocks(rows as u64 * dim as u64);
                self.module.launch(
                    self.mul_weight_f16,
                    (n, 1, 1),
                    256,
                    0,
                    &mut a,
                    "mul_weight_f16",
                )?;
            }
            ElementwiseOp::RmsNorm3dF16 {
                x,
                gamma,
                out,
                n_pos,
                channels,
                stride,
            } => {
                let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5) =
                    (x, gamma, out, n_pos, channels, stride);
                let mut a: [*mut c_void; 6] = [
                    (&raw mut a0).cast(),
                    (&raw mut a1).cast(),
                    (&raw mut a2).cast(),
                    (&raw mut a3).cast(),
                    (&raw mut a4).cast(),
                    (&raw mut a5).cast(),
                ];
                self.module.launch(
                    self.rms_norm3d_f16,
                    (blocks(n_pos as u64), 1, 1),
                    256,
                    0,
                    &mut a,
                    "rms_norm3d_f16",
                )?;
            }
            ElementwiseOp::SiluF16 { x, out, n } => {
                let (mut a0, mut a1, mut a2) = (x, out, n);
                let mut a: [*mut c_void; 3] = [
                    (&raw mut a0).cast(),
                    (&raw mut a1).cast(),
                    (&raw mut a2).cast(),
                ];
                self.module.launch(
                    self.silu_f16,
                    (blocks(n as u64), 1, 1),
                    256,
                    0,
                    &mut a,
                    "silu_f16",
                )?;
            }
            ElementwiseOp::ConcatTimeF16 {
                a,
                b,
                out,
                bc,
                hw,
                a_t,
                b_t,
                a_start,
                a_count,
                b_start,
                b_count,
                a_zero,
            } => {
                let n = bc as u64 * (a_count + b_count) as u64 * hw as u64;
                let (mut a0, mut a1, mut a2) = (a, b, out);
                let mut p: [u32; 9] =
                    [bc, hw, a_t, b_t, a_start, a_count, b_start, b_count, a_zero];
                let mut a: [*mut c_void; 4] = [
                    (&raw mut a0).cast(),
                    (&raw mut a1).cast(),
                    (&raw mut a2).cast(),
                    (&raw mut p).cast(),
                ];
                self.module.launch(
                    self.concat_time_f16,
                    (blocks(n), 1, 1),
                    256,
                    0,
                    &mut a,
                    "concat_time_f16",
                )?;
            }
            ElementwiseOp::AddF16 { x, y, out, n } => {
                let (mut a0, mut a1, mut a2, mut a3) = (x, y, out, n);
                let mut a: [*mut c_void; 4] = [
                    (&raw mut a0).cast(),
                    (&raw mut a1).cast(),
                    (&raw mut a2).cast(),
                    (&raw mut a3).cast(),
                ];
                self.module.launch(
                    self.add_f16,
                    (blocks(n as u64), 1, 1),
                    256,
                    0,
                    &mut a,
                    "add_f16",
                )?;
            }
            ElementwiseOp::GateAddF16 {
                x,
                gate,
                y,
                out,
                rows,
                dim,
            } => {
                let (mut x, mut g, mut y, mut out, mut rows_, mut dim) =
                    (x, gate, y, out, rows, dim);
                let mut a: [*mut c_void; 6] = [
                    (&raw mut x).cast(),
                    (&raw mut g).cast(),
                    (&raw mut y).cast(),
                    (&raw mut out).cast(),
                    (&raw mut rows_).cast(),
                    (&raw mut dim).cast(),
                ];
                self.module.launch(
                    self.gate_add_f16,
                    (blocks(rows as u64 * dim as u64), 1, 1),
                    256,
                    0,
                    &mut a,
                    "gate_add_f16",
                )?;
            }
            ElementwiseOp::Copy { src, dst, bytes } => {
                // SAFETY: device addresses of live buffers the caller owns,
                // on the null stream every kernel here uses.
                crate::nvrtc::drv(
                    unsafe {
                        sys::cuMemcpyDtoDAsync_v2(dst, src, bytes as usize, std::ptr::null_mut())
                    },
                    "copy",
                )?;
            }
        }
        Ok(true)
    }
}
