//! The small ops around a MiniMax-H3 block's matmuls and attention, as CUDA
//! kernels, so a whole block runs on one stream with no wgpu round trips
//! ([`thinfer_core::backend::ElementwiseOp`]). Each matches the wgpu op it
//! replaces, including where that path rounds to bf16 between steps: bf16
//! `[rows, C]` activations, f32 math, round-to-nearest-even stores.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaContext, sys};
use thinfer_core::backend::{ElementwiseOp, MAX_SEGMENTS, RowSegments};

use crate::nvrtc::Module;

const SOURCE: &str = r#"
#include <cuda_bf16.h>

#define MAX_SEG 16

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
    add_bias: sys::CUfunction,
    silu_mul_fused: sys::CUfunction,
}

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
            add_bias: module.function(c"add_bias", 0)?,
            silu_mul_fused: module.function(c"silu_mul_fused", 0)?,
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
        }
        Ok(true)
    }
}
