//! LLM decode / prefill kernels ([`LlmOp`]), the CUDA twins of
//! `thinfer_core::ops::llm`'s WGSL (same math, same uniforms). The routed-
//! expert gemv reads each expert's weight address from a device table, so an
//! expert can live in a VRAM slot or in mapped host memory with no host round
//! trip per layer.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaContext, sys};
use thinfer_core::ops::llm::{LlmOp, LlmWeight};
use thinfer_core::quant::QuantKind;

use crate::nvrtc::Module;

const SOURCE: &str = r#"
#include <cuda_fp16.h>
typedef unsigned char u8;
typedef unsigned int u32;

struct GemvU { u32 n, k, x_stride, x_off, y_stride, y_off, slot_bytes, flags; };
struct RmsU { u32 rows, dim, x_stride, x_off, y_stride, y_off; float eps; u32 flags; };
struct RopeU { u32 rows, heads, row_stride, head_stride, off, half_, p0, p1; };
struct KvU { u32 rows, width, src_stride, src_off, pos0, ring, p1, p2; };
struct AttnU { u32 rows, heads, kv_heads, head_dim, pos0, window; float scale; u32 flags,
               q_stride, q_head_stride, q_off, out_stride, gate_stride, gate_head_stride, gate_off, ring; };
// KV row of absolute position `p` (ring buffer when `ring` > 0).
__device__ __forceinline__ size_t kv_row(u32 p, u32 ring) { return ring ? p % ring : p; }
struct ConvU { u32 rows, channels, kernel, pad; };
struct DgU { u32 rows, heads, p0, p1; };
struct DnU { u32 rows, v_heads, k_heads, conv_width; float eps; u32 p0, p1, p2; };
struct RouterU { u32 rows, experts, top_k, flags; };
struct CombU { u32 rows, dim, top_k, flags; };

__device__ __forceinline__ u32 ld32(const u8* p) {
    return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);
}
__device__ __forceinline__ u32 ld32a(const u8* p) { return *(const u32*)p; }
__device__ __forceinline__ float ldh(const u8* p) {
    unsigned short h = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    return __half2float(__ushort_as_half(h));
}
__device__ __forceinline__ float i8f(u32 b) { return (float)(int)(signed char)(b & 0xFF); }
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// Row bytes per format for k elements.
template <int F> __device__ __forceinline__ u32 row_bytes(u32 k);
template <> __device__ __forceinline__ u32 row_bytes<0>(u32 k) { return k * 4; }
template <> __device__ __forceinline__ u32 row_bytes<1>(u32 k) { return k / 32 * 34; }
template <> __device__ __forceinline__ u32 row_bytes<2>(u32 k) { return k / 32 * 18; }
template <> __device__ __forceinline__ u32 row_bytes<3>(u32 k) { return k / 256 * 144; }
template <> __device__ __forceinline__ u32 row_bytes<4>(u32 k) { return k / 256 * 176; }
template <> __device__ __forceinline__ u32 row_bytes<5>(u32 k) { return k / 256 * 210; }
template <> __device__ __forceinline__ u32 row_bytes<6>(u32 k) { return k / 32 * 24; }
template <> __device__ __forceinline__ u32 row_bytes<7>(u32 k) { return k / 32 * 22; }
template <> __device__ __forceinline__ u32 row_bytes<8>(u32 k) { return k * 2; }

// Eight bf16 weights (16-byte aligned) widened to f32.
__device__ __forceinline__ void bf16x8(const u8* p, float* w) {
    uint4 v = *(const uint4*)p;
    u32 q[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        w[2 * i] = __uint_as_float(q[i] << 16);
        w[2 * i + 1] = __uint_as_float(q[i] & 0xFFFF0000u);
    }
}

// dot of elements [c*256 + l*8, +8) of the row at `r` with x[0..8).
template <int F> __device__ __forceinline__ float dot8(const u8* r, u32 c, u32 l, const float* x);

template <> __device__ __forceinline__ float dot8<0>(const u8* r, u32 c, u32 l, const float* x) {
    const float* w = (const float*)r + c * 256 + l * 8;
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) a += w[i] * x[i];
    return a;
}
template <> __device__ __forceinline__ float dot8<1>(const u8* r, u32 c, u32 l, const float* x) {
    const u8* b = r + (c * 8 + (l >> 2)) * 34;
    float d = ldh(b);
    const u8* q = b + 2 + (l & 3) * 8;
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) a += i8f(q[i]) * x[i];
    return d * a;
}
template <> __device__ __forceinline__ float dot8<2>(const u8* r, u32 c, u32 l, const float* x) {
    const u8* b = r + (c * 8 + (l >> 2)) * 18;
    float d = ldh(b);
    u32 o = (l & 3) * 8;
    u32 byte0 = o >= 16 ? o - 16 : o;
    u32 sh = o >= 16 ? 4 : 0;
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) a += ((float)((b[2 + byte0 + i] >> sh) & 15) - 8.f) * x[i];
    return d * a;
}
__device__ __forceinline__ float2 scale_min_k4(const u8* b, u32 j) {
    if (j < 4) return make_float2((float)(b[4 + j] & 63), (float)(b[8 + j] & 63));
    u32 a = b[8 + j], lo = b[j], hi = b[4 + j];
    return make_float2((float)((a & 0xF) | ((lo >> 6) << 4)), (float)((a >> 4) | ((hi >> 6) << 4)));
}
template <> __device__ __forceinline__ float dot8<3>(const u8* r, u32 c, u32 l, const float* x) {
    const u8* b = r + c * 144;
    float d = ldh(b), dmin = ldh(b + 2);
    u32 j = l >> 2;
    float2 sm = scale_min_k4(b, j);
    u32 sh = (j & 1) * 4;
    const u8* q = b + 16 + (j >> 1) * 32 + (l & 3) * 8;
    u32 q0 = ld32a(q), q1 = ld32a(q + 4);
    float a = 0.f, s = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        a += (float)((q0 >> (8 * i + sh)) & 15) * x[i] + (float)((q1 >> (8 * i + sh)) & 15) * x[4 + i];
        s += x[i] + x[4 + i];
    }
    return d * sm.x * a - dmin * sm.y * s;
}
template <> __device__ __forceinline__ float dot8<4>(const u8* r, u32 c, u32 l, const float* x) {
    const u8* b = r + c * 176;
    float d = ldh(b), dmin = ldh(b + 2);
    u32 j = l >> 2;
    float2 sm = scale_min_k4(b, j);
    u32 sh = (j & 1) * 4;
    u32 o = (l & 3) * 8;
    const u8* q = b + 48 + (j >> 1) * 32 + o;
    u32 q0 = ld32a(q), q1 = ld32a(q + 4);
    u32 h0 = ld32a(b + 16 + o) >> j, h1 = ld32a(b + 20 + o) >> j;
    float a = 0.f, s = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        a += (float)(((q0 >> (8 * i + sh)) & 15) | (((h0 >> (8 * i)) & 1) << 4)) * x[i];
        a += (float)(((q1 >> (8 * i + sh)) & 15) | (((h1 >> (8 * i)) & 1) << 4)) * x[4 + i];
        s += x[i] + x[4 + i];
    }
    return d * sm.x * a - dmin * sm.y * s;
}
template <> __device__ __forceinline__ float dot8<5>(const u8* r, u32 c, u32 l, const float* x) {
    const u8* b = r + c * 210;
    u32 half_ = l >> 4, bk = (l >> 2) & 3, jj = (l & 3) * 8;
    const u8* ql = b + half_ * 64 + (bk & 1) * 32 + jj;
    const u8* qh = b + 128 + half_ * 32 + jj;
    u32 nsh = bk < 2 ? 0 : 4, hsh = bk * 2;
    u32 is = half_ * 8 + bk * 2 + (jj >> 4);
    float sc = i8f(b[192 + is]) * ldh(b + 208);
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int q = (int)(((ql[i] >> nsh) & 15) | (((qh[i] >> hsh) & 3) << 4)) - 32;
        a += (float)q * x[i];
    }
    return sc * a;
}

// Q5_1: element i of a block = nibble of byte i%16 (low i<16, high i>=16)
// with qh bit i as bit 4; f = q * d + m.
template <> __device__ __forceinline__ float dot8<6>(const u8* r, u32 c, u32 l, const float* x) {
    const u8* b = r + (c * 8 + (l >> 2)) * 24;
    float d = ldh(b), m = ldh(b + 2);
    u32 o = (l & 3) * 8;
    u32 qh = ld32(b + 4) >> o;
    u32 byte0 = o >= 16 ? o - 16 : o;
    u32 sh = o >= 16 ? 4 : 0;
    float a = 0.f, s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        a += (float)(((b[8 + byte0 + i] >> sh) & 15) | (((qh >> i) & 1) << 4)) * x[i];
        s += x[i];
    }
    return d * a + m * s;
}

// Q5_0: Q5_1's layout without the min; f = (q - 16) * d.
template <> __device__ __forceinline__ float dot8<7>(const u8* r, u32 c, u32 l, const float* x) {
    const u8* b = r + (c * 8 + (l >> 2)) * 22;
    float d = ldh(b);
    u32 o = (l & 3) * 8;
    u32 qh = ld32(b + 2) >> o;
    u32 byte0 = o >= 16 ? o - 16 : o;
    u32 sh = o >= 16 ? 4 : 0;
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++)
        a += (float)((int)(((b[6 + byte0 + i] >> sh) & 15) | (((qh >> i) & 1) << 4)) - 16) * x[i];
    return d * a;
}
template <> __device__ __forceinline__ float dot8<8>(const u8* r, u32 c, u32 l, const float* x) {
    float w[8];
    bf16x8(r + (c * 256 + l * 8) * 2, w);
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) a += w[i] * x[i];
    return a;
}

// Lanes covering the trailing partial 256-chunk of a k % 256 != 0 row
// (32-element block formats only; K-quant rows are whole super-blocks).
__device__ __forceinline__ bool tail_lane(u32 k, u32 l) { return l * 8 < k % 256; }

template <int F>
__device__ __forceinline__ float row_dot(const u8* row, const float* x, u32 k, u32 l) {
    float acc = 0.f;
    u32 chunks = k / 256 + (tail_lane(k, l) ? 1 : 0);
    for (u32 c = 0; c < chunks; c++) {
        const float* xp = x + c * 256 + l * 8;
        float xv[8];
#pragma unroll
        for (int i = 0; i < 8; i++) xv[i] = xp[i];
        acc += dot8<F>(row, c, l, xv);
    }
    return warp_sum(acc);
}

template <int F>
__device__ void gemv(const u8* w, const float* x, float* y, GemvU u, const float* res) {
    u32 l = threadIdx.x & 31;
    u32 n = blockIdx.x * 8 + (threadIdx.x >> 5);
    u32 p = blockIdx.y;
    if (n >= u.n) return;
    const u8* row = w + (size_t)n * row_bytes<F>(u.k);
    float v = row_dot<F>(row, x + (size_t)p * u.x_stride + u.x_off, u.k, l);
    if (l == 0) {
        size_t o = (size_t)p * u.y_stride + u.y_off + n;
        if (u.flags & 2) v += res[o];
        y[o] = v;
    }
}

template <int F>
__device__ void gemv_experts(const unsigned long long* table, const u32* ids, const float* x, float* y,
                             GemvU u, u32 x_div) {
    u32 l = threadIdx.x & 31;
    u32 n = blockIdx.x * 8 + (threadIdx.x >> 5);
    u32 p = blockIdx.y;
    if (n >= u.n) return;
    const u8* row = (const u8*)table[ids[p]] + (size_t)n * row_bytes<F>(u.k);
    float v = row_dot<F>(row, x + (size_t)(p / x_div) * u.x_stride + u.x_off, u.k, l);
    if (l == 0) y[(size_t)p * u.y_stride + u.y_off + n] = v;
}

// Q8_0 one-row matmul (decode): each lane owns whole 32-element blocks (one
// scale read, 16 two-byte quant loads, 8 float4 activation loads), so a warp
// streams 32 consecutive blocks (~1 KB) per step. Needs 16-byte aligned x
// rows; weight rows are 2-byte aligned (34-byte blocks, even row stride).
extern "C" __global__ void __launch_bounds__(256) gemv_q8_0_blocks(const u8* w, const float* x, float* y, GemvU u,
                                                                 const float* res) {
    u32 l = threadIdx.x & 31;
    u32 n = blockIdx.x * 8 + (threadIdx.x >> 5);
    u32 p = blockIdx.y;
    if (n >= u.n) return;
    u32 nb = u.k / 32;
    const u8* row = w + (size_t)n * nb * 34;
    const float* xr = x + (size_t)p * u.x_stride + u.x_off;
    float acc = 0.f;
    for (u32 b = l; b < nb; b += 32) {
        const u8* blk = row + b * 34;
        const unsigned short* q = (const unsigned short*)(blk + 2);
        const float4* xp = (const float4*)(xr + b * 32);
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            unsigned short a = q[2 * i], c = q[2 * i + 1];
            float4 xv = xp[i];
            s += (float)(signed char)(a & 0xFF) * xv.x + (float)(signed char)(a >> 8) * xv.y
               + (float)(signed char)(c & 0xFF) * xv.z + (float)(signed char)(c >> 8) * xv.w;
        }
        acc += ldh(blk) * s;
    }
    acc = warp_sum(acc);
    if (l == 0) {
        size_t o = (size_t)p * u.y_stride + u.y_off + n;
        if (u.flags & 2) acc += res[o];
        y[o] = acc;
    }
}

#define GEMV(NAME, F) \
extern "C" __global__ void __launch_bounds__(256) gemv_##NAME(const u8* w, const float* x, float* y, GemvU u, const float* res) \
{ gemv<F>(w, x, y, u, res); } \
extern "C" __global__ void __launch_bounds__(256) gemv_experts_##NAME(const unsigned long long* t, const u32* ids, const float* x, float* y, GemvU u, u32 x_div) \
{ gemv_experts<F>(t, ids, x, y, u, x_div); }
GEMV(f32, 0)
GEMV(q8_0, 1)
GEMV(q4_0, 2)
GEMV(q4_k, 3)
GEMV(q5_k, 4)
GEMV(q6_k, 5)
GEMV(q5_1, 6)
GEMV(q5_0, 7)
GEMV(bf16, 8)

__device__ float block_sum(float v, float* sh) {
    v = warp_sum(v);
    u32 w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (l == 0) sh[w] = v;
    __syncthreads();
    float r = l < nw ? sh[l] : 0.f;
    r = warp_sum(r);
    __syncthreads();
    return r;
}
__device__ float block_max(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    u32 w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (l == 0) sh[w] = v;
    __syncthreads();
    float r = l < nw ? sh[l] : -3.0e38f;
    for (int o = 16; o > 0; o >>= 1) r = fmaxf(r, __shfl_xor_sync(0xffffffff, r, o));
    __syncthreads();
    return r;
}

extern "C" __global__ void __launch_bounds__(256) rmsnorm(const float* x, const float* wt, float* y, RmsU u, const float* res) {
    __shared__ float sh[32];
    u32 r = blockIdx.x;
    if (r >= u.rows) return;
    const float* xr = x + (size_t)r * u.x_stride + u.x_off;
    float ss = 0.f;
    for (u32 i = threadIdx.x; i < u.dim; i += blockDim.x) ss += xr[i] * xr[i];
    float inv = rsqrtf(block_sum(ss, sh) / (float)u.dim + u.eps);
    float* yr = y + (size_t)r * u.y_stride + u.y_off;
    for (u32 i = threadIdx.x; i < u.dim; i += blockDim.x) {
        float v = xr[i] * inv;
        if (!(u.flags & 1)) v *= wt[i];
        if (u.flags & 2) v += res[(size_t)r * u.y_stride + u.y_off + i];
        if (u.flags & 4) v *= wt[u.dim];
        yr[i] = v;
    }
}

extern "C" __global__ void rope(float* x, const float* cs, RopeU u) {
    u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= u.rows * u.heads * u.half_) return;
    u32 i = id % u.half_, rh = id / u.half_, h = rh % u.heads, r = rh / u.heads;
    float* b = x + (size_t)r * u.row_stride + h * u.head_stride + u.off;
    float c = cs[(r * u.half_ + i) * 2], s = cs[(r * u.half_ + i) * 2 + 1];
    float x0 = b[i], x1 = b[i + u.half_];
    b[i] = x0 * c - x1 * s;
    b[i + u.half_] = x1 * c + x0 * s;
}

extern "C" __global__ void kv_store(const float* src, __half2* cache, KvU u) {
    u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    u32 hw = u.width / 2;
    if (id >= u.rows * hw) return;
    u32 r = id / hw, i = id % hw;
    const float* s = src + (size_t)r * u.src_stride + u.src_off + 2 * i;
    cache[kv_row(u.pos0 + r, u.ring) * hw + i] = __floats2half2_rn(s[0], s[1]);
}

// One block (256 threads) per (head, row). Keys in blocks of 256: each thread
// scores one key, then each thread accumulates its own output dims.
extern "C" __global__ void __launch_bounds__(256) attention(const float* q, const __half* kc, const __half* vc,
                                                           const float* gate, float* out, AttnU u) {
    __shared__ float qs[512];
    __shared__ float pr[256];
    __shared__ float red[32];
    u32 h = blockIdx.x, t = blockIdx.y, d = u.head_dim, lid = threadIdx.x;
    u32 kvh = h / (u.heads / u.kv_heads);
    const float* qb = q + (size_t)t * u.q_stride + h * u.q_head_stride + u.q_off;
    for (u32 i = lid; i < d; i += 256) qs[i] = qb[i] * u.scale;
    __syncthreads();
    u32 qpos = u.pos0 + t;
    u32 k0 = (u.window > 0 && qpos + 1 > u.window) ? qpos + 1 - u.window : 0;
    size_t kw = (size_t)u.kv_heads * d;
    float m = -3.0e38f, lsum = 0.f;
    float acc0 = 0.f, acc1 = 0.f;
    for (u32 kb = k0; kb <= qpos; kb += 256) {
        u32 key = kb + lid;
        float s = -3.0e38f;
        if (key <= qpos) {
            const __half2* kr = (const __half2*)(kc + kv_row(key, u.ring) * kw + kvh * d);
            float dv = 0.f;
            for (u32 i = 0; i < d / 2; i++) {
                float2 kk = __half22float2(kr[i]);
                dv += qs[2 * i] * kk.x + qs[2 * i + 1] * kk.y;
            }
            s = dv;
        }
        float nm = fmaxf(m, block_max(s, red));
        float p = key <= qpos ? __expf(s - nm) : 0.f;
        pr[lid] = p;
        float bs = block_sum(p, red);
        float corr = __expf(m - nm);
        lsum = lsum * corr + bs;
        m = nm;
        u32 nk = min(256u, qpos + 1 - kb);
        acc0 *= corr;
        acc1 *= corr;
        for (u32 j = 0; j < nk; j++) {
            float pj = pr[j];
            const __half* vb = vc + kv_row(kb + j, u.ring) * kw + kvh * d;
            if (lid < d) acc0 += pj * __half2float(vb[lid]);
            if (lid + 256 < d) acc1 += pj * __half2float(vb[lid + 256]);
        }
        __syncthreads();
    }
    float* ob = out + (size_t)t * u.out_stride + h * d;
    const float* gb = gate + (size_t)t * u.gate_stride + h * u.gate_head_stride + u.gate_off;
    for (int part = 0; part < 2; part++) {
        u32 dd = lid + part * 256;
        if (dd >= d) continue;
        float o = (part == 0 ? acc0 : acc1) / lsum;
        if (u.flags & 1) o = o / (1.f + __expf(-gb[dd]));
        ob[dd] = o;
    }
}

extern "C" __global__ void conv1d_silu(const float* x, const float* wt, float* st, float* y, ConvU u) {
    u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= u.channels) return;
    u32 km = u.kernel - 1;
    float hist[8];
    for (u32 i = 0; i < km; i++) hist[i] = st[c * km + i];
    for (u32 t = 0; t < u.rows; t++) {
        float xv = x[(size_t)t * u.channels + c];
        float acc = wt[c * u.kernel + km] * xv;
        for (u32 i = 0; i < km; i++) acc += wt[c * u.kernel + i] * hist[i];
        for (u32 i = 0; i + 1 < km; i++) hist[i] = hist[i + 1];
        if (km > 0) hist[km - 1] = xv;
        y[(size_t)t * u.channels + c] = acc / (1.f + __expf(-acc));
    }
    for (u32 i = 0; i < km; i++) st[c * km + i] = hist[i];
}

extern "C" __global__ void delta_gates(const float* ab, const float* an, const float* dt, float* gb, DgU u) {
    u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= u.rows * u.heads) return;
    u32 t = id / u.heads, h = id % u.heads, o = t * 2 * u.heads;
    float z = ab[o + h] + dt[h];
    float sp = z > 20.f ? z : log1pf(__expf(z));
    gb[o + h] = an[h] * sp;
    gb[o + u.heads + h] = 1.f / (1.f + __expf(-ab[o + u.heads + h]));
}

// One block of 128 threads per value head; thread j owns state column j.
extern "C" __global__ void __launch_bounds__(128) delta_net(const float* qkv, const float* gb, const float* z,
                                                           const float* nw, float* st, float* out, DnU u) {
    __shared__ float qs[128], ks[128], red[32];
    u32 h = blockIdx.x, j = threadIdx.x;
    u32 kh = h % u.k_heads, kdim = u.k_heads * 128;
    float* sp = st + ((size_t)h * 128 + j) * 128;
    float s[128];
#pragma unroll
    for (int i = 0; i < 128; i++) s[i] = sp[i];
    for (u32 t = 0; t < u.rows; t++) {
        const float* rb = qkv + (size_t)t * u.conv_width;
        float qv = rb[kh * 128 + j], kv = rb[kdim + kh * 128 + j];
        float nq = block_sum(qv * qv, red);
        float nk = block_sum(kv * kv, red);
        qs[j] = qv * rsqrtf(nq + 1e-6f) * rsqrtf(128.f);
        ks[j] = kv * rsqrtf(nk + 1e-6f);
        __syncthreads();
        float vj = rb[2 * kdim + h * 128 + j];
        float e = __expf(gb[t * 2 * u.v_heads + h]);
        float beta = gb[t * 2 * u.v_heads + u.v_heads + h];
        float sk = 0.f, sq = 0.f, kq = 0.f;
#pragma unroll
        for (int i = 0; i < 128; i++) {
            sk += s[i] * ks[i];
            sq += s[i] * qs[i];
            kq += ks[i] * qs[i];
        }
        float delta = (vj - e * sk) * beta;
        float o = e * sq + delta * kq;
#pragma unroll
        for (int i = 0; i < 128; i++) s[i] = e * s[i] + ks[i] * delta;
        float ms = block_sum(o * o, red);
        float zz = z[(size_t)t * u.v_heads * 128 + h * 128 + j];
        out[(size_t)t * u.v_heads * 128 + h * 128 + j] =
            o * rsqrtf(ms / 128.f + u.eps) * nw[j] * (zz / (1.f + __expf(-zz)));
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 128; i++) sp[i] = s[i];
}

extern "C" __global__ void __launch_bounds__(256) router_topk(const float* lg, const float* scale, u32* ids,
                                                              float* wts, RouterU u) {
    __shared__ float pv[1024];
    __shared__ float red[32];
    __shared__ float rv[256];
    __shared__ u32 ri[256];
    u32 r = blockIdx.x, lid = threadIdx.x;
    const float* l = lg + (size_t)r * u.experts;
    float mx = -3.0e38f;
    for (u32 e = lid; e < u.experts; e += 256) mx = fmaxf(mx, l[e]);
    mx = block_max(mx, red);
    float sm = 0.f;
    for (u32 e = lid; e < u.experts; e += 256) { float p = __expf(l[e] - mx); pv[e] = p; sm += p; }
    float total = block_sum(sm, red);
    float picked = 0.f;
    for (u32 k = 0; k < u.top_k; k++) {
        float bv = -1.f;
        u32 bi = 0xFFFFFFFFu;
        for (u32 e = lid; e < u.experts; e += 256) { float p = pv[e]; if (p > bv) { bv = p; bi = e; } }
        rv[lid] = bv;
        ri[lid] = bi;
        __syncthreads();
        for (u32 s = 128; s > 0; s >>= 1) {
            if (lid < s) {
                float ov = rv[lid + s];
                u32 oi = ri[lid + s];
                if (ov > rv[lid] || (ov == rv[lid] && oi < ri[lid])) { rv[lid] = ov; ri[lid] = oi; }
            }
            __syncthreads();
        }
        u32 wi = ri[0];
        float wv = rv[0] / total;
        __syncthreads();
        if (lid == 0) {
            ids[r * u.top_k + k] = wi;
            wts[r * u.top_k + k] = wv;
            pv[wi] = -2.f;
        }
        picked += wv;
        __syncthreads();
    }
    if (lid < u.top_k) {
        float v = wts[r * u.top_k + lid];
        if (u.flags & 1) v /= picked;
        if (u.flags & 2) v *= scale[ids[r * u.top_k + lid]];
        wts[r * u.top_k + lid] = v;
    }
}

// ---- prefill: weights dequantized once per warp, applied to up to MR rows ----
#define MR 8
// Elements [c*256 + l*8, +8) of the row at `r`, dequantized.
template <int F> __device__ __forceinline__ void deq8(const u8* r, u32 c, u32 l, float* w);
template <> __device__ __forceinline__ void deq8<0>(const u8* r, u32 c, u32 l, float* w) {
    const float* s = (const float*)r + c * 256 + l * 8;
#pragma unroll
    for (int i = 0; i < 8; i++) w[i] = s[i];
}
template <> __device__ __forceinline__ void deq8<1>(const u8* r, u32 c, u32 l, float* w) {
    const u8* b = r + (c * 8 + (l >> 2)) * 34;
    float d = ldh(b);
    const u8* q = b + 2 + (l & 3) * 8;
#pragma unroll
    for (int i = 0; i < 8; i++) w[i] = d * i8f(q[i]);
}
template <> __device__ __forceinline__ void deq8<2>(const u8* r, u32 c, u32 l, float* w) {
    const u8* b = r + (c * 8 + (l >> 2)) * 18;
    float d = ldh(b);
    u32 o = (l & 3) * 8;
    u32 byte0 = o >= 16 ? o - 16 : o;
    u32 sh = o >= 16 ? 4 : 0;
#pragma unroll
    for (int i = 0; i < 8; i++) w[i] = d * ((float)((b[2 + byte0 + i] >> sh) & 15) - 8.f);
}
template <> __device__ __forceinline__ void deq8<3>(const u8* r, u32 c, u32 l, float* w) {
    const u8* b = r + c * 144;
    float d = ldh(b), dmin = ldh(b + 2);
    u32 j = l >> 2;
    float2 sm = scale_min_k4(b, j);
    float ds = d * sm.x, dm = dmin * sm.y;
    u32 sh = (j & 1) * 4;
    const u8* q = b + 16 + (j >> 1) * 32 + (l & 3) * 8;
    u32 q0 = ld32a(q), q1 = ld32a(q + 4);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        w[i] = ds * (float)((q0 >> (8 * i + sh)) & 15) - dm;
        w[4 + i] = ds * (float)((q1 >> (8 * i + sh)) & 15) - dm;
    }
}
template <> __device__ __forceinline__ void deq8<4>(const u8* r, u32 c, u32 l, float* w) {
    const u8* b = r + c * 176;
    float d = ldh(b), dmin = ldh(b + 2);
    u32 j = l >> 2;
    float2 sm = scale_min_k4(b, j);
    float ds = d * sm.x, dm = dmin * sm.y;
    u32 sh = (j & 1) * 4;
    u32 o = (l & 3) * 8;
    const u8* q = b + 48 + (j >> 1) * 32 + o;
    u32 q0 = ld32a(q), q1 = ld32a(q + 4);
    u32 h0 = ld32a(b + 16 + o) >> j, h1 = ld32a(b + 20 + o) >> j;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        w[i] = ds * (float)(((q0 >> (8 * i + sh)) & 15) | (((h0 >> (8 * i)) & 1) << 4)) - dm;
        w[4 + i] = ds * (float)(((q1 >> (8 * i + sh)) & 15) | (((h1 >> (8 * i)) & 1) << 4)) - dm;
    }
}
template <> __device__ __forceinline__ void deq8<6>(const u8* r, u32 c, u32 l, float* w) {
    const u8* b = r + (c * 8 + (l >> 2)) * 24;
    float d = ldh(b), m = ldh(b + 2);
    u32 o = (l & 3) * 8;
    u32 qh = ld32(b + 4) >> o;
    u32 byte0 = o >= 16 ? o - 16 : o;
    u32 sh = o >= 16 ? 4 : 0;
#pragma unroll
    for (int i = 0; i < 8; i++) w[i] = (float)(((b[8 + byte0 + i] >> sh) & 15) | (((qh >> i) & 1) << 4)) * d + m;
}
template <> __device__ __forceinline__ void deq8<7>(const u8* r, u32 c, u32 l, float* w) {
    const u8* b = r + (c * 8 + (l >> 2)) * 22;
    float d = ldh(b);
    u32 o = (l & 3) * 8;
    u32 qh = ld32(b + 2) >> o;
    u32 byte0 = o >= 16 ? o - 16 : o;
    u32 sh = o >= 16 ? 4 : 0;
#pragma unroll
    for (int i = 0; i < 8; i++)
        w[i] = (float)((int)(((b[6 + byte0 + i] >> sh) & 15) | (((qh >> i) & 1) << 4)) - 16) * d;
}
template <> __device__ __forceinline__ void deq8<5>(const u8* r, u32 c, u32 l, float* w) {
    const u8* b = r + c * 210;
    u32 half_ = l >> 4, bk = (l >> 2) & 3, jj = (l & 3) * 8;
    const u8* ql = b + half_ * 64 + (bk & 1) * 32 + jj;
    const u8* qh = b + 128 + half_ * 32 + jj;
    u32 nsh = bk < 2 ? 0 : 4, hsh = bk * 2;
    u32 is = half_ * 8 + bk * 2 + (jj >> 4);
    float sc = i8f(b[192 + is]) * ldh(b + 208);
#pragma unroll
    for (int i = 0; i < 8; i++)
        w[i] = sc * (float)((int)(((ql[i] >> nsh) & 15) | (((qh[i] >> hsh) & 3) << 4)) - 32);
}
template <> __device__ __forceinline__ void deq8<8>(const u8* r, u32 c, u32 l, float* w) {
    bf16x8(r + (c * 256 + l * 8) * 2, w);
}

// One warp: `row` against `nr <= MR` activation rows `xs` (16-byte aligned),
// each lane's partial sums in `acc`.
template <int F>
__device__ __forceinline__ void rows_dot(const u8* row, const float* const* xs, u32 nr, u32 k, u32 l,
                                         float* acc) {
    u32 chunks = k / 256 + (tail_lane(k, l) ? 1 : 0);
    for (u32 c = 0; c < chunks; c++) {
        float w[8];
        deq8<F>(row, c, l, w);
#pragma unroll
        for (int r = 0; r < MR; r++) {
            if (r < nr) {
                const float4* xp = (const float4*)(xs[r] + c * 256 + l * 8);
                float4 a = xp[0], b = xp[1];
                acc[r] += w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w
                        + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
            }
        }
    }
}

// Dense matmul for `rows >= 2`: block (8 outputs) x (MR rows).
template <int F>
__device__ void gemv_rows(const u8* w, const float* x, float* y, GemvU u, const float* res, u32 rows) {
    u32 l = threadIdx.x & 31;
    u32 n = blockIdx.x * 8 + (threadIdx.x >> 5);
    u32 r0 = blockIdx.y * MR;
    if (n >= u.n) return;
    u32 nr = min((u32)MR, rows - r0);
    const float* xs[MR];
#pragma unroll
    for (int r = 0; r < MR; r++) xs[r] = x + (size_t)(r0 + min((u32)r, nr - 1)) * u.x_stride + u.x_off;
    float acc[MR];
#pragma unroll
    for (int r = 0; r < MR; r++) acc[r] = 0.f;
    rows_dot<F>(w + (size_t)n * row_bytes<F>(u.k), xs, nr, u.k, l, acc);
#pragma unroll
    for (int r = 0; r < MR; r++) {
        float v = warp_sum(acc[r]);
        if (l == 0 && r < nr) {
            size_t o = (size_t)(r0 + r) * u.y_stride + u.y_off + n;
            if (u.flags & 2) v += res[o];
            y[o] = v;
        }
    }
}

// Routed experts grouped by expert: `tiles` = [count, then per tile
// (expert, pairs in tile, MR pair indices)] from `expert_tiles`.
template <int F>
__device__ void gemv_experts_tiled(const unsigned long long* table, const u32* tiles, const float* x, float* y,
                                   GemvU u, u32 x_div) {
    if (blockIdx.y >= tiles[0]) return;
    const u32* tl = tiles + 1 + blockIdx.y * (MR + 2);
    u32 l = threadIdx.x & 31;
    u32 n = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (n >= u.n) return;
    u32 nr = tl[1];
    const float* xs[MR];
#pragma unroll
    for (int r = 0; r < MR; r++) xs[r] = x + (size_t)(tl[2 + min((u32)r, nr - 1)] / x_div) * u.x_stride + u.x_off;
    float acc[MR];
#pragma unroll
    for (int r = 0; r < MR; r++) acc[r] = 0.f;
    rows_dot<F>((const u8*)table[tl[0]] + (size_t)n * row_bytes<F>(u.k), xs, nr, u.k, l, acc);
#pragma unroll
    for (int r = 0; r < MR; r++) {
        float v = warp_sum(acc[r]);
        if (l == 0 && r < nr) y[(size_t)tl[2 + r] * u.y_stride + u.y_off + n] = v;
    }
}

#define GEMM(NAME, F) \
extern "C" __global__ void __launch_bounds__(256) gemv_rows_##NAME(const u8* w, const float* x, float* y, GemvU u, const float* res, u32 rows) \
{ gemv_rows<F>(w, x, y, u, res, rows); } \
extern "C" __global__ void __launch_bounds__(256) gemv_experts_tiled_##NAME(const unsigned long long* t, const u32* tiles, const float* x, float* y, GemvU u, u32 x_div) \
{ gemv_experts_tiled<F>(t, tiles, x, y, u, x_div); }
GEMM(f32, 0)
GEMM(q8_0, 1)
GEMM(q4_0, 2)
GEMM(q4_k, 3)
GEMM(q5_k, 4)
GEMM(q6_k, 5)
GEMM(q5_1, 6)
GEMM(q5_0, 7)
GEMM(bf16, 8)

// ---- prefill on tensor cores: MT rows x NT outputs per 128-thread block ----
// Weights are dequantized to f16 in shared memory once per (row tile, K
// step); activations go to f16 scaled per (row, K step) by a power of two
// that puts the row's largest magnitude in [2^14, 2^15) (no overflow, full
// f16 precision), and each step's f32 partial sums are unscaled exactly
// before they are accumulated in f32.
#define MT 64
#define NT 64
#define KS 128
#define LDS (KS + 8)

__device__ __forceinline__ void mma16816(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Activation rows `src[r]` (r < nr; the rest repeat the last) against weight
// rows [n0, n0 + NT), written to output rows `dst[r]`.
template <int F>
__device__ void mma_tile(const u8* wb, const float* x, float* y, GemvU u, const float* res, u32 nr,
                         const u32* src, const u32* dst) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* xs = (__half*)smem_raw;
    __half* ws = xs + MT * LDS;
    float* rs = (float*)(ws + NT * LDS);
    u32 tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    u32 g = lane >> 2, t = lane & 3;
    u32 n0 = blockIdx.x * NT;
    size_t rb = row_bytes<F>(u.k);
    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    for (u32 k0 = 0; k0 < u.k; k0 += KS) {
        u32 kv = min((u32)KS, u.k - k0);
        {
            // Two threads per activation row, 64 values each.
            u32 r = tid >> 1, h = tid & 1;
            const float* xr = x + (size_t)src[r] * u.x_stride + u.x_off + k0 + h * 64;
            float amax = 0.f;
            for (u32 i = 0; i < 64; i += 4) {
                if (h * 64 + i < kv) {
                    float4 v = *(const float4*)(xr + i);
                    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
                }
            }
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            int e = 0;
            frexpf(amax, &e);
            int sh = amax > 0.f ? min(15 - e, 100) : 0;
            float sc = ldexpf(1.f, sh);
            if (h == 0) rs[r] = ldexpf(1.f, -sh);
            __half2* d = (__half2*)(xs + r * LDS + h * 64);
            for (u32 i = 0; i < 64; i += 4) {
                float4 v = h * 64 + i < kv ? *(const float4*)(xr + i) : make_float4(0.f, 0.f, 0.f, 0.f);
                d[i / 2] = __floats2half2_rn(v.x * sc, v.y * sc);
                d[i / 2 + 1] = __floats2half2_rn(v.z * sc, v.w * sc);
            }
        }
        // Weights: lane l of a 256-chunk holds elements [l*8, l*8+8); a K
        // step is half a chunk, 16 lanes per weight row.
        u32 c = k0 / 256, lh = (k0 % 256) / 8;
        for (u32 i = tid; i < NT * 16; i += 128) {
            u32 nl = i >> 4, l = lh + (i & 15);
            u32 n = n0 + nl;
            float wv[8];
            if (n < u.n && c * 256 + l * 8 < u.k) {
                deq8<F>(wb + (size_t)n * rb, c, l, wv);
            } else {
#pragma unroll
                for (int q = 0; q < 8; q++) wv[q] = 0.f;
            }
            __half2* d = (__half2*)(ws + nl * LDS + (i & 15) * 8);
#pragma unroll
            for (int q = 0; q < 4; q++) d[q] = __floats2half2_rn(wv[2 * q], wv[2 * q + 1]);
        }
        __syncthreads();
        float part[8][4];
#pragma unroll
        for (int j = 0; j < 8; j++) part[j][0] = part[j][1] = part[j][2] = part[j][3] = 0.f;
        const __half* xa = xs + warp * 16 * LDS;
#pragma unroll
        for (u32 kk = 0; kk < KS; kk += 16) {
            u32 a0 = *(const u32*)(xa + g * LDS + kk + 2 * t);
            u32 a1 = *(const u32*)(xa + (g + 8) * LDS + kk + 2 * t);
            u32 a2 = *(const u32*)(xa + g * LDS + kk + 8 + 2 * t);
            u32 a3 = *(const u32*)(xa + (g + 8) * LDS + kk + 8 + 2 * t);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const __half* wr = ws + (j * 8 + g) * LDS + kk + 2 * t;
                mma16816(part[j], a0, a1, a2, a3, *(const u32*)wr, *(const u32*)(wr + 8));
            }
        }
        float s0 = rs[warp * 16 + g], s1 = rs[warp * 16 + g + 8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            acc[j][0] += part[j][0] * s0;
            acc[j][1] += part[j][1] * s0;
            acc[j][2] += part[j][2] * s1;
            acc[j][3] += part[j][3] * s1;
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            u32 r = warp * 16 + g + hh * 8;
            if (r >= nr) continue;
#pragma unroll
            for (int q = 0; q < 2; q++) {
                u32 n = n0 + j * 8 + 2 * t + q;
                if (n >= u.n) continue;
                size_t o = (size_t)dst[r] * u.y_stride + u.y_off + n;
                float v = acc[j][hh * 2 + q];
                if (res && (u.flags & 2)) v += res[o];
                y[o] = v;
            }
        }
    }
}

template <int F>
__device__ void mma_rows(const u8* w, const float* x, float* y, GemvU u, const float* res, u32 rows) {
    __shared__ u32 idx[MT];
    u32 r0 = blockIdx.y * MT, nr = min((u32)MT, rows - r0);
    if (threadIdx.x < MT) idx[threadIdx.x] = r0 + min((u32)threadIdx.x, nr - 1);
    __syncthreads();
    mma_tile<F>(w, x, y, u, res, nr, idx, idx);
}

// Tiles from `expert_tiles` with tile size MT.
template <int F>
__device__ void mma_experts(const unsigned long long* table, const u32* tiles, const float* x, float* y, GemvU u,
                            u32 x_div) {
    __shared__ u32 src[MT], dst[MT];
    if (blockIdx.y >= tiles[0]) return;
    const u32* tl = tiles + 1 + blockIdx.y * (MT + 2);
    u32 nr = tl[1];
    if (threadIdx.x < MT) {
        u32 p = tl[2 + min((u32)threadIdx.x, nr - 1)];
        src[threadIdx.x] = p / x_div;
        dst[threadIdx.x] = p;
    }
    __syncthreads();
    mma_tile<F>((const u8*)table[tl[0]], x, y, u, nullptr, nr, src, dst);
}

#define MMA(NAME, F) \
extern "C" __global__ void __launch_bounds__(128) mma_rows_##NAME(const u8* w, const float* x, float* y, GemvU u, const float* res, u32 rows) \
{ mma_rows<F>(w, x, y, u, res, rows); } \
extern "C" __global__ void __launch_bounds__(128) mma_experts_##NAME(const unsigned long long* t, const u32* tiles, const float* x, float* y, GemvU u, u32 x_div) \
{ mma_experts<F>(t, tiles, x, y, u, x_div); }
MMA(q8_0, 1)
MMA(q4_0, 2)
MMA(q4_k, 3)
MMA(q5_k, 4)
MMA(q6_k, 5)
MMA(q5_1, 6)
MMA(q5_0, 7)
MMA(bf16, 8)

// Group `pairs` routed ids by expert into tiles of <= mt pairs (one block).
// Pair order inside an expert is arbitrary; every output is computed alone,
// so results do not depend on it.
extern "C" __global__ void __launch_bounds__(256) expert_tiles(const u32* ids, u32 pairs, u32 experts, u32* tiles,
                                                               u32 mt) {
    __shared__ u32 cnt[1024];
    __shared__ u32 tstart[1024];
    for (u32 e = threadIdx.x; e < experts; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (u32 p = threadIdx.x; p < pairs; p += blockDim.x) atomicAdd(&cnt[ids[p]], 1u);
    __syncthreads();
    if (threadIdx.x == 0) {
        u32 at = 0;
        for (u32 e = 0; e < experts; e++) { tstart[e] = at; at += (cnt[e] + mt - 1) / mt; }
        tiles[0] = at;
    }
    __syncthreads();
    for (u32 e = threadIdx.x; e < experts; e += blockDim.x) {
        u32 c = cnt[e];
        for (u32 t = 0; t * mt < c; t++) {
            u32* tl = tiles + 1 + (tstart[e] + t) * (mt + 2);
            tl[0] = e;
            tl[1] = min(mt, c - t * mt);
        }
        cnt[e] = 0;
    }
    __syncthreads();
    for (u32 p = threadIdx.x; p < pairs; p += blockDim.x) {
        u32 e = ids[p];
        u32 slot = atomicAdd(&cnt[e], 1u);
        tiles[1 + (tstart[e] + slot / mt) * (mt + 2) + 2 + slot % mt] = p;
    }
}

// One id from logits[..n] (LlmOp::SampleTopK): k rounds of a block argmax
// (ties: the highest id) pick the candidates in rank order, then thread 0
// replays the host sampler's arithmetic over them in that order.
#define SAMPLE_THREADS 1024
#define SAMPLE_MAX_K 64
extern "C" __global__ void __launch_bounds__(SAMPLE_THREADS) sample_topk(
    const float* logits, u32 n, u32 k, float temp, const float* uniform, u32* out)
{
    __shared__ float bv[SAMPLE_THREADS];
    __shared__ u32 bi[SAMPLE_THREADS];
    __shared__ u32 cand[SAMPLE_MAX_K];
    const u32 t = threadIdx.x;
    const u32 rounds = temp <= 0.f ? 1 : k;
    for (u32 r = 0; r < rounds; ++r) {
        float best = __int_as_float(0xff800000);
        u32 bid = 0xffffffffu;
        for (u32 i = t; i < n; i += SAMPLE_THREADS) {
            bool taken = false;
            for (u32 j = 0; j < r; ++j) taken |= cand[j] == i;
            const float v = logits[i];
            if (!taken && (bid == 0xffffffffu || v > best || (v == best && i > bid))) {
                best = v;
                bid = i;
            }
        }
        bv[t] = best;
        bi[t] = bid;
        __syncthreads();
        for (u32 s = SAMPLE_THREADS / 2; s > 0; s >>= 1) {
            if (t < s) {
                const float ov = bv[t + s];
                const u32 oi = bi[t + s];
                const bool better = oi != 0xffffffffu
                    && (bi[t] == 0xffffffffu || ov > bv[t] || (ov == bv[t] && oi > bi[t]));
                if (better) {
                    bv[t] = ov;
                    bi[t] = oi;
                }
            }
            __syncthreads();
        }
        if (t == 0) cand[r] = bi[0];
        __syncthreads();
    }
    if (t != 0) return;
    if (temp <= 0.f) {
        out[0] = cand[0];
        return;
    }
    const float mx = logits[cand[0]] / temp;
    float p[SAMPLE_MAX_K];
    float sum = 0.f;
    for (u32 i = 0; i < k; ++i) {
        p[i] = expf(logits[cand[i]] / temp - mx);
        sum += p[i];
    }
    float total = 0.f;
    for (u32 i = 0; i < k; ++i) {
        p[i] /= sum;
        total += p[i];
    }
    float rem = uniform[0] * total;
    for (u32 i = 0; i < k; ++i) {
        rem -= p[i];
        if (rem <= 0.f) {
            out[0] = cand[i];
            return;
        }
    }
    out[0] = cand[k - 1];
}

extern "C" __global__ void gather_row_bf16(
    const unsigned short* table, const u32* idx, u32 dim, float* out)
{
    const u32 d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    out[d] = __uint_as_float((u32)table[(size_t)idx[0] * dim + d] << 16);
}

extern "C" __global__ void glu(const float* a, const float* b, float* y, u32 n, u32 gelu) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = a[i], act;
    if (gelu) act = 0.5f * g * (1.f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g)));
    else act = g / (1.f + __expf(-g));
    y[i] = act * b[i];
}

extern "C" __global__ void moe_combine(const float* ex, const float* wts, const float* sh, const float* sg,
                                       const float* res, float* y, CombU u) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= u.rows * u.dim) return;
    u32 r = i / u.dim, d = i % u.dim;
    float v = 0.f;
    for (u32 k = 0; k < u.top_k; k++) v += wts[r * u.top_k + k] * ex[(size_t)(r * u.top_k + k) * u.dim + d];
    if (u.flags & 1) v += sh[i] / (1.f + __expf(-sg[r]));
    if (u.flags & 2) v += res[i];
    y[i] = v;
}
"#;

/// Weight formats with kernels, in [`fmt_index`] order.
const FORMATS: usize = 9;
const FORMAT_NAMES: [&str; FORMATS] = [
    "f32", "q8_0", "q4_0", "q4_k", "q5_k", "q6_k", "q5_1", "q5_0", "bf16",
];

fn fmt_index(f: LlmWeight) -> usize {
    match f {
        LlmWeight::F32 => 0,
        LlmWeight::Quant(QuantKind::Q8_0) => 1,
        LlmWeight::Quant(QuantKind::Q4_0) => 2,
        LlmWeight::Quant(QuantKind::Q4_K) => 3,
        LlmWeight::Quant(QuantKind::Q5_K) => 4,
        LlmWeight::Quant(QuantKind::Q6_K) => 5,
        LlmWeight::Quant(QuantKind::Q5_1) => 6,
        LlmWeight::Quant(QuantKind::Q5_0) => 7,
        LlmWeight::Bf16 => 8,
    }
}

/// Rows per prefill tile (`MR` in the kernels).
const MR: u32 = 8;
/// Rows / outputs per tensor-core prefill tile (`MT` / `NT`), and its
/// dynamic shared memory: f16 activation + weight tiles of `KS` (128) plus
/// padding, and a row scale each.
const MT: u32 = 64;
const NT: u32 = 64;
const MMA_SMEM: u32 = (MT + NT) * (128 + 8) * 2 + MT * 4;
/// Expert ids the tiling kernel indexes (its shared counters; the router
/// kernel declines more).
const EXPERTS_MAX: u32 = 1024;
/// Expert tile scratch: `[count, tiles x (expert, n, MR pairs)]`, sized for
/// the largest routed-pair count the tiled path takes.
const TILE_PAIRS_MAX: u32 = 1 << 15;
const TILE_WORDS: u64 = 1 + TILE_PAIRS_MAX as u64 * (MR as u64 + 2);
// The MT tiling needs fewer words: at most pairs / MT + experts tiles.
const _: () = assert!(
    (TILE_PAIRS_MAX / MT + EXPERTS_MAX) as u64 * (MT as u64 + 2) < TILE_WORDS,
    "tile scratch too small for MT tiles"
);

pub struct LlmKernels {
    module: Module,
    gemv: [sys::CUfunction; FORMATS],
    gemv_experts: [sys::CUfunction; FORMATS],
    gemv_rows: [sys::CUfunction; FORMATS],
    gemv_experts_tiled: [sys::CUfunction; FORMATS],
    /// Tensor-core prefill twins (`None` for f32 weights).
    mma_rows: [Option<sys::CUfunction>; FORMATS],
    mma_experts: [Option<sys::CUfunction>; FORMATS],
    /// `THINFER_LLM_MMA=off` keeps the CUDA-core prefill kernels (A/B).
    mma: bool,
    expert_tiles: sys::CUfunction,
    gemv_q8_0_blocks: sys::CUfunction,
    /// Device scratch for `expert_tiles` (stream ordered: each tiled op
    /// rebuilds it before its own gemv).
    tiles: sys::CUdeviceptr,
    rmsnorm: sys::CUfunction,
    rope: sys::CUfunction,
    kv_store: sys::CUfunction,
    attention: sys::CUfunction,
    conv1d_silu: sys::CUfunction,
    delta_gates: sys::CUfunction,
    delta_net: sys::CUfunction,
    router: sys::CUfunction,
    glu: sys::CUfunction,
    moe_combine: sys::CUfunction,
    sample_topk: sys::CUfunction,
    gather_row_bf16: sys::CUfunction,
}

// SAFETY: CUfunction handles are plain driver handles, valid for the module's
// (and context's) lifetime; launches bind the context first.
unsafe impl Send for LlmKernels {}
unsafe impl Sync for LlmKernels {}

impl Drop for LlmKernels {
    fn drop(&mut self) {
        if self.module.bind().is_ok() {
            // SAFETY: allocated in `new`; the module's users are gone.
            unsafe { sys::cuMemFree_v2(self.tiles) };
        }
    }
}

/// Activation offsets / strides (in f32s) that keep 16-byte vector loads
/// aligned.
fn aligned4(v: &[u32]) -> bool {
    v.iter().all(|x| x.is_multiple_of(4))
}

fn grid1(n: u32) -> (u32, u32, u32) {
    (n.div_ceil(256).max(1), 1, 1)
}

/// `&raw mut` of each argument, as `cuLaunchKernel` wants.
macro_rules! args {
    ($($a:ident),*) => { [$((&raw mut $a).cast::<c_void>()),*] };
}

impl LlmKernels {
    pub fn new(ctx: Arc<CudaContext>, sm: (i32, i32)) -> Result<Self, String> {
        let module = Module::compile(ctx, SOURCE, "thinfer_llm.cu", sm)?;
        let mut gemv = [std::ptr::null_mut(); FORMATS];
        let mut gemv_experts = [std::ptr::null_mut(); FORMATS];
        let mut gemv_rows = [std::ptr::null_mut(); FORMATS];
        let mut gemv_experts_tiled = [std::ptr::null_mut(); FORMATS];
        let mut mma_rows = [None; FORMATS];
        let mut mma_experts = [None; FORMATS];
        for (i, n) in FORMAT_NAMES.iter().enumerate() {
            let f = |name: String| module.function(&std::ffi::CString::new(name).unwrap(), 0);
            gemv[i] = f(format!("gemv_{n}"))?;
            gemv_experts[i] = f(format!("gemv_experts_{n}"))?;
            gemv_rows[i] = f(format!("gemv_rows_{n}"))?;
            gemv_experts_tiled[i] = f(format!("gemv_experts_tiled_{n}"))?;
            if *n != "f32" {
                let g = |name: String| {
                    module.function(&std::ffi::CString::new(name).unwrap(), MMA_SMEM)
                };
                mma_rows[i] = Some(g(format!("mma_rows_{n}"))?);
                mma_experts[i] = Some(g(format!("mma_experts_{n}"))?);
            }
        }
        module.bind()?;
        let mut tiles = 0;
        // SAFETY: plain allocation on the bound context; freed in Drop.
        crate::nvrtc::drv(
            unsafe { sys::cuMemAlloc_v2(&mut tiles, (TILE_WORDS * 4) as usize) },
            "cuMemAlloc",
        )?;
        Ok(Self {
            gemv,
            gemv_experts,
            gemv_rows,
            gemv_experts_tiled,
            mma_rows,
            mma_experts,
            mma: std::env::var("THINFER_LLM_MMA").map_or(true, |v| v != "off"),
            expert_tiles: module.function(c"expert_tiles", 0)?,
            gemv_q8_0_blocks: module.function(c"gemv_q8_0_blocks", 0)?,
            tiles,
            rmsnorm: module.function(c"rmsnorm", 0)?,
            rope: module.function(c"rope", 0)?,
            kv_store: module.function(c"kv_store", 0)?,
            attention: module.function(c"attention", 0)?,
            conv1d_silu: module.function(c"conv1d_silu", 0)?,
            delta_gates: module.function(c"delta_gates", 0)?,
            delta_net: module.function(c"delta_net", 0)?,
            router: module.function(c"router_topk", 0)?,
            glu: module.function(c"glu", 0)?,
            moe_combine: module.function(c"moe_combine", 0)?,
            sample_topk: module.function(c"sample_topk", 0)?,
            gather_row_bf16: module.function(c"gather_row_bf16", 0)?,
            module,
        })
    }

    /// Enqueue `op` on the default stream. `Ok(false)` = unsupported shape.
    pub fn run(&self, op: &LlmOp) -> Result<bool, String> {
        self.module.bind()?;
        let m = &self.module;
        match *op {
            LlmOp::Gemv {
                mut u,
                fmt,
                mut w,
                mut x,
                mut y,
                mut res,
                rows,
            } => {
                if !u.k.is_multiple_of(32) {
                    return Ok(false);
                }
                if let Some(f) = self.mma_rows[fmt_index(fmt)].filter(|_| self.mma && rows > 2 * MR)
                    && aligned4(&[u.x_off, u.x_stride])
                {
                    // Tensor cores: f16 tiles of MT rows x NT outputs.
                    let mut rows = rows;
                    let grid = (u.n.div_ceil(NT), rows.div_ceil(MT), 1);
                    let mut a = args!(w, x, y, u, res, rows);
                    m.launch(f, grid, 128, MMA_SMEM, &mut a, "llm mma rows")?;
                    return Ok(true);
                }
                if rows >= 2 && aligned4(&[u.x_off, u.x_stride]) {
                    // Each weight chunk is dequantized once for MR rows.
                    let mut rows = rows;
                    let grid = (u.n.div_ceil(8), rows.div_ceil(MR), 1);
                    let mut a = args!(w, x, y, u, res, rows);
                    m.launch(
                        self.gemv_rows[fmt_index(fmt)],
                        grid,
                        256,
                        0,
                        &mut a,
                        "llm gemv rows",
                    )?;
                    return Ok(true);
                }
                let f = match fmt {
                    LlmWeight::Quant(QuantKind::Q8_0) if aligned4(&[u.x_off, u.x_stride]) => {
                        self.gemv_q8_0_blocks
                    }
                    _ => self.gemv[fmt_index(fmt)],
                };
                let mut a = args!(w, x, y, u, res);
                m.launch(f, (u.n.div_ceil(8), rows, 1), 256, 0, &mut a, "llm gemv")?;
            }
            LlmOp::GemvExperts {
                mut u,
                fmt,
                mut table,
                mut ids,
                mut x,
                mut y,
                pairs,
                mut x_div,
            } => {
                if !u.k.is_multiple_of(32) {
                    return Ok(false);
                }
                if pairs > 2 * MR && pairs <= TILE_PAIRS_MAX && aligned4(&[u.x_off, u.x_stride]) {
                    // Prefill: pairs grouped by expert, so each expert row is
                    // dequantized once per tile of up to MR pairs.
                    // Ids are < EXPERTS_MAX (the router kernel's limit).
                    let mma = self.mma_experts[fmt_index(fmt)].filter(|_| self.mma);
                    let (mut pairs_, mut experts, mut tiles) = (pairs, EXPERTS_MAX, self.tiles);
                    let mut mt = if mma.is_some() { MT } else { MR };
                    let mut a = args!(ids, pairs_, experts, tiles, mt);
                    m.launch(
                        self.expert_tiles,
                        (1, 1, 1),
                        256,
                        0,
                        &mut a,
                        "llm expert tiles",
                    )?;
                    if let Some(f) = mma {
                        // At most one partial tile per expert; blocks past
                        // the count exit.
                        let max_tiles = pairs.min(pairs / MT + EXPERTS_MAX);
                        let mut a = args!(table, tiles, x, y, u, x_div);
                        m.launch(
                            f,
                            (u.n.div_ceil(NT), max_tiles, 1),
                            128,
                            MMA_SMEM,
                            &mut a,
                            "llm mma experts",
                        )?;
                        return Ok(true);
                    }
                    // Every tile holds >= 1 pair; blocks past the count exit.
                    let max_tiles = pairs;
                    let mut a = args!(table, tiles, x, y, u, x_div);
                    m.launch(
                        self.gemv_experts_tiled[fmt_index(fmt)],
                        (u.n.div_ceil(8), max_tiles, 1),
                        256,
                        0,
                        &mut a,
                        "llm gemv experts tiled",
                    )?;
                    return Ok(true);
                }
                let mut a = args!(table, ids, x, y, u, x_div);
                m.launch(
                    self.gemv_experts[fmt_index(fmt)],
                    (u.n.div_ceil(8), pairs, 1),
                    256,
                    0,
                    &mut a,
                    "llm gemv experts",
                )?;
            }
            LlmOp::RmsNorm {
                mut u,
                mut x,
                mut w,
                mut y,
                mut res,
            } => {
                let mut a = args!(x, w, y, u, res);
                m.launch(self.rmsnorm, (u.rows, 1, 1), 256, 0, &mut a, "llm rmsnorm")?;
            }
            LlmOp::Rope {
                mut u,
                mut x,
                mut cos_sin,
            } => {
                let n = u.rows * u.heads * u.half;
                let mut a = args!(x, cos_sin, u);
                m.launch(self.rope, grid1(n), 256, 0, &mut a, "llm rope")?;
            }
            LlmOp::KvStore {
                mut u,
                mut src,
                mut cache,
            } => {
                let n = u.rows * u.width / 2;
                let mut a = args!(src, cache, u);
                m.launch(self.kv_store, grid1(n), 256, 0, &mut a, "llm kv store")?;
            }
            LlmOp::Attention {
                mut u,
                mut q,
                mut k_cache,
                mut v_cache,
                mut gate,
                mut out,
            } => {
                if u.head_dim > 512 || !u.head_dim.is_multiple_of(2) {
                    return Ok(false);
                }
                let mut a = args!(q, k_cache, v_cache, gate, out, u);
                m.launch(
                    self.attention,
                    (u.heads, u.rows, 1),
                    256,
                    0,
                    &mut a,
                    "llm attention",
                )?;
            }
            LlmOp::Conv1dSilu {
                mut u,
                mut x,
                mut w,
                mut state,
                mut y,
            } => {
                if u.kernel > 9 {
                    return Ok(false);
                }
                let mut a = args!(x, w, state, y, u);
                m.launch(
                    self.conv1d_silu,
                    grid1(u.channels),
                    256,
                    0,
                    &mut a,
                    "llm conv1d",
                )?;
            }
            LlmOp::DeltaGates {
                mut u,
                mut ab,
                mut a_neg,
                mut dt_bias,
                mut gb,
            } => {
                let n = u.rows * u.heads;
                let mut a = args!(ab, a_neg, dt_bias, gb, u);
                m.launch(
                    self.delta_gates,
                    grid1(n),
                    256,
                    0,
                    &mut a,
                    "llm delta gates",
                )?;
            }
            LlmOp::DeltaNet {
                mut u,
                mut qkv,
                mut gb,
                mut z,
                mut norm_w,
                mut state,
                mut out,
            } => {
                let heads = u.v_heads;
                let mut a = args!(qkv, gb, z, norm_w, state, out, u);
                m.launch(
                    self.delta_net,
                    (heads, 1, 1),
                    128,
                    0,
                    &mut a,
                    "llm delta net",
                )?;
            }
            LlmOp::Router {
                mut u,
                mut logits,
                mut scale,
                mut ids,
                mut weights,
            } => {
                if u.experts > 1024 {
                    return Ok(false);
                }
                let mut a = args!(logits, scale, ids, weights, u);
                m.launch(self.router, (u.rows, 1, 1), 256, 0, &mut a, "llm router")?;
            }
            LlmOp::Glu {
                mut n,
                gelu,
                mut a,
                mut b,
                mut y,
            } => {
                let mut g = gelu as u32;
                let count = n;
                let mut ar = args!(a, b, y, n, g);
                m.launch(self.glu, grid1(count), 256, 0, &mut ar, "llm glu")?;
            }
            LlmOp::SampleTopK {
                mut logits,
                mut n,
                mut k,
                temperature,
                mut uniform,
                mut out,
            } => {
                if k == 0 || k > 64 {
                    return Ok(false);
                }
                let mut tp = temperature;
                let mut ar = args!(logits, n, k, tp, uniform, out);
                m.launch(self.sample_topk, (1, 1, 1), 1024, 0, &mut ar, "llm sample")?;
            }
            LlmOp::GatherRowBf16 {
                mut table,
                mut idx,
                mut dim,
                mut out,
            } => {
                let count = dim;
                let mut ar = args!(table, idx, dim, out);
                m.launch(
                    self.gather_row_bf16,
                    grid1(count),
                    256,
                    0,
                    &mut ar,
                    "llm gather",
                )?;
            }
            LlmOp::MoeCombine {
                mut u,
                mut experts,
                mut weights,
                mut shared,
                mut sg,
                mut res,
                mut y,
            } => {
                let n = u.rows * u.dim;
                let mut a = args!(experts, weights, shared, sg, res, y, u);
                m.launch(
                    self.moe_combine,
                    grid1(n),
                    256,
                    0,
                    &mut a,
                    "llm moe combine",
                )?;
            }
        }
        Ok(true)
    }
}
