//! Video sparse attention (VSA) for MiniMax-H3 / FastH3's packed joint
//! sequence, the attention those checkpoints were distilled with (FastVideo
//! `video_sparse_attn_h3`): rows are grouped into 64- or 128-row tiles (text
//! and audio chunked per segment, video as latent cubes), and
//!
//! - every query tile of the dense prefix (text, audio) attends to all keys;
//! - every video query tile attends to all prefix tiles plus the top-k video
//!   tiles by pooled score `mean(q) . mean(k) / sqrt(d)` (per head);
//! - with a learned gate, every row also adds `gate * (softmax(scores) @
//!   mean(v))[its tile]` (the compression branch).
//!
//! Four launches, enqueued in order: tile means, tile scores + compression output, the
//! per-(head, tile) key-tile lists, then a FlashAttention-2 forward over those
//! lists that gathers rows through the tile table (no permute passes) and
//! fuses the gated compression add into its epilogue. Q/K/V/gate/out are bf16
//! `[rows, H, D]`, multiplied as bf16 (bf16 probabilities, as FlashAttention
//! runs bf16 inputs) with f32 accumulate and softmax.

use std::ffi::c_void;
use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, sys};
use thinfer_core::backend::{TileAttnAddrs, TileAttnShape, VsaAddrs, VsaPart, VsaShape, VsaSlab};

use crate::nvrtc::Module;

const D: u32 = 128;
const THREADS: u32 = 128;
/// Query rows per attention block (4 warps x 16), and keys per K/V stage.
const QT: u32 = 64;
const BC: u32 = 32;

// Compiled once per tile size (the FastH3 contract's `vsa_tile_size`): `TILE`
// is defined ahead of this body.
macro_rules! vsa_source {
    () => {
        r#"
#include <cuda_bf16.h>

#define D 128
#define LD (D + 8)
#define NEG_MAX (-3.402823e38f)
#define NEG_INF __int_as_float(0xff800000)

// `tile0`: the launch's first tile (its grid covers a slab of tiles);
// `qrow0`: the packed row at q/gate/out's start (query-side buffers hold just
// the slab's rows); `q_tiles`: tiles in the INT8 Q buffer (one slab's worth).
struct VsaP {
    unsigned rows, heads, n_tiles, n_prefix, topk, has_gate;
    float scale;
    unsigned tile0, qrow0, q_tiles;
};

__device__ __forceinline__ float bf(unsigned short v) { return __uint_as_float((unsigned)v << 16); }
// 2^x in one MUFU op: the attention softmax runs in base 2 on log2(e)-folded
// scores (FlashAttention-2's form; `exp(x) = 2^(x log2 e)`).
__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}
// Tile means of q, k, v: grid (slab tiles, H), one thread per channel.
extern "C" __global__ void __launch_bounds__(128) vsa_pool(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v, const unsigned* __restrict__ tile_rows,
    const unsigned* __restrict__ tile_size, float* __restrict__ pooled, const VsaP p)
{
    const unsigned t = p.tile0 + blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const unsigned n = tile_size[t];
    float sq = 0.f, sk = 0.f, sv = 0.f;
    for (unsigned r = 0; r < n; ++r) {
        const unsigned row = tile_rows[t * TILE + r];
        const size_t off = ((size_t)row * p.heads + h) * D + d;
        sq += bf(q[((size_t)(row - p.qrow0) * p.heads + h) * D + d]);
        sk += bf(k[off]); sv += bf(v[off]);
    }
    const float inv = 1.f / (float)n;
    const size_t o = ((size_t)h * p.n_tiles + t) * D + d;
    const size_t plane = (size_t)p.heads * p.n_tiles * D;
    pooled[o] = sq * inv;
    pooled[plane + o] = sk * inv;
    pooled[2 * plane + o] = sv * inv;
}

__device__ float block_reduce(float x, bool is_max, float* red) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float y = __shfl_xor_sync(0xffffffffu, x, o);
        x = is_max ? fmaxf(x, y) : x + y;
    }
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = x;
    __syncthreads();
    float r = red[0];
    for (int w = 1; w < (int)(blockDim.x >> 5); ++w) r = is_max ? fmaxf(r, red[w]) : r + red[w];
    __syncthreads();
    return r;
}

// Tile scores and the compression output: grid (T, H), smem T floats.
extern "C" __global__ void __launch_bounds__(128) vsa_scores(
    const float* __restrict__ pooled, float* __restrict__ scores, float* __restrict__ coarse, const VsaP p)
{
    extern __shared__ float s[];
    __shared__ __align__(16) float qi[D];
    __shared__ float red[4];
    const unsigned i = blockIdx.x, h = blockIdx.y, T = p.n_tiles;
    const size_t plane = (size_t)p.heads * T * D;
    const float* qp = pooled + (size_t)h * T * D;
    const float* kp = pooled + plane + (size_t)h * T * D;
    const float* vp = pooled + 2 * plane + (size_t)h * T * D;
    qi[threadIdx.x] = qp[(size_t)i * D + threadIdx.x];
    __syncthreads();
    // One warp per key tile, each lane a coalesced float4 slice of the D = 128
    // channels (a thread-per-tile dot product reads 512-byte-strided scalars).
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float4 qv = reinterpret_cast<const float4*>(qi)[lane];
    float mx = NEG_MAX;
    for (unsigned j = warp; j < T; j += blockDim.x >> 5) {
        const float4 kv = reinterpret_cast<const float4*>(kp + (size_t)j * D)[lane];
        float acc = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        acc *= p.scale;
        if (lane == 0) {
            s[j] = acc;
            scores[((size_t)h * T + i) * T + j] = acc;
        }
        mx = fmaxf(mx, acc);
    }
    mx = block_reduce(mx, true, red);
    float sum = 0.f;
    for (unsigned j = threadIdx.x; j < T; j += blockDim.x) {
        const float e = expf(s[j] - mx);
        s[j] = e;
        sum += e;
    }
    sum = block_reduce(sum, false, red);
    const float inv = 1.f / sum;
    float acc = 0.f;
    for (unsigned j = 0; j < T; ++j) acc += s[j] * vp[(size_t)j * D + threadIdx.x];
    coarse[((size_t)h * T + i) * D + threadIdx.x] = acc * inv;
}

// Key-tile lists: grid (T, H), smem T floats. Prefix query tiles take every
// tile; a video query tile takes every prefix tile plus its top-k video tiles
// (ties to the lower index, as a stable descending sort keeps them).
extern "C" __global__ void __launch_bounds__(128) vsa_lists(
    const float* __restrict__ scores, unsigned* __restrict__ lists, unsigned* __restrict__ counts, const VsaP p)
{
    extern __shared__ float vals[];
    __shared__ float bv[4];
    __shared__ unsigned bi[4];
    const unsigned i = blockIdx.x, h = blockIdx.y, T = p.n_tiles, P = p.n_prefix, V = T - P;
    unsigned* list = lists + ((size_t)h * T + i) * T;
    if (i < P || p.topk >= V) {
        for (unsigned j = threadIdx.x; j < T; j += blockDim.x) list[j] = j;
        if (threadIdx.x == 0) counts[h * T + i] = T;
        return;
    }
    const float* row = scores + ((size_t)h * T + i) * T + P;
    for (unsigned j = threadIdx.x; j < V; j += blockDim.x) vals[j] = row[j];
    for (unsigned j = threadIdx.x; j < P; j += blockDim.x) list[j] = j;
    __syncthreads();
    for (unsigned r = 0; r < p.topk; ++r) {
        float best = NEG_INF;
        unsigned idx = 0xffffffffu;
        for (unsigned j = threadIdx.x; j < V; j += blockDim.x) {
            const float x = vals[j];
            if (x > best || (x == best && j < idx)) { best = x; idx = j; }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const unsigned oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ob > best || (ob == best && oi < idx)) { best = ob; idx = oi; }
        }
        if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = best; bi[threadIdx.x >> 5] = idx; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = bv[0];
            unsigned bix = bi[0];
            for (int w = 1; w < 4; ++w)
                if (bv[w] > b || (bv[w] == b && bi[w] < bix)) { b = bv[w]; bix = bi[w]; }
            list[P + r] = P + bix;
            vals[bix] = NEG_INF;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) counts[h * T + i] = P + p.topk;
}

// FlashAttention-2 forward over each query tile's key-tile list: grid
// (T * TILE / QT, H), each block 4 warps x 16 query rows of one tile (a
// 128-row tile as two blocks: one 256-thread block per tile measured 40%
// slower, occupancy), bf16 MMA (bf16 probabilities, as FlashAttention runs
// bf16 inputs). Keys stream in BC = 32 row stages (SPT per tile), double-
// buffered with cp.async straight from the gathered rows, so the next stage
// loads while this one multiplies; the Q rows are staged through stage 0 and
// live in registers. Keys past a tile's size are zero-filled and masked.
// Epilogue adds gate * coarse[tile].
#define BC 32
#define QT 64
#define SPT (TILE / BC)
__device__ __forceinline__ void cp_async16(unsigned short* s, const unsigned short* g, bool pred) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem_addr(s)), "l"(g), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned pack_bf2(float lo, float hi) {
    __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<unsigned*>(&b);
}
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], const unsigned short* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned (&r)[4], const unsigned short* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}

// Head dim is a template parameter from here on (VSA 128, the VAE 64).
#undef D
#undef LD
#define ATTN_ARGS \
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k, \
    const unsigned short* __restrict__ v, const unsigned short* __restrict__ gate, \
    unsigned short* __restrict__ out, const unsigned* __restrict__ tile_rows, \
    const unsigned* __restrict__ tile_size, const unsigned* __restrict__ lists, \
    const unsigned* __restrict__ counts, const float* __restrict__ coarse, const VsaP p

template <int D>
__device__ __forceinline__ void attn_impl(ATTN_ARGS)
{
    extern __shared__ __align__(16) unsigned char smem[];
    // A head dim off the MMA's 16 (the vision tower's 72) computes zero-padded
    // to DP; shared rows are DS wide so the swizzle below stays in the row.
    // Both equal D for 64 and 128.
    constexpr int DP = (D + 15) / 16 * 16, DS = (DP + 63) / 64 * 64;
    // stage s: K [BC][DS] then V [BC][DS]; the Q rows [QT][DS] alias stage 0.
    // 16-byte chunk c of row r sits at chunk c ^ (r & 7), so ldmatrix's eight
    // row addresses hit distinct banks (padding cost the occupancy: 3 blocks
    // per SM fit only without it).
    unsigned short* stage = reinterpret_cast<unsigned short*>(smem);
    constexpr int STAGE = 2 * BC * DS;
    auto at = [](int r, int c8) { return r * DS + ((c8 ^ (r & 7)) << 3); };

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const unsigned i = p.tile0 + blockIdx.x / (TILE / QT), h = blockIdx.y, T = p.n_tiles;
    const unsigned q0 = blockIdx.x % (TILE / QT) * QT, size = tile_size[i];
    // A block past its tile's rows has nothing to do (uniform across the block).
    if (q0 >= size) return;
    const unsigned nq = min(size - q0, (unsigned)QT);
    const unsigned* qrows = tile_rows + i * TILE + q0;
    const size_t rs = (size_t)p.heads * D;
    // q / gate / out hold rows from `qrow0` on (k / v hold every row).
    auto qoff = [&](unsigned row) { return (size_t)(row - p.qrow0) * rs; };
    const unsigned* list = lists + ((size_t)h * T + i) * T;
    const unsigned n_steps = SPT * counts[h * T + i];
    // Ragged tiles skip their padding exactly: a stage with no keys is never
    // loaded, a 16-key block with none is never multiplied, and a warp whose 16
    // query rows are all padding only helps load (all of it contributes 0).
    auto live = [&](unsigned s) { return s % SPT * BC < tile_size[list[s / SPT]]; };
    auto next_live = [&](unsigned s) {
        do ++s; while (s < n_steps && !live(s));
        return s;
    };
    const bool warp_live = (unsigned)(warp * 16) < nq;

    {
        unsigned short* qs = stage;
        for (int x = tid; x < QT * (DP / 8); x += blockDim.x) {
            const int r = x / (DP / 8), c8 = x % (DP / 8);
            const bool ok = (unsigned)r < nq && c8 * 8 < D;
            cp_async16(qs + at(r, c8), q + qoff(qrows[ok ? r : 0]) + h * D + c8 * 8, ok);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }
    unsigned qa[DP / 16][4];
    {
        const int r = warp * 16 + (lane & 15);
        #pragma unroll
        for (int kk = 0; kk < DP / 16; ++kk) ldsm_x4(qa[kk], stage + at(r, kk * 2 + (lane >> 4)));
    }
    __syncthreads(); // stage 0 is free for keys

    // Stage s -> stage buffer.
    auto issue = [&](unsigned s, int buf) {
        const unsigned kt = list[s / SPT], k0 = s % SPT * BC, nk = tile_size[kt];
        unsigned short* ks = stage + buf * STAGE;
        unsigned short* vs = ks + BC * DS;
        for (int x = tid; x < BC * (DP / 8); x += blockDim.x) {
            const int r = x / (DP / 8), c8 = x % (DP / 8);
            const unsigned key = k0 + r;
            const bool ok = key < nk && c8 * 8 < D;
            const size_t off = tile_rows[kt * TILE + (ok ? key : 0)] * rs + h * D + c8 * 8;
            cp_async16(ks + at(r, c8), k + off, ok);
            cp_async16(vs + at(r, c8), v + off, ok);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    float o[DP / 8][4];
    #pragma unroll
    for (int j = 0; j < DP / 8; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m_r[2] = {NEG_MAX, NEG_MAX}, l_r[2] = {0.f, 0.f};
    const float scale2 = p.scale * 1.4426950408889634f;

    if (n_steps > 0) issue(0, 0);
    int buf = 0;
    for (unsigned s = 0; s < n_steps; buf ^= 1) {
        const unsigned nxt = next_live(s);
        if (nxt < n_steps) {
            issue(nxt, buf ^ 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        const unsigned short* ks = stage + buf * STAGE;
        const unsigned short* vs = ks + BC * DS;
        const unsigned k0 = s % SPT * BC, nk = tile_size[list[s / SPT]];
        s = nxt;
        if (!warp_live) {
            __syncthreads();
            continue;
        }

        float sc[BC / 8][4];
        #pragma unroll
        for (int j = 0; j < BC / 8; ++j) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
        #pragma unroll
        for (int kk = 0; kk < DP / 16; ++kk) {
            #pragma unroll
            for (int nj = 0; nj < BC / 16; ++nj) {
                if (k0 + nj * 16 >= nk) continue;
                unsigned b[4];
                ldsm_x4(b, ks + at(nj * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 2 + ((lane >> 3) & 1)));
                mma_bf16(sc[2 * nj], qa[kk], b[0], b[1]);
                mma_bf16(sc[2 * nj + 1], qa[kk], b[2], b[3]);
            }
        }

        float mt[2] = {m_r[0], m_r[1]};
        #pragma unroll
        for (int j = 0; j < BC / 8; ++j) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const unsigned key = k0 + j * 8 + 2 * t + (e & 1);
                const float x = key < nk ? sc[j][e] * scale2 : NEG_MAX;
                sc[j][e] = x;
                mt[e >> 1] = fmaxf(mt[e >> 1], x);
            }
        }
        float alpha[2];
        #pragma unroll
        for (int row = 0; row < 2; ++row) {
            mt[row] = fmaxf(mt[row], __shfl_xor_sync(0xffffffffu, mt[row], 1));
            mt[row] = fmaxf(mt[row], __shfl_xor_sync(0xffffffffu, mt[row], 2));
            alpha[row] = ex2(m_r[row] - mt[row]);
            m_r[row] = mt[row];
            l_r[row] *= alpha[row];
        }
        #pragma unroll
        for (int j = 0; j < BC / 8; ++j) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float pr = ex2(sc[j][e] - m_r[e >> 1]);
                sc[j][e] = pr;
                l_r[e >> 1] += pr;
            }
        }
        unsigned pa[BC / 16][4];
        #pragma unroll
        for (int kb = 0; kb < BC / 16; ++kb) {
            pa[kb][0] = pack_bf2(sc[2 * kb][0], sc[2 * kb][1]);
            pa[kb][1] = pack_bf2(sc[2 * kb][2], sc[2 * kb][3]);
            pa[kb][2] = pack_bf2(sc[2 * kb + 1][0], sc[2 * kb + 1][1]);
            pa[kb][3] = pack_bf2(sc[2 * kb + 1][2], sc[2 * kb + 1][3]);
        }
        #pragma unroll
        for (int j = 0; j < DP / 8; ++j) {
            o[j][0] *= alpha[0]; o[j][1] *= alpha[0];
            o[j][2] *= alpha[1]; o[j][3] *= alpha[1];
        }
        #pragma unroll
        for (int kb = 0; kb < BC / 16; ++kb) {
            if (k0 + kb * 16 >= nk) continue;
            #pragma unroll
            for (int dn = 0; dn < DP / 16; ++dn) {
                unsigned b[4];
                ldsm_x4_t(b, vs + at(kb * 16 + (lane & 7) + (((lane >> 3) & 1) << 3), dn * 2 + (lane >> 4)));
                mma_bf16(o[2 * dn], pa[kb], b[0], b[1]);
                mma_bf16(o[2 * dn + 1], pa[kb], b[2], b[3]);
            }
        }
        __syncthreads(); // this stage is consumed before it is refilled
    }

    float inv[2];
    #pragma unroll
    for (int row = 0; row < 2; ++row) {
        float l = l_r[row];
        l += __shfl_xor_sync(0xffffffffu, l, 1);
        l += __shfl_xor_sync(0xffffffffu, l, 2);
        inv[row] = (l > 0.f) ? 1.f / l : 0.f;
    }
    const float* cz = coarse + ((size_t)h * T + i) * D;
    #pragma unroll
    for (int e = 0; e < 2; ++e) {
        const unsigned r = warp * 16 + g + e * 8;
        if (r >= nq) continue;
        const size_t base = qoff(qrows[r]) + h * D;
        #pragma unroll
        for (int j = 0; j < DP / 8; ++j) {
            const int col = j * 8 + 2 * t;
            if (col >= D) continue;
            float x = o[j][2 * e] * inv[e], y = o[j][2 * e + 1] * inv[e];
            if (p.has_gate) {
                x += bf(gate[base + col]) * cz[col];
                y += bf(gate[base + col + 1]) * cz[col + 1];
            }
            *reinterpret_cast<__nv_bfloat162*>(out + base + col) = __floats2bfloat162_rn(x, y);
        }
    }
}
extern "C" __global__ void __launch_bounds__(128) vsa_attn(ATTN_ARGS)
{ attn_impl<128>(q, k, v, gate, out, tile_rows, tile_size, lists, counts, coarse, p); }

// ---- INT8 QK^T (SageAttention-style), head dim 128 ----
// K is smoothed by its per-head mean over every row: each query's scores
// shift by the same q . mean, which softmax ignores (also under the sparse
// key lists, the shift is per query). Q and the smoothed K are quantized
// symmetric INT8 per (tile, head) into tile-ordered [H][T][TILE][128] buffers;
// QK^T runs on s8 MMA (s32 accumulate), dequantized by q_scale * k_scale.
// Softmax and P.V are the bf16 kernel's.
#define D8 128

// Per-head key mean from the tile means: grid (H), 128 threads.
extern "C" __global__ void __launch_bounds__(128) vsa_kmean(
    const float* __restrict__ pooled, const unsigned* __restrict__ tile_size,
    float* __restrict__ kmean, const VsaP p)
{
    const unsigned h = blockIdx.x, d = threadIdx.x, T = p.n_tiles;
    const float* kp = pooled + (size_t)p.heads * T * D8 + (size_t)h * T * D8;
    float s = 0.f;
    for (unsigned t = 0; t < T; ++t) s += kp[(size_t)t * D8 + d] * (float)tile_size[t];
    kmean[h * D8 + d] = s / (float)p.rows;
}

// Symmetric INT8 of one (tile, head) of `x` (minus `shift` per channel) into
// tile-ordered `dst` at `slot`, scale to `*scale`: 128 threads (one per
// channel, a row per iteration). Rows past the tile's size are zero.
__device__ void quant_tile(const unsigned short* __restrict__ x, size_t row0, float shift,
                           const unsigned* __restrict__ tile_rows, unsigned t, unsigned n,
                           signed char* __restrict__ dst, size_t slot, float* scale, const VsaP& p)
{
    __shared__ float red[4];
    const unsigned h = blockIdx.y, d = threadIdx.x;
    auto at = [&](unsigned r) { return (((size_t)tile_rows[t * TILE + r] - row0) * p.heads + h) * D8 + d; };
    float a = 0.f;
    for (unsigned r = 0; r < n; ++r) a = fmaxf(a, fabsf(bf(x[at(r)]) - shift));
    a = block_reduce(a, true, red);
    const float s = a > 0.f ? a / 127.f : 1.f;
    const size_t base = slot * TILE * D8;
    for (unsigned r = 0; r < TILE; ++r)
        dst[base + (size_t)r * D8 + d] = r < n ? (signed char)__float2int_rn((bf(x[at(r)]) - shift) / s) : 0;
    if (d == 0) *scale = s;
}

// INT8 smoothed K of every tile: grid (T, H), into [H][T][TILE][128].
extern "C" __global__ void __launch_bounds__(128) vsa_quant_k(
    const unsigned short* __restrict__ k, const unsigned* __restrict__ tile_rows,
    const unsigned* __restrict__ tile_size, const float* __restrict__ kmean,
    signed char* __restrict__ ki8, float* __restrict__ scales, const VsaP p)
{
    const unsigned t = blockIdx.x, h = blockIdx.y, T = p.n_tiles;
    quant_tile(k, 0, kmean[h * D8 + threadIdx.x], tile_rows, t, tile_size[t], ki8,
               (size_t)h * T + t, &scales[(size_t)p.heads * T + h * T + t], p);
}

// INT8 Q of one slab's tiles: grid (slab tiles, H), into [H][q_tiles][TILE][128].
extern "C" __global__ void __launch_bounds__(128) vsa_quant_q(
    const unsigned short* __restrict__ q, const unsigned* __restrict__ tile_rows,
    const unsigned* __restrict__ tile_size, signed char* __restrict__ qi8,
    float* __restrict__ scales, const VsaP p)
{
    const unsigned t = p.tile0 + blockIdx.x, h = blockIdx.y;
    quant_tile(q, p.qrow0, 0.f, tile_rows, t, tile_size[t], qi8,
               (size_t)h * p.q_tiles + blockIdx.x, &scales[h * p.n_tiles + t], p);
}

__device__ __forceinline__ void cp_async16b(void* s, const void* g, bool pred) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem_addr(s)), "l"(g), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void mma_s8(int (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm_x4_b(unsigned (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}

// The bf16 kernel's structure with an INT8 K stage ([BC][128 B]) ahead of
// the bf16 V stage ([BC][256 B]); the Q rows (INT8, [QT][128 B]) are staged
// through stage 0. Same grid, ragged-tile skipping and epilogue.
extern "C" __global__ void __launch_bounds__(128) vsa_attn_i8(
    const signed char* __restrict__ qi8, const signed char* __restrict__ ki8,
    const float* __restrict__ scales, const unsigned short* __restrict__ v,
    const unsigned short* __restrict__ gate, unsigned short* __restrict__ out,
    const unsigned* __restrict__ tile_rows, const unsigned* __restrict__ tile_size,
    const unsigned* __restrict__ lists, const unsigned* __restrict__ counts,
    const float* __restrict__ coarse, const VsaP p)
{
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr int KROW = D8, VROW = D8 * 2;           // bytes per K / V row
    constexpr int STAGE = BC * (KROW + VROW);
    // 16-byte chunk c of row r at chunk c ^ (r & 7), as in the bf16 kernel.
    auto at = [](int r, int c, int row) { return r * row + ((c ^ (r & 7)) << 4); };

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const unsigned i = p.tile0 + blockIdx.x / (TILE / QT), h = blockIdx.y, T = p.n_tiles;
    const unsigned q0 = blockIdx.x % (TILE / QT) * QT, size = tile_size[i];
    if (q0 >= size) return;
    const unsigned nq = min(size - q0, (unsigned)QT);
    const unsigned* qrows = tile_rows + i * TILE + q0;
    const size_t rs = (size_t)p.heads * D8;
    // gate / out hold rows from `qrow0` on, INT8 Q the slab's tiles.
    auto qoff = [&](unsigned row) { return (size_t)(row - p.qrow0) * rs; };
    const unsigned* list = lists + ((size_t)h * T + i) * T;
    const unsigned n_steps = SPT * counts[h * T + i];
    auto live = [&](unsigned s) { return s % SPT * BC < tile_size[list[s / SPT]]; };
    auto next_live = [&](unsigned s) {
        do ++s; while (s < n_steps && !live(s));
        return s;
    };
    const bool warp_live = (unsigned)(warp * 16) < nq;
    const float* ks_of = scales + (size_t)p.heads * T + (size_t)h * T;
    const float qscale = scales[h * T + i];

    {
        const signed char* qsrc = qi8 + (((size_t)h * p.q_tiles + (i - p.tile0)) * TILE + q0) * D8;
        for (int x = tid; x < QT * (KROW / 16); x += blockDim.x) {
            const int r = x / (KROW / 16), c = x % (KROW / 16);
            cp_async16b(smem + at(r, c, KROW), qsrc + r * D8 + c * 16, (unsigned)r < nq);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }
    unsigned qa[D8 / 32][4];
    {
        const int r = warp * 16 + (lane & 15);
        #pragma unroll
        for (int kk = 0; kk < D8 / 32; ++kk) ldsm_x4_b(qa[kk], smem + at(r, kk * 2 + (lane >> 4), KROW));
    }
    __syncthreads();

    auto issue = [&](unsigned s, int buf) {
        const unsigned kt = list[s / SPT], k0 = s % SPT * BC, nk = tile_size[kt];
        unsigned char* ks = smem + buf * STAGE;
        unsigned char* vs = ks + BC * KROW;
        const signed char* ksrc = ki8 + (((size_t)h * T + kt) * TILE + k0) * D8;
        for (int x = tid; x < BC * (KROW / 16); x += blockDim.x) {
            const int r = x / (KROW / 16), c = x % (KROW / 16);
            cp_async16b(ks + at(r, c, KROW), ksrc + r * D8 + c * 16, k0 + r < nk);
        }
        for (int x = tid; x < BC * (VROW / 16); x += blockDim.x) {
            const int r = x / (VROW / 16), c = x % (VROW / 16);
            const unsigned key = k0 + r;
            const bool ok = key < nk;
            const size_t off = tile_rows[kt * TILE + (ok ? key : 0)] * rs + h * D8 + c * 8;
            cp_async16b(vs + at(r, c, VROW), v + off, ok);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    float o[D8 / 8][4];
    #pragma unroll
    for (int j = 0; j < D8 / 8; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m_r[2] = {NEG_MAX, NEG_MAX}, l_r[2] = {0.f, 0.f};
    const float scale2 = p.scale * 1.4426950408889634f * qscale;

    if (n_steps > 0) issue(0, 0);
    int buf = 0;
    for (unsigned s = 0; s < n_steps; buf ^= 1) {
        const unsigned nxt = next_live(s);
        if (nxt < n_steps) {
            issue(nxt, buf ^ 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        const unsigned char* ks = smem + buf * STAGE;
        const unsigned char* vs = ks + BC * KROW;
        const unsigned kt = list[s / SPT], k0 = s % SPT * BC, nk = tile_size[kt];
        const float kscale = scale2 * ks_of[kt];
        s = nxt;
        if (!warp_live) {
            __syncthreads();
            continue;
        }

        int si[BC / 8][4];
        #pragma unroll
        for (int j = 0; j < BC / 8; ++j) si[j][0] = si[j][1] = si[j][2] = si[j][3] = 0;
        #pragma unroll
        for (int kk = 0; kk < D8 / 32; ++kk) {
            #pragma unroll
            for (int nj = 0; nj < BC / 16; ++nj) {
                if (k0 + nj * 16 >= nk) continue;
                unsigned b[4];
                ldsm_x4_b(b, ks + at(nj * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 2 + ((lane >> 3) & 1), KROW));
                mma_s8(si[2 * nj], qa[kk], b[0], b[1]);
                mma_s8(si[2 * nj + 1], qa[kk], b[2], b[3]);
            }
        }

        float sc[BC / 8][4];
        float mt[2] = {m_r[0], m_r[1]};
        #pragma unroll
        for (int j = 0; j < BC / 8; ++j) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const unsigned key = k0 + j * 8 + 2 * t + (e & 1);
                const float x = key < nk ? (float)si[j][e] * kscale : NEG_MAX;
                sc[j][e] = x;
                mt[e >> 1] = fmaxf(mt[e >> 1], x);
            }
        }
        float alpha[2];
        #pragma unroll
        for (int row = 0; row < 2; ++row) {
            mt[row] = fmaxf(mt[row], __shfl_xor_sync(0xffffffffu, mt[row], 1));
            mt[row] = fmaxf(mt[row], __shfl_xor_sync(0xffffffffu, mt[row], 2));
            alpha[row] = ex2(m_r[row] - mt[row]);
            m_r[row] = mt[row];
            l_r[row] *= alpha[row];
        }
        #pragma unroll
        for (int j = 0; j < BC / 8; ++j) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float pr = ex2(sc[j][e] - m_r[e >> 1]);
                sc[j][e] = pr;
                l_r[e >> 1] += pr;
            }
        }
        unsigned pa[BC / 16][4];
        #pragma unroll
        for (int kb = 0; kb < BC / 16; ++kb) {
            pa[kb][0] = pack_bf2(sc[2 * kb][0], sc[2 * kb][1]);
            pa[kb][1] = pack_bf2(sc[2 * kb][2], sc[2 * kb][3]);
            pa[kb][2] = pack_bf2(sc[2 * kb + 1][0], sc[2 * kb + 1][1]);
            pa[kb][3] = pack_bf2(sc[2 * kb + 1][2], sc[2 * kb + 1][3]);
        }
        #pragma unroll
        for (int j = 0; j < D8 / 8; ++j) {
            o[j][0] *= alpha[0]; o[j][1] *= alpha[0];
            o[j][2] *= alpha[1]; o[j][3] *= alpha[1];
        }
        #pragma unroll
        for (int kb = 0; kb < BC / 16; ++kb) {
            if (k0 + kb * 16 >= nk) continue;
            #pragma unroll
            for (int dn = 0; dn < D8 / 16; ++dn) {
                unsigned b[4];
                // V holds bf16 pairs: transposed, as in the bf16 kernel.
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
                             : "r"(smem_addr(vs + at(kb * 16 + (lane & 7) + (((lane >> 3) & 1) << 3), dn * 2 + (lane >> 4), VROW))));
                mma_bf16(o[2 * dn], pa[kb], b[0], b[1]);
                mma_bf16(o[2 * dn + 1], pa[kb], b[2], b[3]);
            }
        }
        __syncthreads();
    }

    float inv[2];
    #pragma unroll
    for (int row = 0; row < 2; ++row) {
        float l = l_r[row];
        l += __shfl_xor_sync(0xffffffffu, l, 1);
        l += __shfl_xor_sync(0xffffffffu, l, 2);
        inv[row] = (l > 0.f) ? 1.f / l : 0.f;
    }
    const float* cz = coarse + ((size_t)h * T + i) * D8;
    #pragma unroll
    for (int e = 0; e < 2; ++e) {
        const unsigned r = warp * 16 + g + e * 8;
        if (r >= nq) continue;
        const size_t base = qoff(qrows[r]) + h * D8;
        #pragma unroll
        for (int j = 0; j < D8 / 8; ++j) {
            const int col = j * 8 + 2 * t;
            float x = o[j][2 * e] * inv[e], y = o[j][2 * e + 1] * inv[e];
            if (p.has_gate) {
                x += bf(gate[base + col]) * cz[col];
                y += bf(gate[base + col + 1]) * cz[col + 1];
            }
            *reinterpret_cast<__nv_bfloat162*>(out + base + col) = __floats2bfloat162_rn(x, y);
        }
    }
}

// The caller-built-list pass (VAE, vision tower) runs 64-row tiles only.
#if TILE == 64
extern "C" __global__ void __launch_bounds__(128) vsa_attn_d64(ATTN_ARGS)
{ attn_impl<64>(q, k, v, gate, out, tile_rows, tile_size, lists, counts, coarse, p); }

extern "C" __global__ void __launch_bounds__(128) vsa_attn_d72(ATTN_ARGS)
{ attn_impl<72>(q, k, v, gate, out, tile_rows, tile_size, lists, counts, coarse, p); }
#endif
"#
    };
}
const SOURCE_T64: &str = concat!("#define TILE 64\n", vsa_source!());
const SOURCE_T128: &str = concat!("#define TILE 128\n", vsa_source!());

/// The launch parameters, by value (`struct VsaP`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct VsaP {
    rows: u32,
    heads: u32,
    n_tiles: u32,
    n_prefix: u32,
    topk: u32,
    has_gate: u32,
    scale: f32,
    tile0: u32,
    qrow0: u32,
    q_tiles: u32,
}

/// Scratch bytes for `shape`: pooled q/k/v + compression output (f32
/// `[H, T, D]` each), scores (f32 `[H, T, T]`), lists (u32 `[H, T, T]`) and
/// counts (u32 `[H, T]`).
pub fn scratch_bytes(s: &VsaShape) -> u64 {
    let base = base_scratch(s);
    match int8_layout(s) {
        Some(l) => base + l.total,
        None => base,
    }
}

/// The INT8 path's tile-ordered K (`[H, T, tile, D]` i8) then one slab's Q
/// (`[H, slab_tiles, tile, D]`), a region of its own ([`VsaAddrs::qk_i8`]) so
/// a caller can place it in memory that is dead during the call. 0 on the
/// bf16 path.
pub fn qk_i8_bytes(s: &VsaShape) -> u64 {
    if s.int8_qk {
        k_i8_bytes(s) + i8_tile_bytes(s) * s.slab_tiles as u64
    } else {
        0
    }
}

fn k_i8_bytes(s: &VsaShape) -> u64 {
    i8_tile_bytes(s) * s.n_tiles as u64
}

/// One INT8 tile across every head.
fn i8_tile_bytes(s: &VsaShape) -> u64 {
    s.heads as u64 * s.tile as u64 * s.head_dim as u64
}

/// [`scratch_bytes`] of the bf16 path.
fn base_scratch(s: &VsaShape) -> u64 {
    let (h, t, d) = (s.heads as u64, s.n_tiles as u64, s.head_dim as u64);
    4 * (4 * h * t * d + 2 * h * t * t + h * t)
}

/// The INT8 path's extra scratch, after the bf16 path's: per-head key means
/// (f32 `[H, D]`) then per-(tile, head) Q then K scales (f32 `[2, H, T]`),
/// 16-byte aligned. The INT8 Q/K themselves live in [`qk_i8_bytes`].
struct Int8Layout {
    kmean: u64,
    scales: u64,
    total: u64,
}

fn int8_layout(s: &VsaShape) -> Option<Int8Layout> {
    if !s.int8_qk {
        return None;
    }
    let (h, t, d) = (s.heads as u64, s.n_tiles as u64, s.head_dim as u64);
    let at = |off: u64| off.next_multiple_of(16);
    let kmean = at(base_scratch(s)) - base_scratch(s);
    let scales = at(kmean + 4 * h * d);
    Some(Int8Layout {
        kmean,
        scales,
        total: scales + 8 * h * t,
    })
}

/// The INT8 attention's smem: two stages of INT8 K + bf16 V rows.
const ATTN_I8_SMEM: u32 = 2 * BC * (D + 2 * D);

pub fn accepts(s: &VsaShape) -> bool {
    s.head_dim == D
        && matches!(s.tile, 64 | 128)
        && s.n_tiles > s.n_prefix
        && s.rows > 0
        && s.heads > 0
        && (1..=s.n_tiles).contains(&s.slab_tiles)
}

// The Q rows are staged through K/V stage 0.
const _: () = assert!(QT == 2 * BC);

/// Two double-buffered K/V stages (the Q rows alias stage 0), at the kernel's
/// shared row width (`DS`: the head dim padded to 16, then 64).
fn attn_smem(d: u32) -> u32 {
    2 * 2 * BC * d.next_multiple_of(16).next_multiple_of(64) * 2
}

/// Head dims the attention pass is instantiated for: the DiT's 128, the video
/// VAE's 64, the Qwen3-VL vision tower's 72.
pub fn attention_head_dim_supported(d: u32) -> bool {
    matches!(d, 64 | 72 | 128)
}

/// The VSA forward compiled for one tile size.
struct TileKernels {
    tile: u32,
    module: Module,
    pool: sys::CUfunction,
    scores: sys::CUfunction,
    lists: sys::CUfunction,
    attn: sys::CUfunction,
    kmean: sys::CUfunction,
    quant_k: sys::CUfunction,
    quant_q: sys::CUfunction,
    attn_i8: sys::CUfunction,
}

impl TileKernels {
    fn new(ctx: Arc<CudaContext>, sm: (i32, i32), tile: u32) -> Result<Self, String> {
        let source = match tile {
            64 => SOURCE_T64,
            _ => SOURCE_T128,
        };
        let module = Module::compile(ctx, source, "thinfer_vsa.cu", sm)?;
        let pool = module.function(c"vsa_pool", 0)?;
        // Dynamic smem sized per call (T floats); the attribute caps it.
        let scores = module.function(c"vsa_scores", 48 * 1024)?;
        let lists = module.function(c"vsa_lists", 48 * 1024)?;
        let attn = module.function(c"vsa_attn", attn_smem(D))?;
        let kmean = module.function(c"vsa_kmean", 0)?;
        let quant_k = module.function(c"vsa_quant_k", 0)?;
        let quant_q = module.function(c"vsa_quant_q", 0)?;
        let attn_i8 = module.function(c"vsa_attn_i8", ATTN_I8_SMEM)?;
        Ok(Self {
            tile,
            module,
            pool,
            scores,
            lists,
            attn,
            kmean,
            quant_k,
            quant_q,
            attn_i8,
        })
    }

    /// One step of the forward (see [`VsaPart`]); the scratch tables persist
    /// in `b.scratch` between steps.
    fn part(&self, s: &VsaShape, part: VsaPart, b: &VsaAddrs) -> Result<(), String> {
        let (h, t, d) = (s.heads as u64, s.n_tiles as u64, s.head_dim as u64);
        let mut pooled = b.scratch;
        let mut coarse = pooled + 4 * 3 * h * t * d;
        let mut scores = coarse + 4 * h * t * d;
        let mut lists = scores + 4 * h * t * t;
        let mut counts = lists + 4 * h * t * t;
        let mut tile_rows = b.geometry;
        let mut tile_size = b.geometry + 4 * t * self.tile as u64;
        // (key means, scales, INT8 K, INT8 Q).
        let int8 = int8_layout(s).map(|l| {
            let base = b.scratch + base_scratch(s);
            (
                base + l.kmean,
                base + l.scales,
                b.qk_i8,
                b.qk_i8 + k_i8_bytes(s),
            )
        });
        let slab = match part {
            VsaPart::Pool(x) | VsaPart::Attend(x) => x,
            VsaPart::Select => VsaSlab::whole(s),
        };
        let mut p = VsaP {
            rows: s.rows,
            heads: s.heads,
            n_tiles: s.n_tiles,
            n_prefix: s.n_prefix,
            topk: s.topk,
            has_gate: (b.gate != 0) as u32,
            scale: s.scale,
            tile0: slab.tile0,
            qrow0: slab.row0,
            q_tiles: s.slab_tiles,
        };
        let (mut q, mut k, mut v, mut gate, mut out) = (b.q, b.k, b.v, b.gate, b.out);
        let grid = (s.n_tiles, s.heads, 1);
        let slab_grid = (slab.tiles, s.heads, 1);
        let row_smem = s.n_tiles * 4;
        self.module.bind()?;
        match part {
            VsaPart::Pool(_) => {
                let mut a: [*mut c_void; 7] = [
                    (&raw mut q).cast(),
                    (&raw mut k).cast(),
                    (&raw mut v).cast(),
                    (&raw mut tile_rows).cast(),
                    (&raw mut tile_size).cast(),
                    (&raw mut pooled).cast(),
                    (&raw mut p).cast(),
                ];
                return self
                    .module
                    .launch(self.pool, slab_grid, THREADS, 0, &mut a, "vsa_pool");
            }
            VsaPart::Select => {
                let mut a: [*mut c_void; 4] = [
                    (&raw mut pooled).cast(),
                    (&raw mut scores).cast(),
                    (&raw mut coarse).cast(),
                    (&raw mut p).cast(),
                ];
                self.module
                    .launch(self.scores, grid, THREADS, row_smem, &mut a, "vsa_scores")?;
                let mut a: [*mut c_void; 4] = [
                    (&raw mut scores).cast(),
                    (&raw mut lists).cast(),
                    (&raw mut counts).cast(),
                    (&raw mut p).cast(),
                ];
                self.module
                    .launch(self.lists, grid, THREADS, row_smem, &mut a, "vsa_lists")?;
                let Some((mut kmean, mut scales, mut ki8, _)) = int8 else {
                    return Ok(());
                };
                let mut a: [*mut c_void; 4] = [
                    (&raw mut pooled).cast(),
                    (&raw mut tile_size).cast(),
                    (&raw mut kmean).cast(),
                    (&raw mut p).cast(),
                ];
                self.module
                    .launch(self.kmean, (s.heads, 1, 1), THREADS, 0, &mut a, "vsa_kmean")?;
                let mut a: [*mut c_void; 7] = [
                    (&raw mut k).cast(),
                    (&raw mut tile_rows).cast(),
                    (&raw mut tile_size).cast(),
                    (&raw mut kmean).cast(),
                    (&raw mut ki8).cast(),
                    (&raw mut scales).cast(),
                    (&raw mut p).cast(),
                ];
                return self
                    .module
                    .launch(self.quant_k, grid, THREADS, 0, &mut a, "vsa_quant_k");
            }
            VsaPart::Attend(_) => {}
        }
        let attn_grid = (slab.tiles * (self.tile / QT), s.heads, 1);
        if let Some((_, mut scales, mut ki8, mut qi8)) = int8 {
            let mut a: [*mut c_void; 6] = [
                (&raw mut q).cast(),
                (&raw mut tile_rows).cast(),
                (&raw mut tile_size).cast(),
                (&raw mut qi8).cast(),
                (&raw mut scales).cast(),
                (&raw mut p).cast(),
            ];
            self.module
                .launch(self.quant_q, slab_grid, THREADS, 0, &mut a, "vsa_quant_q")?;
            let mut a: [*mut c_void; 12] = [
                (&raw mut qi8).cast(),
                (&raw mut ki8).cast(),
                (&raw mut scales).cast(),
                (&raw mut v).cast(),
                (&raw mut gate).cast(),
                (&raw mut out).cast(),
                (&raw mut tile_rows).cast(),
                (&raw mut tile_size).cast(),
                (&raw mut lists).cast(),
                (&raw mut counts).cast(),
                (&raw mut coarse).cast(),
                (&raw mut p).cast(),
            ];
            return self.module.launch(
                self.attn_i8,
                attn_grid,
                THREADS,
                ATTN_I8_SMEM,
                &mut a,
                "vsa_attn_i8",
            );
        }
        let mut a: [*mut c_void; 11] = [
            (&raw mut q).cast(),
            (&raw mut k).cast(),
            (&raw mut v).cast(),
            (&raw mut gate).cast(),
            (&raw mut out).cast(),
            (&raw mut tile_rows).cast(),
            (&raw mut tile_size).cast(),
            (&raw mut lists).cast(),
            (&raw mut counts).cast(),
            (&raw mut coarse).cast(),
            (&raw mut p).cast(),
        ];
        self.module.launch(
            self.attn,
            attn_grid,
            THREADS,
            attn_smem(D),
            &mut a,
            "vsa_attn",
        )
    }

    /// The whole forward: pool, select and attend over one slab of every tile.
    fn run(&self, s: &VsaShape, b: &VsaAddrs) -> Result<(), String> {
        let all = VsaSlab::whole(s);
        self.part(s, VsaPart::Pool(all), b)?;
        self.part(s, VsaPart::Select, b)?;
        self.part(s, VsaPart::Attend(all), b)
    }
}

pub struct VsaKernels {
    ctx: Arc<CudaContext>,
    sm: (i32, i32),
    t64: TileKernels,
    /// Compiled on first use (FastH3-Omni only).
    t128: OnceLock<Result<TileKernels, String>>,
    attn_d64: sys::CUfunction,
    attn_d72: sys::CUfunction,
}

// SAFETY: function handles are plain driver handles; every use binds the
// owning context first.
unsafe impl Send for TileKernels {}
unsafe impl Sync for TileKernels {}
unsafe impl Send for VsaKernels {}
unsafe impl Sync for VsaKernels {}

impl VsaKernels {
    pub fn new(ctx: Arc<CudaContext>, sm: (i32, i32)) -> Result<Self, String> {
        let t64 = TileKernels::new(ctx.clone(), sm, 64)?;
        let attn_d64 = t64.module.function(c"vsa_attn_d64", attn_smem(64))?;
        let attn_d72 = t64.module.function(c"vsa_attn_d72", attn_smem(72))?;
        Ok(Self {
            ctx,
            sm,
            t64,
            t128: OnceLock::new(),
            attn_d64,
            attn_d72,
        })
    }

    /// The whole forward. Enqueued only (see `Accelerator::sync`).
    /// `Ok(false)` = shape or alignment not covered.
    pub fn run(&self, s: &VsaShape, b: &VsaAddrs) -> Result<bool, String> {
        let Some(kernels) = self.kernels_for(s, b)? else {
            return Ok(false);
        };
        kernels.run(s, b)?;
        Ok(true)
    }

    /// One step of the forward run in parts (see [`VsaPart`]). Enqueued only.
    pub fn part(&self, s: &VsaShape, part: VsaPart, b: &VsaAddrs) -> Result<bool, String> {
        let slab_ok = match part {
            VsaPart::Pool(x) | VsaPart::Attend(x) => {
                x.tiles > 0 && x.tiles <= s.slab_tiles && x.tile0 + x.tiles <= s.n_tiles
            }
            VsaPart::Select => true,
        };
        let Some(kernels) = self.kernels_for(s, b)?.filter(|_| slab_ok) else {
            return Ok(false);
        };
        kernels.part(s, part, b)?;
        Ok(true)
    }

    fn kernels_for(&self, s: &VsaShape, b: &VsaAddrs) -> Result<Option<&TileKernels>, String> {
        let aligned = [b.q, b.k, b.v, b.out, b.geometry, b.scratch, b.qk_i8]
            .iter()
            .all(|a| a.is_multiple_of(16))
            && b.gate.is_multiple_of(16);
        // Per-row smem score buffers: T floats must fit the 48 KiB cap.
        if !aligned || !accepts(s) || s.n_tiles as u64 * 4 > 48 * 1024 {
            return Ok(None);
        }
        Ok(Some(match s.tile {
            64 => &self.t64,
            _ => self
                .t128
                .get_or_init(|| TileKernels::new(self.ctx.clone(), self.sm, 128))
                .as_ref()
                .map_err(Clone::clone)?,
        }))
    }

    /// The attention pass alone over caller-built 64-row key-tile lists (see
    /// [`TileAttnAddrs`]), head dim 64, 72 or 128, no gate. Enqueued only.
    pub fn run_lists(&self, s: &TileAttnShape, b: &TileAttnAddrs) -> Result<bool, String> {
        let f = match s.head_dim {
            64 => self.attn_d64,
            72 => self.attn_d72,
            128 => self.t64.attn,
            _ => return Ok(false),
        };
        let aligned = [b.q, b.k, b.v, b.out, b.geometry, b.lists]
            .iter()
            .all(|a| a.is_multiple_of(16));
        if !aligned || s.n_tiles == 0 {
            return Ok(false);
        }
        let (h, t) = (s.heads as u64, s.n_tiles as u64);
        let mut p = VsaP {
            rows: s.rows,
            heads: s.heads,
            n_tiles: s.n_tiles,
            scale: s.scale,
            ..Default::default()
        };
        let (mut q, mut k, mut v, mut out) = (b.q, b.k, b.v, b.out);
        let (mut gate, mut coarse) = (0u64, 0u64);
        let mut tile_rows = b.geometry;
        let mut tile_size = b.geometry + 4 * t * self.t64.tile as u64;
        let mut lists = b.lists;
        let mut counts = b.lists + 4 * h * t * t;
        let mut a: [*mut c_void; 11] = [
            (&raw mut q).cast(),
            (&raw mut k).cast(),
            (&raw mut v).cast(),
            (&raw mut gate).cast(),
            (&raw mut out).cast(),
            (&raw mut tile_rows).cast(),
            (&raw mut tile_size).cast(),
            (&raw mut lists).cast(),
            (&raw mut counts).cast(),
            (&raw mut coarse).cast(),
            (&raw mut p).cast(),
        ];
        self.t64.module.bind()?;
        self.t64.module.launch(
            f,
            (s.n_tiles * (self.t64.tile / QT), s.heads, 1),
            THREADS,
            attn_smem(s.head_dim),
            &mut a,
            "tile_attn",
        )?;
        Ok(true)
    }
}
