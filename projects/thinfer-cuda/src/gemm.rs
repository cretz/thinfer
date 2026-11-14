//! Dense 16-bit matmul on tensor cores with exactly the wgpu dense-matmul
//! contract and the coopmat one ([`thinfer_core::ops::GemmDesc`]): A `[M,K]`
//! row-major f16 or bf16, B f16/bf16 as `[N,K]` (n-major, the dequant-once
//! workspace) or `[K,N]` (k-major, loaded dense weights), out `[M,N]` in A's
//! dtype (or f32, the f16 coopmat F32-out form), f32 accumulate, `k` from the
//! `(m, n, k, _)` dims uniform.
//!
//! `mma.sync.m16n8k16` with `ldmatrix` operand loads. A block owns a 128x128
//! output tile (8 warps as 2x4, 64x32 per warp) and walks K in 32-wide
//! strips through double-buffered padded shared memory; the next strip is
//! fetched into registers while the current one multiplies, so one barrier
//! per strip orders everything. The multiply runs in A's dtype: a bf16 B
//! on f16 acts is converted while staging (bf16 A with f16 B declines: the
//! narrowing would drop weight mantissa the wgpu op keeps). That conversion
//! can overflow (bf16-trained weights may exceed 65504), so it raises a
//! device flag and the call declines to the wgpu op, which then overwrites
//! the output: never a silent inf.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaContext, sys};
use thinfer_core::backend::{Conv2dAddrs, Conv3dAddrs, GemmAddrs};
use thinfer_core::ops::{GemmDesc, GemmType};

use crate::nvrtc::Module;

const BM: u32 = 128;
const BN: u32 = 128;
const THREADS: u32 = 256;
/// Below this many rows the tile is mostly empty and the wgpu kernel wins.
pub const MIN_M: u32 = 64;

const SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define BM 128
#define BN 128
#define BK 32
// Tiles are rasterized in groups of GROUP_M row tiles over a 1-D grid, so the
// blocks resident at once share a few A row tiles and B column tiles in L2 (a
// weight wider than L2 is otherwise re-streamed once per row tile).
#define GROUP_M 8
#define LDA (BK + 8)
#define LDBN (BK + 8)
#define LDBK (BN + 8)

struct Dims { unsigned m, n, k, pad; };

__device__ unsigned g_overflow;

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], const unsigned short* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned (&r)[4], const unsigned short* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}
template <bool BF16>
__device__ __forceinline__ void mma16816(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    if (BF16)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    else
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Convert 8 packed 16-bit B values to the multiply dtype.
template <bool C_BF16, bool B_BF16>
__device__ __forceinline__ uint4 conv8(uint4 v, bool& over) {
    if (C_BF16 == B_BF16) return v;
    unsigned w[4] = {v.x, v.y, v.z, v.w};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float lo, hi;
        if (B_BF16) {
            lo = __uint_as_float(w[i] << 16);
            hi = __uint_as_float(w[i] & 0xffff0000u);
            over |= fabsf(lo) > 65504.f || fabsf(hi) > 65504.f;
            __half2 h = __floats2half2_rn(lo, hi);
            w[i] = *reinterpret_cast<unsigned*>(&h);
        } else {
            __half2 h = *reinterpret_cast<__half2*>(&w[i]);
            float2 f = __half22float2(h);
            __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
            w[i] = *reinterpret_cast<unsigned*>(&b);
        }
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
}

// Optional fused epilogue (null pointers = absent): out = resid + gate[col] *
// (acc + bias[col]), one rounding, bf16 output only but for the bf16 bias,
// which an f16 output takes too. `resid` may alias `out` (each element is
// read then written by the same thread).
struct Epi { const unsigned short* bias; const unsigned short* resid; const unsigned short* gate; };

__device__ __forceinline__ float bf_lo(unsigned v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(unsigned v) { return __uint_as_float(v & 0xffff0000u); }

template <bool C_BF16, bool B_BF16, bool NMAJ, bool OUT_F32>
__device__ __forceinline__ void gemm_impl(
    const unsigned short* __restrict__ a, const unsigned short* __restrict__ b,
    void* out, const Dims d, const Epi e)
{
    constexpr int A_TILE = BM * LDA;
    constexpr int B_TILE = NMAJ ? BN * LDBN : BK * LDBK;
    __shared__ __align__(16) unsigned short as[2][A_TILE];
    __shared__ __align__(16) unsigned short bs[2][B_TILE];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int wm0 = (warp >> 2) * 64, wn0 = (warp & 3) * 32;
    const unsigned mt = (d.m + BM - 1) / BM, nt = (d.n + BN - 1) / BN;
    const unsigned per_group = GROUP_M * nt, pid = blockIdx.x;
    const unsigned first_m = (pid / per_group) * GROUP_M;
    const unsigned gsize = min(mt - first_m, (unsigned)GROUP_M);
    const unsigned m0 = (first_m + (pid % per_group) % gsize) * BM;
    const unsigned n0 = ((pid % per_group) / gsize) * BN;
    const unsigned ntiles = (d.k + BK - 1) / BK;

    uint4 ra[2], rb[2];
    bool over = false;
    // Global strip kt -> registers (zero past the edges; K, N are 8-multiples).
    auto fetch = [&](unsigned kt) {
        const unsigned k0 = kt * BK;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int i = tid + j * 256;
            {
                const int r = i >> 2, c = (i & 3) * 8;
                const unsigned gr = m0 + r, gc = k0 + c;
                ra[j] = (gr < d.m && gc < d.k)
                    ? *reinterpret_cast<const uint4*>(a + (size_t)gr * d.k + gc) : make_uint4(0, 0, 0, 0);
            }
            if (NMAJ) {
                const int r = i >> 2, c = (i & 3) * 8;
                const unsigned gn = n0 + r, gk = k0 + c;
                rb[j] = (gn < d.n && gk < d.k)
                    ? *reinterpret_cast<const uint4*>(b + (size_t)gn * d.k + gk) : make_uint4(0, 0, 0, 0);
            } else {
                const int r = i >> 4, c = (i & 15) * 8;
                const unsigned gk = k0 + r, gn = n0 + c;
                rb[j] = (gk < d.k && gn < d.n)
                    ? *reinterpret_cast<const uint4*>(b + (size_t)gk * d.n + gn) : make_uint4(0, 0, 0, 0);
            }
        }
    };
    auto stage = [&](int buf) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int i = tid + j * 256;
            *reinterpret_cast<uint4*>(&as[buf][(i >> 2) * LDA + (i & 3) * 8]) = ra[j];
            const uint4 v = conv8<C_BF16, B_BF16>(rb[j], over);
            if (NMAJ)
                *reinterpret_cast<uint4*>(&bs[buf][(i >> 2) * LDBN + (i & 3) * 8]) = v;
            else
                *reinterpret_cast<uint4*>(&bs[buf][(i >> 4) * LDBK + (i & 15) * 8]) = v;
        }
    };

    float acc[4][4][4];
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi)
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    fetch(0);
    stage(0);
    __syncthreads();
    for (unsigned kt = 0; kt < ntiles; ++kt) {
        const int buf = kt & 1;
        if (kt + 1 < ntiles) fetch(kt + 1);
        #pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            unsigned af[4][4];
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                ldsm_x4(af[mi], &as[buf][(wm0 + mi * 16 + (lane & 15)) * LDA + kk * 16 + (lane >> 4) * 8]);
            unsigned bf[4][2];
            #pragma unroll
            for (int nj = 0; nj < 2; ++nj) {
                unsigned r[4];
                if (NMAJ)
                    ldsm_x4(r, &bs[buf][(wn0 + nj * 16 + (lane & 7) + ((lane >> 4) << 3)) * LDBN
                                        + kk * 16 + ((lane >> 3) & 1) * 8]);
                else
                    ldsm_x4_t(r, &bs[buf][(kk * 16 + (lane & 7) + (((lane >> 3) & 1) << 3)) * LDBK
                                          + wn0 + nj * 16 + (lane >> 4) * 8]);
                bf[2 * nj][0] = r[0]; bf[2 * nj][1] = r[1];
                bf[2 * nj + 1][0] = r[2]; bf[2 * nj + 1][1] = r[3];
            }
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                #pragma unroll
                for (int ni = 0; ni < 4; ++ni)
                    mma16816<C_BF16>(acc[mi][ni], af[mi], bf[ni][0], bf[ni][1]);
        }
        if (kt + 1 < ntiles) stage(buf ^ 1);
        __syncthreads();
    }
    if (over) atomicOr(&g_overflow, 1u);

    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const unsigned col = n0 + wn0 + ni * 8 + 2 * t;
            if (col >= d.n) continue;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const unsigned row = m0 + wm0 + mi * 16 + g + h * 8;
                if (row >= d.m) continue;
                float x = acc[mi][ni][2 * h], y = acc[mi][ni][2 * h + 1];
                if (!OUT_F32 && e.bias) {
                    const unsigned bv = *reinterpret_cast<const unsigned*>(e.bias + col);
                    x += bf_lo(bv); y += bf_hi(bv);
                }
                if (C_BF16 && !OUT_F32) {
                    if (e.resid) {
                        float g0 = 1.f, g1 = 1.f;
                        if (e.gate) {
                            const unsigned gv = *reinterpret_cast<const unsigned*>(e.gate + col);
                            g0 = bf_lo(gv); g1 = bf_hi(gv);
                        }
                        const unsigned rv = *reinterpret_cast<const unsigned*>(e.resid + (size_t)row * d.n + col);
                        x = bf_lo(rv) + g0 * x; y = bf_hi(rv) + g1 * y;
                    }
                }
                if (OUT_F32) {
                    *reinterpret_cast<float2*>(static_cast<float*>(out) + (size_t)row * d.n + col) =
                        make_float2(x, y);
                    continue;
                }
                unsigned packed;
                if (C_BF16) {
                    __nv_bfloat162 v = __floats2bfloat162_rn(x, y);
                    packed = *reinterpret_cast<unsigned*>(&v);
                } else {
                    // Saturate like the wgpu op's f16 store: a finite result past
                    // the f16 range is 65504, never inf.
                    __half2 v = __floats2half2_rn(fminf(fmaxf(x, -65504.f), 65504.f),
                                                  fminf(fmaxf(y, -65504.f), 65504.f));
                    packed = *reinterpret_cast<unsigned*>(&v);
                }
                *reinterpret_cast<unsigned*>(static_cast<unsigned short*>(out) + (size_t)row * d.n + col) =
                    packed;
            }
        }
    }
}

#define GEMM(NAME, CB, BB, NM, O32) \
extern "C" __global__ void __launch_bounds__(256) NAME( \
    const unsigned short* a, const unsigned short* b, void* out, const Dims d, const Epi e) \
{ gemm_impl<CB, BB, NM, O32>(a, b, out, d, e); }

GEMM(gemm_a16_b16_k, false, false, false, false)
GEMM(gemm_a16_b16_n, false, false, true, false)
GEMM(gemm_a16_bb_k, false, true, false, false)
GEMM(gemm_a16_bb_n, false, true, true, false)
GEMM(gemm_ab_bb_k, true, true, false, false)
GEMM(gemm_ab_bb_n, true, true, true, false)
GEMM(gemm_a16_b16_k_o32, false, false, false, true)
GEMM(gemm_a16_b16_n_o32, false, false, true, true)

// 8-bit tensor-core GEMM: out[m, n] = bf16(acc * sa[m] * sb[n] + bias[n]),
// acc = A[m, :] . B[n, :] over per-row quantized operands: INT8 in int32
// (exact), or FP8 e4m3 (`F8`) in f32. FP8 MMAs accumulate in f16 partials
// that are added into the f32 total every F8_PROMOTE_TILES k-tiles (k = 128):
// GeForce parts run the f16-accumulate FP8 MMA at twice the f32-accumulate
// rate, and with operands quantized to |q| <= F8_RANGE (16) a 128-term partial
// is bounded by 16 * 16 * 128 = 32768 < 65504, so it cannot overflow, and its
// f16 rounding (~2^-11 per add) stays far inside e4m3's 3-bit mantissa. A
// `[m, K]`, B n-major `[n, K]`, K a multiple of 16; `d.k` counts K in BYTE
// PAIRS: an m16n8k32 8-bit fragment has exactly the byte layout of an
// m16n8k16 b16 fragment, so the tiling, staging and ldmatrix addressing are
// gemm_impl's (n-major B) as is. FP8 needs sm_89+ (the body traps below).
__device__ __forceinline__ void mma_s8(int (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
#define F8_PROMOTE_TILES 2
// One MMA into the running accumulator: INT8 straight into the int32 total;
// FP8 into the f16x2 partial `h` (h[0] = row g cols 2t, 2t+1; h[1] = row g+8).
__device__ __forceinline__ void mma_q8(int (&d)[4], unsigned (&)[2], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    mma_s8(d, a, b0, b1);
}
__device__ __forceinline__ void mma_q8(float (&)[4], unsigned (&h)[2], const unsigned (&a)[4], unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 890
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(h[0]), "+r"(h[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}
// Fold the f16 partial into the f32 total and restart it (INT8: nothing).
__device__ __forceinline__ void promote_q8(int (&)[4], unsigned (&)[2]) {}
__device__ __forceinline__ void promote_q8(float (&d)[4], unsigned (&h)[2]) {
    const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&h[0]));
    const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&h[1]));
    d[0] += lo.x; d[1] += lo.y; d[2] += hi.x; d[3] += hi.y;
    h[0] = h[1] = 0u;
}
template <bool F> struct Q8Acc { typedef int T; };
template <> struct Q8Acc<true> { typedef float T; };

template <bool F8>
__device__ __forceinline__ void gemm_q8_impl(
    const unsigned short* __restrict__ a, const float* __restrict__ sa,
    const unsigned short* __restrict__ b, const float* __restrict__ sb,
    unsigned short* __restrict__ out, const Dims d, const unsigned short* __restrict__ bias)
{
    constexpr int A_TILE = BM * LDA;
    constexpr int B_TILE = BN * LDBN;
    __shared__ __align__(16) unsigned short as[2][A_TILE];
    __shared__ __align__(16) unsigned short bs[2][B_TILE];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int wm0 = (warp >> 2) * 64, wn0 = (warp & 3) * 32;
    const unsigned mt = (d.m + BM - 1) / BM, nt = (d.n + BN - 1) / BN;
    const unsigned per_group = GROUP_M * nt, pid = blockIdx.x;
    const unsigned first_m = (pid / per_group) * GROUP_M;
    const unsigned gsize = min(mt - first_m, (unsigned)GROUP_M);
    const unsigned m0 = (first_m + (pid % per_group) % gsize) * BM;
    const unsigned n0 = ((pid % per_group) / gsize) * BN;
    const unsigned ntiles = (d.k + BK - 1) / BK;

    uint4 ra[2], rb[2];
    auto fetch = [&](unsigned kt) {
        const unsigned k0 = kt * BK;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int i = tid + j * 256;
            const int r = i >> 2, c = (i & 3) * 8;
            const unsigned gr = m0 + r, gn = n0 + r, gc = k0 + c;
            ra[j] = (gr < d.m && gc < d.k)
                ? *reinterpret_cast<const uint4*>(a + (size_t)gr * d.k + gc) : make_uint4(0, 0, 0, 0);
            rb[j] = (gn < d.n && gc < d.k)
                ? *reinterpret_cast<const uint4*>(b + (size_t)gn * d.k + gc) : make_uint4(0, 0, 0, 0);
        }
    };
    auto stage = [&](int buf) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int i = tid + j * 256;
            *reinterpret_cast<uint4*>(&as[buf][(i >> 2) * LDA + (i & 3) * 8]) = ra[j];
            *reinterpret_cast<uint4*>(&bs[buf][(i >> 2) * LDBN + (i & 3) * 8]) = rb[j];
        }
    };

    typename Q8Acc<F8>::T acc[4][4][4];
    unsigned part[4][4][2];
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi)
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0;
            part[mi][ni][0] = part[mi][ni][1] = 0u;
        }

    fetch(0);
    stage(0);
    __syncthreads();
    for (unsigned kt = 0; kt < ntiles; ++kt) {
        const int buf = kt & 1;
        if (kt + 1 < ntiles) fetch(kt + 1);
        #pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            unsigned af[4][4];
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                ldsm_x4(af[mi], &as[buf][(wm0 + mi * 16 + (lane & 15)) * LDA + kk * 16 + (lane >> 4) * 8]);
            unsigned bf[4][2];
            #pragma unroll
            for (int nj = 0; nj < 2; ++nj) {
                unsigned r[4];
                ldsm_x4(r, &bs[buf][(wn0 + nj * 16 + (lane & 7) + ((lane >> 4) << 3)) * LDBN
                                    + kk * 16 + ((lane >> 3) & 1) * 8]);
                bf[2 * nj][0] = r[0]; bf[2 * nj][1] = r[1];
                bf[2 * nj + 1][0] = r[2]; bf[2 * nj + 1][1] = r[3];
            }
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                #pragma unroll
                for (int ni = 0; ni < 4; ++ni)
                    mma_q8(acc[mi][ni], part[mi][ni], af[mi], bf[ni][0], bf[ni][1]);
        }
        if (F8 && ((kt + 1) % F8_PROMOTE_TILES == 0 || kt + 1 == ntiles)) {
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                #pragma unroll
                for (int ni = 0; ni < 4; ++ni)
                    promote_q8(acc[mi][ni], part[mi][ni]);
        }
        if (kt + 1 < ntiles) stage(buf ^ 1);
        __syncthreads();
    }

    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const unsigned col = n0 + wn0 + ni * 8 + 2 * t;
            if (col >= d.n) continue;
            const float s0 = sb[col], s1 = sb[col + 1];
            float b0 = 0.f, b1 = 0.f;
            if (bias) {
                const unsigned bv = *reinterpret_cast<const unsigned*>(bias + col);
                b0 = bf_lo(bv); b1 = bf_hi(bv);
            }
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const unsigned row = m0 + wm0 + mi * 16 + g + h * 8;
                if (row >= d.m) continue;
                const float r = sa[row];
                // Explicit roundings (no FMA contraction): acc * r * s + b.
                const float x = __fadd_rn(__fmul_rn(__fmul_rn((float)acc[mi][ni][2 * h], r), s0), b0);
                const float y = __fadd_rn(__fmul_rn(__fmul_rn((float)acc[mi][ni][2 * h + 1], r), s1), b1);
                __nv_bfloat162 v = __floats2bfloat162_rn(x, y);
                *reinterpret_cast<unsigned*>(out + (size_t)row * d.n + col) = *reinterpret_cast<unsigned*>(&v);
            }
        }
    }
}

#define GEMM_Q8(NAME, F8) \
extern "C" __global__ void __launch_bounds__(256) NAME( \
    const unsigned short* __restrict__ a, const float* __restrict__ sa, \
    const unsigned short* __restrict__ b, const float* __restrict__ sb, \
    unsigned short* __restrict__ out, const Dims d, const unsigned short* __restrict__ bias) \
{ gemm_q8_impl<F8>(a, sa, b, sb, out, d, bias); }
GEMM_Q8(gemm_i8, false)
GEMM_Q8(gemm_f8, true)

// Per row of a bf16 `[rows, k]`: the row's max |x| and its 256 threads'
// share of the row (one block per row).
__device__ __forceinline__ float row_absmax(const unsigned short* __restrict__ x, size_t base, unsigned k) {
    __shared__ float red[8];
    float m = 0.f;
    for (unsigned i = threadIdx.x; i < k; i += 256) m = fmaxf(m, fabsf(__uint_as_float((unsigned)x[base + i] << 16)));
    #pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    m = red[0];
    #pragma unroll
    for (int w = 1; w < 8; ++w) m = fmaxf(m, red[w]);
    return m;
}

// The FP8 operands' range: |q| <= 16 keeps gemm_f8's f16 MMA partials finite
// (see gemm_q8_impl). e4m3 is floating point, so the range only shifts which
// tiny values go subnormal (below max|x| / 1024), not the relative precision.
#define F8_RANGE 16.f

// Per row of a bf16 `[rows, k]`: q = e4m3(x * F8_RANGE / max|x|) (round to
// nearest), scale = max|x| / F8_RANGE (an all-zero row: q 0, scale 0). One
// block per row; sm_89+.
extern "C" __global__ void __launch_bounds__(256) quant_rows_f8(
    const unsigned short* __restrict__ x, unsigned char* __restrict__ q, float* __restrict__ scale, unsigned k)
{
#if __CUDA_ARCH__ >= 890
    const size_t base = (size_t)blockIdx.x * k;
    const float m = row_absmax(x, base, k);
    const float inv = m > 0.f ? F8_RANGE / m : 0.f;
    for (unsigned i = threadIdx.x; i < k; i += 256) {
        unsigned short r;
        const float v = __uint_as_float((unsigned)x[base + i] << 16) * inv;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.f), "f"(v));
        q[base + i] = (unsigned char)r;
    }
    if (threadIdx.x == 0) scale[blockIdx.x] = m / F8_RANGE;
#else
    __trap();
#endif
}

// Per row of a bf16 `[rows, k]`: q = round(x * 127 / max|x|) as int8, scale =
// max|x| / 127 (an all-zero row: q 0, scale 0). One block per row.
extern "C" __global__ void __launch_bounds__(256) quant_rows_i8(
    const unsigned short* __restrict__ x, signed char* __restrict__ q, float* __restrict__ scale, unsigned k)
{
    const size_t base = (size_t)blockIdx.x * k;
    const float m = row_absmax(x, base, k);
    const float inv = m > 0.f ? 127.f / m : 0.f;
    for (unsigned i = threadIdx.x; i < k; i += 256) {
        const float v = __uint_as_float((unsigned)x[base + i] << 16) * inv;
        q[base + i] = (signed char)max(-127, min(127, __float2int_rn(v)));
    }
    if (threadIdx.x == 0) scale[blockIdx.x] = m / 127.f;
}

// f32 acts x the bf16 n-major dequant workspace, f32 out, for M <= SK_M (a
// short prompt through an f32 encoder): memory-bound on B, so plain f32 FMA,
// each output summed in K order. A block owns SK_M rows x SK_BN columns, a
// thread 4 x 4 of them; A and B stage per SK_BK strip K-major in smem, so a
// warp's A reads broadcast and its B reads are contiguous.
#define SK_M 64
#define SK_BN 64
#define SK_BK 64
#define SK_LD (SK_M + 4)
extern "C" __global__ void __launch_bounds__(256) gemm_af32_bb_n(
    const float* __restrict__ a, const unsigned short* __restrict__ b, float* __restrict__ out, const Dims d)
{
    __shared__ __align__(16) float as[SK_BK][SK_LD];
    __shared__ __align__(16) float bs[SK_BK][SK_LD];
    const int tid = threadIdx.x, tm = tid / 16, tn = tid % 16;
    const unsigned n0 = blockIdx.x * SK_BN, m = d.m, n = d.n, k = d.k;
    float acc[4][4] = {};
    for (unsigned k0 = 0; k0 < k; k0 += SK_BK) {
        // K is a multiple of 8, so a 4- or 8-wide vector is all in or all out.
        for (int x = tid; x < SK_M * (SK_BK / 4); x += blockDim.x) {
            const unsigned r = x / (SK_BK / 4), c = x % (SK_BK / 4) * 4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (r < m && k0 + c < k) v = *reinterpret_cast<const float4*>(a + (size_t)r * k + k0 + c);
            as[c][r] = v.x; as[c + 1][r] = v.y; as[c + 2][r] = v.z; as[c + 3][r] = v.w;
        }
        for (int x = tid; x < SK_BN * (SK_BK / 8); x += blockDim.x) {
            const unsigned r = x / (SK_BK / 8), c = x % (SK_BK / 8) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (n0 + r < n && k0 + c < k) v = *reinterpret_cast<const uint4*>(b + (size_t)(n0 + r) * k + k0 + c);
            const unsigned w[4] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                bs[c + 2 * i][r] = bf_lo(w[i]);
                bs[c + 2 * i + 1][r] = bf_hi(w[i]);
            }
        }
        __syncthreads();
        #pragma unroll 16
        for (int kk = 0; kk < SK_BK; ++kk) {
            const float4 av = *reinterpret_cast<const float4*>(&as[kk][tm * 4]);
            const float4 bv = *reinterpret_cast<const float4*>(&bs[kk][tn * 4]);
            const float ar[4] = {av.x, av.y, av.z, av.w}, br[4] = {bv.x, bv.y, bv.z, bv.w};
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(ar[i], br[j], acc[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned r = tm * 4 + i;
        if (r >= m) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned c = n0 + tn * 4 + j;
            if (c < n) out[(size_t)r * n + c] = acc[i][j];
        }
    }
}

// One-image 2D convolution as an implicit GEMM, the f32-act conv3d op at T = 1
// (see `Conv2dAddrs`): out[co, m] = bias[co] + sum_k w[co, k] * col[k, m],
// k = (ci, dh, dw), m = (ho, wo). Each output sums in k order with fmaf and
// then adds the bias, as the WGSL does, so the tile shape never changes the
// bits. A block owns BM x BN outputs (256 threads of TM x TN, taken in runs of
// four spread across the tile so the float4 smem reads are conflict-free) and
// walks K in CBK strips through double-buffered smem.
struct ConvP { unsigned cin, cout, h, w, h_out, w_out, k, stride, pad, reflect; };
#define CBK 8
__device__ __forceinline__ int reflect_at(int i, int n) {
    return i < 0 ? -i : (i >= n ? 2 * n - 2 - i : i);
}
template <int CM, int CN, int CTM, int CTN>
__device__ __forceinline__ void conv2d_impl(
    const float* __restrict__ x, const unsigned short* __restrict__ w,
    const unsigned short* __restrict__ bias, float* __restrict__ out, const ConvP p)
{
    static_assert((CM / CTM) * (CN / CTN) == 256, "256 threads");
    constexpr int AL = CM * CBK / 256, BL = CN * CBK / 256;   // loads per thread
    constexpr int RS = CM * 4 / CTM, CS = CN * 4 / CTN;          // run strides
    __shared__ __align__(16) float as[2][CBK][CM];
    __shared__ __align__(16) float bs[2][CBK][CN];
    const int tid = threadIdx.x, tx = tid % (CN / CTN), ty = tid / (CN / CTN);
    const unsigned kk2 = p.k * p.k, kdim = p.cin * kk2, m_total = p.h_out * p.w_out;
    const unsigned m0 = blockIdx.x * CN, co0 = blockIdx.y * CM;
    const int H = p.h, W = p.w;
    // Loads: A row `ar`, k `ak0..ak0 + AL`; B one fixed column `bc`, rows
    // `br0 + (256 / CN) s`, so its (ho, wo) is resolved once.
    const unsigned ar = tid % CM, ak0 = tid / CM * AL;
    const unsigned bc = tid % CN, br0 = tid / CN;
    const unsigned gm = m0 + bc;
    const bool col_ok = gm < m_total;
    const int hb = col_ok ? (int)(gm / p.w_out * p.stride) - (int)p.pad : 0;
    const int wb = col_ok ? (int)(gm % p.w_out * p.stride) - (int)p.pad : 0;
    const bool row_ok = co0 + ar < p.cout;
    const unsigned short* wrow = w + (size_t)(co0 + ar) * kdim;
    float ra[AL], rb[BL];
    auto load = [&](unsigned k0) {
        #pragma unroll
        for (int s = 0; s < AL; ++s) {
            const unsigned gk = k0 + ak0 + s;
            ra[s] = row_ok && gk < kdim ? __uint_as_float((unsigned)wrow[gk] << 16) : 0.f;
        }
        #pragma unroll
        for (int s = 0; s < BL; ++s) {
            const unsigned gk = k0 + br0 + (256 / CN) * s;
            float v = 0.f;
            if (col_ok && gk < kdim) {
                const unsigned ci = gk / kk2, r = gk - ci * kk2, dh = r / p.k, dw = r - dh * p.k;
                int hi = hb + (int)dh, wi = wb + (int)dw;
                if (p.reflect) {
                    hi = reflect_at(hi, H);
                    wi = reflect_at(wi, W);
                    v = x[((size_t)ci * H + hi) * W + wi];
                } else if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                    v = x[((size_t)ci * H + hi) * W + wi];
                }
            }
            rb[s] = v;
        }
    };
    auto stash = [&](int buf) {
        #pragma unroll
        for (int s = 0; s < AL; ++s) as[buf][ak0 + s][ar] = ra[s];
        #pragma unroll
        for (int s = 0; s < BL; ++s) bs[buf][br0 + (256 / CN) * s][bc] = rb[s];
    };
    float acc[CTM][CTN];
    #pragma unroll
    for (int i = 0; i < CTM; ++i)
        #pragma unroll
        for (int j = 0; j < CTN; ++j) acc[i][j] = 0.f;
    load(0);
    stash(0);
    __syncthreads();
    int buf = 0;
    for (unsigned k0 = 0; k0 < kdim; k0 += CBK) {
        const bool more = k0 + CBK < kdim;
        if (more) load(k0 + CBK);
        #pragma unroll
        for (int kk = 0; kk < CBK; ++kk) {
            float a[CTM], b[CTN];
            #pragma unroll
            for (int q = 0; q < CTM / 4; ++q) {
                const float4 v = *reinterpret_cast<const float4*>(&as[buf][kk][q * RS + ty * 4]);
                a[4 * q] = v.x; a[4 * q + 1] = v.y; a[4 * q + 2] = v.z; a[4 * q + 3] = v.w;
            }
            #pragma unroll
            for (int q = 0; q < CTN / 4; ++q) {
                const float4 v = *reinterpret_cast<const float4*>(&bs[buf][kk][q * CS + tx * 4]);
                b[4 * q] = v.x; b[4 * q + 1] = v.y; b[4 * q + 2] = v.z; b[4 * q + 3] = v.w;
            }
            #pragma unroll
            for (int i = 0; i < CTM; ++i)
                #pragma unroll
                for (int j = 0; j < CTN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        if (more) stash(buf ^ 1);
        __syncthreads();
        buf ^= 1;
    }
    #pragma unroll
    for (int i = 0; i < CTM; ++i) {
        const unsigned co = co0 + i / 4 * RS + ty * 4 + i % 4;
        if (co >= p.cout) continue;
        const float bv = __uint_as_float((unsigned)bias[co] << 16);
        #pragma unroll
        for (int j = 0; j < CTN; ++j) {
            const unsigned m = m0 + j / 4 * CS + tx * 4 + j % 4;
            if (m < m_total) out[(size_t)co * m_total + m] = acc[i][j] + bv;
        }
    }
}
// Big outputs, and a small tile for the deep levels' few pixels (more blocks).
extern "C" __global__ void __launch_bounds__(256) conv2d_f32(
    const float* x, const unsigned short* w, const unsigned short* bias, float* out, const ConvP p)
{ conv2d_impl<128, 128, 8, 8>(x, w, bias, out, p); }
extern "C" __global__ void __launch_bounds__(256) conv2d_f32_small(
    const float* x, const unsigned short* w, const unsigned short* bias, float* out, const ConvP p)
{ conv2d_impl<64, 64, 4, 4>(x, w, bias, out, p); }

// A 3D convolution with f16 activations, the f16-act conv3d op (see
// `Conv3dAddrs`): conv2d_impl's implicit GEMM with k = (ci, dt, dh, dw) and m
// = (to, ho, wo), zero padding, batch on grid z. Same per-output fmaf order
// and bias add as the WGSL, then the same clamp + round-to-nearest-even f16
// store, so the bits match it.
struct Conv3P {
    unsigned b, cin, cout, t, h, w, t_out, h_out, w_out;
    unsigned kt, kh, kw, pad_t, pad_h, pad_w, st, sh, sw;
};
template <int CM, int CN, int CTM, int CTN>
__device__ __forceinline__ void conv3d_f16_impl(
    const __half* __restrict__ x, const unsigned short* __restrict__ w,
    const unsigned short* __restrict__ bias, __half* __restrict__ out, const Conv3P p)
{
    static_assert((CM / CTM) * (CN / CTN) == 256, "256 threads");
    constexpr int AL = CM * CBK / 256, BL = CN * CBK / 256;
    constexpr int RS = CM * 4 / CTM, CS = CN * 4 / CTN;
    __shared__ __align__(16) float as[2][CBK][CM];
    __shared__ __align__(16) float bs[2][CBK][CN];
    const int tid = threadIdx.x, tx = tid % (CN / CTN), ty = tid / (CN / CTN);
    const unsigned khkw = p.kh * p.kw, ktkhkw = p.kt * khkw, kdim = p.cin * ktkhkw;
    const unsigned hw_out = p.h_out * p.w_out, m_total = p.t_out * hw_out;
    const unsigned m0 = blockIdx.x * CN, co0 = blockIdx.y * CM;
    const int T = p.t, H = p.h, W = p.w;
    const size_t thw = (size_t)T * H * W;
    x += (size_t)blockIdx.z * p.cin * thw;
    out += (size_t)blockIdx.z * p.cout * m_total;
    const unsigned ar = tid % CM, ak0 = tid / CM * AL;
    const unsigned bc = tid % CN, br0 = tid / CN;
    const unsigned gm = m0 + bc;
    const bool col_ok = gm < m_total;
    const unsigned to = col_ok ? gm / hw_out : 0, r2 = gm - to * hw_out;
    const int tb = col_ok ? (int)(to * p.st) - (int)p.pad_t : 0;
    const int hb = col_ok ? (int)(r2 / p.w_out * p.sh) - (int)p.pad_h : 0;
    const int wb = col_ok ? (int)(r2 % p.w_out * p.sw) - (int)p.pad_w : 0;
    const bool row_ok = co0 + ar < p.cout;
    const unsigned short* wrow = w + (size_t)(co0 + ar) * kdim;
    float ra[AL], rb[BL];
    auto load = [&](unsigned k0) {
        #pragma unroll
        for (int s = 0; s < AL; ++s) {
            const unsigned gk = k0 + ak0 + s;
            ra[s] = row_ok && gk < kdim ? __uint_as_float((unsigned)wrow[gk] << 16) : 0.f;
        }
        #pragma unroll
        for (int s = 0; s < BL; ++s) {
            const unsigned gk = k0 + br0 + (256 / CN) * s;
            float v = 0.f;
            if (col_ok && gk < kdim) {
                const unsigned ci = gk / ktkhkw, r = gk - ci * ktkhkw;
                const unsigned dt = r / khkw, r1 = r - dt * khkw, dh = r1 / p.kw, dw = r1 - dh * p.kw;
                const int ti = tb + (int)dt, hi = hb + (int)dh, wi = wb + (int)dw;
                if (ti >= 0 && ti < T && hi >= 0 && hi < H && wi >= 0 && wi < W)
                    v = __half2float(x[ci * thw + ((size_t)ti * H + hi) * W + wi]);
            }
            rb[s] = v;
        }
    };
    auto stash = [&](int buf) {
        #pragma unroll
        for (int s = 0; s < AL; ++s) as[buf][ak0 + s][ar] = ra[s];
        #pragma unroll
        for (int s = 0; s < BL; ++s) bs[buf][br0 + (256 / CN) * s][bc] = rb[s];
    };
    float acc[CTM][CTN];
    #pragma unroll
    for (int i = 0; i < CTM; ++i)
        #pragma unroll
        for (int j = 0; j < CTN; ++j) acc[i][j] = 0.f;
    load(0);
    stash(0);
    __syncthreads();
    int buf = 0;
    for (unsigned k0 = 0; k0 < kdim; k0 += CBK) {
        const bool more = k0 + CBK < kdim;
        if (more) load(k0 + CBK);
        #pragma unroll
        for (int kk = 0; kk < CBK; ++kk) {
            float a[CTM], b[CTN];
            #pragma unroll
            for (int q = 0; q < CTM / 4; ++q) {
                const float4 v = *reinterpret_cast<const float4*>(&as[buf][kk][q * RS + ty * 4]);
                a[4 * q] = v.x; a[4 * q + 1] = v.y; a[4 * q + 2] = v.z; a[4 * q + 3] = v.w;
            }
            #pragma unroll
            for (int q = 0; q < CTN / 4; ++q) {
                const float4 v = *reinterpret_cast<const float4*>(&bs[buf][kk][q * CS + tx * 4]);
                b[4 * q] = v.x; b[4 * q + 1] = v.y; b[4 * q + 2] = v.z; b[4 * q + 3] = v.w;
            }
            #pragma unroll
            for (int i = 0; i < CTM; ++i)
                #pragma unroll
                for (int j = 0; j < CTN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        if (more) stash(buf ^ 1);
        __syncthreads();
        buf ^= 1;
    }
    #pragma unroll
    for (int i = 0; i < CTM; ++i) {
        const unsigned co = co0 + i / 4 * RS + ty * 4 + i % 4;
        if (co >= p.cout) continue;
        const float bv = __uint_as_float((unsigned)bias[co] << 16);
        #pragma unroll
        for (int j = 0; j < CTN; ++j) {
            const unsigned m = m0 + j / 4 * CS + tx * 4 + j % 4;
            if (m < m_total)
                out[(size_t)co * m_total + m] =
                    __float2half_rn(fminf(fmaxf(acc[i][j] + bv, -65504.f), 65504.f));
        }
    }
}
extern "C" __global__ void __launch_bounds__(256) conv3d_f16(
    const __half* x, const unsigned short* w, const unsigned short* bias, __half* out, const Conv3P p)
{ conv3d_f16_impl<128, 128, 8, 8>(x, w, bias, out, p); }
extern "C" __global__ void __launch_bounds__(256) conv3d_f16_small(
    const __half* x, const unsigned short* w, const unsigned short* bias, __half* out, const Conv3P p)
{ conv3d_f16_impl<64, 64, 4, 4>(x, w, bias, out, p); }

// The opt-in tensor-core form (`Conv2dShape::bf16`): the weights as the
// row-major A `[cout, kdim]` (kdim a multiple of 8), the gathered input
// columns rounded to bf16 as B `[kdim, pixels]` staged k-major, f32
// accumulate, then the bias. gemm_impl's tiling: 128 x 128 per block, 8 warps
// of 64 x 32, BK strips double-buffered through registers.
#define CB_LOADS (BK * BN / 256)
extern "C" __global__ void __launch_bounds__(256) conv2d_bf16(
    const float* __restrict__ x, const unsigned short* __restrict__ w,
    const unsigned short* __restrict__ bias, float* __restrict__ out, const ConvP p)
{
    __shared__ __align__(16) unsigned short as[2][BM * LDA];
    __shared__ __align__(16) unsigned short bs[2][BK * LDBK];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int wm0 = (warp >> 2) * 64, wn0 = (warp & 3) * 32;
    const unsigned kk2 = p.k * p.k, kdim = p.cin * kk2, m_total = p.h_out * p.w_out;
    const unsigned m0 = blockIdx.x * BN, co0 = blockIdx.y * BM;
    const unsigned ntiles = (kdim + BK - 1) / BK;
    const int H = p.h, W = p.w;
    // B loads: one fixed column `bc` (its (ho, wo) resolved once), rows
    // `br0 + (256 / BN) s`.
    const unsigned bc = tid % BN, br0 = tid / BN;
    const unsigned gm = m0 + bc;
    const bool col_ok = gm < m_total;
    const int hb = col_ok ? (int)(gm / p.w_out * p.stride) - (int)p.pad : 0;
    const int wb = col_ok ? (int)(gm % p.w_out * p.stride) - (int)p.pad : 0;
    uint4 ra[2];
    float rb[CB_LOADS];
    auto fetch = [&](unsigned kt) {
        const unsigned k0 = kt * BK;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int i = tid + j * 256;
            const unsigned gr = co0 + (i >> 2), gc = k0 + (i & 3) * 8;
            ra[j] = (gr < p.cout && gc < kdim)
                ? *reinterpret_cast<const uint4*>(w + (size_t)gr * kdim + gc) : make_uint4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int s = 0; s < CB_LOADS; ++s) {
            const unsigned gk = k0 + br0 + (256 / BN) * s;
            float v = 0.f;
            if (col_ok && gk < kdim) {
                const unsigned ci = gk / kk2, r = gk - ci * kk2, dh = r / p.k, dw = r - dh * p.k;
                int hi = hb + (int)dh, wi = wb + (int)dw;
                if (p.reflect) {
                    v = x[((size_t)ci * H + reflect_at(hi, H)) * W + reflect_at(wi, W)];
                } else if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                    v = x[((size_t)ci * H + hi) * W + wi];
                }
            }
            rb[s] = v;
        }
    };
    auto stage = [&](int buf) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int i = tid + j * 256;
            *reinterpret_cast<uint4*>(&as[buf][(i >> 2) * LDA + (i & 3) * 8]) = ra[j];
        }
        #pragma unroll
        for (int s = 0; s < CB_LOADS; ++s)
            bs[buf][(br0 + (256 / BN) * s) * LDBK + bc] = __bfloat16_as_ushort(__float2bfloat16_rn(rb[s]));
    };
    float acc[4][4][4];
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi)
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;
    fetch(0);
    stage(0);
    __syncthreads();
    for (unsigned kt = 0; kt < ntiles; ++kt) {
        const int buf = kt & 1;
        if (kt + 1 < ntiles) fetch(kt + 1);
        #pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            unsigned af[4][4];
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                ldsm_x4(af[mi], &as[buf][(wm0 + mi * 16 + (lane & 15)) * LDA + kk * 16 + (lane >> 4) * 8]);
            unsigned bf[4][2];
            #pragma unroll
            for (int nj = 0; nj < 2; ++nj) {
                unsigned r[4];
                ldsm_x4_t(r, &bs[buf][(kk * 16 + (lane & 7) + (((lane >> 3) & 1) << 3)) * LDBK
                                      + wn0 + nj * 16 + (lane >> 4) * 8]);
                bf[2 * nj][0] = r[0]; bf[2 * nj][1] = r[1];
                bf[2 * nj + 1][0] = r[2]; bf[2 * nj + 1][1] = r[3];
            }
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                #pragma unroll
                for (int ni = 0; ni < 4; ++ni)
                    mma16816<true>(acc[mi][ni], af[mi], bf[ni][0], bf[ni][1]);
        }
        if (kt + 1 < ntiles) stage(buf ^ 1);
        __syncthreads();
    }
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned co = co0 + wm0 + mi * 16 + g + h * 8;
            if (co >= p.cout) continue;
            const float bv = __uint_as_float((unsigned)bias[co] << 16);
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const unsigned m = m0 + wn0 + ni * 8 + 2 * t;
                if (m < m_total) out[(size_t)co * m_total + m] = acc[mi][ni][2 * h] + bv;
                if (m + 1 < m_total) out[(size_t)co * m_total + m + 1] = acc[mi][ni][2 * h + 1] + bv;
            }
        }
    }
}

// The tensor-core form of conv3d_f16: conv2d_bf16's mainloop with
// conv3d_f16's 3D gather on f16 operands, f32 accumulate, then the bias and
// the clamped f16 store. The f16 input columns stay exact and the bf16
// weights narrow to f16 (exact but for subnormal-range weights), so only the
// accumulation order differs from conv3d_f16.
__device__ __forceinline__ __half f16_store(float v) {
    return __float2half_rn(fminf(fmaxf(v, -65504.f), 65504.f));
}
__device__ __forceinline__ unsigned bf16x2_to_f16x2(unsigned v) {
    const __half lo = __float2half_rn(__uint_as_float(v << 16));
    const __half hi = __float2half_rn(__uint_as_float(v & 0xffff0000u));
    return (unsigned)__half_as_ushort(lo) | ((unsigned)__half_as_ushort(hi) << 16);
}
extern "C" __global__ void __launch_bounds__(256) conv3d_f16tc(
    const __half* __restrict__ x, const unsigned short* __restrict__ w,
    const unsigned short* __restrict__ bias, __half* __restrict__ out, const Conv3P p)
{
    __shared__ __align__(16) unsigned short as[2][BM * LDA];
    __shared__ __align__(16) unsigned short bs[2][BK * LDBK];
    // Per-tile k-row gather table, built two tiles ahead: `x` = the row's
    // input offset `ci*thw + (dt*H + dh)*W + dw`, `y` = dt | dh << 8 | dw << 16
    // (-1 past kdim). Each warp reads one row per load (a broadcast).
    __shared__ int2 ktab[2][BK];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int wm0 = (warp >> 2) * 64, wn0 = (warp & 3) * 32;
    const unsigned khkw = p.kh * p.kw, ktkhkw = p.kt * khkw, kdim = p.cin * ktkhkw;
    const unsigned hw_out = p.h_out * p.w_out, m_total = p.t_out * hw_out;
    const unsigned m0 = blockIdx.x * BN, co0 = blockIdx.y * BM;
    const unsigned ntiles = (kdim + BK - 1) / BK;
    const int T = p.t, H = p.h, W = p.w;
    const size_t thw = (size_t)T * H * W;
    x += (size_t)blockIdx.z * p.cin * thw;
    out += (size_t)blockIdx.z * p.cout * m_total;
    // B loads: one fixed column `bc` (its (ho, wo) resolved once), rows
    // `br0 + (256 / BN) s`.
    const unsigned bc = tid % BN, br0 = tid / BN;
    const unsigned gm = m0 + bc;
    const bool col_ok = gm < m_total;
    const unsigned to = col_ok ? gm / hw_out : 0, r2 = gm - to * hw_out;
    const int tb = col_ok ? (int)(to * p.st) - (int)p.pad_t : 0;
    const int hb = col_ok ? (int)(r2 / p.w_out * p.sh) - (int)p.pad_h : 0;
    const int wb = col_ok ? (int)(r2 % p.w_out * p.sw) - (int)p.pad_w : 0;
    const long long col_base = ((long long)tb * H + hb) * W + wb;
    const unsigned short* xs = reinterpret_cast<const unsigned short*>(x);
    auto build = [&](unsigned kt) {
        const unsigned gk = kt * BK + (tid & (BK - 1));
        int2 e = make_int2(0, -1);
        if (gk < kdim) {
            const unsigned ci = gk / ktkhkw, r = gk - ci * ktkhkw;
            const unsigned dt = r / khkw, r1 = r - dt * khkw, dh = r1 / p.kw, dw = r1 - dh * p.kw;
            e = make_int2((int)(ci * thw + ((size_t)dt * H + dh) * W + dw), (int)(dt | dh << 8 | dw << 16));
        }
        ktab[kt & 1][tid & (BK - 1)] = e;
    };
    uint4 ra[2];
    unsigned short rb[CB_LOADS];
    auto fetch = [&](unsigned kt) {
        const unsigned k0 = kt * BK;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int i = tid + j * 256;
            const unsigned gr = co0 + (i >> 2), gc = k0 + (i & 3) * 8;
            ra[j] = (gr < p.cout && gc < kdim)
                ? *reinterpret_cast<const uint4*>(w + (size_t)gr * kdim + gc) : make_uint4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int s = 0; s < CB_LOADS; ++s) {
            const int2 e = ktab[kt & 1][br0 + (256 / BN) * s];
            unsigned short v = 0;
            if (col_ok && e.y >= 0) {
                const int ti = tb + (e.y & 0xff), hi = hb + ((e.y >> 8) & 0xff), wi = wb + (e.y >> 16);
                if (ti >= 0 && ti < T && hi >= 0 && hi < H && wi >= 0 && wi < W) v = xs[col_base + e.x];
            }
            rb[s] = v;
        }
    };
    auto stage = [&](int buf) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int i = tid + j * 256;
            const uint4 v = make_uint4(bf16x2_to_f16x2(ra[j].x), bf16x2_to_f16x2(ra[j].y),
                                       bf16x2_to_f16x2(ra[j].z), bf16x2_to_f16x2(ra[j].w));
            *reinterpret_cast<uint4*>(&as[buf][(i >> 2) * LDA + (i & 3) * 8]) = v;
        }
        #pragma unroll
        for (int s = 0; s < CB_LOADS; ++s)
            bs[buf][(br0 + (256 / BN) * s) * LDBK + bc] = rb[s];
    };
    float acc[4][4][4];
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi)
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;
    if (tid < BK) build(0);
    else if (tid < 2 * BK && ntiles > 1) build(1);
    __syncthreads();
    fetch(0);
    stage(0);
    __syncthreads();
    for (unsigned kt = 0; kt < ntiles; ++kt) {
        const int buf = kt & 1;
        if (kt + 1 < ntiles) fetch(kt + 1);
        // Tile kt's table slot was last read by fetch(kt), before the
        // previous iteration's barrier: refill it for tile kt + 2.
        if (tid < BK && kt + 2 < ntiles) build(kt + 2);
        #pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            unsigned af[4][4];
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                ldsm_x4(af[mi], &as[buf][(wm0 + mi * 16 + (lane & 15)) * LDA + kk * 16 + (lane >> 4) * 8]);
            unsigned bf[4][2];
            #pragma unroll
            for (int nj = 0; nj < 2; ++nj) {
                unsigned r[4];
                ldsm_x4_t(r, &bs[buf][(kk * 16 + (lane & 7) + (((lane >> 3) & 1) << 3)) * LDBK
                                      + wn0 + nj * 16 + (lane >> 4) * 8]);
                bf[2 * nj][0] = r[0]; bf[2 * nj][1] = r[1];
                bf[2 * nj + 1][0] = r[2]; bf[2 * nj + 1][1] = r[3];
            }
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                #pragma unroll
                for (int ni = 0; ni < 4; ++ni)
                    mma16816<false>(acc[mi][ni], af[mi], bf[ni][0], bf[ni][1]);
        }
        if (kt + 1 < ntiles) stage(buf ^ 1);
        __syncthreads();
    }
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned co = co0 + wm0 + mi * 16 + g + h * 8;
            if (co >= p.cout) continue;
            const float bv = __uint_as_float((unsigned)bias[co] << 16);
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const unsigned m = m0 + wn0 + ni * 8 + 2 * t;
                if (m < m_total) out[(size_t)co * m_total + m] = f16_store(acc[mi][ni][2 * h] + bv);
                if (m + 1 < m_total)
                    out[(size_t)co * m_total + m + 1] = f16_store(acc[mi][ni][2 * h + 1] + bv);
            }
        }
    }
}
"#;

/// The big conv kernel's square output tile (Cout x pixels); the small
/// kernel's is half of it per side.
const CONV_TILE: u32 = 128;
/// Below this many big tiles the small kernel runs instead.
const CONV_MIN_BIG_TILES: u32 = 128;

/// Most rows the f32-act kernel takes (its block covers them all).
const SKINNY_M: u32 = 64;
const SKINNY_BN: u32 = 64;

pub use thinfer_core::backend::{GemmQ8Addrs, Q8};

/// The dims uniform (`(m, n, k, _)`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GemmDims {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub pad: u32,
}

/// The compiled kernels for one device: one per (A type, B type, B layout,
/// output type).
pub struct GemmKernels {
    module: Module,
    funcs: [sys::CUfunction; 8],
    skinny_f32: sys::CUfunction,
    conv2d: sys::CUfunction,
    conv2d_small: sys::CUfunction,
    conv2d_bf16: sys::CUfunction,
    conv3d_f16: sys::CUfunction,
    conv3d_f16_small: sys::CUfunction,
    conv3d_f16tc: sys::CUfunction,
    /// `[I8, E4m3]` kernels of the 8-bit GEMM and its row quantizer; E4m3
    /// `None` below sm_89.
    gemm_q8: [Option<sys::CUfunction>; 2],
    quant_rows_q8: [Option<sys::CUfunction>; 2],
    overflow: sys::CUdeviceptr,
    /// An unchecked narrowing GEMM ran since the overflow flag was last read.
    unchecked: std::sync::atomic::AtomicBool,
    /// Where [`Self::probe_unchecked_overflow`] copies the flag.
    probe: Arc<OverflowProbe>,
}

/// A pinned host copy of the GEMM overflow flag, written by an async copy
/// enqueued ahead of a fence and read once the fence completes. Shared with
/// the fence futures, which may outlive a scope.
pub struct OverflowProbe {
    host: *mut u32,
    /// An overflow was reported: the device flag is still set and must be
    /// cleared before it is read again.
    reset_due: std::sync::atomic::AtomicBool,
}

// SAFETY: `host` is a pinned allocation owned by this probe; the device only
// writes it through copies the fence orders before any read.
unsafe impl Send for OverflowProbe {}
unsafe impl Sync for OverflowProbe {}

impl OverflowProbe {
    fn new() -> Result<Self, String> {
        let mut host = std::ptr::null_mut();
        // SAFETY: allocates 4 pinned bytes on the bound context.
        crate::nvrtc::drv(
            unsafe { sys::cuMemAllocHost_v2(&mut host, std::mem::size_of::<u32>()) },
            "cuMemAllocHost",
        )?;
        let host = host.cast::<u32>();
        // SAFETY: freshly allocated, 4-byte aligned, no copy in flight.
        unsafe { host.write_volatile(0) };
        Ok(Self {
            host,
            reset_due: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// After the fence that followed the copy: whether the copied flag was
    /// set. A set flag is consumed (cleared here, and on the device before
    /// the next read).
    pub fn take(&self) -> bool {
        // SAFETY: the copy that writes `host` completed before the fence.
        let over = unsafe { self.host.read_volatile() } != 0;
        if over {
            // SAFETY: as above; no copy is in flight for a completed fence.
            unsafe { self.host.write_volatile(0) };
            self.reset_due
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        over
    }
}

impl Drop for OverflowProbe {
    fn drop(&mut self) {
        // SAFETY: allocated by cuMemAllocHost; nothing reads it any more.
        let _ = unsafe { sys::cuMemFreeHost(self.host.cast()) };
    }
}

// SAFETY: function handles are plain driver handles; every use binds the
// owning context first.
unsafe impl Send for GemmKernels {}
unsafe impl Sync for GemmKernels {}

/// Whether [`GemmKernels::run`] takes `desc` at `m` x `n`: the tensor-core
/// forms from [`MIN_M`] rows, the f32-act form (bf16 n-major B, f32 out) up to
/// its block's rows.
pub fn accepts(desc: &GemmDesc, m: u32, n: u32) -> bool {
    n.is_multiple_of(8)
        && if desc.a == GemmType::F32 {
            is_skinny_f32(desc) && m <= SKINNY_M
        } else {
            m >= MIN_M && index(desc).is_some()
        }
}

fn is_skinny_f32(desc: &GemmDesc) -> bool {
    desc.a == GemmType::F32 && desc.b == GemmType::Bf16 && desc.b_nmajor && !desc.out_f32
}

/// The tensor-core kernel for `desc`; `None` for f32 acts (see [`accepts`]),
/// the declined bf16-A x f16-B combo and f32 output on anything but f16 x
/// f16 (no caller has one).
pub fn index(desc: &GemmDesc) -> Option<usize> {
    let form = match (desc.a, desc.b, desc.out_f32) {
        (GemmType::F16, GemmType::F16, false) => 0,
        (GemmType::F16, GemmType::Bf16, false) => 1,
        (GemmType::Bf16, GemmType::Bf16, false) => 2,
        (GemmType::F16, GemmType::F16, true) => 3,
        _ => return None,
    };
    Some(form * 2 + desc.b_nmajor as usize)
}

impl GemmKernels {
    pub fn new(ctx: Arc<CudaContext>, sm: (i32, i32)) -> Result<Self, String> {
        let module = Module::compile(ctx, SOURCE, "thinfer_gemm.cu", sm)?;
        let names = [
            c"gemm_a16_b16_k",
            c"gemm_a16_b16_n",
            c"gemm_a16_bb_k",
            c"gemm_a16_bb_n",
            c"gemm_ab_bb_k",
            c"gemm_ab_bb_n",
            c"gemm_a16_b16_k_o32",
            c"gemm_a16_b16_n_o32",
        ];
        let mut funcs = [std::ptr::null_mut(); 8];
        for (f, name) in funcs.iter_mut().zip(names) {
            *f = module.function(name, 0)?;
        }
        let skinny_f32 = module.function(c"gemm_af32_bb_n", 0)?;
        let conv2d = module.function(c"conv2d_f32", 0)?;
        let conv2d_small = module.function(c"conv2d_f32_small", 0)?;
        let conv2d_bf16 = module.function(c"conv2d_bf16", 0)?;
        let conv3d_f16 = module.function(c"conv3d_f16", 0)?;
        let conv3d_f16_small = module.function(c"conv3d_f16_small", 0)?;
        let conv3d_f16tc = module.function(c"conv3d_f16tc", 0)?;
        let fp8 = sm >= (8, 9);
        let gemm_q8 = [
            Some(module.function(c"gemm_i8", 0)?),
            fp8.then(|| module.function(c"gemm_f8", 0)).transpose()?,
        ];
        let quant_rows_q8 = [
            Some(module.function(c"quant_rows_i8", 0)?),
            fp8.then(|| module.function(c"quant_rows_f8", 0))
                .transpose()?,
        ];
        let overflow = module.global(c"g_overflow")?;
        let probe = Arc::new(OverflowProbe::new()?);
        Ok(Self {
            module,
            funcs,
            skinny_f32,
            conv2d,
            conv2d_small,
            conv2d_bf16,
            conv3d_f16,
            conv3d_f16_small,
            conv3d_f16tc,
            gemm_q8,
            quant_rows_q8,
            overflow,
            unchecked: std::sync::atomic::AtomicBool::new(false),
            probe,
        })
    }

    /// The f16-act 3D convolution of [`Conv3dAddrs`]. A `cin * kt * kh * kw`
    /// that is a multiple of 8 runs on f16 tensor cores with f32 accumulate
    /// (exact inputs; only the accumulation order differs from the WGSL op).
    /// Otherwise f32 FMA in the WGSL op's order with the same f16 store:
    /// identical bits.
    /// Enqueued only.
    pub fn run_conv3d_f16(&self, a: &Conv3dAddrs) -> Result<bool, String> {
        let s = a.shape;
        let m = s.t_out * s.h_out * s.w_out;
        if s.kernel.contains(&0) || s.stride.contains(&0) || m == 0 || s.cout == 0 || s.b == 0 {
            return Ok(false);
        }
        let mut p: [u32; 18] = [
            s.b,
            s.cin,
            s.cout,
            s.t,
            s.h,
            s.w,
            s.t_out,
            s.h_out,
            s.w_out,
            s.kernel[0],
            s.kernel[1],
            s.kernel[2],
            s.pad[0],
            s.pad[1],
            s.pad[2],
            s.stride[0],
            s.stride[1],
            s.stride[2],
        ];
        let (mut x, mut w, mut bias, mut out) = (a.x, a.w, a.bias, a.out);
        let mut args: [*mut c_void; 5] = [
            (&raw mut x).cast(),
            (&raw mut w).cast(),
            (&raw mut bias).cast(),
            (&raw mut out).cast(),
            (&raw mut p).cast(),
        ];
        let kdim = s.cin * s.kernel.iter().product::<u32>();
        let big = m.div_ceil(CONV_TILE) * s.cout.div_ceil(CONV_TILE) * s.b;
        // The tensor-core gather table holds 32-bit input offsets.
        let in_elems = s.cin as u64 * s.t as u64 * s.h as u64 * s.w as u64;
        let (f, tile) = if kdim.is_multiple_of(8) && in_elems < i32::MAX as u64 {
            (self.conv3d_f16tc, BM)
        } else if big < CONV_MIN_BIG_TILES {
            (self.conv3d_f16_small, CONV_TILE / 2)
        } else {
            (self.conv3d_f16, CONV_TILE)
        };
        self.module.bind()?;
        self.module.launch(
            f,
            (m.div_ceil(tile), s.cout.div_ceil(tile), s.b),
            THREADS,
            0,
            &mut args,
            "conv3d_f16",
        )?;
        Ok(true)
    }

    /// The one-image convolution of [`Conv2dAddrs`] (f32 FMA in the WGSL
    /// op's order: identical bits; or, with [`Conv2dShape::bf16`] and a
    /// `cin * k * k` that is a multiple of 8, bf16 tensor cores). Enqueued only.
    ///
    /// [`Conv2dShape::bf16`]: thinfer_core::backend::Conv2dShape::bf16
    pub fn run_conv2d(&self, a: &Conv2dAddrs) -> Result<bool, String> {
        let s = a.shape;
        if s.k == 0 || s.stride == 0 || s.h_out == 0 || s.w_out == 0 || s.cout == 0 {
            return Ok(false);
        }
        let mut p: [u32; 10] = [
            s.cin,
            s.cout,
            s.h,
            s.w,
            s.h_out,
            s.w_out,
            s.k,
            s.stride,
            s.pad,
            s.reflect as u32,
        ];
        let (mut x, mut w, mut bias, mut out) = (a.x, a.w, a.bias, a.out);
        let mut args: [*mut c_void; 5] = [
            (&raw mut x).cast(),
            (&raw mut w).cast(),
            (&raw mut bias).cast(),
            (&raw mut out).cast(),
            (&raw mut p).cast(),
        ];
        let m = s.h_out * s.w_out;
        let big = m.div_ceil(CONV_TILE) * s.cout.div_ceil(CONV_TILE);
        // Too few big tiles to fill the device: the small tile (same bits).
        let (f, tile) = if s.bf16 && (s.cin * s.k * s.k).is_multiple_of(8) {
            (self.conv2d_bf16, BM)
        } else if big < CONV_MIN_BIG_TILES {
            (self.conv2d_small, CONV_TILE / 2)
        } else {
            (self.conv2d, CONV_TILE)
        };
        self.module.bind()?;
        self.module.launch(
            f,
            (m.div_ceil(tile), s.cout.div_ceil(tile), 1),
            THREADS,
            0,
            &mut args,
            "conv2d",
        )?;
        Ok(true)
    }

    /// `out = a @ b` per `desc` on device addresses; `m`/`n` from the caller,
    /// `k` from the caller or (`k == 0`) read from `dims`. `Ok(false)` =
    /// unsupported shape or (checked) a bf16 weight past the f16 range
    /// (caller runs wgpu). Enqueued only, except a checked narrowing form,
    /// which must read its overflow flag and so waits. A non-empty epilogue
    /// (see [`thinfer_core::backend::GemmEpilogue`]) is taken on the bf16-output forms, and a bias
    /// alone on the f16-output ones too.
    pub fn run(&self, desc: &GemmDesc, addrs: &GemmAddrs) -> Result<bool, String> {
        let GemmAddrs {
            a,
            b,
            out,
            dims,
            m,
            n,
            k,
            epilogue: epi,
            checked,
        } = *addrs;
        let skinny = is_skinny_f32(desc);
        let Some(f) = (if skinny {
            Some(self.skinny_f32)
        } else {
            index(desc).map(|i| self.funcs[i])
        }) else {
            return Ok(false);
        };
        let bf16_out = desc.a == GemmType::Bf16 && !desc.out_f32;
        // Bias / gate are read as bf16 pairs, the residual like the output;
        // an f16 output takes the bias only.
        let bias_only = epi.residual == 0 && epi.gate == 0;
        let epi_ok = epi.is_empty()
            || ((bf16_out || (bias_only && !desc.out_f32))
                && [epi.bias, epi.gate].iter().all(|p| p.is_multiple_of(4))
                && epi.residual.is_multiple_of(8));
        if !epi_ok {
            return Ok(false);
        }
        let dm: GemmDims = if k == 0 {
            self.module.read(dims)?
        } else {
            GemmDims { m, n, k, pad: 0 }
        };
        // 16-byte vector loads along K (A, n-major B) or N (k-major B);
        // paired output stores.
        let aligned = [a, b].iter().all(|p| p.is_multiple_of(16)) && out.is_multiple_of(8);
        if !aligned
            || (skinny && (m > SKINNY_M || !epi.is_empty()))
            || dm.m != m
            || dm.n != n
            || m == 0
            || n == 0
            || dm.k == 0
            || !dm.k.is_multiple_of(8)
            || !n.is_multiple_of(8)
        {
            return Ok(false);
        }
        let narrowing = desc.a == GemmType::F16 && desc.b == GemmType::Bf16;
        let narrows = narrowing && checked;
        if narrows {
            self.module.write(self.overflow, 0u32)?;
        } else if narrowing {
            self.unchecked
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let (mut a, mut b, mut out, mut dims) = (a, b, out, dm);
        if skinny {
            let mut args: [*mut c_void; 4] = [
                (&raw mut a).cast(),
                (&raw mut b).cast(),
                (&raw mut out).cast(),
                (&raw mut dims).cast(),
            ];
            self.module.bind()?;
            self.module.launch(
                f,
                (n.div_ceil(SKINNY_BN), 1, 1),
                THREADS,
                0,
                &mut args,
                "gemm",
            )?;
            return Ok(true);
        }
        let mut e = [epi.bias, epi.residual, epi.gate];
        let mut args: [*mut c_void; 5] = [
            (&raw mut a).cast(),
            (&raw mut b).cast(),
            (&raw mut out).cast(),
            (&raw mut dims).cast(),
            (&raw mut e).cast(),
        ];
        self.module.bind()?;
        self.module.launch(
            f,
            (n.div_ceil(BN) * m.div_ceil(BM), 1, 1),
            THREADS,
            0,
            &mut args,
            "gemm",
        )?;
        if narrows && self.take_overflow()? {
            return Ok(false);
        }
        Ok(true)
    }

    /// Enqueue the 8-bit GEMM `out = bf16(acc * sa[m] * sb[n] + bias[n])`
    /// on device addresses (see [`GemmQ8Addrs`]). `Ok(false)` = shape or
    /// format (FP8 below sm_89) not covered.
    pub fn run_q8(&self, p: &GemmQ8Addrs, m: u32, n: u32, k: u32) -> Result<bool, String> {
        let Some(f) = self.gemm_q8[p.format as usize] else {
            return Ok(false);
        };
        if m < MIN_M || n == 0 || !n.is_multiple_of(8) || k == 0 || !k.is_multiple_of(16) {
            return Ok(false);
        }
        let mut dims = GemmDims {
            m,
            n,
            k: k / 2,
            pad: 0,
        };
        let (mut a, mut sa, mut b, mut sb, mut out, mut bias) =
            (p.a, p.sa, p.b, p.sb, p.out, p.bias);
        let mut args: [*mut c_void; 7] = [
            (&raw mut a).cast(),
            (&raw mut sa).cast(),
            (&raw mut b).cast(),
            (&raw mut sb).cast(),
            (&raw mut out).cast(),
            (&raw mut dims).cast(),
            (&raw mut bias).cast(),
        ];
        self.module.bind()?;
        self.module.launch(
            f,
            (n.div_ceil(BN) * m.div_ceil(BM), 1, 1),
            THREADS,
            0,
            &mut args,
            "gemm_q8",
        )?;
        Ok(true)
    }

    /// Enqueue the per-row 8-bit quantization of a bf16 `[rows, k]` (`x`)
    /// into `q` (`[rows, k]` in `format`) and `scale` (f32 `[rows]`): int8 `q
    /// = round(x * 127 / max|row|)`, `scale = max|row| / 127`; e4m3 the same
    /// over 16 (`F8_RANGE`: keeps the f16-accumulate MMA partials finite).
    /// `Ok(false)` = FP8 below sm_89.
    pub fn quant_rows_q8(
        &self,
        format: Q8,
        x: u64,
        q: u64,
        scale: u64,
        rows: u32,
        k: u32,
    ) -> Result<bool, String> {
        let Some(f) = self.quant_rows_q8[format as usize] else {
            return Ok(false);
        };
        let (mut x, mut q, mut scale, mut k) = (x, q, scale, k);
        let mut args: [*mut c_void; 4] = [
            (&raw mut x).cast(),
            (&raw mut q).cast(),
            (&raw mut scale).cast(),
            (&raw mut k).cast(),
        ];
        self.module.bind()?;
        self.module
            .launch(f, (rows, 1, 1), THREADS, 0, &mut args, "quant_rows_q8")?;
        Ok(true)
    }

    /// Whether a narrowing GEMM overflowed since the last call (a blocking
    /// read), clearing the flag.
    pub fn take_overflow(&self) -> Result<bool, String> {
        self.clear_reported()?;
        let over = self.module.read::<u32>(self.overflow)? != 0;
        if over {
            self.module.write(self.overflow, 0u32)?;
        }
        Ok(over)
    }

    /// Without waiting: enqueue a copy of the flag for the unchecked GEMMs
    /// since the last call into the pinned probe, to read (with
    /// [`OverflowProbe::take`]) once a later fence completes. `None` with no
    /// copy when none ran.
    pub fn probe_unchecked_overflow(&self) -> Result<Option<Arc<OverflowProbe>>, String> {
        if !self
            .unchecked
            .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            return Ok(None);
        }
        self.clear_reported()?;
        self.module.bind()?;
        // SAFETY: a 4-byte copy into the probe's pinned word on the default
        // stream, ordered after every GEMM enqueued so far.
        crate::nvrtc::drv(
            unsafe {
                sys::cuMemcpyDtoHAsync_v2(
                    self.probe.host.cast(),
                    self.overflow,
                    std::mem::size_of::<u32>(),
                    std::ptr::null_mut(),
                )
            },
            "cuMemcpyDtoHAsync",
        )?;
        Ok(Some(Arc::clone(&self.probe)))
    }

    /// Clear the device flag once a probe has reported it.
    fn clear_reported(&self) -> Result<(), String> {
        if self
            .probe
            .reset_due
            .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            self.module.write(self.overflow, 0u32)?;
        }
        Ok(())
    }

    /// [`Self::take_overflow`] for the unchecked GEMMs since the last call;
    /// `false` with no read when none ran.
    pub fn take_unchecked_overflow(&self) -> Result<bool, String> {
        if !self
            .unchecked
            .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            return Ok(false);
        }
        self.take_overflow()
    }
}
