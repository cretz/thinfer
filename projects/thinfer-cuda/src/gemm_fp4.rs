//! Native NVFP4 (W4A4) matmul on Blackwell block-scaled tensor cores: the
//! checkpoint's own inference numerics (FastVideo runs these weights through
//! FlashInfer `nvfp4_quantize` + `mm_fp4`), instead of dequantizing the
//! weight to bf16 every step.
//!
//! Two kernels, enqueued in order (after `nvfp4_amax` + `nvfp4_act_scale`
//! when the checkpoint's activation scale is dynamic, see `Nvfp4Format`):
//! 1. `nvfp4_quant`: bf16 activations `[M,K]` -> E2M1 codes `[M,K/2]` (low
//!    nibble = even k, or high for a hi-first checkpoint) + one E4M3 scale
//!    per 16-element block, tiled so each 128-row x 128-k strip's scales are
//!    one contiguous 1 KB ([`act_scale_index`]), against the activation
//!    global scale. Same arithmetic as FlashInfer: `sf = e4m3(gsf * amax /
//!    6)`, `q = e2m1_rne(x * gsf / sf)`.
//! 2. `gemm_nvfp4`: `mma.sync.m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X`
//!    over the activation codes and the weight composite
//!    ([`thinfer_core::ops::nvfp4`]) as stored: packed `[N,K/2]` rows are the
//!    MMA's col-major B operand, and a thread's four k-block scales are one
//!    aligned u32 in the `layout_128x4` swizzle. The epilogue applies
//!    `alpha[n/128] / input_sf` and stores bf16.
//!
//! A block owns a 128x256 output tile (8 consumer warps as 2x4, 64x64 per
//! warp) and walks K in 128-element (64-byte) strips through a 3-stage
//! shared-memory ring that a producer warp fills by TMA and bulk copies,
//! codes AND block scales (a scale read from global inside the k loop stalls
//! every MMA behind it), synchronized per stage by mbarriers. Tiles are
//! rasterized in groups of `GROUP_M` row tiles, so the blocks resident at once
//! share a few A row tiles and B column tiles in L2: a wide weight (qkv, fc1:
//! 58-77 MB) is otherwise re-streamed from DRAM once per row tile. An optional
//! low-rank term (a runtime adapter's `mid @ B^T`, rank a multiple of 16 up to
//! 64) seeds the accumulator through bf16 MMAs before the main loop.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaContext, sys};
use thinfer_core::backend::{GemmNvfp4Addrs, Nvfp4Epilogue, Nvfp4Rows, nvfp4_fuses_lora_rank};
use thinfer_core::ops::nvfp4::Nvfp4Layout;

use crate::elementwise::Seg;
use crate::nvrtc::Module;

const BM: u32 = 128;
const BN: u32 = 256;
const BK: u32 = 128;
/// Mirrors the source's `STAGES` / `STAGE_BYTES` (dynamic shared memory).
const STAGES: u32 = 3;
/// Plus the slack that aligns the ring to the swizzle's 1 KB repeat.
const SMEM: u32 = STAGES * ((BM + BN) * 64 + BM * 8 + (BN / 128) * 1024) + 1024;
const THREADS: u32 = 256;
/// Below this many rows the tile is mostly empty and the dequant path wins.
pub const MIN_M: u32 = 64;

const SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BM 128
#define BN 256
#define ROWB 64
#define GROUP_M 8
#define STAGES 3
// One pipeline stage: A + B code rows, A scale words, B scale chunks.
#define STAGE_BYTES ((BM + BN) * ROWB + BM * 8 + (BN / 128) * 1024)
#define MAX_SEG 16

// `glu`: the weight is a `[gate; value]` composite of 2n rows and the output
// `[m, n]` its SwiGLU (see `gemm_nvfp4`).
struct Dims { unsigned m, n, k, glu; };
// The qk-norm + rope epilogue (`qk_norm_rope`'s slab layout, head_dim 128,
// half-rotation of the leading 2 * rot_pairs channels): off when `q` is 0.
// `wbase` = the composite row of this call's first weight row; `inner` = one
// projection's width.
struct QkN {
    unsigned long long qn, kn, freqs, q, k, v;
    unsigned inner, wbase;
    float eps;
    unsigned rot_pairs;
};
// Row segments of the gated residual (as the elementwise kernels' `Seg`).
struct Seg { unsigned n; unsigned end[MAX_SEG]; unsigned long long a[MAX_SEG]; unsigned long long b[MAX_SEG]; };

__device__ __forceinline__ float bf(unsigned short v) { return __uint_as_float((unsigned)v << 16); }
__device__ __forceinline__ unsigned short rn(float x) { return __bfloat16_as_ushort(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float rnf(float x) { return bf(rn(x)); }

// Byte offset of 16-byte chunk `c` (0..3) of code row `r` in a stage: rows
// are unpadded, the chunk XOR-swizzled by row pair (TMA's 64B swizzle) so
// ldmatrix's eight rows cover all 32 banks.
__device__ __forceinline__ int swz(int r, int c) { return r * ROWB + ((c ^ ((r >> 1) & 3)) << 4); }

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}
// Opaque `CUtensorMap` (a `__grid_constant__` kernel parameter).
struct __align__(64) TMap { unsigned long long v[16]; };

__device__ __forceinline__ void mbar_init(unsigned long long* b, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_addr(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(unsigned long long* b, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(smem_addr(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(unsigned long long* b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(smem_addr(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned long long* b, unsigned parity) {
    asm volatile(
        "{\n.reg .pred p;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra WAIT_%=;\n}\n" :: "r"(smem_addr(b)), "r"(parity) : "memory");
}
// The box at (x bytes, y rows) of `map` -> `dst`, completing on `b`.
__device__ __forceinline__ void tma_2d(unsigned char* dst, const TMap* map, unsigned x, unsigned y,
                                       unsigned long long* b) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(smem_addr(dst)), "l"(reinterpret_cast<unsigned long long>(map)), "r"(x), "r"(y), "r"(smem_addr(b))
        : "memory");
}
__device__ __forceinline__ void bulk_copy(unsigned char* dst, const unsigned char* src, unsigned bytes,
                                          unsigned long long* b) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :: "r"(smem_addr(dst)), "l"(src), "r"(bytes), "r"(smem_addr(b)) : "memory");
}
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], const unsigned char* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned (&r)[4], const unsigned short* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_fp4(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1,
                                        unsigned sa, unsigned sb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}

// Round-to-nearest-even onto E2M1 {0, .5, 1, 1.5, 2, 3, 4, 6}, saturating.
__device__ __forceinline__ unsigned e2m1(float v) {
    const float a = fabsf(v);
    const unsigned c = a <= 0.25f ? 0u : a < 0.75f ? 1u : a <= 1.25f ? 2u : a < 1.75f ? 3u
                     : a <= 2.5f ? 4u : a < 3.5f ? 5u : a <= 5.f ? 6u : 7u;
    return c | (signbit(v) ? 8u : 0u);
}

// Dynamic activation scale (Comfy PTQ): `amax(|x|)` over the whole input into
// `slot[0]` (zeroed first; non-negative floats order as their bits).
extern "C" __global__ void __launch_bounds__(256) nvfp4_amax(
    const unsigned short* __restrict__ x, size_t n, unsigned* __restrict__ slot)
{
    float amax = 0.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n / 2;
         i += (size_t)gridDim.x * blockDim.x) {
        const unsigned w = reinterpret_cast<const unsigned*>(x)[i];
        amax = fmaxf(amax, fmaxf(fabsf(__uint_as_float(w << 16)), fabsf(__uint_as_float(w & 0xffff0000u))));
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    __shared__ float red[8];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 8; ++w) amax = fmaxf(amax, red[w]);
        atomicMax(slot, __float_as_uint(amax));
    }
}

// `slot[1]` = the global scale the quant / GEMM read: `1 / bf16(amax / 2688)`.
extern "C" __global__ void nvfp4_act_scale(unsigned* __restrict__ slot) {
    const float scale = __bfloat162float(__float2bfloat16(__uint_as_float(slot[0]) * (1.f / 2688.f)));
    reinterpret_cast<float*>(slot)[1] = scale > 0.f ? 1.f / scale : 1.f;
}

extern "C" __global__ void __launch_bounds__(256) nvfp4_quant(
    const unsigned short* __restrict__ x, unsigned char* __restrict__ q,
    unsigned char* __restrict__ sf, const float* __restrict__ gsf_p, unsigned m, unsigned k,
    unsigned hi_first)
{
    const unsigned kb = k / 16;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)m * kb) return;
    const size_t row = i / kb, b = i % kb;
    const uint4* src = reinterpret_cast<const uint4*>(x + row * k + b * 16);
    const uint4 v0 = src[0], v1 = src[1];
    const unsigned w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    float f[16];
    float amax = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        f[2 * j] = __uint_as_float(w[j] << 16);
        f[2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u);
        amax = fmaxf(amax, fmaxf(fabsf(f[2 * j]), fabsf(f[2 * j + 1])));
    }
    const float gsf = *gsf_p;
    const __nv_fp8_storage_t s8 = __nv_cvt_float_to_fp8(gsf * (amax * (1.f / 6.f)), __NV_SATFINITE, __NV_E4M3);
    const float sq = __half2float(__half(__nv_cvt_fp8_to_halfraw(s8, __NV_E4M3)));
    const float scale = sq != 0.f ? 1.f / (sq * (1.f / gsf)) : 0.f;
    unsigned long long packed = 0;
    #pragma unroll
    for (int j = 0; j < 16; ++j) packed |= (unsigned long long)e2m1(f[j] * scale) << (4 * (j ^ hi_first));
    *reinterpret_cast<unsigned long long*>(q + row * (k / 2) + b * 8) = packed;
    sf[((row >> 7) * (kb >> 3) + (b >> 3)) * 1024 + (row & 127) * 8 + (b & 7)] = s8;
}

// `w` / `wsf` / `alpha`: the weight's packed rows (`tma_b`), swizzled scales
// and per-group alphas, each already offset to the first of this call's
// `d.n` rows (a composite may be multiplied over a 128-row-aligned
// sub-range). `tma_a` maps the activation codes, `sa` their tiled scales.
// `ldb`: the row stride of the low-rank `lbt` (whose columns are weight
// rows). Under `d.glu` a tile multiplies gate rows [n0, n0 + 128) and value
// rows [n + n0, n + n0 + 128) and stores their SwiGLU. A non-null `resid` makes the
// store `rn(resid + a[seg][col] * rn(acc))` (the separate gate-add pass's
// two roundings). A non-null `bias` (bf16, per weight row) first makes each
// product `rn(rn(acc) + bias)`, as the separate `add_bias` pass.
//
// Strips stream through STAGES stages by TMA (codes, 64B-swizzled = `swz`,
// rows past M/N zero-filled) and bulk copies (scales), each landing on its
// stage's `full` barrier. A warp releases a stage (`empty`) once its
// fragments are read; thread 0 refills it when all eight have, so no
// block-wide barrier sits in the K loop. (A ninth, producer warp would put
// three warps on one sub-partition: 168 registers, accumulator spills.)
extern "C" __global__ void __launch_bounds__(256) gemm_nvfp4(
    const __grid_constant__ TMap tma_a, const __grid_constant__ TMap tma_b,
    const unsigned char* __restrict__ sa, const unsigned char* __restrict__ wsf,
    const float* __restrict__ alpha, unsigned short* __restrict__ out, Dims d,
    const unsigned short* __restrict__ lmid, const unsigned short* __restrict__ lbt, unsigned ldb,
    unsigned lrank, const float* __restrict__ insf, const unsigned short* __restrict__ bias,
    const unsigned short* __restrict__ resid, const Seg seg, const QkN qkn)
{
    // Per stage: A / B code rows, A's scale tile (128 rows x 8 bytes), B's
    // two 512-byte swizzled chunks per 128-column group (consecutive in
    // `layout_128x4`). The swizzle repeats every 512 bytes: align the base.
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __shared__ __align__(8) unsigned long long full[STAGES], empty[STAGES];
    unsigned char* sm = smem_raw + ((1024 - (smem_addr(smem_raw) & 1023)) & 1023);
    auto as = [&](int st) { return sm + st * STAGE_BYTES; };
    auto bs = [&](int st) { return sm + st * STAGE_BYTES + BM * ROWB; };
    auto ssa = [&](int st) { return reinterpret_cast<unsigned*>(sm + st * STAGE_BYTES + (BM + BN) * ROWB); };
    auto ssb = [&](int st) { return sm + st * STAGE_BYTES + (BM + BN) * ROWB + BM * 8; };

    const unsigned kb = d.k / 16;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int wm0 = ((warp >> 2) & 1) * 64, wn0 = (warp & 3) * 64;
    // Output columns per tile: under `d.glu` the tile's two 128-column
    // halves are the gate and value rows of the same 128 outputs.
    const unsigned tw = d.glu ? BN / 2 : BN;
    // Grouped rasterization over a 1-D grid.
    const unsigned mt = (d.m + BM - 1) / BM, nt = (d.n + tw - 1) / tw;
    const unsigned per_group = GROUP_M * nt, pid = blockIdx.x;
    const unsigned first_m = (pid / per_group) * GROUP_M;
    const unsigned gsize = min(mt - first_m, (unsigned)GROUP_M);
    const unsigned m0 = (first_m + (pid % per_group) % gsize) * BM;
    const unsigned n0 = ((pid % per_group) / gsize) * tw;
    const unsigned ntiles = d.k / 128;
    // Weight row of tile column c.
    auto wrow = [&](unsigned c) { return d.glu ? (c >> 7) * d.n + n0 + (c & 127) : n0 + c; };
    // The tile's 128-column scale groups (the second absent when N ends at
    // its first).
    const unsigned groups = d.glu ? 2u : min((d.n - n0) / 128, BN / 128u);
    // A warp's 64 columns lie in one group; past N it only idles.
    const bool warp_live = wn0 < groups * 128;

    if (tid == 0) {
        #pragma unroll
        for (int st = 0; st < STAGES; ++st) {
            mbar_init(&full[st], 1);
            mbar_init(&empty[st], 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    float acc[4][8][4];
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi)
        #pragma unroll
        for (int ni = 0; ni < 8; ++ni)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    // Low-rank term first, seeding the accumulator while registers are free
    // (in the epilogue it pushed the live accumulator into spills): mid rows
    // [BM][lrank] then B^T columns [lrank][BN] at the start of shared memory
    // (rank <= 64 fits, before the pipeline's first fill), then bf16 MMAs in
    // the accumulator's own m16n8 layout. A warp's columns are one 128-column
    // alpha group, so the term enters divided by that group's scale and comes
    // out exact under the epilogue's multiply.
    if (lrank) {
        // Rows padded by 8 elements so ldmatrix's eight rows hit distinct banks.
        const unsigned rc = lrank / 8, mld = lrank + 8;
        constexpr unsigned BLD = BN + 8;
        unsigned short* ms = reinterpret_cast<unsigned short*>(sm);
        unsigned short* bts = ms + BM * mld;
        for (unsigned x = tid; x < BM * rc; x += 256) {
            const unsigned r = x / rc, c = x % rc;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (m0 + r < d.m) v = *reinterpret_cast<const uint4*>(lmid + (size_t)(m0 + r) * lrank + c * 8);
            *reinterpret_cast<uint4*>(ms + r * mld + c * 8) = v;
        }
        for (unsigned x = tid; x < lrank * (BN / 8); x += 256) {
            const unsigned r = x / (BN / 8), c = x % (BN / 8);
            uint4 v = make_uint4(0, 0, 0, 0);
            if (c * 8 < groups * 128) v = *reinterpret_cast<const uint4*>(lbt + (size_t)r * ldb + wrow(c * 8));
            *reinterpret_cast<uint4*>(bts + r * BLD + c * 8) = v;
        }
        __syncthreads();
        if (warp_live) {
            for (unsigned kk = 0; kk < lrank / 16; ++kk) {
                #pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    unsigned af[4];
                    ldsm_x4(af, reinterpret_cast<const unsigned char*>(
                        ms + (wm0 + mi * 16 + (lane & 15)) * mld + kk * 16 + (lane >> 4) * 8));
                    #pragma unroll
                    for (int nj = 0; nj < 4; ++nj) {
                        unsigned r[4];
                        ldsm_x4_t(r, bts + (kk * 16 + (lane & 7) + (((lane >> 3) & 1) << 3)) * BLD
                                         + wn0 + nj * 16 + (lane >> 4) * 8);
                        mma_bf16(acc[mi][2 * nj], af, r[0], r[1]);
                        mma_bf16(acc[mi][2 * nj + 1], af, r[2], r[3]);
                    }
                }
            }
            const float inv_s = insf[0] / alpha[wrow(wn0) >> 7];
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                #pragma unroll
                for (int ni = 0; ni < 8; ++ni)
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) acc[mi][ni][e] *= inv_s;
        }
        __syncthreads(); // the stages are filled next
    }

    // Per group: 128 weight rows (one box) and their 1 KB scale chunk.
    const unsigned bytes = BM * ROWB + BM * 8 + groups * (128 * ROWB + 1024);
    const unsigned char* sa_t = sa + (size_t)(m0 >> 7) * ntiles * 1024;
    auto load = [&](unsigned kt, int st) {
        mbar_expect_tx(&full[st], bytes);
        tma_2d(as(st), &tma_a, kt * ROWB, m0, &full[st]);
        bulk_copy(reinterpret_cast<unsigned char*>(ssa(st)), sa_t + (size_t)kt * 1024, 1024, &full[st]);
        for (unsigned grp = 0; grp < groups; ++grp) {
            const unsigned r = wrow(grp * 128);
            tma_2d(bs(st) + grp * 128 * ROWB, &tma_b, kt * ROWB, r, &full[st]);
            bulk_copy(ssb(st) + grp * 1024, wsf + (size_t)(r >> 7) * (kb >> 2) * 512 + (size_t)kt * 1024, 1024,
                      &full[st]);
        }
    };
    if (tid == 0) {
        // The low-rank staging above was generic-proxy shared memory.
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        for (unsigned kt = 0; kt < min(ntiles, (unsigned)STAGES); ++kt) load(kt, kt);
    }

    for (unsigned kt = 0; kt < ntiles; ++kt) {
        const int st = kt % STAGES;
        // Strip kt + STAGES - 1 refills the stage strip kt - 1 used.
        if (tid == 0 && kt > 0 && kt + STAGES - 1 < ntiles) {
            const int rs = (kt - 1) % STAGES;
            mbar_wait(&empty[rs], ((kt - 1) / STAGES) & 1);
            load(kt + STAGES - 1, rs);
        }
        mbar_wait(&full[st], (kt / STAGES) & 1);
        if (warp_live) {
            const unsigned char* a_s = as(st);
            const unsigned char* b_s = bs(st);
            const unsigned* sa_s = ssa(st);
            const unsigned char* sb_s = ssb(st) + ((wn0 >> 7) << 10);
            // A row's two k64 scale words, both halves of the strip at once.
            uint2 sfa2[4];
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                sfa2[mi] = *reinterpret_cast<const uint2*>(sa_s + (wm0 + mi * 16 + g + (lane & 1) * 8) * 2);
            #pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                unsigned af[4][4], sfa[4];
                #pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    ldsm_x4(af[mi], a_s + swz(wm0 + mi * 16 + (lane & 15), kk * 2 + (lane >> 4)));
                    sfa[mi] = kk ? sfa2[mi].y : sfa2[mi].x;
                }
                unsigned bf[8][2], sfb[8];
                #pragma unroll
                for (int nj = 0; nj < 4; ++nj) {
                    unsigned r[4];
                    ldsm_x4(r, b_s + swz(wn0 + nj * 16 + (lane & 7) + ((lane >> 4) << 3),
                                         kk * 2 + ((lane >> 3) & 1)));
                    bf[2 * nj][0] = r[0]; bf[2 * nj][1] = r[1];
                    bf[2 * nj + 1][0] = r[2]; bf[2 * nj + 1][1] = r[3];
                }
                // Columns n and n + 32 sit in adjacent words of `layout_128x4`
                // (wn0 % 128 is 0 or 64, so n % 32 == ni * 8 + g for ni < 4).
                #pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    const uint2 p = *reinterpret_cast<const uint2*>(
                        sb_s + kk * 512 + (ni * 8 + g) * 16 + ((wn0 & 127) >> 5) * 4);
                    sfb[ni] = p.x;
                    sfb[ni + 4] = p.y;
                }
                #pragma unroll
                for (int mi = 0; mi < 4; ++mi)
                    #pragma unroll
                    for (int ni = 0; ni < 8; ++ni)
                        mma_fp4(acc[mi][ni], af[mi], bf[ni][0], bf[ni][1], sfa[mi], sfb[ni]);
            }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[st]);
    }
    if (!warp_live) return;

    // A warp's 64 columns share one alpha group.
    const float s = alpha[wrow(wn0) >> 7] * (1.f / insf[0]);
    // The bf16 product of accumulator pair (a0, a1) at weight row w, plus
    // `bias[w..w+2]` rounded as the separate `add_bias` pass rounds it.
    auto product = [&](float a0, float a1, unsigned w) {
        __nv_bfloat162 v = __floats2bfloat162_rn(a0 * s, a1 * s);
        if (bias) {
            const unsigned bw = *reinterpret_cast<const unsigned*>(bias + w);
            const unsigned pw = *reinterpret_cast<const unsigned*>(&v);
            v = __floats2bfloat162_rn(bf(pw & 0xffffu) + bf(bw & 0xffffu), bf(pw >> 16) + bf(bw >> 16));
        }
        return *reinterpret_cast<const unsigned*>(&v);
    };

    if (qkn.q) {
        // Park the bf16 product tile, then norm + rope each (row, head) with
        // one warp as `qk_norm_rope` does with one block: channel c = 32j +
        // lane, the sum of squares as `block_sum` adds it (an xor tree per 32
        // channels, the four totals added from 0 in channel order), the normed
        // row in the warp's own 128 floats for the rope partners. Every strip
        // was consumed, so the ring is free. Tile rows padded by 8 elements
        // against bank conflicts.
        constexpr unsigned TLD = BN + 8;
        unsigned short* ts = reinterpret_cast<unsigned short*>(sm);
        float* yw = reinterpret_cast<float*>(sm + BM * TLD * 2) + warp * 128;
        __syncthreads();
        #pragma unroll
        for (int ni = 0; ni < 8; ++ni)
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const unsigned r = wm0 + mi * 16 + g + h * 8, c = wn0 + ni * 8 + 2 * t;
                    *reinterpret_cast<unsigned*>(ts + r * TLD + c) =
                        product(acc[mi][ni][2 * h], acc[mi][ni][2 * h + 1], n0 + c);
                }
        __syncthreads();
        const unsigned short* qn = reinterpret_cast<const unsigned short*>(qkn.qn);
        const unsigned short* kn = reinterpret_cast<const unsigned short*>(qkn.kn);
        const unsigned short* freqs = reinterpret_cast<const unsigned short*>(qkn.freqs);
        // A tile is two heads of one section (inner is a 256 multiple): this
        // warp's norm weights load once, each row's rope pairs once for both.
        const unsigned sec = (qkn.wbase + n0) / qkn.inner, rp = qkn.rot_pairs;
        const unsigned short* gw = sec ? kn : qn;
        float wj[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) wj[j] = gw ? bf(gw[32 * j + lane]) : 1.f;
        unsigned short* dst = reinterpret_cast<unsigned short*>(sec == 2 ? qkn.v : sec ? qkn.k : qkn.q);
        for (unsigned r = warp; r < BM && m0 + r < d.m; r += 8) {
            const unsigned row = m0 + r;
            const size_t o = (size_t)row * qkn.inner + (qkn.wbase + n0) % qkn.inner + lane;
            const unsigned short* src = ts + r * TLD + lane;
            if (sec == 2) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) dst[o + 32 * j] = src[32 * j];
                continue;
            }
            const unsigned short* fr = freqs + (size_t)row * rp * 2;
            float cr[4], ci[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned kk = (32 * j + lane) % rp;
                cr[j] = 32 * j + lane < 2 * rp ? bf(fr[kk * 2]) : 0.f;
                ci[j] = 32 * j + lane < 2 * rp ? bf(fr[kk * 2 + 1]) : 0.f;
            }
            #pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                float x[4], tot = 0.f;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    x[j] = bf(src[hh * 128 + 32 * j]);
                    float ss = x[j] * x[j];
                    #pragma unroll
                    for (int sh = 16; sh > 0; sh >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, sh);
                    tot += ss;
                }
                const float inv = rsqrtf(tot / 128.f + qkn.eps);
                float y[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    y[j] = rnf(x[j] * inv * wj[j]);
                    yw[32 * j + lane] = y[j];
                }
                __syncwarp();
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const unsigned c = 32 * j + lane;
                    const float n = y[j];
                    float out = n;
                    if (c < 2 * rp)
                        out = c < rp ? n * cr[j] - yw[c + rp] * ci[j] : yw[c - rp] * ci[j] + n * cr[j];
                    dst[o + hh * 128 + 32 * j] = rn(out);
                }
                __syncwarp();
            }
        }
        return;
    }

    if (d.glu) {
        // Value warps (tile columns 128..255) park bf16(value) in shared
        // memory; gate warps store `bf16(silu(bf16(gate)) * value)`, the
        // product + `silu_mul_fused` roundings. Every strip was consumed, so
        // the ring is free. Rows padded by 8 elements against bank conflicts.
        constexpr unsigned VLD = 128 + 8;
        unsigned short* vs = reinterpret_cast<unsigned short*>(sm);
        const bool value = wn0 >= 128;
        __syncthreads();
        #pragma unroll
        for (int ni = 0; ni < 8; ++ni)
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    if (!value) continue;
                    const unsigned r = wm0 + mi * 16 + g + h * 8, c = (wn0 & 127) + ni * 8 + 2 * t;
                    *reinterpret_cast<unsigned*>(vs + r * VLD + c) =
                        product(acc[mi][ni][2 * h], acc[mi][ni][2 * h + 1], wrow(128 + c));
                }
        __syncthreads();
        if (value) return;
        #pragma unroll
        for (int ni = 0; ni < 8; ++ni)
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const unsigned r = wm0 + mi * 16 + g + h * 8, c = wn0 + ni * 8 + 2 * t;
                    if (m0 + r >= d.m) continue;
                    const unsigned vw = *reinterpret_cast<const unsigned*>(vs + r * VLD + c);
                    const unsigned gw = product(acc[mi][ni][2 * h], acc[mi][ni][2 * h + 1], wrow(c));
                    float o[2];
                    #pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const float gt = bf(e ? gw >> 16 : gw & 0xffffu);
                        o[e] = gt / (1.f + expf(-gt)) * bf(e ? vw >> 16 : vw & 0xffffu);
                    }
                    *reinterpret_cast<__nv_bfloat162*>(out + (size_t)(m0 + r) * d.n + n0 + c) =
                        __floats2bfloat162_rn(o[0], o[1]);
                }
        return;
    }

    #pragma unroll
    for (int ni = 0; ni < 8; ++ni) {
        const unsigned col = n0 + wn0 + ni * 8 + 2 * t;
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const unsigned row = m0 + wm0 + mi * 16 + g + h * 8;
                if (row >= d.m) continue;
                unsigned pw = product(acc[mi][ni][2 * h], acc[mi][ni][2 * h + 1], col);
                const size_t o = (size_t)row * d.n + col;
                if (resid) {
                    unsigned si = 0;
                    while (si + 1 < seg.n && row >= seg.end[si]) ++si;
                    const unsigned gw = *reinterpret_cast<const unsigned*>(
                        reinterpret_cast<const unsigned short*>(seg.a[si]) + col);
                    const unsigned rw = *reinterpret_cast<const unsigned*>(resid + o);
                    const __nv_bfloat162 v = __floats2bfloat162_rn(
                        bf(rw & 0xffffu) + bf(gw & 0xffffu) * bf(pw & 0xffffu),
                        bf(rw >> 16) + bf(gw >> 16) * bf(pw >> 16));
                    pw = *reinterpret_cast<const unsigned*>(&v);
                }
                *reinterpret_cast<unsigned*>(out + o) = pw;
            }
        }
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Dims {
    m: u32,
    n: u32,
    k: u32,
    glu: u32,
}

/// Mirrors the source's `QkN`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct QkN {
    qn: u64,
    kn: u64,
    freqs: u64,
    q: u64,
    k: u64,
    v: u64,
    inner: u32,
    wbase: u32,
    eps: f32,
    rot_pairs: u32,
}

/// Scratch bytes [`GemmFp4Kernels::run`] needs for an `[m, k]` activation:
/// codes `[M,K/2]`, block scales (whole 128-row tiles, see
/// [`act_scale_index`]), then the dynamic-scale slot (`amax` bits, scale) at
/// [`slot_offset`].
pub fn scratch_bytes(m: u32, k: u32) -> u64 {
    slot_offset(m, k) + 16
}

/// Byte offset of the dynamic-scale slot in the scratch.
pub fn slot_offset(m: u32, k: u32) -> u64 {
    m as u64 * k as u64 / 2 + m.next_multiple_of(BM) as u64 * k as u64 / 16
}

/// Index of row `row`'s 16-element block `b` in the scratch's scale section:
/// `[M/128][K/128][128 rows][8 blocks]`, one strip's A scales per 1 KB.
pub fn act_scale_index(row: usize, b: usize, k: usize) -> usize {
    ((row / 128) * (k / 128) + b / 8) * 1024 + (row % 128) * 8 + b % 8
}

/// A 2-D `u8` tensor map over `rows` rows of `row_bytes`, boxes of 64 bytes x
/// `box_rows` rows, 64B-swizzled, zero-filled out of bounds.
fn tensor_map(
    base: u64,
    row_bytes: u32,
    rows: u32,
    box_rows: u32,
) -> Result<sys::CUtensorMap, String> {
    let mut map = sys::CUtensorMap { opaque: [0; 16] };
    let dims = [row_bytes as u64, rows as u64];
    let strides = [row_bytes as u64];
    let boxes = [64u32, box_rows];
    let steps = [1u32, 1];
    // SAFETY: every pointer is a live local of the rank the call names; the
    // driver only encodes `base` (device memory of this call's operands).
    crate::nvrtc::drv(
        unsafe {
            sys::cuTensorMapEncodeTiled(
                &mut map,
                sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
                2,
                base as *mut c_void,
                dims.as_ptr(),
                strides.as_ptr(),
                boxes.as_ptr(),
                steps.as_ptr(),
                sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
                sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            )
        },
        "nvfp4 tensor map",
    )?;
    Ok(map)
}

/// Whether the kernels cover this shape (the caller then pays to drain): N
/// in whole 128-column scale groups (a tile's second half may be absent).
pub fn accepts(m: u32, n: u32, k: u32) -> bool {
    m >= MIN_M && n.is_multiple_of(128) && k.is_multiple_of(BK)
}

pub struct GemmFp4Kernels {
    module: Module,
    amax: sys::CUfunction,
    act_scale: sys::CUfunction,
    quant: sys::CUfunction,
    gemm: sys::CUfunction,
}

// SAFETY: function handles are plain driver handles; every use binds the
// owning context first.
unsafe impl Send for GemmFp4Kernels {}
unsafe impl Sync for GemmFp4Kernels {}

impl GemmFp4Kernels {
    /// Block-scaled MMA is arch-specific: sm_120 (consumer Blackwell) only.
    pub fn new(ctx: Arc<CudaContext>, sm: (i32, i32)) -> Result<Self, String> {
        if sm != (12, 0) {
            return Err(format!(
                "nvfp4 mma needs sm_120, device is sm_{}{}",
                sm.0, sm.1
            ));
        }
        let module =
            Module::compile_arch(ctx, SOURCE, "thinfer_gemm_fp4.cu", "sm_120a".to_string())?;
        let amax = module.function(c"nvfp4_amax", 0)?;
        let act_scale = module.function(c"nvfp4_act_scale", 0)?;
        let quant = module.function(c"nvfp4_quant", 0)?;
        let gemm = module.function(c"gemm_nvfp4", SMEM)?;
        Ok(Self {
            module,
            amax,
            act_scale,
            quant,
            gemm,
        })
    }

    /// `out[M,N] (bf16) = a[M,K] (bf16) @ W^T` for the NVFP4 composite at `w`
    /// (or its row range `rows`), W4A4 against the activation scale `format`
    /// names (the composite's static one, or this input's amax), plus
    /// `lora`'s low-rank term when its rank is non-zero. `scratch` holds
    /// [`scratch_bytes`]. Enqueued only (see `Accelerator::sync`).
    /// `Ok(false)` = shape, alignment or rank not covered (caller runs the
    /// dequant path).
    pub fn run(&self, g: &GemmNvfp4Addrs) -> Result<bool, String> {
        let GemmNvfp4Addrs {
            a,
            w,
            out,
            scratch,
            m,
            n,
            k,
            lora,
            format,
            rows,
            bias,
            epilogue,
        } = *g;
        let (gate_add, glu, qk) = match epilogue {
            Nvfp4Epilogue::Store => (None, false, None),
            Nvfp4Epilogue::GateAdd(g) => (Some(g), false, None),
            Nvfp4Epilogue::SwiGlu => (None, true, None),
            Nvfp4Epilogue::QkNormRope(q) => (None, false, Some(q)),
        };
        // Weight rows multiplied: a SwiGLU's gate and value halves.
        let wrows = if glu { 2 * n } else { n };
        let Nvfp4Rows { base, total } = rows.unwrap_or(Nvfp4Rows {
            base: 0,
            total: wrows,
        });
        let aligned = [a, w, scratch].iter().all(|p| p.is_multiple_of(16))
            && out.is_multiple_of(4)
            && bias.is_multiple_of(4);
        let lora_ok = lora.rank == 0
            || (nvfp4_fuses_lora_rank(lora.rank)
                && [lora.mid, lora.bt].iter().all(|p| p.is_multiple_of(16)));
        let range_ok =
            base.is_multiple_of(128) && base + wrows <= total && !(glu && rows.is_some());
        // The gated residual reads bf16 pairs at the product's own offsets.
        let gate_ok = gate_add.is_none_or(|g| {
            rows.is_none()
                && g.residual.is_multiple_of(4)
                && g.residual != out
                && g.seg.n > 0
                && g.seg.a[..g.seg.n as usize]
                    .iter()
                    .all(|a| a.is_multiple_of(4))
        });
        // Whole 256-column tiles inside one projection, which is whole heads.
        let qk_ok = qk.is_none_or(|q| {
            q.inner.is_multiple_of(BN)
                && n.is_multiple_of(BN)
                && base.is_multiple_of(BN)
                && base + n <= 3 * q.inner
                && q.rot_pairs <= 64
        });
        if !aligned || !lora_ok || !range_ok || !gate_ok || !qk_ok || !accepts(m, n, k) {
            return Ok(false);
        }
        // Section pointers of the composite's full layout, at row `base`.
        let layout = Nvfp4Layout::new(total as usize, k as usize);
        let group = base as u64 / 128;
        let wq = w + base as u64 * k as u64 / 2;
        let mut wsf = w + layout.scales_offset() as u64 + group * (k as u64 / 64) * 512;
        if !wsf.is_multiple_of(16) {
            return Ok(false);
        }
        let mut alpha = w + layout.alpha_offset() as u64 + group * 4;
        let (mut a, mut out) = (a, out);
        let mut qa = scratch;
        let mut tma_a = tensor_map(qa, k / 2, m, BM)?;
        let mut tma_b = tensor_map(wq, k / 2, wrows, 128)?;
        let mut sa = scratch + m as u64 * k as u64 / 2;
        let mut gsf = w + layout.input_sf_offset() as u64 + group * 4;
        let (mut m_, mut k_) = (m, k);
        self.module.bind()?;
        if format.dynamic_act_scale {
            let mut slot = scratch + slot_offset(m, k);
            // SAFETY: `slot` is 16 bytes of this call's scratch.
            crate::nvrtc::drv(
                unsafe { sys::cuMemsetD32Async(slot, 0, 2, std::ptr::null_mut()) },
                "nvfp4 amax reset",
            )?;
            let mut elems = m as usize * k as usize;
            let mut args: [*mut c_void; 3] = [
                (&raw mut a).cast(),
                (&raw mut elems).cast(),
                (&raw mut slot).cast(),
            ];
            let grid = (elems as u64 / 2).div_ceil(THREADS as u64 * 8).min(1024) as u32;
            self.module
                .launch(self.amax, (grid, 1, 1), THREADS, 0, &mut args, "nvfp4_amax")?;
            let mut args: [*mut c_void; 1] = [(&raw mut slot).cast()];
            self.module.launch(
                self.act_scale,
                (1, 1, 1),
                1,
                0,
                &mut args,
                "nvfp4_act_scale",
            )?;
            gsf = slot + 4;
        }
        let mut hi_first = format.hi_first as u32;
        let blocks = (m as u64 * k as u64 / 16).div_ceil(THREADS as u64) as u32;
        let mut qargs: [*mut c_void; 7] = [
            (&raw mut a).cast(),
            (&raw mut qa).cast(),
            (&raw mut sa).cast(),
            (&raw mut gsf).cast(),
            (&raw mut m_).cast(),
            (&raw mut k_).cast(),
            (&raw mut hi_first).cast(),
        ];
        self.module.launch(
            self.quant,
            (blocks, 1, 1),
            THREADS,
            0,
            &mut qargs,
            "nvfp4_quant",
        )?;
        let mut dims = Dims {
            m,
            n,
            k,
            glu: glu as u32,
        };
        // `bt` is `[rank, total]` bf16: start at column `base`, stride `total`.
        let (mut lmid, mut lbt, mut ldb, mut lrank) =
            (lora.mid, lora.bt + 2 * base as u64, total, lora.rank);
        // `bias` is per composite row: start at row `base`.
        let mut bias = if bias == 0 { 0 } else { bias + 2 * base as u64 };
        let (mut resid, mut seg) = match gate_add {
            Some(g) => (g.residual, Seg::from(&g.seg)),
            None => (0, Seg::from(&Default::default())),
        };
        let mut qkn = qk.map_or(QkN::default(), |q| QkN {
            qn: q.q_norm,
            kn: q.k_norm,
            freqs: q.freqs,
            q: q.q,
            k: q.k,
            v: q.v,
            inner: q.inner,
            wbase: base,
            eps: q.eps,
            rot_pairs: q.rot_pairs,
        });
        let mut args: [*mut c_void; 16] = [
            (&raw mut tma_a).cast(),
            (&raw mut tma_b).cast(),
            (&raw mut sa).cast(),
            (&raw mut wsf).cast(),
            (&raw mut alpha).cast(),
            (&raw mut out).cast(),
            (&raw mut dims).cast(),
            (&raw mut lmid).cast(),
            (&raw mut lbt).cast(),
            (&raw mut ldb).cast(),
            (&raw mut lrank).cast(),
            (&raw mut gsf).cast(),
            (&raw mut bias).cast(),
            (&raw mut resid).cast(),
            (&raw mut seg).cast(),
            (&raw mut qkn).cast(),
        ];
        self.module.launch(
            self.gemm,
            (
                n.div_ceil(if glu { BN / 2 } else { BN }) * m.div_ceil(BM),
                1,
                1,
            ),
            THREADS,
            SMEM,
            &mut args,
            "gemm_nvfp4",
        )?;
        Ok(true)
    }
}
