//! Native NVFP4 (W4A4) matmul on Blackwell block-scaled tensor cores: the
//! checkpoint's own inference numerics (FastVideo runs these weights through
//! FlashInfer `nvfp4_quantize` + `mm_fp4`), instead of dequantizing the
//! weight to bf16 every step.
//!
//! Two kernels, enqueued in order (after `nvfp4_amax` + `nvfp4_act_scale`
//! when the checkpoint's activation scale is dynamic, see `Nvfp4Format`):
//! 1. `nvfp4_quant`: bf16 activations `[M,K]` -> E2M1 codes `[M,K/2]` (low
//!    nibble = even k, or high for a hi-first checkpoint) + one E4M3 scale
//!    per 16-element block, row-major `[M,K/16]`, against the activation
//!    global scale. Same arithmetic as FlashInfer: `sf = e4m3(gsf * amax /
//!    6)`, `q = e2m1_rne(x * gsf / sf)`.
//! 2. `gemm_nvfp4`: `mma.sync.m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X`
//!    over the activation codes and the weight composite
//!    ([`thinfer_core::ops::nvfp4`]) as stored: packed `[N,K/2]` rows are the
//!    MMA's col-major B operand, and a thread's four k-block scales are one
//!    aligned u32 in the `layout_128x4` swizzle. The epilogue applies
//!    `alpha[n/128] / input_sf` and stores bf16.
//!
//! A block owns a 128x128 output tile (8 warps as 2x4, 64x32 per warp) and
//! walks K in 128-element (64-byte) strips through double-buffered padded
//! shared memory filled by `cp.async`, codes AND block scales (a scale read
//! from global inside the k loop stalls every MMA behind it). Tiles are
//! rasterized in groups of `GROUP_M` row tiles, so the blocks resident at once
//! share a few A row tiles and B column tiles in L2: a wide weight (qkv, fc1:
//! 58-77 MB) is otherwise re-streamed from DRAM once per row tile. An optional
//! low-rank term (a runtime adapter's `mid @ B^T`, rank a multiple of 16 up to
//! 64) seeds the accumulator through bf16 MMAs before the main loop.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaContext, sys};
use thinfer_core::backend::{GemmNvfp4Addrs, Nvfp4Rows, nvfp4_fuses_lora_rank};
use thinfer_core::ops::nvfp4::Nvfp4Layout;

use crate::elementwise::Seg;
use crate::nvrtc::Module;

const BM: u32 = 128;
const BN: u32 = 256;
const BK: u32 = 128;
/// Mirrors the source's `STAGES` / `STAGE_BYTES` (dynamic shared memory).
const STAGES: u32 = 2;
const SMEM: u32 = STAGES * ((BM + BN) * 80 + BM * 8 + (BN / 128) * 1024);
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
#define LDS 80
#define GROUP_M 8
#define STAGES 2
// One pipeline stage: A + B code rows, A scale words, B scale chunks.
#define STAGE_BYTES ((BM + BN) * LDS + BM * 8 + (BN / 128) * 1024)
#define MAX_SEG 16

struct Dims { unsigned m, n, k, pad; };
// Row segments of the gated residual (as the elementwise kernels' `Seg`).
struct Seg { unsigned n; unsigned end[MAX_SEG]; unsigned long long a[MAX_SEG]; unsigned long long b[MAX_SEG]; };

__device__ __forceinline__ float bf(unsigned short v) { return __uint_as_float((unsigned)v << 16); }

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], const unsigned char* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}
__device__ __forceinline__ void cp_async16(unsigned char* s, const unsigned char* g, bool pred) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem_addr(s)), "l"(g), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp_async8(void* s, const unsigned char* g, bool pred) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n"
                 :: "r"(smem_addr(s)), "l"(g), "r"(pred ? 8 : 0));
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
    sf[row * kb + b] = s8;
}

// `w` / `wsf` / `alpha`: the weight's packed rows, swizzled scales and
// per-group alphas, each already offset to the first of this call's `d.n`
// rows (a composite may be multiplied over a 128-row-aligned sub-range).
// `ldb`: the row stride of the low-rank `lbt`. A non-null `resid` makes the
// store `rn(resid + a[seg][col] * rn(acc))` (the separate gate-add pass's
// two roundings).
extern "C" __global__ void __launch_bounds__(256) gemm_nvfp4(
    const unsigned char* __restrict__ qa, const unsigned char* __restrict__ sa,
    const unsigned char* __restrict__ w, const unsigned char* __restrict__ wsf,
    const float* __restrict__ alpha, unsigned short* __restrict__ out, Dims d,
    const unsigned short* __restrict__ lmid, const unsigned short* __restrict__ lbt, unsigned ldb,
    unsigned lrank, const float* __restrict__ insf, const unsigned short* __restrict__ resid,
    const Seg seg)
{
    // Per stage: A / B code rows (padded), A's two k64 scale words per row,
    // B's two 512-byte swizzled chunks per 128-column group (consecutive in
    // `layout_128x4`).
    extern __shared__ __align__(16) unsigned char smem[];
    auto as = [&](int st) { return smem + st * STAGE_BYTES; };
    auto bs = [&](int st) { return smem + st * STAGE_BYTES + BM * LDS; };
    auto ssa = [&](int st) { return reinterpret_cast<unsigned*>(smem + st * STAGE_BYTES + (BM + BN) * LDS); };
    auto ssb = [&](int st) { return smem + st * STAGE_BYTES + (BM + BN) * LDS + BM * 8; };

    const unsigned rowb = d.k / 2, kb = d.k / 16;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int wm0 = (warp >> 2) * 64, wn0 = (warp & 3) * 64;
    // Grouped rasterization over a 1-D grid.
    const unsigned mt = (d.m + BM - 1) / BM, nt = (d.n + BN - 1) / BN;
    const unsigned per_group = GROUP_M * nt, pid = blockIdx.x;
    const unsigned first_m = (pid / per_group) * GROUP_M;
    const unsigned gsize = min(mt - first_m, (unsigned)GROUP_M);
    const unsigned m0 = (first_m + (pid % per_group) % gsize) * BM;
    const unsigned n0 = ((pid % per_group) / gsize) * BN;
    const unsigned ntiles = d.k / 128;
    // The tile's 128-column scale groups (the second absent when N ends at
    // its first).
    const unsigned groups = min((d.n - n0) / 128, BN / 128u);
    // A warp's 64 columns lie in one group; past N it only idles.
    const bool warp_live = n0 + wn0 < d.n;

    // Strip kt -> stage st (rows past M/N zero-fill; N is a 128 multiple).
    auto load = [&](unsigned kt, int st) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int i = tid + j * 256;
            const int r = i >> 2, c = (i & 3) * 16;
            const unsigned gr = m0 + r;
            cp_async16(as(st) + r * LDS + c, qa + (size_t)min(gr, d.m - 1) * rowb + kt * ROWB + c, gr < d.m);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = tid + j * 256;
            const int r = i >> 2, c = (i & 3) * 16;
            const unsigned gn = n0 + r;
            cp_async16(bs(st) + r * LDS + c, w + (size_t)min(gn, d.n - 1) * rowb + kt * ROWB + c, gn < d.n);
        }
        if (tid < BM) {
            const unsigned gr = m0 + tid;
            cp_async8(ssa(st) + tid * 2, sa + (size_t)min(gr, d.m - 1) * kb + kt * 8, gr < d.m);
        } else {
            const int x = tid - BM, grp = x >> 6, c = (x & 63) * 16;
            const unsigned char* src = wsf + (size_t)((n0 >> 7) + min((unsigned)grp, groups - 1)) * (kb >> 2) * 512;
            cp_async16(ssb(st) + grp * 1024 + c, src + (size_t)kt * 1024 + c, (unsigned)grp < groups);
        }
    };

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
        unsigned short* ms = reinterpret_cast<unsigned short*>(smem);
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
            if (n0 + c * 8 < d.n) v = *reinterpret_cast<const uint4*>(lbt + (size_t)r * ldb + n0 + c * 8);
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
            const float inv_s = insf[0] / alpha[(n0 + wn0) >> 7];
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                #pragma unroll
                for (int ni = 0; ni < 8; ++ni)
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) acc[mi][ni][e] *= inv_s;
        }
        __syncthreads(); // the stages are refilled by the pipeline next
    }

    // STAGES-deep cp.async pipeline: one commit group per strip (empty past
    // the end, so the wait count stays uniform).
    #pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        if ((unsigned)st < ntiles) load(st, st);
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (unsigned kt = 0; kt < ntiles; ++kt) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2));
        __syncthreads();
        // The stage refilled here was read in the previous iteration, which
        // every thread finished before the barrier above.
        if (kt + STAGES - 1 < ntiles) load(kt + STAGES - 1, (kt + STAGES - 1) % STAGES);
        asm volatile("cp.async.commit_group;\n" ::);
        if (!warp_live) continue;
        const int st = kt % STAGES;
        const unsigned char* a_s = as(st);
        const unsigned char* b_s = bs(st);
        const unsigned* sa_s = ssa(st);
        const unsigned char* sb_s = ssb(st) + ((wn0 >> 7) << 10);
        #pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            unsigned af[4][4], sfa[4];
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                ldsm_x4(af[mi], a_s + (wm0 + mi * 16 + (lane & 15)) * LDS + kk * 32 + (lane >> 4) * 16);
                sfa[mi] = sa_s[(wm0 + mi * 16 + g + (lane & 1) * 8) * 2 + kk];
            }
            unsigned bf[8][2], sfb[8];
            #pragma unroll
            for (int nj = 0; nj < 4; ++nj) {
                unsigned r[4];
                ldsm_x4(r, b_s + (wn0 + nj * 16 + (lane & 7) + ((lane >> 4) << 3)) * LDS
                                + kk * 32 + ((lane >> 3) & 1) * 16);
                bf[2 * nj][0] = r[0]; bf[2 * nj][1] = r[1];
                bf[2 * nj + 1][0] = r[2]; bf[2 * nj + 1][1] = r[3];
            }
            #pragma unroll
            for (int ni = 0; ni < 8; ++ni) {
                const unsigned n = (wn0 + ni * 8 + g) & 127;
                sfb[ni] = *reinterpret_cast<const unsigned*>(sb_s + kk * 512 + (n & 31) * 16 + (n >> 5) * 4);
            }
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                #pragma unroll
                for (int ni = 0; ni < 8; ++ni)
                    mma_fp4(acc[mi][ni], af[mi], bf[ni][0], bf[ni][1], sfa[mi], sfb[ni]);
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    if (!warp_live) return;

    const float inv = 1.f / insf[0];

    #pragma unroll
    for (int ni = 0; ni < 8; ++ni) {
        const unsigned col = n0 + wn0 + ni * 8 + 2 * t;
        const float s = alpha[col >> 7] * inv;
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const unsigned row = m0 + wm0 + mi * 16 + g + h * 8;
                if (row >= d.m) continue;
                __nv_bfloat162 v = __floats2bfloat162_rn(acc[mi][ni][2 * h] * s, acc[mi][ni][2 * h + 1] * s);
                const size_t o = (size_t)row * d.n + col;
                if (resid) {
                    unsigned si = 0;
                    while (si + 1 < seg.n && row >= seg.end[si]) ++si;
                    const unsigned gw = *reinterpret_cast<const unsigned*>(
                        reinterpret_cast<const unsigned short*>(seg.a[si]) + col);
                    const unsigned rw = *reinterpret_cast<const unsigned*>(resid + o);
                    const unsigned pw = *reinterpret_cast<const unsigned*>(&v);
                    v = __floats2bfloat162_rn(
                        bf(rw & 0xffffu) + bf(gw & 0xffffu) * bf(pw & 0xffffu),
                        bf(rw >> 16) + bf(gw >> 16) * bf(pw >> 16));
                }
                *reinterpret_cast<__nv_bfloat162*>(out + o) = v;
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
    pad: u32,
}

/// Scratch bytes [`GemmFp4Kernels::run`] needs for an `[m, k]` activation:
/// codes, block scales, then the dynamic-scale slot (`amax` bits, scale).
pub fn scratch_bytes(m: u32, k: u32) -> u64 {
    slot_offset(m, k) + 16
}

fn slot_offset(m: u32, k: u32) -> u64 {
    (m as u64 * k as u64 / 2 + m as u64 * k as u64 / 16).next_multiple_of(16)
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
            gate_add,
        } = *g;
        let Nvfp4Rows { base, total } = rows.unwrap_or(Nvfp4Rows { base: 0, total: n });
        let aligned = [a, w, scratch].iter().all(|p| p.is_multiple_of(16)) && out.is_multiple_of(4);
        let lora_ok = lora.rank == 0
            || (nvfp4_fuses_lora_rank(lora.rank)
                && [lora.mid, lora.bt].iter().all(|p| p.is_multiple_of(16)));
        let range_ok = base.is_multiple_of(128) && base + n <= total;
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
        if !aligned || !lora_ok || !range_ok || !gate_ok || !accepts(m, n, k) {
            return Ok(false);
        }
        // Section pointers of the composite's full layout, at row `base`.
        let layout = Nvfp4Layout::new(total as usize, k as usize);
        let group = base as u64 / 128;
        let mut wq = w + base as u64 * k as u64 / 2;
        let mut wsf = w + layout.scales_offset() as u64 + group * (k as u64 / 64) * 512;
        let mut alpha = w + layout.alpha_offset() as u64 + group * 4;
        let (mut a, mut out) = (a, out);
        let mut qa = scratch;
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
        let mut dims = Dims { m, n, k, pad: 0 };
        // `bt` is `[rank, total]` bf16: start at column `base`, stride `total`.
        let (mut lmid, mut lbt, mut ldb, mut lrank) =
            (lora.mid, lora.bt + 2 * base as u64, total, lora.rank);
        let (mut resid, mut seg) = match gate_add {
            Some(g) => (g.residual, Seg::from(&g.seg)),
            None => (0, Seg::from(&Default::default())),
        };
        let mut args: [*mut c_void; 14] = [
            (&raw mut qa).cast(),
            (&raw mut sa).cast(),
            (&raw mut wq).cast(),
            (&raw mut wsf).cast(),
            (&raw mut alpha).cast(),
            (&raw mut out).cast(),
            (&raw mut dims).cast(),
            (&raw mut lmid).cast(),
            (&raw mut lbt).cast(),
            (&raw mut ldb).cast(),
            (&raw mut lrank).cast(),
            (&raw mut gsf).cast(),
            (&raw mut resid).cast(),
            (&raw mut seg).cast(),
        ];
        self.module.launch(
            self.gemm,
            (n.div_ceil(BN) * m.div_ceil(BM), 1, 1),
            THREADS,
            SMEM,
            &mut args,
            "gemm_nvfp4",
        )?;
        Ok(true)
    }
}
