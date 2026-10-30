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

template <bool C_BF16, bool B_BF16, bool NMAJ, bool OUT_F32>
__device__ __forceinline__ void gemm_impl(
    const unsigned short* __restrict__ a, const unsigned short* __restrict__ b,
    void* __restrict__ out, const Dims d)
{
    constexpr int A_TILE = BM * LDA;
    constexpr int B_TILE = NMAJ ? BN * LDBN : BK * LDBK;
    __shared__ __align__(16) unsigned short as[2][A_TILE];
    __shared__ __align__(16) unsigned short bs[2][B_TILE];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int wm0 = (warp >> 2) * 64, wn0 = (warp & 3) * 32;
    const unsigned m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
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
                const float x = acc[mi][ni][2 * h], y = acc[mi][ni][2 * h + 1];
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
                    __half2 v = __floats2half2_rn(x, y);
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
    const unsigned short* a, const unsigned short* b, void* out, const Dims d) \
{ gemm_impl<CB, BB, NM, O32>(a, b, out, d); }

GEMM(gemm_a16_b16_k, false, false, false, false)
GEMM(gemm_a16_b16_n, false, false, true, false)
GEMM(gemm_a16_bb_k, false, true, false, false)
GEMM(gemm_a16_bb_n, false, true, true, false)
GEMM(gemm_ab_bb_k, true, true, false, false)
GEMM(gemm_ab_bb_n, true, true, true, false)
GEMM(gemm_a16_b16_k_o32, false, false, false, true)
GEMM(gemm_a16_b16_n_o32, false, false, true, true)
"#;

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
    overflow: sys::CUdeviceptr,
}

// SAFETY: function handles are plain driver handles; every use binds the
// owning context first.
unsafe impl Send for GemmKernels {}
unsafe impl Sync for GemmKernels {}

/// The kernel for `desc`; `None` for the declined bf16-A x f16-B combo and
/// f32 output on anything but f16 x f16 (no caller has one).
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
        let overflow = module.global(c"g_overflow")?;
        Ok(Self {
            module,
            funcs,
            overflow,
        })
    }

    /// `out = a @ b` per `desc` on device addresses; `m`/`n` from the caller,
    /// `k` from the caller or (`k == 0`) read from `dims`. `Ok(false)` =
    /// unsupported shape or a bf16 weight past the f16 range (caller runs
    /// wgpu). Enqueued only, except the narrowing form, which must read its
    /// overflow flag and so waits.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        desc: &GemmDesc,
        a: u64,
        b: u64,
        out: u64,
        dims: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<bool, String> {
        let Some(f) = index(desc) else {
            return Ok(false);
        };
        let dm: GemmDims = if k == 0 {
            self.module.read(dims)?
        } else {
            GemmDims { m, n, k, pad: 0 }
        };
        // 16-byte vector loads along K (A, n-major B) or N (k-major B);
        // paired output stores.
        let aligned = [a, b].iter().all(|p| p.is_multiple_of(16)) && out.is_multiple_of(8);
        if !aligned
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
        let narrows = desc.a == GemmType::F16 && desc.b == GemmType::Bf16;
        if narrows {
            self.module.write(self.overflow, 0u32)?;
        }
        let (mut a, mut b, mut out, mut dims) = (a, b, out, dm);
        let mut args: [*mut c_void; 4] = [
            (&raw mut a).cast(),
            (&raw mut b).cast(),
            (&raw mut out).cast(),
            (&raw mut dims).cast(),
        ];
        self.module.bind()?;
        self.module.launch(
            self.funcs[f],
            (n.div_ceil(BN), m.div_ceil(BM), 1),
            THREADS,
            0,
            &mut args,
            "gemm",
        )?;
        if narrows && self.module.read::<u32>(self.overflow)? != 0 {
            return Ok(false);
        }
        Ok(true)
    }
}
