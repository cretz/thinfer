//! Fused f16 attention on tensor cores, with exactly the wgpu `sdpa_sg`
//! contract (see `thinfer_core::ops::sdpa`): BSHD f16 Q `[B,Sq,Hq,D]`, K/V
//! `[B,Sk,Hkv,D]` (GQA `hkv = hq*Hkv/Hq`), mask modes 0 (none) / 1 (shared
//! `[B,Sq,Sk]`) / 2 (per-head `[B,Hq,Sq,Sk]`) as f16 additive bias, scores
//! `dot*scale + bias`, keys past `Sk` at the same finite `NEG_MAX`, f32 online
//! softmax, output `o / l` (0 when `l == 0`).
//!
//! FlashAttention-2 forward on `mma.sync.m16n8k16` (f16 in, f32 accumulate)
//! with `ldmatrix` operand loads. A block owns BR=128 query rows (8 warps x 16)
//! and streams BC=64 key tiles through padded shared memory (row stride D+8
//! halves keeps `ldmatrix` bank-conflict free). Scores, softmax state, and the
//! running output never leave registers; softmax probabilities are repacked
//! from the score accumulators straight into the P*V A-fragments. The one
//! numeric difference from `sdpa_sg`: probabilities are rounded to f16 for the
//! P*V product (standard FlashAttention practice).
//!
//! Compiled once per process by NVRTC straight to a cubin for the device's
//! compute capability (no driver PTX JIT, so a newer toolkit than the driver
//! still loads under CUDA minor-version compatibility).

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaContext, sys};

use crate::nvrtc::Module;

/// Head dims with a compiled kernel. Anything else declines to wgpu.
pub const SUPPORTED_D: [u32; 2] = [64, 128];
const BR: u32 = 128;
const BC: u32 = 64;
const THREADS: u32 = 256;

const SOURCE: &str = r#"
#include <cuda_fp16.h>

struct SdpaU { unsigned b, h_q, h_kv, s_q, s_k, d; float scale; unsigned has_mask; };

#define BR 128
#define BC 64
#define NEG_MAX (-3.402823e38f)

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], const half* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned (&r)[4], const half* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}
// d += a (16x16 row) * b (16x8 col).
__device__ __forceinline__ void mma16816(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<unsigned*>(&h);
}

// Fragment conventions (g = lane/4, t = lane%4): a C/D tile holds rows g
// (c0,c1) and g+8 (c2,c3) at columns 2t, 2t+1.
template <int D>
__device__ __forceinline__ void sdpa_impl(
    const half* __restrict__ q, const half* __restrict__ k, const half* __restrict__ v,
    const half* __restrict__ mask, half* __restrict__ out, const SdpaU* __restrict__ up)
{
    constexpr int LD = D + 8;
    const SdpaU u = *up;
    extern __shared__ __align__(16) unsigned char smem[];
    half* qs = reinterpret_cast<half*>(smem); // [BR][LD]
    half* ks = qs + BR * LD;                   // [BC][LD]
    half* vs = ks + BC * LD;                   // [BC][LD]

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const unsigned hq = blockIdx.y, bb = blockIdx.z;
    const unsigned hkv = (hq * u.h_kv) / u.h_q;
    const unsigned row0 = blockIdx.x * BR;

    for (int i = tid; i < BR * (D / 8); i += blockDim.x) {
        const int r = i / (D / 8), c8 = i % (D / 8);
        const unsigned sq = row0 + r;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (sq < u.s_q)
            val = *reinterpret_cast<const uint4*>(q + ((size_t)(bb * u.s_q + sq) * u.h_q + hq) * D + c8 * 8);
        *reinterpret_cast<uint4*>(qs + r * LD + c8 * 8) = val;
    }
    __syncthreads();

    unsigned qa[D / 16][4];
    {
        const half* base = qs + (warp * 16 + (lane & 15)) * LD + (lane >> 4) * 8;
        #pragma unroll
        for (int kk = 0; kk < D / 16; ++kk) ldsm_x4(qa[kk], base + kk * 16);
    }

    float o[D / 8][4];
    #pragma unroll
    for (int j = 0; j < D / 8; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m_r[2] = {NEG_MAX, NEG_MAX}, l_r[2] = {0.f, 0.f};
    const unsigned r_lo = row0 + warp * 16 + g, r_hi = r_lo + 8;
    size_t mrow[2];
    {
        const unsigned rc[2] = {min(r_lo, u.s_q - 1u), min(r_hi, u.s_q - 1u)};
        #pragma unroll
        for (int e = 0; e < 2; ++e)
            mrow[e] = (u.has_mask == 2u) ? ((size_t)(bb * u.h_q + hq) * u.s_q + rc[e])
                                         : ((size_t)bb * u.s_q + rc[e]);
    }

    for (unsigned kt = 0; kt < u.s_k; kt += BC) {
        __syncthreads(); // previous K/V tile fully consumed
        for (int i = tid; i < BC * (D / 8); i += blockDim.x) {
            const int r = i / (D / 8), c8 = i % (D / 8);
            const unsigned key = kt + r;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = kv;
            if (key < u.s_k) {
                const size_t off = ((size_t)(bb * u.s_k + key) * u.h_kv + hkv) * D + c8 * 8;
                kv = *reinterpret_cast<const uint4*>(k + off);
                vv = *reinterpret_cast<const uint4*>(v + off);
            }
            *reinterpret_cast<uint4*>(ks + r * LD + c8 * 8) = kv;
            *reinterpret_cast<uint4*>(vs + r * LD + c8 * 8) = vv;
        }
        __syncthreads();

        // S = Q K^T: 16 rows x BC keys per warp, as BC/8 C tiles.
        float s[BC / 8][4];
        #pragma unroll
        for (int j = 0; j < BC / 8; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
        #pragma unroll
        for (int kk = 0; kk < D / 16; ++kk) {
            #pragma unroll
            for (int nj = 0; nj < BC / 16; ++nj) {
                unsigned b[4]; // keys 0-7: (d lo, d hi), keys 8-15: (d lo, d hi)
                ldsm_x4(b, ks + (nj * 16 + (lane & 7) + ((lane >> 4) << 3)) * LD + kk * 16 + ((lane >> 3) & 1) * 8);
                mma16816(s[2 * nj], qa[kk], b[0], b[1]);
                mma16816(s[2 * nj + 1], qa[kk], b[2], b[3]);
            }
        }

        // Online softmax over this tile (two rows per thread, a quad per row).
        float mt[2] = {m_r[0], m_r[1]};
        #pragma unroll
        for (int j = 0; j < BC / 8; ++j) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = e >> 1;
                const unsigned key = kt + j * 8 + 2 * t + (e & 1);
                float x = NEG_MAX;
                if (key < u.s_k) {
                    const float bias = u.has_mask ? __half2float(mask[mrow[row] * u.s_k + key]) : 0.f;
                    x = s[j][e] * u.scale + bias;
                }
                s[j][e] = x;
                mt[row] = fmaxf(mt[row], x);
            }
        }
        float alpha[2];
        #pragma unroll
        for (int row = 0; row < 2; ++row) {
            mt[row] = fmaxf(mt[row], __shfl_xor_sync(0xffffffffu, mt[row], 1));
            mt[row] = fmaxf(mt[row], __shfl_xor_sync(0xffffffffu, mt[row], 2));
            alpha[row] = expf(m_r[row] - mt[row]);
            m_r[row] = mt[row];
            l_r[row] *= alpha[row];
        }
        #pragma unroll
        for (int j = 0; j < BC / 8; ++j) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float p = expf(s[j][e] - m_r[e >> 1]);
                s[j][e] = p;
                l_r[e >> 1] += p;
            }
        }
        // P as A fragments: k-block kb spans C tiles 2kb (keys 0-7) and 2kb+1.
        unsigned pa[BC / 16][4];
        #pragma unroll
        for (int kb = 0; kb < BC / 16; ++kb) {
            pa[kb][0] = pack_h2(s[2 * kb][0], s[2 * kb][1]);
            pa[kb][1] = pack_h2(s[2 * kb][2], s[2 * kb][3]);
            pa[kb][2] = pack_h2(s[2 * kb + 1][0], s[2 * kb + 1][1]);
            pa[kb][3] = pack_h2(s[2 * kb + 1][2], s[2 * kb + 1][3]);
        }
        #pragma unroll
        for (int j = 0; j < D / 8; ++j) {
            o[j][0] *= alpha[0]; o[j][1] *= alpha[0];
            o[j][2] *= alpha[1]; o[j][3] *= alpha[1];
        }
        // O += P V (V row-major [key][d] -> B via ldmatrix.trans).
        #pragma unroll
        for (int kb = 0; kb < BC / 16; ++kb) {
            #pragma unroll
            for (int dn = 0; dn < D / 16; ++dn) {
                unsigned b[4]; // d lo: (keys lo, keys hi), d hi: (keys lo, keys hi)
                ldsm_x4_t(b, vs + (kb * 16 + (lane & 7) + (((lane >> 3) & 1) << 3)) * LD + dn * 16 + (lane >> 4) * 8);
                mma16816(o[2 * dn], pa[kb], b[0], b[1]);
                mma16816(o[2 * dn + 1], pa[kb], b[2], b[3]);
            }
        }
    }

    float inv[2];
    #pragma unroll
    for (int row = 0; row < 2; ++row) {
        float l = l_r[row];
        l += __shfl_xor_sync(0xffffffffu, l, 1);
        l += __shfl_xor_sync(0xffffffffu, l, 2);
        inv[row] = (l > 0.f) ? 1.f / l : 0.f;
    }
    #pragma unroll
    for (int j = 0; j < D / 8; ++j) {
        const int col = j * 8 + 2 * t;
        if (r_lo < u.s_q)
            *reinterpret_cast<__half2*>(out + ((size_t)(bb * u.s_q + r_lo) * u.h_q + hq) * D + col) =
                __floats2half2_rn(o[j][0] * inv[0], o[j][1] * inv[0]);
        if (r_hi < u.s_q)
            *reinterpret_cast<__half2*>(out + ((size_t)(bb * u.s_q + r_hi) * u.h_q + hq) * D + col) =
                __floats2half2_rn(o[j][2] * inv[1], o[j][3] * inv[1]);
    }
}

extern "C" __global__ void __launch_bounds__(256) sdpa_f16_d64(
    const half* q, const half* k, const half* v, const half* mask, half* out, const SdpaU* u)
{ sdpa_impl<64>(q, k, v, mask, out, u); }

extern "C" __global__ void __launch_bounds__(256) sdpa_f16_d128(
    const half* q, const half* k, const half* v, const half* mask, half* out, const SdpaU* u)
{ sdpa_impl<128>(q, k, v, mask, out, u); }
"#;

/// Dynamic shared memory for head dim `d` (must match the kernel's carve-up:
/// Q, K, V tiles at padded row stride `d + 8` halves).
fn smem_bytes(d: u32) -> u32 {
    (BR + 2 * BC) * (d + 8) * 2
}

/// The `sdpa_sg` uniform (`struct U`, 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SdpaUniform {
    pub b: u32,
    pub h_q: u32,
    pub h_kv: u32,
    pub s_q: u32,
    pub s_k: u32,
    pub d: u32,
    pub scale: f32,
    pub has_mask: u32,
}

pub const UNIFORM_LEN: u64 = std::mem::size_of::<SdpaUniform>() as u64;

/// The compiled kernels for one device.
pub struct SdpaKernels {
    module: Module,
    d64: sys::CUfunction,
    d128: sys::CUfunction,
}

// SAFETY: function handles are plain driver handles; every use binds the
// owning context first.
unsafe impl Send for SdpaKernels {}
unsafe impl Sync for SdpaKernels {}

impl SdpaKernels {
    /// Compile and load the kernels on `ctx`'s device (compute capability `sm`).
    pub fn new(ctx: Arc<CudaContext>, sm: (i32, i32)) -> Result<Self, String> {
        let module = Module::compile(ctx, SOURCE, "thinfer_sdpa.cu", sm)?;
        let d64 = module.function(c"sdpa_f16_d64", smem_bytes(64))?;
        let d128 = module.function(c"sdpa_f16_d128", smem_bytes(128))?;
        Ok(Self { module, d64, d128 })
    }

    /// Run on device addresses with shapes `u`. `Ok(false)` = unsupported
    /// shape (caller runs wgpu). Synchronous: output complete on return.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        u: &SdpaUniform,
        q: u64,
        k: u64,
        v: u64,
        mask: u64,
        out: u64,
        uniform: u64,
    ) -> Result<bool, String> {
        let f = match u.d {
            64 => self.d64,
            128 => self.d128,
            _ => return Ok(false),
        };
        // 16-byte vector loads; GQA needs Hq divisible by Hkv like sdpa_sg.
        let aligned = [q, k, v, out].iter().all(|a| a % 16 == 0);
        if !aligned || u.h_kv == 0 || !u.h_q.is_multiple_of(u.h_kv) || u.s_q == 0 || u.s_k == 0 {
            return Ok(false);
        }
        self.module.bind()?;
        let (mut q, mut k, mut v, mut mask, mut out, mut uniform) = (q, k, v, mask, out, uniform);
        let mut args: [*mut c_void; 6] = [
            (&raw mut q).cast(),
            (&raw mut k).cast(),
            (&raw mut v).cast(),
            (&raw mut mask).cast(),
            (&raw mut out).cast(),
            (&raw mut uniform).cast(),
        ];
        self.module.launch_sync(
            f,
            (u.s_q.div_ceil(BR), u.h_q, u.b),
            THREADS,
            smem_bytes(u.d),
            &mut args,
            "sdpa",
        )?;
        Ok(true)
    }

    /// Read the shapes from the uniform buffer in VRAM.
    pub fn read_uniform(&self, uniform: u64) -> Result<SdpaUniform, String> {
        self.module.read(uniform)
    }
}
