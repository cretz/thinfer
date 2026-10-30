//! Video sparse attention (VSA) for MiniMax-H3 / FastH3's packed joint
//! sequence, the attention those checkpoints were distilled with (FastVideo
//! `video_sparse_attn_h3`): rows are grouped into 64-row tiles (text and audio
//! chunked per segment, video as 4x4x4 latent cubes), and
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
use std::sync::Arc;

use cudarc::driver::{CudaContext, sys};
use thinfer_core::backend::{TileAttnAddrs, TileAttnShape, VsaAddrs, VsaShape};

use crate::nvrtc::Module;

/// Rows per tile (the FastH3 contract's `vsa_tile_size`).
pub const TILE: u32 = 64;
const D: u32 = 128;
const THREADS: u32 = 128;

const SOURCE: &str = r#"
#include <cuda_bf16.h>

#define TILE 64
#define D 128
#define LD (D + 8)
#define NEG_MAX (-3.402823e38f)
#define NEG_INF __int_as_float(0xff800000)

struct VsaP {
    unsigned rows, heads, n_tiles, n_prefix, topk, has_gate;
    float scale;
    unsigned pad;
};

__device__ __forceinline__ float bf(unsigned short v) { return __uint_as_float((unsigned)v << 16); }

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}
// Tile means of q, k, v: grid (T, H), one thread per channel.
extern "C" __global__ void __launch_bounds__(128) vsa_pool(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v, const unsigned* __restrict__ tile_rows,
    const unsigned* __restrict__ tile_size, float* __restrict__ pooled, const VsaP p)
{
    const unsigned t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const unsigned n = tile_size[t];
    float sq = 0.f, sk = 0.f, sv = 0.f;
    for (unsigned r = 0; r < n; ++r) {
        const size_t off = ((size_t)tile_rows[t * TILE + r] * p.heads + h) * D + d;
        sq += bf(q[off]); sk += bf(k[off]); sv += bf(v[off]);
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
    __shared__ float qi[D];
    __shared__ float red[4];
    const unsigned i = blockIdx.x, h = blockIdx.y, T = p.n_tiles;
    const size_t plane = (size_t)p.heads * T * D;
    const float* qp = pooled + (size_t)h * T * D;
    const float* kp = pooled + plane + (size_t)h * T * D;
    const float* vp = pooled + 2 * plane + (size_t)h * T * D;
    qi[threadIdx.x] = qp[(size_t)i * D + threadIdx.x];
    __syncthreads();
    float mx = NEG_MAX;
    for (unsigned j = threadIdx.x; j < T; j += blockDim.x) {
        float acc = 0.f;
        for (int c = 0; c < D; ++c) acc += qi[c] * kp[(size_t)j * D + c];
        acc *= p.scale;
        s[j] = acc;
        scores[((size_t)h * T + i) * T + j] = acc;
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

// FlashAttention-2 forward over each query tile's key-tile list: grid (T, H),
// 4 warps x 16 query rows, bf16 MMA (bf16 probabilities, as FlashAttention
// runs bf16 inputs). Keys stream in half-tile (BC = 32) stages, double-
// buffered with cp.async straight from the gathered rows, so the next stage
// loads while this one multiplies; the Q tile is staged through stage 0 and
// lives in registers. Keys past a tile's size are zero-filled and masked.
// Epilogue adds gate * coarse[tile].
#define BC 32
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
    constexpr int LD = D + 8;
    extern __shared__ __align__(16) unsigned char smem[];
    // stage s: K [BC][LD] then V [BC][LD]; the Q tile [TILE][LD] aliases stage 0.
    unsigned short* stage = reinterpret_cast<unsigned short*>(smem);
    constexpr int STAGE = 2 * BC * LD;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const unsigned i = blockIdx.x, h = blockIdx.y, T = p.n_tiles;
    const unsigned nq = tile_size[i];
    const size_t rs = (size_t)p.heads * D;
    const unsigned* list = lists + ((size_t)h * T + i) * T;
    const unsigned n_steps = 2 * counts[h * T + i];

    {
        unsigned short* qs = stage;
        for (int x = tid; x < TILE * (D / 8); x += blockDim.x) {
            const int r = x / (D / 8), c8 = x % (D / 8);
            const bool ok = (unsigned)r < nq;
            cp_async16(qs + r * LD + c8 * 8, q + tile_rows[i * TILE + (ok ? r : 0)] * rs + h * D + c8 * 8, ok);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }
    unsigned qa[D / 16][4];
    {
        const unsigned short* base = stage + (warp * 16 + (lane & 15)) * LD + (lane >> 4) * 8;
        #pragma unroll
        for (int kk = 0; kk < D / 16; ++kk) ldsm_x4(qa[kk], base + kk * 16);
    }
    __syncthreads(); // stage 0 is free for keys

    // Half tile s -> stage buffer.
    auto issue = [&](unsigned s, int buf) {
        const unsigned kt = list[s >> 1], k0 = (s & 1) * BC, nk = tile_size[kt];
        unsigned short* ks = stage + buf * STAGE;
        unsigned short* vs = ks + BC * LD;
        for (int x = tid; x < BC * (D / 8); x += blockDim.x) {
            const int r = x / (D / 8), c8 = x % (D / 8);
            const unsigned key = k0 + r;
            const bool ok = key < nk;
            const size_t off = tile_rows[kt * TILE + (ok ? key : 0)] * rs + h * D + c8 * 8;
            cp_async16(ks + r * LD + c8 * 8, k + off, ok);
            cp_async16(vs + r * LD + c8 * 8, v + off, ok);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    float o[D / 8][4];
    #pragma unroll
    for (int j = 0; j < D / 8; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m_r[2] = {NEG_MAX, NEG_MAX}, l_r[2] = {0.f, 0.f};

    issue(0, 0);
    for (unsigned s = 0; s < n_steps; ++s) {
        const int buf = s & 1;
        if (s + 1 < n_steps) {
            issue(s + 1, buf ^ 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        const unsigned short* ks = stage + buf * STAGE;
        const unsigned short* vs = ks + BC * LD;
        const unsigned k0 = (s & 1) * BC, nk = tile_size[list[s >> 1]];

        float sc[BC / 8][4];
        #pragma unroll
        for (int j = 0; j < BC / 8; ++j) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
        #pragma unroll
        for (int kk = 0; kk < D / 16; ++kk) {
            #pragma unroll
            for (int nj = 0; nj < BC / 16; ++nj) {
                unsigned b[4];
                ldsm_x4(b, ks + (nj * 16 + (lane & 7) + ((lane >> 4) << 3)) * LD + kk * 16 + ((lane >> 3) & 1) * 8);
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
                const float x = key < nk ? sc[j][e] * p.scale : NEG_MAX;
                sc[j][e] = x;
                mt[e >> 1] = fmaxf(mt[e >> 1], x);
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
                const float pr = expf(sc[j][e] - m_r[e >> 1]);
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
        for (int j = 0; j < D / 8; ++j) {
            o[j][0] *= alpha[0]; o[j][1] *= alpha[0];
            o[j][2] *= alpha[1]; o[j][3] *= alpha[1];
        }
        #pragma unroll
        for (int kb = 0; kb < BC / 16; ++kb) {
            #pragma unroll
            for (int dn = 0; dn < D / 16; ++dn) {
                unsigned b[4];
                ldsm_x4_t(b, vs + (kb * 16 + (lane & 7) + (((lane >> 3) & 1) << 3)) * LD + dn * 16 + (lane >> 4) * 8);
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
        const size_t base = tile_rows[i * TILE + r] * rs + h * D;
        #pragma unroll
        for (int j = 0; j < D / 8; ++j) {
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
extern "C" __global__ void __launch_bounds__(128) vsa_attn(ATTN_ARGS)
{ attn_impl<128>(q, k, v, gate, out, tile_rows, tile_size, lists, counts, coarse, p); }

extern "C" __global__ void __launch_bounds__(128) vsa_attn_d64(ATTN_ARGS)
{ attn_impl<64>(q, k, v, gate, out, tile_rows, tile_size, lists, counts, coarse, p); }
"#;

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
    pad: u32,
}

/// Scratch bytes for `shape`: pooled q/k/v + compression output (f32
/// `[H, T, D]` each), scores (f32 `[H, T, T]`), lists (u32 `[H, T, T]`) and
/// counts (u32 `[H, T]`).
pub fn scratch_bytes(s: &VsaShape) -> u64 {
    let (h, t, d) = (s.heads as u64, s.n_tiles as u64, s.head_dim as u64);
    4 * (4 * h * t * d + 2 * h * t * t + h * t)
}

pub fn accepts(s: &VsaShape) -> bool {
    s.head_dim == D && s.n_tiles > s.n_prefix && s.rows > 0 && s.heads > 0
}

/// Two double-buffered half-tile K/V stages (the Q tile aliases stage 0).
fn attn_smem(d: u32) -> u32 {
    2 * 2 * (TILE / 2) * (d + 8) * 2
}

pub struct VsaKernels {
    module: Module,
    pool: sys::CUfunction,
    scores: sys::CUfunction,
    lists: sys::CUfunction,
    attn: sys::CUfunction,
    attn_d64: sys::CUfunction,
}

// SAFETY: function handles are plain driver handles; every use binds the
// owning context first.
unsafe impl Send for VsaKernels {}
unsafe impl Sync for VsaKernels {}

impl VsaKernels {
    pub fn new(ctx: Arc<CudaContext>, sm: (i32, i32)) -> Result<Self, String> {
        let module = Module::compile(ctx, SOURCE, "thinfer_vsa.cu", sm)?;
        let pool = module.function(c"vsa_pool", 0)?;
        // Dynamic smem sized per call (T floats); the attribute caps it.
        let scores = module.function(c"vsa_scores", 48 * 1024)?;
        let lists = module.function(c"vsa_lists", 48 * 1024)?;
        let attn = module.function(c"vsa_attn", attn_smem(D))?;
        let attn_d64 = module.function(c"vsa_attn_d64", attn_smem(64))?;
        Ok(Self {
            module,
            pool,
            scores,
            lists,
            attn,
            attn_d64,
        })
    }

    /// Enqueued only (see `Accelerator::sync`). `Ok(false)` = shape or
    /// alignment not covered.
    pub fn run(&self, s: &VsaShape, b: &VsaAddrs) -> Result<bool, String> {
        let aligned = [b.q, b.k, b.v, b.out, b.geometry, b.scratch]
            .iter()
            .all(|a| a.is_multiple_of(16))
            && b.gate.is_multiple_of(16);
        // Per-row smem score buffers: T floats must fit the 48 KiB cap.
        if !aligned || !accepts(s) || s.n_tiles as u64 * 4 > 48 * 1024 {
            return Ok(false);
        }
        let (h, t, d) = (s.heads as u64, s.n_tiles as u64, s.head_dim as u64);
        let mut pooled = b.scratch;
        let mut coarse = pooled + 4 * 3 * h * t * d;
        let mut scores = coarse + 4 * h * t * d;
        let mut lists = scores + 4 * h * t * t;
        let mut counts = lists + 4 * h * t * t;
        let mut tile_rows = b.geometry;
        let mut tile_size = b.geometry + 4 * t * TILE as u64;
        let mut p = VsaP {
            rows: s.rows,
            heads: s.heads,
            n_tiles: s.n_tiles,
            n_prefix: s.n_prefix,
            topk: s.topk,
            has_gate: (b.gate != 0) as u32,
            scale: s.scale,
            pad: 0,
        };
        let (mut q, mut k, mut v, mut gate, mut out) = (b.q, b.k, b.v, b.gate, b.out);
        let grid = (s.n_tiles, s.heads, 1);
        let row_smem = s.n_tiles * 4;
        self.module.bind()?;
        let mut a: [*mut c_void; 7] = [
            (&raw mut q).cast(),
            (&raw mut k).cast(),
            (&raw mut v).cast(),
            (&raw mut tile_rows).cast(),
            (&raw mut tile_size).cast(),
            (&raw mut pooled).cast(),
            (&raw mut p).cast(),
        ];
        self.module
            .launch(self.pool, grid, THREADS, 0, &mut a, "vsa_pool")?;
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
        self.module
            .launch(self.attn, grid, THREADS, attn_smem(D), &mut a, "vsa_attn")?;
        Ok(true)
    }

    /// The attention pass alone over caller-built key-tile lists (see
    /// [`TileAttnAddrs`]), head dim 64 or 128, no gate. Enqueued only.
    pub fn run_lists(&self, s: &TileAttnShape, b: &TileAttnAddrs) -> Result<bool, String> {
        let f = match s.head_dim {
            64 => self.attn_d64,
            128 => self.attn,
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
        let mut tile_size = b.geometry + 4 * t * TILE as u64;
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
        self.module.bind()?;
        self.module.launch(
            f,
            (s.n_tiles, s.heads, 1),
            THREADS,
            attn_smem(s.head_dim),
            &mut a,
            "tile_attn",
        )?;
        Ok(true)
    }
}
