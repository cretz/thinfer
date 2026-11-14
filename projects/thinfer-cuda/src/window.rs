//! Window attention on tensor cores with exactly the `ops::window_attn`
//! contract (comfy-kitchen `window_attention`, the RelayVSR attention): f16 Q
//! `[B, Tq, H, W, nh, D]`, K/V `[B, Tk, H, W, kvh, D]` (GQA), query plane `t`
//! attends key planes `last - tw + 1 ..= last` (`last = t + Tk - Tq`), each
//! query the `(wh, ww)` window of its plane centered on it (shifted inside the
//! frame with `clamp`, cut at the border without).
//!
//! The `sdpa` FlashAttention-2 core (`mma.sync.m16n8k16`, `ldmatrix`, cp.async
//! double-buffered BC=64 key tiles, f32 online softmax, f16 P) over a GATHERED
//! key sequence: a block owns a `th x tw` (= 128) spatial tile of one query
//! plane; per attended plane its keys are the rectangle covering the union of
//! the tile's windows, and the planes' rectangles are flattened into one key
//! sequence. Each key's `(plane, y, x)` rides in shared memory next to the
//! tile, and scores outside a query's own window are masked to the finite
//! `NEG_MAX` the dense kernel uses past `Sk`.
//!
//! Compiled once per process by NVRTC to a cubin for the device.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use thinfer_core::ops::window_attn::{MAX_PLANES, WindowAttnUniform};

use crate::nvrtc::Module;

const BR: u32 = 128;
const BC: u32 = 64;
const THREADS: u32 = 256;
/// Planes one query plane may attend (the temporal window).
const MAX_TW: u32 = 4;

/// Kernel argument (by value). Mirrors `struct WinU` in [`SOURCE`].
#[repr(C)]
#[derive(Clone, Copy)]
struct WinU {
    tq: u32,
    tk: u32,
    h: u32,
    w: u32,
    nh: u32,
    kvh: u32,
    tw: u32,
    clamp: u32,
    scale: f32,
    tile_h: u32,
    tile_w: u32,
    tiles_x: u32,
    tiles_y: u32,
    _pad: u32,
    win: [[u32; 2]; MAX_PLANES],
}

const SOURCE: &str = r#"
#include <cuda_fp16.h>

#define MAX_PLANES 8
#define MAX_TW 4
struct WinU {
    unsigned tq, tk, h, w, nh, kvh, tw, clamp;
    float scale;
    unsigned tile_h, tile_w, tiles_x, tiles_y, _pad;
    unsigned win[MAX_PLANES][2];
};

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
__device__ __forceinline__ void mma16816(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool full) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem_addr(dst)), "l"(src), "r"(full ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<unsigned*>(&h);
}
// First attended index along an axis for a query at `pos`.
__device__ __forceinline__ int win_lo(int pos, int size, int win, bool clamp) {
    int lo = pos - win / 2;
    if (clamp) lo = min(max(lo, 0), size - win);
    return lo;
}

template <int D>
__device__ __forceinline__ void window_impl(
    const half* __restrict__ q, const half* __restrict__ k, const half* __restrict__ v,
    half* __restrict__ out, const WinU u)
{
    constexpr int LD = D + 8;
    extern __shared__ __align__(16) unsigned char smem[];
    half* qs = reinterpret_cast<half*>(smem); // [BR][LD], then K/V stage 1
    auto kstage = [&](int st) { return st ? qs : qs + BR * LD; };
    // Per stage, each key of the tile: (plane, y', x', wh << 16 | ww) with
    // y' = y + wh / 2 (x' likewise) without `clamp` (so a query at (qy, qx)
    // attends it iff `(unsigned)(y' - qy) < wh` and the same for x), the plain
    // (y, x) with it. No key: plane -1, wh = ww = 0.
    __shared__ int4 kcoord[2][BC];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const unsigned hq = blockIdx.y, bb = blockIdx.z;
    const unsigned hkv = (hq * u.kvh) / u.nh;
    const unsigned tiles = u.tiles_x * u.tiles_y;
    const unsigned tq_i = blockIdx.x / tiles;
    const unsigned tile = blockIdx.x % tiles;
    const int ty0 = (int)((tile / u.tiles_x) * u.tile_h);
    const int tx0 = (int)((tile % u.tiles_x) * u.tile_w);
    const int H = (int)u.h, W = (int)u.w;
    const bool clamp = u.clamp != 0u;

    // Query row r of the tile -> (y, x); token index or -1.
    auto q_tok = [&](int r, int& y, int& x) -> long long {
        y = ty0 + r / (int)u.tile_w;
        x = tx0 + r % (int)u.tile_w;
        if (y >= H || x >= W) return -1;
        return ((long long)(bb * u.tq + tq_i) * H + y) * W + x;
    };

    for (int i = tid; i < BR * (D / 8); i += blockDim.x) {
        const int r = i / (D / 8), c8 = i % (D / 8);
        int y, x;
        const long long tok = q_tok(r, y, x);
        uint4 val = make_uint4(0, 0, 0, 0);
        if (tok >= 0)
            val = *reinterpret_cast<const uint4*>(q + ((size_t)tok * u.nh + hq) * D + c8 * 8);
        *reinterpret_cast<uint4*>(qs + r * LD + c8 * 8) = val;
    }
    __syncthreads();

    unsigned qa[D / 16][4];
    {
        const half* base = qs + (warp * 16 + (lane & 15)) * LD + (lane >> 4) * 8;
        #pragma unroll
        for (int kk = 0; kk < D / 16; ++kk) ldsm_x4(qa[kk], base + kk * 16);
    }
    __syncthreads();

    // Attended planes and the key rectangle of each (union of the tile's
    // windows; lo is monotone in the query coordinate).
    const int last = (int)(tq_i + u.tk - u.tq);
    const int first = max(last - (int)u.tw + 1, 0);
    const int np = last - first + 1;
    int pr_y0[MAX_TW], pr_x0[MAX_TW], pr_cw[MAX_TW], pr_start[MAX_TW + 1];
    int pwh[MAX_TW], pww[MAX_TW];
    const int ty1 = min(ty0 + (int)u.tile_h, H) - 1, tx1 = min(tx0 + (int)u.tile_w, W) - 1;
    pr_start[0] = 0;
    for (int i = 0; i < np; ++i) {
        const int p = first + i;
        const int wi = min(p, MAX_PLANES - 1);
        int wh = (int)u.win[wi][0], ww = (int)u.win[wi][1];
        if (clamp) { wh = min(wh, H); ww = min(ww, W); }
        pwh[i] = wh; pww[i] = ww;
        const int y0 = max(win_lo(ty0, H, wh, clamp), 0);
        const int y1 = min(win_lo(ty1, H, wh, clamp) + wh, H);
        const int x0 = max(win_lo(tx0, W, ww, clamp), 0);
        const int x1 = min(win_lo(tx1, W, ww, clamp) + ww, W);
        pr_y0[i] = y0; pr_x0[i] = x0; pr_cw[i] = max(x1 - x0, 0);
        pr_start[i + 1] = pr_start[i] + max(y1 - y0, 0) * pr_cw[i];
    }
    const unsigned n_keys = (unsigned)pr_start[np];

    auto load_kv = [&](unsigned kt, int st) {
        half* ks = kstage(st);
        half* vs = ks + BC * LD;
        if (tid < BC) {
            const int j = (int)kt + tid;
            int4 c = make_int4(-1, 0, 0, 0);
            if (j < (int)n_keys) {
                int i = 0;
                while (j >= pr_start[i + 1]) ++i;
                const int off = j - pr_start[i];
                const int y = pr_y0[i] + off / pr_cw[i], x = pr_x0[i] + off % pr_cw[i];
                const int hy = clamp ? 0 : pwh[i] / 2, hx = clamp ? 0 : pww[i] / 2;
                c = make_int4(first + i, y + hy, x + hx, (pwh[i] << 16) | pww[i]);
            }
            kcoord[st][tid] = c;
        }
        __syncthreads();
        for (int i = tid; i < BC * (D / 8); i += blockDim.x) {
            const int r = i / (D / 8), c8 = i % (D / 8);
            const int4 c = kcoord[st][r];
            const bool ok = c.x >= 0;
            const int ky = clamp ? c.y : c.y - (c.w >> 16) / 2;
            const int kx = clamp ? c.z : c.z - (c.w & 0xffff) / 2;
            const size_t tok = ok ? ((size_t)(bb * u.tk + c.x) * H + ky) * W + kx : 0;
            const size_t off = (tok * u.kvh + hkv) * D + c8 * 8;
            cp_async16(ks + r * LD + c8 * 8, k + off, ok);
            cp_async16(vs + r * LD + c8 * 8, v + off, ok);
        }
        cp_async_commit();
    };

    float o[D / 8][4];
    #pragma unroll
    for (int j = 0; j < D / 8; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m_r[2] = {NEG_MAX, NEG_MAX}, l_r[2] = {0.f, 0.f};
    const int rq[2] = {warp * 16 + g, warp * 16 + g + 8};
    // A padding query row sits far outside every window (all keys masked).
    int qy[2], qx[2];
    for (int e = 0; e < 2; ++e)
        if (q_tok(rq[e], qy[e], qx[e]) < 0) qy[e] = qx[e] = -(1 << 28);
    // Softmax in base 2: scores pre-scaled by log2(e).
    const float scale2 = u.scale * 1.4426950408889634f;

    load_kv(0, 0);
    int st = 0;
    for (unsigned kt = 0; kt < n_keys; kt += BC, st ^= 1) {
        if (kt + BC < n_keys) {
            load_kv(kt + BC, st ^ 1);
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
        const half* ks = kstage(st);
        const half* vs = ks + BC * LD;

        float s[BC / 8][4];
        #pragma unroll
        for (int j = 0; j < BC / 8; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
        #pragma unroll
        for (int kk = 0; kk < D / 16; ++kk) {
            #pragma unroll
            for (int nj = 0; nj < BC / 16; ++nj) {
                unsigned b[4];
                ldsm_x4(b, ks + (nj * 16 + (lane & 7) + ((lane >> 4) << 3)) * LD + kk * 16 + ((lane >> 3) & 1) * 8);
                mma16816(s[2 * nj], qa[kk], b[0], b[1]);
                mma16816(s[2 * nj + 1], qa[kk], b[2], b[3]);
            }
        }

        float mt[2] = {m_r[0], m_r[1]};
        #pragma unroll
        for (int j = 0; j < BC / 8; ++j) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = e >> 1;
                const int4 c = kcoord[st][j * 8 + 2 * t + (e & 1)];
                const unsigned wh = (unsigned)c.w >> 16, ww = (unsigned)c.w & 0xffffu;
                // Window origin relative to the stored key coordinate: the
                // query itself without clamp (y' already holds the + wh / 2).
                int ly = qy[row], lx = qx[row];
                if (clamp) {
                    ly = win_lo(qy[row], H, (int)wh, true);
                    lx = win_lo(qx[row], W, (int)ww, true);
                }
                const unsigned dy = (unsigned)(c.y - ly), dx = (unsigned)(c.z - lx);
                const float x = (dy < wh && dx < ww) ? s[j][e] * scale2 : NEG_MAX;
                s[j][e] = x;
                mt[row] = fmaxf(mt[row], x);
            }
        }
        float alpha[2];
        #pragma unroll
        for (int row = 0; row < 2; ++row) {
            mt[row] = fmaxf(mt[row], __shfl_xor_sync(0xffffffffu, mt[row], 1));
            mt[row] = fmaxf(mt[row], __shfl_xor_sync(0xffffffffu, mt[row], 2));
            alpha[row] = exp2f(m_r[row] - mt[row]);
            m_r[row] = mt[row];
            l_r[row] *= alpha[row];
        }
        #pragma unroll
        for (int j = 0; j < BC / 8; ++j) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float p = exp2f(s[j][e] - m_r[e >> 1]);
                s[j][e] = p;
                l_r[e >> 1] += p;
            }
        }
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
        #pragma unroll
        for (int kb = 0; kb < BC / 16; ++kb) {
            #pragma unroll
            for (int dn = 0; dn < D / 16; ++dn) {
                unsigned b[4];
                ldsm_x4_t(b, vs + (kb * 16 + (lane & 7) + (((lane >> 3) & 1) << 3)) * LD + dn * 16 + (lane >> 4) * 8);
                mma16816(o[2 * dn], pa[kb], b[0], b[1]);
                mma16816(o[2 * dn + 1], pa[kb], b[2], b[3]);
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
    long long tok[2];
    for (int e = 0; e < 2; ++e) { int y, x; tok[e] = q_tok(rq[e], y, x); }
    #pragma unroll
    for (int j = 0; j < D / 8; ++j) {
        const int col = j * 8 + 2 * t;
        if (tok[0] >= 0)
            *reinterpret_cast<__half2*>(out + ((size_t)tok[0] * u.nh + hq) * D + col) =
                __floats2half2_rn(o[j][0] * inv[0], o[j][1] * inv[0]);
        if (tok[1] >= 0)
            *reinterpret_cast<__half2*>(out + ((size_t)tok[1] * u.nh + hq) * D + col) =
                __floats2half2_rn(o[j][2] * inv[1], o[j][3] * inv[1]);
    }
}

extern "C" __global__ void __launch_bounds__(256) window_f16_d64(
    const half* q, const half* k, const half* v, half* out, const WinU u)
{ window_impl<64>(q, k, v, out, u); }

extern "C" __global__ void __launch_bounds__(256) window_f16_d128(
    const half* q, const half* k, const half* v, half* out, const WinU u)
{ window_impl<128>(q, k, v, out, u); }
"#;

/// Dynamic shared memory for head dim `d`: Q tile + one K/V stage, padded.
fn smem_bytes(d: u32) -> u32 {
    (BR + 2 * BC) * (d + 8) * 2
}

/// Query tile `(th, tw)` (`th * tw = 128`) minimizing gathered keys for the
/// attended windows, and its tile counts.
fn pick_tile(u: &WindowAttnUniform) -> (u32, u32) {
    let (wh, ww) = u.windows[0];
    let (wh, ww) = if u.clamp {
        (wh.min(u.h), ww.min(u.w))
    } else {
        (wh, ww)
    };
    [8u32, 16, 32, 64, 128]
        .into_iter()
        .map(|tw| (BR / tw, tw))
        .min_by_key(|&(th, tw)| {
            let tiles = (u.h.div_ceil(th) * u.w.div_ceil(tw)) as u64;
            let rect = ((th + wh - 1).min(u.h) * (tw + ww - 1).min(u.w)) as u64;
            tiles * rect
        })
        .expect("candidates")
}

pub struct WindowKernels {
    module: Module,
    d64: cudarc::driver::sys::CUfunction,
    d128: cudarc::driver::sys::CUfunction,
}

// SAFETY: plain driver handles; every use binds the module's context first.
unsafe impl Send for WindowKernels {}
unsafe impl Sync for WindowKernels {}

impl WindowKernels {
    pub fn new(ctx: Arc<CudaContext>, sm: (i32, i32)) -> Result<Self, String> {
        let module = Module::compile(ctx, SOURCE, "thinfer_window.cu", sm)?;
        let d64 = module.function(c"window_f16_d64", smem_bytes(64))?;
        let d128 = module.function(c"window_f16_d128", smem_bytes(128))?;
        Ok(Self { module, d64, d128 })
    }

    /// Whether this kernel takes `u` (head dim, plane counts, GQA).
    pub fn accepts(u: &WindowAttnUniform) -> bool {
        matches!(u.hd, 64 | 128)
            && u.kvh > 0
            && u.nh.is_multiple_of(u.kvh)
            && u.temporal_window.min(u.tk) <= MAX_TW
            && u.tq <= u.tk
    }

    /// Enqueue on device addresses (default stream, no wait: the caller
    /// settles). `Ok(false)` = unsupported (caller runs wgpu).
    pub fn run(
        &self,
        u: &WindowAttnUniform,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
    ) -> Result<bool, String> {
        if !Self::accepts(u) || ![q, k, v, out].iter().all(|a| a % 16 == 0) {
            return Ok(false);
        }
        let f = if u.hd == 64 { self.d64 } else { self.d128 };
        let (th, tw) = pick_tile(u);
        let mut win = [[0u32; 2]; MAX_PLANES];
        for (d, s) in win.iter_mut().zip(u.windows) {
            *d = [s.0, s.1];
        }
        let mut wu = WinU {
            tq: u.tq,
            tk: u.tk,
            h: u.h,
            w: u.w,
            nh: u.nh,
            kvh: u.kvh,
            tw: u.temporal_window.min(u.tk),
            clamp: u.clamp as u32,
            scale: u.scale,
            tile_h: th,
            tile_w: tw,
            tiles_x: u.w.div_ceil(tw),
            tiles_y: u.h.div_ceil(th),
            _pad: 0,
            win,
        };
        self.module.bind()?;
        let (mut q, mut k, mut v, mut out) = (q, k, v, out);
        let mut args: [*mut c_void; 5] = [
            (&raw mut q).cast(),
            (&raw mut k).cast(),
            (&raw mut v).cast(),
            (&raw mut out).cast(),
            (&raw mut wu).cast(),
        ];
        let grid = (wu.tiles_x * wu.tiles_y * u.tq, u.nh, u.b);
        self.module
            .launch(f, grid, THREADS, smem_bytes(u.hd), &mut args, "window_attn")?;
        Ok(true)
    }
}
