//! MobileWan ReHyAt (Recurrent Hybrid Attention) GPU kernels.
//!
//! Four F32-only compute kernels backing the MobileWan hybrid attention block:
//! a per-head grouped linear ([`GroupedMatmulF32`]), a polynomial feature
//! expansion ([`PolyExpandF32`]), the fused local-windowed-softmax +
//! linear-recurrent read ([`RehyatAttnF32`]), and the recurrent-state
//! accumulate ([`RehyatStateUpdateF32`]).
//!
//! All buffers are row-major with no batch dim (B=1). Every kernel reads and
//! writes `array<f32>` storage and accumulates internally in f32 — there are no
//! bf16/f16 variants (`WgslConfig` is ignored). Each op follows the sibling-
//! trait pattern (see [`crate::ops::bcast_modulate`]): a trait carrying the
//! kernel id + binding names, a `*Bufs<'a>` struct of borrowed [`BufRef`]s, a
//! `dispatch_*` free function, a shared `LAYOUT`, and a single unit-struct impl.
//!
//! Head dim is a compile-time 128; degree is 3, so the per-head feature dim is
//! `P = 3 * 128 = 384`. The recurrent state `S` is laid out `[h][p][d]` (index
//! `((h*P)+p)*HD + d`) and `z` is `[h][p]` (index `h*P + p`). Zeroing `S`/`z`
//! before the chunk loop is the caller's job, not these kernels'.

use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
use crate::ops::WgslConfig;
use crate::tensor::{ComputeDtype, F32};

// -------------------------------------------------------------------------
// OP 1 — GroupedMatmulF32
// -------------------------------------------------------------------------

/// Per-head ("grouped") linear: for each row `r`, head `h`, output `o`,
/// `y[r, h*Npg+o] = bias[h*Npg+o] + Σ_i x[r, h*Kpg+i] * W[(h*Npg+o), i]`.
/// `W` is row-major `[H*Npg, Kpg]`. One thread per output element.
pub trait GroupedMatmulOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const W: &'static str;
    const BIAS: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n: u32) -> [u32; 3] {
        super::linear_workgroups(n, 64)
    }
}

pub struct GroupedMatmulBufs<'a> {
    pub x: &'a BufRef,
    pub w: &'a BufRef,
    pub bias: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

/// Dispatch the tiled grouped matmul. The grid is one 16x16 workgroup per
/// (output-tile, row-tile, head): `[ceil(npg/16), ceil(rows/16), h]`. `rows`,
/// `h`, `npg` must match the uniform's fields (`kpg` rides in the uniform).
pub(crate) fn dispatch_grouped_matmul<B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &GroupedMatmulBufs<'_>,
    rows: u32,
    h: u32,
    npg: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.w.binding(1),
        bufs.bias.binding(2),
        bufs.out.binding(3),
        bufs.uniform.binding(4),
    ];
    let t = GROUPED_MATMUL_TILE;
    let grid = [npg.div_ceil(t), rows.div_ceil(t), h];
    backend.dispatch(encoder, pipeline, &bindings, grid)
}

const GROUPED_MATMUL_LAYOUT: &[BindingLayout] = &[
    BindingLayout {
        slot: 0,
        kind: BindingKind::StorageRead,
    },
    BindingLayout {
        slot: 1,
        kind: BindingKind::StorageRead,
    },
    BindingLayout {
        slot: 2,
        kind: BindingKind::StorageRead,
    },
    BindingLayout {
        slot: 3,
        kind: BindingKind::StorageReadWrite,
    },
    BindingLayout {
        slot: 4,
        kind: BindingKind::Uniform,
    },
];

/// Output tile edge for the tiled grouped matmul (16x16 tile, 256 threads,
/// 2KB workgroup storage). The Rust-side grid dims MUST match `TILE`.
pub const GROUPED_MATMUL_TILE: u32 = 16;

// Tiled per-head matmul: `out_h = X_h @ W_hᵀ + bias_h` for head `h = wid.z`.
// One 16x16 workgroup computes a 16-row x 16-output tile; X and Wᵀ tiles are
// staged in workgroup memory and reused across the tile (the naive one-thread-
// per-output kernel re-read each x-row `npg` times from global). f32 in and
// out with f32 accumulation - numerically identical to the naive form (same
// products, same summation order over k), so the parity test still holds.
const GROUPED_MATMUL_WGSL: &str = r#"
struct U { rows: u32, h: u32, npg: u32, kpg: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

const TILE: u32 = 16u;
var<workgroup> as_tile: array<f32, 256>; // TILE*TILE, X rows
var<workgroup> bs_tile: array<f32, 256>; // TILE*TILE, Wᵀ (w rows)

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.z;
    let tx = lid.x;
    let ty = lid.y;
    let row = wid.y * TILE + ty; // query row r
    let col = wid.x * TILE + tx; // output o within head

    var acc = 0.0;
    let nk = (u.kpg + TILE - 1u) / TILE;
    for (var kk = 0u; kk < nk; kk = kk + 1u) {
        let kbase = kk * TILE;
        // As[ty][tx] = x[row, h, kbase+tx]
        let ak = kbase + tx;
        var av = 0.0;
        if (row < u.rows && ak < u.kpg) {
            av = x[row * (u.h * u.kpg) + h * u.kpg + ak];
        }
        as_tile[ty * TILE + tx] = av;
        // Bs[ty][tx] = w[(h*npg + (wid.x*TILE+ty)), kbase+tx]  (o indexed by ty)
        let bo = wid.x * TILE + ty;
        let bk = kbase + tx;
        var bv = 0.0;
        if (bo < u.npg && bk < u.kpg) {
            bv = w[(h * u.npg + bo) * u.kpg + bk];
        }
        bs_tile[ty * TILE + tx] = bv;
        workgroupBarrier();
        for (var t = 0u; t < TILE; t = t + 1u) {
            acc = acc + as_tile[ty * TILE + t] * bs_tile[tx * TILE + t];
        }
        workgroupBarrier();
    }
    if (row < u.rows && col < u.npg) {
        let hn = h * u.npg + col;
        out[row * (u.h * u.npg) + hn] = acc + bias[hn];
    }
}
"#;

pub struct GroupedMatmulF32;

impl GroupedMatmulOp for GroupedMatmulF32 {
    const KERNEL_ID: &'static str = "rehyat.grouped_matmul.f32";
    type Dtype = F32;
    const X: &'static str = "rehyat/grouped_matmul/x";
    const W: &'static str = "rehyat/grouped_matmul/w";
    const BIAS: &'static str = "rehyat/grouped_matmul/bias";
    const OUTPUT: &'static str = "rehyat/grouped_matmul/out";
    fn wgsl(_cfg: &WgslConfig) -> &'static str {
        GROUPED_MATMUL_WGSL
    }
    fn layout() -> &'static [BindingLayout] {
        GROUPED_MATMUL_LAYOUT
    }
}

// -------------------------------------------------------------------------
// OP 2 — PolyExpandF32
// -------------------------------------------------------------------------

/// Polynomial feature expansion. For flat index `idx`, the within-head feature
/// is `f = idx % P` and its degree is `d = f / HD ∈ {0,1,2}`; the output raises
/// the input to `d+1` via repeated multiply. One thread per element.
pub trait PolyExpandOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n: u32) -> [u32; 3] {
        super::linear_workgroups(n, 64)
    }
}

pub struct PolyExpandBufs<'a> {
    pub x: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

/// `n_elems` is the element count (`rows * H * P`) and thread count.
pub(crate) fn dispatch_poly_expand<O: PolyExpandOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &PolyExpandBufs<'_>,
    n_elems: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.out.binding(1),
        bufs.uniform.binding(2),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_elems))
}

const POLY_EXPAND_LAYOUT: &[BindingLayout] = &[
    BindingLayout {
        slot: 0,
        kind: BindingKind::StorageRead,
    },
    BindingLayout {
        slot: 1,
        kind: BindingKind::StorageReadWrite,
    },
    BindingLayout {
        slot: 2,
        kind: BindingKind::Uniform,
    },
];

const POLY_EXPAND_WGSL: &str = r#"
struct U { p: u32, hd: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    if (i >= arrayLength(&out)) { return; }
    let f = i % u.p;
    let d = f / u.hd;
    let v = x[i];
    var r = v;
    if (d >= 1u) { r = r * v; }
    if (d >= 2u) { r = r * v; }
    out[i] = r;
}
"#;

pub struct PolyExpandF32;

impl PolyExpandOp for PolyExpandF32 {
    const KERNEL_ID: &'static str = "rehyat.poly_expand.f32";
    type Dtype = F32;
    const X: &'static str = "rehyat/poly_expand/x";
    const OUTPUT: &'static str = "rehyat/poly_expand/out";
    fn wgsl(_cfg: &WgslConfig) -> &'static str {
        POLY_EXPAND_WGSL
    }
    fn layout() -> &'static [BindingLayout] {
        POLY_EXPAND_LAYOUT
    }
}

// -------------------------------------------------------------------------
// OP 3 — RehyatAttnF32
// -------------------------------------------------------------------------

/// Fused local-windowed softmax (online form) plus the linear-recurrent read,
/// producing the final normalized output. `q` is pre-scaled by `1/sqrt(HD)`;
/// no extra scale is applied here. One thread per `(query token, head)`. The
/// windowed exponents are clamped to `>= -60` to match the reference; the
/// recurrent term is added without max-subtraction.
pub trait RehyatAttnOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const Q: &'static str;
    const KWIN: &'static str;
    const VWIN: &'static str;
    const PHIQ: &'static str;
    const S: &'static str;
    const Z: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n: u32) -> [u32; 3] {
        super::linear_workgroups(n, 64)
    }
}

pub struct RehyatAttnBufs<'a> {
    pub q: &'a BufRef,
    pub kwin: &'a BufRef,
    pub vwin: &'a BufRef,
    pub phiq: &'a BufRef,
    pub s: &'a BufRef,
    pub z: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

/// `n_threads` is `ct * H` (one thread per query token per head).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_rehyat_attn<O: RehyatAttnOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &RehyatAttnBufs<'_>,
    n_threads: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.q.binding(0),
        bufs.kwin.binding(1),
        bufs.vwin.binding(2),
        bufs.phiq.binding(3),
        bufs.s.binding(4),
        bufs.z.binding(5),
        bufs.out.binding(6),
        bufs.uniform.binding(7),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_threads))
}

const REHYAT_ATTN_LAYOUT: &[BindingLayout] = &[
    BindingLayout {
        slot: 0,
        kind: BindingKind::StorageRead,
    },
    BindingLayout {
        slot: 1,
        kind: BindingKind::StorageRead,
    },
    BindingLayout {
        slot: 2,
        kind: BindingKind::StorageRead,
    },
    BindingLayout {
        slot: 3,
        kind: BindingKind::StorageRead,
    },
    BindingLayout {
        slot: 4,
        kind: BindingKind::StorageRead,
    },
    BindingLayout {
        slot: 5,
        kind: BindingKind::StorageRead,
    },
    BindingLayout {
        slot: 6,
        kind: BindingKind::StorageReadWrite,
    },
    BindingLayout {
        slot: 7,
        kind: BindingKind::Uniform,
    },
];

// vec4 form of the fused windowed-softmax + linear-recurrent read. HEAD_DIM is
// the compile-time 128, so every per-head span is 32 vec4s: `q`/`kwin`/`vwin`/
// `s_state`/`out` are bound as `array<vec4<f32>>` (128-bit loads/stores; all
// per-head strides are multiples of 4), while `phiq`/`z_state` stay scalar (the
// recurrent loop needs each feature `p` as a scalar weight). The math matches
// the scalar kernel up to vec4 dot/accumulate reassociation (well inside the
// 1e-4 parity band).
const REHYAT_ATTN_WGSL: &str = r#"
struct U { ct: u32, win: u32, h: u32, hd: u32, p: u32, _pad0: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> kwin: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vwin: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> phiq: array<f32>;
@group(0) @binding(4) var<storage, read> s_state: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> z_state: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(7) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let idx = gid.y * (ng.x * 64u) + gid.x;
    if (idx >= u.ct * u.h) { return; }
    let qt = idx / u.h;
    let h = idx % u.h;
    let hd4 = u.hd / 4u; // 32 vec4 per head row

    var acc: array<vec4<f32>, 32>;
    var rnum: array<vec4<f32>, 32>;
    for (var j = 0u; j < hd4; j = j + 1u) {
        acc[j] = vec4<f32>(0.0);
        rnum[j] = vec4<f32>(0.0);
    }

    let qb4 = (qt * (u.h * u.hd) + h * u.hd) / 4u;
    var m = -1.0e30;
    var l = 0.0;
    for (var kt = 0u; kt < u.win; kt = kt + 1u) {
        let kb4 = (kt * (u.h * u.hd) + h * u.hd) / 4u;
        var s = 0.0;
        for (var j = 0u; j < hd4; j = j + 1u) {
            s = s + dot(q[qb4 + j], kwin[kb4 + j]);
        }
        let m_new = max(m, s);
        let corr = exp(max(m - m_new, -60.0));
        let w = exp(max(s - m_new, -60.0));
        for (var j = 0u; j < hd4; j = j + 1u) {
            acc[j] = acc[j] * corr + w * vwin[kb4 + j];
        }
        l = l * corr + w;
        m = m_new;
    }

    var rden = 0.0;
    let pbase = qt * (u.h * u.p) + h * u.p;
    for (var p = 0u; p < u.p; p = p + 1u) {
        let a = phiq[pbase + p];
        rden = rden + a * z_state[h * u.p + p];
        let sb4 = (((h * u.p) + p) * u.hd) / 4u;
        for (var j = 0u; j < hd4; j = j + 1u) {
            rnum[j] = rnum[j] + a * s_state[sb4 + j];
        }
    }

    let denom = l + rden + 1.0e-6;
    let ob4 = (qt * (u.h * u.hd) + h * u.hd) / 4u;
    for (var j = 0u; j < hd4; j = j + 1u) {
        out[ob4 + j] = (acc[j] + rnum[j]) / denom;
    }
}
"#;

pub struct RehyatAttnF32;

impl RehyatAttnOp for RehyatAttnF32 {
    const KERNEL_ID: &'static str = "rehyat.attn.f32";
    type Dtype = F32;
    const Q: &'static str = "rehyat/attn/q";
    const KWIN: &'static str = "rehyat/attn/kwin";
    const VWIN: &'static str = "rehyat/attn/vwin";
    const PHIQ: &'static str = "rehyat/attn/phiq";
    const S: &'static str = "rehyat/attn/s";
    const Z: &'static str = "rehyat/attn/z";
    const OUTPUT: &'static str = "rehyat/attn/out";
    fn wgsl(_cfg: &WgslConfig) -> &'static str {
        REHYAT_ATTN_WGSL
    }
    fn layout() -> &'static [BindingLayout] {
        REHYAT_ATTN_LAYOUT
    }
}

// -------------------------------------------------------------------------
// OP 4 — RehyatStateUpdateF32
// -------------------------------------------------------------------------

/// Recurrent-state accumulate (in-place `+=` into `S` and `z`). For each head
/// `h` and feature `p`, accumulate over the `pre` prefix tokens:
/// `z[h,p] += Σ_t phik[t, h*P+p]` and `S[h,p,d] += Σ_t phik[t, h*P+p] * v[t, h*HD+d]`.
/// One thread per `(head, feature)`.
pub trait RehyatStateUpdateOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const PHIK: &'static str;
    const V: &'static str;
    const S: &'static str;
    const Z: &'static str;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n: u32) -> [u32; 3] {
        super::linear_workgroups(n, 64)
    }
}

pub struct RehyatStateUpdateBufs<'a> {
    pub phik: &'a BufRef,
    pub v: &'a BufRef,
    pub s: &'a BufRef,
    pub z: &'a BufRef,
    pub uniform: &'a BufRef,
}

/// `n_threads` is `H * P` (one thread per head per feature).
pub(crate) fn dispatch_rehyat_state_update<O: RehyatStateUpdateOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &RehyatStateUpdateBufs<'_>,
    n_threads: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.phik.binding(0),
        bufs.v.binding(1),
        bufs.s.binding(2),
        bufs.z.binding(3),
        bufs.uniform.binding(4),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_threads))
}

const REHYAT_STATE_UPDATE_LAYOUT: &[BindingLayout] = &[
    BindingLayout {
        slot: 0,
        kind: BindingKind::StorageRead,
    },
    BindingLayout {
        slot: 1,
        kind: BindingKind::StorageRead,
    },
    BindingLayout {
        slot: 2,
        kind: BindingKind::StorageReadWrite,
    },
    BindingLayout {
        slot: 3,
        kind: BindingKind::StorageReadWrite,
    },
    BindingLayout {
        slot: 4,
        kind: BindingKind::Uniform,
    },
];

const REHYAT_STATE_UPDATE_WGSL: &str = r#"
struct U { pre: u32, h: u32, p: u32, hd: u32 };

@group(0) @binding(0) var<storage, read> phik: array<f32>;
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_state: array<f32>;
@group(0) @binding(3) var<storage, read_write> z_state: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let idx = gid.y * (ng.x * 64u) + gid.x;
    if (idx >= u.h * u.p) { return; }
    let h = idx / u.p;
    let p = idx % u.p;

    var sinc: array<f32, 128>;
    for (var d = 0u; d < u.hd; d = d + 1u) {
        sinc[d] = 0.0;
    }
    var zinc = 0.0;
    for (var t = 0u; t < u.pre; t = t + 1u) {
        let a = phik[t * (u.h * u.p) + h * u.p + p];
        zinc = zinc + a;
        let vbase = t * (u.h * u.hd) + h * u.hd;
        for (var d = 0u; d < u.hd; d = d + 1u) {
            sinc[d] = sinc[d] + a * v[vbase + d];
        }
    }

    z_state[h * u.p + p] = z_state[h * u.p + p] + zinc;
    let sbase = ((h * u.p) + p) * u.hd;
    for (var d = 0u; d < u.hd; d = d + 1u) {
        s_state[sbase + d] = s_state[sbase + d] + sinc[d];
    }
}
"#;

pub struct RehyatStateUpdateF32;

impl RehyatStateUpdateOp for RehyatStateUpdateF32 {
    const KERNEL_ID: &'static str = "rehyat.state_update.f32";
    type Dtype = F32;
    const PHIK: &'static str = "rehyat/state_update/phik";
    const V: &'static str = "rehyat/state_update/v";
    const S: &'static str = "rehyat/state_update/s";
    const Z: &'static str = "rehyat/state_update/z";
    fn wgsl(_cfg: &WgslConfig) -> &'static str {
        REHYAT_STATE_UPDATE_WGSL
    }
    fn layout() -> &'static [BindingLayout] {
        REHYAT_STATE_UPDATE_LAYOUT
    }
}

// -------------------------------------------------------------------------
// Tests — pure-Rust CPU reference + GPU parity on a real wgpu device.
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Binding, PowerPreference, WgpuBackend, WgpuConfig};
    use crate::mem::VramCategory;
    use crate::tensor::GpuBufferId;

    // Head dim / degree / feature dim are compile-time constants in the shader
    // (`array<f32, 128>`), so the tests exercise the real HD=128 path.
    const HD: usize = 128;
    const DEG: usize = 3;
    const P: usize = DEG * HD;

    /// Deterministic pseudo-random f32 pattern (no rng), in roughly [-0.6, 0.6].
    fn pat(i: usize) -> f32 {
        (((i * 37 + 3) % 13) as f32) * 0.1 - 0.6
    }

    /// Uniform bytes, padded to a 16-byte multiple (wgpu uniform alignment).
    fn uni(vals: &[u32]) -> Vec<u8> {
        let mut b: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        while !b.len().is_multiple_of(16) {
            b.push(0);
        }
        b
    }

    // ----- CPU reference implementations (mirror the WGSL exactly) -----

    fn cpu_grouped_matmul(
        x: &[f32],
        w: &[f32],
        bias: &[f32],
        rows: usize,
        h: usize,
        npg: usize,
        kpg: usize,
    ) -> Vec<f32> {
        let mut y = vec![0f32; rows * h * npg];
        for r in 0..rows {
            for hh in 0..h {
                for o in 0..npg {
                    let hn = hh * npg + o;
                    let mut acc = bias[hn];
                    for i in 0..kpg {
                        acc += x[r * (h * kpg) + hh * kpg + i] * w[hn * kpg + i];
                    }
                    y[r * (h * npg) + hn] = acc;
                }
            }
        }
        y
    }

    fn cpu_poly_expand(x: &[f32], rows: usize, h: usize) -> Vec<f32> {
        let mut y = vec![0f32; rows * h * P];
        for (idx, &v) in x.iter().enumerate() {
            let f = idx % P;
            let d = f / HD;
            let mut r = v;
            if d >= 1 {
                r *= v;
            }
            if d >= 2 {
                r *= v;
            }
            y[idx] = r;
        }
        y
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_rehyat_attn(
        q: &[f32],
        kwin: &[f32],
        vwin: &[f32],
        phiq: &[f32],
        s_state: &[f32],
        z_state: &[f32],
        ct: usize,
        win: usize,
        h: usize,
    ) -> Vec<f32> {
        let mut out = vec![0f32; ct * h * HD];
        for qt in 0..ct {
            for hh in 0..h {
                let qbase = qt * (h * HD) + hh * HD;
                let mut acc = vec![0f32; HD];
                let mut m = -1.0e30f32;
                let mut l = 0.0f32;
                for kt in 0..win {
                    let kbase = kt * (h * HD) + hh * HD;
                    let mut s = 0.0f32;
                    for c in 0..HD {
                        s += q[qbase + c] * kwin[kbase + c];
                    }
                    let m_new = m.max(s);
                    let corr = (m - m_new).max(-60.0).exp();
                    let wgt = (s - m_new).max(-60.0).exp();
                    for c in 0..HD {
                        acc[c] = acc[c] * corr + wgt * vwin[kbase + c];
                    }
                    l = l * corr + wgt;
                    m = m_new;
                }
                let mut rden = 0.0f32;
                let mut rnum = vec![0f32; HD];
                let pbase = qt * (h * P) + hh * P;
                for p in 0..P {
                    let a = phiq[pbase + p];
                    rden += a * z_state[hh * P + p];
                    let sbase = (hh * P + p) * HD;
                    for d in 0..HD {
                        rnum[d] += a * s_state[sbase + d];
                    }
                }
                let denom = l + rden + 1.0e-6;
                let obase = qt * (h * HD) + hh * HD;
                for d in 0..HD {
                    out[obase + d] = (acc[d] + rnum[d]) / denom;
                }
            }
        }
        out
    }

    fn cpu_state_update(
        phik: &[f32],
        v: &[f32],
        s_in: &[f32],
        z_in: &[f32],
        pre: usize,
        h: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut s_state = s_in.to_vec();
        let mut z_state = z_in.to_vec();
        for hh in 0..h {
            for p in 0..P {
                let mut zinc = 0.0f32;
                let mut sinc = vec![0f32; HD];
                for t in 0..pre {
                    let a = phik[t * (h * P) + hh * P + p];
                    zinc += a;
                    let vbase = t * (h * HD) + hh * HD;
                    for d in 0..HD {
                        sinc[d] += a * v[vbase + d];
                    }
                }
                z_state[hh * P + p] += zinc;
                let sbase = (hh * P + p) * HD;
                for d in 0..HD {
                    s_state[sbase + d] += sinc[d];
                }
            }
        }
        (s_state, z_state)
    }

    // ----- GPU harness -----

    async fn backend() -> Option<WgpuBackend> {
        let cfg = WgpuConfig {
            power_preference: PowerPreference::HighPerformance,
            ..Default::default()
        };
        WgpuBackend::new_with_config(cfg).await.ok()
    }

    fn alloc(b: &WgpuBackend, bytes: usize) -> GpuBufferId {
        b.allocate_in(bytes.max(4) as u64, VramCategory::Workspace)
            .unwrap()
    }

    fn upload(b: &WgpuBackend, data: &[f32]) -> GpuBufferId {
        let id = alloc(b, data.len() * 4);
        b.write_buffer(id, 0, bytemuck::cast_slice(data)).unwrap();
        id
    }

    fn bind(slot: u32, id: GpuBufferId, bytes: usize) -> Binding {
        Binding {
            slot,
            buffer: id,
            offset: 0,
            size: bytes.max(4) as u64,
        }
    }

    async fn read_f32(b: &WgpuBackend, id: GpuBufferId, n: usize) -> Vec<f32> {
        let raw = b.read_buffer(id, 0, (n * 4) as u64).await.unwrap();
        bytemuck::cast_slice(&raw).to_vec()
    }

    fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn grouped_matmul_gpu_matches_cpu() {
        pollster::block_on(async {
            let Some(b) = backend().await else {
                eprintln!("skip grouped_matmul_gpu_matches_cpu: no GPU adapter");
                return;
            };
            let (rows, h, npg, kpg) = (5usize, 4usize, 6usize, 7usize);
            let x: Vec<f32> = (0..rows * h * kpg).map(pat).collect();
            let w: Vec<f32> = (0..h * npg * kpg).map(|i| pat(i + 11)).collect();
            let bias: Vec<f32> = (0..h * npg).map(|i| pat(i + 5)).collect();
            let expect = cpu_grouped_matmul(&x, &w, &bias, rows, h, npg, kpg);

            let bx = upload(&b, &x);
            let bw = upload(&b, &w);
            let bb = upload(&b, &bias);
            let n_out = rows * h * npg;
            let bo = alloc(&b, n_out * 4);
            let bu = alloc(&b, 16);
            b.write_buffer(
                bu,
                0,
                &uni(&[rows as u32, h as u32, npg as u32, kpg as u32]),
            )
            .unwrap();

            let pipe = b
                .create_pipeline(
                    "t_grouped_matmul",
                    GroupedMatmulF32::wgsl(&WgslConfig::FP32),
                    "main",
                    GroupedMatmulF32::layout(),
                )
                .await
                .unwrap();
            let mut enc = b.create_command_encoder();
            let t = GROUPED_MATMUL_TILE;
            let grid = [
                (npg as u32).div_ceil(t),
                (rows as u32).div_ceil(t),
                h as u32,
            ];
            b.dispatch(
                &mut enc,
                &pipe,
                &[
                    bind(0, bx, x.len() * 4),
                    bind(1, bw, w.len() * 4),
                    bind(2, bb, bias.len() * 4),
                    bind(3, bo, n_out * 4),
                    bind(4, bu, 16),
                ],
                grid,
            )
            .unwrap();
            b.submit(enc).await.unwrap();
            let got = read_f32(&b, bo, n_out).await;

            let err = max_abs_err(&got, &expect);
            eprintln!("[grouped_matmul] max_abs_err={err:.3e}");
            assert!(err <= 1e-4, "grouped_matmul err {err:.3e}");
            for id in [bx, bw, bb, bo, bu] {
                b.free(id);
            }
        });
    }

    #[test]
    fn poly_expand_gpu_matches_cpu() {
        pollster::block_on(async {
            let Some(b) = backend().await else {
                eprintln!("skip poly_expand_gpu_matches_cpu: no GPU adapter");
                return;
            };
            let (rows, h) = (3usize, 2usize);
            let n = rows * h * P;
            let x: Vec<f32> = (0..n).map(pat).collect();
            let expect = cpu_poly_expand(&x, rows, h);

            let bx = upload(&b, &x);
            let bo = alloc(&b, n * 4);
            let bu = alloc(&b, 16);
            b.write_buffer(bu, 0, &uni(&[P as u32, HD as u32, 0, 0]))
                .unwrap();

            let pipe = b
                .create_pipeline(
                    "t_poly_expand",
                    PolyExpandF32::wgsl(&WgslConfig::FP32),
                    "main",
                    PolyExpandF32::layout(),
                )
                .await
                .unwrap();
            let mut enc = b.create_command_encoder();
            b.dispatch(
                &mut enc,
                &pipe,
                &[bind(0, bx, n * 4), bind(1, bo, n * 4), bind(2, bu, 16)],
                crate::ops::linear_workgroups(n as u32, 64),
            )
            .unwrap();
            b.submit(enc).await.unwrap();
            let got = read_f32(&b, bo, n).await;

            let err = max_abs_err(&got, &expect);
            eprintln!("[poly_expand] max_abs_err={err:.3e}");
            assert!(err <= 1e-4, "poly_expand err {err:.3e}");
            for id in [bx, bo, bu] {
                b.free(id);
            }
        });
    }

    #[test]
    fn rehyat_attn_gpu_matches_cpu() {
        pollster::block_on(async {
            let Some(b) = backend().await else {
                eprintln!("skip rehyat_attn_gpu_matches_cpu: no GPU adapter");
                return;
            };
            let (ct, win, h) = (4usize, 3usize, 2usize);
            // q pre-scaled by 1/sqrt(HD); shrink the raw pattern accordingly so
            // the exp() args stay in a sane range.
            let scale = 1.0 / (HD as f32).sqrt();
            let q: Vec<f32> = (0..ct * h * HD).map(|i| pat(i) * scale).collect();
            let kwin: Vec<f32> = (0..win * h * HD).map(|i| pat(i + 7)).collect();
            let vwin: Vec<f32> = (0..win * h * HD).map(|i| pat(i + 3)).collect();
            let phiq: Vec<f32> = (0..ct * h * P).map(|i| pat(i + 9) * 0.05).collect();
            let s_state: Vec<f32> = (0..h * P * HD).map(|i| pat(i + 2) * 0.02).collect();
            let z_state: Vec<f32> = (0..h * P).map(|i| pat(i + 4) * 0.02 + 0.1).collect();
            let expect = cpu_rehyat_attn(&q, &kwin, &vwin, &phiq, &s_state, &z_state, ct, win, h);

            let bq = upload(&b, &q);
            let bk = upload(&b, &kwin);
            let bv = upload(&b, &vwin);
            let bp = upload(&b, &phiq);
            let bs = upload(&b, &s_state);
            let bz = upload(&b, &z_state);
            let n_out = ct * h * HD;
            let bo = alloc(&b, n_out * 4);
            let bu = alloc(&b, 32);
            b.write_buffer(
                bu,
                0,
                &uni(&[
                    ct as u32, win as u32, h as u32, HD as u32, P as u32, 0, 0, 0,
                ]),
            )
            .unwrap();

            let pipe = b
                .create_pipeline(
                    "t_rehyat_attn",
                    RehyatAttnF32::wgsl(&WgslConfig::FP32),
                    "main",
                    RehyatAttnF32::layout(),
                )
                .await
                .unwrap();
            let n_threads = ct * h;
            let mut enc = b.create_command_encoder();
            b.dispatch(
                &mut enc,
                &pipe,
                &[
                    bind(0, bq, q.len() * 4),
                    bind(1, bk, kwin.len() * 4),
                    bind(2, bv, vwin.len() * 4),
                    bind(3, bp, phiq.len() * 4),
                    bind(4, bs, s_state.len() * 4),
                    bind(5, bz, z_state.len() * 4),
                    bind(6, bo, n_out * 4),
                    bind(7, bu, 32),
                ],
                crate::ops::linear_workgroups(n_threads as u32, 64),
            )
            .unwrap();
            b.submit(enc).await.unwrap();
            let got = read_f32(&b, bo, n_out).await;

            let err = max_abs_err(&got, &expect);
            eprintln!("[rehyat_attn] max_abs_err={err:.3e}");
            assert!(err <= 1e-4, "rehyat_attn err {err:.3e}");
            for id in [bq, bk, bv, bp, bs, bz, bo, bu] {
                b.free(id);
            }
        });
    }

    #[test]
    fn rehyat_state_update_gpu_matches_cpu() {
        pollster::block_on(async {
            let Some(b) = backend().await else {
                eprintln!("skip rehyat_state_update_gpu_matches_cpu: no GPU adapter");
                return;
            };
            let (pre, h) = (3usize, 2usize);
            let phik: Vec<f32> = (0..pre * h * P).map(|i| pat(i) * 0.05).collect();
            let v: Vec<f32> = (0..pre * h * HD).map(|i| pat(i + 6)).collect();
            let s_in: Vec<f32> = (0..h * P * HD).map(|i| pat(i + 1) * 0.02).collect();
            let z_in: Vec<f32> = (0..h * P).map(|i| pat(i + 8) * 0.02).collect();
            let (expect_s, expect_z) = cpu_state_update(&phik, &v, &s_in, &z_in, pre, h);

            let bphik = upload(&b, &phik);
            let bv = upload(&b, &v);
            let bs = upload(&b, &s_in);
            let bz = upload(&b, &z_in);
            let bu = alloc(&b, 16);
            b.write_buffer(bu, 0, &uni(&[pre as u32, h as u32, P as u32, HD as u32]))
                .unwrap();

            let pipe = b
                .create_pipeline(
                    "t_rehyat_state_update",
                    RehyatStateUpdateF32::wgsl(&WgslConfig::FP32),
                    "main",
                    RehyatStateUpdateF32::layout(),
                )
                .await
                .unwrap();
            let n_threads = h * P;
            let mut enc = b.create_command_encoder();
            b.dispatch(
                &mut enc,
                &pipe,
                &[
                    bind(0, bphik, phik.len() * 4),
                    bind(1, bv, v.len() * 4),
                    bind(2, bs, s_in.len() * 4),
                    bind(3, bz, z_in.len() * 4),
                    bind(4, bu, 16),
                ],
                crate::ops::linear_workgroups(n_threads as u32, 64),
            )
            .unwrap();
            b.submit(enc).await.unwrap();
            let got_s = read_f32(&b, bs, s_in.len()).await;
            let got_z = read_f32(&b, bz, z_in.len()).await;

            let err_s = max_abs_err(&got_s, &expect_s);
            let err_z = max_abs_err(&got_z, &expect_z);
            eprintln!("[state_update] max_abs_err s={err_s:.3e} z={err_z:.3e}");
            assert!(
                err_s <= 1e-4 && err_z <= 1e-4,
                "state_update s={err_s:.3e} z={err_z:.3e}"
            );
            for id in [bphik, bv, bs, bz, bu] {
                b.free(id);
            }
        });
    }
}
