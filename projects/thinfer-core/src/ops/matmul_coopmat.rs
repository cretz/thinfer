//! Cooperative-matrix (tensor-core) matmul. Maps the dense float matmul onto
//! the GPU's hardware matrix units via WGSL `coop_matNxN` types
//! (`VK_KHR_cooperative_matrix` under the hood). f16 inputs, f32 accumulate -
//! far above the scalar compute-shader matmul's ~3 TFLOPS bf16 floor on a
//! tensor-core GPU.
//!
//! Scope/shape constraints (naga 29 + the KHR extension):
//! - Tiles are SQUARE (`coop_mat8x8` / `coop_mat16x16`), so one
//!   `coopMultiplyAdd` is a `TxTxT` block; the kernel loops K in `T`-steps.
//! - Cooperative ops are SUBGROUP-scoped: one workgroup == one subgroup. The
//!   workgroup size is the device subgroup width, baked at build (the caller
//!   must only use this when subgroup_min == subgroup_max). Each workgroup
//!   computes a `(tm*T) x (tn*T)` register-tiled output block.
//! - Ragged M/N need NO operand padding when the device reports
//!   `cooperativeMatrixRobustBufferAccess` (the RTX 5070 does): out-of-bounds
//!   `coopLoad` tile reads clamp to 0, and the matmul is row/col independent so
//!   the in-bounds outputs are unaffected; the staged store is bounds-checked.
//!   Without robust access, the caller must pad A/B to `T` multiples.
//!
//! Layout: 0=A `[M,K]` f16 row-major, 1=B f16 (`[K,N]` row-major, or `[N,K]`
//! n-major when `cfg.b_col_major`), 2=Out `[M,N]` (f32 or f16), 3=Dims uniform
//! `{m,n,k,_}`. Output dtype is selected by `CoopmatOut`.

use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};

/// Output storage for the coopmat matmul. Accumulation is always f32; this
/// picks how the result tile is written back.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CoopmatOut {
    /// `array<f32>`, one element per cell. Bit-exact reference path; used by
    /// conformance.
    F32,
    /// `array<vec2<f16>>`, two cells per word. The integration target: the
    /// matmul site casts f16 -> bf16 afterwards for the residual stream.
    F16,
}

impl CoopmatOut {
    fn tag(self) -> &'static str {
        match self {
            CoopmatOut::F32 => "f32",
            CoopmatOut::F16 => "f16",
        }
    }
}

/// Register-tiled coopmat GEMM config. One workgroup == one subgroup, computing
/// a `tm x tn` grid of `tile x tile` coop accumulators (a `WM = tm*tile` by
/// `WN = tn*tile` output block). Each K-step `coopLoad`s the A/B subtiles
/// DIRECTLY from global memory (no shared-memory staging - on this naga/wgpu
/// Vulkan stack the hardware coop-load + L2 already serve the redundant reads,
/// and manual shared staging measured strictly slower) and accumulates into
/// register-resident tiles, so each loaded operand is reused `tn`/`tm` times
/// across the grid. Many small workgroups give the occupancy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CoopmatMatmulConfig {
    /// Square coop-matrix tile side (8 or 16); the MMA shape is `tile^3`.
    pub tile: u32,
    /// Subgroup width = workgroup size; uniform min==max required.
    pub subgroup_size: u32,
    /// Coop accumulator tiles in the row / col dim (register blocking factor).
    pub tm: u32,
    pub tn: u32,
    /// B operand layout. `false`: B is `[K,N]` row-major (loaded with
    /// `coopLoadT`) - the standalone form. `true`: B is `[N,K]` row-major (the
    /// natural Linear weight / dequant "n-major" layout), loaded as a
    /// column-major coop tile with `coopLoad` so no transpose is needed.
    pub b_col_major: bool,
    pub out: CoopmatOut,
}

impl CoopmatMatmulConfig {
    pub const KERNEL_ID: &'static str = "matmul.coopmat";

    /// Default for the RTX 5070: a 2x2 accumulator block (32x32 output per
    /// workgroup), the best point found in the sweep (register tiling past 4x4
    /// spills and collapses; 2x2 edges out 1x1 and 4x4).
    pub fn new(tile: u32, subgroup_size: u32, out: CoopmatOut) -> Self {
        Self {
            tile,
            subgroup_size,
            tm: 2,
            tn: 2,
            b_col_major: false,
            out,
        }
    }

    /// Workgroup output block: rows / cols.
    pub fn wm(&self) -> u32 {
        self.tm * self.tile
    }
    pub fn wn(&self) -> u32 {
        self.tn * self.tile
    }

    pub fn hint(&self) -> String {
        format!(
            "coopmat-t{}_sg{}_tm{}_tn{}_{}{}",
            self.tile,
            self.subgroup_size,
            self.tm,
            self.tn,
            self.out.tag(),
            if self.b_col_major { "_bcm" } else { "" },
        )
    }

    /// One workgroup per `WM x WN` output block.
    pub fn workgroups(&self, m: u32, n: u32) -> [u32; 3] {
        [n.div_ceil(self.wn()), m.div_ceil(self.wm()), 1]
    }
}

const LAYOUT: &[BindingLayout] = &[
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
        kind: BindingKind::Uniform,
    },
];

pub fn layout() -> &'static [BindingLayout] {
    LAYOUT
}

/// Build the register-tiled coopmat GEMM WGSL for `cfg`. One subgroup per
/// workgroup computes a `tm x tn` grid of accumulators; each K-step coop-loads
/// the `tm` A subtiles and `tn` B subtiles directly from global (no shared
/// staging) and issues `tm*tn` `coopMultiplyAdd`s, so each loaded operand is
/// reused across the register grid. Output tiles are stored at the end through
/// a small f32 staging region with bounds + dtype conversion (ragged M/N safe;
/// A/B coop-loads of trailing tiles read in-bounds because the workgroup grid
/// is sized by div_ceil and the loads clamp via the row/col guards below).
pub fn build_wgsl(cfg: &CoopmatMatmulConfig) -> String {
    let t = cfg.tile;
    let sg = cfg.subgroup_size;
    let (tm, tn) = (cfg.tm, cfg.tn);
    let mat = format!("coop_mat{t}x{t}");
    let wm = cfg.wm();
    let wn = cfg.wn();
    let nacc = tm * tn;
    let tt = t * t;
    // B load: row-major [K,N] via coopLoadT (stride n), or n-major [N,K] via a
    // column-major coopLoad (stride k) for the natural Linear-weight layout.
    let b_load = if cfg.b_col_major {
        format!("coopLoad<{mat}<f16, B>>(&b[(wg_col + tj * {t}u) * k + kk], k)")
    } else {
        format!("coopLoadT<{mat}<f16, B>>(&b[kk * n + wg_col + tj * {t}u], n)")
    };

    let out_decl = match cfg.out {
        CoopmatOut::F32 => "@group(0) @binding(2) var<storage, read_write> out: array<f32>;",
        CoopmatOut::F16 => "@group(0) @binding(2) var<storage, read_write> out: array<vec2<f16>>;",
    };
    let store_body = match cfg.out {
        CoopmatOut::F32 => format!(
            r#"            var e = lid;
            loop {{
                if (e >= {tt}u) {{ break; }}
                let rr = e / {t}u;
                let cc = e % {t}u;
                let gr = tile_row + rr;
                let gc = tile_col + cc;
                if (gr < m && gc < n) {{ out[gr * n + gc] = cstage[e]; }}
                e += {sg}u;
            }}"#
        ),
        CoopmatOut::F16 => format!(
            r#"            let wpr = {t}u / 2u;
            var e = lid;
            loop {{
                if (e >= {tt}u / 2u) {{ break; }}
                let rr = e / wpr;
                let cw = e % wpr;
                let gr = tile_row + rr;
                let gc = tile_col + cw * 2u;
                if (gr < m && gc < n) {{
                    let base = rr * {t}u + cw * 2u;
                    out[(gr * n + gc) / 2u] =
                        vec2<f16>(f16(cstage[base]), f16(cstage[base + 1u]));
                }}
                e += {sg}u;
            }}"#
        ),
    };

    format!(
        r#"enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> a: array<f16>;
@group(0) @binding(1) var<storage, read> b: array<f16>;
{out_decl}
@group(0) @binding(3) var<uniform> dims: vec4<u32>;

var<workgroup> cstage: array<f32, {tt}>;

@compute @workgroup_size({sg}, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {{
    let m = dims.x;
    let n = dims.y;
    let k = dims.z;
    let wg_row = wid.y * {wm}u;
    let wg_col = wid.x * {wn}u;

    var acc: array<{mat}<f32, C>, {nacc}>;

    // MMA: acc[ti][tj] += A[ti, kk] * B[kk, tj]. A subtiles loaded once per
    // K-step, reused across tn; B subtiles reused across tm. Direct from global.
    var kk = 0u;
    loop {{
        if (kk >= k) {{ break; }}
        var amat: array<{mat}<f16, A>, {tm}>;
        for (var ti = 0u; ti < {tm}u; ti += 1u) {{
            amat[ti] = coopLoadT<{mat}<f16, A>>(&a[(wg_row + ti * {t}u) * k + kk], k);
        }}
        for (var tj = 0u; tj < {tn}u; tj += 1u) {{
            let bmat = {b_load};
            for (var ti = 0u; ti < {tm}u; ti += 1u) {{
                acc[ti * {tn}u + tj] = coopMultiplyAdd(amat[ti], bmat, acc[ti * {tn}u + tj]);
            }}
        }}
        kk += {t}u;
    }}

    // Store each accumulator tile via the shared f32 staging slot.
    for (var ti = 0u; ti < {tm}u; ti += 1u) {{
        for (var tj = 0u; tj < {tn}u; tj += 1u) {{
            coopStoreT(acc[ti * {tn}u + tj], &cstage[0], {t}u);
            workgroupBarrier();
            let tile_row = wg_row + ti * {t}u;
            let tile_col = wg_col + tj * {t}u;
{store_body}
            workgroupBarrier();
        }}
    }}
}}
"#
    )
}

pub struct CoopmatBufs<'a> {
    pub a: &'a BufRef,
    pub b: &'a BufRef,
    pub out: &'a BufRef,
    pub dims: &'a BufRef,
}

pub fn dispatch_coopmat<B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    cfg: &CoopmatMatmulConfig,
    bufs: &CoopmatBufs<'_>,
    m: u32,
    n: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.a.binding(0),
        bufs.b.binding(1),
        bufs.out.binding(2),
        bufs.dims.binding(3),
    ];
    backend.dispatch(encoder, pipeline, &bindings, cfg.workgroups(m, n))
}
