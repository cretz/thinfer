//! DP4A matmul: A is packed int8 with **asymmetric** per-K=32 params
//! `(scale, zero)` (llama.cpp Q8_1-style), B is packed int8 with **symmetric**
//! per-K=32 f32 scale (weights), output is `array<vec2<f16>>` `[M, N]` paired.
//!
//! Asymmetric decomposition (per (m, n), summed over the K dimension):
//!   `O = Σ (qa*sa + za) * qb*sb`
//!     = `Σ_{k_block} sa[m,t] * sb[n,t] * Σ(qa*qb)`        (main DP4A path)
//!     + `Σ_{k_block} za[m,t] * sb[n,t] * Σ(qb)`           (correction)
//!
//! Where `qsum_b[n, k_block] = Σ_{k in block} qb[n, k]` is precomputed at
//! weight-dequant time (see `dequant_i8.rs`) and passed in as a new binding.
//!
//! Kernel structure (onnxruntime DP4AMatMulNBits-style register-resident
//! subtiles): the workgroup owns a `tile x tile` output block and loads one
//! K=32 slice of A and B into shared memory per outer step (2 vec4<u32> per
//! row). The tile is split into 16x16 subtiles with 16 lanes each; every
//! lane owns ONE A row (kept in 2 vec4 registers for the whole K-step) and
//! accumulates 16 output columns. B columns are fetched per unrolled column
//! step either:
//! * `subgroupShuffle` from the lane that owns that column (runtime branch
//!   `sg_size >= 16`; lanes of a subtile are 16 consecutive subgroup lanes,
//!   so the source lane is `(sg_id - sg_id % 16) + col`), or
//! * a single-address shared read (all 16 lanes hit the same word: hardware
//!   broadcast). This branch is also the whole kernel when `use_subgroup`
//!   is false (adapters without `Features::SUBGROUP`).
//!
//! The branch is a uniform `if`/`else` on the `subgroup_size` builtin, NOT a
//! `select()` (select evaluates both arms; that artifact poisoned the old
//! subgroup A/B).
//!
//! Per outer K-step (one K=32 sub-block), per output cell:
//!   dot_i32 = Σ_s dot4I8Packed(a_word_s, b_word_s)        (8 dots, exact)
//!   acc_f32 += f32(dot_i32) * sa * sb + za * sb * qsum_b
//! Identical expression and K-block order as the previous register-blocked
//! kernel: outputs are bit-identical to it.

use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};

/// Tile shape. The output tile is `tile x tile` (square), split into 16x16
/// subtiles; threads = tile*tile/16 (256 at the default tile=64). The K step
/// is fixed at 32 (one K=32 sub-block per barrier, matching the per-block
/// scale granularity).
///
/// `use_subgroup` adds the `sg_size >= 16` shuffle path (B columns fetched
/// from the owning lane's registers instead of shared memory). The emitted
/// WGSL starts at `enable f16;` with no `enable subgroups;` (naga rejects
/// it). On the web (Tint) backend that directive is required, so the model
/// layer prepends `backend.subgroup_enable_directive()` at the build site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatMulI8Config {
    pub tile: u32,
    pub use_subgroup: bool,
}

impl MatMulI8Config {
    pub const DEFAULT: Self = Self {
        tile: 64,
        use_subgroup: false,
    };

    pub const fn threads(&self) -> u32 {
        self.tile * self.tile / 16
    }

    fn validate(&self) {
        assert!(
            self.tile >= 16 && self.tile.is_multiple_of(16),
            "tile must be a multiple of 16 (16x16 subtiles)"
        );
        assert!(
            self.threads() <= 256,
            "threads = tile^2/16 exceeds the web-baseline workgroup cap of 256"
        );
    }
}

impl Default for MatMulI8Config {
    fn default() -> Self {
        Self::DEFAULT
    }
}

pub fn hint(c: &MatMulI8Config) -> String {
    let sg = if c.use_subgroup { "_sg" } else { "" };
    format!("matmul_i8-t{}{}", c.tile, sg)
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
        kind: BindingKind::StorageReadWrite,
    },
    BindingLayout {
        slot: 6,
        kind: BindingKind::Uniform,
    },
    // DIAG bindings (always present; `dbg.enable == 0` makes the kernel
    // skip the trace write entirely). When enable != 0, the single thread
    // that computes output cell (dbg.m, dbg.n) writes 8 f32s per K-block
    // into dbg_out so callers can side-by-side compare GPU intermediates
    // against a CPU recompute from the captured (a, a_p, b, b_sc, b_qs)
    // bytes. Layout: dbg_out[t*8 + 0..8] = (sa, za, sb, qsum, dot, main,
    // corr, acc_running) plus a 16-f32 probe area at index (k/32)*8.
    // dbg_out needs (k/32)*8 + 16 f32s (= 976 for k=3840); callers must
    // size to at least that. With enable=0, callers may pass a 1-element
    // scratch buffer.
    BindingLayout {
        slot: 7,
        kind: BindingKind::StorageReadWrite,
    },
    BindingLayout {
        slot: 8,
        kind: BindingKind::Uniform,
    },
];

pub fn layout() -> &'static [BindingLayout] {
    LAYOUT
}

/// Per-column compute block: fetch B words/params for column `j` of the
/// subtile (shuffle or broadcast form), run the 8-dot DP4A reduction against
/// the lane's register-resident A row, and fold with the asymmetric scale
/// correction. Includes the DIAG trace for the (dbg.m, dbg.n) cell.
fn col_block(j: u32, shuffle: bool) -> String {
    let (b0, b1, sc, qs) = if shuffle {
        (
            format!("subgroupShuffle(own_b0, sg_base + {j}u)"),
            format!("subgroupShuffle(own_b1, sg_base + {j}u)"),
            format!("subgroupShuffle(own_sc, sg_base + {j}u)"),
            format!("subgroupShuffle(own_qs, sg_base + {j}u)"),
        )
    } else {
        (
            format!("tile_b[0][base_b + {j}u]"),
            format!("tile_b[1][base_b + {j}u]"),
            format!("tile_b_scale[base_b + {j}u]"),
            format!("tile_b_qsum[base_b + {j}u]"),
        )
    };
    format!(
        r#"            {{
                let bb0: vec4<u32> = {b0};
                let bb1: vec4<u32> = {b1};
                let b_sc: f32 = {sc};
                let b_qs: f32 = {qs};
                var dt: i32 = dot4I8Packed(own_a0.x, bb0.x);
                dt += dot4I8Packed(own_a0.y, bb0.y);
                dt += dot4I8Packed(own_a0.z, bb0.z);
                dt += dot4I8Packed(own_a0.w, bb0.w);
                dt += dot4I8Packed(own_a1.x, bb1.x);
                dt += dot4I8Packed(own_a1.y, bb1.y);
                dt += dot4I8Packed(own_a1.z, bb1.z);
                dt += dot4I8Packed(own_a1.w, bb1.w);
                acc_{j} = acc_{j} + f32(dt) * sa * b_sc + za * b_sc * b_qs;
                // DIAG: per-K-block trace at one target cell (dbg.m, dbg.n);
                // skipped when dbg.enable == 0.
                if (dbg.enabled != 0u
                    && dbg.m == bm0 + base_a + a_idx
                    && dbg.n == bn0 + base_b + {j}u) {{
                    dbg_out[t * 8u + 0u] = sa;
                    dbg_out[t * 8u + 1u] = za;
                    dbg_out[t * 8u + 2u] = b_sc;
                    dbg_out[t * 8u + 3u] = b_qs;
                    dbg_out[t * 8u + 4u] = f32(dt);
                    dbg_out[t * 8u + 5u] = f32(dt) * sa * b_sc;
                    dbg_out[t * 8u + 6u] = za * b_sc * b_qs;
                    dbg_out[t * 8u + 7u] = acc_{j};
                }}
            }}
"#
    )
}

/// Build the DP4A matmul WGSL.
pub fn build_wgsl(c: &MatMulI8Config) -> String {
    c.validate();
    let tile = c.tile;
    let threads = c.threads();
    // One K=32 sub-block per outer step = 8 u32 words = 2 vec4<u32> per row.
    const KV: u32 = 2;
    let mut decl_acc = String::new();
    let mut bcast_cols = String::new();
    let mut shuffle_cols = String::new();
    let mut write = String::new();
    for j in 0..16 {
        decl_acc += &format!("    var acc_{j}: f32 = 0.0;\n");
        bcast_cols += &col_block(j, false);
        shuffle_cols += &col_block(j, true);
    }
    for j in (0..16).step_by(2) {
        let j1 = j + 1;
        write += &format!(
            r#"        {{
            let col: u32 = bn0 + base_b + {j}u;
            if (col < d.n) {{
                let widx = (row * d.n + col) >> 1u;
                let lo = clamp(acc_{j},  -65504.0, 65504.0);
                let hi = clamp(acc_{j1}, -65504.0, 65504.0);
                out[widx] = vec2<f16>(vec2<f32>(lo, hi));
            }}
        }}
"#
        );
    }
    // Lane-ownership setup + per-column blocks. The shuffle path reads this
    // lane's own B column into registers once, then every lane fetches the 16
    // columns from the owning lanes. Requires the 16 lanes of a subtile to be
    // 16 consecutive subgroup lanes (holds for power-of-2 sg sizes >= 16 with
    // the linear local-index -> lane mapping every adapter uses; smaller
    // sizes take the broadcast branch).
    let compute = if c.use_subgroup {
        format!(
            r#"        if (sg_size >= 16u) {{
            let lane16: u32 = sg_id % 16u;
            let sg_base: u32 = sg_id - lane16;
            let own_b0: vec4<u32> = tile_b[0][base_b + lane16];
            let own_b1: vec4<u32> = tile_b[1][base_b + lane16];
            let own_sc: f32 = tile_b_scale[base_b + lane16];
            let own_qs: f32 = tile_b_qsum[base_b + lane16];
{shuffle_cols}        }} else {{
{bcast_cols}        }}
"#
        )
    } else {
        bcast_cols
    };
    let extra_builtins = if c.use_subgroup {
        "    @builtin(subgroup_size) sg_size: u32,\n\
         \x20   @builtin(subgroup_invocation_id) sg_id: u32,"
    } else {
        ""
    };
    // Shared-mem budget: tile_a + tile_b (2 vec4 per row each) + params.
    let total_shared = tile * KV * 16 * 2 + tile * 8 + tile * 4 + tile * 4;
    const MAX_WORKGROUP_STORAGE: u32 = 16384;
    assert!(
        total_shared <= MAX_WORKGROUP_STORAGE,
        "matmul_i8 exceeds workgroup storage: total={total_shared} > {MAX_WORKGROUP_STORAGE}"
    );
    format!(
        r#"enable f16;

struct Dims {{ m: u32, n: u32, k: u32, _pad: u32 }};
struct Dbg  {{ m: u32, n: u32, enabled: u32, _pad: u32 }};

@group(0) @binding(0) var<storage, read> a: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> a_params: array<vec2<f16>>;
@group(0) @binding(2) var<storage, read> b: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> b_scale: array<f32>;
@group(0) @binding(4) var<storage, read> b_qsum: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<vec2<f16>>;
@group(0) @binding(6) var<uniform> d: Dims;
@group(0) @binding(7) var<storage, read_write> dbg_out: array<f32>;
@group(0) @binding(8) var<uniform> dbg: Dbg;

const TILE: u32 = {tile}u;
const KV: u32 = {KV}u;
const SUBS: u32 = {tile}u / 16u;
const THREADS: u32 = {threads}u;

// One K=32 slice per outer step: [kv word][row], vec4<u32> = 16 packed i8.
// Compute-phase A reads are 16 consecutive vec4s per subtile (conflict-free
// wide loads); B reads are single-address (broadcast) or shuffled.
var<workgroup> tile_a: array<array<vec4<u32>, {tile}u>, {KV}u>;
var<workgroup> tile_b: array<array<vec4<u32>, {tile}u>, {KV}u>;
// (scale, zero) loaded as vec2<f32> after f16->f32 conversion at load time.
// f32 in shared mem keeps subgroup ops free of f16-subgroup features.
var<workgroup> tile_a_params: array<vec2<f32>, {tile}u>;
var<workgroup> tile_b_scale: array<f32, {tile}u>;
var<workgroup> tile_b_qsum: array<f32, {tile}u>;

@compute @workgroup_size({threads}, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
{extra_builtins}
) {{
    let bm0: u32 = wid.y * TILE;
    let bn0: u32 = wid.x * TILE;
    // DIAG probe: tid 0 of EVERY workgroup writes its (wid.x, wid.y) into the
    // 16-slot probe area right after the (d.k/32)*8 trace floats. Last-writer
    // wins; non-zero values prove ANY workgroup reached this write. Also
    // echoes dbg.m / dbg.n / dbg.enabled as f32 to verify the uniform reaches
    // the kernel. dbg_out[probe0+15] = sentinel 1234.0. Writes are discarded
    // (OOB) for dispatches whose dbg_out is the 4-byte placeholder.
    let probe0: u32 = (d.k / 32u) * 8u;
    if (tid == 0u) {{
        dbg_out[probe0 + 0u] = f32(wid.x);
        dbg_out[probe0 + 1u] = f32(wid.y);
        dbg_out[probe0 + 2u] = f32(dbg.enabled);
        dbg_out[probe0 + 3u] = f32(dbg.m);
        dbg_out[probe0 + 4u] = f32(dbg.n);
    }}
    if (tid == 0u && wid.x == 0u && wid.y == 0u) {{
        dbg_out[probe0 + 15u] = 1234.0;
    }}
    // 16x16 subtiles in a SUBS x SUBS grid; 16 lanes per subtile, each lane
    // owns one A row (a_idx) and 16 B columns (base_b .. base_b+15).
    let subtile_id: u32 = tid / 16u;
    let base_a: u32 = (subtile_id / SUBS) * 16u;
    let base_b: u32 = (subtile_id % SUBS) * 16u;
    let a_idx: u32 = tid % 16u;
{decl_acc}
    let n_blocks: u32 = d.k / 32u;
    let v4_per_row: u32 = d.k / 16u;
    let scales_per_row: u32 = d.k / 32u;

    for (var t: u32 = 0u; t < n_blocks; t = t + 1u) {{
        // Load phase: TILE rows x KV vec4 for each of A and B (vec4 global
        // loads, 16B coalesced). At tile=64 this is exactly one item per
        // thread (128 A-loads + 128 B-loads across 256 threads).
        for (var li: u32 = tid; li < TILE * KV * 2u; li = li + THREADS) {{
            let which: u32 = li / (TILE * KV);
            let rem: u32 = li % (TILE * KV);
            let row: u32 = rem / KV;
            let kv: u32 = rem % KV;
            if (which == 0u) {{
                let gr: u32 = bm0 + row;
                var v: vec4<u32> = vec4<u32>(0u);
                if (gr < d.m) {{
                    v = a[gr * v4_per_row + t * KV + kv];
                }}
                tile_a[kv][row] = v;
            }} else {{
                let gc: u32 = bn0 + row;
                var v: vec4<u32> = vec4<u32>(0u);
                if (gc < d.n) {{
                    v = b[gc * v4_per_row + t * KV + kv];
                }}
                tile_b[kv][row] = v;
            }}
        }}
        // Params: first TILE threads load a-params (vec2<f16> -> vec2<f32>),
        // b-scale and b-qsum for their row/column.
        if (tid < TILE) {{
            let gr: u32 = bm0 + tid;
            var ap: vec2<f32> = vec2<f32>(0.0, 0.0);
            if (gr < d.m) {{
                ap = vec2<f32>(a_params[gr * scales_per_row + t]);
            }}
            tile_a_params[tid] = ap;
            let gc: u32 = bn0 + tid;
            var sc: f32 = 0.0;
            var qs: f32 = 0.0;
            if (gc < d.n) {{
                sc = b_scale[gc * scales_per_row + t];
                qs = b_qsum[gc * scales_per_row + t];
            }}
            tile_b_scale[tid] = sc;
            tile_b_qsum[tid] = qs;
        }}
        workgroupBarrier();

        // Compute phase: the lane's A row stays in registers across all 16
        // column steps; the scale fold runs fused per column per K-block.
        let own_a0: vec4<u32> = tile_a[0][base_a + a_idx];
        let own_a1: vec4<u32> = tile_a[1][base_a + a_idx];
        let a_p: vec2<f32> = tile_a_params[base_a + a_idx];
        let sa: f32 = a_p.x;
        let za: f32 = a_p.y;
{compute}
        workgroupBarrier();
    }}

    // Output write: paired vec2<f16>, columns (j, j+1) into one word.
    let row: u32 = bm0 + base_a + a_idx;
    if (row < d.m) {{
{write}    }}
}}
"#
    )
}

pub struct MatMulI8Bufs<'a> {
    pub a: &'a BufRef,
    pub a_params: &'a BufRef,
    pub b: &'a BufRef,
    pub b_scale: &'a BufRef,
    pub b_qsum: &'a BufRef,
    pub out: &'a BufRef,
    pub dims: &'a BufRef,
    /// Storage RW scratch for the diag trace. When `dbg.enable == 0` the
    /// kernel skips all writes and a 1-element scratch is sufficient. When
    /// enabled, size must be `>= (k / 32) * 8` f32s.
    pub dbg_out: &'a BufRef,
    /// Uniform { m: u32, n: u32, enable: u32, _pad: u32 }. `enable == 0`
    /// disables the trace path inside the kernel.
    pub dbg: &'a BufRef,
}

pub fn workgroups(c: &MatMulI8Config, m: u32, n: u32) -> [u32; 3] {
    [n.div_ceil(c.tile), m.div_ceil(c.tile), 1]
}

pub fn dispatch_matmul_i8<B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    cfg: &MatMulI8Config,
    bufs: &MatMulI8Bufs<'_>,
    m: u32,
    n: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.a.binding(0),
        bufs.a_params.binding(1),
        bufs.b.binding(2),
        bufs.b_scale.binding(3),
        bufs.b_qsum.binding(4),
        bufs.out.binding(5),
        bufs.dims.binding(6),
        bufs.dbg_out.binding(7),
        bufs.dbg.binding(8),
    ];
    backend.dispatch(encoder, pipeline, &bindings, workgroups(cfg, m, n))
}
