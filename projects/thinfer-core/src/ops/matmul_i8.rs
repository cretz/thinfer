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
//! The main term still uses `dot4I8Packed` (DP4A); the correction adds one
//! extra fma per K-block per (m, n) cell — negligible vs the DP4A inner loop.
//!
//! Per outer K-step (one K=32 sub-block):
//!   dot_i32 = Σ_s dot4I8Packed(tile_a[m, s], tile_b[n, s])
//!   acc_f32 += f32(dot_i32) * sa * sb + za * sb * qsum_b

use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};

/// Tile shape. `BK` is fixed at 32 (one K=32 sub-block per outer step), so
/// only `bm` / `bn` / `tm` / `tn` are tunable.
///
/// `use_subgroup` enables subgroup-aware reads on the per-row `tile_a` /
/// `tile_a_params` values. Within a subgroup the value loaded for `a_row` is
/// the same across every lane that shares `lid_y`; the hint lets the backend
/// collapse N per-lane shared-mem reads into one fetch + broadcast.
///
/// Two cases, picked at runtime via `@builtin(subgroup_size)`:
/// * `subgroup_size <= wg_x`: the subgroup is contained in a single
///   WG-row, so every lane shares `lid_y` -> `subgroupBroadcastFirst` is
///   correct (value is subgroup-uniform).
/// * `subgroup_size >= wg_x`: the subgroup spans multiple rows. The lane
///   id of THIS thread's row-leader (lid_x=0, same lid_y) within the
///   subgroup is `subgroup_invocation_id - lid_x`. `subgroupShuffle` from
///   that lane reads the row-leader's value (the correct `a_row`).
///
/// Both sizes are powers of 2 and wg_x is a power of 2 in our default
/// configs, so the divisibility relationship always holds in practice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatMulI8Config {
    pub bm: u32,
    pub bn: u32,
    pub tm: u32,
    pub tn: u32,
    pub use_subgroup: bool,
}

impl MatMulI8Config {
    pub const DEFAULT: Self = Self {
        bm: 64,
        bn: 64,
        tm: 8,
        tn: 4,
        use_subgroup: false,
    };

    pub const fn wg_x(&self) -> u32 {
        self.bn / self.tn
    }
    pub const fn wg_y(&self) -> u32 {
        self.bm / self.tm
    }
    pub const fn threads(&self) -> u32 {
        self.wg_x() * self.wg_y()
    }

    fn validate(&self) {
        assert!(self.bm > 0 && self.bn > 0 && self.tm > 0 && self.tn > 0);
        assert!(
            self.bm.is_multiple_of(self.tm),
            "bm must be a multiple of tm"
        );
        assert!(
            self.bn.is_multiple_of(self.tn),
            "bn must be a multiple of tn"
        );
        assert!(
            self.tn.is_multiple_of(2),
            "tn must be even (paired f16 writes)"
        );
        assert!(
            self.bn.is_multiple_of(2),
            "bn must be even (paired f16 writes)"
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
    format!(
        "matmul_i8-bm{}_bn{}_tm{}_tn{}{}",
        c.bm, c.bn, c.tm, c.tn, sg
    )
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

/// Build the DP4A matmul WGSL.
pub fn build_wgsl(c: &MatMulI8Config) -> String {
    c.validate();
    let bm = c.bm;
    let bn = c.bn;
    let tm = c.tm;
    let tn = c.tn;
    let wg_x = c.wg_x();
    let wg_y = c.wg_y();
    let threads = c.threads();
    const BK: u32 = 32;
    const BK_U32: u32 = BK / 4;
    const BK_V4: u32 = BK_U32 / 4;
    let tile_a_u32 = bm * BK_U32;
    let tile_b_u32 = bn * BK_U32;
    // Global A/B loads are vec4<u32> (16B coalesced): one K=32 sub-block row
    // is BK_V4=2 vec4s, so a vec4 never straddles rows. K%32==0 guarantees
    // 16B-divisible binding sizes; storage offsets are 256-aligned.
    let a_loads_per_thread = (tile_a_u32 / 4).div_ceil(threads);
    let b_loads_per_thread = (tile_b_u32 / 4).div_ceil(threads);
    // Subgroup-aware loads of the row-uniform values (av and a_p only depend
    // on a_row, which is uniform across all lanes that share lid_y). Branch
    // on the runtime `subgroup_size` builtin; both branches are correct.
    let (extra_builtins, a_p_load) = if c.use_subgroup {
        (
            "    @builtin(subgroup_size) sg_size: u32,\n\
             \x20   @builtin(subgroup_invocation_id) sg_id: u32,",
            r#"let a_p: vec2<f32> = select(
                subgroupShuffle(tile_a_params[a_row], sg_id - lid_x),
                subgroupBroadcastFirst(tile_a_params[a_row]),
                sg_size <= WG_X
            );"#,
        )
    } else {
        ("", "let a_p: vec2<f32> = tile_a_params[a_row];")
    };
    // Per-row A word for K-word `s`, subgroup-aware (row-uniform across
    // lanes sharing lid_y; see `use_subgroup` doc above).
    let a_word = |i: u32| {
        if c.use_subgroup {
            format!(
                "select(subgroupShuffle(tile_a[(lid_y * TM + {i}u) * BK_U32 + s], \
                 sg_id - lid_x), subgroupBroadcastFirst(tile_a[(lid_y * TM + {i}u) * \
                 BK_U32 + s]), sg_size <= WG_X)"
            )
        } else {
            format!("tile_a[(lid_y * TM + {i}u) * BK_U32 + s]")
        }
    };
    // Fully unrolled register blocking: dots/acc as individual scalars.
    // Dynamically-indexed local arrays (`acc[i * TN + j]`) are not reliably
    // promoted to registers by naga/drivers - the array form measured ~1.3
    // TFLOPS effective on a 5070 (same local-memory spill failure mode the
    // sdpa kernel had with its array<f32, 128> row state).
    let mut decl_acc = String::new();
    let mut decl_dots = String::new();
    let mut inner = String::new();
    let mut fold = String::new();
    let mut write = String::new();
    for i in 0..tm {
        for j in 0..tn {
            decl_acc += &format!("    var acc_{i}_{j}: f32 = 0.0;\n");
            decl_dots += &format!("        var d_{i}_{j}: i32 = 0;\n");
        }
    }
    for i in 0..tm {
        inner += &format!("            let a{i}: u32 = {};\n", a_word(i));
    }
    for j in 0..tn {
        inner +=
            &format!("            let b{j}: u32 = tile_b[(lid_x * TN + {j}u) * BK_U32 + s];\n");
    }
    for i in 0..tm {
        for j in 0..tn {
            inner += &format!("            d_{i}_{j} = d_{i}_{j} + dot4I8Packed(a{i}, b{j});\n");
        }
    }
    for i in 0..tm {
        fold += &format!(
            "        {{\n            let a_row: u32 = lid_y * TM + {i}u;\n            {a_p_load}\n            \
             let sa: f32 = a_p.x;\n            let za: f32 = a_p.y;\n"
        );
        for j in 0..tn {
            fold += &format!(
                r#"            {{
                let b_col: u32 = lid_x * TN + {j}u;
                let b_sc: f32 = tile_b_scale[b_col];
                let b_qs: f32 = tile_b_qsum[b_col];
                acc_{i}_{j} = acc_{i}_{j} + f32(d_{i}_{j}) * sa * b_sc + za * b_sc * b_qs;
                // DIAG: per-K-block trace at one target cell (dbg.m, dbg.n);
                // skipped when dbg.enable == 0.
                if (dbg.enabled != 0u
                    && wid.y * BM <= dbg.m && dbg.m < wid.y * BM + BM
                    && wid.x * BN <= dbg.n && dbg.n < wid.x * BN + BN) {{
                    let tgt_local_m: u32 = dbg.m - wid.y * BM;
                    let tgt_local_n: u32 = dbg.n - wid.x * BN;
                    if (lid_y == tgt_local_m / TM
                        && lid_x == tgt_local_n / TN
                        && {i}u == tgt_local_m % TM
                        && {j}u == tgt_local_n % TN) {{
                        dbg_out[t * 8u + 0u] = sa;
                        dbg_out[t * 8u + 1u] = za;
                        dbg_out[t * 8u + 2u] = b_sc;
                        dbg_out[t * 8u + 3u] = b_qs;
                        dbg_out[t * 8u + 4u] = f32(d_{i}_{j});
                        dbg_out[t * 8u + 5u] = f32(d_{i}_{j}) * sa * b_sc;
                        dbg_out[t * 8u + 6u] = za * b_sc * b_qs;
                        dbg_out[t * 8u + 7u] = acc_{i}_{j};
                    }}
                }}
            }}
"#
            );
        }
        fold += "        }\n";
    }
    for i in 0..tm {
        write += &format!(
            "    {{\n        let row: u32 = bm0 + lid_y * TM + {i}u;\n        if (row < d.m) {{\n"
        );
        for j in (0..tn).step_by(2) {
            let j1 = j + 1;
            write += &format!(
                r#"            {{
                let col: u32 = bn0 + lid_x * TN + {j}u;
                if (col < d.n) {{
                    let widx = (row * d.n + col) >> 1u;
                    let lo = clamp(acc_{i}_{j},  -65504.0, 65504.0);
                    let hi = clamp(acc_{i}_{j1}, -65504.0, 65504.0);
                    out[widx] = vec2<f16>(vec2<f32>(lo, hi));
                }}
            }}
"#
            );
        }
        write += "        }\n    }\n";
    }
    // Shared-mem budget: tile_a + tile_b + tile_a_params + tile_b_scale + tile_b_qsum.
    let total_shared = (tile_a_u32 + tile_b_u32) * 4 + bm * 8 + bn * 4 + bn * 4;
    const MAX_WORKGROUP_STORAGE: u32 = 32768;
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

const BM: u32 = {bm}u;
const BN: u32 = {bn}u;
const BK: u32 = {BK}u;
const BK_U32: u32 = {BK_U32}u;
const BK_V4: u32 = {BK_V4}u;
const TM: u32 = {tm}u;
const TN: u32 = {tn}u;
const WG_X: u32 = {wg_x}u;
const WG_Y: u32 = {wg_y}u;
const THREADS: u32 = {threads}u;

var<workgroup> tile_a: array<u32, {tile_a_u32}u>;
var<workgroup> tile_b: array<u32, {tile_b_u32}u>;
// (scale, zero) loaded as vec2<f32> after f16->f32 conversion at load time.
// f32 in shared mem keeps subgroup ops free of f16-subgroup features.
var<workgroup> tile_a_params: array<vec2<f32>, {bm}u>;
var<workgroup> tile_b_scale: array<f32, {bn}u>;
var<workgroup> tile_b_qsum: array<f32, {bn}u>;

@compute @workgroup_size({threads}, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
{extra_builtins}
) {{
    let lid_x: u32 = tid % WG_X;
    let lid_y: u32 = tid / WG_X;
    let bm0: u32 = wid.y * BM;
    let bn0: u32 = wid.x * BN;
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
{decl_acc}
    let n_blocks: u32 = d.k / BK;
    let v4_per_row: u32 = d.k / 16u;
    let scales_per_row: u32 = d.k / BK;

    for (var t: u32 = 0u; t < n_blocks; t = t + 1u) {{
        // Cooperative tile_a load: BM rows x BK_V4 vec4<u32> (16B coalesced).
        for (var s: u32 = 0u; s < {a_loads_per_thread}u; s = s + 1u) {{
            let idx: u32 = s * THREADS + tid;
            if (idx < BM * BK_V4) {{
                let ar: u32 = idx / BK_V4;
                let ak: u32 = idx % BK_V4;
                let gr: u32 = bm0 + ar;
                var v: vec4<u32> = vec4<u32>(0u);
                if (gr < d.m) {{
                    v = a[gr * v4_per_row + t * BK_V4 + ak];
                }}
                let base: u32 = ar * BK_U32 + ak * 4u;
                tile_a[base] = v.x;
                tile_a[base + 1u] = v.y;
                tile_a[base + 2u] = v.z;
                tile_a[base + 3u] = v.w;
            }}
        }}
        // Cooperative tile_b load: BN cols x BK_V4 vec4<u32> (N-major).
        for (var s: u32 = 0u; s < {b_loads_per_thread}u; s = s + 1u) {{
            let idx: u32 = s * THREADS + tid;
            if (idx < BN * BK_V4) {{
                let bc: u32 = idx / BK_V4;
                let bk: u32 = idx % BK_V4;
                let gc: u32 = bn0 + bc;
                var v: vec4<u32> = vec4<u32>(0u);
                if (gc < d.n) {{
                    v = b[gc * v4_per_row + t * BK_V4 + bk];
                }}
                let base: u32 = bc * BK_U32 + bk * 4u;
                tile_b[base] = v.x;
                tile_b[base + 1u] = v.y;
                tile_b[base + 2u] = v.z;
                tile_b[base + 3u] = v.w;
            }}
        }}
        // Scales/params: BM a-params + BN b-scales + BN b-qsums. First BM
        // threads load a-params (vec2<f16> -> vec2<f32>); first BN threads
        // load b-scale and b-qsum. Idle threads beyond max(BM, BN) skip.
        if (tid < BM) {{
            let gr: u32 = bm0 + tid;
            var v: vec2<f32> = vec2<f32>(0.0, 0.0);
            if (gr < d.m) {{
                v = vec2<f32>(a_params[gr * scales_per_row + t]);
            }}
            tile_a_params[tid] = v;
        }}
        if (tid < BN) {{
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

        // DP4A inner loop, register-blocked and fully unrolled: s is the
        // outer loop; each step pulls TM a-words and TN b-words into
        // registers and runs TM*TN dot4I8Packed on them.
{decl_dots}
        for (var s: u32 = 0u; s < BK_U32; s = s + 1u) {{
{inner}
        }}
        // Scale fold. Asymmetric: acc += dot_i32 * sa * sb + za * sb * qsum_b.
{fold}
        workgroupBarrier();
    }}

    // Output write: paired vec2<f16>, columns (j, j+1) into one word.
{write}
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
    [n.div_ceil(c.bn), m.div_ceil(c.bm), 1]
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
