//! DP4A matmul: A and B as `[M, K]` / `[N, K]` packed int8 + per-K=32 scale
//! buffers, output as `array<vec2<f16>>` `[M, N]` paired.
//!
//! The inner loop uses the WGSL `packed_4x8_integer_dot_product` extension
//! (`dot4I8Packed`) which naga 28 lowers to SPIR-V `SDot` (Vulkan, backed by
//! `VK_KHR_shader_integer_dot_product` on hw that supports it), HLSL `dot4add_i8packed`
//! on D3D12, or `packed_char4` on Metal. Each call computes
//! `sum_{i=0..3} (sext_i8(a, i) * sext_i8(b, i))` in i32 from two u32-packed
//! operands.
//!
//! Per outer K-step (one K=32 sub-block):
//!   dot_i32 = sum_{s=0..BK_U32-1} dot4I8Packed(tile_a[m, s], tile_b[n, s])
//!   acc_f32 += f32(dot_i32) * a_scale[m, t] * b_scale[n, t]
//!
//! `BK` is pinned to 32 (one Q-block sub-block per outer step). The kernel is
//! lean: a single integer fast path, no f16/bf16 branches inside.

use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};

/// Tile shape. `BK` is fixed at 32 (one K=32 sub-block per outer step), so
/// only `bm` / `bn` / `tm` / `tn` are tunable.
///
/// `use_subgroup` enables subgroup-aware reads on the per-row `tile_a` /
/// `tile_a_scale` values. Within a subgroup the value loaded for `a_row`
/// is the same across every lane that shares `lid_y`; the hint lets the
/// backend collapse N per-lane shared-mem reads into one fetch + broadcast.
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
/// Gated by `Features::SUBGROUP` at the device level; numerically identical
/// to the non-subgroup path.
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
        tm: 4,
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
        // Output is paired vec2<f16>: each thread writes its TN cols in pairs.
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
        kind: BindingKind::StorageReadWrite,
    },
    BindingLayout {
        slot: 5,
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
    let tile_a_u32 = bm * BK_U32;
    let tile_b_u32 = bn * BK_U32;
    let a_loads_per_thread = tile_a_u32.div_ceil(threads);
    let b_loads_per_thread = tile_b_u32.div_ceil(threads);
    let acc_size = tm * tn;
    // Subgroup-aware loads of the row-uniform values (av and a_sc only
    // depend on a_row, which is uniform across all lanes that share lid_y).
    // Universal shape: branch on the runtime `subgroup_size` builtin. Both
    // branches are correct; on devices where min==max the compiler folds
    // the branch. No `enable subgroups;` directive (naga 28 treats that as
    // Unimplemented; the builtins themselves are gated by the device's
    // `Features::SUBGROUP`).
    let (extra_builtins, a_sc_load, av_load) = if c.use_subgroup {
        (
            "    @builtin(subgroup_size) sg_size: u32,\n\
             \x20   @builtin(subgroup_invocation_id) sg_id: u32,",
            r#"let a_sc: f32 = select(
                subgroupShuffle(tile_a_scale[a_row], sg_id - lid_x),
                subgroupBroadcastFirst(tile_a_scale[a_row]),
                sg_size <= WG_X
            );"#,
            r#"av_arr[s] = select(
                subgroupShuffle(tile_a[a_row * BK_U32 + s], sg_id - lid_x),
                subgroupBroadcastFirst(tile_a[a_row * BK_U32 + s]),
                sg_size <= WG_X
            );"#,
        )
    } else {
        (
            "",
            "let a_sc: f32 = tile_a_scale[a_row];",
            "av_arr[s] = tile_a[a_row * BK_U32 + s];",
        )
    };
    // Shared-mem budget check matches the F32 matmul: 32 KiB hard cap.
    let total_shared = (tile_a_u32 + tile_b_u32) * 4 + (bm + bn) * 4;
    const MAX_WORKGROUP_STORAGE: u32 = 32768;
    assert!(
        total_shared <= MAX_WORKGROUP_STORAGE,
        "matmul_i8 exceeds workgroup storage: tile_a={} tile_b={} scales={} total={} > {}",
        tile_a_u32 * 4,
        tile_b_u32 * 4,
        (bm + bn) * 4,
        total_shared,
        MAX_WORKGROUP_STORAGE,
    );
    format!(
        r#"enable f16;

struct Dims {{ m: u32, n: u32, k: u32, _pad: u32 }};

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> a_scale: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<u32>;
@group(0) @binding(3) var<storage, read> b_scale: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<vec2<f16>>;
@group(0) @binding(5) var<uniform> d: Dims;

const BM: u32 = {bm}u;
const BN: u32 = {bn}u;
const BK: u32 = {BK}u;
const BK_U32: u32 = {BK_U32}u;
const TM: u32 = {tm}u;
const TN: u32 = {tn}u;
const WG_X: u32 = {wg_x}u;
const WG_Y: u32 = {wg_y}u;
const THREADS: u32 = {threads}u;

var<workgroup> tile_a: array<u32, {tile_a_u32}u>;
var<workgroup> tile_b: array<u32, {tile_b_u32}u>;
var<workgroup> tile_a_scale: array<f32, {bm}u>;
var<workgroup> tile_b_scale: array<f32, {bn}u>;

// Flat 1D workgroup. naga 28 rejects `@builtin(subgroup_invocation_id)`
// (and `subgroup_id`) on multi-dimensional workgroups with
// `InvalidMultiDimensionalSubgroupBuiltIn`. (subgroup_size alone IS allowed
// on multi-dim, but we need invocation_id for the shuffle path.) Threads
// are addressed by `local_invocation_index` (= `tid`); the original 2D
// (lid_x, lid_y) layout is reconstructed locally as (lid_x, lid_y) =
// (tid % WG_X, tid / WG_X). Subgroups are still partitioned in tid order,
// so the row-leader-lane formula `sg_id - lid_x` is unchanged.
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
    var acc: array<f32, {acc_size}u>;
    for (var i: u32 = 0u; i < {acc_size}u; i = i + 1u) {{
        acc[i] = 0.0;
    }}
    let n_blocks: u32 = d.k / BK;
    let words_per_row: u32 = d.k / 4u;
    let scales_per_row: u32 = d.k / BK;

    for (var t: u32 = 0u; t < n_blocks; t = t + 1u) {{
        // Cooperative tile_a load: BM rows x BK_U32 u32 words.
        for (var s: u32 = 0u; s < {a_loads_per_thread}u; s = s + 1u) {{
            let idx: u32 = s * THREADS + tid;
            if (idx < BM * BK_U32) {{
                let ar: u32 = idx / BK_U32;
                let ak: u32 = idx % BK_U32;
                let gr: u32 = bm0 + ar;
                var v: u32 = 0u;
                if (gr < d.m) {{
                    v = a[gr * words_per_row + t * BK_U32 + ak];
                }}
                tile_a[ar * BK_U32 + ak] = v;
            }}
        }}
        // Cooperative tile_b load: BN cols x BK_U32 u32 words (N-major).
        for (var s: u32 = 0u; s < {b_loads_per_thread}u; s = s + 1u) {{
            let idx: u32 = s * THREADS + tid;
            if (idx < BN * BK_U32) {{
                let bc: u32 = idx / BK_U32;
                let bk: u32 = idx % BK_U32;
                let gc: u32 = bn0 + bc;
                var v: u32 = 0u;
                if (gc < d.n) {{
                    v = b[gc * words_per_row + t * BK_U32 + bk];
                }}
                tile_b[bc * BK_U32 + bk] = v;
            }}
        }}
        // Scales: BM + BN f32 values. First BM threads load a_scales, first
        // BN threads load b_scales (independent arrays, no race even when
        // BM == BN). Idle threads beyond max(BM, BN) skip.
        if (tid < BM) {{
            let gr: u32 = bm0 + tid;
            var v: f32 = 0.0;
            if (gr < d.m) {{
                v = a_scale[gr * scales_per_row + t];
            }}
            tile_a_scale[tid] = v;
        }}
        if (tid < BN) {{
            let gc: u32 = bn0 + tid;
            var v: f32 = 0.0;
            if (gc < d.n) {{
                v = b_scale[gc * scales_per_row + t];
            }}
            tile_b_scale[tid] = v;
        }}
        workgroupBarrier();

        // DP4A inner loop. Each thread accumulates TM*TN output cells.
        // Loop order: i (TM) -> s (BK_U32) -> j (TN). The `av` shared-mem read
        // depends only on (a_row, s), so hoisting it out of the j-loop cuts
        // shared-mem reads from 2*TM*TN*BK_U32 (= 256 for default cfg) to
        // (TM*BK_U32 + TM*TN*BK_U32) (= 160). Per-j dot products accumulate
        // into a register-resident `dots` array, finalized once after the s
        // loop with the per-(row, col) scale product.
        //
        // When `use_subgroup` is set, `subgroupBroadcastFirst` marks `av` /
        // `a_sc` as subgroup-uniform. With the wgpu row-major lane layout
        // (lid_x varies fastest), lanes that share `lid_y` share `a_row` and
        // thus the loaded values; the hint lets the backend lower these
        // reads to a single fetch + broadcast across the subgroup. Numerically
        // identical to the non-subgroup branch.
        for (var i: u32 = 0u; i < TM; i = i + 1u) {{
            let a_row: u32 = lid_y * TM + i;
            {a_sc_load}
            // Load all BK_U32 av words for this row into registers.
            var av_arr: array<u32, {BK_U32}u>;
            for (var s: u32 = 0u; s < BK_U32; s = s + 1u) {{
                {av_load}
            }}
            // Per-(row, col) integer accumulators, finalized after the s loop.
            var dots: array<i32, {tn}u>;
            for (var j: u32 = 0u; j < TN; j = j + 1u) {{
                dots[j] = 0;
            }}
            for (var s: u32 = 0u; s < BK_U32; s = s + 1u) {{
                let av: u32 = av_arr[s];
                for (var j: u32 = 0u; j < TN; j = j + 1u) {{
                    let b_col: u32 = lid_x * TN + j;
                    let bv: u32 = tile_b[b_col * BK_U32 + s];
                    dots[j] = dots[j] + dot4I8Packed(av, bv);
                }}
            }}
            for (var j: u32 = 0u; j < TN; j = j + 1u) {{
                let b_col: u32 = lid_x * TN + j;
                let b_sc: f32 = tile_b_scale[b_col];
                acc[i * TN + j] = acc[i * TN + j] + f32(dots[j]) * a_sc * b_sc;
            }}
        }}
        workgroupBarrier();
    }}

    // Output write: paired vec2<f16>, columns (j, j+1) into one word.
    // Saturated narrow at +-65504 to avoid +-inf on any pathological scale
    // (mirrors the F16 matmul output path).
    for (var i: u32 = 0u; i < TM; i = i + 1u) {{
        let row: u32 = bm0 + lid_y * TM + i;
        if (row >= d.m) {{ continue; }}
        for (var j: u32 = 0u; j < TN; j = j + 2u) {{
            let col: u32 = bn0 + lid_x * TN + j;
            if (col >= d.n) {{ continue; }}
            let widx = (row * d.n + col) >> 1u;
            let lo = clamp(acc[i * TN + j],       -65504.0, 65504.0);
            let hi = clamp(acc[i * TN + j + 1u], -65504.0, 65504.0);
            out[widx] = vec2<f16>(vec2<f32>(lo, hi));
        }}
    }}
}}
"#
    )
}

pub struct MatMulI8Bufs<'a> {
    pub a: &'a BufRef,
    pub a_scale: &'a BufRef,
    pub b: &'a BufRef,
    pub b_scale: &'a BufRef,
    pub out: &'a BufRef,
    pub dims: &'a BufRef,
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
        bufs.a_scale.binding(1),
        bufs.b.binding(2),
        bufs.b_scale.binding(3),
        bufs.out.binding(4),
        bufs.dims.binding(5),
    ];
    backend.dispatch(encoder, pipeline, &bindings, workgroups(cfg, m, n))
}
