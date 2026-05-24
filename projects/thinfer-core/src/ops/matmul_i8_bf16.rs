//! Mixed-precision matmul: A is paired packed-i8 with **asymmetric** per-K=32
//! params `(scale, zero)` (llama.cpp Q8_1-style); B is dense bf16 weights
//! stored K-major as `array<u32>` (2 bf16 elements per word). Output is
//! `[M, N]` paired `vec2<f16>`.
//!
//! Asymmetric decomposition (B is symmetric — no zero point on weights):
//!   `O = Σ (qa*sa + za) * b`
//!     = `Σ_{k_block} sa[m,t] * Σ_k qa*b   +   za[m,t] * Σ_k b`
//!                                    main path                 correction
//!
//! `b_sum[n, k_block] = Σ_{k in block} b[n, k]` (f32) is precomputed at
//! weight-load time (host-side scan of the bf16 weight) and passed in as a
//! new binding. Adds one fma per K-block per (m, n) cell on top of the
//! existing inner loop.
//!
//! Used by the I8 acts × bf16 weight surfaces of the DiT block: the noise
//! and context refiners' qkv / proj / ffn_up / ffn_gate / ffn_down sites
//! (refiner weights are never quantized in the GGUF path).

use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};

/// Tile shape. `BK` is fixed at 32 (one K=32 sub-block per outer step), so
/// only `bm` / `bn` / `tm` / `tn` are tunable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatMulI8Bf16Config {
    pub bm: u32,
    pub bn: u32,
    pub tm: u32,
    pub tn: u32,
}

impl MatMulI8Bf16Config {
    pub const DEFAULT: Self = Self {
        bm: 64,
        bn: 64,
        tm: 4,
        tn: 4,
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

impl Default for MatMulI8Bf16Config {
    fn default() -> Self {
        Self::DEFAULT
    }
}

pub fn hint(c: &MatMulI8Bf16Config) -> String {
    format!("matmul_i8_bf16-bm{}_bn{}_tm{}_tn{}", c.bm, c.bn, c.tm, c.tn)
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

/// Build the I8 × bf16 matmul WGSL.
pub fn build_wgsl(c: &MatMulI8Bf16Config) -> String {
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
    let tile_b_size = BK * bn;
    let a_loads_per_thread = tile_a_u32.div_ceil(threads);
    let tile_b_words = (BK * bn) / 2;
    let b_loads_per_thread = tile_b_words.div_ceil(threads);
    let acc_size = tm * tn;
    // Shared-mem budget: tile_a u32 + tile_b f32 + tile_a_params vec2<f32> + tile_b_sum f32.
    let total_shared = tile_a_u32 * 4 + tile_b_size * 4 + bm * 8 + bn * 4;
    const MAX_WORKGROUP_STORAGE: u32 = 32768;
    assert!(
        total_shared <= MAX_WORKGROUP_STORAGE,
        "matmul_i8_bf16 exceeds workgroup storage: total={total_shared} > {MAX_WORKGROUP_STORAGE}"
    );
    format!(
        r#"enable f16;

struct Dims {{ m: u32, n: u32, k: u32, _pad: u32 }};

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> a_params: array<vec2<f16>>;
@group(0) @binding(2) var<storage, read> b: array<u32>;
@group(0) @binding(3) var<storage, read> b_sum: array<f32>;
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
var<workgroup> tile_b: array<f32, {tile_b_size}u>;
var<workgroup> tile_a_params: array<vec2<f32>, {bm}u>;
var<workgroup> tile_b_sum: array<f32, {bn}u>;

fn sext_i8_byte(w: u32, byte_idx: u32) -> i32 {{
    let s: u32 = (3u - byte_idx) * 8u;
    return i32(w << s) >> 24u;
}}

fn unpack_bf16(w: u32, lane: u32) -> f32 {{
    let half: u32 = (w >> (lane * 16u)) & 0xFFFFu;
    return bitcast<f32>(half << 16u);
}}

@compute @workgroup_size({threads}, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
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
    let a_words_per_row: u32 = d.k / 4u;
    let scales_per_a_row: u32 = d.k / BK;

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
                    v = a[gr * a_words_per_row + t * BK_U32 + ak];
                }}
                tile_a[ar * BK_U32 + ak] = v;
            }}
        }}
        // Cooperative tile_b load: (BK*BN)/2 bf16 words.
        for (var s: u32 = 0u; s < {b_loads_per_thread}u; s = s + 1u) {{
            let word_idx: u32 = s * THREADS + tid;
            if (word_idx < (BK * BN) / 2u) {{
                let pair_per_row: u32 = BN / 2u;
                let bk: u32 = word_idx / pair_per_row;
                let bc_pair: u32 = word_idx % pair_per_row;
                let bc0: u32 = bc_pair * 2u;
                let gk: u32 = t * BK + bk;
                let gc0: u32 = bn0 + bc0;
                var w: u32 = 0u;
                if (gk < d.k && gc0 < d.n) {{
                    w = b[(gk * d.n + gc0) >> 1u];
                }}
                let lo = unpack_bf16(w, 0u);
                let hi = unpack_bf16(w, 1u);
                tile_b[bc0 * BK + bk] = lo;
                if (gc0 + 1u < d.n) {{
                    tile_b[(bc0 + 1u) * BK + bk] = hi;
                }}
            }}
        }}
        // a_params: BM (scale, zero) pairs (vec2<f16> -> vec2<f32>).
        if (tid < BM) {{
            let gr: u32 = bm0 + tid;
            var v: vec2<f32> = vec2<f32>(0.0, 0.0);
            if (gr < d.m) {{
                v = vec2<f32>(a_params[gr * scales_per_a_row + t]);
            }}
            tile_a_params[tid] = v;
        }}
        // b_sum: BN per-block sums for this K-block.
        if (tid < BN) {{
            let gc: u32 = bn0 + tid;
            var v: f32 = 0.0;
            if (gc < d.n) {{
                v = b_sum[gc * scales_per_a_row + t];
            }}
            tile_b_sum[tid] = v;
        }}
        workgroupBarrier();

        // Inner compute. For each row: unpack 32 i8 acts into i32 reg array.
        // For each col: p = Σ_k qa[k] * b[k]; acc += p * sa + za * bsum.
        for (var i: u32 = 0u; i < TM; i = i + 1u) {{
            let a_row: u32 = lid_y * TM + i;
            let a_p: vec2<f32> = tile_a_params[a_row];
            let sa: f32 = a_p.x;
            let za: f32 = a_p.y;
            var av: array<i32, {BK}u>;
            for (var s: u32 = 0u; s < BK_U32; s = s + 1u) {{
                let w: u32 = tile_a[a_row * BK_U32 + s];
                av[s * 4u + 0u] = sext_i8_byte(w, 0u);
                av[s * 4u + 1u] = sext_i8_byte(w, 1u);
                av[s * 4u + 2u] = sext_i8_byte(w, 2u);
                av[s * 4u + 3u] = sext_i8_byte(w, 3u);
            }}
            for (var j: u32 = 0u; j < TN; j = j + 1u) {{
                let b_col: u32 = lid_x * TN + j;
                var p: f32 = 0.0;
                for (var k: u32 = 0u; k < BK; k = k + 1u) {{
                    p = p + f32(av[k]) * tile_b[b_col * BK + k];
                }}
                let b_su: f32 = tile_b_sum[b_col];
                acc[i * TN + j] = acc[i * TN + j] + p * sa + za * b_su;
            }}
        }}
        workgroupBarrier();
    }}

    // Output write: paired vec2<f16>, cols (j, j+1) into one word.
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

pub struct MatMulI8Bf16Bufs<'a> {
    pub a: &'a BufRef,
    pub a_params: &'a BufRef,
    pub b: &'a BufRef,
    pub b_sum: &'a BufRef,
    pub out: &'a BufRef,
    pub dims: &'a BufRef,
}

pub fn workgroups(c: &MatMulI8Bf16Config, m: u32, n: u32) -> [u32; 3] {
    [n.div_ceil(c.bn), m.div_ceil(c.bm), 1]
}

pub fn dispatch_matmul_i8_bf16<B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    cfg: &MatMulI8Bf16Config,
    bufs: &MatMulI8Bf16Bufs<'_>,
    m: u32,
    n: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.a.binding(0),
        bufs.a_params.binding(1),
        bufs.b.binding(2),
        bufs.b_sum.binding(3),
        bufs.out.binding(4),
        bufs.dims.binding(5),
    ];
    backend.dispatch(encoder, pipeline, &bindings, workgroups(cfg, m, n))
}
