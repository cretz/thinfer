use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
#[cfg(feature = "conformance")]
use crate::conformance::{
    DTYPES_ACT_BF16, Dtype, OpSpec, OpTest, OpTestContext, TestCase, linspace, t,
};
use crate::ops::{ActDtype, WeightDtype, WgslConfig};
use crate::tensor::{ComputeDtype, F32};

/// Tile/blocking shape for the matmul kernel. The kernel computes a
/// `bm x bn` output tile per workgroup, each thread accumulating a
/// `tm x tn` register block. Workgroup size is therefore
/// `(bn/tn, bm/tm, 1)`; `bn % tn == 0` and `bm % tm == 0` are required.
/// `bk` is the K-strip loaded into shared memory per K-step.
///
/// Default is 16x16x16 / 1x1 (one output per thread, 256-thread WG),
/// matching the original kernel shape so swaps land bit-identically
/// before per-regime tuning starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatMulConfig {
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    pub tm: u32,
    pub tn: u32,
    /// B viewed as `[N, K]` N-major (each row is one N-column's K-strip)
    /// instead of the default `[K, N]` K-major. Coalesces reads when B was
    /// produced by a per-block dequant pass that emits N-major weights to
    /// avoid strided writes. Bf16-weight only; asserts elsewhere.
    pub b_nmajor: bool,
}

impl MatMulConfig {
    pub const DEFAULT: Self = Self {
        bm: 16,
        bn: 16,
        bk: 16,
        tm: 1,
        tn: 1,
        b_nmajor: false,
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
        assert!(self.bm > 0 && self.bn > 0 && self.bk > 0);
        assert!(self.tm > 0 && self.tn > 0);
        assert!(
            self.bm.is_multiple_of(self.tm),
            "bm must be a multiple of tm"
        );
        assert!(
            self.bn.is_multiple_of(self.tn),
            "bn must be a multiple of tn"
        );
    }
}

impl Default for MatMulConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Sibling trait to `Op` for ops with weight bindings, small uniforms, or
/// a non-elementwise output shape. First instance: matmul.
///
/// Layout convention: 0=A, 1=B, 2=Out, 3=Dims uniform `{m, n, k, _pad}` (u32x4).
/// A is row-major `[M, K]`, B is row-major `[K, N]`, Out is row-major `[M, N]`.
pub trait MatmulOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const A: &'static str;
    const B: &'static str;
    const DIMS: &'static str;
    const OUTPUT: &'static str;

    fn config(&self) -> &MatMulConfig;
    fn wgsl(&self, cfg: &WgslConfig) -> String;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(&self, m: u32, n: u32) -> [u32; 3] {
        let c = self.config();
        [n.div_ceil(c.bn), m.div_ceil(c.bm), 1]
    }

    /// Pipeline-cache hint: must capture every cfg field that affects WGSL.
    fn hint(&self, cfg: &WgslConfig) -> String {
        let c = self.config();
        let nm = if c.b_nmajor { "_nm" } else { "" };
        format!(
            "{}-bm{}_bn{}_bk{}_tm{}_tn{}{}",
            cfg.hint(),
            c.bm,
            c.bn,
            c.bk,
            c.tm,
            c.tn,
            nm,
        )
    }
}

pub struct MatmulBufs<'a> {
    pub a: &'a BufRef,
    pub b: &'a BufRef,
    pub dims: &'a BufRef,
    pub out: &'a BufRef,
}

pub fn dispatch_matmul<O: MatmulOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    op: &O,
    bufs: &MatmulBufs<'_>,
    m: u32,
    n: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.a.binding(0),
        bufs.b.binding(1),
        bufs.out.binding(2),
        bufs.dims.binding(3),
    ];
    backend.dispatch(encoder, pipeline, &bindings, op.workgroups(m, n))
}

/// Build the matmul WGSL for a given `(MatMulConfig, WgslConfig)`. The
/// kernel is the classic two-level tiled matmul:
///
/// - Each workgroup owns a `bm x bn` output tile.
/// - Per K-step, the workgroup cooperatively loads a `bm x bk` slab of A
///   and a `bk x bn` slab of B into shared memory.
/// - Each thread accumulates a `tm x tn` register block of outputs.
/// - Bounds checks on every load and every store; non-multiple dims work.
///
/// Variants over `WgslConfig`:
/// - `weight_dtype = Bf16`: B is `array<u32>` with 2 bf16 per word; `load_b`
///   extracts and bitcasts. F32: B is `array<f32>`, direct read.
/// - `bf16_quant_writes = true`: every output store RNE-rounds to bf16
///   (NaN/inf passthrough). Compute and accumulators stay fp32.
fn build_wgsl(c: &MatMulConfig, cfg: &WgslConfig) -> String {
    c.validate();
    if c.b_nmajor {
        assert!(
            matches!(cfg.weight_dtype, WeightDtype::Bf16 | WeightDtype::F16),
            "b_nmajor only valid with Bf16/F16 weight (post-dequant dense workspace), \
             got {:?}",
            cfg.weight_dtype,
        );
    }
    if let WeightDtype::Quant(k) = cfg.weight_dtype {
        let bs = k.block_size();
        let bk = c.bk;
        assert!(
            (bk >= bs && bk.is_multiple_of(bs)) || (bk < bs && bs.is_multiple_of(bk)),
            "matmul Quant({:?}) requires bk ({}) and block_size ({}) to be \
             divisor-aligned (bk%bs==0 or bs%bk==0)",
            k,
            bk,
            bs,
        );
    }
    let bm = c.bm;
    let bn = c.bn;
    let bk = c.bk;
    let tm = c.tm;
    let tn = c.tn;
    let wg_x = c.wg_x();
    let wg_y = c.wg_y();
    let threads = c.threads();
    let tile_a_size = bm * bk;
    let tile_b_size = bk * bn;
    let acc_size = tm * tn;
    let a_loads_per_thread = tile_a_size.div_ceil(threads);
    let b_loads_per_thread = tile_b_size.div_ceil(threads);

    let act_bf16 = cfg.act_dtype == ActDtype::Bf16;
    let act_f16 = cfg.act_dtype == ActDtype::F16;
    let act_packed = act_bf16 || act_f16;
    // F16-tiles path: when the activation dtype is native f16, both tile_a
    // and tile_b live in shared memory as `vec2<f16>` (one slot = 2
    // K-adjacent elements). This halves shared-mem pressure vs the f32
    // tiles used elsewhere, halves the shared-mem read bandwidth in the
    // inner loop (one slot read = 2 multiplies), and keeps the f32
    // accumulator so K-reduction precision is unchanged. K-pair indexing
    // requires bk even and the loads/dequants to emit aligned pairs.
    let tiles_f16 = act_f16;
    if tiles_f16 {
        assert!(
            bk.is_multiple_of(2),
            "f16 tiles require bk % 2 == 0 (got bk={bk})"
        );
    }
    let tile_a_pairs = tile_a_size / 2;
    let tile_b_pairs = tile_b_size / 2;
    // Enforce the same workgroup-storage budget we request from the device
    // (`adapter_limits.max_compute_workgroup_storage_size`, capped by the
    // downlevel default of 32768). Naga does NOT validate `var<workgroup>`
    // size against the device limit at pipeline creation time, so an
    // over-budget kernel silently launches with undefined behavior on the
    // driver - on Intel iGPU we observed dispatches becoming no-ops with the
    // dst buffer retaining stale pool contents (zeros for fresh allocs,
    // saturated f16 for reused slots), masquerading as a numerical quant bug.
    const MAX_WORKGROUP_STORAGE: u32 = 32768;
    let tile_a_bytes = if tiles_f16 {
        tile_a_pairs * 4
    } else {
        tile_a_size * 4
    };
    let tile_b_bytes = if tiles_f16 {
        tile_b_pairs * 4
    } else {
        tile_b_size * 4
    };
    let total_shared = tile_a_bytes + tile_b_bytes;
    assert!(
        total_shared <= MAX_WORKGROUP_STORAGE,
        "matmul kernel exceeds workgroup storage budget: bm={bm} bn={bn} bk={bk} \
         weight={:?} act={:?} tile_a={tile_a_bytes} tile_b={tile_b_bytes} \
         total={total_shared} > limit={MAX_WORKGROUP_STORAGE}",
        cfg.weight_dtype,
        cfg.act_dtype,
    );
    let a_pair_loads_per_thread = tile_a_pairs.div_ceil(threads);
    let b_pair_loads_per_thread = tile_b_pairs.div_ceil(threads);
    if act_packed {
        assert!(
            tn.is_multiple_of(2),
            "matmul packed acts require tn % 2 == 0; got tn={tn}"
        );
        assert!(
            bn.is_multiple_of(2),
            "matmul packed acts require bn % 2 == 0; got bn={bn}"
        );
    }
    // `enable f16;` directive when EITHER acts or weights are native f16. Must
    // precede every other declaration in the WGSL source, so emit it first.
    let weight_f16 = matches!(cfg.weight_dtype, WeightDtype::F16);
    let f16_prelude = if act_f16 || weight_f16 {
        "enable f16;\n"
    } else {
        ""
    };
    let a_elem_decl = if act_bf16 {
        "@group(0) @binding(0) var<storage, read> a: array<u32>;"
    } else if act_f16 {
        "@group(0) @binding(0) var<storage, read> a: array<vec2<f16>>;"
    } else {
        "@group(0) @binding(0) var<storage, read> a: array<f32>;"
    };
    let out_elem_decl = if act_bf16 {
        "@group(0) @binding(2) var<storage, read_write> out: array<u32>;"
    } else if act_f16 {
        "@group(0) @binding(2) var<storage, read_write> out: array<vec2<f16>>;"
    } else {
        "@group(0) @binding(2) var<storage, read_write> out: array<f32>;"
    };
    // `load_a(i)` returns one f32 activation regardless of storage dtype.
    // tile_a stays f32 (matmul accumulators must be f32 — see the K=10240
    // FFN row reductions; an f16 tile_a would only save shared memory
    // while costing precision at every workgroup boundary).
    let load_a = if act_bf16 {
        "fn load_a(i: u32) -> f32 {\n  let pair = a[i >> 1u];\n  let shift = (i & 1u) * 16u;\n  let half = (pair >> shift) & 0xFFFFu;\n  return bitcast<f32>(half << 16u);\n}\n"
    } else if act_f16 {
        // i indexes a single f16 element. Two elements share one vec2<f16>
        // word — fetch the word, widen to vec2<f32>, pick the lane.
        "fn load_a(i: u32) -> f32 {\n  let pair: vec2<f32> = vec2<f32>(a[i >> 1u]);\n  return select(pair.x, pair.y, (i & 1u) == 1u);\n}\n"
    } else {
        "fn load_a(i: u32) -> f32 { return a[i]; }\n"
    };
    let b_elem_decl = match cfg.weight_dtype {
        WeightDtype::F32 => "@group(0) @binding(1) var<storage, read> b: array<f32>;",
        WeightDtype::Bf16 => "@group(0) @binding(1) var<storage, read> b: array<u32>;",
        WeightDtype::F16 => "@group(0) @binding(1) var<storage, read> b: array<vec2<f16>>;",
        WeightDtype::Quant(k) => k.storage_decl(1),
    };
    let load_b = match cfg.weight_dtype {
        WeightDtype::F32 => "fn load_b(i: u32) -> f32 { return b[i]; }\n".to_string(),
        WeightDtype::Bf16 => {
            "fn load_b(i: u32) -> f32 {\n  let pair = b[i >> 1u];\n  let shift = (i & 1u) * 16u;\n  let half = (pair >> shift) & 0xFFFFu;\n  return bitcast<f32>(half << 16u);\n}\n".to_string()
        }
        WeightDtype::F16 => {
            // i indexes one f16 element; two share a vec2<f16> word. Widen to
            // vec2<f32> once and pick the lane. Used only by the scalar B-load
            // path; tiles_f16 + F16 + b_nmajor reads vec2<f16> directly into
            // tile_b (no conversion) below.
            "fn load_b(i: u32) -> f32 {\n  let pair: vec2<f32> = vec2<f32>(b[i >> 1u]);\n  return select(pair.x, pair.y, (i & 1u) == 1u);\n}\n".to_string()
        }
        // Quant path: no per-element load_b. The block-cooperative loader
        // emitted into the kernel body dequants whole blocks straight into
        // tile_b. We only need the dequant helpers (f16_bits_to_f32, b_byte,
        // sext_i8, load_b_block_<scheme>).
        WeightDtype::Quant(k) => k.load_b_block_fn(),
    };
    // Output packing helpers. bf16-packed needs `pack_bf16x2`; f16 acts use
    // the native `vec2<f16>(...)` constructor at the write site, no helper.
    let pack_helpers = if act_bf16 {
        concat!(
            "fn round_bf16(x: f32) -> u32 {\n",
            "  let b = bitcast<u32>(x);\n",
            "  if ((b & 0x7F800000u) == 0x7F800000u) { return (b >> 16u) & 0xFFFFu; }\n",
            "  let l = (b >> 16u) & 1u;\n",
            "  return ((b + 0x7FFFu + l) >> 16u) & 0xFFFFu;\n",
            "}\n",
            "fn pack_bf16x2(lo: f32, hi: f32) -> u32 {\n",
            "  return round_bf16(lo) | (round_bf16(hi) << 16u);\n",
            "}\n",
        )
    } else {
        ""
    };
    let act_store = if cfg.bf16_quant_writes && !act_packed {
        "fn act_store(x: f32) -> f32 {\n  let b = bitcast<u32>(x);\n  if ((b & 0x7F800000u) == 0x7F800000u) { return x; }\n  let l = (b >> 16u) & 1u;\n  return bitcast<f32>((b + 0x7FFFu + l) & 0xFFFF0000u);\n}\n"
    } else {
        "fn act_store(x: f32) -> f32 { return x; }\n"
    };

    // Build the inner B-load loop. F32/Bf16 use the original per-cell
    // cooperative load. Quant uses a block-cooperative path: B is
    // viewed as [N, K] in N-major blocks; each thread dequants whole
    // blocks straight into tile_b. When `tiles_f16` is on, each slot is
    // a `vec2<f16>` covering two K-adjacent elements (br, br+1) for the
    // same N column, so the load iterates pairs instead of cells.
    let b_load_loop = match cfg.weight_dtype {
        // Fast path: F16 dense workspace + f16 tiles + N-major. The on-disk
        // pair layout (vec2<f16> of (gr0, gr0+1) per N-column) matches the
        // tile_b slot exactly, so the inner loop is one paired load = one
        // paired store. Zero conversions, vs the Bf16 workspace path's
        // bf16-unpack -> f32 -> f16-narrow round trip per element. This is
        // the headline win of WeightDtype::F16.
        WeightDtype::F16 if tiles_f16 && c.b_nmajor => {
            // d.k is even (asserted by tiles_f16's bk%2==0 + caller dims).
            // gr0 = k0 + br_pair*2 is always even, so the pair (gr0, gr0+1)
            // sits aligned in one vec2<f16> word at b[gc*(d.k/2) + gr0/2].
            format!(
                r#"        for (var s: u32 = 0u; s < {b_pair_loads_per_thread}u; s = s + 1u) {{
            let pair_idx: u32 = s * THREADS + tid;
            if (pair_idx < (BK * BN) / 2u) {{
                let br_pair: u32 = pair_idx / BN;
                let bc: u32 = pair_idx % BN;
                let gr0: u32 = k0 + br_pair * 2u;
                let gc: u32 = bn0 + bc;
                var w: vec2<f16> = vec2<f16>(0.0, 0.0);
                if (gc < d.n && gr0 < d.k) {{
                    w = b[gc * (d.k >> 1u) + (gr0 >> 1u)];
                }}
                tile_b[br_pair * BN + bc] = w;
            }}
        }}
"#
            )
        }
        WeightDtype::F32 | WeightDtype::Bf16 | WeightDtype::F16 if tiles_f16 => {
            let (lo_idx, hi_idx) = if c.b_nmajor {
                ("gc * d.k + gr0", "gc * d.k + gr0 + 1u")
            } else {
                ("gr0 * d.n + gc", "(gr0 + 1u) * d.n + gc")
            };
            format!(
                r#"        for (var s: u32 = 0u; s < {b_pair_loads_per_thread}u; s = s + 1u) {{
            let pair_idx: u32 = s * THREADS + tid;
            if (pair_idx < (BK * BN) / 2u) {{
                let br_pair: u32 = pair_idx / BN;
                let bc: u32 = pair_idx % BN;
                let gr0: u32 = k0 + br_pair * 2u;
                let gc: u32 = bn0 + bc;
                var lo: f32 = 0.0;
                var hi: f32 = 0.0;
                if (gc < d.n) {{
                    if (gr0 < d.k) {{ lo = load_b({lo_idx}); }}
                    if (gr0 + 1u < d.k) {{ hi = load_b({hi_idx}); }}
                }}
                tile_b[br_pair * BN + bc] = vec2<f16>(vec2<f32>(lo, hi));
            }}
        }}
"#
            )
        }
        WeightDtype::F32 | WeightDtype::Bf16 | WeightDtype::F16 => {
            let idx_expr = if c.b_nmajor {
                "gc * d.k + gr"
            } else {
                "gr * d.n + gc"
            };
            format!(
                r#"        for (var s: u32 = 0u; s < {b_loads_per_thread}u; s = s + 1u) {{
            let idx: u32 = s * THREADS + tid;
            if (idx < BK * BN) {{
                let br: u32 = idx / BN;
                let bc: u32 = idx % BN;
                let gr: u32 = k0 + br;
                let gc: u32 = bn0 + bc;
                var v: f32 = 0.0;
                if (gr < d.k && gc < d.n) {{
                    v = load_b({idx_expr});
                }}
                tile_b[idx] = v;
            }}
        }}
"#
            )
        }
        WeightDtype::Quant(k) => {
            let bs = k.block_size();
            let bpb = k.bytes_per_block();
            let scale_call = k.block_state_call();
            let elem_call = k.block_elem_call();
            let elem4_call = k.block_elem4_call();
            // Validated above: bk and bs are divisor-aligned (bk%bs==0 or
            // bs%bk==0). Two regimes unified:
            //   bk>=bs: one WG K-step covers `bk/bs` whole blocks per column;
            //           each block's elements span the full bs.
            //   bk<bs:  one WG K-step covers a partial bk-wide slice of one
            //           block per column; the slice starts at `(t*bk)%bs`
            //           within the block. State (scales, etc) is recomputed
            //           per strip; cheap on-chip vs the tile-shape win.
            let epb = bk.min(bs); // elems per block in this strip
            let n_blocks_per_strip = (bk / bs).max(1);
            let block_slots = n_blocks_per_strip * bn;
            // TPB = threads-per-block cooperative dequant. Largest pow2
            // factor of epb such that block_slots*TPB <= THREADS, so the
            // WG saturates the B-load without redundant work.
            let mut tpb: u32 = 1;
            while tpb * 2 <= epb && tpb * 2 * block_slots <= threads {
                tpb *= 2;
            }
            let elems_per_thread = epb / tpb;
            let slot_threads = block_slots * tpb;
            let slots_per_thread = slot_threads.div_ceil(threads);
            if tiles_f16 {
                assert!(
                    elems_per_thread.is_multiple_of(2),
                    "f16 tiles + Quant require elems_per_thread % 2 == 0 \
                     (got {elems_per_thread}; epb={epb}, tpb={tpb})"
                );
                assert!(
                    epb.is_multiple_of(2),
                    "f16 tiles + Quant require epb even (got {epb})"
                );
            }
            // vec4 bulk-dequant path. One u32 weight read + 4 nibble/byte
            // extracts per call, vs 4 separate `b_byte` calls in the scalar
            // path. Gated on `elems_per_thread % 4 == 0` (true for all
            // current configs: K-family ept=16, Q8_0/Q4_0 ept=8). Caller-side
            // alignment of `base_elem` is enforced by the cooperative loader:
            // `base_elem = block_elem_start + st * elems_per_thread`, both
            // terms multiples of `elems_per_thread`, so a 4-aligned ept
            // gives 4-aligned `base_elem + i` for i stepping by 4.
            let use_vec4 = elems_per_thread.is_multiple_of(4);
            let inner_dequant = if tiles_f16 && use_vec4 {
                format!(
                    r#"                        for (var i: u32 = 0u; i < {elems_per_thread}u; i = i + 4u) {{
                            let k_strip: u32 = tile_b_k_start + i;
                            let v: vec4<f32> = {elem4_call}(byte0, scale, base_elem + i);
                            tile_b[(k_strip >> 1u) * BN + bc] = vec2<f16>(vec2<f32>(v.x, v.y));
                            tile_b[((k_strip + 2u) >> 1u) * BN + bc] = vec2<f16>(vec2<f32>(v.z, v.w));
                        }}"#
                )
            } else if tiles_f16 {
                format!(
                    r#"                        for (var i: u32 = 0u; i < {elems_per_thread}u; i = i + 2u) {{
                            let k_strip: u32 = tile_b_k_start + i;
                            let lo: f32 = {elem_call}(byte0, scale, base_elem + i);
                            let hi: f32 = {elem_call}(byte0, scale, base_elem + i + 1u);
                            tile_b[(k_strip >> 1u) * BN + bc] = vec2<f16>(vec2<f32>(lo, hi));
                        }}"#
                )
            } else if use_vec4 {
                format!(
                    r#"                        for (var i: u32 = 0u; i < {elems_per_thread}u; i = i + 4u) {{
                            let v: vec4<f32> = {elem4_call}(byte0, scale, base_elem + i);
                            tile_b[(tile_b_k_start + i + 0u) * BN + bc] = v.x;
                            tile_b[(tile_b_k_start + i + 1u) * BN + bc] = v.y;
                            tile_b[(tile_b_k_start + i + 2u) * BN + bc] = v.z;
                            tile_b[(tile_b_k_start + i + 3u) * BN + bc] = v.w;
                        }}"#
                )
            } else {
                format!(
                    r#"                        for (var i: u32 = 0u; i < {elems_per_thread}u; i = i + 1u) {{
                            tile_b[(tile_b_k_start + i) * BN + bc] =
                                {elem_call}(byte0, scale, base_elem + i);
                        }}"#
                )
            };
            let oob_dequant = if tiles_f16 {
                format!(
                    r#"                        for (var i: u32 = 0u; i < {elems_per_thread}u; i = i + 2u) {{
                            let k_strip: u32 = tile_b_k_start + i;
                            tile_b[(k_strip >> 1u) * BN + bc] = vec2<f16>(0.0, 0.0);
                        }}"#
                )
            } else {
                format!(
                    r#"                        for (var i: u32 = 0u; i < {elems_per_thread}u; i = i + 1u) {{
                            tile_b[(tile_b_k_start + i) * BN + bc] = 0.0;
                        }}"#
                )
            };
            format!(
                r#"        // Quant B load. bk={bk}, block_size={bs}, epb={epb},
        // n_blocks_per_strip={n_blocks_per_strip}, TPB={tpb},
        // elems_per_thread={elems_per_thread}, bytes_per_block={bpb}.
        // B is viewed as [N, K] in N-major blocks.
        {{
            let blocks_per_row: u32 = d.k / {bs}u;
            for (var s: u32 = 0u; s < {slots_per_thread}u; s = s + 1u) {{
                let slot: u32 = s * THREADS + tid;
                if (slot < {slot_threads}u) {{
                    let block_slot: u32 = slot / {tpb}u;
                    let st: u32 = slot % {tpb}u;
                    let kc: u32 = block_slot / BN;
                    let bc: u32 = block_slot % BN;
                    let n: u32 = bn0 + bc;
                    let kc_start_in_strip: u32 = kc * {epb}u;
                    let g_k_strip_start: u32 = t * BK + kc_start_in_strip;
                    let block_k_outer: u32 = g_k_strip_start / {bs}u;
                    let block_elem_start: u32 = g_k_strip_start - block_k_outer * {bs}u;
                    let base_elem: u32 = block_elem_start + st * {elems_per_thread}u;
                    let tile_b_k_start: u32 = kc_start_in_strip + st * {elems_per_thread}u;
                    if (n < d.n && block_k_outer < blocks_per_row) {{
                        let global_block_idx: u32 = n * blocks_per_row + block_k_outer;
                        let byte0: u32 = global_block_idx * {bpb}u;
                        let scale = {scale_call}(byte0);
{inner_dequant}
                    }} else {{
{oob_dequant}
                    }}
                }}
            }}
        }}
"#
            )
        }
    };

    let out_write_loop = if act_bf16 {
        // Pair (j, j+1) into one packed word at output column c = bn0+lid.x*TN+j.
        // c is even (bn even, TN even, lid.x*TN even). d.n is even (caller).
        r#"
    for (var i: u32 = 0u; i < TM; i = i + 1u) {
        let row: u32 = bm0 + lid.y * TM + i;
        if (row >= d.m) { continue; }
        for (var j: u32 = 0u; j < TN; j = j + 2u) {
            let col: u32 = bn0 + lid.x * TN + j;
            if (col >= d.n) { continue; }
            let widx = (row * d.n + col) >> 1u;
            out[widx] = pack_bf16x2(acc[i * TN + j], acc[i * TN + j + 1u]);
        }
    }
"#
    } else if act_f16 {
        // Same pairing as bf16-packed, but typed `vec2<f16>` writes. The
        // f32 accumulator pair narrows to f16 at the storage boundary.
        //
        // Saturated narrow: clamp to the f16 finite range (+-65504.0) before
        // the f16 cast so out-of-range accumulator values land at the f16
        // maxima instead of +-inf. Without this clamp, F16 attention-output
        // projection at constant-input rows (cap/x padding rows fed the
        // learned pad_token) overflows on Z-Image proj weights, then the
        // downstream rmsnorm computes `rsqrt(inf) = 0` and `inf * 0 = NaN`,
        // poisoning every later attention because attn-key/value rows go
        // NaN. Upstream model semantics expect attendance on padding rows
        // (transformer.py: `attn_mask[i, :seq_len] = 1` includes inner
        // SEQ_MULTI_OF padding), so masking them is wrong; saturating the
        // F16 narrow is the correct precision-loss-vs-NaN trade-off, since
        // padding rows are discarded post-DiT and image rows attending to
        // saturated padding values incur only a small F16 precision delta
        // (still bounded by the F16 mantissa).
        r#"
    for (var i: u32 = 0u; i < TM; i = i + 1u) {
        let row: u32 = bm0 + lid.y * TM + i;
        if (row >= d.m) { continue; }
        for (var j: u32 = 0u; j < TN; j = j + 2u) {
            let col: u32 = bn0 + lid.x * TN + j;
            if (col >= d.n) { continue; }
            let widx = (row * d.n + col) >> 1u;
            let lo = clamp(acc[i * TN + j], -65504.0, 65504.0);
            let hi = clamp(acc[i * TN + j + 1u], -65504.0, 65504.0);
            out[widx] = vec2<f16>(vec2<f32>(lo, hi));
        }
    }
"#
    } else {
        r#"
    for (var i: u32 = 0u; i < TM; i = i + 1u) {
        let row: u32 = bm0 + lid.y * TM + i;
        if (row >= d.m) { continue; }
        for (var j: u32 = 0u; j < TN; j = j + 1u) {
            let col: u32 = bn0 + lid.x * TN + j;
            if (col >= d.n) { continue; }
            out[row * d.n + col] = act_store(acc[i * TN + j]);
        }
    }
"#
    };
    let (tile_a_decl, tile_b_decl, a_load_loop, inner_loop) = if tiles_f16 {
        // Shared-mem tiles in vec2<f16>: each slot covers two K-adjacent
        // elements for one (row, col) of the output tile. A-load iterates
        // pairs along K; inner loop steps kk by 2 and FMAs both lanes.
        let tile_a_decl = format!("var<workgroup> tile_a: array<vec2<f16>, {tile_a_pairs}u>;");
        let tile_b_decl = format!("var<workgroup> tile_b: array<vec2<f16>, {tile_b_pairs}u>;");
        // Fast path: direct paired vec2<f16> read from `a` storage straight
        // into `tile_a`. Requires d.k even (always true for Z-Image: all
        // model dims are even, and K is a multiple of bk which is itself
        // asserted even). Zero conversions vs the load_a path's vec2<f16>
        // -> vec2<f32> -> pick lane -> vec2<f16> round trip.
        let a_load_loop = format!(
            r#"        for (var s: u32 = 0u; s < {a_pair_loads_per_thread}u; s = s + 1u) {{
            let pair_idx: u32 = s * THREADS + tid;
            if (pair_idx < (BM * BK) / 2u) {{
                let ar: u32 = pair_idx / (BK / 2u);
                let ac_pair: u32 = pair_idx % (BK / 2u);
                let gr: u32 = bm0 + ar;
                let gc0: u32 = k0 + ac_pair * 2u;
                var w: vec2<f16> = vec2<f16>(0.0, 0.0);
                if (gr < d.m && gc0 < d.k) {{
                    w = a[(gr * d.k + gc0) >> 1u];
                }}
                tile_a[ar * (BK / 2u) + ac_pair] = w;
            }}
        }}
"#
        );
        // K-pair inner loop. `av_pair` is loaded once per (i, kk2) and reused
        // across the TN-wide register block.
        //
        // Arithmetic: multiply in native `vec2<f16>` (one vec2 ALU op on
        // Intel Xe / Apple Silicon / RDNA, vs two f32 muls), widen the
        // 2-lane product to `vec2<f32>` at the accumulator-add boundary so
        // the K-reduction stays full-precision. Per the design doc:
        // "widen happens at the reduction boundary, not at every op
        // boundary." Per-multiply precision delta vs prior f32-mul code is
        // bounded by f16 mantissa (10 bits); cumulative dot-product
        // precision is dominated by the f32 accumulator.
        let inner_loop = r#"        for (var kk2: u32 = 0u; kk2 < BK / 2u; kk2 = kk2 + 1u) {
            for (var i: u32 = 0u; i < TM; i = i + 1u) {
                let a_row: u32 = lid.y * TM + i;
                let av_pair: vec2<f16> = tile_a[a_row * (BK / 2u) + kk2];
                for (var j: u32 = 0u; j < TN; j = j + 1u) {
                    let b_col: u32 = lid.x * TN + j;
                    let bv_pair: vec2<f16> = tile_b[kk2 * BN + b_col];
                    let prod: vec2<f32> = vec2<f32>(av_pair * bv_pair);
                    acc[i * TN + j] = acc[i * TN + j] + prod.x + prod.y;
                }
            }
        }
"#
        .to_string();
        (tile_a_decl, tile_b_decl, a_load_loop, inner_loop)
    } else {
        let tile_a_decl = format!("var<workgroup> tile_a: array<f32, {tile_a_size}u>;");
        let tile_b_decl = format!("var<workgroup> tile_b: array<f32, {tile_b_size}u>;");
        let a_load_loop = format!(
            r#"        for (var s: u32 = 0u; s < {a_loads_per_thread}u; s = s + 1u) {{
            let idx: u32 = s * THREADS + tid;
            if (idx < BM * BK) {{
                let ar: u32 = idx / BK;
                let ac: u32 = idx % BK;
                let gr: u32 = bm0 + ar;
                let gc: u32 = k0 + ac;
                var v: f32 = 0.0;
                if (gr < d.m && gc < d.k) {{
                    v = load_a(gr * d.k + gc);
                }}
                tile_a[idx] = v;
            }}
        }}
"#
        );
        let inner_loop = r#"        for (var kk: u32 = 0u; kk < BK; kk = kk + 1u) {
            for (var i: u32 = 0u; i < TM; i = i + 1u) {
                let a_row: u32 = lid.y * TM + i;
                let av: f32 = tile_a[a_row * BK + kk];
                for (var j: u32 = 0u; j < TN; j = j + 1u) {
                    let b_col: u32 = lid.x * TN + j;
                    acc[i * TN + j] = acc[i * TN + j] + av * tile_b[kk * BN + b_col];
                }
            }
        }
"#
        .to_string();
        (tile_a_decl, tile_b_decl, a_load_loop, inner_loop)
    };
    format!(
        r#"{f16_prelude}
struct Dims {{ m: u32, n: u32, k: u32, _pad: u32 }};

{a_elem_decl}
{b_elem_decl}
{out_elem_decl}
@group(0) @binding(3) var<uniform> d: Dims;

{load_a}
{load_b}
{pack_helpers}
{act_store}

const BM: u32 = {bm}u;
const BN: u32 = {bn}u;
const BK: u32 = {bk}u;
const TM: u32 = {tm}u;
const TN: u32 = {tn}u;
const WG_X: u32 = {wg_x}u;
const WG_Y: u32 = {wg_y}u;
const THREADS: u32 = {threads}u;

{tile_a_decl}
{tile_b_decl}

@compute @workgroup_size({wg_x}, {wg_y}, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let tid: u32 = lid.y * WG_X + lid.x;
    let bm0: u32 = wid.y * BM;
    let bn0: u32 = wid.x * BN;
    var acc: array<f32, {acc_size}u>;
    for (var i: u32 = 0u; i < {acc_size}u; i = i + 1u) {{
        acc[i] = 0.0;
    }}

    let num_tiles: u32 = (d.k + BK - 1u) / BK;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {{
        let k0: u32 = t * BK;

{a_load_loop}
{b_load_loop}
        workgroupBarrier();

{inner_loop}
        workgroupBarrier();
    }}

{out_write_loop}
}}
"#
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
        kind: BindingKind::StorageReadWrite,
    },
    BindingLayout {
        slot: 3,
        kind: BindingKind::Uniform,
    },
];

pub struct MatMulF32 {
    pub cfg: MatMulConfig,
}

impl MatMulF32 {
    pub const fn new(cfg: MatMulConfig) -> Self {
        Self { cfg }
    }

    pub const fn default_op() -> Self {
        Self {
            cfg: MatMulConfig::DEFAULT,
        }
    }
}

impl Default for MatMulF32 {
    fn default() -> Self {
        Self::default_op()
    }
}

impl MatmulOp for MatMulF32 {
    const KERNEL_ID: &'static str = "matmul.f32";
    type Dtype = F32;
    const A: &'static str = "matmul/a";
    const B: &'static str = "matmul/b";
    const DIMS: &'static str = "matmul/dims";
    const OUTPUT: &'static str = "matmul/out";

    fn config(&self) -> &MatMulConfig {
        &self.cfg
    }
    fn wgsl(&self, cfg: &WgslConfig) -> String {
        build_wgsl(&self.cfg, cfg)
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl OpTest for MatMulF32 {
    fn dtypes(&self) -> &'static [Dtype] {
        DTYPES_ACT_BF16
    }
    fn test_cases(&self) -> Vec<TestCase> {
        vec![TestCase {
            name: "matmul_basic",
            op: OpSpec::Matmul,
            inputs: vec![
                t("a", [4, 8], linspace(-1.0, 1.0, false)),
                t("b", [8, 6], linspace(-1.5, 1.75, false)),
            ],
        }]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_matmul(MatMulF32::new(self.cfg)))
    }
}
