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
}

impl MatMulConfig {
    pub const DEFAULT: Self = Self {
        bm: 16,
        bn: 16,
        bk: 16,
        tm: 1,
        tn: 1,
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
        format!(
            "{}-bm{}_bn{}_bk{}_tm{}_tn{}",
            cfg.hint(),
            c.bm,
            c.bn,
            c.bk,
            c.tm,
            c.tn
        )
    }
}

pub struct MatmulBufs<'a> {
    pub a: &'a BufRef,
    pub b: &'a BufRef,
    pub dims: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_matmul<O: MatmulOp, B: Backend>(
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

    let act_packed = cfg.act_dtype == ActDtype::Bf16;
    if act_packed {
        assert!(
            tn.is_multiple_of(2),
            "matmul packed-bf16 acts require tn % 2 == 0; got tn={tn}"
        );
        assert!(
            bn.is_multiple_of(2),
            "matmul packed-bf16 acts require bn % 2 == 0; got bn={bn}"
        );
    }
    let a_elem_decl = if act_packed {
        "@group(0) @binding(0) var<storage, read> a: array<u32>;"
    } else {
        "@group(0) @binding(0) var<storage, read> a: array<f32>;"
    };
    let out_elem_decl = if act_packed {
        "@group(0) @binding(2) var<storage, read_write> out: array<u32>;"
    } else {
        "@group(0) @binding(2) var<storage, read_write> out: array<f32>;"
    };
    let load_a = if act_packed {
        "fn load_a(i: u32) -> f32 {\n  let pair = a[i >> 1u];\n  let shift = (i & 1u) * 16u;\n  let half = (pair >> shift) & 0xFFFFu;\n  return bitcast<f32>(half << 16u);\n}\n"
    } else {
        "fn load_a(i: u32) -> f32 { return a[i]; }\n"
    };
    let b_elem_decl = match cfg.weight_dtype {
        WeightDtype::F32 => "@group(0) @binding(1) var<storage, read> b: array<f32>;",
        WeightDtype::Bf16 => "@group(0) @binding(1) var<storage, read> b: array<u32>;",
    };
    let load_b = match cfg.weight_dtype {
        WeightDtype::F32 => "fn load_b(i: u32) -> f32 { return b[i]; }\n",
        WeightDtype::Bf16 => {
            "fn load_b(i: u32) -> f32 {\n  let pair = b[i >> 1u];\n  let shift = (i & 1u) * 16u;\n  let half = (pair >> shift) & 0xFFFFu;\n  return bitcast<f32>(half << 16u);\n}\n"
        }
    };
    let pack_helpers = if act_packed {
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

    let out_write_loop = if act_packed {
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
    format!(
        r#"
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

var<workgroup> tile_a: array<f32, {tile_a_size}u>;
var<workgroup> tile_b: array<f32, {tile_b_size}u>;

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

        for (var s: u32 = 0u; s < {a_loads_per_thread}u; s = s + 1u) {{
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
        for (var s: u32 = 0u; s < {b_loads_per_thread}u; s = s + 1u) {{
            let idx: u32 = s * THREADS + tid;
            if (idx < BK * BN) {{
                let br: u32 = idx / BN;
                let bc: u32 = idx % BN;
                let gr: u32 = k0 + br;
                let gc: u32 = bn0 + bc;
                var v: f32 = 0.0;
                if (gr < d.k && gc < d.n) {{
                    v = load_b(gr * d.n + gc);
                }}
                tile_b[idx] = v;
            }}
        }}
        workgroupBarrier();

        for (var kk: u32 = 0u; kk < BK; kk = kk + 1u) {{
            for (var i: u32 = 0u; i < TM; i = i + 1u) {{
                let a_row: u32 = lid.y * TM + i;
                let av: f32 = tile_a[a_row * BK + kk];
                for (var j: u32 = 0u; j < TN; j = j + 1u) {{
                    let b_col: u32 = lid.x * TN + j;
                    acc[i * TN + j] = acc[i * TN + j] + av * tile_b[kk * BN + b_col];
                }}
            }}
        }}
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
