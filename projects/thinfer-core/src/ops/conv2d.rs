use super::{ActDtype, WeightDtype, WgslConfig};
use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
use crate::tensor::{ComputeDtype, F32};

/// `out = conv2d(x, weight) + bias` over NCHW.
///
/// Shapes (row-major):
/// - `x:      [B, Cin, Hin, Win]`
/// - `weight: [Cout, Cin, kH, kW]` (PyTorch `nn.Conv2d` layout)
/// - `bias:   [Cout]`
/// - `out:    [B, Cout, Hout, Wout]`
///
/// Geometry: `Hout = (Hin + 2*pad_h - kH) / stride_h + 1`, same for W.
///
/// Implicit-GEMM kernel: the conv is computed as
/// `weight[Cout, K] @ im2col[K, M]` with `K = Cin*kH*kW` and
/// `M = Hout*Wout`, which lands the output directly in NCHW (row = Cout,
/// col = spatial) with coalesced spatial stores. The im2col matrix is never
/// materialized: the B-tile loader gathers x elements on the fly (zero-fill
/// at padding). Two-level tiling identical in spirit to `matmul.rs`: each
/// workgroup owns a `bm x bn` output tile (bm over Cout, bn over spatial),
/// each thread accumulates a `tm x tn` register block of explicit scalars
/// (kernel register rule: no dynamically-indexed local arrays in hot loops).
/// Grid z is batch. Bias is always required (callers pass a zero buffer if
/// no bias). Dilation and groups are NOT supported (no Z-Image-Turbo VAE
/// conv uses them).
///
/// Layout: 0=X, 1=W, 2=Bias, 3=Out, 4=Uniform.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Conv2dConfig {
    /// Output-tile rows per workgroup (over Cout).
    pub bm: u32,
    /// Output-tile cols per workgroup (over spatial Hout*Wout).
    pub bn: u32,
    /// K-strip depth per shared-memory step (over Cin*kH*kW).
    pub bk: u32,
    /// Register-block rows per thread.
    pub tm: u32,
    /// Register-block cols per thread.
    pub tn: u32,
}

impl Conv2dConfig {
    pub const DEFAULT: Self = Self {
        bm: 64,
        bn: 64,
        bk: 32,
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

impl Default for Conv2dConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

pub trait Conv2dOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const W: &'static str;
    const BIAS: &'static str;
    const DIMS: &'static str;
    const OUTPUT: &'static str;

    fn config(&self) -> &Conv2dConfig;
    fn wgsl(&self, cfg: &WgslConfig) -> String;
    fn layout() -> &'static [BindingLayout];

    /// Grid: x over spatial tiles, y over Cout tiles, z over batch.
    fn workgroups(&self, cout: u32, m_spatial: u32, batch: u32) -> [u32; 3] {
        let c = self.config();
        [m_spatial.div_ceil(c.bn), cout.div_ceil(c.bm), batch]
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
            c.tn,
        )
    }
}

pub struct Conv2dBufs<'a> {
    pub x: &'a BufRef,
    pub w: &'a BufRef,
    pub bias: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_conv2d<O: Conv2dOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    op: &O,
    bufs: &Conv2dBufs<'_>,
    cout: u32,
    m_spatial: u32,
    batch: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.w.binding(1),
        bufs.bias.binding(2),
        bufs.out.binding(3),
        bufs.uniform.binding(4),
    ];
    backend.dispatch(
        encoder,
        pipeline,
        &bindings,
        op.workgroups(cout, m_spatial, batch),
    )
}

/// Build the conv2d implicit-GEMM WGSL for `(Conv2dConfig, WgslConfig)`.
///
/// Variants over `WgslConfig`:
/// - `weight_dtype = Bf16`: weight and bias are `array<u32>` with 2 bf16 per
///   word; F32: direct `array<f32>` reads.
/// - `act_dtype = F16`: x and out storage are `array<f16>` (scalar element
///   loads: im2col gathers have arbitrary alignment). Shared tiles and
///   accumulators stay f32; the f16 boundary halves global act bandwidth
///   without touching K-reduction precision. F32: plain `array<f32>`.
fn build_wgsl(c: &Conv2dConfig, cfg: &WgslConfig) -> String {
    c.validate();
    assert!(
        !cfg.bf16_quant_writes,
        "conv2d has no bf16-quant-write mode"
    );
    let act_f16 = match cfg.act_dtype {
        ActDtype::F32 => false,
        ActDtype::F16 => true,
        other => panic!("conv2d does not support act dtype {other:?}"),
    };
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
    let a_loads_per_thread = tile_a_size.div_ceil(threads);
    let b_loads_per_thread = tile_b_size.div_ceil(threads);

    // Same workgroup-storage budget rationale as matmul.rs: naga does not
    // validate var<workgroup> size against the device limit, so enforce the
    // 32 KiB downlevel cap here.
    const MAX_WORKGROUP_STORAGE: u32 = 32768;
    let total_shared = (tile_a_size + tile_b_size) * 4;
    assert!(
        total_shared <= MAX_WORKGROUP_STORAGE,
        "conv2d kernel exceeds workgroup storage budget: bm={bm} bn={bn} bk={bk} \
         total={total_shared} > limit={MAX_WORKGROUP_STORAGE}",
    );

    let f16_prelude = if act_f16 { "enable f16;\n" } else { "" };
    let (x_elem_decl, out_elem_decl, load_x, store_expr) = if act_f16 {
        (
            "@group(0) @binding(0) var<storage, read> x: array<f16>;",
            "@group(0) @binding(3) var<storage, read_write> out: array<f16>;",
            "fn load_x(i: u32) -> f32 { return f32(x[i]); }\n",
            // Saturated narrow at the f16 storage boundary (same
            // precision-loss-vs-inf trade as matmul's f16 out path).
            "f16(clamp(v, -65504.0, 65504.0))",
        )
    } else {
        (
            "@group(0) @binding(0) var<storage, read> x: array<f32>;",
            "@group(0) @binding(3) var<storage, read_write> out: array<f32>;",
            "fn load_x(i: u32) -> f32 { return x[i]; }\n",
            "v",
        )
    };
    let (w_elem_decl, load_w) = match cfg.weight_dtype {
        WeightDtype::F32 => (
            concat!(
                "@group(0) @binding(1) var<storage, read> wgt: array<f32>;\n",
                "@group(0) @binding(2) var<storage, read> bias: array<f32>;",
            ),
            concat!(
                "fn load_w(i: u32) -> f32 { return wgt[i]; }\n",
                "fn load_bias(i: u32) -> f32 { return bias[i]; }\n",
            ),
        ),
        WeightDtype::Bf16 => (
            concat!(
                "@group(0) @binding(1) var<storage, read> wgt: array<u32>;\n",
                "@group(0) @binding(2) var<storage, read> bias: array<u32>;",
            ),
            concat!(
                "fn unpack_bf16(pair: u32, i: u32) -> f32 {\n",
                "  let half = (pair >> ((i & 1u) * 16u)) & 0xFFFFu;\n",
                "  return bitcast<f32>(half << 16u);\n",
                "}\n",
                "fn load_w(i: u32) -> f32 { return unpack_bf16(wgt[i >> 1u], i); }\n",
                "fn load_bias(i: u32) -> f32 { return unpack_bf16(bias[i >> 1u], i); }\n",
            ),
        ),
        // conv2d is VAE-only; GGUF quants live in DiT matmuls.
        WeightDtype::F16 => unreachable!("conv2d does not consume f16 weights"),
        WeightDtype::Quant(_) => unreachable!("conv2d does not consume quant weights"),
    };

    // Explicit-scalar register block (kernel register rule). Accumulators,
    // inner-loop FMAs, and the store loop are codegen-unrolled over tm x tn.
    let mut acc_decls = String::new();
    for i in 0..tm {
        for j in 0..tn {
            acc_decls.push_str(&format!("    var acc_{i}_{j}: f32 = 0.0;\n"));
        }
    }
    let mut inner_fmas = String::new();
    for i in 0..tm {
        inner_fmas.push_str(&format!(
            "            let a_{i}: f32 = tile_a[(lid.y * TM + {i}u) * BK + kk];\n"
        ));
    }
    for j in 0..tn {
        inner_fmas.push_str(&format!(
            "            let b_{j}: f32 = tile_b[kk * BN + lid.x * TN + {j}u];\n"
        ));
    }
    for i in 0..tm {
        for j in 0..tn {
            inner_fmas.push_str(&format!(
                "            acc_{i}_{j} = fma(a_{i}, b_{j}, acc_{i}_{j});\n"
            ));
        }
    }
    let mut store_loop = String::new();
    for i in 0..tm {
        store_loop.push_str(&format!(
            r#"    {{
        let row: u32 = bm0 + lid.y * TM + {i}u;
        if (row < u.cout) {{
            let bias_v: f32 = load_bias(row);
            let row_base: u32 = out_base + row * m_total;
"#
        ));
        for j in 0..tn {
            let acc = format!("acc_{i}_{j}");
            store_loop.push_str(&format!(
                r#"            {{
                let col: u32 = bn0 + lid.x * TN + {j}u;
                if (col < m_total) {{
                    let v: f32 = {acc} + bias_v;
                    out[row_base + col] = {store_expr};
                }}
            }}
"#
            ));
        }
        store_loop.push_str("        }\n    }\n");
    }

    format!(
        r#"{f16_prelude}
struct U {{
    b: u32, cin: u32, cout: u32, h_in: u32,
    w_in: u32, h_out: u32, w_out: u32, kh: u32,
    kw: u32, pad_h: u32, pad_w: u32, stride_h: u32,
    stride_w: u32, _pad0: u32, _pad1: u32, _pad2: u32,
}};

{x_elem_decl}
{w_elem_decl}
{out_elem_decl}
@group(0) @binding(4) var<uniform> u: U;

{load_x}
{load_w}

const BM: u32 = {bm}u;
const BN: u32 = {bn}u;
const BK: u32 = {bk}u;
const TM: u32 = {tm}u;
const TN: u32 = {tn}u;
const THREADS: u32 = {threads}u;

var<workgroup> tile_a: array<f32, {tile_a_size}u>;
var<workgroup> tile_b: array<f32, {tile_b_size}u>;

@compute @workgroup_size({wg_x}, {wg_y}, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let tid: u32 = lid.y * {wg_x}u + lid.x;
    let bm0: u32 = wid.y * BM;
    let bn0: u32 = wid.x * BN;
    let khkw: u32 = u.kh * u.kw;
    let kdim: u32 = u.cin * khkw;
    let m_total: u32 = u.h_out * u.w_out;
    let hw_in: u32 = u.h_in * u.w_in;
    let x_base: u32 = wid.z * u.cin * hw_in;
    let out_base: u32 = wid.z * u.cout * m_total;

{acc_decls}
    let num_tiles: u32 = (kdim + BK - 1u) / BK;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {{
        let k0: u32 = t * BK;

        // A-tile: weight [Cout, K] row-major; gr = Cout row, gc = K col.
        for (var s: u32 = 0u; s < {a_loads_per_thread}u; s = s + 1u) {{
            let idx: u32 = s * THREADS + tid;
            if (idx < BM * BK) {{
                let ar: u32 = idx / BK;
                let ac: u32 = idx % BK;
                let gr: u32 = bm0 + ar;
                let gc: u32 = k0 + ac;
                var v: f32 = 0.0;
                if (gr < u.cout && gc < kdim) {{
                    v = load_w(gr * kdim + gc);
                }}
                tile_a[idx] = v;
            }}
        }}
        // B-tile: virtual im2col [K, M]; gr = k -> (ci, dh, dw),
        // gc = spatial -> (ho, wo). Zero-fill at padding / OOB. bc is the
        // fast index, so consecutive threads gather consecutive wi
        // (coalesced x reads).
        for (var s: u32 = 0u; s < {b_loads_per_thread}u; s = s + 1u) {{
            let idx: u32 = s * THREADS + tid;
            if (idx < BK * BN) {{
                let br: u32 = idx / BN;
                let bc: u32 = idx % BN;
                let gr: u32 = k0 + br;
                let gc: u32 = bn0 + bc;
                var v: f32 = 0.0;
                if (gr < kdim && gc < m_total) {{
                    let ci: u32 = gr / khkw;
                    let r: u32 = gr - ci * khkw;
                    let dh: u32 = r / u.kw;
                    let dw: u32 = r - dh * u.kw;
                    let ho: u32 = gc / u.w_out;
                    let wo: u32 = gc - ho * u.w_out;
                    let hi: i32 = i32(ho * u.stride_h + dh) - i32(u.pad_h);
                    let wi: i32 = i32(wo * u.stride_w + dw) - i32(u.pad_w);
                    if (hi >= 0 && hi < i32(u.h_in) && wi >= 0 && wi < i32(u.w_in)) {{
                        v = load_x(x_base + ci * hw_in + u32(hi) * u.w_in + u32(wi));
                    }}
                }}
                tile_b[idx] = v;
            }}
        }}
        workgroupBarrier();

        for (var kk: u32 = 0u; kk < BK; kk = kk + 1u) {{
{inner_fmas}        }}
        workgroupBarrier();
    }}

{store_loop}}}
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

pub struct Conv2dF32 {
    pub cfg: Conv2dConfig,
}

impl Conv2dF32 {
    pub const fn new(cfg: Conv2dConfig) -> Self {
        Self { cfg }
    }

    pub const fn default_op() -> Self {
        Self {
            cfg: Conv2dConfig::DEFAULT,
        }
    }
}

impl Default for Conv2dF32 {
    fn default() -> Self {
        Self::default_op()
    }
}

impl Conv2dOp for Conv2dF32 {
    const KERNEL_ID: &'static str = "conv2d.f32";
    type Dtype = F32;
    const X: &'static str = "conv2d/x";
    const W: &'static str = "conv2d/w";
    const BIAS: &'static str = "conv2d/bias";
    const DIMS: &'static str = "conv2d/dims";
    const OUTPUT: &'static str = "conv2d/out";
    fn config(&self) -> &Conv2dConfig {
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
impl crate::conformance::OpTest for Conv2dF32 {
    fn dtypes(&self) -> &'static [crate::conformance::Dtype] {
        crate::conformance::DTYPES_FP32_ONLY
    }
    fn test_cases(&self) -> Vec<crate::conformance::TestCase> {
        use crate::conformance::{OpSpec, TestCase, linspace, t};
        vec![
            // Tiny 3x3 stride=1 pad=1 - kernel correctness baseline.
            TestCase {
                name: "conv2d_3x3_pad1",
                op: OpSpec::Conv2d {
                    kh: 3,
                    kw: 3,
                    pad_h: 1,
                    pad_w: 1,
                    stride_h: 1,
                    stride_w: 1,
                },
                inputs: vec![
                    t("x", [1, 4, 6, 6], linspace(-1.0, 1.0, false)),
                    t("w", [3, 4, 3, 3], linspace(-0.5, 0.5, false)),
                    t("bias", [3], linspace(-0.25, 0.25, false)),
                ],
            },
            // 1x1 pad=0 - residual shortcut path in VAE.
            TestCase {
                name: "conv2d_1x1",
                op: OpSpec::Conv2d {
                    kh: 1,
                    kw: 1,
                    pad_h: 0,
                    pad_w: 0,
                    stride_h: 1,
                    stride_w: 1,
                },
                inputs: vec![
                    t("x", [1, 4, 4, 4], linspace(-1.0, 1.0, false)),
                    t("w", [3, 4, 1, 1], linspace(-0.5, 0.5, false)),
                    t("bias", [3], linspace(-0.25, 0.25, true)),
                ],
            },
            // Wider cin to exercise a longer accumulation loop. Kept at
            // cin=16 so 144-term fp32 summation stays under 1e-5 tol.
            TestCase {
                name: "conv2d_3x3_widec",
                op: OpSpec::Conv2d {
                    kh: 3,
                    kw: 3,
                    pad_h: 1,
                    pad_w: 1,
                    stride_h: 1,
                    stride_w: 1,
                },
                inputs: vec![
                    t("x", [1, 16, 4, 4], linspace(-1.0, 1.0, false)),
                    t("w", [8, 16, 3, 3], linspace(-0.5, 0.5, false)),
                    t("bias", [8], linspace(-0.25, 0.25, false)),
                ],
            },
            // Spans multiple bm/bn/bk tiles with non-multiple edges:
            // cout=70 (> bm at default 64), spatial 11x13 = 143 cols,
            // K = 24*9 = 216 (> several bk strips, not a bk multiple).
            TestCase {
                name: "conv2d_3x3_multitile",
                op: OpSpec::Conv2d {
                    kh: 3,
                    kw: 3,
                    pad_h: 1,
                    pad_w: 1,
                    stride_h: 1,
                    stride_w: 1,
                },
                inputs: vec![
                    t("x", [1, 24, 11, 13], linspace(-1.0, 1.0, false)),
                    t("w", [70, 24, 3, 3], linspace(-0.05, 0.05, false)),
                    t("bias", [70], linspace(-0.25, 0.25, false)),
                ],
            },
            // Batch > 1 exercises the grid-z path (per-batch x/out bases).
            TestCase {
                name: "conv2d_3x3_batch2",
                op: OpSpec::Conv2d {
                    kh: 3,
                    kw: 3,
                    pad_h: 1,
                    pad_w: 1,
                    stride_h: 1,
                    stride_w: 1,
                },
                inputs: vec![
                    t("x", [2, 4, 5, 5], linspace(-1.0, 1.0, false)),
                    t("w", [6, 4, 3, 3], linspace(-0.5, 0.5, false)),
                    t("bias", [6], linspace(-0.25, 0.25, false)),
                ],
            },
        ]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a crate::conformance::OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_conv2d(Conv2dF32::new(self.cfg)))
    }
}
