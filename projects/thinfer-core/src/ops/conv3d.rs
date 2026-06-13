use super::{ActDtype, WeightDtype, WgslConfig};
use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
use crate::tensor::{ComputeDtype, F32};

/// `out = conv3d(x, weight) + bias` over NCTHW.
///
/// Shapes (row-major):
/// - `x:      [B, Cin, Tin, Hin, Win]`
/// - `weight: [Cout, Cin, kT, kH, kW]` (PyTorch `nn.Conv3d` layout)
/// - `bias:   [Cout]`
/// - `out:    [B, Cout, Tout, Hout, Wout]`
///
/// Geometry: `Tout = (Tin + pad_t - kT) / stride_t + 1` (and H/W analogous,
/// with symmetric `2*pad` along H/W). `pad_t` is a SINGLE front-padding amount
/// (causal time conv: padding is applied only on the low-time side; the back
/// is never padded). The caller assembles any feature-cache frames into `x`
/// before dispatch and passes the residual front pad here. H/W padding is the
/// usual symmetric `pad_h`/`pad_w` on both sides.
///
/// Implicit-GEMM kernel, identical in structure to `conv2d.rs`: the conv is
/// `weight[Cout, K] @ im2col[K, M]` with `K = Cin*kT*kH*kW` and
/// `M = Tout*Hout*Wout`, landing the output directly in NCTHW (row = Cout,
/// col = flattened spatio-temporal). The im2col matrix is virtual: the B-tile
/// loader gathers x on the fly (zero-fill at padding / OOB). Two-level tiling
/// (`bm x bn` per workgroup, `tm x tn` register block per thread); grid z is
/// batch. Bias is always required (pass a zero buffer if absent). Dilation and
/// groups are NOT supported (no Wan VAE conv uses them).
///
/// Layout: 0=X, 1=W, 2=Bias, 3=Out, 4=Uniform.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Conv3dConfig {
    /// Output-tile rows per workgroup (over Cout).
    pub bm: u32,
    /// Output-tile cols per workgroup (over spatio-temporal Tout*Hout*Wout).
    pub bn: u32,
    /// K-strip depth per shared-memory step (over Cin*kT*kH*kW).
    pub bk: u32,
    /// Register-block rows per thread.
    pub tm: u32,
    /// Register-block cols per thread.
    pub tn: u32,
}

impl Conv3dConfig {
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

impl Default for Conv3dConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

pub trait Conv3dOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const W: &'static str;
    const BIAS: &'static str;
    const DIMS: &'static str;
    const OUTPUT: &'static str;

    fn config(&self) -> &Conv3dConfig;
    fn wgsl(&self, cfg: &WgslConfig) -> String;
    fn layout() -> &'static [BindingLayout];

    /// Grid: x over spatio-temporal tiles, y over Cout tiles, z over batch.
    ///
    /// WebGPU caps each grid dimension at 65535. The spatial-tile count is the
    /// only dimension that can blow past it (a 540P clip has millions of output
    /// positions). When it does, spill the overflow into z alongside batch:
    /// `z = batch * x_chunks`, with x capped to a single chunk. The shader
    /// rebuilds `col_tile = (wid.z / batch) * num_workgroups.x + wid.x` and
    /// `batch_idx = wid.z % batch`. The single-chunk path (the only one parity
    /// sizes ever hit) is index-identical to the old flat grid.
    fn workgroups(&self, cout: u32, m_spatial: u32, batch: u32) -> [u32; 3] {
        let c = self.config();
        spill_grid(m_spatial.div_ceil(c.bn), cout.div_ceil(c.bm), batch, 65535)
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

/// Build the dispatch grid for `col_tiles` spatial tiles x `row_tiles` Cout
/// tiles x `batch`, keeping every dimension `<= max_dim` (WebGPU's per-dimension
/// cap). The spatial-tile count is the only one that can overflow; its excess
/// spills into z as `batch * x_chunks` (x capped to one chunk). The shader
/// rebuilds `col_tile = (wid.z / batch) * num_workgroups.x + wid.x`. Chunking
/// can overshoot `col_tiles` (when it does not divide evenly); the shader's
/// `col < m_total` guard drops the extra tiles.
fn spill_grid(col_tiles: u32, row_tiles: u32, batch: u32, max_dim: u32) -> [u32; 3] {
    let x_chunks = col_tiles.div_ceil(max_dim);
    let x = col_tiles.div_ceil(x_chunks);
    [x, row_tiles, batch * x_chunks]
}

pub struct Conv3dBufs<'a> {
    pub x: &'a BufRef,
    pub w: &'a BufRef,
    pub bias: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

pub(crate) fn dispatch_conv3d<O: Conv3dOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    op: &O,
    bufs: &Conv3dBufs<'_>,
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

/// Build the conv3d implicit-GEMM WGSL for `(Conv3dConfig, WgslConfig)`. Dtype
/// variants match `conv2d.rs`: `weight_dtype = Bf16` packs 2 bf16 per word;
/// `act_dtype = F16` stores x/out as `array<f16>` (scalar gathers, f32 accum).
fn build_wgsl(c: &Conv3dConfig, cfg: &WgslConfig) -> String {
    c.validate();
    assert!(
        !cfg.bf16_quant_writes,
        "conv3d has no bf16-quant-write mode"
    );
    let act_f16 = match cfg.act_dtype {
        ActDtype::F32 => false,
        ActDtype::F16 => true,
        other => panic!("conv3d does not support act dtype {other:?}"),
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

    // Same 32 KiB downlevel workgroup-storage cap as matmul/conv2d (naga does
    // not validate var<workgroup> size against the device limit).
    const MAX_WORKGROUP_STORAGE: u32 = 32768;
    let total_shared = (tile_a_size + tile_b_size) * 4;
    assert!(
        total_shared <= MAX_WORKGROUP_STORAGE,
        "conv3d kernel exceeds workgroup storage budget: bm={bm} bn={bn} bk={bk} \
         total={total_shared} > limit={MAX_WORKGROUP_STORAGE}",
    );

    let f16_prelude = if act_f16 { "enable f16;\n" } else { "" };
    let (x_elem_decl, out_elem_decl, load_x, store_expr) = if act_f16 {
        (
            "@group(0) @binding(0) var<storage, read> x: array<f16>;",
            "@group(0) @binding(3) var<storage, read_write> out: array<f16>;",
            "fn load_x(i: u32) -> f32 { return f32(x[i]); }\n",
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
        WeightDtype::F16 => unreachable!("conv3d does not consume f16 weights"),
        WeightDtype::Quant(_) => unreachable!("conv3d does not consume quant weights"),
    };

    // Explicit-scalar register block (kernel register rule), identical to
    // conv2d/matmul: accumulators + inner FMAs + store loop unrolled over
    // tm x tn.
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
    b: u32, cin: u32, cout: u32, t_in: u32,
    h_in: u32, w_in: u32, t_out: u32, h_out: u32,
    w_out: u32, kt: u32, kh: u32, kw: u32,
    pad_t: u32, pad_h: u32, pad_w: u32, stride_t: u32,
    stride_h: u32, stride_w: u32, _pad0: u32, _pad1: u32,
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
    @builtin(num_workgroups) ng: vec3<u32>,
) {{
    let tid: u32 = lid.y * {wg_x}u + lid.x;
    // z carries (spatial-tile chunk, batch): chunk = wid.z / b, batch = wid.z % b.
    // Single-chunk grids leave chunk == 0, so col_tile == wid.x as before.
    let col_tile: u32 = (wid.z / u.b) * ng.x + wid.x;
    let bnum: u32 = wid.z % u.b;
    let bm0: u32 = wid.y * BM;
    let bn0: u32 = col_tile * BN;
    let khkw: u32 = u.kh * u.kw;
    let ktkhkw: u32 = u.kt * khkw;
    let kdim: u32 = u.cin * ktkhkw;
    let hw_out: u32 = u.h_out * u.w_out;
    let m_total: u32 = u.t_out * hw_out;
    let thw_in: u32 = u.t_in * u.h_in * u.w_in;
    let hw_in: u32 = u.h_in * u.w_in;
    let x_base: u32 = bnum * u.cin * thw_in;
    let out_base: u32 = bnum * u.cout * m_total;

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
        // B-tile: virtual im2col [K, M]; gr = k -> (ci, dt, dh, dw),
        // gc = spatio-temporal -> (to, ho, wo). Zero-fill at padding / OOB.
        // bc is the fast index so consecutive threads gather consecutive wi.
        for (var s: u32 = 0u; s < {b_loads_per_thread}u; s = s + 1u) {{
            let idx: u32 = s * THREADS + tid;
            if (idx < BK * BN) {{
                let br: u32 = idx / BN;
                let bc: u32 = idx % BN;
                let gr: u32 = k0 + br;
                let gc: u32 = bn0 + bc;
                var v: f32 = 0.0;
                if (gr < kdim && gc < m_total) {{
                    let ci: u32 = gr / ktkhkw;
                    let kr: u32 = gr - ci * ktkhkw;
                    let dt: u32 = kr / khkw;
                    let kr2: u32 = kr - dt * khkw;
                    let dh: u32 = kr2 / u.kw;
                    let dw: u32 = kr2 - dh * u.kw;
                    let to: u32 = gc / hw_out;
                    let gc2: u32 = gc - to * hw_out;
                    let ho: u32 = gc2 / u.w_out;
                    let wo: u32 = gc2 - ho * u.w_out;
                    let ti: i32 = i32(to * u.stride_t + dt) - i32(u.pad_t);
                    let hi: i32 = i32(ho * u.stride_h + dh) - i32(u.pad_h);
                    let wi: i32 = i32(wo * u.stride_w + dw) - i32(u.pad_w);
                    if (ti >= 0 && ti < i32(u.t_in) && hi >= 0 && hi < i32(u.h_in)
                        && wi >= 0 && wi < i32(u.w_in)) {{
                        v = load_x(x_base + ci * thw_in + u32(ti) * hw_in
                            + u32(hi) * u.w_in + u32(wi));
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

pub struct Conv3dF32 {
    pub cfg: Conv3dConfig,
}

impl Conv3dF32 {
    pub const fn new(cfg: Conv3dConfig) -> Self {
        Self { cfg }
    }

    pub const fn default_op() -> Self {
        Self {
            cfg: Conv3dConfig::DEFAULT,
        }
    }
}

impl Default for Conv3dF32 {
    fn default() -> Self {
        Self::default_op()
    }
}

impl Conv3dOp for Conv3dF32 {
    const KERNEL_ID: &'static str = "conv3d.f32";
    type Dtype = F32;
    const X: &'static str = "conv3d/x";
    const W: &'static str = "conv3d/w";
    const BIAS: &'static str = "conv3d/bias";
    const DIMS: &'static str = "conv3d/dims";
    const OUTPUT: &'static str = "conv3d/out";
    fn config(&self) -> &Conv3dConfig {
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
impl crate::conformance::OpTest for Conv3dF32 {
    fn dtypes(&self) -> &'static [crate::conformance::Dtype] {
        crate::conformance::DTYPES_FP32_ONLY
    }
    fn test_cases(&self) -> Vec<crate::conformance::TestCase> {
        use crate::conformance::{OpSpec, TestCase, linspace, t};
        vec![
            // 3x3x3 symmetric pad - general resnet conv geometry. pad_t=2 with
            // the causal-front convention keeps Tout==Tin for the kt=3 case
            // when the caller front-pads by kt-1.
            TestCase {
                name: "conv3d_3x3x3_pad",
                op: OpSpec::Conv3d {
                    kt: 3,
                    kh: 3,
                    kw: 3,
                    pad_t: 2,
                    pad_h: 1,
                    pad_w: 1,
                    stride_t: 1,
                    stride_h: 1,
                    stride_w: 1,
                },
                inputs: vec![
                    t("x", [1, 4, 3, 5, 5], linspace(-1.0, 1.0, false)),
                    t("w", [3, 4, 3, 3, 3], linspace(-0.3, 0.3, false)),
                    t("bias", [3], linspace(-0.25, 0.25, false)),
                ],
            },
            // 3x1x1 time-only conv (temporal resamplers).
            TestCase {
                name: "conv3d_3x1x1_time",
                op: OpSpec::Conv3d {
                    kt: 3,
                    kh: 1,
                    kw: 1,
                    pad_t: 2,
                    pad_h: 0,
                    pad_w: 0,
                    stride_t: 1,
                    stride_h: 1,
                    stride_w: 1,
                },
                inputs: vec![
                    t("x", [1, 6, 4, 3, 3], linspace(-1.0, 1.0, false)),
                    t("w", [12, 6, 3, 1, 1], linspace(-0.2, 0.2, false)),
                    t("bias", [12], linspace(-0.1, 0.1, false)),
                ],
            },
            // 1x1x1 pointwise (quant/post-quant + channel-change shortcut).
            TestCase {
                name: "conv3d_1x1x1",
                op: OpSpec::Conv3d {
                    kt: 1,
                    kh: 1,
                    kw: 1,
                    pad_t: 0,
                    pad_h: 0,
                    pad_w: 0,
                    stride_t: 1,
                    stride_h: 1,
                    stride_w: 1,
                },
                inputs: vec![
                    t("x", [1, 8, 2, 4, 4], linspace(-1.0, 1.0, false)),
                    t("w", [5, 8, 1, 1, 1], linspace(-0.5, 0.5, true)),
                    t("bias", [5], linspace(-0.25, 0.25, false)),
                ],
            },
            // Temporal stride 2 (downsample3d time_conv).
            TestCase {
                name: "conv3d_3x1x1_stride_t2",
                op: OpSpec::Conv3d {
                    kt: 3,
                    kh: 1,
                    kw: 1,
                    pad_t: 0,
                    pad_h: 0,
                    pad_w: 0,
                    stride_t: 2,
                    stride_h: 1,
                    stride_w: 1,
                },
                inputs: vec![
                    t("x", [1, 4, 5, 3, 3], linspace(-1.0, 1.0, false)),
                    t("w", [4, 4, 3, 1, 1], linspace(-0.2, 0.2, false)),
                    t("bias", [4], linspace(-0.1, 0.1, false)),
                ],
            },
            // Multi-tile: cout=70 (> bm), wide K, non-multiple spatial edges.
            TestCase {
                name: "conv3d_3x3x3_multitile",
                op: OpSpec::Conv3d {
                    kt: 3,
                    kh: 3,
                    kw: 3,
                    pad_t: 2,
                    pad_h: 1,
                    pad_w: 1,
                    stride_t: 1,
                    stride_h: 1,
                    stride_w: 1,
                },
                inputs: vec![
                    t("x", [1, 16, 3, 7, 5], linspace(-1.0, 1.0, false)),
                    t("w", [70, 16, 3, 3, 3], linspace(-0.05, 0.05, false)),
                    t("bias", [70], linspace(-0.25, 0.25, false)),
                ],
            },
        ]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a crate::conformance::OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_conv3d(Conv3dF32::new(self.cfg)))
    }
}

#[cfg(test)]
mod tests {
    use super::spill_grid;
    use std::collections::HashSet;

    /// Replay the shader's index reconstruction over the whole grid and assert
    /// it visits every `(col_tile, batch_idx)` exactly once. The GPU conformance
    /// cases only ever hit the single-chunk path (a >65535-tile buffer is too
    /// large to reference on CPU), so this covers the spill arithmetic directly.
    fn assert_covers(col_tiles: u32, batch: u32, max_dim: u32) {
        let [x, _row, z] = spill_grid(col_tiles, 1, batch, max_dim);
        // x (the split dimension) is always capped; z = batch * x_chunks is only
        // bounded by the cap at realistic sizes (x_chunks stays single digits),
        // so the z cap is asserted separately in `spill_grid_respects_dimension_cap`.
        assert!(x <= max_dim, "split dim over cap: x={x} > {max_dim}");

        let mut seen: HashSet<(u32, u32)> = HashSet::new();
        for wid_z in 0..z {
            // Shader: col_tile = (wid.z / b) * num_workgroups.x + wid.x;
            //         batch_idx = wid.z % b.
            let chunk = wid_z / batch;
            let bnum = wid_z % batch;
            for wid_x in 0..x {
                let col_tile = chunk * x + wid_x;
                if col_tile >= col_tiles {
                    continue; // shader's `col < m_total` guard drops overshoot.
                }
                assert!(
                    seen.insert((col_tile, bnum)),
                    "duplicate visit (col={col_tile}, b={bnum})"
                );
            }
        }
        assert_eq!(
            seen.len() as u32,
            col_tiles * batch,
            "missing tiles: covered {} of {}",
            seen.len(),
            col_tiles * batch
        );
    }

    #[test]
    fn spill_grid_covers_every_tile_exactly_once() {
        // Single-chunk (the parity path) and multi-chunk, with batch 1 and >1.
        // Small max_dim forces chunking without huge allocations; the arithmetic
        // is identical at the real 65535 cap.
        for &max_dim in &[3u32, 8, 65535] {
            for &batch in &[1u32, 2, 3] {
                for &col_tiles in &[1u32, 2, 7, 8, 9, 17, 64, 100] {
                    assert_covers(col_tiles, batch, max_dim);
                }
            }
        }
    }

    #[test]
    fn spill_grid_respects_dimension_cap() {
        // A 540P-scale spatial-tile count must not place anything over the cap.
        let [x, _row, z] = spill_grid(391_680, 1, 1, 65535);
        assert!(x <= 65535 && z <= 65535);
        assert!(x * z >= 391_680, "grid does not cover all tiles");
    }
}
