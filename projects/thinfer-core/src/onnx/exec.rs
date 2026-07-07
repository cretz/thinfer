//! ONNX graph executor on the wgpu backend.
//!
//! [`OnnxModel::load`] parses + plans once for a fixed input shape, compiles the
//! kernel pipelines, and uploads every constant a compute node reads (folding
//! BatchNormalization to a per-channel affine). [`OnnxModel::run`] uploads the
//! inputs, records one command buffer of dispatches over pool-free GPU buffers,
//! submits, and reads the graph outputs back. Weights stay resident across runs
//! (per-frame video re-runs only re-allocate activations).

use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use crate::backend::{
    Backend, Binding, BindingKind, BindingLayout, WgpuBackend, WgpuError, WgpuPipeline,
};
use crate::mem::VramCategory;
use crate::tensor::GpuBufferId;

use super::kernels;
use super::proto::Graph;
use super::shape::{Plan, PlanError, Step};

#[derive(Debug)]
pub enum OnnxError {
    Plan(PlanError),
    Wgpu(WgpuError),
    Unsupported(String),
    Io(String),
}
impl std::fmt::Display for OnnxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OnnxError::Plan(e) => write!(f, "{e}"),
            OnnxError::Wgpu(e) => write!(f, "onnx wgpu: {e:?}"),
            OnnxError::Unsupported(s) => write!(f, "onnx unsupported: {s}"),
            OnnxError::Io(s) => write!(f, "onnx io: {s}"),
        }
    }
}
impl std::error::Error for OnnxError {}
impl From<PlanError> for OnnxError {
    fn from(e: PlanError) -> Self {
        OnnxError::Plan(e)
    }
}
impl From<WgpuError> for OnnxError {
    fn from(e: WgpuError) -> Self {
        OnnxError::Wgpu(e)
    }
}

/// Owned GPU buffer that frees on drop.
struct GpuBuf {
    backend: Arc<WgpuBackend>,
    id: GpuBufferId,
    len: u64,
}
impl GpuBuf {
    fn new(backend: &Arc<WgpuBackend>, bytes: u64) -> Result<Self, WgpuError> {
        let bytes = bytes.max(4);
        let id = backend.allocate_in(bytes, VramCategory::Workspace)?;
        Ok(Self {
            backend: Arc::clone(backend),
            id,
            len: bytes,
        })
    }
    fn binding(&self, slot: u32) -> Binding {
        Binding {
            slot,
            buffer: self.id,
            offset: 0,
            size: self.len,
        }
    }
    fn bufref(&self) -> crate::backend::BufRef {
        crate::backend::BufRef::new(self.id, self.len)
    }
}
impl Drop for GpuBuf {
    fn drop(&mut self) {
        self.backend.free(self.id);
    }
}

fn layout(n_read: usize) -> Vec<BindingLayout> {
    let mut l: Vec<BindingLayout> = (0..n_read)
        .map(|i| BindingLayout {
            slot: i as u32,
            kind: BindingKind::StorageRead,
        })
        .collect();
    l.push(BindingLayout {
        slot: n_read as u32,
        kind: BindingKind::StorageReadWrite,
    });
    l.push(BindingLayout {
        slot: n_read as u32 + 1,
        kind: BindingKind::Uniform,
    });
    l
}

/// i8 DP4A conv pipeline set: the tiled i8 conv plus the three-step activation
/// quantizer (clear the atomic accumulator, atomic max-abs, quantize+pack). Tile
/// shape fixed at the f32 DEFAULT (bm/bn/bk/tm/tn = 64/64/32/4/4).
struct I8ConvPipelines {
    clear: WgpuPipeline,
    maxabs: WgpuPipeline,
    quant: WgpuPipeline,
    conv: WgpuPipeline,
}

impl I8ConvPipelines {
    async fn compile(backend: &WgpuBackend) -> Result<Self, WgpuError> {
        use super::i8conv as k;
        Ok(Self {
            clear: backend
                .create_pipeline("onnx_i8_clear", k::CLEAR_U32, "main", &k::clear_layout())
                .await?,
            maxabs: backend
                .create_pipeline(
                    "onnx_i8_minmax",
                    &k::act_minmax_wgsl(),
                    "main",
                    &k::act_maxabs_layout(),
                )
                .await?,
            quant: backend
                .create_pipeline(
                    "onnx_i8_quant",
                    &k::act_quant_wgsl(),
                    "main",
                    &k::act_quant_layout(),
                )
                .await?,
            conv: backend
                .create_pipeline(
                    "onnx_i8_conv",
                    &k::build_conv_i8_wgsl(64, 64, 32, 4, 4),
                    "main",
                    &k::conv_i8_layout(),
                )
                .await?,
        })
    }
}

/// A tuned implicit-GEMM conv2d pipeline + the tile config it was built with
/// (the op carries the tile shape needed to pick the dispatch grid).
struct TiledConv {
    pipeline: WgpuPipeline,
    op: crate::ops::Conv2dF32,
}

impl TiledConv {
    async fn compile(
        backend: &WgpuBackend,
        label: &str,
        tile: crate::ops::Conv2dConfig,
    ) -> Result<Self, WgpuError> {
        use crate::ops::{Conv2dF32, Conv2dOp, WgslConfig};
        // f32 weights: the HyperSwap convs are compute-bound (f32-FMA-limited)
        // on this GPU, so bf16 weights only add unpack cost (measured slower)
        // and lose precision. Keep f32; the ALU win would need i8 DP4A.
        let op = Conv2dF32::new(tile);
        let pipeline = backend
            .create_pipeline(
                label,
                &op.wgsl(&WgslConfig::FP32),
                "main",
                <Conv2dF32 as Conv2dOp>::layout(),
            )
            .await?;
        Ok(Self { pipeline, op })
    }
}

/// Compiled kernel pipelines (one per op family; no compile-time variants).
struct Pipelines {
    /// Direct conv (groups / dilation): SCRFD depthwise etc.
    conv2d: WgpuPipeline,
    /// Tuned implicit-GEMM conv (group=1, dilation=1): the HyperSwap/ArcFace
    /// bulk. Three tile regimes by output shape (see `pick_tiled_conv`).
    conv2d_tiled: TiledConv,
    conv2d_tiled_wide: TiledConv,
    conv2d_tiled_small_n: TiledConv,
    /// Opt-in i8 DP4A conv (`THINFER_ONNX_I8_CONV`) + its activation quantizer
    /// (clear -> max-abs -> quant). `None` unless the gate is set.
    conv_i8: Option<I8ConvPipelines>,
    convt2d: WgpuPipeline,
    gemm: WgpuPipeline,
    instance_norm: WgpuPipeline,
    channel_affine: WgpuPipeline,
    prelu: WgpuPipeline,
    unary: WgpuPipeline,
    binary: WgpuPipeline,
    expand: WgpuPipeline,
    transpose: WgpuPipeline,
    depth_to_space: WgpuPipeline,
    resize: WgpuPipeline,
    maxpool: WgpuPipeline,
    zero_upsample: WgpuPipeline,
    concat2: WgpuPipeline,
    slice: WgpuPipeline,
    global_avg_pool: WgpuPipeline,
    reduce_sum: WgpuPipeline,
    pad: WgpuPipeline,
    const_fill: WgpuPipeline,
    tile: WgpuPipeline,
    gather: WgpuPipeline,
}

impl Pipelines {
    async fn compile(backend: &WgpuBackend, i8_conv: bool) -> Result<Self, WgpuError> {
        let p = |name: &'static str, src: &'static str, n_read: usize| {
            let lay = layout(n_read);
            async move { backend.create_pipeline(name, src, "main", &lay).await }
        };
        use crate::ops::Conv2dConfig;
        // Tile regimes (mirror the VAE's): default, wide (large spatial), and
        // small-N (tiny Cout like the 3-channel output conv).
        const WIDE: Conv2dConfig = Conv2dConfig {
            bm: 64,
            bn: 128,
            bk: 32,
            tm: 4,
            tn: 8,
            prefetch: false,
        };
        const SMALL_N: Conv2dConfig = Conv2dConfig {
            bm: 4,
            bn: 128,
            bk: 32,
            tm: 1,
            tn: 2,
            prefetch: false,
        };
        // A/B lever: double-buffer the default-tile conv so global-load latency
        // overlaps the FMA loop. The HyperSwap bottleneck convs (cout>=64,
        // spatial < 65536) route to this tile; its 64x64x32 double-buffered
        // budget is exactly 32 KiB.
        let default_tile = Conv2dConfig {
            prefetch: std::env::var_os("THINFER_ONNX_CONV_PREFETCH").is_some(),
            ..Conv2dConfig::DEFAULT
        };
        Ok(Self {
            conv2d: p("onnx_conv2d", kernels::CONV2D, 3).await?,
            conv2d_tiled: TiledConv::compile(backend, "onnx_conv2d_tiled", default_tile).await?,
            conv2d_tiled_wide: TiledConv::compile(backend, "onnx_conv2d_tiled_wide", WIDE).await?,
            conv2d_tiled_small_n: TiledConv::compile(backend, "onnx_conv2d_tiled_small_n", SMALL_N)
                .await?,
            conv_i8: if i8_conv {
                Some(I8ConvPipelines::compile(backend).await?)
            } else {
                None
            },
            convt2d: p("onnx_convt2d", kernels::CONVT2D, 3).await?,
            gemm: p("onnx_gemm", kernels::GEMM, 3).await?,
            instance_norm: p("onnx_instance_norm", kernels::INSTANCE_NORM, 3).await?,
            channel_affine: p("onnx_channel_affine", kernels::CHANNEL_AFFINE, 3).await?,
            prelu: p("onnx_prelu", kernels::PRELU, 2).await?,
            unary: p("onnx_unary", kernels::UNARY, 1).await?,
            binary: p("onnx_binary", kernels::BINARY, 2).await?,
            expand: p("onnx_expand", kernels::EXPAND, 1).await?,
            transpose: p("onnx_transpose", kernels::TRANSPOSE, 1).await?,
            depth_to_space: p("onnx_depth_to_space", kernels::DEPTH_TO_SPACE, 1).await?,
            resize: p("onnx_resize", kernels::RESIZE, 1).await?,
            maxpool: p("onnx_maxpool", kernels::MAXPOOL, 1).await?,
            zero_upsample: p("onnx_zero_upsample", kernels::ZERO_UPSAMPLE, 1).await?,
            concat2: p("onnx_concat2", kernels::CONCAT2, 2).await?,
            slice: p("onnx_slice", kernels::SLICE, 1).await?,
            global_avg_pool: p("onnx_global_avg_pool", kernels::GLOBAL_AVG_POOL, 1).await?,
            reduce_sum: p("onnx_reduce_sum", kernels::REDUCE_SUM, 1).await?,
            pad: p("onnx_pad", kernels::PAD, 1).await?,
            const_fill: p("onnx_const_fill", kernels::CONST_FILL, 0).await?,
            tile: p("onnx_tile", kernels::TILE, 1).await?,
            gather: p("onnx_gather", kernels::GATHER, 2).await?,
        })
    }
}

/// Packs a kernel uniform: push u32/f32 then pad to a 16-byte multiple (WGSL
/// uniform buffers must be 16-aligned in size).
#[derive(Default)]
struct Uni(Vec<u8>);
impl Uni {
    fn u32(mut self, v: u32) -> Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn f32(mut self, v: f32) -> Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn finish(mut self) -> Vec<u8> {
        while !self.0.len().is_multiple_of(16) {
            self.0.push(0);
        }
        self.0
    }
}

pub struct OnnxModel {
    backend: Arc<WgpuBackend>,
    graph: Graph,
    plan: Plan,
    pipelines: Pipelines,
    /// Elementwise-chain fusion: analysis + one codegen'd pipeline per chain
    /// (aligned with `fusion.chains`). Bit-exact; collapses the tail of tiny
    /// elementwise/activation dispatches into single-read-single-write kernels.
    fusion: super::fusion::Fusion,
    fused_pipelines: Vec<WgpuPipeline>,
    /// Resident constant buffers (weights, biases, folded BN affine, etc.),
    /// keyed by value name (BN affine uses synthetic `name__bn_a/b`).
    consts: HashMap<String, Rc<GpuBuf>>,
    /// Per-conv-node i8-quantized weights, present only when the i8 gate is on
    /// and the conv is eligible (group=1, dil=1). Keyed by node index.
    i8_weights: HashMap<usize, I8Weights>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
}

/// Load-time i8-quantized conv weight: packed weights `[Cout, ceil(K/4)]` (4 i8
/// per u32) + a per-Cout f32 scale, both resident. `kw4` = ceil(K/4).
struct I8Weights {
    packed: Rc<GpuBuf>,
    scale: Rc<GpuBuf>,
    qsum: Rc<GpuBuf>,
    kw4: u32,
}

fn pad4(shape: &[i64]) -> [u32; 4] {
    // Right-align into 4D (leading dims become 1), matching the kernels' NCHW
    // broadcast index math.
    let mut out = [1u32; 4];
    for (i, &d) in shape.iter().rev().take(4).enumerate() {
        out[3 - i] = d as u32;
    }
    out
}

fn numel(shape: &[i64]) -> u64 {
    shape.iter().product::<i64>().max(1) as u64
}

impl OnnxModel {
    /// Parse + plan + compile + upload constants. `input_shapes` binds each
    /// graph input to its fixed runtime shape.
    pub async fn load(
        backend: Arc<WgpuBackend>,
        onnx_bytes: &[u8],
        input_shapes: &HashMap<String, Vec<i64>>,
    ) -> Result<Self, OnnxError> {
        Self::load_i8(backend, onnx_bytes, input_shapes, false).await
    }

    /// As [`load`], with an explicit opt-in to the i8 DP4A conv path. Caller-
    /// scoped (not a process-wide env) so it can be enabled for HyperSwap alone
    /// and never for the detector/embedder, whose i8 error is unvalidated.
    pub async fn load_i8(
        backend: Arc<WgpuBackend>,
        onnx_bytes: &[u8],
        input_shapes: &HashMap<String, Vec<i64>>,
        i8_conv: bool,
    ) -> Result<Self, OnnxError> {
        let graph =
            super::proto::parse_model(onnx_bytes).map_err(|e| OnnxError::Io(e.to_string()))?;
        let plan = super::shape::plan(&graph, input_shapes)?;
        let pipelines = Pipelines::compile(&backend, i8_conv).await?;

        // Elementwise-chain fusion (bit-exact; every model). One codegen'd
        // pipeline per chain, compiled once here. `THINFER_ONNX_NO_FUSION` opts
        // out (A/B + safety valve).
        let fusion = if std::env::var_os("THINFER_ONNX_NO_FUSION").is_some() {
            super::fusion::Fusion::empty()
        } else {
            super::fusion::analyze(&graph, &plan)
        };
        let mut fused_pipelines = Vec::with_capacity(fusion.chains.len());
        for (i, chain) in fusion.chains.iter().enumerate() {
            let wgsl = super::fusion::build_wgsl(chain);
            let lay = layout(1 + chain.sides.len());
            let pipe = backend
                .create_pipeline(&format!("onnx_fused_{i}"), &wgsl, "main", &lay)
                .await?;
            fused_pipelines.push(pipe);
        }

        let input_names: Vec<String> = graph
            .inputs
            .iter()
            .filter(|vi| !graph.initializers.iter().any(|t| t.name == vi.name))
            .map(|vi| vi.name.clone())
            .collect();
        let output_names: Vec<String> = graph.outputs.iter().map(|o| o.name.clone()).collect();

        let mut model = OnnxModel {
            backend,
            graph,
            plan,
            pipelines,
            fusion,
            fused_pipelines,
            consts: HashMap::new(),
            i8_weights: HashMap::new(),
            input_names,
            output_names,
        };
        model.upload_constants()?;
        if model.pipelines.conv_i8.is_some() {
            model.quantize_i8_conv_weights()?;
        }
        Ok(model)
    }

    fn upload_f32(&self, _name: &str, data: &[f32]) -> Result<Rc<GpuBuf>, WgpuError> {
        let buf = GpuBuf::new(&self.backend, (data.len() * 4) as u64)?;
        self.backend
            .write_buffer(buf.id, 0, bytemuck::cast_slice(data))?;
        Ok(Rc::new(buf))
    }

    /// Upload every constant a compute node reads as a data operand, plus the
    /// folded BatchNormalization affine. Shape/param consts (Resize sizes,
    /// Expand target) are consumed at plan time and never uploaded.
    fn upload_constants(&mut self) -> Result<(), OnnxError> {
        // Collect (node_idx) of compute steps.
        let compute: Vec<usize> = self
            .plan
            .steps
            .iter()
            .filter_map(|s| match s {
                Step::Compute { node_idx } => Some(*node_idx),
                _ => None,
            })
            .collect();

        let mut to_upload: HashSet<String> = HashSet::new();
        let mut bn_nodes: Vec<usize> = Vec::new();
        let mut ct_nodes: Vec<usize> = Vec::new();
        for &idx in &compute {
            let node = &self.graph.nodes[idx];
            match node.op_type.as_str() {
                "BatchNormalization" => bn_nodes.push(idx),
                // ConvTranspose weight is transformed (flip+transpose) below for
                // the tiled-conv path; only its bias uploads via the generic path.
                "ConvTranspose" if ct_tiled_eligible(node) => {
                    ct_nodes.push(idx);
                    if let Some(b) = node.inputs.get(2).filter(|s| !s.is_empty())
                        && self.plan.consts.contains_key(b)
                    {
                        to_upload.insert(b.clone());
                    }
                }
                _ => {
                    for inp in data_operands(&node.op_type, &node.inputs) {
                        if self.plan.consts.contains_key(inp) {
                            to_upload.insert(inp.to_string());
                        }
                    }
                }
            }
        }

        // A metadata View (Reshape/Expand/Squeeze/etc.) aliasing a constant source
        // (e.g. StyleGAN's learned constant input -> Expand) needs that const on
        // the GPU too, or the alias resolves to nothing. Views are not compute
        // nodes, so add their const sources explicitly.
        for step in &self.plan.steps {
            if let Step::View { src, .. } = step
                && self.plan.consts.contains_key(src)
            {
                to_upload.insert(src.clone());
            }
        }

        let names: Vec<String> = to_upload.into_iter().collect();
        for name in names {
            let data = self.plan.consts[&name].to_f32().into_owned();
            let buf = self.upload_f32(&name, &data)?;
            self.consts.insert(name, buf);
        }

        // Fold BatchNorm to per-channel affine a = gamma/sqrt(var+eps),
        // b = beta - mean*a. Inputs: X, scale(gamma), B(beta), mean, var.
        for idx in bn_nodes {
            let node = &self.graph.nodes[idx];
            let eps = node.attr_f("epsilon", 1e-5);
            let g = self.const_f32(&node.inputs[1])?;
            let beta = self.const_f32(&node.inputs[2])?;
            let mean = self.const_f32(&node.inputs[3])?;
            let var = self.const_f32(&node.inputs[4])?;
            let a: Vec<f32> = g
                .iter()
                .zip(&var)
                .map(|(&gm, &v)| gm / (v + eps).sqrt())
                .collect();
            let b: Vec<f32> = beta
                .iter()
                .zip(&mean)
                .zip(&a)
                .map(|((&bt, &mn), &ai)| bt - mn * ai)
                .collect();
            let out = node.outputs[0].clone();
            let ba = self.upload_f32(&out, &a)?;
            let bb = self.upload_f32(&out, &b)?;
            self.consts.insert(format!("{out}__bn_a"), ba);
            self.consts.insert(format!("{out}__bn_b"), bb);
        }

        // ConvTranspose -> tiled conv: transform the weight `[Cin, Cout, kH, kW]`
        // into a conv weight `[Cout, Cin, kH, kW]` that is spatially flipped and
        // Cin/Cout-transposed. Stored under `{out}__ct_w`.
        for idx in ct_nodes {
            let node = &self.graph.nodes[idx];
            let wname = &node.inputs[1];
            let dims = self.plan.shape_of(wname)?.to_vec(); // [Cin, Cout, kH, kW]
            let (cin, cout, kh, kw) = (
                dims[0] as usize,
                dims[1] as usize,
                dims[2] as usize,
                dims[3] as usize,
            );
            let src = self.const_f32(wname)?; // [Cin, Cout, kH, kW]
            let mut dst = vec![0.0f32; cout * cin * kh * kw];
            for ci in 0..cin {
                for co in 0..cout {
                    for r in 0..kh {
                        for s in 0..kw {
                            let si = ((ci * cout + co) * kh + r) * kw + s;
                            // flip spatial, transpose (ci,co): W'[co,ci,kh-1-r,kw-1-s].
                            let di = ((co * cin + ci) * kh + (kh - 1 - r)) * kw + (kw - 1 - s);
                            dst[di] = src[si];
                        }
                    }
                }
            }
            let out = node.outputs[0].clone();
            let buf = self.upload_f32(&out, &dst)?;
            self.consts.insert(format!("{out}__ct_w"), buf);
        }
        Ok(())
    }

    /// Quantize every eligible (group=1, dil=1) conv's weight to symmetric i8
    /// once at load, storing packed weights + per-Cout scales resident. Runs
    /// only under the i8 gate. The f32 weight const stays uploaded (unused on
    /// the i8 path; a small VRAM cost we accept for load simplicity).
    fn quantize_i8_conv_weights(&mut self) -> Result<(), OnnxError> {
        let eligible: Vec<usize> = self
            .plan
            .steps
            .iter()
            .filter_map(|s| match s {
                Step::Compute { node_idx } => Some(*node_idx),
                _ => None,
            })
            .filter(|&idx| {
                let n = &self.graph.nodes[idx];
                if n.op_type != "Conv"
                    || n.attr_i("group", 1) != 1
                    || !n
                        .attr_ints("dilations")
                        .map(|d| d == [1, 1])
                        .unwrap_or(true)
                {
                    return false;
                }
                // Keep the quality-sensitive input (few in-channels) and output
                // (few out-channels) convs in f32; i8 only the interior bulk.
                let ws = match self.plan.shape_of(&n.inputs[1]) {
                    Ok(s) => s,
                    Err(_) => return false,
                };
                let cin = ws[1];
                let cout = ws[0];
                if cin <= 4 || cout <= 4 {
                    return false;
                }
                // Escape hatch: THINFER_ONNX_I8_MAXHW caps the output H*W eligible
                // for i8 (keep the largest feature maps in f32). Default off --
                // measured net-negative: excluding the big 256^2 convs removes the
                // bulk of the i8 win yet barely changes the grain (it propagates
                // from every layer). Left as a knob, not a default.
                let cap: i64 = std::env::var("THINFER_ONNX_I8_MAXHW")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(i64::MAX);
                let os = match self.plan.shape_of(&n.outputs[0]) {
                    Ok(s) => s,
                    Err(_) => return false,
                };
                os[2] * os[3] <= cap
            })
            .collect();
        for idx in eligible {
            let node = &self.graph.nodes[idx];
            let wname = &node.inputs[1];
            let dims = self.plan.shape_of(wname)?.to_vec(); // [Cout, Cin, kH, kW]
            let cout = dims[0] as usize;
            let k = (dims[1] * dims[2] * dims[3]) as usize;
            let w = self.const_f32(wname)?;
            let (packed, scale, qsum) = super::i8conv::quantize_weight(&w, cout, k);
            let packed_buf = GpuBuf::new(&self.backend, (packed.len() * 4) as u64)?;
            self.backend
                .write_buffer(packed_buf.id, 0, bytemuck::cast_slice(&packed))?;
            let scale_buf = GpuBuf::new(&self.backend, (scale.len() * 4) as u64)?;
            self.backend
                .write_buffer(scale_buf.id, 0, bytemuck::cast_slice(&scale))?;
            let qsum_buf = GpuBuf::new(&self.backend, (qsum.len() * 4) as u64)?;
            self.backend
                .write_buffer(qsum_buf.id, 0, bytemuck::cast_slice(&qsum))?;
            self.i8_weights.insert(
                idx,
                I8Weights {
                    packed: Rc::new(packed_buf),
                    scale: Rc::new(scale_buf),
                    qsum: Rc::new(qsum_buf),
                    kw4: k.div_ceil(4) as u32,
                },
            );
        }
        Ok(())
    }

    fn const_f32(&self, name: &str) -> Result<Vec<f32>, OnnxError> {
        self.plan
            .consts
            .get(name)
            .map(|d| d.to_f32().into_owned())
            .ok_or_else(|| OnnxError::Unsupported(format!("missing const {name}")))
    }

    /// Run one forward. `inputs` maps each input name to its row-major f32 data
    /// (shape per `load`'s `input_shapes`). Returns each output's (shape, data).
    pub async fn run(
        &self,
        inputs: &HashMap<String, Vec<f32>>,
    ) -> Result<HashMap<String, (Vec<i64>, Vec<f32>)>, OnnxError> {
        // Per-run activation buffers (Rc so View aliases share one allocation).
        let mut acts: HashMap<String, Rc<GpuBuf>> = HashMap::new();
        for (name, data) in inputs {
            acts.insert(name.clone(), self.upload_f32(name, data)?);
        }

        let mut encoder = self.backend.create_command_encoder();
        // Keep uniforms + intermediate buffers alive until submit completes.
        let mut keepalive: Vec<Rc<GpuBuf>> = Vec::new();

        // Diagnostic: when set, submit+await after each compute op and tally GPU
        // wall time per op type. Adds per-submit overhead, so it's for relative
        // ranking only - never on in the normal single-submit path.
        let opprof = std::env::var_os("THINFER_ONNX_OPPROF").is_some();
        // When set, Conv/ConvTranspose are bucketed by shape signature
        // (kernel/stride/cin/cout/spatial) so the per-shape cost is visible.
        let opprof_shapes = std::env::var_os("THINFER_ONNX_OPPROF_SHAPES").is_some();
        let mut prof: std::collections::BTreeMap<String, (f64, u32)> =
            std::collections::BTreeMap::new();

        for step in &self.plan.steps {
            match step {
                Step::View { out, src, .. } => {
                    let b = self.value_buf(&acts, src)?;
                    acts.insert(out.clone(), b);
                }
                Step::Compute { node_idx } => {
                    if let Some(&chain_idx) = self.fusion.tail_of.get(node_idx) {
                        // Fused elementwise chain: dispatch once at the tail node.
                        self.dispatch_fused(chain_idx, &mut acts, &mut encoder, &mut keepalive)?;
                    } else if self.fusion.skip.contains(node_idx) {
                        // Subsumed by a fused chain (dispatched at its tail); skip.
                    } else {
                        self.dispatch_node(*node_idx, &mut acts, &mut encoder, &mut keepalive)?;
                    }
                    if opprof {
                        let enc =
                            std::mem::replace(&mut encoder, self.backend.create_command_encoder());
                        let t = web_time::Instant::now();
                        self.backend.submit(enc).await?;
                        let dt = t.elapsed().as_secs_f64() * 1e3;
                        let node = &self.graph.nodes[*node_idx];
                        let key = if opprof_shapes
                            && (node.op_type == "Conv" || node.op_type == "ConvTranspose")
                        {
                            self.conv_prof_key(node)
                        } else {
                            node.op_type.clone()
                        };
                        let e = prof.entry(key).or_default();
                        e.0 += dt;
                        e.1 += 1;
                    }
                }
            }
        }

        self.backend.submit(encoder).await?;
        if opprof {
            let mut rows: Vec<_> = prof.iter().collect();
            rows.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
            for (op, (ms, n)) in rows {
                eprintln!("[onnx-opprof] {op:<22} total={ms:7.1}ms  count={n}");
            }
        }

        let mut out = HashMap::new();
        for name in &self.output_names {
            let buf = self.value_buf(&acts, name)?;
            let shape = self.plan.shape_of(name)?.to_vec();
            let bytes = self
                .backend
                .read_buffer(buf.id, 0, numel(&shape) * 4)
                .await?;
            let data: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
            out.insert(name.clone(), (shape, data));
        }
        drop(keepalive);
        Ok(out)
    }

    /// Resolve a value's buffer: an activation (this run) or a resident const.
    fn value_buf(
        &self,
        acts: &HashMap<String, Rc<GpuBuf>>,
        name: &str,
    ) -> Result<Rc<GpuBuf>, OnnxError> {
        acts.get(name)
            .or_else(|| self.consts.get(name))
            .cloned()
            .ok_or_else(|| OnnxError::Unsupported(format!("no buffer for value {name}")))
    }

    /// Pick the conv tile regime for an output shape: small-N for tiny Cout
    /// (e.g. the 3-channel output conv), wide for large spatial, else default.
    fn pick_tiled_conv(&self, cout: u32, m_spatial: u32) -> &TiledConv {
        if cout <= 4 {
            &self.pipelines.conv2d_tiled_small_n
        } else if m_spatial >= 65536 && cout >= 64 {
            &self.pipelines.conv2d_tiled_wide
        } else {
            &self.pipelines.conv2d_tiled
        }
    }

    /// Diagnostic profiling key for a Conv/ConvTranspose: shape signature so
    /// per-shape cost is visible under THINFER_ONNX_OPPROF_SHAPES. Falls back to
    /// the bare op type if any shape lookup fails.
    fn conv_prof_key(&self, node: &crate::onnx::proto::Node) -> String {
        let (Ok(xs), Ok(ws), Ok(os)) = (
            self.plan.shape_of(&node.inputs[0]),
            self.plan.shape_of(&node.inputs[1]),
            self.plan.shape_of(&node.outputs[0]),
        ) else {
            return node.op_type.clone();
        };
        let strides = node.attr_ints("strides").unwrap_or(&[1, 1]);
        format!(
            "{} k{}x{} s{}x{} cin{} cout{} out{}x{}",
            node.op_type, ws[2], ws[3], strides[0], strides[1], xs[1], os[1], os[2], os[3],
        )
    }

    fn alloc_out(
        &self,
        shape: &[i64],
        keepalive: &mut Vec<Rc<GpuBuf>>,
    ) -> Result<Rc<GpuBuf>, WgpuError> {
        let buf = Rc::new(GpuBuf::new(&self.backend, numel(shape) * 4)?);
        keepalive.push(Rc::clone(&buf));
        Ok(buf)
    }

    fn alloc_bytes(
        &self,
        bytes: u64,
        keepalive: &mut Vec<Rc<GpuBuf>>,
    ) -> Result<Rc<GpuBuf>, WgpuError> {
        let buf = Rc::new(GpuBuf::new(&self.backend, bytes)?);
        keepalive.push(Rc::clone(&buf));
        Ok(buf)
    }

    /// Opt-in i8 DP4A conv: quantize the activation (clear -> max-abs -> pack),
    /// then run the tiled `dot4I8Packed` conv against the load-time i8 weights.
    /// Precondition: `i8_weights[idx]` and `pipelines.conv_i8` both present.
    fn dispatch_conv_i8(
        &self,
        idx: usize,
        acts: &HashMap<String, Rc<GpuBuf>>,
        encoder: &mut crate::backend::CommandEncoderState,
        keepalive: &mut Vec<Rc<GpuBuf>>,
    ) -> Result<Rc<GpuBuf>, OnnxError> {
        let node = &self.graph.nodes[idx];
        let pipes = self.pipelines.conv_i8.as_ref().unwrap();
        let w8 = &self.i8_weights[&idx];
        let x = self.value_buf(acts, &node.inputs[0])?;
        let xs = self.plan.shape_of(&node.inputs[0])?.to_vec();
        let out_shape = self.plan.shape_of(&node.outputs[0])?.to_vec();
        let ws = self.plan.shape_of(&node.inputs[1])?.to_vec();
        let bias = match node.inputs.get(2).filter(|s| !s.is_empty()) {
            Some(b) => self.value_buf(acts, b)?,
            None => self.zero_bias(out_shape[1] as usize, keepalive)?,
        };
        let strides = node.attr_ints("strides").unwrap_or(&[1, 1]);
        let pads = node.attr_ints("pads").unwrap_or(&[0, 0, 0, 0]);
        let (b, cin, h, w) = (xs[0] as u32, xs[1] as u32, xs[2] as u32, xs[3] as u32);
        let n = b * cin * h * w;
        let nwords = n.div_ceil(4);
        let kdim = cin * ws[2] as u32 * ws[3] as u32;

        let mm = self.alloc_bytes(8, keepalive)?; // [max, min] order-preserving u32
        let x_i8 = self.alloc_bytes((nwords * 4) as u64, keepalive)?;
        let a_params = self.alloc_bytes(8, keepalive)?; // [scale, zero]
        let out = self.alloc_out(&out_shape, keepalive)?;

        let u_max = self.write_uniform(
            Uni::default().u32(n).u32(0).u32(0).u32(0).finish(),
            keepalive,
        )?;
        let u_q = self.write_uniform(
            Uni::default().u32(n).u32(nwords).u32(0).u32(0).finish(),
            keepalive,
        )?;
        let u_c = self.write_uniform(
            Uni::default()
                .u32(b)
                .u32(cin)
                .u32(out_shape[1] as u32)
                .u32(h)
                .u32(w)
                .u32(out_shape[2] as u32)
                .u32(out_shape[3] as u32)
                .u32(ws[2] as u32)
                .u32(ws[3] as u32)
                .u32(pads[0] as u32)
                .u32(pads[1] as u32)
                .u32(strides[0] as u32)
                .u32(strides[1] as u32)
                .u32(kdim)
                .u32(w8.kw4)
                .u32(0)
                .finish(),
            keepalive,
        )?;

        let be = self.backend.as_ref();
        be.dispatch(encoder, &pipes.clear, &[mm.binding(0)], [1, 1, 1])?;
        be.dispatch(
            encoder,
            &pipes.maxabs,
            &[x.binding(0), mm.binding(1), u_max.binding(2)],
            crate::ops::linear_workgroups(n, 256),
        )?;
        be.dispatch(
            encoder,
            &pipes.quant,
            &[
                x.binding(0),
                mm.binding(1),
                x_i8.binding(2),
                a_params.binding(3),
                u_q.binding(4),
            ],
            crate::ops::linear_workgroups(nwords, 64),
        )?;
        let cout = out_shape[1] as u32;
        let m_spatial = (out_shape[2] * out_shape[3]) as u32;
        be.dispatch(
            encoder,
            &pipes.conv,
            &[
                x_i8.binding(0),
                a_params.binding(1),
                w8.packed.binding(2),
                w8.scale.binding(3),
                w8.qsum.binding(4),
                bias.binding(5),
                out.binding(6),
                u_c.binding(7),
            ],
            [m_spatial.div_ceil(64), cout.div_ceil(64), b],
        )?;
        Ok(out)
    }

    fn write_uniform(
        &self,
        bytes: Vec<u8>,
        keepalive: &mut Vec<Rc<GpuBuf>>,
    ) -> Result<Rc<GpuBuf>, WgpuError> {
        let u = Rc::new(GpuBuf::new(&self.backend, bytes.len() as u64)?);
        self.backend.write_buffer(u.id, 0, &bytes)?;
        keepalive.push(Rc::clone(&u));
        Ok(u)
    }

    #[allow(clippy::too_many_lines)]
    fn dispatch_node(
        &self,
        idx: usize,
        acts: &mut HashMap<String, Rc<GpuBuf>>,
        encoder: &mut crate::backend::CommandEncoderState,
        keepalive: &mut Vec<Rc<GpuBuf>>,
    ) -> Result<(), OnnxError> {
        let node = &self.graph.nodes[idx];
        let op = node.op_type.as_str();
        let out_name = &node.outputs[0];
        let out_shape = self.plan.shape_of(out_name)?.to_vec();
        let in_shape = |i: usize| self.plan.shape_of(&node.inputs[i]);
        let total = numel(&out_shape) as u32;
        let wg = crate::ops::linear_workgroups(total, 64);

        match op {
            // Tuned implicit-GEMM path for plain (group=1, dilation=1) convs:
            // the HyperSwap/ArcFace bulk. Shared-memory weight + im2col reuse.
            "Conv"
                if node.attr_i("group", 1) == 1
                    && node
                        .attr_ints("dilations")
                        .map(|d| d == [1, 1])
                        .unwrap_or(true) =>
            {
                // Opt-in i8 DP4A path (gate on; weight quantized at load).
                if self.pipelines.conv_i8.is_some() && self.i8_weights.contains_key(&idx) {
                    let out = self.dispatch_conv_i8(idx, acts, encoder, keepalive)?;
                    acts.insert(out_name.clone(), out);
                    return Ok(());
                }
                let x = self.value_buf(acts, &node.inputs[0])?;
                let w = self.value_buf(acts, &node.inputs[1])?;
                let xs = in_shape(0)?.to_vec();
                let ws = in_shape(1)?.to_vec();
                let bias = match node.inputs.get(2).filter(|s| !s.is_empty()) {
                    Some(b) => self.value_buf(acts, b)?,
                    None => self.zero_bias(out_shape[1] as usize, keepalive)?,
                };
                let strides = node.attr_ints("strides").unwrap_or(&[1, 1]);
                let pads = node.attr_ints("pads").unwrap_or(&[0, 0, 0, 0]);
                // conv2d.rs uniform U: b,cin,cout,h_in,w_in,h_out,w_out,kh,kw,
                // pad_h,pad_w,stride_h,stride_w,_,_,_.
                let u = Uni::default()
                    .u32(xs[0] as u32)
                    .u32(xs[1] as u32)
                    .u32(out_shape[1] as u32)
                    .u32(xs[2] as u32)
                    .u32(xs[3] as u32)
                    .u32(out_shape[2] as u32)
                    .u32(out_shape[3] as u32)
                    .u32(ws[2] as u32)
                    .u32(ws[3] as u32)
                    .u32(pads[0] as u32)
                    .u32(pads[1] as u32)
                    .u32(strides[0] as u32)
                    .u32(strides[1] as u32)
                    .u32(0)
                    .u32(0)
                    .u32(0)
                    .finish();
                let uni = self.write_uniform(u, keepalive)?;
                let out = self.alloc_out(&out_shape, keepalive)?;
                let cout = out_shape[1] as u32;
                let m_spatial = (out_shape[2] * out_shape[3]) as u32;
                let conv = self.pick_tiled_conv(cout, m_spatial);
                let (xb, wb, bb, ub, ob) = (
                    x.bufref(),
                    w.bufref(),
                    bias.bufref(),
                    uni.bufref(),
                    out.bufref(),
                );
                let bufs = crate::ops::Conv2dBufs {
                    x: &xb,
                    w: &wb,
                    bias: &bb,
                    uniform: &ub,
                    out: &ob,
                };
                crate::ops::dispatch_conv2d::<crate::ops::Conv2dF32, _>(
                    self.backend.as_ref(),
                    encoder,
                    &conv.pipeline,
                    &conv.op,
                    &bufs,
                    cout,
                    m_spatial,
                    xs[0] as u32,
                )?;
                acts.insert(out_name.clone(), out);
            }
            // ConvTranspose via zero-upsample + tuned conv with the flipped/
            // transposed weight (built at load as `{out}__ct_w`).
            "ConvTranspose" if ct_tiled_eligible(node) => {
                let x = self.value_buf(acts, &node.inputs[0])?;
                let xs = in_shape(0)?.to_vec(); // [n, cin, h, w]
                let ws = in_shape(1)?.to_vec(); // [cin, cout, kh, kw]
                let (cin, kh, kw) = (xs[1] as u32, ws[2] as u32, ws[3] as u32);
                let strides = node.attr_ints("strides").unwrap_or(&[1, 1]);
                let pads = node.attr_ints("pads").unwrap_or(&[0, 0, 0, 0]);
                let (sh, sw) = (strides[0] as u32, strides[1] as u32);
                let up_h = (xs[2] as u32 - 1) * sh + 1;
                let up_w = (xs[3] as u32 - 1) * sw + 1;

                // 1) zero-upsample x -> x_up.
                let up_shape = vec![xs[0], cin as i64, up_h as i64, up_w as i64];
                let x_up = self.alloc_out(&up_shape, keepalive)?;
                let total_up = numel(&up_shape) as u32;
                let zu = Uni::default()
                    .u32(total_up)
                    .u32(xs[0] as u32)
                    .u32(cin)
                    .u32(xs[2] as u32)
                    .u32(xs[3] as u32)
                    .u32(up_h)
                    .u32(up_w)
                    .u32(sh)
                    .u32(sw)
                    .u32(0)
                    .u32(0)
                    .u32(0)
                    .finish();
                let zu_u = self.write_uniform(zu, keepalive)?;
                self.backend.dispatch(
                    encoder,
                    &self.pipelines.zero_upsample,
                    &[x.binding(0), x_up.binding(1), zu_u.binding(2)],
                    crate::ops::linear_workgroups(total_up, 64),
                )?;

                // 2) stride-1 conv with the flipped/transposed weight, pad' = k-1-p.
                let cout = out_shape[1] as u32;
                let ct_w = self
                    .consts
                    .get(&format!("{out_name}__ct_w"))
                    .cloned()
                    .ok_or_else(|| OnnxError::Unsupported("convtranspose weight missing".into()))?;
                let bias = match node.inputs.get(2).filter(|s| !s.is_empty()) {
                    Some(b) => self.value_buf(acts, b)?,
                    None => self.zero_bias(cout as usize, keepalive)?,
                };
                let pad_h = kh - 1 - pads[0] as u32;
                let pad_w = kw - 1 - pads[1] as u32;
                let u = Uni::default()
                    .u32(xs[0] as u32)
                    .u32(cin)
                    .u32(cout)
                    .u32(up_h)
                    .u32(up_w)
                    .u32(out_shape[2] as u32)
                    .u32(out_shape[3] as u32)
                    .u32(kh)
                    .u32(kw)
                    .u32(pad_h)
                    .u32(pad_w)
                    .u32(1)
                    .u32(1)
                    .u32(0)
                    .u32(0)
                    .u32(0)
                    .finish();
                let uni = self.write_uniform(u, keepalive)?;
                let out = self.alloc_out(&out_shape, keepalive)?;
                let m_spatial = (out_shape[2] * out_shape[3]) as u32;
                let conv = self.pick_tiled_conv(cout, m_spatial);
                let (xb, wb, bb, ub, ob) = (
                    x_up.bufref(),
                    ct_w.bufref(),
                    bias.bufref(),
                    uni.bufref(),
                    out.bufref(),
                );
                let bufs = crate::ops::Conv2dBufs {
                    x: &xb,
                    w: &wb,
                    bias: &bb,
                    uniform: &ub,
                    out: &ob,
                };
                crate::ops::dispatch_conv2d::<crate::ops::Conv2dF32, _>(
                    self.backend.as_ref(),
                    encoder,
                    &conv.pipeline,
                    &conv.op,
                    &bufs,
                    cout,
                    m_spatial,
                    xs[0] as u32,
                )?;
                acts.insert(out_name.clone(), out);
            }
            "Conv" | "ConvTranspose" => {
                let x = self.value_buf(acts, &node.inputs[0])?;
                let w = self.value_buf(acts, &node.inputs[1])?;
                let xs = in_shape(0)?.to_vec();
                let ws = in_shape(1)?.to_vec();
                let bias = match node.inputs.get(2).filter(|s| !s.is_empty()) {
                    Some(b) => self.value_buf(acts, b)?,
                    None => self.zero_bias(out_shape[1] as usize, keepalive)?,
                };
                let group = node.attr_i("group", 1) as u32; // direct-kernel path
                let ksh = node
                    .attr_ints("kernel_shape")
                    .map(|v| (v[0] as u32, v[1] as u32))
                    .unwrap_or((ws[2] as u32, ws[3] as u32));
                let strides = node.attr_ints("strides").unwrap_or(&[1, 1]);
                let pads = node.attr_ints("pads").unwrap_or(&[0, 0, 0, 0]);
                let dil = node.attr_ints("dilations").unwrap_or(&[1, 1]);
                let u = Uni::default()
                    .u32(xs[0] as u32)
                    .u32(xs[1] as u32)
                    .u32(out_shape[1] as u32)
                    .u32(xs[2] as u32)
                    .u32(xs[3] as u32)
                    .u32(out_shape[2] as u32)
                    .u32(out_shape[3] as u32)
                    .u32(ksh.0)
                    .u32(ksh.1)
                    .u32(strides[0] as u32)
                    .u32(strides[1] as u32)
                    .u32(pads[0] as u32)
                    .u32(pads[1] as u32)
                    .u32(dil[0] as u32)
                    .u32(dil[1] as u32)
                    .u32(group)
                    .finish();
                let uni = self.write_uniform(u, keepalive)?;
                let out = self.alloc_out(&out_shape, keepalive)?;
                let pipe = if op == "Conv" {
                    &self.pipelines.conv2d
                } else {
                    &self.pipelines.convt2d
                };
                self.backend.dispatch(
                    encoder,
                    pipe,
                    &[
                        x.binding(0),
                        w.binding(1),
                        bias.binding(2),
                        out.binding(3),
                        uni.binding(4),
                    ],
                    wg,
                )?;
                acts.insert(out_name.clone(), out);
            }
            "Gemm" => {
                let a = self.value_buf(acts, &node.inputs[0])?;
                let b = self.value_buf(acts, &node.inputs[1])?;
                let a_sh = in_shape(0)?.to_vec();
                let b_sh = in_shape(1)?.to_vec();
                let trans_b = node.attr_i("transB", 0) != 0;
                let (m, k) = (a_sh[0] as u32, a_sh[1] as u32);
                let n = if trans_b { b_sh[0] } else { b_sh[1] } as u32;
                let bias = node.inputs.get(2).filter(|s| !s.is_empty());
                let bias_buf = match bias {
                    Some(nm) => self.value_buf(acts, nm)?,
                    None => self.zero_bias(n as usize, keepalive)?,
                };
                let u = Uni::default()
                    .u32(m)
                    .u32(n)
                    .u32(k)
                    .u32(trans_b as u32)
                    .u32(bias.is_some() as u32)
                    .u32(0)
                    .u32(0)
                    .u32(0)
                    .finish();
                let uni = self.write_uniform(u, keepalive)?;
                let out = self.alloc_out(&out_shape, keepalive)?;
                self.backend.dispatch(
                    encoder,
                    &self.pipelines.gemm,
                    &[
                        a.binding(0),
                        b.binding(1),
                        bias_buf.binding(2),
                        out.binding(3),
                        uni.binding(4),
                    ],
                    crate::ops::linear_workgroups(m * n, 64),
                )?;
                acts.insert(out_name.clone(), out);
            }
            "InstanceNormalization" => {
                let x = self.value_buf(acts, &node.inputs[0])?;
                let scale = self.value_buf(acts, &node.inputs[1])?;
                let bias = self.value_buf(acts, &node.inputs[2])?;
                let xs = in_shape(0)?.to_vec();
                let (n, c) = (xs[0] as u32, xs[1] as u32);
                let hw = (xs[2] * xs[3]) as u32;
                let eps = node.attr_f("epsilon", 1e-5);
                let u = Uni::default().u32(n).u32(c).u32(hw).f32(eps).finish();
                let uni = self.write_uniform(u, keepalive)?;
                let out = self.alloc_out(&out_shape, keepalive)?;
                self.backend.dispatch(
                    encoder,
                    &self.pipelines.instance_norm,
                    &[
                        x.binding(0),
                        scale.binding(1),
                        bias.binding(2),
                        out.binding(3),
                        uni.binding(4),
                    ],
                    [n * c, 1, 1],
                )?;
                acts.insert(out_name.clone(), out);
            }
            "BatchNormalization" => {
                let x = self.value_buf(acts, &node.inputs[0])?;
                let a = self
                    .consts
                    .get(&format!("{out_name}__bn_a"))
                    .cloned()
                    .ok_or_else(|| OnnxError::Unsupported("BN affine missing".into()))?;
                let b = self
                    .consts
                    .get(&format!("{out_name}__bn_b"))
                    .cloned()
                    .unwrap();
                let xs = in_shape(0)?.to_vec();
                let (c, hw) = channel_hw(&xs);
                let u = Uni::default().u32(total).u32(c).u32(hw).u32(0).finish();
                let uni = self.write_uniform(u, keepalive)?;
                let out = self.alloc_out(&out_shape, keepalive)?;
                self.backend.dispatch(
                    encoder,
                    &self.pipelines.channel_affine,
                    &[
                        x.binding(0),
                        a.binding(1),
                        b.binding(2),
                        out.binding(3),
                        uni.binding(4),
                    ],
                    wg,
                )?;
                acts.insert(out_name.clone(), out);
            }
            "PRelu" => {
                let x = self.value_buf(acts, &node.inputs[0])?;
                let slope = self.value_buf(acts, &node.inputs[1])?;
                let xs = in_shape(0)?.to_vec();
                let (c, hw) = channel_hw(&xs);
                let slope_len = numel(in_shape(1)?) as u32;
                let u = Uni::default()
                    .u32(total)
                    .u32(c)
                    .u32(hw)
                    .u32(slope_len)
                    .finish();
                let uni = self.write_uniform(u, keepalive)?;
                let out = self.alloc_out(&out_shape, keepalive)?;
                self.backend.dispatch(
                    encoder,
                    &self.pipelines.prelu,
                    &[
                        x.binding(0),
                        slope.binding(1),
                        out.binding(2),
                        uni.binding(3),
                    ],
                    wg,
                )?;
                acts.insert(out_name.clone(), out);
            }
            "Relu" | "Sigmoid" | "Tanh" | "LeakyRelu" | "Sqrt" | "Reciprocal" | "HardSigmoid"
            | "Clip" => {
                let x = self.value_buf(acts, &node.inputs[0])?;
                // kind, alpha, beta per UNARY kernel's switch.
                let (kind, alpha, beta) = match op {
                    "Relu" => (0u32, 0.0, 0.0),
                    "Sigmoid" => (1, 0.0, 0.0),
                    "Tanh" => (2, 0.0, 0.0),
                    "LeakyRelu" => (3, node.attr_f("alpha", 0.01), 0.0),
                    "Sqrt" => (5, 0.0, 0.0),
                    "Reciprocal" => (8, 0.0, 0.0),
                    "HardSigmoid" => (6, node.attr_f("alpha", 0.2), node.attr_f("beta", 0.5)),
                    // Clip: min/max come from optional scalar const inputs.
                    _ => {
                        let clip_bound = |i: usize, default: f32| -> f32 {
                            node.inputs
                                .get(i)
                                .filter(|s| !s.is_empty())
                                .and_then(|n| self.plan.consts.get(n))
                                .map(|d| d.to_f32()[0])
                                .unwrap_or(default)
                        };
                        (7, clip_bound(1, -3.4e38), clip_bound(2, 3.4e38))
                    }
                };
                let u = Uni::default()
                    .u32(total)
                    .u32(kind)
                    .f32(alpha)
                    .f32(beta)
                    .finish();
                let uni = self.write_uniform(u, keepalive)?;
                let out = self.alloc_out(&out_shape, keepalive)?;
                self.backend.dispatch(
                    encoder,
                    &self.pipelines.unary,
                    &[x.binding(0), out.binding(1), uni.binding(2)],
                    wg,
                )?;
                acts.insert(out_name.clone(), out);
            }
            "Add" | "Sub" | "Mul" | "Div" | "Max" | "Min" | "Pow" => {
                let a = self.value_buf(acts, &node.inputs[0])?;
                let b = self.value_buf(acts, &node.inputs[1])?;
                let ad = pad4(in_shape(0)?);
                let bd = pad4(in_shape(1)?);
                let od = pad4(&out_shape);
                let opcode = match op {
                    "Add" => 0u32,
                    "Sub" => 1,
                    "Mul" => 2,
                    "Div" => 3,
                    "Max" => 4,
                    "Min" => 5,
                    _ => 6, // Pow
                };
                let mut u = Uni::default().u32(opcode).u32(total).u32(0).u32(0);
                for v in od.iter().chain(ad.iter()).chain(bd.iter()) {
                    u = u.u32(*v);
                }
                let uni = self.write_uniform(u.finish(), keepalive)?;
                let out = self.alloc_out(&out_shape, keepalive)?;
                self.backend.dispatch(
                    encoder,
                    &self.pipelines.binary,
                    &[a.binding(0), b.binding(1), out.binding(2), uni.binding(3)],
                    wg,
                )?;
                acts.insert(out_name.clone(), out);
            }
            "Expand" => {
                let x = self.value_buf(acts, &node.inputs[0])?;
                let xd = pad4(in_shape(0)?);
                let od = pad4(&out_shape);
                let mut u = Uni::default().u32(total).u32(0).u32(0).u32(0);
                for v in od.iter().chain(xd.iter()) {
                    u = u.u32(*v);
                }
                let uni = self.write_uniform(u.finish(), keepalive)?;
                let out = self.alloc_out(&out_shape, keepalive)?;
                self.backend.dispatch(
                    encoder,
                    &self.pipelines.expand,
                    &[x.binding(0), out.binding(1), uni.binding(2)],
                    wg,
                )?;
                acts.insert(out_name.clone(), out);
            }
            "Transpose" => {
                let x = self.value_buf(acts, &node.inputs[0])?;
                let xs = in_shape(0)?.to_vec();
                let r = xs.len();
                let perm: Vec<usize> = match node.attr_ints("perm") {
                    Some(p) => p.iter().map(|&v| v as usize).collect(),
                    None => (0..r).rev().collect(),
                };
                // Left-pad rank to 4 with identity dims.
                let off = 4 - r;
                let id = pad4(&xs);
                let mut perm4 = [0u32, 1, 2, 3];
                for (a, &p) in perm.iter().enumerate() {
                    perm4[off + a] = (off + p) as u32;
                }
                let od = [
                    id[perm4[0] as usize],
                    id[perm4[1] as usize],
                    id[perm4[2] as usize],
                    id[perm4[3] as usize],
                ];
                let mut u = Uni::default().u32(total).u32(0).u32(0).u32(0);
                for v in id.iter().chain(od.iter()).chain(perm4.iter()) {
                    u = u.u32(*v);
                }
                let uni = self.write_uniform(u.finish(), keepalive)?;
                let out = self.alloc_out(&out_shape, keepalive)?;
                self.backend.dispatch(
                    encoder,
                    &self.pipelines.transpose,
                    &[x.binding(0), out.binding(1), uni.binding(2)],
                    wg,
                )?;
                acts.insert(out_name.clone(), out);
            }
            "DepthToSpace" => {
                let x = self.value_buf(acts, &node.inputs[0])?;
                let xs = in_shape(0)?.to_vec();
                let bs = node.attr_i("blocksize", 1) as u32;
                let mode = if node.attr_s("mode") == Some("CRD") {
                    1u32
                } else {
                    0
                };
                let u = Uni::default()
                    .u32(total)
                    .u32(out_shape[0] as u32)
                    .u32(out_shape[1] as u32)
                    .u32(out_shape[2] as u32)
                    .u32(out_shape[3] as u32)
                    .u32(xs[1] as u32)
                    .u32(xs[2] as u32)
                    .u32(xs[3] as u32)
                    .u32(bs)
                    .u32(mode)
                    .u32(0)
                    .u32(0)
                    .finish();
                let uni = self.write_uniform(u, keepalive)?;
                let out = self.alloc_out(&out_shape, keepalive)?;
                self.backend.dispatch(
                    encoder,
                    &self.pipelines.depth_to_space,
                    &[x.binding(0), out.binding(1), uni.binding(2)],
                    wg,
                )?;
                acts.insert(out_name.clone(), out);
            }
            "Resize" => {
                let x = self.value_buf(acts, &node.inputs[0])?;
                let xs = in_shape(0)?.to_vec();
                let mode = if node.attr_s("mode") == Some("linear") {
                    1u32
                } else {
                    0
                };
                let coord = if node.attr_s("coordinate_transformation_mode") == Some("half_pixel") {
                    1u32
                } else {
                    0
                };
                let u = Uni::default()
                    .u32(total)
                    .u32(xs[0] as u32)
                    .u32(xs[1] as u32)
                    .u32(xs[2] as u32)
                    .u32(xs[3] as u32)
                    .u32(out_shape[2] as u32)
                    .u32(out_shape[3] as u32)
                    .u32(mode)
                    .u32(coord)
                    .u32(0)
                    .u32(0)
                    .u32(0)
                    .finish();
                let uni = self.write_uniform(u, keepalive)?;
                let out = self.alloc_out(&out_shape, keepalive)?;
                self.backend.dispatch(
                    encoder,
                    &self.pipelines.resize,
                    &[x.binding(0), out.binding(1), uni.binding(2)],
                    wg,
                )?;
                acts.insert(out_name.clone(), out);
            }
            "MaxPool" => {
                let x = self.value_buf(acts, &node.inputs[0])?;
                let xs = in_shape(0)?.to_vec();
                let ksh = node.attr_ints("kernel_shape").unwrap_or(&[1, 1]);
                let strides = node.attr_ints("strides").unwrap_or(&[1, 1]);
                let pads = node.attr_ints("pads").unwrap_or(&[0, 0, 0, 0]);
                let dil = node.attr_ints("dilations").unwrap_or(&[1, 1]);
                let u = Uni::default()
                    .u32(total)
                    .u32(xs[0] as u32)
                    .u32(xs[1] as u32)
                    .u32(xs[2] as u32)
                    .u32(xs[3] as u32)
                    .u32(out_shape[2] as u32)
                    .u32(out_shape[3] as u32)
                    .u32(ksh[0] as u32)
                    .u32(ksh[1] as u32)
                    .u32(strides[0] as u32)
                    .u32(strides[1] as u32)
                    .u32(pads[0] as u32)
                    .u32(pads[1] as u32)
                    .u32(dil[0] as u32)
                    .u32(dil[1] as u32)
                    .finish();
                let uni = self.write_uniform(u, keepalive)?;
                let out = self.alloc_out(&out_shape, keepalive)?;
                self.backend.dispatch(
                    encoder,
                    &self.pipelines.maxpool,
                    &[x.binding(0), out.binding(1), uni.binding(2)],
                    wg,
                )?;
                acts.insert(out_name.clone(), out);
            }
            "Concat" => {
                let axis = {
                    let a = node.attr_i("axis", 0);
                    (if a < 0 { a + out_shape.len() as i64 } else { a }) as usize
                };
                let outer: i64 = out_shape[..axis].iter().product::<i64>().max(1);
                if node.inputs.len() > 2 && outer == 1 {
                    // Contiguous fast path (N inputs, unit outer dims): the concat
                    // axis is the outermost non-unit one, so each input is a
                    // contiguous run - a plain buffer copy.
                    let out = self.alloc_out(&out_shape, keepalive)?;
                    let mut off_elems: u64 = 0;
                    for inp in &node.inputs {
                        let src = self.value_buf(acts, inp)?;
                        let n = numel(self.plan.shape_of(inp)?);
                        self.backend.copy_buffer_to_buffer(
                            encoder,
                            src.id,
                            0,
                            out.id,
                            off_elems * 4,
                            n * 4,
                        )?;
                        off_elems += n;
                    }
                    acts.insert(out_name.clone(), out);
                } else {
                    // General case: fold the inputs left with the batch-safe
                    // 2-input concat kernel (handles any axis / non-unit outer).
                    let mut acc = self.value_buf(acts, &node.inputs[0])?;
                    let mut acc_shape = in_shape(0)?.to_vec();
                    for (i, inp) in node.inputs.iter().enumerate().skip(1) {
                        let b = self.value_buf(acts, inp)?;
                        let b_shape = in_shape(i)?.to_vec();
                        let mut step_shape = acc_shape.clone();
                        step_shape[axis] += b_shape[axis];
                        let out = if i == node.inputs.len() - 1 {
                            self.alloc_out(&out_shape, keepalive)?
                        } else {
                            self.alloc_out(&step_shape, keepalive)?
                        };
                        self.dispatch_concat2(
                            &acc,
                            &acc_shape,
                            &b,
                            &b_shape,
                            axis,
                            &step_shape,
                            &out,
                            encoder,
                            keepalive,
                        )?;
                        acc = out;
                        acc_shape = step_shape;
                    }
                    acts.insert(out_name.clone(), acc);
                }
            }
            // Batched MatMul with a batch dim of 1 (RTMPose SimCC/GAU head):
            // collapse A's leading dims into M and reuse the GEMM kernel, which
            // reads B as a contiguous `[K, N]` block (valid since B's leading
            // dims are all 1 here). No transpose, no bias.
            "MatMul" => {
                let a = self.value_buf(acts, &node.inputs[0])?;
                let b = self.value_buf(acts, &node.inputs[1])?;
                let a_sh = in_shape(0)?.to_vec();
                let b_sh = in_shape(1)?.to_vec();
                let k = *a_sh.last().unwrap() as u32;
                let n = *b_sh.last().unwrap() as u32;
                let m = (numel(&a_sh) / k.max(1) as u64) as u32;
                let bias = self.zero_bias(n as usize, keepalive)?;
                let u = Uni::default()
                    .u32(m)
                    .u32(n)
                    .u32(k)
                    .u32(0)
                    .u32(0)
                    .u32(0)
                    .u32(0)
                    .u32(0)
                    .finish();
                let uni = self.write_uniform(u, keepalive)?;
                let out = self.alloc_out(&out_shape, keepalive)?;
                self.backend.dispatch(
                    encoder,
                    &self.pipelines.gemm,
                    &[
                        a.binding(0),
                        b.binding(1),
                        bias.binding(2),
                        out.binding(3),
                        uni.binding(4),
                    ],
                    crate::ops::linear_workgroups(m * n, 64),
                )?;
                acts.insert(out_name.clone(), out);
            }
            "Slice" => {
                let x = self.value_buf(acts, &node.inputs[0])?;
                let xs = in_shape(0)?.to_vec();
                let (start, step) = slice_starts_steps(node, &self.plan, &xs)?;
                let out = self.alloc_out(&out_shape, keepalive)?;
                self.dispatch_slice(&x, &out, &xs, &out_shape, start, step, encoder, keepalive)?;
                acts.insert(out_name.clone(), out);
            }
            // Constant-value Pad (up-to-4D). `pads` = [begin.., end..] from an
            // attribute or the 2nd (const) input; optional 3rd input is the fill.
            "Pad" => {
                let x = self.value_buf(acts, &node.inputs[0])?;
                let xs = in_shape(0)?.to_vec();
                let rank = xs.len();
                let pads: Vec<i64> = node
                    .attr_ints("pads")
                    .map(|v| v.to_vec())
                    .or_else(|| {
                        node.inputs
                            .get(1)
                            .filter(|s| !s.is_empty())
                            .and_then(|n| self.plan.consts.get(n))
                            .map(|d| d.to_f32().iter().map(|&f| f.round() as i64).collect())
                    })
                    .ok_or_else(|| OnnxError::Unsupported("Pad without static pads".into()))?;
                let val = node
                    .inputs
                    .get(2)
                    .filter(|s| !s.is_empty())
                    .and_then(|n| self.plan.consts.get(n))
                    .map(|d| d.to_f32()[0])
                    .unwrap_or(0.0);
                let off = 4 - rank;
                let mut begin = [0u32; 4];
                for (i, &p) in pads.iter().take(rank).enumerate() {
                    begin[off + i] = p.max(0) as u32;
                }
                let id = pad4(&xs);
                let od = pad4(&out_shape);
                let mut u = Uni::default().u32(total).f32(val).u32(0).u32(0);
                for v in id.iter().chain(od.iter()).chain(begin.iter()) {
                    u = u.u32(*v);
                }
                let uni = self.write_uniform(u.finish(), keepalive)?;
                let out = self.alloc_out(&out_shape, keepalive)?;
                self.backend.dispatch(
                    encoder,
                    &self.pipelines.pad,
                    &[x.binding(0), out.binding(1), uni.binding(2)],
                    wg,
                )?;
                acts.insert(out_name.clone(), out);
            }
            // ConstantOfShape: fill the output with the scalar `value` attr
            // (default 0). Shape came from the const 1-D input at plan time.
            "ConstantOfShape" => {
                let val = match node.attr("value") {
                    Some(crate::onnx::proto::AttrValue::T(t)) => t.data.to_f32()[0],
                    _ => 0.0,
                };
                let u = Uni::default().u32(total).f32(val).u32(0).u32(0).finish();
                let uni = self.write_uniform(u, keepalive)?;
                let out = self.alloc_out(&out_shape, keepalive)?;
                self.backend.dispatch(
                    encoder,
                    &self.pipelines.const_fill,
                    &[out.binding(0), uni.binding(1)],
                    wg,
                )?;
                acts.insert(out_name.clone(), out);
            }
            // Gather along `axis` with const integer indices (uploaded as u32).
            "Gather" => {
                let data = self.value_buf(acts, &node.inputs[0])?;
                let ds = in_shape(0)?.to_vec();
                let rank = ds.len() as i64;
                let axis = {
                    let a = node.attr_i("axis", 0);
                    (if a < 0 { a + rank } else { a }) as usize
                };
                let axis_len = ds[axis];
                let idx_i64: Vec<i64> = self
                    .plan
                    .consts
                    .get(&node.inputs[1])
                    .ok_or_else(|| OnnxError::Unsupported("Gather with non-const indices".into()))?
                    .as_i64()
                    .iter()
                    .map(|&i| if i < 0 { i + axis_len } else { i })
                    .collect();
                let idx_u32: Vec<u32> = idx_i64.iter().map(|&i| i as u32).collect();
                let outer: u32 = ds[..axis].iter().product::<i64>().max(1) as u32;
                let inner: u32 = ds[axis + 1..].iter().product::<i64>().max(1) as u32;
                let n_idx = idx_u32.len().max(1) as u32;
                let idx_buf = self.alloc_bytes((idx_u32.len().max(1) * 4) as u64, keepalive)?;
                self.backend
                    .write_buffer(idx_buf.id, 0, bytemuck::cast_slice(&idx_u32))?;
                let u = Uni::default()
                    .u32(total)
                    .u32(outer)
                    .u32(n_idx)
                    .u32(inner)
                    .u32(axis_len as u32)
                    .u32(0)
                    .u32(0)
                    .u32(0)
                    .finish();
                let uni = self.write_uniform(u, keepalive)?;
                let out = self.alloc_out(&out_shape, keepalive)?;
                self.backend.dispatch(
                    encoder,
                    &self.pipelines.gather,
                    &[
                        data.binding(0),
                        idx_buf.binding(1),
                        out.binding(2),
                        uni.binding(3),
                    ],
                    wg,
                )?;
                acts.insert(out_name.clone(), out);
            }
            // Tile: repeat the input `repeats` (const) times per axis.
            "Tile" => {
                let x = self.value_buf(acts, &node.inputs[0])?;
                let id = pad4(in_shape(0)?);
                let od = pad4(&out_shape);
                let mut u = Uni::default().u32(total).u32(0).u32(0).u32(0);
                for v in id.iter().chain(od.iter()) {
                    u = u.u32(*v);
                }
                let uni = self.write_uniform(u.finish(), keepalive)?;
                let out = self.alloc_out(&out_shape, keepalive)?;
                self.backend.dispatch(
                    encoder,
                    &self.pipelines.tile,
                    &[x.binding(0), out.binding(1), uni.binding(2)],
                    wg,
                )?;
                acts.insert(out_name.clone(), out);
            }
            // Split along `axis`: each output is a contiguous sub-block, emitted
            // via the slice kernel (start = running offset on that axis).
            "Split" => {
                let x = self.value_buf(acts, &node.inputs[0])?;
                let xs = in_shape(0)?.to_vec();
                let rank = xs.len() as i64;
                let axis = {
                    let a = node.attr_i("axis", 0);
                    (if a < 0 { a + rank } else { a }) as usize
                };
                let off = 4 - xs.len();
                let mut cursor = 0i64;
                for out_val in &node.outputs {
                    let osh = self.plan.shape_of(out_val)?.to_vec();
                    let mut start = [0u32; 4];
                    start[off + axis] = cursor as u32;
                    let out = self.alloc_out(&osh, keepalive)?;
                    self.dispatch_slice(&x, &out, &xs, &osh, start, [1u32; 4], encoder, keepalive)?;
                    acts.insert(out_val.clone(), out);
                    cursor += osh[axis];
                }
            }
            "GlobalAveragePool" => {
                let x = self.value_buf(acts, &node.inputs[0])?;
                let xs = in_shape(0)?.to_vec();
                let nc = (xs[0] * xs[1]) as u32;
                let hw = (xs[2] * xs[3]) as u32;
                let u = Uni::default().u32(nc).u32(hw).u32(0).u32(0).finish();
                let uni = self.write_uniform(u, keepalive)?;
                let out = self.alloc_out(&out_shape, keepalive)?;
                self.backend.dispatch(
                    encoder,
                    &self.pipelines.global_avg_pool,
                    &[x.binding(0), out.binding(1), uni.binding(2)],
                    crate::ops::linear_workgroups(nc, 64),
                )?;
                acts.insert(out_name.clone(), out);
            }
            "ReduceSum" => {
                let x = self.value_buf(acts, &node.inputs[0])?;
                let xs = in_shape(0)?.to_vec();
                let rank = xs.len() as i64;
                let axes: Vec<i64> = node
                    .attr_ints("axes")
                    .map(|a| a.to_vec())
                    .or_else(|| {
                        node.inputs
                            .get(1)
                            .filter(|s| !s.is_empty())
                            .and_then(|n| self.plan.consts.get(n))
                            .map(|d| d.as_i64().to_vec())
                    })
                    .ok_or_else(|| OnnxError::Unsupported("ReduceSum without axes".into()))?;
                // Normalize + sort the axes. Multiple axes are supported only when
                // contiguous (the common trailing-block reduce, e.g. StyleGAN
                // weight demod over [in,kh,kw]): collapse them into one logical
                // axis of length = product, stride = product of the axes after it.
                let mut axset: Vec<usize> = axes
                    .iter()
                    .map(|&a| (if a < 0 { a + rank } else { a }) as usize)
                    .collect();
                axset.sort_unstable();
                axset.dedup();
                let (first, last) = (axset[0], *axset.last().unwrap());
                if axset != (first..=last).collect::<Vec<_>>() {
                    return Err(OnnxError::Unsupported(
                        "ReduceSum over non-contiguous axes".into(),
                    ));
                }
                let alen = xs[first..=last].iter().product::<i64>() as u32;
                let astride: u32 = xs[last + 1..].iter().product::<i64>() as u32;
                let u = Uni::default()
                    .u32(total)
                    .u32(alen)
                    .u32(astride)
                    .u32(0)
                    .finish();
                let uni = self.write_uniform(u, keepalive)?;
                let out = self.alloc_out(&out_shape, keepalive)?;
                self.backend.dispatch(
                    encoder,
                    &self.pipelines.reduce_sum,
                    &[x.binding(0), out.binding(1), uni.binding(2)],
                    wg,
                )?;
                acts.insert(out_name.clone(), out);
            }
            other => return Err(OnnxError::Unsupported(format!("compute op '{other}'"))),
        }
        Ok(())
    }

    /// Dispatch a fused elementwise chain: read `primary` once, apply the whole
    /// op sequence in registers (side operands read per element with broadcast),
    /// write `output` once. Bit-exact vs the per-op path.
    fn dispatch_fused(
        &self,
        chain_idx: usize,
        acts: &mut HashMap<String, Rc<GpuBuf>>,
        encoder: &mut crate::backend::CommandEncoderState,
        keepalive: &mut Vec<Rc<GpuBuf>>,
    ) -> Result<(), OnnxError> {
        let chain = &self.fusion.chains[chain_idx];
        let pipe = &self.fused_pipelines[chain_idx];
        let primary = self.value_buf(acts, &chain.primary)?;
        let sides: Vec<Rc<GpuBuf>> = chain
            .sides
            .iter()
            .map(|s| self.value_buf(acts, s))
            .collect::<Result<_, _>>()?;
        let out = self.alloc_out(&chain.out_shape, keepalive)?;
        let total = numel(&chain.out_shape) as u32;

        // Uniform: total (+pad), out dims, then each side's padded-4D dims.
        let mut u = Uni::default().u32(total).u32(0).u32(0).u32(0);
        for v in pad4(&chain.out_shape) {
            u = u.u32(v);
        }
        for s in &chain.sides {
            for v in pad4(self.plan.shape_of(s)?) {
                u = u.u32(v);
            }
        }
        let uni = self.write_uniform(u.finish(), keepalive)?;

        // Bindings: primary@0, side_k@k+1, out@nsides+1, uniform@nsides+2.
        let nsides = sides.len() as u32;
        let mut bindings = vec![primary.binding(0)];
        for (k, s) in sides.iter().enumerate() {
            bindings.push(s.binding(k as u32 + 1));
        }
        bindings.push(out.binding(nsides + 1));
        bindings.push(uni.binding(nsides + 2));
        self.backend.dispatch(
            encoder,
            pipe,
            &bindings,
            crate::ops::linear_workgroups(total, 64),
        )?;
        acts.insert(chain.output.clone(), out);
        Ok(())
    }

    /// Dispatch the strided-slice kernel: gather `out[oc] = x[start + oc*step]`
    /// per padded-4D axis. Shared by ONNX Slice and Split.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_slice(
        &self,
        x: &Rc<GpuBuf>,
        out: &Rc<GpuBuf>,
        xs: &[i64],
        os: &[i64],
        start: [u32; 4],
        step: [u32; 4],
        encoder: &mut crate::backend::CommandEncoderState,
        keepalive: &mut Vec<Rc<GpuBuf>>,
    ) -> Result<(), OnnxError> {
        let id = pad4(xs);
        let od = pad4(os);
        let total = numel(os) as u32;
        let mut u = Uni::default().u32(total).u32(0).u32(0).u32(0);
        for v in id
            .iter()
            .chain(od.iter())
            .chain(start.iter())
            .chain(step.iter())
        {
            u = u.u32(*v);
        }
        let uni = self.write_uniform(u.finish(), keepalive)?;
        self.backend.dispatch(
            encoder,
            &self.pipelines.slice,
            &[x.binding(0), out.binding(1), uni.binding(2)],
            crate::ops::linear_workgroups(total, 64),
        )?;
        Ok(())
    }

    /// Dispatch the batch-safe 2-input concat kernel joining `a`/`b` along
    /// `axis` into `out` (shape `os`). Shared by the 2-input and N-way paths.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_concat2(
        &self,
        a: &Rc<GpuBuf>,
        a_shape: &[i64],
        b: &Rc<GpuBuf>,
        b_shape: &[i64],
        axis: usize,
        os: &[i64],
        out: &Rc<GpuBuf>,
        encoder: &mut crate::backend::CommandEncoderState,
        keepalive: &mut Vec<Rc<GpuBuf>>,
    ) -> Result<(), OnnxError> {
        let ad = pad4(a_shape);
        let bd = pad4(b_shape);
        let od = pad4(os);
        // pad4 right-aligns to 4D; shift `axis` (over the logical shape) to match.
        let off = 4 - os.len();
        let a_axis = ad[axis + off];
        let total = numel(os) as u32;
        let mut u = Uni::default()
            .u32(total)
            .u32((axis + off) as u32)
            .u32(a_axis)
            .u32(0);
        for v in od.iter().chain(ad.iter()).chain(bd.iter()) {
            u = u.u32(*v);
        }
        let uni = self.write_uniform(u.finish(), keepalive)?;
        self.backend.dispatch(
            encoder,
            &self.pipelines.concat2,
            &[a.binding(0), b.binding(1), out.binding(2), uni.binding(3)],
            crate::ops::linear_workgroups(total, 64),
        )?;
        Ok(())
    }

    fn zero_bias(
        &self,
        n: usize,
        keepalive: &mut Vec<Rc<GpuBuf>>,
    ) -> Result<Rc<GpuBuf>, WgpuError> {
        let buf = Rc::new(GpuBuf::new(&self.backend, (n.max(1) * 4) as u64)?);
        self.backend
            .write_buffer(buf.id, 0, &vec![0u8; n.max(1) * 4])?;
        keepalive.push(Rc::clone(&buf));
        Ok(buf)
    }
}

/// (channels, spatial) for a per-channel op, treating dim 1 as channels.
fn channel_hw(shape: &[i64]) -> (u32, u32) {
    let c = shape.get(1).copied().unwrap_or(1) as u32;
    let hw = shape.iter().skip(2).product::<i64>().max(1) as u32;
    (c, hw)
}

/// Whether a ConvTranspose can use the zero-upsample + tuned-conv path: plain
/// group=1, dilation=1, no output_padding (the HyperSwap convtransposes).
fn ct_tiled_eligible(node: &crate::onnx::proto::Node) -> bool {
    node.attr_i("group", 1) == 1
        && node
            .attr_ints("dilations")
            .map(|d| d == [1, 1])
            .unwrap_or(true)
        && node
            .attr_ints("output_padding")
            .map(|o| o.iter().all(|&v| v == 0))
            .unwrap_or(true)
}

/// Per-axis (start, step) padded to 4D for a Slice node, read from its const
/// starts/ends/axes/steps inputs. Only positive steps are supported (all the
/// slices in these models are forward strides).
fn slice_starts_steps(
    node: &crate::onnx::proto::Node,
    plan: &Plan,
    xs: &[i64],
) -> Result<([u32; 4], [u32; 4]), OnnxError> {
    let cint = |i: usize| -> Vec<i64> {
        plan.consts
            .get(&node.inputs[i])
            .map(|d| d.as_i64().to_vec())
            .unwrap_or_default()
    };
    let starts = cint(1);
    let rank = xs.len() as i64;
    let axes: Vec<i64> = match node.inputs.get(3).filter(|s| !s.is_empty()) {
        Some(_) => cint(3)
            .iter()
            .map(|&a| if a < 0 { a + rank } else { a })
            .collect(),
        None => (0..starts.len() as i64).collect(),
    };
    let steps: Vec<i64> = match node.inputs.get(4).filter(|s| !s.is_empty()) {
        Some(_) => cint(4),
        None => vec![1; starts.len()],
    };
    let off = 4 - xs.len();
    let mut start = [0u32; 4];
    let mut step = [1u32; 4];
    for (k, &ax) in axes.iter().enumerate() {
        if steps[k] <= 0 {
            return Err(OnnxError::Unsupported(
                "Slice with non-positive step".into(),
            ));
        }
        let dim = xs[ax as usize];
        let norm = |v: i64| if v < 0 { v + dim } else { v };
        start[off + ax as usize] = norm(starts[k]).clamp(0, dim) as u32;
        step[off + ax as usize] = steps[k] as u32;
    }
    Ok((start, step))
}

/// Indices of a node's inputs that are real data operands (not shape/param
/// inputs consumed at plan time). Used to decide which constants to upload.
fn data_operands<'a>(op: &str, inputs: &'a [String]) -> Vec<&'a str> {
    let take = |n: usize| -> Vec<&str> {
        inputs
            .iter()
            .take(n)
            .map(|s| s.as_str())
            .filter(|s| !s.is_empty())
            .collect()
    };
    match op {
        // x + shape/param input(s) -> only the data tensor. (Slice starts/ends/
        // axes/steps, Split sizes, ReduceSum axes, Clip min/max are const params
        // read at plan/dispatch time, never uploaded as buffers.)
        "Resize" | "Expand" | "Slice" | "Split" | "ReduceSum" | "Clip" | "Pad" | "Tile"
        | "Gather" => take(1),
        // Shape-only input (consumed at plan time); no data operand to upload.
        "ConstantOfShape" => take(0),
        // All inputs are data operands.
        _ => inputs
            .iter()
            .map(|s| s.as_str())
            .filter(|s| !s.is_empty())
            .collect(),
    }
}
