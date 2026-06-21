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
}

impl Pipelines {
    async fn compile(backend: &WgpuBackend) -> Result<Self, WgpuError> {
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
        };
        const SMALL_N: Conv2dConfig = Conv2dConfig {
            bm: 4,
            bn: 128,
            bk: 32,
            tm: 1,
            tn: 2,
        };
        Ok(Self {
            conv2d: p("onnx_conv2d", kernels::CONV2D, 3).await?,
            conv2d_tiled: TiledConv::compile(backend, "onnx_conv2d_tiled", Conv2dConfig::DEFAULT)
                .await?,
            conv2d_tiled_wide: TiledConv::compile(backend, "onnx_conv2d_tiled_wide", WIDE).await?,
            conv2d_tiled_small_n: TiledConv::compile(backend, "onnx_conv2d_tiled_small_n", SMALL_N)
                .await?,
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
    /// Resident constant buffers (weights, biases, folded BN affine, etc.),
    /// keyed by value name (BN affine uses synthetic `name__bn_a/b`).
    consts: HashMap<String, Rc<GpuBuf>>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
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
        let graph =
            super::proto::parse_model(onnx_bytes).map_err(|e| OnnxError::Io(e.to_string()))?;
        let plan = super::shape::plan(&graph, input_shapes)?;
        let pipelines = Pipelines::compile(&backend).await?;

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
            consts: HashMap::new(),
            input_names,
            output_names,
        };
        model.upload_constants()?;
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
        let mut prof: std::collections::BTreeMap<&str, (f64, u32)> =
            std::collections::BTreeMap::new();

        for step in &self.plan.steps {
            match step {
                Step::View { out, src, .. } => {
                    let b = self.value_buf(&acts, src)?;
                    acts.insert(out.clone(), b);
                }
                Step::Compute { node_idx } => {
                    self.dispatch_node(*node_idx, &mut acts, &mut encoder, &mut keepalive)?;
                    if opprof {
                        let enc =
                            std::mem::replace(&mut encoder, self.backend.create_command_encoder());
                        let t = web_time::Instant::now();
                        self.backend.submit(enc).await?;
                        let dt = t.elapsed().as_secs_f64() * 1e3;
                        let e = prof
                            .entry(self.graph.nodes[*node_idx].op_type.as_str())
                            .or_default();
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

    fn alloc_out(
        &self,
        shape: &[i64],
        keepalive: &mut Vec<Rc<GpuBuf>>,
    ) -> Result<Rc<GpuBuf>, WgpuError> {
        let buf = Rc::new(GpuBuf::new(&self.backend, numel(shape) * 4)?);
        keepalive.push(Rc::clone(&buf));
        Ok(buf)
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
            "Relu" | "Sigmoid" | "Tanh" | "LeakyRelu" => {
                let x = self.value_buf(acts, &node.inputs[0])?;
                let kind = match op {
                    "Relu" => 0u32,
                    "Sigmoid" => 1,
                    "Tanh" => 2,
                    _ => 3,
                };
                let alpha = node.attr_f("alpha", 0.01);
                let u = Uni::default()
                    .u32(total)
                    .u32(kind)
                    .f32(alpha)
                    .u32(0)
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
            "Add" | "Sub" | "Mul" | "Div" => {
                let a = self.value_buf(acts, &node.inputs[0])?;
                let b = self.value_buf(acts, &node.inputs[1])?;
                let ad = pad4(in_shape(0)?);
                let bd = pad4(in_shape(1)?);
                let od = pad4(&out_shape);
                let opcode = match op {
                    "Add" => 0u32,
                    "Sub" => 1,
                    "Mul" => 2,
                    _ => 3,
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
                if node.inputs.len() == 2 {
                    // Batch-safe 2-input concat kernel (the U-Net skip joins).
                    let a = self.value_buf(acts, &node.inputs[0])?;
                    let b = self.value_buf(acts, &node.inputs[1])?;
                    let ad = pad4(in_shape(0)?);
                    let bd = pad4(in_shape(1)?);
                    let od = pad4(&out_shape);
                    // `axis` is over the (possibly <4D) logical shape; pad4 right-
                    // aligns to 4D, so shift the axis index to match.
                    let off = 4 - out_shape.len();
                    let a_axis = ad[axis + off];
                    let mut u = Uni::default()
                        .u32(total)
                        .u32((axis + off) as u32)
                        .u32(a_axis)
                        .u32(0);
                    for v in od.iter().chain(ad.iter()).chain(bd.iter()) {
                        u = u.u32(*v);
                    }
                    let uni = self.write_uniform(u.finish(), keepalive)?;
                    let out = self.alloc_out(&out_shape, keepalive)?;
                    self.backend.dispatch(
                        encoder,
                        &self.pipelines.concat2,
                        &[a.binding(0), b.binding(1), out.binding(2), uni.binding(3)],
                        wg,
                    )?;
                    acts.insert(out_name.clone(), out);
                } else {
                    // Contiguous fallback (N inputs): valid only when the axes
                    // before `axis` are unit (e.g. batch 1, channel concat).
                    let outer: i64 = out_shape[..axis].iter().product::<i64>().max(1);
                    if outer != 1 {
                        return Err(OnnxError::Unsupported(format!(
                            "Concat of {} inputs at axis={axis} with non-unit outer dims {out_shape:?}",
                            node.inputs.len()
                        )));
                    }
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
                }
            }
            other => return Err(OnnxError::Unsupported(format!("compute op '{other}'"))),
        }
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
        // x + shape/param input(s) -> only the data tensor.
        "Resize" | "Expand" => take(1),
        // All inputs are data operands.
        _ => inputs
            .iter()
            .map(|s| s.as_str())
            .filter(|s| !s.is_empty())
            .collect(),
    }
}
