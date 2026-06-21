//! Load-time graph analysis: shape inference + constant folding.
//!
//! Inputs are fixed-size, so every tensor shape in these models is static. A
//! single topological pass (ONNX graphs come pre-sorted) computes, for each
//! value, its concrete shape and - for anything reachable from constants - its
//! folded value. The integer/shape subgraph (Shape -> Gather -> Unsqueeze ->
//! Concat -> Reshape target, etc.) collapses to constants here, so the executor
//! only ever sees genuine compute nodes plus cheap metadata "views".
//!
//! Output: a [`Plan`] = per-value shapes, the constant table (initializers +
//! folded results), and an ordered list of [`Step`]s for the executor.

use std::collections::HashMap;

use super::proto::{AttrValue, Graph, Node, TensorData};

#[derive(Debug)]
pub enum PlanError {
    UnknownValue(String),
    Unsupported(String),
    Shape(String),
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::UnknownValue(s) => write!(f, "onnx: unknown value '{s}'"),
            PlanError::Unsupported(s) => write!(f, "onnx: unsupported: {s}"),
            PlanError::Shape(s) => write!(f, "onnx: shape error: {s}"),
        }
    }
}
impl std::error::Error for PlanError {}

/// One executor action. Folded / constant-producing nodes emit nothing (their
/// outputs live in [`Plan::consts`]); metadata reshapes emit a `View`; real
/// kernels emit a `Compute` carrying the source node index.
#[derive(Clone, Debug)]
pub enum Step {
    /// Output aliases an input buffer under a new (same-numel) shape. No GPU work.
    View {
        out: String,
        src: String,
        shape: Vec<i64>,
    },
    /// Dispatch `graph.nodes[node_idx]`. Resolved I/O shapes are in `Plan::shapes`.
    Compute { node_idx: usize },
}

pub struct Plan {
    /// Concrete shape of every value (graph inputs, initializers, node outputs).
    pub shapes: HashMap<String, Vec<i64>>,
    /// Folded constants: initializers plus everything the shape subgraph
    /// produced. Keyed by value name.
    pub consts: HashMap<String, TensorData>,
    pub steps: Vec<Step>,
}

impl Plan {
    pub fn shape_of(&self, name: &str) -> Result<&[i64], PlanError> {
        self.shapes
            .get(name)
            .map(|v| v.as_slice())
            .ok_or_else(|| PlanError::UnknownValue(name.to_string()))
    }
}

fn numel(shape: &[i64]) -> i64 {
    shape
        .iter()
        .product::<i64>()
        .max(if shape.is_empty() { 1 } else { 0 })
}

/// numpy-style broadcast of two shapes (right-aligned).
fn broadcast(a: &[i64], b: &[i64]) -> Result<Vec<i64>, PlanError> {
    let n = a.len().max(b.len());
    let mut out = vec![0i64; n];
    for i in 0..n {
        let av = if i < n - a.len() {
            1
        } else {
            a[i - (n - a.len())]
        };
        let bv = if i < n - b.len() {
            1
        } else {
            b[i - (n - b.len())]
        };
        out[i] = if av == bv || bv == 1 {
            av
        } else if av == 1 {
            bv
        } else {
            return Err(PlanError::Shape(format!(
                "cannot broadcast {a:?} with {b:?}"
            )));
        };
    }
    Ok(out)
}

/// Resolve a Reshape/Expand-style target shape against the input numel,
/// handling -1 (infer) and 0 (copy input dim, `allowzero=0` semantics).
fn resolve_target(target: &[i64], input: &[i64], allow_zero: bool) -> Result<Vec<i64>, PlanError> {
    let in_numel = numel(input);
    let mut out: Vec<i64> = target.to_vec();
    let mut neg1: Option<usize> = None;
    let mut known: i64 = 1;
    for (i, d) in out.iter_mut().enumerate() {
        if *d == -1 {
            if neg1.is_some() {
                return Err(PlanError::Shape("reshape with two -1".into()));
            }
            neg1 = Some(i);
        } else if *d == 0 && !allow_zero {
            *d = *input
                .get(i)
                .ok_or_else(|| PlanError::Shape("reshape 0-dim out of range".into()))?;
            known *= *d;
        } else {
            known *= *d;
        }
    }
    if let Some(i) = neg1 {
        if known == 0 {
            return Err(PlanError::Shape("reshape -1 with zero divisor".into()));
        }
        out[i] = in_numel / known;
    }
    Ok(out)
}

/// The per-value working state during the analysis pass.
struct Val {
    shape: Vec<i64>,
    data: Option<TensorData>,
}

/// Run the analysis. `input_shapes` binds each graph input name to its concrete
/// runtime shape (replacing any dynamic axes).
pub fn plan(graph: &Graph, input_shapes: &HashMap<String, Vec<i64>>) -> Result<Plan, PlanError> {
    let mut vals: HashMap<String, Val> = HashMap::new();

    // Initializers are constants.
    for t in &graph.initializers {
        vals.insert(
            t.name.clone(),
            Val {
                shape: t.dims.clone(),
                data: Some(t.data.clone()),
            },
        );
    }
    // Graph inputs (skip ones that are also initializers - some exports list both).
    for vi in &graph.inputs {
        if vals.contains_key(&vi.name) {
            continue;
        }
        let shape = match input_shapes.get(&vi.name) {
            Some(s) => s.clone(),
            None => vi.dims.iter().map(|d| d.unwrap_or(1)).collect::<Vec<_>>(),
        };
        vals.insert(vi.name.clone(), Val { shape, data: None });
    }

    let mut steps: Vec<Step> = Vec::new();

    for (idx, node) in graph.nodes.iter().enumerate() {
        process_node(node, idx, &mut vals, &mut steps)?;
    }

    let mut shapes = HashMap::new();
    let mut consts = HashMap::new();
    for (name, v) in vals {
        shapes.insert(name.clone(), v.shape);
        if let Some(d) = v.data {
            consts.insert(name, d);
        }
    }
    Ok(Plan {
        shapes,
        consts,
        steps,
    })
}

fn get<'a>(vals: &'a HashMap<String, Val>, name: &str) -> Result<&'a Val, PlanError> {
    vals.get(name)
        .ok_or_else(|| PlanError::UnknownValue(name.to_string()))
}

/// Is every named input a known constant?
fn all_const(vals: &HashMap<String, Val>, inputs: &[String]) -> bool {
    inputs
        .iter()
        .all(|n| n.is_empty() || vals.get(n).map(|v| v.data.is_some()).unwrap_or(false))
}

fn process_node(
    node: &Node,
    idx: usize,
    vals: &mut HashMap<String, Val>,
    steps: &mut Vec<Step>,
) -> Result<(), PlanError> {
    let op = node.op_type.as_str();

    // --- always-foldable: produce a constant, no Step ---
    match op {
        "Constant" => {
            let t = node
                .attr_t("value")
                .ok_or_else(|| PlanError::Unsupported("Constant without tensor value".into()))?;
            set_const(vals, &node.outputs[0], t.dims.clone(), t.data.clone());
            return Ok(());
        }
        "Shape" => {
            let v = get(vals, &node.inputs[0])?;
            let mut dims: Vec<i64> = v.shape.clone();
            // Optional start/end (opset 15+); slice the shape vector.
            let rank = dims.len() as i64;
            let norm = |x: i64| {
                if x < 0 {
                    (x + rank).clamp(0, rank)
                } else {
                    x.min(rank)
                }
            };
            let start = norm(node.attr_i("start", 0)) as usize;
            let end = node
                .attr("end")
                .and_then(|a| {
                    if let AttrValue::I(i) = a {
                        Some(norm(*i) as usize)
                    } else {
                        None
                    }
                })
                .unwrap_or(dims.len());
            dims = dims[start.min(dims.len())..end.max(start).min(dims.len())].to_vec();
            let len = dims.len() as i64;
            set_const(vals, &node.outputs[0], vec![len], TensorData::I64(dims));
            return Ok(());
        }
        _ => {}
    }

    // --- conditionally-foldable shape-subgraph ops (fold iff inputs const) ---
    if all_const(vals, &node.inputs) && try_fold(node, vals)?.is_some() {
        return Ok(());
    }

    // --- metadata views (no GPU work) ---
    match op {
        "Reshape" => {
            let in_v = get(vals, &node.inputs[0])?;
            let in_shape = in_v.shape.clone();
            let target = get(vals, &node.inputs[1])?
                .data
                .as_ref()
                .ok_or_else(|| PlanError::Unsupported("Reshape with non-const shape".into()))?
                .as_i64()
                .to_vec();
            let allow_zero = node.attr_i("allowzero", 0) != 0;
            let shape = resolve_target(&target, &in_shape, allow_zero)?;
            set_shape(vals, &node.outputs[0], shape.clone());
            steps.push(Step::View {
                out: node.outputs[0].clone(),
                src: node.inputs[0].clone(),
                shape,
            });
            return Ok(());
        }
        "Flatten" => {
            let in_shape = get(vals, &node.inputs[0])?.shape.clone();
            let axis = {
                let a = node.attr_i("axis", 1);
                (if a < 0 { a + in_shape.len() as i64 } else { a }) as usize
            };
            let outer: i64 = in_shape[..axis].iter().product::<i64>().max(1);
            let inner: i64 = in_shape[axis..].iter().product::<i64>().max(1);
            let shape = vec![outer, inner];
            set_shape(vals, &node.outputs[0], shape.clone());
            steps.push(Step::View {
                out: node.outputs[0].clone(),
                src: node.inputs[0].clone(),
                shape,
            });
            return Ok(());
        }
        "Squeeze" | "Unsqueeze" => {
            let in_shape = get(vals, &node.inputs[0])?.shape.clone();
            let axes = squeeze_axes(node, vals, &in_shape, op == "Unsqueeze")?;
            let shape = apply_squeeze(&in_shape, &axes, op == "Unsqueeze");
            set_shape(vals, &node.outputs[0], shape.clone());
            steps.push(Step::View {
                out: node.outputs[0].clone(),
                src: node.inputs[0].clone(),
                shape,
            });
            return Ok(());
        }
        // All-f32 executor: an activation Cast is a no-op relabel.
        "Cast" | "Identity" | "Dropout" => {
            let in_shape = get(vals, &node.inputs[0])?.shape.clone();
            set_shape(vals, &node.outputs[0], in_shape.clone());
            steps.push(Step::View {
                out: node.outputs[0].clone(),
                src: node.inputs[0].clone(),
                shape: in_shape,
            });
            return Ok(());
        }
        _ => {}
    }

    // --- compute nodes: infer output shape(s), emit a Compute step ---
    infer_compute(node, vals)?;
    steps.push(Step::Compute { node_idx: idx });
    Ok(())
}

fn set_const(vals: &mut HashMap<String, Val>, name: &str, shape: Vec<i64>, data: TensorData) {
    vals.insert(
        name.to_string(),
        Val {
            shape,
            data: Some(data),
        },
    );
}
fn set_shape(vals: &mut HashMap<String, Val>, name: &str, shape: Vec<i64>) {
    vals.insert(name.to_string(), Val { shape, data: None });
}

/// Resolve Squeeze/Unsqueeze axes from either the `axes` attribute (opset<13)
/// or the second input (opset>=13).
fn squeeze_axes(
    node: &Node,
    vals: &HashMap<String, Val>,
    in_shape: &[i64],
    unsqueeze: bool,
) -> Result<Vec<i64>, PlanError> {
    let raw: Vec<i64> = if let Some(a) = node.attr_ints("axes") {
        a.to_vec()
    } else if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
        get(vals, &node.inputs[1])?
            .data
            .as_ref()
            .ok_or_else(|| PlanError::Unsupported("Squeeze/Unsqueeze non-const axes".into()))?
            .as_i64()
            .to_vec()
    } else if unsqueeze {
        return Err(PlanError::Unsupported("Unsqueeze without axes".into()));
    } else {
        // Squeeze without axes: drop all size-1 dims.
        in_shape
            .iter()
            .enumerate()
            .filter(|&(_, &d)| d == 1)
            .map(|(i, _)| i as i64)
            .collect()
    };
    let out_rank = if unsqueeze {
        (in_shape.len() + raw.len()) as i64
    } else {
        in_shape.len() as i64
    };
    Ok(raw
        .iter()
        .map(|&a| if a < 0 { a + out_rank } else { a })
        .collect())
}

fn apply_squeeze(in_shape: &[i64], axes: &[i64], unsqueeze: bool) -> Vec<i64> {
    if unsqueeze {
        let out_rank = in_shape.len() + axes.len();
        let mut out = vec![1i64; out_rank];
        let axset: std::collections::HashSet<i64> = axes.iter().copied().collect();
        let mut src = 0;
        for (i, slot) in out.iter_mut().enumerate() {
            if !axset.contains(&(i as i64)) {
                *slot = in_shape[src];
                src += 1;
            }
        }
        out
    } else {
        let axset: std::collections::HashSet<i64> = axes.iter().copied().collect();
        in_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !axset.contains(&(*i as i64)))
            .map(|(_, &d)| d)
            .collect()
    }
}

/// Fold a shape-subgraph node whose inputs are all constant. Returns `Some(())`
/// if it handled the op (and stored the output constant), `None` if the op is
/// not a foldable kind (the caller then treats it as compute / view).
fn try_fold(node: &Node, vals: &mut HashMap<String, Val>) -> Result<Option<()>, PlanError> {
    let op = node.op_type.as_str();
    macro_rules! ints {
        ($name:expr) => {
            get(vals, $name)?.data.as_ref().unwrap().as_i64().to_vec()
        };
    }
    match op {
        "Gather" => {
            let data = ints!(&node.inputs[0]);
            let idx_v = get(vals, &node.inputs[1])?;
            let idx = idx_v.data.as_ref().unwrap().as_i64().to_vec();
            let n = data.len() as i64;
            let picked: Vec<i64> = idx
                .iter()
                .map(|&i| data[(if i < 0 { i + n } else { i }) as usize])
                .collect();
            // Output rank = indices rank (axis 0 gather of a 1-D vector).
            let out_shape = if idx_v.shape.is_empty() {
                vec![]
            } else {
                vec![picked.len() as i64]
            };
            set_const(vals, &node.outputs[0], out_shape, TensorData::I64(picked));
            Ok(Some(()))
        }
        "Concat" => {
            // Only int (shape) concat folds here; activation concat is compute.
            let mut out = Vec::new();
            for inp in &node.inputs {
                out.extend(ints!(inp));
            }
            let len = out.len() as i64;
            set_const(vals, &node.outputs[0], vec![len], TensorData::I64(out));
            Ok(Some(()))
        }
        "Unsqueeze" => {
            let data = ints!(&node.inputs[0]);
            let len = data.len() as i64;
            // Folding a 1-D/scalar shape value: keep the data, bump shape rank.
            let in_shape = get(vals, &node.inputs[0])?.shape.clone();
            let axes = squeeze_axes(node, vals, &in_shape, true)?;
            let shape = apply_squeeze(&in_shape, &axes, true);
            let _ = len;
            set_const(vals, &node.outputs[0], shape, TensorData::I64(data));
            Ok(Some(()))
        }
        "Squeeze" => {
            let data = ints!(&node.inputs[0]);
            let in_shape = get(vals, &node.inputs[0])?.shape.clone();
            let axes = squeeze_axes(node, vals, &in_shape, false)?;
            let shape = apply_squeeze(&in_shape, &axes, false);
            set_const(vals, &node.outputs[0], shape, TensorData::I64(data));
            Ok(Some(()))
        }
        "Slice" => {
            let data = ints!(&node.inputs[0]);
            let starts = ints!(&node.inputs[1]);
            let ends = ints!(&node.inputs[2]);
            // axes/steps optional; for a 1-D shape vector assume axis 0, step 1.
            let n = data.len() as i64;
            let clamp = |x: i64| if x < 0 { (x + n).max(0) } else { x.min(n) };
            let s = clamp(starts[0]) as usize;
            let e = clamp(ends[0]) as usize;
            let out: Vec<i64> = data[s.min(data.len())..e.max(s).min(data.len())].to_vec();
            let len = out.len() as i64;
            set_const(vals, &node.outputs[0], vec![len], TensorData::I64(out));
            Ok(Some(()))
        }
        "Cast" => {
            // Constant cast: keep i64 as i64 (shape math). The only casts in
            // these graphs that fold are int<->int on shape values.
            let v = get(vals, &node.inputs[0])?;
            let (shape, data) = (v.shape.clone(), v.data.clone().unwrap());
            set_const(vals, &node.outputs[0], shape, data);
            Ok(Some(()))
        }
        "Add" | "Sub" | "Mul" | "Div" => {
            let a = ints!(&node.inputs[0]);
            let b = ints!(&node.inputs[1]);
            let n = a.len().max(b.len());
            let out: Vec<i64> = (0..n)
                .map(|i| {
                    let av = a[i % a.len().max(1)];
                    let bv = b[i % b.len().max(1)];
                    match op {
                        "Add" => av + bv,
                        "Sub" => av - bv,
                        "Mul" => av * bv,
                        _ => av / bv,
                    }
                })
                .collect();
            // Result shape = broadcast of operand shapes.
            let sa = get(vals, &node.inputs[0])?.shape.clone();
            let sb = get(vals, &node.inputs[1])?.shape.clone();
            let shape = broadcast(&sa, &sb)?;
            set_const(vals, &node.outputs[0], shape, TensorData::I64(out));
            Ok(Some(()))
        }
        // Not a foldable kind (e.g. Reshape: handled as a view by caller).
        _ => Ok(None),
    }
}

/// Infer the output shape(s) of a compute node and record them.
fn infer_compute(node: &Node, vals: &mut HashMap<String, Val>) -> Result<(), PlanError> {
    let op = node.op_type.as_str();
    let in_shape =
        |i: usize| -> Result<Vec<i64>, PlanError> { Ok(get(vals, &node.inputs[i])?.shape.clone()) };
    match op {
        "Conv" | "MaxPool" => {
            let x = in_shape(0)?;
            let (n, cin, h, w) = (x[0], x[1], x[2], x[3]);
            let cout = if op == "Conv" { in_shape(1)?[0] } else { cin };
            let ksh = node
                .attr_ints("kernel_shape")
                .map(|v| v.to_vec())
                .unwrap_or_else(|| {
                    let w = vals
                        .get(&node.inputs[1])
                        .map(|v| v.shape.clone())
                        .unwrap_or_default();
                    vec![
                        w.get(2).copied().unwrap_or(1),
                        w.get(3).copied().unwrap_or(1),
                    ]
                });
            let strides = node
                .attr_ints("strides")
                .map(|v| v.to_vec())
                .unwrap_or(vec![1, 1]);
            let dil = node
                .attr_ints("dilations")
                .map(|v| v.to_vec())
                .unwrap_or(vec![1, 1]);
            let pads = node
                .attr_ints("pads")
                .map(|v| v.to_vec())
                .unwrap_or(vec![0, 0, 0, 0]);
            let ho = (h + pads[0] + pads[2] - dil[0] * (ksh[0] - 1) - 1) / strides[0] + 1;
            let wo = (w + pads[1] + pads[3] - dil[1] * (ksh[1] - 1) - 1) / strides[1] + 1;
            set_shape(vals, &node.outputs[0], vec![n, cout, ho, wo]);
        }
        "ConvTranspose" => {
            let x = in_shape(0)?;
            let wsh = in_shape(1)?; // [Cin, Cout/group, kH, kW]
            let group = node.attr_i("group", 1);
            let cout = wsh[1] * group;
            let ksh = node
                .attr_ints("kernel_shape")
                .map(|v| v.to_vec())
                .unwrap_or(vec![wsh[2], wsh[3]]);
            let strides = node
                .attr_ints("strides")
                .map(|v| v.to_vec())
                .unwrap_or(vec![1, 1]);
            let dil = node
                .attr_ints("dilations")
                .map(|v| v.to_vec())
                .unwrap_or(vec![1, 1]);
            let pads = node
                .attr_ints("pads")
                .map(|v| v.to_vec())
                .unwrap_or(vec![0, 0, 0, 0]);
            let outpad = node
                .attr_ints("output_padding")
                .map(|v| v.to_vec())
                .unwrap_or(vec![0, 0]);
            let ho =
                strides[0] * (x[2] - 1) + outpad[0] + dil[0] * (ksh[0] - 1) + 1 - pads[0] - pads[2];
            let wo =
                strides[1] * (x[3] - 1) + outpad[1] + dil[1] * (ksh[1] - 1) + 1 - pads[1] - pads[3];
            set_shape(vals, &node.outputs[0], vec![x[0], cout, ho, wo]);
        }
        "Gemm" => {
            let a = in_shape(0)?;
            let b = in_shape(1)?;
            let trans_b = node.attr_i("transB", 0) != 0;
            let m = a[0];
            let n = if trans_b { b[0] } else { b[1] };
            set_shape(vals, &node.outputs[0], vec![m, n]);
        }
        "MatMul" => {
            let a = in_shape(0)?;
            let b = in_shape(1)?;
            let mut shape = a.clone();
            *shape.last_mut().unwrap() = *b.last().unwrap();
            set_shape(vals, &node.outputs[0], shape);
        }
        "InstanceNormalization"
        | "BatchNormalization"
        | "Relu"
        | "LeakyRelu"
        | "PRelu"
        | "Sigmoid"
        | "Tanh"
        | "Softplus"
        | "Elu"
        | "Clip"
        | "Abs"
        | "Sqrt"
        | "Exp"
        | "Neg" => {
            let x = in_shape(0)?;
            set_shape(vals, &node.outputs[0], x);
        }
        "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Max" | "Min" => {
            let a = in_shape(0)?;
            let b = in_shape(1)?;
            set_shape(vals, &node.outputs[0], broadcast(&a, &b)?);
        }
        "Concat" => {
            let axis = node.attr_i("axis", 0);
            let mut out = in_shape(0)?;
            let ax = if axis < 0 {
                axis + out.len() as i64
            } else {
                axis
            } as usize;
            let mut total = 0;
            for inp in &node.inputs {
                total += get(vals, inp)?.shape[ax];
            }
            out[ax] = total;
            set_shape(vals, &node.outputs[0], out);
        }
        "Resize" => {
            let x = in_shape(0)?;
            // inputs: X, roi, scales, sizes (later inputs may be empty "").
            let sizes_in = node.inputs.get(3).filter(|s| !s.is_empty());
            let scales_in = node.inputs.get(2).filter(|s| !s.is_empty());
            let out = if let Some(s) = sizes_in
                .and_then(|n| vals.get(n))
                .and_then(|v| v.data.as_ref())
            {
                s.as_i64().to_vec()
            } else if let Some(sc) = scales_in
                .and_then(|n| vals.get(n))
                .and_then(|v| v.data.as_ref())
            {
                let sc = sc.to_f32();
                x.iter()
                    .zip(sc.iter())
                    .map(|(&d, &s)| (d as f32 * s).floor() as i64)
                    .collect()
            } else {
                return Err(PlanError::Unsupported(
                    "Resize without sizes or scales".into(),
                ));
            };
            set_shape(vals, &node.outputs[0], out);
        }
        "DepthToSpace" => {
            let x = in_shape(0)?;
            let bs = node.attr_i("blocksize", 1);
            set_shape(
                vals,
                &node.outputs[0],
                vec![x[0], x[1] / (bs * bs), x[2] * bs, x[3] * bs],
            );
        }
        "Transpose" => {
            let x = in_shape(0)?;
            let perm: Vec<usize> = match node.attr_ints("perm") {
                Some(p) => p.iter().map(|&v| v as usize).collect(),
                None => (0..x.len()).rev().collect(),
            };
            set_shape(vals, &node.outputs[0], perm.iter().map(|&p| x[p]).collect());
        }
        "Expand" => {
            let x = in_shape(0)?;
            let target = get(vals, &node.inputs[1])?
                .data
                .as_ref()
                .ok_or_else(|| PlanError::Unsupported("Expand non-const shape".into()))?
                .as_i64()
                .to_vec();
            set_shape(vals, &node.outputs[0], broadcast(&x, &target)?);
        }
        other => return Err(PlanError::Unsupported(format!("op '{other}'"))),
    }
    Ok(())
}
