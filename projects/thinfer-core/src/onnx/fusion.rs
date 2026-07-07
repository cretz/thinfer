//! Elementwise-chain fusion for the ONNX executor.
//!
//! HyperSwap (and GAN swappers generally) run a long tail of tiny elementwise /
//! activation ops - AdaIN scale/shift `Mul`+`Add`, gates `Mul(sigmoid(.))`,
//! residual `Add`, `Relu`/`LeakyRelu`/`Sigmoid` - each a separate dispatch that
//! reads and writes the whole activation tensor. On the flat 256^2 profile these
//! ~260 ops are ~half the time, dominated by global round-trips, not compute.
//!
//! This pass finds maximal chains of single-use elementwise ops and collapses
//! each into ONE codegen'd kernel that reads the primary input once, applies the
//! whole op sequence in registers (broadcast side operands read per element),
//! and writes once - eliminating the intermediate round-trips and dispatches.
//! Numerically equivalent to the per-op path: the arithmetic is identical (f32
//! buffer round-trips are lossless) and the activation formulas match
//! `kernels::{UNARY, BINARY}`; a residual ~1e-5 comes only from transcendentals
//! (exp/tanh) compiling slightly differently in the fused vs standalone shader.

use std::collections::{HashMap, HashSet};

use super::proto::{Graph, Node};
use super::shape::Plan;

/// Max side operands (extra broadcast inputs) a fused chain may bind. Bounds the
/// binding count / uniform size; chains that would need more are split.
const MAX_SIDES: usize = 6;

/// One elementwise op in a fused chain. `Binary` combines the running value with
/// a side operand (`side` indexes `FusedChain::sides`); `swap` marks the
/// non-commutative case where the running value is the RHS (`side - v`,
/// `side / v`). `Unary` applies an activation with baked `alpha`/`beta`.
#[derive(Clone, Debug)]
pub enum FusedOp {
    Binary {
        kind: BinKind,
        side: usize,
        swap: bool,
    },
    Unary {
        act: Act,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinKind {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Copy, Debug)]
pub enum Act {
    Relu,
    Sigmoid,
    Tanh,
    Sqrt,
    LeakyRelu { alpha: f32 },
    Clip { min: f32, max: f32 },
    HardSigmoid { alpha: f32, beta: f32 },
}

/// A fused elementwise chain: `primary` (same shape as `output`) run through
/// `ops`, reading `sides` per element with broadcast. Dispatched once at the
/// tail node (all inputs are then guaranteed produced); the other member nodes
/// are skipped by the executor.
#[derive(Clone, Debug)]
pub struct FusedChain {
    pub primary: String,
    pub sides: Vec<String>,
    pub ops: Vec<FusedOp>,
    pub output: String,
    pub out_shape: Vec<i64>,
    /// Node the executor dispatches the chain at (the last member, topologically).
    pub tail_node: usize,
}

/// Result of the fusion analysis: the chains plus the node bookkeeping the
/// executor needs (dispatch at each tail, skip every other member).
pub struct Fusion {
    pub chains: Vec<FusedChain>,
    /// tail node index -> chain index (dispatch the fused kernel here).
    pub tail_of: HashMap<usize, usize>,
    /// Member nodes that are subsumed (skip their normal dispatch); excludes tails.
    pub skip: HashSet<usize>,
}

impl Fusion {
    /// No fusion (every op dispatched individually).
    pub fn empty() -> Self {
        Fusion {
            chains: Vec::new(),
            tail_of: HashMap::new(),
            skip: HashSet::new(),
        }
    }
}

impl Act {
    fn from_node(node: &Node, plan: &Plan) -> Option<Act> {
        let f = |name: &str, d: f32| node.attr_f(name, d);
        Some(match node.op_type.as_str() {
            "Relu" => Act::Relu,
            "Sigmoid" => Act::Sigmoid,
            "Tanh" => Act::Tanh,
            "Sqrt" => Act::Sqrt,
            "LeakyRelu" => Act::LeakyRelu {
                alpha: f("alpha", 0.01),
            },
            "HardSigmoid" => Act::HardSigmoid {
                alpha: f("alpha", 0.2),
                beta: f("beta", 0.5),
            },
            "Clip" => {
                // min/max come from optional scalar const inputs (opset>=11).
                let bound = |i: usize, d: f32| -> f32 {
                    node.inputs
                        .get(i)
                        .filter(|s| !s.is_empty())
                        .and_then(|n| plan.consts.get(n))
                        .map(|c| c.to_f32()[0])
                        .unwrap_or(d)
                };
                // Same open bounds as the unfused Clip (kernels::UNARY).
                Act::Clip {
                    min: bound(1, -3.4e38),
                    max: bound(2, 3.4e38),
                }
            }
            _ => return None,
        })
    }
}

fn bin_kind(op: &str) -> Option<BinKind> {
    Some(match op {
        "Add" => BinKind::Add,
        "Sub" => BinKind::Sub,
        "Mul" => BinKind::Mul,
        "Div" => BinKind::Div,
        _ => return None,
    })
}

/// Whether an op type can participate in a fused elementwise chain.
fn is_fusable(op: &str) -> bool {
    matches!(
        op,
        "Add"
            | "Sub"
            | "Mul"
            | "Div"
            | "Relu"
            | "LeakyRelu"
            | "Sigmoid"
            | "Tanh"
            | "Sqrt"
            | "Clip"
            | "HardSigmoid"
    )
}

/// Find fusable chains. Greedy: each unclaimed fusable node seeds a chain that
/// extends through its single-use elementwise consumers while the running value
/// stays the full-shape ("primary") operand and side operands fit `MAX_SIDES`.
pub fn analyze(graph: &Graph, plan: &Plan) -> Fusion {
    // Occurrence count per value across all node inputs, plus graph outputs (so
    // an output value is never fused away as an interior link), and the single
    // consuming node when the count is exactly 1.
    let mut use_count: HashMap<&str, u32> = HashMap::new();
    let mut consumer: HashMap<&str, usize> = HashMap::new();
    for (idx, node) in graph.nodes.iter().enumerate() {
        for inp in &node.inputs {
            if inp.is_empty() {
                continue;
            }
            *use_count.entry(inp).or_insert(0) += 1;
            consumer.insert(inp, idx);
        }
    }
    for o in &graph.outputs {
        *use_count.entry(o.name.as_str()).or_insert(0) += 1;
    }
    let shape = |name: &str| plan.shapes.get(name).map(|v| v.as_slice()).unwrap_or(&[]);

    let mut chains: Vec<FusedChain> = Vec::new();
    let mut tail_of: HashMap<usize, usize> = HashMap::new();
    let mut skip: HashSet<usize> = HashSet::new();
    let mut claimed: HashSet<usize> = HashSet::new();

    for (seed_idx, seed) in graph.nodes.iter().enumerate() {
        if claimed.contains(&seed_idx) || !is_fusable(&seed.op_type) {
            continue;
        }
        let out_shape = shape(&seed.outputs[0]).to_vec();
        if out_shape.is_empty() {
            continue;
        }

        // Seed: pick the primary (full-shape operand) and the first op.
        let mut sides: Vec<String> = Vec::new();
        let mut ops: Vec<FusedOp> = Vec::new();
        let primary: String;
        if let Some(kind) = bin_kind(&seed.op_type) {
            let (a, b) = (&seed.inputs[0], &seed.inputs[1]);
            let (prim, side, swap) = if shape(a) == out_shape.as_slice() {
                (a.clone(), b.clone(), false)
            } else if shape(b) == out_shape.as_slice() {
                (b.clone(), a.clone(), true)
            } else {
                continue; // neither operand is full-shape; don't seed a chain
            };
            primary = prim;
            sides.push(side);
            ops.push(FusedOp::Binary {
                kind,
                side: 0,
                swap,
            });
        } else {
            // Unary seed.
            let Some(act) = Act::from_node(seed, plan) else {
                continue;
            };
            primary = seed.inputs[0].clone();
            ops.push(FusedOp::Unary { act });
        }

        let mut members = vec![seed_idx];
        let mut running = seed.outputs[0].clone();
        let mut running_shape = out_shape.clone();

        // Extend while the running value has exactly one consumer that is a
        // fusable op preserving the running shape (so `running` stays primary).
        loop {
            if use_count.get(running.as_str()).copied().unwrap_or(0) != 1 {
                break;
            }
            let Some(&c_idx) = consumer.get(running.as_str()) else {
                break;
            };
            if claimed.contains(&c_idx) {
                break;
            }
            let c = &graph.nodes[c_idx];
            if !is_fusable(&c.op_type) {
                break;
            }
            let c_out = shape(&c.outputs[0]);
            if c_out != running_shape.as_slice() {
                break; // output grows: running is not the full-shape operand
            }
            let next_op = if let Some(kind) = bin_kind(&c.op_type) {
                if sides.len() >= MAX_SIDES {
                    break;
                }
                // Which operand is the running value; the other is the side.
                let swap = if c.inputs[0] == running {
                    false
                } else if c.inputs[1] == running {
                    true
                } else {
                    break; // running not a direct operand (shouldn't happen)
                };
                let side = if swap { &c.inputs[0] } else { &c.inputs[1] };
                if side == &running {
                    break; // self-op (x op x); keep it unfused
                }
                sides.push(side.clone());
                FusedOp::Binary {
                    kind,
                    side: sides.len() - 1,
                    swap,
                }
            } else {
                let Some(act) = Act::from_node(c, plan) else {
                    break;
                };
                FusedOp::Unary { act }
            };
            ops.push(next_op);
            members.push(c_idx);
            running = c.outputs[0].clone();
            running_shape = c_out.to_vec();
        }

        // A single op is no win (still one dispatch, one round-trip).
        if ops.len() < 2 {
            continue;
        }
        let tail = *members.last().unwrap();
        for &m in &members {
            claimed.insert(m);
            if m != tail {
                skip.insert(m);
            }
        }
        tail_of.insert(tail, chains.len());
        chains.push(FusedChain {
            primary,
            sides,
            ops,
            output: running,
            out_shape,
            tail_node: tail,
        });
    }

    Fusion {
        chains,
        tail_of,
        skip,
    }
}

/// Codegen the WGSL for one fused chain. Bindings: 0 = primary (full shape),
/// 1..=nsides = side operands (broadcast), nsides+1 = out, nsides+2 = uniform
/// (`total`, out dims, then each side's padded-4D dims).
pub fn build_wgsl(chain: &FusedChain) -> String {
    let nsides = chain.sides.len();
    let mut decls = String::from("@group(0) @binding(0) var<storage, read> p: array<f32>;\n");
    for k in 0..nsides {
        decls.push_str(&format!(
            "@group(0) @binding({}) var<storage, read> side{k}: array<f32>;\n",
            k + 1
        ));
    }
    decls.push_str(&format!(
        "@group(0) @binding({}) var<storage, read_write> out: array<f32>;\n",
        nsides + 1
    ));

    // Uniform: total + padding, out dims, one vec4 per side.
    let mut uni =
        String::from("struct U {\n  total: u32, _a: u32, _b: u32, _c: u32,\n  od: vec4<u32>,\n");
    for k in 0..nsides {
        uni.push_str(&format!("  s{k}: vec4<u32>,\n"));
    }
    uni.push_str("};\n");

    // Op sequence, unrolled over registers.
    let mut body = String::new();
    for op in &chain.ops {
        match op {
            FusedOp::Binary { kind, side, swap } => {
                body.push_str(&format!(
                    "  {{\n    let sd = u.s{side};\n    let si = ((select(c0, 0u, sd.x == 1u) * sd.y \
                     + select(c1, 0u, sd.y == 1u)) * sd.z + select(c2, 0u, sd.z == 1u)) * sd.w \
                     + select(c3, 0u, sd.w == 1u);\n    let s = side{side}[si];\n"
                ));
                let expr = match (kind, swap) {
                    (BinKind::Add, _) => "v + s",
                    (BinKind::Mul, _) => "v * s",
                    (BinKind::Sub, false) => "v - s",
                    (BinKind::Sub, true) => "s - v",
                    (BinKind::Div, false) => "v / s",
                    (BinKind::Div, true) => "s / v",
                };
                body.push_str(&format!("    v = {expr};\n  }}\n"));
            }
            FusedOp::Unary { act } => {
                let expr = match act {
                    Act::Relu => "max(v, 0.0)".to_string(),
                    Act::Sigmoid => "1.0 / (1.0 + exp(-v))".to_string(),
                    Act::Tanh => "tanh(v)".to_string(),
                    Act::Sqrt => "sqrt(v)".to_string(),
                    Act::LeakyRelu { alpha } => {
                        format!("select({alpha:?} * v, v, v > 0.0)")
                    }
                    Act::Clip { min, max } => format!("clamp(v, {min:?}, {max:?})"),
                    Act::HardSigmoid { alpha, beta } => {
                        format!("clamp({alpha:?} * v + {beta:?}, 0.0, 1.0)")
                    }
                };
                body.push_str(&format!("  v = {expr};\n"));
            }
        }
    }

    format!(
        r#"{uni}
{decls}
@group(0) @binding({ubind}) var<uniform> u: U;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {{
  let i = gid.y * (ng.x * 64u) + gid.x;
  if (i >= u.total) {{ return; }}
  let c3 = i % u.od.w;
  let c2 = (i / u.od.w) % u.od.z;
  let c1 = (i / (u.od.w * u.od.z)) % u.od.y;
  let c0 = i / (u.od.w * u.od.z * u.od.y);
  var v: f32 = p[i];
{body}  out[i] = v;
}}
"#,
        ubind = nsides + 2,
    )
}
