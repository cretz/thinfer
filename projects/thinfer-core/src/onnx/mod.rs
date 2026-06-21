//! ONNX graph import + execution on the thinfer wgpu backend.
//!
//! Scope: a focused executor for the inference-only, fixed-input-shape conv-net
//! models the face-swap pipeline needs (SCRFD detector, ArcFace embedder,
//! HyperSwap swapper). It is not a general ONNX runtime: it const-folds the
//! shape subgraph at load (all input shapes are known), so only compute nodes
//! reach the GPU. See `proto` for the wire reader, `shape` for the load-time
//! inference + fold, `exec` for the BatchScope dispatch.

pub mod exec;
pub mod kernels;
pub mod proto;
pub mod shape;

pub use exec::{OnnxError, OnnxModel};

#[cfg(test)]
mod proto_tests {
    use super::proto;

    /// Parse a real ONNX file when `THINFER_ONNX_TEST` points at one; otherwise
    /// skip. Lets us validate the hand-rolled wire reader against the actual
    /// face-swap models without committing a fixture.
    #[test]
    fn parse_real_model_if_present() {
        let Ok(path) = std::env::var("THINFER_ONNX_TEST") else {
            eprintln!("THINFER_ONNX_TEST unset; skipping");
            return;
        };
        let bytes = std::fs::read(&path).expect("read onnx");
        let g = proto::parse_model(&bytes).expect("parse onnx");
        eprintln!(
            "parsed {}: {} nodes, {} initializers, {} inputs, {} outputs",
            path,
            g.nodes.len(),
            g.initializers.len(),
            g.inputs.len(),
            g.outputs.len(),
        );
        assert!(!g.nodes.is_empty());
        assert!(!g.initializers.is_empty());
        // Spot-check: every initializer decoded to a non-empty payload whose
        // element count matches its declared dims.
        for t in &g.initializers {
            assert_eq!(
                t.data.len(),
                t.numel(),
                "initializer {} decoded {} elems but dims say {}",
                t.name,
                t.data.len(),
                t.numel(),
            );
        }
        for n in &g.nodes {
            assert!(!n.op_type.is_empty(), "node with empty op_type");
        }
    }

    /// Plan a real model: bind input shapes, run shape-inference + fold, then
    /// report the distinct compute op types and the view/compute step counts.
    /// Asserts the analysis covers every node (no Unsupported).
    #[test]
    fn plan_real_model_if_present() {
        use super::shape;
        use std::collections::{BTreeSet, HashMap};
        let Ok(path) = std::env::var("THINFER_ONNX_TEST") else {
            return;
        };
        let bytes = std::fs::read(&path).expect("read onnx");
        let g = proto::parse_model(&bytes).expect("parse onnx");
        // Bind dynamic inputs to the face-swap runtime shapes.
        let mut ins: HashMap<String, Vec<i64>> = HashMap::new();
        for vi in &g.inputs {
            let dims: Vec<i64> = vi.dims.iter().map(|d| d.unwrap_or(0)).collect();
            let bound = match (vi.name.as_str(), dims.as_slice()) {
                ("input.1", _) => vec![1, 3, 640, 640], // SCRFD dynamic HxW
                (_, d) => d.iter().map(|&x| if x <= 0 { 1 } else { x }).collect(),
            };
            ins.insert(vi.name.clone(), bound);
        }
        let plan = shape::plan(&g, &ins).expect("plan");
        let mut compute_ops: BTreeSet<&str> = BTreeSet::new();
        let mut n_view = 0;
        let mut n_compute = 0;
        for step in &plan.steps {
            match step {
                shape::Step::View { .. } => n_view += 1,
                shape::Step::Compute { node_idx } => {
                    n_compute += 1;
                    compute_ops.insert(g.nodes[*node_idx].op_type.as_str());
                }
            }
        }
        eprintln!(
            "plan {}: {} compute steps, {} view steps, {} consts; compute ops: {:?}",
            path,
            n_compute,
            n_view,
            plan.consts.len(),
            compute_ops,
        );
        // Every graph output must have a resolved shape.
        for o in &g.outputs {
            let s = plan.shape_of(&o.name).expect("output shape");
            eprintln!("  output {}: {:?}", o.name, s);
            assert!(s.iter().all(|&d| d > 0), "non-positive output dim");
        }
    }
}
