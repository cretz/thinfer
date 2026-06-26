use super::{ActDtype, Op, WgslConfig};
use crate::backend::{BindingKind, BindingLayout};
use crate::tensor::F32;
use crate::{act_bf16_prelude, act_f16_prelude, act_store_bf16q, act_store_f32};

// Per-head sigmoid gating: `out[i] = x[i] * 2*sigmoid(gate[i / head_dim])`.
//
// LTX-2's `apply_gated_attention` multiplies the attention output by a per-head
// learned gate (`out = attn_out.view(B,T,H,D) * (2*sigmoid(to_gate_logits(x)))`,
// `ltx_core/model/transformer/ops.py:PytorchGatedAttention`). The attention
// output is laid out `[rows, heads*head_dim]` and the gate `[rows, heads]`, so
// the flat element index `i` maps to gate slot `i / head_dim` exactly: with
// `x[i] = x[(row*heads + head)*head_dim + d]`, `i / head_dim = row*heads + head`,
// which is the gate's flat index. `head_dim = len(x) / len(gate)` is recovered
// in-kernel, so no uniform is needed (the same trick as `gelu`'s arrayLength
// guard). The `2*sigmoid` is folded in here.
//
// F32 acts (the once-per-request Gemma connector) AND packed f16/bf16 (the LTX
// DiT's gated attention under f16 acts). The packed variants process one word
// (two consecutive elements) at a time; they read the gate slot from element
// `2w` and apply it to BOTH elements of the word. This is correct iff `head_dim`
// is even (so a head boundary never falls between the two elements of a word) --
// true for the LTX DiT (head_dim 128 video / 64 audio). Do NOT use the packed
// path with an odd head_dim.

macro_rules! f32_body {
    () => {
        r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    if (i >= arrayLength(&out)) { return; }
    let head_dim = arrayLength(&x) / arrayLength(&gate);
    let g = gate[i / head_dim];
    let s = 2.0 / (1.0 + exp(-g));
    out[i] = act_store(x[i] * s);
}
"#
    };
}

const WGSL_F32: &str = concat!(act_store_f32!(), f32_body!());
const WGSL_F32_BF16: &str = concat!(act_store_bf16q!(), f32_body!());

// Packed body: one word = two consecutive elements `2w`, `2w+1`. `head_dim`
// (elems) = arrayLength(&x)/arrayLength(&gate) (both arrays 2-per-word, so the
// word ratio equals the elem ratio). The gate slot `(2w)/head_dim` is shared by
// both elements (head_dim even); read it from the packed gate buffer (slot/2
// word, slot&1 lane). bf16 stores words as `u32`; f16 as `vec2<f16>`.
const WGSL_BF16_PACKED: &str = concat!(
    act_bf16_prelude!(),
    r#"
@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> gate: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let w = gid.y * (ng.x * 64u) + gid.x;
    if (w >= arrayLength(&out)) { return; }
    let head_dim = arrayLength(&x) / arrayLength(&gate);
    let gslot = (w * 2u) / head_dim;
    let gw = unpack_bf16x2(gate[gslot / 2u]);
    let g = select(gw.x, gw.y, (gslot & 1u) == 1u);
    let s = 2.0 / (1.0 + exp(-g));
    let xv = unpack_bf16x2(x[w]);
    out[w] = pack_bf16x2(xv.x * s, xv.y * s);
}
"#
);

const WGSL_F16_PACKED: &str = concat!(
    act_f16_prelude!(),
    r#"
@group(0) @binding(0) var<storage, read> x: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read> gate: array<vec2<f16>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec2<f16>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let w = gid.y * (ng.x * 64u) + gid.x;
    if (w >= arrayLength(&out)) { return; }
    let head_dim = arrayLength(&x) / arrayLength(&gate);
    let gslot = (w * 2u) / head_dim;
    let gw = unpack_f16x2_f32(gate[gslot / 2u]);
    let g = select(gw.x, gw.y, (gslot & 1u) == 1u);
    let s = 2.0 / (1.0 + exp(-g));
    let xv = unpack_f16x2_f32(x[w]);
    out[w] = pack_f32_to_f16x2(xv.x * s, xv.y * s);
}
"#
);

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
];

pub struct GatedHeadMulF32;

impl Op for GatedHeadMulF32 {
    const KERNEL_ID: &'static str = "gated_head_mul.f32";
    type Dtype = F32;
    const INPUTS: &'static [&'static str] = &["gated_head_mul/x", "gated_head_mul/gate"];
    const OUTPUT: &'static str = "gated_head_mul/out";
    fn wgsl(cfg: &WgslConfig) -> &'static str {
        match (cfg.act_dtype, cfg.bf16_quant_writes) {
            (ActDtype::F32, false) => WGSL_F32,
            (ActDtype::F32, true) => WGSL_F32_BF16,
            (ActDtype::Bf16, _) => WGSL_BF16_PACKED,
            (ActDtype::F16, _) => WGSL_F16_PACKED,
            (ActDtype::I8, _) => unreachable!("ActDtype::I8 is never a block-level act dtype"),
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl crate::conformance::OpTest for GatedHeadMulF32 {
    fn dtypes(&self) -> &'static [crate::conformance::Dtype] {
        crate::conformance::DTYPES_FP32_ONLY
    }
    fn test_cases(&self) -> Vec<crate::conformance::TestCase> {
        use crate::conformance::{OpSpec, TestCase, linspace, t};
        // rows=2, heads=4, head_dim=8 -> x [2*4*8=64], gate [2*4=8]. The kernel
        // recovers head_dim = len(x)/len(gate) = 8.
        vec![TestCase {
            name: "gated_head_mul_basic",
            op: OpSpec::GatedHeadMul,
            inputs: vec![
                t("x", [64], linspace(-2.0, 2.0, false)),
                t("gate", [8], linspace(-3.0, 3.0, true)),
            ],
        }]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a crate::conformance::OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_elementwise::<GatedHeadMulF32>())
    }
}
