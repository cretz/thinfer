use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
#[cfg(feature = "conformance")]
use crate::conformance::{
    DTYPES_FP32_ONLY, Dtype, OpSpec, OpTest, OpTestContext, TestCase, linspace, t,
};
use crate::ops::{ActDtype, WgslConfig};
use crate::tensor::{ComputeDtype, F32};

/// MemBlock input assembly for the LightTAE / TAEHV video decoder. Given a
/// frame-major activation `x` shaped `[T, C, H, W]`, produce `[T, 2C, H, W]`
/// where output channels `[0, C)` are the current frame `x[t]` and `[C, 2C)`
/// are the previous frame `x[t-1]`. This is exactly the reference
/// `torch.cat([x, past], dim=1)` with the causal `past` = one-frame shift
/// (`F.pad(..., (...,1,0))[:, :T]`), fused into a single dispatch over a single
/// input (no separate shifted/zeroed buffer).
///
/// The `t == 0` "past" is the carry frame `prev` (`[C, H, W]`) when `has_prev`
/// is set (temporal-chunk tiling: `prev` is the previous chunk's trailing input
/// frame so chunk boundaries see real history), else zero (the untiled / first-
/// chunk case, bit-identical to the original zero-pad). `prev` is always bound;
/// pass any same-or-larger buffer with `has_prev = 0` when there is no carry.
///
/// Layout: 0=x, 1=Out, 2=Uniform, 3=Prev.
pub trait MemCatOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> String;
    fn layout() -> &'static [BindingLayout];

    /// One thread per output element.
    fn workgroups(n_out: u32) -> [u32; 3] {
        super::linear_workgroups(n_out, 64)
    }
}

pub struct MemCatBufs<'a> {
    pub x: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
    /// Carry frame `[C, H, W]` for the `t == 0` "past" (used iff the uniform's
    /// `has_prev != 0`). Always bound; pass `x` itself when there is no carry.
    pub prev: &'a BufRef,
}

pub(crate) fn dispatch_memcat<O: MemCatOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &MemCatBufs<'_>,
    n_out: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.out.binding(1),
        bufs.uniform.binding(2),
        bufs.prev.binding(3),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_out))
}

fn build_wgsl(cfg: &WgslConfig) -> String {
    assert!(
        !cfg.bf16_quant_writes,
        "memcat has no bf16-quant-write mode"
    );
    let (f16_prelude, x_decl, out_decl, prev_decl, load_fns, store_expr) = match cfg.act_dtype {
        ActDtype::F32 => (
            "",
            "@group(0) @binding(0) var<storage, read> x: array<f32>;",
            "@group(0) @binding(1) var<storage, read_write> out: array<f32>;",
            "@group(0) @binding(3) var<storage, read> prev: array<f32>;",
            "fn load_x(i: u32) -> f32 { return x[i]; }\nfn load_prev(i: u32) -> f32 { return prev[i]; }\n",
            "v",
        ),
        ActDtype::F16 => (
            "enable f16;\n",
            "@group(0) @binding(0) var<storage, read> x: array<f16>;",
            "@group(0) @binding(1) var<storage, read_write> out: array<f16>;",
            "@group(0) @binding(3) var<storage, read> prev: array<f16>;",
            "fn load_x(i: u32) -> f32 { return f32(x[i]); }\nfn load_prev(i: u32) -> f32 { return f32(prev[i]); }\n",
            "f16(clamp(v, -65504.0, 65504.0))",
        ),
        other => panic!("memcat does not support act dtype {other:?}"),
    };

    format!(
        r#"{f16_prelude}
struct U {{ t: u32, c: u32, h: u32, w: u32, has_prev: u32, }};

{x_decl}
{out_decl}
@group(0) @binding(2) var<uniform> u: U;
{prev_decl}

{load_fns}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {{
    let o: u32 = gid.y * (ng.x * 64u) + gid.x;
    let hw: u32 = u.h * u.w;
    let c2: u32 = u.c * 2u;
    let n_out: u32 = u.t * c2 * hw;
    if (o >= n_out) {{ return; }}

    // Decode flat output index over [T, 2C, H, W].
    let s: u32 = o % hw;
    var rem: u32 = o / hw;
    let cc: u32 = rem % c2;
    let tt: u32 = rem / c2;

    var v: f32 = 0.0;
    if (cc < u.c) {{
        // current frame
        v = load_x((tt * u.c + cc) * hw + s);
    }} else {{
        // previous frame: x[t-1] in-batch, or the carry frame at the chunk edge
        // (t == 0), or zero when there is no carry.
        let pc: u32 = cc - u.c;
        if (tt > 0u) {{
            v = load_x(((tt - 1u) * u.c + pc) * hw + s);
        }} else if (u.has_prev != 0u) {{
            v = load_prev(pc * hw + s);
        }}
    }}
    out[o] = {store_expr};
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
        kind: BindingKind::StorageReadWrite,
    },
    BindingLayout {
        slot: 2,
        kind: BindingKind::Uniform,
    },
    BindingLayout {
        slot: 3,
        kind: BindingKind::StorageRead,
    },
];

pub struct MemCatF32;

impl MemCatOp for MemCatF32 {
    const KERNEL_ID: &'static str = "memcat.f32";
    type Dtype = F32;
    const X: &'static str = "memcat/x";
    const OUTPUT: &'static str = "memcat/out";
    fn wgsl(cfg: &WgslConfig) -> String {
        build_wgsl(cfg)
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl OpTest for MemCatF32 {
    fn dtypes(&self) -> &'static [Dtype] {
        // Pure data movement, F32 act path only (no bf16-write / packed mode).
        DTYPES_FP32_ONLY
    }
    fn test_cases(&self) -> Vec<TestCase> {
        // x = [T=2, C=3, H=2, W=2]; checks the t=0 zero-pad and t=1 shift.
        vec![TestCase {
            name: "memcat_basic",
            op: OpSpec::Memcat,
            inputs: vec![t("x", [2, 3, 2, 2], linspace(-4.0, 4.0, false))],
        }]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_memcat::<MemCatF32>())
    }
}
