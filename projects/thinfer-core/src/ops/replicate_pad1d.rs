use super::{ActDtype, WgslConfig};
use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
use crate::tensor::{ComputeDtype, F32};

/// Edge-replicate padding along the length axis of an NCL tensor:
/// `out[b, c, j] = x[b, c, clamp(j - lpad, 0, Lin-1)]`, producing
/// `[B, C, Lin + lpad + rpad]`.
///
/// BigVGAN's anti-alias `UpSample1d`/`DownSample1d` and the Hann resampler do
/// `F.pad(x, (lpad, rpad), mode="replicate")` before a zero-pad (transposed)
/// conv. Materializing the replicate pad as its own gather keeps
/// [`super::conv1d`]/[`super::conv_transpose1d`] pure zero-pad. One thread per
/// output element.
///
/// Layout: 0=X, 1=Out, 2=Uniform `{b, channels, lin, lout, lpad, _, _, _}`.
pub trait ReplicatePad1dOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const DIMS: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> String;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n_out: u32) -> [u32; 3] {
        crate::ops::linear_workgroups(n_out, 64)
    }
}

pub struct ReplicatePad1dBufs<'a> {
    pub x: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

/// Uniform `[b, channels, lin, lout, src_off (i32 bits), 0, 0, 0]` (32 bytes).
/// `out[c, j] = x[c, clamp(j + src_off, 0, lin-1)]`.
fn uniform_bytes(b: u32, channels: u32, lin: u32, lout: u32, src_off: i32) -> [u8; 32] {
    let fields: [u32; 8] = [b, channels, lin, lout, src_off as u32, 0, 0, 0];
    let mut bytes = [0u8; 32];
    for (i, v) in fields.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Edge-replicate pad: `lpad`/`rpad` samples on each side (`src_off = -lpad`).
pub fn replicate_pad1d_uniform_bytes(
    b: u32,
    channels: u32,
    lin: u32,
    lpad: u32,
    rpad: u32,
) -> [u8; 32] {
    uniform_bytes(b, channels, lin, lin + lpad + rpad, -(lpad as i32))
}

/// Crop/slice `lout` samples starting at `start` (`src_off = start`; in-range so
/// no replication). The same clamped-gather kernel covers both pad and crop.
pub fn crop1d_uniform_bytes(b: u32, channels: u32, lin: u32, start: u32, lout: u32) -> [u8; 32] {
    uniform_bytes(b, channels, lin, lout, start as i32)
}

pub(crate) fn dispatch_replicate_pad1d<O: ReplicatePad1dOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &ReplicatePad1dBufs<'_>,
    n_out: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.out.binding(1),
        bufs.uniform.binding(2),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_out))
}

fn build_wgsl(cfg: &WgslConfig) -> String {
    assert!(
        !cfg.bf16_quant_writes,
        "replicate_pad1d has no bf16-quant-write mode"
    );
    let (f16_prelude, x_decl, out_decl, load_x, store_expr) = match cfg.act_dtype {
        ActDtype::F32 => (
            "",
            "@group(0) @binding(0) var<storage, read> x: array<f32>;",
            "@group(0) @binding(1) var<storage, read_write> out: array<f32>;",
            "fn load_x(i: u32) -> f32 { return x[i]; }\n",
            "v",
        ),
        other => panic!("replicate_pad1d does not support act dtype {other:?}"),
    };

    format!(
        r#"{f16_prelude}
struct U {{ b: u32, channels: u32, lin: u32, lout: u32, src_off: i32, _p0: u32, _p1: u32, _p2: u32 }};

{x_decl}
{out_decl}
@group(0) @binding(2) var<uniform> u: U;

{load_x}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {{
    let idx: u32 = gid.y * (ng.x * 64u) + gid.x;
    let n_out: u32 = u.b * u.channels * u.lout;
    if (idx >= n_out) {{ return; }}
    let bc: u32 = idx / u.lout;
    let j: u32 = idx - bc * u.lout;
    // clamp(j + src_off, 0, lin-1): replicate-pad (src_off<0) or crop (src_off>=0).
    let src_s: i32 = i32(j) + u.src_off;
    let src: u32 = u32(clamp(src_s, 0, i32(u.lin) - 1));
    let v: f32 = load_x(bc * u.lin + src);
    out[idx] = {store_expr};
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
];

pub struct ReplicatePad1dF32;

impl ReplicatePad1dOp for ReplicatePad1dF32 {
    const KERNEL_ID: &'static str = "replicate_pad1d.f32";
    type Dtype = F32;
    const X: &'static str = "replicate_pad1d/x";
    const DIMS: &'static str = "replicate_pad1d/dims";
    const OUTPUT: &'static str = "replicate_pad1d/out";
    fn wgsl(cfg: &WgslConfig) -> String {
        build_wgsl(cfg)
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl crate::conformance::OpTest for ReplicatePad1dF32 {
    fn dtypes(&self) -> &'static [crate::conformance::Dtype] {
        crate::conformance::DTYPES_FP32_ONLY
    }
    fn test_cases(&self) -> Vec<crate::conformance::TestCase> {
        use crate::conformance::{OpSpec, TestCase, linspace, t};
        vec![TestCase {
            name: "replicate_pad1d_basic",
            op: OpSpec::ReplicatePad1d { lpad: 5, rpad: 6 },
            inputs: vec![t("x", [1, 3, 10], linspace(-1.0, 1.0, false))],
        }]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a crate::conformance::OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_replicate_pad1d::<ReplicatePad1dF32>())
    }
}
