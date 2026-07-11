use super::{ActDtype, WgslConfig};
use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
use crate::tensor::{ComputeDtype, F32};

/// BigVGAN v2 SnakeBeta activation over NCL (per-channel `alpha`/`beta`, both
/// stored log-scale): `out = x + (1/(exp(beta)+eps)) * sin(exp(alpha)*x)^2`,
/// `eps = 1e-9`.
///
/// Shapes (row-major): `x: [B, C, L]`, `alpha: [C]`, `beta: [C]`, `out: [B,C,L]`.
/// The channel of a flat element `i` is `(i / L) % C`. One thread per element.
///
/// `alpha`/`beta` are bound as f32 activation buffers (the whole vocoder tail
/// runs f32; the loader reads the bf16 params to f32 and uploads them). No weight
/// dtype variants.
///
/// Layout: 0=X, 1=Alpha, 2=Beta, 3=Out, 4=Uniform `{n, channels, inner, eps}`.
pub trait SnakeBetaOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const X: &'static str;
    const ALPHA: &'static str;
    const BETA: &'static str;
    const DIMS: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> String;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n: u32) -> [u32; 3] {
        crate::ops::linear_workgroups(n, 64)
    }
}

pub struct SnakeBetaBufs<'a> {
    pub x: &'a BufRef,
    pub alpha: &'a BufRef,
    pub beta: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

/// Uniform `[n, channels, inner, eps_bits]` (16 bytes).
pub fn snake_beta_uniform_bytes(n: u32, channels: u32, inner: u32, eps: f32) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&n.to_le_bytes());
    bytes[4..8].copy_from_slice(&channels.to_le_bytes());
    bytes[8..12].copy_from_slice(&inner.to_le_bytes());
    bytes[12..16].copy_from_slice(&eps.to_bits().to_le_bytes());
    bytes
}

pub(crate) fn dispatch_snake_beta<O: SnakeBetaOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &SnakeBetaBufs<'_>,
    n: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.x.binding(0),
        bufs.alpha.binding(1),
        bufs.beta.binding(2),
        bufs.out.binding(3),
        bufs.uniform.binding(4),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n))
}

fn build_wgsl(cfg: &WgslConfig) -> String {
    assert!(
        !cfg.bf16_quant_writes,
        "snake_beta has no bf16-quant-write mode"
    );
    let (f16_prelude, x_decl, out_decl, load_x, store_expr) = match cfg.act_dtype {
        ActDtype::F32 => (
            "",
            "@group(0) @binding(0) var<storage, read> x: array<f32>;",
            "@group(0) @binding(3) var<storage, read_write> out: array<f32>;",
            "fn load_x(i: u32) -> f32 { return x[i]; }\n",
            "v",
        ),
        other => panic!("snake_beta does not support act dtype {other:?}"),
    };

    format!(
        r#"{f16_prelude}
struct U {{ n: u32, channels: u32, inner: u32, eps_bits: u32 }};

{x_decl}
@group(0) @binding(1) var<storage, read> alpha: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
{out_decl}
@group(0) @binding(4) var<uniform> u: U;

{load_x}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {{
    let i: u32 = gid.y * (ng.x * 64u) + gid.x;
    if (i >= u.n) {{ return; }}
    let c: u32 = (i / u.inner) % u.channels;
    let eps: f32 = bitcast<f32>(u.eps_bits);
    let a: f32 = exp(alpha[c]);
    let b: f32 = exp(beta[c]);
    let xv: f32 = load_x(i);
    let s: f32 = sin(a * xv);
    let v: f32 = xv + (1.0 / (b + eps)) * s * s;
    out[i] = {store_expr};
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

pub struct SnakeBetaF32;

impl SnakeBetaOp for SnakeBetaF32 {
    const KERNEL_ID: &'static str = "snake_beta.f32";
    type Dtype = F32;
    const X: &'static str = "snake_beta/x";
    const ALPHA: &'static str = "snake_beta/alpha";
    const BETA: &'static str = "snake_beta/beta";
    const DIMS: &'static str = "snake_beta/dims";
    const OUTPUT: &'static str = "snake_beta/out";
    fn wgsl(cfg: &WgslConfig) -> String {
        build_wgsl(cfg)
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

#[cfg(feature = "conformance")]
impl crate::conformance::OpTest for SnakeBetaF32 {
    fn dtypes(&self) -> &'static [crate::conformance::Dtype] {
        crate::conformance::DTYPES_FP32_ONLY
    }
    fn test_cases(&self) -> Vec<crate::conformance::TestCase> {
        use crate::conformance::{OpSpec, TestCase, linspace, t};
        vec![TestCase {
            name: "snake_beta_basic",
            op: OpSpec::SnakeBeta { eps: 1e-9 },
            inputs: vec![
                t("x", [1, 4, 8], linspace(-3.0, 3.0, false)),
                t("alpha", [4], linspace(-0.5, 0.5, false)),
                t("beta", [4], linspace(-0.3, 0.3, true)),
            ],
        }]
    }
    fn run_test<'a>(
        &self,
        ctx: &'a crate::conformance::OpTestContext<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>> {
        Box::pin(ctx.run_snake_beta::<SnakeBetaF32>())
    }
}
