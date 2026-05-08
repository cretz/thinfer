use crate::backend::{Backend, Binding, BindingLayout, BufRef};
use crate::tensor::ComputeDtype;

pub mod add;
pub mod bcast_add;
pub mod bcast_affine;
pub mod bcast_fma;
pub mod conv2d;
pub mod group_norm;
pub mod layernorm;
pub mod matmul;
pub mod mul;
pub mod rmsnorm;
pub mod rope;
pub mod scatter_pad_rows;
pub mod sdpa;
pub mod silu;
pub mod silu_mul;
pub mod softmax;
pub mod tanh;
pub mod transpose12;
pub mod upsample2d_nearest;

pub use add::AddF32;
pub(crate) use bcast_add::dispatch_bcast_add;
pub use bcast_add::{BcastAddBufs, BcastAddF32, BcastAddOp};
pub(crate) use bcast_affine::dispatch_bcast_affine;
pub use bcast_affine::{BcastAffineBufs, BcastAffineF32, BcastAffineOp};
pub(crate) use bcast_fma::dispatch_bcast_fma;
pub use bcast_fma::{BcastFmaBufs, BcastFmaF32, BcastFmaOp};
pub(crate) use conv2d::dispatch_conv2d;
pub use conv2d::{Conv2dBufs, Conv2dF32, Conv2dOp};
pub(crate) use group_norm::dispatch_group_norm;
pub use group_norm::{GroupNormBufs, GroupNormF32, GroupNormOp};
pub(crate) use layernorm::dispatch_layernorm;
pub use layernorm::{LayerNormBufs, LayerNormF32, LayerNormOp};
pub(crate) use matmul::dispatch_matmul;
pub use matmul::{MatMulConfig, MatMulF32, MatmulBufs, MatmulOp};
pub use mul::MulF32;
pub(crate) use rmsnorm::dispatch_rmsnorm;
pub use rmsnorm::{RmsNormBufs, RmsNormF32, RmsNormOp};
pub(crate) use rope::dispatch_rope;
pub use rope::{RopeBufs, RopeF32, RopeF32HalfRot, RopeOp};
pub(crate) use scatter_pad_rows::dispatch_scatter_pad_rows;
pub use scatter_pad_rows::{ScatterPadRowsBufs, ScatterPadRowsF32, ScatterPadRowsOp};
pub(crate) use sdpa::dispatch_sdpa;
pub use sdpa::{SdpaBufs, SdpaF32, SdpaF32LargeD, SdpaOp};
pub use silu::SiluF32;
pub use silu_mul::SiluMulF32;
pub(crate) use softmax::dispatch_softmax;
pub use softmax::{SoftmaxBufs, SoftmaxF32, SoftmaxOp};
pub use tanh::TanhF32;
pub(crate) use transpose12::dispatch_transpose12;
pub use transpose12::{Transpose12Bufs, Transpose12F32, Transpose12Op};
pub(crate) use upsample2d_nearest::dispatch_upsample2d_nearest;
pub use upsample2d_nearest::{Upsample2dNearestBufs, Upsample2dNearestF32, Upsample2dNearestOp};

/// WebGPU caps dispatch size at 65535 per dimension. For elementwise kernels
/// with `wgsize` invocations per workgroup operating on N elements, we need
/// ceil(N/wgsize) workgroups; once that exceeds 65535 we have to distribute
/// across the Y dimension. Shaders that consume the result use the
/// `LINEAR_INDEX_PREAMBLE` snippet (or its equivalent) to recover the linear
/// element index. Pass `wgsize` matching the shader's `@workgroup_size`.
pub fn linear_workgroups(n: u32, wgsize: u32) -> [u32; 3] {
    const MAX_DIM: u32 = 65535;
    let total = n.div_ceil(wgsize);
    if total <= MAX_DIM {
        [total, 1, 1]
    } else {
        let y = total.div_ceil(MAX_DIM);
        [MAX_DIM, y, 1]
    }
}

/// Drop-in WGSL preamble for kernels that index a flat output array. Replaces
/// `let i = gid.x;` with a 2D-aware computation. Requires the shader's `main`
/// to declare `@builtin(num_workgroups) ng: vec3<u32>` alongside `gid`.
/// `WG_X` is the shader's `@workgroup_size(X)` (almost always 64).
pub const LINEAR_INDEX_64: &str = "let i = gid.y * (ng.x * 64u) + gid.x;";

/// On-GPU storage layout for weight bindings. F32: declared `array<f32>`,
/// read directly. Bf16: declared `array<u32>` with 2 bf16 elements packed
/// per word, unpacked via a per-binding `load_*` helper that bitcasts the
/// upper-16 bits into f32. Compute is always fp32 in M1; this only affects
/// memory layout and the load path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum WeightDtype {
    #[default]
    F32,
    Bf16,
}

impl WeightDtype {
    pub fn hint(&self) -> &'static str {
        match self {
            Self::F32 => "wf32",
            Self::Bf16 => "wbf16",
        }
    }
}

/// Selects which compiled variant of an op's WGSL to use. Each op trait's
/// `wgsl(&WgslConfig)` returns a single `&'static str`.
///
/// `bf16_quant_writes`: round every activation-producing store to bf16 precision
/// (RNE, NaN/inf passthrough). Compute and accumulator state stay fp32 — only
/// the final memory write quantizes. Matches PyTorch's bf16 dtype semantics.
///
/// `weight_dtype`: on-GPU storage format for weight bindings. See
/// [`WeightDtype`]. Ops without weight bindings ignore this.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WgslConfig {
    pub bf16_quant_writes: bool,
    pub weight_dtype: WeightDtype,
}

impl WgslConfig {
    pub const FP32: Self = Self {
        bf16_quant_writes: false,
        weight_dtype: WeightDtype::F32,
    };
    pub const BF16_QUANT_WRITES: Self = Self {
        bf16_quant_writes: true,
        weight_dtype: WeightDtype::F32,
    };

    /// Short tag for `KernelKey` hints and pipeline-cache disambiguation.
    /// Must change whenever any cfg field changes - the same kernel id with
    /// different `WgslConfig` values needs distinct cache entries.
    pub fn hint(&self) -> &'static str {
        match (self.bf16_quant_writes, self.weight_dtype) {
            (false, WeightDtype::F32) => "",
            (true, WeightDtype::F32) => "bf16q",
            (false, WeightDtype::Bf16) => "wbf16",
            (true, WeightDtype::Bf16) => "bf16q-wbf16",
        }
    }
}

/// WGSL prelude defining `act_store(x)`. Identity in the f32 path; RNE-round
/// to bf16 (NaN/inf passthrough) in the `bf16_quant_writes` path. Used as a
/// literal-producing macro so it composes with `concat!`.
#[macro_export]
macro_rules! act_store_f32 {
    () => {
        "fn act_store(x: f32) -> f32 { return x; }\n"
    };
}
#[macro_export]
macro_rules! act_store_bf16q {
    () => {
        concat!(
            "fn act_store(x: f32) -> f32 { ",
            "let b = bitcast<u32>(x); ",
            "if ((b & 0x7F800000u) == 0x7F800000u) { return x; } ",
            "let l = (b >> 16u) & 1u; ",
            "return bitcast<f32>((b + 0x7FFFu + l) & 0xFFFF0000u); ",
            "}\n",
        )
    };
}

/// Emits a pair of `&'static str` consts from a single WGSL body literal:
/// the fp32 variant and the bf16-quantized-writes variant. Body uses
/// `act_store(x)` placeholders at activation-producing stores. No weight
/// bindings - for ops with weights, see `weight_op_wgsl!` which threads
/// both `act_store` and the `load_*` weight-binding helper.
#[macro_export]
macro_rules! wgsl_with_bf16_variant {
    ($vis:vis $fp32:ident, $bf16q:ident = $body:expr) => {
        $vis const $fp32: &str = concat!($crate::act_store_f32!(), $body);
        $vis const $bf16q: &str = concat!($crate::act_store_bf16q!(), $body);
    };
}

/// Emits 4 `&'static str` consts covering the cross-product of
/// `bf16_quant_writes` x `weight_dtype`. Each op passes raw string literal
/// chunks for the WGSL @compute body plus the per-encoding binding-and-helper
/// preamble. Caller's `wgsl()` matches on `WgslConfig` to pick the variant.
///
/// `body` must reference `act_store(x)` at output stores and `load_<name>(i)`
/// when reading a weight binding. `f32_bindings` and `bf16_bindings` declare
/// the matching `@binding(...) ...` and the corresponding `load_*` helpers.
#[macro_export]
macro_rules! weight_op_wgsl {
    (
        $vis:vis ($f32:ident, $bf16q:ident, $wbf16:ident, $bf16q_wbf16:ident);
        body = $body:expr;
        f32_bindings = $f32b:expr;
        bf16_bindings = $bf16b:expr;
    ) => {
        $vis const $f32: &str = concat!($crate::act_store_f32!(), $f32b, $body);
        $vis const $bf16q: &str = concat!($crate::act_store_bf16q!(), $f32b, $body);
        $vis const $wbf16: &str = concat!($crate::act_store_f32!(), $bf16b, $body);
        $vis const $bf16q_wbf16: &str = concat!($crate::act_store_bf16q!(), $bf16b, $body);
    };
}

/// Like `weight_op_wgsl!` but omits the `bf16_quant_writes` variants. For ops
/// (conv2d, group_norm) whose outputs are not activations crossing the bf16
/// quantization boundary — their `wgsl()` asserts `!bf16_quant_writes`.
#[macro_export]
macro_rules! weight_op_wgsl_no_bf16q {
    (
        $vis:vis ($f32:ident, $wbf16:ident);
        body = $body:expr;
        f32_bindings = $f32b:expr;
        bf16_bindings = $bf16b:expr;
    ) => {
        $vis const $f32: &str = concat!($crate::act_store_f32!(), $f32b, $body);
        $vis const $wbf16: &str = concat!($crate::act_store_f32!(), $bf16b, $body);
    };
}

/// All an op declares is static metadata + WGSL. `dispatch_op` is generic over
/// this trait — adding a new (elementwise / single-output) op is wgsl + a few
/// const lines, no per-op glue.
///
/// Constraints assumed by `dispatch_op`:
/// - Input slots are 0..INPUTS.len(); output slot is INPUTS.len() (last).
/// - All bindings are storage buffers in `Self::Dtype`'s storage layout.
///
/// Ops that need uniforms, weights, dynamic shape, or multi-output get a
/// second trait when we hit one.
pub trait Op {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const INPUTS: &'static [&'static str];
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n: u32) -> [u32; 3] {
        linear_workgroups(n, 64)
    }

    /// Compile-time variants (workgroup, tile, fusion). Empty when unique per kernel id.
    fn hint() -> &'static str {
        ""
    }
}

/// Generic dispatch — works for any `Op`. Caller owns the encoder and the
/// pipeline (looked up via the runtime's `PipelineCache`). Sync; no awaits.
pub(crate) fn dispatch_op<O: Op, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    inputs: &[BufRef],
    output: BufRef,
) -> Result<(), B::Error> {
    debug_assert_eq!(inputs.len(), O::INPUTS.len());
    let n_elems = (output.len / O::Dtype::SIZE as u64) as u32;
    let mut bindings: Vec<Binding> = inputs
        .iter()
        .enumerate()
        .map(|(i, b)| b.binding(i as u32))
        .collect();
    bindings.push(output.binding(inputs.len() as u32));
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n_elems))
}
