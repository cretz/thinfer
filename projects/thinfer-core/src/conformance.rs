//! Pytorch-delta op harness, shared between native and web conformance runs.
//!
//! Op files own their `impl OpTest` (cases + one-line GPU invocation). All
//! pipeline/alloc/dispatch/readback machinery lives in `OpTestContext::run_*`
//! (one method per op-shape-family), not per op.
//!
//! Trait + spec types are exposed under the `conformance` feature so external
//! drivers (`thinfer-conformance`, `thinfer-web` tests) can build a registry
//! and run it against the same WgpuBackend.

use crate::backend::{Backend, BindingLayout, BufRef, WgpuBackend, WgpuError};
use crate::ops::{
    AddF32, BcastAddBufs, BcastAddF32, BcastAddOp, BcastAffineBufs, BcastAffineF32, BcastAffineOp,
    BcastFmaBufs, BcastFmaF32, BcastFmaOp, BcastModulateBufs, BcastModulateF32, BcastModulateOp,
    BcastMulF32, Conv2dBufs, Conv2dF32, Conv2dOp, Conv3dBufs, Conv3dF32, Conv3dOp, GeluMulF32,
    LayerNormBufs, LayerNormF32, LayerNormOp, MatMulF32, MatmulBufs, MatmulOp, MemCatBufs,
    MemCatOp, MulF32, Op, ReluF32, RmsNorm3dBufs, RmsNorm3dF32, RmsNorm3dOp, RmsNormBufs,
    RmsNormF32, RmsNormOp, RopeBufs, RopeF32, RopeF32HalfRot, RopeOp, SdpaBufs, SdpaF32,
    SdpaF32LargeD, SdpaOp, SiluF32, SiluMulF32, SoftmaxBufs, SoftmaxF32, SoftmaxOp, TanhF32,
    Transpose12Bufs, Transpose12F32, Transpose12Op, WgslConfig, dispatch_bcast_add,
    dispatch_bcast_affine, dispatch_bcast_fma, dispatch_bcast_modulate, dispatch_conv2d,
    dispatch_conv3d, dispatch_layernorm, dispatch_matmul, dispatch_memcat, dispatch_op,
    dispatch_rmsnorm, dispatch_rmsnorm3d, dispatch_rope, dispatch_sdpa, dispatch_softmax,
    dispatch_transpose12,
};
use crate::tensor::ComputeDtype;
use safetensors::SafeTensors;
use serde::Serialize;
use std::pin::Pin;

// ---------- spec ----------

#[derive(Serialize, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    /// fp32 storage and fp32 outputs — baseline parity.
    Fp32,
    /// fp32 storage, but every activation-producing store is RNE-quantized
    /// to bf16. Mirrors pytorch's `--dtype bf16` module-output semantics.
    /// Ops that don't participate in a bf16 boundary (VAE-only ops like
    /// conv2d, transpose12, softmax) opt out via `OpTest::dtypes`.
    #[serde(rename = "bf16w")]
    Bf16Writes,
    /// Packed bf16 activation storage: `array<u32>` on the GPU, 2 elems/word.
    /// Halves activation VRAM and bandwidth. Inputs and outputs are stored as
    /// native bf16 (2 bytes/elem) in the safetensors fixture; both sides land
    /// on the same bf16 bit pattern so tolerance stays at fp32 levels. Ops
    /// roll into this dtype incrementally via `OpTest::dtypes` overrides.
    #[serde(rename = "bf16p")]
    Bf16Packed,
}

impl Dtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Dtype::Fp32 => "fp32",
            Dtype::Bf16Writes => "bf16w",
            Dtype::Bf16Packed => "bf16p",
        }
    }

    /// `WgslConfig` the op's `wgsl()` is invoked with for this dtype.
    pub fn wgsl_config(self) -> &'static WgslConfig {
        match self {
            Dtype::Fp32 => &WgslConfig::FP32,
            Dtype::Bf16Writes => &WgslConfig::BF16_QUANT_WRITES,
            Dtype::Bf16Packed => &WgslConfig::BF16_PACKED,
        }
    }

    /// Bytes per activation element on-disk and on-GPU for this dtype.
    pub const fn bytes_per_elem(self) -> u64 {
        match self {
            Dtype::Fp32 | Dtype::Bf16Writes => 4,
            Dtype::Bf16Packed => 2,
        }
    }
}

/// Default dtype list every bf16-capable op runs in. Fp32-only ops override
/// `OpTest::dtypes` to `DTYPES_FP32_ONLY`. Ops that have rolled into packed
/// bf16 storage override to `DTYPES_ACT_BF16`.
pub const DTYPES_DEFAULT: &[Dtype] = &[Dtype::Fp32, Dtype::Bf16Writes];
pub const DTYPES_ACT_BF16: &[Dtype] = &[Dtype::Fp32, Dtype::Bf16Writes, Dtype::Bf16Packed];
pub const DTYPES_FP32_ONLY: &[Dtype] = &[Dtype::Fp32];

#[derive(Serialize, Clone)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum OpSpec {
    Add,
    Mul,
    Silu,
    Relu,
    /// MemBlock input assembly: `[T,C,H,W]` -> `[T,2C,H,W]` (current frame ++
    /// previous frame on the channel axis, zero at t=0). See `ops::memcat`.
    Memcat,
    #[serde(rename = "silu_mul")]
    SiluMul,
    #[serde(rename = "gelu_mul")]
    GeluMul,
    Tanh,
    Matmul,
    Rmsnorm {
        eps: f32,
    },
    Layernorm {
        eps: f32,
    },
    Softmax,
    Rope,
    #[serde(rename = "rope_halfrot")]
    RopeHalfRot,
    Sdpa {
        scale: f32,
    },
    Transpose12,
    #[serde(rename = "bcast_affine")]
    BcastAffine {
        bias: f32,
    },
    #[serde(rename = "bcast_fma")]
    BcastFma,
    #[serde(rename = "bcast_modulate")]
    BcastModulate {
        bias: f32,
    },
    #[serde(rename = "bcast_add")]
    BcastAdd,
    #[serde(rename = "bcast_mul")]
    BcastMul,
    Conv2d {
        kh: u32,
        kw: u32,
        pad_h: u32,
        pad_w: u32,
        stride_h: u32,
        stride_w: u32,
    },
    Conv3d {
        kt: u32,
        kh: u32,
        kw: u32,
        pad_t: u32,
        pad_h: u32,
        pad_w: u32,
        stride_t: u32,
        stride_h: u32,
        stride_w: u32,
    },
    #[serde(rename = "rmsnorm3d")]
    RmsNorm3d,
}

#[derive(Serialize, Clone)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Fill {
    Linspace {
        lo: f32,
        hi: f32,
        #[serde(default, skip_serializing_if = "is_false")]
        flip: bool,
    },
}

fn is_false(b: &bool) -> bool {
    !*b
}

pub fn linspace(lo: f32, hi: f32, flip: bool) -> Fill {
    Fill::Linspace { lo, hi, flip }
}

#[derive(Serialize, Clone)]
pub struct Tensor {
    pub name: &'static str,
    pub shape: Vec<usize>,
    pub fill: Fill,
}

pub fn t(name: &'static str, shape: impl Into<Vec<usize>>, fill: Fill) -> Tensor {
    Tensor {
        name,
        shape: shape.into(),
        fill,
    }
}

/// Per-op spec entry authored in op files (no dtype — parameterization is
/// declared by the op via `OpTest::dtypes`).
#[derive(Clone)]
pub struct TestCase {
    pub name: &'static str,
    pub op: OpSpec,
    pub inputs: Vec<Tensor>,
}

/// Serialization shape of a single case: `TestCase` plus the dtype list the
/// op wants this case generated for. Built by the harness from
/// `(OpTest::test_cases, OpTest::dtypes)`.
#[derive(Serialize, Clone)]
pub struct SpecCase {
    pub name: &'static str,
    #[serde(flatten)]
    pub op: OpSpec,
    pub inputs: Vec<Tensor>,
    pub dtypes: Vec<Dtype>,
}

#[derive(Serialize)]
pub struct SpecPayload<'a> {
    pub cases: &'a [SpecCase],
}

// ---------- trait ----------

pub trait OpTest {
    fn test_cases(&self) -> Vec<TestCase>;
    /// Dtypes this op runs in. Default: fp32 + bf16-writes. Fp32-only ops
    /// (VAE conv2d, transpose12, softmax) override to `DTYPES_FP32_ONLY`.
    fn dtypes(&self) -> &'static [Dtype] {
        DTYPES_DEFAULT
    }
    fn run_test<'a>(
        &self,
        ctx: &'a OpTestContext<'a>,
    ) -> Pin<Box<dyn std::future::Future<Output = Vec<u8>> + 'a>>;
}

// ---------- context ----------

pub struct OpTestContext<'a> {
    pub backend: &'a WgpuBackend,
    pub st: &'a SafeTensors<'a>,
    pub case: &'a TestCase,
    /// Which dtype variant of `case` is being run now. `run_*` helpers pass
    /// `dtype.wgsl_config()` into `O::wgsl(...)`.
    pub dtype: Dtype,
}

impl<'a> OpTestContext<'a> {
    fn input_bytes(&self, name: &str) -> &'a [u8] {
        let key = format!("{}/{}", self.case.name, name);
        self.st
            .tensor(&key)
            .unwrap_or_else(|e| panic!("safetensors missing {key}: {e}"))
            .data()
    }

    fn alloc_with(&self, bytes: &[u8]) -> BufRef {
        let len = bytes.len() as u64;
        let id = self.backend.allocate(len).expect("allocate");
        self.backend
            .write_buffer(id, 0, bytes)
            .expect("write_buffer");
        BufRef::new(id, len)
    }

    fn alloc_empty(&self, len: u64) -> BufRef {
        let id = self.backend.allocate(len).expect("allocate");
        BufRef::new(id, len)
    }

    /// Common scaffolding: build pipeline, alloc each `case.inputs` entry into a
    /// `BufRef`, optionally alloc a uniform buffer, alloc the output, run a
    /// caller-supplied dispatch closure inside a fresh command encoder, submit,
    /// readback, and free everything. Each per-op `run_*` is just uniform
    /// packing + a one-line dispatch call.
    async fn run_op<F>(
        &self,
        wgsl: &str,
        layout: fn() -> &'static [BindingLayout],
        uniform: Option<&[u8]>,
        out_len: u64,
        dispatch: F,
    ) -> Vec<u8>
    where
        F: FnOnce(
            &WgpuBackend,
            &mut <WgpuBackend as Backend>::CommandEncoder,
            &<WgpuBackend as Backend>::Pipeline,
            &[BufRef],
            Option<BufRef>,
            BufRef,
        ) -> Result<(), WgpuError>,
    {
        let pipeline = self
            .backend
            .create_pipeline(self.case.name, wgsl, "main", layout())
            .await
            .expect("pipeline");
        let inputs: Vec<BufRef> = self
            .case
            .inputs
            .iter()
            .map(|t| self.alloc_with(self.input_bytes(t.name)))
            .collect();
        let uniform_buf = uniform.map(|b| self.alloc_with(b));
        let out = self.alloc_empty(out_len);

        let mut enc = self.backend.create_command_encoder();
        dispatch(self.backend, &mut enc, &pipeline, &inputs, uniform_buf, out).expect("dispatch");
        self.backend.submit(enc).await.expect("submit");
        let got = self
            .backend
            .read_buffer(out.id, 0, out.len)
            .await
            .expect("read");

        let mut to_free = inputs;
        if let Some(u) = uniform_buf {
            to_free.push(u);
        }
        to_free.push(out);
        for b in &to_free {
            self.backend.free(b.id);
        }
        got
    }

    pub async fn run_elementwise<O: Op>(&self) -> Vec<u8> {
        let out_len = self
            .case
            .inputs
            .iter()
            .map(|t| self.input_bytes(t.name).len())
            .max()
            .unwrap() as u64;
        debug_assert_eq!(out_len % O::Dtype::SIZE as u64, 0);
        self.run_op(
            O::wgsl(self.dtype.wgsl_config()),
            O::layout,
            None,
            out_len,
            |b, e, p, ins, _u, out| dispatch_op::<O, _>(b, e, p, ins, out),
        )
        .await
    }

    pub async fn run_memcat<O: MemCatOp>(&self) -> Vec<u8> {
        // Input x is [T, C, H, W]; output [T, 2C, H, W].
        let s = &self.case.inputs[0].shape;
        let (tt, c, h, w) = (s[0] as u32, s[1] as u32, s[2] as u32, s[3] as u32);
        let out_len =
            (tt as u64) * 2 * (c as u64) * (h as u64) * (w as u64) * self.dtype.bytes_per_elem();
        // U = { t, c, h, w, has_prev }; has_prev = 0 (no carry frame, the untiled
        // zero-pad path). Padded to 32 bytes for the uniform min binding size.
        let mut u = [0u8; 32];
        u[0..16].copy_from_slice(&pack_u32x4(tt, c, h, w));
        let n_out = tt * 2 * c * h * w;
        self.run_op(
            &O::wgsl(self.dtype.wgsl_config()),
            O::layout,
            Some(&u),
            out_len,
            |b, e, p, ins, uf, out| {
                dispatch_memcat::<O, _>(
                    b,
                    e,
                    p,
                    &MemCatBufs {
                        x: &ins[0],
                        uniform: uf.as_ref().unwrap(),
                        out: &out,
                        // No carry: bind x itself (unread when has_prev = 0).
                        prev: &ins[0],
                    },
                    n_out,
                )
            },
        )
        .await
    }

    pub async fn run_matmul<O: MatmulOp>(&self, op: O) -> Vec<u8> {
        let a_shape = &self.case.inputs[0].shape;
        let b_shape = &self.case.inputs[1].shape;
        let (m, k, n) = (a_shape[0] as u32, a_shape[1] as u32, b_shape[1] as u32);
        let out_len = (m as u64) * (n as u64) * self.dtype.bytes_per_elem();
        let u = pack_u32x4(m, n, k, 0);
        let wgsl = op.wgsl(self.dtype.wgsl_config());
        self.run_op(
            &wgsl,
            O::layout,
            Some(&u),
            out_len,
            |b, e, p, ins, uf, out| {
                dispatch_matmul::<O, _>(
                    b,
                    e,
                    p,
                    &op,
                    &MatmulBufs {
                        a: &ins[0],
                        b: &ins[1],
                        dims: uf.as_ref().unwrap(),
                        out: &out,
                    },
                    m,
                    n,
                )
            },
        )
        .await
    }

    pub async fn run_rmsnorm<O: RmsNormOp>(&self) -> Vec<u8> {
        let eps = match self.case.op {
            OpSpec::Rmsnorm { eps } => eps,
            _ => panic!("run_rmsnorm called with non-rmsnorm OpSpec"),
        };
        let x_shape = &self.case.inputs[0].shape;
        let (n_rows, dim) = (x_shape[0] as u32, x_shape[1] as u32);
        let out_len = self.input_bytes(self.case.inputs[0].name).len() as u64;
        let u = pack_u32x4_eps(n_rows, dim, eps);
        self.run_op(
            O::wgsl(self.dtype.wgsl_config()),
            O::layout,
            Some(&u),
            out_len,
            |b, e, p, ins, uf, out| {
                dispatch_rmsnorm::<O, _>(
                    b,
                    e,
                    p,
                    &RmsNormBufs {
                        x: &ins[0],
                        w: &ins[1],
                        uniform: uf.as_ref().unwrap(),
                        out: &out,
                    },
                    n_rows,
                )
            },
        )
        .await
    }

    pub async fn run_layernorm<O: LayerNormOp>(&self) -> Vec<u8> {
        let eps = match self.case.op {
            OpSpec::Layernorm { eps } => eps,
            _ => panic!("run_layernorm called with non-layernorm OpSpec"),
        };
        let x_shape = &self.case.inputs[0].shape;
        let (n_rows, dim) = (x_shape[0] as u32, x_shape[1] as u32);
        let out_len = self.input_bytes(self.case.inputs[0].name).len() as u64;
        let u = pack_u32x4_eps(n_rows, dim, eps);
        self.run_op(
            O::wgsl(self.dtype.wgsl_config()),
            O::layout,
            Some(&u),
            out_len,
            |b, e, p, ins, uf, out| {
                dispatch_layernorm::<O, _>(
                    b,
                    e,
                    p,
                    &LayerNormBufs {
                        x: &ins[0],
                        uniform: uf.as_ref().unwrap(),
                        out: &out,
                    },
                    n_rows,
                )
            },
        )
        .await
    }

    pub async fn run_sdpa<O: SdpaOp>(&self) -> Vec<u8> {
        let scale = match self.case.op {
            OpSpec::Sdpa { scale } => scale,
            _ => panic!("run_sdpa called with non-sdpa OpSpec"),
        };
        let q_shape = &self.case.inputs[0].shape;
        let k_shape = &self.case.inputs[1].shape;
        let (bsz, s_q, h_q, d) = (
            q_shape[0] as u32,
            q_shape[1] as u32,
            q_shape[2] as u32,
            q_shape[3] as u32,
        );
        let (s_k, h_kv) = (k_shape[1] as u32, k_shape[2] as u32);
        let out_len = self.input_bytes(self.case.inputs[0].name).len() as u64;
        let mut u = [0u8; 32];
        u[0..4].copy_from_slice(&bsz.to_le_bytes());
        u[4..8].copy_from_slice(&h_q.to_le_bytes());
        u[8..12].copy_from_slice(&h_kv.to_le_bytes());
        u[12..16].copy_from_slice(&s_q.to_le_bytes());
        u[16..20].copy_from_slice(&s_k.to_le_bytes());
        u[20..24].copy_from_slice(&d.to_le_bytes());
        u[24..28].copy_from_slice(&scale.to_le_bytes());
        u[28..32].copy_from_slice(&1u32.to_le_bytes()); // has_mask: tests bind a real mask tensor
        self.run_op(
            O::wgsl(self.dtype.wgsl_config()),
            O::layout,
            Some(&u),
            out_len,
            |b, e, p, ins, uf, out| {
                dispatch_sdpa::<O, _>(
                    b,
                    e,
                    p,
                    &SdpaBufs {
                        q: &ins[0],
                        k: &ins[1],
                        v: &ins[2],
                        mask: &ins[3],
                        uniform: uf.as_ref().unwrap(),
                        out: &out,
                    },
                    bsz,
                    s_q,
                    h_q,
                )
            },
        )
        .await
    }

    pub async fn run_rope<O: RopeOp>(&self) -> Vec<u8> {
        let x_shape = &self.case.inputs[0].shape;
        let f_shape = &self.case.inputs[1].shape;
        let (rows, heads, dim) = (x_shape[0] as u32, x_shape[1] as u32, x_shape[2] as u32);
        debug_assert_eq!(dim % 2, 0, "rope dim must be even");
        debug_assert_eq!(f_shape[0] as u32, rows);
        debug_assert_eq!(f_shape[1] as u32, dim);
        let pairs = dim / 2;
        let out_len = self.input_bytes(self.case.inputs[0].name).len() as u64;
        let u = pack_u32x4(rows, heads, pairs, 0);
        self.run_op(
            O::wgsl(self.dtype.wgsl_config()),
            O::layout,
            Some(&u),
            out_len,
            |b, e, p, ins, uf, out| {
                dispatch_rope::<O, _>(
                    b,
                    e,
                    p,
                    &RopeBufs {
                        x: &ins[0],
                        freqs: &ins[1],
                        uniform: uf.as_ref().unwrap(),
                        out: &out,
                    },
                    rows,
                    heads,
                    pairs,
                )
            },
        )
        .await
    }

    pub async fn run_softmax<O: SoftmaxOp>(&self) -> Vec<u8> {
        let x_shape = &self.case.inputs[0].shape;
        let (n_rows, dim) = (x_shape[0] as u32, x_shape[1] as u32);
        let out_len = self.input_bytes(self.case.inputs[0].name).len() as u64;
        let u = pack_u32x4(n_rows, dim, 0, 0);
        self.run_op(
            O::wgsl(self.dtype.wgsl_config()),
            O::layout,
            Some(&u),
            out_len,
            |b, e, p, ins, uf, out| {
                dispatch_softmax::<O, _>(
                    b,
                    e,
                    p,
                    &SoftmaxBufs {
                        x: &ins[0],
                        uniform: uf.as_ref().unwrap(),
                        out: &out,
                    },
                    n_rows,
                )
            },
        )
        .await
    }

    pub async fn run_bcast_affine<O: BcastAffineOp>(&self) -> Vec<u8> {
        let bias = match self.case.op {
            OpSpec::BcastAffine { bias } => bias,
            _ => panic!("run_bcast_affine called with non-bcast_affine OpSpec"),
        };
        let (c, n_elems, out_len) = bcast_shapes(self);
        let mut u = [0u8; 16];
        u[0..4].copy_from_slice(&c.to_le_bytes());
        u[4..8].copy_from_slice(&bias.to_le_bytes());
        self.run_op(
            O::wgsl(self.dtype.wgsl_config()),
            O::layout,
            Some(&u),
            out_len,
            |b, e, p, ins, uf, out| {
                dispatch_bcast_affine::<O, _>(
                    b,
                    e,
                    p,
                    &BcastAffineBufs {
                        x: &ins[0],
                        s: &ins[1],
                        uniform: uf.as_ref().unwrap(),
                        out: &out,
                    },
                    n_elems,
                )
            },
        )
        .await
    }

    pub async fn run_bcast_add<O: BcastAddOp>(&self) -> Vec<u8> {
        let (c, n_elems, out_len) = bcast_shapes(self);
        let u = pack_u32x4(c, 0, 0, 0);
        self.run_op(
            O::wgsl(self.dtype.wgsl_config()),
            O::layout,
            Some(&u),
            out_len,
            |b, e, p, ins, uf, out| {
                dispatch_bcast_add::<O, _>(
                    b,
                    e,
                    p,
                    &BcastAddBufs {
                        x: &ins[0],
                        s: &ins[1],
                        uniform: uf.as_ref().unwrap(),
                        out: &out,
                    },
                    n_elems,
                )
            },
        )
        .await
    }

    pub async fn run_bcast_fma<O: BcastFmaOp>(&self) -> Vec<u8> {
        let (c, n_elems, out_len) = bcast_shapes(self);
        let u = pack_u32x4(c, 0, 0, 0);
        self.run_op(
            O::wgsl(self.dtype.wgsl_config()),
            O::layout,
            Some(&u),
            out_len,
            |b, e, p, ins, uf, out| {
                dispatch_bcast_fma::<O, _>(
                    b,
                    e,
                    p,
                    &BcastFmaBufs {
                        x: &ins[0],
                        s: &ins[1],
                        y: &ins[2],
                        uniform: uf.as_ref().unwrap(),
                        out: &out,
                    },
                    n_elems,
                )
            },
        )
        .await
    }

    pub async fn run_bcast_modulate<O: BcastModulateOp>(&self) -> Vec<u8> {
        let bias = match self.case.op {
            OpSpec::BcastModulate { bias } => bias,
            _ => panic!("run_bcast_modulate called with non-bcast_modulate OpSpec"),
        };
        let (c, n_elems, out_len) = bcast_shapes(self);
        let mut u = [0u8; 16];
        u[0..4].copy_from_slice(&c.to_le_bytes());
        u[4..8].copy_from_slice(&bias.to_le_bytes());
        self.run_op(
            O::wgsl(self.dtype.wgsl_config()),
            O::layout,
            Some(&u),
            out_len,
            |b, e, p, ins, uf, out| {
                dispatch_bcast_modulate::<O, _>(
                    b,
                    e,
                    p,
                    &BcastModulateBufs {
                        x: &ins[0],
                        s: &ins[1],
                        t: &ins[2],
                        uniform: uf.as_ref().unwrap(),
                        out: &out,
                    },
                    n_elems,
                )
            },
        )
        .await
    }

    pub async fn run_conv2d<O: Conv2dOp>(&self, op: O) -> Vec<u8> {
        let (kh, kw, pad_h, pad_w, stride_h, stride_w) = match self.case.op {
            OpSpec::Conv2d {
                kh,
                kw,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
            } => (kh, kw, pad_h, pad_w, stride_h, stride_w),
            _ => panic!("run_conv2d called with non-conv2d OpSpec"),
        };
        let x_shape = &self.case.inputs[0].shape;
        let w_shape = &self.case.inputs[1].shape;
        let (b, cin, h_in, w_in) = (
            x_shape[0] as u32,
            x_shape[1] as u32,
            x_shape[2] as u32,
            x_shape[3] as u32,
        );
        let cout = w_shape[0] as u32;
        assert_eq!(w_shape[1] as u32, cin, "weight cin mismatch");
        assert_eq!(w_shape[2] as u32, kh, "weight kh mismatch");
        assert_eq!(w_shape[3] as u32, kw, "weight kw mismatch");
        let h_out = (h_in + 2 * pad_h - kh) / stride_h + 1;
        let w_out = (w_in + 2 * pad_w - kw) / stride_w + 1;
        let n_out = b * cout * h_out * w_out;
        let out_len = (n_out as u64) * O::Dtype::SIZE as u64;
        let mut u = [0u8; 64];
        let fields: [u32; 13] = [
            b, cin, cout, h_in, w_in, h_out, w_out, kh, kw, pad_h, pad_w, stride_h, stride_w,
        ];
        for (i, v) in fields.iter().enumerate() {
            u[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        self.run_op(
            &op.wgsl(self.dtype.wgsl_config()),
            O::layout,
            Some(&u),
            out_len,
            |bk, e, p, ins, uf, out| {
                dispatch_conv2d::<O, _>(
                    bk,
                    e,
                    p,
                    &op,
                    &Conv2dBufs {
                        x: &ins[0],
                        w: &ins[1],
                        bias: &ins[2],
                        uniform: uf.as_ref().unwrap(),
                        out: &out,
                    },
                    cout,
                    h_out * w_out,
                    b,
                )
            },
        )
        .await
    }

    pub async fn run_rmsnorm3d<O: RmsNorm3dOp>(&self) -> Vec<u8> {
        match self.case.op {
            OpSpec::RmsNorm3d => {}
            _ => panic!("run_rmsnorm3d called with non-rmsnorm3d OpSpec"),
        };
        let s = &self.case.inputs[0].shape;
        // NCTHW: channels = dim 1, stride = product of the trailing dims.
        let (b, channels) = (s[0] as u32, s[1] as u32);
        let stride: u32 = s[2..].iter().map(|&d| d as u32).product();
        let n_pos = b * stride;
        let out_len = self.input_bytes(self.case.inputs[0].name).len() as u64;
        let u = pack_u32x4(n_pos, channels, stride, 0);
        self.run_op(
            &O::wgsl(self.dtype.wgsl_config()),
            O::layout,
            Some(&u),
            out_len,
            |bk, e, p, ins, uf, out| {
                dispatch_rmsnorm3d::<O, _>(
                    bk,
                    e,
                    p,
                    &RmsNorm3dBufs {
                        x: &ins[0],
                        w: &ins[1],
                        uniform: uf.as_ref().unwrap(),
                        out: &out,
                    },
                    n_pos,
                )
            },
        )
        .await
    }

    #[allow(clippy::many_single_char_names)]
    pub async fn run_conv3d<O: Conv3dOp>(&self, op: O) -> Vec<u8> {
        let (kt, kh, kw, pad_t, pad_h, pad_w, stride_t, stride_h, stride_w) = match self.case.op {
            OpSpec::Conv3d {
                kt,
                kh,
                kw,
                pad_t,
                pad_h,
                pad_w,
                stride_t,
                stride_h,
                stride_w,
            } => (
                kt, kh, kw, pad_t, pad_h, pad_w, stride_t, stride_h, stride_w,
            ),
            _ => panic!("run_conv3d called with non-conv3d OpSpec"),
        };
        let x_shape = &self.case.inputs[0].shape;
        let w_shape = &self.case.inputs[1].shape;
        let (b, cin, t_in, h_in, w_in) = (
            x_shape[0] as u32,
            x_shape[1] as u32,
            x_shape[2] as u32,
            x_shape[3] as u32,
            x_shape[4] as u32,
        );
        let cout = w_shape[0] as u32;
        assert_eq!(w_shape[1] as u32, cin, "weight cin mismatch");
        assert_eq!(w_shape[2] as u32, kt, "weight kt mismatch");
        assert_eq!(w_shape[3] as u32, kh, "weight kh mismatch");
        assert_eq!(w_shape[4] as u32, kw, "weight kw mismatch");
        // `pad_t` is the single front-pad (causal convention); H/W are
        // symmetric. The python ref applies the same asymmetric time pad.
        let t_out = (t_in + pad_t - kt) / stride_t + 1;
        let h_out = (h_in + 2 * pad_h - kh) / stride_h + 1;
        let w_out = (w_in + 2 * pad_w - kw) / stride_w + 1;
        let n_out = b * cout * t_out * h_out * w_out;
        let out_len = (n_out as u64) * O::Dtype::SIZE as u64;
        let mut u = [0u8; 80];
        let fields: [u32; 18] = [
            b, cin, cout, t_in, h_in, w_in, t_out, h_out, w_out, kt, kh, kw, pad_t, pad_h, pad_w,
            stride_t, stride_h, stride_w,
        ];
        for (i, v) in fields.iter().enumerate() {
            u[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        self.run_op(
            &op.wgsl(self.dtype.wgsl_config()),
            O::layout,
            Some(&u),
            out_len,
            |bk, e, p, ins, uf, out| {
                dispatch_conv3d::<O, _>(
                    bk,
                    e,
                    p,
                    &op,
                    &Conv3dBufs {
                        x: &ins[0],
                        w: &ins[1],
                        bias: &ins[2],
                        uniform: uf.as_ref().unwrap(),
                        out: &out,
                    },
                    cout,
                    t_out * h_out * w_out,
                    b,
                )
            },
        )
        .await
    }

    pub async fn run_transpose12<O: Transpose12Op>(&self) -> Vec<u8> {
        let s = &self.case.inputs[0].shape;
        let (d0, d1, d2, d3) = (s[0] as u32, s[1] as u32, s[2] as u32, s[3] as u32);
        let total = d0 * d1 * d2 * d3;
        let out_len = self.input_bytes(self.case.inputs[0].name).len() as u64;
        let u = pack_u32x4(d0, d1, d2, d3);
        self.run_op(
            O::wgsl(self.dtype.wgsl_config()),
            O::layout,
            Some(&u),
            out_len,
            |b, e, p, ins, uf, out| {
                dispatch_transpose12::<O, _>(
                    b,
                    e,
                    p,
                    &Transpose12Bufs {
                        input: &ins[0],
                        uniform: uf.as_ref().unwrap(),
                        out: &out,
                    },
                    total,
                )
            },
        )
        .await
    }
}

fn bcast_shapes(ctx: &OpTestContext<'_>) -> (u32, u32, u64) {
    let x_shape = &ctx.case.inputs[0].shape;
    let s_shape = &ctx.case.inputs[1].shape;
    let c = *s_shape.last().expect("s shape") as u32;
    let n_elems: u32 = x_shape.iter().map(|&d| d as u32).product();
    let out_len = ctx.input_bytes(ctx.case.inputs[0].name).len() as u64;
    (c, n_elems, out_len)
}

fn pack_u32x4(a: u32, b: u32, c: u32, d: u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&a.to_le_bytes());
    out[4..8].copy_from_slice(&b.to_le_bytes());
    out[8..12].copy_from_slice(&c.to_le_bytes());
    out[12..16].copy_from_slice(&d.to_le_bytes());
    out
}

fn pack_u32x4_eps(a: u32, b: u32, eps: f32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&a.to_le_bytes());
    out[4..8].copy_from_slice(&b.to_le_bytes());
    out[8..12].copy_from_slice(&eps.to_le_bytes());
    out
}

// ---------- registry / diff ----------

pub fn registry() -> Vec<Box<dyn OpTest>> {
    vec![
        Box::new(AddF32),
        Box::new(MulF32),
        Box::new(SiluF32),
        Box::new(ReluF32),
        Box::new(crate::ops::MemCatF32),
        Box::new(SiluMulF32),
        Box::new(GeluMulF32),
        Box::new(TanhF32),
        // tn=2 (not the DEFAULT tn=1) so the bf16-packed variant can pack
        // two output columns per thread. Otherwise unchanged geometry.
        Box::new(MatMulF32::new(crate::ops::MatMulConfig {
            tn: 2,
            ..crate::ops::MatMulConfig::DEFAULT
        })),
        Box::new(RmsNormF32),
        Box::new(LayerNormF32),
        Box::new(SoftmaxF32),
        Box::new(RopeF32),
        Box::new(RopeF32HalfRot),
        Box::new(SdpaF32),
        Box::new(SdpaF32LargeD),
        Box::new(Transpose12F32),
        Box::new(BcastAffineF32),
        Box::new(BcastFmaF32),
        Box::new(BcastModulateF32),
        Box::new(BcastAddF32),
        Box::new(BcastMulF32),
        Box::new(Conv2dF32::default_op()),
        Box::new(Conv3dF32::default_op()),
        Box::new(RmsNorm3dF32),
    ]
}

pub fn tol(d: Dtype) -> f32 {
    match d {
        Dtype::Fp32 => 1e-5,
        // bf16-writes / bf16-packed: pytorch ref RNE-rounds to bf16; our
        // shaders do the same on every activation store. Both sides land on
        // the same bf16 bit pattern, so diff is bit-exact in practice. Keep
        // at fp32-level tol so kernel reorderings (tiled matmul changing
        // accumulator order, etc.) get caught instead of silently absorbed.
        Dtype::Bf16Writes => 1e-5,
        Dtype::Bf16Packed => 1e-5,
    }
}

pub fn diff_max_abs(d: Dtype, got: &[u8], expected: &[u8]) -> f32 {
    assert_eq!(got.len(), expected.len(), "byte length mismatch");
    match d {
        Dtype::Fp32 | Dtype::Bf16Writes => got
            .chunks_exact(4)
            .zip(expected.chunks_exact(4))
            .map(|(g, e)| {
                let g = f32::from_le_bytes([g[0], g[1], g[2], g[3]]);
                let e = f32::from_le_bytes([e[0], e[1], e[2], e[3]]);
                (g - e).abs()
            })
            .fold(0.0_f32, f32::max),
        Dtype::Bf16Packed => got
            .chunks_exact(2)
            .zip(expected.chunks_exact(2))
            .map(|(g, e)| {
                // bf16 lives in the upper 16 bits of an f32 bitcast.
                let g = f32::from_bits((u16::from_le_bytes([g[0], g[1]]) as u32) << 16);
                let e = f32::from_bits((u16::from_le_bytes([e[0], e[1]]) as u32) << 16);
                (g - e).abs()
            })
            .fold(0.0_f32, f32::max),
    }
}
