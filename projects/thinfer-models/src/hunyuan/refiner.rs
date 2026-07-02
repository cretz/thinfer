//! HunyuanVideo 1.5 SingleTokenRefiner (`txt_in` of the DiT): refines the
//! Qwen2.5-VL text hidden states `[seq, 3584]` into joint DiT tokens `[seq,
//! 2048]`, conditioned on the diffusion timestep. 2 `IndividualTokenRefinerBlock`s
//! (affine LayerNorm, fused qkv + full bidirectional self-attn, qk-norm OFF, silu
//! MLP, 2-gate adaLN-Zero). Source: `modules/token_refiner.py`.
//!
//! `c = t_embedder(t) + c_embedder(mean(x))`; `x = input_embedder(x)`; then per
//! block: `x += gate_msa * proj(attn(LN1(x)))`, `x += gate_mlp * mlp(LN2(x))`,
//! where `gate_{msa,mlp} = adaLN(c).chunk(2)`. Timestep embed = cos-first sinusoid
//! (freq_dim 256, max_period 1e4, NO 1000 scale) -> 2-layer silu MLP. Bf16 weights
//! (narrowed from the fp16 DiT checkpoint), f32 acts for parity.

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError};
use thinfer_core::ops::{
    ActDtype, AddF32, BcastAddF32, BcastAddOp, BcastFmaF32, BcastFmaOp, BcastMulF32, LayerNormF32,
    LayerNormOp, MatMulConfig, MatMulF32, MatmulOp, Op, QkvSplitF32, QkvSplitOp, SdpaF32, SdpaOp,
    SiluF32, WeightDtype, WgslConfig,
};
use thinfer_core::residency::{GpuView, ResidencyError, WeightHandle, WeightResidency};
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace, WsBuf};

use crate::common::loader::{LoadError, register_linear, register_passthrough};
use crate::common::seq::{act_readback_to_f32, act_upload_bytes};

pub const IN_CHANNELS: usize = 3584;
pub const HIDDEN: usize = 2048;
pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = HIDDEN / HEADS; // 128
pub const DEPTH: usize = 2;
pub const MLP_HIDDEN: usize = HIDDEN * 4; // 8192
const FREQ_DIM: usize = 256;
const MAX_PERIOD: f32 = 10_000.0;
const LN_EPS: f32 = 1e-6;

// ============================================================================
// Weights
// ============================================================================

/// `weight [out,in]` (registered transposed to `[in,out]`) + `bias [out]`.
#[derive(Clone)]
struct LinearW {
    weight: WeightId,
    bias: WeightId,
}

fn linear_w(prefix: &str) -> LinearW {
    LinearW {
        weight: WeightId(format!("{prefix}.weight")),
        bias: WeightId(format!("{prefix}.bias")),
    }
}

/// Affine LayerNorm params `[dim]`.
#[derive(Clone)]
struct LnW {
    weight: WeightId,
    bias: WeightId,
}

fn ln_w(prefix: &str) -> LnW {
    LnW {
        weight: WeightId(format!("{prefix}.weight")),
        bias: WeightId(format!("{prefix}.bias")),
    }
}

struct BlockW {
    norm1: LnW,
    qkv: LinearW,
    proj: LinearW,
    norm2: LnW,
    fc1: LinearW,
    fc2: LinearW,
    adaln: LinearW,
}

struct RefinerW {
    input_embedder: LinearW,
    t_mlp0: LinearW,
    t_mlp2: LinearW,
    c_linear1: LinearW,
    c_linear2: LinearW,
    blocks: Vec<BlockW>,
}

impl RefinerW {
    fn new() -> Self {
        let blocks = (0..DEPTH)
            .map(|i| {
                let p = format!("txt_in.individual_token_refiner.blocks.{i}");
                BlockW {
                    norm1: ln_w(&format!("{p}.norm1")),
                    qkv: linear_w(&format!("{p}.self_attn_qkv")),
                    proj: linear_w(&format!("{p}.self_attn_proj")),
                    norm2: ln_w(&format!("{p}.norm2")),
                    fc1: linear_w(&format!("{p}.mlp.fc1")),
                    fc2: linear_w(&format!("{p}.mlp.fc2")),
                    adaln: linear_w(&format!("{p}.adaLN_modulation.1")),
                }
            })
            .collect();
        Self {
            input_embedder: linear_w("txt_in.input_embedder"),
            t_mlp0: linear_w("txt_in.t_embedder.mlp.0"),
            t_mlp2: linear_w("txt_in.t_embedder.mlp.2"),
            c_linear1: linear_w("txt_in.c_embedder.linear_1"),
            c_linear2: linear_w("txt_in.c_embedder.linear_2"),
            blocks,
        }
    }
}

#[derive(Clone, Copy)]
struct LinearH {
    weight: WeightHandle,
    bias: WeightHandle,
}

#[derive(Clone, Copy)]
struct LnH {
    weight: WeightHandle,
    bias: WeightHandle,
}

struct BlockH {
    norm1: LnH,
    qkv: LinearH,
    proj: LinearH,
    norm2: LnH,
    fc1: LinearH,
    fc2: LinearH,
    adaln: LinearH,
}

struct RefinerH {
    input_embedder: LinearH,
    t_mlp0: LinearH,
    t_mlp2: LinearH,
    c_linear1: LinearH,
    c_linear2: LinearH,
    blocks: Vec<BlockH>,
}

fn reg_linear<S: WeightSource>(
    res: &WeightResidency<S>,
    w: &LinearW,
) -> Result<LinearH, LoadError> {
    Ok(LinearH {
        weight: register_linear(res, &w.weight)?,
        bias: register_passthrough(res, &w.bias)?,
    })
}

fn reg_ln<S: WeightSource>(res: &WeightResidency<S>, w: &LnW) -> Result<LnH, LoadError> {
    Ok(LnH {
        weight: register_passthrough(res, &w.weight)?,
        bias: register_passthrough(res, &w.bias)?,
    })
}

impl RefinerH {
    fn register<S: WeightSource>(
        res: &WeightResidency<S>,
        w: &RefinerW,
    ) -> Result<Self, LoadError> {
        let mut blocks = Vec::with_capacity(w.blocks.len());
        for b in &w.blocks {
            blocks.push(BlockH {
                norm1: reg_ln(res, &b.norm1)?,
                qkv: reg_linear(res, &b.qkv)?,
                proj: reg_linear(res, &b.proj)?,
                norm2: reg_ln(res, &b.norm2)?,
                fc1: reg_linear(res, &b.fc1)?,
                fc2: reg_linear(res, &b.fc2)?,
                adaln: reg_linear(res, &b.adaln)?,
            });
        }
        Ok(Self {
            input_embedder: reg_linear(res, &w.input_embedder)?,
            t_mlp0: reg_linear(res, &w.t_mlp0)?,
            t_mlp2: reg_linear(res, &w.t_mlp2)?,
            c_linear1: reg_linear(res, &w.c_linear1)?,
            c_linear2: reg_linear(res, &w.c_linear2)?,
            blocks,
        })
    }
}

#[derive(Clone, Copy)]
struct LinearBufs {
    weight: BufRef,
    bias: BufRef,
}

#[derive(Clone, Copy)]
struct LnBufs {
    weight: BufRef,
    bias: BufRef,
}

struct BlockBufs {
    norm1: LnBufs,
    qkv: LinearBufs,
    proj: LinearBufs,
    norm2: LnBufs,
    fc1: LinearBufs,
    fc2: LinearBufs,
    adaln: LinearBufs,
}

struct RefinerBufs {
    input_embedder: LinearBufs,
    t_mlp0: LinearBufs,
    t_mlp2: LinearBufs,
    c_linear1: LinearBufs,
    c_linear2: LinearBufs,
    blocks: Vec<BlockBufs>,
}

// ============================================================================
// Pipelines
// ============================================================================

pub struct HunyuanRefinerPipelines {
    act: ActDtype,
    matmul: MatMulF32,
    matmul_pl: thinfer_core::backend::WgpuPipeline,
    layernorm: thinfer_core::backend::WgpuPipeline,
    sdpa: thinfer_core::backend::WgpuPipeline,
    silu: thinfer_core::backend::WgpuPipeline,
    add: thinfer_core::backend::WgpuPipeline,
    bcast_mul: thinfer_core::backend::WgpuPipeline,
    bcast_add: thinfer_core::backend::WgpuPipeline,
    bcast_fma: thinfer_core::backend::WgpuPipeline,
    qkv_split: thinfer_core::backend::WgpuPipeline,
}

impl HunyuanRefinerPipelines {
    fn act_size(&self) -> u64 {
        match self.act {
            ActDtype::F32 => 4,
            ActDtype::F16 => 2,
            other => unreachable!("refiner acts f32/f16, got {other:?}"),
        }
    }

    pub async fn compile_with(backend: &WgpuBackend, act: ActDtype) -> Result<Self, WgpuError> {
        let cfg = &WgslConfig {
            bf16_quant_writes: false,
            act_dtype: act,
            weight_dtype: WeightDtype::Bf16,
        };
        let matmul = MatMulF32::new(MatMulConfig::DEFAULT);
        Ok(Self {
            act,
            matmul_pl: backend
                .create_pipeline(
                    "hunyuan_refiner_matmul",
                    &matmul.wgsl(cfg),
                    "main",
                    <MatMulF32 as MatmulOp>::layout(),
                )
                .await?,
            matmul,
            layernorm: backend
                .create_pipeline(
                    "hunyuan_refiner_layernorm",
                    <LayerNormF32 as LayerNormOp>::wgsl(cfg),
                    "main",
                    <LayerNormF32 as LayerNormOp>::layout(),
                )
                .await?,
            sdpa: backend
                .create_pipeline(
                    "hunyuan_refiner_sdpa",
                    <SdpaF32 as SdpaOp>::wgsl(cfg),
                    "main",
                    <SdpaF32 as SdpaOp>::layout(),
                )
                .await?,
            silu: backend
                .create_pipeline(
                    "hunyuan_refiner_silu",
                    SiluF32::wgsl(cfg),
                    "main",
                    SiluF32::layout(),
                )
                .await?,
            add: backend
                .create_pipeline(
                    "hunyuan_refiner_add",
                    AddF32::wgsl(cfg),
                    "main",
                    AddF32::layout(),
                )
                .await?,
            bcast_mul: backend
                .create_pipeline(
                    "hunyuan_refiner_bcast_mul",
                    <BcastMulF32 as BcastAddOp>::wgsl(cfg),
                    "main",
                    <BcastMulF32 as BcastAddOp>::layout(),
                )
                .await?,
            bcast_add: backend
                .create_pipeline(
                    "hunyuan_refiner_bcast_add",
                    <BcastAddF32 as BcastAddOp>::wgsl(cfg),
                    "main",
                    <BcastAddF32 as BcastAddOp>::layout(),
                )
                .await?,
            bcast_fma: backend
                .create_pipeline(
                    "hunyuan_refiner_bcast_fma",
                    <BcastFmaF32 as BcastFmaOp>::wgsl(cfg),
                    "main",
                    <BcastFmaF32 as BcastFmaOp>::layout(),
                )
                .await?,
            qkv_split: backend
                .create_pipeline(
                    "hunyuan_refiner_qkv_split",
                    <QkvSplitF32 as QkvSplitOp>::wgsl(cfg),
                    "main",
                    <QkvSplitF32 as QkvSplitOp>::layout(),
                )
                .await?,
        })
    }
}

// ============================================================================
// Op wrappers
// ============================================================================

/// `x[rows,k] @ wᵀ + bias -> [rows,n]` (bf16 weight already transposed to [k,n]).
fn linear<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &HunyuanRefinerPipelines,
    x: BatchBuf<'w>,
    w: &LinearBufs,
    rows: u32,
    n: u32,
    k: u32,
) -> Result<BatchBuf<'w>, WgpuError> {
    let pre = scope.alloc((rows * n) as u64 * pl.act_size())?;
    let dims = scope.u32x4_uniform(rows, n, k, 0)?;
    let wv = scope.import_copy(w.weight);
    scope.matmul(&pl.matmul_pl, &pl.matmul, x, wv, dims, pre, rows, n)?;
    let out = scope.alloc((rows * n) as u64 * pl.act_size())?;
    let u = scope.u32x4_uniform(n, 0, 0, 0)?;
    let bv = scope.import_copy(w.bias);
    scope.bcast_add::<BcastAddF32>(&pl.bcast_add, pre, bv, u, out, rows * n)?;
    Ok(out)
}

/// Affine LayerNorm: `LN(x) * weight[c] + bias[c]` over `dim`.
fn affine_ln<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &HunyuanRefinerPipelines,
    x: BatchBuf<'w>,
    w: &LnBufs,
    rows: u32,
    dim: u32,
) -> Result<BatchBuf<'w>, WgpuError> {
    let normed = scope.alloc((rows * dim) as u64 * pl.act_size())?;
    let lu = scope.u32x4_uniform(rows, dim, LN_EPS.to_bits(), 0)?;
    scope.layernorm::<LayerNormF32>(&pl.layernorm, x, lu, normed, rows)?;
    let scaled = scope.alloc((rows * dim) as u64 * pl.act_size())?;
    let au = scope.u32x4_uniform(dim, 0, 0, 0)?;
    let wv = scope.import_copy(w.weight);
    // BcastMulF32 (bf16-weight-aware) = `normed * weight[c]`; bcast_affine reads
    // its scale as f32 and would misread the bf16 LN weight.
    scope.bcast_add::<BcastMulF32>(&pl.bcast_mul, normed, wv, au, scaled, rows * dim)?;
    let out = scope.alloc((rows * dim) as u64 * pl.act_size())?;
    let bu = scope.u32x4_uniform(dim, 0, 0, 0)?;
    let bv = scope.import_copy(w.bias);
    scope.bcast_add::<BcastAddF32>(&pl.bcast_add, scaled, bv, bu, out, rows * dim)?;
    Ok(out)
}

fn silu<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &HunyuanRefinerPipelines,
    x: BatchBuf<'w>,
    n: u32,
) -> Result<BatchBuf<'w>, WgpuError> {
    let out = scope.alloc(n as u64 * pl.act_size())?;
    scope.dispatch_op::<SiluF32>(&pl.silu, &[x], out)?;
    Ok(out)
}

/// `x + gate[c] * y` over `dim` (gate `[1,dim]` broadcast over rows).
fn gate_residual<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &HunyuanRefinerPipelines,
    x: BatchBuf<'w>,
    gate: BatchBuf<'w>,
    y: BatchBuf<'w>,
    rows: u32,
    dim: u32,
) -> Result<BatchBuf<'w>, WgpuError> {
    let out = scope.alloc((rows * dim) as u64 * pl.act_size())?;
    let u = scope.u32x4_uniform(dim, 0, 0, 0)?;
    scope.bcast_fma::<BcastFmaF32>(&pl.bcast_fma, x, gate, y, u, out, rows * dim)?;
    Ok(out)
}

fn sdpa_uniform_bytes(s: u32, scale: f32) -> [u8; 32] {
    let f = [1u32, HEADS as u32, HEADS as u32, s, s, HEAD_DIM as u32];
    let mut bytes = [0u8; 32];
    for (i, v) in f.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    bytes[24..28].copy_from_slice(&scale.to_le_bytes());
    bytes // has_mask = 0
}

/// Full bidirectional multi-head attention over `[seq, HIDDEN]` q/k/v (no mask).
fn attention<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &HunyuanRefinerPipelines,
    q: BatchBuf<'w>,
    k: BatchBuf<'w>,
    v: BatchBuf<'w>,
    seq: u32,
) -> Result<BatchBuf<'w>, WgpuError> {
    let out = scope.alloc((seq * HIDDEN as u32) as u64 * pl.act_size())?;
    let mask = scope.write_uniform(&0f32.to_le_bytes())?;
    let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    let u = scope.write_uniform(&sdpa_uniform_bytes(seq, scale))?;
    scope.sdpa::<SdpaF32>(&pl.sdpa, q, k, v, mask, u, out, 1, seq, HEADS as u32)?;
    Ok(out)
}

// ============================================================================
// Refiner
// ============================================================================

pub struct HunyuanRefiner {
    pub pipelines: HunyuanRefinerPipelines,
    handles: RefinerH,
}

#[derive(Debug)]
pub enum HunyuanRefinerError<SE: core::fmt::Debug> {
    Wgpu(WgpuError),
    Load(LoadError),
    Residency(ResidencyError<SE, WgpuError>),
}
impl<SE: core::fmt::Debug> From<WgpuError> for HunyuanRefinerError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}
impl<SE: core::fmt::Debug> From<LoadError> for HunyuanRefinerError<SE> {
    fn from(e: LoadError) -> Self {
        Self::Load(e)
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for HunyuanRefinerError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

/// Per-stage taps (NCTHW-free, row-major `[*, dim]`).
#[derive(Default)]
pub struct HunyuanRefinerTaps<'a> {
    pub t_emb: Option<&'a mut Vec<f32>>,
    pub c_emb: Option<&'a mut Vec<f32>>,
    pub cond: Option<&'a mut Vec<f32>>,
    pub embedded: Option<&'a mut Vec<f32>>,
    pub block0: Option<&'a mut Vec<f32>>,
}

/// Cos-first sinusoid timestep embedding `[256]`: `a_j = t * exp(-ln(1e4)*j/128)`,
/// `[cos(a) ++ sin(a)]` (NO 1000 scale, unlike the diffusers `Timesteps` path).
pub(crate) fn timestep_sinusoid(t: f32) -> Vec<f32> {
    let half = FREQ_DIM / 2;
    let mut out = vec![0.0f32; FREQ_DIM];
    for j in 0..half {
        let freq = (-(MAX_PERIOD.ln()) * j as f32 / half as f32).exp();
        let a = t * freq;
        out[j] = a.cos();
        out[half + j] = a.sin();
    }
    out
}

impl HunyuanRefiner {
    pub fn new<S: WeightSource>(
        pipelines: HunyuanRefinerPipelines,
        residency: &WeightResidency<S>,
    ) -> Result<Self, LoadError> {
        let handles = RefinerH::register(residency, &RefinerW::new())?;
        Ok(Self { pipelines, handles })
    }

    /// Refine `text [seq, 3584]` (host f32) at timestep `t` into `[seq, 2048]`.
    pub async fn refine<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        text: &[f32],
        seq: usize,
        t: f32,
        mut taps: Option<&mut HunyuanRefinerTaps<'_>>,
    ) -> Result<Vec<f32>, HunyuanRefinerError<S::Error>> {
        assert_eq!(text.len(), seq * IN_CHANNELS, "text size");
        let act = self.pipelines.act;
        let asz = self.pipelines.act_size();

        // Host-side: mean over tokens (for c_embedder) + the timestep sinusoid.
        let mut c_mean = vec![0.0f32; IN_CHANNELS];
        for s in 0..seq {
            for c in 0..IN_CHANNELS {
                c_mean[c] += text[s * IN_CHANNELS + c];
            }
        }
        for v in &mut c_mean {
            *v /= seq as f32;
        }
        let t_sin = timestep_sinusoid(t);

        let upload = |slice: &[f32]| -> Result<WsBuf<WgpuBackend>, WgpuError> {
            let bytes = act_upload_bytes(act, slice);
            let buf = workspace.alloc(bytes.len() as u64)?;
            backend.write_buffer(buf.id(), 0, &bytes)?;
            Ok(buf)
        };
        let text_in = upload(text)?;
        let cmean_in = upload(&c_mean)?;
        let tsin_in = upload(&t_sin)?;

        let mut pins: Vec<GpuView> = Vec::new();
        let bufs = self.acquire(residency, backend, &mut pins).await?;

        let want = |f: fn(&HunyuanRefinerTaps) -> bool| taps.as_ref().is_some_and(|t| f(t));
        let mut persists: Vec<(WsBuf<WgpuBackend>, usize)> = Vec::new(); // (buf, n_elems) parity taps
        let out_persist;
        {
            let scope = workspace.batch();
            let pl = &self.pipelines;
            let seqr = seq as u32;
            let (hidden, inc, mlp) = (HIDDEN as u32, IN_CHANNELS as u32, MLP_HIDDEN as u32);

            // c = t_embedder(t) + c_embedder(mean(x)).
            let tb = scope.import_copy(tsin_in.as_buf_ref());
            let t0 = linear(&scope, pl, tb, &bufs.t_mlp0, 1, hidden, FREQ_DIM as u32)?;
            let t0a = silu(&scope, pl, t0, hidden)?;
            let t_emb = linear(&scope, pl, t0a, &bufs.t_mlp2, 1, hidden, hidden)?;
            let cb = scope.import_copy(cmean_in.as_buf_ref());
            let c0 = linear(&scope, pl, cb, &bufs.c_linear1, 1, hidden, inc)?;
            let c0a = silu(&scope, pl, c0, hidden)?;
            let c_emb = linear(&scope, pl, c0a, &bufs.c_linear2, 1, hidden, hidden)?;
            let cond = scope.alloc(hidden as u64 * asz)?;
            scope.dispatch_op::<AddF32>(&pl.add, &[t_emb, c_emb], cond)?;

            if want(|t| t.t_emb.is_some()) {
                persists.push((persist(&scope, workspace, t_emb, HIDDEN, asz)?, HIDDEN));
            }
            if want(|t| t.c_emb.is_some()) {
                persists.push((persist(&scope, workspace, c_emb, HIDDEN, asz)?, HIDDEN));
            }
            if want(|t| t.cond.is_some()) {
                persists.push((persist(&scope, workspace, cond, HIDDEN, asz)?, HIDDEN));
            }

            // x = input_embedder(text).
            let xin = scope.import_copy(text_in.as_buf_ref());
            let mut x = linear(&scope, pl, xin, &bufs.input_embedder, seqr, hidden, inc)?;
            if want(|t| t.embedded.is_some()) {
                persists.push((
                    persist(&scope, workspace, x, seq * HIDDEN, asz)?,
                    seq * HIDDEN,
                ));
            }

            for (bi, b) in bufs.blocks.iter().enumerate() {
                // gates = adaLN(silu(c)).chunk(2): [1, 2*hidden].
                let cs = silu(&scope, pl, cond, hidden)?;
                let gates = linear(&scope, pl, cs, &b.adaln, 1, 2 * hidden, hidden)?;
                let gate_msa = scope.alloc(hidden as u64 * asz)?;
                scope.copy_buffer_to_buffer(gates, 0, gate_msa, 0, hidden as u64 * asz)?;
                let gate_mlp = scope.alloc(hidden as u64 * asz)?;
                scope.copy_buffer_to_buffer(
                    gates,
                    hidden as u64 * asz,
                    gate_mlp,
                    0,
                    hidden as u64 * asz,
                )?;

                // attn sublayer.
                let n1 = affine_ln(&scope, pl, x, &b.norm1, seqr, hidden)?;
                let qkv = linear(&scope, pl, n1, &b.qkv, seqr, 3 * hidden, hidden)?;
                let q = scope.alloc((seqr * hidden) as u64 * asz)?;
                let k = scope.alloc((seqr * hidden) as u64 * asz)?;
                let v = scope.alloc((seqr * hidden) as u64 * asz)?;
                let qu = scope.write_uniform(&{
                    let mut by = [0u8; 16];
                    by[0..4].copy_from_slice(&seqr.to_le_bytes());
                    by[4..8].copy_from_slice(&hidden.to_le_bytes());
                    by
                })?;
                scope.qkv_split::<QkvSplitF32>(&pl.qkv_split, qkv, q, k, v, qu, seqr * hidden)?;
                let attn = attention(&scope, pl, q, k, v, seqr)?;
                let proj = linear(&scope, pl, attn, &b.proj, seqr, hidden, hidden)?;
                x = gate_residual(&scope, pl, x, gate_msa, proj, seqr, hidden)?;

                // mlp sublayer.
                let n2 = affine_ln(&scope, pl, x, &b.norm2, seqr, hidden)?;
                let h1 = linear(&scope, pl, n2, &b.fc1, seqr, mlp, hidden)?;
                let h1a = silu(&scope, pl, h1, seqr * mlp)?;
                let h2 = linear(&scope, pl, h1a, &b.fc2, seqr, hidden, mlp)?;
                x = gate_residual(&scope, pl, x, gate_mlp, h2, seqr, hidden)?;

                if bi == 0 && want(|t| t.block0.is_some()) {
                    persists.push((
                        persist(&scope, workspace, x, seq * HIDDEN, asz)?,
                        seq * HIDDEN,
                    ));
                }
            }

            let ws = workspace.alloc((seq * HIDDEN) as u64 * asz)?;
            let dst = scope.import_copy(ws.as_buf_ref());
            scope.copy_buffer_to_buffer(x, 0, dst, 0, (seq * HIDDEN) as u64 * asz)?;
            out_persist = ws;
            scope.submit_void().await?;
        }

        // Tap readback (in push order: t_emb, c_emb, cond, embedded, block0 - each
        // only present if its tap was requested).
        if let Some(t) = taps.as_mut() {
            let mut it = persists.iter();
            for sink in [
                t.t_emb.as_deref_mut(),
                t.c_emb.as_deref_mut(),
                t.cond.as_deref_mut(),
                t.embedded.as_deref_mut(),
                t.block0.as_deref_mut(),
            ]
            .into_iter()
            .flatten()
            {
                let (ws, n) = it.next().expect("tap count matches");
                *sink = read_acts(backend, &ws.as_buf_ref(), *n, act).await?;
            }
        }
        read_acts(backend, &out_persist.as_buf_ref(), seq * HIDDEN, act)
            .await
            .map_err(Into::into)
    }

    async fn acquire<'r, S: WeightSource>(
        &self,
        residency: &'r WeightResidency<S>,
        backend: &WgpuBackend,
        pins: &mut Vec<GpuView<'r>>,
    ) -> Result<RefinerBufs, ResidencyError<S::Error, WgpuError>> {
        let h = &self.handles;
        let mut blocks = Vec::with_capacity(h.blocks.len());
        for b in &h.blocks {
            blocks.push(BlockBufs {
                norm1: acq_ln(residency, backend, b.norm1, pins).await?,
                qkv: acq_lin(residency, backend, b.qkv, pins).await?,
                proj: acq_lin(residency, backend, b.proj, pins).await?,
                norm2: acq_ln(residency, backend, b.norm2, pins).await?,
                fc1: acq_lin(residency, backend, b.fc1, pins).await?,
                fc2: acq_lin(residency, backend, b.fc2, pins).await?,
                adaln: acq_lin(residency, backend, b.adaln, pins).await?,
            });
        }
        Ok(RefinerBufs {
            input_embedder: acq_lin(residency, backend, h.input_embedder, pins).await?,
            t_mlp0: acq_lin(residency, backend, h.t_mlp0, pins).await?,
            t_mlp2: acq_lin(residency, backend, h.t_mlp2, pins).await?,
            c_linear1: acq_lin(residency, backend, h.c_linear1, pins).await?,
            c_linear2: acq_lin(residency, backend, h.c_linear2, pins).await?,
            blocks,
        })
    }
}

async fn acq_lin<'r, S: WeightSource>(
    residency: &'r WeightResidency<S>,
    backend: &WgpuBackend,
    h: LinearH,
    pins: &mut Vec<GpuView<'r>>,
) -> Result<LinearBufs, ResidencyError<S::Error, WgpuError>> {
    let wv = residency.acquire(h.weight, backend).await?;
    let bv = residency.acquire(h.bias, backend).await?;
    let bufs = LinearBufs {
        weight: wv.buf(),
        bias: bv.buf(),
    };
    pins.push(wv);
    pins.push(bv);
    Ok(bufs)
}

async fn acq_ln<'r, S: WeightSource>(
    residency: &'r WeightResidency<S>,
    backend: &WgpuBackend,
    h: LnH,
    pins: &mut Vec<GpuView<'r>>,
) -> Result<LnBufs, ResidencyError<S::Error, WgpuError>> {
    let wv = residency.acquire(h.weight, backend).await?;
    let bv = residency.acquire(h.bias, backend).await?;
    let bufs = LnBufs {
        weight: wv.buf(),
        bias: bv.buf(),
    };
    pins.push(wv);
    pins.push(bv);
    Ok(bufs)
}

/// Copy a scope-local activation into a workspace buffer that outlives the submit.
fn persist<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    workspace: &Workspace<WgpuBackend>,
    buf: BatchBuf<'w>,
    n: usize,
    asz: u64,
) -> Result<WsBuf<WgpuBackend>, WgpuError> {
    let ws = workspace.alloc(n as u64 * asz)?;
    let dst = scope.import_copy(ws.as_buf_ref());
    scope.copy_buffer_to_buffer(buf, 0, dst, 0, n as u64 * asz)?;
    Ok(ws)
}

async fn read_acts(
    backend: &WgpuBackend,
    buf: &BufRef,
    n: usize,
    act: ActDtype,
) -> Result<Vec<f32>, WgpuError> {
    let asz = match act {
        ActDtype::F32 => 4,
        ActDtype::F16 => 2,
        other => unreachable!("refiner acts f32/f16, got {other:?}"),
    };
    let bytes = backend
        .read_buffer(buf.id, buf.offset, (n * asz) as u64)
        .await?;
    Ok(act_readback_to_f32(act, &bytes, n))
}
