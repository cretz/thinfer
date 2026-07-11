//! Krea 2 Turbo single-stream MMDiT. Ground truth:
//! `third-party/stable-diffusion.cpp/src/model/diffusion/krea2.hpp`
//! (`Krea2Model::forward` + the Krea* blocks).
//!
//! Text and image tokens run as ONE `[txt ++ img]` sequence through all 28
//! blocks. Each block: shared 6-way adaLN from the timestep (`tproj`) PLUS a
//! per-block learned offset (`mod.lin`); gated GQA attention (48 q / 12 kv, per-
//! head `(1+w)` QK-RMSNorm, interleaved RoPE, a full-width sigmoid OUTPUT gate);
//! and a SwiGLU MLP. `KreaRMSNorm` is gemma-style `rms(x)*(1+w)` everywhere.
//!
//! The text side (`txtfusion` over the 12 Qwen3-VL taps + `txtmlp`) depends only
//! on the encoder output, NOT the timestep, so [`KreaDit::prepare_txt`] runs it
//! ONCE per generation into `[txt_tok, DIM]` features that every denoise step
//! reuses. bf16 acts throughout (the residual stream exceeds f16 range).

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError, WgpuPipeline};
use thinfer_core::ops::{
    ActDtype, BcastFmaF32, BcastModulateF32, GeluF32, Op, RmsNormGemmaF32, SigmoidMulF32,
    WeightDtype, WgslConfig,
};
use thinfer_core::quant::{QuantKind, QuantKind::Q8_0};
use thinfer_core::residency::{GpuView, ResidencyError, WeightResidency};
use thinfer_core::trace;
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace, WsBuf};

use crate::common::block::{
    ActBuf, Block, BlockPipelines, BlockWgslConfigs, CoopmatSites, DenseActSites, alloc_act,
    alloc_matmul_out_buf, op_add, op_rope, op_sdpa_f16,
};
use crate::common::embedders::{LinearBiasBufs, rmsnorm_uniform};
use crate::common::seq;
use crate::krea::config;
use crate::krea::loader::{
    KreaAttnHandles, KreaBlockHandles, KreaDitHandles, KreaMlpHandles, KreaTextBlockHandles,
};
use crate::krea::rope::KreaRope;

/// `BlockPipelines` + the extra elementwise pipelines Krea needs beyond the
/// common set: GELU (time/text MLPs), fused `a*sigmoid(b)` (attn output gate),
/// and the gemma `rms(x)*(1+w)` RMSNorm (F32 scales under bf16 acts).
pub struct KreaDitPipelines {
    pub block: BlockPipelines,
    pub gelu: WgpuPipeline,
    pub sigmoid_mul: WgpuPipeline,
    pub rmsnorm_gemma: WgpuPipeline,
}

impl KreaDitPipelines {
    pub async fn compile(
        backend: &WgpuBackend,
        cfgs: &BlockWgslConfigs,
    ) -> Result<Self, WgpuError> {
        let block = BlockPipelines::compile(backend, cfgs).await?;
        let gelu = backend
            .create_pipeline(
                "krea_gelu",
                <GeluF32 as Op>::wgsl(&cfgs.ops),
                "main",
                <GeluF32 as Op>::layout(),
            )
            .await?;
        let sigmoid_mul = backend
            .create_pipeline(
                "krea_sigmoid_mul",
                <SigmoidMulF32 as Op>::wgsl(&cfgs.ops),
                "main",
                <SigmoidMulF32 as Op>::layout(),
            )
            .await?;
        // Krea's `KreaRMSNorm` scales ship F32 but narrow to bf16 on GPU upload
        // (residency), so the gemma norm compiles the bf16-weight variant (same
        // dtype as the block ops).
        let rmsnorm_gemma = backend
            .create_pipeline(
                "krea_rmsnorm_gemma",
                <RmsNormGemmaF32 as thinfer_core::ops::RmsNormOp>::wgsl(&cfgs.ops),
                "main",
                <RmsNormGemmaF32 as thinfer_core::ops::RmsNormOp>::layout(),
            )
            .await?;
        Ok(Self {
            block,
            gelu,
            sigmoid_mul,
            rmsnorm_gemma,
        })
    }
}

/// DiT block config. bf16 acts (residual stream has large-outlier channels
/// beyond f16 range). Block matmuls are `quant` dequant-once; the F16 embedders
/// (`first`, `last.modulation`) ride the bf16 `Adaln` site. `quant` selects the
/// GGUF quant family loaded (Q8_0 canary / Q4_K_M footprint).
pub fn block_cfgs(quant: QuantKind) -> BlockWgslConfigs {
    let ops = WgslConfig {
        bf16_quant_writes: false,
        act_dtype: ActDtype::Bf16,
        weight_dtype: WeightDtype::Bf16,
    };
    let q = WgslConfig {
        weight_dtype: WeightDtype::Quant(quant),
        ..ops
    };
    BlockWgslConfigs {
        matmul_qkv: q,
        matmul_qkv_self: q,
        matmul_proj: q,
        matmul_ffn_up: q,
        matmul_ffn_down: q,
        // The top-level linears routed through `Site::Adaln` (first, tmlp_0/2,
        // tproj, txtmlp_1/3) are ALL Q8_0/Q4_K in the GGUF, not F16 -- so this
        // site must dequant-once like the block matmuls. (`last.modulation.lin`
        // is the only genuinely-F16 top weight, and it is read directly, not
        // through a matmul.) Reading Q8_0 bytes as bf16 here produced all-NaN.
        matmul_adaln: q,
        ops,
        i8_sdpa: false,
        dense_acts: DenseActSites::default(),
        // Krea's big DiT keeps a bf16 residual (f16 diverges), so the f16-only
        // i8-DP4A matmul path is unavailable. Coopmat (tensor cores, f16-cast at
        // the matmul boundary, bf16 residual preserved) is the compute lever for
        // the dense-bf16 sites (~1.3 TFLOPS -> ~6-7). All four block matmul sites
        // have f16-safe A-sides here: qkv/ffn_up/proj read modulated post-norm
        // (O(1)) activations; ffn_down reads the SwiGLU product (Wan ships coopmat
        // there too). NOT the top-level adaln embedders (single-row, no win).
        // Requires fast_sdpa (builds the bf16<->f16 casts). Native-only; the web
        // substrate must keep this off (coopmat is desktop-Vulkan here).
        // Default ON (measured ~4.7x on the DiT compute, quality-imperceptible:
        // same fidelity, trajectory diverges only as a seed nudge would).
        // THINFER_KREA_NO_COOPMAT=1 falls back to dense bf16 (fast_sdpa stays on).
        coopmat_acts: if std::env::var("THINFER_KREA_NO_COOPMAT").is_ok() {
            CoopmatSites::default()
        } else {
            CoopmatSites {
                proj: true,
                ffn_down: true,
                qkv: true,
                ffn_up: true,
            }
        },
        large_d_sdpa: false,
        fast_sdpa: true,
        decode_sdpa: false,
    }
}

// --- matmul site routing (Krea linears are bias-free except the embedders) ----

/// Which compiled matmul site a projection routes through.
#[derive(Clone, Copy)]
enum Site {
    /// Attention q/k/v (quant dequant-once).
    Qkv,
    /// Attention out-proj + the sigmoid gate proj (quant).
    Proj,
    /// SwiGLU gate/up (quant).
    FfnUp,
    /// SwiGLU down (quant).
    FfnDown,
    /// bf16 embedders + the quant top-level MLPs that have no dedicated site.
    Adaln,
}

/// `[rows, n] = x[rows, k] @ wᵀ` through `site`.
fn matmul_site<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    w: BatchBuf<'wsp>,
    rows: u32,
    n: u32,
    k: u32,
    site: Site,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let out = alloc_matmul_out_buf(scope, bp, rows * n)?;
    let dims = scope.u32x4_uniform(rows, n, k, 0)?;
    // Each Quant site may run coopmat (f16-cast tensor cores) when its step was
    // compiled; the dispatch falls back to dequant-once bf16 when it wasn't (or
    // M < WM). i8 is never used here (bf16-residual DiT). Adaln has no coopmat.
    #[allow(clippy::type_complexity)]
    let (coopmat, dq, pipe, op): (_, _, _, _) = match site {
        Site::Qkv => (
            bp.coopmat_qkv.as_ref(),
            bp.dequant_qkv.as_ref(),
            &bp.matmul_qkv,
            &bp.matmuls.qkv,
        ),
        Site::Proj => (
            bp.coopmat_proj.as_ref(),
            bp.dequant_proj.as_ref(),
            &bp.matmul_proj,
            &bp.matmuls.proj,
        ),
        Site::FfnUp => (
            bp.coopmat_ffn_up.as_ref(),
            bp.dequant_ffn_up.as_ref(),
            &bp.matmul_ffn_up,
            &bp.matmuls.ffn_up,
        ),
        Site::FfnDown => (
            bp.coopmat_ffn_down.as_ref(),
            bp.dequant_ffn_down.as_ref(),
            &bp.matmul_ffn_down,
            &bp.matmuls.ffn_down,
        ),
        Site::Adaln => (
            None,
            bp.dequant_adaln.as_ref(),
            &bp.matmul_adaln,
            &bp.matmuls.adaln,
        ),
    };
    Block::dispatch_matmul_site_coopmat(
        scope, bp, x, w, out, dims, None, coopmat, None, None, dq, pipe, op, rows, n, k,
    )?;
    Ok(out)
}

/// Bias-free linear -> dense act `[rows, n]`.
fn linear<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    w: BufRef,
    rows: u32,
    n: u32,
    k: u32,
    site: Site,
) -> Result<ActBuf<'wsp>, WgpuError> {
    let wv = scope.import_copy(w);
    let out = matmul_site(scope, bp, x, wv, rows, n, k, site)?;
    Ok(ActBuf::dense(out))
}

/// Biased linear `x @ wᵀ + bias`.
#[allow(clippy::too_many_arguments)]
fn linear_bias<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    w: &LinearBiasBufs,
    rows: u32,
    n: u32,
    k: u32,
    site: Site,
) -> Result<ActBuf<'wsp>, WgpuError> {
    let wv = scope.import_copy(w.weight);
    let pre = matmul_site(scope, bp, x, wv, rows, n, k, site)?;
    let bv = scope.import_copy(w.bias);
    let out = alloc_act(scope, bp, rows, n)?;
    let u = scope.u32x4_uniform(n, 0, 0, 0)?;
    scope.bcast_add::<thinfer_core::ops::BcastAddF32>(
        &bp.bcast_add,
        pre,
        bv,
        u,
        out.data,
        rows * n,
    )?;
    Ok(out)
}

// --- elementwise wrappers -----------------------------------------------------

/// `out = x * (1 + scale) + shift`, scale/shift `[dim]` broadcast over rows.
#[allow(clippy::too_many_arguments)]
fn op_modulate<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    scale: BatchBuf<'wsp>,
    shift: BatchBuf<'wsp>,
    dst: ActBuf<'wsp>,
    rows: u32,
    dim: u32,
) -> Result<(), WgpuError> {
    let u = scope.u32x4_uniform(dim, 1.0_f32.to_bits(), 0, 0)?;
    scope.bcast_modulate::<BcastModulateF32>(
        &bp.bcast_modulate,
        x.data,
        scale,
        shift,
        u,
        dst.data,
        rows * dim,
    )
}

/// `out = x + gate * y`, gate `[dim]` broadcast over rows.
#[allow(clippy::too_many_arguments)]
fn op_gate_residual<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    gate: BatchBuf<'wsp>,
    y: ActBuf<'wsp>,
    dst: ActBuf<'wsp>,
    rows: u32,
    dim: u32,
) -> Result<(), WgpuError> {
    let u = scope.u32x4_uniform(dim, 0, 0, 0)?;
    scope.bcast_fma::<BcastFmaF32>(&bp.bcast_fma, x.data, gate, y.data, u, dst.data, rows * dim)
}

/// Gemma RMSNorm `out = rms(x) * (1 + w)`, `w` an F32 scale `[dim]`.
fn op_rmsnorm_gemma<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipes: &KreaDitPipelines,
    x: ActBuf<'wsp>,
    w: BatchBuf<'wsp>,
    dst: ActBuf<'wsp>,
    rows: u32,
    dim: u32,
) -> Result<(), WgpuError> {
    let u = rmsnorm_uniform(scope, rows, dim, config::NORM_EPS)?;
    scope.rmsnorm::<RmsNormGemmaF32>(&pipes.rmsnorm_gemma, x.data, w, u, dst.data, rows)
}

/// Fused attention output gate `out = a * sigmoid(b)`.
fn op_sigmoid_mul<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipes: &KreaDitPipelines,
    a: ActBuf<'wsp>,
    b: ActBuf<'wsp>,
    dst: ActBuf<'wsp>,
) -> Result<(), WgpuError> {
    scope.dispatch_op::<SigmoidMulF32>(&pipes.sigmoid_mul, &[a.data, b.data], dst.data)
}

/// Slice a `[dim]` chunk `k` out of a `[1, n*dim]` buffer.
fn mod_chunk<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    src: BatchBuf<'wsp>,
    k: u32,
    dim: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let dst = alloc_act(scope, bp, 1, dim)?;
    let row = bp.act_bytes(dim);
    scope.copy_buffer_to_buffer(src, k as u64 * row, dst.data, 0, row)?;
    Ok(dst.data)
}

// --- gated attention (shared by DiT blocks and txtfusion) ---------------------

/// Gated GQA attention: per-head `(1+w)` QK-RMSNorm, optional interleaved RoPE
/// (`freqs = None` for txtfusion), SDPA, a full-width sigmoid output gate, then
/// the output projection. `x` is the (already modulated/normed) input
/// `[b*s, heads_q*hd]`-compatible `[rows, dim]`. Returns `[rows, dim]`.
#[allow(clippy::too_many_arguments)]
fn gated_attention<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipes: &KreaDitPipelines,
    a: ActBuf<'wsp>,
    w: &AttnBufs,
    b: u32,
    s: u32,
    h_q: u32,
    h_kv: u32,
    hd: u32,
    freqs: Option<BatchBuf<'wsp>>,
) -> Result<ActBuf<'wsp>, WgpuError> {
    let bp = &pipes.block;
    let rows = b * s;
    let dim = h_q * hd;
    let kvdim = h_kv * hd;
    let scale = 1.0 / (hd as f32).sqrt();

    let proj_norm_rope = |wq: BufRef,
                          norm_w: BufRef,
                          heads: u32,
                          nn: u32,
                          site: Site|
     -> Result<ActBuf<'wsp>, WgpuError> {
        let p = linear(scope, bp, a, wq, rows, nn, dim, site)?;
        let normed = alloc_act(scope, bp, rows * heads, hd)?;
        let nw = scope.import_copy(norm_w);
        op_rmsnorm_gemma(scope, pipes, p, nw, normed, rows * heads, hd)?;
        match freqs {
            Some(f) => {
                let roped = alloc_act(scope, bp, rows, nn)?;
                // Per-batch freqs: op_rope applies the same [s, hd] table to each
                // of the b batches (rows = b*s), broadcasting over heads.
                op_rope(scope, bp, normed, f, roped, rows, heads, hd)?;
                Ok(roped)
            }
            None => Ok(normed),
        }
    };

    let q = proj_norm_rope(w.wq, w.qnorm, h_q, dim, Site::Qkv)?;
    let k = proj_norm_rope(w.wk, w.knorm, h_kv, kvdim, Site::Qkv)?;
    let v = linear(scope, bp, a, w.wv, rows, kvdim, dim, Site::Qkv)?;

    let sa = alloc_act(scope, bp, rows, dim)?;
    // No causal mask (full attention); has_mask=0 so the mask binding is unread.
    let mask = v.data; // any same-size buffer; unread at has_mask=0
    // Fast mixed-precision path: casts the post-norm/rope (O(1), f16-safe) Q/K/V
    // to f16 for the subgroup SDPA, residual stays bf16. Falls back to the f32
    // sdpa when fast pipelines weren't built.
    op_sdpa_f16(
        scope, bp, q, k, v, mask, sa, b, s, s, h_q, h_kv, hd, scale, 0,
    )?;

    // Output gate: sa * sigmoid(gate(a)), then out-proj.
    let gate = linear(scope, bp, a, w.gate, rows, dim, dim, Site::Proj)?;
    let gated = alloc_act(scope, bp, rows, dim)?;
    op_sigmoid_mul(scope, pipes, sa, gate, gated)?;
    linear(scope, bp, gated, w.wo, rows, dim, dim, Site::Proj)
}

/// SwiGLU MLP `down(silu(gate(x)) * up(x))`. `inner` is the intermediate width.
fn swiglu<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipes: &KreaDitPipelines,
    x: ActBuf<'wsp>,
    w: &MlpBufs,
    rows: u32,
    dim: u32,
    inner: u32,
) -> Result<ActBuf<'wsp>, WgpuError> {
    let bp = &pipes.block;
    let g = linear(scope, bp, x, w.gate, rows, inner, dim, Site::FfnUp)?;
    let u = linear(scope, bp, x, w.up, rows, inner, dim, Site::FfnUp)?;
    let h = alloc_act(scope, bp, rows, inner)?;
    scope.dispatch_op::<thinfer_core::ops::SiluMulF32>(&bp.silu_mul, &[g.data, u.data], h.data)?;
    linear(scope, bp, h, w.down, rows, dim, inner, Site::FfnDown)
}

// --- per-block weight views ---------------------------------------------------

struct AttnBufs {
    wq: BufRef,
    wk: BufRef,
    wv: BufRef,
    gate: BufRef,
    wo: BufRef,
    qnorm: BufRef,
    knorm: BufRef,
}
struct MlpBufs {
    gate: BufRef,
    up: BufRef,
    down: BufRef,
}

struct AttnViews<'a> {
    wq: GpuView<'a>,
    wk: GpuView<'a>,
    wv: GpuView<'a>,
    gate: GpuView<'a>,
    wo: GpuView<'a>,
    qnorm: GpuView<'a>,
    knorm: GpuView<'a>,
}
struct MlpViews<'a> {
    gate: GpuView<'a>,
    up: GpuView<'a>,
    down: GpuView<'a>,
}

impl<'a> AttnViews<'a> {
    async fn acquire<S: WeightSource>(
        h: &KreaAttnHandles,
        r: &'a WeightResidency<S>,
        b: &WgpuBackend,
    ) -> Result<Self, ResidencyError<S::Error, WgpuError>> {
        Ok(Self {
            wq: r.acquire(h.wq, b).await?,
            wk: r.acquire(h.wk, b).await?,
            wv: r.acquire(h.wv, b).await?,
            gate: r.acquire(h.gate, b).await?,
            wo: r.acquire(h.wo, b).await?,
            qnorm: r.acquire(h.qnorm, b).await?,
            knorm: r.acquire(h.knorm, b).await?,
        })
    }
    fn bufs(&self) -> AttnBufs {
        AttnBufs {
            wq: self.wq.buf(),
            wk: self.wk.buf(),
            wv: self.wv.buf(),
            gate: self.gate.buf(),
            wo: self.wo.buf(),
            qnorm: self.qnorm.buf(),
            knorm: self.knorm.buf(),
        }
    }
}
impl<'a> MlpViews<'a> {
    async fn acquire<S: WeightSource>(
        h: &KreaMlpHandles,
        r: &'a WeightResidency<S>,
        b: &WgpuBackend,
    ) -> Result<Self, ResidencyError<S::Error, WgpuError>> {
        Ok(Self {
            gate: r.acquire(h.gate, b).await?,
            up: r.acquire(h.up, b).await?,
            down: r.acquire(h.down, b).await?,
        })
    }
    fn bufs(&self) -> MlpBufs {
        MlpBufs {
            gate: self.gate.buf(),
            up: self.up.buf(),
            down: self.down.buf(),
        }
    }
}

struct BlockViews<'a> {
    prenorm: GpuView<'a>,
    postnorm: GpuView<'a>,
    mod_lin: GpuView<'a>,
    attn: AttnViews<'a>,
    mlp: MlpViews<'a>,
}
struct BlockBufs {
    prenorm: BufRef,
    postnorm: BufRef,
    mod_lin: BufRef,
    attn: AttnBufs,
    mlp: MlpBufs,
}
impl<'a> BlockViews<'a> {
    async fn acquire<S: WeightSource>(
        h: &KreaBlockHandles,
        r: &'a WeightResidency<S>,
        b: &WgpuBackend,
    ) -> Result<Self, ResidencyError<S::Error, WgpuError>> {
        Ok(Self {
            prenorm: r.acquire(h.prenorm, b).await?,
            postnorm: r.acquire(h.postnorm, b).await?,
            mod_lin: r.acquire(h.mod_lin, b).await?,
            attn: AttnViews::acquire(&h.attn, r, b).await?,
            mlp: MlpViews::acquire(&h.mlp, r, b).await?,
        })
    }
    fn bufs(&self) -> BlockBufs {
        BlockBufs {
            prenorm: self.prenorm.buf(),
            postnorm: self.postnorm.buf(),
            mod_lin: self.mod_lin.buf(),
            attn: self.attn.bufs(),
            mlp: self.mlp.bufs(),
        }
    }
}

/// One image-stream block: shared-mod (tvec + per-block offset) chunked into 6,
/// gated GQA attn (with RoPE) + gate residual, SwiGLU + gate residual.
#[allow(clippy::too_many_arguments)]
fn block_forward<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipes: &KreaDitPipelines,
    hidden_in: BatchBuf<'wsp>,
    tvec: BatchBuf<'wsp>,
    freqs: BatchBuf<'wsp>,
    joint: u32,
    hidden_out: BatchBuf<'wsp>,
    bufs: &'wsp BlockBufs,
) -> Result<(), WgpuError> {
    let bp = &pipes.block;
    let dim = config::DIM as u32;
    let hd = config::HEAD_DIM as u32;
    let hq = config::N_HEADS as u32;
    let hkv = config::N_KV_HEADS as u32;
    let inner = config::MLP_INNER as u32;
    let hidden = ActBuf::dense(hidden_in);

    // mods = chunk6(tvec + mod.lin)  [6 x dim]
    let modp = alloc_act(scope, bp, 1, 6 * dim)?;
    let ml = scope.import_copy(bufs.mod_lin);
    op_add(scope, bp, ActBuf::dense(tvec), ActBuf::dense(ml), modp)?;
    let sig = |k| mod_chunk(scope, bp, modp.data, k, dim);
    let (scale_msa, shift_msa, gate_msa) = (sig(0)?, sig(1)?, sig(2)?);
    let (scale_mlp, shift_mlp, gate_mlp) = (sig(3)?, sig(4)?, sig(5)?);

    // attn: modulate(prenorm(hidden)) -> gated attn -> gate residual
    let n1 = alloc_act(scope, bp, joint, dim)?;
    let pw = scope.import_copy(bufs.prenorm);
    op_rmsnorm_gemma(scope, pipes, hidden, pw, n1, joint, dim)?;
    let a = alloc_act(scope, bp, joint, dim)?;
    op_modulate(scope, bp, n1, scale_msa, shift_msa, a, joint, dim)?;
    let attn = gated_attention(
        scope,
        pipes,
        a,
        &bufs.attn,
        1,
        joint,
        hq,
        hkv,
        hd,
        Some(freqs),
    )?;
    let h1 = alloc_act(scope, bp, joint, dim)?;
    op_gate_residual(scope, bp, hidden, gate_msa, attn, h1, joint, dim)?;

    // mlp: modulate(postnorm(h1)) -> swiglu -> gate residual
    let n2 = alloc_act(scope, bp, joint, dim)?;
    let qw = scope.import_copy(bufs.postnorm);
    op_rmsnorm_gemma(scope, pipes, h1, qw, n2, joint, dim)?;
    let m = alloc_act(scope, bp, joint, dim)?;
    op_modulate(scope, bp, n2, scale_mlp, shift_mlp, m, joint, dim)?;
    let mlp = swiglu(scope, pipes, m, &bufs.mlp, joint, dim, inner)?;
    op_gate_residual(
        scope,
        bp,
        h1,
        gate_mlp,
        mlp,
        ActBuf::dense(hidden_out),
        joint,
        dim,
    )?;
    Ok(())
}

// --- text-fusion block (no modulation, no rope; over text width) --------------

/// One `txtfusion` block: prenorm -> gated MHA (no rope) -> residual; postnorm
/// -> SwiGLU -> residual. `b` batches independent attention groups (layerwise =
/// per-token over the 12-layer axis; refiner = one group over the tokens).
#[allow(clippy::too_many_arguments)]
fn text_block_forward<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipes: &KreaDitPipelines,
    x_in: BatchBuf<'wsp>,
    b: u32,
    s: u32,
    x_out: BatchBuf<'wsp>,
    bufs: &'wsp TextBlockBufs,
) -> Result<(), WgpuError> {
    let bp = &pipes.block;
    let dim = config::TEXT_DIM as u32;
    let hd = config::TEXT_HEAD_DIM as u32;
    let heads = config::TEXT_HEADS as u32;
    let inner = config::TEXT_MLP_INNER as u32;
    let rows = b * s;
    let x = ActBuf::dense(x_in);

    let n1 = alloc_act(scope, bp, rows, dim)?;
    let pw = scope.import_copy(bufs.prenorm);
    op_rmsnorm_gemma(scope, pipes, x, pw, n1, rows, dim)?;
    let attn = gated_attention(scope, pipes, n1, &bufs.attn, b, s, heads, heads, hd, None)?;
    let h1 = alloc_act(scope, bp, rows, dim)?;
    op_add(scope, bp, x, attn, h1)?;

    let n2 = alloc_act(scope, bp, rows, dim)?;
    let qw = scope.import_copy(bufs.postnorm);
    op_rmsnorm_gemma(scope, pipes, h1, qw, n2, rows, dim)?;
    let mlp = swiglu(scope, pipes, n2, &bufs.mlp, rows, dim, inner)?;
    op_add(scope, bp, h1, mlp, ActBuf::dense(x_out))?;
    Ok(())
}

struct TextBlockBufs {
    prenorm: BufRef,
    postnorm: BufRef,
    attn: AttnBufs,
    mlp: MlpBufs,
}
struct TextBlockViews<'a> {
    prenorm: GpuView<'a>,
    postnorm: GpuView<'a>,
    attn: AttnViews<'a>,
    mlp: MlpViews<'a>,
}
impl<'a> TextBlockViews<'a> {
    async fn acquire<S: WeightSource>(
        h: &KreaTextBlockHandles,
        r: &'a WeightResidency<S>,
        b: &WgpuBackend,
    ) -> Result<Self, ResidencyError<S::Error, WgpuError>> {
        Ok(Self {
            prenorm: r.acquire(h.prenorm, b).await?,
            postnorm: r.acquire(h.postnorm, b).await?,
            attn: AttnViews::acquire(&h.attn, r, b).await?,
            mlp: MlpViews::acquire(&h.mlp, r, b).await?,
        })
    }
    fn bufs(&self) -> TextBlockBufs {
        TextBlockBufs {
            prenorm: self.prenorm.buf(),
            postnorm: self.postnorm.buf(),
            attn: self.attn.bufs(),
            mlp: self.mlp.bufs(),
        }
    }
}

// --- driver -------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct DitOutput {
    /// Velocity in patch-token space `[img_seq, PACKED_CH=64]`.
    pub velocity: Vec<f32>,
    pub img_seq: usize,
}

#[derive(Debug)]
pub enum DitError<SE: core::fmt::Debug> {
    Wgpu(WgpuError),
    Residency(ResidencyError<SE, WgpuError>),
}
impl<SE: core::fmt::Debug> From<WgpuError> for DitError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for DitError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

pub struct KreaDit {
    rope: KreaRope,
}

fn upload_act(
    scratch: &Workspace<WgpuBackend>,
    backend: &WgpuBackend,
    bp: &BlockPipelines,
    host: &[f32],
    n: u32,
) -> Result<thinfer_core::workspace::WsBuf<WgpuBackend>, WgpuError> {
    let buf = scratch.alloc(bp.act_bytes(n))?;
    let bytes = seq::act_upload_bytes(bp.act_dtype, host);
    backend.write_buffer(buf.id, 0, &bytes)?;
    Ok(buf)
}

fn upload_freqs(
    scratch: &Workspace<WgpuBackend>,
    backend: &WgpuBackend,
    bp: &BlockPipelines,
    freqs: &[f32],
) -> Result<thinfer_core::workspace::WsBuf<WgpuBackend>, WgpuError> {
    let bytes = seq::freqs_upload_bytes(bp.act_dtype, freqs);
    let buf = scratch.alloc(bytes.len() as u64)?;
    backend.write_buffer(buf.id, 0, &bytes)?;
    Ok(buf)
}

/// Read an act buffer back to host f32 `[n]`.
async fn download_act(
    backend: &WgpuBackend,
    bp: &BlockPipelines,
    buf: &WsBuf<WgpuBackend>,
    n: u32,
) -> Result<Vec<f32>, WgpuError> {
    let bytes = backend.read_buffer(buf.id(), 0, bp.act_bytes(n)).await?;
    Ok(seq::act_readback_to_f32(bp.act_dtype, &bytes, n as usize))
}

/// `get_timestep_embedding(t, 256, flip_sin_to_cos=True, scale=1000,
/// max_period=10000)`: `[cos(a) ++ sin(a)]`, `a_j = 1000*t*exp(-ln(10000)*
/// j/128)`.
fn timestep_sinusoid(t: f32) -> Vec<f32> {
    const HALF: usize = config::TIMESTEP_DIM / 2;
    let mut out = vec![0.0_f32; 2 * HALF];
    for j in 0..HALF {
        let freq = (-(config::TIMESTEP_MAX_PERIOD.ln()) * j as f32 / HALF as f32).exp();
        let a = config::TIMESTEP_SCALE * t * freq;
        out[j] = a.cos();
        out[HALF + j] = a.sin();
    }
    out
}

impl KreaDit {
    pub fn new() -> Self {
        Self {
            rope: KreaRope::new(),
        }
    }

    /// Run `txtfusion` + `txtmlp` ONCE over the 12 encoder taps
    /// `[txt_tok*TEXT_LAYERS*TEXT_DIM]` (token-major, then layer, then dim).
    /// Returns the projected text features `[txt_tok*DIM]` the denoise loop
    /// reuses. The projector `Linear(12->1)` contraction is done on host (its 12
    /// weights are read from the source); everything else is on GPU.
    pub async fn prepare_txt<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipes: &KreaDitPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        handles: &KreaDitHandles,
        taps: &[f32],
    ) -> Result<Vec<f32>, DitError<S::Error>> {
        let bp = &pipes.block;
        let td = config::TEXT_DIM;
        let tl = config::TEXT_LAYERS;
        assert_eq!(taps.len() % (tl * td), 0, "taps not a multiple of 12*2560");
        let txt_tok = taps.len() / (tl * td);
        let tf = &handles.top.txtfusion;

        // layerwise blocks: attention over the 12-layer axis, batched per token.
        // Input rows = txt_tok*12, batch = txt_tok, seq = 12.
        let rows = (txt_tok * tl) as u32;
        let mut cur = upload_act(scratch, backend, bp, taps, rows * td as u32)?;
        for h in tf.layerwise.iter() {
            let out = scratch.alloc(bp.act_bytes(rows * td as u32))?;
            let v = TextBlockViews::acquire(h, residency, backend).await?;
            let b = v.bufs();
            let scope = scratch.batch();
            let x = scope.import_copy(cur.as_buf_ref());
            let o = scope.import_copy(out.as_buf_ref());
            text_block_forward(&scope, pipes, x, txt_tok as u32, tl as u32, o, &b)?;
            scope.submit_void().await?;
            cur = out;
        }

        // projector Linear(12->1): out[tok, d] = sum_l cur[tok, l, d] * proj[l].
        // Read the 12 projector weights (F32) + the layerwise output to host,
        // contract, re-upload. Runs once per generation.
        let proj: Vec<f32> = {
            // Passthrough F32 scales narrow to bf16 on GPU upload, so the
            // projector buffer is `tl` bf16 values (2 bytes each).
            let pv = residency.acquire(tf.projector, backend).await?;
            let nbytes = (tl * 2) as u64;
            let pbuf = scratch.alloc(nbytes)?;
            {
                let scope = scratch.batch();
                let src = scope.import_copy(pv.buf());
                let dst = scope.import_copy(pbuf.as_buf_ref());
                scope.copy_buffer_to_buffer(src, 0, dst, 0, nbytes)?;
                scope.submit_void().await?;
            }
            let bytes = backend.read_buffer(pbuf.id(), 0, nbytes).await?;
            bytes
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect()
        };
        let lw = download_act(backend, bp, &cur, rows * td as u32).await?;
        let mut collapsed = vec![0.0_f32; txt_tok * td];
        for tok in 0..txt_tok {
            for (l, &pw) in proj.iter().enumerate() {
                let base = (tok * tl + l) * td;
                let dst = tok * td;
                for d in 0..td {
                    collapsed[dst + d] += lw[base + d] * pw;
                }
            }
        }

        // refiner blocks: attention over the tokens (batch 1, seq txt_tok).
        let rrows = txt_tok as u32;
        let mut cur = upload_act(scratch, backend, bp, &collapsed, rrows * td as u32)?;
        for h in tf.refiner.iter() {
            let out = scratch.alloc(bp.act_bytes(rrows * td as u32))?;
            let v = TextBlockViews::acquire(h, residency, backend).await?;
            let b = v.bufs();
            let scope = scratch.batch();
            let x = scope.import_copy(cur.as_buf_ref());
            let o = scope.import_copy(out.as_buf_ref());
            text_block_forward(&scope, pipes, x, 1, txt_tok as u32, o, &b)?;
            scope.submit_void().await?;
            cur = out;
        }

        // txtmlp: gemma-rmsnorm -> Linear(2560->6144)+b -> gelu(tanh) ->
        // Linear(6144->6144)+b. Output [txt_tok, DIM].
        let dim = config::DIM as u32;
        let out = scratch.alloc(bp.act_bytes(rrows * dim))?;
        {
            let nv = residency.acquire(handles.top.txtmlp_norm, backend).await?;
            let l1 = handles.top.txtmlp_1.acquire(residency, backend).await?;
            let l3 = handles.top.txtmlp_3.acquire(residency, backend).await?;
            let (w1, w3) = (l1.bufs(), l3.bufs());
            let scope = scratch.batch();
            let x = ActBuf::dense(scope.import_copy(cur.as_buf_ref()));
            let normed = alloc_act(&scope, bp, rrows, td as u32)?;
            let nw = scope.import_copy(nv.buf());
            op_rmsnorm_gemma(&scope, pipes, x, nw, normed, rrows, td as u32)?;
            let h1 = linear_bias(&scope, bp, normed, &w1, rrows, dim, td as u32, Site::Adaln)?;
            let g = alloc_act(&scope, bp, rrows, dim)?;
            scope.dispatch_op::<GeluF32>(&pipes.gelu, &[h1.data], g.data)?;
            let o = linear_bias(&scope, bp, g, &w3, rrows, dim, dim, Site::Adaln)?;
            let dst = scope.import_copy(out.as_buf_ref());
            scope.copy_buffer_to_buffer(o.data, 0, dst, 0, bp.act_bytes(rrows * dim))?;
            scope.submit_void().await?;
        }
        Ok(download_act(backend, bp, &out, rrows * dim).await?)
    }

    /// One denoise-step forward. `img_tokens` = packed latents `[img_seq, 64]`;
    /// `txt_features` = the `prepare_txt` output `[txt_seq, DIM]`. Returns the
    /// velocity over the image tokens.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipes: &KreaDitPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        handles: &KreaDitHandles,
        img_tokens: &[f32],
        txt_features: &[f32],
        timestep: f32,
        gh: usize,
        gw: usize,
    ) -> Result<DitOutput, DitError<S::Error>> {
        let bp = &pipes.block;
        let diag = tracing::enabled!(target: "thinfer::diag", tracing::Level::DEBUG);
        let dim = config::DIM as u32;
        let packed = config::PACKED_CH as u32;
        let img_seq = (gh * gw) as u32;
        assert_eq!(img_tokens.len(), gh * gw * config::PACKED_CH);
        assert_eq!(txt_features.len() % config::DIM, 0);
        let txt_seq = (txt_features.len() / config::DIM) as u32;
        let joint = txt_seq + img_seq;

        let img_buf = upload_act(scratch, backend, bp, img_tokens, img_seq * packed)?;
        let txt_buf = upload_act(scratch, backend, bp, txt_features, txt_seq * dim)?;

        // hidden = concat[ txt_features , first(img_tokens) ]  [joint, dim]
        let hidden = scratch.alloc(bp.act_bytes(joint * dim))?;
        {
            let fv = handles.top.first.acquire(residency, backend).await?;
            let w = fv.bufs();
            let scope = scratch.batch();
            let img = ActBuf::dense(scope.import_copy(img_buf.as_buf_ref()));
            let ie = linear_bias(&scope, bp, img, &w, img_seq, dim, packed, Site::Adaln)?;
            let dst = scope.import_copy(hidden.as_buf_ref());
            let txt = scope.import_copy(txt_buf.as_buf_ref());
            scope.copy_buffer_to_buffer(txt, 0, dst, 0, bp.act_bytes(txt_seq * dim))?;
            scope.copy_buffer_to_buffer(
                ie.data,
                0,
                dst,
                bp.act_bytes(txt_seq * dim),
                bp.act_bytes(img_seq * dim),
            )?;
            scope.submit_void().await?;
        }

        // timestep: temb = tmlp.2(silu?gelu... KreaTimeMLP uses gelu). temb ->
        // tvec = tproj.1(gelu(temb)). temb is also the final-layer mod source.
        let sin = timestep_sinusoid(timestep);
        let sin_buf = upload_act(scratch, backend, bp, &sin, config::TIMESTEP_DIM as u32)?;
        let temb = scratch.alloc(bp.act_bytes(dim))?;
        let tvec = scratch.alloc(bp.act_bytes(6 * dim))?;
        {
            let t0 = handles.top.tmlp_0.acquire(residency, backend).await?;
            let t2 = handles.top.tmlp_2.acquire(residency, backend).await?;
            let tp = handles.top.tproj.acquire(residency, backend).await?;
            let (w0, w2, wp) = (t0.bufs(), t2.bufs(), tp.bufs());
            let scope = scratch.batch();
            let x = ActBuf::dense(scope.import_copy(sin_buf.as_buf_ref()));
            let h0 = linear_bias(
                &scope,
                bp,
                x,
                &w0,
                1,
                dim,
                config::TIMESTEP_DIM as u32,
                Site::Adaln,
            )?;
            let g0 = alloc_act(&scope, bp, 1, dim)?;
            scope.dispatch_op::<GeluF32>(&pipes.gelu, &[h0.data], g0.data)?;
            let tb = linear_bias(&scope, bp, g0, &w2, 1, dim, dim, Site::Adaln)?;
            let tdst = scope.import_copy(temb.as_buf_ref());
            scope.copy_buffer_to_buffer(tb.data, 0, tdst, 0, bp.act_bytes(dim))?;
            // tvec = tproj.1(gelu(temb))
            let gt = alloc_act(&scope, bp, 1, dim)?;
            scope.dispatch_op::<GeluF32>(&pipes.gelu, &[tb.data], gt.data)?;
            let tv = linear_bias(&scope, bp, gt, &wp, 1, 6 * dim, dim, Site::Adaln)?;
            let vdst = scope.import_copy(tvec.as_buf_ref());
            scope.copy_buffer_to_buffer(tv.data, 0, vdst, 0, bp.act_bytes(6 * dim))?;
            scope.submit_void().await?;
        }

        // rope freqs over [txt ++ img]
        let freqs = self.rope.freqs(txt_seq as usize, gh, gw);
        let freqs_buf = upload_freqs(scratch, backend, bp, &freqs)?;

        // Diag: scan a buffer for non-finite values (readback, diag-gated only).
        if diag {
            let h = download_act(backend, bp, &hidden, joint * dim).await?;
            let t = download_act(backend, bp, &tvec, 6 * dim).await?;
            tracing::event!(
                target: "thinfer::diag", tracing::Level::DEBUG,
                hidden_nonfinite = h.iter().filter(|v| !v.is_finite()).count(),
                tvec_nonfinite = t.iter().filter(|v| !v.is_finite()).count(),
                "krea DiT pre-block (concat + timestep)"
            );
        }

        // 28 blocks (one submit each).
        let mut hidden = hidden;
        let mut first_nan_block: Option<usize> = None;
        for (i, bh) in handles.blocks.iter().enumerate() {
            let out = scratch.alloc(bp.act_bytes(joint * dim))?;
            let v = BlockViews::acquire(bh, residency, backend).await?;
            let bufs = v.bufs();
            let scope = scratch.batch();
            let hin = scope.import_copy(hidden.as_buf_ref());
            let tvecb = scope.import_copy(tvec.as_buf_ref());
            let fb = scope.import_copy(freqs_buf.as_buf_ref());
            let hout = scope.import_copy(out.as_buf_ref());
            let _s = trace::scope!("krea.block", i = i).entered();
            block_forward(&scope, pipes, hin, tvecb, fb, joint, hout, &bufs)?;
            scope.submit_void().await?;
            hidden = out;
            if diag && first_nan_block.is_none() {
                let h = download_act(backend, bp, &hidden, joint * dim).await?;
                if h.iter().any(|v| !v.is_finite()) {
                    first_nan_block = Some(i);
                    let bad = h.iter().filter(|v| !v.is_finite()).count();
                    tracing::event!(
                        target: "thinfer::diag", tracing::Level::DEBUG,
                        block = i, nonfinite = bad, total = h.len(),
                        "krea DiT: FIRST non-finite block output"
                    );
                }
            }
        }

        // final layer: scale/shift = last.modulation.lin[2,dim] + temb (broadcast),
        // modulate(last.norm(hidden)), then last.linear -> [joint, PACKED_CH];
        // slice the image rows.
        let vel = scratch.alloc(bp.act_bytes(img_seq * packed))?;
        {
            let nv = residency.acquire(handles.top.last_norm, backend).await?;
            let mv = residency.acquire(handles.top.last_mod, backend).await?;
            let lv = handles.top.last_linear.acquire(residency, backend).await?;
            let lw = lv.bufs();
            let scope = scratch.batch();
            let h = ActBuf::dense(scope.import_copy(hidden.as_buf_ref()));
            let tv = scope.import_copy(temb.as_buf_ref());
            // scale = last_mod[row0] + temb ; shift = last_mod[row1] + temb
            let mb = scope.import_copy(mv.buf());
            let row = bp.act_bytes(dim);
            let scale = alloc_act(&scope, bp, 1, dim)?;
            let shift = alloc_act(&scope, bp, 1, dim)?;
            let m0 = scope.subview(&mb, 0, row);
            let m1 = scope.subview(&mb, row, row);
            op_add(&scope, bp, ActBuf::dense(m0), ActBuf::dense(tv), scale)?;
            op_add(&scope, bp, ActBuf::dense(m1), ActBuf::dense(tv), shift)?;
            let nw = scope.import_copy(nv.buf());
            let normed = alloc_act(&scope, bp, joint, dim)?;
            op_rmsnorm_gemma(&scope, pipes, h, nw, normed, joint, dim)?;
            let modded = alloc_act(&scope, bp, joint, dim)?;
            op_modulate(
                &scope, bp, normed, scale.data, shift.data, modded, joint, dim,
            )?;
            let out = linear_bias(&scope, bp, modded, &lw, joint, packed, dim, Site::Proj)?;
            // slice image rows [txt_seq.. ] -> velocity
            let vdst = scope.import_copy(vel.as_buf_ref());
            scope.copy_buffer_to_buffer(
                out.data,
                bp.act_bytes(txt_seq * packed),
                vdst,
                0,
                bp.act_bytes(img_seq * packed),
            )?;
            scope.submit_void().await?;
        }

        let velocity = download_act(backend, bp, &vel, img_seq * packed).await?;
        Ok(DitOutput {
            velocity,
            img_seq: img_seq as usize,
        })
    }
}

impl Default for KreaDit {
    fn default() -> Self {
        Self::new()
    }
}

// Q8_0 is the default DiT quant (canary + quality tier); Q4_K_M is a footprint
// option wired at the manifest/config layer.
pub const DEFAULT_QUANT: QuantKind = Q8_0;
