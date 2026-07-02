//! Wan / SkyReels-V2 DiT transformer block forward (`SkyReelsV2TransformerBlock`,
//! `transformer_skyreels_v2.py`). Full-attention video DiT block:
//!
//! ```text
//! shift_msa,scale_msa,gate_msa,c_shift,c_scale,c_gate = mod   // [inner] vectors
//! // 1. self-attention (RoPE3D, interleaved-pair)
//! n  = norm1(x) * (1 + scale_msa) + shift_msa                 // FP32 LayerNorm, no affine
//! a  = self_attn(n)                                           // q/k/v biased, qk RMSNorm-across-heads
//! x  = x + gate_msa * a
//! // 2. cross-attention to the umT5 text states (no RoPE, no gate)
//! n  = norm2(x)                                               // FP32 LayerNorm, affine
//! x  = x + cross_attn(n, text)
//! // 3. feed-forward (gelu-tanh, non-gated)
//! n  = norm3(x) * (1 + c_scale) + c_shift                     // FP32 LayerNorm, no affine
//! x  = x + c_gate * proj_out(gelu_new(proj_in(n)))
//! ```
//!
//! Arch deltas vs the Z-Image DiT block (`z_image/block.rs`):
//! - Block norms are LayerNorm (mean-subtract), not RMSNorm; norm1/norm3 have
//!   no affine, norm2 (cross_attn_norm) has weight+bias.
//! - Separate q/k/v projections WITH bias (Z-Image fuses QKV, no bias); bias is
//!   a post-matmul channel-broadcast `bcast_add`.
//! - `qk_norm = rms_norm_across_heads`: RMSNorm over the full inner dim applied
//!   to `[rows, inner]` BEFORE the head split (not per-head like Z-Image).
//! - RoPE3D is interleaved-pair (`RopeF32`), NOT half-rot.
//! - A cross-attention stage to the umT5 text states (no RoPE, no mask).
//! - Non-gated gelu-tanh FFN (`proj_out(gelu_new(proj_in(x)))`), reusing the new
//!   [`GeluF32`] op (single-input sibling of umT5's `GeluMulF32`).
//! - Modulation is CHANNEL-BROADCAST: the distilled T2V line feeds one scalar
//!   timestep, so scale/shift/gate are `[inner]` vectors uniform over all
//!   tokens. norm modulation is `bcast_affine`(bias=1) + `bcast_add`; the gated
//!   residual is a single `bcast_fma` (matches the Z-Image modulation form).
//!
//! The `scale_shift_table` add (`table[6,inner] + timestep_proj`) that produces
//! the six modulation signals lives in the driver; this block consumes the six
//! ready `[inner]` vectors via [`WanMod`].

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError, WgpuPipeline};
use thinfer_core::ops::{
    BcastAddF32, BcastFmaF32, BcastModulateF32, BcastMulF32, GeluF32, LayerNormF32, MatMulF32,
    RopeF32,
};
use thinfer_core::residency::{GpuView, ResidencyError, WeightHandle, WeightResidency};
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::{BatchBuf, BatchScope};

use crate::common::block::{
    ActBuf, Block, BlockPipelines, alloc_act, alloc_matmul_out_buf, copy_tap, op_add, op_rmsnorm,
    op_sdpa, op_sdpa_f16, op_sdpa_f16_win,
};

/// Query rows per FFN chunk. The FFN is position-wise, so it is processed in
/// row tiles to cap the peak `rows * ffn_dim` transient working set: a long clip
/// (81f @ 832x480 = 32760 rows) would otherwise reserve multiple GB of FFN
/// scratch and starve the streamed 14B weight cache on the 8GB card. 8192 keeps
/// each chunk's FFN intermediate near ~0.45GB (bf16) while staying large enough
/// for full matmul efficiency; short clips collapse to a single chunk. The
/// denoise driver reads this to size its transient-reserve estimate.
pub(crate) const FFN_TILE_ROWS: u32 = 8192;

/// Wan-family-invariant DiT constants (the same across SkyReels-1.3B, FastWan
/// 5B, Wan2.2-14B): only `num_heads`, `ffn_dim`, `num_layers`, and the latent
/// channel count differ between variants -- those live in [`WanDitConfig`]. The
/// `qk_norm rms_norm_across_heads`, `cross_attn_norm`, gelu-tanh FFN, and patch
/// structure are shared.
pub mod config {
    /// Per-head dim. 128 across the whole Wan family (variants scale `num_heads`,
    /// not `head_dim`), so the rope axis split below is a fixed function of it.
    pub const HEAD_DIM: usize = 128;
    /// umT5-XXL hidden width (the text encoder, shared by every Wan variant).
    pub const TEXT_DIM: usize = 4096;
    pub const FREQ_DIM: usize = 256;
    pub const EPS: f32 = 1e-6;
    /// Text context length the DiT cross-attends to (umT5 output rows).
    pub const TEXT_SEQ: usize = 512;

    /// 3D patch (`p_t, p_h, p_w`) the `patch_embedding` Conv3d folds latent
    /// voxels into tokens with. Kernel == stride == patch, so the conv is an
    /// affine patchify (see `wan/patchify.rs`).
    pub const PATCH_T: usize = 1;
    pub const PATCH_H: usize = 2;
    pub const PATCH_W: usize = 2;

    /// RoPE3D theta and the per-axis (t, h, w) sub-dimensions of `HEAD_DIM`.
    /// `h = w = 2 * (head_dim / 6)`, `t = head_dim - h - w` (diffusers
    /// `WanRotaryPosEmbed`). At head_dim 128: t=44, h=w=42.
    pub const ROPE_THETA: f32 = 10_000.0;
    pub const ROPE_H_DIM: usize = 2 * (HEAD_DIM / 6); // 42
    pub const ROPE_W_DIM: usize = 2 * (HEAD_DIM / 6); // 42
    pub const ROPE_T_DIM: usize = HEAD_DIM - ROPE_H_DIM - ROPE_W_DIM; // 44
    /// Max per-axis grid length the freq tables are precomputed to
    /// (`rope_max_seq_len`).
    pub const ROPE_MAX_SEQ_LEN: usize = 1024;
}

/// Per-variant Wan DiT geometry. Threaded through the driver / block / loader /
/// condition embedder so a new variant (e.g. Wan2.2-14B) is a new constructor,
/// not a code fork. Family-invariant numbers stay in [`config`].
#[derive(Clone, Copy, Debug)]
pub struct WanDitConfig {
    pub num_heads: usize,
    pub ffn_dim: usize,
    pub num_layers: usize,
    /// Latent channels in / out (== the VAE `z_dim`): 48 for the Wan2.2-TI2V
    /// high-compression VAE, 16 for the Wan2.1 VAE.
    pub in_channels: usize,
    pub out_channels: usize,
}

impl WanDitConfig {
    /// FastWan2.2-TI2V-5B-FullAttn (== LongLive-2.0-5B base). Audited against the
    /// `transformer/config.json` (`WanTransformer3DModel`): num_attention_heads
    /// 24, attention_head_dim 128 -> inner 3072; num_layers 30; ffn_dim 14336;
    /// in/out_channels 48.
    pub fn fastwan_ti2v_5b() -> Self {
        Self {
            num_heads: 24,
            ffn_dim: 14336,
            num_layers: 30,
            in_channels: 48,
            out_channels: 48,
        }
    }

    /// LongLive-2.0-5B: the causal/AR finetune of the FastWan base, structurally
    /// identical (verified tensor-by-tensor against the real `model_bf16.pt`:
    /// dim 3072 / ffn 14336 / 24 heads / 30 blocks / in=out 48). Named separately
    /// so the variant registry selects it by intent and the two can diverge later
    /// without silently sharing a number.
    pub fn longlive_2_0_5b() -> Self {
        Self::fastwan_ti2v_5b()
    }

    /// Wan2.2-T2V-A14B (each MoE expert). Audited against the upstream
    /// `high/low_noise_model/config.json` (`WanModel`): dim 5120 (num_heads 40 *
    /// head_dim 128), num_layers 40, ffn_dim 13824, in/out 16 (the Wan2.1 VAE
    /// z_dim). Both experts share this config; they differ only in weights. The
    /// 14B is NOT a layer-count tweak of the 5B (different heads/ffn/channels +
    /// the Wan2.1 VAE instead of the TI2V high-compression one).
    pub fn wan22_14b() -> Self {
        Self {
            num_heads: 40,
            ffn_dim: 13824,
            num_layers: 40,
            in_channels: 16,
            out_channels: 16,
        }
    }

    /// Model dim `num_heads * HEAD_DIM`.
    pub fn inner(&self) -> usize {
        self.num_heads * config::HEAD_DIM
    }
}

// ---------------------------------------------------------------------------
// Pipelines: reuse the DiT BlockPipelines + the gelu-tanh op
// ---------------------------------------------------------------------------

/// Wan DiT reuses the DiT `BlockPipelines` (layernorm, rmsnorm, rope, sdpa, the
/// matmul sites, mul/add/bcast_add) and adds the one op the Z-Image block lacks:
/// the non-gated gelu-tanh FFN activation.
pub struct WanDitPipelines {
    pub block: BlockPipelines,
    pub gelu: WgpuPipeline,
}

impl WanDitPipelines {
    pub async fn compile(
        backend: &WgpuBackend,
        cfgs: &crate::common::block::BlockWgslConfigs,
    ) -> Result<Self, WgpuError> {
        let block = BlockPipelines::compile(backend, cfgs).await?;
        let gelu = backend
            .create_pipeline(
                "wan_gelu",
                <GeluF32 as thinfer_core::ops::Op>::wgsl(&cfgs.ops),
                "main",
                <GeluF32 as thinfer_core::ops::Op>::layout(),
            )
            .await?;
        Ok(Self { block, gelu })
    }
}

// ---------------------------------------------------------------------------
// Shape + per-block weight buffers
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct WanDitBlockShape {
    pub dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub inner: usize,
    pub ffn_dim: usize,
    /// Latent token count (post-patchify `f * pp_h * pp_w`), B=1.
    pub seq: usize,
    /// Text context length (umT5 output rows the cross-attention reads).
    pub text_seq: usize,
    pub norm_eps: f32,
}

impl WanDitBlockShape {
    pub fn new(cfg: &WanDitConfig, seq: usize, text_seq: usize) -> Self {
        Self {
            dim: cfg.inner(),
            n_heads: cfg.num_heads,
            head_dim: config::HEAD_DIM,
            inner: cfg.inner(),
            ffn_dim: cfg.ffn_dim,
            seq,
            text_seq,
            norm_eps: config::EPS,
        }
    }

    fn sdpa_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

/// One attention stage's projection + norm buffers (self or cross). q/k/v/o are
/// square `[inner, inner]` linears WITH bias; `norm_q`/`norm_k` are
/// RMSNorm-across-heads weights `[inner]`.
#[derive(Clone, Debug)]
pub struct WanAttnBufs {
    pub q_w: BufRef,
    pub q_b: BufRef,
    pub k_w: BufRef,
    pub k_b: BufRef,
    pub v_w: BufRef,
    pub v_b: BufRef,
    pub o_w: BufRef,
    pub o_b: BufRef,
    pub norm_q: BufRef,
    pub norm_k: BufRef,
}

/// All resident GPU buffers for one Wan DiT block.
#[derive(Clone, Debug)]
pub struct WanDitBlockBufs {
    pub self_attn: WanAttnBufs,
    pub cross_attn: WanAttnBufs,
    /// norm2 (cross_attn_norm) affine weight + bias `[inner]`.
    pub norm2_w: BufRef,
    pub norm2_b: BufRef,
    /// FFN `proj_in` (`net.0.proj`, `[ffn_dim, inner]`) + bias `[ffn_dim]`.
    pub ffn_up_w: BufRef,
    pub ffn_up_b: BufRef,
    /// FFN `proj_out` (`net.2`, `[inner, ffn_dim]`) + bias `[inner]`.
    pub ffn_down_w: BufRef,
    pub ffn_down_b: BufRef,
}

/// Residency handles mirroring [`WanAttnBufs`] (one self/cross attention
/// stage). q/k/v/o linears with bias + `norm_q`/`norm_k` gains.
#[derive(Clone, Debug)]
pub struct WanAttnHandles {
    pub q_w: WeightHandle,
    pub q_b: WeightHandle,
    pub k_w: WeightHandle,
    pub k_b: WeightHandle,
    pub v_w: WeightHandle,
    pub v_b: WeightHandle,
    pub o_w: WeightHandle,
    pub o_b: WeightHandle,
    pub norm_q: WeightHandle,
    pub norm_k: WeightHandle,
}

/// Residency handles for one Wan DiT block, mirroring [`WanDitBlockBufs`] plus
/// the per-block `scale_shift_table` `[6, inner]` the driver sums with
/// `timestep_proj` to produce the six modulation signals.
#[derive(Clone, Debug)]
pub struct WanDitBlockHandles {
    pub self_attn: WanAttnHandles,
    pub cross_attn: WanAttnHandles,
    pub norm2_w: WeightHandle,
    pub norm2_b: WeightHandle,
    pub ffn_up_w: WeightHandle,
    pub ffn_up_b: WeightHandle,
    pub ffn_down_w: WeightHandle,
    pub ffn_down_b: WeightHandle,
    pub scale_shift_table: WeightHandle,
}

pub struct WanAttnViews<'a> {
    q_w: GpuView<'a>,
    q_b: GpuView<'a>,
    k_w: GpuView<'a>,
    k_b: GpuView<'a>,
    v_w: GpuView<'a>,
    v_b: GpuView<'a>,
    o_w: GpuView<'a>,
    o_b: GpuView<'a>,
    norm_q: GpuView<'a>,
    norm_k: GpuView<'a>,
}

pub struct WanDitBlockViews<'a> {
    self_attn: WanAttnViews<'a>,
    cross_attn: WanAttnViews<'a>,
    norm2_w: GpuView<'a>,
    norm2_b: GpuView<'a>,
    ffn_up_w: GpuView<'a>,
    ffn_up_b: GpuView<'a>,
    ffn_down_w: GpuView<'a>,
    ffn_down_b: GpuView<'a>,
    scale_shift_table: GpuView<'a>,
}

impl WanAttnHandles {
    async fn acquire<'a, S: WeightSource>(
        &self,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<WanAttnViews<'a>, ResidencyError<S::Error, WgpuError>> {
        Ok(WanAttnViews {
            q_w: residency.acquire(self.q_w, backend).await?,
            q_b: residency.acquire(self.q_b, backend).await?,
            k_w: residency.acquire(self.k_w, backend).await?,
            k_b: residency.acquire(self.k_b, backend).await?,
            v_w: residency.acquire(self.v_w, backend).await?,
            v_b: residency.acquire(self.v_b, backend).await?,
            o_w: residency.acquire(self.o_w, backend).await?,
            o_b: residency.acquire(self.o_b, backend).await?,
            norm_q: residency.acquire(self.norm_q, backend).await?,
            norm_k: residency.acquire(self.norm_k, backend).await?,
        })
    }

    async fn prefetch<S: WeightSource>(
        &self,
        residency: &WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<(), ResidencyError<S::Error, WgpuError>> {
        for h in [
            self.q_w,
            self.q_b,
            self.k_w,
            self.k_b,
            self.v_w,
            self.v_b,
            self.o_w,
            self.o_b,
            self.norm_q,
            self.norm_k,
        ] {
            residency.prefetch(h, backend).await?;
        }
        Ok(())
    }
}

impl WanAttnViews<'_> {
    fn bufs(&self) -> WanAttnBufs {
        WanAttnBufs {
            q_w: self.q_w.buf(),
            q_b: self.q_b.buf(),
            k_w: self.k_w.buf(),
            k_b: self.k_b.buf(),
            v_w: self.v_w.buf(),
            v_b: self.v_b.buf(),
            o_w: self.o_w.buf(),
            o_b: self.o_b.buf(),
            norm_q: self.norm_q.buf(),
            norm_k: self.norm_k.buf(),
        }
    }
}

impl WanDitBlockHandles {
    pub async fn acquire<'a, S: WeightSource>(
        &self,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<WanDitBlockViews<'a>, ResidencyError<S::Error, WgpuError>> {
        Ok(WanDitBlockViews {
            self_attn: self.self_attn.acquire(residency, backend).await?,
            cross_attn: self.cross_attn.acquire(residency, backend).await?,
            norm2_w: residency.acquire(self.norm2_w, backend).await?,
            norm2_b: residency.acquire(self.norm2_b, backend).await?,
            ffn_up_w: residency.acquire(self.ffn_up_w, backend).await?,
            ffn_up_b: residency.acquire(self.ffn_up_b, backend).await?,
            ffn_down_w: residency.acquire(self.ffn_down_w, backend).await?,
            ffn_down_b: residency.acquire(self.ffn_down_b, backend).await?,
            scale_shift_table: residency.acquire(self.scale_shift_table, backend).await?,
        })
    }

    /// Stream every weight to GPU without pinning (overlap prefetch).
    pub async fn prefetch<S: WeightSource>(
        &self,
        residency: &WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<(), ResidencyError<S::Error, WgpuError>> {
        self.self_attn.prefetch(residency, backend).await?;
        self.cross_attn.prefetch(residency, backend).await?;
        for h in [
            self.norm2_w,
            self.norm2_b,
            self.ffn_up_w,
            self.ffn_up_b,
            self.ffn_down_w,
            self.ffn_down_b,
            self.scale_shift_table,
        ] {
            residency.prefetch(h, backend).await?;
        }
        Ok(())
    }
}

impl<'a> WanDitBlockViews<'a> {
    pub fn bufs(&self) -> WanDitBlockBufs {
        WanDitBlockBufs {
            self_attn: self.self_attn.bufs(),
            cross_attn: self.cross_attn.bufs(),
            norm2_w: self.norm2_w.buf(),
            norm2_b: self.norm2_b.buf(),
            ffn_up_w: self.ffn_up_w.buf(),
            ffn_up_b: self.ffn_up_b.buf(),
            ffn_down_w: self.ffn_down_w.buf(),
            ffn_down_b: self.ffn_down_b.buf(),
        }
    }

    /// The per-block `scale_shift_table` `[6, inner]` (driver builds modulation).
    pub fn scale_shift_table(&self) -> BufRef {
        self.scale_shift_table.buf()
    }
}

/// The six modulation signals, each a channel vector `[inner]`
/// (`scale_shift_table[k] + timestep_proj[k]`, already summed by the driver).
/// The distilled T2V line feeds one scalar timestep, so modulation is uniform
/// over all tokens: each signal broadcasts over the `seq` rows (no per-token
/// materialization, the SkyReels-V2-DF `[rows, inner]` hog is gone).
#[derive(Clone)]
pub struct WanMod<'wsp> {
    pub shift_msa: BatchBuf<'wsp>,
    pub scale_msa: BatchBuf<'wsp>,
    pub gate_msa: BatchBuf<'wsp>,
    pub c_shift_mlp: BatchBuf<'wsp>,
    pub c_scale_mlp: BatchBuf<'wsp>,
    pub c_gate_mlp: BatchBuf<'wsp>,
}

/// GPU tap destinations for one Wan DiT block (parity diagnostics).
#[derive(Default, Clone)]
pub struct WanDitBlockTaps {
    /// Pre-modulation self-attn LayerNorm output (`norm1(x)`, before
    /// `*(1+scale)+shift`). Separates a LayerNorm bug from a modulation bug.
    pub norm1_premod: Option<BufRef>,
    /// The self-attn modulation channel vectors `[inner]` (scale_msa, shift_msa).
    pub mod_scale: Option<BufRef>,
    pub mod_shift: Option<BufRef>,
    pub norm1: Option<BufRef>,
    pub self_q: Option<BufRef>,
    pub self_k: Option<BufRef>,
    pub self_v: Option<BufRef>,
    pub self_sa: Option<BufRef>,
    pub after_self: Option<BufRef>,
    pub norm2: Option<BufRef>,
    pub cross_sa: Option<BufRef>,
    pub after_cross: Option<BufRef>,
    pub norm3: Option<BufRef>,
    pub ffn_gelu: Option<BufRef>,
    pub ffn_down: Option<BufRef>,
}

// ---------------------------------------------------------------------------
// Local op helpers (the ones common::block keeps private). All compose the
// reusable BlockPipelines kernels; nothing Wan-specific except GeluF32.
// ---------------------------------------------------------------------------

/// `out = layernorm(x)` (mean-subtract, no affine). FP32 LayerNorm, eps folded.
fn op_layernorm<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    dst: ActBuf<'wsp>,
    rows: u32,
    dim: u32,
    eps: f32,
) -> Result<(), WgpuError> {
    let u = scope.u32x4_uniform(rows, dim, eps.to_bits(), 0)?;
    scope.layernorm::<LayerNormF32>(&bp.layernorm, x.data, u, dst.data, rows)
}

/// Channel-broadcast bias add `out[i] = x[i] + bias[i % dim]`. `bias` is a
/// resident weight view `[dim]`.
fn op_bias_add<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    bias: BatchBuf<'wsp>,
    dst: ActBuf<'wsp>,
    rows: u32,
    dim: u32,
) -> Result<(), WgpuError> {
    let u = scope.u32x4_uniform(dim, 0, 0, 0)?;
    scope.bcast_add::<BcastAddF32>(&bp.bcast_add, x.data, bias, u, dst.data, rows * dim)
}

/// Channel-broadcast modulation `out = x * (1 + scale) + shift`, where `scale`
/// and `shift` are `[dim]` vectors broadcast over the `rows` tokens (uniform
/// scalar-t). Both are runtime ACTIVATIONS (`scale_shift_table + timestep_proj`),
/// so the fused `bcast_modulate` op reads them in the act dtype (bias=1 folds the
/// `1 +`). NOT `bcast_add`, which reads its broadcast vector as a weight and
/// would reinterpret the f16 shift act as bf16, dropping it.
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

/// Channel-broadcast gated residual `out = x + gate * y`, `gate` a `[dim]`
/// vector broadcast over rows. Single `bcast_fma` dispatch; `out` must not alias
/// `x` or `y`.
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

/// One matmul site: `out = input @ wᵀ` through `dispatch_matmul_site` (handles
/// dense / dequant-once / DP4A per the site's compiled pipelines).
#[allow(clippy::too_many_arguments)]
fn lin<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    input: ActBuf<'wsp>,
    w: BatchBuf<'wsp>,
    out: BatchBuf<'wsp>,
    rows: u32,
    n: u32,
    k: u32,
    coopmat: Option<&crate::common::block::CoopmatStep>,
    i8: Option<&WgpuPipeline>,
    dq_i8: Option<&crate::common::block::DequantStep>,
    dq: Option<&crate::common::block::DequantStep>,
    pipe: &WgpuPipeline,
    inst: &MatMulF32,
) -> Result<(), WgpuError> {
    let dims = scope.u32x4_uniform(rows, n, k, 0)?;
    Block::dispatch_matmul_site_coopmat(
        scope, bp, input, w, out, dims, coopmat, i8, dq_i8, dq, pipe, inst, rows, n, k,
    )
}

// ---------------------------------------------------------------------------
// Block forward
// ---------------------------------------------------------------------------

/// Which QKV projection site a [`WanDitBlock::biased_proj`] call belongs to.
/// They are separate matmul pipelines so self-attn can run the DP4A i8 weight
/// while cross-attn (un-normed text K/V) stays dense; identical when i8 is off.
#[derive(Clone, Copy)]
enum QkvSite {
    SelfAttn,
    Cross,
}

pub struct WanDitBlock {
    pub shape: WanDitBlockShape,
}

impl WanDitBlock {
    pub fn new(shape: WanDitBlockShape) -> Self {
        assert_eq!(
            shape.n_heads * shape.head_dim,
            shape.inner,
            "inner must equal n_heads * head_dim"
        );
        Self { shape }
    }

    /// Append one block's dispatches to `scope`.
    /// - `x_in`: latent stream `[seq, inner]`.
    /// - `text`: umT5 text states `[text_seq, inner]` (already projected to the
    ///   DiT dim by the condition embedder).
    /// - `freqs`: RoPE3D table `[seq, head_dim]` (interleaved cos/sin) the caller
    ///   built for the latent grid.
    /// - `m`: the six per-token modulation signals.
    #[allow(clippy::too_many_arguments)]
    pub fn forward<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &WanDitPipelines,
        x_in: BatchBuf<'wsp>,
        text: BatchBuf<'wsp>,
        freqs: BatchBuf<'wsp>,
        m: &WanMod<'wsp>,
        y_out: BatchBuf<'wsp>,
        bufs: &WanDitBlockBufs,
        taps: &WanDitBlockTaps,
    ) -> Result<(), WgpuError> {
        let s = &self.shape;
        let bp = &pipelines.block;
        let rows = s.seq as u32;
        let trows = s.text_seq as u32;
        let dim = s.dim as u32;
        let eps = s.norm_eps;
        let scale = s.sdpa_scale();
        let x_in = ActBuf::dense(x_in);
        let y_out = ActBuf::dense(y_out);

        // ============== 1. self-attention ==============
        let n1 = alloc_act(scope, bp, rows, dim)?;
        op_layernorm(scope, bp, x_in, n1, rows, dim, eps)?;
        copy_tap(
            scope,
            n1.data,
            taps.norm1_premod.as_ref(),
            bp.act_bytes(rows * dim),
        )?;
        copy_tap(
            scope,
            m.scale_msa,
            taps.mod_scale.as_ref(),
            bp.act_bytes(dim),
        )?;
        copy_tap(
            scope,
            m.shift_msa,
            taps.mod_shift.as_ref(),
            bp.act_bytes(dim),
        )?;
        let n1m = alloc_act(scope, bp, rows, dim)?;
        op_modulate(scope, bp, n1, m.scale_msa, m.shift_msa, n1m, rows, dim)?;
        copy_tap(
            scope,
            n1m.data,
            taps.norm1.as_ref(),
            bp.act_bytes(rows * dim),
        )?;

        let sa = self.attention(
            scope,
            pipelines,
            n1m,
            n1m,
            Some(freqs),
            &bufs.self_attn,
            rows,
            rows,
            scale,
            QkvSite::SelfAttn,
            taps.self_q.as_ref(),
            taps.self_k.as_ref(),
            taps.self_v.as_ref(),
            taps.self_sa.as_ref(),
        )?;

        let x1 = alloc_act(scope, bp, rows, dim)?;
        op_gate_residual(scope, bp, x_in, m.gate_msa, sa, x1, rows, dim)?;
        copy_tap(
            scope,
            x1.data,
            taps.after_self.as_ref(),
            bp.act_bytes(rows * dim),
        )?;

        // ============== 2. cross-attention + 3. FFN (shared tail) ==============
        self.cross_ffn(
            scope, pipelines, x1, text, m, y_out, bufs, taps, rows, trows,
        )
    }

    /// Cross-attention to the umT5 text states + the gelu-tanh FFN, the shared
    /// tail after self-attention. `x1` is the residual stream after the self-attn
    /// gated residual; `y_out` receives the block output. Factored out of
    /// [`Self::forward`] so the AR path ([`Self::self_attn_ar`] + this) reuses the
    /// exact same op sequence (FastWan stays numerically byte-identical: the ops
    /// and their order are verbatim, only the self-attention differs in AR).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cross_ffn<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &WanDitPipelines,
        x1: ActBuf<'wsp>,
        text: BatchBuf<'wsp>,
        m: &WanMod<'wsp>,
        y_out: ActBuf<'wsp>,
        bufs: &WanDitBlockBufs,
        taps: &WanDitBlockTaps,
        rows: u32,
        trows: u32,
    ) -> Result<(), WgpuError> {
        let s = &self.shape;
        let bp = &pipelines.block;
        let dim = s.dim as u32;
        let inner = s.inner as u32;
        let dff = s.ffn_dim as u32;
        let eps = s.norm_eps;
        let scale = s.sdpa_scale();

        // ============== 2. cross-attention (to text) ==============
        // norm2 = affine LayerNorm: ln -> *weight -> +bias (channel-broadcast).
        let n2 = alloc_act(scope, bp, rows, dim)?;
        op_layernorm(scope, bp, x1, n2, rows, dim, eps)?;
        let n2w = alloc_act(scope, bp, rows, dim)?;
        let w2 = scope.import_copy(bufs.norm2_w);
        // norm2 weight is a stored bf16 weight, not an act: use `bcast_mul`
        // (decodes by weight_dtype) NOT `bcast_affine` (reads its scale as an
        // f16 act, which reinterprets the bf16 weight bits -> wrong scale).
        let u_mul = scope.u32x4_uniform(dim, 0, 0, 0)?;
        scope.bcast_add::<BcastMulF32>(&bp.bcast_mul, n2.data, w2, u_mul, n2w.data, rows * dim)?;
        let b2 = scope.import_copy(bufs.norm2_b);
        let n2wb = alloc_act(scope, bp, rows, dim)?;
        op_bias_add(scope, bp, n2w, b2, n2wb, rows, dim)?;
        copy_tap(
            scope,
            n2wb.data,
            taps.norm2.as_ref(),
            bp.act_bytes(rows * dim),
        )?;

        let ca = self.attention(
            scope,
            pipelines,
            n2wb,
            ActBuf::dense(text),
            None,
            &bufs.cross_attn,
            rows,
            trows,
            scale,
            QkvSite::Cross,
            None,
            None,
            None,
            taps.cross_sa.as_ref(),
        )?;
        // residual, no gate
        let x2 = alloc_act(scope, bp, rows, dim)?;
        op_add(scope, bp, x1, ca, x2)?;
        copy_tap(
            scope,
            x2.data,
            taps.after_cross.as_ref(),
            bp.act_bytes(rows * dim),
        )?;

        // ============== 3. feed-forward (gelu-tanh, non-gated) ==============
        // Tiled over query rows. The FFN is purely position-wise (every op acts
        // per row), so processing the rows in chunks is BIT-EXACT and bounds the
        // peak transient working set to `FFN_TILE_ROWS * ffn_dim` instead of
        // `rows * ffn_dim`. Without this, a long clip (e.g. 81f @ 832x480 = 32760
        // rows) reserves multiple GB of FFN scratch, which on the 8GB card starves
        // the 14B weight cache and the whole DiT thrashes on weight re-streaming.
        // The weights are imported once; the per-chunk matmul re-runs the Q8
        // dequant pre-pass, a negligible cost paid only for multi-chunk clips.
        // Taps force a single whole-rows chunk so the parity copy stays correct
        // (they are None in production, where chunking is active).
        let ffn_tile = if taps.norm3.is_some() || taps.ffn_gelu.is_some() || taps.ffn_down.is_some()
        {
            rows
        } else {
            FFN_TILE_ROWS.min(rows).max(1)
        };
        let wi = scope.import_copy(bufs.ffn_up_w);
        let bi = scope.import_copy(bufs.ffn_up_b);
        let wo = scope.import_copy(bufs.ffn_down_w);
        let bo = scope.import_copy(bufs.ffn_down_b);
        let mut r0 = 0u32;
        while r0 < rows {
            let rc = (rows - r0).min(ffn_tile);
            let x2_c = ActBuf::dense(scope.subview(
                &x2.data,
                bp.act_bytes(r0 * dim),
                bp.act_bytes(rc * dim),
            ));
            let y_c = ActBuf::dense(scope.subview(
                &y_out.data,
                bp.act_bytes(r0 * dim),
                bp.act_bytes(rc * dim),
            ));

            let n3 = alloc_act(scope, bp, rc, dim)?;
            op_layernorm(scope, bp, x2_c, n3, rc, dim, eps)?;
            let n3m = alloc_act(scope, bp, rc, dim)?;
            op_modulate(scope, bp, n3, m.c_scale_mlp, m.c_shift_mlp, n3m, rc, dim)?;
            copy_tap(scope, n3m.data, taps.norm3.as_ref(), bp.act_bytes(rc * dim))?;

            // up: [rc, inner] @ [ffn_dim, inner]ᵀ + bias -> [rc, ffn_dim]
            let up = alloc_matmul_out_buf(scope, bp, rc * dff)?;
            lin(
                scope,
                bp,
                n3m,
                wi,
                up,
                rc,
                dff,
                inner,
                None,
                bp.matmul_i8_ffn_up.as_ref(),
                bp.dequant_i8_ffn_up.as_ref(),
                bp.dequant_ffn_up.as_ref(),
                &bp.matmul_ffn_up,
                &bp.matmuls.ffn_up,
            )?;
            let upb = alloc_act(scope, bp, rc, dff)?;
            op_bias_add(scope, bp, ActBuf::dense(up), bi, upb, rc, dff)?;
            // gelu-tanh
            let gelu = alloc_act(scope, bp, rc, dff)?;
            scope.dispatch_op::<GeluF32>(&pipelines.gelu, &[upb.data], gelu.data)?;
            copy_tap(
                scope,
                gelu.data,
                taps.ffn_gelu.as_ref(),
                bp.act_bytes(rc * dff),
            )?;
            // down: [rc, ffn_dim] @ [inner, ffn_dim]ᵀ + bias -> [rc, inner]
            let down = alloc_matmul_out_buf(scope, bp, rc * dim)?;
            lin(
                scope,
                bp,
                gelu,
                wo,
                down,
                rc,
                dim,
                dff,
                bp.coopmat_ffn_down.as_ref(),
                bp.matmul_i8_ffn_down.as_ref(),
                bp.dequant_i8_ffn_down.as_ref(),
                bp.dequant_ffn_down.as_ref(),
                &bp.matmul_ffn_down,
                &bp.matmuls.ffn_down,
            )?;
            let downb = alloc_act(scope, bp, rc, dim)?;
            op_bias_add(scope, bp, ActBuf::dense(down), bo, downb, rc, dim)?;
            copy_tap(
                scope,
                downb.data,
                taps.ffn_down.as_ref(),
                bp.act_bytes(rc * dim),
            )?;

            op_gate_residual(scope, bp, x2_c, m.c_gate_mlp, downb, y_c, rc, dim)?;
            r0 += rc;
        }
        Ok(())
    }

    /// AR (LongLive) self-attention for one chunk against a windowed KV cache.
    /// Mirrors the `use_relative_rope=False` release path of
    /// `CausalWanSelfAttention.forward`: q and the chunk's k are RoPE'd at the
    /// chunk's ABSOLUTE frame position (carried in `chunk_freqs`), and the query
    /// attends over `[prefix (committed, already-RoPE'd) ++ this chunk's k]` with
    /// NO materialized mask (causality is the window contents). Produces the
    /// post-self-attn residual `x1_out` and exports the chunk's roped-k / v
    /// (`roped_k_out` / `v_out`) so the clean pass can commit them to the cache.
    ///
    /// `prefix_k` / `prefix_v` are the `prefix_rows` committed window tokens
    /// uploaded from the host store (K already roped, V raw); `window_k` /
    /// `window_v` are `window_rows = prefix_rows + chunk_rows` scratch buffers the
    /// driver pre-allocates. When `prefix_rows == 0` (first chunk) the window is
    /// just the chunk itself.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn self_attn_ar<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &WanDitPipelines,
        x_in: BatchBuf<'wsp>,
        chunk_freqs: BatchBuf<'wsp>,
        m: &WanMod<'wsp>,
        prefix_k: BatchBuf<'wsp>,
        prefix_v: BatchBuf<'wsp>,
        prefix_rows: u32,
        window_k: BatchBuf<'wsp>,
        window_v: BatchBuf<'wsp>,
        roped_k_out: BatchBuf<'wsp>,
        v_out: BatchBuf<'wsp>,
        x1_out: BatchBuf<'wsp>,
        chunk_rows: u32,
        window_rows: u32,
        w: &WanAttnBufs,
    ) -> Result<(), WgpuError> {
        let s = &self.shape;
        let bp = &pipelines.block;
        let dim = s.dim as u32;
        let inner = s.inner as u32;
        let eps = s.norm_eps;
        let nh = s.n_heads as u32;
        let hd = s.head_dim as u32;
        let scale = s.sdpa_scale();
        let x_in = ActBuf::dense(x_in);

        // norm1 (no affine) -> modulate by (scale_msa, shift_msa).
        let n1 = alloc_act(scope, bp, chunk_rows, dim)?;
        op_layernorm(scope, bp, x_in, n1, chunk_rows, dim, eps)?;
        let n1m = alloc_act(scope, bp, chunk_rows, dim)?;
        op_modulate(
            scope,
            bp,
            n1,
            m.scale_msa,
            m.shift_msa,
            n1m,
            chunk_rows,
            dim,
        )?;

        // q/k/v projections + bias, qk-norm (RMSNorm across the full inner dim).
        let q = self.biased_proj(
            scope,
            bp,
            n1m,
            w.q_w,
            w.q_b,
            chunk_rows,
            inner,
            QkvSite::SelfAttn,
        )?;
        let k = self.biased_proj(
            scope,
            bp,
            n1m,
            w.k_w,
            w.k_b,
            chunk_rows,
            inner,
            QkvSite::SelfAttn,
        )?;
        let vv = self.biased_proj(
            scope,
            bp,
            n1m,
            w.v_w,
            w.v_b,
            chunk_rows,
            inner,
            QkvSite::SelfAttn,
        )?;
        let qn = alloc_act(scope, bp, chunk_rows, inner)?;
        let nq = scope.import_copy(w.norm_q);
        op_rmsnorm(scope, bp, q, nq, qn, chunk_rows, inner, eps)?;
        let kn = alloc_act(scope, bp, chunk_rows, inner)?;
        let nk = scope.import_copy(w.norm_k);
        op_rmsnorm(scope, bp, k, nk, kn, chunk_rows, inner, eps)?;

        // RoPE3D q -> qr; k -> roped_k_out (also the cache-commit export). V is
        // never roped; copy straight into v_out.
        let qr = alloc_act(scope, bp, chunk_rows, inner)?;
        self.rope(scope, bp, qn, chunk_freqs, qr, chunk_rows)?;
        self.rope(
            scope,
            bp,
            kn,
            chunk_freqs,
            ActBuf::dense(roped_k_out),
            chunk_rows,
        )?;
        scope.copy_buffer_to_buffer(vv.data, 0, v_out, 0, bp.act_bytes(chunk_rows * inner))?;

        // Assemble the attended window K/V = [prefix ++ this chunk]. With no
        // prefix (first chunk) window == chunk, so the SDPA is bidirectional over
        // the chunk (matches the upstream block-causal mask within one chunk).
        let prefix_bytes = bp.act_bytes(prefix_rows * inner);
        let chunk_bytes = bp.act_bytes(chunk_rows * inner);
        if prefix_rows > 0 {
            scope.copy_buffer_to_buffer(prefix_k, 0, window_k, 0, prefix_bytes)?;
            scope.copy_buffer_to_buffer(prefix_v, 0, window_v, 0, prefix_bytes)?;
        }
        scope.copy_buffer_to_buffer(roped_k_out, 0, window_k, prefix_bytes, chunk_bytes)?;
        scope.copy_buffer_to_buffer(v_out, 0, window_v, prefix_bytes, chunk_bytes)?;

        // SDPA: q = chunk rows, kv = the window. Mode 0 (no mask): the window
        // only ever holds tokens at or before the chunk, so causality is the
        // cache contents, not a materialized mask.
        let sa = alloc_act(scope, bp, chunk_rows, inner)?;
        let no_mask = scope.alloc(16)?;
        op_sdpa_f16(
            scope,
            bp,
            qr,
            ActBuf::dense(window_k),
            ActBuf::dense(window_v),
            no_mask,
            sa,
            1,
            chunk_rows,
            window_rows,
            nh,
            nh,
            hd,
            scale,
            0,
        )?;

        // output projection + bias, then the gated residual into x1_out.
        let sa_proj = self.attn_out_proj(scope, bp, sa, w, chunk_rows, inner)?;
        op_gate_residual(
            scope,
            bp,
            x_in,
            m.gate_msa,
            sa_proj,
            ActBuf::dense(x1_out),
            chunk_rows,
            dim,
        )?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Activation-tiled block (large-token path). The block's only all-to-all
    // op is the self-attention SDPA; everything else is per-token-row. So the
    // driver runs the block in three movements, each in its own scope(s) so
    // the workspace pool recycles tile transients between submits:
    //
    //   pass A (row-tiled): norm1 -> modulate -> q/k/v proj+bias -> qk-norm
    //                       -> rope, writing the full `qx`/`kx`/`v` buffers.
    //   barrier:            cross-attn K/V projected once + GLOBAL self-SDPA.
    //   pass B (row-tiled): self o-proj -> gate residual -> cross-attn ->
    //                       FFN, writing the residual stream out.
    //
    // The math is identical to `forward`; only the buffer lifetime changes
    // (the heavy `[tile, ffn_dim]` FFN transients no longer all live at once).
    // -----------------------------------------------------------------------

    /// Pass A for one row-tile: produce the rotated `qx`/`kx` and `v` slices
    /// for `tr` query rows. `x_in`/`freqs`/`qx`/`kx`/`v` are the tile slices
    /// (`tr` rows) of the full-sequence buffers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn self_qkv_tile<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &WanDitPipelines,
        x_in: BatchBuf<'wsp>,
        freqs: BatchBuf<'wsp>,
        m: &WanMod<'wsp>,
        qx: BatchBuf<'wsp>,
        kx: BatchBuf<'wsp>,
        v: BatchBuf<'wsp>,
        tr: u32,
        w: &WanAttnBufs,
    ) -> Result<(), WgpuError> {
        let s = &self.shape;
        let bp = &pipelines.block;
        let dim = s.dim as u32;
        let inner = s.inner as u32;
        let eps = s.norm_eps;
        let x_in = ActBuf::dense(x_in);

        // norm1 (no affine) -> modulate by (scale_msa, shift_msa).
        let n1 = alloc_act(scope, bp, tr, dim)?;
        op_layernorm(scope, bp, x_in, n1, tr, dim, eps)?;
        let n1m = alloc_act(scope, bp, tr, dim)?;
        op_modulate(scope, bp, n1, m.scale_msa, m.shift_msa, n1m, tr, dim)?;

        // q/k/v projections + bias, qk-norm (RMSNorm across the full inner dim).
        let q = self.biased_proj(scope, bp, n1m, w.q_w, w.q_b, tr, inner, QkvSite::SelfAttn)?;
        let k = self.biased_proj(scope, bp, n1m, w.k_w, w.k_b, tr, inner, QkvSite::SelfAttn)?;
        let vv = self.biased_proj(scope, bp, n1m, w.v_w, w.v_b, tr, inner, QkvSite::SelfAttn)?;
        let qn = alloc_act(scope, bp, tr, inner)?;
        let nq = scope.import_copy(w.norm_q);
        op_rmsnorm(scope, bp, q, nq, qn, tr, inner, eps)?;
        let kn = alloc_act(scope, bp, tr, inner)?;
        let nk = scope.import_copy(w.norm_k);
        op_rmsnorm(scope, bp, k, nk, kn, tr, inner, eps)?;

        // RoPE3D into the persistent qx/kx slices; v carried unrotated.
        self.rope(scope, bp, qn, freqs, ActBuf::dense(qx), tr)?;
        self.rope(scope, bp, kn, freqs, ActBuf::dense(kx), tr)?;
        scope.copy_buffer_to_buffer(vv.data, 0, v, 0, bp.act_bytes(tr * inner))?;
        Ok(())
    }

    /// Project the umT5 text states to the cross-attention K/V once (shared by
    /// every query tile in pass B). `ck` is `norm_k(to_k(text)+b_k)`; `cv` is
    /// `to_v(text)+b_v`. No RoPE (cross-attn is unrotated).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cross_kv<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &WanDitPipelines,
        text: BatchBuf<'wsp>,
        w: &WanAttnBufs,
        ck: BatchBuf<'wsp>,
        cv: BatchBuf<'wsp>,
        trows: u32,
    ) -> Result<(), WgpuError> {
        let s = &self.shape;
        let bp = &pipelines.block;
        let inner = s.inner as u32;
        let eps = s.norm_eps;
        let text = ActBuf::dense(text);
        let k = self.biased_proj(scope, bp, text, w.k_w, w.k_b, trows, inner, QkvSite::Cross)?;
        let vv = self.biased_proj(scope, bp, text, w.v_w, w.v_b, trows, inner, QkvSite::Cross)?;
        let nk = scope.import_copy(w.norm_k);
        op_rmsnorm(scope, bp, k, nk, ActBuf::dense(ck), trows, inner, eps)?;
        scope.copy_buffer_to_buffer(vv.data, 0, cv, 0, bp.act_bytes(trows * inner))?;
        Ok(())
    }

    /// Pass B for one row-tile: consume the self-attention output `sa` slice
    /// (`tr` rows) and produce the block's residual-stream output slice.
    /// `ck`/`cv` are the full precomputed cross-attn K/V (`trows` text rows).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn post_attn_tile<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &WanDitPipelines,
        x_in: BatchBuf<'wsp>,
        sa: BatchBuf<'wsp>,
        m: &WanMod<'wsp>,
        ck: BatchBuf<'wsp>,
        cv: BatchBuf<'wsp>,
        y_out: BatchBuf<'wsp>,
        tr: u32,
        trows: u32,
        bufs: &WanDitBlockBufs,
    ) -> Result<(), WgpuError> {
        let s = &self.shape;
        let bp = &pipelines.block;
        let dim = s.dim as u32;
        let inner = s.inner as u32;
        let dff = s.ffn_dim as u32;
        let eps = s.norm_eps;
        let nh = s.n_heads as u32;
        let hd = s.head_dim as u32;
        let scale = s.sdpa_scale();
        let x_in = ActBuf::dense(x_in);

        // self-attn output projection + bias, then gated residual.
        let sa_proj =
            self.attn_out_proj(scope, bp, ActBuf::dense(sa), &bufs.self_attn, tr, inner)?;
        let x1 = alloc_act(scope, bp, tr, dim)?;
        op_gate_residual(scope, bp, x_in, m.gate_msa, sa_proj, x1, tr, dim)?;

        // cross-attention: norm2 (affine) -> q proj -> SDPA against ck/cv.
        let n2 = alloc_act(scope, bp, tr, dim)?;
        op_layernorm(scope, bp, x1, n2, tr, dim, eps)?;
        let n2w = alloc_act(scope, bp, tr, dim)?;
        let w2 = scope.import_copy(bufs.norm2_w);
        let u_mul = scope.u32x4_uniform(dim, 0, 0, 0)?;
        scope.bcast_add::<BcastMulF32>(&bp.bcast_mul, n2.data, w2, u_mul, n2w.data, tr * dim)?;
        let b2 = scope.import_copy(bufs.norm2_b);
        let n2wb = alloc_act(scope, bp, tr, dim)?;
        op_bias_add(scope, bp, n2w, b2, n2wb, tr, dim)?;

        let cq_b = self.biased_proj(
            scope,
            bp,
            n2wb,
            bufs.cross_attn.q_w,
            bufs.cross_attn.q_b,
            tr,
            inner,
            QkvSite::Cross,
        )?;
        let cq = alloc_act(scope, bp, tr, inner)?;
        let ncq = scope.import_copy(bufs.cross_attn.norm_q);
        op_rmsnorm(scope, bp, cq_b, ncq, cq, tr, inner, eps)?;
        let ca = alloc_act(scope, bp, tr, inner)?;
        let no_mask = scope.alloc(16)?;
        op_sdpa(
            scope,
            bp,
            cq,
            ActBuf::dense(ck),
            ActBuf::dense(cv),
            no_mask,
            ca,
            1,
            tr,
            trows,
            nh,
            nh,
            hd,
            scale,
            0,
        )?;
        let ca_proj = self.attn_out_proj(scope, bp, ca, &bufs.cross_attn, tr, inner)?;
        // cross-attn residual: no gate.
        let x2 = alloc_act(scope, bp, tr, dim)?;
        op_add(scope, bp, x1, ca_proj, x2)?;

        // feed-forward (gelu-tanh, non-gated), gated residual into y_out.
        let n3 = alloc_act(scope, bp, tr, dim)?;
        op_layernorm(scope, bp, x2, n3, tr, dim, eps)?;
        let n3m = alloc_act(scope, bp, tr, dim)?;
        op_modulate(scope, bp, n3, m.c_scale_mlp, m.c_shift_mlp, n3m, tr, dim)?;
        let up = alloc_matmul_out_buf(scope, bp, tr * dff)?;
        let wi = scope.import_copy(bufs.ffn_up_w);
        lin(
            scope,
            bp,
            n3m,
            wi,
            up,
            tr,
            dff,
            inner,
            None,
            bp.matmul_i8_ffn_up.as_ref(),
            bp.dequant_i8_ffn_up.as_ref(),
            bp.dequant_ffn_up.as_ref(),
            &bp.matmul_ffn_up,
            &bp.matmuls.ffn_up,
        )?;
        let bi = scope.import_copy(bufs.ffn_up_b);
        let upb = alloc_act(scope, bp, tr, dff)?;
        op_bias_add(scope, bp, ActBuf::dense(up), bi, upb, tr, dff)?;
        let gelu = alloc_act(scope, bp, tr, dff)?;
        scope.dispatch_op::<GeluF32>(&pipelines.gelu, &[upb.data], gelu.data)?;
        let down = alloc_matmul_out_buf(scope, bp, tr * dim)?;
        let wo = scope.import_copy(bufs.ffn_down_w);
        lin(
            scope,
            bp,
            gelu,
            wo,
            down,
            tr,
            dim,
            dff,
            bp.coopmat_ffn_down.as_ref(),
            bp.matmul_i8_ffn_down.as_ref(),
            bp.dequant_i8_ffn_down.as_ref(),
            bp.dequant_ffn_down.as_ref(),
            &bp.matmul_ffn_down,
            &bp.matmuls.ffn_down,
        )?;
        let bo = scope.import_copy(bufs.ffn_down_b);
        let downb = alloc_act(scope, bp, tr, dim)?;
        op_bias_add(scope, bp, ActBuf::dense(down), bo, downb, tr, dim)?;
        op_gate_residual(
            scope,
            bp,
            x2,
            m.c_gate_mlp,
            downb,
            ActBuf::dense(y_out),
            tr,
            dim,
        )?;
        Ok(())
    }

    /// Global self-attention barrier for the tiled path: `sa = softmax(qx kxᵀ)
    /// v` over the full token sequence (the one all-to-all op in the block).
    #[allow(clippy::too_many_arguments)]
    /// Global self-attention over the whole token sequence. When `window > 0`
    /// the attention is restricted to a temporal sliding window: each query
    /// attends only to keys within `±window` latent frames, where `period` is
    /// the token count per latent frame (frame-major `(f, h, w)` layout). This
    /// breaks the O(frames^2) cost at long clips; it changes the output, so the
    /// caller gates it on the run's `attn_window` flag.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn self_sdpa<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &WanDitPipelines,
        qx: BatchBuf<'wsp>,
        kx: BatchBuf<'wsp>,
        v: BatchBuf<'wsp>,
        sa: BatchBuf<'wsp>,
        rows: u32,
        period: u32,
        window: u32,
    ) -> Result<(), WgpuError> {
        let s = &self.shape;
        let bp = &pipelines.block;
        let nh = s.n_heads as u32;
        let hd = s.head_dim as u32;
        let scale = s.sdpa_scale();
        let no_mask = scope.alloc(16)?;
        op_sdpa_f16_win(
            scope,
            bp,
            ActBuf::dense(qx),
            ActBuf::dense(kx),
            ActBuf::dense(v),
            no_mask,
            ActBuf::dense(sa),
            1,
            rows,
            rows,
            nh,
            nh,
            hd,
            scale,
            0,
            period,
            window,
            // Pure-video self-attention: no text tokens, so `txt_start = s_k`
            // leaves the joint-windowing branches dead (bit-identical windowing).
            rows,
        )
    }

    /// `out = bias + proj(x)` through the attention output matmul site. Shared
    /// by the self/cross output projections in the tiled path (the dense tail
    /// of [`Self::attention`], factored so pass B can reuse it).
    fn attn_out_proj<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        bp: &BlockPipelines,
        sa: ActBuf<'wsp>,
        w: &WanAttnBufs,
        rows: u32,
        inner: u32,
    ) -> Result<ActBuf<'wsp>, WgpuError> {
        let proj = alloc_matmul_out_buf(scope, bp, rows * inner)?;
        let ow = scope.import_copy(w.o_w);
        lin(
            scope,
            bp,
            sa,
            ow,
            proj,
            rows,
            inner,
            inner,
            bp.coopmat_proj.as_ref(),
            bp.matmul_i8_proj.as_ref(),
            bp.dequant_i8_proj.as_ref(),
            bp.dequant_proj.as_ref(),
            &bp.matmul_proj,
            &bp.matmuls.proj,
        )?;
        let ob = scope.import_copy(w.o_b);
        let projb = alloc_act(scope, bp, rows, inner)?;
        op_bias_add(scope, bp, ActBuf::dense(proj), ob, projb, rows, inner)?;
        Ok(projb)
    }

    /// Shared self/cross attention. `q_src` provides the queries `[q_rows,
    /// inner]`; `kv_src` provides keys/values `[kv_rows, inner]` (== `q_src` for
    /// self-attention). `freqs` is `Some` only for self-attention (RoPE3D); cross
    /// attention runs no positional rotation and no mask.
    #[allow(clippy::too_many_arguments)]
    fn attention<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &WanDitPipelines,
        q_src: ActBuf<'wsp>,
        kv_src: ActBuf<'wsp>,
        freqs: Option<BatchBuf<'wsp>>,
        w: &WanAttnBufs,
        q_rows: u32,
        kv_rows: u32,
        scale: f32,
        site: QkvSite,
        tap_q: Option<&BufRef>,
        tap_k: Option<&BufRef>,
        tap_v: Option<&BufRef>,
        tap_sa: Option<&BufRef>,
    ) -> Result<ActBuf<'wsp>, WgpuError> {
        let s = &self.shape;
        let bp = &pipelines.block;
        let inner = s.inner as u32;
        let hd = s.head_dim as u32;
        let nh = s.n_heads as u32;
        let eps = s.norm_eps;

        // q = norm_q(to_q(q_src) + b_q); k = norm_k(to_k(kv_src) + b_k);
        // v = to_v(kv_src) + b_v. qk-norm is RMSNorm over the FULL inner dim
        // (`rms_norm_across_heads`), applied before the head split.
        let q = self.biased_proj(scope, bp, q_src, w.q_w, w.q_b, q_rows, inner, site)?;
        let k = self.biased_proj(scope, bp, kv_src, w.k_w, w.k_b, kv_rows, inner, site)?;
        let v = self.biased_proj(scope, bp, kv_src, w.v_w, w.v_b, kv_rows, inner, site)?;
        copy_tap(scope, q.data, tap_q, bp.act_bytes(q_rows * inner))?;
        copy_tap(scope, k.data, tap_k, bp.act_bytes(kv_rows * inner))?;
        copy_tap(scope, v.data, tap_v, bp.act_bytes(kv_rows * inner))?;

        let qn = alloc_act(scope, bp, q_rows, inner)?;
        let nq = scope.import_copy(w.norm_q);
        op_rmsnorm(scope, bp, q, nq, qn, q_rows, inner, eps)?;
        let kn = alloc_act(scope, bp, kv_rows, inner)?;
        let nk = scope.import_copy(w.norm_k);
        op_rmsnorm(scope, bp, k, nk, kn, kv_rows, inner, eps)?;

        // RoPE3D (interleaved-pair) on q/k, self-attention only.
        let (qx, kx) = match freqs {
            Some(f) => {
                let qr = alloc_act(scope, bp, q_rows, inner)?;
                let kr = alloc_act(scope, bp, kv_rows, inner)?;
                self.rope(scope, bp, qn, f, qr, q_rows)?;
                self.rope(scope, bp, kn, f, kr, kv_rows)?;
                (qr, kr)
            }
            None => (qn, kn),
        };

        let sa = alloc_act(scope, bp, q_rows, inner)?;
        // No mask (mode 0): self-attn is bidirectional for num_frame_per_block==1
        // (DF base), cross-attn is unmasked by construction. The kernel still
        // binds the mask storage slot but never reads it (read gated on
        // has_mask != 0), so a tiny dummy storage buffer suffices.
        let no_mask = scope.alloc(16)?;
        op_sdpa(
            scope, bp, qx, kx, v, no_mask, sa, 1, q_rows, kv_rows, nh, nh, hd, scale, 0,
        )?;
        copy_tap(scope, sa.data, tap_sa, bp.act_bytes(q_rows * inner))?;

        // output projection + bias
        self.attn_out_proj(scope, bp, sa, w, q_rows, inner)
    }

    /// `out = x @ wᵀ + bias` through the qkv matmul site (`site` selects the
    /// self- vs cross-attention pipeline; see [`QkvSite`]).
    #[allow(clippy::too_many_arguments)]
    fn biased_proj<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        bp: &BlockPipelines,
        x: ActBuf<'wsp>,
        w: BufRef,
        b: BufRef,
        rows: u32,
        n: u32,
        site: QkvSite,
    ) -> Result<ActBuf<'wsp>, WgpuError> {
        let dim = self.shape.dim as u32;
        let out = alloc_matmul_out_buf(scope, bp, rows * n)?;
        let wv = scope.import_copy(w);
        // Self-attn qkv may be i8 (normed A-side); cross-attn qkv stays dense
        // (un-normed umT5 text K/V), or runs coopmat when opted in.
        let (coopmat, mm_i8, dq_i8, dq, pipe, inst) = match site {
            QkvSite::SelfAttn => (
                None,
                bp.matmul_i8_qkv_self.as_ref(),
                bp.dequant_i8_qkv_self.as_ref(),
                bp.dequant_qkv_self.as_ref(),
                &bp.matmul_qkv_self,
                &bp.matmuls.qkv_self,
            ),
            QkvSite::Cross => (
                bp.coopmat_qkv.as_ref(),
                bp.matmul_i8_qkv.as_ref(),
                bp.dequant_i8_qkv.as_ref(),
                bp.dequant_qkv.as_ref(),
                &bp.matmul_qkv,
                &bp.matmuls.qkv,
            ),
        };
        lin(
            scope, bp, x, wv, out, rows, n, dim, coopmat, mm_i8, dq_i8, dq, pipe, inst,
        )?;
        let bv = scope.import_copy(b);
        let biased = alloc_act(scope, bp, rows, n)?;
        op_bias_add(scope, bp, ActBuf::dense(out), bv, biased, rows, n)?;
        Ok(biased)
    }

    /// RoPE3D, interleaved-pair (`RopeF32`), per-head over `inner = n_heads *
    /// head_dim`.
    fn rope<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        bp: &BlockPipelines,
        src: ActBuf<'wsp>,
        freqs: BatchBuf<'wsp>,
        dst: ActBuf<'wsp>,
        rows: u32,
    ) -> Result<(), WgpuError> {
        let nh = self.shape.n_heads as u32;
        let hd = self.shape.head_dim as u32;
        let pairs = hd / 2;
        let u = scope.u32x4_uniform(rows, nh, pairs, 0)?;
        scope.rope::<RopeF32>(&bp.rope, src.data, freqs, u, dst.data, rows, nh, pairs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inner_matches_heads() {
        let cfg = WanDitConfig::fastwan_ti2v_5b();
        assert_eq!(cfg.inner(), cfg.num_heads * config::HEAD_DIM);
        assert_eq!(cfg.inner(), 3072);
    }

    #[test]
    fn shape_builds() {
        let sh = WanDitBlockShape::new(&WanDitConfig::fastwan_ti2v_5b(), 1024, config::TEXT_SEQ);
        assert_eq!(sh.inner, 3072);
        assert!((sh.sdpa_scale() - 1.0 / (128f32).sqrt()).abs() < 1e-9);
    }
}
