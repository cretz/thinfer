//! Wan / SkyReels-V2 DiT transformer block forward (`SkyReelsV2TransformerBlock`,
//! `transformer_skyreels_v2.py`). Full-attention video DiT block:
//!
//! ```text
//! shift_msa,scale_msa,gate_msa,c_shift,c_scale,c_gate = mod   // per-token (DF)
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
//! - Diffusion-Forcing modulation is PER-TOKEN: scale/shift/gate are full
//!   `[rows, inner]` tensors (one timestep per latent frame, broadcast over the
//!   frame's spatial tokens by the driver), so modulation is full-elementwise
//!   `Mul`/`Add`, not the channel-broadcast `bcast_affine`/`bcast_fma`.
//!
//! The `scale_shift_table` add (`table[6,inner] + temb`) that produces the six
//! modulation signals lives in the driver; this block consumes the six ready
//! `[rows, inner]` tensors via [`WanMod`].

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError, WgpuPipeline};
use thinfer_core::ops::{
    BcastAddF32, BcastAffineF32, GeluF32, LayerNormF32, MatMulF32, MulF32, RopeF32,
};
use thinfer_core::residency::{GpuView, ResidencyError, WeightHandle, WeightResidency};
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::{BatchBuf, BatchScope};

use crate::common::block::{
    ActBuf, Block, BlockPipelines, alloc_act, alloc_matmul_out_buf, copy_tap, op_add, op_rmsnorm,
    op_sdpa,
};

/// Audited against the SkyReels-V2-DF-1.3B-540P-Diffusers `transformer/
/// config.json` (see `wan-plan.md` "Pinned config").
pub mod config {
    pub const NUM_HEADS: usize = 12;
    pub const HEAD_DIM: usize = 128;
    /// `num_heads * head_dim`. Equals the model dim for the 1.3B DiT.
    pub const INNER: usize = NUM_HEADS * HEAD_DIM; // 1536
    pub const DIM: usize = INNER;
    pub const FFN_DIM: usize = 8960;
    pub const NUM_LAYERS: usize = 30;
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
    /// Latent channels in / out (Wan2.1 VAE `z_dim`).
    pub const IN_CHANNELS: usize = 16;
    pub const OUT_CHANNELS: usize = 16;

    /// RoPE3D theta and the per-axis (t, h, w) sub-dimensions of `HEAD_DIM`.
    /// `h = w = 2 * (head_dim / 6)`, `t = head_dim - h - w` (diffusers
    /// `SkyReelsV2RotaryPosEmbed`). At head_dim 128: t=44, h=w=42.
    pub const ROPE_THETA: f32 = 10_000.0;
    pub const ROPE_H_DIM: usize = 2 * (HEAD_DIM / 6); // 42
    pub const ROPE_W_DIM: usize = 2 * (HEAD_DIM / 6); // 42
    pub const ROPE_T_DIM: usize = HEAD_DIM - ROPE_H_DIM - ROPE_W_DIM; // 44
    /// Max per-axis grid length the freq tables are precomputed to
    /// (`rope_max_seq_len`).
    pub const ROPE_MAX_SEQ_LEN: usize = 1024;
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
    pub fn new(seq: usize, text_seq: usize) -> Self {
        Self {
            dim: config::DIM,
            n_heads: config::NUM_HEADS,
            head_dim: config::HEAD_DIM,
            inner: config::INNER,
            ffn_dim: config::FFN_DIM,
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

/// The six per-token Diffusion-Forcing modulation signals, each `[rows, inner]`
/// (`scale_shift_table + temb`, already summed by the driver).
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

/// Full-elementwise `out = a * b`.
fn op_mul<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    a: ActBuf<'wsp>,
    b: ActBuf<'wsp>,
    dst: ActBuf<'wsp>,
) -> Result<(), WgpuError> {
    scope.dispatch_op::<MulF32>(&bp.mul, &[a.data, b.data], dst.data)
}

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

/// Per-token modulation `out = x * (1 + scale) + shift` (full-elementwise; DF
/// modulation varies per row). Composes mul + add + add through two scratch
/// buffers from `scope`.
#[allow(clippy::too_many_arguments)]
fn op_modulate<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    scale: ActBuf<'wsp>,
    shift: ActBuf<'wsp>,
    dst: ActBuf<'wsp>,
    rows: u32,
    dim: u32,
) -> Result<(), WgpuError> {
    // Distinct scratch per stage: elementwise ops bind input read-only and
    // output read-write, so input and output must not alias the same buffer.
    let xs = alloc_act(scope, bp, rows, dim)?;
    op_mul(scope, bp, x, scale, xs)?; // x * scale
    let xss = alloc_act(scope, bp, rows, dim)?;
    op_add(scope, bp, xs, shift, xss)?; // x * scale + shift
    op_add(scope, bp, x, xss, dst)?; // x + (x * scale + shift) = x * (1 + scale) + shift
    Ok(())
}

/// Gated residual `out = x + gate * y` (full-elementwise). Writes into a scratch
/// then adds; `out` may alias `x`.
#[allow(clippy::too_many_arguments)]
fn op_gate_residual<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    gate: ActBuf<'wsp>,
    y: ActBuf<'wsp>,
    dst: ActBuf<'wsp>,
    rows: u32,
    dim: u32,
) -> Result<(), WgpuError> {
    let gy = alloc_act(scope, bp, rows, dim)?;
    op_mul(scope, bp, gate, y, gy)?;
    op_add(scope, bp, x, gy, dst)?;
    Ok(())
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
    i8: Option<&WgpuPipeline>,
    dq_i8: Option<&crate::common::block::DequantStep>,
    dq: Option<&crate::common::block::DequantStep>,
    pipe: &WgpuPipeline,
    inst: &MatMulF32,
) -> Result<(), WgpuError> {
    let dims = scope.u32x4_uniform(rows, n, k, 0)?;
    Block::dispatch_matmul_site(
        scope, bp, input, w, out, dims, i8, dq_i8, dq, pipe, inst, rows, n, k,
    )
}

// ---------------------------------------------------------------------------
// Block forward
// ---------------------------------------------------------------------------

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
        let inner = s.inner as u32;
        let dff = s.ffn_dim as u32;
        let eps = s.norm_eps;
        let scale = s.sdpa_scale();
        let x_in = ActBuf::dense(x_in);
        let y_out = ActBuf::dense(y_out);

        // ============== 1. self-attention ==============
        let n1 = alloc_act(scope, bp, rows, dim)?;
        op_layernorm(scope, bp, x_in, n1, rows, dim, eps)?;
        let n1m = alloc_act(scope, bp, rows, dim)?;
        op_modulate(
            scope,
            bp,
            n1,
            ActBuf::dense(m.scale_msa),
            ActBuf::dense(m.shift_msa),
            n1m,
            rows,
            dim,
        )?;
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
            taps.self_q.as_ref(),
            taps.self_k.as_ref(),
            taps.self_v.as_ref(),
            taps.self_sa.as_ref(),
        )?;

        let x1 = alloc_act(scope, bp, rows, dim)?;
        op_gate_residual(
            scope,
            bp,
            x_in,
            ActBuf::dense(m.gate_msa),
            sa,
            x1,
            rows,
            dim,
        )?;
        copy_tap(
            scope,
            x1.data,
            taps.after_self.as_ref(),
            bp.act_bytes(rows * dim),
        )?;

        // ============== 2. cross-attention (to text) ==============
        // norm2 = affine LayerNorm: ln -> *weight -> +bias (channel-broadcast).
        let n2 = alloc_act(scope, bp, rows, dim)?;
        op_layernorm(scope, bp, x1, n2, rows, dim, eps)?;
        let n2w = alloc_act(scope, bp, rows, dim)?;
        let w2 = scope.import_copy(bufs.norm2_w);
        let u_aff = scope.u32x4_uniform(dim, 0, 0, 0)?;
        scope.bcast_affine::<BcastAffineF32>(
            &bp.bcast_affine,
            n2.data,
            w2,
            u_aff,
            n2w.data,
            rows * dim,
        )?;
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
        let n3 = alloc_act(scope, bp, rows, dim)?;
        op_layernorm(scope, bp, x2, n3, rows, dim, eps)?;
        let n3m = alloc_act(scope, bp, rows, dim)?;
        op_modulate(
            scope,
            bp,
            n3,
            ActBuf::dense(m.c_scale_mlp),
            ActBuf::dense(m.c_shift_mlp),
            n3m,
            rows,
            dim,
        )?;
        copy_tap(
            scope,
            n3m.data,
            taps.norm3.as_ref(),
            bp.act_bytes(rows * dim),
        )?;

        // up: [rows, inner] @ [ffn_dim, inner]ᵀ + bias -> [rows, ffn_dim]
        let up = alloc_matmul_out_buf(scope, bp, rows * dff)?;
        let wi = scope.import_copy(bufs.ffn_up_w);
        lin(
            scope,
            bp,
            n3m,
            wi,
            up,
            rows,
            dff,
            inner,
            bp.matmul_i8_ffn_up.as_ref(),
            bp.dequant_i8_ffn_up.as_ref(),
            bp.dequant_ffn_up.as_ref(),
            &bp.matmul_ffn_up,
            &bp.matmuls.ffn_up,
        )?;
        let bi = scope.import_copy(bufs.ffn_up_b);
        let upb = alloc_act(scope, bp, rows, dff)?;
        op_bias_add(scope, bp, ActBuf::dense(up), bi, upb, rows, dff)?;
        // gelu-tanh
        let gelu = alloc_act(scope, bp, rows, dff)?;
        scope.dispatch_op::<GeluF32>(&pipelines.gelu, &[upb.data], gelu.data)?;
        copy_tap(
            scope,
            gelu.data,
            taps.ffn_gelu.as_ref(),
            bp.act_bytes(rows * dff),
        )?;
        // down: [rows, ffn_dim] @ [inner, ffn_dim]ᵀ + bias -> [rows, inner]
        let down = alloc_matmul_out_buf(scope, bp, rows * dim)?;
        let wo = scope.import_copy(bufs.ffn_down_w);
        lin(
            scope,
            bp,
            gelu,
            wo,
            down,
            rows,
            dim,
            dff,
            bp.matmul_i8_ffn_down.as_ref(),
            bp.dequant_i8_ffn_down.as_ref(),
            bp.dequant_ffn_down.as_ref(),
            &bp.matmul_ffn_down,
            &bp.matmuls.ffn_down,
        )?;
        let bo = scope.import_copy(bufs.ffn_down_b);
        let downb = alloc_act(scope, bp, rows, dim)?;
        op_bias_add(scope, bp, ActBuf::dense(down), bo, downb, rows, dim)?;
        copy_tap(
            scope,
            downb.data,
            taps.ffn_down.as_ref(),
            bp.act_bytes(rows * dim),
        )?;

        op_gate_residual(
            scope,
            bp,
            x2,
            ActBuf::dense(m.c_gate_mlp),
            downb,
            y_out,
            rows,
            dim,
        )?;
        Ok(())
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
        let q = self.biased_proj(scope, bp, q_src, w.q_w, w.q_b, q_rows, inner)?;
        let k = self.biased_proj(scope, bp, kv_src, w.k_w, w.k_b, kv_rows, inner)?;
        let v = self.biased_proj(scope, bp, kv_src, w.v_w, w.v_b, kv_rows, inner)?;
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
        let proj = alloc_matmul_out_buf(scope, bp, q_rows * inner)?;
        let ow = scope.import_copy(w.o_w);
        lin(
            scope,
            bp,
            sa,
            ow,
            proj,
            q_rows,
            inner,
            inner,
            bp.matmul_i8_proj.as_ref(),
            bp.dequant_i8_proj.as_ref(),
            bp.dequant_proj.as_ref(),
            &bp.matmul_proj,
            &bp.matmuls.proj,
        )?;
        let ob = scope.import_copy(w.o_b);
        let projb = alloc_act(scope, bp, q_rows, inner)?;
        op_bias_add(scope, bp, ActBuf::dense(proj), ob, projb, q_rows, inner)?;
        Ok(projb)
    }

    /// `out = x @ wᵀ + bias` through the qkv matmul site.
    fn biased_proj<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        bp: &BlockPipelines,
        x: ActBuf<'wsp>,
        w: BufRef,
        b: BufRef,
        rows: u32,
        n: u32,
    ) -> Result<ActBuf<'wsp>, WgpuError> {
        let dim = self.shape.dim as u32;
        let out = alloc_matmul_out_buf(scope, bp, rows * n)?;
        let wv = scope.import_copy(w);
        lin(
            scope,
            bp,
            x,
            wv,
            out,
            rows,
            n,
            dim,
            bp.matmul_i8_qkv.as_ref(),
            bp.dequant_i8_qkv.as_ref(),
            bp.dequant_qkv.as_ref(),
            &bp.matmul_qkv,
            &bp.matmuls.qkv,
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
        assert_eq!(config::INNER, config::NUM_HEADS * config::HEAD_DIM);
        assert_eq!(config::DIM, config::INNER);
    }

    #[test]
    fn shape_builds() {
        let sh = WanDitBlockShape::new(1024, config::TEXT_SEQ);
        assert_eq!(sh.inner, 1536);
        assert!((sh.sdpa_scale() - 1.0 / (128f32).sqrt()).abs() < 1e-9);
    }
}
