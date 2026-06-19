//! Wan2.2 DiT stack driver (`WanTransformer3DModel.forward`,
//! `transformer_wan.py`). Single-stream video DiT, `B = 1`:
//!
//! ```text
//! x      = patch_linear(patchify(image)) + bias        // [n_tok, inner]
//! temb, timestep_proj, text = condition_embedder(timestep, text_states)
//! freqs  = rope3d(grid)                                 // [n_tok, head_dim]
//! for blk in blocks:                                    // 30 Wan blocks
//!     mod6 = blk.scale_shift_table[6] + timestep_proj   // [6, inner] vectors
//!     x    = blk(x, text, freqs, mod6)
//! shift, scale = model.scale_shift_table[2] + temb      // [inner] vectors
//! x      = norm_out(x) * (1 + scale) + shift
//! tokens = proj_out(x)                                  // [n_tok, out_ch*p_t*p_h*p_w]
//! image  = unpatchify(tokens)                           // [out_ch, F, H, W]
//! ```
//!
//! The timestep is a single scalar uniform over the clip, so `temb`,
//! `timestep_proj`, and the final `shift`/`scale` are channel vectors `[inner]`
//! (`[6, inner]` for the per-block six), broadcast over the `n_tok` rows inside
//! the block via `bcast_affine`/`bcast_fma`. No per-token materialization (the
//! SkyReels-V2-DF path broadcast the six bases to `[n_tok, inner]` up front, a
//! `6 * n_tok * inner` resident hog; that is gone).
//!
//! Residency: each block pages its weights via [`WeightResidency`]; the loop
//! awaits each block's GPU fence concurrently with streaming the next block's
//! weights (the same overlap model as `z_image/dit.rs`). Activations persist
//! across submits as caller-owned `WsBuf`s.

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError};
use thinfer_core::ops::{ActDtype, BcastAddF32, BcastModulateF32, LayerNormF32};
use thinfer_core::residency::{ResidencyError, WeightResidency};
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace, WsBuf};
use tracing::Instrument;

use thinfer_core::trace::{self, PHASE};

use crate::common::block::{ActBuf, BlockPipelines, alloc_act, alloc_matmul_out_buf};
use crate::common::embedders::{LinearBiasBufs, LinearBiasHandles};
use crate::common::seq::{act_upload_bytes, freqs_upload_bytes};
use crate::wan::condition_embedder::{
    ConditionEmbedder, ConditionEmbedderHandles, ConditionEmbedderOut,
};
use crate::wan::dit_block::{
    WanDitBlock, WanDitBlockBufs, WanDitBlockHandles, WanDitBlockShape, WanDitBlockTaps,
    WanDitConfig, WanDitPipelines, WanMod, config,
};
use crate::wan::patchify::{self, PatchGrid};
use crate::wan::rope3d::WanRope3d;

// ---------------------------------------------------------------------------
// Residency handles (file -> handles loader is the step-5 glue)
// ---------------------------------------------------------------------------

/// Every resident weight group of one Wan DiT (module-level + per block).
#[derive(Clone, Debug)]
pub struct LoadedWanDitHandles {
    /// `patch_embedding` conv folded to a linear `[inner, in_ch*p_t*p_h*p_w]`
    /// (transposed `[patch_in, inner]`) + bias `[inner]`.
    pub patch: LinearBiasHandles,
    pub condition: ConditionEmbedderHandles,
    pub blocks: Vec<WanDitBlockHandles>,
    /// Model-level `scale_shift_table` `[2, inner]` (final norm modulation).
    pub scale_shift_table: thinfer_core::residency::WeightHandle,
    /// `proj_out` `[out_ch*p_t*p_h*p_w, inner]` (transposed) + bias.
    pub proj_out: LinearBiasHandles,
}

// ---------------------------------------------------------------------------
// Shape + inputs + outputs
// ---------------------------------------------------------------------------

/// Runtime shape derived from the latent grid.
#[derive(Clone, Copy, Debug)]
pub struct WanDitShape {
    pub grid: PatchGrid,
    pub text_seq: usize,
    /// Patch tokens `ppf * pph * ppw`.
    pub n_tok: usize,
}

impl WanDitShape {
    pub fn new(c: usize, f: usize, h: usize, w: usize, text_seq: usize) -> Self {
        let grid = PatchGrid::new(c, f, h, w);
        Self {
            n_tok: grid.n_tok(),
            grid,
            text_seq,
        }
    }
}

/// Per-forward inputs. All host-side; the driver uploads.
pub struct WanDitInputs<'a> {
    /// Latent `[C, F, H, W]` row-major f32.
    pub image: &'a [f32],
    /// umT5 text states `[text_seq, text_dim]` row-major f32.
    pub text: &'a [f32],
    /// Scalar diffusion timestep, uniform over the whole clip (the distilled
    /// T2V line is plain flow-matching, not per-frame Diffusion Forcing).
    pub timestep: f32,
}

/// One DiT forward output, ready for VAE decode.
pub struct WanDitOutput {
    /// `[out_ch, F, H, W]` row-major f32.
    pub image: Vec<f32>,
}

/// Optional per-stage readbacks (ad-hoc bringup diff vs pyref). The committed
/// gate is the single end-state `video_e2e`; these localize divergence.
#[derive(Default)]
pub struct WanDitTaps<'a> {
    pub patch_x: Option<&'a mut Vec<f32>>,
    pub temb: Option<&'a mut Vec<f32>>,
    pub timestep_proj: Option<&'a mut Vec<f32>>,
    pub text_proj: Option<&'a mut Vec<f32>>,
    /// Residual stream after each block (`len == num_layers`, empty sub-vecs
    /// for blocks that did not fire).
    pub per_block: Option<&'a mut Vec<Vec<f32>>>,
    /// Per-op taps inside block 0.
    pub block0: Option<WanDitBlockTaps>,
    pub final_norm: Option<&'a mut Vec<f32>>,
    pub proj_out: Option<&'a mut Vec<f32>>,
}

#[derive(Debug)]
pub enum WanDitError<SE: core::fmt::Debug> {
    Wgpu(WgpuError),
    Residency(ResidencyError<SE, WgpuError>),
}

impl<SE: core::fmt::Debug> From<WgpuError> for WanDitError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}

impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for WanDitError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Cross-submit residual-stream buffer (dense activation at the pipeline dtype).
struct ResStream {
    data: WsBuf<WgpuBackend>,
}

impl ResStream {
    fn alloc(
        scratch: &Workspace<WgpuBackend>,
        pipelines: &BlockPipelines,
        rows: u32,
        dim: u32,
    ) -> Result<Self, WgpuError> {
        Ok(Self {
            data: scratch.alloc(pipelines.act_bytes(rows * dim))?,
        })
    }

    fn as_act_ref(&self) -> BufRef {
        self.data.as_buf_ref()
    }
}

pub struct WanDit {
    pub shape: WanDitShape,
    pub cfg: WanDitConfig,
    pub handles: LoadedWanDitHandles,
    block: WanDitBlock,
    condition_embedder: ConditionEmbedder,
    rope: WanRope3d,
}

impl WanDit {
    pub fn assemble(handles: LoadedWanDitHandles, shape: WanDitShape, cfg: WanDitConfig) -> Self {
        let block = WanDitBlock::new(WanDitBlockShape::new(&cfg, shape.n_tok, shape.text_seq));
        Self {
            shape,
            cfg,
            handles,
            block,
            condition_embedder: ConditionEmbedder::from_cfg(&cfg),
            rope: WanRope3d::new(),
        }
    }

    pub async fn forward<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipelines: &WanDitPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        inputs: &WanDitInputs<'_>,
    ) -> Result<WanDitOutput, WanDitError<S::Error>> {
        self.forward_with_taps(
            backend,
            pipelines,
            residency,
            scratch,
            inputs,
            WanDitTaps::default(),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn forward_with_taps<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipelines: &WanDitPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        inputs: &WanDitInputs<'_>,
        mut taps: WanDitTaps<'_>,
    ) -> Result<WanDitOutput, WanDitError<S::Error>> {
        let bp = &pipelines.block;
        let s = self.shape;
        let inner = self.cfg.inner() as u32;
        let rows = s.n_tok as u32;
        let ppf = s.grid.ppf;
        let text_seq = s.text_seq as u32;

        // --- 1. patchify image + front-door linear -> x [n_tok, inner] ---
        let patch_in = s.grid.patch_in() as u32;
        let tokens = patchify::patchify(inputs.image, &s.grid);
        let tok_bytes = act_upload_bytes(bp.act_dtype, &tokens);
        let tok_buf = scratch.alloc(tok_bytes.len() as u64)?;
        backend.write_buffer(tok_buf.id, 0, &tok_bytes)?;
        let x = ResStream::alloc(scratch, bp, rows, inner)?;
        {
            let views = self.handles.patch.acquire(residency, backend).await?;
            let scope = scratch.batch();
            self.linear_bias(
                &scope,
                bp,
                scope.import_copy(tok_buf.as_buf_ref()),
                &views.bufs(),
                rows,
                patch_in,
                inner,
                scope.import_copy(x.as_act_ref()),
            )?;
            scope.submit_void().await?;
        }
        read_tap(
            backend,
            &x.as_act_ref(),
            (rows * inner) as usize,
            bp.act_dtype,
            &mut taps.patch_x,
        )
        .await?;

        // --- 2. condition embedder -> temb, timestep_proj, text ---
        let text_bytes = act_upload_bytes(bp.act_dtype, inputs.text);
        let text_in = scratch.alloc(text_bytes.len() as u64)?;
        backend.write_buffer(text_in.id, 0, &text_bytes)?;
        // temb [1, inner], timestep_proj [1, 6*inner] (scalar-t channel vectors).
        let temb = scratch.alloc(bp.act_bytes(inner))?;
        let tproj = scratch.alloc(bp.act_bytes(6 * inner))?;
        let text = ResStream::alloc(scratch, bp, text_seq, inner)?;
        {
            let views = self.handles.condition.acquire(residency, backend).await?;
            let scope = scratch.batch();
            let out = ConditionEmbedderOut {
                temb: scope.import_copy(temb.as_buf_ref()),
                timestep_proj: scope.import_copy(tproj.as_buf_ref()),
                text: scope.import_copy(text.as_act_ref()),
            };
            self.condition_embedder.forward(
                &scope,
                bp,
                &pipelines.gelu,
                inputs.timestep,
                scope.import_copy(text_in.as_buf_ref()),
                text_seq,
                &out,
                &views.bufs(),
            )?;
            scope.submit_void().await?;
        }
        read_tap(
            backend,
            &temb.as_buf_ref(),
            inner as usize,
            bp.act_dtype,
            &mut taps.temb,
        )
        .await?;
        read_tap(
            backend,
            &tproj.as_buf_ref(),
            (6 * inner) as usize,
            bp.act_dtype,
            &mut taps.timestep_proj,
        )
        .await?;
        read_tap(
            backend,
            &text.as_act_ref(),
            (text_seq * inner) as usize,
            bp.act_dtype,
            &mut taps.text_proj,
        )
        .await?;

        // --- 3. RoPE3D freqs for the latent grid ---
        // Pack to the act dtype: the F16/Bf16 rope kernels read freqs in the act
        // dtype (packed), not f32. (Mirrors z_image; the f32-only `lookup_bytes`
        // would feed f16 kernels reinterpreted garbage -> inf -> NaN softmax.)
        let freqs_bytes =
            freqs_upload_bytes(bp.act_dtype, &self.rope.lookup(ppf, s.grid.pph, s.grid.ppw));
        let freqs = scratch.alloc(freqs_bytes.len() as u64)?;
        backend.write_buffer(freqs.id, 0, &freqs_bytes)?;

        // --- 4. main transformer blocks (residency-paged) ---
        // The six modulation vectors `scale_shift_table[k] + timestep_proj[k]`
        // are built per block inside its scope (`build_mod`); each is `[inner]`
        // and broadcasts over the rows in the block. No up-front per-token
        // materialization.
        let mut x_cur = x;
        // Activation-tiling tier: above ~one tile's worth of tokens, run each
        // block as pass A (row-tiled q/k/v) -> global self-SDPA -> pass B
        // (row-tiled cross-attn + FFN), so the heavy `[tile, ffn_dim]` FFN
        // transients recycle through the pool instead of all living at once.
        // Below the threshold (the e2e gate's tiny grids), n_tiles == 1 keeps
        // the original single-scope path bit-identical. Diag taps force the
        // untiled path (intra-block taps are single-scope only).
        let n_tiles = if taps.block0.is_some() {
            1
        } else {
            rows.div_ceil(dit_tile_rows()).max(1)
        };
        let tile = if n_tiles > 1 {
            Some(TileBufs::alloc(scratch, bp, rows, text_seq, inner)?)
        } else {
            None
        };
        let _lr = trace::scope!("wan.dit.blocks", n = self.handles.blocks.len()).entered();
        let mut pending = if self.handles.blocks.is_empty() {
            None
        } else {
            Some(
                self.handles.blocks[0]
                    .acquire(residency, backend)
                    .instrument(tracing::debug_span!(target: PHASE, "wan.acquire", idx = 0_usize))
                    .await?,
            )
        };
        for idx in 0..self.handles.blocks.len() {
            let _bs = trace::scope!(format_args!("block.{idx}")).entered();
            if idx > 0
                && let Some(sink) = taps.per_block.as_deref_mut()
            {
                let prev = idx - 1;
                if sink.len() <= prev {
                    sink.resize(prev + 1, Vec::new());
                }
                read_into_f32(
                    backend,
                    &x_cur.as_act_ref(),
                    (rows * inner) as usize,
                    bp.act_dtype,
                    &mut sink[prev],
                )
                .await?;
            }
            let nxt = ResStream::alloc(scratch, bp, rows, inner)?;
            let views = pending.take().expect("pending block acquire missing");
            let block_taps = if idx == 0 {
                taps.block0.clone().unwrap_or_default()
            } else {
                WanDitBlockTaps::default()
            };
            {
                let next_idx = idx + 1;
                let next_acquire = async {
                    match self.handles.blocks.get(next_idx) {
                        Some(h) => {
                            let span =
                                tracing::debug_span!(target: PHASE, "wan.acquire", idx = next_idx);
                            Ok::<_, ResidencyError<S::Error, WgpuError>>(Some(
                                h.acquire(residency, backend).instrument(span).await?,
                            ))
                        }
                        None => Ok(None),
                    }
                };
                let prefetch_after = async {
                    match self.handles.blocks.get(idx + 2) {
                        Some(h) => {
                            let span =
                                tracing::debug_span!(target: PHASE, "wan.prefetch", idx = idx + 2);
                            h.prefetch(residency, backend).instrument(span).await
                        }
                        None => Ok(()),
                    }
                };
                if let Some(tb) = tile.as_ref() {
                    // Tiled path owns its own (serial) submits; overlap weight
                    // streaming for the next block(s) with the whole movement.
                    let block_bufs = views.bufs();
                    let sst = views.scale_shift_table();
                    let compute = self
                        .forward_block_tiled(
                            pipelines,
                            scratch,
                            &block_bufs,
                            sst,
                            tproj.as_buf_ref(),
                            freqs.as_buf_ref(),
                            text.as_act_ref(),
                            x_cur.as_act_ref(),
                            nxt.as_act_ref(),
                            tb,
                            n_tiles,
                        )
                        .instrument(tracing::debug_span!(target: PHASE, "wan.tiled", idx));
                    let (c_res, n_res, p_res) =
                        futures::join!(compute, next_acquire, prefetch_after);
                    c_res?;
                    p_res?;
                    pending = n_res?;
                } else {
                    let scope = scratch.batch();
                    // Build the six modulation vectors: scale_shift_table[k] +
                    // timestep_proj[k], each `[inner]` (broadcast over the rows).
                    let tproj_h = scope.import_copy(tproj.as_buf_ref());
                    let m =
                        self.build_mod(&scope, bp, views.scale_shift_table(), tproj_h, inner)?;
                    let nxt_h = scope.import_copy(nxt.as_act_ref());
                    self.block.forward(
                        &scope,
                        pipelines,
                        scope.import_copy(x_cur.as_act_ref()),
                        scope.import_copy(text.as_act_ref()),
                        scope.import_copy(freqs.as_buf_ref()),
                        &m,
                        nxt_h,
                        &views.bufs(),
                        &block_taps,
                    )?;
                    let submit_fut = scope
                        .submit_deferred()
                        .instrument(tracing::debug_span!(target: PHASE, "wan.submit", idx));
                    let (s_res, n_res, p_res) =
                        futures::join!(submit_fut, next_acquire, prefetch_after);
                    s_res?;
                    p_res?;
                    pending = n_res?;
                }
            }
            x_cur = nxt;
        }
        // Last block's residual, if requested.
        if let Some(sink) = taps.per_block.as_deref_mut() {
            let last = self.handles.blocks.len();
            if last > 0 {
                if sink.len() < last {
                    sink.resize(last, Vec::new());
                }
                read_into_f32(
                    backend,
                    &x_cur.as_act_ref(),
                    (rows * inner) as usize,
                    bp.act_dtype,
                    &mut sink[last - 1],
                )
                .await?;
            }
        }

        // --- 6. final norm + modulation + proj_out ---
        let proj_w = self.cfg.out_channels * config::PATCH_T * config::PATCH_H * config::PATCH_W;
        let proj_out = ResStream::alloc(scratch, bp, rows, proj_w as u32)?;
        // Persist the post-modulation activation when a tap wants it (readback
        // after submit; scope-local buffers do not survive the submit).
        let final_norm_ws = match taps.final_norm {
            Some(_) => Some(scratch.alloc(bp.act_bytes(rows * inner))?),
            None => None,
        };
        {
            let sst = residency
                .acquire(self.handles.scale_shift_table, backend)
                .await?;
            let pv = self.handles.proj_out.acquire(residency, backend).await?;
            let scope = scratch.batch();
            // shift = table[0] + temb ; scale = table[1] + temb (all [inner]).
            // Same bf16-decoding `mod_signal` path as the per-block modulation.
            let temb_h = scope.import_copy(temb.as_buf_ref());
            let shift = self.mod_signal(&scope, bp, sst.buf(), 0, temb_h, inner)?;
            let scale = self.mod_signal(&scope, bp, sst.buf(), 1, temb_h, inner)?;
            // norm_out (FP32 LayerNorm, no affine).
            let x_h = ActBuf::dense(scope.import_copy(x_cur.as_act_ref()));
            let normed = alloc_act(&scope, bp, rows, inner)?;
            let ln_u = scope.u32x4_uniform(rows, inner, config::EPS.to_bits(), 0)?;
            scope.layernorm::<LayerNormF32>(&bp.layernorm, x_h.data, ln_u, normed.data, rows)?;
            // out = normed * (1 + scale) + shift.
            let modded = alloc_act(&scope, bp, rows, inner)?;
            self.modulate(&scope, bp, normed, scale, shift, modded, rows, inner)?;
            if let Some(fnw) = &final_norm_ws {
                let dst = scope.import_copy(fnw.as_buf_ref());
                scope.copy_buffer_to_buffer(modded.data, 0, dst, 0, bp.act_bytes(rows * inner))?;
            }
            // proj_out: [rows, inner] @ [proj_w, inner]ᵀ + bias.
            self.linear_bias(
                &scope,
                bp,
                modded.data,
                &pv.bufs(),
                rows,
                inner,
                proj_w as u32,
                scope.import_copy(proj_out.as_act_ref()),
            )?;
            scope.submit_void().await?;
        }
        if let (Some(fnw), Some(sink)) = (&final_norm_ws, taps.final_norm.as_deref_mut()) {
            read_into_f32(
                backend,
                &fnw.as_buf_ref(),
                (rows * inner) as usize,
                bp.act_dtype,
                sink,
            )
            .await?;
        }
        read_tap(
            backend,
            &proj_out.as_act_ref(),
            (rows * proj_w as u32) as usize,
            bp.act_dtype,
            &mut taps.proj_out,
        )
        .await?;

        // --- 7. unpatchify on CPU -> [out_ch, F, H, W] ---
        let mut tokens_out = Vec::new();
        read_into_f32(
            backend,
            &proj_out.as_act_ref(),
            (rows * proj_w as u32) as usize,
            bp.act_dtype,
            &mut tokens_out,
        )
        .await?;
        let image = patchify::unpatchify(&tokens_out, &s.grid, self.cfg.out_channels);
        Ok(WanDitOutput { image })
    }

    /// Run one transformer block in the activation-tiled regime: pass A
    /// (row-tiled q/k/v projection + RoPE) -> global self-SDPA barrier ->
    /// pass B (row-tiled o-proj + cross-attn + FFN). Each movement submits on
    /// its own so the pool recycles tile transients between submits; the only
    /// resolution-growing residents are `qx`/`kx`/`v`/`sa` (the cost of an
    /// exact global attention). Numerically identical to [`WanDitBlock::forward`].
    #[allow(clippy::too_many_arguments)]
    async fn forward_block_tiled(
        &self,
        pipelines: &WanDitPipelines,
        scratch: &Workspace<WgpuBackend>,
        bufs: &WanDitBlockBufs,
        sst: BufRef,
        tproj: BufRef,
        freqs: BufRef,
        text: BufRef,
        x_in: BufRef,
        y_out: BufRef,
        tb: &TileBufs,
        n_tiles: u32,
    ) -> Result<(), WgpuError> {
        let bp = &pipelines.block;
        let inner = self.cfg.inner() as u32;
        let rows = self.shape.n_tok as u32;
        let text_seq = self.shape.text_seq as u32;
        let hd = config::HEAD_DIM as u32;

        // Modulation vectors for this block, persisted across the tile scopes.
        self.fill_mod(bp, scratch, sst, tproj, inner, &tb.m).await?;

        // Cross-attention K/V projected once (shared by every query tile).
        {
            let scope = scratch.batch();
            let text_h = scope.import_copy(text);
            let ck = scope.import_copy(tb.ck.as_buf_ref());
            let cv = scope.import_copy(tb.cv.as_buf_ref());
            self.block.cross_kv(
                &scope,
                pipelines,
                text_h,
                &bufs.cross_attn,
                ck,
                cv,
                text_seq,
            )?;
            scope.submit_void().await?;
        }

        // Pass A: per-tile q/k/v projection + RoPE into the full qx/kx/v.
        for t in 0..n_tiles {
            let (r0, tr) = tile_range(rows, n_tiles, t);
            let scope = scratch.batch();
            let m = mk_mod(&scope, &tb.m);
            let x_slice = scope.import_copy(act_slice(x_in, r0, tr, inner, bp));
            let f_slice = scope.import_copy(act_slice(freqs, r0, tr, hd, bp));
            let qx = scope.import_copy(act_slice(tb.qx.as_buf_ref(), r0, tr, inner, bp));
            let kx = scope.import_copy(act_slice(tb.kx.as_buf_ref(), r0, tr, inner, bp));
            let v = scope.import_copy(act_slice(tb.v.as_buf_ref(), r0, tr, inner, bp));
            self.block.self_qkv_tile(
                &scope,
                pipelines,
                x_slice,
                f_slice,
                &m,
                qx,
                kx,
                v,
                tr,
                &bufs.self_attn,
            )?;
            scope.submit_void().await?;
        }

        // Barrier: global self-attention over the whole sequence.
        {
            let scope = scratch.batch();
            let qx = scope.import_copy(tb.qx.as_buf_ref());
            let kx = scope.import_copy(tb.kx.as_buf_ref());
            let v = scope.import_copy(tb.v.as_buf_ref());
            let sa = scope.import_copy(tb.sa.as_buf_ref());
            self.block
                .self_sdpa(&scope, pipelines, qx, kx, v, sa, rows)?;
            scope.submit_void().await?;
        }

        // Pass B: per-tile o-proj + gated residual + cross-attn + FFN -> y_out.
        for t in 0..n_tiles {
            let (r0, tr) = tile_range(rows, n_tiles, t);
            let scope = scratch.batch();
            let m = mk_mod(&scope, &tb.m);
            let x_slice = scope.import_copy(act_slice(x_in, r0, tr, inner, bp));
            let sa_slice = scope.import_copy(act_slice(tb.sa.as_buf_ref(), r0, tr, inner, bp));
            let y_slice = scope.import_copy(act_slice(y_out, r0, tr, inner, bp));
            let ck = scope.import_copy(tb.ck.as_buf_ref());
            let cv = scope.import_copy(tb.cv.as_buf_ref());
            self.block.post_attn_tile(
                &scope, pipelines, x_slice, sa_slice, &m, ck, cv, y_slice, tr, text_seq, bufs,
            )?;
            scope.submit_void().await?;
        }
        Ok(())
    }

    /// Build the six modulation vectors for one block into the persistent
    /// `[inner]` buffers `m` (so the tile scopes can broadcast them). Same
    /// `scale_shift_table[k] + timestep_proj[k]` sum as [`Self::build_mod`].
    async fn fill_mod(
        &self,
        bp: &BlockPipelines,
        scratch: &Workspace<WgpuBackend>,
        sst: BufRef,
        tproj: BufRef,
        inner: u32,
        m: &[WsBuf<WgpuBackend>; 6],
    ) -> Result<(), WgpuError> {
        let scope = scratch.batch();
        let tproj_h = scope.import_copy(tproj);
        let built = self.build_mod(&scope, bp, sst, tproj_h, inner)?;
        let srcs = [
            built.shift_msa,
            built.scale_msa,
            built.gate_msa,
            built.c_shift_mlp,
            built.c_scale_mlp,
            built.c_gate_mlp,
        ];
        let row_b = bp.act_bytes(inner);
        for (i, src) in srcs.into_iter().enumerate() {
            let dst = scope.import_copy(m[i].as_buf_ref());
            scope.copy_buffer_to_buffer(src, 0, dst, 0, row_b)?;
        }
        scope.submit_void().await
    }

    /// Build the six modulation vectors `scale_shift_table[k] +
    /// timestep_proj[k]`, each `[inner]`. `table` is the block's `[6, inner]`
    /// bf16 weight (a raw `BufRef`, NOT pre-imported as an act buffer); `tproj`
    /// the imported `[6, inner]` act-dtype projected timestep. The add is done
    /// via [`Self::mod_signal`] (`bcast_add`, which decodes the bf16 table
    /// operand) - a byte copy of the table into an act buffer would reinterpret
    /// its bf16 bits as f16 and corrupt the modulation.
    fn build_mod<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        bp: &BlockPipelines,
        table: BufRef,
        tproj: BatchBuf<'wsp>,
        inner: u32,
    ) -> Result<WanMod<'wsp>, WgpuError> {
        let sig = |k: u32| -> Result<BatchBuf<'wsp>, WgpuError> {
            let p = self.mod_row(scope, bp, tproj, k, inner)?;
            self.mod_signal(scope, bp, table, k, p, inner)
        };
        Ok(WanMod {
            shift_msa: sig(0)?,
            scale_msa: sig(1)?,
            gate_msa: sig(2)?,
            c_shift_mlp: sig(3)?,
            c_scale_mlp: sig(4)?,
            c_gate_mlp: sig(5)?,
        })
    }

    /// Copy the `[inner]` channel vector at row `k` of an ACT-dtype `[*, inner]`
    /// buffer into a fresh scope buffer (used for the `timestep_proj` / `temb`
    /// rows, which are activations - never for the bf16 `scale_shift_table`).
    fn mod_row<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        bp: &BlockPipelines,
        src: BatchBuf<'wsp>,
        k: u32,
        inner: u32,
    ) -> Result<BatchBuf<'wsp>, WgpuError> {
        let dst = alloc_act(scope, bp, 1, inner)?;
        let row_b = bp.act_bytes(inner);
        scope.copy_buffer_to_buffer(src, k as u64 * row_b, dst.data, 0, row_b)?;
        Ok(dst.data)
    }

    /// `out = x + table[k]`, where `x` is an `[inner]` ACT vector and `table` is
    /// the bf16 `scale_shift_table` weight (`[*, inner]`). Row `k` is sliced
    /// directly out of the weight buffer and added via `bcast_add`, whose
    /// `(act=F16, weight=Bf16)` kernel decodes the bf16 table operand. This is
    /// the bias-add path the q/k/v/ffn biases already use; routing the table
    /// through a plain act `op_add` instead would read its bf16 bytes as f16.
    fn mod_signal<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        bp: &BlockPipelines,
        table: BufRef,
        k: u32,
        x: BatchBuf<'wsp>,
        inner: u32,
    ) -> Result<BatchBuf<'wsp>, WgpuError> {
        // scale_shift_table is always bf16 (registered passthrough, never
        // transcoded), so a row is `inner` bf16 elements = `inner * 2` bytes.
        let row_b = inner as u64 * 2;
        let row = BufRef::view(table.id, table.offset + k as u64 * row_b, row_b);
        let s = scope.import_copy(row);
        let dst = alloc_act(scope, bp, 1, inner)?;
        let u = scope.u32x4_uniform(inner, 0, 0, 0)?;
        scope.bcast_add::<BcastAddF32>(&bp.bcast_add, x, s, u, dst.data, inner)?;
        Ok(dst.data)
    }

    /// `out = x * (1 + scale) + shift` with `scale`/`shift` `[inner]` channel
    /// vectors broadcast over the `rows` tokens. `scale`/`shift` are runtime ACTS
    /// (`scale_shift_table + timestep_proj`), so this uses the fused
    /// `bcast_modulate` (reads both in the act dtype; bias=1 folds the `1 +`).
    /// NOT `bcast_affine` + `bcast_add`: `bcast_add` reads its broadcast vector as
    /// a weight and would reinterpret the f16 `shift` act as bf16, dropping it
    /// (the same BUG #3 the per-block `op_modulate` was migrated to fix).
    #[allow(clippy::too_many_arguments)]
    fn modulate<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        bp: &BlockPipelines,
        x: ActBuf<'wsp>,
        scale: BatchBuf<'wsp>,
        shift: BatchBuf<'wsp>,
        dst: ActBuf<'wsp>,
        rows: u32,
        inner: u32,
    ) -> Result<(), WgpuError> {
        let u = scope.u32x4_uniform(inner, 1.0_f32.to_bits(), 0, 0)?;
        scope.bcast_modulate::<BcastModulateF32>(
            &bp.bcast_modulate,
            x.data,
            scale,
            shift,
            u,
            dst.data,
            rows * inner,
        )
    }

    /// `out = x @ wᵀ + b` through the qkv matmul site (dense front-door).
    #[allow(clippy::too_many_arguments)]
    fn linear_bias<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        bp: &BlockPipelines,
        x: BatchBuf<'wsp>,
        w: &LinearBiasBufs,
        rows: u32,
        in_dim: u32,
        out_dim: u32,
        out: BatchBuf<'wsp>,
    ) -> Result<(), WgpuError> {
        let pre = alloc_matmul_out_buf(scope, bp, rows * out_dim)?;
        let dims = scope.u32x4_uniform(rows, out_dim, in_dim, 0)?;
        let weight = scope.import_copy(w.weight);
        scope.matmul(
            &bp.matmul_qkv,
            &bp.matmuls.qkv,
            x,
            weight,
            dims,
            pre,
            rows,
            out_dim,
        )?;
        let u = scope.u32x4_uniform(out_dim, 0, 0, 0)?;
        let bias = scope.import_copy(w.bias);
        scope.bcast_add::<BcastAddF32>(&bp.bcast_add, pre, bias, u, out, rows * out_dim)
    }
}

// ---------------------------------------------------------------------------
// Activation tiling
// ---------------------------------------------------------------------------

/// Row-tile granularity for the DiT activation-tiling tier. A block's pass-B
/// transient envelope is `~(17 * inner + 3 * ffn_dim) * tile_rows * act_bytes`
/// (the FFN `[tile, ffn_dim]` buffers dominate); at 1024 rows / f16 / the 5B
/// geometry that is ~190 MiB, comfortably inside the `DIT_WORKSPACE_RESERVE`
/// envelope while keeping the submit count low. Sequences at or under one tile
/// (the e2e gate) run untiled.
const DIT_TILE_ROWS: u32 = 1024;

/// Effective tile-row threshold, overridable via `THINFER_DIT_TILE_ROWS` (read
/// once). Diagnostics use it to force tiling ON at a small grid (validate the
/// tiled path against the parity gate's ground truth) or OFF at a large grid
/// (A/B the tier vs an untiled reference). Falls back to [`DIT_TILE_ROWS`].
fn dit_tile_rows() -> u32 {
    use std::sync::OnceLock;
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("THINFER_DIT_TILE_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DIT_TILE_ROWS)
    })
}

/// Cross-submit buffers for one block's tiled execution, reused across all
/// blocks (every block has the same geometry). `qx`/`kx`/`v`/`sa` are full
/// `[rows, inner]`; `ck`/`cv` the `[text_seq, inner]` cross-attn K/V; `m` the
/// six `[inner]` modulation vectors.
struct TileBufs {
    qx: WsBuf<WgpuBackend>,
    kx: WsBuf<WgpuBackend>,
    v: WsBuf<WgpuBackend>,
    sa: WsBuf<WgpuBackend>,
    ck: WsBuf<WgpuBackend>,
    cv: WsBuf<WgpuBackend>,
    m: [WsBuf<WgpuBackend>; 6],
}

impl TileBufs {
    fn alloc(
        scratch: &Workspace<WgpuBackend>,
        bp: &BlockPipelines,
        rows: u32,
        text_seq: u32,
        inner: u32,
    ) -> Result<Self, WgpuError> {
        let act = |n: u32| scratch.alloc(bp.act_bytes(n));
        Ok(Self {
            qx: act(rows * inner)?,
            kx: act(rows * inner)?,
            v: act(rows * inner)?,
            sa: act(rows * inner)?,
            ck: act(text_seq * inner)?,
            cv: act(text_seq * inner)?,
            m: [
                act(inner)?,
                act(inner)?,
                act(inner)?,
                act(inner)?,
                act(inner)?,
                act(inner)?,
            ],
        })
    }
}

/// Row range `[r0, r0 + tr)` for tile `t` of `n_tiles`, distributing the
/// remainder across the first tiles so every tile is within one row of even.
fn tile_range(rows: u32, n_tiles: u32, t: u32) -> (u32, u32) {
    let base = rows / n_tiles;
    let rem = rows % n_tiles;
    let r0 = t * base + t.min(rem);
    let tr = base + if t < rem { 1 } else { 0 };
    (r0, tr)
}

/// View rows `[row0, row0 + rows)` of a row-major `[*, dim]` activation buffer.
fn act_slice(base: BufRef, row0: u32, rows: u32, dim: u32, bp: &BlockPipelines) -> BufRef {
    BufRef::view(
        base.id,
        base.offset + row0 as u64 * bp.act_bytes(dim),
        bp.act_bytes(rows * dim),
    )
}

/// Re-import the six persistent modulation vectors into `scope` as a [`WanMod`]
/// (the per-tile broadcast operands). Index order matches [`WanDit::fill_mod`].
fn mk_mod<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    m: &[WsBuf<WgpuBackend>; 6],
) -> WanMod<'wsp> {
    WanMod {
        shift_msa: scope.import_copy(m[0].as_buf_ref()),
        scale_msa: scope.import_copy(m[1].as_buf_ref()),
        gate_msa: scope.import_copy(m[2].as_buf_ref()),
        c_shift_mlp: scope.import_copy(m[3].as_buf_ref()),
        c_scale_mlp: scope.import_copy(m[4].as_buf_ref()),
        c_gate_mlp: scope.import_copy(m[5].as_buf_ref()),
    }
}

// ---------------------------------------------------------------------------
// Readback helpers
// ---------------------------------------------------------------------------

/// Read a GPU buffer of `n` activation elements into `sink` as f32.
pub(crate) async fn read_into_f32(
    backend: &WgpuBackend,
    buf: &BufRef,
    n: usize,
    act: ActDtype,
    sink: &mut Vec<f32>,
) -> Result<(), WgpuError> {
    let bytes = backend
        .read_buffer(
            buf.id,
            buf.offset,
            (n * act.bytes_per_elem() as usize) as u64,
        )
        .await?;
    sink.clear();
    sink.reserve(n);
    match act {
        ActDtype::F32 => {
            for c in bytes.chunks_exact(4) {
                sink.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        ActDtype::Bf16 => {
            for c in bytes.chunks_exact(2) {
                let h = u16::from_le_bytes([c[0], c[1]]);
                sink.push(f32::from_bits((h as u32) << 16));
            }
        }
        ActDtype::F16 => {
            for c in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                sink.push(half::f16::from_bits(bits).to_f32());
            }
        }
        ActDtype::I8 => unreachable!("I8 is never a Wan DiT act dtype"),
    }
    Ok(())
}

/// Read into an optional tap sink (no-op when the tap is `None`).
async fn read_tap(
    backend: &WgpuBackend,
    buf: &BufRef,
    n: usize,
    act: ActDtype,
    sink: &mut Option<&mut Vec<f32>>,
) -> Result<(), WgpuError> {
    if let Some(s) = sink.as_deref_mut() {
        read_into_f32(backend, buf, n, act, s).await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_derives_grid() {
        let sh = WanDitShape::new(16, 5, 16, 16, 512);
        assert_eq!((sh.grid.ppf, sh.grid.pph, sh.grid.ppw), (5, 8, 8));
        assert_eq!(sh.n_tok, 5 * 8 * 8);
    }
}
