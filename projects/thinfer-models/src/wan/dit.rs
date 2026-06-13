//! Wan / SkyReels-V2 DiT stack driver (`SkyReelsV2Transformer3DModel.forward`,
//! `transformer_skyreels_v2.py`). Single-stream video DiT, `B = 1`:
//!
//! ```text
//! x      = patch_linear(patchify(image)) + bias        // [n_tok, inner]
//! temb, timestep_proj, text = condition_embedder(timesteps, fps, text_states)
//! freqs  = rope3d(grid)                                 // [n_tok, head_dim]
//! for blk in blocks:                                    // 30 SkyReelsV2 blocks
//!     mod6 = blk.scale_shift_table[6] + timestep_proj   // per-token (DF)
//!     x    = blk(x, text, freqs, mod6)
//! shift, scale = model.scale_shift_table[2] + temb      // per-token (DF)
//! x      = norm_out(x) * (1 + scale) + shift
//! tokens = proj_out(x)                                  // [n_tok, out_ch*p_t*p_h*p_w]
//! image  = unpatchify(tokens)                           // [out_ch, F, H, W]
//! ```
//!
//! Diffusion Forcing means the timestep is per latent frame: `temb`,
//! `timestep_proj`, and the final `shift`/`scale` are per-frame signals
//! broadcast over each frame's spatial tokens (`pph * ppw`) to per-token
//! `[n_tok, inner]` tensors, which is what [`WanDitBlock`] consumes. The driver
//! materializes the broadcast once (the six modulation bases + the final temb)
//! and reuses it across all 30 blocks.
//!
//! Residency: each block pages its weights via [`WeightResidency`]; the loop
//! awaits each block's GPU fence concurrently with streaming the next block's
//! weights (the same overlap model as `z_image/dit.rs`). Activations persist
//! across submits as caller-owned `WsBuf`s.
//!
//! Per-frame-broadcast modulation costs `6 * n_tok * inner` resident floats,
//! inherent to DF (diffusers materializes the same 4D temb). At 540P that is
//! large; a row-broadcast op that lets blocks read the compact `[f, inner]`
//! form directly is the native-memory optimization, deferred (parity shapes are
//! tiny). See `wan-plan.md`.

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError};
use thinfer_core::ops::{ActDtype, BcastAddF32, LayerNormF32, MulF32};
use thinfer_core::residency::{ResidencyError, WeightResidency};
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace, WsBuf};
use tracing::Instrument;

use thinfer_core::trace::{self, PHASE};

use crate::common::block::{ActBuf, BlockPipelines, alloc_act, alloc_matmul_out_buf, op_add};
use crate::common::embedders::{LinearBiasBufs, LinearBiasHandles};
use crate::common::seq::{act_upload_bytes, freqs_upload_bytes};
use crate::wan::condition_embedder::{
    ConditionEmbedder, ConditionEmbedderHandles, ConditionEmbedderOut,
};
use crate::wan::dit_block::{
    WanDitBlock, WanDitBlockHandles, WanDitBlockShape, WanDitBlockTaps, WanDitPipelines, WanMod,
    config,
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
    /// Spatial tokens per frame `pph * ppw` (the DF broadcast factor).
    pub spatial: usize,
}

impl WanDitShape {
    pub fn new(c: usize, f: usize, h: usize, w: usize, text_seq: usize) -> Self {
        let grid = PatchGrid::new(c, f, h, w);
        Self {
            n_tok: grid.n_tok(),
            spatial: grid.pph * grid.ppw,
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
    /// Per-frame noise levels (`len == ppf`, Diffusion Forcing).
    pub timesteps: &'a [f32],
    /// `fps_embedding` bucket (DF, `inject_sample_info`).
    pub fps: usize,
}

/// One DiT forward output, ready for VAE decode.
pub struct WanDitOutput {
    /// `[out_ch, F, H, W]` row-major f32.
    pub image: Vec<f32>,
}

/// Optional per-stage readbacks (ad-hoc bringup diff vs pyref). The committed
/// gate is the single end-state `video_e2e_parity`; these localize divergence.
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
    pub handles: LoadedWanDitHandles,
    block: WanDitBlock,
    condition_embedder: ConditionEmbedder,
    rope: WanRope3d,
}

impl WanDit {
    pub fn assemble(handles: LoadedWanDitHandles, shape: WanDitShape) -> Self {
        let block = WanDitBlock::new(WanDitBlockShape::new(shape.n_tok, shape.text_seq));
        Self {
            shape,
            handles,
            block,
            condition_embedder: ConditionEmbedder::skyreels_df(),
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
        let inner = config::INNER as u32;
        let rows = s.n_tok as u32;
        let ppf = s.grid.ppf;
        let spatial = s.spatial as u32;
        let text_seq = s.text_seq as u32;
        assert_eq!(
            inputs.timesteps.len(),
            ppf,
            "Diffusion Forcing needs one timestep per latent frame"
        );

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
        let temb = scratch.alloc(bp.act_bytes(ppf as u32 * inner))?;
        let tproj = scratch.alloc(bp.act_bytes(ppf as u32 * 6 * inner))?;
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
                inputs.timesteps,
                inputs.fps,
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
            (ppf as u32 * inner) as usize,
            bp.act_dtype,
            &mut taps.temb,
        )
        .await?;
        read_tap(
            backend,
            &tproj.as_buf_ref(),
            (ppf as u32 * 6 * inner) as usize,
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

        // --- 4. broadcast the 6 modulation bases + temb to per-token [rows, *] ---
        // Each base is `timestep_proj[:, k]` (or temb) repeated over spatial.
        let mod_base: Vec<WsBuf<WgpuBackend>> = (0..6)
            .map(|_| scratch.alloc(bp.act_bytes(rows * inner)))
            .collect::<Result<_, _>>()?;
        let temb_tok = scratch.alloc(bp.act_bytes(rows * inner))?;
        {
            let scope = scratch.batch();
            let elem = bp.act_dtype.bytes_per_elem();
            let row_b = inner as u64 * elem;
            let tproj_h = scope.import_copy(tproj.as_buf_ref());
            let temb_h = scope.import_copy(temb.as_buf_ref());
            for (k, base) in mod_base.iter().enumerate() {
                let dst = scope.import_copy(base.as_buf_ref());
                for t in 0..rows as u64 {
                    let frame = t / spatial as u64;
                    let src_off = (frame * 6 + k as u64) * row_b;
                    scope.copy_buffer_to_buffer(tproj_h, src_off, dst, t * row_b, row_b)?;
                }
            }
            let temb_dst = scope.import_copy(temb_tok.as_buf_ref());
            for t in 0..rows as u64 {
                let frame = t / spatial as u64;
                scope.copy_buffer_to_buffer(temb_h, frame * row_b, temb_dst, t * row_b, row_b)?;
            }
            scope.submit_void().await?;
        }

        // --- 5. main transformer blocks (residency-paged) ---
        let mut x_cur = x;
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
                let scope = scratch.batch();
                // Build the six per-token modulation signals: base_k + table[k].
                let table = scope.import_copy(views.scale_shift_table());
                let m = self.build_mod(&scope, bp, &mod_base, table, rows, inner)?;
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
                let submit_fut = scope
                    .submit_deferred()
                    .instrument(tracing::debug_span!(target: PHASE, "wan.submit", idx));
                let (s_res, n_res, p_res) =
                    futures::join!(submit_fut, next_acquire, prefetch_after);
                s_res?;
                p_res?;
                pending = n_res?;
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
        let proj_w = config::OUT_CHANNELS * config::PATCH_T * config::PATCH_H * config::PATCH_W;
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
            // shift = table[0] + temb_tok ; scale = table[1] + temb_tok.
            let table = scope.import_copy(sst.buf());
            let temb_h = ActBuf::dense(scope.import_copy(temb_tok.as_buf_ref()));
            let shift = self.add_table_row(&scope, bp, temb_h, table, 0, rows, inner)?;
            let scale = self.add_table_row(&scope, bp, temb_h, table, 1, rows, inner)?;
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
        let image = patchify::unpatchify(&tokens_out, &s.grid, config::OUT_CHANNELS);
        Ok(WanDitOutput { image })
    }

    /// Build the six per-token modulation signals `base_k + scale_shift_table[k]`
    /// (`table` is the block's `[6, inner]` weight, imported into `scope`).
    fn build_mod<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        bp: &BlockPipelines,
        mod_base: &[WsBuf<WgpuBackend>],
        table: BatchBuf<'wsp>,
        rows: u32,
        inner: u32,
    ) -> Result<WanMod<'wsp>, WgpuError> {
        let mut out = Vec::with_capacity(6);
        for (k, base) in mod_base.iter().enumerate() {
            let base_h = ActBuf::dense(scope.import_copy(base.as_buf_ref()));
            out.push(self.add_table_row(scope, bp, base_h, table, k, rows, inner)?);
        }
        Ok(WanMod {
            shift_msa: out[0],
            scale_msa: out[1],
            gate_msa: out[2],
            c_shift_mlp: out[3],
            c_scale_mlp: out[4],
            c_gate_mlp: out[5],
        })
    }

    /// `out = x + table[k]` where `table` is `[n, inner]` and row `k` broadcasts
    /// over the `rows` of `x` (channel-broadcast bias add on row `k`).
    #[allow(clippy::too_many_arguments)]
    fn add_table_row<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        bp: &BlockPipelines,
        x: ActBuf<'wsp>,
        table: BatchBuf<'wsp>,
        k: usize,
        rows: u32,
        inner: u32,
    ) -> Result<BatchBuf<'wsp>, WgpuError> {
        // Copy row k of the table into an [inner] scratch, then bias-add.
        let row = alloc_act(scope, bp, 1, inner)?;
        let row_b = bp.act_bytes(inner);
        scope.copy_buffer_to_buffer(table, k as u64 * row_b, row.data, 0, row_b)?;
        let dst = alloc_act(scope, bp, rows, inner)?;
        let u = scope.u32x4_uniform(inner, 0, 0, 0)?;
        scope.bcast_add::<BcastAddF32>(
            &bp.bcast_add,
            x.data,
            row.data,
            u,
            dst.data,
            rows * inner,
        )?;
        Ok(dst.data)
    }

    /// `out = x * (1 + scale) + shift` (full-elementwise, per-token modulation).
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
        // Distinct scratch per stage: elementwise ops bind input read-only and
        // output read-write, so input and output must not alias the same buffer.
        let xs = alloc_act(scope, bp, rows, inner)?;
        scope.dispatch_op::<MulF32>(&bp.mul, &[x.data, scale], xs.data)?; // x * scale
        let xss = alloc_act(scope, bp, rows, inner)?;
        op_add(scope, bp, xs, ActBuf::dense(shift), xss)?; // x * scale + shift
        op_add(scope, bp, x, xss, dst)?; // x + (x * scale + shift) = x * (1 + scale) + shift
        Ok(())
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
        assert_eq!(sh.spatial, 64);
    }
}
