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
use crate::wan::kv_cache::{ChunkPlan, KvStore};

/// Per-layer clean-pass K/V commit sinks for [`WanDit::forward_ar`]: when `Some`,
/// the chunk's roped-k (`.0`) and v (`.1`) bytes are read back per layer (one
/// `Vec<u8>` per DiT layer) so the cache tail can be committed. `None` on the 4
/// denoise steps (read-only window).
type ChunkKvCommit<'a> = Option<(&'a mut Vec<Vec<u8>>, &'a mut Vec<Vec<u8>>)>;

use crate::wan::patchify::{self, PatchGrid};
use crate::wan::rope3d::WanRope3d;

/// Per-request host cache of each block's cross-attention text K/V:
/// `(norm_k(to_k(text) + b_k), to_v(text) + b_v)` raw act bytes, one pair
/// per DiT layer, each `[text_seq, inner]`. The umT5 text states are constant
/// for the whole request, so the first forward that runs a layer computes the
/// pair in-scope (the block's weights are already resident) and reads it back;
/// every later forward replays the exact bytes, skipping the two `[text_seq,
/// dim] @ [inner, dim]T` projections + k-norm per layer. The byte round-trip
/// keeps the SDPA inputs bit-identical. Used by [`WanDit::forward_ar`] (one
/// cache per unique prompt) and by the tiled denoise step loop via
/// [`WanDit::forward_cached`] (one cache per DiT weight set - the K/V depend
/// on the block weights, so MoE experts must not share one). Sizing:
/// `num_layers * 2 * text_seq * inner * act_bytes` host RAM (~189 MB at 30
/// layers, 512 x 3072 f16; ~420 MB at 40 layers, 512 x 5120); dropped with
/// the request.
pub struct WanCrossKvCache {
    layers: Vec<Option<(Vec<u8>, Vec<u8>)>>,
}

impl WanCrossKvCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: vec![None; num_layers],
        }
    }
}

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
    /// DreamID-V only (`cfg.ref_conv`): `ref_conv` Conv2d `[inner, out_ch, p_h,
    /// p_w]` folded to a linear `[inner, out_ch*p_h*p_w]` + bias. Patchifies the
    /// source-face latent into prefix tokens. `None` on every plain Wan model.
    pub ref_conv: Option<LinearBiasHandles>,
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
    /// DreamID-V source-face latent `[out_channels, 1, h, w]` row-major f32 (the
    /// single-frame reference the `ref_conv` patchifies into prefix tokens).
    /// `Some` on every DreamID-V forward (`cfg.ref_conv`), the negative pass
    /// feeding a zeros latent; `None` on every plain Wan model.
    pub img_ref: Option<&'a [f32]>,
    /// umT5 text states `[text_seq, text_dim]` row-major f32.
    pub text: &'a [f32],
    /// AnyFlow flow-map TARGET timestep (`r`, the sigma this forward's
    /// velocity integrates to). `Some` iff the config's `delta_embedder` is
    /// set; `None` on the plain Wan models.
    pub r_timestep: Option<f32>,
    /// Scalar diffusion timestep, uniform over the whole clip (the distilled
    /// T2V line is plain flow-matching, not per-frame Diffusion Forcing).
    pub timestep: f32,
    /// Temporal self-attention window radius in latent frames. `Some(W)`
    /// restricts each query to keys within `±W` latent frames (the O(frames^2)
    /// attack for long clips); `None` runs full self-attention. Set per run from
    /// the `--attn-window` flag.
    pub attn_window: Option<u32>,
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
    /// Per-op taps inside block `tap_block` (default 0). Lets the diag dump
    /// localize the late-block drift onset (~block 17-18) by tapping that block's
    /// sub-ops, fed the captured `per_block[tap_block - 1]` residual as input.
    pub block0: Option<WanDitBlockTaps>,
    /// Which block the `block0` sub-op taps apply to (default 0).
    pub tap_block: usize,
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

/// Boxed completion fence carried ACROSS blocks: block N's last pass-B submit,
/// awaited inside block N+1's pass-A pipeline instead of drained at the end of
/// block N (`forward_block_tiled`). Eliminates the per-block GPU drain: block
/// N+1's setup/pass-A submits land while block N's tail executes. The future
/// owns the submitting scope's workspace guards (and, via the caller's
/// wrapper, the retiring residual-stream input), so it MUST be awaited --
/// dropping it un-awaited would release pool buffers the GPU may still read.
type TailFence<'w> =
    core::pin::Pin<Box<dyn core::future::Future<Output = Result<(), WgpuError>> + 'w>>;

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
        // DreamID-V (`cfg.ref_conv`) prepends `ref_rows = pph * ppw` source-face
        // tokens to the sequence, so the block runs over the full (ref ++ video)
        // row count. `ref_rows == 0` on every plain Wan model, leaving the block
        // shape (and thus its forward) bit-identical.
        let ref_rows = if cfg.ref_conv {
            shape.grid.pph * shape.grid.ppw
        } else {
            0
        };
        let block = WanDitBlock::new(WanDitBlockShape::new(
            &cfg,
            shape.n_tok + ref_rows,
            shape.text_seq,
        ));
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
            None,
        )
        .await
    }

    /// Like [`Self::forward`], threading the per-request cross-attention K/V
    /// cache through the tiled block loop: the first forward fills it, every
    /// later forward replays the request-constant bytes (bit-identical; see
    /// [`WanCrossKvCache`]). Pass one cache per DiT weight set for the whole
    /// step loop. The untiled tier ignores it (single-scope path unchanged).
    #[allow(clippy::too_many_arguments)]
    pub async fn forward_cached<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipelines: &WanDitPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        inputs: &WanDitInputs<'_>,
        cross_kv: &mut WanCrossKvCache,
    ) -> Result<WanDitOutput, WanDitError<S::Error>> {
        self.forward_with_taps(
            backend,
            pipelines,
            residency,
            scratch,
            inputs,
            WanDitTaps::default(),
            Some(cross_kv),
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
        mut cross_kv: Option<&mut WanCrossKvCache>,
    ) -> Result<WanDitOutput, WanDitError<S::Error>> {
        let bp = &pipelines.block;
        let s = self.shape;
        let inner = self.cfg.inner() as u32;
        let ppf = s.grid.ppf;
        let text_seq = s.text_seq as u32;
        // DreamID-V prefix geometry: `ref_rows` source-face tokens ride in front
        // of the `n_tok` video patch tokens, and RoPE runs over the grown grid
        // `(ppf + 1, pph, ppw)` (ref = frame 0, video = frames 1..ppf). Both are
        // zero / identity on every plain Wan model, so `rows == n_tok` and the
        // whole forward stays bit-identical there.
        let n_tok = s.n_tok as u32;
        let ref_rows = if self.cfg.ref_conv {
            (s.grid.pph * s.grid.ppw) as u32
        } else {
            0
        };
        let rows = n_tok + ref_rows;
        let ppf_full = ppf + if self.cfg.ref_conv { 1 } else { 0 };

        // --- 1. patchify image + front-door linear -> x [rows, inner] ---
        // The video patch-embed output lands AFTER the ref prefix (offset 0 when
        // no ref); the ref_conv prefix tokens are written to rows `0..ref_rows`.
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
                n_tok,
                patch_in,
                inner,
                scope.import_copy(act_slice(x.as_act_ref(), ref_rows, n_tok, inner, bp)),
            )?;
            scope.submit_void().await?;
        }
        // DreamID-V: patchify the single-frame source-face latent spatially and
        // project it through `ref_conv` (a 2x2 conv folded to a linear over
        // `out_channels * p_h * p_w`), writing the prefix tokens to rows
        // `0..ref_rows`. `ref_conv` is `None` on every plain Wan model.
        if let Some(ref_conv) = &self.handles.ref_conv {
            let img_ref = inputs
                .img_ref
                .expect("DreamID-V forward requires img_ref when ref_conv is set");
            let ref_grid = PatchGrid::new(self.cfg.out_channels, 1, s.grid.h, s.grid.w);
            let ref_tokens = patchify::patchify(img_ref, &ref_grid);
            let ref_in = ref_grid.patch_in() as u32;
            let ref_bytes = act_upload_bytes(bp.act_dtype, &ref_tokens);
            let ref_buf = scratch.alloc(ref_bytes.len() as u64)?;
            backend.write_buffer(ref_buf.id, 0, &ref_bytes)?;
            let views = ref_conv.acquire(residency, backend).await?;
            let scope = scratch.batch();
            self.linear_bias(
                &scope,
                bp,
                scope.import_copy(ref_buf.as_buf_ref()),
                &views.bufs(),
                ref_rows,
                ref_in,
                inner,
                scope.import_copy(act_slice(x.as_act_ref(), 0, ref_rows, inner, bp)),
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
                inputs.r_timestep,
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
        let freqs_bytes = freqs_upload_bytes(
            bp.act_dtype,
            &self.rope.lookup(ppf_full, s.grid.pph, s.grid.ppw),
        );
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
        // DreamID-V runs untiled: the activation-tiled path derives its geometry
        // (rows, self-attn period) from `self.shape.n_tok`, which excludes the
        // ref prefix. Parity first; the prefix-aware tiled path is a perf
        // follow-up. Plain Wan models are unaffected (`ref_conv` is false).
        let n_tiles = if taps.block0.is_some() || self.cfg.ref_conv {
            1
        } else {
            rows.div_ceil(dit_tile_rows()).max(1)
        };
        let tile = if n_tiles > 1 {
            let tb = TileBufs::alloc(scratch, bp, rows, text_seq, inner)?;
            // Per-block prepared site weights (dequant once per block; every
            // tile reuses them). Shapes are block-invariant, so the buffers
            // are allocated once here and refilled inside the block loop.
            let ptw = super::dit_block::PreparedTileWeights::alloc(scratch, bp, &self.block.shape)?;
            Some((tb, ptw))
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
        // Cross-block tail fence (tiled path): block N's last pass-B submit,
        // awaited inside block N+1's pass-A pipeline. See `TailFence`.
        let mut carried: Option<TailFence<'_>> = None;
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
            let block_taps = if idx == taps.tap_block {
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
                if let Some((tb, ptw)) = tile.as_ref() {
                    // Tiled path owns its own (serial) submits; overlap weight
                    // streaming for the next block(s) with the whole movement.
                    let block_bufs = views.bufs();
                    let sst = views.scale_shift_table();
                    let ckv_slot = cross_kv.as_deref_mut().map(|c| &mut c.layers[idx]);
                    let compute = self
                        .forward_block_tiled(
                            backend,
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
                            ptw,
                            n_tiles,
                            inputs.attn_window.unwrap_or(0),
                            ckv_slot,
                            carried.take(),
                        )
                        .instrument(tracing::debug_span!(target: PHASE, "wan.tiled", idx));
                    let (c_res, n_res, p_res) =
                        futures::join!(compute, next_acquire, prefetch_after);
                    let tail = c_res?;
                    p_res?;
                    pending = n_res?;
                    // Carry this block's tail fence into the next block; the
                    // retiring input buffer rides inside it (pool reuse only
                    // after the GPU is done reading it). The host-readback
                    // per-block taps need the tail complete -> drain inline.
                    let prev_x = std::mem::replace(&mut x_cur, nxt);
                    match tail {
                        Some(t) if taps.per_block.is_none() => {
                            carried = Some(Box::pin(async move {
                                let _hold = prev_x;
                                t.await
                            }));
                        }
                        Some(t) => t.await?,
                        None => {}
                    }
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
                    x_cur = nxt;
                }
            }
        }
        // Drain the last block's carried tail before anything reads x_cur on
        // the host (and so its workspace guards resolve before the head).
        if let Some(t) = carried.take() {
            t.await?;
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

        // --- 6. strip the ref prefix, then final norm + modulation + proj_out ---
        // The head runs on the VIDEO rows only: DreamID-V slices off the `ref_rows`
        // source-face prefix (`x[ref_rows..]`); on plain Wan models `ref_rows == 0`
        // and `head_rows == rows`, so `x_head` is the whole stream (bit-identical).
        let head_rows = n_tok;
        let x_head = act_slice(x_cur.as_act_ref(), ref_rows, head_rows, inner, bp);
        let proj_w = self.cfg.out_channels * config::PATCH_T * config::PATCH_H * config::PATCH_W;
        let proj_out = ResStream::alloc(scratch, bp, head_rows, proj_w as u32)?;
        // Persist the post-modulation activation when a tap wants it (readback
        // after submit; scope-local buffers do not survive the submit).
        let final_norm_ws = match taps.final_norm {
            Some(_) => Some(scratch.alloc(bp.act_bytes(head_rows * inner))?),
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
            let x_h = ActBuf::dense(scope.import_copy(x_head));
            let normed = alloc_act(&scope, bp, head_rows, inner)?;
            let ln_u = scope.u32x4_uniform(head_rows, inner, config::EPS.to_bits(), 0)?;
            scope.layernorm::<LayerNormF32>(
                &bp.layernorm,
                x_h.data,
                ln_u,
                normed.data,
                head_rows,
            )?;
            // out = normed * (1 + scale) + shift.
            let modded = alloc_act(&scope, bp, head_rows, inner)?;
            self.modulate(&scope, bp, normed, scale, shift, modded, head_rows, inner)?;
            if let Some(fnw) = &final_norm_ws {
                let dst = scope.import_copy(fnw.as_buf_ref());
                scope.copy_buffer_to_buffer(
                    modded.data,
                    0,
                    dst,
                    0,
                    bp.act_bytes(head_rows * inner),
                )?;
            }
            // proj_out: [head_rows, inner] @ [proj_w, inner]ᵀ + bias.
            self.linear_bias(
                &scope,
                bp,
                modded.data,
                &pv.bufs(),
                head_rows,
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
                (head_rows * inner) as usize,
                bp.act_dtype,
                sink,
            )
            .await?;
        }
        read_tap(
            backend,
            &proj_out.as_act_ref(),
            (head_rows * proj_w as u32) as usize,
            bp.act_dtype,
            &mut taps.proj_out,
        )
        .await?;

        // --- 7. unpatchify on CPU -> [out_ch, F, H, W] ---
        let mut tokens_out = Vec::new();
        read_into_f32(
            backend,
            &proj_out.as_act_ref(),
            (head_rows * proj_w as u32) as usize,
            bp.act_dtype,
            &mut tokens_out,
        )
        .await?;
        let image = patchify::unpatchify(&tokens_out, &s.grid, self.cfg.out_channels);
        Ok(WanDitOutput { image })
    }

    /// AR (LongLive) forward for ONE chunk against the windowed KV cache. Same
    /// backbone as [`Self::forward`] (patchify, condition embedder, the 30 blocks,
    /// final norm + proj_out, unpatchify); the ONLY difference is the per-block
    /// self-attention, which runs [`WanDitBlock::self_attn_ar`] over `[committed
    /// window prefix ++ this chunk]` instead of the full-sequence self-SDPA. The
    /// chunk's q/k rotate at the chunk's absolute frame position (`plan
    /// .chunk_start_frame` + `plan.temporal_offset`, release `use_relative_rope=
    /// False`). `store` supplies the committed prefix K/V (host-resident); when
    /// `commit` is `Some`, the chunk's roped-k / v are read back per layer into it
    /// (the clean recache pass writes these to the cache tail).
    ///
    /// `cross_kv` caches the per-layer cross-attention text K/V across this
    /// request's forwards (see [`WanCrossKvCache`]); pass the same cache for
    /// every forward that shares the prompt.
    ///
    /// `self.shape` must be the per-CHUNK shape (`n_tok == chunk frames * pph *
    /// ppw`). Serial block residency (acquire -> compute -> next); the AR perf
    /// path (prefetch overlap / activation tiling) is a follow-up, the e2e gate
    /// runs at small chunk-token counts.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward_ar<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipelines: &WanDitPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        inputs: &WanDitInputs<'_>,
        store: &dyn KvStore,
        plan: &ChunkPlan,
        cross_kv: &mut WanCrossKvCache,
        mut commit: ChunkKvCommit<'_>,
        mut block_res_diag: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<WanDitOutput, WanDitError<S::Error>> {
        let bp = &pipelines.block;
        let s = self.shape;
        let inner = self.cfg.inner() as u32;
        let rows = s.n_tok as u32;
        let ppf = s.grid.ppf;
        let text_seq = s.text_seq as u32;
        let token_bytes = inner as usize * bp.act_dtype.bytes_per_elem() as usize;
        let num_layers = self.handles.blocks.len();

        // --- patchify chunk + front-door linear -> x [chunk_rows, inner] ---
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

        // --- condition embedder -> temb, timestep_proj, text ---
        let text_bytes = act_upload_bytes(bp.act_dtype, inputs.text);
        let text_in = scratch.alloc(text_bytes.len() as u64)?;
        backend.write_buffer(text_in.id, 0, &text_bytes)?;
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
                inputs.r_timestep,
                scope.import_copy(text_in.as_buf_ref()),
                text_seq,
                &out,
                &views.bufs(),
            )?;
            scope.submit_void().await?;
        }

        // --- per-chunk RoPE3D freqs at the chunk's ABSOLUTE temporal position ---
        let t_start = plan.chunk_start_frame + plan.temporal_offset as usize;
        let freqs_bytes = freqs_upload_bytes(
            bp.act_dtype,
            &self
                .rope
                .lookup_temporal(ppf, s.grid.pph, s.grid.ppw, t_start),
        );
        let freqs = scratch.alloc(freqs_bytes.len() as u64)?;
        backend.write_buffer(freqs.id, 0, &freqs_bytes)?;

        // Window geometry (token counts) from the chunk plan.
        let prefix_rows = plan.prefix_tokens as u32;
        let window_rows = plan.window_tokens as u32;
        debug_assert_eq!(plan.tail.len as u32, rows, "chunk tokens must match shape");

        // --- transformer blocks (serial residency; AR self-attn) ---
        let _lr = trace::scope!("wan.dit.ar_blocks", n = num_layers).entered();
        let mut x_cur = x;
        for idx in 0..num_layers {
            let _bs = trace::scope!(format_args!("ar_block.{idx}")).entered();
            let views = self.handles.blocks[idx]
                .acquire(residency, backend)
                .instrument(tracing::debug_span!(target: PHASE, "wan.acquire", idx))
                .await?;
            let block_bufs = views.bufs();

            let nxt = ResStream::alloc(scratch, bp, rows, inner)?;
            let window_k = scratch.alloc(bp.act_bytes(window_rows * inner))?;
            let window_v = scratch.alloc(bp.act_bytes(window_rows * inner))?;
            // Upload this layer's committed window prefix (K already roped, V
            // raw) straight into the window buffers, one write per committed
            // segment (no host gather, no GPU staging copy); the chunk's fresh
            // K/V land after it in-scope. Empty for the first chunk.
            let mut dst = 0u64;
            for seg in &plan.prefix {
                backend.write_buffer(
                    window_k.id,
                    dst,
                    &store.k(idx)[seg.byte_range(token_bytes)],
                )?;
                backend.write_buffer(
                    window_v.id,
                    dst,
                    &store.v(idx)[seg.byte_range(token_bytes)],
                )?;
                dst += (seg.len * token_bytes) as u64;
            }

            // Cross-attention text K/V: replay the request-constant bytes when
            // cached, else compute in-scope and read back after the submit.
            let ckv_bytes = bp.act_bytes(text_seq * inner);
            let cross_k = scratch.alloc(ckv_bytes)?;
            let cross_v = scratch.alloc(ckv_bytes)?;
            let ckv_cached = match &cross_kv.layers[idx] {
                Some((kb, vb)) => {
                    backend.write_buffer(cross_k.id, 0, kb)?;
                    backend.write_buffer(cross_v.id, 0, vb)?;
                    true
                }
                None => false,
            };

            let roped_k = scratch.alloc(bp.act_bytes(rows * inner))?;
            let v_out = scratch.alloc(bp.act_bytes(rows * inner))?;
            {
                let scope = scratch.batch();
                let tproj_h = scope.import_copy(tproj.as_buf_ref());
                let m = self.build_mod(&scope, bp, views.scale_shift_table(), tproj_h, inner)?;
                let x1 = scratch.alloc(bp.act_bytes(rows * inner))?;
                let cross_k_h = scope.import_copy(cross_k.as_buf_ref());
                let cross_v_h = scope.import_copy(cross_v.as_buf_ref());
                if !ckv_cached {
                    self.block.cross_kv(
                        &scope,
                        pipelines,
                        scope.import_copy(text.as_act_ref()),
                        &block_bufs.cross_attn,
                        cross_k_h,
                        cross_v_h,
                        text_seq,
                    )?;
                }
                self.block.self_attn_ar(
                    &scope,
                    pipelines,
                    scope.import_copy(x_cur.as_act_ref()),
                    scope.import_copy(freqs.as_buf_ref()),
                    &m,
                    prefix_rows,
                    scope.import_copy(window_k.as_buf_ref()),
                    scope.import_copy(window_v.as_buf_ref()),
                    scope.import_copy(roped_k.as_buf_ref()),
                    scope.import_copy(v_out.as_buf_ref()),
                    scope.import_copy(x1.as_buf_ref()),
                    rows,
                    window_rows,
                    &block_bufs.self_attn,
                )?;
                self.block.cross_ffn(
                    &scope,
                    pipelines,
                    ActBuf::dense(scope.import_copy(x1.as_buf_ref())),
                    scope.import_copy(text.as_act_ref()),
                    &m,
                    ActBuf::dense(scope.import_copy(nxt.as_act_ref())),
                    &block_bufs,
                    &WanDitBlockTaps::default(),
                    rows,
                    text_seq,
                    Some((cross_k_h, cross_v_h)),
                )?;
                scope.submit_void().await?;
            }
            if !ckv_cached {
                cross_kv.layers[idx] = Some((
                    backend.read_buffer(cross_k.id, 0, ckv_bytes).await?,
                    backend.read_buffer(cross_v.id, 0, ckv_bytes).await?,
                ));
            }
            // Clean-pass commit: read back the chunk's roped-k / v for this layer.
            if let Some((k_sink, v_sink)) = commit.as_mut() {
                let n = (rows * inner) as u64 * bp.act_dtype.bytes_per_elem();
                k_sink[idx] = backend.read_buffer(roped_k.id, 0, n).await?;
                v_sink[idx] = backend.read_buffer(v_out.id, 0, n).await?;
            }
            x_cur = nxt;
            // Per-block residual readback (localization): the token-space stream
            // [n_tok, inner] after block `idx`, matching the pyref block hooks.
            if let Some(sink) = block_res_diag.as_deref_mut() {
                let mut v = Vec::new();
                read_into_f32(
                    backend,
                    &x_cur.as_act_ref(),
                    (rows * inner) as usize,
                    bp.act_dtype,
                    &mut v,
                )
                .await?;
                sink.push(v);
            }
        }

        // --- final norm + modulation + proj_out -> velocity ---
        let proj_w = self.cfg.out_channels * config::PATCH_T * config::PATCH_H * config::PATCH_W;
        let proj_out = ResStream::alloc(scratch, bp, rows, proj_w as u32)?;
        {
            let sst = residency
                .acquire(self.handles.scale_shift_table, backend)
                .await?;
            let pv = self.handles.proj_out.acquire(residency, backend).await?;
            let scope = scratch.batch();
            let temb_h = scope.import_copy(temb.as_buf_ref());
            let shift = self.mod_signal(&scope, bp, sst.buf(), 0, temb_h, inner)?;
            let scale = self.mod_signal(&scope, bp, sst.buf(), 1, temb_h, inner)?;
            let x_h = ActBuf::dense(scope.import_copy(x_cur.as_act_ref()));
            let normed = alloc_act(&scope, bp, rows, inner)?;
            let ln_u = scope.u32x4_uniform(rows, inner, config::EPS.to_bits(), 0)?;
            scope.layernorm::<LayerNormF32>(&bp.layernorm, x_h.data, ln_u, normed.data, rows)?;
            let modded = alloc_act(&scope, bp, rows, inner)?;
            self.modulate(&scope, bp, normed, scale, shift, modded, rows, inner)?;
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
    ///
    /// `ckv_slot` is this block's [`WanCrossKvCache`] entry: a hit replays the
    /// cached ck/cv bytes (skipping the two text projections + k-norm), a miss
    /// computes them and reads the bytes back after the drain. `None` disables
    /// caching (single-forward diag paths).
    ///
    /// Returns block N's last pass-B fence (`Some`) for the caller to carry
    /// into block N+1 instead of draining here; `None` when the fn had to
    /// drain internally (first-step cross-KV persist readback).
    #[allow(clippy::too_many_arguments)]
    async fn forward_block_tiled<'w>(
        &self,
        backend: &WgpuBackend,
        pipelines: &WanDitPipelines,
        scratch: &'w Workspace<WgpuBackend>,
        bufs: &WanDitBlockBufs,
        sst: BufRef,
        tproj: BufRef,
        freqs: BufRef,
        text: BufRef,
        x_in: BufRef,
        y_out: BufRef,
        tb: &TileBufs,
        ptw: &super::dit_block::PreparedTileWeights,
        n_tiles: u32,
        window: u32,
        ckv_slot: Option<&mut Option<(Vec<u8>, Vec<u8>)>>,
        carry: Option<TailFence<'w>>,
    ) -> Result<Option<TailFence<'w>>, WgpuError> {
        let bp = &pipelines.block;
        let inner = self.cfg.inner() as u32;
        let rows = self.shape.n_tok as u32;
        let text_seq = self.shape.text_seq as u32;
        let hd = config::HEAD_DIM as u32;

        // Diag-gated per-phase wall clocks: encode (CPU dispatch building) vs
        // await (fence waits) per pass, to localize non-GPU block time.
        let diag = tracing::enabled!(target: "thinfer::diag", tracing::Level::DEBUG);
        let t0 = diag.then(trace::Instant::now);

        // Setup - one scope, submitted WITHOUT a CPU await: the six modulation
        // vectors, the cross-attn K/V (replayed from the cache when hit), and
        // this block's site weights dequanted into the persistent prepared
        // buffers (all shared by every tile in passes A and B). Queue order
        // already sequences these writes after the previous block's tile reads
        // and before this block's, so the two hard submit-await syncs the
        // setup used to pay per block (each serialized behind the prior
        // block's pass-B drain) are gone; the fence seeds the pass-A pipeline.
        let ckv_replay = matches!(ckv_slot.as_deref(), Some(Some(_)));
        if let Some(Some((kb, vb))) = ckv_slot.as_deref() {
            // Byte replay of the request-constant text K/V (bit-identical to
            // recomputing it). queue.write_buffer lands in queue order: after
            // the prior block's submitted reads, before this block's.
            let (ck, cv) = (tb.ck.as_buf_ref(), tb.cv.as_buf_ref());
            backend.write_buffer(ck.id, ck.offset, kb)?;
            backend.write_buffer(cv.id, cv.offset, vb)?;
        }
        let setup_fut = {
            let scope = scratch.batch();
            self.fill_mod(&scope, bp, sst, tproj, inner, &tb.m)?;
            if !ckv_replay {
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
            }
            self.block
                .fill_prepared_weights(&scope, pipelines, bufs, ptw)?;
            scope.submit_deferred()
        };
        let t_setup = t0.as_ref().map(|t| t.elapsed().as_secs_f64() * 1e3);

        // Pass A: per-tile q/k/v projection + RoPE into the full qx/kx/v.
        // Depth-2 submit pipeline: tile t is submitted before tile t-1's
        // completion is awaited, so the CPU encodes/submits the next tile
        // while the GPU runs the current one. The serial submit-and-wait it
        // replaces left the GPU idle between every tile - at video dims
        // (32760 rows = 32 tiles, ~67 fences per block) that idle was the
        // dominant per-block cost, bigger than the tile compute itself.
        // Depth 2 keeps at most two tiles' transients out of the pool, which
        // the DiT workspace reserve accounts for. The setup fence rides the
        // same pipeline (awaited when tile 0's submit lands), and the PRIOR
        // block's carried tail fence rides ahead of it (cross-block depth-2:
        // this block's setup/pass-A submits landed while that tail executed).
        let mut pending_tile: Option<TailFence<'w>> = Some(match carry {
            Some(c) => Box::pin(async move {
                c.await?;
                setup_fut.await
            }),
            None => Box::pin(setup_fut),
        });
        let (mut enc_a, mut await_a) = (0.0f64, 0.0f64);
        for t in 0..n_tiles {
            let te = diag.then(trace::Instant::now);
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
                ptw,
            )?;
            let fut: TailFence<'w> = Box::pin(scope.submit_deferred());
            if let Some(t) = te {
                enc_a += t.elapsed().as_secs_f64() * 1e3;
            }
            let ta = diag.then(trace::Instant::now);
            if let Some(prev) = pending_tile.replace(fut) {
                prev.await?;
            }
            if let Some(t) = ta {
                await_a += t.elapsed().as_secs_f64() * 1e3;
            }
        }
        // Barrier: global self-attention over the whole sequence. Submitted
        // before the leftover pass-A fence is awaited (queue order already
        // sequences it after every pass-A tile), so the GPU queue never runs
        // dry between the passes; its fence seeds the pass-B pipeline.
        let sdpa_fut = {
            let scope = scratch.batch();
            let qx = scope.import_copy(tb.qx.as_buf_ref());
            let kx = scope.import_copy(tb.kx.as_buf_ref());
            let v = scope.import_copy(tb.v.as_buf_ref());
            let sa = scope.import_copy(tb.sa.as_buf_ref());
            // Tokens per latent frame (frame-major `(f, h, w)`): the windowed
            // kernel's `period`. `window == 0` runs full self-attention.
            let period = (self.shape.grid.pph * self.shape.grid.ppw) as u32;
            self.block
                .self_sdpa(&scope, pipelines, qx, kx, v, sa, rows, period, window)?;
            scope.submit_deferred()
        };
        if let Some(last) = pending_tile {
            let ta = diag.then(trace::Instant::now);
            last.await?;
            if let Some(t) = ta {
                await_a += t.elapsed().as_secs_f64() * 1e3;
            }
        }
        let t_sdpa = t0.as_ref().map(|t| t.elapsed().as_secs_f64() * 1e3);

        // Pass B: per-tile o-proj + gated residual + cross-attn + FFN -> y_out.
        // Same depth-2 submit pipeline as pass A, seeded with the SDPA fence.
        let mut pending_tile: Option<TailFence<'w>> = Some(Box::pin(sdpa_fut));
        let (mut enc_b, mut await_b) = (0.0f64, 0.0f64);
        for t in 0..n_tiles {
            let te = diag.then(trace::Instant::now);
            let (r0, tr) = tile_range(rows, n_tiles, t);
            let scope = scratch.batch();
            let m = mk_mod(&scope, &tb.m);
            let x_slice = scope.import_copy(act_slice(x_in, r0, tr, inner, bp));
            let sa_slice = scope.import_copy(act_slice(tb.sa.as_buf_ref(), r0, tr, inner, bp));
            let y_slice = scope.import_copy(act_slice(y_out, r0, tr, inner, bp));
            let ck = scope.import_copy(tb.ck.as_buf_ref());
            let cv = scope.import_copy(tb.cv.as_buf_ref());
            self.block.post_attn_tile(
                &scope, pipelines, x_slice, sa_slice, &m, ck, cv, y_slice, tr, text_seq, bufs, ptw,
            )?;
            let fut: TailFence<'w> = Box::pin(scope.submit_deferred());
            if let Some(t) = te {
                enc_b += t.elapsed().as_secs_f64() * 1e3;
            }
            let ta = diag.then(trace::Instant::now);
            if let Some(prev) = pending_tile.replace(fut) {
                prev.await?;
            }
            if let Some(t) = ta {
                await_b += t.elapsed().as_secs_f64() * 1e3;
            }
        }
        // Cache miss: persist the just-computed ck/cv bytes for the later
        // steps' replay. This is the one case that must DRAIN the tail here
        // (the readback needs the ck/cv writes complete); first-forward-only
        // cost. Otherwise the tail fence is RETURNED for the caller to carry
        // into the next block's pass-A pipeline (no per-block GPU drain).
        if let Some(slot) = ckv_slot
            && slot.is_none()
        {
            if let Some(last) = pending_tile.take() {
                let ta = diag.then(trace::Instant::now);
                last.await?;
                if let Some(t) = ta {
                    await_b += t.elapsed().as_secs_f64() * 1e3;
                }
            }
            let (ck, cv) = (tb.ck.as_buf_ref(), tb.cv.as_buf_ref());
            let n = bp.act_bytes(text_seq * inner);
            *slot = Some((
                backend.read_buffer(ck.id, ck.offset, n).await?,
                backend.read_buffer(cv.id, cv.offset, n).await?,
            ));
        }
        if let (Some(setup), Some(sdpa), Some(t)) = (t_setup, t_sdpa, t0.as_ref()) {
            let total = t.elapsed().as_secs_f64() * 1e3;
            tracing::debug!(
                target: "thinfer::diag",
                setup_enc_ms = setup,
                ckv_replay,
                pass_a_enc_ms = enc_a,
                pass_a_await_ms = await_a,
                sdpa_ms = sdpa - setup - enc_a - await_a,
                pass_b_enc_ms = enc_b,
                pass_b_await_ms = await_b,
                total_ms = total,
                other_ms = total - sdpa - enc_b - await_b,
                "tiled block phases"
            );
        }
        Ok(pending_tile)
    }

    /// Encode the six modulation vectors for one block into the persistent
    /// `[inner]` buffers `m` (so the tile scopes can broadcast them). Same
    /// `scale_shift_table[k] + timestep_proj[k]` sum as [`Self::build_mod`];
    /// dispatched into the caller's (setup) scope.
    fn fill_mod(
        &self,
        scope: &BatchScope<'_, WgpuBackend>,
        bp: &BlockPipelines,
        sst: BufRef,
        tproj: BufRef,
        inner: u32,
        m: &[WsBuf<WgpuBackend>; 6],
    ) -> Result<(), WgpuError> {
        let tproj_h = scope.import_copy(tproj);
        let built = self.build_mod(scope, bp, sst, tproj_h, inner)?;
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
        Ok(())
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
            &bp.matmul_module,
            &bp.matmuls.module,
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
