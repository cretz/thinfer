//! Z-Image DiT stack driver: composes XEmbedder + noise_refiner (2) + concat
//! with CapEmbedder + context_refiner (2) + main layers (30) + FinalLayer.
//!
//! Mirrors `ZImageTransformer2DModel.forward` (`src/zimage/transformer.py`).
//! Single-batch (B=1) v1: one image, one caption per forward. The
//! `next_work` scheduler integration is deferred; the driver pages weights
//! per-module via `WeightResidency`, owns one `BatchScope` per module,
//! submits + awaits per-module so weights can be evicted between modules.
//! Workspace activations persist across submits within one DiT forward as
//! caller-owned `WsBuf`s; each scope `import`s them as needed.

use std::collections::VecDeque;

use std::future::Future;
use std::pin::Pin;

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError};
use thinfer_core::mem::VramCategory;
use thinfer_core::ops::{ActDtype, ScatterPadRowsF32};
use thinfer_core::residency::{ResidencyError, WeightResidency};
use thinfer_core::trace::{self, PHASE};
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::{ScopePacker, Workspace, WsBuf};
use tracing::Instrument;

use crate::z_image::block::{
    ActBufRef, Block, BlockConfig, BlockDebugTaps, BlockHandles, BlockPipelines,
};

/// Residual-stream buffer: a dense activation buffer at the pipeline's
/// native dtype. Mirrors `block::ActBuf` at the WsBuf (cross-submit) level.
struct ResStream {
    data: WsBuf<WgpuBackend>,
}

impl ResStream {
    /// Allocate one residual-stream buffer sized for `rows * dim` acts.
    fn alloc(
        scratch: &Workspace<WgpuBackend>,
        pipelines: &BlockPipelines,
        rows: u32,
        dim: u32,
    ) -> Result<Self, WgpuError> {
        let data = scratch.alloc(pipelines.act_bytes(rows * dim))?;
        Ok(Self { data })
    }

    /// Build the `ActBufRef` view that crosses into `BatchScope` imports.
    fn as_act_ref(&self) -> ActBufRef {
        ActBufRef::dense(self.data.as_buf_ref())
    }
}

/// Import an `ActBufRef` into a `BatchScope`, producing a dense
/// `ActBuf<'wsp>`.
fn import_act<'wsp>(
    scope: &thinfer_core::workspace::BatchScope<'wsp, WgpuBackend>,
    r: ActBufRef,
) -> crate::z_image::block::ActBuf<'wsp> {
    crate::z_image::block::ActBuf::dense(scope.import_copy(r.data))
}
use crate::z_image::config;
use crate::z_image::embedders::{
    CapEmbedder, CapEmbedderConfig, CapEmbedderHandles, LinearBiasHandles, XEmbedder,
    XEmbedderConfig,
};
use crate::z_image::final_layer::{FinalLayer, FinalLayerConfig, FinalLayerHandles};
use crate::z_image::loader::LoadedDitHandles;
use crate::z_image::rope_embedder::RopeEmbedder;
use crate::z_image::seq;
use crate::z_image::t_embedder::{
    TEmbedderWeightHandles, TimestepEmbedder, TimestepEmbedderConfig,
};

pub struct ZImageDit {
    pub x_embedder: XEmbedder,
    pub x_embedder_handles: LinearBiasHandles,
    pub cap_embedder: CapEmbedder,
    pub cap_embedder_handles: CapEmbedderHandles,
    pub t_embedder: TimestepEmbedder,
    pub t_embedder_handles: TEmbedderWeightHandles,
    pub noise_refiner: Vec<Block>,
    pub noise_refiner_handles: Vec<BlockHandles>,
    pub context_refiner: Vec<Block>,
    pub context_refiner_handles: Vec<BlockHandles>,
    pub layers: Vec<Block>,
    pub layers_handles: Vec<BlockHandles>,
    pub final_layer: FinalLayer,
    pub final_layer_handles: FinalLayerHandles,
    pub rope: RopeEmbedder,
    /// `[1, dim]` learned pad token for image positions.
    pub x_pad_token: thinfer_core::residency::WeightHandle,
    /// `[1, dim]` learned pad token for caption positions.
    pub cap_pad_token: thinfer_core::residency::WeightHandle,
}

/// Caller-provided per-forward inputs. All host-side; the driver uploads.
pub struct DitInputs<'a> {
    /// `[C, F, H, W]` row-major f32.
    pub image: &'a [f32],
    pub size: (usize, usize, usize, usize), // (C, F, H, W)
    /// `[cap_len, cap_feat_dim]` row-major f32.
    pub cap_feats: &'a [f32],
    pub cap_len: usize,
    /// Scalar timestep (will be multiplied by `t_scale` inside).
    pub timestep: f32,
    pub patch_size: usize,
    pub f_patch_size: usize,
}

/// Optional intermediate readbacks for the `dit_parity` narrowing test.
#[derive(Default)]
pub struct DitTaps<'a> {
    pub cap_embed_normed: Option<&'a mut Vec<f32>>,
    pub cap_embed_pre_bias: Option<&'a mut Vec<f32>>,
    pub cap_embedded: Option<&'a mut Vec<f32>>,
    pub main_layer_0_out: Option<&'a mut Vec<f32>>,
    pub main_layer_14_out: Option<&'a mut Vec<f32>>,
    pub ctx_refiner_0_out: Option<&'a mut Vec<f32>>,
    pub last_ctx_refiner_out: Option<&'a mut Vec<f32>>,
    pub unified_in: Option<&'a mut Vec<f32>>,
    pub last_main_layer_out: Option<&'a mut Vec<f32>>,
    pub final_layer_out: Option<&'a mut Vec<f32>>,
    /// Per-op readbacks inside `dit.layers.0`. When `Some`, forward_taps
    /// is called for the first main block with scratch-allocated tap
    /// buffers, then read back to f32 here. Fields left `None` skip that
    /// tap. Used to narrow which op first produces NaN under a new
    /// weight encoding (e.g. Q8_0).
    pub block0: Option<&'a mut Block0Taps>,
    /// Per-op readbacks inside `context_refiner.0` (first ctx-refiner block,
    /// modulation=false). Same struct as [`Block0Taps`]; adaln_* fields stay
    /// `None`-shaped after the run since this block has no adaln path.
    pub ctx_block0: Option<&'a mut Block0Taps>,
    /// Per-op readbacks inside the LAST main DiT block (block N-1). Same
    /// struct shape as block0; fields populated by the same machinery.
    /// DIAG: lets us compare per-quant-event slope at first vs last block
    /// to see if loss-per-event grows with heavy-tail activations.
    pub block_last: Option<&'a mut Block0Taps>,
    /// DIAG: per-op readbacks at arbitrary main-block indices, keyed by
    /// `(block_idx, taps)`. Engine matches each iter idx against the
    /// keys; matched blocks get the same per-op tap treatment as
    /// `block0`/`block_last`. Used to instrument the damage-zone blocks
    /// (24-28) without growing `DitTaps` with per-block fields. Block
    /// indices that match neither 0, last, nor any extra are skipped
    /// (zero-cost).
    pub extra_blocks: Option<&'a mut [(usize, Block0Taps)]>,
    /// DIAG: full residual stream readback after every main block. When
    /// `Some`, engine forces sync-submit at every block boundary and
    /// reads `u_cur` to f32. Length = number of main blocks (30). Empty
    /// sub-vecs for indices that didn't fire (e.g. early-return paths).
    pub per_main_block_residual: Option<&'a mut Vec<Vec<f32>>>,
}

/// Selects which `Block0Taps` sink (if any) the engine reads back into
/// after a given main-block iteration. `None` means the iteration does
/// not produce per-op taps.
enum ProbeRoute {
    None,
    Block0,
    BlockLast,
    Extra(usize),
}

/// Per-op readbacks inside `dit.layers.0` (first DiT main block). All values
/// are decoded to f32 regardless of the kernel's activation dtype.
#[derive(Default)]
pub struct Block0Taps {
    pub adaln_input: Option<Vec<f32>>,
    pub adaln_pre: Option<Vec<f32>>,
    pub adaln_full: Option<Vec<f32>>,
    pub scale_msa: Option<Vec<f32>>,
    pub gate_msa: Option<Vec<f32>>,
    pub scale_mlp: Option<Vec<f32>>,
    pub gate_mlp: Option<Vec<f32>>,
    pub attn_norm1_out: Option<Vec<f32>>,
    pub modulated_attn_in: Option<Vec<f32>>,
    pub attn_q: Option<Vec<f32>>,
    pub attn_k: Option<Vec<f32>>,
    pub attn_v: Option<Vec<f32>>,
    pub attn_q_norm: Option<Vec<f32>>,
    pub attn_k_norm: Option<Vec<f32>>,
    pub attn_q_rope: Option<Vec<f32>>,
    pub attn_k_rope: Option<Vec<f32>>,
    pub attn_sdpa: Option<Vec<f32>>,
    pub attn_out: Option<Vec<f32>>,
    pub attn_norm2_out: Option<Vec<f32>>,
    pub x_mid: Option<Vec<f32>>,
    pub ffn_norm1_out: Option<Vec<f32>>,
    pub modulated_ffn_in: Option<Vec<f32>>,
    pub ffn_raw: Option<Vec<f32>>,
    pub ffn_norm2_out: Option<Vec<f32>>,
    /// Raw f16 matmul output written by `matmul_i8_qkv` before `act_quant`
    /// transcodes it to the paired (i8, scale) form. Decoded f16 -> f32.
    /// Lets us see whether matmul_i8 itself is producing the wrong magnitude
    /// vs. the act_quant / paired-readback path.
    pub attn_qkv_f16_pre_quant: Option<Vec<f32>>,
    /// Same idea for the attention output projection (`matmul_i8_proj`).
    pub attn_proj_f16_pre_quant: Option<Vec<f32>>,
    /// Pre-act_quant f16 scratch from FFN w1 matmul (decoded to f32).
    pub ffn_h1_f16_pre_quant: Option<Vec<f32>>,
    /// Pre-act_quant f16 scratch from FFN w3 matmul.
    pub ffn_h3_f16_pre_quant: Option<Vec<f32>>,
    /// Pre-act_quant f16 scratch from FFN w2 matmul.
    pub ffn_h2_f16_pre_quant: Option<Vec<f32>>,
    /// DIAG raw byte snapshots (i8 / f32 packed as bytes).
    pub proj_sa_data_head: Option<Vec<u8>>,
    pub proj_sa_scale_head: Option<Vec<u8>>,
    pub proj_wo_b_i8_head: Option<Vec<u8>>,
    pub proj_wo_b_scale_head: Option<Vec<u8>>,
    /// DIAG raw byte snapshots at the QKV-matmul site under I8 acts. Used
    /// by the parity test's block-26 byte-level matmul_i8 audit to CPU-
    /// recompute one output element from the actual operand bytes and
    /// cross-check against the GPU's `attn_qkv_f16_pre_quant`. Captured
    /// post-act_quant for acts and post-dequant_i8 for weights, so the
    /// formula audited is exactly `acc = dot(a_i8, b_i8)*sa*sb + za*sb*b_qsum`
    /// with sa/za read from `a_params` (vec2<f16>), sb from `b_scale` (f16),
    /// b_qsum from `b_qsum` (f32).
    pub qkv_attn_in_data_head: Option<Vec<u8>>,
    pub qkv_attn_in_params_head: Option<Vec<u8>>,
    pub qkv_b_i8_head: Option<Vec<u8>>,
    pub qkv_b_scale_head: Option<Vec<u8>>,
    pub qkv_b_qsum_head: Option<Vec<u8>>,
    /// (k/32) K-blocks * 8 f32 (960 f32 at dim=3840) + 16-f32 probe area.
    /// Per K-block: (sa, za, sb, qsum, dot, main, corr, acc_running).
    pub qkv_dbg_trace_head: Option<Vec<u8>>,
}

/// Result of one DiT forward, ready for VAE decode.
pub struct DitOutput {
    /// `[out_channels, F, H, W]` row-major f32.
    pub image: Vec<f32>,
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

/// Runtime parameters that determine block sequence lengths.
#[derive(Clone, Copy, Debug)]
pub struct DitShape {
    pub patch_in: usize,
    pub out_channels: usize,
    pub seq_x: usize,
    pub seq_cap: usize,
}

impl DitShape {
    pub fn for_image(
        c_lat: usize,
        h_lat: usize,
        w_lat: usize,
        cap_ori_len: usize,
        patch_size: usize,
        f_patch_size: usize,
    ) -> Self {
        let n_tok = (h_lat / patch_size) * (w_lat / patch_size); // F=1
        let pad = seq::pad_len(n_tok);
        let seq_x = n_tok + pad;
        let cap_padded = cap_ori_len + seq::pad_len(cap_ori_len);
        let patch_in = patch_size * patch_size * f_patch_size * c_lat;
        Self {
            patch_in,
            out_channels: patch_in,
            seq_x,
            seq_cap: cap_padded,
        }
    }
}

impl ZImageDit {
    pub fn assemble(handles: LoadedDitHandles, shape: DitShape) -> Self {
        let dim = config::DIM;
        let head_dim = config::HEAD_DIM;
        let n_heads = config::N_HEADS;
        let n_kv_heads = config::N_KV_HEADS;
        let ffn = config::FFN_HIDDEN;
        let eps = config::NORM_EPS;
        let aed = config::ADALN_EMBED_DIM;
        let seq_u = shape.seq_x + shape.seq_cap;

        let block_cfg = |seq: usize, modulation: bool| BlockConfig {
            dim,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_hidden: ffn,
            batch: 1,
            seq,
            norm_eps: eps,
            adaln_embed_dim: aed,
            modulation,
        };

        let noise_refiner = (0..config::N_REFINER_LAYERS)
            .map(|_| Block::new(block_cfg(shape.seq_x, true)))
            .collect::<Vec<_>>();
        let context_refiner = (0..config::N_REFINER_LAYERS)
            .map(|_| Block::new(block_cfg(shape.seq_cap, false)))
            .collect::<Vec<_>>();
        let layers = (0..config::N_LAYERS)
            .map(|_| Block::new(block_cfg(seq_u, true)))
            .collect::<Vec<_>>();

        let x_embedder = XEmbedder::new(XEmbedderConfig {
            patch_in: shape.patch_in,
            dim,
        });
        let cap_embedder = CapEmbedder::new(CapEmbedderConfig {
            cap_feat_dim: config::CAP_FEAT_DIM,
            dim,
            norm_eps: eps,
        });
        let t_embedder = TimestepEmbedder::new(TimestepEmbedderConfig::z_image());
        let final_layer = FinalLayer::new(FinalLayerConfig {
            dim,
            adaln_embed_dim: aed,
            out_channels: shape.out_channels,
            norm_eps: 1e-6,
        });
        let rope = RopeEmbedder::new(
            config::ROPE_THETA,
            config::ROPE_AXES_DIMS,
            config::ROPE_AXES_LENS,
        );

        Self {
            x_embedder,
            x_embedder_handles: handles.x_embedder,
            cap_embedder,
            cap_embedder_handles: handles.cap_embedder,
            t_embedder,
            t_embedder_handles: handles.t_embedder,
            noise_refiner,
            noise_refiner_handles: handles.noise_refiner,
            context_refiner,
            context_refiner_handles: handles.context_refiner,
            layers,
            layers_handles: handles.layers,
            final_layer,
            final_layer_handles: handles.final_layer,
            rope,
            x_pad_token: handles.x_pad_token,
            cap_pad_token: handles.cap_pad_token,
        }
    }

    pub async fn forward<'a, S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        main_pipelines: &[BlockPipelines],
        encoder_pipelines: &BlockPipelines,
        embedder_pipelines: &BlockPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        inputs: &DitInputs<'a>,
    ) -> Result<DitForwardLayout, DitError<S::Error>> {
        self.forward_with_taps(
            backend,
            main_pipelines,
            encoder_pipelines,
            embedder_pipelines,
            residency,
            scratch,
            inputs,
            DitTaps::default(),
        )
        .await
    }

    /// `main_pipelines` is used for the 30 main DiT blocks (Q8 weights in
    /// Q8 mode, bf16 otherwise). `encoder_pipelines` is used for
    /// t_embedder, noise_refiner, context_refiner. `embedder_pipelines` is
    /// the dense-input front-door set used by x_embedder, cap_embedder,
    /// and final_layer; it mirrors `encoder_pipelines`. All sets share
    /// `act_dtype` so the residual stream is contiguous.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward_with_taps<'a, S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        main_pipelines: &[BlockPipelines],
        encoder_pipelines: &BlockPipelines,
        embedder_pipelines: &BlockPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        inputs: &DitInputs<'a>,
        mut taps: DitTaps<'_>,
    ) -> Result<DitForwardLayout, DitError<S::Error>> {
        debug_assert_eq!(
            main_pipelines.len(),
            self.layers.len(),
            "main_pipelines must hold one BlockPipelines per main layer",
        );
        debug_assert_eq!(
            main_pipelines[0].act_dtype, encoder_pipelines.act_dtype,
            "main/encoder pipelines must share act_dtype",
        );
        debug_assert_eq!(
            embedder_pipelines.act_dtype, encoder_pipelines.act_dtype,
            "embedder_pipelines must mirror encoder act_dtype",
        );
        // Single alias for sizing — act_dtype is identical across the
        // two pipeline sets, so either works. Routing of kernel pipelines
        // is explicit at each call site below.
        let pipelines = encoder_pipelines;
        let (c, f, h, w) = inputs.size;
        let dim = config::DIM as u32;
        let head_dim = config::HEAD_DIM;
        let cap_feat_dim = config::CAP_FEAT_DIM;

        // --- 1. patchify image ---
        let patch_in = inputs.patch_size * inputs.patch_size * inputs.f_patch_size * c;
        let cap_padded_len = inputs.cap_len + seq::pad_len(inputs.cap_len);
        let px = seq::patchify(
            inputs.image,
            c,
            f,
            h,
            w,
            inputs.patch_size,
            inputs.f_patch_size,
            cap_padded_len,
        );
        let seq_x = px.padded_len as u32;

        // Upstream probes (host-side inputs feeding the embedders + noise).
        if tracing::enabled!(target: trace::DIAG, tracing::Level::INFO) {
            diag_print_stats("00a_dit_input_image_raw", inputs.image);
            diag_print_stats("00b_x_tokens_after_patchify", &px.tokens);
            diag_print_stats("00c_cap_feats_from_text_encoder", inputs.cap_feats);
            tracing::info!(
                target: trace::DIAG,
                "  [DIT-PROBE-SHAPE] timestep={} seq_x={seq_x} cap_len={} patch_in={patch_in} \
                 image_size=({c},{f},{h},{w})",
                inputs.timestep, inputs.cap_len,
            );
        }

        // --- 2. upload x tokens, run XEmbedder ---
        let x_tok_bytes = act_upload_bytes(embedder_pipelines.act_dtype, &px.tokens);
        let x_tok = scratch.alloc(x_tok_bytes.len() as u64)?;
        backend.write_buffer(x_tok.id, 0, &x_tok_bytes)?;
        let x_act = ResStream::alloc(scratch, pipelines, seq_x, dim)?;
        {
            let views = self.x_embedder_handles.acquire(residency, backend).await?;
            let bufs = views.bufs();
            let x_tok_ref = x_tok.as_buf_ref();
            let out_ref = x_act.data.as_buf_ref();
            let scope = scratch.batch();
            self.x_embedder.forward(
                &scope,
                embedder_pipelines,
                scope.import(&x_tok_ref),
                seq_x,
                scope.import(&out_ref),
                &bufs,
            )?;
            scope.submit_void().await?;
        }
        diag_probe_dense(
            backend,
            &x_act.data.as_buf_ref(),
            (seq_x * dim) as usize,
            embedder_pipelines.act_dtype,
            "01_x_embedder_out",
        )
        .await?;
        // --- 3. scatter pad token into pad rows ---
        {
            let pad = residency.acquire(self.x_pad_token, backend).await?;
            scatter_pad_rows(
                backend,
                embedder_pipelines,
                scratch,
                x_act.as_act_ref(),
                pad.buf(),
                seq_x,
                dim,
                &px.pad_mask,
            )
            .await?;
        }
        diag_probe_dense(
            backend,
            &x_act.data.as_buf_ref(),
            (seq_x * dim) as usize,
            embedder_pipelines.act_dtype,
            "02_x_after_scatter_pad",
        )
        .await?;

        // --- 4. rope freqs for x ---
        let x_freqs_bytes =
            seq::freqs_upload_bytes(pipelines.act_dtype, &self.rope.lookup(&px.pos_ids));
        let x_freqs = scratch.alloc(x_freqs_bytes.len() as u64)?;
        backend.write_buffer(x_freqs.id, 0, &x_freqs_bytes)?;

        // --- 5. x attn mask --- bsz=1 -> all zero.
        let x_mask_bytes = seq::attn_mask_zero_bytes_act(px.padded_len, pipelines.act_dtype);
        let x_mask = scratch.alloc(x_mask_bytes.len() as u64)?;
        backend.write_buffer(x_mask.id, 0, &x_mask_bytes)?;

        // --- t_embedder for adaln_input ---
        let t_emb = scratch.alloc(pipelines.act_bytes(config::ADALN_EMBED_DIM as u32))?;
        {
            let views = self.t_embedder_handles.acquire(residency, backend).await?;
            let bufs = views.bufs();
            let t_emb_ref = t_emb.as_buf_ref();
            let scope = scratch.batch();
            self.t_embedder.forward(
                &scope,
                pipelines,
                inputs.timestep * config::T_SCALE,
                scope.import(&t_emb_ref),
                &bufs,
            )?;
            scope.submit_void().await?;
        }
        diag_probe_dense(
            backend,
            &t_emb.as_buf_ref(),
            config::ADALN_EMBED_DIM,
            pipelines.act_dtype,
            "00d_t_embedder_out",
        )
        .await?;

        // Depth-K ring of in-flight submits shared by the refiner loops, the
        // cap glue, and the main layer loop. Deferred submits keep the queue
        // fed across section boundaries (no CPU fence wait per section); each
        // hold-bag keeps the submit's consumed buffers and weight pins alive
        // until the GPU finishes. Static depth=2: higher depths help GPU
        // saturation when per-block GPU >> CPU-encode, but each extra slot
        // pins another block's weights + live workspace (in-flight guards
        // aren't spillable). At 768x768 / 2 GiB, depth=3 wedges (verified:
        // BudgetTooSmall at block.3 with the elastic gate accepting depth=3
        // transiently before mid-phase workspace expansion eats the residency
        // headroom). Keep static depth=2; the idx+2 prefetch in each loop is
        // the overlap mechanism.
        const SUBMIT_DEPTH: usize = 2;
        let mut pending_submits: SubmitRing<'_> = VecDeque::new();

        // --- 6. noise_refiner ---
        // Block expression scopes the rollup span guard `_nr_scope` so it
        // exits when the loop is done; x_cur escapes as the block's value.
        let x_cur: ResStream = {
            let _nr_scope =
                trace::scope!("dit.noise_refiner", n = self.noise_refiner.len()).entered();
            let mut x_cur: ResStream = x_act;
            let mut pending = if self.noise_refiner_handles.is_empty() {
                None
            } else {
                Some(
                    self.noise_refiner_handles[0]
                        .acquire(residency, backend)
                        .instrument(tracing::debug_span!(target: PHASE, "dit.acquire", phase = "noise_refiner", idx = 0_usize))
                        .await?,
                )
            };
            for (idx, blk) in self.noise_refiner.iter().enumerate() {
                let _blk_scope = trace::scope!(format_args!("block.{idx}")).entered();
                while pending_submits.len() >= SUBMIT_DEPTH {
                    let (fut, _holds) = pending_submits.pop_front().unwrap();
                    fut.await?;
                }
                let nxt = ResStream::alloc(scratch, pipelines, seq_x, dim)?;
                let views = pending
                    .take()
                    .expect("pending noise_refiner acquire missing");
                let bufs = views.bufs();
                let x_cur_ref = x_cur.as_act_ref();
                let freqs_ref = x_freqs.as_buf_ref();
                let mask_ref = x_mask.as_buf_ref();
                let t_emb_ref = t_emb.as_buf_ref();
                let nxt_ref = nxt.as_act_ref();
                let next_idx = idx + 1;
                let scope = scratch.batch();
                let x_cur_act = import_act(&scope, x_cur_ref);
                let nxt_act = import_act(&scope, nxt_ref);
                blk.forward(
                    &scope,
                    pipelines,
                    x_cur_act,
                    scope.import_copy(freqs_ref),
                    scope.import_copy(mask_ref),
                    Some(scope.import_copy(t_emb_ref)),
                    nxt_act,
                    &bufs,
                )?;
                let next_acquire = async {
                    match self.noise_refiner_handles.get(next_idx) {
                        Some(h) => {
                            let span = tracing::debug_span!(target: PHASE, "dit.acquire", phase = "noise_refiner", idx = next_idx);
                            Ok::<_, ResidencyError<S::Error, WgpuError>>(Some(
                                h.acquire(residency, backend).instrument(span).await?,
                            ))
                        }
                        None => Ok(None),
                    }
                };
                let prefetch_after = async {
                    match self.noise_refiner_handles.get(idx + 2) {
                        Some(h) => {
                            let span = tracing::debug_span!(target: PHASE, "dit.prefetch", phase = "noise_refiner", idx = idx + 2);
                            h.prefetch(residency, backend).instrument(span).await
                        }
                        None => Ok(()),
                    }
                };
                // queue.submit happens synchronously inside submit_deferred;
                // the GPU runs this block while the joins below upload the
                // next blocks' weights.
                let submit_fut = scope.submit_deferred().instrument(
                    tracing::debug_span!(target: PHASE, "dit.submit", phase = "noise_refiner", idx),
                );
                let (n_res, p_res) = futures::join!(next_acquire, prefetch_after);
                p_res?;
                pending = n_res?;
                let prev_x = std::mem::replace(&mut x_cur, nxt);
                pending_submits.push_back((
                    Box::pin(submit_fut),
                    vec![Box::new(prev_x) as Box<dyn RingHold + '_>, Box::new(views)],
                ));
                diag_probe_resstream(
                    backend,
                    pipelines,
                    &x_cur,
                    seq_x,
                    dim,
                    &format!("04_noise_refiner_block{idx}_out"),
                )
                .await?;
            }
            x_cur
        };

        // --- 7-8. cap pad meta + CapEmbedder ---
        let cm = seq::cap_meta(inputs.cap_len);
        let seq_cap = cm.padded_len as u32;
        let cap_feats_padded = seq::pad_cap_feats(
            inputs.cap_feats,
            cap_feat_dim,
            inputs.cap_len,
            cm.padded_len,
        );
        let cap_in_bytes = act_upload_bytes(embedder_pipelines.act_dtype, &cap_feats_padded);
        let cap_in = scratch.alloc(cap_in_bytes.len() as u64)?;
        backend.write_buffer(cap_in.id, 0, &cap_in_bytes)?;
        let cap_act = ResStream::alloc(scratch, pipelines, seq_cap, dim)?;
        let (cap_intermediate_normed, cap_intermediate_pre_bias) = {
            while pending_submits.len() >= SUBMIT_DEPTH {
                let (fut, _holds) = pending_submits.pop_front().unwrap();
                fut.await?;
            }
            let views = self
                .cap_embedder_handles
                .acquire(residency, backend)
                .await?;
            let bufs = views.bufs();
            let cap_in_ref = cap_in.as_buf_ref();
            let out_ref = cap_act.data.as_buf_ref();
            let scope = scratch.batch();
            let inter = self.cap_embedder.forward(
                &scope,
                embedder_pipelines,
                scope.import_copy(cap_in_ref),
                seq_cap,
                scope.import_copy(out_ref),
                &bufs,
            )?;
            let (outs, fut) = scope.submit_many_deferred(&[inter.normed, inter.pre_bias]);
            pending_submits.push_back((
                Box::pin(fut),
                vec![Box::new(views) as Box<dyn RingHold + '_>],
            ));
            let mut it = outs.into_iter();
            let normed = it.next().unwrap();
            let pre_bias = it.next().unwrap();
            (normed, pre_bias)
        };
        diag_probe_dense(
            backend,
            &cap_act.data.as_buf_ref(),
            (seq_cap * dim) as usize,
            embedder_pipelines.act_dtype,
            "05_cap_embedder_out",
        )
        .await?;
        if let Some(sink) = taps.cap_embed_normed.as_deref_mut() {
            read_into_f32(
                backend,
                &cap_intermediate_normed,
                (seq_cap * cap_feat_dim as u32) as usize,
                embedder_pipelines.act_dtype,
                sink,
            )
            .await?;
        }
        if let Some(sink) = taps.cap_embed_pre_bias.as_deref_mut() {
            read_into_f32(
                backend,
                &cap_intermediate_pre_bias,
                (seq_cap * dim) as usize,
                embedder_pipelines.act_dtype,
                sink,
            )
            .await?;
        }
        {
            while pending_submits.len() >= SUBMIT_DEPTH {
                let (fut, _holds) = pending_submits.pop_front().unwrap();
                fut.await?;
            }
            let pad = residency.acquire(self.cap_pad_token, backend).await?;
            let fut = scatter_pad_rows_deferred(
                backend,
                embedder_pipelines,
                scratch,
                cap_act.as_act_ref(),
                pad.buf(),
                seq_cap,
                dim,
                &cm.pad_mask,
            )?;
            pending_submits
                .push_back((Box::pin(fut), vec![Box::new(pad) as Box<dyn RingHold + '_>]));
        }
        diag_probe_resstream(
            backend,
            pipelines,
            &cap_act,
            seq_cap,
            dim,
            "06_cap_act_post_pad",
        )
        .await?;

        if let Some(sink) = taps.cap_embedded.as_deref_mut() {
            read_resstream_to_f32(
                backend,
                pipelines,
                &cap_act,
                seq_cap,
                dim,
                sink,
                "cap_embedded",
            )
            .await?;
        }

        // --- 9. cap rope freqs + attn mask ---
        let cap_freqs_bytes =
            seq::freqs_upload_bytes(pipelines.act_dtype, &self.rope.lookup(&cm.pos_ids));
        let cap_freqs = scratch.alloc(cap_freqs_bytes.len() as u64)?;
        backend.write_buffer(cap_freqs.id, 0, &cap_freqs_bytes)?;
        let cap_mask_bytes = seq::attn_mask_zero_bytes_act(cm.padded_len, pipelines.act_dtype);
        let cap_mask = scratch.alloc(cap_mask_bytes.len() as u64)?;
        backend.write_buffer(cap_mask.id, 0, &cap_mask_bytes)?;

        // --- 10. context_refiner ---
        // See noise_refiner block above: span guard `_cr_scope` is bounded by
        // the inner block; cap_cur escapes as the block expression's value.
        let cap_cur: ResStream = {
            let _cr_scope =
                trace::scope!("dit.context_refiner", n = self.context_refiner.len()).entered();
            let mut cap_cur: ResStream = cap_act;
            let mut pending = if self.context_refiner_handles.is_empty() {
                None
            } else {
                Some(
                    self.context_refiner_handles[0]
                        .acquire(residency, backend)
                        .instrument(tracing::debug_span!(target: PHASE, "dit.acquire", phase = "context_refiner", idx = 0_usize))
                        .await?,
                )
            };
            for (idx, blk) in self.context_refiner.iter().enumerate() {
                let _blk_scope = trace::scope!(format_args!("block.{idx}")).entered();
                while pending_submits.len() >= SUBMIT_DEPTH {
                    let (fut, _holds) = pending_submits.pop_front().unwrap();
                    fut.await?;
                }
                if idx == 1
                    && let Some(sink) = taps.ctx_refiner_0_out.as_deref_mut()
                {
                    read_resstream_to_f32(
                        backend,
                        pipelines,
                        &cap_cur,
                        seq_cap,
                        dim,
                        sink,
                        "ctx_refiner_0_out",
                    )
                    .await?;
                }
                let nxt = ResStream::alloc(scratch, pipelines, seq_cap, dim)?;
                let views = pending
                    .take()
                    .expect("pending context_refiner acquire missing");
                let bufs = views.bufs();
                let cap_cur_ref = cap_cur.as_act_ref();
                let freqs_ref = cap_freqs.as_buf_ref();
                let mask_ref = cap_mask.as_buf_ref();
                let nxt_ref = nxt.as_act_ref();
                let next_idx = idx + 1;
                // Allocate per-op tap bufs only for ctx_refiner block 0 and
                // only when the caller requested ctx_block0. Sized against
                // the ctx-refiner block's own seq (= seq_cap), not seq_u.
                let cb0_bufs = if idx == 0
                    && let Some(req) = taps.ctx_block0.as_deref()
                {
                    Some(Block0TapBufs::alloc(
                        req, scratch, pipelines, seq_cap, dim, blk.cfg,
                    )?)
                } else {
                    None
                };
                let cb0_block_taps = cb0_bufs
                    .as_ref()
                    .map(Block0TapBufs::to_block_debug_taps)
                    .unwrap_or(BlockDebugTaps::EMPTY);
                let scope = scratch.batch();
                let cap_cur_act = import_act(&scope, cap_cur_ref);
                let nxt_act = import_act(&scope, nxt_ref);
                blk.forward_taps(
                    &scope,
                    pipelines,
                    cap_cur_act,
                    scope.import_copy(freqs_ref),
                    scope.import_copy(mask_ref),
                    None,
                    nxt_act,
                    &bufs,
                    &cb0_block_taps,
                )?;
                let next_acquire = async {
                    match self.context_refiner_handles.get(next_idx) {
                        Some(h) => {
                            let span = tracing::debug_span!(target: PHASE, "dit.acquire", phase = "context_refiner", idx = next_idx);
                            Ok::<_, ResidencyError<S::Error, WgpuError>>(Some(
                                h.acquire(residency, backend).instrument(span).await?,
                            ))
                        }
                        None => Ok(None),
                    }
                };
                let prefetch_after = async {
                    match self.context_refiner_handles.get(idx + 2) {
                        Some(h) => {
                            let span = tracing::debug_span!(target: PHASE, "dit.prefetch", phase = "context_refiner", idx = idx + 2);
                            h.prefetch(residency, backend).instrument(span).await
                        }
                        None => Ok(()),
                    }
                };
                let submit_fut = scope.submit_deferred().instrument(
                    tracing::debug_span!(target: PHASE, "dit.submit", phase = "context_refiner", idx),
                );
                let (n_res, p_res) = futures::join!(next_acquire, prefetch_after);
                p_res?;
                pending = n_res?;
                if let (Some(tap_bufs), Some(sink)) = (cb0_bufs, taps.ctx_block0.as_deref_mut()) {
                    // Tap readback maps after the deferred submit in queue
                    // order, so it observes this block's writes.
                    tap_bufs.read_back(backend, pipelines, sink).await?;
                }
                let prev_cap = std::mem::replace(&mut cap_cur, nxt);
                pending_submits.push_back((
                    Box::pin(submit_fut),
                    vec![
                        Box::new(prev_cap) as Box<dyn RingHold + '_>,
                        Box::new(views),
                    ],
                ));
                diag_probe_resstream(
                    backend,
                    pipelines,
                    &cap_cur,
                    seq_cap,
                    dim,
                    &format!("07_context_refiner_block{idx}_out"),
                )
                .await?;
            }
            cap_cur
        };

        if let Some(sink) = taps.last_ctx_refiner_out.as_deref_mut() {
            read_resstream_to_f32(
                backend,
                pipelines,
                &cap_cur,
                seq_cap,
                dim,
                sink,
                "last_ctx_refiner_out",
            )
            .await?;
        }

        // --- 11. concat unified = [x; cap] ---
        let seq_u = seq_x + seq_cap;
        let unified = ResStream::alloc(scratch, pipelines, seq_u, dim)?;
        let row_bytes = pipelines.act_bytes(dim);
        let freq_row = pipelines.act_bytes(head_dim as u32);
        let unified_freqs = scratch.alloc(seq_u as u64 * freq_row)?;
        {
            while pending_submits.len() >= SUBMIT_DEPTH {
                let (fut, _holds) = pending_submits.pop_front().unwrap();
                fut.await?;
            }
            let x_cur_ref = x_cur.data.as_buf_ref();
            let cap_cur_ref = cap_cur.data.as_buf_ref();
            let x_freqs_ref = x_freqs.as_buf_ref();
            let cap_freqs_ref = cap_freqs.as_buf_ref();
            let unified_ref = unified.data.as_buf_ref();
            let unified_freqs_ref = unified_freqs.as_buf_ref();
            let scope = scratch.batch();
            let xh = scope.import_copy(x_cur_ref);
            let ch = scope.import_copy(cap_cur_ref);
            let xfh = scope.import_copy(x_freqs_ref);
            let cfh = scope.import_copy(cap_freqs_ref);
            let uh = scope.import_copy(unified_ref);
            let ufh = scope.import_copy(unified_freqs_ref);
            scope.copy_buffer_to_buffer(xh, 0, uh, 0, (seq_x as u64) * row_bytes)?;
            scope.copy_buffer_to_buffer(
                ch,
                0,
                uh,
                (seq_x as u64) * row_bytes,
                (seq_cap as u64) * row_bytes,
            )?;
            scope.copy_buffer_to_buffer(xfh, 0, ufh, 0, (seq_x as u64) * freq_row)?;
            scope.copy_buffer_to_buffer(
                cfh,
                0,
                ufh,
                (seq_x as u64) * freq_row,
                (seq_cap as u64) * freq_row,
            )?;
            let fut = scope.submit_deferred();
            pending_submits.push_back((Box::pin(fut), Vec::new()));
        }
        diag_probe_resstream(
            backend,
            pipelines,
            &unified,
            seq_u,
            dim,
            "08_unified_in_pre_main_block0",
        )
        .await?;

        if let Some(sink) = taps.unified_in.as_deref_mut() {
            read_resstream_to_f32(backend, pipelines, &unified, seq_u, dim, sink, "unified_in")
                .await?;
        }

        let u_mask = seq::attn_mask_zero_bytes_act(seq_u as usize, pipelines.act_dtype);
        let unified_mask = scratch.alloc(u_mask.len() as u64)?;
        backend.write_buffer(unified_mask.id, 0, &u_mask)?;

        // --- 12. main layers ---
        let _lr_scope = trace::scope!("dit.layers", n = self.layers.len()).entered();
        let mut u_cur: ResStream = unified;
        let mut pending = if self.layers_handles.is_empty() {
            None
        } else {
            Some(
                self.layers_handles[0]
                    .acquire(residency, backend)
                    .instrument(tracing::debug_span!(target: PHASE, "dit.acquire", phase = "layers", idx = 0_usize))
                    .await?,
            )
        };
        let last_main_idx = self.layers.len().saturating_sub(1);
        for (idx, blk) in self.layers.iter().enumerate() {
            let _blk_scope = trace::scope!(format_args!("block.{idx}")).entered();
            // Cap simultaneous pinned BlockViews before adding more.
            // Drain to len < SUBMIT_DEPTH so that after this iteration's
            // push_back the ring sits at exactly SUBMIT_DEPTH.
            while pending_submits.len() >= SUBMIT_DEPTH {
                let (fut, _holds) = pending_submits.pop_front().unwrap();
                fut.await?;
            }
            // DIAG per-block trajectory: when DIAG INFO is enabled, probe
            // residual stream at EVERY block boundary so we get a slope-vs-
            // block-index curve. Without DIAG, fall back to the prior sparse
            // probe set (cheap stats only at canary boundaries).
            let fire_probe = idx > 0
                && (matches!(idx, 1 | 2 | 6 | 11 | 16 | 21 | 26 | 27 | 28 | 29)
                    || tracing::enabled!(target: trace::DIAG, tracing::Level::INFO));
            if fire_probe {
                // u_cur at top of iter idx = output of block (idx-1).
                let prev = idx - 1;
                diag_probe_resstream_split(
                    backend,
                    pipelines,
                    &u_cur,
                    seq_x,
                    seq_cap,
                    dim,
                    &format!("09_main_block{prev}_out"),
                )
                .await?;
            }
            // DIAG full-residual readback into per-block sink. Fires at
            // every block boundary (idx > 0), captures block (idx-1).
            // Forces queue drain so the read sees committed writes from
            // the previous iter's pipelined submit.
            if idx > 0
                && let Some(sink) = taps.per_main_block_residual.as_deref_mut()
            {
                while let Some((fut, _holds)) = pending_submits.pop_front() {
                    fut.await?;
                }
                let prev = idx - 1;
                if sink.len() <= prev {
                    sink.resize(prev + 1, Vec::new());
                }
                read_resstream_to_f32(
                    backend,
                    pipelines,
                    &u_cur,
                    seq_u,
                    dim,
                    &mut sink[prev],
                    "per_block_residual",
                )
                .await?;
            }
            if idx == 1
                && let Some(sink) = taps.main_layer_0_out.as_deref_mut()
            {
                read_resstream_to_f32(
                    backend,
                    pipelines,
                    &u_cur,
                    seq_u,
                    dim,
                    sink,
                    "main_layer_0_out",
                )
                .await?;
            }
            if idx == 15
                && let Some(sink) = taps.main_layer_14_out.as_deref_mut()
            {
                read_resstream_to_f32(
                    backend,
                    pipelines,
                    &u_cur,
                    seq_u,
                    dim,
                    sink,
                    "main_layer_14_out",
                )
                .await?;
            }
            let nxt = ResStream::alloc(scratch, pipelines, seq_u, dim)?;
            let views = pending.take().expect("pending layers acquire missing");
            let bufs = views.bufs();
            // Per-op probe taps for this block. `block0`, `block_last`,
            // and `extra_blocks` are all routed through the same
            // `Block0TapBufs` machinery; `probe_route` records which
            // sink to feed at read_back time.
            let extra_pos = taps
                .extra_blocks
                .as_deref()
                .and_then(|extras| extras.iter().position(|(b, _)| *b == idx));
            let (probe_bufs, probe_route) = if idx == 0 && taps.block0.is_some() {
                let bufs = taps
                    .block0
                    .as_deref()
                    .map(|req| {
                        Block0TapBufs::alloc(
                            req,
                            scratch,
                            &main_pipelines[idx],
                            seq_u,
                            dim,
                            blk.cfg,
                        )
                    })
                    .transpose()?;
                (bufs, ProbeRoute::Block0)
            } else if idx == last_main_idx && taps.block_last.is_some() {
                let bufs = taps
                    .block_last
                    .as_deref()
                    .map(|req| {
                        Block0TapBufs::alloc(
                            req,
                            scratch,
                            &main_pipelines[idx],
                            seq_u,
                            dim,
                            blk.cfg,
                        )
                    })
                    .transpose()?;
                (bufs, ProbeRoute::BlockLast)
            } else if let (Some(extras), Some(pos)) = (taps.extra_blocks.as_deref(), extra_pos) {
                let req = &extras[pos].1;
                let bufs = Some(Block0TapBufs::alloc(
                    req,
                    scratch,
                    &main_pipelines[idx],
                    seq_u,
                    dim,
                    blk.cfg,
                )?);
                (bufs, ProbeRoute::Extra(pos))
            } else {
                (None, ProbeRoute::None)
            };
            let b0_block_taps = probe_bufs
                .as_ref()
                .map(Block0TapBufs::to_block_debug_taps)
                .unwrap_or(BlockDebugTaps::EMPTY);
            let u_cur_ref = u_cur.as_act_ref();
            let freqs_ref = unified_freqs.as_buf_ref();
            let mask_ref = unified_mask.as_buf_ref();
            let t_emb_ref = t_emb.as_buf_ref();
            let nxt_ref = nxt.as_act_ref();
            let next_idx = idx + 1;
            // Diag-tap fast path bypass: when per-op taps are requested
            // (block 0 OR last block), read_back must observe THIS block's
            // writes via the same WsBufs the scope owned, so we cannot move
            // that scope's guards into a deferred future. Fall back to sync
            // submit for this iteration. (Production runs never set taps.)
            let needs_sync = probe_bufs.is_some();
            // Per-block packer budget: VRAM ceiling minus currently-pinned
            // weights (including this block's `views` + any ring-held weights
            // + ongoing concurrent acquire) and a fixed safety pad. Read once
            // at block start; eviction may shift pins during the block but
            // the budget here is the bound on this scope's WORKSPACE growth,
            // not weights.
            // Pad=0: `phase_peaks` is doc'd as a conservative upper bound;
            // the arbiter reclaim chain (workspace alloc evicts unpinned
            // weights under pressure) is the safety net if it under-counts.
            // Subtract only RENTED workspace (live WsBufs) — idle pool bytes
            // are reusable and will satisfy the next phase's allocs in-place
            // without growing VRAM further.
            let vram_ceiling = residency.budget().vram_bytes;
            let mem = backend.mem_account();
            let pinned_weights = mem.vram_current(VramCategory::Weights);
            let workspace_total =
                mem.vram_current(VramCategory::Workspace) + mem.vram_current(VramCategory::Staging);
            let rented_workspace = workspace_total.saturating_sub(scratch.pool_bytes());
            let packer_budget = vram_ceiling
                .saturating_sub(pinned_weights)
                .saturating_sub(rented_workspace);
            let mut packer = ScopePacker::new(scratch, packer_budget);
            blk.forward_taps_packed(
                &mut packer,
                &main_pipelines[idx],
                u_cur_ref,
                freqs_ref,
                mask_ref,
                Some(t_emb_ref),
                nxt_ref,
                &bufs,
                &b0_block_taps,
            )?;
            // Acquire the next block's weights concurrently with this
            // block's GPU work. Also prefetch idx+2 (unpinned) so the LRU
            // is pre-warmed and the next iter's acquire is a hit. Prefetch
            // drops the view immediately, leaving the entry in Gpu state
            // with pin_count=0 (evictable under pressure). The packer
            // budget already excludes pinned weights, so adding an
            // evictable resident doesn't tighten this scope's headroom.
            let next_acquire = async {
                match self.layers_handles.get(next_idx) {
                    Some(h) => {
                        let span = tracing::debug_span!(target: PHASE, "dit.acquire", phase = "layers", idx = next_idx);
                        Ok::<_, ResidencyError<S::Error, WgpuError>>(Some(
                            h.acquire(residency, backend).instrument(span).await?,
                        ))
                    }
                    None => Ok(None),
                }
            };
            let prefetch_after = async {
                match self.layers_handles.get(idx + 2) {
                    Some(h) => {
                        let span = tracing::debug_span!(target: PHASE, "dit.prefetch", phase = "layers", idx = idx + 2);
                        h.prefetch(residency, backend).instrument(span).await
                    }
                    None => Ok(()),
                }
            };
            if needs_sync {
                // Drain any in-flight submits first so the read_back we're
                // about to do observes a quiet queue; then sync-submit and
                // read taps before moving on.
                while let Some((fut, _holds)) = pending_submits.pop_front() {
                    fut.await?;
                }
                let submit_fut = packer.finish_void().instrument(
                    tracing::debug_span!(target: PHASE, "dit.submit", phase = "layers", idx),
                );
                let (s_res, n_res, p_res) =
                    futures::join!(submit_fut, next_acquire, prefetch_after);
                p_res?;
                pending = n_res?;
                s_res?;
                if let Some(tap_bufs) = probe_bufs {
                    let sink_opt: Option<&mut Block0Taps> = match probe_route {
                        ProbeRoute::Block0 => taps.block0.as_deref_mut(),
                        ProbeRoute::BlockLast => taps.block_last.as_deref_mut(),
                        ProbeRoute::Extra(pos) => taps
                            .extra_blocks
                            .as_deref_mut()
                            .map(|extras| &mut extras[pos].1),
                        ProbeRoute::None => None,
                    };
                    if let Some(sink) = sink_opt {
                        tap_bufs.read_back(backend, pipelines, sink).await?;
                    }
                }
                u_cur = nxt;
                // Keep `views` alive for the duration of forward_taps; it
                // drops here. Sync submit already awaited GPU completion,
                // so weight pins are free to release.
                drop(views);
            } else {
                // Pipelined path: scope.submit_deferred() runs
                // backend.submit's synchronous prelude (queue.submit
                // happens here), returning a completion future. Move the
                // future plus the buffers it transitively depends on
                // (prior u_cur, the block's BlockViews) into the ring.
                // The ring is bounded by the drain-at-top above, so no
                // trailing pop here.
                // packer.finish_void synchronously calls queue.submit; the
                // returned future just awaits the fence. So GPU is already
                // running this block when we join next_acquire + prefetch
                // for upcoming blocks (uploads overlap with compute).
                let submit_fut = packer.finish_void();
                let (n_res, p_res) = futures::join!(next_acquire, prefetch_after);
                p_res?;
                pending = n_res?;
                let prev_u = std::mem::replace(&mut u_cur, nxt);
                pending_submits.push_back((
                    Box::pin(submit_fut),
                    vec![Box::new(prev_u) as Box<dyn RingHold + '_>, Box::new(views)],
                ));
            }
        }
        // Drain remaining in-flight submits before any downstream read or
        // the final_layer encoder consumes `u_cur`.
        while let Some((fut, _holds)) = pending_submits.pop_front() {
            fut.await?;
        }
        diag_probe_resstream_split(
            backend,
            pipelines,
            &u_cur,
            seq_x,
            seq_cap,
            dim,
            "10_main_block29_out_post_drain",
        )
        .await?;

        if let Some(sink) = taps.last_main_layer_out.as_deref_mut() {
            read_resstream_to_f32(
                backend,
                pipelines,
                &u_cur,
                seq_u,
                dim,
                sink,
                "last_main_layer_out",
            )
            .await?;
        }
        // Per-block-residual sink: also capture last block (N-1) here
        // since the iter-top probe only fires for idx > 0 (i.e. captures
        // blocks 0..N-2 only). Pads sink to length last_main_idx + 1.
        if let Some(sink) = taps.per_main_block_residual.as_deref_mut() {
            if sink.len() <= last_main_idx {
                sink.resize(last_main_idx + 1, Vec::new());
            }
            read_resstream_to_f32(
                backend,
                pipelines,
                &u_cur,
                seq_u,
                dim,
                &mut sink[last_main_idx],
                "per_block_residual_last",
            )
            .await?;
        }

        // --- 13. final layer ---
        // `u_cur` and `t_emb` are already dense at the shared act dtype;
        // final_layer consumes them directly.
        let oc = self.final_layer.cfg.out_channels as u32;
        let final_out = scratch.alloc(embedder_pipelines.act_bytes(seq_u * oc))?;
        {
            let views = self.final_layer_handles.acquire(residency, backend).await?;
            let bufs = views.bufs();
            let u_ref = u_cur.data.as_buf_ref();
            let t_ref = t_emb.as_buf_ref();
            let final_out_ref = final_out.as_buf_ref();
            let scope = scratch.batch();
            self.final_layer.forward(
                &scope,
                embedder_pipelines,
                scope.import(&u_ref),
                scope.import(&t_ref),
                seq_u,
                scope.import(&final_out_ref),
                &bufs,
            )?;
            scope.submit_void().await?;
        }
        diag_probe_dense(
            backend,
            &final_out.as_buf_ref(),
            (seq_u * oc) as usize,
            embedder_pipelines.act_dtype,
            "11_final_layer_out",
        )
        .await?;
        // Image-only prefix: the first `seq_x_ori * oc` cells are what
        // decode_image() actually feeds into the latent / PNG. The full
        // probe above includes padded image rows + caption rows, whose
        // aggregate can mask divergence in the image-only window.
        diag_probe_dense(
            backend,
            &final_out.as_buf_ref(),
            px.ori_len * (oc as usize),
            embedder_pipelines.act_dtype,
            "11b_final_layer_out_img_only",
        )
        .await?;

        if let Some(sink) = taps.final_layer_out.as_deref_mut() {
            read_into_f32_tagged(
                backend,
                &final_out,
                (seq_u * oc) as usize,
                embedder_pipelines.act_dtype,
                sink,
                "final_layer_out",
            )
            .await?;
        }

        Ok(DitForwardLayout {
            final_out,
            act_dtype: embedder_pipelines.act_dtype,
            patch_in,
            out_channels: self.final_layer.cfg.out_channels,
            image_size: (c, f, h, w),
            seq_x_ori: px.ori_len,
            seq_x_padded: px.padded_len,
            seq_cap_padded: cm.padded_len,
            patch_size: inputs.patch_size,
            f_patch_size: inputs.f_patch_size,
        })
    }

    pub fn decode_image(&self, layout: &DitForwardLayout, bytes: &[u8]) -> DitOutput {
        let (c, f, h, w) = layout.image_size;
        let oc = layout.out_channels;
        let row = oc;
        let total_rows = layout.seq_x_padded + layout.seq_cap_padded;
        let stride = layout.act_dtype.bytes_per_elem() as usize;
        debug_assert_eq!(bytes.len(), total_rows * row * stride);
        let mut tokens = Vec::<f32>::with_capacity(layout.seq_x_ori * row);
        match layout.act_dtype {
            ActDtype::F32 => {
                for i in 0..layout.seq_x_ori {
                    for j in 0..row {
                        let idx = (i * row + j) * 4;
                        tokens.push(f32::from_le_bytes([
                            bytes[idx],
                            bytes[idx + 1],
                            bytes[idx + 2],
                            bytes[idx + 3],
                        ]));
                    }
                }
            }
            ActDtype::Bf16 => {
                for i in 0..layout.seq_x_ori {
                    for j in 0..row {
                        let idx = (i * row + j) * 2;
                        let half = u16::from_le_bytes([bytes[idx], bytes[idx + 1]]);
                        tokens.push(f32::from_bits((half as u32) << 16));
                    }
                }
            }
            ActDtype::F16 => {
                for i in 0..layout.seq_x_ori {
                    for j in 0..row {
                        let idx = (i * row + j) * 2;
                        let bits = u16::from_le_bytes([bytes[idx], bytes[idx + 1]]);
                        tokens.push(half::f16::from_bits(bits).to_f32());
                    }
                }
            }
            ActDtype::I8 => unreachable!("I8 is never a block act_dtype"),
        }
        let image = seq::unpatchify(&tokens, c, f, h, w, layout.patch_size, layout.f_patch_size);
        DitOutput { image }
    }
}

pub struct DitForwardLayout {
    pub final_out: WsBuf<WgpuBackend>,
    /// Activation storage dtype the kernels were compiled for. Drives byte
    /// stride in `decode_image` and readback length in the pipeline driver.
    pub act_dtype: ActDtype,
    pub patch_in: usize,
    pub out_channels: usize,
    pub image_size: (usize, usize, usize, usize),
    pub seq_x_ori: usize,
    pub seq_x_padded: usize,
    pub seq_cap_padded: usize,
    pub patch_size: usize,
    pub f_patch_size: usize,
}

/// Encode `slice` for upload as activation storage. F32: 4 bytes per elem
/// little-endian. Bf16: 2 bytes per elem, RNE-rounded; consecutive pairs land
/// in the same `array<u32>` word when read by kernels (LE -> low half = even
/// index, high half = odd index, matching `pack_bf16x2(lo, hi)` in WGSL).
fn act_upload_bytes(act: ActDtype, slice: &[f32]) -> Vec<u8> {
    match act {
        ActDtype::F32 => {
            let mut bytes = vec![0u8; slice.len() * 4];
            for (i, v) in slice.iter().enumerate() {
                bytes[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
            }
            bytes
        }
        ActDtype::Bf16 => {
            let mut bytes = vec![0u8; slice.len() * 2];
            for (i, v) in slice.iter().enumerate() {
                let h = round_f32_to_bf16(*v);
                bytes[i * 2..(i + 1) * 2].copy_from_slice(&h.to_le_bytes());
            }
            bytes
        }
        ActDtype::F16 => {
            let mut bytes = vec![0u8; slice.len() * 2];
            for (i, v) in slice.iter().enumerate() {
                let h = half::f16::from_f32(*v).to_bits();
                bytes[i * 2..(i + 1) * 2].copy_from_slice(&h.to_le_bytes());
            }
            bytes
        }
        ActDtype::I8 => unreachable!(
            "ActDtype::I8 act upload needs a per-row scale - this path is only used by the embedder upload which has no caller-supplied scale (step 8 block.rs plumbing handles I8 acts at op boundaries instead)"
        ),
    }
}

/// f32 -> bf16 round-to-nearest-even, NaN/inf passthrough (NaN canonicalized
/// to a quiet bf16 NaN with the sign bit preserved). Mirrors the WGSL
/// `round_bf16` helper exactly.
fn round_f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let exp = (bits >> 23) & 0xff;
    if exp == 0xff {
        // NaN or inf: keep sign+exp, force a quiet NaN payload if mantissa nonzero.
        let mant = bits & 0x7f_ffff;
        let top = (bits >> 16) as u16;
        if mant == 0 { top } else { top | 0x0040 }
    } else {
        let rounding = 0x7fff + ((bits >> 16) & 1);
        ((bits.wrapping_add(rounding)) >> 16) as u16
    }
}

/// Replace masked rows in `dst` (fp32 `[n_rows, dim]`) with the single-row
/// vector at `pad_token` (bf16-packed `[dim]` from residency). Dispatches the
/// `scatter_pad_rows` kernel which decodes bf16 -> fp32 inline. `pub` for the
/// conformance smoke test.
/// Type-erased member of a pending-submit hold-bag. Blanket impl: anything
/// can be held; dropping the bag releases buffers and weight pins.
trait RingHold {}
impl<T> RingHold for T {}

/// Depth-bounded ring of in-flight submits spanning dit_forward sections.
/// Each entry pairs a submit completion future with the hold-bag that keeps
/// the submit's input buffers and weight views alive until the fence
/// resolves.
type SubmitRing<'a> = VecDeque<(
    Pin<Box<dyn Future<Output = Result<(), WgpuError>> + 'a>>,
    Vec<Box<dyn RingHold + 'a>>,
)>;

pub async fn scatter_pad_rows(
    backend: &WgpuBackend,
    pipelines: &BlockPipelines,
    scratch: &Workspace<WgpuBackend>,
    dst: ActBufRef,
    pad_token: BufRef,
    n_rows: u32,
    dim: u32,
    pad_mask: &[u8],
) -> Result<(), WgpuError> {
    scatter_pad_rows_deferred(
        backend, pipelines, scratch, dst, pad_token, n_rows, dim, pad_mask,
    )?
    .await
}

/// Like [`scatter_pad_rows`] but returns the completion future synchronously
/// (queue.submit already issued); the internal mask/uniform scratch buffers
/// ride inside the future until the fence resolves.
#[allow(clippy::too_many_arguments)]
pub fn scatter_pad_rows_deferred<'wsp>(
    backend: &WgpuBackend,
    pipelines: &BlockPipelines,
    scratch: &'wsp Workspace<WgpuBackend>,
    dst: ActBufRef,
    pad_token: BufRef,
    n_rows: u32,
    dim: u32,
    pad_mask: &[u8],
) -> Result<impl Future<Output = Result<(), WgpuError>> + 'wsp, WgpuError> {
    assert_eq!(
        pad_mask.len(),
        n_rows as usize,
        "pad_mask len {} != n_rows {}",
        pad_mask.len(),
        n_rows
    );
    debug_assert_eq!(pad_token.len, (dim as u64).div_ceil(2) * 4);
    {
        let elem_bytes = pipelines.act_dtype.bytes_per_elem();
        debug_assert_eq!(dst.data.len, (n_rows as u64) * (dim as u64) * elem_bytes);
    }

    let mask_u32: Vec<u32> = pad_mask.iter().map(|&m| m as u32).collect();
    let mask_bytes: &[u8] = bytemuck::cast_slice(&mask_u32);
    let mask_buf = scratch.alloc(mask_bytes.len() as u64)?;
    backend.write_buffer(mask_buf.id, 0, mask_bytes)?;

    let uniform_vals = [n_rows, dim, 0u32, 0u32];
    let uniform_bytes: &[u8] = bytemuck::cast_slice(&uniform_vals);
    let uniform_buf = scratch.alloc(uniform_bytes.len() as u64)?;
    backend.write_buffer(uniform_buf.id, 0, uniform_bytes)?;

    let mask_ref = mask_buf.as_buf_ref();
    let uniform_ref = uniform_buf.as_buf_ref();

    let scope = scratch.batch();
    let dispatch_elems = match pipelines.act_dtype {
        ActDtype::F32 => n_rows * dim,
        ActDtype::Bf16 | ActDtype::F16 => n_rows * (dim / 2),
        ActDtype::I8 => unreachable!("I8 is never a block act_dtype"),
    };
    scope.scatter_pad_rows::<ScatterPadRowsF32>(
        &pipelines.scatter_pad_rows,
        scope.import_copy(pad_token),
        scope.import_copy(mask_ref),
        scope.import_copy(uniform_ref),
        scope.import_copy(dst.data),
        dispatch_elems,
    )?;
    let fut = scope.submit_deferred();
    Ok(async move {
        let _hold = (mask_buf, uniform_buf);
        fut.await
    })
}

/// Scratch-allocated tap buffers for a single block's per-op readbacks. One
/// `WsBuf` per requested slot in [`Block0Taps`]; the buffers live in the
/// caller's `Workspace` and survive the block-forward submit so we can read
/// them back. Sized via `BlockPipelines::act_bytes` so packed-bf16 and f32
/// paths share the same code.
/// Per-tap scratch for one activation-bearing slot in `Block0TapBufs`. Holds
/// the data buffer plus a paired scale buffer for taps whose source is a
/// paired sdpa_i8 I/O slot (`attn_sdpa` under i8_sdpa). `rows` and `inner`
/// are the per-tap 2-D shape used to size readback and decode paired taps
/// via `seq::act_readback_i8_to_f32`. Different taps have different `(rows,
/// inner)` (e.g. `attn_q_norm` is `(rows*hq, hd)` while `attn_q` is
/// `(rows, hq*hd)`).
struct ActTapBuf {
    data: WsBuf<WgpuBackend>,
    scale: Option<WsBuf<WgpuBackend>>,
    rows: u32,
    inner: u32,
}

struct Block0TapBufs {
    adaln_input: Option<(WsBuf<WgpuBackend>, u32)>,
    adaln_pre: Option<(WsBuf<WgpuBackend>, u32)>,
    adaln_full: Option<(WsBuf<WgpuBackend>, u32)>,
    scale_msa: Option<(WsBuf<WgpuBackend>, u32)>,
    gate_msa: Option<(WsBuf<WgpuBackend>, u32)>,
    scale_mlp: Option<(WsBuf<WgpuBackend>, u32)>,
    gate_mlp: Option<(WsBuf<WgpuBackend>, u32)>,
    attn_norm1_out: Option<ActTapBuf>,
    modulated_attn_in: Option<ActTapBuf>,
    attn_q: Option<ActTapBuf>,
    attn_k: Option<ActTapBuf>,
    attn_v: Option<ActTapBuf>,
    attn_q_norm: Option<ActTapBuf>,
    attn_k_norm: Option<ActTapBuf>,
    attn_q_rope: Option<ActTapBuf>,
    attn_k_rope: Option<ActTapBuf>,
    attn_sdpa: Option<ActTapBuf>,
    attn_out: Option<ActTapBuf>,
    attn_norm2_out: Option<ActTapBuf>,
    x_mid: Option<ActTapBuf>,
    ffn_norm1_out: Option<ActTapBuf>,
    modulated_ffn_in: Option<ActTapBuf>,
    ffn_raw: Option<ActTapBuf>,
    ffn_norm2_out: Option<ActTapBuf>,
    /// Pre-act_quant snapshot of the qkv matmul scratch (raw f16, `rows *
    /// n_qkv` elements). Only populated under I8. `n_elems` here is the
    /// element count, NOT byte count.
    attn_qkv_f16_pre_quant: Option<(WsBuf<WgpuBackend>, u32)>,
    /// Same idea for the attention output projection scratch (raw f16,
    /// `rows * dim` elements).
    attn_proj_f16_pre_quant: Option<(WsBuf<WgpuBackend>, u32)>,
    /// Pre-act_quant f16 scratch of FFN w1 (`rows * hid` elements).
    ffn_h1_f16_pre_quant: Option<(WsBuf<WgpuBackend>, u32)>,
    /// Pre-act_quant f16 scratch of FFN w3 (`rows * hid` elements).
    ffn_h3_f16_pre_quant: Option<(WsBuf<WgpuBackend>, u32)>,
    /// Pre-act_quant f16 scratch of FFN w2 (`rows * dim` elements).
    ffn_h2_f16_pre_quant: Option<(WsBuf<WgpuBackend>, u32)>,
    /// DIAG raw byte snapshots, sized in `alloc`. Each tuple is (buf, n_bytes).
    proj_sa_data_head: Option<(WsBuf<WgpuBackend>, u64)>,
    proj_sa_scale_head: Option<(WsBuf<WgpuBackend>, u64)>,
    proj_wo_b_i8_head: Option<(WsBuf<WgpuBackend>, u64)>,
    proj_wo_b_scale_head: Option<(WsBuf<WgpuBackend>, u64)>,
    /// QKV-site byte snapshots (block 26 matmul_i8 audit). All `(buf, n_bytes)`.
    qkv_attn_in_data_head: Option<(WsBuf<WgpuBackend>, u64)>,
    qkv_attn_in_params_head: Option<(WsBuf<WgpuBackend>, u64)>,
    qkv_b_i8_head: Option<(WsBuf<WgpuBackend>, u64)>,
    qkv_b_scale_head: Option<(WsBuf<WgpuBackend>, u64)>,
    qkv_b_qsum_head: Option<(WsBuf<WgpuBackend>, u64)>,
    qkv_dbg_trace_head: Option<(WsBuf<WgpuBackend>, u64)>,
}

impl Block0TapBufs {
    fn alloc(
        req: &Block0Taps,
        scratch: &Workspace<WgpuBackend>,
        pipelines: &BlockPipelines,
        seq_u: u32,
        dim: u32,
        cfg: BlockConfig,
    ) -> Result<Self, WgpuError> {
        let rows = seq_u; // cfg.batch=1; cfg.seq=seq_u
        let hq = cfg.n_heads as u32;
        let hkv = cfg.n_kv_heads as u32;
        let hd = cfg.head_dim as u32;
        // AdaLN slots follow the block act dtype.
        let mk_ada =
            |want: bool, n_elems: u32| -> Result<Option<(WsBuf<WgpuBackend>, u32)>, WgpuError> {
                if want {
                    let buf = scratch.alloc(pipelines.act_bytes(n_elems))?;
                    Ok(Some((buf, n_elems)))
                } else {
                    Ok(None)
                }
            };
        // Dense activation-bearing slots.
        let mk_act = |want: bool, r: u32, inner: u32| -> Result<Option<ActTapBuf>, WgpuError> {
            if !want {
                return Ok(None);
            }
            let data = scratch.alloc(pipelines.act_bytes(r * inner))?;
            Ok(Some(ActTapBuf {
                data,
                scale: None,
                rows: r,
                inner,
            }))
        };
        // Paired slot for sources that are sdpa_i8 I/O (packed-i8 data +
        // per-(rows, inner/32) f32 scale) so `act_readback_i8_to_f32` can
        // decode at read_back time.
        let mk_act_paired =
            |want: bool, r: u32, inner: u32| -> Result<Option<ActTapBuf>, WgpuError> {
                if !want {
                    return Ok(None);
                }
                if !pipelines.i8_sdpa() {
                    return mk_act(true, r, inner);
                }
                let data = scratch.alloc(r as u64 * inner as u64)?;
                let scale = scratch.alloc(BlockPipelines::i8_scale_bytes(r, inner))?;
                Ok(Some(ActTapBuf {
                    data,
                    scale: Some(scale),
                    rows: r,
                    inner,
                }))
            };
        // AdaLN chunks are per-batch single rows of `dim`.
        let chunk_elems = cfg.batch as u32 * dim;
        // AdaLN pre/full are `[batch, 4*dim]`.
        let adaln_full_elems = cfg.batch as u32 * 4 * dim;
        // AdaLN input is `[batch, adaln_embed_dim]`.
        let adaln_input_elems = cfg.batch as u32 * cfg.adaln_embed_dim as u32;
        Ok(Self {
            adaln_input: mk_ada(req.adaln_input.is_some(), adaln_input_elems)?,
            adaln_pre: mk_ada(req.adaln_pre.is_some(), adaln_full_elems)?,
            adaln_full: mk_ada(req.adaln_full.is_some(), adaln_full_elems)?,
            scale_msa: mk_ada(req.scale_msa.is_some(), chunk_elems)?,
            gate_msa: mk_ada(req.gate_msa.is_some(), chunk_elems)?,
            scale_mlp: mk_ada(req.scale_mlp.is_some(), chunk_elems)?,
            gate_mlp: mk_ada(req.gate_mlp.is_some(), chunk_elems)?,
            attn_norm1_out: mk_act(req.attn_norm1_out.is_some(), rows, dim)?,
            modulated_attn_in: mk_act(req.modulated_attn_in.is_some(), rows, dim)?,
            attn_q: mk_act(req.attn_q.is_some(), rows, hq * hd)?,
            attn_k: mk_act(req.attn_k.is_some(), rows, hkv * hd)?,
            attn_v: mk_act(req.attn_v.is_some(), rows, hkv * hd)?,
            attn_q_norm: mk_act(req.attn_q_norm.is_some(), rows * hq, hd)?,
            attn_k_norm: mk_act(req.attn_k_norm.is_some(), rows * hkv, hd)?,
            attn_q_rope: mk_act(req.attn_q_rope.is_some(), rows, hq * hd)?,
            attn_k_rope: mk_act(req.attn_k_rope.is_some(), rows, hkv * hd)?,
            attn_sdpa: mk_act_paired(req.attn_sdpa.is_some(), rows, hq * hd)?,
            attn_out: mk_act(req.attn_out.is_some(), rows, dim)?,
            attn_norm2_out: mk_act(req.attn_norm2_out.is_some(), rows, dim)?,
            x_mid: mk_act(req.x_mid.is_some(), rows, dim)?,
            ffn_norm1_out: mk_act(req.ffn_norm1_out.is_some(), rows, dim)?,
            modulated_ffn_in: mk_act(req.modulated_ffn_in.is_some(), rows, dim)?,
            ffn_raw: mk_act(req.ffn_raw.is_some(), rows, dim)?,
            ffn_norm2_out: mk_act(req.ffn_norm2_out.is_some(), rows, dim)?,
            attn_qkv_f16_pre_quant: if req.attn_qkv_f16_pre_quant.is_some() {
                let n_qkv = 3 * hq * hd;
                let n_elems = rows * n_qkv;
                let buf = scratch.alloc(pipelines.act_bytes(n_elems))?;
                Some((buf, n_elems))
            } else {
                None
            },
            attn_proj_f16_pre_quant: if req.attn_proj_f16_pre_quant.is_some() {
                let n_elems = rows * dim;
                let buf = scratch.alloc(pipelines.act_bytes(n_elems))?;
                Some((buf, n_elems))
            } else {
                None
            },
            ffn_h1_f16_pre_quant: if req.ffn_h1_f16_pre_quant.is_some() {
                let hid = cfg.ffn_hidden as u32;
                let n_elems = rows * hid;
                let buf = scratch.alloc(pipelines.act_bytes(n_elems))?;
                Some((buf, n_elems))
            } else {
                None
            },
            ffn_h3_f16_pre_quant: if req.ffn_h3_f16_pre_quant.is_some() {
                let hid = cfg.ffn_hidden as u32;
                let n_elems = rows * hid;
                let buf = scratch.alloc(pipelines.act_bytes(n_elems))?;
                Some((buf, n_elems))
            } else {
                None
            },
            ffn_h2_f16_pre_quant: if req.ffn_h2_f16_pre_quant.is_some() {
                let n_elems = rows * dim;
                let buf = scratch.alloc(pipelines.act_bytes(n_elems))?;
                Some((buf, n_elems))
            } else {
                None
            },
            // DIAG heads: small probe buffers. sa heads only exist under
            // i8_sdpa (sa is paired there); the matmul-site heads need the
            // DP4A path on the relevant site. Sizes are chosen so one full
            // K-row (3840 i8 = 3840 bytes) fits for data and a single row of
            // scales (120 f32 = 480 bytes) fits for scale, x4 rows.
            proj_sa_data_head: if req.proj_sa_data_head.is_some() && pipelines.i8_sdpa() {
                let n_bytes: u64 = 4 * 3840;
                Some((scratch.alloc(n_bytes)?, n_bytes))
            } else {
                None
            },
            proj_sa_scale_head: if req.proj_sa_scale_head.is_some() && pipelines.i8_sdpa() {
                let n_bytes: u64 = 4 * 120 * 4;
                Some((scratch.alloc(n_bytes)?, n_bytes))
            } else {
                None
            },
            proj_wo_b_i8_head: if req.proj_wo_b_i8_head.is_some()
                && pipelines.matmul_i8_proj.is_some()
            {
                let n_bytes: u64 = 4 * 3840;
                Some((scratch.alloc(n_bytes)?, n_bytes))
            } else {
                None
            },
            proj_wo_b_scale_head: if req.proj_wo_b_scale_head.is_some()
                && pipelines.matmul_i8_proj.is_some()
            {
                let n_bytes: u64 = 4 * 120 * 4;
                Some((scratch.alloc(n_bytes)?, n_bytes))
            } else {
                None
            },
            // QKV-site heads. K = `dim` → dim/32 K-blocks of 32. Capture FULL
            // acts rows (1024-row oversize covers padding mismatch) and the
            // FULL fused QKV weight (3*dim N-rows) so e2e_parity can pick any
            // outlier (row, col) for CPU recompute. Params/scale/qsum per row
            // = dim/32 entries of 4 bytes.
            qkv_attn_in_data_head: if req.qkv_attn_in_data_head.is_some()
                && pipelines.matmul_i8_qkv.is_some()
            {
                let n_bytes: u64 = 1024 * dim as u64;
                Some((scratch.alloc(n_bytes)?, n_bytes))
            } else {
                None
            },
            qkv_attn_in_params_head: if req.qkv_attn_in_params_head.is_some()
                && pipelines.matmul_i8_qkv.is_some()
            {
                let n_bytes: u64 = 1024 * (dim as u64 / 32) * 4;
                Some((scratch.alloc(n_bytes)?, n_bytes))
            } else {
                None
            },
            qkv_b_i8_head: if req.qkv_b_i8_head.is_some() && pipelines.matmul_i8_qkv.is_some() {
                let n_bytes: u64 = 3 * dim as u64 * dim as u64;
                Some((scratch.alloc(n_bytes)?, n_bytes))
            } else {
                None
            },
            qkv_b_scale_head: if req.qkv_b_scale_head.is_some() && pipelines.matmul_i8_qkv.is_some()
            {
                let n_bytes: u64 = 3 * dim as u64 * (dim as u64 / 32) * 4;
                Some((scratch.alloc(n_bytes)?, n_bytes))
            } else {
                None
            },
            qkv_b_qsum_head: if req.qkv_b_qsum_head.is_some() && pipelines.matmul_i8_qkv.is_some() {
                let n_bytes: u64 = 3 * dim as u64 * (dim as u64 / 32) * 4;
                Some((scratch.alloc(n_bytes)?, n_bytes))
            } else {
                None
            },
            qkv_dbg_trace_head: if req.qkv_dbg_trace_head.is_some()
                && pipelines.matmul_i8_qkv.is_some()
            {
                let n_bytes: u64 = (dim as u64 / 32) * 8 * 4 + 16 * 4;
                Some((scratch.alloc(n_bytes)?, n_bytes))
            } else {
                None
            },
        })
    }

    fn to_block_debug_taps(&self) -> BlockDebugTaps {
        let r =
            |slot: &Option<(WsBuf<WgpuBackend>, u32)>| slot.as_ref().map(|(b, _)| b.as_buf_ref());
        let a = |slot: &Option<ActTapBuf>| {
            slot.as_ref().map(|t| crate::z_image::block::ActTapBufRef {
                data: t.data.as_buf_ref(),
                scale: t.scale.as_ref().map(|s| s.as_buf_ref()),
            })
        };
        BlockDebugTaps {
            adaln_input: r(&self.adaln_input),
            adaln_pre: r(&self.adaln_pre),
            adaln_full: r(&self.adaln_full),
            scale_msa: r(&self.scale_msa),
            gate_msa: r(&self.gate_msa),
            scale_mlp: r(&self.scale_mlp),
            gate_mlp: r(&self.gate_mlp),
            attn_norm1_out: a(&self.attn_norm1_out),
            modulated_attn_in: a(&self.modulated_attn_in),
            attn_q: a(&self.attn_q),
            attn_k: a(&self.attn_k),
            attn_v: a(&self.attn_v),
            attn_q_norm: a(&self.attn_q_norm),
            attn_k_norm: a(&self.attn_k_norm),
            attn_q_rope: a(&self.attn_q_rope),
            attn_k_rope: a(&self.attn_k_rope),
            attn_sdpa: a(&self.attn_sdpa),
            attn_out: a(&self.attn_out),
            attn_norm2_out: a(&self.attn_norm2_out),
            x_mid: a(&self.x_mid),
            ffn_norm1_out: a(&self.ffn_norm1_out),
            modulated_ffn_in: a(&self.modulated_ffn_in),
            ffn_raw: a(&self.ffn_raw),
            ffn_norm2_out: a(&self.ffn_norm2_out),
            attn_qkv_f16_pre_quant: self
                .attn_qkv_f16_pre_quant
                .as_ref()
                .map(|(b, _)| b.as_buf_ref()),
            attn_proj_f16_pre_quant: self
                .attn_proj_f16_pre_quant
                .as_ref()
                .map(|(b, _)| b.as_buf_ref()),
            ffn_h1_f16_pre_quant: self
                .ffn_h1_f16_pre_quant
                .as_ref()
                .map(|(b, _)| b.as_buf_ref()),
            ffn_h3_f16_pre_quant: self
                .ffn_h3_f16_pre_quant
                .as_ref()
                .map(|(b, _)| b.as_buf_ref()),
            ffn_h2_f16_pre_quant: self
                .ffn_h2_f16_pre_quant
                .as_ref()
                .map(|(b, _)| b.as_buf_ref()),
            proj_sa_data_head: self.proj_sa_data_head.as_ref().map(|(b, _)| b.as_buf_ref()),
            proj_sa_scale_head: self
                .proj_sa_scale_head
                .as_ref()
                .map(|(b, _)| b.as_buf_ref()),
            proj_wo_b_i8_head: self.proj_wo_b_i8_head.as_ref().map(|(b, _)| b.as_buf_ref()),
            proj_wo_b_scale_head: self
                .proj_wo_b_scale_head
                .as_ref()
                .map(|(b, _)| b.as_buf_ref()),
            qkv_attn_in_data_head: self
                .qkv_attn_in_data_head
                .as_ref()
                .map(|(b, _)| b.as_buf_ref()),
            qkv_attn_in_params_head: self
                .qkv_attn_in_params_head
                .as_ref()
                .map(|(b, _)| b.as_buf_ref()),
            qkv_b_i8_head: self.qkv_b_i8_head.as_ref().map(|(b, _)| b.as_buf_ref()),
            qkv_b_scale_head: self.qkv_b_scale_head.as_ref().map(|(b, _)| b.as_buf_ref()),
            qkv_b_qsum_head: self.qkv_b_qsum_head.as_ref().map(|(b, _)| b.as_buf_ref()),
            qkv_dbg_trace_head: self
                .qkv_dbg_trace_head
                .as_ref()
                .map(|(b, _)| b.as_buf_ref()),
        }
    }

    async fn read_back(
        self,
        backend: &WgpuBackend,
        pipelines: &BlockPipelines,
        sink: &mut Block0Taps,
    ) -> Result<(), WgpuError> {
        // AdaLN slots follow the block act dtype.
        let ada = pipelines.act_dtype;
        macro_rules! rd_ada {
            ($field:ident) => {
                if let Some((buf, n)) = self.$field {
                    let v = sink.$field.get_or_insert_with(Vec::new);
                    read_into_f32_tagged(
                        backend,
                        &buf.as_buf_ref(),
                        n as usize,
                        ada,
                        v,
                        concat!("block0.", stringify!($field)),
                    )
                    .await?;
                }
            };
        }
        // Activation slots: paired taps (sdpa_i8 I/O sources) decode via
        // seq::act_readback_i8_to_f32; dense taps read at block dtype.
        macro_rules! rd_act {
            ($field:ident) => {
                if let Some(t) = self.$field {
                    let v = sink.$field.get_or_insert_with(Vec::new);
                    if let Some(scale) = t.scale.as_ref() {
                        let d = t.data.as_buf_ref();
                        let s = scale.as_buf_ref();
                        let data_bytes = backend
                            .read_buffer(d.id, d.offset, (t.rows * t.inner) as u64)
                            .await?;
                        let scale_bytes = backend
                            .read_buffer(
                                s.id,
                                s.offset,
                                BlockPipelines::i8_scale_bytes(t.rows, t.inner),
                            )
                            .await?;
                        let out = seq::act_readback_i8_to_f32(
                            &data_bytes,
                            &scale_bytes,
                            t.rows as usize,
                            t.inner as usize,
                        );
                        v.clear();
                        v.extend(out);
                    } else {
                        read_into_f32_tagged(
                            backend,
                            &t.data.as_buf_ref(),
                            (t.rows * t.inner) as usize,
                            pipelines.act_dtype,
                            v,
                            concat!("block0.", stringify!($field)),
                        )
                        .await?;
                    }
                }
            };
        }
        rd_ada!(adaln_input);
        rd_ada!(adaln_pre);
        rd_ada!(adaln_full);
        rd_ada!(scale_msa);
        rd_ada!(gate_msa);
        rd_ada!(scale_mlp);
        rd_ada!(gate_mlp);
        rd_act!(attn_norm1_out);
        rd_act!(modulated_attn_in);
        rd_act!(attn_q);
        rd_act!(attn_k);
        rd_act!(attn_v);
        rd_act!(attn_q_norm);
        rd_act!(attn_k_norm);
        rd_act!(attn_q_rope);
        rd_act!(attn_k_rope);
        rd_act!(attn_sdpa);
        rd_act!(attn_out);
        rd_act!(attn_norm2_out);
        rd_act!(x_mid);
        rd_act!(ffn_norm1_out);
        rd_act!(modulated_ffn_in);
        rd_act!(ffn_raw);
        rd_act!(ffn_norm2_out);
        if let Some((buf, n)) = self.attn_qkv_f16_pre_quant {
            let v = sink.attn_qkv_f16_pre_quant.get_or_insert_with(Vec::new);
            read_into_f32_tagged(
                backend,
                &buf.as_buf_ref(),
                n as usize,
                pipelines.act_dtype,
                v,
                "block0.attn_qkv_f16_pre_quant",
            )
            .await?;
        }
        if let Some((buf, n)) = self.attn_proj_f16_pre_quant {
            let v = sink.attn_proj_f16_pre_quant.get_or_insert_with(Vec::new);
            read_into_f32_tagged(
                backend,
                &buf.as_buf_ref(),
                n as usize,
                pipelines.act_dtype,
                v,
                "block0.attn_proj_f16_pre_quant",
            )
            .await?;
        }
        if let Some((buf, n)) = self.ffn_h1_f16_pre_quant {
            let v = sink.ffn_h1_f16_pre_quant.get_or_insert_with(Vec::new);
            read_into_f32_tagged(
                backend,
                &buf.as_buf_ref(),
                n as usize,
                pipelines.act_dtype,
                v,
                "block.ffn_h1_f16_pre_quant",
            )
            .await?;
        }
        if let Some((buf, n)) = self.ffn_h3_f16_pre_quant {
            let v = sink.ffn_h3_f16_pre_quant.get_or_insert_with(Vec::new);
            read_into_f32_tagged(
                backend,
                &buf.as_buf_ref(),
                n as usize,
                pipelines.act_dtype,
                v,
                "block.ffn_h3_f16_pre_quant",
            )
            .await?;
        }
        if let Some((buf, n)) = self.ffn_h2_f16_pre_quant {
            let v = sink.ffn_h2_f16_pre_quant.get_or_insert_with(Vec::new);
            read_into_f32_tagged(
                backend,
                &buf.as_buf_ref(),
                n as usize,
                pipelines.act_dtype,
                v,
                "block.ffn_h2_f16_pre_quant",
            )
            .await?;
        }
        macro_rules! rd_bytes {
            ($field:ident) => {
                if let Some((buf, n_bytes)) = self.$field {
                    let b = buf.as_buf_ref();
                    let bytes = backend.read_buffer(b.id, b.offset, n_bytes).await?;
                    *sink.$field.get_or_insert_with(Vec::new) = bytes;
                }
            };
        }
        rd_bytes!(proj_sa_data_head);
        rd_bytes!(proj_sa_scale_head);
        rd_bytes!(proj_wo_b_i8_head);
        rd_bytes!(proj_wo_b_scale_head);
        rd_bytes!(qkv_attn_in_data_head);
        rd_bytes!(qkv_attn_in_params_head);
        rd_bytes!(qkv_b_i8_head);
        rd_bytes!(qkv_b_scale_head);
        rd_bytes!(qkv_b_qsum_head);
        rd_bytes!(qkv_dbg_trace_head);
        Ok(())
    }
}

/// Decode a packed-bf16 or f32 GPU buffer into a Rust `Vec<f32>`. The buffer's
/// physical byte length is `n_f32 * act.bytes_per_elem()`. Used by tap
/// readbacks where the activation dtype is set by `BlockPipelines` at compile
/// time and the host code wants finite-value diagnostics regardless.
async fn read_into_f32(
    backend: &WgpuBackend,
    buf: &BufRef,
    n_f32: usize,
    act: ActDtype,
    sink: &mut Vec<f32>,
) -> Result<(), WgpuError> {
    read_into_f32_tagged(backend, buf, n_f32, act, sink, "").await
}

/// Like [`read_into_f32`] but also emits a `[TAP-RAW]` tracing line with the
/// first/last 32 raw bytes (hex) of the readback when `tag` is non-empty and
/// DIAG-level tracing is enabled. The hex dump is the only signal that
/// distinguishes "buffer is 0x00 throughout" (write never landed / wrong
/// alias) from "buffer has data but decode is wrong" (dtype/stride mismatch).
async fn read_into_f32_tagged(
    backend: &WgpuBackend,
    buf: &BufRef,
    n_f32: usize,
    act: ActDtype,
    sink: &mut Vec<f32>,
    tag: &str,
) -> Result<(), WgpuError> {
    let bytes_per_elem = act.bytes_per_elem() as usize;
    let bytes = backend
        .read_buffer(buf.id, buf.offset, (n_f32 * bytes_per_elem) as u64)
        .await?;
    if !tag.is_empty() && tracing::enabled!(target: trace::DIAG, tracing::Level::INFO) {
        let n_bytes = bytes.len();
        let head_n = 32.min(n_bytes);
        let tail_lo = n_bytes.saturating_sub(32);
        let to_hex = |slice: &[u8]| -> String {
            let mut s = String::with_capacity(slice.len() * 3);
            for (i, b) in slice.iter().enumerate() {
                if i > 0 && i % 4 == 0 {
                    s.push(' ');
                }
                s.push_str(&format!("{:02x}", b));
            }
            s
        };
        tracing::info!(
            target: trace::DIAG,
            "  [TAP-RAW] {tag}: act={act:?} n_bytes={n_bytes} head=[{}] tail=[{}]",
            to_hex(&bytes[..head_n]),
            to_hex(&bytes[tail_lo..]),
        );
    }
    sink.clear();
    sink.reserve(n_f32);
    match act {
        ActDtype::F32 => {
            for c in bytes.chunks_exact(4) {
                sink.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        ActDtype::Bf16 => {
            for c in bytes.chunks_exact(2) {
                let half = u16::from_le_bytes([c[0], c[1]]);
                sink.push(f32::from_bits((half as u32) << 16));
            }
        }
        ActDtype::F16 => {
            for c in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                sink.push(half::f16::from_bits(bits).to_f32());
            }
        }
        ActDtype::I8 => unreachable!("I8 is never a block act_dtype"),
    }
    Ok(())
}

/// Whole-buffer stats + 4-row buckets + head/tail samples, gated on DIAG.
/// Cheap host-side reduction over an already-decoded f32 buffer.
fn diag_print_stats(label: &str, v: &[f32]) {
    if v.is_empty() {
        return;
    }
    let mut nan = 0usize;
    let mut pinf = 0usize;
    let mut ninf = 0usize;
    let mut zeros = 0usize;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut n_fin = 0usize;
    for &x in v {
        if x.is_nan() {
            nan += 1;
            continue;
        }
        if x == f32::INFINITY {
            pinf += 1;
            continue;
        }
        if x == f32::NEG_INFINITY {
            ninf += 1;
            continue;
        }
        if x == 0.0 {
            zeros += 1;
        }
        n_fin += 1;
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
        sum += x as f64;
        sum_abs += x.abs() as f64;
    }
    let (mean, mean_abs) = if n_fin > 0 {
        (sum / n_fin as f64, sum_abs / n_fin as f64)
    } else {
        (0.0, 0.0)
    };
    let n = v.len();
    let mut b_means = [0.0f64; 4];
    for (bi, b) in b_means.iter_mut().enumerate() {
        let lo = bi * n / 4;
        let hi = (bi + 1) * n / 4;
        let mut s = 0.0f64;
        let mut c = 0usize;
        for &x in &v[lo..hi] {
            if x.is_finite() {
                s += x.abs() as f64;
                c += 1;
            }
        }
        *b = if c > 0 { s / c as f64 } else { 0.0 };
    }
    let head_n = 8.min(n);
    let tail_lo = n.saturating_sub(8);
    tracing::info!(
        target: trace::DIAG,
        "  [DIT-PROBE] {label}: len={n} nan={nan} +inf={pinf} -inf={ninf} \
         zeros={zeros} min={:.4e} max={:.4e} mean={:.4e} mean_abs={:.4e} \
         buckets=[{:.4e},{:.4e},{:.4e},{:.4e}] head={:?} tail={:?}",
        min,
        max,
        mean,
        mean_abs,
        b_means[0],
        b_means[1],
        b_means[2],
        b_means[3],
        &v[..head_n],
        &v[tail_lo..],
    );
}

/// DIAG-gated probe: read a dense buffer back at `act` dtype and print stats.
/// Zero cost when DIAG INFO is disabled.
async fn diag_probe_dense(
    backend: &WgpuBackend,
    buf: &BufRef,
    n_elems: usize,
    act: ActDtype,
    label: &str,
) -> Result<(), WgpuError> {
    if !tracing::enabled!(target: trace::DIAG, tracing::Level::INFO) {
        return Ok(());
    }
    let mut v = Vec::<f32>::with_capacity(n_elems);
    read_into_f32(backend, buf, n_elems, act, &mut v).await?;
    diag_print_stats(label, &v);
    Ok(())
}

/// DIAG-gated probe: read a `ResStream` back to f32 and print stats. Zero
/// cost when DIAG INFO is disabled.
async fn diag_probe_resstream(
    backend: &WgpuBackend,
    pipelines: &BlockPipelines,
    src: &ResStream,
    rows: u32,
    dim: u32,
    label: &str,
) -> Result<(), WgpuError> {
    if !tracing::enabled!(target: trace::DIAG, tracing::Level::INFO) {
        return Ok(());
    }
    let mut v = Vec::<f32>::new();
    read_resstream_to_f32(backend, pipelines, src, rows, dim, &mut v, "").await?;
    diag_print_stats(label, &v);
    Ok(())
}

/// DIAG-gated probe variant that splits the residual stream into the
/// image-row window (rows `0..img_rows`) and the caption-row window
/// (rows `img_rows..total_rows`) and emits one summary per window in
/// addition to the combined one. Aggregate stats across the whole
/// stream can mask divergence localized to one window — pyref dumps
/// the image-row sub-tensor only, so a window-restricted comparison
/// is the apples-to-apples check.
async fn diag_probe_resstream_split(
    backend: &WgpuBackend,
    pipelines: &BlockPipelines,
    src: &ResStream,
    img_rows: u32,
    ctx_rows: u32,
    dim: u32,
    label: &str,
) -> Result<(), WgpuError> {
    if !tracing::enabled!(target: trace::DIAG, tracing::Level::INFO) {
        return Ok(());
    }
    let total = img_rows + ctx_rows;
    let mut v = Vec::<f32>::new();
    read_resstream_to_f32(backend, pipelines, src, total, dim, &mut v, "").await?;
    diag_print_stats(label, &v);
    let split = (img_rows as usize) * (dim as usize);
    if split <= v.len() {
        diag_print_stats(&format!("{label}__img"), &v[..split]);
        diag_print_stats(&format!("{label}__ctx"), &v[split..]);
        // Per-row mean_abs for the first 4 and last 4 image rows — a
        // transpose / row-permutation / unpatchify bug shows up as wildly
        // uneven row magnitudes here even when the aggregate looks fine.
        let d = dim as usize;
        let nrows = img_rows as usize;
        let probe_first = nrows.min(4);
        let probe_last = if nrows > 8 { 4 } else { 0 };
        for r in 0..probe_first {
            let row = &v[r * d..(r + 1) * d];
            let (mut s, mut mx) = (0.0f64, 0f32);
            for &x in row {
                if x.is_finite() {
                    s += x.abs() as f64;
                    if x.abs() > mx {
                        mx = x.abs();
                    }
                }
            }
            tracing::info!(
                target: trace::DIAG,
                "  [DIT-PROBE-ROW] {label}__img row={r} mean_abs={:.4e} max_abs={:.4e} head={:?}",
                s / d as f64,
                mx,
                &row[..8.min(d)],
            );
        }
        for off in 0..probe_last {
            let r = nrows - probe_last + off;
            let row = &v[r * d..(r + 1) * d];
            let (mut s, mut mx) = (0.0f64, 0f32);
            for &x in row {
                if x.is_finite() {
                    s += x.abs() as f64;
                    if x.abs() > mx {
                        mx = x.abs();
                    }
                }
            }
            tracing::info!(
                target: trace::DIAG,
                "  [DIT-PROBE-ROW] {label}__img row={r} mean_abs={:.4e} max_abs={:.4e} head={:?}",
                s / d as f64,
                mx,
                &row[..8.min(d)],
            );
        }
    }
    Ok(())
}

async fn read_resstream_to_f32(
    backend: &WgpuBackend,
    pipelines: &BlockPipelines,
    src: &ResStream,
    rows: u32,
    dim: u32,
    sink: &mut Vec<f32>,
    tag: &str,
) -> Result<(), WgpuError> {
    read_into_f32_tagged(
        backend,
        &src.data.as_buf_ref(),
        (rows * dim) as usize,
        pipelines.act_dtype,
        sink,
        tag,
    )
    .await
}
