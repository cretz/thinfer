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

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError};
use thinfer_core::ops::{ActDtype, ScatterPadRowsF32};
use thinfer_core::residency::{ResidencyError, WeightResidency};
use thinfer_core::trace::{self, PHASE};
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::{Workspace, WsBuf};
use tracing::Instrument;

use crate::z_image::block::{Block, BlockConfig, BlockHandles, BlockPipelines};
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
        pipelines: &BlockPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        inputs: &DitInputs<'a>,
    ) -> Result<DitForwardLayout, DitError<S::Error>> {
        self.forward_with_taps(
            backend,
            pipelines,
            residency,
            scratch,
            inputs,
            DitTaps::default(),
        )
        .await
    }

    pub async fn forward_with_taps<'a, S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipelines: &BlockPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        inputs: &DitInputs<'a>,
        mut taps: DitTaps<'_>,
    ) -> Result<DitForwardLayout, DitError<S::Error>> {
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

        // --- 2. upload x tokens, run XEmbedder ---
        let x_tok_bytes = act_upload_bytes(pipelines.act_dtype, &px.tokens);
        let x_tok = scratch.alloc(x_tok_bytes.len() as u64)?;
        backend.write_buffer(x_tok.id, 0, &x_tok_bytes)?;
        let x_act = scratch.alloc(pipelines.act_bytes(seq_x * dim))?;
        {
            let views = self.x_embedder_handles.acquire(residency, backend).await?;
            let bufs = views.bufs();
            let x_tok_ref = x_tok.as_buf_ref();
            let x_act_ref = x_act.as_buf_ref();
            let scope = scratch.batch();
            self.x_embedder.forward(
                &scope,
                pipelines,
                scope.import(&x_tok_ref),
                seq_x,
                scope.import(&x_act_ref),
                &bufs,
            )?;
            scope.submit_void().await?;
        }
        // --- 3. scatter pad token into pad rows ---
        {
            let pad = residency.acquire(self.x_pad_token, backend).await?;
            scatter_pad_rows(
                backend,
                pipelines,
                scratch,
                x_act.as_buf_ref(),
                pad.buf(),
                seq_x,
                dim,
                &px.pad_mask,
            )
            .await?;
        }

        // --- 4. rope freqs for x ---
        let x_freqs_bytes =
            seq::act_upload_bytes(pipelines.act_dtype, &self.rope.lookup(&px.pos_ids));
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

        // --- 6. noise_refiner ---
        // Block expression scopes the rollup span guard `_nr_scope` so it
        // exits when the loop is done; x_cur escapes as the block's value.
        let x_cur: WsBuf<WgpuBackend> = {
            let _nr_scope =
                trace::scope!("dit.noise_refiner", n = self.noise_refiner.len()).entered();
            let mut x_cur: WsBuf<WgpuBackend> = x_act;
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
                let nxt = scratch.alloc(pipelines.act_bytes(seq_x * dim))?;
                let views = pending
                    .take()
                    .expect("pending noise_refiner acquire missing");
                let bufs = views.bufs();
                let x_cur_ref = x_cur.as_buf_ref();
                let freqs_ref = x_freqs.as_buf_ref();
                let mask_ref = x_mask.as_buf_ref();
                let t_emb_ref = t_emb.as_buf_ref();
                let nxt_ref = nxt.as_buf_ref();
                let next_idx = idx + 1;
                let submit_res = {
                    let scope = scratch.batch();
                    blk.forward(
                        &scope,
                        pipelines,
                        scope.import(&x_cur_ref),
                        scope.import(&freqs_ref),
                        scope.import(&mask_ref),
                        Some(scope.import(&t_emb_ref)),
                        scope.import(&nxt_ref),
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
                    let submit_fut = scope.submit_void().instrument(
                        tracing::debug_span!(target: PHASE, "dit.submit", phase = "noise_refiner", idx),
                    );
                    let (s_res, n_res, p_res) =
                        futures::join!(submit_fut, next_acquire, prefetch_after);
                    p_res?;
                    pending = n_res?;
                    s_res
                };
                submit_res?;
                x_cur = nxt;
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
        let cap_in_bytes = act_upload_bytes(pipelines.act_dtype, &cap_feats_padded);
        let cap_in = scratch.alloc(cap_in_bytes.len() as u64)?;
        backend.write_buffer(cap_in.id, 0, &cap_in_bytes)?;
        let cap_act = scratch.alloc(pipelines.act_bytes(seq_cap * dim))?;
        let (cap_intermediate_normed, cap_intermediate_pre_bias) = {
            let views = self
                .cap_embedder_handles
                .acquire(residency, backend)
                .await?;
            let bufs = views.bufs();
            let cap_in_ref = cap_in.as_buf_ref();
            let cap_act_ref = cap_act.as_buf_ref();
            let scope = scratch.batch();
            let inter = self.cap_embedder.forward(
                &scope,
                pipelines,
                scope.import(&cap_in_ref),
                seq_cap,
                scope.import(&cap_act_ref),
                &bufs,
            )?;
            let outs = scope.submit_many(&[inter.normed, inter.pre_bias]).await?;
            let mut it = outs.into_iter();
            let normed = it.next().unwrap();
            let pre_bias = it.next().unwrap();
            (normed, pre_bias)
        };
        if let Some(sink) = taps.cap_embed_normed.as_deref_mut() {
            read_into_f32(
                backend,
                &cap_intermediate_normed,
                (seq_cap * cap_feat_dim as u32) as usize,
                sink,
            )
            .await?;
        }
        if let Some(sink) = taps.cap_embed_pre_bias.as_deref_mut() {
            read_into_f32(
                backend,
                &cap_intermediate_pre_bias,
                (seq_cap * dim) as usize,
                sink,
            )
            .await?;
        }
        {
            let pad = residency.acquire(self.cap_pad_token, backend).await?;
            scatter_pad_rows(
                backend,
                pipelines,
                scratch,
                cap_act.as_buf_ref(),
                pad.buf(),
                seq_cap as u32,
                dim,
                &cm.pad_mask,
            )
            .await?;
        }

        if let Some(sink) = taps.cap_embedded.as_deref_mut() {
            read_into_f32(backend, &cap_act, (seq_cap * dim) as usize, sink).await?;
        }

        // --- 9. cap rope freqs + attn mask ---
        let cap_freqs_bytes =
            seq::act_upload_bytes(pipelines.act_dtype, &self.rope.lookup(&cm.pos_ids));
        let cap_freqs = scratch.alloc(cap_freqs_bytes.len() as u64)?;
        backend.write_buffer(cap_freqs.id, 0, &cap_freqs_bytes)?;
        let cap_mask_bytes = seq::attn_mask_zero_bytes_act(cm.padded_len, pipelines.act_dtype);
        let cap_mask = scratch.alloc(cap_mask_bytes.len() as u64)?;
        backend.write_buffer(cap_mask.id, 0, &cap_mask_bytes)?;

        // --- 10. context_refiner ---
        // See noise_refiner block above: span guard `_cr_scope` is bounded by
        // the inner block; cap_cur escapes as the block expression's value.
        let cap_cur: WsBuf<WgpuBackend> = {
            let _cr_scope =
                trace::scope!("dit.context_refiner", n = self.context_refiner.len()).entered();
            let mut cap_cur: WsBuf<WgpuBackend> = cap_act;
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
                if idx == 1
                    && let Some(sink) = taps.ctx_refiner_0_out.as_deref_mut()
                {
                    read_into_f32(backend, &cap_cur, (seq_cap * dim) as usize, sink).await?;
                }
                let nxt = scratch.alloc(pipelines.act_bytes(seq_cap * dim))?;
                let views = pending
                    .take()
                    .expect("pending context_refiner acquire missing");
                let bufs = views.bufs();
                let cap_cur_ref = cap_cur.as_buf_ref();
                let freqs_ref = cap_freqs.as_buf_ref();
                let mask_ref = cap_mask.as_buf_ref();
                let nxt_ref = nxt.as_buf_ref();
                let next_idx = idx + 1;
                let submit_res = {
                    let scope = scratch.batch();
                    blk.forward(
                        &scope,
                        pipelines,
                        scope.import(&cap_cur_ref),
                        scope.import(&freqs_ref),
                        scope.import(&mask_ref),
                        None,
                        scope.import(&nxt_ref),
                        &bufs,
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
                    let submit_fut = scope.submit_void().instrument(
                        tracing::debug_span!(target: PHASE, "dit.submit", phase = "context_refiner", idx),
                    );
                    let (s_res, n_res, p_res) =
                        futures::join!(submit_fut, next_acquire, prefetch_after);
                    p_res?;
                    pending = n_res?;
                    s_res
                };
                submit_res?;
                cap_cur = nxt;
            }
            cap_cur
        };

        if let Some(sink) = taps.last_ctx_refiner_out.as_deref_mut() {
            read_into_f32(backend, &cap_cur, (seq_cap * dim) as usize, sink).await?;
        }

        // --- 11. concat unified = [x; cap] ---
        let seq_u = seq_x + seq_cap;
        let unified = scratch.alloc(pipelines.act_bytes(seq_u * dim))?;
        let row_bytes = pipelines.act_bytes(dim);
        let freq_row = (head_dim as u64) * pipelines.act_dtype.bytes_per_elem();
        let unified_freqs = scratch.alloc(seq_u as u64 * freq_row)?;
        {
            let x_cur_ref = x_cur.as_buf_ref();
            let cap_cur_ref = cap_cur.as_buf_ref();
            let x_freqs_ref = x_freqs.as_buf_ref();
            let cap_freqs_ref = cap_freqs.as_buf_ref();
            let unified_ref = unified.as_buf_ref();
            let unified_freqs_ref = unified_freqs.as_buf_ref();
            let scope = scratch.batch();
            let xh = scope.import(&x_cur_ref);
            let ch = scope.import(&cap_cur_ref);
            let xfh = scope.import(&x_freqs_ref);
            let cfh = scope.import(&cap_freqs_ref);
            let uh = scope.import(&unified_ref);
            let ufh = scope.import(&unified_freqs_ref);
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
            scope.submit_void().await?;
        }

        if let Some(sink) = taps.unified_in.as_deref_mut() {
            read_into_f32(backend, &unified, (seq_u * dim) as usize, sink).await?;
        }

        let u_mask = seq::attn_mask_zero_bytes_act(seq_u as usize, pipelines.act_dtype);
        let unified_mask = scratch.alloc(u_mask.len() as u64)?;
        backend.write_buffer(unified_mask.id, 0, &u_mask)?;

        // --- 12. main layers ---
        let _lr_scope = trace::scope!("dit.layers", n = self.layers.len()).entered();
        let mut u_cur: WsBuf<WgpuBackend> = unified;
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
        for (idx, blk) in self.layers.iter().enumerate() {
            let _blk_scope = trace::scope!(format_args!("block.{idx}")).entered();
            if idx == 1
                && let Some(sink) = taps.main_layer_0_out.as_deref_mut()
            {
                read_into_f32(backend, &u_cur, (seq_u * dim) as usize, sink).await?;
            }
            if idx == 15
                && let Some(sink) = taps.main_layer_14_out.as_deref_mut()
            {
                read_into_f32(backend, &u_cur, (seq_u * dim) as usize, sink).await?;
            }
            let nxt = scratch.alloc(pipelines.act_bytes(seq_u * dim))?;
            let views = pending.take().expect("pending layers acquire missing");
            let bufs = views.bufs();
            let u_cur_ref = u_cur.as_buf_ref();
            let freqs_ref = unified_freqs.as_buf_ref();
            let mask_ref = unified_mask.as_buf_ref();
            let t_emb_ref = t_emb.as_buf_ref();
            let nxt_ref = nxt.as_buf_ref();
            let next_idx = idx + 1;
            let submit_res = {
                let scope = scratch.batch();
                blk.forward(
                    &scope,
                    pipelines,
                    scope.import(&u_cur_ref),
                    scope.import(&freqs_ref),
                    scope.import(&mask_ref),
                    Some(scope.import(&t_emb_ref)),
                    scope.import(&nxt_ref),
                    &bufs,
                )?;
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
                let submit_fut = scope.submit_void().instrument(
                    tracing::debug_span!(target: PHASE, "dit.submit", phase = "layers", idx),
                );
                let (s_res, n_res, p_res) =
                    futures::join!(submit_fut, next_acquire, prefetch_after);
                p_res?;
                pending = n_res?;
                s_res
            };
            submit_res?;
            u_cur = nxt;
        }

        if let Some(sink) = taps.last_main_layer_out.as_deref_mut() {
            read_into_f32(backend, &u_cur, (seq_u * dim) as usize, sink).await?;
        }

        // --- 13. final layer ---
        let oc = self.final_layer.cfg.out_channels as u32;
        let final_out = scratch.alloc(pipelines.act_bytes(seq_u * oc))?;
        {
            let views = self.final_layer_handles.acquire(residency, backend).await?;
            let bufs = views.bufs();
            let u_cur_ref = u_cur.as_buf_ref();
            let t_emb_ref = t_emb.as_buf_ref();
            let final_out_ref = final_out.as_buf_ref();
            let scope = scratch.batch();
            self.final_layer.forward(
                &scope,
                pipelines,
                scope.import(&u_cur_ref),
                scope.import(&t_emb_ref),
                seq_u,
                scope.import(&final_out_ref),
                &bufs,
            )?;
            scope.submit_void().await?;
        }

        if let Some(sink) = taps.final_layer_out.as_deref_mut() {
            read_into_f32(backend, &final_out, (seq_u * oc) as usize, sink).await?;
        }

        Ok(DitForwardLayout {
            final_out,
            act_dtype: pipelines.act_dtype,
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
pub async fn scatter_pad_rows(
    backend: &WgpuBackend,
    pipelines: &BlockPipelines,
    scratch: &Workspace<WgpuBackend>,
    dst: BufRef,
    pad_token: BufRef,
    n_rows: u32,
    dim: u32,
    pad_mask: &[u8],
) -> Result<(), WgpuError> {
    assert_eq!(
        pad_mask.len(),
        n_rows as usize,
        "pad_mask len {} != n_rows {}",
        pad_mask.len(),
        n_rows
    );
    let elem_bytes = pipelines.act_dtype.bytes_per_elem();
    debug_assert_eq!(dst.len, (n_rows as u64) * (dim as u64) * elem_bytes);
    debug_assert_eq!(pad_token.len, (dim as u64).div_ceil(2) * 4);

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
        ActDtype::Bf16 => n_rows * (dim / 2),
    };
    scope.scatter_pad_rows::<ScatterPadRowsF32>(
        &pipelines.scatter_pad_rows,
        scope.import(&pad_token),
        scope.import(&mask_ref),
        scope.import(&uniform_ref),
        scope.import(&dst),
        dispatch_elems,
    )?;
    scope.submit_void().await?;
    Ok(())
}

async fn read_into_f32(
    backend: &WgpuBackend,
    buf: &BufRef,
    n_f32: usize,
    sink: &mut Vec<f32>,
) -> Result<(), WgpuError> {
    let bytes = backend
        .read_buffer(buf.id, buf.offset, (n_f32 * 4) as u64)
        .await?;
    sink.clear();
    sink.reserve(n_f32);
    for c in bytes.chunks_exact(4) {
        sink.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    Ok(())
}
