//! Ideogram-4 single-stream DiT driver (`Ideogram4Transformer.forward`).
//!
//! One sample (B=1). The packed sequence is `[text tokens][image tokens]`
//! (no left-pad: a single prompt carries its own length, and the upstream
//! `max_text_tokens` pad rows have `segment_id=-1` so they never influence the
//! image outputs that are read back). All `segment_id`s are then `1`, so
//! attention is full bidirectional and needs no mask buffer (a zero additive
//! mask, identical to the Z-Image B=1 path).
//!
//! Assembly is region-wise, which is exactly equivalent to the upstream
//! mask-twice formulation for B=1 with a contiguous `[text][image]` layout:
//! - text rows = `llm_cond_proj(llm_cond_norm(features))` + indicator[0]
//! - image rows = `input_proj(noise)` + indicator[1]
//!
//! (input_proj over zero-text rows then *image_mask, and llm_cond over
//! zero-image rows then *text_mask, both collapse to "don't add here".)
//!
//! Modulation is the model-agnostic `common::block::Block` (sandwich-norm,
//! tanh-gated 4-chunk AdaLN, QK-RMSNorm, SwiGLU) with `rope_halfrot=true` for
//! the 3-axis MRoPE. The final layer reuses the Z-Image `FinalLayer`
//! (LayerNorm-no-affine + adaln scale + Linear), which matches Ideogram-4's
//! exactly (double-silu on the shared `adaln_input`).

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError};
use thinfer_core::residency::{ResidencyError, WeightResidency};
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::{BatchScope, Workspace, WsBuf};

use crate::common::block::{ActBuf, Block, BlockConfig, BlockHandles, BlockPipelines};
use crate::common::embedders::{LinearBiasBufs, LinearBiasHandles, linear_bias};
use crate::common::seq::{
    act_readback_to_f32, act_upload_bytes, attn_mask_pad_tail_bytes_act, freqs_upload_bytes,
};
use crate::z_image::embedders::{CapEmbedder, CapEmbedderConfig};
use crate::z_image::final_layer::{FinalLayer, FinalLayerConfig};

use super::loader::DitHandles;
use super::packing::{ImageGrid, build_position_ids};
use super::t_embedder::TEmbedder;
use super::text_encoder::config as enc_config;
use super::{config, mrope};

/// Per-forward host inputs (the driver uploads them).
pub struct DitInputs<'a> {
    /// `[num_text, LLM_FEATURES_DIM]` row-major f32 encoder taps (text rows).
    pub llm_features: &'a [f32],
    pub num_text: usize,
    /// `[num_image, IN_CHANNELS]` row-major f32 noise at image positions.
    pub noise: &'a [f32],
    pub grid: ImageGrid,
    /// Scalar flow-matching time in [0, 1].
    pub timestep: f32,
}

/// Optional intermediate readbacks for parity localization (all f32).
#[derive(Default)]
pub struct DitTaps<'a> {
    pub adaln_input: Option<&'a mut Vec<f32>>,
    /// `h` after assembly + indicator, before block 0. `[seq, DIM]`.
    pub h_assembled: Option<&'a mut Vec<f32>>,
    pub block0_out: Option<&'a mut Vec<f32>>,
    pub block_last_out: Option<&'a mut Vec<f32>>,
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

pub struct Ideogram4Dit {
    pub t_embedder: TEmbedder,
    pub llm_cond: CapEmbedder,
    pub final_layer: FinalLayer,
    pub layers: Vec<Block>,
    pub handles: DitHandles,
}

impl Ideogram4Dit {
    /// `seq = num_text + num_image` (the real packed length). Blocks are built
    /// at the EVEN-padded length (`seq.next_multiple_of(2)`): F16 acts pack two
    /// elems per `array<u32>` word, so the SDPA mask needs an even sequence (the
    /// same constraint the Qwen3 encoder even-pads for). The pad rows sit AFTER
    /// the image tokens and are masked out as attention keys; the velocity is
    /// sliced from the image region, ahead of the pad.
    pub fn assemble(handles: DitHandles, seq: usize) -> Self {
        let seq_pad = seq.next_multiple_of(2);
        let block_cfg = BlockConfig {
            dim: config::DIM,
            n_heads: config::N_HEADS,
            n_kv_heads: config::N_HEADS, // no GQA (hq == hkv)
            head_dim: config::HEAD_DIM,
            ffn_hidden: config::FFN_HIDDEN,
            batch: 1,
            seq: seq_pad,
            norm_eps: config::NORM_EPS,
            adaln_embed_dim: config::ADALN_DIM,
            modulation: true,
            rope_halfrot: true,
        };
        let layers = (0..config::N_LAYERS)
            .map(|_| Block::new(block_cfg))
            .collect();
        Self {
            t_embedder: TEmbedder::new(),
            llm_cond: CapEmbedder::new(CapEmbedderConfig {
                cap_feat_dim: enc_config::LLM_FEATURES_DIM,
                dim: config::DIM,
                norm_eps: 1e-6, // llm_cond_norm eps (NOT the block 1e-5)
            }),
            final_layer: FinalLayer::new(FinalLayerConfig {
                dim: config::DIM,
                adaln_embed_dim: config::ADALN_DIM,
                out_channels: config::IN_CHANNELS,
                norm_eps: 1e-6, // final LayerNorm eps
            }),
            layers,
            handles,
        }
    }

    /// Deterministic pin-priority list for the denoise phase (consumed by
    /// `WeightResidency::set_pin_plan`). Same rationale as `ZImageDit`: the DiT
    /// reads every weight once per step, so per pinned byte the saved per-step
    /// upload is identical no matter which bytes; the order only shapes the
    /// residual streaming demand. Front: the per-step constant
    /// (`embed_image_indicator`) + the small embedders / final layer (some pay a
    /// per-acquire Q8 transcode, so residency also saves prep dispatches). Then
    /// the 34 main blocks role-major / block-minor, largest roles first: every
    /// block keeps the same role subset resident and streams the rest, keeping
    /// per-block stream demand flat across the step.
    pub fn pin_priority(&self) -> Vec<thinfer_core::residency::WeightHandle> {
        use thinfer_core::residency::WeightHandle;
        let h = &self.handles;
        let mut out: Vec<WeightHandle> = Vec::new();
        let lb = |out: &mut Vec<WeightHandle>, l: &LinearBiasHandles| {
            out.push(l.weight);
            out.push(l.bias);
        };
        out.push(h.embed_image_indicator);
        lb(&mut out, &h.t_embedder.mlp_in);
        lb(&mut out, &h.t_embedder.mlp_out);
        lb(&mut out, &h.t_embedder.adaln_proj);
        out.push(h.llm_cond.norm_weight);
        lb(&mut out, &h.llm_cond.linear);
        lb(&mut out, &h.input_proj);
        lb(&mut out, &h.final_layer.linear);
        lb(&mut out, &h.final_layer.adaln);
        // Main layers: one role across all 34 blocks, then the next role
        // (largest matmuls first). qkv + the three FFN linears are the big Q8_0
        // weights; adaln/norms are tiny.
        type RoleFn = fn(&BlockHandles) -> Option<thinfer_core::residency::WeightHandle>;
        let roles: [RoleFn; 6] = [
            |b| Some(b.ffn_w1),
            |b| Some(b.ffn_w3),
            |b| Some(b.attn_qkv),
            |b| Some(b.ffn_w2),
            |b| Some(b.attn_to_out),
            |b| b.adaln.as_ref().map(|a| a.weight),
        ];
        for role in roles {
            out.extend(h.layers.iter().filter_map(role));
        }
        let small_roles: [RoleFn; 7] = [
            |b| b.adaln.as_ref().map(|a| a.bias),
            |b| Some(b.attention_norm1),
            |b| Some(b.attention_norm2),
            |b| Some(b.ffn_norm1),
            |b| Some(b.ffn_norm2),
            |b| Some(b.attn_norm_q),
            |b| Some(b.attn_norm_k),
        ];
        for role in small_roles {
            out.extend(h.layers.iter().filter_map(role));
        }
        out
    }

    /// Conservative VRAM reserve for the denoise-phase workspace, used to size
    /// the pin plan's budget headroom. All 34 blocks share one pipeline set and
    /// one (even-padded) seq, so the per-block peak is uniform; take the max of
    /// each block's four phase peaks (the packer's single-scope fast path) plus
    /// the cross-submit persistents (ping-pong `cur`/`nxt`, freqs, mask, adaln),
    /// approximated as a handful of `seq * dim` activation buffers. Err high:
    /// overestimating only shrinks the pin set, underestimating starves the
    /// packer.
    pub fn workspace_reserve_estimate(&self, main_pipelines: &BlockPipelines) -> u64 {
        let block_peak = self
            .layers
            .iter()
            .map(|blk| blk.phase_peaks(main_pipelines).iter().sum::<u64>())
            .max()
            .unwrap_or(0);
        let persist = self.layers.first().map_or(0, |blk| {
            let cfg = &blk.cfg;
            8 * main_pipelines.act_bytes((cfg.rows() * cfg.dim) as u32)
        });
        block_peak + persist
    }

    /// Run one velocity forward. Returns `[num_image, IN_CHANNELS]` row-major
    /// f32 velocity at the image positions.
    pub async fn forward<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        main_pipelines: &BlockPipelines,
        dense_pipelines: &BlockPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        inputs: &DitInputs<'_>,
    ) -> Result<Vec<f32>, DitError<S::Error>> {
        self.forward_with_taps(
            backend,
            main_pipelines,
            dense_pipelines,
            residency,
            scratch,
            inputs,
            DitTaps::default(),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn forward_with_taps<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        main_pipelines: &BlockPipelines,
        dense_pipelines: &BlockPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        inputs: &DitInputs<'_>,
        mut taps: DitTaps<'_>,
    ) -> Result<Vec<f32>, DitError<S::Error>> {
        let dim = config::DIM as u32;
        let num_text = inputs.num_text;
        let num_image = inputs.grid.num_image_tokens();
        let seq = num_text + num_image;
        // Even-pad for the F16 mask word-packing (see `assemble`). Pad rows are
        // appended after the image region and masked as keys.
        let seq_pad = seq.next_multiple_of(2);
        let n_pad = seq_pad - seq;
        assert_eq!(
            self.layers[0].cfg.seq, seq_pad,
            "assembled seq must match padded inputs"
        );
        assert_eq!(
            inputs.llm_features.len(),
            num_text * enc_config::LLM_FEATURES_DIM,
            "llm_features shape"
        );
        assert_eq!(
            inputs.noise.len(),
            num_image * config::IN_CHANNELS,
            "noise shape"
        );

        // --- adaln_input (shared modulation conditioning) ---
        let adaln = scratch.alloc(dense_pipelines.act_bytes(config::ADALN_DIM as u32))?;
        {
            let views = self.handles.t_embedder.acquire(residency, backend).await?;
            let bufs = views.bufs();
            let out_ref = adaln.as_buf_ref();
            let scope = scratch.batch();
            self.t_embedder.forward(
                &scope,
                dense_pipelines,
                inputs.timestep,
                scope.import(&out_ref),
                &bufs,
            )?;
            scope.submit_void().await?;
        }
        if let Some(sink) = taps.adaln_input.as_deref_mut() {
            *sink = read_to_f32(backend, dense_pipelines, &adaln, config::ADALN_DIM as u32).await?;
        }

        // --- assemble h = [text: llm_cond | image: input_proj] + indicator ---
        // Each region's embedder writes to a temp; bcast_add then folds the
        // indicator row in while writing to h (distinct in/out buffers: an
        // in-place bcast aliases STORAGE_READ_ONLY + READ_WRITE on one buffer).
        let h = scratch.alloc(dense_pipelines.act_bytes(seq_pad as u32 * dim))?;
        let row_bytes = dense_pipelines.act_bytes(dim);
        let h_base = h.as_buf_ref().offset;
        // Zero the trailing pad rows: they are masked as keys, but their q/k/v
        // must be finite (a -inf mask added to a NaN score still yields NaN), and
        // their value rows must not feed garbage anywhere. Zero h -> qkv = bias.
        if n_pad > 0 {
            let zeros = vec![0u8; n_pad * row_bytes as usize];
            backend.write_buffer(h.id(), seq as u64 * row_bytes, &zeros)?;
        }
        let ind = residency
            .acquire(self.handles.embed_image_indicator, backend)
            .await?;
        let ind_buf = ind.buf();
        let row0 = BufRef::view(ind_buf.id, ind_buf.offset, row_bytes);
        let row1 = BufRef::view(ind_buf.id, ind_buf.offset + row_bytes, row_bytes);

        // text rows: llm_cond_proj(llm_cond_norm(features)) + indicator[0]
        if num_text > 0 {
            let feat_bytes = act_upload_bytes(dense_pipelines.act_dtype, inputs.llm_features);
            let feat = scratch.alloc(feat_bytes.len() as u64)?;
            backend.write_buffer(feat.id(), 0, &feat_bytes)?;
            let views = self.handles.llm_cond.acquire(residency, backend).await?;
            let bufs = views.bufs();
            let feat_ref = feat.as_buf_ref();
            let dst = BufRef::view(h.id(), h_base, num_text as u64 * row_bytes);
            let scope = scratch.batch();
            let tmp = scope.alloc(num_text as u64 * row_bytes)?;
            self.llm_cond.forward(
                &scope,
                dense_pipelines,
                scope.import(&feat_ref),
                num_text as u32,
                tmp,
                &bufs,
            )?;
            add_bcast_row(
                &scope,
                dense_pipelines,
                tmp,
                row0,
                scope.import(&dst),
                num_text as u32,
                dim,
            )?;
            scope.submit_void().await?;
        }

        // image rows: input_proj(noise) + indicator[1]
        {
            let noise_bytes = act_upload_bytes(dense_pipelines.act_dtype, inputs.noise);
            let noise = scratch.alloc(noise_bytes.len() as u64)?;
            backend.write_buffer(noise.id(), 0, &noise_bytes)?;
            let views = self.handles.input_proj.acquire(residency, backend).await?;
            let bufs: LinearBiasBufs = views.bufs();
            let noise_ref = noise.as_buf_ref();
            let dst = BufRef::view(
                h.id(),
                h_base + num_text as u64 * row_bytes,
                num_image as u64 * row_bytes,
            );
            let scope = scratch.batch();
            let tmp = scope.alloc(num_image as u64 * row_bytes)?;
            linear_bias(
                &scope,
                dense_pipelines,
                scope.import(&noise_ref),
                &bufs,
                num_image as u32,
                config::IN_CHANNELS as u32,
                dim,
                tmp,
            )?;
            add_bcast_row(
                &scope,
                dense_pipelines,
                tmp,
                row1,
                scope.import(&dst),
                num_image as u32,
                dim,
            )?;
            scope.submit_void().await?;
        }
        drop(ind);
        if let Some(sink) = taps.h_assembled.as_deref_mut() {
            *sink = read_to_f32(backend, dense_pipelines, &h, seq_pad as u32 * dim).await?;
        }

        // --- rope freqs (3-axis MRoPE) + tail-masked mask ---
        // Pad positions get (0,0,0): they are masked as keys and discarded as
        // outputs, so their rope angle is irrelevant (just finite).
        let mut pos_ids = build_position_ids(num_text, inputs.grid);
        pos_ids.extend(std::iter::repeat_n(0_i64, n_pad * 3));
        let freqs_f32 = mrope::build_freqs_dit(&pos_ids);
        let freqs_bytes = freqs_upload_bytes(main_pipelines.act_dtype, &freqs_f32);
        let freqs = scratch.alloc(freqs_bytes.len() as u64)?;
        backend.write_buffer(freqs.id(), 0, &freqs_bytes)?;
        let mask_bytes = attn_mask_pad_tail_bytes_act(seq_pad, n_pad, main_pipelines.act_dtype);
        let mask = scratch.alloc(mask_bytes.len() as u64)?;
        backend.write_buffer(mask.id(), 0, &mask_bytes)?;

        // --- 34 transformer blocks (ping-pong) ---
        let mut cur = h;
        let last = self.layers.len() - 1;
        for (idx, blk) in self.layers.iter().enumerate() {
            let nxt = scratch.alloc(main_pipelines.act_bytes(seq_pad as u32 * dim))?;
            let views = self.handles.layers[idx].acquire(residency, backend).await?;
            let bufs = views.bufs();
            let cur_ref = cur.as_buf_ref();
            let nxt_ref = nxt.as_buf_ref();
            let freqs_ref = freqs.as_buf_ref();
            let mask_ref = mask.as_buf_ref();
            let adaln_ref = adaln.as_buf_ref();
            {
                let scope = scratch.batch();
                let x_act = ActBuf::dense(scope.import_copy(cur_ref));
                let y_act = ActBuf::dense(scope.import_copy(nxt_ref));
                blk.forward(
                    &scope,
                    main_pipelines,
                    x_act,
                    scope.import_copy(freqs_ref),
                    scope.import_copy(mask_ref),
                    Some(scope.import_copy(adaln_ref)),
                    y_act,
                    &bufs,
                )?;
                scope.submit_void().await?;
            }
            cur = nxt;
            if idx == 0
                && let Some(sink) = taps.block0_out.as_deref_mut()
            {
                *sink = read_to_f32(backend, main_pipelines, &cur, seq_pad as u32 * dim).await?;
            }
            if idx == last
                && let Some(sink) = taps.block_last_out.as_deref_mut()
            {
                *sink = read_to_f32(backend, main_pipelines, &cur, seq_pad as u32 * dim).await?;
            }
        }

        // --- final layer -> velocity [seq_pad, IN_CHANNELS], slice image tail ---
        let oc = config::IN_CHANNELS as u32;
        let out = scratch.alloc(dense_pipelines.act_bytes(seq_pad as u32 * oc))?;
        {
            let views = self.handles.final_layer.acquire(residency, backend).await?;
            let bufs = views.bufs();
            let x_ref = cur.as_buf_ref();
            let c_ref = adaln.as_buf_ref();
            let out_ref = out.as_buf_ref();
            let scope = scratch.batch();
            self.final_layer.forward(
                &scope,
                dense_pipelines,
                scope.import(&x_ref),
                scope.import(&c_ref),
                seq_pad as u32,
                scope.import(&out_ref),
                &bufs,
            )?;
            scope.submit_void().await?;
        }
        let full = read_to_f32(backend, dense_pipelines, &out, seq_pad as u32 * oc).await?;
        // Velocity = the image region [num_text .. num_text+num_image], ahead of
        // the trailing pad rows.
        let img_start = num_text * config::IN_CHANNELS;
        let img_end = (num_text + num_image) * config::IN_CHANNELS;
        Ok(full[img_start..img_end].to_vec())
    }
}

/// `out = input + row[i % dim]` over `n_rows` rows of `dim` channels (`input`
/// and `out` must be distinct buffers; `row` is read at the weight dtype).
fn add_bcast_row<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    input: thinfer_core::workspace::BatchBuf<'wsp>,
    row: BufRef,
    out: thinfer_core::workspace::BatchBuf<'wsp>,
    n_rows: u32,
    dim: u32,
) -> Result<(), WgpuError> {
    use thinfer_core::ops::BcastAddF32;
    let mut u = [0u8; 16];
    u[0..4].copy_from_slice(&dim.to_le_bytes());
    let uni = scope.write_uniform(&u)?;
    let vec = scope.import_copy(row);
    scope.bcast_add::<BcastAddF32>(&pipelines.bcast_add, input, vec, uni, out, n_rows * dim)
}

async fn read_to_f32(
    backend: &WgpuBackend,
    pipelines: &BlockPipelines,
    buf: &WsBuf<WgpuBackend>,
    n_elems: u32,
) -> Result<Vec<f32>, WgpuError> {
    let bytes = backend
        .read_buffer(buf.id(), 0, pipelines.act_bytes(n_elems))
        .await?;
    Ok(act_readback_to_f32(
        pipelines.act_dtype,
        &bytes,
        n_elems as usize,
    ))
}
