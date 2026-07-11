//! Krea 2 text encoder: Qwen3-VL-4B run text-only, tapping 12 hidden-state
//! layers for the DiT's `txtfusion`. The text backbone is structurally identical
//! to the Qwen3-4B encoder already ported in [`crate::z_image::text_encoder`]
//! (36 layers, hidden 2560, GQA 32/8, head_dim 128, SwiGLU 9728, per-head Q/K
//! RMSNorm, half-rot RoPE) -- so this module REUSES its `Qwen3Block`, weight
//! handles, embed lookup, and GGUF rename map. The only Krea differences:
//!
//! - RoPE theta is 5e6 (Qwen3-VL), not Qwen3's 1e6. For a text-only prompt the
//!   Qwen3-VL mrope collapses to 1-axis token-position RoPE, so only theta
//!   changes.
//! - The DiT consumes 12 tapped hidden states, not the single `hidden_states[-2]`
//!   that Z-Image reads. The taps are `hidden_states[{2,5,8,...,35}]` = the
//!   residual stream after that many decoder layers (index 0 = embeddings). The
//!   max is 35, so we run 35 layers (exactly what `register_qwen3_handles`
//!   registers) and snapshot the residual after each selected layer count.
//!
//! Output: `[txt_tok, TEXT_LAYERS=12, TEXT_DIM=2560]` (token-major, then layer in
//! `ENCODER_SELECT_LAYERS` order, then dim), ready for `KreaDit::prepare_txt`.

use thinfer_core::backend::{Backend, WgpuBackend, WgpuError};
use thinfer_core::residency::{ResidencyError, WeightResidency};
use thinfer_core::trace;
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::Workspace;

use crate::common::block::BlockPipelines;
use crate::common::rope_embedder::RopeEmbedder;
use crate::common::seq;
use crate::krea::config as krea_config;
use crate::z_image::text_encoder::{
    Qwen3Block, Qwen3BlockShape, Qwen3ForwardError, Qwen3Handles, config as q3, embed_lookup,
};

/// Krea RoPE base for the Qwen3-VL text backbone.
pub const ENCODER_ROPE_THETA: f32 = 5_000_000.0;

pub struct KreaTextEncoder {
    rope: RopeEmbedder,
}

impl KreaTextEncoder {
    pub fn new(max_seq: usize) -> Self {
        let seq_len = max_seq.max(1);
        Self {
            rope: RopeEmbedder::new(ENCODER_ROPE_THETA, [q3::HEAD_DIM, 0, 0], [seq_len, 1, 1]),
        }
    }

    /// Encode `token_ids` and return the 12 tapped hidden states
    /// `[txt_tok * TEXT_LAYERS * TEXT_DIM]`.
    pub async fn forward<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipelines: &BlockPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        handles: &Qwen3Handles,
        source: &S,
        token_ids: &[u32],
    ) -> Result<Vec<f32>, Qwen3ForwardError<S::Error>> {
        debug_assert_eq!(handles.layers.len(), q3::HIDDEN_STATES_LAYER);
        let seq = token_ids.len();
        assert!(seq > 0, "KreaTextEncoder::forward: empty token list");
        // Pad to an even row count (F16 mask/act layouts pack 2 elems/word); the
        // pad row repeats the last token, is causally invisible, and is sliced
        // off after readback.
        let mut ids = token_ids.to_vec();
        if !ids.len().is_multiple_of(2) {
            ids.push(*ids.last().expect("non-empty"));
        }
        let seq_pad = ids.len();
        assert!(
            seq_pad <= self.rope.axes_lens[0],
            "prompt length {seq_pad} exceeds encoder max_seq {}",
            self.rope.axes_lens[0]
        );

        let embeds = embed_lookup(source, &ids)
            .await
            .map_err(Qwen3ForwardError::Embed)?;
        let diag = tracing::enabled!(target: "thinfer::diag", tracing::Level::DEBUG);
        if diag {
            let bad = embeds.iter().filter(|v| !v.is_finite()).count();
            tracing::event!(
                target: "thinfer::diag", tracing::Level::DEBUG,
                len = embeds.len(), nonfinite = bad,
                "krea encoder: embed_lookup output"
            );
        }

        let dim = q3::HIDDEN as u32;
        let rows = seq_pad as u32;
        let act_bytes = pipelines.act_bytes(rows * dim);
        let shape = Qwen3BlockShape::default_from_config(seq_pad);
        let block = Qwen3Block::new(shape);

        // which decoder-layer-count snapshots we keep, and their position in the
        // output layer axis.
        let selects = krea_config::ENCODER_SELECT_LAYERS;
        let n_taps = selects.len();
        // one GPU snapshot buffer per tap.
        let mut tap_bufs = Vec::with_capacity(n_taps);
        for _ in 0..n_taps {
            tap_bufs.push(scratch.alloc(act_bytes)?);
        }

        let x_buf = scratch.alloc(act_bytes)?;
        backend.write_buffer(
            x_buf.id,
            0,
            &seq::act_upload_bytes(pipelines.act_dtype, &embeds),
        )?;

        // rope freqs (theta 5e6) + causal mask.
        let mut pos_ids = vec![0_i32; seq_pad * 3];
        for (i, p) in pos_ids.chunks_mut(3).enumerate() {
            p[0] = i as i32;
        }
        let freqs_bytes = seq::freqs_upload_bytes(pipelines.act_dtype, &self.rope.lookup(&pos_ids));
        let freqs_buf = scratch.alloc(freqs_bytes.len() as u64)?;
        backend.write_buffer(freqs_buf.id, 0, &freqs_bytes)?;
        let mask_bytes = seq::causal_mask_bytes_act(seq_pad, pipelines.act_dtype);
        let mask_buf = scratch.alloc(mask_bytes.len() as u64)?;
        backend.write_buffer(mask_buf.id, 0, &mask_bytes)?;

        // run 35 layers, snapshotting the residual after each selected count.
        let mut cur = x_buf;
        let mut first_nan_reported = false;
        let mut pending = if handles.layers.is_empty() {
            None
        } else {
            Some(handles.layers[0].acquire(residency, backend).await?)
        };
        let freqs_ref = freqs_buf.as_buf_ref();
        let mask_ref = mask_buf.as_buf_ref();
        for (idx, _h) in handles.layers.iter().enumerate() {
            let _g = trace::scope!(format!("krea_enc.layer.{idx}")).entered();
            let views = pending.take().expect("pending acquire missing");
            let bufs = views.bufs();
            let cur_ref = cur.as_buf_ref();
            let nxt = scratch.alloc(act_bytes)?;
            let nxt_ref = nxt.as_buf_ref();
            {
                let scope = scratch.batch();
                let cur_h = scope.import(&cur_ref);
                let freqs_h = scope.import(&freqs_ref);
                let mask_h = scope.import(&mask_ref);
                let nxt_h = scope.import(&nxt_ref);
                block.forward(&scope, pipelines, cur_h, freqs_h, mask_h, nxt_h, &bufs)?;
                // hidden_states[idx+1] = residual after this layer.
                if let Some(t) = selects.iter().position(|&s| s == idx + 1) {
                    let tb_ref = tap_bufs[t].as_buf_ref();
                    let dst = scope.import(&tb_ref);
                    let src = scope.import(&nxt_ref);
                    scope.copy_buffer_to_buffer(src, 0, dst, 0, act_bytes)?;
                }
                let next_acquire = async {
                    match handles.layers.get(idx + 1) {
                        Some(h) => Ok::<_, ResidencyError<S::Error, WgpuError>>(Some(
                            h.acquire(residency, backend).await?,
                        )),
                        None => Ok(None),
                    }
                };
                let (submit_res, next_res) = futures::join!(scope.submit_void(), next_acquire);
                submit_res?;
                pending = next_res?;
            }
            cur = nxt;
            if diag && !first_nan_reported {
                let bytes = backend.read_buffer(cur.id(), 0, act_bytes).await?;
                let hs =
                    seq::act_readback_to_f32(pipelines.act_dtype, &bytes, seq_pad * dim as usize);
                let bad = hs.iter().filter(|v| !v.is_finite()).count();
                if bad > 0 {
                    first_nan_reported = true;
                    tracing::event!(
                        target: "thinfer::diag", tracing::Level::DEBUG,
                        layer = idx, nonfinite = bad, total = hs.len(),
                        "krea encoder: FIRST non-finite layer output"
                    );
                }
            }
        }

        // readback the 12 taps and interleave per token: out[(tok*12 + l)*dim + d].
        let td = q3::HIDDEN;
        let txt_tok = seq; // real tokens (drop the parity pad row)
        let mut out = vec![0.0_f32; txt_tok * n_taps * td];
        for (l, tb) in tap_bufs.iter().enumerate() {
            let bytes = backend.read_buffer(tb.id(), 0, act_bytes).await?;
            let hs = seq::act_readback_to_f32(pipelines.act_dtype, &bytes, seq_pad * td);
            for tok in 0..txt_tok {
                let src = &hs[tok * td..tok * td + td];
                let dst = &mut out[(tok * n_taps + l) * td..(tok * n_taps + l) * td + td];
                dst.copy_from_slice(src);
            }
        }
        Ok(out)
    }
}
