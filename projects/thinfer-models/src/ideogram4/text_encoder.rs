//! Qwen3-VL-8B language-model text encoder for Ideogram-4.
//!
//! Ideogram-4 conditions on `text_encoder.language_model` (the LM tower of
//! Qwen3-VL-8B-Instruct, no vision) run over the prompt tokens, capturing the
//! hidden state AFTER each of 13 tapped layers and concatenating them per token
//! into a `(seq, 4096*13 = 53248)` feature tensor (the DiT then RMSNorm's +
//! projects this). See `pipeline_ideogram4.py::_encode_text` /
//! `_get_qwen3_vl_embeddings`.
//!
//! The per-layer compute is identical to Z-Image's Qwen3 (GQA 32Q/8KV,
//! head_dim 128, per-head QK-RMSNorm, half-rot RoPE, SwiGLU), so this reuses
//! `z_image::text_encoder`'s `Qwen3Block` and weight/handle plumbing verbatim.
//! Deltas vs the 4B encoder: HIDDEN 4096, FFN 12288, rope_theta 5e6, ALL 36
//! layers run (the 4B path stops at `hidden_states[-2]`; here layer 35 IS a
//! tap), and the multi-layer stack instead of a single `[-2]` readback.
//!
//! For text-only positions Qwen3-VL's MRoPE collapses to standard 1D RoPE (all
//! three position axes are equal), so the single-axis RoPE table suffices. The
//! engine runs the encoder over the prompt tokens only: in the upstream packed
//! sequence the image-token positions are masked out of attention and their
//! features zeroed, so text-position features are independent of them.

use thinfer_core::backend::{Backend, WgpuBackend, WgpuError};
use thinfer_core::residency::{ResidencyError, WeightResidency};
use thinfer_core::trace::{self, PHASE};
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::Workspace;
use tracing::Instrument;

use crate::common::block::BlockPipelines;
use crate::common::rope_embedder::RopeEmbedder;
use crate::common::seq;
use crate::z_image::text_encoder::{
    EmbedLookupError, LoadError, Qwen3BlockShape, Qwen3BlockViews, Qwen3Handles, Qwen3Weights,
    embed_lookup_hidden, register_one,
};

use thinfer_core::residency::TransposePolicy;

/// Audited against `Qwen/Qwen3-VL-8B-Instruct/config.json` (`text_config`).
pub mod config {
    pub const HIDDEN: usize = 4096;
    pub const N_LAYERS: usize = 36;
    pub const N_HEADS: usize = 32;
    pub const N_KV_HEADS: usize = 8;
    pub const HEAD_DIM: usize = 128;
    pub const FFN_HIDDEN: usize = 12288;
    pub const VOCAB: usize = 151936;
    pub const RMS_NORM_EPS: f32 = 1e-6;
    pub const ROPE_THETA: f32 = 5_000_000.0;

    /// Layers whose post-residual output is captured and stacked
    /// (`QWEN3_VL_ACTIVATION_LAYERS`). Layer index = decoder-layer output (the
    /// embedding is not tapped). Layer 35 (the last) IS included, so all 36
    /// layers must run.
    pub const TAP_LAYERS: [usize; 13] = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35];
    pub const N_TAPS: usize = TAP_LAYERS.len();
    /// Stacked feature width fed to the DiT (`llm_features_dim`).
    pub const LLM_FEATURES_DIM: usize = HIDDEN * N_TAPS; // 53248
}

/// Register all 36 Qwen3-VL-8B LM layers. Mirrors
/// `z_image::text_encoder::register_qwen3_handles` but runs the FULL stack
/// (layer 35 is a tap) and uses the 8B layer count. The embed table stays
/// unregistered (gathered per-row from disk). `mlp_down` is never transcoded
/// (Qwen3 massive-activation precision; same rationale as the 4B encoder).
pub fn register_handles<S: WeightSource>(
    residency: &WeightResidency<S>,
    transcode: Option<thinfer_core::quant::QuantKind>,
) -> Result<Qwen3Handles, LoadError> {
    let weights = Qwen3Weights::new();
    debug_assert_eq!(weights.layers.len(), config::N_LAYERS);
    let mut layers = Vec::with_capacity(config::N_LAYERS);
    for b in weights.layers.iter() {
        layers.push(crate::z_image::text_encoder::Qwen3BlockHandles {
            input_layernorm: register_one(
                residency,
                &b.input_layernorm,
                TransposePolicy::None,
                None,
            )?,
            post_attention_layernorm: register_one(
                residency,
                &b.post_attention_layernorm,
                TransposePolicy::None,
                None,
            )?,
            q_proj: register_one(residency, &b.q_proj, TransposePolicy::Linear2D, transcode)?,
            k_proj: register_one(residency, &b.k_proj, TransposePolicy::Linear2D, transcode)?,
            v_proj: register_one(residency, &b.v_proj, TransposePolicy::Linear2D, transcode)?,
            o_proj: register_one(residency, &b.o_proj, TransposePolicy::Linear2D, transcode)?,
            q_norm: register_one(residency, &b.q_norm, TransposePolicy::None, None)?,
            k_norm: register_one(residency, &b.k_norm, TransposePolicy::None, None)?,
            mlp_gate: register_one(residency, &b.mlp_gate, TransposePolicy::Linear2D, transcode)?,
            mlp_up: register_one(residency, &b.mlp_up, TransposePolicy::Linear2D, transcode)?,
            mlp_down: register_one(residency, &b.mlp_down, TransposePolicy::Linear2D, None)?,
        });
    }
    Ok(Qwen3Handles { layers })
}

fn block_shape(seq: usize) -> Qwen3BlockShape {
    Qwen3BlockShape {
        dim: config::HIDDEN,
        n_heads: config::N_HEADS,
        n_kv_heads: config::N_KV_HEADS,
        head_dim: config::HEAD_DIM,
        ffn_hidden: config::FFN_HIDDEN,
        seq,
        norm_eps: config::RMS_NORM_EPS,
    }
}

/// Stacked multi-layer encoder features.
#[derive(Clone, Debug)]
pub struct EncoderOutput {
    /// `[seq, LLM_FEATURES_DIM]` row-major fp32. Feature index for token `t`,
    /// hidden channel `h`, tap `j` is `t*LLM_FEATURES_DIM + h*N_TAPS + j`
    /// (matching upstream `permute(B,L,H,n_taps).reshape(B,L,H*n_taps)`).
    pub features: Vec<f32>,
    /// The 13 raw tapped layer outputs, each `[seq, HIDDEN]` row-major fp32,
    /// in `TAP_LAYERS` order. Empty unless `encode` was asked to keep them
    /// (parity localization: which tap diverges). The pad row is dropped.
    pub taps: Vec<Vec<f32>>,
    pub seq: usize,
}

#[derive(Debug)]
pub enum EncodeError<SE: core::fmt::Debug> {
    Embed(EmbedLookupError),
    Wgpu(WgpuError),
    Residency(ResidencyError<SE, WgpuError>),
}

impl<SE: core::fmt::Debug> From<WgpuError> for EncodeError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for EncodeError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

pub struct Qwen3VlEncoder {
    rope: RopeEmbedder,
}

impl Qwen3VlEncoder {
    pub fn new(max_seq: usize) -> Self {
        let seq_len = max_seq.max(1);
        Self {
            // 1-axis RoPE: text-only positions make all 3 MRoPE axes equal.
            rope: RopeEmbedder::new(
                config::ROPE_THETA,
                [config::HEAD_DIM, 0, 0],
                [seq_len, 1, 1],
            ),
        }
    }

    /// Run the 36-layer LM over `token_ids`, capture the 13 tapped layer
    /// outputs, and stack them into `[seq, LLM_FEATURES_DIM]`.
    pub async fn encode<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipelines: &BlockPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        handles: &Qwen3Handles,
        source: &S,
        token_ids: &[u32],
        keep_taps: bool,
    ) -> Result<EncoderOutput, EncodeError<S::Error>> {
        assert_eq!(handles.layers.len(), config::N_LAYERS);
        let seq = token_ids.len();
        assert!(seq > 0, "Qwen3VlEncoder::encode: empty token list");

        // Even-pad: F16 act/mask layouts pack two elems per u32 word (mask
        // word offsets require even s_k). The pad repeats the last token and
        // is causally invisible; its output row is sliced off after readback.
        let mut ids = token_ids.to_vec();
        if !ids.len().is_multiple_of(2) {
            ids.push(*ids.last().expect("non-empty"));
        }
        let seq_pad = ids.len();
        assert!(
            seq_pad <= self.rope.axes_lens[0],
            "prompt length {seq_pad} exceeds encoder rope max_seq {}",
            self.rope.axes_lens[0]
        );

        let embeds = embed_lookup_hidden(source, &ids, config::HIDDEN)
            .instrument(tracing::debug_span!(target: PHASE, "qwen3vl.embed_lookup", seq))
            .await
            .map_err(EncodeError::Embed)?;

        let block = crate::z_image::text_encoder::Qwen3Block::new(block_shape(seq_pad));
        let act_bytes = pipelines.act_bytes((seq_pad * config::HIDDEN) as u32);

        let x_buf = scratch.alloc(act_bytes)?;
        let embed_bytes = seq::act_upload_bytes(pipelines.act_dtype, &embeds);
        backend.write_buffer(x_buf.id, 0, &embed_bytes)?;

        // 1-axis rope freqs on positions 0..seq_pad.
        let mut pos_ids = vec![0_i32; seq_pad * 3];
        for (i, p) in pos_ids.chunks_exact_mut(3).enumerate() {
            p[0] = i as i32;
        }
        let freqs_bytes = seq::freqs_upload_bytes(pipelines.act_dtype, &self.rope.lookup(&pos_ids));
        let freqs_buf = scratch.alloc(freqs_bytes.len() as u64)?;
        backend.write_buffer(freqs_buf.id, 0, &freqs_bytes)?;

        let mask_bytes = seq::causal_mask_bytes_act(seq_pad, pipelines.act_dtype);
        let mask_buf = scratch.alloc(mask_bytes.len() as u64)?;
        backend.write_buffer(mask_buf.id, 0, &mask_bytes)?;

        // Per-tap readback (host f32, padded seq). 13 captures of [seq_pad, HIDDEN].
        let mut taps: Vec<Vec<f32>> = Vec::with_capacity(config::N_TAPS);
        let tap_set = config::TAP_LAYERS;

        let mut cur = x_buf;
        let mut pending: Option<Qwen3BlockViews<'_>> =
            Some(handles.layers[0].acquire(residency, backend).await?);
        let freqs_ref = freqs_buf.as_buf_ref();
        let mask_ref = mask_buf.as_buf_ref();
        for idx in 0..config::N_LAYERS {
            let _g = trace::scope!(format!("qwen3vl.layer.{idx}")).entered();
            let views = pending.take().expect("pending acquire missing");
            let bufs = views.bufs();
            let cur_ref = cur.as_buf_ref();
            let nxt = scratch.alloc(act_bytes)?;
            let nxt_ref = nxt.as_buf_ref();
            let scope = scratch.batch();
            let cur_h = scope.import(&cur_ref);
            let freqs_h = scope.import(&freqs_ref);
            let mask_h = scope.import(&mask_ref);
            let nxt_h = scope.import(&nxt_ref);
            block.forward(&scope, pipelines, cur_h, freqs_h, mask_h, nxt_h, &bufs)?;

            let next_idx = idx + 1;
            let next_acquire = async {
                match handles.layers.get(next_idx) {
                    Some(h) => Ok::<_, ResidencyError<S::Error, WgpuError>>(Some(
                        h.acquire(residency, backend)
                            .instrument(tracing::debug_span!(target: PHASE, "qwen3vl.acquire", idx = next_idx))
                            .await?,
                    )),
                    None => Ok(None),
                }
            };
            let submit_fut = scope
                .submit_void()
                .instrument(tracing::debug_span!(target: PHASE, "qwen3vl.submit", idx));
            let (submit_res, next_res) = futures::join!(submit_fut, next_acquire);
            submit_res?;
            pending = next_res?;

            if tap_set.contains(&idx) {
                let bytes = backend.read_buffer(nxt.id(), 0, act_bytes).await?;
                taps.push(seq::act_readback_to_f32(
                    pipelines.act_dtype,
                    &bytes,
                    seq_pad * config::HIDDEN,
                ));
            }

            drop(views);
            cur = nxt;
        }
        debug_assert_eq!(taps.len(), config::N_TAPS);

        // Stack: features[t, h*N_TAPS + j] = taps[j][t, h], dropping the pad row.
        let h = config::HIDDEN;
        let n = config::N_TAPS;
        let mut features = vec![0f32; seq * config::LLM_FEATURES_DIM];
        for (j, tap) in taps.iter().enumerate() {
            for t in 0..seq {
                let src = &tap[t * h..t * h + h];
                let base = t * config::LLM_FEATURES_DIM + j;
                for (hi, &val) in src.iter().enumerate() {
                    features[base + hi * n] = val;
                }
            }
        }
        // Localization taps: same data, sliced to the unpadded seq, one Vec
        // per tapped layer (no interleave). Only kept on request.
        let taps_out = if keep_taps {
            taps.iter().map(|tap| tap[..seq * h].to_vec()).collect()
        } else {
            Vec::new()
        };
        Ok(EncoderOutput {
            features,
            taps: taps_out,
            seq,
        })
    }
}
