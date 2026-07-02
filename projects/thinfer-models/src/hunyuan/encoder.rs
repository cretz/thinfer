//! HunyuanVideo 1.5 text conditioning. The encoder is the Qwen2.5-VL-7B text
//! stack REUSED verbatim from `qwen_image` (same arch, same cached GGUF, GREEN
//! via that port's `encoder_parity`); this module adds only the Hunyuan-specific
//! prompt template + the exact hidden-state the DiT was distilled against.
//!
//! Upstream (`hyvideo` `TextEncoder`, `hunyuan_video_pipeline.py`): the LLM is
//! prompted with a fixed video chat template, tokenized via `apply_chat_template`
//! (+ generation prompt), the system/user-header tokens are cropped off the
//! hidden states (`crop_start`), and the DiT conditions on
//! `hidden_states[-3]` with NO final norm (`hidden_state_skip_layer=2`,
//! `apply_final_norm=False`). The crop itself is done host-side by the caller
//! (it owns the tokenizer); this module owns the template strings + the
//! skip-layer extraction.
//!
//! NB: the encoder -> DiT seam has no component parity gate (the DiT/VAE gates
//! feed a pinned random text tensor); fidelity rests on reproducing the template
//! + skip-layer exactly. A serve eyeball is the check.

use thinfer_core::backend::{WgpuBackend, WgpuError};
use thinfer_core::residency::WeightResidency;
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::Workspace;

use crate::common::block::BlockPipelines;
use crate::qwen_image::text_encoder::{
    EncoderHandles, ForwardError, TextEncoder, config as enc_config, encoder_block_cfgs,
    register_handles,
};
use crate::z_image::text_encoder::LoadError;

/// The fixed video system prompt (verbatim from upstream
/// `PROMPT_TEMPLATE_ENCODE_VIDEO_JSON`). The 9-space runs between the numbered
/// items are an upstream Python line-continuation artifact and ARE load-bearing:
/// the system block is cropped from the output, but the kept prompt tokens
/// attend back to it under causal attention, so the exact tokenization matters.
pub const VIDEO_SYSTEM_PROMPT: &str = "You are a helpful assistant. Describe the video by detailing the following aspects:         1. The main content and theme of the video.         2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.         3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.         4. background environment, light, style and atmosphere.         5. camera angles, movements, and transitions used in the video.";

/// The user-turn marker upstream crops on (`calculate_crop_start`): every token
/// through this marker (the system block + the user header) is dropped from the
/// encoder hidden states. The caller tokenizes this and finds the subsequence.
pub const USER_MARKER: &str = "<|im_start|>user\n";

/// `hidden_state_skip_layer=2` (pipeline config) -> the DiT conditions on
/// `hidden_states[-3]`: the residual AFTER 26 of the 28 decoder layers, with NO
/// final norm (`apply_final_norm=False`). In `TextEncoder::forward`'s
/// `layer_outputs` (one entry per layer, post-residual) that is index
/// `N_LAYERS - 1 - 2 = 25`.
pub const HIDDEN_STATE_SKIP_LAYER: usize = 2;
pub const SKIP_LAYER_OUTPUT_INDEX: usize = enc_config::N_LAYERS - 1 - HIDDEN_STATE_SKIP_LAYER;

/// Build the Qwen2.5 ChatML string for a video prompt: system(video template) +
/// user(prompt) + the assistant generation prompt. Matches
/// `apply_chat_template([{system}, {user}], add_generation_prompt=True)` for the
/// string-content messages this path uses (verified against the cached
/// Qwen2.5-VL `chat_template`).
pub fn build_chat_prompt(user_prompt: &str) -> String {
    format!(
        "<|im_start|>system\n{VIDEO_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
}

/// Failure building the Hunyuan text encoder: weight registration or pipeline
/// compilation.
#[derive(Debug)]
pub enum EncoderLoadError {
    Register(LoadError),
    Compile(WgpuError),
}

impl From<LoadError> for EncoderLoadError {
    fn from(e: LoadError) -> Self {
        Self::Register(e)
    }
}

impl From<WgpuError> for EncoderLoadError {
    fn from(e: WgpuError) -> Self {
        Self::Compile(e)
    }
}

/// The Qwen2.5-VL text encoder configured for HunyuanVideo 1.5: same weights +
/// block config as `qwen_image`, but the DiT consumes `hidden_states[-3]`
/// (skip-layer 2, no final norm) instead of the post-norm last layer.
pub struct HunyuanTextEncoder {
    enc: TextEncoder,
    pipelines: BlockPipelines,
    handles: EncoderHandles,
}

impl HunyuanTextEncoder {
    /// Register the encoder weights + compile its pipelines. `max_seq` sizes the
    /// rope table (the cropped+even-padded prompt length).
    pub async fn load<S: WeightSource>(
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        max_seq: usize,
    ) -> Result<Self, EncoderLoadError> {
        let handles = register_handles(residency, None)?;
        let pipelines = BlockPipelines::compile(backend, &encoder_block_cfgs()).await?;
        Ok(Self {
            enc: TextEncoder::new(max_seq),
            pipelines,
            handles,
        })
    }

    /// Encode the (already cropped) prompt token ids into the DiT conditioning
    /// `[seq, 3584]` f32 = `hidden_states[-3]` (no final norm).
    pub async fn encode<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        token_ids: &[u32],
    ) -> Result<Vec<f32>, ForwardError<S::Error>> {
        let mut out = self
            .enc
            .forward(
                backend,
                &self.pipelines,
                residency,
                workspace,
                &self.handles,
                residency.source(),
                token_ids,
                true, // capture per-layer outputs to pull hidden_states[-3]
            )
            .await?;
        // layer_outputs[idx] is [seq_pad, HIDDEN] (even-pad rows trail); take the
        // skip-layer residual and slice to the real cropped length.
        let mut hidden = out.layer_outputs.swap_remove(SKIP_LAYER_OUTPUT_INDEX);
        hidden.truncate(out.seq * enc_config::HIDDEN);
        Ok(hidden)
    }
}
