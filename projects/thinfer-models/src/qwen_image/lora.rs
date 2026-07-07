//! Adapter (LoRA) key renames for Qwen-Image(-Edit)-Rapid. Unlike Krea (whose
//! base is an sd.cpp-named GGUF), the Qwen-Image base DiT tensor keys are already
//! 1:1 with the diffusers `QwenImageTransformer2DModel` module paths (see
//! `loader.rs`). So the ONLY thing a community adapter needs remapped is its
//! top-level PREFIX: PEFT/diffusers LoRAs ship `transformer.{X}`, while the
//! generic [`crate::common::lora`] fold discovers a site by matching
//! `diffusion_model.{X}.lora_{A,down}.weight` against the base tensor `{X}.weight`.
//! The module path `{X}` is identical on both sides, so a pure prefix swap
//! (`transformer.` -> `diffusion_model.`) is enough.
//!
//! ComfyUI-format Qwen-Image LoRAs already carry the `diffusion_model.` prefix and
//! fold with no rename at all (their keys pass through this map untouched). Kohya
//! `lora_unet_...` (underscore) naming is a different transform and is NOT handled
//! here (matching Krea). Both `.lora_A/.lora_B` (ai-toolkit) and
//! `.lora_down/.lora_up` (LightX2V) half-suffixes are covered so either convention
//! folds once the prefix is mapped.

use std::collections::HashMap;

use thinfer_core::weight::WeightId;

use crate::qwen_image::config;

/// The four LoRA half-suffixes across the two conventions the fold understands.
/// Renaming preserves the suffix; only the module prefix changes.
const LORA_SUFFIXES: [&str; 4] = [
    ".lora_A.weight",
    ".lora_B.weight",
    ".lora_down.weight",
    ".lora_up.weight",
];

/// Every linear submodule inside a Qwen-Image dual-stream block, relative to the
/// `transformer_blocks.N` prefix (see `loader::register_block`). The four norms
/// (`attn.norm_{q,k,added_q,added_k}`) are not linears, so no LoRA touches them.
const BLOCK_LINEARS: [&str; 14] = [
    "img_mod.1",
    "txt_mod.1",
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
];

/// Top-level (non-block) linears (see `loader::register_handles`). `txt_norm` is a
/// norm, so it is omitted.
const TOP_LINEARS: [&str; 6] = [
    "img_in",
    "txt_in",
    "time_text_embed.timestep_embedder.linear_1",
    "time_text_embed.timestep_embedder.linear_2",
    "norm_out.linear",
    "proj_out",
];

/// `transformer.{X}{suffix}` -> `diffusion_model.{X}{suffix}` for every base
/// linear site and both half-conventions. The module path `{X}` is preserved
/// (the base already uses diffusers naming); only the prefix is swapped. Pass to
/// `RenamedSource::with_passthrough` before `discover_specs`.
pub fn lora_key_renames() -> HashMap<WeightId, WeightId> {
    let mut out = HashMap::with_capacity(
        (config::N_LAYERS * BLOCK_LINEARS.len() + TOP_LINEARS.len()) * LORA_SUFFIXES.len(),
    );
    let mut insert_module = |module: &str| {
        for suffix in LORA_SUFFIXES {
            out.insert(
                WeightId(format!("transformer.{module}{suffix}")),
                WeightId(format!("diffusion_model.{module}{suffix}")),
            );
        }
    };
    for i in 0..config::N_LAYERS {
        for sub in BLOCK_LINEARS {
            insert_module(&format!("transformer_blocks.{i}.{sub}"));
        }
    }
    for module in TOP_LINEARS {
        insert_module(module);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_a_representative_site_of_each_kind() {
        let m = lora_key_renames();
        let check = |from: &str, to: &str| {
            assert_eq!(
                m.get(&WeightId(from.to_string())),
                Some(&WeightId(to.to_string())),
                "{from}"
            );
        };
        // Block attention (both halves) + the `.0` in to_out.
        check(
            "transformer.transformer_blocks.5.attn.to_q.lora_A.weight",
            "diffusion_model.transformer_blocks.5.attn.to_q.lora_A.weight",
        );
        check(
            "transformer.transformer_blocks.59.attn.to_out.0.lora_B.weight",
            "diffusion_model.transformer_blocks.59.attn.to_out.0.lora_B.weight",
        );
        // The added (text-stream) projections + the LightX2V down/up convention.
        check(
            "transformer.transformer_blocks.0.attn.add_k_proj.lora_down.weight",
            "diffusion_model.transformer_blocks.0.attn.add_k_proj.lora_down.weight",
        );
        // MLP.
        check(
            "transformer.transformer_blocks.12.img_mlp.net.0.proj.lora_up.weight",
            "diffusion_model.transformer_blocks.12.img_mlp.net.0.proj.lora_up.weight",
        );
        // Top-level.
        check(
            "transformer.img_in.lora_A.weight",
            "diffusion_model.img_in.lora_A.weight",
        );
        check(
            "transformer.proj_out.lora_B.weight",
            "diffusion_model.proj_out.lora_B.weight",
        );
    }

    #[test]
    fn count_is_every_site_times_four_halves() {
        // 60*14 block linears + 6 top-level, times the four half-suffixes.
        let expect = (config::N_LAYERS * BLOCK_LINEARS.len() + TOP_LINEARS.len()) * 4;
        assert_eq!(lora_key_renames().len(), expect);
        assert_eq!(expect, 3384);
    }
}
