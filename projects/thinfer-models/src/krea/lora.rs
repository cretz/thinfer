//! Adapter (LoRA) key renames for Krea 2 Turbo. Community Krea LoRAs (ai-toolkit,
//! the "Krea 2 Official Loras" on Civitai) ship DIFFUSERS module names
//! (`transformer.transformer_blocks.N.attn.to_q`, `img_in`, `final_layer`,
//! `ff.gate`, `time_embed`), while our base DiT is the sd.cpp-native `krea2` GGUF
//! (`blocks.N.attn.wq`, `first`, `last.linear`, `mlp.gate`, `tmlp`). The generic
//! [`crate::common::lora`] fold discovers a site by matching
//! `diffusion_model.{X}.lora_{A,B}.weight` against the base tensor `{X}.weight`,
//! so a diffusers-named adapter folds ZERO sites until its keys are mapped to the
//! base's naming.
//!
//! [`lora_key_renames`] is that map (like `qwen3_gguf_renames` for the encoder):
//! wrap the adapter source in `RenamedSource::with_passthrough(adapter,
//! lora_key_renames())` before `discover_specs`. Only the module path changes;
//! the `.lora_A.weight` / `.lora_B.weight` suffix (which the fold reads to pair
//! `A`/`B`) is preserved, and the `transformer.` prefix becomes `diffusion_model.`
//! (the prefix the fold keys on). Keys the map does not cover pass through
//! unchanged and are simply ignored by discovery.

use std::collections::HashMap;

use thinfer_core::weight::WeightId;

use crate::krea::config;

/// The two LoRA weight halves the fold pairs. Renaming preserves these suffixes.
const LORA_SUFFIXES: [&str; 2] = [".lora_A.weight", ".lora_B.weight"];

/// The eight linear submodules inside every Krea block (image or text), as
/// `(diffusers, sd.cpp)` path fragments relative to the block prefix. Attention
/// q/k/v/gate/out + the SwiGLU gate/up/down. The norms and the modulation offset
/// are not linears, so no LoRA touches them.
const BLOCK_SUBMODULES: [(&str, &str); 8] = [
    ("attn.to_q", "attn.wq"),
    ("attn.to_k", "attn.wk"),
    ("attn.to_v", "attn.wv"),
    ("attn.to_gate", "attn.gate"),
    ("attn.to_out.0", "attn.wo"),
    ("ff.gate", "mlp.gate"),
    ("ff.up", "mlp.up"),
    ("ff.down", "mlp.down"),
];

/// Top-level (non-block) linears, as `(diffusers, sd.cpp)` module paths. The
/// text-fusion `projector` is a 1-D vector in the base, so the fold skips it;
/// mapping it anyway is harmless and keeps the map complete.
const TOP_MODULES: [(&str, &str); 8] = [
    ("img_in", "first"),
    ("final_layer.linear", "last.linear"),
    ("time_embed.linear_1", "tmlp.0"),
    ("time_embed.linear_2", "tmlp.2"),
    ("time_mod_proj", "tproj.1"),
    ("txt_in.linear_1", "txtmlp.1"),
    ("txt_in.linear_2", "txtmlp.3"),
    ("text_fusion.projector", "txtfusion.projector"),
];

/// diffusers-adapter-id -> base-fold-id renames for a Krea 2 adapter. Emits one
/// entry per `(site, lora half)`: `transformer.{diffusers}{suffix}` ->
/// `diffusion_model.{sdcpp}{suffix}`. Covers all 28 image blocks, the 2+2
/// text-fusion blocks, and the top-level linears. Pass to
/// `RenamedSource::with_passthrough`.
pub fn lora_key_renames() -> HashMap<WeightId, WeightId> {
    // 28 image + 2 layerwise + 2 refiner text blocks, 8 submodules each, plus the
    // top-level modules, times the two lora halves.
    let n_blocks = config::N_LAYERS + config::N_LAYERWISE_BLOCKS + config::N_REFINER_BLOCKS;
    let mut out =
        HashMap::with_capacity((n_blocks * BLOCK_SUBMODULES.len() + TOP_MODULES.len()) * 2);

    let mut insert_module = |diffusers: &str, sdcpp: &str| {
        for suffix in LORA_SUFFIXES {
            out.insert(
                WeightId(format!("transformer.{diffusers}{suffix}")),
                WeightId(format!("diffusion_model.{sdcpp}{suffix}")),
            );
        }
    };

    // Blocks: image `transformer_blocks.N` -> `blocks.N`; text-fusion blocks keep
    // their layerwise/refiner naming under `txtfusion`.
    let block_prefixes = std::iter::empty()
        .chain(
            (0..config::N_LAYERS)
                .map(|i| (format!("transformer_blocks.{i}"), format!("blocks.{i}"))),
        )
        .chain((0..config::N_LAYERWISE_BLOCKS).map(|i| {
            (
                format!("text_fusion.layerwise_blocks.{i}"),
                format!("txtfusion.layerwise_blocks.{i}"),
            )
        }))
        .chain((0..config::N_REFINER_BLOCKS).map(|i| {
            (
                format!("text_fusion.refiner_blocks.{i}"),
                format!("txtfusion.refiner_blocks.{i}"),
            )
        }));
    for (diff_prefix, sd_prefix) in block_prefixes {
        for (diff_sub, sd_sub) in BLOCK_SUBMODULES {
            insert_module(
                &format!("{diff_prefix}.{diff_sub}"),
                &format!("{sd_prefix}.{sd_sub}"),
            );
        }
    }

    for (diff, sdcpp) in TOP_MODULES {
        insert_module(diff, sdcpp);
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
        // Image block attn + ff, both halves.
        check(
            "transformer.transformer_blocks.5.attn.to_q.lora_A.weight",
            "diffusion_model.blocks.5.attn.wq.lora_A.weight",
        );
        check(
            "transformer.transformer_blocks.27.attn.to_out.0.lora_B.weight",
            "diffusion_model.blocks.27.attn.wo.lora_B.weight",
        );
        check(
            "transformer.transformer_blocks.0.ff.down.lora_A.weight",
            "diffusion_model.blocks.0.mlp.down.lora_A.weight",
        );
        // Text-fusion layerwise + refiner.
        check(
            "transformer.text_fusion.layerwise_blocks.1.attn.to_k.lora_A.weight",
            "diffusion_model.txtfusion.layerwise_blocks.1.attn.wk.lora_A.weight",
        );
        check(
            "transformer.text_fusion.refiner_blocks.0.ff.up.lora_B.weight",
            "diffusion_model.txtfusion.refiner_blocks.0.mlp.up.lora_B.weight",
        );
        // Top-level.
        check(
            "transformer.img_in.lora_A.weight",
            "diffusion_model.first.lora_A.weight",
        );
        check(
            "transformer.final_layer.linear.lora_B.weight",
            "diffusion_model.last.linear.lora_B.weight",
        );
        check(
            "transformer.time_mod_proj.lora_A.weight",
            "diffusion_model.tproj.1.lora_A.weight",
        );
    }

    #[test]
    fn count_is_every_site_times_two_halves() {
        // 264 diffusers sites (28*8 image + 2*8 + 2*8 text + 8 top), 2 halves each.
        let expect = ((config::N_LAYERS + config::N_LAYERWISE_BLOCKS + config::N_REFINER_BLOCKS)
            * BLOCK_SUBMODULES.len()
            + TOP_MODULES.len())
            * 2;
        assert_eq!(lora_key_renames().len(), expect);
        assert_eq!(expect, 528);
    }
}
