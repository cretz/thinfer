//! Adapter (LoRA) key renames for the Wan-family DiT (AnyFlow-Wan2.1-T2V-14B and
//! any other diffusers-named Wan base). The base DiT carries the diffusers tensor
//! names (`blocks.N.attn1.to_q.weight`, `attn2.*` cross-attn, `ffn.net.0.proj` /
//! `ffn.net.2`), while community Wan LoRAs ship in TWO conventions:
//!
//!  1. ComfyUI / original-Wan (the dominant Civitai format):
//!     `diffusion_model.blocks.N.{self_attn,cross_attn}.{q,k,v,o}.lora_{down,up}
//!     .weight` + `diffusion_model.blocks.N.ffn.{0,2}.lora_{down,up}.weight`.
//!  2. PEFT / diffusers: `transformer.blocks.N.{attn1,attn2}.{to_q,to_k,to_v,
//!     to_out.0}.lora_{A,B}.weight` + `transformer.blocks.N.ffn.net.{0.proj,2}`.
//!
//! The generic [`crate::common::lora`] fold discovers a site by matching
//! `diffusion_model.{X}.lora_{A,down}.weight` against the base tensor `{X}.weight`,
//! so both conventions must land on `diffusion_model.{diffusers-base}`. This map
//! does exactly that: convention 1 renames the original-Wan submodule names to
//! diffusers under the SAME `diffusion_model.` prefix; convention 2 swaps the
//! `transformer.` prefix for `diffusion_model.` (the submodule names already
//! match the base). A LoRA that already uses `diffusion_model.` + diffusers names
//! passes through untouched and folds directly. Keys the map does not cover pass
//! through unchanged and are ignored by discovery.
//!
//! Only the attn q/k/v/o + ffn up/down linears are mapped -- the minimal,
//! transfer-safe fold set. LoRAs that also touch `patch_embedding` / norms /
//! `condition_embedder` / `proj_out` degrade cross-LoRA compatibility on distilled
//! Wan bases, so those sites are intentionally left unmapped (ignored if present).

use std::collections::HashMap;

use thinfer_core::weight::WeightId;

/// The four LoRA weight-half suffixes across both conventions (ai-toolkit/PEFT
/// `lora_A`/`lora_B`, ComfyUI `lora_down`/`lora_up`). Renaming preserves the
/// suffix; the fold pairs the halves.
const LORA_SUFFIXES: [&str; 4] = [
    ".lora_down.weight",
    ".lora_up.weight",
    ".lora_A.weight",
    ".lora_B.weight",
];

/// Per-block linear submodules as `(original-Wan, diffusers)` path fragments
/// relative to `blocks.N`. Self-attn `attn1`, cross-attn `attn2`, SwiGLU-less MLP
/// `ffn.net.0.proj` (up) / `ffn.net.2` (down). Norms + `scale_shift_table` are not
/// linears, so no LoRA folds them.
const BLOCK_SUBMODULES: [(&str, &str); 10] = [
    ("self_attn.q", "attn1.to_q"),
    ("self_attn.k", "attn1.to_k"),
    ("self_attn.v", "attn1.to_v"),
    ("self_attn.o", "attn1.to_out.0"),
    ("cross_attn.q", "attn2.to_q"),
    ("cross_attn.k", "attn2.to_k"),
    ("cross_attn.v", "attn2.to_v"),
    ("cross_attn.o", "attn2.to_out.0"),
    ("ffn.0", "ffn.net.0.proj"),
    ("ffn.2", "ffn.net.2"),
];

/// adapter-id -> base-fold-id renames for a Wan LoRA over `num_layers` blocks.
/// Emits, per `(block, submodule, lora-half)`, both convention entries:
///   `diffusion_model.blocks.N.{wan}{suffix}`  -> `diffusion_model.blocks.N.{diff}{suffix}`
///   `transformer.blocks.N.{diff}{suffix}`     -> `diffusion_model.blocks.N.{diff}{suffix}`
/// Pass to `RenamedSource::with_passthrough` before `discover_specs`.
pub fn lora_key_renames(num_layers: usize) -> HashMap<WeightId, WeightId> {
    // num_layers blocks * 10 linears * 4 suffixes * 2 conventions.
    let mut out =
        HashMap::with_capacity(num_layers * BLOCK_SUBMODULES.len() * LORA_SUFFIXES.len() * 2);
    for n in 0..num_layers {
        for (wan_sub, diff_sub) in BLOCK_SUBMODULES {
            for suffix in LORA_SUFFIXES {
                let base = format!("diffusion_model.blocks.{n}.{diff_sub}{suffix}");
                // Convention 1: original-Wan submodule names, same prefix.
                out.insert(
                    WeightId(format!("diffusion_model.blocks.{n}.{wan_sub}{suffix}")),
                    WeightId(base.clone()),
                );
                // Convention 2: diffusers submodule names, `transformer.` prefix.
                out.insert(
                    WeightId(format!("transformer.blocks.{n}.{diff_sub}{suffix}")),
                    WeightId(base),
                );
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_both_conventions_onto_the_diffusers_base() {
        let m = lora_key_renames(40);
        let check = |from: &str, to: &str| {
            assert_eq!(
                m.get(&WeightId(from.to_string())),
                Some(&WeightId(to.to_string())),
                "{from}"
            );
        };
        // ComfyUI / original-Wan (self + cross attn, ffn up/down, down/up halves).
        check(
            "diffusion_model.blocks.0.self_attn.q.lora_down.weight",
            "diffusion_model.blocks.0.attn1.to_q.lora_down.weight",
        );
        check(
            "diffusion_model.blocks.39.self_attn.o.lora_up.weight",
            "diffusion_model.blocks.39.attn1.to_out.0.lora_up.weight",
        );
        check(
            "diffusion_model.blocks.5.cross_attn.k.lora_down.weight",
            "diffusion_model.blocks.5.attn2.to_k.lora_down.weight",
        );
        check(
            "diffusion_model.blocks.5.ffn.0.lora_down.weight",
            "diffusion_model.blocks.5.ffn.net.0.proj.lora_down.weight",
        );
        check(
            "diffusion_model.blocks.5.ffn.2.lora_up.weight",
            "diffusion_model.blocks.5.ffn.net.2.lora_up.weight",
        );
        // PEFT / diffusers (`transformer.` prefix, A/B halves).
        check(
            "transformer.blocks.12.attn1.to_v.lora_A.weight",
            "diffusion_model.blocks.12.attn1.to_v.lora_A.weight",
        );
        check(
            "transformer.blocks.12.attn2.to_out.0.lora_B.weight",
            "diffusion_model.blocks.12.attn2.to_out.0.lora_B.weight",
        );
        // A LoRA already on the base convention is not in the map (passes through).
        assert!(!m.contains_key(&WeightId(
            "diffusion_model.blocks.0.attn1.to_q.lora_down.weight".into()
        )));
    }
}
