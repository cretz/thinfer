//! LTX-2.3 distilled LoRA fold. The fold itself is the shared, model-agnostic
//! [`crate::common::lora`] (auto-discovery, stacking, both key conventions);
//! this module just re-exports it so the LTX pipeline and its `discover_specs`/
//! `LoraFoldSource` call sites keep their `ltx::lora::*` paths.
//!
//! LTX usage: the published Sulphur GGUF is `sulphur_dev` (the BASE checkpoint);
//! the 8-step CFG-free distilled path only converges with the distill LoRA
//! folded on top (ComfyUI `t2v distilled` loads dev + the distill LoRA at
//! strength 1.0). The LoRA carries no `.alpha` (ai-toolkit "rerank" already baked
//! alpha/rank into A/B), so strength 1.0 = a plain `B @ A`. Cross-attn + adaLN
//! modules are zeroed in the checkpoint and auto-skipped by discovery.

pub use crate::common::lora::*;
