//! LTX-2.3 distilled LoRA fold. The fold itself is the shared, model-agnostic
//! [`crate::common::lora`] (auto-discovery, stacking, both key conventions);
//! this module just re-exports it so the LTX pipeline and its `discover_specs`/
//! `LoraFoldSource` call sites keep their `ltx::lora::*` paths.
//!
//! LTX usage: a `dev` DiT is the BASE checkpoint; the 8-step CFG-free distilled
//! path only converges with a distill LoRA folded on top (ComfyUI `t2v
//! distilled` loads dev + the distill LoRA at strength 1.0). Rank-reduced
//! ("rerank") LoRAs carry no `.alpha` (alpha/rank baked into A/B), so strength
//! 1.0 = a plain `B @ A`; zeroed modules are auto-skipped by discovery.

pub use crate::common::lora::*;
