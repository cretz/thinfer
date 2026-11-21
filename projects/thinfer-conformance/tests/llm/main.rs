//! LLM parity suite. Gated on `llm-e2e` (needs the unsloth GGUFs + official
//! config/tokenizer in the HF cache + `uv`).

mod bench;
mod gemma4_bench;
mod gemma4_parity;
mod gemv;
mod parity;
mod qwen36_bench;
mod qwen36_parity;
mod session;
mod util;
mod window_ring;
