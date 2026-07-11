//! Native prompt-rewriter (Qwen3-VL-8B-Instruct) conformance suite. Bundled into
//! one test binary (cargo auto-discovers `tests/<dir>/main.rs`); submodules are
//! the individual tests. Gated on the `qwen3-lm` feature (needs the Q5_K_M GGUF
//! in the HF cache; skips cleanly when uncached).

mod kv_generate;
mod load_smoke;
mod logits_parity;
