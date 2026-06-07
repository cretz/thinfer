//! Z-Image prompt formatting (Qwen3 chat template, single user message).
//!
//! Z-Image's pipeline (`src/zimage/pipeline.py:111`) calls
//! `tokenizer.apply_chat_template(messages=[{role: user, content: prompt}],
//! add_generation_prompt=True, enable_thinking=True)`. Rendering that template
//! for our case (single user message, no tools, no system message) collapses
//! to the literal string format below: byte-equal to running the Jinja
//! template in transformers, verified for both empty and non-empty prompts.
//!
//! The actual tokenization is platform-specific (HF `tokenizers` crate on
//! native, JS-side `tokenizers.js` on web) and goes through
//! `thinfer_core::tokenizer::Tokenizer`. This module just produces the
//! template-wrapped string for that trait to consume.

pub fn format_qwen3_prompt(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_matches_upstream_jinja() {
        // Rendered output of upstream chat_template with
        // add_generation_prompt=True, enable_thinking=True, for our two cases.
        assert_eq!(
            format_qwen3_prompt("A red apple on a table"),
            "<|im_start|>user\nA red apple on a table<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            format_qwen3_prompt(""),
            "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n"
        );
    }
}
