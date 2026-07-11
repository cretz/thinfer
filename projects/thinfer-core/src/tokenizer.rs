//! Tokenizer trait. Backend-agnostic interface that model code (e.g.
//! `Qwen3Encoder`) consumes via `&[u32]` token IDs.
//!
//! Implementations live outside core to keep wasm builds light:
//! - `thinfer-native`: HF `tokenizers` crate via `HfTokenizer`.
//! - `thinfer-web`: JS-side `tokenizers.js` via the browser's network-cached
//!   copy (or wasm-compiled tokenizers when size budget allows).
//!
//! Tokenization input is the *already-chat-template-formatted* prompt string.
//! Chat-template wrapping is model-specific and lives in `thinfer-models`
//! (e.g. `z_image::tokenizer::format_qwen3_prompt`); the tokenizer itself is
//! template-agnostic.

#[derive(Debug)]
pub enum TokenizerError {
    Load(String),
    Encode(String),
    Decode(String),
}

impl core::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Load(s) => write!(f, "tokenizer load: {s}"),
            Self::Encode(s) => write!(f, "tokenizer encode: {s}"),
            Self::Decode(s) => write!(f, "tokenizer decode: {s}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

/// Backend-agnostic tokenizer. Implementations are not required to be
/// thread-safe; callers serialize as needed.
pub trait Tokenizer {
    /// Tokenize `text`, with no chat-template wrapping (callers handle
    /// templating upstream). `add_special_tokens` selects whether the
    /// tokenizer's configured specials (e.g. a trailing T5/umT5 `</s>` EOS) are
    /// inserted: chat-template models pass `false` (their specials are already
    /// literal text in the wrapped prompt), while umT5/T5 pass `true` to append
    /// the EOS the reference encoders expect.
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError>;

    /// Detokenize `ids` back to a String. `skip_special_tokens` drops the
    /// tokenizer's configured special tokens (e.g. ChatML `<|im_end|>`) from the
    /// output, as the HF `tokenizers` `decode` does. Used by generator loops
    /// (e.g. the Qwen3 rewriter) to turn produced token ids into a caption.
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError>;
}
