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
}

impl core::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Load(s) => write!(f, "tokenizer load: {s}"),
            Self::Encode(s) => write!(f, "tokenizer encode: {s}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

/// Backend-agnostic tokenizer. Implementations are not required to be
/// thread-safe; callers serialize as needed.
pub trait Tokenizer {
    /// Tokenize `text` exactly: no special-token insertion, no chat-template
    /// wrapping. Callers handle templating upstream.
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError>;
}
