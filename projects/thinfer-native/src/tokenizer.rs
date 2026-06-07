//! Native `thinfer_core::tokenizer::Tokenizer` impl backed by the HF
//! `tokenizers` crate.
//!
//! Construction is async (file open + read of `tokenizer.json`); after that
//! the parsed structure is sync and held for the session.

use std::path::Path;
use thinfer_core::tokenizer::{Tokenizer, TokenizerError};
use tokenizers::Tokenizer as HfInner;

pub struct HfTokenizer {
    inner: HfInner,
}

impl HfTokenizer {
    /// Load and parse `tokenizer.json` from disk. The raw JSON bytes are
    /// dropped after parse; only the in-memory vocab/merges remain.
    pub async fn from_path(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let bytes = tokio::fs::read(path.as_ref())
            .await
            .map_err(|e| TokenizerError::Load(e.to_string()))?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TokenizerError> {
        let inner = HfInner::from_bytes(bytes).map_err(|e| TokenizerError::Load(e.to_string()))?;
        Ok(Self { inner })
    }
}

impl Tokenizer for HfTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| TokenizerError::Encode(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }
}
