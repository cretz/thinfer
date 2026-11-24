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

    /// A Qwen2 byte-level BPE from `vocab.json` + `merges.txt` +
    /// `tokenizer_config.json` (repos that ship no `tokenizer.json`): the
    /// same tokenizer `transformers` builds when it converts the slow one
    /// (NFC, the Qwen2 split regex, byte-level BPE, added tokens from the
    /// config).
    pub async fn from_bpe_files(
        vocab: impl AsRef<Path>,
        merges: impl AsRef<Path>,
        config: impl AsRef<Path>,
    ) -> Result<Self, TokenizerError> {
        let read = |p: &Path| {
            let p = p.to_path_buf();
            async move {
                tokio::fs::read_to_string(&p)
                    .await
                    .map_err(|e| TokenizerError::Load(format!("{}: {e}", p.display())))
            }
        };
        let vocab: serde_json::Value = serde_json::from_str(&read(vocab.as_ref()).await?)
            .map_err(|e| TokenizerError::Load(format!("vocab.json: {e}")))?;
        let merges: Vec<String> = read(merges.as_ref())
            .await?
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with("#version"))
            .map(str::to_string)
            .collect();
        let cfg: serde_json::Value = serde_json::from_str(&read(config.as_ref()).await?)
            .map_err(|e| TokenizerError::Load(format!("tokenizer_config.json: {e}")))?;
        let mut added: Vec<serde_json::Value> = cfg["added_tokens_decoder"]
            .as_object()
            .map(|m| {
                m.iter()
                    .filter_map(|(id, t)| {
                        let mut t = t.clone();
                        t["id"] = serde_json::json!(id.parse::<u32>().ok()?);
                        Some(t)
                    })
                    .collect()
            })
            .unwrap_or_default();
        added.sort_by_key(|t| t["id"].as_u64());
        let json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added,
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": QWEN2_SPLIT}, "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}
            ]},
            "post_processor": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true},
            "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
            "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "",
                      "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false,
                      "ignore_merges": false, "vocab": vocab, "merges": merges}
        });
        Self::from_bytes(json.to_string().as_bytes())
    }
}

/// The Qwen2 pre-tokenizer split (as `transformers` writes it).
const QWEN2_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

impl Tokenizer for HfTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError> {
        let enc = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| TokenizerError::Encode(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| TokenizerError::Decode(e.to_string()))
    }
}
