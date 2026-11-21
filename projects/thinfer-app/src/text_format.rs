//! Constrained decoding: the reply must match a JSON Schema or a Lark grammar.
//! llguidance computes the allowed-token mask per step (disallowed logits are
//! dropped before sampling) and the forced continuation after each token (the
//! keys, brackets and punctuation a schema fixes), which the caller feeds as
//! one multi-token forward instead of decoding it token by token
//! ("jump-forward").
//!
//! Schemas and grammars are job data: their text and the compiler's messages
//! about them never reach logs or (server) error events. [`check`] gives the
//! compiler's detail to a local caller.

use std::path::Path;
use std::sync::Arc;

use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::TokEnv;
use llguidance::{Matcher, ParserFactory};
use toktrie_hf_tokenizers::ByteTokenizer;

/// The shape the reply must take.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "serde", serde(tag = "type", rename_all = "camelCase"))]
pub enum TextFormat {
    /// A JSON Schema (the llguidance subset: types, properties, required,
    /// enum/const, items, anyOf/oneOf, $ref/$defs, string patterns/lengths,
    /// numeric bounds).
    JsonSchema {
        #[cfg_attr(feature = "serve", schema(value_type = Object))]
        schema: serde_json::Value,
    },
    /// A Lark grammar (llguidance dialect; `start` is the entry rule).
    Lark { grammar: String },
}

impl TextFormat {
    pub fn grammar(&self) -> TopLevelGrammar {
        match self {
            Self::JsonSchema { schema } => TopLevelGrammar::from_json_schema(schema.clone()),
            Self::Lark { grammar } => TopLevelGrammar::from_lark(grammar.clone()),
        }
    }
}

/// Error text for the request path (no job data).
pub const REJECTED: &str = "output format rejected: the schema/grammar did not compile \
                        (run it through the local CLI for the compiler's detail)";
const FAILED: &str = "output format: generation left the grammar";

/// The tokenizer as llguidance sees it (token byte strings + a trie).
/// `n_vocab` is the logits row length (models pad past the tokenizer's vocab;
/// `None` = the tokenizer's own size).
pub fn tok_env(
    tokenizer_json: &Path,
    n_vocab: Option<usize>,
    stop: &[u32],
) -> Result<TokEnv, String> {
    let mut bt = ByteTokenizer::from_file(tokenizer_json)
        .map_err(|e| format!("constraint tokenizer: {e}"))?;
    if !stop.is_empty() {
        bt.set_eos_tokens(stop);
    }
    bt.into_tok_env(n_vocab)
        .map_err(|e| format!("constraint tokenizer: {e}"))
}

fn factory(env: &TokEnv) -> Result<ParserFactory, String> {
    let mut f = ParserFactory::new_simple(env).map_err(|e| format!("constraint parser: {e}"))?;
    f.set_stderr_log_level(0).set_buffer_log_level(0);
    Ok(f)
}

/// Compile `format` against the model's tokenizer, returning the compiler's
/// detail on failure. For a local caller (the CLI) only: the detail quotes the
/// schema/grammar.
pub fn check(format: &TextFormat, tokenizer_json: &Path, stop: &[u32]) -> Result<(), String> {
    let env = tok_env(tokenizer_json, None, stop)?;
    factory(&env)?
        .create_parser(format.grammar())
        .map(|_| ())
        .map_err(|e| format!("invalid output format: {e}"))
}

/// Whether `err` is the request path's (detail-free) compile rejection.
pub fn is_rejection(err: &str) -> bool {
    err == REJECTED
}

/// The compiler's detail for a format `model` rejected (local callers only;
/// needs the tokenizer cached).
pub fn explain(model: crate::model::TextModelId, format: &TextFormat) -> String {
    let tokenizer = crate::download::resolve_role(
        model.manifest(),
        thinfer_models::llm::manifest::role::TOKENIZER,
    );
    match tokenizer.and_then(|t| check(format, &t, &[])) {
        Ok(()) => REJECTED.to_string(),
        Err(e) => e,
    }
}

/// One request's constraint state.
pub struct Constraint {
    matcher: Matcher,
    /// Token that opens the constrained part (the model's think-close marker
    /// in thinking mode); `None` once open. Before it the reply is free.
    gate: Option<u32>,
}

impl Constraint {
    /// Compile `grammar`; a failure returns `rejected` (detail-free).
    pub fn new(
        grammar: TopLevelGrammar,
        env: &Arc<TokEnv>,
        gate: Option<u32>,
        rejected: &str,
    ) -> Result<Self, String> {
        let parser = factory(env)?
            .create_parser(grammar)
            .map_err(|_| rejected.to_string())?;
        Ok(Self {
            matcher: Matcher::new(Ok(parser)),
            gate,
        })
    }

    /// Drop every logit the grammar disallows (a no-op while gated).
    pub fn mask(&mut self, logits: &mut [f32]) -> Result<(), String> {
        if self.gate.is_some() {
            return Ok(());
        }
        let allowed = self
            .matcher
            .compute_mask_or_eos()
            .map_err(|_| FAILED.to_string())?;
        allowed.iter_unset_entries(|i| {
            if let Some(l) = logits.get_mut(i) {
                *l = f32::NEG_INFINITY;
            }
        });
        // Rows past the tokenizer's vocab (model padding) are never valid.
        if let Some(pad) = logits.get_mut(allowed.len()..) {
            pad.fill(f32::NEG_INFINITY);
        }
        Ok(())
    }

    /// Record a sampled (or forced) token. Returns the tokens the grammar now
    /// forces, already consumed: the caller appends them to the reply and
    /// feeds them as one forward.
    pub fn accept(&mut self, tok: u32) -> Result<Vec<u32>, String> {
        if let Some(g) = self.gate {
            if tok == g {
                self.gate = None;
                return Ok(self.matcher.consume_ff_tokens());
            }
            return Ok(Vec::new());
        }
        self.matcher
            .consume_token(tok)
            .map_err(|_| FAILED.to_string())?;
        Ok(self.matcher.consume_ff_tokens())
    }

    /// Past the gate: the grammar is shaping the reply.
    pub fn is_open(&self) -> bool {
        self.gate.is_none()
    }

    /// The grammar is complete and admits nothing more.
    pub fn finished(&self) -> bool {
        self.gate.is_none() && self.matcher.is_stopped()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny byte-level tokenizer.json: one token per printable ASCII byte
    /// past space (byte-level maps those to themselves; id = byte - 0x21),
    /// a multi-byte token for `"name"` (94) and `</s>` (EOS, 95).
    fn tokenizer_file(dir: &Path) -> std::path::PathBuf {
        let mut vocab = serde_json::Map::new();
        let mut id = 0u32;
        for b in 0x21u8..0x7f {
            vocab.insert((b as char).to_string(), id.into());
            id += 1;
        }
        vocab.insert("\"name\"".into(), id.into());
        let eos = id + 1;
        vocab.insert("</s>".into(), eos.into());
        let json = serde_json::json!({
            "version": "1.0",
            "added_tokens": [{"id": eos, "content": "</s>", "single_word": false,
                "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": false,
                "trim_offsets": true, "use_regex": true},
            "model": {"type": "BPE", "vocab": vocab, "merges": [], "dropout": null,
                "unk_token": null, "continuing_subword_prefix": null,
                "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false},
        });
        let p = dir.join("tokenizer.json");
        std::fs::write(&p, json.to_string()).unwrap();
        p
    }

    #[test]
    fn enum_schema_forces_and_masks() {
        let dir = tempfile::tempdir().unwrap();
        let tj = tokenizer_file(dir.path());
        let eos = 95;
        let env = Arc::new(tok_env(&tj, Some(96), &[eos]).unwrap());
        let format = TextFormat::JsonSchema {
            schema: serde_json::json!({
                "type": "object",
                "properties": {"name": {"enum": ["ab", "ac"]}},
                "required": ["name"],
                "additionalProperties": false,
            }),
        };
        let mut c = Constraint::new(format.grammar(), &env, None, REJECTED).unwrap();
        // The opening `{` is forced at once, and the whole key soon after.
        let mut out = Vec::new();
        let mut logits = vec![0.0f32; 96];
        for _ in 0..64 {
            logits.iter_mut().for_each(|l| *l = 0.0);
            c.mask(&mut logits).unwrap();
            let allowed: Vec<usize> = (0..96).filter(|&i| logits[i].is_finite()).collect();
            assert!(!allowed.is_empty());
            // Always pick the highest allowed id that is not EOS unless EOS is
            // the only option: exercises a deterministic path.
            let tok = if allowed == [eos as usize] {
                eos
            } else {
                *allowed.iter().rfind(|&&i| i != eos as usize).unwrap() as u32
            };
            if tok == eos {
                break;
            }
            out.push(tok);
            out.extend(c.accept(tok).unwrap());
            if c.finished() {
                break;
            }
        }
        let text: String = out
            .iter()
            .map(|&t| {
                if t == 94 {
                    "\"name\"".to_string()
                } else {
                    ((t as u8 + 0x21) as char).to_string()
                }
            })
            .collect();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let name = v["name"].as_str().unwrap();
        assert!(name == "ab" || name == "ac", "{text}");
    }

    #[test]
    fn bad_schema_error_has_no_detail() {
        let dir = tempfile::tempdir().unwrap();
        let tj = tokenizer_file(dir.path());
        let env = Arc::new(tok_env(&tj, Some(96), &[95]).unwrap());
        let bad = TextFormat::Lark {
            grammar: "start: secret_rule_name_xyz".into(),
        };
        let err = Constraint::new(bad.grammar(), &env, None, REJECTED)
            .err()
            .unwrap();
        assert!(!err.contains("secret_rule_name_xyz"));
        assert!(check(&bad, &tj, &[95]).is_err());
    }

    #[test]
    fn gate_leaves_reply_free_until_marker() {
        let dir = tempfile::tempdir().unwrap();
        let tj = tokenizer_file(dir.path());
        let env = Arc::new(tok_env(&tj, Some(96), &[95]).unwrap());
        let format = TextFormat::Lark {
            grammar: "start: \"x\"".into(),
        };
        let gate = 2; // '#'
        let mut c = Constraint::new(format.grammar(), &env, Some(gate), REJECTED).unwrap();
        let mut logits = vec![0.0f32; 96];
        c.mask(&mut logits).unwrap();
        assert!(logits.iter().all(|l| l.is_finite()));
        assert!(c.accept(10).unwrap().is_empty());
        // Opening the gate: the one-token grammar is forced outright.
        assert_eq!(c.accept(gate).unwrap(), vec![b'x' as u32 - 0x21]);
        assert!(c.finished());
    }
}
