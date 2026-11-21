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

use crate::model::TextModelEngine;
use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::TokEnv;
use llguidance::{Matcher, ParserFactory};
use toktrie_hf_tokenizers::ByteTokenizer;

pub use thinfer_client::text_format::TextFormat;

/// The llguidance grammar for `format`.
pub fn top_level(format: &TextFormat) -> TopLevelGrammar {
    match format {
        TextFormat::JsonSchema { schema } => TopLevelGrammar::from_json_schema(strict(schema)),
        TextFormat::Lark { grammar } => TopLevelGrammar::from_lark(grammar.clone()),
    }
}

/// A JSON Schema with thinfer's defaults: single-line JSON spaced the usual
/// way (`", "` between items, `": "` after keys, no other whitespace) and
/// closed objects. llguidance's own defaults leave objects open to extra keys
/// and allow whitespace anywhere, and a model can wander into either (an
/// invented key, then whitespace until the token budget runs out). Fully
/// compact separators are worse: models mostly saw `": "`, and forced into
/// `":"` one wrote an empty string where it wrote prose with the space. The
/// caller's own choices win: an explicit `additionalProperties` (or
/// `unevaluatedProperties`) is kept, and `x-guidance` keys override ours.
/// Only objects that declare `properties` are closed: a bare
/// `{"type": "object"}` stays a free-form map.
pub fn strict(schema: &serde_json::Value) -> serde_json::Value {
    let mut s = schema.clone();
    close_objects(&mut s);
    if let Some(root) = s.as_object_mut() {
        let mut guidance = serde_json::json!({
            "whitespace_flexible": false,
            "key_separator": ": ",
            "item_separator": ", ",
        });
        if let Some(theirs) = root.get("x-guidance").and_then(|g| g.as_object()) {
            let ours = guidance.as_object_mut().expect("object");
            for (k, v) in theirs {
                ours.insert(k.clone(), v.clone());
            }
        }
        root.insert("x-guidance".into(), guidance);
    }
    s
}

/// Add `additionalProperties: false` to every object schema with
/// `properties` that sets neither it nor `unevaluatedProperties`, walking
/// only schema positions (never `enum` / `const` / `default` data, never a
/// `properties` map's names).
fn close_objects(v: &mut serde_json::Value) {
    use serde_json::Value;
    let Value::Object(o) = v else {
        return;
    };
    if o.contains_key("properties")
        && !o.contains_key("additionalProperties")
        && !o.contains_key("unevaluatedProperties")
    {
        o.insert("additionalProperties".into(), Value::Bool(false));
    }
    const ONE: &[&str] = &[
        "items",
        "additionalItems",
        "additionalProperties",
        "unevaluatedItems",
        "unevaluatedProperties",
        "contains",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
    ];
    const MAP: &[&str] = &[
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        "dependentSchemas",
    ];
    const LIST: &[&str] = &["anyOf", "oneOf", "allOf", "prefixItems"];
    for (k, child) in o.iter_mut() {
        let k = k.as_str();
        if ONE.contains(&k) {
            match child {
                // `items` may still be a list (draft 4-7 tuples).
                Value::Array(a) => a.iter_mut().for_each(close_objects),
                c => close_objects(c),
            }
        } else if MAP.contains(&k) {
            if let Value::Object(m) = child {
                m.values_mut().for_each(close_objects);
            }
        } else if LIST.contains(&k)
            && let Value::Array(a) = child
        {
            a.iter_mut().for_each(close_objects);
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
        .create_parser(top_level(format))
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
        tokenizer_file_with(dir, false)
    }

    /// [`tokenizer_file`], optionally with a space token (`Ġ`, id 96).
    fn tokenizer_file_with(dir: &Path, space: bool) -> std::path::PathBuf {
        let mut vocab = serde_json::Map::new();
        let mut id = 0u32;
        for b in 0x21u8..0x7f {
            vocab.insert((b as char).to_string(), id.into());
            id += 1;
        }
        vocab.insert("\"name\"".into(), id.into());
        let eos = id + 1;
        vocab.insert("</s>".into(), eos.into());
        if space {
            vocab.insert("\u{120}".into(), (eos + 1).into());
        }
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
                // The toy vocab has no space-led tokens: compact separators.
                "x-guidance": {"key_separator": ":", "item_separator": ","},
            }),
        };
        let mut c = Constraint::new(top_level(&format), &env, None, REJECTED).unwrap();
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
        let err = Constraint::new(top_level(&bad), &env, None, REJECTED)
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
        let mut c = Constraint::new(top_level(&format), &env, Some(gate), REJECTED).unwrap();
        let mut logits = vec![0.0f32; 96];
        c.mask(&mut logits).unwrap();
        assert!(logits.iter().all(|l| l.is_finite()));
        assert!(c.accept(10).unwrap().is_empty());
        // Opening the gate: the one-token grammar is forced outright.
        assert_eq!(c.accept(gate).unwrap(), vec![b'x' as u32 - 0x21]);
        assert!(c.finished());
    }

    #[test]
    fn strict_closes_declared_objects_only_and_keeps_caller_choices() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                // A property literally named `properties`: a name, not a schema.
                "properties": {"type": "string"},
                "inner": {"type": "object", "properties": {"x": {"type": "integer"}}},
                "free": {"type": "object"},
                "open": {"type": "object", "properties": {}, "additionalProperties": true},
                "list": {"type": "array", "items": {"properties": {"y": {}}}},
                "choice": {"anyOf": [{"properties": {"z": {}}}, {"type": "null"}]},
                "fixed": {"enum": [{"properties": {"data": 1}}]},
            },
            "$defs": {"d": {"properties": {"w": {}}}},
            "x-guidance": {"key_separator": ":"},
        });
        let s = strict(&schema);
        assert_eq!(s["additionalProperties"], false);
        let p = &s["properties"];
        assert!(p.get("additionalProperties").is_none(), "names untouched");
        assert_eq!(p["inner"]["additionalProperties"], false);
        assert!(
            p["free"].get("additionalProperties").is_none(),
            "free-form map stays open"
        );
        assert_eq!(
            p["open"]["additionalProperties"], true,
            "explicit choice kept"
        );
        assert_eq!(p["list"]["items"]["additionalProperties"], false);
        assert_eq!(p["choice"]["anyOf"][0]["additionalProperties"], false);
        assert!(
            p["fixed"]["enum"][0].get("additionalProperties").is_none(),
            "enum data untouched"
        );
        assert_eq!(s["$defs"]["d"]["additionalProperties"], false);
        assert_eq!(
            s["x-guidance"],
            serde_json::json!({
                "whitespace_flexible": false,
                "key_separator": ":",
                "item_separator": ", ",
            }),
            "theirs override ours, ours fill the rest"
        );
    }

    /// The allowed ids after feeding `text` (one token per character) under
    /// `grammar`, following forced runs.
    fn allowed_after(
        grammar: TopLevelGrammar,
        env: &Arc<TokEnv>,
        vocab: usize,
        text: &str,
    ) -> Vec<usize> {
        let tok = |c: char| if c == ' ' { 96 } else { c as u32 - 0x21 };
        let want: Vec<u32> = text.chars().map(tok).collect();
        let mut c = Constraint::new(grammar, env, None, REJECTED).unwrap();
        let mut logits = vec![0.0f32; vocab];
        let mut i = 0;
        while i < want.len() {
            logits.iter_mut().for_each(|l| *l = 0.0);
            c.mask(&mut logits).unwrap();
            assert!(
                logits[want[i] as usize].is_finite(),
                "{text}: step {i} refused; allowed {:?}",
                (0..vocab)
                    .filter(|&k| logits[k].is_finite())
                    .collect::<Vec<_>>()
            );
            let forced = c.accept(want[i]).unwrap();
            i += 1;
            for f in forced {
                assert_eq!(Some(&f), want.get(i), "{text}: forced run diverged");
                i += 1;
            }
        }
        logits.iter_mut().for_each(|l| *l = 0.0);
        c.mask(&mut logits).unwrap();
        (0..vocab).filter(|&i| logits[i].is_finite()).collect()
    }

    #[test]
    fn strict_schema_refuses_extra_keys_and_free_whitespace() {
        let dir = tempfile::tempdir().unwrap();
        let tj = tokenizer_file_with(dir.path(), true);
        let env = Arc::new(tok_env(&tj, Some(97), &[95]).unwrap());
        // The toy vocab has no space-led tokens: compact separators (the
        // default spacing is checked on a real tokenizer below).
        let schema = serde_json::json!({"type": "object",
            "properties": {"a": {"type": "integer"}},
            "x-guidance": {"key_separator": ":", "item_separator": ","}});
        let (comma, space) = (',' as usize - 0x21, 96);
        let strict_g = || {
            top_level(&TextFormat::JsonSchema {
                schema: schema.clone(),
            })
        };
        let loose_g = || TopLevelGrammar::from_json_schema(schema.clone());
        // Closed: after the only key's value, no `,` (llguidance's default
        // would let the model start an invented key there).
        assert!(!allowed_after(strict_g(), &env, 97, "{\"a\":1").contains(&comma));
        assert!(allowed_after(loose_g(), &env, 97, "{\"a\":1").contains(&comma));
        // No free whitespace.
        assert!(!allowed_after(strict_g(), &env, 97, "{").contains(&space));
        assert!(allowed_after(loose_g(), &env, 97, "{").contains(&space));
    }

    /// Whether `grammar` accepts all of `text` (Gemma's tokenization, token by
    /// token, no jump-forward) and may end there. `None`: tokenizer not cached.
    fn gemma_accepts(grammar: TopLevelGrammar, text: &str) -> Option<bool> {
        use crate::model::TextModelEngine;
        use thinfer_core::tokenizer::Tokenizer;
        use thinfer_models::llm::manifest::role;
        let m = crate::model::TextModelId::Gemma4_26bA4b.manifest();
        let (Ok(tj), Ok(gc)) = (
            crate::download::resolve_role(m, role::TOKENIZER),
            crate::download::resolve_role(m, role::GENERATION_CONFIG),
        ) else {
            eprintln!("skipped: gemma tokenizer not cached");
            return None;
        };
        let tok = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(thinfer_native::tokenizer::HfTokenizer::from_path(&tj))
            .unwrap();
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(gc).unwrap()).unwrap();
        let stop: Vec<u32> = match &cfg["eos_token_id"] {
            serde_json::Value::Array(a) => a.iter().map(|v| v.as_u64().unwrap() as u32).collect(),
            v => vec![v.as_u64().unwrap() as u32],
        };
        let env = tok_env(&tj, None, &stop).unwrap();
        let mut m = Matcher::new(Ok(factory(&env).unwrap().create_parser(grammar).unwrap()));
        for t in tok.encode(text, false).unwrap() {
            if !m.compute_mask_or_eos().unwrap().is_allowed(t) {
                return Some(false);
            }
            m.consume_token(t).unwrap();
        }
        let end = m.compute_mask_or_eos().unwrap();
        Some(m.is_stopped() || stop.iter().any(|&s| end.is_allowed(s)))
    }

    #[test]
    fn strict_schema_takes_spaced_single_line_json_only() {
        let schema = serde_json::json!({"type": "object", "properties": {
            "a": {"type": "integer"}, "b": {"type": "string"}}});
        let strict_g = || {
            top_level(&TextFormat::JsonSchema {
                schema: schema.clone(),
            })
        };
        let loose_g = || TopLevelGrammar::from_json_schema(schema.clone());
        for (text, strict, loose) in [
            (r#"{"a": 1, "b": "x"}"#, true, true),
            (r#"{"a":1,"b":"x"}"#, false, true),
            (r#"{ "a": 1}"#, false, true),
            (r#"{"a": 1,  "b": "x"}"#, false, true),
            ("{\"a\": 1,\n\"b\": \"x\"}", false, true),
            (r#"{"a": 1, "c": 2}"#, false, true),
        ] {
            let Some(got) = gemma_accepts(strict_g(), text) else {
                return;
            };
            assert_eq!(got, strict, "strict {text:?}");
            assert_eq!(
                gemma_accepts(loose_g(), text),
                Some(loose),
                "loose {text:?}"
            );
        }
    }

    /// A schema parsed from the wire keeps its property order, and the reply
    /// is generated in it (not sorted: `caption` first would leave a reply
    /// that opened with `type` no way to add the other fields).
    #[test]
    fn properties_keep_the_schema_order() {
        let schema: serde_json::Value = serde_json::from_str(
            r#"{"type": "object", "properties": {"type": {"type": "string"},
                "prompt": {"type": "string"}, "caption": {"type": "string"}},
                "required": ["type", "prompt", "caption"]}"#,
        )
        .unwrap();
        let keys: Vec<&String> = schema["properties"].as_object().unwrap().keys().collect();
        assert_eq!(keys, ["type", "prompt", "caption"]);
        let g = || {
            top_level(&TextFormat::JsonSchema {
                schema: schema.clone(),
            })
        };
        let Some(declared) =
            gemma_accepts(g(), r#"{"type": "image", "prompt": "p", "caption": "c"}"#)
        else {
            return;
        };
        assert!(declared, "the schema's order");
        assert_eq!(
            gemma_accepts(g(), r#"{"caption": "c", "prompt": "p", "type": "image"}"#),
            Some(false),
            "sorted order"
        );
    }
}
