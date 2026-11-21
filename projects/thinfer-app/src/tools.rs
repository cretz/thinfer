//! Tool calling. The caller defines tools (a name, a description and a JSON
//! Schema for the arguments); the model answers in prose or with calls. Each
//! model writes calls in its own trained syntax (Gemma 4: `<|tool_call>call:
//! name{key:value}<tool_call|>` with `<|"|>`-quoted strings; Qwen3.6:
//! `<tool_call><function=name><parameter=key>value</parameter>...`), so the
//! definitions compile into a grammar over that syntax: every call names a
//! defined tool and its arguments match that tool's schema, by construction.
//! The calls are parsed back out of the reply.
//!
//! Tools, calls and their arguments are job data: never logged, and the
//! request path's errors about them carry no detail ([`explain`] gives the
//! detail to a local caller).

use serde_json::{Map, Value};

use crate::model::TextModelId;
use crate::text_format::TextFormat;
pub use thinfer_client::tools::{FinishReason, ToolCall, ToolChoice, ToolDef, wipe};

/// Error text for the request path (no job data).
pub const REJECTED: &str = "tool definitions rejected (run them through the local CLI \
                        for the detail)";

/// Whether `err` is the request path's (detail-free) rejection.
pub fn is_rejection(err: &str) -> bool {
    err == REJECTED
}

/// A model's tool-call syntax and the token ids it needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Syntax {
    /// `<|tool_call>call:name{key:value,...}<tool_call|>`; strings between
    /// `<|"|>` tokens. `open`/`close`/`quote` are those special tokens.
    Gemma4 { open: u32, close: u32, quote: u32 },
    /// `<tool_call>\n<function=name>\n<parameter=key>\nvalue\n</parameter>\n
    /// ...</function>\n</tool_call>`; string values raw, others JSON.
    /// `open`/`close` are the `<tool_call>`/`</tool_call>` tokens (added
    /// tokens: the constraint sees them as special).
    Qwen { open: u32, close: u32 },
}

impl Syntax {
    /// The syntax of `model`, with token ids from `id_of` (token text ->
    /// its single id).
    pub fn of(
        model: TextModelId,
        id_of: impl Fn(&str) -> Result<u32, String>,
    ) -> Result<Self, String> {
        Ok(match model {
            TextModelId::Gemma4_26bA4b => Self::Gemma4 {
                open: id_of("<|tool_call>")?,
                close: id_of("<tool_call|>")?,
                quote: id_of("<|\"|>")?,
            },
            TextModelId::Qwen36_35bA3b => Self::Qwen {
                open: id_of("<tool_call>")?,
                close: id_of("</tool_call>")?,
            },
        })
    }

    /// The token that opens the reply's calls (nothing after it is prose).
    pub fn open(self) -> u32 {
        match self {
            Self::Gemma4 { open, .. } | Self::Qwen { open, .. } => open,
        }
    }

    /// What the prompt gets after tool results so the reply starts as a
    /// fresh one would: Gemma's non-thinking replies open with an empty
    /// thought channel (its template's generation prompt writes one for a new
    /// turn, not after results, and the model then writes it itself).
    pub fn after_results(self, think: bool) -> &'static str {
        match (self, think) {
            (Self::Gemma4 { .. }, false) => "<|channel>thought\n<channel|>",
            _ => "",
        }
    }

    /// The text that closes one call (where a continuation after the calls
    /// starts; see [`thinfer_models::llm::chat::continuation`]).
    pub fn call_close(self) -> &'static str {
        match self {
            Self::Gemma4 { .. } => "<tool_call|>",
            Self::Qwen { .. } => "</tool_call>",
        }
    }
}

/// The template's `tools`: OpenAI function-tool objects.
pub fn template_tools(tools: &[ToolDef]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            })
        })
        .collect()
}

/// The Lark grammar for a reply under `tools` / `choice` (prose shaped by
/// `format` when set), with the compiler-level detail on failure (local
/// callers only: it names tools and keywords). `stop` are the model's stop
/// tokens: only the grammar's end admits them.
pub fn grammar(
    syntax: Syntax,
    tools: &[ToolDef],
    choice: &ToolChoice,
    format: Option<&TextFormat>,
    stop: &[u32],
) -> Result<String, String> {
    if tools.is_empty() {
        return Err("tool choice set without tools".into());
    }
    if tools.len() > 64 {
        return Err("at most 64 tools".into());
    }
    let mut names = std::collections::HashSet::new();
    for t in tools {
        if !valid_name(&t.name) {
            return Err(format!(
                "tool name {:?}: use [A-Za-z_][A-Za-z0-9_-]*, at most 64 characters",
                t.name
            ));
        }
        if !names.insert(t.name.as_str()) {
            return Err(format!("tool {:?} is defined twice", t.name));
        }
    }
    let callable: Vec<&ToolDef> = match choice {
        ToolChoice::Tool(name) => {
            let t = tools
                .iter()
                .find(|t| &t.name == name)
                .ok_or_else(|| format!("tool choice names an undefined tool {name:?}"))?;
            vec![t]
        }
        _ => tools.iter().collect(),
    };
    let mut g = Grammar::new(syntax, stop);
    // Prose may lead into calls: a model writes its own markup (Gemma's
    // thought channel) or a line of reasoning (Qwen's template invites it)
    // before calling.
    let prose = g.text();
    let reply = match format {
        None => prose.clone(),
        Some(TextFormat::JsonSchema { schema }) => {
            g.add(format!("%json {}", serde_json::to_string(schema).unwrap()))
        }
        Some(TextFormat::Lark { .. }) => {
            return Err("a Lark reply format cannot be combined with tools".into());
        }
    };
    let mut fns = Vec::new();
    for t in &callable {
        fns.push(
            g.function(t)
                .map_err(|e| format!("tool {:?}: {e}", t.name))?,
        );
    }
    let calls = g.calls(&fns.join(" | "));
    let start = match (choice, format.is_some()) {
        (ToolChoice::None, _) => reply,
        (ToolChoice::Auto, false) => format!("{prose} {calls}?"),
        (ToolChoice::Auto, true) => format!("{reply} | {prose} {calls}"),
        (ToolChoice::Required | ToolChoice::Tool(_), _) => format!("{prose} {calls}"),
    };
    Ok(g.finish(&start))
}

/// The compiler's detail for tools `model` rejected (local callers only).
pub fn explain(
    model: TextModelId,
    tools: &[ToolDef],
    choice: &ToolChoice,
    format: Option<&TextFormat>,
) -> String {
    // The detail does not depend on token ids.
    let syntax = match model {
        TextModelId::Gemma4_26bA4b => Syntax::Gemma4 {
            open: 0,
            close: 0,
            quote: 0,
        },
        TextModelId::Qwen36_35bA3b => Syntax::Qwen { open: 0, close: 0 },
    };
    match grammar(syntax, tools, choice, format, &[]) {
        Ok(_) => REJECTED.to_string(),
        Err(e) => e,
    }
}

/// The request path's grammar: [`grammar`] with the detail dropped.
pub fn request_grammar(
    syntax: Syntax,
    tools: &[ToolDef],
    choice: &ToolChoice,
    format: Option<&TextFormat>,
    stop: &[u32],
) -> Result<String, String> {
    grammar(syntax, tools, choice, format, stop).map_err(|_| REJECTED.to_string())
}

fn valid_name(s: &str) -> bool {
    let mut c = s.chars();
    !s.is_empty()
        && s.len() <= 64
        && c.next()
            .is_some_and(|f| f.is_ascii_alphabetic() || f == '_')
        && c.all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
}

fn valid_key(s: &str) -> bool {
    let mut c = s.chars();
    !s.is_empty()
        && s.len() <= 64
        && c.next()
            .is_some_and(|f| f.is_ascii_alphabetic() || f == '_')
        && c.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

/// A JSON string as a Lark literal (Lark strings use JSON syntax).
fn lit(s: &str) -> String {
    serde_json::to_string(s).unwrap()
}

/// Keywords that describe but do not constrain.
const IGNORED: &[&str] = &[
    "description",
    "title",
    "default",
    "examples",
    "format",
    "$schema",
    "$comment",
    "deprecated",
    "readOnly",
    "writeOnly",
];
const HANDLED: &[&str] = &[
    "type",
    "enum",
    "const",
    "properties",
    "required",
    "items",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "anyOf",
    "oneOf",
    "nullable",
    "minimum",
    "maximum",
    "additionalProperties",
];

/// Most array items a min/max count unrolls to.
const MAX_UNROLL: u64 = 32;
/// Most integers a minimum/maximum range enumerates.
const MAX_INT_RANGE: i64 = 256;

/// Lark rules under construction.
struct Grammar {
    syntax: Syntax,
    /// Token ids prose excludes: the call markers and the stop tokens.
    not_prose: Vec<u32>,
    rules: Vec<String>,
    n: usize,
    any: Option<String>,
}

impl Grammar {
    fn new(syntax: Syntax, stop: &[u32]) -> Self {
        let (open, close) = match syntax {
            Syntax::Gemma4 { open, close, .. } | Syntax::Qwen { open, close } => (open, close),
        };
        let mut not_prose = vec![open, close];
        not_prose.extend_from_slice(stop);
        not_prose.sort_unstable();
        not_prose.dedup();
        Self {
            syntax,
            not_prose,
            rules: Vec::new(),
            n: 0,
            any: None,
        }
    }

    /// A new rule `body`; returns its name.
    fn add(&mut self, body: String) -> String {
        self.n += 1;
        let name = format!("r{}", self.n);
        self.rules.push(format!("{name}: {body}"));
        name
    }

    /// A new terminal; returns its name.
    fn terminal(&mut self, body: String) -> String {
        self.n += 1;
        let name = format!("T{}", self.n);
        self.rules.push(format!("{name}: {body}"));
        name
    }

    fn finish(self, start: &str) -> String {
        let mut out = format!("start: {start}\n");
        for r in self.rules {
            out.push_str(&r);
            out.push('\n');
        }
        out
    }

    /// Free prose: any tokens but the call markers and the stop tokens (the
    /// reply ends where the grammar allows it). Token-level, not text: the
    /// model's own markup in a reply (Gemma's thought channel, say) stays
    /// open to it.
    fn text(&mut self) -> String {
        let ids: Vec<String> = self.not_prose.iter().map(u32::to_string).collect();
        self.add(format!("<[^{}]>*", ids.join(",")))
    }

    /// One or more calls, each one of `fns`.
    fn calls(&mut self, fns: &str) -> String {
        match self.syntax {
            Syntax::Gemma4 { open, close, .. } => {
                let one = self.add(format!("<[{open}]> \"call:\" ({fns}) <[{close}]>"));
                self.add(format!("{one}+"))
            }
            Syntax::Qwen { open, close } => {
                let one = self.add(format!(
                    "<[{open}]> \"\\n<function=\" ({fns}) \"</function>\\n\" <[{close}]>"
                ));
                self.add(format!("{one} (\"\\n\" {one})*"))
            }
        }
    }

    /// One tool's name and arguments (inside the call markup).
    fn function(&mut self, t: &ToolDef) -> Result<String, String> {
        let p = &t.parameters;
        let obj = p
            .as_object()
            .ok_or("parameters must be a JSON Schema object")?;
        check_keywords(obj)?;
        if obj.get("type").is_some_and(|ty| ty != "object") {
            return Err("parameters must have type object".into());
        }
        let props = properties(obj)?;
        match self.syntax {
            Syntax::Gemma4 { .. } => {
                let body = self.gemma_members(&props)?;
                Ok(self.add(format!("{} \"{{\" {body} \"}}\"", lit(&t.name))))
            }
            Syntax::Qwen { .. } => {
                let mut parts = Vec::new();
                for (key, schema, required) in &props {
                    let v = self.qwen_param(schema).map_err(|e| format!("{key}: {e}"))?;
                    let item = format!("{} {v}", lit(&format!("<parameter={key}>\n")));
                    parts.push(if *required {
                        item
                    } else {
                        format!("({item})?")
                    });
                }
                Ok(self.add(format!(
                    "{} {}",
                    lit(&format!("{}>\n", t.name)),
                    parts.join(" ")
                )))
            }
        }
    }

    // ---- Gemma 4 values ---------------------------------------------------

    /// `key:value` members of an object in key order (the template's
    /// `dictsort`), commas between the present ones. Returns an expression
    /// (possibly empty: `""`-free, so `(...)?` forms).
    fn gemma_members(&mut self, props: &[(String, &Value, bool)]) -> Result<String, String> {
        // From the end: `first` = members from i on with none before,
        // `rest` = members from i on after an earlier one.
        let (mut first, mut rest): (Option<String>, Option<String>) = (None, None);
        for (key, schema, required) in props.iter().rev() {
            let v = self
                .gemma_value(schema)
                .map_err(|e| format!("{key}: {e}"))?;
            let member = format!("{} {v}", lit(&format!("{key}:")));
            let tail = rest.clone().map(|r| format!(" {r}")).unwrap_or_default();
            let with_first = format!("{member}{tail}");
            let with_rest = format!("\",\" {member}{tail}");
            let (f, r) = if *required {
                (with_first, with_rest)
            } else {
                (
                    alt(Some(with_first), first.clone()),
                    alt(Some(with_rest), rest.clone()),
                )
            };
            first = Some(self.add(f));
            rest = Some(self.add(r));
        }
        Ok(first.unwrap_or_default())
    }

    fn quoted(&self, inner: &str) -> String {
        let Syntax::Gemma4 { quote, .. } = self.syntax else {
            unreachable!("gemma value outside the gemma syntax")
        };
        format!("<[{quote}]> {inner} <[{quote}]>")
    }

    /// A literal value in Gemma's argument syntax.
    fn gemma_literal(&self, v: &Value) -> Result<String, String> {
        Ok(match v {
            Value::String(s) if s.is_empty() => self.quoted(""),
            Value::String(s) => self.quoted(&lit(s)),
            Value::Number(n) => lit(&n.to_string()),
            Value::Bool(b) => lit(&b.to_string()),
            Value::Null => lit("null"),
            _ => return Err("enum/const values must be scalars".into()),
        })
    }

    fn gemma_value(&mut self, schema: &Value) -> Result<String, String> {
        let Some(obj) = schema.as_object() else {
            return match schema {
                Value::Bool(true) => Ok(self.gemma_any()),
                _ => Err("a schema must be an object".into()),
            };
        };
        check_keywords(obj)?;
        if let Some(c) = obj.get("const") {
            return Ok(self.add(self.gemma_literal(c)?));
        }
        if let Some(e) = obj.get("enum") {
            let alts = e
                .as_array()
                .ok_or("enum must be a list")?
                .iter()
                .map(|v| self.gemma_literal(v))
                .collect::<Result<Vec<_>, _>>()?;
            if alts.is_empty() {
                return Err("enum is empty".into());
            }
            return Ok(self.add(alts.join(" | ")));
        }
        if let Some(subs) = obj.get("anyOf").or_else(|| obj.get("oneOf")) {
            let alts = subs
                .as_array()
                .ok_or("anyOf/oneOf must be a list")?
                .iter()
                .map(|s| self.gemma_value(s))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(self.add(alts.join(" | ")));
        }
        let types = types(obj)?;
        if types.is_empty() {
            return Ok(self.gemma_any());
        }
        let mut alts = Vec::new();
        for ty in types {
            alts.push(match ty.as_str() {
                "string" => {
                    let re = string_regex(obj)?;
                    let t = self.terminal(re);
                    self.quoted(&t)
                }
                "integer" => self.integer(obj)?,
                "number" => {
                    if obj.contains_key("minimum") || obj.contains_key("maximum") {
                        return Err("number bounds are not supported (use an integer \
                                    range or describe them)"
                            .into());
                    }
                    self.terminal(NUMBER.into())
                }
                "boolean" => "(\"true\" | \"false\")".into(),
                "null" => lit("null"),
                "array" => {
                    let item = match obj.get("items") {
                        Some(s) => self.gemma_value(s)?,
                        None => self.gemma_any(),
                    };
                    self.array(obj, &item)?
                }
                "object" => {
                    if obj.get("properties").is_none() {
                        self.gemma_any_object()
                    } else {
                        let props = properties(obj)?;
                        let body = self.gemma_members(&props)?;
                        format!("\"{{\" {body} \"}}\"")
                    }
                }
                other => return Err(format!("unknown type {other:?}")),
            });
        }
        Ok(self.add(alts.join(" | ")))
    }

    /// Any value (an unconstrained schema).
    fn gemma_any(&mut self) -> String {
        if let Some(a) = &self.any {
            return a.clone();
        }
        self.n += 1;
        let name = format!("r{}", self.n);
        self.any = Some(name.clone());
        let s = self.terminal(r"/(?s:.*)/".into());
        let num = self.terminal(NUMBER.into());
        let key = self.terminal(r"/[A-Za-z_][A-Za-z0-9_]*/".into());
        let quoted = self.quoted(&s);
        self.rules.push(format!(
            "{name}: {quoted} | {num} | \"true\" | \"false\" | \"null\" \
             | \"[\" ({name} (\",\" {name})*)? \"]\" \
             | \"{{\" ({key} \":\" {name} (\",\" {key} \":\" {name})*)? \"}}\""
        ));
        name
    }

    fn gemma_any_object(&mut self) -> String {
        let any = self.gemma_any();
        let key = self.terminal(r"/[A-Za-z_][A-Za-z0-9_]*/".into());
        format!("\"{{\" ({key} \":\" {any} (\",\" {key} \":\" {any})*)? \"}}\"")
    }

    /// An integer: a literal per value of a small bounded range, else any.
    fn integer(&mut self, obj: &Map<String, Value>) -> Result<String, String> {
        let (lo, hi) = (
            obj.get("minimum").and_then(Value::as_i64),
            obj.get("maximum").and_then(Value::as_i64),
        );
        match (lo, hi) {
            (None, None) => Ok(self.terminal(INTEGER.into())),
            (Some(lo), Some(hi)) if hi >= lo && hi - lo < MAX_INT_RANGE => Ok(format!(
                "({})",
                (lo..=hi)
                    .map(|i| lit(&i.to_string()))
                    .collect::<Vec<_>>()
                    .join(" | ")
            )),
            _ => Err(format!(
                "integer bounds must give both minimum and maximum, at most \
                 {MAX_INT_RANGE} values apart"
            )),
        }
    }

    /// `[item, ...]` with the schema's item-count bounds.
    fn array(&mut self, obj: &Map<String, Value>, item: &str) -> Result<String, String> {
        let min = obj.get("minItems").and_then(Value::as_u64).unwrap_or(0);
        let max = obj.get("maxItems").and_then(Value::as_u64);
        if min > MAX_UNROLL || max.is_some_and(|m| m > MAX_UNROLL || m < min) {
            return Err(format!(
                "minItems/maxItems must be ordered and at most {MAX_UNROLL}"
            ));
        }
        let more = format!("\",\" {item}");
        let body = match (min, max) {
            (_, Some(0)) => String::new(),
            (0, None) => format!("({item} ({more})*)?"),
            (m, None) => {
                let mut s = item.to_string();
                for _ in 1..m {
                    s += &format!(" {more}");
                }
                format!("{s} ({more})*")
            }
            (m, Some(x)) => {
                // `m` required, then `x - m` optional, nested.
                let mut opt = String::new();
                for _ in m.max(1)..x {
                    opt = format!("({more} {opt})?");
                }
                if m == 0 {
                    format!("({item} {opt})?")
                } else {
                    let mut s = item.to_string();
                    for _ in 1..m {
                        s += &format!(" {more}");
                    }
                    format!("{s} {opt}")
                }
            }
        };
        Ok(format!("\"[\" {body} \"]\""))
    }

    // ---- Qwen values --------------------------------------------------------

    /// One parameter's value and its closing markup. Strings are raw text
    /// (the template prints them as is), everything else JSON.
    fn qwen_param(&mut self, schema: &Value) -> Result<String, String> {
        const END: &str = "\n</parameter>\n";
        let obj = schema.as_object();
        if let Some(obj) = obj {
            check_keywords(obj)?;
        }
        let is_string = obj.is_some_and(|o| {
            let t = types(o).unwrap_or_default();
            let strings = o
                .get("enum")
                .and_then(Value::as_array)
                .is_some_and(|e| !e.is_empty() && e.iter().all(Value::is_string))
                || o.get("const").is_some_and(Value::is_string);
            strings || (!t.is_empty() && t.iter().all(|x| x == "string" || x == "null"))
        });
        if !is_string {
            return Ok(self.add(format!(
                "%json {} {}",
                serde_json::to_string(schema).unwrap(),
                lit(END)
            )));
        }
        let obj = obj.expect("string schemas are objects");
        let values: Option<Vec<&str>> = obj
            .get("enum")
            .and_then(Value::as_array)
            .map(|e| e.iter().filter_map(Value::as_str).collect())
            .or_else(|| obj.get("const").and_then(Value::as_str).map(|c| vec![c]));
        if let Some(values) = values {
            let alts: Vec<String> = values.iter().map(|v| lit(&format!("{v}{END}"))).collect();
            return Ok(self.add(format!("({})", alts.join(" | "))));
        }
        // Raw text up to the first closing tag (lazy: the tag ends it).
        let body = string_regex(obj)?;
        let inner = &body[1..body.len() - 1];
        self.n += 1;
        let name = format!("r{}", self.n);
        self.rules
            .push(format!(r"{name}[lazy]: /{inner}\n<\/parameter>\n/"));
        Ok(name)
    }
}

const INTEGER: &str = r"/-?(0|[1-9][0-9]*)/";
const NUMBER: &str = r"/-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?/";

/// `a | b` where either side may be empty (`None`).
fn alt(a: Option<String>, b: Option<String>) -> String {
    match (a, b) {
        (Some(a), Some(b)) => format!("({a}) | ({b})"),
        (Some(x), None) | (None, Some(x)) => format!("({x})?"),
        (None, None) => unreachable!("alt of two empty sides"),
    }
}

/// A string's Lark regex (`/.../`) from its length bounds.
fn string_regex(obj: &Map<String, Value>) -> Result<String, String> {
    let min = obj.get("minLength").and_then(Value::as_u64);
    let max = obj.get("maxLength").and_then(Value::as_u64);
    Ok(match (min, max) {
        (None, None) => r"/(?s:.*)/".into(),
        (lo, hi) => {
            let lo = lo.unwrap_or(0);
            if hi.is_some_and(|h| h < lo) || lo > 10_000 || hi.is_some_and(|h| h > 10_000) {
                return Err("minLength/maxLength must be ordered and at most 10000".into());
            }
            match hi {
                Some(h) => format!(r"/(?s:.{{{lo},{h}}})/"),
                None => format!(r"/(?s:.{{{lo},}})/"),
            }
        }
    })
}

/// Reject keywords the grammar cannot enforce (so a call never violates its
/// schema silently).
fn check_keywords(obj: &Map<String, Value>) -> Result<(), String> {
    for k in obj.keys() {
        if !HANDLED.contains(&k.as_str()) && !IGNORED.contains(&k.as_str()) {
            return Err(format!("schema keyword {k:?} is not supported"));
        }
    }
    if let Some(ap) = obj.get("additionalProperties")
        && ap != &Value::Bool(false)
    {
        return Err("additionalProperties must be false or absent (objects are closed)".into());
    }
    Ok(())
}

/// The schema's `type`s (a string or list, plus `nullable`).
fn types(obj: &Map<String, Value>) -> Result<Vec<String>, String> {
    let mut t: Vec<String> = match obj.get("type") {
        None => Vec::new(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or("type list of non-strings")
            })
            .collect::<Result<_, _>>()?,
        Some(_) => return Err("type must be a string or a list".into()),
    };
    if obj.get("nullable") == Some(&Value::Bool(true)) && !t.iter().any(|x| x == "null") {
        t.push("null".into());
    }
    if t.is_empty() && obj.contains_key("properties") {
        t.push("object".into());
    }
    Ok(t)
}

/// An object schema's `(key, schema, required)` in key order.
fn properties(obj: &Map<String, Value>) -> Result<Vec<(String, &Value, bool)>, String> {
    let required: Vec<&str> = match obj.get("required") {
        None => Vec::new(),
        Some(r) => r
            .as_array()
            .ok_or("required must be a list")?
            .iter()
            .map(|v| v.as_str().ok_or("required lists property names"))
            .collect::<Result<_, _>>()?,
    };
    let props = match obj.get("properties") {
        None => return Ok(Vec::new()),
        Some(p) => p.as_object().ok_or("properties must be an object")?,
    };
    for r in &required {
        if !props.contains_key(*r) {
            return Err(format!("required property {r:?} is not defined"));
        }
    }
    let mut out: Vec<(String, &Value, bool)> = props
        .iter()
        .map(|(k, v)| (k.clone(), v, required.contains(&k.as_str())))
        .collect();
    for (k, ..) in &out {
        if !valid_key(k) {
            return Err(format!(
                "property name {k:?}: use [A-Za-z_][A-Za-z0-9_]*, at most 64 characters"
            ));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

// ---- Parsing calls back -----------------------------------------------------

/// The calls in `text`: the reply from the syntax's opening token on, decoded
/// with special tokens kept. Ids are `call_<id_base>_<index>`.
pub fn parse_calls(
    syntax: Syntax,
    tools: &[ToolDef],
    text: &str,
    id_base: u64,
) -> Result<Vec<ToolCall>, String> {
    let calls = match syntax {
        Syntax::Gemma4 { .. } => parse_gemma(text)?,
        Syntax::Qwen { .. } => parse_qwen(tools, text)?,
    };
    if calls.is_empty() {
        return Err("tool calls: none found".into());
    }
    Ok(calls
        .into_iter()
        .enumerate()
        .map(|(i, (name, arguments))| ToolCall {
            id: format!("call_{id_base:x}_{i}"),
            name,
            arguments,
        })
        .collect())
}

const BAD: &str = "tool calls: unexpected markup";

fn parse_gemma(text: &str) -> Result<Vec<(String, Value)>, String> {
    const OPEN: &str = "<|tool_call>call:";
    const CLOSE: &str = "<tool_call|>";
    let mut out = Vec::new();
    let mut rest = text.trim_start();
    while !rest.is_empty() {
        rest = rest.strip_prefix(OPEN).ok_or(BAD)?;
        let brace = rest.find('{').ok_or(BAD)?;
        let name = rest[..brace].to_string();
        let mut p = GemmaParser {
            s: &rest[brace..],
            bare_keys: true,
        };
        let args = p.value()?;
        rest = p.s.strip_prefix(CLOSE).ok_or(BAD)?.trim_start();
        out.push((name, args));
    }
    Ok(out)
}

/// Gemma's argument syntax: JSON-like, keys bare (or `<|"|>`-quoted),
/// strings between `<|"|>` markers.
struct GemmaParser<'a> {
    s: &'a str,
    bare_keys: bool,
}

impl GemmaParser<'_> {
    const Q: &'static str = "<|\"|>";

    fn eat(&mut self, p: &str) -> bool {
        match self.s.strip_prefix(p) {
            Some(r) => {
                self.s = r;
                true
            }
            None => false,
        }
    }

    fn string(&mut self) -> Result<String, String> {
        let end = self.s.find(Self::Q).ok_or(BAD)?;
        let v = self.s[..end].to_string();
        self.s = &self.s[end + Self::Q.len()..];
        Ok(v)
    }

    fn value(&mut self) -> Result<Value, String> {
        if self.eat(Self::Q) {
            return Ok(Value::String(self.string()?));
        }
        if self.eat("{") {
            let mut m = Map::new();
            if self.eat("}") {
                return Ok(Value::Object(m));
            }
            loop {
                let key = if self.eat(Self::Q) {
                    self.string()?
                } else if self.bare_keys {
                    let end = self.s.find(':').ok_or(BAD)?;
                    let k = self.s[..end].to_string();
                    self.s = &self.s[end..];
                    k
                } else {
                    return Err(BAD.into());
                };
                if !self.eat(":") {
                    return Err(BAD.into());
                }
                let v = self.value()?;
                m.insert(key, v);
                if self.eat("}") {
                    return Ok(Value::Object(m));
                }
                if !self.eat(",") {
                    return Err(BAD.into());
                }
            }
        }
        if self.eat("[") {
            let mut a = Vec::new();
            if self.eat("]") {
                return Ok(Value::Array(a));
            }
            loop {
                a.push(self.value()?);
                if self.eat("]") {
                    return Ok(Value::Array(a));
                }
                if !self.eat(",") {
                    return Err(BAD.into());
                }
            }
        }
        for (word, v) in [
            ("true", Value::Bool(true)),
            ("false", Value::Bool(false)),
            ("null", Value::Null),
        ] {
            if self.eat(word) {
                return Ok(v);
            }
        }
        let end = self
            .s
            .find(|c: char| !(c.is_ascii_digit() || "+-.eE".contains(c)))
            .unwrap_or(self.s.len());
        let n: serde_json::Number = self.s[..end].parse().map_err(|_| BAD)?;
        self.s = &self.s[end..];
        Ok(Value::Number(n))
    }
}

fn parse_qwen(tools: &[ToolDef], text: &str) -> Result<Vec<(String, Value)>, String> {
    const OPEN: &str = "<tool_call>\n<function=";
    const END_FN: &str = "</function>\n</tool_call>";
    const OPEN_P: &str = "<parameter=";
    const END_P: &str = "\n</parameter>\n";
    let mut out = Vec::new();
    let mut rest = text.trim_start();
    while !rest.is_empty() {
        rest = rest.strip_prefix(OPEN).ok_or(BAD)?;
        let gt = rest.find(">\n").ok_or(BAD)?;
        let name = rest[..gt].to_string();
        rest = &rest[gt + 2..];
        let schema = tools
            .iter()
            .find(|t| t.name == name)
            .map(|t| &t.parameters)
            .ok_or(BAD)?;
        let mut args = Map::new();
        while let Some(r) = rest.strip_prefix(OPEN_P) {
            let gt = r.find(">\n").ok_or(BAD)?;
            let key = r[..gt].to_string();
            let r = &r[gt + 2..];
            let end = r.find(END_P).ok_or(BAD)?;
            let raw = &r[..end];
            rest = &r[end + END_P.len()..];
            let prop = schema.get("properties").and_then(|p| p.get(&key));
            let is_string = prop.and_then(|p| p.as_object()).is_some_and(|o| {
                types(o).unwrap_or_default().iter().any(|t| t == "string")
                    || o.get("enum")
                        .and_then(Value::as_array)
                        .is_some_and(|e| e.iter().all(Value::is_string))
                    || o.get("const").is_some_and(Value::is_string)
            });
            let v = if is_string {
                Value::String(raw.to_string())
            } else {
                serde_json::from_str(raw).map_err(|_| BAD)?
            };
            args.insert(key, v);
        }
        rest = rest.strip_prefix(END_FN).ok_or(BAD)?;
        rest = rest.strip_prefix('\n').unwrap_or(rest);
        out.push((name, Value::Object(args)));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TextModelEngine;

    fn tools() -> Vec<ToolDef> {
        vec![
            ToolDef {
                name: "generate_image".into(),
                description: "Paint a scene".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "what to paint"},
                        "style": {"enum": ["anime", "photo"]},
                        "count": {"type": "integer", "minimum": 1, "maximum": 4},
                        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
                        "hd": {"type": "boolean"}
                    },
                    "required": ["prompt", "style"]
                }),
            },
            ToolDef {
                name: "wait".into(),
                description: String::new(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            },
        ]
    }

    #[test]
    fn rejects_what_it_cannot_enforce() {
        let syn = Syntax::Qwen { open: 1, close: 2 };
        let mut t = tools();
        t[0].parameters["properties"]["prompt"]["pattern"] = "^a".into();
        let e = grammar(syn, &t, &ToolChoice::Auto, None, &[]).unwrap_err();
        assert!(e.contains("pattern"), "{e}");
        let mut t = tools();
        t[1].name = "bad name".into();
        assert!(grammar(syn, &t, &ToolChoice::Auto, None, &[]).is_err());
        let t = tools();
        assert!(grammar(syn, &t, &ToolChoice::Tool("nope".into()), None, &[]).is_err());
        // The request path's error names nothing.
        let e = request_grammar(syn, &[], &ToolChoice::Auto, None, &[]).unwrap_err();
        assert!(is_rejection(&e));
    }

    #[test]
    fn parses_gemma_calls() {
        let text = "<|tool_call>call:generate_image{count:2,hd:true,prompt:<|\"|>a red fox, snow<|\"|>,style:<|\"|>photo<|\"|>,tags:[<|\"|>winter<|\"|>]}<tool_call|><|tool_call>call:wait{}<tool_call|>";
        let calls = parse_calls(
            Syntax::Gemma4 {
                open: 0,
                close: 0,
                quote: 0,
            },
            &tools(),
            text,
            7,
        )
        .unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "generate_image");
        assert_eq!(
            calls[0].arguments,
            serde_json::json!({"count": 2, "hd": true, "prompt": "a red fox, snow",
                "style": "photo", "tags": ["winter"]})
        );
        assert_eq!(calls[1].arguments, serde_json::json!({}));
        assert_eq!(calls[0].id, "call_7_0");
    }

    /// Feed `text` (tokenized by the model's own tokenizer) through the
    /// compiled grammar: `Ok` when every token is allowed and the reply may
    /// end there, else the byte offset where it was refused.
    fn run_grammar(
        model: TextModelId,
        choice: &ToolChoice,
        text: &str,
    ) -> Option<Result<(), String>> {
        use thinfer_core::tokenizer::Tokenizer;
        use thinfer_models::llm::manifest::role;
        let m = model.manifest();
        let (Ok(tj), Ok(gc)) = (
            crate::download::resolve_role(m, role::TOKENIZER),
            crate::download::resolve_role(m, role::GENERATION_CONFIG),
        ) else {
            eprintln!("skipped: {model} tokenizer not cached");
            return None;
        };
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let tok = rt
            .block_on(thinfer_native::tokenizer::HfTokenizer::from_path(&tj))
            .unwrap();
        let gen_cfg: Value = serde_json::from_str(&std::fs::read_to_string(gc).unwrap()).unwrap();
        let stop: Vec<u32> = match &gen_cfg["eos_token_id"] {
            Value::Array(a) => a.iter().map(|v| v.as_u64().unwrap() as u32).collect(),
            v => vec![v.as_u64().unwrap() as u32],
        };
        let id = |t: &str| match tok.encode(t, false).as_deref() {
            Ok([i]) => Ok(*i),
            _ => Err(format!("{t} not one token")),
        };
        let syntax = Syntax::of(model, id).unwrap();
        let src = grammar(syntax, &tools(), choice, None, &stop).unwrap();
        let env = std::sync::Arc::new(crate::text_format::tok_env(&tj, None, &stop).unwrap());
        let mut f = llguidance::ParserFactory::new_simple(&env).unwrap();
        f.set_stderr_log_level(0).set_buffer_log_level(0);
        let parser = f
            .create_parser(llguidance::api::TopLevelGrammar::from_lark(src.clone()))
            .unwrap_or_else(|e| {
                panic!(
                    "{e}
{src}"
                )
            });
        // Token by token, no jump-forward: the grammar alone decides.
        let mut m = llguidance::Matcher::new(Ok(parser));
        let ids = tok.encode(text, false).unwrap();
        for (i, &t) in ids.iter().enumerate() {
            let allowed = m.compute_mask_or_eos().unwrap();
            if !allowed.is_allowed(t) {
                return Some(Err(format!(
                    "refused after {:?}",
                    tok.decode(&ids[..i], false).unwrap()
                )));
            }
            m.consume_token(t).unwrap();
        }
        let allowed = m.compute_mask_or_eos().unwrap();
        if m.is_stopped() || stop.iter().any(|&s| allowed.is_allowed(s)) {
            Some(Ok(()))
        } else {
            Some(Err("cannot end here".into()))
        }
    }

    fn expect(model: TextModelId, choice: &ToolChoice, text: &str, ok: bool) {
        if let Some(r) = run_grammar(model, choice, text) {
            assert_eq!(r.is_ok(), ok, "{model} {choice:?} {text:?}: {r:?}");
        }
    }

    #[test]
    fn gemma_grammar_admits_exactly_valid_calls() {
        let m = TextModelId::Gemma4_26bA4b;
        let q = |s: &str| format!("<|\"|>{s}<|\"|>");
        let call = |args: &str| format!("<|tool_call>call:generate_image{{{args}}}<tool_call|>");
        let good = call(&format!(
            "count:3,hd:false,prompt:{},style:{},tags:[{},{}]",
            q("a fox, in snow"),
            q("photo"),
            q("winter"),
            q("night")
        ));
        expect(m, &ToolChoice::Auto, &good, true);
        expect(m, &ToolChoice::Required, &good, true);
        // Optional members may go; two calls in a row.
        let minimal = call(&format!("prompt:{},style:{}", q("x"), q("anime")));
        expect(
            m,
            &ToolChoice::Auto,
            &format!("{minimal}<|tool_call>call:wait{{}}<tool_call|>"),
            true,
        );
        // Prose: fine under auto, not under required.
        expect(m, &ToolChoice::Auto, "Sure, here is a story.", true);
        expect(m, &ToolChoice::Required, "Sure, here is a story.", false);
        // Missing a required member, a bad enum, out of range, unknown tool,
        // members out of key order, too many items.
        expect(
            m,
            &ToolChoice::Auto,
            &call(&format!("style:{}", q("photo"))),
            false,
        );
        expect(
            m,
            &ToolChoice::Auto,
            &call(&format!("prompt:{},style:{}", q("x"), q("oil"))),
            false,
        );
        expect(
            m,
            &ToolChoice::Auto,
            &call(&format!("count:9,prompt:{},style:{}", q("x"), q("photo"))),
            false,
        );
        expect(
            m,
            &ToolChoice::Auto,
            "<|tool_call>call:fly{}<tool_call|>",
            false,
        );
        expect(
            m,
            &ToolChoice::Auto,
            &call(&format!("style:{},prompt:{}", q("photo"), q("x"))),
            false,
        );
        expect(
            m,
            &ToolChoice::Auto,
            &call(&format!(
                "prompt:{},style:{},tags:[{},{},{},{}]",
                q("x"),
                q("photo"),
                q("a"),
                q("b"),
                q("c"),
                q("d")
            )),
            false,
        );
        // The model's thought channel before a call.
        expect(
            m,
            &ToolChoice::Required,
            &format!("<|channel>thought\n<channel|>{minimal}"),
            true,
        );
        // A named tool only.
        expect(m, &ToolChoice::Tool("wait".into()), &minimal, false);
        expect(
            m,
            &ToolChoice::Tool("wait".into()),
            "<|tool_call>call:wait{}<tool_call|>",
            true,
        );
    }

    #[test]
    fn qwen_grammar_admits_exactly_valid_calls() {
        let m = TextModelId::Qwen36_35bA3b;
        let p = |k: &str, v: &str| {
            format!(
                "<parameter={k}>
{v}
</parameter>
"
            )
        };
        let call = |name: &str, params: &str| {
            format!(
                "<tool_call>
<function={name}>
{params}</function>
</tool_call>"
            )
        };
        let good = call(
            "generate_image",
            &[
                p("count", "2"),
                p("hd", "true"),
                p(
                    "prompt",
                    "a fox
in the snow",
                ),
                p("style", "anime"),
                p("tags", "[\"winter\"]"),
            ]
            .concat(),
        );
        expect(m, &ToolChoice::Auto, &good, true);
        let minimal = call(
            "generate_image",
            &[p("prompt", "x"), p("style", "photo")].concat(),
        );
        expect(
            m,
            &ToolChoice::Required,
            &format!(
                "{minimal}
{}",
                call("wait", "")
            ),
            true,
        );
        expect(m, &ToolChoice::Auto, "Here you go.", true);
        expect(m, &ToolChoice::Required, "Here you go.", false);
        // A line of reasoning before the call (the template's format).
        expect(
            m,
            &ToolChoice::Required,
            &format!("Let me render it.\n\n{minimal}"),
            true,
        );
        expect(
            m,
            &ToolChoice::Auto,
            &call("generate_image", &p("style", "photo")),
            false,
        );
        expect(
            m,
            &ToolChoice::Auto,
            &call(
                "generate_image",
                &[p("prompt", "x"), p("style", "oil")].concat(),
            ),
            false,
        );
        expect(
            m,
            &ToolChoice::Auto,
            &call(
                "generate_image",
                &[p("count", "7"), p("prompt", "x"), p("style", "photo")].concat(),
            ),
            false,
        );
        expect(m, &ToolChoice::Auto, &call("fly", ""), false);
    }

    /// The continuation after a reply of tool calls, rendered by the
    /// model's official template, is exactly what a full render places after
    /// the calls (expected strings from the template under Python Jinja2).
    #[test]
    fn tool_result_continuation_matches_the_official_templates() {
        use thinfer_models::llm::chat::{ChatMessage, continuation};
        use thinfer_models::llm::manifest::role;
        let result = [ChatMessage::tool_result(
            "call_1_0",
            "get_weather",
            "{\"temp\": 7}",
        )];
        for (model, close, want) in [
            (
                TextModelId::Gemma4_26bA4b,
                "<tool_call|>",
                "<|tool_response>response:get_weather{value:<|\"|>{\"temp\": 7}<|\"|>}<tool_response|>",
            ),
            (
                TextModelId::Qwen36_35bA3b,
                "</tool_call>",
                "<|im_end|>\n<|im_start|>user\n<tool_response>\n{\"temp\": 7}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            ),
        ] {
            let Ok(t) = crate::download::resolve_role(model.manifest(), role::CHAT_TEMPLATE) else {
                eprintln!("skipped: {model} template not cached");
                continue;
            };
            let t = std::fs::read_to_string(t).unwrap();
            let got = continuation(&t, &result, &[], close, false, "").unwrap();
            assert_eq!(got, want, "{model}");
        }
    }

    #[test]
    fn parses_qwen_calls() {
        let text = "<tool_call>\n<function=generate_image>\n<parameter=count>\n2\n</parameter>\n<parameter=prompt>\na fox\nin snow\n</parameter>\n<parameter=style>\nanime\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=wait>\n</function>\n</tool_call>";
        let calls = parse_calls(Syntax::Qwen { open: 0, close: 0 }, &tools(), text, 1).unwrap();
        assert_eq!(
            calls[0].arguments,
            serde_json::json!({"count": 2, "prompt": "a fox\nin snow", "style": "anime"})
        );
        assert_eq!(calls[1].name, "wait");
    }
}
