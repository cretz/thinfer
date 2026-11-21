//! Tool calling: the tools a text job offers, the calls a reply makes, and
//! how a reply ended. Tools, calls and their arguments are job data: `Debug`
//! redacts them and [`wipe`] zeroes their strings.

use serde_json::Value;

/// One callable tool.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct ToolDef {
    /// `[A-Za-z_][A-Za-z0-9_-]*`, at most 64 characters, unique.
    pub name: String,
    /// What the tool does and when to call it (the model reads this).
    #[serde(default)]
    pub description: String,
    /// JSON Schema of the arguments object (`type: object` with
    /// `properties` / `required`; omitted = no arguments). Supported: type
    /// (incl. lists and `nullable`), enum, const, properties, required,
    /// items, minItems/maxItems, minLength/maxLength, anyOf/oneOf;
    /// integer minimum/maximum spanning at most 256 values; descriptive
    /// keywords (description, title, default, examples, format) are
    /// ignored. Objects are closed (no extra keys).
    #[serde(default = "no_arguments")]
    #[cfg_attr(feature = "utoipa", schema(value_type = Object))]
    pub parameters: Value,
}

fn no_arguments() -> Value {
    serde_json::json!({"type": "object", "properties": {}})
}

/// Whether and how the model may call tools.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub enum ToolChoice {
    /// Reply in prose or call tools: the model decides.
    #[default]
    Auto,
    /// Call at least one tool.
    Required,
    /// Reply in prose (the tools stay visible to the model).
    None,
    /// Call this tool (one or more calls).
    Tool(String),
}

/// A call the model made.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    /// Unique in the conversation: the result answers it (`toolCallId`).
    pub id: String,
    pub name: String,
    /// The arguments object (matches the tool's schema).
    #[cfg_attr(feature = "utoipa", schema(value_type = Object))]
    pub arguments: Value,
}

/// Redacted: names and arguments are job data.
impl std::fmt::Debug for ToolCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ToolCall { .. }")
    }
}

/// Wipe every string in `v` (tool schemas, arguments: job data).
pub fn wipe(v: &mut Value) {
    match v {
        Value::String(s) => zeroize::Zeroize::zeroize(s),
        Value::Array(a) => a.iter_mut().for_each(wipe),
        Value::Object(m) => m.values_mut().for_each(wipe),
        _ => {}
    }
}

/// How a text reply ended.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub enum FinishReason {
    /// The model ended its reply.
    #[default]
    Stop,
    /// The token budget ran out.
    Length,
    /// The reply is tool calls: run them and continue with their results.
    ToolCalls,
}
