//! Constrained text output: the shape a reply must take.

/// The shape the reply must take.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum TextFormat {
    /// A JSON Schema (the llguidance subset: types, properties, required,
    /// enum/const, items, anyOf/oneOf, $ref/$defs, string patterns/lengths,
    /// numeric bounds).
    JsonSchema {
        #[cfg_attr(feature = "utoipa", schema(value_type = Object))]
        schema: serde_json::Value,
    },
    /// A Lark grammar (llguidance dialect; `start` is the entry rule).
    Lark { grammar: String },
}
