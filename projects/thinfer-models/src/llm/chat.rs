//! Chat prompt rendering with the model's own Jinja chat template (from the
//! official repo), the same template transformers and llama.cpp render.

use minijinja::{Environment, Error, ErrorKind, Value, context};

/// One chat turn.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatMessage {
    /// `system`, `user`, `assistant` or `tool` (a tool call's result).
    pub role: String,
    pub content: String,
    /// An assistant turn's tool calls.
    pub tool_calls: Vec<ChatToolCall>,
    /// A `tool` turn: the call it answers, and that call's tool name.
    pub tool_call_id: Option<String>,
    pub name: Option<String>,
}

/// One tool call of an assistant turn.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatToolCall {
    pub id: String,
    pub name: String,
    /// The arguments object.
    pub arguments: serde_json::Value,
}

impl ChatMessage {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    /// The result of tool call `id` (to tool `name`).
    pub fn tool_result(id: &str, name: &str, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: Some(id.into()),
            name: Some(name.into()),
            ..Self::new("tool", content)
        }
    }

    /// The template's view: the OpenAI chat shape the official templates
    /// read (`tool_calls[].function.{name, arguments}` with the arguments
    /// as a mapping).
    fn to_value(&self) -> Value {
        let mut m = serde_json::Map::new();
        m.insert("role".into(), self.role.clone().into());
        m.insert("content".into(), self.content.clone().into());
        if !self.tool_calls.is_empty() {
            let calls: Vec<serde_json::Value> = self
                .tool_calls
                .iter()
                .map(|c| {
                    serde_json::json!({
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments},
                    })
                })
                .collect();
            m.insert("tool_calls".into(), calls.into());
        }
        if let Some(id) = &self.tool_call_id {
            m.insert("tool_call_id".into(), id.clone().into());
        }
        if let Some(n) = &self.name {
            m.insert("name".into(), n.clone().into());
        }
        Value::from_serialize(serde_json::Value::Object(m))
    }
}

/// Render `messages` through `template`, ending with the assistant generation
/// prompt. `tools` (OpenAI function-tool objects) is the template's `tools`
/// (none when empty). `thinking` sets the template's `enable_thinking`
/// (Qwen) / reasoning switch; templates without one ignore it. `bos` is the
/// template's `bos_token` (empty for templates that do not emit one).
pub fn render(
    template: &str,
    messages: &[ChatMessage],
    tools: &[serde_json::Value],
    thinking: bool,
    bos: &str,
) -> Result<String, String> {
    // HF's `{% generation %}` marks assistant spans for training masks; it is
    // not Jinja and renders as its body.
    let src = strip_tag(&strip_tag(template, "generation"), "endgeneration");
    let mut env = Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
        Err(Error::new(ErrorKind::InvalidOperation, msg))
    });
    env.add_template("chat", &src)
        .map_err(|e| format!("chat template: {e}"))?;
    let msgs: Vec<Value> = messages.iter().map(ChatMessage::to_value).collect();
    let tools = match tools.is_empty() {
        true => Value::from(()),
        false => Value::from_serialize(tools),
    };
    env.get_template("chat")
        .and_then(|t| {
            t.render(context! {
                messages => msgs,
                tools => tools,
                add_generation_prompt => true,
                enable_thinking => thinking,
                bos_token => bos,
            })
        })
        .map_err(|e| format!("render chat template: {e}"))
}

/// Stand-ins for an earlier exchange (private-use code points: never in a
/// real message, untouched by templates' `trim`).
const PRIOR_USER: &str = "\u{e000}thinfer-prior-user\u{e000}";
const PRIOR_REPLY: &str = "\u{e000}thinfer-prior-reply\u{e000}";

/// The template text that continues a conversation whose last turn was an
/// assistant reply (already in the model's state) with `messages`: the
/// reply's end-of-turn markup, the new turns, and the generation prompt.
/// Rendered after a stand-in exchange and cut right after the stand-in
/// reply, so the result is exactly what a full render would append.
///
/// When `messages` opens with `tool` turns, the state's last reply ended in
/// tool calls: the stand-in reply makes one call per leading result (same
/// ids and names, empty arguments) and the cut falls after the last call's
/// closing markup `call_close` (the model's syntax), so the results render
/// as the template places them after its calls.
pub fn continuation(
    template: &str,
    messages: &[ChatMessage],
    tools: &[serde_json::Value],
    call_close: &str,
    thinking: bool,
    bos: &str,
) -> Result<String, String> {
    let results: Vec<&ChatMessage> = messages.iter().take_while(|m| m.role == "tool").collect();
    let mut prior = ChatMessage::new("assistant", PRIOR_REPLY);
    if !results.is_empty() {
        prior.content = String::new();
        prior.tool_calls = results
            .iter()
            .map(|r| ChatToolCall {
                id: r.tool_call_id.clone().unwrap_or_default(),
                name: r.name.clone().unwrap_or_default(),
                arguments: serde_json::json!({}),
            })
            .collect();
    }
    let calls = prior.tool_calls.len();
    let mut all = vec![ChatMessage::new("user", PRIOR_USER), prior];
    all.extend(messages.iter().cloned());
    let full = render(template, &all, tools, thinking, bos)?;
    let user = full
        .find(PRIOR_USER)
        .ok_or("chat template dropped the prior turn")?;
    let cut = match calls {
        0 => {
            let at = full[user..]
                .find(PRIOR_REPLY)
                .ok_or("chat template dropped the prior reply")?;
            user + at + PRIOR_REPLY.len()
        }
        n => {
            if call_close.is_empty() {
                return Err("tool results need the model's call markup".into());
            }
            let mut at = user;
            for _ in 0..n {
                at += full[at..]
                    .find(call_close)
                    .ok_or("chat template dropped the prior tool calls")?
                    + call_close.len();
            }
            at
        }
    };
    Ok(full[cut..].to_string())
}

/// Remove every `{%- name %}` / `{% name %}` / `{%- name -%}` tag.
fn strip_tag(src: &str, name: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(i) = rest.find("{%") {
        let Some(j) = rest[i..].find("%}") else { break };
        let inner = rest[i + 2..i + j].trim_matches(|c: char| c == '-' || c.is_whitespace());
        out.push_str(&rest[..i]);
        if inner != name {
            out.push_str(&rest[i..i + j + 2]);
        }
        rest = &rest[i + j + 2..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_generation_tags() {
        let s = "a{%- generation %}b{% endgeneration -%}c{% if x %}d{% endif %}";
        assert_eq!(
            strip_tag(&strip_tag(s, "generation"), "endgeneration"),
            "abc{% if x %}d{% endif %}"
        );
    }

    #[test]
    fn renders_chatml_like_template() {
        let t = "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content.strip() }}<|im_end|>\n{%- endfor %}{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}";
        let out = render(t, &[ChatMessage::new("user", " hi ")], &[], false, "").unwrap();
        assert_eq!(out, "<|im_start|>user\nhi<|im_end|><|im_start|>assistant");
    }

    #[test]
    fn continuation_is_the_full_render_suffix() {
        let t = "{{ bos_token }}{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content.strip() }}<|im_end|>\n{%- endfor %}{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}";
        let next = [ChatMessage::new("user", "again")];
        let out = continuation(t, &next, &[], "", false, "<s>").unwrap();
        assert_eq!(
            out,
            "<|im_end|><|im_start|>user\nagain<|im_end|><|im_start|>assistant"
        );
    }

    /// A toy template with tool calls (`<call>name</call>`) and results.
    const TOOLS_T: &str = "{%- if tools %}[tools:{% for t in tools %}{{ t.function.name }}{% endfor %}]{% endif %}{%- for m in messages %}{%- if m.role == 'tool' %}<result>{{ m.content }}</result>{%- else %}<|{{ m.role }}|>{{ m.content }}{%- if m.tool_calls %}{% for c in m.tool_calls %}<call>{{ c.function.name }}{{ c.function.arguments | tojson }}</call>{% endfor %}{% endif %}<|end|>{%- endif %}{%- endfor %}{%- if add_generation_prompt %}<|assistant|>{%- endif %}";

    #[test]
    fn renders_tools_and_calls() {
        let tools = [serde_json::json!({"type": "function", "function": {"name": "roll"}})];
        let mut a = ChatMessage::new("assistant", "");
        a.tool_calls.push(ChatToolCall {
            id: "c1".into(),
            name: "roll".into(),
            arguments: serde_json::json!({"sides": 6}),
        });
        let msgs = [
            ChatMessage::new("user", "roll"),
            a,
            ChatMessage::tool_result("c1", "roll", "4"),
        ];
        let out = render(TOOLS_T, &msgs, &tools, false, "").unwrap();
        assert_eq!(
            out,
            "[tools:roll]<|user|>roll<|end|><|assistant|><call>roll{\"sides\":6}</call><|end|><result>4</result><|assistant|>"
        );
    }

    #[test]
    fn continuation_after_tool_calls_starts_past_the_calls() {
        let next = [ChatMessage::tool_result("c1", "roll", "4")];
        let out = continuation(TOOLS_T, &next, &[], "</call>", false, "").unwrap();
        assert_eq!(out, "<|end|><result>4</result><|assistant|>");
    }
}
