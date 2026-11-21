//! Chat prompt rendering with the model's own Jinja chat template (from the
//! official repo), the same template transformers and llama.cpp render.
//!
//! Messages are job data. Every copy this module makes is wiped: messages
//! and calls on drop, the strings handed to the template ([`Strings`]) once
//! the render is done, and the rendered text as its buffer grows
//! ([`WipingBuf`]). Strings the template engine derives internally (filter
//! results such as `trim`, captured blocks) are outside our reach.

use std::sync::Arc;

use minijinja::{Environment, Error, ErrorKind, Value, context};
use zeroize::Zeroize;

/// One chat turn. Wiped on drop (move fields out with `std::mem::take`).
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

/// One tool call of an assistant turn. Wiped on drop.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatToolCall {
    pub id: String,
    pub name: String,
    /// The arguments object.
    pub arguments: serde_json::Value,
}

impl Drop for ChatMessage {
    fn drop(&mut self) {
        self.content.zeroize();
        if let Some(n) = &mut self.name {
            n.zeroize();
        }
    }
}

impl Drop for ChatToolCall {
    fn drop(&mut self) {
        self.name.zeroize();
        wipe_json(&mut self.arguments);
    }
}

/// Wipe every string in `v`, object keys included (the object is left
/// empty: its keys cannot be wiped in place).
pub fn wipe_json(v: &mut serde_json::Value) {
    match v {
        serde_json::Value::String(s) => s.zeroize(),
        serde_json::Value::Array(a) => a.iter_mut().for_each(wipe_json),
        serde_json::Value::Object(m) => {
            for (mut k, mut v) in std::mem::take(m) {
                k.zeroize();
                wipe_json(&mut v);
            }
        }
        _ => {}
    }
}

/// The template's values, built so every string in them is an `Arc<str>`
/// this keeps: on drop, once the render has released its references, each
/// is zeroed in place.
#[derive(Default)]
struct Strings(Vec<Arc<str>>);

impl Strings {
    fn str(&mut self, s: &str) -> Value {
        let a: Arc<str> = Arc::from(s);
        self.0.push(a.clone());
        Value::from(a)
    }

    fn json(&mut self, v: &serde_json::Value) -> Value {
        use serde_json::Value as J;
        match v {
            J::Null => Value::from(()),
            J::Bool(b) => Value::from(*b),
            J::Number(n) => match (n.as_i64(), n.as_u64()) {
                (Some(i), _) => Value::from(i),
                (None, Some(u)) => Value::from(u),
                _ => Value::from(n.as_f64().unwrap_or(0.0)),
            },
            J::String(s) => self.str(s),
            J::Array(a) => Value::from(a.iter().map(|x| self.json(x)).collect::<Vec<_>>()),
            J::Object(m) => Value::from_iter(
                m.iter()
                    .map(|(k, v)| (self.str(k), self.json(v)))
                    .collect::<Vec<_>>(),
            ),
        }
    }

    /// The template's view of `m`: the OpenAI chat shape the official
    /// templates read (`tool_calls[].function.{name, arguments}` with the
    /// arguments as a mapping).
    fn message(&mut self, m: &ChatMessage) -> Value {
        let mut fields = vec![
            (self.str("role"), self.str(&m.role)),
            (self.str("content"), self.str(&m.content)),
        ];
        if !m.tool_calls.is_empty() {
            let calls: Vec<Value> = m
                .tool_calls
                .iter()
                .map(|c| {
                    let function = Value::from_iter([
                        (self.str("name"), self.str(&c.name)),
                        (self.str("arguments"), self.json(&c.arguments)),
                    ]);
                    Value::from_iter([
                        (self.str("id"), self.str(&c.id)),
                        (self.str("type"), self.str("function")),
                        (self.str("function"), function),
                    ])
                })
                .collect();
            fields.push((self.str("tool_calls"), Value::from(calls)));
        }
        if let Some(id) = &m.tool_call_id {
            fields.push((self.str("tool_call_id"), self.str(id)));
        }
        if let Some(n) = &m.name {
            fields.push((self.str("name"), self.str(n)));
        }
        Value::from_iter(fields)
    }
}

/// Strings a render left shared (so unwiped); tests assert it stays 0.
#[cfg(test)]
static UNWIPED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl Drop for Strings {
    fn drop(&mut self) {
        for a in &mut self.0 {
            let Some(s) = Arc::get_mut(a) else {
                #[cfg(test)]
                UNWIPED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                continue;
            };
            // SAFETY: zero bytes are valid UTF-8, so the str stays valid.
            unsafe { s.as_bytes_mut() }.zeroize();
        }
    }
}

/// The render's output buffer: growing moves the text into a larger
/// allocation and wipes the old one; dropped unclaimed, it wipes itself.
#[derive(Default)]
struct WipingBuf(Vec<u8>);

impl std::io::Write for WipingBuf {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        let need = self.0.len() + b.len();
        if need > self.0.capacity() {
            let mut grown = Vec::with_capacity(need.max(self.0.capacity() * 2).max(4096));
            grown.extend_from_slice(&self.0);
            self.0.zeroize();
            self.0 = grown;
        }
        self.0.extend_from_slice(b);
        Ok(b.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Drop for WipingBuf {
    fn drop(&mut self) {
        self.0.zeroize();
    }
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
        let mut m = Self::new("tool", content);
        m.tool_call_id = Some(id.into());
        m.name = Some(name.into());
        m
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
    // Declared before the values so it drops after them (and after the
    // render released them): only then are its strings unshared.
    let mut strings = Strings::default();
    let msgs: Vec<Value> = messages.iter().map(|m| strings.message(m)).collect();
    let tools = match tools.is_empty() {
        true => Value::from(()),
        false => Value::from(tools.iter().map(|t| strings.json(t)).collect::<Vec<_>>()),
    };
    let mut out = WipingBuf::default();
    env.get_template("chat")
        .and_then(|t| {
            t.render_captured_to(
                context! {
                    messages => msgs,
                    tools => tools,
                    add_generation_prompt => true,
                    enable_thinking => thinking,
                    bos_token => bos,
                },
                &mut out,
            )
            .map(drop)
        })
        .map_err(|e| format!("render chat template: {e}"))?;
    String::from_utf8(std::mem::take(&mut out.0)).map_err(|e| {
        let mut bytes = e.into_bytes();
        bytes.zeroize();
        "render chat template: output is not UTF-8".to_string()
    })
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
    // Clones are wiped on drop (`ChatMessage`), and so is the full render.
    let mut all = vec![ChatMessage::new("user", PRIOR_USER), prior];
    all.extend(messages.iter().cloned());
    let mut full = zeroize::Zeroizing::new(render(template, &all, tools, thinking, bos)?);
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
    Ok(full.split_off(cut))
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
    fn render_releases_every_string_for_wiping() {
        let tpl = "{% for m in messages %}{{ m.role }}:{{ m.content }}\
            {% for c in m.tool_calls or [] %}[{{ c.function.name }} {{ c.function.arguments | tojson }}]{% endfor %}\
            {% endfor %}{% for t in tools or [] %}<{{ t.name }}>{% endfor %}";
        let mut call = ChatMessage::new("assistant", "");
        call.tool_calls.push(ChatToolCall {
            id: "c1".into(),
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Lisbon"}),
        });
        let msgs = [ChatMessage::new("user", "secret prompt"), call];
        let tools = [serde_json::json!({"name": "get_weather"})];
        let before = UNWIPED.load(std::sync::atomic::Ordering::SeqCst);
        let out = render(tpl, &msgs, &tools, false, "").unwrap();
        assert!(out.contains("user:secret prompt"), "{out}");
        assert!(out.contains("[get_weather {\"city\":\"Lisbon\"}]"), "{out}");
        assert!(out.contains("<get_weather>"), "{out}");
        assert_eq!(UNWIPED.load(std::sync::atomic::Ordering::SeqCst), before);
    }

    #[test]
    fn continuation_after_tool_calls_starts_past_the_calls() {
        let next = [ChatMessage::tool_result("c1", "roll", "4")];
        let out = continuation(TOOLS_T, &next, &[], "</call>", false, "").unwrap();
        assert_eq!(out, "<|end|><result>4</result><|assistant|>");
    }
}
