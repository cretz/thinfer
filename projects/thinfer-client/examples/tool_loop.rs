//! A text session with one tool, end to end: the model calls the tool, the
//! result goes back, the model answers; then the session is exported and
//! imported again and the conversation continues from the import.
//!
//! cargo run --release -p thinfer-client --example tool_loop -- https://localhost:8080
//!
//! Set `THINFER_TOKEN` for an `auth_token` server, and `THINFER_INSECURE=1`
//! to accept a self-signed certificate (local testing only).

use thinfer_client::tools::{FinishReason, ToolCall, ToolDef};
use thinfer_client::wire::{JobSpec, TextMessage, TextSpec};
use thinfer_client::{Client, Error, Output, Session, Update};

fn weather(call: &ToolCall) -> String {
    let city = call.arguments["city"].as_str().unwrap_or("?");
    format!("{{\"city\":\"{city}\",\"sky\":\"thick fog\",\"celsius\":9}}")
}

async fn reply(client: &Client, spec: TextSpec) -> Result<Output, Error> {
    let out = client
        .run(&JobSpec::Text(spec), |u| {
            if let Update::Text(t) = u {
                eprint!("{t}");
            }
        })
        .await?;
    eprintln!();
    Ok(out)
}

/// Send `messages` from `node` until the model answers in prose; returns
/// the last node.
async fn turn(
    client: &Client,
    session: &Session,
    mut node: Option<String>,
    mut messages: Vec<TextMessage>,
    tools: &[ToolDef],
) -> Result<String, Error> {
    loop {
        let mut spec = TextSpec::new(messages);
        spec.session = Some(session.at(node.as_deref()));
        spec.tools = tools.to_vec();
        spec.max_tokens = Some(300);
        let out = reply(client, spec).await?;
        node = out.session_node().map(str::to_string);
        if out.finish_reason() != FinishReason::ToolCalls {
            return Ok(node.expect("session reply has a node"));
        }
        // Continue from the reply's node with just the results.
        messages = out
            .tool_calls
            .iter()
            .map(|c| {
                eprintln!("[tool call {}({})]", c.name, c.arguments);
                TextMessage::tool_result(c, weather(c))
            })
            .collect();
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let url = std::env::args()
        .nth(1)
        .unwrap_or("https://localhost:8080".into());
    let mut b = Client::builder(&url)
        .danger_accept_invalid_certs(std::env::var("THINFER_INSECURE").is_ok_and(|v| v == "1"));
    if let Ok(t) = std::env::var("THINFER_TOKEN") {
        b = b.token(t);
    }
    let client = b.build()?;

    let tools = vec![ToolDef {
        name: "get_weather".into(),
        description: "Current weather in a city.".into(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }),
    }];

    let session = client.create_session(None, Some(600)).await?;
    let node = turn(
        &client,
        &session,
        None,
        vec![
            TextMessage::system("You narrate a visual novel. Use tools when they help."),
            TextMessage::user(
                "Check the weather in Lisbon, then describe the opening scene in two sentences.",
            ),
        ],
        &tools,
    )
    .await?;

    // Export (still sealed under the session key), drop the server copy,
    // import it back, and continue from the same node.
    let blob = client.export_session(&session).await?;
    eprintln!("[exported {} bytes]", blob.len());
    client.delete_session(&session).await?;
    let restored = client
        .import_session(
            &blob,
            thinfer_client::JobKey::from_b64(&session.key().to_b64())?,
        )
        .await?;
    let info = client.session_info(&restored).await?;
    eprintln!("[imported as {} with {} nodes]", info.id, info.nodes.len());

    turn(
        &client,
        &restored,
        Some(node),
        vec![TextMessage::user(
            "In one sentence: what is the weather doing in the scene?",
        )],
        &tools,
    )
    .await?;
    client.delete_session(&restored).await
}
