//! `thinfer generate text`: chat reply from an LLM, streamed to stdout. Thin
//! clap adapter over `thinfer_app::TextRequest`, plus the tool loop: with
//! `--tools`, a reply of tool calls runs each call's command and continues
//! the conversation with the results until the model answers in prose.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::Args;
use thinfer_app::config::ResidencyBudget;
use thinfer_app::model::{TEXT_DEFAULT_RAM, TEXT_DEFAULT_VRAM, TextModelId};
use thinfer_app::progress::ProgressSink;
use thinfer_app::remote::RemoteExecutor;
use thinfer_app::request::{StateChunk, TextSession};
use thinfer_app::text_format::{self, TextFormat};
use thinfer_app::tools::{self, ToolCall, ToolChoice, ToolDef};
use thinfer_app::wire::{JobSpec, TextMessage, TextSpec};
use thinfer_app::{JobRequest, JobSummary, TextRequest, parse_budget};

use super::CliSink;

#[derive(Args)]
pub struct GenerateText {
    #[arg(long, default_value_t = TextModelId::DEFAULT, value_enum)]
    pub model: TextModelId,
    /// The user message.
    #[arg(long)]
    pub prompt: String,
    /// Optional system message.
    #[arg(long)]
    pub system: Option<String>,
    /// Upper bound on generated tokens (per reply).
    #[arg(long, default_value_t = 2048)]
    pub max_tokens: u32,
    /// Let the model reason (thinking mode) before it answers.
    #[arg(long)]
    pub think: bool,
    /// Sampling temperature (0 = greedy). Default: the model card's value.
    #[arg(long)]
    pub temperature: Option<f32>,
    #[arg(long)]
    pub top_p: Option<f32>,
    #[arg(long)]
    pub top_k: Option<u32>,
    #[arg(long)]
    pub min_p: Option<f32>,
    #[arg(long)]
    pub presence_penalty: Option<f32>,
    #[arg(long)]
    pub repetition_penalty: Option<f32>,
    #[arg(long)]
    pub seed: Option<u64>,
    /// Constrain the reply (after any thinking) to this JSON Schema file.
    #[arg(long, conflicts_with = "grammar")]
    pub schema: Option<PathBuf>,
    /// Constrain the reply (after any thinking) to this Lark grammar file.
    #[arg(long)]
    pub grammar: Option<PathBuf>,
    /// Tools the model may call: a JSON array of `{name, description,
    /// parameters (JSON Schema), command}`. `command` (an argv list) runs per
    /// call with the arguments JSON on stdin; its stdout is the result. With
    /// a tool that has no command, the calls print as JSON on stdout and the
    /// command exits (run them yourself).
    #[arg(long)]
    pub tools: Option<PathBuf>,
    /// `auto` (default), `required`, `none`, or a tool name (call that tool).
    #[arg(long, requires = "tools")]
    pub tool_choice: Option<String>,
    /// Most rounds of tool calls before giving up.
    #[arg(long, default_value_t = 8, requires = "tools")]
    pub max_tool_rounds: u32,
    /// Seconds one tool command may run.
    #[arg(long, default_value_t = 120, requires = "tools")]
    pub tool_timeout: u64,
    /// Also write the (final) reply to this file.
    #[arg(long)]
    pub output: Option<PathBuf>,
    #[arg(long)]
    pub ram_budget: Option<String>,
    #[arg(long)]
    pub vram_budget: Option<String>,
    #[arg(long, default_value_t = false)]
    pub download_as_needed: bool,
    #[command(flatten)]
    pub remote: super::RemoteArgs,
}

/// The `--schema` / `--grammar` file as a [`TextFormat`].
fn read_format(args: &GenerateText) -> Result<Option<TextFormat>, String> {
    let read =
        |p: &PathBuf| std::fs::read_to_string(p).map_err(|e| format!("read {}: {e}", p.display()));
    if let Some(p) = &args.schema {
        let schema = serde_json::from_str(&read(p)?)
            .map_err(|e| format!("--schema {}: {e}", p.display()))?;
        return Ok(Some(TextFormat::JsonSchema { schema }));
    }
    match &args.grammar {
        Some(p) => Ok(Some(TextFormat::Lark { grammar: read(p)? })),
        None => Ok(None),
    }
}

/// One `--tools` entry: the definition the model sees, and the command that
/// runs it here.
struct ToolEntry {
    def: ToolDef,
    command: Option<Vec<String>>,
}

fn read_tools(args: &GenerateText) -> Result<Vec<ToolEntry>, String> {
    let Some(p) = &args.tools else {
        return Ok(Vec::new());
    };
    let text = std::fs::read_to_string(p).map_err(|e| format!("read {}: {e}", p.display()))?;
    let bad = |e: String| format!("--tools {}: {e}", p.display());
    let entries: Vec<serde_json::Value> =
        serde_json::from_str(&text).map_err(|e| bad(e.to_string()))?;
    let mut tools = Vec::new();
    for mut e in entries {
        let command = match e.as_object_mut().and_then(|o| o.remove("command")) {
            None => None,
            Some(c) => Some(
                serde_json::from_value::<Vec<String>>(c)
                    .map_err(|_| bad("command must be a list of strings".into()))?,
            ),
        };
        let def: ToolDef = serde_json::from_value(e).map_err(|e| bad(e.to_string()))?;
        tools.push(ToolEntry { def, command });
    }
    if tools.is_empty() {
        return Err(format!("--tools {}: no tools", p.display()));
    }
    for t in &tools {
        if t.command.as_ref().is_some_and(|c| c.is_empty()) {
            return Err(format!("tool {:?}: empty command", t.def.name));
        }
    }
    Ok(tools)
}

fn tool_choice(arg: Option<&str>) -> ToolChoice {
    match arg {
        None | Some("auto") => ToolChoice::Auto,
        Some("required") => ToolChoice::Required,
        Some("none") => ToolChoice::None,
        Some(name) => ToolChoice::Tool(name.to_string()),
    }
}

/// Where replies come from: this machine (keeping the conversation's model
/// state between tool rounds) or a `--remote` server (stateless: the whole
/// conversation each round).
enum Runner {
    /// A fresh device per reply (as the server does per job): a round
    /// leaves no VRAM behind for the next model load.
    Local {
        budget: ResidencyBudget,
        /// The model state of the conversation so far (page-locked chunks,
        /// in RAM only), and how many messages it covers.
        state: Option<(Vec<StateChunk>, String, usize)>,
    },
    Remote(RemoteExecutor),
}

pub async fn run_text(args: GenerateText) -> Result<(), String> {
    let format = read_format(&args)?;
    let tools = read_tools(&args)?;
    let defs: Vec<ToolDef> = tools.iter().map(|t| t.def.clone()).collect();
    let choice = tool_choice(args.tool_choice.as_deref());
    let mut convo = Vec::new();
    if let Some(s) = &args.system {
        convo.push(TextMessage::new("system", s.as_str()));
    }
    convo.push(TextMessage::new("user", args.prompt.as_str()));

    let mut runner = match &args.remote.remote {
        Some(url) => Runner::Remote(
            RemoteExecutor::new(url, args.remote.remote_token.clone())?
                .with_priority(args.remote.priority),
        ),
        None => {
            let ram = parse_budget(
                "--ram-budget",
                Some(args.ram_budget.as_deref().unwrap_or(TEXT_DEFAULT_RAM)),
            )?;
            let vram = parse_budget(
                "--vram-budget",
                Some(args.vram_budget.as_deref().unwrap_or(TEXT_DEFAULT_VRAM)),
            )?;
            let budget = ResidencyBudget {
                ram_bytes: ram,
                vram_bytes: vram,
            };
            let messages = convo.iter().cloned().map(TextMessage::into_chat).collect();
            let probe = request(&args, &defs, &choice, &format, messages, None, budget);
            probe.validate()?;
            let files = probe.required_files();
            super::ensure_files(&files, args.download_as_needed).await?;
            Runner::Local {
                budget,
                state: None,
            }
        }
    };

    let sink = CliSink::new();
    for round in 0..=args.max_tool_rounds {
        let summary =
            match run_once(&args, &defs, &choice, &format, &convo, &mut runner, &sink).await {
                Ok(s) => s,
                // The request path keeps the compiler's message out (it quotes
                // job data); this is the user's own terminal, so show it.
                Err(e) if tools::is_rejection(&e) => {
                    return Err(tools::explain(args.model, &defs, &choice, format.as_ref()));
                }
                Err(e) if text_format::is_rejection(&e) && format.is_some() => {
                    return Err(text_format::explain(args.model, format.as_ref().unwrap()));
                }
                Err(e) => return Err(e),
            };
        sink.footer(&summary);
        if summary.tool_calls.is_empty() {
            return Ok(());
        }
        let calls = summary.tool_calls;
        // A tool this machine cannot run: hand the calls to the caller.
        if calls.iter().any(|c| {
            tools
                .iter()
                .all(|t| t.def.name != c.name || t.command.is_none())
        }) {
            println!(
                "{}",
                serde_json::json!({ "toolCalls": serde_json::to_value(&calls).unwrap() })
            );
            return Ok(());
        }
        if round == args.max_tool_rounds {
            return Err(format!(
                "the model was still calling tools after {} rounds (--max-tool-rounds)",
                args.max_tool_rounds
            ));
        }
        let mut results = Vec::new();
        for call in &calls {
            let entry = tools.iter().find(|t| t.def.name == call.name).unwrap();
            let command = entry.command.as_ref().unwrap();
            sink.note(&format!("tool {}({})", call.name, call.arguments));
            let out = run_tool(command, call, Duration::from_secs(args.tool_timeout)).await;
            sink.note(&format!("tool {} -> {} bytes", call.name, out.len()));
            results.push(TextMessage {
                tool_call_id: Some(call.id.clone()),
                name: Some(call.name.clone()),
                ..TextMessage::new("tool", out)
            });
        }
        convo.push(TextMessage {
            tool_calls: calls,
            ..TextMessage::new("assistant", "")
        });
        convo.extend(results);
    }
    unreachable!("the loop returns by its last round")
}

/// The local request for `messages` (resuming `session` when given).
fn request(
    args: &GenerateText,
    defs: &[ToolDef],
    choice: &ToolChoice,
    format: &Option<TextFormat>,
    messages: Vec<thinfer_app::ChatMessage>,
    session: Option<Arc<TextSession>>,
    budget: ResidencyBudget,
) -> TextRequest {
    TextRequest {
        model: args.model,
        messages,
        max_tokens: args.max_tokens,
        think: args.think,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        min_p: args.min_p,
        presence_penalty: args.presence_penalty,
        repetition_penalty: args.repetition_penalty,
        seed: args.seed,
        budget,
        output: args.output.clone(),
        format: format.clone(),
        session,
        tools: defs.to_vec(),
        tool_choice: choice.clone(),
    }
}

/// One reply to `convo`.
async fn run_once(
    args: &GenerateText,
    defs: &[ToolDef],
    choice: &ToolChoice,
    format: &Option<TextFormat>,
    convo: &[TextMessage],
    runner: &mut Runner,
    sink: &CliSink,
) -> Result<JobSummary, String> {
    match runner {
        Runner::Remote(executor) => {
            let spec = JobSpec::Text(TextSpec {
                model: Some(args.model),
                messages: convo.to_vec(),
                max_tokens: Some(args.max_tokens),
                think: Some(args.think),
                temperature: args.temperature,
                top_p: args.top_p,
                top_k: args.top_k,
                min_p: args.min_p,
                presence_penalty: args.presence_penalty,
                repetition_penalty: args.repetition_penalty,
                seed: args.seed,
                format: format.clone(),
                session: None,
                tools: defs.to_vec(),
                tool_choice: (!defs.is_empty()).then(|| choice.clone()),
            });
            let output = args.output.clone().unwrap_or_default();
            executor.run(&spec, &output, sink).await
        }
        Runner::Local { budget, state } => {
            // With tools the reply's model state is kept (in RAM) so the next
            // round prefills only the tool results.
            let (messages, session) = match (defs.is_empty(), state.take()) {
                (true, _) => (convo.to_vec(), None),
                (false, None) => (
                    convo.to_vec(),
                    Some(TextSession {
                        path: Vec::new(),
                        weights: None,
                    }),
                ),
                (false, Some((path, weights, covered))) => (
                    convo[covered..].to_vec(),
                    Some(TextSession {
                        path,
                        weights: Some(weights),
                    }),
                ),
            };
            let messages = messages.into_iter().map(TextMessage::into_chat).collect();
            let req = request(
                args,
                defs,
                choice,
                format,
                messages,
                session.map(Arc::new),
                *budget,
            );
            req.validate()?;
            let mut job = JobRequest::Text(req);
            let executor = super::new_local().await?;
            let mut summary = executor.run(&job, sink).await?;
            if std::env::var_os("THINFER_TRACE").is_some() {
                thinfer_app::report_mem(executor.backend(), budget.ram_bytes, budget.vram_bytes);
            }
            drop(executor);
            // Keep the chain plus this reply's chunk for the next round.
            if let (JobRequest::Text(req), Some(save)) = (&mut job, summary.session.take())
                && let Some(Ok(mut chain)) = req.session.take().map(Arc::try_unwrap)
                && let Ok(save) = Arc::try_unwrap(save)
            {
                chain.path.push(save.chunk);
                // The reply's assistant turn (pushed next) is in the state.
                *state = Some((chain.path, save.weights, convo.len() + 1));
            }
            Ok(summary)
        }
    }
}

/// Run one call's command: arguments JSON on stdin, stdout is the result
/// (an error description when it fails, so the model can react).
async fn run_tool(command: &[String], call: &ToolCall, timeout: Duration) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    /// Most result bytes handed to the model.
    const MAX_OUTPUT: usize = 256 << 10;
    let spawned = tokio::process::Command::new(&command[0])
        .args(&command[1..])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn();
    let mut child = match spawned {
        Ok(c) => c,
        Err(e) => return format!("error: could not start the tool: {e}"),
    };
    let input = serde_json::to_vec(&call.arguments).unwrap_or_default();
    let mut stdin = child.stdin.take().expect("piped stdin");
    let mut stdout = child.stdout.take().expect("piped stdout");
    let mut stderr = child.stderr.take().expect("piped stderr");
    let work = async {
        // A tool may exit without reading its input: ignore a broken pipe.
        let _ = stdin.write_all(&input).await;
        drop(stdin);
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let (a, b) = tokio::join!(stdout.read_to_end(&mut out), stderr.read_to_end(&mut err));
        let status = child.wait().await;
        (a.and(b).and(status), out, err)
    };
    match tokio::time::timeout(timeout, work).await {
        Err(_) => format!("error: the tool timed out after {}s", timeout.as_secs()),
        Ok((Err(e), ..)) => format!("error: {e}"),
        Ok((Ok(status), out, err)) => {
            let text =
                |b: &[u8]| String::from_utf8_lossy(&b[..b.len().min(MAX_OUTPUT)]).into_owned();
            if status.success() {
                let mut s = text(&out);
                if out.len() > MAX_OUTPUT {
                    s.push_str("\n[output truncated]");
                }
                s
            } else {
                format!(
                    "error (exit {}): {}",
                    status.code().unwrap_or(-1),
                    text(&err).trim()
                )
            }
        }
    }
}
