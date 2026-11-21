//! Text generation (chat): render the conversation through the model's chat
//! template, prefill it in chunks, then sample token by token, streaming the
//! decoded text through the sink. Prompt and reply text are never logged.

use std::sync::Arc;
use std::time::Instant;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::secmem::Locked;
use thinfer_core::tokenizer::Tokenizer;
use thinfer_core::workspace::Workspace;
use thinfer_models::llm::chat;
use thinfer_models::llm::decoder::{Blocks, Decoder, MAX_STEP, Want};
use thinfer_models::llm::gemma4::{Gemma4Config, Gemma4Model};
use thinfer_models::llm::kernels::LlmPipelines;
use thinfer_models::llm::manifest::role;
use thinfer_models::llm::qwen35::{Qwen35Config, Qwen35Model};
use thinfer_models::llm::sampler::Sampler;
use thinfer_native::MmapFileOpener;
use thinfer_native::tokenizer::HfTokenizer;
use zeroize::{Zeroize, Zeroizing};

use crate::config::random_seed;
use crate::download::resolve_role;
use crate::model::TextModelId;
use crate::progress::ProgressSink;
use crate::request::{JobSummary, SessionSave, TextRequest, TextStats};
use crate::text_format::{self, Constraint};
use crate::tools::{self, FinishReason, Syntax};

/// Prompt tokens per prefill forward: one decoder step (bounds the
/// activation workspace: a step's routed-expert pairs are `step * top_k` rows).
const PREFILL_CHUNK: usize = MAX_STEP as usize;

pub async fn run(
    backend: &Arc<WgpuBackend>,
    req: &TextRequest,
    sink: &dyn ProgressSink,
) -> Result<JobSummary, String> {
    req.validate()?;
    let manifest = req.model.manifest();
    let tokenizer_json = resolve_role(manifest, role::TOKENIZER)?;
    let tokenizer = HfTokenizer::from_path(&tokenizer_json)
        .await
        .map_err(|e| format!("load tokenizer: {e:?}"))?;
    let template = std::fs::read_to_string(resolve_role(manifest, role::CHAT_TEMPLATE)?)
        .map_err(|e| format!("read chat template: {e}"))?;
    let config: serde_json::Value = read_json(&resolve_role(manifest, role::CONFIG)?)?;
    let gen_cfg: serde_json::Value = read_json(&resolve_role(manifest, role::GENERATION_CONFIG)?)?;
    let stop = eos_ids(&gen_cfg);
    let syntax = Syntax::of(req.model, |t| single_token(&tokenizer, t))?;
    // The reply's grammar: the tools' call syntax (prose shaped by the format,
    // if any), else the format alone.
    let grammar = match (req.tools.is_empty(), &req.format) {
        (false, format) => Some((
            llguidance::api::TopLevelGrammar::from_lark(tools::request_grammar(
                syntax,
                &req.tools,
                &req.tool_choice,
                format.as_ref(),
                &stop,
            )?),
            tools::REJECTED,
        )),
        (true, Some(f)) => Some((f.grammar(), text_format::REJECTED)),
        (true, None) => None,
    };
    // The constraint's view of the tokenizer, built (and the grammar
    // compiled) before any weights load: a bad schema fails fast.
    let grammar_env = match grammar {
        Some((g, rejected)) => {
            let env = Arc::new(text_format::tok_env(&tokenizer_json, None, &stop)?);
            let gate = if req.think {
                Some(single_token(&tokenizer, req.model.think_close())?)
            } else {
                None
            };
            Constraint::new(g.clone(), &env, gate, rejected)?;
            Some(GrammarEnv {
                grammar: g,
                rejected,
                env,
                gate,
            })
        }
        None => None,
    };
    let tool_defs = tools::template_tools(&req.tools);

    // The template's `bos_token` (Gemma opens every prompt with it).
    let bos = match gen_cfg.get("bos_token_id").and_then(|v| v.as_u64()) {
        Some(id) => tokenizer
            .decode(&[id as u32], false)
            .map_err(|e| format!("decode bos: {e:?}"))?,
        None => String::new(),
    };
    // A resumed session already holds the earlier turns: render only what
    // follows its last reply.
    let resume_at = req
        .session
        .as_ref()
        .and_then(|s| s.path.last())
        .map_or(0, |c| c.end);
    let mut prompt = if resume_at > 0 {
        chat::continuation(
            &template,
            &req.messages,
            &tool_defs,
            syntax.call_close(),
            req.think,
            &bos,
        )?
    } else {
        chat::render(&template, &req.messages, &tool_defs, req.think, &bos)?
    };
    if req.messages.last().is_some_and(|m| m.role == "tool") {
        // Grown into a fresh buffer: the old one (prompt text) is wiped.
        let tail = syntax.after_results(req.think);
        let mut grown = String::with_capacity(prompt.len() + tail.len());
        grown.push_str(&prompt);
        grown.push_str(tail);
        prompt.zeroize();
        prompt = grown;
    }
    let ids = tokenizer.encode(&prompt, false);
    prompt.zeroize();
    let ids = Locked::new(ids.map_err(|e| format!("tokenize: {e:?}"))?);
    let max_ctx = resume_at + (ids.len() + req.max_tokens as usize + 1) as u32;
    let seed = req.seed.unwrap_or_else(random_seed);
    tracing::info!(
        target: thinfer_core::trace::DIAG,
        model = %req.model, prompt_tokens = ids.len(), max_tokens = req.max_tokens,
        think = req.think, seed, resume_at, ram_budget = req.budget.ram_bytes,
        vram_budget = req.budget.vram_bytes, "generate start",
    );

    let gguf = resolve_role(manifest, role::GGUF)?;
    tracing::info!(
        target: thinfer_core::trace::DIAG,
        file = %gguf.file_name().unwrap_or_default().to_string_lossy(),
        "llm weights",
    );
    // A session's state is only valid for the exact weights that made it.
    let weights = weights_identity(&gguf)?;
    if let Some(built) = req.session.as_ref().and_then(|s| s.weights.as_ref())
        && *built != weights
    {
        return Err("session was built with different model weights".into());
    }
    let source = GgufSource::open(
        MmapFileOpener::new(&gguf)
            .await
            .map_err(|e| format!("open gguf: {e:?}"))?,
    )
    .await
    .map_err(|e| format!("parse gguf: {e:?}"))?;
    sink.note(&format!(
        "Loading {} ({} prompt tokens)",
        req.model,
        ids.len()
    ));
    let pipes = LlmPipelines::compile(backend)
        .await
        .map_err(|e| format!("compile llm pipelines: {e:?}"))?;
    let load_err = |e| format!("model load: {e:?}");
    // Inside what the OS keeps resident: past it allocations are demoted to
    // system memory and every read crosses PCIe.
    let mut budget = req.budget;
    budget.vram_bytes =
        thinfer_native::gpu_budget::resident_vram_budget(backend.wgpu_device(), budget.vram_bytes);
    if budget.vram_bytes < req.budget.vram_bytes {
        tracing::info!(
            target: thinfer_core::trace::DIAG,
            requested = req.budget.vram_bytes, resident = budget.vram_bytes,
            "vram budget lowered to the OS-resident headroom",
        );
    }
    let job = Job {
        backend,
        req,
        sink,
        pipes,
        ids,
        stop,
        tokenizer,
        seed,
        max_ctx,
        grammar_env,
        syntax,
        weights,
    };
    match req.model {
        TextModelId::Qwen36_35bA3b => {
            let cfg = Qwen35Config::from_json(&config)?;
            let model =
                Qwen35Model::load(backend, source, cfg, budget, max_ctx).map_err(load_err)?;
            job.generate(model).await
        }
        TextModelId::Gemma4_26bA4b => {
            let cfg = Gemma4Config::from_json(&config)?;
            let model = Gemma4Model::load(backend, source, cfg, budget, max_ctx)
                .await
                .map_err(load_err)?;
            job.generate(model).await
        }
    }
}

/// A compiled reply grammar's inputs: the grammar, its detail-free
/// rejection, the constraint's tokenizer view and the think gate.
struct GrammarEnv {
    grammar: llguidance::api::TopLevelGrammar,
    rejected: &'static str,
    env: Arc<llguidance::toktrie::TokEnv>,
    gate: Option<u32>,
}

/// Everything a generate needs besides the model.
struct Job<'a> {
    backend: &'a Arc<WgpuBackend>,
    req: &'a TextRequest,
    sink: &'a dyn ProgressSink,
    pipes: LlmPipelines,
    /// Prompt token ids (job data).
    ids: Locked<u32>,
    stop: Vec<u32>,
    tokenizer: HfTokenizer,
    seed: u64,
    max_ctx: u32,
    /// The reply grammar (tools / format), when set.
    grammar_env: Option<GrammarEnv>,
    /// The model's tool-call syntax.
    syntax: Syntax,
    /// Weights identity (a session node records it).
    weights: String,
}

impl Job<'_> {
    /// Prefill the prompt in chunks, then sample until a stop token or the
    /// token budget, streaming the reply.
    async fn generate<B: Blocks>(
        self,
        mut model: Decoder<GgufSource<MmapFileOpener>, B>,
    ) -> Result<JobSummary, String> {
        let Job {
            backend,
            req,
            sink,
            pipes,
            ids,
            stop,
            tokenizer,
            seed,
            max_ctx,
            grammar_env,
            syntax,
            weights,
        } = self;
        let accel = model
            .enable_accel(backend)
            .map_err(|e| format!("enable accelerator: {e:?}"))?;
        tracing::info!(target: thinfer_core::trace::DIAG, accel, "llm path");
        let ws = Workspace::new(backend.clone(), model.residency().arbiter().clone());
        let mut state = model
            .new_state(backend, &ws, max_ctx)
            .map_err(|e| format!("state: {e:?}"))?;
        if let Some(session) = &req.session {
            let chain: Vec<_> = session.path.iter().collect();
            state
                .import(&chain)
                .map_err(|e| format!("resume session: {e:?}"))?;
        }
        let resume_at = state.pos;

        let t0 = Instant::now();
        let mut logits = Zeroizing::new(Vec::new());
        for (i, chunk) in ids.chunks(PREFILL_CHUNK).enumerate() {
            if sink.cancelled() {
                return Err("cancelled".into());
            }
            let last = (i + 1) * PREFILL_CHUNK >= ids.len();
            let out = model
                .forward(
                    backend,
                    &pipes,
                    &ws,
                    &mut state,
                    chunk,
                    Want {
                        logits: last,
                        ..Default::default()
                    },
                )
                .await
                .map_err(|e| format!("prefill: {e:?}"))?;
            if last {
                logits = out.logits;
            }
        }
        let prefill_secs = t0.elapsed().as_secs_f64();

        let mut constraint = match grammar_env {
            Some(g) => Some(Constraint::new(g.grammar, &g.env, g.gate, g.rejected)?),
            None => None,
        };
        let mut sampler = Sampler::new(req.sampling(seed));
        // Fixed size (never reallocates): the generated ids are job data.
        let mut produced = Locked::<u32>::zeroed(req.max_tokens as usize);
        let mut n = 0;
        let mut text = Detok::default();
        // Where the reply's tool calls start: nothing from there on is prose
        // (it is parsed, not streamed).
        let mut calls_at: Option<usize> = None;
        let mut out_of_tokens = false;
        let t1 = Instant::now();
        loop {
            if sink.cancelled() {
                return Err("cancelled".into());
            }
            if let Some(c) = &mut constraint {
                c.mask(&mut logits)?;
            }
            let tok = sampler.sample(&mut logits);
            if stop.contains(&tok) {
                break;
            }
            // Penalties see sampled tokens only, never grammar-forced ones.
            sampler.accept(tok);
            let forced = match &mut constraint {
                Some(c) => Zeroizing::new(c.accept(tok)?),
                None => Zeroizing::new(Vec::new()),
            };
            let start = n;
            for &t in std::iter::once(&tok).chain(forced.iter()) {
                if n >= req.max_tokens as usize {
                    break;
                }
                if t == syntax.open()
                    && calls_at.is_none()
                    && !req.tools.is_empty()
                    && constraint.as_ref().is_some_and(|c| c.is_open())
                {
                    calls_at = Some(n);
                }
                produced[n] = t;
                n += 1;
            }
            sink.text(&text.push(&tokenizer, &produced[..calls_at.unwrap_or(n)])?);
            if n >= req.max_tokens as usize {
                out_of_tokens = true;
                break;
            }
            if constraint.as_ref().is_some_and(|c| c.finished()) {
                break;
            }
            // The sampled token plus any forced run in one forward per chunk
            // (jump-forward): a fixed key costs one pass, not one per token.
            for chunk in produced[start..n].chunks(PREFILL_CHUNK) {
                logits = model
                    .forward(
                        backend,
                        &pipes,
                        &ws,
                        &mut state,
                        chunk,
                        Want {
                            logits: true,
                            ..Default::default()
                        },
                    )
                    .await
                    .map_err(|e| format!("decode: {e:?}"))?
                    .logits;
            }
        }
        sink.text(&text.flush(&tokenizer, &produced[..calls_at.unwrap_or(n)])?);
        let decode_secs = t1.elapsed().as_secs_f64();
        let (finish_reason, tool_calls) = match (out_of_tokens, calls_at) {
            (true, _) => (FinishReason::Length, Vec::new()),
            (false, Some(at)) => {
                let calls = Zeroizing::new(
                    tokenizer
                        .decode(&produced[at..n], false)
                        .map_err(|e| format!("detokenize: {e:?}"))?,
                );
                let id_base = random_seed() & 0xffff_ffff;
                let parsed = tools::parse_calls(syntax, &req.tools, &calls, id_base)?;
                (FinishReason::ToolCalls, parsed)
            }
            (false, None) => (FinishReason::Stop, Vec::new()),
        };
        let n = n as u32;
        sink.note(&format!(
            "{} prompt tokens in {prefill_secs:.2}s ({:.1} tok/s), {n} generated in {decode_secs:.2}s ({:.1} tok/s)",
            ids.len(),
            ids.len() as f64 / prefill_secs.max(1e-9),
            n.saturating_sub(1) as f64 / decode_secs.max(1e-9),
        ));

        // The positions this request added, for the session's new node.
        let session = match &req.session {
            Some(_) => Some(Arc::new(SessionSave {
                chunk: state
                    .export(resume_at)
                    .await
                    .map_err(|e| format!("save session: {e:?}"))?,
                weights,
            })),
            None => None,
        };

        let output = req.output.clone().unwrap_or_default();
        if let Some(path) = &req.output {
            let reply = std::mem::take(&mut *text.full).into_bytes();
            crate::progress::write_artifact(sink, path, reply)?;
        }
        Ok(JobSummary {
            output,
            seed: Some(seed),
            text: Some(TextStats {
                prompt_tokens: ids.len() as u32,
                generated_tokens: n,
                prefill_secs,
                decode_secs,
                finish_reason,
            }),
            session,
            tool_calls,
            ..Default::default()
        })
    }
}

/// The weights file's identity: name and byte length (cheap; a different
/// quant, merge or re-upload changes either).
fn weights_identity(gguf: &std::path::Path) -> Result<String, String> {
    let len = std::fs::metadata(gguf)
        .map_err(|e| format!("stat {}: {e}", gguf.display()))?
        .len();
    Ok(format!(
        "{}:{len}",
        gguf.file_name().unwrap_or_default().to_string_lossy()
    ))
}

/// The id of a special token that encodes to exactly one id.
fn single_token(tokenizer: &HfTokenizer, text: &str) -> Result<u32, String> {
    match tokenizer.encode(text, false).as_deref() {
        Ok([id]) => Ok(*id),
        _ => Err(format!("{text:?} is not a single token")),
    }
}

fn read_json(path: &std::path::Path) -> Result<serde_json::Value, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_str(&s).map_err(|e| format!("parse {}: {e}", path.display()))
}

/// `eos_token_id` of a `generation_config.json` (a number or a list).
fn eos_ids(gen_cfg: &serde_json::Value) -> Vec<u32> {
    match gen_cfg.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => n.as_u64().map(|v| v as u32).into_iter().collect(),
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .filter_map(|v| v.as_u64())
            .map(|v| v as u32)
            .collect(),
        _ => Vec::new(),
    }
}

/// Incremental detokenization (the vLLM / TGI scheme): each token decodes
/// only the short window `ids[prefix..]` and emits what it adds past
/// `ids[prefix..read]` (decoding both sides keeps tokenizer-specific leading
/// space handling out of the delta), holding back a trailing partial UTF-8
/// sequence (decoded as U+FFFD) until a later token completes it. O(1) per
/// token. Every string here is reply text: all are wiped.
#[derive(Default)]
struct Detok {
    prefix: usize,
    read: usize,
    /// The reply so far, for the artifact.
    full: Zeroizing<String>,
}

impl Detok {
    fn push(&mut self, tok: &HfTokenizer, ids: &[u32]) -> Result<Zeroizing<String>, String> {
        self.step(tok, ids, false)
    }

    /// Emit everything left, a trailing partial sequence included.
    fn flush(&mut self, tok: &HfTokenizer, ids: &[u32]) -> Result<Zeroizing<String>, String> {
        self.step(tok, ids, true)
    }

    fn step(
        &mut self,
        tok: &HfTokenizer,
        ids: &[u32],
        last: bool,
    ) -> Result<Zeroizing<String>, String> {
        // Special tokens kept: thinking markers (Gemma's `<|channel>` /
        // `<channel|>`) are special, and stop tokens never enter `ids`.
        let decode = |w: &[u32]| {
            tok.decode(w, false)
                .map(Zeroizing::new)
                .map_err(|e| format!("detokenize: {e:?}"))
        };
        let prev = decode(&ids[self.prefix..self.read])?;
        let cur = decode(&ids[self.prefix..])?;
        let mut delta = Zeroizing::new(String::new());
        if cur.len() > prev.len() && (last || !cur.ends_with('\u{FFFD}')) {
            // A merge that rewrote the window's start (rare) emits nothing.
            if let Some(d) = cur.strip_prefix(prev.as_str()) {
                delta.push_str(d);
                push_wiped(&mut self.full, d);
            }
            self.prefix = self.read;
            self.read = ids.len();
        }
        Ok(delta)
    }
}

/// Append to a wiped string, wiping the old buffer whenever it has to grow
/// (a plain `push_str` reallocation would leave an unwiped copy behind).
fn push_wiped(s: &mut Zeroizing<String>, add: &str) {
    if s.len() + add.len() > s.capacity() {
        let mut grown = String::with_capacity((s.len() + add.len()).max(4096) * 2);
        grown.push_str(s);
        s.zeroize();
        **s = grown;
    }
    s.push_str(add);
}
