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
/// The engine's chat rendering (its session `Tail` is shared with the server).
pub use thinfer_models::llm::chat;
use thinfer_models::llm::decoder::{Blocks, Decoder, MAX_STEP, Want};
use thinfer_models::llm::experts::HostTier;
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
use crate::model::TextModelEngine;
use crate::model::TextModelId;
use crate::progress::ProgressSink;
use crate::request::{JobSummary, SessionSave, TextRequest, TextSession, TextStats};
use crate::text_format::{self, Constraint};
use crate::tools::{self, FinishReason, Syntax};

/// The engine's chat turn for a wire message.
pub fn chat_message(m: thinfer_client::wire::TextMessage) -> chat::ChatMessage {
    let mut out = chat::ChatMessage::new(&m.role, m.content);
    out.tool_calls = m
        .tool_calls
        .into_iter()
        .map(|mut c| chat::ChatToolCall {
            id: std::mem::take(&mut c.id),
            name: std::mem::take(&mut c.name),
            arguments: c.arguments.take(),
        })
        .collect();
    out.tool_call_id = m.tool_call_id;
    out.name = m.name;
    out
}

/// Prompt tokens per prefill forward: one decoder step (bounds the
/// activation workspace: a step's routed-expert pairs are `step * top_k` rows).
const PREFILL_CHUNK: usize = MAX_STEP as usize;

/// A text model kept loaded between jobs on a warm device (see
/// [`crate::LocalExecutor`]): its parsed tokenizer, template and configs, the
/// constraint's tokenizer view, and (once a job loaded it) the compiled
/// pipelines and the decoder with its resident weights and expert pool.
/// Pristine model data only: every request's tokens, state and reply live in
/// the job and go with it.
///
/// [`Self::release_device`] frees the model's VRAM but keeps its host side
/// (see [`Parked`]), so the next text job skips the host work.
pub struct WarmLlm {
    model: TextModelId,
    assets: Assets,
    loaded: Option<Loaded>,
    parked: Option<Parked>,
    /// When a text job last finished on it.
    last_used: Instant,
}

impl WarmLlm {
    /// Free the loaded model's VRAM, keeping what the host holds for it:
    /// the parsed assets, the compiled pipelines and the page-locked expert
    /// cache.
    pub fn release_device(&mut self) {
        if let Some(l) = self.loaded.take() {
            self.parked = Some(l.park());
        }
    }

    /// Its model is on the device (not released to its host side).
    pub fn is_loaded(&self) -> bool {
        self.loaded.is_some()
    }

    /// Time since a text job last used it.
    pub fn idle(&self) -> std::time::Duration {
        self.last_used.elapsed()
    }

    /// Page-locked host bytes held for it (the expert cache).
    pub fn host_bytes(&self) -> u64 {
        self.parked
            .as_ref()
            .and_then(|p| p.host.as_ref())
            .map_or(0, HostTier::bytes)
    }
}

/// A released model's host side, for the next load of the same weights:
/// its compiled pipelines (no buffers) and page-locked expert cache.
struct Parked {
    weights: String,
    pipes: LlmPipelines,
    host: Option<HostTier>,
}

/// A model's host-side files, parsed.
struct Assets {
    tokenizer_json: std::path::PathBuf,
    tokenizer: HfTokenizer,
    template: String,
    config: serde_json::Value,
    gen_cfg: serde_json::Value,
    /// The constraint's tokenizer view (built on the first grammar job).
    env: Option<Arc<llguidance::toktrie::TokEnv>>,
}

impl Assets {
    async fn load(model: TextModelId) -> Result<Self, String> {
        let manifest = model.manifest();
        let tokenizer_json = resolve_role(manifest, role::TOKENIZER)?;
        let tokenizer = HfTokenizer::from_path(&tokenizer_json)
            .await
            .map_err(|e| format!("load tokenizer: {e:?}"))?;
        let template = std::fs::read_to_string(resolve_role(manifest, role::CHAT_TEMPLATE)?)
            .map_err(|e| format!("read chat template: {e}"))?;
        Ok(Self {
            tokenizer_json,
            tokenizer,
            template,
            config: read_json(&resolve_role(manifest, role::CONFIG)?)?,
            gen_cfg: read_json(&resolve_role(manifest, role::GENERATION_CONFIG)?)?,
            env: None,
        })
    }
}

/// A loaded decoder and what it was sized for.
struct Loaded {
    weights: String,
    /// The requested budget it was sized from.
    budget: thinfer_core::policy::ResidencyBudget,
    max_ctx: u32,
    pipes: LlmPipelines,
    model: AnyModel,
}

impl Loaded {
    /// Free the model's VRAM; keep its host side.
    fn park(self) -> Parked {
        let Loaded {
            weights,
            pipes,
            mut model,
            ..
        } = self;
        let host = match &mut model {
            AnyModel::Gemma4(m) => m.take_host_tier(),
            AnyModel::Qwen35(m) => m.take_host_tier(),
        };
        Parked {
            weights,
            pipes,
            host,
        }
    }
}

enum AnyModel {
    Gemma4(Gemma4Model<GgufSource<MmapFileOpener>>),
    Qwen35(Qwen35Model<GgufSource<MmapFileOpener>>),
}

/// A loaded model's context, rounded up so a conversation that grows a
/// little (a session's next turn) still fits it.
const CTX_STEP: u32 = 1024;

pub async fn run(
    backend: &Arc<WgpuBackend>,
    req: &TextRequest,
    sink: &dyn ProgressSink,
    warm: &mut Option<WarmLlm>,
) -> Result<JobSummary, String> {
    req.validate()?;
    // Another model's warm state goes first (its VRAM and RAM).
    if warm.as_ref().is_some_and(|w| w.model != req.model) {
        *warm = None;
    }
    let mut w = match warm.take() {
        Some(w) => w,
        None => WarmLlm {
            model: req.model,
            assets: Assets::load(req.model).await?,
            loaded: None,
            parked: None,
            last_used: Instant::now(),
        },
    };
    let out = run_with(backend, req, sink, &mut w).await;
    // Kept only after a clean job: a failure may leave the model mid-update.
    if out.is_ok() {
        w.last_used = Instant::now();
        *warm = Some(w);
    }
    out
}

async fn run_with(
    backend: &Arc<WgpuBackend>,
    req: &TextRequest,
    sink: &dyn ProgressSink,
    w: &mut WarmLlm,
) -> Result<JobSummary, String> {
    let manifest = req.model.manifest();
    let WarmLlm {
        assets,
        loaded,
        parked,
        ..
    } = w;
    let Assets {
        tokenizer_json,
        tokenizer,
        template,
        config,
        gen_cfg,
        env: env_cache,
    } = assets;
    let stop = eos_ids(gen_cfg);
    let syntax = Syntax::of(req.model, |t| single_token(tokenizer, t))?;
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
        (true, Some(f)) => Some((text_format::top_level(f), text_format::REJECTED)),
        (true, None) => None,
    };
    // The constraint's view of the tokenizer, built (and the grammar
    // compiled) before any weights load: a bad schema fails fast.
    let grammar_env = match grammar {
        Some((g, rejected)) => {
            let env = match env_cache {
                Some(e) => e.clone(),
                None => env_cache
                    .insert(Arc::new(text_format::tok_env(tokenizer_json, None, &stop)?))
                    .clone(),
            };
            let gate = if req.think {
                Some(single_token(tokenizer, req.model.think_close())?)
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
    // A prefix-only request (`max_tokens` 0) renders no generation prompt:
    // later requests continue its turns.
    let prefix_only = req.max_tokens == 0;
    let mut prompt = match &req.session {
        Some(s) if resume_at > 0 => chat::resume(
            template,
            &s.tail,
            &req.messages,
            &tool_defs,
            syntax.call_close(),
            !prefix_only,
            req.think,
            &bos,
        )?,
        _ if prefix_only => {
            chat::render_prefix(template, &req.messages, &tool_defs, req.think, &bos)?
        }
        _ => chat::render(template, &req.messages, &tool_defs, req.think, &bos)?,
    };
    if !prefix_only && req.messages.last().is_some_and(|m| m.role == "tool") {
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
    // The loaded model when it is the same weights, requested budget and big
    // enough; else it goes (its memory first) and the model loads fresh.
    let fits = loaded
        .as_ref()
        .is_some_and(|l| l.weights == weights && l.budget == req.budget && l.max_ctx >= max_ctx);
    if fits {
        sink.note(&format!(
            "Using loaded {} ({} prompt tokens)",
            req.model,
            ids.len()
        ));
    } else {
        // The old model's VRAM goes; its host side (or the parked one) is
        // reused when it is for these weights.
        let carried = loaded
            .take()
            .map(Loaded::park)
            .or_else(|| parked.take())
            .filter(|p| p.weights == weights);
        *parked = None;
        // Inside what the OS keeps resident (measured with the old model
        // gone): past it allocations are demoted to system memory and every
        // read crosses PCIe.
        let mut budget = req.budget;
        budget.vram_bytes = thinfer_native::gpu_budget::resident_vram_budget(
            backend.wgpu_device(),
            budget.vram_bytes,
        );
        if budget.vram_bytes < req.budget.vram_bytes {
            tracing::info!(
                target: thinfer_core::trace::DIAG,
                requested = req.budget.vram_bytes, resident = budget.vram_bytes,
                "vram budget lowered to the OS-resident headroom",
            );
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
        let (pipes, host) = match carried {
            Some(p) => {
                tracing::info!(
                    target: thinfer_core::trace::DIAG,
                    host_bytes = p.host.as_ref().map_or(0, HostTier::bytes),
                    "llm host side reused",
                );
                (p.pipes, p.host)
            }
            None => (
                LlmPipelines::compile(backend)
                    .await
                    .map_err(|e| format!("compile llm pipelines: {e:?}"))?,
                None,
            ),
        };
        let load_err = |e| format!("model load: {e:?}");
        let ctx = max_ctx.div_ceil(CTX_STEP) * CTX_STEP;
        let mut model = match req.model {
            TextModelId::Qwen36_35bA3b => AnyModel::Qwen35(
                Qwen35Model::load(
                    backend,
                    source,
                    Qwen35Config::from_json(config)?,
                    budget,
                    ctx,
                )
                .map_err(load_err)?,
            ),
            TextModelId::Gemma4_26bA4b => AnyModel::Gemma4(
                Gemma4Model::load(
                    backend,
                    source,
                    Gemma4Config::from_json(config)?,
                    budget,
                    ctx,
                )
                .await
                .map_err(load_err)?,
            ),
        };
        let accel = match &mut model {
            AnyModel::Gemma4(m) => m.enable_accel(backend, host),
            AnyModel::Qwen35(m) => m.enable_accel(backend, host),
        };
        accel.map_err(|e| format!("enable accelerator: {e:?}"))?;
        *loaded = Some(Loaded {
            weights: weights.clone(),
            budget: req.budget,
            max_ctx: ctx,
            pipes,
            model,
        });
    }
    let Loaded { pipes, model, .. } = loaded.as_mut().expect("loaded above");
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
    match model {
        AnyModel::Qwen35(m) => job.generate(m).await,
        AnyModel::Gemma4(m) => job.generate(m).await,
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
    pipes: &'a LlmPipelines,
    /// Prompt token ids (job data).
    ids: Locked<u32>,
    stop: Vec<u32>,
    tokenizer: &'a HfTokenizer,
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
        model: &mut Decoder<GgufSource<MmapFileOpener>, B>,
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
            .enable_accel(backend, None)
            .map_err(|e| format!("enable accelerator: {e:?}"))?;
        tracing::info!(target: thinfer_core::trace::DIAG, accel, "llm path");
        let ws = Workspace::new(backend.clone(), model.residency().arbiter().clone());
        let mut state = model
            .new_state(backend, &ws, max_ctx)
            .map_err(|e| format!("state: {e:?}"))?;
        // Dropped before the state and workspace (declared after them): on
        // an early exit it waits for queued CUDA work on their memory.
        let _drain = AccelDrain(backend.clone());
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
            // Logits for the last chunk only, and none for a prefix (nothing
            // is sampled from it).
            let last = (i + 1) * PREFILL_CHUNK >= ids.len() && req.max_tokens > 0;
            let out = model
                .forward(
                    backend,
                    pipes,
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
            // A prefix-only request (`max_tokens` 0) generates nothing.
            if req.max_tokens == 0 {
                break;
            }
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
            sink.text(&text.push(tokenizer, &produced[..calls_at.unwrap_or(n)])?);
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
                        pipes,
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
        sink.text(&text.flush(tokenizer, &produced[..calls_at.unwrap_or(n)])?);
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
            Some(s) => Some(Arc::new(SessionSave {
                chunk: state
                    .export(resume_at)
                    .await
                    .map_err(|e| format!("save session: {e:?}"))?,
                weights,
                tail: new_tail(req, s, resume_at > 0),
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

/// Waits for the accelerator's queued work when dropped. CUDA kernels are
/// enqueued without a wait, so an early exit (cancel, error) can leave some
/// reading or writing buffers that are about to be freed.
struct AccelDrain(Arc<WgpuBackend>);

impl Drop for AccelDrain {
    fn drop(&mut self) {
        use thinfer_core::backend::Backend;
        if let Some(acc) = self.0.accelerator()
            && let Err(e) = acc.join_ahead().and_then(|()| acc.sync())
        {
            tracing::warn!(target: thinfer_core::trace::ADAPTER, "accelerator drain failed: {e}");
        }
    }
}

/// How a session request's new node ends: after its reply, or (a prefix-only
/// request) after its turns, extending an open parent's turns.
fn new_tail(req: &TextRequest, session: &TextSession, resumed: bool) -> chat::Tail {
    if req.max_tokens > 0 {
        return chat::Tail::Reply;
    }
    let mut roles: Vec<String> = Vec::new();
    let mut after_reply = false;
    if resumed {
        match &session.tail {
            chat::Tail::Reply => after_reply = true,
            chat::Tail::Open {
                roles: r,
                after_reply: a,
            } => {
                roles.extend(r.iter().cloned());
                after_reply = *a;
            }
        }
    }
    roles.extend(req.messages.iter().map(|m| m.role.clone()));
    chat::Tail::Open { roles, after_reply }
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
