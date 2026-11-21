//! Shared LLM end-to-end perf bench: a ~100-word user prompt through the
//! model's own chat template, timed prefill (time to first token) then greedy
//! decode, then the same prefill again with the experts cached (the compute
//! floor without disk). Prints tokens/s, phase splits, expert cache stats and
//! the decoded text (sanity). `THINFER_TRACE=1` adds the per-pipeline GPU
//! rollup; `THINFER_LLM_BENCH_TOKENS` / `THINFER_LLM_BENCH_VRAM_MB` /
//! `THINFER_LLM_RAM_GB` set the decode length and budgets;
//! `THINFER_LLM_BENCH_REPEAT` repeats the prompt (a long chat's prefill).

#![cfg(feature = "llm-e2e")]

use std::future::Future;
use std::time::Instant;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::manifest::ModelManifest;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::tokenizer::Tokenizer;
use thinfer_core::workspace::Workspace;
use thinfer_models::llm::chat::{ChatMessage, render};
use thinfer_models::llm::decoder::{Blocks, Decoder, Want};
use thinfer_models::llm::kernels::LlmPipelines;
use thinfer_models::llm::manifest::role;
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;
use thinfer_native::tokenizer::HfTokenizer;

use crate::parity::Source;
use crate::util::backend;

const PROMPT: &str = "I am planning a three day trip to Kyoto in late autumn with my partner, who \
loves gardens and quiet temples, while I am more interested in food markets, small bars and old \
wooden neighborhoods. We will arrive by train from Tokyo on a Thursday morning and leave on Sunday \
evening. We do not want to rush, and we prefer walking or buses over taxis. Please suggest a \
relaxed day by day plan that balances both of our interests, mentions roughly how long to spend \
in each place, and includes one or two tips for avoiding the biggest crowds during the peak \
foliage season.";

fn env_or<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

pub async fn run<B, F, Fut>(name: &str, manifest: &ModelManifest, load: F)
where
    B: Blocks,
    F: FnOnce(std::sync::Arc<WgpuBackend>, Source, ResidencyBudget, u32) -> Fut,
    Fut: Future<Output = Decoder<Source, B>>,
{
    let _trace = thinfer_core::trace::init_from_env();
    let resolve = |r: &str| cache::resolve(manifest.get(r).expect("role"));
    let (Some(gguf), Some(tok), Some(template), Some(gen_cfg)) = (
        resolve(role::GGUF),
        resolve(role::TOKENIZER),
        resolve(role::CHAT_TEMPLATE),
        resolve(role::GENERATION_CONFIG),
    ) else {
        eprintln!("skipped[{name}]: model files not in HF cache");
        return;
    };
    let new_tokens: usize = env_or("THINFER_LLM_BENCH_TOKENS", 64);
    let vram: u64 = env_or("THINFER_LLM_BENCH_VRAM_MB", 5632);
    // A long chat's re-prefill: the prompt repeated (~125 tokens each).
    let prompt = vec![PROMPT; env_or("THINFER_LLM_BENCH_REPEAT", 1usize).max(1)].join(
        "

",
    );
    let tokenizer = HfTokenizer::from_path(&tok).await.expect("tokenizer");
    let gen_cfg_v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(gen_cfg).unwrap()).unwrap();
    let ids_of = |k: &str| -> Vec<u32> {
        match gen_cfg_v.get(k) {
            Some(serde_json::Value::Number(n)) => vec![n.as_u64().unwrap() as u32],
            Some(serde_json::Value::Array(a)) => a
                .iter()
                .filter_map(|v| v.as_u64())
                .map(|v| v as u32)
                .collect(),
            _ => Vec::new(),
        }
    };
    let stop = ids_of("eos_token_id");
    let bos = match ids_of("bos_token_id").first() {
        Some(&id) => tokenizer.decode(&[id], false).expect("bos"),
        None => String::new(),
    };
    let chat = render(
        &std::fs::read_to_string(template).expect("read template"),
        &[ChatMessage::new("user", &prompt)],
        &[],
        false,
        &bos,
    )
    .expect("render");
    let ids = tokenizer.encode(&chat, false).expect("encode");

    let backend = backend().await;
    let pipes = LlmPipelines::compile(&backend).await.expect("pipelines");
    let source = GgufSource::open(MmapFileOpener::new(&gguf).await.expect("open"))
        .await
        .expect("parse gguf");
    let max_ctx = (ids.len() + new_tokens + 8) as u32;
    let t_load = Instant::now();
    // As the app does: inside what the OS keeps resident.
    let budget = ResidencyBudget {
        ram_bytes: crate::util::ram_budget(),
        vram_bytes: thinfer_native::gpu_budget::resident_vram_budget(
            backend.wgpu_device(),
            vram << 20,
        ),
    };
    eprintln!(
        "vram budget: {} MiB (requested {vram})",
        budget.vram_bytes >> 20
    );
    let mut model = load(backend.clone(), source, budget, max_ctx).await;
    let ws = Workspace::new(backend.clone(), model.residency().arbiter().clone());
    let accel = model.enable_accel(&backend, None).expect("accel");
    eprintln!(
        "accel: {accel}, experts {:.1} GB, {} VRAM slots",
        model.experts_total_bytes() as f64 / 1e9,
        model.experts.capacity()
    );
    let mut st = model.new_state(&backend, &ws, max_ctx).expect("state");
    eprintln!("load: {:.2}s", t_load.elapsed().as_secs_f64());

    let logits_only = Want {
        logits: true,
        ..Default::default()
    };
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32
    };
    let t0 = Instant::now();
    let out = model
        .forward(&backend, &pipes, &ws, &mut st, &ids, logits_only)
        .await
        .expect("prefill");
    let ttft = t0.elapsed().as_secs_f64();
    eprintln!("prefill phases: {:?}", model.times);
    model.times = Default::default();
    let hs = model.experts.host_stats();
    eprintln!(
        "prefill host cache: {hs:?} ({:.2} GB/s source reads)",
        hs.read_bytes as f64 / hs.read_secs.max(1e-9) / 1e9
    );
    let (f0, h0) = (model.experts.fills, model.experts.hits);
    eprintln!(
        "prefill {} tokens: {ttft:.2}s ({:.1} tok/s), expert fills {f0} hits {h0}",
        ids.len(),
        ids.len() as f64 / ttft
    );

    let mut next = argmax(&out.logits);
    let mut produced = vec![next];
    let t1 = Instant::now();
    for _ in 1..new_tokens {
        let out = model
            .forward(&backend, &pipes, &ws, &mut st, &[next], logits_only)
            .await
            .expect("decode");
        next = argmax(&out.logits);
        produced.push(next);
        if stop.contains(&next) {
            break;
        }
    }
    let dt = t1.elapsed().as_secs_f64();
    eprintln!("decode phases: {:?}", model.times);
    let n = produced.len() - 1;
    let (f1, h1) = (model.experts.fills - f0, model.experts.hits - h0);
    eprintln!(
        "decode {n} tokens: {dt:.2}s ({:.2} tok/s, {:.1} ms/token), expert fills {f1} hits {h1} (hit rate {:.1}%)",
        n as f64 / dt,
        dt * 1000.0 / n as f64,
        100.0 * h1 as f64 / (h1 + f1).max(1) as f64
    );
    eprintln!(
        "vram ahead: {} copied, {} used",
        model.experts.ahead, model.experts.ahead_used
    );
    let hs = model.experts.host_stats();
    eprintln!(
        "host cache: {hs:?} ({:.2} GB/s source reads)",
        hs.read_bytes as f64 / hs.read_secs.max(1e-9) / 1e9
    );
    eprintln!(
        "text: {}",
        tokenizer.decode(&produced, true).unwrap_or_default()
    );
    // The same prefill again on a fresh state, experts now cached in RAM /
    // VRAM: the compute floor without disk reads. The decoded state is
    // dropped first: the expert pool leaves room for one state only.
    model.times = Default::default();
    drop(st);
    let mut st = model.new_state(&backend, &ws, max_ctx).expect("state");
    let t2 = Instant::now();
    model
        .forward(&backend, &pipes, &ws, &mut st, &ids, logits_only)
        .await
        .expect("warm prefill");
    eprintln!(
        "warm prefill {} tokens: {:.2}s, phases {:?}",
        ids.len(),
        t2.elapsed().as_secs_f64(),
        model.times
    );
    if let Some(h) = _trace {
        h.dump(&mut std::io::stderr()).expect("rollup");
    }
}
