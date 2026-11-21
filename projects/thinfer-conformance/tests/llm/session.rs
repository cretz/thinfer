//! Session state round trip on the real models: a sequence exported as
//! [`StateChunk`]s and imported into a fresh state must continue exactly as
//! the uninterrupted state does (same logits, same greedy tokens). Covers both
//! state shapes: per-position KV rows (Gemma 4, Qwen3.6's attention layers)
//! and fixed recurrent state (Qwen3.6's Gated DeltaNet layers), over a
//! two-link chain (root + child), with the continuation rendered by
//! `chat::continuation` as the app does.

#![cfg(feature = "llm-e2e")]

use std::future::Future;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::manifest::ModelManifest;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::tokenizer::Tokenizer;
use thinfer_core::workspace::Workspace;
use thinfer_models::llm::chat::{ChatMessage, continuation, render};
use thinfer_models::llm::decoder::{Blocks, Decoder, LlmState, Want};
use thinfer_models::llm::gemma4::{Gemma4Config, Gemma4Model};
use thinfer_models::llm::kernels::LlmPipelines;
use thinfer_models::llm::manifest::{GEMMA4_MANIFEST, QWEN36_MANIFEST, role};
use thinfer_models::llm::qwen35::{Qwen35Config, Qwen35Model};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;
use thinfer_native::tokenizer::HfTokenizer;

use crate::parity::Source;
use crate::util::backend;

/// Greedy tokens decoded per turn.
const TURN: usize = 12;

fn config(manifest: &ModelManifest) -> serde_json::Value {
    let p = cache::resolve(manifest.get(role::CONFIG).expect("role")).expect("config cached");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn gemma4_session() {
    run(
        "gemma4_session",
        &GEMMA4_MANIFEST,
        |backend, source, budget, max_ctx| async move {
            let cfg = Gemma4Config::from_json(&config(&GEMMA4_MANIFEST)).expect("config");
            Gemma4Model::load(&backend, source, cfg, budget, max_ctx)
                .await
                .expect("load")
        },
    )
    .await;
}

#[tokio::test(flavor = "current_thread")]
async fn qwen36_session() {
    run(
        "qwen36_session",
        &QWEN36_MANIFEST,
        |backend, source, budget, max_ctx| async move {
            let cfg = Qwen35Config::from_json(&config(&QWEN36_MANIFEST)).expect("config");
            Qwen35Model::load(&backend, source, cfg, budget, max_ctx).expect("load")
        },
    )
    .await;
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}

async fn run<B, F, Fut>(name: &str, manifest: &ModelManifest, load: F)
where
    B: Blocks,
    F: FnOnce(std::sync::Arc<WgpuBackend>, Source, ResidencyBudget, u32) -> Fut,
    Fut: Future<Output = Decoder<Source, B>>,
{
    let resolve = |r: &str| cache::resolve(manifest.get(r).expect("role"));
    let (Some(gguf), Some(tok), Some(template)) = (
        resolve(role::GGUF),
        resolve(role::TOKENIZER),
        resolve(role::CHAT_TEMPLATE),
    ) else {
        eprintln!("skipped[{name}]: model files not in HF cache");
        return;
    };
    let template = std::fs::read_to_string(template).expect("template");
    let tokenizer = HfTokenizer::from_path(&tok).await.expect("tokenizer");
    let encode = |s: &str| tokenizer.encode(s, false).expect("encode");
    // BOS-less render is fine here: both paths see the same tokens.
    let first = encode(
        &render(
            &template,
            &[ChatMessage::new("user", "Name three colors, briefly.")],
            &[],
            false,
            "",
        )
        .expect("render"),
    );
    let next = encode(
        &continuation(
            &template,
            &[ChatMessage::new("user", "Now three animals.")],
            &[],
            "",
            false,
            "",
        )
        .expect("continuation"),
    );

    let backend = backend().await;
    let pipes = LlmPipelines::compile(&backend).await.expect("pipelines");
    let source = GgufSource::open(MmapFileOpener::new(&gguf).await.expect("open"))
        .await
        .expect("parse gguf");
    let max_ctx = (first.len() + next.len() + 3 * TURN + 8) as u32;
    let budget = ResidencyBudget {
        ram_bytes: crate::util::ram_budget(),
        vram_bytes: thinfer_native::gpu_budget::resident_vram_budget(
            backend.wgpu_device(),
            5 << 30,
        ),
    };
    let mut model = load(backend.clone(), source, budget, max_ctx).await;
    let ws = Workspace::new(backend.clone(), model.residency().arbiter().clone());
    let accel = model.enable_accel(&backend).expect("accel");
    eprintln!(
        "[{name}] accel {accel}, prompt {} + {} tokens",
        first.len(),
        next.len()
    );
    let want = Want {
        logits: true,
        ..Default::default()
    };

    // Feed `tokens`, then decode TURN greedy tokens; returns (logits after
    // the feed, decoded tokens).
    macro_rules! turn {
        ($st:expr, $tokens:expr) => {{
            let mut logits = model
                .forward(&backend, &pipes, &ws, $st, $tokens, want)
                .await
                .expect("forward")
                .logits
                .to_vec();
            let fed = logits.clone();
            let mut out = Vec::new();
            for _ in 0..TURN {
                let t = argmax(&logits);
                out.push(t);
                logits = model
                    .forward(&backend, &pipes, &ws, $st, &[t], want)
                    .await
                    .expect("decode")
                    .logits
                    .to_vec();
            }
            (fed, out)
        }};
    }
    let fresh = |model: &Decoder<Source, B>| -> LlmState {
        model.new_state(&backend, &ws, max_ctx).expect("state")
    };

    // Uninterrupted: turn 1, turn 2, then one more token.
    let mut a = fresh(&model);
    let (_, t1) = turn!(&mut a, &first);
    let root = a.export(0).await.expect("export root");
    let (ref2, t2) = turn!(&mut a, &next);
    let child = a.export(root.end).await.expect("export child");
    let probe = [t2[TURN - 1]];
    let ref3 = model
        .forward(&backend, &pipes, &ws, &mut a, &probe, want)
        .await
        .expect("probe")
        .logits
        .to_vec();
    drop(a);

    // Resumed from the root chunk: turn 2 must match.
    let mut b = fresh(&model);
    b.import(&[&root]).expect("import root");
    assert_eq!(b.pos, root.end);
    let (got2, u2) = turn!(&mut b, &next);
    drop(b);
    // Resumed from the root + child chain: the probe must match.
    let mut c = fresh(&model);
    c.import(&[&root, &child]).expect("import chain");
    let got3 = model
        .forward(&backend, &pipes, &ws, &mut c, &probe, want)
        .await
        .expect("probe")
        .logits
        .to_vec();

    // Panics on non-finite input: `f32::max` would skip a NaN.
    let max_diff = |x: &[f32], y: &[f32]| {
        assert!(
            x.iter().chain(y).all(|v| v.is_finite()),
            "[{name}] non-finite logits"
        );
        x.iter()
            .zip(y)
            .map(|(p, q)| (p - q).abs())
            .fold(0f32, f32::max)
    };
    let (d2, d3) = (max_diff(&ref2, &got2), max_diff(&ref3, &got3));
    eprintln!(
        "[{name}] root {} rows ({:.1} MB), child {} rows ({:.1} MB); turn1 {:?}",
        root.end,
        root.bytes.len() as f64 / 1e6,
        child.end - child.start,
        child.bytes.len() as f64 / 1e6,
        tokenizer.decode(&t1, false).unwrap_or_default(),
    );
    eprintln!(
        "[{name}] max |logit diff| resume-root {d2:e}, resume-chain {d3:e}; turn2 equal {}",
        t2 == u2
    );
    assert_eq!(t2, u2, "resumed greedy continuation differs");
    assert_eq!(argmax(&ref3), argmax(&got3));
    assert!(
        d2 <= 1e-3 && d3 <= 1e-3,
        "resumed logits drift: {d2} / {d3}"
    );
}
