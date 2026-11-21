//! Gemma 4's sliding-window KV rings on the real model, over a prompt long
//! enough to wrap them (ring = window + `MAX_STEP` rows): the ring state must
//! reproduce the linear reference layout exactly (same logits, same greedy
//! tokens), and a session chain exported from ring states (window rows
//! trimmed to the last `window`) must resume it exactly.

#![cfg(feature = "llm-e2e")]

use thinfer_core::format::gguf::GgufSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::tokenizer::Tokenizer;
use thinfer_core::workspace::Workspace;
use thinfer_models::llm::chat::{ChatMessage, render};
use thinfer_models::llm::decoder::{LlmState, MAX_STEP, Want};
use thinfer_models::llm::gemma4::{Gemma4Config, Gemma4Model};
use thinfer_models::llm::kernels::LlmPipelines;
use thinfer_models::llm::manifest::{GEMMA4_MANIFEST, role};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;
use thinfer_native::tokenizer::HfTokenizer;

use crate::util::backend;

/// Greedy tokens decoded after the prompt.
const TURN: usize = 12;

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}

/// Largest absolute difference. Panics on non-finite input: `f32::max`
/// would skip a NaN and report no drift.
fn max_diff(x: &[f32], y: &[f32]) -> f32 {
    assert!(
        x.iter().chain(y).all(|v| v.is_finite()),
        "non-finite logits"
    );
    x.iter()
        .zip(y)
        .map(|(p, q)| (p - q).abs())
        .fold(0f32, f32::max)
}

/// A prompt whose answer needs both its start (outside the window by the
/// end) and its end.
fn long_prompt() -> String {
    let colors = ["red", "blue", "green", "amber", "violet", "silver", "teal"];
    let things = [
        "lantern", "harbor", "orchard", "bridge", "library", "market",
    ];
    let mut s = String::from("The secret word is PELICAN. Remember it.\n\n");
    for i in 0..170 {
        s += &format!(
            "Entry {i}: the {} {} opened at hour {}.\n",
            colors[i % colors.len()],
            things[(i * 5) % things.len()],
            (i * 7) % 24
        );
    }
    s + "\nWhat was the secret word, and what opened in entry 169? Answer briefly."
}

#[tokio::test(flavor = "current_thread")]
async fn gemma4_window_ring() {
    let m = &GEMMA4_MANIFEST;
    let resolve = |r: &str| cache::resolve(m.get(r).expect("role"));
    let (Some(gguf), Some(tok), Some(template), Some(cfg)) = (
        resolve(role::GGUF),
        resolve(role::TOKENIZER),
        resolve(role::CHAT_TEMPLATE),
        resolve(role::CONFIG),
    ) else {
        eprintln!("skipped[gemma4_window_ring]: model files not in HF cache");
        return;
    };
    let template = std::fs::read_to_string(template).expect("template");
    let tokenizer = HfTokenizer::from_path(&tok).await.expect("tokenizer");
    let prompt = tokenizer
        .encode(
            &render(
                &template,
                &[ChatMessage::new("user", long_prompt())],
                &[],
                false,
                "",
            )
            .expect("render"),
            false,
        )
        .expect("encode");
    let cfg = Gemma4Config::from_json(
        &serde_json::from_str(&std::fs::read_to_string(cfg).unwrap()).unwrap(),
    )
    .expect("config");
    let window = cfg.window;
    let max_ctx = (prompt.len() + 2 * TURN + 8) as u32;
    assert!(
        prompt.len() as u32 > window + MAX_STEP,
        "prompt ({} tokens) must wrap the {}-row ring",
        prompt.len(),
        window + MAX_STEP
    );

    let backend = backend().await;
    let pipes = LlmPipelines::compile(&backend).await.expect("pipelines");
    let source = GgufSource::open(MmapFileOpener::new(&gguf).await.expect("open"))
        .await
        .expect("parse gguf");
    let budget = ResidencyBudget {
        ram_bytes: crate::util::ram_budget(),
        vram_bytes: thinfer_native::gpu_budget::resident_vram_budget(
            backend.wgpu_device(),
            5 << 30,
        ),
    };
    let mut model = Gemma4Model::load(&backend, source, cfg, budget, max_ctx)
        .await
        .expect("load");
    let ws = Workspace::new(backend.clone(), model.residency().arbiter().clone());
    let accel = model.enable_accel(&backend).expect("accel");
    eprintln!(
        "[ring] accel {accel}, prompt {} tokens, window {window}, ring {}",
        prompt.len(),
        window + MAX_STEP
    );
    let want = Want {
        logits: true,
        ..Default::default()
    };

    // Both layouts run the same steps (the ring would split the prompt into
    // MAX_STEP steps by itself; the reference must match its numerics).
    macro_rules! feed {
        ($st:expr, $tokens:expr) => {{
            let mut logits = Vec::new();
            for step in $tokens.chunks(MAX_STEP as usize) {
                logits = model
                    .forward(&backend, &pipes, &ws, $st, step, want)
                    .await
                    .expect("forward")
                    .logits
                    .to_vec();
            }
            logits
        }};
    }
    macro_rules! decode {
        ($st:expr, $logits:expr) => {{
            let mut logits = $logits;
            let mut out = Vec::new();
            for _ in 0..TURN {
                let t = argmax(&logits);
                out.push(t);
                logits = feed!($st, [t]);
            }
            (out, logits)
        }};
    }

    // Reference: window buffers held whole.
    let mut lin: LlmState = model
        .new_state_unwindowed(&backend, &ws, max_ctx)
        .expect("state");
    let lin_fed = feed!(&mut lin, prompt);
    let lin_root = lin.export(0).await.expect("export");
    let (lin_toks, lin_next) = decode!(&mut lin, lin_fed.clone());
    drop(lin);

    // Ring.
    let mut ring = model.new_state(&backend, &ws, max_ctx).expect("state");
    let ring_fed = feed!(&mut ring, prompt);
    let root = ring.export(0).await.expect("export root");
    let (ring_toks, ring_next) = decode!(&mut ring, ring_fed.clone());
    let child = ring.export(root.end).await.expect("export child");
    // One more token on the uninterrupted ring, then the same on the chain
    // resumed into a fresh ring state.
    let probe = [argmax(&ring_next)];
    let ref_probe = feed!(&mut ring, probe);
    drop(ring);
    let mut resumed = model.new_state(&backend, &ws, max_ctx).expect("state");
    resumed.import(&[&root, &child]).expect("import chain");
    assert_eq!(resumed.pos, child.end);
    let got_probe = feed!(&mut resumed, probe);
    let d_resume = max_diff(&ref_probe, &got_probe);

    let (d_fed, d_next) = (
        max_diff(&lin_fed, &ring_fed),
        max_diff(&lin_next, &ring_next),
    );
    eprintln!(
        "[ring] root chunk {:.1} MB (linear layout {:.1} MB), child {:.1} MB; reply {:?}",
        root.bytes.len() as f64 / 1e6,
        lin_root.bytes.len() as f64 / 1e6,
        child.bytes.len() as f64 / 1e6,
        tokenizer.decode(&ring_toks, false).unwrap_or_default(),
    );
    eprintln!(
        "[ring] max |logit diff| ring vs linear: prompt {d_fed:e}, after decode {d_next:e}; tokens equal {}; resumed chain {d_resume:e}",
        lin_toks == ring_toks
    );
    assert_eq!(lin_toks, ring_toks, "ring greedy tokens differ from linear");
    assert!(
        d_fed == 0.0 && d_next == 0.0,
        "ring changes the numerics: {d_fed} / {d_next}"
    );
    assert_eq!(d_resume, 0.0, "resumed ring chain drifts");
    assert!(
        root.bytes.len() < lin_root.bytes.len(),
        "window rows not trimmed"
    );
}
