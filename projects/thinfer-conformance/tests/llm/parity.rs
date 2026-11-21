//! Shared LLM parity run: the engine's per-layer residual stream and last-row
//! logits vs a transformers reference (`thinfer_pytorch_ref.llm.<module>`)
//! over the SAME GGUF (bf16, one layer resident). Two engine runs: one
//! prefill of the whole prompt (every row compared), and prefill of all but
//! the last token + one decode step (the last row compared), which exercises
//! the KV cache and any recurrent state carry.

#![cfg(feature = "llm-e2e")]

use std::future::Future;
use std::path::{Path, PathBuf};
use std::process::Command;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::manifest::ModelManifest;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::workspace::Workspace;
use thinfer_models::llm::decoder::{Blocks, Decoder, Want};
use thinfer_models::llm::kernels::LlmPipelines;
use thinfer_models::llm::manifest::role;
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::util::{backend, compare, read_f32, read_i32};

pub type Source = GgufSource<MmapFileOpener>;

/// Tolerances for f32 engine acts over quantized weights vs the bf16
/// reference over the same dequantized weights.
pub struct Tol {
    /// Bound on a layer's MEDIAN row relative error. A row whose 8th expert
    /// is a near tie (weights ~1e-3 apart) can flip on the reference's bf16
    /// rounding and diverge from there on (a step in that row's error, flat
    /// before); a real defect moves every row, so the median gates.
    pub layer_rel: f64,
    pub logit_cos: f64,
}

/// Run parity for one model. `load` builds the engine from the GGUF source,
/// budget and max context. Skips (returns) when the files are not cached.
pub async fn run<B, F, Fut>(name: &str, manifest: &ModelManifest, module: &str, tol: Tol, load: F)
where
    B: Blocks,
    F: FnOnce(std::sync::Arc<WgpuBackend>, Source, ResidencyBudget, u32) -> Fut,
    Fut: Future<Output = Decoder<Source, B>>,
{
    let _trace = thinfer_core::trace::init_from_env();
    let resolve = |r: &str| cache::resolve(manifest.get(r).expect("role"));
    let (Some(gguf), Some(config), Some(tok)) = (
        resolve(role::GGUF),
        resolve(role::CONFIG),
        resolve(role::TOKENIZER),
    ) else {
        eprintln!("skipped[{name}]: model files not in HF cache");
        return;
    };
    let layers: u32 = std::env::var("THINFER_LLM_PARITY_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("config.txt");
    let tag = format!(
        "{} layers={layers}",
        gguf.file_name().unwrap().to_string_lossy()
    );
    if std::fs::read_to_string(&marker).is_ok_and(|c| c == tag) {
        eprintln!("{name}: reusing cached pyref dumps ({})", tmp.display());
    } else {
        run_python_ref(module, &gguf, &config, &tok, &tmp, layers);
        std::fs::write(&marker, &tag).expect("write marker");
    }
    let meta = std::fs::read_to_string(tmp.join("meta.txt")).expect("meta");
    let m: Vec<usize> = meta
        .split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect();
    let (t, hidden, ref_layers) = (m[0], m[1], m[2]);
    let ids: Vec<u32> = read_i32(&tmp.join("ids.bin"))
        .iter()
        .map(|&i| i as u32)
        .collect();
    assert_eq!(ids.len(), t);

    let backend = backend().await;
    let pipes = LlmPipelines::compile(&backend).await.expect("pipelines");
    let source = GgufSource::open(MmapFileOpener::new(&gguf).await.expect("open"))
        .await
        .expect("parse gguf");
    let max_ctx = 256;
    let budget = ResidencyBudget {
        ram_bytes: crate::util::ram_budget(),
        vram_bytes: 5632 << 20,
    };
    let mut model = load(backend.clone(), source, budget, max_ctx).await;
    let dims = model.blocks.dims().clone();
    assert_eq!(dims.hidden as usize, hidden);
    let ws = Workspace::new(backend.clone(), model.residency().arbiter().clone());
    let t_host = std::time::Instant::now();
    let accel = model.enable_accel(&backend, None).expect("accel");
    eprintln!(
        "accel: {accel}, experts {:.1} GB ({:.2}s)",
        model.experts_total_bytes() as f64 / 1e9,
        t_host.elapsed().as_secs_f64()
    );
    let full = ref_layers == dims.layers as usize;
    let want = Want {
        layer_taps: true,
        logits: full,
        diag_layer: std::env::var("THINFER_LLM_DIAG_LAYER")
            .ok()
            .and_then(|v| v.parse().ok()),
    };

    // Prefill: every row of every compared layer.
    let mut st = model.new_state(&backend, &ws, max_ctx).expect("state");
    let t0 = std::time::Instant::now();
    let prefill = model
        .forward(&backend, &pipes, &ws, &mut st, &ids, want)
        .await
        .expect("prefill");
    let out = &prefill;
    eprintln!("prefill {t} tokens: {:.2}s", t0.elapsed().as_secs_f64());
    dump_diag(&tmp, "prefill", &out.diag);
    // THINFER_LLM_DUMP_TAPS: the engine's prefill taps next to the reference's.
    if std::env::var("THINFER_LLM_DUMP_TAPS").is_ok() {
        for (i, tap) in out.layer_taps.iter().enumerate() {
            std::fs::write(
                tmp.join(format!("engine_layer{i}.bin")),
                bytemuck::cast_slice(tap),
            )
            .expect("write tap");
        }
    }
    let mut fails = Vec::new();
    for (i, tap) in out.layer_taps.iter().take(ref_layers).enumerate() {
        let exp = read_f32(&tmp.join(format!("layer{i}.bin")));
        compare(&format!("prefill layer {i}"), &exp, tap);
        let med = median_row_rel(&exp, tap, hidden);
        if med > tol.layer_rel {
            fails.push(format!(
                "prefill layer {i}: median row rel {:.3}%",
                med * 100.0
            ));
        }
    }
    if full {
        check_logits(&tmp, &out.logits, "prefill", tol.logit_cos, &mut fails);
    }

    // Prefill t-1 + decode 1: the last row. Gated against the engine's own
    // full prefill (same weights and math through the KV cache / recurrent
    // state instead of in-scope rows; the prefill is gated against the
    // reference above), reported against the reference.
    let mut st = model.new_state(&backend, &ws, max_ctx).expect("state");
    model
        .forward(
            &backend,
            &pipes,
            &ws,
            &mut st,
            &ids[..t - 1],
            Want::default(),
        )
        .await
        .expect("prefill t-1");
    let t0 = std::time::Instant::now();
    let out = model
        .forward(&backend, &pipes, &ws, &mut st, &ids[t - 1..], want)
        .await
        .expect("decode");
    eprintln!("decode 1 token: {:.3}s", t0.elapsed().as_secs_f64());
    dump_diag(&tmp, "decode", &out.diag);
    for (i, tap) in out.layer_taps.iter().take(ref_layers).enumerate() {
        let exp = read_f32(&tmp.join(format!("layer{i}.bin")));
        compare(&format!("decode layer {i}"), &exp[(t - 1) * hidden..], tap);
        let own = &prefill.layer_taps[i][(t - 1) * hidden..];
        let (rel, _) = compare(&format!("decode layer {i} vs prefill"), own, tap);
        if rel > DECODE_VS_PREFILL_REL {
            fails.push(format!(
                "decode layer {i} vs prefill: rel {:.4}%",
                rel * 100.0
            ));
        }
    }
    if full {
        check_logits(&tmp, &out.logits, "decode", tol.logit_cos, &mut fails);
    }
    eprintln!(
        "experts: {} fills, {} hits, {} slots",
        model.experts.fills,
        model.experts.hits,
        model.experts.capacity()
    );
    assert!(fails.is_empty(), "{name} parity:\n{}", fails.join("\n"));
}

/// Decode vs the engine's own prefill, last row: only kernel rounding differs
/// (one-row vs multi-row matmuls), unless a near-tied expert flips between them.
const DECODE_VS_PREFILL_REL: f64 = 0.01;

/// Median over rows of `|got - exp| / |exp|` (rows of `width`).
fn median_row_rel(exp: &[f32], got: &[f32], width: usize) -> f64 {
    let mut rels: Vec<f64> = exp
        .chunks(width)
        .zip(got.chunks(width))
        .map(|(e, g)| {
            let num: f64 = e.iter().zip(g).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
            let den: f64 = e.iter().map(|a| (*a as f64).powi(2)).sum();
            (num / den.max(1e-30)).sqrt()
        })
        .collect();
    rels.sort_by(f64::total_cmp);
    rels[rels.len() / 2]
}

/// Write each diag tensor to `<dir>/diag_<run>_<name>.bin` (raw f32 / u32 bits).
fn dump_diag(dir: &Path, run: &str, diag: &[(String, Vec<f32>)]) {
    for (name, v) in diag {
        std::fs::write(
            dir.join(format!("diag_{run}_{name}.bin")),
            bytemuck::cast_slice(v),
        )
        .expect("write diag");
    }
}

fn check_logits(dir: &Path, got: &[f32], label: &str, cos_tol: f64, fails: &mut Vec<String>) {
    let exp = read_f32(&dir.join("logits.bin"));
    let (_, cos) = compare(&format!("{label} logits"), &exp, got);
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    let (e, g) = (argmax(&exp), argmax(got));
    eprintln!("[{label}] top-1 ref {e} engine {g}");
    if cos < cos_tol || e != g {
        fails.push(format!("{label} logits: cos {cos:.5}, top-1 {e} vs {g}"));
    }
}

fn run_python_ref(module: &str, gguf: &Path, config: &Path, tok: &Path, out: &Path, layers: u32) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let mut cmd = Command::new("uv");
    cmd.args(["run", "--directory", py_dir.to_str().unwrap()]);
    for dep in ["gguf", "transformers>=5.8", "tokenizers"] {
        cmd.args(["--with", dep]);
    }
    cmd.args(["python", "-m", &format!("thinfer_pytorch_ref.llm.{module}")]);
    cmd.arg("--gguf").arg(gguf);
    cmd.arg("--config").arg(config);
    cmd.arg("--tokenizer").arg(tok);
    cmd.arg("--out")
        .arg(out)
        .args(["--layers", &layers.to_string()]);
    let status = cmd.status().expect("failed to spawn `uv run`");
    assert!(status.success(), "{module} pyref failed");
}
