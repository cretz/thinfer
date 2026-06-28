//! LTX-2.3 Gemma-3-12B text-encoder parity: the engine encoder (Q8_0 weights from
//! the gemma GGUF + bf16 acts) vs a bf16 reference dequantized from the SAME GGUF
//! (`ltx/gen_encoder_ref.py`, streamed layer-by-layer), fed the SAME tokens.
//!
//! Compares ALL 49 hidden states (embedding layer + 48 decoder layers) that
//! FeatureExtractor V2 consumes. Both sides start from identical Q8_0 bytes, so
//! the band is the engine's bf16-act rounding accumulated over 48 layers plus the
//! Gemma `(1+w)` norm bake (engine bakes +1 in f32 at load; HF folds it in bf16).
//! The residual grows past f16 (~1.9e5) so bf16 acts are mandatory.

#![cfg(feature = "ltx-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::format::union::RenamedSource;
use thinfer_core::ops::WeightDtype;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::quant::QuantKind;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::ltx::gemma;
use thinfer_models::ltx::loader::{UnitOffsetSource, gemma_gguf_renames, gemma_norm_offset_ids};
use thinfer_models::ltx::manifest::{self, role};
use thinfer_models::ltx::text_encoder::{
    GemmaEncoder, GemmaEncoderPipelines, gemma_encoder_cfgs, register_handles,
};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, read_u32, report};

const PROMPT: &str = "a red fox leaps over a snowy log at dawn";

// F32-act engine vs bf16-weight reference, same Q8_0 weights. Two-tier:
//
// - Early/well-conditioned states (0..WELL_CONDITIONED): tight rel -- this is
//   where the algorithm is proven (layer 0 is bit-exact vs an independent HF
//   recompute; the first ~16 states stay <6%).
// - All states: slope ~1 (the engine tracks the reference in direction+scale).
//
// The LATE raw-residual states are deliberately NOT held to a tight rel: gemma's
// residual grows to ~1e5 with heavy outlier-channel cancellation, so the bf16
// weight rounding is ill-conditioned there -- bf16-vs-f32 *within the reference*
// already diverges ~97% rel by layer 48, and same-weight accumulation-order
// differences (GPU tiled matmul vs torch) blow up comparably. The conditioning
// that actually feeds the DiT is the FE V2 aggregate embed (per-token RMS + a
// learned Linear over the stacked states), which is robust to this raw-residual
// chaos (bf16-vs-f32 cos 0.995); that is the tight gate (see FE V2 parity).
const SLOPE_TOL: f64 = 0.10;
const REL_TOL: f64 = 0.06;
const WELL_CONDITIONED: usize = 16;

#[tokio::test(flavor = "current_thread")]
async fn encoder_parity() {
    let _trace = thinfer_core::trace::init_from_env();

    let enc_fr = manifest::MANIFEST
        .get(role::ENCODER_GGUF)
        .expect("encoder gguf role");
    let conn_fr = manifest::MANIFEST
        .get(role::CONNECTOR)
        .expect("connector role");
    let tok_fr = manifest::MANIFEST
        .get(role::TOKENIZER)
        .expect("tokenizer role");
    let (Some(gguf_path), Some(conn_path), Some(tok_path)) = (
        cache::resolve(enc_fr),
        cache::resolve(conn_fr),
        cache::resolve(tok_fr),
    ) else {
        eprintln!(
            "skipped[ltx encoder_parity]: gemma gguf / connector / tokenizer not in HF cache"
        );
        return;
    };

    // --- python reference (owns tokenization; writes token_ids + all 49 states) ---
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ltx_encoder_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("prompt.txt");
    // Discriminator busts caches from the old degenerate `AutoTokenizer(gguf_file=)`
    // dumps when the pyref switched to the product tokenizer.json.
    let marker_val = format!("{PROMPT}\ntok=product");
    let hidden_path = tmp.join("all_hidden.bin");
    let cached =
        hidden_path.exists() && std::fs::read_to_string(&marker).is_ok_and(|p| p == marker_val);
    if cached {
        eprintln!(
            "ltx encoder-parity: reusing cached pyref dumps ({})",
            tmp.display()
        );
    } else {
        for ent in std::fs::read_dir(&tmp).into_iter().flatten().flatten() {
            let _ = std::fs::remove_file(ent.path());
        }
        run_python_ref(&gguf_path, &conn_path, &tok_path, &tmp);
        std::fs::write(&marker, &marker_val).expect("write marker");
    }

    let ids = read_u32(&tmp.join("token_ids.bin"));
    let seq = ids.len();
    assert!(seq > 0, "pyref produced no tokens");
    let meta = std::fs::read_to_string(tmp.join("meta.txt")).expect("meta.txt");
    let m: Vec<usize> = meta
        .split_whitespace()
        .take(3)
        .map(|s| s.parse().expect("meta int"))
        .collect();
    assert_eq!(m[0], seq, "pyref seq mismatch");
    let n_states = m[1];
    let hidden_dim = m[2];
    assert_eq!(hidden_dim, gemma::HIDDEN, "hidden mismatch");
    assert_eq!(n_states, gemma::N_LAYERS + 1, "n_states mismatch");
    eprintln!("ltx encoder-parity: {seq} tokens, {n_states} states, hidden {hidden_dim}");

    // --- engine encoder: gemma GGUF Q8_0 + (1+w) norm bake + bf16 acts ---
    let opener = MmapFileOpener::new(&gguf_path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", gguf_path.display()));
    let gguf = GgufSource::open(opener).await.expect("parse gemma gguf");
    let renamed = RenamedSource::with_passthrough(gguf, gemma_gguf_renames());
    let source = UnitOffsetSource::new(renamed, gemma_norm_offset_ids());
    let budget = ResidencyBudget {
        ram_bytes: 16 << 30,
        vram_bytes: 6 << 30,
    };
    let residency = WeightResidency::new(source, budget);
    let handles = register_handles(&residency, None).expect("register encoder handles");

    let backend = Arc::new(
        WgpuBackend::new_with_config(WgpuConfig {
            power_preference: match std::env::var("THINFER_POWER_PREF").as_deref() {
                Ok("low" | "lowpower" | "integrated") => PowerPreference::LowPower,
                Ok("none") => PowerPreference::None,
                _ => PowerPreference::HighPerformance,
            },
            timestamps: std::env::var("THINFER_TRACE").is_ok(),
            disable_coopmat: std::env::var("THINFER_NO_COOPMAT").is_ok(),
        })
        .await
        .expect("wgpu adapter unavailable for tests"),
    );

    // F32 acts + large_d_sdpa (head_dim 256) + uniform Q8_0 weights; see
    // `gemma_encoder_cfgs` for why F32 (not bf16) is mandatory here.
    let cfgs = gemma_encoder_cfgs(WeightDtype::Quant(QuantKind::Q8_0));
    let pipelines = GemmaEncoderPipelines::compile(&backend, &cfgs)
        .await
        .expect("compile gemma encoder pipelines");

    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));
    let out = GemmaEncoder
        .forward(
            &backend,
            &pipelines,
            &residency,
            &workspace,
            &handles,
            residency.source(),
            &ids,
        )
        .await
        .expect("engine gemma encoder forward");
    assert_eq!(out.seq, seq);
    assert_eq!(out.states.len(), n_states);

    // Diagnostic: dump the engine's 49 states for offline op-bisection.
    if std::env::var("THINFER_LTX_DUMP_ENGINE").is_ok() {
        let mut bytes = Vec::with_capacity(n_states * seq * hidden_dim * 4);
        for st in &out.states {
            for v in st {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        let p = tmp.join("engine_states.bin");
        std::fs::write(&p, &bytes).expect("write engine states");
        eprintln!("wrote engine states -> {}", p.display());
    }

    let exp = read_f32(&hidden_path); // [n_states, seq, hidden]
    assert_eq!(exp.len(), n_states * seq * hidden_dim, "all_hidden size");

    eprintln!("---- ltx gemma encoder parity (per state) ----");
    let mut failures = Vec::new();
    for (i, st) in out.states.iter().enumerate() {
        assert_eq!(st.len(), seq * hidden_dim, "state {i} size");
        let slice = &exp[i * seq * hidden_dim..(i + 1) * seq * hidden_dim];
        let (slope, rel) = report(&format!("state[{i:02}]"), slice, st);
        // Slope tracks the reference for every state; tight rel only where the
        // residual is well-conditioned (see the band note above).
        if !slope.is_finite() || (slope - 1.0).abs() > SLOPE_TOL {
            failures.push(format!("state[{i:02}] slope off: {slope:.6}"));
        }
        if i < WELL_CONDITIONED && rel > REL_TOL {
            failures.push(format!(
                "state[{i:02}] (well-conditioned) rel={:.3}% > {REL_TOL}",
                rel * 100.0
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "ltx encoder parity (slope 1+-{SLOPE_TOL} all states, rel {REL_TOL} for states <{WELL_CONDITIONED}):\n{}",
        failures.join("\n")
    );
}

fn run_python_ref(gguf: &Path, connector: &Path, tokenizer: &Path, out_dir: &Path) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let status = Command::new("uv")
        .args([
            "run",
            "--directory",
            py_dir.to_str().unwrap(),
            "--with",
            "gguf",
            "--with",
            "tokenizers",
            "python",
            "-m",
            "thinfer_pytorch_ref.ltx.gen_encoder_ref",
            "--gguf",
            gguf.to_str().unwrap(),
            "--connector",
            connector.to_str().unwrap(),
            "--tokenizer",
            tokenizer.to_str().unwrap(),
            "--prompt",
            PROMPT,
            "--out",
            out_dir.to_str().unwrap(),
        ])
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "ltx gemma encoder pyref failed");
}
