//! Ideogram-4 text-encoder parity: the engine Qwen3-VL-8B encoder (Q8_0
//! weights from the GGUF + F16 acts + DP4A/flash-sdpa) vs a bf16 reference
//! dequantized from the SAME GGUF (`gen_encoder_ref.py`), fed the SAME tokens.
//!
//! Because both sides start from identical Q8_0 bytes, the band is dominated by
//! F16-act vs bf16-act rounding (much tighter than zimage's qwen3_parity, which
//! compares against a different-checkpoint bf16 HF model). Linfits every one of
//! the 13 taps + the stacked 53248-dim features, so a regression names the
//! first diverging tap.

#![cfg(feature = "ideogram4-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::format::union::RenamedSource;
use thinfer_core::ops::{ActDtype, WeightDtype, WgslConfig};
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::quant::QuantKind;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::common::block::{BlockPipelines, BlockWgslConfigs, DenseActSites};
use thinfer_models::ideogram4::manifest::{self, role};
use thinfer_models::ideogram4::text_encoder::{self, Qwen3VlEncoder, register_handles};
use thinfer_models::z_image::qwen3_gguf_renames;
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, read_u32, report};

const PROMPT: &str = "a red apple on a wooden table";

/// Final-feature gate (the 13-tap stack IS the DiT conditioning surface).
/// Measured-baseline-plus-margin with bf16 acts + Q8_0 weights from the same
/// GGUF the pyref dequantizes (2026-06-24): feats slope 0.9904, rel 7.46%.
/// The rel is dominated by Qwen3's massive-activation channels, whose
/// magnitude grows toward the final layer (per-tap: ~0.8% at layer 0, ~3%
/// mid, 5.3% at layer 35); bf16 tracks them to rounding (f16 crushes them to
/// 12%, slope 0.97 -- see the act_dtype comment). The clean ~0.99 slope means
/// it is rounding noise, not a directional bug, and the DiT's `llm_cond_norm`
/// RMSNorm down-weights the outlier channels. The per-tap injection check
/// below is the real bug-catcher; the e2e image arbitrates perceptual impact.
/// Do not loosen to make red go green; re-measure + re-document on a config
/// change.
const FEATS_SLOPE_TOL: f64 = 0.02;
const FEATS_REL_TOL: f64 = 0.09;
/// Per-tap injection detector: no tap may multiply rel error past this over its
/// predecessor (above a floor), catching a single-layer bug that washes out.
const TAP_REL_JUMP_TOL: f64 = 3.0;
const TAP_REL_FLOOR: f64 = 0.005;

#[tokio::test(flavor = "current_thread")]
async fn encoder_parity() {
    let _trace = thinfer_core::trace::init_from_env();

    // --- resolve the encoder GGUF from the HF cache (skip cleanly if absent) ---
    let fr = manifest::MANIFEST
        .get(role::ENCODER_GGUF_Q8_0)
        .expect("encoder gguf role");
    let Some(gguf_path) = cache::resolve(fr) else {
        eprintln!(
            "skipped[encoder_parity]: {}/{} not in HF cache",
            fr.repo, fr.path
        );
        return;
    };

    // --- python reference (owns tokenization; writes token_ids + dumps) ---
    // The bf16 CPU forward of the 8B encoder is a pure function of the prompt,
    // so reuse the dumps across runs; a prompt change invalidates them.
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ideo_encoder_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("prompt.txt");
    let feats_path = tmp.join("py_ideo_feats.bin");
    let cached = feats_path.exists() && std::fs::read_to_string(&marker).is_ok_and(|p| p == PROMPT);
    if cached {
        eprintln!(
            "encoder-parity: reusing cached pyref dumps ({})",
            tmp.display()
        );
    } else {
        for ent in std::fs::read_dir(&tmp).into_iter().flatten().flatten() {
            if ent.file_name().to_string_lossy().starts_with("py_ideo_") {
                let _ = std::fs::remove_file(ent.path());
            }
        }
        run_python_ref(&gguf_path, &tmp);
        std::fs::write(&marker, PROMPT).expect("write marker");
    }

    let ids = read_u32(&tmp.join("token_ids.bin"));
    let seq = ids.len();
    assert!(seq > 0, "pyref produced no tokens");
    eprintln!("encoder-parity: {seq} tokens");

    // --- engine encoder: GGUF Q8_0 (no transcode) + F16/F32 acts ---
    let opener = MmapFileOpener::new(&gguf_path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", gguf_path.display()));
    let gguf = GgufSource::open(opener).await.expect("parse encoder gguf");
    let source = RenamedSource::with_passthrough(gguf, qwen3_gguf_renames());
    let budget = ResidencyBudget {
        ram_bytes: 10 << 30,
        vram_bytes: 6 << 30,
    };
    let residency = WeightResidency::new(source, budget);
    // GGUF weights are already Q8_0; no upload-time transcode.
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

    // bf16 acts (not f16): Qwen3's residual stream develops massive activations
    // from layer ~6; f16 crushes those outlier channels (~12% rel, slope 0.97),
    // exactly the umT5 rationale. bf16 has f32's exponent range, so it tracks
    // the bf16 reference to rounding. The encoder runs once per image, so the
    // bf16-act (dequant-Q8_0-weight) matmul cost is amortized.
    let act = ActDtype::Bf16;
    eprintln!("encoder-parity: act_dtype={act:?}");
    let ops = WgslConfig {
        bf16_quant_writes: false,
        act_dtype: act,
        weight_dtype: WeightDtype::Bf16,
    };
    // Every encoder matmul site is Q8_0 in the GGUF (ffn_down included).
    let q8 = WgslConfig {
        weight_dtype: WeightDtype::Quant(QuantKind::Q8_0),
        ..ops
    };
    let cfgs = BlockWgslConfigs {
        matmul_qkv: q8,
        matmul_qkv_self: q8,
        matmul_proj: q8,
        matmul_ffn_up: q8,
        matmul_ffn_down: q8,
        matmul_adaln: ops,
        ops,
        i8_sdpa: false,
        dense_acts: DenseActSites::default(),
        coopmat_acts: Default::default(),
        large_d_sdpa: false,
        fast_sdpa: false,
        decode_sdpa: false,
    };
    let pipelines = BlockPipelines::compile(&backend, &cfgs)
        .await
        .expect("compile encoder pipelines");

    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));
    // encode() even-pads an odd token count, so the rope table must span the
    // padded length.
    let encoder = Qwen3VlEncoder::new(seq.next_multiple_of(2));
    let out = encoder
        .encode(
            &backend,
            &pipelines,
            &residency,
            &workspace,
            &handles,
            residency.source(),
            &ids,
            true, // keep_taps for localization
        )
        .await
        .expect("engine encoder forward");
    assert_eq!(out.seq, seq);
    assert_eq!(out.taps.len(), text_encoder::config::N_TAPS);

    // --- compare every tap, then the stacked features ---
    eprintln!("---- ideogram encoder parity: per-tap ----");
    let mut rels = Vec::with_capacity(out.taps.len());
    for (j, tap) in out.taps.iter().enumerate() {
        let exp = read_f32(&tmp.join(format!("py_ideo_tap{j}.bin")));
        let (_slope, rel) = report(&format!("tap{j:02}"), &exp, tap);
        rels.push(rel);
    }

    eprintln!("---- ideogram encoder parity: stacked features ----");
    let exp_feats = read_f32(&feats_path);
    let (feats_slope, feats_rel) = report("feats", &exp_feats, &out.features);

    let mut failures: Vec<String> = Vec::new();
    for j in 1..rels.len() {
        if rels[j] > rels[j - 1].max(TAP_REL_FLOOR) * TAP_REL_JUMP_TOL {
            failures.push(format!(
                "tap{j:02} injects error: rel {:.3}% vs tap{:02} {:.3}% (jump {TAP_REL_JUMP_TOL}x)",
                rels[j] * 100.0,
                j - 1,
                rels[j - 1] * 100.0
            ));
        }
    }
    if (feats_slope - 1.0).abs() > FEATS_SLOPE_TOL || feats_rel > FEATS_REL_TOL {
        failures.push(format!(
            "features diverge: slope={feats_slope:.6} rel={:.3}% (tol slope 1+-{FEATS_SLOPE_TOL}, rel {FEATS_REL_TOL})",
            feats_rel * 100.0
        ));
    }
    assert!(
        failures.is_empty(),
        "encoder parity:\n{}",
        failures.join("\n")
    );
}

fn run_python_ref(gguf: &Path, out_dir: &Path) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let status = Command::new("uv")
        .args([
            "run",
            "--directory",
            py_dir.to_str().unwrap(),
            "--with",
            "gguf",
            "python",
            "-m",
            "thinfer_pytorch_ref.ideogram4.gen_encoder_ref",
            "--gguf",
            gguf.to_str().unwrap(),
            "--prompt",
            PROMPT,
            "--out",
            out_dir.to_str().unwrap(),
        ])
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "ideogram encoder pyref failed");
}
