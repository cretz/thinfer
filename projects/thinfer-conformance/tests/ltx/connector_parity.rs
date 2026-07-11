//! LTX-2.3 conditioning-tail parity: FeatureExtractor V2 + the 8-layer gated
//! embeddings connector (per modality). The engine FE V2 + connector (Q8_0/bf16
//! weights from the DiT GGUF, bf16 FE aggregate embeds from the connector
//! safetensors, F32 acts) vs the upstream `Embeddings1DConnector` reference
//! (`ltx/gen_encoder_ref.py --dit-gguf`, same weights bf16-rounded).
//!
//! Both sides consume the SAME Gemma hidden states (the pyref's `all_hidden`
//! dump), so this isolates the new FE V2 + connector code from the encoder's
//! late-residual drift (encoder is gated by `encoder_parity`). The band is the
//! engine's f32 GPU-tiled compute vs torch over 8 connector attention layers.

#![cfg(feature = "ltx-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::ltx::connector::{
    self, AUDIO, ConnectorPipelines, VIDEO, connector_forward, fe_aggregate,
    feature_extractor_v2_flatten, register_connector, register_fe,
};
use thinfer_models::ltx::gemma;
use thinfer_models::ltx::manifest::{self, role};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, read_u32, report};

const PROMPT: &str = "a red fox leaps over a snowy log at dawn";

// Engine (f32 GPU-tiled) vs torch over the SAME bf16 weights + same input
// states: bit-tight. Measured rel ~0.006% / slope 1.000000 over the full 8-layer
// connector; the band keeps generous headroom for GPU accumulation-order drift.
const SLOPE_TOL: f64 = 0.01;
const REL_TOL: f64 = 0.005;

#[tokio::test(flavor = "current_thread")]
async fn connector_parity() {
    let _trace = thinfer_core::trace::init_from_env();

    let enc_fr = manifest::MANIFEST
        .get(role::ENCODER_GGUF)
        .expect("encoder role");
    let conn_fr = manifest::MANIFEST
        .get(role::CONNECTOR)
        .expect("connector role");
    let dit_fr = manifest::MANIFEST
        .get(role::DIT_GGUF_Q8_0)
        .expect("dit role");
    let tok_fr = manifest::MANIFEST
        .get(role::TOKENIZER)
        .expect("tokenizer role");
    let (Some(gguf_path), Some(conn_path), Some(dit_path), Some(tok_path)) = (
        cache::resolve(enc_fr),
        cache::resolve(conn_fr),
        cache::resolve(dit_fr),
        cache::resolve(tok_fr),
    ) else {
        eprintln!("skipped[ltx connector_parity]: gemma/connector/dit/tokenizer not in HF cache");
        return;
    };

    // --- python reference (FE V2 + connector from the same weights) ---
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ltx_connector_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("prompt.txt");
    // Discriminator busts caches from the old degenerate `AutoTokenizer(gguf_file=)`
    // dumps when the pyref switched to the product tokenizer.json.
    let marker_val = format!("{PROMPT}\ntok=product,leftpad");
    let vc_path = tmp.join("video_connected.bin");
    let cached =
        vc_path.exists() && std::fs::read_to_string(&marker).is_ok_and(|p| p == marker_val);
    if cached {
        eprintln!(
            "ltx connector-parity: reusing cached pyref dumps ({})",
            tmp.display()
        );
    } else {
        for ent in std::fs::read_dir(&tmp).into_iter().flatten().flatten() {
            let _ = std::fs::remove_file(ent.path());
        }
        run_python_ref(&gguf_path, &conn_path, &dit_path, &tok_path, &tmp);
        std::fs::write(&marker, &marker_val).expect("write marker");
    }

    let ids = read_u32(&tmp.join("token_ids.bin"));
    let seq = ids.len();
    assert!(seq > 0, "pyref produced no tokens");
    let cm = std::fs::read_to_string(tmp.join("conn_meta.txt")).expect("conn_meta.txt");
    let m: Vec<usize> = cm
        .split_whitespace()
        .take(3)
        .map(|s| s.parse().expect("meta int"))
        .collect();
    let (s_conn, v_dim, a_dim) = (m[0], m[1], m[2]);
    assert_eq!(s_conn, connector::CONN_SEQ, "connector seq");
    assert_eq!(v_dim, VIDEO.inner_dim);
    assert_eq!(a_dim, AUDIO.inner_dim);

    // Gemma hidden states (49 x [seq, 3840]) from the pyref dump.
    let all_hidden = read_f32(&tmp.join("all_hidden.bin"));
    let n_states = gemma::N_LAYERS + 1;
    assert_eq!(
        all_hidden.len(),
        n_states * seq * gemma::HIDDEN,
        "all_hidden size"
    );
    let states: Vec<Vec<f32>> = (0..n_states)
        .map(|l| all_hidden[l * seq * gemma::HIDDEN..(l + 1) * seq * gemma::HIDDEN].to_vec())
        .collect();

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

    let pipelines = ConnectorPipelines::compile(&backend)
        .await
        .expect("compile connector pipelines");

    let flat = feature_extractor_v2_flatten(&states, seq);

    // --- FE V2 aggregate embeds (own residency over the connector safetensors;
    // dropped before the connector phase to free its ~2.3GB of VRAM) ---
    let (video_embed, audio_embed) = {
        let opener = MmapFileOpener::new(&conn_path)
            .await
            .unwrap_or_else(|e| panic!("open {}: {e}", conn_path.display()));
        let src = SafetensorsSource::open(opener)
            .await
            .expect("parse connector safetensors");
        let residency = WeightResidency::new(
            src,
            ResidencyBudget {
                ram_bytes: 16 << 30,
                vram_bytes: 5 << 30,
            },
        );
        let fe = register_fe(&residency).expect("register FE handles");
        let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));
        let v = fe_aggregate(
            &backend,
            &pipelines,
            &residency,
            &workspace,
            &flat,
            seq,
            fe.video_w,
            fe.video_b,
            VIDEO.out_dim,
        )
        .await
        .expect("fe video");
        let a = fe_aggregate(
            &backend,
            &pipelines,
            &residency,
            &workspace,
            &flat,
            seq,
            fe.audio_w,
            fe.audio_b,
            AUDIO.out_dim,
        )
        .await
        .expect("fe audio");
        (v, a)
    };

    // --- connector blocks (own residency over the DiT GGUF) ---
    let opener = MmapFileOpener::new(&dit_path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", dit_path.display()));
    let src = GgufSource::open(opener).await.expect("parse dit gguf");
    let residency = WeightResidency::new(
        src,
        ResidencyBudget {
            ram_bytes: 16 << 30,
            vram_bytes: 5 << 30,
        },
    );
    let video_h = register_connector(&residency, VIDEO).expect("register video connector");
    let audio_h = register_connector(&residency, AUDIO).expect("register audio connector");
    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    let video_kv = connector_forward(
        &backend,
        &pipelines,
        &residency,
        &workspace,
        &video_h,
        VIDEO,
        &video_embed,
        seq,
    )
    .await
    .expect("video connector forward");
    let audio_kv = connector_forward(
        &backend,
        &pipelines,
        &residency,
        &workspace,
        &audio_h,
        AUDIO,
        &audio_embed,
        seq,
    )
    .await
    .expect("audio connector forward");

    let video_exp = read_f32(&vc_path);
    let audio_exp = read_f32(&tmp.join("audio_connected.bin"));
    assert_eq!(video_kv.len(), video_exp.len(), "video_kv size");
    assert_eq!(audio_kv.len(), audio_exp.len(), "audio_kv size");

    eprintln!("---- ltx connector parity ----");
    let mut failures = Vec::new();
    for (label, exp, got) in [
        ("video", &video_exp, &video_kv),
        ("audio", &audio_exp, &audio_kv),
    ] {
        let (slope, rel) = report(label, exp, got);
        if !slope.is_finite() || (slope - 1.0).abs() > SLOPE_TOL {
            failures.push(format!("{label} slope off: {slope:.6}"));
        }
        if rel > REL_TOL {
            failures.push(format!("{label} rel={:.3}% > {REL_TOL}", rel * 100.0));
        }
    }
    assert!(
        failures.is_empty(),
        "ltx connector parity (slope 1+-{SLOPE_TOL}, rel {REL_TOL}):\n{}",
        failures.join("\n")
    );
}

fn run_python_ref(gguf: &Path, connector: &Path, dit: &Path, tokenizer: &Path, out_dir: &Path) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let status = Command::new("uv")
        .args([
            "run",
            "--directory",
            py_dir.to_str().unwrap(),
            "--with",
            "gguf",
            "--with",
            "einops",
            "--with",
            "tokenizers",
            "python",
            "-m",
            "thinfer_pytorch_ref.ltx.gen_encoder_ref",
            "--gguf",
            gguf.to_str().unwrap(),
            "--connector",
            connector.to_str().unwrap(),
            "--dit-gguf",
            dit.to_str().unwrap(),
            "--tokenizer",
            tokenizer.to_str().unwrap(),
            "--prompt",
            PROMPT,
            "--out",
            out_dir.to_str().unwrap(),
        ])
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "ltx connector pyref failed");
}
