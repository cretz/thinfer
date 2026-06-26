//! Gemma-3 encoder perf harness (NOT a parity/health gate -- a measurement
//! vehicle for the once-per-request text-encoder front-end). Loads the real
//! Q8_0 Gemma GGUF, runs `GemmaEncoder::forward` on a synthetic real-token
//! sequence, and dumps the `THINFER_TRACE` rollup so the stream-vs-compute
//! split (and the absolute encoder wall) is visible.
//!
//! The encoder streams the whole ~12.5GB Q8_0 tower once per request at a tiny
//! real-token count (<< the 1024 left-pad), so it is expected to be streaming-
//! bound: per-layer compute (M = n_real rows) is trivial next to the ~260MB/
//! layer weight upload. This harness quantifies that and is the rollup the
//! worklog calls for when comparing a smaller-quant encoder.
//!
//! Env knobs (all optional):
//!   LTX_ENC_SEQ        (default 18)  synthetic real-token count
//!   THINFER_E2E_VRAM_GB (default 6)  VRAM budget GB
//!
//! Run: THINFER_TRACE=1 THINFER_POWER_PREF=high cargo test -p thinfer-conformance
//!   --features ltx-e2e --release encoder_perf -- --nocapture --test-threads=1

#![cfg(feature = "ltx-e2e")]

use std::sync::Arc;
use std::time::Instant;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::format::union::RenamedSource;
use thinfer_core::ops::WeightDtype;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::quant::QuantKind;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::ltx::loader::{UnitOffsetSource, gemma_gguf_renames, gemma_norm_offset_ids};
use thinfer_models::ltx::manifest::{self, role};
use thinfer_models::ltx::text_encoder::{
    GemmaEncoder, GemmaEncoderPipelines, gemma_encoder_cfgs, register_handles,
};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[tokio::test(flavor = "current_thread")]
async fn encoder_perf() {
    let trace = thinfer_core::trace::init_from_env();

    // LTX_ENC_QUANT selects the encoder GGUF: q8_0 (default) or q4_k_m (the
    // mixed Q4_K/Q6_K variant that exercises the per-site dequant dispatch).
    let enc_role = match std::env::var("LTX_ENC_QUANT").ok().as_deref() {
        Some("q4_k_m") | Some("q4") => role::ENCODER_GGUF_Q4,
        _ => role::ENCODER_GGUF,
    };
    let fr = manifest::MANIFEST.get(enc_role).expect("role");
    let Some(gemma_path) = cache::resolve(fr) else {
        eprintln!("skipped[ltx encoder_perf]: gemma GGUF {enc_role} not in HF cache");
        return;
    };
    eprintln!("ltx encoder_perf: encoder={enc_role} ({})", fr.path);

    let seq = env_usize("LTX_ENC_SEQ", 18);
    let vram_gb = env_usize("THINFER_E2E_VRAM_GB", 6);
    // Synthetic real tokens (values irrelevant to timing; just valid rows).
    let ids: Vec<u32> = (0..seq).map(|i| (i as u32 * 37 + 11) % 200_000).collect();

    let backend = Arc::new(
        WgpuBackend::new_with_config(WgpuConfig {
            power_preference: match std::env::var("THINFER_POWER_PREF").as_deref() {
                Ok("low" | "lowpower" | "integrated") => PowerPreference::LowPower,
                Ok("none") => PowerPreference::None,
                _ => PowerPreference::HighPerformance,
            },
            timestamps: std::env::var("THINFER_TRACE").is_ok(),
        })
        .await
        .expect("wgpu adapter unavailable for tests"),
    );

    let budget = ResidencyBudget {
        ram_bytes: 16 << 30,
        vram_bytes: (vram_gb as u64) << 30,
    };

    let opener = MmapFileOpener::new(&gemma_path).await.expect("open gemma");
    let gguf = GgufSource::open(opener).await.expect("parse gemma gguf");
    let renamed = RenamedSource::with_passthrough(gguf, gemma_gguf_renames());
    let source = UnitOffsetSource::new(renamed, gemma_norm_offset_ids());
    let residency = WeightResidency::new(source, budget);
    let handles = register_handles(&residency, None).expect("register encoder");
    let cfgs = gemma_encoder_cfgs(WeightDtype::Quant(QuantKind::Q8_0));

    let t_compile = Instant::now();
    let pipelines = GemmaEncoderPipelines::compile(&backend, &cfgs)
        .await
        .expect("compile gemma pipelines");
    let compile_s = t_compile.elapsed().as_secs_f64();

    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    // Cold forward: streams the full tower from disk (residency budget << 12.5GB
    // so nothing stays resident -- this is the real once-per-request cost).
    let t_fwd = Instant::now();
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
        .expect("gemma encoder forward");
    let fwd_s = t_fwd.elapsed().as_secs_f64();

    assert_eq!(out.states.len(), 49, "expected 49 hidden states");
    assert!(
        out.states.iter().all(|s| s.iter().all(|v| v.is_finite())),
        "non-finite hidden state"
    );

    eprintln!(
        "ltx encoder_perf: seq={seq} budget={vram_gb}G compile={compile_s:.2}s forward(cold)={fwd_s:.2}s"
    );

    if let Some(h) = &trace {
        let _ = h.dump(&mut std::io::stderr());
    }
}
