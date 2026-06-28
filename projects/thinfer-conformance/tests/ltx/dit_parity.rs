//! LTX-2.3 dual-stream DiT one-block parity: the engine `forward_block_dumped`
//! (Q8_0 matmuls, F32 acts, bf16 norm/bias/table weights) vs the upstream
//! `BasicAVTransformerBlock` (`ltx/gen_dit_ref.py`, same DiT GGUF weights
//! bf16-rounded). Both sides consume the SAME post-patchify token streams,
//! caption KV, AdaLN modulation vectors, and rope position bounds (dumped by the
//! pyref); the engine rebuilds the split/half-rot freqs from those bounds. This
//! isolates the block + rope-application (5 attn sublayers, per-head gating,
//! av-cross, X0-velocity) from the timestep/adaln and patchifier paths (P3b/P4).
//!
//! The band is the engine's f32 GPU compute vs torch over one full block; the
//! residual grows large with outlier cancellation, so slope ~1 is the tight gate
//! and rel carries headroom for accumulation-order drift.

#![cfg(feature = "ltx-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::ltx::config as dit;
use thinfer_models::ltx::dit::{
    DitPipelines, HostFreqs, HostStreamMod, Streams, build_split_freqs, forward_block_dumped,
    register_block,
};
use thinfer_models::ltx::manifest::{self, role};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, report};

// Same bf16 weights both sides -> bit-tight. Measured: video slope 0.999997 /
// rel 0.008%, audio slope 1.000000 / rel 0.001% over the full block (5 attn +
// 2 FFN + av-cross + split rope). The band keeps ~60x headroom for GPU
// accumulation-order drift.
const SLOPE_TOL: f64 = 0.01;
const REL_TOL: f64 = 0.005;

#[tokio::test(flavor = "current_thread")]
async fn dit_parity() {
    let _trace = thinfer_core::trace::init_from_env();

    let dit_fr = manifest::MANIFEST
        .get(role::DIT_GGUF_Q8_0)
        .expect("dit role");
    let Some(dit_path) = cache::resolve(dit_fr) else {
        eprintln!("skipped[ltx dit_parity]: DiT GGUF not in HF cache");
        return;
    };

    let tmp = ensure_pyref(&dit_path);
    let meta = std::fs::read_to_string(tmp.join("meta.txt")).expect("meta.txt");
    let m: Vec<usize> = meta
        .split_whitespace()
        .map(|s| s.parse().expect("meta int"))
        .collect();
    let s = Streams {
        video_tokens: m[0],
        audio_tokens: m[1],
        video_text: m[2],
        audio_text: m[3],
    };

    let vx = read_f32(&tmp.join("vx_in.bin"));
    let ax = read_f32(&tmp.join("ax_in.bin"));
    let vtext = read_f32(&tmp.join("vtext.bin"));
    let atext = read_f32(&tmp.join("atext.bin"));
    let v_pos = read_f32(&tmp.join("v_pos.bin"));
    let a_pos = read_f32(&tmp.join("a_pos.bin"));

    let vmod = load_mod(&tmp, "vmod");
    let amod = load_mod(&tmp, "amod");
    let freqs = build_freqs(s, &v_pos, &a_pos);

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

    let pipelines = DitPipelines::compile(&backend)
        .await
        .expect("compile dit pipelines");

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
    let handles = register_block(&residency, 0).expect("register block 0");
    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    let (vx_out, ax_out) = forward_block_dumped(
        &backend, &pipelines, &residency, &workspace, &handles, s, &vx, &ax, &vtext, &atext, &vmod,
        &amod, &freqs,
    )
    .await
    .expect("dit block forward");

    let vx_exp = read_f32(&tmp.join("vx_out.bin"));
    let ax_exp = read_f32(&tmp.join("ax_out.bin"));
    assert_eq!(vx_out.len(), vx_exp.len(), "vx_out size");
    assert_eq!(ax_out.len(), ax_exp.len(), "ax_out size");

    eprintln!("---- ltx dit parity ----");
    let mut failures = Vec::new();
    for (label, exp, got) in [("video", &vx_exp, &vx_out), ("audio", &ax_exp, &ax_out)] {
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
        "ltx dit parity (slope 1+-{SLOPE_TOL}, rel {REL_TOL}):\n{}",
        failures.join("\n")
    );
}

/// Load the 16 AdaLN modulation vectors for one stream from the pyref dumps.
fn load_mod(dir: &Path, prefix: &str) -> HostStreamMod {
    let r = |name: &str| read_f32(&dir.join(format!("{prefix}_{name}.bin")));
    HostStreamMod {
        msa_scale: r("msa_scale"),
        msa_shift: r("msa_shift"),
        msa_gate: r("msa_gate"),
        cq_scale: r("cq_scale"),
        cq_shift: r("cq_shift"),
        cq_gate: r("cq_gate"),
        ckv_scale: r("ckv_scale"),
        ckv_shift: r("ckv_shift"),
        mlp_scale: r("mlp_scale"),
        mlp_shift: r("mlp_shift"),
        mlp_gate: r("mlp_gate"),
        a2v_scale: r("a2v_scale"),
        a2v_shift: r("a2v_shift"),
        v2a_scale: r("v2a_scale"),
        v2a_shift: r("v2a_shift"),
        av_gate: r("av_gate"),
    }
}

/// Sigma the pyref drives the timestep/adaln with (`gen_dit_ref.py` `SIGMA`).
const PYREF_SIGMA: f32 = 0.7;

/// Rebuild the split/half-rot rope freqs from the dumped position bounds (shared
/// by the block + full-forward tests).
fn build_freqs(s: Streams, v_pos: &[f32], a_pos: &[f32]) -> HostFreqs {
    let theta = dit::ROPE_THETA;
    let vmax: Vec<f64> = dit::ROPE_MAX_POS.iter().map(|&x| x as f64).collect();
    let amax = [dit::AUDIO_ROPE_MAX_POS[0] as f64];
    let cross_max = [dit::AUDIO_ROPE_MAX_POS[0] as f64]; // cross_pe_max_pos = max(20,20)
    let v_temporal = &v_pos[0..s.video_tokens * 2]; // axis-0 slice for cross_pe
    HostFreqs {
        video_self: build_split_freqs(
            v_pos,
            3,
            s.video_tokens,
            &vmax,
            dit::DIM,
            dit::N_HEADS,
            dit::HEAD_DIM,
            theta,
        ),
        audio_self: build_split_freqs(
            a_pos,
            1,
            s.audio_tokens,
            &amax,
            dit::AUDIO_DIM,
            dit::AUDIO_N_HEADS,
            dit::AUDIO_HEAD_DIM,
            theta,
        ),
        video_cross: build_split_freqs(
            v_temporal,
            1,
            s.video_tokens,
            &cross_max,
            dit::AUDIO_CROSS_ATTENTION_DIM,
            dit::N_HEADS,
            dit::AUDIO_HEAD_DIM,
            theta,
        ),
        audio_cross: build_split_freqs(
            a_pos,
            1,
            s.audio_tokens,
            &cross_max,
            dit::AUDIO_CROSS_ATTENTION_DIM,
            dit::AUDIO_N_HEADS,
            dit::AUDIO_HEAD_DIM,
            theta,
        ),
    }
}

/// Full DiT forward parity: the engine `DitModel::forward` (patchify -> 1 block
/// -> output stage, on-device timestep/adaln) vs the upstream `LTXModel.forward`
/// (`gen_dit_ref.py`, num_layers=1) velocity prediction. End-to-end DiT minus
/// only the patchifier position-construction (positions dumped) and the
/// multi-block loop (1 block here; trivially repeated for the 48-layer run).
#[tokio::test(flavor = "current_thread")]
async fn dit_full_parity() {
    use thinfer_models::ltx::dit::DitModel;

    let _trace = thinfer_core::trace::init_from_env();
    let dit_fr = manifest::MANIFEST
        .get(role::DIT_GGUF_Q8_0)
        .expect("dit role");
    let Some(dit_path) = cache::resolve(dit_fr) else {
        eprintln!("skipped[ltx dit_full_parity]: DiT GGUF not in HF cache");
        return;
    };
    let tmp = ensure_pyref(&dit_path);
    let meta = std::fs::read_to_string(tmp.join("meta.txt")).expect("meta.txt");
    let m: Vec<usize> = meta
        .split_whitespace()
        .map(|s| s.parse().expect("meta int"))
        .collect();
    let s = Streams {
        video_tokens: m[0],
        audio_tokens: m[1],
        video_text: m[2],
        audio_text: m[3],
    };

    let latent_v = read_f32(&tmp.join("v_latent.bin"));
    let latent_a = read_f32(&tmp.join("a_latent.bin"));
    let vtext = read_f32(&tmp.join("vtext.bin"));
    let atext = read_f32(&tmp.join("atext.bin"));
    let v_pos = read_f32(&tmp.join("v_pos.bin"));
    let a_pos = read_f32(&tmp.join("a_pos.bin"));
    let freqs = build_freqs(s, &v_pos, &a_pos);

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
    let pipelines = DitPipelines::compile(&backend)
        .await
        .expect("compile dit pipelines");
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
    let model = DitModel::register(&backend, &residency, 1)
        .await
        .expect("register dit model");
    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    let (vel_v, vel_a) = model
        .forward(
            &backend,
            &pipelines,
            &residency,
            &workspace,
            s,
            &latent_v,
            &latent_a,
            &vtext,
            &atext,
            PYREF_SIGMA,
            &freqs,
        )
        .await
        .expect("dit full forward");

    let v_exp = read_f32(&tmp.join("v_vel.bin"));
    let a_exp = read_f32(&tmp.join("a_vel.bin"));
    assert_eq!(vel_v.len(), v_exp.len(), "vel_v size");
    assert_eq!(vel_a.len(), a_exp.len(), "vel_a size");

    eprintln!("---- ltx dit full parity ----");
    let mut failures = Vec::new();
    for (label, exp, got) in [("video", &v_exp, &vel_v), ("audio", &a_exp, &vel_a)] {
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
        "ltx dit full parity:\n{}",
        failures.join("\n")
    );
}

/// On-device timestep/AdaLN parity: the engine `cond` path (8 adaln modules +
/// per-block table-add) computes the block-0 modulation from sigma; compare to
/// the pyref's dumped modulation vectors. Isolates the conditioning (P3b) from
/// the block (`dit_parity`).
#[tokio::test(flavor = "current_thread")]
async fn modulation_parity() {
    use thinfer_models::ltx::cond::{
        assemble_block_mod, compute_shared_timestep, read_block_tables, register_timestep,
    };

    let _trace = thinfer_core::trace::init_from_env();
    let dit_fr = manifest::MANIFEST
        .get(role::DIT_GGUF_Q8_0)
        .expect("dit role");
    let Some(dit_path) = cache::resolve(dit_fr) else {
        eprintln!("skipped[ltx modulation_parity]: DiT GGUF not in HF cache");
        return;
    };
    let tmp = ensure_pyref(&dit_path);

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
    let pipelines = DitPipelines::compile(&backend)
        .await
        .expect("compile dit pipelines");
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
    let th = register_timestep(&residency).expect("register timestep");
    let block0 = register_block(&residency, 0).expect("register block 0");
    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    let shared = compute_shared_timestep(
        &backend,
        &pipelines,
        &residency,
        &workspace,
        &th,
        PYREF_SIGMA,
        PYREF_SIGMA,
    )
    .await
    .expect("compute shared timestep");
    let tables = read_block_tables(&backend, &residency, &block0)
        .await
        .expect("read block tables");
    let (vmod, amod) = assemble_block_mod(&shared, &tables);

    let exp_v = load_mod(&tmp, "vmod");
    let exp_a = load_mod(&tmp, "amod");

    eprintln!("---- ltx modulation parity ----");
    let mut failures = Vec::new();
    let mut check = |label: &str, exp: &[f32], got: &[f32]| {
        let (slope, rel) = report(label, exp, got);
        // bf16 adaln matmuls both sides -> bit-exact (measured rel 0.000-0.001%,
        // slope 1.000000 for all 32 vectors). Tight band, generous headroom.
        if !slope.is_finite() || (slope - 1.0).abs() > 0.001 {
            failures.push(format!("{label} slope off: {slope:.6}"));
        }
        if rel > 0.002 {
            failures.push(format!("{label} rel={:.3}% > 0.2%", rel * 100.0));
        }
    };
    for (label, e, g) in mod_pairs("v", &exp_v, &vmod) {
        check(&label, e, g);
    }
    for (label, e, g) in mod_pairs("a", &exp_a, &amod) {
        check(&label, e, g);
    }
    assert!(
        failures.is_empty(),
        "ltx modulation parity:\n{}",
        failures.join("\n")
    );
}

/// The 16 (label, expected, got) modulation-vector pairs for one stream.
fn mod_pairs<'a>(
    s: &'a str,
    e: &'a HostStreamMod,
    g: &'a HostStreamMod,
) -> Vec<(String, &'a [f32], &'a [f32])> {
    let p = |n: &str, ev: &'a [f32], gv: &'a [f32]| (format!("{s}.{n}"), ev, gv);
    vec![
        p("msa_scale", &e.msa_scale, &g.msa_scale),
        p("msa_shift", &e.msa_shift, &g.msa_shift),
        p("msa_gate", &e.msa_gate, &g.msa_gate),
        p("cq_scale", &e.cq_scale, &g.cq_scale),
        p("cq_shift", &e.cq_shift, &g.cq_shift),
        p("cq_gate", &e.cq_gate, &g.cq_gate),
        p("ckv_scale", &e.ckv_scale, &g.ckv_scale),
        p("ckv_shift", &e.ckv_shift, &g.ckv_shift),
        p("mlp_scale", &e.mlp_scale, &g.mlp_scale),
        p("mlp_shift", &e.mlp_shift, &g.mlp_shift),
        p("mlp_gate", &e.mlp_gate, &g.mlp_gate),
        p("a2v_scale", &e.a2v_scale, &g.a2v_scale),
        p("a2v_shift", &e.a2v_shift, &g.a2v_shift),
        p("v2a_scale", &e.v2a_scale, &g.v2a_scale),
        p("v2a_shift", &e.v2a_shift, &g.v2a_shift),
        p("av_gate", &e.av_gate, &g.av_gate),
    ]
}

/// Run (or reuse cached) the 1-block pyref dumps; returns the tmp dir.
fn ensure_pyref(dit_path: &Path) -> PathBuf {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ltx_dit_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("done.txt");
    if tmp.join("v_vel.bin").exists() && marker.exists() {
        eprintln!(
            "ltx dit-parity: reusing cached pyref dumps ({})",
            tmp.display()
        );
    } else {
        for ent in std::fs::read_dir(&tmp).into_iter().flatten().flatten() {
            let _ = std::fs::remove_file(ent.path());
        }
        run_python_ref(dit_path, &tmp);
        std::fs::write(&marker, "ok").expect("write marker");
    }
    tmp
}

fn run_python_ref(dit: &Path, out_dir: &Path) {
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
            "thinfer_pytorch_ref.ltx.gen_dit_ref",
            "--dit-gguf",
            dit.to_str().unwrap(),
            "--out",
            out_dir.to_str().unwrap(),
        ])
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "ltx dit pyref failed");
}
