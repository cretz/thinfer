//! HunyuanVideo 1.5 SingleTokenRefiner parity: the engine `HunyuanRefiner` (txt_in
//! weights from the lightx2v DiT safetensors, fp16 narrowed to bf16, f32 acts) vs
//! the upstream `SingleTokenRefiner` (`gen_refiner_ref.py`, same fp16 bytes as
//! f32). Refines a fixed-seed `[16, 3584]` text-hidden at a fixed timestep and
//! compares per-stage taps + the final `[16, 2048]`. The band absorbs the fp16 ->
//! bf16 weight narrowing (the engine's weight storage), like the VAE gate.

#![cfg(feature = "hunyuan-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::ops::ActDtype;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::hunyuan::refiner::{
    HIDDEN, HunyuanRefiner, HunyuanRefinerPipelines, HunyuanRefinerTaps, IN_CHANNELS,
};
use thinfer_native::MmapFileOpener;

use crate::parity_util::{read_f32, report, resolve_hf};

const SEQ: usize = 16;
const TIMESTEP: f32 = 500.0;

const SLOPE_TOL: f64 = 0.02;
// fp16 weights narrowed to bf16 (the engine storage) vs fp16-as-f32 pyref; the
// bf16 mantissa loss compounds over the refiner matmuls. Measured well inside.
const REL_TOL: f64 = 0.03;

#[tokio::test(flavor = "current_thread")]
async fn refiner_parity() {
    let _trace = thinfer_core::trace::init_from_env();

    let Some(dit_path) = resolve_hf(
        "THINFER_HUNYUAN_DIT",
        "models--lightx2v--Hy1.5-Distill-Models",
        "hy1.5_t2v_480p_lightx2v_4step.safetensors",
    ) else {
        eprintln!("skipped[hunyuan refiner_parity]: lightx2v DiT not in HF cache");
        return;
    };

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("hunyuan_refiner_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("seq.txt");
    let dims = SEQ.to_string();
    let cached = tmp.join("refined.bin").exists()
        && std::fs::read_to_string(&marker).is_ok_and(|d| d == dims);
    if cached {
        eprintln!("hunyuan refiner-parity: reusing cached pyref dumps");
    } else {
        for ent in std::fs::read_dir(&tmp).into_iter().flatten().flatten() {
            let _ = std::fs::remove_file(ent.path());
        }
        run_python_ref(&dit_path, &tmp);
        std::fs::write(&marker, &dims).expect("write marker");
    }

    let text = read_f32(&tmp.join("text_in.bin"));
    assert_eq!(text.len(), SEQ * IN_CHANNELS, "text size");

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

    let pipelines = HunyuanRefinerPipelines::compile_with(&backend, ActDtype::F32)
        .await
        .expect("compile refiner pipelines");

    let opener = MmapFileOpener::new(&dit_path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", dit_path.display()));
    let src = SafetensorsSource::open(opener)
        .await
        .expect("parse DiT safetensors");
    let residency = WeightResidency::new(
        src,
        ResidencyBudget {
            ram_bytes: 24 << 30,
            vram_bytes: 5 << 30,
        },
    );
    let refiner = HunyuanRefiner::new(pipelines, &residency).expect("build refiner");
    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));

    let mut t_emb = Vec::new();
    let mut c_emb = Vec::new();
    let mut cond = Vec::new();
    let mut embedded = Vec::new();
    let mut block0 = Vec::new();
    let mut taps = HunyuanRefinerTaps {
        t_emb: Some(&mut t_emb),
        c_emb: Some(&mut c_emb),
        cond: Some(&mut cond),
        embedded: Some(&mut embedded),
        block0: Some(&mut block0),
    };
    let refined = refiner
        .refine(
            &backend,
            &residency,
            &workspace,
            &text,
            SEQ,
            TIMESTEP,
            Some(&mut taps),
        )
        .await
        .expect("refine");

    eprintln!("---- hunyuan refiner parity ----");
    let mut failures = Vec::new();
    let mut check = |label: &str, exp: &[f32], got: &[f32]| {
        assert_eq!(
            exp.len(),
            got.len(),
            "[{label}] len exp={} got={}",
            exp.len(),
            got.len()
        );
        let (slope, rel) = report(label, exp, got);
        if !slope.is_finite() || (slope - 1.0).abs() > SLOPE_TOL {
            failures.push(format!("{label} slope off: {slope:.6}"));
        }
        if rel > REL_TOL {
            failures.push(format!(
                "{label} rel={:.3}% > {:.3}%",
                rel * 100.0,
                REL_TOL * 100.0
            ));
        }
    };

    check("t_emb", &read_f32(&tmp.join("t_emb.bin")), &t_emb);
    check("c_emb", &read_f32(&tmp.join("c_emb.bin")), &c_emb);
    check("cond", &read_f32(&tmp.join("cond.bin")), &cond);
    check("embedded", &read_f32(&tmp.join("embedded.bin")), &embedded);
    check("block0", &read_f32(&tmp.join("block0.bin")), &block0);
    check("refined", &read_f32(&tmp.join("refined.bin")), &refined);
    assert_eq!(refined.len(), SEQ * HIDDEN, "refined size");

    assert!(
        failures.is_empty(),
        "hunyuan refiner parity:\n{}",
        failures.join("\n")
    );
}

fn run_python_ref(dit: &Path, out_dir: &Path) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let status = Command::new("uv")
        .args([
            "run",
            "--directory",
            py_dir.to_str().unwrap(),
            "--with",
            "einops",
            "--with",
            "loguru",
            "python",
            "-m",
            "thinfer_pytorch_ref.hunyuan.gen_refiner_ref",
            "--dit",
            dit.to_str().unwrap(),
            "--out",
            out_dir.to_str().unwrap(),
            "--seq",
            &SEQ.to_string(),
        ])
        .status()
        .expect("failed to spawn `uv run`");
    assert!(status.success(), "hunyuan refiner pyref failed");
}
