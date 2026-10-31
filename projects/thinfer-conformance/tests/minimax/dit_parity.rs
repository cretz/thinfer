//! MiniMax-H3 truncated DiT parity: the engine `forward_tapped` vs the torch
//! reference `minimax/gen_dit_ref.py`, stage by stage, over the SAME Q4_K_M
//! GGUF weights (the pyref dequantizes them to f32).
//!
//! The truncation is `n_layers = 2`: token refiner (2 blocks) + the first two
//! transformer blocks + the final layer, on a 232-row packed t2va sequence
//! (48 text + 40 audio + 144 video) at step 1 of a 4-step run, so video and
//! audio sit at different timesteps and every modulation segment is distinct.
//!
//! Stages, in order (the first failing one localizes the bug):
//! - layout: positions / tags / per-row timestep index / the run's distinct
//!   timesteps, against upstream `packing` + `scheduler` (host, exact-ish);
//! - adaln: the precomputed per-block and final modulation tables (host);
//! - x_embed, refiner{0,1}, refiner_out, block{0,1}, final_norm, video/audio
//!   velocity (GPU, production config: Q4_K dequant to bf16, bf16 acts), plus
//!   the intra-block sublayers of refiner block 0 and main block 0 (the
//!   engine's `BlockProbe`), so a failure names the sublayer.
//!
//! This gate found the noise bug: H3's bf16 block called `op_sdpa`, whose
//! dense fallback ran the f16-only subgroup kernel on bf16 Q/K/V bits.
//!
//! `dit_parity` runs the base checkpoint; `dit_parity_turbo` adds the lightx2v
//! turbo LoRA, which the pyref applies the PEFT way on the diffusers module
//! layout it was trained on (separate to_q/to_k/to_v, `[value; gate]` fc1,
//! alpha/rank scaling) -- an independent route to the same delta the engine
//! synthesizes onto the fused checkpoint tensors.

#![cfg(feature = "minimax-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf;
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::ops::{ActDtype, WeightDtype};
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::quant::QuantKind;
use thinfer_core::residency::WeightResidency;
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::Workspace;
use thinfer_models::common::merge::MergedSource;
use thinfer_models::minimax::DitConfig;
use thinfer_models::minimax::dit::{
    COOPMAT_SITES, DitPipelines, DitShape, dit_cfgs, probe_layout, register_handles,
};
use thinfer_models::minimax::loader::{
    ModulationTables, build_modulation_tables, register_video_patch_proj,
};
use thinfer_models::minimax::lora::LoraSource;
use thinfer_models::minimax::manifest::{self, role};
use thinfer_models::minimax::pipeline::{
    DenoisePlan, ForwardTaps, forward, forward_tapped, upload_modulation_tables,
};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, read_i32, read_u32, report};

/// Host-side stages (layout, modulation tables) are f32 vs f32 over identical
/// dequantized weights: anything beyond accumulation-order noise is a bug.
const HOST_REL_TOL: f64 = 1e-4;
/// GPU stages run bf16 activations against an f32 reference. Measured (base and
/// turbo alike): every stage slope within 0.005 of 1, rel 0.3-1.3% inside the
/// blocks, video velocity 0.74%, audio velocity 1.8% (32 channels, the
/// smallest head). Before the SDPA fix the first attention output sat at 94%.
const SLOPE_TOL: f64 = 0.02;
const REL_TOL: f64 = 0.03;
/// Full depth (`dit_parity_deep`: 50 bf16 blocks, residual rms ~15k on real
/// text) drifts the final norm to ~6% and the 32-channel audio head to ~9%
/// (measured); every block and the video velocity stay within [`REL_TOL`].
const DEEP_FINAL_REL_TOL: f64 = 0.10;
const DEEP_AUDIO_REL_TOL: f64 = 0.12;
/// Turbo adapter delta (turbo minus base forward), see `gpu_stages`. SLOPE is
/// the gate: measured 0.983 (video) / 1.042 (audio) after the contiguous-qkv fix;
/// the old unscaled adapter measured 10.5 / 12.8 and a dropped one reads 0. REL
/// is only a sanity bound: the delta is a small fraction of the output, so the
/// bf16 noise of two separate forwards leaves it at ~29% (video) / ~229% (audio,
/// the tiny 32-channel head) even when correct.
const DELTA_SLOPE_TOL: f64 = 0.1;
const DELTA_REL_TOL: f64 = 4.0;

#[tokio::test(flavor = "current_thread")]
async fn dit_parity() {
    let _trace = thinfer_core::trace::init_from_env();
    let Some(dit_path) = dit_gguf() else {
        eprintln!("skipped[minimax dit_parity]: DiT GGUF not in HF cache");
        return;
    };
    let dir = ensure_pyref(&dit_path, None, "minimax_dit_parity", &[]);
    let base = open_gguf(&dit_path).await;
    run_parity("base", &dir, base, None).await;
}

#[tokio::test(flavor = "current_thread")]
async fn dit_parity_turbo() {
    let _trace = thinfer_core::trace::init_from_env();
    let Some(dit_path) = dit_gguf() else {
        eprintln!("skipped[minimax dit_parity_turbo]: DiT GGUF not in HF cache");
        return;
    };
    let lora_fr = manifest::MANIFEST
        .get(role::TURBO_LORA)
        .expect("turbo lora role");
    let Some(lora_path) = cache::resolve(lora_fr) else {
        eprintln!("skipped[minimax dit_parity_turbo]: turbo LoRA not in HF cache");
        return;
    };
    let dir = ensure_pyref(&dit_path, Some(&lora_path), "minimax_dit_parity_turbo", &[]);
    // Same inputs and seed as the turbo set, adapter off: the delta baseline.
    let base_dir = ensure_pyref(&dit_path, None, "minimax_dit_parity", &[]);
    let base = open_gguf(&dit_path).await;
    let opener = MmapFileOpener::new(&lora_path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", lora_path.display()));
    let lora = SafetensorsSource::open(opener)
        .await
        .expect("parse turbo lora safetensors");
    let cfg = truncated_cfg(&dir);
    let src = MergedSource::new(
        base,
        LoraSource::turbo(&cfg, lora)
            .await
            .expect("stack turbo lora"),
    );
    run_parity("turbo", &dir, src, Some(&base_dir)).await;
}

/// Full depth: all 50 blocks, streamed through the 4 GiB budget (eviction on),
/// over the REAL conditioning the encoder gate dumped (`encoder_parity` runs
/// first). The truncated gate sees neither: 2 resident blocks on randn text.
#[tokio::test(flavor = "current_thread")]
async fn dit_parity_deep() {
    let _trace = thinfer_core::trace::init_from_env();
    let Some(dit_path) = dit_gguf() else {
        eprintln!("skipped[minimax dit_parity_deep]: DiT GGUF not in HF cache");
        return;
    };
    let text = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("minimax_encoder_parity/hidden.bin");
    if !text.exists() {
        eprintln!("skipped[minimax dit_parity_deep]: run encoder_parity first (real conditioning)");
        return;
    }
    let extra = [
        "--blocks".as_ref(),
        "50".as_ref(),
        "--text".as_ref(),
        text.as_os_str(),
    ];
    let dir = ensure_pyref(&dit_path, None, "minimax_dit_parity_deep", &extra);
    let base = open_gguf(&dit_path).await;
    run_parity("deep", &dir, base, None).await;
}

fn dit_gguf() -> Option<PathBuf> {
    let fr = manifest::MANIFEST
        .get(role::DIT_GGUF_Q4_K_M)
        .expect("dit role");
    cache::resolve(fr)
}

async fn open_gguf(path: &Path) -> gguf::GgufSource<MmapFileOpener> {
    let opener = MmapFileOpener::new(path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    gguf::GgufSource::open(opener)
        .await
        .expect("parse dit gguf")
}

/// `Nt frames lat_h lat_w audio_latents steps step n_blocks`.
fn meta(dir: &Path) -> [usize; 8] {
    let s = std::fs::read_to_string(dir.join("meta.txt")).expect("meta.txt");
    let v: Vec<usize> = s
        .split_whitespace()
        .map(|t| t.parse().expect("meta int"))
        .collect();
    v.try_into().expect("meta has 8 fields")
}

fn truncated_cfg(dir: &Path) -> DitConfig {
    let mut cfg = DitConfig::h3_33b();
    cfg.n_layers = meta(dir)[7];
    cfg
}

/// Collects every stage's numbers first, then asserts once, so one run shows
/// the whole divergence profile rather than stopping at the first stage.
#[derive(Default)]
struct Gate {
    failures: Vec<String>,
}

impl Gate {
    fn stage(&mut self, label: &str, exp: &[f32], got: &[f32], slope_tol: f64, rel_tol: f64) {
        let (slope, rel) = report(label, exp, got);
        if !slope.is_finite() || (slope - 1.0).abs() > slope_tol {
            self.failures.push(format!("{label}: slope {slope:.5}"));
        }
        // NaN-safe: a non-finite rel fails too.
        if rel.is_nan() || rel > rel_tol {
            self.failures.push(format!(
                "{label}: rel {:.3}% > {:.3}%",
                rel * 100.0,
                rel_tol * 100.0
            ));
        }
    }

    fn check(&mut self, ok: bool, msg: String) {
        if !ok {
            self.failures.push(msg);
        }
    }

    fn finish(self, name: &str) {
        assert!(
            self.failures.is_empty(),
            "minimax dit parity ({name}) failed:\n{}",
            self.failures.join("\n")
        );
    }
}

/// Host-only stages (layout + modulation tables), no GPU. Cheap enough to run
/// alongside GPU work, and the first two things the GPU stages depend on.
#[tokio::test(flavor = "current_thread")]
async fn dit_parity_host() {
    let Some(dit_path) = dit_gguf() else {
        eprintln!("skipped[minimax dit_parity_host]: DiT GGUF not in HF cache");
        return;
    };
    let dir = ensure_pyref(&dit_path, None, "minimax_dit_parity", &[]);
    let src = open_gguf(&dit_path).await;
    let cfg = truncated_cfg(&dir);
    let plan = build_plan(&dir, &cfg);
    let mut gate = Gate::default();
    host_stages(&dir, &cfg, &plan, &src, &mut gate).await;
    gate.finish("host");
}

fn build_plan(dir: &Path, cfg: &DitConfig) -> DenoisePlan {
    let [n_text, frames, lat_h, lat_w, audio_latents, steps, ..] = meta(dir);
    DenoisePlan::build::<std::convert::Infallible>(
        cfg,
        n_text,
        frames,
        lat_h,
        lat_w,
        audio_latents,
        steps,
    )
    .expect("build denoise plan")
}

/// `base_ref` (turbo only): the base-checkpoint pyref dir, for the adapter-delta
/// stages.
async fn run_parity<S: WeightSource>(name: &str, dir: &Path, src: S, base_ref: Option<&Path>)
where
    S::Error: core::fmt::Debug,
{
    let [.., steps, step, n_blocks] = meta(dir);
    let cfg = truncated_cfg(dir);
    let plan = build_plan(dir, &cfg);
    eprintln!(
        "---- minimax dit parity ({name}): {} rows, {n_blocks} blocks, step {step}/{steps} ----",
        plan.packed.seq_len
    );
    let mut gate = Gate::default();
    let tables = host_stages(dir, &cfg, &plan, &src, &mut gate).await;
    gpu_stages(dir, &cfg, &plan, src, &tables, base_ref, &mut gate).await;
    gate.finish(name);
}

async fn host_stages<S: WeightSource>(
    dir: &Path,
    cfg: &DitConfig,
    plan: &DenoisePlan,
    src: &S,
    gate: &mut Gate,
) -> ModulationTables
where
    S::Error: core::fmt::Debug,
{
    let step = meta(dir)[6];
    // --- layout + schedule vs upstream ---
    let pos_exp = read_f32(&dir.join("positions.bin"));
    let pos_got: Vec<f32> = plan.packed.position_ids.iter().flatten().copied().collect();
    let pos_err = pos_exp
        .iter()
        .zip(&pos_got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("[   positions] max abs err {pos_err:.3e}");
    gate.check(
        pos_exp.len() == pos_got.len() && pos_err <= 1e-4,
        format!("positions differ (max abs {pos_err})"),
    );
    gate.check(
        read_i32(&dir.join("tags.bin")) == plan.packed.token_tags,
        "token tags differ".into(),
    );
    let ts_exp = read_f32(&dir.join("timesteps.bin"));
    eprintln!(
        "[   timesteps] ref {ts_exp:?}\n[   timesteps] got {:?}",
        plan.modulation.timesteps
    );
    gate.check(
        ts_exp.len() == plan.modulation.timesteps.len()
            && ts_exp
                .iter()
                .zip(&plan.modulation.timesteps)
                .all(|(a, b)| (a - b).abs() <= 1e-6),
        "distinct timesteps differ".into(),
    );
    gate.check(
        read_u32(&dir.join("ts_index.bin")) == plan.modulation.per_step_row_index[step],
        "per-row timestep index differs".into(),
    );

    // --- modulation tables (host) ---
    let tables = build_modulation_tables(cfg, src, &plan.modulation.timesteps)
        .await
        .expect("build modulation tables");
    for (i, got) in tables.blocks.iter().enumerate() {
        let exp = read_f32(&dir.join(format!("adaln_block{i}.bin")));
        gate.stage(&format!("adaln_block{i}"), &exp, got, 1e-4, HOST_REL_TOL);
    }
    let exp = read_f32(&dir.join("adaln_final.bin"));
    gate.stage("adaln_final", &exp, &tables.final_layer, 1e-4, HOST_REL_TOL);
    tables
}

async fn gpu_stages<S: WeightSource>(
    dir: &Path,
    cfg: &DitConfig,
    plan: &DenoisePlan,
    src: S,
    tables: &ModulationTables,
    base_ref: Option<&Path>,
    gate: &mut Gate,
) where
    S::Error: core::fmt::Debug,
{
    let [.., step, n_blocks] = meta(dir);
    // --- GPU forward, production config ---
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
    let residency = WeightResidency::new(
        src,
        ResidencyBudget {
            ram_bytes: 6 << 30,
            vram_bytes: 4 << 30,
        },
    );
    let mut handles = register_handles(cfg, &residency, None).expect("register dit handles");
    handles.video_patch_proj =
        register_video_patch_proj(cfg, &residency).expect("register video_patch_proj");
    eprintln!("runtime adapter on {} blocks", handles.lora_block_count());
    let pipelines = DitPipelines::compile(
        &backend,
        &dit_cfgs(
            WeightDtype::Quant(QuantKind::Q4_K),
            ActDtype::Bf16,
            COOPMAT_SITES,
        ),
    )
    .await
    .expect("compile dit pipelines");
    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));
    let (table, _held) = upload_modulation_tables(&backend, &workspace, tables, ActDtype::Bf16)
        .expect("upload table");

    let video = read_f32(&dir.join("video_in.bin"));
    let audio = read_f32(&dir.join("audio_in.bin"));
    let text = read_f32(&dir.join("text_in.bin"));
    let mut taps = ForwardTaps::default();
    let v = forward_tapped(
        cfg,
        &backend,
        &pipelines,
        &residency,
        &workspace,
        &handles,
        &table,
        plan,
        step,
        &video,
        &audio,
        &text,
        Some(&mut taps),
        None,
    )
    .await
    .expect("dit forward");

    // Block 0 of each stack also reports its intra-block intermediates, so a
    // divergence names the sublayer, not just the block.
    let probed = |prefix: &str, rows: usize| -> Vec<String> {
        probe_layout(DitShape { cfg: *cfg, rows })
            .iter()
            .map(|(n, _)| format!("{prefix}.{n}"))
            .collect()
    };
    let mut stages: Vec<String> = vec!["x_embed".into()];
    stages.extend(probed("refiner0", plan.num_text_rows));
    stages.extend((0..cfg.token_refiner_layers).map(|j| format!("refiner{j}")));
    stages.push("refiner_out".into());
    stages.extend(probed("block0", plan.packed.seq_len));
    stages.extend((0..n_blocks).map(|i| format!("block{i}")));
    stages.push("final_norm".into());
    // The engine side is written next to the reference (`got_<stage>.bin`) so a
    // divergence can be dissected offline without another GPU run.
    let save = |s: &str, v: &[f32]| {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(dir.join(format!("got_{s}.bin")), bytes).expect("write engine tap");
    };
    let deep = n_blocks > 2;
    for s in &stages {
        let exp = read_f32(&dir.join(format!("{s}.bin")));
        let got = taps.get(s).unwrap_or_else(|| panic!("no tap {s}"));
        save(s, got);
        let rel = if deep && s == "final_norm" {
            DEEP_FINAL_REL_TOL
        } else {
            REL_TOL
        };
        gate.stage(s, &exp, got, SLOPE_TOL, rel);
    }
    save("video_vel", &v.video);
    save("audio_vel", &v.audio);
    gate.stage(
        "video_vel",
        &read_f32(&dir.join("video_vel.bin")),
        &v.video,
        SLOPE_TOL,
        REL_TOL,
    );
    gate.stage(
        "audio_vel",
        &read_f32(&dir.join("audio_vel.bin")),
        &v.audio,
        SLOPE_TOL,
        if deep { DEEP_AUDIO_REL_TOL } else { REL_TOL },
    );

    // The adapter's own contribution. At alpha/rank = 1/16 the whole delta is
    // only ~1-3% of the output after two blocks, the same order as the bf16
    // noise above, so the absolute stages cannot tell "adapter right" from
    // "adapter missing". Differencing two engine forwards (same residency, same
    // kernels) cancels the common-mode noise and gates the delta directly.
    let Some(base_dir) = base_ref else {
        return;
    };
    let v0 = forward(
        cfg,
        &backend,
        &pipelines,
        &residency,
        &workspace,
        &handles.without_lora(),
        &table,
        plan,
        step,
        &video,
        &audio,
        &text,
        None,
    )
    .await
    .expect("base-only dit forward");
    for (label, got_t, got_b) in [
        ("video_delta", &v.video, &v0.video),
        ("audio_delta", &v.audio, &v0.audio),
    ] {
        let file = format!("{}.bin", label.replace("delta", "vel"));
        let ref_t = read_f32(&dir.join(&file));
        let ref_b = read_f32(&base_dir.join(&file));
        let exp: Vec<f32> = ref_t.iter().zip(&ref_b).map(|(t, b)| t - b).collect();
        let got: Vec<f32> = got_t.iter().zip(got_b.iter()).map(|(t, b)| t - b).collect();
        gate.stage(label, &exp, &got, DELTA_SLOPE_TOL, DELTA_REL_TOL);
    }
}

/// Run (or reuse) the pyref dumps. Each (weights, adapter) pair gets its own
/// dir; delete it to force a regeneration after changing the reference.
fn ensure_pyref(dit: &Path, lora: Option<&Path>, dir: &str, extra: &[&std::ffi::OsStr]) -> PathBuf {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(dir);
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("done.txt");
    if marker.exists() {
        eprintln!(
            "minimax dit-parity: reusing cached pyref dumps ({})",
            tmp.display()
        );
        return tmp;
    }
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let mut cmd = Command::new("uv");
    cmd.args([
        "run",
        "--directory",
        py_dir.to_str().unwrap(),
        "--with",
        "gguf",
        "python",
        "-m",
        "thinfer_pytorch_ref.minimax.gen_dit_ref",
        "--dit-gguf",
        dit.to_str().unwrap(),
        "--out",
        tmp.to_str().unwrap(),
    ]);
    if let Some(l) = lora {
        cmd.args(["--lora", l.to_str().unwrap()]);
    }
    cmd.args(extra);
    // The reference is a few CPU blocks; keep it off the user's other cores.
    cmd.env("OMP_NUM_THREADS", "4");
    let status = cmd
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "minimax dit pyref failed");
    std::fs::write(&marker, "ok").expect("write marker");
    tmp
}
