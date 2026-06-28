//! Qwen3 text-encoder parity: engine encoder (Q8_0-transcoded weights +
//! F16 acts + DP4A/flash-sdpa) vs HF transformers bf16 reference, fed the
//! SAME even-padded token ids.
//!
//! Localization instrument: linfits every per-layer residual (35 layers)
//! and every layer-0 per-op intermediate against the python dumps, so one
//! run names the first diverging op instead of bisecting by hand. The
//! python side hooks the layer-0 submodules and recomputes rope manually
//! (`gen_qwen3_parity_ref.py`).

#![cfg(feature = "zimage-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::safetensors::ShardedSafetensorsSource;
use thinfer_core::ops::{ActDtype, WeightDtype, WgslConfig};
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::quant::QuantKind;
use thinfer_core::residency::WeightResidency;
use thinfer_core::tokenizer::Tokenizer;
use thinfer_core::trace;
use thinfer_core::workspace::Workspace;
use thinfer_models::common::block::{BlockPipelines, BlockWgslConfigs, DenseActSites};
use thinfer_models::z_image::manifest::{self, role};
use thinfer_models::z_image::text_encoder::{
    self, Qwen3BlockOpsHost, Qwen3Encoder, Qwen3Taps, register_qwen3_handles,
};
use thinfer_models::z_image::tokenizer::format_qwen3_prompt;
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;
use thinfer_native::tokenizer::HfTokenizer;

use crate::e2e_parity::{linfit, read_f32, summarize};

const PROMPT: &str = "a red apple on a wooden table";
/// Final-hidden gate, measured-baseline-plus-margin (2026-06-05 chosen
/// config: Q8_0 weights at 6 sites, mlp_down bf16, i8 acts on).
/// Measured: rel 4.90% slope 1.0020 (GPU `weight_prep` transcode,
/// 2026-06-06); the earlier CPU-transcode build measured 5.27% / 1.0065.
/// The paths differ only in Q8_0 tie-breaking (WGSL division is 2.5 ULP,
/// so ~0.03% of quants flip by +-1 at round-to-nearest boundaries; both
/// are valid encodings - see ops::weight_prep). Decomposition: all-dense
/// acts gave 5.04%, so i8 acts cost only ~0.23pp; the ~5% residual is
/// Q8_0 weight quant + f16 vs the HF-bf16 pyref (this test measures
/// kernel fidelity PLUS load-time quantization loss, unlike same-GGUF
/// DiT parity). Per-op forensics: 42/40960 attention-sink outlier cells
/// carry ~99% of the sq-err; normal cells sit ~2.7% rel; error injects
/// at layer 6 (massive-activation onset) and decays by layer 33. Slope
/// near 1 everywhere = quantization noise, not a bug. PNG e2e arbitrates
/// perceptual impact. Do not loosen to make red go green; re-measure and
/// re-document if the config changes.
const FINAL_SLOPE_TOL: f64 = 0.02;
const FINAL_REL_TOL: f64 = 0.06;
/// Injection detector: no layer may multiply rel error by more than this
/// over its predecessor (above a noise floor). Catches a single-layer bug
/// signature even when it partially washes out by the final hidden.
const LAYER_REL_JUMP_TOL: f64 = 3.0;
const LAYER_REL_FLOOR: f64 = 0.005;

#[tokio::test(flavor = "current_thread")]
async fn qwen3_parity() {
    let _trace = trace::init_from_env();

    // --- resolve weights from HF cache (skip cleanly when absent) ---
    let needed = [
        role::TEXT_ENCODER_SHARD_1,
        role::TEXT_ENCODER_SHARD_2,
        role::TEXT_ENCODER_SHARD_3,
        role::TOKENIZER_JSON,
    ];
    let mut resolved: Vec<(&str, PathBuf)> = Vec::with_capacity(needed.len());
    for r in needed {
        let fr = manifest::MANIFEST.get(r).expect("role in manifest");
        match cache::resolve(fr) {
            Some(p) => resolved.push((r, p)),
            None => {
                eprintln!(
                    "skipped[qwen3_parity]: {}/{} not in HF cache",
                    fr.repo, fr.path
                );
                return;
            }
        }
    }
    let path_of =
        |role_name: &str| -> &Path { &resolved.iter().find(|(r, _)| *r == role_name).unwrap().1 };

    // --- tokenize once; even-pad exactly like Qwen3Encoder::forward ---
    let tokenizer = HfTokenizer::from_path(path_of(role::TOKENIZER_JSON))
        .await
        .expect("tokenizer load");
    let mut ids = tokenizer
        .encode(&format_qwen3_prompt(PROMPT), false)
        .expect("tokenize");
    if !ids.len().is_multiple_of(2) {
        ids.push(*ids.last().unwrap());
    }
    let seq = ids.len();
    eprintln!("qwen3-parity: {seq} tokens (even-padded)");

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("qwen3_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let ids_path = tmp.join("token_ids.bin");
    let id_bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();

    // --- python reference (cached) ---
    // The bf16 CPU forward of the 4B encoder takes minutes; the dumps are
    // pure functions of the token ids, so reuse them across runs. Any ids
    // change (prompt/tokenizer edits) invalidates and re-runs.
    let last_hs = tmp.join(format!(
        "py_qwen3_hs{}.bin",
        text_encoder::config::HIDDEN_STATES_LAYER
    ));
    let cached = std::fs::read(&ids_path).is_ok_and(|prev| prev == id_bytes) && last_hs.exists();
    if cached {
        eprintln!(
            "qwen3-parity: reusing cached pyref dumps ({})",
            tmp.display()
        );
    } else {
        // Clear stale dumps so a missing hook can't be masked by an old file.
        if let Ok(rd) = std::fs::read_dir(&tmp) {
            for ent in rd.flatten() {
                let name = ent.file_name();
                if name.to_string_lossy().starts_with("py_qwen3_") {
                    let _ = std::fs::remove_file(ent.path());
                }
            }
        }
        std::fs::write(&ids_path, &id_bytes).expect("write ids");
        run_python_ref(&ids_path, &tmp, &[0]);
    }

    // --- engine encoder: Q8_0 transcode + F16/F32 acts (prod cfg) ---
    let weight_roles = [
        role::TEXT_ENCODER_SHARD_1,
        role::TEXT_ENCODER_SHARD_2,
        role::TEXT_ENCODER_SHARD_3,
    ];
    let mut openers: Vec<MmapFileOpener> = Vec::with_capacity(weight_roles.len());
    for r in weight_roles {
        let path = path_of(r);
        openers.push(
            MmapFileOpener::new(path)
                .await
                .unwrap_or_else(|e| panic!("open {}: {e}", path.display())),
        );
    }
    let source = ShardedSafetensorsSource::open(openers)
        .await
        .expect("parse text-encoder shards");
    let budget = ResidencyBudget {
        ram_bytes: 2 << 30,
        vram_bytes: 2 << 30,
    };
    let residency = WeightResidency::new(source, budget);
    let handles =
        register_qwen3_handles(&residency, Some(QuantKind::Q8_0)).expect("register qwen3");

    let cfg = WgpuConfig {
        power_preference: match std::env::var("THINFER_POWER_PREF")
            .ok()
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("low" | "lowpower" | "integrated") => PowerPreference::LowPower,
            Some("none") => PowerPreference::None,
            _ => PowerPreference::HighPerformance,
        },
        timestamps: std::env::var("THINFER_TRACE").is_ok(),
        disable_coopmat: std::env::var("THINFER_NO_COOPMAT").is_ok(),
    };
    let backend = Arc::new(
        WgpuBackend::new_with_config(cfg)
            .await
            .expect("wgpu adapter unavailable for tests"),
    );

    // Mirror ZImageModel::load's encoder pipeline-set choice.
    let act = if backend.supports_shader_f16() {
        ActDtype::F16
    } else {
        ActDtype::F32
    };
    let ops = WgslConfig {
        bf16_quant_writes: manifest::current_recipe().bf16_quant_writes,
        act_dtype: act,
        weight_dtype: WeightDtype::Bf16,
    };
    let matmul = WgslConfig {
        weight_dtype: WeightDtype::Quant(QuantKind::Q8_0),
        ..ops
    };
    let cfgs = BlockWgslConfigs {
        matmul_qkv: matmul,
        matmul_qkv_self: matmul,
        matmul_proj: matmul,
        matmul_ffn_up: matmul,
        // Mirrors pipeline.rs encoder_cfgs: ffn_down weights stay bf16
        // (never transcoded; see `register_qwen3_handles`).
        matmul_ffn_down: ops,
        matmul_adaln: ops,
        ops,
        i8_sdpa: false,
        dense_acts: DenseActSites::default(),
        large_d_sdpa: false,
    };
    eprintln!("qwen3-parity: act_dtype={act:?}");
    let pipelines = BlockPipelines::compile(&backend, &cfgs)
        .await
        .expect("compile encoder pipelines");

    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));
    let encoder = Qwen3Encoder::new(seq);
    let mut taps = Qwen3Taps {
        want_layer_outputs: true,
        tap_block: Some(0),
        ..Default::default()
    };
    let out = encoder
        .forward_taps(
            &backend,
            &pipelines,
            &residency,
            &workspace,
            &handles,
            residency.source(),
            &ids,
            Some(&mut taps),
        )
        .await
        .expect("engine qwen3 forward");
    assert_eq!(out.seq, seq);

    // --- compare: embeds, per-layer residuals, layer-0 per-op ---
    let n_layers = text_encoder::config::HIDDEN_STATES_LAYER;
    assert_eq!(taps.layer_outputs.len(), n_layers);

    let report = |label: &str, exp: &[f32], got: &[f32]| -> (f64, f64) {
        assert_eq!(
            exp.len(),
            got.len(),
            "[{label}] length mismatch exp={} got={}",
            exp.len(),
            got.len()
        );
        let (slope, bias, rmse, cnt) = linfit(exp, got);
        let mean_abs = exp.iter().map(|x| x.abs() as f64).sum::<f64>() / (exp.len().max(1) as f64);
        let rel = if mean_abs > 0.0 { rmse / mean_abs } else { 0.0 };
        eprintln!(
            "[{label}] slope={slope:.6} bias={bias:+.4e} rmse={rmse:.4e} rel={:.3}% n={cnt}",
            rel * 100.0
        );
        (slope, rel)
    };

    let py = |name: &str| -> Vec<f32> { read_f32(&tmp.join(name)) };

    eprintln!("---- qwen3 parity: embeds ----");
    report("embeds", &py("py_qwen3_hs0.bin"), &taps.embeds);

    eprintln!("---- qwen3 parity: layer-0 per-op ----");
    let b0 = &taps.block_ops;
    for (name, got) in op_taps(b0) {
        let path = tmp.join(format!("py_qwen3_l0_{name}.bin"));
        if path.exists() {
            report(&format!("l0.{name}"), &read_f32(&path), got);
        } else {
            summarize(&format!("l0.{name} (no py dump)"), got);
        }
    }
    // Engine-only intermediates (no module to hook on the py side).
    summarize("l0.after_attn (engine only)", &b0.after_attn);
    summarize("l0.gu (engine only)", &b0.gu);

    eprintln!("---- qwen3 parity: per-layer residual ----");
    let write_f32 = |name: &str, v: &[f32]| {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(tmp.join(name), bytes).expect("write engine tap");
    };
    write_f32("eng_qwen3_hs0.bin", &taps.embeds);
    let mut rels: Vec<f64> = Vec::with_capacity(n_layers);
    let mut final_stats = (0.0f64, 0.0f64);
    for (i, got) in taps.layer_outputs.iter().enumerate() {
        let stats = report(
            &format!("layer{i:02}"),
            &py(&format!("py_qwen3_hs{}.bin", i + 1)),
            got,
        );
        write_f32(&format!("eng_qwen3_hs{}.bin", i + 1), got);
        rels.push(stats.1);
        if i == n_layers - 1 {
            final_stats = stats;
        }
    }

    // --- forensics: outlier-rounding vs normal-elem smear ---
    // Injection layer = first layer whose rel jumps >3x over its
    // predecessor (fallback: max-rel layer). Compare error structure
    // there and at the final hidden: f16 rounding concentrates squared
    // error on the massive-activation outlier elems; act_quant
    // block-crush smears it over normal elems sharing their rows.
    let suspect = rels
        .windows(2)
        .position(|w| w[1] > w[0].max(1e-6) * 3.0)
        .map(|p| p + 1)
        .unwrap_or_else(|| {
            rels.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap()
        });
    eprintln!("---- qwen3 parity: forensics (suspect layer {suspect:02}) ----");
    for i in [suspect.saturating_sub(1), suspect, n_layers - 1] {
        forensics(
            &format!("layer{i:02}"),
            &py(&format!("py_qwen3_hs{}.bin", i + 1)),
            &taps.layer_outputs[i],
            seq,
        );
    }

    // --- exact per-op localization at the suspect layer ---
    // Rerun the forward with the per-op taps pointed at the suspect layer
    // (weights are resident; the rerun is seconds) and compare every op
    // against the matching pyref hook dumps. The first op in dataflow
    // order (n1 -> q/k/v -> qn/kn -> qr/kr -> sa -> proj -> n2 ->
    // gate/up -> down) whose error structure shows the injection names
    // the culprit op; everything upstream of it is exonerated.
    if !tmp.join(format!("py_qwen3_l{suspect}_down.bin")).exists() {
        eprintln!("qwen3-parity: generating pyref per-op dumps for layer {suspect}");
        run_python_ref(&ids_path, &tmp, &[0, suspect]);
    }
    let mut taps_s = Qwen3Taps {
        tap_block: Some(suspect),
        ..Default::default()
    };
    let out_s = encoder
        .forward_taps(
            &backend,
            &pipelines,
            &residency,
            &workspace,
            &handles,
            residency.source(),
            &ids,
            Some(&mut taps_s),
        )
        .await
        .expect("engine qwen3 forward (suspect-layer taps)");
    assert_eq!(out_s.seq, seq);
    eprintln!("---- qwen3 parity: layer-{suspect} per-op ----");
    for (name, got) in op_taps(&taps_s.block_ops) {
        let path = tmp.join(format!("py_qwen3_l{suspect}_{name}.bin"));
        if path.exists() {
            let exp = read_f32(&path);
            report(&format!("l{suspect}.{name}"), &exp, got);
            forensics(&format!("l{suspect}.{name}"), &exp, got, seq);
        } else {
            summarize(&format!("l{suspect}.{name} (no py dump)"), got);
        }
    }
    summarize(
        &format!("l{suspect}.after_attn (engine only)"),
        &taps_s.block_ops.after_attn,
    );
    summarize(
        &format!("l{suspect}.gu (engine only)"),
        &taps_s.block_ops.gu,
    );

    let mut failures: Vec<String> = Vec::new();
    for i in 1..n_layers {
        if rels[i] > rels[i - 1].max(LAYER_REL_FLOOR) * LAYER_REL_JUMP_TOL {
            failures.push(format!(
                "layer{i:02} injects error: rel {:.3}% vs layer{:02} {:.3}% \
                 (jump tol {LAYER_REL_JUMP_TOL}x over floor {LAYER_REL_FLOOR})",
                rels[i] * 100.0,
                i - 1,
                rels[i - 1] * 100.0
            ));
        }
    }
    let (slope, rel) = final_stats;
    if (slope - 1.0).abs() > FINAL_SLOPE_TOL || rel > FINAL_REL_TOL {
        failures.push(format!(
            "final hidden diverges: slope={slope:.6} rel={:.3}% \
             (tol slope 1+-{FINAL_SLOPE_TOL}, rel {FINAL_REL_TOL})",
            rel * 100.0
        ));
    }
    assert!(
        failures.is_empty(),
        "qwen3 parity:\n{}",
        failures.join("\n")
    );
}

/// Per-op tap fields in dataflow order, paired with their pyref dump names.
fn op_taps(b: &Qwen3BlockOpsHost) -> [(&'static str, &Vec<f32>); 14] {
    [
        ("n1", &b.n1),
        ("q", &b.q),
        ("k", &b.k),
        ("v", &b.v),
        ("qn", &b.qn),
        ("kn", &b.kn),
        ("qr", &b.qr),
        ("kr", &b.kr),
        ("sa", &b.sa),
        ("proj", &b.proj),
        ("n2", &b.n2),
        ("gate", &b.gate),
        ("up", &b.up),
        ("down", &b.down),
    ]
}

/// Per-element error structure for one layer tap: splits elements into
/// outliers (top 0.1% by |pyref|, the massive-activation channels) vs
/// normal, and reports where the squared-error mass lives, plus per-token
/// and per-channel concentration. f16 rounding puts the error mass on the
/// outlier elems themselves; act_quant block-crush smears it across normal
/// elems sharing quant blocks/rows with them.
fn forensics(label: &str, exp: &[f32], got: &[f32], seq: usize) {
    let n = exp.len();
    assert_eq!(n, got.len(), "[{label}] forensics length mismatch");
    let hidden = n / seq;
    assert_eq!(seq * hidden, n, "[{label}] not row-divisible");

    let mut mags: Vec<f32> = exp.iter().map(|x| x.abs()).collect();
    mags.sort_unstable_by(f32::total_cmp);
    let mag_q = |p: f64| mags[((n - 1) as f64 * p) as usize] as f64;
    let cut = mag_q(0.999);
    eprintln!(
        "[{label}] |exp| p50={:.3e} p99={:.3e} p99.9={:.3e} max={:.3e} (outlier cut {cut:.3e})",
        mag_q(0.5),
        mag_q(0.99),
        mag_q(0.999),
        mag_q(1.0)
    );

    // (count, sum sq err, sum |exp|, |err| samples); [0]=normal [1]=outlier
    let mut sub: [(usize, f64, f64, Vec<f64>); 2] =
        [(0, 0.0, 0.0, Vec::new()), (0, 0.0, 0.0, Vec::new())];
    let mut row_sq = vec![0.0f64; seq];
    let mut ch_sq = vec![0.0f64; hidden];
    let mut ch_max = vec![0.0f32; hidden];
    for i in 0..n {
        let d = got[i] as f64 - exp[i] as f64;
        let s = &mut sub[usize::from(exp[i].abs() as f64 >= cut)];
        s.0 += 1;
        s.1 += d * d;
        s.2 += exp[i].abs() as f64;
        s.3.push(d.abs());
        row_sq[i / hidden] += d * d;
        ch_sq[i % hidden] += d * d;
        ch_max[i % hidden] = ch_max[i % hidden].max(exp[i].abs());
    }
    let total_sq = (sub[0].1 + sub[1].1).max(1e-30);
    for (name, s) in ["normal ", "outlier"].into_iter().zip(sub.iter_mut()) {
        s.3.sort_unstable_by(f64::total_cmp);
        let errs = &s.3;
        let err_q = |p: f64| errs[((errs.len().max(1) - 1) as f64 * p) as usize];
        let rmse = (s.1 / s.0.max(1) as f64).sqrt();
        let mean_abs = (s.2 / s.0.max(1) as f64).max(1e-30);
        eprintln!(
            "[{label}] {name}: n={} rel={:.3}% err p50={:.3e} p99={:.3e} max={:.3e} \
             sq-err share={:.1}%",
            s.0,
            100.0 * rmse / mean_abs,
            err_q(0.5),
            err_q(0.99),
            err_q(1.0),
            100.0 * s.1 / total_sq
        );
    }

    let mut rows: Vec<(usize, f64)> = row_sq
        .iter()
        .map(|&s| (s / hidden as f64).sqrt())
        .enumerate()
        .collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    let med = rows[rows.len() / 2].1;
    let worst: Vec<String> = rows
        .iter()
        .take(5)
        .map(|(r, e)| format!("t{r}={e:.3e}"))
        .collect();
    eprintln!(
        "[{label}] row rmse: median={med:.3e} worst {}",
        worst.join(" ")
    );

    let mut chs: Vec<(usize, f64)> = ch_sq
        .iter()
        .map(|&s| (s / seq as f64).sqrt())
        .enumerate()
        .collect();
    chs.sort_by(|a, b| b.1.total_cmp(&a.1));
    let worst: Vec<String> = chs
        .iter()
        .take(8)
        .map(|&(c, e)| format!("ch{c} rmse={e:.3e} max|exp|={:.3e}", ch_max[c]))
        .collect();
    eprintln!("[{label}] worst channels: {}", worst.join("; "));
}

fn run_python_ref(ids_path: &Path, out_dir: &Path, tap_layers: &[usize]) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let layers = tap_layers
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let status = Command::new("uv")
        .args([
            "run",
            "--directory",
            py_dir.to_str().unwrap(),
            "python",
            "-m",
            "thinfer_pytorch_ref.zimage.gen_qwen3_parity_ref",
            "--token-ids",
            ids_path.to_str().unwrap(),
            "--out",
            out_dir.to_str().unwrap(),
            "--tap-layers",
            &layers,
        ])
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "pytorch qwen3-parity ref failed");
}
