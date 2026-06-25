//! Qwen-Image-Edit text-encoder parity: the engine LM edit path (3-axis MRoPE +
//! vision-embed scatter into the `<|image_pad|>` slots) vs the HF Qwen2.5-VL-7B
//! LM dequantized from the SAME Q8_0 GGUF (`gen_encoder_edit_ref.py`), fed the
//! SAME tokens + the SAME vision-tower embeds (the pyref computes + dumps them;
//! the engine consumes them rather than running its own vision tower here).
//!
//! Isolates LM + MRoPE + scatter from the vision tower. Compares the final
//! `hidden_states[-1]` (post output_norm) -- the DiT edit-conditioning surface.
//! Same band as `encoder_parity` (bf16-act rounding; Qwen2.5 massive acts).

#![cfg(feature = "qwen-image-e2e")]

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
use thinfer_models::common::block::{BlockWgslConfigs, DenseActSites};
use thinfer_models::qwen_image::manifest::{self, role};
use thinfer_models::qwen_image::text_encoder::{
    EditEncoderPipelines, TextEncoder, qwen2vl_gguf_renames, register_handles,
};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, read_u32, report};

const PROMPT: &str = "make the apple green";
const GH: usize = 8;
const GW: usize = 8;

// bf16-act engine vs bf16 reference, same Q8_0 weights (mirror encoder_parity).
const SLOPE_TOL: f64 = 0.03;
const REL_TOL: f64 = 0.08;

#[tokio::test(flavor = "current_thread")]
async fn encoder_edit_parity() {
    let _trace = thinfer_core::trace::init_from_env();

    let enc_fr = manifest::MANIFEST
        .get(role::ENCODER_GGUF_Q8_0)
        .expect("encoder gguf role");
    let mm_fr = manifest::MANIFEST
        .get(role::MMPROJ_F16)
        .expect("mmproj role");
    let tok_fr = manifest::MANIFEST
        .get(role::TOKENIZER)
        .expect("tokenizer role");
    let (Some(gguf_path), Some(mmproj_path), Some(tok_path)) = (
        cache::resolve(enc_fr),
        cache::resolve(mm_fr),
        cache::resolve(tok_fr),
    ) else {
        eprintln!("skipped[encoder_edit_parity]: encoder/mmproj gguf or tokenizer not in HF cache");
        return;
    };
    let tok_dir = tok_path.parent().expect("tokenizer dir");

    // --- python reference (owns tokenization + vision tower; dumps embeds) ---
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("qwen_encoder_edit_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("config.txt");
    let cfg_tag = format!("{PROMPT}|{GH}x{GW}");
    let cached = tmp.join("hidden.bin").exists()
        && std::fs::read_to_string(&marker).is_ok_and(|c| c == cfg_tag);
    if cached {
        eprintln!(
            "encoder-edit-parity: reusing cached pyref ({})",
            tmp.display()
        );
    } else {
        for ent in std::fs::read_dir(&tmp).into_iter().flatten().flatten() {
            let _ = std::fs::remove_file(ent.path());
        }
        run_python_ref(&gguf_path, &mmproj_path, tok_dir, &tmp);
        std::fs::write(&marker, &cfg_tag).expect("write marker");
    }

    let ids = read_u32(&tmp.join("token_ids.bin"));
    let seq = ids.len();
    assert!(seq > 0, "pyref produced no tokens");
    let meta = std::fs::read_to_string(tmp.join("meta.txt")).expect("meta.txt");
    let m: Vec<usize> = meta
        .split_whitespace()
        .take(6)
        .map(|s| s.parse().expect("meta int"))
        .collect();
    let (seq_m, hidden_dim, pad_start, n_img, mgh, mgw) = (m[0], m[1], m[2], m[3], m[4], m[5]);
    assert_eq!(seq_m, seq, "pyref seq mismatch");
    assert_eq!(n_img, mgh * mgw);
    eprintln!(
        "encoder-edit-parity: {seq} tokens, hidden {hidden_dim}, image_pad_start={pad_start}, \
         merged {mgh}x{mgw} ({n_img} img tokens)"
    );

    let vision_embeds = read_f32(&tmp.join("vision_embeds.bin"));
    assert_eq!(vision_embeds.len(), n_img * hidden_dim);

    // --- engine encoder: GGUF Q8_0 (no transcode) + bf16 acts + MRoPE ---
    let opener = MmapFileOpener::new(&gguf_path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", gguf_path.display()));
    let gguf = GgufSource::open(opener).await.expect("parse encoder gguf");
    let source = RenamedSource::with_passthrough(gguf, qwen2vl_gguf_renames());
    let budget = ResidencyBudget {
        ram_bytes: 12 << 30,
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
        })
        .await
        .expect("wgpu adapter unavailable for tests"),
    );

    // bf16 acts (residual-overflow guard; same lesson as umT5/Qwen3).
    let ops = WgslConfig {
        bf16_quant_writes: false,
        act_dtype: ActDtype::Bf16,
        weight_dtype: WeightDtype::Bf16,
    };
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
        large_d_sdpa: false,
    };
    let pipelines = EditEncoderPipelines::compile(&backend, &cfgs)
        .await
        .expect("compile edit encoder pipelines");

    let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));
    let encoder = TextEncoder::new(seq.next_multiple_of(2));
    let out = encoder
        .forward_edit(
            &backend,
            &pipelines,
            &residency,
            &workspace,
            &handles,
            residency.source(),
            &ids,
            pad_start,
            &vision_embeds,
            (mgh, mgw),
            false,
        )
        .await
        .expect("engine encoder forward_edit");
    assert_eq!(out.seq, seq);
    assert_eq!(out.hidden.len(), seq * hidden_dim);

    eprintln!("---- qwen-image encoder edit parity ----");
    let exp = read_f32(&tmp.join("hidden.bin"));
    let (slope, rel) = report("hidden", &exp, &out.hidden);

    let mut failures = Vec::new();
    if (slope - 1.0).abs() > SLOPE_TOL || rel > REL_TOL {
        failures.push(format!(
            "hidden diverges: slope={slope:.6} rel={:.3}% (tol slope 1+-{SLOPE_TOL}, rel {REL_TOL})",
            rel * 100.0
        ));
    }
    assert!(
        failures.is_empty(),
        "encoder edit parity:\n{}",
        failures.join("\n")
    );
}

fn run_python_ref(gguf: &Path, mmproj: &Path, tok_dir: &Path, out_dir: &Path) {
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
            "thinfer_pytorch_ref.qwen_image.gen_encoder_edit_ref",
            "--gguf",
            gguf.to_str().unwrap(),
            "--mmproj",
            mmproj.to_str().unwrap(),
            "--tokenizer-dir",
            tok_dir.to_str().unwrap(),
            "--prompt",
            PROMPT,
            "--out",
            out_dir.to_str().unwrap(),
            "--gh",
            &GH.to_string(),
            "--gw",
            &GW.to_string(),
        ])
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "qwen-image encoder edit pyref failed");
}
