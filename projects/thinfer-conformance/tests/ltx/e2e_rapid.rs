//! ltx2-rapid (LTX-2 19B merge) t2v end-to-end HEALTH + perf gate. Drives the
//! net-new 19B conditioning chain on the real weights and asserts the DiT output
//! latents are finite + non-degenerate, then reports the `gpu_ms by pipeline`
//! rollup (the perf number). NOT a parity test (a community merge has no upstream
//! pyref); the 19B pieces are validated component-wise + against ComfyUI on the
//! same GGUF. Chain: tokenize -> Gemma encoder -> FE V1 (range-norm, single
//! bias-free aggregate) -> 2-layer ungated connector (both streams, inner 3840) ->
//! caption projection (3840 -> 4096/2048) -> DitModel(register_variant 19B) 8-step
//! single-stage denoise. The video VAE decode is opt-in (`THINFER_RAPID_VAE=1`);
//! the LTX-2 (non-.3) autoencoder differs structurally from our LTX-2.3 decoder
//! and is a separate follow-up, so the default gate stops at the DiT latents.
//!
//! Dims default tiny (9x64x64) for a cheap health run; scale to a product decode
//! with `THINFER_E2E_{FRAMES,HEIGHT,WIDTH,VRAM_GB}` to read the real per-step DiT
//! cost (the ~5s-video perf number). Serial GPU; `THINFER_TRACE=1
//! THINFER_POWER_PREF=high` for the rollup.

#![cfg(feature = "ltx-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::format::union::RenamedSource;
use thinfer_core::ops::WeightDtype;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::quant::QuantKind;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::ltx::LtxVariant;
use thinfer_models::ltx::connector::{
    ConnectorPipelines, RAPID_AUDIO, RAPID_VIDEO, connector_forward, fe_aggregate_v1,
    feature_extractor_v1_flatten, register_connector, register_fe_v1,
};
use thinfer_models::ltx::dit::{DitModel, DitPipelines, caption_project, register_caption_proj};
use thinfer_models::ltx::loader::{UnitOffsetSource, gemma_gguf_renames, gemma_norm_offset_ids};
use thinfer_models::ltx::manifest::{self, role};
use thinfer_models::ltx::pipeline::{build_dit_freqs, denoise_loop, streams_for};
use thinfer_models::ltx::sampler::{self, AudioLatentDims, STAGE1_SIGMAS, VideoLatentDims};
use thinfer_models::ltx::text_encoder::{
    GemmaEncoder, GemmaEncoderPipelines, gemma_encoder_cfgs, register_handles,
};
use thinfer_models::ltx::video_vae::{
    LATENT_CHANNELS, LtxVaeConfig, LtxVaeDecoder, LtxVaePipelines, load_latent_stats,
};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::read_u32;

const PROMPT: &str = "a red fox leaps over a snowy log at dawn";
const FRAMES: usize = 9;
const HEIGHT: usize = 64;
const WIDTH: usize = 64;
const FPS: f64 = 24.0;
const SEED: u64 = 42;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[tokio::test(flavor = "current_thread")]
async fn rapid_t2v_e2e_health() {
    let trace_handle = thinfer_core::trace::init_from_env();

    // Prompt is overridable (`THINFER_RAPID_PROMPT`) so a run can A/B whether the
    // conditioning chain actually steers the output.
    let prompt = std::env::var("THINFER_RAPID_PROMPT").unwrap_or_else(|_| PROMPT.to_string());
    eprintln!("ltx rapid e2e: prompt={prompt:?}");

    let resolve = |r: &str| cache::resolve(manifest::LTX2_RAPID_MANIFEST.get(r).expect("role"));
    let vae_path = resolve(role::VIDEO_VAE);
    let (Some(gemma_path), Some(conn_path), Some(dit_path), Some(tok_path)) = (
        resolve(role::ENCODER_GGUF),
        resolve(role::CONNECTOR),
        resolve(role::DIT_GGUF_Q5_K_M),
        resolve(role::TOKENIZER),
    ) else {
        eprintln!("skipped[ltx rapid_t2v_e2e_health]: rapid weight files not in HF cache");
        return;
    };

    // --- token ids (cheap tokenize-only pyref, cached). Uses the PRODUCT
    // tokenizer.json (what the engine feeds), NOT the degenerate GGUF tokenizer. ---
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ltx_rapid_e2e");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("prompt.txt");
    let marker_val = format!("{prompt}\ntok=product");
    let cached = tmp.join("token_ids.bin").exists()
        && std::fs::read_to_string(&marker).is_ok_and(|p| p == marker_val);
    if !cached {
        run_tokens_ref(&tok_path, &tmp, &prompt);
        std::fs::write(&marker, &marker_val).expect("write marker");
    }
    let ids = read_u32(&tmp.join("token_ids.bin"));
    assert!(!ids.is_empty(), "no tokens");
    let seq = ids.len();
    eprintln!("ltx rapid e2e: {seq} tokens");

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

    let vram_gb = env_usize("THINFER_E2E_VRAM_GB", 6);
    let budget = move || ResidencyBudget {
        ram_bytes: 16 << 30,
        vram_bytes: (vram_gb as u64) << 30,
    };

    let frames = env_usize("THINFER_E2E_FRAMES", FRAMES);
    let height = env_usize("THINFER_E2E_HEIGHT", HEIGHT);
    let width = env_usize("THINFER_E2E_WIDTH", WIDTH);
    let vd = VideoLatentDims::from_pixels(frames, height, width);
    let ad = AudioLatentDims::from_video(frames, FPS);
    eprintln!(
        "ltx rapid e2e: video latent {}x{}x{} ({} tok), audio {} tok",
        vd.frames,
        vd.height,
        vd.width,
        vd.tokens(),
        ad.tokens(),
    );

    // === Phase A: Gemma-3 text encoder -> 49 hidden states (host) ===
    let states = {
        let opener = MmapFileOpener::new(&gemma_path).await.expect("open gemma");
        let gguf = GgufSource::open(opener).await.expect("parse gemma gguf");
        let renamed = RenamedSource::with_passthrough(gguf, gemma_gguf_renames());
        let source = UnitOffsetSource::new(renamed, gemma_norm_offset_ids());
        let residency = WeightResidency::new(source, budget());
        let handles = register_handles(&residency, None).expect("register encoder");
        let cfgs = gemma_encoder_cfgs(WeightDtype::Quant(QuantKind::Q8_0));
        let pipelines = GemmaEncoderPipelines::compile(&backend, &cfgs)
            .await
            .expect("compile gemma pipelines");
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
            .expect("gemma encoder forward");
        residency.evict_all_and_free(&*backend);
        out.states
    };

    // === Phase B: FE V1 + 2-layer ungated connector (connector safetensors) ===
    // FE V1 = per-layer range-norm -> single bias-free aggregate embed [seq,3840],
    // one shared stream fed to BOTH connectors (each frames with its own registers).
    let flat = feature_extractor_v1_flatten(&states, seq);
    drop(states);
    let conn_pipes = ConnectorPipelines::compile_rapid(&backend)
        .await
        .expect("compile rapid connector pipelines");
    let (video_conn, audio_conn) = {
        let src = open_st(&conn_path).await;
        let residency = WeightResidency::new(src, budget());
        let fe_w = register_fe_v1(&residency).expect("register FE V1");
        let video_h = register_connector(&residency, RAPID_VIDEO).expect("register video conn");
        let audio_h = register_connector(&residency, RAPID_AUDIO).expect("register audio conn");
        let ws = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));
        // out_dim = gemma HIDDEN = 3840 (single shared aggregate).
        let agg = fe_aggregate_v1(
            &backend,
            &conn_pipes,
            &residency,
            &ws,
            &flat,
            seq,
            fe_w,
            3840,
        )
        .await
        .expect("fe v1 aggregate");
        if let Some(dir) = std::env::var_os("THINFER_RAPID_DUMP_COND").map(PathBuf::from) {
            std::fs::create_dir_all(&dir).expect("dump dir");
            let bytes: Vec<u8> = agg.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(dir.join("agg.bin"), bytes).unwrap();
            eprintln!("DIAG dumped agg[{}] (seq={seq})", agg.len());
        }
        let vconn = connector_forward(
            &backend,
            &conn_pipes,
            &residency,
            &ws,
            &video_h,
            RAPID_VIDEO,
            &agg,
            seq,
        )
        .await
        .expect("video connector");
        let aconn = connector_forward(
            &backend,
            &conn_pipes,
            &residency,
            &ws,
            &audio_h,
            RAPID_AUDIO,
            &agg,
            seq,
        )
        .await
        .expect("audio connector");
        residency.evict_all_and_free(&*backend);
        (vconn, aconn)
    };

    // === Phases C-D: DiT GGUF resident: caption projection + 8-step denoise ===
    let opener = MmapFileOpener::new(&dit_path).await.expect("open dit");
    let dit_src = GgufSource::open(opener).await.expect("parse dit gguf");
    let dit_res = WeightResidency::new(dit_src, budget());
    let dit_pipes = DitPipelines::compile(&backend).await.expect("compile dit");
    let cap_h = register_caption_proj(&dit_res).expect("register caption proj");
    let dit_model = DitModel::register_variant(
        &backend,
        &dit_res,
        thinfer_models::ltx::config::NUM_LAYERS,
        LtxVariant::ltx2_rapid_19b(),
    )
    .await
    .expect("register rapid dit");
    let dit_ws = Workspace::new(Arc::clone(&backend), Arc::clone(dit_res.arbiter()));

    // caption projection: connector out [1024,3840] -> vtext[1024,4096]/atext[1024,2048].
    let rows = thinfer_models::ltx::connector::CONN_SEQ;
    let (vtext, atext) = caption_project(
        &backend,
        &dit_pipes,
        &dit_res,
        &dit_ws,
        &cap_h,
        &video_conn,
        &audio_conn,
        rows,
    )
    .await
    .expect("caption projection");
    assert!(vtext.iter().all(|v| v.is_finite()), "vtext non-finite");
    assert!(atext.iter().all(|v| v.is_finite()), "atext non-finite");

    // DIAG: dump connector output + caption-projected vtext for an offline numpy
    // check of caption_projection (isolates a GPU bug from a real math nulling).
    if let Some(dir) = std::env::var_os("THINFER_RAPID_DUMP_COND").map(PathBuf::from) {
        std::fs::create_dir_all(&dir).expect("dump dir");
        let as_le = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        std::fs::write(dir.join("video_conn.bin"), as_le(&video_conn)).unwrap();
        std::fs::write(dir.join("vtext.bin"), as_le(&vtext)).unwrap();
        eprintln!(
            "DIAG dumped video_conn[{}] vtext[{}] to {}",
            video_conn.len(),
            vtext.len(),
            dir.display()
        );
    }

    // DIAG: per-row RMS of a real token row vs a register row, at the connector
    // output (3840) and post caption-proj vtext (4096). norm_out should leave every
    // connector-output row at unit RMS; a big real-vs-register RMS gap localizes
    // where the caption signal balance breaks.
    {
        let rms = |v: &[f32], row: usize, d: usize| -> f64 {
            let s: f64 = v[row * d..(row + 1) * d]
                .iter()
                .map(|&x| (x as f64).powi(2))
                .sum();
            (s / d as f64).sqrt()
        };
        let cd = RAPID_VIDEO.inner_dim; // 3840 (rapid connector inner; NOT 22B's 4096)
        let vd_ = thinfer_models::ltx::config::DIM; // 4096
        // LEFT-pad: real tokens occupy the TRAILING rows; registers lead. Sample the
        // last real row (`rows-1`) vs a leading register row (0).
        let real_row = rows - 1;
        eprintln!(
            "DIAG conn[3840] real_row{real_row}_rms={:.4} reg_row0_rms={:.4} | vtext[4096] real_row{real_row}_rms={:.4} reg_row0_rms={:.4}",
            rms(&video_conn, real_row, cd),
            rms(&video_conn, 0, cd),
            rms(&vtext, real_row, vd_),
            rms(&vtext, 0, vd_),
        );
    }

    // Single-stage 8-step denoise directly at the target res.
    let mut s = streams_for(vd, ad);
    // LEFT-pad caption: the real prompt tokens are the TRAILING `seq` rows; the
    // leading `rows-seq` rows are learnable registers.
    let vd_dim = thinfer_models::ltx::config::DIM;
    let ad_dim = thinfer_models::ltx::config::AUDIO_DIM;
    let (vtext, atext) = if std::env::var_os("THINFER_RAPID_REAL_ONLY").is_some() {
        // Truncate the caption KV to just the real prompt tokens (drop the register
        // pad rows) so cross-attn attends only to them.
        s.video_text = seq;
        s.audio_text = seq;
        eprintln!("DIAG: caption truncated to {seq} real tokens (registers dropped)");
        (
            vtext[(rows - seq) * vd_dim..].to_vec(),
            atext[(rows - seq) * ad_dim..].to_vec(),
        )
    } else if let Ok(cl) = std::env::var("THINFER_RAPID_CAP_LEN") {
        // Attend to a SHORTER caption (real tokens + fewer leading register rows) to
        // test whether the 1024-length register padding dilutes conditioning.
        let n: usize = cl.parse().unwrap_or(rows).clamp(seq, rows);
        s.video_text = n;
        s.audio_text = n;
        eprintln!(
            "DIAG: caption length {n} ({seq} real + {} registers)",
            n - seq
        );
        (
            vtext[(rows - n) * vd_dim..].to_vec(),
            atext[(rows - n) * ad_dim..].to_vec(),
        )
    } else if let Ok(amp) = std::env::var("THINFER_RAPID_AMP") {
        // Amplify the real prompt-token caption rows (keep registers) to test
        // whether the ~1006 null-register rows dilute conditioning in cross-attn.
        let f: f32 = amp.parse().unwrap_or(1.0);
        eprintln!("DIAG: amplifying {seq} real caption rows x{f}");
        let mut v = vtext.clone();
        let mut a = atext.clone();
        v[(rows - seq) * vd_dim..].iter_mut().for_each(|x| *x *= f);
        a[(rows - seq) * ad_dim..].iter_mut().for_each(|x| *x *= f);
        (v, a)
    } else {
        (vtext, atext)
    };
    let freqs = build_dit_freqs(vd, ad, FPS);
    let lat_v = sampler::gaussian_noise(vd.elems(), sampler::substream_seed(SEED, 0));
    let lat_a = sampler::gaussian_noise(ad.elems(), sampler::substream_seed(SEED, 1));
    let (lat_v_final, lat_a_final) = denoise_loop(
        &backend,
        &dit_pipes,
        &dit_res,
        &dit_ws,
        &dit_model,
        s,
        &STAGE1_SIGMAS,
        lat_v,
        lat_a,
        &vtext,
        &atext,
        &freqs,
        None, // no i2v conditioning (t2v health check)
        None,
    )
    .await
    .expect("denoise");

    // --- Health assertions on the DiT latents ---
    assert!(
        lat_v_final.iter().all(|v| v.is_finite()),
        "video latents non-finite"
    );
    assert!(
        lat_a_final.iter().all(|v| v.is_finite()),
        "audio latents non-finite"
    );
    let n = lat_v_final.len() as f64;
    let m = lat_v_final.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = lat_v_final
        .iter()
        .map(|&v| (v as f64 - m).powi(2))
        .sum::<f64>()
        / n;
    eprintln!(
        "ltx rapid e2e: video latent [{},{}] mean={m:.4} var={var:.6} min={:.3} max={:.3}",
        vd.tokens(),
        thinfer_models::ltx::config::IN_CHANNELS,
        lat_v_final.iter().cloned().fold(f32::INFINITY, f32::min),
        lat_v_final
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max),
    );
    assert!(var > 1e-8, "degenerate (flat) video latent: var={var}");
    eprintln!("ltx rapid e2e: DiT + conditioning chain OK (finite, non-degenerate)");

    drop(dit_model);
    dit_res.evict_all_and_free(&*backend);
    drop(dit_res);

    // === Phase E: video VAE decode (opt-in) -> frames ===
    // The LTX-2 (non-.3) decoder (`LtxVaeConfig::ltx2_rapid`). Off by default so the
    // health gate stays cheap; `THINFER_RAPID_VAE=1` exercises the full pixel path.
    if std::env::var("THINFER_RAPID_VAE").is_ok() {
        let Some(vae_path) = vae_path else {
            eprintln!("skipped[rapid vae]: video vae not in cache");
            return;
        };
        let cthw = sampler::video_tokens_to_cthw(&lat_v_final, vd);
        let src = open_st(&vae_path).await;
        let residency = WeightResidency::new(src, budget());
        let (mean, std) = load_latent_stats(&residency, &backend)
            .await
            .expect("load latent stats");
        assert_eq!(mean.len(), LATENT_CHANNELS);
        let pipes = LtxVaePipelines::compile(&backend)
            .await
            .expect("compile vae");
        let decoder = LtxVaeDecoder::new_with_config(
            pipes,
            &residency,
            mean,
            std,
            LtxVaeConfig::ltx2_rapid(),
        )
        .expect("vae decoder");
        let ws = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));
        let video = decoder
            .decode(
                &backend, &residency, &ws, &cthw, vd.frames, vd.height, vd.width,
            )
            .await
            .expect("vae decode");
        let f_px = 8 * (vd.frames - 1) + 1;
        let h_px = 32 * vd.height;
        let w_px = 32 * vd.width;
        assert_eq!(video.len(), 3 * f_px * h_px * w_px, "video shape");
        assert!(video.iter().all(|v| v.is_finite()), "video non-finite");
        let vn = video.len() as f64;
        let vm = video.iter().map(|&v| v as f64).sum::<f64>() / vn;
        let vvar = video.iter().map(|&v| (v as f64 - vm).powi(2)).sum::<f64>() / vn;
        eprintln!(
            "ltx rapid e2e: decoded video [3,{f_px},{h_px},{w_px}] mean={vm:.4} var={vvar:.6} min={:.3} max={:.3}",
            video.iter().cloned().fold(f32::INFINITY, f32::min),
            video.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        );
        assert!(vvar > 1e-6, "degenerate (flat) video: var={vvar}");

        // Stage per-frame PNGs for a visual eyeball (`THINFER_E2E_PNG_DIR`).
        // VAE output is [C=3, T=f_px, H=h_px, W=w_px] contiguous.
        if let Some(dir) = std::env::var_os("THINFER_E2E_PNG_DIR").map(PathBuf::from) {
            std::fs::create_dir_all(&dir).expect("png dir");
            let frame_len = h_px * w_px;
            let per_c = f_px * frame_len;
            for f in 0..f_px {
                let mut chw = vec![0.0f32; frame_len * 3];
                for c in 0..3 {
                    for p in 0..frame_len {
                        chw[c * frame_len + p] = video[c * per_c + f * frame_len + p];
                    }
                }
                let png =
                    thinfer_models::z_image::pipeline::encode_png(&chw, w_px as u32, h_px as u32)
                        .expect("encode png");
                std::fs::write(dir.join(format!("rapid_{f:03}.png")), png).expect("write png");
            }
            eprintln!("ltx rapid e2e: staged {f_px} frames to {}", dir.display());
        }
        residency.evict_all_and_free(&*backend);
    }

    // Dump the `gpu_ms by pipeline` rollup (THINFER_TRACE) so a traced run localizes
    // the DiT step cost (matmul vs SDPA vs dequant). init_from_env only COLLECTS;
    // the handle must be dumped explicitly.
    if let Some(h) = &trace_handle {
        h.dump(&mut std::io::stderr()).ok();
    }
}

async fn open_st(path: &Path) -> SafetensorsSource<MmapFileOpener> {
    let opener = MmapFileOpener::new(path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    SafetensorsSource::open(opener)
        .await
        .unwrap_or_else(|e| panic!("parse {}: {e:?}", path.display()))
}

fn run_tokens_ref(tokenizer: &Path, out_dir: &Path, prompt: &str) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let status = Command::new("uv")
        .args([
            "run",
            "--directory",
            py_dir.to_str().unwrap(),
            "--with",
            "tokenizers",
            "python",
            "-m",
            "thinfer_pytorch_ref.ltx.gen_tokens_ref",
            "--tokenizer",
            tokenizer.to_str().unwrap(),
            "--prompt",
            prompt,
            "--out",
            out_dir.to_str().unwrap(),
        ])
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "ltx tokens pyref failed");
}
