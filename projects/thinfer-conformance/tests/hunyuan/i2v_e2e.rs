//! HunyuanVideo 1.5 causal I2V (minWM WorldPlay dmd) end-to-end HEALTH gate:
//! real weights, engine-only (no pyref -- the minWM reference stack requires
//! CUDA+flash-attn; component faithfulness rides the shared T2V parity gates
//! for the common blocks). Drives the full chain the product path runs:
//! Qwen2.5-VL text encode -> SigLIP vision encode -> VAE-encode a synthetic
//! first frame -> chunked AR denoise (4 Euler steps + recache per chunk over
//! the host-staged KV cache) -> full VAE decode. Asserts finiteness, global
//! variance, and temporal motion; stages PNGs via `THINFER_E2E_PNG_DIR` for
//! the eyeball.
//!
//! Dims default tiny-OOD (448x256, 13 frames = one 4-latent-frame chunk) so
//! the gate is minutes; `THINFER_E2E_{WIDTH,HEIGHT,FRAMES,VRAM_GB}` scale it
//! up to the 832x480 product regime for the real quality eyeball.

#![cfg(feature = "hunyuan-e2e")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::format::union::RenamedSource;
use thinfer_core::ops::ActDtype;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::tokenizer::Tokenizer;
use thinfer_core::workspace::Workspace;
use thinfer_models::hunyuan::config::ar as arcfg;
use thinfer_models::hunyuan::dit::HunyuanDitPipelines;
use thinfer_models::hunyuan::dit::ar::HunyuanArDit;
use thinfer_models::hunyuan::encoder::{HunyuanTextEncoder, USER_MARKER, build_chat_prompt};
use thinfer_models::hunyuan::refiner::{HunyuanRefiner, HunyuanRefinerPipelines};
use thinfer_models::hunyuan::scheduler::FlowMatchSchedule;
use thinfer_models::hunyuan::siglip::{self, SiglipEncoder};
use thinfer_models::hunyuan::vae::encode::HunyuanVaeEncoder;
use thinfer_models::hunyuan::vae::{HunyuanVaeDecoder, HunyuanVaePipelines};
use thinfer_models::qwen_image::text_encoder::qwen2vl_gguf_renames;
use thinfer_native::MmapFileOpener;
use thinfer_native::tokenizer::HfTokenizer;

use crate::parity_util::resolve_hf;

const PROMPT: &str = "The subject slowly turns its head to the right while the \
camera holds a steady medium shot, soft natural lighting, realistic style.";

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Deterministic synthetic first frame: a vertical color gradient with a
/// bright disc, normalized to [-1, 1] CHW.
fn synthetic_frame(w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; 3 * h * w];
    let (cx, cy, r) = (w as f32 * 0.4, h as f32 * 0.45, h as f32 * 0.2);
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / w as f32;
            let fy = y as f32 / h as f32;
            let mut px = [0.2 + 0.6 * fy, 0.3 + 0.4 * fx, 0.7 - 0.5 * fy];
            let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
            if d < r {
                px = [0.9, 0.8, 0.3];
            }
            for c in 0..3 {
                out[c * h * w + y * w + x] = px[c] * 2.0 - 1.0;
            }
        }
    }
    out
}

/// Bilinear-resample a CHW [-1,1] frame (good enough for the synthetic input).
fn resample(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; 3 * dh * dw];
    for c in 0..3 {
        for y in 0..dh {
            for x in 0..dw {
                let fx = (x as f32 + 0.5) * sw as f32 / dw as f32 - 0.5;
                let fy = (y as f32 + 0.5) * sh as f32 / dh as f32 - 0.5;
                let (x0, y0) = (fx.floor().max(0.0) as usize, fy.floor().max(0.0) as usize);
                let (x1, y1) = ((x0 + 1).min(sw - 1), (y0 + 1).min(sh - 1));
                let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
                let at = |xx: usize, yy: usize| src[c * sh * sw + yy * sw + xx];
                let v = at(x0, y0) * (1.0 - tx) * (1.0 - ty)
                    + at(x1, y0) * tx * (1.0 - ty)
                    + at(x0, y1) * (1.0 - tx) * ty
                    + at(x1, y1) * tx * ty;
                out[c * dh * dw + y * dw + x] = v;
            }
        }
    }
    out
}

fn find_subsequence(hay: &[u32], needle: &[u32]) -> Option<usize> {
    if needle.is_empty() || needle.len() > hay.len() {
        return None;
    }
    (0..=hay.len() - needle.len()).find(|&i| &hay[i..i + needle.len()] == needle)
}

#[tokio::test(flavor = "current_thread")]
async fn i2v_e2e_health() {
    let _trace = thinfer_core::trace::init_from_env();

    let Some(dit_path) = resolve_hf(
        "THINFER_HUNYUAN_I2V_DIT",
        "models--MIN-Lab--minWM",
        "HY15/TI2V/dmd/diffusion_pytorch_model.safetensors",
    ) else {
        eprintln!("skipped[hunyuan i2v_e2e_health]: minWM dmd DiT not in HF cache");
        return;
    };
    let Some(siglip_path) = resolve_hf(
        "THINFER_HUNYUAN_SIGLIP",
        "models--Comfy-Org--sigclip_vision_384",
        "sigclip_vision_patch14_384.safetensors",
    ) else {
        eprintln!("skipped[hunyuan i2v_e2e_health]: sigclip vision tower not in HF cache");
        return;
    };
    let Some(vae_path) = resolve_hf(
        "THINFER_HUNYUAN_VAE",
        "models--Comfy-Org--HunyuanVideo_1.5_repackaged",
        "split_files/vae/hunyuanvideo15_vae_fp16.safetensors",
    ) else {
        eprintln!("skipped[hunyuan i2v_e2e_health]: Comfy VAE not in HF cache");
        return;
    };
    let Some(enc_path) = resolve_hf(
        "THINFER_HUNYUAN_ENCODER",
        "models--Phil2Sat--Qwen-Image-Edit-Rapid-AIO-GGUF",
        "Qwen2.5-VL-7B-Instruct-abliterated/Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf",
    ) else {
        eprintln!("skipped[hunyuan i2v_e2e_health]: encoder GGUF not in HF cache");
        return;
    };
    let Some(tok_path) = resolve_hf(
        "THINFER_HUNYUAN_TOKENIZER",
        "models--Qwen--Qwen2.5-VL-7B-Instruct",
        "tokenizer.json",
    ) else {
        eprintln!("skipped[hunyuan i2v_e2e_health]: tokenizer not in HF cache");
        return;
    };

    // Tiny-OOD health default: one 4-latent-frame chunk at 448x256.
    let width = env_usize("THINFER_E2E_WIDTH", 448);
    let height = env_usize("THINFER_E2E_HEIGHT", 256);
    let frames = env_usize("THINFER_E2E_FRAMES", 13);
    let vram_gb = env_usize("THINFER_E2E_VRAM_GB", 5) as u64;
    let grid_t = (frames - 1) / 4 + 1;
    let (grid_h, grid_w) = (height / 16, width / 16);
    assert!(
        grid_t.is_multiple_of(arcfg::CHUNK_LATENT_FRAMES),
        "frames must give latent frames divisible by 4"
    );
    let budget = ResidencyBudget {
        ram_bytes: 28 << 30,
        vram_bytes: vram_gb << 30,
    };
    eprintln!("i2v e2e: {width}x{height} f{frames} (grid {grid_t}x{grid_h}x{grid_w})");

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

    let frame_px = synthetic_frame(width, height);
    let siglip_px = resample(
        &frame_px,
        width,
        height,
        siglip::IMAGE_SIZE,
        siglip::IMAGE_SIZE,
    );

    // `THINFER_I2V_T2V_PROBE=1`: run the model TEXT-ONLY (no SigLIP tokens, no
    // cond latent/mask -- the upstream `mask_type="t2v"` shape). The dmd
    // checkpoint was trained i2v, so this probes whether text-only is usable at
    // all; a scene prompt replaces the motion prompt so there is something to
    // depict.
    let t2v_probe = std::env::var("THINFER_I2V_T2V_PROBE").is_ok_and(|v| v != "0");
    // `THINFER_E2E_PROMPT` overrides (drift is content-dependent; a motion-heavy
    // prompt stresses the AR cache harder than the static probe scenes).
    let prompt_env = std::env::var("THINFER_E2E_PROMPT").ok();
    let prompt = prompt_env.as_deref().unwrap_or(if t2v_probe {
        "A single bright yellow rubber duck floats on calm deep-blue water, \
         gentle ripples spreading outward, soft daylight, realistic style, \
         static camera."
    } else {
        PROMPT
    });

    // --- text encode (real engine encoder, same template/crop as the app) ---
    let t0 = Instant::now();
    let tokenizer = HfTokenizer::from_path(&tok_path).await.expect("tokenizer");
    let full = build_chat_prompt(prompt);
    let ids_full = tokenizer.encode(&full, false).expect("tokenize");
    let marker = tokenizer.encode(USER_MARKER, false).expect("marker");
    let crop = find_subsequence(&ids_full, &marker)
        .map(|i| i + marker.len())
        .unwrap_or(0);
    let token_ids = &ids_full[crop..];
    let seq = token_ids.len();
    let text = {
        let gguf = GgufSource::open(MmapFileOpener::new(&enc_path).await.expect("open enc"))
            .await
            .expect("parse encoder gguf");
        let renamed = RenamedSource::with_passthrough(gguf, qwen2vl_gguf_renames());
        let res = WeightResidency::new(renamed, budget);
        let enc = HunyuanTextEncoder::load(&backend, &res, seq + 1)
            .await
            .expect("load encoder");
        let ws = Workspace::new(Arc::clone(&backend), Arc::clone(res.arbiter()));
        let out = enc
            .encode(&backend, &res, &ws, token_ids)
            .await
            .expect("encode prompt");
        res.evict_all_and_free(&*backend);
        out
    };
    eprintln!(
        "text encode: {} tokens, {:.1}s",
        seq,
        t0.elapsed().as_secs_f32()
    );

    // --- VAE-encode the synthetic first frame (skipped by the t2v probe) ---
    let t0 = Instant::now();
    let vae_pl = HunyuanVaePipelines::compile_with(&backend, ActDtype::F32)
        .await
        .expect("compile vae");
    let cond0 = if t2v_probe {
        None
    } else {
        let src = SafetensorsSource::open(MmapFileOpener::new(&vae_path).await.expect("open vae"))
            .await
            .expect("parse vae");
        let res = WeightResidency::new(src, budget);
        let enc = HunyuanVaeEncoder::new(&res).expect("build vae encoder");
        let ws = Workspace::new(Arc::clone(&backend), Arc::clone(res.arbiter()));
        let out = enc
            .encode_frame(&backend, &res, &ws, &vae_pl, &frame_px, height, width)
            .await
            .expect("vae encode");
        res.evict_all_and_free(&*backend);
        assert_eq!(out.len(), 32 * grid_h * grid_w, "cond latent size");
        assert!(out.iter().all(|v| v.is_finite()), "cond latent finite");
        let cond_std = std(&out);
        eprintln!(
            "vae encode: std={cond_std:.3}, {:.1}s",
            t0.elapsed().as_secs_f32()
        );
        assert!(cond_std > 0.05, "cond latent degenerate (std {cond_std})");
        Some(out)
    };

    // --- DiT pipelines (shared by SigLIP + AR denoise) ---
    // i8 DP4A is the shipping default; `THINFER_HY_I8=0` bisects to pure bf16.
    let i8 = std::env::var("THINFER_HY_I8")
        .map(|v| v != "0")
        .unwrap_or(true);
    let dit_pipelines = HunyuanDitPipelines::compile_with(&backend, ActDtype::Bf16, i8)
        .await
        .expect("compile dit");

    // --- SigLIP vision encode (skipped by the t2v probe) ---
    let t0 = Instant::now();
    let vision = if t2v_probe {
        None
    } else {
        let src = SafetensorsSource::open(
            MmapFileOpener::new(&siglip_path)
                .await
                .expect("open siglip"),
        )
        .await
        .expect("parse siglip");
        let res = WeightResidency::new(src, budget);
        let sig = SiglipEncoder::new(&res).expect("build siglip");
        let ws = Workspace::new(Arc::clone(&backend), Arc::clone(res.arbiter()));
        let out = sig
            .encode(&backend, &res, &ws, &dit_pipelines.bp, &siglip_px)
            .await
            .expect("siglip encode");
        res.evict_all_and_free(&*backend);
        assert!(out.iter().all(|v| v.is_finite()), "vision tokens finite");
        let vis_std = std(&out);
        eprintln!(
            "siglip encode: std={vis_std:.3}, {:.1}s",
            t0.elapsed().as_secs_f32()
        );
        assert!(vis_std > 0.05, "vision tokens degenerate (std {vis_std})");
        Some(out)
    };

    // --- chunked AR denoise ---
    let t0 = Instant::now();
    let latent = {
        let src = SafetensorsSource::open(MmapFileOpener::new(&dit_path).await.expect("open dit"))
            .await
            .expect("parse dit");
        let res = WeightResidency::new(src, budget);
        // Streaming under pressure (as the app driver): workspace growth evicts
        // unpinned resident weights via the arbiter's reclaim chain, so the
        // budget holds without a predicted reserve.
        res.arbiter().register(
            thinfer_core::arbiter::RECLAIM_EVICTABLE_WEIGHTS,
            res.reclaimer(Arc::clone(&backend)),
        );
        let refiner = HunyuanRefiner::new(
            HunyuanRefinerPipelines::compile_with(&backend, ActDtype::F32)
                .await
                .expect("compile refiner"),
            &res,
        )
        .expect("build refiner");
        let dit = HunyuanArDit::new(dit_pipelines, refiner, &res, i8).expect("build ar dit");
        let ws = Workspace::new(Arc::clone(&backend), Arc::clone(res.arbiter()));
        let schedule =
            FlowMatchSchedule::build(&arcfg::DENOISING_STEP_LIST, arcfg::FLOW_SHIFT, 1000);
        let noise = seeded_gaussian(32 * grid_t * grid_h * grid_w, 42);
        let progress =
            |c: u32, nc: u32, i: u32, n: u32| eprintln!("  ar chunk {c}/{nc} forward {i}/{n}");
        let out = dit
            .generate(
                &backend,
                &res,
                &ws,
                &text,
                seq,
                vision.as_deref(),
                cond0.as_deref(),
                &noise,
                (grid_t, grid_h, grid_w),
                &schedule,
                Some(&progress),
                None,
            )
            .await
            .expect("ar generate");
        drop(ws);
        res.evict_all_and_free(&*backend);
        out
    };
    assert!(latent.iter().all(|v| v.is_finite()), "latent finite");
    let lat_std = std(&latent);
    eprintln!(
        "ar denoise: std={lat_std:.3}, {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    assert!(lat_std > 0.05, "denoised latent degenerate (std {lat_std})");

    // `THINFER_E2E_SKIP_DECODE=1`: drift-bisect mode. The latent-domain
    // per-chunk stats (`THINFER_AR_DIAG=1`) carry the drift verdict; skipping
    // the full-VAE decode cuts ~15-20 min per run at product dims.
    if std::env::var("THINFER_E2E_SKIP_DECODE").is_ok_and(|v| v != "0") {
        eprintln!("skipping VAE decode (THINFER_E2E_SKIP_DECODE)");
        return;
    }

    // --- VAE decode + health checks ---
    let t0 = Instant::now();
    let video = {
        let src = SafetensorsSource::open(MmapFileOpener::new(&vae_path).await.expect("open vae"))
            .await
            .expect("parse vae");
        let res = WeightResidency::new(src, budget);
        let vae = HunyuanVaeDecoder::new(vae_pl, &res).expect("build vae");
        let ws = Workspace::new(Arc::clone(&backend), Arc::clone(res.arbiter()));
        let out = vae
            .decode(&backend, &res, &ws, &latent, grid_t, grid_h, grid_w)
            .await
            .expect("vae decode");
        res.evict_all_and_free(&*backend);
        out
    };
    eprintln!("vae decode: {:.1}s", t0.elapsed().as_secs_f32());
    let f_px = 4 * (grid_t - 1) + 1;
    assert_eq!(video.len(), 3 * f_px * height * width, "video size");
    assert!(video.iter().all(|v| v.is_finite()), "video finite");
    let vid_std = std(&video);
    assert!(vid_std > 0.05, "video degenerate (std {vid_std})");
    // Temporal motion: the last frame must differ from the first (a frozen clip
    // means the AR chunks ignored the denoise).
    let frame_len = height * width;
    let per_c = f_px * frame_len;
    let mut diff = 0.0f64;
    for c in 0..3 {
        for p in 0..frame_len {
            let a = video[c * per_c + p];
            let b = video[c * per_c + (f_px - 1) * frame_len + p];
            diff += ((a - b) as f64).abs();
        }
    }
    let mad = diff / (3 * frame_len) as f64;
    eprintln!("i2v health: video std={vid_std:.3}, first-vs-last MAD={mad:.4}");
    assert!(mad > 1e-3, "no temporal motion (MAD {mad})");

    // Opt-in PNG staging for the eyeball.
    if let Some(dir) = std::env::var_os("THINFER_E2E_PNG_DIR").map(PathBuf::from) {
        std::fs::create_dir_all(&dir).expect("png dir");
        for f in 0..f_px {
            let mut chw = vec![0.0f32; frame_len * 3];
            for p in 0..frame_len {
                for c in 0..3 {
                    chw[c * frame_len + p] = video[c * per_c + f * frame_len + p];
                }
            }
            let png =
                thinfer_models::z_image::pipeline::encode_png(&chw, width as u32, height as u32)
                    .expect("encode png");
            std::fs::write(dir.join(format!("i2v_{f:03}.png")), png).expect("write png");
        }
        eprintln!("staged {} frames to {}", f_px, dir.display());
    }
}

/// Deterministic standard-normal noise (splitmix64 -> Box-Muller); mirrors the
/// app driver's seeding (no parity constraint on the RNG).
fn seeded_gaussian(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next_u64 = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut unit = || ((next_u64() >> 11) as f64 + 1.0) * (1.0 / (1u64 << 53) as f64);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = unit();
        let u2 = unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        out.push((r * theta.cos()) as f32);
        if out.len() < n {
            out.push((r * theta.sin()) as f32);
        }
    }
    out
}

fn std(x: &[f32]) -> f32 {
    let n = x.len().max(1) as f64;
    let mean = x.iter().map(|v| *v as f64).sum::<f64>() / n;
    let var = x.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / n;
    var.sqrt() as f32
}
