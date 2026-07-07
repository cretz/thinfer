//! LTX-2.3 distilled silent-video t2v end-to-end HEALTH gate (P4 first
//! deliverable). Drives the full chain on the real weights at minimal dims and
//! asserts the output is a finite, non-degenerate video. NOT a parity test: there
//! is no full-pipeline pyref (the 48-block DiT ref OOMs); each component is parity-
//! gated separately (`encoder_parity`, `connector_parity`, `dit_parity`,
//! `vae_parity`, `upsampler_parity`). This proves they compose: tokenize -> Gemma
//! encoder -> FE V2 -> 8-layer connector -> denoise -> video VAE decode -> frames.
//! Default = single-stage (8-step denoise at the target res, the product
//! default); set `THINFER_E2E_UPSCALE=1` for the two-stage path (stage1 8 steps
//! half-res -> latent upscale x2 -> stage2 renoise + 3-step refine full-res).
//! Joint AV: the audio stream runs through the DiT and its
//! decode tail (audio VAE -> mel -> vocoder -> 48kHz wav) is also exercised here;
//! both video and audio are asserted finite + non-degenerate.
//!
//! Per-phase scoped `WeightResidency` (one per weight file, built -> used ->
//! dropped) so only one large weight set is VRAM-resident at a time (Gemma, then
//! the DiT GGUF across conditioning+denoise, then the VAE), matching the upstream
//! `DiffusionStage` lifecycle.

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
use thinfer_models::ltx::audio_vae::{
    AudioVaeDecoder, AudioVaePipelines, load_latent_stats as load_audio_latent_stats,
};
use thinfer_models::ltx::connector::{
    AUDIO, ConnectorPipelines, VIDEO, connector_forward, fe_aggregate,
    feature_extractor_v2_flatten, register_connector, register_fe,
};
use thinfer_models::ltx::dit::{DitModel, DitPipelines};
use thinfer_models::ltx::loader::{UnitOffsetSource, gemma_gguf_renames, gemma_norm_offset_ids};
use thinfer_models::ltx::manifest::{self, role};
use thinfer_models::ltx::pipeline::{
    build_dit_freqs, denoise_loop, normalize_cthw, streams_for, un_normalize_cthw,
};
use thinfer_models::ltx::sampler::{
    self, AudioLatentDims, STAGE1_SIGMAS, STAGE2_SIGMAS, VideoLatentDims,
};
use thinfer_models::ltx::text_encoder::{
    GemmaEncoder, GemmaEncoderPipelines, gemma_encoder_cfgs, register_handles,
};
use thinfer_models::ltx::upsampler::{LtxUpsampler, LtxUpsamplerPipelines};
use thinfer_models::ltx::video_vae::{
    LATENT_CHANNELS, LtxVaeDecoder, LtxVaePipelines, load_latent_stats,
};
use thinfer_models::ltx::vocoder::{Vocoder, VocoderPipelines};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::read_u32;

const PROMPT: &str = "a red fox leaps over a snowy log at dawn";

// Minimal two-stage dims: frames 8k+1, H/W divisible by 64 (two-stage halves to a
// /32-divisible stage-1 res). 9 frames -> 2 latent frames; 64x64 full -> stage1
// 32x32 (latent 1x1) -> upscale -> full latent 2x2. Overridable via
// `THINFER_E2E_{FRAMES,HEIGHT,WIDTH,VRAM_GB}` to drive the product-size decode
// (e.g. 121x320x512 @ 6GB) through the real chain -- the health gate keeps the
// tiny defaults so it stays cheap.
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
async fn t2v_e2e_health() {
    let _trace = thinfer_core::trace::init_from_env();

    let resolve = |r: &str| cache::resolve(manifest::MANIFEST.get(r).expect("role"));
    let (
        Some(gemma_path),
        Some(conn_path),
        Some(dit_path),
        Some(vae_path),
        Some(upscaler_path),
        Some(audio_vae_path),
    ) = (
        resolve(role::ENCODER_GGUF),
        resolve(role::CONNECTOR),
        resolve(role::DIT_GGUF_Q8_0),
        resolve(role::VIDEO_VAE),
        resolve(role::UPSCALER),
        resolve(role::AUDIO_VAE),
    )
    else {
        eprintln!("skipped[ltx t2v_e2e_health]: one or more LTX weight files not in HF cache");
        return;
    };

    // --- token ids (cheap tokenize-only pyref, cached) ---
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ltx_e2e");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("prompt.txt");
    let cached = tmp.join("token_ids.bin").exists()
        && std::fs::read_to_string(&marker).is_ok_and(|p| p == PROMPT);
    if !cached {
        run_tokens_ref(&gemma_path, &tmp);
        std::fs::write(&marker, PROMPT).expect("write marker");
    }
    let ids = read_u32(&tmp.join("token_ids.bin"));
    assert!(!ids.is_empty(), "no tokens");
    eprintln!("ltx e2e: {} tokens", ids.len());

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

    // Latent grids: stage 1 (half res), full res, audio (duration-derived).
    let frames = env_usize("THINFER_E2E_FRAMES", FRAMES);
    let height = env_usize("THINFER_E2E_HEIGHT", HEIGHT);
    let width = env_usize("THINFER_E2E_WIDTH", WIDTH);
    let vd_s1 = VideoLatentDims::from_pixels(frames, height / 2, width / 2);
    let vd_full = VideoLatentDims::from_pixels(frames, height, width);
    let ad = AudioLatentDims::from_video(frames, FPS);
    eprintln!(
        "ltx e2e: stage1 video latent {}x{}x{} ({} tok), full {}x{}x{} ({} tok), audio {} tok",
        vd_s1.frames,
        vd_s1.height,
        vd_s1.width,
        vd_s1.tokens(),
        vd_full.frames,
        vd_full.height,
        vd_full.width,
        vd_full.tokens(),
        ad.tokens(),
    );

    // === Phase: VAE per-channel latent stats (scoped; reopened for decode) ===
    let (mean, std) = {
        let src = open_st(&vae_path).await;
        let residency = WeightResidency::new(src, budget());
        let stats = load_latent_stats(&residency, &backend)
            .await
            .expect("load latent stats");
        residency.evict_all_and_free(&*backend);
        stats
    };
    assert_eq!(mean.len(), LATENT_CHANNELS);

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
    let seq = ids.len();

    // The connector pipelines are reused across the FE (connector ST) and the
    // connector blocks (DiT GGUF) phases.
    let conn_pipes = ConnectorPipelines::compile(&backend)
        .await
        .expect("compile connector pipelines");

    // === Phase B: FeatureExtractor V2 -> aggregate embeds (host) ===
    let flat = feature_extractor_v2_flatten(&states, seq);
    drop(states);
    let (video_embed, audio_embed) = {
        let src = open_st(&conn_path).await;
        let residency = WeightResidency::new(src, budget());
        let fe = register_fe(&residency).expect("register FE");
        let workspace = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));
        let v = fe_aggregate(
            &backend,
            &conn_pipes,
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
            &conn_pipes,
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
        residency.evict_all_and_free(&*backend);
        (v, a)
    };

    // === Phases C-F: DiT GGUF resident for connector blocks + both denoise stages ===
    let opener = MmapFileOpener::new(&dit_path).await.expect("open dit");
    let dit_src = GgufSource::open(opener).await.expect("parse dit gguf");
    let dit_res = WeightResidency::new(dit_src, budget());
    let video_h = register_connector(&dit_res, VIDEO).expect("register video connector");
    let audio_h = register_connector(&dit_res, AUDIO).expect("register audio connector");
    let dit_pipes = DitPipelines::compile(&backend).await.expect("compile dit");
    let dit_model = DitModel::register(&backend, &dit_res, thinfer_models::ltx::config::NUM_LAYERS)
        .await
        .expect("register dit model");
    let dit_ws = Workspace::new(Arc::clone(&backend), Arc::clone(dit_res.arbiter()));

    // Connector cross-attn KV (all 1024 positions valid; registers fill pads).
    let vtext = connector_forward(
        &backend,
        &conn_pipes,
        &dit_res,
        &dit_ws,
        &video_h,
        VIDEO,
        &video_embed,
        seq,
    )
    .await
    .expect("video connector");
    let atext = connector_forward(
        &backend,
        &conn_pipes,
        &dit_res,
        &dit_ws,
        &audio_h,
        AUDIO,
        &audio_embed,
        seq,
    )
    .await
    .expect("audio connector");

    // Render path: default single-stage (8-step denoise at the target res, the
    // product default), or the opt-in two-stage upscale refine when
    // `THINFER_E2E_UPSCALE` is set. Both compose the same components; the
    // upscale path additionally exercises the latent upsampler + stage-2 renoise.
    let upscale = std::env::var("THINFER_E2E_UPSCALE").is_ok();
    eprintln!(
        "ltx e2e: render path = {}",
        if upscale {
            "two-stage upscale"
        } else {
            "single-stage"
        }
    );
    let (lat_v_final, lat_a_final) = if upscale {
        // --- Stage 1: pure-noise init, 8-step denoise at half res ---
        let s1 = streams_for(vd_s1, ad);
        let freqs_s1 = build_dit_freqs(vd_s1, ad, FPS);
        let lat_v = sampler::gaussian_noise(vd_s1.elems(), sampler::substream_seed(SEED, 0));
        let lat_a = sampler::gaussian_noise(ad.elems(), sampler::substream_seed(SEED, 1));
        let (lat_v, lat_a) = denoise_loop(
            &backend,
            &dit_pipes,
            &dit_res,
            &dit_ws,
            &dit_model,
            s1,
            &STAGE1_SIGMAS,
            lat_v,
            lat_a,
            &vtext,
            &atext,
            &freqs_s1,
            None, // no i2v conditioning
            None,
        )
        .await
        .expect("stage1 denoise");
        assert!(
            lat_v.iter().all(|v| v.is_finite()),
            "stage1 video non-finite"
        );

        // --- Upscale x2 (un-normalize -> upsample -> re-normalize) ---
        let mut cthw = sampler::video_tokens_to_cthw(&lat_v, vd_s1);
        let thw_s1 = vd_s1.tokens();
        un_normalize_cthw(&mut cthw, &mean, &std, thw_s1);
        // The upscaler is its own phase sandwiched between the two denoise stages.
        // Free the DiT's GPU-cached block weights so the upsampler (~1GB) has the
        // full card; stage2 re-streams the DiT from disk anyway (budget << 22.8GB
        // model). Without it the DiT cache (~budget) + upsampler OOMs 8GB.
        dit_res.evict_all_and_free(&*backend);
        let mut upscaled = {
            let src = open_st(&upscaler_path).await;
            let residency = WeightResidency::new(src, budget());
            let pipes = LtxUpsamplerPipelines::compile(&backend)
                .await
                .expect("compile upsampler");
            let ups = LtxUpsampler::new(pipes, &residency).expect("build upsampler");
            let ws = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));
            let up = ups
                .forward(
                    &backend,
                    &residency,
                    &ws,
                    &cthw,
                    vd_s1.frames,
                    vd_s1.height,
                    vd_s1.width,
                )
                .await
                .expect("upsample");
            residency.evict_all_and_free(&*backend);
            up
        };
        let thw_full = vd_full.tokens();
        assert_eq!(upscaled.len(), LATENT_CHANNELS * thw_full, "upscaled size");
        normalize_cthw(&mut upscaled, &mean, &std, thw_full);
        let lat_v_full = sampler::video_cthw_to_tokens(&upscaled, vd_full);

        // --- Stage 2: renoise to STAGE2_SIGMAS[0], 3-step refine at full res ---
        let renoise = STAGE2_SIGMAS[0];
        let noise_v = sampler::gaussian_noise(lat_v_full.len(), sampler::substream_seed(SEED, 2));
        let lat_v2 = sampler::renoise(&lat_v_full, &noise_v, renoise);
        let noise_a = sampler::gaussian_noise(lat_a.len(), sampler::substream_seed(SEED, 3));
        let lat_a2 = sampler::renoise(&lat_a, &noise_a, renoise);
        let s2 = streams_for(vd_full, ad);
        let freqs_s2 = build_dit_freqs(vd_full, ad, FPS);
        denoise_loop(
            &backend,
            &dit_pipes,
            &dit_res,
            &dit_ws,
            &dit_model,
            s2,
            &STAGE2_SIGMAS,
            lat_v2,
            lat_a2,
            &vtext,
            &atext,
            &freqs_s2,
            None, // no i2v conditioning
            None,
        )
        .await
        .expect("stage2 denoise")
    } else {
        // --- Single-stage: 8-step denoise directly at the target res ---
        let s = streams_for(vd_full, ad);
        let freqs = build_dit_freqs(vd_full, ad, FPS);
        let lat_v = sampler::gaussian_noise(vd_full.elems(), sampler::substream_seed(SEED, 0));
        let lat_a = sampler::gaussian_noise(ad.elems(), sampler::substream_seed(SEED, 1));
        denoise_loop(
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
            None, // no i2v conditioning
            None,
        )
        .await
        .expect("denoise")
    };
    assert!(
        lat_v_final.iter().all(|v| v.is_finite()),
        "video latents non-finite"
    );

    // Free the DiT before the VAE phase (release its streamed GPU residents).
    drop(dit_model);
    dit_res.evict_all_and_free(&*backend);
    drop(dit_res);

    // === Phase G: video VAE decode -> frames ===
    let cthw_final = sampler::video_tokens_to_cthw(&lat_v_final, vd_full);
    let video = {
        let src = open_st(&vae_path).await;
        let residency = WeightResidency::new(src, budget());
        let pipes = LtxVaePipelines::compile(&backend)
            .await
            .expect("compile vae");
        let decoder =
            LtxVaeDecoder::new(pipes, &residency, mean.clone(), std.clone()).expect("vae decoder");
        let ws = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));
        decoder
            .decode(
                &backend,
                &residency,
                &ws,
                &cthw_final,
                vd_full.frames,
                vd_full.height,
                vd_full.width,
            )
            .await
            .expect("vae decode")
    };

    // --- Health assertions: shape, finiteness, non-degeneracy ---
    let f_px = 8 * (vd_full.frames - 1) + 1;
    let h_px = 32 * vd_full.height;
    let w_px = 32 * vd_full.width;
    assert_eq!(video.len(), 3 * f_px * h_px * w_px, "video shape");
    assert!(video.iter().all(|v| v.is_finite()), "video has non-finite");
    let n = video.len() as f64;
    let m = video.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = video.iter().map(|&v| (v as f64 - m).powi(2)).sum::<f64>() / n;
    eprintln!(
        "ltx e2e: decoded video [3,{f_px},{h_px},{w_px}] mean={m:.4} var={var:.6} min={:.3} max={:.3}",
        video.iter().cloned().fold(f32::INFINITY, f32::min),
        video.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );
    assert!(var > 1e-6, "degenerate (flat) video output: var={var}");

    // === Phase H: audio decode tail (audio VAE -> vocoder), joint-AV health ===
    // The DiT audio latent is token-major [Ta, IN_CHANNELS]; the 128 features are
    // (channel, mel) = c*16 + m. Reshape to the audio VAE's [8, Ta, 16] CTF layout
    // (normalized; the audio VAE un-normalizes with its own stats). Audio VAE +
    // vocoder both load from the one audio_vae safetensors file.
    let ta = ad.frames;
    const AC: usize = 8;
    const AM: usize = 16;
    let mut audio_ctf = vec![0f32; AC * ta * AM];
    for c in 0..AC {
        for t in 0..ta {
            for m in 0..AM {
                audio_ctf[(c * ta + t) * AM + m] = lat_a_final[t * (AC * AM) + c * AM + m];
            }
        }
    }
    let wav = {
        let src = open_st(&audio_vae_path).await;
        let residency = WeightResidency::new(src, budget());
        let (a_mean, a_std) = load_audio_latent_stats(&residency, &backend)
            .await
            .expect("load audio latent stats");
        let avae_pipes = AudioVaePipelines::compile(&backend)
            .await
            .expect("compile audio vae");
        let decoder = AudioVaeDecoder::new(avae_pipes, &residency, a_mean, a_std)
            .expect("build audio vae decoder");
        let ws = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));
        let mel = decoder
            .decode(&backend, &residency, &ws, &audio_ctf, ta)
            .await
            .expect("audio vae decode");
        let mel_frames = 4 * ta - 3;
        let voc_pipes = VocoderPipelines::compile(&backend)
            .await
            .expect("compile vocoder");
        let vocoder = Vocoder::new(voc_pipes, &residency, &backend)
            .await
            .expect("build vocoder");
        let wav = vocoder
            .decode(&backend, &residency, &ws, &mel, mel_frames)
            .await
            .expect("vocoder decode");
        residency.evict_all_and_free(&*backend);
        wav
    };
    assert!(wav.iter().all(|v| v.is_finite()), "audio has non-finite");
    let an = wav.len() as f64;
    let am = wav.iter().map(|&v| v as f64).sum::<f64>() / an;
    let avar = wav.iter().map(|&v| (v as f64 - am).powi(2)).sum::<f64>() / an;
    eprintln!(
        "ltx e2e: decoded audio [2,{}] mean={am:.4} var={avar:.6} min={:.3} max={:.3}",
        wav.len() / 2,
        wav.iter().cloned().fold(f32::INFINITY, f32::min),
        wav.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );
    assert!(avar > 1e-9, "degenerate (silent) audio output: var={avar}");
}

async fn open_st(path: &Path) -> SafetensorsSource<MmapFileOpener> {
    let opener = MmapFileOpener::new(path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    SafetensorsSource::open(opener)
        .await
        .unwrap_or_else(|e| panic!("parse {}: {e:?}", path.display()))
}

fn run_tokens_ref(gemma: &Path, out_dir: &Path) {
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let status = Command::new("uv")
        .args([
            "run",
            "--directory",
            py_dir.to_str().unwrap(),
            "--with",
            "gguf",
            "--with",
            "transformers",
            "python",
            "-m",
            "thinfer_pytorch_ref.ltx.gen_tokens_ref",
            "--gguf",
            gemma.to_str().unwrap(),
            "--prompt",
            PROMPT,
            "--out",
            out_dir.to_str().unwrap(),
        ])
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "ltx tokens pyref failed");
}
