//! LTX-2.3 distilled joint audio-video driver (the product-path analog of the
//! conformance `t2v_e2e_health`). Models can't open files, so the file IO + the
//! per-phase `WeightResidency` lifecycle live here; `thinfer_models::ltx::*`
//! stays pure glue. The chain: tokenize -> Gemma-3 encoder -> FeatureExtractor
//! V2 -> 8-layer connector -> denoise -> video VAE decode (+ audio VAE ->
//! vocoder) -> H.264 + AAC MP4. The distilled denoise has two modes: the default
//! single-stage (8 steps at the target res) and the opt-in `upscale` two-stage
//! refine (stage 1 at half res -> latent upscale x2 -> stage 2 renoise + 3-step
//! refine at full res), which is only the cheaper route to HIGH res.
//!
//! Each weight file gets its own scoped residency (built -> used -> dropped) so
//! only one large weight set is VRAM-resident at a time (Gemma, then the DiT
//! GGUF across conditioning + the denoise stage(s), then the VAEs), matching the
//! upstream `DiffusionStage` lifecycle and the parity tests. On the upscale path
//! the DiT is evicted before the upscaler (it would otherwise OOM the 8GB card);
//! the DiT is evicted again before the VAE phase on both paths.

use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::format::union::RenamedSource;
use thinfer_core::ops::WeightDtype;
use thinfer_core::quant::QuantKind;
use thinfer_core::residency::WeightResidency;
use thinfer_core::tokenizer::Tokenizer;
use thinfer_core::weight::{WeightCatalog, WeightId, WeightReader, WeightSource};
use thinfer_core::workspace::Workspace;
use thinfer_models::ltx::LtxVariant;
use thinfer_models::ltx::audio_vae::{
    AudioVaeDecoder, AudioVaePipelines, load_latent_stats as load_audio_latent_stats,
};
use thinfer_models::ltx::config;
use thinfer_models::ltx::connector::{
    AUDIO, CONN_SEQ, ConnectorPipelines, RAPID_AUDIO, RAPID_VIDEO, VIDEO, connector_forward,
    fe_aggregate, fe_aggregate_v1, feature_extractor_v1_flatten, feature_extractor_v2_flatten,
    register_connector, register_fe, register_fe_v1,
};
use thinfer_models::ltx::dit::{DitModel, DitPipelines, caption_project, register_caption_proj};
use thinfer_models::ltx::loader::{UnitOffsetSource, gemma_gguf_renames, gemma_norm_offset_ids};
use thinfer_models::ltx::lora;
use thinfer_models::ltx::manifest::role;
use thinfer_models::ltx::pipeline::I2vCond;
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
    LATENT_CHANNELS, LtxVaeConfig, LtxVaeDecoder, LtxVaeEncoder, LtxVaeEncoderConfig,
    LtxVaePipelines, load_latent_stats,
};
use thinfer_models::ltx::vocoder::{Vocoder, VocoderPipelines};
use thinfer_models::wan::pipeline::WanVideo;
use thinfer_native::MmapFileOpener;
use thinfer_native::tokenizer::HfTokenizer;

use crate::config::{ResidencyBudget, random_seed};
use crate::download::resolve_role;
use crate::progress::{ProgressSink, Stage};
use crate::request::{JobSummary, VideoFormat, VideoRequest};

/// Audio sample rate the vocoder emits (48kHz stereo).
const AUDIO_SR: u32 = 48_000;

/// VRAM weight-residency cap for LTX on thin (8GB) hardware. The DiT is 22.8GB
/// and always streams per-block regardless of budget, so a high budget does not
/// reduce streaming -- it just pins more resident weight, stealing the device
/// slack the stage-2 full-res denoise needs for its activation peak. At the
/// in-distribution default (1280x704 two-stage), a 5G budget (serve's default)
/// OOMs stage 2; ~2G leaves enough slack to fit while the per-step time is
/// unchanged (streaming dominates either way). Callers requesting LESS keep it;
/// the cap only lowers an over-large budget. Revisit for >8GB cards.
const LTX_VRAM_BUDGET_CAP: u64 = 2 * 1024 * 1024 * 1024;

/// Run an LTX-2.3 joint audio-video generate to completion. `req.model` must be
/// the LTX id (the caller dispatches on [`crate::model::VideoModelId::is_ltx`]).
pub async fn run(
    backend: &Arc<WgpuBackend>,
    req: &VideoRequest,
    sink: &dyn ProgressSink,
) -> Result<JobSummary, String> {
    let plan = req.resolve()?;
    for w in &plan.warnings {
        tracing::warn!(target: thinfer_core::trace::DIAG, "{w}");
        sink.note(w);
    }
    let manifest = req.model.manifest();
    // ltx2-rapid (LTX-2 19B merge): 19B conditioning tail + LTX-2 VAEs, single-stage
    // only (no spatial upscaler ships), video-only for now (its audio VAE is
    // comfy-prefixed -- a separate follow-up).
    let rapid = req.model.is_ltx_rapid();
    let frames = plan.frames as usize;
    let height = req.height as usize;
    let width = req.width as usize;
    let fps = plan.fps as f64;
    // Cap the weight-residency budget so the stage-2 full-res denoise keeps the
    // device slack it needs at the widescreen default (see LTX_VRAM_BUDGET_CAP).
    let budget = ResidencyBudget {
        vram_bytes: req.budget.vram_bytes.min(LTX_VRAM_BUDGET_CAP),
        ..req.budget
    };

    // --- resolve every weight file (caller ensured they are cached) ---
    let enc_role = req.ltx_encoder_role();
    let gemma_path = resolve_role(manifest, enc_role)?;
    let tok_path = resolve_role(manifest, role::TOKENIZER)?;
    let conn_path = resolve_role(manifest, role::CONNECTOR)?;
    let dit_path = resolve_role(manifest, req.model.ltx_dit_role())?;
    let vae_path = resolve_role(manifest, role::VIDEO_VAE)?;
    let audio_vae_path = resolve_role(manifest, role::AUDIO_VAE)?;

    // --- Phase 0: optional prompt rewrite (ltx2-rapid only). Like HunyuanVideo
    //     1.5, the 19B merge is trained on long, structured captions -- a terse
    //     raw prompt is out-of-distribution and collapses to the model's portrait
    //     prior. Expand it via the on-device Qwen3-VL rewriter (phase-scoped,
    //     evicted before the encoder); fall back to the original on any failure.
    //     The 22B distilled/Sulphur path is unchanged. Prompt text is never logged.
    let raw_prompt = req.prompts[0].trim();
    if raw_prompt.is_empty() {
        return Err("empty prompt produced no tokens".into());
    }
    let rewritten = if rapid {
        crate::rewrite::maybe_rewrite_prompt(
            req.rewrite,
            req.rewrite_quality,
            crate::rewrite::LTX_REWRITE_SYSTEM_PROMPT,
            raw_prompt,
            backend,
            manifest,
            req.budget.vram_bytes,
            sink,
        )
        .await
    } else {
        None
    };
    let prompt: &str = rewritten.as_deref().unwrap_or(raw_prompt);

    // --- tokenize (Gemma fast tokenizer; matches the pyref add_special_tokens) ---
    let tokenizer = HfTokenizer::from_path(&tok_path)
        .await
        .map_err(|e| format!("load tokenizer {}: {e:?}", tok_path.display()))?;
    let ids = tokenizer
        .encode(prompt, true)
        .map_err(|e| format!("tokenize: {e:?}"))?;
    if ids.is_empty() {
        return Err("empty prompt produced no tokens".into());
    }
    let seq = ids.len();

    let seed = req.seed.unwrap_or_else(random_seed);
    tracing::info!(
        target: thinfer_core::trace::DIAG,
        model = %req.model, width = req.width, height = req.height,
        frames = plan.frames, fps = plan.fps, seed, tokens = seq,
        ram_budget = budget.ram_bytes, vram_budget = budget.vram_bytes,
        encoder = enc_role,
        "ltx generate start",
    );
    sink.note(&format!(
        "Generating {}x{} joint AV, {} frames @ {} fps (~{:.1}s), seed {} ({})",
        req.width,
        req.height,
        plan.frames,
        plan.fps,
        plan.frames as f32 / plan.fps as f32,
        seed,
        req.model,
    ));

    // Latent grids: full res (target) + audio (duration-derived). The opt-in
    // upscale path derives its half-res stage-1 grid inside the branch below.
    let vd_full = VideoLatentDims::from_pixels(frames, height, width);
    let ad = AudioLatentDims::from_video(frames, fps);

    sink.stage(Stage::TextEncode);

    // === Phase: video VAE per-channel latent stats (scoped; reopened to decode) ===
    let (mean, std) = {
        let src = open_st(&vae_path).await?;
        let residency = WeightResidency::new(src, budget);
        let stats = load_latent_stats(&residency, backend)
            .await
            .map_err(|e| format!("load video latent stats: {e:?}"))?;
        residency.evict_all_and_free(&**backend);
        stats
    };

    // === Phase A: Gemma-3 text encoder -> 49 hidden states (host) ===
    let states = {
        let opener = open_mmap(&gemma_path).await?;
        let gguf = GgufSource::open(opener)
            .await
            .map_err(|e| format!("parse gemma gguf: {e:?}"))?;
        let renamed = RenamedSource::with_passthrough(gguf, gemma_gguf_renames());
        let source = UnitOffsetSource::new(renamed, gemma_norm_offset_ids());
        let residency = WeightResidency::new(source, budget);
        let handles =
            register_handles(&residency, None).map_err(|e| format!("register encoder: {e:?}"))?;
        let cfgs = gemma_encoder_cfgs(WeightDtype::Quant(QuantKind::Q8_0));
        let pipelines = GemmaEncoderPipelines::compile(backend, &cfgs)
            .await
            .map_err(|e| format!("compile gemma pipelines: {e:?}"))?;
        let workspace = Workspace::new(Arc::clone(backend), Arc::clone(residency.arbiter()));
        let out = GemmaEncoder
            .forward(
                backend,
                &pipelines,
                &residency,
                &workspace,
                &handles,
                residency.source(),
                &ids,
            )
            .await
            .map_err(|e| format!("gemma encoder forward: {e:?}"))?;
        residency.evict_all_and_free(&**backend);
        out.states
    };

    // Which LTX-2 line: 22B (LTX-2.3 / Sulphur) or the 19B (ltx2-rapid) merge.
    let variant = if rapid {
        LtxVariant::ltx2_rapid_19b()
    } else {
        LtxVariant::ltx_2_3_22b()
    };

    // FeatureExtractor: V1 (19B range-norm, single stream) or V2 (22B per-token RMS).
    let flat = if rapid {
        feature_extractor_v1_flatten(&states, seq)
    } else {
        feature_extractor_v2_flatten(&states, seq)
    };
    drop(states);

    // Connector pipelines: bf16 block matmuls (19B, from safetensors) vs Q8_0 (22B,
    // from the DiT GGUF).
    let conn_pipes = if rapid {
        ConnectorPipelines::compile_rapid(backend).await
    } else {
        ConnectorPipelines::compile(backend).await
    }
    .map_err(|e| format!("compile connector pipelines: {e:?}"))?;

    // Conditioning tail -> cross-attn caption KV (`vtext`/`atext`) + the resident
    // DiT (`dit_res`/`dit_model`/`dit_pipes`/`dit_ws`). The 19B and 22B lines differ
    // in where the connector lives and how the 3840 Gemma channels reach the
    // cross-attn width, so each builds the KV its own way and converges here.
    let (vtext, atext, dit_res, dit_pipes, dit_model, dit_ws) = if rapid {
        // === Phase B (19B): single bias-free FE V1 aggregate + 2-layer ungated
        // 3840 connector, BOTH from the connector safetensors -> [1024, 3840]. ===
        let (video_conn, audio_conn) = {
            let src = open_st(&conn_path).await?;
            let residency = WeightResidency::new(src, budget);
            let fe_w = register_fe_v1(&residency).map_err(|e| format!("register FE V1: {e:?}"))?;
            let video_h = register_connector(&residency, RAPID_VIDEO)
                .map_err(|e| format!("register video conn: {e:?}"))?;
            let audio_h = register_connector(&residency, RAPID_AUDIO)
                .map_err(|e| format!("register audio conn: {e:?}"))?;
            let ws = Workspace::new(Arc::clone(backend), Arc::clone(residency.arbiter()));
            // Single shared aggregate (out = Gemma HIDDEN = 3840) fed to both.
            let agg = fe_aggregate_v1(
                backend,
                &conn_pipes,
                &residency,
                &ws,
                &flat,
                seq,
                fe_w,
                config::CAPTION_CHANNELS,
            )
            .await
            .map_err(|e| format!("fe v1 aggregate: {e:?}"))?;
            let v = connector_forward(
                backend,
                &conn_pipes,
                &residency,
                &ws,
                &video_h,
                RAPID_VIDEO,
                &agg,
                seq,
            )
            .await
            .map_err(|e| format!("video connector: {e:?}"))?;
            let a = connector_forward(
                backend,
                &conn_pipes,
                &residency,
                &ws,
                &audio_h,
                RAPID_AUDIO,
                &agg,
                seq,
            )
            .await
            .map_err(|e| format!("audio connector: {e:?}"))?;
            residency.evict_all_and_free(&**backend);
            (v, a)
        };
        // === Phase C (19B): DiT GGUF resident: caption projection (3840 -> 4096/
        // 2048) + register the 19B variant. No Sulphur fold (rapid is pre-distilled). ===
        let opener = open_mmap(&dit_path).await?;
        let gguf = GgufSource::open(opener)
            .await
            .map_err(|e| format!("parse dit gguf: {e:?}"))?;
        let dit_res = WeightResidency::new(DitSource::Plain(gguf), budget);
        let dit_pipes = DitPipelines::compile(backend)
            .await
            .map_err(|e| format!("compile dit: {e:?}"))?;
        let cap_h =
            register_caption_proj(&dit_res).map_err(|e| format!("register caption proj: {e:?}"))?;
        let dit_model = DitModel::register_variant(backend, &dit_res, config::NUM_LAYERS, variant)
            .await
            .map_err(|e| format!("register dit model: {e:?}"))?;
        let dit_ws = Workspace::new(Arc::clone(backend), Arc::clone(dit_res.arbiter()));
        let (vtext, atext) = caption_project(
            backend,
            &dit_pipes,
            &dit_res,
            &dit_ws,
            &cap_h,
            &video_conn,
            &audio_conn,
            CONN_SEQ,
        )
        .await
        .map_err(|e| format!("caption projection: {e:?}"))?;
        (vtext, atext, dit_res, dit_pipes, dit_model, dit_ws)
    } else {
        // === Phase B (22B): FeatureExtractor V2 -> per-modality aggregate embeds ===
        let (video_embed, audio_embed) = {
            let src = open_st(&conn_path).await?;
            let residency = WeightResidency::new(src, budget);
            let fe = register_fe(&residency).map_err(|e| format!("register FE: {e:?}"))?;
            let workspace = Workspace::new(Arc::clone(backend), Arc::clone(residency.arbiter()));
            let v = fe_aggregate(
                backend,
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
            .map_err(|e| format!("fe video: {e:?}"))?;
            let a = fe_aggregate(
                backend,
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
            .map_err(|e| format!("fe audio: {e:?}"))?;
            residency.evict_all_and_free(&**backend);
            (v, a)
        };

        // === Phases C-F: DiT GGUF resident for connector blocks + denoise ===
        let opener = open_mmap(&dit_path).await?;
        let gguf = GgufSource::open(opener)
            .await
            .map_err(|e| format!("parse dit gguf: {e:?}"))?;
        // Sulphur ships the BASE (`dev`) DiT; fold the distill LoRA in so the 8-step
        // CFG-free distilled sampler converges (see `ltx::lora`). LTX distilled DiTs
        // are already distilled -> plain passthrough.
        let dit_src = if req.model.is_sulphur() {
            let mut stack = Vec::new();
            for (role_name, strength) in req.model.sulphur_distill_stack() {
                let lora_path = resolve_role(manifest, role_name)?;
                let lora = open_st(&lora_path).await?;
                let specs = lora::discover_specs(&gguf, &lora)
                    .await
                    .map_err(|e| format!("discover distill lora {role_name} specs: {e:?}"))?;
                tracing::info!(
                    target: thinfer_core::trace::DIAG,
                    lora = role_name, strength, sites = specs.len(),
                    "sulphur distill lora in stack",
                );
                stack.push((lora, strength, specs));
            }
            let folded = lora::LoraFoldSource::new(gguf, stack)
                .map_err(|e| format!("build distill lora fold: {e}"))?;
            tracing::info!(
                target: thinfer_core::trace::DIAG,
                sites = folded.fold_count(), "sulphur distill lora folded",
            );
            DitSource::Folded(Box::new(folded))
        } else {
            DitSource::Plain(gguf)
        };
        let dit_res = WeightResidency::new(dit_src, budget);
        let video_h = register_connector(&dit_res, VIDEO)
            .map_err(|e| format!("register video conn: {e:?}"))?;
        let audio_h = register_connector(&dit_res, AUDIO)
            .map_err(|e| format!("register audio conn: {e:?}"))?;
        let dit_pipes = DitPipelines::compile(backend)
            .await
            .map_err(|e| format!("compile dit: {e:?}"))?;
        let dit_model = DitModel::register(backend, &dit_res, config::NUM_LAYERS)
            .await
            .map_err(|e| format!("register dit model: {e:?}"))?;
        let dit_ws = Workspace::new(Arc::clone(backend), Arc::clone(dit_res.arbiter()));

        // Connector cross-attn KV (all 1024 positions valid; registers fill pads).
        let vtext = connector_forward(
            backend,
            &conn_pipes,
            &dit_res,
            &dit_ws,
            &video_h,
            VIDEO,
            &video_embed,
            seq,
        )
        .await
        .map_err(|e| format!("video connector: {e:?}"))?;
        let atext = connector_forward(
            backend,
            &conn_pipes,
            &dit_res,
            &dit_ws,
            &audio_h,
            AUDIO,
            &audio_embed,
            seq,
        )
        .await
        .map_err(|e| format!("audio connector: {e:?}"))?;
        (vtext, atext, dit_res, dit_pipes, dit_model, dit_ws)
    };

    // === Optional Phase F2: native I2V frame-0 latent (ltx2-rapid only) ===
    // Encode the input image to a clean, normalized frame-0 latent; the denoise
    // then holds the leading latent-frame to it (the frame-0 anchor that rescues
    // the merge's weak compositional T2V). Video-only, single-stage. `clean_v` is
    // token-major with the encoded frame-0 in the leading `H'*W'` tokens; `mask_v`
    // is 1.0 (free) everywhere except `1-strength` on those frame-0 tokens.
    let i2v = if let Some(img) = req.input_image.as_ref().filter(|_| rapid) {
        let rgb = image::load_from_memory(&img.0)
            .map_err(|e| format!("decode input image: {e}"))?
            .to_rgb8();
        sink.note(&format!(
            "I2V frame-0 {}x{} (strength {:.2})",
            rgb.width(),
            rgb.height(),
            req.strength
        ));
        let resized = image::imageops::resize(
            &rgb,
            width as u32,
            height as u32,
            image::imageops::FilterType::CatmullRom,
        );
        // CHW [-1,1], single frame [3, H, W].
        let mut frame_px = vec![0.0f32; 3 * width * height];
        for (x, y, px) in resized.enumerate_pixels() {
            for c in 0..3 {
                frame_px[c * width * height + y as usize * width + x as usize] =
                    px.0[c] as f32 / 127.5 - 1.0;
            }
        }
        // Encode -> normalized latent CTHW [128, 1, H/32, W/32].
        let z = {
            let src = open_st(&vae_path).await?;
            let residency = WeightResidency::new(src, budget);
            let pipes = LtxVaePipelines::compile(backend)
                .await
                .map_err(|e| format!("compile vae encoder: {e:?}"))?;
            let enc = LtxVaeEncoder::new(
                pipes,
                &residency,
                mean.clone(),
                std.clone(),
                LtxVaeEncoderConfig::ltx2_rapid(),
            )
            .map_err(|e| format!("build vae encoder: {e:?}"))?;
            let ws = Workspace::new(Arc::clone(backend), Arc::clone(residency.arbiter()));
            let out = enc
                .encode(backend, &residency, &ws, &frame_px, 1, height, width)
                .await
                .map_err(|e| format!("vae encode: {e:?}"))?;
            residency.evict_all_and_free(&**backend);
            out
        };
        // z CTHW [128,1,H',W'] -> token-major clean_v [tokens,128] with the
        // leading latent-frame-0 tokens filled (token t = f*H'*W' + h*W' + w).
        let ftok = vd_full.height * vd_full.width;
        let tokens = vd_full.tokens();
        let c = LATENT_CHANNELS;
        if z.len() != c * ftok {
            return Err(format!("i2v latent size {} != {}", z.len(), c * ftok));
        }
        let mut clean = vec![0.0f32; c * tokens];
        for t in 0..ftok {
            for ch in 0..c {
                clean[t * c + ch] = z[ch * ftok + t];
            }
        }
        let mut mask = vec![1.0f32; tokens];
        for m in &mut mask[..ftok] {
            *m = 1.0 - req.strength;
        }
        Some((clean, mask))
    } else {
        None
    };

    // Two render paths (upstream `LTX2_DISTILLED` vs `LTX2_TWO_STAGE`): the
    // default single-stage 8-step denoise at the target res, or the opt-in 2x
    // spatial-upscale refine. Two-stage halves stage-1's res, so at low res it
    // goes out of distribution -- hence single-stage is the default; two-stage is
    // for cheaply reaching HIGH res.
    let (lat_v_final, lat_a_final) = if req.upscale && !rapid {
        let upscaler_path = resolve_role(manifest, role::UPSCALER)?;
        let vd_s1 = VideoLatentDims::from_pixels(frames, height / 2, width / 2);
        let s1_steps = STAGE1_SIGMAS.len() - 1;
        let s2_steps = STAGE2_SIGMAS.len() - 1;
        let total_steps = (s1_steps + s2_steps) as u32;

        // --- Stage 1: pure-noise init, denoise at half res ---
        let s1 = streams_for(vd_s1, ad);
        let freqs_s1 = build_dit_freqs(vd_s1, ad, fps);
        let lat_v = sampler::gaussian_noise(vd_s1.elems(), sampler::substream_seed(seed, 0));
        let lat_a = sampler::gaussian_noise(ad.elems(), sampler::substream_seed(seed, 1));
        let prog_s1 = |step: usize| {
            sink.stage(Stage::Step {
                i: step as u32 + 1,
                n: total_steps,
            })
        };
        let (lat_v, lat_a) = denoise_loop(
            backend,
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
            None, // upscale path is t2v only (rapid, the i2v model, is single-stage)
            Some(&prog_s1),
        )
        .await
        .map_err(|e| format!("stage1 denoise: {e:?}"))?;

        // --- Upscale x2 (un-normalize -> upsample -> re-normalize) ---
        let mut cthw = sampler::video_tokens_to_cthw(&lat_v, vd_s1);
        let thw_s1 = vd_s1.tokens();
        un_normalize_cthw(&mut cthw, &mean, &std, thw_s1);
        // Free the DiT's GPU-cached block weights so the upsampler has the full
        // card; stage 2 re-streams the DiT from disk anyway (budget << 22.8GB).
        dit_res.evict_all_and_free(&**backend);
        let mut upscaled = {
            let src = open_st(&upscaler_path).await?;
            let residency = WeightResidency::new(src, budget);
            let pipes = LtxUpsamplerPipelines::compile(backend)
                .await
                .map_err(|e| format!("compile upsampler: {e:?}"))?;
            let ups =
                LtxUpsampler::new(pipes, &residency).map_err(|e| format!("build ups: {e:?}"))?;
            let ws = Workspace::new(Arc::clone(backend), Arc::clone(residency.arbiter()));
            let up = ups
                .forward(
                    backend,
                    &residency,
                    &ws,
                    &cthw,
                    vd_s1.frames,
                    vd_s1.height,
                    vd_s1.width,
                )
                .await
                .map_err(|e| format!("upsample: {e:?}"))?;
            residency.evict_all_and_free(&**backend);
            up
        };
        let thw_full = vd_full.tokens();
        if upscaled.len() != LATENT_CHANNELS * thw_full {
            return Err(format!(
                "upscaled size {} != expected {}",
                upscaled.len(),
                LATENT_CHANNELS * thw_full
            ));
        }
        normalize_cthw(&mut upscaled, &mean, &std, thw_full);
        let lat_v_full = sampler::video_cthw_to_tokens(&upscaled, vd_full);

        // --- Stage 2: renoise to STAGE2_SIGMAS[0], refine at full res ---
        let renoise = STAGE2_SIGMAS[0];
        let noise_v = sampler::gaussian_noise(lat_v_full.len(), sampler::substream_seed(seed, 2));
        let lat_v2 = sampler::renoise(&lat_v_full, &noise_v, renoise);
        let noise_a = sampler::gaussian_noise(lat_a.len(), sampler::substream_seed(seed, 3));
        let lat_a2 = sampler::renoise(&lat_a, &noise_a, renoise);
        let s2 = streams_for(vd_full, ad);
        let freqs_s2 = build_dit_freqs(vd_full, ad, fps);
        let prog_s2 = |step: usize| {
            sink.stage(Stage::Step {
                i: s1_steps as u32 + step as u32 + 1,
                n: total_steps,
            })
        };
        denoise_loop(
            backend,
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
            None, // upscale path is t2v only
            Some(&prog_s2),
        )
        .await
        .map_err(|e| format!("stage2 denoise: {e:?}"))?
    } else {
        // --- Single-stage distilled: 8-step denoise directly at the target res ---
        let total_steps = (STAGE1_SIGMAS.len() - 1) as u32;
        let s = streams_for(vd_full, ad);
        let freqs = build_dit_freqs(vd_full, ad, fps);
        let mut lat_v = sampler::gaussian_noise(vd_full.elems(), sampler::substream_seed(seed, 0));
        let lat_a = sampler::gaussian_noise(ad.elems(), sampler::substream_seed(seed, 1));
        // I2V: seed the conditioned tokens with the clean frame-0 latent (the
        // same blend the per-step X0 hard-replace uses), then hold them there.
        let cond = i2v.as_ref().map(|(clean, mask)| {
            sampler::blend_clean(&mut lat_v, clean, mask, LATENT_CHANNELS);
            I2vCond {
                clean_v: clean.as_slice(),
                mask_v: mask.as_slice(),
            }
        });
        let prog = |step: usize| {
            sink.stage(Stage::Step {
                i: step as u32 + 1,
                n: total_steps,
            })
        };
        denoise_loop(
            backend,
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
            cond,
            Some(&prog),
        )
        .await
        .map_err(|e| format!("denoise: {e:?}"))?
    };

    // Free the DiT before the VAE phases (release its streamed GPU residents).
    drop(dit_model);
    dit_res.evict_all_and_free(&**backend);
    drop(dit_res);

    sink.stage(Stage::VaeDecode);

    // === Phase G: video VAE decode -> frames ===
    // The DiT is fully evicted above, so the whole card is free for the decode.
    // Use the UNCAPPED requested budget (not the 2G DiT-streaming cap): the VAE
    // tiler sizes tiles from this budget, so the 2G cap forced tiny tiles (seed
    // OOMed down to 2 latent cells -> hundreds of overlapping tiles, ~320s at
    // 768x512x49). The cap exists only to leave device slack while the 22.8G DiT
    // streams per-block; it has no reason to throttle the standalone VAE phase.
    let vae_budget = ResidencyBudget {
        vram_bytes: req.budget.vram_bytes,
        ..budget
    };
    let cthw_final = sampler::video_tokens_to_cthw(&lat_v_final, vd_full);
    let mut video = {
        let src = open_st(&vae_path).await?;
        let residency = WeightResidency::new(src, vae_budget);
        let pipes = LtxVaePipelines::compile(backend)
            .await
            .map_err(|e| format!("compile vae: {e:?}"))?;
        // The 19B (ltx2-rapid) uses the LTX-2 (non-.3) decoder schedule.
        let cfg = if rapid {
            LtxVaeConfig::ltx2_rapid()
        } else {
            LtxVaeConfig::distilled()
        };
        let decoder =
            LtxVaeDecoder::new_with_config(pipes, &residency, mean.clone(), std.clone(), cfg)
                .map_err(|e| format!("vae decoder: {e:?}"))?;
        let ws = Workspace::new(Arc::clone(backend), Arc::clone(residency.arbiter()));
        let out = decoder
            .decode(
                backend,
                &residency,
                &ws,
                &cthw_final,
                vd_full.frames,
                vd_full.height,
                vd_full.width,
            )
            .await
            .map_err(|e| format!("vae decode: {e:?}"))?;
        residency.evict_all_and_free(&**backend);
        out
    };
    if !video.iter().all(|v| v.is_finite()) {
        return Err("video decode produced non-finite output".into());
    }
    // WanVideo is the shared CTHW [-1,1] frame carrier; honor its clamp contract.
    for v in &mut video {
        *v = v.clamp(-1.0, 1.0);
    }
    let f_px = 8 * (vd_full.frames - 1) + 1;
    let h_px = 32 * vd_full.height;
    let w_px = 32 * vd_full.width;
    if video.len() != 3 * f_px * h_px * w_px {
        return Err(format!(
            "video shape {} != 3*{f_px}*{h_px}*{w_px}",
            video.len()
        ));
    }
    let wan_video = WanVideo {
        frames: video,
        num_frames: f_px,
        height: h_px,
        width: w_px,
    };

    // === Phase H: audio decode tail (audio VAE -> vocoder) ===
    // Skipped entirely when audio is off (req.audio=false) -- the audio VAE +
    // ~108-conv vocoder is a sizable chunk of the run, so a video-only request
    // saves it. The DiT audio latent is token-major [Ta, IN_CHANNELS]; the 128
    // features are (channel, mel) = c*16 + m. Reshape to the audio VAE's
    // [8, Ta, 16] CTF layout. Audio VAE + vocoder share one residency.
    // rapid: video-only (its audio VAE is comfy-prefixed; a follow-up).
    let wav = if req.audio && !rapid {
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
        let src = open_st(&audio_vae_path).await?;
        let residency = WeightResidency::new(src, budget);
        let (a_mean, a_std) = load_audio_latent_stats(&residency, backend)
            .await
            .map_err(|e| format!("load audio latent stats: {e:?}"))?;
        let avae_pipes = AudioVaePipelines::compile(backend)
            .await
            .map_err(|e| format!("compile audio vae: {e:?}"))?;
        let decoder = AudioVaeDecoder::new(avae_pipes, &residency, a_mean, a_std)
            .map_err(|e| format!("build audio vae: {e:?}"))?;
        let ws = Workspace::new(Arc::clone(backend), Arc::clone(residency.arbiter()));
        let mel = decoder
            .decode(backend, &residency, &ws, &audio_ctf, ta)
            .await
            .map_err(|e| format!("audio vae decode: {e:?}"))?;
        let mel_frames = 4 * ta - 3;
        let voc_pipes = VocoderPipelines::compile(backend)
            .await
            .map_err(|e| format!("compile vocoder: {e:?}"))?;
        let vocoder = Vocoder::new(voc_pipes, &residency, backend)
            .await
            .map_err(|e| format!("build vocoder: {e:?}"))?;
        let wav = vocoder
            .decode(backend, &residency, &ws, &mel, mel_frames)
            .await
            .map_err(|e| format!("vocoder decode: {e:?}"))?;
        residency.evict_all_and_free(&**backend);
        if !wav.iter().all(|v| v.is_finite()) {
            return Err("audio decode produced non-finite output".into());
        }
        Some(wav)
    } else {
        None
    };

    // --- Encode: H.264 video (+ optional AAC audio) -> MP4 (or silent PNG frames) ---
    match req.format {
        VideoFormat::Mp4 => {
            let audio = match &wav {
                Some(wav) => {
                    sink.note("Encoding MP4 (H.264 + AAC)");
                    Some(crate::codec::encode_aac_stereo(wav, AUDIO_SR)?)
                }
                None => {
                    sink.note("Encoding MP4 (H.264, no audio)");
                    None
                }
            };
            crate::codec::encode_mp4_with_audio(&wan_video, plan.fps, audio, &req.output)?;
        }
        VideoFormat::PngFrames => {
            // Codec-free debug view: silent frames; the audio is dropped.
            sink.note("Writing PNG frames (audio dropped for png-frames output)");
            crate::codec::write_png_frames(&wan_video, &req.output)?;
        }
    }
    tracing::info!(target: thinfer_core::trace::DIAG, path = %req.output.display(), "wrote output");

    Ok(JobSummary {
        output: req.output.clone(),
        width: w_px as u32,
        height: h_px as u32,
        frames: f_px as u32,
        fps: Some(plan.fps),
        seed: Some(seed),
    })
}

async fn open_mmap(path: &std::path::Path) -> Result<MmapFileOpener, String> {
    MmapFileOpener::new(path)
        .await
        .map_err(|e| format!("open {}: {e}", path.display()))
}

async fn open_st(path: &std::path::Path) -> Result<SafetensorsSource<MmapFileOpener>, String> {
    let opener = open_mmap(path).await?;
    SafetensorsSource::open(opener)
        .await
        .map_err(|e| format!("parse {}: {e:?}", path.display()))
}

/// The DiT weight source: the plain GGUF (LTX distilled) or the GGUF with the
/// Sulphur distill LoRA folded in. One concrete type so the DiT-resident phase
/// (connector + denoise) stays monomorphic over `WeightResidency<DitSource>`.
type Gguf = GgufSource<MmapFileOpener>;
type LoraFold = lora::LoraFoldSource<Gguf, SafetensorsSource<MmapFileOpener>>;

enum DitSource {
    Plain(Gguf),
    // Boxed: the folded source (two sub-sources + catalog + fold cache) is much
    // larger than the plain GGUF handle.
    Folded(Box<LoraFold>),
}

impl WeightSource for DitSource {
    type Reader = DitReader;
    // Inner source errors only need Debug (per the trait); format them at the
    // arm so the wrapper stays a single error type with no write-only fields.
    type Error = String;

    fn catalog(&self) -> &WeightCatalog {
        match self {
            DitSource::Plain(s) => s.catalog(),
            DitSource::Folded(s) => s.catalog(),
        }
    }

    async fn open(&self, id: &WeightId) -> Result<Self::Reader, Self::Error> {
        match self {
            DitSource::Plain(s) => s
                .open(id)
                .await
                .map(DitReader::Plain)
                .map_err(|e| format!("{e:?}")),
            DitSource::Folded(s) => s
                .open(id)
                .await
                .map(DitReader::Folded)
                .map_err(|e| format!("{e:?}")),
        }
    }
}

enum DitReader {
    Plain(<Gguf as WeightSource>::Reader),
    Folded(<LoraFold as WeightSource>::Reader),
}

impl WeightReader for DitReader {
    type Error = String;

    fn len(&self) -> u64 {
        match self {
            DitReader::Plain(r) => r.len(),
            DitReader::Folded(r) => r.len(),
        }
    }

    async fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), Self::Error> {
        match self {
            DitReader::Plain(r) => r.read_at(offset, dst).await.map_err(|e| format!("{e:?}")),
            DitReader::Folded(r) => r.read_at(offset, dst).await.map_err(|e| format!("{e:?}")),
        }
    }

    fn will_read(&mut self, offset: u64, len: u64) {
        match self {
            DitReader::Plain(r) => r.will_read(offset, len),
            DitReader::Folded(r) => r.will_read(offset, len),
        }
    }
}
