//! HunyuanVideo 1.5 T2V driver (the product-path analog of the conformance
//! `t2v_e2e`). Models can't open files, so the file IO + per-phase
//! `WeightResidency` lifecycle live here; `thinfer_models::hunyuan::*` stays pure
//! glue. The chain: build the video chat prompt -> Qwen2.5-VL encoder
//! (hidden_states[-3]) -> seeded latent noise -> 4-step flow-match Euler denoise
//! (lightx2v) -> causal-conv VAE decode -> H.264 MP4 (silent).
//!
//! Three scoped residencies, one large weight set VRAM-resident at a time (the
//! 8GB card holds neither the encoder, the DiT, nor the VAE alongside another):
//! encoder GGUF -> DiT safetensors -> VAE safetensors, each built/used/evicted.
//! The DiT must be `evict_all_and_free`'d before the VAE phase (drop alone does
//! not return pooled VRAM); the DiT itself streams per-block (16.7GB never
//! co-resides).

use std::sync::Arc;

use thinfer_core::arbiter::RECLAIM_EVICTABLE_WEIGHTS;
use thinfer_core::backend::WgpuBackend;
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::format::pytorch::PytorchSource;
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::format::union::RenamedSource;
use thinfer_core::ops::ActDtype;
use thinfer_core::residency::WeightResidency;
use thinfer_core::tokenizer::Tokenizer;
use thinfer_core::workspace::Workspace;
use thinfer_models::common::vae_taehv::{
    TaehvDecoder, TaehvDecoderWeights, TaehvPipelines, register_decoder_taehv,
};
use thinfer_models::hunyuan::dit::{HunyuanDit, HunyuanDitPipelines};
use thinfer_models::hunyuan::encoder::{HunyuanTextEncoder, USER_MARKER, build_chat_prompt};
use thinfer_models::hunyuan::manifest::role;
use thinfer_models::hunyuan::refiner::{HunyuanRefiner, HunyuanRefinerPipelines};
use thinfer_models::hunyuan::scheduler::FlowMatchSchedule;
use thinfer_models::hunyuan::vae::{HunyuanVaeDecoder, HunyuanVaePipelines};
use thinfer_models::hunyuan::vae_tiny::taehv_config;
use thinfer_models::qwen_image::text_encoder::qwen2vl_gguf_renames;
use thinfer_models::wan::pipeline::WanVideo;
use thinfer_native::MmapFileOpener;
use thinfer_native::tokenizer::HfTokenizer;

use crate::config::random_seed;
use crate::download::resolve_role;
use crate::model::VaeChoice;
use crate::progress::{ProgressSink, Stage};
use crate::request::{JobSummary, VideoFormat, VideoRequest};

/// VAE spatial / temporal downscale (latent grid -> pixels). 16x spatial, 4x
/// temporal: `pixels = [3, 4*(T-1)+1, 16*H, 16*W]`.
const VAE_SPATIAL: usize = 16;
const VAE_TEMPORAL: usize = 4;
const LATENT_CHANNELS: usize = 32;

/// Run a HunyuanVideo 1.5 T2V generate to completion. `req.model` must be the
/// Hunyuan id (the caller dispatches on [`crate::model::VideoModelId::is_hunyuan`]).
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
    let frames = plan.frames as usize;
    let height = req.height as usize;
    let width = req.width as usize;

    // Latent grid: temporal (frames-1)/4+1, spatial /16.
    let grid_t = (frames - 1) / VAE_TEMPORAL + 1;
    let grid_h = height / VAE_SPATIAL;
    let grid_w = width / VAE_SPATIAL;

    // --- resolve every weight file (caller ensured they are cached) ---
    let enc_path = resolve_role(manifest, role::ENCODER_GGUF_Q8_0)?;
    let tok_path = resolve_role(manifest, role::TOKENIZER)?;
    let dit_path = resolve_role(manifest, role::DIT)?;
    // VAE file follows the `--vae` choice (only the selected one is downloaded).

    // --- host text prep: video chat template -> tokenize -> crop the system /
    // user-header block (upstream `crop_start`), leaving prompt + gen-prompt ---
    let tokenizer = HfTokenizer::from_path(&tok_path)
        .await
        .map_err(|e| format!("load tokenizer {}: {e:?}", tok_path.display()))?;
    let raw_prompt = req.prompts[0].trim();
    if raw_prompt.is_empty() {
        return Err("empty prompt".into());
    }
    // Phase 0 prompt rewrite: HunyuanVideo 1.5 is trained on long, structured
    // captions, so a short raw prompt is out-of-distribution and yields
    // incoherent video. Expand it via the rewrite endpoint; fall back to the
    // original on any failure (rewriting never blocks a run). Prompt text is
    // never logged.
    let rewritten = crate::rewrite::maybe_rewrite_prompt(
        req.rewrite,
        req.rewrite_quality,
        crate::rewrite::T2V_REWRITE_SYSTEM_PROMPT,
        raw_prompt,
        backend,
        manifest,
        req.budget.vram_bytes,
        sink,
    )
    .await;
    let prompt = rewritten.as_deref().unwrap_or(raw_prompt);
    let full = build_chat_prompt(prompt);
    let ids_full = tokenizer
        .encode(&full, false)
        .map_err(|e| format!("tokenize: {e:?}"))?;
    let marker = tokenizer
        .encode(USER_MARKER, false)
        .map_err(|e| format!("tokenize marker: {e:?}"))?;
    let crop_start = find_subsequence(&ids_full, &marker)
        .map(|i| i + marker.len())
        .unwrap_or(0);
    let token_ids = &ids_full[crop_start..];
    if token_ids.is_empty() {
        return Err("prompt produced no tokens after crop".into());
    }
    let seq = token_ids.len();

    let seed = req.seed.unwrap_or_else(random_seed);
    tracing::info!(
        target: thinfer_core::trace::DIAG,
        model = %req.model, width = req.width, height = req.height,
        frames = plan.frames, fps = plan.fps, seed, tokens = seq,
        grid_t, grid_h, grid_w,
        ram_budget = req.budget.ram_bytes, vram_budget = req.budget.vram_bytes,
        "hunyuan generate start",
    );
    sink.note(&format!(
        "Generating {}x{} video, {} frames @ {} fps (~{:.1}s), seed {} ({})",
        req.width,
        req.height,
        plan.frames,
        plan.fps,
        plan.frames as f32 / plan.fps as f32,
        seed,
        req.model,
    ));

    // Honor a cancel requested during the (multi-minute) rewrite before starting
    // the encoder phase. Denoise polls cancel per step; the phases between are
    // checked at their boundaries so a mid-rewrite/mid-encode cancel is prompt.
    if sink.cancelled() {
        return Err("cancelled".into());
    }

    // === Phase A: Qwen2.5-VL text encoder -> [seq, 3584] (scoped) ===
    sink.stage(Stage::TextEncode);
    let text = {
        let enc = open_mmap(&enc_path).await?;
        let gguf = GgufSource::open(enc)
            .await
            .map_err(|e| format!("parse encoder gguf: {e:?}"))?;
        let renamed = RenamedSource::with_passthrough(gguf, qwen2vl_gguf_renames());
        let residency = WeightResidency::new(renamed, req.budget);
        // even-pad headroom for the rope table (the encoder pads to even seq).
        let encoder = HunyuanTextEncoder::load(backend, &residency, seq + 1)
            .await
            .map_err(|e| format!("load encoder: {e:?}"))?;
        let workspace = Workspace::new(Arc::clone(backend), Arc::clone(residency.arbiter()));
        let out = encoder
            .encode(backend, &residency, &workspace, token_ids)
            .await
            .map_err(|e| format!("encode prompt: {e:?}"))?;
        residency.evict_all_and_free(&**backend);
        out
    };
    if sink.cancelled() {
        return Err("cancelled".into());
    }

    // === Phase B: DiT denoise (4-step flow-match Euler, scoped + evicted) ===
    // `denoise` drives the per-step UI updates via the callback below (its first
    // tick is step 1 with the true denominator = the fixed lightx2v schedule len,
    // which ignores `req.steps`), so no pre-loop tick here -- one would duplicate
    // the "step 1/4" the callback emits at the top of the loop.
    let latent_init = seeded_gaussian(LATENT_CHANNELS * grid_t * grid_h * grid_w, seed);
    let latent = {
        let dit_src = open_st(&dit_path).await?;
        let dit_res = WeightResidency::new(dit_src, req.budget);
        // The DiT streams weights per block while the activation-tiled forward
        // grows a large workspace on the same device. Register the weight
        // reclaimer so workspace pressure EVICTS unpinned residents (streaming
        // in/out under the one budget), replacing the old predicted-reserve
        // carve-out.
        dit_res.arbiter().register(
            RECLAIM_EVICTABLE_WEIGHTS,
            dit_res.reclaimer(Arc::clone(backend)),
        );
        let refiner = HunyuanRefiner::new(
            HunyuanRefinerPipelines::compile_with(backend, ActDtype::F32)
                .await
                .map_err(|e| format!("compile refiner: {e:?}"))?,
            &dit_res,
        )
        .map_err(|e| format!("build refiner: {e:?}"))?;
        let dit = HunyuanDit::new(
            HunyuanDitPipelines::compile_with(backend, ActDtype::Bf16, req.i8_matmul)
                .await
                .map_err(|e| format!("compile dit: {e:?}"))?,
            refiner,
            &dit_res,
            req.i8_matmul,
        )
        .map_err(|e| format!("build dit: {e:?}"))?;
        let dit_ws = Workspace::new(Arc::clone(backend), Arc::clone(dit_res.arbiter()));
        let schedule = FlowMatchSchedule::lightx2v_t2v_480p();
        // Per-step UI progress: each denoise step is ~minutes at 480p, so the SSE
        // step counter must advance per step (the single pre-loop tick above left
        // the UI frozen at step 1 for the whole run).
        let step_progress = |i: u32, n: u32| sink.stage(Stage::Step { i, n });
        // Cooperative cancel: denoise polls this at each step boundary (~minutes
        // apart) and aborts with HunyuanDitError::Cancelled, which the worker maps
        // to a cancelled job state.
        let cancel = || sink.cancelled();
        // Temporal attention window radius in LATENT frames. Unset -> the model
        // default (eyeballed W=3 for Hunyuan, 2026-07-01); an explicit
        // `--attn-window N` opts into the O(frames²)→O(frames·N) joint windowed
        // attention (image queries see ±N frames + all text). `0` = full.
        let window = req
            .attn_window
            .or_else(|| req.model.default_attn_window())
            .unwrap_or(0);
        // Log the EFFECTIVE window (0 = full attention) so the run's attention
        // mode is observable, not inferred from per-step timing.
        tracing::info!(
            target: thinfer_core::trace::DIAG,
            attn_window = window,
            requested = ?req.attn_window,
            "hunyuan denoise attention window",
        );
        let latent = dit
            .denoise(
                backend,
                &dit_res,
                &dit_ws,
                &text,
                seq,
                &latent_init,
                (grid_t, grid_h, grid_w),
                &schedule,
                window,
                Some(&step_progress),
                Some(&cancel),
            )
            .await
            .map_err(|e| format!("denoise: {e:?}"))?;
        drop(dit_ws);
        dit_res.evict_all_and_free(&**backend); // return all DiT VRAM before VAE
        latent
    };

    // === Phase C: VAE decode -> raw [-1, 1] CTHW video ===
    // Tiny (TAEHV) is the fast default (~seconds, draft quality); Full is the
    // conv3d parity VAE (~minutes). Both return CTHW [-1, 1] of the same shape.
    sink.stage(Stage::VaeDecode);
    let mut video = match req.vae {
        // Both tiny choices share the TAEHV decoder (identical arch + config);
        // only the checkpoint differs -- base `taehv1_5` vs the Hunyuan-1.5
        // fine-tune (`TinyFt`, better fidelity at the same seconds-fast decode).
        VaeChoice::Tiny | VaeChoice::TinyFt => {
            let tiny_role = if req.vae == VaeChoice::TinyFt {
                role::TINY_VAE_FT
            } else {
                role::TINY_VAE
            };
            let tiny_path = resolve_role(manifest, tiny_role)?;
            let tiny_src = PytorchSource::open(open_mmap(&tiny_path).await?)
                .await
                .map_err(|e| format!("parse tiny vae {}: {e:?}", tiny_path.display()))?;
            let vae_res = WeightResidency::new(tiny_src, req.budget);
            let handles = register_decoder_taehv(&vae_res, &TaehvDecoderWeights::new())
                .map_err(|e| format!("register tiny vae: {e:?}"))?;
            let decoder = TaehvDecoder {
                pipelines: TaehvPipelines::compile(backend)
                    .await
                    .map_err(|e| format!("compile tiny vae: {e:?}"))?,
                handles,
                cfg: taehv_config(),
            };
            let mut vae_ws = Workspace::new(Arc::clone(backend), Arc::clone(vae_res.arbiter()));
            let out = decoder
                .decode(
                    backend,
                    &vae_res,
                    &mut vae_ws,
                    &latent,
                    grid_t,
                    grid_h,
                    grid_w,
                    None,
                )
                .await
                .map_err(|e| format!("tiny vae decode: {e:?}"))?;
            vae_res.evict_all_and_free(&**backend);
            out
        }
        VaeChoice::Full => {
            let vae_path = resolve_role(manifest, role::VAE)?;
            let vae_src = open_st(&vae_path).await?;
            let vae_res = WeightResidency::new(vae_src, req.budget);
            // All-F32 for reliability. The f16 up-stage seam (new_mixed) is correct
            // at parity dims (vae_tiling f16_upstage rel 0.049%) but caused a DEVICE
            // LOSS at production resolution in serve (BufferMap/device-lost during
            // VAE decode, 2026-06-29) -- root cause TBD: f16 up-tiles size by
            // act_size=2 so they ~2x vs f32 (a single conv3d dispatch can trip the
            // 2s TDR), and/or two co-compiled pipeline sets + the whole-tensor f32
            // mid overshoot the VRAM budget at 480p. Re-enable once the f16 up-stage
            // is TDR/budget-capped and validated at full res. See worklog.
            let vae = HunyuanVaeDecoder::new(
                HunyuanVaePipelines::compile_with(backend, ActDtype::F32)
                    .await
                    .map_err(|e| format!("compile vae: {e:?}"))?,
                &vae_res,
            )
            .map_err(|e| format!("build vae: {e:?}"))?;
            let vae_ws = Workspace::new(Arc::clone(backend), Arc::clone(vae_res.arbiter()));
            let out = vae
                .decode(backend, &vae_res, &vae_ws, &latent, grid_t, grid_h, grid_w)
                .await
                .map_err(|e| format!("vae decode: {e:?}"))?;
            vae_res.evict_all_and_free(&**backend);
            out
        }
    };
    if !video.iter().all(|v| v.is_finite()) {
        return Err("video decode produced non-finite output".into());
    }
    // WanVideo is the shared CTHW [-1,1] frame carrier; honor its clamp contract.
    for v in &mut video {
        *v = v.clamp(-1.0, 1.0);
    }
    let f_px = VAE_TEMPORAL * (grid_t - 1) + 1;
    let h_px = VAE_SPATIAL * grid_h;
    let w_px = VAE_SPATIAL * grid_w;
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

    // --- Encode: silent H.264 MP4 (or PNG frames) ---
    match req.format {
        VideoFormat::Mp4 => {
            sink.note("Encoding MP4 (H.264)");
            crate::codec::encode_mp4(&wan_video, plan.fps, &req.output)?;
        }
        VideoFormat::PngFrames => {
            sink.note("Writing PNG frames");
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

/// Run a HunyuanVideo 1.5 causal I2V generate (minWM WorldPlay dmd) to
/// completion. `req.model` must be the TI2V id (`is_hunyuan_i2v`); the frame
/// grid chunks by 4 (resolve enforces). `input_image` is OPTIONAL: with it the
/// run is I2V (SigLIP + VAE-encoded first frame conditioning); without it the
/// model runs text-only (the upstream `mask_type="t2v"` shape, probe-validated
/// to produce coherent prompt-following video).
///
/// Phases (one heavy weight set resident at a time, like the T2V run):
/// text encode (Qwen2.5-VL) -> [SigLIP vision encode -> VAE-encode the first
/// frame, image runs only] -> chunked AR denoise (4 Euler steps + recache per
/// 4-latent-frame chunk over the host-staged KV cache) -> VAE decode -> MP4.
pub async fn run_i2v(
    backend: &Arc<WgpuBackend>,
    req: &VideoRequest,
    sink: &dyn ProgressSink,
) -> Result<JobSummary, String> {
    use thinfer_models::hunyuan::config::ar as arcfg;
    use thinfer_models::hunyuan::dit::ar::{HunyuanArDit, kv_cache_bytes};
    use thinfer_models::hunyuan::siglip::{self, SiglipEncoder};
    use thinfer_models::hunyuan::vae::encode::HunyuanVaeEncoder;

    let plan = req.resolve()?;
    for w in &plan.warnings {
        tracing::warn!(target: thinfer_core::trace::DIAG, "{w}");
        sink.note(w);
    }
    let manifest = req.model.manifest();
    let frames = plan.frames as usize;
    let height = req.height as usize;
    let width = req.width as usize;
    let grid_t = (frames - 1) / VAE_TEMPORAL + 1;
    let grid_h = height / VAE_SPATIAL;
    let grid_w = width / VAE_SPATIAL;
    if !grid_t.is_multiple_of(arcfg::CHUNK_LATENT_FRAMES) {
        return Err(format!(
            "latent frame count {grid_t} must chunk by {} (frames 13, 29, 45, 61, 77, ...)",
            arcfg::CHUNK_LATENT_FRAMES
        ));
    }

    let enc_path = resolve_role(manifest, role::ENCODER_GGUF_Q8_0)?;
    let tok_path = resolve_role(manifest, role::TOKENIZER)?;
    let dit_path = resolve_role(manifest, role::DIT_I2V)?;

    // --- input image (optional): decode once, resize twice (video dims for the
    // VAE conditioning latent; 384x384 for SigLIP). None = text-only run. ---
    let to_chw_norm = |img: &image::RgbImage, w: usize, h: usize| -> Vec<f32> {
        let mut out = vec![0.0f32; 3 * w * h];
        for (x, y, px) in img.enumerate_pixels() {
            for c in 0..3 {
                out[c * w * h + y as usize * w + x as usize] = px.0[c] as f32 / 127.5 - 1.0;
            }
        }
        out
    };
    let pixels = match req.input_image.as_ref() {
        Some(input_image) => {
            let rgb = image::load_from_memory(&input_image.0)
                .map_err(|e| format!("decode first frame: {e}"))?
                .to_rgb8();
            sink.note(&format!(
                "Loaded first frame {}x{}",
                rgb.width(),
                rgb.height()
            ));
            let frame_px = {
                let resized = image::imageops::resize(
                    &rgb,
                    width as u32,
                    height as u32,
                    image::imageops::FilterType::CatmullRom,
                );
                to_chw_norm(&resized, width, height)
            };
            let siglip_px = {
                let resized = image::imageops::resize(
                    &rgb,
                    siglip::IMAGE_SIZE as u32,
                    siglip::IMAGE_SIZE as u32,
                    image::imageops::FilterType::CatmullRom,
                );
                to_chw_norm(&resized, siglip::IMAGE_SIZE, siglip::IMAGE_SIZE)
            };
            Some((frame_px, siglip_px))
        }
        None => {
            sink.note("No input image: text-only generation");
            None
        }
    };

    // --- host text prep (same template/crop as T2V) ---
    let tokenizer = HfTokenizer::from_path(&tok_path)
        .await
        .map_err(|e| format!("load tokenizer {}: {e:?}", tok_path.display()))?;
    let raw_prompt = req.prompts[0].trim();
    if raw_prompt.is_empty() {
        return Err("empty prompt".into());
    }
    // Rewrite applies to TEXT-ONLY runs (a plain t2v rewrite, same as the T2V
    // model). Image runs skip it: the text-only rewriter can't see the image
    // and would describe a scene that contradicts it (image-aware rewrite is a
    // follow-up).
    let rewritten = if pixels.is_none() {
        crate::rewrite::maybe_rewrite_prompt(
            req.rewrite,
            req.rewrite_quality,
            crate::rewrite::T2V_REWRITE_SYSTEM_PROMPT,
            raw_prompt,
            backend,
            manifest,
            req.budget.vram_bytes,
            sink,
        )
        .await
    } else {
        if req.rewrite {
            sink.note("Prompt rewrite is skipped on image-conditioned runs (text-only rewriter)");
        }
        None
    };
    let prompt = rewritten.as_deref().unwrap_or(raw_prompt);
    let full = build_chat_prompt(prompt);
    let ids_full = tokenizer
        .encode(&full, false)
        .map_err(|e| format!("tokenize: {e:?}"))?;
    let marker = tokenizer
        .encode(USER_MARKER, false)
        .map_err(|e| format!("tokenize marker: {e:?}"))?;
    let crop_start = find_subsequence(&ids_full, &marker)
        .map(|i| i + marker.len())
        .unwrap_or(0);
    let token_ids = &ids_full[crop_start..];
    if token_ids.is_empty() {
        return Err("prompt produced no tokens after crop".into());
    }
    let seq = token_ids.len();

    // --- RAM feasibility: the host-staged KV cache is the dominant non-weight
    // RAM cost (54 layers x K+V x bf16 over every txt+vision+latent token).
    // Respect the request's RAM budget rather than discovering an OOM later.
    let kv_bytes = kv_cache_bytes((grid_t, grid_h, grid_w), seq);
    if kv_bytes > req.budget.ram_bytes {
        return Err(format!(
            "the causal KV cache needs ~{:.1} GB host RAM at {} frames ({}x{}), over the \
             --ram-budget ({:.1} GB); reduce frames (e.g. {}) or raise the budget",
            kv_bytes as f64 / 1e9,
            frames,
            width,
            height,
            req.budget.ram_bytes as f64 / 1e9,
            // Largest legal frame count whose KV fits the budget.
            {
                let row = 54 * 2 * 2048 * 2u64;
                let tokens_max = req.budget.ram_bytes / row;
                let img_max =
                    tokens_max.saturating_sub((arcfg::VISION_TOKENS + seq) as u64) as usize;
                let t_max = (img_max / (grid_h * grid_w)).max(arcfg::CHUNK_LATENT_FRAMES);
                let t_fit = (t_max / arcfg::CHUNK_LATENT_FRAMES) * arcfg::CHUNK_LATENT_FRAMES;
                4 * t_fit.max(arcfg::CHUNK_LATENT_FRAMES) - 3
            },
        ));
    }

    let seed = req.seed.unwrap_or_else(random_seed);
    tracing::info!(
        target: thinfer_core::trace::DIAG,
        model = %req.model, width = req.width, height = req.height,
        frames = plan.frames, fps = plan.fps, seed, tokens = seq,
        grid_t, grid_h, grid_w, kv_cache_gb = kv_bytes as f64 / 1e9,
        ram_budget = req.budget.ram_bytes, vram_budget = req.budget.vram_bytes,
        "hunyuan i2v generate start",
    );
    sink.note(&format!(
        "Generating {}x{} video from the input image, {} frames @ {} fps (~{:.1}s), seed {} ({})",
        req.width,
        req.height,
        plan.frames,
        plan.fps,
        plan.frames as f32 / plan.fps as f32,
        seed,
        req.model,
    ));
    if sink.cancelled() {
        return Err("cancelled".into());
    }

    // === Phase A: Qwen2.5-VL text encoder -> [seq, 3584] (scoped) ===
    sink.stage(Stage::TextEncode);
    let text = {
        let enc = open_mmap(&enc_path).await?;
        let gguf = GgufSource::open(enc)
            .await
            .map_err(|e| format!("parse encoder gguf: {e:?}"))?;
        let renamed = RenamedSource::with_passthrough(gguf, qwen2vl_gguf_renames());
        let residency = WeightResidency::new(renamed, req.budget);
        let encoder = HunyuanTextEncoder::load(backend, &residency, seq + 1)
            .await
            .map_err(|e| format!("load encoder: {e:?}"))?;
        let workspace = Workspace::new(Arc::clone(backend), Arc::clone(residency.arbiter()));
        let out = encoder
            .encode(backend, &residency, &workspace, token_ids)
            .await
            .map_err(|e| format!("encode prompt: {e:?}"))?;
        residency.evict_all_and_free(&**backend);
        out
    };
    if sink.cancelled() {
        return Err("cancelled".into());
    }

    // === Phase B: VAE-encode the first frame -> cond latent [32, H/16, W/16]
    // (image runs only) ===
    let cond0 = match &pixels {
        Some((frame_px, _)) => {
            sink.note("Encoding the first frame (VAE)");
            let vae_path = resolve_role(manifest, role::VAE)?;
            let vae_src = open_st(&vae_path).await?;
            let vae_res = WeightResidency::new(vae_src, req.budget);
            let enc = HunyuanVaeEncoder::new(&vae_res)
                .map_err(|e| format!("build vae encoder: {e:?}"))?;
            let pl = HunyuanVaePipelines::compile_with(backend, ActDtype::F32)
                .await
                .map_err(|e| format!("compile vae: {e:?}"))?;
            let ws = Workspace::new(Arc::clone(backend), Arc::clone(vae_res.arbiter()));
            let out = enc
                .encode_frame(backend, &vae_res, &ws, &pl, frame_px, height, width)
                .await
                .map_err(|e| format!("vae encode: {e:?}"))?;
            vae_res.evict_all_and_free(&**backend);
            Some(out)
        }
        None => None,
    };
    if sink.cancelled() {
        return Err("cancelled".into());
    }

    // === Phase C + D: SigLIP vision encode (image runs only), then the chunked
    // AR denoise. The DiT block pipelines are compiled once and shared. ===
    let dit_pipelines = thinfer_models::hunyuan::dit::HunyuanDitPipelines::compile_with(
        backend,
        ActDtype::Bf16,
        req.i8_matmul,
    )
    .await
    .map_err(|e| format!("compile dit: {e:?}"))?;

    let vision = match &pixels {
        Some((_, siglip_px)) => {
            sink.note("Encoding the first frame (SigLIP)");
            let siglip_path = resolve_role(manifest, role::SIGLIP)?;
            let sig_src = open_st(&siglip_path).await?;
            let sig_res = WeightResidency::new(sig_src, req.budget);
            let sig = SiglipEncoder::new(&sig_res).map_err(|e| format!("build siglip: {e:?}"))?;
            let ws = Workspace::new(Arc::clone(backend), Arc::clone(sig_res.arbiter()));
            let out = sig
                .encode(backend, &sig_res, &ws, &dit_pipelines.bp, siglip_px)
                .await
                .map_err(|e| format!("siglip encode: {e:?}"))?;
            sig_res.evict_all_and_free(&**backend);
            Some(out)
        }
        None => None,
    };
    if sink.cancelled() {
        return Err("cancelled".into());
    }

    let latent_init = seeded_gaussian(LATENT_CHANNELS * grid_t * grid_h * grid_w, seed);
    let latent = {
        let dit_src = open_st(&dit_path).await?;
        let dit_res = WeightResidency::new(dit_src, req.budget);
        // Weights join the VRAM arbiter's reclaim chain so workspace growth (the
        // KV ping-pong buffers + chunk transients) EVICTS unpinned residents
        // instead of overshooting the budget -- streaming under pressure, not a
        // predicted reserve.
        dit_res.arbiter().register(
            RECLAIM_EVICTABLE_WEIGHTS,
            dit_res.reclaimer(Arc::clone(backend)),
        );
        let refiner = HunyuanRefiner::new(
            HunyuanRefinerPipelines::compile_with(backend, ActDtype::F32)
                .await
                .map_err(|e| format!("compile refiner: {e:?}"))?,
            &dit_res,
        )
        .map_err(|e| format!("build refiner: {e:?}"))?;
        let dit = HunyuanArDit::new(dit_pipelines, refiner, &dit_res, req.i8_matmul)
            .map_err(|e| format!("build ar dit: {e:?}"))?;
        let dit_ws = Workspace::new(Arc::clone(backend), Arc::clone(dit_res.arbiter()));
        let schedule =
            FlowMatchSchedule::build(&arcfg::DENOISING_STEP_LIST, arcfg::FLOW_SHIFT, 1000);
        // AR progress is chunk-shaped (the web renders "Denoising chunk c/N step
        // s/M"); the final per-chunk "step" is the KV recache pass.
        let step_progress = |chunk: u32, num_chunks: u32, step: u32, num_steps: u32| {
            sink.stage(Stage::ChunkStep {
                chunk,
                num_chunks,
                step,
                num_steps,
            })
        };
        let cancel = || sink.cancelled();
        let latent = dit
            .generate(
                backend,
                &dit_res,
                &dit_ws,
                &text,
                seq,
                vision.as_deref(),
                cond0.as_deref(),
                &latent_init,
                (grid_t, grid_h, grid_w),
                &schedule,
                Some(&step_progress),
                Some(&cancel),
            )
            .await
            .map_err(|e| format!("ar denoise: {e:?}"))?;
        drop(dit_ws);
        dit_res.evict_all_and_free(&**backend);
        latent
    };

    // === Phase E: VAE decode (same choices as T2V) ===
    sink.stage(Stage::VaeDecode);
    let mut video = match req.vae {
        VaeChoice::Tiny | VaeChoice::TinyFt => {
            let tiny_role = if req.vae == VaeChoice::TinyFt {
                role::TINY_VAE_FT
            } else {
                role::TINY_VAE
            };
            let tiny_path = resolve_role(manifest, tiny_role)?;
            let tiny_src = PytorchSource::open(open_mmap(&tiny_path).await?)
                .await
                .map_err(|e| format!("parse tiny vae {}: {e:?}", tiny_path.display()))?;
            let vae_res = WeightResidency::new(tiny_src, req.budget);
            let handles = register_decoder_taehv(&vae_res, &TaehvDecoderWeights::new())
                .map_err(|e| format!("register tiny vae: {e:?}"))?;
            let decoder = TaehvDecoder {
                pipelines: TaehvPipelines::compile(backend)
                    .await
                    .map_err(|e| format!("compile tiny vae: {e:?}"))?,
                handles,
                cfg: taehv_config(),
            };
            let mut vae_ws = Workspace::new(Arc::clone(backend), Arc::clone(vae_res.arbiter()));
            let out = decoder
                .decode(
                    backend,
                    &vae_res,
                    &mut vae_ws,
                    &latent,
                    grid_t,
                    grid_h,
                    grid_w,
                    None,
                )
                .await
                .map_err(|e| format!("tiny vae decode: {e:?}"))?;
            vae_res.evict_all_and_free(&**backend);
            out
        }
        VaeChoice::Full => {
            let vae_path = resolve_role(manifest, role::VAE)?;
            let vae_src = open_st(&vae_path).await?;
            let vae_res = WeightResidency::new(vae_src, req.budget);
            let vae = HunyuanVaeDecoder::new(
                HunyuanVaePipelines::compile_with(backend, ActDtype::F32)
                    .await
                    .map_err(|e| format!("compile vae: {e:?}"))?,
                &vae_res,
            )
            .map_err(|e| format!("build vae: {e:?}"))?;
            let vae_ws = Workspace::new(Arc::clone(backend), Arc::clone(vae_res.arbiter()));
            let out = vae
                .decode(backend, &vae_res, &vae_ws, &latent, grid_t, grid_h, grid_w)
                .await
                .map_err(|e| format!("vae decode: {e:?}"))?;
            vae_res.evict_all_and_free(&**backend);
            out
        }
    };
    if !video.iter().all(|v| v.is_finite()) {
        return Err("video decode produced non-finite output".into());
    }
    for v in &mut video {
        *v = v.clamp(-1.0, 1.0);
    }
    let f_px = VAE_TEMPORAL * (grid_t - 1) + 1;
    let h_px = VAE_SPATIAL * grid_h;
    let w_px = VAE_SPATIAL * grid_w;
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
    match req.format {
        VideoFormat::Mp4 => {
            sink.note("Encoding MP4 (H.264)");
            crate::codec::encode_mp4(&wan_video, plan.fps, &req.output)?;
        }
        VideoFormat::PngFrames => {
            sink.note("Writing PNG frames");
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

/// First index where `needle` occurs as a contiguous subsequence of `hay`.
fn find_subsequence(hay: &[u32], needle: &[u32]) -> Option<usize> {
    if needle.is_empty() || needle.len() > hay.len() {
        return None;
    }
    (0..=hay.len() - needle.len()).find(|&i| &hay[i..i + needle.len()] == needle)
}

/// Deterministic standard-normal noise of length `n` from `seed` (splitmix64 ->
/// Box-Muller). The product path owns the seed (no parity constraint on the RNG;
/// the parity gate pins its own noise tensor).
fn seeded_gaussian(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next_u64 = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    // Uniform in (0, 1]: take the top 53 bits (avoid 0 for the log).
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
