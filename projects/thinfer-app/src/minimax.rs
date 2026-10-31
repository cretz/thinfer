//! MiniMax-H3 t2va driver (text -> video + audio).
//!
//! Models can't open files, so the file IO and the per-phase
//! `WeightResidency` lifecycle live here; `thinfer_models::minimax::*` stays
//! pure glue. The chain: tokenize -> truncated Qwen3-VL-32B conditioner ->
//! packed-sequence denoise -> video VAE (a ViT) + audio VAE (a BigVGAN) ->
//! H.264 + AAC MP4.
//!
//! Three things make this pipeline unlike the others here:
//!
//! - **One packed sequence.** Text, audio and video rows share a single 1-D
//!   sequence under full self-attention. There is no cross-attention, so there
//!   is no conditioning tail to build: the conditioner's hidden state is simply
//!   projected into the same stream.
//! - **AdaLN is precomputed at load.** `blocks.N.adaln_proj` is ~13B of the 33B
//!   and reads only the timestep embedding, so the whole modulation table for
//!   the run is built once (~78 MB) and those weights are never streamed. That
//!   is what takes the streamed DiT from ~18.5 GiB to ~10.8 GiB per step.
//! - **Adapters are applied at RUNTIME**, not folded: the turbo distillation
//!   and any user (vault) LoRAs, stacked into one low-rank pair per site.
//!   Folding would re-encode every Q4_K site to Q8_0 and nearly double both the
//!   streaming and the host cache, and cannot touch an NVFP4 site at all; see
//!   `minimax::dit::BlockLora`.
//!
//! Each weight file gets its own scoped residency (built -> used -> dropped) so
//! only one large set is VRAM-resident at a time: the 13.6 GiB conditioner,
//! then the DiT, then the two VAEs.

use std::sync::Arc;

use thinfer_core::backend::{Backend, WgpuBackend};
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::format::union::UnionSource;
use thinfer_core::ops::{ActDtype, WeightDtype};
use thinfer_core::quant::QuantKind;
use thinfer_core::residency::WeightResidency;
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::Workspace;
use thinfer_models::common::bigvgan::read_weight_f32;
use thinfer_models::common::merge::MergedSource;
use thinfer_models::minimax::DitConfig;
use thinfer_models::minimax::audio_vae::{AudioVae, AudioVaeConfig};
use thinfer_models::minimax::dit::{
    COOPMAT_SITES, DitPipelines, dit_cfgs, register_handles as register_dit,
};
use thinfer_models::minimax::fasth3::FastH3Source;
use thinfer_models::minimax::loader::{build_modulation_tables, register_video_patch_proj};
use thinfer_models::minimax::lora::{Adapter, LoraSource, TURBO_ALPHA};
use thinfer_models::minimax::manifest as mmf;
use thinfer_models::minimax::manifest::role;
use thinfer_models::minimax::packing;
use thinfer_models::minimax::pipeline::{
    DenoisePlan, LONG_SEQUENCE_WEIGHT_BUDGET, Latents, MAX_PACKED_ROWS, audio_rows_to_latents,
    denoise, max_packed_rows, upload_modulation_tables, video_rows_to_latents,
};
use thinfer_models::minimax::scheduler::JointSchedule;
use thinfer_models::minimax::text_encoder::{
    EncoderPipelines, ImageRows, MiniMaxTextEncoder, config as enc_config, encoder_cfgs,
    encoder_cfgs_long, register_handles as register_encoder,
};
use thinfer_models::minimax::video_vae::{
    DecodedTokens, DecoderPipelines, PixelBlock, TilePolicy, VideoVaeConfig, decode_tokens,
    load_consts, register_handles as register_vae,
};
use thinfer_models::wan::pipeline::WanVideo;
use thinfer_native::MmapFileOpener;
use thinfer_native::tokenizer::HfTokenizer;

use crate::config::{ResidencyBudget, random_seed};
use crate::download::resolve_role;
use crate::executor::BytesOpener;
use crate::minimax_ref;
use crate::progress::{ProgressSink, Stage};
use crate::request::{JobSummary, VideoFormat, VideoRequest};

/// H3 renders at a fixed 24 fps; the rotary clock and the 40-latents/s audio
/// grid are both tied to it, so this is not a user knob.
pub const FPS: u32 = 24;
/// Audio sample rate the audio VAE emits.
const AUDIO_SR: u32 = 32_000;

/// VRAM weight-residency cap.
///
/// The 33B DiT streams whatever the budget, and the block's activation peak is
/// large (the fused SwiGLU is `rows x 28672`), so the weight LRU must not fill
/// to the ceiling: a residency sized near the device leaves nothing for the
/// workspace and the forward OOMs mid-block rather than evicting. 4G on the 8GB
/// card leaves the activations room and costs nothing, since the DiT is far
/// larger than any cache we could hold anyway.
const VRAM_BUDGET_CAP: u64 = 4 * 1024 * 1024 * 1024;

/// Run a MiniMax-H3 t2va generate to completion.
pub async fn run(
    backend: &Arc<WgpuBackend>,
    req: &VideoRequest,
    sink: &dyn ProgressSink,
) -> Result<JobSummary, String> {
    let fasth3 = req.model.is_fasth3();
    // FastH3's NVFP4 block runs row-chunked on the accelerator, which lifts the
    // packed-row limit; the backend block (base H3, or no accelerator) keeps
    // the old one. A blank clip length is capped to it.
    let row_limit = max_packed_rows(fasth3 && backend.accelerator().is_some());
    let plan_meta = req.resolve_minimax(row_limit)?;
    for w in &plan_meta.warnings {
        tracing::warn!(target: thinfer_core::trace::DIAG, "{w}");
        sink.note(w);
    }
    let manifest = req.model.manifest();
    let cfg = DitConfig {
        vsa_int8_qk: req.attn == crate::model::AttnPrecision::Int8,
        ..req.model.minimax_dit_config()
    };
    let vae_cfg = VideoVaeConfig::h3();

    // --- geometry ---
    // Frames snap up to `17n + 5` (latent `5n + 2`); the spatial axes are
    // whole latent voxels after the 16x compression.
    let frames = packing::align_num_frames(plan_meta.frames as usize)
        .map_err(|e| format!("frame count: {e:?}"))?;
    let latent_frames =
        packing::video_latent_num_frames(frames).map_err(|e| format!("latent frames: {e:?}"))?;
    let (height, width) = (req.height as usize, req.width as usize);
    if !height.is_multiple_of(32) || !width.is_multiple_of(32) {
        return Err(format!(
            "MiniMax-H3 needs both axes a multiple of 32, got {width}x{height}"
        ));
    }
    let (lat_h, lat_w) = (
        height / vae_cfg.spatial_compression,
        width / vae_cfg.spatial_compression,
    );
    let audio_latents = packing::audio_latent_num_frames(frames);

    let prompt = req.prompts[0].trim();
    if prompt.is_empty() {
        return Err("empty prompt produced no tokens".into());
    }

    // --- resolve every weight file (caller ensured they are cached) ---
    let enc_path = resolve_role(manifest, role::ENCODER_GGUF)?;
    let tok_path = resolve_role(manifest, role::TOKENIZER)?;
    let vae_path = resolve_role(manifest, role::VIDEO_VAE)?;
    let audio_vae_path = resolve_role(manifest, role::AUDIO_VAE)?;

    // --- tokenize. NO chat template and NO special tokens: H3 conditions on
    //     the prompt verbatim (the same finding as LTX). Adding a BOS shifts
    //     every text row's rotary position and with it the whole media clock.
    let tokenizer = HfTokenizer::from_path(&tok_path)
        .await
        .map_err(|e| format!("load tokenizer {}: {e:?}", tok_path.display()))?;
    // Reference images (FastH3-Omni): each adds a labelled vision block to the
    // prompt and one merged token per 32x32 pixels, and as many DiT rows.
    if !req.ref_images.is_empty() && cfg.pdd.is_none() {
        return Err(format!("{} does not take reference images", req.model));
    }
    let refs = req
        .ref_images
        .iter()
        .map(|b| minimax_ref::prepare(&b.0))
        .collect::<Result<Vec<_>, _>>()?;
    let ref_tokens: Vec<usize> = refs
        .iter()
        .map(|r| (r.height / 32) * (r.width / 32))
        .collect();
    let prompt_in = minimax_ref::assemble(&tokenizer, prompt, &ref_tokens)?;
    let ids = prompt_in.ids;
    if ids.len() == ref_tokens.iter().map(|n| n + 2).sum::<usize>() {
        return Err("empty prompt produced no tokens".into());
    }
    let num_text = ids.len();

    // Refuse a canvas we cannot run BEFORE loading 13.6 GiB of conditioner.
    let rows = packing::sequence_rows(&cfg, num_text, latent_frames, lat_h, lat_w, audio_latents)
        + ref_tokens.iter().sum::<usize>();
    if rows > row_limit {
        return Err(format!(
            "{width}x{height} at {frames} frames packs {rows} rows, over the {row_limit}-row \
             limit: one block's activations would not fit. Reduce the resolution or the duration."
        ));
    }

    // FastH3 runs its trained rungs; the base runs `steps` turbo evaluations.
    let steps = if let Some(pdd) = cfg.pdd {
        pdd.steps
    } else if fasth3 {
        mmf::FASTH3_RUNGS.len()
    } else {
        req.steps.max(1) as usize
    };
    let seed = req.seed.unwrap_or_else(random_seed);
    let budget = ResidencyBudget {
        vram_bytes: req.budget.vram_bytes.min(VRAM_BUDGET_CAP),
        ..req.budget
    };
    tracing::info!(
        target: thinfer_core::trace::DIAG,
        model = %req.model, width, height, frames, latent_frames, audio_latents,
        rows, steps, seed, tokens = num_text,
        ram_budget = budget.ram_bytes, vram_budget = budget.vram_bytes,
        "minimax-h3 generate start",
    );
    sink.note(&format!(
        "Generating {width}x{height} joint AV, {frames} frames @ {FPS} fps (~{:.1}s), \
         {rows} packed rows, seed {seed}",
        frames as f32 / FPS as f32,
    ));

    // The conditioner leaves the disk mostly idle: warm the DiT's files (then
    // the video VAE's) in the page cache meanwhile, so the first denoise step
    // does not read them cold. Skipped without the RAM to hold them.
    let dit_roles: &[&str] = if cfg.pdd.is_some() {
        &[role::FASTH3_OMNI]
    } else if fasth3 {
        &[role::FASTH3_TRIM_NVFP4, role::FASTH3_TRIM_REST]
    } else {
        &[role::DIT_GGUF_Q4_K_M, role::TURBO_LORA]
    };
    let mut warm = dit_roles
        .iter()
        .map(|r| resolve_role(manifest, r))
        .collect::<Result<Vec<_>, _>>()?;
    warm.push(vae_path.clone());
    let _prefetch = thinfer_native::prefetch::Prefetch::start(warm);

    // === Phase A: the truncated Qwen3-VL-32B conditioner ===
    sink.stage(Stage::TextEncode);
    let vision = if refs.is_empty() {
        Vec::new()
    } else {
        sink.note(&format!("Encoding {} reference image(s)", refs.len()));
        minimax_ref::vision_features(backend, manifest, budget, &refs).await?
    };
    let images: Vec<ImageRows<'_>> = prompt_in
        .image_starts
        .iter()
        .zip(&vision)
        .map(|(&start, vision)| ImageRows { start, vision })
        .collect();
    let text = {
        let opener = open_mmap(&enc_path).await?;
        let src = GgufSource::open(opener)
            .await
            .map_err(|e| format!("parse encoder gguf: {e:?}"))?;
        let residency = WeightResidency::new(src, budget);
        let handles =
            register_encoder(&residency, None).map_err(|e| format!("register encoder: {e:?}"))?;
        // Reference images make the prompt thousands of tokens long: run it
        // on bf16 acts (tensor-core matmuls and attention). Text alone keeps
        // the f32 path.
        let cfgs = if images.is_empty() {
            encoder_cfgs()
        } else {
            encoder_cfgs_long()
        };
        let pipelines = EncoderPipelines::compile(backend, &cfgs)
            .await
            .map_err(|e| format!("compile encoder pipelines: {e:?}"))?;
        let ws = Workspace::new(Arc::clone(backend), Arc::clone(residency.arbiter()));
        let out = MiniMaxTextEncoder
            .forward(
                backend,
                &pipelines,
                &residency,
                &ws,
                &handles,
                residency.source(),
                &ids,
                &images,
            )
            .await
            .map_err(|e| format!("encoder forward: {e:?}"))?;
        residency.evict_all_and_free(&**backend);
        debug_assert_eq!(out.hidden.len(), num_text * enc_config::HIDDEN);
        out.hidden
    };
    drop(images);
    drop(vision);
    let (conditions, cond_rows) = if refs.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        minimax_ref::latent_conditions(backend, &cfg, &vae_path, budget, &refs).await?
    };
    drop(refs);
    let cond = Conditioning {
        text_tags: prompt_in.tags,
        conditions,
        cond_rows,
    };

    check_cancel(sink)?;
    // === Phase B: the DiT ===
    let dims = PlanDims {
        rows,
        row_limit,
        latent_frames,
        lat_h,
        lat_w,
        audio_latents,
    };
    let latents = if let Some(pdd) = cfg.pdd {
        let file = open_st(&resolve_role(manifest, role::FASTH3_OMNI)?).await?;
        let src = FastH3Source::omni_comfy(&cfg, file)
            .map_err(|e| format!("map fasth3-omni checkpoint: {e:?}"))?;
        let schedule =
            JointSchedule::from_pdd(&pdd).map_err(|e| format!("fasth3-omni schedule: {e}"))?;
        let users = user_adapters(req).await?;
        run_dit_users(
            backend, &cfg, src, users, schedule, dims, &text, &cond, seed, budget, sink,
        )
        .await?
    } else if fasth3 {
        let nvfp4 = open_st(&resolve_role(manifest, role::FASTH3_TRIM_NVFP4)?).await?;
        let rest = open_st(&resolve_role(manifest, role::FASTH3_TRIM_REST)?).await?;
        let src = FastH3Source::new(&cfg, UnionSource::new(nvfp4, rest))
            .map_err(|e| format!("map fasth3 checkpoint: {e:?}"))?;
        let schedule = JointSchedule::from_rungs(
            &mmf::FASTH3_RUNGS,
            mmf::FASTH3_SHIFT_VIDEO,
            mmf::FASTH3_SHIFT_AUDIO,
        )
        .map_err(|e| format!("fasth3 schedule: {e}"))?;
        let users = user_adapters(req).await?;
        run_dit_users(
            backend, &cfg, src, users, schedule, dims, &text, &cond, seed, budget, sink,
        )
        .await?
    } else {
        let opener = open_mmap(&resolve_role(manifest, role::DIT_GGUF_Q4_K_M)?).await?;
        let base = GgufSource::open(opener)
            .await
            .map_err(|e| format!("parse dit gguf: {e:?}"))?;
        let turbo_path = resolve_role(manifest, role::TURBO_LORA)?;
        // `steps` is model evaluations; the grid adds the terminal sigma.
        let schedule = JointSchedule::new(steps + 1).map_err(|e| format!("schedule: {e}"))?;
        let users = user_adapters(req).await?;
        if users.is_empty() {
            let turbo = SafetensorsSource::open(open_mmap(&turbo_path).await?)
                .await
                .map_err(|e| format!("parse turbo lora: {e:?}"))?;
            let lora = LoraSource::turbo(&cfg, turbo)
                .await
                .map_err(|e| format!("turbo lora: {e:?}"))?;
            let src = MergedSource::new(base, lora);
            run_dit(
                backend, &cfg, src, schedule, dims, &text, &cond, seed, budget, sink, true,
            )
            .await?
        } else {
            // Stacked with user adapters, the turbo shares their in-RAM type.
            let bytes = std::fs::read(&turbo_path)
                .map_err(|e| format!("read {}: {e}", turbo_path.display()))?;
            let turbo = SafetensorsSource::open(BytesOpener::new(bytes))
                .await
                .map_err(|e| format!("parse turbo lora: {e:?}"))?;
            let adapters = std::iter::once(Adapter {
                source: turbo,
                strength: 1.0,
                alpha: Some(TURBO_ALPHA),
            })
            .chain(users)
            .collect();
            let src = MergedSource::new(base, lora_source(&cfg, adapters, sink).await?);
            run_dit(
                backend, &cfg, src, schedule, dims, &text, &cond, seed, budget, sink, true,
            )
            .await?
        }
    };
    drop(text);

    check_cancel(sink)?;
    // === Phase C: video VAE (a 36-layer ViT, tiled) ===
    sink.stage(Stage::VaeDecode);
    let grid = (latent_frames, lat_h, lat_w);
    // The NVFP4 decoders need the accelerator's FP4 GEMM (NVIDIA Blackwell);
    // elsewhere the fp16 decoder runs instead (they share the latent space).
    let fp4 = backend
        .accelerator()
        .and_then(|a| a.accepts_gemm_nvfp4(128, 2048, 2048))
        .is_some();
    let nvfp4 = req.vae.minimax_nvfp4().filter(|_| fp4);
    if req.vae.minimax_nvfp4().is_some() && !fp4 {
        sink.note("NVFP4 decoder needs an NVIDIA Blackwell GPU with CUDA; using the fp16 decoder");
    }
    let decoded = match nvfp4 {
        None => {
            let src = open_st(&vae_path).await?;
            decode_video(backend, src, &vae_cfg, &cfg, &latents.video, grid, budget).await?
        }
        // FastVideo's NVFP4 decoder over the fp16 VAE (latent stats, mask
        // token).
        Some((role, layers)) => {
            let vae_cfg = VideoVaeConfig {
                decoder_layers: layers,
                ..vae_cfg
            };
            let decoder = open_st(&resolve_role(manifest, role)?).await?;
            let host = open_st(&vae_path).await?;
            let src = FastH3Source::vae_decoder(&vae_cfg, UnionSource::new(decoder, host))
                .map_err(|e| format!("map nvfp4 video decoder: {e:?}"))?;
            decode_video(backend, src, &vae_cfg, &cfg, &latents.video, grid, budget).await?
        }
    };
    tracing::info!(target: thinfer_core::trace::DIAG, "video vae done");
    // The stitch and the H.264 encode are host-only: they run on a worker
    // while the audio VAE has the GPU.
    let format = req.format;
    let stitcher = std::thread::Builder::new()
        .name("h3-vae-stitch".into())
        .spawn(move || -> Result<_, String> {
            let video = clip_frames(decoded.stitch(), frames)?;
            tracing::info!(target: thinfer_core::trace::DIAG, "video frames ready");
            let dims = (video.height, video.width, video.num_frames);
            let clip = match format {
                VideoFormat::Mp4 => Clip::H264(crate::codec::encode_h264(&video, FPS)?),
                VideoFormat::PngFrames => Clip::Frames(video),
            };
            Ok((dims, clip))
        })
        .map_err(|e| format!("spawn vae stitch: {e}"))?;

    check_cancel(sink)?;
    // === Phase D: audio VAE (the BigVGAN family we already have) ===
    let wav = if req.audio {
        sink.note("Decoding audio");
        let src = open_st(&audio_vae_path).await?;
        let residency = WeightResidency::new(src, budget);
        let vae = AudioVae::new(backend, &residency, AudioVaeConfig::h3())
            .await
            .map_err(|e| format!("build audio vae: {e:?}"))?;
        let pipelines = AudioVae::compile(backend)
            .await
            .map_err(|e| format!("compile audio vae: {e:?}"))?;
        let ws = Workspace::new(Arc::clone(backend), Arc::clone(residency.arbiter()));
        let z = audio_rows_to_latents(&cfg, &latents.audio, audio_latents);
        let wav = vae
            .decode(backend, &residency, &ws, &pipelines, &z, audio_latents)
            .await
            .map_err(|e| format!("audio vae decode: {e:?}"))?;
        residency.evict_all_and_free(&**backend);
        if !wav.iter().all(|v| v.is_finite()) {
            return Err("audio decode produced non-finite output".into());
        }
        Some(wav)
    } else {
        None
    };

    tracing::info!(target: thinfer_core::trace::DIAG, "audio vae done");
    let ((h_px, w_px, f_px), clip) = stitcher
        .join()
        .map_err(|_| "vae stitch panicked".to_string())??;

    check_cancel(sink)?;
    // --- MP4 (the worker's H.264 + optional AAC audio), or PNG frames ---
    match clip {
        Clip::H264(video) => {
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
            crate::progress::write_artifact(sink, &req.output, &video.mux(audio)?)?;
        }
        Clip::Frames(video) => {
            sink.note("Writing PNG frames (audio dropped for png-frames output)");
            crate::codec::write_png_frames(&video, &req.output)?;
        }
    }
    tracing::info!(target: thinfer_core::trace::DIAG, path = %req.output.display(), "wrote output");

    Ok(JobSummary {
        output: req.output.clone(),
        width: w_px as u32,
        height: h_px as u32,
        frames: f_px as u32,
        fps: Some(FPS),
        seed: Some(seed),
    })
}

/// Phase C's GPU half (the video VAE's ViT passes, read back) over any source
/// carrying the H3 decoder names; the host stitch follows on a worker.
/// The decoded clip in its output form.
enum Clip {
    H264(crate::codec::H264Stream),
    Frames(WanVideo),
}

/// Bail between phases when the job was cancelled (the worker reports any
/// error on a cancel-requested job as cancelled).
fn check_cancel(sink: &dyn ProgressSink) -> Result<(), String> {
    if sink.cancelled() {
        return Err("cancelled".into());
    }
    Ok(())
}

async fn decode_video<S: WeightSource>(
    backend: &Arc<WgpuBackend>,
    src: S,
    vae_cfg: &VideoVaeConfig,
    cfg: &DitConfig,
    video_rows: &[f32],
    (latent_frames, lat_h, lat_w): (usize, usize, usize),
    budget: ResidencyBudget,
) -> Result<DecodedTokens, String>
where
    S::Error: core::fmt::Debug,
{
    let residency = WeightResidency::new(src, budget);
    let handles =
        register_vae(vae_cfg, &residency).map_err(|e| format!("register video vae: {e:?}"))?;
    let pipelines = DecoderPipelines::compile(backend)
        .await
        .map_err(|e| format!("compile video vae pipelines: {e:?}"))?;
    let ws = Workspace::new(Arc::clone(backend), Arc::clone(residency.arbiter()));
    let read = |id: &'static str| read_weight_f32(backend, &residency, id, vae_cfg.latent_channels);
    let mean = read("latents_mean")
        .await
        .map_err(|e| format!("read latents_mean: {e:?}"))?;
    let std = read("latents_std")
        .await
        .map_err(|e| format!("read latents_std: {e:?}"))?;
    tracing::debug!(
        target: thinfer_core::trace::DIAG,
        mean = ?&mean[..4],
        std = ?&std[..4],
        "minimax video latent stats",
    );
    let z = video_rows_to_latents(cfg, video_rows, latent_frames, lat_h, lat_w, &mean, &std);
    let consts = load_consts(vae_cfg, backend, &residency, &handles)
        .await
        .map_err(|e| format!("read register/mask tokens: {e:?}"))?;
    let out = decode_tokens(
        vae_cfg,
        backend,
        &pipelines,
        &residency,
        &ws,
        &handles,
        &consts,
        TilePolicy::default(),
        &z,
        latent_frames,
        lat_h,
        lat_w,
    )
    .await
    .map_err(|e| format!("video vae decode: {e:?}"))?;
    residency.evict_all_and_free(&**backend);
    Ok(out)
}

/// The decoded clip as `[-1, 1]` planar frames, trimmed to the `frames` asked
/// for (the decoder overshoots whenever the count misses a whole chunk).
fn clip_frames(pixels: PixelBlock, frames: usize) -> Result<WanVideo, String> {
    if !pixels.data.iter().all(|v| v.is_finite()) {
        return Err("video decode produced non-finite output".into());
    }
    let (h_px, w_px) = (pixels.height, pixels.width);
    let f_px = pixels.frames.min(frames);
    let planar = pixels.to_signed_planar();
    let plane = pixels.frames * h_px * w_px;
    let keep = f_px * h_px * w_px;
    let video: Vec<f32> = (0..3)
        .flat_map(|c| planar[c * plane..c * plane + keep].iter().copied())
        .collect();
    Ok(WanVideo {
        frames: video,
        num_frames: f_px,
        height: h_px,
        width: w_px,
    })
}

/// The latent grid of one t2va request, as the denoise plan needs it.
#[derive(Clone, Copy)]
struct PlanDims {
    rows: usize,
    row_limit: usize,
    latent_frames: usize,
    lat_h: usize,
    lat_w: usize,
    audio_latents: usize,
}

/// What the conditioner produced for the DiT: the text rows' modality tags
/// (a reference image's vision block tags as video), and each reference
/// image's clean latent as a conditioning block plus its patchified rows.
struct Conditioning {
    text_tags: Vec<i32>,
    conditions: Vec<packing::Condition>,
    cond_rows: Vec<f32>,
}

/// [`run_dit`] for a distill (no turbo adapter), with the request's vault
/// adapters stacked onto `src` when there are any.
#[allow(clippy::too_many_arguments)]
async fn run_dit_users<S: WeightSource>(
    backend: &Arc<WgpuBackend>,
    cfg: &DitConfig,
    src: S,
    users: Vec<Adapter<SafetensorsSource<BytesOpener>>>,
    schedule: JointSchedule,
    d: PlanDims,
    text: &[f32],
    cond: &Conditioning,
    seed: u64,
    budget: ResidencyBudget,
    sink: &dyn ProgressSink,
) -> Result<Latents, String>
where
    S::Error: core::fmt::Debug,
{
    if users.is_empty() {
        return run_dit(
            backend, cfg, src, schedule, d, text, cond, seed, budget, sink, false,
        )
        .await;
    }
    let src = MergedSource::new(src, lora_source(cfg, users, sink).await?);
    run_dit(
        backend, cfg, src, schedule, d, text, cond, seed, budget, sink, false,
    )
    .await
}

/// Phase B, the DiT, over any source carrying the H3 tensor names: the base
/// GGUF with its turbo adapter merged in (`turbo`, required on every block),
/// or a distillation presented under the same names, either with any user
/// adapters stacked in.
#[allow(clippy::too_many_arguments)]
async fn run_dit<S: WeightSource>(
    backend: &Arc<WgpuBackend>,
    cfg: &DitConfig,
    src: S,
    schedule: JointSchedule,
    d: PlanDims,
    text: &[f32],
    cond: &Conditioning,
    seed: u64,
    budget: ResidencyBudget,
    sink: &dyn ProgressSink,
    turbo: bool,
) -> Result<Latents, String>
where
    S::Error: core::fmt::Debug,
{
    let plan = DenoisePlan::build_with::<String>(
        cfg,
        &cond.text_tags,
        d.latent_frames,
        d.lat_h,
        d.lat_w,
        d.audio_latents,
        &cond.conditions,
        schedule,
        d.row_limit,
    )
    .map_err(|e| format!("build denoise plan: {e:?}"))?;
    // Past the backend block's limit the activations need the room the weight
    // LRU would take; the DiT streams block by block either way.
    let budget = if d.rows > MAX_PACKED_ROWS {
        ResidencyBudget {
            vram_bytes: budget.vram_bytes.min(LONG_SEQUENCE_WEIGHT_BUDGET),
            ..budget
        }
    } else {
        budget
    };

    // The modulation table is built over exactly the plan's distinct
    // timesteps, in the plan's order: the block stack indexes it by that
    // position. Host-side, streaming the projections once.
    sink.note("Precomputing the AdaLN modulation table");
    let tables = build_modulation_tables(cfg, &src, &plan.modulation.timesteps)
        .await
        .map_err(|e| format!("build modulation tables: {e:?}"))?;

    let residency = WeightResidency::new(src, budget);
    let mut handles =
        register_dit(cfg, &residency, None).map_err(|e| format!("register dit: {e:?}"))?;
    // The comfy converter reshapes any tensor whose row length is not a
    // multiple of the 256-element superblock, and the upload transpose reads
    // the SHAPE: left alone this scrambles every video row with no NaN to
    // catch it. Registering at the true shape is a no-op for other sources.
    handles.video_patch_proj = register_video_patch_proj(cfg, &residency)
        .map_err(|e| format!("register video_patch_proj: {e:?}"))?;
    let want = cfg.n_layers + cfg.token_refiner_layers;
    if turbo && handles.lora_block_count() != want {
        return Err(format!(
            "turbo adapter covers {} blocks, expected {want}; the checkpoint and the adapter disagree",
            handles.lora_block_count(),
        ));
    }

    let cfgs = dit_cfgs(
        WeightDtype::Quant(QuantKind::Q4_K),
        ActDtype::Bf16,
        COOPMAT_SITES,
    );
    let pipelines = DitPipelines::compile(backend, &cfgs)
        .await
        .map_err(|e| format!("compile dit pipelines: {e:?}"))?;
    let ws = Workspace::new(Arc::clone(backend), Arc::clone(residency.arbiter()));
    let (table, _held) = upload_modulation_tables(backend, &ws, &tables, ActDtype::Bf16)
        .map_err(|e| format!("upload modulation table: {e:?}"))?;
    drop(tables);

    let out = denoise(
        cfg,
        backend,
        &pipelines,
        &residency,
        &ws,
        &handles,
        &table,
        &plan,
        &cond.cond_rows,
        text,
        seed,
        |step, total| {
            sink.stage(Stage::Step {
                i: step as u32,
                n: total as u32,
            })
        },
        Some(&|| sink.cancelled()),
    )
    .await
    .map_err(|e| format!("denoise: {e:?}"))?;
    residency.evict_all_and_free(&**backend);
    Ok(out)
}

/// The request's vault adapters, decrypted to RAM here, at the DiT phase that
/// uses them (never earlier), and dropped with the DiT source.
async fn user_adapters(
    req: &VideoRequest,
) -> Result<Vec<Adapter<SafetensorsSource<BytesOpener>>>, String> {
    if req.lora.is_empty() {
        return Ok(Vec::new());
    }
    #[cfg(feature = "vault")]
    {
        let password = req
            .vault_password
            .as_ref()
            .ok_or("a vault password is required to use adapters")?;
        let vault = crate::vault::Vault::new(&req.vault_dir);
        let mut out = Vec::with_capacity(req.lora.len());
        for l in &req.lora {
            let bytes = vault
                .open(password.expose(), &req.model.to_string(), &l.id)
                .map_err(|e| e.to_string())?;
            let source = SafetensorsSource::open(BytesOpener::new(bytes))
                .await
                .map_err(|e| format!("parse adapter {}: {e:?}", l.id))?;
            out.push(Adapter {
                source,
                strength: l.weight,
                alpha: None,
            });
        }
        Ok(out)
    }
    #[cfg(not(feature = "vault"))]
    Err("this build has no adapter (vault) support".into())
}

/// Stack `adapters` onto the DiT's sites, refusing a set that touches none.
async fn lora_source<L: WeightSource>(
    cfg: &DitConfig,
    adapters: Vec<Adapter<L>>,
    sink: &dyn ProgressSink,
) -> Result<LoraSource<L>, String> {
    let n = adapters.len();
    let src = LoraSource::new(cfg, adapters)
        .await
        .map_err(|e| format!("adapters: {e:?}"))?;
    if src.site_count() == 0 {
        return Err(
            "the adapter(s) match no MiniMax-H3 DiT site (is this a LoRA for a different model?)"
                .into(),
        );
    }
    if !src.skipped().is_empty() {
        tracing::debug!(target: thinfer_core::trace::DIAG, skipped = ?src.skipped(), "adapter modules skipped");
        sink.note(&format!(
            "{} adapter module(s) have no runtime site here and were skipped \
             (AdaLN / boundary projections, or blocks this checkpoint removed)",
            src.skipped().len(),
        ));
    }
    sink.note(&format!(
        "Applying {n} adapter(s) at runtime on {} sites",
        src.site_count()
    ));
    Ok(src)
}

pub(crate) async fn open_mmap(path: &std::path::Path) -> Result<MmapFileOpener, String> {
    MmapFileOpener::new(path)
        .await
        .map_err(|e| format!("open {}: {e}", path.display()))
}

pub(crate) async fn open_st(
    path: &std::path::Path,
) -> Result<SafetensorsSource<MmapFileOpener>, String> {
    let opener = open_mmap(path).await?;
    SafetensorsSource::open(opener)
        .await
        .map_err(|e| format!("parse {}: {e:?}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The manifest must carry every role the driver resolves, or a generate
    /// fails only after the download step has already "succeeded".
    #[test]
    fn driver_roles_are_all_runtime_roles() {
        use thinfer_models::minimax::manifest as mmf;
        for r in [
            role::ENCODER_GGUF,
            role::TOKENIZER,
            role::DIT_GGUF_Q4_K_M,
            role::TURBO_LORA,
            role::VIDEO_VAE,
            role::AUDIO_VAE,
        ] {
            assert!(mmf::MANIFEST.get(r).is_some(), "missing role {r}");
            assert!(
                mmf::RUNTIME_ROLES_T2VA_Q4.contains(&r),
                "{r} is resolved by the driver but is not a runtime role, so the \
                 downloader will not fetch it"
            );
        }
    }
}
