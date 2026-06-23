//! The local execution path: load weights, run the model, encode the output.
//! Holds the wgpu backend so a long-running host (thinfer-serve) builds it once
//! and reuses it across jobs; the CLI just builds one per invocation. Callers
//! ensure the request's files are cached first (see [`crate::download`]); `run`
//! assumes they are present and opens them.

use std::path::Path;
use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::residency::WeightResidency;
use thinfer_models::wan::manifest as wanmf;
use thinfer_models::wan::pipeline::{
    self as wanpipe, GenerationParams as WanParams, Shot, VaeChoice, WanModel, WanVideo,
};
use thinfer_models::wan::source::{WanSource, open_longlive_source};
use thinfer_models::z_image::manifest::role as zrole;
use thinfer_models::z_image::pipeline::{self as zpipe, GenerationParams as ZParams, ZImageModel};
use thinfer_models::z_image::source::{GgufOpeners, ZImageSource};
use thinfer_native::MmapFileOpener;
use thinfer_native::tokenizer::HfTokenizer;

use crate::codec;
use crate::config::{BackendConfig, init_backend, random_seed};
use crate::download::{resolve_file, resolve_role};
use crate::progress::{ProgressSink, Stage};
use crate::request::{
    FaceSwapRequest, ImageRequest, JobRequest, JobSummary, VideoFormat, VideoRequest,
};

/// Runs jobs on this machine's GPU. Construct once; the backend is reused.
pub struct LocalExecutor {
    backend: Arc<WgpuBackend>,
}

impl LocalExecutor {
    pub async fn new(cfg: BackendConfig) -> Result<Self, String> {
        Ok(Self {
            backend: init_backend(cfg).await?,
        })
    }

    /// The wgpu backend, for the caller's end-of-run memory rollup.
    pub fn backend(&self) -> &Arc<WgpuBackend> {
        &self.backend
    }

    /// Execute a request to completion, reporting progress through `sink`.
    pub async fn run(
        &self,
        req: &JobRequest,
        sink: &dyn ProgressSink,
    ) -> Result<JobSummary, String> {
        match req {
            JobRequest::Image(r) => self.run_image(r, sink).await,
            JobRequest::Video(r) => self.run_video(r, sink).await,
            JobRequest::FaceSwap(r) => self.run_faceswap(r, sink).await,
        }
    }

    async fn run_image(
        &self,
        req: &ImageRequest,
        sink: &dyn ProgressSink,
    ) -> Result<JobSummary, String> {
        req.validate()?;
        let manifest = req.model.manifest();
        let variant = req.model.variant();

        let mut weight_openers: Vec<MmapFileOpener> =
            Vec::with_capacity(variant.weight_roles.len());
        for role in variant.weight_roles {
            weight_openers.push(open_mmap(&resolve_role(manifest, role)?).await?);
        }
        let mut open_gguf = Vec::with_capacity(2);
        for role in [variant.dit_gguf_role, variant.te_gguf_role]
            .into_iter()
            .flatten()
        {
            open_gguf.push(open_mmap(&resolve_role(manifest, role)?).await?);
        }
        let gguf_openers = match (open_gguf.pop(), open_gguf.pop()) {
            (Some(te), Some(dit)) => Some(GgufOpeners { dit, te }),
            (None, None) => None,
            _ => return Err("variant must set both gguf roles or neither".into()),
        };
        let source = ZImageSource::open(weight_openers, gguf_openers)
            .await
            .map_err(|e| format!("parse weight files: {e:?}"))?;

        let tokenizer = load_tokenizer(&resolve_role(manifest, zrole::TOKENIZER_JSON)?).await?;

        let seed = req.seed.unwrap_or_else(random_seed);
        tracing::info!(
            target: thinfer_core::trace::DIAG,
            model = %req.model, prompt = %req.prompt, width = req.width, height = req.height,
            steps = req.steps, seed, ram_budget = req.budget.ram_bytes,
            vram_budget = req.budget.vram_bytes, "generate start",
        );
        sink.note(&format!(
            "Generating {}x{} image, {} steps, seed {} ({})",
            req.width, req.height, req.steps, seed, req.model,
        ));

        let progress = |ev: zpipe::ProgressEvent| sink.stage(map_z(ev));
        let params = ZParams {
            prompt: req.prompt.clone(),
            height: req.height,
            width: req.width,
            steps: req.steps,
            seed,
        };
        let residency = WeightResidency::new(source, req.budget);
        let model = {
            let _s = tracing::info_span!("model_load").entered();
            ZImageModel::load(self.backend.clone(), residency, tokenizer)
                .await
                .map_err(|e| format!("model load: {e:?}"))?
        };
        let png = model
            .generate(&params, Some(&progress))
            .await
            .map_err(|e| format!("generate: {e:?}"))?;
        tokio::fs::write(&req.output, &png)
            .await
            .map_err(|e| format!("write {}: {e}", req.output.display()))?;
        tracing::info!(target: thinfer_core::trace::DIAG, path = %req.output.display(), bytes = png.len(), "wrote output");

        Ok(JobSummary {
            output: req.output.clone(),
            width: req.width,
            height: req.height,
            frames: 1,
            fps: None,
            seed: Some(seed),
        })
    }

    async fn run_video(
        &self,
        req: &VideoRequest,
        sink: &dyn ProgressSink,
    ) -> Result<JobSummary, String> {
        let plan = req.resolve()?;
        let manifest = req.model.manifest();
        let variant = req.model.variant();
        let vae: VaeChoice = req.vae.into();

        let mut weight_openers: Vec<MmapFileOpener> =
            Vec::with_capacity(variant.weight_roles.len());
        for role in variant.weight_roles {
            weight_openers.push(open_mmap(&resolve_role(manifest, role)?).await?);
        }
        // The tiny decoder is an extra safetensors shard (disjoint keys), unioned
        // into the same catalog. Only when selected, so the parity path is
        // byte-for-byte the source it always was.
        if vae == VaeChoice::Tiny {
            weight_openers.push(open_mmap(&resolve_role(manifest, wanmf::role::TINY_VAE)?).await?);
        }

        let tokenizer =
            load_tokenizer(&resolve_role(manifest, wanmf::role::TOKENIZER_JSON)?).await?;

        let seed = req.seed.unwrap_or_else(random_seed);
        tracing::info!(
            target: thinfer_core::trace::DIAG,
            model = %req.model, prompts = ?req.prompts, shots = plan.shots.len().max(1),
            width = req.width, height = req.height, frames = plan.frames, fps = plan.fps,
            seed, ram_budget = req.budget.ram_bytes, vram_budget = req.budget.vram_bytes,
            "generate video start",
        );
        sink.note(&format!(
            "Generating {}x{} video, {} frames, {} fps (~{:.1}s), {} shot(s), seed {} ({})",
            req.width,
            req.height,
            plan.frames,
            plan.fps,
            plan.frames as f32 / plan.fps as f32,
            plan.shots.len().max(1),
            seed,
            req.model,
        ));

        let progress = |ev: wanpipe::ProgressEvent| sink.stage(map_wan(ev));
        let params = WanParams {
            // Single-shot reads this; multi-shot reads `shots` and ignores it
            // (kept = the first shot's caption for logging consistency).
            prompt: req.prompts[0].clone(),
            height: req.height,
            width: req.width,
            num_frames: plan.frames,
            seed,
            sampler: req.sampler.into_engine(req.steps),
        };

        let video = if req.model.is_ar() {
            let dit_opener = open_mmap(&resolve_role(manifest, wanmf::role::LONGLIVE_DIT)?).await?;
            let num_layers =
                thinfer_models::wan::dit_block::WanDitConfig::longlive_2_0_5b().num_layers;
            let source = open_longlive_source(dit_opener, weight_openers, num_layers)
                .await
                .map_err(|e| format!("parse LongLive weights: {e:?}"))?;
            self.run_wan(
                source,
                req,
                tokenizer,
                &params,
                &plan.shots,
                vae,
                true,
                &progress,
            )
            .await?
        } else {
            // GGUF deferred: bringup is safetensors-only (the union path in
            // wan::source is retained for a published FastWan GGUF).
            let source = WanSource::open(weight_openers, None)
                .await
                .map_err(|e| format!("parse weight files: {e:?}"))?;
            self.run_wan(
                source,
                req,
                tokenizer,
                &params,
                &plan.shots,
                vae,
                false,
                &progress,
            )
            .await?
        };

        match req.format {
            VideoFormat::Mp4 => {
                sink.note("Encoding MP4 (H.264)");
                codec::encode_mp4(&video, plan.fps, &req.output)?;
            }
            VideoFormat::PngFrames => {
                sink.note("Writing PNG frames");
                codec::write_png_frames(&video, &req.output)?;
            }
        }
        tracing::info!(target: thinfer_core::trace::DIAG, path = %req.output.display(), "wrote output");

        Ok(JobSummary {
            output: req.output.clone(),
            width: video.width as u32,
            height: video.height as u32,
            frames: video.num_frames as u32,
            fps: Some(plan.fps),
            seed: Some(seed),
        })
    }

    /// Load + denoise a Wan model from `source` (FastWan DMD or LongLive AR).
    /// Generic over the weight source so safetensors and `.pt`-backed paths
    /// share the load/generate/error tail.
    #[allow(clippy::too_many_arguments)]
    async fn run_wan<S: thinfer_core::weight::WeightSource>(
        &self,
        source: S,
        req: &VideoRequest,
        tokenizer: HfTokenizer,
        params: &WanParams,
        shots: &[Shot],
        vae: VaeChoice,
        ar: bool,
        progress: &dyn Fn(wanpipe::ProgressEvent),
    ) -> Result<WanVideo, String> {
        let residency = WeightResidency::new(source, req.budget);
        let model = {
            let _s = tracing::info_span!("model_load").entered();
            WanModel::load(
                self.backend.clone(),
                residency,
                tokenizer,
                vae,
                req.i8_matmul,
            )
            .await
            .map_err(|e| format!("model load: {e:?}"))?
        };
        if ar {
            model
                .generate_ar(params, shots, vae, Some(progress))
                .await
                .map_err(|e| format!("generate: {e:?}"))
        } else {
            model
                .generate(params, vae, Some(progress))
                .await
                .map_err(|e| format!("generate: {e:?}"))
        }
    }

    async fn run_faceswap(
        &self,
        req: &FaceSwapRequest,
        sink: &dyn ProgressSink,
    ) -> Result<JobSummary, String> {
        req.validate()?;
        let scrfd = std::fs::read(resolve_file(&crate::model::SCRFD)?)
            .map_err(|e| format!("read scrfd: {e}"))?;
        let arcface = std::fs::read(resolve_file(&crate::model::ARCFACE)?)
            .map_err(|e| format!("read arcface: {e}"))?;
        let hyperswap = std::fs::read(resolve_file(&req.model.file())?)
            .map_err(|e| format!("read hyperswap: {e}"))?;

        let source = codec::load_image(&req.source_image)?;
        sink.note(&format!("Loaded source {}x{}", source.w, source.h));

        tracing::info!(
            target: thinfer_core::trace::DIAG, model = %req.model,
            ram_budget = req.budget.ram_bytes, vram_budget = req.budget.vram_bytes, "face-swap start",
        );
        let swapper = thinfer_models::faceswap::FaceSwapper::load(
            self.backend.clone(),
            &scrfd,
            &arcface,
            &hyperswap,
        )
        .await
        .map_err(|e| format!("load models: {e}"))?;
        let embedding = swapper
            .source_embedding(&source)
            .await
            .map_err(|e| format!("source embedding (no face in source image?): {e}"))?;
        sink.note("Extracted source face embedding");

        let (w, h, fps, n) =
            codec::swap_video_streaming(&swapper, &embedding, &req.input_video, &req.output, sink)
                .await?;
        tracing::info!(target: thinfer_core::trace::DIAG, path = %req.output.display(), "wrote output");

        Ok(JobSummary {
            output: req.output.clone(),
            width: w as u32,
            height: h as u32,
            frames: n as u32,
            fps: Some(fps),
            seed: None,
        })
    }
}

async fn open_mmap(path: &Path) -> Result<MmapFileOpener, String> {
    MmapFileOpener::new(path)
        .await
        .map_err(|e| format!("open {}: {e}", path.display()))
}

async fn load_tokenizer(path: &Path) -> Result<HfTokenizer, String> {
    HfTokenizer::from_path(path)
        .await
        .map_err(|e| format!("tokenizer {}: {e:?}", path.display()))
}

fn map_z(ev: zpipe::ProgressEvent) -> Stage {
    match ev {
        zpipe::ProgressEvent::TextEncode => Stage::TextEncode,
        zpipe::ProgressEvent::Step { i, n } => Stage::Step { i, n },
        zpipe::ProgressEvent::VaeDecode => Stage::VaeDecode,
    }
}

fn map_wan(ev: wanpipe::ProgressEvent) -> Stage {
    match ev {
        wanpipe::ProgressEvent::TextEncode => Stage::TextEncode,
        wanpipe::ProgressEvent::Step { i, n } => Stage::Step { i, n },
        wanpipe::ProgressEvent::ChunkStep {
            chunk,
            num_chunks,
            step,
            num_steps,
        } => Stage::ChunkStep {
            chunk,
            num_chunks,
            step,
            num_steps,
        },
        wanpipe::ProgressEvent::VaeDecode => Stage::VaeDecode,
    }
}
