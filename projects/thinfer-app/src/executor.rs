//! The local execution path: load weights, run the model, encode the output.
//! Holds the wgpu backend so a long-running host (thinfer-serve) builds it once
//! and reuses it across jobs; the CLI just builds one per invocation. Callers
//! ensure the request's files are cached first (see [`crate::download`]); `run`
//! assumes they are present and opens them.

use std::path::Path;
use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::format::union::{RenamedSource, UnionSource};
use thinfer_core::residency::WeightResidency;
use thinfer_core::tokenizer::Tokenizer;
use thinfer_models::ideogram4::lora::LoraFoldSource;
use thinfer_models::ideogram4::manifest::role as idrole;
use thinfer_models::ideogram4::pipeline::{
    self as idpipe, GenerationParams as IdParams, Ideogram4Pipeline,
};
use thinfer_models::qwen_image::manifest::role as qrole;
use thinfer_models::qwen_image::pipeline::{self as qpipe, QwenImagePipeline};
use thinfer_models::qwen_image::text_encoder::qwen2vl_gguf_renames;
use thinfer_models::wan::manifest as wanmf;
use thinfer_models::wan::pipeline::{
    self as wanpipe, GenerationParams as WanParams, Shot, VaeChoice, WanModel, WanVariant, WanVideo,
};
use thinfer_models::wan::source::{WanSource, open_longlive_source, open_wan22_source};
use thinfer_models::z_image::manifest::role as zrole;
use thinfer_models::z_image::pipeline::{self as zpipe, GenerationParams as ZParams, ZImageModel};
use thinfer_models::z_image::qwen3_gguf_renames;
use thinfer_models::z_image::source::{GgufOpeners, ZImageSource};
use thinfer_models::z_image::tokenizer::format_qwen3_prompt;
use thinfer_native::MmapFileOpener;
use thinfer_native::tokenizer::HfTokenizer;

use crate::codec;
use crate::config::{BackendConfig, init_backend, random_seed};
use crate::download::{resolve_file, resolve_role};
use crate::model::VideoModelId;
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
        match req.model.kind() {
            crate::model::ImageKind::ZImage => self.run_zimage(req, sink).await,
            crate::model::ImageKind::Ideogram4 => self.run_ideogram4(req, sink).await,
            crate::model::ImageKind::QwenImageEdit => self.run_qwen_image_edit(req, sink).await,
            crate::model::ImageKind::QwenImage => self.run_qwen_image(req, sink).await,
        }
    }

    async fn run_zimage(
        &self,
        req: &ImageRequest,
        sink: &dyn ProgressSink,
    ) -> Result<JobSummary, String> {
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
            model = %req.model, width = req.width, height = req.height,
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

    /// Ideogram-4 + turbotime LoRA. Distinct from Z-Image: three weight files
    /// (Qwen3-VL encoder GGUF, DiT GGUF folded with the LoRA, FLUX.2 VAE) unioned
    /// into one residency, the DiT GGUF wrapped by [`LoraFoldSource`].
    async fn run_ideogram4(
        &self,
        req: &ImageRequest,
        sink: &dyn ProgressSink,
    ) -> Result<JobSummary, String> {
        let manifest = req.model.manifest();
        // Ideogram-4 runs Q8_0 throughout: the LoRA fold dequantizes the base
        // DiT and re-quantizes to Q8_0 (near-lossless). The encoder is Q8_0 too
        // (the quantized Qwen3 GGUFs are mixed-precision and the encoder pipeline
        // probes one dtype uniformly).
        let (dit_role, fold_target) = (idrole::DIT_GGUF_Q8_0, thinfer_core::quant::QuantKind::Q8_0);
        let enc_role = idrole::ENCODER_GGUF_Q8_0;
        let enc = open_mmap(&resolve_role(manifest, enc_role)?).await?;
        let dit = open_mmap(&resolve_role(manifest, dit_role)?).await?;
        let vae = open_mmap(&resolve_role(manifest, idrole::VAE)?).await?;
        let lora = open_mmap(&resolve_role(manifest, idrole::LORA)?).await?;
        let tokenizer = load_tokenizer(&resolve_role(manifest, idrole::TOKENIZER)?).await?;

        // union(encoder-renamed-gguf, lora-folded-dit-gguf, vae-safetensors).
        let enc_src = RenamedSource::with_passthrough(
            GgufSource::open(enc)
                .await
                .map_err(|e| format!("parse encoder gguf: {e:?}"))?,
            qwen3_gguf_renames(),
        );
        let dit_src = LoraFoldSource::new(
            GgufSource::open(dit)
                .await
                .map_err(|e| format!("parse dit gguf: {e:?}"))?,
            SafetensorsSource::open(lora)
                .await
                .map_err(|e| format!("parse lora safetensors: {e:?}"))?,
            fold_target,
        );
        let vae_src = SafetensorsSource::open(vae)
            .await
            .map_err(|e| format!("parse vae safetensors: {e:?}"))?;
        let source = UnionSource::new(UnionSource::new(dit_src, enc_src), vae_src);

        let token_ids = tokenizer
            .encode(&format_qwen3_prompt(&req.prompt), false)
            .map_err(|e| format!("tokenize: {e:?}"))?;
        if token_ids.is_empty() {
            return Err("empty prompt produced no tokens".into());
        }

        let seed = req.seed.unwrap_or_else(random_seed);
        tracing::info!(
            target: thinfer_core::trace::DIAG,
            model = %req.model, width = req.width, height = req.height,
            steps = req.steps, seed, tokens = token_ids.len(), "ideogram4 generate start",
        );
        sink.note(&format!(
            "Generating {}x{} image, {} steps, seed {} ({})",
            req.width, req.height, req.steps, seed, req.model,
        ));

        let progress = |ev: idpipe::ProgressEvent| sink.stage(map_ideo(ev));
        let params = IdParams::new(token_ids, req.height, req.width, req.steps, seed);
        let residency = WeightResidency::new(source, req.budget);
        let pipeline = {
            let _s = tracing::info_span!("model_load").entered();
            Ideogram4Pipeline::load(self.backend.clone(), residency, req.i8_matmul)
                .await
                .map_err(|e| format!("model load: {e:?}"))?
        };
        let png = pipeline
            .generate(&params, None, Some(&progress))
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

    /// Qwen-Image-Edit-Rapid (image->image, CFG-free 4-step). Four sources union
    /// into one residency: DiT GGUF (Q8_0, 1:1 keys), the Qwen2.5-VL encoder GGUF
    /// (renamed), the f16 mmproj (native keys), and the full Qwen-Image KL VAE
    /// (safetensors, encoder+decoder). Host-side input prep (image preprocess +
    /// edit-template tokenization) lives in [`crate::preprocess`].
    async fn run_qwen_image_edit(
        &self,
        req: &ImageRequest,
        sink: &dyn ProgressSink,
    ) -> Result<JobSummary, String> {
        let manifest = req.model.manifest();
        let input_image = req
            .input_image
            .as_ref()
            .ok_or("--input-image is required for qwen-image-edit-rapid")?;

        // --- 4-source union: dit(1:1) + encoder(renamed) + mmproj(native) + vae ---
        let dit = open_mmap(&resolve_role(manifest, qrole::DIT_GGUF_Q8_0)?).await?;
        let enc = open_mmap(&resolve_role(manifest, qrole::ENCODER_GGUF_Q8_0)?).await?;
        let mmproj = open_mmap(&resolve_role(manifest, qrole::MMPROJ_F16)?).await?;
        let vae = open_mmap(&resolve_role(manifest, qrole::VAE)?).await?;
        let tokenizer = load_tokenizer(&resolve_role(manifest, qrole::TOKENIZER)?).await?;

        let dit_src = GgufSource::open(dit)
            .await
            .map_err(|e| format!("parse dit gguf: {e:?}"))?;
        let enc_src = RenamedSource::with_passthrough(
            GgufSource::open(enc)
                .await
                .map_err(|e| format!("parse encoder gguf: {e:?}"))?,
            qwen2vl_gguf_renames(),
        );
        let mmproj_src = GgufSource::open(mmproj)
            .await
            .map_err(|e| format!("parse mmproj gguf: {e:?}"))?;
        let vae_src = SafetensorsSource::open(vae)
            .await
            .map_err(|e| format!("parse vae safetensors: {e:?}"))?;
        let source = UnionSource::new(
            UnionSource::new(UnionSource::new(dit_src, enc_src), mmproj_src),
            vae_src,
        );

        // --- host-side input prep: decode source, preprocess, tokenize ---
        let rgb = {
            let dynimg = image::open(input_image)
                .map_err(|e| format!("decode {}: {e}", input_image.display()))?;
            dynimg.to_rgb8()
        };
        sink.note(&format!(
            "Loaded reference {}x{}",
            rgb.width(),
            rgb.height()
        ));
        let inputs = crate::preprocess::prepare_edit_inputs(&rgb, &req.prompt, &tokenizer)?;

        let seed = req.seed.unwrap_or_else(random_seed);
        tracing::info!(
            target: thinfer_core::trace::DIAG,
            model = %req.model, width = req.width, height = req.height,
            steps = req.steps, seed, tokens = inputs.token_ids.len(),
            image_pad_start = inputs.image_pad_start,
            vit_grid = ?inputs.vit_grid, vae_dims = ?inputs.vae_dims,
            "qwen-image-edit generate start",
        );
        sink.note(&format!(
            "Generating {}x{} image, {} steps, seed {} ({})",
            req.width, req.height, req.steps, seed, req.model,
        ));

        let residency = WeightResidency::new(source, req.budget);
        let pipeline = {
            let _s = tracing::info_span!("model_load").entered();
            // The edit encoder even-pads the sequence (mask layout wants even
            // s_k), so size the rope table to the next even length.
            let max_seq = (inputs.token_ids.len() + 1) & !1;
            QwenImagePipeline::load(
                self.backend.clone(),
                residency,
                max_seq,
                req.i8_matmul,
                true,
            )
            .await
            .map_err(|e| format!("model load: {e:?}"))?
        };
        let progress = |ev: qpipe::ProgressEvent| sink.stage(map_qwen(ev));
        let rgb_out = pipeline
            .generate_edit_rgb(
                &inputs.token_ids,
                inputs.image_pad_start,
                &inputs.vit_pixels,
                inputs.vit_grid,
                &inputs.vae_image,
                inputs.vae_dims,
                req.height,
                req.width,
                req.steps,
                seed,
                Some(&progress),
            )
            .await
            .map_err(|e| format!("generate: {e:?}"))?;
        let png = thinfer_models::z_image::pipeline::encode_png(&rgb_out, req.width, req.height)
            .map_err(|e| format!("encode png: {e}"))?;
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

    /// Qwen-Image-Rapid (text->image, CFG-free 4-step). Same MMDiT as the edit
    /// path but text-only: three sources union into one residency (DiT GGUF
    /// Q8_0, the renamed Qwen2.5-VL encoder GGUF, the Qwen-Image KL VAE), no
    /// mmproj / vision tower. Host-side prep is just the t2i-template tokenize
    /// (see [`crate::preprocess::tokenize_t2i`]).
    async fn run_qwen_image(
        &self,
        req: &ImageRequest,
        sink: &dyn ProgressSink,
    ) -> Result<JobSummary, String> {
        let manifest = req.model.manifest();

        // --- 3-source union: dit(1:1) + encoder(renamed) + vae ---
        let dit = open_mmap(&resolve_role(manifest, qrole::DIT_GGUF_Q8_0)?).await?;
        let enc = open_mmap(&resolve_role(manifest, qrole::ENCODER_GGUF_Q8_0)?).await?;
        let vae = open_mmap(&resolve_role(manifest, qrole::VAE)?).await?;
        let tokenizer = load_tokenizer(&resolve_role(manifest, qrole::TOKENIZER)?).await?;

        let dit_src = GgufSource::open(dit)
            .await
            .map_err(|e| format!("parse dit gguf: {e:?}"))?;
        let enc_src = RenamedSource::with_passthrough(
            GgufSource::open(enc)
                .await
                .map_err(|e| format!("parse encoder gguf: {e:?}"))?,
            qwen2vl_gguf_renames(),
        );
        let vae_src = SafetensorsSource::open(vae)
            .await
            .map_err(|e| format!("parse vae safetensors: {e:?}"))?;
        let source = UnionSource::new(UnionSource::new(dit_src, enc_src), vae_src);

        let token_ids = crate::preprocess::tokenize_t2i(&req.prompt, &tokenizer)?;

        let seed = req.seed.unwrap_or_else(random_seed);
        tracing::info!(
            target: thinfer_core::trace::DIAG,
            model = %req.model, width = req.width, height = req.height,
            steps = req.steps, seed, tokens = token_ids.len(), "qwen-image generate start",
        );
        sink.note(&format!(
            "Generating {}x{} image, {} steps, seed {} ({})",
            req.width, req.height, req.steps, seed, req.model,
        ));

        let residency = WeightResidency::new(source, req.budget);
        let pipeline = {
            let _s = tracing::info_span!("model_load").entered();
            // The encoder even-pads to the next even length; size the rope table
            // for it (+2 headroom, matching the e2e gate).
            let max_seq = token_ids.len() + 2;
            QwenImagePipeline::load(
                self.backend.clone(),
                residency,
                max_seq,
                req.i8_matmul,
                false,
            )
            .await
            .map_err(|e| format!("model load: {e:?}"))?
        };
        let progress = |ev: qpipe::ProgressEvent| sink.stage(map_qwen(ev));
        let rgb_out = pipeline
            .generate_rgb(
                &token_ids,
                req.height,
                req.width,
                req.steps,
                seed,
                Some(&progress),
            )
            .await
            .map_err(|e| format!("generate: {e:?}"))?;
        let png = thinfer_models::z_image::pipeline::encode_png(&rgb_out, req.width, req.height)
            .map_err(|e| format!("encode png: {e}"))?;
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
        // LTX-2.3 is a distinct joint-AV pipeline (Gemma encoder, dual-stream
        // DiT, two VAEs + vocoder); it shares none of the Wan variant/sampler/vae
        // machinery, so it dispatches to its own driver before any Wan lookup.
        if req.model.is_ltx() {
            return crate::ltx::run(&self.backend, req, sink).await;
        }
        // HunyuanVideo 1.5: its own encoder/refiner/DiT/VAE pipeline, no Wan
        // variant/sampler/vae machinery. Dispatch before any Wan lookup; the
        // causal I2V variant has its own driver (chunked AR + image cond).
        if req.model.is_hunyuan_i2v() {
            return crate::hunyuan::run_i2v(&self.backend, req, sink).await;
        }
        if req.model.is_hunyuan() {
            return crate::hunyuan::run(&self.backend, req, sink).await;
        }
        let plan = req.resolve()?;
        for w in &plan.warnings {
            tracing::warn!(target: thinfer_core::trace::DIAG, "{w}");
            sink.note(w);
        }
        let manifest = req.model.manifest();
        let variant = req.model.variant();
        // Wan2.2-A14B has only the full Wan2.1 VAE (no tiny decoder path).
        let vae: VaeChoice = if matches!(req.model, VideoModelId::Wan22T2vA14b) {
            VaeChoice::Full
        } else {
            req.vae.into()
        };

        let mut weight_openers: Vec<MmapFileOpener> =
            Vec::with_capacity(variant.weight_roles.len());
        for role in variant.weight_roles {
            weight_openers.push(open_mmap(&resolve_role(manifest, role)?).await?);
        }
        // The tiny decoder is an extra safetensors shard (disjoint keys), unioned
        // into the same catalog. Only when selected, so the parity path is
        // byte-for-byte the source it always was.
        // `vae` here is the Wan-mapped choice (`req.vae.into()`), so a Hunyuan-only
        // `TinyFt` request has already collapsed to `Tiny` -- this load covers it.
        if vae == VaeChoice::Tiny {
            weight_openers
                .push(open_mmap(&resolve_role(manifest, req.model.wan_tiny_vae_role())?).await?);
        }

        let tokenizer =
            load_tokenizer(&resolve_role(manifest, wanmf::role::TOKENIZER_JSON)?).await?;

        let seed = req.seed.unwrap_or_else(random_seed);
        tracing::info!(
            target: thinfer_core::trace::DIAG,
            model = %req.model, shots = plan.shots.len().max(1),
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
        let cancel = || sink.cancelled();
        let params = WanParams {
            // Single-shot reads this; multi-shot reads `shots` and ignores it
            // (kept = the first shot's caption for logging consistency).
            prompt: req.prompts[0].clone(),
            height: req.height,
            width: req.width,
            num_frames: plan.frames,
            seed,
            sampler: req.sampler.into_engine(req.steps),
            // Unset -> the model default (Some(3) for Wan22, on the long-clip
            // tiled path only); an explicit 0 flows through as full attention.
            // Single source for both CLI and serve so they cannot drift.
            attn_window: req.attn_window.or_else(|| req.model.default_attn_window()),
            // AnyFlow any-step: the user's --steps drives the flow-map schedule.
            // The fixed distill samplers ignore it.
            steps: Some(req.steps),
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
                WanVariant::fastwan_ti2v_5b(),
                &progress,
                &cancel,
            )
            .await?
        } else if matches!(req.model, VideoModelId::Wan22T2vA14b) {
            // Wan2.2-A14B MoE: two folded GGUF experts + the safetensors tail
            // (umT5 reused from the FastWan bundle + the Wan2.1 VAE; both already in
            // `weight_openers` from the variant tail roles).
            let hi_gguf =
                open_mmap(&resolve_role(manifest, variant.gguf_high_role.unwrap())?).await?;
            let lo_gguf =
                open_mmap(&resolve_role(manifest, variant.gguf_low_role.unwrap())?).await?;
            let hi_lora =
                open_mmap(&resolve_role(manifest, variant.lora_high_role.unwrap())?).await?;
            let lo_lora =
                open_mmap(&resolve_role(manifest, variant.lora_low_role.unwrap())?).await?;
            let num_layers = thinfer_models::wan::dit_block::WanDitConfig::wan22_14b().num_layers;
            let source = open_wan22_source(
                hi_gguf,
                hi_lora,
                lo_gguf,
                lo_lora,
                weight_openers,
                num_layers,
            )
            .await
            .map_err(|e| format!("parse Wan2.2 weights: {e:?}"))?;
            self.run_wan(
                source,
                req,
                tokenizer,
                &params,
                &plan.shots,
                vae,
                false,
                WanVariant::wan22_t2v_a14b(),
                &progress,
                &cancel,
            )
            .await?
        } else {
            // GGUF deferred: bringup is safetensors-only (the union path in
            // wan::source is retained for a published FastWan GGUF).
            let source = WanSource::open(weight_openers, None)
                .await
                .map_err(|e| format!("parse weight files: {e:?}"))?;
            let wan_variant = if matches!(req.model, VideoModelId::AnyflowT2v14b) {
                // All-safetensors like FastWan (3 DiT shards lead the opener
                // list); the variant flips the 14B geometry + delta embedder +
                // Q8 block transcode + the any-step denoise.
                WanVariant::anyflow_t2v_14b()
            } else {
                WanVariant::fastwan_ti2v_5b()
            };
            self.run_wan(
                source,
                req,
                tokenizer,
                &params,
                &plan.shots,
                vae,
                false,
                wan_variant,
                &progress,
                &cancel,
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
        variant: WanVariant,
        progress: &dyn Fn(wanpipe::ProgressEvent),
        cancel: &dyn Fn() -> bool,
    ) -> Result<WanVideo, String> {
        let residency = WeightResidency::new(source, req.budget);
        let model = {
            let _s = tracing::info_span!("model_load").entered();
            WanModel::load_variant(
                self.backend.clone(),
                residency,
                tokenizer,
                vae,
                variant,
                None,
                req.i8_matmul,
            )
            .await
            .map_err(|e| format!("model load: {e:?}"))?
        };
        // Cooperative cancellation: the denoise loop polls `cancel` between steps
        // and aborts with GenerateError::Cancelled, which the worker maps to a
        // cancelled job state (the AR path is not yet cancel-wired).
        if ar {
            model
                .generate_ar(params, shots, vae, Some(progress))
                .await
                .map_err(|e| format!("generate: {e:?}"))
        } else {
            model
                .generate(params, vae, Some(progress), Some(&cancel))
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

fn map_qwen(ev: qpipe::ProgressEvent) -> Stage {
    match ev {
        qpipe::ProgressEvent::TextEncode => Stage::TextEncode,
        qpipe::ProgressEvent::Step { i, n } => Stage::Step { i, n },
        qpipe::ProgressEvent::VaeDecode => Stage::VaeDecode,
    }
}

fn map_ideo(ev: idpipe::ProgressEvent) -> Stage {
    match ev {
        idpipe::ProgressEvent::TextEncode => Stage::TextEncode,
        idpipe::ProgressEvent::Step { i, n } => Stage::Step { i, n },
        idpipe::ProgressEvent::VaeDecode => Stage::VaeDecode,
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
