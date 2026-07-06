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
#[cfg(feature = "vault")]
use thinfer_core::format::safetensors::ShardedSafetensorsSource;
use thinfer_core::format::union::{RenamedSource, UnionSource};
use thinfer_core::residency::WeightResidency;
use thinfer_core::tokenizer::Tokenizer;
use thinfer_core::weight::WeightSource;
#[cfg(feature = "vault")]
use thinfer_models::common::lora as vault_lora;
use thinfer_models::ideogram4::lora::LoraFoldSource;
use thinfer_models::ideogram4::manifest::role as idrole;
use thinfer_models::ideogram4::pipeline::{
    self as idpipe, GenerationParams as IdParams, Ideogram4Pipeline,
};
use thinfer_models::krea::manifest::role as krole;
use thinfer_models::krea::pipeline::{self as kreapipe, KreaPipeline};
use thinfer_models::qwen_image::manifest::role as qrole;
use thinfer_models::qwen_image::pipeline::{self as qpipe, QwenImagePipeline};
use thinfer_models::qwen_image::text_encoder::qwen2vl_gguf_renames;
use thinfer_models::wan::manifest as wanmf;
use thinfer_models::wan::pipeline::{
    self as wanpipe, GenerationParams as WanParams, Shot, VaeChoice, WanModel, WanVariant, WanVideo,
};
use thinfer_models::wan::source::{
    WanSource, open_dreamidv_source, open_longlive_source, open_wan22_source,
};
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
    FaceSwapRequest, ImageRequest, JobRequest, JobSummary, VideoFormat, VideoInput, VideoRequest,
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
            crate::model::ImageKind::Krea2 => self.run_krea(req, sink).await,
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

    /// Krea 2 Turbo t2i: DiT GGUF (1:1 krea2 keys) + Qwen3-VL-4B encoder GGUF
    /// (renamed) + Wan2.1 VAE, unioned into one residency; shared Qwen-Image t2i
    /// tokenize template; 8-step turbo CFG-free denoise. Mirrors `run_qwen_image`.
    /// Krea 2 Turbo t2i: DiT GGUF (1:1 krea2 keys) + Qwen3-VL-4B encoder GGUF
    /// (renamed) + Wan2.1 VAE, unioned into one residency; shared Qwen-Image t2i
    /// tokenize template; 8-step turbo CFG-free denoise. Mirrors `run_qwen_image`.
    async fn run_krea(
        &self,
        req: &ImageRequest,
        sink: &dyn ProgressSink,
    ) -> Result<JobSummary, String> {
        let manifest = req.model.manifest();
        let dit = open_mmap(&resolve_role(manifest, krole::DIT_GGUF_Q8_0)?).await?;
        let enc = open_mmap(&resolve_role(manifest, krole::ENCODER_GGUF)?).await?;
        let vae = open_mmap(&resolve_role(manifest, krole::VAE)?).await?;
        let tokenizer = load_tokenizer(&resolve_role(manifest, krole::TOKENIZER)?).await?;

        let dit_src = GgufSource::open(dit)
            .await
            .map_err(|e| format!("parse dit gguf: {e:?}"))?;
        let enc_src = RenamedSource::with_passthrough(
            GgufSource::open(enc)
                .await
                .map_err(|e| format!("parse encoder gguf: {e:?}"))?,
            qwen3_gguf_renames(),
        );
        let vae_src = SafetensorsSource::open(vae)
            .await
            .map_err(|e| format!("parse vae safetensors: {e:?}"))?;

        let token_ids = crate::preprocess::tokenize_t2i(&req.prompt, &tokenizer)?;
        let seed = req.seed.unwrap_or_else(random_seed);
        tracing::info!(
            target: thinfer_core::trace::DIAG,
            model = %req.model, width = req.width, height = req.height,
            steps = req.steps, seed, tokens = token_ids.len(),
            adapters = req.lora.len(), "krea generate start",
        );
        sink.note(&format!(
            "Generating {}x{} image, {} steps, seed {} ({})",
            req.width, req.height, req.steps, seed, req.model,
        ));

        // Fold any request-time user adapters into the DiT (encrypted vault).
        // The folded source is a distinct type from the plain GGUF, so each arm
        // hands its concrete source to the shared generic finish.
        #[cfg(feature = "vault")]
        if !req.lora.is_empty() {
            let folded = self.krea_fold_adapters(dit_src, req, sink).await?;
            let source = UnionSource::new(UnionSource::new(folded, enc_src), vae_src);
            return self
                .krea_generate(source, req, &token_ids, seed, sink)
                .await;
        }
        #[cfg(not(feature = "vault"))]
        if !req.lora.is_empty() {
            return Err("this build has no adapter (vault) support".into());
        }
        let source = UnionSource::new(UnionSource::new(dit_src, enc_src), vae_src);
        self.krea_generate(source, req, &token_ids, seed, sink)
            .await
    }

    /// Load the Krea pipeline over `source` (plain DiT or an adapter-folded DiT)
    /// and run the 8-step turbo denoise -> PNG. Generic over the source so the
    /// fold path and the base path share one body.
    async fn krea_generate<S: WeightSource>(
        &self,
        source: S,
        req: &ImageRequest,
        token_ids: &[u32],
        seed: u64,
        sink: &dyn ProgressSink,
    ) -> Result<JobSummary, String> {
        let residency = WeightResidency::new(source, req.budget);
        let pipeline = {
            let _s = tracing::info_span!("model_load").entered();
            let max_seq = token_ids.len() + 2;
            KreaPipeline::load(
                self.backend.clone(),
                residency,
                max_seq,
                thinfer_core::quant::QuantKind::Q8_0,
            )
            .await
            .map_err(|e| format!("model load: {e:?}"))?
        };
        let progress = |ev: kreapipe::ProgressEvent| sink.stage(map_krea(ev));
        let rgb_out = pipeline
            .generate_rgb(
                token_ids,
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

    /// Decrypt this request's vault adapters and wrap `base` (the Krea DiT GGUF)
    /// in the generic LoRA fold. Each adapter's plaintext bytes live only for the
    /// fold build; the password is re-derived here and never logged. `validate`
    /// has already guaranteed a password is present.
    #[cfg(feature = "vault")]
    async fn krea_fold_adapters<B: WeightSource>(
        &self,
        base: B,
        req: &ImageRequest,
        sink: &dyn ProgressSink,
    ) -> Result<
        vault_lora::LoraFoldSource<B, RenamedSource<ShardedSafetensorsSource<BytesOpener>>>,
        String,
    > {
        let password = req
            .vault_password
            .as_ref()
            .ok_or("a vault password is required to use adapters")?;
        let vault = crate::vault::Vault::new(&req.vault_dir);
        let model_id = req.model.to_string();
        // Community Krea LoRAs use diffusers module names; map them onto the base
        // DiT's sd.cpp keys so the generic fold discovers their sites (else 0).
        let lora_renames = thinfer_models::krea::lora::lora_key_renames();
        let mut stacks = Vec::with_capacity(req.lora.len());
        for l in &req.lora {
            let bytes = vault
                .open(password.expose(), &model_id, &l.id)
                .map_err(|e| e.to_string())?;
            let raw = ShardedSafetensorsSource::open(vec![BytesOpener::new(bytes)])
                .await
                .map_err(|e| format!("parse adapter {}: {e:?}", l.id))?;
            let adapter = RenamedSource::with_passthrough(raw, lora_renames.clone());
            let specs = vault_lora::discover_specs(&base, &adapter)
                .await
                .map_err(|e| format!("discover adapter {} sites: {e:?}", l.id))?;
            stacks.push((adapter, l.weight, specs));
        }
        let sites: usize = stacks.iter().map(|(_, _, s)| s.len()).sum();
        if sites == 0 {
            return Err(format!(
                "adapter(s) folded 0 sites into {} -- the LoRA keys match no {} DiT \
                 tensor (is this a LoRA for a different model?)",
                req.model, req.model,
            ));
        }
        sink.note(&format!(
            "Folding {} adapter(s) into {} ({} sites); first-touch quantizes each site",
            stacks.len(),
            req.model,
            sites,
        ));
        vault_lora::LoraFoldSource::new(base, stacks)
            .map_err(|e| format!("build adapter fold: {e}"))
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
        // DreamID-V video face-swap: its own wan::dreamidv pipeline (target video +
        // live DWPose mask + source-face image, image-CFG denoise), no Wan
        // variant/sampler/prompt machinery. Dispatch before any Wan lookup.
        if req.model.is_dreamidv() {
            return self.run_dreamidv(req, sink).await;
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
                WanVariant::longlive_2_0_5b(),
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

        let source = codec::load_image_bytes(&req.source_image.0)?;
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

        // Materialize the target video in RAM (decrypting an encrypted spill with
        // its RAM-held ephemeral key). The plaintext lives only for this decode.
        let video_bytes = resolve_video_bytes(&req.input_video)?;
        let (w, h, fps, n) = codec::swap_video_streaming(
            &swapper,
            &embedding,
            video_bytes.as_ref(),
            &req.output,
            sink,
        )
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

    /// DreamID-V-Wan-1.3B-Faster diffusion video face-swap. Decodes the target
    /// video (RAM), generates the DWPose face-mask clip live, loads the source
    /// face image, and runs the `wan::dreamidv` pipeline (VAE-encode the three
    /// inputs -> 48-ch image-CFG denoise -> VAE-decode) -> MP4. The DiT `.pth` +
    /// Wan2.1 VAE + the two pose ONNX nets all download as needed (the DiT is read
    /// directly, no offline conversion); the baked umT5 context is an in-tree
    /// constant ([`baked_context`]).
    async fn run_dreamidv(
        &self,
        req: &VideoRequest,
        sink: &dyn ProgressSink,
    ) -> Result<JobSummary, String> {
        use thinfer_models::wan::dreamidv::{DreamIdvInputs, DreamIdvPipeline, RgbFrames};

        let plan = req.resolve()?; // validates the source image + target video + dims
        let manifest = req.model.manifest();

        // --- 1. decode the target video into RAM frames ---
        let input_video = req
            .input_video
            .as_ref()
            .ok_or("dreamid-v requires a target video")?;
        let video_bytes = resolve_video_bytes(input_video)?;
        sink.note("Decoding target video");
        let (mut frames, _src_fps) = codec::decode_mp4_frames(video_bytes.as_ref())?;
        drop(video_bytes);

        // Clamp to the requested target on the 4k+1 grid, bounded by the actual
        // decoded length (the pipeline needs `frames == 4k + 1`).
        let cap = largest_4k_plus_1((plan.frames as usize).min(frames.len()));
        if cap == 0 {
            return Err("target video is too short (need at least 1 frame)".into());
        }
        frames.truncate(cap);
        let (fw, fh) = (frames[0].w, frames[0].h);

        // --- 2. source face image (small; RAM only, never on disk) ---
        let source_face = req
            .source_image
            .as_ref()
            .ok_or("dreamid-v requires a source face image")?;
        let source_img = codec::load_image_bytes(&source_face.0)?;
        sink.note(&format!(
            "Loaded source face {}x{}",
            source_img.w, source_img.h
        ));

        // --- 3. live DWPose face-mask clip (yolox + dw-ll_ucoco RTMPose) ---
        let yolox = std::fs::read(resolve_role(manifest, wanmf::role::YOLOX_ONNX)?)
            .map_err(|e| format!("read yolox onnx: {e}"))?;
        let dwpose = std::fs::read(resolve_role(manifest, wanmf::role::DWPOSE_ONNX)?)
            .map_err(|e| format!("read dwpose onnx: {e}"))?;
        sink.note(&format!(
            "Generating face masks for {} frames (DWPose)",
            cap
        ));
        let masks = thinfer_models::faceswap::dwpose::face_mask_video(
            self.backend.clone(),
            &yolox,
            &dwpose,
            &frames,
        )
        .await
        .map_err(|e| format!("face-mask (DWPose): {e:?}"))?;

        // --- 4. flatten to interleaved RGB [n, h, w, 3] the pipeline expects ---
        let mut video_rgb = Vec::with_capacity(cap * fh * fw * 3);
        for f in &frames {
            if (f.w, f.h) == (fw, fh) {
                video_rgb.extend_from_slice(&f.to_rgb8());
            } else {
                video_rgb.extend_from_slice(&f.resize(fw, fh).to_rgb8());
            }
        }
        let mut mask_rgb = Vec::with_capacity(cap * fh * fw * 3);
        for m in &masks {
            if (m.w, m.h) == (fw, fh) {
                mask_rgb.extend_from_slice(&m.to_rgb8());
            } else {
                mask_rgb.extend_from_slice(&m.resize(fw, fh).to_rgb8());
            }
        }
        let src_rgb = source_img.to_rgb8();

        // --- 5. resolve the downloaded DiT `.pth` + Wan2.1 VAE (baked context is
        //         an in-tree model constant, not a download) ---
        let dit_path = resolve_role(manifest, wanmf::role::DIT_DREAMIDV)?;
        let (context, ctx_rows) = thinfer_models::wan::dreamidv::baked_context();
        let vae_path = resolve_role(manifest, wanmf::role::VAE_WAN21)?;
        let dit_opener = open_mmap(&dit_path).await?;
        let vae_opener = open_mmap(&vae_path).await?;
        let num_layers = thinfer_models::wan::dit_block::WanDitConfig::dreamid_v().num_layers;
        let source = open_dreamidv_source(dit_opener, vec![vae_opener], num_layers)
            .await
            .map_err(|e| format!("open dreamidv source: {e:?}"))?;

        let seed = req.seed.unwrap_or_else(random_seed);
        let guide_scale = req
            .guide_scale
            .unwrap_or_else(|| req.model.default_guide_scale());
        let steps = req.steps;
        let target_area = req.width as f64 * req.height as f64;
        tracing::info!(
            target: thinfer_core::trace::DIAG,
            model = %req.model, frames = cap, width = fw, height = fh,
            steps, guide_scale, seed, ram_budget = req.budget.ram_bytes,
            vram_budget = req.budget.vram_bytes, "dreamid-v generate start",
        );
        sink.note(&format!(
            "Face-swapping {cap} frames, {steps} steps, guide {guide_scale}, seed {seed} ({})",
            req.model,
        ));

        // --- 6. load + run the pipeline ---
        let residency = WeightResidency::new(source, req.budget);
        let pipeline = {
            let _s = tracing::info_span!("model_load").entered();
            // i8 DP4A on the DiT is a ~24% speedup but visibly degrades the
            // swapped face (identity is sensitive), so DreamID stays bf16 by
            // default regardless of the generic `i8_matmul` flag. The capability
            // is retained on `DreamIdvPipeline::load` for A/B measurement.
            DreamIdvPipeline::load(self.backend.clone(), residency, &context, ctx_rows, false)
                .await
                .map_err(|e| format!("model load: {e:?}"))?
        };
        let inputs = DreamIdvInputs {
            video: RgbFrames {
                data: &video_rgb,
                frames: cap,
                h: fh,
                w: fw,
            },
            mask: RgbFrames {
                data: &mask_rgb,
                frames: cap,
                h: fh,
                w: fw,
            },
            image: RgbFrames {
                data: &src_rgb,
                frames: 1,
                h: source_img.h,
                w: source_img.w,
            },
            target_area,
            steps,
            guide_scale,
            seed,
        };
        sink.stage(Stage::VaeDecode);
        let out = pipeline
            .generate(&inputs, None)
            .await
            .map_err(|e| format!("generate: {e:?}"))?;

        // --- 7. encode the swapped clip -> MP4 ---
        let video = WanVideo {
            frames: out.frames,
            num_frames: out.num_frames,
            height: out.height,
            width: out.width,
        };
        let fps = plan.fps;
        sink.note("Encoding MP4 (H.264)");
        codec::encode_mp4(&video, fps, &req.output)?;
        tracing::info!(target: thinfer_core::trace::DIAG, path = %req.output.display(), "wrote output");

        Ok(JobSummary {
            output: req.output.clone(),
            width: out.width as u32,
            height: out.height as u32,
            frames: out.num_frames as u32,
            fps: Some(fps),
            seed: Some(seed),
        })
    }
}

/// Largest `4k + 1` not exceeding `n` (the DreamID-V temporal grid), or 0 when
/// `n == 0`.
fn largest_4k_plus_1(n: usize) -> usize {
    if n == 0 { 0 } else { 4 * ((n - 1) / 4) + 1 }
}

/// Materialize a target video's bytes in RAM. A `Ram` upload is borrowed
/// directly; an `Encrypted` spill is read + decrypted with its RAM-held ephemeral
/// key (the plaintext lives only for the decode, and is never logged).
fn resolve_video_bytes(input: &VideoInput) -> Result<std::borrow::Cow<'_, [u8]>, String> {
    match input {
        VideoInput::Ram(b) => Ok(std::borrow::Cow::Borrowed(b)),
        VideoInput::Encrypted { path, key, nonce } => {
            #[cfg(feature = "vault")]
            {
                let ct = std::fs::read(path).map_err(|e| format!("read encrypted upload: {e}"))?;
                let pt = crate::vault::ephemeral_unseal(key, nonce, &ct)
                    .map_err(|e| format!("decrypt upload: {e}"))?;
                Ok(std::borrow::Cow::Owned(pt))
            }
            #[cfg(not(feature = "vault"))]
            {
                let _ = (path, key, nonce);
                Err("encrypted video spill requires the vault feature".into())
            }
        }
    }
}

async fn open_mmap(path: &Path) -> Result<MmapFileOpener, String> {
    MmapFileOpener::new(path)
        .await
        .map_err(|e| format!("open {}: {e}", path.display()))
}

/// A `FileOpener` over decrypted adapter bytes held in RAM: the vault decrypts a
/// blob to a `Vec<u8>`, this wraps it so `ShardedSafetensorsSource` can parse it
/// like a file. Cheap to reopen (an `Arc` clone per per-tensor reader). The
/// plaintext lives only as long as the fold source that reads it.
#[cfg(feature = "vault")]
#[derive(Clone)]
struct BytesOpener(Arc<[u8]>);

#[cfg(feature = "vault")]
impl BytesOpener {
    fn new(bytes: Vec<u8>) -> Self {
        Self(Arc::from(bytes.into_boxed_slice()))
    }
}

#[cfg(feature = "vault")]
impl thinfer_core::weight::FileOpener for BytesOpener {
    type Reader = BytesReader;
    type Error = std::convert::Infallible;
    async fn open(&self) -> Result<BytesReader, Self::Error> {
        Ok(BytesReader {
            bytes: Arc::clone(&self.0),
        })
    }
}

#[cfg(feature = "vault")]
struct BytesReader {
    bytes: Arc<[u8]>,
}

#[cfg(feature = "vault")]
impl thinfer_core::weight::WeightReader for BytesReader {
    type Error = std::convert::Infallible;
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }
    async fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), Self::Error> {
        let off = offset as usize;
        dst.copy_from_slice(&self.bytes[off..off + dst.len()]);
        Ok(())
    }
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

fn map_krea(ev: kreapipe::ProgressEvent) -> Stage {
    match ev {
        kreapipe::ProgressEvent::TextEncode => Stage::TextEncode,
        kreapipe::ProgressEvent::Step { i, n } => Stage::Step { i, n },
        kreapipe::ProgressEvent::VaeDecode => Stage::VaeDecode,
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
