//! Job request types and their validation/resolution. One `JobRequest` per
//! modality; every front end (CLI flags, serve JSON) builds these and hands
//! them to a [`crate::executor::LocalExecutor`]. Resolution that can fail
//! (output format, frame grid, clip chain) is exposed standalone so a server can
//! reject a bad request at submit time rather than mid-job.

use std::path::PathBuf;

use thinfer_core::manifest::FileRef;
use thinfer_core::policy::ResidencyBudget;

use crate::model::{
    AttnPrecision, EncoderQuant, ImageModelId, RefImageSize, RewriteQuality, SuperRes, SwapModel,
    VaeChoice, VideoModelId, VideoSampler,
};

/// Both image and video models narrow dims by `vae_scale = vae_factor*2 = 16`.
const VAE_SCALE: u32 = 16;

/// Prompt rows a MiniMax-H3 default clip length budgets for (tokens are not
/// known at resolve time). Generous for a typical prompt.
const MINIMAX_TEXT_ROW_RESERVE: usize = 256;

/// Validate a pixel dimension: positive and a multiple of the VAE scale.
pub fn validate_dim(name: &str, v: u32) -> Result<(), String> {
    if v == 0 {
        return Err(format!("--{name} must be > 0"));
    }
    if !v.is_multiple_of(VAE_SCALE) {
        return Err(format!(
            "--{name} must be a multiple of {VAE_SCALE} (got {v})"
        ));
    }
    Ok(())
}

/// Resolve an output format: an explicit choice wins, otherwise infer from the
/// path extension. A missing/unrecognized extension is a hard error (we never
/// silently write the wrong container). `from_ext` gets the lower-cased
/// extension; `known` lists recognized extensions for the failure message.
pub fn resolve_output_format<T: Copy>(
    explicit: Option<T>,
    output: &std::path::Path,
    from_ext: impl Fn(&str) -> Option<T>,
    known: &str,
) -> Result<T, String> {
    if let Some(f) = explicit {
        return Ok(f);
    }
    let ext = output.extension().and_then(|e| e.to_str()).ok_or_else(|| {
        format!(
            "cannot infer output format: {} has no file extension. Pass an explicit format or use a known extension ({known}).",
            output.display(),
        )
    })?;
    from_ext(&ext.to_ascii_lowercase()).ok_or_else(|| {
        format!("cannot infer output format from extension {ext:?}; known: {known}. Pass an explicit format.")
    })
}

/// Image output container. PNG-only today (the only encoder shipped; every
/// parity baseline is PNG).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum ImageFormat {
    Png,
}

impl ImageFormat {
    pub fn from_ext(ext: &str) -> Option<Self> {
        match ext {
            "png" => Some(Self::Png),
            _ => None,
        }
    }
    pub const KNOWN: &'static str = "png";
}

/// Video output container.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum VideoFormat {
    Mp4,
    /// Raw per-frame PNG sequence: the output path is a directory, frames land
    /// as `frame{n:03}.png`. Not inferable from an extension, so it must be set
    /// explicitly. Bypasses the H.264 encode entirely (the codec-free view of
    /// the VAE decode).
    PngFrames,
}

impl VideoFormat {
    pub fn from_ext(ext: &str) -> Option<Self> {
        match ext {
            "mp4" => Some(Self::Mp4),
            _ => None,
        }
    }
    pub const KNOWN: &'static str = "mp4";
}

/// A secret string (a vault password) that never appears in logs. `Debug`
/// redacts it, so a `{:?}` on any request that carries one -- serve access logs,
/// a panic dump -- cannot leak it (stricter than the prompt handling, which
/// relies on callers not printing the struct). The plaintext is reachable only
/// via [`Secret::expose`], used at the single crypto call site.
#[derive(Clone)]
pub struct Secret(String);

impl Secret {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted>")
    }
}

/// A vault adapter to fold into the DiT for one generation: the entry id (from
/// `vault list`) and its blend weight (ComfyUI `LoraLoaderModelOnly` strength).
#[derive(Clone, Debug)]
pub struct LoraRef {
    pub id: String,
    pub weight: f32,
}

/// An uploaded target video (DreamID-V / face-swap). RAM-FIRST: the mp4 bytes
/// ride in memory and decode straight from RAM. A large upload instead spills to
/// an on-disk blob AES-256-GCM sealed under a per-request EPHEMERAL key held only
/// in RAM (see `vault::ephemeral_seal`), decrypted back to RAM on read for
/// decode. The raw plaintext video NEVER lands on disk, and the bytes are never
/// logged (`Debug` shows only the size / a redaction).
#[derive(Clone)]
pub enum VideoInput {
    /// The mp4 container bytes, held in RAM.
    Ram(Vec<u8>),
    /// A spilled upload: ciphertext at `path`, sealed under `key` (RAM only)
    /// with base64 `nonce`. The whole blob is read + decrypted to RAM at decode
    /// time (GCM is not seekable; "decrypt-on-read" = one transient decrypt).
    Encrypted {
        path: PathBuf,
        /// The ephemeral AES-256-GCM key. RAM only; dropped with the request.
        key: [u8; 32],
        /// Base64 GCM nonce.
        nonce: String,
    },
}

impl std::fmt::Debug for VideoInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never render the bytes or the key (no-media-logging): size / path only.
        match self {
            VideoInput::Ram(b) => write!(f, "VideoInput::Ram({} bytes)", b.len()),
            VideoInput::Encrypted { path, .. } => {
                write!(
                    f,
                    "VideoInput::Encrypted({}, <key redacted>)",
                    path.display()
                )
            }
        }
    }
}

/// Image container bytes (PNG/JPEG) held in RAM: a source face (DreamID-V /
/// face-swap), an edit reference (Qwen-Image-Edit), or an i2v first frame
/// (Hunyuan). Small enough to always ride in memory (unlike the large target
/// video), so a decrypted image NEVER lands on disk. Bytes are never logged
/// (`Debug` shows only the size), matching the video's no-media-logging
/// discipline. Decode with `image::load_from_memory`.
#[derive(Clone)]
pub struct ImageBytes(pub Vec<u8>);

impl std::fmt::Debug for ImageBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ImageBytes({} bytes)", self.0.len())
    }
}

impl Drop for ImageBytes {
    fn drop(&mut self) {
        zeroize::Zeroize::zeroize(&mut self.0);
    }
}

impl Drop for VideoInput {
    fn drop(&mut self) {
        match self {
            VideoInput::Ram(b) => zeroize::Zeroize::zeroize(b),
            VideoInput::Encrypted { key, .. } => zeroize::Zeroize::zeroize(key),
        }
    }
}

/// Generate an image from a prompt (Z-Image t2i).
#[derive(Clone)]
pub struct ImageRequest {
    pub model: ImageModelId,
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    /// `None` -> randomized at run time; the resolved value is reported back.
    pub seed: Option<u64>,
    /// DP4A i8 matmul on the DP4A-safe DiT sites (qkv + ffn_up). Default true;
    /// `false` forces the bf16/dequant-once reference path. Ideogram-4 only
    /// (Z-Image ignores it).
    pub i8_matmul: bool,
    /// Run a bf16-source Z-Image DiT (the `zimage-turbo-bf16` variant)
    /// at FULL bf16 precision instead of the default load-time Q8 transcode.
    /// Default false (Q8: ~6x faster matmul, quality == the Q8 GGUF variant);
    /// `true` = max precision, ~6x slower. No effect on the GGUF variants
    /// (already quant) or non-Z-Image models.
    pub full_precision: bool,
    /// Input images, in prompt order: exactly one for the `QwenImageEdit` kind
    /// (the image to edit), up to `MAX_CONDITION_IMAGES` condition images for
    /// Qwen-Image-2.1 (edit / composition; none = text-to-image), rejected
    /// for the t2i kinds. Held in RAM ([`ImageBytes`]); a decrypted image
    /// never touches disk.
    pub input_images: Vec<ImageBytes>,
    /// User adapters to fold into the DiT for this generation, in fold order
    /// (each id is a vault entry for THIS model). Empty = the plain base model.
    /// Only models where [`ImageModelId::supports_adapters`] is true accept a
    /// non-empty list.
    pub lora: Vec<LoraRef>,
    /// The vault password, required when `lora` is non-empty. Re-derives the key
    /// for this request only; never persisted, never logged (see [`Secret`]).
    pub vault_password: Option<Secret>,
    /// Where the encrypted adapter vault lives (shared by CLI + serve). Only read
    /// when `lora` is non-empty.
    pub vault_dir: PathBuf,
    pub budget: ResidencyBudget,
    pub output: PathBuf,
    pub format: ImageFormat,
}

impl ImageRequest {
    /// Files this request needs in the HF cache.
    pub fn required_files(&self) -> Vec<FileRef> {
        match self.model.kind() {
            crate::model::ImageKind::ZImage => {
                self.model.variant().files().map(|(_, f)| *f).collect()
            }
            // Ideogram-4 + Qwen-Image(-Edit) source by role (encoder + DiT GGUFs,
            // VAE, mmproj/LoRA, tokenizer), not via the Z-Image variant registry.
            crate::model::ImageKind::Ideogram4
            | crate::model::ImageKind::QwenImageEdit
            | crate::model::ImageKind::QwenImage
            | crate::model::ImageKind::Krea2
            | crate::model::ImageKind::QwenImage21 => {
                let m = self.model.manifest();
                self.model
                    .required_roles()
                    .iter()
                    .map(|r| *m.get(r).expect("image role in manifest"))
                    .collect()
            }
        }
    }

    /// Validate everything that can fail before any GPU/network work.
    pub fn validate(&self) -> Result<(), String> {
        validate_dim("height", self.height)?;
        validate_dim("width", self.width)?;
        if self.steps == 0 {
            return Err("--steps must be > 0".into());
        }
        let n = self.input_images.len();
        match self.model.kind() {
            crate::model::ImageKind::QwenImageEdit => {
                if n != 1 {
                    return Err(format!(
                        "{} edits exactly one --input-image (got {n})",
                        self.model
                    ));
                }
            }
            // Optional: an edit / composition when present, text-to-image when not.
            crate::model::ImageKind::QwenImage21 => {
                let max = thinfer_models::qwen_image21::config::MAX_CONDITION_IMAGES;
                if n > max {
                    return Err(format!(
                        "{} takes at most {max} input images (got {n})",
                        self.model
                    ));
                }
            }
            _ => {
                if n > 0 {
                    return Err(format!("--input-image is not supported by {}", self.model));
                }
            }
        }
        if !self.lora.is_empty() {
            if !self.model.supports_adapters() {
                return Err(format!("{} does not support adapters", self.model));
            }
            if self.vault_password.is_none() {
                return Err("a vault password is required to use adapters".into());
            }
        }
        Ok(())
    }
}

/// Generate a video from one or more prompts (t2v; several prompts = chained
/// FastH3 / MiniMax-H3 clips).
#[derive(Clone)]
pub struct VideoRequest {
    pub model: VideoModelId,
    pub prompts: Vec<String>,
    pub width: u32,
    pub height: u32,
    /// Verbatim frame counts. Mutually exclusive with `durations` (the caller
    /// enforces, as clap does); at most one of the two is non-empty.
    pub frames: Vec<u32>,
    pub durations: Vec<f32>,
    /// `None` -> the model's preferred fps.
    pub fps: Option<u32>,
    pub seed: Option<u64>,
    /// Hunyuan i2v / ltx2-rapid i2v first-frame conditioning image. Held in RAM
    /// ([`ImageBytes`]); a decrypted image never touches disk.
    pub input_image: Option<ImageBytes>,
    /// LTX native-I2V frame-0 conditioning strength (`0.0..=1.0`; default 1.0).
    /// `1.0` fully locks the encoded input frame through the denoise (the frame-0
    /// anchor); lower values let it drift toward the model's own frame-0. Only
    /// used when `input_image` is set on an LTX i2v model; ignored otherwise.
    pub strength: f32,
    /// FastH3-Omni only: reference images (PNG/JPEG, at most 9) the prompt can
    /// name as `<Picture 1>`, `<Picture 2>`, ... in order. Held in RAM
    /// ([`ImageBytes`]); a decrypted image never touches disk.
    pub ref_images: Vec<ImageBytes>,
    /// FastH3-Omni only: how each reference image is sized (rejected for every
    /// other model when not the default).
    pub ref_image_size: RefImageSize,
    /// FastWan denoise sampler (ignored on the AR path).
    pub sampler: VideoSampler,
    /// UniPC denoise step count (1..=`VIDEO_MAX_STEPS`); DMD ignores it.
    pub steps: u32,
    /// Temporal self-attention window radius in LATENT frames. `Some(W)`
    /// restricts DiT self-attention to keys within `±W` latent frames (trades
    /// long-range temporal coherence for breaking the O(frames^2) attention cost
    /// on long clips); `None` (default) runs full self-attention. Honored only on
    /// the activation-tiled long-clip path; short clips run full attention.
    pub attn_window: Option<u32>,
    /// MiniMax-H3 family attention precision (rejected for every other model).
    pub attn: AttnPrecision,
    pub vae: VaeChoice,
    /// LTX-2.3 text-encoder quantization (Q8 baseline or Q4_K_M fast). Applies to
    /// every LTX model (shared Gemma encoder); ignored by Wan.
    pub encoder: EncoderQuant,
    pub i8_matmul: bool,
    /// Decode + mux an audio track (LTX joint-AV only; ignored by the silent Wan
    /// models). `false` skips the audio VAE + vocoder tail and writes a
    /// video-only MP4 -- faster, and the choice users want when they only need
    /// the video.
    pub audio: bool,
    /// LTX-2.3 distilled only: opt in to the 2x spatial-upscale refine path
    /// (denoise at half res -> latent upscale x2 -> renoise + refine at full
    /// res). Default `false` = single-stage denoise directly at the target
    /// `width`x`height`, which stays in-distribution at low res and skips the
    /// upscaler model swap. Two-stage is only the cheaper route to HIGH res.
    /// Ignored by the Wan models.
    pub upscale: bool,
    /// MiniMax-H3 / FastH3: RelayVSR super-resolution of the finished video by
    /// this factor (every frame, before encoding). `None` = native res.
    pub super_res: Option<SuperRes>,
    /// LTX joint-AV: opt in to a finer denoise schedule for better audio. The
    /// distilled 8-step sigma table takes very coarse late steps, where the
    /// delicate audio residual under-resolves and drifts to gibberish; this
    /// subdivides the coarse jumps into more Euler substeps (keeping the
    /// distilled anchors exactly). Every LTX path honors it: the single-stage
    /// schedule (ltx2-rapid always; the 22B line when `upscale` is off) and BOTH
    /// stage schedules on the two-stage 22B line, 8+3 -> 16+11 steps (~2.5x
    /// denoise). Wan ignores it. Default `false`.
    pub audio_hq: bool,
    /// MiniMax-H3 / FastH3: run the VAE encoder's convolutions (reference
    /// images, first frame) in f32, the exact path. By default they run in
    /// bf16 on tensor cores (the precision the reference runs that encoder
    /// in, ~3x faster), which needs the CUDA accelerator (f32 without it).
    pub vae_encode_f32: bool,
    /// MiniMax-H3 GGUF models (minimax-h3, minimax-h3-turbo(-ref),
    /// hyperflow-flash(-ref)): run every block GEMM in bf16 with unfolded
    /// adapters, the exact path. By default the sites run as 8-bit tensor-core
    /// GEMMs (INT8 qkv/fc1, FP8 out/fc2) and adapters fold into the bf16
    /// weights (a different but equally good sample, ~30% faster steps; needs
    /// the CUDA accelerator, bf16 without it). Default `false`.
    pub bf16_gemm: bool,
    /// HunyuanVideo 1.5 only: expand a short prompt into a detailed, structured
    /// caption via the rewrite endpoint before encoding (the model is trained on
    /// long captions; short raw prompts are out-of-distribution). Default `true`;
    /// gracefully falls back to the original prompt if the endpoint is
    /// unreachable. Other models ignore it.
    pub rewrite: bool,
    /// Which rewriter model to use when `rewrite` is set (HunyuanVideo 1.5 only):
    /// `Fast` = the ~2.5GB Qwen3-VL-4B (default, budget-honest), `Full` = the
    /// ~5.85GB Qwen3-VL-8B. Ignored when `rewrite` is false or by other models.
    pub rewrite_quality: RewriteQuality,
    /// User adapters (vault LoRAs) to fold into the DiT for this generation, in
    /// fold order (each id is a vault entry for THIS model). Empty = the plain
    /// base model. Only models where [`VideoModelId::supports_adapters`] is true
    /// accept a non-empty list.
    pub lora: Vec<LoraRef>,
    /// The vault password, required when `lora` is non-empty. Re-derives the key
    /// for this request only; never persisted, never logged (see [`Secret`]).
    pub vault_password: Option<Secret>,
    /// Where the encrypted adapter vault lives (shared by CLI + serve). Only read
    /// when `lora` is non-empty.
    pub vault_dir: PathBuf,
    pub budget: ResidencyBudget,
    pub output: PathBuf,
    pub format: VideoFormat,
}

/// The resolved video plan: total frame count, the clip chain, and the
/// effective fps.
#[derive(Clone, Debug)]
pub struct VideoPlan {
    pub frames: u32,
    /// MiniMax-H3 / FastH3 past one clip: the chained clips in order (empty =
    /// one clip of `frames`).
    pub chain: Vec<ChainSegment>,
    pub fps: u32,
    /// Non-fatal notices the user should see (e.g. a requested duration capped
    /// to the per-resolution VRAM envelope). Surfaced via the progress sink at
    /// generate time so a silent cap is never silent.
    pub warnings: Vec<String>,
}

/// One clip of a chained MiniMax-H3 / FastH3 video. Every clip after the
/// first starts from the previous one's last frame (FL2VA first-frame
/// conditioning, as FastVideo's DreamVerse chains FastH3 segments), repeats
/// it, and the output drops the repeat (and its audio).
#[derive(Clone, Debug)]
pub struct ChainSegment {
    pub prompt: String,
    pub frames: u32,
}

impl VideoRequest {
    pub fn required_files(&self) -> Result<Vec<FileRef>, String> {
        // MiniMax-H3 sources by role (conditioner + tokenizer + DiT + the
        // required turbo adapter + both VAEs), not via the Wan variant registry.
        if self.model.is_minimax() {
            let m = self.model.manifest();
            // The opt-in NVFP4 decoder rides with (not instead of) the fp16 VAE,
            // which still supplies the latent stats.
            let decoder = self.vae.minimax_nvfp4().map(|(r, _)| r);
            // A keyframe (first frame, or a chained clip's handoff) also goes
            // through the vision tower.
            let vision = (self.minimax_keyframes()
                && !self
                    .model
                    .minimax_roles()
                    .contains(&thinfer_models::minimax::manifest::role::VISION))
            .then_some(thinfer_models::minimax::manifest::role::VISION);
            let mut files = self
                .model
                .minimax_roles()
                .iter()
                .copied()
                .chain(decoder)
                .chain(vision)
                .map(|r| {
                    m.get(r)
                        .copied()
                        .ok_or_else(|| format!("MiniMax-H3 manifest missing role {r}"))
                })
                .collect::<Result<Vec<_>, String>>()?;
            if let Some(sr) = self.super_res {
                use thinfer_models::wan::relayvsr::manifest as vsr;
                let (dit, flash) = vsr::roles(sr.factor());
                for r in [dit, flash] {
                    files.push(*vsr::MANIFEST.get(r).ok_or("RelayVSR manifest role")?);
                }
            }
            return Ok(files);
        }
        // LTX sources by role (joint-AV: encoder + tokenizer + connector + DiT +
        // both VAEs + upscaler), not via the Wan variant registry.
        if self.model.is_ltx() {
            let m = self.model.manifest();
            // Resolve the selected text encoder (Q8_0 baseline or Q4_K_M) in place
            // of the role list's default, so the chosen file is fetched.
            let enc_role = self.ltx_encoder_role();
            let mut files: Vec<FileRef> = self
                .model
                .ltx_runtime_roles()
                .iter()
                .map(|r| {
                    let r = if *r == thinfer_models::ltx::manifest::role::ENCODER_GGUF {
                        enc_role
                    } else {
                        r
                    };
                    m.get(r)
                        .copied()
                        .ok_or_else(|| format!("LTX manifest missing role {r}"))
                })
                .collect::<Result<_, _>>()?;
            // Dev folds a distill-LoRA stack into the dev DiT at load; fetch
            // every LoRA in the env-selected stack (default = condsafe alone).
            if self.model.folds_distill_lora() {
                for (role, _strength) in self.model.distill_stack() {
                    files.push(
                        m.get(role)
                            .copied()
                            .ok_or_else(|| format!("dev manifest missing distill role {role}"))?,
                    );
                }
            }
            // ltx2-rapid optional prompt rewriter (the 19B merge needs long
            // captions, like Hunyuan): fetch the selected rewriter GGUF +
            // tokenizer only when rewriting is enabled (opt-in footprint).
            if self.rewrite && self.model.is_ltx_rapid() {
                use thinfer_models::hunyuan::manifest::role as hyrole;
                for r in [self.rewrite_quality.gguf_role(), hyrole::REWRITER_TOKENIZER] {
                    files.push(
                        m.get(r)
                            .copied()
                            .ok_or_else(|| format!("LTX manifest missing rewriter role {r}"))?,
                    );
                }
            }
            return Ok(files);
        }
        // Hunyuan sources by role (DiT + VAE + shared Qwen2.5-VL encoder GGUF +
        // tokenizer), not via the Wan variant registry. The VAE role follows the
        // `--vae` choice so Tiny pulls only the ~5MB TAEHV (not the ~500MB full
        // VAE) -- download footprint is first-class.
        if self.model.is_hunyuan() {
            use thinfer_models::hunyuan::manifest::role as hyrole;
            let m = self.model.manifest();
            let i2v = self.model.is_hunyuan_i2v();
            let vae_role = match self.vae {
                VaeChoice::Tiny => hyrole::TINY_VAE,
                VaeChoice::TinyFt => hyrole::TINY_VAE_FT,
                // NVFP4 is rejected for Hunyuan in `resolve`.
                VaeChoice::Full | VaeChoice::Nvfp4 | VaeChoice::Nvfp4Fast => hyrole::VAE,
            };
            let mut roles = vec![
                hyrole::ENCODER_GGUF_Q8_0,
                if i2v { hyrole::DIT_I2V } else { hyrole::DIT },
                vae_role,
                hyrole::TOKENIZER,
            ];
            if i2v && self.input_image.is_some() {
                // Image conditioning additionally needs the SigLIP vision tower
                // and the FULL VAE (its encoder half conditions the first
                // frame) even when a tiny decoder is selected. A text-only run
                // on the TI2V model needs neither.
                roles.push(hyrole::SIGLIP);
                if vae_role != hyrole::VAE {
                    roles.push(hyrole::VAE);
                }
            }
            // Prompt rewriter is opt-in: only pull the SELECTED size's GGUF +
            // tokenizer when the request enables rewriting (download footprint is
            // first-class -- a rewrite-off run fetches neither, and Fast pulls the
            // ~2.5GB 4B rather than the ~5.85GB 8B). On the TI2V model the
            // rewriter runs only for TEXT-ONLY requests (the text-only rewriter
            // cannot see a conditioning image).
            if self.rewrite && (!i2v || self.input_image.is_none()) {
                roles.push(self.rewrite_quality.gguf_role());
                roles.push(hyrole::REWRITER_TOKENIZER);
            }
            return roles
                .iter()
                .map(|r| {
                    m.get(r)
                        .copied()
                        .ok_or_else(|| format!("Hunyuan manifest missing role {r}"))
                })
                .collect();
        }
        let mut files: Vec<FileRef> = self.model.variant().files().map(|(_, f)| *f).collect();
        // `TinyFt` (Hunyuan-only) degrades to Wan's tiny decoder here. The role
        // is per-model: the tiny decoder must match the variant's VAE latent
        // space (taew2_1 for the z16 AnyFlow, lighttaew2_2 for the z48 line).
        if matches!(self.vae, VaeChoice::Tiny | VaeChoice::TinyFt) {
            files.push(
                *self
                    .model
                    .manifest()
                    .get(self.model.wan_tiny_vae_role())
                    .ok_or("manifest missing tiny VAE role")?,
            );
        }
        Ok(files)
    }

    /// LTX text-encoder GGUF role for this request: the `encoder` choice, or
    /// Q4 when `THINFER_LTX_ENCODER=q4_k_m` is set (a power-user/test override that
    /// forces Q4 even on the default request). LTX-only.
    pub fn ltx_encoder_role(&self) -> &'static str {
        let env_q4 = matches!(
            std::env::var("THINFER_LTX_ENCODER").ok().as_deref(),
            Some("q4_k_m") | Some("q4")
        );
        let encoder = if env_q4 {
            EncoderQuant::Q4
        } else {
            self.encoder
        };
        self.model.ltx_encoder_role(encoder)
    }

    /// Validate dims + resolve fps and the shot plan. Fails fast on every user
    /// error so a server can 400 at submit.
    pub fn resolve(&self) -> Result<VideoPlan, String> {
        if self.super_res.is_some() && !self.model.is_minimax() {
            return Err(format!(
                "--super-res applies to the MiniMax-H3 / FastH3 models; {} does not take it",
                self.model
            ));
        }
        if self.attn != AttnPrecision::Bf16 && !self.model.is_minimax() {
            return Err(format!(
                "--attn {} applies to the MiniMax-H3 family's attention; {} does not take it",
                self.attn, self.model
            ));
        }
        if self.bf16_gemm && !self.model.is_minimax_gguf() {
            return Err(format!(
                "--bf16-gemm applies to the MiniMax-H3 GGUF models (minimax-h3, minimax-h3-turbo, \
                 minimax-h3-turbo-ref, hyperflow-flash, hyperflow-flash-ref); {} does not take it",
                self.model
            ));
        }
        if self.vae_encode_f32 && !self.model.is_minimax() {
            return Err(format!(
                "--vae-encode-f32 applies to the MiniMax-H3 / FastH3 reference encode; {} does not take it",
                self.model
            ));
        }
        if self.vae.minimax_nvfp4().is_some() && !self.model.is_minimax() {
            return Err(format!(
                "--vae {} is a MiniMax-H3 / FastH3 decoder; {} does not take it",
                self.vae, self.model
            ));
        }
        // First-frame image conditioning exists only on the causal Hunyuan TI2V
        // model, where it is OPTIONAL: with an image the run is I2V; without it
        // the model runs text-only (the upstream `mask_type="t2v"` shape --
        // probe-validated to produce coherent prompt-following video despite the
        // i2v-trained checkpoint). Every other model rejects it.
        if !(self.model.is_hunyuan_i2v() || self.model.is_ltx_i2v() || self.model.minimax_fl2va())
            && self.input_image.is_some()
        {
            return Err(format!(
                "--input-image is only supported by hunyuan-video-1.5-ti2v, ltx2-rapid, the FL2VA \
                 MiniMax line (fasth3-trim, fasth3-v2, minimax-h3, hyperflow-flash) (got {})",
                self.model
            ));
        }
        if self.prompts.is_empty() {
            return Err("at least one prompt is required".into());
        }
        if !self.lora.is_empty() {
            if !self.model.supports_adapters() {
                return Err(format!("{} does not support adapters", self.model));
            }
            if self.vault_password.is_none() {
                return Err("a vault password is required to use adapters".into());
            }
        }
        let fps = self.fps.unwrap_or_else(|| self.model.fps());
        if fps == 0 {
            return Err("--fps must be > 0".into());
        }
        // LTX: its own grid (dims /64, frames 8k+1), single prompt only, and the
        // FastWan sampler/vae/shot machinery does not apply.
        if self.model.is_ltx() {
            return self.resolve_ltx(fps);
        }
        // MiniMax-H3: its own grid (dims /32, frames 17n+5), single prompt, and
        // none of the FastWan sampler/vae/shot machinery.
        if self.model.is_minimax() {
            // The row-chunked accelerated block where one exists; the driver
            // re-resolves with the limit of the backend it has.
            return self.resolve_minimax(self.model.minimax_row_limit(true));
        }
        validate_dim("height", self.height)?;
        validate_dim("width", self.width)?;
        if self.steps == 0 {
            return Err("--steps must be > 0".into());
        }
        let mut frames = resolve_frames(
            self.model,
            &self.prompts,
            &self.frames,
            &self.durations,
            fps,
        )?;
        // Per-resolution VRAM activation-envelope cap (Wan2.2-A14B: the 14B DiT
        // runs the full token grid per step, so the safe clip length depends on
        // the dims -- see `VideoModelId::max_frames`). Explicit `--frames` over
        // budget is a hard error (fail fast at submit instead of device-losing
        // mid-denoise); `--duration` and the default cap DOWN with a warning.
        // Uncapped models (FastWan) return `None` -> no-op.
        let mut warnings = Vec::new();
        if let Some(max_frames) = self.model.max_frames(self.width, self.height)
            && frames > max_frames
        {
            if !self.frames.is_empty() {
                return Err(format!(
                    "{} at {}x{} fits at most {max_frames} frames on the 8GB card \
                     (got {frames}); lower --frames, or reduce --width/--height for \
                     a longer clip",
                    self.model, self.width, self.height
                ));
            }
            warnings.push(format!(
                "{}x{} fits at most {max_frames} frames (~{:.1}s) on the 8GB card; \
                 the requested ~{:.1}s was capped. Lower the resolution for a longer clip.",
                self.width,
                self.height,
                max_frames as f32 / fps as f32,
                frames as f32 / fps as f32,
            ));
            frames = max_frames;
        }
        Ok(VideoPlan {
            frames,
            chain: Vec::new(),
            fps,
            warnings,
        })
    }

    /// MiniMax-H3 plan resolution: single-prompt t2va on the H3 grid (dims a
    /// multiple of 32, frames `17n + 5`). Fixed 24 fps: the video and audio rows
    /// share one 40-units-per-second rotary clock and that shared clock IS the
    /// a/v alignment, so fps is not a user knob here. No shots, no sampler/vae
    /// knobs.
    ///
    /// The packed-row ceiling is NOT enforced here: it depends on the tokenized
    /// prompt length, which resolve does not have. The driver checks it before
    /// loading any weights.
    pub fn resolve_minimax(&self, row_limit: usize) -> Result<VideoPlan, String> {
        if self.prompts.len() > 1 && !self.model.minimax_fl2va() {
            return Err(format!(
                "{} takes a single prompt (chained clips need FL2VA: fasth3-trim, fasth3-v2, \
                 minimax-h3)",
                self.model
            ));
        }
        if self.input_image.is_some() && !self.model.minimax_fl2va() {
            return Err(format!(
                "{} takes no --input-image (FL2VA first frames: fasth3-trim, fasth3-v2, minimax-h3)",
                self.model
            ));
        }
        let max_refs = crate::minimax_ref::MAX_REFERENCES;
        if !self.ref_images.is_empty() && !self.model.takes_refs() {
            return Err(format!(
                "{} does not take reference images (fasth3-omni, hyperflow-flash-ref, \
                 minimax-h3-turbo-ref do)",
                self.model
            ));
        }
        if self.ref_image_size != RefImageSize::default() && !self.model.takes_refs() {
            return Err(format!(
                "--ref-image-size applies to reference images; {} does not take them",
                self.model
            ));
        }
        if self.ref_images.len() > max_refs {
            return Err(format!("at most {max_refs} reference images"));
        }
        let mult = self.model.dim_multiple();
        for (name, v) in [("height", self.height), ("width", self.width)] {
            if !v.is_multiple_of(mult) || v == 0 {
                return Err(format!(
                    "--{name} for {} must be a positive multiple of {mult} (got {v})",
                    self.model
                ));
            }
        }
        let fps = crate::minimax::FPS;
        if self.fps.is_some_and(|f| f != fps) {
            return Err(format!(
                "{} renders at a fixed {fps} fps (the audio and video rows share one \
                 rotary clock); --fps is not supported",
                self.model
            ));
        }
        let mut warnings = Vec::new();
        if self.minimax_chains() {
            return self.resolve_minimax_chain(row_limit);
        }
        let requested = match (self.frames.first(), self.durations.first()) {
            (Some(&f), _) => f,
            (None, Some(&d)) => ((d * fps as f32).round() as u32).max(1),
            (None, None) => {
                // A blank length is the model default, shortened to what the
                // canvas fits (the prompt's rows are not known yet: reserve a
                // typical prompt). An explicit length over the cap still fails
                // in the driver, with the exact row count.
                use thinfer_models::minimax::packing;
                let default = self.model.default_frames();
                let fit = packing::max_fitting_frames(
                    &self.model.minimax_dit_config(),
                    MINIMAX_TEXT_ROW_RESERVE,
                    (self.height / 16) as usize,
                    (self.width / 16) as usize,
                    default as usize,
                    row_limit,
                )
                .ok_or_else(|| {
                    format!(
                        "{}x{} is too large for {}: not even the shortest clip fits",
                        self.width, self.height, self.model
                    )
                })? as u32;
                if fit < default {
                    warnings.push(format!(
                        "{}x{} fits at most {fit} frames (~{:.1}s) on this model; \
                         the default clip was shortened.",
                        self.width,
                        self.height,
                        fit as f32 / fps as f32,
                    ));
                }
                fit
            }
        };
        let frames = thinfer_models::minimax::packing::align_num_frames(requested as usize)
            .map_err(|e| format!("frame count {requested}: {e:?}"))? as u32;
        if frames != requested {
            warnings.push(format!(
                "{} renders {frames} frames (~{:.1}s): its clip length grid is 17n+5, \
                 so the requested {requested} was rounded up.",
                self.model,
                frames as f32 / fps as f32,
            ));
        }
        Ok(VideoPlan {
            frames,
            chain: Vec::new(),
            fps,
            warnings,
        })
    }

    /// The frames each `--frames` / `--duration` value asks for (no grid snap).
    fn requested_frames(&self, fps: u32) -> Result<Vec<u32>, String> {
        if self.durations.is_empty() {
            return Ok(self.frames.clone());
        }
        self.durations
            .iter()
            .map(|&d| {
                if !(d.is_finite() && d > 0.0) {
                    return Err(format!(
                        "--duration must be a positive number of seconds (got {d})"
                    ));
                }
                Ok(((d * fps as f32).round() as u32).max(1))
            })
            .collect()
    }

    /// Whether a MiniMax-H3 / FastH3 request runs as chained clips: several
    /// prompts, or one past the trained clip length (FL2VA models only).
    pub fn minimax_chains(&self) -> bool {
        let one_clip = thinfer_models::minimax::packing::max_clip_frames() as u32;
        self.model.minimax_fl2va()
            && (self.prompts.len() > 1
                || self
                    .requested_frames(crate::minimax::FPS)
                    .is_ok_and(|v| v.first().is_some_and(|&f| f > one_clip)))
    }

    /// Whether a MiniMax-H3 / FastH3 request conditions on a keyframe.
    pub fn minimax_keyframes(&self) -> bool {
        self.model.is_minimax() && (self.input_image.is_some() || self.minimax_chains())
    }

    /// The chained plan (see [`ChainSegment`]): one length per prompt, or one
    /// total split evenly across them; a single prompt past one clip splits
    /// into as few equal clips as fit the canvas.
    fn resolve_minimax_chain(&self, row_limit: usize) -> Result<VideoPlan, String> {
        use thinfer_models::minimax::packing;
        let fps = crate::minimax::FPS;
        let mut warnings = Vec::new();
        // Each clip carries its keyframe twice: vision tokens in the prompt and
        // as many latent condition rows.
        let keyframe_rows = 2 * (self.height / 32 * (self.width / 32)) as usize;
        let cap = packing::max_fitting_frames(
            &self.model.minimax_dit_config(),
            MINIMAX_TEXT_ROW_RESERVE + keyframe_rows,
            (self.height / 16) as usize,
            (self.width / 16) as usize,
            packing::max_clip_frames(),
            row_limit,
        )
        .ok_or_else(|| {
            format!(
                "{}x{} is too large for chained {} clips: not even the shortest clip fits \
                 with its keyframe",
                self.width, self.height, self.model
            )
        })? as u32;
        let align = |f: u32| {
            packing::align_num_frames(f as usize)
                .map(|v| v as u32)
                .map_err(|e| format!("frame count {f}: {e:?}"))
        };
        // A clip after the first repeats its keyframe, so it adds `frames - 1`.
        let split = |total: u32, parts: u32| total.saturating_sub(1).div_ceil(parts) + 1;
        let values = self.requested_frames(fps)?;
        let n = self.prompts.len() as u32;
        let lens: Vec<u32> = match (n, values.len()) {
            (1, 1) => {
                let parts = values[0].saturating_sub(1).div_ceil(cap - 1).max(1);
                vec![align(split(values[0], parts))?; parts as usize]
            }
            (_, 0) => vec![align(self.model.default_frames())?; n as usize],
            (_, 1) => vec![align(split(values[0], n))?; n as usize],
            (_, m) if m == n as usize => {
                values.iter().map(|&f| align(f)).collect::<Result<_, _>>()?
            }
            (_, m) => {
                return Err(format!(
                    "expected 1 --duration/--frames value (split evenly) or {n} (one per \
                     --prompt), got {m}"
                ));
            }
        };
        if let Some(&f) = lens.iter().find(|&&f| f > cap) {
            return Err(format!(
                "a {f}-frame clip does not fit: at most {cap} frames (~{:.1}s) per chained clip \
                 at {}x{}",
                cap as f32 / fps as f32,
                self.width,
                self.height
            ));
        }
        let frames = lens[0] + lens[1..].iter().map(|f| f - 1).sum::<u32>();
        let asked: u32 = match values.len() {
            1 => values[0],
            0 => frames,
            _ => values[0] + values[1..].iter().map(|f| f.saturating_sub(1)).sum::<u32>(),
        };
        if frames != asked {
            warnings.push(format!(
                "{} renders {frames} frames (~{:.1}s) as {} chained clips on its 17n+5 grid \
                 (asked {asked}).",
                self.model,
                frames as f32 / fps as f32,
                lens.len(),
            ));
        }
        let chain = lens
            .iter()
            .enumerate()
            .map(|(i, &frames)| ChainSegment {
                prompt: self.prompts[if n == 1 { 0 } else { i }].clone(),
                frames,
            })
            .collect();
        Ok(VideoPlan {
            frames,
            chain,
            fps,
            warnings,
        })
    }

    /// LTX-2.3 plan resolution: single-prompt t2v on the LTX grid (dims a
    /// multiple of 64, frames `8k+1`). No chain, no sampler/vae knobs. Returns
    /// an empty shot list (the single-clip path).
    fn resolve_ltx(&self, fps: u32) -> Result<VideoPlan, String> {
        let mult = self.model.dim_multiple();
        let min = VideoModelId::LTX_MIN_DIM;
        for (name, v) in [("height", self.height), ("width", self.width)] {
            if !v.is_multiple_of(mult) || v < min {
                let (dw, dh) = self.model.video_defaults();
                return Err(format!(
                    "--{name} for {} must be a multiple of {mult} and at least {min} \
                     (got {v}); the default is {dw}x{dh}",
                    self.model
                ));
            }
        }
        if self.prompts.len() != 1 {
            return Err(format!("{} takes a single prompt", self.model));
        }
        if self.input_image.is_some() && !(0.0..=1.0).contains(&self.strength) {
            return Err(format!(
                "--strength must be in 0.0..=1.0 (got {})",
                self.strength
            ));
        }
        if self.frames.len() > 1 || self.durations.len() > 1 {
            return Err(format!("{} takes a single --frames/--duration", self.model));
        }
        // The two-stage stage-2 denoise runs the DiT at full res, so the safe
        // clip length depends on the dims: cap frames to the 8GB envelope (see
        // `ltx_max_frames`). Explicit `--frames` over budget is a hard error
        // (fail fast at submit instead of device-losing mid-denoise); `--duration`
        // (approximate) and the default are capped down to fit.
        let max_frames = self.model.ltx_max_frames(self.width, self.height);
        // The requested length BEFORE the VRAM cap (explicit --frames errors above
        // the cap; --duration and the default cap DOWN silently, so we record the
        // pre-cap target to warn when it is reduced).
        let requested = if let Some(&f) = self.frames.first() {
            self.model.validate_frames(f)?;
            if f > max_frames {
                return Err(format!(
                    "{} at {}x{} fits at most {max_frames} frames on the 8GB card \
                     (got {f}); lower --frames, or reduce --width/--height for a \
                     longer clip",
                    self.model, self.width, self.height
                ));
            }
            f
        } else if let Some(&d) = self.durations.first() {
            if !(d.is_finite() && d > 0.0) {
                return Err(format!("--duration must be positive seconds (got {d})"));
            }
            self.model.snap_frames((d * fps as f32).round() as u32)
        } else {
            self.model.default_frames()
        };
        let frames = requested.min(max_frames);
        let mut warnings = Vec::new();
        if frames < requested {
            warnings.push(format!(
                "{}x{} fits at most {frames} frames (~{:.1}s) on the 8GB card; the \
                 requested ~{:.1}s was capped. For a longer clip lower the resolution \
                 (e.g. 1024x576 fits ~3s, 768x512 the full ~5s).",
                self.width,
                self.height,
                frames as f32 / fps as f32,
                requested as f32 / fps as f32,
            ));
        }
        // Below the adherence floor the model still produces a clean-looking clip
        // that quietly drops requested content, so say so rather than let it pass
        // as a plain "faster" option.
        if !self.model.ltx_dims_hold_adherence(self.width, self.height) {
            let side = VideoModelId::LTX_ADHERENCE_MIN_SIDE;
            warnings.push(format!(
                "{w}x{h} is below the 768x512 prompt-adherence floor (needs both \
                 >={floor} pixels and a short side >={side}; this is {px} px with a \
                 short side of {short}). The clip will still look clean, but the \
                 SCENE is what degrades: backgrounds thin out, then dissolve, then \
                 get replaced by an unrelated setting, while the subject survives. \
                 It also drifts off photoreal toward cartoon. Going lower is a way \
                 to get a LONGER clip, not a materially faster one -- the frame cap \
                 grows as dims shrink, so the default clip lengthens and eats most \
                 of the saving. Prefer 768x512 or above.",
                w = self.width,
                h = self.height,
                floor = VideoModelId::LTX_ADHERENCE_FLOOR_PX,
                px = self.width * self.height,
                short = self.width.min(self.height),
            ));
        }
        Ok(VideoPlan {
            frames,
            chain: Vec::new(),
            fps,
            warnings,
        })
    }
}

/// Compositing/quality knobs for a face-swap run. All default off so the
/// baseline (feather-only paste) is unchanged; each changes the visual output,
/// so it is user-opt-in.
#[derive(Clone, Copy, Debug)]
pub struct FaceSwapOptions {
    /// Intersect the paste mask with HyperSwap's own confidence mask (free).
    pub hyperswap_mask: bool,
    /// Intersect with an XSeg occlusion mask so occluders (hands/hair/glasses)
    /// show the original frame through. Pulls in the XSeg model + one ONNX
    /// forward per face.
    pub occlusion: bool,
    /// Run the GFPGAN face enhancer on the swapped face (restores GAN detail).
    /// Pulls in the GFPGAN model + one ONNX forward per face.
    pub enhance: bool,
    /// Run SCRFD detection every Nth frame, reusing the previous frame's faces in
    /// between (the per-frame HyperSwap still runs). 1 = detect every frame
    /// (default). >1 trades a little tracking accuracy on fast head motion for
    /// speed. 0 is treated as 1.
    pub detect_stride: u32,
    /// Output H.264 bitrate as a multiple of the source video bitrate. Default
    /// 1.15 (matches the input's size/quality with a small cushion for the second
    /// encode). Falls back to a bits-per-pixel target if the source bitrate is
    /// unknown.
    pub bitrate_scale: f32,
    /// Optional swap+output window in seconds of the source clip: only frames in
    /// [start, end) are swapped and written (rebased to t=0). `None` = clip
    /// start/end.
    pub start_secs: Option<f32>,
    pub end_secs: Option<f32>,
    /// DreamID-V's knobs (ignored by HyperSwap).
    pub diffusion: DiffusionSwapOptions,
}

/// DreamID-V (diffusion face swap) settings.
#[derive(Clone, Copy, Debug)]
pub struct DiffusionSwapOptions {
    /// Denoise steps (the distill is authored for 16).
    pub steps: u32,
    /// Image-CFG guidance scale on the source face.
    pub guide_scale: f32,
    /// `None` = random.
    pub seed: Option<u64>,
    /// Working size: the clip is resized toward this pixel area (its own
    /// aspect kept), on the /16 grid. The reference default is 832x480.
    pub width: u32,
    pub height: u32,
    /// Opt-in: the unguided forward every this many steps, its guidance
    /// delta reused in between (1 = every step, the reference).
    pub cfg_every: u32,
    /// Opt-in: run on the clip's face region only (the masks' union, with
    /// context) at up to its native size, pasted back into the full-resolution
    /// frames with a feathered edge. Far fewer tokens; the output keeps the
    /// source resolution.
    pub face_crop: bool,
}

impl Default for DiffusionSwapOptions {
    fn default() -> Self {
        Self {
            steps: thinfer_models::wan::dreamidv::DEFAULT_STEPS,
            guide_scale: thinfer_models::wan::dreamidv::DEFAULT_GUIDE_SCALE,
            seed: None,
            width: 832,
            height: 480,
            cfg_every: 1,
            face_crop: false,
        }
    }
}

/// The most DreamID-V denoise steps a request may ask for.
pub const DREAMIDV_MAX_STEPS: u32 = 50;

impl Default for FaceSwapOptions {
    fn default() -> Self {
        Self {
            hyperswap_mask: false,
            occlusion: false,
            enhance: false,
            detect_stride: 1,
            bitrate_scale: 1.15,
            start_secs: None,
            end_secs: None,
            diffusion: DiffusionSwapOptions::default(),
        }
    }
}

/// Swap a face from a source image into every frame of an input video. The
/// target video is RAM-first (encrypted spill for large uploads; see
/// [`VideoInput`]); the small source image rides in RAM ([`ImageBytes`]).
#[derive(Clone, Debug)]
pub struct FaceSwapRequest {
    pub model: SwapModel,
    pub input_video: VideoInput,
    pub source_image: ImageBytes,
    pub output: PathBuf,
    pub budget: ResidencyBudget,
    pub options: FaceSwapOptions,
}

impl FaceSwapRequest {
    pub fn required_files(&self) -> Vec<FileRef> {
        let Some(swap) = self.model.file() else {
            return SwapModel::dreamidv_files();
        };
        let mut files = vec![crate::model::SCRFD, crate::model::ARCFACE, swap];
        if self.options.occlusion {
            files.push(crate::model::XSEG);
        }
        if self.options.enhance {
            files.push(crate::model::GFPGAN);
        }
        files
    }

    pub fn validate(&self) -> Result<(), String> {
        let ok = self
            .output
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("mp4"))
            .unwrap_or(false);
        if !ok {
            return Err("--output must be a .mp4 file".into());
        }
        let o = &self.options;
        if o.bitrate_scale <= 0.0 || !o.bitrate_scale.is_finite() {
            return Err("--bitrate-scale must be a positive number".into());
        }
        if let Some(s) = o.start_secs
            && (s < 0.0 || !s.is_finite())
        {
            return Err("--start must be a non-negative number of seconds".into());
        }
        if let Some(e) = o.end_secs
            && (e < 0.0 || !e.is_finite())
        {
            return Err("--end must be a non-negative number of seconds".into());
        }
        if let (Some(s), Some(e)) = (o.start_secs, o.end_secs)
            && e <= s
        {
            return Err("--end must be greater than --start".into());
        }
        if self.model.is_dreamidv() {
            if o.hyperswap_mask || o.occlusion || o.enhance || o.detect_stride > 1 {
                return Err(
                    "--hyperswap-mask, --occlusion-mask, --enhance and --detect-stride are HyperSwap options; dreamid-v makes its own face mask"
                        .into(),
                );
            }
            let d = &o.diffusion;
            validate_dim("width", d.width)?;
            validate_dim("height", d.height)?;
            if !(1..=DREAMIDV_MAX_STEPS).contains(&d.steps) {
                return Err(format!("--steps must be 1..={DREAMIDV_MAX_STEPS}"));
            }
            if d.cfg_every == 0 || d.cfg_every > d.steps {
                return Err("--cfg-every must be 1..=--steps".into());
            }
            if d.guide_scale < 0.0 || !d.guide_scale.is_finite() {
                return Err("--guide-scale must be a non-negative number".into());
            }
        }
        Ok(())
    }
}

/// A unit of work.
#[derive(Clone, Debug)]
pub enum JobRequest {
    Image(ImageRequest),
    /// Boxed: the video request is several times the others' size.
    Video(Box<VideoRequest>),
    FaceSwap(FaceSwapRequest),
}

impl JobRequest {
    /// Files this request needs in the HF cache.
    pub fn required_files(&self) -> Result<Vec<FileRef>, String> {
        match self {
            JobRequest::Image(r) => Ok(r.required_files()),
            JobRequest::Video(r) => r.required_files(),
            JobRequest::FaceSwap(r) => Ok(r.required_files()),
        }
    }
}

/// What a finished job produced.
#[derive(Clone, Debug)]
pub struct JobSummary {
    pub output: PathBuf,
    pub width: u32,
    pub height: u32,
    /// Frame count (1 for an image).
    pub frames: u32,
    /// Playback fps (video / face-swap only).
    pub fps: Option<u32>,
    /// The seed actually used (image/video; `None` for face-swap).
    pub seed: Option<u64>,
}

/// Resolve frames/durations into the clip's frame count. These models take one
/// prompt and one length: `--frames` must sit on the model grid, `--duration`
/// snaps to it, and neither gives the model default.
pub fn resolve_frames(
    model: VideoModelId,
    prompts: &[String],
    frames: &[u32],
    durations: &[f32],
    fps: u32,
) -> Result<u32, String> {
    if prompts.len() != 1 {
        return Err(format!("{model} takes a single --prompt"));
    }
    if frames.len() + durations.len() > 1 {
        return Err(format!("{model} takes a single --frames/--duration"));
    }
    if let Some(&d) = durations.first() {
        if !(d.is_finite() && d > 0.0) {
            return Err(format!(
                "--duration must be a positive number of seconds (got {d})"
            ));
        }
        return Ok(model.snap_frames((d * fps as f32).round() as u32));
    }
    match frames.first() {
        Some(&f) => {
            model.validate_frames(f)?;
            Ok(f)
        }
        None => Ok(model.default_frames()),
    }
}

// Prompts are user content: never rendered by `Debug` (structural fields only)
// and wiped when the request drops (the job ended).
impl std::fmt::Debug for ImageRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageRequest")
            .field("model", &self.model)
            .field(
                "prompt",
                &format_args!("<{} chars redacted>", self.prompt.len()),
            )
            .field("width", &self.width)
            .field("height", &self.height)
            .field("steps", &self.steps)
            .field("seed", &self.seed)
            .finish_non_exhaustive()
    }
}

impl Drop for ImageRequest {
    fn drop(&mut self) {
        zeroize::Zeroize::zeroize(&mut self.prompt);
    }
}

impl std::fmt::Debug for VideoRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoRequest")
            .field("model", &self.model)
            .field(
                "prompts",
                &format_args!("<{} redacted>", self.prompts.len()),
            )
            .field("width", &self.width)
            .field("height", &self.height)
            .field("frames", &self.frames)
            .field("seed", &self.seed)
            .finish_non_exhaustive()
    }
}

impl Drop for VideoRequest {
    fn drop(&mut self) {
        for p in &mut self.prompts {
            zeroize::Zeroize::zeroize(p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{VideoFormat, VideoRequest, resolve_frames};
    use crate::model::VideoModelId::FastwanTi2v5b as Fw;

    const FPS: u32 = 24;

    fn prompts(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("prompt {i}")).collect()
    }

    #[test]
    fn frames_resolve_on_the_model_grid() {
        assert_eq!(
            resolve_frames(Fw, &prompts(1), &[61], &[], FPS).unwrap(),
            61
        );
        assert_eq!(
            resolve_frames(Fw, &prompts(1), &[], &[], FPS).unwrap(),
            Fw.default_frames()
        );
        assert!(resolve_frames(Fw, &prompts(1), &[60], &[], FPS).is_err());
        // 2.5s @ 24fps = 60 -> 61 (nearest 4k+1).
        assert_eq!(
            resolve_frames(Fw, &prompts(1), &[], &[2.5], FPS).unwrap(),
            61
        );
    }

    #[test]
    fn multiple_prompts_or_lengths_are_an_error() {
        assert!(resolve_frames(Fw, &prompts(2), &[], &[], FPS).is_err());
        assert!(resolve_frames(Fw, &prompts(1), &[61, 61], &[], FPS).is_err());
        assert!(resolve_frames(Fw, &prompts(1), &[61], &[2.5], FPS).is_err());
    }

    // A minimal LTX t2v request for resolve() coverage. Only the fields resolve
    // reads matter; the rest take inert defaults.
    fn ltx_req(width: u32, height: u32, frames: Vec<u32>, durations: Vec<f32>) -> VideoRequest {
        use crate::model::{AttnPrecision, EncoderQuant, VaeChoice, VideoModelId, VideoSampler};
        use thinfer_core::policy::ResidencyBudget;
        VideoRequest {
            model: VideoModelId::Ltx23Distilled,
            prompts: vec!["a candid clip".into()],
            width,
            height,
            frames,
            durations,
            fps: None,
            seed: None,
            input_image: None,
            strength: 1.0,
            ref_images: Vec::new(),
            ref_image_size: Default::default(),
            sampler: VideoSampler::UniPc,
            steps: 8,
            attn_window: None,
            vae: VaeChoice::Full,
            encoder: EncoderQuant::Q8,
            i8_matmul: true,
            attn: AttnPrecision::Bf16,
            audio: true,
            upscale: true,
            super_res: None,
            audio_hq: false,
            vae_encode_f32: false,
            bf16_gemm: false,
            rewrite: false,
            rewrite_quality: crate::model::RewriteQuality::Fast,
            lora: vec![],
            vault_password: None,
            vault_dir: std::path::PathBuf::new(),
            budget: ResidencyBudget {
                ram_bytes: 5 << 30,
                vram_bytes: 5 << 30,
            },
            output: std::path::PathBuf::from("out.mp4"),
            format: VideoFormat::Mp4,
        }
    }

    #[test]
    fn ltx_duration_cap_warns_and_caps() {
        // 5s @ 1280x704 cannot fit on the 8GB card (max 49f ~2s). The request
        // must succeed (cap down), land at 49f, AND surface a warning so the
        // silent shortfall is never silent.
        let plan = ltx_req(1280, 704, vec![], vec![5.0]).resolve().unwrap();
        assert_eq!(plan.frames, 49);
        assert_eq!(plan.warnings.len(), 1, "capped duration must warn");
        assert!(plan.warnings[0].contains("capped"), "{:?}", plan.warnings);

        // The unset default (121f target) is the same silent-cap path -> warns.
        let plan = ltx_req(1280, 704, vec![], vec![]).resolve().unwrap();
        assert_eq!(plan.frames, 49);
        assert_eq!(plan.warnings.len(), 1);

        // A duration that fits leaves no LENGTH warning. 512x320 is under the
        // adherence floor, so the one warning it does carry is that one.
        let plan = ltx_req(512, 320, vec![], vec![2.0]).resolve().unwrap();
        assert_eq!(plan.warnings.len(), 1, "{:?}", plan.warnings);
        assert!(!plan.warnings[0].contains("capped"), "{:?}", plan.warnings);

        // An explicit over-budget --frames stays a hard error (fail fast), not a
        // silent cap.
        assert!(ltx_req(1280, 704, vec![121], vec![]).resolve().is_err());
    }

    #[test]
    fn ltx_warns_below_the_adherence_floor() {
        use crate::model::VideoModelId::{Ltx23Distilled as Ltx, Wan22T2vA14b as Wan};
        // The floor is a PIXEL COUNT, not a short side: 512x512 has the same 512
        // short side as the in-regime 768x512 but far fewer pixels, and measures
        // off-regime. Pin that it is judged on area.
        assert!(Ltx.ltx_dims_hold_adherence(768, 512));
        assert!(Ltx.ltx_dims_hold_adherence(1280, 704));
        assert!(Ltx.ltx_dims_hold_adherence(896, 512));
        // Both conditions bind, and each one alone would mis-classify a measured
        // rung: 896x448 has MORE pixels than the in-regime 768x512 yet goes
        // cartoon (short side 448), and 640x512 has the 512 short side yet
        // collapses (only 328k px). A pure-area or pure-short-side rule passes
        // one of these two.
        assert!(!Ltx.ltx_dims_hold_adherence(896, 448));
        assert!(!Ltx.ltx_dims_hold_adherence(640, 512));
        assert!(!Ltx.ltx_dims_hold_adherence(512, 512));
        // Portrait is the same rule transposed.
        assert!(Ltx.ltx_dims_hold_adherence(512, 768));
        assert!(!Ltx.ltx_dims_hold_adherence(448, 896));
        // Non-LTX models have their own regimes; the floor must not leak onto them.
        assert!(Wan.ltx_dims_hold_adherence(512, 320));

        // At the floor exactly: no adherence warning (49f fits, so no cap either).
        let plan = ltx_req(768, 512, vec![49], vec![]).resolve().unwrap();
        assert!(plan.warnings.is_empty(), "{:?}", plan.warnings);

        // Below it: warns, and says what the user actually loses. Critically it
        // must NOT read as a speed tradeoff -- lower dims buy clip length, not
        // time, and selling them as "faster" is what sent a user off-regime.
        let plan = ltx_req(640, 384, vec![49], vec![]).resolve().unwrap();
        assert_eq!(plan.warnings.len(), 1, "{:?}", plan.warnings);
        let w = &plan.warnings[0];
        assert!(w.contains("adherence floor"), "{w}");
        // It must name the SCENE as what degrades: the subject survives, so a
        // warning about generic "quality" would not match what the user sees.
        assert!(w.contains("SCENE"), "{w}");
        // And it must not sell low dims as a speed win.
        assert!(
            w.contains("LONGER clip, not a materially faster one"),
            "{w}"
        );
    }

    #[test]
    fn ltx_frame_cap_tracks_resolution() {
        use crate::model::VideoModelId::Ltx23Distilled as Ltx;
        // The two proven-safe envelope points on the 8GB card (these become the
        // per-resolution default clip lengths).
        assert_eq!(Ltx.ltx_max_frames(1280, 704), 49);
        assert_eq!(Ltx.ltx_max_frames(1024, 576), 73);
        // Low res fits the full ideal 5s (121f) and then some.
        assert!(Ltx.ltx_max_frames(512, 320) >= 121);
        // The device-lost config (73f @ 1280x704) is above the cap, so it is
        // rejected rather than attempted.
        assert!(Ltx.ltx_max_frames(1280, 704) < 73);
    }

    // A minimal Wan2.2-A14B t2v request for resolve() coverage.
    fn wan22_req(width: u32, height: u32, frames: Vec<u32>, durations: Vec<f32>) -> VideoRequest {
        use crate::model::{AttnPrecision, EncoderQuant, VaeChoice, VideoModelId, VideoSampler};
        use thinfer_core::policy::ResidencyBudget;
        VideoRequest {
            model: VideoModelId::Wan22T2vA14b,
            prompts: vec!["a candid clip".into()],
            width,
            height,
            frames,
            durations,
            fps: None,
            seed: None,
            input_image: None,
            strength: 1.0,
            ref_images: Vec::new(),
            ref_image_size: Default::default(),
            sampler: VideoSampler::UniPc,
            steps: 4,
            attn_window: None,
            vae: VaeChoice::Full,
            encoder: EncoderQuant::Q8,
            i8_matmul: true,
            attn: AttnPrecision::Bf16,
            audio: false,
            upscale: false,
            super_res: None,
            audio_hq: false,
            vae_encode_f32: false,
            bf16_gemm: false,
            rewrite: false,
            rewrite_quality: crate::model::RewriteQuality::Fast,
            lora: vec![],
            vault_password: None,
            vault_dir: std::path::PathBuf::new(),
            budget: ResidencyBudget {
                ram_bytes: 5 << 30,
                vram_bytes: 5 << 30,
            },
            output: std::path::PathBuf::from("out.mp4"),
            format: VideoFormat::Mp4,
        }
    }

    #[test]
    fn wan22_frame_cap_tracks_resolution() {
        use crate::model::VideoModelId::Wan22T2vA14b as W;
        // The cap is now the model's 81f @ 832x480 design envelope (32760 latent
        // cells = f_lat*h/16*w/16), NOT a TDR-fault guard (the crash is fixed by
        // op_sdpa query chunking). At 832x480 (1560 cells/lat-frame) that is
        // f_lat 21 -> 81f; at 512x288 (576/lat-frame) f_lat 56 -> 221f; at 256x256
        // (256/lat-frame) f_lat 127 -> 505f. All on the 4k+1 grid.
        assert_eq!(W.max_frames(832, 480), Some(81));
        assert_eq!(W.max_frames(512, 288), Some(221));
        assert_eq!(W.max_frames(256, 256), Some(505));
        // The default config (832x480 x 33f, the validated length) sits well under
        // the envelope, so no cap warning.
        let plan = wan22_req(832, 480, vec![], vec![]).resolve().unwrap();
        assert_eq!(plan.frames, 33);
        assert!(plan.warnings.is_empty(), "{:?}", plan.warnings);
    }

    #[test]
    fn wan22_over_envelope_frames_error_duration_caps() {
        // Explicit over-cap --frames at 832x480 (85f > the 81f cap) is a hard error
        // (fail fast at submit rather than start a multi-hour job that overruns the
        // model's design envelope).
        assert!(wan22_req(832, 480, vec![85], vec![]).resolve().is_err());
        // A duration over the envelope caps DOWN to the cap with a warning
        // (6s @ 16fps = 97f > the 81f cap).
        let plan = wan22_req(832, 480, vec![], vec![6.0]).resolve().unwrap();
        assert_eq!(plan.frames, 81);
        assert_eq!(plan.warnings.len(), 1, "capped duration must warn");
        // A request inside the envelope is untouched and silent.
        let plan = wan22_req(832, 480, vec![49], vec![]).resolve().unwrap();
        assert_eq!(plan.frames, 49);
        assert!(plan.warnings.is_empty(), "{:?}", plan.warnings);
    }

    #[test]
    fn fastwan_snaps_to_4k_plus_1() {
        assert_eq!(Fw.snap_frames(97), 97);
        assert_eq!(Fw.snap_frames(60), 61);
        assert_eq!(Fw.snap_frames(50), 49);
        assert_eq!(Fw.snap_frames(1), 1);
        assert_eq!(Fw.snap_frames(0), 1);
        // Default is 5s @ 24fps = 120 -> 121 (nearest 4k+1).
        assert_eq!(Fw.default_frames(), 121);
    }

    #[test]
    fn snapped_frames_always_validate() {
        for raw in 1u32..400 {
            Fw.validate_frames(Fw.snap_frames(raw))
                .unwrap_or_else(|e| panic!("{Fw} snap({raw}) not legal: {e}"));
        }
    }

    #[test]
    fn validate_rejects_off_grid() {
        assert!(Fw.validate_frames(96).is_err());
        assert!(Fw.validate_frames(97).is_ok());
    }

    /// A FastH3-Trim request at 832x480 with `prompts` and `durations`.
    fn trim_req(prompts: usize, durations: Vec<f32>) -> VideoRequest {
        let mut r = ltx_req(832, 480, vec![], durations);
        r.model = crate::model::VideoModelId::Fasth3Trim;
        r.prompts = (0..prompts).map(|i| format!("clip {i}")).collect();
        r
    }

    #[test]
    fn fasth3_chains_past_one_clip() {
        let row_limit = crate::model::VideoModelId::Fasth3Trim.minimax_row_limit(true);
        // 15s and under stays one clip.
        let plan = trim_req(1, vec![15.0]).resolve_minimax(row_limit).unwrap();
        assert!(plan.chain.is_empty());
        assert_eq!(plan.frames, 362);
        // 30s: as few equal clips as fit, each on the 17n+5 grid, together
        // (minus the repeated keyframes) at least the ask.
        let plan = trim_req(1, vec![30.0]).resolve_minimax(row_limit).unwrap();
        assert!(plan.chain.len() >= 2, "{:?}", plan.chain);
        let lens: Vec<u32> = plan.chain.iter().map(|c| c.frames).collect();
        assert!(lens.iter().all(|&f| f % 17 == 5 && f <= 362), "{lens:?}");
        assert!(lens.windows(2).all(|w| w[0] == w[1]), "{lens:?}");
        assert_eq!(
            plan.frames,
            lens[0] + lens[1..].iter().map(|f| f - 1).sum::<u32>()
        );
        assert!(plan.frames >= 720);
        assert!(plan.chain.iter().all(|c| c.prompt == "clip 0"));
        // One length per prompt, each its own clip.
        let plan = trim_req(2, vec![5.0, 4.0])
            .resolve_minimax(row_limit)
            .unwrap();
        let lens: Vec<u32> = plan.chain.iter().map(|c| c.frames).collect();
        assert_eq!(lens, vec![124, 107]);
        assert_eq!(plan.chain[1].prompt, "clip 1");
        assert_eq!(plan.frames, 124 + 106);
        // A per-prompt clip past the cap is an error, not a silent split.
        assert!(
            trim_req(2, vec![20.0, 5.0])
                .resolve_minimax(row_limit)
                .is_err()
        );
        // The Ref2VA model has no FL2VA keyframes, so no chain.
        let mut omni = trim_req(2, vec![5.0, 5.0]);
        omni.model = crate::model::VideoModelId::Fasth3Omni;
        assert!(omni.resolve_minimax(row_limit).is_err());
    }
}
