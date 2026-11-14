//! `thinfer generate video`. Thin clap adapter over
//! `thinfer_app::VideoRequest`; the frame-plan resolution, denoise, and MP4
//! encode all live in `thinfer-app`.

use std::path::PathBuf;

use clap::Args;
use thinfer_app::config::ResidencyBudget;
use thinfer_app::model::{
    AttnPrecision, EncoderQuant, RefImageSize, RewriteQuality, SuperRes, VaeChoice, VideoModelId,
    VideoSampler,
};
use thinfer_app::request::{ImageBytes, VideoFormat, VideoRequest};
use thinfer_app::wire::{JobSpec, VideoSpec};
use thinfer_app::{JobRequest, parse_budget, resolve_output_format};

/// FastWan is CFG-free (no `--guidance-scale`/`--negative-prompt`). The denoise
/// sampler is selectable: `--sampler unipc` (default, `--steps` configurable,
/// matches the public FastWan Spaces) or `--sampler dmd` (the fixed 3-step
/// byte-parity reference; `--steps` is ignored). Dims must be /32; frames must be
/// 4k+1 (the causal-VAE temporal grid). For a fast first run, pass `--frames 5`.
#[derive(Args)]
pub struct GenerateVideo {
    /// Model identifier. Defaults to `fastwan-ti2v-5b` (a fast 5B DMD distill,
    /// 960x544, no prompt rewrite; the e2e-validated path).
    #[arg(long, default_value_t = VideoModelId::DEFAULT, value_enum)]
    pub model: VideoModelId,
    /// Text prompt. Repeat `--prompt` for chained clips (fasth3-trim,
    /// minimax-h3): each prompt drives one clip. Required.
    #[arg(long)]
    pub prompt: Vec<String>,
    /// fasth3-omni only: a reference image (PNG/JPEG) the prompt can name as
    /// `<Picture N>` (N = its position among the --ref-image flags, from 1).
    /// Repeat for several, up to 9.
    #[arg(long = "ref-image")]
    pub ref_images: Vec<PathBuf>,
    /// fasth3-omni only: how each reference image is sized. `match` (default,
    /// as in ComfyUI: the canvas's pixel area, never upscaled), `max` (short
    /// edge capped at 2048) or `diffusers` (short edge exactly 2048, the
    /// reference; ~2.4x slower with one reference, no better identity).
    #[arg(long, value_enum, default_value_t)]
    pub ref_image_size: RefImageSize,
    /// Output video file (e.g. `out.mp4`).
    #[arg(long)]
    pub output: PathBuf,
    /// Output format. Defaults to inferring from the `--output` extension.
    #[arg(long, value_enum)]
    pub output_format: Option<VideoFormat>,
    /// Frame width in pixels. Defaults per model (FastWan 960, LTX 256).
    /// FastWan needs a multiple of 32; LTX a multiple of 64.
    #[arg(long)]
    pub width: Option<u32>,
    /// Frame height in pixels. Defaults per model (FastWan 544, LTX 256).
    /// FastWan needs a multiple of 32; LTX a multiple of 64.
    #[arg(long)]
    pub height: Option<u32>,
    /// Output frame count. Must be `4 * k + 1` (LTX `8 * k + 1`). Mutually
    /// exclusive with `--duration`. Chained clips take one value for the whole
    /// video or one per `--prompt`.
    #[arg(long, conflicts_with = "duration")]
    pub frames: Vec<u32>,
    /// Target clip length in seconds (mutually exclusive with `--frames`):
    /// frames = round(duration * fps), snapped to the model's legal grid.
    #[arg(long, conflicts_with = "frames")]
    pub duration: Vec<f32>,
    /// Playback fps written into the MP4 (and used to convert `--duration`).
    /// Defaults to the model's preferred fps (Wan TI2V = 24).
    #[arg(long)]
    pub fps: Option<u32>,
    /// Seed. Omit for a randomized seed.
    #[arg(long)]
    pub seed: Option<u64>,
    /// Denoise sampler: `unipc` (multistep, `--steps` configurable) or `dmd`
    /// (fixed 3-step byte-parity path). Unset -> the model default (`dmd` for
    /// fastwan2.1-t2v-1.3b, else `unipc`).
    #[arg(long, value_enum)]
    pub sampler: Option<VideoSampler>,
    /// UniPC denoise steps (1..=8). Ignored when
    /// `--sampler dmd`. Unset -> the model default.
    #[arg(long)]
    pub steps: Option<u32>,
    /// Temporal self-attention window radius in LATENT frames (Wan2.2 14B). Each
    /// query attends only to keys within `±N` latent frames, breaking the
    /// O(frames^2) self-attention cost on long clips at the price of long-range
    /// temporal coherence. Unset = the model default (3 for Wan2.2 14B, long
    /// clips only); `0` forces full attention. Honored only on the activation-
    /// tiled long-clip path.
    #[arg(long)]
    pub attn_window: Option<u32>,
    /// First-frame conditioning image (PNG/JPEG). Optional, on
    /// hunyuan-video-1.5-ti2v, ltx2-rapid, fasth3-trim and minimax-h3 (FL2VA,
    /// stretched to the canvas): with it the run animates the image (I2V);
    /// without it the model generates from the prompt alone. FL2VA models also
    /// chain clips past ~15s (or one per `--prompt`), each from the previous
    /// clip's last frame.
    #[arg(long)]
    pub input_image: Option<PathBuf>,
    /// LTX native-I2V frame-0 conditioning strength (0.0..=1.0, default 1.0):
    /// 1.0 locks the input frame through the denoise; lower lets it drift toward
    /// the model's own first frame. Used only with --input-image on ltx2-rapid.
    #[arg(long, default_value_t = 1.0)]
    pub strength: f32,
    /// Host RAM budget for the weight residency manager. e.g. `8G`, `512M`.
    #[arg(long)]
    pub ram_budget: Option<String>,
    /// GPU VRAM budget for the weight residency manager.
    #[arg(long)]
    pub vram_budget: Option<String>,
    /// Skip the TTY consent prompt and download missing weight files.
    #[arg(long, default_value_t = false)]
    pub download_as_needed: bool,
    /// VAE decoder. Default `tiny` (LightTAE): ~0.4GB, ~50x faster decode at
    /// near-identical quality. `full` is the bit-clean parity path.
    /// MiniMax-H3 / FastH3 default to `nvfp4` (FastVideo's NVFP4 decoder:
    /// indistinguishable frames, ~2x faster decode; NVIDIA Blackwell, else the
    /// fp16 decoder runs instead); `full` = the original fp16 decoder,
    /// `nvfp4-fast` = 8 layers, fastest, softer.
    #[arg(long, value_enum)]
    pub vae: Option<VaeChoice>,
    /// MiniMax-H3 family attention precision (FastH3's sparse attention, the
    /// GGUF models' dense attention). Default `int8-fp8` (SageAttention2-
    /// style INT8 QK^T + FP8 P.V, no visible quality change; GPUs below sm_89
    /// run `int8`); `int8` keeps P.V in bf16 (~10% slower overall); `bf16` is
    /// the reference path. Each is a different sample per seed.
    #[arg(long, value_enum)]
    pub attn: Option<AttnPrecision>,
    /// LTX-2.3 text encoder (all LTX models). `q8` (default) is the
    /// conditioning-quality baseline; `q4` is Q4_K_M -- ~2.8x faster encode
    /// (~16s -> ~6s) for lower-precision conditioning. Ignored by Wan.
    #[arg(long, value_enum, default_value_t = EncoderQuant::Q8)]
    pub encoder: EncoderQuant,
    /// Disable the default DP4A int8 matmul on the quantization-safe DiT sites
    /// (forces the bf16 reference path throughout).
    #[arg(long)]
    pub no_i8_matmul: bool,
    /// Skip the audio track (LTX joint-AV only): decode video only, for a faster
    /// silent MP4. Ignored by the silent Wan models.
    #[arg(long)]
    pub no_audio: bool,
    /// LTX-2.3 distilled only: opt in to the 2x spatial-upscale refine path
    /// (half-res denoise -> latent upscale -> refine), the cheaper route to high
    /// res. Default off = single-stage denoise at the target res. Ignored by Wan.
    #[arg(long)]
    pub upscale: bool,
    /// MiniMax-H3 / FastH3: RelayVSR super-resolution of the finished video,
    /// 2x (LightX2V's retrain for H3 output) or 4x. Runs after generation,
    /// before encoding; default off (native resolution).
    #[arg(long, value_enum)]
    pub super_res: Option<SuperRes>,
    /// LTX joint-AV: refine the denoise schedule for better audio. The distilled
    /// 8-step table under-resolves the delicate audio residual (speech drifts to
    /// gibberish); this subdivides the coarse sigma jumps into more Euler
    /// substeps, keeping the distilled anchors exactly. Applies to every LTX
    /// line: the single-stage schedule on ltx2-rapid, and BOTH stage schedules on
    /// the two-stage 22B line (8+3 -> 16+11 steps, ~2.5x denoise). Ignored by Wan.
    #[arg(long)]
    pub audio_hq: bool,
    /// MiniMax-H3 / FastH3: run the reference / first-frame VAE encode in f32
    /// (the exact path). By default it runs in bf16 on tensor cores, the
    /// precision the reference runs that encoder in (~3x faster encode).
    #[arg(long)]
    pub vae_encode_f32: bool,
    /// MiniMax-H3 GGUF models (minimax-h3, minimax-h3-turbo(-ref),
    /// hyperflow-flash(-ref)): run every block GEMM in bf16 with unfolded
    /// adapters, the exact path. By default each block's sites run as 8-bit
    /// tensor-core GEMMs (INT8 qkv/fc1, FP8 out/fc2) and turbo / LoRA adapters
    /// fold into the bf16 weights (~30% faster steps, a different but equally
    /// good sample).
    #[arg(long)]
    pub bf16_gemm: bool,
    /// HunyuanVideo 1.5 only: skip prompt rewriting and send the raw prompt. By
    /// default a short prompt is expanded into a detailed, structured caption by
    /// the on-device Qwen3-VL rewriter (the model is trained on long captions;
    /// raw short prompts are out-of-distribution and yield incoherent video).
    #[arg(long)]
    pub no_rewrite: bool,
    /// HunyuanVideo 1.5 only: which rewriter model to use. `fast` (default) runs
    /// the ~2.5GB Qwen3-VL-4B (quick even under a tight VRAM budget); `full` runs
    /// the ~5.85GB Qwen3-VL-8B (slightly richer captions, much slower). Ignored
    /// with `--no-rewrite` or on non-Hunyuan models.
    #[arg(long, value_enum, default_value_t = RewriteQuality::Fast)]
    pub rewrite_quality: RewriteQuality,
    /// Fold a stored adapter into the DiT, as `NAME_OR_ID[:WEIGHT]` (repeatable,
    /// applied in order). Resolved against the vault for `--model`; needs the
    /// vault password (hidden prompt, or `THINFER_VAULT_PASSWORD`). Local runs
    /// only (not `--remote`). AnyFlow, FastWan2.1-1.3B, Wan2.1-1.3B, MiniMax-H3
    /// and FastH3-Trim accept adapters (H3: applied at runtime, any layout).
    #[arg(long = "lora", value_name = "NAME[:WEIGHT]")]
    pub lora: Vec<String>,
    /// Vault directory for `--lora`. Defaults to the shared location
    /// (`THINFER_VAULT_DIR`, else `<hf-cache>/vault`).
    #[arg(long)]
    pub vault_dir: Option<PathBuf>,
    #[command(flatten)]
    pub remote: super::RemoteArgs,
}

/// Read a file and base64-encode it for a remote job spec, or `None` when the
/// path is absent.
fn base64_of(path: Option<&std::path::Path>) -> Result<Option<String>, String> {
    path.map(base64_file).transpose()
}

/// Read a file and base64-encode it for a remote job spec.
pub(super) fn base64_file(path: &std::path::Path) -> Result<String, String> {
    use base64::Engine;
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

pub async fn run_video(args: GenerateVideo) -> Result<(), String> {
    let format = resolve_output_format(
        args.output_format,
        &args.output,
        VideoFormat::from_ext,
        VideoFormat::KNOWN,
    )?;

    if args.remote.remote.is_some() {
        if format != VideoFormat::Mp4 {
            return Err("--remote only produces MP4 (png-frames is a local debug format)".into());
        }
        if !args.lora.is_empty() {
            return Err(
                "--lora (adapters) is not supported over --remote; run locally or use the web UI"
                    .into(),
            );
        }
        let spec = JobSpec::Video(VideoSpec {
            model: Some(args.model),
            prompts: args.prompt,
            width: args.width,
            height: args.height,
            frames: (!args.frames.is_empty()).then_some(args.frames),
            durations: (!args.duration.is_empty()).then_some(args.duration),
            fps: args.fps,
            seed: args.seed,
            sampler: args.sampler,
            steps: args.steps,
            attn_window: args.attn_window,
            vae: args.vae,
            attn: args.attn,
            input_image: base64_of(args.input_image.as_deref())?,
            strength: Some(args.strength),
            ref_images: Some(
                args.ref_images
                    .iter()
                    .map(|p| Ok(base64_of(Some(p))?.unwrap_or_default()))
                    .collect::<Result<Vec<_>, String>>()?,
            )
            .filter(|v| !v.is_empty()),
            ref_image_size: Some(args.ref_image_size),
            encoder: Some(args.encoder),
            i8_matmul: Some(!args.no_i8_matmul),
            audio: Some(!args.no_audio),
            upscale: Some(args.upscale || args.model.two_stage_default()),
            super_res: args.super_res,
            audio_hq: Some(args.audio_hq),
            vae_encode_f32: Some(args.vae_encode_f32),
            bf16_gemm: Some(args.bf16_gemm),
            rewrite: Some(!args.no_rewrite),
            rewrite_quality: Some(args.rewrite_quality),
            // Remote path defers coopmat to the server default; local runs read
            // THINFER_NO_COOPMAT via BackendConfig.
            disable_coopmat: None,
            // Adapters over --remote are guarded above.
            lora: Vec::new(),
            password: None,
        });
        return super::run_remote(&args.remote, spec, args.output).await;
    }
    let ram = parse_budget("--ram-budget", args.ram_budget.as_deref())?;
    let vram = parse_budget("--vram-budget", args.vram_budget.as_deref())?;

    // Resolve any --lora names/ids against the vault (needs the password).
    let (lora, vault_password) = if args.lora.is_empty() {
        (Vec::new(), None)
    } else {
        let (refs, password) =
            super::image::resolve_loras(args.model, args.vault_dir.as_deref(), &args.lora)?;
        (refs, Some(password))
    };

    let (def_w, def_h) = args.model.video_defaults();
    let req = VideoRequest {
        model: args.model,
        prompts: args.prompt,
        width: args.width.unwrap_or(def_w),
        height: args.height.unwrap_or(def_h),
        frames: args.frames,
        durations: args.duration,
        fps: args.fps,
        seed: args.seed,
        input_image: match &args.input_image {
            Some(p) => Some(ImageBytes(
                std::fs::read(p).map_err(|e| format!("read {}: {e}", p.display()))?,
            )),
            None => None,
        },
        strength: args.strength,
        ref_images: args
            .ref_images
            .iter()
            .map(|p| {
                std::fs::read(p)
                    .map(ImageBytes)
                    .map_err(|e| format!("read {}: {e}", p.display()))
            })
            .collect::<Result<_, _>>()?,
        ref_image_size: args.ref_image_size,
        sampler: args.sampler.unwrap_or(args.model.default_sampler()),
        steps: args.steps.unwrap_or(args.model.default_steps()),
        attn_window: args.attn_window,
        vae: args.vae.unwrap_or(args.model.default_vae()),
        attn: args.attn.unwrap_or(args.model.default_attn()),
        encoder: args.encoder,
        i8_matmul: !args.no_i8_matmul,
        audio: !args.no_audio,
        // LTX defaults to two-stage (single-stage at the widescreen default OOMs
        // 8GB and low-res single-stage is OOD); `--upscale` is store-true, so OR
        // with the model default. Wan ignores upscale.
        upscale: args.upscale || args.model.two_stage_default(),
        super_res: args.super_res,
        audio_hq: args.audio_hq,
        vae_encode_f32: args.vae_encode_f32,
        bf16_gemm: args.bf16_gemm,
        rewrite: !args.no_rewrite,
        rewrite_quality: args.rewrite_quality,
        lora,
        vault_password,
        vault_dir: thinfer_app::vault::resolve_dir(args.vault_dir.as_deref()),
        budget: ResidencyBudget {
            ram_bytes: ram,
            vram_bytes: vram,
        },
        output: args.output,
        format,
    };
    // Fail fast on dims / shot-plan errors before any download or GPU work.
    req.resolve()?;
    let files = req.required_files()?;
    super::run_job(
        JobRequest::Video(Box::new(req)),
        &files,
        args.download_as_needed,
        ram,
        vram,
    )
    .await
}
