//! `thinfer generate video` (FastWan2.2-TI2V-5B t2v) plus the CLI-only MP4
//! encode/mux. The engine yields CTHW f32 frames; this module turns them into a
//! single `.mp4` (H.264 via openh264, muxed with the `mp4` crate). Encode lives
//! here, not in `thinfer-native`/`thinfer-models`, so the codec dep stays in the
//! one crate that is already the glue layer. Web video will use the browser's
//! WebCodecs encoder instead (nothing here is shared down).

use std::path::{Path, PathBuf};

use clap::{Args, ValueEnum};
use thinfer_core::manifest::{FileRef, ModelManifest};
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::trace::DIAG;
use thinfer_models::wan::manifest as wanmf;
use thinfer_models::wan::pipeline::{
    GenerationParams, ProgressEvent, VaeChoice, WanModel, WanVideo,
};
use thinfer_models::wan::source::{WanSource, open_longlive_source};
use thinfer_models::z_image::pipeline::encode_png;
use thinfer_native::tokenizer::HfTokenizer;
use thinfer_native::{MmapFileOpener, cache};

use super::{
    PercentLogger, backend_for_stats, confirm_download, init_backend, parse_budget, random_seed,
    report_mem, resolve_output_format, validate_dim,
};

/// Output container.
#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum VideoFormat {
    Mp4,
    /// Raw per-frame PNG sequence: `--output` is a directory, frames land as
    /// `frame{n:03}.png`. Not inferable from an extension, so it must be set
    /// explicitly (`--output-format png-frames`). Bypasses the H.264 encode
    /// path entirely, which makes it the tool for inspecting exactly what the
    /// VAE decoded (no codec in the way).
    PngFrames,
}

impl VideoFormat {
    fn from_ext(ext: &str) -> Option<Self> {
        match ext {
            "mp4" => Some(Self::Mp4),
            _ => None,
        }
    }
    const KNOWN: &'static str = "mp4";
}

/// Defaults: 960x544, 97 frames (= 4*24+1, ~4.0s @ 24fps). FastWan is
/// DMD-distilled (fixed 3-step schedule, baked in) and CFG-free, so there is no
/// `--steps`/`--guidance-scale`/`--negative-prompt`. Dims must be /32 (TI2V VAE
/// 16x spatial * patch 2); frames must be 4k+1 (the causal-VAE temporal grid).
/// For a fast first run, pass `--frames 5` (still full self-attention, so its
/// wall-clock is the perf signal).
#[derive(Args)]
pub struct GenerateVideo {
    /// Model identifier. Defaults to `fastwan-ti2v-5b` (safetensors, the
    /// e2e-validated path). GGUF variants are deferred.
    #[arg(long, default_value_t = VideoModelId::FastwanTi2v5b, value_enum)]
    pub model: VideoModelId,
    #[arg(long)]
    pub prompt: String,
    /// Output video file. A single file (e.g. `out.mp4`); no frame dir.
    #[arg(long)]
    pub output: PathBuf,
    /// Output format. Defaults to inferring from the `--output` extension;
    /// errors if the extension is missing or unrecognized.
    #[arg(long, value_enum)]
    pub output_format: Option<VideoFormat>,
    /// Frame width in pixels. Must be divisible by 32.
    #[arg(long, default_value_t = 960)]
    pub width: u32,
    /// Frame height in pixels. Must be divisible by 32.
    #[arg(long, default_value_t = 544)]
    pub height: u32,
    /// Output frame count. Must be `4 * k + 1` (causal-VAE temporal grid).
    #[arg(long, default_value_t = 97)]
    pub frames: u32,
    /// Playback frames-per-second written into the MP4. Output-only metadata;
    /// the DMD model takes no fps conditioning.
    #[arg(long, default_value_t = 24)]
    pub fps: u32,
    /// Seed. Omit for a randomized seed.
    #[arg(long)]
    pub seed: Option<u64>,
    /// img2vid conditioning image. Not yet wired (engine path is t2v-only).
    #[arg(long)]
    pub input_image: Option<PathBuf>,
    /// Host RAM budget for the weight residency manager. e.g. `8G`, `512M`,
    /// raw bytes.
    #[arg(long)]
    pub ram_budget: Option<String>,
    /// GPU VRAM budget for the weight residency manager.
    #[arg(long)]
    pub vram_budget: Option<String>,
    /// Skip the TTY consent prompt and download missing weight files.
    #[arg(long, default_value_t = false)]
    pub download_as_needed: bool,
    /// VAE decoder. Default `tiny` (LightTAE `lighttaew2_2`): a ~0.4GB tiny
    /// decoder, ~50x faster decode at near-identical quality; downloads its own
    /// weight file. `full` is the real AutoencoderKLWan (the bit-clean parity
    /// path). Both temporally tile the decode to hold the VRAM budget.
    #[arg(long, value_enum, default_value_t = VaeChoiceArg::Tiny)]
    pub vae: VaeChoiceArg,
}

/// CLI mirror of [`VaeChoice`] (clap `ValueEnum`).
#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum VaeChoiceArg {
    Full,
    Tiny,
}

impl From<VaeChoiceArg> for VaeChoice {
    fn from(v: VaeChoiceArg) -> Self {
        match v {
            VaeChoiceArg::Full => VaeChoice::Full,
            VaeChoiceArg::Tiny => VaeChoice::Tiny,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum VideoModelId {
    /// FastWan2.2-TI2V-5B-FullAttn, DMD-distilled (3-step, CFG-free), fp32/bf16
    /// safetensors. The e2e-validated path. GGUF variants are deferred.
    #[value(name = "fastwan-ti2v-5b")]
    FastwanTi2v5b,
    /// LongLive-2.0-5B: the causal/AR (autoregressive) long-video finetune of the
    /// FastWan base (same 5B geometry). 4-step FlowUniPC per chunk over a windowed
    /// KV cache. DiT is a standalone `.pt`; umT5 + VAE reuse the FastWan base. The
    /// frame count must give a latent frame count divisible by 8 (the chunk size):
    /// e.g. `--frames 29` (8 latent frames) or `--frames 61` (16).
    #[value(name = "longlive-2.0-5b")]
    Longlive205b,
}

impl VideoModelId {
    fn manifest(self) -> &'static ModelManifest {
        &thinfer_models::wan::manifest::MANIFEST
    }

    fn variant(self) -> &'static wanmf::VariantFiles {
        wanmf::variant(&self.to_string()).expect("CLI VideoModelId missing from VARIANTS registry")
    }

    /// AR (LongLive) path: the `.pt` DiT + windowed-KV-cache chunk loop.
    fn is_ar(self) -> bool {
        matches!(self, VideoModelId::Longlive205b)
    }
}

impl std::fmt::Display for VideoModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VideoModelId::FastwanTi2v5b => f.write_str("fastwan-ti2v-5b"),
            VideoModelId::Longlive205b => f.write_str("longlive-2.0-5b"),
        }
    }
}

/// Load the model from `source` and run the appropriate denoise (FastWan DMD vs
/// LongLive AR). Generic over the weight source so both the safetensors
/// (`WanSource`) and the `.pt`-backed (`LongLiveSource`) paths share the load +
/// generate + error-mapping tail.
async fn build_and_generate<S: thinfer_core::weight::WeightSource>(
    backend: std::sync::Arc<thinfer_core::backend::WgpuBackend>,
    source: S,
    budget: ResidencyBudget,
    tokenizer: HfTokenizer,
    vae: VaeChoice,
    gen_params: &GenerationParams,
    progress: &dyn Fn(ProgressEvent),
    ar: bool,
) -> Result<WanVideo, String> {
    let residency = WeightResidency::new(source, budget);
    let model = {
        let _s = tracing::info_span!("model_load").entered();
        WanModel::load(backend, residency, tokenizer, vae)
            .await
            .map_err(|e| format!("model load: {e:?}"))?
    };
    if ar {
        model
            .generate_ar(gen_params, vae, Some(progress))
            .await
            .map_err(|e| format!("generate: {e:?}"))
    } else {
        model
            .generate(gen_params, vae, Some(progress))
            .await
            .map_err(|e| format!("generate: {e:?}"))
    }
}

pub async fn run_video(args: GenerateVideo) -> Result<(), String> {
    validate_dim("height", args.height)?;
    validate_dim("width", args.width)?;
    // Causal VAE temporal grid: frame count must be 4k+1.
    if args.frames == 0 || args.frames % 4 != 1 {
        return Err(format!(
            "--frames must be 4*k + 1 (got {}); e.g. 1, 5, 9, ..., 97",
            args.frames
        ));
    }
    if args.fps == 0 {
        return Err("--fps must be > 0".into());
    }
    if args.input_image.is_some() {
        return Err("--input-image (img2vid) not yet wired; the engine path is t2v-only".into());
    }
    // Resolve up front so a bad extension fails before any download / GPU work.
    let format = resolve_output_format(
        args.output_format,
        &args.output,
        VideoFormat::from_ext,
        VideoFormat::KNOWN,
    )?;

    let ram_bytes = parse_budget("--ram-budget", args.ram_budget.as_deref())?;
    let vram_bytes = parse_budget("--vram-budget", args.vram_budget.as_deref())?;
    let budget = ResidencyBudget {
        ram_bytes,
        vram_bytes,
    };

    let manifest = args.model.manifest();
    let variant = args.model.variant();
    let vae: VaeChoice = args.vae.into();
    let mut all_files: Vec<FileRef> = variant.files().map(|(_, f)| *f).collect();
    // The tiny decoder weight is not in the static variant file set; pull it
    // (and only it) in when `--vae tiny` so the parity path never downloads it.
    if vae == VaeChoice::Tiny {
        all_files.push(
            *manifest
                .get(wanmf::role::TINY_VAE)
                .ok_or("manifest missing tiny VAE role")?,
        );
    }
    let (_resolved, missing) = cache::resolve_all(all_files.iter());

    if !missing.is_empty()
        && !confirm_download(&missing, args.download_as_needed).map_err(|e| e.to_string())?
    {
        return Err("declined download; rerun with --download-as-needed or `hf download …`".into());
    }
    for f in &missing {
        let progress = std::sync::Arc::new(PercentLogger::new(format!("{}/{}", f.repo, f.path)));
        cache::download_with_progress(f, Some(progress))
            .await
            .map_err(|e| format!("{e:?}"))?;
    }

    let resolve_role = |role: &str| -> Result<PathBuf, String> {
        let r = manifest
            .get(role)
            .ok_or_else(|| format!("manifest missing role {role}"))?;
        cache::resolve(r)
            .ok_or_else(|| format!("{}/{} not in cache after download", r.repo, r.path))
    };

    let mut weight_openers: Vec<MmapFileOpener> = Vec::with_capacity(variant.weight_roles.len());
    for role in variant.weight_roles {
        let path = resolve_role(role)?;
        weight_openers.push(
            MmapFileOpener::new(&path)
                .await
                .map_err(|e| format!("open {}: {e}", path.display()))?,
        );
    }
    // Append the tiny decoder as an extra safetensors shard (its `decoder.{N}`
    // keys are disjoint from the real VAE's `decoder.conv_in/...`), so both live
    // in one catalog. Only when selected, so the parity path is byte-for-byte
    // the same source it always was.
    if vae == VaeChoice::Tiny {
        let path = resolve_role(wanmf::role::TINY_VAE)?;
        weight_openers.push(
            MmapFileOpener::new(&path)
                .await
                .map_err(|e| format!("open {}: {e}", path.display()))?,
        );
    }
    let tokenizer_path = resolve_role(wanmf::role::TOKENIZER_JSON)?;
    let tokenizer = HfTokenizer::from_path(&tokenizer_path)
        .await
        .map_err(|e| format!("tokenizer {}: {e:?}", tokenizer_path.display()))?;

    tracing::info!(
        target: DIAG,
        model = %args.model,
        prompt = %args.prompt,
        width = args.width,
        height = args.height,
        frames = args.frames,
        fps = args.fps,
        seed = ?args.seed,
        ram_budget = ram_bytes,
        vram_budget = vram_bytes,
        "generate video start",
    );

    let backend = init_backend().await?;
    let stats = backend_for_stats(&backend);

    let seed = args.seed.unwrap_or_else(random_seed);
    let t_run = std::time::Instant::now();
    let stamp = move || format!("[{:6.1}s]", t_run.elapsed().as_secs_f64());
    eprintln!(
        "{} Generating {}x{} video, {} frames, {} fps, seed {} ({})",
        stamp(),
        args.width,
        args.height,
        args.frames,
        args.fps,
        seed,
        args.model,
    );
    let progress = move |ev: ProgressEvent| match ev {
        ProgressEvent::TextEncode => eprintln!("{} Encoding prompt", stamp()),
        ProgressEvent::Step { i, n } => eprintln!("{} Denoising step {i}/{n}", stamp()),
        ProgressEvent::ChunkStep {
            chunk,
            num_chunks,
            step,
            num_steps,
        } => eprintln!(
            "{} Denoising chunk {chunk}/{num_chunks} step {step}/{num_steps}",
            stamp()
        ),
        ProgressEvent::VaeDecode => eprintln!("{} Decoding latents (VAE)", stamp()),
    };
    let gen_params = GenerationParams {
        prompt: args.prompt,
        height: args.height,
        width: args.width,
        num_frames: args.frames,
        seed,
    };
    // FastWan loads the all-safetensors `WanSource`; LongLive unions the renamed
    // `.pt` DiT over the base safetensors (umT5 + VAE) via `LongLiveSource`. Both
    // share the generic load + generate tail.
    let video = if args.model.is_ar() {
        let pt_path = resolve_role(wanmf::role::LONGLIVE_DIT)?;
        let dit_opener = MmapFileOpener::new(&pt_path)
            .await
            .map_err(|e| format!("open {}: {e}", pt_path.display()))?;
        let num_layers = thinfer_models::wan::dit_block::WanDitConfig::longlive_2_0_5b().num_layers;
        let source = open_longlive_source(dit_opener, weight_openers, num_layers)
            .await
            .map_err(|e| format!("parse LongLive weights: {e:?}"))?;
        build_and_generate(
            backend,
            source,
            budget,
            tokenizer,
            vae,
            &gen_params,
            &progress,
            true,
        )
        .await?
    } else {
        // GGUF is deferred: bringup is safetensors-only (the union path in
        // `wan::source` is retained for when a published FastWan GGUF is wired).
        let source = WanSource::open(weight_openers, None)
            .await
            .map_err(|e| format!("parse weight files: {e:?}"))?;
        build_and_generate(
            backend,
            source,
            budget,
            tokenizer,
            vae,
            &gen_params,
            &progress,
            false,
        )
        .await?
    };

    match format {
        VideoFormat::Mp4 => {
            eprintln!("{} Encoding MP4 (H.264)", stamp());
            encode_mp4(&video, args.fps, &args.output)?;
        }
        VideoFormat::PngFrames => {
            eprintln!("{} Writing PNG frames", stamp());
            write_png_frames(&video, &args.output)?;
        }
    }
    eprintln!(
        "{} Wrote {} ({}x{}, {} frames @ {} fps, seed {}) in {:.1}s",
        stamp(),
        args.output.display(),
        video.width,
        video.height,
        video.num_frames,
        args.fps,
        seed,
        t_run.elapsed().as_secs_f64(),
    );
    tracing::info!(target: DIAG, path = %args.output.display(), "wrote output");
    if let Some(b) = stats {
        report_mem(&b, ram_bytes, vram_bytes);
    }
    Ok(())
}

// --- MP4 encode (CLI-only) ---------------------------------------------------

/// Encode CTHW f32 `[-1, 1]` frames to an H.264 MP4 at `fps`. Each frame is
/// converted to RGB8, encoded with openh264, and muxed with the `mp4` crate.
fn encode_mp4(video: &WanVideo, fps: u32, out: &Path) -> Result<(), String> {
    use openh264::OpenH264API;
    use openh264::encoder::{Encoder, EncoderConfig};
    use openh264::formats::{RgbSliceU8, YUVBuffer};

    let (w, h, n) = (video.width, video.height, video.num_frames);
    // openh264 wants even dimensions for 4:2:0; our dims are /16 so this holds.
    // The default config (120 kbps target, frame-skip enabled) is built for
    // low-bandwidth real-time camera streams: at video resolution it produces
    // heavy block artifacts, and frame-skip can DROP frames -- but our muxer
    // emits exactly one sample per encode() call, so a dropped frame becomes a
    // broken empty sample. Configure for offline archival quality: a
    // resolution/fps-scaled bitrate, no frame skipping, and the real frame rate
    // so rate control budgets per second correctly.
    const BITS_PER_PIXEL: f64 = 0.2; // visually near-lossless for H.264 at these dims
    let bitrate = ((w as f64) * (h as f64) * (fps as f64) * BITS_PER_PIXEL).max(1_000_000.0) as u32;
    let cfg = EncoderConfig::new()
        .set_bitrate_bps(bitrate)
        .enable_skip_frame(false)
        .max_frame_rate(fps as f32);
    let mut enc = Encoder::with_api_config(OpenH264API::from_source(), cfg)
        .map_err(|e| format!("openh264 init: {e}"))?;

    let mut samples: Vec<(bool, Vec<u8>)> = Vec::with_capacity(n);
    let mut sps: Option<Vec<u8>> = None;
    let mut pps: Option<Vec<u8>> = None;
    let mut rgb = vec![0u8; w * h * 3];

    for t in 0..n {
        frame_to_rgb(video, t, &mut rgb);
        let yuv = YUVBuffer::from_rgb_source(RgbSliceU8::new(&rgb, (w, h)));
        let bitstream = enc
            .encode(&yuv)
            .map_err(|e| format!("openh264 encode frame {t}: {e}"))?;
        let annexb = bitstream.to_vec();
        let (sample, is_key) = annexb_to_avcc(&annexb, &mut sps, &mut pps);
        samples.push((is_key, sample));
    }

    let sps = sps.ok_or("openh264 produced no SPS")?;
    let pps = pps.ok_or("openh264 produced no PPS")?;
    mux_mp4(out, w, h, fps, &sps, &pps, &samples)
}

/// Mux length-prefixed (AVCC) H.264 samples into an MP4 file.
fn mux_mp4(
    out: &Path,
    w: usize,
    h: usize,
    fps: u32,
    sps: &[u8],
    pps: &[u8],
    samples: &[(bool, Vec<u8>)],
) -> Result<(), String> {
    use mp4::{
        AvcConfig, FourCC, MediaConfig, Mp4Config, Mp4Sample, Mp4Writer, TrackConfig, TrackType,
    };

    let config = Mp4Config {
        major_brand: FourCC::from(*b"isom"),
        minor_version: 512,
        compatible_brands: vec![
            FourCC::from(*b"isom"),
            FourCC::from(*b"iso2"),
            FourCC::from(*b"avc1"),
            FourCC::from(*b"mp41"),
        ],
        timescale: 1000,
    };
    let mut writer = Mp4Writer::write_start(std::io::Cursor::new(Vec::<u8>::new()), &config)
        .map_err(|e| format!("mp4 start: {e}"))?;
    // Track timescale = fps so each frame is exactly one tick (no rounding).
    writer
        .add_track(&TrackConfig {
            track_type: TrackType::Video,
            timescale: fps,
            language: "und".to_string(),
            media_conf: MediaConfig::AvcConfig(AvcConfig {
                width: w as u16,
                height: h as u16,
                seq_param_set: sps.to_vec(),
                pic_param_set: pps.to_vec(),
            }),
        })
        .map_err(|e| format!("mp4 add_track: {e}"))?;

    for (i, (is_key, bytes)) in samples.iter().enumerate() {
        writer
            .write_sample(
                1,
                &Mp4Sample {
                    start_time: i as u64,
                    duration: 1,
                    rendering_offset: 0,
                    is_sync: *is_key,
                    bytes: mp4::Bytes::copy_from_slice(bytes),
                },
            )
            .map_err(|e| format!("mp4 write_sample {i}: {e}"))?;
    }
    writer.write_end().map_err(|e| format!("mp4 end: {e}"))?;
    let data = writer.into_writer().into_inner();
    std::fs::write(out, &data).map_err(|e| format!("write {}: {e}", out.display()))
}

/// Fill `out` (RGB8, `w*h*3`) from frame `t` of the CTHW `[-1, 1]` tensor.
fn frame_to_rgb(video: &WanVideo, t: usize, out: &mut [u8]) {
    let (w, h, n) = (video.width, video.height, video.num_frames);
    let plane = n * h * w; // per-channel stride
    let base = t * h * w; // frame offset within a channel
    for px in 0..(w * h) {
        let idx = base + px;
        let o = px * 3;
        out[o] = to_u8(video.frames[idx]);
        out[o + 1] = to_u8(video.frames[plane + idx]);
        out[o + 2] = to_u8(video.frames[2 * plane + idx]);
    }
}

fn to_u8(v: f32) -> u8 {
    ((v.clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0).round() as u8
}

/// Write each decoded frame as `frame{n:03}.png` into directory `dir`. `frames`
/// is the CTHW `[-1, 1]` tensor; we gather each frame to channel-planar `[3, H,
/// W]` and reuse the shared `encode_png` (same `[-1,1]` mapping as the MP4 RGB
/// path). This is the codec-free view of the raw decode.
fn write_png_frames(video: &WanVideo, dir: &Path) -> Result<(), String> {
    let (w, h, n) = (video.width, video.height, video.num_frames);
    let per = h * w;
    let plane = n * per; // per-channel stride
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let mut chw = vec![0.0f32; 3 * per];
    for t in 0..n {
        let base = t * per;
        for c in 0..3 {
            chw[c * per..(c + 1) * per]
                .copy_from_slice(&video.frames[c * plane + base..c * plane + base + per]);
        }
        let png = encode_png(&chw, w as u32, h as u32)?;
        let p = dir.join(format!("frame{t:03}.png"));
        std::fs::write(&p, &png).map_err(|e| format!("write {}: {e}", p.display()))?;
    }
    Ok(())
}

/// Convert one Annex B access unit to AVCC (4-byte length-prefixed NALs),
/// capturing the first SPS/PPS into `sps`/`pps` (excluded from the sample, they
/// live in the avcC config box). Returns the sample bytes + whether it is an
/// IDR keyframe.
fn annexb_to_avcc(
    annexb: &[u8],
    sps: &mut Option<Vec<u8>>,
    pps: &mut Option<Vec<u8>>,
) -> (Vec<u8>, bool) {
    let mut out = Vec::with_capacity(annexb.len());
    let mut is_key = false;
    for nal in nal_units(annexb) {
        if nal.is_empty() {
            continue;
        }
        match nal[0] & 0x1F {
            7 => {
                if sps.is_none() {
                    *sps = Some(nal.to_vec());
                }
            }
            8 => {
                if pps.is_none() {
                    *pps = Some(nal.to_vec());
                }
            }
            nal_type => {
                if nal_type == 5 {
                    is_key = true;
                }
                out.extend_from_slice(&(nal.len() as u32).to_be_bytes());
                out.extend_from_slice(nal);
            }
        }
    }
    (out, is_key)
}

/// Split an Annex B buffer into NAL payloads (start codes stripped). Handles
/// both 3-byte (`00 00 01`) and 4-byte (`00 00 00 01`) start codes.
fn nal_units(annexb: &[u8]) -> Vec<&[u8]> {
    let n = annexb.len();
    let mut sc_ends = Vec::new(); // index just past each `00 00 01`
    let mut i = 0;
    while i + 3 <= n {
        if annexb[i] == 0 && annexb[i + 1] == 0 && annexb[i + 2] == 1 {
            sc_ends.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut units = Vec::with_capacity(sc_ends.len());
    for (k, &start) in sc_ends.iter().enumerate() {
        let mut end = if k + 1 < sc_ends.len() {
            sc_ends[k + 1] - 3 // back up over the next `00 00 01`
        } else {
            n
        };
        // Drop the single leading zero of a 4-byte start code.
        if end > start && annexb[end - 1] == 0 {
            end -= 1;
        }
        if end > start {
            units.push(&annexb[start..end]);
        }
    }
    units
}
