//! `thinfer generate face-swap`: swap a face from a source image into every
//! frame of an input video. Decodes the input (mp4 + openh264), runs the
//! HyperSwap pipeline (`thinfer_models::faceswap`) per frame on the GPU ONNX
//! executor, and re-encodes an mp4. The three ONNX models (SCRFD detector,
//! ArcFace embedder, HyperSwap swapper) download from the FaceFusion HF repos.

use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use clap::{Args, ValueEnum};
use thinfer_core::manifest::FileRef;
use thinfer_core::trace::DIAG;
use thinfer_models::faceswap::FaceSwapper;
use thinfer_models::faceswap::image::Image;
use thinfer_native::cache;

use super::video::{AudioPassthrough, annexb_to_avcc, mux_mp4};
use super::{
    PercentLogger, backend_for_stats, confirm_download, init_backend, parse_budget, report_mem,
};

/// HyperSwap checkpoint. 1a/1b/1c share architecture + speed; different visual
/// character (FaceFusion `models-3.3.0`).
#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum SwapModel {
    #[value(name = "hyperswap-1a")]
    Hyperswap1a,
    #[value(name = "hyperswap-1b")]
    Hyperswap1b,
    #[value(name = "hyperswap-1c")]
    Hyperswap1c,
}

impl SwapModel {
    fn file(self) -> FileRef {
        let path = match self {
            SwapModel::Hyperswap1a => "hyperswap_1a_256.onnx",
            SwapModel::Hyperswap1b => "hyperswap_1b_256.onnx",
            SwapModel::Hyperswap1c => "hyperswap_1c_256.onnx",
        };
        FileRef::new("facefusion/models-3.3.0", path)
    }
}

const SCRFD: FileRef = FileRef::new("deepghs/insightface", "buffalo_s/det_500m.onnx");
const ARCFACE: FileRef = FileRef::new("facefusion/models-3.0.0", "arcface_w600k_r50.onnx");

#[derive(Args)]
pub struct GenerateFaceSwap {
    /// Input video (mp4 / H.264). Every frame is processed.
    #[arg(long)]
    pub input_video: PathBuf,
    /// Source image (PNG/JPEG): the face to paste into the video.
    #[arg(long)]
    pub source_image: PathBuf,
    /// Output video file (.mp4).
    #[arg(long)]
    pub output: PathBuf,
    /// Swap model checkpoint.
    #[arg(long, value_enum, default_value_t = SwapModel::Hyperswap1a)]
    pub model: SwapModel,
    /// Host RAM budget. e.g. `8G`, `512M`, raw bytes. Face-swap streams the clip
    /// frame-by-frame, so peak RAM is a few frames regardless; the flag exists
    /// for parity with `generate video`/`image` and feeds the `[mem]` rollup.
    #[arg(long)]
    pub ram_budget: Option<String>,
    /// GPU VRAM budget (parity flag; see `--ram-budget`).
    #[arg(long)]
    pub vram_budget: Option<String>,
    /// Skip the TTY consent prompt and download missing model files.
    #[arg(long, default_value_t = false)]
    pub download_as_needed: bool,
}

pub async fn run_faceswap(args: GenerateFaceSwap) -> Result<(), String> {
    if args
        .output
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        != Some("mp4".to_string())
    {
        return Err("--output must be a .mp4 file".into());
    }

    // Resolve + download the three ONNX models.
    let files = [SCRFD, ARCFACE, args.model.file()];
    let (_resolved, missing) = cache::resolve_all(files.iter());
    if !missing.is_empty()
        && !confirm_download(&missing, args.download_as_needed).map_err(|e| e.to_string())?
    {
        return Err("declined download; rerun with --download-as-needed".into());
    }
    for f in &missing {
        let progress = Arc::new(PercentLogger::new(format!("{}/{}", f.repo, f.path)));
        cache::download_with_progress(f, Some(progress))
            .await
            .map_err(|e| format!("{e:?}"))?;
    }
    let path_of = |f: &FileRef| -> Result<PathBuf, String> {
        cache::resolve(f)
            .ok_or_else(|| format!("{}/{} not in cache after download", f.repo, f.path))
    };
    let scrfd_bytes = read(&path_of(&SCRFD)?)?;
    let arcface_bytes = read(&path_of(&ARCFACE)?)?;
    let hyperswap_bytes = read(&path_of(&args.model.file())?)?;

    let ram_bytes = parse_budget("--ram-budget", args.ram_budget.as_deref())?;
    let vram_bytes = parse_budget("--vram-budget", args.vram_budget.as_deref())?;

    let t_run = std::time::Instant::now();
    let stamp = move || format!("[{:6.1}s]", t_run.elapsed().as_secs_f64());

    // Source image.
    let source = load_image(&args.source_image)?;
    eprintln!("{} Loaded source {}x{}", stamp(), source.w, source.h);

    let backend = init_backend().await?;
    let stats = backend_for_stats(&backend);
    tracing::info!(
        target: DIAG,
        model = ?args.model,
        ram_budget = ram_bytes,
        vram_budget = vram_bytes,
        "face-swap start",
    );

    let swapper = FaceSwapper::load(backend, &scrfd_bytes, &arcface_bytes, &hyperswap_bytes)
        .await
        .map_err(|e| format!("load models: {e}"))?;

    let embedding = swapper
        .source_embedding(&source)
        .await
        .map_err(|e| format!("source embedding (no face in source image?): {e}"))?;
    eprintln!("{} Extracted source face embedding", stamp());

    // Stream decode -> swap -> encode one frame at a time (never materialize the
    // whole clip), remuxing the source audio track through verbatim.
    let (w, h, fps, n) = swap_video_streaming(
        &swapper,
        &embedding,
        &args.input_video,
        &args.output,
        &stamp,
    )
    .await?;

    eprintln!(
        "{} Wrote {} ({}x{}, {} frames @ {} fps) in {:.1}s",
        stamp(),
        args.output.display(),
        w,
        h,
        n,
        fps,
        t_run.elapsed().as_secs_f64(),
    );
    tracing::info!(target: DIAG, path = %args.output.display(), "wrote output");
    if let Some(b) = stats {
        report_mem(&b, ram_bytes, vram_bytes);
    }
    Ok(())
}

fn read(p: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(p).map_err(|e| format!("read {}: {e}", p.display()))
}

/// Decode a PNG/JPEG source image into an RGB `Image`.
fn load_image(path: &Path) -> Result<Image, String> {
    let dynimg = image::open(path).map_err(|e| format!("decode {}: {e}", path.display()))?;
    let rgb = dynimg.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    Ok(Image::from_rgb8(w, h, rgb.as_raw()))
}

// --- Streaming swap (mp4 demux -> per-frame swap -> mp4 mux) -----------------

/// Decode the input, swap each frame, and encode the output one frame at a time
/// so only a few frames are ever in RAM (the 4K clip otherwise materialized
/// ~30GB). The source AAC audio track is remuxed through verbatim. Returns the
/// output `(width, height, fps, frame_count)`.
async fn swap_video_streaming(
    swapper: &FaceSwapper,
    embedding: &[f32],
    input: &Path,
    output: &Path,
    stamp: &dyn Fn() -> String,
) -> Result<(usize, usize, u32, usize), String> {
    use mp4::{Mp4Reader, TrackType};
    use openh264::OpenH264API;
    use openh264::decoder::{Decoder, DecoderConfig, Flush};

    let file = std::fs::File::open(input).map_err(|e| format!("open {}: {e}", input.display()))?;
    let size = file.metadata().map_err(|e| e.to_string())?.len();
    let mut mp4 = Mp4Reader::read_header(BufReader::new(file), size)
        .map_err(|e| format!("mp4 parse: {e}"))?;

    // Pull the video track's metadata before any mutable read borrow.
    let (track_id, sps, pps, fps) = {
        let track = mp4
            .tracks()
            .iter()
            .find(|(_, t)| matches!(t.track_type(), Ok(TrackType::Video)))
            .ok_or("no video track in input")?
            .1;
        let sps = track
            .sequence_parameter_set()
            .map_err(|e| format!("missing SPS: {e}"))?
            .to_vec();
        let pps = track
            .picture_parameter_set()
            .map_err(|e| format!("missing PPS: {e}"))?
            .to_vec();
        let fr = track.frame_rate();
        let fps = if fr.is_finite() && fr > 0.0 {
            fr.round() as u32
        } else {
            24
        };
        (track.track_id(), sps, pps, fps.max(1))
    };
    let count = mp4
        .tracks()
        .get(&track_id)
        .map(|t| t.sample_count())
        .unwrap_or(0);

    // Audio is read up front (compressed AAC, small) for verbatim passthrough.
    let audio = extract_audio(&mut mp4)?;
    match &audio {
        Some(a) => eprintln!(
            "{} Audio: {} AAC sample(s), passthrough",
            stamp(),
            a.samples.len()
        ),
        None => eprintln!("{} Audio: video-only output", stamp()),
    }

    // Annex B SPS/PPS prefix, prepended to every access unit (decoders tolerate
    // repeats; keeps the first IDR self-contained).
    let mut prefix = Vec::new();
    for nal in [&sps, &pps] {
        prefix.extend_from_slice(&[0, 0, 0, 1]);
        prefix.extend_from_slice(nal);
    }

    // Disable per-decode auto-flush: with B-frames (High/Main profile) openh264
    // reorders internally and holds the reorder depth, so frames come ready in
    // display order and the tail is drained with flush_remaining after the last
    // AU (the crate's recommended B-frame-safe pattern).
    let mut decoder = Decoder::with_api_config(
        OpenH264API::from_source(),
        DecoderConfig::new().flush_after_decode(Flush::NoFlush),
    )
    .map_err(|e| format!("openh264 decoder init: {e}"))?;

    eprintln!("{} Streaming {count} frames at {fps} fps", stamp());
    let mut sink = Mp4VideoSink::new(fps);
    let mut rgb: Vec<u8> = Vec::new();
    let mut done = 0usize;
    for sid in 1..=count {
        let Some(sample) = mp4
            .read_sample(track_id, sid)
            .map_err(|e| format!("read sample {sid}: {e}"))?
        else {
            continue;
        };
        let mut au = prefix.clone();
        avcc_to_annexb(&sample.bytes, &mut au);
        if let Some(yuv) = decoder
            .decode(&au)
            .map_err(|e| format!("decode sample {sid}: {e}"))?
        {
            swap_and_push(swapper, embedding, &yuv, &mut sink, &mut rgb).await?;
            done += 1;
            if done.is_multiple_of(10) {
                eprintln!("{} Swapped frame {done}/{count}", stamp());
            }
        }
    }
    for yuv in decoder
        .flush_remaining()
        .map_err(|e| format!("decode flush: {e}"))?
    {
        swap_and_push(swapper, embedding, &yuv, &mut sink, &mut rgb).await?;
        done += 1;
    }
    if done == 0 {
        return Err("input video has no decodable frames".into());
    }

    eprintln!(
        "{} Encoding MP4 (H.264){}",
        stamp(),
        if audio.is_some() { " + audio" } else { "" }
    );
    let (out_w, out_h, n, enc_sps, enc_pps, samples) = sink.finish()?;
    mux_mp4(
        output, out_w, out_h, fps, &enc_sps, &enc_pps, &samples, audio,
    )?;
    Ok((out_w, out_h, fps, n))
}

/// Decode one YUV frame to an `Image`, swap, and feed it to the encoder sink.
async fn swap_and_push(
    swapper: &FaceSwapper,
    embedding: &[f32],
    yuv: &openh264::decoder::DecodedYUV<'_>,
    sink: &mut Mp4VideoSink,
    rgb: &mut Vec<u8>,
) -> Result<(), String> {
    use openh264::formats::YUVSource;
    let (w, h) = yuv.dimensions();
    rgb.resize(w * h * 3, 0);
    yuv.write_rgb8(rgb);
    let img = Image::from_rgb8(w, h, rgb);
    let swapped = swapper
        .swap_frame(&img, embedding)
        .await
        .map_err(|e| format!("swap frame: {e}"))?;
    sink.push(&swapped)
}

/// Read the input's audio track (if any) for verbatim remux. Returns `None`
/// (audio dropped, with a warning) when there is no audio, the codec is not AAC,
/// or the AAC object type cannot be remuxed losslessly. The `mp4` crate models
/// AAC as only `{profile, freq_index, chan_conf}` and discards the rest of the
/// AudioSpecificConfig, so HE-AAC (SBR/PS) extension data is lost: those streams
/// would produce a broken esds, so we drop them rather than emit unplayable
/// audio. The common phone-clip case (AAC-LC) round-trips cleanly. The track id
/// + config are copied out before the mutable per-sample reads (borrow checker).
fn extract_audio<R: std::io::Read + std::io::Seek>(
    mp4: &mut mp4::Mp4Reader<R>,
) -> Result<Option<AudioPassthrough>, String> {
    use mp4::{AudioObjectType, MediaType, TrackType};

    let Some(id) = mp4
        .tracks()
        .iter()
        .find_map(|(&id, t)| matches!(t.track_type(), Ok(TrackType::Audio)).then_some(id))
    else {
        return Ok(None); // no audio track at all
    };

    let (config, timescale, count) = {
        let t = mp4.tracks().get(&id).ok_or("audio track vanished")?;
        if !matches!(t.media_type(), Ok(MediaType::AAC)) {
            eprintln!(
                "[face-swap] dropping audio: input codec is not AAC (only AAC passthrough is supported)"
            );
            return Ok(None);
        }
        let profile = t
            .audio_profile()
            .map_err(|e| format!("audio profile: {e}"))?;
        // Only the object types whose config is fully `{profile,freq_index,
        // chan_conf}` survive the round-trip; HE-AAC and friends do not.
        if !matches!(
            profile,
            AudioObjectType::AacMain
                | AudioObjectType::AacLowComplexity
                | AudioObjectType::AacScalableSampleRate
                | AudioObjectType::AacLongTermPrediction
        ) {
            eprintln!(
                "[face-swap] dropping audio: cannot losslessly remux {profile:?} (e.g. HE-AAC); \
                 re-encode the input audio to AAC-LC to keep it"
            );
            return Ok(None);
        }
        let config = mp4::AacConfig {
            bitrate: t.bitrate(),
            profile,
            freq_index: t
                .sample_freq_index()
                .map_err(|e| format!("audio freq index: {e}"))?,
            chan_conf: t
                .channel_config()
                .map_err(|e| format!("audio channels: {e}"))?,
        };
        (config, t.timescale(), t.sample_count())
    };

    let mut samples = Vec::with_capacity(count as usize);
    for sid in 1..=count {
        if let Some(s) = mp4
            .read_sample(id, sid)
            .map_err(|e| format!("read audio sample {sid}: {e}"))?
        {
            samples.push(s);
        }
    }
    Ok(Some(AudioPassthrough {
        config,
        timescale,
        samples,
    }))
}

/// Convert AVCC (4-byte length-prefixed NALs) to Annex B (start-code prefixed),
/// appending to `out`.
fn avcc_to_annexb(avcc: &[u8], out: &mut Vec<u8>) {
    let mut i = 0;
    while i + 4 <= avcc.len() {
        let len = u32::from_be_bytes([avcc[i], avcc[i + 1], avcc[i + 2], avcc[i + 3]]) as usize;
        i += 4;
        if i + len > avcc.len() {
            break;
        }
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&avcc[i..i + len]);
        i += len;
    }
}

// --- Output encode (streaming RGB8 -> H.264 samples) ------------------------

/// Streaming H.264 encoder sink: each swapped frame is encoded immediately (one
/// frame in RAM at a time) and accumulated as a compressed AVCC sample. The
/// output resolution is fixed on the first frame, downscaled to fit openh264's
/// encoder cap (max 3840 long edge / 2160 short edge), aspect-preserved.
struct Mp4VideoSink {
    fps: u32,
    enc: Option<openh264::encoder::Encoder>,
    out_w: usize,
    out_h: usize,
    samples: Vec<(bool, Vec<u8>)>,
    sps: Option<Vec<u8>>,
    pps: Option<Vec<u8>>,
}

impl Mp4VideoSink {
    fn new(fps: u32) -> Self {
        Self {
            fps,
            enc: None,
            out_w: 0,
            out_h: 0,
            samples: Vec::new(),
            sps: None,
            pps: None,
        }
    }

    /// Encode one swapped frame. The first call fixes the output resolution and
    /// builds the encoder; the input resolution is assumed constant thereafter.
    fn push(&mut self, frame: &Image) -> Result<(), String> {
        use openh264::OpenH264API;
        use openh264::encoder::{BitRate, Encoder, EncoderConfig, FrameRate};
        use openh264::formats::{RgbSliceU8, YUVBuffer};

        if self.enc.is_none() {
            let (ow, oh) = fit_encode_dims(frame.w, frame.h);
            if (ow, oh) != (frame.w, frame.h) {
                eprintln!(
                    "[face-swap] downscaling output {}x{} -> {ow}x{oh} (openh264 encoder cap)",
                    frame.w, frame.h
                );
            }
            const BITS_PER_PIXEL: f64 = 0.2;
            let bitrate = ((ow as f64) * (oh as f64) * (self.fps as f64) * BITS_PER_PIXEL)
                .max(1_000_000.0) as u32;
            let cfg = EncoderConfig::new()
                .bitrate(BitRate::from_bps(bitrate))
                .skip_frames(false)
                .max_frame_rate(FrameRate::from_hz(self.fps as f32));
            self.enc = Some(
                Encoder::with_api_config(OpenH264API::from_source(), cfg)
                    .map_err(|e| format!("openh264 init: {e}"))?,
            );
            (self.out_w, self.out_h) = (ow, oh);
        }

        let rgb = if (self.out_w, self.out_h) == (frame.w, frame.h) {
            frame.to_rgb8()
        } else {
            frame.resize(self.out_w, self.out_h).to_rgb8()
        };
        let yuv = YUVBuffer::from_rgb_source(RgbSliceU8::new(&rgb, (self.out_w, self.out_h)));
        let bitstream = self
            .enc
            .as_mut()
            .unwrap()
            .encode(&yuv)
            .map_err(|e| format!("openh264 encode frame {}: {e}", self.samples.len()))?;
        let (sample, is_key) = annexb_to_avcc(&bitstream.to_vec(), &mut self.sps, &mut self.pps);
        self.samples.push((is_key, sample));
        Ok(())
    }

    /// Output `(width, height, frame_count, sps, pps, samples)`.
    #[allow(clippy::type_complexity)]
    fn finish(
        self,
    ) -> Result<(usize, usize, usize, Vec<u8>, Vec<u8>, Vec<(bool, Vec<u8>)>), String> {
        let sps = self.sps.ok_or("openh264 produced no SPS")?;
        let pps = self.pps.ok_or("openh264 produced no PPS")?;
        let n = self.samples.len();
        Ok((self.out_w, self.out_h, n, sps, pps, self.samples))
    }
}

/// Largest aspect-preserving size within openh264's encoder cap (max 3840 on the
/// long edge, 2160 on the short edge), rounded to even (4:2:0 needs even dims).
/// Returns the input (even-clamped) when it already fits.
fn fit_encode_dims(w: usize, h: usize) -> (usize, usize) {
    const MAX_LONG: f64 = 3840.0;
    const MAX_SHORT: f64 = 2160.0;
    let (long, short) = (w.max(h) as f64, w.min(h) as f64);
    let s = (MAX_LONG / long).min(MAX_SHORT / short).min(1.0);
    let even = |x: f64| ((x.round() as usize) & !1).max(2);
    (even(w as f64 * s), even(h as f64 * s))
}
