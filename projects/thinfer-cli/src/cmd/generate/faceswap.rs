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

use super::{PercentLogger, confirm_download, init_backend};

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

    let t_run = std::time::Instant::now();
    let stamp = move || format!("[{:6.1}s]", t_run.elapsed().as_secs_f64());

    // Source image.
    let source = load_image(&args.source_image)?;
    eprintln!("{} Loaded source {}x{}", stamp(), source.w, source.h);

    // Decode input video.
    eprintln!("{} Decoding input video", stamp());
    let (frames, fps) = decode_video(&args.input_video)?;
    if frames.is_empty() {
        return Err("input video has no decodable frames".into());
    }
    eprintln!(
        "{} Decoded {} frames at {} fps ({}x{})",
        stamp(),
        frames.len(),
        fps,
        frames[0].w,
        frames[0].h
    );

    let backend = init_backend().await?;
    tracing::info!(
        target: DIAG,
        model = ?args.model,
        frames = frames.len(),
        fps,
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

    let (w, h) = (frames[0].w, frames[0].h);
    let mut out_frames: Vec<Vec<u8>> = Vec::with_capacity(frames.len());
    let n = frames.len();
    for (i, frame) in frames.iter().enumerate() {
        let swapped = swapper
            .swap_frame(frame, &embedding)
            .await
            .map_err(|e| format!("swap frame {i}: {e}"))?;
        out_frames.push(swapped.to_rgb8());
        if (i + 1) % 10 == 0 || i + 1 == n {
            eprintln!("{} Swapped frame {}/{}", stamp(), i + 1, n);
        }
    }

    eprintln!("{} Encoding MP4 (H.264)", stamp());
    encode_rgb8_mp4(&out_frames, w, h, fps, &args.output)?;
    eprintln!(
        "{} Wrote {} ({}x{}, {} frames @ {} fps) in {:.1}s",
        stamp(),
        args.output.display(),
        w,
        h,
        out_frames.len(),
        fps,
        t_run.elapsed().as_secs_f64(),
    );
    tracing::info!(target: DIAG, path = %args.output.display(), "wrote output");
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

// --- Video decode (mp4 demux + openh264) ------------------------------------

/// Decode an mp4/H.264 video into RGB frames. Returns the frames + integer fps.
fn decode_video(path: &Path) -> Result<(Vec<Image>, u32), String> {
    use mp4::{Mp4Reader, TrackType};
    use openh264::OpenH264API;
    use openh264::decoder::Decoder;
    use openh264::formats::YUVSource;

    let file = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
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

    // Annex B SPS/PPS prefix, prepended to every access unit (decoders tolerate
    // repeats; keeps the first IDR self-contained).
    let mut prefix = Vec::new();
    for nal in [&sps, &pps] {
        prefix.extend_from_slice(&[0, 0, 0, 1]);
        prefix.extend_from_slice(nal);
    }

    let mut decoder = Decoder::new().map_err(|e| format!("openh264 decoder init: {e}"))?;
    let _ = OpenH264API::from_source(); // ensure the source build links
    let mut frames = Vec::with_capacity(count as usize);
    let mut rgb: Vec<u8> = Vec::new();
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
            let (w, h) = yuv.dimensions();
            rgb.resize(w * h * 3, 0);
            yuv.write_rgb8(&mut rgb);
            frames.push(Image::from_rgb8(w, h, &rgb));
        }
    }
    Ok((frames, fps))
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

// --- Output encode (RGB8 frames -> mp4) -------------------------------------

/// Encode interleaved-RGB8 frames to an H.264 mp4 at `fps`. Mirrors
/// `video::encode_mp4` but takes ready RGB frames; reuses the shared muxer.
fn encode_rgb8_mp4(
    frames: &[Vec<u8>],
    w: usize,
    h: usize,
    fps: u32,
    out: &Path,
) -> Result<(), String> {
    use openh264::OpenH264API;
    use openh264::encoder::{Encoder, EncoderConfig};
    use openh264::formats::{RgbSliceU8, YUVBuffer};

    const BITS_PER_PIXEL: f64 = 0.2;
    let bitrate = ((w as f64) * (h as f64) * (fps as f64) * BITS_PER_PIXEL).max(1_000_000.0) as u32;
    let cfg = EncoderConfig::new()
        .set_bitrate_bps(bitrate)
        .enable_skip_frame(false)
        .max_frame_rate(fps as f32);
    let mut enc = Encoder::with_api_config(OpenH264API::from_source(), cfg)
        .map_err(|e| format!("openh264 init: {e}"))?;

    let mut samples: Vec<(bool, Vec<u8>)> = Vec::with_capacity(frames.len());
    let mut sps: Option<Vec<u8>> = None;
    let mut pps: Option<Vec<u8>> = None;
    for (t, rgb) in frames.iter().enumerate() {
        let yuv = YUVBuffer::from_rgb_source(RgbSliceU8::new(rgb, (w, h)));
        let bitstream = enc
            .encode(&yuv)
            .map_err(|e| format!("openh264 encode frame {t}: {e}"))?;
        let (sample, is_key) =
            super::video::annexb_to_avcc(&bitstream.to_vec(), &mut sps, &mut pps);
        samples.push((is_key, sample));
    }
    let sps = sps.ok_or("openh264 produced no SPS")?;
    let pps = pps.ok_or("openh264 produced no PPS")?;
    super::video::mux_mp4(out, w, h, fps, &sps, &pps, &samples)
}
