//! Container encode/decode: H.264 (openh264) + MP4 mux, the codec-free PNG
//! frame dump, and the face-swap streaming decode -> swap -> encode loop. This
//! is the one place the openh264/mp4 deps live (web uses the browser's
//! WebCodecs instead). Per-frame progress goes through a [`ProgressSink`];
//! descriptive lines go through [`ProgressSink::note`] with their original text.

use std::io::BufReader;
use std::path::Path;

use thinfer_models::faceswap::FaceSwapper;
use thinfer_models::faceswap::image::Image;
use thinfer_models::wan::pipeline::WanVideo;
use thinfer_models::z_image::pipeline::encode_png;

use crate::progress::{ProgressSink, Stage};

// visually near-lossless H.264 at our dims.
const BITS_PER_PIXEL: f64 = 0.2;

/// Decode a PNG/JPEG source image into an RGB [`Image`].
pub fn load_image(path: &Path) -> Result<Image, String> {
    let dynimg = image::open(path).map_err(|e| format!("decode {}: {e}", path.display()))?;
    let rgb = dynimg.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    Ok(Image::from_rgb8(w, h, rgb.as_raw()))
}

// --- t2v MP4 encode ----------------------------------------------------------

/// Encode CTHW f32 `[-1, 1]` frames to an H.264 MP4 at `fps`.
pub fn encode_mp4(video: &WanVideo, fps: u32, out: &Path) -> Result<(), String> {
    encode_mp4_with_audio(video, fps, None, out)
}

/// Encode CTHW f32 `[-1, 1]` frames to an H.264 MP4 at `fps`, optionally muxing
/// a pre-encoded AAC audio track (the LTX joint-AV path). The CTHW container is
/// reused as a plain frame carrier; the audio rides alongside, faststart-muxed.
pub fn encode_mp4_with_audio(
    video: &WanVideo,
    fps: u32,
    audio: Option<AudioPassthrough>,
    out: &Path,
) -> Result<(), String> {
    use openh264::OpenH264API;
    use openh264::encoder::{BitRate, Encoder, EncoderConfig, FrameRate};
    use openh264::formats::{RgbSliceU8, YUVBuffer};

    let (w, h, n) = (video.width, video.height, video.num_frames);
    // openh264 wants even dims for 4:2:0; ours are /16. The default config
    // (120 kbps, frame-skip on) is built for low-bandwidth camera streams and
    // would drop frames -- our muxer emits one sample per encode(), so a drop
    // becomes a broken sample. Configure for offline archival quality.
    let bitrate = ((w as f64) * (h as f64) * (fps as f64) * BITS_PER_PIXEL).max(1_000_000.0) as u32;
    let cfg = EncoderConfig::new()
        .bitrate(BitRate::from_bps(bitrate))
        .skip_frames(false)
        .max_frame_rate(FrameRate::from_hz(fps as f32));
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
        let (sample, is_key) = annexb_to_avcc(&bitstream.to_vec(), &mut sps, &mut pps);
        samples.push((is_key, sample));
    }

    let sps = sps.ok_or("openh264 produced no SPS")?;
    let pps = pps.ok_or("openh264 produced no PPS")?;
    mux_mp4(out, w, h, fps, &sps, &pps, &samples, audio)
}

/// Encode planar stereo f32 PCM `[2, n]` (channel-major, `[-1, 1]`) to AAC-LC
/// and package it as an [`AudioPassthrough`] ready for [`mux_mp4`]. The LTX
/// vocoder emits exactly this layout (left then right). One AAC packet = 1024
/// samples/channel; the audio track timescale is the sample rate, so each
/// packet spans 1024 ticks. A trailing zero frame flushes the encoder's
/// lookahead so the tail is not truncated.
pub fn encode_aac_stereo(wav_planar: &[f32], sample_rate: u32) -> Result<AudioPassthrough, String> {
    use fdk_aac::enc::{
        AudioObjectType, BitRate, ChannelMode, Encoder as AacEncoder, EncoderParams, Transport,
    };

    const FRAME: usize = 1024; // AAC-LC samples per channel per packet.
    let per_ch = wav_planar.len() / 2;
    if per_ch == 0 {
        return Err("empty audio".into());
    }
    // ~128 kbps stereo: transparent for speech/foley at 48kHz.
    let bitrate = 128_000;
    let enc = AacEncoder::new(EncoderParams {
        bit_rate: BitRate::Cbr(bitrate),
        sample_rate,
        transport: Transport::Raw,
        channels: ChannelMode::Stereo,
        audio_object_type: AudioObjectType::Mpeg4LowComplexity,
    })
    .map_err(|e| format!("aac encoder init: {e}"))?;

    // Interleave L/R to i16, zero-padded to a whole number of frames plus one
    // flush frame.
    let frames = per_ch.div_ceil(FRAME) + 1;
    let mut pcm = vec![0i16; frames * FRAME * 2];
    let to_i16 = |v: f32| (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
    for i in 0..per_ch {
        pcm[i * 2] = to_i16(wav_planar[i]);
        pcm[i * 2 + 1] = to_i16(wav_planar[per_ch + i]);
    }

    let mut samples = Vec::with_capacity(frames);
    let mut out_buf = vec![0u8; 8192];
    let mut pos = 0usize; // interleaved-sample cursor.
    while pos < pcm.len() {
        let info = enc
            .encode(&pcm[pos..], &mut out_buf)
            .map_err(|e| format!("aac encode: {e}"))?;
        if info.input_consumed == 0 && info.output_size == 0 {
            break; // encoder made no progress: done.
        }
        pos += info.input_consumed;
        if info.output_size > 0 {
            let start = (samples.len() * FRAME) as u64;
            samples.push(mp4::Mp4Sample {
                start_time: start,
                duration: FRAME as u32,
                rendering_offset: 0,
                is_sync: true,
                bytes: mp4::Bytes::copy_from_slice(&out_buf[..info.output_size]),
            });
        }
    }
    if samples.is_empty() {
        return Err("aac encoder produced no packets".into());
    }
    Ok(AudioPassthrough {
        config: mp4::AacConfig {
            bitrate,
            profile: mp4::AudioObjectType::AacLowComplexity,
            freq_index: freq_index_for(sample_rate)?,
            chan_conf: mp4::ChannelConfig::Stereo,
        },
        timescale: sample_rate,
        samples,
    })
}

/// Map a sample rate to the MP4/AAC `SampleFreqIndex`. Only the rates the LTX
/// vocoder emits are needed (48kHz); extend if other rates appear.
fn freq_index_for(sample_rate: u32) -> Result<mp4::SampleFreqIndex, String> {
    match sample_rate {
        48000 => Ok(mp4::SampleFreqIndex::Freq48000),
        44100 => Ok(mp4::SampleFreqIndex::Freq44100),
        24000 => Ok(mp4::SampleFreqIndex::Freq24000),
        16000 => Ok(mp4::SampleFreqIndex::Freq16000),
        other => Err(format!("unsupported audio sample rate {other}")),
    }
}

/// Write each decoded frame as `frame{n:03}.png` into directory `dir`. The
/// codec-free view of the raw decode.
pub fn write_png_frames(video: &WanVideo, dir: &Path) -> Result<(), String> {
    let (w, h, n) = (video.width, video.height, video.num_frames);
    let per = h * w;
    let plane = n * per;
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

/// Fill `out` (RGB8, `w*h*3`) from frame `t` of the CTHW `[-1, 1]` tensor.
fn frame_to_rgb(video: &WanVideo, t: usize, out: &mut [u8]) {
    let (w, h, n) = (video.width, video.height, video.num_frames);
    let plane = n * h * w;
    let base = t * h * w;
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

// --- MP4 mux (shared by t2v + face-swap) -------------------------------------

/// An audio track copied verbatim from an input container, remuxed without
/// re-encoding. `samples` carry their own `start_time`/`duration` in
/// `timescale` units.
pub struct AudioPassthrough {
    pub config: mp4::AacConfig,
    pub timescale: u32,
    pub samples: Vec<mp4::Mp4Sample>,
}

/// Mux length-prefixed (AVCC) H.264 video samples into an MP4 file, optionally
/// adding a verbatim audio track.
pub fn mux_mp4(
    out: &Path,
    w: usize,
    h: usize,
    fps: u32,
    sps: &[u8],
    pps: &[u8],
    samples: &[(bool, Vec<u8>)],
    audio: Option<AudioPassthrough>,
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

    if let Some(a) = &audio {
        writer
            .add_track(&TrackConfig {
                track_type: TrackType::Audio,
                timescale: a.timescale,
                language: "und".to_string(),
                media_conf: MediaConfig::AacConfig(a.config.clone()),
            })
            .map_err(|e| format!("mp4 add audio track: {e}"))?;
    }

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
    if let Some(a) = audio {
        for (i, s) in a.samples.iter().enumerate() {
            writer
                .write_sample(2, s)
                .map_err(|e| format!("mp4 write audio sample {i}: {e}"))?;
        }
    }
    writer.write_end().map_err(|e| format!("mp4 end: {e}"))?;
    let data = faststart(writer.into_writer().into_inner());
    std::fs::write(out, &data).map_err(|e| format!("write {}: {e}", out.display()))
}

/// Relocate the `moov` box ahead of `mdat` ("faststart") so the file plays
/// progressively and seeks reliably in browsers / strict players. Chunk offsets
/// in `stco`/`co64` are absolute, so they shift by the inserted `moov` length.
/// Any parse surprise returns the input untouched (a valid moov-at-end file).
fn faststart(data: Vec<u8>) -> Vec<u8> {
    let Some(boxes) = top_level_boxes(&data) else {
        return data;
    };
    let moov = boxes.iter().find(|b| &b.kind == b"moov");
    let mdat = boxes.iter().find(|b| &b.kind == b"mdat");
    let (Some(moov), Some(mdat)) = (moov, mdat) else {
        return data;
    };
    if moov.start < mdat.start {
        return data; // already faststart
    }
    let shift = moov.len as u64;
    let mut moov_bytes = data[moov.start..moov.start + moov.len].to_vec();
    if !patch_chunk_offsets(&mut moov_bytes, shift) {
        return data;
    }
    // ftyp first, then the relocated moov, then everything else in order.
    let mut out = Vec::with_capacity(data.len());
    for b in &boxes {
        if &b.kind == b"moov" {
            continue;
        }
        out.extend_from_slice(&data[b.start..b.start + b.len]);
        if &b.kind == b"ftyp" {
            out.extend_from_slice(&moov_bytes);
        }
    }
    out
}

/// A top-level (or sibling) MP4 box: its 4-byte type and byte range.
struct Mp4Box {
    kind: [u8; 4],
    start: usize,
    len: usize,
}

/// Parse the boxes at the top level of `data` (size+type framing, with 64-bit
/// largesize and size-0-to-EOF handled). `None` on any malformed length.
fn top_level_boxes(data: &[u8]) -> Option<Vec<Mp4Box>> {
    let mut boxes = Vec::new();
    let mut p = 0;
    while p + 8 <= data.len() {
        let size = u32::from_be_bytes(data[p..p + 4].try_into().ok()?);
        let kind: [u8; 4] = data[p + 4..p + 8].try_into().ok()?;
        let len = match size {
            0 => data.len() - p,
            1 => u64::from_be_bytes(data.get(p + 8..p + 16)?.try_into().ok()?) as usize,
            n => n as usize,
        };
        if len < 8 || p + len > data.len() {
            return None;
        }
        boxes.push(Mp4Box {
            kind,
            start: p,
            len,
        });
        p += len;
    }
    Some(boxes)
}

/// Recursively add `shift` to every `stco`/`co64` chunk offset inside `buf`
/// (a `moov` box payload, walked through its container boxes). Returns false on
/// a malformed box so the caller can bail out safely.
fn patch_chunk_offsets(buf: &mut [u8], shift: u64) -> bool {
    let mut p = 0;
    while p + 8 <= buf.len() {
        let size = u32::from_be_bytes(buf[p..p + 4].try_into().unwrap());
        let kind: [u8; 4] = buf[p + 4..p + 8].try_into().unwrap();
        let (len, header) = match size {
            0 => (buf.len() - p, 8usize),
            1 => {
                let Some(b) = buf.get(p + 8..p + 16) else {
                    return false;
                };
                (u64::from_be_bytes(b.try_into().unwrap()) as usize, 16usize)
            }
            n => (n as usize, 8usize),
        };
        if len < header || p + len > buf.len() {
            return false;
        }
        match &kind {
            // Container boxes on the path to the sample tables: recurse.
            b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" => {
                if !patch_chunk_offsets(&mut buf[p + header..p + len], shift) {
                    return false;
                }
            }
            // stco: full box (4) + entry_count (4) + entry_count * u32 offsets.
            b"stco" => {
                let base = p + header + 4;
                let Some(cnt_bytes) = buf.get(base..base + 4) else {
                    return false;
                };
                let count = u32::from_be_bytes(cnt_bytes.try_into().unwrap()) as usize;
                let mut q = base + 4;
                for _ in 0..count {
                    let Some(v) = buf.get(q..q + 4) else {
                        return false;
                    };
                    let nv = (u32::from_be_bytes(v.try_into().unwrap()) as u64 + shift) as u32;
                    buf[q..q + 4].copy_from_slice(&nv.to_be_bytes());
                    q += 4;
                }
            }
            // co64: same shape with 64-bit offsets.
            b"co64" => {
                let base = p + header + 4;
                let Some(cnt_bytes) = buf.get(base..base + 4) else {
                    return false;
                };
                let count = u32::from_be_bytes(cnt_bytes.try_into().unwrap()) as usize;
                let mut q = base + 4;
                for _ in 0..count {
                    let Some(v) = buf.get(q..q + 8) else {
                        return false;
                    };
                    let nv = u64::from_be_bytes(v.try_into().unwrap()) + shift;
                    buf[q..q + 8].copy_from_slice(&nv.to_be_bytes());
                    q += 8;
                }
            }
            _ => {}
        }
        p += len;
    }
    true
}

// --- Annex B <-> AVCC --------------------------------------------------------

/// Convert one Annex B access unit to AVCC (4-byte length-prefixed NALs),
/// capturing the first SPS/PPS into `sps`/`pps` (they live in the avcC box).
/// Returns the sample bytes + whether it is an IDR keyframe.
pub fn annexb_to_avcc(
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
/// both 3-byte and 4-byte start codes.
fn nal_units(annexb: &[u8]) -> Vec<&[u8]> {
    let n = annexb.len();
    let mut sc_ends = Vec::new();
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
            sc_ends[k + 1] - 3
        } else {
            n
        };
        if end > start && annexb[end - 1] == 0 {
            end -= 1;
        }
        if end > start {
            units.push(&annexb[start..end]);
        }
    }
    units
}

/// Convert AVCC (4-byte length-prefixed NALs) to Annex B, appending to `out`.
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

// --- Face-swap streaming (mp4 demux -> per-frame swap -> mp4 mux) -------------

/// Decode the input, swap each frame, and encode the output one frame at a time
/// so only a few frames are ever in RAM (a 4K clip otherwise materializes
/// ~30GB). The source AAC audio track is remuxed verbatim. Returns the output
/// `(width, height, fps, frame_count)`.
pub async fn swap_video_streaming(
    swapper: &FaceSwapper,
    embedding: &[f32],
    input: &Path,
    output: &Path,
    sink: &dyn ProgressSink,
) -> Result<(usize, usize, u32, usize), String> {
    use mp4::{Mp4Reader, TrackType};
    use openh264::OpenH264API;
    use openh264::decoder::{Decoder, DecoderConfig, Flush};

    let file = std::fs::File::open(input).map_err(|e| format!("open {}: {e}", input.display()))?;
    let size = file.metadata().map_err(|e| e.to_string())?.len();
    let mut mp4 = Mp4Reader::read_header(BufReader::new(file), size)
        .map_err(|e| format!("mp4 parse: {e}"))?;

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

    let audio = extract_audio(&mut mp4)?;
    match &audio {
        Some(a) => sink.note(&format!(
            "Audio: {} AAC sample(s), passthrough",
            a.samples.len()
        )),
        None => sink.note("Audio: video-only output"),
    }

    // Annex B SPS/PPS prefix, prepended to every access unit.
    let mut prefix = Vec::new();
    for nal in [&sps, &pps] {
        prefix.extend_from_slice(&[0, 0, 0, 1]);
        prefix.extend_from_slice(nal);
    }

    // Disable per-decode auto-flush: with B-frames openh264 reorders internally,
    // so frames come ready in display order and the tail is drained with
    // flush_remaining after the last AU (the B-frame-safe pattern).
    let mut decoder = Decoder::with_api_config(
        OpenH264API::from_source(),
        DecoderConfig::new().flush_after_decode(Flush::NoFlush),
    )
    .map_err(|e| format!("openh264 decoder init: {e}"))?;

    sink.note(&format!("Streaming {count} frames at {fps} fps"));
    let mut video_sink = Mp4VideoSink::new(fps);
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
            swap_and_push(swapper, embedding, &yuv, &mut video_sink, &mut rgb).await?;
            done += 1;
            sink.stage(Stage::FrameSwapped {
                done: done as u32,
                total: count,
            });
        }
    }
    for yuv in decoder
        .flush_remaining()
        .map_err(|e| format!("decode flush: {e}"))?
    {
        swap_and_push(swapper, embedding, &yuv, &mut video_sink, &mut rgb).await?;
        done += 1;
        sink.stage(Stage::FrameSwapped {
            done: done as u32,
            total: count,
        });
    }
    if done == 0 {
        return Err("input video has no decodable frames".into());
    }

    sink.note(&format!(
        "Encoding MP4 (H.264){}",
        if audio.is_some() { " + audio" } else { "" }
    ));
    let (out_w, out_h, n, enc_sps, enc_pps, samples) = video_sink.finish()?;
    mux_mp4(
        output, out_w, out_h, fps, &enc_sps, &enc_pps, &samples, audio,
    )?;
    Ok((out_w, out_h, fps, n))
}

/// Decode one YUV frame to an [`Image`], swap, and feed it to the encoder sink.
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
/// (audio dropped, with a warning to stderr) when there is no audio, the codec
/// is not AAC, or the AAC object type cannot be remuxed losslessly (HE-AAC).
fn extract_audio<R: std::io::Read + std::io::Seek>(
    mp4: &mut mp4::Mp4Reader<R>,
) -> Result<Option<AudioPassthrough>, String> {
    use mp4::{AudioObjectType, MediaType, TrackType};

    let Some(id) = mp4
        .tracks()
        .iter()
        .find_map(|(&id, t)| matches!(t.track_type(), Ok(TrackType::Audio)).then_some(id))
    else {
        return Ok(None);
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

/// Streaming H.264 encoder sink: each swapped frame is encoded immediately (one
/// frame in RAM) and accumulated as a compressed AVCC sample. Output resolution
/// is fixed on the first frame, downscaled to fit openh264's encoder cap
/// (max 3840 long edge / 2160 short edge), aspect-preserved.
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

/// Largest aspect-preserving size within openh264's encoder cap (3840 long /
/// 2160 short edge), rounded to even (4:2:0 needs even dims).
fn fit_encode_dims(w: usize, h: usize) -> (usize, usize) {
    const MAX_LONG: f64 = 3840.0;
    const MAX_SHORT: f64 = 2160.0;
    let (long, short) = (w.max(h) as f64, w.min(h) as f64);
    let s = (MAX_LONG / long).min(MAX_SHORT / short).min(1.0);
    let even = |x: f64| ((x.round() as usize) & !1).max(2);
    (even(w as f64 * s), even(h as f64 * s))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bx(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut v = ((8 + payload.len()) as u32).to_be_bytes().to_vec();
        v.extend_from_slice(kind);
        v.extend_from_slice(payload);
        v
    }

    fn find(hay: &[u8], needle: &[u8]) -> usize {
        hay.windows(needle.len())
            .position(|w| w == needle)
            .expect("box type present")
    }

    // ftyp|mdat|moov (moov last) -> ftyp|moov|mdat, with the single stco chunk
    // offset shifted by the relocated moov's length.
    #[test]
    fn faststart_relocates_moov_and_shifts_chunk_offsets() {
        let ftyp = bx(b"ftyp", b"isom");
        let mdat = bx(b"mdat", &[0xAAu8; 32]);
        // In the moov-at-end layout the mdat payload starts right after ftyp.
        let mdat_payload_off = (ftyp.len() + 8) as u32;
        let mut stco_payload = vec![0, 0, 0, 0, 0, 0, 0, 1]; // version/flags, count=1
        stco_payload.extend_from_slice(&mdat_payload_off.to_be_bytes());
        let moov = bx(
            b"moov",
            &bx(
                b"trak",
                &bx(
                    b"mdia",
                    &bx(b"minf", &bx(b"stbl", &bx(b"stco", &stco_payload))),
                ),
            ),
        );
        let moov_len = moov.len() as u32;

        let mut input = Vec::new();
        input.extend_from_slice(&ftyp);
        input.extend_from_slice(&mdat);
        input.extend_from_slice(&moov);

        let out = faststart(input);
        assert!(
            find(&out, b"moov") < find(&out, b"mdat"),
            "moov must precede mdat"
        );
        let stco = find(&out, b"stco");
        let off = u32::from_be_bytes(out[stco + 12..stco + 16].try_into().unwrap());
        assert_eq!(
            off,
            mdat_payload_off + moov_len,
            "stco offset shifted by moov len"
        );
    }

    #[test]
    fn faststart_leaves_already_faststart_files_untouched() {
        // ftyp|moov|mdat (moov already first) is returned byte-identical.
        let ftyp = bx(b"ftyp", b"isom");
        let moov = bx(b"moov", &bx(b"trak", b""));
        let mdat = bx(b"mdat", &[0u8; 8]);
        let mut input = Vec::new();
        input.extend_from_slice(&ftyp);
        input.extend_from_slice(&moov);
        input.extend_from_slice(&mdat);
        assert_eq!(faststart(input.clone()), input);
    }
}
