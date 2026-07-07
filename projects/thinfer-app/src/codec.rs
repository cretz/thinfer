//! Container encode/decode: H.264 (openh264) + MP4 mux, the codec-free PNG
//! frame dump, and the face-swap streaming decode -> swap -> encode loop. This
//! is the one place the openh264/mp4 deps live (web uses the browser's
//! WebCodecs instead). Per-frame progress goes through a [`ProgressSink`];
//! descriptive lines go through [`ProgressSink::note`] with their original text.

use std::path::Path;

use thinfer_models::faceswap::image::Image;
use thinfer_models::faceswap::{Face, FaceSwapper};
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

/// Decode PNG/JPEG source-image container bytes (held in RAM) into an RGB
/// [`Image`]. The RAM-first counterpart of [`load_image`] (face-swap / DreamID-V
/// source face; the bytes never touch disk).
pub fn load_image_bytes(bytes: &[u8]) -> Result<Image, String> {
    let dynimg = image::load_from_memory(bytes).map_err(|e| format!("decode source image: {e}"))?;
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

/// The byte offset where a box's payload starts (past its size+type header),
/// accounting for a 64-bit `largesize`. `abs` is the box's absolute start.
fn box_payload_start(data: &[u8], abs: usize) -> usize {
    match u32::from_be_bytes(data[abs..abs + 4].try_into().unwrap()) {
        1 => abs + 16, // 64-bit largesize
        _ => abs + 8,
    }
}

/// Iterate the child boxes in `data[start..end)`, yielding `(abs_start, kind,
/// abs_end)` for each. Handles 32/64-bit sizes and size-0-to-end. Stops on any
/// malformed length (returns what parsed so far).
fn child_boxes(data: &[u8], mut p: usize, end: usize) -> Vec<([u8; 4], usize, usize)> {
    let mut out = Vec::new();
    while p + 8 <= end {
        let size = u32::from_be_bytes(data[p..p + 4].try_into().unwrap());
        let kind: [u8; 4] = data[p + 4..p + 8].try_into().unwrap();
        let len = match size {
            0 => end - p,
            1 => match data.get(p + 8..p + 16) {
                Some(b) => u64::from_be_bytes(b.try_into().unwrap()) as usize,
                None => break,
            },
            n => n as usize,
        };
        if len < 8 || p + len > end {
            break;
        }
        out.push((kind, p, p + len));
        p += len;
    }
    out
}

/// Whether a `trak` box (`data[start..end)`) is an AUDIO track: its
/// `mdia/hdlr` handler_type is `soun`.
fn trak_is_audio(data: &[u8], start: usize, end: usize) -> bool {
    let Some((_, ms, me)) = child_boxes(data, box_payload_start(data, start), end)
        .into_iter()
        .find(|(k, ..)| k == b"mdia")
    else {
        return false;
    };
    let Some((_, hs, _)) = child_boxes(data, box_payload_start(data, ms), me)
        .into_iter()
        .find(|(k, ..)| k == b"hdlr")
    else {
        return false;
    };
    // hdlr payload: version+flags (4), pre_defined (4), handler_type (4).
    let ht = box_payload_start(data, hs) + 8;
    data.get(ht..ht + 4) == Some(b"soun".as_slice())
}

/// Neutralize every AUDIO trak inside `moov` so the strict `mp4` demuxer never
/// parses an audio sample-entry box it rejects (some AAC muxings), even though we
/// only need the H.264 video track. Each audio `trak` box type is rewritten IN
/// PLACE to `free` (an ignorable box): the moov length is unchanged, so `mdat`
/// stays put and the video `stco` chunk offsets (absolute into the file) remain
/// valid regardless of faststart. Returns the input untouched (borrowed) when
/// there is no audio trak or on any parse surprise (the caller then surfaces the
/// actionable hint if the read still fails).
fn strip_audio_traks(input: &[u8]) -> std::borrow::Cow<'_, [u8]> {
    use std::borrow::Cow;
    let Some(boxes) = top_level_boxes(input) else {
        return Cow::Borrowed(input);
    };
    let Some(moov) = boxes.iter().find(|b| &b.kind == b"moov") else {
        return Cow::Borrowed(input);
    };
    let audio: Vec<usize> = child_boxes(
        input,
        box_payload_start(input, moov.start),
        moov.start + moov.len,
    )
    .into_iter()
    .filter(|(k, s, e)| k == b"trak" && trak_is_audio(input, *s, *e))
    .map(|(_, s, _)| s)
    .collect();
    if audio.is_empty() {
        return Cow::Borrowed(input);
    }
    let mut out = input.to_vec();
    for s in audio {
        out[s + 4..s + 8].copy_from_slice(b"free");
    }
    Cow::Owned(out)
}

/// First child box of kind `kind` directly inside the box at `parent_start`
/// (whose contents end at `parent_end`). Returns `(kind, abs_start, abs_end)`.
fn find_child(
    data: &[u8],
    parent_start: usize,
    parent_end: usize,
    kind: &[u8; 4],
) -> Option<([u8; 4], usize, usize)> {
    child_boxes(data, box_payload_start(data, parent_start), parent_end)
        .into_iter()
        .find(|(k, ..)| k == kind)
}

/// Some H.264 muxers emit the `avc1` sample entry with a `colr`/`pasp`/`fiel`
/// box BEFORE the `avcC` config box. The strict `mp4` demuxer reads only the
/// FIRST child of `avc1` and rejects the track ("avcc not found") if it is not
/// `avcC`, even though the clip is perfectly decodable H.264. Reorder the `avc1`
/// child boxes IN PLACE so `avcC` leads: every child keeps its exact bytes and
/// length and the region is only permuted, so the `avc1` box size and all
/// downstream chunk offsets (`stco`/`co64`) are unchanged. Returns the input
/// borrowed when no reorder is needed or on any parse surprise.
fn reorder_avcc_first(input: &[u8]) -> std::borrow::Cow<'_, [u8]> {
    use std::borrow::Cow;
    let Some(boxes) = top_level_boxes(input) else {
        return Cow::Borrowed(input);
    };
    let Some(moov) = boxes.iter().find(|b| &b.kind == b"moov") else {
        return Cow::Borrowed(input);
    };

    // Every avc1 child region that needs reordering: (children_start, avc1_end).
    let mut regions: Vec<(usize, usize)> = Vec::new();
    for (k, ts, te) in child_boxes(
        input,
        box_payload_start(input, moov.start),
        moov.start + moov.len,
    ) {
        if &k != b"trak" {
            continue;
        }
        // trak -> mdia -> minf -> stbl -> stsd.
        let Some((_, ms, me)) = find_child(input, ts, te, b"mdia") else {
            continue;
        };
        let Some((_, is, ie)) = find_child(input, ms, me, b"minf") else {
            continue;
        };
        let Some((_, bs, be)) = find_child(input, is, ie, b"stbl") else {
            continue;
        };
        let Some((_, ss, se)) = find_child(input, bs, be, b"stsd") else {
            continue;
        };
        // stsd is a FullBox: skip version+flags (4) + entry_count (4) to reach
        // the sample-entry boxes; then find the avc1 entry (if any).
        for (ek, es, ee) in child_boxes(input, box_payload_start(input, ss) + 8, se) {
            if &ek != b"avc1" {
                continue;
            }
            // avc1's children start after the 78-byte visual sample entry header
            // (the fields the `mp4` crate reads before the first child box).
            let children_start = box_payload_start(input, es) + 78;
            let kids = child_boxes(input, children_start, ee);
            match kids.iter().position(|(kk, ..)| kk == b"avcC") {
                Some(0) | None => {} // already first, or no avcC to hoist
                Some(_) => regions.push((children_start, ee)),
            }
        }
    }
    if regions.is_empty() {
        return Cow::Borrowed(input);
    }

    let mut out = input.to_vec();
    for (children_start, avc1_end) in regions {
        let kids = child_boxes(input, children_start, avc1_end);
        let mut reordered = Vec::with_capacity(avc1_end - children_start);
        // avcC first, then every other child in its original order.
        for (k, s, e) in kids.iter().filter(|(k, ..)| k == b"avcC") {
            let _ = k;
            reordered.extend_from_slice(&input[*s..*e]);
        }
        for (k, s, e) in &kids {
            if k != b"avcC" {
                reordered.extend_from_slice(&input[*s..*e]);
            }
        }
        // Only rewrite when the children tile the region exactly (they always do
        // for well-formed avc1); otherwise leave it for the reader to reject.
        if reordered.len() == avc1_end - children_start {
            out[children_start..avc1_end].copy_from_slice(&reordered);
        }
    }
    Cow::Owned(out)
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
fn avcc_to_annexb(avcc: &[u8], length_size: usize, out: &mut Vec<u8>) {
    let mut i = 0;
    while i + length_size <= avcc.len() {
        let mut len = 0usize;
        for &b in &avcc[i..i + length_size] {
            len = (len << 8) | b as usize;
        }
        i += length_size;
        if i + len > avcc.len() {
            break;
        }
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&avcc[i..i + len]);
        i += len;
    }
}

/// (all SPS NALs, all PPS NALs, NAL length-prefix width in bytes).
type ParamSets = (Vec<Vec<u8>>, Vec<Vec<u8>>, usize);

/// Every SPS and PPS NAL from every `avcC` config record in the file, plus the
/// NAL length-prefix width (bytes). A clip can carry MORE than one parameter set
/// -- a second `stsd` `avc1` entry written after a mid-recording resolution or
/// orientation change, or several sets in one record. Slices after the switch
/// reference the later set, so prepending only the FIRST record's first SPS/PPS
/// makes openh264 report `dsNoParamSets` (Native error bit 0x10) and lose every
/// frame past the switch. Collect them all (deduped) so the prefix we prepend to
/// each access unit always contains the set the slice needs. `None` when no avcC
/// parsed (the caller falls back to the demuxer's single SPS/PPS).
fn collect_avcc_param_sets(input: &[u8]) -> Option<ParamSets> {
    let boxes = top_level_boxes(input)?;
    let moov = boxes.iter().find(|b| &b.kind == b"moov")?;
    let mut sps_all: Vec<Vec<u8>> = Vec::new();
    let mut pps_all: Vec<Vec<u8>> = Vec::new();
    let mut length_size = 4usize;
    for (k, ts, te) in child_boxes(
        input,
        box_payload_start(input, moov.start),
        moov.start + moov.len,
    ) {
        if &k != b"trak" {
            continue;
        }
        let Some((_, ms, me)) = find_child(input, ts, te, b"mdia") else {
            continue;
        };
        let Some((_, is, ie)) = find_child(input, ms, me, b"minf") else {
            continue;
        };
        let Some((_, bs, be)) = find_child(input, is, ie, b"stbl") else {
            continue;
        };
        let Some((_, ss, se)) = find_child(input, bs, be, b"stsd") else {
            continue;
        };
        for (ek, es, ee) in child_boxes(input, box_payload_start(input, ss) + 8, se) {
            if &ek != b"avc1" {
                continue;
            }
            // avc1 children start past the 78-byte visual sample entry header.
            let children_start = box_payload_start(input, es) + 78;
            let Some((_, cs, ce)) = child_boxes(input, children_start, ee)
                .into_iter()
                .find(|(kk, ..)| kk == b"avcC")
            else {
                continue;
            };
            let rec = &input[box_payload_start(input, cs)..ce];
            if let Some(ls) = parse_avcc_record(rec, &mut sps_all, &mut pps_all) {
                length_size = ls;
            }
        }
    }
    if sps_all.is_empty() && pps_all.is_empty() {
        return None;
    }
    Some((sps_all, pps_all, length_size))
}

/// Parse one AVCDecoderConfigurationRecord, appending its SPS/PPS NALs (deduped)
/// and returning the NAL length-prefix width in bytes. Layout: `[0]` version,
/// `[1..4]` profile/compat/level, `[4]` `xxxxxx|lengthSizeMinusOne(2b)`, `[5]`
/// `xxx|numSPS(5b)`, then each SPS as `u16 len + bytes`, then `[numPPS: u8]`,
/// then each PPS as `u16 len + bytes`.
fn parse_avcc_record(rec: &[u8], sps: &mut Vec<Vec<u8>>, pps: &mut Vec<Vec<u8>>) -> Option<usize> {
    if rec.len() < 6 {
        return None;
    }
    let length_size = (rec[4] & 0x03) as usize + 1;
    let push = |dst: &mut Vec<Vec<u8>>, nal: &[u8]| {
        if !nal.is_empty() && !dst.iter().any(|e| e.as_slice() == nal) {
            dst.push(nal.to_vec());
        }
    };
    let mut i = 6;
    for _ in 0..(rec[5] & 0x1F) {
        if i + 2 > rec.len() {
            return Some(length_size);
        }
        let l = u16::from_be_bytes([rec[i], rec[i + 1]]) as usize;
        i += 2;
        if i + l > rec.len() {
            return Some(length_size);
        }
        push(sps, &rec[i..i + l]);
        i += l;
    }
    if i >= rec.len() {
        return Some(length_size);
    }
    let num_pps = rec[i];
    i += 1;
    for _ in 0..num_pps {
        if i + 2 > rec.len() {
            break;
        }
        let l = u16::from_be_bytes([rec[i], rec[i + 1]]) as usize;
        i += 2;
        if i + l > rec.len() {
            break;
        }
        push(pps, &rec[i..i + l]);
        i += l;
    }
    Some(length_size)
}

/// Build the Annex B parameter-set prefix (all SPS then all PPS, each with a
/// 4-byte start code) prepended to every access unit before decode. Falls back
/// to the demuxer's single `sps`/`pps` when no multi-set `avcC` could be parsed.
fn build_param_set_prefix(container: &[u8], sps: &[u8], pps: &[u8]) -> (Vec<u8>, usize) {
    let (sps_all, pps_all, length_size) = collect_avcc_param_sets(container)
        .unwrap_or_else(|| (vec![sps.to_vec()], vec![pps.to_vec()], 4));
    let mut prefix = Vec::new();
    for nal in sps_all.iter().chain(pps_all.iter()) {
        prefix.extend_from_slice(&[0, 0, 0, 1]);
        prefix.extend_from_slice(nal);
    }
    (prefix, length_size)
}

/// Build the H.264 decoder with error concealment enabled. The safe `openh264`
/// wrapper hardcodes concealment OFF, so a single lost reference (a corrupt or
/// dropped inter-frame -- common in phone-recorded clips) makes EVERY following
/// frame fail until the next clean IDR; with IDRs ~2s apart the output collapses
/// to ~1 distinct frame every couple seconds (and a long enough run trips the
/// per-sample error guard and aborts the job). `ERROR_CON_SLICE_MV_COPY_CROSS_IDR`
/// lets the decoder rebuild references (motion-copy concealment, recovering across
/// IDRs) and keep producing frames. `NoFlush` (drain via `flush_remaining`) is
/// B-frame safe. Concealment is a no-op on clean streams (error-free frames keep
/// returning `dsErrorFree`), so it does not change well-formed clips.
fn make_h264_decoder() -> Result<openh264::decoder::Decoder, String> {
    use openh264::OpenH264API;
    use openh264::decoder::{Decoder, DecoderConfig, Flush};
    let mut decoder = Decoder::with_api_config(
        OpenH264API::from_source(),
        DecoderConfig::new().flush_after_decode(Flush::NoFlush),
    )
    .map_err(|e| format!("openh264 decoder init: {e}"))?;
    // SAFETY: raw_api()/set_option are unsafe FFI. We only set the documented
    // DECODER_OPTION_ERROR_CON_IDC to a valid ERROR_CON_IDC; `idc` outlives the
    // call and SetOption copies the int out, so the pointer is sound.
    unsafe {
        let mut idc: openh264_sys2::ERROR_CON_IDC =
            openh264_sys2::ERROR_CON_SLICE_MV_COPY_CROSS_IDR;
        let _ = decoder.raw_api().set_option(
            openh264_sys2::DECODER_OPTION_ERROR_CON_IDC,
            std::ptr::addr_of_mut!(idc).cast(),
        );
    }
    Ok(decoder)
}

/// Feed one Annex B access unit and return the decoded frame, KEEPING concealed
/// frames. The safe `Decoder::decode` turns any nonzero decode state into an Err
/// and discards the frame -- but with concealment on, a frame flagged
/// `dsDataErrorConcealed` (lost reference filled in) is still valid and usable;
/// discarding it is exactly what froze glitchy/corrupt clips. We call the raw
/// `decode_frame_no_delay` and read the picture directly, ignoring the state.
/// `None` means the decoder buffered the AU for reorder (no frame ready yet).
fn decode_au_concealed(
    decoder: &mut openh264::decoder::Decoder,
    au: &[u8],
    rgb: &mut Vec<u8>,
) -> Option<Image> {
    let mut dst = [std::ptr::null_mut::<u8>(); 3];
    let mut info = openh264_sys2::SBufferInfo::default();
    // SAFETY: raw FFI. `decode_frame_no_delay` fills `dst` (plane pointers into the
    // decoder's internal picture, valid until the next decode call) and `info`; we
    // copy pixels out immediately in `yuv_ptrs_to_rgb`. The returned DECODING_STATE
    // is intentionally ignored (concealed frames report nonzero but are valid).
    unsafe {
        decoder.raw_api().decode_frame_no_delay(
            au.as_ptr(),
            au.len() as i32,
            std::ptr::from_mut(&mut dst).cast(),
            std::ptr::from_mut(&mut info),
        );
    }
    yuv_ptrs_to_rgb(&dst, &info, rgb)
}

/// Drain the decoder's reorder buffer after the last AU, keeping concealed frames
/// (see [`decode_au_concealed`]). Bounded well above any real reorder depth.
fn flush_all_concealed(decoder: &mut openh264::decoder::Decoder, rgb: &mut Vec<u8>) -> Vec<Image> {
    let mut out = Vec::new();
    for _ in 0..64 {
        let mut dst = [std::ptr::null_mut::<u8>(); 3];
        let mut info = openh264_sys2::SBufferInfo::default();
        // SAFETY: as decode_au_concealed; flush_frame emits one buffered picture.
        unsafe {
            decoder.raw_api().flush_frame(
                std::ptr::from_mut(&mut dst).cast(),
                std::ptr::from_mut(&mut info),
            );
        }
        match yuv_ptrs_to_rgb(&dst, &info, rgb) {
            Some(img) => out.push(img),
            None => break,
        }
    }
    out
}

/// Convert an openh264 I420 picture (plane pointers + `SBufferInfo`) to an RGB
/// [`Image`]. `None` when no frame is ready (`iBufferStatus != 1`) or a plane is
/// null. Matches openh264's own limited-range BT.601 YUV->RGB (`write_rgb8`), so
/// output is identical to the safe path on clean frames.
fn yuv_ptrs_to_rgb(
    dst: &[*mut u8; 3],
    info: &openh264_sys2::SBufferInfo,
    rgb: &mut Vec<u8>,
) -> Option<Image> {
    if info.iBufferStatus != 1 || dst[0].is_null() || dst[1].is_null() || dst[2].is_null() {
        return None;
    }
    // SAFETY: the union holds the system-memory buffer when a frame is ready.
    let sys = unsafe { info.UsrData.sSystemBuffer };
    let (w, h) = (sys.iWidth as usize, sys.iHeight as usize);
    let (ys, cs) = (sys.iStride[0] as usize, sys.iStride[1] as usize);
    if w == 0 || h == 0 || ys == 0 || cs == 0 {
        return None;
    }
    // SAFETY: with iBufferStatus==1 the planes hold a full I420 picture at these
    // strides (Y: h rows x ys; U/V: h/2 rows x cs), matching the safe decoder.
    let (y, u, v) = unsafe {
        (
            std::slice::from_raw_parts(dst[0], ys * h),
            std::slice::from_raw_parts(dst[1], cs * h / 2),
            std::slice::from_raw_parts(dst[2], cs * h / 2),
        )
    };
    const Y_MUL: f32 = 255.0 / 219.0;
    const RV_MUL: f32 = 255.0 / 224.0 * 1.402;
    const GV_MUL: f32 = -255.0 / 224.0 * 1.402 * 0.299 / 0.687;
    const GU_MUL: f32 = -255.0 / 224.0 * 1.772 * 0.114 / 0.587;
    const BU_MUL: f32 = 255.0 / 224.0 * 1.772;
    rgb.resize(w * h * 3, 0);
    for row in 0..h {
        for col in 0..w {
            let t = (row * w + col) * 3;
            let yv = Y_MUL * (f32::from(y[row * ys + col]) - 16.0);
            let uu = f32::from(u[(row / 2) * cs + col / 2]) - 128.0;
            let vv = f32::from(v[(row / 2) * cs + col / 2]) - 128.0;
            rgb[t] = RV_MUL.mul_add(vv, yv) as u8;
            rgb[t + 1] = GV_MUL.mul_add(vv, GU_MUL.mul_add(uu, yv)) as u8;
            rgb[t + 2] = BU_MUL.mul_add(uu, yv) as u8;
        }
    }
    Some(Image::from_rgb8(w, h, rgb))
}

// --- RAM mp4 decode (DreamID-V target video) ---------------------------------

/// Decode EVERY video frame of an in-RAM mp4 into RGB [`Image`]s (RAM-first: no
/// disk touch). Returns `(frames, fps)`. Reuses the openh264 decoder + mp4
/// demux; any audio track is ignored (DreamID-V re-encodes a silent clip). Used
/// by the DreamID-V face-swap path, which needs the whole clip in RAM to VAE-
/// encode it (unlike the frame-streaming HyperSwap path).
pub fn decode_mp4_frames(input: &[u8]) -> Result<(Vec<Image>, u32), String> {
    use std::io::Cursor;

    use mp4::{Mp4Reader, TrackType};

    // The `mp4` demuxer parses the whole moov (incl. the audio track) strictly, so
    // an unusual audio sample-entry box (some AAC muxings) would fail the whole read
    // even though we only need the H.264 video track. Neutralize the audio trak(s)
    // first (in-place `free` rewrite, moov length preserved). If a read still fails,
    // surface an actionable hint rather than the raw box error.
    let prepared = reorder_avcc_first(input);
    let demuxed = strip_audio_traks(prepared.as_ref());
    let demuxed = demuxed.as_ref();
    let size = demuxed.len() as u64;
    let mut mp4 = Mp4Reader::read_header(Cursor::new(demuxed), size).map_err(|e| {
        format!(
            "could not parse the MP4 video container ({e}). The file may be malformed \
             or use an unsupported box layout; re-mux/re-export it as a standard \
             H.264 MP4 and retry."
        )
    })?;

    let (track_id, sps, pps, fps) = {
        let track = mp4
            .tracks()
            .iter()
            .find(|(_, t)| matches!(t.track_type(), Ok(TrackType::Video)))
            .ok_or("no video track in input")?
            .1;
        let sps = track
            .sequence_parameter_set()
            .map_err(|e| {
                format!(
                    "missing H.264 SPS ({e}); the video track is likely HEVC/H.265 or \
                     another non-AVC codec, which is not supported -- re-export as H.264"
                )
            })?
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

    // Prepend EVERY avcC parameter set (all SPS/PPS across all avc1 entries), not
    // just the first, so a mid-clip param-set switch does not lose the rest.
    let (prefix, length_size) = build_param_set_prefix(demuxed, &sps, &pps);
    let mut decoder = make_h264_decoder()?;

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
        avcc_to_annexb(&sample.bytes, length_size, &mut au);
        // Concealing decode: a lost reference is filled in, not turned into a hard
        // error, so a glitchy/corrupt clip keeps decoding instead of freezing.
        if let Some(img) = decode_au_concealed(&mut decoder, &au, &mut rgb) {
            frames.push(img);
        }
    }
    frames.extend(flush_all_concealed(&mut decoder, &mut rgb));
    if frames.is_empty() {
        return Err("input video has no decodable frames".into());
    }
    Ok((frames, fps))
}

// --- Face-swap streaming (mp4 demux -> per-frame swap -> mp4 mux) -------------

/// Decode the input, swap each frame, and encode the output one frame at a time
/// so only a few frames are ever in RAM (a 4K clip otherwise materializes
/// ~30GB). The source AAC audio track is remuxed verbatim. Returns the output
/// `(width, height, fps, frame_count)`.
/// Encode + trim knobs for a streaming face-swap.
#[derive(Clone, Copy, Debug)]
pub struct StreamOptions {
    /// Detect faces every Nth frame (>=1), reusing between. See `detect_stride`.
    pub detect_stride: u32,
    /// Output H.264 bitrate = source video bitrate * this factor. ~1.15 keeps
    /// quality (offsets the second encode + a weaker encoder) without the 2x bloat
    /// of a fixed high target. Falls back to a bits-per-pixel target when the
    /// source bitrate is unknown.
    pub bitrate_scale: f32,
    /// Optional swap+output window, in seconds of the SOURCE clip: only frames in
    /// [start, end) are swapped and written (rebased to t=0). `None` bounds mean
    /// clip start / clip end.
    pub start_secs: Option<f32>,
    pub end_secs: Option<f32>,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            detect_stride: 1,
            bitrate_scale: 1.15,
            start_secs: None,
            end_secs: None,
        }
    }
}

pub async fn swap_video_streaming(
    swapper: &FaceSwapper,
    embedding: &[f32],
    input: &[u8],
    output: &Path,
    opts: StreamOptions,
    sink: &dyn ProgressSink,
) -> Result<(usize, usize, u32, usize), String> {
    use std::io::Cursor;

    use mp4::{Mp4Reader, TrackType};

    // RAM-first: the uploaded mp4 rides in memory (decrypted from an encrypted
    // spill upstream if it was large), so demux over a Cursor -- no plaintext
    // video ever hits disk here.
    //
    // The `mp4` demuxer parses the whole moov (incl. the audio track) strictly, so
    // an unusual audio sample-entry box (some AAC muxings) would fail the whole read
    // even though we only need the H.264 video track. Neutralize the audio trak(s)
    // first (in-place `free` rewrite, moov length preserved). If a read still fails,
    // surface an actionable hint rather than the raw box error.
    //
    // The bytes are owned here (not borrowed) so the decode thread below can take
    // them: the pipeline runs demux+decode on one thread and encode on another,
    // overlapping the ~77ms/frame codec CPU with the GPU swap.
    //
    // First reorder the avc1 sample entry so `avcC` leads (some muxers put a
    // colr/pasp box first, which the strict demuxer rejects). Audio is extracted
    // from these audio-INTACT bytes for passthrough BEFORE the audio trak is
    // neutralized for the video demux -- stripping first would drop the audio.
    let prepared = reorder_avcc_first(input);
    let audio = extract_audio_from_bytes(prepared.as_ref());
    let demuxed: Vec<u8> = strip_audio_traks(prepared.as_ref()).into_owned();
    let size = demuxed.len() as u64;
    let mp4 = Mp4Reader::read_header(Cursor::new(demuxed.as_slice()), size).map_err(|e| {
        format!(
            "could not parse the MP4 video container ({e}). The file may be malformed \
             or use an unsupported box layout; re-mux/re-export it as a standard \
             H.264 MP4 and retry."
        )
    })?;

    let (track_id, sps, pps, fps) = {
        let track = mp4
            .tracks()
            .iter()
            .find(|(_, t)| matches!(t.track_type(), Ok(TrackType::Video)))
            .ok_or("no video track in input")?
            .1;
        let sps = track
            .sequence_parameter_set()
            .map_err(|e| {
                format!(
                    "missing H.264 SPS ({e}); the video track is likely HEVC/H.265 or \
                     another non-AVC codec, which is not supported -- re-export as H.264"
                )
            })?
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
    // Source video bitrate -> scaled encoder target. 0/unknown -> None (the
    // encoder falls back to its bits-per-pixel target).
    let src_bitrate = mp4
        .tracks()
        .get(&track_id)
        .map(|t| t.bitrate())
        .unwrap_or(0);
    let target_bitrate = (src_bitrate > 0).then(|| {
        (f64::from(src_bitrate) * f64::from(opts.bitrate_scale.max(0.1))).max(1_000_000.0) as u32
    });

    // Trim window -> frame range [start_frame, end_frame). Bounds snap to whole
    // frames at the source fps; the output is rebased to t=0.
    let count_usize = count as usize;
    let to_frame = |s: f32| (s.max(0.0) * fps as f32).round() as usize;
    let start_frame = opts.start_secs.map(to_frame).unwrap_or(0).min(count_usize);
    let end_frame = opts
        .end_secs
        .map(to_frame)
        .unwrap_or(count_usize)
        .clamp(start_frame, count_usize);
    let out_count = end_frame - start_frame;
    if out_count == 0 {
        return Err(format!(
            "trim window selects no frames (start={:?}s end={:?}s, clip is {} frames @ {fps}fps)",
            opts.start_secs, opts.end_secs, count
        ));
    }
    // Trim the passthrough audio to the same window (rebased) so it stays in sync.
    let audio = if opts.start_secs.is_some() || opts.end_secs.is_some() {
        trim_audio(
            audio,
            start_frame as f32 / fps as f32,
            end_frame as f32 / fps as f32,
        )
    } else {
        audio
    };

    match &audio {
        Some(a) => sink.note(&format!(
            "Audio: {} AAC sample(s), passthrough",
            a.samples.len()
        )),
        None => sink.note("Audio: video-only output"),
    }

    // Annex B parameter-set prefix, prepended to every access unit. Include EVERY
    // avcC parameter set (all SPS/PPS across all avc1 entries), not just the first,
    // so a mid-clip param-set switch (a second stsd entry after a resolution /
    // orientation change) does not make openh264 lose every frame past the switch.
    let (prefix, length_size) = build_param_set_prefix(&demuxed, &sps, &pps);

    // --- 3-stage pipeline: (1) demux+decode+YUV->RGB on a thread, (2) detect+
    // swap on this task (GPU), (3) encode on a thread. Bounded channels keep only
    // a few frames resident (streaming stays RAM-cheap) while the ~77ms/frame
    // codec CPU overlaps the GPU swap -> per-frame wall ~= max(gpu, codec)
    // instead of their serial sum. ---
    type EncodeResult =
        Result<(usize, usize, usize, Vec<u8>, Vec<u8>, Vec<(bool, Vec<u8>)>), String>;

    if out_count == count_usize {
        sink.note(&format!("Streaming {count} frames at {fps} fps"));
    } else {
        sink.note(&format!(
            "Streaming {out_count} of {count} frames (trim {start_frame}..{end_frame}) at {fps} fps"
        ));
    }
    drop(mp4); // release the borrow of `demuxed` so the decode thread can own it.

    // Stage 1: emits owned `Image`s in display order; a decode error is sent as
    // the final `Err` item and also surfaces via the thread join.
    let (img_tx, img_rx) = std::sync::mpsc::sync_channel::<Result<Image, String>>(3);
    let decode_thread = std::thread::spawn(move || {
        let result = (|| -> Result<(), String> {
            let mut mp4 = Mp4Reader::read_header(Cursor::new(demuxed.as_slice()), size)
                .map_err(|e| format!("re-parse video for streaming: {e}"))?;
            // B-frame-safe (NoFlush + flush_remaining) with error concealment on.
            let mut decoder = make_h264_decoder()?;
            let mut rgb: Vec<u8> = Vec::new();
            // Frames must be DECODED in order (H.264 references), but we only need
            // to EMIT up to `end_frame` display frames -- stop early past the trim
            // end instead of decoding the whole tail. Front-trim frames are still
            // decoded (they may be references) and dropped by the consumer.
            let mut emitted = 0usize;
            for sid in 1..=count {
                let Some(sample) = mp4
                    .read_sample(track_id, sid)
                    .map_err(|e| format!("read sample {sid}: {e}"))?
                else {
                    continue;
                };
                let mut au = prefix.clone();
                avcc_to_annexb(&sample.bytes, length_size, &mut au);
                // Concealing decode: a lost reference (corrupt/dropped inter-frame
                // in a phone clip) is filled in instead of turned into a hard error,
                // so the stream keeps decoding rather than freezing until the next
                // IDR. `None` = frame buffered for B-frame reorder (not ready yet).
                let Some(frame) = decode_au_concealed(&mut decoder, &au, &mut rgb) else {
                    continue;
                };
                if img_tx.send(Ok(frame)).is_err() {
                    return Ok(()); // consumer gone
                }
                emitted += 1;
                if emitted >= end_frame {
                    return Ok(());
                }
            }
            for frame in flush_all_concealed(&mut decoder, &mut rgb) {
                if img_tx.send(Ok(frame)).is_err() {
                    return Ok(());
                }
                emitted += 1;
                if emitted >= end_frame {
                    return Ok(());
                }
            }
            Ok(())
        })();
        if let Err(e) = result {
            let _ = img_tx.send(Err(e));
        }
    });

    // Stage 3: encode each swapped frame immediately (one frame resident).
    let (swap_tx, swap_rx) = std::sync::mpsc::sync_channel::<Image>(3);
    let encode_thread = std::thread::spawn(move || -> EncodeResult {
        let mut video_sink = Mp4VideoSink::new(fps, target_bitrate);
        for frame in swap_rx {
            video_sink.push(&frame)?;
        }
        video_sink.finish()
    });

    // Stage 2 (this task): detect (on a stride), then swap a WINDOW of frames at
    // once so the HyperSwap forward batches every frame's crop into one GPU
    // dispatch (the swapper is badly GPU-underutilized at one crop at a time).
    // Detection cadence and frame order are unchanged vs the serial loop; only
    // the swap is grouped. `swap_window` collects up to SWAP_BATCH frames + their
    // detected faces, batch-swaps them in place, and streams them to the encoder
    // in display order.
    let stride = opts.detect_stride.max(1);
    let mut cached_faces: Vec<Face> = Vec::new();
    let mut done = 0usize; // frames emitted (drives progress)
    let mut win_frames: Vec<Image> = Vec::new();
    let mut win_faces: Vec<Vec<Face>> = Vec::new();
    let mut pipeline_err: Option<String> = None;
    let mut aborted = false;

    // Batch-swap the current window and stream it to the encoder. Returns false
    // if the encoder is gone (caller should stop).
    async fn flush_window(
        swapper: &FaceSwapper,
        embedding: &[f32],
        frames: &mut Vec<Image>,
        faces: &mut Vec<Vec<Face>>,
        swap_tx: &std::sync::mpsc::SyncSender<Image>,
        done: &mut usize,
        count: u32,
        sink: &dyn ProgressSink,
    ) -> Result<bool, String> {
        if frames.is_empty() {
            return Ok(true);
        }
        swapper
            .swap_predetected_multi(frames, faces, embedding)
            .await
            .map_err(|e| format!("swap frame: {e}"))?;
        faces.clear();
        for f in frames.drain(..) {
            if swap_tx.send(f).is_err() {
                return Ok(false); // encoder died; its error surfaces at join
            }
            *done += 1;
            // Progress is emitted every STATUS_EVERY frames (plus the last one) to
            // keep the SSE stream light on long clips -- not once per frame.
            const STATUS_EVERY: usize = 50;
            if done.is_multiple_of(STATUS_EVERY) || *done as u32 == count {
                sink.stage(Stage::FrameSwapped {
                    done: *done as u32,
                    total: count,
                });
            }
        }
        Ok(true)
    }

    for (idx, item) in img_rx.into_iter().enumerate() {
        let img = match item {
            Ok(img) => img,
            Err(e) => {
                pipeline_err = Some(e);
                break;
            }
        };
        if idx < start_frame {
            continue; // front-trim: decoded for references, not swapped/emitted
        }
        if idx >= end_frame {
            break; // past the trim end (the decode thread stops here too)
        }
        if cached_faces.is_empty() || idx.is_multiple_of(stride as usize) {
            match swapper.detect(&img).await {
                Ok(f) => cached_faces = f,
                Err(e) => {
                    pipeline_err = Some(format!("detect: {e}"));
                    break;
                }
            }
        }
        win_frames.push(img);
        win_faces.push(cached_faces.clone());
        if win_frames.len() >= thinfer_models::faceswap::SWAP_BATCH {
            match flush_window(
                swapper,
                embedding,
                &mut win_frames,
                &mut win_faces,
                &swap_tx,
                &mut done,
                out_count as u32,
                sink,
            )
            .await
            {
                Ok(true) => {}
                Ok(false) => {
                    aborted = true;
                    break;
                }
                Err(e) => {
                    pipeline_err = Some(e);
                    break;
                }
            }
        }
    }
    // Swap the trailing partial window (unless we already hit an error / abort).
    if pipeline_err.is_none()
        && !aborted
        && let Err(e) = flush_window(
            swapper,
            embedding,
            &mut win_frames,
            &mut win_faces,
            &swap_tx,
            &mut done,
            out_count as u32,
            sink,
        )
        .await
    {
        pipeline_err = Some(e);
    }
    drop(swap_tx); // close the encoder's input so it can finish

    let encoded = encode_thread
        .join()
        .map_err(|_| "encode thread panicked".to_string())?;
    let _ = decode_thread.join(); // decode errors surface via `img_rx` above

    if let Some(e) = pipeline_err {
        return Err(e);
    }
    let (out_w, out_h, n, enc_sps, enc_pps, samples) = encoded?;
    if n == 0 {
        return Err("input video has no decodable frames".into());
    }

    sink.note(&format!(
        "Encoding MP4 (H.264){}",
        if audio.is_some() { " + audio" } else { "" }
    ));
    mux_mp4(
        output, out_w, out_h, fps, &enc_sps, &enc_pps, &samples, audio,
    )?;
    Ok((out_w, out_h, fps, n))
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

/// Extract the AAC audio track from a full mp4 in RAM, for passthrough. Call
/// this on the audio-INTACT bytes (before [`strip_audio_traks`] neutralizes the
/// audio trak for the strict video demux): stripping first would leave nothing
/// to pass through. Returns `None` when there is no audio track, the codec is
/// not passthrough-able AAC, or the container's audio box is one the demuxer
/// cannot parse (a soft failure -> video-only output, never a hard error).
fn extract_audio_from_bytes(bytes: &[u8]) -> Option<AudioPassthrough> {
    use std::io::Cursor;
    let mut mp4 = mp4::Mp4Reader::read_header(Cursor::new(bytes), bytes.len() as u64).ok()?;
    extract_audio(&mut mp4).ok().flatten()
}

/// Trim a passthrough audio track to the window `[start_sec, end_sec)` of the
/// SOURCE timeline and rebase it to t=0, so it stays in sync with a trimmed
/// video. Samples are kept by their `start_time` (in the track timescale).
/// Returns `None` if nothing remains in the window.
fn trim_audio(
    audio: Option<AudioPassthrough>,
    start_sec: f32,
    end_sec: f32,
) -> Option<AudioPassthrough> {
    let mut a = audio?;
    let ts = f64::from(a.timescale);
    let start_tick = (f64::from(start_sec) * ts) as u64;
    let end_tick = (f64::from(end_sec) * ts) as u64;
    a.samples.retain_mut(|s| {
        let keep = s.start_time >= start_tick && s.start_time < end_tick;
        if keep {
            s.start_time -= start_tick;
        }
        keep
    });
    (!a.samples.is_empty()).then_some(a)
}

/// Streaming H.264 encoder sink: each swapped frame is encoded immediately (one
/// frame in RAM) and accumulated as a compressed AVCC sample. Output resolution
/// is fixed on the first frame, downscaled to fit openh264's encoder cap
/// (max 3840 long edge / 2160 short edge), aspect-preserved.
struct Mp4VideoSink {
    fps: u32,
    /// Target H.264 bitrate (bps). `None` -> a bits-per-pixel target from the
    /// output dims (used when the source bitrate could not be read).
    target_bitrate: Option<u32>,
    enc: Option<openh264::encoder::Encoder>,
    out_w: usize,
    out_h: usize,
    samples: Vec<(bool, Vec<u8>)>,
    sps: Option<Vec<u8>>,
    pps: Option<Vec<u8>>,
}

impl Mp4VideoSink {
    fn new(fps: u32, target_bitrate: Option<u32>) -> Self {
        Self {
            fps,
            target_bitrate,
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
            // Prefer the (scaled) source bitrate so the output tracks the input's
            // size/quality; fall back to a bits-per-pixel target if it is unknown.
            let bitrate = self.target_bitrate.unwrap_or_else(|| {
                ((ow as f64) * (oh as f64) * (self.fps as f64) * BITS_PER_PIXEL).max(1_000_000.0)
                    as u32
            });
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

    /// Build an AVCDecoderConfigurationRecord: version/profile/compat/level, then
    /// `lengthSizeMinusOne`, the SPS list, and the PPS list.
    fn avcc_record(length_minus_one: u8, sps: &[&[u8]], pps: &[&[u8]]) -> Vec<u8> {
        let mut r = vec![1, 0x64, 0x00, 0x1f];
        r.push(0xFC | (length_minus_one & 0x03));
        r.push(0xE0 | (sps.len() as u8 & 0x1F));
        for s in sps {
            r.extend_from_slice(&(s.len() as u16).to_be_bytes());
            r.extend_from_slice(s);
        }
        r.push(pps.len() as u8);
        for p in pps {
            r.extend_from_slice(&(p.len() as u16).to_be_bytes());
            r.extend_from_slice(p);
        }
        r
    }

    #[test]
    fn parse_avcc_record_reads_all_sets_and_length_size() {
        let rec = avcc_record(1, &[b"AAA", b"BBB"], &[b"P"]);
        let (mut sps, mut pps) = (Vec::new(), Vec::new());
        let ls = parse_avcc_record(&rec, &mut sps, &mut pps).expect("parses");
        assert_eq!(ls, 2, "lengthSizeMinusOne 1 -> 2-byte prefix");
        assert_eq!(sps, vec![b"AAA".to_vec(), b"BBB".to_vec()]);
        assert_eq!(pps, vec![b"P".to_vec()]);
    }

    #[test]
    fn avcc_to_annexb_honors_length_size() {
        // Two NALs with 2-byte length prefixes (not the usual 4).
        let mut avcc = Vec::new();
        avcc.extend_from_slice(&3u16.to_be_bytes());
        avcc.extend_from_slice(b"foo");
        avcc.extend_from_slice(&2u16.to_be_bytes());
        avcc.extend_from_slice(b"hi");
        let mut out = Vec::new();
        avcc_to_annexb(&avcc, 2, &mut out);
        assert_eq!(out, [0, 0, 0, 1, b'f', b'o', b'o', 0, 0, 0, 1, b'h', b'i']);
    }

    // A clip with TWO avc1 sample-description entries (a mid-recording param-set
    // switch) must contribute BOTH entries' SPS/PPS to the prefix -- prepending
    // only the first is what made openh264 lose every frame past the switch.
    #[test]
    fn collect_gathers_param_sets_from_all_avc1_entries() {
        let mk_avc1 = |rec: Vec<u8>| {
            let mut payload = vec![0u8; 78]; // VisualSampleEntry header
            payload.extend_from_slice(&bx(b"avcC", &rec));
            bx(b"avc1", &payload)
        };
        let avc1_a = mk_avc1(avcc_record(3, &[b"SPS_A"], &[b"PPS_A"]));
        let avc1_b = mk_avc1(avcc_record(3, &[b"SPS_B"], &[b"PPS_B"]));
        let mut stsd_payload = vec![0, 0, 0, 0, 0, 0, 0, 2]; // version/flags, entry_count=2
        stsd_payload.extend_from_slice(&avc1_a);
        stsd_payload.extend_from_slice(&avc1_b);
        let moov = bx(
            b"moov",
            &bx(
                b"trak",
                &bx(
                    b"mdia",
                    &bx(b"minf", &bx(b"stbl", &bx(b"stsd", &stsd_payload))),
                ),
            ),
        );
        let mut input = bx(b"ftyp", b"isom");
        input.extend_from_slice(&moov);

        let (sps, pps, ls) = collect_avcc_param_sets(&input).expect("param sets parsed");
        assert_eq!(ls, 4);
        assert_eq!(sps, vec![b"SPS_A".to_vec(), b"SPS_B".to_vec()]);
        assert_eq!(pps, vec![b"PPS_A".to_vec(), b"PPS_B".to_vec()]);
    }

    // Encode a handful of distinct frames, mux, then decode: exercises the raw
    // concealing decode path (decode_au_concealed + flush_all_concealed + YUV->RGB)
    // on a clean stream. Every frame must round-trip (no drops, correct dims).
    #[test]
    fn concealing_decode_roundtrips_a_clean_clip() {
        let (w, h, n, fps) = (64usize, 48usize, 12usize, 15u32);
        let mut sink = Mp4VideoSink::new(fps, None);
        for i in 0..n {
            let mut rgb = vec![0u8; w * h * 3];
            for y in 0..h {
                for x in 0..w {
                    let t = (y * w + x) * 3;
                    rgb[t] = if x == (i * 3) % w { 255 } else { 0 }; // moving bar
                    rgb[t + 1] = ((i * 20) % 256) as u8;
                    rgb[t + 2] = 128;
                }
            }
            sink.push(&Image::from_rgb8(w, h, &rgb))
                .expect("encode frame");
        }
        let (ow, oh, cnt, sps, pps, samples) = sink.finish().expect("finish encode");
        assert_eq!(cnt, n);
        let tmp =
            std::env::temp_dir().join(format!("thinfer_dec_roundtrip_{}.mp4", std::process::id()));
        mux_mp4(&tmp, ow, oh, fps, &sps, &pps, &samples, None).expect("mux");
        let bytes = std::fs::read(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);
        let (frames, got_fps) = decode_mp4_frames(&bytes).expect("decode");
        assert_eq!(got_fps, fps);
        assert_eq!(frames.len(), n, "all encoded frames decode back (no drops)");
        assert_eq!((frames[0].w, frames[0].h), (ow, oh));
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

    /// Build a `trak` box with the given `mdia/hdlr` handler_type.
    fn trak(handler: &[u8; 4]) -> Vec<u8> {
        // hdlr payload: version+flags (4) + pre_defined (4) + handler_type (4) + 0.
        let mut hdlr = vec![0u8; 8];
        hdlr.extend_from_slice(handler);
        bx(b"trak", &bx(b"mdia", &bx(b"hdlr", &hdlr)))
    }

    #[test]
    fn strip_audio_rewrites_only_the_soun_trak_to_free() {
        let ftyp = bx(b"ftyp", b"isom");
        let vtrak = trak(b"vide");
        let atrak = trak(b"soun");
        let moov = bx(b"moov", &[vtrak.clone(), atrak.clone()].concat());
        let mdat = bx(b"mdat", &[0xAAu8; 16]);
        let mut input = Vec::new();
        for b in [&ftyp, &moov, &mdat] {
            input.extend_from_slice(b);
        }

        let out = strip_audio_traks(&input);
        // Same length (in-place rewrite), mdat untouched.
        assert_eq!(out.len(), input.len(), "length must be preserved");
        assert_eq!(&out[input.len() - mdat.len()..], mdat.as_slice());
        // The audio trak header is now `free`; exactly one `trak` remains (video).
        let n_trak = out.windows(4).filter(|w| *w == b"trak").count();
        let n_free = out.windows(4).filter(|w| *w == b"free").count();
        assert_eq!(n_trak, 1, "only the video trak should remain a trak");
        assert_eq!(n_free, 1, "the audio trak became a free box");
        // The `soun` handler bytes still sit inside the now-ignored free box.
        assert!(out.windows(4).any(|w| w == b"vide"));
    }

    #[test]
    fn strip_audio_is_a_noop_without_audio() {
        let ftyp = bx(b"ftyp", b"isom");
        let moov = bx(b"moov", &trak(b"vide"));
        let mut input = Vec::new();
        input.extend_from_slice(&ftyp);
        input.extend_from_slice(&moov);
        // Borrowed (no copy) when there's nothing to strip.
        assert!(matches!(
            strip_audio_traks(&input),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    /// Build a video `trak` whose stbl/stsd holds one `avc1` sample entry with
    /// the given child boxes (in order) after its 78-byte visual sample header.
    fn video_trak_with_avc1_children(children: &[Vec<u8>]) -> Vec<u8> {
        let mut avc1_payload = vec![0u8; 78]; // visual sample entry fields (opaque here)
        for c in children {
            avc1_payload.extend_from_slice(c);
        }
        let avc1 = bx(b"avc1", &avc1_payload);
        // stsd is a FullBox: version+flags (4) + entry_count (4) then the entries.
        let mut stsd_payload = vec![0, 0, 0, 0, 0, 0, 0, 1];
        stsd_payload.extend_from_slice(&avc1);
        let hdlr = {
            let mut h = vec![0u8; 8];
            h.extend_from_slice(b"vide");
            bx(b"hdlr", &h)
        };
        let mdia = bx(
            b"mdia",
            &[hdlr, bx(b"minf", &bx(b"stbl", &bx(b"stsd", &stsd_payload)))].concat(),
        );
        bx(b"trak", &mdia)
    }

    #[test]
    fn reorder_hoists_avcc_ahead_of_a_leading_colr_box() {
        // avc1 children as some muxers emit them: colr BEFORE avcC.
        let colr = bx(b"colr", b"nclx____");
        let avcc = bx(b"avcC", b"\x01dummy-config");
        let trak = video_trak_with_avc1_children(&[colr.clone(), avcc.clone()]);
        let moov = bx(b"moov", &trak);
        let mut input = bx(b"ftyp", b"isom");
        input.extend_from_slice(&moov);

        let out = reorder_avcc_first(&input);
        assert!(matches!(out, std::borrow::Cow::Owned(_)), "should rewrite");
        let out = out.into_owned();
        // Length preserved (pure permutation of the avc1 child region).
        assert_eq!(out.len(), input.len());
        // avcC now precedes colr; both boxes still present intact. `find`
        // locates the 4-byte box TYPE, which sits 4 bytes past the box start
        // (after the size field), so back up to slice the whole box.
        let avcc_box = find(&out, b"avcC") - 4;
        let colr_box = find(&out, b"colr") - 4;
        assert!(avcc_box < colr_box, "avcC must lead its avc1 siblings");
        assert_eq!(&out[avcc_box..avcc_box + avcc.len()], avcc.as_slice());
        assert_eq!(&out[colr_box..colr_box + colr.len()], colr.as_slice());
    }

    #[test]
    fn reorder_is_a_noop_when_avcc_already_leads() {
        let avcc = bx(b"avcC", b"\x01cfg");
        let pasp = bx(b"pasp", b"____");
        let trak = video_trak_with_avc1_children(&[avcc, pasp]);
        let moov = bx(b"moov", &trak);
        let mut input = bx(b"ftyp", b"isom");
        input.extend_from_slice(&moov);
        // avcC is already first -> borrowed, no copy.
        assert!(matches!(
            reorder_avcc_first(&input),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    // Audio passthrough must survive the reorder+demux path: extract runs on the
    // audio-INTACT bytes, so a real AAC-LC track round-trips. This guards the
    // regression where strip_audio_traks ran BEFORE extraction and silently
    // dropped all audio (video-only output).
    #[test]
    fn audio_extracts_from_intact_bytes_but_not_after_stripping() {
        use mp4::{AacConfig, AudioObjectType, AvcConfig, ChannelConfig, SampleFreqIndex};

        // Mux a 1-frame H.264 + AAC-LC clip via the mp4 writer (avcC is written
        // first, so this exercises audio, not the reorder path).
        let config = mp4::Mp4Config {
            major_brand: str::parse("isom").unwrap(),
            minor_version: 0,
            compatible_brands: vec![str::parse("isom").unwrap()],
            timescale: 1000,
        };
        let mut writer =
            mp4::Mp4Writer::write_start(std::io::Cursor::new(Vec::new()), &config).unwrap();
        writer
            .add_track(&mp4::TrackConfig {
                track_type: mp4::TrackType::Video,
                timescale: 30,
                language: "und".to_string(),
                media_conf: mp4::MediaConfig::AvcConfig(AvcConfig {
                    width: 16,
                    height: 16,
                    seq_param_set: vec![0x67, 0x42, 0x00, 0x0a, 0xf8, 0x41, 0xa2],
                    pic_param_set: vec![0x68, 0xce, 0x3c, 0x80],
                }),
            })
            .unwrap();
        writer
            .add_track(&mp4::TrackConfig {
                track_type: mp4::TrackType::Audio,
                timescale: 44100,
                language: "und".to_string(),
                media_conf: mp4::MediaConfig::AacConfig(AacConfig {
                    bitrate: 128000,
                    profile: AudioObjectType::AacLowComplexity,
                    freq_index: SampleFreqIndex::Freq44100,
                    chan_conf: ChannelConfig::Stereo,
                }),
            })
            .unwrap();
        // One video sample, two audio samples (opaque bytes; the demuxer treats
        // AAC frames as raw payloads).
        writer
            .write_sample(
                1,
                &mp4::Mp4Sample {
                    start_time: 0,
                    duration: 1,
                    rendering_offset: 0,
                    is_sync: true,
                    bytes: vec![0u8; 4].into(),
                },
            )
            .unwrap();
        for i in 0..2 {
            writer
                .write_sample(
                    2,
                    &mp4::Mp4Sample {
                        start_time: i * 1024,
                        duration: 1024,
                        rendering_offset: 0,
                        is_sync: true,
                        bytes: vec![i as u8; 6].into(),
                    },
                )
                .unwrap();
        }
        writer.write_end().unwrap();
        let clip = writer.into_writer().into_inner();

        // Intact bytes: audio is recovered (the fixed path).
        let recovered = extract_audio_from_bytes(&clip).expect("audio should passthrough");
        assert_eq!(recovered.samples.len(), 2, "both AAC samples pass through");
        assert_eq!(recovered.samples[1].bytes.as_ref(), &[1u8; 6]);

        // Strip-first bytes: audio is gone (the bug this fix avoids by extracting
        // before stripping).
        let stripped = strip_audio_traks(&clip).into_owned();
        assert!(
            extract_audio_from_bytes(&stripped).is_none(),
            "stripping before extraction drops audio (regression guard)"
        );
    }

    fn box_offset(data: &[u8], kind: &[u8]) -> usize {
        // Box TYPE sits 4 bytes past the box start (after the size field).
        find(data, kind) - 4
    }

    // A muxed video+audio clip must be faststart (moov before mdat) -- else mobile
    // browsers show a 0:00 white frame with no MediaError (the file still
    // downloads + plays in desktop players). Guards faststart across BOTH tracks.
    #[test]
    fn mux_with_audio_is_faststart() {
        use mp4::{AacConfig, AudioObjectType, ChannelConfig, Mp4Sample, SampleFreqIndex};

        let sps = vec![0x67, 0x42, 0x00, 0x0a, 0xf8, 0x41, 0xa2];
        let pps = vec![0x68, 0xce, 0x3c, 0x80];
        let samples: Vec<(bool, Vec<u8>)> = (0..8).map(|i| (i == 0, vec![i as u8; 32])).collect();
        let audio = AudioPassthrough {
            config: AacConfig {
                bitrate: 128000,
                profile: AudioObjectType::AacLowComplexity,
                freq_index: SampleFreqIndex::Freq44100,
                chan_conf: ChannelConfig::Stereo,
            },
            timescale: 44100,
            samples: (0..8)
                .map(|i| Mp4Sample {
                    start_time: i as u64 * 1024,
                    duration: 1024,
                    rendering_offset: 0,
                    is_sync: true,
                    bytes: vec![i as u8; 6].into(),
                })
                .collect(),
        };

        let tmp =
            std::env::temp_dir().join(format!("thinfer_mux_faststart_{}.mp4", std::process::id()));
        mux_mp4(&tmp, 16, 16, 24, &sps, &pps, &samples, Some(audio)).unwrap();
        let data = std::fs::read(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);

        let moov = box_offset(&data, b"moov");
        let mdat = box_offset(&data, b"mdat");
        assert!(
            moov < mdat,
            "moov (@{moov}) must precede mdat (@{mdat}) for inline playback"
        );
    }

    #[test]
    fn trim_audio_keeps_window_and_rebases() {
        use mp4::{AacConfig, AudioObjectType, ChannelConfig, Mp4Sample, SampleFreqIndex};
        // 10 samples of 1024 ticks each at 44100 Hz (~0.0232s apart). Window
        // [0.05s, 0.15s) -> ticks [2205, 6615) -> keeps samples 3,4,5,6.
        let mk = |i: u64| Mp4Sample {
            start_time: i * 1024,
            duration: 1024,
            rendering_offset: 0,
            is_sync: true,
            bytes: vec![i as u8; 4].into(),
        };
        let audio = AudioPassthrough {
            config: AacConfig {
                bitrate: 128000,
                profile: AudioObjectType::AacLowComplexity,
                freq_index: SampleFreqIndex::Freq44100,
                chan_conf: ChannelConfig::Stereo,
            },
            timescale: 44100,
            samples: (0..10).map(mk).collect(),
        };
        let out = trim_audio(Some(audio), 0.05, 0.15).expect("window is non-empty");
        assert_eq!(out.samples.len(), 4, "samples 3..=6 fall in the window");
        // First kept sample (index 3, start 3072) rebases to 3072-2205=867.
        assert_eq!(out.samples[0].start_time, 3072 - 2205);
        assert_eq!(out.samples[0].bytes.as_ref(), &[3u8; 4]);
        // A window past the end yields no audio (None, not an empty track).
        let audio2 = AudioPassthrough {
            config: out.config.clone(),
            timescale: 44100,
            samples: (0..10).map(mk).collect(),
        };
        assert!(trim_audio(Some(audio2), 100.0, 200.0).is_none());
    }
}
