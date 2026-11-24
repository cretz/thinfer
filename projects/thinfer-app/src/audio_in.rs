//! Reference audio input: decode any common container (WAV, FLAC, MP3, AAC /
//! M4A, Ogg Vorbis) and bring it to the audio VAE's format the way the H3
//! reference does (`prepare_reference_waveform`): trim to the clip, mono to
//! stereo, then torchaudio's default sinc resample to the model rate.

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use zeroize::Zeroizing;

/// The containers and codecs [`decode`] reads, for the model catalog.
pub const FORMATS: &[&str] = &["wav", "flac", "mp3", "m4a", "aac", "ogg"];

/// Decoded audio: `channels` planar f32 tracks at `rate` Hz.
pub struct Decoded {
    pub channels: Vec<Vec<f32>>,
    pub rate: u32,
}

impl Decoded {
    pub fn samples(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }
}

/// Decode the first audio track of `bytes`.
pub fn decode(bytes: &[u8]) -> Result<Decoded, String> {
    // The source owns a wiped-on-drop copy: request media never outlives the job.
    let cursor = std::io::Cursor::new(Zeroizing::new(bytes.to_vec()));
    let mss = MediaSourceStream::new(Box::new(cursor), MediaSourceStreamOptions::default());
    let probed = symphonia::default::get_probe()
        .format(
            &Hint::new(),
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| format!("reference audio: unrecognized format ({e})"))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("reference audio: no audio track")?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("reference audio: unsupported codec ({e})"))?;
    let mut rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut channels: Vec<Vec<f32>> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(format!("reference audio: {e}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let buf = match decoder.decode(&packet) {
            Ok(b) => b,
            // A corrupt frame is skipped, as players do.
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(format!("reference audio: {e}")),
        };
        let spec = *buf.spec();
        rate = spec.rate;
        let n = spec.channels.count();
        let frames = buf.frames();
        let mut sb = SampleBuffer::<f32>::new(buf.capacity() as u64, spec);
        sb.copy_planar_ref(buf);
        if channels.is_empty() {
            channels = vec![Vec::new(); n];
        }
        if channels.len() != n {
            return Err("reference audio: channel count changes mid-stream".into());
        }
        for (c, track) in channels.iter_mut().enumerate() {
            track.extend_from_slice(&sb.samples()[c * frames..][..frames]);
        }
    }
    if channels.is_empty() || channels[0].is_empty() || rate == 0 {
        return Err("reference audio: no samples".into());
    }
    Ok(Decoded { channels, rate })
}

/// The reference's `prepare_reference_waveform`: keep at most
/// `max_seconds`, mono to stereo (more channels keep the front pair), then
/// resample to `target_rate`. Returns `[2, samples]` planar.
pub fn prepare(audio: Decoded, target_rate: u32, max_seconds: f64) -> (Vec<f32>, usize) {
    let keep = ((max_seconds * audio.rate as f64) as usize).min(audio.samples());
    let mut tracks: Vec<Vec<f32>> = audio
        .channels
        .into_iter()
        .take(2)
        .map(|mut t| {
            t.truncate(keep);
            t
        })
        .collect();
    if tracks.len() == 1 {
        tracks.push(tracks[0].clone());
    }
    let tracks: Vec<Vec<f32>> = if audio.rate == target_rate {
        tracks
    } else {
        let kernel = SincKernel::new(audio.rate, target_rate);
        tracks.iter().map(|t| kernel.apply(t)).collect()
    };
    let n = tracks[0].len();
    (tracks.concat(), n)
}

/// torchaudio's `_get_sinc_resample_kernel` at its defaults
/// (`sinc_interp_hann`, `lowpass_filter_width = 6`, `rolloff = 0.99`), the
/// resampler the reference uses.
struct SincKernel {
    orig: usize,
    new: usize,
    width: usize,
    /// `[new, 2 * width + orig]` row-major.
    taps: Vec<f32>,
}

impl SincKernel {
    const LOWPASS_WIDTH: f64 = 6.0;
    const ROLLOFF: f64 = 0.99;

    fn new(orig: u32, new: u32) -> Self {
        let g = gcd(orig, new);
        let (orig, new) = ((orig / g) as usize, (new / g) as usize);
        // `transforms.Resample` builds its kernel in f64 (no dtype given).
        let base = orig.min(new) as f64 * Self::ROLLOFF;
        let lw = Self::LOWPASS_WIDTH;
        let width = (lw * orig as f64 / base).ceil() as usize;
        let k = 2 * width + orig;
        let mut taps = Vec::with_capacity(new * k);
        let scale = base / orig as f64;
        for i in 0..new {
            let shift = -(i as f64) / new as f64;
            for j in 0..k {
                let idx = (j as f64 - width as f64) / orig as f64;
                let t = ((shift + idx) * base).clamp(-lw, lw);
                let window = (t * std::f64::consts::PI / lw / 2.0).cos().powi(2);
                let t = t * std::f64::consts::PI;
                let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
                taps.push((sinc * window * scale) as f32);
            }
        }
        Self {
            orig,
            new,
            width,
            taps,
        }
    }

    /// torchaudio's `_apply_sinc_resample_kernel`: zero-pad `(width, width +
    /// orig)`, a stride-`orig` conv with `new` phases, interleave, trim to
    /// `ceil(new * len / orig)`.
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let k = 2 * self.width + self.orig;
        let at = |i: isize| -> f32 {
            if i < 0 || i as usize >= x.len() {
                0.0
            } else {
                x[i as usize]
            }
        };
        let target = (self.new * x.len()).div_ceil(self.orig);
        let mut out = Vec::with_capacity(target);
        'outer: for frame in 0.. {
            let start = (frame * self.orig) as isize - self.width as isize;
            for phase in 0..self.new {
                if out.len() == target {
                    break 'outer;
                }
                let taps = &self.taps[phase * k..][..k];
                let acc: f32 = taps
                    .iter()
                    .enumerate()
                    .map(|(j, w)| w * at(start + j as isize))
                    .sum();
                out.push(acc);
            }
        }
        out
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let data = (samples.len() * 2) as u32;
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&(36 + data).to_le_bytes());
        b.extend_from_slice(b"WAVEfmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&channels.to_le_bytes());
        b.extend_from_slice(&rate.to_le_bytes());
        b.extend_from_slice(&(rate * channels as u32 * 2).to_le_bytes());
        b.extend_from_slice(&(channels * 2).to_le_bytes());
        b.extend_from_slice(&16u16.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&data.to_le_bytes());
        for s in samples {
            b.extend_from_slice(&s.to_le_bytes());
        }
        b
    }

    #[test]
    fn decodes_interleaved_wav_to_planar() {
        let d = decode(&wav(16_000, 2, &[1000, -1000, 2000, -2000])).unwrap();
        assert_eq!(d.rate, 16_000);
        assert_eq!(d.channels.len(), 2);
        assert!((d.channels[0][1] - 2000.0 / 32768.0).abs() < 1e-6);
        assert!((d.channels[1][0] + 1000.0 / 32768.0).abs() < 1e-6);
    }

    /// Mono doubles to stereo; the trim applies before the resample.
    #[test]
    fn prepare_trims_and_doubles_mono() {
        let audio = Decoded {
            channels: vec![vec![0.25; 32_000 * 3]],
            rate: 32_000,
        };
        let (wave, n) = prepare(audio, 32_000, 2.0);
        assert_eq!(n, 64_000);
        assert_eq!(wave.len(), 2 * n);
        assert!(wave.iter().all(|&v| v == 0.25));
    }

    /// 48 kHz -> 32 kHz keeps the length rule, a DC level (the kernel sums to
    /// ~1 away from the edges), and a 1 kHz tone's amplitude.
    #[test]
    fn resample_keeps_length_dc_and_tone() {
        let k = SincKernel::new(48_000, 32_000);
        assert_eq!((k.orig, k.new), (3, 2));
        let dc = k.apply(&vec![1.0; 4800]);
        assert_eq!(dc.len(), 3200);
        assert!(dc[100..3100].iter().all(|v| (v - 1.0).abs() < 2e-3));

        let tone: Vec<f32> = (0..48_000)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48_000.0).sin())
            .collect();
        let out = SincKernel::new(48_000, 32_000).apply(&tone);
        let peak = out[1000..31_000].iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!((peak - 1.0).abs() < 1e-2, "peak {peak}");
        // 44.1 kHz: the awkward ratio (441 / 320, a 9-sample half width).
        let k = SincKernel::new(44_100, 32_000);
        assert_eq!((k.orig, k.new, k.width), (441, 320, 9));
        assert_eq!(k.apply(&vec![0.0; 44_100]).len(), 32_000);
    }
}
