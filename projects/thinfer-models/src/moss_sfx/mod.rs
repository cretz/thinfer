//! MOSS-SoundEffect v2.0: text to 48 kHz sound effects.
//!
//! Reference: `OpenMOSS/MOSS-TTS/moss_soundeffect_v2` (`MossSoundEffectPipeline`,
//! `WanAudioPipeline`).
//!
//! - The prompt (cleaned like the reference prompter, plus the training-time
//!   `" duration: X.Xs"` tag) runs through Qwen3-1.7B; the last hidden states
//!   (final-normed) are padded to [`TEXT_LEN`] rows with a zero tail. The CFG
//!   negative is the empty prompt: all zeros.
//! - A Wan2.1-1.3B DiT over the 1D DAC latent ([`crate::wan`] with
//!   [`WanGeometry::Audio`](crate::wan::dit_block::WanGeometry)) denoises a
//!   FIXED [`FULL_FRAMES`] latent (30 s) whatever the requested length:
//!   flow matching, shift 5, Euler, CFG 4, 100 steps.
//! - The DAC decoder turns latent frames into samples; only the frames the
//!   requested length needs (plus the decoder's receptive field) are decoded,
//!   which is exact.
//! - Opt-in seamless loop: each step rolls the clip's frames by a different
//!   offset before the DiT and back after, so no frame stays at the seam, and
//!   the decode reads the frames circularly.

pub mod dac;
pub mod manifest;
pub mod pipeline;
pub mod text;

/// Output rate (Hz).
pub const SAMPLE_RATE: u32 = 48_000;
/// Samples per latent frame (the DAC encoder strides 2 * 3 * 4 * 5 * 8).
pub const HOP: usize = 960;
/// Latent frames per second.
pub const FRAME_RATE: usize = SAMPLE_RATE as usize / HOP;
/// Latent channels.
pub const LATENT_CH: usize = 128;
/// The latent every run denoises (`max_inference_seconds`).
pub const FULL_SECONDS: f32 = 30.0;
pub const FULL_FRAMES: usize = 30 * FRAME_RATE;
/// Text rows the DiT cross-attends to (`WanPrompter(text_len=512)`).
pub const TEXT_LEN: usize = 512;

/// The reference's recommended sampling.
pub const DEFAULT_STEPS: u32 = 100;
pub const DEFAULT_GUIDANCE: f32 = 4.0;
pub const DEFAULT_SECONDS: f32 = 10.0;
/// Flow-match shift (`scheduler_config.json`).
pub const SHIFT: f64 = 5.0;
/// `num_train_timesteps`.
const TRAIN_TIMESTEPS: f64 = 1000.0;

/// Latent frames past a decode window the DAC decoder can see. Its receptive
/// field is under 12 frames (the k=7 input conv, then per stage the k=2s
/// transposed conv and three k=7 residual convs dilated 1/3/9 at that
/// stage's rate); twice that keeps windowed decodes bit-exact.
pub const DECODE_MARGIN: usize = 24;

/// Round a requested length like the reference (`round(seconds, 1)`).
pub fn round_seconds(seconds: f32) -> f32 {
    (seconds * 10.0).round() / 10.0
}

/// Samples of a `seconds` clip (`int(sample_rate * seconds)`).
pub fn samples_for(seconds: f32) -> usize {
    (SAMPLE_RATE as f64 * seconds as f64) as usize
}

/// Latent frames covering `samples`.
pub fn frames_for(samples: usize) -> usize {
    samples.div_ceil(HOP)
}

/// The prompt the text encoder reads: the reference's `_format` (strip +
/// duration tag), then its tokenizer clean (`whitespace_clean(basic_clean)`:
/// HTML unescaped twice, whitespace runs collapsed). `ftfy.fix_text` only
/// repairs mojibake, which a typed prompt does not carry.
pub fn prompt_text(prompt: &str, seconds: f32) -> String {
    let p = format!("{} duration: {:.1}s", prompt.trim(), round_seconds(seconds));
    let p = html_unescape(&html_unescape(&p));
    p.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Python `html.unescape` for the character references prompts carry:
/// numeric (`&#38;`, `&#x26;`) and the common named ones. Anything else stays
/// as written.
fn html_unescape(s: &str) -> String {
    const NAMED: [(&str, char); 8] = [
        ("amp", '&'),
        ("lt", '<'),
        ("gt", '>'),
        ("quot", '"'),
        ("apos", '\''),
        ("nbsp", '\u{a0}'),
        ("hellip", '\u{2026}'),
        ("mdash", '\u{2014}'),
    ];
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        let end = rest[1..]
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '#'))
            .map(|e| e + 1)
            .unwrap_or(rest.len());
        let name = &rest[1..end];
        let ch = if let Some(num) = name.strip_prefix('#') {
            let v = match num.strip_prefix(['x', 'X']) {
                Some(hex) => u32::from_str_radix(hex, 16).ok(),
                None => num.parse().ok(),
            };
            v.and_then(char::from_u32)
        } else {
            NAMED.iter().find(|(n, _)| *n == name).map(|&(_, c)| c)
        };
        match ch {
            Some(c) => {
                out.push(c);
                rest = &rest[end..];
                rest = rest.strip_prefix(';').unwrap_or(rest);
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// The step schedule: `(sigma, timestep)` per step, then 0 to finish
/// (`FlowMatchScheduler(extra_one_step=True, sigma_min=0)`).
pub fn schedule(steps: u32) -> Vec<f32> {
    (0..steps)
        .map(|i| {
            let s = 1.0 - i as f64 / steps as f64;
            (SHIFT * s / (1.0 + (SHIFT - 1.0) * s)) as f32
        })
        .collect()
}

/// The DiT timestep of `sigma`.
pub fn timestep(sigma: f32) -> f32 {
    (sigma as f64 * TRAIN_TIMESTEPS) as f32
}

/// Seamless-loop offset of step `i` over a loop of `frames`: a golden-ratio
/// walk, so consecutive steps put the seam far apart.
pub fn loop_offset(i: u32, frames: usize) -> usize {
    const PHI: f64 = 0.618_033_988_749_895;
    ((i as f64 * PHI).fract() * frames as f64) as usize % frames.max(1)
}

/// Rotate the first `frames` columns of a `[LATENT_CH, cols]` latent left by
/// `by` (column `by` moves to 0); the rest stay put.
pub fn roll_left(x: &mut [f32], cols: usize, frames: usize, by: usize) {
    for row in x.chunks_exact_mut(cols) {
        row[..frames].rotate_left(by);
    }
}

/// Inverse of [`roll_left`].
pub fn roll_right(x: &mut [f32], cols: usize, frames: usize, by: usize) {
    for row in x.chunks_exact_mut(cols) {
        row[..frames].rotate_right(by);
    }
}

/// The decode window of a clip: `[LATENT_CH, frames + 2 * margin]` from a
/// `[LATENT_CH, cols]` latent. A loop wraps (its tail before, its head after),
/// so the decoded clip joins end to start; a plain clip takes the frames that
/// follow it (or zeros past the latent) and nothing before (the reference
/// decodes from frame 0 with the convs' zero padding). Returns the window and
/// the frame the clip starts at inside it.
pub fn decode_window(
    x: &[f32],
    cols: usize,
    frames: usize,
    looped: bool,
    margin: usize,
) -> (Vec<f32>, usize) {
    let (lead, len) = if looped {
        (margin, frames + 2 * margin)
    } else {
        (0, (frames + margin).min(cols))
    };
    let mut w = Vec::with_capacity(LATENT_CH * len);
    for row in x.chunks_exact(cols) {
        for j in 0..len {
            let f = if looped {
                (j + frames - lead % frames) % frames
            } else {
                j
            };
            w.push(row[f]);
        }
    }
    (w, lead)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_carries_the_duration_tag() {
        assert_eq!(
            prompt_text("  A dog  barking\nin a park. ", 10.04),
            "A dog barking in a park. duration: 10.0s"
        );
        assert_eq!(
            prompt_text("rain &amp;amp; thunder &#x2014; far", 2.0),
            "rain & thunder \u{2014} far duration: 2.0s"
        );
        assert_eq!(prompt_text("R&D lab", 1.0), "R&D lab duration: 1.0s");
    }

    /// `linspace(1, 0, n + 1)[:-1]` shifted by 5: starts at 1, never reaches 0.
    #[test]
    fn schedule_matches_the_reference() {
        let s = schedule(100);
        assert_eq!(s.len(), 100);
        assert_eq!(s[0], 1.0);
        let last = 0.01f64;
        assert!((s[99] as f64 - 5.0 * last / (1.0 + 4.0 * last)).abs() < 1e-7);
        assert!(s.windows(2).all(|w| w[1] < w[0]));
        assert_eq!(timestep(s[0]), 1000.0);
    }

    #[test]
    fn roll_round_trips_and_spares_the_tail() {
        let cols = 6;
        let mut x: Vec<f32> = (0..2 * cols).map(|v| v as f32).collect();
        let orig = x.clone();
        roll_left(&mut x, cols, 4, 1);
        assert_eq!(&x[..cols], &[1.0, 2.0, 3.0, 0.0, 4.0, 5.0]);
        roll_right(&mut x, cols, 4, 1);
        assert_eq!(x, orig);
    }

    #[test]
    fn loop_window_wraps_both_ends() {
        let cols = 8;
        let x: Vec<f32> = (0..LATENT_CH * cols).map(|v| (v % cols) as f32).collect();
        let (w, lead) = decode_window(&x, cols, 4, true, 2);
        assert_eq!(lead, 2);
        assert_eq!(&w[..8], &[2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0]);
        let (w, lead) = decode_window(&x, cols, 4, false, 2);
        assert_eq!(lead, 0);
        assert_eq!(&w[..6], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(w.len(), LATENT_CH * 6);
    }
}
