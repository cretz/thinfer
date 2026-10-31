//! Packed-sequence geometry for MiniMax-H3.
//!
//! The transformer runs over a single packed 1-D sequence holding every modality
//! at once. For the text / keyframe tasks the row order is
//!
//! ```text
//! [ text (L) | keyframe conditions (C) | target audio (A) | target video (V) ]
//! ```
//!
//! Everything here exists to place a row in that sequence and give it its
//! `(t, h, w)` rotary coordinate.
//!
//! **The grids are built in f64.** Video and audio share one 40-units-per-second
//! rotary clock (video advances `5/3` rotary units per pixel frame at 24 fps,
//! audio one unit per latent at 40 latents/s) and that shared clock IS the
//! audio/video alignment. Accumulating it in f32 drifts the two apart.
//!
//! **No padding is ever emitted.** Upstream pads to a multiple of 64 and keeps
//! the tail as a separate attention document; a padless sequence needs no
//! attention mask at all, which keeps the fast unmasked sdpa path available.

use super::{DitConfig, MODALITY_NUM, TAG_AUDIO, TAG_VIDEO};

/// MiniMax-H3 generates at a fixed 24 fps.
pub const FPS: usize = 24;
/// Released for a 768-pixel short edge only.
pub const SHORT_EDGE: f64 = 768.0;
/// Soft area cap; both axes are then rounded to [`CANVAS_MULTIPLE`], so the
/// final area can land slightly above this budget.
pub const MAX_PIXELS: f64 = 768.0 * 1344.0;
pub const CANVAS_MULTIPLE: usize = 32;
pub const MIN_ASPECT_RATIO: f64 = 0.25;
pub const MAX_ASPECT_RATIO: f64 = 4.0;
pub const MIN_DURATION: f64 = 5.0;
pub const MAX_DURATION: f64 = 15.0;

/// The video VAE encodes 17 pixel frames per chunk and drops the 3 trailing
/// latent frames of every chunk, so `17n + 5` pixel frames map to `5n + 2`
/// latent frames.
pub const FRAMES_PER_CHUNK: usize = 17;
pub const LATENTS_PER_CHUNK: usize = 5;

/// The audio VAE hops 800 samples at 32 kHz: 40 latents per second. Stereo is
/// carried as two channel-major blocks of audio rows.
pub const AUDIO_LATENTS_PER_SECOND: usize = 40;
pub const AUDIO_CHANNELS: usize = 2;
pub const AUDIO_SAMPLE_RATE: u32 = 32_000;

/// H3 conditions on the unnormalized hidden state its Qwen3-VL conditioner
/// produces after the 50th of its 64 decoder layers (`hidden_states[0]` is the
/// embedding output).
pub const TEXT_ENCODER_LAYER: usize = 50;

/// Keyframe latents are noised to this level and held there for every step.
pub const KEYFRAME_NOISE_AUG: f32 = 0.999;

/// One latent frame spans `5/3 * frames_per_latent`; the `(1, 4, 4, 4, 4)`
/// pattern mirrors the VAE's 17-pixel-frames-to-5-latent-frames grouping.
const ROPE_FRAME_RESCALE: f64 = 5.0 / 3.0;
const ROPE_FRAMES_PER_LATENT: [f64; 5] = [1.0, 4.0, 4.0, 4.0, 4.0];
/// The spatial axes are normalized by the square root of the latent area and
/// scaled by 32, so a square canvas spans `[0, 32)`.
const ROPE_SPATIAL_SCALE: f64 = 32.0;

/// Round half to even, matching Python's `round`. Rust's `f64::round` is
/// half-away-from-zero, which would disagree on exact `.5` canvas and audio
/// sizes and silently shift the geometry off the reference.
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

#[derive(Debug, PartialEq)]
pub enum GeometryError {
    NonPositiveAspect(f64, f64),
    /// `(width, height, ratio)`.
    AspectOutOfRange(f64, f64, f64),
    CanvasNotAligned(usize, usize),
    NoFrames,
    /// Frame counts must be `17n + 5` for the video VAE to chunk them.
    UnalignedFrames(usize),
    PatchMismatch([usize; 5], (usize, usize, usize)),
}

impl core::fmt::Display for GeometryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonPositiveAspect(w, h) => {
                write!(f, "aspect ratio must be positive, got {w}:{h}")
            }
            Self::AspectOutOfRange(w, h, r) => write!(
                f,
                "MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got {w}:{h} ({r})"
            ),
            Self::CanvasNotAligned(h, w) => {
                write!(f, "height and width must be multiples of 32, got {h}x{w}")
            }
            Self::NoFrames => write!(f, "num_frames must be positive"),
            Self::UnalignedFrames(n) => {
                write!(f, "num_frames must be of the form 17n + 5, got {n}")
            }
            Self::PatchMismatch(shape, patch) => write!(
                f,
                "latents of shape {shape:?} are not divisible by the patch {patch:?}"
            ),
        }
    }
}

impl core::error::Error for GeometryError {}

/// Where a keyframe conditioning block sits on the rotary time axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyframeAnchor {
    First,
    Last,
}

/// One clean video-latent conditioning block in the packed sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Condition {
    /// FL2VA: a frame of the target canvas pinned at a clip end.
    Keyframe(KeyframeAnchor),
    /// Ref2VA: an image at its own latent size, on its own aspect-normalized
    /// spatial grid, holding ONE rotary time slot (references take slots
    /// `L, L + 1, ...` after the `L` text rows and the target starts after).
    Reference {
        latent_height: usize,
        latent_width: usize,
    },
}

impl Condition {
    /// Packed rows this block adds at `patch`, given the target's latent size.
    pub fn rows(&self, patch: (usize, usize, usize), target: (usize, usize)) -> usize {
        let (h, w) = match *self {
            Condition::Keyframe(_) => target,
            Condition::Reference {
                latent_height,
                latent_width,
            } => (latent_height, latent_width),
        };
        (h / patch.1) * (w / patch.2)
    }
}

/// An aspect-normalized `(h, w)` patch grid, row-major (`np.meshgrid(...,
/// indexing="ij")`).
fn frame_position_grid(
    latent_height: usize,
    latent_width: usize,
    patch: (usize, usize, usize),
) -> Vec<[f64; 2]> {
    let sqrt_area = ((latent_height * latent_width) as f64).sqrt();
    let height_grid = spatial_position_grid(latent_height, patch.1, sqrt_area);
    let width_grid = spatial_position_grid(latent_width, patch.2, sqrt_area);
    height_grid
        .iter()
        .flat_map(|&h| width_grid.iter().map(move |&w| [h, w]))
        .collect()
}

/// The structural description of one packed sequence.
#[derive(Clone, Debug)]
pub struct PackedSequence {
    pub seq_len: usize,
    /// `(t, h, w)` rotary coordinate per row, narrowed from the f64 host grid.
    pub position_ids: Vec<[f32; 3]>,
    pub token_tags: Vec<i32>,
    /// Conditioning rows first, then target rows.
    pub video_indices: Vec<u32>,
    /// Reference rows first (ref2va), then target rows.
    pub audio_indices: Vec<u32>,
    pub text_indices: Vec<u32>,
    pub num_condition_video_rows: usize,
    pub num_condition_audio_rows: usize,
}

/// Resolve a display aspect ratio into a canvas, returning `(height, width)`.
///
/// The short edge starts at 768, the area is capped at `768 * 1344`, and both
/// axes are rounded to the nearest multiple of 32. Only the ratio of the
/// arguments matters.
pub fn resolve_canvas_size(aspect_w: f64, aspect_h: f64) -> Result<(usize, usize), GeometryError> {
    if aspect_w <= 0.0 || aspect_h <= 0.0 {
        return Err(GeometryError::NonPositiveAspect(aspect_w, aspect_h));
    }
    let ratio = aspect_w / aspect_h;
    if !(MIN_ASPECT_RATIO..=MAX_ASPECT_RATIO).contains(&ratio) {
        return Err(GeometryError::AspectOutOfRange(aspect_w, aspect_h, ratio));
    }

    let (mut width, mut height) = if ratio >= 1.0 {
        (SHORT_EDGE * ratio, SHORT_EDGE)
    } else {
        (SHORT_EDGE, SHORT_EDGE / ratio)
    };

    let area = width * height;
    if area > MAX_PIXELS {
        let scale = (MAX_PIXELS / area).sqrt();
        width *= scale;
        height *= scale;
    }

    let m = CANVAS_MULTIPLE as f64;
    Ok((
        (round_half_even(height / m) * m).max(m) as usize,
        (round_half_even(width / m) * m).max(m) as usize,
    ))
}

/// Snap a frame count up to the next `17n + 5` the video VAE can encode.
pub fn align_num_frames(mut n: usize) -> Result<usize, GeometryError> {
    if n < 1 {
        return Err(GeometryError::NoFrames);
    }
    while n % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK {
        n += 1;
    }
    Ok(n)
}

/// Latent frames produced for a `17n + 5` frame count: `5n + 2`.
pub fn video_latent_num_frames(n: usize) -> Result<usize, GeometryError> {
    if n % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK {
        return Err(GeometryError::UnalignedFrames(n));
    }
    Ok((n - LATENTS_PER_CHUNK) / FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2)
}

/// Audio latents covering `n` video frames at 24 fps on the 40 Hz latent grid.
pub fn audio_latent_num_frames(n: usize) -> usize {
    round_half_even(n as f64 / FPS as f64 * AUDIO_LATENTS_PER_SECOND as f64) as usize
}

/// One aspect-normalized spatial rotary axis: `dim / patch` coordinates centred
/// on the unit interval, scaled by 32, right endpoint excluded.
///
/// This is numpy's `linspace(..., endpoint=False)`, i.e.
/// `start + arange(n) * (stop - start) / n`, which differs from the torch/MLX
/// formulation. Do not "simplify" it to a step-accumulating loop.
fn spatial_position_grid(dim: usize, patch: usize, sqrt_area: f64) -> Vec<f64> {
    let ratio = dim as f64 / sqrt_area;
    let left = (1.0 - ratio) / 2.0;
    let n = dim / patch;
    (0..n)
        .map(|i| (left + i as f64 * ratio / n as f64) * ROPE_SPATIAL_SCALE)
        .collect()
}

/// Rotary time of every latent frame from `origin`. Spacing cycles
/// `5/3 * (1, 4, 4, 4, 4)`, accumulated.
fn temporal_position_grid(num_latent_frames: usize, origin: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(num_latent_frames);
    let mut acc = origin;
    for i in 0..num_latent_frames {
        out.push(acc);
        acc += ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % ROPE_FRAMES_PER_LATENT.len()];
    }
    out
}

/// Rotary time spanned by `num_latent_frames` latent frames.
///
/// Only the `Last` keyframe anchor reads this, which the T2VA path never builds.
/// NB the reference sums this PAIRWISE (numpy) and notes the two summation
/// orders differ in the last ulp from 16 latent frames onwards; this is a plain
/// f64 sum and is not bit-exact to it. Revisit when keyframe conditioning lands.
fn temporal_position_span(num_latent_frames: usize) -> f64 {
    (0..num_latent_frames)
        .map(|i| ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % ROPE_FRAMES_PER_LATENT.len()])
        .sum()
}

/// Build the `[text | target audio | conditions | target video]` layout.
///
/// The reference orders conditions before the target audio; attention is
/// full and every row carries its own rotary position, tag and timestep, so
/// the row ORDER is free, and this one keeps all video rows (conditions then
/// target) contiguous for the video input projection.
///
/// `text_token_tags` is the modality tag of every text row: text is tagged
/// [`TAG_TEXT`], except the rows of an image's vision block, which H3 tags
/// [`TAG_VIDEO`]. That tag is what the AdaLN modulation keys off.
pub fn build_packed_sequence(
    text_token_tags: &[i32],
    num_latent_frames: usize,
    latent_height: usize,
    latent_width: usize,
    num_audio_latents: usize,
    patch: (usize, usize, usize),
    conditions: &[Condition],
) -> PackedSequence {
    let (_, ph, pw) = patch;
    let rows_per_frame = (latent_height / ph) * (latent_width / pw);
    let num_text = text_token_tags.len();
    let target = (latent_height, latent_width);
    let num_condition_rows: usize = conditions.iter().map(|c| c.rows(patch, target)).sum();
    let num_audio_rows = num_audio_latents * AUDIO_CHANNELS;
    let num_video_rows = num_latent_frames * rows_per_frame;
    let seq_len = num_text + num_condition_rows + num_audio_rows + num_video_rows;

    let audio_start = num_text;
    let condition_start = audio_start + num_audio_rows;
    let video_start = condition_start + num_condition_rows;

    let mut pos = vec![[0.0f64; 3]; seq_len];

    // Text rows sit on the time axis at their row index, so text length shifts
    // the whole media clock: media time starts at `num_text`, plus one slot
    // per reference image.
    for (i, p) in pos.iter_mut().take(num_text).enumerate() {
        p[0] = i as f64;
    }
    let references = conditions
        .iter()
        .filter(|c| matches!(c, Condition::Reference { .. }))
        .count();
    let origin = (num_text + references) as f64;

    let frame_grid = frame_position_grid(latent_height, latent_width, patch);
    debug_assert_eq!(frame_grid.len(), rows_per_frame);

    let (mut lo, mut slot) = (condition_start, num_text as f64);
    for condition in conditions {
        let (time, grid) = match *condition {
            Condition::Keyframe(KeyframeAnchor::First) => (origin, frame_grid.clone()),
            Condition::Keyframe(KeyframeAnchor::Last) => (
                origin + temporal_position_span(num_latent_frames) - ROPE_FRAME_RESCALE,
                frame_grid.clone(),
            ),
            Condition::Reference {
                latent_height,
                latent_width,
            } => {
                slot += 1.0;
                (
                    slot - 1.0,
                    frame_position_grid(latent_height, latent_width, patch),
                )
            }
        };
        for (row, g) in grid.iter().enumerate() {
            pos[lo + row] = [time, g[0], g[1]];
        }
        lo += grid.len();
    }

    // Audio rows are channel-major and share the video's rotary clock: one unit
    // per latent at 40 latents/s equals 24 fps * 5/3. They carry no height
    // coordinate and are pinned to the two extremes of the width grid.
    let w_first = frame_grid.first().map_or(0.0, |g| g[1]);
    let w_last = frame_grid.last().map_or(0.0, |g| g[1]);
    for i in 0..num_audio_rows {
        let latent = i % num_audio_latents.max(1);
        let w = if i < num_audio_latents {
            w_first
        } else {
            w_last
        };
        pos[audio_start + i] = [origin + latent as f64, 0.0, w];
    }

    let time_grid = temporal_position_grid(num_latent_frames, origin);
    for (f, &t) in time_grid.iter().enumerate() {
        let base = video_start + f * rows_per_frame;
        for (row, g) in frame_grid.iter().enumerate() {
            pos[base + row] = [t, g[0], g[1]];
        }
    }

    let video_indices: Vec<u32> = (condition_start as u32..seq_len as u32).collect();
    let audio_indices: Vec<u32> = (audio_start as u32..condition_start as u32).collect();
    let text_indices: Vec<u32> = (0..num_text as u32).collect();

    let mut token_tags = vec![TAG_VIDEO; seq_len];
    token_tags[..num_text].copy_from_slice(text_token_tags);
    for &i in &audio_indices {
        token_tags[i as usize] = TAG_AUDIO;
    }
    for &i in &video_indices {
        token_tags[i as usize] = TAG_VIDEO;
    }

    PackedSequence {
        seq_len,
        position_ids: pos
            .into_iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect(),
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        num_condition_video_rows: num_condition_rows,
        num_condition_audio_rows: 0,
    }
}

/// Assign a timestep to every row and reduce to the transformer's
/// `(distinct timesteps, per-row index)` pair.
///
/// One forward serves rows at different noise levels: generated video and audio
/// rows step down their own schedules while conditioning rows stay pinned at
/// their noise-augmentation level. Text rows never reach an output head and
/// inherit the video timestep.
pub fn build_row_timesteps(
    layout: &PackedSequence,
    video_timestep: f32,
    audio_timestep: f32,
    condition_video_timestep: f32,
    condition_audio_timestep: f32,
) -> (Vec<f32>, Vec<u32>) {
    let mut rows = vec![video_timestep; layout.seq_len];
    let n_cond_v = layout.num_condition_video_rows;
    let n_cond_a = layout.num_condition_audio_rows;

    for &i in &layout.video_indices[..n_cond_v] {
        rows[i as usize] = condition_video_timestep;
    }
    for &i in &layout.audio_indices[n_cond_a..] {
        rows[i as usize] = audio_timestep;
    }
    for &i in &layout.audio_indices[..n_cond_a] {
        rows[i as usize] = condition_audio_timestep;
    }

    let mut distinct: Vec<f32> = rows.clone();
    distinct.sort_by(|a, b| a.partial_cmp(b).expect("timesteps are finite"));
    distinct.dedup();
    let index: Vec<u32> = rows
        .iter()
        .map(|t| {
            distinct
                .binary_search_by(|p| p.partial_cmp(t).expect("timesteps are finite"))
                .expect("every row timestep is in the distinct set") as u32
        })
        .collect();
    (distinct, index)
}

/// Row of the modulation table addressed by a `(timestep, modality)` pair.
/// Padding rows carry [`super::TAG_PAD`] and are clamped so they cannot index
/// backwards; they never reach the outputs.
pub fn modality_row(timestep_index: u32, tag: i32) -> u32 {
    timestep_index * MODALITY_NUM as u32 + tag.max(0) as u32
}

/// Pack `(C, F, H, W)` video latents into transformer rows, frame-major then
/// row-major. Returns `num_patches` rows of `C * prod(patch)`.
pub fn patchify_video_latents(
    latents: &[f32],
    channels: usize,
    frames: usize,
    height: usize,
    width: usize,
    patch: (usize, usize, usize),
) -> Result<Vec<f32>, GeometryError> {
    let (pt, ph, pw) = patch;
    if !frames.is_multiple_of(pt) || !height.is_multiple_of(ph) || !width.is_multiple_of(pw) {
        return Err(GeometryError::PatchMismatch(
            [1, channels, frames, height, width],
            patch,
        ));
    }
    let row_width = channels * pt * ph * pw;
    let mut out = vec![0.0f32; latents.len()];
    let mut w_at = 0usize;
    for f in (0..frames).step_by(pt) {
        for h in (0..height).step_by(ph) {
            for w in (0..width).step_by(pw) {
                // Row layout is (c, pt, ph, pw), matching the reference's
                // transpose(0, 2, 4, 6, 1, 3, 5, 7).
                for c in 0..channels {
                    for dt in 0..pt {
                        for dh in 0..ph {
                            for dw in 0..pw {
                                let src =
                                    ((c * frames + f + dt) * height + h + dh) * width + w + dw;
                                out[w_at] = latents[src];
                                w_at += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    debug_assert_eq!(w_at, out.len());
    debug_assert_eq!(out.len() % row_width, 0);
    Ok(out)
}

/// Inverse of [`patchify_video_latents`]. Returns `(C, F, H, W)`.
pub fn unpatchify_video_tokens(
    rows: &[f32],
    channels: usize,
    frames: usize,
    height: usize,
    width: usize,
    patch: (usize, usize, usize),
) -> Vec<f32> {
    let (pt, ph, pw) = patch;
    let mut out = vec![0.0f32; rows.len()];
    let mut r_at = 0usize;
    for f in (0..frames).step_by(pt) {
        for h in (0..height).step_by(ph) {
            for w in (0..width).step_by(pw) {
                for c in 0..channels {
                    for dt in 0..pt {
                        for dh in 0..ph {
                            for dw in 0..pw {
                                let dst =
                                    ((c * frames + f + dt) * height + h + dh) * width + w + dw;
                                out[dst] = rows[r_at];
                                r_at += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    debug_assert_eq!(r_at, rows.len());
    out
}

/// Row count a t2va request produces, for budgeting before any allocation.
pub fn sequence_rows(
    cfg: &DitConfig,
    num_text: usize,
    num_latent_frames: usize,
    latent_height: usize,
    latent_width: usize,
    num_audio_latents: usize,
) -> usize {
    let (_, ph, pw) = cfg.patch;
    let rows_per_frame = (latent_height / ph) * (latent_width / pw);
    num_text + num_latent_frames * rows_per_frame + num_audio_latents * AUDIO_CHANNELS
}

/// Longest `17n + 5` clip of at most `frames` frames whose packed sequence
/// (with `num_text` prompt rows) fits `limit` rows, or `None` when not even the
/// shortest clip does. Larger canvases trade clip length for resolution.
pub fn max_fitting_frames(
    cfg: &DitConfig,
    num_text: usize,
    latent_height: usize,
    latent_width: usize,
    frames: usize,
    limit: usize,
) -> Option<usize> {
    let mut best = None;
    let mut n = LATENTS_PER_CHUNK;
    while n <= frames.max(LATENTS_PER_CHUNK) {
        let latents = video_latent_num_frames(n).ok()?;
        let rows = sequence_rows(
            cfg,
            num_text,
            latents,
            latent_height,
            latent_width,
            audio_latent_num_frames(n),
        );
        if rows > limit {
            break;
        }
        best = Some(n);
        n += FRAMES_PER_CHUNK;
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minimax::TAG_TEXT;

    /// 5s at 832x480 fits the cap; 768p only fits a short clip.
    #[test]
    fn max_fitting_frames_trades_length_for_canvas() {
        let cfg = DitConfig::fasth3_trim();
        let fit = |h: usize, w: usize| max_fitting_frames(&cfg, 256, h / 16, w / 16, 124, 16384);
        assert_eq!(fit(480, 832), Some(124));
        let hd = fit(768, 1344).unwrap();
        assert!((22..124).contains(&hd), "1344x768 fits {hd} frames");
    }

    #[test]
    fn canvas_16_9_is_1344x768() {
        assert_eq!(resolve_canvas_size(16.0, 9.0).unwrap(), (768, 1344));
    }

    #[test]
    fn canvas_is_square_at_1_1() {
        assert_eq!(resolve_canvas_size(1.0, 1.0).unwrap(), (768, 768));
    }

    #[test]
    fn canvas_axes_are_always_multiples_of_32() {
        for (w, h) in [
            (16.0, 9.0),
            (9.0, 16.0),
            (4.0, 3.0),
            (21.0, 9.0),
            (1.0, 2.0),
        ] {
            let (ch, cw) = resolve_canvas_size(w, h).unwrap();
            assert_eq!(ch % 32, 0, "{w}:{h} height {ch}");
            assert_eq!(cw % 32, 0, "{w}:{h} width {cw}");
        }
    }

    #[test]
    fn canvas_rejects_extreme_aspects() {
        assert!(resolve_canvas_size(5.0, 1.0).is_err());
        assert!(resolve_canvas_size(0.0, 1.0).is_err());
    }

    #[test]
    fn frames_snap_to_17n_plus_5() {
        assert_eq!(align_num_frames(120).unwrap(), 124);
        assert_eq!(align_num_frames(5).unwrap(), 5);
        assert_eq!(align_num_frames(22).unwrap(), 22);
        for n in 1..300 {
            let a = align_num_frames(n).unwrap();
            assert!(a >= n);
            assert_eq!(a % 17, 5, "{n} -> {a}");
        }
    }

    #[test]
    fn latent_frames_are_5n_plus_2() {
        assert_eq!(video_latent_num_frames(5).unwrap(), 2);
        assert_eq!(video_latent_num_frames(22).unwrap(), 7);
        assert_eq!(video_latent_num_frames(124).unwrap(), 37);
        assert!(video_latent_num_frames(123).is_err());
    }

    #[test]
    fn audio_latents_track_24fps_at_40hz() {
        assert_eq!(audio_latent_num_frames(124), 207);
        assert_eq!(audio_latent_num_frames(24), 40);
    }

    /// The reference canvas: 5s 16:9 is 37,296 video rows + 414 audio.
    #[test]
    fn reference_5s_16x9_row_counts() {
        let (h, w) = resolve_canvas_size(16.0, 9.0).unwrap();
        let frames = align_num_frames(120).unwrap();
        let lf = video_latent_num_frames(frames).unwrap();
        let (lh, lw) = (h / 16, w / 16);
        assert_eq!((lh, lw), (48, 84));
        let rows_per_frame = (lh / 2) * (lw / 2);
        assert_eq!(rows_per_frame, 1008);
        assert_eq!(lf * rows_per_frame, 37_296);
        assert_eq!(audio_latent_num_frames(frames) * AUDIO_CHANNELS, 414);
    }

    fn tiny_layout(num_text: usize) -> PackedSequence {
        build_packed_sequence(&vec![TAG_TEXT; num_text], 2, 4, 4, 3, (1, 2, 2), &[])
    }

    #[test]
    fn layout_row_order_and_tags() {
        let l = tiny_layout(3);
        // 3 text + 0 cond + 6 audio (3 latents x 2ch) + 8 video (2 frames x 4)
        assert_eq!(l.seq_len, 3 + 6 + 8);
        assert_eq!(l.text_indices, vec![0, 1, 2]);
        assert_eq!(l.audio_indices, (3..9).collect::<Vec<u32>>());
        assert_eq!(l.video_indices, (9..17).collect::<Vec<u32>>());
        assert_eq!(&l.token_tags[..3], &[TAG_TEXT; 3]);
        assert!(l.token_tags[3..9].iter().all(|&t| t == TAG_AUDIO));
        assert!(l.token_tags[9..].iter().all(|&t| t == TAG_VIDEO));
    }

    /// Text length shifts the whole media clock: media time starts at num_text.
    #[test]
    fn text_length_shifts_the_media_clock() {
        for num_text in [0, 1, 7] {
            let l = tiny_layout(num_text);
            let first_audio = l.position_ids[l.audio_indices[0] as usize][0];
            let first_video = l.position_ids[l.video_indices[0] as usize][0];
            assert_eq!(first_audio, num_text as f32);
            assert_eq!(first_video, num_text as f32);
        }
    }

    /// Audio advances one rotary unit per latent; video advances 5/3 per pixel
    /// frame. The first video latent covers 1 frame, the rest 4.
    #[test]
    fn audio_and_video_share_one_rotary_clock() {
        let l = tiny_layout(0);
        let a0 = l.position_ids[l.audio_indices[0] as usize][0];
        let a1 = l.position_ids[l.audio_indices[1] as usize][0];
        assert_eq!(a1 - a0, 1.0);

        let rows_per_frame = 4;
        let v0 = l.position_ids[l.video_indices[0] as usize][0];
        let v1 = l.position_ids[l.video_indices[rows_per_frame] as usize][0];
        assert!(((v1 - v0) as f64 - 5.0 / 3.0).abs() < 1e-6);
    }

    /// Audio rows carry no height coordinate and sit on the two width extremes.
    #[test]
    fn audio_rows_pin_to_width_extremes() {
        let l = tiny_layout(0);
        let n = 3;
        for (i, &idx) in l.audio_indices.iter().enumerate() {
            let p = l.position_ids[idx as usize];
            assert_eq!(p[1], 0.0, "audio carries no height");
            if i < n {
                assert_eq!(p[2], l.position_ids[l.audio_indices[0] as usize][2]);
            }
        }
        let left = l.position_ids[l.audio_indices[0] as usize][2];
        let right = l.position_ids[l.audio_indices[n] as usize][2];
        assert_ne!(left, right, "the two channels take opposite extremes");
    }

    /// A square canvas spans [0, 32) on both spatial axes.
    #[test]
    fn square_canvas_spans_0_to_32() {
        let g = spatial_position_grid(4, 2, 4.0);
        assert_eq!(g.len(), 2);
        assert!((g[0] - 0.0).abs() < 1e-12);
        assert!((g[1] - 16.0).abs() < 1e-12);
    }

    #[test]
    fn row_timesteps_collapse_to_distinct_levels() {
        let l = tiny_layout(2);
        let (distinct, index) = build_row_timesteps(&l, 0.7, 0.4, 0.999, 1.0);
        // No conditioning rows: only the video (text inherits it) and audio levels.
        assert_eq!(distinct, vec![0.4, 0.7]);
        assert_eq!(index.len(), l.seq_len);
        for &i in &l.audio_indices {
            assert_eq!(distinct[index[i as usize] as usize], 0.4);
        }
        for &i in &l.video_indices {
            assert_eq!(distinct[index[i as usize] as usize], 0.7);
        }
        for &i in &l.text_indices {
            assert_eq!(
                distinct[index[i as usize] as usize], 0.7,
                "text inherits video"
            );
        }
    }

    #[test]
    fn modality_row_clamps_padding() {
        assert_eq!(modality_row(2, TAG_VIDEO), 6);
        assert_eq!(modality_row(2, TAG_AUDIO), 8);
        assert_eq!(modality_row(0, super::super::TAG_PAD), 0);
    }

    #[test]
    fn patchify_roundtrips() {
        let (c, f, h, w) = (2usize, 2usize, 4usize, 4usize);
        let src: Vec<f32> = (0..c * f * h * w).map(|i| i as f32).collect();
        let rows = patchify_video_latents(&src, c, f, h, w, (1, 2, 2)).unwrap();
        assert_eq!(rows.len(), src.len());
        let back = unpatchify_video_tokens(&rows, c, f, h, w, (1, 2, 2));
        assert_eq!(back, src);
    }

    #[test]
    fn patchify_row_width_matches_config() {
        let cfg = DitConfig::h3_33b();
        let (c, f, h, w) = (cfg.latents_dim, 2, 4, 4);
        let src = vec![0.0f32; c * f * h * w];
        let rows = patchify_video_latents(&src, c, f, h, w, cfg.patch).unwrap();
        assert_eq!(rows.len() / cfg.video_patch_dim(), 2 * 2 * 2);
    }

    #[test]
    fn round_half_even_matches_python() {
        assert_eq!(round_half_even(0.5), 0.0);
        assert_eq!(round_half_even(1.5), 2.0);
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(-0.5), -0.0);
        assert_eq!(round_half_even(2.4), 2.0);
        assert_eq!(round_half_even(2.6), 3.0);
    }
}
