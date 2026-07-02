//! Overlap-blend tiling geometry for video-VAE decoders (LTX, Hunyuan, ...).
//!
//! Pure host-side planning + blend math, shared across models. A decoder that
//! is spatially+temporally LOCAL (causal convs + pixel-shuffle upsample, no
//! global attention over the tiled region) decodes each overlapping latent tile
//! independently and blends seams with a partition-of-unity weight window:
//! linear feather ramps spatially, trapezoidal masks temporally. The two blend
//! windows are separable (every temporal tile pairs with every spatial tile), so
//! the per-output weight is the product of a temporal weight-sum and a spatial
//! weight-sum, each accumulated once.
//!
//! Per-model knobs are arguments, not baked constants: `spatial_scale` /
//! `temporal_scale` (latent cell -> output pixel/frame), `channels` (the tiled
//! tensor's channel count). Budget-seeding + the OOM-retry loop stay per-model
//! (each has its own peak-bytes calibration). A single tile (`f <= tf`,
//! `h,w <= tile`) reduces to unit weights -> bit-identical to an untiled decode.

/// Min/max latent tile side (cells). Min 2; max caps a roomy budget from a
/// TDR-prone megatile. (Shared defaults; a model may clamp tighter.)
pub const TILE_MIN: u32 = 2;
pub const TILE_MAX: u32 = 24;
/// Min/max latent temporal tile depth (frames).
pub const TEMPORAL_TILE_MIN: u32 = 2;
pub const TEMPORAL_TILE_MAX: usize = 16;

/// One temporal tile in a decode plan: latent frame range `[l0, l1)`, the output
/// frame range `[o0, o1)` it lands in, and the trapezoidal blend mask's
/// output-space ramp lengths (`lr`/`rr`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TempTile {
    pub l0: usize,
    pub l1: usize,
    pub o0: usize,
    pub o1: usize,
    pub lr: usize,
    pub rr: usize,
}

/// Plan temporal tiles over `f` latent frames with latent depth `tf` and latent
/// overlap `overlap`, for a causal temporal upsampler of scale `s`
/// (`temporal_scale`). A single tile (`f <= tf`) spans all frames with unit
/// ramps (-> bit-identical decode). Else: `split_by_size` then the causal shift
/// (each tile after the first starts one latent frame earlier, left ramp +1)
/// then map to output frames (`o0 = l0*s`, `o1 = (l1-1)*s+1`, `lr = 1+(lr-1)*s`,
/// `rr = rr*s`). The decoder emits exactly `o1-o0` frames per tile (the
/// architectural first-frame-drop makes `(l1-l0-1)*s+1` outputs).
pub fn plan_temporal_tiles(
    f: usize,
    tf: usize,
    overlap: usize,
    temporal_scale: usize,
) -> Vec<TempTile> {
    let s = temporal_scale;
    let map = |l0: usize, l1: usize, lr: usize, rr: usize| TempTile {
        l0,
        l1,
        o0: l0 * s,
        o1: (l1 - 1) * s + 1,
        lr: if lr == 0 { 0 } else { 1 + (lr - 1) * s },
        rr: rr * s,
    };
    if f <= tf {
        return vec![map(0, f, 0, 0)];
    }
    let step = tf.saturating_sub(overlap).max(1);
    let amount = ((f + tf - 2 * overlap).saturating_sub(1) / step).max(2);
    // Raw `split_by_size` intervals (start, end, left_ramp, right_ramp).
    let mut raw: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(amount);
    raw.push((0, tf, 0, overlap));
    for i in 1..amount - 1 {
        raw.push((i * step, i * step + tf, overlap, overlap));
    }
    raw.push(((amount - 1) * step, f, overlap, 0));
    // Causal shift on every tile after the first; then map to output space.
    raw.into_iter()
        .enumerate()
        .map(|(i, (st, en, lr, rr))| {
            if i == 0 {
                map(st, en, lr, rr)
            } else {
                map(st - 1, en, lr + 1, rr)
            }
        })
        .collect()
}

/// 1-D trapezoidal blend mask of `length` with linear `ramp_left` fade-in
/// (starting from 0) and `ramp_right` fade-out, holding 1 between. Mirrors
/// upstream `compute_trapezoidal_mask_1d(..., left_starts_from_0=True)`.
pub fn trapezoid_mask(length: usize, ramp_left: usize, ramp_right: usize) -> Vec<f32> {
    let rl = ramp_left.min(length);
    let rr = ramp_right.min(length);
    let mut m = vec![1.0f32; length];
    if rl > 0 {
        for (i, v) in m.iter_mut().take(rl).enumerate() {
            *v *= i as f32 / rl as f32;
        }
    }
    if rr > 0 {
        for k in 0..rr {
            m[length - rr + k] *= 1.0 - (k + 1) as f32 / (rr + 1) as f32;
        }
    }
    m
}

/// Tiles covering `[0, n)` latent cells: `(start, extent)` pairs stepping by
/// `tile - overlap`, each extent capped at `tile`. A single `(0, n)` when
/// `n <= tile` (budget-fits-whole / parity fast path -> bit-identical).
pub fn plan_tiles(n: u32, tile: u32, overlap: u32) -> Vec<(u32, u32)> {
    if n <= tile {
        return vec![(0, n)];
    }
    let step = (tile - overlap).max(1);
    let mut tiles = Vec::new();
    let mut start = 0;
    loop {
        let ext = (n - start).min(tile);
        tiles.push((start, ext));
        if start + ext >= n {
            break;
        }
        start += step;
    }
    tiles
}

/// Per-output-pixel feather weights along one tiled axis (length `ext*scale`):
/// ramps 0->1 over the `overlap*scale` band on any edge abutting a neighbor,
/// holds 1 elsewhere. Adjacent tiles' complementary ramps sum to ~1 over the
/// shared overlap (partition of unity); a tiny floor keeps `wsum` positive.
pub fn feather_1d(ext: u32, overlap: u32, scale: u32, has_prev: bool, has_next: bool) -> Vec<f32> {
    let len = (ext * scale) as usize;
    let ramp = ((overlap * scale) as usize).min(len).max(1) as f32;
    (0..len)
        .map(|i| {
            let mut wt = 1.0f32;
            if has_prev {
                wt = wt.min((i as f32 + 0.5) / ramp);
            }
            if has_next {
                wt = wt.min(((len - i) as f32 - 0.5) / ramp);
            }
            wt.clamp(0.0, 1.0).max(1e-4)
        })
        .collect()
}

/// Gather a spatio-temporal sub-tile `[channels, tlen, hext, wext]` (contiguous
/// CTHW) from a full tensor `[channels, f, h, w]` at offset `(t0, r0, c0)`.
#[allow(clippy::too_many_arguments)]
pub fn gather_subtile(
    z: &[f32],
    channels: usize,
    f: usize,
    h: usize,
    w: usize,
    t0: usize,
    tlen: usize,
    r0: usize,
    c0: usize,
    hext: usize,
    wext: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; channels * tlen * hext * wext];
    for c in 0..channels {
        for t in 0..tlen {
            let src_ct = (c * f + (t0 + t)) * h * w;
            let dst_ct = (c * tlen + t) * hext * wext;
            for yy in 0..hext {
                let src = src_ct + (r0 + yy) * w + c0;
                let dst = dst_ct + yy * wext;
                out[dst..dst + wext].copy_from_slice(&z[src..src + wext]);
            }
        }
    }
    out
}

/// Accumulate one spatial tile's feather window into the spatial weight sum
/// `[oh*ow]` at output offset `(y0, x0)`. `wh`/`ww` hold one weight per output
/// pixel along H/W. Channel/time/temporal-tile independent, so done once.
pub fn accumulate_spatial_wsum(
    wsum_s: &mut [f32],
    ow: usize,
    y0: usize,
    x0: usize,
    wh: &[f32],
    ww: &[f32],
) {
    for (yy, &wy) in wh.iter().enumerate() {
        let row = (y0 + yy) * ow + x0;
        for (xx, &wx) in ww.iter().enumerate() {
            wsum_s[row + xx] += wy * wx;
        }
    }
}

/// Blend-accumulate a decoded pixel tile `[out_channels, tlen, th, tw]` into
/// `video` `[out_channels, f_px, oh, ow]` at output offset `(t0, y0, x0)`,
/// weighting each voxel by `tmask[t] * wh[y] * ww[x]` (the separable temporal x
/// spatial blend window). Normalization by the weight-sum product happens once
/// in the caller.
#[allow(clippy::too_many_arguments)]
pub fn blend_tile(
    video: &mut [f32],
    pix: &[f32],
    out_channels: usize,
    oh: usize,
    ow: usize,
    plane: usize,
    t0: usize,
    y0: usize,
    x0: usize,
    tlen: usize,
    th: usize,
    tw: usize,
    tmask: &[f32],
    wh: &[f32],
    ww: &[f32],
) {
    for c in 0..out_channels {
        for (t, &wt) in tmask.iter().enumerate().take(tlen) {
            let dst_ct = c * plane + (t0 + t) * oh * ow;
            let src_ct = (c * tlen + t) * th * tw;
            for (yy, &wy) in wh.iter().enumerate() {
                let dst_row = dst_ct + (y0 + yy) * ow + x0;
                let src_row = src_ct + yy * tw;
                let wyt = wt * wy;
                for (xx, &wx) in ww.iter().enumerate() {
                    video[dst_row + xx] += pix[src_row + xx] * (wyt * wx);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_tiles_single_when_fits() {
        assert_eq!(plan_tiles(8, 8, 2), vec![(0, 8)]);
        assert_eq!(plan_tiles(5, 8, 2), vec![(0, 5)]);
    }

    #[test]
    fn plan_tiles_covers_and_overlaps() {
        let t = plan_tiles(16, 6, 2); // step 4
        assert_eq!(t, [(0, 6), (4, 6), (8, 6), (12, 4)]);
        let mut covered = [false; 16];
        for (s, e) in t {
            for i in s..s + e {
                covered[i as usize] = true;
            }
        }
        assert!(covered.iter().all(|&c| c));
    }

    #[test]
    fn temporal_single_tile_is_unit() {
        // f <= tf -> one tile spanning all output frames with no ramps (scale 8).
        let t = plan_temporal_tiles(3, 8, 2, 8);
        assert_eq!(t.len(), 1);
        let tt = t[0];
        assert_eq!((tt.l0, tt.l1), (0, 3));
        assert_eq!((tt.o0, tt.o1), (0, 8 * 2 + 1)); // 8*(3-1)+1 = 17
        assert_eq!((tt.lr, tt.rr), (0, 0));
        assert!(
            trapezoid_mask(tt.o1 - tt.o0, tt.lr, tt.rr)
                .iter()
                .all(|&w| w == 1.0)
        );
    }

    #[test]
    fn temporal_tiles_cover_output_and_match_decoder_len() {
        // Exercise both temporal scales (LTX=8, Hunyuan=4).
        for s in [8usize, 4usize] {
            let (f, tf, ov) = (16usize, 6usize, 2usize);
            let tiles = plan_temporal_tiles(f, tf, ov, s);
            assert!(tiles.len() > 1);
            let f_px = s * (f - 1) + 1;
            assert_eq!(tiles.first().unwrap().o0, 0);
            assert_eq!(tiles.last().unwrap().o1, f_px);
            let mut wsum = vec![0.0f32; f_px];
            for tt in &tiles {
                // Mask length == decoder output length for the latent depth.
                let dec_len = s * (tt.l1 - tt.l0 - 1) + 1;
                assert_eq!(tt.o1 - tt.o0, dec_len, "mask vs decoder length (scale {s})");
                let m = trapezoid_mask(tt.o1 - tt.o0, tt.lr, tt.rr);
                for (i, &w) in m.iter().enumerate() {
                    wsum[tt.o0 + i] += w;
                }
            }
            assert!(wsum.iter().all(|&w| w > 1e-3), "wsum {wsum:?} (scale {s})");
        }
    }

    #[test]
    fn trapezoid_mask_ramps() {
        let m = trapezoid_mask(8, 2, 2);
        assert_eq!(m[0], 0.0);
        assert!((m[1] - 0.5).abs() < 1e-6);
        assert_eq!(m[2], 1.0);
        assert_eq!(m[5], 1.0);
        assert!((m[6] - (1.0 - 1.0 / 3.0)).abs() < 1e-6);
        assert!((m[7] - (1.0 - 2.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn feather_partition_of_unity_on_shared_overlap() {
        let overlap = 2u32;
        let scale = 32u32;
        let left = feather_1d(6, overlap, scale, false, true);
        let right = feather_1d(6, overlap, scale, true, false);
        let band = (overlap * scale) as usize;
        for k in 0..band {
            let l = left[left.len() - band + k];
            let r = right[k];
            assert!((l + r - 1.0).abs() < 0.05, "sum {} at {k}", l + r);
        }
    }
}
