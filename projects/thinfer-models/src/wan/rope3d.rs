//! Wan / SkyReels-V2 RoPE3D frequency table over the latent patch grid.
//!
//! Mirrors `SkyReelsV2RotaryPosEmbed` (`transformer_skyreels_v2.py`). The head
//! dimension splits into three axes `(t, h, w)` with sub-dims `t=44, h=w=42` at
//! `head_dim=128`; each axis gets a 1D rotary table (`get_1d_rotary_pos_embed`,
//! `repeat_interleave_real=True`), and per token the three axes' rows are
//! concatenated to a `[head_dim]` `(re, im)`-interleaved vector.
//!
//! `repeat_interleave_real` produces exactly the `(cos_j, sin_j)` per-pair
//! layout the interleaved-pair `RopeF32` kernel consumes (the same layout the
//! Z-Image [`RopeEmbedder`] builds), so we reuse it: one table per axis with
//! `axes_dims = [t, h, w]`, then look up `(f, h, w)` coordinates per token.
//!
//! Token order matches the DiT's patchify (`flatten(2).transpose(1, 2)` of the
//! conv output): `(f, h, w)` row-major over `(ppf, pph, ppw)`.

use crate::common::rope_embedder::RopeEmbedder;
use crate::wan::dit_block::{WanGeometry, config};

/// Longest audio latent the 1D table covers (frames; MOSS runs 1500 = 30 s).
pub const AUDIO_ROPE_LEN: usize = 4096;

/// RoPE3D builder for one latent grid shape. Holds the per-axis cis tables;
/// `lookup` produces the `[n_tok, head_dim]` interleaved freqs the DiT block's
/// self-attention rotates q/k with (the driver packs them to the act dtype via
/// `freqs_upload_bytes` before upload).
#[derive(Clone, Debug)]
pub struct WanRope3d {
    inner: RopeEmbedder,
}

impl WanRope3d {
    /// Video: tables sized to `ROPE_MAX_SEQ_LEN` per axis (diffusers
    /// precomputes to `rope_max_seq_len` then slices per grid), axis order
    /// `(t, h, w)`. Audio: one temporal axis over the whole head dim (the
    /// `h`/`w` axes are empty, so a `[F, 1, 1]` grid's ids `(f, 0, 0)` rotate
    /// every pair by `f`).
    pub fn new(geometry: WanGeometry) -> Self {
        let (axes_dims, axes_lens) = match geometry {
            WanGeometry::Video => (
                [config::ROPE_T_DIM, config::ROPE_H_DIM, config::ROPE_W_DIM],
                [config::ROPE_MAX_SEQ_LEN; 3],
            ),
            WanGeometry::Audio => ([config::HEAD_DIM, 0, 0], [AUDIO_ROPE_LEN, 1, 1]),
        };
        Self {
            inner: RopeEmbedder::new(config::ROPE_THETA, axes_dims, axes_lens),
        }
    }

    /// Per-token `(f, h, w)` coordinate ids `[n_tok * 3]` for the patch grid,
    /// in the patchify token order `(ppf, pph, ppw)` row-major.
    pub fn grid_ids(ppf: usize, pph: usize, ppw: usize) -> Vec<i32> {
        Self::grid_ids_temporal(ppf, pph, ppw, 0)
    }

    /// Like [`Self::grid_ids`] but with the temporal axis shifted by `t_start`
    /// frames (`f -> f + t_start`); spatial axes are unchanged.
    pub fn grid_ids_temporal(ppf: usize, pph: usize, ppw: usize, t_start: usize) -> Vec<i32> {
        let mut ids = Vec::with_capacity(ppf * pph * ppw * 3);
        for f in 0..ppf {
            for h in 0..pph {
                for w in 0..ppw {
                    ids.push((f + t_start) as i32);
                    ids.push(h as i32);
                    ids.push(w as i32);
                }
            }
        }
        ids
    }

    /// `[n_tok, head_dim]` interleaved `(re, im)` f32 freqs (host floats). The
    /// driver packs these to the act dtype (`freqs_upload_bytes`) before upload;
    /// the f16/bf16 rope kernels read freqs in the act dtype, not f32.
    pub fn lookup(&self, ppf: usize, pph: usize, ppw: usize) -> Vec<f32> {
        self.inner.lookup(&Self::grid_ids(ppf, pph, ppw))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_dims_sum_to_head_dim() {
        assert_eq!(
            config::ROPE_T_DIM + config::ROPE_H_DIM + config::ROPE_W_DIM,
            config::HEAD_DIM
        );
        // All axis dims must be even for the (re, im) pairing.
        for d in [config::ROPE_T_DIM, config::ROPE_H_DIM, config::ROPE_W_DIM] {
            assert!(d.is_multiple_of(2));
        }
    }

    #[test]
    fn grid_ids_token_order() {
        // 2x1x2 grid -> tokens (f,h,w): (0,0,0),(0,0,1),(1,0,0),(1,0,1).
        let ids = WanRope3d::grid_ids(2, 1, 2);
        assert_eq!(ids, vec![0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1]);
    }

    #[test]
    fn grid_ids_temporal_shifts_only_time() {
        // Same 2x1x2 grid at t_start=3: temporal ids become 3,3,4,4; h/w intact.
        let ids = WanRope3d::grid_ids_temporal(2, 1, 2, 3);
        assert_eq!(ids, vec![3, 0, 0, 3, 0, 1, 4, 0, 0, 4, 0, 1]);
    }

    #[test]
    fn lookup_shape_and_origin() {
        let r = WanRope3d::new(WanGeometry::Video);
        let f = r.lookup(2, 3, 4);
        let n_tok = 2 * 3 * 4;
        assert_eq!(f.len(), n_tok * config::HEAD_DIM);
        // Token 0 sits at coord (0,0,0): every (re, im) pair is (1, 0).
        for j in 0..config::HEAD_DIM / 2 {
            assert!((f[2 * j] - 1.0).abs() < 1e-6);
            assert!(f[2 * j + 1].abs() < 1e-6);
        }
    }

    /// MOSS `precompute_freqs_cis_1d(128)`: pair j of frame f rotates by
    /// `f / 10000^(2j / 128)` across the whole head dim.
    #[test]
    fn audio_rotates_every_pair_by_frame() {
        let r = WanRope3d::new(WanGeometry::Audio);
        let f = r.lookup(1500, 1, 1);
        assert_eq!(f.len(), 1500 * config::HEAD_DIM);
        let row = &f[1499 * config::HEAD_DIM..][..config::HEAD_DIM];
        for j in 0..config::HEAD_DIM / 2 {
            let a = 1499.0 / 10_000f64.powf(2.0 * j as f64 / config::HEAD_DIM as f64);
            assert!((row[2 * j] as f64 - a.cos()).abs() < 1e-5, "pair {j}");
            assert!((row[2 * j + 1] as f64 - a.sin()).abs() < 1e-5, "pair {j}");
        }
    }
}
