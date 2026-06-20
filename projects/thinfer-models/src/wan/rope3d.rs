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
use crate::wan::dit_block::config;

/// RoPE3D builder for one latent grid shape. Holds the per-axis cis tables;
/// `lookup` produces the `[n_tok, head_dim]` interleaved freqs the DiT block's
/// self-attention rotates q/k with (the driver packs them to the act dtype via
/// `freqs_upload_bytes` before upload).
#[derive(Clone, Debug)]
pub struct WanRope3d {
    inner: RopeEmbedder,
}

impl WanRope3d {
    /// Tables sized to `ROPE_MAX_SEQ_LEN` per axis (diffusers precomputes to
    /// `rope_max_seq_len` then slices per grid). Axis order is `(t, h, w)`.
    pub fn new() -> Self {
        let axes_dims = [config::ROPE_T_DIM, config::ROPE_H_DIM, config::ROPE_W_DIM];
        let axes_lens = [config::ROPE_MAX_SEQ_LEN; 3];
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
    /// frames (`f -> f + t_start`); spatial axes are unchanged. LongLive AR
    /// rotates each chunk's q/k at its ABSOLUTE frame position (release
    /// `use_relative_rope=False`), and the integer multi-shot RoPE offset folds
    /// straight into `t_start` (`chunk_start_frame + shot_index * 8`). The cached
    /// prefix keys were already RoPE'd at their own absolute positions, so only
    /// the current chunk uses this.
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

    /// `[n_tok, head_dim]` freqs with the temporal axis shifted by `t_start`
    /// frames (see [`Self::grid_ids_temporal`]). Used to rotate one AR chunk's
    /// q/k at its absolute frame position (plus the integer multi-shot offset).
    pub fn lookup_temporal(&self, ppf: usize, pph: usize, ppw: usize, t_start: usize) -> Vec<f32> {
        self.inner
            .lookup(&Self::grid_ids_temporal(ppf, pph, ppw, t_start))
    }
}

impl Default for WanRope3d {
    fn default() -> Self {
        Self::new()
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
    fn lookup_temporal_matches_base_offset() {
        // A chunk of `ppf` frames at absolute start `t0` must rotate identically
        // to the corresponding slice of a base lookup grown to cover `t0 + ppf`.
        let r = WanRope3d::new();
        let (ppf, pph, ppw, t0) = (2usize, 3, 4, 5usize);
        let shifted = r.lookup_temporal(ppf, pph, ppw, t0);
        let base = r.lookup(t0 + ppf, pph, ppw);
        let hd = config::HEAD_DIM;
        // shifted token (f,h,w) == base token (f+t0, h, w).
        for f in 0..ppf {
            for h in 0..pph {
                for w in 0..ppw {
                    let si = ((f * pph + h) * ppw + w) * hd;
                    let bi = (((f + t0) * pph + h) * ppw + w) * hd;
                    assert_eq!(
                        &shifted[si..si + hd],
                        &base[bi..bi + hd],
                        "f={f} h={h} w={w}"
                    );
                }
            }
        }
    }

    #[test]
    fn lookup_shape_and_origin() {
        let r = WanRope3d::new();
        let f = r.lookup(2, 3, 4);
        let n_tok = 2 * 3 * 4;
        assert_eq!(f.len(), n_tok * config::HEAD_DIM);
        // Token 0 sits at coord (0,0,0): every (re, im) pair is (1, 0).
        for j in 0..config::HEAD_DIM / 2 {
            assert!((f[2 * j] - 1.0).abs() < 1e-6);
            assert!(f[2 * j + 1].abs() < 1e-6);
        }
    }
}
