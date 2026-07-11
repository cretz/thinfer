//! Krea 3-axis complex RoPE (Flux-style joint ids). Reference:
//! `krea2.hpp::gen_krea2_pe` = `Rope::concat_ids(gen_flux_txt_ids,
//! gen_flux_img_ids)` then `embed_nd(theta, axes_dim)`.
//!
//! Interleaved-pair complex family (NOT half-rot), so the shared
//! [`RopeEmbedder`] and `op_rope` kernel consume it directly. Axes `(t, h, w)`
//! with sub-dims 32/48/48 (sum = head_dim 128), `theta = 1000`.
//!
//! Ids are applied to the CONCATENATED `[txt ++ img]` sequence: text tokens sit
//! at `(0, 0, 0)` (origin on every axis, identity rotation); image tokens at
//! `(0, r, c)` over the patch grid, row-major and UNCENTERED (`gen_flux_img_ids`
//! uses raw `0..gh`, `0..gw` indices; no `scale_rope`).

use crate::common::rope_embedder::RopeEmbedder;

use super::config;

/// Max absolute RoPE position; covers the largest patch grid we generate
/// (e.g. 2048px / 16 = 128 tokens/side, well under this).
const ROPE_MAX_POS: usize = 4096;

/// Krea RoPE freq generator. Holds the shared per-axis cis tables; the per-call
/// work is id construction + a table lookup.
#[derive(Clone, Debug)]
pub struct KreaRope {
    inner: RopeEmbedder,
}

impl KreaRope {
    pub fn new() -> Self {
        Self {
            inner: RopeEmbedder::new(
                config::ROPE_THETA,
                config::AXES_DIMS_ROPE,
                [ROPE_MAX_POS; 3],
            ),
        }
    }

    /// `head_dim` = sum of the axis sub-dims (= 128).
    pub fn head_dim(&self) -> usize {
        self.inner.head_dim()
    }

    /// Freqs for the full `[txt ++ img]` sequence,
    /// `[(txt_len + gh*gw), head_dim]` interleaved `(re, im)`. Text tokens sit at
    /// the origin; image tokens at `(0, r, c)` over the `gh x gw` patch grid,
    /// row-major.
    pub fn freqs(&self, txt_len: usize, gh: usize, gw: usize) -> Vec<f32> {
        let n = txt_len + gh * gw;
        let mut ids = Vec::with_capacity(n * 3);
        for _ in 0..txt_len {
            ids.extend_from_slice(&[0, 0, 0]);
        }
        for r in 0..gh {
            for c in 0..gw {
                ids.extend_from_slice(&[0, r as i32, c as i32]);
            }
        }
        self.inner.lookup_signed(&ids)
    }
}

impl Default for KreaRope {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_dim_is_128() {
        assert_eq!(KreaRope::new().head_dim(), config::HEAD_DIM);
        assert_eq!(
            config::AXES_DIMS_ROPE.iter().sum::<usize>(),
            config::HEAD_DIM
        );
    }

    #[test]
    fn text_tokens_are_identity() {
        let rope = KreaRope::new();
        let (txt_len, gh, gw) = (3, 4, 6);
        let f = rope.freqs(txt_len, gh, gw);
        assert_eq!(f.len(), (txt_len + gh * gw) * config::HEAD_DIM);
        // Every text token sits at (0,0,0) => identity: interleaved (1,0) pairs.
        for (j, &fj) in f.iter().take(txt_len * config::HEAD_DIM).enumerate() {
            let want = if j % 2 == 0 { 1.0 } else { 0.0 };
            assert!((fj - want).abs() < 1e-6, "lane {j} = {fj}");
        }
    }

    #[test]
    fn image_first_token_is_origin_identity() {
        // First image token is grid (0,0) => (0,0,0) => also identity.
        let rope = KreaRope::new();
        let (txt_len, gh, gw) = (2, 3, 3);
        let f = rope.freqs(txt_len, gh, gw);
        let off = txt_len * config::HEAD_DIM;
        for j in 0..config::HEAD_DIM {
            let want = if j % 2 == 0 { 1.0 } else { 0.0 };
            assert!((f[off + j] - want).abs() < 1e-6, "img lane {j}");
        }
    }
}
