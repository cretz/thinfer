//! CPU-side patchify, pos-id grids, pad lengths, unpatchify (Z-Image layout).
//!
//! Mirrors `ZImageTransformer2DModel.patchify_and_embed` / `unpatchify`
//! (`src/zimage/transformer.py`). v1 is single-batch (one image + one
//! caption per forward); for `bsz>1` the per-item pad lengths force a
//! different concat scheme. Activation-dtype encode/decode and attention-mask
//! byte builders are model-agnostic and live in `crate::common::seq`.
//!
//! Conventions
//!
//! - Image input: `[C, F, H, W]` row-major. Patch sizes `pH=pW=patch_size`,
//!   `pF=f_patch_size`. Token order: `(F/pF, H/pH, W/pW) -> token` with the
//!   inner `(pF, pH, pW, C)` collapsed.
//! - Pad length per stream = `(-ori_len) % SEQ_MULTI_OF`.
//! - Position-id grid is `[seq, 3]` i32, axis order `(temporal, h, w)`. Caption
//!   positions live on the temporal axis only (`start=(1, 0, 0)`), images
//!   start at `(cap_padded_len + 1, 0, 0)`. Padded-out positions are
//!   `(0, 0, 0)` (matches upstream).

use crate::z_image::config::SEQ_MULTI_OF;

#[derive(Clone, Debug)]
pub struct PatchifyOut {
    /// `[n_padded, pF*pH*pW*C]` row-major f32 image tokens.
    pub tokens: Vec<f32>,
    /// Original `(F, H, W)` for unpatchify.
    pub size: (usize, usize, usize),
    /// Per-token pos ids `[n_padded, 3]` i32.
    pub pos_ids: Vec<i32>,
    /// `[n_padded]` 0/1; 1 = pad position (gets x_pad_token substituted).
    pub pad_mask: Vec<u8>,
    pub ori_len: usize,
    pub padded_len: usize,
}

/// Single-image patchify. Image is `[C, F, H, W]` row-major f32.
pub fn patchify(
    image: &[f32],
    c: usize,
    f: usize,
    h: usize,
    w: usize,
    patch_size: usize,
    f_patch_size: usize,
    cap_padded_len: usize,
) -> PatchifyOut {
    let p_h = patch_size;
    let p_w = patch_size;
    let p_f = f_patch_size;
    assert!(f.is_multiple_of(p_f) && h.is_multiple_of(p_h) && w.is_multiple_of(p_w));

    let f_tok = f / p_f;
    let h_tok = h / p_h;
    let w_tok = w / p_w;
    let n_tok = f_tok * h_tok * w_tok;
    let patch = p_f * p_h * p_w * c;

    // Reshape [C, F, H, W] -> [f_tok, h_tok, w_tok, p_f, p_h, p_w, C]. Upstream
    // permutes (1,3,5,2,4,6,0) of [C, f_tok, p_f, h_tok, p_h, w_tok, p_w], i.e.
    // it puts token-grid first, then within-patch (p_f, p_h, p_w), then C.
    let mut tokens_ori = vec![0.0_f32; n_tok * patch];
    let stride_c = f * h * w;
    let stride_f = h * w;
    let stride_h = w;
    for ft in 0..f_tok {
        for ht in 0..h_tok {
            for wt in 0..w_tok {
                let tok = (ft * h_tok + ht) * w_tok + wt;
                let dst_base = tok * patch;
                for pf in 0..p_f {
                    for ph in 0..p_h {
                        for pw in 0..p_w {
                            let f_idx = ft * p_f + pf;
                            let h_idx = ht * p_h + ph;
                            let w_idx = wt * p_w + pw;
                            for ci in 0..c {
                                let src =
                                    ci * stride_c + f_idx * stride_f + h_idx * stride_h + w_idx;
                                let inner = ((pf * p_h + ph) * p_w + pw) * c + ci;
                                tokens_ori[dst_base + inner] = image[src];
                            }
                        }
                    }
                }
            }
        }
    }

    let pad_len = ori_pad_len(n_tok);
    let padded_len = n_tok + pad_len;
    let mut tokens = vec![0.0_f32; padded_len * patch];
    tokens[..n_tok * patch].copy_from_slice(&tokens_ori);
    // Upstream pads by repeating the last token; mask flag matters more than the
    // payload (those rows get replaced by `x_pad_token` after the embedder).
    if pad_len > 0 && n_tok > 0 {
        let last = (n_tok - 1) * patch;
        for i in 0..pad_len {
            let dst = (n_tok + i) * patch;
            tokens[dst..dst + patch].copy_from_slice(&tokens_ori[last..last + patch]);
        }
    }

    let mut pos_ids = vec![0_i32; padded_len * 3];
    let base = (cap_padded_len + 1) as i32;
    for ft in 0..f_tok {
        for ht in 0..h_tok {
            for wt in 0..w_tok {
                let tok = (ft * h_tok + ht) * w_tok + wt;
                pos_ids[tok * 3] = base + ft as i32;
                pos_ids[tok * 3 + 1] = ht as i32;
                pos_ids[tok * 3 + 2] = wt as i32;
            }
        }
    }
    // Padded positions all `(0,0,0)`, already zeroed.

    let mut pad_mask = vec![0_u8; padded_len];
    for i in 0..pad_len {
        pad_mask[n_tok + i] = 1;
    }

    PatchifyOut {
        tokens,
        size: (f, h, w),
        pos_ids,
        pad_mask,
        ori_len: n_tok,
        padded_len,
    }
}

/// Single-caption pos-id + pad-mask build. Cap features themselves stay where
/// the caller put them; this only emits the metadata.
#[derive(Clone, Debug)]
pub struct CapMeta {
    pub pos_ids: Vec<i32>,
    pub pad_mask: Vec<u8>,
    pub ori_len: usize,
    pub padded_len: usize,
}

pub fn cap_meta(cap_ori_len: usize) -> CapMeta {
    let pad_len = ori_pad_len(cap_ori_len);
    let padded_len = cap_ori_len + pad_len;
    let mut pos_ids = vec![0_i32; padded_len * 3];
    for i in 0..padded_len {
        pos_ids[i * 3] = (i + 1) as i32; // temporal axis, start=1
    }
    let mut pad_mask = vec![0_u8; padded_len];
    for i in 0..pad_len {
        pad_mask[cap_ori_len + i] = 1;
    }
    CapMeta {
        pos_ids,
        pad_mask,
        ori_len: cap_ori_len,
        padded_len,
    }
}

/// Repeat the final caption row into the padded tail (matches upstream cap
/// padding).
pub fn pad_cap_feats(
    cap: &[f32],
    cap_feat_dim: usize,
    ori_len: usize,
    padded_len: usize,
) -> Vec<f32> {
    assert_eq!(cap.len(), ori_len * cap_feat_dim);
    let mut out = vec![0.0_f32; padded_len * cap_feat_dim];
    out[..cap.len()].copy_from_slice(cap);
    if padded_len > ori_len && ori_len > 0 {
        let last = (ori_len - 1) * cap_feat_dim;
        for i in 0..(padded_len - ori_len) {
            let dst = (ori_len + i) * cap_feat_dim;
            out[dst..dst + cap_feat_dim].copy_from_slice(&cap[last..last + cap_feat_dim]);
        }
    }
    out
}

pub fn pad_len(ori_len: usize) -> usize {
    ori_pad_len(ori_len)
}

fn ori_pad_len(ori_len: usize) -> usize {
    let m = SEQ_MULTI_OF;
    (m - (ori_len % m)) % m
}

/// Inverse of `patchify` for a single image. `tokens` is `[ori_len, patch]`
/// (caller slices off the pad rows first).
pub fn unpatchify(
    tokens: &[f32],
    out_channels: usize,
    f: usize,
    h: usize,
    w: usize,
    patch_size: usize,
    f_patch_size: usize,
) -> Vec<f32> {
    let p_h = patch_size;
    let p_w = patch_size;
    let p_f = f_patch_size;
    let f_tok = f / p_f;
    let h_tok = h / p_h;
    let w_tok = w / p_w;
    let patch = p_f * p_h * p_w * out_channels;
    assert_eq!(tokens.len(), f_tok * h_tok * w_tok * patch);
    let mut img = vec![0.0_f32; out_channels * f * h * w];
    let stride_c = f * h * w;
    let stride_f = h * w;
    let stride_h = w;
    for ft in 0..f_tok {
        for ht in 0..h_tok {
            for wt in 0..w_tok {
                let tok = (ft * h_tok + ht) * w_tok + wt;
                let src_base = tok * patch;
                for pf in 0..p_f {
                    for ph in 0..p_h {
                        for pw in 0..p_w {
                            for ci in 0..out_channels {
                                let inner = ((pf * p_h + ph) * p_w + pw) * out_channels + ci;
                                let f_idx = ft * p_f + pf;
                                let h_idx = ht * p_h + ph;
                                let w_idx = wt * p_w + pw;
                                let dst =
                                    ci * stride_c + f_idx * stride_f + h_idx * stride_h + w_idx;
                                img[dst] = tokens[src_base + inner];
                            }
                        }
                    }
                }
            }
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patchify_unpatchify_roundtrip_3d() {
        // 2 channels, F=2, H=4, W=4, patch=2, f_patch=1 -> 2*2*2=8 tokens of 8 floats.
        let (c, f, h, w) = (2, 2, 4, 4);
        let n = c * f * h * w;
        let image: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let px = patchify(&image, c, f, h, w, 2, 1, 0);
        assert_eq!(px.ori_len, 8);
        assert_eq!(px.padded_len, 32); // SEQ_MULTI_OF=32 rounds up from 8
        let tokens_ori: Vec<f32> = px.tokens[..px.ori_len * 8].to_vec();
        let back = unpatchify(&tokens_ori, c, f, h, w, 2, 1);
        assert_eq!(back, image, "patchify->unpatchify must be identity");
    }

    #[test]
    fn pad_len_arithmetic() {
        assert_eq!(pad_len(0), 0);
        assert_eq!(pad_len(32), 0);
        assert_eq!(pad_len(1), 31);
        assert_eq!(pad_len(31), 1);
        assert_eq!(pad_len(33), 31);
    }

    #[test]
    fn cap_meta_layout() {
        let cm = cap_meta(10);
        assert_eq!(cm.ori_len, 10);
        assert_eq!(cm.padded_len, 32);
        // temporal axis starts at 1; h, w axes zero.
        assert_eq!(cm.pos_ids[0..3], [1, 0, 0]);
        assert_eq!(cm.pos_ids[9 * 3..9 * 3 + 3], [10, 0, 0]);
        // pad rows: mask=1, pos still indexed (matches upstream).
        assert_eq!(cm.pad_mask[10], 1);
        assert_eq!(cm.pad_mask[31], 1);
        assert_eq!(cm.pad_mask[9], 0);
    }

    #[test]
    fn image_pos_ids_start_after_cap() {
        let (c, f, h, w) = (1, 1, 4, 4);
        let img = vec![0.0_f32; c * f * h * w];
        let cap_padded = 32;
        let px = patchify(&img, c, f, h, w, 2, 1, cap_padded);
        // First image token: temporal=cap_padded+1, h=0, w=0.
        assert_eq!(px.pos_ids[0..3], [33, 0, 0]);
    }
}
