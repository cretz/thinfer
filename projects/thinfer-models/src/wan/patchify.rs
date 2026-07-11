//! CPU patchify / unpatchify for the Wan DiT (`B = 1`).
//!
//! `patch_embedding` is `nn.Conv3d(in_ch, inner, kernel=stride=(p_t,p_h,p_w))`.
//! Because kernel == stride == patch, the conv is an affine patchify: each
//! output token is a single linear over one non-overlapping voxel patch. So we
//! fold the conv into the DiT front-door linear -- patchify emits token vectors
//! `[n_tok, in_ch*p_t*p_h*p_w]`, and the loader reshapes the conv weight
//! `[inner, in_ch, p_t, p_h, p_w]` to `[inner, in_ch*p_t*p_h*p_w]` (row-major,
//! already `(ic, kt, kh, kw)` order) so the existing matmul + bias reproduces
//! the conv exactly. No Conv3d GPU op needed.
//!
//! Token order mirrors the conv output `flatten(2).transpose(1, 2)`:
//! `(f, h, w)` row-major over the patch grid `(ppf, pph, ppw)`. The patch
//! vector is `(ic, kt, kh, kw)` row-major, matching the reshaped conv weight.
//!
//! `unpatchify` inverts the DiT's output reshape (`proj_out` produces
//! `[n_tok, out_ch*p_t*p_h*p_w]`; diffusers reshapes to
//! `[ppf, pph, ppw, p_t, p_h, p_w, out_ch]`, permutes to `[out_ch, ppf, p_t,
//! pph, p_h, ppw, p_w]`, and flattens back to `[out_ch, F, H, W]`).

use crate::wan::dit_block::config;

/// Latent voxel grid `[C, F, H, W]` and its patch grid `(ppf, pph, ppw)`.
#[derive(Clone, Copy, Debug)]
pub struct PatchGrid {
    pub c: usize,
    pub f: usize,
    pub h: usize,
    pub w: usize,
    pub ppf: usize,
    pub pph: usize,
    pub ppw: usize,
}

impl PatchGrid {
    /// Build from latent dims, validating divisibility by the patch.
    pub fn new(c: usize, f: usize, h: usize, w: usize) -> Self {
        let (pt, ph, pw) = (config::PATCH_T, config::PATCH_H, config::PATCH_W);
        assert!(
            f.is_multiple_of(pt) && h.is_multiple_of(ph) && w.is_multiple_of(pw),
            "latent dims ({f},{h},{w}) not divisible by patch ({pt},{ph},{pw})"
        );
        Self {
            c,
            f,
            h,
            w,
            ppf: f / pt,
            pph: h / ph,
            ppw: w / pw,
        }
    }

    /// Number of patch tokens `ppf * pph * ppw`.
    pub fn n_tok(&self) -> usize {
        self.ppf * self.pph * self.ppw
    }

    /// Patch-vector width `C * p_t * p_h * p_w` (the front-door linear's K).
    pub fn patch_in(&self) -> usize {
        self.c * config::PATCH_T * config::PATCH_H * config::PATCH_W
    }
}

/// Patchify `[C, F, H, W]` row-major into tokens `[n_tok, patch_in]`, token
/// order `(f, h, w)` row-major, patch-vector order `(ic, kt, kh, kw)`.
pub fn patchify(image: &[f32], grid: &PatchGrid) -> Vec<f32> {
    let (pt, ph, pw) = (config::PATCH_T, config::PATCH_H, config::PATCH_W);
    let PatchGrid {
        c,
        f,
        h,
        w,
        ppf,
        pph,
        ppw,
    } = *grid;
    debug_assert_eq!(image.len(), c * f * h * w);
    let patch_in = grid.patch_in();
    let mut out = vec![0.0_f32; grid.n_tok() * patch_in];
    let idx = |ic: usize, ff: usize, hh: usize, ww: usize| ((ic * f + ff) * h + hh) * w + ww;
    let mut t = 0usize;
    for tf in 0..ppf {
        for th in 0..pph {
            for tw in 0..ppw {
                let base = t * patch_in;
                let mut o = 0usize;
                for ic in 0..c {
                    for kt in 0..pt {
                        for kh in 0..ph {
                            for kw in 0..pw {
                                out[base + o] =
                                    image[idx(ic, tf * pt + kt, th * ph + kh, tw * pw + kw)];
                                o += 1;
                            }
                        }
                    }
                }
                t += 1;
            }
        }
    }
    out
}

/// Unpatchify `proj_out` tokens `[n_tok, out_ch*p_t*p_h*p_w]` (token order
/// `(f, h, w)`, vec order `(p_t, p_h, p_w, out_ch)`) back to `[out_ch, F, H, W]`
/// row-major.
pub fn unpatchify(tokens: &[f32], grid: &PatchGrid, out_ch: usize) -> Vec<f32> {
    let (pt, ph, pw) = (config::PATCH_T, config::PATCH_H, config::PATCH_W);
    let PatchGrid {
        f,
        h,
        w,
        ppf,
        pph,
        ppw,
        ..
    } = *grid;
    let vec_w = out_ch * pt * ph * pw;
    debug_assert_eq!(tokens.len(), grid.n_tok() * vec_w);
    let mut out = vec![0.0_f32; out_ch * f * h * w];
    let oidx = |oc: usize, ff: usize, hh: usize, ww: usize| ((oc * f + ff) * h + hh) * w + ww;
    let mut t = 0usize;
    for tf in 0..ppf {
        for th in 0..pph {
            for tw in 0..ppw {
                let base = t * vec_w;
                for pt_i in 0..pt {
                    for ph_i in 0..ph {
                        for pw_i in 0..pw {
                            let voxel = ((pt_i * ph + ph_i) * pw + pw_i) * out_ch;
                            for oc in 0..out_ch {
                                out[oidx(oc, tf * pt + pt_i, th * ph + ph_i, tw * pw + pw_i)] =
                                    tokens[base + voxel + oc];
                            }
                        }
                    }
                }
                t += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_grid_shapes() {
        // patch (1,2,2): latent 16x2x4x6 -> grid 2x2x3, n_tok=12.
        let g = PatchGrid::new(16, 2, 4, 6);
        assert_eq!((g.ppf, g.pph, g.ppw), (2, 2, 3));
        assert_eq!(g.n_tok(), 12);
        assert_eq!(
            g.patch_in(),
            16 * config::PATCH_T * config::PATCH_H * config::PATCH_W
        );
    }

    #[test]
    fn patchify_gathers_patch_voxels() {
        // C=1, F=1, H=2, W=2, patch (1,2,2) -> 1 token, vec=(kh,kw) row-major.
        let img = vec![10.0, 11.0, 12.0, 13.0]; // [h,w] row-major
        let g = PatchGrid::new(1, 1, 2, 2);
        let tok = patchify(&img, &g);
        // patch vec order (ic, kt, kh, kw): (0,0,0,0),(0,0,0,1),(0,0,1,0),(0,0,1,1)
        assert_eq!(tok, vec![10.0, 11.0, 12.0, 13.0]);
    }

    #[test]
    fn patchify_two_channels_interleave() {
        // C=2, single 2x2 spatial patch. Vec is channel-major: all of c0's
        // 4 voxels then c1's 4 voxels.
        let img = vec![
            0.0, 1.0, 2.0, 3.0, // c0
            100.0, 101.0, 102.0, 103.0, // c1
        ];
        let g = PatchGrid::new(2, 1, 2, 2);
        let tok = patchify(&img, &g);
        assert_eq!(tok, vec![0.0, 1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 103.0]);
    }

    #[test]
    fn unpatchify_scatters_voxels() {
        // out_ch=1, patch (1,2,2), grid 1x1x1 -> token vec (pt,ph,pw,oc).
        let g = PatchGrid::new(1, 1, 2, 2);
        let tokens = vec![20.0, 21.0, 22.0, 23.0]; // (ph,pw) order, oc=1
        let img = unpatchify(&tokens, &g, 1);
        assert_eq!(img, vec![20.0, 21.0, 22.0, 23.0]);
    }

    #[test]
    fn unpatchify_two_out_channels() {
        // out_ch=2, patch (1,2,2), single token (latent 1x2x2). Vec order
        // (p_h, p_w, oc): voxels (0,0),(0,1),(1,0),(1,1), each with oc0,oc1.
        let g = PatchGrid::new(2, 1, 2, 2);
        let tokens = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let img = unpatchify(&tokens, &g, 2);
        // out [oc, h, w] (f=1): oc0 -> [1,3,5,7]; oc1 -> [2,4,6,8].
        assert_eq!(img, vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]);
    }
}
