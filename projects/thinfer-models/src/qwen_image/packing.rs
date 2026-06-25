//! Qwen-Image latent <-> patch-token packing (host-side, B=1).
//!
//! Ground truth: `pipeline_qwenimage_edit.py::{_pack_latents,_unpack_latents}`.
//! The DiT operates on `[num_tokens, IN_CHANNELS=64]` patch tokens; the VAE
//! latent is `[z_dim=16, H, W]`. A 2x2 patch of the latent becomes one token's
//! 64-vec, ordered CHANNEL-major then patch-row then patch-col:
//!   `vec[c*4 + ph*2 + pw] = latent[c, gh*2+ph, gw*2+pw]`.
//! (NB this differs from ideogram4's patch-major `[ph,pw,c]` packing.)
//!
//! `_pack_latents`: `view(C, H/2, 2, W/2, 2) -> permute(H/2, W/2, C, 2, 2) ->
//! reshape((H/2)*(W/2), C*4)`. `_unpack_latents` is the exact inverse. Token
//! order is row-major over the patch grid `(gh, gw)`.

use super::config;

/// `z_dim` latent channels per patch (= [`config::OUT_CHANNELS`]).
const C: usize = config::OUT_CHANNELS; // 16
/// Patch side (2x2). `C * P * P = IN_CHANNELS` (64).
const P: usize = config::PATCH_SIZE; // 2

/// Pack a spatial latent `[C, H, W]` (row-major CHW) into patch tokens
/// `[(H/2)*(W/2), C*4]`. `H` and `W` must be even.
pub fn pack_latents(latent: &[f32], h: usize, w: usize) -> Vec<f32> {
    assert_eq!(h % P, 0, "height {h} not divisible by patch {P}");
    assert_eq!(w % P, 0, "width {w} not divisible by patch {P}");
    assert_eq!(latent.len(), C * h * w, "latent len != C*H*W");
    let (gh, gw) = (h / P, w / P);
    let plane = h * w;
    let tok_dim = C * P * P; // 64
    let mut out = vec![0.0_f32; gh * gw * tok_dim];
    for ghi in 0..gh {
        for gwi in 0..gw {
            let tok = (ghi * gw + gwi) * tok_dim;
            for c in 0..C {
                for ph in 0..P {
                    for pw in 0..P {
                        let y = ghi * P + ph;
                        let x = gwi * P + pw;
                        out[tok + c * P * P + ph * P + pw] = latent[c * plane + y * w + x];
                    }
                }
            }
        }
    }
    out
}

/// Inverse of [`pack_latents`]: patch tokens `[(H/2)*(W/2), C*4]` back to a
/// spatial latent `[C, H, W]`. `H`, `W` are the target spatial dims (even).
pub fn unpack_latents(tokens: &[f32], h: usize, w: usize) -> Vec<f32> {
    assert_eq!(h % P, 0, "height {h} not divisible by patch {P}");
    assert_eq!(w % P, 0, "width {w} not divisible by patch {P}");
    let (gh, gw) = (h / P, w / P);
    let tok_dim = C * P * P;
    assert_eq!(tokens.len(), gh * gw * tok_dim, "tokens len != grid*64");
    let plane = h * w;
    let mut out = vec![0.0_f32; C * plane];
    for ghi in 0..gh {
        for gwi in 0..gw {
            let tok = (ghi * gw + gwi) * tok_dim;
            for c in 0..C {
                for ph in 0..P {
                    for pw in 0..P {
                        let y = ghi * P + ph;
                        let x = gwi * P + pw;
                        out[c * plane + y * w + x] = tokens[tok + c * P * P + ph * P + pw];
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_round_trips() {
        let (h, w) = (4, 6);
        let latent: Vec<f32> = (0..C * h * w).map(|i| i as f32 * 0.5 - 3.0).collect();
        let packed = pack_latents(&latent, h, w);
        assert_eq!(packed.len(), (h / P) * (w / P) * (C * P * P));
        let back = unpack_latents(&packed, h, w);
        assert_eq!(back, latent);
    }

    #[test]
    fn pack_layout_is_channel_major() {
        // A single 2x2 grid (h=w=2 => one token). vec[c*4 + ph*2 + pw].
        let (h, w) = (2, 2);
        let latent: Vec<f32> = (0..C * h * w).map(|i| i as f32).collect();
        let packed = pack_latents(&latent, h, w);
        // Token 0, channel c, patch (ph,pw): latent index = c*4 + ph*2 + pw
        // (since plane=4, y=ph, x=pw). So packed[c*4+ph*2+pw] == that value.
        for c in 0..C {
            for ph in 0..P {
                for pw in 0..P {
                    let want = (c * (h * w) + ph * w + pw) as f32;
                    assert_eq!(packed[c * 4 + ph * 2 + pw], want);
                }
            }
        }
    }

    #[test]
    fn in_channels_invariant() {
        assert_eq!(C * P * P, config::IN_CHANNELS);
    }
}
