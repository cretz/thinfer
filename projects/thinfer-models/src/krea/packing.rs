//! Krea latent <-> patch-token packing (host-side, B=1). Flux-style channel-
//! major patchify: `rearrange(latent, "c (gh ph) (gw pw) -> (gh gw) (c ph pw)")`,
//! i.e. `vec[c*4 + ph*2 + pw] = latent[c, gh*2+ph, gw*2+pw]`. The DiT operates on
//! `[num_tokens, PACKED_CH=64]` tokens; the VAE latent is `[LATENT_CH=16, H, W]`.
//! Token order is row-major over the patch grid `(gh, gw)`.

use super::config;

const C: usize = config::LATENT_CH; // 16
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

/// Inverse of [`pack_latents`]: patch tokens back to a spatial latent
/// `[C, H, W]`. `H`, `W` are the target spatial dims (even).
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
        assert_eq!(unpack_latents(&packed, h, w), latent);
    }

    #[test]
    fn packed_ch_invariant() {
        assert_eq!(C * P * P, config::PACKED_CH);
    }
}
