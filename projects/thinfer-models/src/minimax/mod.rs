//! MiniMax-H3: 33B dense single-stream joint audio-video diffusion transformer.
//!
//! One packed 1-D sequence carries text, conditioning, audio and video rows at
//! once and a single forward per step predicts the velocity of every row. There
//! is NO cross-attention and no per-modality block weights; modality shows up
//! only in the two input patch projections, the per-row AdaLN tag, and the two
//! output heads.
//!
//! Ground truth: `third-party/minimax-h3-mlx` (clean-room port validated against
//! the diffusers reference) plus its vendored `reference/diffusers/`.
//!
//! Two structural facts drive the whole port:
//!
//! - **AdaLN is precomputable.** `blocks.{i}.adaln_proj.linear` is `[96768,
//!   2688]` x 50 = ~13B of the 33B, and its only input is the timestep
//!   embedding. For a fixed schedule the entire modulation table is built once
//!   at load (~77 MB) and those weights are then never streamed. See [`adaln`].
//! - **The conditioner is truncated.** H3 reads the UNNORMALIZED
//!   `hidden_states[50]` of Qwen3-VL-32B's 64 decoder layers, so layers 50..63,
//!   the final norm and `lm_head` are never evaluated. See [`text_encoder`].

pub mod adaln;
pub mod audio_vae;
pub mod dit;
pub mod fasth3;
pub mod loader;
pub mod lora;
pub mod manifest;
pub mod packing;
pub mod pipeline;
pub mod rope3d;
pub mod scheduler;
pub mod text_encoder;
pub mod video_vae;
pub mod vsa;

/// DiT architecture constants, from the upstream checkpoint `config.json`
/// (`MiniMaxH3DiTModel`). Field names follow the ORIGINAL checkpoint, not the
/// diffusers port, so a config file reads verbatim.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DitConfig {
    pub hidden: usize,
    pub n_layers: usize,
    pub token_refiner_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub ffn_hidden: usize,
    /// Video latent channels.
    pub latents_dim: usize,
    /// Audio latent channels.
    pub audio_latents_dim: usize,
    /// `(t, h, w)` patch. Temporal patch is 1: latent frames are not merged.
    pub patch: (usize, usize, usize),
    /// Width of the conditioner's hidden state (Qwen3-VL-32B).
    pub text_dim: usize,
    /// Sinusoidal timestep embedding width.
    pub timestep_input_dim: usize,
    pub time_embed_hidden: usize,
    pub time_embed_dim: usize,
    /// `6 * MODALITY_NUM * hidden` = the per-block AdaLN projection width.
    pub adaln_out: usize,
    /// `2 * hidden` = the final layer's `shift`/`scale`.
    pub final_adaln_out: usize,
    /// Rank-reduced AdaLN (FastH3 Trim): every projection reads the shared
    /// `adaln_basis(silu(temb))` (`rank` wide, no further silu) instead of
    /// `silu(temb)`. `None` = the released full-width projections.
    pub adaln_rank: Option<usize>,
    /// Video sparse attention: the fraction of video key tiles each video
    /// query tile drops (FastH3's `vsa_sparsity`, trained with 64-row tiles
    /// and a learned compression gate). `None` = dense attention.
    pub vsa_sparsity: Option<f64>,
    /// Blocks of the released 50-block H3 that a trimmed checkpoint removed
    /// (the rest are renumbered densely). Maps base-H3 adapters onto it.
    pub dropped_blocks: &'static [usize],
    /// Frequencies per rotary axis. Three axes share one `inv_freq` buffer.
    pub rope_inv_freq_len: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub qk_norm_eps: f32,
    pub final_norm_eps: f32,
}

impl DitConfig {
    /// The released FL2VA / Ref2VA checkpoints. Both share this topology; they
    /// differ only in weights and in which conditioning tasks they accept.
    pub const fn h3_33b() -> Self {
        Self {
            hidden: 5376,
            n_layers: 50,
            token_refiner_layers: 2,
            n_heads: 56,
            head_dim: 128,
            ffn_hidden: 14336,
            latents_dim: 24,
            audio_latents_dim: 32,
            patch: (1, 2, 2),
            text_dim: 5120,
            timestep_input_dim: 256,
            time_embed_hidden: 5376,
            time_embed_dim: 2688,
            adaln_out: 96768,
            final_adaln_out: 10752,
            adaln_rank: None,
            vsa_sparsity: None,
            dropped_blocks: &[],
            rope_inv_freq_len: 16,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            final_norm_eps: 1e-5,
        }
    }

    /// FastVideo FastH3-Trim: the H3 topology with 8 blocks pruned (42 kept,
    /// renumbered) and the per-block AdaLN projections reduced to rank 16 over a
    /// shared basis.
    pub const fn fasth3_trim() -> Self {
        let mut c = Self::h3_33b();
        c.n_layers = 42;
        c.adaln_rank = Some(16);
        // `fastvideo_inference.json`: VIDEO_SPARSE_ATTN_H3, tile 64.
        c.vsa_sparsity = Some(0.8);
        // `transformer/config.json` keeps every block but these.
        c.dropped_blocks = &[6, 7, 9, 13, 15, 16, 22, 23];
        c
    }

    /// This checkpoint's index for block `i` of the released 50-block H3, or
    /// `None` when the trim removed it.
    pub fn block_for_base(&self, i: usize) -> Option<usize> {
        if self.dropped_blocks.contains(&i) || i >= self.n_layers + self.dropped_blocks.len() {
            return None;
        }
        Some(i - self.dropped_blocks.iter().filter(|&&d| d < i).count())
    }

    /// Attention inner width (`n_heads * head_dim`). NB this is 7168, which is
    /// deliberately WIDER than `hidden` (5376): the attention inner dimension is
    /// not tied to the residual stream.
    pub const fn inner_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    /// Row width of a patchified video latent (`latents_dim * prod(patch)`).
    pub const fn video_patch_dim(&self) -> usize {
        let (t, h, w) = self.patch;
        self.latents_dim * t * h * w
    }

    /// Head channels rotated by RoPE: three axes, doubled for the rotate-half
    /// convention. The remaining `head_dim - rotary_dim` channels pass through.
    pub const fn rotary_dim(&self) -> usize {
        2 * 3 * self.rope_inv_freq_len
    }
}

/// Every row of the packed sequence carries a modality tag. One set of AdaLN
/// modulation parameters is kept per `(timestep, modality)` pair.
pub const MODALITY_NUM: usize = 3;

/// Modality tags. Padding rows carry [`TAG_PAD`] and are clamped to 0 before
/// indexing the modulation table; we emit no padding, so this is defensive.
pub const TAG_VIDEO: i32 = 0;
pub const TAG_TEXT: i32 = 1;
pub const TAG_AUDIO: i32 = 2;
pub const TAG_PAD: i32 = -1;

/// Sigma shifts of the released checkpoints: video and audio run independent
/// schedules over one shared transformer forward.
pub const SIGMA_SHIFT_VIDEO: f32 = 12.0;
pub const SIGMA_SHIFT_AUDIO: f32 = 3.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_is_wider_than_the_residual_stream() {
        let c = DitConfig::h3_33b();
        assert_eq!(c.inner_dim(), 7168);
        assert!(c.inner_dim() > c.hidden);
    }

    /// `adaln_out` must be exactly six modulation params per modality, or the
    /// precomputed table is sliced wrong and every block is silently mismodulated.
    #[test]
    fn adaln_width_is_six_params_per_modality() {
        let c = DitConfig::h3_33b();
        assert_eq!(c.adaln_out, 6 * MODALITY_NUM * c.hidden);
        assert_eq!(c.final_adaln_out, 2 * c.hidden);
    }

    #[test]
    fn rotary_leaves_channels_unrotated() {
        let c = DitConfig::h3_33b();
        assert_eq!(c.rotary_dim(), 96);
        assert!(
            c.rotary_dim() < c.head_dim,
            "some channels must pass through"
        );
    }

    #[test]
    fn video_patch_row_width() {
        assert_eq!(DitConfig::h3_33b().video_patch_dim(), 96);
    }
}
