//! Qwen3-VL-Instruct text generator (HunyuanVideo prompt rewriter).
//!
//! This is the NATIVE prompt-rewriter LM: a text-only Qwen3 decoder stack that
//! rewrites a user's terse prompt into the richer phrasing the HunyuanVideo DiT
//! was trained on. The cached assets are the text-only `qwen3vl` GGUFs
//! (`unsloth/Qwen3-VL-{8B,4B}-Instruct-GGUF`); they carry NO vision tensors, so
//! the load path is a plain causal-LM stack: a token embedding, N decoder
//! layers, a final RMSNorm, and an `lm_head` (untied `output.weight` on the 8B;
//! the 4B ties the head to `token_embd`).
//!
//! The stack is runtime-parameterized by [`Qwen3LmConfig`] so the SAME code loads
//! either size (they differ only in `hidden` / `ffn_hidden` and whether the head
//! is tied). The decoder layers reuse the Z-Image Qwen3 block machinery
//! ([`crate::z_image::text_encoder`]) since the per-layer shape (GQA 32 Q / 8 KV
//! heads, head_dim 128, per-head Q/K RMSNorm, SwiGLU FFN) is identical; only the
//! dimensions and the presence of the final norm + lm_head differ.

pub mod forward;
pub mod generate;

/// Runtime architecture config for a Qwen3-VL text-only rewriter.
///
/// Both sizes are the `qwen3vl` arch (GQA 32/8, head_dim 128, per-head Q/K
/// RMSNorm, SwiGLU, rope theta 5e6, rms eps 1e-6); they differ only in the two
/// widths and whether `lm_head` is tied to the token embedding. Values are the
/// GGUF metadata of the cached `unsloth/Qwen3-VL-{8B,4B}-Instruct-GGUF` Q5_K_M
/// files (dumped 2026-07-01).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen3LmConfig {
    pub hidden: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_hidden: usize,
    pub vocab: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// `lm_head` shares the token-embedding matrix (the GGUF ships no separate
    /// `output.weight`). The 4B ties; the 8B does not.
    pub tied_embeddings: bool,
}

impl Qwen3LmConfig {
    /// Qwen3-VL-8B-Instruct (`hidden` 4096, `ffn` 12288, untied lm_head).
    pub const fn qwen3_vl_8b() -> Self {
        Self {
            hidden: 4096,
            n_layers: 36,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            ffn_hidden: 12288,
            vocab: 151936,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            tied_embeddings: false,
        }
    }

    /// Qwen3-VL-4B-Instruct (`hidden` 2560, `ffn` 9728, TIED lm_head = the fast
    /// default rewriter: ~2.5GB Q5_K_M vs the 8B's ~5.85GB).
    pub const fn qwen3_vl_4b() -> Self {
        Self {
            hidden: 2560,
            n_layers: 36,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            ffn_hidden: 9728,
            vocab: 151936,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            tied_embeddings: true,
        }
    }

    /// GQA key/value width in elements (`n_kv_heads * head_dim`).
    pub const fn kv_width(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}
