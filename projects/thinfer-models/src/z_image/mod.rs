//! Z-Image-Turbo. fp16 ema-only, M1 target.
//!
//! Source: github.com/Tongyi-MAI/Z-Image (`src/zimage/transformer.py`).
//! Per-model details live in `projects/thinfer-working-area/zimage-plan.md`.
//!
//! Engine-wide rules (Module shape, residency, op trait) live in
//! `thinfer-core`; this module is glue: shapes, weight-name maps, the per-block
//! op recipe. Cross-model abstractions belong in `thinfer-core`, not here.

use thinfer_core::format::union::QkvTriple;
use thinfer_core::weight::WeightId;

pub mod audit;
pub mod block;
pub mod dit;
pub mod embedders;
pub mod final_layer;
pub mod loader;
pub mod manifest;
pub mod pipeline;
pub mod rope_embedder;
pub mod scheduler;
pub mod seq;
pub mod source;
pub mod t_embedder;
pub mod text_encoder;
pub mod tokenizer;
pub mod vae;

/// Static config matching `ZImageTransformer2DModel` defaults for the turbo
/// checkpoint. Audited 2026-05-09 against
/// `models/Z-Image-Turbo/diffusion_pytorch_model-ema-only-fp16.safetensors`.
pub mod config {
    pub const DIM: usize = 3840;
    pub const N_HEADS: usize = 30;
    pub const N_KV_HEADS: usize = 30;
    pub const HEAD_DIM: usize = DIM / N_HEADS; // 128
    /// `int(dim / 3 * 8)` per `FeedForward`. Not a clean multiple; bake in.
    pub const FFN_HIDDEN: usize = 10240;
    pub const N_LAYERS: usize = 30;
    pub const N_REFINER_LAYERS: usize = 2;
    pub const NORM_EPS: f32 = 1e-5;
    pub const ADALN_EMBED_DIM: usize = 256;
    pub const FREQUENCY_EMBEDDING_SIZE: usize = 256;
    pub const T_EMBEDDER_MID: usize = 1024;
    pub const T_SCALE: f32 = 1000.0;
    pub const CAP_FEAT_DIM: usize = 2560;
    /// Z-Image-Turbo checkpoint sets `rope_theta=256.0` (NOT the 10000.0
    /// LLaMA/Qwen3 default). Mismatch produces near-collapse DiT output
    /// (see gray-PNG bug fix 2026-05-15).
    pub const ROPE_THETA: f32 = 256.0;
    /// 3-axis RoPE: temporal, height, width. Sum == HEAD_DIM. Z-Image-Turbo
    /// checkpoint sets `[32, 48, 48]` (NOT diffusers' default `[32, 56, 56]`
    /// or our prior `[16, 56, 56]`). Same gray-PNG bug.
    pub const ROPE_AXES_DIMS: [usize; 3] = [32, 48, 48];
    /// Per-axis precomputed-table lengths (max coordinate index per axis).
    pub const ROPE_AXES_LENS: [usize; 3] = [1536, 512, 512];
    /// Sequence-length padding multiple (per upstream `SEQ_MULTI_OF`).
    pub const SEQ_MULTI_OF: usize = 32;
    /// adaLN out width = 4 * dim (scale_msa, gate_msa, scale_mlp, gate_mlp).
    pub const ADALN_OUT: usize = 4 * DIM;
    /// On-disk patching variant present in the checkpoint. Other entries
    /// (`all_x_embedder.<ps>-<fps>.*`) would appear if more variants were
    /// trained; turbo ships only this one.
    pub const PATCH_KEY: &str = "2-1";
}

/// Which transformer-block stack a block belongs to. Drives weight-key prefix
/// and whether the adaLN modulation pathway is active.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    /// Main DiT stack (`layers.<i>`), modulation=True.
    Main,
    /// Per-image noise refiner (`noise_refiner.<i>`), modulation=True.
    NoiseRefiner,
    /// Per-text context refiner (`context_refiner.<i>`), modulation=False.
    ContextRefiner,
}

impl BlockKind {
    pub fn prefix(self) -> &'static str {
        match self {
            Self::Main => "layers",
            Self::NoiseRefiner => "noise_refiner",
            Self::ContextRefiner => "context_refiner",
        }
    }

    pub fn modulated(self) -> bool {
        matches!(self, Self::Main | Self::NoiseRefiner)
    }
}

/// Resolved `WeightId`s for one transformer block. String-keyed lookup happens
/// once at load; forward path threads the resolved ids.
///
/// Names mirror `ZImageAttention` / `FeedForward` field paths so this also
/// serves as the audit map for fixture verification.
#[derive(Clone, Debug)]
pub struct BlockWeights {
    pub attention_norm1: WeightId,
    pub attention_norm2: WeightId,
    pub ffn_norm1: WeightId,
    pub ffn_norm2: WeightId,
    /// Canonical upstream-schema fused QKV: `[3*hq*head_dim, dim]` in source
    /// layout. Q rows [0, H), K rows [H, 2H), V rows [2H, 3H). GGUF Z-Image
    /// ships this directly; split safetensors checkpoints go through
    /// `SplitToFusedQkvSource` upstream of the loader.
    pub attn_qkv: WeightId,
    pub attn_to_out: WeightId,
    pub attn_norm_q: WeightId,
    pub attn_norm_k: WeightId,
    pub ffn_w1: WeightId,
    pub ffn_w2: WeightId,
    pub ffn_w3: WeightId,
    /// `Some` iff `kind.modulated()`. Single Linear (weight + bias),
    /// `[ADALN_OUT, ADALN_EMBED_DIM]`.
    pub adaln_modulation: Option<AdaLnWeights>,
}

#[derive(Clone, Debug)]
pub struct AdaLnWeights {
    pub weight: WeightId,
    pub bias: WeightId,
}

impl BlockWeights {
    pub fn new(kind: BlockKind, idx: usize) -> Self {
        let p = format!("{}.{}", kind.prefix(), idx);
        let id = |suffix: &str| WeightId(format!("{p}.{suffix}"));
        Self {
            attention_norm1: id("attention_norm1.weight"),
            attention_norm2: id("attention_norm2.weight"),
            ffn_norm1: id("ffn_norm1.weight"),
            ffn_norm2: id("ffn_norm2.weight"),
            attn_qkv: id("attention.qkv.weight"),
            attn_to_out: id("attention.out.weight"),
            attn_norm_q: id("attention.norm_q.weight"),
            attn_norm_k: id("attention.norm_k.weight"),
            ffn_w1: id("feed_forward.w1.weight"),
            ffn_w2: id("feed_forward.w2.weight"),
            ffn_w3: id("feed_forward.w3.weight"),
            adaln_modulation: kind.modulated().then(|| AdaLnWeights {
                weight: id("adaLN_modulation.0.weight"),
                bias: id("adaLN_modulation.0.bias"),
            }),
        }
    }
}

/// Module-level (non-block) weights.
#[derive(Clone, Debug)]
pub struct ModelWeights {
    /// `t_embedder.mlp.{0,2}.{weight,bias}`. Linear -> SiLU -> Linear,
    /// `[FREQUENCY_EMBEDDING_SIZE -> T_EMBEDDER_MID -> ADALN_EMBED_DIM]`.
    /// adaLN time-embed source: every modulated block reads this output.
    pub t_embedder: TEmbedderWeights,
    /// `cap_embedder.0.weight` is RMSNorm gain over `CAP_FEAT_DIM`;
    /// `cap_embedder.1.{weight,bias}` is `Linear(CAP_FEAT_DIM, DIM)`.
    pub cap_embedder: CapEmbedderWeights,
    /// `all_x_embedder.<patch>-<f_patch>.{weight,bias}`. Single variant in
    /// checkpoint (`2-1`).
    pub x_embedder: LinearWeights,
    /// `all_final_layer.<patch>-<f_patch>.linear.{weight,bias}` and
    /// `all_final_layer.<patch>-<f_patch>.adaLN_modulation.1.{weight,bias}`.
    /// `adaLN_modulation.0` is SiLU (no params).
    pub final_layer: FinalLayerWeights,
    /// Learned [1, DIM] pad tokens substituted into masked positions.
    pub x_pad_token: WeightId,
    pub cap_pad_token: WeightId,
}

#[derive(Clone, Debug)]
pub struct TEmbedderWeights {
    pub fc1_weight: WeightId,
    pub fc1_bias: WeightId,
    pub fc2_weight: WeightId,
    pub fc2_bias: WeightId,
}

#[derive(Clone, Debug)]
pub struct CapEmbedderWeights {
    pub norm_weight: WeightId,
    pub linear_weight: WeightId,
    pub linear_bias: WeightId,
}

#[derive(Clone, Debug)]
pub struct LinearWeights {
    pub weight: WeightId,
    pub bias: WeightId,
}

#[derive(Clone, Debug)]
pub struct FinalLayerWeights {
    pub linear: LinearWeights,
    pub adaln: AdaLnWeights,
}

impl ModelWeights {
    pub fn new() -> Self {
        let id = |s: &str| WeightId(s.to_string());
        let patch = config::PATCH_KEY;
        Self {
            t_embedder: TEmbedderWeights {
                fc1_weight: id("t_embedder.mlp.0.weight"),
                fc1_bias: id("t_embedder.mlp.0.bias"),
                fc2_weight: id("t_embedder.mlp.2.weight"),
                fc2_bias: id("t_embedder.mlp.2.bias"),
            },
            cap_embedder: CapEmbedderWeights {
                norm_weight: id("cap_embedder.0.weight"),
                linear_weight: id("cap_embedder.1.weight"),
                linear_bias: id("cap_embedder.1.bias"),
            },
            x_embedder: LinearWeights {
                weight: WeightId(format!("all_x_embedder.{patch}.weight")),
                bias: WeightId(format!("all_x_embedder.{patch}.bias")),
            },
            final_layer: FinalLayerWeights {
                linear: LinearWeights {
                    weight: WeightId(format!("all_final_layer.{patch}.linear.weight")),
                    bias: WeightId(format!("all_final_layer.{patch}.linear.bias")),
                },
                adaln: AdaLnWeights {
                    weight: WeightId(format!("all_final_layer.{patch}.adaLN_modulation.1.weight")),
                    bias: WeightId(format!("all_final_layer.{patch}.adaLN_modulation.1.bias")),
                },
            },
            x_pad_token: id("x_pad_token"),
            cap_pad_token: id("cap_pad_token"),
        }
    }
}

impl Default for ModelWeights {
    fn default() -> Self {
        Self::new()
    }
}

/// Every QKV triple in a Z-Image DiT, paired with the canonical fused id the
/// engine asks for. Pass to `SplitToFusedQkvSource::new` over a safetensors
/// source that ships split `to_q`/`to_k`/`to_v` (dimitribarbot's checkpoint).
/// Sources that already ship the fused entry (`unsloth/Z-Image-Turbo-GGUF`)
/// bypass the adapter or simply have no matching split entries to fuse.
/// Renames every block's split `attention.to_out.0.weight` (dimitribarbot
/// safetensors schema) to canonical `attention.out.weight` (upstream /
/// unsloth GGUF schema). Pass to `RenamedSource::new` over a safetensors
/// source so the engine can ask for the canonical id everywhere.
pub fn dit_to_out_renames() -> std::collections::HashMap<WeightId, WeightId> {
    let mut out =
        std::collections::HashMap::with_capacity(config::N_LAYERS + 2 * config::N_REFINER_LAYERS);
    for kind in [
        BlockKind::Main,
        BlockKind::NoiseRefiner,
        BlockKind::ContextRefiner,
    ] {
        let n = match kind {
            BlockKind::Main => config::N_LAYERS,
            BlockKind::NoiseRefiner | BlockKind::ContextRefiner => config::N_REFINER_LAYERS,
        };
        for i in 0..n {
            let p = format!("{}.{}", kind.prefix(), i);
            out.insert(
                WeightId(format!("{p}.attention.to_out.0.weight")),
                WeightId(format!("{p}.attention.out.weight")),
            );
        }
    }
    out
}

/// GGUF -> canonical id renames for `unsloth/Z-Image-Turbo-GGUF`. The GGUF
/// ships canonical names for almost everything; the exceptions are the
/// flattened single-patch-variant modules (`x_embedder.*` /
/// `final_layer.*` vs canonical `all_x_embedder.<patch>.*` /
/// `all_final_layer.<patch>.*`) and the swapped q/k norm spelling
/// (`attention.q_norm` vs canonical `attention.norm_q`). Pass to
/// `RenamedSource::with_passthrough` over the GGUF source.
pub fn dit_gguf_renames() -> std::collections::HashMap<WeightId, WeightId> {
    let patch = config::PATCH_KEY;
    let mut out = std::collections::HashMap::with_capacity(
        6 + 2 * (config::N_LAYERS + 2 * config::N_REFINER_LAYERS),
    );
    for suffix in ["weight", "bias"] {
        out.insert(
            WeightId(format!("x_embedder.{suffix}")),
            WeightId(format!("all_x_embedder.{patch}.{suffix}")),
        );
        out.insert(
            WeightId(format!("final_layer.linear.{suffix}")),
            WeightId(format!("all_final_layer.{patch}.linear.{suffix}")),
        );
        out.insert(
            WeightId(format!("final_layer.adaLN_modulation.1.{suffix}")),
            WeightId(format!(
                "all_final_layer.{patch}.adaLN_modulation.1.{suffix}"
            )),
        );
    }
    for kind in [
        BlockKind::Main,
        BlockKind::NoiseRefiner,
        BlockKind::ContextRefiner,
    ] {
        let n = match kind {
            BlockKind::Main => config::N_LAYERS,
            BlockKind::NoiseRefiner | BlockKind::ContextRefiner => config::N_REFINER_LAYERS,
        };
        for i in 0..n {
            let p = format!("{}.{}", kind.prefix(), i);
            out.insert(
                WeightId(format!("{p}.attention.q_norm.weight")),
                WeightId(format!("{p}.attention.norm_q.weight")),
            );
            out.insert(
                WeightId(format!("{p}.attention.k_norm.weight")),
                WeightId(format!("{p}.attention.norm_k.weight")),
            );
        }
    }
    out
}

/// GGUF -> canonical id renames for the Qwen3 text encoder GGUF
/// (`worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF`). llama.cpp tensor
/// naming (`blk.{i}.attn_q.weight`, `token_embd.weight`) maps to the HF
/// Qwen3 ids the encoder registers (`model.layers.{i}.self_attn.q_proj.
/// weight`, `model.embed_tokens.weight`). `output_norm` is left unmapped:
/// the encoder stops at `hidden_states[-2]` and never reads it.
pub fn qwen3_gguf_renames() -> std::collections::HashMap<WeightId, WeightId> {
    use crate::z_image::text_encoder::config::N_LAYERS;
    const SITES: [(&str, &str); 11] = [
        ("attn_norm", "input_layernorm"),
        ("ffn_norm", "post_attention_layernorm"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("attn_q_norm", "self_attn.q_norm"),
        ("attn_k_norm", "self_attn.k_norm"),
        ("ffn_gate", "mlp.gate_proj"),
        ("ffn_up", "mlp.up_proj"),
        ("ffn_down", "mlp.down_proj"),
    ];
    let mut out = std::collections::HashMap::with_capacity(1 + SITES.len() * N_LAYERS);
    out.insert(
        WeightId("token_embd.weight".into()),
        WeightId("model.embed_tokens.weight".into()),
    );
    for i in 0..N_LAYERS {
        for (gguf, hf) in SITES {
            out.insert(
                WeightId(format!("blk.{i}.{gguf}.weight")),
                WeightId(format!("model.layers.{i}.{hf}.weight")),
            );
        }
    }
    out
}

pub fn dit_qkv_triples() -> Vec<QkvTriple> {
    let mut out = Vec::with_capacity(config::N_LAYERS + 2 * config::N_REFINER_LAYERS);
    for kind in [
        BlockKind::Main,
        BlockKind::NoiseRefiner,
        BlockKind::ContextRefiner,
    ] {
        let n = match kind {
            BlockKind::Main => config::N_LAYERS,
            BlockKind::NoiseRefiner | BlockKind::ContextRefiner => config::N_REFINER_LAYERS,
        };
        for i in 0..n {
            let p = format!("{}.{}", kind.prefix(), i);
            out.push(QkvTriple {
                fused: WeightId(format!("{p}.attention.qkv.weight")),
                q: WeightId(format!("{p}.attention.to_q.weight")),
                k: WeightId(format!("{p}.attention.to_k.weight")),
                v: WeightId(format!("{p}.attention.to_v.weight")),
            });
        }
    }
    out
}
