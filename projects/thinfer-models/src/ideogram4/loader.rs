//! Ideogram-4 DiT weight registration from the DiT GGUF.
//!
//! GGUF tensor keys are 1:1 with the modeling-module paths (no rename map).
//! Per-layer matmuls (`attention.{qkv,o}`, `feed_forward.{w1,w2,w3}`,
//! `adaln_modulation`) are Q8_0; everything else (norms, biases, embeds, the
//! module-level projections, `t_embedding`, `final_layer`) is BF16.
//!
//! Quant routing mirrors the Z-Image DiT (`z_image/loader.rs`): the four block
//! projection matmuls register as-is (`Quant -> TransposePolicy::None`, the
//! dequant-once / DP4A path reads block-major `[N,K]`); the block adaln
//! modulation registers dense (dequant Q8_0 -> bf16, M=1 modulation matmul is
//! always bf16); the bf16 module-level linears transpose `[N,K] -> [K,N]`.

use thinfer_core::residency::WeightResidency;
use thinfer_core::weight::{WeightId, WeightSource};

use crate::common::block::{AdaLnHandles, BlockHandles};
use crate::common::embedders::LinearBiasHandles;
use crate::common::loader::{
    LoadError, register_linear, register_linear_dense_opt_ring, register_linear_transcode,
    register_passthrough,
};
use crate::z_image::embedders::CapEmbedderHandles;
use crate::z_image::final_layer::FinalLayerHandles;

use super::config;
use super::t_embedder::TEmbedderHandles;

/// All DiT weights, resolved to residency handles.
#[derive(Clone, Debug)]
pub struct DitHandles {
    /// `input_proj`: noise patch `Linear(128 -> 4608, bias)`.
    pub input_proj: LinearBiasHandles,
    /// `llm_cond_norm` (RMSNorm 53248) + `llm_cond_proj` (Linear 53248 -> 4608,
    /// bias). Shares the `CapEmbedder` norm+linear shape.
    pub llm_cond: CapEmbedderHandles,
    /// `t_embedding` MLP + `adaln_proj`.
    pub t_embedder: TEmbedderHandles,
    /// `embed_image_indicator`: `nn.Embedding(2, 4608)` (passthrough).
    pub embed_image_indicator: thinfer_core::residency::WeightHandle,
    /// `final_layer`: LayerNorm(no affine) + adaln scale + Linear(4608 -> 128).
    pub final_layer: FinalLayerHandles,
    /// The 34 transformer blocks.
    pub layers: Vec<BlockHandles>,
}

fn id(s: impl Into<String>) -> WeightId {
    WeightId(s.into())
}

fn register_linear_bias<S: WeightSource>(
    residency: &WeightResidency<S>,
    weight: &str,
    bias: &str,
) -> Result<LinearBiasHandles, LoadError> {
    Ok(LinearBiasHandles {
        weight: register_linear(residency, &id(weight))?,
        bias: register_passthrough(residency, &id(bias))?,
    })
}

fn register_block<S: WeightSource>(
    residency: &WeightResidency<S>,
    i: usize,
) -> Result<BlockHandles, LoadError> {
    let p = format!("layers.{i}");
    // Block projection matmuls are Q8_0 in the file: register as-is (no
    // transcode); the dequant-once bf16 path (or DP4A under f16 acts) reads
    // the block-major layout.
    let lin = |suffix: &str| {
        register_linear_transcode(residency, &id(format!("{p}.{suffix}")), None, None)
    };
    let pass = |suffix: &str| register_passthrough(residency, &id(format!("{p}.{suffix}")));
    Ok(BlockHandles {
        attention_norm1: pass("attention_norm1.weight")?,
        attention_norm2: pass("attention_norm2.weight")?,
        ffn_norm1: pass("ffn_norm1.weight")?,
        ffn_norm2: pass("ffn_norm2.weight")?,
        attn_qkv: lin("attention.qkv.weight")?,
        attn_to_out: lin("attention.o.weight")?,
        attn_norm_q: pass("attention.norm_q.weight")?,
        attn_norm_k: pass("attention.norm_k.weight")?,
        ffn_w1: lin("feed_forward.w1.weight")?,
        ffn_w2: lin("feed_forward.w2.weight")?,
        ffn_w3: lin("feed_forward.w3.weight")?,
        adaln: Some(AdaLnHandles {
            weight: register_linear_dense_opt_ring(
                residency,
                &id(format!("{p}.adaln_modulation.weight")),
                None,
            )?,
            bias: pass("adaln_modulation.bias")?,
        }),
    })
}

/// Register every DiT weight. `register_handles` opens nothing; it walks the
/// source catalog and registers residency handles (lazy upload on acquire).
pub fn register_handles<S: WeightSource>(
    residency: &WeightResidency<S>,
) -> Result<DitHandles, LoadError> {
    let layers = (0..config::N_LAYERS)
        .map(|i| register_block(residency, i))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(DitHandles {
        input_proj: register_linear_bias(residency, "input_proj.weight", "input_proj.bias")?,
        llm_cond: CapEmbedderHandles {
            norm_weight: register_passthrough(residency, &id("llm_cond_norm.weight"))?,
            linear: register_linear_bias(residency, "llm_cond_proj.weight", "llm_cond_proj.bias")?,
        },
        t_embedder: TEmbedderHandles {
            mlp_in: register_linear_bias(
                residency,
                "t_embedding.mlp_in.weight",
                "t_embedding.mlp_in.bias",
            )?,
            mlp_out: register_linear_bias(
                residency,
                "t_embedding.mlp_out.weight",
                "t_embedding.mlp_out.bias",
            )?,
            adaln_proj: register_linear_bias(residency, "adaln_proj.weight", "adaln_proj.bias")?,
        },
        embed_image_indicator: register_passthrough(
            residency,
            &id("embed_image_indicator.weight"),
        )?,
        final_layer: FinalLayerHandles {
            linear: register_linear_bias(
                residency,
                "final_layer.linear.weight",
                "final_layer.linear.bias",
            )?,
            adaln: register_linear_bias(
                residency,
                "final_layer.adaln_modulation.weight",
                "final_layer.adaln_modulation.bias",
            )?,
        },
        layers,
    })
}
