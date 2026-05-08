//! Residency-aware Z-Image DiT loader. Walks the typed `WeightId` bundles in
//! `mod.rs`, looks each tensor up in the source's `WeightCatalog`, builds a
//! `WeightMeta` (with decode + transpose policy) and registers it with the
//! `WeightResidency` manager. Returns `LoadedDitHandles`.
//!
//! No bytes flow here. Decode + transpose + GPU upload happen lazily on
//! `WeightResidency::acquire`, which the per-block forward calls.
//!
//! Linear weight tensors get `TransposePolicy::Linear2D` (matmul convention is
//! `A @ B` with B in `[K, N]`; PyTorch ships `[N, K]`). RMSNorm gains, biases,
//! and pad tokens are 1-D and use `TransposePolicy::None`.

use thinfer_core::residency::{
    ResidencyError, TransposePolicy, WeightHandle, WeightMeta, WeightResidency,
};
use thinfer_core::tensor::StorageEncoding;
use thinfer_core::weight::{WeightId, WeightSource};

use crate::z_image::block::{AdaLnHandles, BlockHandles};
use crate::z_image::embedders::{CapEmbedderHandles, LinearBiasHandles};
use crate::z_image::final_layer::FinalLayerHandles;
use crate::z_image::t_embedder::TEmbedderWeightHandles;
use crate::z_image::{
    AdaLnWeights, BlockKind, BlockWeights, CapEmbedderWeights, FinalLayerWeights, LinearWeights,
    ModelWeights, TEmbedderWeights, config,
};

#[derive(Clone, Debug)]
pub struct LoadedDitHandles {
    pub x_embedder: LinearBiasHandles,
    pub cap_embedder: CapEmbedderHandles,
    pub t_embedder: TEmbedderWeightHandles,
    pub final_layer: FinalLayerHandles,
    pub noise_refiner: Vec<BlockHandles>,
    pub context_refiner: Vec<BlockHandles>,
    pub layers: Vec<BlockHandles>,
    pub x_pad_token: WeightHandle,
    pub cap_pad_token: WeightHandle,
}

#[derive(Debug)]
pub enum LoadError {
    UnknownWeight(WeightId),
    /// On-disk encoding can't be decoded into fp32 (quantized, or fp16 which
    /// we intentionally don't support for Z-Image).
    Undecodable {
        id: WeightId,
        encoding: Option<StorageEncoding>,
        label: String,
    },
}

pub fn register_dit_handles<S: WeightSource>(
    residency: &WeightResidency<S>,
) -> Result<LoadedDitHandles, LoadError> {
    let mw = ModelWeights::new();
    Ok(LoadedDitHandles {
        x_embedder: register_linear_bias(residency, &mw.x_embedder)?,
        cap_embedder: register_cap_embedder(residency, &mw.cap_embedder)?,
        t_embedder: register_t_embedder(residency, &mw.t_embedder)?,
        final_layer: register_final_layer(residency, &mw.final_layer)?,
        noise_refiner: register_block_stack(residency, BlockKind::NoiseRefiner)?,
        context_refiner: register_block_stack(residency, BlockKind::ContextRefiner)?,
        layers: register_block_stack(residency, BlockKind::Main)?,
        x_pad_token: register_passthrough(residency, &mw.x_pad_token)?,
        cap_pad_token: register_passthrough(residency, &mw.cap_pad_token)?,
    })
}

fn register_block_stack<S: WeightSource>(
    residency: &WeightResidency<S>,
    kind: BlockKind,
) -> Result<Vec<BlockHandles>, LoadError> {
    let n = match kind {
        BlockKind::Main => config::N_LAYERS,
        BlockKind::NoiseRefiner | BlockKind::ContextRefiner => config::N_REFINER_LAYERS,
    };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(register_block(residency, kind, i)?);
    }
    Ok(out)
}

fn register_block<S: WeightSource>(
    residency: &WeightResidency<S>,
    kind: BlockKind,
    idx: usize,
) -> Result<BlockHandles, LoadError> {
    let w = BlockWeights::new(kind, idx);
    let adaln = match w.adaln_modulation {
        Some(a) => Some(register_adaln(residency, &a)?),
        None => None,
    };
    Ok(BlockHandles {
        attention_norm1: register_passthrough(residency, &w.attention_norm1)?,
        attention_norm2: register_passthrough(residency, &w.attention_norm2)?,
        ffn_norm1: register_passthrough(residency, &w.ffn_norm1)?,
        ffn_norm2: register_passthrough(residency, &w.ffn_norm2)?,
        attn_to_q: register_linear(residency, &w.attn_to_q)?,
        attn_to_k: register_linear(residency, &w.attn_to_k)?,
        attn_to_v: register_linear(residency, &w.attn_to_v)?,
        attn_to_out: register_linear(residency, &w.attn_to_out)?,
        attn_norm_q: register_passthrough(residency, &w.attn_norm_q)?,
        attn_norm_k: register_passthrough(residency, &w.attn_norm_k)?,
        ffn_w1: register_linear(residency, &w.ffn_w1)?,
        ffn_w2: register_linear(residency, &w.ffn_w2)?,
        ffn_w3: register_linear(residency, &w.ffn_w3)?,
        adaln,
    })
}

fn register_adaln<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &AdaLnWeights,
) -> Result<AdaLnHandles, LoadError> {
    Ok(AdaLnHandles {
        weight: register_linear(residency, &w.weight)?,
        bias: register_passthrough(residency, &w.bias)?,
    })
}

fn register_linear_bias<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &LinearWeights,
) -> Result<LinearBiasHandles, LoadError> {
    Ok(LinearBiasHandles {
        weight: register_linear(residency, &w.weight)?,
        bias: register_passthrough(residency, &w.bias)?,
    })
}

fn register_cap_embedder<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &CapEmbedderWeights,
) -> Result<CapEmbedderHandles, LoadError> {
    Ok(CapEmbedderHandles {
        norm_weight: register_passthrough(residency, &w.norm_weight)?,
        linear: LinearBiasHandles {
            weight: register_linear(residency, &w.linear_weight)?,
            bias: register_passthrough(residency, &w.linear_bias)?,
        },
    })
}

fn register_t_embedder<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &TEmbedderWeights,
) -> Result<TEmbedderWeightHandles, LoadError> {
    Ok(TEmbedderWeightHandles {
        fc1_weight: register_linear(residency, &w.fc1_weight)?,
        fc1_bias: register_passthrough(residency, &w.fc1_bias)?,
        fc2_weight: register_linear(residency, &w.fc2_weight)?,
        fc2_bias: register_passthrough(residency, &w.fc2_bias)?,
    })
}

fn register_final_layer<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &FinalLayerWeights,
) -> Result<FinalLayerHandles, LoadError> {
    Ok(FinalLayerHandles {
        linear: LinearBiasHandles {
            weight: register_linear(residency, &w.linear.weight)?,
            bias: register_passthrough(residency, &w.linear.bias)?,
        },
        adaln: LinearBiasHandles {
            weight: register_linear(residency, &w.adaln.weight)?,
            bias: register_passthrough(residency, &w.adaln.bias)?,
        },
    })
}

pub(crate) fn register_linear<S: WeightSource>(
    residency: &WeightResidency<S>,
    id: &WeightId,
) -> Result<WeightHandle, LoadError> {
    register_one(residency, id, TransposePolicy::Linear2D)
}

pub(crate) fn register_passthrough<S: WeightSource>(
    residency: &WeightResidency<S>,
    id: &WeightId,
) -> Result<WeightHandle, LoadError> {
    register_one(residency, id, TransposePolicy::None)
}

pub(crate) fn register_one<S: WeightSource>(
    residency: &WeightResidency<S>,
    id: &WeightId,
    transpose: TransposePolicy,
) -> Result<WeightHandle, LoadError> {
    let entry = residency
        .source()
        .catalog()
        .get(id)
        .ok_or_else(|| LoadError::UnknownWeight(id.clone()))?;
    let encoding = entry.encoding.ok_or_else(|| LoadError::Undecodable {
        id: id.clone(),
        encoding: None,
        label: entry.encoding_label.clone(),
    })?;
    // Reject quantized / fp16 encodings up front via a probe `Decoder::new`.
    if thinfer_core::weight::Decoder::new(encoding).is_err() {
        return Err(LoadError::Undecodable {
            id: id.clone(),
            encoding: Some(encoding),
            label: entry.encoding_label.clone(),
        });
    }
    Ok(residency.register(WeightMeta {
        id: id.clone(),
        shape: entry.shape.clone(),
        encoding,
        on_disk_bytes: entry.size,
        transpose,
    }))
}

impl<SE: core::fmt::Debug, BE: core::fmt::Debug> From<ResidencyError<SE, BE>> for LoadError {
    fn from(_: ResidencyError<SE, BE>) -> Self {
        // Registration is sync and infallible at the residency layer; this is
        // unreachable but the trait makes call sites cleaner.
        unreachable!("register doesn't fail")
    }
}
