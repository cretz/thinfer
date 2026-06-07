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
    RegisterRingError, ResidencyError, RingId, TransposePolicy, WeightHandle, WeightMeta,
    WeightResidency,
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

/// Load-time requantize target for the refiner matmul weights. The unsloth
/// GGUFs quantize only the 30 main-layer matmuls; refiner matmuls stay bf16
/// in the file. When the main path is quant-flavored, transcode refiner
/// linears to Q8_0 at upload so they ride the same quant matmul path
/// (DP4A `matmul_i8` or the dequant workspace fallback) instead of the
/// untuned bf16 kernel, and re-upload half the bytes. Q8_0 regardless of
/// the main scheme: quality headroom is free and one encoder is enough.
/// `ZImageModel::load` mirrors this when building the refiner pipeline cfgs.
pub fn refiner_transcode_target<S: WeightSource>(
    residency: &WeightResidency<S>,
) -> Option<thinfer_core::quant::QuantKind> {
    let catalog = residency.source().catalog();
    let any_quant = (0..config::N_LAYERS).any(|li| {
        let bw = BlockWeights::new(BlockKind::Main, li);
        [&bw.attn_qkv, &bw.attn_to_out, &bw.ffn_w1, &bw.ffn_w2]
            .into_iter()
            .any(|id| {
                matches!(
                    catalog.get(id).and_then(|e| e.encoding),
                    Some(StorageEncoding::Quant(_))
                )
            })
    });
    any_quant.then_some(thinfer_core::quant::QuantKind::Q8_0)
}

pub fn register_dit_handles<S: WeightSource>(
    residency: &WeightResidency<S>,
    refiner_transcode: Option<thinfer_core::quant::QuantKind>,
) -> Result<LoadedDitHandles, LoadError> {
    let mw = ModelWeights::new();
    Ok(LoadedDitHandles {
        x_embedder: register_linear_bias(residency, &mw.x_embedder)?,
        cap_embedder: register_cap_embedder(residency, &mw.cap_embedder)?,
        t_embedder: register_t_embedder(residency, &mw.t_embedder)?,
        final_layer: register_final_layer(residency, &mw.final_layer)?,
        noise_refiner: register_block_stack(residency, BlockKind::NoiseRefiner, refiner_transcode)?,
        context_refiner: register_block_stack(
            residency,
            BlockKind::ContextRefiner,
            refiner_transcode,
        )?,
        layers: register_block_stack(residency, BlockKind::Main, None)?,
        x_pad_token: register_passthrough(residency, &mw.x_pad_token)?,
        cap_pad_token: register_passthrough(residency, &mw.cap_pad_token)?,
    })
}

fn register_block_stack<S: WeightSource>(
    residency: &WeightResidency<S>,
    kind: BlockKind,
    transcode: Option<thinfer_core::quant::QuantKind>,
) -> Result<Vec<BlockHandles>, LoadError> {
    let n = match kind {
        BlockKind::Main => config::N_LAYERS,
        BlockKind::NoiseRefiner | BlockKind::ContextRefiner => config::N_REFINER_LAYERS,
    };
    // Only the 30-block main loop sees sawtooth GPU util under memory
    // pressure; refiners are 2 blocks each and stay on the pool path.
    let rings = if matches!(kind, BlockKind::Main) {
        Some(block_ring_set(kind))
    } else {
        None
    };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(register_block(
            residency,
            kind,
            i,
            rings.as_ref(),
            transcode,
        )?);
    }
    Ok(out)
}

/// One `RingId` per `(BlockKind, weight-role)`. Allocator-stable IDs so
/// repeated calls on the same `WeightResidency` hit the same ring. Base
/// offset varies by `BlockKind`; refiners get their own range so a future
/// extension can ring them too without colliding with main-layer IDs.
struct BlockRingSet {
    attention_norm1: RingId,
    attention_norm2: RingId,
    ffn_norm1: RingId,
    ffn_norm2: RingId,
    attn_qkv: RingId,
    attn_to_out: RingId,
    attn_norm_q: RingId,
    attn_norm_k: RingId,
    ffn_w1: RingId,
    ffn_w2: RingId,
    ffn_w3: RingId,
    adaln_weight: RingId,
    adaln_bias: RingId,
}

fn block_ring_set(kind: BlockKind) -> BlockRingSet {
    let base = match kind {
        BlockKind::Main => 0u32,
        BlockKind::NoiseRefiner => 100,
        BlockKind::ContextRefiner => 200,
    };
    BlockRingSet {
        attention_norm1: RingId(base),
        attention_norm2: RingId(base + 1),
        ffn_norm1: RingId(base + 2),
        ffn_norm2: RingId(base + 3),
        attn_qkv: RingId(base + 4),
        attn_to_out: RingId(base + 5),
        attn_norm_q: RingId(base + 6),
        attn_norm_k: RingId(base + 7),
        ffn_w1: RingId(base + 8),
        ffn_w2: RingId(base + 9),
        ffn_w3: RingId(base + 10),
        adaln_weight: RingId(base + 11),
        adaln_bias: RingId(base + 12),
    }
}

fn register_block<S: WeightSource>(
    residency: &WeightResidency<S>,
    kind: BlockKind,
    idx: usize,
    rings: Option<&BlockRingSet>,
    transcode: Option<thinfer_core::quant::QuantKind>,
) -> Result<BlockHandles, LoadError> {
    let w = BlockWeights::new(kind, idx);
    let adaln = match w.adaln_modulation {
        Some(a) => Some(register_adaln(residency, &a, rings)?),
        None => None,
    };
    Ok(BlockHandles {
        attention_norm1: register_passthrough_opt_ring(
            residency,
            &w.attention_norm1,
            rings.map(|r| r.attention_norm1),
        )?,
        attention_norm2: register_passthrough_opt_ring(
            residency,
            &w.attention_norm2,
            rings.map(|r| r.attention_norm2),
        )?,
        ffn_norm1: register_passthrough_opt_ring(
            residency,
            &w.ffn_norm1,
            rings.map(|r| r.ffn_norm1),
        )?,
        ffn_norm2: register_passthrough_opt_ring(
            residency,
            &w.ffn_norm2,
            rings.map(|r| r.ffn_norm2),
        )?,
        attn_qkv: register_linear_transcode(
            residency,
            &w.attn_qkv,
            rings.map(|r| r.attn_qkv),
            transcode,
        )?,
        attn_to_out: register_linear_transcode(
            residency,
            &w.attn_to_out,
            rings.map(|r| r.attn_to_out),
            transcode,
        )?,
        attn_norm_q: register_passthrough_opt_ring(
            residency,
            &w.attn_norm_q,
            rings.map(|r| r.attn_norm_q),
        )?,
        attn_norm_k: register_passthrough_opt_ring(
            residency,
            &w.attn_norm_k,
            rings.map(|r| r.attn_norm_k),
        )?,
        ffn_w1: register_linear_transcode(
            residency,
            &w.ffn_w1,
            rings.map(|r| r.ffn_w1),
            transcode,
        )?,
        ffn_w2: register_linear_transcode(
            residency,
            &w.ffn_w2,
            rings.map(|r| r.ffn_w2),
            transcode,
        )?,
        ffn_w3: register_linear_transcode(
            residency,
            &w.ffn_w3,
            rings.map(|r| r.ffn_w3),
            transcode,
        )?,
        adaln,
    })
}

fn register_adaln<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &AdaLnWeights,
    rings: Option<&BlockRingSet>,
) -> Result<AdaLnHandles, LoadError> {
    Ok(AdaLnHandles {
        weight: register_linear_dense_opt_ring(
            residency,
            &w.weight,
            rings.map(|r| r.adaln_weight),
        )?,
        bias: register_passthrough_opt_ring(residency, &w.bias, rings.map(|r| r.adaln_bias))?,
    })
}

/// Dense-consumed linear: the matmul site reads bf16 `[K, N]` regardless of
/// file encoding. Quant files (GGUF checkpoints quantize the AdaLN
/// modulation weights) dequant to dense bf16 at upload (`encoding Quant +
/// TransposePolicy::Linear2D`, see `WeightMeta::gpu_encoding`); bf16/F32
/// files ride the plain `Linear2D` path. Never registers a quant GPU
/// layout: the adaln matmul pipeline compiles `WeightDtype::Bf16`
/// unconditionally (M=1 modulation matmul, see `BlockMatmuls::for_cfgs`).
fn register_linear_dense_opt_ring<S: WeightSource>(
    residency: &WeightResidency<S>,
    id: &WeightId,
    ring: Option<RingId>,
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
    match encoding {
        StorageEncoding::Bf16 | StorageEncoding::F32 => {}
        StorageEncoding::Quant(k) => {
            assert_eq!(entry.shape.0.len(), 2, "dense-linear quant must be 2-D");
            assert_eq!(
                entry.shape.0[1] % k.block_size() as usize,
                0,
                "dense-linear dequant requires K % block_size == 0 ({id:?})"
            );
        }
        _ => {
            return Err(LoadError::Undecodable {
                id: id.clone(),
                encoding: Some(encoding),
                label: entry.encoding_label.clone(),
            });
        }
    }
    let meta = WeightMeta {
        id: id.clone(),
        shape: entry.shape.clone(),
        encoding,
        on_disk_bytes: entry.size,
        transpose: TransposePolicy::Linear2D,
        transcode: None,
    };
    Ok(match ring {
        Some(r) => residency.register_in_ring(meta, r)?,
        None => residency.register(meta),
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
    register_linear_opt_ring(residency, id, None)
}

fn register_linear_opt_ring<S: WeightSource>(
    residency: &WeightResidency<S>,
    id: &WeightId,
    ring: Option<RingId>,
) -> Result<WeightHandle, LoadError> {
    register_linear_transcode(residency, id, ring, None)
}

fn register_linear_transcode<S: WeightSource>(
    residency: &WeightResidency<S>,
    id: &WeightId,
    ring: Option<RingId>,
    transcode: Option<thinfer_core::quant::QuantKind>,
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
    // Linear weight tensors: bf16 (transposed to [K, N]; or requantized to
    // the GGUF block layout when `transcode` is set) or GGUF quant (already
    // block-major [N, K], no transpose). fp16/i8/i4 not supported here yet.
    let (encoding, transpose, transcode) = match encoding {
        StorageEncoding::Bf16 if transcode.is_some() => {
            // Quant block layout is [N, K] N-major: keep the file's row
            // order, no transpose. K must be whole 32-elem blocks.
            assert_eq!(entry.shape.0.len(), 2, "transcode target must be 2-D");
            assert_eq!(
                entry.shape.0[1] % 32,
                0,
                "transcode requires K % 32 == 0 ({id:?})"
            );
            (encoding, TransposePolicy::None, transcode)
        }
        StorageEncoding::Bf16 | StorageEncoding::F32 => (encoding, TransposePolicy::Linear2D, None),
        StorageEncoding::Quant(_) => (encoding, TransposePolicy::None, None),
        _ => {
            return Err(LoadError::Undecodable {
                id: id.clone(),
                encoding: Some(encoding),
                label: entry.encoding_label.clone(),
            });
        }
    };
    let meta = WeightMeta {
        id: id.clone(),
        shape: entry.shape.clone(),
        encoding,
        on_disk_bytes: entry.size,
        transpose,
        transcode,
    };
    Ok(match ring {
        Some(r) => residency.register_in_ring(meta, r)?,
        None => residency.register(meta),
    })
}

pub(crate) fn register_passthrough<S: WeightSource>(
    residency: &WeightResidency<S>,
    id: &WeightId,
) -> Result<WeightHandle, LoadError> {
    register_passthrough_opt_ring(residency, id, None)
}

fn register_passthrough_opt_ring<S: WeightSource>(
    residency: &WeightResidency<S>,
    id: &WeightId,
    ring: Option<RingId>,
) -> Result<WeightHandle, LoadError> {
    register_one(residency, id, TransposePolicy::None, ring)
}

pub(crate) fn register_one<S: WeightSource>(
    residency: &WeightResidency<S>,
    id: &WeightId,
    transpose: TransposePolicy,
    ring: Option<RingId>,
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
    let meta = WeightMeta {
        id: id.clone(),
        shape: entry.shape.clone(),
        encoding,
        on_disk_bytes: entry.size,
        transpose,
        transcode: None,
    };
    Ok(match ring {
        Some(r) => residency.register_in_ring(meta, r)?,
        None => residency.register(meta),
    })
}

impl<SE: core::fmt::Debug, BE: core::fmt::Debug> From<ResidencyError<SE, BE>> for LoadError {
    fn from(_: ResidencyError<SE, BE>) -> Self {
        // Registration is sync and infallible at the residency layer; this is
        // unreachable but the trait makes call sites cleaner.
        unreachable!("register doesn't fail")
    }
}

impl From<RegisterRingError> for LoadError {
    fn from(e: RegisterRingError) -> Self {
        // `GrowthAfterAlloc` is structurally impossible from this loader:
        // every `register_in_ring` call runs before any inference (no
        // acquire happens during registration). Treat as a programmer
        // bug rather than a typed variant the loader has to thread.
        panic!("ring registration error: {e:?}");
    }
}
