//! Residency-aware weight registration primitives shared across models. Each
//! helper looks a tensor up in the source's `WeightCatalog`, builds a
//! `WeightMeta` (with decode + transpose policy) and registers it with the
//! `WeightResidency` manager. No bytes flow here: decode + transpose + GPU
//! upload happen lazily on `WeightResidency::acquire`.
//!
//! Linear weight tensors get `TransposePolicy::Linear2D` (matmul convention is
//! `A @ B` with B in `[K, N]`; PyTorch ships `[N, K]`). RMSNorm gains, biases,
//! and pad tokens are 1-D and use `TransposePolicy::None`. Per-model handle
//! bundles and the walk over a model's typed weight ids live in the model's
//! own `loader` module.

use thinfer_core::residency::{
    RegisterRingError, ResidencyError, RingId, TransposePolicy, WeightHandle, WeightMeta,
    WeightResidency,
};
use thinfer_core::tensor::StorageEncoding;
use thinfer_core::weight::{WeightId, WeightSource};

#[derive(Debug)]
pub enum LoadError {
    UnknownWeight(WeightId),
    /// On-disk encoding can't be decoded into fp32 (quantized, or fp16 which
    /// we intentionally don't support for the dense path).
    Undecodable {
        id: WeightId,
        encoding: Option<StorageEncoding>,
        label: String,
    },
}

/// Dense-consumed linear: the matmul site reads bf16 `[K, N]` regardless of
/// file encoding. Quant files (GGUF checkpoints quantize the AdaLN
/// modulation weights) dequant to dense bf16 at upload (`encoding Quant +
/// TransposePolicy::Linear2D`, see `WeightMeta::gpu_encoding`); bf16/F32
/// files ride the plain `Linear2D` path. Never registers a quant GPU
/// layout: the adaln matmul pipeline compiles `WeightDtype::Bf16`
/// unconditionally (M=1 modulation matmul).
pub(crate) fn register_linear_dense_opt_ring<S: WeightSource>(
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

pub(crate) fn register_linear_transcode<S: WeightSource>(
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

pub(crate) fn register_passthrough_opt_ring<S: WeightSource>(
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
