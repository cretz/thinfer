//! Z-Image weight-source construction, shared by every host (CLI, web,
//! e2e tests). Owns the safetensors schema adapters and the GGUF-over-
//! safetensors union so the recipe exists exactly once.

use thinfer_core::format::gguf::{self, GgufSource};
use thinfer_core::format::safetensors::{self, ShardedSafetensorsSource};
use thinfer_core::format::union::{
    RenamedSource, SplitToFusedQkvSource, UnionError, UnionReader, UnionSource,
};
use thinfer_core::weight::{FileOpener, WeightCatalog, WeightId, WeightSource};

/// Safetensors side: sharded files plus schema adapters. Z-Image canonical
/// schema is fused `attention.qkv.weight` / `attention.out.weight`;
/// checkpoints that ship split `to_q`/`to_k`/`to_v` or `to_out.0`
/// (dimitribarbot) flow through the adapters, canonical checkpoints see
/// passthrough.
pub type SafetensorsSide<O> = RenamedSource<SplitToFusedQkvSource<ShardedSafetensorsSource<O>>>;
/// One GGUF with its naming divergences mapped to canonical ids. Two
/// instances on quant variants: the COMPLETE DiT (quantized matmuls +
/// AdaLN as-tagged, bf16 refiners/embedders, F32 norms/biases) and the
/// Qwen3 text encoder (Q8_0 matmuls + token_embd, F32 norms). Residency
/// decodes every encoding at upload, so nothing is filtered.
type GgufSide<O> = RenamedSource<GgufSource<O>>;
/// Quant-variant source: TE GGUF over DiT GGUF over the safetensors side
/// (which supplies only the VAE). Tensor namespaces are disjoint, so the
/// union order is just lookup order.
type QuantSide<O> = UnionSource<GgufSide<O>, UnionSource<GgufSide<O>, SafetensorsSide<O>>>;

/// The two GGUF openers a quant variant loads from. Both present exactly
/// when the variant has `dit_gguf_role` + `te_gguf_role` set.
pub struct GgufOpeners<O> {
    pub dit: O,
    pub te: O,
}

/// The one weight source `ZImageModel` loads from. `Plain` for bf16
/// safetensors variants, `Quantized` when the variant takes the DiT and
/// text encoder from GGUFs unioned over the safetensors source that
/// supplies the VAE (quant variants ship no DiT/TE safetensors at all).
// Both arms boxed: catalogs + rename maps make either source hundreds of
// bytes (clippy::large_enum_variant); one model-lifetime alloc each.
pub enum ZImageSource<O: FileOpener> {
    Plain(Box<SafetensorsSide<O>>),
    Quantized(Box<QuantSide<O>>),
}

#[derive(Debug)]
pub enum SourceOpenError<E: core::fmt::Debug> {
    Safetensors(safetensors::SourceError<E>),
    Gguf(gguf::SourceError<E>),
}

impl<O: FileOpener> ZImageSource<O> {
    /// Parse and adapt a variant's weight files. `weight_openers` are the
    /// safetensors shards in `VariantFiles::weight_roles` order;
    /// `gguf_openers` is `Some` exactly when the variant has GGUF roles.
    pub async fn open(
        weight_openers: Vec<O>,
        gguf_openers: Option<GgufOpeners<O>>,
    ) -> Result<Self, SourceOpenError<O::Error>> {
        let sharded = ShardedSafetensorsSource::open(weight_openers)
            .await
            .map_err(SourceOpenError::Safetensors)?;
        let fused = SplitToFusedQkvSource::new(sharded, crate::z_image::dit_qkv_triples());
        let st = RenamedSource::with_passthrough(fused, crate::z_image::dit_to_out_renames());
        Ok(match gguf_openers {
            None => Self::Plain(Box::new(st)),
            Some(g) => {
                let dit = GgufSource::open(g.dit)
                    .await
                    .map_err(SourceOpenError::Gguf)?;
                let te = GgufSource::open(g.te)
                    .await
                    .map_err(SourceOpenError::Gguf)?;
                Self::Quantized(Box::new(UnionSource::new(
                    RenamedSource::with_passthrough(te, crate::z_image::qwen3_gguf_renames()),
                    UnionSource::new(
                        RenamedSource::with_passthrough(dit, crate::z_image::dit_gguf_renames()),
                        st,
                    ),
                )))
            }
        })
    }
}

impl<O: FileOpener> WeightSource for ZImageSource<O> {
    // Plain readers ride the union's nested `Fallback` arms so both
    // variants share one reader/error type and callers stay monomorphic
    // over the enum.
    type Reader = <QuantSide<O> as WeightSource>::Reader;
    type Error = <QuantSide<O> as WeightSource>::Error;

    fn catalog(&self) -> &WeightCatalog {
        match self {
            Self::Plain(s) => s.catalog(),
            Self::Quantized(s) => s.catalog(),
        }
    }

    async fn open(&self, id: &WeightId) -> Result<Self::Reader, Self::Error> {
        match self {
            Self::Plain(s) => s
                .open(id)
                .await
                .map(|r| UnionReader::Fallback(UnionReader::Fallback(r)))
                .map_err(|e| UnionError::Fallback(UnionError::Fallback(e))),
            Self::Quantized(s) => s.open(id).await,
        }
    }
}
