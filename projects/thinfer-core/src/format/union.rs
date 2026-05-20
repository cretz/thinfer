//! Compose two `WeightSource`s into one. Tensors that exist in both are
//! resolved by the priority chosen at construction (the "primary"). The
//! merged catalog reflects that priority.
//!
//! Use case: stitch a quantized GGUF source for matmul tensors on top of a
//! safetensors source for norms/biases. The GGUF source is primary so its
//! Q8_0 entries shadow the bf16 entries with the same `WeightId`.

use crate::tensor::{Shape, StorageEncoding};
use crate::weight::{WeightCatalog, WeightEntry, WeightId, WeightReader, WeightSource};
use std::collections::HashMap;

/// Hides any catalog entry whose encoding isn't `StorageEncoding::Quant`,
/// plus any entry whose id contains one of `excluded_substrings`. Use
/// over a GGUF source so a `UnionSource(gguf, safetensors)` falls through
/// to safetensors for non-quant tensors (norms, biases) AND for tensors
/// the engine intentionally keeps unquantized even when the GGUF file
/// quantized them (e.g. AdaLN modulation weights, which the Z-Image
/// engine reads as bf16).
pub struct QuantOnlySource<S: WeightSource> {
    inner: S,
    catalog: WeightCatalog,
}

impl<S: WeightSource> QuantOnlySource<S> {
    /// Hide only non-Quant entries. Convenience for callers that don't
    /// need name-based exclusion.
    pub fn new(inner: S) -> Self {
        Self::with_excluded_substrings(inner, &[])
    }

    /// Hide non-Quant entries plus any id whose string contains one of
    /// `excluded_substrings`. The name filter applies even to Quant
    /// entries — that's the whole point (Q8 GGUFs from unsloth quantize
    /// AdaLN tensors that the engine still wants as bf16).
    pub fn with_excluded_substrings(inner: S, excluded_substrings: &[&str]) -> Self {
        let mut catalog = WeightCatalog::new();
        for (id, entry) in &inner.catalog().entries {
            if !matches!(entry.encoding, Some(StorageEncoding::Quant(_))) {
                continue;
            }
            if excluded_substrings.iter().any(|s| id.0.contains(s)) {
                continue;
            }
            catalog.entries.insert(id.clone(), entry.clone());
        }
        Self { inner, catalog }
    }

    /// Hide non-Quant entries, then keep only those whose id contains one
    /// of `allowed_substrings`. The engine-side equivalent of "only these
    /// specific weight roles are expected to come from the Quant source —
    /// everything else falls through". Use this when the upstream GGUF
    /// quantizes more tensors than the engine treats as quantized.
    pub fn with_allowed_substrings(inner: S, allowed_substrings: &[&str]) -> Self {
        let mut catalog = WeightCatalog::new();
        for (id, entry) in &inner.catalog().entries {
            if !matches!(entry.encoding, Some(StorageEncoding::Quant(_))) {
                continue;
            }
            if !allowed_substrings.iter().any(|s| id.0.contains(s)) {
                continue;
            }
            catalog.entries.insert(id.clone(), entry.clone());
        }
        Self { inner, catalog }
    }
}

impl<S: WeightSource> WeightSource for QuantOnlySource<S> {
    type Reader = S::Reader;
    type Error = S::Error;
    fn catalog(&self) -> &WeightCatalog {
        &self.catalog
    }
    async fn open(&self, id: &WeightId) -> Result<Self::Reader, Self::Error> {
        self.inner.open(id).await
    }
}

/// Re-keys another source's catalog. Entries appear under their renamed
/// id; `open(renamed)` resolves back to the underlying id. Tensors not in
/// `rename` are dropped from the catalog (this is the explicit-allowlist
/// posture: only the keys we know how to translate are surfaced).
pub struct RenamedSource<S: WeightSource> {
    inner: S,
    /// renamed -> original.
    back: HashMap<WeightId, WeightId>,
    catalog: WeightCatalog,
}

impl<S: WeightSource> RenamedSource<S> {
    /// `rename: original -> renamed`. Original ids that don't exist in the
    /// inner catalog are silently dropped (allows passing a superset map
    /// that covers more tensors than this particular file contains).
    pub fn new(inner: S, rename: HashMap<WeightId, WeightId>) -> Self {
        let mut catalog = WeightCatalog::new();
        let mut back: HashMap<WeightId, WeightId> = HashMap::new();
        for (orig, renamed) in &rename {
            if let Some(entry) = inner.catalog().get(orig) {
                catalog.entries.insert(renamed.clone(), entry.clone());
                back.insert(renamed.clone(), orig.clone());
            }
        }
        Self {
            inner,
            back,
            catalog,
        }
    }

    /// Same as [`Self::new`] but every inner-catalog entry NOT in `rename`
    /// is republished under its original id (passthrough). Use when the
    /// caller wants a thin adapter that re-keys a few names while leaving
    /// the rest of the source intact.
    pub fn with_passthrough(inner: S, rename: HashMap<WeightId, WeightId>) -> Self {
        let mut catalog = WeightCatalog::new();
        let mut back: HashMap<WeightId, WeightId> = HashMap::new();
        for (orig, renamed) in &rename {
            if let Some(entry) = inner.catalog().get(orig) {
                catalog.entries.insert(renamed.clone(), entry.clone());
                back.insert(renamed.clone(), orig.clone());
            }
        }
        for (id, entry) in &inner.catalog().entries {
            if !rename.contains_key(id) {
                catalog.entries.insert(id.clone(), entry.clone());
            }
        }
        Self {
            inner,
            back,
            catalog,
        }
    }
}

impl<S: WeightSource> WeightSource for RenamedSource<S> {
    type Reader = S::Reader;
    type Error = S::Error;
    fn catalog(&self) -> &WeightCatalog {
        &self.catalog
    }
    async fn open(&self, id: &WeightId) -> Result<Self::Reader, Self::Error> {
        let orig = self.back.get(id).unwrap_or(id);
        self.inner.open(orig).await
    }
}

/// Wraps two sources. `primary` wins on `WeightId` collisions. The
/// catalog is rebuilt at construction from both sources' catalogs.
pub struct UnionSource<P: WeightSource, F: WeightSource> {
    primary: P,
    fallback: F,
    catalog: WeightCatalog,
    /// `true` means the entry came from `primary`. Routes `open()`.
    in_primary: HashMap<WeightId, bool>,
}

impl<P: WeightSource, F: WeightSource> UnionSource<P, F> {
    pub fn new(primary: P, fallback: F) -> Self {
        let mut catalog = WeightCatalog::new();
        let mut in_primary: HashMap<WeightId, bool> = HashMap::new();
        for (id, entry) in &fallback.catalog().entries {
            catalog.entries.insert(id.clone(), entry.clone());
            in_primary.insert(id.clone(), false);
        }
        for (id, entry) in &primary.catalog().entries {
            catalog.entries.insert(id.clone(), entry.clone());
            in_primary.insert(id.clone(), true);
        }
        Self {
            primary,
            fallback,
            catalog,
            in_primary,
        }
    }

    pub fn primary(&self) -> &P {
        &self.primary
    }
    pub fn fallback(&self) -> &F {
        &self.fallback
    }
}

/// Reader enum: dispatches `read_at` to the underlying source's reader.
pub enum UnionReader<A: WeightReader, B: WeightReader> {
    Primary(A),
    Fallback(B),
}

impl<A: WeightReader, B: WeightReader> WeightReader for UnionReader<A, B> {
    type Error = UnionError<A::Error, B::Error>;
    fn len(&self) -> u64 {
        match self {
            Self::Primary(a) => a.len(),
            Self::Fallback(b) => b.len(),
        }
    }
    async fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), Self::Error> {
        match self {
            Self::Primary(a) => a.read_at(offset, dst).await.map_err(UnionError::Primary),
            Self::Fallback(b) => b.read_at(offset, dst).await.map_err(UnionError::Fallback),
        }
    }
}

#[derive(Debug)]
pub enum UnionError<A: core::fmt::Debug, B: core::fmt::Debug> {
    Primary(A),
    Fallback(B),
    /// Tensor not in either catalog. Distinct from each source's own
    /// `UnknownTensor` since we want callers to know neither side had it.
    UnknownTensor(WeightId),
}

impl<P: WeightSource, F: WeightSource> WeightSource for UnionSource<P, F> {
    type Reader = UnionReader<P::Reader, F::Reader>;
    type Error = UnionError<P::Error, F::Error>;

    fn catalog(&self) -> &WeightCatalog {
        &self.catalog
    }

    async fn open(&self, id: &WeightId) -> Result<Self::Reader, Self::Error> {
        match self.in_primary.get(id) {
            Some(true) => self
                .primary
                .open(id)
                .await
                .map(UnionReader::Primary)
                .map_err(UnionError::Primary),
            Some(false) => self
                .fallback
                .open(id)
                .await
                .map(UnionReader::Fallback)
                .map_err(UnionError::Fallback),
            None => Err(UnionError::UnknownTensor(id.clone())),
        }
    }
}

/// Adapter that lazily presents canonical fused QKV tensors over a split-QKV
/// underlying source (e.g. dimitribarbot's safetensors checkpoint that ships
/// `attention.to_q.weight`, `to_k.weight`, `to_v.weight`). For each configured
/// triple, exposes one fused `attention.qkv.weight` entry whose bytes are the
/// row-concatenation of the three sub-tensors along the N axis (`[3*H, K]`
/// row-major: q rows, then k rows, then v rows).
///
/// All non-fused keys passthrough to the inner source. Triples are configured
/// at construction; tensors not in any triple keep their original
/// catalog entry.
pub struct SplitToFusedQkvSource<S: WeightSource> {
    inner: S,
    catalog: WeightCatalog,
    /// Fused id -> (q_id, k_id, v_id, q_size, k_size).
    /// Sizes (in bytes) are needed at `open()` time so the reader can route
    /// offsets to the right sub-tensor without re-reading the catalog.
    fused: HashMap<WeightId, FusedSpec>,
}

#[derive(Clone, Debug)]
struct FusedSpec {
    q: WeightId,
    k: WeightId,
    v: WeightId,
    q_size: u64,
    k_size: u64,
    v_size: u64,
}

/// One QKV triple to fuse: `(fused_id, q_id, k_id, v_id)`. Caller assembles
/// the list (typically per-block, per-prefix).
pub struct QkvTriple {
    pub fused: WeightId,
    pub q: WeightId,
    pub k: WeightId,
    pub v: WeightId,
}

impl<S: WeightSource> SplitToFusedQkvSource<S> {
    /// Triples whose sub-tensors are not all present in the inner catalog are
    /// silently dropped (the caller's triple list may be a superset covering
    /// schemas the current file does not use).
    pub fn new(inner: S, triples: Vec<QkvTriple>) -> Self {
        let mut catalog = WeightCatalog::new();
        for (id, entry) in &inner.catalog().entries {
            catalog.entries.insert(id.clone(), entry.clone());
        }
        let mut fused: HashMap<WeightId, FusedSpec> = HashMap::new();
        for t in triples {
            let (Some(eq), Some(ek), Some(ev)) = (
                inner.catalog().get(&t.q),
                inner.catalog().get(&t.k),
                inner.catalog().get(&t.v),
            ) else {
                continue;
            };
            // Shape sanity: split tensors are [H, K] each (PyTorch
            // [out, in]); fused is [3H, K]. K must match across q/k/v.
            assert_eq!(
                eq.shape.0.len(),
                2,
                "split QKV adapter: q tensor must be 2-D, got {:?}",
                eq.shape.0
            );
            assert_eq!(
                eq.shape.0[1], ek.shape.0[1],
                "split QKV adapter: q/k K mismatch"
            );
            assert_eq!(
                eq.shape.0[1], ev.shape.0[1],
                "split QKV adapter: q/v K mismatch"
            );
            assert_eq!(
                eq.encoding, ek.encoding,
                "split QKV adapter: q/k encoding mismatch"
            );
            assert_eq!(
                eq.encoding, ev.encoding,
                "split QKV adapter: q/v encoding mismatch"
            );
            let fused_n = eq.shape.0[0] + ek.shape.0[0] + ev.shape.0[0];
            let fused_shape = Shape(vec![fused_n, eq.shape.0[1]]);
            // The fused entry's offset is ignored (the adapter routes opens
            // through `fused`); size is the byte sum so `WeightEntry::size`
            // stays consistent for catalog consumers.
            catalog.entries.insert(
                t.fused.clone(),
                WeightEntry {
                    offset: 0,
                    size: eq.size + ek.size + ev.size,
                    encoding: eq.encoding,
                    encoding_label: eq.encoding_label.clone(),
                    shape: fused_shape,
                },
            );
            // Drop the three split entries from the catalog: the engine will
            // never ask for them directly, and surfacing both fused + split
            // would inflate the audit's `extra` list.
            catalog.entries.remove(&t.q);
            catalog.entries.remove(&t.k);
            catalog.entries.remove(&t.v);
            fused.insert(
                t.fused,
                FusedSpec {
                    q: t.q,
                    k: t.k,
                    v: t.v,
                    q_size: eq.size,
                    k_size: ek.size,
                    v_size: ev.size,
                },
            );
        }
        Self {
            inner,
            catalog,
            fused,
        }
    }
}

impl<S: WeightSource> WeightSource for SplitToFusedQkvSource<S> {
    type Reader = SplitToFusedQkvReader<S>;
    type Error = S::Error;

    fn catalog(&self) -> &WeightCatalog {
        &self.catalog
    }

    async fn open(&self, id: &WeightId) -> Result<Self::Reader, Self::Error> {
        match self.fused.get(id) {
            Some(spec) => {
                let q = self.inner.open(&spec.q).await?;
                let k = self.inner.open(&spec.k).await?;
                let v = self.inner.open(&spec.v).await?;
                Ok(SplitToFusedQkvReader::Fused(FusedReader {
                    q,
                    k,
                    v,
                    q_size: spec.q_size,
                    k_size: spec.k_size,
                    v_size: spec.v_size,
                }))
            }
            None => self.inner.open(id).await.map(SplitToFusedQkvReader::Pass),
        }
    }
}

/// Reader returned by `SplitToFusedQkvSource::open`. `Pass` is a transparent
/// inner reader; `Fused` holds three sub-readers opened eagerly and stitches
/// `read_at` ranges across slab boundaries.
pub enum SplitToFusedQkvReader<S: WeightSource> {
    Pass(S::Reader),
    Fused(FusedReader<S>),
}

pub struct FusedReader<S: WeightSource> {
    q: S::Reader,
    k: S::Reader,
    v: S::Reader,
    q_size: u64,
    k_size: u64,
    v_size: u64,
}

impl<S: WeightSource> WeightReader for SplitToFusedQkvReader<S> {
    type Error = <S::Reader as WeightReader>::Error;

    fn len(&self) -> u64 {
        match self {
            Self::Pass(r) => r.len(),
            Self::Fused(f) => f.q_size + f.k_size + f.v_size,
        }
    }

    async fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), Self::Error> {
        match self {
            Self::Pass(r) => r.read_at(offset, dst).await,
            Self::Fused(f) => f.read_at(offset, dst).await,
        }
    }
}

impl<S: WeightSource> FusedReader<S> {
    async fn read_at(
        &mut self,
        mut offset: u64,
        mut dst: &mut [u8],
    ) -> Result<(), <S::Reader as WeightReader>::Error> {
        let q_end = self.q_size;
        let k_end = q_end + self.k_size;
        let v_end = k_end + self.v_size;
        while !dst.is_empty() {
            if offset >= v_end {
                // Past end-of-tensor: residency layer must not request this.
                // Fail loud rather than silently zero-filling.
                panic!("SplitToFusedQkvSource::read_at: offset {offset} past tensor end {v_end}");
            }
            let (take, sub_offset) = if offset < q_end {
                (((q_end - offset) as usize).min(dst.len()), offset)
            } else if offset < k_end {
                (((k_end - offset) as usize).min(dst.len()), offset - q_end)
            } else {
                (((v_end - offset) as usize).min(dst.len()), offset - k_end)
            };
            let (head, tail) = dst.split_at_mut(take);
            if offset < q_end {
                self.q.read_at(sub_offset, head).await?;
            } else if offset < k_end {
                self.k.read_at(sub_offset, head).await?;
            } else {
                self.v.read_at(sub_offset, head).await?;
            }
            offset += take as u64;
            dst = tail;
        }
        Ok(())
    }
}
