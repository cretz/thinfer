//! Union of two weight sources behind one [`WeightSource`].
//!
//! A model whose adapter is applied at RUNTIME (rather than folded into the
//! base tensors) needs the base weights and the adapter weights to share one
//! residency: two residencies would carry two independent VRAM budgets over one
//! device and each would size its evictions blind to the other.
//!
//! The overlay's ids win on collision, which is what makes this usable for
//! genuine overrides as well as disjoint unions.

use std::collections::HashSet;

use thinfer_core::weight::{WeightCatalog, WeightId, WeightReader, WeightSource};

pub struct MergedSource<A: WeightSource, B: WeightSource> {
    base: A,
    overlay: B,
    catalog: WeightCatalog,
    /// Ids served by `overlay`. Kept explicitly so a collision resolves the
    /// same way at `open` as it did when the catalog was built.
    from_overlay: HashSet<WeightId>,
}

impl<A: WeightSource, B: WeightSource> MergedSource<A, B> {
    pub fn new(base: A, overlay: B) -> Self {
        let mut catalog = WeightCatalog::new();
        for (k, v) in &base.catalog().entries {
            catalog.entries.insert(k.clone(), v.clone());
        }
        let mut from_overlay = HashSet::new();
        for (k, v) in &overlay.catalog().entries {
            catalog.entries.insert(k.clone(), v.clone());
            from_overlay.insert(k.clone());
        }
        Self {
            base,
            overlay,
            catalog,
            from_overlay,
        }
    }

    pub fn base(&self) -> &A {
        &self.base
    }

    pub fn overlay_len(&self) -> usize {
        self.from_overlay.len()
    }
}

#[derive(Debug)]
pub enum MergedError<AE: core::fmt::Debug, BE: core::fmt::Debug> {
    Base(AE),
    Overlay(BE),
}

pub enum MergedReader<A: WeightReader, B: WeightReader> {
    Base(A),
    Overlay(B),
}

impl<A: WeightReader, B: WeightReader> WeightReader for MergedReader<A, B> {
    type Error = MergedError<A::Error, B::Error>;

    fn len(&self) -> u64 {
        match self {
            Self::Base(r) => r.len(),
            Self::Overlay(r) => r.len(),
        }
    }

    async fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), Self::Error> {
        match self {
            Self::Base(r) => r.read_at(offset, dst).await.map_err(MergedError::Base),
            Self::Overlay(r) => r.read_at(offset, dst).await.map_err(MergedError::Overlay),
        }
    }

    fn will_read(&mut self, offset: u64, len: u64) {
        match self {
            Self::Base(r) => r.will_read(offset, len),
            Self::Overlay(r) => r.will_read(offset, len),
        }
    }
}

impl<A: WeightSource, B: WeightSource> WeightSource for MergedSource<A, B> {
    type Reader = MergedReader<A::Reader, B::Reader>;
    type Error = MergedError<A::Error, B::Error>;

    fn catalog(&self) -> &WeightCatalog {
        &self.catalog
    }

    async fn open(&self, id: &WeightId) -> Result<Self::Reader, Self::Error> {
        if self.from_overlay.contains(id) {
            self.overlay
                .open(id)
                .await
                .map(MergedReader::Overlay)
                .map_err(MergedError::Overlay)
        } else {
            self.base
                .open(id)
                .await
                .map(MergedReader::Base)
                .map_err(MergedError::Base)
        }
    }
}
