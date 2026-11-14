//! ComfyUI `int8_tensorwise` + `convrot` linears served as GGUF Q8_0.
//!
//! A convrot linear stores `Wq` int8 `[N, K]` plus `weight_scale` f32 `[N, 1]`
//! for the ROTATED weight `W_rot = W @ H_blk^T`, `H_blk` the block-diagonal
//! normalized regular Hadamard of group 256 (comfy-kitchen `_rotate_weight`).
//! `H` is symmetric and orthogonal, so `x @ W^T = (x @ H_blk) @ W_rot^T`: the
//! caller rotates the activation per 256 group (`ops::hadamard`) and the
//! weight runs as-is. `s * Wq` per row is exactly a Q8_0 stream whose every
//! 32-block carries the row scale as its f16 `d` (the int8 codes are copied
//! unchanged; the only loss is the f32 -> f16 scale rounding, <= 2^-11
//! relative per row, the scales being normal f16), so the existing Q8_0 matmul
//! paths serve it.
//!
//! The adapter rewrites the catalog: every `X.weight` I8 tensor with an
//! `X.weight_scale` sibling becomes a Q8_0 entry; everything else passes
//! through.

use std::collections::HashSet;
use std::sync::Arc;

use half::f16;
use thinfer_core::quant::QuantKind;
use thinfer_core::tensor::StorageEncoding;
use thinfer_core::weight::{WeightCatalog, WeightId, WeightReader, WeightSource};

use crate::common::lora::VecReader;

/// Hadamard group of every convrot checkpoint we load (`convrot_groupsize`).
pub const CONVROT_GROUP: usize = 256;

pub struct ConvRotSource<S: WeightSource> {
    base: S,
    catalog: WeightCatalog,
    /// Weight ids served as repacked Q8_0.
    convrot: HashSet<WeightId>,
}

#[derive(Debug)]
pub enum ConvRotError<E: core::fmt::Debug> {
    Base(E),
    Read(String),
}

pub enum ConvRotReader<S: WeightSource> {
    Base(S::Reader),
    Packed(VecReader),
}

impl<S: WeightSource> WeightReader for ConvRotReader<S> {
    type Error = ConvRotError<<S::Reader as WeightReader>::Error>;
    fn len(&self) -> u64 {
        match self {
            Self::Base(r) => r.len(),
            Self::Packed(r) => r.len(),
        }
    }
    async fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), Self::Error> {
        match self {
            Self::Base(r) => r.read_at(offset, dst).await.map_err(ConvRotError::Base),
            Self::Packed(r) => {
                let Ok(()) = r.read_at(offset, dst).await;
                Ok(())
            }
        }
    }
    fn will_read(&mut self, offset: u64, len: u64) {
        if let Self::Base(r) = self {
            r.will_read(offset, len);
        }
    }
}

fn scale_id(weight: &WeightId) -> Option<WeightId> {
    weight
        .0
        .strip_suffix(".weight")
        .map(|stem| WeightId(format!("{stem}.weight_scale")))
}

impl<S: WeightSource> ConvRotSource<S> {
    pub fn new(base: S) -> Self {
        let mut catalog = WeightCatalog::new();
        let mut convrot = HashSet::new();
        for (id, entry) in &base.catalog().entries {
            let mut entry = entry.clone();
            let is_convrot = entry.encoding == Some(StorageEncoding::I8)
                && entry.shape.0.len() == 2
                && entry.shape.0[1] % CONVROT_GROUP == 0
                && scale_id(id).is_some_and(|s| base.catalog().get(&s).is_some());
            if is_convrot {
                let elems: u64 = entry.shape.0.iter().map(|&d| d as u64).product();
                entry.encoding = Some(StorageEncoding::Quant(QuantKind::Q8_0));
                entry.encoding_label = "Q8_0(convrot)".into();
                entry.size = QuantKind::Q8_0.bytes_for_elements(elems);
                convrot.insert(id.clone());
            }
            catalog.entries.insert(id.clone(), entry);
        }
        Self {
            base,
            catalog,
            convrot,
        }
    }

    /// The convrot linears (their inputs must be rotated), shareable after the
    /// source moves into a residency.
    pub fn convrot_ids(&self) -> ConvRotIds {
        Arc::new(self.convrot.clone())
    }

    async fn read_all(&self, id: &WeightId) -> Result<Vec<u8>, ConvRotError<S::Error>> {
        let mut r = self.base.open(id).await.map_err(ConvRotError::Base)?;
        let mut bytes = vec![0u8; r.len() as usize];
        r.read_at(0, &mut bytes)
            .await
            .map_err(|e| ConvRotError::Read(format!("{id:?}: {e:?}")))?;
        Ok(bytes)
    }
}

/// Repack int8 `[n, k]` + per-row f32 scales into Q8_0 blocks (`d` = row scale).
pub fn pack_q8_0(codes: &[u8], scales: &[f32], k: usize) -> Vec<u8> {
    const BLOCK: usize = 32;
    const BLOCK_BYTES: usize = 34;
    let row_bytes = k / BLOCK * BLOCK_BYTES;
    let mut out = vec![0u8; scales.len() * row_bytes];
    let pack_rows = |codes: &[u8], scales: &[f32], out: &mut [u8]| {
        for ((src, &s), dst) in codes.chunks(k).zip(scales).zip(out.chunks_mut(row_bytes)) {
            let d = f16::from_f32(s).to_le_bytes();
            for (sb, db) in src.chunks(BLOCK).zip(dst.chunks_mut(BLOCK_BYTES)) {
                db[..2].copy_from_slice(&d);
                db[2..].copy_from_slice(sb);
            }
        }
    };
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(8);
    let rows_per = scales.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for ((c, sc), o) in codes
            .chunks(rows_per * k)
            .zip(scales.chunks(rows_per))
            .zip(out.chunks_mut(rows_per * row_bytes))
        {
            s.spawn(move || pack_rows(c, sc, o));
        }
    });
    out
}

impl<S: WeightSource> WeightSource for ConvRotSource<S> {
    type Reader = ConvRotReader<S>;
    type Error = ConvRotError<S::Error>;

    fn catalog(&self) -> &WeightCatalog {
        &self.catalog
    }

    async fn open(&self, id: &WeightId) -> Result<Self::Reader, Self::Error> {
        if !self.convrot.contains(id) {
            let r = self.base.open(id).await.map_err(ConvRotError::Base)?;
            return Ok(ConvRotReader::Base(r));
        }
        let k = self.catalog.entries[id].shape.0[1];
        let codes = self.read_all(id).await?;
        let sid = scale_id(id).expect("convrot ids end in .weight");
        let scale_bytes = self.read_all(&sid).await?;
        let scales: Vec<f32> = scale_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        // The repack is ~0.2 GB of byte shuffling per block at RelayVSR's 5B:
        // off the engine thread, so it overlaps the GPU running the block.
        let packed = super::seq::off_thread(move || pack_q8_0(&codes, &scales, k)).await;
        Ok(ConvRotReader::Packed(VecReader::from_vec(packed)))
    }
}

/// Which linears of a [`ConvRotSource`] need rotated inputs.
pub type ConvRotIds = Arc<HashSet<WeightId>>;

#[cfg(test)]
mod tests {
    use super::*;

    /// Each 32-block of a row carries that row's scale as f16 and its codes.
    #[test]
    fn packs_row_scale_into_every_block() {
        let k = 64;
        let codes: Vec<u8> = (0..2 * k)
            .map(|i| (i as i8).wrapping_sub(60) as u8)
            .collect();
        let scales = [0.5f32, 0.25];
        let out = pack_q8_0(&codes, &scales, k);
        assert_eq!(out.len(), 2 * 2 * 34);
        for row in 0..2 {
            for b in 0..2 {
                let blk = &out[(row * 2 + b) * 34..][..34];
                assert_eq!(f16::from_le_bytes([blk[0], blk[1]]).to_f32(), scales[row]);
                assert_eq!(&blk[2..], &codes[row * k + b * 32..][..32]);
            }
        }
    }
}
