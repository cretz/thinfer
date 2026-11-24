//! `torch.nn.utils.weight_norm` pairs served as plain weights.
//!
//! A weight-normed layer stores `weight_g` (`[d0, 1, ..]`) and `weight_v`
//! (`[d0, ..]`) for `weight = g * v / ||v||`, the norm taken per index of
//! dim 0 over every other dim (`dim=0`, the default: output channels of a
//! Conv1d, INPUT channels of a ConvTranspose1d, whose weight is `[in, out,
//! k]`). The adapter rewrites the catalog: each `X.weight_g` + `X.weight_v`
//! pair becomes one f32 `X.weight` folded on read; everything else passes
//! through.

use std::collections::HashMap;

use thinfer_core::tensor::StorageEncoding;
use thinfer_core::weight::{WeightCatalog, WeightId, WeightReader, WeightSource};

use crate::common::lora::VecReader;

pub struct WeightNormSource<S: WeightSource> {
    base: S,
    catalog: WeightCatalog,
    /// Folded weight -> its `(g, v)` pair.
    pairs: HashMap<WeightId, (WeightId, WeightId)>,
}

#[derive(Debug)]
pub enum WeightNormError<E: core::fmt::Debug> {
    Base(E),
    Read(String),
}

pub enum WeightNormReader<S: WeightSource> {
    Base(S::Reader),
    Folded(VecReader),
}

impl<S: WeightSource> WeightReader for WeightNormReader<S> {
    type Error = WeightNormError<<S::Reader as WeightReader>::Error>;
    fn len(&self) -> u64 {
        match self {
            Self::Base(r) => r.len(),
            Self::Folded(r) => r.len(),
        }
    }
    async fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), Self::Error> {
        match self {
            Self::Base(r) => r.read_at(offset, dst).await.map_err(WeightNormError::Base),
            Self::Folded(r) => {
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

impl<S: WeightSource> WeightNormSource<S> {
    pub fn new(base: S) -> Self {
        let mut catalog = WeightCatalog::new();
        let mut pairs = HashMap::new();
        for (id, entry) in &base.catalog().entries {
            if let Some(stem) = id.0.strip_suffix(".weight_v") {
                let g = WeightId(format!("{stem}.weight_g"));
                if base.catalog().get(&g).is_some() {
                    let mut e = entry.clone();
                    let elems: u64 = e.shape.0.iter().map(|&d| d as u64).product();
                    e.encoding = Some(StorageEncoding::F32);
                    e.encoding_label = "F32(weight_norm)".into();
                    e.size = elems * 4;
                    let w = WeightId(format!("{stem}.weight"));
                    catalog.entries.insert(w.clone(), e);
                    pairs.insert(w, (g, id.clone()));
                    continue;
                }
            }
            let is_g_of_pair = id.0.strip_suffix(".weight_g").is_some_and(|stem| {
                base.catalog()
                    .get(&WeightId(format!("{stem}.weight_v")))
                    .is_some()
            });
            if !is_g_of_pair {
                catalog.entries.insert(id.clone(), entry.clone());
            }
        }
        Self {
            base,
            catalog,
            pairs,
        }
    }

    async fn read_f32(&self, id: &WeightId) -> Result<Vec<f32>, WeightNormError<S::Error>> {
        let entry = self
            .base
            .catalog()
            .get(id)
            .ok_or_else(|| WeightNormError::Read(format!("missing {}", id.0)))?;
        if entry.encoding != Some(StorageEncoding::F32) {
            return Err(WeightNormError::Read(format!(
                "{}: weight_norm pairs must be f32, got {}",
                id.0, entry.encoding_label
            )));
        }
        let mut r = self.base.open(id).await.map_err(WeightNormError::Base)?;
        let mut bytes = vec![0u8; r.len() as usize];
        r.read_at(0, &mut bytes)
            .await
            .map_err(|e| WeightNormError::Read(format!("{}: {e:?}", id.0)))?;
        Ok(bytemuck::pod_collect_to_vec(&bytes))
    }
}

/// `g[i] * v[i, ..] / ||v[i, ..]||` per index `i` of dim 0 (f64 norm).
pub fn fold(g: &[f32], v: &[f32]) -> Vec<f32> {
    let per = v.len() / g.len();
    let mut w = Vec::with_capacity(v.len());
    for (gi, row) in g.iter().zip(v.chunks_exact(per)) {
        let norm = row.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
        let s = *gi as f64 / norm;
        w.extend(row.iter().map(|&x| (x as f64 * s) as f32));
    }
    w
}

impl<S: WeightSource> WeightSource for WeightNormSource<S> {
    type Reader = WeightNormReader<S>;
    type Error = WeightNormError<S::Error>;

    fn catalog(&self) -> &WeightCatalog {
        &self.catalog
    }

    async fn open(&self, id: &WeightId) -> Result<Self::Reader, Self::Error> {
        let Some((g, v)) = self.pairs.get(id) else {
            let r = self.base.open(id).await.map_err(WeightNormError::Base)?;
            return Ok(WeightNormReader::Base(r));
        };
        let g = self.read_f32(g).await?;
        let v = self.read_f32(v).await?;
        if g.is_empty() || !v.len().is_multiple_of(g.len()) {
            return Err(WeightNormError::Read(format!(
                "{}: g {} does not divide v {}",
                id.0,
                g.len(),
                v.len()
            )));
        }
        let w = super::seq::off_thread(move || fold(&g, &v)).await;
        Ok(WeightNormReader::Folded(VecReader::from_vec(
            bytemuck::cast_slice(&w).to_vec(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each dim-0 slice is rescaled to norm `g`.
    #[test]
    fn folds_per_leading_index() {
        let g = [2.0f32, 0.5];
        let v = [3.0f32, 4.0, 0.0, -1.0];
        let w = fold(&g, &v);
        assert_eq!(w, vec![1.2, 1.6, 0.0, -0.5]);
    }
}
