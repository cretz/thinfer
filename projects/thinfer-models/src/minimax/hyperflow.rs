//! HyperFlow H3 Flash (`VideoRebirthAI/hyperflow-h3-flash`): MiniMax-H3
//! compressed to 47 blocks (dropped blocks removed and renumbered in the GGUF),
//! with low-rank factored linears and two-time `(t, r)` conditioning, sampled
//! in 4 / 6 / 8 CFG-free steps.
//!
//! What differs from the base H3 checkpoint, and where it lands:
//! - **Time embedding**: `temb(t, r) = emb_t(t) + gate * (emb_r(r) - emb_t(t))`
//!   (`endpoint_time_embedder`, `hyperflow_gate`), 16 wide; the AdaLN
//!   projections read `silu(temb)` as on the base. Every distinct `(t, r)` pair
//!   gets one modulation-table row (`DitConfig::endpoint_gate`).
//! - **Low-rank linears**: a site is either the dense `<site>.weight` or
//!   `<site>.down.weight` (`[rank, in]`) + `<site>.up.weight` (`[out, rank]`),
//!   `y = up(down(x))`. A factored block splits q / k / v; [`HyperFlowSource`]
//!   serves them as ONE fused qkv site: `qkv_proj.down` stacks the three downs
//!   (`[rq + rk + rv, in]`) and `qkv_proj.up` is block-diagonal (`[3 * inner,
//!   rq + rk + rv]`), assembled from whole quant super-blocks plus all-zero
//!   blocks (exact: a zero block decodes to zeros), so the block forward keeps
//!   its fused `qkv -> split` path.
//! - **Schedule**: a fixed raw sigma grid per step count (shifts 12 video / 3
//!   audio as on the base), Euler, endpoint `r_i = 1 - sigma_{i+1}`.

use std::collections::HashMap;

use thinfer_core::quant::QuantKind;
use thinfer_core::tensor::StorageEncoding;
use thinfer_core::weight::{WeightCatalog, WeightId, WeightReader, WeightSource};

use crate::common::lora::VecReader;

/// Raw (unshifted) sigma grids by model evaluations (`schedule.py`
/// `SIGMA_GRIDS`); 6 is the release default.
pub fn sigma_grid(steps: usize) -> Option<&'static [f32]> {
    match steps {
        4 => Some(&[1.0, 0.681872, 0.205019, 0.036496, 0.0]),
        6 => Some(&[1.0, 0.825217, 0.471725, 0.147401, 0.049626, 0.015517, 0.0]),
        8 => Some(&[
            1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0,
        ]),
        _ => None,
    }
}

pub const DEFAULT_STEPS: usize = 6;

/// How a synthesized fused-qkv tensor is assembled.
enum Synth {
    /// Rows of each part, back to back (the stacked downs).
    Stack(Vec<WeightId>),
    /// Block-diagonal: each part's rows land in their own K range, zero
    /// blocks elsewhere. `(part, k)` in order, `kind` shared.
    BlockDiag {
        parts: Vec<(WeightId, usize)>,
        kind: QuantKind,
    },
}

pub struct HyperFlowSource<S: WeightSource> {
    base: S,
    catalog: WeightCatalog,
    synth: HashMap<WeightId, Synth>,
}

#[derive(Debug)]
pub enum HyperFlowError<E: core::fmt::Debug> {
    Base(E),
    Read(String),
    Layout(String),
}

pub enum HyperFlowReader<S: WeightSource> {
    Base(S::Reader),
    Synth(VecReader),
}

impl<S: WeightSource> WeightReader for HyperFlowReader<S> {
    type Error = HyperFlowError<<S::Reader as WeightReader>::Error>;
    fn len(&self) -> u64 {
        match self {
            Self::Base(r) => r.len(),
            Self::Synth(r) => r.len(),
        }
    }
    async fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), Self::Error> {
        match self {
            Self::Base(r) => r.read_at(offset, dst).await.map_err(HyperFlowError::Base),
            Self::Synth(r) => {
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

impl<S: WeightSource> HyperFlowSource<S> {
    /// Wrap the GGUF, adding the fused qkv `down`/`up` of every block whose
    /// q / k / v are factored.
    pub fn new(base: S) -> Result<Self, HyperFlowError<S::Error>> {
        let mut catalog = WeightCatalog {
            entries: base.catalog().entries.clone(),
        };
        let mut synth = HashMap::new();
        for i in 0.. {
            let p = format!("blocks.{i}.attn");
            let id = |s: &str| WeightId(format!("{p}.{s}"));
            if base.catalog().get(&id("q_norm.weight")).is_none() {
                break;
            }
            if base.catalog().get(&id("qkv_proj.weight")).is_some() {
                continue;
            }
            let get = |s: &str| {
                base.catalog()
                    .get(&id(s))
                    .cloned()
                    .ok_or_else(|| HyperFlowError::Layout(format!("{p}.{s} missing")))
            };
            let downs = ["q", "k", "v"].map(|x| id(&format!("{x}_proj.down.weight")));
            let ups = ["q", "k", "v"].map(|x| id(&format!("{x}_proj.up.weight")));
            let de: Vec<_> = ["q", "k", "v"]
                .iter()
                .map(|x| get(&format!("{x}_proj.down.weight")))
                .collect::<Result<_, _>>()?;
            let ue: Vec<_> = ["q", "k", "v"]
                .iter()
                .map(|x| get(&format!("{x}_proj.up.weight")))
                .collect::<Result<_, _>>()?;
            let quant = |e: &thinfer_core::weight::WeightEntry| match e.encoding {
                Some(StorageEncoding::Quant(k)) => Some(k),
                _ => None,
            };
            let (Some(dk), Some(uk)) = (quant(&de[0]), quant(&ue[0])) else {
                return Err(HyperFlowError::Layout(format!("{p}: unquantized factors")));
            };
            if de.iter().any(|e| quant(e) != Some(dk)) || ue.iter().any(|e| quant(e) != Some(uk)) {
                return Err(HyperFlowError::Layout(format!(
                    "{p}: mixed quant kinds in q/k/v"
                )));
            }
            // down `[rank, in]` x3 -> `[sum rank, in]`
            let ranks: Vec<usize> = de.iter().map(|e| e.shape.0[0]).collect();
            let k_in = de[0].shape.0[1];
            let r_tot: usize = ranks.iter().sum();
            let mut down = de[0].clone();
            down.shape.0 = vec![r_tot, k_in];
            down.size = de.iter().map(|e| e.size).sum();
            down.encoding_label = format!("{}(qkv stack)", de[0].encoding_label);
            // up `[out, rank]` x3 -> block-diagonal `[3 * out, sum rank]`
            let n_out = ue[0].shape.0[0];
            let bs = uk.block_size() as usize;
            if ranks.iter().any(|r| r % bs != 0) {
                return Err(HyperFlowError::Layout(format!(
                    "{p}: rank not a whole quant block"
                )));
            }
            let mut up = ue[0].clone();
            up.shape.0 = vec![3 * n_out, r_tot];
            up.size = uk.bytes_for_elements((3 * n_out * r_tot) as u64);
            up.encoding_label = format!("{}(qkv block-diag)", ue[0].encoding_label);
            let did = id("qkv_proj.down.weight");
            let uid = id("qkv_proj.up.weight");
            catalog.entries.insert(did.clone(), down);
            catalog.entries.insert(uid.clone(), up);
            synth.insert(did, Synth::Stack(downs.to_vec()));
            synth.insert(
                uid,
                Synth::BlockDiag {
                    parts: ups.into_iter().zip(ranks).collect(),
                    kind: uk,
                },
            );
        }
        Ok(Self {
            base,
            catalog,
            synth,
        })
    }

    async fn read_all(&self, id: &WeightId) -> Result<Vec<u8>, HyperFlowError<S::Error>> {
        let mut r = self.base.open(id).await.map_err(HyperFlowError::Base)?;
        let mut bytes = vec![0u8; r.len() as usize];
        r.read_at(0, &mut bytes)
            .await
            .map_err(|e| HyperFlowError::Read(format!("{id:?}: {e:?}")))?;
        Ok(bytes)
    }

    async fn build(&self, s: &Synth) -> Result<Vec<u8>, HyperFlowError<S::Error>> {
        match s {
            Synth::Stack(parts) => {
                let mut out = Vec::new();
                for p in parts {
                    out.extend(self.read_all(p).await?);
                }
                Ok(out)
            }
            Synth::BlockDiag { parts, kind } => {
                let bs = kind.block_size() as usize;
                let bb = kind.bytes_per_block() as usize;
                let k_tot: usize = parts.iter().map(|(_, k)| k).sum();
                let row_bytes = k_tot / bs * bb;
                let mut out = Vec::new();
                let mut k0 = 0;
                for (id, k) in parts {
                    let bytes = self.read_all(id).await?;
                    let part_row = k / bs * bb;
                    for src in bytes.chunks_exact(part_row) {
                        let start = out.len();
                        out.resize(start + row_bytes, 0);
                        let at = start + k0 / bs * bb;
                        out[at..at + part_row].copy_from_slice(src);
                    }
                    k0 += k;
                }
                Ok(out)
            }
        }
    }
}

impl<S: WeightSource> WeightSource for HyperFlowSource<S> {
    type Reader = HyperFlowReader<S>;
    type Error = HyperFlowError<S::Error>;

    fn catalog(&self) -> &WeightCatalog {
        &self.catalog
    }

    async fn open(&self, id: &WeightId) -> Result<Self::Reader, Self::Error> {
        match self.synth.get(id) {
            Some(s) => Ok(HyperFlowReader::Synth(VecReader::from_vec(
                self.build(s).await?,
            ))),
            None => Ok(HyperFlowReader::Base(
                self.base.open(id).await.map_err(HyperFlowError::Base)?,
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grids_end_at_zero_and_decrease() {
        for steps in [4, 6, 8] {
            let g = sigma_grid(steps).unwrap();
            assert_eq!(g.len(), steps + 1);
            assert_eq!((g[0], g[steps]), (1.0, 0.0));
            assert!(g.windows(2).all(|w| w[1] < w[0]));
        }
        assert!(sigma_grid(5).is_none());
    }
}
