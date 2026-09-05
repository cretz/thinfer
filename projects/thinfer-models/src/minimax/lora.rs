//! The MiniMax-H3 turbo LoRA, re-presented so the shared fold machinery can
//! consume it unchanged.
//!
//! The lightx2v distillation is diffusers/PEFT-named and split per projection
//! (`transformer_blocks.N.attn.to_q/to_k/to_v`), while the checkpoint keeps ONE
//! fused, per-head interleaved `blocks.N.attn.qkv_proj`. Three separate rank-128
//! pairs therefore have to become one delta on a fused tensor.
//!
//! That is exactly representable without ever materializing the dense delta.
//! For a scatter `P_x` placing projection `x`'s rows into the fused layout,
//!
//! ```text
//! delta = P_q B_q A_q + P_k B_k A_k + P_v B_v A_v
//!       = [P_q B_q | P_k B_k | P_v B_v] @ [A_q ; A_k ; A_v]
//! ```
//!
//! which is a single rank-384 pair. So this module presents a synthetic LoRA
//! source whose `qkv_proj` entries are that concatenation, and
//! [`crate::common::lora`] folds it like any other adapter.
//!
//! The feed-forward needs no such work: `ff.net.0.proj` already emits the FUSED
//! 28672-wide `[gate; value]` delta, matching `mlp.fc1` directly.
//!
//! Tensors are synthesized ON DEMAND rather than up front: the full set is
//! ~1 GiB, one tensor is ~17 MiB.

use std::collections::HashMap;

use half::bf16;

use thinfer_core::tensor::{Shape, StorageEncoding};
use thinfer_core::weight::{WeightCatalog, WeightEntry, WeightId, WeightReader, WeightSource};

use super::DitConfig;

/// Per-projection LoRA rank of the released adapter. Read from the tensors
/// rather than trusted, but pinned here for the fused-rank arithmetic.
pub const LORA_RANK: usize = 128;

/// One fused-`qkv_proj` site: the three source pairs feeding it.
struct QkvSite {
    block: usize,
    /// `transformer_blocks` or `token_refiner.refiner_blocks`.
    lora_prefix: String,
    /// `blocks.{i}` or `token_refiner.blocks.{i}`.
    base_prefix: String,
}

/// A synthetic view of the turbo LoRA in the checkpoint's own tensor naming.
///
/// Entries are keyed `diffusion_model.{base}.lora_{A,B}.weight`, which is what
/// [`crate::common::lora::discover_specs`] looks for.
pub struct TurboLoraSource<L: WeightSource> {
    inner: L,
    catalog: WeightCatalog,
    /// Synthetic id -> how to build it.
    plan: HashMap<WeightId, Synth>,
}

#[derive(Clone)]
enum Synth {
    /// Pass the inner tensor through under a new name.
    Rename(WeightId),
    /// Stack `[A_q; A_k; A_v]` into `[3 * rank, K]`.
    QkvA {
        parts: [WeightId; 3],
        rank: usize,
        k: usize,
    },
    /// Scatter `B_q|B_k|B_v` columns into the per-head interleaved row layout.
    QkvB {
        parts: [WeightId; 3],
        rank: usize,
        heads: usize,
        head_dim: usize,
    },
}

#[derive(Debug)]
pub enum LoraError<LE: core::fmt::Debug> {
    Lora(LE),
    Read(String),
    Missing(String),
}

fn ids(prefix: &str, site: &str) -> [WeightId; 2] {
    [
        WeightId(format!("{prefix}.{site}.lora_A.default.weight")),
        WeightId(format!("{prefix}.{site}.lora_B.default.weight")),
    ]
}

impl<L: WeightSource> TurboLoraSource<L> {
    /// Build the synthetic catalog. Blocks whose LoRA tensors are absent are
    /// skipped rather than erroring: the adapter need not cover every site.
    pub fn new(cfg: &DitConfig, inner: L) -> Self {
        let mut catalog = WeightCatalog::new();
        let mut plan = HashMap::new();

        let sites: Vec<QkvSite> = (0..cfg.n_layers)
            .map(|i| QkvSite {
                block: i,
                lora_prefix: format!("transformer_blocks.{i}"),
                base_prefix: format!("blocks.{i}"),
            })
            .chain((0..cfg.token_refiner_layers).map(|i| QkvSite {
                block: i,
                lora_prefix: format!("token_refiner.refiner_blocks.{i}"),
                base_prefix: format!("token_refiner.blocks.{i}"),
            }))
            .collect();

        for site in &sites {
            let _ = site.block;
            // --- fused qkv ---
            let qkv: Vec<[WeightId; 2]> = ["to_q", "to_k", "to_v"]
                .iter()
                .map(|p| ids(&site.lora_prefix, &format!("attn.{p}")))
                .collect();
            let have_qkv = qkv
                .iter()
                .all(|[a, b]| inner.catalog().get(a).is_some() && inner.catalog().get(b).is_some());
            if have_qkv {
                let rank = inner
                    .catalog()
                    .get(&qkv[0][0])
                    .and_then(|e| e.shape.0.first().copied())
                    .unwrap_or(LORA_RANK);
                let fused_rank = 3 * rank;
                let a_id = WeightId(format!(
                    "diffusion_model.{}.attn.qkv_proj.lora_A.weight",
                    site.base_prefix
                ));
                let b_id = WeightId(format!(
                    "diffusion_model.{}.attn.qkv_proj.lora_B.weight",
                    site.base_prefix
                ));
                catalog.entries.insert(
                    a_id.clone(),
                    entry(vec![fused_rank, cfg.hidden], fused_rank * cfg.hidden),
                );
                catalog.entries.insert(
                    b_id.clone(),
                    entry(
                        vec![3 * cfg.inner_dim(), fused_rank],
                        3 * cfg.inner_dim() * fused_rank,
                    ),
                );
                plan.insert(
                    a_id,
                    Synth::QkvA {
                        parts: [qkv[0][0].clone(), qkv[1][0].clone(), qkv[2][0].clone()],
                        rank,
                        k: cfg.hidden,
                    },
                );
                plan.insert(
                    b_id,
                    Synth::QkvB {
                        parts: [qkv[0][1].clone(), qkv[1][1].clone(), qkv[2][1].clone()],
                        rank,
                        heads: cfg.n_heads,
                        head_dim: cfg.head_dim,
                    },
                );
            }

            // --- straight renames ---
            for (lora_site, base_site) in [
                ("attn.to_out.0", "attn.out_proj"),
                ("ff.net.0.proj", "mlp.fc1"),
                ("ff.net.2", "mlp.fc2"),
            ] {
                let [a, b] = ids(&site.lora_prefix, lora_site);
                let (Some(ae), Some(be)) = (inner.catalog().get(&a), inner.catalog().get(&b))
                else {
                    continue;
                };
                for (src, e, suffix) in [(a, ae, "lora_A"), (b, be, "lora_B")] {
                    let dst = WeightId(format!(
                        "diffusion_model.{}.{base_site}.{suffix}.weight",
                        site.base_prefix
                    ));
                    catalog.entries.insert(
                        dst.clone(),
                        entry(e.shape.0.clone(), e.shape.0.iter().product()),
                    );
                    plan.insert(dst, Synth::Rename(src));
                }
            }
        }

        Self {
            inner,
            catalog,
            plan,
        }
    }

    pub fn site_count(&self) -> usize {
        self.plan.len() / 2
    }

    async fn read_bf16(&self, id: &WeightId) -> Result<Vec<bf16>, LoraError<L::Error>> {
        let entry = self
            .inner
            .catalog()
            .get(id)
            .ok_or_else(|| LoraError::Missing(id.0.clone()))?;
        let elems: usize = entry.shape.0.iter().product();
        let mut bytes = vec![0u8; entry.size as usize];
        let mut reader = self.inner.open(id).await.map_err(LoraError::Lora)?;
        reader
            .read_at(0, &mut bytes)
            .await
            .map_err(|e| LoraError::Read(format!("{}: {e:?}", id.0)))?;
        Ok(bytes[..elems * 2]
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
            .collect())
    }
}

fn entry(shape: Vec<usize>, elems: usize) -> WeightEntry {
    WeightEntry {
        offset: 0,
        size: (elems * 2) as u64,
        encoding: Some(StorageEncoding::Bf16),
        encoding_label: "BF16".to_string(),
        shape: Shape(shape),
    }
}

impl<L: WeightSource> WeightSource for TurboLoraSource<L> {
    type Reader = super::super::common::lora::VecReader;
    type Error = LoraError<L::Error>;

    fn catalog(&self) -> &WeightCatalog {
        &self.catalog
    }

    async fn open(&self, id: &WeightId) -> Result<Self::Reader, Self::Error> {
        let synth = self
            .plan
            .get(id)
            .ok_or_else(|| LoraError::Missing(id.0.clone()))?
            .clone();
        let values: Vec<bf16> = match synth {
            Synth::Rename(src) => self.read_bf16(&src).await?,
            Synth::QkvA { parts, rank, k } => {
                let mut out = Vec::with_capacity(3 * rank * k);
                for p in &parts {
                    out.extend_from_slice(&self.read_bf16(p).await?);
                }
                out
            }
            Synth::QkvB {
                parts,
                rank,
                heads,
                head_dim,
            } => {
                // Fused B is [heads * 3 * head_dim, 3 * rank]. Projection `x`'s
                // rows land at `h * 3 * head_dim + x * head_dim`, and its
                // columns occupy the `x`-th rank block; everything else is 0.
                let fused_rank = 3 * rank;
                let rows = heads * 3 * head_dim;
                let mut out = vec![bf16::ZERO; rows * fused_rank];
                for (x, p) in parts.iter().enumerate() {
                    let b = self.read_bf16(p).await?;
                    for h in 0..heads {
                        for d in 0..head_dim {
                            let src = (h * head_dim + d) * rank;
                            let dst = (h * 3 * head_dim + x * head_dim + d) * fused_rank + x * rank;
                            out[dst..dst + rank].copy_from_slice(&b[src..src + rank]);
                        }
                    }
                }
                out
            }
        };
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        Ok(super::super::common::lora::VecReader::from_vec(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fused pair must reproduce the same delta the three separate pairs
    /// would, scattered per head. This is the whole point of the module, and a
    /// wrong scatter is silent: it produces a plausible but wrong model.
    #[test]
    fn fused_pair_reproduces_the_per_head_scatter() {
        // Tiny stand-in: 2 heads x head_dim 2, k = 3, rank 1.
        let (heads, head_dim, k, rank) = (2usize, 2usize, 3usize, 1usize);
        let inner = heads * head_dim; // 4
        // A_x [rank, k], B_x [inner, rank] for x in q,k,v.
        let a = [
            vec![1.0f32, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let b = [
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ];

        // Reference: scatter each projection's delta into the fused layout.
        let rows = heads * 3 * head_dim;
        let mut want = vec![0.0f32; rows * k];
        for (x, (ax, bx)) in a.iter().zip(&b).enumerate() {
            for h in 0..heads {
                for d in 0..head_dim {
                    let r = h * 3 * head_dim + x * head_dim + d;
                    for kk in 0..k {
                        want[r * k + kk] += bx[h * head_dim + d] * ax[kk];
                    }
                }
            }
        }

        // Fused: A_cat [3 * rank, k], B_cat [rows, 3 * rank].
        let fused_rank = 3 * rank;
        let mut a_cat = Vec::new();
        for ax in &a {
            a_cat.extend_from_slice(ax);
        }
        let mut b_cat = vec![0.0f32; rows * fused_rank];
        for (x, bx) in b.iter().enumerate() {
            for h in 0..heads {
                for d in 0..head_dim {
                    let src = (h * head_dim + d) * rank;
                    let dst = (h * 3 * head_dim + x * head_dim + d) * fused_rank + x * rank;
                    b_cat[dst..dst + rank].copy_from_slice(&bx[src..src + rank]);
                }
            }
        }
        let mut got = vec![0.0f32; rows * k];
        for r in 0..rows {
            for kk in 0..k {
                let mut acc = 0.0;
                for j in 0..fused_rank {
                    acc += b_cat[r * fused_rank + j] * a_cat[j * k + kk];
                }
                got[r * k + kk] = acc;
            }
        }
        assert_eq!(got, want);
        assert_eq!(inner, 4);
    }

    /// The fused layout is per-head interleaved, NOT three contiguous blocks.
    /// Head 0's k rows must sit at rows 2..4, not at `inner`.
    #[test]
    fn scatter_is_per_head_interleaved_not_block_contiguous() {
        let (heads, head_dim) = (2usize, 2usize);
        let row_of = |h: usize, x: usize, d: usize| h * 3 * head_dim + x * head_dim + d;
        assert_eq!(row_of(0, 0, 0), 0); // h0 q
        assert_eq!(row_of(0, 1, 0), 2); // h0 k
        assert_eq!(row_of(0, 2, 0), 4); // h0 v
        assert_eq!(row_of(1, 0, 0), 6); // h1 q
        assert_eq!(heads * 3 * head_dim, 12);
    }
}
