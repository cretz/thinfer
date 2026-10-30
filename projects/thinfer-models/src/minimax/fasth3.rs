//! FastH3 (FastVideo's 8-step DMD distillation of MiniMax-H3) presented under
//! the H3 DiT's own tensor names, so the existing block stack, AdaLN precompute
//! and loader run it unchanged.
//!
//! The checkpoint is diffusers-named and split in two files: the block linears
//! as NVFP4 (`{prefix}::_nvfp4_weight` / `_nvfp4_weight_scale` / `_nvfp4_alpha`)
//! and everything else bf16/f32. The H3 DiT reads the ORIGINAL names and, for
//! attention and the FFN, FUSED tensors:
//!
//! - `blocks.N.attn.qkv_proj` = contiguous `[q; k; v]` rows (the layout the
//!   engine consumes, see `dit`), assembled from `to_q | to_k | to_v`;
//! - `blocks.N.mlp.fc1` = `[gate; value]`, from diffusers' `[value; gate]`
//!   `ff.fc_in` with its halves swapped.
//!
//! Both are pure 128-row reorderings, and a 128-row group is exactly one NVFP4
//! scale tile with its own alpha ([`thinfer_core::ops::nvfp4`]), so an NVFP4
//! composite is assembled by copying byte ranges: no requantization. The bf16
//! token refiner gets the same fusions as plain row concatenations.
//!
//! Every tensor is served by a [`SpanReader`] over byte ranges of the source
//! tensors, so a composite is never materialized on the host: streaming it
//! costs the same as streaming the source bytes.

use std::collections::HashMap;
use std::ops::Range;

use thinfer_core::ops::nvfp4::{BLOCK, Nvfp4Layout};
use thinfer_core::tensor::{Shape, StorageEncoding};
use thinfer_core::weight::{WeightCatalog, WeightEntry, WeightId, WeightReader, WeightSource};

use super::DitConfig;

#[derive(Debug)]
pub enum FastH3Error<E: core::fmt::Debug> {
    Source(E),
    Missing(String),
}

/// How one engine-named tensor is produced.
#[derive(Clone, Debug)]
enum Plan {
    /// A source tensor under a new name.
    Rename(WeightId),
    /// Row ranges of 2-D bf16 source tensors, concatenated.
    Rows {
        parts: Vec<(WeightId, Range<usize>)>,
        row_bytes: usize,
    },
    /// An NVFP4 composite from row ranges (multiples of 128) of NVFP4 linears
    /// (by source prefix), all with the same `k`.
    Nvfp4 {
        parts: Vec<(String, Range<usize>)>,
        k: usize,
    },
}

/// See the module docs. `inner` is the union of the checkpoint's two files.
pub struct FastH3Source<S: WeightSource> {
    inner: S,
    catalog: WeightCatalog,
    plan: HashMap<WeightId, Plan>,
}

fn id(s: impl Into<String>) -> WeightId {
    WeightId(s.into())
}

impl<S: WeightSource> FastH3Source<S> {
    pub fn new(cfg: &DitConfig, inner: S) -> Result<Self, FastH3Error<S::Error>> {
        let mut out = Self {
            inner,
            catalog: WeightCatalog::new(),
            plan: HashMap::new(),
        };
        for (engine, source) in [
            ("video_patch_proj", "proj_in"),
            ("audio_patch_proj", "audio_proj_in"),
            ("condition_proj", "context_embedder"),
            ("time_embedder.proj_in", "time_embedder.linear_1"),
            ("time_embedder.proj_out", "time_embedder.linear_2"),
            ("final_layer.adaln_proj.linear", "norm_out.linear"),
            ("final_layer.video_out", "proj_out"),
            ("final_layer.audio_out", "audio_proj_out"),
        ] {
            out.rename(&format!("{engine}.weight"), &format!("{source}.weight"))?;
            out.rename(&format!("{engine}.bias"), &format!("{source}.bias"))?;
        }
        out.rename("final_layer.norm.weight", "norm_out.norm.weight")?;
        out.rename(
            "token_refiner.final_norm.weight",
            "token_refiner.final_norm.weight",
        )?;
        if cfg.adaln_rank.is_some() {
            out.rename("adaln_basis.weight", "adaln_basis.weight")?;
        }

        let (inner, ffn) = (cfg.inner_dim(), cfg.ffn_hidden);
        for j in 0..cfg.token_refiner_layers {
            let (e, t) = (
                format!("token_refiner.blocks.{j}"),
                format!("token_refiner.refiner_blocks.{j}"),
            );
            out.block_norms(&e, &t)?;
            out.rows(
                &format!("{e}.attn.qkv_proj.weight"),
                ["to_q", "to_k", "to_v"]
                    .map(|p| (id(format!("{t}.attn.{p}.weight")), 0..inner))
                    .to_vec(),
                cfg.hidden,
            )?;
            out.rename(
                &format!("{e}.attn.out_proj.weight"),
                &format!("{t}.attn.to_out.0.weight"),
            )?;
            let fc_in = id(format!("{t}.ff.net.0.proj.weight"));
            out.rows(
                &format!("{e}.mlp.fc1.weight"),
                vec![(fc_in.clone(), ffn..2 * ffn), (fc_in, 0..ffn)],
                cfg.hidden,
            )?;
            out.rename(
                &format!("{e}.mlp.fc2.weight"),
                &format!("{t}.ff.net.2.weight"),
            )?;
        }

        for i in 0..cfg.n_layers {
            let (e, t) = (format!("blocks.{i}"), format!("transformer_blocks.{i}"));
            out.block_norms(&e, &t)?;
            out.rename(
                &format!("{e}.adaln_proj.linear.weight"),
                &format!("{t}.adaln_proj.linear.weight"),
            )?;
            out.rename(
                &format!("{e}.adaln_proj.linear.bias"),
                &format!("{t}.adaln_proj.linear.bias"),
            )?;
            out.nvfp4(
                &format!("{e}.attn.qkv_proj.weight"),
                ["to_q", "to_k", "to_v"]
                    .map(|p| (format!("{t}.attn.{p}"), 0..inner))
                    .to_vec(),
                cfg.hidden,
            )?;
            // The VSA compression gate (`to_gate_compress`, same input as q/k/v).
            out.nvfp4(
                &format!("{e}.attn.gate_proj.weight"),
                vec![(format!("{t}.attn.to_gate_compress"), 0..inner)],
                cfg.hidden,
            )?;
            out.nvfp4(
                &format!("{e}.attn.out_proj.weight"),
                vec![(format!("{t}.attn.to_out"), 0..cfg.hidden)],
                inner,
            )?;
            let fc_in = format!("{t}.ff.fc_in");
            out.nvfp4(
                &format!("{e}.mlp.fc1.weight"),
                vec![(fc_in.clone(), ffn..2 * ffn), (fc_in, 0..ffn)],
                cfg.hidden,
            )?;
            out.nvfp4(
                &format!("{e}.mlp.fc2.weight"),
                vec![(format!("{t}.ff.fc_out"), 0..cfg.hidden)],
                ffn,
            )?;
        }
        Ok(out)
    }

    fn source_entry(&self, src: &WeightId) -> Result<&WeightEntry, FastH3Error<S::Error>> {
        self.inner
            .catalog()
            .get(src)
            .ok_or_else(|| FastH3Error::Missing(src.0.clone()))
    }

    fn rename(&mut self, engine: &str, source: &str) -> Result<(), FastH3Error<S::Error>> {
        let src = id(source);
        let entry = self.source_entry(&src)?.clone();
        self.catalog
            .entries
            .insert(id(engine), WeightEntry { offset: 0, ..entry });
        self.plan.insert(id(engine), Plan::Rename(src));
        Ok(())
    }

    /// Per-block norms: identical roles, diffusers names.
    fn block_norms(&mut self, e: &str, t: &str) -> Result<(), FastH3Error<S::Error>> {
        self.rename(&format!("{e}.norm1.weight"), &format!("{t}.norm1.weight"))?;
        self.rename(&format!("{e}.norm2.weight"), &format!("{t}.norm2.weight"))?;
        self.rename(
            &format!("{e}.attn.q_norm.weight"),
            &format!("{t}.attn.norm_q.weight"),
        )?;
        self.rename(
            &format!("{e}.attn.k_norm.weight"),
            &format!("{t}.attn.norm_k.weight"),
        )
    }

    fn rows(
        &mut self,
        engine: &str,
        parts: Vec<(WeightId, Range<usize>)>,
        k: usize,
    ) -> Result<(), FastH3Error<S::Error>> {
        for (src, _) in &parts {
            let e = self.source_entry(src)?;
            if e.encoding != Some(StorageEncoding::Bf16) || e.shape.0.get(1) != Some(&k) {
                return Err(FastH3Error::Missing(format!(
                    "{} is not bf16 [_, {k}]",
                    src.0
                )));
            }
        }
        let n: usize = parts.iter().map(|(_, r)| r.len()).sum();
        self.catalog.entries.insert(
            id(engine),
            WeightEntry {
                offset: 0,
                size: (n * k * 2) as u64,
                encoding: Some(StorageEncoding::Bf16),
                encoding_label: "BF16".into(),
                shape: Shape(vec![n, k]),
            },
        );
        self.plan.insert(
            id(engine),
            Plan::Rows {
                parts,
                row_bytes: k * 2,
            },
        );
        Ok(())
    }

    fn nvfp4(
        &mut self,
        engine: &str,
        parts: Vec<(String, Range<usize>)>,
        k: usize,
    ) -> Result<(), FastH3Error<S::Error>> {
        for (prefix, r) in &parts {
            let packed = self.source_entry(&id(format!("{prefix}::_nvfp4_weight")))?;
            if packed.shape.0.get(1) != Some(&(k / 2)) || r.start % 128 != 0 || r.end % 128 != 0 {
                return Err(FastH3Error::Missing(format!(
                    "{prefix}: not an NVFP4 [_, {k}] in 128-row groups"
                )));
            }
        }
        let n: usize = parts.iter().map(|(_, r)| r.len()).sum();
        let layout = Nvfp4Layout::new(n, k);
        self.catalog.entries.insert(
            id(engine),
            WeightEntry {
                offset: 0,
                size: layout.total_len() as u64,
                encoding: Some(StorageEncoding::Nvfp4),
                encoding_label: "NVFP4".into(),
                shape: Shape(vec![n, k]),
            },
        );
        self.plan.insert(id(engine), Plan::Nvfp4 { parts, k });
        Ok(())
    }

    /// The byte spans (into the opened source tensors) that make up `plan`'s
    /// tensor, in order. Source tensors are opened once each.
    async fn spans(
        &self,
        plan: &Plan,
    ) -> Result<(Vec<S::Reader>, Vec<Span>), FastH3Error<S::Error>> {
        let mut ids: Vec<WeightId> = Vec::new();
        let mut index = |wid: WeightId| match ids.iter().position(|x| *x == wid) {
            Some(i) => i,
            None => {
                ids.push(wid);
                ids.len() - 1
            }
        };
        let mut spans = Vec::new();
        match plan {
            Plan::Rename(src) => {
                let len = self.source_entry(src)?.size;
                spans.push(Span {
                    src: index(src.clone()),
                    offset: 0,
                    len,
                });
            }
            Plan::Rows { parts, row_bytes } => {
                for (src, r) in parts {
                    spans.push(Span {
                        src: index(src.clone()),
                        offset: (r.start * row_bytes) as u64,
                        len: (r.len() * row_bytes) as u64,
                    });
                }
            }
            Plan::Nvfp4 { parts, k } => {
                let n: usize = parts.iter().map(|(_, r)| r.len()).sum();
                // Same K throughout, so the same padded block width: a source's
                // 128-row tile rows are byte-identical in size to the output's.
                let tile_row_bytes = (Nvfp4Layout::new(n, *k).blocks_padded() * 128) as u64;
                debug_assert_eq!(k / BLOCK % 4, 0, "scale tiles assumed unpadded along K");
                let part = |p: &str, s: &str| id(format!("{p}::{s}"));
                for (prefix, r) in parts {
                    spans.push(Span {
                        src: index(part(prefix, "_nvfp4_weight")),
                        offset: (r.start * k / 2) as u64,
                        len: (r.len() * k / 2) as u64,
                    });
                }
                for (prefix, r) in parts {
                    spans.push(Span {
                        src: index(part(prefix, "_nvfp4_weight_scale")),
                        offset: r.start as u64 / 128 * tile_row_bytes,
                        len: r.len() as u64 / 128 * tile_row_bytes,
                    });
                }
                for name in ["_nvfp4_alpha", "_nvfp4_input_global_sf"] {
                    for (prefix, r) in parts {
                        let src = index(part(prefix, name));
                        for _ in 0..r.len() / 128 {
                            spans.push(Span {
                                src,
                                offset: 0,
                                len: 4,
                            });
                        }
                    }
                }
            }
        }
        let mut readers = Vec::with_capacity(ids.len());
        for wid in &ids {
            readers.push(self.inner.open(wid).await.map_err(FastH3Error::Source)?);
        }
        Ok((readers, spans))
    }
}

/// One contiguous byte range of an opened source tensor.
#[derive(Clone, Copy, Debug)]
struct Span {
    src: usize,
    offset: u64,
    len: u64,
}

/// A synthesized tensor read straight from its source byte ranges: the
/// composite is never materialized on the host.
pub struct SpanReader<R: WeightReader> {
    readers: Vec<R>,
    spans: Vec<Span>,
    len: u64,
}

impl<R: WeightReader> WeightReader for SpanReader<R> {
    type Error = R::Error;

    fn len(&self) -> u64 {
        self.len
    }

    async fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), Self::Error> {
        let end = offset + dst.len() as u64;
        let (mut start, mut written, mut s0) = (offset, 0usize, 0u64);
        for span in &self.spans {
            let s1 = s0 + span.len;
            if s1 > start && s0 < end {
                let b = end.min(s1);
                let n = (b - start) as usize;
                self.readers[span.src]
                    .read_at(span.offset + (start - s0), &mut dst[written..written + n])
                    .await?;
                written += n;
                start = b;
            }
            if s1 >= end {
                break;
            }
            s0 = s1;
        }
        Ok(())
    }

    /// Forward the readahead hint to every source range the window covers.
    fn will_read(&mut self, offset: u64, len: u64) {
        let end = offset + len;
        let mut s0 = 0u64;
        for span in &self.spans {
            let s1 = s0 + span.len;
            if s1 > offset && s0 < end {
                let (a, b) = (offset.max(s0), end.min(s1));
                self.readers[span.src].will_read(span.offset + (a - s0), b - a);
            }
            if s1 >= end {
                break;
            }
            s0 = s1;
        }
    }
}

impl<S: WeightSource> WeightSource for FastH3Source<S> {
    type Reader = SpanReader<S::Reader>;
    type Error = FastH3Error<S::Error>;

    fn catalog(&self) -> &WeightCatalog {
        &self.catalog
    }

    async fn open(&self, wid: &WeightId) -> Result<Self::Reader, Self::Error> {
        let plan = self
            .plan
            .get(wid)
            .ok_or_else(|| FastH3Error::Missing(wid.0.clone()))?;
        let len = self.catalog.entries[wid].size;
        let (readers, spans) = self.spans(plan).await?;
        debug_assert_eq!(spans.iter().map(|s| s.len).sum::<u64>(), len);
        Ok(SpanReader {
            readers,
            spans,
            len,
        })
    }
}
