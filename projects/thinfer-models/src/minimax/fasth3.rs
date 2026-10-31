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
        names: Nvfp4Names,
    },
    /// One step's fused PDD head (f32): `rows_0 + sum_j w_j rows_j` over
    /// `head_rows`-row heads of an f32 source `[heads * head_rows, cols]`.
    Pdd {
        src: WeightId,
        head_rows: usize,
        cols: usize,
        mix: Vec<(usize, f32)>,
    },
}

/// Where an NVFP4 linear's parts live, as suffixes of its prefix.
#[derive(Clone, Copy, Debug)]
struct Nvfp4Names {
    packed: &'static str,
    scales: &'static str,
    tail: Nvfp4Tail,
}

/// Where the composite's per-128-row `alpha` (weight dequant multiplier) and
/// `input_sf` (static activation scale) come from.
#[derive(Clone, Copy, Debug)]
enum Nvfp4Tail {
    /// Stored f32 scalars, copied.
    Stored {
        alpha: &'static str,
        input_sf: &'static str,
    },
    /// A stored weight `global_sf` (alpha is its inverse) and a unit
    /// activation scale: computed at open.
    InverseUnit { global_sf: &'static str },
}

/// FastVideo / ModelOpt export.
const FASTVIDEO: Nvfp4Names = Nvfp4Names {
    packed: "::_nvfp4_weight",
    scales: "::_nvfp4_weight_scale",
    tail: Nvfp4Tail::Stored {
        alpha: "::_nvfp4_alpha",
        input_sf: "::_nvfp4_input_global_sf",
    },
};

/// Comfy `TensorCoreNVFP4Layout`: `weight_scale_2` is the dequant multiplier
/// (FastVideo's alpha); there is no static input scale (activations scale
/// dynamically, see `Nvfp4Format`), so that slot repeats it unread.
const COMFY: Nvfp4Names = Nvfp4Names {
    packed: ".weight",
    scales: ".weight_scale",
    tail: Nvfp4Tail::Stored {
        alpha: ".weight_scale_2",
        input_sf: ".weight_scale_2",
    },
};

/// FastVideo's packed NVFP4 VAE decoders (`fastvideo_h3_decoder_nvfp4_packed_v1`,
/// FlashInfer order and `layout_128x4` scales, `act_scale = "unit"`).
const DECODER: Nvfp4Names = Nvfp4Names {
    packed: ".packed_weight",
    scales: ".weight_inv_scale",
    tail: Nvfp4Tail::InverseUnit {
        global_sf: ".weight_global_sf",
    },
};

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
        if matches!(cfg.adaln, super::AdalnInput::Basis(_)) {
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
                FASTVIDEO,
            )?;
            // The VSA compression gate (`to_gate_compress`, same input as q/k/v).
            out.nvfp4(
                &format!("{e}.attn.gate_proj.weight"),
                vec![(format!("{t}.attn.to_gate_compress"), 0..inner)],
                cfg.hidden,
                FASTVIDEO,
            )?;
            out.nvfp4(
                &format!("{e}.attn.out_proj.weight"),
                vec![(format!("{t}.attn.to_out"), 0..cfg.hidden)],
                inner,
                FASTVIDEO,
            )?;
            let fc_in = format!("{t}.ff.fc_in");
            out.nvfp4(
                &format!("{e}.mlp.fc1.weight"),
                vec![(fc_in.clone(), ffn..2 * ffn), (fc_in, 0..ffn)],
                cfg.hidden,
                FASTVIDEO,
            )?;
            out.nvfp4(
                &format!("{e}.mlp.fc2.weight"),
                vec![(format!("{t}.ff.fc_out"), 0..cfg.hidden)],
                ffn,
                FASTVIDEO,
            )?;
        }
        Ok(out)
    }

    /// FastH3-Omni as the Comfy NVFP4 file ships it ([`DitConfig::fasth3_omni`]).
    /// Tensor names are already the H3 originals (fused `qkv_proj`, `fc1` as
    /// `[gate; up]`), so everything renames to itself except the compression
    /// gate; the five block linears become composites over the Comfy parts,
    /// and the PDD heads become one fused head per step.
    pub fn omni_comfy(cfg: &DitConfig, inner: S) -> Result<Self, FastH3Error<S::Error>> {
        let pdd = cfg
            .pdd
            .ok_or_else(|| FastH3Error::Missing("omni config without PDD heads".into()))?;
        let mut out = Self {
            inner,
            catalog: WeightCatalog::new(),
            plan: HashMap::new(),
        };
        let (inner_dim, ffn) = (cfg.inner_dim(), cfg.ffn_hidden);
        let linears = [
            ("attn.qkv_proj", "attn.qkv_proj", 3 * inner_dim, cfg.hidden),
            (
                "attn.gate_proj",
                "attn.to_gate_compress",
                inner_dim,
                cfg.hidden,
            ),
            ("attn.out_proj", "attn.out_proj", cfg.hidden, inner_dim),
            ("mlp.fc1", "mlp.fc1", 2 * ffn, cfg.hidden),
            ("mlp.fc2", "mlp.fc2", cfg.hidden, ffn),
        ];
        let mut quantized = Vec::new();
        for i in 0..cfg.n_layers {
            for (engine, source, n, k) in linears {
                let prefix = format!("blocks.{i}.{source}");
                out.nvfp4(
                    &format!("blocks.{i}.{engine}.weight"),
                    vec![(prefix.clone(), 0..n)],
                    k,
                    COMFY,
                )?;
                quantized.push(prefix);
            }
        }
        let heads = ["final_layer.video_out", "final_layer.audio_out"];
        let names: Vec<String> = out
            .inner
            .catalog()
            .entries
            .keys()
            .map(|w| w.0.clone())
            .collect();
        for name in names {
            let owned = quantized.iter().any(|p| name.starts_with(&format!("{p}.")));
            if !owned && !heads.iter().any(|h| name.starts_with(h)) {
                out.rename(&name, &name)?;
            }
        }
        for (head, shift) in [
            ("final_layer.video_out", super::SIGMA_SHIFT_VIDEO),
            ("final_layer.audio_out", super::SIGMA_SHIFT_AUDIO),
        ] {
            let w = out.source_entry(&id(format!("{head}.weight")))?.clone();
            let (rows, cols) = (w.shape.0[0], w.shape.0[1]);
            let head_rows = rows / pdd.heads;
            for step in 0..pdd.steps {
                let mix: Vec<(usize, f32)> = pdd
                    .mix(step, shift as f64)
                    .into_iter()
                    .map(|(j, w)| (j, w as f32))
                    .collect();
                for (suffix, c) in [("weight", cols), ("bias", 1)] {
                    let engine = id(format!("{head}.step{step}.{suffix}"));
                    let shape = if c == 1 {
                        vec![head_rows]
                    } else {
                        vec![head_rows, c]
                    };
                    out.catalog.entries.insert(
                        engine.clone(),
                        WeightEntry {
                            offset: 0,
                            size: (head_rows * c * 4) as u64,
                            encoding: Some(StorageEncoding::F32),
                            encoding_label: "F32".into(),
                            shape: Shape(shape),
                        },
                    );
                    out.plan.insert(
                        engine,
                        Plan::Pdd {
                            src: id(format!("{head}.{suffix}")),
                            head_rows,
                            cols: c,
                            mix: mix.clone(),
                        },
                    );
                }
            }
        }
        Ok(out)
    }

    /// FastVideo's NVFP4 H3 video decoder (`decoder-nvfp4.safetensors`, either
    /// tier: `cfg.decoder_layers` = its `student_layers`) under the engine's VAE
    /// names. `inner` = that file unioned with the H3 video VAE, which supplies
    /// what the decoder file lacks (`mask_token`, `latents_mean` / `_std`).
    /// Every other tensor comes from the decoder file: QAD retrained the norms
    /// and projections along with the quantized linears.
    ///
    /// The six block linears become NVFP4 composites: `to_qkv` = `[q; k; v]`
    /// slabs (not the dense file's per-head interleave: the block reads it with
    /// `headwise = false`), and `ff.w1` = `[gate; up]` from the SwiGLU's
    /// `[up; gate]` `net.0.proj`.
    pub fn vae_decoder(
        cfg: &super::video_vae::VideoVaeConfig,
        inner: S,
    ) -> Result<Self, FastH3Error<S::Error>> {
        let mut out = Self {
            inner,
            catalog: WeightCatalog::new(),
            plan: HashMap::new(),
        };
        let (dim, ffn) = (cfg.dim(), cfg.ffn_inner());
        for i in 0..cfg.decoder_layers {
            let p = format!("decoder.transformer_blocks.{i}");
            for t in ["norm1.weight", "norm2.weight", "scale1", "scale2"] {
                out.rename(&format!("{p}.{t}"), &format!("{p}.{t}"))?;
            }
            let qkv = ["to_q", "to_k", "to_v"].map(|s| format!("{p}.attn.{s}"));
            out.nvfp4(
                &format!("{p}.attn.to_qkv.weight"),
                qkv.iter().map(|s| (s.clone(), 0..dim)).collect(),
                dim,
                DECODER,
            )?;
            out.rows(
                &format!("{p}.attn.to_qkv.bias"),
                qkv.iter()
                    .map(|s| (id(format!("{s}.bias")), 0..dim))
                    .collect(),
                1,
            )?;
            let to_out = format!("{p}.attn.to_out.0");
            out.nvfp4(
                &format!("{p}.attn.to_out.weight"),
                vec![(to_out.clone(), 0..dim)],
                dim,
                DECODER,
            )?;
            out.rename(&format!("{p}.attn.to_out.bias"), &format!("{to_out}.bias"))?;
            let up = format!("{p}.ff.net.0.proj");
            out.nvfp4(
                &format!("{p}.ff.w1.weight"),
                vec![(up.clone(), ffn..2 * ffn), (up.clone(), 0..ffn)],
                dim,
                DECODER,
            )?;
            let up_bias = id(format!("{up}.bias"));
            out.rows(
                &format!("{p}.ff.w1.bias"),
                vec![(up_bias.clone(), ffn..2 * ffn), (up_bias, 0..ffn)],
                1,
            )?;
            let down = format!("{p}.ff.net.2");
            out.nvfp4(
                &format!("{p}.ff.w2.weight"),
                vec![(down.clone(), 0..dim)],
                ffn,
                DECODER,
            )?;
            out.rename(&format!("{p}.ff.w2.bias"), &format!("{down}.bias"))?;
        }
        for (engine, source) in [
            ("decoder.x_embedder.weight", "decoder.proj_in.weight"),
            ("decoder.x_embedder.bias", "decoder.proj_in.bias"),
        ] {
            out.rename(engine, source)?;
        }
        for name in [
            "decoder.register_tokens",
            "decoder.mask_token",
            "decoder.norm_out.weight",
            "decoder.norm_out.bias",
            "decoder.proj_out.weight",
            "decoder.proj_out.bias",
            "post_quant_conv.weight",
            "post_quant_conv.bias",
            "latents_mean",
            "latents_std",
        ] {
            out.rename(name, name)?;
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

    /// Row ranges of bf16 `[_, k]` sources, concatenated. `k = 1` takes 1-D
    /// vectors (biases) and yields one.
    fn rows(
        &mut self,
        engine: &str,
        parts: Vec<(WeightId, Range<usize>)>,
        k: usize,
    ) -> Result<(), FastH3Error<S::Error>> {
        for (src, _) in &parts {
            let e = self.source_entry(src)?;
            let row_k = if k == 1 && e.shape.0.len() == 1 {
                Some(&1)
            } else {
                e.shape.0.get(1)
            };
            if e.encoding != Some(StorageEncoding::Bf16) || row_k != Some(&k) {
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
                shape: Shape(if k == 1 { vec![n] } else { vec![n, k] }),
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
        names: Nvfp4Names,
    ) -> Result<(), FastH3Error<S::Error>> {
        for (prefix, r) in &parts {
            let packed = self.source_entry(&id(format!("{prefix}{}", names.packed)))?;
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
        self.plan
            .insert(id(engine), Plan::Nvfp4 { parts, k, names });
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
            Plan::Pdd { .. } => unreachable!("fused PDD heads are computed in `open`"),
            Plan::Rows { parts, row_bytes } => {
                for (src, r) in parts {
                    spans.push(Span {
                        src: index(src.clone()),
                        offset: (r.start * row_bytes) as u64,
                        len: (r.len() * row_bytes) as u64,
                    });
                }
            }
            Plan::Nvfp4 { parts, k, names } => {
                let n: usize = parts.iter().map(|(_, r)| r.len()).sum();
                // Same K throughout, so the same padded block width: a source's
                // 128-row tile rows are byte-identical in size to the output's.
                let tile_row_bytes = (Nvfp4Layout::new(n, *k).blocks_padded() * 128) as u64;
                debug_assert_eq!(k / BLOCK % 4, 0, "scale tiles assumed unpadded along K");
                let part = |p: &str, s: &str| id(format!("{p}{s}"));
                for (prefix, r) in parts {
                    spans.push(Span {
                        src: index(part(prefix, names.packed)),
                        offset: (r.start * k / 2) as u64,
                        len: (r.len() * k / 2) as u64,
                    });
                }
                for (prefix, r) in parts {
                    spans.push(Span {
                        src: index(part(prefix, names.scales)),
                        offset: r.start as u64 / 128 * tile_row_bytes,
                        len: r.len() as u64 / 128 * tile_row_bytes,
                    });
                }
                if let Nvfp4Tail::Stored { alpha, input_sf } = names.tail {
                    for name in [alpha, input_sf] {
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

/// A synthesized tensor read straight from its source byte ranges, then a
/// small computed `tail` (an NVFP4 composite's derived scales, or a whole
/// fused PDD head): the streamed part is never materialized on the host.
pub struct SpanReader<R: WeightReader> {
    readers: Vec<R>,
    spans: Vec<Span>,
    tail: Vec<u8>,
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
        // Whatever lies past the spans is the tail.
        let rest = dst.len() - written;
        if rest > 0 {
            let at = (start - (self.len - self.tail.len() as u64)) as usize;
            dst[written..].copy_from_slice(&self.tail[at..at + rest]);
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
        if let Plan::Pdd {
            src,
            head_rows,
            cols,
            mix,
        } = plan
        {
            let tail = self.fused_head(src, *head_rows, *cols, mix).await?;
            debug_assert_eq!(tail.len() as u64, len);
            return Ok(SpanReader {
                readers: Vec::new(),
                spans: Vec::new(),
                tail,
                len,
            });
        }
        let (readers, spans) = self.spans(plan).await?;
        let tail = match plan {
            Plan::Nvfp4 {
                parts,
                names:
                    Nvfp4Names {
                        tail: Nvfp4Tail::InverseUnit { global_sf },
                        ..
                    },
                ..
            } => self.inverse_unit_tail(parts, global_sf).await?,
            _ => Vec::new(),
        };
        debug_assert_eq!(
            spans.iter().map(|s| s.len).sum::<u64>() + tail.len() as u64,
            len
        );
        Ok(SpanReader {
            readers,
            spans,
            tail,
            len,
        })
    }
}

impl<S: WeightSource> FastH3Source<S> {
    /// An [`Nvfp4Tail::InverseUnit`] composite's tail: per 128-row group, alpha
    /// `= 1 / global_sf` of its part, then the unit activation scales.
    async fn inverse_unit_tail(
        &self,
        parts: &[(String, Range<usize>)],
        global_sf: &str,
    ) -> Result<Vec<u8>, FastH3Error<S::Error>> {
        let mut alpha = Vec::new();
        for (prefix, r) in parts {
            let src = id(format!("{prefix}{global_sf}"));
            let mut b = [0u8; 4];
            self.inner
                .open(&src)
                .await
                .map_err(FastH3Error::Source)?
                .read_at(0, &mut b)
                .await
                .map_err(|e| FastH3Error::Missing(format!("{}: {e:?}", src.0)))?;
            let inv = (1.0 / f32::from_le_bytes(b)).to_le_bytes();
            alpha.extend((0..r.len() / 128).flat_map(|_| inv));
        }
        let unit = 1.0f32.to_le_bytes();
        let groups = alpha.len() / 4;
        alpha.extend((0..groups).flat_map(|_| unit));
        Ok(alpha)
    }

    /// `rows_0 + sum_j w_j rows_j` (the file stores head 0 whole and every
    /// other head as its offset from head 0, so `sum w_j = 1` folds head 0's
    /// share into the leading term), f32 bytes.
    async fn fused_head(
        &self,
        src: &WeightId,
        head_rows: usize,
        cols: usize,
        mix: &[(usize, f32)],
    ) -> Result<Vec<u8>, FastH3Error<S::Error>> {
        if self.source_entry(src)?.encoding != Some(StorageEncoding::F32) {
            return Err(FastH3Error::Missing(format!(
                "{}: PDD heads must be f32",
                src.0
            )));
        }
        let head_bytes = head_rows * cols * 4;
        let mut reader = self.inner.open(src).await.map_err(FastH3Error::Source)?;
        let mut read_head = async |j: usize| -> Result<Vec<f32>, FastH3Error<S::Error>> {
            let mut b = vec![0u8; head_bytes];
            reader
                .read_at((j * head_bytes) as u64, &mut b)
                .await
                .map_err(|e| FastH3Error::Missing(format!("{}: {e:?}", src.0)))?;
            Ok(b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };
        let mut out = read_head(0).await?;
        for &(j, w) in mix.iter().filter(|(j, _)| *j > 0) {
            for (o, v) in out.iter_mut().zip(read_head(j).await?) {
                *o += w * v;
            }
        }
        Ok(out.iter().flat_map(|v| v.to_le_bytes()).collect())
    }
}
