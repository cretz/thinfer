//! Wan / SkyReels-V2 weight-source construction, shared by every host (CLI,
//! web, e2e). Owns the GGUF-over-safetensors union so the recipe exists once.
//! Mirrors `z_image::source::ZImageSource`.
//!
//! The diffusers safetensors bundle already uses the canonical names the DiT
//! loader (`wan/loader.rs`), umT5 encoder (`wan/umt5.rs`), and VAE
//! (`wan/vae.rs`) ask for, so the `Plain` arm needs no rename adapters (unlike
//! Z-Image, which fuses QKV). The `Quantized` arm unions the DiT GGUF and the
//! umT5-XXL GGUF over the safetensors source that still supplies the VAE; each
//! GGUF carries llama.cpp/ComfyUI tensor names mapped back to canonical via
//! [`dit_gguf_renames`] / [`umt5_gguf_renames`].

use std::collections::HashMap;

use thinfer_core::format::gguf::{self, GgufSource};
use thinfer_core::format::safetensors::{self, ShardedSafetensorsSource};
use thinfer_core::format::union::{RenamedSource, UnionError, UnionReader, UnionSource};
use thinfer_core::weight::{FileOpener, WeightCatalog, WeightId, WeightSource};

use crate::wan::dit_block::WanDitConfig;
use crate::wan::umt5::config as umt5_config;

/// Safetensors side: sharded diffusers files, names already canonical.
pub type SafetensorsSide<O> = ShardedSafetensorsSource<O>;
/// One GGUF with its naming divergences mapped to canonical ids.
type GgufSide<O> = RenamedSource<GgufSource<O>>;
/// Quant-variant source: umT5 GGUF over DiT GGUF over the safetensors side
/// (which supplies only the VAE). Namespaces are disjoint, so union order is
/// just lookup order.
type QuantSide<O> = UnionSource<GgufSide<O>, UnionSource<GgufSide<O>, SafetensorsSide<O>>>;

/// The two GGUF openers a quant variant loads from. Both present exactly when
/// the variant has `dit_gguf_role` + `umt5_gguf_role` set.
pub struct GgufOpeners<O> {
    pub dit: O,
    pub umt5: O,
}

/// The one weight source `WanModel` loads from. `Plain` for the fp32/bf16
/// safetensors bundle; `Quantized` when the DiT + umT5 come from GGUFs unioned
/// over the safetensors source that supplies the VAE.
// Both arms boxed: catalogs + rename maps make either source large
// (clippy::large_enum_variant); one model-lifetime alloc each.
pub enum WanSource<O: FileOpener> {
    Plain(Box<SafetensorsSide<O>>),
    Quantized(Box<QuantSide<O>>),
}

#[derive(Debug)]
pub enum SourceOpenError<E: core::fmt::Debug> {
    Safetensors(safetensors::SourceError<E>),
    Gguf(gguf::SourceError<E>),
}

impl<O: FileOpener> WanSource<O> {
    /// Parse and adapt a variant's weight files. `weight_openers` are the
    /// safetensors shards in `VariantFiles::weight_roles` order; `gguf_openers`
    /// is `Some` exactly when the variant has GGUF roles.
    pub async fn open(
        weight_openers: Vec<O>,
        gguf_openers: Option<GgufOpeners<O>>,
    ) -> Result<Self, SourceOpenError<O::Error>> {
        let sharded = ShardedSafetensorsSource::open(weight_openers)
            .await
            .map_err(SourceOpenError::Safetensors)?;
        Ok(match gguf_openers {
            None => Self::Plain(Box::new(sharded)),
            Some(g) => {
                let dit = GgufSource::open(g.dit)
                    .await
                    .map_err(SourceOpenError::Gguf)?;
                let umt5 = GgufSource::open(g.umt5)
                    .await
                    .map_err(SourceOpenError::Gguf)?;
                Self::Quantized(Box::new(UnionSource::new(
                    RenamedSource::with_passthrough(umt5, umt5_gguf_renames()),
                    UnionSource::new(
                        RenamedSource::with_passthrough(
                            dit,
                            dit_gguf_renames(WanDitConfig::fastwan_ti2v_5b().num_layers),
                        ),
                        sharded,
                    ),
                )))
            }
        })
    }
}

impl<O: FileOpener> WeightSource for WanSource<O> {
    // Plain readers ride the union's nested `Fallback` arms so both variants
    // share one reader/error type and callers stay monomorphic over the enum.
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

// ---------------------------------------------------------------------------
// GGUF -> canonical rename maps
// ---------------------------------------------------------------------------
//
// Maps are `original (GGUF) -> renamed (canonical)`, the direction
// `RenamedSource::with_passthrough` consumes. GGUF is DEFERRED for FastWan (the
// active path is `Plain` safetensors); these maps are unexercised until a
// FastWan2.2-TI2V-5B GGUF exists. The structure below is Wan-family-general
// (original-Wan single-file names -> diffusers canonical), so it is reusable
// as the starting point; re-verify the exact tensor names against the real
// FastWan GGUF dump before enabling the quant e2e variant.

/// Original-Wan single-file DiT tensor names -> diffusers canonical (the ids
/// `wan/loader.rs` registers): `self_attn`/`cross_attn`/`norm3`/`ffn.0`/
/// `head.head`/`modulation`/`time_embedding.0`... -> diffusers. `patch_embedding.*`
/// is already canonical and rides the passthrough. `num_layers` is the variant's
/// block count (FastWan2.2-TI2V-5B: 30). FastWan has no fps conditioning, so no
/// `fps_embedding` entry (unlike the SkyReels-DF GGUF this was first shaped from).
pub fn dit_gguf_renames(num_layers: usize) -> HashMap<WeightId, WeightId> {
    // `pair` expands a weight+bias linear into `(orig, canon)` entries; nested
    // `fn` so it doesn't capture `e` (avoids a long-lived mutable borrow).
    fn pair(e: &mut Vec<(String, String)>, o: &str, c: &str) {
        for s in ["weight", "bias"] {
            e.push((format!("{o}.{s}"), format!("{c}.{s}")));
        }
    }
    let mut e: Vec<(String, String)> = Vec::new();

    // Module-level embedders / projections / head.
    pair(
        &mut e,
        "time_embedding.0",
        "condition_embedder.time_embedder.linear_1",
    );
    pair(
        &mut e,
        "time_embedding.2",
        "condition_embedder.time_embedder.linear_2",
    );
    pair(&mut e, "time_projection.1", "condition_embedder.time_proj");
    pair(
        &mut e,
        "text_embedding.0",
        "condition_embedder.text_embedder.linear_1",
    );
    pair(
        &mut e,
        "text_embedding.2",
        "condition_embedder.text_embedder.linear_2",
    );
    pair(&mut e, "head.head", "proj_out");
    // `head.modulation` is the model-level scale_shift_table (weight-only).
    e.push(("head.modulation".into(), "scale_shift_table".into()));

    // Per-block: original-Wan -> diffusers (attn1=self, attn2=cross, norm3 ->
    // norm2, ffn.0/2 -> ffn.net.0.proj/net.2, modulation -> scale_shift_table).
    for i in 0..num_layers {
        let b = format!("blocks.{i}");
        for (gg, df) in [("self_attn", "attn1"), ("cross_attn", "attn2")] {
            for (gq, dq) in [
                ("q", "to_q"),
                ("k", "to_k"),
                ("v", "to_v"),
                ("o", "to_out.0"),
            ] {
                pair(&mut e, &format!("{b}.{gg}.{gq}"), &format!("{b}.{df}.{dq}"));
            }
            for nq in ["norm_q", "norm_k"] {
                e.push((
                    format!("{b}.{gg}.{nq}.weight"),
                    format!("{b}.{df}.{nq}.weight"),
                ));
            }
        }
        pair(&mut e, &format!("{b}.norm3"), &format!("{b}.norm2"));
        pair(
            &mut e,
            &format!("{b}.ffn.0"),
            &format!("{b}.ffn.net.0.proj"),
        );
        pair(&mut e, &format!("{b}.ffn.2"), &format!("{b}.ffn.net.2"));
        e.push((format!("{b}.modulation"), format!("{b}.scale_shift_table")));
    }
    e.into_iter()
        .map(|(o, c)| (WeightId(o), WeightId(c)))
        .collect()
}

/// `city96/umt5-xxl-encoder-gguf` tensor names -> the canonical umT5 ids
/// (`encoder.block.{i}...`) the encoder registers.
pub fn umt5_gguf_renames() -> HashMap<WeightId, WeightId> {
    // llama.cpp T5 naming (`enc.blk.{i}.attn_q.weight`, `ffn_gate`/`ffn_up`/
    // `ffn_down`, `attn_rel_b`, `token_embd`, `enc.output_norm`) -> the HF umT5
    // ids the encoder reads. Gated-gelu: `wi_0` is the gelu branch (ffn_gate),
    // `wi_1` the linear branch (ffn_up). relpos bias + embed are read directly
    // (not residency-registered), so they must be in this map too. Verified
    // against the Q4_K_M dump (242 tensors = 24 * 10 + 2; relpos on every
    // block).
    let mut e: Vec<(String, String)> = vec![
        ("token_embd.weight".into(), "shared.weight".into()),
        (
            "enc.output_norm.weight".into(),
            "encoder.final_layer_norm.weight".into(),
        ),
    ];
    const SITES: [(&str, &str); 9] = [
        ("attn_norm", "layer.0.layer_norm"),
        ("attn_q", "layer.0.SelfAttention.q"),
        ("attn_k", "layer.0.SelfAttention.k"),
        ("attn_v", "layer.0.SelfAttention.v"),
        ("attn_o", "layer.0.SelfAttention.o"),
        (
            "attn_rel_b",
            "layer.0.SelfAttention.relative_attention_bias",
        ),
        ("ffn_norm", "layer.1.layer_norm"),
        ("ffn_gate", "layer.1.DenseReluDense.wi_0"),
        ("ffn_up", "layer.1.DenseReluDense.wi_1"),
    ];
    for i in 0..umt5_config::N_LAYERS {
        for (gg, hf) in SITES {
            e.push((
                format!("enc.blk.{i}.{gg}.weight"),
                format!("encoder.block.{i}.{hf}.weight"),
            ));
        }
        // ffn_down -> wo (kept separate so the SITES table stays weight-only and
        // uniform).
        e.push((
            format!("enc.blk.{i}.ffn_down.weight"),
            format!("encoder.block.{i}.layer.1.DenseReluDense.wo.weight"),
        ));
    }
    e.into_iter()
        .map(|(o, c)| (WeightId(o), WeightId(c)))
        .collect()
}
