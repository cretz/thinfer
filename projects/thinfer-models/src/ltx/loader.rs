//! LTX text-encoder loading helpers: the Gemma-3 GGUF -> HF tensor-name map and
//! the `(1 + weight)` RMSNorm bake.
//!
//! Gemma3 RMSNorm is `x_normed * (1 + weight)` (HF `Gemma3RMSNorm`), NOT the
//! engine's standard `x_normed * weight`. Rather than fork every rmsnorm WGSL
//! variant, [`UnitOffsetSource`] adds 1.0 to each element of the Gemma norm
//! tensors AT LOAD, so the standard `op_rmsnorm` computes the Gemma form. The
//! norm tensors are F32 in the GGUF, so the raw bytes are f32 lanes. (Only the
//! Gemma encoder norms get the bake; the connector + DiT use standard rms.)

use std::collections::{HashMap, HashSet};

use thinfer_core::weight::{WeightCatalog, WeightId, WeightReader, WeightSource};

use super::gemma::N_LAYERS;

/// Per-layer GGUF (`gemma3` native) -> HF (`model.layers.{i}.*`) site map.
const SITES: &[(&str, &str)] = &[
    ("attn_norm.weight", "input_layernorm.weight"),
    ("attn_q.weight", "self_attn.q_proj.weight"),
    ("attn_k.weight", "self_attn.k_proj.weight"),
    ("attn_v.weight", "self_attn.v_proj.weight"),
    ("attn_output.weight", "self_attn.o_proj.weight"),
    ("attn_q_norm.weight", "self_attn.q_norm.weight"),
    ("attn_k_norm.weight", "self_attn.k_norm.weight"),
    (
        "post_attention_norm.weight",
        "post_attention_layernorm.weight",
    ),
    ("ffn_norm.weight", "pre_feedforward_layernorm.weight"),
    ("ffn_gate.weight", "mlp.gate_proj.weight"),
    ("ffn_up.weight", "mlp.up_proj.weight"),
    ("ffn_down.weight", "mlp.down_proj.weight"),
    ("post_ffw_norm.weight", "post_feedforward_layernorm.weight"),
];

/// HF site suffixes whose Gemma3RMSNorm weight needs the `+1` bake.
const NORM_SUFFIXES: &[&str] = &[
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
];

/// GGUF (`gemma3` native) -> HF (`model.*`) tensor-name map. Re-keys the encoder
/// source so the shared embed gather (`model.embed_tokens.weight`) + the HF-named
/// handles work unchanged. `output_norm` -> `model.norm`.
pub fn gemma_gguf_renames() -> HashMap<WeightId, WeightId> {
    let mut m = HashMap::new();
    m.insert(
        WeightId("token_embd.weight".into()),
        WeightId("model.embed_tokens.weight".into()),
    );
    m.insert(
        WeightId("output_norm.weight".into()),
        WeightId("model.norm.weight".into()),
    );
    for i in 0..N_LAYERS {
        for (g, h) in SITES {
            m.insert(
                WeightId(format!("blk.{i}.{g}")),
                WeightId(format!("model.layers.{i}.{h}")),
            );
        }
    }
    m
}

/// HF names (post-rename) of every Gemma norm weight that takes the `+1` bake:
/// the 6 per-layer norms + the final `model.norm`.
pub fn gemma_norm_offset_ids() -> HashSet<WeightId> {
    let mut s = HashSet::new();
    s.insert(WeightId("model.norm.weight".into()));
    for i in 0..N_LAYERS {
        for suf in NORM_SUFFIXES {
            s.insert(WeightId(format!("model.layers.{i}.{suf}")));
        }
    }
    s
}

/// Wraps a `WeightSource` so reads of the configured ids have `+1.0` added to
/// every f32 lane. See the module note: this realizes Gemma's `x * (1 + w)`
/// RMSNorm via the standard `x * w` kernel. Non-listed ids pass through verbatim.
pub struct UnitOffsetSource<S> {
    inner: S,
    offset_ids: HashSet<WeightId>,
}

impl<S: WeightSource> UnitOffsetSource<S> {
    pub fn new(inner: S, offset_ids: HashSet<WeightId>) -> Self {
        Self { inner, offset_ids }
    }
}

impl<S: WeightSource> WeightSource for UnitOffsetSource<S> {
    type Reader = UnitOffsetReader<S::Reader>;
    type Error = S::Error;

    fn catalog(&self) -> &WeightCatalog {
        self.inner.catalog()
    }

    async fn open(&self, id: &WeightId) -> Result<Self::Reader, Self::Error> {
        let offset = self.offset_ids.contains(id);
        Ok(UnitOffsetReader {
            inner: self.inner.open(id).await?,
            offset,
        })
    }
}

/// Reader that adds 1.0 to each f32 lane of every byte range it returns, iff the
/// tensor was flagged. Norm tensors are F32 and read on 4-byte boundaries, so
/// the per-lane add is exact regardless of how the upload path chunks the read.
pub struct UnitOffsetReader<R> {
    inner: R,
    offset: bool,
}

impl<R: WeightReader> WeightReader for UnitOffsetReader<R> {
    type Error = R::Error;

    fn len(&self) -> u64 {
        self.inner.len()
    }

    async fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), Self::Error> {
        self.inner.read_at(offset, dst).await?;
        if self.offset {
            debug_assert_eq!(offset % 4, 0, "unit-offset norm read must be f32-aligned");
            debug_assert_eq!(dst.len() % 4, 0, "unit-offset norm read must be whole f32s");
            for lane in dst.chunks_exact_mut(4) {
                let v = f32::from_le_bytes([lane[0], lane[1], lane[2], lane[3]]) + 1.0;
                lane.copy_from_slice(&v.to_le_bytes());
            }
        }
        Ok(())
    }

    fn will_read(&mut self, offset: u64, len: u64) {
        self.inner.will_read(offset, len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    #[test]
    fn renames_cover_layers_and_norm_ids_align() {
        let m = gemma_gguf_renames();
        assert_eq!(
            m[&WeightId("token_embd.weight".into())].0,
            "model.embed_tokens.weight"
        );
        assert_eq!(
            m[&WeightId("output_norm.weight".into())].0,
            "model.norm.weight"
        );
        assert_eq!(
            m[&WeightId("blk.0.post_ffw_norm.weight".into())].0,
            "model.layers.0.post_feedforward_layernorm.weight"
        );
        // 2 top-level + N_LAYERS * 13 per-layer.
        assert_eq!(m.len(), 2 + N_LAYERS * SITES.len());

        // Every offset id is a value in the rename map (so the bake actually
        // lands on a real registered tensor).
        let hf_names: HashSet<&str> = m.values().map(|v| v.0.as_str()).collect();
        for id in gemma_norm_offset_ids() {
            assert!(hf_names.contains(id.0.as_str()), "stray offset id {id:?}");
        }
        // 6 norms/layer + final norm.
        assert_eq!(
            gemma_norm_offset_ids().len(),
            1 + N_LAYERS * NORM_SUFFIXES.len()
        );
    }

    // Minimal in-memory source to exercise the +1 byte transform.
    struct MemReader {
        bytes: Vec<u8>,
    }
    impl WeightReader for MemReader {
        type Error = Infallible;
        fn len(&self) -> u64 {
            self.bytes.len() as u64
        }
        async fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), Infallible> {
            let o = offset as usize;
            dst.copy_from_slice(&self.bytes[o..o + dst.len()]);
            Ok(())
        }
    }
    struct MemSource {
        catalog: WeightCatalog,
        bytes: Vec<u8>,
    }
    impl WeightSource for MemSource {
        type Reader = MemReader;
        type Error = Infallible;
        fn catalog(&self) -> &WeightCatalog {
            &self.catalog
        }
        async fn open(&self, _id: &WeightId) -> Result<MemReader, Infallible> {
            Ok(MemReader {
                bytes: self.bytes.clone(),
            })
        }
    }

    #[test]
    fn unit_offset_adds_one_to_flagged_only() {
        let vals = [0.0f32, 1.5, -2.25, 10.0];
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let id = WeightId("model.norm.weight".into());
        let src = MemSource {
            catalog: WeightCatalog::default(),
            bytes: bytes.clone(),
        };
        let mut offset_ids = HashSet::new();
        offset_ids.insert(id.clone());
        let wrapped = UnitOffsetSource::new(src, offset_ids);

        use futures::FutureExt;
        let read = |id: &WeightId| {
            let mut r = wrapped.open(id).now_or_never().unwrap().unwrap();
            let mut buf = vec![0u8; bytes.len()];
            r.read_at(0, &mut buf).now_or_never().unwrap().unwrap();
            buf.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<_>>()
        };
        // flagged -> +1
        assert_eq!(read(&id), vec![1.0, 2.5, -1.25, 11.0]);
        // not flagged -> passthrough
        assert_eq!(read(&WeightId("other".into())), vals.to_vec());
    }
}
