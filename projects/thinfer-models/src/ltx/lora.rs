//! Folds the LTX-2.3 distilled LoRA into the Sulphur-2 dev DiT at load.
//!
//! The published Sulphur GGUF is `sulphur_dev` -- the BASE (full) checkpoint, not
//! the distilled one. The distilled fast path (8-step, CFG-free) only converges
//! when the distill LoRA is applied to the dev base: the Sulphur ComfyUI
//! `t2v distilled` workflow loads dev + the distill LoRA at strength 1.0 with
//! CFG=1, whereas the `t2v base` workflow (no LoRA) needs ~50 steps + CFG 3.6.
//! Running dev un-distilled through the 8-step CFG-free sampler is what produced
//! the undercooked / blurry eyeball. This module folds the LoRA so the existing
//! distilled pipeline converges.
//!
//! LoRA = `SulphurAI/Sulphur-2-base/distill_loras/ltx-2.3-22b-distilled-lora-1.1
//! _fro90_ceil72_condsafe.safetensors` (bf16, ai-toolkit keys
//! `diffusion_model.{X}.lora_{A,B}.weight`, base key `{X}.weight`). It carries NO
//! `.alpha` tensors, so the ComfyUI convention is `alpha = rank` -> scale 1.0
//! (the "rerank" pass already baked the original alpha/rank into the A/B values);
//! we apply a plain `B @ A`. The checkpoint was rank-reduced per-tensor by
//! Frobenius retention, so the rank VARIES per site (attn ~36, ff ~45-72,
//! gate_logits ~14-27) -- we read it from `lora_A.shape[0]`, never assume it.
//!
//! Sites are discovered from the LoRA itself: every key whose base `{X}.weight`
//! exists and whose `lora_B` is not all-zero is folded. In this checkpoint the
//! cross-attn (`attn2`, `audio_attn2`, `audio_to_video_attn`,
//! `video_to_audio_attn`) and all adaLN modules are zeroed (so they are skipped),
//! leaving video/audio self-attn (`attn1`/`audio_attn1`, incl `to_gate_logits`),
//! the per-modality FFNs, and the patchify/proj_out projections. Auto-discovery
//! keeps this honest if a future LoRA revision touches a different set.
//!
//! Each site is dequantized to f32, the LoRA delta added, then re-encoded with
//! the SAME `[N, K]` shape and an exact (un-padded) byte length, so the folded
//! source is a byte-shape drop-in -- the DiT loader (`ltx::dit::register_*`) and
//! the residency upload path see a normal tensor. Quant matmul sites (attn/ff)
//! re-encode to Q8_0 (what `register_attn`/`register_ff` already hint, so it is a
//! drop-in for the Q8 base AND the q4 file's Q4_K/Q6_K, avoiding Q4_K alignment +
//! Q6_K requant); the bf16 patchify/proj_out are preserved as bf16. See
//! [`fold_out_enc`]. Folded bytes are computed once per tensor and cached in RAM
//! (residency re-acquires across denoise steps; recomputing `B @ A` each time
//! would be catastrophic).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use thinfer_core::quant::{QuantKind, dequantize_row, quantize_row};
use thinfer_core::tensor::StorageEncoding;
use thinfer_core::weight::{
    DecodeError, WeightCatalog, WeightId, WeightReader, WeightSource, decode_into,
};

/// One fold target: the base tensor, its LoRA `A`/`B` ids, `[N, K]` dims, and the
/// per-tensor rank (read from `lora_A.shape[0]`, not assumed).
#[derive(Clone, Debug)]
pub struct FoldSpec {
    base: WeightId,
    a: WeightId,
    b: WeightId,
    n: usize,
    k: usize,
    rank: usize,
}

#[derive(Debug)]
pub enum FoldError<BE: core::fmt::Debug, LE: core::fmt::Debug> {
    Base(BE),
    Lora(LE),
    Decode(DecodeError),
    /// A reader-side `read_at` failed (reader error types differ from the source
    /// error types, so they ride a formatted string).
    Read(String),
    /// A site was mis-shaped or an encoding was missing/unsupported.
    Shape(String),
}

fn id(s: impl Into<String>) -> WeightId {
    WeightId(s.into())
}

/// Encoding the fold emits for a site, given the base encoding. Quant matmul
/// sites (attn/ff) re-encode to Q8_0 -- the DiT loader (`register_attn`/
/// `register_ff`) hints Q8_0 and the dequant-once/DP4A path reads it, so this is
/// a drop-in for ANY quant base kind (Q8_0 or the q4 file's Q4_K/Q6_K) and dodges
/// Q4_K's 256-elem alignment + Q6_K requant. Non-quant sites (the bf16
/// patchify/proj_out, registered with no quant hint) are preserved as-is.
fn fold_out_enc(base: StorageEncoding) -> StorageEncoding {
    match base {
        StorageEncoding::Quant(_) => StorageEncoding::Quant(QuantKind::Q8_0),
        other => other,
    }
}

/// Exact tight byte length for `elems` values in `enc` (no GGUF tensor padding):
/// the size the folded catalog entry advertises and `compute_fold` emits.
fn exact_bytes(enc: StorageEncoding, elems: usize) -> Result<usize, String> {
    Ok(match enc {
        StorageEncoding::Quant(kind) => kind.bytes_for_elements(elems as u64) as usize,
        StorageEncoding::Bf16 | StorageEncoding::F16 => elems * 2,
        StorageEncoding::F32 => elems * 4,
        other => return Err(format!("unsupported fold encoding {other:?}")),
    })
}

/// Discover the fold set from the LoRA catalog. For each `diffusion_model.{X}
/// .lora_A.weight` whose `lora_B` and base `{X}.weight` both exist, build a spec
/// -- unless `lora_B` is entirely zero (a deliberately-zeroed site, e.g. the
/// cross-attn and adaLN modules in this checkpoint), which is skipped.
pub async fn discover_specs<B: WeightSource, L: WeightSource>(
    base: &B,
    lora: &L,
) -> Result<Vec<FoldSpec>, FoldError<B::Error, L::Error>> {
    let mut specs = Vec::new();
    // Deterministic order: sort the lora keys so the fold (and its logs) are
    // reproducible across runs.
    let mut a_keys: Vec<&String> = lora
        .catalog()
        .entries
        .keys()
        .map(|k| &k.0)
        .filter(|k| k.starts_with("diffusion_model.") && k.ends_with(".lora_A.weight"))
        .collect();
    a_keys.sort();

    for a_key in a_keys {
        // diffusion_model.{X}.lora_A.weight -> X
        let x = a_key
            .strip_prefix("diffusion_model.")
            .and_then(|s| s.strip_suffix(".lora_A.weight"))
            .expect("filtered above");
        let a_id = id(a_key.clone());
        let b_id = id(format!("diffusion_model.{x}.lora_B.weight"));
        let base_id = id(format!("{x}.weight"));

        let (Some(ae), Some(be), Some(base_e)) = (
            lora.catalog().get(&a_id),
            lora.catalog().get(&b_id),
            base.catalog().get(&base_id),
        ) else {
            // No matching lora_B or no such base tensor -> not a foldable site.
            continue;
        };

        // base [N, K]; lora_A [rank, K]; lora_B [N, rank].
        if base_e.shape.0.len() != 2 {
            continue;
        }
        let (n, k) = (base_e.shape.0[0], base_e.shape.0[1]);
        let rank = ae.shape.0[0];
        if ae.shape.0 != vec![rank, k] || be.shape.0 != vec![n, rank] {
            return Err(FoldError::Shape(format!(
                "lora site {x}: A={:?} B={:?} vs base [{n}, {k}]",
                ae.shape.0, be.shape.0
            )));
        }

        // Skip a zeroed site (no contribution): read the small B tensor and test.
        let b_bytes = {
            let mut r = lora.open(&b_id).await.map_err(FoldError::Lora)?;
            let len = r.len() as usize;
            let mut buf = vec![0u8; len];
            r.read_at(0, &mut buf)
                .await
                .map_err(|e| FoldError::Read(format!("lora {b_id:?}: {e:?}")))?;
            buf
        };
        let mut b_f32 = vec![0f32; n * rank];
        let benc = be
            .encoding
            .ok_or_else(|| FoldError::Shape(format!("lora {b_id:?} has no encoding")))?;
        decode_into(benc, &b_bytes, bytemuck::cast_slice_mut(&mut b_f32))
            .map_err(FoldError::Decode)?;
        if b_f32.iter().all(|&v| v == 0.0) {
            continue;
        }

        specs.push(FoldSpec {
            base: base_id,
            a: a_id,
            b: b_id,
            n,
            k,
            rank,
        });
    }
    Ok(specs)
}

/// Wraps the base DiT GGUF + one or more distill/content LoRAs, serving
/// `re-encode(dequant(base) + Σ_i strength_i · (B_i @ A_i))` at the discovered
/// sites (encoding + shape preserved) and passing every other tensor straight
/// through. The Sulphur distilled ComfyUI workflow STACKS LoRAs at different
/// strengths (e.g. `condsafe @ 0.7` + `distilled-lora-384 @ 0.5`), so the fold
/// accumulates all of them onto each base before re-encoding. See the module note.
pub struct LoraFoldSource<B: WeightSource, L: WeightSource> {
    base: B,
    loras: Vec<L>,
    strengths: Vec<f32>,
    catalog: WeightCatalog,
    /// base id -> the `(lora index, spec)` pairs folded into it (a site may be
    /// touched by several stacked LoRAs).
    folds: HashMap<WeightId, Vec<(usize, FoldSpec)>>,
    /// Compute-once cache of folded bytes, keyed by base id.
    cache: Mutex<HashMap<WeightId, Arc<[u8]>>>,
}

impl<B: WeightSource, L: WeightSource> LoraFoldSource<B, L> {
    /// Build the folded source from one or more `(lora, strength, specs)` stacks
    /// (each from [`discover_specs`] on that lora). The folded catalog republishes
    /// each touched site with its `[N, K]` shape, the [`fold_out_enc`] encoding,
    /// and an EXACT (un-padded) byte size, so the recomputed bytes the reader
    /// serves match the length. A site folded by several LoRAs is republished once.
    pub fn new(base: B, stacks: Vec<(L, f32, Vec<FoldSpec>)>) -> Result<Self, String> {
        let mut catalog = WeightCatalog::new();
        for (k, v) in &base.catalog().entries {
            catalog.entries.insert(k.clone(), v.clone());
        }
        let mut folds: HashMap<WeightId, Vec<(usize, FoldSpec)>> = HashMap::new();
        let mut loras = Vec::with_capacity(stacks.len());
        let mut strengths = Vec::with_capacity(stacks.len());
        for (idx, (lora, strength, specs)) in stacks.into_iter().enumerate() {
            for spec in specs {
                let entry = catalog
                    .entries
                    .get_mut(&spec.base)
                    .ok_or_else(|| format!("fold base {:?} absent from catalog", spec.base))?;
                let enc = entry
                    .encoding
                    .ok_or_else(|| format!("fold base {:?} has no encoding", spec.base))?;
                let out_enc = fold_out_enc(enc);
                entry.offset = 0;
                entry.encoding = Some(out_enc);
                entry.encoding_label = match out_enc {
                    StorageEncoding::Quant(k) => k.hint().to_string(),
                    _ => entry.encoding_label.clone(),
                };
                entry.size = exact_bytes(out_enc, spec.n * spec.k)? as u64;
                folds.entry(spec.base.clone()).or_default().push((idx, spec));
            }
            loras.push(lora);
            strengths.push(strength);
        }
        Ok(Self {
            base,
            loras,
            strengths,
            catalog,
            folds,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Number of distinct base sites that will be folded.
    pub fn fold_count(&self) -> usize {
        self.folds.len()
    }

    async fn lora_to_f32(
        &self,
        lora: &L,
        id: &WeightId,
        expect: usize,
    ) -> Result<Vec<f32>, FoldError<B::Error, L::Error>> {
        let mut r = lora.open(id).await.map_err(FoldError::Lora)?;
        let len = r.len() as usize;
        let mut bytes = vec![0u8; len];
        r.read_at(0, &mut bytes)
            .await
            .map_err(|e| FoldError::Read(format!("lora {id:?}: {e:?}")))?;
        let enc = lora
            .catalog()
            .get(id)
            .and_then(|e| e.encoding)
            .ok_or_else(|| FoldError::Shape(format!("lora {id:?} has no encoding")))?;
        let mut out = vec![0f32; expect];
        let written = decode_into(enc, &bytes, bytemuck::cast_slice_mut(&mut out))
            .map_err(FoldError::Decode)?;
        if written != expect * 4 {
            return Err(FoldError::Shape(format!(
                "lora {id:?}: decoded {written} bytes, expected {}",
                expect * 4
            )));
        }
        Ok(out)
    }

    /// `re-encode(dequant(base) + Σ_i strength_i · (B_i @ A_i))` for one base
    /// site, accumulating every stacked LoRA that touches it, in the base's
    /// encoding. `specs` are the `(lora index, spec)` pairs for this base.
    async fn compute_fold(
        &self,
        base_id: &WeightId,
        specs: &[(usize, FoldSpec)],
    ) -> Result<Arc<[u8]>, FoldError<B::Error, L::Error>> {
        let (n, k) = {
            let s = &specs[0].1;
            (s.n, s.k)
        };
        // base -> f32 [N*K].
        let base_bytes = {
            let mut r = self.base.open(base_id).await.map_err(FoldError::Base)?;
            let len = r.len() as usize;
            let mut buf = vec![0u8; len];
            r.read_at(0, &mut buf)
                .await
                .map_err(|e| FoldError::Read(format!("base {base_id:?}: {e:?}")))?;
            buf
        };
        let base_enc = self
            .base
            .catalog()
            .get(base_id)
            .and_then(|e| e.encoding)
            .ok_or_else(|| FoldError::Shape(format!("base {base_id:?} has no encoding")))?;
        let mut acc = vec![0f32; n * k];
        match base_enc {
            StorageEncoding::Quant(kind) => dequantize_row(kind, &base_bytes, &mut acc),
            StorageEncoding::Bf16 | StorageEncoding::F16 | StorageEncoding::F32 => {
                decode_into(base_enc, &base_bytes, bytemuck::cast_slice_mut(&mut acc))
                    .map_err(FoldError::Decode)?;
            }
            other => {
                return Err(FoldError::Shape(format!(
                    "base encoding {other:?} unsupported"
                )));
            }
        }

        // Each stacked LoRA: A [rank, K], B [N, rank] -> f32, then
        // acc += strength · (B @ A).
        for (idx, spec) in specs {
            debug_assert_eq!((spec.n, spec.k), (n, k), "stacked fold dims must match base");
            let lora = &self.loras[*idx];
            let a = self.lora_to_f32(lora, &spec.a, spec.rank * k).await?;
            let b = self.lora_to_f32(lora, &spec.b, n * spec.rank).await?;
            fold_add(&mut acc, &b, &a, k, spec.rank, self.strengths[*idx]);
        }

        // f32 -> fold output encoding (quant sites -> Q8_0, bf16 sites preserved).
        let out = encode_rows(fold_out_enc(base_enc), &acc, n, k).map_err(FoldError::Shape)?;
        Ok(Arc::from(out.into_boxed_slice()))
    }
}

/// `acc[n,:] += strength * (B[n,:] @ A)` where `B` is `[N, rank]`, `A` is
/// `[rank, K]`, `acc` is `[N, K]` (row-major). Each output row is independent;
/// fan rows across threads on native (the full-DiT fold is billions of FLOPs).
/// Skips zero `B[n,r]` so a low-magnitude row is cheap. `strength` is the
/// ComfyUI `LoraLoaderModelOnly` weight (the distilled workflow stacks the
/// distill LoRAs below 1.0).
fn fold_add(acc: &mut [f32], b: &[f32], a: &[f32], k: usize, rank: usize, strength: f32) {
    let row = |acc_row: &mut [f32], b_row: &[f32]| {
        for (r, &bnr) in b_row.iter().enumerate() {
            if bnr == 0.0 {
                continue;
            }
            let scaled = strength * bnr;
            let a_row = &a[r * k..(r + 1) * k];
            for (dst, &av) in acc_row.iter_mut().zip(a_row) {
                *dst += scaled * av;
            }
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        let rows = acc.len() / k;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(rows.max(1));
        if threads > 1 {
            let chunk_rows = rows.div_ceil(threads);
            std::thread::scope(|s| {
                for (acc_chunk, b_chunk) in acc
                    .chunks_mut(chunk_rows * k)
                    .zip(b.chunks(chunk_rows * rank))
                {
                    let row = &row;
                    s.spawn(move || {
                        for (ar, br) in acc_chunk.chunks_mut(k).zip(b_chunk.chunks_exact(rank)) {
                            row(ar, br);
                        }
                    });
                }
            });
            return;
        }
    }
    for (ar, br) in acc.chunks_mut(k).zip(b.chunks_exact(rank)) {
        row(ar, br);
    }
}

/// Encode `acc` `[N, K]` (row-major) to `enc`'s bytes, fanning rows across
/// threads. Quant encodings stay row-local (K is block-aligned for the GGUF
/// tensors). Bf16/F16/F32 narrow element-wise (RNE for bf16, matching every
/// other f32->bf16 path in core).
fn encode_rows(enc: StorageEncoding, acc: &[f32], n: usize, k: usize) -> Result<Vec<u8>, String> {
    let row_bytes = exact_bytes(enc, k)?;
    let mut out = vec![0u8; n * row_bytes];
    let encode_row = |src: &[f32], dst: &mut [u8]| match enc {
        StorageEncoding::Quant(kind) => {
            let mut buf = Vec::with_capacity(dst.len());
            quantize_row(kind, src, &mut buf);
            dst.copy_from_slice(&buf);
        }
        StorageEncoding::Bf16 => {
            for (v, o) in src.iter().zip(dst.chunks_exact_mut(2)) {
                o.copy_from_slice(&half::bf16::from_f32(*v).to_le_bytes());
            }
        }
        StorageEncoding::F16 => {
            for (v, o) in src.iter().zip(dst.chunks_exact_mut(2)) {
                o.copy_from_slice(&half::f16::from_f32(*v).to_le_bytes());
            }
        }
        StorageEncoding::F32 => {
            for (v, o) in src.iter().zip(dst.chunks_exact_mut(4)) {
                o.copy_from_slice(&v.to_le_bytes());
            }
        }
        _ => unreachable!("exact_bytes rejected other encodings"),
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        let threads = std::thread::available_parallelism()
            .map(|t| t.get())
            .unwrap_or(1)
            .min(n.max(1));
        if threads > 1 {
            let chunk_rows = n.div_ceil(threads);
            std::thread::scope(|s| {
                for (src_chunk, dst_chunk) in acc
                    .chunks(chunk_rows * k)
                    .zip(out.chunks_mut(chunk_rows * row_bytes))
                {
                    let encode_row = &encode_row;
                    s.spawn(move || {
                        for (sr, dr) in src_chunk.chunks(k).zip(dst_chunk.chunks_mut(row_bytes)) {
                            encode_row(sr, dr);
                        }
                    });
                }
            });
            return Ok(out);
        }
    }
    for (sr, dr) in acc.chunks(k).zip(out.chunks_mut(row_bytes)) {
        encode_row(sr, dr);
    }
    Ok(out)
}

impl<B: WeightSource, L: WeightSource> WeightSource for LoraFoldSource<B, L> {
    type Reader = LoraFoldReader<B>;
    type Error = FoldError<B::Error, L::Error>;

    fn catalog(&self) -> &WeightCatalog {
        &self.catalog
    }

    async fn open(&self, id: &WeightId) -> Result<Self::Reader, Self::Error> {
        let Some(specs) = self.folds.get(id) else {
            let r = self.base.open(id).await.map_err(FoldError::Base)?;
            return Ok(LoraFoldReader::Base(r));
        };
        if let Some(bytes) = self.cache.lock().expect("lora cache").get(id).cloned() {
            return Ok(LoraFoldReader::Folded(VecReader::new(bytes)));
        }
        let bytes = self.compute_fold(id, specs).await?;
        self.cache
            .lock()
            .expect("lora cache")
            .insert(id.clone(), Arc::clone(&bytes));
        Ok(LoraFoldReader::Folded(VecReader::new(bytes)))
    }
}

/// Reader over an in-memory folded tensor.
pub struct VecReader {
    bytes: Arc<[u8]>,
}

impl VecReader {
    fn new(bytes: Arc<[u8]>) -> Self {
        Self { bytes }
    }
}

impl WeightReader for VecReader {
    type Error = std::convert::Infallible;
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }
    async fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), Self::Error> {
        let off = offset as usize;
        dst.copy_from_slice(&self.bytes[off..off + dst.len()]);
        Ok(())
    }
}

pub enum LoraFoldReader<B: WeightSource> {
    Base(B::Reader),
    Folded(VecReader),
}

impl<B: WeightSource> WeightReader for LoraFoldReader<B> {
    type Error = FoldReaderError<<B::Reader as WeightReader>::Error>;
    fn len(&self) -> u64 {
        match self {
            Self::Base(r) => r.len(),
            Self::Folded(r) => r.len(),
        }
    }
    async fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), Self::Error> {
        match self {
            Self::Base(r) => r.read_at(offset, dst).await.map_err(FoldReaderError::Base),
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

#[derive(Debug)]
pub enum FoldReaderError<BE: core::fmt::Debug> {
    Base(BE),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;
    use thinfer_core::quant::QuantKind;
    use thinfer_core::tensor::Shape;
    use thinfer_core::weight::WeightEntry;

    fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect()
    }

    #[test]
    fn fold_add_is_b_times_a() {
        // acc[2,3] += B[2,1] @ A[1,3]; rank 1, scale 1.
        let mut acc = vec![0f32; 6];
        let b = [1.0f32, 2.0]; // [N=2, rank=1]
        let a = [1.0f32, 2.0, 3.0]; // [rank=1, K=3]
        fold_add(&mut acc, &b, &a, 3, 1, 1.0);
        assert_eq!(acc, vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0]);
        // Half strength halves the delta.
        let mut acc = vec![0f32; 6];
        fold_add(&mut acc, &b, &a, 3, 1, 0.5);
        assert_eq!(acc, vec![0.5, 1.0, 1.5, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn fold_add_rank2_accumulates() {
        // rank 2: B[1,2]=[2,3], A[2,3]=[[1,0,1],[0,1,1]] -> [2,3,5].
        let mut acc = vec![10.0f32, 0.0, 0.0]; // base row added onto
        let b = [2.0f32, 3.0];
        let a = [1.0f32, 0.0, 1.0, 0.0, 1.0, 1.0];
        fold_add(&mut acc, &b, &a, 3, 2, 1.0);
        assert_eq!(acc, vec![12.0, 3.0, 5.0]);
    }

    #[test]
    fn encode_rows_bf16_roundtrips() {
        // bf16-representable values survive f32 -> bf16 -> f32 exactly.
        let vals = [1.0f32, -2.5, 0.5, 0.25, 0.0, 8.0];
        let bytes = encode_rows(StorageEncoding::Bf16, &vals, 2, 3).unwrap();
        assert_eq!(bytes.len(), 6 * 2);
        let mut back = vec![0f32; 6];
        decode_into(
            StorageEncoding::Bf16,
            &bytes,
            bytemuck::cast_slice_mut(&mut back),
        )
        .unwrap();
        assert_eq!(back, vals.to_vec());
    }

    #[test]
    fn encode_rows_q8_0_size_and_recovery() {
        // K=64 (two Q8_0 blocks/row), 2 rows. Round-trip stays close.
        let k = 64;
        let vals: Vec<f32> = (0..2 * k).map(|i| (i as f32 % 7.0) - 3.0).collect();
        let bytes = encode_rows(StorageEncoding::Quant(QuantKind::Q8_0), &vals, 2, k).unwrap();
        assert_eq!(
            bytes.len(),
            QuantKind::Q8_0.bytes_for_elements((2 * k) as u64) as usize
        );
        let mut back = vec![0f32; 2 * k];
        dequantize_row(QuantKind::Q8_0, &bytes, &mut back);
        for (a, b) in vals.iter().zip(&back) {
            assert!((a - b).abs() < 0.1, "{a} vs {b}");
        }
    }

    // Minimal in-memory source for discover_specs.
    struct MemSource {
        catalog: WeightCatalog,
        bytes: HashMap<WeightId, Vec<u8>>,
    }
    impl MemSource {
        fn new() -> Self {
            Self {
                catalog: WeightCatalog::new(),
                bytes: HashMap::new(),
            }
        }
        fn put(&mut self, name: &str, shape: Vec<usize>, enc: StorageEncoding, bytes: Vec<u8>) {
            self.catalog.entries.insert(
                WeightId(name.into()),
                WeightEntry {
                    offset: 0,
                    size: bytes.len() as u64,
                    encoding: Some(enc),
                    encoding_label: String::new(),
                    shape: Shape(shape),
                },
            );
            self.bytes.insert(WeightId(name.into()), bytes);
        }
    }
    impl WeightSource for MemSource {
        type Reader = VecReader;
        type Error = Infallible;
        fn catalog(&self) -> &WeightCatalog {
            &self.catalog
        }
        async fn open(&self, id: &WeightId) -> Result<VecReader, Infallible> {
            Ok(VecReader::new(Arc::from(
                self.bytes
                    .get(id)
                    .cloned()
                    .unwrap_or_default()
                    .into_boxed_slice(),
            )))
        }
    }

    #[test]
    fn discover_specs_skips_zero_and_unmatched_reads_rank() {
        use futures::FutureExt;
        let mut base = MemSource::new();
        // Foldable base tensors [N,K] (bytes content irrelevant for discovery).
        base.put(
            "x.weight",
            vec![2, 3],
            StorageEncoding::Bf16,
            bf16_bytes(&[0.0; 6]),
        );
        base.put(
            "z.weight",
            vec![2, 3],
            StorageEncoding::Bf16,
            bf16_bytes(&[0.0; 6]),
        );
        // No base for "y" -> the y lora site must be skipped.

        let mut lora = MemSource::new();
        // x: nonzero B, rank 2 -> kept.
        lora.put(
            "diffusion_model.x.lora_A.weight",
            vec![2, 3],
            StorageEncoding::Bf16,
            bf16_bytes(&[0.0; 6]),
        );
        lora.put(
            "diffusion_model.x.lora_B.weight",
            vec![2, 2],
            StorageEncoding::Bf16,
            bf16_bytes(&[0.0, 1.0, 0.0, 0.0]),
        );
        // z: all-zero B -> skipped.
        lora.put(
            "diffusion_model.z.lora_A.weight",
            vec![1, 3],
            StorageEncoding::Bf16,
            bf16_bytes(&[0.0; 3]),
        );
        lora.put(
            "diffusion_model.z.lora_B.weight",
            vec![2, 1],
            StorageEncoding::Bf16,
            bf16_bytes(&[0.0, 0.0]),
        );
        // y: nonzero B but no base tensor -> skipped.
        lora.put(
            "diffusion_model.y.lora_A.weight",
            vec![1, 3],
            StorageEncoding::Bf16,
            bf16_bytes(&[0.0; 3]),
        );
        lora.put(
            "diffusion_model.y.lora_B.weight",
            vec![2, 1],
            StorageEncoding::Bf16,
            bf16_bytes(&[1.0, 1.0]),
        );

        let specs = discover_specs(&base, &lora)
            .now_or_never()
            .unwrap()
            .expect("discover");
        assert_eq!(specs.len(), 1, "only x folds");
        let s = &specs[0];
        assert_eq!(s.base.0, "x.weight");
        assert_eq!((s.n, s.k, s.rank), (2, 3, 2));
    }
}
