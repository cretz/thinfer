//! Folds the ostris turbotime LoRA into the Ideogram-4 DiT at load.
//!
//! The LoRA removes CFG (single conditional transformer, no unconditional
//! branch). It ships as `ostris/ideogram_4_turbotime_v1.safetensors` (rank 128,
//! bf16, ai-toolkit, NO `.alpha` -> scale 1.0): 408 tensors = 6 sites x {A,B} x
//! 34 layers. Sites per layer: `adaln_modulation`, `attention.qkv`,
//! `attention.o`, `feed_forward.{w1,w2,w3}`. Key format
//! `diffusion_model.layers.{i}.{site}.lora_{A,B}.weight`.
//!
//! `LoraFoldSource` wraps the DiT GGUF (base) + the LoRA safetensors and, for
//! each of the 204 base matmul tensors that has a LoRA pair, serves
//! `quantize_target(dequant(base) + B@A)`. `lora_A` is `[rank, K]`, `lora_B` is
//! `[N, rank]`, so `B@A = [N, K]` matches the GGUF base ne-order (block-major
//! `[N, K]`). All other base tensors pass through untouched.
//!
//! The fold output quant (`target`) is the runtime choice: Q8_0 (parity canary,
//! near-lossless) or Q4_K (default -- ~half the bytes/bandwidth). The folded
//! sites re-report `StorageEncoding::Quant(target)`, so the folded source is a
//! drop-in for a quantized DiT GGUF: the encoding-driven loader and the
//! pipeline's per-site encoding probe see `target` and route through the proven
//! dequant-once / DP4A matmul path (the DiT runs head_dim 256 large-D SDPA,
//! which has no bf16 variant, so block matmuls MUST be a quant-weight + F16-act
//! path, not bf16-weight/f16-act -- Q4_K and Q8_0 both satisfy this, as does
//! Z-Image's Q4_K DiT). The pyref folds in bf16 then the engine re-quantizes;
//! Q8_0's extra rounding is within the loose e2e band (the parity path), Q4_K
//! is eyeballed.
//!
//! Fold output is computed once per tensor and cached in RAM (residency may
//! re-acquire across denoise steps; recomputing `B@A` each time would be
//! catastrophic under paging). The cache is `target`-quant (~9GB Q8_0 / ~5GB
//! Q4_K for the full DiT), not the ~18GB a bf16 fold would hold.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use thinfer_core::quant::{QuantKind, dequantize_row, quantize_row};
use thinfer_core::tensor::{Shape, StorageEncoding};
use thinfer_core::weight::{
    DecodeError, WeightCatalog, WeightEntry, WeightId, WeightReader, WeightSource, decode_into,
};

use super::config;

/// LoRA rank (ostris turbotime). Asserted against the safetensors at fold time.
const RANK: usize = 128;

/// The six per-layer matmul sites that carry a LoRA pair, as the suffix shared
/// between the base key (`layers.{i}.{suffix}.weight`) and the LoRA keys
/// (`diffusion_model.layers.{i}.{suffix}.lora_{A,B}.weight`).
const LORA_SITES: [&str; 6] = [
    "adaln_modulation",
    "attention.qkv",
    "attention.o",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
];

/// One fold target: the base tensor plus its LoRA `A`/`B` ids and `[N, K]` dims.
#[derive(Clone, Debug)]
struct FoldSpec {
    base: WeightId,
    a: WeightId,
    b: WeightId,
    n: usize,
    k: usize,
}

/// Wraps a base DiT source + a LoRA source, serving `quantize(dequant(base) +
/// B@A)` at the 204 LoRA matmul sites and passing everything else through to
/// `base`. `target` is the fold output quant: Q8_0 (the parity canary, near-
/// lossless) or Q4_K (the runtime default -- ~half the bytes/bandwidth of Q8_0;
/// the engine's proven dequant-once / DP4A path reads it, same as Z-Image's Q4_K
/// DiT). The base GGUF's own encoding is irrelevant to the output: every fold
/// site is dequantized to f32, LoRA-added, then re-quantized to `target`.
pub struct LoraFoldSource<B: WeightSource, L: WeightSource> {
    base: B,
    lora: L,
    target: QuantKind,
    catalog: WeightCatalog,
    folds: HashMap<WeightId, FoldSpec>,
    /// Compute-once cache of folded `target`-quant bytes, keyed by folded id.
    cache: Mutex<HashMap<WeightId, Arc<[u8]>>>,
}

#[derive(Debug)]
pub enum FoldError<BE: core::fmt::Debug, LE: core::fmt::Debug> {
    Base(BE),
    Lora(LE),
    Decode(DecodeError),
    /// A reader-side `read_at` failed (reader error types differ from the
    /// source error types, so they ride a formatted string).
    Read(String),
    /// A configured LoRA site was missing or mis-shaped (programmer/checkpoint
    /// error: the site list and rank are fixed).
    Shape(String),
}

fn id(s: impl Into<String>) -> WeightId {
    WeightId(s.into())
}

impl<B: WeightSource, L: WeightSource> LoraFoldSource<B, L> {
    /// Build the folded catalog, re-quantizing fold outputs to `target`. Sites
    /// whose base or LoRA tensors are absent are skipped (so a partial
    /// checkpoint degrades to "no fold there" rather than failing the whole
    /// load); a complete turbotime LoRA folds all 204.
    pub fn new(base: B, lora: L, target: QuantKind) -> Self {
        let mut catalog = WeightCatalog::new();
        for (k, v) in &base.catalog().entries {
            catalog.entries.insert(k.clone(), v.clone());
        }
        let mut folds = HashMap::new();
        for layer in 0..config::N_LAYERS {
            for site in LORA_SITES {
                let base_id = id(format!("layers.{layer}.{site}.weight"));
                let a_id = id(format!(
                    "diffusion_model.layers.{layer}.{site}.lora_A.weight"
                ));
                let b_id = id(format!(
                    "diffusion_model.layers.{layer}.{site}.lora_B.weight"
                ));
                let (Some(be), Some(ae), Some(bbe)) = (
                    base.catalog().get(&base_id),
                    lora.catalog().get(&a_id),
                    lora.catalog().get(&b_id),
                ) else {
                    continue;
                };
                // base [N, K]; lora_A [rank, K]; lora_B [N, rank].
                let (n, k) = (be.shape.0[0], be.shape.0[1]);
                debug_assert_eq!(ae.shape.0, vec![RANK, k], "lora_A shape {a_id:?}");
                debug_assert_eq!(bbe.shape.0, vec![n, RANK], "lora_B shape {b_id:?}");
                // Republish the base entry as `target` quant, same [N, K] shape.
                // K is a whole number of target blocks (4608/12288/512 are all
                // divisible by 256, so Q4_K's 256-elem super-block fits; Q8_0's
                // 32 trivially).
                let bs = target.block_size() as usize;
                debug_assert_eq!(
                    k % bs,
                    0,
                    "fold site K must be {bs}-block aligned for {target:?}"
                );
                let q_bytes = target.bytes_for_elements((n * k) as u64);
                catalog.entries.insert(
                    base_id.clone(),
                    WeightEntry {
                        offset: 0,
                        size: q_bytes,
                        encoding: Some(StorageEncoding::Quant(target)),
                        encoding_label: target.hint().to_string(),
                        shape: Shape(vec![n, k]),
                    },
                );
                folds.insert(
                    base_id.clone(),
                    FoldSpec {
                        base: base_id,
                        a: a_id,
                        b: b_id,
                        n,
                        k,
                    },
                );
            }
        }
        Self {
            base,
            lora,
            target,
            catalog,
            folds,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// How many sites will be folded (204 for a complete turbotime LoRA).
    pub fn fold_count(&self) -> usize {
        self.folds.len()
    }

    async fn read_full_base(
        &self,
        id: &WeightId,
    ) -> Result<Vec<u8>, FoldError<B::Error, L::Error>> {
        let mut r = self.base.open(id).await.map_err(FoldError::Base)?;
        let len = r.len() as usize;
        let mut buf = vec![0u8; len];
        r.read_at(0, &mut buf)
            .await
            .map_err(|e| FoldError::Read(format!("base {id:?}: {e:?}")))?;
        Ok(buf)
    }

    async fn read_full_lora(
        &self,
        id: &WeightId,
    ) -> Result<Vec<u8>, FoldError<B::Error, L::Error>> {
        let mut r = self.lora.open(id).await.map_err(FoldError::Lora)?;
        let len = r.len() as usize;
        let mut buf = vec![0u8; len];
        r.read_at(0, &mut buf)
            .await
            .map_err(|e| FoldError::Read(format!("lora {id:?}: {e:?}")))?;
        Ok(buf)
    }

    /// Compute the folded `target`-quant bytes for one site:
    /// `quantize(dequant(base) + B@A)`.
    async fn compute_fold(
        &self,
        spec: &FoldSpec,
    ) -> Result<Arc<[u8]>, FoldError<B::Error, L::Error>> {
        let (n, k) = (spec.n, spec.k);
        // base -> f32 [N*K] (Q8_0 dequant, or bf16/f32 decode for robustness).
        let base_bytes = self.read_full_base(&spec.base).await?;
        let base_enc = self
            .base
            .catalog()
            .get(&spec.base)
            .and_then(|e| e.encoding)
            .ok_or_else(|| FoldError::Shape(format!("base {:?} has no encoding", spec.base)))?;
        let mut acc = vec![0f32; n * k];
        match base_enc {
            StorageEncoding::Quant(kind) => dequantize_row(kind, &base_bytes, &mut acc),
            StorageEncoding::Bf16 | StorageEncoding::F32 => {
                let dst = bytemuck::cast_slice_mut(&mut acc);
                decode_into(base_enc, &base_bytes, dst).map_err(FoldError::Decode)?;
            }
            other => {
                return Err(FoldError::Shape(format!(
                    "base encoding {other:?} unsupported"
                )));
            }
        }

        // lora A [rank, K], B [N, rank] -> f32.
        let a = self.lora_to_f32(&spec.a, RANK * k).await?;
        let b = self.lora_to_f32(&spec.b, n * RANK).await?;

        // acc[n,k] += sum_r B[n,r] * A[r,k]. Parallel over output rows (each
        // row is independent; A is shared read-only). scale alpha/rank = 1.0.
        fold_add(&mut acc, &b, &a, k);

        // f32 -> `target` quant blocks. K is target-block-aligned, so each row's
        // blocks stay row-local: fan the rows across threads. Q4_K quant (per-32
        // scale/min search) is the fold hot path -- serial it stalls minutes.
        let out = quantize_rows(self.target, &acc, n, k);
        Ok(Arc::from(out.into_boxed_slice()))
    }

    async fn lora_to_f32(
        &self,
        id: &WeightId,
        expect: usize,
    ) -> Result<Vec<f32>, FoldError<B::Error, L::Error>> {
        let bytes = self.read_full_lora(id).await?;
        let enc = self
            .lora
            .catalog()
            .get(id)
            .and_then(|e| e.encoding)
            .ok_or_else(|| FoldError::Shape(format!("lora {id:?} has no encoding")))?;
        let mut out = vec![0f32; expect];
        let dst = bytemuck::cast_slice_mut(&mut out);
        let written = decode_into(enc, &bytes, dst).map_err(FoldError::Decode)?;
        if written != expect * 4 {
            return Err(FoldError::Shape(format!(
                "lora {id:?}: decoded {written} bytes, expected {}",
                expect * 4
            )));
        }
        Ok(out)
    }
}

/// `acc[n,:] += B[n,:] @ A` where `B` is `[N, RANK]`, `A` is `[RANK, K]`, and
/// `acc` is `[N, K]` (row-major). Each output row is independent; on native we
/// fan the rows across threads (the full-DiT fold is ~2.3 TFLOP, minutes
/// single-threaded). The inner k-loop vectorizes and reuses A's rows.
fn fold_add(acc: &mut [f32], b: &[f32], a: &[f32], k: usize) {
    let row = |acc_row: &mut [f32], b_row: &[f32]| {
        for (r, &bnr) in b_row.iter().enumerate() {
            if bnr == 0.0 {
                continue;
            }
            let a_row = &a[r * k..(r + 1) * k];
            for (dst, &av) in acc_row.iter_mut().zip(a_row) {
                *dst += bnr * av;
            }
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(acc.len().max(1));
        if threads > 1 {
            let rows = acc.len() / k;
            let chunk_rows = rows.div_ceil(threads);
            std::thread::scope(|s| {
                for (acc_chunk, b_chunk) in acc
                    .chunks_mut(chunk_rows * k)
                    .zip(b.chunks(chunk_rows * RANK))
                {
                    let row = &row;
                    s.spawn(move || {
                        for (ar, br) in acc_chunk.chunks_mut(k).zip(b_chunk.chunks_exact(RANK)) {
                            row(ar, br);
                        }
                    });
                }
            });
            return;
        }
    }
    for (ar, br) in acc.chunks_mut(k).zip(b.chunks_exact(RANK)) {
        row(ar, br);
    }
}

/// Quantize `acc` `[N, K]` (row-major) to `target` block bytes, fanning rows
/// across threads. K is target-block-aligned, so each row's output blocks are
/// disjoint and row-local. Q4_K's per-32 scale/min search dominates the fold;
/// serial over the full DiT it costs minutes (the regression that stalled the
/// first job), so this mirrors `fold_add`'s row chunking.
fn quantize_rows(target: QuantKind, acc: &[f32], n: usize, k: usize) -> Vec<u8> {
    let row_bytes = target.bytes_for_elements(k as u64) as usize;
    let mut out = vec![0u8; n * row_bytes];
    let quant = |src: &[f32], dst: &mut [u8]| {
        let mut buf = Vec::with_capacity(dst.len());
        quantize_row(target, src, &mut buf);
        dst.copy_from_slice(&buf);
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
                    let quant = &quant;
                    s.spawn(move || quant(src_chunk, dst_chunk));
                }
            });
            return out;
        }
    }
    quant(acc, &mut out);
    out
}

impl<B: WeightSource, L: WeightSource> WeightSource for LoraFoldSource<B, L> {
    type Reader = LoraFoldReader<B>;
    type Error = FoldError<B::Error, L::Error>;

    fn catalog(&self) -> &WeightCatalog {
        &self.catalog
    }

    async fn open(&self, id: &WeightId) -> Result<Self::Reader, Self::Error> {
        let Some(spec) = self.folds.get(id) else {
            // Passthrough: a non-folded base tensor.
            let r = self.base.open(id).await.map_err(FoldError::Base)?;
            return Ok(LoraFoldReader::Base(r));
        };
        // Cache hit?
        if let Some(bytes) = self.cache.lock().expect("lora cache").get(id).cloned() {
            return Ok(LoraFoldReader::Folded(VecReader::new(bytes)));
        }
        let bytes = self.compute_fold(spec).await?;
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
                // VecReader is infallible (Infallible has no inhabitants).
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
