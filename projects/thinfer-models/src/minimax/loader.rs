//! MiniMax-H3 load-time work: the AdaLN precompute and two checkpoint-shape
//! corrections the GGUF conversion forces on us.
//!
//! **The comfy converter reshapes tensors whose row length is not a multiple of
//! the 256-element K-quant superblock.** `blocks.N.adaln_proj.linear.weight` is
//! logically `[96768, 2688]`, but 2688 is 10.5 superblocks, so it ships as
//! `[1016064, 256]` with the true shape recorded only in a
//! `comfy.gguf.orig_shape.*` metadata key. `video_patch_proj.weight` gets the
//! same treatment (`[5376, 96]` -> `[2016, 256]`) even though it is F16.
//!
//! The element order is preserved by the reshape, so reading is unaffected, but
//! the `Linear2D` upload transpose reads the SHAPE. Left alone it would
//! transpose `video_patch_proj` as a `[2016, 256]` matrix and scramble every
//! video row with no NaN to catch it. [`register_video_patch_proj`] restores the
//! real shape.
//!
//! **The AdaLN precompute runs on the host.** There are only a handful of
//! distinct timesteps, so the whole table is a `[T, 2688] x [2688, 96768]`
//! product per block: a few GFLOP total, against ~7 GiB of weight reads that
//! dominate either way. Streaming the Q4 blocks two output rows at a time keeps
//! peak memory at one tensor's byte buffer instead of the ~1 GiB a full f32
//! dequantization would cost, and the arithmetic stays f32-exact, which matters
//! because every block reads the same `temb` (see [`super::adaln`]).

use half::f16;

use thinfer_core::quant::{QuantKind, dequantize_row};
use thinfer_core::residency::{TransposePolicy, WeightHandle, WeightMeta, WeightResidency};
use thinfer_core::tensor::{Shape, StorageEncoding};
use thinfer_core::weight::{WeightId, WeightReader, WeightSource};

use super::adaln::{ModulationLayout, timestep_embedding};
use super::{AdalnInput, DitConfig};
use crate::z_image::text_encoder::LoadError;

#[derive(Debug)]
pub enum LoaderError<SE: core::fmt::Debug> {
    Load(LoadError),
    Read(SE),
    /// A tensor read failed. The reader's error type is independent of the
    /// source's, so it is rendered rather than nested.
    ReadAt(String),
    /// A tensor was absent, or present at a shape we cannot interpret.
    BadTensor {
        id: String,
        detail: String,
    },
    /// The conditioner is not the truncated build H3 reads.
    EncoderDepth {
        got: usize,
        want: usize,
    },
}

impl<SE: core::fmt::Debug> From<LoadError> for LoaderError<SE> {
    fn from(e: LoadError) -> Self {
        Self::Load(e)
    }
}

/// H3 reads `hidden_states[50]`, so a checkpoint MUST carry at least 50 decoder
/// layers. A shorter one silently conditions on the wrong depth; a full
/// 64-layer stack merely wastes space and is allowed.
pub fn check_encoder_depth<SE: core::fmt::Debug>(
    block_count: usize,
) -> Result<(), LoaderError<SE>> {
    let want = super::text_encoder::config::N_LAYERS;
    if block_count < want {
        return Err(LoaderError::EncoderDepth {
            got: block_count,
            want,
        });
    }
    Ok(())
}

/// Register `video_patch_proj.weight` at its TRUE `[hidden, video_patch_dim]`
/// shape rather than the converter's superblock-friendly reshape.
pub fn register_video_patch_proj<S: WeightSource>(
    cfg: &DitConfig,
    residency: &WeightResidency<S>,
) -> Result<WeightHandle, LoadError> {
    let id = WeightId("video_patch_proj.weight".to_string());
    let entry = residency
        .source()
        .catalog()
        .get(&id)
        .ok_or_else(|| LoadError::UnknownWeight(id.clone()))?;
    let encoding = entry.encoding.ok_or_else(|| LoadError::Undecodable {
        id: id.clone(),
        encoding: None,
        label: entry.encoding_label.clone(),
    })?;
    let (n, k) = (cfg.hidden, cfg.video_patch_dim());
    assert_eq!(
        entry.shape.0.iter().product::<usize>(),
        n * k,
        "video_patch_proj element count must survive the converter's reshape"
    );
    Ok(residency.register(WeightMeta {
        id,
        shape: Shape(vec![n, k]),
        encoding,
        on_disk_bytes: entry.size,
        transpose: TransposePolicy::Linear2D,
        transcode: None,
    }))
}

/// The precomputed modulation table, host-side and ready to upload.
///
/// `blocks[i]` is `[num_timesteps * MODALITY_NUM, 6 * hidden]` and
/// `final_layer` is `[num_timesteps, 2 * hidden]`. Both are the projection
/// output verbatim: the checkpoint already emits `[t][modality][param][hidden]`
/// order, so no permutation is involved.
pub struct ModulationTables {
    pub layout: ModulationLayout,
    pub blocks: Vec<Vec<f32>>,
    pub final_layer: Vec<f32>,
}

impl ModulationTables {
    pub fn bytes(&self, bytes_per_element: usize) -> usize {
        self.layout
            .total_bytes(self.blocks.len(), bytes_per_element)
    }
}

/// Read a whole tensor and decode it to f32.
///
/// GGUF pads tensor data to its 32-byte alignment, so the catalog `size` can
/// exceed the real element bytes; every arm slices to the tight length.
async fn read_tensor_f32<S: WeightSource>(
    source: &S,
    id: &WeightId,
    elements: usize,
) -> Result<Vec<f32>, LoaderError<S::Error>> {
    let entry = source
        .catalog()
        .get(id)
        .ok_or_else(|| LoaderError::BadTensor {
            id: id.0.clone(),
            detail: "absent from the catalog".into(),
        })?;
    let encoding = entry.encoding.ok_or_else(|| LoaderError::BadTensor {
        id: id.0.clone(),
        detail: format!("undecodable encoding {}", entry.encoding_label),
    })?;
    let mut bytes = vec![0u8; entry.size as usize];
    let mut reader = source.open(id).await.map_err(LoaderError::Read)?;
    reader
        .read_at(0, &mut bytes)
        .await
        .map_err(|e| LoaderError::ReadAt(format!("{}: {e:?}", id.0)))?;
    Ok(decode_f32(encoding, &bytes, elements))
}

fn decode_f32(encoding: StorageEncoding, bytes: &[u8], elements: usize) -> Vec<f32> {
    match encoding {
        StorageEncoding::F32 => bytes[..elements * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        StorageEncoding::F16 => bytes[..elements * 2]
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        StorageEncoding::Bf16 => bytes[..elements * 2]
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        StorageEncoding::Quant(kind) => {
            let mut out = vec![0f32; elements];
            dequantize_row(
                kind,
                &bytes[..kind.bytes_for_elements(elements as u64) as usize],
                &mut out,
            );
            out
        }
        enc => panic!("MiniMax-H3 checkpoints carry no {enc:?} tensors"),
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `y = x @ w^T + b` for a host `[rows, k]` against `[n, k]` row-major.
fn linear_f32(x: &[f32], w: &[f32], bias: &[f32], rows: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * n];
    for j in 0..n {
        let wr = &w[j * k..(j + 1) * k];
        for r in 0..rows {
            let xr = &x[r * k..(r + 1) * k];
            let mut acc = bias[j];
            for i in 0..k {
                acc += xr[i] * wr[i];
            }
            out[r * n + j] = acc;
        }
    }
    out
}

/// Build the whole run's modulation table.
///
/// `timesteps` must be every DISTINCT noise level the schedule will present,
/// unscaled in `[0, 1]`. The transformer is handed the same table at every
/// step, so this is built once.
pub async fn build_modulation_tables<S: WeightSource>(
    cfg: &DitConfig,
    source: &S,
    timesteps: &[f32],
) -> Result<ModulationTables, LoaderError<S::Error>> {
    let t = timesteps.len();
    assert!(t > 0, "modulation table needs at least one timestep");
    let (a, k) = match cfg.adaln {
        AdalnInput::Table(r) => (table_rows(source, timesteps, r).await?, r),
        AdalnInput::Full => (silu_temb(cfg, source, timesteps).await?, cfg.time_embed_dim),
        // A rank-reduced checkpoint maps silu(temb) through the shared basis
        // (no further silu).
        AdalnInput::Basis(r) => {
            let a = silu_temb(cfg, source, timesteps).await?;
            let basis = read_tensor_f32(
                source,
                &WeightId("adaln_basis.weight".into()),
                r * cfg.time_embed_dim,
            )
            .await?;
            let a = linear_f32(&a, &basis, &vec![0.0; r], t, cfg.time_embed_dim, r);
            (a, r)
        }
    };

    let mut blocks = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        blocks.push(
            project_adaln(
                source,
                &a,
                t,
                k,
                cfg.adaln_out,
                &format!("blocks.{i}.adaln_proj"),
            )
            .await?,
        );
    }
    let final_layer = project_adaln(
        source,
        &a,
        t,
        k,
        cfg.final_adaln_out,
        "final_layer.adaln_proj",
    )
    .await?;

    Ok(ModulationTables {
        layout: ModulationLayout::new(t, cfg.hidden),
        blocks,
        final_layer,
    })
}

/// The curve table's rows at each `t` (`t = 1 - sigma`), linearly
/// interpolated: `pos = clamp(t, 0, 1) * (rows - 1)`, lerp rows `floor(pos)`
/// and the next (Comfy `MiniMaxH3`'s lookup), `[timesteps, rank]`.
async fn table_rows<S: WeightSource>(
    source: &S,
    timesteps: &[f32],
    rank: usize,
) -> Result<Vec<f32>, LoaderError<S::Error>> {
    let id = WeightId("adaln_t_table".into());
    let rows = source
        .catalog()
        .get(&id)
        .map(|e| e.shape.0[0])
        .ok_or_else(|| LoaderError::BadTensor {
            id: id.0.clone(),
            detail: "absent from the catalog".into(),
        })?;
    let table = read_tensor_f32(source, &id, rows * rank).await?;
    let mut out = Vec::with_capacity(timesteps.len() * rank);
    for &t in timesteps {
        let pos = t.clamp(0.0, 1.0) * (rows - 1) as f32;
        let i0 = (pos.floor() as usize).min(rows - 2);
        let w = pos - i0 as f32;
        let (r0, r1) = (
            &table[i0 * rank..][..rank],
            &table[(i0 + 1) * rank..][..rank],
        );
        out.extend(r0.iter().zip(r1).map(|(a, b)| a + w * (b - a)));
    }
    Ok(out)
}

/// `silu(temb)`, `temb = proj_out(silu(proj_in(sinusoid)))`, f32 throughout.
async fn silu_temb<S: WeightSource>(
    cfg: &DitConfig,
    source: &S,
    timesteps: &[f32],
) -> Result<Vec<f32>, LoaderError<S::Error>> {
    let t = timesteps.len();
    let sinusoid = timestep_embedding(timesteps, cfg.timestep_input_dim);
    let temb = {
        let w_in = read_tensor_f32(
            source,
            &WeightId("time_embedder.proj_in.weight".into()),
            cfg.time_embed_hidden * cfg.timestep_input_dim,
        )
        .await?;
        let b_in = read_tensor_f32(
            source,
            &WeightId("time_embedder.proj_in.bias".into()),
            cfg.time_embed_hidden,
        )
        .await?;
        let mut h = linear_f32(
            &sinusoid,
            &w_in,
            &b_in,
            t,
            cfg.timestep_input_dim,
            cfg.time_embed_hidden,
        );
        for v in &mut h {
            *v = silu(*v);
        }
        let w_out = read_tensor_f32(
            source,
            &WeightId("time_embedder.proj_out.weight".into()),
            cfg.time_embed_dim * cfg.time_embed_hidden,
        )
        .await?;
        let b_out = read_tensor_f32(
            source,
            &WeightId("time_embedder.proj_out.bias".into()),
            cfg.time_embed_dim,
        )
        .await?;
        linear_f32(
            &h,
            &w_out,
            &b_out,
            t,
            cfg.time_embed_hidden,
            cfg.time_embed_dim,
        )
    };

    // Every AdaLN projection reads silu(temb), not temb.
    Ok(temb.iter().map(|&v| silu(v)).collect())
}

/// One `adaln_proj.linear` applied to `a` `[t, k]`, streaming the weight rather
/// than materializing it: a full f32 dequantization of one block's projection is
/// ~1 GiB, and there are 50 of them.
async fn project_adaln<S: WeightSource>(
    source: &S,
    a: &[f32],
    t: usize,
    k: usize,
    n: usize,
    prefix: &str,
) -> Result<Vec<f32>, LoaderError<S::Error>> {
    let wid = WeightId(format!("{prefix}.linear.weight"));
    let bias = read_tensor_f32(source, &WeightId(format!("{prefix}.linear.bias")), n).await?;

    let entry = source
        .catalog()
        .get(&wid)
        .ok_or_else(|| LoaderError::BadTensor {
            id: wid.0.clone(),
            detail: "absent from the catalog".into(),
        })?;
    let encoding = entry.encoding.ok_or_else(|| LoaderError::BadTensor {
        id: wid.0.clone(),
        detail: format!("undecodable encoding {}", entry.encoding_label),
    })?;
    // The converter's reshape preserves element order, so the flat stream is
    // still the `[n, k]` row-major weight whatever dims the header claims.
    if entry.shape.0.iter().product::<usize>() != n * k {
        return Err(LoaderError::BadTensor {
            id: wid.0.clone(),
            detail: format!("{:?} does not hold {n} x {k} elements", entry.shape.0),
        });
    }

    let mut bytes = vec![0u8; entry.size as usize];
    let mut reader = source.open(&wid).await.map_err(LoaderError::Read)?;
    reader
        .read_at(0, &mut bytes)
        .await
        .map_err(|e| LoaderError::ReadAt(format!("{}: {e:?}", wid.0)))?;

    let mut out = vec![0f32; t * n];
    match encoding {
        StorageEncoding::Quant(kind) => {
            // Decode a whole number of output rows at a time. K-quant blocks
            // are 256 elements and `k` need not be a multiple of that, so the
            // chunk is `rows_per_chunk` output rows chosen to land on a block
            // boundary.
            let rows_per_chunk = rows_per_quant_chunk(kind, k);
            let chunk_elems = rows_per_chunk * k;
            let chunk_bytes = kind.bytes_for_elements(chunk_elems as u64) as usize;
            let mut buf = vec![0f32; chunk_elems];
            let mut j = 0usize;
            let mut at = 0usize;
            while j < n {
                let rows = rows_per_chunk.min(n - j);
                let take = if rows == rows_per_chunk {
                    chunk_bytes
                } else {
                    kind.bytes_for_elements((rows * k) as u64) as usize
                };
                dequantize_row(kind, &bytes[at..at + take], &mut buf[..rows * k]);
                accumulate(&mut out, a, &buf[..rows * k], &bias, t, k, n, j, rows);
                j += rows;
                at += take;
            }
        }
        enc => {
            let w = decode_f32(enc, &bytes, n * k);
            accumulate(&mut out, a, &w, &bias, t, k, n, 0, n);
        }
    }
    Ok(out)
}

/// Output rows that span a whole number of quant blocks, so chunked decoding
/// never straddles a block. `k` is 2688 and a K-quant block is 256, so this is
/// 2 rows; a `k` already block-aligned gives 1.
fn rows_per_quant_chunk(kind: QuantKind, k: usize) -> usize {
    let block = kind.block_size() as usize;
    let mut rows = 1;
    while !(rows * k).is_multiple_of(block) {
        rows += 1;
    }
    rows
}

#[allow(clippy::too_many_arguments)]
fn accumulate(
    out: &mut [f32],
    a: &[f32],
    w: &[f32],
    bias: &[f32],
    t: usize,
    k: usize,
    n: usize,
    j0: usize,
    rows: usize,
) {
    for r in 0..rows {
        let wr = &w[r * k..(r + 1) * k];
        let j = j0 + r;
        for ti in 0..t {
            let ar = &a[ti * k..(ti + 1) * k];
            let mut acc = bias[j];
            for i in 0..k {
                acc += ar[i] * wr[i];
            }
            out[ti * n + j] = acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_depth_rejects_only_short_stacks() {
        assert!(check_encoder_depth::<std::convert::Infallible>(50).is_ok());
        assert!(check_encoder_depth::<std::convert::Infallible>(64).is_ok());
        assert!(check_encoder_depth::<std::convert::Infallible>(49).is_err());
    }

    /// `k = 2688` is 10.5 K-quant superblocks, which is exactly why the
    /// converter reshapes the tensor. Two output rows land on a block boundary.
    #[test]
    fn quant_chunking_lands_on_block_boundaries() {
        let c = DitConfig::h3_33b();
        assert_eq!(c.time_embed_dim, 2688);
        assert!(!c.time_embed_dim.is_multiple_of(256));
        let rows = rows_per_quant_chunk(QuantKind::Q4_K, c.time_embed_dim);
        assert_eq!(rows, 2);
        assert!((rows * c.time_embed_dim).is_multiple_of(256));
    }

    /// A block-aligned width needs no multi-row chunk.
    #[test]
    fn block_aligned_width_chunks_one_row() {
        assert_eq!(rows_per_quant_chunk(QuantKind::Q4_K, 5376), 1);
    }

    #[test]
    fn linear_matches_a_hand_computed_product() {
        // x [1, 2] @ w[2, 2]^T + b
        let x = [1.0f32, 2.0];
        let w = [1.0f32, 0.0, 0.0, 1.0];
        let b = [10.0f32, 20.0];
        assert_eq!(linear_f32(&x, &w, &b, 1, 2, 2), vec![11.0, 22.0]);
    }

    /// The table row layout must agree with what the block stack indexes.
    #[test]
    fn table_width_matches_the_modulation_layout() {
        let c = DitConfig::h3_33b();
        let layout = ModulationLayout::new(5, c.hidden);
        assert_eq!(layout.rows() * layout.row_width(), 5 * c.adaln_out);
        assert_eq!(layout.rows(), 5 * super::super::MODALITY_NUM);
    }
}
