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

use super::adaln::{ModulationLayout, TimeKey, timestep_embedding};
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
/// One `adaln_proj.linear` low-rank pair of an adapter (larryvrh's turbo
/// LoRA ships them for every block and the final layer): `out += scale * B @
/// (A @ silu(temb))`. The projections never run at sampling time, so the pair
/// folds into the precomputed table exactly (ComfyUI injects the same delta at
/// run time).
pub struct AdalnPair {
    /// `[rank, k]`.
    a: Vec<f32>,
    /// `[n, rank]`.
    b: Vec<f32>,
    rank: usize,
    scale: f32,
}

impl AdalnPair {
    /// Add the delta for each of the `t` rows of `x` (`[t, k]`) to `out`
    /// (`[t, n]`).
    fn apply(&self, x: &[f32], t: usize, k: usize, n: usize, out: &mut [f32]) {
        for ti in 0..t {
            let xr = &x[ti * k..][..k];
            let mid: Vec<f32> = (0..self.rank)
                .map(|r| {
                    self.a[r * k..][..k]
                        .iter()
                        .zip(xr)
                        .map(|(w, v)| w * v)
                        .sum::<f32>()
                        * self.scale
                })
                .collect();
            let row = &mut out[ti * n..][..n];
            for (j, o) in row.iter_mut().enumerate() {
                let bj = &self.b[j * self.rank..][..self.rank];
                *o += bj.iter().zip(&mid).map(|(w, m)| w * m).sum::<f32>();
            }
        }
    }
}

/// Every AdaLN pair of an adapter, per block and for the final layer.
pub struct AdalnLora {
    pub blocks: Vec<Option<AdalnPair>>,
    pub final_layer: Option<AdalnPair>,
}

/// Read the `*.adaln_proj.linear.lora_{A,B}.weight` pairs of `adapter` (alpha
/// == rank, so the scale is `strength`). `None` when it carries none.
pub async fn read_adaln_lora<S: WeightSource>(
    cfg: &DitConfig,
    adapter: &S,
    strength: f32,
) -> Result<Option<AdalnLora>, LoaderError<S::Error>> {
    let pair = |prefix: String, n: usize| async move {
        let a_id = WeightId(format!("{prefix}.adaln_proj.linear.lora_A.weight"));
        let b_id = WeightId(format!("{prefix}.adaln_proj.linear.lora_B.weight"));
        let Some(rank) = adapter.catalog().get(&a_id).map(|e| e.shape.0[0]) else {
            return Ok::<_, LoaderError<S::Error>>(None);
        };
        let k = cfg.time_embed_dim;
        Ok(Some(AdalnPair {
            a: read_tensor_f32(adapter, &a_id, rank * k).await?,
            b: read_tensor_f32(adapter, &b_id, n * rank).await?,
            rank,
            scale: strength,
        }))
    };
    let mut blocks = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        blocks.push(pair(format!("blocks.{i}"), cfg.adaln_out).await?);
    }
    let final_layer = pair("final_layer".into(), cfg.final_adaln_out).await?;
    let any = final_layer.is_some() || blocks.iter().any(Option::is_some);
    Ok(any.then_some(AdalnLora {
        blocks,
        final_layer,
    }))
}

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
    keys: &[TimeKey],
    adaln_lora: Option<&AdalnLora>,
) -> Result<ModulationTables, LoaderError<S::Error>> {
    let t = keys.len();
    let timesteps: Vec<f32> = keys.iter().map(|k| k.t).collect();
    let timesteps = &timesteps[..];
    assert!(t > 0, "modulation table needs at least one timestep");
    let (a, k) = match cfg.adaln {
        AdalnInput::Table(r) => (table_rows(source, timesteps, r).await?, r),
        AdalnInput::Full => (silu_temb(cfg, source, keys).await?, cfg.time_embed_dim),
        // A rank-reduced checkpoint maps silu(temb) through the shared basis
        // (no further silu).
        AdalnInput::Basis(r) => {
            let a = silu_temb(cfg, source, keys).await?;
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

    // Every block's projection, then the final layer's. Block `i + 1` is read
    // while block `i` projects on a worker: the read (~150 MB per Q4 block on
    // the full checkpoint) overlaps the math instead of alternating with it.
    let projs: Vec<(String, usize)> = (0..cfg.n_layers)
        .map(|i| (format!("blocks.{i}.adaln_proj"), cfg.adaln_out))
        .chain([("final_layer.adaln_proj".to_string(), cfg.final_adaln_out)])
        .collect();
    let a = std::sync::Arc::new(a);
    let mut outs = Vec::with_capacity(projs.len());
    let mut next = Some(read_adaln(source, &projs[0].0, projs[0].1, k).await?);
    for i in 0..projs.len() {
        let w = next.take().expect("read ahead");
        let shared = std::sync::Arc::clone(&a);
        let compute = crate::common::seq::off_thread(move || w.project(&shared, t));
        let read = async {
            match projs.get(i + 1) {
                Some((prefix, n)) => read_adaln(source, prefix, *n, k).await.map(Some),
                None => Ok(None),
            }
        };
        let (out, read) = futures::join!(compute, read);
        next = read?;
        outs.push(out);
    }
    let mut final_layer = outs.pop().expect("final layer projection");
    let mut blocks = outs;
    for (i, out) in blocks.iter_mut().enumerate() {
        if let Some(p) = adaln_lora.and_then(|l| l.blocks[i].as_ref()) {
            p.apply(&a, t, k, cfg.adaln_out, out);
        }
    }
    if let Some(p) = adaln_lora.and_then(|l| l.final_layer.as_ref()) {
        p.apply(&a, t, k, cfg.final_adaln_out, &mut final_layer);
    }

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
/// Two-time checkpoints mix in the endpoint embedder at each key's `r`:
/// `temb(t) + gate * (temb_r(r) - temb(t))` (HyperFlow `time_embedding`).
async fn silu_temb<S: WeightSource>(
    cfg: &DitConfig,
    source: &S,
    keys: &[TimeKey],
) -> Result<Vec<f32>, LoaderError<S::Error>> {
    let ts: Vec<f32> = keys.iter().map(|k| k.t).collect();
    let mut temb = embed_time(cfg, source, "time_embedder", &ts).await?;
    if cfg.two_time {
        let rs: Vec<f32> = keys.iter().map(|k| k.r).collect();
        let temb_r = embed_time(cfg, source, "endpoint_time_embedder", &rs).await?;
        let gate = read_tensor_f32(source, &WeightId("hyperflow_gate".into()), 1).await?[0];
        for (a, b) in temb.iter_mut().zip(&temb_r) {
            *a += gate * (b - *a);
        }
    }
    // Every AdaLN projection reads silu(temb), not temb.
    Ok(temb.iter().map(|&v| silu(v)).collect())
}

/// `proj_out(silu(proj_in(sinusoid(t))))` of the time embedder `name`, f32.
async fn embed_time<S: WeightSource>(
    cfg: &DitConfig,
    source: &S,
    name: &str,
    timesteps: &[f32],
) -> Result<Vec<f32>, LoaderError<S::Error>> {
    let t = timesteps.len();
    let sinusoid = timestep_embedding(timesteps, cfg.timestep_input_dim);
    let w = |s: &str| WeightId(format!("{name}.{s}"));
    let temb = {
        let w_in = read_tensor_f32(
            source,
            &w("proj_in.weight"),
            cfg.time_embed_hidden * cfg.timestep_input_dim,
        )
        .await?;
        let b_in = read_tensor_f32(source, &w("proj_in.bias"), cfg.time_embed_hidden).await?;
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
            &w("proj_out.weight"),
            cfg.time_embed_dim * cfg.time_embed_hidden,
        )
        .await?;
        let b_out = read_tensor_f32(source, &w("proj_out.bias"), cfg.time_embed_dim).await?;
        linear_f32(
            &h,
            &w_out,
            &b_out,
            t,
            cfg.time_embed_hidden,
            cfg.time_embed_dim,
        )
    };
    Ok(temb)
}

/// One `adaln_proj.linear` as read from the file: the encoded `[n, k]` weight
/// and the f32 bias (see [`AdalnWeight::project`]).
struct AdalnWeight {
    bytes: Vec<u8>,
    encoding: StorageEncoding,
    bias: Vec<f32>,
    n: usize,
    k: usize,
}

/// Read `{prefix}.linear` (`n` outputs over `k`).
async fn read_adaln<S: WeightSource>(
    source: &S,
    prefix: &str,
    n: usize,
    k: usize,
) -> Result<AdalnWeight, LoaderError<S::Error>> {
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
    Ok(AdalnWeight {
        bytes,
        encoding,
        bias,
        n,
        k,
    })
}

impl AdalnWeight {
    /// The projection applied to `a` `[t, k]`, decoding the weight a few rows
    /// at a time rather than materializing it (a full f32 dequantization of
    /// one block's projection is ~1 GiB, and there are 50 of them).
    fn project(&self, a: &[f32], t: usize) -> Vec<f32> {
        let AdalnWeight {
            bytes,
            encoding,
            bias,
            n,
            k,
        } = self;
        let (encoding, n, k) = (*encoding, *n, *k);
        // Decode a whole number of output rows at a time. K-quant blocks are
        // 256 elements and `k` need not be a multiple of that, so a quant
        // chunk is the fewest output rows landing on a block boundary.
        let chunk = match encoding {
            StorageEncoding::Quant(kind) => rows_per_quant_chunk(kind, k) * ROW_TILE,
            _ => DENSE_CHUNK_ROWS,
        };
        // Every output is its own sum over `k`: the output rows split into
        // contiguous groups across workers, each decoding only its own rows
        // (identical bits to one pass).
        let group = n.div_ceil(chunk).div_ceil(PROJECT_PARTS) * chunk;
        let lanes = lane_groups(a, t, k);
        let lanes = &lanes;
        let parts = crate::common::seq::par_map(n.div_ceil(group), |g| {
            let (j0, j1) = (g * group, ((g + 1) * group).min(n));
            let mut part = vec![0f32; t * (j1 - j0)];
            let mut j = j0;
            while j < j1 {
                let rows = chunk.min(j1 - j);
                let at = encoded_bytes(encoding, j * k);
                let w = decode_f32(encoding, &bytes[at..], rows * k);
                accumulate(
                    &mut part,
                    lanes,
                    &w,
                    &bias[j..],
                    t,
                    k,
                    j1 - j0,
                    j - j0,
                    rows,
                );
                j += rows;
            }
            part
        });
        let mut out = vec![0f32; t * n];
        for (g, part) in parts.iter().enumerate() {
            let (j0, cols) = (g * group, part.len() / t);
            for ti in 0..t {
                out[ti * n + j0..][..cols].copy_from_slice(&part[ti * cols..][..cols]);
            }
        }
        out
    }
}

/// Worker groups [`AdalnWeight::project`] splits its output rows into.
const PROJECT_PARTS: usize = 8;

/// Output rows a dense [`AdalnWeight::project`] weight decodes at a time.
const DENSE_CHUNK_ROWS: usize = 16;

/// Bytes `elements` leading elements take in `encoding` (whole quant blocks).
fn encoded_bytes(encoding: StorageEncoding, elements: usize) -> usize {
    match encoding {
        StorageEncoding::F32 => elements * 4,
        StorageEncoding::Quant(kind) => kind.bytes_for_elements(elements as u64) as usize,
        _ => elements * 2,
    }
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

/// Timesteps one [`accumulate`] lane group carries.
const LANES: usize = 8;
/// Weight rows [`accumulate`] runs side by side.
const ROW_TILE: usize = 4;

/// `a` `[t, k]` regrouped for [`accumulate`]: group `g` holds, per `i < k`,
/// the lanes `a[g * LANES + l][i]` (zero past `t`).
fn lane_groups(a: &[f32], t: usize, k: usize) -> Vec<Vec<[f32; LANES]>> {
    (0..t.div_ceil(LANES))
        .map(|g| {
            (0..k)
                .map(|i| {
                    std::array::from_fn(|l| {
                        let ti = g * LANES + l;
                        if ti < t { a[ti * k + i] } else { 0.0 }
                    })
                })
                .collect()
        })
        .collect()
}

/// `out[ti, col0 + r] = bias[r] + a[ti] . w[r]` for `rows` weight rows, `out`
/// being `[t, cols]`, `a` as [`lane_groups`]. Every output is still one sum
/// in `k` order (`acc + a * w`, two roundings), so the bits match a plain
/// scalar loop; the timestep lanes and `ROW_TILE` rows are independent
/// chains, which is what lets it vectorize instead of waiting on one add.
#[allow(clippy::too_many_arguments)]
fn accumulate(
    out: &mut [f32],
    at: &[Vec<[f32; LANES]>],
    w: &[f32],
    bias: &[f32],
    t: usize,
    k: usize,
    cols: usize,
    col0: usize,
    rows: usize,
) {
    for (g, ag) in at.iter().enumerate() {
        for r0 in (0..rows).step_by(ROW_TILE) {
            let live = (rows - r0).min(ROW_TILE);
            // Rows past `live` repeat the last live row (results dropped).
            let wr: [&[f32]; ROW_TILE] =
                std::array::from_fn(|q| &w[(r0 + q.min(live - 1)) * k..][..k]);
            let mut acc: [[f32; LANES]; ROW_TILE] =
                std::array::from_fn(|q| [bias[r0 + q.min(live - 1)]; LANES]);
            for (i, av) in ag.iter().enumerate() {
                for q in 0..ROW_TILE {
                    let wv = wr[q][i];
                    for l in 0..LANES {
                        acc[q][l] += av[l] * wv;
                    }
                }
            }
            for (q, row) in acc.iter().enumerate().take(live) {
                for (l, &v) in row.iter().enumerate() {
                    let ti = g * LANES + l;
                    if ti < t {
                        out[ti * cols + col0 + r0 + q] = v;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lane-grouped, row-tiled accumulate is bit-identical to one scalar
    /// sum per output in `k` order (ragged `t` and row counts included).
    #[test]
    fn accumulate_matches_scalar_bits() {
        let (k, rows) = (37, 7);
        let mut seed = 0x2545_f491_u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for t in [1, 7, 8, 11] {
            let a: Vec<f32> = (0..t * k).map(|_| rnd()).collect();
            let w: Vec<f32> = (0..rows * k).map(|_| rnd() * 3.0).collect();
            let bias: Vec<f32> = (0..rows).map(|_| rnd()).collect();
            let mut got = vec![f32::NAN; t * rows];
            accumulate(
                &mut got,
                &lane_groups(&a, t, k),
                &w,
                &bias,
                t,
                k,
                rows,
                0,
                rows,
            );
            for ti in 0..t {
                for r in 0..rows {
                    let mut acc = bias[r];
                    for i in 0..k {
                        acc += a[ti * k + i] * w[r * k + i];
                    }
                    assert_eq!(
                        got[ti * rows + r].to_bits(),
                        acc.to_bits(),
                        "t {t} ti {ti} r {r}"
                    );
                }
            }
        }
    }

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
