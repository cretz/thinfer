//! umT5-XXL text encoder (`google/umt5-xxl`, encoder-only). Shared prompt
//! conditioner for the Wan video family (SkyReels-V2, Wan2.x, NAVA).
//!
//! Encoder-decoder T5 lineage, encoder stack only. Arch deltas vs the Qwen3
//! decoder we already run (`z_image/text_encoder.rs`):
//! - T5LayerNorm == RMSNorm with NO mean subtraction and NO bias (reuse
//!   `op_rmsnorm`); eps 1e-6.
//! - Bidirectional self-attention (no causal mask). The only additive bias is
//!   the per-layer relative-position bias (umT5 keeps a distinct
//!   `relative_attention_bias` per block, unlike vanilla T5's shared layer-0
//!   table), expanded by `relpos_bias` into a per-head SDPA mask
//!   (`has_mask == 2`).
//! - SDPA scale == 1.0 (T5 folds the `1/sqrt(d_kv)` into init; there is no
//!   query scaling).
//! - Separate q/k/v/o projections, no fused qkv, no per-head q/k norm, no RoPE.
//!   `inner == n_heads * d_kv == 4096 == d_model`.
//! - Gated-GELU FFN: `wo(gelu_new(wi_0(x)) * wi_1(x))` (reuse `GeluMulF32`,
//!   the gate*up half mirroring SwiGLU's `SiluMulF32`).
//! - Runs `final_layer_norm` after the last block (output is the encoder last
//!   hidden state, unlike Qwen3 where we stop at `hidden_states[-2]`).
//!
//! Weight names (HF UMT5 encoder convention):
//! `shared.weight` (vocab embed), `encoder.block.{i}.layer.0.{layer_norm,
//! SelfAttention.{q,k,v,o,relative_attention_bias}}`,
//! `encoder.block.{i}.layer.1.{layer_norm, DenseReluDense.{wi_0,wi_1,wo}}`,
//! `encoder.final_layer_norm`.

use thinfer_core::backend::{Backend, WgpuBackend, WgpuError, WgpuPipeline};
use thinfer_core::ops::{ActDtype, GeluMulF32, Op, RelposBiasF32, RelposBiasOp, relpos_bucket_map};
use thinfer_core::residency::{
    GpuView, ResidencyError, TransposePolicy, WeightHandle, WeightMeta, WeightResidency,
};
use thinfer_core::tensor::StorageEncoding;
use thinfer_core::trace::{self, PHASE};
use thinfer_core::weight::{Decoder, WeightId, WeightReader, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace};
use tracing::Instrument;

use crate::common::block::{
    ActBuf, Block, BlockPipelines, BlockWgslConfigs, alloc_act, alloc_matmul_out_buf, copy_tap,
    op_add, op_rmsnorm, op_sdpa,
};
use crate::common::seq;

/// Audited against `google/umt5-xxl/config.json` and the SkyReels-V2-DF
/// Diffusers bundle's `text_encoder/config.json`.
pub mod config {
    pub const D_MODEL: usize = 4096;
    pub const D_FF: usize = 10240;
    pub const D_KV: usize = 64;
    pub const N_HEADS: usize = 64;
    pub const N_LAYERS: usize = 24;
    pub const VOCAB: usize = 256384;
    pub const NUM_BUCKETS: u32 = 32;
    pub const MAX_DISTANCE: u32 = 128;
    pub const LN_EPS: f32 = 1e-6;
    /// `n_heads * d_kv`. Equals `D_MODEL` for umT5-XXL (q/k/v/o are square).
    pub const INNER: usize = N_HEADS * D_KV;
    /// Vocab embedding tensor (T5 ties encoder embed to `shared`).
    pub const EMBED_TOKENS: &str = "shared.weight";
}

// ---------------------------------------------------------------------------
// Weight names + handles
// ---------------------------------------------------------------------------

/// Resolved weight names for one umT5 encoder block.
#[derive(Clone, Debug)]
pub struct Umt5BlockWeights {
    pub attn_norm: WeightId,
    pub q: WeightId,
    pub k: WeightId,
    pub v: WeightId,
    pub o: WeightId,
    pub relpos_bias: WeightId,
    pub ffn_norm: WeightId,
    pub wi_0: WeightId,
    pub wi_1: WeightId,
    pub wo: WeightId,
}

impl Umt5BlockWeights {
    pub fn new(idx: usize) -> Self {
        let p = format!("encoder.block.{idx}");
        let id = |s: &str| WeightId(format!("{p}.{s}"));
        Self {
            attn_norm: id("layer.0.layer_norm.weight"),
            q: id("layer.0.SelfAttention.q.weight"),
            k: id("layer.0.SelfAttention.k.weight"),
            v: id("layer.0.SelfAttention.v.weight"),
            o: id("layer.0.SelfAttention.o.weight"),
            relpos_bias: id("layer.0.SelfAttention.relative_attention_bias.weight"),
            ffn_norm: id("layer.1.layer_norm.weight"),
            wi_0: id("layer.1.DenseReluDense.wi_0.weight"),
            wi_1: id("layer.1.DenseReluDense.wi_1.weight"),
            wo: id("layer.1.DenseReluDense.wo.weight"),
        }
    }
}

/// Per-block residency handles (all but the relative-position-bias table,
/// which is CPU-decoded to f32 and uploaded compactly per layer - see
/// [`Umt5Encoder::forward_taps`]).
#[derive(Clone, Copy, Debug)]
pub struct Umt5BlockHandles {
    pub attn_norm: WeightHandle,
    pub q: WeightHandle,
    pub k: WeightHandle,
    pub v: WeightHandle,
    pub o: WeightHandle,
    pub ffn_norm: WeightHandle,
    pub wi_0: WeightHandle,
    pub wi_1: WeightHandle,
    pub wo: WeightHandle,
}

#[derive(Clone, Debug)]
pub struct Umt5Handles {
    pub layers: Vec<Umt5BlockHandles>,
    pub final_norm: WeightHandle,
}

#[derive(Debug)]
pub enum LoadError {
    UnknownWeight(WeightId),
    Undecodable {
        id: WeightId,
        encoding: Option<StorageEncoding>,
        label: String,
    },
}

/// Register all encoder weights with residency. `transcode`: optional
/// load-time requantize target for the 7 matmul weights per layer (q/k/v/o +
/// wi_0/wi_1/wo), mirroring `register_qwen3_handles`. Norms stay dense. The
/// `relative_attention_bias` table and the `shared` embed table are NOT
/// registered (decoded directly, see module docs).
pub fn register_umt5_handles<S: WeightSource>(
    residency: &WeightResidency<S>,
    transcode: Option<thinfer_core::quant::QuantKind>,
) -> Result<Umt5Handles, LoadError> {
    let mut layers = Vec::with_capacity(config::N_LAYERS);
    for i in 0..config::N_LAYERS {
        let w = Umt5BlockWeights::new(i);
        let lin = |id: &WeightId| register_one(residency, id, TransposePolicy::Linear2D, transcode);
        let norm = |id: &WeightId| register_one(residency, id, TransposePolicy::None, None);
        layers.push(Umt5BlockHandles {
            attn_norm: norm(&w.attn_norm)?,
            q: lin(&w.q)?,
            k: lin(&w.k)?,
            v: lin(&w.v)?,
            o: lin(&w.o)?,
            ffn_norm: norm(&w.ffn_norm)?,
            wi_0: lin(&w.wi_0)?,
            wi_1: lin(&w.wi_1)?,
            wo: lin(&w.wo)?,
        });
    }
    let final_norm = register_one(
        residency,
        &WeightId("encoder.final_layer_norm.weight".into()),
        TransposePolicy::None,
        None,
    )?;
    Ok(Umt5Handles { layers, final_norm })
}

fn register_one<S: WeightSource>(
    residency: &WeightResidency<S>,
    id: &WeightId,
    transpose: TransposePolicy,
    transcode: Option<thinfer_core::quant::QuantKind>,
) -> Result<WeightHandle, LoadError> {
    let entry = residency
        .source()
        .catalog()
        .get(id)
        .ok_or_else(|| LoadError::UnknownWeight(id.clone()))?;
    let encoding = entry.encoding.ok_or_else(|| LoadError::Undecodable {
        id: id.clone(),
        encoding: None,
        label: entry.encoding_label.clone(),
    })?;
    // Mirror `text_encoder::register_one`: transcode targets keep the file's
    // [N, K] row order (GGUF block layout is N-major). In-file quant registers
    // as-is; bf16/f32 keeps the requested transpose.
    let (transpose, transcode) = match (encoding, transcode) {
        (StorageEncoding::Bf16, Some(k)) => {
            assert_eq!(entry.shape.0.len(), 2, "transcode target must be 2-D");
            assert_eq!(
                entry.shape.0[1] % 32,
                0,
                "transcode requires K % 32 == 0 ({id:?})"
            );
            (TransposePolicy::None, Some(k))
        }
        (StorageEncoding::Quant(_), _) => (TransposePolicy::None, None),
        (StorageEncoding::Bf16 | StorageEncoding::F32, None) => (transpose, None),
        _ => {
            return Err(LoadError::Undecodable {
                id: id.clone(),
                encoding: Some(encoding),
                label: entry.encoding_label.clone(),
            });
        }
    };
    Ok(residency.register(WeightMeta {
        id: id.clone(),
        shape: entry.shape.clone(),
        encoding,
        on_disk_bytes: entry.size,
        transpose,
        transcode,
    }))
}

// ---------------------------------------------------------------------------
// CPU tensor loads (embedding gather + relpos-bias tables)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum TensorLoadError {
    Missing(WeightId),
    BadShape { id: WeightId, shape: Vec<usize> },
    Undecodable(StorageEncoding),
    TokenOutOfRange { id: u32, vocab: usize },
    Source(String),
    Reader(String),
    Decode(thinfer_core::weight::DecodeError),
}

/// CPU gather over `shared.weight` (the `[VOCAB, D_MODEL]` embedding). Reads
/// only the prompt's rows, decoded to f32. Mirrors
/// `text_encoder::embed_lookup`. Output is `[token_ids.len(), D_MODEL]` f32.
pub async fn embed_lookup<S: WeightSource>(
    source: &S,
    token_ids: &[u32],
) -> Result<Vec<f32>, TensorLoadError> {
    let id = WeightId(config::EMBED_TOKENS.into());
    let entry = source
        .catalog()
        .get(&id)
        .ok_or_else(|| TensorLoadError::Missing(id.clone()))?;
    let shape = &entry.shape.0;
    if shape.len() != 2 || shape[1] != config::D_MODEL {
        return Err(TensorLoadError::BadShape {
            id: id.clone(),
            shape: shape.clone(),
        });
    }
    let vocab = shape[0];
    let encoding = entry
        .encoding
        .ok_or(TensorLoadError::Undecodable(StorageEncoding::F32))?;
    let quant = match encoding {
        StorageEncoding::Bf16 | StorageEncoding::F32 => None,
        StorageEncoding::Quant(k) => {
            if !config::D_MODEL.is_multiple_of(k.block_size() as usize) {
                return Err(TensorLoadError::Undecodable(encoding));
            }
            Some(k)
        }
        enc => return Err(TensorLoadError::Undecodable(enc)),
    };
    let row_src_bytes: u64 = match (encoding, quant) {
        (_, Some(k)) => k.bytes_for_elements(config::D_MODEL as u64),
        (StorageEncoding::Bf16, _) => (config::D_MODEL as u64) * 2,
        _ => (config::D_MODEL as u64) * 4,
    };

    let mut reader = source
        .open(&id)
        .await
        .map_err(|e| TensorLoadError::Source(format!("{e:?}")))?;
    let mut out = vec![0f32; token_ids.len() * config::D_MODEL];
    let mut row_src = vec![0u8; row_src_bytes as usize];
    let row_dst_bytes = config::D_MODEL * 4;
    for (i, &tok) in token_ids.iter().enumerate() {
        if (tok as usize) >= vocab {
            return Err(TensorLoadError::TokenOutOfRange { id: tok, vocab });
        }
        let off = (tok as u64) * row_src_bytes;
        reader
            .read_at(off, &mut row_src)
            .await
            .map_err(|e| TensorLoadError::Reader(format!("{e:?}")))?;
        if let Some(k) = quant {
            let dst = &mut out[i * config::D_MODEL..(i + 1) * config::D_MODEL];
            thinfer_core::quant::dequantize_row(k, &row_src, dst);
            continue;
        }
        let dst_byte_off = i * row_dst_bytes;
        let out_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut out[..]);
        let mut decoder = Decoder::new(encoding).map_err(TensorLoadError::Decode)?;
        let n = decoder
            .feed(
                &row_src,
                &mut out_bytes[dst_byte_off..dst_byte_off + row_dst_bytes],
            )
            .map_err(TensorLoadError::Decode)?;
        decoder.finish().map_err(TensorLoadError::Decode)?;
        debug_assert_eq!(n, row_dst_bytes);
    }
    Ok(out)
}

/// Decode a whole small tensor to f32. Used for the per-layer
/// `relative_attention_bias` table `[NUM_BUCKETS, N_HEADS]` (~2 KiB), which
/// feeds `relpos_bias` as an f32 binding (the op reads `array<f32>`). Supports
/// bf16/f32 (the SkyReels Diffusers bundle ships umT5 fp32); quant errors out.
pub async fn load_tensor_f32<S: WeightSource>(
    source: &S,
    id: &WeightId,
) -> Result<Vec<f32>, TensorLoadError> {
    let entry = source
        .catalog()
        .get(id)
        .ok_or_else(|| TensorLoadError::Missing(id.clone()))?;
    let n_elems: usize = entry.shape.0.iter().product();
    let encoding = entry
        .encoding
        .ok_or(TensorLoadError::Undecodable(StorageEncoding::F32))?;
    if !matches!(encoding, StorageEncoding::Bf16 | StorageEncoding::F32) {
        return Err(TensorLoadError::Undecodable(encoding));
    }
    let mut reader = source
        .open(id)
        .await
        .map_err(|e| TensorLoadError::Source(format!("{e:?}")))?;
    let mut src = vec![0u8; entry.size as usize];
    reader
        .read_at(0, &mut src)
        .await
        .map_err(|e| TensorLoadError::Reader(format!("{e:?}")))?;
    let mut out = vec![0f32; n_elems];
    let out_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut out[..]);
    let mut decoder = Decoder::new(encoding).map_err(TensorLoadError::Decode)?;
    decoder
        .feed(&src, out_bytes)
        .map_err(TensorLoadError::Decode)?;
    decoder.finish().map_err(TensorLoadError::Decode)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Pipelines: reuse the DiT BlockPipelines + the two umT5-specific ops
// ---------------------------------------------------------------------------

/// umT5 reuses the DiT `BlockPipelines` (rmsnorm, sdpa, the 4 matmul sites)
/// and adds the two ops the DiT block lacks: gated-gelu and the relpos-bias
/// mask expander.
pub struct Umt5Pipelines {
    pub block: BlockPipelines,
    pub gelu_mul: WgpuPipeline,
    pub relpos_bias: WgpuPipeline,
}

impl Umt5Pipelines {
    pub async fn compile(
        backend: &WgpuBackend,
        cfgs: &BlockWgslConfigs,
    ) -> Result<Self, WgpuError> {
        let block = BlockPipelines::compile(backend, cfgs).await?;
        let gelu_mul = backend
            .create_pipeline(
                "umt5_gelu_mul",
                <GeluMulF32 as Op>::wgsl(&cfgs.ops),
                "main",
                <GeluMulF32 as Op>::layout(),
            )
            .await?;
        let relpos_bias = backend
            .create_pipeline(
                "umt5_relpos_bias",
                <RelposBiasF32 as RelposBiasOp>::wgsl(&cfgs.ops),
                "main",
                <RelposBiasF32 as RelposBiasOp>::layout(),
            )
            .await?;
        Ok(Self {
            block,
            gelu_mul,
            relpos_bias,
        })
    }

    fn act_dtype(&self) -> ActDtype {
        self.block.act_dtype
    }
}

// ---------------------------------------------------------------------------
// Block forward
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Umt5BlockShape {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub inner: usize,
    /// Sequence length (B=1).
    pub seq: usize,
    pub norm_eps: f32,
}

impl Umt5BlockShape {
    pub fn default_from_config(seq: usize) -> Self {
        Self {
            d_model: config::D_MODEL,
            n_heads: config::N_HEADS,
            d_kv: config::D_KV,
            d_ff: config::D_FF,
            inner: config::INNER,
            seq,
            norm_eps: config::LN_EPS,
        }
    }
}

pub struct Umt5BlockViews<'a> {
    pub attn_norm: GpuView<'a>,
    pub q: GpuView<'a>,
    pub k: GpuView<'a>,
    pub v: GpuView<'a>,
    pub o: GpuView<'a>,
    pub ffn_norm: GpuView<'a>,
    pub wi_0: GpuView<'a>,
    pub wi_1: GpuView<'a>,
    pub wo: GpuView<'a>,
}

impl Umt5BlockHandles {
    pub async fn acquire<'a, S: WeightSource>(
        &self,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<Umt5BlockViews<'a>, ResidencyError<S::Error, WgpuError>> {
        Ok(Umt5BlockViews {
            attn_norm: residency.acquire(self.attn_norm, backend).await?,
            q: residency.acquire(self.q, backend).await?,
            k: residency.acquire(self.k, backend).await?,
            v: residency.acquire(self.v, backend).await?,
            o: residency.acquire(self.o, backend).await?,
            ffn_norm: residency.acquire(self.ffn_norm, backend).await?,
            wi_0: residency.acquire(self.wi_0, backend).await?,
            wi_1: residency.acquire(self.wi_1, backend).await?,
            wo: residency.acquire(self.wo, backend).await?,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Umt5BlockBufs {
    pub attn_norm: thinfer_core::backend::BufRef,
    pub q: thinfer_core::backend::BufRef,
    pub k: thinfer_core::backend::BufRef,
    pub v: thinfer_core::backend::BufRef,
    pub o: thinfer_core::backend::BufRef,
    pub ffn_norm: thinfer_core::backend::BufRef,
    pub wi_0: thinfer_core::backend::BufRef,
    pub wi_1: thinfer_core::backend::BufRef,
    pub wo: thinfer_core::backend::BufRef,
}

impl Umt5BlockViews<'_> {
    pub fn bufs(&self) -> Umt5BlockBufs {
        Umt5BlockBufs {
            attn_norm: self.attn_norm.buf(),
            q: self.q.buf(),
            k: self.k.buf(),
            v: self.v.buf(),
            o: self.o.buf(),
            ffn_norm: self.ffn_norm.buf(),
            wi_0: self.wi_0.buf(),
            wi_1: self.wi_1.buf(),
            wo: self.wo.buf(),
        }
    }
}

/// GPU tap destinations for one umT5 block (parity diagnostics).
#[derive(Default, Clone)]
pub struct Umt5BlockTaps {
    pub n1: Option<thinfer_core::backend::BufRef>,
    pub q: Option<thinfer_core::backend::BufRef>,
    pub k: Option<thinfer_core::backend::BufRef>,
    pub v: Option<thinfer_core::backend::BufRef>,
    pub sa: Option<thinfer_core::backend::BufRef>,
    pub proj: Option<thinfer_core::backend::BufRef>,
    pub after_attn: Option<thinfer_core::backend::BufRef>,
    pub n2: Option<thinfer_core::backend::BufRef>,
    pub wi0: Option<thinfer_core::backend::BufRef>,
    pub wi1: Option<thinfer_core::backend::BufRef>,
    pub gu: Option<thinfer_core::backend::BufRef>,
    pub wo: Option<thinfer_core::backend::BufRef>,
}

/// One umT5 encoder block: pre-norm bidirectional self-attn (relpos-bias
/// per-head mask, scale 1.0) -> residual -> pre-norm gated-GELU FFN ->
/// residual.
pub struct Umt5Block {
    pub shape: Umt5BlockShape,
}

impl Umt5Block {
    pub fn new(shape: Umt5BlockShape) -> Self {
        Self { shape }
    }

    /// Append one block's dispatches to `scope`. `mask` is the per-head relpos
    /// bias `[H, S, S]` (act dtype) the caller built via `relpos_bias`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &Umt5Pipelines,
        x_in: BatchBuf<'wsp>,
        mask: BatchBuf<'wsp>,
        y_out: BatchBuf<'wsp>,
        bufs: &'wsp Umt5BlockBufs,
        taps: &Umt5BlockTaps,
    ) -> Result<(), WgpuError> {
        let s = self.shape;
        let bp = &pipelines.block;
        let rows = s.seq as u32;
        let dim = s.d_model as u32;
        let inner = s.inner as u32;
        let hd = s.d_kv as u32;
        let nh = s.n_heads as u32;
        let dff = s.d_ff as u32;
        let eps = s.norm_eps;
        let x_in = ActBuf::dense(x_in);
        let y_out = ActBuf::dense(y_out);

        let an = scope.import(&bufs.attn_norm);
        let q_w = scope.import(&bufs.q);
        let k_w = scope.import(&bufs.k);
        let v_w = scope.import(&bufs.v);
        let o_w = scope.import(&bufs.o);
        let fn_w = scope.import(&bufs.ffn_norm);
        let wi0_w = scope.import(&bufs.wi_0);
        let wi1_w = scope.import(&bufs.wi_1);
        let wo_w = scope.import(&bufs.wo);

        // --- attn pre-norm (T5LayerNorm == RMSNorm) ---
        let n1 = alloc_act(scope, bp, rows, dim)?;
        op_rmsnorm(scope, bp, x_in, an, n1, rows, dim, eps)?;
        copy_tap(scope, n1.data, taps.n1.as_ref(), bp.act_bytes(rows * dim))?;

        // --- q/k/v projections (separate, square) ---
        let q_s = alloc_matmul_out_buf(scope, bp, rows * inner)?;
        let k_s = alloc_matmul_out_buf(scope, bp, rows * inner)?;
        let v_s = alloc_matmul_out_buf(scope, bp, rows * inner)?;
        for (w, out) in [(q_w, q_s), (k_w, k_s), (v_w, v_s)] {
            let dims = scope.u32x4_uniform(rows, inner, dim, 0)?;
            Block::dispatch_matmul_site(
                scope,
                bp,
                n1,
                w,
                out,
                dims,
                bp.matmul_i8_qkv.as_ref(),
                bp.dequant_i8_qkv.as_ref(),
                bp.dequant_qkv.as_ref(),
                &bp.matmul_qkv,
                &bp.matmuls.qkv,
                rows,
                inner,
                dim,
            )?;
        }
        let q = ActBuf::dense(q_s);
        let k = ActBuf::dense(k_s);
        let v = ActBuf::dense(v_s);
        copy_tap(scope, q.data, taps.q.as_ref(), bp.act_bytes(rows * inner))?;
        copy_tap(scope, k.data, taps.k.as_ref(), bp.act_bytes(rows * inner))?;
        copy_tap(scope, v.data, taps.v.as_ref(), bp.act_bytes(rows * inner))?;

        // --- bidirectional sdpa, scale 1.0, per-head relpos mask (mode 2) ---
        let sa = alloc_act(scope, bp, rows, inner)?;
        op_sdpa(
            scope, bp, q, k, v, mask, sa, 1, rows, rows, nh, nh, hd, 1.0, 2,
        )?;
        copy_tap(scope, sa.data, taps.sa.as_ref(), bp.act_bytes(rows * inner))?;

        // --- o_proj + residual ---
        let proj_s = alloc_matmul_out_buf(scope, bp, rows * dim)?;
        let dims_o = scope.u32x4_uniform(rows, dim, inner, 0)?;
        Block::dispatch_matmul_site(
            scope,
            bp,
            sa,
            o_w,
            proj_s,
            dims_o,
            bp.matmul_i8_proj.as_ref(),
            bp.dequant_i8_proj.as_ref(),
            bp.dequant_proj.as_ref(),
            &bp.matmul_proj,
            &bp.matmuls.proj,
            rows,
            dim,
            inner,
        )?;
        copy_tap(scope, proj_s, taps.proj.as_ref(), bp.act_bytes(rows * dim))?;
        let after_attn = alloc_act(scope, bp, rows, dim)?;
        op_add(scope, bp, x_in, ActBuf::dense(proj_s), after_attn)?;
        copy_tap(
            scope,
            after_attn.data,
            taps.after_attn.as_ref(),
            bp.act_bytes(rows * dim),
        )?;

        // --- ffn pre-norm ---
        let n2 = alloc_act(scope, bp, rows, dim)?;
        op_rmsnorm(scope, bp, after_attn, fn_w, n2, rows, dim, eps)?;
        copy_tap(scope, n2.data, taps.n2.as_ref(), bp.act_bytes(rows * dim))?;

        // --- gated-GELU FFN: wo(gelu_new(wi_0(x)) * wi_1(x)) ---
        let wi0_s = alloc_matmul_out_buf(scope, bp, rows * dff)?;
        let dims_wi0 = scope.u32x4_uniform(rows, dff, dim, 0)?;
        Block::dispatch_matmul_site(
            scope,
            bp,
            n2,
            wi0_w,
            wi0_s,
            dims_wi0,
            bp.matmul_i8_ffn_up.as_ref(),
            bp.dequant_i8_ffn_up.as_ref(),
            bp.dequant_ffn_up.as_ref(),
            &bp.matmul_ffn_up,
            &bp.matmuls.ffn_up,
            rows,
            dff,
            dim,
        )?;
        let wi1_s = alloc_matmul_out_buf(scope, bp, rows * dff)?;
        let dims_wi1 = scope.u32x4_uniform(rows, dff, dim, 0)?;
        Block::dispatch_matmul_site(
            scope,
            bp,
            n2,
            wi1_w,
            wi1_s,
            dims_wi1,
            bp.matmul_i8_ffn_up.as_ref(),
            bp.dequant_i8_ffn_up.as_ref(),
            bp.dequant_ffn_up.as_ref(),
            &bp.matmul_ffn_up,
            &bp.matmuls.ffn_up,
            rows,
            dff,
            dim,
        )?;
        copy_tap(scope, wi0_s, taps.wi0.as_ref(), bp.act_bytes(rows * dff))?;
        copy_tap(scope, wi1_s, taps.wi1.as_ref(), bp.act_bytes(rows * dff))?;
        let gu = alloc_act(scope, bp, rows, dff)?;
        scope.dispatch_op::<GeluMulF32>(&pipelines.gelu_mul, &[wi0_s, wi1_s], gu.data)?;
        copy_tap(scope, gu.data, taps.gu.as_ref(), bp.act_bytes(rows * dff))?;
        let wo_s = alloc_matmul_out_buf(scope, bp, rows * dim)?;
        let dims_wo = scope.u32x4_uniform(rows, dim, dff, 0)?;
        Block::dispatch_matmul_site(
            scope,
            bp,
            gu,
            wo_w,
            wo_s,
            dims_wo,
            bp.matmul_i8_ffn_down.as_ref(),
            bp.dequant_i8_ffn_down.as_ref(),
            bp.dequant_ffn_down.as_ref(),
            &bp.matmul_ffn_down,
            &bp.matmuls.ffn_down,
            rows,
            dim,
            dff,
        )?;
        copy_tap(scope, wo_s, taps.wo.as_ref(), bp.act_bytes(rows * dim))?;
        op_add(scope, bp, after_attn, ActBuf::dense(wo_s), y_out)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Encoder driver
// ---------------------------------------------------------------------------

/// umT5 encoder: CPU embed lookup, N bidirectional blocks (per-layer relpos
/// bias), final RMSNorm. Output is the encoder last hidden state
/// `[seq, D_MODEL]` f32.
pub struct Umt5Encoder {
    max_seq: usize,
}

impl Umt5Encoder {
    pub fn new(max_seq: usize) -> Self {
        Self {
            max_seq: max_seq.max(1),
        }
    }

    pub async fn forward<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipelines: &Umt5Pipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        handles: &Umt5Handles,
        source: &S,
        token_ids: &[u32],
    ) -> Result<Umt5Output, Umt5ForwardError<S::Error>> {
        self.forward_taps(
            backend, pipelines, residency, scratch, handles, source, token_ids, None,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn forward_taps<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipelines: &Umt5Pipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        handles: &Umt5Handles,
        source: &S,
        token_ids: &[u32],
        mut taps: Option<&mut Umt5Taps>,
    ) -> Result<Umt5Output, Umt5ForwardError<S::Error>> {
        let seq = token_ids.len();
        assert!(seq > 0, "Umt5Encoder::forward: empty token list");
        // Even-pad for the packed act/mask layouts (2 elems/word; mask rows
        // need s_k even). The pad row repeats the last token; its output row
        // is sliced off after readback. Bidirectional attention means the pad
        // row is visible to real rows, so its relpos column must be benign:
        // it is (a duplicate key), and we slice the pad query row out. This
        // matches how the pyref must be fed (same even-padded ids).
        let mut ids = token_ids.to_vec();
        if !ids.len().is_multiple_of(2) {
            ids.push(*ids.last().expect("non-empty token list"));
        }
        let seq_pad = ids.len();
        assert!(
            seq_pad <= self.max_seq.max(seq_pad),
            "prompt length {seq_pad} exceeds Umt5Encoder max_seq {}",
            self.max_seq
        );

        let act = pipelines.act_dtype();
        let dim = config::D_MODEL as u32;
        let nh = config::N_HEADS as u32;
        let rows = seq_pad as u32;

        // --- CPU embed lookup (-> f32) ---
        let embeds = embed_lookup(source, &ids)
            .instrument(tracing::debug_span!(target: PHASE, "umt5.embed_lookup", seq))
            .await
            .map_err(Umt5ForwardError::Tensor)?;
        if let Some(t) = taps.as_deref_mut() {
            t.embeds = embeds.clone();
        }

        let shape = Umt5BlockShape::default_from_config(seq_pad);
        let block = Umt5Block::new(shape);
        let act_bytes = pipelines.block.act_bytes(rows * dim);

        // --- upload embeds to the activation buffer ---
        let x_buf = scratch.alloc(act_bytes)?;
        let embed_bytes = seq::act_upload_bytes(act, &embeds);
        backend.write_buffer(x_buf.id, 0, &embed_bytes)?;

        // --- bucket map (shared across layers), uploaded once as u32 ---
        let bucket_map =
            relpos_bucket_map(seq_pad, true, config::NUM_BUCKETS, config::MAX_DISTANCE);
        let bm_bytes: Vec<u8> = bucket_map.iter().flat_map(|x| x.to_le_bytes()).collect();
        let bm_buf = scratch.alloc(bm_bytes.len() as u64)?;
        backend.write_buffer(bm_buf.id, 0, &bm_bytes)?;

        // --- preload per-layer relpos tables (tiny: [32, 64] each) ---
        let mut tables: Vec<Vec<f32>> = Vec::with_capacity(config::N_LAYERS);
        for i in 0..config::N_LAYERS {
            let id = WeightId(format!(
                "encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight"
            ));
            tables.push(
                load_tensor_f32(source, &id)
                    .await
                    .map_err(Umt5ForwardError::Tensor)?,
            );
        }

        // One reusable per-op tap buffer set, filled by EVERY block and read
        // back after each block's submit (incremental, so memory stays at +12
        // buffers rather than +12*N_LAYERS).
        let want_block_ops = taps.as_deref().map(|t| t.want_block_ops).unwrap_or(false);
        let inner = config::INNER as u32;
        let dff = config::D_FF as u32;
        let tap_sizes: [u32; 12] = [
            rows * dim,   // n1
            rows * inner, // q
            rows * inner, // k
            rows * inner, // v
            rows * inner, // sa
            rows * dim,   // proj
            rows * dim,   // after_attn
            rows * dim,   // n2
            rows * dff,   // wi0
            rows * dff,   // wi1
            rows * dff,   // gu
            rows * dim,   // wo
        ];
        let mut tap_wsbufs = Vec::with_capacity(if want_block_ops { 12 } else { 0 });
        let block_taps = if want_block_ops {
            for n in tap_sizes {
                tap_wsbufs.push(scratch.alloc(pipelines.block.act_bytes(n))?);
            }
            let r = |i: usize| Some(tap_wsbufs[i].as_buf_ref());
            Some(Umt5BlockTaps {
                n1: r(0),
                q: r(1),
                k: r(2),
                v: r(3),
                sa: r(4),
                proj: r(5),
                after_attn: r(6),
                n2: r(7),
                wi0: r(8),
                wi1: r(9),
                gu: r(10),
                wo: r(11),
            })
        } else {
            None
        };

        // relpos mask elem/word count for the dispatch.
        let mask_n = if act.is_packed() {
            nh * rows * (rows / 2)
        } else {
            nh * rows * rows
        };

        let mut cur = x_buf;
        let mut pending: Option<Umt5BlockViews<'_>> = if handles.layers.is_empty() {
            None
        } else {
            Some(
                handles.layers[0]
                    .acquire(residency, backend)
                    .instrument(tracing::debug_span!(target: PHASE, "umt5.acquire", idx = 0_usize))
                    .await?,
            )
        };
        let bm_ref = bm_buf.as_buf_ref();
        for (idx, _h) in handles.layers.iter().enumerate() {
            let _g = trace::scope!(format!("umt5.layer.{idx}")).entered();
            let views = pending.take().expect("pending acquire missing");
            let bufs = views.bufs();

            // Upload this layer's relpos table (f32) and build the mask.
            let table_bytes: Vec<u8> = tables[idx].iter().flat_map(|x| x.to_le_bytes()).collect();
            let table_buf = scratch.alloc(table_bytes.len() as u64)?;
            backend.write_buffer(table_buf.id, 0, &table_bytes)?;
            let mask_buf = scratch.alloc(pipelines.block.act_bytes(nh * rows * rows))?;

            let cur_ref = cur.as_buf_ref();
            let nxt = scratch.alloc(act_bytes)?;
            let nxt_ref = nxt.as_buf_ref();
            let table_ref = table_buf.as_buf_ref();
            let mask_ref = mask_buf.as_buf_ref();
            let scope = scratch.batch();
            let cur_h = scope.import(&cur_ref);
            let nxt_h = scope.import(&nxt_ref);
            let table_h = scope.import(&table_ref);
            let bm_h = scope.import(&bm_ref);
            let mask_h = scope.import(&mask_ref);
            // relpos uniform {H, S, _, _}.
            let relpos_u = scope.u32x4_uniform(nh, rows, 0, 0)?;
            scope.relpos_bias::<RelposBiasF32>(
                &pipelines.relpos_bias,
                table_h,
                bm_h,
                relpos_u,
                mask_h,
                mask_n,
            )?;
            match block_taps.as_ref() {
                Some(t) => block.forward(&scope, pipelines, cur_h, mask_h, nxt_h, &bufs, t)?,
                None => block.forward(
                    &scope,
                    pipelines,
                    cur_h,
                    mask_h,
                    nxt_h,
                    &bufs,
                    &Umt5BlockTaps::default(),
                )?,
            }

            let next_idx = idx + 1;
            let next_acquire = async {
                match handles.layers.get(next_idx) {
                    Some(h) => {
                        let span =
                            tracing::debug_span!(target: PHASE, "umt5.acquire", idx = next_idx);
                        Ok::<_, ResidencyError<S::Error, WgpuError>>(Some(
                            h.acquire(residency, backend).instrument(span).await?,
                        ))
                    }
                    None => Ok(None),
                }
            };
            let submit_fut = scope
                .submit_void()
                .instrument(tracing::debug_span!(target: PHASE, "umt5.submit", idx));
            let (submit_res, next_res) = futures::join!(submit_fut, next_acquire);
            submit_res?;
            pending = next_res?;

            if let Some(t) = taps.as_deref_mut()
                && t.want_layer_outputs
            {
                let bytes = backend.read_buffer(nxt.id(), 0, act_bytes).await?;
                t.layer_outputs.push(seq::act_readback_to_f32(
                    act,
                    &bytes,
                    seq_pad * config::D_MODEL,
                ));
            }

            // Incremental per-op readback: the reusable tap buffers hold THIS
            // block's intermediates; read them before the next block overwrites.
            if want_block_ops {
                let mut decoded: Vec<Vec<f32>> = Vec::with_capacity(tap_wsbufs.len());
                for (b, n) in tap_wsbufs.iter().zip(tap_sizes) {
                    let bytes = backend
                        .read_buffer(b.id(), 0, pipelines.block.act_bytes(n))
                        .await?;
                    decoded.push(seq::act_readback_to_f32(act, &bytes, n as usize));
                }
                let mut it = decoded.into_iter();
                let mut nx = || it.next().expect("12 tap fields");
                let ops = Umt5BlockOpsHost {
                    n1: nx(),
                    q: nx(),
                    k: nx(),
                    v: nx(),
                    sa: nx(),
                    proj: nx(),
                    after_attn: nx(),
                    n2: nx(),
                    wi0: nx(),
                    wi1: nx(),
                    gu: nx(),
                    wo: nx(),
                };
                if let Some(t) = taps.as_deref_mut() {
                    t.block_ops.push(ops);
                }
            }

            drop(views);
            cur = nxt;
        }

        // --- final_layer_norm, then readback (drop the even-pad row) ---
        let final_view = residency.acquire(handles.final_norm, backend).await?;
        let fn_buf = final_view.buf();
        let normed = scratch.alloc(act_bytes)?;
        {
            let cur_ref = cur.as_buf_ref();
            let normed_ref = normed.as_buf_ref();
            let scope = scratch.batch();
            let cur_h = scope.import(&cur_ref);
            let fn_h = scope.import(&fn_buf);
            let out_h = scope.import(&normed_ref);
            op_rmsnorm(
                &scope,
                &pipelines.block,
                ActBuf::dense(cur_h),
                fn_h,
                ActBuf::dense(out_h),
                rows,
                dim,
                config::LN_EPS,
            )?;
            scope.submit_void().await?;
        }
        drop(final_view);

        let bytes = backend.read_buffer(normed.id(), 0, act_bytes).await?;
        let mut hidden = seq::act_readback_to_f32(act, &bytes, seq_pad * config::D_MODEL);
        hidden.truncate(seq * config::D_MODEL);
        Ok(Umt5Output { hidden, seq })
    }
}

/// Host-side parity tap sinks for [`Umt5Encoder::forward_taps`].
#[derive(Default)]
pub struct Umt5Taps {
    /// Capture each block's post-residual output `[seq_pad, D_MODEL]`.
    pub want_layer_outputs: bool,
    pub layer_outputs: Vec<Vec<f32>>,
    /// Capture per-op intermediates of EVERY block, read back incrementally
    /// after each block's submit (one reusable GPU tap set). `block_ops[i]` is
    /// block `i`'s op trace; length == N_LAYERS when set.
    pub want_block_ops: bool,
    pub block_ops: Vec<Umt5BlockOpsHost>,
    /// f32 embeds as gathered, `[seq_pad, D_MODEL]`.
    pub embeds: Vec<f32>,
}

/// Decoded per-op intermediates for the tapped layer.
#[derive(Default)]
pub struct Umt5BlockOpsHost {
    pub n1: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub sa: Vec<f32>,
    pub proj: Vec<f32>,
    pub after_attn: Vec<f32>,
    pub n2: Vec<f32>,
    pub wi0: Vec<f32>,
    pub wi1: Vec<f32>,
    pub gu: Vec<f32>,
    pub wo: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Umt5Output {
    /// `[seq, D_MODEL]` row-major f32 (encoder last hidden state, post
    /// `final_layer_norm`).
    pub hidden: Vec<f32>,
    pub seq: usize,
}

#[derive(Debug)]
pub enum Umt5ForwardError<SE: core::fmt::Debug> {
    Tensor(TensorLoadError),
    Wgpu(WgpuError),
    Residency(ResidencyError<SE, WgpuError>),
}

impl<SE: core::fmt::Debug> From<WgpuError> for Umt5ForwardError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}

impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for Umt5ForwardError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_weight_names() {
        let w = Umt5BlockWeights::new(3);
        assert_eq!(w.q.0, "encoder.block.3.layer.0.SelfAttention.q.weight");
        assert_eq!(
            w.relpos_bias.0,
            "encoder.block.3.layer.0.SelfAttention.relative_attention_bias.weight"
        );
        assert_eq!(
            w.wi_0.0,
            "encoder.block.3.layer.1.DenseReluDense.wi_0.weight"
        );
        assert_eq!(w.wo.0, "encoder.block.3.layer.1.DenseReluDense.wo.weight");
    }

    #[test]
    fn inner_equals_d_model() {
        assert_eq!(config::INNER, config::D_MODEL);
        assert_eq!(config::N_HEADS * config::D_KV, config::INNER);
    }
}
