//! LTX-2.3 text-conditioning tail: FeatureExtractor V2 + the 8-layer gated
//! embeddings connector (per modality: video 4096, audio 2048). Consumes the
//! Gemma encoder's 49 hidden states and produces the DiT cross-attention KV
//! caption tensors (`video [1024, 4096]`, `audio [1024, 2048]`, all positions
//! valid; learnable registers fill the right-pad slots).
//!
//! Ground truth: `third-party/LTX-2/.../text_encoders/gemma/`
//! (`feature_extractor.py` V2, `embeddings_connector.py`, `embeddings_processor.py`)
//! + `model/transformer/{attention,ops,rope,feed_forward}.py`.
//!
//! Weight provenance (verified on disk):
//! - FE V2 aggregate embeds (`text_embedding_projection.{video,audio}_aggregate_
//!   embed.{weight,bias}`, bf16 `[out, 188160]`/`[out]`) live in the CONNECTOR
//!   SAFETENSORS.
//! - The 8 connector blocks + `learnable_registers` live in the DiT GGUF under
//!   `{video,audio}_embeddings_connector.*` (matmuls Q8_0, norms/biases/registers
//!   F32). The DiT GGUF is the connector's weight file, NOT the safetensors.
//!
//! Precision: F32 acts, bf16 weights (F32 GGUF tensors upload bf16 per
//! `residency::WeightMeta::gpu_encoding`). Same regime as the Gemma encoder; the
//! connector runs once per request so the cost is irrelevant.
//!
//! Connector block (`_BasicTransformerBlock1D`):
//! ```text
//! n = rms_norm(h)                          // weightless, eps 1e-6, over inner
//! q,k,v = to_{q,k,v}(n) + bias             // Linear inner->inner
//! q,k   = {q,k}_norm(q,k)                   // RMSNorm over FULL inner (with weight)
//! q,k   = rope_split(q,k)                   // half-rot, per-head freq slice
//! a     = sdpa(q,k,v)                       // 32 heads, non-causal (mask zeroed)
//! a     = a * 2*sigmoid(to_gate_logits(n))  // per-head gate
//! h     = h + to_out(a) + bias
//! n2    = rms_norm(h)                       // weightless
//! h     = h + net2(gelu_tanh(net0.proj(n2) + b) ) + b   // FF, inner->4*inner->inner
//! ```
//! After 8 blocks: `h = rms_norm(h)` (weightless `norm_out`).
//!
//! RoPE is SPLIT (half-rot) but with PER-HEAD-DISTINCT frequencies (the freq grid
//! spans the full inner_dim, so head h owns the `[h*pairs, (h+1)*pairs)` slice).
//! The shared half-rot kernel broadcasts one freq row across all heads, so we feed
//! it `rows = S*heads, heads = 1`: each (position, head) becomes its own kernel row
//! and reads its own freq row from a `[S*heads, head_dim]` table. Bit-identical to
//! per-head freqs, no kernel change.

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError, WgpuPipeline};
use thinfer_core::ops::{
    BcastAddF32, GatedHeadMulF32, GeluF32, MatMulConfig, MatMulF32, MatmulOp, Op, WeightDtype,
    WgslConfig,
};
use thinfer_core::residency::{
    GpuView, ResidencyError, TransposePolicy, WeightHandle, WeightResidency,
};
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace};

use super::config as dit;
use super::gemma;
use crate::common::block::{
    ActBuf, Block, BlockPipelines, BlockWgslConfigs, DenseActSites, DequantStep, alloc_act,
    alloc_matmul_out_buf, op_add, op_rmsnorm, op_rope_halfrot, op_sdpa,
};
use crate::common::embedders::bcast_add_uniform;
use crate::common::seq;
use crate::z_image::text_encoder::{LoadError, register_one};

/// Connector sequence length (LTX `max_length`); registers fill pad slots.
pub const CONN_SEQ: usize = 1024;
/// FE V2 RMS eps + flattened width (`HIDDEN * N_STATES`).
const FE_EPS: f32 = 1e-6;
const N_STATES: usize = gemma::N_LAYERS + 1;
const FE_FLAT: usize = gemma::HIDDEN * N_STATES; // 3840 * 49 = 188160

/// One connector modality (video / audio) geometry.
#[derive(Clone, Copy, Debug)]
pub struct Modality {
    /// GGUF prefix: `video_embeddings_connector` / `audio_embeddings_connector`.
    pub prefix: &'static str,
    pub inner_dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    /// Aggregate-embed Linear out width (= inner_dim).
    pub out_dim: usize,
}

pub const VIDEO: Modality = Modality {
    prefix: "video_embeddings_connector",
    inner_dim: dit::DIM,
    n_heads: dit::CONNECTOR_N_HEADS,
    head_dim: dit::CONNECTOR_HEAD_DIM,
    out_dim: dit::DIM,
};

pub const AUDIO: Modality = Modality {
    prefix: "audio_embeddings_connector",
    inner_dim: dit::AUDIO_DIM,
    n_heads: dit::AUDIO_CONNECTOR_N_HEADS,
    head_dim: dit::AUDIO_CONNECTOR_HEAD_DIM,
    out_dim: dit::AUDIO_DIM,
};

// ---------------------------------------------------------------------------
// FeatureExtractor V2 (per-token RMS over D per layer -> stack -> aggregate)
// ---------------------------------------------------------------------------

/// Per-token RMS-normalize each of the 49 hidden states over D and flatten
/// C-order to `[seq, FE_FLAT]` with `flat[t][d*49 + l] = normed`. Host-side
/// (the encoder already returns states on the host); the heavy aggregate-embed
/// Linears run on GPU. `states[l]` is `[seq, HIDDEN]` row-major.
pub fn feature_extractor_v2_flatten(states: &[Vec<f32>], seq: usize) -> Vec<f32> {
    assert_eq!(states.len(), N_STATES, "FE V2 needs all 49 states");
    let d = gemma::HIDDEN;
    let l_n = N_STATES;
    let mut flat = vec![0.0f32; seq * FE_FLAT];
    for (l, state) in states.iter().enumerate() {
        assert_eq!(state.len(), seq * d, "state {l} size");
        for t in 0..seq {
            let row = &state[t * d..(t + 1) * d];
            let mean_sq = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / d as f64;
            let inv = (1.0 / (mean_sq + FE_EPS as f64).sqrt()) as f32;
            let base = t * FE_FLAT;
            for (dd, &v) in row.iter().enumerate() {
                flat[base + dd * l_n + l] = v * inv;
            }
        }
    }
    flat
}

/// FE V2 aggregate-embed handles (from the connector safetensors).
#[derive(Clone, Debug)]
pub struct FeHandles {
    pub video_w: WeightHandle,
    pub video_b: WeightHandle,
    pub audio_w: WeightHandle,
    pub audio_b: WeightHandle,
}

pub fn register_fe<S: WeightSource>(
    residency: &WeightResidency<S>,
) -> Result<FeHandles, LoadError> {
    let lin = |s: &str| {
        register_one(
            residency,
            &WeightId(format!("text_embedding_projection.{s}.weight")),
            TransposePolicy::Linear2D,
            None,
        )
    };
    let bias = |s: &str| {
        register_one(
            residency,
            &WeightId(format!("text_embedding_projection.{s}.bias")),
            TransposePolicy::None,
            None,
        )
    };
    Ok(FeHandles {
        video_w: lin("video_aggregate_embed")?,
        video_b: bias("video_aggregate_embed")?,
        audio_w: lin("audio_aggregate_embed")?,
        audio_b: bias("audio_aggregate_embed")?,
    })
}

// ---------------------------------------------------------------------------
// Connector block weights / handles (from the DiT GGUF)
// ---------------------------------------------------------------------------

struct BlockWeightIds {
    q_w: WeightId,
    q_b: WeightId,
    k_w: WeightId,
    k_b: WeightId,
    v_w: WeightId,
    v_b: WeightId,
    o_w: WeightId,
    o_b: WeightId,
    q_norm: WeightId,
    k_norm: WeightId,
    gate_w: WeightId,
    gate_b: WeightId,
    ff0_w: WeightId,
    ff0_b: WeightId,
    ff2_w: WeightId,
    ff2_b: WeightId,
}

impl BlockWeightIds {
    fn new(prefix: &str, i: usize) -> Self {
        let p = format!("{prefix}.transformer_1d_blocks.{i}");
        let id = |s: &str| WeightId(format!("{p}.{s}"));
        Self {
            q_w: id("attn1.to_q.weight"),
            q_b: id("attn1.to_q.bias"),
            k_w: id("attn1.to_k.weight"),
            k_b: id("attn1.to_k.bias"),
            v_w: id("attn1.to_v.weight"),
            v_b: id("attn1.to_v.bias"),
            o_w: id("attn1.to_out.0.weight"),
            o_b: id("attn1.to_out.0.bias"),
            q_norm: id("attn1.q_norm.weight"),
            k_norm: id("attn1.k_norm.weight"),
            gate_w: id("attn1.to_gate_logits.weight"),
            gate_b: id("attn1.to_gate_logits.bias"),
            ff0_w: id("ff.net.0.proj.weight"),
            ff0_b: id("ff.net.0.proj.bias"),
            ff2_w: id("ff.net.2.weight"),
            ff2_b: id("ff.net.2.bias"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BlockHandles {
    q_w: WeightHandle,
    q_b: WeightHandle,
    k_w: WeightHandle,
    k_b: WeightHandle,
    v_w: WeightHandle,
    v_b: WeightHandle,
    o_w: WeightHandle,
    o_b: WeightHandle,
    q_norm: WeightHandle,
    k_norm: WeightHandle,
    gate_w: WeightHandle,
    gate_b: WeightHandle,
    ff0_w: WeightHandle,
    ff0_b: WeightHandle,
    ff2_w: WeightHandle,
    ff2_b: WeightHandle,
}

#[derive(Clone, Debug)]
pub struct ConnectorHandles {
    layers: Vec<BlockHandles>,
    /// `learnable_registers` `[128, inner]` (F32 in GGUF -> bf16; read to host).
    registers: WeightHandle,
}

/// Register one modality's 8 connector blocks + registers from the DiT GGUF.
pub fn register_connector<S: WeightSource>(
    residency: &WeightResidency<S>,
    m: Modality,
) -> Result<ConnectorHandles, LoadError> {
    let q8 = Some(thinfer_core::quant::QuantKind::Q8_0);
    // Matmul weights are Q8_0 in the GGUF -> registered as-is (transcode moot).
    let lin = |id: &WeightId| register_one(residency, id, TransposePolicy::Linear2D, q8);
    let dense = |id: &WeightId| register_one(residency, id, TransposePolicy::None, None);
    let mut layers = Vec::with_capacity(dit::CONNECTOR_NUM_LAYERS);
    for b in (0..dit::CONNECTOR_NUM_LAYERS).map(|i| BlockWeightIds::new(m.prefix, i)) {
        layers.push(BlockHandles {
            q_w: lin(&b.q_w)?,
            q_b: dense(&b.q_b)?,
            k_w: lin(&b.k_w)?,
            k_b: dense(&b.k_b)?,
            v_w: lin(&b.v_w)?,
            v_b: dense(&b.v_b)?,
            o_w: lin(&b.o_w)?,
            o_b: dense(&b.o_b)?,
            q_norm: dense(&b.q_norm)?,
            k_norm: dense(&b.k_norm)?,
            gate_w: lin(&b.gate_w)?,
            gate_b: dense(&b.gate_b)?,
            ff0_w: lin(&b.ff0_w)?,
            ff0_b: dense(&b.ff0_b)?,
            ff2_w: lin(&b.ff2_w)?,
            ff2_b: dense(&b.ff2_b)?,
        });
    }
    let registers = dense(&WeightId(format!("{}.learnable_registers", m.prefix)))?;
    Ok(ConnectorHandles { layers, registers })
}

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

/// Connector block config: Q8_0 matmuls, F32 acts, bf16 norm/bias weights,
/// dense acts (no DP4A act-quant; once-per-request and the gated-attn / gelu
/// A-sides can carry outliers). `head_dim` 128 (video) fits `SdpaF32`.
fn connector_block_cfgs() -> BlockWgslConfigs {
    use thinfer_core::ops::ActDtype;
    let ops = WgslConfig {
        bf16_quant_writes: false,
        act_dtype: ActDtype::F32,
        weight_dtype: WeightDtype::Bf16,
    };
    let mm = WgslConfig {
        weight_dtype: WeightDtype::Quant(thinfer_core::quant::QuantKind::Q8_0),
        ..ops
    };
    BlockWgslConfigs {
        matmul_qkv: mm,
        matmul_qkv_self: mm,
        matmul_proj: mm,
        matmul_ffn_up: mm,
        matmul_ffn_down: mm,
        matmul_adaln: ops,
        ops,
        i8_sdpa: false,
        dense_acts: DenseActSites {
            qkv: true,
            qkv_self: true,
            proj: true,
            ffn_up: true,
            ffn_down: true,
        },
        coopmat_acts: crate::common::block::CoopmatSites::default(),
        large_d_sdpa: false,
        fast_sdpa: false,
        decode_sdpa: false,
    }
}

pub struct ConnectorPipelines {
    pub block: BlockPipelines,
    /// Bf16-weight matmul for the FE V2 aggregate embeds (K = 188160).
    fe_matmul: WgpuPipeline,
    fe_matmul_op: MatMulF32,
    gelu: WgpuPipeline,
    gate: WgpuPipeline,
}

impl ConnectorPipelines {
    pub async fn compile(backend: &WgpuBackend) -> Result<Self, WgpuError> {
        let cfgs = connector_block_cfgs();
        let block = BlockPipelines::compile(backend, &cfgs).await?;
        let fe_cfg = WgslConfig {
            weight_dtype: WeightDtype::Bf16,
            ..cfgs.ops
        };
        let fe_matmul_op = MatMulF32::new(MatMulConfig::DEFAULT);
        let fe_wgsl = fe_matmul_op.wgsl(&fe_cfg);
        let fe_matmul = backend
            .create_pipeline(
                "ltx_fe_aggregate_matmul",
                &fe_wgsl,
                "main",
                <MatMulF32 as MatmulOp>::layout(),
            )
            .await?;
        let gelu = backend
            .create_pipeline(
                "ltx_conn_gelu",
                <GeluF32 as Op>::wgsl(&cfgs.ops),
                "main",
                <GeluF32 as Op>::layout(),
            )
            .await?;
        let gate = backend
            .create_pipeline(
                "ltx_conn_gate",
                <GatedHeadMulF32 as Op>::wgsl(&cfgs.ops),
                "main",
                <GatedHeadMulF32 as Op>::layout(),
            )
            .await?;
        Ok(Self {
            block,
            fe_matmul,
            fe_matmul_op,
            gelu,
            gate,
        })
    }
}

// ---------------------------------------------------------------------------
// RoPE freqs (SPLIT / half-rot, per-head-distinct, float64 grid)
// ---------------------------------------------------------------------------

/// Build the connector's per-head SPLIT-rope freq table as `[seq*heads, head_dim]`
/// interleaved `(cos, sin)` per pair. The freq grid spans the full inner_dim:
/// `indices[m] = theta^(m/(inner/2 - 1)) * pi/2`, `m in 0..inner/2`; head h pair j
/// reads `m = h*(head_dim/2) + j`. Angle = `indices[m] * (2*pos/max_pos - 1)`.
/// Computed in f64 then narrowed (float64 frequencies_precision).
fn build_connector_freqs(m: Modality, seq: usize) -> Vec<f32> {
    use std::f64::consts::PI;
    let theta = dit::ROPE_THETA;
    let max_pos = dit::CONNECTOR_MAX_POS as f64;
    let half = m.inner_dim / 2;
    let pairs = m.head_dim / 2;
    let indices: Vec<f64> = (0..half)
        .map(|i| theta.powf(i as f64 / (half - 1) as f64) * PI / 2.0)
        .collect();
    let mut out = vec![0.0f32; seq * m.n_heads * m.head_dim];
    for pos in 0..seq {
        let frac = 2.0 * (pos as f64) / max_pos - 1.0;
        for h in 0..m.n_heads {
            for j in 0..pairs {
                let angle = indices[h * pairs + j] * frac;
                let base = (pos * m.n_heads + h) * m.head_dim + 2 * j;
                out[base] = angle.cos() as f32;
                out[base + 1] = angle.sin() as f32;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum ConnError<SE: core::fmt::Debug> {
    Wgpu(WgpuError),
    Residency(ResidencyError<SE, WgpuError>),
}
impl<SE: core::fmt::Debug> From<WgpuError> for ConnError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for ConnError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

/// FE V2 aggregate-embed Linear for one modality: `out = (flat * sqrt(out/D)) @ W^T
/// + b`. The per-modality rescale folds into the host flat (cheap; seq is small).
/// Returns `[seq, out_dim]` host f32.
#[allow(clippy::too_many_arguments)]
pub async fn fe_aggregate<S: WeightSource>(
    backend: &WgpuBackend,
    pipelines: &ConnectorPipelines,
    residency: &WeightResidency<S>,
    scratch: &Workspace<WgpuBackend>,
    flat: &[f32],
    seq: usize,
    w: WeightHandle,
    b: WeightHandle,
    out_dim: usize,
) -> Result<Vec<f32>, ConnError<S::Error>> {
    let scale = (out_dim as f32 / gemma::HIDDEN as f32).sqrt();
    let scaled: Vec<f32> = flat.iter().map(|v| v * scale).collect();
    let bp = &pipelines.block;
    let flat_buf = scratch.alloc(bp.act_bytes((seq * FE_FLAT) as u32))?;
    backend.write_buffer(
        flat_buf.id,
        0,
        &seq::act_upload_bytes(bp.act_dtype, &scaled),
    )?;
    let out_bytes = bp.act_bytes((seq * out_dim) as u32);
    let out_buf = scratch.alloc(out_bytes)?;

    let w_view = residency.acquire(w, backend).await?;
    let b_view = residency.acquire(b, backend).await?;
    {
        let scope = scratch.batch();
        let x = scope.import_copy(flat_buf.as_buf_ref());
        let wv = scope.import_copy(w_view.buf());
        let pre = alloc_matmul_out_buf(&scope, bp, (seq * out_dim) as u32)?;
        let dims = scope.u32x4_uniform(seq as u32, out_dim as u32, FE_FLAT as u32, 0)?;
        scope.matmul(
            &pipelines.fe_matmul,
            &pipelines.fe_matmul_op,
            x,
            wv,
            dims,
            pre,
            seq as u32,
            out_dim as u32,
        )?;
        let bv = scope.import_copy(b_view.buf());
        let bu = bcast_add_uniform(&scope, out_dim as u32)?;
        let out_h = scope.import_copy(out_buf.as_buf_ref());
        scope.bcast_add::<BcastAddF32>(
            &bp.bcast_add,
            pre,
            bv,
            bu,
            out_h,
            (seq * out_dim) as u32,
        )?;
        scope.submit_void().await?;
    }
    let bytes = backend.read_buffer(out_buf.id(), 0, out_bytes).await?;
    Ok(seq::act_readback_to_f32(
        bp.act_dtype,
        &bytes,
        seq * out_dim,
    ))
}

/// Frame `[n_real, inner]` aggregate embeds into the `[CONN_SEQ, inner]` connector
/// input: valid tokens at rows `0..n_real`, then `learnable_registers[row % 128]`
/// (the registers are tiled with period 128 over the full sequence, replaced where
/// the binary mask is 0). `registers` is `[128, inner]` row-major host f32.
fn frame_with_registers(
    embeds: &[f32],
    n_real: usize,
    inner: usize,
    registers: &[f32],
) -> Vec<f32> {
    let mut framed = vec![0.0f32; CONN_SEQ * inner];
    framed[..n_real * inner].copy_from_slice(&embeds[..n_real * inner]);
    for row in n_real..CONN_SEQ {
        let reg = (row % dit::CONNECTOR_NUM_LEARNABLE_REGISTERS) * inner;
        framed[row * inner..(row + 1) * inner].copy_from_slice(&registers[reg..reg + inner]);
    }
    framed
}

/// bf16(1.0) packed two-per-word, enough words for the widest inner_dim. Used as
/// the weight for the weightless connector rms_norms (`rms_norm(x)` with no gain):
/// the F32-act/bf16-weight rmsnorm kernel reads the gain as packed bf16, so a
/// plain f32 ones buffer would be misread.
fn bf16_ones_words(inner: usize) -> Vec<u8> {
    let words = inner / 2;
    let mut v = Vec::with_capacity(words * 4);
    for _ in 0..words {
        v.extend_from_slice(&0x3F80_3F80u32.to_le_bytes());
    }
    v
}

struct ConnBufs {
    q_w: BufRef,
    q_b: BufRef,
    k_w: BufRef,
    k_b: BufRef,
    v_w: BufRef,
    v_b: BufRef,
    o_w: BufRef,
    o_b: BufRef,
    q_norm: BufRef,
    k_norm: BufRef,
    gate_w: BufRef,
    gate_b: BufRef,
    ff0_w: BufRef,
    ff0_b: BufRef,
    ff2_w: BufRef,
    ff2_b: BufRef,
}

/// Run the 8-layer gated connector for one modality. `aggregate` is the FE V2
/// output `[n_real, inner]` (host). Returns `[CONN_SEQ, inner]` host f32 (the DiT
/// cross-attn caption KV; all positions valid).
pub async fn connector_forward<S: WeightSource>(
    backend: &WgpuBackend,
    pipelines: &ConnectorPipelines,
    residency: &WeightResidency<S>,
    scratch: &Workspace<WgpuBackend>,
    handles: &ConnectorHandles,
    m: Modality,
    aggregate: &[f32],
    n_real: usize,
) -> Result<Vec<f32>, ConnError<S::Error>> {
    let bp = &pipelines.block;
    let inner = m.inner_dim;
    let rows = CONN_SEQ as u32;
    let dimu = inner as u32;

    // Registers (host) -> frame the connector input. The F32 GGUF tensor uploads
    // bf16 (gpu_encoding), so read the buffer as bf16 (2 bytes/elem), not at the
    // F32 act sizing; the engine thus uses bf16-rounded registers (matched in
    // the pyref).
    let reg_elems = dit::CONNECTOR_NUM_LEARNABLE_REGISTERS * inner;
    let reg_view = residency.acquire(handles.registers, backend).await?;
    let reg_bytes = backend
        .read_buffer(reg_view.buf().id, 0, reg_elems as u64 * 2)
        .await?;
    let registers =
        seq::act_readback_to_f32(thinfer_core::ops::ActDtype::Bf16, &reg_bytes, reg_elems);
    drop(reg_view);
    let framed = frame_with_registers(aggregate, n_real, inner, &registers);

    let act_bytes = bp.act_bytes((CONN_SEQ * inner) as u32);
    let mut cur = scratch.alloc(act_bytes)?;
    backend.write_buffer(cur.id, 0, &seq::act_upload_bytes(bp.act_dtype, &framed))?;

    // Per-head SPLIT-rope freqs `[rows*heads, head_dim]` (heads collapsed into
    // rows for the kernel; see module note).
    let freqs = build_connector_freqs(m, CONN_SEQ);
    let freqs_bytes = seq::freqs_upload_bytes(bp.act_dtype, &freqs);
    let freqs_buf = scratch.alloc(freqs_bytes.len() as u64)?;
    backend.write_buffer(freqs_buf.id, 0, &freqs_bytes)?;

    // Non-causal full attention (registers replace pads -> mask zeroed). The
    // SdpaF32 kernel still indexes the mask via `select(0.0, mask[...], ...)`
    // (the read is unconditional even at `has_mask = 0`), so the buffer must be
    // in-bounds: a full `[s_q, s_k]` zero mask (all-zero additive = no masking).
    let mask_elems = CONN_SEQ * CONN_SEQ;
    let mask_buf = scratch.alloc(bp.act_bytes(mask_elems as u32))?;
    backend.write_buffer(
        mask_buf.id,
        0,
        &vec![0u8; bp.act_bytes(mask_elems as u32) as usize],
    )?;

    // bf16-ones gain for the weightless rms_norms.
    let ones = bf16_ones_words(inner);
    let ones_buf = scratch.alloc(ones.len() as u64)?;
    backend.write_buffer(ones_buf.id, 0, &ones)?;

    for layer in &handles.layers {
        let views = LayerViews::acquire(layer, residency, backend).await?;
        let b = views.bufs();
        let nxt = scratch.alloc(act_bytes)?;
        {
            let scope = scratch.batch();
            let cur_h = scope.import_copy(cur.as_buf_ref());
            let nxt_h = scope.import_copy(nxt.as_buf_ref());
            let freqs_h = scope.import_copy(freqs_buf.as_buf_ref());
            let mask_h = scope.import_copy(mask_buf.as_buf_ref());
            let ones_h = scope.import_copy(ones_buf.as_buf_ref());
            conn_block_forward(
                &scope, bp, pipelines, m, cur_h, nxt_h, freqs_h, mask_h, ones_h, &b,
            )?;
            scope.submit_void().await?;
        }
        drop(views);
        cur = nxt;
    }

    // norm_out (weightless).
    let normed = scratch.alloc(act_bytes)?;
    {
        let scope = scratch.batch();
        let cur_h = ActBuf::dense(scope.import_copy(cur.as_buf_ref()));
        let ones_h = scope.import_copy(ones_buf.as_buf_ref());
        let out_h = ActBuf::dense(scope.import_copy(normed.as_buf_ref()));
        op_rmsnorm(&scope, bp, cur_h, ones_h, out_h, rows, dimu, dit::NORM_EPS)?;
        scope.submit_void().await?;
    }
    let bytes = backend.read_buffer(normed.id(), 0, act_bytes).await?;
    Ok(seq::act_readback_to_f32(
        bp.act_dtype,
        &bytes,
        CONN_SEQ * inner,
    ))
}

/// One connector block (`_BasicTransformerBlock1D`): weightless pre-norm ->
/// gated self-attn (full-inner qk-norm, per-head split rope, non-causal sdpa,
/// per-head sigmoid gate) -> residual -> weightless pre-norm -> gelu-tanh FF ->
/// residual. `cur`/`nxt` are dense `[CONN_SEQ, inner]`; `b` holds this block's
/// weight bufs. The `'wsp` lifetime ties every transient to `scope`.
#[allow(clippy::too_many_arguments)]
fn conn_block_forward<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    pipelines: &ConnectorPipelines,
    m: Modality,
    cur: BatchBuf<'wsp>,
    nxt: BatchBuf<'wsp>,
    freqs: BatchBuf<'wsp>,
    mask: BatchBuf<'wsp>,
    ones: BatchBuf<'wsp>,
    b: &ConnBufs,
) -> Result<(), WgpuError> {
    let rows = CONN_SEQ as u32;
    let dimu = m.inner_dim as u32;
    let hq = m.n_heads as u32;
    let hd = m.head_dim as u32;
    let ff_hidden = (4 * m.inner_dim) as u32;
    let scale = 1.0 / (m.head_dim as f32).sqrt();
    let cur_h = ActBuf::dense(cur);

    // matmul site helper (Q8_0 dense path: i8 pipes None under dense_acts).
    let mm = |src: ActBuf<'wsp>,
              w: BufRef,
              out_w: u32,
              in_w: u32,
              dq: Option<&DequantStep>,
              mp: &WgpuPipeline,
              mop: &MatMulF32|
     -> Result<BatchBuf<'wsp>, WgpuError> {
        let scratch_b = alloc_matmul_out_buf(scope, bp, rows * out_w)?;
        let dims = scope.u32x4_uniform(rows, out_w, in_w, 0)?;
        let w_h = scope.import_copy(w);
        Block::dispatch_matmul_site(
            scope, bp, src, w_h, scratch_b, dims, None, None, dq, mp, mop, rows, out_w, in_w,
        )?;
        Ok(scratch_b)
    };
    let add_bias =
        |pre: BatchBuf<'wsp>, bias: BufRef, out_w: u32| -> Result<BatchBuf<'wsp>, WgpuError> {
            let dst = alloc_act(scope, bp, rows, out_w)?;
            let bu = bcast_add_uniform(scope, out_w)?;
            let bv = scope.import_copy(bias);
            scope.bcast_add::<BcastAddF32>(&bp.bcast_add, pre, bv, bu, dst.data, rows * out_w)?;
            Ok(dst.data)
        };

    // --- attn ---
    let n1 = alloc_act(scope, bp, rows, dimu)?;
    op_rmsnorm(scope, bp, cur_h, ones, n1, rows, dimu, dit::NORM_EPS)?;

    let q = mm(
        n1,
        b.q_w,
        dimu,
        dimu,
        bp.dequant_qkv.as_ref(),
        &bp.matmul_qkv,
        &bp.matmuls.qkv,
    )?;
    let q = add_bias(q, b.q_b, dimu)?;
    let k = mm(
        n1,
        b.k_w,
        dimu,
        dimu,
        bp.dequant_qkv.as_ref(),
        &bp.matmul_qkv,
        &bp.matmuls.qkv,
    )?;
    let k = add_bias(k, b.k_b, dimu)?;
    let v = mm(
        n1,
        b.v_w,
        dimu,
        dimu,
        bp.dequant_qkv.as_ref(),
        &bp.matmul_qkv,
        &bp.matmuls.qkv,
    )?;
    let v = add_bias(v, b.v_b, dimu)?;

    // q/k RMSNorm over the FULL inner_dim (with weight), then per-head rope.
    let qn = alloc_act(scope, bp, rows, dimu)?;
    let qnw = scope.import_copy(b.q_norm);
    op_rmsnorm(
        scope,
        bp,
        ActBuf::dense(q),
        qnw,
        qn,
        rows,
        dimu,
        dit::NORM_EPS,
    )?;
    let kn = alloc_act(scope, bp, rows, dimu)?;
    let knw = scope.import_copy(b.k_norm);
    op_rmsnorm(
        scope,
        bp,
        ActBuf::dense(k),
        knw,
        kn,
        rows,
        dimu,
        dit::NORM_EPS,
    )?;

    // rope: collapse heads into rows so the kernel reads per-(pos,head) freqs.
    let qr = alloc_act(scope, bp, rows, dimu)?;
    op_rope_halfrot(scope, bp, qn, freqs, qr, rows * hq, 1, hd)?;
    let kr = alloc_act(scope, bp, rows, dimu)?;
    op_rope_halfrot(scope, bp, kn, freqs, kr, rows * hq, 1, hd)?;

    let sa = alloc_act(scope, bp, rows, dimu)?;
    op_sdpa(
        scope,
        bp,
        qr,
        kr,
        ActBuf::dense(v),
        mask,
        sa,
        1,
        rows,
        rows,
        hq,
        hq,
        hd,
        scale,
        0,
    )?;

    // per-head gate: attn * 2*sigmoid(to_gate_logits(n1)).
    let gate_pre = mm(
        n1,
        b.gate_w,
        hq,
        dimu,
        bp.dequant_qkv.as_ref(),
        &bp.matmul_qkv,
        &bp.matmuls.qkv,
    )?;
    let gate = add_bias(gate_pre, b.gate_b, hq)?;
    let gated = alloc_act(scope, bp, rows, dimu)?;
    scope.dispatch_op::<GatedHeadMulF32>(&pipelines.gate, &[sa.data, gate], gated.data)?;

    let proj = mm(
        gated,
        b.o_w,
        dimu,
        dimu,
        bp.dequant_proj.as_ref(),
        &bp.matmul_proj,
        &bp.matmuls.proj,
    )?;
    let proj = add_bias(proj, b.o_b, dimu)?;
    let after_attn = alloc_act(scope, bp, rows, dimu)?;
    op_add(scope, bp, cur_h, ActBuf::dense(proj), after_attn)?;

    // --- ff: net0.proj -> gelu-tanh -> net2 ---
    let n2 = alloc_act(scope, bp, rows, dimu)?;
    op_rmsnorm(scope, bp, after_attn, ones, n2, rows, dimu, dit::NORM_EPS)?;
    let h1 = mm(
        n2,
        b.ff0_w,
        ff_hidden,
        dimu,
        bp.dequant_ffn_up.as_ref(),
        &bp.matmul_ffn_up,
        &bp.matmuls.ffn_up,
    )?;
    let h1 = add_bias(h1, b.ff0_b, ff_hidden)?;
    let g = alloc_act(scope, bp, rows, ff_hidden)?;
    scope.dispatch_op::<GeluF32>(&pipelines.gelu, &[h1], g.data)?;
    let h2 = mm(
        g,
        b.ff2_w,
        dimu,
        ff_hidden,
        bp.dequant_ffn_down.as_ref(),
        &bp.matmul_ffn_down,
        &bp.matmuls.ffn_down,
    )?;
    let h2 = add_bias(h2, b.ff2_b, dimu)?;
    op_add(scope, bp, after_attn, ActBuf::dense(h2), ActBuf::dense(nxt))?;
    Ok(())
}

struct LayerViews<'a> {
    q_w: GpuView<'a>,
    q_b: GpuView<'a>,
    k_w: GpuView<'a>,
    k_b: GpuView<'a>,
    v_w: GpuView<'a>,
    v_b: GpuView<'a>,
    o_w: GpuView<'a>,
    o_b: GpuView<'a>,
    q_norm: GpuView<'a>,
    k_norm: GpuView<'a>,
    gate_w: GpuView<'a>,
    gate_b: GpuView<'a>,
    ff0_w: GpuView<'a>,
    ff0_b: GpuView<'a>,
    ff2_w: GpuView<'a>,
    ff2_b: GpuView<'a>,
}

impl<'a> LayerViews<'a> {
    async fn acquire<S: WeightSource>(
        h: &BlockHandles,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<Self, ResidencyError<S::Error, WgpuError>> {
        Ok(Self {
            q_w: residency.acquire(h.q_w, backend).await?,
            q_b: residency.acquire(h.q_b, backend).await?,
            k_w: residency.acquire(h.k_w, backend).await?,
            k_b: residency.acquire(h.k_b, backend).await?,
            v_w: residency.acquire(h.v_w, backend).await?,
            v_b: residency.acquire(h.v_b, backend).await?,
            o_w: residency.acquire(h.o_w, backend).await?,
            o_b: residency.acquire(h.o_b, backend).await?,
            q_norm: residency.acquire(h.q_norm, backend).await?,
            k_norm: residency.acquire(h.k_norm, backend).await?,
            gate_w: residency.acquire(h.gate_w, backend).await?,
            gate_b: residency.acquire(h.gate_b, backend).await?,
            ff0_w: residency.acquire(h.ff0_w, backend).await?,
            ff0_b: residency.acquire(h.ff0_b, backend).await?,
            ff2_w: residency.acquire(h.ff2_w, backend).await?,
            ff2_b: residency.acquire(h.ff2_b, backend).await?,
        })
    }
    fn bufs(&self) -> ConnBufs {
        ConnBufs {
            q_w: self.q_w.buf(),
            q_b: self.q_b.buf(),
            k_w: self.k_w.buf(),
            k_b: self.k_b.buf(),
            v_w: self.v_w.buf(),
            v_b: self.v_b.buf(),
            o_w: self.o_w.buf(),
            o_b: self.o_b.buf(),
            q_norm: self.q_norm.buf(),
            k_norm: self.k_norm.buf(),
            gate_w: self.gate_w.buf(),
            gate_b: self.gate_b.buf(),
            ff0_w: self.ff0_w.buf(),
            ff0_b: self.ff0_b.buf(),
            ff2_w: self.ff2_w.buf(),
            ff2_b: self.ff2_b.buf(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fe_flatten_layout_and_rms() {
        // 2 tokens, fake states where layer l is constant value (l+1) per token.
        let seq = 2;
        let d = gemma::HIDDEN;
        let states: Vec<Vec<f32>> = (0..N_STATES)
            .map(|l| vec![(l + 1) as f32; seq * d])
            .collect();
        let flat = super::feature_extractor_v2_flatten(&states, seq);
        assert_eq!(flat.len(), seq * FE_FLAT);
        // A constant row RMS-normalizes to ~1 (mean_sq = c^2 -> v/sqrt(c^2) = 1).
        // flat[t=0][d=0, l=5] -> index 5.
        let v = flat[5];
        assert!((v - 1.0).abs() < 1e-3, "rms-normed constant -> 1, got {v}");
    }

    #[test]
    fn freqs_endpoints() {
        // pos at the middle of max_pos -> frac 0 -> all cos=1, sin=0.
        let mid = dit::CONNECTOR_MAX_POS / 2;
        let f = build_connector_freqs(VIDEO, mid + 1);
        let base = (mid * VIDEO.n_heads) * VIDEO.head_dim;
        assert!((f[base] - 1.0).abs() < 1e-5);
        assert!(f[base + 1].abs() < 1e-5);
    }

    #[test]
    fn registers_fill_pads() {
        let inner = 4;
        let n_real = 1;
        let embeds = vec![9.0f32; n_real * inner];
        let registers: Vec<f32> = (0..dit::CONNECTOR_NUM_LEARNABLE_REGISTERS * inner)
            .map(|i| i as f32)
            .collect();
        let framed = frame_with_registers(&embeds, n_real, inner, &registers);
        assert_eq!(&framed[0..inner], &[9.0; 4]);
        // row 1 -> register 1 -> [4,5,6,7]
        assert_eq!(&framed[inner..2 * inner], &[4.0, 5.0, 6.0, 7.0]);
    }
}
