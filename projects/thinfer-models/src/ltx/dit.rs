//! LTX-2.3 dual-stream audio-video DiT (`AVTransformer3DModel`), one block.
//!
//! Ground truth: `third-party/LTX-2/.../model/transformer/` (`transformer.py`
//! `BasicAVTransformerBlock`, `attention.py`, `ops.py`, `rope.py`, `model.py`).
//! The GGUF tensor names are diffusers/ltx-core-native (`transformer_blocks.N.*`),
//! verified on disk; no comfy rename.
//!
//! Each block runs FIVE attention sublayers in this order (`transformer.py:253`):
//! (1) video self-attn (`attn1`, q=k=v=video); (2) video<->text cross (`attn2`,
//! q=video, kv=video text); (3) audio self-attn (`audio_attn1`); (4) audio<->text
//! cross (`audio_attn2`); (5) audio<->video cross (`audio_to_video_attn` q=video/
//! kv=audio + `video_to_audio_attn` q=audio/kv=video), then per-modality gelu-tanh
//! FFNs. Every attention has per-head sigmoid gating (`apply_gated_attention`,
//! `to_gate_logits`) AND an outer AdaLN gate at its residual. The norms are
//! WEIGHTLESS rms_norm (`ada_zero = rms_norm(x)*(1+scale)+shift`); qk-norm is a
//! learned rms_norm over the FULL inner dim.
//!
//! RoPE is SPLIT / half-rot (Qwen3-style `(k, k+D/2)` pairing, reuse
//! `op_rope_halfrot`) with PER-HEAD-DISTINCT freqs: the freq grid spans the full
//! inner dim, head h owns slice `[h*pairs, (h+1)*pairs)`. As in the connector we
//! collapse heads into the kernel's row axis (`rows = S*heads, heads = 1`) so each
//! (pos, head) reads its own freq row. Self/audio self use the 3-axis (video) /
//! 1-axis (audio) `pe`; av-cross uses temporal-only `cross_pe` with SEPARATE q and
//! k freqs. Text cross-attn gets NO rope.
//!
//! Precision: F32 acts, Q8_0 matmuls, bf16 norm/bias/table weights (same regime
//! as the connector; per-head gate + gelu A-sides carry outliers so no i8 here in
//! the parity path). The base model predicts VELOCITY; X0 = latent - v*timestep is
//! applied by the sampler, not here.

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError, WgpuPipeline};
use thinfer_core::ops::{
    ActDtype, BcastAddF32, BcastFmaF32, BcastModulateF32, GatedHeadMulF32, GeluF32, Op,
    WeightDtype, WgslConfig,
};
use thinfer_core::quant::QuantKind;
use thinfer_core::residency::{
    GpuView, ResidencyError, TransposePolicy, WeightHandle, WeightResidency,
};
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace};

use super::config as dit;
use crate::common::block::{
    ActBuf, Block, BlockPipelines, BlockWgslConfigs, DenseActSites, DequantStep, alloc_act,
    alloc_matmul_out_buf, op_rmsnorm, op_rope_halfrot, op_sdpa,
};
use crate::common::embedders::bcast_add_uniform;
use crate::z_image::text_encoder::{LoadError, register_one};

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// One attention sublayer's projection geometry. `inner = heads * head_dim` is
/// the q/k/v width and the qk-norm width; `q_dim`/`kv_dim` are the (possibly
/// asymmetric) input widths; `out_dim = q_dim` (`to_out` projects back).
#[derive(Clone, Copy, Debug)]
pub struct AttnGeom {
    pub q_dim: usize,
    pub kv_dim: usize,
    pub heads: usize,
    pub head_dim: usize,
}

impl AttnGeom {
    pub fn inner(&self) -> usize {
        self.heads * self.head_dim
    }
    fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

/// The six per-block attention geometries (video/audio streams + the av-cross
/// pair). Heads/head_dim for the av-cross both come from the AUDIO stream
/// (`transformer.py:153,165`); only the q/kv input widths differ by direction.
pub mod geom {
    use super::AttnGeom;
    use super::dit;

    pub const VIDEO_SELF: AttnGeom = AttnGeom {
        q_dim: dit::DIM,
        kv_dim: dit::DIM,
        heads: dit::N_HEADS,
        head_dim: dit::HEAD_DIM,
    };
    pub const VIDEO_CROSS: AttnGeom = AttnGeom {
        q_dim: dit::DIM,
        kv_dim: dit::CROSS_ATTENTION_DIM,
        heads: dit::N_HEADS,
        head_dim: dit::HEAD_DIM,
    };
    pub const AUDIO_SELF: AttnGeom = AttnGeom {
        q_dim: dit::AUDIO_DIM,
        kv_dim: dit::AUDIO_DIM,
        heads: dit::AUDIO_N_HEADS,
        head_dim: dit::AUDIO_HEAD_DIM,
    };
    pub const AUDIO_CROSS: AttnGeom = AttnGeom {
        q_dim: dit::AUDIO_DIM,
        kv_dim: dit::AUDIO_CROSS_ATTENTION_DIM,
        heads: dit::AUDIO_N_HEADS,
        head_dim: dit::AUDIO_HEAD_DIM,
    };
    /// audio_to_video_attn: Q from video, K/V from audio, out back to video.
    pub const A2V: AttnGeom = AttnGeom {
        q_dim: dit::DIM,
        kv_dim: dit::AUDIO_DIM,
        heads: dit::AUDIO_N_HEADS,
        head_dim: dit::AUDIO_HEAD_DIM,
    };
    /// video_to_audio_attn: Q from audio, K/V from video, out back to audio.
    pub const V2A: AttnGeom = AttnGeom {
        q_dim: dit::AUDIO_DIM,
        kv_dim: dit::DIM,
        heads: dit::AUDIO_N_HEADS,
        head_dim: dit::AUDIO_HEAD_DIM,
    };
}

// ---------------------------------------------------------------------------
// RoPE freqs (SPLIT / half-rot, per-head-distinct, float64 grid)
// ---------------------------------------------------------------------------

/// Build the per-head SPLIT-rope freq table `[seq*heads, head_dim]` (interleaved
/// `(cos, sin)` per pair) from physical-coord position bounds, mirroring
/// `rope.py` `precompute_freqs_cis` (`generate_freq_grid_np` -> `generate_freqs`
/// -> `split_freqs_cis`) for `rope_type=split`, `frequencies_precision=float64`.
///
/// - `positions`: `[n_dims, seq, 2]` start/end grid bounds (already in physical
///   pixel/second units). Middle index = `(start+end)/2`.
/// - `max_pos`: per-axis fractional-position divisor (`[20,2048,2048]` video,
///   `[20]` audio / cross).
/// - `inner_dim`: the freq grid spans the full inner dim; head h owns slice
///   `[h*pairs, (h+1)*pairs)` of the per-token `inner/2` angle vector.
///
/// `indices[k] = theta^(k/(count-1)) * pi/2`, `count = inner/(2*n_dims)`, in f64.
/// `angle = indices[k] * (2*frac[d] - 1)`; the per-token angle vector is
/// `[<pad zeros>, (k,d)-flattened]` (front pad so the head-0 low pairs are
/// identity, matching the `split_freqs_cis` front pad).
pub fn build_split_freqs(
    positions: &[f32],
    n_dims: usize,
    seq: usize,
    max_pos: &[f64],
    inner_dim: usize,
    n_heads: usize,
    head_dim: usize,
    theta: f64,
) -> Vec<f32> {
    assert_eq!(positions.len(), n_dims * seq * 2, "positions shape");
    assert_eq!(max_pos.len(), n_dims, "max_pos per axis");
    let pairs = head_dim / 2;
    let half = inner_dim / 2;
    let count = inner_dim / (2 * n_dims);
    // f64 frequency grid: theta^linspace(0,1,count) * pi/2.
    let indices: Vec<f64> = (0..count)
        .map(|k| {
            let e = if count == 1 {
                0.0
            } else {
                k as f64 / (count - 1) as f64
            };
            theta.powf(e) * std::f64::consts::PI / 2.0
        })
        .collect();
    let raw_len = count * n_dims;
    let pad = half - raw_len;
    let mut out = vec![0.0f32; seq * n_heads * head_dim];
    for p in 0..seq {
        // fractional positions per axis (middle index / max_pos).
        let mut frac = vec![0.0f64; n_dims];
        for (d, fr) in frac.iter_mut().enumerate() {
            let base = (d * seq + p) * 2;
            let mid = (positions[base] as f64 + positions[base + 1] as f64) / 2.0;
            *fr = mid / max_pos[d];
        }
        // per-token angle vector [half]: front pad zeros, then (k,d) flattened.
        let mut angle = vec![0.0f64; half];
        for k in 0..count {
            for (d, &fr) in frac.iter().enumerate() {
                angle[pad + k * n_dims + d] = indices[k] * (2.0 * fr - 1.0);
            }
        }
        for h in 0..n_heads {
            for j in 0..pairs {
                let a = angle[h * pairs + j];
                let base = (p * n_heads + h) * head_dim + 2 * j;
                out[base] = a.cos() as f32;
                out[base + 1] = a.sin() as f32;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Per-block weights
// ---------------------------------------------------------------------------

struct AttnWeightIds {
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
}

impl AttnWeightIds {
    fn new(block_prefix: &str, attn: &str) -> Self {
        let id = |s: &str| WeightId(format!("{block_prefix}.{attn}.{s}"));
        Self {
            q_w: id("to_q.weight"),
            q_b: id("to_q.bias"),
            k_w: id("to_k.weight"),
            k_b: id("to_k.bias"),
            v_w: id("to_v.weight"),
            v_b: id("to_v.bias"),
            o_w: id("to_out.0.weight"),
            o_b: id("to_out.0.bias"),
            q_norm: id("q_norm.weight"),
            k_norm: id("k_norm.weight"),
            gate_w: id("to_gate_logits.weight"),
            gate_b: id("to_gate_logits.bias"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AttnHandles {
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
}

/// FFN weights for one modality (`ff.net.0.proj` -> gelu-tanh -> `ff.net.2`).
#[derive(Clone, Copy, Debug)]
pub struct FfHandles {
    up_w: WeightHandle,
    up_b: WeightHandle,
    down_w: WeightHandle,
    down_b: WeightHandle,
}

/// AdaLN modulation tables for one block (loaded for the timestep/adaln path).
#[derive(Clone, Copy, Debug)]
pub struct TableHandles {
    pub scale_shift: WeightHandle,
    pub audio_scale_shift: WeightHandle,
    pub prompt_scale_shift: WeightHandle,
    pub audio_prompt_scale_shift: WeightHandle,
    pub a2v_ca_video: WeightHandle,
    pub a2v_ca_audio: WeightHandle,
}

#[derive(Clone, Copy, Debug)]
pub struct BlockHandles {
    pub attn1: AttnHandles,
    pub attn2: AttnHandles,
    pub audio_attn1: AttnHandles,
    pub audio_attn2: AttnHandles,
    pub a2v: AttnHandles,
    pub v2a: AttnHandles,
    pub ff: FfHandles,
    pub audio_ff: FfHandles,
    pub tables: TableHandles,
}

fn register_attn<S: WeightSource>(
    residency: &WeightResidency<S>,
    block_prefix: &str,
    attn: &str,
) -> Result<AttnHandles, LoadError> {
    let ids = AttnWeightIds::new(block_prefix, attn);
    let q8 = Some(QuantKind::Q8_0);
    let lin = |id: &WeightId| register_one(residency, id, TransposePolicy::Linear2D, q8);
    let dense = |id: &WeightId| register_one(residency, id, TransposePolicy::None, None);
    Ok(AttnHandles {
        q_w: lin(&ids.q_w)?,
        q_b: dense(&ids.q_b)?,
        k_w: lin(&ids.k_w)?,
        k_b: dense(&ids.k_b)?,
        v_w: lin(&ids.v_w)?,
        v_b: dense(&ids.v_b)?,
        o_w: lin(&ids.o_w)?,
        o_b: dense(&ids.o_b)?,
        q_norm: dense(&ids.q_norm)?,
        k_norm: dense(&ids.k_norm)?,
        gate_w: lin(&ids.gate_w)?,
        gate_b: dense(&ids.gate_b)?,
    })
}

fn register_ff<S: WeightSource>(
    residency: &WeightResidency<S>,
    block_prefix: &str,
    ff: &str,
) -> Result<FfHandles, LoadError> {
    let q8 = Some(QuantKind::Q8_0);
    let lin = |s: &str| {
        register_one(
            residency,
            &WeightId(format!("{block_prefix}.{ff}.{s}")),
            TransposePolicy::Linear2D,
            q8,
        )
    };
    let dense = |s: &str| {
        register_one(
            residency,
            &WeightId(format!("{block_prefix}.{ff}.{s}")),
            TransposePolicy::None,
            None,
        )
    };
    Ok(FfHandles {
        up_w: lin("net.0.proj.weight")?,
        up_b: dense("net.0.proj.bias")?,
        down_w: lin("net.2.weight")?,
        down_b: dense("net.2.bias")?,
    })
}

/// Register one DiT block's weights from the GGUF (`transformer_blocks.{i}.*`).
pub fn register_block<S: WeightSource>(
    residency: &WeightResidency<S>,
    i: usize,
) -> Result<BlockHandles, LoadError> {
    let p = format!("transformer_blocks.{i}");
    let dense = |s: &str| {
        register_one(
            residency,
            &WeightId(format!("{p}.{s}")),
            TransposePolicy::None,
            None,
        )
    };
    Ok(BlockHandles {
        attn1: register_attn(residency, &p, "attn1")?,
        attn2: register_attn(residency, &p, "attn2")?,
        audio_attn1: register_attn(residency, &p, "audio_attn1")?,
        audio_attn2: register_attn(residency, &p, "audio_attn2")?,
        a2v: register_attn(residency, &p, "audio_to_video_attn")?,
        v2a: register_attn(residency, &p, "video_to_audio_attn")?,
        ff: register_ff(residency, &p, "ff")?,
        audio_ff: register_ff(residency, &p, "audio_ff")?,
        tables: TableHandles {
            scale_shift: dense("scale_shift_table")?,
            audio_scale_shift: dense("audio_scale_shift_table")?,
            prompt_scale_shift: dense("prompt_scale_shift_table")?,
            audio_prompt_scale_shift: dense("audio_prompt_scale_shift_table")?,
            a2v_ca_video: dense("scale_shift_table_a2v_ca_video")?,
            a2v_ca_audio: dense("scale_shift_table_a2v_ca_audio")?,
        },
    })
}

/// Top-level I/O weights: `patchify_proj` (latent channels -> inner) and the
/// output stage (`scale_shift_table[2]` + `norm_out` LayerNorm-no-affine +
/// `proj_out`), per modality. Matmul weights BF16; tables/biases F32.
#[derive(Clone, Copy, Debug)]
pub struct IoHandles {
    patchify_v_w: WeightHandle,
    patchify_v_b: WeightHandle,
    patchify_a_w: WeightHandle,
    patchify_a_b: WeightHandle,
    /// Output `scale_shift_table` `[2, DIM]` (added to embedded_timestep).
    sst_out_v: WeightHandle,
    sst_out_a: WeightHandle,
    proj_out_v_w: WeightHandle,
    proj_out_v_b: WeightHandle,
    proj_out_a_w: WeightHandle,
    proj_out_a_b: WeightHandle,
}

/// Register the top-level patchify + output weights.
pub fn register_io<S: WeightSource>(
    residency: &WeightResidency<S>,
) -> Result<IoHandles, LoadError> {
    let lin = |s: &str| {
        register_one(
            residency,
            &WeightId(s.into()),
            TransposePolicy::Linear2D,
            None,
        )
    };
    let den = |s: &str| register_one(residency, &WeightId(s.into()), TransposePolicy::None, None);
    Ok(IoHandles {
        patchify_v_w: lin("patchify_proj.weight")?,
        patchify_v_b: den("patchify_proj.bias")?,
        patchify_a_w: lin("audio_patchify_proj.weight")?,
        patchify_a_b: den("audio_patchify_proj.bias")?,
        sst_out_v: den("scale_shift_table")?,
        sst_out_a: den("audio_scale_shift_table")?,
        proj_out_v_w: lin("proj_out.weight")?,
        proj_out_v_b: den("proj_out.bias")?,
        proj_out_a_w: lin("audio_proj_out.weight")?,
        proj_out_a_b: den("audio_proj_out.bias")?,
    })
}

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

/// Resolve the DiT activation dtype + DP4A choice from `LTX_DIT_ACTS`.
///
/// The parity path is `f32` acts, Q8_0 dequant-once matmuls, dense SDPA -- but
/// that path runs the dense `SdpaF32` (16 ms/disp) + dense f32 matmul, which the
/// trace shows are ~81% of DiT GPU time. The perf paths unlock the proven levers:
/// - `f16`   : F16 acts -> the fast `sdpa_sg` subgroup SDPA + f16 matmul.
/// - `f16dp4a`: F16 acts + DP4A i8 on the VIDEO-stream matmuls -- video self +
///   text-cross qkv, the a2v cross (writes video), and the video ffn_up (all
///   A-sides normed). The AUDIO stream stays fully dense: i8 on the small audio
///   dim is too lossy at the block-residual level (audio block rel 0.86% > the
///   0.5% gate; fully-dense audio = 0.32%). proj / ffn_down / gate dense too.
///
/// f16 acts put the residual stream in f16 (5-bit exponent, max 65504); the
/// worklog flags this as a dead-end for the big WAN/Qwen DiTs (residual outliers
/// overflow). LTX-2.3 distilled does NOT overflow -- validated against the
/// upstream pyref (`dit_full_parity`: f16 slope 1.0006, rel 0.093%; bit-clean to
/// the f32 path). **`f16dp4a` is the DEFAULT** (env unset): F16 acts + DP4A i8 on
/// the video-stream qkv + ffn_up, ~1.6x at real res over plain f16 and validated
/// quality-clean (block rel <=0.37%, eyeballed on a real browser gen). `f16` opts
/// out of the i8 (dense f16 matmuls, the bit-cleaner perf path); `f32` is the
/// bit-exact parity reference. Without shader-f16 the default falls back to f32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DitActs {
    F32,
    F16,
    F16Dp4a,
}

impl DitActs {
    fn from_env(backend: &WgpuBackend) -> Self {
        let want = std::env::var("LTX_DIT_ACTS").unwrap_or_default();
        let f16_ok = backend.supports_shader_f16();
        match want.as_str() {
            "f32" => Self::F32,
            // Explicit dense f16 (opt OUT of DP4A i8 on the video-stream matmuls).
            "f16" if f16_ok => Self::F16,
            // Default (env unset) + "f16dp4a"/"dp4a": DP4A is the default perf path.
            "" | "f16dp4a" | "dp4a" if f16_ok => Self::F16Dp4a,
            _ => {
                if !want.is_empty() {
                    tracing::warn!(
                        "LTX_DIT_ACTS={want}: shader_f16 unsupported or unknown value; using f32"
                    );
                }
                Self::F32
            }
        }
    }
    fn act_dtype(self) -> ActDtype {
        match self {
            Self::F32 => ActDtype::F32,
            Self::F16 | Self::F16Dp4a => ActDtype::F16,
        }
    }
}

/// DiT block config. Q8_0 matmuls, bf16 norm/bias weights. `act` selects f32
/// (parity) or f16 (fast `sdpa_sg` + f16 matmul); `dp4a` opts the outlier-free
/// self-attn qkv + ffn_up sites out of `dense_acts` (DP4A i8). `head_dim` 128
/// (video) / 64 (audio) both fit `SdpaF32`/`sdpa_sg`, so `large_d_sdpa` stays off.
fn dit_block_cfgs(act: ActDtype, dp4a: bool) -> BlockWgslConfigs {
    let ops = WgslConfig {
        bf16_quant_writes: false,
        act_dtype: act,
        weight_dtype: WeightDtype::Bf16,
    };
    let mm = WgslConfig {
        weight_dtype: WeightDtype::Quant(QuantKind::Q8_0),
        ..ops
    };
    // `dp4a` BUILDS the i8 pipelines for the qkv + ffn_up sites; the per-call
    // routing (`I8Site` in `biased_proj`, driven by `attention`/`feed_forward`)
    // decides which projections actually use them -- the VIDEO-stream qkv/ffn_up
    // (all normed A-sides), with proj (post-gate) + ffn_down (post-gelu) + the
    // whole AUDIO stream kept dense.
    let dense_acts = DenseActSites {
        qkv: !dp4a,
        proj: true,
        ffn_up: !dp4a,
        ffn_down: true,
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
        dense_acts,
        large_d_sdpa: false,
    }
}

pub struct DitPipelines {
    pub block: BlockPipelines,
    gelu: WgpuPipeline,
    gate: WgpuPipeline,
    /// Silu for the timestep embedder activations.
    pub(crate) silu: WgpuPipeline,
    /// Dense (dequant-once) matmul dequant step per quant kind. The Q8_0 DiT
    /// file is uniform, but the Q4_K_M variant ships each block MIXED (q/k/o/gate
    /// Q4_K, to_v + video ffn_down Q6_K). The block matmul reads the f16/bf16
    /// dequant workspace regardless of source kind, so only the dequant step
    /// varies; this map lets `biased_proj` pick the step matching each weight's
    /// on-disk kind. Built for every K-quant kind the GGUF recipes use (the
    /// pipeline cache dedups, so the unused kinds cost ~nothing). DP4A i8 stays
    /// Q8_0-only (its i8 dequant is scheme-specific); the Q4 variant runs dense.
    dense_dequant: std::collections::HashMap<QuantKind, DequantStep>,
}

impl DitPipelines {
    /// Dense dequant step for `kind`, or `None` if not built (non-quant weight).
    fn dense_dequant(&self, kind: QuantKind) -> Option<&DequantStep> {
        self.dense_dequant.get(&kind)
    }

    pub async fn compile(backend: &WgpuBackend) -> Result<Self, WgpuError> {
        let acts = DitActs::from_env(backend);
        tracing::info!(?acts, "ltx dit acts");
        let cfgs = dit_block_cfgs(acts.act_dtype(), acts == DitActs::F16Dp4a);
        let block = BlockPipelines::compile(backend, &cfgs).await?;
        // Dense dequant steps keyed by quant kind. Target matches the block's
        // act dtype (F16 acts -> f16 workspace, else bf16), exactly as
        // `BlockPipelines::compile` chooses for its own per-site steps.
        let dequant_target = if acts.act_dtype() == ActDtype::F16 {
            thinfer_core::ops::dequant::DequantTarget::F16
        } else {
            thinfer_core::ops::dequant::DequantTarget::Bf16
        };
        let dq_layout = thinfer_core::ops::dequant::layout();
        let mut dense_dequant = std::collections::HashMap::new();
        for scheme in [
            QuantKind::Q8_0,
            QuantKind::Q4_K,
            QuantKind::Q5_K,
            QuantKind::Q6_K,
        ] {
            let wgsl = thinfer_core::ops::dequant::build_wgsl(scheme, dequant_target);
            let pipeline = backend
                .create_pipeline(
                    &format!("ltx_dit_dequant_{}", scheme.hint()),
                    &wgsl,
                    "main",
                    dq_layout,
                )
                .await?;
            dense_dequant.insert(scheme, DequantStep { pipeline, scheme });
        }
        let gelu = backend
            .create_pipeline(
                "ltx_dit_gelu",
                <GeluF32 as Op>::wgsl(&cfgs.ops),
                "main",
                <GeluF32 as Op>::layout(),
            )
            .await?;
        let gate = backend
            .create_pipeline(
                "ltx_dit_gate",
                <GatedHeadMulF32 as Op>::wgsl(&cfgs.ops),
                "main",
                <GatedHeadMulF32 as Op>::layout(),
            )
            .await?;
        let silu = backend
            .create_pipeline(
                "ltx_dit_silu",
                <thinfer_core::ops::SiluF32 as Op>::wgsl(&cfgs.ops),
                "main",
                <thinfer_core::ops::SiluF32 as Op>::layout(),
            )
            .await?;
        Ok(Self {
            block,
            gelu,
            gate,
            silu,
            dense_dequant,
        })
    }
}

/// On-disk quant kind of each matmul weight in a DiT block. Uniform across the
/// 48 blocks (probed once from block 0) but MIXED within a block for the Q4_K_M
/// variant. The Q8_0 baseline reports `Q8_0` for every site.
#[derive(Clone, Copy, Debug)]
pub struct BlockQuantKinds {
    q: QuantKind,
    k: QuantKind,
    v: QuantKind,
    o: QuantKind,
    gate: QuantKind,
    vff_up: QuantKind,
    vff_down: QuantKind,
    aff_up: QuantKind,
    aff_down: QuantKind,
}

impl BlockQuantKinds {
    /// Probe each matmul weight's on-disk quant kind from block 0 of the source
    /// catalog. The Q8_0 file reports Q8_0 everywhere (behaviour unchanged); the
    /// Q4_K_M variant reports the mixed per-site kinds. Non-quant sites (none in
    /// practice for these matmul weights) fall back to Q8_0.
    fn probe<S: WeightSource>(residency: &WeightResidency<S>) -> Self {
        let probe = |suffix: &str| -> QuantKind {
            let id = WeightId(format!("transformer_blocks.0.{suffix}"));
            match residency
                .source()
                .catalog()
                .get(&id)
                .and_then(|e| e.encoding)
            {
                Some(thinfer_core::tensor::StorageEncoding::Quant(k)) => k,
                _ => QuantKind::Q8_0,
            }
        };
        Self {
            q: probe("attn1.to_q.weight"),
            k: probe("attn1.to_k.weight"),
            v: probe("attn1.to_v.weight"),
            o: probe("attn1.to_out.0.weight"),
            gate: probe("attn1.to_gate_logits.weight"),
            vff_up: probe("ff.net.0.proj.weight"),
            vff_down: probe("ff.net.2.weight"),
            aff_up: probe("audio_ff.net.0.proj.weight"),
            aff_down: probe("audio_ff.net.2.weight"),
        }
    }
}

/// Resolved dense dequant steps for one attention sublayer's matmul weights.
#[derive(Clone, Copy)]
struct AttnDequant<'a> {
    q: Option<&'a DequantStep>,
    k: Option<&'a DequantStep>,
    v: Option<&'a DequantStep>,
    o: Option<&'a DequantStep>,
    gate: Option<&'a DequantStep>,
}

/// Resolved dense dequant steps for one FFN's matmul weights.
#[derive(Clone, Copy)]
struct FfDequant<'a> {
    up: Option<&'a DequantStep>,
    down: Option<&'a DequantStep>,
}

// ---------------------------------------------------------------------------
// Modulation inputs (per-stream AdaLN vectors, [inner] each, dumped/computed)
// ---------------------------------------------------------------------------

/// Per-stream AdaLN modulation vectors for one block. Each is an `[inner]`
/// channel vector broadcast over tokens (uniform scalar-timestep path). Order
/// matches `get_ada_values` row layout; see `transformer.py`.
pub struct StreamMod<'wsp> {
    /// self-attn (slice 0:3 -> shift,scale,gate).
    pub msa_scale: BatchBuf<'wsp>,
    pub msa_shift: BatchBuf<'wsp>,
    pub msa_gate: BatchBuf<'wsp>,
    /// text cross-attn q-side (slice 6:9 -> shift_q,scale_q,gate).
    pub cq_scale: BatchBuf<'wsp>,
    pub cq_shift: BatchBuf<'wsp>,
    pub cq_gate: BatchBuf<'wsp>,
    /// text cross-attn kv-side (prompt table + prompt_timestep -> shift_kv,scale_kv).
    pub ckv_scale: BatchBuf<'wsp>,
    pub ckv_shift: BatchBuf<'wsp>,
    /// FFN (slice 3:6 -> shift,scale,gate).
    pub mlp_scale: BatchBuf<'wsp>,
    pub mlp_shift: BatchBuf<'wsp>,
    pub mlp_gate: BatchBuf<'wsp>,
    /// av-cross: this stream's scale/shift when it is the A2V participant
    /// (a2v slice 0:2 -> scale,shift) and the V2A participant (slice 2:4).
    pub a2v_scale: BatchBuf<'wsp>,
    pub a2v_shift: BatchBuf<'wsp>,
    pub v2a_scale: BatchBuf<'wsp>,
    pub v2a_shift: BatchBuf<'wsp>,
    /// av-cross gate (row 4): video stream carries gate_a2v, audio carries gate_v2a.
    pub av_gate: BatchBuf<'wsp>,
}

// ---------------------------------------------------------------------------
// Block forward
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum DitError<SE: core::fmt::Debug> {
    Wgpu(WgpuError),
    Residency(ResidencyError<SE, WgpuError>),
    Load(LoadError),
}
impl<SE: core::fmt::Debug> From<WgpuError> for DitError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}
impl<SE: core::fmt::Debug> From<LoadError> for DitError<SE> {
    fn from(e: LoadError) -> Self {
        Self::Load(e)
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for DitError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

/// bf16(1.0) packed two-per-word, enough words for the widest inner dim. Gain for
/// the weightless `ada_zero` rms_norms (read as packed bf16 by the F32-act path).
fn bf16_ones_words(inner: usize) -> Vec<u8> {
    let words = inner / 2;
    let mut v = Vec::with_capacity(words * 4);
    for _ in 0..words {
        v.extend_from_slice(&0x3F80_3F80u32.to_le_bytes());
    }
    v
}

// ---------------------------------------------------------------------------
// Local op helpers
// ---------------------------------------------------------------------------

/// `out = x * (1 + scale) + shift` (AdaLN affine; scale/shift are act vectors
/// broadcast over rows). One `bcast_modulate` dispatch (bias=1 folds the `1 +`).
#[allow(clippy::too_many_arguments)]
fn op_modulate<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: BatchBuf<'wsp>,
    scale: BatchBuf<'wsp>,
    shift: BatchBuf<'wsp>,
    dst: BatchBuf<'wsp>,
    rows: u32,
    dim: u32,
) -> Result<(), WgpuError> {
    let u = scope.u32x4_uniform(dim, 1.0_f32.to_bits(), 0, 0)?;
    scope.bcast_modulate::<BcastModulateF32>(
        &bp.bcast_modulate,
        x,
        scale,
        shift,
        u,
        dst,
        rows * dim,
    )
}

/// `out = x + gate * y` (gated residual; gate is an act vector broadcast over
/// rows). `out` must not alias `x` or `y`.
#[allow(clippy::too_many_arguments)]
fn op_gate_residual<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: BatchBuf<'wsp>,
    gate: BatchBuf<'wsp>,
    y: BatchBuf<'wsp>,
    dst: BatchBuf<'wsp>,
    rows: u32,
    dim: u32,
) -> Result<(), WgpuError> {
    let u = scope.u32x4_uniform(dim, 0, 0, 0)?;
    scope.bcast_fma::<BcastFmaF32>(&bp.bcast_fma, x, gate, y, u, dst, rows * dim)
}

/// Channel-broadcast bias add `out[i] = x[i] + bias[i % dim]`. `bias` is a
/// resident bf16 weight view (`BcastAddF32` reads the broadcast vector as a
/// weight).
fn op_bias_add<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: BatchBuf<'wsp>,
    bias: BatchBuf<'wsp>,
    dst: BatchBuf<'wsp>,
    rows: u32,
    dim: u32,
) -> Result<(), WgpuError> {
    let u = bcast_add_uniform(scope, dim)?;
    scope.bcast_add::<BcastAddF32>(&bp.bcast_add, x, bias, u, dst, rows * dim)
}

/// `ada_zero(x) = rms_norm_weightless(x) * (1 + scale) + shift`. `ones` is the
/// bf16-ones gain for the weightless rms_norm.
#[allow(clippy::too_many_arguments)]
fn ada_zero<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    ones: BatchBuf<'wsp>,
    x: ActBuf<'wsp>,
    scale: BatchBuf<'wsp>,
    shift: BatchBuf<'wsp>,
    rows: u32,
    dim: u32,
    eps: f32,
) -> Result<ActBuf<'wsp>, WgpuError> {
    let n = alloc_act(scope, bp, rows, dim)?;
    op_rmsnorm(scope, bp, x, ones, n, rows, dim, eps)?;
    let out = alloc_act(scope, bp, rows, dim)?;
    op_modulate(scope, bp, n.data, scale, shift, out.data, rows, dim)?;
    Ok(out)
}

/// `out = src @ w^T + bias` through the qkv matmul site (dense Q8_0). All DiT
/// projections share one site (uniform cfg); `n`=out width, `k`=in width.
///
/// `i8` selects the DP4A i8 matmul route for the outlier-safe sites (self-attn
/// qkv and ffn_up); the cross-attn / gate / proj / ffn_down projections pass
/// `I8Site::None`. On the F32/F16 (non-dp4a) paths the i8 pipelines are `None`,
/// so every site falls back to the dense dequant-once matmul automatically.
#[derive(Clone, Copy, PartialEq, Eq)]
enum I8Site {
    None,
    Qkv,
    FfnUp,
}

#[allow(clippy::too_many_arguments)]
fn biased_proj<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    src: ActBuf<'wsp>,
    w: BufRef,
    b: BufRef,
    rows: u32,
    n: u32,
    k: u32,
    i8: I8Site,
    dq_dense: Option<&DequantStep>,
) -> Result<ActBuf<'wsp>, WgpuError> {
    let pre = alloc_matmul_out_buf(scope, bp, rows * n)?;
    let dims = scope.u32x4_uniform(rows, n, k, 0)?;
    let w_h = scope.import_copy(w);
    let (mi8, di8) = match i8 {
        I8Site::None => (None, None),
        I8Site::Qkv => (bp.matmul_i8_qkv.as_ref(), bp.dequant_i8_qkv.as_ref()),
        I8Site::FfnUp => (bp.matmul_i8_ffn_up.as_ref(), bp.dequant_i8_ffn_up.as_ref()),
    };
    Block::dispatch_matmul_site(
        scope,
        bp,
        src,
        w_h,
        pre,
        dims,
        mi8,
        di8,
        dq_dense,
        &bp.matmul_qkv,
        &bp.matmuls.qkv,
        rows,
        n,
        k,
    )?;
    let bv = scope.import_copy(b);
    let out = alloc_act(scope, bp, rows, n)?;
    op_bias_add(scope, bp, pre, bv, out.data, rows, n)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Weight buffers / views
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct AttnBufs {
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
}

struct AttnViews<'a> {
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
}

impl<'a> AttnViews<'a> {
    async fn acquire<S: WeightSource>(
        h: &AttnHandles,
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
        })
    }
    fn bufs(&self) -> AttnBufs {
        AttnBufs {
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
        }
    }
}

#[derive(Clone, Copy)]
struct FfBufs {
    up_w: BufRef,
    up_b: BufRef,
    down_w: BufRef,
    down_b: BufRef,
}

struct FfViews<'a> {
    up_w: GpuView<'a>,
    up_b: GpuView<'a>,
    down_w: GpuView<'a>,
    down_b: GpuView<'a>,
}

impl<'a> FfViews<'a> {
    async fn acquire<S: WeightSource>(
        h: &FfHandles,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<Self, ResidencyError<S::Error, WgpuError>> {
        Ok(Self {
            up_w: residency.acquire(h.up_w, backend).await?,
            up_b: residency.acquire(h.up_b, backend).await?,
            down_w: residency.acquire(h.down_w, backend).await?,
            down_b: residency.acquire(h.down_b, backend).await?,
        })
    }
    fn bufs(&self) -> FfBufs {
        FfBufs {
            up_w: self.up_w.buf(),
            up_b: self.up_b.buf(),
            down_w: self.down_w.buf(),
            down_b: self.down_b.buf(),
        }
    }
}

struct BlockBufs {
    attn1: AttnBufs,
    attn2: AttnBufs,
    audio_attn1: AttnBufs,
    audio_attn2: AttnBufs,
    a2v: AttnBufs,
    v2a: AttnBufs,
    ff: FfBufs,
    audio_ff: FfBufs,
}

/// All resident views for one block (held while its scope runs).
struct BlockViewsAll<'a> {
    attn1: AttnViews<'a>,
    attn2: AttnViews<'a>,
    audio_attn1: AttnViews<'a>,
    audio_attn2: AttnViews<'a>,
    a2v: AttnViews<'a>,
    v2a: AttnViews<'a>,
    ff: FfViews<'a>,
    audio_ff: FfViews<'a>,
}

impl<'a> BlockViewsAll<'a> {
    async fn acquire<S: WeightSource>(
        h: &BlockHandles,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<Self, ResidencyError<S::Error, WgpuError>> {
        Ok(Self {
            attn1: AttnViews::acquire(&h.attn1, residency, backend).await?,
            attn2: AttnViews::acquire(&h.attn2, residency, backend).await?,
            audio_attn1: AttnViews::acquire(&h.audio_attn1, residency, backend).await?,
            audio_attn2: AttnViews::acquire(&h.audio_attn2, residency, backend).await?,
            a2v: AttnViews::acquire(&h.a2v, residency, backend).await?,
            v2a: AttnViews::acquire(&h.v2a, residency, backend).await?,
            ff: FfViews::acquire(&h.ff, residency, backend).await?,
            audio_ff: FfViews::acquire(&h.audio_ff, residency, backend).await?,
        })
    }
    fn bufs(&self) -> BlockBufs {
        BlockBufs {
            attn1: self.attn1.bufs(),
            attn2: self.attn2.bufs(),
            audio_attn1: self.audio_attn1.bufs(),
            audio_attn2: self.audio_attn2.bufs(),
            a2v: self.a2v.bufs(),
            v2a: self.v2a.bufs(),
            ff: self.ff.bufs(),
            audio_ff: self.audio_ff.bufs(),
        }
    }
}

// ---------------------------------------------------------------------------
// Attention sublayer
// ---------------------------------------------------------------------------

/// One attention sublayer: biased q/k/v projections, learned qk rms-norm, SPLIT
/// rope (per-(pos,head) freq rows; q via `q_freqs`, k via `k_freqs`), non-causal
/// SDPA, per-head sigmoid gate (`to_gate_logits(gate_src)`), then biased
/// `to_out`. Returns `[q_rows, q_dim]`. No outer AdaLN gate here (the caller
/// applies it at the residual).
#[allow(clippy::too_many_arguments)]
fn attention<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipes: &DitPipelines,
    g: AttnGeom,
    q_src: ActBuf<'wsp>,
    kv_src: ActBuf<'wsp>,
    gate_src: ActBuf<'wsp>,
    q_freqs: Option<BatchBuf<'wsp>>,
    k_freqs: Option<BatchBuf<'wsp>>,
    w: &AttnBufs,
    q_rows: u32,
    kv_rows: u32,
    eps: f32,
    qkv_i8: bool,
    dq: AttnDequant<'_>,
) -> Result<ActBuf<'wsp>, WgpuError> {
    let bp = &pipes.block;
    let inner = g.inner() as u32;
    let heads = g.heads as u32;
    let hd = g.head_dim as u32;
    let q_dim = g.q_dim as u32;
    let kv_dim = g.kv_dim as u32;
    // DP4A for the q/k/v projections when `qkv_i8`. Their A-sides are all normed
    // (self-attn: the sandwich-normed residual; text-cross: the connector output,
    // `norm_output=true`, modulated; av-cross: ada_zero-normed) -> DP4A-safe
    // (validated: cross-attn i8 adds <0.02% rel). gate (tiny) + to_out (post-gate
    // outliers) stay dense (`I8Site::None`). The i8 dequant is Q8_0-specific, so
    // any non-Q8_0 weight (the Q4_K_M variant) falls to the dense per-kind path.
    let qkv_site = |d: Option<&DequantStep>| {
        if qkv_i8 && d.is_some_and(|s| s.scheme == QuantKind::Q8_0) {
            I8Site::Qkv
        } else {
            I8Site::None
        }
    };

    let q = biased_proj(
        scope,
        bp,
        q_src,
        w.q_w,
        w.q_b,
        q_rows,
        inner,
        q_dim,
        qkv_site(dq.q),
        dq.q,
    )?;
    let k = biased_proj(
        scope,
        bp,
        kv_src,
        w.k_w,
        w.k_b,
        kv_rows,
        inner,
        kv_dim,
        qkv_site(dq.k),
        dq.k,
    )?;
    let v = biased_proj(
        scope,
        bp,
        kv_src,
        w.v_w,
        w.v_b,
        kv_rows,
        inner,
        kv_dim,
        qkv_site(dq.v),
        dq.v,
    )?;

    // learned qk rms-norm over the FULL inner dim.
    let qn = alloc_act(scope, bp, q_rows, inner)?;
    let qnw = scope.import_copy(w.q_norm);
    op_rmsnorm(scope, bp, q, qnw, qn, q_rows, inner, eps)?;
    let kn = alloc_act(scope, bp, kv_rows, inner)?;
    let knw = scope.import_copy(w.k_norm);
    op_rmsnorm(scope, bp, k, knw, kn, kv_rows, inner, eps)?;

    // SPLIT rope (heads collapsed into kernel rows; see module note).
    let qx = match q_freqs {
        Some(f) => {
            let r = alloc_act(scope, bp, q_rows, inner)?;
            op_rope_halfrot(scope, bp, qn, f, r, q_rows * heads, 1, hd)?;
            r
        }
        None => qn,
    };
    let kx = match k_freqs {
        Some(f) => {
            let r = alloc_act(scope, bp, kv_rows, inner)?;
            op_rope_halfrot(scope, bp, kn, f, r, kv_rows * heads, 1, hd)?;
            r
        }
        None => kn,
    };

    // non-causal SDPA (no mask; dummy 16-byte slot, has_mask=0).
    let sa = alloc_act(scope, bp, q_rows, inner)?;
    let no_mask = scope.alloc(16)?;
    op_sdpa(
        scope,
        bp,
        qx,
        kx,
        v,
        no_mask,
        sa,
        1,
        q_rows,
        kv_rows,
        heads,
        heads,
        hd,
        g.scale(),
        0,
    )?;

    // per-head gate: out = sa * 2*sigmoid(to_gate_logits(gate_src)).
    let gate_logits = biased_proj(
        scope,
        bp,
        gate_src,
        w.gate_w,
        w.gate_b,
        q_rows,
        heads,
        q_dim,
        I8Site::None,
        dq.gate,
    )?;
    let gated = alloc_act(scope, bp, q_rows, inner)?;
    scope.dispatch_op::<GatedHeadMulF32>(&pipes.gate, &[sa.data, gate_logits.data], gated.data)?;

    // to_out projection + bias -> [q_rows, q_dim].
    biased_proj(
        scope,
        bp,
        gated,
        w.o_w,
        w.o_b,
        q_rows,
        q_dim,
        inner,
        I8Site::None,
        dq.o,
    )
}

/// One modality's gelu-tanh FFN: `net.0.proj` (-> 4*inner) -> gelu-tanh ->
/// `net.2` (-> inner). Returns `[rows, inner]` (the un-gated FFN output).
fn feed_forward<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipes: &DitPipelines,
    x: ActBuf<'wsp>,
    w: &FfBufs,
    rows: u32,
    inner: u32,
    up_i8: bool,
    dq: FfDequant<'_>,
) -> Result<ActBuf<'wsp>, WgpuError> {
    let bp = &pipes.block;
    let ff_hidden = 4 * inner;
    // ffn_up A-side is the sandwich-normed residual (DP4A-safe); ffn_down reads
    // the post-gelu product (outlier-prone), so it stays dense. `up_i8` is set
    // for the VIDEO ffn only -- the audio stream (small dim, fewer tokens) is too
    // sensitive to i8 at the block-residual level, so it stays fully dense. i8 is
    // Q8_0-specific; a non-Q8_0 ffn_up (Q4_K_M variant) runs the dense per-kind path.
    let up = if up_i8 && dq.up.is_some_and(|s| s.scheme == QuantKind::Q8_0) {
        I8Site::FfnUp
    } else {
        I8Site::None
    };
    let h1 = biased_proj(
        scope, bp, x, w.up_w, w.up_b, rows, ff_hidden, inner, up, dq.up,
    )?;
    let g = alloc_act(scope, bp, rows, ff_hidden)?;
    scope.dispatch_op::<GeluF32>(&pipes.gelu, &[h1.data], g.data)?;
    biased_proj(
        scope,
        bp,
        g,
        w.down_w,
        w.down_b,
        rows,
        inner,
        ff_hidden,
        I8Site::None,
        dq.down,
    )
}

// ---------------------------------------------------------------------------
// Block orchestration (5 attn sublayers + 2 FFNs)
// ---------------------------------------------------------------------------

/// Token counts for one block forward (B=1): video / audio latent tokens and
/// video / audio text-context rows.
#[derive(Clone, Copy, Debug)]
pub struct Streams {
    pub video_tokens: usize,
    pub audio_tokens: usize,
    pub video_text: usize,
    pub audio_text: usize,
}

/// RoPE freq buffers for one block (already on GPU, `[seq*heads, head_dim]`
/// interleaved). `video_self`/`audio_self` are the 3-axis / 1-axis self-attn
/// `pe`; `video_cross`/`audio_cross` are the temporal-only av-cross `cross_pe`.
#[derive(Clone, Copy)]
struct Freqs<'wsp> {
    video_self: BatchBuf<'wsp>,
    audio_self: BatchBuf<'wsp>,
    video_cross: BatchBuf<'wsp>,
    audio_cross: BatchBuf<'wsp>,
}

/// Run one full dual-stream DiT block. `vx`/`ax` are the residual streams
/// (`[video_tokens, DIM]` / `[audio_tokens, AUDIO_DIM]`); `vtext`/`atext` are the
/// cross-attn caption KV. Writes the block outputs into `vx_out`/`ax_out`.
#[allow(clippy::too_many_arguments)]
fn block_forward<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipes: &DitPipelines,
    s: Streams,
    vx_in: ActBuf<'wsp>,
    ax_in: ActBuf<'wsp>,
    vtext: ActBuf<'wsp>,
    atext: ActBuf<'wsp>,
    vmod: &StreamMod<'wsp>,
    amod: &StreamMod<'wsp>,
    freqs: Freqs<'wsp>,
    ones_v: BatchBuf<'wsp>,
    ones_a: BatchBuf<'wsp>,
    bufs: &BlockBufs,
    quant: BlockQuantKinds,
    vx_out: ActBuf<'wsp>,
    ax_out: ActBuf<'wsp>,
) -> Result<(), WgpuError> {
    let bp = &pipes.block;
    let eps = dit::NORM_EPS;
    let vd = dit::DIM as u32;
    let ad = dit::AUDIO_DIM as u32;
    let sv = s.video_tokens as u32;
    let sa = s.audio_tokens as u32;

    // Resolve the dense dequant step for each matmul weight from its probed
    // on-disk quant kind. Uniform across attn types (q/k/o/gate one kind, v
    // another for Q4_K_M); the two FFNs differ only in video vs audio ffn_down.
    let adq = AttnDequant {
        q: pipes.dense_dequant(quant.q),
        k: pipes.dense_dequant(quant.k),
        v: pipes.dense_dequant(quant.v),
        o: pipes.dense_dequant(quant.o),
        gate: pipes.dense_dequant(quant.gate),
    };
    let vff_dq = FfDequant {
        up: pipes.dense_dequant(quant.vff_up),
        down: pipes.dense_dequant(quant.vff_down),
    };
    let aff_dq = FfDequant {
        up: pipes.dense_dequant(quant.aff_up),
        down: pipes.dense_dequant(quant.aff_down),
    };

    // ===================== video self + text cross =====================
    let norm_vx = ada_zero(
        scope,
        bp,
        ones_v,
        vx_in,
        vmod.msa_scale,
        vmod.msa_shift,
        sv,
        vd,
        eps,
    )?;
    let vx_msa = attention(
        scope,
        pipes,
        geom::VIDEO_SELF,
        norm_vx,
        norm_vx,
        norm_vx,
        Some(freqs.video_self),
        Some(freqs.video_self),
        &bufs.attn1,
        sv,
        sv,
        eps,
        true,
        adq,
    )?;
    let vx1 = alloc_act(scope, bp, sv, vd)?;
    op_gate_residual(
        scope,
        bp,
        vx_in.data,
        vmod.msa_gate,
        vx_msa.data,
        vx1.data,
        sv,
        vd,
    )?;
    // post_sa: x_normed = rms_norm_weightless(vx1).
    let vx_normed = alloc_act(scope, bp, sv, vd)?;
    op_rmsnorm(scope, bp, vx1, ones_v, vx_normed, sv, vd, eps)?;
    // text cross-attn with q/kv AdaLN.
    let attn_in = alloc_act(scope, bp, sv, vd)?;
    op_modulate(
        scope,
        bp,
        vx_normed.data,
        vmod.cq_scale,
        vmod.cq_shift,
        attn_in.data,
        sv,
        vd,
    )?;
    let enc = alloc_act(scope, bp, s.video_text as u32, vd)?;
    op_modulate(
        scope,
        bp,
        vtext.data,
        vmod.ckv_scale,
        vmod.ckv_shift,
        enc.data,
        s.video_text as u32,
        vd,
    )?;
    let vca = attention(
        scope,
        pipes,
        geom::VIDEO_CROSS,
        attn_in,
        enc,
        attn_in,
        None,
        None,
        &bufs.attn2,
        sv,
        s.video_text as u32,
        eps,
        true,
        adq,
    )?;
    let vx2 = alloc_act(scope, bp, sv, vd)?;
    op_gate_residual(
        scope,
        bp,
        vx1.data,
        vmod.cq_gate,
        vca.data,
        vx2.data,
        sv,
        vd,
    )?;

    // ===================== audio self + text cross =====================
    let norm_ax = ada_zero(
        scope,
        bp,
        ones_a,
        ax_in,
        amod.msa_scale,
        amod.msa_shift,
        sa,
        ad,
        eps,
    )?;
    let ax_msa = attention(
        scope,
        pipes,
        geom::AUDIO_SELF,
        norm_ax,
        norm_ax,
        norm_ax,
        Some(freqs.audio_self),
        Some(freqs.audio_self),
        &bufs.audio_attn1,
        sa,
        sa,
        eps,
        false,
        adq,
    )?;
    let ax1 = alloc_act(scope, bp, sa, ad)?;
    op_gate_residual(
        scope,
        bp,
        ax_in.data,
        amod.msa_gate,
        ax_msa.data,
        ax1.data,
        sa,
        ad,
    )?;
    let ax_normed = alloc_act(scope, bp, sa, ad)?;
    op_rmsnorm(scope, bp, ax1, ones_a, ax_normed, sa, ad, eps)?;
    let a_attn_in = alloc_act(scope, bp, sa, ad)?;
    op_modulate(
        scope,
        bp,
        ax_normed.data,
        amod.cq_scale,
        amod.cq_shift,
        a_attn_in.data,
        sa,
        ad,
    )?;
    let a_enc = alloc_act(scope, bp, s.audio_text as u32, ad)?;
    op_modulate(
        scope,
        bp,
        atext.data,
        amod.ckv_scale,
        amod.ckv_shift,
        a_enc.data,
        s.audio_text as u32,
        ad,
    )?;
    let aca = attention(
        scope,
        pipes,
        geom::AUDIO_CROSS,
        a_attn_in,
        a_enc,
        a_attn_in,
        None,
        None,
        &bufs.audio_attn2,
        sa,
        s.audio_text as u32,
        eps,
        false,
        adq,
    )?;
    let ax2 = alloc_act(scope, bp, sa, ad)?;
    op_gate_residual(
        scope,
        bp,
        ax1.data,
        amod.cq_gate,
        aca.data,
        ax2.data,
        sa,
        ad,
    )?;

    // ===================== audio <-> video cross =====================
    // vx2/ax2 are the pre-av snapshots (each residual writes a fresh buffer, so
    // the pre-av bindings stay valid for V2A's KV).
    let a2v_vx = ada_zero(
        scope,
        bp,
        ones_v,
        vx2,
        vmod.a2v_scale,
        vmod.a2v_shift,
        sv,
        vd,
        eps,
    )?;
    let a2v_ax = ada_zero(
        scope,
        bp,
        ones_a,
        ax2,
        amod.a2v_scale,
        amod.a2v_shift,
        sa,
        ad,
        eps,
    )?;
    let a2v_out = attention(
        scope,
        pipes,
        geom::A2V,
        a2v_vx,
        a2v_ax,
        a2v_vx,
        Some(freqs.video_cross),
        Some(freqs.audio_cross),
        &bufs.a2v,
        sv,
        sa,
        eps,
        true,
        adq,
    )?;
    let vx3 = alloc_act(scope, bp, sv, vd)?;
    op_gate_residual(
        scope,
        bp,
        vx2.data,
        vmod.av_gate,
        a2v_out.data,
        vx3.data,
        sv,
        vd,
    )?;

    let v2a_ax = ada_zero(
        scope,
        bp,
        ones_a,
        ax2,
        amod.v2a_scale,
        amod.v2a_shift,
        sa,
        ad,
        eps,
    )?;
    let v2a_vx = ada_zero(
        scope,
        bp,
        ones_v,
        vx2,
        vmod.v2a_scale,
        vmod.v2a_shift,
        sv,
        vd,
        eps,
    )?;
    let v2a_out = attention(
        scope,
        pipes,
        geom::V2A,
        v2a_ax,
        v2a_vx,
        v2a_ax,
        Some(freqs.audio_cross),
        Some(freqs.video_cross),
        &bufs.v2a,
        sa,
        sv,
        eps,
        false,
        adq,
    )?;
    let ax3 = alloc_act(scope, bp, sa, ad)?;
    op_gate_residual(
        scope,
        bp,
        ax2.data,
        amod.av_gate,
        v2a_out.data,
        ax3.data,
        sa,
        ad,
    )?;

    // ===================== per-modality FFNs =====================
    let vx_ff_in = ada_zero(
        scope,
        bp,
        ones_v,
        vx3,
        vmod.mlp_scale,
        vmod.mlp_shift,
        sv,
        vd,
        eps,
    )?;
    let vx_ff = feed_forward(scope, pipes, vx_ff_in, &bufs.ff, sv, vd, true, vff_dq)?;
    op_gate_residual(
        scope,
        bp,
        vx3.data,
        vmod.mlp_gate,
        vx_ff.data,
        vx_out.data,
        sv,
        vd,
    )?;

    let ax_ff_in = ada_zero(
        scope,
        bp,
        ones_a,
        ax3,
        amod.mlp_scale,
        amod.mlp_shift,
        sa,
        ad,
        eps,
    )?;
    let ax_ff = feed_forward(
        scope,
        pipes,
        ax_ff_in,
        &bufs.audio_ff,
        sa,
        ad,
        false,
        aff_dq,
    )?;
    op_gate_residual(
        scope,
        bp,
        ax3.data,
        amod.mlp_gate,
        ax_ff.data,
        ax_out.data,
        sa,
        ad,
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Public 1-block entry (parity: modulation + freqs supplied; see ltx-plan P3)
// ---------------------------------------------------------------------------

/// Host AdaLN modulation vectors for one stream (each `[inner]`). Ordered to
/// match `StreamMod`; see `as_slices`.
#[derive(Clone, Debug, Default)]
pub struct HostStreamMod {
    pub msa_scale: Vec<f32>,
    pub msa_shift: Vec<f32>,
    pub msa_gate: Vec<f32>,
    pub cq_scale: Vec<f32>,
    pub cq_shift: Vec<f32>,
    pub cq_gate: Vec<f32>,
    pub ckv_scale: Vec<f32>,
    pub ckv_shift: Vec<f32>,
    pub mlp_scale: Vec<f32>,
    pub mlp_shift: Vec<f32>,
    pub mlp_gate: Vec<f32>,
    pub a2v_scale: Vec<f32>,
    pub a2v_shift: Vec<f32>,
    pub v2a_scale: Vec<f32>,
    pub v2a_shift: Vec<f32>,
    pub av_gate: Vec<f32>,
}

impl HostStreamMod {
    fn as_slices(&self) -> [&[f32]; 16] {
        [
            &self.msa_scale,
            &self.msa_shift,
            &self.msa_gate,
            &self.cq_scale,
            &self.cq_shift,
            &self.cq_gate,
            &self.ckv_scale,
            &self.ckv_shift,
            &self.mlp_scale,
            &self.mlp_shift,
            &self.mlp_gate,
            &self.a2v_scale,
            &self.a2v_shift,
            &self.v2a_scale,
            &self.v2a_shift,
            &self.av_gate,
        ]
    }
}

/// RoPE freq tables for one block, host-built (`[seq*heads, head_dim]`).
pub struct HostFreqs {
    pub video_self: Vec<f32>,
    pub audio_self: Vec<f32>,
    pub video_cross: Vec<f32>,
    pub audio_cross: Vec<f32>,
}

type WsBuf = thinfer_core::workspace::WsBuf<WgpuBackend>;

/// Bytes per activation element for the given dtype (F32 = 4, F16/Bf16 = 2).
pub(crate) fn act_bytes(act: ActDtype) -> u64 {
    match act {
        ActDtype::F32 => 4,
        _ => 2,
    }
}

fn upload(
    backend: &WgpuBackend,
    workspace: &Workspace<WgpuBackend>,
    data: &[f32],
    act: ActDtype,
) -> Result<WsBuf, WgpuError> {
    let b = workspace.alloc(data.len() as u64 * act_bytes(act))?;
    backend.write_buffer(b.id(), 0, &crate::common::seq::act_upload_bytes(act, data))?;
    Ok(b)
}

/// Run one DiT block with externally-supplied AdaLN modulation + rope freqs
/// (the P3 parity entry: isolates the block + rope-apply from the timestep/adaln
/// and patchifier paths). Returns `(vx_out, ax_out)` host f32.
#[allow(clippy::too_many_arguments)]
pub async fn forward_block_dumped<S: WeightSource>(
    backend: &WgpuBackend,
    pipes: &DitPipelines,
    residency: &WeightResidency<S>,
    workspace: &Workspace<WgpuBackend>,
    handles: &BlockHandles,
    s: Streams,
    vx: &[f32],
    ax: &[f32],
    vtext: &[f32],
    atext: &[f32],
    vmod: &HostStreamMod,
    amod: &HostStreamMod,
    freqs: &HostFreqs,
) -> Result<(Vec<f32>, Vec<f32>), DitError<S::Error>> {
    let vd = dit::DIM;
    let ad = dit::AUDIO_DIM;
    assert_eq!(vx.len(), s.video_tokens * vd, "vx shape");
    assert_eq!(ax.len(), s.audio_tokens * ad, "ax shape");
    assert_eq!(vtext.len(), s.video_text * vd, "vtext shape");
    assert_eq!(atext.len(), s.audio_text * ad, "atext shape");

    // Acquire every weight view (held across the scope).
    let v = handles;
    let attn1 = AttnViews::acquire(&v.attn1, residency, backend).await?;
    let attn2 = AttnViews::acquire(&v.attn2, residency, backend).await?;
    let audio_attn1 = AttnViews::acquire(&v.audio_attn1, residency, backend).await?;
    let audio_attn2 = AttnViews::acquire(&v.audio_attn2, residency, backend).await?;
    let a2v = AttnViews::acquire(&v.a2v, residency, backend).await?;
    let v2a = AttnViews::acquire(&v.v2a, residency, backend).await?;
    let ff = FfViews::acquire(&v.ff, residency, backend).await?;
    let audio_ff = FfViews::acquire(&v.audio_ff, residency, backend).await?;
    let bufs = BlockBufs {
        attn1: attn1.bufs(),
        attn2: attn2.bufs(),
        audio_attn1: audio_attn1.bufs(),
        audio_attn2: audio_attn2.bufs(),
        a2v: a2v.bufs(),
        v2a: v2a.bufs(),
        ff: ff.bufs(),
        audio_ff: audio_ff.bufs(),
    };

    // Upload inputs / freqs / modulation / ones gains (in the block act dtype).
    let act = pipes.block.act_dtype;
    let ab = act_bytes(act);
    let vx_b = upload(backend, workspace, vx, act)?;
    let ax_b = upload(backend, workspace, ax, act)?;
    let vtext_b = upload(backend, workspace, vtext, act)?;
    let atext_b = upload(backend, workspace, atext, act)?;
    let vsf = upload(backend, workspace, &freqs.video_self, act)?;
    let asf = upload(backend, workspace, &freqs.audio_self, act)?;
    let vcf = upload(backend, workspace, &freqs.video_cross, act)?;
    let acf = upload(backend, workspace, &freqs.audio_cross, act)?;
    let ones_v_b = workspace.alloc(bf16_ones_words(vd).len() as u64)?;
    backend.write_buffer(ones_v_b.id(), 0, &bf16_ones_words(vd))?;
    let ones_a_b = workspace.alloc(bf16_ones_words(ad).len() as u64)?;
    backend.write_buffer(ones_a_b.id(), 0, &bf16_ones_words(ad))?;
    let mut vmod_bufs = Vec::with_capacity(16);
    for sl in vmod.as_slices() {
        vmod_bufs.push(upload(backend, workspace, sl, act)?);
    }
    let mut amod_bufs = Vec::with_capacity(16);
    for sl in amod.as_slices() {
        amod_bufs.push(upload(backend, workspace, sl, act)?);
    }
    let vx_out_b = workspace.alloc(vx.len() as u64 * ab)?;
    let ax_out_b = workspace.alloc(ax.len() as u64 * ab)?;

    {
        let scope = workspace.batch();
        let imp = |b: &WsBuf| scope.import_copy(b.as_buf_ref());
        let vi: Vec<_> = vmod_bufs.iter().map(&imp).collect();
        let ai: Vec<_> = amod_bufs.iter().map(&imp).collect();
        let vmod_s = StreamMod {
            msa_scale: vi[0],
            msa_shift: vi[1],
            msa_gate: vi[2],
            cq_scale: vi[3],
            cq_shift: vi[4],
            cq_gate: vi[5],
            ckv_scale: vi[6],
            ckv_shift: vi[7],
            mlp_scale: vi[8],
            mlp_shift: vi[9],
            mlp_gate: vi[10],
            a2v_scale: vi[11],
            a2v_shift: vi[12],
            v2a_scale: vi[13],
            v2a_shift: vi[14],
            av_gate: vi[15],
        };
        let amod_s = StreamMod {
            msa_scale: ai[0],
            msa_shift: ai[1],
            msa_gate: ai[2],
            cq_scale: ai[3],
            cq_shift: ai[4],
            cq_gate: ai[5],
            ckv_scale: ai[6],
            ckv_shift: ai[7],
            mlp_scale: ai[8],
            mlp_shift: ai[9],
            mlp_gate: ai[10],
            a2v_scale: ai[11],
            a2v_shift: ai[12],
            v2a_scale: ai[13],
            v2a_shift: ai[14],
            av_gate: ai[15],
        };
        let fr = Freqs {
            video_self: imp(&vsf),
            audio_self: imp(&asf),
            video_cross: imp(&vcf),
            audio_cross: imp(&acf),
        };
        block_forward(
            &scope,
            pipes,
            s,
            ActBuf::dense(imp(&vx_b)),
            ActBuf::dense(imp(&ax_b)),
            ActBuf::dense(imp(&vtext_b)),
            ActBuf::dense(imp(&atext_b)),
            &vmod_s,
            &amod_s,
            fr,
            imp(&ones_v_b),
            imp(&ones_a_b),
            &bufs,
            BlockQuantKinds::probe(residency),
            ActBuf::dense(imp(&vx_out_b)),
            ActBuf::dense(imp(&ax_out_b)),
        )?;
        scope.submit_void().await?;
    }

    let vbytes = backend
        .read_buffer(vx_out_b.id(), 0, vx.len() as u64 * ab)
        .await?;
    let abytes = backend
        .read_buffer(ax_out_b.id(), 0, ax.len() as u64 * ab)
        .await?;
    Ok((
        crate::common::seq::act_readback_to_f32(act, &vbytes, vx.len()),
        crate::common::seq::act_readback_to_f32(act, &abytes, ax.len()),
    ))
}

// ---------------------------------------------------------------------------
// Full DiT forward (patchify -> N blocks -> output), streams persistent on GPU
// ---------------------------------------------------------------------------

use super::cond::{self, BlockTables, TimestepHandles};

/// Upload a host `[n]` f32 vector into a persistent workspace buffer (act dtype).
fn ws_upload(
    backend: &WgpuBackend,
    workspace: &Workspace<WgpuBackend>,
    data: &[f32],
    act: ActDtype,
) -> Result<WsBuf, WgpuError> {
    let b = workspace.alloc(data.len() as u64 * act_bytes(act))?;
    backend.write_buffer(b.id(), 0, &crate::common::seq::act_upload_bytes(act, data))?;
    Ok(b)
}

/// Upload one stream's 16 modulation vectors; returns the workspace bufs.
fn upload_mod(
    backend: &WgpuBackend,
    workspace: &Workspace<WgpuBackend>,
    m: &HostStreamMod,
    act: ActDtype,
) -> Result<Vec<WsBuf>, WgpuError> {
    m.as_slices()
        .iter()
        .map(|sl| ws_upload(backend, workspace, sl, act))
        .collect()
}

fn stream_mod_from<'wsp>(scope: &BatchScope<'wsp, WgpuBackend>, b: &[WsBuf]) -> StreamMod<'wsp> {
    let i: Vec<_> = b
        .iter()
        .map(|w| scope.import_copy(w.as_buf_ref()))
        .collect();
    StreamMod {
        msa_scale: i[0],
        msa_shift: i[1],
        msa_gate: i[2],
        cq_scale: i[3],
        cq_shift: i[4],
        cq_gate: i[5],
        ckv_scale: i[6],
        ckv_shift: i[7],
        mlp_scale: i[8],
        mlp_shift: i[9],
        mlp_gate: i[10],
        a2v_scale: i[11],
        a2v_shift: i[12],
        v2a_scale: i[13],
        v2a_shift: i[14],
        av_gate: i[15],
    }
}

/// `out = patchify_proj(latent) + bias` (bf16 matmul, latent channels -> inner).
fn patchify<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipes: &DitPipelines,
    latent: ActBuf<'wsp>,
    w: BufRef,
    b: BufRef,
    rows: u32,
    out: u32,
    in_ch: u32,
) -> Result<ActBuf<'wsp>, WgpuError> {
    let bp = &pipes.block;
    let pre = alloc_matmul_out_buf(scope, bp, rows * out)?;
    let dims = scope.u32x4_uniform(rows, out, in_ch, 0)?;
    let wv = scope.import_copy(w);
    scope.matmul(
        &pipes.block.matmul_adaln,
        &pipes.block.matmuls.adaln,
        latent.data,
        wv,
        dims,
        pre,
        rows,
        out,
    )?;
    let bv = scope.import_copy(b);
    let dst = alloc_act(scope, bp, rows, out)?;
    op_bias_add(scope, bp, pre, bv, dst.data, rows, out)?;
    Ok(dst)
}

/// Output stage: `proj_out(norm_out(x) * (1+scale) + shift)`. `scale`/`shift`
/// are act vectors (`scale_shift_table_out[k] + embedded_timestep`). `norm_out`
/// is LayerNorm with NO affine. Returns `[rows, out_ch]`.
#[allow(clippy::too_many_arguments)]
fn output_stage<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipes: &DitPipelines,
    x: ActBuf<'wsp>,
    scale: BatchBuf<'wsp>,
    shift: BatchBuf<'wsp>,
    proj_w: BufRef,
    proj_b: BufRef,
    rows: u32,
    dim: u32,
    out_ch: u32,
    eps: f32,
) -> Result<ActBuf<'wsp>, WgpuError> {
    use thinfer_core::ops::LayerNormF32;
    let bp = &pipes.block;
    let n = alloc_act(scope, bp, rows, dim)?;
    let u = scope.u32x4_uniform(rows, dim, eps.to_bits(), 0)?;
    scope.layernorm::<LayerNormF32>(&bp.layernorm, x.data, u, n.data, rows)?;
    let m = alloc_act(scope, bp, rows, dim)?;
    op_modulate(scope, bp, n.data, scale, shift, m.data, rows, dim)?;
    let pre = alloc_matmul_out_buf(scope, bp, rows * out_ch)?;
    let dims = scope.u32x4_uniform(rows, out_ch, dim, 0)?;
    let wv = scope.import_copy(proj_w);
    scope.matmul(
        &pipes.block.matmul_adaln,
        &pipes.block.matmuls.adaln,
        m.data,
        wv,
        dims,
        pre,
        rows,
        out_ch,
    )?;
    let bv = scope.import_copy(proj_b);
    let dst = alloc_act(scope, bp, rows, out_ch)?;
    op_bias_add(scope, bp, pre, bv, dst.data, rows, out_ch)?;
    Ok(dst)
}

/// Read a resident weight back to host as bf16-rounded f32 (`n` elems).
async fn read_weight_bf16<S: WeightSource>(
    backend: &WgpuBackend,
    residency: &WeightResidency<S>,
    h: WeightHandle,
    n: usize,
) -> Result<Vec<f32>, DitError<S::Error>> {
    let view = residency.acquire(h, backend).await?;
    let bytes = backend
        .read_buffer(view.buf().id, 0, (n * 2) as u64)
        .await?;
    Ok(crate::common::seq::act_readback_to_f32(
        ActDtype::Bf16,
        &bytes,
        n,
    ))
}

/// The full DiT model handles: timestep/adaln modules, the per-block weights +
/// host-cached modulation tables, and the top-level I/O.
pub struct DitModel {
    pub timestep: TimestepHandles,
    pub blocks: Vec<BlockHandles>,
    pub tables: Vec<BlockTables>,
    pub io: IoHandles,
    /// On-disk quant kind per matmul site (probed from block 0; uniform across
    /// blocks). Q8_0 for the baseline file; mixed Q4_K/Q6_K for the Q4_K_M variant.
    quant: BlockQuantKinds,
}

impl DitModel {
    /// Register `num_layers` blocks + timestep + I/O and cache the per-block
    /// modulation tables on host (read once; constant across denoise steps).
    pub async fn register<S: WeightSource>(
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        num_layers: usize,
    ) -> Result<Self, DitError<S::Error>> {
        let timestep = cond::register_timestep(residency)?;
        let io = register_io(residency)?;
        let mut blocks = Vec::with_capacity(num_layers);
        let mut tables = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let h = register_block(residency, i)?;
            tables.push(cond::read_block_tables(backend, residency, &h).await?);
            blocks.push(h);
        }
        let quant = BlockQuantKinds::probe(residency);
        Ok(Self {
            timestep,
            blocks,
            tables,
            io,
            quant,
        })
    }

    /// Full forward for one denoise step: patchify the raw latents, run every
    /// block (streams kept on GPU), apply the output stage. Returns the velocity
    /// predictions `(video [Tv,128], audio [Ta,128])` host f32. `sigma` is the
    /// scalar timestep (uniform); `freqs` the rope tables for the latent grid.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipes: &DitPipelines,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        s: Streams,
        latent_v: &[f32],
        latent_a: &[f32],
        vtext: &[f32],
        atext: &[f32],
        sigma: f32,
        freqs: &HostFreqs,
    ) -> Result<(Vec<f32>, Vec<f32>), DitError<S::Error>> {
        let vd = dit::DIM;
        let ad = dit::AUDIO_DIM;
        let inc = dit::IN_CHANNELS;
        let outc = dit::OUT_CHANNELS;
        let eps = dit::NORM_EPS;
        let sv = s.video_tokens as u32;
        let sa = s.audio_tokens as u32;

        // Timestep / adaln (shared across blocks).
        let shared = cond::compute_shared_timestep(
            backend,
            pipes,
            residency,
            workspace,
            &self.timestep,
            sigma,
            sigma,
        )
        .await?;

        // Persistent buffers: latents, text, freqs, ones (in the block act dtype).
        let act = pipes.block.act_dtype;
        let ab = act_bytes(act);
        let latv = ws_upload(backend, workspace, latent_v, act)?;
        let lata = ws_upload(backend, workspace, latent_a, act)?;
        let vtext_b = ws_upload(backend, workspace, vtext, act)?;
        let atext_b = ws_upload(backend, workspace, atext, act)?;
        let vsf = ws_upload(backend, workspace, &freqs.video_self, act)?;
        let asf = ws_upload(backend, workspace, &freqs.audio_self, act)?;
        let vcf = ws_upload(backend, workspace, &freqs.video_cross, act)?;
        let acf = ws_upload(backend, workspace, &freqs.audio_cross, act)?;
        let ones_v_b = workspace.alloc(bf16_ones_words(vd).len() as u64)?;
        backend.write_buffer(ones_v_b.id(), 0, &bf16_ones_words(vd))?;
        let ones_a_b = workspace.alloc(bf16_ones_words(ad).len() as u64)?;
        backend.write_buffer(ones_a_b.id(), 0, &bf16_ones_words(ad))?;

        // Patchify both streams into the persistent residual buffers.
        let io = &self.io;
        let iov = (
            residency.acquire(io.patchify_v_w, backend).await?,
            residency.acquire(io.patchify_v_b, backend).await?,
            residency.acquire(io.patchify_a_w, backend).await?,
            residency.acquire(io.patchify_a_b, backend).await?,
        );
        let mut vx_b = workspace.alloc(s.video_tokens as u64 * vd as u64 * ab)?;
        let mut ax_b = workspace.alloc(s.audio_tokens as u64 * ad as u64 * ab)?;
        {
            let scope = workspace.batch();
            let lv = ActBuf::dense(scope.import_copy(latv.as_buf_ref()));
            let la = ActBuf::dense(scope.import_copy(lata.as_buf_ref()));
            let vx = patchify(
                &scope,
                pipes,
                lv,
                iov.0.buf(),
                iov.1.buf(),
                sv,
                vd as u32,
                inc as u32,
            )?;
            let ax = patchify(
                &scope,
                pipes,
                la,
                iov.2.buf(),
                iov.3.buf(),
                sa,
                ad as u32,
                inc as u32,
            )?;
            let vh = scope.import_copy(vx_b.as_buf_ref());
            let ah = scope.import_copy(ax_b.as_buf_ref());
            scope.copy_buffer_to_buffer(
                vx.data,
                0,
                vh,
                0,
                s.video_tokens as u64 * vd as u64 * ab,
            )?;
            scope.copy_buffer_to_buffer(
                ax.data,
                0,
                ah,
                0,
                s.audio_tokens as u64 * ad as u64 * ab,
            )?;
            scope.submit_void().await?;
        }
        drop(iov);

        // Run every block. The 22.8GB DiT exceeds the VRAM budget, so each block's
        // weights stream from disk on acquire; to keep that streaming off the
        // critical path we prefetch block i+1's weights WHILE block i's GPU work
        // runs (`join!` the submit with the next acquire), turning per-block wall
        // from compute+stream into ~max(compute, stream). (Pattern mirrors the LTX
        // text encoder + z-image DiT.) Residual streams persist on GPU between
        // blocks; modulation is assembled + uploaded per block.
        let n_blocks = self.blocks.len();
        let mut pending = if n_blocks == 0 {
            None
        } else {
            Some(BlockViewsAll::acquire(&self.blocks[0], residency, backend).await?)
        };
        for i in 0..n_blocks {
            let (vmod_h, amod_h) = cond::assemble_block_mod(&shared, &self.tables[i]);
            let vmod_b = upload_mod(backend, workspace, &vmod_h, act)?;
            let amod_b = upload_mod(backend, workspace, &amod_h, act)?;
            let views = pending.take().expect("pending block views");
            let bufs = views.bufs();
            let vx_out = workspace.alloc(s.video_tokens as u64 * vd as u64 * ab)?;
            let ax_out = workspace.alloc(s.audio_tokens as u64 * ad as u64 * ab)?;
            let scope = workspace.batch();
            let vmod_s = stream_mod_from(&scope, &vmod_b);
            let amod_s = stream_mod_from(&scope, &amod_b);
            let fr = Freqs {
                video_self: scope.import_copy(vsf.as_buf_ref()),
                audio_self: scope.import_copy(asf.as_buf_ref()),
                video_cross: scope.import_copy(vcf.as_buf_ref()),
                audio_cross: scope.import_copy(acf.as_buf_ref()),
            };
            block_forward(
                &scope,
                pipes,
                s,
                ActBuf::dense(scope.import_copy(vx_b.as_buf_ref())),
                ActBuf::dense(scope.import_copy(ax_b.as_buf_ref())),
                ActBuf::dense(scope.import_copy(vtext_b.as_buf_ref())),
                ActBuf::dense(scope.import_copy(atext_b.as_buf_ref())),
                &vmod_s,
                &amod_s,
                fr,
                scope.import_copy(ones_v_b.as_buf_ref()),
                scope.import_copy(ones_a_b.as_buf_ref()),
                &bufs,
                self.quant,
                ActBuf::dense(scope.import_copy(vx_out.as_buf_ref())),
                ActBuf::dense(scope.import_copy(ax_out.as_buf_ref())),
            )?;
            // Prefetch block i+1's weights concurrently with block i's submit.
            let next_acquire = async {
                match self.blocks.get(i + 1) {
                    Some(h) => Ok::<_, ResidencyError<S::Error, WgpuError>>(Some(
                        BlockViewsAll::acquire(h, residency, backend).await?,
                    )),
                    None => Ok(None),
                }
            };
            let submit_fut = scope.submit_void();
            let (submit_res, next_res) = futures::join!(submit_fut, next_acquire);
            submit_res?;
            pending = next_res?;
            drop(views);
            vx_b = vx_out;
            ax_b = ax_out;
        }

        // Output stage: per-stream scale/shift = sst_out_row + embedded_timestep.
        let sst_v = read_weight_bf16(backend, residency, io.sst_out_v, 2 * vd).await?;
        let sst_a = read_weight_bf16(backend, residency, io.sst_out_a, 2 * ad).await?;
        let add_emb = |row: &[f32], emb: &[f32]| -> Vec<f32> {
            row.iter().zip(emb).map(|(a, b)| a + b).collect()
        };
        let v_shift = ws_upload(
            backend,
            workspace,
            &add_emb(&sst_v[0..vd], &shared.v_emb),
            act,
        )?;
        let v_scale = ws_upload(
            backend,
            workspace,
            &add_emb(&sst_v[vd..2 * vd], &shared.v_emb),
            act,
        )?;
        let a_shift = ws_upload(
            backend,
            workspace,
            &add_emb(&sst_a[0..ad], &shared.a_emb),
            act,
        )?;
        let a_scale = ws_upload(
            backend,
            workspace,
            &add_emb(&sst_a[ad..2 * ad], &shared.a_emb),
            act,
        )?;
        let projv = (
            residency.acquire(io.proj_out_v_w, backend).await?,
            residency.acquire(io.proj_out_v_b, backend).await?,
            residency.acquire(io.proj_out_a_w, backend).await?,
            residency.acquire(io.proj_out_a_b, backend).await?,
        );
        let vel_v_b = workspace.alloc(s.video_tokens as u64 * outc as u64 * ab)?;
        let vel_a_b = workspace.alloc(s.audio_tokens as u64 * outc as u64 * ab)?;
        {
            let scope = workspace.batch();
            let vx = ActBuf::dense(scope.import_copy(vx_b.as_buf_ref()));
            let ax = ActBuf::dense(scope.import_copy(ax_b.as_buf_ref()));
            let vel_v = output_stage(
                &scope,
                pipes,
                vx,
                scope.import_copy(v_scale.as_buf_ref()),
                scope.import_copy(v_shift.as_buf_ref()),
                projv.0.buf(),
                projv.1.buf(),
                sv,
                vd as u32,
                outc as u32,
                eps,
            )?;
            let vel_a = output_stage(
                &scope,
                pipes,
                ax,
                scope.import_copy(a_scale.as_buf_ref()),
                scope.import_copy(a_shift.as_buf_ref()),
                projv.2.buf(),
                projv.3.buf(),
                sa,
                ad as u32,
                outc as u32,
                eps,
            )?;
            let vh = scope.import_copy(vel_v_b.as_buf_ref());
            let ah = scope.import_copy(vel_a_b.as_buf_ref());
            scope.copy_buffer_to_buffer(
                vel_v.data,
                0,
                vh,
                0,
                s.video_tokens as u64 * outc as u64 * ab,
            )?;
            scope.copy_buffer_to_buffer(
                vel_a.data,
                0,
                ah,
                0,
                s.audio_tokens as u64 * outc as u64 * ab,
            )?;
            scope.submit_void().await?;
        }
        drop(projv);

        let vbytes = backend
            .read_buffer(vel_v_b.id(), 0, s.video_tokens as u64 * outc as u64 * ab)
            .await?;
        let abytes = backend
            .read_buffer(vel_a_b.id(), 0, s.audio_tokens as u64 * outc as u64 * ab)
            .await?;
        Ok((
            crate::common::seq::act_readback_to_f32(act, &vbytes, s.video_tokens * outc),
            crate::common::seq::act_readback_to_f32(act, &abytes, s.audio_tokens * outc),
        ))
    }
}
