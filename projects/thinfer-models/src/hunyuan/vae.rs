//! HunyuanVideo 1.5 VAE decode (`AutoencoderKLConv3D` decoder, Wan-VAE residual
//! family). Whole-tensor SINGLE-SUBMIT decode (correctness-first; the LTX
//! `video_vae.rs` template, not the wan per-frame `feat_cache` streaming). Source
//! of truth: `third-party/HunyuanVideo-1.5/.../hunyuanvideo_15_vae.py` `Decoder`.
//!
//! 16x spatial / 4x temporal, z32, decoder block_out (reversed) [1024,1024,512,
//! 256,128], num_res_blocks 2 (=> 3 resnets per up stage). Graph: conv_in (+
//! `repeat_interleave` residual) -> mid(resnet, causal-attn, resnet) -> 5 up
//! stages (3 resnets + optional upsample) -> norm_out -> silu -> conv_out -> raw
//! `[-1,1]` video (the pipeline maps to `[0,1]` via `*0.5+0.5`). The Hunyuan
//! decoder DIVERGES from the wan decoder in two ways this fork reproduces:
//! - `mid.attn_1` is CAUSAL spatio-temporal (frame i attends 0..i over f*h*w
//!   tokens): the `SdpaF32LargeDCausal` kernel clamps the key loop per frame
//!   (no materialized `[N,N]` mask; fits production frame counts).
//! - `Upsample` = one causal conv -> `out*factor` then pixelshuffle (temporal
//!   upsamplers split the first latent frame spatial-only with half channels) +
//!   `repeat_interleave` residual: the `HunyuanUpsample3d` op.
//!
//! Causal `CausalConv3d` = `F.pad(mode='replicate')` front-pad T by kt-1 +
//! symmetric H/W; realized with the conv3d op `pad_mode=1` (replicate-edge
//! clamp), so a kt=3 causal conv is ONE `conv3d` over the whole `[C,T,H,W]` (no
//! pre-pad, no feat_cache). RMS_norm = channel-first RMS (`* sqrt(C) * gamma`) =
//! the wan `RmsNorm3dF32` op. Single whole-tensor submit (correctness-first;
//! parity vs `gen_vae_decode_ref.py`). Mid-attn causality is on the fly (no
//! materialized mask). Production up-stage tiling is the remaining perf follow-up.

pub mod encode;

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError};
use thinfer_core::ops::{
    ActDtype, AddF32, Conv3dConfig, Conv3dF32, Conv3dOp, HunyuanUpsample3dF32, HunyuanUpsample3dOp,
    Op, RmsNorm3dF32, RmsNorm3dOp, SdpaF32LargeDCausal, SdpaOp, SiluF32, Transpose12F32,
    Transpose12Op, WeightDtype, WgslConfig,
};
use thinfer_core::residency::{GpuView, ResidencyError, WeightHandle, WeightResidency};
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace, WsBuf};

use crate::common::loader::{LoadError, register_passthrough};
use crate::common::seq::{act_readback_to_f32, act_upload_bytes};
use crate::common::vae_tiling::{
    TEMPORAL_TILE_MAX, TEMPORAL_TILE_MIN, TILE_MAX, TILE_MIN, accumulate_spatial_wsum, blend_tile,
    feather_1d, gather_subtile, plan_temporal_tiles, plan_tiles, trapezoid_mask,
};

// ============================================================================
// Config (fixed for this checkpoint; decoder block_out reversed).
// ============================================================================

/// Latent (z) channels, decoder input.
pub const LATENT_CHANNELS: usize = 32;
/// `conv_in` output / mid width (= reversed block_out[0]).
pub const MID_CHANNELS: usize = 1024;
/// `conv_in` residual = `z.repeat_interleave(MID_CHANNELS / LATENT_CHANNELS)`.
const REPEAT: usize = MID_CHANNELS / LATENT_CHANNELS; // 32
/// Pipeline pre-scale: the decoder network consumes `latent / SCALING_FACTOR`.
pub const SCALING_FACTOR: f32 = 1.03682;
/// Output RGB channels (`conv_out`).
pub const OUT_CHANNELS: usize = 3;
/// Resnets per up stage (`num_res_blocks + 1`).
const RES_PER_STAGE: usize = 3;

// --- up-stage tiling (overlap-blend; geometry in `common::vae_tiling`) --------
/// Output pixels per latent cell (16x spatial) and frames per latent frame
/// (4x temporal = two 2x causal upsamples composing `f -> 4*(f-1)+1`).
const SPATIAL_SCALE: usize = 16;
const TEMPORAL_SCALE: usize = 4;
/// Spatial tile side aimed for before introducing temporal tiling (latent
/// cells; 8 = 128 px). Below this the temporal depth is capped so the spatial
/// tile can re-grow, balancing seams across axes.
const SPATIAL_COMFORT: u32 = 8;
/// Upload/readback staging + fragmentation headroom held out of the tile budget.
const VAE_STAGING_RESERVE: u64 = 256 * 1024 * 1024;
/// Fraction of the workspace budget the INITIAL tile is sized to (margin off the
/// OOM boundary; the strict-budget retry corrects any under-shoot).
const SEED_SAFETY: f64 = 0.82;
/// Budget step-down per OOM re-seed (balanced shrink, not an axis halve).
const OOM_SHRINK: f64 = 0.7;
/// Reference budget when the residency budget is unbounded (`u64::MAX`).
const DEFAULT_REF_BUDGET: u64 = 4 * 1024 * 1024 * 1024;
/// Approx peak up-stage workspace bytes per (latent tile area * output frame) at
/// f32 acts. The up-stages carry the whole decode's activation peak (128ch at
/// 16x spatial); calibrated to seed the first tile only (the adaptive retry
/// fixes any error without a device OOM). ~9e6 anchors the LTX 32x decoder; the
/// Hunyuan 16x stack peaks at a similar per-output-frame area cost.
const PEAK_BYTES_PER_AREA_FRAME_F32: f64 = 9.0e6;

/// One decoder up stage: `RES_PER_STAGE` channel-preserving resnets then an
/// optional upsampler. `ch` = stage channels (resnet width = upsampler input).
struct UpStage {
    ch: usize,
    /// `Some((out_c, temporal))` upsamples to `out_c` (temporal = 8x channel
    /// pixelshuffle + frame split; spatial-only = 4x); `None` = final stage.
    upsample: Option<(usize, bool)>,
}

/// Decoder up stages (block_out reversed [1024,1024,512,256,128]; ffactor 16
/// spatial / 4 temporal -> temporal-up on stages 0,1; spatial-up on 0..3).
const UP_STAGES: [UpStage; 5] = [
    UpStage {
        ch: 1024,
        upsample: Some((1024, true)),
    },
    UpStage {
        ch: 1024,
        upsample: Some((512, true)),
    },
    UpStage {
        ch: 512,
        upsample: Some((256, false)),
    },
    UpStage {
        ch: 256,
        upsample: Some((128, false)),
    },
    UpStage {
        ch: 128,
        upsample: None,
    },
];

// ============================================================================
// Weight ids / handles.
// ============================================================================

#[derive(Clone)]
struct ConvW {
    weight: WeightId,
    bias: WeightId,
}

fn conv_w(prefix: &str) -> ConvW {
    ConvW {
        weight: WeightId(format!("{prefix}.weight")),
        bias: WeightId(format!("{prefix}.bias")),
    }
}

/// RMS_norm: channel-first gamma `[C,1,1,1]` (gamma only, no bias).
#[derive(Clone)]
struct RmsW {
    gamma: WeightId,
}

fn rms_w(prefix: &str) -> RmsW {
    RmsW {
        gamma: WeightId(format!("{prefix}.gamma")),
    }
}

/// `ResnetBlock` (channel-preserving in the decoder): norm1 -> silu -> conv1 ->
/// norm2 -> silu -> conv2 + x.
#[derive(Clone)]
struct ResnetW {
    norm1: RmsW,
    conv1: ConvW,
    norm2: RmsW,
    conv2: ConvW,
}

fn resnet_w(prefix: &str) -> ResnetW {
    ResnetW {
        norm1: rms_w(&format!("{prefix}.norm1")),
        conv1: conv_w(&format!("{prefix}.conv1.conv")),
        norm2: rms_w(&format!("{prefix}.norm2")),
        conv2: conv_w(&format!("{prefix}.conv2.conv")),
    }
}

/// `AttnBlock`: norm -> (q,k,v 1x1x1) -> causal sdpa -> proj_out (1x1x1) + x.
/// q/k/v/proj are SEPARATE convs (not a fused to_qkv).
#[derive(Clone)]
struct AttnW {
    norm: RmsW,
    q: ConvW,
    k: ConvW,
    v: ConvW,
    proj: ConvW,
}

fn attn_w(prefix: &str) -> AttnW {
    AttnW {
        norm: rms_w(&format!("{prefix}.norm")),
        q: conv_w(&format!("{prefix}.q")),
        k: conv_w(&format!("{prefix}.k")),
        v: conv_w(&format!("{prefix}.v")),
        proj: conv_w(&format!("{prefix}.proj_out")),
    }
}

struct MidW {
    block_1: ResnetW,
    attn_1: AttnW,
    block_2: ResnetW,
}

/// One up stage: `RES_PER_STAGE` resnets + optional upsampler conv.
struct UpStageW {
    resnets: Vec<ResnetW>,
    upsample_conv: Option<ConvW>,
}

struct DecoderW {
    conv_in: ConvW,
    mid: MidW,
    up: Vec<UpStageW>,
    norm_out: RmsW,
    conv_out: ConvW,
}

impl DecoderW {
    fn new() -> Self {
        let up = UP_STAGES
            .iter()
            .enumerate()
            .map(|(i, st)| UpStageW {
                resnets: (0..RES_PER_STAGE)
                    .map(|j| resnet_w(&format!("decoder.up.{i}.block.{j}")))
                    .collect(),
                upsample_conv: st
                    .upsample
                    .map(|_| conv_w(&format!("decoder.up.{i}.upsample.conv.conv"))),
            })
            .collect();
        Self {
            conv_in: conv_w("decoder.conv_in.conv"),
            mid: MidW {
                block_1: resnet_w("decoder.mid.block_1"),
                attn_1: attn_w("decoder.mid.attn_1"),
                block_2: resnet_w("decoder.mid.block_2"),
            },
            up,
            norm_out: rms_w("decoder.norm_out"),
            conv_out: conv_w("decoder.conv_out.conv"),
        }
    }
}

#[derive(Clone, Copy)]
struct ConvH {
    weight: WeightHandle,
    bias: WeightHandle,
}

#[derive(Clone, Copy)]
struct RmsH {
    gamma: WeightHandle,
}

#[derive(Clone, Copy)]
struct ResnetH {
    norm1: RmsH,
    conv1: ConvH,
    norm2: RmsH,
    conv2: ConvH,
}

#[derive(Clone, Copy)]
struct AttnH {
    norm: RmsH,
    q: ConvH,
    k: ConvH,
    v: ConvH,
    proj: ConvH,
}

struct MidH {
    block_1: ResnetH,
    attn_1: AttnH,
    block_2: ResnetH,
}

struct UpStageH {
    resnets: Vec<ResnetH>,
    upsample_conv: Option<ConvH>,
}

struct DecoderH {
    conv_in: ConvH,
    mid: MidH,
    up: Vec<UpStageH>,
    norm_out: RmsH,
    conv_out: ConvH,
}

fn reg_conv<S: WeightSource>(res: &WeightResidency<S>, w: &ConvW) -> Result<ConvH, LoadError> {
    Ok(ConvH {
        weight: register_passthrough(res, &w.weight)?,
        bias: register_passthrough(res, &w.bias)?,
    })
}

fn reg_rms<S: WeightSource>(res: &WeightResidency<S>, w: &RmsW) -> Result<RmsH, LoadError> {
    Ok(RmsH {
        gamma: register_passthrough(res, &w.gamma)?,
    })
}

fn reg_resnet<S: WeightSource>(
    res: &WeightResidency<S>,
    w: &ResnetW,
) -> Result<ResnetH, LoadError> {
    Ok(ResnetH {
        norm1: reg_rms(res, &w.norm1)?,
        conv1: reg_conv(res, &w.conv1)?,
        norm2: reg_rms(res, &w.norm2)?,
        conv2: reg_conv(res, &w.conv2)?,
    })
}

fn reg_attn<S: WeightSource>(res: &WeightResidency<S>, w: &AttnW) -> Result<AttnH, LoadError> {
    Ok(AttnH {
        norm: reg_rms(res, &w.norm)?,
        q: reg_conv(res, &w.q)?,
        k: reg_conv(res, &w.k)?,
        v: reg_conv(res, &w.v)?,
        proj: reg_conv(res, &w.proj)?,
    })
}

impl DecoderH {
    fn register<S: WeightSource>(
        res: &WeightResidency<S>,
        w: &DecoderW,
    ) -> Result<Self, LoadError> {
        let mut up = Vec::with_capacity(w.up.len());
        for st in &w.up {
            up.push(UpStageH {
                resnets: st
                    .resnets
                    .iter()
                    .map(|r| reg_resnet(res, r))
                    .collect::<Result<_, _>>()?,
                upsample_conv: st
                    .upsample_conv
                    .as_ref()
                    .map(|c| reg_conv(res, c))
                    .transpose()?,
            });
        }
        Ok(Self {
            conv_in: reg_conv(res, &w.conv_in)?,
            mid: MidH {
                block_1: reg_resnet(res, &w.mid.block_1)?,
                attn_1: reg_attn(res, &w.mid.attn_1)?,
                block_2: reg_resnet(res, &w.mid.block_2)?,
            },
            up,
            norm_out: reg_rms(res, &w.norm_out)?,
            conv_out: reg_conv(res, &w.conv_out)?,
        })
    }
}

// Post-acquire GPU buffers (Copy; bound into the scope).
#[derive(Clone, Copy)]
struct ConvBufs {
    weight: BufRef,
    bias: BufRef,
}

#[derive(Clone, Copy)]
struct RmsBufs {
    gamma: BufRef,
}

#[derive(Clone, Copy)]
struct ResnetBufs {
    norm1: RmsBufs,
    conv1: ConvBufs,
    norm2: RmsBufs,
    conv2: ConvBufs,
}

#[derive(Clone, Copy)]
struct AttnBufs {
    norm: RmsBufs,
    q: ConvBufs,
    k: ConvBufs,
    v: ConvBufs,
    proj: ConvBufs,
}

struct MidBufs {
    block_1: ResnetBufs,
    attn_1: AttnBufs,
    block_2: ResnetBufs,
}

struct UpStageBufs {
    resnets: Vec<ResnetBufs>,
    upsample_conv: Option<ConvBufs>,
}

struct DecoderBufs {
    conv_in: ConvBufs,
    mid: MidBufs,
    up: Vec<UpStageBufs>,
    norm_out: RmsBufs,
    conv_out: ConvBufs,
}

// ============================================================================
// Pipelines
// ============================================================================

pub struct HunyuanVaePipelines {
    act: ActDtype,
    conv3d: Conv3dF32,
    conv3d_pipeline: thinfer_core::backend::WgpuPipeline,
    rmsnorm3d: thinfer_core::backend::WgpuPipeline,
    silu: thinfer_core::backend::WgpuPipeline,
    add: thinfer_core::backend::WgpuPipeline,
    /// Causal mid-block SDPA. `SdpaF32LargeDCausal` is f32-only, and the mid
    /// block runs whole-tensor at latent resolution where f32 is cheap, so this
    /// is built ONLY for an F32 act set. An F16 set (the conv-heavy up-stages,
    /// which have no attention) leaves it `None`.
    sdpa_causal: Option<thinfer_core::backend::WgpuPipeline>,
    transpose12: thinfer_core::backend::WgpuPipeline,
    upsample: thinfer_core::backend::WgpuPipeline,
}

impl HunyuanVaePipelines {
    fn act_size(&self) -> u32 {
        match self.act {
            ActDtype::F32 => 4,
            ActDtype::F16 => 2,
            other => unreachable!("hunyuan vae acts are f32 or f16, got {other:?}"),
        }
    }

    /// Compile with the parity-default act dtype (`f32` exact; `f16` = production
    /// perf, validated to a band). Parity pins `f32`.
    pub async fn compile_with(backend: &WgpuBackend, act: ActDtype) -> Result<Self, WgpuError> {
        let cfg = &WgslConfig {
            bf16_quant_writes: false,
            act_dtype: act,
            weight_dtype: WeightDtype::Bf16,
        };
        let conv3d = Conv3dF32::new(Conv3dConfig::DEFAULT);
        Ok(Self {
            act,
            conv3d_pipeline: backend
                .create_pipeline(
                    "hunyuan_vae_conv3d",
                    &conv3d.wgsl(cfg),
                    "main",
                    <Conv3dF32 as Conv3dOp>::layout(),
                )
                .await?,
            conv3d,
            rmsnorm3d: backend
                .create_pipeline(
                    "hunyuan_vae_rmsnorm3d",
                    &<RmsNorm3dF32 as RmsNorm3dOp>::wgsl(cfg),
                    "main",
                    <RmsNorm3dF32 as RmsNorm3dOp>::layout(),
                )
                .await?,
            silu: backend
                .create_pipeline(
                    "hunyuan_vae_silu",
                    SiluF32::wgsl(cfg),
                    "main",
                    SiluF32::layout(),
                )
                .await?,
            add: backend
                .create_pipeline(
                    "hunyuan_vae_add",
                    AddF32::wgsl(cfg),
                    "main",
                    AddF32::layout(),
                )
                .await?,
            sdpa_causal: if matches!(act, ActDtype::F32) {
                Some(
                    backend
                        .create_pipeline(
                            "hunyuan_vae_sdpa_causal",
                            <SdpaF32LargeDCausal as SdpaOp>::wgsl(cfg),
                            "main",
                            <SdpaF32LargeDCausal as SdpaOp>::layout(),
                        )
                        .await?,
                )
            } else {
                None
            },
            transpose12: backend
                .create_pipeline(
                    "hunyuan_vae_transpose12",
                    <Transpose12F32 as Transpose12Op>::wgsl(cfg),
                    "main",
                    <Transpose12F32 as Transpose12Op>::layout(),
                )
                .await?,
            upsample: backend
                .create_pipeline(
                    "hunyuan_vae_upsample3d",
                    HunyuanUpsample3dF32::wgsl(cfg),
                    "main",
                    <HunyuanUpsample3dF32 as HunyuanUpsample3dOp>::layout(),
                )
                .await?,
        })
    }
}

// ============================================================================
// Shapes / uniforms
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape {
    c: u32,
    t: u32,
    h: u32,
    w: u32,
}

impl Shape {
    fn elems(&self) -> u32 {
        self.c * self.t * self.h * self.w
    }
    fn bytes(&self, act_size: u32) -> u64 {
        self.elems() as u64 * act_size as u64
    }
    /// `T*H*W` (rmsnorm position count and channel stride, B=1).
    fn thw(&self) -> u32 {
        self.t * self.h * self.w
    }
}

/// Conv3d uniform (20 u32). `pad_mode` (index 18): 0 = zero-fill, 1 =
/// replicate-edge (causal `F.pad(mode='replicate')`).
#[allow(clippy::too_many_arguments)]
fn conv3d_uniform_bytes(
    cin: u32,
    cout: u32,
    t_in: u32,
    h_in: u32,
    w_in: u32,
    t_out: u32,
    h_out: u32,
    w_out: u32,
    ker: (u32, u32, u32),
    pad: (u32, u32, u32),
    pad_mode: u32,
) -> [u8; 80] {
    let fields: [u32; 20] = [
        1, cin, cout, t_in, h_in, w_in, t_out, h_out, w_out, ker.0, ker.1, ker.2, pad.0, pad.1,
        pad.2, 1, 1, 1, pad_mode, 0,
    ];
    let mut bytes = [0u8; 80];
    for (i, v) in fields.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn rmsnorm3d_uniform_bytes(n_pos: u32, channels: u32, stride: u32) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&n_pos.to_le_bytes());
    bytes[4..8].copy_from_slice(&channels.to_le_bytes());
    bytes[8..12].copy_from_slice(&stride.to_le_bytes());
    bytes
}

/// Causal large-D SDPA uniform (12 u32 = 48 bytes, 16-aligned). `has_mask` is
/// unused by the causal kernel (the mask binding is a scratch); `period` =
/// tokens per latent frame (`h*w`) for the on-the-fly causal prefix cutoff;
/// `row_off` = the global first query row of this dispatch (q/out are bound as
/// subviews from `row_off`, so a chunked query range still clamps causality by
/// the GLOBAL frame).
fn sdpa_causal_uniform_bytes(
    s_q: u32,
    s_k: u32,
    d: u32,
    scale: f32,
    period: u32,
    row_off: u32,
) -> [u8; 48] {
    let fields_u32: [(usize, u32); 6] = [
        (0, 1),    // b
        (4, 1),    // h_q
        (8, 1),    // h_kv
        (12, s_q), // s_q
        (16, s_k), // s_k
        (20, d),   // d
    ];
    let mut bytes = [0u8; 48];
    for (off, v) in fields_u32 {
        bytes[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
    bytes[24..28].copy_from_slice(&scale.to_le_bytes());
    // bytes[28..32] = has_mask (0, unused); bytes[32..36] = period;
    // bytes[36..40] = row_off.
    bytes[32..36].copy_from_slice(&period.to_le_bytes());
    bytes[36..40].copy_from_slice(&row_off.to_le_bytes());
    bytes
}

/// Query-row chunk size for the causal mid-attention. The whole-tensor mid-attn
/// at real f (e.g. 480p latent = 32760 tokens, D=1024) is one multi-second
/// dispatch that trips the 2s GPU watchdog (TDR -> device lost). Splitting the
/// query range into per-submit chunks (each flushed) keeps every dispatch under
/// the watchdog; the per-chunk `row_off` preserves exact causal cutoffs. Target
/// ~32 chunks; env-overridable (`THINFER_VAE_MID_CHUNK_ROWS`) to force many small
/// chunks for the f=2 parity test.
fn vae_mid_chunk_rows(n: u32) -> u32 {
    use std::sync::OnceLock;
    static V: OnceLock<Option<u32>> = OnceLock::new();
    let override_rows = *V.get_or_init(|| {
        std::env::var("THINFER_VAE_MID_CHUNK_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&r| r > 0)
    });
    override_rows.unwrap_or_else(|| n.div_ceil(32).max(1))
}

// ============================================================================
// Op wrappers (all run inside one BatchScope)
// ============================================================================

/// A conv3d with explicit geometry. `pad_mode=1` = replicate-edge (causal). For
/// a kt=3 causal conv pass `pad=(2,1,1)` so `t_out == t_in` (front-pad 2, none
/// back). 1x1x1 convs pass `pad=(0,0,0)`, `pad_mode` irrelevant.
#[allow(clippy::too_many_arguments)]
fn conv3d<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &HunyuanVaePipelines,
    x: BatchBuf<'w>,
    s: Shape,
    w: &ConvBufs,
    cout: u32,
    ker: (u32, u32, u32),
    pad: (u32, u32, u32),
    pad_mode: u32,
) -> Result<(BatchBuf<'w>, Shape), WgpuError> {
    let (kt, kh, kw) = ker;
    let (pt, ph, pw) = pad;
    let t_out = s.t + pt - kt + 1;
    let h_out = s.h + 2 * ph - kh + 1;
    let w_out = s.w + 2 * pw - kw + 1;
    let out_shape = Shape {
        c: cout,
        t: t_out,
        h: h_out,
        w: w_out,
    };
    let out = scope.alloc(out_shape.bytes(pl.act_size()))?;
    let u = scope.write_uniform(&conv3d_uniform_bytes(
        s.c, cout, s.t, s.h, s.w, t_out, h_out, w_out, ker, pad, pad_mode,
    ))?;
    let wb = scope.import_copy(w.weight);
    let bb = scope.import_copy(w.bias);
    scope.conv3d(
        &pl.conv3d_pipeline,
        &pl.conv3d,
        x,
        wb,
        bb,
        u,
        out,
        cout,
        out_shape.thw(),
        1,
    )?;
    Ok((out, out_shape))
}

/// kt=3 causal conv (replicate front-pad time + symmetric H/W). Preserves dims.
fn conv3d_k3<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &HunyuanVaePipelines,
    x: BatchBuf<'w>,
    s: Shape,
    w: &ConvBufs,
    cout: u32,
) -> Result<(BatchBuf<'w>, Shape), WgpuError> {
    conv3d(scope, pl, x, s, w, cout, (3, 3, 3), (2, 1, 1), 1)
}

/// 1x1x1 pointwise conv (attn q/k/v/proj). No padding.
fn conv3d_1x1x1<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &HunyuanVaePipelines,
    x: BatchBuf<'w>,
    s: Shape,
    w: &ConvBufs,
    cout: u32,
) -> Result<(BatchBuf<'w>, Shape), WgpuError> {
    conv3d(scope, pl, x, s, w, cout, (1, 1, 1), (0, 0, 0), 0)
}

fn rmsnorm3d<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &HunyuanVaePipelines,
    x: BatchBuf<'w>,
    s: Shape,
    w: &RmsBufs,
) -> Result<BatchBuf<'w>, WgpuError> {
    let out = scope.alloc(s.bytes(pl.act_size()))?;
    let u = scope.write_uniform(&rmsnorm3d_uniform_bytes(s.thw(), s.c, s.thw()))?;
    let gamma = scope.import_copy(w.gamma);
    scope.rmsnorm3d::<RmsNorm3dF32>(&pl.rmsnorm3d, x, gamma, u, out, s.thw())?;
    Ok(out)
}

fn silu<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &HunyuanVaePipelines,
    x: BatchBuf<'w>,
    s: Shape,
) -> Result<BatchBuf<'w>, WgpuError> {
    let out = scope.alloc(s.bytes(pl.act_size()))?;
    scope.dispatch_op::<SiluF32>(&pl.silu, &[x], out)?;
    Ok(out)
}

fn add<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &HunyuanVaePipelines,
    a: BatchBuf<'w>,
    b: BatchBuf<'w>,
    s: Shape,
) -> Result<BatchBuf<'w>, WgpuError> {
    let out = scope.alloc(s.bytes(pl.act_size()))?;
    scope.dispatch_op::<AddF32>(&pl.add, &[a, b], out)?;
    Ok(out)
}

fn transpose12<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &HunyuanVaePipelines,
    x: BatchBuf<'w>,
    d0: u32,
    d1: u32,
    d2: u32,
    d3: u32,
) -> Result<BatchBuf<'w>, WgpuError> {
    let out = scope.alloc((d0 * d1 * d2 * d3) as u64 * pl.act_size() as u64)?;
    let u = scope.u32x4_uniform(d0, d1, d2, d3)?;
    scope.transpose12::<Transpose12F32>(&pl.transpose12, x, u, out, d0 * d1 * d2 * d3)?;
    Ok(out)
}

/// ResnetBlock (channel-preserving): norm1 -> silu -> conv1 -> norm2 -> silu ->
/// conv2 + x.
fn resnet<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &HunyuanVaePipelines,
    x: BatchBuf<'w>,
    s: Shape,
    w: &ResnetBufs,
) -> Result<BatchBuf<'w>, WgpuError> {
    let n1 = rmsnorm3d(scope, pl, x, s, &w.norm1)?;
    let a1 = silu(scope, pl, n1, s)?;
    let (c1, _) = conv3d_k3(scope, pl, a1, s, &w.conv1, s.c)?;
    let n2 = rmsnorm3d(scope, pl, c1, s, &w.norm2)?;
    let a2 = silu(scope, pl, n2, s)?;
    let (c2, _) = conv3d_k3(scope, pl, a2, s, &w.conv2, s.c)?;
    add(scope, pl, x, c2, s)
}

/// Causal spatio-temporal mid-attention (`AttnBlock`). q/k/v 1x1x1 ->
/// single-head sdpa over (f*h*w) tokens (D=C) -> proj_out + x. Causality (frame
/// i attends frames 0..=i) is enforced on the fly by the causal large-D kernel
/// (key-loop clamped to `(query_frame+1)*period`), so there is NO materialized
/// `[N, N]` mask: `period = h*w` tokens per frame. `mask_scratch` is a 1-element
/// dummy for the unused binding slot 3.
fn mid_attention<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &HunyuanVaePipelines,
    x: BatchBuf<'w>,
    s: Shape,
    w: &AttnBufs,
    mask_scratch: BatchBuf<'w>,
) -> Result<BatchBuf<'w>, WgpuError> {
    let c = s.c;
    let n = s.thw(); // tokens
    let asz = pl.act_size() as u64;

    let normed = rmsnorm3d(scope, pl, x, s, &w.norm)?;
    let (q, _) = conv3d_1x1x1(scope, pl, normed, s, &w.q, c)?;
    let (k, _) = conv3d_1x1x1(scope, pl, normed, s, &w.k, c)?;
    let (v, _) = conv3d_1x1x1(scope, pl, normed, s, &w.v, c)?;

    // [1, C, N] -> [1, N, C] tokens (single head, head_dim = C).
    let qt = transpose12(scope, pl, q, 1, c, n, 1)?;
    let kt = transpose12(scope, pl, k, 1, c, n, 1)?;
    let vt = transpose12(scope, pl, v, 1, c, n, 1)?;

    let attn = scope.alloc((c * n) as u64 * asz)?;
    let scale = 1.0_f32 / (c as f32).sqrt();
    let period = s.h * s.w;
    // Chunk the query range so no single causal dispatch trips the 2s GPU
    // watchdog at real f (whole-tensor is one multi-second dispatch -> TDR).
    // Each chunk binds q/out as a row-subview from its global `row_off` (so the
    // causal frame cutoff stays exact) and flushes to land in its own submit.
    let row_bytes = c as u64 * asz;
    let chunk = vae_mid_chunk_rows(n);
    let mut r0 = 0u32;
    while r0 < n {
        let cr = (n - r0).min(chunk);
        let off = r0 as u64 * row_bytes;
        let len = cr as u64 * row_bytes;
        let q_c = scope.subview(&qt, off, len);
        let o_c = scope.subview(&attn, off, len);
        let u = scope.write_uniform(&sdpa_causal_uniform_bytes(cr, n, c, scale, period, r0))?;
        let sdpa_causal = pl
            .sdpa_causal
            .as_ref()
            .expect("mid_attention requires the F32 mid pipeline set");
        scope.sdpa::<SdpaF32LargeDCausal>(
            sdpa_causal,
            q_c,
            kt,
            vt,
            mask_scratch,
            u,
            o_c,
            1,
            cr,
            1,
        )?;
        r0 += cr;
        if chunk < n {
            scope.flush()?;
        }
    }

    // [1, N, C] -> [1, C, N] -> proj_out (1x1x1) + residual.
    let restored = transpose12(scope, pl, attn, 1, n, c, 1)?;
    let (proj, _) = conv3d_1x1x1(scope, pl, restored, s, &w.proj, c)?;
    add(scope, pl, x, proj, s)
}

fn upsample_uniform_bytes(in_c: u32, out_c: u32, temporal: u32, s: Shape, r: u32) -> [u8; 32] {
    let fields: [u32; 8] = [in_c, out_c, temporal, s.t, s.h, s.w, r, 0];
    let mut bytes = [0u8; 32];
    for (i, v) in fields.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Hunyuan `Upsample`: causal k3 conv (`in_c -> out_c*factor`) then the fused
/// pixelshuffle + repeat_interleave residual (`HunyuanUpsample3d`). `temporal`
/// = 8x channel split with first-frame handling (F -> 2F-1); else 4x spatial
/// (F unchanged). H/W double. `in_c = s.c`.
fn upsample<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &HunyuanVaePipelines,
    x: BatchBuf<'w>,
    s: Shape,
    conv: &ConvBufs,
    out_c: u32,
    temporal: bool,
) -> Result<(BatchBuf<'w>, Shape), WgpuError> {
    let in_c = s.c;
    let factor = if temporal { 8 } else { 4 };
    let (h, _) = conv3d_k3(scope, pl, x, s, conv, out_c * factor)?;
    let out_shape = Shape {
        c: out_c,
        t: if temporal { 2 * s.t - 1 } else { s.t },
        h: s.h * 2,
        w: s.w * 2,
    };
    let out = scope.alloc(out_shape.bytes(pl.act_size()))?;
    let r = 4 * out_c / in_c;
    let u = scope.write_uniform(&upsample_uniform_bytes(in_c, out_c, temporal as u32, s, r))?;
    scope.hunyuan_upsample3d::<HunyuanUpsample3dF32>(
        &pl.upsample,
        h,
        x,
        u,
        out,
        out_shape.elems(),
    )?;
    Ok((out, out_shape))
}

/// Run the decoder up-stages (`RES_PER_STAGE` resnets + optional upsampler each)
/// then `norm_out -> silu -> conv_out`, returning the in-scope `conv_out` buffer
/// `[3, F, H, W]` + shape. Pure local convs/upsamples -> tileable. When `up_taps`
/// is given, each up-stage output `(buf, shape)` is pushed (caller persists them
/// before submit, for parity bisection). Input `x` is the post-mid latent
/// `[MID_CHANNELS, f, h, w]` (whole tensor or one tile).
fn run_upstages<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &HunyuanVaePipelines,
    bufs: &DecoderBufs,
    mut x: BatchBuf<'w>,
    mut s: Shape,
    mut up_taps: Option<&mut Vec<(BatchBuf<'w>, Shape)>>,
) -> Result<(BatchBuf<'w>, Shape), WgpuError> {
    for (i, st) in UP_STAGES.iter().enumerate() {
        debug_assert_eq!(s.c as usize, st.ch, "up stage {i} input channels");
        let ub = &bufs.up[i];
        for rb in &ub.resnets {
            x = resnet(scope, pl, x, s, rb)?;
        }
        if let Some((out_c, temporal)) = st.upsample {
            let conv = ub.upsample_conv.as_ref().expect("upsampler conv");
            let (ux, us) = upsample(scope, pl, x, s, conv, out_c as u32, temporal)?;
            x = ux;
            s = us;
        }
        if let Some(t) = up_taps.as_deref_mut() {
            t.push((x, s));
        }
    }
    let n = rmsnorm3d(scope, pl, x, s, &bufs.norm_out)?;
    let a = silu(scope, pl, n, s)?;
    conv3d_k3(scope, pl, a, s, &bufs.conv_out, OUT_CHANNELS as u32)
}

// ============================================================================
// Decoder
// ============================================================================

pub struct HunyuanVaeDecoder {
    /// Mid-block + parity pipeline set. MUST be F32 (the causal mid-attn is
    /// f32-only). `decode_mid` (whole-tensor, latent resolution = cheap) and the
    /// tap/parity decode paths use this.
    pub pipelines: HunyuanVaePipelines,
    /// Optional dedicated up-stage pipeline set. The conv-heavy up-stages (~87%
    /// of decode, no attention) run through this when present; an F16 set roughly
    /// halves conv bandwidth. `None` = run the up-stages on `pipelines` (the
    /// all-F32 parity path).
    up_pipelines: Option<HunyuanVaePipelines>,
    handles: DecoderH,
}

#[derive(Debug)]
pub enum HunyuanVaeError<SE: core::fmt::Debug> {
    Wgpu(WgpuError),
    Load(LoadError),
    Residency(ResidencyError<SE, WgpuError>),
}
impl<SE: core::fmt::Debug> From<WgpuError> for HunyuanVaeError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}
impl<SE: core::fmt::Debug> From<LoadError> for HunyuanVaeError<SE> {
    fn from(e: LoadError) -> Self {
        Self::Load(e)
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for HunyuanVaeError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

/// Per-stage taps for parity bisection (NCTHW f32, row-major).
#[derive(Default)]
pub struct HunyuanVaeTaps<'a> {
    pub conv_in: Option<&'a mut Vec<f32>>,
    pub mid_block1: Option<&'a mut Vec<f32>>,
    pub mid_attn: Option<&'a mut Vec<f32>>,
    pub mid: Option<&'a mut Vec<f32>>,
    /// One sink per up stage output (`len == UP_STAGES.len()` once filled).
    pub up: Option<&'a mut Vec<Vec<f32>>>,
    /// `conv_out` output `[3, F, Hout, Wout]` (== video, raw [-1,1]).
    pub conv_out: Option<&'a mut Vec<f32>>,
}

impl HunyuanVaeDecoder {
    pub fn new<S: WeightSource>(
        pipelines: HunyuanVaePipelines,
        residency: &WeightResidency<S>,
    ) -> Result<Self, LoadError> {
        Self::new_mixed(pipelines, None, residency)
    }

    /// Build with a distinct up-stage pipeline set (e.g. F16 up-stages over an
    /// F32 mid). `up` MUST be `None` or carry the SAME weights (only the act
    /// dtype differs); the mid set stays F32 for the causal attn. The driver uses
    /// `Some(f16)` for perf; parity passes `None` (all-F32).
    pub fn new_mixed<S: WeightSource>(
        pipelines: HunyuanVaePipelines,
        up_pipelines: Option<HunyuanVaePipelines>,
        residency: &WeightResidency<S>,
    ) -> Result<Self, LoadError> {
        let weights = DecoderW::new();
        let handles = DecoderH::register(residency, &weights)?;
        Ok(Self {
            pipelines,
            up_pipelines,
            handles,
        })
    }

    /// The pipeline set for the conv-heavy up-stages (the F16 set when mixed,
    /// else the F32 mid set).
    fn up_pl(&self) -> &HunyuanVaePipelines {
        self.up_pipelines.as_ref().unwrap_or(&self.pipelines)
    }

    /// Decode `latents` (CTHW row-major, `32*f*h*w` f32, NORMALIZED = engine
    /// input) into the raw video `[3, 4*(f-1)+1, 16*h, 16*w]` f32 ([-1,1]; the
    /// pipeline maps to [0,1] via `*0.5+0.5`). Pre-scales by `1/SCALING_FACTOR`.
    /// Fills `taps` (parity bisection). Single whole-tensor submit (tiny parity
    /// dims; production up-stage tiling is a follow-up). Mid-attn causality is
    /// enforced on the fly (no materialized mask), so any `f` is valid here.
    pub async fn decode_with_taps<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        latents: &[f32],
        f: usize,
        h: usize,
        w: usize,
        mut taps: Option<&mut HunyuanVaeTaps<'_>>,
    ) -> Result<Vec<f32>, HunyuanVaeError<S::Error>> {
        assert_eq!(latents.len(), LATENT_CHANNELS * f * h * w, "latent size");
        let act = self.pipelines.act;

        // Host pre-scale + build the repeat_interleave residual `z.repeat_interleave
        // (REPEAT)` (out channel c <- z[c / REPEAT]).
        let thw = f * h * w;
        let mut z = vec![0.0f32; latents.len()];
        for (i, &v) in latents.iter().enumerate() {
            z[i] = v / SCALING_FACTOR;
        }
        let mut z_rep = vec![0.0f32; MID_CHANNELS * thw];
        for c in 0..MID_CHANNELS {
            let src = (c / REPEAT) * thw;
            z_rep[c * thw..c * thw + thw].copy_from_slice(&z[src..src + thw]);
        }

        let mut pins: Vec<GpuView> = Vec::new();
        let bufs = self.acquire(residency, backend, &mut pins).await?;

        let z_bytes = act_upload_bytes(act, &z);
        let z_in = workspace.alloc(z_bytes.len() as u64)?;
        backend.write_buffer(z_in.id(), 0, &z_bytes)?;
        let rep_bytes = act_upload_bytes(act, &z_rep);
        let rep_in = workspace.alloc(rep_bytes.len() as u64)?;
        backend.write_buffer(rep_in.id(), 0, &rep_bytes)?;

        let act_size = self.pipelines.act_size();

        // Causality is enforced on the fly inside the mid-attn kernel (no
        // materialized [N, N] mask). A 1-element scratch fills the unused
        // mask binding slot.
        let mask_scratch = workspace.alloc(act_size as u64)?;

        let mut p_conv_in: Option<(WsBuf<WgpuBackend>, Shape)> = None;
        let mut p_block1: Option<(WsBuf<WgpuBackend>, Shape)> = None;
        let mut p_attn: Option<(WsBuf<WgpuBackend>, Shape)> = None;
        let mut p_mid: Option<(WsBuf<WgpuBackend>, Shape)> = None;
        let mut p_up: Vec<(WsBuf<WgpuBackend>, Shape)> = Vec::new();
        let conv_out_persist;
        let conv_out_shape;
        {
            let scope = workspace.batch();
            let pl = &self.pipelines;
            let s_in = Shape {
                c: LATENT_CHANNELS as u32,
                t: f as u32,
                h: h as u32,
                w: w as u32,
            };

            // conv_in (k3 causal, 32 -> 1024) + repeat_interleave residual.
            let zb = scope.import_copy(z_in.as_buf_ref());
            let (ci, cs) = conv3d_k3(&scope, pl, zb, s_in, &bufs.conv_in, MID_CHANNELS as u32)?;
            let repb = scope.import_copy(rep_in.as_buf_ref());
            let mut x = add(&scope, pl, ci, repb, cs)?;
            let s = cs;
            let want = |f: fn(&HunyuanVaeTaps) -> bool| taps.as_ref().is_some_and(|t| f(t));
            if want(|t| t.conv_in.is_some()) {
                p_conv_in = Some((persist(&scope, workspace, x, s, act_size)?, s));
            }

            // mid: resnet -> causal attn -> resnet.
            x = resnet(&scope, pl, x, s, &bufs.mid.block_1)?;
            if want(|t| t.mid_block1.is_some()) {
                p_block1 = Some((persist(&scope, workspace, x, s, act_size)?, s));
            }
            let mask = scope.import_copy(mask_scratch.as_buf_ref());
            x = mid_attention(&scope, pl, x, s, &bufs.mid.attn_1, mask)?;
            if want(|t| t.mid_attn.is_some()) {
                p_attn = Some((persist(&scope, workspace, x, s, act_size)?, s));
            }
            x = resnet(&scope, pl, x, s, &bufs.mid.block_2)?;
            if want(|t| t.mid.is_some()) {
                p_mid = Some((persist(&scope, workspace, x, s, act_size)?, s));
            }

            // up stages + norm_out + conv_out (shared with the tiled decode).
            let want_up = want(|t| t.up.is_some());
            let mut up_tap_bufs: Vec<(BatchBuf, Shape)> = Vec::new();
            let (co, cos) =
                run_upstages(&scope, pl, &bufs, x, s, want_up.then_some(&mut up_tap_bufs))?;
            if want_up {
                for (b, sh) in &up_tap_bufs {
                    p_up.push((persist(&scope, workspace, *b, *sh, act_size)?, *sh));
                }
            }
            conv_out_shape = cos;
            conv_out_persist = persist(&scope, workspace, co, cos, act_size)?;
            scope.submit_void().await?;
        }

        if let Some(t) = taps.as_mut() {
            for (p, sink) in [
                (&p_conv_in, t.conv_in.as_deref_mut()),
                (&p_block1, t.mid_block1.as_deref_mut()),
                (&p_attn, t.mid_attn.as_deref_mut()),
                (&p_mid, t.mid.as_deref_mut()),
            ] {
                if let (Some((ws, sh)), Some(sink)) = (p, sink) {
                    *sink = read_acts(backend, &ws.as_buf_ref(), sh.elems() as usize, act).await?;
                }
            }
            if let Some(sink) = t.up.as_deref_mut() {
                sink.clear();
                for (ws, sh) in &p_up {
                    sink.push(
                        read_acts(backend, &ws.as_buf_ref(), sh.elems() as usize, act).await?,
                    );
                }
            }
        }
        let video = read_acts(
            backend,
            &conv_out_persist.as_buf_ref(),
            conv_out_shape.elems() as usize,
            act,
        )
        .await?;
        if let Some(sink) = taps.as_mut().and_then(|t| t.conv_out.as_deref_mut()) {
            *sink = video.clone();
        }
        Ok(video)
    }

    /// Production decode (no taps): whole-tensor `conv_in + mid` (the causal
    /// mid-attn needs the global receptive field, so this phase is NOT tiled),
    /// then the up-stages + `norm_out` + `conv_out` decoded in overlapping
    /// spatial + temporal tiles and blended. Returns the raw video
    /// `[3, 4*(f-1)+1, 16*h, 16*w]` f32 ([-1,1]). A single tile (latent fits the
    /// budget) is bit-identical to `decode_with_taps` (the parity path).
    pub async fn decode<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        latents: &[f32],
        f: usize,
        h: usize,
        w: usize,
    ) -> Result<Vec<f32>, HunyuanVaeError<S::Error>> {
        assert_eq!(latents.len(), LATENT_CHANNELS * f * h * w, "latent size");

        // Conv weights acquired once; pins held across phase 1 + every up tile.
        let mut pins: Vec<GpuView> = Vec::new();
        let bufs = self.acquire(residency, backend, &mut pins).await?;

        // Phase 1: whole-tensor conv_in + mid -> post-mid latent [1024, f, h, w].
        let post_mid = self
            .decode_mid(backend, workspace, &bufs, latents, f, h, w)
            .await?;
        workspace.drain_pool();

        // Phase 2: tiled up-stages. Budget -> tile plan; strict-budget so an
        // over-budget tile fails AT the budget (the tiler reshrinks) instead of
        // a device OOM. Mirrors the LTX VAE / Wan step-loop residency mechanism.
        let budget = residency.arbiter().budget_bytes();
        let eff_budget = if budget == u64::MAX {
            DEFAULT_REF_BUDGET
        } else {
            budget
        };
        let weight_footprint = weight_footprint_bytes();
        let workspace_budget = eff_budget.saturating_sub(weight_footprint + VAE_STAGING_RESERVE);
        if budget != u64::MAX {
            residency.set_transient_reserve(eff_budget.saturating_sub(weight_footprint));
            workspace.set_strict_budget(true);
        }
        // Up-stage tiles run on the up pipeline set (F16 when mixed), so size the
        // tile budget by ITS act bytes (f16 = half -> larger tiles fit).
        let act_size = self.up_pl().act_size();
        let mut seed_budget = (workspace_budget as f64 * SEED_SAFETY) as u64;
        let (mut tf, mut tile) = initial_tiles(seed_budget, f, act_size);

        loop {
            tracing::info!(
                target: thinfer_core::trace::DIAG,
                acts = ?self.up_pl().act, f, h, w,
                seed_budget_mb = seed_budget / (1024 * 1024),
                tf, tile, "hunyuan vae up-stage decode attempt",
            );
            match self
                .decode_upstages_tiled(backend, workspace, &bufs, &post_mid, f, h, w, tf, tile)
                .await
            {
                Ok(v) => return Ok(v),
                Err(WgpuError::Allocate { .. } | WgpuError::BudgetExceeded { .. })
                    if tile > TILE_MIN || tf as u32 > TEMPORAL_TILE_MIN =>
                {
                    workspace.drain_pool();
                    let prev = (tf, tile);
                    loop {
                        seed_budget = ((seed_budget as f64 * OOM_SHRINK) as u64).max(1);
                        let (ntf, ntile) = initial_tiles(seed_budget, f, act_size);
                        tf = ntf;
                        tile = ntile;
                        let floored = tile == TILE_MIN && tf as u32 == TEMPORAL_TILE_MIN;
                        if (tf, tile) != prev || floored {
                            break;
                        }
                    }
                    tracing::warn!(
                        target: thinfer_core::trace::DIAG,
                        from_tf = prev.0, from_tile = prev.1, to_tf = tf, to_tile = tile,
                        "hunyuan vae up-stage OOM; re-seeding smaller",
                    );
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    /// Decode with an EXPLICIT up-stage tile plan (`tf` latent frames, `tile`
    /// latent cells), bypassing budget seeding. Phase 1 is still whole-tensor.
    /// Used by the tiled-vs-whole parity test to force a deterministic multi-tile
    /// case, and by any caller that wants a fixed plan instead of budget-driven.
    #[allow(clippy::too_many_arguments)]
    pub async fn decode_with_tiles<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        latents: &[f32],
        f: usize,
        h: usize,
        w: usize,
        tf: usize,
        tile: u32,
    ) -> Result<Vec<f32>, HunyuanVaeError<S::Error>> {
        assert_eq!(latents.len(), LATENT_CHANNELS * f * h * w, "latent size");
        let mut pins: Vec<GpuView> = Vec::new();
        let bufs = self.acquire(residency, backend, &mut pins).await?;
        let post_mid = self
            .decode_mid(backend, workspace, &bufs, latents, f, h, w)
            .await?;
        workspace.drain_pool();
        Ok(self
            .decode_upstages_tiled(backend, workspace, &bufs, &post_mid, f, h, w, tf, tile)
            .await?)
    }

    /// Phase 1: pre-scale + `repeat_interleave` residual + `conv_in` + mid
    /// (resnet -> causal attn -> resnet), whole tensor, one submit. Returns the
    /// post-mid latent `[MID_CHANNELS, f, h, w]` host f32.
    async fn decode_mid(
        &self,
        backend: &WgpuBackend,
        workspace: &Workspace<WgpuBackend>,
        bufs: &DecoderBufs,
        latents: &[f32],
        f: usize,
        h: usize,
        w: usize,
    ) -> Result<Vec<f32>, WgpuError> {
        use thinfer_core::backend::Backend;
        let act = self.pipelines.act;
        let act_size = self.pipelines.act_size();
        let thw = f * h * w;

        let mut z = vec![0.0f32; latents.len()];
        for (i, &v) in latents.iter().enumerate() {
            z[i] = v / SCALING_FACTOR;
        }
        let mut z_rep = vec![0.0f32; MID_CHANNELS * thw];
        for c in 0..MID_CHANNELS {
            let src = (c / REPEAT) * thw;
            z_rep[c * thw..c * thw + thw].copy_from_slice(&z[src..src + thw]);
        }

        let z_bytes = act_upload_bytes(act, &z);
        let z_in = workspace.alloc(z_bytes.len() as u64)?;
        backend.write_buffer(z_in.id(), 0, &z_bytes)?;
        let rep_bytes = act_upload_bytes(act, &z_rep);
        let rep_in = workspace.alloc(rep_bytes.len() as u64)?;
        backend.write_buffer(rep_in.id(), 0, &rep_bytes)?;
        let mask_scratch = workspace.alloc(act_size as u64)?;

        let post_mid_persist;
        let post_mid_shape;
        {
            let scope = workspace.batch();
            let pl = &self.pipelines;
            let s_in = Shape {
                c: LATENT_CHANNELS as u32,
                t: f as u32,
                h: h as u32,
                w: w as u32,
            };
            let zb = scope.import_copy(z_in.as_buf_ref());
            let (ci, cs) = conv3d_k3(&scope, pl, zb, s_in, &bufs.conv_in, MID_CHANNELS as u32)?;
            let repb = scope.import_copy(rep_in.as_buf_ref());
            let mut x = add(&scope, pl, ci, repb, cs)?;
            let s = cs;
            x = resnet(&scope, pl, x, s, &bufs.mid.block_1)?;
            let mask = scope.import_copy(mask_scratch.as_buf_ref());
            x = mid_attention(&scope, pl, x, s, &bufs.mid.attn_1, mask)?;
            x = resnet(&scope, pl, x, s, &bufs.mid.block_2)?;
            post_mid_shape = s;
            post_mid_persist = persist(&scope, workspace, x, s, act_size)?;
            scope.submit_void().await?;
        }
        read_acts(
            backend,
            &post_mid_persist.as_buf_ref(),
            post_mid_shape.elems() as usize,
            act,
        )
        .await
    }

    /// Decode the up-stages of `post_mid [MID_CHANNELS, f, h, w]` in overlapping
    /// spatial (`tile` latent cells) + temporal (`tf` latent frames) tiles,
    /// blending seams (spatial feather ramps, temporal trapezoid masks). A single
    /// tile (`f<=tf`, `h,w<=tile`) reduces to unit weights -> bit-identical to the
    /// whole-tensor up-stage decode.
    #[allow(clippy::too_many_arguments)]
    async fn decode_upstages_tiled(
        &self,
        backend: &WgpuBackend,
        workspace: &Workspace<WgpuBackend>,
        bufs: &DecoderBufs,
        post_mid: &[f32],
        f: usize,
        h: usize,
        w: usize,
        tf: usize,
        tile: u32,
    ) -> Result<Vec<f32>, WgpuError> {
        let overlap = (tile / 4).clamp(1, tile.saturating_sub(1).max(1));
        let t_overlap = ((tf / 4).max(1)).min(tf.saturating_sub(1).max(1));
        let scale = SPATIAL_SCALE as u32;
        let f_px = TEMPORAL_SCALE * (f - 1) + 1;
        let (oh, ow) = (h * SPATIAL_SCALE, w * SPATIAL_SCALE);
        let plane = f_px * oh * ow;

        let ttiles = plan_temporal_tiles(f, tf, t_overlap, TEMPORAL_SCALE);
        let tiles_h = plan_tiles(h as u32, tile, overlap);
        let tiles_w = plan_tiles(w as u32, tile, overlap);
        let single = ttiles.len() == 1 && tiles_h.len() == 1 && tiles_w.len() == 1;
        tracing::info!(
            target: thinfer_core::trace::DIAG,
            n_temporal = ttiles.len(), n_h = tiles_h.len(), n_w = tiles_w.len(),
            total_decodes = ttiles.len() * tiles_h.len() * tiles_w.len(),
            t_overlap, overlap, "hunyuan vae up-stage tile plan",
        );

        let mut video = vec![0.0f32; OUT_CHANNELS * plane];
        let mut wsum_s = vec![0.0f32; oh * ow];
        let mut wsum_t = vec![0.0f32; f_px];

        for &(r0, hext) in &tiles_h {
            let wh = feather_1d(hext, overlap, scale, r0 > 0, r0 + hext < h as u32);
            for &(c0, wext) in &tiles_w {
                let ww = feather_1d(wext, overlap, scale, c0 > 0, c0 + wext < w as u32);
                accumulate_spatial_wsum(
                    &mut wsum_s,
                    ow,
                    (r0 as usize) * SPATIAL_SCALE,
                    (c0 as usize) * SPATIAL_SCALE,
                    &wh,
                    &ww,
                );
            }
        }
        for tt in &ttiles {
            let tmask = trapezoid_mask(tt.o1 - tt.o0, tt.lr, tt.rr);
            for (i, &wv) in tmask.iter().enumerate() {
                wsum_t[tt.o0 + i] += wv;
            }
        }

        for tt in &ttiles {
            let tmask = trapezoid_mask(tt.o1 - tt.o0, tt.lr, tt.rr);
            let tflen = tt.l1 - tt.l0;
            let f_px_tile = tt.o1 - tt.o0;
            for &(r0, hext) in &tiles_h {
                let wh = feather_1d(hext, overlap, scale, r0 > 0, r0 + hext < h as u32);
                for &(c0, wext) in &tiles_w {
                    let ww = feather_1d(wext, overlap, scale, c0 > 0, c0 + wext < w as u32);
                    let (he, we) = (hext as usize, wext as usize);
                    let sub = gather_subtile(
                        post_mid,
                        MID_CHANNELS,
                        f,
                        h,
                        w,
                        tt.l0,
                        tflen,
                        r0 as usize,
                        c0 as usize,
                        he,
                        we,
                    );
                    let (pix, cos) = self
                        .run_upstages_tile(backend, workspace, bufs, &sub, tflen, he, we)
                        .await?;
                    debug_assert_eq!(
                        (cos.t as usize, cos.h as usize, cos.w as usize),
                        (f_px_tile, he * SPATIAL_SCALE, we * SPATIAL_SCALE),
                        "tile output shape vs plan",
                    );
                    blend_tile(
                        &mut video,
                        &pix,
                        OUT_CHANNELS,
                        oh,
                        ow,
                        plane,
                        tt.o0,
                        (r0 as usize) * SPATIAL_SCALE,
                        (c0 as usize) * SPATIAL_SCALE,
                        f_px_tile,
                        he * SPATIAL_SCALE,
                        we * SPATIAL_SCALE,
                        &tmask,
                        &wh,
                        &ww,
                    );
                    if !single {
                        workspace.drain_pool();
                    }
                }
            }
        }

        for c in 0..OUT_CHANNELS {
            let base = c * plane;
            for (t, &wt_raw) in wsum_t.iter().enumerate() {
                let toff = base + t * oh * ow;
                let wt = wt_raw.max(1e-8);
                for (p, &ws) in wsum_s.iter().enumerate() {
                    video[toff + p] /= wt * ws;
                }
            }
        }
        Ok(video)
    }

    /// Run the up-stage conv stack on one post-mid tile `[MID_CHANNELS, tflen,
    /// he, we]`, returning the decoded pixel tile `[3, f_px_tile, he*16, we*16]`
    /// host f32 + its shape.
    async fn run_upstages_tile(
        &self,
        backend: &WgpuBackend,
        workspace: &Workspace<WgpuBackend>,
        bufs: &DecoderBufs,
        sub: &[f32],
        tflen: usize,
        he: usize,
        we: usize,
    ) -> Result<(Vec<f32>, Shape), WgpuError> {
        use thinfer_core::backend::Backend;
        let pl = self.up_pl();
        let act = pl.act;
        let act_size = pl.act_size();
        let in_bytes = act_upload_bytes(act, sub);
        let in_buf = workspace.alloc(in_bytes.len() as u64)?;
        backend.write_buffer(in_buf.id(), 0, &in_bytes)?;
        let s = Shape {
            c: MID_CHANNELS as u32,
            t: tflen as u32,
            h: he as u32,
            w: we as u32,
        };
        let conv_out_persist;
        let conv_out_shape;
        {
            let scope = workspace.batch();
            let xb = scope.import_copy(in_buf.as_buf_ref());
            let (co, cos) = run_upstages(&scope, pl, bufs, xb, s, None)?;
            conv_out_shape = cos;
            conv_out_persist = persist(&scope, workspace, co, cos, act_size)?;
            scope.submit_void().await?;
        }
        let pix = read_acts(
            backend,
            &conv_out_persist.as_buf_ref(),
            conv_out_shape.elems() as usize,
            act,
        )
        .await?;
        Ok((pix, conv_out_shape))
    }

    async fn acquire<'r, S: WeightSource>(
        &self,
        residency: &'r WeightResidency<S>,
        backend: &WgpuBackend,
        pins: &mut Vec<GpuView<'r>>,
    ) -> Result<DecoderBufs, ResidencyError<S::Error, WgpuError>> {
        let h = &self.handles;
        let mut up = Vec::with_capacity(h.up.len());
        for st in &h.up {
            let mut resnets = Vec::with_capacity(st.resnets.len());
            for r in &st.resnets {
                resnets.push(acquire_resnet(residency, backend, r, pins).await?);
            }
            let upsample_conv = match &st.upsample_conv {
                Some(c) => Some(acquire_conv(residency, backend, *c, pins).await?),
                None => None,
            };
            up.push(UpStageBufs {
                resnets,
                upsample_conv,
            });
        }
        Ok(DecoderBufs {
            conv_in: acquire_conv(residency, backend, h.conv_in, pins).await?,
            mid: MidBufs {
                block_1: acquire_resnet(residency, backend, &h.mid.block_1, pins).await?,
                attn_1: acquire_attn(residency, backend, &h.mid.attn_1, pins).await?,
                block_2: acquire_resnet(residency, backend, &h.mid.block_2, pins).await?,
            },
            up,
            norm_out: acquire_rms(residency, backend, h.norm_out, pins).await?,
            conv_out: acquire_conv(residency, backend, h.conv_out, pins).await?,
        })
    }
}

async fn acquire_conv<'r, S: WeightSource>(
    residency: &'r WeightResidency<S>,
    backend: &WgpuBackend,
    h: ConvH,
    pins: &mut Vec<GpuView<'r>>,
) -> Result<ConvBufs, ResidencyError<S::Error, WgpuError>> {
    let wv = residency.acquire(h.weight, backend).await?;
    let bv = residency.acquire(h.bias, backend).await?;
    let bufs = ConvBufs {
        weight: wv.buf(),
        bias: bv.buf(),
    };
    pins.push(wv);
    pins.push(bv);
    Ok(bufs)
}

async fn acquire_rms<'r, S: WeightSource>(
    residency: &'r WeightResidency<S>,
    backend: &WgpuBackend,
    h: RmsH,
    pins: &mut Vec<GpuView<'r>>,
) -> Result<RmsBufs, ResidencyError<S::Error, WgpuError>> {
    let gv = residency.acquire(h.gamma, backend).await?;
    let bufs = RmsBufs { gamma: gv.buf() };
    pins.push(gv);
    Ok(bufs)
}

async fn acquire_resnet<'r, S: WeightSource>(
    residency: &'r WeightResidency<S>,
    backend: &WgpuBackend,
    h: &ResnetH,
    pins: &mut Vec<GpuView<'r>>,
) -> Result<ResnetBufs, ResidencyError<S::Error, WgpuError>> {
    Ok(ResnetBufs {
        norm1: acquire_rms(residency, backend, h.norm1, pins).await?,
        conv1: acquire_conv(residency, backend, h.conv1, pins).await?,
        norm2: acquire_rms(residency, backend, h.norm2, pins).await?,
        conv2: acquire_conv(residency, backend, h.conv2, pins).await?,
    })
}

async fn acquire_attn<'r, S: WeightSource>(
    residency: &'r WeightResidency<S>,
    backend: &WgpuBackend,
    h: &AttnH,
    pins: &mut Vec<GpuView<'r>>,
) -> Result<AttnBufs, ResidencyError<S::Error, WgpuError>> {
    Ok(AttnBufs {
        norm: acquire_rms(residency, backend, h.norm, pins).await?,
        q: acquire_conv(residency, backend, h.q, pins).await?,
        k: acquire_conv(residency, backend, h.k, pins).await?,
        v: acquire_conv(residency, backend, h.v, pins).await?,
        proj: acquire_conv(residency, backend, h.proj, pins).await?,
    })
}

/// Copy a scope-local stage activation into a workspace buffer that outlives the
/// submit, for a post-submit readback.
fn persist<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    workspace: &Workspace<WgpuBackend>,
    buf: BatchBuf<'w>,
    s: Shape,
    act_size: u32,
) -> Result<WsBuf<WgpuBackend>, WgpuError> {
    let ws = workspace.alloc(s.bytes(act_size))?;
    let dst = scope.import_copy(ws.as_buf_ref());
    scope.copy_buffer_to_buffer(buf, 0, dst, 0, s.bytes(act_size))?;
    Ok(ws)
}

/// Decoder conv-weight footprint (bf16 weights), summed from the fixed config.
/// Pinned resident across all up tiles; the per-tile workspace gets the budget
/// minus this minus staging. An overestimate is safe (smaller tiles).
fn weight_footprint_bytes() -> u64 {
    let k3 = |cin: u64, cout: u64| (cout * cin * 27 + cout) * 2;
    let p1 = |cin: u64, cout: u64| (cout * cin + cout) * 2;
    let mid = MID_CHANNELS as u64;
    let mut total = k3(LATENT_CHANNELS as u64, mid); // conv_in
    // mid: block_1 (2 k3) + attn (q/k/v/proj 1x1x1) + block_2 (2 k3).
    total += 4 * k3(mid, mid) + 4 * p1(mid, mid);
    let mut feature = mid;
    for st in &UP_STAGES {
        let ch = st.ch as u64;
        total += 2 * RES_PER_STAGE as u64 * k3(ch, ch);
        feature = ch;
        if let Some((out_c, temporal)) = st.upsample {
            let factor = if temporal { 8 } else { 4 };
            total += k3(ch, out_c as u64 * factor);
            feature = out_c as u64;
        }
    }
    total + k3(feature, OUT_CHANNELS as u64) // conv_out (norm gammas negligible)
}

/// Peak up-stage workspace bytes per (latent tile area * output frame); scales
/// with the act dtype (f16 halves the f32-anchored estimate).
fn peak_bytes_per_area_frame(act_size: u32) -> f64 {
    PEAK_BYTES_PER_AREA_FRAME_F32 * act_size as f64 / 4.0
}

/// Largest latent tile side whose estimated up-stage workspace fits the budget
/// for `f_px` output frames. Clamped to `[TILE_MIN, TILE_MAX]`.
fn initial_tile(workspace_budget: u64, f_px: usize, act_size: u32) -> u32 {
    let denom = (peak_bytes_per_area_frame(act_size) * f_px.max(1) as f64).max(1.0);
    let t = (workspace_budget as f64 / denom).sqrt() as u32;
    t.clamp(TILE_MIN, TILE_MAX)
}

/// Seed the (temporal-depth, spatial-side) tile pair from the budget. Prefer no
/// temporal tiling; else cap the depth so the spatial tile re-grows to
/// `SPATIAL_COMFORT` (balancing seams). The adaptive OOM-retry in `decode`
/// corrects any estimate error.
fn initial_tiles(workspace_budget: u64, f: usize, act_size: u32) -> (usize, u32) {
    let f_px_full = TEMPORAL_SCALE * (f - 1) + 1;
    let tile_full = initial_tile(workspace_budget, f_px_full, act_size);
    if tile_full >= SPATIAL_COMFORT || f <= TEMPORAL_TILE_MIN as usize {
        return (f, tile_full);
    }
    let cap = workspace_budget as f64 / peak_bytes_per_area_frame(act_size);
    let f_px_target = (cap / (SPATIAL_COMFORT as f64).powi(2)).max(1.0);
    // f_px_tile = 4*(tf-1)+1 <= f_px_target  =>  tf <= (f_px_target-1)/4 + 1.
    let tf = (((f_px_target - 1.0) / TEMPORAL_SCALE as f64 + 1.0) as usize)
        .clamp(TEMPORAL_TILE_MIN as usize, TEMPORAL_TILE_MAX.min(f));
    let f_px_tile = TEMPORAL_SCALE * (tf - 1) + 1;
    (tf, initial_tile(workspace_budget, f_px_tile, act_size))
}

async fn read_acts(
    backend: &WgpuBackend,
    buf: &BufRef,
    n: usize,
    act: ActDtype,
) -> Result<Vec<f32>, WgpuError> {
    let act_size = match act {
        ActDtype::F32 => 4,
        ActDtype::F16 => 2,
        other => unreachable!("hunyuan vae acts are f32 or f16, got {other:?}"),
    };
    let bytes = backend
        .read_buffer(buf.id, buf.offset, (n * act_size) as u64)
        .await?;
    Ok(act_readback_to_f32(act, &bytes, n))
}
