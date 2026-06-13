//! Wan 3D causal video VAE (`AutoencoderKLWan`, Wan2.1 config; `is_residual`
//! false). Source: `third-party/diffusers/.../autoencoder_kl_wan.py`.
//!
//! Config (SkyReels-V2-DF-1.3B / Wan2.1):
//! - `base_dim = 96`, `z_dim = 16`, `dim_mult = [1, 2, 4, 4]`,
//!   `num_res_blocks = 2`, `temperal_downsample = [F, T, T]` (so the decoder
//!   upsamples temporally on blocks 0,1 and spatially-only on block 2; block 3
//!   does neither). 4x temporal / 8x spatial compression.
//! - `WanRMS_norm` (channel-dim RMS, no bias, `* sqrt(C) * gamma`) everywhere a
//!   norm appears (no GroupNorm, unlike Z-Image).
//! - All convs are `WanCausalConv3d` (front-padded time, symmetric H/W) EXCEPT
//!   the spatial resample convs, which are per-frame `nn.Conv2d` applied over a
//!   `[B*T, C, H, W]` reshape (so we reuse the 2D conv kernel for those).
//!
//! Decode (`_decode`): `post_quant_conv -> [per latent frame] decoder(...)`,
//! cat over time, `clamp[-1, 1]`. The decoder is conv_in -> mid(resnet, attn,
//! resnet) -> 4 up_blocks -> norm_out -> silu -> conv_out. Encode is the mirror
//! image: conv_in -> 4 down stages -> mid -> norm_out -> silu -> conv_out ->
//! quant_conv, fed one 4-frame chunk at a time.
//!
//! Causality is realized with a host-side `feat_cache`: the per-frame loop keeps
//! the trailing input frames each causal conv needs and front-assembles them
//! before dispatch (see the forward driver). This file declares the typed
//! `WeightId` bundles, residency handles, GPU views, `BufRef` bundles, and the
//! compiled-pipeline set. The forward driver lands in a follow-up.

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError};
use thinfer_core::cache::KernelKey;
use thinfer_core::ops::{
    ActDtype, AddF32, BcastAddF32, BcastAddOp, ConcatTimeF32, ConcatTimeOp, Conv2dConfig,
    Conv2dF32, Conv2dOp, Conv3dConfig, Conv3dF32, Conv3dOp, MatMulConfig, MatMulF32, MatmulOp, Op,
    RmsNorm3dF32, RmsNorm3dOp, SdpaF32LargeD, SdpaOp, SiluF32, Transpose12F32, Transpose12Op,
    Upsample2dNearestF32, Upsample2dNearestOp, WgslConfig,
};
use thinfer_core::residency::{GpuView, ResidencyError, WeightHandle, WeightResidency};
use thinfer_core::trace::PHASE;
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, ScopePacker, Workspace, WsBuf};
use tracing::Instrument;

use crate::common::loader::{LoadError, register_passthrough};

// ============================================================================
// Config: derived dims for the Wan2.1 VAE encoder + decoder.
// ============================================================================

pub mod config {
    pub const BASE_DIM: usize = 96;
    pub const Z_DIM: usize = 16;
    pub const IN_CHANNELS: usize = 3;
    pub const OUT_CHANNELS: usize = 3;
    pub const DIM_MULT: [usize; 4] = [1, 2, 4, 4];
    pub const NUM_RES_BLOCKS: usize = 2;
    /// Decoder up_blocks / encoder mid use `num_res_blocks + 1` resnets.
    pub const RESNETS_PER_UP_BLOCK: usize = NUM_RES_BLOCKS + 1;
    /// `temperal_downsample` (encoder, low->high res order).
    pub const TEMPORAL_DOWNSAMPLE: [bool; 3] = [false, true, true];
    /// `temperal_upsample` = reversed `temperal_downsample` (decoder order).
    pub const TEMPORAL_UPSAMPLE: [bool; 3] = [true, true, false];
    /// RMS norm eps floor (matches `F.normalize` default in `WanRMS_norm`).
    pub const NORM_EPS: f32 = 1e-12;
    pub const ATTN_HEADS: usize = 1;
    /// 8x spatial / 4x temporal.
    pub const SPATIAL_COMPRESSION: usize = 8;
    pub const TEMPORAL_COMPRESSION: usize = 4;

    /// Baked latent normalization (16-vector each); decode pre-scales the latent
    /// by `z * std + mean` per channel before `post_quant_conv`.
    pub const LATENTS_MEAN: [f32; Z_DIM] = [
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715,
        0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    ];
    pub const LATENTS_STD: [f32; Z_DIM] = [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652,
        1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
    ];

    /// Decoder per-stage channel dims: `[dim_mult[-1]] + dim_mult[::-1]`, times
    /// `base_dim`. Length `N_UP_BLOCKS + 1` = 5: `[384, 384, 384, 192, 96]`.
    pub const N_UP_BLOCKS: usize = DIM_MULT.len();
    pub const fn dec_dim(i: usize) -> usize {
        // [dim_mult[3], dim_mult[3], dim_mult[2], dim_mult[1], dim_mult[0]]
        let mult = if i == 0 {
            DIM_MULT[DIM_MULT.len() - 1]
        } else {
            DIM_MULT[DIM_MULT.len() - i]
        };
        BASE_DIM * mult
    }
    /// Channels feeding the mid block / conv_in output (= `dec_dim(0)` = 384).
    pub const MID_CHANNELS: usize = dec_dim(0);

    /// Up_block `i` output channels (`dims[i+1]`).
    pub const fn up_out_channels(i: usize) -> usize {
        dec_dim(i + 1)
    }
    /// Up_block `i` input channels. For `i > 0` the previous block's upsampler
    /// halved the channel count (its spatial conv maps `out -> out/2`), so the
    /// stored block input is `dims[i] / 2`. `i == 0` consumes the mid output.
    pub const fn up_in_channels(i: usize) -> usize {
        if i == 0 { dec_dim(0) } else { dec_dim(i) / 2 }
    }
    /// Up_block `i` has an upsampler on all but the last block.
    pub const fn up_has_upsampler(i: usize) -> bool {
        i + 1 < N_UP_BLOCKS
    }
    /// Whether up_block `i`'s upsampler upsamples in time (`upsample3d`).
    pub const fn up_temporal(i: usize) -> bool {
        up_has_upsampler(i) && TEMPORAL_UPSAMPLE[i]
    }
    /// Upsampler spatial conv output channels (`out_dim / 2`).
    pub const fn up_resample_out(i: usize) -> usize {
        up_out_channels(i) / 2
    }

    /// Encoder per-stage channel dims: `[1] + dim_mult` times `base_dim`.
    /// Length `N_DOWN_STAGES + 1` = 5: `[96, 96, 192, 384, 384]`.
    pub const N_DOWN_STAGES: usize = DIM_MULT.len();
    pub const fn enc_dim(i: usize) -> usize {
        let mult = if i == 0 { 1 } else { DIM_MULT[i - 1] };
        BASE_DIM * mult
    }
    pub const fn down_in_channels(i: usize) -> usize {
        enc_dim(i)
    }
    pub const fn down_out_channels(i: usize) -> usize {
        enc_dim(i + 1)
    }
    pub const fn down_has_downsampler(i: usize) -> bool {
        i + 1 < N_DOWN_STAGES
    }
    /// Whether down stage `i`'s downsampler downsamples in time (`downsample3d`).
    pub const fn down_temporal(i: usize) -> bool {
        down_has_downsampler(i) && TEMPORAL_DOWNSAMPLE[i]
    }
    /// Encoder mid / norm_out channels (= last enc_dim = 384).
    pub const ENC_MID_CHANNELS: usize = enc_dim(N_DOWN_STAGES);
    /// Encoder latent output channels before quant_conv (`z_dim * 2`).
    pub const Z_DIM_X2: usize = Z_DIM * 2;
}

// ============================================================================
// Weight ID bundles
// ============================================================================

/// `WanCausalConv3d` (or per-frame `nn.Conv2d`): weight `[Cout, Cin, kT, kH,
/// kW]` (or `[Cout, Cin, kH, kW]`) + bias `[Cout]`. Always passthrough (no
/// transpose), like Z-Image conv2d.
#[derive(Clone, Debug)]
pub struct ConvWeights {
    pub weight: WeightId,
    pub bias: WeightId,
}

/// `WanRMS_norm`: per-channel gain only (no bias).
#[derive(Clone, Debug)]
pub struct RmsWeights {
    pub gamma: WeightId,
}

/// `WanResidualBlock`: norm1 -> conv1 -> norm2 -> conv2 (+ conv_shortcut when
/// `in != out`).
#[derive(Clone, Debug)]
pub struct ResnetWeights {
    pub norm1: RmsWeights,
    pub conv1: ConvWeights,
    pub norm2: RmsWeights,
    pub conv2: ConvWeights,
    pub conv_shortcut: Option<ConvWeights>,
}

/// `WanAttentionBlock`: channel RMS norm, fused `to_qkv` 1x1 conv (`C -> 3C`),
/// `proj` 1x1 conv (`C -> C`). Single head over `H*W` spatial tokens per frame.
#[derive(Clone, Debug)]
pub struct AttnWeights {
    pub norm: RmsWeights,
    pub to_qkv: ConvWeights,
    pub proj: ConvWeights,
}

/// `WanMidBlock`: resnet -> attention -> resnet.
#[derive(Clone, Debug)]
pub struct MidBlockWeights {
    pub resnets: [ResnetWeights; 2],
    pub attention: AttnWeights,
}

/// `WanResample` (decoder upsample): a per-frame spatial conv (`Conv2d`, kept in
/// `resample.1`) plus, for `upsample3d`, a temporal `time_conv` causal 3D conv.
#[derive(Clone, Debug)]
pub struct UpsampleWeights {
    pub spatial_conv: ConvWeights,
    pub time_conv: Option<ConvWeights>,
}

/// `WanUpBlock`: `num_res_blocks + 1` resnets, optional upsampler.
#[derive(Clone, Debug)]
pub struct UpBlockWeights {
    pub resnets: Vec<ResnetWeights>,
    pub upsampler: Option<UpsampleWeights>,
}

/// `WanResample` (encoder downsample): a per-frame strided spatial conv plus,
/// for `downsample3d`, a temporal `time_conv` (stride-2 in time).
#[derive(Clone, Debug)]
pub struct DownsampleWeights {
    pub spatial_conv: ConvWeights,
    pub time_conv: Option<ConvWeights>,
}

/// One encoder down stage: `num_res_blocks` resnets + optional downsampler.
#[derive(Clone, Debug)]
pub struct DownStageWeights {
    pub resnets: Vec<ResnetWeights>,
    pub downsampler: Option<DownsampleWeights>,
}

#[derive(Clone, Debug)]
pub struct VaeDecoderWeights {
    /// `AutoencoderKLWan.post_quant_conv` (`z_dim -> z_dim`, 1x1), applied
    /// before the decoder proper.
    pub post_quant_conv: ConvWeights,
    pub conv_in: ConvWeights,
    pub mid_block: MidBlockWeights,
    pub up_blocks: Vec<UpBlockWeights>,
    pub norm_out: RmsWeights,
    pub conv_out: ConvWeights,
}

#[derive(Clone, Debug)]
pub struct VaeEncoderWeights {
    pub conv_in: ConvWeights,
    pub down_stages: Vec<DownStageWeights>,
    pub mid_block: MidBlockWeights,
    pub norm_out: RmsWeights,
    pub conv_out: ConvWeights,
    /// `AutoencoderKLWan.quant_conv` (`2*z_dim -> 2*z_dim`, 1x1).
    pub quant_conv: ConvWeights,
}

impl VaeDecoderWeights {
    pub fn new() -> Self {
        let conv = conv_ids;
        let rms = rms_ids;

        let mid_block = MidBlockWeights {
            resnets: [
                resnet_ids("decoder.mid_block.resnets.0", false),
                resnet_ids("decoder.mid_block.resnets.1", false),
            ],
            attention: attn_ids("decoder.mid_block.attentions.0"),
        };

        let mut up_blocks = Vec::with_capacity(config::N_UP_BLOCKS);
        for i in 0..config::N_UP_BLOCKS {
            let cin = config::up_in_channels(i);
            let cout = config::up_out_channels(i);
            let mut resnets = Vec::with_capacity(config::RESNETS_PER_UP_BLOCK);
            for j in 0..config::RESNETS_PER_UP_BLOCK {
                let rin = if j == 0 { cin } else { cout };
                resnets.push(resnet_ids(
                    &format!("decoder.up_blocks.{i}.resnets.{j}"),
                    rin != cout,
                ));
            }
            let upsampler = config::up_has_upsampler(i).then(|| {
                let p = format!("decoder.up_blocks.{i}.upsamplers.0");
                UpsampleWeights {
                    spatial_conv: conv(&format!("{p}.resample.1")),
                    time_conv: config::up_temporal(i).then(|| conv(&format!("{p}.time_conv"))),
                }
            });
            up_blocks.push(UpBlockWeights { resnets, upsampler });
        }

        Self {
            post_quant_conv: conv("post_quant_conv"),
            conv_in: conv("decoder.conv_in"),
            mid_block,
            up_blocks,
            norm_out: rms("decoder.norm_out"),
            conv_out: conv("decoder.conv_out"),
        }
    }
}

impl Default for VaeDecoderWeights {
    fn default() -> Self {
        Self::new()
    }
}

impl VaeEncoderWeights {
    pub fn new() -> Self {
        let conv = conv_ids;
        let rms = rms_ids;

        // Encoder `down_blocks` is a flat ModuleList: each stage contributes
        // `num_res_blocks` resnets then (for all but the last) a downsampler, so
        // stage `i` starts at flat index `i * (num_res_blocks + 1)`.
        let mut down_stages = Vec::with_capacity(config::N_DOWN_STAGES);
        for i in 0..config::N_DOWN_STAGES {
            let cin = config::down_in_channels(i);
            let cout = config::down_out_channels(i);
            let base = i * (config::NUM_RES_BLOCKS + 1);
            let mut resnets = Vec::with_capacity(config::NUM_RES_BLOCKS);
            for j in 0..config::NUM_RES_BLOCKS {
                let rin = if j == 0 { cin } else { cout };
                resnets.push(resnet_ids(
                    &format!("encoder.down_blocks.{}", base + j),
                    rin != cout,
                ));
            }
            let downsampler = config::down_has_downsampler(i).then(|| {
                let p = format!("encoder.down_blocks.{}", base + config::NUM_RES_BLOCKS);
                DownsampleWeights {
                    spatial_conv: conv(&format!("{p}.resample.1")),
                    time_conv: config::down_temporal(i).then(|| conv(&format!("{p}.time_conv"))),
                }
            });
            down_stages.push(DownStageWeights {
                resnets,
                downsampler,
            });
        }

        let mid_block = MidBlockWeights {
            resnets: [
                resnet_ids("encoder.mid_block.resnets.0", false),
                resnet_ids("encoder.mid_block.resnets.1", false),
            ],
            attention: attn_ids("encoder.mid_block.attentions.0"),
        };

        Self {
            conv_in: conv("encoder.conv_in"),
            down_stages,
            mid_block,
            norm_out: rms("encoder.norm_out"),
            conv_out: conv("encoder.conv_out"),
            quant_conv: conv("quant_conv"),
        }
    }
}

impl Default for VaeEncoderWeights {
    fn default() -> Self {
        Self::new()
    }
}

fn conv_ids(prefix: &str) -> ConvWeights {
    ConvWeights {
        weight: WeightId(format!("{prefix}.weight")),
        bias: WeightId(format!("{prefix}.bias")),
    }
}

fn rms_ids(prefix: &str) -> RmsWeights {
    // `WanRMS_norm.gamma`.
    RmsWeights {
        gamma: WeightId(format!("{prefix}.gamma")),
    }
}

fn resnet_ids(prefix: &str, has_shortcut: bool) -> ResnetWeights {
    ResnetWeights {
        norm1: rms_ids(&format!("{prefix}.norm1")),
        conv1: conv_ids(&format!("{prefix}.conv1")),
        norm2: rms_ids(&format!("{prefix}.norm2")),
        conv2: conv_ids(&format!("{prefix}.conv2")),
        conv_shortcut: has_shortcut.then(|| conv_ids(&format!("{prefix}.conv_shortcut"))),
    }
}

fn attn_ids(prefix: &str) -> AttnWeights {
    AttnWeights {
        norm: rms_ids(&format!("{prefix}.norm")),
        to_qkv: conv_ids(&format!("{prefix}.to_qkv")),
        proj: conv_ids(&format!("{prefix}.proj")),
    }
}

// ============================================================================
// Residency handles (no GPU allocation; bytes flow on `acquire`)
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub struct ConvHandles {
    pub weight: WeightHandle,
    pub bias: WeightHandle,
}

#[derive(Clone, Copy, Debug)]
pub struct RmsHandles {
    pub gamma: WeightHandle,
}

#[derive(Clone, Copy, Debug)]
pub struct ResnetHandles {
    pub norm1: RmsHandles,
    pub conv1: ConvHandles,
    pub norm2: RmsHandles,
    pub conv2: ConvHandles,
    pub conv_shortcut: Option<ConvHandles>,
}

#[derive(Clone, Copy, Debug)]
pub struct AttnHandles {
    pub norm: RmsHandles,
    pub to_qkv: ConvHandles,
    pub proj: ConvHandles,
}

#[derive(Clone, Copy, Debug)]
pub struct MidBlockHandles {
    pub resnets: [ResnetHandles; 2],
    pub attention: AttnHandles,
}

#[derive(Clone, Copy, Debug)]
pub struct UpsampleHandles {
    pub spatial_conv: ConvHandles,
    pub time_conv: Option<ConvHandles>,
}

#[derive(Clone, Debug)]
pub struct UpBlockHandles {
    pub resnets: Vec<ResnetHandles>,
    pub upsampler: Option<UpsampleHandles>,
}

#[derive(Clone, Copy, Debug)]
pub struct DownsampleHandles {
    pub spatial_conv: ConvHandles,
    pub time_conv: Option<ConvHandles>,
}

#[derive(Clone, Debug)]
pub struct DownStageHandles {
    pub resnets: Vec<ResnetHandles>,
    pub downsampler: Option<DownsampleHandles>,
}

#[derive(Clone, Debug)]
pub struct VaeDecoderHandles {
    pub post_quant_conv: ConvHandles,
    pub conv_in: ConvHandles,
    pub mid_block: MidBlockHandles,
    pub up_blocks: Vec<UpBlockHandles>,
    pub norm_out: RmsHandles,
    pub conv_out: ConvHandles,
}

#[derive(Clone, Debug)]
pub struct VaeEncoderHandles {
    pub conv_in: ConvHandles,
    pub down_stages: Vec<DownStageHandles>,
    pub mid_block: MidBlockHandles,
    pub norm_out: RmsHandles,
    pub conv_out: ConvHandles,
    pub quant_conv: ConvHandles,
}

// ============================================================================
// Registration: build handles from weight names (no GPU upload yet).
// ============================================================================

fn reg_conv<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &ConvWeights,
) -> Result<ConvHandles, LoadError> {
    Ok(ConvHandles {
        weight: register_passthrough(residency, &w.weight)?,
        bias: register_passthrough(residency, &w.bias)?,
    })
}

fn reg_rms<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &RmsWeights,
) -> Result<RmsHandles, LoadError> {
    Ok(RmsHandles {
        gamma: register_passthrough(residency, &w.gamma)?,
    })
}

fn reg_resnet<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &ResnetWeights,
) -> Result<ResnetHandles, LoadError> {
    Ok(ResnetHandles {
        norm1: reg_rms(residency, &w.norm1)?,
        conv1: reg_conv(residency, &w.conv1)?,
        norm2: reg_rms(residency, &w.norm2)?,
        conv2: reg_conv(residency, &w.conv2)?,
        conv_shortcut: match &w.conv_shortcut {
            Some(c) => Some(reg_conv(residency, c)?),
            None => None,
        },
    })
}

fn reg_attn<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &AttnWeights,
) -> Result<AttnHandles, LoadError> {
    Ok(AttnHandles {
        norm: reg_rms(residency, &w.norm)?,
        to_qkv: reg_conv(residency, &w.to_qkv)?,
        proj: reg_conv(residency, &w.proj)?,
    })
}

fn reg_mid<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &MidBlockWeights,
) -> Result<MidBlockHandles, LoadError> {
    Ok(MidBlockHandles {
        resnets: [
            reg_resnet(residency, &w.resnets[0])?,
            reg_resnet(residency, &w.resnets[1])?,
        ],
        attention: reg_attn(residency, &w.attention)?,
    })
}

pub fn register_decoder<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &VaeDecoderWeights,
) -> Result<VaeDecoderHandles, LoadError> {
    let mut up_blocks = Vec::with_capacity(w.up_blocks.len());
    for ub in &w.up_blocks {
        let mut resnets = Vec::with_capacity(ub.resnets.len());
        for r in &ub.resnets {
            resnets.push(reg_resnet(residency, r)?);
        }
        let upsampler = match &ub.upsampler {
            Some(u) => Some(UpsampleHandles {
                spatial_conv: reg_conv(residency, &u.spatial_conv)?,
                time_conv: match &u.time_conv {
                    Some(c) => Some(reg_conv(residency, c)?),
                    None => None,
                },
            }),
            None => None,
        };
        up_blocks.push(UpBlockHandles { resnets, upsampler });
    }
    Ok(VaeDecoderHandles {
        post_quant_conv: reg_conv(residency, &w.post_quant_conv)?,
        conv_in: reg_conv(residency, &w.conv_in)?,
        mid_block: reg_mid(residency, &w.mid_block)?,
        up_blocks,
        norm_out: reg_rms(residency, &w.norm_out)?,
        conv_out: reg_conv(residency, &w.conv_out)?,
    })
}

pub fn register_encoder<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &VaeEncoderWeights,
) -> Result<VaeEncoderHandles, LoadError> {
    let mut down_stages = Vec::with_capacity(w.down_stages.len());
    for ds in &w.down_stages {
        let mut resnets = Vec::with_capacity(ds.resnets.len());
        for r in &ds.resnets {
            resnets.push(reg_resnet(residency, r)?);
        }
        let downsampler = match &ds.downsampler {
            Some(d) => Some(DownsampleHandles {
                spatial_conv: reg_conv(residency, &d.spatial_conv)?,
                time_conv: match &d.time_conv {
                    Some(c) => Some(reg_conv(residency, c)?),
                    None => None,
                },
            }),
            None => None,
        };
        down_stages.push(DownStageHandles {
            resnets,
            downsampler,
        });
    }
    Ok(VaeEncoderHandles {
        conv_in: reg_conv(residency, &w.conv_in)?,
        down_stages,
        mid_block: reg_mid(residency, &w.mid_block)?,
        norm_out: reg_rms(residency, &w.norm_out)?,
        conv_out: reg_conv(residency, &w.conv_out)?,
        quant_conv: reg_conv(residency, &w.quant_conv)?,
    })
}

// ============================================================================
// GpuView bundles (pin guards; `bufs()` materializes BufRefs)
// ============================================================================

pub struct ConvViews<'a> {
    pub weight: GpuView<'a>,
    pub bias: GpuView<'a>,
}
pub struct RmsViews<'a> {
    pub gamma: GpuView<'a>,
}
pub struct ResnetViews<'a> {
    pub norm1: RmsViews<'a>,
    pub conv1: ConvViews<'a>,
    pub norm2: RmsViews<'a>,
    pub conv2: ConvViews<'a>,
    pub conv_shortcut: Option<ConvViews<'a>>,
}
pub struct AttnViews<'a> {
    pub norm: RmsViews<'a>,
    pub to_qkv: ConvViews<'a>,
    pub proj: ConvViews<'a>,
}
pub struct MidBlockViews<'a> {
    pub resnets: [ResnetViews<'a>; 2],
    pub attention: AttnViews<'a>,
}
pub struct UpsampleViews<'a> {
    pub spatial_conv: ConvViews<'a>,
    pub time_conv: Option<ConvViews<'a>>,
}
pub struct UpBlockViews<'a> {
    pub resnets: Vec<ResnetViews<'a>>,
    pub upsampler: Option<UpsampleViews<'a>>,
}
pub struct DownsampleViews<'a> {
    pub spatial_conv: ConvViews<'a>,
    pub time_conv: Option<ConvViews<'a>>,
}
pub struct DownStageViews<'a> {
    pub resnets: Vec<ResnetViews<'a>>,
    pub downsampler: Option<DownsampleViews<'a>>,
}
pub struct VaeDecoderViews<'a> {
    pub post_quant_conv: ConvViews<'a>,
    pub conv_in: ConvViews<'a>,
    pub mid_block: MidBlockViews<'a>,
    pub up_blocks: Vec<UpBlockViews<'a>>,
    pub norm_out: RmsViews<'a>,
    pub conv_out: ConvViews<'a>,
}
pub struct VaeEncoderViews<'a> {
    pub conv_in: ConvViews<'a>,
    pub down_stages: Vec<DownStageViews<'a>>,
    pub mid_block: MidBlockViews<'a>,
    pub norm_out: RmsViews<'a>,
    pub conv_out: ConvViews<'a>,
    pub quant_conv: ConvViews<'a>,
}

// ============================================================================
// BufRef bundles (post-acquire, for forward driver)
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub struct ConvBufs {
    pub weight: thinfer_core::backend::BufRef,
    pub bias: thinfer_core::backend::BufRef,
}
#[derive(Clone, Copy, Debug)]
pub struct RmsBufs {
    pub gamma: thinfer_core::backend::BufRef,
}
#[derive(Clone, Copy, Debug)]
pub struct ResnetBufs {
    pub norm1: RmsBufs,
    pub conv1: ConvBufs,
    pub norm2: RmsBufs,
    pub conv2: ConvBufs,
    pub conv_shortcut: Option<ConvBufs>,
}
#[derive(Clone, Copy, Debug)]
pub struct AttnBufs {
    pub norm: RmsBufs,
    pub to_qkv: ConvBufs,
    pub proj: ConvBufs,
}
#[derive(Clone, Copy, Debug)]
pub struct MidBlockBufs {
    pub resnets: [ResnetBufs; 2],
    pub attention: AttnBufs,
}
#[derive(Clone, Copy, Debug)]
pub struct UpsampleBufs {
    pub spatial_conv: ConvBufs,
    pub time_conv: Option<ConvBufs>,
}
#[derive(Clone, Debug)]
pub struct UpBlockBufs {
    pub resnets: Vec<ResnetBufs>,
    pub upsampler: Option<UpsampleBufs>,
}
#[derive(Clone, Copy, Debug)]
pub struct DownsampleBufs {
    pub spatial_conv: ConvBufs,
    pub time_conv: Option<ConvBufs>,
}
#[derive(Clone, Debug)]
pub struct DownStageBufs {
    pub resnets: Vec<ResnetBufs>,
    pub downsampler: Option<DownsampleBufs>,
}
#[derive(Clone, Debug)]
pub struct VaeDecoderBufs {
    pub post_quant_conv: ConvBufs,
    pub conv_in: ConvBufs,
    pub mid_block: MidBlockBufs,
    pub up_blocks: Vec<UpBlockBufs>,
    pub norm_out: RmsBufs,
    pub conv_out: ConvBufs,
}
#[derive(Clone, Debug)]
pub struct VaeEncoderBufs {
    pub conv_in: ConvBufs,
    pub down_stages: Vec<DownStageBufs>,
    pub mid_block: MidBlockBufs,
    pub norm_out: RmsBufs,
    pub conv_out: ConvBufs,
    pub quant_conv: ConvBufs,
}

impl ConvViews<'_> {
    pub fn bufs(&self) -> ConvBufs {
        ConvBufs {
            weight: self.weight.buf(),
            bias: self.bias.buf(),
        }
    }
}
impl RmsViews<'_> {
    pub fn bufs(&self) -> RmsBufs {
        RmsBufs {
            gamma: self.gamma.buf(),
        }
    }
}
impl ResnetViews<'_> {
    pub fn bufs(&self) -> ResnetBufs {
        ResnetBufs {
            norm1: self.norm1.bufs(),
            conv1: self.conv1.bufs(),
            norm2: self.norm2.bufs(),
            conv2: self.conv2.bufs(),
            conv_shortcut: self.conv_shortcut.as_ref().map(|c| c.bufs()),
        }
    }
}
impl AttnViews<'_> {
    pub fn bufs(&self) -> AttnBufs {
        AttnBufs {
            norm: self.norm.bufs(),
            to_qkv: self.to_qkv.bufs(),
            proj: self.proj.bufs(),
        }
    }
}
impl MidBlockViews<'_> {
    pub fn bufs(&self) -> MidBlockBufs {
        MidBlockBufs {
            resnets: [self.resnets[0].bufs(), self.resnets[1].bufs()],
            attention: self.attention.bufs(),
        }
    }
}
impl UpBlockViews<'_> {
    pub fn bufs(&self) -> UpBlockBufs {
        UpBlockBufs {
            resnets: self.resnets.iter().map(|r| r.bufs()).collect(),
            upsampler: self.upsampler.as_ref().map(|u| UpsampleBufs {
                spatial_conv: u.spatial_conv.bufs(),
                time_conv: u.time_conv.as_ref().map(|c| c.bufs()),
            }),
        }
    }
}
impl DownStageViews<'_> {
    pub fn bufs(&self) -> DownStageBufs {
        DownStageBufs {
            resnets: self.resnets.iter().map(|r| r.bufs()).collect(),
            downsampler: self.downsampler.as_ref().map(|d| DownsampleBufs {
                spatial_conv: d.spatial_conv.bufs(),
                time_conv: d.time_conv.as_ref().map(|c| c.bufs()),
            }),
        }
    }
}
impl VaeDecoderViews<'_> {
    pub fn bufs(&self) -> VaeDecoderBufs {
        VaeDecoderBufs {
            post_quant_conv: self.post_quant_conv.bufs(),
            conv_in: self.conv_in.bufs(),
            mid_block: self.mid_block.bufs(),
            up_blocks: self.up_blocks.iter().map(|u| u.bufs()).collect(),
            norm_out: self.norm_out.bufs(),
            conv_out: self.conv_out.bufs(),
        }
    }
}
impl VaeEncoderViews<'_> {
    pub fn bufs(&self) -> VaeEncoderBufs {
        VaeEncoderBufs {
            conv_in: self.conv_in.bufs(),
            down_stages: self.down_stages.iter().map(|d| d.bufs()).collect(),
            mid_block: self.mid_block.bufs(),
            norm_out: self.norm_out.bufs(),
            conv_out: self.conv_out.bufs(),
            quant_conv: self.quant_conv.bufs(),
        }
    }
}

// ============================================================================
// `acquire` impls (page weights resident; return pin guards)
// ============================================================================

impl ConvHandles {
    pub async fn acquire<'r, S: WeightSource>(
        &self,
        residency: &'r WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<ConvViews<'r>, ResidencyError<S::Error, WgpuError>> {
        Ok(ConvViews {
            weight: residency.acquire(self.weight, backend).await?,
            bias: residency.acquire(self.bias, backend).await?,
        })
    }
}

impl RmsHandles {
    pub async fn acquire<'r, S: WeightSource>(
        &self,
        residency: &'r WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<RmsViews<'r>, ResidencyError<S::Error, WgpuError>> {
        Ok(RmsViews {
            gamma: residency.acquire(self.gamma, backend).await?,
        })
    }
}

impl ResnetHandles {
    pub async fn acquire<'r, S: WeightSource>(
        &self,
        residency: &'r WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<ResnetViews<'r>, ResidencyError<S::Error, WgpuError>> {
        Ok(ResnetViews {
            norm1: self.norm1.acquire(residency, backend).await?,
            conv1: self.conv1.acquire(residency, backend).await?,
            norm2: self.norm2.acquire(residency, backend).await?,
            conv2: self.conv2.acquire(residency, backend).await?,
            conv_shortcut: match self.conv_shortcut {
                Some(c) => Some(c.acquire(residency, backend).await?),
                None => None,
            },
        })
    }
}

impl AttnHandles {
    pub async fn acquire<'r, S: WeightSource>(
        &self,
        residency: &'r WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<AttnViews<'r>, ResidencyError<S::Error, WgpuError>> {
        Ok(AttnViews {
            norm: self.norm.acquire(residency, backend).await?,
            to_qkv: self.to_qkv.acquire(residency, backend).await?,
            proj: self.proj.acquire(residency, backend).await?,
        })
    }
}

impl MidBlockHandles {
    pub async fn acquire<'r, S: WeightSource>(
        &self,
        residency: &'r WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<MidBlockViews<'r>, ResidencyError<S::Error, WgpuError>> {
        Ok(MidBlockViews {
            resnets: [
                self.resnets[0].acquire(residency, backend).await?,
                self.resnets[1].acquire(residency, backend).await?,
            ],
            attention: self.attention.acquire(residency, backend).await?,
        })
    }
}

impl UpBlockHandles {
    pub async fn acquire<'r, S: WeightSource>(
        &self,
        residency: &'r WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<UpBlockViews<'r>, ResidencyError<S::Error, WgpuError>> {
        let mut resnets = Vec::with_capacity(self.resnets.len());
        for r in &self.resnets {
            resnets.push(r.acquire(residency, backend).await?);
        }
        let upsampler = match &self.upsampler {
            Some(u) => Some(UpsampleViews {
                spatial_conv: u.spatial_conv.acquire(residency, backend).await?,
                time_conv: match &u.time_conv {
                    Some(c) => Some(c.acquire(residency, backend).await?),
                    None => None,
                },
            }),
            None => None,
        };
        Ok(UpBlockViews { resnets, upsampler })
    }
}

impl DownStageHandles {
    pub async fn acquire<'r, S: WeightSource>(
        &self,
        residency: &'r WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<DownStageViews<'r>, ResidencyError<S::Error, WgpuError>> {
        let mut resnets = Vec::with_capacity(self.resnets.len());
        for r in &self.resnets {
            resnets.push(r.acquire(residency, backend).await?);
        }
        let downsampler = match &self.downsampler {
            Some(d) => Some(DownsampleViews {
                spatial_conv: d.spatial_conv.acquire(residency, backend).await?,
                time_conv: match &d.time_conv {
                    Some(c) => Some(c.acquire(residency, backend).await?),
                    None => None,
                },
            }),
            None => None,
        };
        Ok(DownStageViews {
            resnets,
            downsampler,
        })
    }
}

impl VaeDecoderHandles {
    pub async fn acquire<'r, S: WeightSource>(
        &self,
        residency: &'r WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<VaeDecoderViews<'r>, ResidencyError<S::Error, WgpuError>> {
        let mut up_blocks = Vec::with_capacity(self.up_blocks.len());
        for ub in &self.up_blocks {
            up_blocks.push(ub.acquire(residency, backend).await?);
        }
        Ok(VaeDecoderViews {
            post_quant_conv: self.post_quant_conv.acquire(residency, backend).await?,
            conv_in: self.conv_in.acquire(residency, backend).await?,
            mid_block: self.mid_block.acquire(residency, backend).await?,
            up_blocks,
            norm_out: self.norm_out.acquire(residency, backend).await?,
            conv_out: self.conv_out.acquire(residency, backend).await?,
        })
    }
}

impl VaeEncoderHandles {
    pub async fn acquire<'r, S: WeightSource>(
        &self,
        residency: &'r WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<VaeEncoderViews<'r>, ResidencyError<S::Error, WgpuError>> {
        let mut down_stages = Vec::with_capacity(self.down_stages.len());
        for ds in &self.down_stages {
            down_stages.push(ds.acquire(residency, backend).await?);
        }
        Ok(VaeEncoderViews {
            conv_in: self.conv_in.acquire(residency, backend).await?,
            down_stages,
            mid_block: self.mid_block.acquire(residency, backend).await?,
            norm_out: self.norm_out.acquire(residency, backend).await?,
            conv_out: self.conv_out.acquire(residency, backend).await?,
            quant_conv: self.quant_conv.acquire(residency, backend).await?,
        })
    }
}

// ============================================================================
// Compiled pipelines
// ============================================================================

/// One compiled conv3d variant: pipeline + the op (tile config) it was built
/// from.
pub struct Conv3dPipeline {
    pub pipeline: thinfer_core::backend::WgpuPipeline,
    pub op: Conv3dF32,
}

impl Conv3dPipeline {
    async fn compile(
        backend: &WgpuBackend,
        label: &str,
        cfg: &WgslConfig,
        tile: Conv3dConfig,
    ) -> Result<Self, WgpuError> {
        use thinfer_core::backend::Backend;
        let op = Conv3dF32::new(tile);
        let pipeline = backend
            .create_pipeline(
                label,
                &op.wgsl(cfg),
                "main",
                <Conv3dF32 as Conv3dOp>::layout(),
            )
            .await?;
        Ok(Self { pipeline, op })
    }
}

/// One compiled conv2d variant (the per-frame spatial resample convs).
pub struct Conv2dPipeline {
    pub pipeline: thinfer_core::backend::WgpuPipeline,
    pub op: Conv2dF32,
}

impl Conv2dPipeline {
    async fn compile(
        backend: &WgpuBackend,
        label: &str,
        cfg: &WgslConfig,
        tile: Conv2dConfig,
    ) -> Result<Self, WgpuError> {
        use thinfer_core::backend::Backend;
        let op = Conv2dF32::new(tile);
        let pipeline = backend
            .create_pipeline(
                label,
                &op.wgsl(cfg),
                "main",
                <Conv2dF32 as Conv2dOp>::layout(),
            )
            .await?;
        Ok(Self { pipeline, op })
    }
}

/// Mid-block 1-head attention linear matmul tile (tokens x C @ C x C). Same
/// config serves q/k/v/proj.
const VAE_MATMUL_CFG: MatMulConfig = MatMulConfig {
    bm: 64,
    bn: 64,
    bk: 16,
    tm: 4,
    tn: 4,
    b_nmajor: false,
};

/// Small-N conv3d tile: conv_out (cout=3) / latent convs (cout=16/32). The
/// default bm=64 tile idles most rows when cout is tiny.
const CONV3D_TILE_SMALL_N: Conv3dConfig = Conv3dConfig {
    bm: 4,
    bn: 128,
    bk: 32,
    tm: 1,
    tn: 2,
};

/// Every WGSL pipeline the Wan VAE forward dispatches. Shared encoder + decoder.
pub struct WanVaePipelines {
    /// Activation storage dtype: F16 when the device has `SHADER_F16`, else F32.
    pub act_dtype: ActDtype,
    pub act_size: u64,
    pub conv3d: Conv3dPipeline,
    pub conv3d_small_n: Conv3dPipeline,
    pub conv2d: Conv2dPipeline,
    /// Time-axis concat for `feat_cache` assembly (see `concat_time` op).
    pub concat_time: thinfer_core::backend::WgpuPipeline,
    pub rmsnorm3d: thinfer_core::backend::WgpuPipeline,
    pub silu: thinfer_core::backend::WgpuPipeline,
    pub upsample: thinfer_core::backend::WgpuPipeline,
    pub add: thinfer_core::backend::WgpuPipeline,
    pub matmul: thinfer_core::backend::WgpuPipeline,
    pub matmul_op: MatMulF32,
    pub bcast_add: thinfer_core::backend::WgpuPipeline,
    pub sdpa_large_d: thinfer_core::backend::WgpuPipeline,
    pub transpose12: thinfer_core::backend::WgpuPipeline,
}

impl WanVaePipelines {
    pub async fn compile(backend: &WgpuBackend) -> Result<Self, WgpuError> {
        use thinfer_core::backend::Backend;
        let act_dtype = if backend.supports_shader_f16() {
            ActDtype::F16
        } else {
            ActDtype::F32
        };
        let cfg = &WgslConfig {
            bf16_quant_writes: false,
            act_dtype,
            weight_dtype: thinfer_core::ops::WeightDtype::Bf16,
        };
        Ok(Self {
            act_dtype,
            act_size: act_dtype.bytes_per_elem(),
            conv3d: Conv3dPipeline::compile(backend, "wan_vae_conv3d", cfg, Conv3dConfig::DEFAULT)
                .await?,
            conv3d_small_n: Conv3dPipeline::compile(
                backend,
                "wan_vae_conv3d_small_n",
                cfg,
                CONV3D_TILE_SMALL_N,
            )
            .await?,
            conv2d: Conv2dPipeline::compile(backend, "wan_vae_conv2d", cfg, Conv2dConfig::DEFAULT)
                .await?,
            concat_time: backend
                .create_pipeline(
                    "wan_vae_concat_time",
                    &<ConcatTimeF32 as ConcatTimeOp>::wgsl(cfg),
                    "main",
                    <ConcatTimeF32 as ConcatTimeOp>::layout(),
                )
                .await?,
            rmsnorm3d: backend
                .create_pipeline(
                    "wan_vae_rmsnorm3d",
                    &<RmsNorm3dF32 as RmsNorm3dOp>::wgsl(cfg),
                    "main",
                    <RmsNorm3dF32 as RmsNorm3dOp>::layout(),
                )
                .await?,
            silu: backend
                .create_pipeline(
                    "wan_vae_silu",
                    SiluF32::wgsl(cfg),
                    "main",
                    SiluF32::layout(),
                )
                .await?,
            upsample: backend
                .create_pipeline(
                    "wan_vae_upsample",
                    <Upsample2dNearestF32 as Upsample2dNearestOp>::wgsl(cfg),
                    "main",
                    <Upsample2dNearestF32 as Upsample2dNearestOp>::layout(),
                )
                .await?,
            add: backend
                .create_pipeline("wan_vae_add", AddF32::wgsl(cfg), "main", AddF32::layout())
                .await?,
            matmul: {
                let op = MatMulF32::new(VAE_MATMUL_CFG);
                backend
                    .create_pipeline(
                        "wan_vae_matmul",
                        &op.wgsl(cfg),
                        "main",
                        <MatMulF32 as MatmulOp>::layout(),
                    )
                    .await?
            },
            matmul_op: MatMulF32::new(VAE_MATMUL_CFG),
            bcast_add: backend
                .create_pipeline(
                    "wan_vae_bcast_add",
                    <BcastAddF32 as BcastAddOp>::wgsl(cfg),
                    "main",
                    <BcastAddF32 as BcastAddOp>::layout(),
                )
                .await?,
            sdpa_large_d: backend
                .create_pipeline(
                    "wan_vae_sdpa_large_d",
                    <SdpaF32LargeD as SdpaOp>::wgsl(cfg),
                    "main",
                    <SdpaF32LargeD as SdpaOp>::layout(),
                )
                .await?,
            transpose12: backend
                .create_pipeline(
                    "wan_vae_transpose12",
                    <Transpose12F32 as Transpose12Op>::wgsl(cfg),
                    "main",
                    <Transpose12F32 as Transpose12Op>::layout(),
                )
                .await?,
        })
    }

    pub fn kernel_keys() -> [KernelKey; 11] {
        let kk = |id: &'static str| KernelKey {
            kernel_id: id,
            hint: String::new(),
        };
        [
            kk(<Conv3dF32 as Conv3dOp>::KERNEL_ID),
            kk(<Conv2dF32 as Conv2dOp>::KERNEL_ID),
            kk(<ConcatTimeF32 as ConcatTimeOp>::KERNEL_ID),
            kk(<RmsNorm3dF32 as RmsNorm3dOp>::KERNEL_ID),
            kk(<SiluF32 as Op>::KERNEL_ID),
            kk(<Upsample2dNearestF32 as Upsample2dNearestOp>::KERNEL_ID),
            kk(<AddF32 as Op>::KERNEL_ID),
            kk(<MatMulF32 as MatmulOp>::KERNEL_ID),
            kk(<BcastAddF32 as BcastAddOp>::KERNEL_ID),
            kk(<SdpaF32LargeD as SdpaOp>::KERNEL_ID),
            kk(<Transpose12F32 as Transpose12Op>::KERNEL_ID),
        ]
    }
}

// ============================================================================
// Forward driver (decoder). Encoder lands in a follow-up.
// ============================================================================
//
// The decoder runs `_decode` (`autoencoder_kl_wan.py`) one latent frame at a
// time: each frame is its own `BatchScope` + submit (the VAE single-heavy-submit
// rule), and causality is carried between frames by a host-side `feat_cache`
// (one entry per `kt=3` causal conv, assembled in front of the next frame's
// conv with `concat_time`). All activations are NCTHW row-major; the only
// excursions are inside the spatial resample (per-frame NTCHW for the 2D conv)
// and the mid-block attention (token layout for sdpa). B is fixed at 1 (one
// video), which lets the attention channel-split use contiguous sub-ranges.

/// NCTHW activation shape carried through the decoder.
#[derive(Clone, Copy, Debug)]
struct Shape5 {
    b: u32,
    c: u32,
    t: u32,
    h: u32,
    w: u32,
}

impl Shape5 {
    fn elems(&self) -> u32 {
        self.b * self.c * self.t * self.h * self.w
    }
    fn bytes(&self, act_size: u64) -> u64 {
        self.elems() as u64 * act_size
    }
    /// Positions for `rmsnorm3d` (one thread per `(b, t, h, w)`).
    fn n_pos(&self) -> u32 {
        self.b * self.t * self.h * self.w
    }
    /// Per-channel time stride for `rmsnorm3d` (`T*H*W`).
    fn thw(&self) -> u32 {
        self.t * self.h * self.w
    }
}

/// Per-submit work cap for the decoder's `ScopePacker` (NOT the VRAM ceiling).
/// At native res a single full-res frame is ~2.3s of GPU work; one submit that
/// long trips the Windows 2s GPU watchdog (TDR). The packer accumulates phase
/// outputs into a scope until their bytes exceed this cap, then cuts a fresh
/// submit, so each submit stays well under the watchdog. Sized so small
/// (parity, 64px) frames stay a single submit -- bit-identical to no packing --
/// while full-res resnets (~400 MB output each) land one-per-submit.
const VAE_SUBMIT_BUDGET_BYTES: u64 = 128 * 1024 * 1024;

// Spatial tiling of the decode (native res). A full-frame decode keeps the
// whole activation pyramid + per-conv `FeatCache` live at output res, which
// overruns the VRAM budget at 540P+. The arbiter budget can only reclaim *idle*
// buffers; the decode's caches + chain are all simultaneously *live*, so nothing
// is reclaimable and a full frame overshoots the budget straight to a hardware
// OOM. Tiling is the only lever that shrinks the live set itself: the decode
// runs in overlapping latent tiles whose size is derived from the budget
// (`vae_tile_dims`), bounding the per-tile working set -- and thus peak VRAM --
// regardless of output resolution. Adjacent tiles overlap by a halo that absorbs
// each tile's interior zero-pad seam error and is feather-blended away. When
// both latent dims fit one tile (parity res), the plan is a single full tile
// with unit weights -> bit-identical to the untiled path. Sizes are in LATENT
// pixels (output is 8x).

/// Largest square latent tile (and its overlap) whose decode working set fits
/// the VRAM budget. The live set is ~linear in tile AREA and in act-dtype bytes.
/// Calibrated from a 27x27-latent f16 tile that ran at ~2.3 GiB workspace live
/// -> ~3.2 MiB per latent-px^2 at f16. (A 64x64 tile only tells us it exceeded
/// the card, an unusable lower bound -- don't calibrate off a crash.) Solve for
/// the largest tile using a fraction of the budget; the remainder (~1/4) covers
/// resident weights + I/O staging + fragmentation. An unbounded budget falls
/// back to a 2 GiB reference so we still tile sanely.
fn vae_tile_dims(budget_bytes: u64, act_size: u64) -> (u32, u32) {
    /// Live workspace bytes per latent-px^2 at f16 (measured, rounded up).
    const BYTES_PER_LAT_AREA_F16: u64 = 3_200_000;
    /// Fraction of budget (3/4) spent on the tile working set.
    const SAFETY_NUM: u64 = 3;
    const SAFETY_DEN: u64 = 4;
    /// Upper cap so a huge budget doesn't pick a TDR-prone megatile.
    const TILE_MAX: u32 = 96;
    /// Floor: an 8-latent (64px) tile fits any usable device.
    const TILE_MIN: u32 = 8;

    let budget = if budget_bytes == u64::MAX {
        2 * 1024 * 1024 * 1024
    } else {
        budget_bytes
    };
    let per_area = (BYTES_PER_LAT_AREA_F16 * act_size / 2).max(1);
    let max_area = (budget / SAFETY_DEN * SAFETY_NUM) / per_area;
    let tile = ((max_area as f64).sqrt() as u32).clamp(TILE_MIN, TILE_MAX);
    // Overlap ~1/4 of the tile (>=4 latent / 32px) for a smooth feather seam.
    let overlap = (tile / 4).max(4);
    (tile, overlap)
}

/// Tiles covering `[0, n)` latent pixels along one axis: `(start, extent)` pairs
/// stepping by `tile - overlap`, each extent capped at `tile` and clamped to the
/// end. A single `(0, n)` tile when `n <= tile` (the parity-res fast path).
fn plan_tiles(n: u32, tile: u32, overlap: u32) -> Vec<(u32, u32)> {
    if n <= tile {
        return vec![(0, n)];
    }
    let step = tile - overlap;
    let mut tiles = Vec::new();
    let mut start = 0;
    loop {
        let ext = (n - start).min(tile);
        tiles.push((start, ext));
        if start + ext >= n {
            break;
        }
        start += step;
    }
    tiles
}

/// Per-output-pixel feather weights along one tiled axis (length `ext * 8`).
/// Linearly ramps 0->1 over the `overlap`-wide band on any edge that abuts a
/// neighbor (`has_prev`/`has_next`) and holds 1 elsewhere; true image borders
/// get no ramp. Two adjacent tiles' complementary ramps sum to ~1 over their
/// shared overlap (partition of unity); the small floor keeps `wsum` positive.
fn feather_1d(ext: u32, overlap: u32, has_prev: bool, has_next: bool) -> Vec<f32> {
    let len = (ext * 8) as usize;
    let ramp = ((overlap * 8) as usize).min(len) as f32;
    (0..len)
        .map(|i| {
            let mut w = 1.0f32;
            if has_prev {
                w = w.min((i as f32 + 0.5) / ramp);
            }
            if has_next {
                w = w.min(((len - i) as f32 - 0.5) / ramp);
            }
            w.clamp(0.0, 1.0).max(1e-4)
        })
        .collect()
}

/// Activation bytes of a `[1, c, t, h, w]` tensor: the `ScopePacker` phase peak.
fn peak_bytes(c: u32, t: u32, h: u32, w: u32, asz: u64) -> u64 {
    c as u64 * t as u64 * h as u64 * w as u64 * asz
}

#[derive(Debug)]
pub enum WanVaeError {
    Wgpu(WgpuError),
}

impl From<WgpuError> for WanVaeError {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}

// ---------------------------------------------------------------------------
// feat_cache
// ---------------------------------------------------------------------------

/// One `feat_cache` slot. Mirrors the per-conv entry in `_feat_map`: `None`
/// before the first frame, `Rep` for the one-shot `upsample3d` zero-prefix
/// marker, or the trailing `t` (<= 2) frames of a conv's input held in a
/// workspace buffer that survives across per-frame submits.
enum FeatEntry {
    None,
    Rep,
    Frames { buf: WsBuf<WgpuBackend>, t: u32 },
}

/// Per-decode causal-conv cache. `idx` resets to 0 at the start of each frame
/// and is advanced by exactly the convs that pyref advances `feat_idx` for
/// (conv_in, every resnet conv1/conv2, upsample3d time_conv, conv_out - never
/// conv_shortcut / quant / qkv / proj). Entries grow lazily on frame 0 and are
/// then indexed in lockstep.
struct FeatCache {
    entries: Vec<FeatEntry>,
    idx: usize,
}

impl FeatCache {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            idx: 0,
        }
    }
    fn reset(&mut self) {
        self.idx = 0;
    }
    /// Claim the next slot, growing with `None` on the first frame.
    fn next(&mut self) -> usize {
        let i = self.idx;
        if i == self.entries.len() {
            self.entries.push(FeatEntry::None);
        }
        self.idx += 1;
        i
    }
}

// ---------------------------------------------------------------------------
// uniform builders
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn conv3d_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    inb: (u32, u32, u32, u32, u32), // (b, cin, t_in, h_in, w_in)
    out: (u32, u32, u32, u32),      // (cout, t_out, h_out, w_out)
    ker: (u32, u32, u32),           // (kt, kh, kw)
    pad: (u32, u32, u32),           // (pad_t, pad_h, pad_w)
    stride: (u32, u32, u32),        // (stride_t, stride_h, stride_w)
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let fields: [u32; 20] = [
        inb.0, inb.1, out.0, inb.2, inb.3, inb.4, out.1, out.2, out.3, ker.0, ker.1, ker.2, pad.0,
        pad.1, pad.2, stride.0, stride.1, stride.2, 0, 0,
    ];
    let mut bytes = [0u8; 80];
    for (i, v) in fields.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    scope.write_uniform(&bytes)
}

#[allow(clippy::too_many_arguments)]
fn concat_time_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    b: u32,
    c: u32,
    h: u32,
    w: u32,
    a_t: u32,
    b_t: u32,
    a_start: u32,
    a_count: u32,
    b_start: u32,
    b_count: u32,
    a_zero: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let fields: [u32; 12] = [
        b, c, h, w, a_t, b_t, a_start, a_count, b_start, b_count, a_zero, 0,
    ];
    let mut bytes = [0u8; 48];
    for (i, v) in fields.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    scope.write_uniform(&bytes)
}

fn rmsnorm3d_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    n_pos: u32,
    channels: u32,
    stride: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&n_pos.to_le_bytes());
    bytes[4..8].copy_from_slice(&channels.to_le_bytes());
    bytes[8..12].copy_from_slice(&stride.to_le_bytes());
    scope.write_uniform(&bytes)
}

fn conv2d_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    inb: (u32, u32, u32, u32),           // (b, cin, h_in, w_in)
    out: (u32, u32, u32),                // (cout, h_out, w_out)
    ker: (u32, u32, u32, u32, u32, u32), // (kh, kw, pad_h, pad_w, stride_h, stride_w)
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let fields: [u32; 16] = [
        inb.0, inb.1, out.0, inb.2, inb.3, out.1, out.2, ker.0, ker.1, ker.2, ker.3, ker.4, ker.5,
        0, 0, 0,
    ];
    let mut bytes = [0u8; 64];
    for (i, v) in fields.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    scope.write_uniform(&bytes)
}

fn upsample_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    b: u32,
    c: u32,
    h_in: u32,
    w_in: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&b.to_le_bytes());
    bytes[4..8].copy_from_slice(&c.to_le_bytes());
    bytes[8..12].copy_from_slice(&h_in.to_le_bytes());
    bytes[12..16].copy_from_slice(&w_in.to_le_bytes());
    scope.write_uniform(&bytes)
}

fn transpose12_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    d0: u32,
    d1: u32,
    d2: u32,
    d3: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    scope.u32x4_uniform(d0, d1, d2, d3)
}

#[allow(clippy::too_many_arguments)]
fn sdpa_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    b: u32,
    h_q: u32,
    h_kv: u32,
    s_q: u32,
    s_k: u32,
    d: u32,
    scale: f32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&b.to_le_bytes());
    bytes[4..8].copy_from_slice(&h_q.to_le_bytes());
    bytes[8..12].copy_from_slice(&h_kv.to_le_bytes());
    bytes[12..16].copy_from_slice(&s_q.to_le_bytes());
    bytes[16..20].copy_from_slice(&s_k.to_le_bytes());
    bytes[20..24].copy_from_slice(&d.to_le_bytes());
    bytes[24..28].copy_from_slice(&scale.to_le_bytes());
    // has_mask = 0.
    scope.write_uniform(&bytes)
}

// ---------------------------------------------------------------------------
// low-level op wrappers (all run inside one frame's BatchScope)
// ---------------------------------------------------------------------------

/// Pick the conv3d tile regime: tiny-`cout` convs (conv_out=3, latent=16/32)
/// idle most rows of the default bm=64 tile.
fn conv3d_pipeline(pl: &WanVaePipelines, cout: u32) -> &Conv3dPipeline {
    if cout <= 32 {
        &pl.conv3d_small_n
    } else {
        &pl.conv3d
    }
}

/// `conv3d(x, w) + bias` with explicit geometry. Front-pads time by `pad_t`
/// (causal), symmetric `pad_h`/`pad_w`. Output NCTHW.
#[allow(clippy::too_many_arguments)]
fn conv3d_run<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pl: &WanVaePipelines,
    x: BatchBuf<'wsp>,
    in_shape: Shape5,
    w: &'wsp ConvBufs,
    cout: u32,
    ker: (u32, u32, u32),
    pad: (u32, u32, u32),
    stride: (u32, u32, u32),
) -> Result<(BatchBuf<'wsp>, Shape5), WgpuError> {
    let (kt, kh, kw) = ker;
    let (pad_t, pad_h, pad_w) = pad;
    let (st, sh, sw) = stride;
    let t_out = (in_shape.t + pad_t - kt) / st + 1;
    let h_out = (in_shape.h + 2 * pad_h - kh) / sh + 1;
    let w_out = (in_shape.w + 2 * pad_w - kw) / sw + 1;
    let out_shape = Shape5 {
        b: in_shape.b,
        c: cout,
        t: t_out,
        h: h_out,
        w: w_out,
    };
    let out = scope.alloc(out_shape.bytes(pl.act_size))?;
    let u = conv3d_uniform(
        scope,
        (in_shape.b, in_shape.c, in_shape.t, in_shape.h, in_shape.w),
        (cout, t_out, h_out, w_out),
        ker,
        pad,
        stride,
    )?;
    let wb = scope.import(&w.weight);
    let bb = scope.import(&w.bias);
    let conv = conv3d_pipeline(pl, cout);
    scope.conv3d(
        &conv.pipeline,
        &conv.op,
        x,
        wb,
        bb,
        u,
        out,
        cout,
        t_out * h_out * w_out,
        in_shape.b,
    )?;
    Ok((out, out_shape))
}

/// 1x1x1 pointwise conv (post_quant, conv_shortcut, attention qkv/proj). Never
/// cached (kt=1 has no causal front pad).
fn conv3d_1x1x1<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pl: &WanVaePipelines,
    x: BatchBuf<'wsp>,
    in_shape: Shape5,
    w: &'wsp ConvBufs,
    cout: u32,
) -> Result<(BatchBuf<'wsp>, Shape5), WgpuError> {
    conv3d_run(
        scope,
        pl,
        x,
        in_shape,
        w,
        cout,
        (1, 1, 1),
        (0, 0, 0),
        (1, 1, 1),
    )
}

fn rmsnorm3d_run<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pl: &WanVaePipelines,
    x: BatchBuf<'wsp>,
    shape: Shape5,
    w: &'wsp RmsBufs,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let out = scope.alloc(shape.bytes(pl.act_size))?;
    let u = rmsnorm3d_uniform(scope, shape.n_pos(), shape.c, shape.thw())?;
    let gamma = scope.import(&w.gamma);
    scope.rmsnorm3d::<RmsNorm3dF32>(&pl.rmsnorm3d, x, gamma, u, out, shape.n_pos())?;
    Ok(out)
}

fn silu_run<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pl: &WanVaePipelines,
    x: BatchBuf<'wsp>,
    shape: Shape5,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let out = scope.alloc(shape.bytes(pl.act_size))?;
    scope.dispatch_op::<SiluF32>(&pl.silu, &[x], out)?;
    Ok(out)
}

fn add_run<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pl: &WanVaePipelines,
    a: BatchBuf<'wsp>,
    b: BatchBuf<'wsp>,
    shape: Shape5,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let out = scope.alloc(shape.bytes(pl.act_size))?;
    scope.dispatch_op::<AddF32>(&pl.add, &[a, b], out)?;
    Ok(out)
}

/// `transpose12` over a flat `[d0, d1, d2, d3]` view: `out[d0,d2,d1,d3] =
/// in[d0,d1,d2,d3]`.
fn transpose12_run<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pl: &WanVaePipelines,
    x: BatchBuf<'wsp>,
    d0: u32,
    d1: u32,
    d2: u32,
    d3: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let out = scope.alloc((d0 * d1 * d2 * d3) as u64 * pl.act_size)?;
    let u = transpose12_uniform(scope, d0, d1, d2, d3)?;
    scope.transpose12::<Transpose12F32>(&pl.transpose12, x, u, out, d0 * d1 * d2 * d3)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// causal conv with feat_cache
// ---------------------------------------------------------------------------

/// A `kt=3` causal conv (`conv_in`, resnet conv1/conv2, conv_out) with
/// `feat_cache`. Assembles the previous frame's trailing input frames in front
/// of `x` (`concat_time`), runs the conv so `Tout == Tin`, then stashes this
/// frame's trailing frames for the next call. `kh`/`kw` are 3 (symmetric pad 1)
/// or 1 (time-only, pad 0). `retire` collects displaced cache buffers so they
/// outlive the submit that still reads them.
#[allow(clippy::too_many_arguments)]
fn causal_conv3d<'wsp>(
    workspace: &'wsp Workspace<WgpuBackend>,
    scope: &BatchScope<'wsp, WgpuBackend>,
    pl: &WanVaePipelines,
    fc: &mut FeatCache,
    retire: &mut Vec<FeatEntry>,
    x: BatchBuf<'wsp>,
    in_shape: Shape5,
    w: &'wsp ConvBufs,
    cout: u32,
    kh: u32,
    kw: u32,
) -> Result<(BatchBuf<'wsp>, Shape5), WgpuError> {
    let idx = fc.next();
    let old_t = match &fc.entries[idx] {
        FeatEntry::Frames { t, .. } => *t,
        _ => 0,
    };
    let tin = in_shape.t;
    let cin = in_shape.c;
    let (h, ww) = (in_shape.h, in_shape.w);
    let asz = pl.act_size;

    // Assemble [old_cache ++ x] in front of the conv (or just x on frame 0).
    let (assembled, assembled_t) = if old_t > 0 {
        let old_ref = match &fc.entries[idx] {
            FeatEntry::Frames { buf, .. } => buf.as_buf_ref(),
            _ => unreachable!(),
        };
        let a = scope.import_copy(old_ref);
        let asm = Shape5 {
            b: in_shape.b,
            c: cin,
            t: old_t + tin,
            h,
            w: ww,
        };
        let out = scope.alloc(asm.bytes(asz))?;
        let u = concat_time_uniform(
            scope, in_shape.b, cin, h, ww, old_t, tin, 0, old_t, 0, tin, 0,
        )?;
        scope.concat_time::<ConcatTimeF32>(&pl.concat_time, a, x, u, out, asm.elems())?;
        (out, old_t + tin)
    } else {
        (x, tin)
    };
    let assembled_shape = Shape5 {
        b: in_shape.b,
        c: cin,
        t: assembled_t,
        h,
        w: ww,
    };
    // kt=3 with front pad (2 - old_t) over the assembled tensor keeps Tout=Tin.
    let pad_t = 2 - old_t;
    let pad_h = (kh - 1) / 2;
    let pad_w = (kw - 1) / 2;
    let (out, out_shape) = conv3d_run(
        scope,
        pl,
        assembled,
        assembled_shape,
        w,
        cout,
        (3, kh, kw),
        (pad_t, pad_h, pad_w),
        (1, 1, 1),
    )?;

    // Build next cache = trailing <=2 frames of THIS conv input `x`, padding
    // with the previous cache's last frame when only one frame is available.
    let new_t = if tin >= 2 || old_t > 0 { 2 } else { 1 };
    let cache_shape = Shape5 {
        b: in_shape.b,
        c: cin,
        t: new_t,
        h,
        w: ww,
    };
    let new_buf = workspace.alloc(cache_shape.bytes(asz))?;
    let new_ref = new_buf.as_buf_ref();
    let nb = scope.import_copy(new_ref);
    if tin >= 2 {
        // A = x tail (last 2 frames); B unused (bind x).
        let u = concat_time_uniform(scope, in_shape.b, cin, h, ww, tin, tin, tin - 2, 2, 0, 0, 0)?;
        scope.concat_time::<ConcatTimeF32>(&pl.concat_time, x, x, u, nb, cache_shape.elems())?;
    } else if old_t > 0 {
        // A = old cache last frame, B = x (single frame).
        let old_ref = match &fc.entries[idx] {
            FeatEntry::Frames { buf, .. } => buf.as_buf_ref(),
            _ => unreachable!(),
        };
        let a = scope.import_copy(old_ref);
        let u = concat_time_uniform(
            scope,
            in_shape.b,
            cin,
            h,
            ww,
            old_t,
            tin,
            old_t - 1,
            1,
            0,
            1,
            0,
        )?;
        scope.concat_time::<ConcatTimeF32>(&pl.concat_time, a, x, u, nb, cache_shape.elems())?;
    } else {
        // First frame, single input frame: cache just that frame.
        let u = concat_time_uniform(scope, in_shape.b, cin, h, ww, tin, tin, 0, 1, 0, 0, 0)?;
        scope.concat_time::<ConcatTimeF32>(&pl.concat_time, x, x, u, nb, cache_shape.elems())?;
    }

    retire.push(std::mem::replace(
        &mut fc.entries[idx],
        FeatEntry::Frames {
            buf: new_buf,
            t: new_t,
        },
    ));
    Ok((out, out_shape))
}

// ---------------------------------------------------------------------------
// resnet / attention / upsampler / up_block
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn resnet_forward<'wsp>(
    workspace: &'wsp Workspace<WgpuBackend>,
    scope: &BatchScope<'wsp, WgpuBackend>,
    pl: &WanVaePipelines,
    fc: &mut FeatCache,
    retire: &mut Vec<FeatEntry>,
    x: BatchBuf<'wsp>,
    in_shape: Shape5,
    cout: u32,
    w: &'wsp ResnetBufs,
) -> Result<(BatchBuf<'wsp>, Shape5), WgpuError> {
    // Shortcut (1x1x1 conv when channels change, else identity).
    let (skip, _) = match &w.conv_shortcut {
        Some(cs) => conv3d_1x1x1(scope, pl, x, in_shape, cs, cout)?,
        None => (x, in_shape),
    };
    let h0 = rmsnorm3d_run(scope, pl, x, in_shape, &w.norm1)?;
    let h1 = silu_run(scope, pl, h0, in_shape)?;
    let (h2, sh2) = causal_conv3d(
        workspace, scope, pl, fc, retire, h1, in_shape, &w.conv1, cout, 3, 3,
    )?;
    let h3 = rmsnorm3d_run(scope, pl, h2, sh2, &w.norm2)?;
    let h4 = silu_run(scope, pl, h3, sh2)?;
    let (h5, sh5) = causal_conv3d(
        workspace, scope, pl, fc, retire, h4, sh2, &w.conv2, cout, 3, 3,
    )?;
    let out = add_run(scope, pl, skip, h5, sh5)?;
    Ok((out, sh5))
}

/// Mid-block single-head spatial self-attention (`WanAttentionBlock`). Decoder
/// mid always runs at `T=1` (temporal upsampling happens later, in the up
/// blocks), and `B=1`, so the channel split feeds contiguous sub-ranges. qkv /
/// proj are 1x1 convs over NCTHW; sdpa runs over `H*W` tokens with `D=C`.
fn mid_attention_forward<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pl: &WanVaePipelines,
    x: BatchBuf<'wsp>,
    shape: Shape5,
    w: &'wsp AttnBufs,
) -> Result<(BatchBuf<'wsp>, Shape5), WgpuError> {
    debug_assert_eq!(shape.b, 1, "mid attention assumes B=1");
    debug_assert_eq!(shape.t, 1, "decoder mid attention runs at T=1");
    let c = shape.c;
    let hw = shape.h * shape.w;
    let asz = pl.act_size;

    let normed = rmsnorm3d_run(scope, pl, x, shape, &w.norm)?;
    let (qkv, _) = conv3d_1x1x1(scope, pl, normed, shape, &w.to_qkv, 3 * c)?;

    // Split the 3C channels into contiguous q/k/v blocks (B=T=1) and turn each
    // [1, C, H*W] block into [1, H*W, C] sdpa tokens via transpose12.
    let block_bytes = (c * hw) as u64 * asz;
    let mut tokens = [None, None, None];
    for (g, slot) in tokens.iter_mut().enumerate() {
        let blk = scope.alloc(block_bytes)?;
        scope.copy_buffer_to_buffer(qkv, g as u64 * block_bytes, blk, 0, block_bytes)?;
        // [1, C, H*W] -> [1, H*W, C].
        *slot = Some(transpose12_run(scope, pl, blk, 1, c, hw, 1)?);
    }
    let q = tokens[0].unwrap();
    let k = tokens[1].unwrap();
    let v = tokens[2].unwrap();

    let mask = scope.write_uniform(&0f32.to_le_bytes())?;
    let attn = scope.alloc(block_bytes)?;
    let scale = 1.0_f32 / (c as f32).sqrt();
    let u = sdpa_uniform(scope, 1, 1, 1, hw, hw, c, scale)?;
    scope.sdpa::<SdpaF32LargeD>(&pl.sdpa_large_d, q, k, v, mask, u, attn, 1, hw, 1)?;

    // [1, H*W, C] -> [1, C, H*W] -> NCTHW, then proj (1x1) + residual.
    let restored = transpose12_run(scope, pl, attn, 1, hw, c, 1)?;
    let (proj, _) = conv3d_1x1x1(scope, pl, restored, shape, &w.proj, c)?;
    let out = add_run(scope, pl, x, proj, shape)?;
    Ok((out, shape))
}

/// Per-frame spatial resample (`nearest-exact` 2x + 3x3 conv, `out -> out/2`),
/// applied over every time frame. NCTHW in, NCTHW out with `H,W` doubled. The
/// 2D conv runs in NTCHW (batch `= B*T`); `nearest-exact` equals plain
/// `nearest` for integer 2x scale, so `Upsample2dNearestF32` is exact here.
fn spatial_resample<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pl: &WanVaePipelines,
    x: BatchBuf<'wsp>,
    shape: Shape5,
    w: &'wsp ConvBufs,
    cout: u32,
) -> Result<(BatchBuf<'wsp>, Shape5), WgpuError> {
    let (b, c, t, h, ww) = (shape.b, shape.c, shape.t, shape.h, shape.w);
    let bt = b * t;
    let hw = h * ww;
    // NCTHW -> NTCHW.
    let ntchw = transpose12_run(scope, pl, x, b, c, t, hw)?;

    // nearest 2x over [B*T, C, H, W].
    let up_h = h * 2;
    let up_w = ww * 2;
    let up_shape = Shape5 {
        b: bt,
        c,
        t: 1,
        h: up_h,
        w: up_w,
    };
    let up = scope.alloc(up_shape.bytes(pl.act_size))?;
    let uu = upsample_uniform(scope, bt, c, h, ww)?;
    scope.upsample2d_nearest::<Upsample2dNearestF32>(
        &pl.upsample,
        ntchw,
        uu,
        up,
        up_shape.elems(),
    )?;

    // 3x3 conv (pad 1, stride 1): out channels = cout, spatial unchanged.
    let conv_shape = Shape5 {
        b: bt,
        c: cout,
        t: 1,
        h: up_h,
        w: up_w,
    };
    let cout_buf = scope.alloc(conv_shape.bytes(pl.act_size))?;
    let cu = conv2d_uniform(
        scope,
        (bt, c, up_h, up_w),
        (cout, up_h, up_w),
        (3, 3, 1, 1, 1, 1),
    )?;
    let wb = scope.import(&w.weight);
    let bb = scope.import(&w.bias);
    scope.conv2d(
        &pl.conv2d.pipeline,
        &pl.conv2d.op,
        up,
        wb,
        bb,
        cu,
        cout_buf,
        cout,
        up_h * up_w,
        bt,
    )?;

    // NTCHW -> NCTHW.
    let out = transpose12_run(scope, pl, cout_buf, b, t, cout, up_h * up_w)?;
    Ok((
        out,
        Shape5 {
            b,
            c: cout,
            t,
            h: up_h,
            w: up_w,
        },
    ))
}

/// Decoder up_block: `num_res_blocks + 1` resnets then an optional upsampler.
/// `temporal` selects the `upsample3d` path (time_conv + 2x time interleave)
/// over `upsample2d` (spatial only). `out_c` is the block output channels;
/// the upsampler's spatial conv then halves to `out_c/2`. Each resnet and the
/// upsampler is a `ScopePacker` phase: at native res they land in separate
/// submits (watchdog-safe), at parity res they collapse into one.
#[allow(clippy::too_many_arguments)]
async fn up_block_forward<'wsp>(
    workspace: &'wsp Workspace<WgpuBackend>,
    packer: &mut ScopePacker<'wsp, WgpuBackend>,
    pl: &WanVaePipelines,
    fc: &mut FeatCache,
    retire: &mut Vec<FeatEntry>,
    mut x: BatchBuf<'wsp>,
    mut shape: Shape5,
    out_c: u32,
    temporal: bool,
    w: &'wsp UpBlockBufs,
    asz: u64,
) -> Result<(BatchBuf<'wsp>, Shape5), WgpuError> {
    for resnet in &w.resnets {
        // Resnets preserve t/h/w; only the channel count moves to out_c.
        let peak = peak_bytes(out_c, shape.t, shape.h, shape.w, asz);
        x = packer.advance(&[x], peak).await?.pop().unwrap();
        let (y, sh) = resnet_forward(
            workspace,
            packer.scope(),
            pl,
            fc,
            retire,
            x,
            shape,
            out_c,
            resnet,
        )?;
        x = y;
        shape = sh;
    }
    let Some(up) = &w.upsampler else {
        return Ok((x, shape));
    };

    // Upsampler phase: spatial 2x always, time 2x when temporal; out_c/2 chans.
    let up_t = if temporal { shape.t * 2 } else { shape.t };
    let up_peak = peak_bytes(out_c / 2, up_t, shape.h * 2, shape.w * 2, asz);
    x = packer.advance(&[x], up_peak).await?.pop().unwrap();
    let scope = packer.scope();

    // upsample3d: causal time_conv (out_c -> 2*out_c) then 2x time interleave.
    // First frame is a one-shot "Rep" marker: no time_conv, no doubling.
    if temporal {
        let tc = up
            .time_conv
            .as_ref()
            .expect("temporal upsampler has time_conv");
        let idx = fc.next();
        // Extract the slot kind without holding a borrow across the mutation.
        let (is_first, old_t) = match &fc.entries[idx] {
            FeatEntry::None => (true, 0),
            FeatEntry::Rep => (false, 0), // Rep -> conv with full front pad (no prepend).
            FeatEntry::Frames { t, .. } => (false, *t),
        };
        if is_first {
            // First frame: mark Rep, skip time_conv + doubling entirely.
            fc.entries[idx] = FeatEntry::Rep;
        } else {
            {
                let cin = shape.c;
                let (h, ww) = (shape.h, shape.w);
                let tin = shape.t;
                let asz = pl.act_size;

                // Assemble cached frames in front (Rep -> none; front pad covers it).
                let (assembled, assembled_t) = if old_t > 0 {
                    let old_ref = match &fc.entries[idx] {
                        FeatEntry::Frames { buf, .. } => buf.as_buf_ref(),
                        _ => unreachable!(),
                    };
                    let a = scope.import_copy(old_ref);
                    let asm = Shape5 {
                        b: shape.b,
                        c: cin,
                        t: old_t + tin,
                        h,
                        w: ww,
                    };
                    let out = scope.alloc(asm.bytes(asz))?;
                    let u = concat_time_uniform(
                        scope, shape.b, cin, h, ww, old_t, tin, 0, old_t, 0, tin, 0,
                    )?;
                    scope.concat_time::<ConcatTimeF32>(
                        &pl.concat_time,
                        a,
                        x,
                        u,
                        out,
                        asm.elems(),
                    )?;
                    (out, old_t + tin)
                } else {
                    (x, tin)
                };
                let asm_shape = Shape5 {
                    b: shape.b,
                    c: cin,
                    t: assembled_t,
                    h,
                    w: ww,
                };
                let pad_t = 2 - old_t;
                let (conv, conv_shape) = conv3d_run(
                    scope,
                    pl,
                    assembled,
                    asm_shape,
                    tc,
                    2 * cin,
                    (3, 1, 1),
                    (pad_t, 0, 0),
                    (1, 1, 1),
                )?;
                debug_assert_eq!(conv_shape.t, tin);

                // Build next cache = trailing <=2 frames of `x`, zero-filling
                // when only one frame is available (Rep) or padding with the
                // previous cache's last frame.
                let new_t = 2u32;
                let cache_shape = Shape5 {
                    b: shape.b,
                    c: cin,
                    t: new_t,
                    h,
                    w: ww,
                };
                let new_buf = workspace.alloc(cache_shape.bytes(asz))?;
                let nb = scope.import_copy(new_buf.as_buf_ref());
                if tin >= 2 {
                    let u = concat_time_uniform(
                        scope,
                        shape.b,
                        cin,
                        h,
                        ww,
                        tin,
                        tin,
                        tin - 2,
                        2,
                        0,
                        0,
                        0,
                    )?;
                    scope.concat_time::<ConcatTimeF32>(
                        &pl.concat_time,
                        x,
                        x,
                        u,
                        nb,
                        cache_shape.elems(),
                    )?;
                } else if old_t > 0 {
                    let old_ref = match &fc.entries[idx] {
                        FeatEntry::Frames { buf, .. } => buf.as_buf_ref(),
                        _ => unreachable!(),
                    };
                    let a = scope.import_copy(old_ref);
                    let u = concat_time_uniform(
                        scope,
                        shape.b,
                        cin,
                        h,
                        ww,
                        old_t,
                        tin,
                        old_t - 1,
                        1,
                        0,
                        1,
                        0,
                    )?;
                    scope.concat_time::<ConcatTimeF32>(
                        &pl.concat_time,
                        a,
                        x,
                        u,
                        nb,
                        cache_shape.elems(),
                    )?;
                } else {
                    // Rep: prepend a zero frame in front of the single input frame.
                    let u =
                        concat_time_uniform(scope, shape.b, cin, h, ww, tin, tin, 0, 1, 0, 1, 1)?;
                    scope.concat_time::<ConcatTimeF32>(
                        &pl.concat_time,
                        x,
                        x,
                        u,
                        nb,
                        cache_shape.elems(),
                    )?;
                }
                retire.push(std::mem::replace(
                    &mut fc.entries[idx],
                    FeatEntry::Frames {
                        buf: new_buf,
                        t: new_t,
                    },
                ));

                // 2x time interleave: conv out [b, 2C, t, hw] with channel =
                // g*C + ch maps to out[b, ch, 2t+g, hw]. Realized as two
                // transpose12s: split g out of the channel block, then move it
                // inside the time axis.
                let hw = h * ww;
                // [b, 2, C, t*hw] -> [b, C, 2, t*hw].
                let p = transpose12_run(scope, pl, conv, shape.b, 2, cin, tin * hw)?;
                // [b*C, 2, t, hw] -> [b*C, t, 2, hw] == [b, C, 2t, hw].
                let q = transpose12_run(scope, pl, p, shape.b * cin, 2, tin, hw)?;
                x = q;
                shape = Shape5 {
                    b: shape.b,
                    c: cin,
                    t: tin * 2,
                    h,
                    w: ww,
                };
            }
        }
    }

    // Spatial resample (every frame), out_c -> out_c/2, H/W doubled.
    spatial_resample(scope, pl, x, shape, &up.spatial_conv, out_c / 2)
}

// ---------------------------------------------------------------------------
// stage taps (e2e bisection)
// ---------------------------------------------------------------------------
//
// Each stage activation is a scope-local `BatchBuf` that does not survive the
// frame/chunk submit, so a tapped stage is first copied into a workspace buffer
// inside the submit (`persist_stage`) and read back afterwards (`read_into_f32`).
// To keep readback cost off the hot path, taps capture exactly one frame/chunk
// (`frame`/`chunk`); the final stage (`conv_out` for decode, `quant_conv` for
// encode) is already exposed by the `decode`/`encode` return value.

/// Per-stage decoder taps. `frame` selects which decoded frame to capture; each
/// `Some` sink is filled with that frame's stage output (NCTHW f32, row-major).
#[derive(Default)]
pub struct WanVaeDecodeTaps<'a> {
    pub frame: usize,
    pub post_quant: Option<&'a mut Vec<f32>>,
    pub conv_in: Option<&'a mut Vec<f32>>,
    pub mid: Option<&'a mut Vec<f32>>,
    /// One sink per up_block output (`len == up_blocks.len()` once filled).
    pub up_blocks: Option<&'a mut Vec<Vec<f32>>>,
}

/// Per-stage encoder taps. `chunk` selects which input chunk to capture; each
/// `Some` sink is filled with that chunk's stage output (NCTHW f32, row-major).
#[derive(Default)]
pub struct WanVaeEncodeTaps<'a> {
    pub chunk: usize,
    pub conv_in: Option<&'a mut Vec<f32>>,
    /// One sink per down_stage output (`len == down_stages.len()` once filled).
    pub down_stages: Option<&'a mut Vec<Vec<f32>>>,
    pub mid: Option<&'a mut Vec<f32>>,
    pub conv_out: Option<&'a mut Vec<f32>>,
}

/// Copy a scope-local stage activation into a workspace buffer that outlives the
/// submit, returning `(buffer, n_elems)` for a post-submit `read_into_f32`.
fn persist_stage<'wsp>(
    workspace: &'wsp Workspace<WgpuBackend>,
    scope: &BatchScope<'wsp, WgpuBackend>,
    buf: BatchBuf<'wsp>,
    shape: Shape5,
    asz: u64,
) -> Result<(WsBuf<WgpuBackend>, usize), WgpuError> {
    let bytes = shape.bytes(asz);
    let ws = workspace.alloc(bytes)?;
    let dst = scope.import_copy(ws.as_buf_ref());
    scope.copy_buffer_to_buffer(buf, 0, dst, 0, bytes)?;
    Ok((ws, shape.elems() as usize))
}

/// Read a persisted stage buffer into an optional tap sink (no-op when either is
/// `None`).
async fn read_stage(
    backend: &WgpuBackend,
    asz: u64,
    persisted: &Option<(WsBuf<WgpuBackend>, usize)>,
    sink: Option<&mut Vec<f32>>,
) -> Result<(), WgpuError> {
    if let (Some((ws, n)), Some(s)) = (persisted, sink) {
        let act = if asz == 2 {
            ActDtype::F16
        } else {
            ActDtype::F32
        };
        read_into_f32(backend, &ws.as_buf_ref(), *n, act, s).await?;
    }
    Ok(())
}

/// Read a GPU buffer of `n` activation elements into `sink` as f32.
async fn read_into_f32(
    backend: &WgpuBackend,
    buf: &BufRef,
    n: usize,
    act: ActDtype,
    sink: &mut Vec<f32>,
) -> Result<(), WgpuError> {
    let bytes = backend
        .read_buffer(buf.id, buf.offset, n as u64 * act.bytes_per_elem())
        .await?;
    *sink = act_bytes_to_f32_vec(act.bytes_per_elem(), &bytes);
    Ok(())
}

// ---------------------------------------------------------------------------
// per-frame decoder + top-level decode
// ---------------------------------------------------------------------------

/// Decode one latent frame `[1, z_dim, 1, h, w]` (already pre-scaled) into a
/// `[1, 3, Tout, h*8, w*8]` chunk written to `out_ref`. Runs the whole frame in
/// one BatchScope + submit. `Tout` is 1 on frame 0 (Rep markers suppress the
/// temporal doubling) and 4 afterwards.
#[allow(clippy::too_many_arguments)]
async fn decode_frame(
    backend: &WgpuBackend,
    workspace: &Workspace<WgpuBackend>,
    pl: &WanVaePipelines,
    bufs: &VaeDecoderBufs,
    fc: &mut FeatCache,
    frame_in: &BufRef,
    h_in: u32,
    w_in: u32,
    out_ref: &BufRef,
    frame_idx: usize,
    taps: Option<&mut WanVaeDecodeTaps<'_>>,
) -> Result<Shape5, WanVaeError> {
    fc.reset();
    // Old cache buffers displaced this frame; held until the submit completes.
    let mut retire: Vec<FeatEntry> = Vec::new();
    let asz = pl.act_size;
    // Which stages to persist this frame (only the selected frame, only the
    // sinks the caller asked for).
    let (want_post, want_conv_in, want_mid, want_ups) = match &taps {
        Some(t) if t.frame == frame_idx => (
            t.post_quant.is_some(),
            t.conv_in.is_some(),
            t.mid.is_some(),
            t.up_blocks.is_some(),
        ),
        _ => (false, false, false, false),
    };
    let mut p_post = None;
    let mut p_conv_in = None;
    let mut p_mid = None;
    let mut p_ups: Vec<(WsBuf<WgpuBackend>, usize)> = Vec::new();
    let out_shape;
    {
        // ScopePacker splits the frame into multiple submits so no single one
        // exceeds the GPU watchdog at native res (see VAE_SUBMIT_BUDGET_BYTES).
        // At parity res every phase fits the budget -> one submit, identical to
        // the unpacked path. Carried activations (`advance`) survive cuts via
        // the packer's hold-bag; `retire`/persisted taps are workspace-owned.
        let mut packer = ScopePacker::new(workspace, VAE_SUBMIT_BUDGET_BYTES);
        let in_shape = Shape5 {
            b: 1,
            c: config::Z_DIM as u32,
            t: 1,
            h: h_in,
            w: w_in,
        };
        let z = packer.scope().import_copy(*frame_in);

        // post_quant_conv (1x1x1, pointwise in time so per-frame == whole).
        let z = packer
            .advance(&[z], peak_bytes(config::Z_DIM as u32, 1, h_in, w_in, asz))
            .await?
            .pop()
            .unwrap();
        let (x, sh) = conv3d_1x1x1(
            packer.scope(),
            pl,
            z,
            in_shape,
            &bufs.post_quant_conv,
            config::Z_DIM as u32,
        )?;
        if want_post {
            p_post = Some(persist_stage(workspace, packer.scope(), x, sh, asz)?);
        }

        // conv_in (causal 3x3x3).
        let mid_c = config::MID_CHANNELS as u32;
        let x = packer
            .advance(&[x], peak_bytes(mid_c, sh.t, sh.h, sh.w, asz))
            .await?
            .pop()
            .unwrap();
        let (mut x, mut sh) = causal_conv3d(
            workspace,
            packer.scope(),
            pl,
            fc,
            &mut retire,
            x,
            sh,
            &bufs.conv_in,
            mid_c,
            3,
            3,
        )?;
        if want_conv_in {
            p_conv_in = Some(persist_stage(workspace, packer.scope(), x, sh, asz)?);
        }

        // mid_block: resnet -> attention -> resnet (latent res, T=1).
        x = packer
            .advance(&[x], peak_bytes(mid_c, sh.t, sh.h, sh.w, asz))
            .await?
            .pop()
            .unwrap();
        let (m0, s0) = resnet_forward(
            workspace,
            packer.scope(),
            pl,
            fc,
            &mut retire,
            x,
            sh,
            mid_c,
            &bufs.mid_block.resnets[0],
        )?;
        let m0 = packer
            .advance(&[m0], peak_bytes(mid_c, s0.t, s0.h, s0.w, asz))
            .await?
            .pop()
            .unwrap();
        let (m1, s1) =
            mid_attention_forward(packer.scope(), pl, m0, s0, &bufs.mid_block.attention)?;
        let m1 = packer
            .advance(&[m1], peak_bytes(mid_c, s1.t, s1.h, s1.w, asz))
            .await?
            .pop()
            .unwrap();
        let (m2, s2) = resnet_forward(
            workspace,
            packer.scope(),
            pl,
            fc,
            &mut retire,
            m1,
            s1,
            mid_c,
            &bufs.mid_block.resnets[1],
        )?;
        x = m2;
        sh = s2;
        if want_mid {
            p_mid = Some(persist_stage(workspace, packer.scope(), x, sh, asz)?);
        }

        // up_blocks 0..N (each resnet/upsampler is its own packer phase).
        for (i, ub) in bufs.up_blocks.iter().enumerate() {
            let out_c = config::up_out_channels(i) as u32;
            let temporal = config::up_temporal(i);
            let (y, s) = up_block_forward(
                workspace,
                &mut packer,
                pl,
                fc,
                &mut retire,
                x,
                sh,
                out_c,
                temporal,
                ub,
                asz,
            )
            .await?;
            x = y;
            sh = s;
            if want_ups {
                p_ups.push(persist_stage(workspace, packer.scope(), x, sh, asz)?);
            }
        }

        // norm_out -> silu -> conv_out (causal 3x3x3, cout=3).
        x = packer
            .advance(&[x], peak_bytes(sh.c, sh.t, sh.h, sh.w, asz))
            .await?
            .pop()
            .unwrap();
        let n = rmsnorm3d_run(packer.scope(), pl, x, sh, &bufs.norm_out)?;
        let a = silu_run(packer.scope(), pl, n, sh)?;
        let (cout, cout_shape) = causal_conv3d(
            workspace,
            packer.scope(),
            pl,
            fc,
            &mut retire,
            a,
            sh,
            &bufs.conv_out,
            config::OUT_CHANNELS as u32,
            3,
            3,
        )?;
        out_shape = cout_shape;

        let dst = packer.scope().import_copy(*out_ref);
        packer
            .scope()
            .copy_buffer_to_buffer(cout, 0, dst, 0, cout_shape.bytes(pl.act_size))?;
        packer
            .finish_void()
            .instrument(
                tracing::debug_span!(target: PHASE, "wan_vae.decode_frame", frame = frame_idx),
            )
            .await
            .map_err(WanVaeError::Wgpu)?;
    }
    // `retire` drops here (post-submit): displaced caches return to the pool.
    drop(retire);

    // Post-submit tap readback (selected frame only).
    if let Some(t) = taps
        && t.frame == frame_idx
    {
        read_stage(backend, asz, &p_post, t.post_quant.as_deref_mut()).await?;
        read_stage(backend, asz, &p_conv_in, t.conv_in.as_deref_mut()).await?;
        read_stage(backend, asz, &p_mid, t.mid.as_deref_mut()).await?;
        if let Some(sink) = t.up_blocks.as_deref_mut() {
            sink.clear();
            for (ws, n) in &p_ups {
                let act = if asz == 2 {
                    ActDtype::F16
                } else {
                    ActDtype::F32
                };
                let mut v = Vec::new();
                read_into_f32(backend, &ws.as_buf_ref(), *n, act, &mut v).await?;
                sink.push(v);
            }
        }
    }
    Ok(out_shape)
}

/// Decode one spatial latent tile (`[r0, r0+hext) x [c0, c0+wext)`) across all
/// `f` frame groups and feather-blend its output into the host `video`
/// accumulator (weighted sum) + `wsum` (per-pixel weight, channel-independent).
/// The tile carries its own `FeatCache` across frames; `weights_h`/`weights_w`
/// are the separable feather ramps (`feather_1d`) for this tile's edges. `taps`
/// is only meaningful for the single full-frame tile (parity bisection). The
/// caller normalizes (`video /= wsum`) and clamps once after every tile lands.
#[allow(clippy::too_many_arguments)]
async fn decode_tile<S: WeightSource>(
    backend: &WgpuBackend,
    workspace: &mut Workspace<WgpuBackend>,
    pl: &WanVaePipelines,
    bufs: &VaeDecoderBufs,
    latents: &[f32],
    f: usize,
    h_in: usize,
    w_in: usize,
    tile: (u32, u32, u32, u32), // (r0, c0, hext, wext) in latent pixels
    weights_h: &[f32],
    weights_w: &[f32],
    t_total: usize,
    video: &mut [f32],
    wsum: &mut [f32],
    mut taps: Option<&mut WanVaeDecodeTaps<'_>>,
) -> Result<(), WanVaeDecodeError<S::Error>> {
    let (r0, c0, hext, wext) = tile;
    let (r0, c0, hext, wext) = (r0 as usize, c0 as usize, hext as usize, wext as usize);
    let z_dim = config::Z_DIM;
    let asz = pl.act_size;
    let (h_out, w_out) = (h_in * 8, w_in * 8);
    let hw_out = h_out * w_out;
    let (toh, tow) = (hext * 8, wext * 8); // tile output H/W
    let tile_hw = toh * tow;
    let tile_elems = z_dim * hext * wext;

    let mut fc = FeatCache::new();
    let mut t_off = 0usize;
    for i in 0..f {
        // Slice + denormalize this tile's latent frame (CTHW, channel-major over
        // the F axis), uploading as act bytes.
        let mut scaled = vec![0.0_f32; tile_elems];
        for c in 0..z_dim {
            let std = config::LATENTS_STD[c];
            let mean = config::LATENTS_MEAN[c];
            for hh in 0..hext {
                for ww in 0..wext {
                    let src = (c * f + i) * h_in * w_in + (r0 + hh) * w_in + (c0 + ww);
                    scaled[(c * hext + hh) * wext + ww] = latents[src] * std + mean;
                }
            }
        }
        let in_bytes = f32s_to_act_bytes(asz, &scaled);
        let in_buf = workspace.alloc(in_bytes.len() as u64)?;
        backend.write_buffer(in_buf.id(), 0, &in_bytes)?;

        let tout = if i == 0 { 1 } else { 4 };
        let out_bytes = (3 * tout * tile_hw) as u64 * asz;
        let out_buf = workspace.alloc(out_bytes)?;
        let out_ref = out_buf.as_buf_ref();

        let shape = decode_frame(
            backend,
            &*workspace,
            pl,
            bufs,
            &mut fc,
            &in_buf.as_buf_ref(),
            hext as u32,
            wext as u32,
            &out_ref,
            i,
            taps.as_deref_mut(),
        )
        .await?;
        debug_assert_eq!(shape.t as usize, tout);

        let host = backend
            .read_buffer(out_ref.id, out_ref.offset, out_bytes)
            .instrument(tracing::debug_span!(target: PHASE, "wan_vae.readback", frame = i))
            .await?;
        let vals = act_bytes_to_f32_vec(asz, &host);

        // Feather-blend this group's [3, tout, toh, tow] block into the video at
        // the running time offset, accumulating per-pixel weight once (ch 0).
        for ch in 0..3 {
            for tt in 0..tout {
                let t = t_off + tt;
                for (hh, &wh) in weights_h.iter().enumerate().take(toh) {
                    let oh = r0 * 8 + hh;
                    for (ww, &wwt) in weights_w.iter().enumerate().take(tow) {
                        let ow = c0 * 8 + ww;
                        let wgt = wh * wwt;
                        let src = (ch * tout + tt) * tile_hw + hh * tow + ww;
                        let pix = t * hw_out + oh * w_out + ow;
                        video[ch * t_total * hw_out + pix] += vals[src] * wgt;
                        if ch == 0 {
                            wsum[pix] += wgt;
                        }
                    }
                }
            }
        }
        t_off += tout;
    }
    Ok(())
}

/// Wan VAE decoder. Decodes `[z_dim, F, h, w]` latents (B=1) into a video
/// `[3, 4F-3, h*8, w*8]` in `[-1, 1]`, frame group by frame group. Applies the
/// baked `z * std + mean` per-channel latent denormalization host-side before
/// upload (mirrors the Wan pipeline's pre-decode scaling). At native res the
/// decode runs in overlapping spatial tiles (`plan_tiles`) feather-blended into
/// the output, bounding peak VRAM to one tile's working set; at parity res it
/// collapses to a single unit-weight tile (bit-identical to the untiled path).
pub struct WanVaeDecoder {
    pub pipelines: WanVaePipelines,
    pub handles: VaeDecoderHandles,
}

#[derive(Debug)]
pub enum WanVaeDecodeError<SE: core::fmt::Debug> {
    Forward(WanVaeError),
    Residency(ResidencyError<SE, WgpuError>),
}

impl<SE: core::fmt::Debug> From<WanVaeError> for WanVaeDecodeError<SE> {
    fn from(e: WanVaeError) -> Self {
        Self::Forward(e)
    }
}
impl<SE: core::fmt::Debug> From<WgpuError> for WanVaeDecodeError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Forward(WanVaeError::Wgpu(e))
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for WanVaeDecodeError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

impl WanVaeDecoder {
    /// Decode `latents` (CTHW row-major, `z_dim * f * h_in * w_in` f32) into a
    /// host video tensor CTHW `[3, 4*f-3, h_in*8, w_in*8]` f32 clamped to
    /// `[-1, 1]`.
    pub async fn decode<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &mut Workspace<WgpuBackend>,
        latents: &[f32],
        f: usize,
        h_in: usize,
        w_in: usize,
    ) -> Result<Vec<f32>, WanVaeDecodeError<S::Error>> {
        self.decode_with_taps(backend, residency, workspace, latents, f, h_in, w_in, None)
            .await
    }

    /// `decode` with per-stage taps captured on `taps.frame` for e2e bisection.
    #[allow(clippy::too_many_arguments)]
    pub async fn decode_with_taps<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &mut Workspace<WgpuBackend>,
        latents: &[f32],
        f: usize,
        h_in: usize,
        w_in: usize,
        mut taps: Option<&mut WanVaeDecodeTaps<'_>>,
    ) -> Result<Vec<f32>, WanVaeDecodeError<S::Error>> {
        let z_dim = config::Z_DIM;
        let frame_elems = z_dim * h_in * w_in;
        assert_eq!(
            latents.len(),
            frame_elems * f,
            "decode: expected {} latents, got {}",
            frame_elems * f,
            latents.len()
        );
        let (h_out, w_out) = (h_in * 8, w_in * 8);

        let views = self
            .handles
            .acquire(residency, backend)
            .instrument(tracing::debug_span!(target: PHASE, "wan_vae.acquire"))
            .await?;
        let bufs = views.bufs();

        // Output video: CTHW with the time axis concatenated across frame
        // groups (group 0 -> 1 frame, each later group -> 4 frames).
        let t_total = if f == 0 { 0 } else { 4 * f - 3 };
        let hw_out = h_out * w_out;
        // `video` accumulates feather-weighted tile outputs; `wsum` the matching
        // per-pixel weights (channel-independent). Normalized + clamped below.
        let mut video = vec![0.0_f32; 3 * t_total * hw_out];
        let mut wsum = vec![0.0_f32; t_total * hw_out];

        // Latent tile size derived from the live VRAM budget (not a fixed
        // const): bigger tiles on a roomy device, smaller on thin hardware, so
        // peak VRAM tracks the budget instead of a hardcoded ceiling.
        let budget = residency.arbiter().budget_bytes();
        let (tile, overlap) = vae_tile_dims(budget, self.pipelines.act_size);

        // Overlapping latent tiles per axis; a single full tile at parity res.
        let tiles_h = plan_tiles(h_in as u32, tile, overlap);
        let tiles_w = plan_tiles(w_in as u32, tile, overlap);
        let single = tiles_h.len() == 1 && tiles_w.len() == 1;
        for &(r0, hext) in &tiles_h {
            let weights_h = feather_1d(hext, overlap, r0 > 0, (r0 + hext) < h_in as u32);
            for &(c0, wext) in &tiles_w {
                let weights_w = feather_1d(wext, overlap, c0 > 0, (c0 + wext) < w_in as u32);
                // Taps (parity bisection) only apply to the single full tile.
                let tile_taps = if single { taps.as_deref_mut() } else { None };
                decode_tile::<S>(
                    backend,
                    workspace,
                    &self.pipelines,
                    &bufs,
                    latents,
                    f,
                    h_in,
                    w_in,
                    (r0, c0, hext, wext),
                    &weights_h,
                    &weights_w,
                    t_total,
                    &mut video,
                    &mut wsum,
                    tile_taps,
                )
                .await?;
                // Return this tile's idle buffers to the pool before the next
                // tile grows it, so the live set stays bounded to one tile.
                if !single {
                    workspace.drain_pool();
                }
            }
        }

        // Normalize the feather blend (wsum is unit everywhere for a single
        // tile -> exact passthrough) and clamp to [-1, 1].
        for ch in 0..3 {
            let base = ch * t_total * hw_out;
            for pix in 0..(t_total * hw_out) {
                video[base + pix] = (video[base + pix] / wsum[pix]).clamp(-1.0, 1.0);
            }
        }

        Ok(video)
    }
}

// ===========================================================================
// Forward driver (encoder).
// ===========================================================================
//
// The encoder runs `_encode` (`autoencoder_kl_wan.py`) one input chunk at a
// time: chunk 0 is a single frame, every later chunk is the next 4 frames, and
// each chunk emits exactly one latent frame (4x temporal compression). Each
// chunk is its own `BatchScope` + submit (the VAE single-heavy-submit rule),
// with a host-side `feat_cache` carrying causality across chunks. The cache
// covers the same kt=3 causal convs as the decoder (conv_in, every resnet
// conv1/conv2, conv_out) plus the per-stage `downsample3d` time_conv. The two
// temporal down stages collapse a 4-frame chunk 4 -> 2 -> 1 before the mid
// block, so the mid block (and its attention) run at T=1, exactly like the
// decoder; `mid_attention_forward` is reused unchanged.
//
// quant_conv (1x1x1, 2*z_dim -> 2*z_dim) is pointwise in time, so applying it
// per chunk equals applying it to the whole concatenated latent.

// ---------------------------------------------------------------------------
// downsample helpers
// ---------------------------------------------------------------------------

/// Per-frame spatial downsample (`ZeroPad2d((0, 1, 0, 1))` + 3x3 stride-2
/// `Conv2d`, channels unchanged), applied over every time frame. NCTHW in,
/// NCTHW out with `H, W` halved. The 2D conv runs in NTCHW (batch `= B*T`).
/// The right/bottom-only zero pad is realized implicitly: a `pad=0 stride=2`
/// conv whose `h_out = h_in/2` reads `hi = ho*2 + dh` and the kernel zero-fills
/// every `hi >= h_in` gather (the bottom/right pad), so no asymmetric-pad op is
/// needed.
fn spatial_downsample<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pl: &WanVaePipelines,
    x: BatchBuf<'wsp>,
    shape: Shape5,
    w: &'wsp ConvBufs,
) -> Result<(BatchBuf<'wsp>, Shape5), WgpuError> {
    let (b, c, t, h, ww) = (shape.b, shape.c, shape.t, shape.h, shape.w);
    let bt = b * t;
    let hw = h * ww;
    // NCTHW -> NTCHW.
    let ntchw = transpose12_run(scope, pl, x, b, c, t, hw)?;

    let dn_h = h / 2;
    let dn_w = ww / 2;
    let conv_shape = Shape5 {
        b: bt,
        c,
        t: 1,
        h: dn_h,
        w: dn_w,
    };
    let cout_buf = scope.alloc(conv_shape.bytes(pl.act_size))?;
    let cu = conv2d_uniform(scope, (bt, c, h, ww), (c, dn_h, dn_w), (3, 3, 0, 0, 2, 2))?;
    let wb = scope.import(&w.weight);
    let bb = scope.import(&w.bias);
    scope.conv2d(
        &pl.conv2d.pipeline,
        &pl.conv2d.op,
        ntchw,
        wb,
        bb,
        cu,
        cout_buf,
        c,
        dn_h * dn_w,
        bt,
    )?;

    // NTCHW -> NCTHW.
    let out = transpose12_run(scope, pl, cout_buf, b, t, c, dn_h * dn_w)?;
    Ok((
        out,
        Shape5 {
            b,
            c,
            t,
            h: dn_h,
            w: dn_w,
        },
    ))
}

/// `downsample3d` temporal conv (`WanCausalConv3d(c, c, (3,1,1), stride=(2,1,1),
/// padding=0)`). Unlike the kt=3 stride-1 causal convs this has no internal
/// front pad; causality is carried by manually prepending the previous chunk's
/// last (spatially-downsampled) frame. On the first chunk the cache is empty:
/// the frame is stashed and `x` passes through untouched (pyref stores
/// `x.clone()` and skips the conv). Later chunks prepend the cached frame, run
/// the stride-2 conv (`Tout = (Tin + 1 - 3) / 2 + 1`), and restash `x`'s last
/// frame.
#[allow(clippy::too_many_arguments)]
fn downsample_temporal<'wsp>(
    workspace: &'wsp Workspace<WgpuBackend>,
    scope: &BatchScope<'wsp, WgpuBackend>,
    pl: &WanVaePipelines,
    fc: &mut FeatCache,
    retire: &mut Vec<FeatEntry>,
    x: BatchBuf<'wsp>,
    in_shape: Shape5,
    w: &'wsp ConvBufs,
) -> Result<(BatchBuf<'wsp>, Shape5), WgpuError> {
    let idx = fc.next();
    let cin = in_shape.c;
    let (h, ww) = (in_shape.h, in_shape.w);
    let tin = in_shape.t;
    let asz = pl.act_size;

    let has_cache = matches!(fc.entries[idx], FeatEntry::Frames { .. });

    // Stash this chunk's last frame for the next chunk (always 1 frame).
    let cache_shape = Shape5 {
        b: in_shape.b,
        c: cin,
        t: 1,
        h,
        w: ww,
    };
    let new_buf = workspace.alloc(cache_shape.bytes(asz))?;
    let nb = scope.import_copy(new_buf.as_buf_ref());
    let u = concat_time_uniform(scope, in_shape.b, cin, h, ww, tin, tin, tin - 1, 1, 0, 0, 0)?;
    scope.concat_time::<ConcatTimeF32>(&pl.concat_time, x, x, u, nb, cache_shape.elems())?;

    if !has_cache {
        // First chunk: cache the frame, pass `x` through unchanged.
        retire.push(std::mem::replace(
            &mut fc.entries[idx],
            FeatEntry::Frames { buf: new_buf, t: 1 },
        ));
        return Ok((x, in_shape));
    }

    // Prepend the cached frame, then a stride-2 kt=3 conv with no front pad.
    let old_ref = match &fc.entries[idx] {
        FeatEntry::Frames { buf, .. } => buf.as_buf_ref(),
        _ => unreachable!(),
    };
    let a = scope.import_copy(old_ref);
    let asm = Shape5 {
        b: in_shape.b,
        c: cin,
        t: tin + 1,
        h,
        w: ww,
    };
    let asm_buf = scope.alloc(asm.bytes(asz))?;
    let au = concat_time_uniform(scope, in_shape.b, cin, h, ww, 1, tin, 0, 1, 0, tin, 0)?;
    scope.concat_time::<ConcatTimeF32>(&pl.concat_time, a, x, au, asm_buf, asm.elems())?;

    let (out, out_shape) = conv3d_run(
        scope,
        pl,
        asm_buf,
        asm,
        w,
        cin,
        (3, 1, 1),
        (0, 0, 0),
        (2, 1, 1),
    )?;

    retire.push(std::mem::replace(
        &mut fc.entries[idx],
        FeatEntry::Frames { buf: new_buf, t: 1 },
    ));
    Ok((out, out_shape))
}

/// Encoder down stage: `num_res_blocks` resnets (`in_c -> out_c` on the first,
/// then `out_c -> out_c`) then an optional downsampler (spatial 2x always,
/// plus the temporal conv for `downsample3d`).
#[allow(clippy::too_many_arguments)]
fn down_stage_forward<'wsp>(
    workspace: &'wsp Workspace<WgpuBackend>,
    scope: &BatchScope<'wsp, WgpuBackend>,
    pl: &WanVaePipelines,
    fc: &mut FeatCache,
    retire: &mut Vec<FeatEntry>,
    mut x: BatchBuf<'wsp>,
    mut shape: Shape5,
    out_c: u32,
    temporal: bool,
    w: &'wsp DownStageBufs,
) -> Result<(BatchBuf<'wsp>, Shape5), WgpuError> {
    for resnet in &w.resnets {
        let (y, sh) = resnet_forward(workspace, scope, pl, fc, retire, x, shape, out_c, resnet)?;
        x = y;
        shape = sh;
    }
    let Some(ds) = &w.downsampler else {
        return Ok((x, shape));
    };

    // Spatial first (per-frame), then the temporal conv for downsample3d.
    let (y, sh) = spatial_downsample(scope, pl, x, shape, &ds.spatial_conv)?;
    x = y;
    shape = sh;
    if temporal {
        let tc = ds
            .time_conv
            .as_ref()
            .expect("temporal downsampler has time_conv");
        let (y, sh) = downsample_temporal(workspace, scope, pl, fc, retire, x, shape, tc)?;
        x = y;
        shape = sh;
    }
    Ok((x, shape))
}

// ---------------------------------------------------------------------------
// per-chunk encoder + top-level encode
// ---------------------------------------------------------------------------

/// Encode one input chunk `[1, 3, tin, h, w]` (`tin` = 1 on chunk 0, else 4)
/// into one latent frame `[1, 2*z_dim, 1, h/8, w/8]` written to `out_ref`.
/// Runs the whole chunk in one BatchScope + submit.
#[allow(clippy::too_many_arguments)]
async fn encode_chunk(
    backend: &WgpuBackend,
    workspace: &Workspace<WgpuBackend>,
    pl: &WanVaePipelines,
    bufs: &VaeEncoderBufs,
    fc: &mut FeatCache,
    chunk_in: &BufRef,
    tin: u32,
    h_in: u32,
    w_in: u32,
    out_ref: &BufRef,
    chunk_idx: usize,
    taps: Option<&mut WanVaeEncodeTaps<'_>>,
) -> Result<Shape5, WanVaeError> {
    fc.reset();
    let mut retire: Vec<FeatEntry> = Vec::new();
    let asz = pl.act_size;
    let (want_conv_in, want_downs, want_mid, want_conv_out) = match &taps {
        Some(t) if t.chunk == chunk_idx => (
            t.conv_in.is_some(),
            t.down_stages.is_some(),
            t.mid.is_some(),
            t.conv_out.is_some(),
        ),
        _ => (false, false, false, false),
    };
    let mut p_conv_in = None;
    let mut p_downs: Vec<(WsBuf<WgpuBackend>, usize)> = Vec::new();
    let mut p_mid = None;
    let mut p_conv_out = None;
    let out_shape;
    {
        let scope = workspace.batch();
        let x0 = scope.import(chunk_in);
        let in_shape = Shape5 {
            b: 1,
            c: config::IN_CHANNELS as u32,
            t: tin,
            h: h_in,
            w: w_in,
        };

        // conv_in (causal 3x3x3).
        let (mut x, mut sh) = causal_conv3d(
            workspace,
            &scope,
            pl,
            fc,
            &mut retire,
            x0,
            in_shape,
            &bufs.conv_in,
            config::enc_dim(0) as u32,
            3,
            3,
        )?;
        if want_conv_in {
            p_conv_in = Some(persist_stage(workspace, &scope, x, sh, asz)?);
        }

        // down stages 0..N.
        for (i, ds) in bufs.down_stages.iter().enumerate() {
            let out_c = config::down_out_channels(i) as u32;
            let temporal = config::down_temporal(i);
            let (y, s) = down_stage_forward(
                workspace,
                &scope,
                pl,
                fc,
                &mut retire,
                x,
                sh,
                out_c,
                temporal,
                ds,
            )?;
            x = y;
            sh = s;
            if want_downs {
                p_downs.push(persist_stage(workspace, &scope, x, sh, asz)?);
            }
        }

        // mid_block: resnet -> attention -> resnet (T=1 here).
        let (m0, s0) = resnet_forward(
            workspace,
            &scope,
            pl,
            fc,
            &mut retire,
            x,
            sh,
            config::ENC_MID_CHANNELS as u32,
            &bufs.mid_block.resnets[0],
        )?;
        let (m1, s1) = mid_attention_forward(&scope, pl, m0, s0, &bufs.mid_block.attention)?;
        let (m2, s2) = resnet_forward(
            workspace,
            &scope,
            pl,
            fc,
            &mut retire,
            m1,
            s1,
            config::ENC_MID_CHANNELS as u32,
            &bufs.mid_block.resnets[1],
        )?;
        x = m2;
        sh = s2;
        if want_mid {
            p_mid = Some(persist_stage(workspace, &scope, x, sh, asz)?);
        }

        // norm_out -> silu -> conv_out (causal 3x3x3, cout=2*z_dim).
        let n = rmsnorm3d_run(&scope, pl, x, sh, &bufs.norm_out)?;
        let a = silu_run(&scope, pl, n, sh)?;
        let (c_out, c_sh) = causal_conv3d(
            workspace,
            &scope,
            pl,
            fc,
            &mut retire,
            a,
            sh,
            &bufs.conv_out,
            config::Z_DIM_X2 as u32,
            3,
            3,
        )?;
        if want_conv_out {
            p_conv_out = Some(persist_stage(workspace, &scope, c_out, c_sh, asz)?);
        }

        // quant_conv (1x1x1, pointwise in time so per-chunk == whole).
        let (q, q_sh) = conv3d_1x1x1(
            &scope,
            pl,
            c_out,
            c_sh,
            &bufs.quant_conv,
            config::Z_DIM_X2 as u32,
        )?;
        out_shape = q_sh;

        let dst = scope.import(out_ref);
        scope.copy_buffer_to_buffer(q, 0, dst, 0, q_sh.bytes(pl.act_size))?;
        scope
            .submit_void()
            .instrument(tracing::debug_span!(target: PHASE, "wan_vae.submit", phase = "encode_chunk", chunk = chunk_idx))
            .await
            .map_err(WanVaeError::Wgpu)?;
    }
    drop(retire);

    if let Some(t) = taps
        && t.chunk == chunk_idx
    {
        read_stage(backend, asz, &p_conv_in, t.conv_in.as_deref_mut()).await?;
        read_stage(backend, asz, &p_mid, t.mid.as_deref_mut()).await?;
        read_stage(backend, asz, &p_conv_out, t.conv_out.as_deref_mut()).await?;
        if let Some(sink) = t.down_stages.as_deref_mut() {
            sink.clear();
            for (ws, n) in &p_downs {
                let act = if asz == 2 {
                    ActDtype::F16
                } else {
                    ActDtype::F32
                };
                let mut v = Vec::new();
                read_into_f32(backend, &ws.as_buf_ref(), *n, act, &mut v).await?;
                sink.push(v);
            }
        }
    }
    Ok(out_shape)
}

/// Wan VAE encoder. Encodes a video `[3, F, h, w]` in `[-1, 1]` (B=1, F = 4k+1)
/// into latent distribution params `[2*z_dim, k+1, h/8, w/8]` (mean ++ logvar),
/// chunk by chunk (chunk 0 = frame 0, each later chunk = the next 4 frames).
pub struct WanVaeEncoder {
    pub pipelines: WanVaePipelines,
    pub handles: VaeEncoderHandles,
}

#[derive(Debug)]
pub enum WanVaeEncodeError<SE: core::fmt::Debug> {
    Forward(WanVaeError),
    Residency(ResidencyError<SE, WgpuError>),
}

impl<SE: core::fmt::Debug> From<WanVaeError> for WanVaeEncodeError<SE> {
    fn from(e: WanVaeError) -> Self {
        Self::Forward(e)
    }
}
impl<SE: core::fmt::Debug> From<WgpuError> for WanVaeEncodeError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Forward(WanVaeError::Wgpu(e))
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for WanVaeEncodeError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

impl WanVaeEncoder {
    /// Encode `video` (CTHW row-major, `3 * f * h_in * w_in` f32, `f = 4k+1`)
    /// into host latent params CTHW `[2*z_dim, k+1, h_in/8, w_in/8]` f32.
    pub async fn encode<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &mut Workspace<WgpuBackend>,
        video: &[f32],
        f: usize,
        h_in: usize,
        w_in: usize,
    ) -> Result<Vec<f32>, WanVaeEncodeError<S::Error>> {
        self.encode_with_taps(backend, residency, workspace, video, f, h_in, w_in, None)
            .await
    }

    /// `encode` with per-stage taps captured on `taps.chunk` for e2e bisection.
    #[allow(clippy::too_many_arguments)]
    pub async fn encode_with_taps<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &mut Workspace<WgpuBackend>,
        video: &[f32],
        f: usize,
        h_in: usize,
        w_in: usize,
        mut taps: Option<&mut WanVaeEncodeTaps<'_>>,
    ) -> Result<Vec<f32>, WanVaeEncodeError<S::Error>> {
        let hw_in = h_in * w_in;
        assert_eq!(
            video.len(),
            3 * f * hw_in,
            "encode: expected {} input samples, got {}",
            3 * f * hw_in,
            video.len()
        );
        assert!(
            f >= 1 && (f - 1).is_multiple_of(4),
            "encode: F must be 4k+1, got {f}"
        );
        let asz = self.pipelines.act_size;
        let (h_out, w_out) = (h_in / 8, w_in / 8);
        let hw_out = h_out * w_out;
        let zc = config::Z_DIM_X2;

        let views = self
            .handles
            .acquire(residency, backend)
            .instrument(tracing::debug_span!(target: PHASE, "wan_vae.acquire"))
            .await?;
        let bufs = views.bufs();
        let mut fc = FeatCache::new();

        let n_chunks = 1 + (f - 1) / 4;
        // Latent CTHW: time axis concatenated across chunks (1 latent frame each).
        let mut latent = vec![0.0_f32; zc * n_chunks * hw_out];
        for i in 0..n_chunks {
            // Slice this chunk's frames (chunk 0 = frame 0; chunk i = frames
            // [1+4(i-1), 1+4i)) into NCTHW [1, 3, tin, h, w].
            let (f0, tin) = if i == 0 {
                (0usize, 1usize)
            } else {
                (1 + 4 * (i - 1), 4usize)
            };
            let mut chunk = vec![0.0_f32; 3 * tin * hw_in];
            for c in 0..3 {
                for tt in 0..tin {
                    let src = (c * f + (f0 + tt)) * hw_in;
                    let dst = (c * tin + tt) * hw_in;
                    chunk[dst..dst + hw_in].copy_from_slice(&video[src..src + hw_in]);
                }
            }
            let in_bytes = f32s_to_act_bytes(asz, &chunk);
            let in_buf = workspace.alloc(in_bytes.len() as u64)?;
            backend.write_buffer(in_buf.id(), 0, &in_bytes)?;

            let out_bytes = (zc * hw_out) as u64 * asz;
            let out_buf = workspace.alloc(out_bytes)?;
            let out_ref = out_buf.as_buf_ref();

            let shape = encode_chunk(
                backend,
                &*workspace,
                &self.pipelines,
                &bufs,
                &mut fc,
                &in_buf.as_buf_ref(),
                tin as u32,
                h_in as u32,
                w_in as u32,
                &out_ref,
                i,
                taps.as_deref_mut(),
            )
            .await?;
            debug_assert_eq!(shape.t, 1);
            debug_assert_eq!(shape.c as usize, zc);

            let host = backend
                .read_buffer(out_ref.id, out_ref.offset, out_bytes)
                .instrument(tracing::debug_span!(target: PHASE, "wan_vae.readback", chunk = i))
                .await?;
            let vals = act_bytes_to_f32_vec(asz, &host);

            // Scatter this chunk's [zc, 1, hw] block into the CTHW latent at
            // time index i.
            for ch in 0..zc {
                let src = ch * hw_out;
                let dst = (ch * n_chunks + i) * hw_out;
                latent[dst..dst + hw_out].copy_from_slice(&vals[src..src + hw_out]);
            }
        }

        Ok(latent)
    }
}

/// Act-dtype-aware readback conversion (`act_size == 2` decodes f16).
fn act_bytes_to_f32_vec(act_size: u64, bytes: &[u8]) -> Vec<f32> {
    if act_size == 2 {
        bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()
    } else {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

/// Act-dtype-aware upload conversion (`act_size == 2` encodes f16).
fn f32s_to_act_bytes(act_size: u64, vals: &[f32]) -> Vec<u8> {
    if act_size == 2 {
        let mut out = Vec::with_capacity(vals.len() * 2);
        for v in vals {
            out.extend_from_slice(&half::f16::from_f32(*v).to_le_bytes());
        }
        out
    } else {
        let mut out = Vec::with_capacity(vals.len() * 4);
        for v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
}
