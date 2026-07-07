//! Flux-style `AutoencoderKL` decoder (the diffusers KL-VAE family).
//!
//! Shared by every model in the tree that ships a diffusers-layout KL VAE:
//! Z-Image (`latent_channels=16`, `block_out=[128,256,512,512]`, no
//! post-quant-conv) and Ideogram-4 (`latent_channels=32`,
//! `block_out=[96,192,384,384]`, 1x1 post-quant-conv). The decoder graph is
//! identical across them; only the channel widths, latent depth, the optional
//! post-quant conv, and the host-side latent pre-transform differ. Those are
//! carried in `KlVaeConfig`; the weight-key spelling is the diffusers schema
//! for all members (`decoder.conv_in`, `decoder.up_blocks.{i}.resnets.{j}.*`,
//! `decoder.mid_block.attentions.0.*`, `decoder.conv_norm_out`,
//! `decoder.conv_out`, and a top-level `post_quant_conv` when present).
//!
//! Sequence (`Decoder.forward`):
//! ```text
//! z = post_quant_conv(z)?                      [1, lat, h, w]  (1x1, optional)
//! x = conv_in(z)                               [1, mid, h, w]
//! x = mid_block(x)                             resnet -> attn -> resnet
//! for i in 0..N: x = up_blocks[i](x)           R resnets + (nearest2x + conv)
//! x = silu(group_norm(x))
//! x = conv_out(x)                              [1, 3, h*F, w*F]
//! ```
//!
//! `mid` is the last entry of `block_out_channels`; the up-block stack walks
//! that list in reverse. Mid-block attention is single-head over `[B, H*W, C]`
//! (sdpa, `scale = 1/sqrt(C)`). No reductions leave f32.
//!
//! This file declares typed `WeightId` bundles, residency handles, GPU views,
//! `BufRef` bundles, the compiled pipeline set, and the forward driver
//! (single-shot + spatially-tiled). The per-model glue (which `KlVaeConfig`,
//! the latent unpatch / denorm) stays in the model module.

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError, WgpuPipeline};
use thinfer_core::cache::KernelKey;
use thinfer_core::ops::{
    ActDtype, AddF32, BcastAddF32, BcastAddOp, Conv2dConfig, Conv2dF32, Conv2dOp, GroupNormF32,
    GroupNormOp, MatMulConfig, MatMulF32, MatmulOp, Op, SdpaF32LargeD, SdpaOp, SiluF32,
    Transpose12F32, Transpose12Op, Upsample2dNearestF32, Upsample2dNearestOp, WgslConfig,
};
use thinfer_core::residency::{GpuView, ResidencyError, WeightHandle, WeightResidency};
use thinfer_core::trace::{self, PHASE};
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace, WsBuf};
use tracing::Instrument;

use crate::common::loader::{LoadError, register_linear, register_passthrough};

// ============================================================================
// Architecture config
// ============================================================================

/// Host-side transform applied to the latent before it enters the decoder.
/// Z-Image folds a scalar `(z / scaling) + shift` here; Ideogram-4 does a
/// per-channel denorm + unpatch in its own pipeline and feeds spatial latents
/// straight through (`None`).
#[derive(Clone, Copy, Debug)]
pub enum LatentPretransform {
    None,
    /// `z' = z / scaling + shift`.
    Scalar {
        scaling: f32,
        shift: f32,
    },
}

/// Per-model arch knobs for the KL-VAE decoder. Everything else (resnets per
/// up-block, group count, eps, out channels, attention heads) is constant
/// across the diffusers KL family and lives as module consts below.
#[derive(Clone, Copy, Debug)]
pub struct KlVaeConfig {
    /// Encoder-order block widths, ascending. The decoder walks them reversed:
    /// `up_block i` outputs `block_out_channels[len-1-i]`. The last entry is
    /// the mid-block width (= `conv_in` output). Length is the resolution count.
    pub block_out_channels: &'static [usize],
    /// `z_channels`: depth of the spatial latent fed to the decoder.
    pub latent_channels: usize,
    /// 1x1 `post_quant_conv(lat -> lat)` applied before `conv_in` (Flux2 VAE).
    pub has_post_quant_conv: bool,
    /// Host pre-transform applied inside `decode`.
    pub pretransform: LatentPretransform,
}

impl KlVaeConfig {
    pub const fn n_up_blocks(&self) -> usize {
        self.block_out_channels.len()
    }
    /// Mid-block / `conv_in`-output channel count (last block width).
    pub const fn mid_channels(&self) -> usize {
        self.block_out_channels[self.block_out_channels.len() - 1]
    }
    /// Output channels of up_block `i` (reversed block list).
    pub const fn up_block_out_channels(&self, i: usize) -> usize {
        self.block_out_channels[self.block_out_channels.len() - 1 - i]
    }
    /// Input channels of up_block `i` (= prev block's out; mid for `i=0`).
    pub const fn up_block_in_channels(&self, i: usize) -> usize {
        if i == 0 {
            self.mid_channels()
        } else {
            self.up_block_out_channels(i - 1)
        }
    }
    /// True when up_block `i` has an upsampler (all but the last).
    pub const fn up_block_has_upsampler(&self, i: usize) -> bool {
        i + 1 < self.n_up_blocks()
    }
    /// Total spatial scale factor `2^(num_resolutions - 1)`.
    pub const fn vae_scale_factor(&self) -> usize {
        1 << (self.block_out_channels.len() - 1)
    }
}

/// `layers_per_block + 1` in the diffusers decoder ctor. Both family members
/// use `layers_per_block = 2`, so 3 resnets per up-block.
pub const RESNETS_PER_UP_BLOCK: usize = 3;
pub const NORM_NUM_GROUPS: usize = 32;
pub const NORM_EPS: f32 = 1e-6;
pub const OUT_CHANNELS: usize = 3;

// ============================================================================
// Weight IDs
// ============================================================================

/// `nn.Conv2d` (3x3 or 1x1) weight + bias.
#[derive(Clone, Debug)]
pub struct Conv2dWeights {
    pub weight: WeightId,
    pub bias: WeightId,
}

/// `nn.GroupNorm` affine weight + bias.
#[derive(Clone, Debug)]
pub struct GroupNormWeights {
    pub weight: WeightId,
    pub bias: WeightId,
}

#[derive(Clone, Debug)]
pub struct LinearWeights {
    pub weight: WeightId,
    pub bias: WeightId,
}

/// `ResnetBlock2D` weights. `conv_shortcut` is Some when in/out channels differ.
#[derive(Clone, Debug)]
pub struct ResnetWeights {
    pub norm1: GroupNormWeights,
    pub conv1: Conv2dWeights,
    pub norm2: GroupNormWeights,
    pub conv2: Conv2dWeights,
    pub conv_shortcut: Option<Conv2dWeights>,
}

/// Mid-block single-head self-attention (`to_q/k/v/out.0` are all `[C, C]`).
#[derive(Clone, Debug)]
pub struct MidAttentionWeights {
    pub group_norm: GroupNormWeights,
    pub to_q: LinearWeights,
    pub to_k: LinearWeights,
    pub to_v: LinearWeights,
    pub to_out: LinearWeights,
}

#[derive(Clone, Debug)]
pub struct MidBlockWeights {
    pub resnets: [ResnetWeights; 2],
    pub attention: MidAttentionWeights,
}

#[derive(Clone, Debug)]
pub struct UpBlockWeights {
    pub resnets: Vec<ResnetWeights>,
    /// 3x3 conv applied after nearest-neighbor 2x upsample. Some on all but
    /// the last `UpDecoderBlock2D`.
    pub upsampler_conv: Option<Conv2dWeights>,
}

#[derive(Clone, Debug)]
pub struct VaeDecoderWeights {
    /// 1x1 `post_quant_conv` (Flux2 VAE only).
    pub post_quant_conv: Option<Conv2dWeights>,
    pub conv_in: Conv2dWeights,
    pub mid_block: MidBlockWeights,
    pub up_blocks: Vec<UpBlockWeights>,
    pub conv_norm_out: GroupNormWeights,
    pub conv_out: Conv2dWeights,
}

impl VaeDecoderWeights {
    pub fn new(cfg: &KlVaeConfig) -> Self {
        let id = |s: String| WeightId(s);
        let conv = |p: &str| Conv2dWeights {
            weight: id(format!("{p}.weight")),
            bias: id(format!("{p}.bias")),
        };
        let gn = |p: &str| GroupNormWeights {
            weight: id(format!("{p}.weight")),
            bias: id(format!("{p}.bias")),
        };
        let lin = |p: &str| LinearWeights {
            weight: id(format!("{p}.weight")),
            bias: id(format!("{p}.bias")),
        };
        let resnet = |p: String, has_shortcut: bool| ResnetWeights {
            norm1: gn(&format!("{p}.norm1")),
            conv1: conv(&format!("{p}.conv1")),
            norm2: gn(&format!("{p}.norm2")),
            conv2: conv(&format!("{p}.conv2")),
            conv_shortcut: has_shortcut.then(|| conv(&format!("{p}.conv_shortcut"))),
        };

        let mid_block = MidBlockWeights {
            resnets: [
                resnet("decoder.mid_block.resnets.0".into(), false),
                resnet("decoder.mid_block.resnets.1".into(), false),
            ],
            attention: MidAttentionWeights {
                group_norm: gn("decoder.mid_block.attentions.0.group_norm"),
                to_q: lin("decoder.mid_block.attentions.0.to_q"),
                to_k: lin("decoder.mid_block.attentions.0.to_k"),
                to_v: lin("decoder.mid_block.attentions.0.to_v"),
                to_out: lin("decoder.mid_block.attentions.0.to_out.0"),
            },
        };

        let mut up_blocks = Vec::with_capacity(cfg.n_up_blocks());
        for i in 0..cfg.n_up_blocks() {
            let cin = cfg.up_block_in_channels(i);
            let cout = cfg.up_block_out_channels(i);
            let mut resnets = Vec::with_capacity(RESNETS_PER_UP_BLOCK);
            for j in 0..RESNETS_PER_UP_BLOCK {
                let resnet_in = if j == 0 { cin } else { cout };
                let has_shortcut = resnet_in != cout;
                resnets.push(resnet(
                    format!("decoder.up_blocks.{i}.resnets.{j}"),
                    has_shortcut,
                ));
            }
            let upsampler_conv = cfg
                .up_block_has_upsampler(i)
                .then(|| conv(&format!("decoder.up_blocks.{i}.upsamplers.0.conv")));
            up_blocks.push(UpBlockWeights {
                resnets,
                upsampler_conv,
            });
        }

        Self {
            // Top-level diffusers key (not under `decoder.`).
            post_quant_conv: cfg.has_post_quant_conv.then(|| conv("post_quant_conv")),
            conv_in: conv("decoder.conv_in"),
            mid_block,
            up_blocks,
            conv_norm_out: gn("decoder.conv_norm_out"),
            conv_out: conv("decoder.conv_out"),
        }
    }
}

// ============================================================================
// Residency handles (no GPU allocation; bytes flow on `acquire`)
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub struct Conv2dHandles {
    pub weight: WeightHandle,
    pub bias: WeightHandle,
}

#[derive(Clone, Copy, Debug)]
pub struct GroupNormHandles {
    pub weight: WeightHandle,
    pub bias: WeightHandle,
}

#[derive(Clone, Copy, Debug)]
pub struct LinearHandles {
    pub weight: WeightHandle,
    pub bias: WeightHandle,
}

#[derive(Clone, Copy, Debug)]
pub struct ResnetHandles {
    pub norm1: GroupNormHandles,
    pub conv1: Conv2dHandles,
    pub norm2: GroupNormHandles,
    pub conv2: Conv2dHandles,
    pub conv_shortcut: Option<Conv2dHandles>,
}

#[derive(Clone, Copy, Debug)]
pub struct MidAttentionHandles {
    pub group_norm: GroupNormHandles,
    pub to_q: LinearHandles,
    pub to_k: LinearHandles,
    pub to_v: LinearHandles,
    pub to_out: LinearHandles,
}

#[derive(Clone, Copy, Debug)]
pub struct MidBlockHandles {
    pub resnets: [ResnetHandles; 2],
    pub attention: MidAttentionHandles,
}

#[derive(Clone, Debug)]
pub struct UpBlockHandles {
    pub resnets: Vec<ResnetHandles>,
    pub upsampler_conv: Option<Conv2dHandles>,
}

#[derive(Clone, Debug)]
pub struct VaeDecoderHandles {
    pub post_quant_conv: Option<Conv2dHandles>,
    pub conv_in: Conv2dHandles,
    pub mid_block: MidBlockHandles,
    pub up_blocks: Vec<UpBlockHandles>,
    pub conv_norm_out: GroupNormHandles,
    pub conv_out: Conv2dHandles,
}

// ============================================================================
// BufRef bundles (post-acquire, for forward driver)
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub struct Conv2dBufs {
    pub weight: BufRef,
    pub bias: BufRef,
}

#[derive(Clone, Copy, Debug)]
pub struct GroupNormBufs {
    pub weight: BufRef,
    pub bias: BufRef,
}

#[derive(Clone, Copy, Debug)]
pub struct LinearBufs {
    pub weight: BufRef,
    pub bias: BufRef,
}

#[derive(Clone, Copy, Debug)]
pub struct ResnetBufs {
    pub norm1: GroupNormBufs,
    pub conv1: Conv2dBufs,
    pub norm2: GroupNormBufs,
    pub conv2: Conv2dBufs,
    pub conv_shortcut: Option<Conv2dBufs>,
}

#[derive(Clone, Copy, Debug)]
pub struct MidAttentionBufs {
    pub group_norm: GroupNormBufs,
    pub to_q: LinearBufs,
    pub to_k: LinearBufs,
    pub to_v: LinearBufs,
    pub to_out: LinearBufs,
}

#[derive(Clone, Copy, Debug)]
pub struct MidBlockBufs {
    pub resnets: [ResnetBufs; 2],
    pub attention: MidAttentionBufs,
}

#[derive(Clone, Debug)]
pub struct UpBlockBufs {
    pub resnets: Vec<ResnetBufs>,
    pub upsampler_conv: Option<Conv2dBufs>,
}

#[derive(Clone, Debug)]
pub struct VaeDecoderBufs {
    pub post_quant_conv: Option<Conv2dBufs>,
    pub conv_in: Conv2dBufs,
    pub mid_block: MidBlockBufs,
    pub up_blocks: Vec<UpBlockBufs>,
    pub conv_norm_out: GroupNormBufs,
    pub conv_out: Conv2dBufs,
}

// ============================================================================
// GpuView bundles (pin guards; bufs() materializes BufRefs)
// ============================================================================

pub struct Conv2dViews<'a> {
    pub weight: GpuView<'a>,
    pub bias: GpuView<'a>,
}
impl Conv2dViews<'_> {
    pub fn bufs(&self) -> Conv2dBufs {
        Conv2dBufs {
            weight: self.weight.buf(),
            bias: self.bias.buf(),
        }
    }
}

pub struct GroupNormViews<'a> {
    pub weight: GpuView<'a>,
    pub bias: GpuView<'a>,
}
impl GroupNormViews<'_> {
    pub fn bufs(&self) -> GroupNormBufs {
        GroupNormBufs {
            weight: self.weight.buf(),
            bias: self.bias.buf(),
        }
    }
}

pub struct LinearViews<'a> {
    pub weight: GpuView<'a>,
    pub bias: GpuView<'a>,
}
impl LinearViews<'_> {
    pub fn bufs(&self) -> LinearBufs {
        LinearBufs {
            weight: self.weight.buf(),
            bias: self.bias.buf(),
        }
    }
}

pub struct ResnetViews<'a> {
    pub norm1: GroupNormViews<'a>,
    pub conv1: Conv2dViews<'a>,
    pub norm2: GroupNormViews<'a>,
    pub conv2: Conv2dViews<'a>,
    pub conv_shortcut: Option<Conv2dViews<'a>>,
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

pub struct MidAttentionViews<'a> {
    pub group_norm: GroupNormViews<'a>,
    pub to_q: LinearViews<'a>,
    pub to_k: LinearViews<'a>,
    pub to_v: LinearViews<'a>,
    pub to_out: LinearViews<'a>,
}
impl MidAttentionViews<'_> {
    pub fn bufs(&self) -> MidAttentionBufs {
        MidAttentionBufs {
            group_norm: self.group_norm.bufs(),
            to_q: self.to_q.bufs(),
            to_k: self.to_k.bufs(),
            to_v: self.to_v.bufs(),
            to_out: self.to_out.bufs(),
        }
    }
}

pub struct MidBlockViews<'a> {
    pub resnets: [ResnetViews<'a>; 2],
    pub attention: MidAttentionViews<'a>,
}
impl MidBlockViews<'_> {
    pub fn bufs(&self) -> MidBlockBufs {
        MidBlockBufs {
            resnets: [self.resnets[0].bufs(), self.resnets[1].bufs()],
            attention: self.attention.bufs(),
        }
    }
}

pub struct UpBlockViews<'a> {
    pub resnets: Vec<ResnetViews<'a>>,
    pub upsampler_conv: Option<Conv2dViews<'a>>,
}
impl UpBlockViews<'_> {
    pub fn bufs(&self) -> UpBlockBufs {
        UpBlockBufs {
            resnets: self.resnets.iter().map(|r| r.bufs()).collect(),
            upsampler_conv: self.upsampler_conv.as_ref().map(|c| c.bufs()),
        }
    }
}

pub struct VaeDecoderViews<'a> {
    pub post_quant_conv: Option<Conv2dViews<'a>>,
    pub conv_in: Conv2dViews<'a>,
    pub mid_block: MidBlockViews<'a>,
    pub up_blocks: Vec<UpBlockViews<'a>>,
    pub conv_norm_out: GroupNormViews<'a>,
    pub conv_out: Conv2dViews<'a>,
}
impl VaeDecoderViews<'_> {
    pub fn bufs(&self) -> VaeDecoderBufs {
        VaeDecoderBufs {
            post_quant_conv: self.post_quant_conv.as_ref().map(|c| c.bufs()),
            conv_in: self.conv_in.bufs(),
            mid_block: self.mid_block.bufs(),
            up_blocks: self.up_blocks.iter().map(|u| u.bufs()).collect(),
            conv_norm_out: self.conv_norm_out.bufs(),
            conv_out: self.conv_out.bufs(),
        }
    }
}

// ============================================================================
// `acquire` impls
// ============================================================================

impl Conv2dHandles {
    pub async fn acquire<'r, S: WeightSource>(
        &self,
        residency: &'r WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<Conv2dViews<'r>, ResidencyError<S::Error, WgpuError>> {
        Ok(Conv2dViews {
            weight: residency.acquire(self.weight, backend).await?,
            bias: residency.acquire(self.bias, backend).await?,
        })
    }
}

impl GroupNormHandles {
    pub async fn acquire<'r, S: WeightSource>(
        &self,
        residency: &'r WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<GroupNormViews<'r>, ResidencyError<S::Error, WgpuError>> {
        Ok(GroupNormViews {
            weight: residency.acquire(self.weight, backend).await?,
            bias: residency.acquire(self.bias, backend).await?,
        })
    }
}

impl LinearHandles {
    pub async fn acquire<'r, S: WeightSource>(
        &self,
        residency: &'r WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<LinearViews<'r>, ResidencyError<S::Error, WgpuError>> {
        Ok(LinearViews {
            weight: residency.acquire(self.weight, backend).await?,
            bias: residency.acquire(self.bias, backend).await?,
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

impl MidAttentionHandles {
    pub async fn acquire<'r, S: WeightSource>(
        &self,
        residency: &'r WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<MidAttentionViews<'r>, ResidencyError<S::Error, WgpuError>> {
        Ok(MidAttentionViews {
            group_norm: self.group_norm.acquire(residency, backend).await?,
            to_q: self.to_q.acquire(residency, backend).await?,
            to_k: self.to_k.acquire(residency, backend).await?,
            to_v: self.to_v.acquire(residency, backend).await?,
            to_out: self.to_out.acquire(residency, backend).await?,
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
        let upsampler_conv = match self.upsampler_conv {
            Some(c) => Some(c.acquire(residency, backend).await?),
            None => None,
        };
        Ok(UpBlockViews {
            resnets,
            upsampler_conv,
        })
    }
}

impl VaeDecoderHandles {
    pub async fn acquire<'r, S: WeightSource>(
        &self,
        residency: &'r WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<VaeDecoderViews<'r>, ResidencyError<S::Error, WgpuError>> {
        let post_quant_conv = match self.post_quant_conv {
            Some(c) => Some(
                c.acquire(residency, backend)
                    .instrument(tracing::debug_span!(target: PHASE, "vae.acquire.part", part = "post_quant_conv"))
                    .await?,
            ),
            None => None,
        };
        let conv_in = self
            .conv_in
            .acquire(residency, backend)
            .instrument(tracing::debug_span!(target: PHASE, "vae.acquire.part", part = "conv_in"))
            .await?;
        let mid_block = self
            .mid_block
            .acquire(residency, backend)
            .instrument(tracing::debug_span!(target: PHASE, "vae.acquire.part", part = "mid_block"))
            .await?;
        let mut up_blocks = Vec::with_capacity(self.up_blocks.len());
        for (idx, u) in self.up_blocks.iter().enumerate() {
            up_blocks.push(
                u.acquire(residency, backend)
                    .instrument(tracing::debug_span!(target: PHASE, "vae.acquire.part", part = "up_block", idx))
                    .await?,
            );
        }
        let conv_norm_out = self
            .conv_norm_out
            .acquire(residency, backend)
            .instrument(
                tracing::debug_span!(target: PHASE, "vae.acquire.part", part = "conv_norm_out"),
            )
            .await?;
        let conv_out = self
            .conv_out
            .acquire(residency, backend)
            .instrument(tracing::debug_span!(target: PHASE, "vae.acquire.part", part = "conv_out"))
            .await?;
        Ok(VaeDecoderViews {
            post_quant_conv,
            conv_in,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
        })
    }
}

// ============================================================================
// Registration: build handles from weight names (no GPU upload yet).
// ============================================================================

fn reg_conv<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &Conv2dWeights,
) -> Result<Conv2dHandles, LoadError> {
    // Conv2d weight is 4D `[Cout, Cin, kH, kW]`; no transpose. Bias is 1D.
    Ok(Conv2dHandles {
        weight: register_passthrough(residency, &w.weight)?,
        bias: register_passthrough(residency, &w.bias)?,
    })
}

fn reg_gn<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &GroupNormWeights,
) -> Result<GroupNormHandles, LoadError> {
    Ok(GroupNormHandles {
        weight: register_passthrough(residency, &w.weight)?,
        bias: register_passthrough(residency, &w.bias)?,
    })
}

fn reg_linear<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &LinearWeights,
) -> Result<LinearHandles, LoadError> {
    // 2D Linear: transposed-at-load for the matmul `A @ B` convention.
    Ok(LinearHandles {
        weight: register_linear(residency, &w.weight)?,
        bias: register_passthrough(residency, &w.bias)?,
    })
}

fn reg_resnet<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &ResnetWeights,
) -> Result<ResnetHandles, LoadError> {
    Ok(ResnetHandles {
        norm1: reg_gn(residency, &w.norm1)?,
        conv1: reg_conv(residency, &w.conv1)?,
        norm2: reg_gn(residency, &w.norm2)?,
        conv2: reg_conv(residency, &w.conv2)?,
        conv_shortcut: match &w.conv_shortcut {
            Some(c) => Some(reg_conv(residency, c)?),
            None => None,
        },
    })
}

fn reg_mid_attention<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &MidAttentionWeights,
) -> Result<MidAttentionHandles, LoadError> {
    Ok(MidAttentionHandles {
        group_norm: reg_gn(residency, &w.group_norm)?,
        to_q: reg_linear(residency, &w.to_q)?,
        to_k: reg_linear(residency, &w.to_k)?,
        to_v: reg_linear(residency, &w.to_v)?,
        to_out: reg_linear(residency, &w.to_out)?,
    })
}

pub fn register_vae_decoder_handles<S: WeightSource>(
    residency: &WeightResidency<S>,
    cfg: &KlVaeConfig,
) -> Result<VaeDecoderHandles, LoadError> {
    let mw = VaeDecoderWeights::new(cfg);
    let mid_block = MidBlockHandles {
        resnets: [
            reg_resnet(residency, &mw.mid_block.resnets[0])?,
            reg_resnet(residency, &mw.mid_block.resnets[1])?,
        ],
        attention: reg_mid_attention(residency, &mw.mid_block.attention)?,
    };
    let mut up_blocks = Vec::with_capacity(mw.up_blocks.len());
    for ub in &mw.up_blocks {
        let mut resnets = Vec::with_capacity(ub.resnets.len());
        for r in &ub.resnets {
            resnets.push(reg_resnet(residency, r)?);
        }
        let upsampler_conv = match &ub.upsampler_conv {
            Some(c) => Some(reg_conv(residency, c)?),
            None => None,
        };
        up_blocks.push(UpBlockHandles {
            resnets,
            upsampler_conv,
        });
    }
    let post_quant_conv = match &mw.post_quant_conv {
        Some(c) => Some(reg_conv(residency, c)?),
        None => None,
    };
    Ok(VaeDecoderHandles {
        post_quant_conv,
        conv_in: reg_conv(residency, &mw.conv_in)?,
        mid_block,
        up_blocks,
        conv_norm_out: reg_gn(residency, &mw.conv_norm_out)?,
        conv_out: reg_conv(residency, &mw.conv_out)?,
    })
}

// ============================================================================
// Pipeline cache
// ============================================================================

/// One compiled conv2d variant: pipeline + the op (tile config) it was
/// built from. The decoder holds one instance per tile-shape regime.
pub struct ConvPipeline {
    pub pipeline: WgpuPipeline,
    pub op: Conv2dF32,
}

impl ConvPipeline {
    async fn compile(
        backend: &WgpuBackend,
        label: &str,
        cfg: &WgslConfig,
        tile: Conv2dConfig,
    ) -> Result<Self, WgpuError> {
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

/// Wide-spatial conv tile: large `M = Hout*Wout` with cout >= 64 (the
/// image-space resnet convs). Fatter bn raises arithmetic intensity per
/// gathered im2col element.
const CONV_TILE_WIDE: Conv2dConfig = Conv2dConfig {
    bm: 64,
    bn: 128,
    bk: 32,
    tm: 4,
    tn: 8,
    prefetch: false,
};
/// Small-N conv tile: conv_out (cout=3). The default bm=64 tile would idle
/// 15/16 of each workgroup's rows.
const CONV_TILE_SMALL_N: Conv2dConfig = Conv2dConfig {
    bm: 4,
    bn: 128,
    bk: 32,
    tm: 1,
    tn: 2,
    prefetch: false,
};

/// WGSL pipelines needed by the KL-VAE decoder forward. Built once at runtime
/// init (or lazily on first decode). Arch-independent: the same set serves
/// every `KlVaeConfig` (post-quant 1x1 reuses the default conv2d).
pub struct VaeDecoderPipelines {
    /// Activation storage dtype: F16 when the device has `SHADER_F16`, else
    /// F32. Weights are bf16 either way. All reductions stay f32 in-kernel.
    pub act_dtype: ActDtype,
    /// Bytes per activation element (2 or 4).
    pub act_size: u64,
    pub conv2d: ConvPipeline,
    pub conv2d_wide: ConvPipeline,
    pub conv2d_small_n: ConvPipeline,
    pub group_norm: WgpuPipeline,
    pub silu: WgpuPipeline,
    pub upsample: WgpuPipeline,
    pub add: WgpuPipeline,
    /// Mid-block 1-head self-attention QKV/proj. Single config covers all four
    /// linears (same shape: spatial tokens x C channels).
    pub matmul: WgpuPipeline,
    pub matmul_op: MatMulF32,
    pub bcast_add: WgpuPipeline,
    pub sdpa_large_d: WgpuPipeline,
    pub transpose12: WgpuPipeline,
}

/// Mid-block linear matmul tile (tokens x C @ C x C). tn must be even for the
/// f16-act tiles path; the same config serves the f32 fallback.
const VAE_MATMUL_CFG: MatMulConfig = MatMulConfig {
    bm: 64,
    bn: 64,
    bk: 16,
    tm: 4,
    tn: 4,
    b_nmajor: false,
};

impl VaeDecoderPipelines {
    /// Compile every WGSL pipeline the VAE decoder dispatches. Activations
    /// run f16 when the device supports `SHADER_F16` (reductions stay f32
    /// in-kernel; stores saturate at +-65504), f32 otherwise. Weights are
    /// stored bf16 on-GPU to halve VRAM either way.
    pub async fn compile(backend: &WgpuBackend) -> Result<Self, WgpuError> {
        let act_dtype = if backend.supports_shader_f16() {
            thinfer_core::ops::ActDtype::F16
        } else {
            thinfer_core::ops::ActDtype::F32
        };
        let cfg = &WgslConfig {
            bf16_quant_writes: false,
            act_dtype,
            weight_dtype: thinfer_core::ops::WeightDtype::Bf16,
        };
        Ok(Self {
            act_dtype,
            act_size: act_dtype.bytes_per_elem(),
            conv2d: ConvPipeline::compile(backend, "vae_conv2d", cfg, Conv2dConfig::DEFAULT)
                .await?,
            conv2d_wide: ConvPipeline::compile(backend, "vae_conv2d_wide", cfg, CONV_TILE_WIDE)
                .await?,
            conv2d_small_n: ConvPipeline::compile(
                backend,
                "vae_conv2d_small_n",
                cfg,
                CONV_TILE_SMALL_N,
            )
            .await?,
            group_norm: backend
                .create_pipeline(
                    "vae_group_norm",
                    <GroupNormF32 as GroupNormOp>::wgsl(cfg),
                    "main",
                    <GroupNormF32 as GroupNormOp>::layout(),
                )
                .await?,
            silu: backend
                .create_pipeline("vae_silu", SiluF32::wgsl(cfg), "main", SiluF32::layout())
                .await?,
            upsample: backend
                .create_pipeline(
                    "vae_upsample",
                    <Upsample2dNearestF32 as Upsample2dNearestOp>::wgsl(cfg),
                    "main",
                    <Upsample2dNearestF32 as Upsample2dNearestOp>::layout(),
                )
                .await?,
            add: backend
                .create_pipeline("vae_add", AddF32::wgsl(cfg), "main", AddF32::layout())
                .await?,
            matmul: {
                let op = MatMulF32::new(VAE_MATMUL_CFG);
                backend
                    .create_pipeline(
                        "vae_matmul",
                        &op.wgsl(cfg),
                        "main",
                        <MatMulF32 as MatmulOp>::layout(),
                    )
                    .await?
            },
            matmul_op: MatMulF32::new(VAE_MATMUL_CFG),
            bcast_add: backend
                .create_pipeline(
                    "vae_bcast_add",
                    <BcastAddF32 as BcastAddOp>::wgsl(cfg),
                    "main",
                    <BcastAddF32 as BcastAddOp>::layout(),
                )
                .await?,
            sdpa_large_d: backend
                .create_pipeline(
                    "vae_sdpa_large_d",
                    <SdpaF32LargeD as SdpaOp>::wgsl(cfg),
                    "main",
                    <SdpaF32LargeD as SdpaOp>::layout(),
                )
                .await?,
            transpose12: backend
                .create_pipeline(
                    "vae_transpose12",
                    <Transpose12F32 as Transpose12Op>::wgsl(cfg),
                    "main",
                    <Transpose12F32 as Transpose12Op>::layout(),
                )
                .await?,
        })
    }

    pub fn kernel_keys() -> [KernelKey; 9] {
        let kk = |id: &'static str| KernelKey {
            kernel_id: id,
            hint: String::new(),
        };
        [
            kk(<Conv2dF32 as Conv2dOp>::KERNEL_ID),
            kk(<GroupNormF32 as GroupNormOp>::KERNEL_ID),
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
// Forward driver
// ============================================================================

/// Runtime shape inputs to the decoder front. `h_in` / `w_in` are the latent
/// spatial dims (i.e. the input to `post_quant_conv`/`conv_in`); image-side
/// dims are `h_in * vae_scale_factor` and `w_in * vae_scale_factor`.
#[derive(Clone, Copy, Debug)]
pub struct VaeForwardConfig {
    pub batch: usize,
    pub h_in: usize,
    pub w_in: usize,
}

#[derive(Debug)]
pub enum VaeForwardError {
    Wgpu(WgpuError),
}

impl From<WgpuError> for VaeForwardError {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}

fn conv2d_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    cfg_in: (u32, u32, u32, u32),           // (b, cin, h_in, w_in)
    cfg_out: (u32, u32, u32),               // (cout, h_out, w_out)
    kernel: (u32, u32, u32, u32, u32, u32), // (kh, kw, pad_h, pad_w, stride_h, stride_w)
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 64];
    let fields: [u32; 16] = [
        cfg_in.0, cfg_in.1, cfg_out.0, cfg_in.2, cfg_in.3, cfg_out.1, cfg_out.2, kernel.0,
        kernel.1, kernel.2, kernel.3, kernel.4, kernel.5, 0, 0, 0,
    ];
    for (i, v) in fields.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    scope.write_uniform(&bytes)
}

fn group_norm_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    b: u32,
    c: u32,
    g: u32,
    h: u32,
    w: u32,
    eps: f32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&b.to_le_bytes());
    bytes[4..8].copy_from_slice(&c.to_le_bytes());
    bytes[8..12].copy_from_slice(&g.to_le_bytes());
    bytes[12..16].copy_from_slice(&h.to_le_bytes());
    bytes[16..20].copy_from_slice(&w.to_le_bytes());
    bytes[20..24].copy_from_slice(&eps.to_le_bytes());
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

/// Activation tensor shape used throughout the VAE decoder. Public only
/// to make `VaeStageSample::shape` reachable from diag consumers (tests).
#[derive(Clone, Copy, Debug)]
pub struct ActShape {
    pub b: u32,
    pub c: u32,
    pub h: u32,
    pub w: u32,
}

impl ActShape {
    fn bytes(self, act_size: u64) -> u64 {
        (self.b * self.c * self.h * self.w) as u64 * act_size
    }
    fn elems(self) -> u32 {
        self.b * self.c * self.h * self.w
    }
}

/// Conv2d with bias. Output channel count `cout` and `(kh, kw, pad)` describe
/// the kernel; stride is fixed at 1 (no decoder conv strides differently).
#[allow(clippy::too_many_arguments)]
fn conv2d_forward<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &VaeDecoderPipelines,
    x_in: BatchBuf<'wsp>,
    in_shape: ActShape,
    weights: &'wsp Conv2dBufs,
    cout: u32,
    kh: u32,
    kw: u32,
    pad: u32,
) -> Result<(BatchBuf<'wsp>, ActShape), WgpuError> {
    // stride=1 keeps spatial dims when pad == (k-1)/2; for 1x1 with pad=0
    // dims are also preserved. No other shape is used by the decoder.
    let h_out = in_shape.h + 2 * pad - kh + 1;
    let w_out = in_shape.w + 2 * pad - kw + 1;
    let out_shape = ActShape {
        b: in_shape.b,
        c: cout,
        h: h_out,
        w: w_out,
    };
    let out = scope.alloc(out_shape.bytes(pipelines.act_size))?;
    let u = conv2d_uniform(
        scope,
        (in_shape.b, in_shape.c, in_shape.h, in_shape.w),
        (cout, h_out, w_out),
        (kh, kw, pad, pad, 1, 1),
    )?;
    let w = scope.import(&weights.weight);
    let bias = scope.import(&weights.bias);
    let m_spatial = h_out * w_out;
    // Tile-regime selection: see CONV_TILE_WIDE / CONV_TILE_SMALL_N.
    let conv = if cout <= 4 {
        &pipelines.conv2d_small_n
    } else if m_spatial >= 65536 && cout >= 64 {
        &pipelines.conv2d_wide
    } else {
        &pipelines.conv2d
    };
    scope.conv2d(
        &conv.pipeline,
        &conv.op,
        x_in,
        w,
        bias,
        u,
        out,
        cout,
        m_spatial,
        in_shape.b,
    )?;
    Ok((out, out_shape))
}

fn group_norm_forward<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &VaeDecoderPipelines,
    x_in: BatchBuf<'wsp>,
    shape: ActShape,
    weights: &'wsp GroupNormBufs,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let out = scope.alloc(shape.bytes(pipelines.act_size))?;
    let u = group_norm_uniform(
        scope,
        shape.b,
        shape.c,
        NORM_NUM_GROUPS as u32,
        shape.h,
        shape.w,
        NORM_EPS,
    )?;
    let w = scope.import(&weights.weight);
    let bias = scope.import(&weights.bias);
    scope.group_norm::<GroupNormF32>(
        &pipelines.group_norm,
        x_in,
        w,
        bias,
        u,
        out,
        shape.b * NORM_NUM_GROUPS as u32,
    )?;
    Ok(out)
}

fn silu_forward<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &VaeDecoderPipelines,
    x_in: BatchBuf<'wsp>,
    shape: ActShape,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let out = scope.alloc(shape.bytes(pipelines.act_size))?;
    scope.dispatch_op::<SiluF32>(&pipelines.silu, &[x_in], out)?;
    Ok(out)
}

fn add_forward<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &VaeDecoderPipelines,
    a: BatchBuf<'wsp>,
    b: BatchBuf<'wsp>,
    shape: ActShape,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let out = scope.alloc(shape.bytes(pipelines.act_size))?;
    scope.dispatch_op::<AddF32>(&pipelines.add, &[a, b], out)?;
    Ok(out)
}

fn bcast_add_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    c: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&c.to_le_bytes());
    scope.write_uniform(&bytes)
}

fn transpose12_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    d0: u32,
    d1: u32,
    d2: u32,
    d3: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&d0.to_le_bytes());
    bytes[4..8].copy_from_slice(&d1.to_le_bytes());
    bytes[8..12].copy_from_slice(&d2.to_le_bytes());
    bytes[12..16].copy_from_slice(&d3.to_le_bytes());
    scope.write_uniform(&bytes)
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
    has_mask: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&b.to_le_bytes());
    bytes[4..8].copy_from_slice(&h_q.to_le_bytes());
    bytes[8..12].copy_from_slice(&h_kv.to_le_bytes());
    bytes[12..16].copy_from_slice(&s_q.to_le_bytes());
    bytes[16..20].copy_from_slice(&s_k.to_le_bytes());
    bytes[20..24].copy_from_slice(&d.to_le_bytes());
    bytes[24..28].copy_from_slice(&scale.to_le_bytes());
    bytes[28..32].copy_from_slice(&has_mask.to_le_bytes());
    scope.write_uniform(&bytes)
}

/// 4-byte zero-filled scratch for the SDPA mask binding when `has_mask=0`.
/// The kernel never reads from it (gated by `select(0.0, ..., has_mask != 0)`)
/// but WGSL still requires a non-empty storage buffer of the declared type.
fn sdpa_mask_stub<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    scope.write_uniform(&0f32.to_le_bytes())
}

/// Linear with bias on a 2D row-major tensor: `out [rows, out_dim] = x [rows,
/// in_dim] @ W [in_dim, out_dim] + b [out_dim]`. Weight is uploaded transposed
/// at load (`TransposePolicy::Linear2D`).
fn linear_bias_forward<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &VaeDecoderPipelines,
    x: BatchBuf<'wsp>,
    w: &'wsp LinearBufs,
    rows: u32,
    in_dim: u32,
    out_dim: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let out_bytes = (rows * out_dim) as u64 * pipelines.act_size;
    let pre = scope.alloc(out_bytes)?;
    let dims = scope.u32x4_uniform(rows, out_dim, in_dim, 0)?;
    let w_b = scope.import(&w.weight);
    scope.matmul(
        &pipelines.matmul,
        &pipelines.matmul_op,
        x,
        w_b,
        dims,
        pre,
        rows,
        out_dim,
    )?;
    let out = scope.alloc(out_bytes)?;
    let ba_u = bcast_add_uniform(scope, out_dim)?;
    let bias_b = scope.import(&w.bias);
    scope.bcast_add::<BcastAddF32>(&pipelines.bcast_add, pre, bias_b, ba_u, out, rows * out_dim)?;
    Ok(out)
}

/// Mid-block self-attention (one head, D=C).
///
/// ```text
/// residual = x
/// x = group_norm(x)                     # [B, C, H, W]
/// x = view(B, C, H*W).transpose(1,2)    # [B, H*W, C]
/// q, k, v = to_q(x), to_k(x), to_v(x)   # all [B, H*W, C]
/// x = sdpa(q, k, v)                     # scale = 1/sqrt(C)
/// x = to_out[0](x)                      # [B, H*W, C]
/// x = transpose(1,2).view(B, C, H, W)
/// return residual + x
/// ```
///
/// NCHW<->BHWC uses `transpose12` over a virtual `[B, C, H*W, 1]` view so the
/// inner contiguous axis stays length 1; `q/k/v` feed sdpa as `[B, S, H_q=1,
/// D=C]` (head dim is 1, the reshape is a view).
fn mid_attention_forward<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &VaeDecoderPipelines,
    x_in: BatchBuf<'wsp>,
    in_shape: ActShape,
    bufs: &'wsp MidAttentionBufs,
) -> Result<(BatchBuf<'wsp>, ActShape), WgpuError> {
    let b = in_shape.b;
    let c = in_shape.c;
    let h = in_shape.h;
    let w = in_shape.w;
    let hw = h * w;

    let normed = group_norm_forward(scope, pipelines, x_in, in_shape, &bufs.group_norm)?;

    let tokens_bytes = (b * hw * c) as u64 * pipelines.act_size;
    let tokens = scope.alloc(tokens_bytes)?;
    let t12_u_fwd = transpose12_uniform(scope, b, c, hw, 1)?;
    scope.transpose12::<Transpose12F32>(
        &pipelines.transpose12,
        normed,
        t12_u_fwd,
        tokens,
        b * c * hw,
    )?;

    let rows = b * hw;
    let q = linear_bias_forward(scope, pipelines, tokens, &bufs.to_q, rows, c, c)?;
    let k = linear_bias_forward(scope, pipelines, tokens, &bufs.to_k, rows, c, c)?;
    let v = linear_bias_forward(scope, pipelines, tokens, &bufs.to_v, rows, c, c)?;

    let mask = sdpa_mask_stub(scope)?;
    let attn_out = scope.alloc(tokens_bytes)?;
    let scale = 1.0_f32 / (c as f32).sqrt();
    let sdpa_u = sdpa_uniform(scope, b, 1, 1, hw, hw, c, scale, 0)?;
    scope.sdpa::<SdpaF32LargeD>(
        &pipelines.sdpa_large_d,
        q,
        k,
        v,
        mask,
        sdpa_u,
        attn_out,
        b,
        hw,
        1,
    )?;

    let projected = linear_bias_forward(scope, pipelines, attn_out, &bufs.to_out, rows, c, c)?;

    let restored = scope.alloc(tokens_bytes)?;
    let t12_u_bwd = transpose12_uniform(scope, b, hw, c, 1)?;
    scope.transpose12::<Transpose12F32>(
        &pipelines.transpose12,
        projected,
        t12_u_bwd,
        restored,
        b * c * hw,
    )?;

    let out = add_forward(scope, pipelines, x_in, restored, in_shape)?;
    Ok((out, in_shape))
}

fn upsample_forward<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &VaeDecoderPipelines,
    x_in: BatchBuf<'wsp>,
    shape: ActShape,
) -> Result<(BatchBuf<'wsp>, ActShape), WgpuError> {
    let out_shape = ActShape {
        b: shape.b,
        c: shape.c,
        h: shape.h * 2,
        w: shape.w * 2,
    };
    let out = scope.alloc(out_shape.bytes(pipelines.act_size))?;
    let u = upsample_uniform(scope, shape.b, shape.c, shape.h, shape.w)?;
    scope.upsample2d_nearest::<Upsample2dNearestF32>(
        &pipelines.upsample,
        x_in,
        u,
        out,
        out_shape.elems(),
    )?;
    Ok((out, out_shape))
}

fn resnet_forward<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &VaeDecoderPipelines,
    x_in: BatchBuf<'wsp>,
    in_shape: ActShape,
    out_c: u32,
    bufs: &'wsp ResnetBufs,
    diag: &mut Option<&mut Vec<StagedReadback>>,
    prefix: &str,
) -> Result<(BatchBuf<'wsp>, ActShape), WgpuError> {
    let h0 = group_norm_forward(scope, pipelines, x_in, in_shape, &bufs.norm1)?;
    if !prefix.is_empty() {
        stage_diag(
            scope,
            pipelines.act_size,
            diag,
            &format!("{prefix}.h0"),
            h0,
            in_shape,
        )?;
    }
    let h1 = silu_forward(scope, pipelines, h0, in_shape)?;
    let (h2, h2_shape) =
        conv2d_forward(scope, pipelines, h1, in_shape, &bufs.conv1, out_c, 3, 3, 1)?;
    if !prefix.is_empty() {
        stage_diag(
            scope,
            pipelines.act_size,
            diag,
            &format!("{prefix}.h2"),
            h2,
            h2_shape,
        )?;
    }
    let h3 = group_norm_forward(scope, pipelines, h2, h2_shape, &bufs.norm2)?;
    if !prefix.is_empty() {
        stage_diag(
            scope,
            pipelines.act_size,
            diag,
            &format!("{prefix}.h3"),
            h3,
            h2_shape,
        )?;
    }
    let h4 = silu_forward(scope, pipelines, h3, h2_shape)?;
    let (h5, h5_shape) =
        conv2d_forward(scope, pipelines, h4, h2_shape, &bufs.conv2, out_c, 3, 3, 1)?;
    if !prefix.is_empty() {
        stage_diag(
            scope,
            pipelines.act_size,
            diag,
            &format!("{prefix}.h5"),
            h5,
            h5_shape,
        )?;
    }
    let skip = match &bufs.conv_shortcut {
        Some(cs) => {
            let (s, _) = conv2d_forward(scope, pipelines, x_in, in_shape, cs, out_c, 1, 1, 0)?;
            s
        }
        None => x_in,
    };
    if !prefix.is_empty() {
        stage_diag(
            scope,
            pipelines.act_size,
            diag,
            &format!("{prefix}.skip"),
            skip,
            h5_shape,
        )?;
    }
    let out = add_forward(scope, pipelines, skip, h5, h5_shape)?;
    Ok((out, h5_shape))
}

/// Run `post_quant_conv` (if present) + `conv_in` + mid_block (resnet ->
/// attention -> resnet). Returns the post-mid-block activation
/// `[B, mid, h_in, w_in]` as a `BatchBuf` handle inside the caller's scope.
fn decoder_front<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &VaeDecoderPipelines,
    arch: &KlVaeConfig,
    cfg: &VaeForwardConfig,
    bufs: &'wsp VaeDecoderBufs,
    latents_in: BatchBuf<'wsp>,
) -> Result<(BatchBuf<'wsp>, ActShape), VaeForwardError> {
    let b = cfg.batch as u32;
    let lat_c = arch.latent_channels as u32;
    let mid_c = arch.mid_channels() as u32;
    let latent_shape = ActShape {
        b,
        c: lat_c,
        h: cfg.h_in as u32,
        w: cfg.w_in as u32,
    };

    // Optional 1x1 post_quant_conv (lat -> lat) before conv_in.
    let (z, z_shape) = match &bufs.post_quant_conv {
        Some(pq) => conv2d_forward(
            scope,
            pipelines,
            latents_in,
            latent_shape,
            pq,
            lat_c,
            1,
            1,
            0,
        )?,
        None => (latents_in, latent_shape),
    };

    let (x0, sh0) = conv2d_forward(scope, pipelines, z, z_shape, &bufs.conv_in, mid_c, 3, 3, 1)?;
    let (x1, sh1) = resnet_forward(
        scope,
        pipelines,
        x0,
        sh0,
        mid_c,
        &bufs.mid_block.resnets[0],
        &mut None,
        "",
    )?;
    let (x2, sh2) = mid_attention_forward(scope, pipelines, x1, sh1, &bufs.mid_block.attention)?;
    let (x3, sh3) = resnet_forward(
        scope,
        pipelines,
        x2,
        sh2,
        mid_c,
        &bufs.mid_block.resnets[1],
        &mut None,
        "",
    )?;
    Ok((x3, sh3))
}

type StagedReadback = (
    String,
    ActShape,
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<u8>, WgpuError>> + Send>>,
);

/// Per stage, max bytes we read back. Per `[[feedback-vae-diag-hazard]]`,
/// MiB-scale `read_buffer_via_encoder` probes inside VAE crash the device.
/// 1 KiB = 256 fp32 elements is enough to detect zeros/NaNs/magnitude drift
/// vs a reference without touching the dispatch-side memory pressure.
const STAGE_DIAG_MAX_BYTES: u64 = 1024;

fn stage_diag<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    act_size: u64,
    diag: &mut Option<&mut Vec<StagedReadback>>,
    label: &str,
    buf: BatchBuf<'wsp>,
    shape: ActShape,
) -> Result<(), WgpuError> {
    let Some(sink) = diag.as_deref_mut() else {
        return Ok(());
    };
    let bytes = shape.bytes(act_size).min(STAGE_DIAG_MAX_BYTES);
    let fut = scope.read_buffer_via_encoder(buf, 0, bytes)?;
    sink.push((label.to_string(), shape, Box::pin(fut)));
    Ok(())
}

/// Public per-stage VAE decoder diagnostic sample. `head` holds up to 256
/// fp32 elements read from the start of the stage's output buffer; `shape`
/// is the full ActShape so callers can compute the full element count.
#[derive(Clone, Debug)]
pub struct VaeStageSample {
    pub label: String,
    pub shape: ActShape,
    pub head: Vec<f32>,
}

/// VAE decoder back-half. Records per up_block (and the tail) into its own
/// `BatchScope` and submits between blocks via `on_submitted_work_done` so
/// each submit's GPU time stays well under the Windows TDR window. The
/// up_block-i output is copied into a workspace-pool-allocated carry `WsBuf`
/// that survives across scopes and feeds the next block's input.
/// See [[feedback-no-multi-submit-vae]] for the TDR background.
async fn decoder_back(
    workspace: &Workspace<WgpuBackend>,
    pipelines: &VaeDecoderPipelines,
    arch: &KlVaeConfig,
    bufs: &VaeDecoderBufs,
    front_in: &BufRef,
    front_shape: ActShape,
    image_out: &BufRef,
    mut diag: Option<&mut Vec<StagedReadback>>,
) -> Result<(), VaeForwardError> {
    let mut cur_ref: BufRef = *front_in;
    let mut cur_shape: ActShape = front_shape;
    // Long-lived carry buffer holding the previous up_block's output; kept
    // alive across the next scope so its WsBuf doesn't return to the pool
    // while the next scope still imports it. Drops at function end.
    let mut _carry: Option<WsBuf<WgpuBackend>> = None;

    for (i, ub) in bufs.up_blocks.iter().enumerate() {
        let _scope_guard = trace::scope!(format!("up_block.{i}")).entered();
        let out_c = arch.up_block_out_channels(i) as u32;
        let has_up = ub.upsampler_conv.is_some();
        let next_shape = ActShape {
            b: cur_shape.b,
            c: out_c,
            h: if has_up { cur_shape.h * 2 } else { cur_shape.h },
            w: if has_up { cur_shape.w * 2 } else { cur_shape.w },
        };
        let next_carry = workspace
            .alloc(next_shape.bytes(pipelines.act_size))
            .map_err(VaeForwardError::Wgpu)?;
        let next_carry_ref = next_carry.as_buf_ref();

        // Sub-scope per resnet (and per upsample+up_conv if present) so each
        // sub-step's intermediates return to the workspace pool before the
        // next one allocates. At 512+ image dims the up_block working set
        // otherwise pins many 100-MiB activations in one scope and trips
        // wgpu device-lost. `_sub_in_carry` holds the prior sub-step's output
        // across the scope boundary; reassignment drops it back to the pool.
        let mut sub_in_ref = cur_ref;
        let mut sub_in_shape = cur_shape;
        let mut _sub_in_carry: Option<WsBuf<WgpuBackend>> = None;
        let n_steps = ub.resnets.len() + usize::from(has_up);
        for step in 0..n_steps {
            let is_last = step + 1 == n_steps;
            let is_upsample_step = has_up && is_last;
            let step_out_shape = if is_upsample_step {
                next_shape
            } else {
                ActShape {
                    b: sub_in_shape.b,
                    c: out_c,
                    h: sub_in_shape.h,
                    w: sub_in_shape.w,
                }
            };
            // Last step writes directly into the pre-allocated `next_carry`;
            // middle steps get a fresh carry that survives one iter and then
            // drops when reassigned below.
            let (step_dst_ref, step_dst_keepalive) = if is_last {
                (next_carry_ref, None)
            } else {
                let c = workspace
                    .alloc(step_out_shape.bytes(pipelines.act_size))
                    .map_err(VaeForwardError::Wgpu)?;
                let r = c.as_buf_ref();
                (r, Some(c))
            };

            {
                let scope = workspace.batch();
                let x_in = scope.import(&sub_in_ref);
                if i == 0 && step == 0 {
                    stage_diag(
                        &scope,
                        pipelines.act_size,
                        &mut diag,
                        "front_in",
                        x_in,
                        sub_in_shape,
                    )
                    .map_err(VaeForwardError::Wgpu)?;
                }
                let (out_buf, _out_s) = if is_upsample_step {
                    let up_conv = ub.upsampler_conv.as_ref().expect("has_up implies up_conv");
                    let (xu, shape_u) = upsample_forward(&scope, pipelines, x_in, sub_in_shape)?;
                    stage_diag(
                        &scope,
                        pipelines.act_size,
                        &mut diag,
                        &format!("up{i}.upsample"),
                        xu,
                        shape_u,
                    )
                    .map_err(VaeForwardError::Wgpu)?;
                    let (xc, shape_c) =
                        conv2d_forward(&scope, pipelines, xu, shape_u, up_conv, out_c, 3, 3, 1)?;
                    stage_diag(
                        &scope,
                        pipelines.act_size,
                        &mut diag,
                        &format!("up{i}.upconv"),
                        xc,
                        shape_c,
                    )
                    .map_err(VaeForwardError::Wgpu)?;
                    (xc, shape_c)
                } else {
                    let j = step;
                    let resnet = &ub.resnets[j];
                    let probe_prefix = if diag.is_some() {
                        format!("up{i}.resnet{j}")
                    } else {
                        String::new()
                    };
                    let (r_buf, r_shape) = resnet_forward(
                        &scope,
                        pipelines,
                        x_in,
                        sub_in_shape,
                        out_c,
                        resnet,
                        &mut diag,
                        &probe_prefix,
                    )?;
                    stage_diag(
                        &scope,
                        pipelines.act_size,
                        &mut diag,
                        &format!("up{i}.resnet{j}"),
                        r_buf,
                        r_shape,
                    )
                    .map_err(VaeForwardError::Wgpu)?;
                    (r_buf, r_shape)
                };
                let dst_buf = scope.import(&step_dst_ref);
                scope
                    .copy_buffer_to_buffer(
                        out_buf,
                        0,
                        dst_buf,
                        0,
                        step_out_shape.bytes(pipelines.act_size),
                    )
                    .map_err(VaeForwardError::Wgpu)?;
                scope
                    .submit_void()
                    .instrument(tracing::debug_span!(
                        target: PHASE,
                        "vae.submit",
                        phase = "up_block_step",
                        idx = i,
                        step = step,
                    ))
                    .await
                    .map_err(VaeForwardError::Wgpu)?;
            }

            sub_in_ref = step_dst_ref;
            sub_in_shape = step_out_shape;
            // Reassigning drops the previous middle-step carry back to the
            // pool. Last step has `None` here; the surviving handle is
            // `next_carry`, which moves into `_carry` below.
            if !is_last {
                _sub_in_carry = step_dst_keepalive;
            }
        }
        debug_assert_eq!(sub_in_shape.b, next_shape.b);
        debug_assert_eq!(sub_in_shape.c, next_shape.c);
        debug_assert_eq!(sub_in_shape.h, next_shape.h);
        debug_assert_eq!(sub_in_shape.w, next_shape.w);

        _carry = Some(next_carry);
        cur_ref = next_carry_ref;
        cur_shape = next_shape;
    }

    // Tail: conv_norm_out -> silu -> conv_out into caller-supplied image_out.
    {
        let _scope_guard = trace::scope!("vae_tail").entered();
        let scope = workspace.batch();
        let cur_buf = scope.import(&cur_ref);
        let img_buf = scope.import(image_out);
        let h_gn = group_norm_forward(&scope, pipelines, cur_buf, cur_shape, &bufs.conv_norm_out)?;
        stage_diag(
            &scope,
            pipelines.act_size,
            &mut diag,
            "conv_norm_out",
            h_gn,
            cur_shape,
        )
        .map_err(VaeForwardError::Wgpu)?;
        let h_silu = silu_forward(&scope, pipelines, h_gn, cur_shape)?;
        stage_diag(
            &scope,
            pipelines.act_size,
            &mut diag,
            "silu_out",
            h_silu,
            cur_shape,
        )
        .map_err(VaeForwardError::Wgpu)?;

        let out_c = OUT_CHANNELS as u32;
        let out_shape = ActShape {
            b: cur_shape.b,
            c: out_c,
            h: cur_shape.h,
            w: cur_shape.w,
        };
        let u = conv2d_uniform(
            &scope,
            (cur_shape.b, cur_shape.c, cur_shape.h, cur_shape.w),
            (out_c, out_shape.h, out_shape.w),
            (3, 3, 1, 1, 1, 1),
        )?;
        let w = scope.import(&bufs.conv_out.weight);
        let bias = scope.import(&bufs.conv_out.bias);
        // conv_out is cout=3: small-N tile regime.
        scope.conv2d(
            &pipelines.conv2d_small_n.pipeline,
            &pipelines.conv2d_small_n.op,
            h_silu,
            w,
            bias,
            u,
            img_buf,
            out_c,
            out_shape.h * out_shape.w,
            cur_shape.b,
        )?;
        scope
            .submit_void()
            .instrument(tracing::debug_span!(target: PHASE, "vae.submit", phase = "vae_tail"))
            .await
            .map_err(VaeForwardError::Wgpu)?;
    }

    Ok(())
}

/// KL-VAE decoder forward (single-shot, no tiling). Builds one BatchScope for
/// the front (post_quant + conv_in + mid_block) then runs the back-half. The
/// spatial latent `latents_in` is `[B, lat, h_in, w_in]` (already pre-
/// transformed by the caller); `image_out` receives `[B, 3, h_in*F, w_in*F]`
/// fp32 RGB in `[-1, 1]`.
pub async fn decoder_forward(
    workspace: &Workspace<WgpuBackend>,
    pipelines: &VaeDecoderPipelines,
    arch: &KlVaeConfig,
    cfg: &VaeForwardConfig,
    bufs: &VaeDecoderBufs,
    latents_in: &BufRef,
    image_out: &BufRef,
) -> Result<(), VaeForwardError> {
    let b = cfg.batch as u32;
    let h_in = cfg.h_in as u32;
    let w_in = cfg.w_in as u32;
    let front_shape = ActShape {
        b,
        c: arch.mid_channels() as u32,
        h: h_in,
        w: w_in,
    };
    let front_carry = workspace
        .alloc(front_shape.bytes(pipelines.act_size))
        .map_err(VaeForwardError::Wgpu)?;
    let front_carry_ref = front_carry.as_buf_ref();
    {
        let scope = workspace.batch();
        let lat = scope.import(latents_in);
        let f_out = scope.import(&front_carry_ref);
        let (front_buf, fshape) = decoder_front(&scope, pipelines, arch, cfg, bufs, lat)?;
        debug_assert_eq!(
            fshape.bytes(pipelines.act_size),
            front_shape.bytes(pipelines.act_size)
        );
        scope
            .copy_buffer_to_buffer(
                front_buf,
                0,
                f_out,
                0,
                front_shape.bytes(pipelines.act_size),
            )
            .map_err(VaeForwardError::Wgpu)?;
        scope
            .submit_void()
            .instrument(tracing::debug_span!(target: PHASE, "vae.submit", phase = "front"))
            .await
            .map_err(VaeForwardError::Wgpu)?;
    }
    decoder_back(
        workspace,
        pipelines,
        arch,
        bufs,
        &front_carry_ref,
        front_shape,
        image_out,
        None,
    )
    .await?;
    Ok(())
}

/// Tiled-decode knob. Tile size is chosen in latent space; image-space tile
/// is `latent_tile * vae_scale_factor`. `overlap` is the latent-space halo
/// each tile shares with its neighbours.
#[derive(Clone, Copy, Debug)]
pub struct VaeTileConfig {
    pub latent_tile: u32,
    pub overlap: u32,
}

impl Default for VaeTileConfig {
    fn default() -> Self {
        // 64 latent -> 512 image tile; last-up-block activation is bounded at
        // [1, mid, 512, 512]. Overlap 8 latent -> 64 image px feather.
        Self {
            latent_tile: 64,
            overlap: 8,
        }
    }
}

/// Compute tile start positions along one axis. Mirrors `intabai/web/src/sd15/vae.ts`.
fn tile_starts(dim: u32, tile: u32) -> Vec<u32> {
    if dim <= tile {
        return vec![0];
    }
    let stride = tile.saturating_sub(8); // overlap-agnostic spacing helper
    let n = (dim - tile).div_ceil(stride.max(1)) + 1;
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        let v = if n == 1 {
            0
        } else {
            ((i as u64 * (dim - tile) as u64) + (n as u64 - 1) / 2) / (n as u64 - 1)
        };
        out.push(v as u32);
    }
    out
}

/// 1D linear-ramp blend weights of length `len`. Ramps 0 -> 1 over `blend`
/// pixels at the start (unless `first_edge`) and 1 -> 0 over `blend` at the
/// end (unless `last_edge`). Mirrors `intabai/web/src/sd15/vae.ts:ramp1D`.
fn ramp_1d(len: u32, blend: u32, first_edge: bool, last_edge: bool) -> Vec<f32> {
    let blend_f = (blend + 1) as f32;
    (0..len)
        .map(|i| {
            let mut w = 1.0_f32;
            if !first_edge && i < blend {
                w *= (i as f32 + 1.0) / blend_f;
            }
            if !last_edge && i >= len - blend {
                w *= (len as f32 - i as f32) / blend_f;
            }
            w
        })
        .collect()
}

/// Tiled KL-VAE decoder. Hybrid scheme:
/// - post_quant + conv_in + mid_block (which includes a global self-attention)
///   run once on the full latent; the result is read back to host.
/// - up_blocks + tail run per latent-space tile from a sliced+uploaded
///   `[B, mid, tile_h, tile_w]` input. Each tile produces a `[B, 3,
///   tile_h*F, tile_w*F]` patch, blended into a host accumulator with a
///   linear-ramp feather (matches diffusers `tiled_decode`).
///
/// Output is host-side CHW `[3, h_in*F, w_in*F]` in `[-1, 1]`. The caller
/// clamps + converts to u8. Setting `latent_tile >= max(h_in, w_in)` collapses
/// to a single tile (equals `decoder_forward` plus a host roundtrip).
#[allow(clippy::too_many_arguments)]
pub async fn decoder_forward_tiled(
    backend: &WgpuBackend,
    workspace: &Workspace<WgpuBackend>,
    pipelines: &VaeDecoderPipelines,
    arch: &KlVaeConfig,
    cfg: &VaeForwardConfig,
    bufs: &VaeDecoderBufs,
    latents_in: &BufRef,
    tile_cfg: VaeTileConfig,
    mut diag_sink: Option<&mut Vec<VaeStageSample>>,
) -> Result<Vec<f32>, VaeForwardError> {
    let b = cfg.batch as u32;
    assert_eq!(b, 1, "tiled decode supports B=1 only (matches single-shot)");

    let scale_factor = arch.vae_scale_factor() as u32;
    let mid_c = arch.mid_channels() as u32;
    let h_in = cfg.h_in as u32;
    let w_in = cfg.w_in as u32;
    let img_h = h_in * scale_factor;
    let img_w = w_in * scale_factor;
    let out_c = OUT_CHANNELS as u32;
    let plane = (img_h as usize) * (img_w as usize);

    // ---------------- Front: post_quant + conv_in + mid_block on full latent ----------------
    // mid_out is allocated raw (outside the workspace pool) because it has to
    // survive across the front BatchScope and into each per-tile back pass.
    let act_size = pipelines.act_size;
    let mid_elems = (b * mid_c * h_in * w_in) as u64;
    let mid_bytes = mid_elems * act_size;
    let mid_id = backend.allocate(mid_bytes).map_err(VaeForwardError::Wgpu)?;
    let mid_buf_ref = BufRef::new(mid_id, mid_bytes);

    {
        let scope = workspace.batch();
        let lat = scope.import(latents_in);
        let mid = scope.import(&mid_buf_ref);
        let (front_buf, front_shape) = decoder_front(&scope, pipelines, arch, cfg, bufs, lat)?;
        debug_assert_eq!(front_shape.b, b);
        debug_assert_eq!(front_shape.c, mid_c);
        debug_assert_eq!(front_shape.h, h_in);
        debug_assert_eq!(front_shape.w, w_in);
        scope
            .copy_buffer_to_buffer(front_buf, 0, mid, 0, mid_bytes)
            .map_err(VaeForwardError::Wgpu)?;
        scope
            .submit_void()
            .instrument(tracing::debug_span!(target: PHASE, "vae.submit", phase = "front"))
            .await
            .map_err(VaeForwardError::Wgpu)?;
    }

    let mid_host_bytes = backend
        .read_buffer(mid_id, 0, mid_bytes)
        .instrument(tracing::debug_span!(target: PHASE, "vae.readback", phase = "front_mid"))
        .await
        .map_err(VaeForwardError::Wgpu)?;
    backend.free(mid_id);
    // Kept as raw act-dtype bytes: the tile slicer below copies rows
    // byte-wise and re-uploads without a host dtype round-trip.
    debug_assert_eq!(mid_host_bytes.len() as u64, mid_elems * act_size);

    // ---------------- Tile loop ----------------
    let tile = tile_cfg.latent_tile;
    let blend_img = tile_cfg.overlap * scale_factor;
    let y_starts = tile_starts(h_in, tile);
    let x_starts = tile_starts(w_in, tile);

    let mut acc = vec![0.0_f32; (out_c as usize) * plane];
    let mut weights = vec![0.0_f32; plane];

    let n_tiles = y_starts.len() * x_starts.len();
    let mut tile_idx: usize = 0;
    for (yi, &ly) in y_starts.iter().enumerate() {
        let is_first_y = yi == 0;
        let is_last_y = yi + 1 == y_starts.len();
        for (xi, &lx) in x_starts.iter().enumerate() {
            let is_first_x = xi == 0;
            let is_last_x = xi + 1 == x_starts.len();
            let _tile_guard = trace::scope!(format!("tile.{tile_idx}")).entered();
            tracing::debug!(target: PHASE, idx = tile_idx, of = n_tiles, "vae.tile.begin");

            let tile_h = tile.min(h_in - ly);
            let tile_w = tile.min(w_in - lx);

            // ---- Slice mid_out -> host tile bytes (CHW contiguous) ----
            // Byte-wise row copies in the act dtype; no host dtype round-trip.
            let asz = act_size as usize;
            let tile_in_elems = (b * mid_c * tile_h * tile_w) as usize;
            let mut tile_in_host = vec![0u8; tile_in_elems * asz];
            for c in 0..mid_c {
                for dy in 0..tile_h {
                    let src_row = ((c as usize) * (h_in as usize) * (w_in as usize)
                        + ((ly + dy) as usize) * (w_in as usize)
                        + (lx as usize))
                        * asz;
                    let dst_row = ((c as usize) * (tile_h as usize) * (tile_w as usize)
                        + (dy as usize) * (tile_w as usize))
                        * asz;
                    let row_bytes = (tile_w as usize) * asz;
                    tile_in_host[dst_row..dst_row + row_bytes]
                        .copy_from_slice(&mid_host_bytes[src_row..src_row + row_bytes]);
                }
            }

            // Pre-allocate per-tile front_in + image_out via the workspace pool;
            // bind their BufRefs in locals BEFORE the scope so `scope.import`
            // can borrow them for `'wsp`.
            let tile_front_in = workspace
                .alloc((tile_in_elems as u64) * act_size)
                .map_err(VaeForwardError::Wgpu)?;
            backend
                .write_buffer(tile_front_in.id(), 0, &tile_in_host)
                .map_err(VaeForwardError::Wgpu)?;

            let tile_img_h = tile_h * scale_factor;
            let tile_img_w = tile_w * scale_factor;
            let tile_img_bytes = (b * out_c * tile_img_h * tile_img_w) as u64 * act_size;
            let tile_image_out = workspace
                .alloc(tile_img_bytes)
                .map_err(VaeForwardError::Wgpu)?;

            let tile_front_ref = tile_front_in.as_buf_ref();
            let tile_image_ref = tile_image_out.as_buf_ref();

            let mut local_diag: Vec<StagedReadback> = Vec::new();
            let diag_arg: Option<&mut Vec<StagedReadback>> = if diag_sink.is_some() {
                Some(&mut local_diag)
            } else {
                None
            };
            decoder_back(
                workspace,
                pipelines,
                arch,
                bufs,
                &tile_front_ref,
                ActShape {
                    b,
                    c: mid_c,
                    h: tile_h,
                    w: tile_w,
                },
                &tile_image_ref,
                diag_arg,
            )
            .await?;

            // Read the tile RGB out via a separate submit. Folding the staging
            // copy into the compute submit caused the readback to come back as
            // the 0xAB pre-fill sentinel for late tile dispatches and produced
            // gray PNGs.
            let tile_img_host_bytes = backend
                .read_buffer(tile_image_ref.id, tile_image_ref.offset, tile_img_bytes)
                .instrument(tracing::debug_span!(target: PHASE, "vae.readback", phase = "tile", idx = tile_idx))
                .await
                .map_err(VaeForwardError::Wgpu)?;
            let tile_img_host = act_bytes_to_f32_vec(act_size, &tile_img_host_bytes);
            // tile_front_in / tile_image_out drop at end of iter -> back to pool.

            // Drain per-tile diag samples into the caller's sink. For
            // multi-tile decodes we tag the label with the tile index.
            if let Some(sink) = diag_sink.as_deref_mut() {
                for (label, shape, fut) in local_diag.drain(..) {
                    let bytes = fut.await.map_err(VaeForwardError::Wgpu)?;
                    let label = if n_tiles == 1 {
                        label
                    } else {
                        format!("tile{tile_idx}.{label}")
                    };
                    sink.push(VaeStageSample {
                        label,
                        shape,
                        head: act_bytes_to_f32_vec(act_size, &bytes),
                    });
                }
                // Also dump a head sample of the per-tile final image_out
                // (post-conv_out), the byte-for-byte counterpart to py's
                // `decoder.conv_out` output.
                let head_n = (STAGE_DIAG_MAX_BYTES as usize / 4).min(tile_img_host.len());
                let label = if n_tiles == 1 {
                    "conv_out".to_string()
                } else {
                    format!("tile{tile_idx}.conv_out")
                };
                sink.push(VaeStageSample {
                    label,
                    shape: ActShape {
                        b,
                        c: out_c,
                        h: tile_img_h,
                        w: tile_img_w,
                    },
                    head: tile_img_host[..head_n].to_vec(),
                });
            }

            // ---- Blend tile into accumulator ----
            let v_ramp = ramp_1d(tile_img_h, blend_img, is_first_y, is_last_y);
            let h_ramp = ramp_1d(tile_img_w, blend_img, is_first_x, is_last_x);
            let tile_origin_y = ly * scale_factor;
            let tile_origin_x = lx * scale_factor;
            let tile_plane = (tile_img_h as usize) * (tile_img_w as usize);

            for ty in 0..tile_img_h {
                let oy = tile_origin_y + ty;
                if oy >= img_h {
                    break;
                }
                let w_row = v_ramp[ty as usize];
                for tx in 0..tile_img_w {
                    let ox = tile_origin_x + tx;
                    if ox >= img_w {
                        break;
                    }
                    let w = w_row * h_ramp[tx as usize];
                    let o_idx = (oy as usize) * (img_w as usize) + (ox as usize);
                    let t_idx = (ty as usize) * (tile_img_w as usize) + (tx as usize);
                    for c in 0..out_c as usize {
                        acc[c * plane + o_idx] += tile_img_host[c * tile_plane + t_idx] * w;
                    }
                    weights[o_idx] += w;
                }
            }
            tile_idx += 1;
        }
    }

    // ---- Normalize ----
    for i in 0..plane {
        let w = weights[i];
        if w > 0.0 {
            let inv = 1.0 / w;
            for c in 0..out_c as usize {
                acc[c * plane + i] *= inv;
            }
        }
    }

    Ok(acc)
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    debug_assert_eq!(bytes.len() % 4, 0);
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

/// Act-dtype-aware readback conversion: `act_size == 2` decodes f16, else f32.
fn act_bytes_to_f32_vec(act_size: u64, bytes: &[u8]) -> Vec<f32> {
    if act_size == 2 {
        bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()
    } else {
        bytes_to_f32_vec(bytes)
    }
}

/// Act-dtype-aware upload conversion: `act_size == 2` encodes f16, else f32.
fn f32s_to_act_bytes(act_size: u64, vals: &[f32]) -> Vec<u8> {
    if act_size == 2 {
        let mut out = Vec::with_capacity(vals.len() * 2);
        for v in vals {
            out.extend_from_slice(&half::f16::from_f32(*v).to_le_bytes());
        }
        out
    } else {
        f32_slice_as_bytes(vals).to_vec()
    }
}

fn f32_slice_as_bytes(s: &[f32]) -> &[u8] {
    // Safe: f32 is plain-old-data, alignment of &[f32] is >= alignment of u8.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

/// High-level KL-VAE decoder. Owns the compiled pipelines, residency handles,
/// and the per-model arch config so the orchestrator can call `decode` without
/// re-wiring the lower-level `decoder_forward_tiled` plumbing.
pub struct VaeDecoder {
    pub pipelines: VaeDecoderPipelines,
    pub handles: VaeDecoderHandles,
    pub tile_cfg: VaeTileConfig,
    pub arch: KlVaeConfig,
}

#[derive(Debug)]
pub enum VaeDecodeError<SE: core::fmt::Debug> {
    Forward(VaeForwardError),
    Residency(ResidencyError<SE, WgpuError>),
}

impl<SE: core::fmt::Debug> From<VaeForwardError> for VaeDecodeError<SE> {
    fn from(e: VaeForwardError) -> Self {
        Self::Forward(e)
    }
}

impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for VaeDecodeError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

impl<SE: core::fmt::Debug> From<WgpuError> for VaeDecodeError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Forward(VaeForwardError::Wgpu(e))
    }
}

impl VaeDecoder {
    /// Apply the configured host pre-transform to the raw spatial latent.
    fn pretransform(&self, latents: &[f32]) -> Vec<f32> {
        match self.arch.pretransform {
            LatentPretransform::None => latents.to_vec(),
            LatentPretransform::Scalar { scaling, shift } => {
                latents.iter().map(|z| z / scaling + shift).collect()
            }
        }
    }

    /// Decode `[lat, h_in, w_in]` spatial latents (B=1 assumed; `latents`
    /// length must equal `latent_channels * h_in * w_in`). Applies the
    /// configured pre-transform internally. Returns CHW `[3, h_in*F, w_in*F]`
    /// fp32 RGB in `[-1, 1]`.
    pub async fn decode<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        scratch: &mut Workspace<WgpuBackend>,
        latents: &[f32],
        h_in: usize,
        w_in: usize,
    ) -> Result<Vec<f32>, VaeDecodeError<S::Error>> {
        self.decode_inner(backend, residency, scratch, latents, h_in, w_in, None)
            .await
    }

    /// Like `decode`, but additionally captures per-stage head samples inside
    /// `decoder_back` (front_in, every up_block resnet/upsample/upconv output,
    /// conv_norm_out, silu_out, final conv_out). Each sample is a small head
    /// slice (<=256 fp32 elements) per the VAE diag hazard rule. Order =
    /// data-flow order; multi-tile labels are prefixed with `tile{i}.`.
    pub async fn decode_with_diag<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        scratch: &mut Workspace<WgpuBackend>,
        latents: &[f32],
        h_in: usize,
        w_in: usize,
        diag_sink: &mut Vec<VaeStageSample>,
    ) -> Result<Vec<f32>, VaeDecodeError<S::Error>> {
        self.decode_inner(
            backend,
            residency,
            scratch,
            latents,
            h_in,
            w_in,
            Some(diag_sink),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn decode_inner<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        scratch: &mut Workspace<WgpuBackend>,
        latents: &[f32],
        h_in: usize,
        w_in: usize,
        diag_sink: Option<&mut Vec<VaeStageSample>>,
    ) -> Result<Vec<f32>, VaeDecodeError<S::Error>> {
        let elems = self.arch.latent_channels * h_in * w_in;
        assert_eq!(
            latents.len(),
            elems,
            "VaeDecoder::decode: expected {elems} latents, got {}",
            latents.len()
        );
        let scaled = self.pretransform(latents);
        let bytes = f32s_to_act_bytes(self.pipelines.act_size, &scaled);
        let latents_buf = scratch.alloc(bytes.len() as u64)?;
        backend.write_buffer(latents_buf.id, 0, &bytes)?;
        let cfg = VaeForwardConfig {
            batch: 1,
            h_in,
            w_in,
        };
        let views = self
            .handles
            .acquire(residency, backend)
            .instrument(tracing::debug_span!(target: PHASE, "vae.acquire"))
            .await?;
        let bufs = views.bufs();
        Ok(decoder_forward_tiled(
            backend,
            &*scratch,
            &self.pipelines,
            &self.arch,
            &cfg,
            &bufs,
            &latents_buf,
            self.tile_cfg,
            diag_sink,
        )
        .await?)
    }
}
