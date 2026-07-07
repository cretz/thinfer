//! LTX-2.3 causal video VAE decode (`CausalVideoAutoencoder`, decoder only).
//!
//! Source of truth: `third-party/LTX-2/.../model/video_vae/{video_vae,resnet,
//! sampling,convolution,ops}.py` + the on-disk safetensors `__metadata__.config`
//! (authoritative; it CORRECTS ltx-plan): `timestep_conditioning=false`,
//! `causal_decoder=false`, `normalize_latent_channels=false`, `use_quant_conv=
//! false`, `norm_layer=pixel_norm`, `patch_size=4`, `latent_channels=128`,
//! `spatial_padding_mode=zeros`, `scaling_factor=1.0`. So the decoder is a PLAIN
//! conv stack -- NOT timestep-conditioned, NOT noise-injecting (no temb, no
//! scale_shift_table, no inject_noise).
//!
//! Unlike the Wan VAE (`wan/vae.rs`), this decoder is NOT causal: every conv uses
//! `causal=False` = symmetric REPLICATE temporal padding (one frame each side for
//! k=3) plus zeros spatial padding. There is no per-frame `feat_cache` streaming;
//! the whole `[128, F, H, W]` latent decodes in one pass. Net-new ops vs the Wan
//! core: `PixelNorm3d` (weightless channel-RMS) + `DepthToSpace3d` (pixel-shuffle
//! upsampler). Temporal replicate pad reuses `concat_time`; the final 4x4 spatial
//! unpatchify + per-channel un-normalize run host-side.
//!
//! Decoder graph (feature_channels start = base*8 = 1024):
//!   conv_in 128->1024 -> up_blocks[0..9] -> pixel_norm -> silu -> conv_out
//!   1024->48 -> (host) unpatchify 4x4 -> 3ch video `[3, F, 32h, 32w]`.
//! up_blocks (flat, = reversed `decoder_blocks`): res_x(2) / compress_all(m2) /
//! res_x(2) / compress_all(m1) / res_x(4) / compress_time(m2) / res_x(6) /
//! compress_space(m2) / res_x(4). `compress_*` = `DepthToSpaceUpsample`: a k3
//! conv that expands channels then a depth-to-space shuffle (+ first-frame drop
//! when it upsamples time).

use thinfer_core::backend::{BufRef, WgpuBackend, WgpuError};
use thinfer_core::ops::{
    ActDtype, AddF32, ConcatTimeF32, ConcatTimeOp, Conv3dConfig, Conv3dF32, Conv3dOp,
    DepthToSpace3dF32, DepthToSpace3dOp, Op, PixelNorm3dF32, PixelNorm3dOp, SiluF32, WeightDtype,
    WgslConfig,
};
use thinfer_core::residency::{GpuView, ResidencyError, WeightHandle, WeightResidency};
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::{BatchScope, Workspace};

use crate::common::loader::{LoadError, register_passthrough};
use crate::common::vae_tiling::{
    accumulate_spatial_wsum, blend_tile, feather_1d, gather_subtile, plan_temporal_tiles,
    plan_tiles, trapezoid_mask,
};

// ============================================================================
// Config: decoder geometry derived from the safetensors config.
// ============================================================================

/// Per-channel latent un-normalization eps + PixelNorm eps (torch default 1e-8).
const PIXEL_NORM_EPS: f32 = 1e-8;
/// Latent channels (`latent_channels`) and decoder input.
pub const LATENT_CHANNELS: usize = 128;
/// Final spatial pixel-unshuffle factor (`patch_size`).
pub const PATCH_SIZE: usize = 4;
/// Output RGB channels.
pub const OUT_CHANNELS: usize = 3;
/// Packed RGB patch channels (`out_channels * patch_size^2 = 3*16 = 48`): the
/// decoder `conv_out` output width AND the encoder `conv_in` input width (the
/// host `patchify` / `unpatchify` boundary).
pub const CONV_OUT_CHANNELS: usize = OUT_CHANNELS * PATCH_SIZE * PATCH_SIZE;
/// Encoder `conv_out` width: 128 latent means + 1 uniform log-variance
/// (`latent_log_var=uniform`); encode takes the first [`LATENT_CHANNELS`] means.
pub const ENC_CONV_OUT_CHANNELS: usize = LATENT_CHANNELS + 1;
/// Spatial / temporal compression (latent cell -> pixels / frames).
pub const SPATIAL_SCALE: usize = 32;
pub const TEMPORAL_SCALE: usize = 8;

/// One decoder up_block, in flat order (= reversed `decoder_blocks`).
#[derive(Clone, Debug)]
enum UpBlockKind {
    /// `res_x` = `UNetMidBlock3D`: `n_layers` channel-preserving resnets.
    ResX { n_layers: usize, channels: usize },
    /// `compress_*` = `DepthToSpaceUpsample`: a k3 conv `in -> conv_out_c` then a
    /// `(p1,p2,p3)` depth-to-space shuffle, dropping the leading frame when
    /// `p1 == 2`. Output channels = `in / multiplier`. `residual` adds the
    /// input's own shuffle-and-tile up-shortcut (`compress_all` w/ `residual:
    /// True`; LTX-2 rapid VAE, off for the shipped distilled schedule).
    Upsample {
        conv_out_c: usize,
        out_c: usize,
        p: [usize; 3],
        t_drop: usize,
        residual: bool,
    },
}

#[derive(Clone, Debug)]
pub struct LtxVaeConfig {
    up_blocks: Vec<UpBlockKind>,
    /// `conv_in` output channels (= base_channels * 8).
    mid_channels: usize,
    /// Conv spatial-border `pad_mode` (`decoder_spatial_padding_mode`): 0=zeros
    /// (distilled, explicit in its config), 2=reflect (LTX-2 rapid, the decoder
    /// configurator default when the config omits `spatial_padding_mode`).
    spatial_pad_mode: u32,
}

impl LtxVaeConfig {
    /// The shipped distilled VAE config (verified on disk). `decoder_blocks` in
    /// config order; the decoder walks them reversed.
    pub fn distilled() -> Self {
        // (name, param): res_x param = num_layers; compress_* param = multiplier.
        let decoder_blocks: [(&str, usize); 9] = [
            ("res_x", 4),
            ("compress_space", 2),
            ("res_x", 6),
            ("compress_time", 2),
            ("res_x", 4),
            ("compress_all", 1),
            ("res_x", 2),
            ("compress_all", 2),
            ("res_x", 2),
        ];
        let mut feature = LATENT_CHANNELS * 8; // base_channels(128) * 8 = 1024
        let mut up_blocks = Vec::with_capacity(decoder_blocks.len());
        for (name, param) in decoder_blocks.iter().rev() {
            let block = match *name {
                "res_x" => UpBlockKind::ResX {
                    n_layers: *param,
                    channels: feature,
                },
                "compress_space" | "compress_time" | "compress_all" => {
                    let p = match *name {
                        "compress_space" => [1usize, 2, 2],
                        "compress_time" => [2, 1, 1],
                        _ => [2, 2, 2],
                    };
                    let prod = p[0] * p[1] * p[2];
                    let multiplier = *param;
                    let conv_out_c = prod * feature / multiplier;
                    let out_c = feature / multiplier;
                    let blk = UpBlockKind::Upsample {
                        conv_out_c,
                        out_c,
                        p,
                        t_drop: if p[0] == 2 { 1 } else { 0 },
                        // Distilled decoder_blocks carry no `residual` key.
                        residual: false,
                    };
                    feature = out_c;
                    blk
                }
                other => panic!("unknown decoder block: {other}"),
            };
            up_blocks.push(block);
        }
        Self {
            up_blocks,
            mid_channels: LATENT_CHANNELS * 8,
            // Distilled config sets `spatial_padding_mode=zeros`.
            spatial_pad_mode: 0,
        }
    }

    /// The LTX-2 (non-.3) VAE decoder used by ltx2-rapid (verified on disk from
    /// `Kijai/LTXV2_comfy/VAE/LTX2_video_vae_bf16.safetensors`). Same `conv_in`
    /// 128->1024, patch_size 4, and 32x spatial / 8x temporal scale as distilled,
    /// but a different (symmetric) block schedule: 4 uniform `res_x` (5 layers) with
    /// 3 `compress_all` (`[2,2,2]`, mult 2) between them -> feature 1024,512,256,128.
    /// Reconstructed from the tensor shapes: up_blocks conv `(4096,1024)`,
    /// `(2048,512)`, `(1024,256)`; res_blocks 5 per stage.
    pub fn ltx2_rapid() -> Self {
        let decoder_blocks: [(&str, usize); 7] = [
            ("res_x", 5),
            ("compress_all", 2),
            ("res_x", 5),
            ("compress_all", 2),
            ("res_x", 5),
            ("compress_all", 2),
            ("res_x", 5),
        ];
        let mut feature = LATENT_CHANNELS * 8; // 1024
        let mut up_blocks = Vec::with_capacity(decoder_blocks.len());
        for (name, param) in decoder_blocks.iter().rev() {
            let block = match *name {
                "res_x" => UpBlockKind::ResX {
                    n_layers: *param,
                    channels: feature,
                },
                "compress_all" => {
                    let p = [2usize, 2, 2];
                    let prod = p[0] * p[1] * p[2];
                    let multiplier = *param;
                    let conv_out_c = prod * feature / multiplier;
                    let out_c = feature / multiplier;
                    let blk = UpBlockKind::Upsample {
                        conv_out_c,
                        out_c,
                        p,
                        t_drop: 1,
                        // LTX-2 rapid `compress_all` sets `residual: True`.
                        residual: true,
                    };
                    feature = out_c;
                    blk
                }
                other => panic!("unknown decoder block: {other}"),
            };
            up_blocks.push(block);
        }
        Self {
            up_blocks,
            mid_channels: LATENT_CHANNELS * 8,
            // The rapid config omits `spatial_padding_mode`; the decoder
            // configurator defaults it to reflect (encoder defaults to zeros).
            spatial_pad_mode: 2,
        }
    }

    /// Resident weight bytes (bf16): every conv `weight [cout,cin,3,3,3] + bias
    /// [cout]`, walked in forward order tracking the feature width. Used to carve
    /// the weight footprint out of the VRAM budget before sizing the per-tile
    /// decode workspace (so the budget is honored at any size).
    fn weight_footprint_bytes(&self) -> u64 {
        let conv_bytes = |cin: u64, cout: u64| (cout * cin * 27 + cout) * 2;
        let mut total = conv_bytes(LATENT_CHANNELS as u64, self.mid_channels as u64); // conv_in
        let mut feature = self.mid_channels as u64;
        for blk in &self.up_blocks {
            match blk {
                UpBlockKind::ResX { n_layers, channels } => {
                    let ch = *channels as u64;
                    total += 2 * (*n_layers as u64) * conv_bytes(ch, ch);
                    feature = ch;
                }
                UpBlockKind::Upsample {
                    conv_out_c, out_c, ..
                } => {
                    total += conv_bytes(feature, *conv_out_c as u64);
                    feature = *out_c as u64;
                }
            }
        }
        total + conv_bytes(feature, CONV_OUT_CHANNELS as u64) // conv_out
    }
}

// ============================================================================
// Weight ids / handles. PixelNorm is weightless; resnets have no shortcut (all
// channel-preserving); upsamplers have no time_conv. So a conv = {weight, bias}
// is the only leaf.
// ============================================================================

#[derive(Clone, Debug)]
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

/// All decoder conv weight ids, in forward order.
struct DecoderW {
    conv_in: ConvW,
    /// `up_blocks[i]`: a list of convs (res_x = 2*n_layers convs `conv1,conv2`;
    /// upsample = 1 conv `conv`).
    up_blocks: Vec<Vec<ConvW>>,
    conv_out: ConvW,
}

impl DecoderW {
    fn new(cfg: &LtxVaeConfig) -> Self {
        let mut up_blocks = Vec::with_capacity(cfg.up_blocks.len());
        for (i, blk) in cfg.up_blocks.iter().enumerate() {
            let mut convs = Vec::new();
            match blk {
                UpBlockKind::ResX { n_layers, .. } => {
                    for j in 0..*n_layers {
                        let p = format!("decoder.up_blocks.{i}.res_blocks.{j}");
                        convs.push(conv_w(&format!("{p}.conv1.conv")));
                        convs.push(conv_w(&format!("{p}.conv2.conv")));
                    }
                }
                UpBlockKind::Upsample { .. } => {
                    convs.push(conv_w(&format!("decoder.up_blocks.{i}.conv.conv")));
                }
            }
            up_blocks.push(convs);
        }
        Self {
            conv_in: conv_w("decoder.conv_in.conv"),
            up_blocks,
            conv_out: conv_w("decoder.conv_out.conv"),
        }
    }
}

#[derive(Clone, Copy)]
struct ConvH {
    weight: WeightHandle,
    bias: WeightHandle,
}

struct DecoderH {
    conv_in: ConvH,
    up_blocks: Vec<Vec<ConvH>>,
    conv_out: ConvH,
}

fn reg_conv<S: WeightSource>(res: &WeightResidency<S>, w: &ConvW) -> Result<ConvH, LoadError> {
    Ok(ConvH {
        weight: register_passthrough(res, &w.weight)?,
        bias: register_passthrough(res, &w.bias)?,
    })
}

impl DecoderH {
    fn register<S: WeightSource>(
        res: &WeightResidency<S>,
        w: &DecoderW,
    ) -> Result<Self, LoadError> {
        let mut up_blocks = Vec::with_capacity(w.up_blocks.len());
        for convs in &w.up_blocks {
            let mut hs = Vec::with_capacity(convs.len());
            for c in convs {
                hs.push(reg_conv(res, c)?);
            }
            up_blocks.push(hs);
        }
        Ok(Self {
            conv_in: reg_conv(res, &w.conv_in)?,
            up_blocks,
            conv_out: reg_conv(res, &w.conv_out)?,
        })
    }
}

/// Post-acquire conv buffers (Copy; bound into the scope).
#[derive(Clone, Copy)]
struct ConvBufs {
    weight: BufRef,
    bias: BufRef,
}

/// All decoder bufs after one `acquire` (pins held in the parallel `Vec<GpuView>`).
struct DecoderBufs {
    conv_in: ConvBufs,
    up_blocks: Vec<Vec<ConvBufs>>,
    conv_out: ConvBufs,
}

// ============================================================================
// Pipelines
// ============================================================================

pub struct LtxVaePipelines {
    act: ActDtype,
    conv3d: Conv3dF32,
    conv3d_pipeline: thinfer_core::backend::WgpuPipeline,
    pixel_norm: thinfer_core::backend::WgpuPipeline,
    depth_to_space: thinfer_core::backend::WgpuPipeline,
    silu: thinfer_core::backend::WgpuPipeline,
    add: thinfer_core::backend::WgpuPipeline,
    concat_time: thinfer_core::backend::WgpuPipeline,
}

impl LtxVaePipelines {
    /// Activation storage size in bytes (4 = f32, 2 = f16). Drives every
    /// workspace alloc and the tile-peak estimate.
    fn act_size(&self) -> u32 {
        match self.act {
            ActDtype::F32 => 4,
            ActDtype::F16 => 2,
            other => unreachable!("ltx vae acts are f32 or f16, got {other:?}"),
        }
    }

    /// Compile with the act dtype from `LTX_VAE_ACTS` (default `f16`; `f32` =
    /// the bit-exact parity reference). F16 acts halve every activation, so far
    /// more area-frames fit per VRAM tile (fewer tiles, less overlap recompute)
    /// AND the conv stack runs ~2x faster. The whole-tensor single-tile path is
    /// still bit-exact at f32 (`vae_parity`); f16 is validated to a tight band.
    pub async fn compile(backend: &WgpuBackend) -> Result<Self, WgpuError> {
        let act = match std::env::var("LTX_VAE_ACTS").ok().as_deref() {
            Some("f32") => ActDtype::F32,
            _ => ActDtype::F16,
        };
        Self::compile_with(backend, act).await
    }

    pub async fn compile_with(backend: &WgpuBackend, act: ActDtype) -> Result<Self, WgpuError> {
        use thinfer_core::backend::Backend;
        // bf16 safetensors weights; acts f32 (parity) or f16 (production perf).
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
                    "ltx_vae_conv3d",
                    &conv3d.wgsl(cfg),
                    "main",
                    <Conv3dF32 as Conv3dOp>::layout(),
                )
                .await?,
            conv3d,
            pixel_norm: backend
                .create_pipeline(
                    "ltx_vae_pixel_norm3d",
                    &<PixelNorm3dF32 as PixelNorm3dOp>::wgsl(cfg),
                    "main",
                    <PixelNorm3dF32 as PixelNorm3dOp>::layout(),
                )
                .await?,
            depth_to_space: backend
                .create_pipeline(
                    "ltx_vae_depth_to_space3d",
                    &<DepthToSpace3dF32 as DepthToSpace3dOp>::wgsl(cfg),
                    "main",
                    <DepthToSpace3dF32 as DepthToSpace3dOp>::layout(),
                )
                .await?,
            silu: backend
                .create_pipeline(
                    "ltx_vae_silu",
                    SiluF32::wgsl(cfg),
                    "main",
                    SiluF32::layout(),
                )
                .await?,
            add: backend
                .create_pipeline("ltx_vae_add", AddF32::wgsl(cfg), "main", AddF32::layout())
                .await?,
            concat_time: backend
                .create_pipeline(
                    "ltx_vae_concat_time",
                    &<ConcatTimeF32 as ConcatTimeOp>::wgsl(cfg),
                    "main",
                    <ConcatTimeF32 as ConcatTimeOp>::layout(),
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
    /// `T*H*W` (channel stride and pixel-norm position count, B=1).
    fn thw(&self) -> u32 {
        self.t * self.h * self.w
    }
}

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

#[allow(clippy::too_many_arguments)]
fn concat_time_uniform_bytes(
    c: u32,
    h: u32,
    w: u32,
    a_t: u32,
    b_t: u32,
    a_start: u32,
    a_count: u32,
    b_start: u32,
    b_count: u32,
) -> [u8; 48] {
    let fields: [u32; 12] = [
        1, c, h, w, a_t, b_t, a_start, a_count, b_start, b_count, 0, 0,
    ];
    let mut bytes = [0u8; 48];
    for (i, v) in fields.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn pixel_norm_uniform_bytes(n_pos: u32, channels: u32, stride: u32) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&n_pos.to_le_bytes());
    bytes[4..8].copy_from_slice(&channels.to_le_bytes());
    bytes[8..12].copy_from_slice(&stride.to_le_bytes());
    bytes[12..16].copy_from_slice(&PIXEL_NORM_EPS.to_bits().to_le_bytes());
    bytes
}

#[allow(clippy::too_many_arguments)]
fn depth_to_space_uniform_bytes(
    cin: u32,
    t_in: u32,
    h_in: u32,
    w_in: u32,
    p: [u32; 3],
    t_drop: u32,
    cout: u32,
    t_out: u32,
    h_out: u32,
    w_out: u32,
    base_cout: u32,
) -> [u8; 64] {
    let fields: [u32; 16] = [
        cin, t_in, h_in, w_in, p[0], p[1], p[2], t_drop, cout, t_out, h_out, w_out, base_cout, 0,
        0, 0,
    ];
    let mut bytes = [0u8; 64];
    for (i, v) in fields.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    bytes
}

// ============================================================================
// Op wrappers (all run inside one BatchScope)
// ============================================================================

/// Temporal padding for a k3 conv (both pad time by 2 -> conv restores T).
/// `Symmetric` (`causal=False`, decoder) pads one frame each side; `Causal`
/// (`causal=True`, encoder) front-pads two frames (repeat frame 0), no rear pad.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TimePad {
    Symmetric,
    Causal,
}

/// Symmetric replicate temporal pad by 1 (`causal=False`): `[C,T,H,W]` ->
/// `[C,T+2,H,W]` = `[x0, x0..x_{T-1}, x_{T-1}]`, via two `concat_time` passes.
fn replicate_pad_time<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &LtxVaePipelines,
    x: thinfer_core::workspace::BatchBuf<'w>,
    s: Shape,
) -> Result<thinfer_core::workspace::BatchBuf<'w>, WgpuError> {
    // tmp [C,T+1,H,W] = [x0, x0..x_{T-1}].
    let tmp_shape = Shape { t: s.t + 1, ..s };
    let tmp = scope.alloc(tmp_shape.bytes(pl.act_size()))?;
    let u1 = scope.write_uniform(&concat_time_uniform_bytes(
        s.c, s.h, s.w, s.t, s.t, 0, 1, 0, s.t,
    ))?;
    scope.concat_time::<ConcatTimeF32>(&pl.concat_time, x, x, u1, tmp, tmp_shape.elems())?;
    // padded [C,T+2,H,W] = [tmp, x_{T-1}].
    let pad_shape = Shape { t: s.t + 2, ..s };
    let padded = scope.alloc(pad_shape.bytes(pl.act_size()))?;
    let u2 = scope.write_uniform(&concat_time_uniform_bytes(
        s.c,
        s.h,
        s.w,
        s.t + 1,
        s.t,
        0,
        s.t + 1,
        s.t - 1,
        1,
    ))?;
    scope.concat_time::<ConcatTimeF32>(&pl.concat_time, tmp, x, u2, padded, pad_shape.elems())?;
    Ok(padded)
}

/// Causal temporal pad by 2 (`causal=True`): `[C,T,H,W]` -> `[C,T+2,H,W]` =
/// `[x0, x0, x0..x_{T-1}]` (repeat frame 0 twice up front, no rear pad), via two
/// `concat_time` passes. This is the encoder's `CausalConv3d` front-pad; the conv
/// (pad_t=0, stride 1) then restores T.
fn causal_pad_time<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &LtxVaePipelines,
    x: thinfer_core::workspace::BatchBuf<'w>,
    s: Shape,
) -> Result<thinfer_core::workspace::BatchBuf<'w>, WgpuError> {
    // tmp [C,2,H,W] = [x0, x0].
    let tmp_shape = Shape { t: 2, ..s };
    let tmp = scope.alloc(tmp_shape.bytes(pl.act_size()))?;
    let u1 = scope.write_uniform(&concat_time_uniform_bytes(
        s.c, s.h, s.w, s.t, s.t, 0, 1, 0, 1,
    ))?;
    scope.concat_time::<ConcatTimeF32>(&pl.concat_time, x, x, u1, tmp, tmp_shape.elems())?;
    // padded [C,T+2,H,W] = [x0, x0, x0..x_{T-1}].
    let pad_shape = Shape { t: s.t + 2, ..s };
    let padded = scope.alloc(pad_shape.bytes(pl.act_size()))?;
    let u2 = scope.write_uniform(&concat_time_uniform_bytes(
        s.c, s.h, s.w, 2, s.t, 0, 2, 0, s.t,
    ))?;
    scope.concat_time::<ConcatTimeF32>(&pl.concat_time, tmp, x, u2, padded, pad_shape.elems())?;
    Ok(padded)
}

/// k3 conv, stride 1, with time pre-padded by 2 (mode-dependent: `Symmetric` =
/// decoder `causal=False` replicate; `Causal` = encoder `causal=True` front-pad)
/// so the conv restores T. The conv then pads H/W (+1 each side) with `pad_mode`
/// (0=zeros for the distilled VAE + every encoder conv, 2=reflect for the LTX-2
/// rapid decoder). `pad_mode` only affects the spatial border (conv pad_t=0).
/// `[in.c, T, H, W]` -> `[cout, T, H, W]`.
fn conv3d_k3<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &LtxVaePipelines,
    x: thinfer_core::workspace::BatchBuf<'w>,
    s: Shape,
    w: &ConvBufs,
    cout: u32,
    pad_mode: u32,
    time_pad: TimePad,
) -> Result<(thinfer_core::workspace::BatchBuf<'w>, Shape), WgpuError> {
    let padded = match time_pad {
        TimePad::Symmetric => replicate_pad_time(scope, pl, x, s)?,
        TimePad::Causal => causal_pad_time(scope, pl, x, s)?,
    };
    let out_shape = Shape { c: cout, ..s };
    let out = scope.alloc(out_shape.bytes(pl.act_size()))?;
    let u = scope.write_uniform(&conv3d_uniform_bytes(
        s.c,
        cout,
        s.t + 2,
        s.h,
        s.w,
        s.t,
        s.h,
        s.w,
        (3, 3, 3),
        (0, 1, 1),
        pad_mode,
    ))?;
    let wb = scope.import_copy(w.weight);
    let bb = scope.import_copy(w.bias);
    scope.conv3d(
        &pl.conv3d_pipeline,
        &pl.conv3d,
        padded,
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

fn pixel_norm<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &LtxVaePipelines,
    x: thinfer_core::workspace::BatchBuf<'w>,
    s: Shape,
) -> Result<thinfer_core::workspace::BatchBuf<'w>, WgpuError> {
    let out = scope.alloc(s.bytes(pl.act_size()))?;
    let u = scope.write_uniform(&pixel_norm_uniform_bytes(s.thw(), s.c, s.thw()))?;
    scope.pixel_norm3d::<PixelNorm3dF32>(&pl.pixel_norm, x, u, out, s.thw())?;
    Ok(out)
}

fn silu<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &LtxVaePipelines,
    x: thinfer_core::workspace::BatchBuf<'w>,
    s: Shape,
) -> Result<thinfer_core::workspace::BatchBuf<'w>, WgpuError> {
    let out = scope.alloc(s.bytes(pl.act_size()))?;
    scope.dispatch_op::<SiluF32>(&pl.silu, &[x], out)?;
    Ok(out)
}

fn add<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &LtxVaePipelines,
    a: thinfer_core::workspace::BatchBuf<'w>,
    b: thinfer_core::workspace::BatchBuf<'w>,
    s: Shape,
) -> Result<thinfer_core::workspace::BatchBuf<'w>, WgpuError> {
    let out = scope.alloc(s.bytes(pl.act_size()))?;
    scope.dispatch_op::<AddF32>(&pl.add, &[a, b], out)?;
    Ok(out)
}

/// One channel-preserving `res_x` resnet layer (`ResnetBlock3D`, in==out):
/// `x + conv2(silu(norm2(conv1(silu(norm1(x))))))`. Shared by the decoder
/// `up_blocks` and the encoder `down_blocks` (they are the same module); the
/// only difference is the conv temporal padding (`Symmetric` decode / `Causal`
/// encode). `cout == s.c`.
#[allow(clippy::too_many_arguments)]
fn resnet_layer<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &LtxVaePipelines,
    x: thinfer_core::workspace::BatchBuf<'w>,
    s: Shape,
    conv1: &ConvBufs,
    conv2: &ConvBufs,
    pad_mode: u32,
    time_pad: TimePad,
) -> Result<thinfer_core::workspace::BatchBuf<'w>, WgpuError> {
    let cout = s.c;
    let n1 = pixel_norm(scope, pl, x, s)?;
    let a1 = silu(scope, pl, n1, s)?;
    let (c1, _) = conv3d_k3(scope, pl, a1, s, conv1, cout, pad_mode, time_pad)?;
    let n2 = pixel_norm(scope, pl, c1, s)?;
    let a2 = silu(scope, pl, n2, s)?;
    let (c2, _) = conv3d_k3(scope, pl, a2, s, conv2, cout, pad_mode, time_pad)?;
    add(scope, pl, x, c2, s)
}

/// Depth-to-space shuffle of `x` into `out_c` channels: a `(p1,p2,p3)` pixel
/// shuffle (`b (c p1 p2 p3) t h w -> b c (t p1)(h p2)(w p3)`, + leading-frame
/// drop when p1==2). `base_c` is the pre-tile channel count (`s.c / prod(p)`):
/// pass `base_c == out_c` for the plain shuffle (`residual=False` conv path), or
/// `base_c = s.c / prod` with `out_c = base_c * repeat` to tile the shuffle up
/// for the residual up-shortcut (`x_in` in `DepthToSpaceUpsample(residual=True)`).
#[allow(clippy::too_many_arguments)]
fn depth_to_space<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &LtxVaePipelines,
    x: thinfer_core::workspace::BatchBuf<'w>,
    s: Shape,
    p: [u32; 3],
    t_drop: u32,
    out_c: u32,
    base_c: u32,
) -> Result<(thinfer_core::workspace::BatchBuf<'w>, Shape), WgpuError> {
    let out_shape = Shape {
        c: out_c,
        t: s.t * p[0] - t_drop,
        h: s.h * p[1],
        w: s.w * p[2],
    };
    let out = scope.alloc(out_shape.bytes(pl.act_size()))?;
    let u = scope.write_uniform(&depth_to_space_uniform_bytes(
        s.c,
        s.t,
        s.h,
        s.w,
        p,
        t_drop,
        out_c,
        out_shape.t,
        out_shape.h,
        out_shape.w,
        base_c,
    ))?;
    scope.depth_to_space3d::<DepthToSpace3dF32>(
        &pl.depth_to_space,
        x,
        u,
        out,
        out_shape.elems(),
    )?;
    Ok((out, out_shape))
}

// ============================================================================
// Decoder
// ============================================================================

pub struct LtxVaeDecoder {
    pub pipelines: LtxVaePipelines,
    cfg: LtxVaeConfig,
    handles: DecoderH,
    /// Baked per-channel latent un-normalization (`per_channel_statistics`).
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
}

#[derive(Debug)]
pub enum LtxVaeError<SE: core::fmt::Debug> {
    Wgpu(WgpuError),
    Load(LoadError),
    Residency(ResidencyError<SE, WgpuError>),
}
impl<SE: core::fmt::Debug> From<WgpuError> for LtxVaeError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}
impl<SE: core::fmt::Debug> From<LoadError> for LtxVaeError<SE> {
    fn from(e: LoadError) -> Self {
        Self::Load(e)
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for LtxVaeError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

/// Load the per-channel latent normalization stats (`per_channel_statistics.
/// {mean,std}-of-means`, bf16 `[128]`) from the VAE source. The DiT denoises in
/// the normalized latent space; the upsampler operates UN-normalized, so the
/// two-stage orchestration needs these to un/re-normalize around the upscale
/// (`upsample_video`), and `LtxVaeDecoder::new` takes them for the decode
/// un-normalize. Returns `(mean, std)`, each `[128]` f32.
pub async fn load_latent_stats<S: WeightSource>(
    residency: &WeightResidency<S>,
    backend: &WgpuBackend,
) -> Result<(Vec<f32>, Vec<f32>), LtxVaeError<S::Error>> {
    use thinfer_core::backend::Backend;
    let read = |id: &str| -> Result<_, LoadError> {
        register_passthrough(residency, &WeightId(id.into()))
    };
    let mean_h = read("per_channel_statistics.mean-of-means")?;
    let std_h = read("per_channel_statistics.std-of-means")?;
    let mut out = Vec::new();
    for h in [mean_h, std_h] {
        let view = residency.acquire(h, backend).await?;
        let bytes = backend
            .read_buffer(view.buf().id, 0, (LATENT_CHANNELS * 2) as u64)
            .await?;
        out.push(crate::common::seq::act_readback_to_f32(
            ActDtype::Bf16,
            &bytes,
            LATENT_CHANNELS,
        ));
    }
    let std = out.pop().expect("std");
    let mean = out.pop().expect("mean");
    Ok((mean, std))
}

/// Per-stage taps for parity bisection (NCTHW f32, row-major). Each `Some` sink
/// is filled with that stage's output.
#[derive(Default)]
pub struct LtxVaeTaps<'a> {
    pub conv_in: Option<&'a mut Vec<f32>>,
    /// One sink per up_block output (`len == up_blocks.len()` once filled).
    pub up_blocks: Option<&'a mut Vec<Vec<f32>>>,
    /// `conv_out` output, pre-unpatchify `[48, F, H, W]`.
    pub conv_out: Option<&'a mut Vec<f32>>,
}

impl LtxVaeDecoder {
    /// Register the decoder weights against `residency` and capture the baked
    /// per-channel latent stats.
    pub fn new<S: WeightSource>(
        pipelines: LtxVaePipelines,
        residency: &WeightResidency<S>,
        latents_mean: Vec<f32>,
        latents_std: Vec<f32>,
    ) -> Result<Self, LoadError> {
        Self::new_with_config(
            pipelines,
            residency,
            latents_mean,
            latents_std,
            LtxVaeConfig::distilled(),
        )
    }

    /// [`Self::new`] with an explicit decoder config (ltx2-rapid uses
    /// [`LtxVaeConfig::ltx2_rapid`]; the shipped 22B path uses `distilled`).
    pub fn new_with_config<S: WeightSource>(
        pipelines: LtxVaePipelines,
        residency: &WeightResidency<S>,
        latents_mean: Vec<f32>,
        latents_std: Vec<f32>,
        cfg: LtxVaeConfig,
    ) -> Result<Self, LoadError> {
        let weights = DecoderW::new(&cfg);
        let handles = DecoderH::register(residency, &weights)?;
        assert_eq!(latents_mean.len(), LATENT_CHANNELS);
        assert_eq!(latents_std.len(), LATENT_CHANNELS);
        Ok(Self {
            pipelines,
            cfg,
            handles,
            latents_mean,
            latents_std,
        })
    }

    /// Decode `latents` (CTHW row-major, `128 * f * h * w` f32, NORMALIZED) into
    /// a host video `[3, 8*(f-1)+1, 32*h, 32*w]` f32 (raw decoder output, no
    /// clamp; the caller maps to `[0,1]` via `(x+1)/2` + clamp).
    ///
    /// The decode runs in OVERLAPPING SPATIAL + TEMPORAL TILES sized from the
    /// residency VRAM budget (the whole-tensor single-submit otherwise OOMs
    /// above tiny dims -- its activation peak is unbounded by the weight
    /// budget). Each tile decodes through the full conv stack; outputs are
    /// feather/trapezoid-blended over the overlap so seams (each tile's interior
    /// pad error) vanish. This is the engine-side analog of upstream LTX
    /// `SpatialTilingConfig` + `TemporalTilingConfig` (overlap tiles + blend) --
    /// the in-place chunk path is CUDA-specific. A single tile (latent dims fit
    /// the budget) is bit-identical to the untiled decode, so the parity gate
    /// (`decode_with_taps`, tiny dims) is unaffected.
    ///
    /// Two knobs, both budget-sized then HALVED on OOM (adaptive -> converges
    /// without measured calibration): the spatial tile side (latent cells) and
    /// the temporal tile depth (latent frames). The spatial tile shrinks first;
    /// once it hits the floor the temporal tile shrinks (and the spatial tile is
    /// re-grown for the now-shallower per-tile frame count). With both knobs the
    /// peak is tileable below ANY budget down to the resident conv-weight floor
    /// (~1.4GB; a budget under that hard-fails -- weight streaming is not done).
    pub async fn decode<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        latents: &[f32],
        f: usize,
        h: usize,
        w: usize,
    ) -> Result<Vec<f32>, LtxVaeError<S::Error>> {
        assert_eq!(latents.len(), LATENT_CHANNELS * f * h * w, "latent size");
        let z = self.un_normalize(latents, f, h, w);

        // Acquire every conv weight once; pins held across all tiles.
        let mut pins: Vec<GpuView> = Vec::new();
        let bufs = self.acquire(residency, backend, &mut pins).await?;

        // Budget -> initial tiles. The budget (u64::MAX = unbounded -> a 4 GiB
        // reference) is the total VRAM ceiling; carve out the resident weights
        // (exact, from the conv geometry) + staging, the rest is the per-tile
        // workspace. `set_transient_reserve` holds the non-weight VRAM so the
        // arbiter caps weight residency at the footprint and the workspace +
        // staging cannot push the true peak past the budget (same mechanism the
        // Wan VAE / DiT step loop use). Respected at any budget, including small.
        let budget = residency.arbiter().budget_bytes();
        let eff_budget = if budget == u64::MAX {
            DEFAULT_REF_BUDGET
        } else {
            budget
        };
        let weight_footprint = self.cfg.weight_footprint_bytes();
        let workspace_budget = eff_budget.saturating_sub(weight_footprint + VAE_STAGING_RESERVE);
        if budget != u64::MAX {
            residency.set_transient_reserve(eff_budget.saturating_sub(weight_footprint));
            // Hard ceiling: an over-budget tile alloc fails AT the budget (the
            // tiler shrinks below) instead of overshooting into a device OOM.
            workspace.set_strict_budget(true);
        }
        let act_size = self.pipelines.act_size();
        // Seed the tile pair below the workspace budget by a safety margin: the
        // peak estimate is close but doesn't model alloc fragmentation / staging
        // exactly, so seeding AT the budget edge OOMs on the first tile and forces
        // a costly reshrink. The margin lands the common case on the first try.
        let mut seed_budget = (workspace_budget as f64 * SEED_SAFETY) as u64;
        let (mut tf, mut tile) = initial_tiles(seed_budget, f, act_size);

        loop {
            tracing::info!(
                target: thinfer_core::trace::DIAG,
                acts = ?self.pipelines.act, f, h, w,
                seed_budget_mb = seed_budget / (1024 * 1024),
                tf, tile, "ltx vae decode attempt",
            );
            match self
                .decode_tiled(backend, workspace, &bufs, &z, f, h, w, tf, tile)
                .await
            {
                Ok(v) => return Ok(v),
                Err(WgpuError::Allocate { .. } | WgpuError::BudgetExceeded { .. })
                    if tile > TILE_MIN || tf as u32 > TEMPORAL_TILE_MIN =>
                {
                    // Per-tile workspace didn't fit. OOM hits at alloc (pre-submit),
                    // so the failed tiles left nothing live -- drain the pool and
                    // re-seed from a smaller budget. Re-seeding (vs halving one
                    // axis) keeps the spatial/temporal split BALANCED: it shrinks
                    // whichever axis the estimate over-allocated instead of
                    // collapsing the spatial tile (which would explode the tile
                    // count). Step the budget down until the plan strictly shrinks.
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
                        "ltx vae decode OOM; re-seeding smaller",
                    );
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    /// Host per-channel latent un-normalize (`z * std + mean`), CTHW.
    fn un_normalize(&self, latents: &[f32], f: usize, h: usize, w: usize) -> Vec<f32> {
        let mut z = vec![0.0f32; latents.len()];
        let thw = f * h * w;
        for c in 0..LATENT_CHANNELS {
            let (std, mean) = (self.latents_std[c], self.latents_mean[c]);
            for i in 0..thw {
                z[c * thw + i] = latents[c * thw + i] * std + mean;
            }
        }
        z
    }

    /// Decode `z` (un-normalized CTHW) in overlapping spatial (`tile` latent
    /// cells) AND temporal (`tf` latent frames) tiles, with budget-independent
    /// overlaps. Spatial seams blend with feather ramps; temporal seams with
    /// trapezoidal masks (mirroring upstream `map_temporal_slice`). The two
    /// blend windows are separable (every temporal tile pairs with every spatial
    /// tile, a full product grid), so the per-output weight = temporal-weight-sum
    /// x spatial-weight-sum -- accumulated as two 1-D sums and divided as a
    /// product. A single tile (`h,w <= tile` and `f <= tf`) reduces to one full
    /// decode with unit weights -> bit-identical to the untiled path.
    #[allow(clippy::too_many_arguments)]
    async fn decode_tiled(
        &self,
        backend: &WgpuBackend,
        workspace: &Workspace<WgpuBackend>,
        bufs: &DecoderBufs,
        z: &[f32],
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
        let plane = f_px * oh * ow; // per-channel stride of the output.

        let ttiles = plan_temporal_tiles(f, tf, t_overlap, TEMPORAL_SCALE);
        let tiles_h = plan_tiles(h as u32, tile, overlap);
        let tiles_w = plan_tiles(w as u32, tile, overlap);
        let single = ttiles.len() == 1 && tiles_h.len() == 1 && tiles_w.len() == 1;
        tracing::info!(
            target: thinfer_core::trace::DIAG,
            n_temporal = ttiles.len(), n_h = tiles_h.len(), n_w = tiles_w.len(),
            total_decodes = ttiles.len() * tiles_h.len() * tiles_w.len(),
            t_overlap, overlap, "ltx vae tile plan",
        );

        let mut video = vec![0.0f32; OUT_CHANNELS * plane];
        // Separable weight sums: spatial per output pixel `[oh*ow]`, temporal per
        // output frame `[f_px]`. Each accumulated once (independent of the other
        // axis), then the true per-output weight is their product.
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
                    let z_sub = gather_subtile(
                        z,
                        LATENT_CHANNELS,
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
                    let (packed, cos) = self
                        .run_graph(backend, workspace, bufs, &z_sub, tflen, he, we, None)
                        .await?;
                    let pix = unpatchify_4x4(&packed, cos); // [3, f_px_tile, he*32, we*32]
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
                    // Free this tile's idle workspace before the next grows it.
                    if !single {
                        workspace.drain_pool();
                    }
                }
            }
        }

        // Normalize by the separable weight product (unit everywhere for a
        // single tile). A tiny floor keeps the divisor positive at ramp zeros.
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

    /// Run the decoder conv stack on one (un-normalized) latent tile `z`
    /// `[128, f, h, w]`, returning the packed `conv_out` `[48, F, H, W]` host
    /// f32 + its shape (the caller unpatchifies). Fills `taps` if given (parity
    /// bisection; only used on the single full-tile path).
    #[allow(clippy::too_many_arguments)]
    async fn run_graph(
        &self,
        backend: &WgpuBackend,
        workspace: &Workspace<WgpuBackend>,
        bufs: &DecoderBufs,
        z: &[f32],
        f: usize,
        h: usize,
        w: usize,
        mut taps: Option<&mut LtxVaeTaps<'_>>,
    ) -> Result<(Vec<f32>, Shape), WgpuError> {
        use thinfer_core::backend::Backend;

        let act = self.pipelines.act;
        let act_size = self.pipelines.act_size();
        let in_bytes = crate::common::seq::act_upload_bytes(act, z);
        let in_buf = workspace.alloc(in_bytes.len() as u64)?;
        backend.write_buffer(in_buf.id(), 0, &in_bytes)?;

        // Per-stage tap persistence (workspace buffers that outlive the submit).
        let mut p_conv_in: Option<(thinfer_core::workspace::WsBuf<WgpuBackend>, Shape)> = None;
        let mut p_ups: Vec<(thinfer_core::workspace::WsBuf<WgpuBackend>, Shape)> = Vec::new();
        let conv_out_shape;
        let conv_out_persist;
        {
            let scope = workspace.batch();
            let pl = &self.pipelines;
            let pad_mode = self.cfg.spatial_pad_mode;
            let mut x = scope.import_copy(in_buf.as_buf_ref());
            let mut s = Shape {
                c: LATENT_CHANNELS as u32,
                t: f as u32,
                h: h as u32,
                w: w as u32,
            };

            // conv_in (k3, 128 -> 1024).
            let (cx, cs) = conv3d_k3(
                &scope,
                pl,
                x,
                s,
                &bufs.conv_in,
                self.cfg.mid_channels as u32,
                pad_mode,
                TimePad::Symmetric,
            )?;
            x = cx;
            s = cs;
            if taps.as_ref().is_some_and(|t| t.conv_in.is_some()) {
                p_conv_in = Some((persist(&scope, workspace, x, s, act_size)?, s));
            }

            // up_blocks.
            let want_ups = taps.as_ref().is_some_and(|t| t.up_blocks.is_some());
            for (i, blk) in self.cfg.up_blocks.iter().enumerate() {
                match blk {
                    UpBlockKind::ResX { n_layers, channels } => {
                        debug_assert_eq!(s.c, *channels as u32, "res_x preserves channels");
                        for j in 0..*n_layers {
                            let conv1 = &bufs.up_blocks[i][2 * j];
                            let conv2 = &bufs.up_blocks[i][2 * j + 1];
                            x = resnet_layer(
                                &scope,
                                pl,
                                x,
                                s,
                                conv1,
                                conv2,
                                pad_mode,
                                TimePad::Symmetric,
                            )?;
                        }
                    }
                    UpBlockKind::Upsample {
                        conv_out_c,
                        out_c,
                        p,
                        t_drop,
                        residual,
                    } => {
                        let conv = &bufs.up_blocks[i][0];
                        let pp = [p[0] as u32, p[1] as u32, p[2] as u32];
                        let td = *t_drop as u32;
                        // Main path: k3 conv `in -> conv_out_c` then shuffle to out_c.
                        let (cx, cs) = conv3d_k3(
                            &scope,
                            pl,
                            x,
                            s,
                            conv,
                            *conv_out_c as u32,
                            pad_mode,
                            TimePad::Symmetric,
                        )?;
                        let (dx, ds) = depth_to_space(
                            &scope,
                            pl,
                            cx,
                            cs,
                            pp,
                            td,
                            *out_c as u32,
                            *out_c as u32,
                        )?;
                        // Residual up-shortcut: shuffle the block INPUT into
                        // `s.c / prod` channels and tile it up to out_c, then add.
                        x = if *residual {
                            let prod = pp[0] * pp[1] * pp[2];
                            let base_c = s.c / prod;
                            let (rx, _) =
                                depth_to_space(&scope, pl, x, s, pp, td, *out_c as u32, base_c)?;
                            add(&scope, pl, dx, rx, ds)?
                        } else {
                            dx
                        };
                        s = ds;
                    }
                }
                if want_ups {
                    p_ups.push((persist(&scope, workspace, x, s, act_size)?, s));
                }
            }

            // norm_out -> silu -> conv_out (k3, 1024-stage-out -> 48).
            let n = pixel_norm(&scope, pl, x, s)?;
            let a = silu(&scope, pl, n, s)?;
            let (co, cos) = conv3d_k3(
                &scope,
                pl,
                a,
                s,
                &bufs.conv_out,
                CONV_OUT_CHANNELS as u32,
                pad_mode,
                TimePad::Symmetric,
            )?;
            conv_out_shape = cos;
            conv_out_persist = persist(&scope, workspace, co, cos, act_size)?;

            scope.submit_void().await?;
        }

        // Tap readback.
        if let Some(t) = taps.as_mut() {
            if let (Some((ws, sh)), Some(sink)) = (&p_conv_in, t.conv_in.as_deref_mut()) {
                *sink = read_acts(backend, &ws.as_buf_ref(), sh.elems() as usize, act).await?;
            }
            if let Some(sink) = t.up_blocks.as_deref_mut() {
                sink.clear();
                for (ws, sh) in &p_ups {
                    sink.push(
                        read_acts(backend, &ws.as_buf_ref(), sh.elems() as usize, act).await?,
                    );
                }
            }
            if let Some(sink) = t.conv_out.as_deref_mut() {
                *sink = read_acts(
                    backend,
                    &conv_out_persist.as_buf_ref(),
                    conv_out_shape.elems() as usize,
                    act,
                )
                .await?;
            }
        }

        let packed = read_acts(
            backend,
            &conv_out_persist.as_buf_ref(),
            conv_out_shape.elems() as usize,
            act,
        )
        .await?;
        Ok((packed, conv_out_shape))
    }

    /// Single-tile decode with parity taps (no tiling). The parity gate runs at
    /// tiny dims, so this stays the exact whole-tensor path; production decode
    /// (`decode`) tiles for VRAM.
    #[allow(clippy::too_many_arguments)]
    pub async fn decode_with_taps<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        latents: &[f32],
        f: usize,
        h: usize,
        w: usize,
        taps: Option<&mut LtxVaeTaps<'_>>,
    ) -> Result<Vec<f32>, LtxVaeError<S::Error>> {
        assert_eq!(latents.len(), LATENT_CHANNELS * f * h * w, "latent size");
        let z = self.un_normalize(latents, f, h, w);
        let mut pins: Vec<GpuView> = Vec::new();
        let bufs = self.acquire(residency, backend, &mut pins).await?;
        let (packed, cos) = self
            .run_graph(backend, workspace, &bufs, &z, f, h, w, taps)
            .await?;
        Ok(unpatchify_4x4(&packed, cos))
    }

    async fn acquire<'r, S: WeightSource>(
        &self,
        residency: &'r WeightResidency<S>,
        backend: &WgpuBackend,
        pins: &mut Vec<GpuView<'r>>,
    ) -> Result<DecoderBufs, ResidencyError<S::Error, WgpuError>> {
        let conv_in = acquire_conv(residency, backend, self.handles.conv_in, pins).await?;
        let mut up_blocks = Vec::with_capacity(self.handles.up_blocks.len());
        for convs in &self.handles.up_blocks {
            let mut bs = Vec::with_capacity(convs.len());
            for h in convs {
                bs.push(acquire_conv(residency, backend, *h, pins).await?);
            }
            up_blocks.push(bs);
        }
        let conv_out = acquire_conv(residency, backend, self.handles.conv_out, pins).await?;
        Ok(DecoderBufs {
            conv_in,
            up_blocks,
            conv_out,
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

/// Copy a scope-local stage activation into a workspace buffer that outlives the
/// submit, for a post-submit readback.
fn persist<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    workspace: &Workspace<WgpuBackend>,
    buf: thinfer_core::workspace::BatchBuf<'w>,
    s: Shape,
    act_size: u32,
) -> Result<thinfer_core::workspace::WsBuf<WgpuBackend>, WgpuError> {
    let ws = workspace.alloc(s.bytes(act_size))?;
    let dst = scope.import_copy(ws.as_buf_ref());
    scope.copy_buffer_to_buffer(buf, 0, dst, 0, s.bytes(act_size))?;
    Ok(ws)
}

/// Read `n` activations from `buf` (stored in `act` dtype) back as f32.
async fn read_acts(
    backend: &WgpuBackend,
    buf: &BufRef,
    n: usize,
    act: ActDtype,
) -> Result<Vec<f32>, WgpuError> {
    use thinfer_core::backend::Backend;
    let act_size = match act {
        ActDtype::F32 => 4,
        ActDtype::F16 => 2,
        other => unreachable!("ltx vae acts are f32 or f16, got {other:?}"),
    };
    let bytes = backend
        .read_buffer(buf.id, buf.offset, (n * act_size) as u64)
        .await?;
    Ok(crate::common::seq::act_readback_to_f32(act, &bytes, n))
}

/// Final spatial pixel-unshuffle (`unpatchify`, patch 4): packed `[48, T, H, W]`
/// -> `[3, T, 4H, 4W]`. einops `(c r q) f h w -> c (h q) (w r)`, q,r = 4: packed
/// channel `cp = (c*4 + r)*4 + q` lands at `(c, h*4+q, w*4+r)`.
fn unpatchify_4x4(packed: &[f32], s: Shape) -> Vec<f32> {
    let (t, h, w) = (s.t as usize, s.h as usize, s.w as usize);
    let (oh, ow) = (h * PATCH_SIZE, w * PATCH_SIZE);
    let mut out = vec![0.0f32; OUT_CHANNELS * t * oh * ow];
    let in_thw = t * h * w;
    let out_thw = t * oh * ow;
    for cp in 0..CONV_OUT_CHANNELS {
        let c = cp / (PATCH_SIZE * PATCH_SIZE);
        let rem = cp % (PATCH_SIZE * PATCH_SIZE);
        let r = rem / PATCH_SIZE;
        let q = rem % PATCH_SIZE;
        for tt in 0..t {
            for hh in 0..h {
                for ww in 0..w {
                    let src = cp * in_thw + (tt * h + hh) * w + ww;
                    let oh_i = hh * PATCH_SIZE + q;
                    let ow_i = ww * PATCH_SIZE + r;
                    let dst = c * out_thw + (tt * oh + oh_i) * ow + ow_i;
                    out[dst] = packed[src];
                }
            }
        }
    }
    out
}

// ============================================================================
// Encoder (image/video -> latent), for native I2V frame-0 conditioning.
//
// Mirror of the upstream `VideoEncoder` (video_vae.py). The conv/norm/silu run
// on the GPU (reusing the decoder pipelines); the `SpaceToDepthDownsample`
// space-to-depth + grouped-mean skip and the initial `patchify` run host-side
// between per-block scopes (encode runs ONCE per generation, so bounded host
// rearranges cost nothing and add no kernels -- same pattern as the Hunyuan VAE
// encoder). Every conv is `causal=True` (front-pad time by 2) with ZEROS spatial
// padding (the encoder default; the config omits `spatial_padding_mode`).
// ============================================================================

/// One encoder `down_block` (config order).
#[derive(Clone, Debug)]
enum EncBlock {
    /// `res_x` = `UNetMidBlock3D`: `n_layers` channel-preserving resnets.
    ResX { n_layers: usize, channels: usize },
    /// `compress_*_res` = `SpaceToDepthDownsample`: a k3 conv `in_c ->
    /// conv_out_c` then a `stride` space-to-depth (-> `out_c`), plus a grouped-
    /// mean space-to-depth skip of the block input. `stride` selects the
    /// compressed dims (space / time / all).
    Compress {
        in_c: usize,
        out_c: usize,
        /// Conv output channels (`out_c / prod(stride)`); space-to-depth then
        /// re-expands by `prod(stride)` back to `out_c`.
        conv_out_c: usize,
        /// Skip group size (`in_c * prod(stride) / out_c`): the space-to-depth of
        /// the input has `in_c * prod` channels, mean-pooled in contiguous groups.
        group_size: usize,
        stride: [usize; 3],
    },
}

#[derive(Clone, Debug)]
pub struct LtxVaeEncoderConfig {
    blocks: Vec<EncBlock>,
    /// Final feature width feeding `conv_out` (2048 for rapid).
    feature_out: usize,
}

impl LtxVaeEncoderConfig {
    /// The LTX-2 rapid encoder, verified from the on-disk `encoder_blocks`:
    /// res_x(4), compress_space_res, res_x(6), compress_time_res, res_x(6),
    /// compress_all_res, res_x(2), compress_all_res, res_x(2). Every
    /// `compress_*_res` doubles channels (multiplier 2); features run
    /// 128 -> 256 -> 512 -> 1024 -> 2048. Net 32x spatial / 8x temporal.
    pub fn ltx2_rapid() -> Self {
        // (name, multiplier-or-num_layers).
        let schedule: [(&str, usize); 9] = [
            ("res_x", 4),
            ("compress_space", 2),
            ("res_x", 6),
            ("compress_time", 2),
            ("res_x", 6),
            ("compress_all", 2),
            ("res_x", 2),
            ("compress_all", 2),
            ("res_x", 2),
        ];
        let mut feature = LATENT_CHANNELS; // conv_in output = 128.
        let mut blocks = Vec::with_capacity(schedule.len());
        for (name, param) in schedule {
            match name {
                "res_x" => blocks.push(EncBlock::ResX {
                    n_layers: param,
                    channels: feature,
                }),
                "compress_space" | "compress_time" | "compress_all" => {
                    let stride = match name {
                        "compress_space" => [1usize, 2, 2],
                        "compress_time" => [2, 1, 1],
                        _ => [2, 2, 2],
                    };
                    let prod = stride[0] * stride[1] * stride[2];
                    let in_c = feature;
                    let out_c = in_c * param; // multiplier.
                    blocks.push(EncBlock::Compress {
                        in_c,
                        out_c,
                        conv_out_c: out_c / prod,
                        group_size: in_c * prod / out_c,
                        stride,
                    });
                    feature = out_c;
                }
                other => panic!("unknown encoder block: {other}"),
            }
        }
        Self {
            blocks,
            feature_out: feature,
        }
    }
}

/// All encoder conv weight ids, in forward order (mirrors [`DecoderW`]).
struct EncoderW {
    conv_in: ConvW,
    /// `down_blocks[i]`: res_x = `2*n_layers` convs (conv1,conv2 flat); compress
    /// = 1 conv (`down_blocks.{i}.conv.conv`).
    down_blocks: Vec<Vec<ConvW>>,
    conv_out: ConvW,
}

impl EncoderW {
    fn new(cfg: &LtxVaeEncoderConfig) -> Self {
        let mut down_blocks = Vec::with_capacity(cfg.blocks.len());
        for (i, blk) in cfg.blocks.iter().enumerate() {
            let mut convs = Vec::new();
            match blk {
                EncBlock::ResX { n_layers, .. } => {
                    for j in 0..*n_layers {
                        let p = format!("encoder.down_blocks.{i}.res_blocks.{j}");
                        convs.push(conv_w(&format!("{p}.conv1.conv")));
                        convs.push(conv_w(&format!("{p}.conv2.conv")));
                    }
                }
                EncBlock::Compress { .. } => {
                    convs.push(conv_w(&format!("encoder.down_blocks.{i}.conv.conv")));
                }
            }
            down_blocks.push(convs);
        }
        Self {
            conv_in: conv_w("encoder.conv_in.conv"),
            down_blocks,
            conv_out: conv_w("encoder.conv_out.conv"),
        }
    }
}

struct EncoderH {
    conv_in: ConvH,
    down_blocks: Vec<Vec<ConvH>>,
    conv_out: ConvH,
}

impl EncoderH {
    fn register<S: WeightSource>(
        res: &WeightResidency<S>,
        w: &EncoderW,
    ) -> Result<Self, LoadError> {
        let mut down_blocks = Vec::with_capacity(w.down_blocks.len());
        for convs in &w.down_blocks {
            let mut hs = Vec::with_capacity(convs.len());
            for c in convs {
                hs.push(reg_conv(res, c)?);
            }
            down_blocks.push(hs);
        }
        Ok(Self {
            conv_in: reg_conv(res, &w.conv_in)?,
            down_blocks,
            conv_out: reg_conv(res, &w.conv_out)?,
        })
    }
}

/// Prepend a duplicate of frame 0 (`SpaceToDepthDownsample`'s temporal-stride
/// front-pad, distinct from the conv's causal pad): `[C,T,H,W] -> [C,T+1,H,W]` =
/// `[x0, x0..x_{T-1}]`.
fn dup_first_frame<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &LtxVaePipelines,
    x: thinfer_core::workspace::BatchBuf<'w>,
    s: Shape,
) -> Result<(thinfer_core::workspace::BatchBuf<'w>, Shape), WgpuError> {
    let out_shape = Shape { t: s.t + 1, ..s };
    let out = scope.alloc(out_shape.bytes(pl.act_size()))?;
    let u = scope.write_uniform(&concat_time_uniform_bytes(
        s.c, s.h, s.w, s.t, s.t, 0, 1, 0, s.t,
    ))?;
    scope.concat_time::<ConcatTimeF32>(&pl.concat_time, x, x, u, out, out_shape.elems())?;
    Ok((out, out_shape))
}

pub struct LtxVaeEncoder {
    pipelines: LtxVaePipelines,
    cfg: LtxVaeEncoderConfig,
    handles: EncoderH,
    /// Baked per-channel latent normalization (`per_channel_statistics`); encode
    /// applies `(mean - mean-of-means) / std-of-means` (inverse of the decoder
    /// un-normalize), so the output shares the DiT's normalized latent space.
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
}

impl LtxVaeEncoder {
    pub fn new<S: WeightSource>(
        pipelines: LtxVaePipelines,
        residency: &WeightResidency<S>,
        latents_mean: Vec<f32>,
        latents_std: Vec<f32>,
        cfg: LtxVaeEncoderConfig,
    ) -> Result<Self, LoadError> {
        assert_eq!(latents_mean.len(), LATENT_CHANNELS);
        assert_eq!(latents_std.len(), LATENT_CHANNELS);
        let handles = EncoderH::register(residency, &EncoderW::new(&cfg))?;
        Ok(Self {
            pipelines,
            cfg,
            handles,
            latents_mean,
            latents_std,
        })
    }

    /// Encode a NORMALIZED-`[-1,1]` video `[3, t, hp, wp]` (row-major CTHW;
    /// `t == 1 + 8*k`, `hp`/`wp` multiples of 32) into the normalized latent
    /// `[128, 1 + (t-1)/8, hp/32, wp/32]` f32 (distribution mean, per-channel
    /// normalized). Single frame (`t == 1`) is the I2V frame-0 case.
    pub async fn encode<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        frame: &[f32],
        t: usize,
        hp: usize,
        wp: usize,
    ) -> Result<Vec<f32>, LtxVaeError<S::Error>> {
        self.encode_with_taps(backend, residency, workspace, frame, t, hp, wp, None)
            .await
    }

    /// [`Self::encode`] with an optional per-`down_block` output sink (parity
    /// bisection): each `Some` sink is filled with one host `[C,T,H,W]` vector
    /// per encoder block, in forward order.
    #[allow(clippy::too_many_arguments)]
    pub async fn encode_with_taps<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        frame: &[f32],
        t: usize,
        hp: usize,
        wp: usize,
        mut block_taps: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Vec<f32>, LtxVaeError<S::Error>> {
        assert_eq!(frame.len(), OUT_CHANNELS * t * hp * wp, "frame [3,t,hp,wp]");
        assert!(
            hp.is_multiple_of(SPATIAL_SCALE) && wp.is_multiple_of(SPATIAL_SCALE),
            "hp/wp must be multiples of {SPATIAL_SCALE}"
        );
        assert!((t - 1).is_multiple_of(TEMPORAL_SCALE), "t must be 1 + 8*k");
        if let Some(taps) = block_taps.as_deref_mut() {
            taps.clear();
        }

        // Host patchify: [3, t, hp, wp] -> [48, t, hp/4, wp/4].
        let (mut host, mut s) = patchify_4x4(frame, t, hp, wp);

        for (bi, blk) in self.cfg.blocks.iter().enumerate() {
            match blk {
                EncBlock::ResX { n_layers, channels } => {
                    host = self
                        .run_res_x(backend, residency, workspace, bi, *n_layers, &mut s, &host)
                        .await?;
                    debug_assert_eq!(s.c as usize, *channels, "res_x channel width");
                }
                EncBlock::Compress {
                    in_c,
                    out_c,
                    conv_out_c,
                    group_size,
                    stride,
                } => {
                    host = self
                        .run_compress(
                            backend,
                            residency,
                            workspace,
                            bi,
                            *in_c,
                            *out_c,
                            *conv_out_c,
                            *group_size,
                            *stride,
                            &mut s,
                            &host,
                        )
                        .await?;
                }
            }
            if let Some(taps) = block_taps.as_deref_mut() {
                taps.push(host.clone());
            }
        }

        // norm_out (pixel_norm) -> silu -> conv_out (feature -> 129).
        let raw = self
            .run_conv_out(backend, residency, workspace, &mut s, &host)
            .await?;

        // Means = first 128 channels (channel-major, so contiguous); normalize.
        let plane = (s.t * s.h * s.w) as usize;
        let mut z = raw[..LATENT_CHANNELS * plane].to_vec();
        for c in 0..LATENT_CHANNELS {
            let (mean, std) = (self.latents_mean[c], self.latents_std[c]);
            for v in &mut z[c * plane..(c + 1) * plane] {
                *v = (*v - mean) / std;
            }
        }
        Ok(z)
    }

    /// A `res_x` block: `n_layers` resnets in one scope (conv_in prepended for
    /// block 0). Host in -> host out; `s` updated (conv_in changes 48 -> 128).
    #[allow(clippy::too_many_arguments)]
    async fn run_res_x<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        bi: usize,
        n_layers: usize,
        s: &mut Shape,
        host: &[f32],
    ) -> Result<Vec<f32>, LtxVaeError<S::Error>> {
        use thinfer_core::backend::Backend;
        let pl = &self.pipelines;
        let act = pl.act;
        let act_size = pl.act_size();
        let mut pins: Vec<GpuView> = Vec::new();
        let conv_in = if bi == 0 {
            Some(acquire_conv(residency, backend, self.handles.conv_in, &mut pins).await?)
        } else {
            None
        };
        let mut convs = Vec::with_capacity(2 * n_layers);
        for h in &self.handles.down_blocks[bi] {
            convs.push(acquire_conv(residency, backend, *h, &mut pins).await?);
        }

        let up = crate::common::seq::act_upload_bytes(act, host);
        let x_up = workspace.alloc(up.len() as u64)?;
        backend.write_buffer(x_up.id(), 0, &up)?;

        let (out_ws, out_s);
        {
            let scope = workspace.batch();
            let mut x = scope.import_copy(x_up.as_buf_ref());
            let mut cur = *s;
            if let Some(ci) = &conv_in {
                let (y, ns) = conv3d_k3(
                    &scope,
                    pl,
                    x,
                    cur,
                    ci,
                    LATENT_CHANNELS as u32,
                    0,
                    TimePad::Causal,
                )?;
                x = y;
                cur = ns;
            }
            for j in 0..n_layers {
                x = resnet_layer(
                    &scope,
                    pl,
                    x,
                    cur,
                    &convs[2 * j],
                    &convs[2 * j + 1],
                    0,
                    TimePad::Causal,
                )?;
            }
            out_s = cur;
            out_ws = persist(&scope, workspace, x, cur, act_size)?;
            scope.submit_void().await?;
        }
        *s = out_s;
        Ok(read_acts(backend, &out_ws.as_buf_ref(), out_s.elems() as usize, act).await?)
    }

    /// A `compress_*_res` block. GPU: optional temporal frame-0 dup, then the k3
    /// causal conv. Host: space-to-depth of both the conv output (-> `out_c`) and
    /// the (dup'd) block input, grouped-mean the latter for the skip, and add.
    /// `s` updated to the compressed shape.
    #[allow(clippy::too_many_arguments)]
    async fn run_compress<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        bi: usize,
        in_c: usize,
        out_c: usize,
        conv_out_c: usize,
        group_size: usize,
        stride: [usize; 3],
        s: &mut Shape,
        host: &[f32],
    ) -> Result<Vec<f32>, LtxVaeError<S::Error>> {
        use thinfer_core::backend::Backend;
        let pl = &self.pipelines;
        let act = pl.act;
        let act_size = pl.act_size();
        let temporal = stride[0] == 2;

        let mut pins: Vec<GpuView> = Vec::new();
        let conv = acquire_conv(
            residency,
            backend,
            self.handles.down_blocks[bi][0],
            &mut pins,
        )
        .await?;

        let up = crate::common::seq::act_upload_bytes(act, host);
        let x_up = workspace.alloc(up.len() as u64)?;
        backend.write_buffer(x_up.id(), 0, &up)?;

        // Skip source = the (dup'd) block input; conv output = the main path.
        let (skip_ws, skip_s, conv_ws, conv_s);
        {
            let scope = workspace.batch();
            let mut x = scope.import_copy(x_up.as_buf_ref());
            let mut xs = *s;
            if temporal {
                let (y, ns) = dup_first_frame(&scope, pl, x, xs)?;
                x = y;
                xs = ns;
            }
            skip_ws = persist(&scope, workspace, x, xs, act_size)?;
            skip_s = xs;
            let (cx, cs) = conv3d_k3(
                &scope,
                pl,
                x,
                xs,
                &conv,
                conv_out_c as u32,
                0,
                TimePad::Causal,
            )?;
            conv_ws = persist(&scope, workspace, cx, cs, act_size)?;
            conv_s = cs;
            scope.submit_void().await?;
        }
        let skip_host =
            read_acts(backend, &skip_ws.as_buf_ref(), skip_s.elems() as usize, act).await?;
        let conv_host =
            read_acts(backend, &conv_ws.as_buf_ref(), conv_s.elems() as usize, act).await?;
        drop(pins);

        // Main: space-to-depth conv output (conv_out_c * prod = out_c channels).
        let main = space_to_depth3d(
            &conv_host,
            conv_out_c,
            conv_s.t as usize,
            conv_s.h as usize,
            conv_s.w as usize,
            stride,
        );
        // Skip: space-to-depth input (in_c * prod channels), grouped-mean -> out_c.
        let sd_skip = space_to_depth3d(
            &skip_host,
            in_c,
            skip_s.t as usize,
            skip_s.h as usize,
            skip_s.w as usize,
            stride,
        );
        let out_t = skip_s.t as usize / stride[0];
        let out_h = skip_s.h as usize / stride[1];
        let out_w = skip_s.w as usize / stride[2];
        let plane = out_t * out_h * out_w;
        let skip = group_mean_plane(&sd_skip, out_c, group_size, plane);
        debug_assert_eq!(main.len(), out_c * plane, "compress main channel math");

        let mut out = main;
        for (a, b) in out.iter_mut().zip(skip.iter()) {
            *a += b;
        }
        *s = Shape {
            c: out_c as u32,
            t: out_t as u32,
            h: out_h as u32,
            w: out_w as u32,
        };
        Ok(out)
    }

    /// Final `pixel_norm -> silu -> conv_out` (feature -> 129). Returns the raw
    /// `[129, T', H', W']` host output; `s` updated to `c = 129`.
    async fn run_conv_out<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        s: &mut Shape,
        host: &[f32],
    ) -> Result<Vec<f32>, LtxVaeError<S::Error>> {
        use thinfer_core::backend::Backend;
        let pl = &self.pipelines;
        let act = pl.act;
        let act_size = pl.act_size();
        debug_assert_eq!(s.c as usize, self.cfg.feature_out, "conv_out input width");

        let mut pins: Vec<GpuView> = Vec::new();
        let conv_out = acquire_conv(residency, backend, self.handles.conv_out, &mut pins).await?;

        let up = crate::common::seq::act_upload_bytes(act, host);
        let x_up = workspace.alloc(up.len() as u64)?;
        backend.write_buffer(x_up.id(), 0, &up)?;

        let (out_ws, out_s);
        {
            let scope = workspace.batch();
            let x = scope.import_copy(x_up.as_buf_ref());
            let n = pixel_norm(&scope, pl, x, *s)?;
            let a = silu(&scope, pl, n, *s)?;
            let (co, cos) = conv3d_k3(
                &scope,
                pl,
                a,
                *s,
                &conv_out,
                ENC_CONV_OUT_CHANNELS as u32,
                0,
                TimePad::Causal,
            )?;
            out_s = cos;
            out_ws = persist(&scope, workspace, co, cos, act_size)?;
            scope.submit_void().await?;
        }
        *s = out_s;
        Ok(read_acts(backend, &out_ws.as_buf_ref(), out_s.elems() as usize, act).await?)
    }
}

/// Host `patchify` (patch 4, inverse of [`unpatchify_4x4`]): `[3, t, hp, wp]` ->
/// `[48, t, hp/4, wp/4]`. einops `c (h q)(w r) -> (c r q) h w`, q,r = 4: pixel
/// `(c, hp=h*4+q, wp=w*4+r)` -> packed channel `(c*4 + r)*4 + q` at `(h, w)`.
fn patchify_4x4(frame: &[f32], t: usize, hp: usize, wp: usize) -> (Vec<f32>, Shape) {
    let (ho, wo) = (hp / PATCH_SIZE, wp / PATCH_SIZE);
    let mut out = vec![0.0f32; CONV_OUT_CHANNELS * t * ho * wo];
    let in_thw = t * hp * wp;
    let out_thw = t * ho * wo;
    for c in 0..OUT_CHANNELS {
        for r in 0..PATCH_SIZE {
            for q in 0..PATCH_SIZE {
                let cp = (c * PATCH_SIZE + r) * PATCH_SIZE + q;
                for tt in 0..t {
                    for h in 0..ho {
                        for w in 0..wo {
                            let src = c * in_thw
                                + (tt * hp + h * PATCH_SIZE + q) * wp
                                + w * PATCH_SIZE
                                + r;
                            let dst = cp * out_thw + (tt * ho + h) * wo + w;
                            out[dst] = frame[src];
                        }
                    }
                }
            }
        }
    }
    (
        out,
        Shape {
            c: CONV_OUT_CHANNELS as u32,
            t: t as u32,
            h: ho as u32,
            w: wo as u32,
        },
    )
}

/// Host space-to-depth (inverse of the decoder `DepthToSpace3d`): `[c, td, hd,
/// wd]` -> `[c*p1*p2*p3, td/p1, hd/p2, wd/p3]`. einops `c (d p1)(h p2)(w p3) ->
/// (c p1 p2 p3) d h w`: element `(c, d*p1+i1, h*p2+i2, w*p3+i3)` -> channel
/// `((c*p1+i1)*p2+i2)*p3+i3` at `(d, h, w)`.
fn space_to_depth3d(
    x: &[f32],
    c: usize,
    td: usize,
    hd: usize,
    wd: usize,
    p: [usize; 3],
) -> Vec<f32> {
    let (p1, p2, p3) = (p[0], p[1], p[2]);
    let (to, ho, wo) = (td / p1, hd / p2, wd / p3);
    let out_thw = to * ho * wo;
    let mut out = vec![0.0f32; c * p1 * p2 * p3 * out_thw];
    for ci in 0..c {
        for d in 0..to {
            for i1 in 0..p1 {
                for h in 0..ho {
                    for i2 in 0..p2 {
                        for w in 0..wo {
                            for i3 in 0..p3 {
                                let src = ci * td * hd * wd
                                    + (d * p1 + i1) * hd * wd
                                    + (h * p2 + i2) * wd
                                    + (w * p3 + i3);
                                let oc = ((ci * p1 + i1) * p2 + i2) * p3 + i3;
                                out[oc * out_thw + (d * ho + h) * wo + w] = x[src];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Contiguous grouped channel mean: `view(out_c, group, plane).mean(dim=1)`
/// (out channel `o` averages input channels `[o*group, (o+1)*group)`).
fn group_mean_plane(x: &[f32], out_c: usize, group: usize, plane: usize) -> Vec<f32> {
    assert_eq!(x.len(), out_c * group * plane, "group_mean size");
    let mut out = vec![0.0f32; out_c * plane];
    let inv = 1.0 / group as f32;
    for o in 0..out_c {
        for g in 0..group {
            let base = (o * group + g) * plane;
            for p in 0..plane {
                out[o * plane + p] += x[base + p];
            }
        }
    }
    for v in &mut out {
        *v *= inv;
    }
    out
}

// ============================================================================
// Spatial tiling: overlap tiles + feather blend, sized from the VRAM budget.
// Mirrors upstream LTX `SpatialTilingConfig` (overlap + blend) -- the same
// overlap/feather math the in-repo Wan VAE decode uses. Latent-cell granularity
// (1 cell = 32 px) matches upstream's /32 tile/overlap rule.
// ============================================================================

/// Min/max latent tile side (cells). Min 2 = 64 px (upstream floor); max caps a
/// roomy budget from picking a TDR-prone megatile.
const TILE_MIN: u32 = 2;
const TILE_MAX: u32 = 24;
/// Min/max latent temporal tile depth (frames). Min 2 = 9 output frames
/// (upstream temporal floor is 16 video frames = 2 latent); max caps the depth.
const TEMPORAL_TILE_MIN: u32 = 2;
const TEMPORAL_TILE_MAX: usize = 16;
/// Spatial tile side we aim to keep before introducing temporal tiling (cells;
/// 6 = 192 px ~ upstream's overlap-respecting floor). Below this the temporal
/// tile is capped so the spatial tile can re-grow to it.
const SPATIAL_COMFORT: u32 = 6;
/// Upload/readback staging + fragmentation headroom.
const VAE_STAGING_RESERVE: u64 = 256 * 1024 * 1024;
/// Fraction of the workspace budget the INITIAL tile is sized to: the peak
/// estimate is accurate to ~edge but doesn't model alloc fragmentation, so a
/// margin keeps the first attempt off the OOM boundary.
const SEED_SAFETY: f64 = 0.82;
/// Budget step-down per OOM re-seed (balanced shrink, not an axis halve).
const OOM_SHRINK: f64 = 0.7;
/// Reference budget when the residency budget is unbounded (`u64::MAX`).
const DEFAULT_REF_BUDGET: u64 = 4 * 1024 * 1024 * 1024;
/// Calibration: approx peak decode workspace bytes per (latent tile area * output
/// frame). Re-measured at 512x320x121 + temporal tiling (a tile-6/25-frame
/// attempt overshot a ~4.3 GiB workspace budget, recovering at tile 3 -> actual
/// ~7-8e6/area/frame, ~1.8x the original 256x256 anchor's 4.2e6). Only seeds the
/// INITIAL tile; the strict-budget adaptive retry corrects any error WITHOUT a
/// device OOM, so an over-estimate just means one extra (cheap) shrink.
const PEAK_BYTES_PER_AREA_FRAME_F32: f64 = 9.0e6;

/// Peak workspace bytes per (latent tile area * output frame) for `act_size`-byte
/// activations. The whole conv-stack activation set scales with the act dtype, so
/// f16 (2 bytes) roughly halves the peak vs the f32-measured anchor -> ~2x more
/// area-frames fit per tile (fewer tiles, less overlap recompute).
fn peak_bytes_per_area_frame(act_size: u32) -> f64 {
    PEAK_BYTES_PER_AREA_FRAME_F32 * act_size as f64 / 4.0
}

/// Largest latent tile whose estimated decode workspace fits `workspace_budget`
/// for `f_px` output frames. Clamped to `[TILE_MIN, TILE_MAX]`.
fn initial_tile(workspace_budget: u64, f_px: usize, act_size: u32) -> u32 {
    let denom = (peak_bytes_per_area_frame(act_size) * f_px.max(1) as f64).max(1.0);
    let t = (workspace_budget as f64 / denom).sqrt() as u32;
    t.clamp(TILE_MIN, TILE_MAX)
}

/// Seed the (temporal-depth, spatial-side) tile pair from the workspace budget.
/// Prefer no temporal tiling: if a comfortable spatial tile already fits with
/// the full frame count, keep `tf = f`. Otherwise cap the temporal depth so the
/// spatial tile can re-grow to `SPATIAL_COMFORT`, balancing seams across axes
/// (total tile count is ~budget-fixed either way). The adaptive OOM-retry in
/// `decode` corrects any estimate error.
fn initial_tiles(workspace_budget: u64, f: usize, act_size: u32) -> (usize, u32) {
    let f_px_full = TEMPORAL_SCALE * (f - 1) + 1;
    let tile_full = initial_tile(workspace_budget, f_px_full, act_size);
    if tile_full >= SPATIAL_COMFORT || f <= TEMPORAL_TILE_MIN as usize {
        return (f, tile_full);
    }
    // Per-temporal-tile output-frame budget at the comfort spatial tile.
    let cap = workspace_budget as f64 / peak_bytes_per_area_frame(act_size);
    let f_px_target = (cap / (SPATIAL_COMFORT as f64).powi(2)).max(1.0);
    // f_px_tile = 8*(tf-1)+1 <= f_px_target  =>  tf <= (f_px_target-1)/8 + 1.
    let tf = (((f_px_target - 1.0) / TEMPORAL_SCALE as f64 + 1.0) as usize)
        .clamp(TEMPORAL_TILE_MIN as usize, TEMPORAL_TILE_MAX.min(f));
    let f_px_tile = TEMPORAL_SCALE * (tf - 1) + 1;
    (tf, initial_tile(workspace_budget, f_px_tile, act_size))
}

// Tiling geometry (plan/feather/trapezoid/gather/blend) is shared across video
// VAEs in `common::vae_tiling`; LTX passes `TEMPORAL_SCALE`/`SPATIAL_SCALE` /
// channel counts as arguments. CPU `vae_tiling::tests` gate the math.
