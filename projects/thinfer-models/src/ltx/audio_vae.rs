//! LTX-2.3 audio VAE decoder (latent mel -> mel spectrogram). 2D causal-conv
//! autoencoder over a stereo mel, separate from the 3D video VAE. Decodes a
//! `[8, frames, 16]` latent to a `[2, 4*frames-3, 64]` mel that feeds the vocoder
//! (P5 vocoder is a separate, larger module).
//!
//! Ground truth: `ltx_core/model/audio_vae/audio_vae.py` (`AudioDecoder`),
//! `causal_conv_2d.py`, `resnet.py`, `upsample.py`, `normalization.py`. Config +
//! tensors disk-verified from `audio_vae.safetensors` `__metadata__.config`
//! (prefix `audio_vae.decoder.*`): ch 128, ch_mult [1,2,4], z 8, latent mel_bins
//! 16 -> decoded 64, out_ch 2, `norm_type=pixel` (PixelNorm = channel-RMS eps
//! 1e-6, NO affine), `causality_axis=height` (=time), NO attention
//! (`mid_block_add_attention=false`, `attn_resolutions=[]`), NO temb/noise.
//!
//! ZERO net-new ops (same reuse strategy as `upsampler.rs`): the audio CausalConv2d
//! over `[B, C, time, freq]` is run as a `Conv3dF32` over `[C, time, freq, 1]` with
//! kernel `(3,3,1)`. The kernel's front-only temporal pad (`pad_t=2`) IS the causal
//! past-only window the audio model wants; `pad_h=1` gives the symmetric freq pad;
//! the stored `[Cout,Cin,3,3]` conv2d weight is byte-identical to `[*,*,3,3,1]`.
//! PixelNorm reuses `PixelNorm3dF32` (eps 1e-6). Upsample = nearest x2 on both axes
//! (`Upsample2dNearestF32`) -> causal conv -> drop the first time frame.
//!
//! Graph: conv_in 8->512 -> mid(resnet,resnet) -> up[2]=3 resblk@512 +ups ->
//! up[1]=3 resblk 512->256 +ups -> up[0]=3 resblk 256->128 -> norm_out -> silu ->
//! conv_out 128->2. Two upsamples (time 2->3->5, freq 16->32->64 for frames=2).
//! ResnetBlock: `pixelnorm->silu->conv ->pixelnorm->silu->conv + (1x1 shortcut)`.

use thinfer_core::backend::{BufRef, WgpuBackend, WgpuError};
use thinfer_core::ops::{
    ActDtype, AddF32, Conv3dConfig, Conv3dF32, Conv3dOp, Op, PixelNorm3dF32, PixelNorm3dOp,
    SiluF32, Upsample2dNearestF32, Upsample2dNearestOp, WeightDtype, WgslConfig,
};
use thinfer_core::residency::{GpuView, ResidencyError, WeightHandle, WeightResidency};
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace};

use crate::common::loader::{LoadError, register_passthrough};

/// Disk-verified config (the distilled checkpoint is the only variant).
pub const LATENT_CHANNELS: usize = 8;
pub const LATENT_MEL_BINS: usize = 16;
/// `latent_channels * latent_mel_bins` = the patchified per-frame width the
/// per-channel stats normalize over.
pub const PATCH_WIDTH: usize = LATENT_CHANNELS * LATENT_MEL_BINS;
pub const OUT_CHANNELS: usize = 2;
pub const MEL_BINS: usize = 64;
const BASE_CH: usize = 128;
const CH_MULT: [usize; 3] = [1, 2, 4];
/// `num_res_blocks + 1` ResnetBlocks per up stage.
const BLOCKS_PER_STAGE: usize = 3;
/// `nn.GroupNorm`-free PixelNorm eps (`normalization.py`).
const PIXEL_NORM_EPS: f32 = 1e-6;

// ============================================================================
// Pipelines
// ============================================================================

pub struct AudioVaePipelines {
    conv3d: Conv3dF32,
    conv3d_pipeline: thinfer_core::backend::WgpuPipeline,
    pixel_norm: thinfer_core::backend::WgpuPipeline,
    silu: thinfer_core::backend::WgpuPipeline,
    add: thinfer_core::backend::WgpuPipeline,
    upsample: thinfer_core::backend::WgpuPipeline,
}

impl AudioVaePipelines {
    pub async fn compile(backend: &WgpuBackend) -> Result<Self, WgpuError> {
        use thinfer_core::backend::Backend;
        let cfg = &WgslConfig {
            bf16_quant_writes: false,
            act_dtype: ActDtype::F32,
            weight_dtype: WeightDtype::Bf16,
        };
        let conv3d = Conv3dF32::new(Conv3dConfig::DEFAULT);
        Ok(Self {
            conv3d_pipeline: backend
                .create_pipeline(
                    "ltx_audio_vae_conv3d",
                    &conv3d.wgsl(cfg),
                    "main",
                    <Conv3dF32 as Conv3dOp>::layout(),
                )
                .await?,
            conv3d,
            pixel_norm: backend
                .create_pipeline(
                    "ltx_audio_vae_pixel_norm3d",
                    &<PixelNorm3dF32 as PixelNorm3dOp>::wgsl(cfg),
                    "main",
                    <PixelNorm3dF32 as PixelNorm3dOp>::layout(),
                )
                .await?,
            silu: backend
                .create_pipeline(
                    "ltx_audio_vae_silu",
                    SiluF32::wgsl(cfg),
                    "main",
                    SiluF32::layout(),
                )
                .await?,
            add: backend
                .create_pipeline(
                    "ltx_audio_vae_add",
                    AddF32::wgsl(cfg),
                    "main",
                    AddF32::layout(),
                )
                .await?,
            upsample: backend
                .create_pipeline(
                    "ltx_audio_vae_upsample2d",
                    <Upsample2dNearestF32 as Upsample2dNearestOp>::wgsl(cfg),
                    "main",
                    <Upsample2dNearestF32 as Upsample2dNearestOp>::layout(),
                )
                .await?,
        })
    }
}

// ============================================================================
// Shapes / uniforms ([C, time, freq] row-major f32; conv runs as [C,t,f,1])
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape {
    c: u32,
    t: u32,
    f: u32,
}

impl Shape {
    fn elems(&self) -> u32 {
        self.c * self.t * self.f
    }
    fn bytes(&self) -> u64 {
        self.elems() as u64 * 4
    }
    fn tf(&self) -> u32 {
        self.t * self.f
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
) -> [u8; 80] {
    let fields: [u32; 20] = [
        1, cin, cout, t_in, h_in, w_in, t_out, h_out, w_out, ker.0, ker.1, ker.2, pad.0, pad.1,
        pad.2, 1, 1, 1, 0, 0,
    ];
    let mut bytes = [0u8; 80];
    for (i, v) in fields.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn pixel_norm_uniform_bytes(n_pos: u32, channels: u32, stride: u32, eps: f32) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&n_pos.to_le_bytes());
    bytes[4..8].copy_from_slice(&channels.to_le_bytes());
    bytes[8..12].copy_from_slice(&stride.to_le_bytes());
    bytes[12..16].copy_from_slice(&eps.to_bits().to_le_bytes());
    bytes
}

fn upsample_uniform_bytes(c: u32, h_in: u32, w_in: u32) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&1u32.to_le_bytes());
    bytes[4..8].copy_from_slice(&c.to_le_bytes());
    bytes[8..12].copy_from_slice(&h_in.to_le_bytes());
    bytes[12..16].copy_from_slice(&w_in.to_le_bytes());
    bytes
}

// ============================================================================
// Op wrappers
// ============================================================================

struct ConvBufs {
    weight: BufRef,
    bias: BufRef,
}

/// Causal conv: `Conv3dF32` over `[c, t, f, 1]` with kernel `(kt, kf, 1)`. For k3
/// (`kt=kf=3`) `pad_t=2` is the causal past-only time window and `pad_h=1` is the
/// symmetric freq pad; for the 1x1 `nin_shortcut` (`kt=kf=1`) pad is 0.
fn conv_causal<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &AudioVaePipelines,
    x: BatchBuf<'w>,
    s: Shape,
    w: &ConvBufs,
    cout: u32,
    k: u32,
) -> Result<(BatchBuf<'w>, Shape), WgpuError> {
    let out_shape = Shape { c: cout, ..s };
    let out = scope.alloc(out_shape.bytes())?;
    let (pad_t, pad_f) = if k == 3 { (2, 1) } else { (0, 0) };
    let u = scope.write_uniform(&conv3d_uniform_bytes(
        s.c,
        cout,
        s.t,
        s.f,
        1,
        s.t,
        s.f,
        1,
        (k, k, 1),
        (pad_t, pad_f, 0),
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
        out_shape.tf(),
        1,
    )?;
    Ok((out, out_shape))
}

/// PixelNorm: channel-RMS `x / sqrt(mean_c(x^2) + eps)` over the C axis at each
/// `(t, f)` (eps 1e-6, weightless). Reuses `PixelNorm3dF32` with `n_pos = t*f`,
/// channel stride `t*f`.
fn pixel_norm<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &AudioVaePipelines,
    x: BatchBuf<'w>,
    s: Shape,
) -> Result<BatchBuf<'w>, WgpuError> {
    let out = scope.alloc(s.bytes())?;
    let u = scope.write_uniform(&pixel_norm_uniform_bytes(
        s.tf(),
        s.c,
        s.tf(),
        PIXEL_NORM_EPS,
    ))?;
    scope.pixel_norm3d::<PixelNorm3dF32>(&pl.pixel_norm, x, u, out, s.tf())?;
    Ok(out)
}

fn silu<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &AudioVaePipelines,
    x: BatchBuf<'w>,
    s: Shape,
) -> Result<BatchBuf<'w>, WgpuError> {
    let out = scope.alloc(s.bytes())?;
    scope.dispatch_op::<SiluF32>(&pl.silu, &[x], out)?;
    Ok(out)
}

fn add<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &AudioVaePipelines,
    a: BatchBuf<'w>,
    b: BatchBuf<'w>,
    s: Shape,
) -> Result<BatchBuf<'w>, WgpuError> {
    let out = scope.alloc(s.bytes())?;
    scope.dispatch_op::<AddF32>(&pl.add, &[a, b], out)?;
    Ok(out)
}

/// One ResnetBlock: `pixelnorm->silu->conv1 ->pixelnorm->silu->conv2 + shortcut`.
/// The shortcut is a 1x1 causal conv (`nin_shortcut`) iff channels change.
fn resnet_block<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &AudioVaePipelines,
    x: BatchBuf<'w>,
    s: Shape,
    w: &ResnetBufs,
    cout: u32,
) -> Result<(BatchBuf<'w>, Shape), WgpuError> {
    let h = pixel_norm(scope, pl, x, s)?;
    let h = silu(scope, pl, h, s)?;
    let (h, hs) = conv_causal(scope, pl, h, s, &w.conv1, cout, 3)?;
    let h = pixel_norm(scope, pl, h, hs)?;
    let h = silu(scope, pl, h, hs)?;
    let (h, hs) = conv_causal(scope, pl, h, hs, &w.conv2, cout, 3)?;
    let residual = match &w.nin_shortcut {
        Some(sc) => conv_causal(scope, pl, x, s, sc, cout, 1)?.0,
        None => x,
    };
    let out = add(scope, pl, h, residual, hs)?;
    Ok((out, hs))
}

/// Upsample: nearest x2 on both (time, freq) -> causal conv k3 -> drop the first
/// time frame (`x[:, :, 1:, :]`, the causal-axis crop). `[c,t,f] -> [c,2t-1,2f]`.
fn upsample<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    pl: &AudioVaePipelines,
    x: BatchBuf<'w>,
    s: Shape,
    w: &ConvBufs,
) -> Result<(BatchBuf<'w>, Shape), WgpuError> {
    // nearest x2 over [C, h=t, w=f].
    let up_shape = Shape {
        c: s.c,
        t: s.t * 2,
        f: s.f * 2,
    };
    let up = scope.alloc(up_shape.bytes())?;
    let u = scope.write_uniform(&upsample_uniform_bytes(s.c, s.t, s.f))?;
    scope.upsample2d_nearest::<Upsample2dNearestF32>(&pl.upsample, x, u, up, up_shape.elems())?;
    // causal conv (same channels).
    let (conv, conv_s) = conv_causal(scope, pl, up, up_shape, w, s.c, 3)?;
    // drop the first time frame: [c, 2t, 2f] -> [c, 2t-1, 2f], per-channel copy
    // (channels are not contiguous after the crop).
    let out_s = Shape {
        t: conv_s.t - 1,
        ..conv_s
    };
    let out = scope.alloc(out_s.bytes())?;
    let row = (conv_s.f) as u64 * 4; // bytes per time row (per channel)
    let src_chan = conv_s.tf() as u64 * 4;
    let dst_chan = out_s.tf() as u64 * 4;
    for c in 0..conv_s.c as u64 {
        // copy time rows [1..t) of channel c.
        scope.copy_buffer_to_buffer(conv, c * src_chan + row, out, c * dst_chan, dst_chan)?;
    }
    Ok((out, out_s))
}

// ============================================================================
// Weights / handles
// ============================================================================

fn conv_ids(prefix: &str) -> (WeightId, WeightId) {
    // CausalConv2d wraps nn.Conv2d in `.conv`, so the leaf is `{prefix}.conv.*`.
    (
        WeightId(format!("{prefix}.conv.weight")),
        WeightId(format!("{prefix}.conv.bias")),
    )
}

struct ResnetW {
    conv1: (WeightId, WeightId),
    conv2: (WeightId, WeightId),
    nin_shortcut: Option<(WeightId, WeightId)>,
}

impl ResnetW {
    fn new(prefix: &str, c_in: usize, c_out: usize) -> Self {
        Self {
            conv1: conv_ids(&format!("{prefix}.conv1")),
            conv2: conv_ids(&format!("{prefix}.conv2")),
            nin_shortcut: (c_in != c_out).then(|| conv_ids(&format!("{prefix}.nin_shortcut"))),
        }
    }
}

struct UpStageW {
    blocks: Vec<ResnetW>,
    upsample: Option<(WeightId, WeightId)>,
}

struct DecoderW {
    conv_in: (WeightId, WeightId),
    mid: Vec<ResnetW>,
    up: Vec<UpStageW>,
    conv_out: (WeightId, WeightId),
}

impl DecoderW {
    fn new() -> Self {
        let pfx = "audio_vae.decoder";
        let base_out = BASE_CH * CH_MULT[CH_MULT.len() - 1]; // 512
        let mid = vec![
            ResnetW::new(&format!("{pfx}.mid.block_1"), base_out, base_out),
            ResnetW::new(&format!("{pfx}.mid.block_2"), base_out, base_out),
        ];
        // Up stages stored ascending [0,1,2]; forward walks 2,1,0. block_in threads
        // from the mid output (512), each stage's first block changes channels.
        let mut up: Vec<UpStageW> = Vec::with_capacity(CH_MULT.len());
        for _ in 0..CH_MULT.len() {
            up.push(UpStageW {
                blocks: Vec::new(),
                upsample: None,
            });
        }
        let mut block_in = base_out;
        for level in (0..CH_MULT.len()).rev() {
            let block_out = BASE_CH * CH_MULT[level];
            let mut blocks = Vec::with_capacity(BLOCKS_PER_STAGE);
            for i in 0..BLOCKS_PER_STAGE {
                blocks.push(ResnetW::new(
                    &format!("{pfx}.up.{level}.block.{i}"),
                    block_in,
                    block_out,
                ));
                block_in = block_out;
            }
            let upsample =
                (level != 0).then(|| conv_ids(&format!("{pfx}.up.{level}.upsample.conv")));
            up[level] = UpStageW { blocks, upsample };
        }
        Self {
            conv_in: conv_ids(&format!("{pfx}.conv_in")),
            mid,
            up,
            conv_out: conv_ids(&format!("{pfx}.conv_out")),
        }
    }
}

#[derive(Clone, Copy)]
struct ConvH {
    weight: WeightHandle,
    bias: WeightHandle,
}

struct ResnetH {
    conv1: ConvH,
    conv2: ConvH,
    nin_shortcut: Option<ConvH>,
}

struct UpStageH {
    blocks: Vec<ResnetH>,
    upsample: Option<ConvH>,
}

struct DecoderH {
    conv_in: ConvH,
    mid: Vec<ResnetH>,
    up: Vec<UpStageH>,
    conv_out: ConvH,
}

fn reg_conv<S: WeightSource>(
    res: &WeightResidency<S>,
    ids: &(WeightId, WeightId),
) -> Result<ConvH, LoadError> {
    Ok(ConvH {
        weight: register_passthrough(res, &ids.0)?,
        bias: register_passthrough(res, &ids.1)?,
    })
}

fn reg_resnet<S: WeightSource>(
    res: &WeightResidency<S>,
    w: &ResnetW,
) -> Result<ResnetH, LoadError> {
    Ok(ResnetH {
        conv1: reg_conv(res, &w.conv1)?,
        conv2: reg_conv(res, &w.conv2)?,
        nin_shortcut: w
            .nin_shortcut
            .as_ref()
            .map(|s| reg_conv(res, s))
            .transpose()?,
    })
}

impl DecoderH {
    fn register<S: WeightSource>(
        res: &WeightResidency<S>,
        w: &DecoderW,
    ) -> Result<Self, LoadError> {
        let mid = w
            .mid
            .iter()
            .map(|b| reg_resnet(res, b))
            .collect::<Result<_, _>>()?;
        let mut up = Vec::with_capacity(w.up.len());
        for stage in &w.up {
            up.push(UpStageH {
                blocks: stage
                    .blocks
                    .iter()
                    .map(|b| reg_resnet(res, b))
                    .collect::<Result<_, _>>()?,
                upsample: stage
                    .upsample
                    .as_ref()
                    .map(|s| reg_conv(res, s))
                    .transpose()?,
            });
        }
        Ok(Self {
            conv_in: reg_conv(res, &w.conv_in)?,
            mid,
            up,
            conv_out: reg_conv(res, &w.conv_out)?,
        })
    }
}

// Acquired bufs -----------------------------------------------------------------

struct ResnetBufs {
    conv1: ConvBufs,
    conv2: ConvBufs,
    nin_shortcut: Option<ConvBufs>,
}

struct UpStageBufs {
    blocks: Vec<ResnetBufs>,
    upsample: Option<ConvBufs>,
}

struct DecoderBufs {
    conv_in: ConvBufs,
    mid: Vec<ResnetBufs>,
    up: Vec<UpStageBufs>,
    conv_out: ConvBufs,
}

async fn acquire_conv<'r, S: WeightSource>(
    res: &'r WeightResidency<S>,
    backend: &WgpuBackend,
    h: ConvH,
    pins: &mut Vec<GpuView<'r>>,
) -> Result<ConvBufs, ResidencyError<S::Error, WgpuError>> {
    let wv = res.acquire(h.weight, backend).await?;
    let bv = res.acquire(h.bias, backend).await?;
    let bufs = ConvBufs {
        weight: wv.buf(),
        bias: bv.buf(),
    };
    pins.push(wv);
    pins.push(bv);
    Ok(bufs)
}

async fn acquire_resnet<'r, S: WeightSource>(
    res: &'r WeightResidency<S>,
    backend: &WgpuBackend,
    h: &ResnetH,
    pins: &mut Vec<GpuView<'r>>,
) -> Result<ResnetBufs, ResidencyError<S::Error, WgpuError>> {
    let nin_shortcut = match &h.nin_shortcut {
        Some(c) => Some(acquire_conv(res, backend, *c, pins).await?),
        None => None,
    };
    Ok(ResnetBufs {
        conv1: acquire_conv(res, backend, h.conv1, pins).await?,
        conv2: acquire_conv(res, backend, h.conv2, pins).await?,
        nin_shortcut,
    })
}

// ============================================================================
// Decoder
// ============================================================================

pub struct AudioVaeDecoder {
    pub pipelines: AudioVaePipelines,
    handles: DecoderH,
    /// Per-channel latent stats over the patchified 128 (= 8ch x 16mel).
    mean: Vec<f32>,
    std: Vec<f32>,
}

#[derive(Debug)]
pub enum AudioVaeError<SE: core::fmt::Debug> {
    Wgpu(WgpuError),
    Load(LoadError),
    Residency(ResidencyError<SE, WgpuError>),
}
impl<SE: core::fmt::Debug> From<WgpuError> for AudioVaeError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}
impl<SE: core::fmt::Debug> From<LoadError> for AudioVaeError<SE> {
    fn from(e: LoadError) -> Self {
        Self::Load(e)
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for AudioVaeError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

/// Load the audio VAE per-channel latent stats (`audio_vae.per_channel_statistics.
/// {mean,std}-of-means`, bf16 `[128]`).
pub async fn load_latent_stats<S: WeightSource>(
    residency: &WeightResidency<S>,
    backend: &WgpuBackend,
) -> Result<(Vec<f32>, Vec<f32>), AudioVaeError<S::Error>> {
    use thinfer_core::backend::Backend;
    let read = |id: &str| register_passthrough(residency, &WeightId(id.into()));
    let mean_h = read("audio_vae.per_channel_statistics.mean-of-means")?;
    let std_h = read("audio_vae.per_channel_statistics.std-of-means")?;
    let mut out = Vec::new();
    for h in [mean_h, std_h] {
        let view = residency.acquire(h, backend).await?;
        let bytes = backend
            .read_buffer(view.buf().id, 0, (PATCH_WIDTH * 2) as u64)
            .await?;
        out.push(crate::common::seq::act_readback_to_f32(
            ActDtype::Bf16,
            &bytes,
            PATCH_WIDTH,
        ));
    }
    let std = out.pop().expect("std");
    let mean = out.pop().expect("mean");
    Ok((mean, std))
}

impl AudioVaeDecoder {
    pub fn new<S: WeightSource>(
        pipelines: AudioVaePipelines,
        residency: &WeightResidency<S>,
        mean: Vec<f32>,
        std: Vec<f32>,
    ) -> Result<Self, LoadError> {
        assert_eq!(mean.len(), PATCH_WIDTH);
        assert_eq!(std.len(), PATCH_WIDTH);
        let handles = DecoderH::register(residency, &DecoderW::new())?;
        Ok(Self {
            pipelines,
            handles,
            mean,
            std,
        })
    }

    /// Decode a `[8, frames, 16]` latent (CTF row-major, NORMALIZED) into a mel
    /// `[2, 4*frames-3, 64]` host f32. Host-side per-channel un-normalize
    /// (`x*std+mean` over the patchified 128) then the conv decode.
    pub async fn decode<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        latent: &[f32],
        frames: usize,
    ) -> Result<Vec<f32>, AudioVaeError<S::Error>> {
        use thinfer_core::backend::Backend;
        assert_eq!(
            latent.len(),
            LATENT_CHANNELS * frames * LATENT_MEL_BINS,
            "audio latent size"
        );

        // Host un-normalize: latent[c, t, f] *= std[c*16+f] + mean[c*16+f].
        let tf = frames * LATENT_MEL_BINS;
        let mut z = latent.to_vec();
        for c in 0..LATENT_CHANNELS {
            for t in 0..frames {
                for fb in 0..LATENT_MEL_BINS {
                    let idx = c * tf + t * LATENT_MEL_BINS + fb;
                    let p = c * LATENT_MEL_BINS + fb;
                    z[idx] = z[idx] * self.std[p] + self.mean[p];
                }
            }
        }

        let mut pins: Vec<GpuView> = Vec::new();
        let bufs = self.acquire(residency, backend, &mut pins).await?;

        let in_bytes: Vec<u8> = z.iter().flat_map(|v| v.to_le_bytes()).collect();
        let in_buf = workspace.alloc(in_bytes.len() as u64)?;
        backend.write_buffer(in_buf.id(), 0, &in_bytes)?;

        // Output mel [2, 4*frames-3, 64].
        let t_out = 4 * frames - 3;
        let out_elems = OUT_CHANNELS * t_out * MEL_BINS;
        let out_buf = workspace.alloc((out_elems * 4) as u64)?;
        {
            let scope = workspace.batch();
            let pl = &self.pipelines;
            let x = scope.import_copy(in_buf.as_buf_ref());
            let s = Shape {
                c: LATENT_CHANNELS as u32,
                t: frames as u32,
                f: LATENT_MEL_BINS as u32,
            };

            // conv_in 8 -> 512.
            let (mut h, mut hs) = conv_causal(
                &scope,
                pl,
                x,
                s,
                &bufs.conv_in,
                (BASE_CH * CH_MULT[2]) as u32,
                3,
            )?;
            // mid (resnet, resnet), channel-preserving.
            for blk in &bufs.mid {
                let (nh, ns) = resnet_block(&scope, pl, h, hs, blk, hs.c)?;
                h = nh;
                hs = ns;
            }
            // up stages 2,1,0.
            for level in (0..CH_MULT.len()).rev() {
                let block_out = (BASE_CH * CH_MULT[level]) as u32;
                let stage = &bufs.up[level];
                for blk in &stage.blocks {
                    let (nh, ns) = resnet_block(&scope, pl, h, hs, blk, block_out)?;
                    h = nh;
                    hs = ns;
                }
                if let Some(up) = &stage.upsample {
                    let (nh, ns) = upsample(&scope, pl, h, hs, up)?;
                    h = nh;
                    hs = ns;
                }
            }
            // norm_out -> silu -> conv_out 128 -> 2.
            let h = pixel_norm(&scope, pl, h, hs)?;
            let h = silu(&scope, pl, h, hs)?;
            let (mel, mel_s) =
                conv_causal(&scope, pl, h, hs, &bufs.conv_out, OUT_CHANNELS as u32, 3)?;
            debug_assert_eq!(mel_s.t as usize, t_out);
            debug_assert_eq!(mel_s.f as usize, MEL_BINS);
            let dst = scope.import_copy(out_buf.as_buf_ref());
            scope.copy_buffer_to_buffer(mel, 0, dst, 0, (out_elems * 4) as u64)?;
            scope.submit_void().await?;
        }
        drop(pins);

        let bytes = backend
            .read_buffer(out_buf.id(), 0, (out_elems * 4) as u64)
            .await?;
        Ok(crate::common::seq::act_readback_to_f32(
            ActDtype::F32,
            &bytes,
            out_elems,
        ))
    }

    async fn acquire<'r, S: WeightSource>(
        &self,
        res: &'r WeightResidency<S>,
        backend: &WgpuBackend,
        pins: &mut Vec<GpuView<'r>>,
    ) -> Result<DecoderBufs, ResidencyError<S::Error, WgpuError>> {
        let conv_in = acquire_conv(res, backend, self.handles.conv_in, pins).await?;
        let mut mid = Vec::with_capacity(self.handles.mid.len());
        for b in &self.handles.mid {
            mid.push(acquire_resnet(res, backend, b, pins).await?);
        }
        let mut up = Vec::with_capacity(self.handles.up.len());
        for stage in &self.handles.up {
            let mut blocks = Vec::with_capacity(stage.blocks.len());
            for b in &stage.blocks {
                blocks.push(acquire_resnet(res, backend, b, pins).await?);
            }
            let upsample = match &stage.upsample {
                Some(c) => Some(acquire_conv(res, backend, *c, pins).await?),
                None => None,
            };
            up.push(UpStageBufs { blocks, upsample });
        }
        let conv_out = acquire_conv(res, backend, self.handles.conv_out, pins).await?;
        Ok(DecoderBufs {
            conv_in,
            mid,
            up,
            conv_out,
        })
    }
}
