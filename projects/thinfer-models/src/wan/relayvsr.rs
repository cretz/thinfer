//! RelayVSR video super-resolution (`Comfy-Org/RelayVSR`; ComfyUI
//! `comfy/ldm/wan/model_vsr.py` + `nodes_wan.py` RelayVSRConditioning /
//! VSRDecode): x2 or x4 upscale of a whole clip.
//!
//! 1. Keyframes = every 15th frame + the last. A Wan2.2-TI2V-5B (window
//!    self-attention, convrot int8 linears, baked text context) makes ONE
//!    forward at t=1000 on Gaussian noise with each keyframe's projected LR
//!    pixels (`lq_proj_in`) added to the patch tokens; the keyframe latent is
//!    `noise - v` (euler, 1 step, cfg 1).
//! 2. The Flash decoder ([`super::relay_flash`]) restores every frame from the
//!    two keyframe latents around it and the frame's own LR pixels.
//!
//! The x2 pair is LightX2V's retrain "for the step after H3 generation".

use std::collections::HashMap;
use std::sync::Arc;

use half::bf16;
use thinfer_core::backend::{Backend, WgpuBackend, WgpuError};
use thinfer_core::format::safetensors::{SafetensorsSource, SourceError};
use thinfer_core::format::union::{RenamedSource, UnionSource};
use thinfer_core::ops::{ActDtype, SiluF32, Transpose12F32, WeightDtype};
use thinfer_core::quant::QuantKind;
use thinfer_core::residency::{ResidencyError, WeightResidency};
use thinfer_core::trace;
use thinfer_core::weight::{
    FileOpener, OffsetView, WeightCatalog, WeightId, WeightReader, WeightSource,
};
use thinfer_core::workspace::{BatchScope, Workspace};

use crate::common::autoencoder_kl::conv2d_uniform;
use crate::common::block::{ActBuf, DenseActSites, alloc_act, op_rmsnorm};
use crate::common::convrot::ConvRotSource;
use crate::common::embedders::{LinearBiasHandles, linear_bias};
use crate::common::seq::act_upload_bytes;
use crate::wan::dit::{
    LoadedWanDitHandles, WanDit, WanDitError, WanDitInputs, WanDitShape, read_into_f32,
};
use crate::wan::dit_block::{WanDitConfig, WanDitPipelines, config as dit_config};
use crate::wan::loader::{
    LoadError, WanI8Sites, register_conv_as_linear_bias, register_linear_bias,
    register_passthrough, register_wan_dit_handles,
};
use crate::wan::pipeline::{SiteOverride, block_cfgs, gaussian_noise};
use crate::wan::relay_flash::{self, FlashDecoder, FlashHandles, FlashPipelines};
use crate::wan::source::dit_gguf_renames;

pub const KEYFRAME_INTERVAL: usize = 15;
/// Keyframe-model hidden width.
const DIM: u32 = 3072;
const LQ_HIDDEN: u32 = 768;
const LQ_EPS: f32 = 1e-6;
/// The one diffusion timestep RelayVSR runs (sigma 1 x 1000).
const TIMESTEP: f32 = 1000.0;

/// `range(0, F - 1, 15) + [F - 1]`.
pub fn keyframes(frames: usize) -> Vec<usize> {
    let mut k: Vec<usize> = (0..frames.saturating_sub(1))
        .step_by(KEYFRAME_INTERVAL)
        .collect();
    k.push(frames - 1);
    k
}

/// Latent grid `(h, w)` for an LR frame: whole 2x2 patches of the model's
/// pixel stride `ls = 16 / up`.
pub fn latent_size(h: usize, w: usize, ls: usize) -> (usize, usize) {
    let s = ls * 2;
    (h.div_ceil(s) * 2, w.div_ceil(s) * 2)
}

// ---------------------------------------------------------------------------
// Source
// ---------------------------------------------------------------------------

/// Serves the Flash decoder's fused `feed_forward.gate_up_proj.weight`
/// `[2 * ffn, dim]` as `gate_proj` (rows `0..ffn`) + `up_proj` (rows
/// `ffn..`): byte views of the same tensor, so the SwiGLU halves are two plain
/// linears.
pub struct FlashSplitSource<S: WeightSource> {
    base: S,
    catalog: WeightCatalog,
    /// split id -> (fused id, byte offset, byte len)
    views: HashMap<WeightId, (WeightId, u64, u64)>,
}

impl<S: WeightSource> FlashSplitSource<S> {
    pub fn new(base: S) -> Self {
        let mut catalog = WeightCatalog {
            entries: base.catalog().entries.clone(),
        };
        let mut views = HashMap::new();
        for (id, e) in &base.catalog().entries {
            let Some(stem) = id.0.strip_suffix("feed_forward.gate_up_proj.weight") else {
                continue;
            };
            let half = e.size / 2;
            for (i, name) in ["gate_proj", "up_proj"].into_iter().enumerate() {
                let sid = WeightId(format!("{stem}feed_forward.{name}.weight"));
                let mut se = e.clone();
                se.size = half;
                se.offset = e.offset + half * i as u64;
                se.shape.0[0] /= 2;
                catalog.entries.insert(sid.clone(), se);
                views.insert(sid, (id.clone(), half * i as u64, half));
            }
        }
        Self {
            base,
            catalog,
            views,
        }
    }
}

impl<S: WeightSource> WeightSource for FlashSplitSource<S> {
    type Reader = OffsetView<S::Reader>;
    type Error = S::Error;

    fn catalog(&self) -> &WeightCatalog {
        &self.catalog
    }

    async fn open(&self, id: &WeightId) -> Result<Self::Reader, Self::Error> {
        match self.views.get(id) {
            Some((fused, off, len)) => {
                Ok(OffsetView::new(self.base.open(fused).await?, *off, *len))
            }
            None => {
                let r = self.base.open(id).await?;
                let len = r.len();
                Ok(OffsetView::new(r, 0, len))
            }
        }
    }
}

pub type RelayVsrSource<O> = UnionSource<
    RenamedSource<ConvRotSource<SafetensorsSource<O>>>,
    FlashSplitSource<SafetensorsSource<O>>,
>;

/// Union of the keyframe DiT (convrot -> Q8_0, original-Wan names renamed to
/// the canonical ones the Wan loader registers) and the Flash decoder.
pub async fn open_relayvsr_source<O: FileOpener>(
    dit: O,
    flash: O,
) -> Result<RelayVsrSource<O>, SourceError<O::Error>> {
    let dit = SafetensorsSource::open(dit).await?;
    let flash = SafetensorsSource::open(flash).await?;
    let num_layers = WanDitConfig::relayvsr().num_layers;
    Ok(UnionSource::new(
        RenamedSource::with_passthrough(ConvRotSource::new(dit), dit_gguf_renames(num_layers)),
        FlashSplitSource::new(flash),
    ))
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum RelayVsrError<SE: core::fmt::Debug> {
    Load(LoadError),
    Wgpu(WgpuError),
    Residency(ResidencyError<SE, WgpuError>),
    Dit(WanDitError<SE>),
    Input(String),
    Cancelled,
}

impl<SE: core::fmt::Debug> From<LoadError> for RelayVsrError<SE> {
    fn from(e: LoadError) -> Self {
        Self::Load(e)
    }
}
impl<SE: core::fmt::Debug> From<WgpuError> for RelayVsrError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for RelayVsrError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}
impl<SE: core::fmt::Debug> From<WanDitError<SE>> for RelayVsrError<SE> {
    fn from(e: WanDitError<SE>) -> Self {
        Self::Dit(e)
    }
}

/// `lq_proj_in`: 1x1 conv (as a linear over the pixel-unshuffled LR) ->
/// RMSNorm -> SiLU -> 3x3/s2 conv -> RMSNorm -> SiLU -> Linear.
struct LqHandles {
    conv1: LinearBiasHandles,
    norm1: thinfer_core::residency::WeightHandle,
    conv2_w: thinfer_core::residency::WeightHandle,
    conv2_b: thinfer_core::residency::WeightHandle,
    norm2: thinfer_core::residency::WeightHandle,
    linear: LinearBiasHandles,
}

pub struct UpscaleInputs<'a> {
    /// Interleaved u8 RGB `[frames, height, width, 3]`.
    pub frames: &'a [u8],
    pub num_frames: usize,
    pub height: usize,
    pub width: usize,
    pub seed: u64,
    /// Scale on what the decoder adds to the bicubic upscale (1 = model).
    pub detail: f32,
}

pub struct UpscaledVideo {
    /// Interleaved u8 RGB `[num_frames, height, width, 3]`. Job plaintext:
    /// page-locked on a server, zeroized on drop.
    pub frames: thinfer_core::secmem::Locked<u8>,
    pub num_frames: usize,
    pub height: usize,
    pub width: usize,
}

/// Bring-up captures (parity vs the pyref).
#[derive(Default)]
pub struct RelayVsrDiag {
    /// Replaces the seeded noise `[48, K, lh, lw]` when set.
    pub noise_override: Option<Vec<f32>>,
    pub lq_tokens: Vec<f32>,
    pub velocity: Vec<f32>,
    pub kf_latent: Vec<f32>,
    /// Wall time per phase (`lq_proj`, `keyframes`, `flash`), ms.
    pub phase_ms: Vec<(&'static str, f64)>,
}

pub struct RelayVsrPipeline<S: WeightSource> {
    backend: Arc<WgpuBackend>,
    residency: WeightResidency<S>,
    up: usize,
    cfg: WanDitConfig,
    dit_handles: LoadedWanDitHandles,
    dit_pipelines: WanDitPipelines,
    lq: LqHandles,
    flash: FlashHandles,
    flash_pipelines: FlashPipelines,
    /// Baked `positive_conditioning` `[512, 4096]`.
    context: Vec<f32>,
}

fn wid(s: &str) -> WeightId {
    WeightId(s.to_string())
}

impl<S: WeightSource> RelayVsrPipeline<S> {
    /// `up` = 2 or 4, matching the DiT + Flash files the source was opened with.
    pub async fn load(
        backend: Arc<WgpuBackend>,
        residency: WeightResidency<S>,
        up: usize,
    ) -> Result<Self, RelayVsrError<S::Error>> {
        if up != 2 && up != 4 {
            return Err(RelayVsrError::Input(format!(
                "upscale factor {up} (2 or 4)"
            )));
        }
        // The window attention runs on the accelerator (tensor cores); the
        // WGSL fallback is ~300x slower, so a frame-sized dispatch would hit
        // the OS GPU watchdog and lose the device. Refuse instead.
        if backend.accelerator().is_none() || !backend.supports_shader_f16() {
            return Err(RelayVsrError::Input(
                "RelayVSR needs the CUDA accelerator and shader-f16 (window attention)".into(),
            ));
        }
        let cfg = WanDitConfig::relayvsr();
        // Block linears already arrive as Q8_0 (convrot repack): no load-time
        // transcode. Every site reads dense acts: the keyframe forward is one
        // deferred accelerator sequence per block (each site's Q8_0 weight
        // dequantized once, then a tensor-core GEMM), which has no DP4A arm.
        let i8_sites = WanI8Sites {
            qkv_self: None,
            ffn_up: None,
        };
        let dit_handles = register_wan_dit_handles(&residency, &cfg, "", None, i8_sites)?;
        let act = if backend.supports_shader_f16() {
            ActDtype::F16
        } else {
            ActDtype::F32
        };
        let q8w = WeightDtype::Quant(QuantKind::Q8_0);
        let dit_pipelines = WanDitPipelines::compile(
            &backend,
            &block_cfgs(
                q8w,
                act,
                SiteOverride {
                    qkv_self: Some(q8w),
                    ffn_up: Some(q8w),
                    dense_acts: DenseActSites {
                        qkv: true,
                        qkv_self: true,
                        proj: true,
                        ffn_up: true,
                        ffn_down: true,
                    },
                },
                false,
                false,
            ),
            false,
            true,
        )
        .await?;
        let lq = LqHandles {
            conv1: register_conv_as_linear_bias(
                &residency,
                &wid("lq_proj_in.conv1.weight"),
                &wid("lq_proj_in.conv1.bias"),
            )?,
            norm1: register_passthrough(&residency, &wid("lq_proj_in.norm1.weight"))?,
            conv2_w: register_passthrough(&residency, &wid("lq_proj_in.conv2.weight"))?,
            conv2_b: register_passthrough(&residency, &wid("lq_proj_in.conv2.bias"))?,
            norm2: register_passthrough(&residency, &wid("lq_proj_in.norm2.weight"))?,
            linear: register_linear_bias(
                &residency,
                &wid("lq_proj_in.linear.weight"),
                &wid("lq_proj_in.linear.bias"),
            )?,
        };
        let flash = relay_flash::register_flash(&residency, up as u32)?;
        let flash_pipelines = FlashPipelines::compile(&backend).await?;
        let context = read_bf16(&residency, "positive_conditioning")
            .await
            .map_err(|e| RelayVsrError::Input(format!("positive_conditioning: {e}")))?;
        Ok(Self {
            backend,
            residency,
            up,
            cfg,
            dit_handles,
            dit_pipelines,
            lq,
            flash,
            flash_pipelines,
            context,
        })
    }

    pub fn upscale_factor(&self) -> usize {
        self.up
    }

    /// LR frame `idx` as padded NCHW `[3, hp, wp]` in [-1, 1] (replicate pad).
    fn lr_frame(inputs: &UpscaleInputs<'_>, idx: usize, hp: usize, wp: usize) -> Vec<f32> {
        let (h, w) = (inputs.height, inputs.width);
        let src = &inputs.frames[idx * h * w * 3..][..h * w * 3];
        let mut out = vec![0f32; 3 * hp * wp];
        for y in 0..hp {
            let sy = y.min(h - 1);
            for x in 0..wp {
                let sx = x.min(w - 1);
                for c in 0..3 {
                    out[(c * hp + y) * wp + x] =
                        src[(sy * w + sx) * 3 + c] as f32 / 255.0 * 2.0 - 1.0;
                }
            }
        }
        out
    }

    /// `lq_proj_in` of one keyframe -> tokens `[gh * gw, 3072]` f32.
    async fn lq_tokens(
        &self,
        workspace: &Workspace<WgpuBackend>,
        lr: &[f32],
        hp: usize,
        wp: usize,
    ) -> Result<Vec<f32>, RelayVsrError<S::Error>> {
        let fp = &self.flash_pipelines;
        let bp = &fp.block;
        // x4: bilinear x4 first (align_corners=False), then unshuffle by 16.
        let (img, ih, iw) = if self.up == 4 {
            (bilinear_up(lr, 3, hp, wp, 4), hp * 4, wp * 4)
        } else {
            (lr.to_vec(), hp, wp)
        };
        let r = 4 * self.up;
        let (h1, w1) = (ih / r, iw / r);
        let cin = 3 * r * r;
        // pixel_unshuffle -> rows `[h1*w1, 3*r*r]`, channel index c*r*r + i*r + j.
        let mut rows1 = vec![0f32; h1 * w1 * cin];
        for y in 0..h1 {
            for x in 0..w1 {
                let row = &mut rows1[(y * w1 + x) * cin..][..cin];
                for c in 0..3 {
                    for i in 0..r {
                        for j in 0..r {
                            row[c * r * r + i * r + j] = img[(c * ih + y * r + i) * iw + x * r + j];
                        }
                    }
                }
            }
        }
        let (h2, w2) = ((h1 + 2 - 3) / 2 + 1, (w1 + 2 - 3) / 2 + 1);
        let (n1, n2) = ((h1 * w1) as u32, (h2 * w2) as u32);
        let in_bytes = act_upload_bytes(fp.act, &rows1);
        let tin = workspace.alloc(in_bytes.len() as u64)?;
        self.backend.write_buffer(tin.id(), 0, &in_bytes)?;
        let out = workspace.alloc(bp.act_bytes(n2 * DIM))?;
        {
            let c1 = self
                .lq
                .conv1
                .acquire(&self.residency, &self.backend)
                .await?;
            let nm1 = self
                .residency
                .acquire(self.lq.norm1, &*self.backend)
                .await?;
            let c2w = self
                .residency
                .acquire(self.lq.conv2_w, &*self.backend)
                .await?;
            let c2b = self
                .residency
                .acquire(self.lq.conv2_b, &*self.backend)
                .await?;
            let nm2 = self
                .residency
                .acquire(self.lq.norm2, &*self.backend)
                .await?;
            let lin = self
                .lq
                .linear
                .acquire(&self.residency, &self.backend)
                .await?;
            let scope = workspace.batch();
            let x = scope.import_copy(tin.as_buf_ref());
            let a = alloc_act(&scope, bp, n1, LQ_HIDDEN)?;
            linear_bias(&scope, bp, x, &c1.bufs(), n1, cin as u32, LQ_HIDDEN, a.data)?;
            let a = rms_silu(&scope, bp, a, scope.import_copy(nm1.buf()), n1, LQ_HIDDEN)?;
            // rows -> NCHW, 3x3/s2 conv, back to rows
            let nchw = scope.alloc(bp.act_bytes(n1 * LQ_HIDDEN))?;
            let tu = scope.u32x4_uniform(1, n1, LQ_HIDDEN, 1)?;
            scope.transpose12::<Transpose12F32>(&fp.transpose, a.data, tu, nchw, n1 * LQ_HIDDEN)?;
            let conv = scope.alloc(bp.act_bytes(DIM * n2))?;
            let cu = conv2d_uniform(
                &scope,
                (1, LQ_HIDDEN, h1 as u32, w1 as u32),
                (DIM, h2 as u32, w2 as u32),
                (3, 3, 1, 1, 2, 2),
            )?;
            scope.conv2d(
                &fp.conv.pipeline,
                &fp.conv.op,
                nchw,
                scope.import_copy(c2w.buf()),
                scope.import_copy(c2b.buf()),
                cu,
                conv,
                DIM,
                n2,
                1,
            )?;
            let t2 = scope.alloc(bp.act_bytes(DIM * n2))?;
            let tu2 = scope.u32x4_uniform(1, DIM, n2, 1)?;
            scope.transpose12::<Transpose12F32>(&fp.transpose, conv, tu2, t2, DIM * n2)?;
            let b = rms_silu(
                &scope,
                bp,
                ActBuf::dense(t2),
                scope.import_copy(nm2.buf()),
                n2,
                DIM,
            )?;
            linear_bias(
                &scope,
                bp,
                b.data,
                &lin.bufs(),
                n2,
                DIM,
                DIM,
                scope.import_copy(out.as_buf_ref()),
            )?;
            scope.submit_void().await?;
        }
        let mut tokens = Vec::new();
        read_into_f32(
            &self.backend,
            &out.as_buf_ref(),
            (n2 * DIM) as usize,
            fp.act,
            &mut tokens,
        )
        .await?;
        Ok(tokens)
    }

    pub async fn upscale(
        &self,
        inputs: &UpscaleInputs<'_>,
        mut diag: Option<&mut RelayVsrDiag>,
        cancel: crate::common::Cancel<'_>,
    ) -> Result<UpscaledVideo, RelayVsrError<S::Error>> {
        let (nf, h, w) = (inputs.num_frames, inputs.height, inputs.width);
        if nf == 0 || inputs.frames.len() != nf * h * w * 3 {
            return Err(RelayVsrError::Input(format!(
                "{} bytes for {nf}x{h}x{w} RGB",
                inputs.frames.len()
            )));
        }
        let up = self.up;
        let ls = 16 / up;
        let kfs = keyframes(nf);
        let (lh, lw) = latent_size(h, w, ls);
        let (hp, wp) = (lh * ls, lw * ls);
        let (gh, gw) = (lh / 2, lw / 2);
        let workspace = Workspace::new(
            Arc::clone(&self.backend),
            Arc::clone(self.residency.arbiter()),
        );
        let bp = &self.dit_pipelines.block;
        tracing::info!(
            target: thinfer_core::trace::DIAG,
            frames = nf, height = h, width = w, up, keyframes = kfs.len(), latent_h = lh, latent_w = lw,
            "relayvsr start",
        );

        // --- 1. keyframe LR projection -> token_add ---
        let mut phase_ms = Vec::new();
        let mut t_phase = std::time::Instant::now();
        let backend = &self.backend;
        let mut lap = |name: &'static str, phase_ms: &mut Vec<(&'static str, f64)>| {
            phase_ms.push((name, t_phase.elapsed().as_secs_f64() * 1e3));
            t_phase = std::time::Instant::now();
            tracing::info!(
                target: thinfer_core::trace::DIAG,
                phase = name,
                mem = ?thinfer_core::backend::Backend::mem_account(&**backend).snapshot(),
                "relayvsr phase done",
            );
        };
        let mut lq = Vec::with_capacity(kfs.len() * gh * gw * DIM as usize);
        {
            let _s = trace::scope!("relayvsr.lq_proj").entered();
            for &k in &kfs {
                let lr = Self::lr_frame(inputs, k, hp, wp);
                lq.extend(self.lq_tokens(&workspace, &lr, hp, wp).await?);
            }
        }
        let lq_bytes = act_upload_bytes(bp.act_dtype, &lq);
        let token_add = workspace.alloc(lq_bytes.len() as u64)?;
        self.backend.write_buffer(token_add.id(), 0, &lq_bytes)?;
        if let Some(d) = diag.as_deref_mut() {
            d.lq_tokens = lq;
        }
        if cancel.is_some_and(|c| c()) {
            return Err(RelayVsrError::Cancelled);
        }

        // --- 2. one keyframe forward: latent = noise - v ---
        let ch = self.cfg.in_channels;
        let n_lat = ch * kfs.len() * lh * lw;
        let noise = match diag.as_deref_mut().and_then(|d| d.noise_override.take()) {
            Some(n) if n.len() == n_lat => n,
            Some(n) => {
                return Err(RelayVsrError::Input(format!(
                    "noise override {} != {n_lat}",
                    n.len()
                )));
            }
            None => gaussian_noise(n_lat, inputs.seed),
        };
        let shape = WanDitShape::new(ch, kfs.len(), lh, lw, dit_config::TEXT_SEQ);
        let dit = WanDit::assemble(self.dit_handles.clone(), shape, self.cfg);
        lap("lq_proj", &mut phase_ms);
        let v = {
            let _s = trace::scope!("relayvsr.keyframes").entered();
            // The untiled block path: at f16 acts each block is one deferred
            // accelerator sequence (no per-op drains).
            dit.forward_many(
                &self.backend,
                &self.dit_pipelines,
                &self.residency,
                &workspace,
                &[WanDitInputs {
                    image: &noise,
                    img_ref: None,
                    text: &self.context,
                    r_timestep: None,
                    timestep: TIMESTEP,
                    attn_window: None,
                    token_add: Some(token_add.as_buf_ref()),
                }],
            )
            .await?
            .swap_remove(0)
            .image
        };
        lap("keyframes", &mut phase_ms);
        let latent: Vec<f32> = noise.iter().zip(&v).map(|(n, v)| n - v).collect();
        if let Some(d) = diag.as_deref_mut() {
            d.velocity = v;
            d.kf_latent = latent.clone();
        }
        drop(token_add);
        self.residency.evict_all_and_free(&*self.backend);
        workspace.drain_pool();
        if cancel.is_some_and(|c| c()) {
            return Err(RelayVsrError::Cancelled);
        }

        // --- 3. Flash decode: every keyframe window in lockstep ---
        // Windows share nothing but the weights (history resets per window),
        // so step `j` decodes frame `first + j` of every window as one batch.
        // Only the last window can be shorter, so the windows still running
        // at any step are a prefix of them.
        let _s = trace::scope!("relayvsr.flash").entered();
        let pairs: Vec<(usize, usize)> = if kfs.len() == 1 {
            vec![(kfs[0], kfs[0])]
        } else {
            kfs.windows(2).map(|p| (p[0], p[1])).collect()
        };
        let dec = FlashDecoder::new(
            &self.backend,
            &self.flash_pipelines,
            &workspace,
            &self.residency,
            &self.flash,
            lh as u32,
            lw as u32,
            pairs.len().max(kfs.len()) as u32,
        )
        .await?;
        let plane = lh * lw;
        let kf_latent = |i: usize| -> Vec<f32> {
            let mut out = vec![0f32; ch * plane];
            for c in 0..ch {
                out[c * plane..][..plane]
                    .copy_from_slice(&latent[(c * kfs.len() + i) * plane..][..plane]);
            }
            out
        };
        let (oh, ow) = (h * up, w * up);
        let mut frames = thinfer_core::secmem::Locked::<u8>::zeroed(nf * oh * ow * 3);
        let conds = dec
            .conditions(&(0..kfs.len()).map(kf_latent).collect::<Vec<_>>())
            .await?;
        // A step's composes (bicubic LR + pixel-shuffled detail, CPU, one
        // worker per frame) run while the GPU decodes the next step; they land
        // once that step is back. `gpu`/`wait`: decode vs compose-wait wall time.
        let detail = inputs.detail;
        type Composing = Vec<(usize, std::thread::JoinHandle<Vec<u8>>)>;
        let mut composing: Composing = Vec::new();
        let (mut gpu, mut wait) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
        let mut land = |frames: &mut [u8], c: Composing| {
            let t = std::time::Instant::now();
            for (fr, worker) in c {
                let img = worker
                    .join()
                    .unwrap_or_else(|p| std::panic::resume_unwind(p));
                frames[fr * oh * ow * 3..][..oh * ow * 3].copy_from_slice(&img);
            }
            wait += t.elapsed();
        };
        let steps = pairs.iter().map(|&(f, l)| l - f + 1).max().unwrap_or(0);
        let mut history = Vec::new();
        let head_len = plane * relay_flash::HEAD_OUT as usize;
        for j in 0..steps {
            if cancel.is_some_and(|c| c()) {
                land(&mut frames, std::mem::take(&mut composing));
                return Err(RelayVsrError::Cancelled);
            }
            let b = pairs.iter().take_while(|&&(f, l)| f + j <= l).count();
            debug_assert!(pairs[b..].iter().all(|&(f, l)| f + j > l));
            let lrs: Vec<Vec<f32>> = pairs[..b]
                .iter()
                .map(|&(f, _)| Self::lr_frame(inputs, f + j, hp, wp))
                .collect();
            let t = std::time::Instant::now();
            let head = dec
                .frames(
                    &lrs.concat(),
                    b as u32,
                    hp as u32,
                    wp as u32,
                    &conds,
                    &mut history,
                )
                .await;
            gpu += t.elapsed();
            land(&mut frames, std::mem::take(&mut composing));
            let head = Arc::new(head?);
            // A window's first frame is the previous window's last.
            for (wi, lr) in lrs.into_iter().enumerate() {
                if wi > 0 && j == 0 {
                    continue;
                }
                let head = Arc::clone(&head);
                composing.push((
                    pairs[wi].0 + j,
                    std::thread::spawn(move || {
                        relay_flash::compose_frame(
                            &head[wi * head_len..][..head_len],
                            lh,
                            lw,
                            &lr,
                            hp,
                            wp,
                            up,
                            oh,
                            ow,
                            detail,
                        )
                    }),
                ));
            }
        }
        land(&mut frames, std::mem::take(&mut composing));
        tracing::debug!(
            target: thinfer_core::trace::DIAG,
            gpu_ms = gpu.as_millis() as u64,
            compose_wait_ms = wait.as_millis() as u64,
            "relayvsr flash"
        );
        drop(dec);
        lap("flash", &mut phase_ms);
        for (name, ms) in &phase_ms {
            tracing::info!(target: thinfer_core::trace::DIAG, phase = name, ms = *ms as u64, "relayvsr phase");
        }
        if let Some(d) = diag {
            d.phase_ms = phase_ms;
        }
        self.residency.evict_all_and_free(&*self.backend);
        Ok(UpscaledVideo {
            frames,
            num_frames: nf,
            height: oh,
            width: ow,
        })
    }
}

fn rms_silu<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    bp: &crate::common::block::BlockPipelines,
    x: ActBuf<'w>,
    w: thinfer_core::workspace::BatchBuf<'w>,
    rows: u32,
    dim: u32,
) -> Result<ActBuf<'w>, WgpuError> {
    let n = alloc_act(scope, bp, rows, dim)?;
    op_rmsnorm(scope, bp, x, w, n, rows, dim, LQ_EPS)?;
    let s = alloc_act(scope, bp, rows, dim)?;
    scope.dispatch_op::<SiluF32>(&bp.silu, &[n.data], s.data)?;
    Ok(s)
}

/// Read a whole bf16 tensor from the residency's source as f32.
async fn read_bf16<S: WeightSource>(
    residency: &WeightResidency<S>,
    name: &str,
) -> Result<Vec<f32>, String> {
    let id = wid(name);
    let mut r = residency
        .source()
        .open(&id)
        .await
        .map_err(|e| format!("{e:?}"))?;
    let mut bytes = vec![0u8; r.len() as usize];
    r.read_at(0, &mut bytes)
        .await
        .map_err(|e| format!("{e:?}"))?;
    Ok(bytes
        .chunks_exact(2)
        .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect())
}

/// torch `F.interpolate(mode="bilinear", align_corners=False)` of NCHW
/// `[c, h, w]` by an integer factor.
pub fn bilinear_up(x: &[f32], c: usize, h: usize, w: usize, up: usize) -> Vec<f32> {
    let (oh, ow) = (h * up, w * up);
    let tap = |o: usize, n: usize| -> (usize, usize, f32) {
        let src = ((o as f32 + 0.5) / up as f32 - 0.5).max(0.0);
        let i0 = (src.floor() as usize).min(n - 1);
        let i1 = (i0 + 1).min(n - 1);
        (i0, i1, src - i0 as f32)
    };
    let ys: Vec<_> = (0..oh).map(|y| tap(y, h)).collect();
    let xs: Vec<_> = (0..ow).map(|x| tap(x, w)).collect();
    let mut out = vec![0f32; c * oh * ow];
    for ch in 0..c {
        let p = &x[ch * h * w..][..h * w];
        for (y, &(y0, y1, fy)) in ys.iter().enumerate() {
            for (xo, &(x0, x1, fx)) in xs.iter().enumerate() {
                let top = p[y0 * w + x0] * (1.0 - fx) + p[y0 * w + x1] * fx;
                let bot = p[y1 * w + x0] * (1.0 - fx) + p[y1 * w + x1] * fx;
                out[(ch * oh + y) * ow + xo] = top * (1.0 - fy) + bot * fy;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyframe_schedule_matches_reference() {
        assert_eq!(keyframes(1), vec![0]);
        assert_eq!(keyframes(31), vec![0, 15, 30]);
        assert_eq!(keyframes(121), vec![0, 15, 30, 45, 60, 75, 90, 105, 120]);
        assert_eq!(
            keyframes(124),
            vec![0, 15, 30, 45, 60, 75, 90, 105, 120, 123]
        );
    }

    #[test]
    fn latent_grid_rounds_to_whole_patches() {
        assert_eq!(latent_size(480, 832, 8), (60, 104));
        assert_eq!(latent_size(481, 832, 8), (62, 104));
        assert_eq!(latent_size(480, 832, 4), (120, 208));
    }
}

/// `Comfy-Org/RelayVSR` files by upscale factor.
pub mod manifest {
    use thinfer_core::manifest::{FileRef, ModelManifest};

    const REPO: &str = "Comfy-Org/RelayVSR";

    pub mod role {
        /// x2 keyframe DiT: LightX2V's retrain for the step after H3
        /// generation (int8 convrot, 4.8G).
        pub const DIT_X2: &str = "dit/x2";
        pub const FLASH_X2: &str = "flash/x2";
        pub const DIT_X4: &str = "dit/x4";
        pub const FLASH_X4: &str = "flash/x4";
    }

    pub static MANIFEST: ModelManifest = ModelManifest {
        id: "relayvsr",
        files: &[
            (
                role::DIT_X2,
                FileRef::new(
                    REPO,
                    "diffusion_models/relayvsr_x2_lightx2v_int8_convrot.safetensors",
                ),
            ),
            (
                role::FLASH_X2,
                FileRef::new(REPO, "vae/relayvsr_flash_vae_x2_fp32.safetensors"),
            ),
            (
                role::DIT_X4,
                FileRef::new(
                    REPO,
                    "diffusion_models/relayvsr_x4_int8_convrot.safetensors",
                ),
            ),
            (
                role::FLASH_X4,
                FileRef::new(REPO, "vae/relayvsr_flash_vae_x4_fp32.safetensors"),
            ),
        ],
    };

    /// `(dit, flash)` roles for an upscale factor (2 or 4).
    pub fn roles(up: usize) -> (&'static str, &'static str) {
        if up == 4 {
            (role::DIT_X4, role::FLASH_X4)
        } else {
            (role::DIT_X2, role::FLASH_X2)
        }
    }
}
