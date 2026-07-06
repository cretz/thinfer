//! DreamID-V-Wan-1.3B-Faster pipeline: diffusion video face swap.
//!
//! A custom orchestrator (like `krea` / `qwen_image`) reusing the shared Wan ops
//! rather than `WanModel::generate`: there is NO live text encoder (the "chang
//! face" umT5 context is baked) and the denoise is image-CFG with two DiT
//! forwards per step. It wires the EXISTING pieces:
//!   - `WanVaeEncoder` / `WanVaeDecoder` (Wan2.1 z16 VAE, `wan/vae.rs`),
//!   - `WanDit` extended with the `ref_conv` prefix + 48-ch patch input
//!     (`wan/dit.rs`, gated on `cfg.ref_conv`),
//!   - `FlowUniPc` (`wan/unipc.rs`, shift 5).
//!
//! Pipeline (`dreamidv_wan_faster/wan_swapface.py::DreamIDV.generate`):
//!   1. VAE-encode the target video (Normalize 0.5), the face-mask video (NO
//!      normalize), and the white-padded source image (Normalize 0.5) into z16
//!      latents (mode + `(z - latents_mean) / latents_std`, matching the
//!      reference `WanVAE.encode`).
//!   2. `y = channel-cat(video_latent, mask_latent)` (32 ch). Per step the 48-ch
//!      DiT input is `cat(noise[16], y[32])`; `img_ref` is the image latent.
//!   3. Denoise (UniPC shift 5): `pos_tiv = fwd(img_ref = face)`, `pos_tv =
//!      fwd(img_ref = zeros)`, `noise_pred = pos_tiv + s * (pos_tiv - pos_tv)`.
//!   4. VAE-decode the clean latent into RGB in `[-1, 1]`.
//!
//! Preprocessing mirrors the reference: `NaResize` to `sqrt(W*H)` area
//! (downsample-only) + `DivisibleCrop` to multiples of `(vae_stride * patch) =
//! (16, 16)`, layout `[c, t, h, w]`. Resampling here is bilinear; the torch
//! reference uses bicubic (image) / bilinear plus LANCZOS pre-fit (source face),
//! so exact resampling is a PARITY concern the torch gate drives, not this
//! health path.

use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::ops::{ActDtype, WeightDtype};
use thinfer_core::trace;
use thinfer_core::residency::WeightResidency;
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::Workspace;

use crate::wan::dit::{WanDit, WanDitError, WanDitInputs, WanDitShape, WanDitTaps};
use crate::wan::dit_block::{WanDitConfig, WanDitPipelines, config as dit_config};
use crate::wan::loader::{WanI8Sites, register_wan_dit_handles};
use crate::common::block::DenseActSites;
use crate::wan::pipeline::{SiteOverride, WanVariant, block_cfgs, gaussian_noise};
use crate::wan::unipc::{FlowUniPc, UniPcConfig};
use crate::wan::vae::{
    VaeDecoderWeights, VaeEncoderWeights, WanVaeConfig, WanVaeDecodeError, WanVaeDecoder,
    WanVaeEncodeError, WanVaeEncoder, WanVaePipelines, register_decoder, register_encoder,
};

/// Baked-context text length the DiT cross-attends to (umT5 rows, zero-padded).
const TEXT_SEQ: usize = dit_config::TEXT_SEQ;
/// `vae_stride[1..] * patch_size[1..]` = the spatial grid the input must be a
/// multiple of (8 spatial VAE stride * 2 patch = 16 per axis).
const CROP_FACTOR: usize = 16;
/// Image-CFG guidance scale on the source-face reference (`guide_scale_img`).
pub const DEFAULT_GUIDE_SCALE: f32 = 4.0;
/// Faster variant step count.
pub const DEFAULT_STEPS: u32 = 16;

/// Rows of the baked "chang face" umT5 context (`[CONTEXT_ROWS, TEXT_DIM]`).
const CONTEXT_ROWS: usize = 4;
/// The baked context embedded as raw bf16 `[CONTEXT_ROWS, TEXT_DIM]`
/// (little-endian), extracted from the reference `context.pth`. It is a fixed
/// model constant that the DreamID-V HF *weights* repo does not ship (it lives in
/// the code repo), so it rides in-tree as this ~32 KiB asset rather than as a
/// download. No prompt / text encoder is involved at inference.
static CONTEXT_BF16: &[u8] = include_bytes!("dreamidv_context.bf16");

/// Decode the baked umT5 context to host f32 rows `([CONTEXT_ROWS * TEXT_DIM],
/// CONTEXT_ROWS)`, ready for [`DreamIdvPipeline::load`].
pub fn baked_context() -> (Vec<f32>, usize) {
    debug_assert_eq!(CONTEXT_BF16.len(), CONTEXT_ROWS * dit_config::TEXT_DIM * 2);
    let flat = CONTEXT_BF16
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect();
    (flat, CONTEXT_ROWS)
}

#[derive(Debug)]
pub enum DreamIdvError<SE: core::fmt::Debug> {
    Load(crate::wan::loader::LoadError),
    VaeLoad(crate::common::loader::LoadError),
    Wgpu(thinfer_core::backend::WgpuError),
    Dit(WanDitError<SE>),
    Encode(WanVaeEncodeError<SE>),
    Decode(WanVaeDecodeError<SE>),
    /// A preprocessed input collapsed to zero size (all frames cropped away, or a
    /// resolution below one 16x16 grid cell).
    DegenerateInput(String),
}

impl<SE: core::fmt::Debug> From<crate::wan::loader::LoadError> for DreamIdvError<SE> {
    fn from(e: crate::wan::loader::LoadError) -> Self {
        Self::Load(e)
    }
}
impl<SE: core::fmt::Debug> From<crate::common::loader::LoadError> for DreamIdvError<SE> {
    fn from(e: crate::common::loader::LoadError) -> Self {
        Self::VaeLoad(e)
    }
}
impl<SE: core::fmt::Debug> From<thinfer_core::backend::WgpuError> for DreamIdvError<SE> {
    fn from(e: thinfer_core::backend::WgpuError) -> Self {
        Self::Wgpu(e)
    }
}
impl<SE: core::fmt::Debug> From<WanDitError<SE>> for DreamIdvError<SE> {
    fn from(e: WanDitError<SE>) -> Self {
        Self::Dit(e)
    }
}
impl<SE: core::fmt::Debug> From<WanVaeEncodeError<SE>> for DreamIdvError<SE> {
    fn from(e: WanVaeEncodeError<SE>) -> Self {
        Self::Encode(e)
    }
}
impl<SE: core::fmt::Debug> From<WanVaeDecodeError<SE>> for DreamIdvError<SE> {
    fn from(e: WanVaeDecodeError<SE>) -> Self {
        Self::Decode(e)
    }
}

/// One raw RGB media input: interleaved `u8` `[frames, h, w, 3]` (a single image
/// is `frames == 1`).
pub struct RgbFrames<'a> {
    pub data: &'a [u8],
    pub frames: usize,
    pub h: usize,
    pub w: usize,
}

/// Generate inputs. `video` and `mask` must share frame count and dims; `frames`
/// must be `4k + 1` (truncate before calling).
pub struct DreamIdvInputs<'a> {
    pub video: RgbFrames<'a>,
    pub mask: RgbFrames<'a>,
    pub image: RgbFrames<'a>,
    /// `sqrt(W * H)` target-area resolution the `NaResize` downsamples toward
    /// (the reference `size = (W, H)`; area = `W * H`).
    pub target_area: f64,
    pub steps: u32,
    pub guide_scale: f32,
    pub seed: u64,
}

/// Decoded output: RGB CTHW `[3, frames, height, width]` in `[-1, 1]`.
pub struct DreamIdvVideo {
    pub frames: Vec<f32>,
    pub num_frames: usize,
    pub height: usize,
    pub width: usize,
}

/// Optional intermediate captures for the health gate (finiteness + non-constant
/// checks). All host CTHW f32.
#[derive(Default)]
pub struct DreamIdvDiag {
    /// z16 video latent `[16, F_lat, h, w]`.
    pub video_latent: Vec<f32>,
    /// z16 mask latent `[16, F_lat, h, w]`.
    pub mask_latent: Vec<f32>,
    /// z16 source-face latent `[16, 1, h, w]`.
    pub image_latent: Vec<f32>,
    /// The clean pre-VAE denoised latent `[16, F_lat, h, w]`.
    pub denoised_latent: Vec<f32>,
}

/// DreamID-V pipeline over a residency that unions the renamed DiT safetensors
/// and the Wan2.1 VAE (see `wan::source::open_dreamidv_source`).
pub struct DreamIdvPipeline<S: WeightSource> {
    backend: Arc<WgpuBackend>,
    residency: WeightResidency<S>,
    cfg: WanDitConfig,
    dit_handles: crate::wan::dit::LoadedWanDitHandles,
    dit_pipelines: WanDitPipelines,
    vae_encoder: WanVaeEncoder,
    vae_decoder: WanVaeDecoder,
    /// Baked "chang face" umT5 context, zero-padded to `[TEXT_SEQ, TEXT_DIM]`.
    context: Vec<f32>,
}

impl<S: WeightSource> DreamIdvPipeline<S> {
    /// Build the pipeline. `residency` must already register from a source that
    /// supplies the DiT (canonical names + `ref_conv` / `patch_embedding`) and the
    /// Wan2.1 VAE. `context` is the baked umT5 embedding `[L, TEXT_DIM]` (any
    /// `L <= TEXT_SEQ`); it is zero-padded to `[TEXT_SEQ, TEXT_DIM]`.
    pub async fn load(
        backend: Arc<WgpuBackend>,
        residency: WeightResidency<S>,
        context: &[f32],
        context_rows: usize,
        i8_matmul: bool,
    ) -> Result<Self, DreamIdvError<S::Error>> {
        let variant = WanVariant::dreamid_v();
        let cfg = variant.dit;
        let vae_cfg = variant.vae.clone();

        // DP4A on the DP4A-safe normed matmul sites (self-attn qkv + ffn_up): those
        // A-sides are norm-conditioned (no massive outliers), so Q8_0 weights + the
        // i8 `matmul_i8` path cut their matmul ~6x while holding parity (the same
        // default the main Wan path uses). Cross-attn qkv + proj + ffn_down stay
        // bf16 dense (proj/ffn_down ride coopmat). Cross-attn here reads the tiny
        // baked context (not un-normed umT5), so it is not outlier-heavy. Opt-out
        // (`i8_matmul = false`) forces the whole DiT to the bf16 reference path.
        let q8 = i8_matmul.then_some(thinfer_core::quant::QuantKind::Q8_0);
        let i8_sites = WanI8Sites {
            qkv_self: q8,
            ffn_up: q8,
        };
        let dit_handles =
            register_wan_dit_handles(&residency, &cfg, "", variant.block_transcode, i8_sites)?;
        // Acts F16 when the device supports shader-f16 (the 1536-dim residual fits
        // f16), else F32. F32 `.pth` narrows to bf16 at upload for the dense sites.
        let act = variant.act_pref.unwrap_or(if backend.supports_shader_f16() {
            ActDtype::F16
        } else {
            ActDtype::F32
        });
        let dit_w = variant
            .block_transcode
            .map(WeightDtype::Quant)
            .unwrap_or(WeightDtype::Bf16);
        // `fast_sdpa = true` engages the tensor-core (coopmat) proj/ffn_down matmuls
        // + the f16 SDPA fast path -- the sanctioned Wan DiT hot-path lever (the main
        // `WanModel` path always enables it). The compile gate falls back on devices
        // without a coopmat config / subgroups, and `THINFER_NO_COOPMAT` disables it.
        let dit_pipelines = WanDitPipelines::compile(
            &backend,
            &block_cfgs(
                dit_w,
                act,
                SiteOverride {
                    qkv_self: i8_sites.qkv_self.map(WeightDtype::Quant),
                    ffn_up: i8_sites.ffn_up.map(WeightDtype::Quant),
                    dense_acts: DenseActSites::default(),
                },
                true,
                false,
            ),
        )
        .await?;

        // VAE encoder + decoder share one residency + config.
        let before_enc = residency.total_registered_bytes();
        let enc_handles = register_encoder(&residency, &VaeEncoderWeights::new(&vae_cfg))?;
        let enc_footprint = residency.total_registered_bytes() - before_enc;
        let before_dec = residency.total_registered_bytes();
        let dec_handles = register_decoder(&residency, &VaeDecoderWeights::new(&vae_cfg))?;
        let dec_footprint = residency.total_registered_bytes() - before_dec;
        let vae_encoder = WanVaeEncoder {
            pipelines: WanVaePipelines::compile(&backend).await?,
            handles: enc_handles,
            cfg: vae_cfg.clone(),
            weight_footprint: enc_footprint,
        };
        let vae_decoder = WanVaeDecoder {
            pipelines: WanVaePipelines::compile(&backend).await?,
            handles: dec_handles,
            cfg: vae_cfg,
            weight_footprint: dec_footprint,
        };

        // Zero-pad the baked context to [TEXT_SEQ, TEXT_DIM].
        let td = dit_config::TEXT_DIM;
        assert!(
            context_rows <= TEXT_SEQ && context.len() == context_rows * td,
            "context must be [L<=512, {td}]; got {} rows, {} elems",
            context_rows,
            context.len()
        );
        let mut ctx = vec![0.0_f32; TEXT_SEQ * td];
        ctx[..context.len()].copy_from_slice(context);

        Ok(Self {
            backend,
            residency,
            cfg,
            dit_handles,
            dit_pipelines,
            vae_encoder,
            vae_decoder,
            context: ctx,
        })
    }

    /// Run the full face-swap pipeline. Returns the decoded RGB video; captures
    /// intermediates into `diag` when provided.
    pub async fn generate(
        &self,
        inputs: &DreamIdvInputs<'_>,
        mut diag: Option<&mut DreamIdvDiag>,
    ) -> Result<DreamIdvVideo, DreamIdvError<S::Error>> {
        let mut workspace = Workspace::new(
            Arc::clone(&self.backend),
            Arc::clone(self.residency.arbiter()),
        );

        // --- 1. preprocess + VAE-encode the three inputs ---
        // Target dims come from the VIDEO (NaResize sqrt-area + DivisibleCrop);
        // the mask reuses them, the source image is white-padded to the video
        // aspect so all three share the latent grid.
        let (video_rgb, ph, pw) = preprocess(&inputs.video, inputs.target_area, false)?;
        let vf = inputs.video.frames;
        let (mask_rgb, mh, mw) = preprocess(&inputs.mask, inputs.target_area, true)?;
        if (mh, mw) != (ph, pw) || inputs.mask.frames != vf {
            return Err(DreamIdvError::DegenerateInput(format!(
                "mask geometry {mh}x{mw}x{} != video {ph}x{pw}x{vf}",
                inputs.mask.frames
            )));
        }
        // Source image: white-pad to the video aspect, then the same
        // NaResize/DivisibleCrop/Normalize as the video.
        let padded = white_pad_to_aspect(&inputs.image, pw, ph);
        let (image_rgb, ih, iw) = preprocess(
            &RgbFrames {
                data: &padded.data,
                frames: 1,
                h: padded.h,
                w: padded.w,
            },
            inputs.target_area,
            false,
        )?;
        if (ih, iw) != (ph, pw) {
            return Err(DreamIdvError::DegenerateInput(format!(
                "image geometry {ih}x{iw} != video {ph}x{pw}"
            )));
        }

        let (video_lat, mask_lat, image_lat) = {
            let _e = trace::scope!("vae_encode").entered();
            let video_lat = self
                .encode_z16(&mut workspace, &video_rgb, vf, ph, pw)
                .await?;
            let mask_lat = self
                .encode_z16(&mut workspace, &mask_rgb, vf, ph, pw)
                .await?;
            let image_lat = self
                .encode_z16(&mut workspace, &image_rgb, 1, ph, pw)
                .await?;
            (video_lat, mask_lat, image_lat)
        };

        let z = self.vae_encoder.cfg.z_dim; // 16
        let f_lat = (vf - 1) / self.vae_encoder.cfg.temporal_compression + 1;
        let h_lat = ph / self.vae_encoder.cfg.spatial_compression;
        let w_lat = pw / self.vae_encoder.cfg.spatial_compression;
        debug_assert_eq!(video_lat.len(), z * f_lat * h_lat * w_lat);
        debug_assert_eq!(image_lat.len(), z * h_lat * w_lat);

        if let Some(d) = diag.as_deref_mut() {
            d.video_latent = video_lat.clone();
            d.mask_latent = mask_lat.clone();
            d.image_latent = image_lat.clone();
        }
        self.residency.evict_all_and_free(&*self.backend);
        workspace.drain_pool();

        // --- 2. y = cat(video, mask) [32 ch]; assemble the DiT ---
        let hw = h_lat * w_lat;
        let plane = f_lat * hw;
        let mut y = vec![0.0_f32; 2 * z * plane]; // [32, F_lat, h, w]
        y[..z * plane].copy_from_slice(&video_lat);
        y[z * plane..].copy_from_slice(&mask_lat);

        let shape = WanDitShape::new(self.cfg.in_channels, f_lat, h_lat, w_lat, TEXT_SEQ);
        let dit = WanDit::assemble(self.dit_handles.clone(), shape, self.cfg);
        let zeros_ref = vec![0.0_f32; image_lat.len()];

        // --- 3. image-CFG UniPC denoise ---
        let mut unipc = FlowUniPc::new(&UniPcConfig::dreamid_v(inputs.steps));
        let mut latents = gaussian_noise(z * plane, inputs.seed); // z16 noise
        let scale = inputs.guide_scale;
        {
            let _denoise = trace::scope!("denoise").entered();
            for i in 0..unipc.num_steps() {
                let _step = trace::scope!("step", i = i).entered();
                let t = unipc.timestep(i);
                // 48-ch input = cat(noise[16], y[32]).
                let mut x_in = vec![0.0_f32; 3 * z * plane];
                x_in[..z * plane].copy_from_slice(&latents);
                x_in[z * plane..].copy_from_slice(&y);

                let out_tiv = self
                    .forward(&dit, &workspace, &x_in, Some(&image_lat), t)
                    .await?;
                let out_tv = self
                    .forward(&dit, &workspace, &x_in, Some(&zeros_ref), t)
                    .await?;
                // noise_pred = pos_tiv + scale * (pos_tiv - pos_tv).
                let noise_pred: Vec<f32> = out_tiv
                    .iter()
                    .zip(&out_tv)
                    .map(|(&a, &b)| a + scale * (a - b))
                    .collect();
                latents = unipc.step(&noise_pred, &latents);
            }
        }

        if let Some(d) = diag {
            d.denoised_latent = latents.clone();
        }
        self.residency.evict_all_and_free(&*self.backend);
        workspace.drain_pool();

        // --- 4. VAE-decode -> RGB [-1, 1] ---
        let rgb = self
            .vae_decoder
            .decode(
                &self.backend,
                &self.residency,
                &mut workspace,
                &latents,
                f_lat,
                h_lat,
                w_lat,
            )
            .await?;
        let out_frames = self.vae_decoder.cfg.temporal_compression * f_lat
            - (self.vae_decoder.cfg.temporal_compression - 1);
        Ok(DreamIdvVideo {
            frames: rgb,
            num_frames: out_frames,
            height: ph,
            width: pw,
        })
    }

    /// One DiT forward returning the velocity latent `[16, F_lat, h, w]`.
    async fn forward(
        &self,
        dit: &WanDit,
        workspace: &Workspace<WgpuBackend>,
        image: &[f32],
        img_ref: Option<&[f32]>,
        timestep: f32,
    ) -> Result<Vec<f32>, DreamIdvError<S::Error>> {
        let inputs = WanDitInputs {
            image,
            img_ref,
            text: &self.context,
            timestep,
            r_timestep: None,
            attn_window: None,
        };
        let out = dit
            .forward(
                &self.backend,
                &self.dit_pipelines,
                &self.residency,
                workspace,
                &inputs,
            )
            .await?;
        Ok(out.image)
    }

    /// Parity/diag accessor (no forward-logic change): run ONE DiT forward on
    /// caller-supplied latents and return the velocity `[out_channels, F_lat, h,
    /// w]` plus, per block, the token-space residual stream `[rows, inner]` (ref
    /// prefix first). `image` is the 48-ch input `[in_channels, F_lat, h, w]` laid
    /// out EXACTLY as [`Self::generate`] builds `x_in` (noise[16] ++ video[16] ++
    /// mask[16]); `img_ref` is the z16 source-face latent `[out_channels, 1, h,
    /// w]`; the baked context is the pipeline's own padded `self.context`. Drives
    /// the torch Stage-A parity gate (`dreamidv_dit_parity`), mirroring the
    /// LongLive `forward_velocity_at` accessor. Reuses `forward_with_taps`.
    pub async fn dit_forward_parity(
        &self,
        f_lat: usize,
        h_lat: usize,
        w_lat: usize,
        image: &[f32],
        img_ref: &[f32],
        timestep: f32,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>), DreamIdvError<S::Error>> {
        let workspace = Workspace::new(
            Arc::clone(&self.backend),
            Arc::clone(self.residency.arbiter()),
        );
        let shape = WanDitShape::new(self.cfg.in_channels, f_lat, h_lat, w_lat, TEXT_SEQ);
        let dit = WanDit::assemble(self.dit_handles.clone(), shape, self.cfg);
        let inputs = WanDitInputs {
            image,
            img_ref: Some(img_ref),
            text: &self.context,
            timestep,
            r_timestep: None,
            attn_window: None,
        };
        let mut per_block: Vec<Vec<f32>> = Vec::new();
        let taps = WanDitTaps {
            per_block: Some(&mut per_block),
            ..Default::default()
        };
        let out = dit
            .forward_with_taps(
                &self.backend,
                &self.dit_pipelines,
                &self.residency,
                &workspace,
                &inputs,
                taps,
                None,
            )
            .await?;
        Ok((out.image, per_block))
    }

    /// Parity accessor (Stage B): VAE-encode a caller-supplied preprocessed CTHW
    /// RGB clip `[3, f, h, w]` (values in `[-1, 1]`) into the normalized z16
    /// latent `[16, F_lat, h/8, w/8]` (mode `mu` then `(mu - mean) / std`),
    /// matching the reference `WanVAE.encode`. Drives `dreamidv_vae_parity`.
    pub async fn vae_encode_parity(
        &self,
        rgb: &[f32],
        f: usize,
        h: usize,
        w: usize,
    ) -> Result<Vec<f32>, DreamIdvError<S::Error>> {
        let mut workspace = Workspace::new(
            Arc::clone(&self.backend),
            Arc::clone(self.residency.arbiter()),
        );
        self.encode_z16(&mut workspace, rgb, f, h, w).await
    }

    /// VAE-encode a preprocessed CTHW RGB clip `[3, f, h, w]` (`[-1, 1]` or
    /// `[0, 1]` for the mask) into the normalized z16 latent `[16, F_lat, h/8,
    /// w/8]` (mode + `(z - mean) / std`).
    async fn encode_z16(
        &self,
        workspace: &mut Workspace<WgpuBackend>,
        rgb: &[f32],
        f: usize,
        h: usize,
        w: usize,
    ) -> Result<Vec<f32>, DreamIdvError<S::Error>> {
        let moments = self
            .vae_encoder
            .encode(&self.backend, &self.residency, workspace, rgb, f, h, w)
            .await?;
        Ok(normalize_moments(&moments, &self.vae_encoder.cfg))
    }
}

// ---------------------------------------------------------------------------
// Host preprocessing (mirrors dreamidv_wan_faster/utils/na_resize.py)
// ---------------------------------------------------------------------------

/// Padded RGB output of [`white_pad_to_aspect`], interleaved `[h, w, 3]`.
struct PaddedRgb {
    data: Vec<u8>,
    h: usize,
    w: usize,
}

/// Fit the source image inside `(target_w, target_h)` preserving aspect, then
/// white-pad the border out to `(target_w, target_h)` (reference: LANCZOS resize
/// + `ImageOps.expand(fill=(255,255,255))`). Resampling is bilinear here.
fn white_pad_to_aspect(img: &RgbFrames<'_>, target_w: usize, target_h: usize) -> PaddedRgb {
    let (iw, ih) = (img.w as f64, img.h as f64);
    let (tw, th) = (target_w as f64, target_h as f64);
    let img_ratio = iw / ih;
    let target_ratio = tw / th;
    let (new_w, new_h) = if img_ratio > target_ratio {
        (target_w, ((target_w as f64) / img_ratio).round() as usize)
    } else {
        (((target_h as f64) * img_ratio).round() as usize, target_h)
    };
    let new_w = new_w.max(1);
    let new_h = new_h.max(1);
    let resized = resize_bilinear_u8(img.data, img.h, img.w, new_h, new_w);

    // Center-pad to (target_h, target_w) with white.
    let mut out = vec![255u8; target_h * target_w * 3];
    let off_x = (target_w - new_w) / 2;
    let off_y = (target_h - new_h) / 2;
    for y in 0..new_h {
        for x in 0..new_w {
            let src = (y * new_w + x) * 3;
            let dst = ((y + off_y) * target_w + (x + off_x)) * 3;
            out[dst..dst + 3].copy_from_slice(&resized[src..src + 3]);
        }
    }
    PaddedRgb {
        data: out,
        h: target_h,
        w: target_w,
    }
}

/// `to_tensor` + `NaResize(sqrt(area))` (downsample-only) + `DivisibleCrop(16)` +
/// optional `Normalize(0.5, 0.5)`. Returns CTHW `[3, frames, h', w']` f32 and the
/// cropped `(h', w')`. `no_normalize` keeps `[0, 1]` (the mask branch).
fn preprocess<SE: core::fmt::Debug>(
    frames: &RgbFrames<'_>,
    target_area: f64,
    no_normalize: bool,
) -> Result<(Vec<f32>, usize, usize), DreamIdvError<SE>> {
    // NaResize scale (geometric, downsample-only).
    let area = (frames.h * frames.w) as f64;
    let mut scale = (target_area / area).sqrt();
    if scale >= 1.0 {
        scale = 1.0; // downsample_only
    }
    let rh = ((frames.h as f64) * scale).round() as usize;
    let rw = ((frames.w as f64) * scale).round() as usize;
    // DivisibleCrop to multiples of CROP_FACTOR (center crop).
    let ch = rh - (rh % CROP_FACTOR);
    let cw = rw - (rw % CROP_FACTOR);
    if ch == 0 || cw == 0 {
        return Err(DreamIdvError::DegenerateInput(format!(
            "preprocess collapsed to {ch}x{cw} (resized {rh}x{rw}, factor {CROP_FACTOR})"
        )));
    }
    let crop_y = (rh - ch) / 2;
    let crop_x = (rw - cw) / 2;

    let plane = ch * cw;
    // Output CTHW: [3, frames, ch, cw].
    let mut out = vec![0.0_f32; 3 * frames.frames * plane];
    for f in 0..frames.frames {
        let src = &frames.data[f * frames.h * frames.w * 3..(f + 1) * frames.h * frames.w * 3];
        let resized = resize_bilinear_u8(src, frames.h, frames.w, rh, rw);
        for y in 0..ch {
            for x in 0..cw {
                let s = ((y + crop_y) * rw + (x + crop_x)) * 3;
                for c in 0..3 {
                    let mut v = resized[s + c] as f32 / 255.0;
                    if !no_normalize {
                        v = v * 2.0 - 1.0; // Normalize(0.5, 0.5)
                    }
                    // CTHW: channel-major, then frame, then h, w.
                    out[((c * frames.frames + f) * ch + y) * cw + x] = v;
                }
            }
        }
    }
    Ok((out, ch, cw))
}

/// Bilinear-resize an interleaved RGB `u8` image `[sh, sw, 3]` to `[dh, dw, 3]`.
fn resize_bilinear_u8(src: &[u8], sh: usize, sw: usize, dh: usize, dw: usize) -> Vec<u8> {
    if (sh, sw) == (dh, dw) {
        return src.to_vec();
    }
    let mut out = vec![0u8; dh * dw * 3];
    // Half-pixel-centered sampling (align_corners = False, as torchvision).
    let sx = sw as f32 / dw as f32;
    let sy = sh as f32 / dh as f32;
    for y in 0..dh {
        let fy = ((y as f32 + 0.5) * sy - 0.5).max(0.0);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let wy = fy - y0 as f32;
        for x in 0..dw {
            let fx = ((x as f32 + 0.5) * sx - 0.5).max(0.0);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let wx = fx - x0 as f32;
            for c in 0..3 {
                let p00 = src[(y0 * sw + x0) * 3 + c] as f32;
                let p01 = src[(y0 * sw + x1) * 3 + c] as f32;
                let p10 = src[(y1 * sw + x0) * 3 + c] as f32;
                let p11 = src[(y1 * sw + x1) * 3 + c] as f32;
                let top = p00 + (p01 - p00) * wx;
                let bot = p10 + (p11 - p10) * wx;
                let v = top + (bot - top) * wy;
                out[(y * dw + x) * 3 + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Take the VAE moments `[2*z, F_lat, h, w]` (mean ++ logvar), keep the mode
/// (mean = channels `0..z`), and normalize per channel `(m - latents_mean) /
/// latents_std`. Mirrors the reference `WanVAE.encode` (`(mu - mean) * 1/std`).
fn normalize_moments(moments: &[f32], cfg: &WanVaeConfig) -> Vec<f32> {
    let z = cfg.z_dim;
    let plane = moments.len() / (2 * z);
    debug_assert_eq!(moments.len(), 2 * z * plane);
    let mut out = vec![0.0_f32; z * plane];
    for c in 0..z {
        let mean = cfg.latents_mean[c];
        let std = cfg.latents_std[c];
        let src = &moments[c * plane..(c + 1) * plane]; // mode = mean channels
        let dst = &mut out[c * plane..(c + 1) * plane];
        for (o, &m) in dst.iter_mut().zip(src) {
            *o = (m - mean) / std;
        }
    }
    out
}
