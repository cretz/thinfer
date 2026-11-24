//! Load + generate for MOSS-SoundEffect v2.0 (see the module docs). One
//! residency serves all three weight files (DiT, text encoder, DAC), so one
//! budget covers them; the text encoder runs first and is evicted before the
//! DiT's 2 x `steps` forwards.

use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::ops::ActDtype;
use thinfer_core::residency::WeightResidency;
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::Workspace;
use zeroize::Zeroizing;

use super::dac::DacDecoder;
use super::text::TextEncoder;
use super::{
    DECODE_MARGIN, FULL_FRAMES, LATENT_CH, TEXT_LEN, decode_window, frames_for, loop_offset,
    roll_left, roll_right, samples_for, schedule, timestep,
};
use crate::common::bigvgan::BigVganPipelines;
use crate::llm::kernels::LlmPipelines;
use crate::wan::dit::{LoadedWanDitHandles, WanDit, WanDitInputs, WanDitShape};
use crate::wan::dit_block::{WanDitConfig, WanDitPipelines};
use crate::wan::loader::{WanI8Sites, register_wan_dit_handles};
use crate::wan::pipeline::{SiteOverride, block_cfgs, gaussian_noise};

#[derive(Debug)]
pub enum SfxError {
    Load(String),
    Text(String),
    Dit(String),
    Decode(String),
    Cancelled,
}

/// One generation.
pub struct SfxParams<'a> {
    /// Token ids of [`super::prompt_text`].
    pub text_ids: &'a [u32],
    /// Clip length (rounded to 0.1 s, at most 30).
    pub seconds: f32,
    pub steps: u32,
    pub guidance: f32,
    pub seed: u64,
    /// Seamless loop: the clip's end runs into its start.
    pub looped: bool,
    /// Initial latent override `[LATENT_CH, FULL_FRAMES]` (parity).
    pub noise: Option<&'a [f32]>,
}

/// Per-step hook: `(steps done, steps)`; `false` stops the run.
pub type StepHook<'a> = &'a mut dyn FnMut(u32, u32) -> bool;

pub struct SfxModel<S: WeightSource> {
    backend: Arc<WgpuBackend>,
    residency: WeightResidency<S>,
    cfg: WanDitConfig,
    dit: WanDitPipelines,
    dit_handles: LoadedWanDitHandles,
    text: TextEncoder,
    llm: LlmPipelines,
    dac: DacDecoder,
    bv: BigVganPipelines,
}

impl<S: WeightSource> SfxModel<S> {
    /// `residency` serves the DiT (diffusers names), the text encoder
    /// (`model.*`) and the weight-norm-folded DAC (`state_dict.*`).
    pub async fn load(
        backend: Arc<WgpuBackend>,
        residency: WeightResidency<S>,
    ) -> Result<Self, SfxError> {
        let cfg = WanDitConfig::moss_sfx();
        let dit_handles =
            register_wan_dit_handles(&residency, &cfg, "", None, WanI8Sites::default())
                .map_err(|e| SfxError::Load(format!("dit: {e:?}")))?;
        let act = if backend.supports_shader_f16() {
            ActDtype::F16
        } else {
            ActDtype::F32
        };
        // f32 checkpoint: every DiT matmul narrows to bf16 at upload.
        let dit = WanDitPipelines::compile(
            &backend,
            &block_cfgs(
                thinfer_core::ops::WeightDtype::Bf16,
                act,
                SiteOverride::default(),
                true,
                false,
            ),
            false,
            false,
        )
        .await
        .map_err(|e| SfxError::Load(format!("dit pipelines: {e:?}")))?;
        let llm = LlmPipelines::compile(&backend)
            .await
            .map_err(|e| SfxError::Load(format!("llm pipelines: {e:?}")))?;
        let text = TextEncoder::load(&backend, &residency)
            .await
            .map_err(|e| SfxError::Load(format!("text encoder: {e:?}")))?;
        let dac = DacDecoder::new(&residency)
            .await
            .map_err(|e| SfxError::Load(format!("dac: {e:?}")))?;
        let bv = BigVganPipelines::compile_with(
            &backend,
            "moss.dac",
            thinfer_core::ops::WeightDtype::F32,
        )
        .await
        .map_err(|e| SfxError::Load(format!("dac pipelines: {e:?}")))?;
        tracing::info!(?act, "MOSS-SoundEffect loaded");
        Ok(Self {
            backend,
            residency,
            cfg,
            dit,
            dit_handles,
            text,
            llm,
            dac,
            bv,
        })
    }

    pub fn residency(&self) -> &WeightResidency<S> {
        &self.residency
    }

    fn workspace(&self) -> Workspace<WgpuBackend> {
        Workspace::new(self.backend.clone(), Arc::clone(self.residency.arbiter()))
    }

    /// `[TEXT_LEN, 2048]` prompt states (zero tail). The text and DiT phases
    /// never overlap and their weights (3.4 + 2.8 GB) do not fit an 8 GB card
    /// together with the accelerator's own memory, which the residency budget
    /// does not see: the device is cleared before the encoder runs (a warm
    /// model still holds the last job's DiT) and after it.
    pub async fn encode_text(&self, ids: &[u32]) -> Result<Zeroizing<Vec<f32>>, SfxError> {
        self.residency.evict_all_and_free(&*self.backend);
        let ws = self.workspace();
        let out = self
            .text
            .encode(&self.backend, &self.residency, &self.llm, &ws, ids)
            .await
            .map_err(|e| SfxError::Text(format!("{e:?}")))?;
        drop(ws);
        self.residency.evict_all_and_free(&*self.backend);
        Ok(out)
    }

    fn dit_model(&self) -> WanDit {
        let shape = WanDitShape::new(self.cfg.geometry, LATENT_CH, FULL_FRAMES, 1, 1, TEXT_LEN);
        WanDit::assemble(self.dit_handles.clone(), shape, self.cfg)
    }

    /// Keep the DiT forward's working set free on every weight admission (the
    /// Wan pipeline's reserve, at this row count).
    fn reserve_dit_workspace(&self) {
        let rows = FULL_FRAMES as u64;
        let dim = self.cfg.inner() as u64;
        let dff = self.cfg.ffn_dim as u64;
        let bpe = self.dit.block.act_bytes(1);
        let tile = (crate::wan::dit_block::FFN_TILE_ROWS as u64).min(rows);
        let attn = rows * 6 * dim * bpe;
        let ffn = rows * 2 * dim * bpe + 2 * (tile * 2 * dff * bpe);
        self.residency.set_transient_reserve(
            crate::wan::pipeline::PREFETCH_STAGING_DEPTH
                * self.residency.vram_staging_reserve_bytes()
                + attn.max(ffn)
                + 4 * dim * dff * bpe,
        );
    }

    /// The DiT velocity of a `[LATENT_CH, FULL_FRAMES]` latent under each
    /// text in `texts` (the CFG pair: one pass over the blocks, each block
    /// running every text while resident, all on the accelerator at f16).
    pub async fn velocities(
        &self,
        x: &[f32],
        sigma: f32,
        texts: &[&[f32]],
    ) -> Result<Vec<Vec<f32>>, SfxError> {
        let ws = self.workspace();
        let dit = self.dit_model();
        let inputs: Vec<WanDitInputs<'_>> = texts
            .iter()
            .map(|text| WanDitInputs {
                image: x,
                img_ref: None,
                text,
                r_timestep: None,
                timestep: timestep(sigma),
                attn_window: None,
                token_add: None,
            })
            .collect();
        let out = dit
            .forward_many(&self.backend, &self.dit, &self.residency, &ws, &inputs)
            .await
            .map_err(|e| SfxError::Dit(format!("{e:?}")))?;
        Ok(out.into_iter().map(|o| o.image).collect())
    }

    /// `[LATENT_CH, frames]` latents to `frames * HOP` samples.
    pub async fn decode(&self, latent: &[f32]) -> Result<Vec<f32>, SfxError> {
        let ws = self.workspace();
        self.dac
            .decode(&self.backend, &self.residency, &ws, &self.bv, latent)
            .await
            .map_err(|e| SfxError::Decode(format!("{e:?}")))
    }

    /// The denoised `[LATENT_CH, FULL_FRAMES]` latent of `p`.
    pub async fn denoise(
        &self,
        p: &SfxParams<'_>,
        on_step: StepHook<'_>,
    ) -> Result<Vec<f32>, SfxError> {
        let cond = self.encode_text(p.text_ids).await?;
        let uncond = Zeroizing::new(vec![0f32; cond.len()]);
        let n = LATENT_CH * FULL_FRAMES;
        let mut x = match p.noise {
            Some(v) if v.len() == n => v.to_vec(),
            Some(v) => {
                return Err(SfxError::Dit(format!("noise override {} != {n}", v.len())));
            }
            None => gaussian_noise(n, p.seed),
        };
        let clip = frames_for(samples_for(p.seconds)).min(FULL_FRAMES);
        let sigmas = schedule(p.steps);
        self.reserve_dit_workspace();
        for (i, &sigma) in sigmas.iter().enumerate() {
            let shift = if p.looped {
                loop_offset(i as u32, clip)
            } else {
                0
            };
            roll_left(&mut x, FULL_FRAMES, clip, shift);
            let texts: &[&[f32]] = if p.guidance != 1.0 {
                &[&cond, &uncond]
            } else {
                &[&cond]
            };
            let mut vs = self.velocities(&x, sigma, texts).await?;
            let mut v = vs.swap_remove(0);
            if let Some(vu) = vs.first() {
                for (a, u) in v.iter_mut().zip(vu) {
                    *a = u + p.guidance * (*a - u);
                }
            }
            let next = sigmas.get(i + 1).copied().unwrap_or(0.0);
            let dt = next - sigma;
            for (xv, vv) in x.iter_mut().zip(&v) {
                *xv += vv * dt;
            }
            roll_right(&mut x, FULL_FRAMES, clip, shift);
            if !on_step(i as u32 + 1, p.steps) {
                return Err(SfxError::Cancelled);
            }
        }
        Ok(x)
    }

    /// Samples of `p` (`seconds * 48000`).
    pub async fn generate(
        &self,
        p: &SfxParams<'_>,
        on_step: StepHook<'_>,
    ) -> Result<Vec<f32>, SfxError> {
        let x = self.denoise(p, on_step).await?;
        self.decode_clip(&x, p.seconds, p.looped).await
    }

    /// Decode the clip of a denoised latent: only its frames plus the
    /// decoder's receptive field (wrapped for a loop).
    pub async fn decode_clip(
        &self,
        x: &[f32],
        seconds: f32,
        looped: bool,
    ) -> Result<Vec<f32>, SfxError> {
        let samples = samples_for(seconds);
        let clip = frames_for(samples).min(FULL_FRAMES);
        let (window, lead) = decode_window(x, FULL_FRAMES, clip, looped, DECODE_MARGIN);
        let wav = self.decode(&window).await?;
        let start = lead * super::HOP;
        Ok(wav[start..start + samples].to_vec())
    }
}
