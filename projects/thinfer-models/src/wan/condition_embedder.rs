//! Wan2.2 condition embedder (`WanTimeTextImageEmbedding`, diffusers
//! `transformer_wan.py`). Produces the three signals the DiT consumes from a
//! SINGLE scalar timestep (the distilled T2V line is plain flow-matching: one
//! noise level for the whole clip, not the per-frame Diffusion-Forcing schedule
//! the abandoned SkyReels-V2-DF path used):
//!
//! - `temb` `[1, inner]`: sinusoidal time embed -> `TimestepEmbedding` MLP
//!   (`Linear(freq_dim, inner)` -> SiLU -> `Linear(inner, inner)`). Feeds the
//!   final-layer modulation.
//! - `timestep_proj` `[1, 6*inner]`: `Linear(inner, 6*inner)` over `SiLU(temb)`.
//!   The per-block adaLN-style modulation source.
//! - `text` `[text_seq, inner]`: `PixArtAlphaTextProjection` over the umT5
//!   states (`Linear(text_dim, inner)` -> gelu-tanh -> `Linear(inner, inner)`).
//!
//! Sinusoidal embed (`get_1d_sincos_pos_embed_from_grid`, `flip_sin_to_cos`):
//! `emb = cat(cos(t * omega), sin(t * omega))`, `omega[i] = 1/10000^(i/(D/2))`,
//! computed in f64 (matching diffusers' float64 path) then cast.
//!
//! No `fps`/`inject_sample_info` (SkyReels-V2-DF only) and no `image_embedder`
//! (I2V only): the FastWan/LongLive base is T2V, time + text only.

use thinfer_core::backend::{WgpuBackend, WgpuError, WgpuPipeline};
use thinfer_core::ops::{GeluF32, SiluF32};
use thinfer_core::residency::{ResidencyError, WeightResidency};
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::{BatchBuf, BatchScope};

use crate::common::block::BlockPipelines;
use crate::common::embedders::{LinearBiasBufs, LinearBiasHandles, LinearBiasViews};
use crate::wan::dit_block::{WanDitConfig, config};

/// Resident GPU buffers for the condition embedder. Every linear's weight is
/// `[in, out]` (transposed from PyTorch `[out, in]`).
#[derive(Clone, Copy, Debug)]
pub struct ConditionEmbedderBufs {
    pub time_linear_1: LinearBiasBufs,
    pub time_linear_2: LinearBiasBufs,
    pub time_proj: LinearBiasBufs,
    pub text_linear_1: LinearBiasBufs,
    pub text_linear_2: LinearBiasBufs,
}

/// Output destinations the driver owns (imported from cross-submit `WsBuf`s).
#[derive(Clone, Copy)]
pub struct ConditionEmbedderOut<'wsp> {
    /// `[1, inner]`.
    pub temb: BatchBuf<'wsp>,
    /// `[1, 6*inner]`.
    pub timestep_proj: BatchBuf<'wsp>,
    /// `[text_seq, inner]`.
    pub text: BatchBuf<'wsp>,
}

pub struct ConditionEmbedder {
    pub freq_dim: usize,
    pub inner: usize,
    pub text_dim: usize,
}

impl ConditionEmbedder {
    pub fn from_cfg(cfg: &WanDitConfig) -> Self {
        Self {
            freq_dim: config::FREQ_DIM,
            inner: cfg.inner(),
            text_dim: config::TEXT_DIM,
        }
    }

    /// `emb = cat(cos(t*omega), sin(t*omega))`, row-major `[freq_dim]`. f64 trig
    /// then cast, matching diffusers.
    fn sincos(&self, t: f32) -> Vec<f32> {
        let half = self.freq_dim / 2;
        let mut out = vec![0.0_f32; self.freq_dim];
        for i in 0..half {
            let w = 1.0 / 10000_f64.powf(i as f64 / half as f64);
            let arg = t as f64 * w;
            out[i] = arg.cos() as f32;
            out[half + i] = arg.sin() as f32;
        }
        out
    }

    /// `t` is the scalar diffusion timestep. `text` is the umT5 states
    /// `[text_seq, text_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        gelu: &WgpuPipeline,
        t: f32,
        text: BatchBuf<'wsp>,
        text_seq: u32,
        out: &ConditionEmbedderOut<'wsp>,
        bufs: &ConditionEmbedderBufs,
    ) -> Result<(), WgpuError> {
        let inner = self.inner as u32;
        let freq = self.freq_dim as u32;
        let six = 6 * inner;

        // --- time path: sincos -> Linear -> SiLU -> Linear = temb [1, inner] ---
        let emb_bytes = crate::common::seq::act_upload_bytes(pipelines.act_dtype, &self.sincos(t));
        let emb = scope.write_uniform(&emb_bytes)?;
        let h1 = self.linear_bias(scope, pipelines, emb, &bufs.time_linear_1, 1, freq, inner)?;
        let h1a = scope.alloc(pipelines.act_bytes(inner))?;
        scope.dispatch_op::<SiluF32>(&pipelines.silu, &[h1], h1a)?;
        self.linear_bias_into(
            scope,
            pipelines,
            h1a,
            &bufs.time_linear_2,
            1,
            inner,
            inner,
            out.temb,
        )?;

        // --- timestep_proj = Linear(SiLU(temb), 6*inner) [1, 6*inner] ---
        let ta = scope.alloc(pipelines.act_bytes(inner))?;
        scope.dispatch_op::<SiluF32>(&pipelines.silu, &[out.temb], ta)?;
        self.linear_bias_into(
            scope,
            pipelines,
            ta,
            &bufs.time_proj,
            1,
            inner,
            six,
            out.timestep_proj,
        )?;

        // --- text path: Linear -> gelu-tanh -> Linear = text [text_seq, inner] ---
        let td = self.text_dim as u32;
        let t1 = self.linear_bias(
            scope,
            pipelines,
            text,
            &bufs.text_linear_1,
            text_seq,
            td,
            inner,
        )?;
        let t1a = scope.alloc(pipelines.act_bytes(text_seq * inner))?;
        scope.dispatch_op::<GeluF32>(gelu, &[t1], t1a)?;
        self.linear_bias_into(
            scope,
            pipelines,
            t1a,
            &bufs.text_linear_2,
            text_seq,
            inner,
            inner,
            out.text,
        )?;
        Ok(())
    }

    /// `out = x @ wᵀ + b`, allocating the result in the scope pool.
    #[allow(clippy::too_many_arguments)]
    fn linear_bias<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        x: BatchBuf<'wsp>,
        w: &LinearBiasBufs,
        rows: u32,
        in_dim: u32,
        out_dim: u32,
    ) -> Result<BatchBuf<'wsp>, WgpuError> {
        let dst = scope.alloc(pipelines.act_bytes(rows * out_dim))?;
        self.linear_bias_into(scope, pipelines, x, w, rows, in_dim, out_dim, dst)?;
        Ok(dst)
    }

    /// `out = x @ wᵀ + b` into a caller-provided destination.
    #[allow(clippy::too_many_arguments)]
    fn linear_bias_into<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        x: BatchBuf<'wsp>,
        w: &LinearBiasBufs,
        rows: u32,
        in_dim: u32,
        out_dim: u32,
        out: BatchBuf<'wsp>,
    ) -> Result<(), WgpuError> {
        let pre = scope.alloc(pipelines.act_bytes(rows * out_dim))?;
        let dims = scope.u32x4_uniform(rows, out_dim, in_dim, 0)?;
        let weight = scope.import_copy(w.weight);
        scope.matmul(
            &pipelines.matmul_qkv,
            &pipelines.matmuls.qkv,
            x,
            weight,
            dims,
            pre,
            rows,
            out_dim,
        )?;
        let ba_u = bcast_add_uniform(scope, out_dim)?;
        let bias = scope.import_copy(w.bias);
        scope.bcast_add::<thinfer_core::ops::BcastAddF32>(
            &pipelines.bcast_add,
            pre,
            bias,
            ba_u,
            out,
            rows * out_dim,
        )
    }
}

// ---------------------------------------------------------------------------
// Residency handles / views
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ConditionEmbedderHandles {
    pub time_linear_1: LinearBiasHandles,
    pub time_linear_2: LinearBiasHandles,
    pub time_proj: LinearBiasHandles,
    pub text_linear_1: LinearBiasHandles,
    pub text_linear_2: LinearBiasHandles,
}

pub struct ConditionEmbedderViews<'a> {
    time_linear_1: LinearBiasViews<'a>,
    time_linear_2: LinearBiasViews<'a>,
    time_proj: LinearBiasViews<'a>,
    text_linear_1: LinearBiasViews<'a>,
    text_linear_2: LinearBiasViews<'a>,
}

impl ConditionEmbedderHandles {
    pub async fn acquire<'a, S: WeightSource>(
        &self,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<ConditionEmbedderViews<'a>, ResidencyError<S::Error, WgpuError>> {
        Ok(ConditionEmbedderViews {
            time_linear_1: self.time_linear_1.acquire(residency, backend).await?,
            time_linear_2: self.time_linear_2.acquire(residency, backend).await?,
            time_proj: self.time_proj.acquire(residency, backend).await?,
            text_linear_1: self.text_linear_1.acquire(residency, backend).await?,
            text_linear_2: self.text_linear_2.acquire(residency, backend).await?,
        })
    }
}

impl ConditionEmbedderViews<'_> {
    pub fn bufs(&self) -> ConditionEmbedderBufs {
        ConditionEmbedderBufs {
            time_linear_1: self.time_linear_1.bufs(),
            time_linear_2: self.time_linear_2.bufs(),
            time_proj: self.time_proj.bufs(),
            text_linear_1: self.text_linear_1.bufs(),
            text_linear_2: self.text_linear_2.bufs(),
        }
    }
}

fn bcast_add_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    c: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&c.to_le_bytes());
    scope.write_uniform(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sincos_origin_and_shape() {
        let ce = ConditionEmbedder::from_cfg(&WanDitConfig::fastwan_ti2v_5b());
        let e = ce.sincos(0.0);
        assert_eq!(e.len(), config::FREQ_DIM);
        let half = config::FREQ_DIM / 2;
        // t=0: cos=1, sin=0 across the board.
        for i in 0..half {
            assert!((e[i] - 1.0).abs() < 1e-6);
            assert!(e[half + i].abs() < 1e-6);
        }
        // omega[0]=1 so frame value at t=5 is (cos 5, sin 5) in the first slot.
        let e5 = ce.sincos(5.0);
        assert!((e5[0] - 5.0_f32.cos()).abs() < 1e-5);
        assert!((e5[half] - 5.0_f32.sin()).abs() < 1e-5);
    }
}
