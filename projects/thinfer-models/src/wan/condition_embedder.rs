//! Wan / SkyReels-V2 condition embedder (`SkyReelsV2TimeTextImageEmbedding`,
//! `transformer_skyreels_v2.py`). Produces the three signals the DiT consumes:
//!
//! - `temb` `[f, inner]`: sinusoidal time embed -> `TimestepEmbedding` MLP
//!   (`Linear(freq_dim, inner)` -> SiLU -> `Linear(inner, inner)`). Per latent
//!   frame under Diffusion Forcing. Feeds the final-layer modulation.
//! - `timestep_proj` `[f, 6*inner]`: `Linear(inner, 6*inner)` over `SiLU(temb)`,
//!   plus the DF `fps` contribution. The per-block adaLN-style modulation
//!   source.
//! - `text` `[text_seq, inner]`: `PixArtAlphaTextProjection` over the umT5
//!   states (`Linear(text_dim, inner)` -> gelu-tanh -> `Linear(inner, inner)`).
//!
//! Sinusoidal embed (`get_1d_sincos_pos_embed_from_grid`, `flip_sin_to_cos`):
//! `emb = cat(cos(t * omega), sin(t * omega))`, `omega[i] = 1/10000^(i/(D/2))`,
//! computed in f64 (matching diffusers' float64 path) then cast.
//!
//! `fps` (DF only, `inject_sample_info`): a binary bucket selecting one of two
//! `fps_embedding` rows, run through `fps_projection`
//! (`Linear(inner, inner)` -> SiLU -> `Linear(inner, 6*inner)`), then added to
//! every frame's `timestep_proj`. The embedding lookup is a `onehot[1,2] @
//! table[2, inner]` matmul so dtypes route through the matmul site cleanly.

use thinfer_core::backend::{BufRef, WgpuBackend, WgpuError, WgpuPipeline};
use thinfer_core::ops::{BcastAddF32, GeluF32, SiluF32};
use thinfer_core::residency::{GpuView, ResidencyError, WeightHandle, WeightResidency};
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::{BatchBuf, BatchScope};

use crate::common::block::BlockPipelines;
use crate::common::embedders::{LinearBiasBufs, LinearBiasHandles, LinearBiasViews};
use crate::wan::dit_block::config;

/// Resident GPU buffers for the condition embedder. Every linear's weight is
/// `[in, out]` (transposed from PyTorch `[out, in]`); `fps_embedding` is the
/// raw embedding table `[2, inner]` (`[in=2, out=inner]`, no transpose).
#[derive(Clone, Copy, Debug)]
pub struct ConditionEmbedderBufs {
    pub time_linear_1: LinearBiasBufs,
    pub time_linear_2: LinearBiasBufs,
    pub time_proj: LinearBiasBufs,
    pub text_linear_1: LinearBiasBufs,
    pub text_linear_2: LinearBiasBufs,
    /// `fps_embedding.weight` `[2, inner]`. Present iff `inject_sample_info`.
    pub fps_embedding: Option<BufRef>,
    /// `fps_projection.net.0` (`Linear(inner, inner)`, linear-silu).
    pub fps_proj_in: Option<LinearBiasBufs>,
    /// `fps_projection.net.2` (`Linear(inner, 6*inner)`).
    pub fps_proj_out: Option<LinearBiasBufs>,
}

/// Output destinations the driver owns (imported from cross-submit `WsBuf`s).
#[derive(Clone, Copy)]
pub struct ConditionEmbedderOut<'wsp> {
    /// `[f, inner]`.
    pub temb: BatchBuf<'wsp>,
    /// `[f, 6*inner]`.
    pub timestep_proj: BatchBuf<'wsp>,
    /// `[text_seq, inner]`.
    pub text: BatchBuf<'wsp>,
}

pub struct ConditionEmbedder {
    pub freq_dim: usize,
    pub inner: usize,
    pub text_dim: usize,
    pub inject_sample_info: bool,
}

impl ConditionEmbedder {
    /// SkyReels-V2-DF-1.3B config (`inject_sample_info = true`).
    pub fn skyreels_df() -> Self {
        Self {
            freq_dim: config::FREQ_DIM,
            inner: config::INNER,
            text_dim: config::TEXT_DIM,
            inject_sample_info: true,
        }
    }

    /// `emb = cat(cos(t*omega), sin(t*omega))` for each frame timestep,
    /// row-major `[f, freq_dim]`. f64 trig then cast, matching diffusers.
    fn sincos(&self, timesteps: &[f32]) -> Vec<f32> {
        let half = self.freq_dim / 2;
        let omega: Vec<f64> = (0..half)
            .map(|i| 1.0 / 10000_f64.powf(i as f64 / half as f64))
            .collect();
        let mut out = vec![0.0_f32; timesteps.len() * self.freq_dim];
        for (frame, &t) in timesteps.iter().enumerate() {
            let base = frame * self.freq_dim;
            for (i, &w) in omega.iter().enumerate() {
                let arg = t as f64 * w;
                out[base + i] = arg.cos() as f32;
                out[base + half + i] = arg.sin() as f32;
            }
        }
        out
    }

    /// `timesteps` are the per-frame noise levels (DF). `fps` is the binary
    /// `fps_embedding` bucket (ignored when `inject_sample_info` is false).
    /// `text` is the umT5 states `[text_seq, text_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        gelu: &WgpuPipeline,
        timesteps: &[f32],
        fps: usize,
        text: BatchBuf<'wsp>,
        text_seq: u32,
        out: &ConditionEmbedderOut<'wsp>,
        bufs: &ConditionEmbedderBufs,
    ) -> Result<(), WgpuError> {
        let f = timesteps.len() as u32;
        let inner = self.inner as u32;
        let freq = self.freq_dim as u32;
        let six = 6 * inner;

        // --- time path: sincos -> Linear -> SiLU -> Linear = temb [f, inner] ---
        let emb_bytes =
            crate::common::seq::act_upload_bytes(pipelines.act_dtype, &self.sincos(timesteps));
        let emb = scope.write_uniform(&emb_bytes)?;
        let h1 = self.linear_bias(scope, pipelines, emb, &bufs.time_linear_1, f, freq, inner)?;
        let h1a = scope.alloc(pipelines.act_bytes(f * inner))?;
        scope.dispatch_op::<SiluF32>(&pipelines.silu, &[h1], h1a)?;
        self.linear_bias_into(
            scope,
            pipelines,
            h1a,
            &bufs.time_linear_2,
            f,
            inner,
            inner,
            out.temb,
        )?;

        // --- timestep_proj = Linear(SiLU(temb), 6*inner) ---
        let ta = scope.alloc(pipelines.act_bytes(f * inner))?;
        scope.dispatch_op::<SiluF32>(&pipelines.silu, &[out.temb], ta)?;
        let tproj = self.linear_bias(scope, pipelines, ta, &bufs.time_proj, f, inner, six)?;

        // --- fps contribution (DF), broadcast over the f frames ---
        if self.inject_sample_info {
            let table = bufs
                .fps_embedding
                .expect("inject_sample_info needs fps_embedding");
            let proj_in = bufs
                .fps_proj_in
                .expect("inject_sample_info needs fps_proj_in");
            let proj_out = bufs
                .fps_proj_out
                .expect("inject_sample_info needs fps_proj_out");

            // fps_emb [1, inner] = onehot[1, 2] @ table[2, inner].
            let mut onehot = crate::common::seq::act_upload_bytes(pipelines.act_dtype, &[0.0, 0.0]);
            let stride = pipelines.act_dtype.bytes_per_elem() as usize;
            let one = crate::common::seq::act_upload_bytes(pipelines.act_dtype, &[1.0]);
            onehot[fps * stride..fps * stride + stride].copy_from_slice(&one);
            let oh = scope.write_uniform(&onehot)?;
            let fps_emb = scope.alloc(pipelines.act_bytes(inner))?;
            let dims = scope.u32x4_uniform(1, inner, 2, 0)?;
            let tw = scope.import_copy(table);
            scope.matmul(
                &pipelines.matmul_qkv,
                &pipelines.matmuls.qkv,
                oh,
                tw,
                dims,
                fps_emb,
                1,
                inner,
            )?;

            // fps_projection: Linear(inner, inner) -> SiLU -> Linear(inner, 6*inner).
            let p0 = self.linear_bias(scope, pipelines, fps_emb, &proj_in, 1, inner, inner)?;
            let p0a = scope.alloc(pipelines.act_bytes(inner))?;
            scope.dispatch_op::<SiluF32>(&pipelines.silu, &[p0], p0a)?;
            let fps_contrib = self.linear_bias(scope, pipelines, p0a, &proj_out, 1, inner, six)?;

            // timestep_proj += fps_contrib (broadcast [6*inner] over f rows).
            let ba_u = bcast_add_uniform(scope, six)?;
            scope.bcast_add::<BcastAddF32>(
                &pipelines.bcast_add,
                tproj,
                fps_contrib,
                ba_u,
                out.timestep_proj,
                f * six,
            )?;
        } else {
            scope.copy_buffer_to_buffer(
                tproj,
                0,
                out.timestep_proj,
                0,
                pipelines.act_bytes(f * six),
            )?;
        }

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
        scope.bcast_add::<BcastAddF32>(&pipelines.bcast_add, pre, bias, ba_u, out, rows * out_dim)
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
    pub fps_embedding: Option<WeightHandle>,
    pub fps_proj_in: Option<LinearBiasHandles>,
    pub fps_proj_out: Option<LinearBiasHandles>,
}

pub struct ConditionEmbedderViews<'a> {
    time_linear_1: LinearBiasViews<'a>,
    time_linear_2: LinearBiasViews<'a>,
    time_proj: LinearBiasViews<'a>,
    text_linear_1: LinearBiasViews<'a>,
    text_linear_2: LinearBiasViews<'a>,
    fps_embedding: Option<GpuView<'a>>,
    fps_proj_in: Option<LinearBiasViews<'a>>,
    fps_proj_out: Option<LinearBiasViews<'a>>,
}

impl ConditionEmbedderHandles {
    pub async fn acquire<'a, S: WeightSource>(
        &self,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<ConditionEmbedderViews<'a>, ResidencyError<S::Error, WgpuError>> {
        let fps_embedding = match self.fps_embedding {
            Some(h) => Some(residency.acquire(h, backend).await?),
            None => None,
        };
        let fps_proj_in = match &self.fps_proj_in {
            Some(h) => Some(h.acquire(residency, backend).await?),
            None => None,
        };
        let fps_proj_out = match &self.fps_proj_out {
            Some(h) => Some(h.acquire(residency, backend).await?),
            None => None,
        };
        Ok(ConditionEmbedderViews {
            time_linear_1: self.time_linear_1.acquire(residency, backend).await?,
            time_linear_2: self.time_linear_2.acquire(residency, backend).await?,
            time_proj: self.time_proj.acquire(residency, backend).await?,
            text_linear_1: self.text_linear_1.acquire(residency, backend).await?,
            text_linear_2: self.text_linear_2.acquire(residency, backend).await?,
            fps_embedding,
            fps_proj_in,
            fps_proj_out,
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
            fps_embedding: self.fps_embedding.as_ref().map(|v| v.buf()),
            fps_proj_in: self.fps_proj_in.as_ref().map(|v| v.bufs()),
            fps_proj_out: self.fps_proj_out.as_ref().map(|v| v.bufs()),
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
        let ce = ConditionEmbedder::skyreels_df();
        let e = ce.sincos(&[0.0, 5.0]);
        assert_eq!(e.len(), 2 * config::FREQ_DIM);
        let half = config::FREQ_DIM / 2;
        // t=0: cos=1, sin=0 across the board.
        for i in 0..half {
            assert!((e[i] - 1.0).abs() < 1e-6);
            assert!(e[half + i].abs() < 1e-6);
        }
        // omega[0]=1 so the first cos/sin pair of frame 1 is (cos 5, sin 5).
        assert!((e[config::FREQ_DIM] - 5.0_f32.cos()).abs() < 1e-5);
        assert!((e[config::FREQ_DIM + half] - 5.0_f32.sin()).abs() < 1e-5);
    }
}
