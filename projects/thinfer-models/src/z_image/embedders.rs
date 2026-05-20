//! `XEmbedder` and `CapEmbedder`: Linear-with-bias inputs to the DiT.
//!
//! - `XEmbedder` is `nn.Linear(pF*pH*pW*C_in, dim, bias=True)`. Input is the
//!   patchified image `[N_tokens, patch_in]`, output `[N_tokens, dim]`.
//! - `CapEmbedder` is `RMSNorm(cap_feat_dim) -> Linear(cap_feat_dim, dim, bias=True)`.
//!   Input `[N_cap_tokens, cap_feat_dim]`, output `[N_cap_tokens, dim]`.
//!
//! Both Linears use `matmul + bcast_add` (additive bias broadcast over rows).
//!
//! All scratch (matmul output, uniforms) is allocated through `BatchScope` so
//! pool-reuse-during-pending-dispatch is unconstructible. Caller passes the
//! output buffer in as an import.

use thinfer_core::backend::{BufRef, WgpuBackend, WgpuError};
use thinfer_core::ops::{BcastAddF32, RmsNormF32};
use thinfer_core::residency::{GpuView, ResidencyError, WeightHandle, WeightResidency};
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::{BatchBuf, BatchScope};

use crate::z_image::block::BlockPipelines;

#[derive(Clone, Copy, Debug)]
pub struct LinearBiasBufs {
    /// `[in_dim, out_dim]` (transposed from PyTorch `[out_dim, in_dim]`).
    pub weight: BufRef,
    /// `[out_dim]`.
    pub bias: BufRef,
}

#[derive(Clone, Copy, Debug)]
pub struct XEmbedderConfig {
    pub patch_in: usize,
    pub dim: usize,
}

pub struct XEmbedder {
    pub cfg: XEmbedderConfig,
}

impl XEmbedder {
    pub fn new(cfg: XEmbedderConfig) -> Self {
        Self { cfg }
    }

    /// Dispatch `x [n_tokens, patch_in] @ W + b -> out [n_tokens, dim]` into
    /// the scope's encoder. Caller submits the scope.
    pub fn forward<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        x: BatchBuf<'wsp>,
        n_tokens: u32,
        out: BatchBuf<'wsp>,
        bufs: &'wsp LinearBiasBufs,
    ) -> Result<(), WgpuError> {
        linear_bias(
            scope,
            pipelines,
            x,
            bufs,
            n_tokens,
            self.cfg.patch_in as u32,
            self.cfg.dim as u32,
            out,
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CapEmbedderConfig {
    pub cap_feat_dim: usize,
    pub dim: usize,
    pub norm_eps: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct CapEmbedderBufs {
    /// RMSNorm gain `[cap_feat_dim]`.
    pub norm_weight: BufRef,
    pub linear: LinearBiasBufs,
}

pub struct CapEmbedder {
    pub cfg: CapEmbedderConfig,
}

/// Scope-local handles to the two intermediates a debug caller may want to
/// read back: post-RMSNorm and post-matmul-pre-bias. Forward to
/// `scope.submit_many([cap_act_h?, normed, pre_bias])` to move them out as
/// `WsBuf`s after submit; production callers ignore the field.
#[derive(Clone, Copy)]
pub struct CapEmbedderIntermediateHandles<'wsp> {
    pub normed: BatchBuf<'wsp>,
    pub pre_bias: BatchBuf<'wsp>,
}

impl CapEmbedder {
    pub fn new(cfg: CapEmbedderConfig) -> Self {
        Self { cfg }
    }

    /// `cap [n_tokens, cap_feat_dim] -> out [n_tokens, dim]`. Returns scoped
    /// handles to the two intermediates (for taps via `submit_many`).
    pub fn forward<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        cap: BatchBuf<'wsp>,
        n_tokens: u32,
        out: BatchBuf<'wsp>,
        bufs: &'wsp CapEmbedderBufs,
    ) -> Result<CapEmbedderIntermediateHandles<'wsp>, WgpuError> {
        let cfd = self.cfg.cap_feat_dim as u32;
        let normed = scope.alloc(pipelines.act_bytes(n_tokens * cfd))?;
        let rms_u = rmsnorm_uniform(scope, n_tokens, cfd, self.cfg.norm_eps)?;
        let norm_w = scope.import(&bufs.norm_weight);
        scope.rmsnorm::<RmsNormF32>(&pipelines.rmsnorm, cap, norm_w, rms_u, normed, n_tokens)?;
        let pre_bias = linear_no_bias(
            scope,
            pipelines,
            normed,
            &bufs.linear,
            n_tokens,
            cfd,
            self.cfg.dim as u32,
        )?;
        let ba_u = bcast_add_uniform(scope, self.cfg.dim as u32)?;
        let bias = scope.import(&bufs.linear.bias);
        scope.bcast_add::<BcastAddF32>(
            &pipelines.bcast_add,
            pre_bias,
            bias,
            ba_u,
            out,
            n_tokens * self.cfg.dim as u32,
        )?;
        Ok(CapEmbedderIntermediateHandles { normed, pre_bias })
    }
}

fn linear_no_bias<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    x: BatchBuf<'wsp>,
    w: &'wsp LinearBiasBufs,
    n_rows: u32,
    in_dim: u32,
    out_dim: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let pre = scope.alloc(pipelines.act_bytes(n_rows * out_dim))?;
    let dims = scope.u32x4_uniform(n_rows, out_dim, in_dim, 0)?;
    let weight = scope.import(&w.weight);
    scope.matmul(
        &pipelines.matmul_qkv,
        &pipelines.matmuls.qkv,
        x,
        weight,
        dims,
        pre,
        n_rows,
        out_dim,
    )?;
    Ok(pre)
}

#[allow(clippy::too_many_arguments)]
fn linear_bias<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    x: BatchBuf<'wsp>,
    w: &'wsp LinearBiasBufs,
    n_rows: u32,
    in_dim: u32,
    out_dim: u32,
    out: BatchBuf<'wsp>,
) -> Result<(), WgpuError> {
    let pre = linear_no_bias(scope, pipelines, x, w, n_rows, in_dim, out_dim)?;
    let ba_u = bcast_add_uniform(scope, out_dim)?;
    let bias = scope.import(&w.bias);
    scope.bcast_add::<BcastAddF32>(&pipelines.bcast_add, pre, bias, ba_u, out, n_rows * out_dim)
}

#[derive(Clone, Debug)]
pub struct LinearBiasHandles {
    pub weight: WeightHandle,
    pub bias: WeightHandle,
}

#[derive(Clone, Debug)]
pub struct CapEmbedderHandles {
    pub norm_weight: WeightHandle,
    pub linear: LinearBiasHandles,
}

pub struct LinearBiasViews<'a> {
    pub weight: GpuView<'a>,
    pub bias: GpuView<'a>,
}

pub struct CapEmbedderViews<'a> {
    pub norm_weight: GpuView<'a>,
    pub linear: LinearBiasViews<'a>,
}

impl LinearBiasHandles {
    pub async fn acquire<'a, S: WeightSource>(
        &self,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<LinearBiasViews<'a>, ResidencyError<S::Error, WgpuError>> {
        Ok(LinearBiasViews {
            weight: residency.acquire(self.weight, backend).await?,
            bias: residency.acquire(self.bias, backend).await?,
        })
    }
}

impl CapEmbedderHandles {
    pub async fn acquire<'a, S: WeightSource>(
        &self,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<CapEmbedderViews<'a>, ResidencyError<S::Error, WgpuError>> {
        Ok(CapEmbedderViews {
            norm_weight: residency.acquire(self.norm_weight, backend).await?,
            linear: self.linear.acquire(residency, backend).await?,
        })
    }
}

impl LinearBiasViews<'_> {
    pub fn bufs(&self) -> LinearBiasBufs {
        LinearBiasBufs {
            weight: self.weight.buf(),
            bias: self.bias.buf(),
        }
    }
}

impl CapEmbedderViews<'_> {
    pub fn bufs(&self) -> CapEmbedderBufs {
        CapEmbedderBufs {
            norm_weight: self.norm_weight.buf(),
            linear: self.linear.bufs(),
        }
    }
}

fn rmsnorm_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    n_rows: u32,
    d: u32,
    eps: f32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&n_rows.to_le_bytes());
    bytes[4..8].copy_from_slice(&d.to_le_bytes());
    bytes[8..12].copy_from_slice(&eps.to_le_bytes());
    scope.write_uniform(&bytes)
}

fn bcast_add_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    c: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&c.to_le_bytes());
    scope.write_uniform(&bytes)
}
