//! Linear-with-bias building blocks shared across models: the `[in, out]`
//! weight + `[out]` bias buffer/handle/view triple and the `matmul (+ bcast
//! bias)` dispatch helpers. Both Linears use `matmul + bcast_add` (additive
//! bias broadcast over rows).
//!
//! All scratch (matmul output, uniforms) is allocated through `BatchScope` so
//! pool-reuse-during-pending-dispatch is unconstructible.

use thinfer_core::backend::{BufRef, WgpuBackend, WgpuError};
use thinfer_core::ops::BcastAddF32;
use thinfer_core::residency::{GpuView, ResidencyError, WeightHandle, WeightResidency};
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::{BatchBuf, BatchScope};

use crate::common::block::BlockPipelines;

#[derive(Clone, Copy, Debug)]
pub struct LinearBiasBufs {
    /// `[in_dim, out_dim]` (transposed from PyTorch `[out_dim, in_dim]`).
    pub weight: BufRef,
    /// `[out_dim]`.
    pub bias: BufRef,
}

pub(crate) fn linear_no_bias<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    x: BatchBuf<'wsp>,
    w: &LinearBiasBufs,
    n_rows: u32,
    in_dim: u32,
    out_dim: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let pre = scope.alloc(pipelines.act_bytes(n_rows * out_dim))?;
    let dims = scope.u32x4_uniform(n_rows, out_dim, in_dim, 0)?;
    let weight = scope.import_copy(w.weight);
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
pub(crate) fn linear_bias<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    x: BatchBuf<'wsp>,
    w: &LinearBiasBufs,
    n_rows: u32,
    in_dim: u32,
    out_dim: u32,
    out: BatchBuf<'wsp>,
) -> Result<(), WgpuError> {
    let pre = linear_no_bias(scope, pipelines, x, w, n_rows, in_dim, out_dim)?;
    let ba_u = bcast_add_uniform(scope, out_dim)?;
    let bias = scope.import_copy(w.bias);
    scope.bcast_add::<BcastAddF32>(&pipelines.bcast_add, pre, bias, ba_u, out, n_rows * out_dim)
}

#[derive(Clone, Debug)]
pub struct LinearBiasHandles {
    pub weight: WeightHandle,
    pub bias: WeightHandle,
}

pub struct LinearBiasViews<'a> {
    pub weight: GpuView<'a>,
    pub bias: GpuView<'a>,
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

impl LinearBiasViews<'_> {
    pub fn bufs(&self) -> LinearBiasBufs {
        LinearBiasBufs {
            weight: self.weight.buf(),
            bias: self.bias.buf(),
        }
    }
}

pub(crate) fn rmsnorm_uniform<'wsp>(
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

pub(crate) fn bcast_add_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    c: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&c.to_le_bytes());
    scope.write_uniform(&bytes)
}
