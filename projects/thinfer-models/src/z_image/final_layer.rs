//! `FinalLayer`: LayerNorm (no affine) -> adaLN scale -> Linear+bias.
//!
//! Mirrors `FinalLayer` in `src/zimage/transformer.py`:
//! ```text
//! scale = 1.0 + Linear(SiLU(c))                  // [B, dim]
//! x = LayerNorm(x) * scale.unsqueeze(1)          // [B, S, dim]
//! out = Linear(x)                                 // [B, S, out_channels]
//! ```
//!
//! Single-batch (B=1): scale broadcasts over `S` rows via `bcast_affine` with
//! `bias=0.0` (multiply-only). adaLN MLP is `SiLU(c) -> Linear(adaln_embed -> dim, bias=True)`
//! producing `[1, dim]`. Output linear is `Linear(dim, out_channels, bias=True)`.

use thinfer_core::backend::{WgpuBackend, WgpuError};
use thinfer_core::ops::{BcastAddF32, BcastAffineF32, LayerNormF32, SiluF32};
use thinfer_core::residency::{ResidencyError, WeightResidency};
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::{BatchBuf, BatchScope};

use crate::z_image::block::BlockPipelines;
use crate::z_image::embedders::{LinearBiasBufs, LinearBiasHandles, LinearBiasViews};

#[derive(Clone, Copy, Debug)]
pub struct FinalLayerConfig {
    pub dim: usize,
    pub adaln_embed_dim: usize,
    pub out_channels: usize,
    /// `nn.LayerNorm` defaults to `eps=1e-5`; upstream `FinalLayer` overrides
    /// to `1e-6`.
    pub norm_eps: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct FinalLayerBufs {
    /// Output projection `Linear(dim, out_channels, bias=True)`. Weight is
    /// transposed at upload to `[dim, out_channels]`.
    pub linear: LinearBiasBufs,
    /// adaLN modulation `Linear(adaln_embed_dim, dim, bias=True)` (the SiLU
    /// before it has no params). Weight transposed `[adaln_embed_dim, dim]`.
    pub adaln: LinearBiasBufs,
}

pub struct FinalLayer {
    pub cfg: FinalLayerConfig,
}

impl FinalLayer {
    pub fn new(cfg: FinalLayerConfig) -> Self {
        Self { cfg }
    }

    /// `x [S, dim]`, `c [1, adaln_embed_dim]` -> `out [S, out_channels]`.
    /// Single-batch only. Caller owns `out` (typically imported from a
    /// `WsBuf` so it survives `scope.submit`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        x: BatchBuf<'wsp>,
        c: BatchBuf<'wsp>,
        seq: u32,
        out: BatchBuf<'wsp>,
        bufs: &'wsp FinalLayerBufs,
    ) -> Result<(), WgpuError> {
        let dim = self.cfg.dim as u32;
        let aed = self.cfg.adaln_embed_dim as u32;
        let oc = self.cfg.out_channels as u32;
        let eps = self.cfg.norm_eps;
        let act_bytes = (seq * dim) as u64 * 4;
        let scale_bytes = (dim * 4) as u64;

        // adaLN MLP: SiLU(c) -> Linear -> [1, dim], then `1 + ...`. The bcast
        // unit absorbs the `+1` via `bcast_affine(bias=1.0)` over LN output.
        let cs = scope.alloc((aed * 4) as u64)?;
        scope.dispatch_op::<SiluF32>(&pipelines.silu, &[c], cs)?;
        let pre = scope.alloc(scale_bytes)?;
        let dims_adaln = scope.u32x4_uniform(1, dim, aed, 0)?;
        let adaln_w = scope.import(&bufs.adaln.weight);
        scope.matmul(
            &pipelines.matmul_adaln,
            &pipelines.matmuls.adaln,
            cs,
            adaln_w,
            dims_adaln,
            pre,
            1,
            dim,
        )?;
        let scale = scope.alloc(scale_bytes)?;
        let adaln_b = scope.import(&bufs.adaln.bias);
        let adaln_ba_u = bcast_add_uniform(scope, dim)?;
        scope.bcast_add::<BcastAddF32>(
            &pipelines.bcast_add,
            pre,
            adaln_b,
            adaln_ba_u,
            scale,
            dim,
        )?;

        // LayerNorm(x): [S, dim] -> [S, dim].
        let normed = scope.alloc(act_bytes)?;
        let ln_u = ln_uniform(scope, seq, dim, eps)?;
        scope.layernorm::<LayerNormF32>(&pipelines.layernorm, x, ln_u, normed, seq)?;

        // x = LN(x) * (1 + scale). bcast_affine(x, scale, bias=1.0) =
        // x * (scale + 1.0), broadcast over channel dim.
        let scaled = scope.alloc(act_bytes)?;
        let bcast_u = bcast_affine_uniform(scope, dim, 1.0)?;
        scope.bcast_affine::<BcastAffineF32>(
            &pipelines.bcast_affine,
            normed,
            scale,
            bcast_u,
            scaled,
            seq * dim,
        )?;

        // Output projection: [S, dim] @ [dim, out_channels] + bias.
        let pre_out = scope.alloc((seq * oc) as u64 * 4)?;
        let dims_out = scope.u32x4_uniform(seq, oc, dim, 0)?;
        let lin_w = scope.import(&bufs.linear.weight);
        scope.matmul(
            &pipelines.matmul_proj,
            &pipelines.matmuls.proj,
            scaled,
            lin_w,
            dims_out,
            pre_out,
            seq,
            oc,
        )?;
        // Bias broadcast over rows via bcast_add.
        let ba_u = bcast_add_uniform(scope, oc)?;
        let lin_b = scope.import(&bufs.linear.bias);
        scope.bcast_add::<BcastAddF32>(&pipelines.bcast_add, pre_out, lin_b, ba_u, out, seq * oc)
    }
}

#[derive(Clone, Debug)]
pub struct FinalLayerHandles {
    pub linear: LinearBiasHandles,
    pub adaln: LinearBiasHandles,
}

pub struct FinalLayerViews<'a> {
    pub linear: LinearBiasViews<'a>,
    pub adaln: LinearBiasViews<'a>,
}

impl FinalLayerHandles {
    pub async fn acquire<'a, S: WeightSource>(
        &self,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<FinalLayerViews<'a>, ResidencyError<S::Error, WgpuError>> {
        Ok(FinalLayerViews {
            linear: self.linear.acquire(residency, backend).await?,
            adaln: self.adaln.acquire(residency, backend).await?,
        })
    }
}

impl FinalLayerViews<'_> {
    pub fn bufs(&self) -> FinalLayerBufs {
        FinalLayerBufs {
            linear: self.linear.bufs(),
            adaln: self.adaln.bufs(),
        }
    }
}

fn ln_uniform<'wsp>(
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

fn bcast_affine_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    c: u32,
    bias: f32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&c.to_le_bytes());
    bytes[4..8].copy_from_slice(&bias.to_le_bytes());
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
