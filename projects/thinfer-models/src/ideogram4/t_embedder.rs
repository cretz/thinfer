//! Ideogram-4 timestep conditioning: `Ideogram4EmbedScalar` + `adaln_proj`.
//!
//! Mirrors `modeling_ideogram4.py`:
//! ```text
//! scaled = 1e4 * (t - 0) / (1 - 0)                  # input_range (0,1)
//! emb    = cat(sin(scaled*freq), cos(scaled*freq))  # freq base 1e4, sin FIRST
//! t_emb  = mlp_out(silu(mlp_in(emb)))               # 4608->4608->4608
//! adaln_input = silu(adaln_proj(t_emb))             # 4608->512
//! ```
//! `adaln_input` (`[1, ADALN_DIM]`) is the shared modulation conditioning fed to
//! every block AND the final layer (broadcast over the token axis). Two distinct
//! `1e4` constants: the input scaling and the sinusoid frequency base.
//!
//! Single-batch (B=1): the sinusoid is built on CPU (one row), the three linears
//! run via the M=1-tuned `matmul_adaln` pipeline (bf16 weights in every pipeline
//! set, so this is identical under the quant-DiT and dense pipeline sets).

use thinfer_core::backend::{WgpuBackend, WgpuError};
use thinfer_core::ops::{ActDtype, BcastAddF32, SiluF32};
use thinfer_core::residency::{ResidencyError, WeightResidency};
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::{BatchBuf, BatchScope};

use crate::common::block::BlockPipelines;
use crate::common::embedders::{LinearBiasBufs, LinearBiasHandles, LinearBiasViews};

use super::config;

/// `Ideogram4EmbedScalar` frequency table + the silu-MLP that maps the sinusoid
/// to `t_emb`, then `adaln_proj` to `adaln_input`.
pub struct TEmbedder {
    /// `exp(-i * ln(1e4) / (half-1))`, `half = DIM/2`. Built once.
    freqs: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct TEmbedderBufs {
    pub mlp_in: LinearBiasBufs,
    pub mlp_out: LinearBiasBufs,
    pub adaln_proj: LinearBiasBufs,
}

impl TEmbedder {
    pub fn new() -> Self {
        let half = config::DIM / 2;
        let log_scale = (1e4_f32).ln();
        let freqs = (0..half)
            .map(|i| (-(i as f32) * log_scale / (half as f32 - 1.0)).exp())
            .collect();
        Self { freqs }
    }

    /// `emb = cat(sin(scaled*freq), cos(scaled*freq))`, `scaled = 1e4 * t`.
    /// Encoded for the act dtype (the matmul_adaln A-side). `I8` never reaches
    /// here (adaln runs at the AdaLN-effective dtype, bf16 for Ideogram-4).
    fn compute_embed(&self, t: f32, act: ActDtype) -> Vec<u8> {
        let scaled = 1e4 * t;
        let half = self.freqs.len();
        let stride = act.bytes_per_elem() as usize;
        let mut bytes = vec![0u8; config::DIM * stride];
        let write = |bytes: &mut [u8], idx: usize, v: f32| match act {
            ActDtype::F32 => bytes[idx * 4..(idx + 1) * 4].copy_from_slice(&v.to_le_bytes()),
            ActDtype::Bf16 => {
                bytes[idx * 2..(idx + 1) * 2].copy_from_slice(&round_f32_to_bf16(v).to_le_bytes())
            }
            ActDtype::F16 => bytes[idx * 2..(idx + 1) * 2]
                .copy_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes()),
            ActDtype::I8 => unreachable!("t_embedder runs at the AdaLN-effective dtype, never I8"),
        };
        for (i, &f) in self.freqs.iter().enumerate() {
            let arg = scaled * f;
            write(&mut bytes, i, arg.sin());
            write(&mut bytes, half + i, arg.cos());
        }
        bytes
    }

    /// Writes `[1, ADALN_DIM]` `adaln_input` to `out`. Single-batch only.
    pub fn forward<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        t: f32,
        out: BatchBuf<'wsp>,
        bufs: &'wsp TEmbedderBufs,
    ) -> Result<(), WgpuError> {
        let dim = config::DIM as u32;
        let ad = config::ADALN_DIM as u32;

        let embed_bytes = self.compute_embed(t, pipelines.act_dtype);
        let embed = scope.write_uniform(&embed_bytes)?;

        // mlp_in: [1, DIM] @ [DIM, DIM] + b -> silu
        let h1 = self.linear(scope, pipelines, embed, &bufs.mlp_in, dim, dim)?;
        let h1a = scope.alloc(pipelines.act_bytes(dim))?;
        scope.dispatch_op::<SiluF32>(&pipelines.silu, &[h1], h1a)?;
        // mlp_out: [1, DIM] @ [DIM, DIM] + b -> t_emb
        let t_emb = self.linear(scope, pipelines, h1a, &bufs.mlp_out, dim, dim)?;
        // adaln_proj: [1, DIM] @ [DIM, ADALN_DIM] + b -> silu -> adaln_input
        let proj = self.linear(scope, pipelines, t_emb, &bufs.adaln_proj, dim, ad)?;
        scope.dispatch_op::<SiluF32>(&pipelines.silu, &[proj], out)
    }

    /// One M=1 `Linear(in->out)+bias` via the adaln matmul pipeline (bf16
    /// weight). Returns the freshly allocated output buffer.
    fn linear<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        x: BatchBuf<'wsp>,
        w: &LinearBiasBufs,
        in_dim: u32,
        out_dim: u32,
    ) -> Result<BatchBuf<'wsp>, WgpuError> {
        let pre = scope.alloc(pipelines.act_bytes(out_dim))?;
        let dims = scope.u32x4_uniform(1, out_dim, in_dim, 0)?;
        let wbuf = scope.import_copy(w.weight);
        scope.matmul(
            &pipelines.matmul_adaln,
            &pipelines.matmuls.adaln,
            x,
            wbuf,
            dims,
            pre,
            1,
            out_dim,
        )?;
        let outb = scope.alloc(pipelines.act_bytes(out_dim))?;
        let bias = scope.import_copy(w.bias);
        let ba_u = bcast_add_uniform(scope, out_dim)?;
        scope.bcast_add::<BcastAddF32>(&pipelines.bcast_add, pre, bias, ba_u, outb, out_dim)?;
        Ok(outb)
    }
}

impl Default for TEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct TEmbedderHandles {
    pub mlp_in: LinearBiasHandles,
    pub mlp_out: LinearBiasHandles,
    pub adaln_proj: LinearBiasHandles,
}

pub struct TEmbedderViews<'a> {
    pub mlp_in: LinearBiasViews<'a>,
    pub mlp_out: LinearBiasViews<'a>,
    pub adaln_proj: LinearBiasViews<'a>,
}

impl TEmbedderHandles {
    pub async fn acquire<'a, S: WeightSource>(
        &self,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<TEmbedderViews<'a>, ResidencyError<S::Error, WgpuError>> {
        Ok(TEmbedderViews {
            mlp_in: self.mlp_in.acquire(residency, backend).await?,
            mlp_out: self.mlp_out.acquire(residency, backend).await?,
            adaln_proj: self.adaln_proj.acquire(residency, backend).await?,
        })
    }
}

impl TEmbedderViews<'_> {
    pub fn bufs(&self) -> TEmbedderBufs {
        TEmbedderBufs {
            mlp_in: self.mlp_in.bufs(),
            mlp_out: self.mlp_out.bufs(),
            adaln_proj: self.adaln_proj.bufs(),
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

/// f32 -> bf16 RNE matching the WGSL `round_bf16` helper.
fn round_f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let exp = (bits >> 23) & 0xff;
    if exp == 0xff {
        let mant = bits & 0x7f_ffff;
        let top = (bits >> 16) as u16;
        if mant == 0 { top } else { top | 0x0040 }
    } else {
        let rounding = 0x7fff + ((bits >> 16) & 1);
        ((bits.wrapping_add(rounding)) >> 16) as u16
    }
}
