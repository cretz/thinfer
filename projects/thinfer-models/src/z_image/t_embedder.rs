//! `TimestepEmbedder`: scalar timestep -> `[B, ADALN_EMBED_DIM]` embedding for
//! the modulated blocks' adaLN input.
//!
//! Mirrors `TimestepEmbedder` in `src/zimage/transformer.py`:
//!   `embed = cat(cos(t * freqs), sin(t * freqs))`  (`freqs` length `dim/2`)
//!   `out = Linear(mid -> out)(SiLU(Linear(in -> mid)(embed)))`
//!
//! `embed` is `B * FREQUENCY_EMBEDDING_SIZE = 256` floats per step (B=1). Run
//! it on CPU; the per-step `writeBuffer` of 1 KiB is far cheaper than the
//! kernel-dispatch + bind-group overhead of two trig kernels.
//!
//! MLP runs on GPU via existing matmul + add (bias) + silu. Bias `add` is
//! correct only because B=1 matches the block pipeline constraint.

use thinfer_core::backend::{BufRef, WgpuBackend, WgpuError};
use thinfer_core::ops::{ActDtype, BcastAddF32, SiluF32};
use thinfer_core::residency::{GpuView, ResidencyError, WeightHandle, WeightResidency};
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::{BatchBuf, BatchScope};

use crate::z_image::block::BlockPipelines;

#[derive(Clone, Copy, Debug)]
pub struct TimestepEmbedderConfig {
    pub freq_dim: usize,
    pub mid_dim: usize,
    pub out_dim: usize,
    pub max_period: f32,
}

impl TimestepEmbedderConfig {
    pub const fn z_image() -> Self {
        Self {
            freq_dim: super::config::FREQUENCY_EMBEDDING_SIZE,
            mid_dim: super::config::T_EMBEDDER_MID,
            out_dim: super::config::ADALN_EMBED_DIM,
            max_period: 10_000.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TEmbedderWeightBufs {
    /// Linear `[freq_dim -> mid_dim]` weight, transposed at upload `[freq_dim, mid_dim]`.
    pub fc1_weight: BufRef,
    /// `[mid_dim]`.
    pub fc1_bias: BufRef,
    /// Linear `[mid_dim -> out_dim]` weight, transposed `[mid_dim, out_dim]`.
    pub fc2_weight: BufRef,
    /// `[out_dim]`.
    pub fc2_bias: BufRef,
}

pub struct TimestepEmbedder {
    pub cfg: TimestepEmbedderConfig,
    /// `exp(-log(max_period) * arange(0, half) / half)`, `half = freq_dim / 2`.
    /// Pre-built once. Per-step CPU cost is one mul + one cos + one sin per
    /// element (256 ops total at default sizes).
    freqs: Vec<f32>,
}

impl TimestepEmbedder {
    pub fn new(cfg: TimestepEmbedderConfig) -> Self {
        assert!(cfg.freq_dim.is_multiple_of(2), "freq_dim must be even");
        let half = cfg.freq_dim / 2;
        let log_max = cfg.max_period.ln();
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-log_max * (i as f32) / (half as f32)).exp())
            .collect();
        Self { cfg, freqs }
    }

    /// Build `embed = cat(cos(t * freqs), sin(t * freqs))` in CPU memory.
    /// `t` is the (already `t_scale`d) timestep, single value (B=1). Encoded
    /// for the matmul's activation storage dtype: f32 LE or bf16-packed (RNE).
    fn compute_embed(&self, t: f32, act: ActDtype) -> Vec<u8> {
        let half = self.freqs.len();
        let stride = act.bytes_per_elem() as usize;
        let mut bytes = vec![0u8; self.cfg.freq_dim * stride];
        let write = |bytes: &mut [u8], idx: usize, v: f32| match act {
            ActDtype::F32 => {
                bytes[idx * 4..(idx + 1) * 4].copy_from_slice(&v.to_le_bytes());
            }
            ActDtype::Bf16 => {
                let h = round_f32_to_bf16(v);
                bytes[idx * 2..(idx + 1) * 2].copy_from_slice(&h.to_le_bytes());
            }
        };
        for (i, &f) in self.freqs.iter().enumerate() {
            let arg = t * f;
            write(&mut bytes, i, arg.cos());
            write(&mut bytes, half + i, arg.sin());
        }
        bytes
    }

    /// `t` is the scalar timestep (after `t_scale`). Writes `[1, out_dim]`
    /// floats to `out`. Single-batch only.
    pub fn forward<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        t: f32,
        out: BatchBuf<'wsp>,
        bufs: &'wsp TEmbedderWeightBufs,
    ) -> Result<(), WgpuError> {
        let freq = self.cfg.freq_dim as u32;
        let mid = self.cfg.mid_dim as u32;
        let outd = self.cfg.out_dim as u32;
        let freq_bytes = pipelines.act_bytes(freq);
        let mid_bytes = pipelines.act_bytes(mid);

        let embed_bytes = self.compute_embed(t, pipelines.act_dtype);
        debug_assert_eq!(embed_bytes.len() as u64, freq_bytes);
        let embed = scope.write_uniform(&embed_bytes)?;

        // Linear 1: [1, freq] @ [freq, mid] -> [1, mid]
        let fc1_pre = scope.alloc(mid_bytes)?;
        let dims_fc1 = scope.u32x4_uniform(1, mid, freq, 0)?;
        let fc1_w = scope.import(&bufs.fc1_weight);
        scope.matmul(
            &pipelines.matmul_adaln,
            &pipelines.matmuls.adaln,
            embed,
            fc1_w,
            dims_fc1,
            fc1_pre,
            1,
            mid,
        )?;
        let fc1 = scope.alloc(mid_bytes)?;
        let fc1_b = scope.import(&bufs.fc1_bias);
        let fc1_ba_u = bcast_add_uniform(scope, mid)?;
        scope.bcast_add::<BcastAddF32>(&pipelines.bcast_add, fc1_pre, fc1_b, fc1_ba_u, fc1, mid)?;

        // SiLU
        let act = scope.alloc(mid_bytes)?;
        scope.dispatch_op::<SiluF32>(&pipelines.silu, &[fc1], act)?;

        // Linear 2: [1, mid] @ [mid, out] -> [1, out]
        let fc2_pre = scope.alloc(pipelines.act_bytes(outd))?;
        let dims_fc2 = scope.u32x4_uniform(1, outd, mid, 0)?;
        let fc2_w = scope.import(&bufs.fc2_weight);
        scope.matmul(
            &pipelines.matmul_adaln,
            &pipelines.matmuls.adaln,
            act,
            fc2_w,
            dims_fc2,
            fc2_pre,
            1,
            outd,
        )?;
        let fc2_b = scope.import(&bufs.fc2_bias);
        let fc2_ba_u = bcast_add_uniform(scope, outd)?;
        scope.bcast_add::<BcastAddF32>(
            &pipelines.bcast_add,
            fc2_pre,
            fc2_b,
            fc2_ba_u,
            out,
            outd,
        )?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct TEmbedderWeightHandles {
    pub fc1_weight: WeightHandle,
    pub fc1_bias: WeightHandle,
    pub fc2_weight: WeightHandle,
    pub fc2_bias: WeightHandle,
}

pub struct TEmbedderWeightViews<'a> {
    pub fc1_weight: GpuView<'a>,
    pub fc1_bias: GpuView<'a>,
    pub fc2_weight: GpuView<'a>,
    pub fc2_bias: GpuView<'a>,
}

impl TEmbedderWeightHandles {
    pub async fn acquire<'a, S: WeightSource>(
        &self,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<TEmbedderWeightViews<'a>, ResidencyError<S::Error, WgpuError>> {
        Ok(TEmbedderWeightViews {
            fc1_weight: residency.acquire(self.fc1_weight, backend).await?,
            fc1_bias: residency.acquire(self.fc1_bias, backend).await?,
            fc2_weight: residency.acquire(self.fc2_weight, backend).await?,
            fc2_bias: residency.acquire(self.fc2_bias, backend).await?,
        })
    }
}

/// f32 -> bf16 RNE; matches the WGSL `round_bf16` helper bit-for-bit so
/// host-uploaded activations agree with kernel-rounded ones.
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

fn bcast_add_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    c: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&c.to_le_bytes());
    scope.write_uniform(&bytes)
}

impl TEmbedderWeightViews<'_> {
    pub fn bufs(&self) -> TEmbedderWeightBufs {
        TEmbedderWeightBufs {
            fc1_weight: self.fc1_weight.buf(),
            fc1_bias: self.fc1_bias.buf(),
            fc2_weight: self.fc2_weight.buf(),
            fc2_bias: self.fc2_bias.buf(),
        }
    }
}
