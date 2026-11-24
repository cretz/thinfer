//! The DAC VAE decoder (`vae_128d_48k.pth`, `DAC(continuous=True)`):
//! `post_quant_conv` (k=1), a k=7 conv to 2048 channels, five up blocks
//! (Snake, a transposed conv `k = 2s` by the rates 8/5/4/3/2, three residual
//! units dilated 1/3/9), Snake, a k=7 conv to one channel, tanh.
//!
//! Runs f32 (weights too: compile the BigVGAN pipelines with
//! `WeightDtype::F32`) on the BigVGAN kernels, one submit per residual unit so no submit
//! grows with the clip (the MiniMax DAC encoder's scheme). The checkpoint's
//! convs are weight-normed: serve it through
//! [`WeightNormSource`](crate::common::weight_norm::WeightNormSource), which
//! folds each `weight_g` / `weight_v` pair into `weight`. Snake's `alpha` is
//! raw (some are negative), so it runs the raw-parameter Snake op.

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError};
use thinfer_core::ops::ActDtype;
use thinfer_core::residency::{GpuView, ResidencyError, WeightHandle, WeightResidency};
use thinfer_core::weight::{WeightId, WeightReader, WeightSource, decode_into};
use thinfer_core::workspace::{Workspace, WsBuf};

use super::LATENT_CH;
use crate::common::bigvgan::{
    BigVganPipelines, Cl, accel_covers, add, conv_transpose1d, conv1d, persist_cl, snake1d,
};
use crate::common::loader::{LoadError, register_keep_f32};
use crate::common::seq::act_upload_bytes;

/// Where the checkpoint's tensors sit: `torch.save({"state_dict": ..,
/// "metadata": ..})`, flattened with dots.
pub const PREFIX: &str = "state_dict.";
/// Upsampling strides (`decoder_rates`); their product is the hop.
pub const RATES: [usize; 5] = [8, 5, 4, 3, 2];
/// Channels after the input conv (`decoder_dim`); each block halves them.
const DIM: usize = 2048;
const DILATIONS: [u32; 3] = [1, 3, 9];

#[derive(Debug)]
pub enum DacError<SE: core::fmt::Debug> {
    Wgpu(WgpuError),
    Load(LoadError),
    Residency(ResidencyError<SE, WgpuError>),
    Weight(String),
}

impl<SE: core::fmt::Debug> From<WgpuError> for DacError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}
impl<SE: core::fmt::Debug> From<LoadError> for DacError<SE> {
    fn from(e: LoadError) -> Self {
        Self::Load(e)
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for DacError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

/// One conv layer: weights plus geometry. `transposed` convs carry their
/// `output_padding`.
struct Conv {
    weight: WeightHandle,
    bias: WeightHandle,
    cout: u32,
    k: u32,
    stride: u32,
    dilation: u32,
    pad: u32,
    transposed: Option<u32>,
}

impl Conv {
    fn plain<S: WeightSource>(
        res: &WeightResidency<S>,
        prefix: &str,
        cout: usize,
        k: usize,
        dilation: u32,
    ) -> Result<Self, LoadError> {
        Ok(Self {
            weight: register_keep_f32(res, &WeightId(format!("{prefix}.weight")))?,
            bias: register_keep_f32(res, &WeightId(format!("{prefix}.bias")))?,
            cout: cout as u32,
            k: k as u32,
            stride: 1,
            dilation,
            pad: dilation * (k as u32 - 1) / 2,
            transposed: None,
        })
    }

    /// `WNConvTranspose1d(k = 2s, stride s, padding ceil(s/2), output_padding
    /// s % 2)`: exactly `s` samples out per sample in.
    fn up<S: WeightSource>(
        res: &WeightResidency<S>,
        prefix: &str,
        cout: usize,
        s: usize,
    ) -> Result<Self, LoadError> {
        Ok(Self {
            weight: register_keep_f32(res, &WeightId(format!("{prefix}.weight")))?,
            bias: register_keep_f32(res, &WeightId(format!("{prefix}.bias")))?,
            cout: cout as u32,
            k: 2 * s as u32,
            stride: s as u32,
            dilation: 1,
            pad: s.div_ceil(2) as u32,
            transposed: Some((s % 2) as u32),
        })
    }

    async fn acquire<'r, S: WeightSource>(
        &self,
        res: &'r WeightResidency<S>,
        backend: &WgpuBackend,
        pins: &mut Vec<GpuView<'r>>,
    ) -> Result<(BufRef, BufRef), ResidencyError<S::Error, WgpuError>> {
        let w = res.acquire(self.weight, backend).await?;
        let b = res.acquire(self.bias, backend).await?;
        let out = (w.buf(), b.buf());
        pins.push(w);
        pins.push(b);
        Ok(out)
    }

    fn run<'w>(
        &self,
        scope: &thinfer_core::workspace::BatchScope<'w, WgpuBackend>,
        pl: &BigVganPipelines,
        x: Cl<'w>,
        w: BufRef,
        b: BufRef,
    ) -> Result<Cl<'w>, WgpuError> {
        match self.transposed {
            Some(op) => conv_transpose1d(
                scope,
                pl,
                x,
                w,
                b,
                self.cout,
                self.k,
                self.stride,
                self.pad,
                op,
                1,
            ),
            None => conv1d(
                scope,
                pl,
                x,
                w,
                b,
                self.cout,
                self.k,
                1,
                self.dilation,
                self.pad,
                1,
            ),
        }
    }
}

/// Snake `alpha` (raw).
struct Snake(Vec<f32>);

struct Unit {
    snake1: Snake,
    conv1: Conv,
    snake2: Snake,
    conv2: Conv,
}

struct Block {
    snake: Snake,
    up: Conv,
    units: Vec<Unit>,
}

pub struct DacDecoder {
    post_quant: Conv,
    conv_in: Conv,
    blocks: Vec<Block>,
    snake_out: Snake,
    conv_out: Conv,
}

impl DacDecoder {
    pub async fn new<S: WeightSource>(
        residency: &WeightResidency<S>,
    ) -> Result<Self, DacError<S::Error>> {
        let src = residency.source();
        let snake = async |name: String| -> Result<Snake, DacError<S::Error>> {
            Ok(Snake(read_host(src, &name).await?))
        };
        let post_quant = Conv::plain(
            residency,
            &format!("{PREFIX}post_quant_conv"),
            LATENT_CH,
            1,
            1,
        )?;
        let conv_in = Conv::plain(residency, &format!("{PREFIX}decoder.model.0"), DIM, 7, 1)?;
        let mut blocks = Vec::with_capacity(RATES.len());
        let mut dim = DIM;
        for (i, &s) in RATES.iter().enumerate() {
            let p = format!("{PREFIX}decoder.model.{}.block", i + 1);
            let out = dim / 2;
            let mut units = Vec::with_capacity(DILATIONS.len());
            for (u, &dil) in DILATIONS.iter().enumerate() {
                let q = format!("{p}.{}.block", u + 2);
                units.push(Unit {
                    snake1: snake(format!("{q}.0.alpha")).await?,
                    conv1: Conv::plain(residency, &format!("{q}.1"), out, 7, dil)?,
                    snake2: snake(format!("{q}.2.alpha")).await?,
                    conv2: Conv::plain(residency, &format!("{q}.3"), out, 1, 1)?,
                });
            }
            blocks.push(Block {
                snake: snake(format!("{p}.0.alpha")).await?,
                up: Conv::up(residency, &format!("{p}.1"), out, s)?,
                units,
            });
            dim = out;
        }
        let last = RATES.len() + 1;
        Ok(Self {
            post_quant,
            conv_in,
            blocks,
            snake_out: snake(format!("{PREFIX}decoder.model.{last}.alpha")).await?,
            conv_out: Conv::plain(
                residency,
                &format!("{PREFIX}decoder.model.{}", last + 1),
                1,
                7,
                1,
            )?,
        })
    }

    /// `[LATENT_CH, frames]` latents to `frames * HOP` samples in [-1, 1].
    pub async fn decode<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        pl: &BigVganPipelines,
        latent: &[f32],
    ) -> Result<Vec<f32>, DacError<S::Error>> {
        let frames = latent.len() / LATENT_CH;
        let mut pins: Vec<GpuView> = Vec::new();
        let mut held: Vec<WsBuf<WgpuBackend>> = Vec::new();
        let mut upload = |v: &[f32]| -> Result<BufRef, WgpuError> {
            let buf = workspace.alloc((v.len() * 4) as u64)?;
            backend.write_buffer(buf.id(), 0, &act_upload_bytes(ActDtype::F32, v))?;
            let r = buf.as_buf_ref();
            held.push(buf);
            Ok(r)
        };

        let input = upload(latent)?;
        // With an accelerator covering the tower, every scope below runs on
        // it (its uploads land at each scope's `accel_begin` drain).
        let accel = accel_covers(&workspace.batch());
        let begin = |scope: &thinfer_core::workspace::BatchScope<'_, WgpuBackend>| {
            if accel {
                scope.accel_begin()?;
            }
            Ok::<(), WgpuError>(())
        };
        let (pw, pb) = self
            .post_quant
            .acquire(residency, backend, &mut pins)
            .await?;
        let (w, b) = self.conv_in.acquire(residency, backend, &mut pins).await?;
        let mut cur = {
            let scope = workspace.batch();
            begin(&scope)?;
            let x = Cl {
                buf: scope.import_copy(input),
                c: LATENT_CH as u32,
                l: frames as u32,
            };
            let y = self.post_quant.run(&scope, pl, x, pw, pb)?;
            let y = self.conv_in.run(&scope, pl, y, w, b)?;
            let p = persist_cl(&scope, workspace, y)?;
            scope.submit_void().await?;
            p
        };

        for block in &self.blocks {
            let a = upload(&block.snake.0)?;
            let (w, b) = block.up.acquire(residency, backend, &mut pins).await?;
            cur = {
                let scope = workspace.batch();
                begin(&scope)?;
                let x = cur.import(&scope);
                let y = snake1d(&scope, pl, x, a)?;
                let y = block.up.run(&scope, pl, y, w, b)?;
                let p = persist_cl(&scope, workspace, y)?;
                scope.submit_void().await?;
                p
            };
            for unit in &block.units {
                let a1 = upload(&unit.snake1.0)?;
                let a2 = upload(&unit.snake2.0)?;
                let (w1, b1) = unit.conv1.acquire(residency, backend, &mut pins).await?;
                let (w2, b2) = unit.conv2.acquire(residency, backend, &mut pins).await?;
                cur = {
                    let scope = workspace.batch();
                    begin(&scope)?;
                    let x = cur.import(&scope);
                    let y = snake1d(&scope, pl, x, a1)?;
                    let y = unit.conv1.run(&scope, pl, y, w1, b1)?;
                    let y = snake1d(&scope, pl, y, a2)?;
                    let y = unit.conv2.run(&scope, pl, y, w2, b2)?;
                    let y = add(&scope, pl, y, x)?;
                    let p = persist_cl(&scope, workspace, y)?;
                    scope.submit_void().await?;
                    p
                };
            }
        }

        let a = upload(&self.snake_out.0)?;
        let (w, b) = self.conv_out.acquire(residency, backend, &mut pins).await?;
        let mut out = {
            let scope = workspace.batch();
            begin(&scope)?;
            let x = cur.import(&scope);
            let y = snake1d(&scope, pl, x, a)?;
            let y = self.conv_out.run(&scope, pl, y, w, b)?;
            let p = persist_cl(&scope, workspace, y)?;
            scope.submit_void().await?;
            p.read(backend).await?
        };
        out.iter_mut().for_each(|v| *v = v.tanh());
        drop(cur);
        workspace.drain_pool();
        drop(held);
        drop(pins);
        Ok(out)
    }
}

/// A whole weight, decoded to host f32.
async fn read_host<S: WeightSource>(src: &S, name: &str) -> Result<Vec<f32>, DacError<S::Error>> {
    let err = |m: String| DacError::Weight(format!("{name}: {m}"));
    let id = WeightId(name.into());
    let enc = src
        .catalog()
        .get(&id)
        .and_then(|e| e.encoding)
        .ok_or_else(|| err("missing or undecodable".into()))?;
    let mut r = src.open(&id).await.map_err(|e| err(format!("{e:?}")))?;
    let mut bytes = vec![0u8; r.len() as usize];
    r.read_at(0, &mut bytes)
        .await
        .map_err(|e| err(format!("{e:?}")))?;
    let mut out = vec![0u8; bytes.len() * 2];
    let n = decode_into(enc, &bytes, &mut out).map_err(|e| err(format!("{e:?}")))?;
    Ok(out[..n]
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moss_sfx::HOP;

    #[test]
    fn rates_make_the_hop() {
        assert_eq!(RATES.iter().product::<usize>(), HOP);
    }

    /// `k = 2s`, `pad = ceil(s/2)`, `output_padding = s % 2`: `L * s` out.
    #[test]
    fn up_convs_multiply_exactly() {
        for s in RATES {
            let (k, pad, op) = (2 * s, s.div_ceil(2), s % 2);
            for l in [1usize, 7, 1500] {
                assert_eq!((l - 1) * s + k - 2 * pad + op, l * s, "stride {s}");
            }
        }
    }
}
