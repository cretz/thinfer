//! MiniMax-H3 audio VAE ENCODER: waveform to normalized latents, for audio
//! references (Ref2VA voice and sound conditioning).
//!
//! Reference: ComfyUI `comfy/ldm/minimax/audio_vae.py`, `MiniMaxH3AudioVAE.encode`.
//!
//! - A DAC encoder tower (Snake1d, dilated residual units, strided
//!   downsampling `2, 4, 4, 5, 5` = the 800-sample hop) runs on the GPU with
//!   the BigVGAN kernels, one submit per residual unit so no submit grows with
//!   the clip.
//! - The `pre_block` attention projection (2048 -> 32 channels) and
//!   `mean_proj` run on the host: 40 rows per second of audio, a few hundred
//!   at most, cheaper than another pipeline set.
//! - Encoding returns the posterior MEAN (no sampling), normalized with the
//!   checkpoint's `latents_mean` / `latents_std`, as the reference does.
//! - Like the decoder, the model is mono: each stereo channel is its own run.

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError};
use thinfer_core::ops::ActDtype;
use thinfer_core::residency::{GpuView, ResidencyError, WeightHandle, WeightResidency};
use thinfer_core::weight::{WeightId, WeightReader, WeightSource, decode_into};
use thinfer_core::workspace::{Workspace, WsBuf};

use super::audio_vae::AudioVaeConfig;
use crate::common::bigvgan::{BigVganPipelines, Cl, add, conv1d, persist_cl, snake_beta};
use crate::common::loader::{LoadError, register_passthrough};
use crate::common::seq::{act_readback_to_f32, act_upload_bytes, par_map};

/// Downsampling strides; their product is the hop.
const RATES: [usize; 5] = [2, 4, 4, 5, 5];
/// Channels after `block.0`; each stage doubles them.
const BASE_DIM: usize = 64;
/// The three residual units of every stage.
const DILATIONS: [u32; 3] = [1, 3, 9];
/// Width the tower ends at (`latent_dim`, the attention projection's input).
const TOWER_DIM: usize = 2048;
const HEADS: usize = 8;
const LN_EPS: f32 = 1e-5;
/// Host matmul worker cap: the user's machine is shared.
const MAX_THREADS: usize = 8;

#[derive(Debug)]
pub enum AudioEncoderError<SE: core::fmt::Debug> {
    Wgpu(WgpuError),
    Load(LoadError),
    Residency(ResidencyError<SE, WgpuError>),
    /// A host-read weight is missing, undecodable or out of its range.
    Weight(String),
}

impl<SE: core::fmt::Debug> From<WgpuError> for AudioEncoderError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}
impl<SE: core::fmt::Debug> From<LoadError> for AudioEncoderError<SE> {
    fn from(e: LoadError) -> Self {
        Self::Load(e)
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for AudioEncoderError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

/// A GPU conv layer: streamed weights plus its geometry.
struct Conv {
    weight: WeightHandle,
    bias: WeightHandle,
    cout: u32,
    k: u32,
    stride: u32,
    dilation: u32,
    pad: u32,
}

impl Conv {
    fn register<S: WeightSource>(
        res: &WeightResidency<S>,
        prefix: &str,
        cout: usize,
        k: usize,
        stride: usize,
        dilation: u32,
        pad: usize,
    ) -> Result<Self, LoadError> {
        Ok(Self {
            weight: register_passthrough(res, &WeightId(format!("{prefix}.weight")))?,
            bias: register_passthrough(res, &WeightId(format!("{prefix}.bias")))?,
            cout: cout as u32,
            k: k as u32,
            stride: stride as u32,
            dilation,
            pad: pad as u32,
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
}

/// `Snake1d`: `x + 1/(alpha + eps) * sin(alpha * x)^2`. That is SnakeBeta with
/// `alpha == beta`, stored log-scale, so the kernel is shared; every released
/// alpha is positive (checked at load).
struct Snake {
    log_alpha: Vec<f32>,
}

struct Unit {
    snake1: Snake,
    conv1: Conv,
    snake2: Snake,
    conv2: Conv,
}

struct Stage {
    units: Vec<Unit>,
    snake: Snake,
    down: Conv,
}

/// `y = x W^T + b`, `W` row-major `[n_out, n_in]`.
struct Linear {
    w: Vec<f32>,
    b: Vec<f32>,
    n_in: usize,
}

impl Linear {
    fn n_out(&self) -> usize {
        self.b.len()
    }

    /// `rows` rows of `x`, split across host threads.
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let rows = x.len() / self.n_in;
        let n_out = self.n_out();
        let threads = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .clamp(1, MAX_THREADS)
            .min(rows.max(1));
        let per = rows.div_ceil(threads);
        par_map(threads, |t| {
            let lo = (t * per).min(rows);
            let hi = ((t + 1) * per).min(rows);
            let mut out = Vec::with_capacity((hi - lo) * n_out);
            for r in lo..hi {
                let xr = &x[r * self.n_in..][..self.n_in];
                for (o, b) in self.b.iter().enumerate() {
                    out.push(b + dot(xr, &self.w[o * self.n_in..][..self.n_in]));
                }
            }
            out
        })
        .concat()
    }
}

struct LayerNorm {
    g: Vec<f32>,
    b: Vec<f32>,
}

impl LayerNorm {
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let d = self.g.len();
        let mut out = Vec::with_capacity(x.len());
        for row in x.chunks_exact(d) {
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
            let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
            let inv = 1.0 / (var + LN_EPS as f64).sqrt();
            for ((&v, g), b) in row.iter().zip(&self.g).zip(&self.b) {
                out.push(((v as f64 - mean) * inv) as f32 * g + b);
            }
        }
        out
    }
}

/// `pre_block` (`AttnProjection`) + `mean_proj` + normalization, host f32.
struct Head {
    norm1: LayerNorm,
    /// `qkv` with the reference's `cat(q_bias, zero_k_bias, v_bias)` bias.
    qkv: Linear,
    attn_proj: Linear,
    norm3: LayerNorm,
    proj: Linear,
    norm2: LayerNorm,
    mlp_norm: LayerNorm,
    w0: Linear,
    w1: Linear,
    w2: Linear,
    mean_proj: Linear,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
}

impl Head {
    /// `x`: `[t, TOWER_DIM]` row-major. Returns `[t, latent_channels]`.
    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let d = TOWER_DIM;
        let hd = d / HEADS;
        let out_dim = self.attn_proj.n_in;
        let bin = hd / out_dim;

        // Causal attention, then the mean over heads and the adaptive average
        // pool `hd -> out_dim` (exact bins of `bin`) in one reduction.
        let qkv = self.qkv.apply(&self.norm1.apply(x));
        let scale = 1.0 / (hd as f32).sqrt();
        let heads = par_map(HEADS, |h| {
            let mut out = vec![0f32; t * hd];
            let mut s = vec![0f32; t];
            for i in 0..t {
                let q = &qkv[i * 3 * d + h * hd..][..hd];
                let mut m = f32::NEG_INFINITY;
                for (j, sj) in s.iter_mut().enumerate().take(i + 1) {
                    *sj = dot(q, &qkv[j * 3 * d + d + h * hd..][..hd]) * scale;
                    m = m.max(*sj);
                }
                let mut sum = 0f32;
                for sj in s.iter_mut().take(i + 1) {
                    *sj = (*sj - m).exp();
                    sum += *sj;
                }
                let o = &mut out[i * hd..][..hd];
                for (j, sj) in s.iter().enumerate().take(i + 1) {
                    let p = sj / sum;
                    let v = &qkv[j * 3 * d + 2 * d + h * hd..][..hd];
                    for (oe, ve) in o.iter_mut().zip(v) {
                        *oe += p * ve;
                    }
                }
            }
            out
        });
        let mut pooled = vec![0f32; t * out_dim];
        let denom = (HEADS * bin) as f32;
        for (i, row) in pooled.chunks_exact_mut(out_dim).enumerate() {
            for (c, p) in row.iter_mut().enumerate() {
                let mut acc = 0f32;
                for head in &heads {
                    acc += head[i * hd + c * bin..][..bin].iter().sum::<f32>();
                }
                *p = acc / denom;
            }
        }

        let attn = self.attn_proj.apply(&pooled);
        let mut y = self.proj.apply(&self.norm3.apply(x));
        y.iter_mut().zip(&attn).for_each(|(a, b)| *a += b);

        let m = self.mlp_norm.apply(&self.norm2.apply(&y));
        let mut g = self.w0.apply(&m);
        g.iter_mut()
            .zip(self.w1.apply(&m))
            .for_each(|(a, b)| *a = gelu_tanh(*a) * b);
        y.iter_mut()
            .zip(self.w2.apply(&g))
            .for_each(|(a, b)| *a += b);

        let mut z = self.mean_proj.apply(&y);
        let c = self.latents_mean.len();
        for row in z.chunks_exact_mut(c) {
            for ((v, mean), std) in row
                .iter_mut()
                .zip(&self.latents_mean)
                .zip(&self.latents_std)
            {
                *v = (*v - mean) / std;
            }
        }
        z
    }
}

pub struct AudioEncoder {
    cfg: AudioVaeConfig,
    conv_in: Conv,
    stages: Vec<Stage>,
    snake_out: Snake,
    conv_out: Conv,
    head: Head,
}

impl AudioEncoder {
    pub async fn new<S: WeightSource>(
        residency: &WeightResidency<S>,
        cfg: AudioVaeConfig,
    ) -> Result<Self, AudioEncoderError<S::Error>> {
        let src = residency.source();
        let snake = async |name: String| -> Result<Snake, AudioEncoderError<S::Error>> {
            let alpha = read_host(src, &name).await?;
            if let Some(bad) = alpha.iter().find(|a| a.is_nan() || **a <= 0.0) {
                return Err(AudioEncoderError::Weight(format!(
                    "{name}: alpha {bad} is not positive"
                )));
            }
            Ok(Snake {
                log_alpha: alpha.iter().map(|a| a.ln()).collect(),
            })
        };
        let linear = async |name: &str| -> Result<Linear, AudioEncoderError<S::Error>> {
            let b = read_host(src, &format!("{name}.bias")).await?;
            let w = read_host(src, &format!("{name}.weight")).await?;
            Ok(Linear {
                n_in: w.len() / b.len(),
                w,
                b,
            })
        };
        let norm = async |name: &str| -> Result<LayerNorm, AudioEncoderError<S::Error>> {
            Ok(LayerNorm {
                g: read_host(src, &format!("{name}.weight")).await?,
                b: read_host(src, &format!("{name}.bias")).await?,
            })
        };

        let conv_in = Conv::register(residency, "encoder.block.0", BASE_DIM, 7, 1, 1, 3)?;
        let mut stages = Vec::with_capacity(RATES.len());
        let mut dim = BASE_DIM;
        for (i, &stride) in RATES.iter().enumerate() {
            let p = format!("encoder.block.{}.block", i + 1);
            let mut units = Vec::with_capacity(DILATIONS.len());
            for (u, &dil) in DILATIONS.iter().enumerate() {
                let q = format!("{p}.{u}.block");
                units.push(Unit {
                    snake1: snake(format!("{q}.0.alpha")).await?,
                    conv1: Conv::register(
                        residency,
                        &format!("{q}.1"),
                        dim,
                        7,
                        1,
                        dil,
                        3 * dil as usize,
                    )?,
                    snake2: snake(format!("{q}.2.alpha")).await?,
                    conv2: Conv::register(residency, &format!("{q}.3"), dim, 1, 1, 1, 0)?,
                });
            }
            let n = DILATIONS.len();
            stages.push(Stage {
                units,
                snake: snake(format!("{p}.{n}.alpha")).await?,
                down: Conv::register(
                    residency,
                    &format!("{p}.{}", n + 1),
                    2 * dim,
                    2 * stride,
                    stride,
                    1,
                    stride.div_ceil(2),
                )?,
            });
            dim *= 2;
        }
        assert_eq!(dim, TOWER_DIM, "the tower ends at the latent width");
        let last = RATES.len() + 1;
        let snake_out = snake(format!("encoder.block.{last}.alpha")).await?;
        let conv_out = Conv::register(
            residency,
            &format!("encoder.block.{}", last + 1),
            TOWER_DIM,
            3,
            1,
            1,
            1,
        )?;

        let q_bias = read_host(src, "pre_block.attn.q_bias").await?;
        let k_bias = read_host(src, "pre_block.attn.zero_k_bias").await?;
        let v_bias = read_host(src, "pre_block.attn.v_bias").await?;
        let head = Head {
            norm1: norm("pre_block.norm1").await?,
            qkv: Linear {
                w: read_host(src, "pre_block.attn.qkv.weight").await?,
                b: [q_bias, k_bias, v_bias].concat(),
                n_in: TOWER_DIM,
            },
            attn_proj: linear("pre_block.attn.proj").await?,
            norm3: norm("pre_block.norm3").await?,
            proj: linear("pre_block.proj").await?,
            norm2: norm("pre_block.norm2").await?,
            mlp_norm: norm("pre_block.mlp.norm").await?,
            w0: linear("pre_block.mlp.w0").await?,
            w1: linear("pre_block.mlp.w1").await?,
            w2: linear("pre_block.mlp.w2").await?,
            mean_proj: linear("mean_proj").await?,
            latents_mean: read_host(src, "latents_mean").await?,
            latents_std: read_host(src, "latents_std").await?,
        };
        let c = cfg.latent_channels;
        let shapes_ok = head.qkv.w.len() == 3 * TOWER_DIM * TOWER_DIM
            && head.proj.n_out() == c
            && head.attn_proj.n_out() == c
            && (TOWER_DIM / HEADS).is_multiple_of(head.attn_proj.n_in)
            && head.mean_proj.n_out() == c
            && head.latents_mean.len() == c
            && head.latents_std.len() == c;
        if !shapes_ok {
            return Err(AudioEncoderError::Weight(
                "pre_block / mean_proj shapes do not match the H3 audio VAE".into(),
            ));
        }
        Ok(Self {
            cfg,
            conv_in,
            stages,
            snake_out,
            conv_out,
            head,
        })
    }

    pub fn cfg(&self) -> &AudioVaeConfig {
        &self.cfg
    }

    /// Latent frames for `samples` waveform samples (right-padded to the hop).
    pub fn num_latents(&self, samples: usize) -> usize {
        samples.div_ceil(self.cfg.hop_length())
    }

    /// Encode `[channels, samples]` planar 32 kHz audio in `[-1, 1]`. Returns
    /// the normalized latents as DiT rows, `[channels * t, latent_channels]`
    /// channel-major (channel 0's `t` rows, then channel 1's), and `t`.
    pub async fn encode<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        pipelines: &BigVganPipelines,
        wave: &[f32],
        samples: usize,
    ) -> Result<(Vec<f32>, usize), AudioEncoderError<S::Error>> {
        let ch = self.cfg.channels;
        assert_eq!(wave.len(), ch * samples, "audio encoder input size");
        let t = self.num_latents(samples);
        let mut rows = Vec::with_capacity(ch * t * self.cfg.latent_channels);
        for s in 0..ch {
            let mut x = wave[s * samples..][..samples].to_vec();
            x.resize(t * self.cfg.hop_length(), 0.0);
            let y = self
                .tower(backend, residency, workspace, pipelines, &x)
                .await?;
            debug_assert_eq!(y.len(), TOWER_DIM * t);
            // `[TOWER_DIM, t]` -> `[t, TOWER_DIM]`.
            let mut xt = vec![0f32; y.len()];
            for (c, row) in y.chunks_exact(t).enumerate() {
                for (i, &v) in row.iter().enumerate() {
                    xt[i * TOWER_DIM + c] = v;
                }
            }
            rows.extend(self.head.forward(&xt, t));
        }
        Ok((rows, t))
    }

    /// The DAC tower over one mono channel (length a multiple of the hop):
    /// `[TOWER_DIM, len / hop]` host f32. Public for parity taps.
    pub async fn tower<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        pl: &BigVganPipelines,
        x: &[f32],
    ) -> Result<Vec<f32>, AudioEncoderError<S::Error>> {
        let mut pins: Vec<GpuView> = Vec::new();
        let mut held: Vec<WsBuf<WgpuBackend>> = Vec::new();
        let mut upload = |v: &[f32]| -> Result<BufRef, WgpuError> {
            let buf = workspace.alloc((v.len() * 4) as u64)?;
            backend.write_buffer(buf.id(), 0, &act_upload_bytes(ActDtype::F32, v))?;
            let r = buf.as_buf_ref();
            held.push(buf);
            Ok(r)
        };

        let input = upload(x)?;
        let (w, b) = self.conv_in.acquire(residency, backend, &mut pins).await?;
        let mut cur = {
            let scope = workspace.batch();
            let x = Cl {
                buf: scope.import_copy(input),
                c: 1,
                l: x.len() as u32,
            };
            let y = run_conv(&scope, pl, x, &self.conv_in, w, b)?;
            let p = persist_cl(&scope, workspace, y)?;
            scope.submit_void().await?;
            p
        };

        for stage in &self.stages {
            for unit in &stage.units {
                let a1 = upload(&unit.snake1.log_alpha)?;
                let a2 = upload(&unit.snake2.log_alpha)?;
                let (w1, b1) = unit.conv1.acquire(residency, backend, &mut pins).await?;
                let (w2, b2) = unit.conv2.acquire(residency, backend, &mut pins).await?;
                cur = {
                    let scope = workspace.batch();
                    let x = cur.import(&scope);
                    let y = snake_beta(&scope, pl, x, a1, a1)?;
                    let y = run_conv(&scope, pl, y, &unit.conv1, w1, b1)?;
                    let y = snake_beta(&scope, pl, y, a2, a2)?;
                    let y = run_conv(&scope, pl, y, &unit.conv2, w2, b2)?;
                    let y = add(&scope, pl, y, x)?;
                    let p = persist_cl(&scope, workspace, y)?;
                    scope.submit_void().await?;
                    p
                };
            }
            let a = upload(&stage.snake.log_alpha)?;
            let (w, b) = stage.down.acquire(residency, backend, &mut pins).await?;
            cur = {
                let scope = workspace.batch();
                let x = cur.import(&scope);
                let y = snake_beta(&scope, pl, x, a, a)?;
                let y = run_conv(&scope, pl, y, &stage.down, w, b)?;
                let p = persist_cl(&scope, workspace, y)?;
                scope.submit_void().await?;
                p
            };
        }

        let a = upload(&self.snake_out.log_alpha)?;
        let (w, b) = self.conv_out.acquire(residency, backend, &mut pins).await?;
        let out = {
            let scope = workspace.batch();
            let x = cur.import(&scope);
            let y = snake_beta(&scope, pl, x, a, a)?;
            let y = run_conv(&scope, pl, y, &self.conv_out, w, b)?;
            let out = workspace.alloc(y.bytes())?;
            let dst = scope.import_copy(out.as_buf_ref());
            scope.copy_buffer_to_buffer(y.buf, 0, dst, 0, y.bytes())?;
            scope.submit_void().await?;
            let bytes = backend.read_buffer(out.id(), 0, y.bytes()).await?;
            act_readback_to_f32(ActDtype::F32, &bytes, y.n() as usize)
        };
        drop(cur);
        workspace.drain_pool();
        drop(held);
        drop(pins);
        Ok(out)
    }
}

fn run_conv<'w>(
    scope: &thinfer_core::workspace::BatchScope<'w, WgpuBackend>,
    pl: &BigVganPipelines,
    x: Cl<'w>,
    c: &Conv,
    w: BufRef,
    b: BufRef,
) -> Result<Cl<'w>, WgpuError> {
    conv1d(
        scope, pl, x, w, b, c.cout, c.k, c.stride, c.dilation, c.pad, 1,
    )
}

/// A whole weight, decoded to host f32.
async fn read_host<S: WeightSource>(
    src: &S,
    name: &str,
) -> Result<Vec<f32>, AudioEncoderError<S::Error>> {
    let err = |m: String| AudioEncoderError::Weight(format!("{name}: {m}"));
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
    // Any float encoding decodes to at most 2x its bytes as f32.
    let mut out = vec![0u8; bytes.len() * 2];
    let n = decode_into(enc, &bytes, &mut out).map_err(|e| err(format!("{e:?}")))?;
    Ok(out[..n]
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ac, bc) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = ac
        .remainder()
        .iter()
        .zip(bc.remainder())
        .map(|(x, y)| x * y)
        .sum();
    for (x, y) in ac.zip(bc) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// `GELU(approximate="tanh")`.
fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2 / pi)
    0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The strides must multiply to the decoder's hop, or the latent rate
    /// (and with it the audio/video clock) drifts.
    #[test]
    fn strides_make_the_hop() {
        assert_eq!(
            RATES.iter().product::<usize>(),
            AudioVaeConfig::h3().hop_length()
        );
        assert_eq!(BASE_DIM << RATES.len(), TOWER_DIM);
        assert_eq!(TOWER_DIM, AudioVaeConfig::h3().latent_dim);
    }

    /// Every strided conv (`k = 2s`, `pad = ceil(s/2)`) divides the length
    /// exactly, so a hop-aligned input lands on whole latents.
    #[test]
    fn strided_convs_divide_exactly() {
        let mut l = 800 * 7;
        for &s in &RATES {
            let next = (l + 2 * s.div_ceil(2) - 2 * s) / s + 1;
            assert_eq!(next * s, l, "stride {s}");
            l = next;
        }
        assert_eq!(l, 7);
    }

    #[test]
    fn layer_norm_and_linear_match_hand_values() {
        let ln = LayerNorm {
            g: vec![1.0, 2.0],
            b: vec![0.0, 1.0],
        };
        let y = ln.apply(&[1.0, 3.0]);
        assert!((y[0] + 1.0).abs() < 1e-3 && (y[1] - 3.0).abs() < 1e-3);
        let lin = Linear {
            w: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            b: vec![0.5, -0.5],
            n_in: 3,
        };
        assert_eq!(
            lin.apply(&[1.0, 1.0, 1.0, 0.0, 1.0, 0.0]),
            [6.5, 14.5, 2.5, 4.5]
        );
    }
}
