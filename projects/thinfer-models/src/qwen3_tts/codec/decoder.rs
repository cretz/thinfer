//! Codes -> 24 kHz samples: RVQ lookup (host, f32) -> causal conv 512 ->
//! 1024 -> an 8-layer window-72 transformer (RMSNorm, SwiGLU) -> two causal
//! 2x transposed-conv + ConvNeXt upsamplers -> a SnakeBeta vocoder (1536 ->
//! 96 channels, rates 8/5/4/3). Long inputs decode in chunks of [`CHUNK`]
//! frames with [`LEFT_CONTEXT`] frames of left context, as the reference does.

use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::residency::WeightResidency;
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::Workspace;

use super::{
    Conv, Convs, Lin, Norm, Scope, Stage, Xf, XfDims, glue_u, mat_f32, read_f32, transpose,
};
use crate::common::bigvgan::{self as bv, Cl};
use crate::llm::decoder::{LlmError, OwnedBuf};
use crate::llm::kernels::LlmPipelines;
use crate::qwen3_tts::{CODE_GROUPS, CODEBOOK, SAMPLES_PER_FRAME};

pub const CHUNK: usize = 300;
pub const LEFT_CONTEXT: usize = 25;

const CB_DIM: usize = 256;
const CODE_DIM: usize = 512;
const LATENT: u32 = 1024;
const DEC_DIM: u32 = 1536;
const RATES: [u32; 4] = [8, 5, 4, 3];
const UPS: usize = 2;
const XF: XfDims = XfDims {
    hidden: 512,
    heads: 16,
    head_dim: 64,
    ffn: 1024,
    window: 72,
    eps: 1e-5,
    theta: 10_000.0,
};

struct Snake {
    alpha: OwnedBuf,
    beta: OwnedBuf,
}

struct ResUnit {
    act1: Snake,
    conv1: Conv,
    act2: Snake,
    conv2: Conv,
}

struct DecBlock {
    act: Snake,
    up: Conv,
    units: Vec<ResUnit>,
}

struct Upsample {
    up: Conv,
    dw: Conv,
    ln_w: OwnedBuf,
    ln_b: OwnedBuf,
    pw1: Conv,
    pw2: Conv,
    gamma: OwnedBuf,
}

pub struct CodecDecoder {
    /// `[16][2048][256]` normalized codebooks.
    codebooks: Vec<f32>,
    /// `[512, 256]` output projections (first codebook / the rest).
    proj_first: Vec<f32>,
    proj_rest: Vec<f32>,
    pre_conv: Conv,
    xf: Xf,
    in_proj: Lin,
    norm: Norm,
    out_proj: Lin,
    ups: Vec<Upsample>,
    dec_in: Conv,
    blocks: Vec<DecBlock>,
    act_out: Snake,
    dec_out: Conv,
    cv: Convs,
}

impl CodecDecoder {
    pub async fn load<S: WeightSource>(
        backend: &Arc<WgpuBackend>,
        res: &WeightResidency<S>,
    ) -> Result<Self, LlmError> {
        let src = res.source();
        let buf =
            |name: String| async move { OwnedBuf::f32s(backend, &read_f32(src, &name).await?) };
        let snake = |p: String| async move {
            Ok::<_, LlmError>(Snake {
                alpha: buf(format!("{p}.alpha")).await?,
                beta: buf(format!("{p}.beta")).await?,
            })
        };
        let conv = |p: &str, cout, k, stride, dil, groups| {
            Conv::new(res, p, cout, k, stride, dil, groups, true)
        };
        // Codebooks: embedding_sum / max(cluster_usage, eps).
        let mut codebooks = Vec::with_capacity(CODE_GROUPS * CODEBOOK as usize * CB_DIM);
        for g in 0..CODE_GROUPS {
            let p = if g == 0 {
                "decoder.quantizer.rvq_first.vq.layers.0._codebook".to_string()
            } else {
                format!("decoder.quantizer.rvq_rest.vq.layers.{}._codebook", g - 1)
            };
            let sum = read_f32(src, &format!("{p}.embedding_sum")).await?;
            let usage = read_f32(src, &format!("{p}.cluster_usage")).await?;
            for (row, &u) in sum.chunks(CB_DIM).zip(&usage) {
                let d = u.max(1e-5);
                codebooks.extend(row.iter().map(|v| v / d));
            }
        }
        let mut ups = Vec::with_capacity(UPS);
        for i in 0..UPS {
            let p = format!("decoder.upsample.{i}");
            ups.push(Upsample {
                up: conv(&format!("{p}.0.conv"), LATENT, 2, 2, 1, 1)?,
                dw: conv(&format!("{p}.1.dwconv.conv"), LATENT, 7, 1, 1, LATENT)?,
                ln_w: buf(format!("{p}.1.norm.weight")).await?,
                ln_b: buf(format!("{p}.1.norm.bias")).await?,
                pw1: conv(&format!("{p}.1.pwconv1"), 4 * LATENT, 1, 1, 1, 1)?,
                pw2: conv(&format!("{p}.1.pwconv2"), LATENT, 1, 1, 1, 1)?,
                gamma: buf(format!("{p}.1.gamma")).await?,
            });
        }
        let mut blocks = Vec::with_capacity(RATES.len());
        for (i, &rate) in RATES.iter().enumerate() {
            let p = format!("decoder.decoder.{}.block", i + 1);
            let out = DEC_DIM >> (i + 1);
            let mut units = Vec::with_capacity(3);
            for (j, dil) in [1u32, 3, 9].into_iter().enumerate() {
                let q = format!("{p}.{}", j + 2);
                units.push(ResUnit {
                    act1: snake(format!("{q}.act1")).await?,
                    conv1: conv(&format!("{q}.conv1.conv"), out, 7, 1, dil, 1)?,
                    act2: snake(format!("{q}.act2")).await?,
                    conv2: conv(&format!("{q}.conv2.conv"), out, 1, 1, 1, 1)?,
                });
            }
            blocks.push(DecBlock {
                act: snake(format!("{p}.0")).await?,
                up: conv(&format!("{p}.1.conv"), out, 2 * rate, rate, 1, 1)?,
                units,
            });
        }
        let proj = |n: &str| {
            let name = format!("decoder.quantizer.{n}.output_proj.weight");
            async move { read_f32(src, &name).await }
        };
        let lin = |n: &str, out: u32| {
            let n = n.to_string();
            async move {
                Ok::<_, LlmError>(Lin {
                    w: mat_f32(res, &format!("{n}.weight"))?,
                    bias: Some(read_f32(src, &format!("{n}.bias")).await?),
                    out,
                })
            }
        };
        Ok(Self {
            codebooks,
            proj_first: proj("rvq_first").await?,
            proj_rest: proj("rvq_rest").await?,
            pre_conv: conv("decoder.pre_conv.conv", LATENT, 3, 1, 1, 1)?,
            xf: Xf::load(backend, res, "decoder.pre_transformer", 8, false, true, XF).await?,
            in_proj: lin("decoder.pre_transformer.input_proj", XF.hidden).await?,
            norm: Norm::Rms(buf("decoder.pre_transformer.norm.weight".into()).await?),
            out_proj: lin("decoder.pre_transformer.output_proj", LATENT).await?,
            ups,
            dec_in: conv("decoder.decoder.0.conv", DEC_DIM, 7, 1, 1, 1)?,
            blocks,
            act_out: snake("decoder.decoder.5".into()).await?,
            dec_out: conv("decoder.decoder.6.conv", 1, 7, 1, 1, 1)?,
            cv: Convs::new(backend, "qtts_codec_dec").await?,
        })
    }

    fn convs(&self) -> Vec<&Conv> {
        let mut v = vec![&self.pre_conv, &self.dec_in, &self.dec_out];
        for u in &self.ups {
            v.extend([&u.up, &u.dw, &u.pw1, &u.pw2]);
        }
        for b in &self.blocks {
            v.push(&b.up);
            for u in &b.units {
                v.extend([&u.conv1, &u.conv2]);
            }
        }
        v
    }

    /// Samples for `codes` (`[frames][16]`), clamped to [-1, 1].
    pub async fn decode<S: WeightSource>(
        &self,
        backend: &Arc<WgpuBackend>,
        res: &WeightResidency<S>,
        p: &LlmPipelines,
        ws: &Workspace<WgpuBackend>,
        codes: &[[u32; CODE_GROUPS]],
    ) -> Result<Vec<f32>, LlmError> {
        let mut out = Vec::with_capacity(codes.len() * SAMPLES_PER_FRAME as usize);
        let mut start = 0;
        while start < codes.len() {
            let end = (start + CHUNK).min(codes.len());
            let ctx = if start > LEFT_CONTEXT {
                LEFT_CONTEXT
            } else {
                start
            };
            let wav = self
                .forward(backend, res, p, ws, &codes[start - ctx..end])
                .await?;
            out.extend_from_slice(&wav[ctx * SAMPLES_PER_FRAME as usize..]);
            start = end;
        }
        out.iter_mut().for_each(|v| *v = v.clamp(-1.0, 1.0));
        Ok(out)
    }

    /// RVQ: `[512, T]` (the first codebook's projection plus the rest's).
    fn quantized(&self, codes: &[[u32; CODE_GROUPS]]) -> Vec<f32> {
        let t = codes.len();
        let cb = |g: usize, c: u32| {
            let o = (g * CODEBOOK as usize + c as usize) * CB_DIM;
            &self.codebooks[o..o + CB_DIM]
        };
        let mut first = vec![0f32; t * CB_DIM];
        let mut rest = vec![0f32; t * CB_DIM];
        for (f, fr) in codes.iter().enumerate() {
            first[f * CB_DIM..(f + 1) * CB_DIM].copy_from_slice(cb(0, fr[0]));
            for (g, &c) in fr.iter().enumerate().skip(1) {
                rest[f * CB_DIM..(f + 1) * CB_DIM]
                    .iter_mut()
                    .zip(cb(g, c))
                    .for_each(|(o, v)| *o += v);
            }
        }
        let mut y = vec![0f32; CODE_DIM * t];
        for (proj, src) in [(&self.proj_first, &first), (&self.proj_rest, &rest)] {
            for co in 0..CODE_DIM {
                let w = &proj[co * CB_DIM..(co + 1) * CB_DIM];
                for f in 0..t {
                    let x = &src[f * CB_DIM..(f + 1) * CB_DIM];
                    y[co * t + f] += w.iter().zip(x).map(|(a, b)| a * b).sum::<f32>();
                }
            }
        }
        y
    }

    async fn forward<S: WeightSource>(
        &self,
        backend: &Arc<WgpuBackend>,
        res: &WeightResidency<S>,
        p: &LlmPipelines,
        ws: &Workspace<WgpuBackend>,
        codes: &[[u32; CODE_GROUPS]],
    ) -> Result<Vec<f32>, LlmError> {
        let t = codes.len() as u32;
        let convs = self.convs();
        let q = self.quantized(codes);
        let pre: Stage<'_> = Box::new(|s, x, a| self.cv.conv(s, &self.pre_conv, x, a));
        let (pre, _, _) = self
            .cv
            .stages(backend, res, ws, &convs, &q, (CODE_DIM as u32, t), &[pre])
            .await?;
        let rows = transpose(&pre, LATENT as usize, t as usize);
        let xf = self
            .xf
            .run(
                backend,
                res,
                p,
                &self.cv,
                ws,
                &rows,
                t,
                Some(&self.in_proj),
                Some(&self.norm),
                Some(&self.out_proj),
            )
            .await?;
        let cl = transpose(&xf, t as usize, LATENT as usize);
        let (wav, _, _) = self
            .cv
            .stages(
                backend,
                res,
                ws,
                &convs,
                &cl,
                (LATENT, t),
                &self.vocoder_stages(),
            )
            .await?;
        Ok(wav)
    }

    /// ConvNeXt upsampler: 2x causal transposed conv, then
    /// `x + gamma * pw2(gelu(pw1(ln(dwconv(x)))))`.
    fn upsample<'s>(
        &self,
        s: &Scope<'s>,
        u: &Upsample,
        x: Cl<'s>,
        a: &super::Acq<'_>,
    ) -> Result<Cl<'s>, LlmError> {
        let cv = &self.cv;
        let imp = |b: &OwnedBuf| s.import_copy(b.buf_ref());
        let x = cv.conv_up(s, &u.up, x, a)?;
        let h = cv.conv(s, &u.dw, x, a)?;
        let n = s.alloc(h.bytes())?;
        cv.dispatch(
            s,
            &cv.glue.ln,
            [h.buf, imp(&u.ln_w), imp(&u.ln_b), n],
            glue_u(h.n(), h.c, 1, 1e-6, h.l, 1, h.l),
            h.l,
        )?;
        let h1 = cv.conv(s, &u.pw1, Cl { buf: n, ..h }, a)?;
        let g = cv.unary(s, &cv.glue.gelu, h1)?;
        let h2 = cv.conv(s, &u.pw2, g, a)?;
        let o = s.alloc(h2.bytes())?;
        cv.dispatch(
            s,
            &cv.glue.scale_add,
            [h2.buf, imp(&u.gamma), x.buf, o],
            glue_u(h2.n(), h2.c, h2.l, 0.0, 0, 0, 0),
            h2.n(),
        )?;
        Ok(Cl { buf: o, ..h2 })
    }

    fn snake<'s>(&self, s: &Scope<'s>, a: &Snake, x: Cl<'s>) -> Result<Cl<'s>, LlmError> {
        Ok(bv::snake_beta(
            s,
            &self.cv.bv,
            x,
            a.alpha.buf_ref(),
            a.beta.buf_ref(),
        )?)
    }

    /// The vocoder as scope-sized stages: the upsamplers, the input conv,
    /// each decoder block's upsample and residual units, the output. From
    /// the input conv on, a stage runs on the accelerator when it covers the
    /// tower (the upsamplers' depthwise convs and glue stay on wgpu).
    fn vocoder_stages(&self) -> Vec<Stage<'_>> {
        let cv = &self.cv;
        let accel = |s: &Scope<'_>| -> Result<(), LlmError> {
            if bv::accel_covers(s) {
                s.accel_begin()?;
            }
            Ok(())
        };
        let mut v: Vec<Stage<'_>> = Vec::new();
        for u in &self.ups {
            v.push(Box::new(move |s, x, a| self.upsample(s, u, x, a)));
        }
        v.push(Box::new(move |s, x, a| {
            accel(s)?;
            cv.conv(s, &self.dec_in, x, a)
        }));
        for blk in &self.blocks {
            v.push(Box::new(move |s, x, a| {
                accel(s)?;
                let h = self.snake(s, &blk.act, x)?;
                cv.conv_up(s, &blk.up, h, a)
            }));
            for u in &blk.units {
                v.push(Box::new(move |s, x, a| {
                    accel(s)?;
                    let h = self.snake(s, &u.act1, x)?;
                    let h = cv.conv(s, &u.conv1, h, a)?;
                    let h = self.snake(s, &u.act2, h)?;
                    let h = cv.conv(s, &u.conv2, h, a)?;
                    Ok(bv::add(s, &cv.bv, h, x)?)
                }));
            }
        }
        v.push(Box::new(move |s, x, a| {
            accel(s)?;
            let h = self.snake(s, &self.act_out, x)?;
            cv.conv(s, &self.dec_out, h, a)
        }));
        v
    }
}
