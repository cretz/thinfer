//! 24 kHz samples -> codes (the Mimi encoder of the 12 Hz codec), for
//! in-context voice cloning: a causal SEANet conv stack (ELU, residual blocks,
//! 4/5/6/8 strided downsampling, 64 -> 1024 channels) -> an 8-layer window-250
//! transformer (LayerNorm, GELU) -> a 2x causal downsample to 12.5 Hz -> split
//! RVQ nearest-centroid search (1 semantic + 15 acoustic codebooks, host).

use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::residency::WeightResidency;
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::Workspace;

use super::{Acq, Conv, Convs, Scope, Stage, Xf, XfDims, read_f32, transpose};
use crate::common::bigvgan::{self as bv, Cl};
use crate::llm::decoder::LlmError;
use crate::llm::kernels::LlmPipelines;
use crate::qwen3_tts::{CODE_GROUPS, CODEBOOK, SAMPLES_PER_FRAME};

const FILTERS: u32 = 64;
const RATIOS: [u32; 4] = [4, 5, 6, 8];
const HIDDEN: u32 = 512;
const CB_DIM: usize = 256;
const ACOUSTIC: usize = CODE_GROUPS - 1;
const XF: XfDims = XfDims {
    hidden: HIDDEN,
    heads: 8,
    head_dim: 64,
    ffn: 2048,
    window: 250,
    eps: 1e-5,
    theta: 10_000.0,
};

/// SEANet residual block: `x + conv2(elu(conv1(elu(x))))`.
struct ResBlock {
    conv1: Conv,
    conv2: Conv,
}

struct Level {
    res: ResBlock,
    down: Conv,
}

/// One RVQ: its input projection `[256, 512]` and codebooks `[n][2048][256]`
/// with their squared norms.
struct Rvq {
    proj: Vec<f32>,
    books: Vec<f32>,
    norms: Vec<f32>,
}

impl Rvq {
    async fn load<S: WeightSource>(src: &S, prefix: &str, n: usize) -> Result<Self, LlmError> {
        let proj = read_f32(src, &format!("{prefix}.input_proj.weight")).await?;
        let mut books = Vec::with_capacity(n * CODEBOOK as usize * CB_DIM);
        for i in 0..n {
            let p = format!("{prefix}.layers.{i}.codebook");
            let sum = read_f32(src, &format!("{p}.embed_sum")).await?;
            let usage = read_f32(src, &format!("{p}.cluster_usage")).await?;
            for (row, &u) in sum.chunks(CB_DIM).zip(&usage) {
                let d = u.max(1e-5);
                books.extend(row.iter().map(|v| v / d));
            }
        }
        let norms = books
            .chunks(CB_DIM)
            .map(|r| r.iter().map(|v| v * v).sum())
            .collect();
        Ok(Self { proj, books, norms })
    }

    /// Codes of `x` (`[512, T]`) into `out[f][g0..]` for each codebook in
    /// turn, subtracting each chosen centroid from the residual.
    fn encode(&self, x: &[f32], t: usize, out: &mut [[u32; CODE_GROUPS]], g0: usize) {
        let n = self.norms.len() / CODEBOOK as usize;
        let mut r = vec![0f32; t * CB_DIM];
        for o in 0..CB_DIM {
            let w = &self.proj[o * HIDDEN as usize..(o + 1) * HIDDEN as usize];
            for f in 0..t {
                r[f * CB_DIM + o] = (0..HIDDEN as usize).map(|i| w[i] * x[i * t + f]).sum();
            }
        }
        for (f, frame) in out.iter_mut().enumerate().take(t) {
            let v = &mut r[f * CB_DIM..(f + 1) * CB_DIM];
            for (q, code) in frame[g0..g0 + n].iter_mut().enumerate() {
                let base = q * CODEBOOK as usize;
                // argmin |v - e|^2 = argmin |e|^2 - 2 v.e
                let (best, _) = (0..CODEBOOK as usize)
                    .map(|c| {
                        let e = &self.books[(base + c) * CB_DIM..(base + c + 1) * CB_DIM];
                        let dot: f32 = v.iter().zip(e).map(|(a, b)| a * b).sum();
                        (c, self.norms[base + c] - 2.0 * dot)
                    })
                    .fold((0, f32::INFINITY), |m, x| if x.1 < m.1 { x } else { m });
                *code = best as u32;
                let e = &self.books[(base + best) * CB_DIM..(base + best + 1) * CB_DIM];
                v.iter_mut().zip(e).for_each(|(a, b)| *a -= b);
            }
        }
    }
}

pub struct CodecEncoder {
    conv_in: Conv,
    levels: Vec<Level>,
    conv_out: Conv,
    xf: Xf,
    down: Conv,
    semantic: Rvq,
    acoustic: Rvq,
    cv: Convs,
}

impl CodecEncoder {
    pub async fn load<S: WeightSource>(
        backend: &Arc<WgpuBackend>,
        res: &WeightResidency<S>,
    ) -> Result<Self, LlmError> {
        let p = "encoder.encoder.layers";
        let conv = |i: usize, sub: &str, cout, k, stride, dil| {
            Conv::new(
                res,
                &format!("{p}.{i}{sub}.conv"),
                cout,
                k,
                stride,
                dil,
                1,
                true,
            )
        };
        let mut levels = Vec::with_capacity(RATIOS.len());
        let mut idx = 1;
        for (j, &r) in RATIOS.iter().enumerate() {
            let dim = FILTERS << j;
            levels.push(Level {
                res: ResBlock {
                    conv1: conv(idx, ".block.1", dim / 2, 3, 1, 1)?,
                    conv2: conv(idx, ".block.3", dim, 1, 1, 1)?,
                },
                down: conv(idx + 2, "", dim * 2, 2 * r, r, 1)?,
            });
            idx += 3;
        }
        let q = "encoder.quantizer";
        Ok(Self {
            conv_in: conv(0, "", FILTERS, 7, 1, 1)?,
            levels,
            conv_out: conv(idx + 1, "", HIDDEN, 3, 1, 1)?,
            xf: Xf::load(
                backend,
                res,
                "encoder.encoder_transformer",
                8,
                true,
                false,
                XF,
            )
            .await?,
            down: Conv::new(res, "encoder.downsample.conv", HIDDEN, 4, 2, 1, 1, false)?.replicate(),
            semantic: Rvq::load(
                res.source(),
                &format!("{q}.semantic_residual_vector_quantizer"),
                1,
            )
            .await?,
            acoustic: Rvq::load(
                res.source(),
                &format!("{q}.acoustic_residual_vector_quantizer"),
                ACOUSTIC,
            )
            .await?,
            cv: Convs::new(backend, "qtts_codec_enc").await?,
        })
    }

    fn convs(&self) -> Vec<&Conv> {
        let mut v = vec![&self.conv_in, &self.conv_out, &self.down];
        for l in &self.levels {
            v.extend([&l.res.conv1, &l.res.conv2, &l.down]);
        }
        v
    }

    /// Codes `[frames][16]` of 24 kHz mono samples (`ceil(len / 1920)`
    /// frames, as the reference trims them). `trace` (parity) collects the
    /// SEANet, transformer and downsample outputs.
    pub async fn encode<S: WeightSource>(
        &self,
        backend: &Arc<WgpuBackend>,
        res: &WeightResidency<S>,
        p: &LlmPipelines,
        ws: &Workspace<WgpuBackend>,
        wav: &[f32],
        mut trace: Option<&mut Vec<(&'static str, Vec<f32>)>>,
    ) -> Result<Vec<[u32; CODE_GROUPS]>, LlmError> {
        if wav.is_empty() {
            return Err(LlmError::Weights("empty reference clip".into()));
        }
        let convs = self.convs();
        let (h, c, t) = self
            .cv
            .stages(
                backend,
                res,
                ws,
                &convs,
                wav,
                (1, wav.len() as u32),
                &self.seanet_stages(),
            )
            .await?;
        if let Some(tr) = trace.as_deref_mut() {
            tr.push(("seanet", h.clone()));
        }
        let rows = transpose(&h, c as usize, t as usize);
        let y = self
            .xf
            .run(backend, res, p, &self.cv, ws, &rows, t, None, None, None)
            .await?;
        if let Some(tr) = trace.as_deref_mut() {
            tr.push(("xf", y.clone()));
        }
        let cl = transpose(&y, t as usize, HIDDEN as usize);
        let down: Stage<'_> = Box::new(|s, x, a| self.cv.conv(s, &self.down, x, a));
        let (z, _, frames) = self
            .cv
            .stages(backend, res, ws, &convs, &cl, (HIDDEN, t), &[down])
            .await?;
        if let Some(tr) = trace {
            tr.push(("down", z.clone()));
        }
        let mut codes = self.quantize(&z, frames as usize);
        codes.truncate(wav.len().div_ceil(SAMPLES_PER_FRAME as usize));
        Ok(codes)
    }

    /// Split RVQ codes of the downsampled embeddings `z` (`[512, frames]`).
    pub fn quantize(&self, z: &[f32], frames: usize) -> Vec<[u32; CODE_GROUPS]> {
        let mut codes = vec![[0u32; CODE_GROUPS]; frames];
        self.semantic.encode(z, frames, &mut codes, 0);
        self.acoustic.encode(z, frames, &mut codes, 1);
        codes
    }

    fn elu<'s>(&self, s: &Scope<'s>, x: Cl<'s>) -> Result<Cl<'s>, LlmError> {
        self.cv.unary(s, &self.cv.glue.elu, x)
    }

    fn seanet_stages(&self) -> Vec<Stage<'_>> {
        let cv = &self.cv;
        let mut v: Vec<Stage<'_>> = vec![Box::new(|s, x, a| cv.conv(s, &self.conv_in, x, a))];
        for l in &self.levels {
            v.push(Box::new(move |s, x, a: &Acq<'_>| {
                let h = self.elu(s, x)?;
                let h = cv.conv(s, &l.res.conv1, h, a)?;
                let h = self.elu(s, h)?;
                let h = cv.conv(s, &l.res.conv2, h, a)?;
                let x = bv::add(s, &cv.bv, h, x)?;
                let h = self.elu(s, x)?;
                cv.conv(s, &l.down, h, a)
            }));
        }
        v.push(Box::new(|s, x, a| {
            let h = self.elu(s, x)?;
            cv.conv(s, &self.conv_out, h, a)
        }));
        v
    }
}
