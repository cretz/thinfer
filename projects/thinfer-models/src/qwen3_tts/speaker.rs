//! Speaker x-vector of a reference clip (Base checkpoints): a 128-bin log
//! mel at 24 kHz into an ECAPA-TDNN, on the host in f32 (one short clip per
//! job; every conv is a contiguous axpy over time, which vectorizes).

use thinfer_core::tensor::StorageEncoding;
use thinfer_core::weight::{WeightId, WeightReader, WeightSource};

pub const SAMPLE_RATE: u32 = 24_000;
const N_FFT: usize = 1024;
const HOP: usize = 256;
const N_MELS: usize = 128;
const F_MAX: f64 = 12_000.0;
const CH: usize = 512;
const CH_CAT: usize = 1536;
const SCALE: usize = 8;
pub const DIM: usize = 2048;

/// `[Cout, Cin / groups, K]` + bias, with its dilation.
struct Conv {
    w: Vec<f32>,
    b: Vec<f32>,
    cout: usize,
    cin: usize,
    k: usize,
    dil: usize,
}

impl Conv {
    /// `same` conv with reflect padding over `x` (`[cin, t]`), then ReLU when
    /// `relu`.
    fn run(&self, x: &[f32], t: usize, relu: bool) -> Vec<f32> {
        let pad = self.dil * (self.k - 1) / 2;
        let tp = t + 2 * pad;
        let mut xp = vec![0f32; self.cin * tp];
        for c in 0..self.cin {
            let src = &x[c * t..(c + 1) * t];
            let dst = &mut xp[c * tp..(c + 1) * tp];
            dst[pad..pad + t].copy_from_slice(src);
            for j in 0..pad {
                dst[pad - 1 - j] = src[(j + 1).min(t - 1)];
                dst[pad + t + j] = src[t.saturating_sub(2 + j)];
            }
        }
        let mut y = vec![0f32; self.cout * t];
        for o in 0..self.cout {
            let row = &mut y[o * t..(o + 1) * t];
            row.fill(self.b[o]);
            for i in 0..self.cin {
                for k in 0..self.k {
                    let wv = self.w[(o * self.cin + i) * self.k + k];
                    let xs = &xp[i * tp + k * self.dil..i * tp + k * self.dil + t];
                    row.iter_mut().zip(xs).for_each(|(r, v)| *r += wv * v);
                }
            }
            if relu {
                row.iter_mut().for_each(|v| *v = v.max(0.0));
            }
        }
        y
    }
}

struct SeRes2Block {
    tdnn1: Conv,
    res2: Vec<Conv>,
    tdnn2: Conv,
    se1: Conv,
    se2: Conv,
}

pub struct SpeakerEncoder {
    block0: Conv,
    blocks: Vec<SeRes2Block>,
    mfa: Conv,
    asp_tdnn: Conv,
    asp_conv: Conv,
    fc: Conv,
    mel_basis: Vec<f32>,
    window: Vec<f32>,
}

async fn tensor<S: WeightSource>(src: &S, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
    let id = WeightId(name.to_string());
    let e = src
        .catalog()
        .get(&id)
        .cloned()
        .ok_or_else(|| format!("missing tensor {name}"))?;
    let mut r = src.open(&id).await.map_err(|e| format!("{e:?}"))?;
    let mut b = vec![0u8; e.size as usize];
    r.read_at(0, &mut b).await.map_err(|e| format!("{e:?}"))?;
    let v = match e.encoding {
        Some(StorageEncoding::Bf16) => b
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        Some(StorageEncoding::F32) => bytemuck::cast_slice(&b).to_vec(),
        other => return Err(format!("{name}: unsupported encoding {other:?}")),
    };
    Ok((v, e.shape.0.clone()))
}

async fn conv<S: WeightSource>(src: &S, prefix: &str, dil: usize) -> Result<Conv, String> {
    let (w, s) = tensor(src, &format!("{prefix}.weight")).await?;
    let (b, _) = tensor(src, &format!("{prefix}.bias")).await?;
    Ok(Conv {
        w,
        b,
        cout: s[0],
        cin: s[1],
        k: s[2],
        dil,
    })
}

impl SpeakerEncoder {
    pub async fn load<S: WeightSource>(src: &S) -> Result<Self, String> {
        let p = "speaker_encoder";
        let mut blocks = Vec::new();
        for (i, dil) in [(1usize, 2usize), (2, 3), (3, 4)] {
            let q = format!("{p}.blocks.{i}");
            let mut res2 = Vec::new();
            for j in 0..SCALE - 1 {
                res2.push(conv(src, &format!("{q}.res2net_block.blocks.{j}.conv"), dil).await?);
            }
            blocks.push(SeRes2Block {
                tdnn1: conv(src, &format!("{q}.tdnn1.conv"), 1).await?,
                res2,
                tdnn2: conv(src, &format!("{q}.tdnn2.conv"), 1).await?,
                se1: conv(src, &format!("{q}.se_block.conv1"), 1).await?,
                se2: conv(src, &format!("{q}.se_block.conv2"), 1).await?,
            });
        }
        Ok(Self {
            block0: conv(src, &format!("{p}.blocks.0.conv"), 1).await?,
            blocks,
            mfa: conv(src, &format!("{p}.mfa.conv"), 1).await?,
            asp_tdnn: conv(src, &format!("{p}.asp.tdnn.conv"), 1).await?,
            asp_conv: conv(src, &format!("{p}.asp.conv"), 1).await?,
            fc: conv(src, &format!("{p}.fc"), 1).await?,
            mel_basis: mel_basis(),
            window: (0..N_FFT)
                .map(|i| {
                    (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos())
                        as f32
                })
                .collect(),
        })
    }

    /// Log mel `[128, frames]` of 24 kHz mono samples (reference
    /// `mel_spectrogram`: reflect pad (n_fft - hop) / 2, no centering,
    /// magnitude with 1e-9 inside the sqrt, log of max(x, 1e-5)).
    pub fn mel(&self, wav: &[f32]) -> (Vec<f32>, usize) {
        let pad = (N_FFT - HOP) / 2;
        let n = wav.len();
        let mut y = vec![0f32; n + 2 * pad];
        y[pad..pad + n].copy_from_slice(wav);
        for j in 0..pad {
            y[pad - 1 - j] = wav[(j + 1).min(n - 1)];
            y[pad + n + j] = wav[n.saturating_sub(2 + j)];
        }
        let frames = (y.len() - N_FFT) / HOP + 1;
        let bins = N_FFT / 2 + 1;
        // DFT basis rows (cos, sin) per bin.
        let mut cosb = vec![0f32; bins * N_FFT];
        let mut sinb = vec![0f32; bins * N_FFT];
        for k in 0..bins {
            for i in 0..N_FFT {
                let a = 2.0 * std::f64::consts::PI * (k * i) as f64 / N_FFT as f64;
                cosb[k * N_FFT + i] = a.cos() as f32;
                sinb[k * N_FFT + i] = a.sin() as f32;
            }
        }
        let mut mel = vec![0f32; N_MELS * frames];
        let mut frame = vec![0f32; N_FFT];
        let mut mag = vec![0f32; bins];
        for f in 0..frames {
            for (o, (&v, &w)) in frame
                .iter_mut()
                .zip(y[f * HOP..f * HOP + N_FFT].iter().zip(&self.window))
            {
                *o = v * w;
            }
            for k in 0..bins {
                let c = &cosb[k * N_FFT..(k + 1) * N_FFT];
                let s = &sinb[k * N_FFT..(k + 1) * N_FFT];
                let re: f32 = frame.iter().zip(c).map(|(a, b)| a * b).sum();
                let im: f32 = frame.iter().zip(s).map(|(a, b)| a * b).sum();
                mag[k] = (re * re + im * im + 1e-9).sqrt();
            }
            for m in 0..N_MELS {
                let row = &self.mel_basis[m * bins..(m + 1) * bins];
                let v: f32 = row.iter().zip(&mag).map(|(a, b)| a * b).sum();
                mel[m * frames + f] = v.max(1e-5).ln();
            }
        }
        (mel, frames)
    }

    /// x-vector `[2048]` of 24 kHz mono samples.
    pub fn embed(&self, wav: &[f32]) -> Vec<f32> {
        let (mel, t) = self.mel(wav);
        let x0 = self.block0.run(&mel, t, true);
        let mut outs = Vec::new();
        let mut x = x0;
        for b in &self.blocks {
            let res = x.clone();
            let h = b.tdnn1.run(&x, t, true);
            // Res2Net: chunk 0 passes, chunk i = conv(chunk i (+ previous out)).
            let w = CH / SCALE;
            let mut r2 = vec![0f32; CH * t];
            r2[..w * t].copy_from_slice(&h[..w * t]);
            let mut prev: Vec<f32> = Vec::new();
            for (i, c) in b.res2.iter().enumerate() {
                let part = &h[(i + 1) * w * t..(i + 2) * w * t];
                let inp: Vec<f32> = if i == 0 {
                    part.to_vec()
                } else {
                    part.iter().zip(&prev).map(|(a, b)| a + b).collect()
                };
                prev = c.run(&inp, t, true);
                r2[(i + 1) * w * t..(i + 2) * w * t].copy_from_slice(&prev);
            }
            let h = b.tdnn2.run(&r2, t, true);
            // Squeeze-excitation over the time mean.
            let mean: Vec<f32> = h
                .chunks(t)
                .map(|r| r.iter().sum::<f32>() / t as f32)
                .collect();
            let s1 = b.se1.run(&mean, 1, true);
            let s2 = b.se2.run(&s1, 1, false);
            let mut y = vec![0f32; CH * t];
            for c in 0..CH {
                let g = 1.0 / (1.0 + (-s2[c]).exp());
                for j in 0..t {
                    y[c * t + j] = h[c * t + j] * g + res[c * t + j];
                }
            }
            outs.push(y.clone());
            x = y;
        }
        let cat: Vec<f32> = outs.concat();
        let h = self.mfa.run(&cat, t, true);
        // Attentive statistics pooling.
        let stats = |h: &[f32], wts: Option<&[f32]>| -> (Vec<f32>, Vec<f32>) {
            let mut mean = vec![0f32; CH_CAT];
            let mut std = vec![0f32; CH_CAT];
            for c in 0..CH_CAT {
                let r = &h[c * t..(c + 1) * t];
                let wv = |j: usize| wts.map_or(1.0 / t as f32, |w| w[c * t + j]);
                let m: f32 = (0..t).map(|j| wv(j) * r[j]).sum();
                let v: f32 = (0..t).map(|j| wv(j) * (r[j] - m) * (r[j] - m)).sum();
                mean[c] = m;
                std[c] = v.max(1e-12).sqrt();
            }
            (mean, std)
        };
        let (m0, s0) = stats(&h, None);
        let mut att_in = Vec::with_capacity(3 * CH_CAT * t);
        att_in.extend_from_slice(&h);
        for v in [&m0, &s0] {
            for &x in v.iter() {
                att_in.extend(std::iter::repeat_n(x, t));
            }
        }
        let a = self.asp_tdnn.run(&att_in, t, true);
        let a: Vec<f32> = a.iter().map(|v| v.tanh()).collect();
        let mut a = self.asp_conv.run(&a, t, false);
        debug_assert_eq!(a.len(), CH_CAT * t);
        for row in a.chunks_mut(t) {
            let mx = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let sum: f32 = row
                .iter_mut()
                .map(|v| {
                    *v = (*v - mx).exp();
                    *v
                })
                .sum();
            row.iter_mut().for_each(|v| *v /= sum);
        }
        let (m1, s1) = stats(&h, Some(&a));
        let pooled: Vec<f32> = m1.into_iter().chain(s1).collect();
        self.fc.run(&pooled, 1, false)
    }
}

/// librosa `filters.mel(sr=24000, n_fft=1024, n_mels=128, fmin=0,
/// fmax=12000)`: Slaney mel scale and area normalization, `[128, 513]`.
fn mel_basis() -> Vec<f32> {
    let bins = N_FFT / 2 + 1;
    let hz_to_mel = |f: f64| {
        let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
        let min_log_mel = min_log_hz / f_sp;
        let logstep = (6.4f64).ln() / 27.0;
        if f >= min_log_hz {
            min_log_mel + (f / min_log_hz).ln() / logstep
        } else {
            f / f_sp
        }
    };
    let mel_to_hz = |m: f64| {
        let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
        let min_log_mel = min_log_hz / f_sp;
        let logstep = (6.4f64).ln() / 27.0;
        if m >= min_log_mel {
            min_log_hz * (logstep * (m - min_log_mel)).exp()
        } else {
            f_sp * m
        }
    };
    let fft_freqs: Vec<f64> = (0..bins)
        .map(|i| i as f64 * SAMPLE_RATE as f64 / N_FFT as f64)
        .collect();
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(F_MAX));
    let pts: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    let mut w = vec![0f32; N_MELS * bins];
    for m in 0..N_MELS {
        let (l, c, r) = (pts[m], pts[m + 1], pts[m + 2]);
        let enorm = 2.0 / (r - l);
        for (k, &f) in fft_freqs.iter().enumerate() {
            let lower = (f - l) / (c - l);
            let upper = (r - f) / (r - c);
            w[m * bins + k] = (lower.min(upper).max(0.0) * enorm) as f32;
        }
    }
    w
}
