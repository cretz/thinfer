//! LTX-2.3 audio vocoder (`VocoderWithBWE`, log-mel -> 48kHz stereo waveform).
//!
//! Two BigVGAN v2 generators (see [`crate::common::bigvgan`], which carries the
//! whole conv tower): a main generator (init 1536, 6 up stages, x160 -> 16kHz)
//! and a BWE generator (init 512, 5 stages, x120 -> 48kHz residual). What lives
//! HERE is only the LTX-specific glue the shared tower does not have: the main
//! output clamp, the causal STFT -> mel front-end that drives the BWE, a Hann
//! sinc resampler, and the residual + skip + clamp tail.
//!
//! Ground truth: `ltx_core/model/audio_vae/vocoder.py` (`Vocoder`, `AMPBlock1`,
//! `Activation1d`, `UpSample1d`/`DownSample1d`, `MelSTFT`, `VocoderWithBWE`).
//! Config from the on-disk `audio_vae.safetensors` `vocoder`/`bwe` blocks; tensors
//! under `vocoder.vocoder.*`, `vocoder.bwe_generator.*`, `vocoder.mel_stft.*`
//! (the extra `vocoder.` prefix is on disk; pyref strips ONE).
//!
//! The mel front-end and the resampler run on host: they are light next to the
//! ~108 convolutions, and the resampler's filter is `persistent=False` upstream
//! so it is regenerated rather than loaded.

use thinfer_core::backend::WgpuBackend;
use thinfer_core::residency::WeightResidency;
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::Workspace;

use crate::common::bigvgan::{
    BigVgan, BigVganCfg, BigVganError, BigVganPipelines, read_weight_f32,
};

// ============================================================================
// Config (disk-verified)
// ============================================================================

pub const MEL_BINS: usize = 64;
pub const OUT_CHANNELS: usize = 2;
pub const INPUT_SR: usize = 16000;
pub const OUTPUT_SR: usize = 48000;
/// mel_stft: n_fft 512, hop 80, win 512, mel 64.
pub const N_FFT: usize = 512;
pub const HOP: usize = 80;

const N_FREQS: usize = N_FFT / 2 + 1; // 257

/// Tensor-name and shape config of the two generators.
fn main_cfg() -> BigVganCfg {
    BigVganCfg {
        prefix: "vocoder.vocoder".into(),
        in_channels: OUT_CHANNELS * MEL_BINS,
        out_channels: OUT_CHANNELS,
        init_ch: 1536,
        up_rates: vec![5, 2, 2, 2, 2, 2],
        up_kernels: vec![11, 4, 4, 4, 4, 4],
        ups_nested: false,
        acts_interleaved: false,
        act_post: "act_post".into(),
        in_proj: None,
        up_filter: "vocoder.vocoder.act_post.upsample.filter".into(),
        down_filter: "vocoder.vocoder.act_post.downsample.lowpass.filter".into(),
    }
}

fn bwe_cfg() -> BigVganCfg {
    BigVganCfg {
        prefix: "vocoder.bwe_generator".into(),
        in_channels: OUT_CHANNELS * MEL_BINS,
        out_channels: OUT_CHANNELS,
        init_ch: 512,
        up_rates: vec![6, 5, 2, 2, 2],
        up_kernels: vec![12, 11, 4, 4, 4],
        ups_nested: false,
        acts_interleaved: false,
        act_post: "act_post".into(),
        in_proj: None,
        up_filter: "vocoder.vocoder.act_post.upsample.filter".into(),
        down_filter: "vocoder.vocoder.act_post.downsample.lowpass.filter".into(),
    }
}

pub type VocoderPipelines = BigVganPipelines;
pub type VocoderError<SE> = BigVganError<SE>;

pub struct Vocoder {
    pub pipelines: VocoderPipelines,
    main: BigVgan,
    bwe: BigVgan,
    /// STFT bases (host f32): forward_basis [514,1,512], mel_basis [64,257].
    forward_basis: Vec<f32>,
    mel_basis: Vec<f32>,
}

impl Vocoder {
    pub async fn new<S: WeightSource>(
        pipelines: VocoderPipelines,
        residency: &WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<Self, VocoderError<S::Error>> {
        let main = BigVgan::new(backend, residency, main_cfg()).await?;
        let bwe = BigVgan::new(backend, residency, bwe_cfg()).await?;
        let forward_basis = read_weight_f32(
            backend,
            residency,
            "vocoder.mel_stft.stft_fn.forward_basis",
            N_FREQS * 2 * N_FFT,
        )
        .await?;
        let mel_basis = read_weight_f32(
            backend,
            residency,
            "vocoder.mel_stft.mel_basis",
            MEL_BINS * N_FREQS,
        )
        .await?;
        Ok(Self {
            pipelines,
            main,
            bwe,
            forward_basis,
            mel_basis,
        })
    }

    /// Rearrange a host mel `[2, frames, 64]` into the tower's `[128, frames]`
    /// and run one generator.
    async fn run<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        generator: &BigVgan,
        mel: &[f32],
        frames: usize,
    ) -> Result<Vec<f32>, VocoderError<S::Error>> {
        let mut x0 = vec![0.0f32; OUT_CHANNELS * MEL_BINS * frames];
        for s in 0..OUT_CHANNELS {
            for c in 0..MEL_BINS {
                for t in 0..frames {
                    x0[(s * MEL_BINS + c) * frames + t] = mel[(s * frames + t) * MEL_BINS + c];
                }
            }
        }
        generator
            .run(backend, residency, workspace, &self.pipelines, &x0, frames)
            .await
    }

    /// Decode a log-mel `[2, T, 64]` (host f32) into a 48kHz stereo waveform
    /// `[2, T_wav]` (host f32, clamped [-1,1]).
    pub async fn decode<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        mel: &[f32],
        frames: usize,
    ) -> Result<Vec<f32>, VocoderError<S::Error>> {
        assert_eq!(mel.len(), OUT_CHANNELS * frames * MEL_BINS, "mel size");
        let mut x_main = self
            .run(backend, residency, workspace, &self.main, mel, frames)
            .await?;
        let l16 = x_main.len() / OUT_CHANNELS;
        for v in &mut x_main {
            *v = v.clamp(-1.0, 1.0); // main use_tanh_at_final=false -> clamp.
        }
        let output_length = l16 * OUTPUT_SR / INPUT_SR;

        // Pad x_main to a hop multiple, causal STFT -> log-mel for the BWE drive.
        let rem = l16 % HOP;
        let l16_pad = if rem != 0 { l16 + (HOP - rem) } else { l16 };
        let padded = if l16_pad != l16 {
            let mut p = vec![0.0f32; OUT_CHANNELS * l16_pad];
            for c in 0..OUT_CHANNELS {
                p[c * l16_pad..c * l16_pad + l16].copy_from_slice(&x_main[c * l16..(c + 1) * l16]);
            }
            p
        } else {
            x_main.clone()
        };
        let (bwe_mel, t_frames) = self.compute_mel(&padded, l16_pad);

        let residual = self
            .run(backend, residency, workspace, &self.bwe, &bwe_mel, t_frames)
            .await?;
        let skip = self.hann_resample(&x_main, l16);
        let l48 = residual.len() / OUT_CHANNELS;
        assert_eq!(
            skip.len(),
            OUT_CHANNELS * l48,
            "resampler vs residual length"
        );

        let mut wav = vec![0.0f32; OUT_CHANNELS * output_length];
        for c in 0..OUT_CHANNELS {
            for t in 0..output_length {
                let v = residual[c * l48 + t] + skip[c * l48 + t];
                wav[c * output_length + t] = v.clamp(-1.0, 1.0);
            }
        }
        Ok(wav)
    }

    // -- host STFT mel ------------------------------------------------------

    /// Causal log-mel: pad left (win-hop), conv with `forward_basis` (stride hop),
    /// magnitude, mel_basis projection, log(clamp(.,1e-5)). Returns `([2, T, 64]`,
    /// t_frames) in the BWE input layout.
    fn compute_mel(&self, x: &[f32], l: usize) -> (Vec<f32>, usize) {
        let left_pad = N_FFT - HOP; // causal
        let t_frames = (l + left_pad - N_FFT) / HOP + 1;
        let mut out = vec![0.0f32; OUT_CHANNELS * t_frames * MEL_BINS];
        for ch in 0..OUT_CHANNELS {
            for f in 0..t_frames {
                let start = f as isize * HOP as isize - left_pad as isize;
                let mut spec = vec![0.0f32; N_FREQS * 2];
                for (b, sp) in spec.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    let wbase = b * N_FFT;
                    for n in 0..N_FFT {
                        let idx = start + n as isize;
                        let xv = if idx >= 0 && (idx as usize) < l {
                            x[ch * l + idx as usize]
                        } else {
                            0.0
                        };
                        acc += self.forward_basis[wbase + n] * xv;
                    }
                    *sp = acc;
                }
                for m in 0..MEL_BINS {
                    let mut mel_v = 0.0f32;
                    let mbase = m * N_FREQS;
                    for k in 0..N_FREQS {
                        let re = spec[k];
                        let im = spec[N_FREQS + k];
                        mel_v += self.mel_basis[mbase + k] * (re * re + im * im).sqrt();
                    }
                    out[(ch * t_frames + f) * MEL_BINS + m] = mel_v.max(1e-5).ln();
                }
            }
        }
        (out, t_frames)
    }

    // -- host Hann resampler (UpSample1d ratio 3, persistent=False) ---------

    fn hann_resample(&self, x: &[f32], l: usize) -> Vec<f32> {
        let ratio = OUTPUT_SR / INPUT_SR; // 3
        let rolloff = 0.99f64;
        let lp_width = 6usize;
        let width = (lp_width as f64 / rolloff).ceil() as usize; // 7
        let ksize = 2 * width * ratio + 1; // 43
        let pad_amt = width; // 7
        let pad_left = 2 * width * ratio; // 42
        let pad_right = ksize - ratio; // 40
        let mut filt = vec![0.0f32; ksize];
        for (i, fv) in filt.iter_mut().enumerate() {
            let t = (i as f64 / ratio as f64 - width as f64) * rolloff;
            let tc = t.clamp(-(lp_width as f64), lp_width as f64);
            let window = (tc * std::f64::consts::PI / lp_width as f64 / 2.0)
                .cos()
                .powi(2);
            let sinc = if t == 0.0 {
                1.0
            } else {
                (std::f64::consts::PI * t).sin() / (std::f64::consts::PI * t)
            };
            *fv = (sinc * window * rolloff / ratio as f64) as f32;
        }
        let lp = l + 2 * pad_amt;
        let lout_t = (lp - 1) * ratio + ksize;
        let crop_len = lout_t - pad_left - pad_right;
        let mut out = vec![0.0f32; OUT_CHANNELS * crop_len];
        for ch in 0..OUT_CHANNELS {
            let get = |idx: isize| -> f32 { x[ch * l + idx.clamp(0, l as isize - 1) as usize] };
            for lo in 0..crop_len {
                let pos = lo + pad_left;
                let mut acc = 0.0f32;
                for (kk, &fk) in filt.iter().enumerate() {
                    if pos >= kk && (pos - kk).is_multiple_of(ratio) {
                        let li = (pos - kk) / ratio;
                        if li < lp {
                            acc += fk * get(li as isize - pad_amt as isize);
                        }
                    }
                }
                out[ch * crop_len + lo] = acc * ratio as f32;
            }
        }
        out
    }
}
