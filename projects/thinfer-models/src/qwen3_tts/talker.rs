//! The talker and code predictor: two Qwen3 decoder stacks on the LLM
//! kernels (bf16 weights read in place, f32 activations, f16 KV caches).
//!
//! One frame = one talker step (codebook 0 logits + its last hidden row),
//! then the code predictor's 2-row prefill `[talker hidden, embed(code 0)]`
//! and 14 one-row steps (codebooks 1..15). Codes are sampled on the host;
//! the next talker input is the sum of the frame's 16 code embeddings plus
//! the next streamed text row. With an accelerator a whole frame runs on its
//! stream (inputs written and logits read in stream order, no wgpu submit);
//! without one every sub-step is its own submit.

use std::sync::Arc;

use thinfer_core::Backend;
use thinfer_core::backend::WgpuBackend;
use thinfer_core::residency::{GpuView, WeightResidency};
use thinfer_core::tensor::StorageEncoding;
use thinfer_core::weight::{WeightId, WeightReader, WeightSource};
use thinfer_core::workspace::{Workspace, WsBuf};
use zeroize::Zeroizing;

use super::prompt::{Part, TalkerPrompt};
use super::{
    CODE_GROUPS, CODEBOOK, PREDICTOR, TALKER, TALKER_VOCAB, TtsSampling, codec_ids, text_ids,
};
use crate::llm::decoder::{LlmError, OwnedBuf, Registry, Scope, W, acquire, mat};
use crate::llm::dense::{DenseStack, Io, Rows, bf16_row, bf16_vec, rope_rows};
use crate::llm::kernels::{Act, LlmPipelines, act_bytes};
use crate::llm::sampler::{Sampler, SamplingParams};

/// Every weight of the talker side of a Qwen3-TTS checkpoint.
pub struct TalkerWeights {
    talker: DenseStack,
    predictor: DenseStack,
    codec_head: W,
    /// Predictor heads for codebooks 1..15.
    heads: Vec<W>,
    proj: W,
    proj_bias: Vec<f32>,
    fc1: W,
    fc1_bias: Vec<f32>,
    fc2: W,
    fc2_bias: Vec<f32>,
    /// The predictor's codebook 1..15 embedding tables (bf16 `[2048, 2048]`
    /// each) on the device, when an accelerator runs the LLM kernels: the
    /// predictor then samples and embeds on the device, one sync per frame.
    groups: Option<Vec<OwnedBuf>>,
}

/// A tensor's raw bytes.
async fn raw<S: WeightSource>(src: &S, name: &str) -> Result<Vec<u8>, LlmError> {
    let id = WeightId(name.to_string());
    let mut r = src
        .open(&id)
        .await
        .map_err(|e| LlmError::Weights(format!("{name}: {e:?}")))?;
    let mut b = vec![0u8; r.len() as usize];
    r.read_at(0, &mut b)
        .await
        .map_err(|e| LlmError::Weights(format!("{name}: {e:?}")))?;
    Ok(b)
}

impl TalkerWeights {
    /// Register the matrices (bf16 in place) and read the vectors.
    pub async fn load<S: WeightSource>(
        backend: &Arc<WgpuBackend>,
        residency: &WeightResidency<S>,
    ) -> Result<Self, LlmError> {
        let reg = Registry { residency };
        let src = residency.source();
        let talker = DenseStack::load(backend, residency, "talker.model", TALKER).await?;
        let predictor =
            DenseStack::load(backend, residency, "talker.code_predictor.model", PREDICTOR).await?;
        let heads = (0..CODE_GROUPS - 1)
            .map(|i| reg.dense(&format!("talker.code_predictor.lm_head.{i}.weight")))
            .collect::<Result<_, _>>()?;
        let groups = if backend.accelerator().is_some_and(|a| a.accepts_llm()) {
            let mut v = Vec::with_capacity(CODE_GROUPS - 1);
            for g in 0..CODE_GROUPS - 1 {
                let name = format!("talker.code_predictor.model.codec_embedding.{g}.weight");
                let e = src.catalog().get(&WeightId(name.clone())).cloned();
                if e.as_ref().and_then(|e| e.encoding) != Some(StorageEncoding::Bf16) {
                    return Err(LlmError::Weights(format!("{name}: expected bf16")));
                }
                v.push(OwnedBuf::bytes(backend, &raw(src, &name).await?)?);
            }
            Some(v)
        } else {
            None
        };
        Ok(Self {
            groups,
            talker,
            predictor,
            codec_head: reg.dense("talker.codec_head.weight")?,
            heads,
            proj: reg.dense("talker.code_predictor.small_to_mtp_projection.weight")?,
            proj_bias: bf16_vec(src, "talker.code_predictor.small_to_mtp_projection.bias").await?,
            fc1: reg.dense("talker.text_projection.linear_fc1.weight")?,
            fc1_bias: bf16_vec(src, "talker.text_projection.linear_fc1.bias").await?,
            fc2: reg.dense("talker.text_projection.linear_fc2.weight")?,
            fc2_bias: bf16_vec(src, "talker.text_projection.linear_fc2.bias").await?,
        })
    }

    fn all(&self) -> Vec<W> {
        let mut v = self.talker.weights();
        v.extend(self.predictor.weights());
        v.extend(self.heads.iter().copied());
        v.extend([self.codec_head, self.proj, self.fc1, self.fc2]);
        v
    }
}

/// Embedding rows read from the checkpoint on demand (bf16, widened).
struct Embeddings<'a, S: WeightSource> {
    src: &'a S,
}

impl<S: WeightSource> Embeddings<'_, S> {
    async fn row(&self, name: &str, idx: u32, out: &mut [f32]) -> Result<(), LlmError> {
        bf16_row(self.src, name, idx, out).await
    }

    /// Talker codec embedding of `id` (codebook 0 codes and control ids).
    async fn codec(&self, id: u32, out: &mut [f32]) -> Result<(), LlmError> {
        self.row("talker.model.codec_embedding.weight", id, out)
            .await
    }

    /// Predictor codec embedding `g` (codebook `g + 1`) of `code`.
    async fn group(&self, g: usize, code: u32, out: &mut [f32]) -> Result<(), LlmError> {
        self.row(
            &format!("talker.code_predictor.model.codec_embedding.{g}.weight"),
            code,
            out,
        )
        .await
    }

    async fn text(&self, tok: u32, out: &mut [f32]) -> Result<(), LlmError> {
        self.row("talker.model.text_embedding.weight", tok, out)
            .await
    }

    /// Sum of a frame's 16 code embeddings.
    async fn frame(&self, codes: &[u32], out: &mut [f32]) -> Result<(), LlmError> {
        let mut tmp = vec![0f32; out.len()];
        self.codec(codes[0], out).await?;
        for (g, &c) in codes[1..].iter().enumerate() {
            self.group(g, c, &mut tmp).await?;
            out.iter_mut().zip(&tmp).for_each(|(o, t)| *o += t);
        }
        Ok(())
    }
}

/// What a generation needs besides the prompt.
pub struct GenSpec<'a> {
    pub prompt: &'a TalkerPrompt,
    /// The speaker vector for [`Part::Speaker`] (`[TALKER.hidden]`).
    pub speaker: Option<&'a [f32]>,
    /// In-context reference codes, `[frames][16]`, for [`Part::RefFrame`].
    pub ref_codes: &'a [[u32; CODE_GROUPS]],
    pub sampling: TtsSampling,
    pub seed: u64,
    pub max_frames: u32,
    /// Keep codebook-0 logits of the first this-many frames (parity).
    pub tap_logits: usize,
    /// Teacher forcing (parity): take these codes instead of sampling and
    /// stop after the last one.
    pub force: Option<&'a [[u32; CODE_GROUPS]]>,
}

/// Per-frame hook: called with the frame count so far; `false` stops.
pub trait FrameSink {
    fn frame(&mut self, frames: u32) -> bool;
}

/// Max rows of one stack forward (the prompt; text projection chunks).
const MAX_ROWS: u32 = 1024;

/// Persistent buffers of one generation.
struct Bufs {
    tk: Vec<[WsBuf<WgpuBackend>; 2]>,
    pk: Vec<[WsBuf<WgpuBackend>; 2]>,
    /// Stack input rows `[MAX_ROWS, 2048]` (host-written).
    x: WsBuf<WgpuBackend>,
    /// Rope table rows `[MAX_ROWS, 64, 2]`.
    rope: WsBuf<WgpuBackend>,
    /// Talker last hidden row (normed) `[2048]`.
    hid: WsBuf<WgpuBackend>,
    /// Logits `[3072]` (codebook 0) or `[2048]`.
    logits: WsBuf<WgpuBackend>,
    /// Predictor projection bias, two rows.
    pbias: WsBuf<WgpuBackend>,
    /// Text-projection output rows (read back).
    tout: WsBuf<WgpuBackend>,
    /// Text-projection bias rows (fc1, fc2), `[MAX_ROWS, 2048]` each.
    tb1: WsBuf<WgpuBackend>,
    tb2: WsBuf<WgpuBackend>,
    /// `1.0` rows for the SiLU (glu with unit gate partner).
    ones: WsBuf<WgpuBackend>,
    /// Device sampling: a frame's draws (`[15]` f32) and codes (`[16]` u32).
    uni: WsBuf<WgpuBackend>,
    codes: WsBuf<WgpuBackend>,
}

pub struct Talker<'w, S: WeightSource> {
    pub w: &'w TalkerWeights,
    pub residency: &'w WeightResidency<S>,
    pub backend: &'w Arc<WgpuBackend>,
    pub p: &'w LlmPipelines,
    pub ws: &'w Workspace<WgpuBackend>,
}

/// Generation result: codes `[frames][16]` (job data, wiped on drop).
pub struct Codes {
    pub codes: Zeroizing<Vec<[u32; CODE_GROUPS]>>,
    /// Codebook-0 logits of the first [`GenSpec::tap_logits`] frames.
    pub logits: Vec<Vec<f32>>,
}

impl<S: WeightSource> Talker<'_, S> {
    fn emb(&self) -> Embeddings<'_, S> {
        Embeddings {
            src: self.residency.source(),
        }
    }

    async fn views(&self, ws: &[W]) -> Result<Vec<GpuView<'_>>, LlmError> {
        let mut v = Vec::with_capacity(ws.len());
        for &w in ws {
            v.push(acquire(self.residency, self.backend, w).await?);
        }
        Ok(v)
    }

    async fn bufs(&self, max_ctx: u32) -> Result<Bufs, LlmError> {
        let ws = self.ws;
        let h = TALKER.hidden;
        let pb: Vec<f32> = self.w.proj_bias.repeat(2);
        let pbias = ws.alloc(act_bytes(2, PREDICTOR.hidden))?;
        self.backend
            .write_buffer(pbias.id(), 0, bytemuck::cast_slice(&pb))?;
        let ones = ws.alloc(act_bytes(MAX_ROWS, h))?;
        self.backend.write_buffer(
            ones.id(),
            0,
            bytemuck::cast_slice(&vec![1f32; (MAX_ROWS * h) as usize]),
        )?;
        let tb = |b: &[f32]| -> Result<WsBuf<WgpuBackend>, LlmError> {
            let buf = ws.alloc(act_bytes(MAX_ROWS, h))?;
            self.backend.write_buffer(
                buf.id(),
                0,
                bytemuck::cast_slice(&b.repeat(MAX_ROWS as usize)),
            )?;
            Ok(buf)
        };
        let bufs = Bufs {
            tk: self.w.talker.kv(ws, max_ctx)?,
            pk: self.w.predictor.kv(ws, CODE_GROUPS as u32 + 1)?,
            x: ws.alloc(act_bytes(MAX_ROWS, h))?,
            rope: ws.alloc(act_bytes(MAX_ROWS, TALKER.head_dim))?,
            hid: ws.alloc(act_bytes(1, h))?,
            logits: ws.alloc(act_bytes(1, TALKER_VOCAB))?,
            pbias,
            tout: ws.alloc(act_bytes(MAX_ROWS, h))?,
            tb1: tb(&self.w.fc1_bias)?,
            tb2: tb(&self.w.fc2_bias)?,
            ones,
            uni: ws.alloc(act_bytes(1, CODE_GROUPS as u32))?,
            codes: ws.alloc(act_bytes(1, CODE_GROUPS as u32))?,
        };
        // Queue writes land at the next submit; an accelerator reads them
        // without one, so flush and wait here.
        self.ws.batch().submit_void().await?;
        self.backend.wait_idle();
        Ok(bufs)
    }

    /// Project text tokens through `text_embedding` + `text_projection`.
    async fn project_text(&self, b: &Bufs, toks: &[u32]) -> Result<Vec<Vec<f32>>, LlmError> {
        let h = TALKER.hidden;
        let emb = self.emb();
        let fw = self.views(&[self.w.fc1, self.w.fc2]).await?;
        let mut out = Vec::with_capacity(toks.len());
        for chunk in toks.chunks(MAX_ROWS as usize) {
            let t = chunk.len() as u32;
            let mut x = Zeroizing::new(vec![0f32; (t * h) as usize]);
            for (i, &tok) in chunk.iter().enumerate() {
                emb.text(tok, &mut x[i * h as usize..(i + 1) * h as usize])
                    .await?;
            }
            let s = self.ws.batch();
            let io = Io {
                backend: self.backend,
            };
            let deferred = s.accel_begin()?;
            io.put(&s, &b.x, bytemuck::cast_slice(&x[..]))?;
            let xi = s.import_copy(b.x.as_buf_ref());
            let y1 = s.alloc(act_bytes(t, h))?;
            let tb1 = s.import_copy(b.tb1.as_buf_ref());
            self.p.gemv(
                &s,
                mat(&s, &fw[0], self.w.fc1.fmt),
                Act::dense(xi, h),
                Act::dense(y1, h),
                Some(tb1),
                t,
                h,
                h,
            )?;
            let a = s.alloc(act_bytes(t, h))?;
            let ones = s.import_copy(b.ones.as_buf_ref());
            self.p.glu(&s, y1, ones, a, t * h, false)?;
            let tb2 = s.import_copy(b.tb2.as_buf_ref());
            let to = s.import_copy(b.tout.as_buf_ref());
            self.p.gemv(
                &s,
                mat(&s, &fw[1], self.w.fc2.fmt),
                Act::dense(a, h),
                Act::dense(to, h),
                Some(tb2),
                t,
                h,
                h,
            )?;
            let rows = if deferred {
                let v = Io::get_deferred(&s, &b.tout, (t * h) as usize)?;
                s.submit_void().await?;
                v
            } else {
                s.submit_void().await?;
                let bytes = self
                    .backend
                    .read_buffer(b.tout.id(), 0, act_bytes(t, h))
                    .await?;
                bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
            };
            out.extend(rows.chunks(h as usize).map(<[f32]>::to_vec));
        }
        Ok(out)
    }

    /// Generate codes for `spec`.
    pub async fn generate(
        &self,
        spec: &GenSpec<'_>,
        sink: &mut dyn FrameSink,
    ) -> Result<Codes, LlmError> {
        let h = TALKER.hidden as usize;
        let prompt = spec.prompt;
        let rows = prompt.rows.len() as u32;
        if rows == 0 || rows > MAX_ROWS {
            return Err(LlmError::Weights(format!(
                "prompt of {rows} rows (max {MAX_ROWS})"
            )));
        }
        let max_ctx = rows + spec.max_frames + 1;
        let b = self.bufs(max_ctx).await?;
        let emb = self.emb();

        // Text rows: every distinct text id the prompt and stream use.
        let mut toks: Vec<u32> = prompt
            .rows
            .iter()
            .flatten()
            .filter_map(|p| match p {
                Part::Text(t) => Some(*t),
                _ => None,
            })
            .chain(prompt.trailing.iter().copied())
            .chain([text_ids::TTS_PAD])
            .collect();
        toks.sort_unstable();
        toks.dedup();
        let proj = self.project_text(&b, &toks).await?;
        let text_row = |t: u32| &proj[toks.binary_search(&t).expect("projected")];

        // Prefill rows.
        let mut x = Zeroizing::new(vec![0f32; rows as usize * h]);
        let mut tmp = vec![0f32; h];
        for (r, parts) in prompt.rows.iter().enumerate() {
            let row = &mut x[r * h..(r + 1) * h];
            for part in parts {
                match *part {
                    Part::Text(t) => row.iter_mut().zip(text_row(t)).for_each(|(o, v)| *o += v),
                    Part::Codec(id) => {
                        emb.codec(id, &mut tmp).await?;
                        row.iter_mut().zip(&tmp).for_each(|(o, v)| *o += v);
                    }
                    Part::Speaker => {
                        let sp = spec.speaker.ok_or_else(|| {
                            LlmError::Weights("prompt needs a speaker vector".into())
                        })?;
                        row.iter_mut().zip(sp).for_each(|(o, v)| *o += v);
                    }
                    Part::RefFrame(i) => {
                        let f = spec.ref_codes.get(i).ok_or_else(|| {
                            LlmError::Weights("reference frame out of range".into())
                        })?;
                        emb.frame(f, &mut tmp).await?;
                        row.iter_mut().zip(&tmp).for_each(|(o, v)| *o += v);
                    }
                }
            }
        }

        let sp = spec.sampling;
        let mut s0 = Sampler::new(SamplingParams {
            temperature: sp.temperature,
            top_k: sp.top_k,
            top_p: sp.top_p,
            min_p: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: sp.repetition_penalty,
            seed: spec.seed,
        });
        let sub = SamplingParams {
            temperature: sp.sub_temperature,
            top_k: sp.sub_top_k,
            top_p: sp.sub_top_p,
            min_p: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            seed: 0,
        };
        let mut s_sub = Sampler::new(SamplingParams {
            seed: spec.seed.wrapping_add(1),
            ..sub
        });

        // The predictor samples on the device when it can reproduce the host
        // sampler there (plain top-k: no top-p, a k the kernel holds).
        let device_sub = (self.w.groups.is_some()
            && spec.force.is_none()
            && sp.sub_top_p >= 1.0
            && (1..=64).contains(&sp.sub_top_k))
        .then_some((sp.sub_top_k, sp.sub_temperature));
        let tv = self.views(&self.w.talker.weights()).await?;
        let pv = self.views(&self.w.predictor.weights()).await?;
        let hv = self.views(&self.w.heads).await?;
        let mv = self.views(&[self.w.codec_head, self.w.proj]).await?;

        let mut codes: Zeroizing<Vec<[u32; CODE_GROUPS]>> = Zeroizing::new(Vec::new());
        let mut tapped = Vec::new();
        let mut pos = 0u32;
        let mut input_rows = rows;
        let mut next = x;
        loop {
            let g = codes.len() as u32;
            let s = self.ws.batch();
            let io = Io {
                backend: self.backend,
            };
            let deferred = s.accel_begin()?;
            // Talker step.
            io.put(&s, &b.x, bytemuck::cast_slice(&next[..]))?;
            io.put(
                &s,
                &b.rope,
                bytemuck::cast_slice(&rope_rows(&TALKER, pos, input_rows)),
            )?;
            let xi = s.import_copy(b.x.as_buf_ref());
            let rp = s.import_copy(b.rope.as_buf_ref());
            // The normed last row stays in `hid` for the predictor.
            let hid = s.import_copy(b.hid.as_buf_ref());
            self.w.talker.encode(
                &s,
                self.p,
                &tv,
                &b.tk,
                rp,
                xi,
                input_rows,
                pos,
                hid,
                Rows::Last,
            )?;
            let lg = s.import_copy(b.logits.as_buf_ref());
            let hdu = TALKER.hidden;
            self.p.gemv(
                &s,
                mat(&s, &mv[0], self.w.codec_head.fmt),
                Act::dense(hid, hdu),
                Act::dense(lg, TALKER_VOCAB),
                None,
                1,
                TALKER_VOCAB,
                hdu,
            )?;
            let (mut logits, s) = self.read(s, deferred, &b, TALKER_VOCAB).await?;
            pos += input_rows;
            // Codebook 0: control ids suppressed except eos (not before 2 frames).
            if tapped.len() < spec.tap_logits {
                tapped.push(logits.clone());
            }
            for (i, l) in logits.iter_mut().enumerate().skip(CODEBOOK as usize) {
                if i as u32 != codec_ids::EOS || g < 2 {
                    *l = f32::NEG_INFINITY;
                }
            }
            let forced = spec.force.map(|f| f.get(g as usize));
            let c0 = match forced {
                Some(Some(f)) => f[0],
                Some(None) => codec_ids::EOS,
                None => s0.sample(&mut logits),
            };
            s0.accept(c0);
            if c0 == codec_ids::EOS || g >= spec.max_frames {
                if let Some(s) = s {
                    s.submit_void().await?;
                }
                break;
            }
            let mut frame = [0u32; CODE_GROUPS];
            frame[0] = c0;
            let s = self
                .predict(
                    s,
                    deferred,
                    &b,
                    &pv,
                    &hv,
                    &mv[1],
                    &mut frame,
                    &mut s_sub,
                    forced.flatten(),
                    device_sub.filter(|_| deferred),
                )
                .await?;
            codes.push(frame);
            // Next talker input: the frame's code embeddings + streamed text.
            let mut nx = Zeroizing::new(vec![0f32; h]);
            emb.frame(&frame, &mut nx).await?;
            let tt = prompt
                .trailing
                .get(g as usize)
                .copied()
                .unwrap_or(text_ids::TTS_PAD);
            nx.iter_mut().zip(text_row(tt)).for_each(|(o, v)| *o += v);
            next = nx;
            input_rows = 1;
            if let Some(s) = s {
                s.submit_void().await?;
            }
            if !sink.frame(codes.len() as u32) {
                return Err(LlmError::Weights("cancelled".into()));
            }
        }
        Ok(Codes {
            codes,
            logits: tapped,
        })
    }

    /// Read `n` logits (and keep `hid` written): on an accelerator in stream
    /// order (the scope stays open), else submit and read (`None` scope).
    async fn read<'s>(
        &self,
        s: Scope<'s>,
        deferred: bool,
        b: &Bufs,
        n: u32,
    ) -> Result<(Vec<f32>, Option<Scope<'s>>), LlmError> {
        if deferred {
            let v = Io::get_deferred(&s, &b.logits, n as usize)?;
            Ok((v, Some(s)))
        } else {
            s.submit_void().await?;
            let bytes = self
                .backend
                .read_buffer(b.logits.id(), 0, act_bytes(1, n))
                .await?;
            Ok((bytemuck::cast_slice::<u8, f32>(&bytes).to_vec(), None))
        }
    }

    /// Codebooks 1..15 of `frame` (codebook 0 set) from the talker hidden.
    #[allow(clippy::too_many_arguments)]
    async fn predict<'s>(
        &'s self,
        mut s: Option<Scope<'s>>,
        deferred: bool,
        b: &Bufs,
        pv: &[GpuView<'_>],
        hv: &[GpuView<'_>],
        projv: &GpuView<'_>,
        frame: &mut [u32; CODE_GROUPS],
        sampler: &mut Sampler,
        forced: Option<&[u32; CODE_GROUPS]>,
        device: Option<(u32, f32)>,
    ) -> Result<Option<Scope<'s>>, LlmError> {
        let (th, ph) = (TALKER.hidden, PREDICTOR.hidden);
        let emb = self.emb();
        let mut e = Zeroizing::new(vec![0f32; th as usize]);
        // Device sampling draws the frame's uniforms from the host stream in
        // the order the host sampler would consume them.
        let draws: Option<Zeroizing<Vec<f32>>> = device.map(|(_, temp)| {
            Zeroizing::new(
                (0..CODE_GROUPS - 1)
                    .map(|_| if temp > 0.0 { sampler.uniform() } else { 0.0 })
                    .collect(),
            )
        });
        let at = |sc: &Scope<'s>, buf: &WsBuf<WgpuBackend>, i: usize| {
            sc.subview(&sc.import_copy(buf.as_buf_ref()), i as u64 * 4, 4)
        };
        for step in 0..CODE_GROUPS - 1 {
            let sc = match s.take() {
                Some(sc) => sc,
                None => {
                    let sc = self.ws.batch();
                    sc.accel_begin()?;
                    sc
                }
            };
            let io = Io {
                backend: self.backend,
            };
            // Input embedding: codebook 0 through the talker table, then
            // codebook `step` through predictor table `step - 1`.
            if let (Some(_), true, Some(groups)) = (&draws, step > 0, &self.w.groups) {
                // Codebook `step`'s embedding of the code sampled last step.
                let table = sc.import_copy(groups[step - 1].buf_ref());
                let xi = sc.import_copy(b.x.as_buf_ref());
                self.p
                    .gather_row_bf16(&sc, table, at(&sc, &b.codes, step), th, xi)?;
            } else {
                if step == 0 {
                    emb.codec(frame[0], &mut e).await?;
                } else {
                    emb.group(step - 1, frame[step], &mut e).await?;
                }
                io.put(&sc, &b.x, bytemuck::cast_slice(&e[..]))?;
            }
            let (t, pos) = if step == 0 {
                (2u32, 0u32)
            } else {
                (1, step as u32 + 1)
            };
            io.put(
                &sc,
                &b.rope,
                bytemuck::cast_slice(&rope_rows(&PREDICTOR, pos, t)),
            )?;
            // Projection 2048 -> 1024 (+ bias): row 0 of the prefill is the
            // talker hidden, the embedding row follows.
            let y = sc.alloc(act_bytes(t, ph))?;
            let pm = mat(&sc, projv, self.w.proj.fmt);
            let bias = sc.import_copy(b.pbias.as_buf_ref());
            let xi = sc.import_copy(b.x.as_buf_ref());
            if step == 0 {
                let hid = sc.import_copy(b.hid.as_buf_ref());
                self.p.gemv(
                    &sc,
                    pm,
                    Act::dense(hid, th),
                    Act::dense(y, ph),
                    Some(bias),
                    1,
                    ph,
                    th,
                )?;
                let y1 = sc.subview(&y, act_bytes(1, ph), act_bytes(1, ph));
                let b1 = sc.subview(&bias, act_bytes(1, ph), act_bytes(1, ph));
                self.p.gemv(
                    &sc,
                    pm,
                    Act::dense(xi, th),
                    Act::dense(y1, ph),
                    Some(b1),
                    1,
                    ph,
                    th,
                )?;
            } else {
                self.p.gemv(
                    &sc,
                    pm,
                    Act::dense(xi, th),
                    Act::dense(y, ph),
                    Some(bias),
                    1,
                    ph,
                    th,
                )?;
            }
            let rp = sc.import_copy(b.rope.as_buf_ref());
            let last = sc.alloc(act_bytes(1, ph))?;
            self.w
                .predictor
                .encode(&sc, self.p, pv, &b.pk, rp, y, t, pos, last, Rows::Last)?;
            let lg = sc.import_copy(b.logits.as_buf_ref());
            self.p.gemv(
                &sc,
                mat(&sc, &hv[step], self.w.heads[step].fmt),
                Act::dense(last, ph),
                Act::dense(lg, CODEBOOK),
                None,
                1,
                CODEBOOK,
                ph,
            )?;
            sc.recycle(last);
            if let (Some((k, temp)), Some(d)) = (device, &draws) {
                if step == 0 {
                    io.put(&sc, &b.uni, bytemuck::cast_slice(&d[..]))?;
                }
                self.p.sample_topk(
                    &sc,
                    lg,
                    CODEBOOK,
                    k,
                    temp,
                    at(&sc, &b.uni, step),
                    at(&sc, &b.codes, step + 1),
                )?;
                s = Some(sc);
                continue;
            }
            let (mut logits, rest) = self.read(sc, deferred, b, CODEBOOK).await?;
            s = rest;
            frame[step + 1] = match forced {
                Some(f) => f[step + 1],
                None => sampler.sample(&mut logits),
            };
        }
        if device.is_some()
            && let Some(sc) = &s
        {
            // The frame's 15 codes, one read.
            let v = Io::get_deferred(sc, &b.codes, CODE_GROUPS)?;
            for (f, c) in frame.iter_mut().zip(&v).skip(1) {
                *f = c.to_bits();
            }
        }
        Ok(s)
    }

    /// Weights to pin (all of them are touched every frame).
    pub fn pin_plan(w: &TalkerWeights) -> Vec<thinfer_core::residency::WeightHandle> {
        w.all().into_iter().map(|w| w.h).collect()
    }
}
