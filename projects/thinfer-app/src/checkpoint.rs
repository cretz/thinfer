//! Pausing and resuming a job between steps.
//!
//! A pipeline that can pause keeps, at each step boundary (encode, each
//! denoise step, the VAE decode), the state its remaining steps need: latents,
//! the conditioning, the seed. When the job's owner asks it to pause
//! ([`ProgressSink::pause`]), it stops at the next boundary
//! ([`PauseMode::StepEnd`]), or abandons the step in flight at the next poll
//! ([`PauseMode::Now`]) and keeps the state from before it. That state goes to
//! [`ProgressSink::suspend`] (the server seals it under the job key before it
//! reaches disk) and the run returns `Err(`[`SUSPENDED`]`)`. Weights, adapter
//! folds and device memory are not kept: the resumed run loads them as a cold
//! start would, takes the state back ([`ProgressSink::take_resume`]) and
//! continues from the step after it. Every step is recomputed from exactly
//! the inputs it would have had, so the result matches an uninterrupted run
//! bit for bit.
//!
//! The format is plain little-endian fields behind a header naming the
//! pipeline and its layout version; it is written straight into page-locked
//! memory (it is job data) and only ever read back by the same server build.

use thinfer_core::secmem::Locked;

pub use thinfer_client::wire::PauseMode;

use thinfer_models::common::{SolverState, StepControl, StepState};

use crate::progress::ProgressSink;

/// The error a paused run returns (the worker tells it from a failure by the
/// suspend it just took).
pub const SUSPENDED: &str = "paused";

const MAGIC: &[u8; 8] = b"thnfckpt";

/// Per-block poll: stop now (cancel, or a pause that does not wait for the
/// step to finish).
pub fn interrupted(sink: &dyn ProgressSink) -> bool {
    sink.cancelled() || sink.pause() == Some(PauseMode::Now)
}

/// At a step boundary: stop here (cancel, or any pause).
pub fn stop_here(sink: &dyn ProgressSink) -> bool {
    sink.cancelled() || sink.pause().is_some()
}

/// End a run that stopped early: paused, with the state `write` records under
/// `tag` (the error to return is [`SUSPENDED`]), or cancelled.
pub fn stop(sink: &dyn ProgressSink, tag: &str, write: impl Fn(&mut Writer<'_>)) -> String {
    if sink.cancelled() || sink.pause().is_none() {
        return "cancelled".into();
    }
    let state = encode(tag, write);
    if state.lock_failed() {
        return "could not lock the paused job's state in RAM".into();
    }
    match sink.suspend(&state) {
        Ok(()) => SUSPENDED.into(),
        Err(e) => e,
    }
}

/// For pipelines that resume from a step and a latent (their conditioning
/// is recomputed on resume): the seed the paused run drew and where it
/// stood, if this run resumes one. `tag` names the pipeline and its layout.
pub fn take_steps(sink: &dyn ProgressSink, tag: &str) -> Result<Option<Resumed>, String> {
    let Some(buf) = sink.take_resume() else {
        return Ok(None);
    };
    let mut r = Reader::open(&buf, tag)?;
    let seed = r.u64()?;
    let step = r.usize()?;
    let latent = r.f32s()?;
    let vecs = (0..r.usize()?)
        .map(|_| r.opt_f32s())
        .collect::<Result<_, _>>()?;
    let ints = (0..r.usize()?).map(|_| r.u64()).collect::<Result<_, _>>()?;
    let prompt = if r.bool()? {
        Some(zeroize::Zeroizing::new(
            String::from_utf8(r.bytes()?.to_vec()).map_err(|_| "checkpoint prompt is not UTF-8")?,
        ))
    } else {
        None
    };
    r.finish()?;
    Ok(Some(Resumed {
        seed,
        state: StepState {
            step,
            latent,
            solver: SolverState { vecs, ints },
        },
        prompt,
    }))
}

/// A run resuming from [`take_steps`].
pub struct Resumed {
    /// The seed the paused run drew.
    pub seed: u64,
    pub state: StepState,
    /// The prompt it conditioned on, when it was one it derived (a rewrite):
    /// the resumed run uses it as-is.
    pub prompt: Option<zeroize::Zeroizing<String>>,
}

/// [`stop`] with the state [`take_steps`] reads back.
pub fn stop_steps(sink: &dyn ProgressSink, tag: &str, seed: u64, state: &StepState) -> String {
    stop_steps_with(sink, tag, seed, None, state)
}

/// [`stop_steps`], keeping the prompt the run conditioned on.
pub fn stop_steps_with(
    sink: &dyn ProgressSink,
    tag: &str,
    seed: u64,
    prompt: Option<&str>,
    state: &StepState,
) -> String {
    stop(sink, tag, |w| {
        w.u64(seed);
        w.usize(state.step);
        w.f32s(&state.latent);
        w.usize(state.solver.vecs.len());
        for v in &state.solver.vecs {
            w.opt_f32s(v.as_deref());
        }
        w.usize(state.solver.ints.len());
        for &i in &state.solver.ints {
            w.u64(i);
        }
        w.bool(prompt.is_some());
        if let Some(p) = prompt {
            w.bytes(p.as_bytes());
        }
    })
}

/// A [`StepControl`] polling `sink` at each step boundary.
pub fn step_control<'a>(
    resume: Option<StepState>,
    stop_here: &'a dyn Fn() -> bool,
) -> StepControl<'a> {
    StepControl {
        resume,
        stop_here: Some(stop_here),
    }
}

/// Serialize a checkpoint: `write` runs twice, once to size it and once to
/// fill the page-locked buffer.
pub fn encode(tag: &str, write: impl Fn(&mut Writer<'_>)) -> Locked<u8> {
    let mut size = Writer { out: None, len: 0 };
    size.header(tag);
    write(&mut size);
    let mut buf = Locked::<u8>::zeroed(size.len);
    let mut w = Writer {
        out: Some(&mut buf),
        len: 0,
    };
    w.header(tag);
    write(&mut w);
    debug_assert_eq!(w.len, size.len, "checkpoint writer is not deterministic");
    buf
}

/// Field writer (see [`encode`]).
pub struct Writer<'a> {
    out: Option<&'a mut [u8]>,
    len: usize,
}

impl Writer<'_> {
    fn put(&mut self, b: &[u8]) {
        if let Some(out) = self.out.as_deref_mut() {
            out[self.len..][..b.len()].copy_from_slice(b);
        }
        self.len += b.len();
    }

    fn header(&mut self, tag: &str) {
        self.put(MAGIC);
        self.bytes(tag.as_bytes());
    }

    pub fn u64(&mut self, v: u64) {
        self.put(&v.to_le_bytes());
    }

    pub fn usize(&mut self, v: usize) {
        self.u64(v as u64);
    }

    pub fn bool(&mut self, v: bool) {
        self.put(&[u8::from(v)]);
    }

    pub fn bytes(&mut self, v: &[u8]) {
        self.usize(v.len());
        self.put(v);
    }

    pub fn f32s(&mut self, v: &[f32]) {
        self.usize(v.len());
        match self.out.as_deref_mut() {
            Some(out) => {
                let dst = &mut out[self.len..][..v.len() * 4];
                for (d, x) in dst.chunks_exact_mut(4).zip(v) {
                    d.copy_from_slice(&x.to_le_bytes());
                }
                self.len += v.len() * 4;
            }
            None => self.len += v.len() * 4,
        }
    }

    pub fn i32s(&mut self, v: &[i32]) {
        self.usize(v.len());
        for x in v {
            self.put(&x.to_le_bytes());
        }
    }

    /// `None` or a slice.
    pub fn opt_f32s(&mut self, v: Option<&[f32]>) {
        self.bool(v.is_some());
        if let Some(v) = v {
            self.f32s(v);
        }
    }
}

/// Field reader over a checkpoint [`encode`] wrote under the same tag.
pub struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    /// Check the header: the checkpoint must be this pipeline's, at this
    /// layout version (`tag`).
    pub fn open(buf: &'a [u8], tag: &str) -> Result<Self, String> {
        let mut r = Self { buf, at: 0 };
        if r.take(MAGIC.len())? != MAGIC {
            return Err("not a paused-job checkpoint".into());
        }
        let found = r.bytes()?;
        if found != tag.as_bytes() {
            return Err(format!(
                "the paused job's checkpoint is for {}, not {tag}",
                String::from_utf8_lossy(found)
            ));
        }
        Ok(r)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or("the paused job's checkpoint is truncated")?;
        let b = &self.buf[self.at..end];
        self.at = end;
        Ok(b)
    }

    pub fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }

    pub fn usize(&mut self) -> Result<usize, String> {
        usize::try_from(self.u64()?).map_err(|_| "checkpoint length overflows".to_string())
    }

    pub fn bool(&mut self) -> Result<bool, String> {
        match self.take(1)?[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err("the paused job's checkpoint is corrupt".into()),
        }
    }

    pub fn bytes(&mut self) -> Result<&'a [u8], String> {
        let n = self.usize()?;
        self.take(n)
    }

    pub fn f32s(&mut self) -> Result<Vec<f32>, String> {
        let n = self.usize()?;
        let b = self.take(n.checked_mul(4).ok_or("checkpoint length overflows")?)?;
        Ok(b.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect())
    }

    pub fn i32s(&mut self) -> Result<Vec<i32>, String> {
        let n = self.usize()?;
        let b = self.take(n.checked_mul(4).ok_or("checkpoint length overflows")?)?;
        Ok(b.chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect())
    }

    pub fn opt_f32s(&mut self) -> Result<Option<Vec<f32>>, String> {
        if self.bool()? {
            self.f32s().map(Some)
        } else {
            Ok(None)
        }
    }

    /// Every byte was read.
    pub fn finish(self) -> Result<(), String> {
        if self.at == self.buf.len() {
            Ok(())
        } else {
            Err("the paused job's checkpoint has trailing data".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_round_trip_bit_exactly() {
        let latents = [
            0.0f32,
            -0.0,
            1.5e-38,
            f32::MIN_POSITIVE / 3.0,
            -7.25,
            f32::MAX,
        ];
        let buf = encode("test/1", |w| {
            w.usize(3);
            w.bool(true);
            w.f32s(&latents);
            w.i32s(&[-1, 0, i32::MAX]);
            w.opt_f32s(None);
            w.opt_f32s(Some(&latents[..2]));
            w.bytes(b"x");
        });
        let mut r = Reader::open(&buf, "test/1").unwrap();
        assert_eq!(r.usize().unwrap(), 3);
        assert!(r.bool().unwrap());
        let back = r.f32s().unwrap();
        assert!(
            back.iter()
                .zip(&latents)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert_eq!(r.i32s().unwrap(), [-1, 0, i32::MAX]);
        assert_eq!(r.opt_f32s().unwrap(), None);
        assert_eq!(r.opt_f32s().unwrap().unwrap().len(), 2);
        assert_eq!(r.bytes().unwrap(), b"x");
        r.finish().unwrap();
    }

    #[test]
    fn a_foreign_or_truncated_checkpoint_is_refused() {
        let buf = encode("a/1", |w| w.f32s(&[1.0; 4]));
        assert!(Reader::open(&buf, "a/2").is_err());
        assert!(Reader::open(b"garbage!", "a/1").is_err());
        let mut r = Reader::open(&buf[..buf.len() - 1], "a/1").unwrap();
        assert!(r.f32s().is_err());
        let mut r = Reader::open(&buf, "a/1").unwrap();
        r.usize().unwrap();
        assert!(r.finish().is_err());
    }
}
