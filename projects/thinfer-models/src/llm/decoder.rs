//! The routed-MoE decoder forward shared by every LLM architecture: token
//! embedding, dense weight residency, routed-expert residency (VRAM slot
//! pool, page-locked host tier, read-ahead from a next-layer prediction), the
//! two submit structures (wgpu: two submits per layer; accelerator: one
//! stream per forward with a router sync per layer), the head, and the
//! per-sequence state. An architecture supplies [`Blocks`]: its dense
//! weights, the token mixer + router (+ dense / shared FFN), the next-layer
//! router prediction and the combine of the routed experts' output.

use std::sync::Arc;

use thinfer_core::Backend;
use thinfer_core::backend::{BufRef, WgpuBackend, WgpuError};
use thinfer_core::ops::llm::LlmWeight;
use thinfer_core::quant::{QuantKind, dequantize_row};
use thinfer_core::residency::{
    GpuView, TransposePolicy, WeightHandle, WeightMeta, WeightResidency,
};
use thinfer_core::secmem::Locked;
use thinfer_core::tensor::{GpuBufferId, StorageEncoding};
use thinfer_core::weight::{WeightId, WeightReader, WeightSource};
use thinfer_core::workspace::{BatchScope, Workspace, WsBuf};
use zeroize::{Zeroize, Zeroizing};

use crate::llm::experts::{DOWN, ExpertCache, ExpertError, ExpertTensors, GATE, Prefetch, UP};
use crate::llm::kernels::{Act, LlmPipelines, Mat, act_bytes};

pub type Buf<'s> = thinfer_core::workspace::BatchBuf<'s>;
pub type Scope<'s> = BatchScope<'s, WgpuBackend>;

#[derive(Debug)]
pub enum LlmError {
    Gpu(WgpuError),
    Weights(String),
    Experts(ExpertError),
}

impl From<WgpuError> for LlmError {
    fn from(e: WgpuError) -> Self {
        Self::Gpu(e)
    }
}

impl From<ExpertError> for LlmError {
    fn from(e: ExpertError) -> Self {
        Self::Experts(e)
    }
}

/// A registered dense weight and its encoding.
#[derive(Clone, Copy, Debug)]
pub struct W {
    pub h: WeightHandle,
    pub fmt: LlmWeight,
}

/// Which outputs a forward returns.
#[derive(Clone, Copy, Debug, Default)]
pub struct Want {
    /// Read back the residual stream after every layer (parity).
    pub layer_taps: bool,
    /// Compute + read back the last row's logits.
    pub logits: bool,
    /// Read back this layer's intermediates into [`Output::diag`] (wgpu path).
    pub diag_layer: Option<u32>,
}

#[derive(Default)]
pub struct Output {
    pub layer_taps: Vec<Vec<f32>>,
    /// Named intermediates of `Want::diag_layer` (f32, row-major).
    pub diag: Vec<(String, Vec<f32>)>,
    /// Job data (it encodes the context): wiped on drop.
    pub logits: Zeroizing<Vec<f32>>,
}

impl Output {
    /// One line per captured tensor: non-finite count and max magnitude.
    pub fn summary(&self) -> String {
        let line = |n: &str, v: &[f32]| {
            let bad = v.iter().filter(|x| !x.is_finite()).count();
            let max = v.iter().fold(0f32, |m, x| m.max(x.abs()));
            format!("{n}: n={} non-finite={bad} max|x|={max}", v.len())
        };
        let mut l: Vec<String> = self.diag.iter().map(|(n, v)| line(n, v)).collect();
        for (i, v) in self.layer_taps.iter().enumerate() {
            l.push(line(&format!("layer{i}"), v));
        }
        l.join("\n")
    }
}

/// A readback as `T`s (logits, routed expert ids: job data), wiping the raw bytes.
fn take<T: bytemuck::Pod + Zeroize>(mut bytes: Vec<u8>) -> Zeroizing<Vec<T>> {
    let v = Zeroizing::new(bytemuck::cast_slice(&bytes).to_vec());
    bytes.zeroize();
    v
}

/// Accumulated host wall seconds per forward phase (each phase ends on a GPU
/// completion, so this is the critical path split).
#[derive(Clone, Copy, Debug, Default)]
pub struct PhaseTimes {
    /// Accelerator path: waiting for each layer's router ids (GPU drain).
    pub sync: f64,
    /// Expert residency (source reads + uploads).
    pub fill: f64,
    /// Encode + submit + GPU completion of the pipelined layer scopes.
    pub gpu: f64,
    /// Accelerator path: acquiring the dense weights (part of `gpu`).
    pub dense: f64,
    /// Read-ahead accuracy: a layer's routed experts (unique per forward)
    /// that the previous layer's prediction named.
    pub predicted: u64,
    pub routed: u64,
}

/// Shape facts the driver needs.
#[derive(Clone, Debug)]
pub struct Dims {
    pub hidden: u32,
    pub layers: u32,
    pub experts: u32,
    pub top_k: u32,
    pub moe_ff: u32,
    pub vocab: u32,
    pub eps: f32,
    /// Routed-expert activation: GELU (tanh) instead of SiLU.
    pub gelu: bool,
}

/// What a layer's mixer half leaves for its routed-expert half.
pub struct Mid<'s> {
    /// Routed experts' input rows.
    pub hn: Buf<'s>,
    /// Router top-k ids / weights, `[t, top_k]`.
    pub ids: Buf<'s>,
    pub wts: Buf<'s>,
    /// Router logits (diagnostics).
    pub logits: Buf<'s>,
    /// Everything the combine reads besides the expert outputs.
    pub combine: Combine<'s>,
}

/// The combine's inputs: the residual stream after the mixer, the routing
/// weights and arch-specific extras (e.g. the shared expert's output).
pub struct Combine<'s> {
    pub x2: Buf<'s>,
    pub wts: Buf<'s>,
    pub aux: Vec<Buf<'s>>,
}

/// One architecture's layers.
pub trait Blocks {
    fn dims(&self) -> &Dims;
    /// Dense weights of layer `li`, in the order the encode hooks index its
    /// views.
    fn layer_weights(&self, li: usize) -> Vec<W>;
    /// Dense weights of the output head, in the order [`Self::encode_head`]
    /// indexes its views.
    fn head_weights(&self) -> Vec<W>;
    /// Per-layer state buffers (KV caches, recurrent state), in the order
    /// the mixer indexes them. All start zeroed.
    fn state_layout(&self, li: usize) -> Vec<StateBuf>;
    /// Rope cos/sin tables for rows at positions `pos0..` (one per rope
    /// kind the mixer uses).
    fn rope_tables(&self, pos0: u32, rows: u32) -> Vec<Vec<f32>>;
    /// Multiplier on the token embedding rows.
    fn embed_scale(&self) -> f32 {
        1.0
    }
    /// `tanh(x / cap) * cap` on the logits.
    fn logit_softcap(&self) -> Option<f32> {
        None
    }
    /// Token mixer, router + top-k (into `ids_out` when given) and any dense
    /// FFN half of layer `li` on residual `x`.
    #[allow(clippy::too_many_arguments)]
    fn encode_mixer<'s>(
        &self,
        s: &Scope<'s>,
        p: &LlmPipelines,
        state: &[LayerBuf],
        pos: u32,
        li: usize,
        v: &[GpuView<'_>],
        x: Buf<'s>,
        rope: &[WsBuf<WgpuBackend>],
        t: u32,
        ids_out: Option<Buf<'s>>,
    ) -> Result<Mid<'s>, LlmError>;
    /// Layer `li`'s router top-k into `ids`, run on the previous layer's
    /// `mid` (a prediction of `li`'s experts one layer early, for read-ahead).
    fn encode_prediction<'s>(
        &self,
        s: &Scope<'s>,
        p: &LlmPipelines,
        li: usize,
        v: &[GpuView<'_>],
        mid: &Mid<'s>,
        t: u32,
        ids: Buf<'s>,
    ) -> Result<(), LlmError>;
    /// The next residual from the routed experts' output `ed` (`[t * top_k,
    /// hidden]`, pair order) and the combine inputs.
    #[allow(clippy::too_many_arguments)]
    fn encode_combine<'s>(
        &self,
        s: &Scope<'s>,
        p: &LlmPipelines,
        li: usize,
        v: &[GpuView<'_>],
        c: &Combine<'s>,
        ed: Buf<'s>,
        t: u32,
    ) -> Result<Buf<'s>, LlmError>;
    /// Logits `[1, vocab]` of residual row `x`.
    fn encode_head<'s>(
        &self,
        s: &Scope<'s>,
        p: &LlmPipelines,
        v: &[GpuView<'_>],
        x: Act<'s>,
    ) -> Result<Buf<'s>, LlmError>;
}

/// Most rows one forward step runs. Sliding-window KV rings hold this many
/// rows past their window, so a step's new keys never overwrite a key its own
/// queries still read; [`Decoder::forward`] splits longer inputs.
pub const MAX_STEP: u32 = 256;

/// How one per-sequence state buffer is shaped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateBuf {
    /// `row` bytes per position, indexed by absolute position (KV caches):
    /// a sequence's prefix rows never change once written.
    Rows(u64),
    /// `row` bytes per position, of which only the last `window` are ever
    /// read again (sliding-window KV): a ring of `window + MAX_STEP` rows.
    Window { row: u64, window: u32 },
    /// Fixed size (recurrent / conv state): only its value at the sequence's
    /// end matters.
    Fixed(u64),
}

impl StateBuf {
    /// Rows held for a `max_ctx` sequence (0 for fixed state).
    fn capacity(self, max_ctx: u32) -> u32 {
        match self {
            Self::Rows(_) => max_ctx,
            Self::Window { window, .. } => max_ctx.min(window + MAX_STEP),
            Self::Fixed(_) => 0,
        }
    }

    pub fn bytes(self, max_ctx: u32) -> u64 {
        match self {
            Self::Rows(row) | Self::Window { row, .. } => row * self.capacity(max_ctx) as u64,
            Self::Fixed(b) => b,
        }
    }

    /// The positions of `[start, end)` a [`StateChunk`] keeps: all of them,
    /// or a window's last `window` (no later query reads further back).
    fn kept(self, start: u32, end: u32) -> std::ops::Range<u32> {
        match self {
            Self::Rows(_) => start..end,
            Self::Window { window, .. } => start.max(end.saturating_sub(window))..end,
            Self::Fixed(_) => end..end,
        }
    }

    /// Bytes of a [`StateChunk`] covering `[start, end)`.
    fn chunk_bytes(self, start: u32, end: u32) -> u64 {
        match self {
            Self::Rows(row) | Self::Window { row, .. } => row * self.kept(start, end).len() as u64,
            Self::Fixed(b) => b,
        }
    }
}

/// One state buffer of a layer, as the encode hooks see it.
pub struct LayerBuf {
    pub buf: WsBuf<WgpuBackend>,
    /// Ring capacity in rows (position `p` in row `p % ring`); 0 = linear or
    /// fixed state.
    pub ring: u32,
}

/// `(offset, len)` byte segments of positions `range` in a buffer of
/// `row`-byte rows that is a ring of `ring` rows (0 = linear): one segment,
/// or two when the range wraps.
fn row_segments(range: std::ops::Range<u32>, row: u64, ring: u32) -> Vec<(u64, u64)> {
    let n = range.end - range.start;
    if n == 0 {
        return Vec::new();
    }
    if ring == 0 {
        return vec![(range.start as u64 * row, n as u64 * row)];
    }
    assert!(n <= ring, "range longer than the ring");
    let a = range.start % ring;
    let first = (ring - a).min(n);
    let mut out = vec![(a as u64 * row, first as u64 * row)];
    if first < n {
        out.push((0, (n - first) as u64 * row));
    }
    out
}

/// A segment of a sequence's state: every per-position buffer's rows for
/// positions `[start, end)` (a window buffer's last `window` of them) plus
/// every fixed buffer's value at `end`, layer by layer in layout order. A chain of chunks from position 0 restores a
/// sequence exactly (see [`LlmState::import`]). Job data: wiped on drop.
pub struct StateChunk {
    pub start: u32,
    pub end: u32,
    /// Page-locked (never swapped), wiped on drop.
    pub bytes: Locked<u8>,
}

/// Per-sequence state: each layer's buffers (KV caches, recurrent state)
/// and the next position. Job data: zeroed on the GPU on drop, before the
/// buffers go back to the pool.
pub struct LlmState {
    pub pos: u32,
    pub max_ctx: u32,
    backend: Arc<WgpuBackend>,
    layers: Vec<Vec<LayerBuf>>,
    layout: Vec<Vec<StateBuf>>,
}

impl LlmState {
    fn buffer_ids(&self) -> Vec<GpuBufferId> {
        self.layers.iter().flatten().map(|b| b.buf.id()).collect()
    }

    fn buffers(&self) -> impl Iterator<Item = (&LayerBuf, StateBuf)> {
        self.layers
            .iter()
            .flatten()
            .zip(self.layout.iter().flatten().copied())
    }

    /// Whether some window ring is shorter than the context (so a forward
    /// step must stay within [`MAX_STEP`] rows).
    fn wraps(&self) -> bool {
        self.buffers()
            .any(|(b, l)| matches!(l, StateBuf::Window { .. }) && b.ring < self.max_ctx)
    }

    /// The chunk of positions `[start, pos)` (plus the fixed state at `pos`).
    pub async fn export(&self, start: u32) -> Result<StateChunk, LlmError> {
        assert!(start <= self.pos, "export start past the sequence end");
        let end = self.pos;
        let total: u64 = self.buffers().map(|(_, l)| l.chunk_bytes(start, end)).sum();
        let mut bytes = Locked::<u8>::zeroed(total as usize);
        let mut at = 0usize;
        for (b, l) in self.buffers() {
            let segments = match l {
                StateBuf::Rows(row) | StateBuf::Window { row, .. } => {
                    row_segments(l.kept(start, end), row, b.ring)
                }
                StateBuf::Fixed(n) => vec![(0, n)],
            };
            for (offset, len) in segments {
                let part = Zeroizing::new(self.backend.read_buffer(b.buf.id(), offset, len).await?);
                bytes[at..at + part.len()].copy_from_slice(&part);
                at += part.len();
            }
        }
        debug_assert_eq!(at, bytes.len());
        Ok(StateChunk {
            start,
            end: self.pos,
            bytes,
        })
    }

    /// Restore a sequence from a chain of chunks starting at position 0
    /// (each starting where the previous ended); the fixed buffers take the
    /// last chunk's values. The state must be fresh (`pos == 0`).
    pub fn import(&mut self, chain: &[&StateChunk]) -> Result<(), LlmError> {
        let bad = |m: &str| Err(LlmError::Weights(format!("session state: {m}")));
        if self.pos != 0 {
            return bad("import into a used state");
        }
        let mut at = 0;
        for c in chain {
            if c.start != at || c.end < c.start {
                return bad("chunks are not a contiguous chain from 0");
            }
            let want: u64 = self
                .buffers()
                .map(|(_, l)| l.chunk_bytes(c.start, c.end))
                .sum();
            if c.bytes.len() as u64 != want {
                return bad("chunk size does not match this model's state layout");
            }
            at = c.end;
        }
        if at > self.max_ctx {
            return bad("chain is longer than the context");
        }
        let last = chain.len().saturating_sub(1);
        // In chain order, so a ring keeps each slot's latest position.
        for (ci, c) in chain.iter().enumerate() {
            let mut off = 0usize;
            for (b, l) in self.buffers() {
                let len = l.chunk_bytes(c.start, c.end) as usize;
                let part = &c.bytes[off..off + len];
                off += len;
                match l {
                    StateBuf::Rows(row) | StateBuf::Window { row, .. } => {
                        let mut from = 0usize;
                        for (offset, n) in row_segments(l.kept(c.start, c.end), row, b.ring) {
                            let n = n as usize;
                            self.backend
                                .write_buffer(b.buf.id(), offset, &part[from..from + n])?;
                            from += n;
                        }
                    }
                    StateBuf::Fixed(_) if ci == last => {
                        self.backend.write_buffer(b.buf.id(), 0, part)?
                    }
                    StateBuf::Fixed(_) => {}
                }
            }
        }
        // Flush the queued writes (a submit) and wait: the CUDA path reads
        // these buffers without a wgpu submit in between.
        self.backend.clear_buffers(&[])?;
        self.backend.wait_idle();
        self.pos = at;
        Ok(())
    }
}

impl Drop for LlmState {
    fn drop(&mut self) {
        if let Err(e) = self.backend.clear_buffers(&self.buffer_ids()) {
            tracing::error!(target: thinfer_core::trace::DIAG, error = ?e, "llm state wipe failed");
        }
    }
}

/// A small GPU buffer the model owns (weights it derives at load), freed on
/// drop.
pub struct OwnedBuf {
    backend: Arc<WgpuBackend>,
    pub id: GpuBufferId,
    pub len: u64,
}

impl OwnedBuf {
    pub fn f32s(backend: &Arc<WgpuBackend>, v: &[f32]) -> Result<Self, LlmError> {
        let len = (v.len() * 4) as u64;
        let id = backend.allocate_in(len, thinfer_core::mem::VramCategory::Weights)?;
        backend.write_buffer(id, 0, bytemuck::cast_slice(v))?;
        Ok(Self {
            backend: backend.clone(),
            id,
            len,
        })
    }

    pub fn buf_ref(&self) -> BufRef {
        BufRef::new(self.id, self.len)
    }
}

impl Drop for OwnedBuf {
    fn drop(&mut self) {
        self.backend.free(self.id);
    }
}

/// Registers GGUF tensors with a residency (dense weights) and describes
/// routed-expert tensors.
pub struct Registry<'r, S: WeightSource> {
    pub residency: &'r WeightResidency<S>,
}

impl<S: WeightSource> Registry<'_, S> {
    fn entry(&self, id: &WeightId) -> Result<thinfer_core::weight::WeightEntry, LlmError> {
        self.residency
            .source()
            .catalog()
            .get(id)
            .cloned()
            .ok_or_else(|| LlmError::Weights(format!("missing tensor {}", id.0)))
    }

    /// Register dense weight `name`.
    pub fn dense(&self, name: &str) -> Result<W, LlmError> {
        let id = WeightId(name.to_string());
        let e = self.entry(&id)?;
        let fmt = fmt_of(e.encoding, &id)?;
        let h = self.residency.register(WeightMeta {
            id,
            shape: e.shape.clone(),
            encoding: e.encoding.expect("checked by fmt_of"),
            on_disk_bytes: e.size,
            transpose: TransposePolicy::None,
            transcode: None,
            keep_f32: true,
        });
        Ok(W { h, fmt })
    }

    /// Format and bytes per expert of a 3D `[experts, n, k]` expert tensor.
    pub fn experts(&self, name: &str) -> Result<(WeightId, LlmWeight, u64), LlmError> {
        let id = WeightId(name.to_string());
        let e = self.entry(&id)?;
        let fmt = fmt_of(e.encoding, &id)?;
        let (n, k) = (e.shape.0[1] as u64, e.shape.0[2] as u64);
        Ok((id, fmt, n * fmt.row_bytes(k)))
    }

    /// A small f32 tensor read whole to the host (norm / scale vectors the
    /// model folds into buffers it owns).
    pub async fn f32_vec(&self, name: &str) -> Result<Vec<f32>, LlmError> {
        let id = WeightId(name.to_string());
        let e = self.entry(&id)?;
        if e.encoding != Some(StorageEncoding::F32) {
            return Err(LlmError::Weights(format!("{name}: expected f32")));
        }
        let mut r = self
            .residency
            .source()
            .open(&id)
            .await
            .map_err(|e| LlmError::Weights(format!("{e:?}")))?;
        let mut b = vec![0u8; e.size as usize];
        r.read_at(0, &mut b)
            .await
            .map_err(|e| LlmError::Weights(format!("{e:?}")))?;
        Ok(bytemuck::cast_slice(&b).to_vec())
    }
}

fn fmt_of(enc: Option<StorageEncoding>, id: &WeightId) -> Result<LlmWeight, LlmError> {
    match enc {
        Some(StorageEncoding::F32) => Ok(LlmWeight::F32),
        Some(StorageEncoding::Quant(q)) => Ok(LlmWeight::Quant(q)),
        other => Err(LlmError::Weights(format!(
            "{}: unsupported encoding {other:?}",
            id.0
        ))),
    }
}

/// RAM kept out of the host expert cache for the rest of a request
/// (embedding rows, logits, tokenizer, staging).
const HOST_RESERVE: u64 = 1 << 30;

/// Activation scratch reserved out of the VRAM budget on top of weights and
/// state (one prefill chunk's buffers, uniforms, staging).
const WORKSPACE_RESERVE: u64 = 768 << 20;

/// Enqueue an accelerator copy of `len` bytes from `src` to `dst` (deferred
/// mode, stream ordered).
fn copy_on_stream<'s>(s: &Scope<'s>, dst: Buf<'s>, src: Buf<'s>, len: u64) -> Result<(), LlmError> {
    let (Some(d), Some(sa), Some(acc)) = (s.accel_addr(dst), s.accel_addr(src), s.accelerator())
    else {
        return Err(LlmError::Weights("copy: not device-addressable".into()));
    };
    acc.copy(d, sa, len).map_err(LlmError::Weights)?;
    Ok(())
}

/// Wait for the accelerator stream and read `n` u32 from `buf`.
fn read_on_stream<'s>(s: &Scope<'s>, buf: Buf<'s>, n: usize) -> Result<Vec<u32>, LlmError> {
    let (Some(a), Some(acc)) = (s.accel_addr(buf), s.accelerator()) else {
        return Err(LlmError::Weights("read: not device-addressable".into()));
    };
    let mut v = vec![0u32; n];
    acc.read(a, bytemuck::cast_slice_mut(&mut v))
        .map_err(LlmError::Weights)?;
    Ok(v)
}

/// Sorted distinct expert ids of a router readback (validated).
fn unique_experts(mut ids: Vec<u32>, experts: u32, layer: usize) -> Result<Vec<u32>, LlmError> {
    if let Some(&bad) = ids.iter().find(|&&e| e >= experts) {
        return Err(LlmError::Weights(format!(
            "layer {layer}: router produced expert id {bad}"
        )));
    }
    ids.sort_unstable();
    ids.dedup();
    Ok(ids)
}

pub async fn acquire<'a, S: WeightSource>(
    residency: &'a WeightResidency<S>,
    backend: &WgpuBackend,
    w: W,
) -> Result<GpuView<'a>, LlmError> {
    residency
        .acquire(w.h, backend)
        .await
        .map_err(|e| LlmError::Weights(format!("{e:?}")))
}

async fn views<'a, S: WeightSource>(
    residency: &'a WeightResidency<S>,
    backend: &WgpuBackend,
    ws: Vec<W>,
) -> Result<Vec<GpuView<'a>>, LlmError> {
    let mut v = Vec::with_capacity(ws.len());
    for w in ws {
        v.push(acquire(residency, backend, w).await?);
    }
    Ok(v)
}

/// A dense weight view as a gemv matrix.
pub fn mat<'s>(s: &Scope<'s>, v: &GpuView<'_>, fmt: LlmWeight) -> Mat<'s> {
    Mat {
        buf: s.import_copy(v.buf()),
        fmt,
        slot_bytes: 0,
    }
}

/// [`Combine`] carried across a submit (wgpu path).
struct Carry {
    hn: WsBuf<WgpuBackend>,
    x2: WsBuf<WgpuBackend>,
    wts: WsBuf<WgpuBackend>,
    aux: Vec<WsBuf<WgpuBackend>>,
}

pub struct Decoder<S: WeightSource, B: Blocks> {
    pub blocks: B,
    residency: WeightResidency<S>,
    embed: (WeightId, QuantKind),
    pub experts: ExpertCache,
    /// Host wall time per forward phase, accumulated (see [`PhaseTimes`]).
    pub times: PhaseTimes,
}

impl<S: WeightSource, B: Blocks> Decoder<S, B> {
    /// Pin every dense weight (they are touched every token) and size the
    /// routed-expert pool from what the VRAM budget leaves after them, the
    /// state and the workspace.
    pub fn new(
        backend: &Arc<WgpuBackend>,
        residency: WeightResidency<S>,
        blocks: B,
        experts: Vec<ExpertTensors>,
        max_ctx: u32,
    ) -> Result<Self, LlmError> {
        let d = blocks.dims().clone();
        let embed_id = WeightId("token_embd.weight".into());
        let embed_q = match residency
            .source()
            .catalog()
            .get(&embed_id)
            .map(|e| e.encoding)
        {
            Some(Some(StorageEncoding::Quant(q))) => q,
            other => {
                return Err(LlmError::Weights(format!(
                    "token_embd: unsupported encoding {other:?}"
                )));
            }
        };
        let dense = residency.total_registered_bytes();
        let mut all: Vec<WeightHandle> = (0..d.layers as usize)
            .flat_map(|i| blocks.layer_weights(i))
            .chain(blocks.head_weights())
            .map(|w| w.h)
            .collect();
        all.dedup();
        residency.set_pin_plan(&all, dense);

        let state: u64 = (0..d.layers as usize)
            .flat_map(|i| blocks.state_layout(i))
            .map(|b| b.bytes(max_ctx))
            .sum();
        let left = residency
            .budget()
            .vram_bytes
            .saturating_sub(dense + state + WORKSPACE_RESERVE);
        let slot = ExpertCache::slot_bytes(&experts).iter().sum::<u64>().max(1);
        let capacity = (left / slot) as usize;
        let need = d.experts as usize;
        if capacity < need {
            return Err(LlmError::Weights(format!(
                "VRAM budget too small: {capacity} expert slots fit, one layer can route to {need}"
            )));
        }
        let experts = ExpertCache::new(backend, experts, capacity)?;
        Ok(Self {
            blocks,
            residency,
            embed: (embed_id, embed_q),
            experts,
            times: PhaseTimes::default(),
        })
    }

    pub fn residency(&self) -> &WeightResidency<S> {
        &self.residency
    }

    /// Fresh per-sequence state (all buffers zeroed).
    pub fn new_state(
        &self,
        backend: &Arc<WgpuBackend>,
        ws: &Workspace<WgpuBackend>,
        max_ctx: u32,
    ) -> Result<LlmState, LlmError> {
        self.state_with(backend, ws, max_ctx, true)
    }

    /// [`Self::new_state`] with every window buffer held whole (linear, no
    /// ring): the reference layout the ring must reproduce exactly. Its
    /// chunks are not interchangeable with ring states' chunks.
    pub fn new_state_unwindowed(
        &self,
        backend: &Arc<WgpuBackend>,
        ws: &Workspace<WgpuBackend>,
        max_ctx: u32,
    ) -> Result<LlmState, LlmError> {
        self.state_with(backend, ws, max_ctx, false)
    }

    fn state_with(
        &self,
        backend: &Arc<WgpuBackend>,
        ws: &Workspace<WgpuBackend>,
        max_ctx: u32,
        windowed: bool,
    ) -> Result<LlmState, LlmError> {
        let layout: Vec<Vec<StateBuf>> = (0..self.blocks.dims().layers as usize)
            .map(|li| {
                let mut l = self.blocks.state_layout(li);
                if !windowed {
                    for b in &mut l {
                        if let StateBuf::Window { row, .. } = *b {
                            *b = StateBuf::Rows(row);
                        }
                    }
                }
                l
            })
            .collect();
        let mut layers = Vec::new();
        for l in &layout {
            layers.push(
                l.iter()
                    .map(|b| {
                        Ok::<_, LlmError>(LayerBuf {
                            buf: ws.alloc(b.bytes(max_ctx))?,
                            ring: match b {
                                StateBuf::Window { .. } => b.capacity(max_ctx),
                                _ => 0,
                            },
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        let state = LlmState {
            pos: 0,
            max_ctx,
            backend: backend.clone(),
            layers,
            layout,
        };
        // Pooled buffers carry stale bytes; recurrent state must start at 0.
        // Waited: the CUDA path reads these without a wgpu submit in between.
        backend.clear_buffers(&state.buffer_ids())?;
        backend.wait_idle();
        Ok(state)
    }

    /// Embedding rows for `tokens` (CPU dequant gather), f32 `[T, hidden]`.
    async fn embed(&self, tokens: &[u32]) -> Result<Zeroizing<Vec<f32>>, LlmError> {
        let h = self.blocks.dims().hidden as usize;
        let (id, q) = (&self.embed.0, self.embed.1);
        let rb = q.bytes_for_elements(h as u64) as usize;
        let mut r = self
            .residency
            .source()
            .open(id)
            .await
            .map_err(|e| LlmError::Weights(format!("{e:?}")))?;
        let mut out = Zeroizing::new(vec![0f32; tokens.len() * h]);
        let mut row = vec![0u8; rb];
        for (i, &t) in tokens.iter().enumerate() {
            r.read_at(t as u64 * rb as u64, &mut row)
                .await
                .map_err(|e| LlmError::Weights(format!("{e:?}")))?;
            dequantize_row(q, &row, &mut out[i * h..(i + 1) * h]);
        }
        let scale = self.blocks.embed_scale();
        if scale != 1.0 {
            out.iter_mut().for_each(|v| *v *= scale);
        }
        Ok(out)
    }

    /// Embeddings + rope tables of `tokens` at the state's position.
    async fn inputs(
        &self,
        backend: &WgpuBackend,
        ws: &Workspace<WgpuBackend>,
        state: &LlmState,
        tokens: &[u32],
    ) -> Result<(WsBuf<WgpuBackend>, Vec<WsBuf<WgpuBackend>>), LlmError> {
        let t = tokens.len() as u32;
        let emb = self.embed(tokens).await?;
        let x0 = ws.alloc(act_bytes(t, self.blocks.dims().hidden))?;
        backend.write_buffer(x0.id(), 0, bytemuck::cast_slice(&emb[..]))?;
        let mut rope = Vec::new();
        for table in self.blocks.rope_tables(state.pos, t) {
            let b = ws.alloc((table.len() * 4) as u64)?;
            backend.write_buffer(b.id(), 0, bytemuck::cast_slice(&table))?;
            rope.push(b);
        }
        Ok((x0, rope))
    }

    /// Run the forward on the accelerator's LLM kernels when it has them:
    /// one stream per token with a short per-layer sync to fill the routed
    /// experts by DMA from a page-locked host cache sized by the RAM budget
    /// (less [`HOST_RESERVE`] for the rest of the request). `Ok(false)` = no
    /// accelerator LLM kernels (the wgpu path stays).
    pub fn enable_accel(&mut self, backend: &WgpuBackend) -> Result<bool, LlmError> {
        if !backend.accelerator().is_some_and(|a| a.accepts_llm()) {
            return Ok(false);
        }
        let host = self
            .residency
            .budget()
            .ram_bytes
            .saturating_sub(HOST_RESERVE)
            .min(self.experts_total_bytes());
        let experts = self.blocks.dims().experts;
        self.experts.enable_accel(backend, experts, host)?;
        Ok(true)
    }

    /// Bytes of every routed expert.
    pub fn experts_total_bytes(&self) -> u64 {
        self.experts.total_bytes(self.blocks.dims().experts)
    }

    /// Run `tokens` (appended at `state.pos`) through the model. Inputs
    /// longer than [`MAX_STEP`] run as several steps when a window ring
    /// wraps (the output is the last step's).
    pub async fn forward(
        &mut self,
        backend: &WgpuBackend,
        p: &LlmPipelines,
        ws: &Workspace<WgpuBackend>,
        state: &mut LlmState,
        tokens: &[u32],
        want: Want,
    ) -> Result<Output, LlmError> {
        assert!(
            state.pos + tokens.len() as u32 <= state.max_ctx,
            "context overflow"
        );
        if tokens.len() as u32 > MAX_STEP && state.wraps() {
            assert!(
                !want.layer_taps && want.diag_layer.is_none(),
                "per-layer readback needs a single forward step"
            );
            let mut out = Output::default();
            for (i, step) in tokens.chunks(MAX_STEP as usize).enumerate() {
                let last = (i + 1) * MAX_STEP as usize >= tokens.len();
                let w = Want {
                    logits: want.logits && last,
                    ..want
                };
                out = self.forward_step(backend, p, ws, state, step, w).await?;
            }
            return Ok(out);
        }
        self.forward_step(backend, p, ws, state, tokens, want).await
    }

    /// One forward step of `tokens` (at most [`MAX_STEP`] when a ring wraps).
    async fn forward_step(
        &mut self,
        backend: &WgpuBackend,
        p: &LlmPipelines,
        ws: &Workspace<WgpuBackend>,
        state: &mut LlmState,
        tokens: &[u32],
        want: Want,
    ) -> Result<Output, LlmError> {
        let mut out = if self.experts.is_accel() {
            self.forward_accel(backend, p, ws, state, tokens, want)
                .await?
        } else {
            self.forward_wgpu(backend, p, ws, state, tokens, want)
                .await?
        };
        if let Some(cap) = self.blocks.logit_softcap() {
            out.logits
                .iter_mut()
                .for_each(|v| *v = (*v / cap).tanh() * cap);
        }
        state.pos += tokens.len() as u32;
        Ok(out)
    }

    /// Whole forward as one deferred accelerator stream. After each layer's
    /// router the stream is synced and its ids read, the missing experts are
    /// DMA-copied into VRAM slots (address table rewritten), then the expert
    /// kernels read them from VRAM. The next layer's predicted experts are
    /// read from the source meanwhile.
    async fn forward_accel(
        &mut self,
        backend: &WgpuBackend,
        p: &LlmPipelines,
        ws: &Workspace<WgpuBackend>,
        state: &mut LlmState,
        tokens: &[u32],
        want: Want,
    ) -> Result<Output, LlmError> {
        let d = self.blocks.dims().clone();
        let t = tokens.len() as u32;
        let (hd, n) = (d.hidden, d.layers as usize);
        let pairs = t * d.top_k;
        let mut out = Output::default();
        let (x0, rope) = self.inputs(backend, ws, state, tokens).await?;

        let t_gpu = std::time::Instant::now();
        let mut layer_views = Vec::with_capacity(n);
        for li in 0..n {
            layer_views.push(views(&self.residency, backend, self.blocks.layer_weights(li)).await?);
        }
        let head_views = views(&self.residency, backend, self.blocks.head_weights()).await?;
        self.times.dense += t_gpu.elapsed().as_secs_f64();
        let s = ws.batch();
        if !s.accel_begin()? {
            return Err(LlmError::Weights("accelerator unavailable".into()));
        }
        let log = s.alloc(n as u64 * pairs as u64 * 4)?;
        // Each layer's router run one layer early (on the previous layer's
        // state): the experts to read ahead while the GPU works.
        let pred_log = s.alloc(n as u64 * pairs as u64 * 4)?;
        let taps = if want.layer_taps {
            Some(s.alloc(n as u64 * act_bytes(t, hd))?)
        } else {
            None
        };
        let mut x = s.import_copy(x0.as_buf_ref());
        let source = self.residency.source();
        let mut ahead: Option<(Vec<u32>, Option<Prefetch<'_>>)> = None;
        for li in 0..n {
            let rows = |b| s.subview(b, li as u64 * pairs as u64 * 4, pairs as u64 * 4);
            let ids = rows(&log);
            let v = &layer_views[li];
            let m = self.blocks.encode_mixer(
                &s,
                p,
                &state.layers[li],
                state.pos,
                li,
                v,
                x,
                &rope,
                t,
                Some(ids),
            )?;
            let pred = match layer_views.get(li + 1) {
                Some(next) => {
                    let ids = rows(&pred_log);
                    self.blocks
                        .encode_prediction(&s, p, li + 1, next, &m, t, ids)?;
                    Some(ids)
                }
                None => None,
            };
            let t_sync = std::time::Instant::now();
            let routed = read_on_stream(&s, ids, pairs as usize)?;
            let pred = match pred {
                Some(b) => Some(read_on_stream(&s, b, pairs as usize)?),
                None => None,
            };
            self.times.sync += t_sync.elapsed().as_secs_f64();
            let t_fill = std::time::Instant::now();
            let uniq = unique_experts(routed, d.experts, li)?;
            if let Some((named, fetch)) = ahead.take() {
                self.times.routed += uniq.len() as u64;
                self.times.predicted += uniq
                    .iter()
                    .filter(|e| named.binary_search(e).is_ok())
                    .count() as u64;
                if let Some(f) = fetch {
                    self.experts.finish(f).await?;
                }
            }
            self.experts.ensure(backend, source, li, &uniq).await?;
            // Read the next layer's predicted experts while the GPU runs this
            // layer's experts and the next mixer.
            if let Some(pred) = pred {
                let named = unique_experts(pred, d.experts, li + 1)?;
                // Host-resident ones over the bus now, the rest from the source.
                self.experts.prefetch_vram(backend, li + 1, &named)?;
                let fetch = self.experts.prefetch(source, li + 1, &named);
                ahead = Some((named, fetch));
            }
            self.times.fill += t_fill.elapsed().as_secs_f64();
            let ed = self.encode_experts_table(&s, p, li, m.hn, m.ids, t)?;
            x = self
                .blocks
                .encode_combine(&s, p, li, v, &m.combine, ed, t)?;
            for b in [m.hn, m.logits, ed, m.wts] {
                s.recycle(b);
            }
            if let Some(tb) = taps {
                let bytes = act_bytes(t, hd);
                let dst = s.subview(&tb, li as u64 * bytes, bytes);
                copy_on_stream(&s, dst, x, bytes)?;
            }
        }
        let lg = self
            .blocks
            .encode_head(&s, p, &head_views, Act::dense(x, hd).at((t - 1) * hd))?;
        let mut keep = vec![lg];
        keep.extend(taps);
        let outs = s.submit_many(&keep).await?;
        drop((layer_views, head_views));
        self.times.gpu += t_gpu.elapsed().as_secs_f64();

        if want.logits {
            out.logits = take(backend.read_buffer(outs[0].id(), 0, outs[0].len()).await?);
        }
        if let Some(tb) = outs.get(1) {
            let all = backend.read_buffer(tb.id(), 0, tb.len()).await?;
            let all: &[f32] = bytemuck::cast_slice(&all);
            let per = (t * hd) as usize;
            out.layer_taps = all.chunks(per).map(<[f32]>::to_vec).collect();
        }
        Ok(out)
    }

    /// Routed experts of layer `li` through the device address table
    /// (accelerator path): `[t * top_k, hidden]` outputs in pair order.
    fn encode_experts_table<'s>(
        &self,
        s: &Scope<'s>,
        p: &LlmPipelines,
        li: usize,
        hn: Buf<'s>,
        ids: Buf<'s>,
        t: u32,
    ) -> Result<Buf<'s>, LlmError> {
        let d = self.blocks.dims();
        let (hd, kk, ff) = (d.hidden, d.top_k, d.moe_ff);
        let pairs = t * kk;
        let table = |k: usize| {
            let (b, off, len) = self.experts.table(li, k).expect("host tier attached");
            s.import_copy(BufRef::view(b, off, len))
        };
        let fmt = |k: usize| self.experts.fmt(li, k);
        let eg = s.alloc(act_bytes(pairs, ff))?;
        let eu = s.alloc(act_bytes(pairs, ff))?;
        for (k, y) in [(GATE, eg), (UP, eu)] {
            p.gemv_experts(
                s,
                fmt(k),
                table(k),
                ids,
                Act::dense(hn, hd),
                Act::dense(y, ff),
                pairs,
                kk,
                ff,
                hd,
            )?;
        }
        let ea = s.alloc(act_bytes(pairs, ff))?;
        p.glu(s, eg, eu, ea, pairs * ff, d.gelu)?;
        let ed = s.alloc(act_bytes(pairs, hd))?;
        p.gemv_experts(
            s,
            fmt(DOWN),
            table(DOWN),
            ids,
            Act::dense(ea, ff),
            Act::dense(ed, hd),
            pairs,
            1,
            hd,
            ff,
        )?;
        for b in [eg, eu, ea] {
            s.recycle(b);
        }
        Ok(ed)
    }

    /// Pipelined wgpu forward: each submit carries the previous layer's
    /// routed experts + combine and this layer's mixer and router, plus the
    /// router-id readback, so a layer costs one GPU round trip.
    async fn forward_wgpu(
        &mut self,
        backend: &WgpuBackend,
        p: &LlmPipelines,
        ws: &Workspace<WgpuBackend>,
        state: &mut LlmState,
        tokens: &[u32],
        want: Want,
    ) -> Result<Output, LlmError> {
        let d = self.blocks.dims().clone();
        let t = tokens.len() as u32;
        let (hd, n) = (d.hidden, d.layers as usize);
        let mut out = Output::default();
        let (x0, rope) = self.inputs(backend, ws, state, tokens).await?;

        let mut carry: Option<(Carry, Zeroizing<Vec<u32>>)> = None;
        for k in 0..=n {
            // Routed experts of layer k - 1 resident before its matmuls encode.
            let mut slots = Vec::new();
            if let Some((_, routed)) = &carry {
                let t_fill = std::time::Instant::now();
                slots = self.route_slots(backend, k - 1, routed).await?;
                self.times.fill += t_fill.elapsed().as_secs_f64();
            }
            let t_gpu = std::time::Instant::now();
            let lv = match k < n {
                true => Some(views(&self.residency, backend, self.blocks.layer_weights(k)).await?),
                false => None,
            };
            let prev = match k > 0 {
                true => {
                    Some(views(&self.residency, backend, self.blocks.layer_weights(k - 1)).await?)
                }
                false => None,
            };
            let head = match k == n && want.logits {
                true => Some(views(&self.residency, backend, self.blocks.head_weights()).await?),
                false => None,
            };
            let s = ws.batch();
            let mut reads = Vec::new();
            let x_in = match (&carry, &prev) {
                (Some((cy, _)), Some(pv)) => {
                    let hn = s.import_copy(cy.hn.as_buf_ref());
                    let ed = self.encode_experts_gather(&s, p, k - 1, hn, &slots, t)?;
                    let combine = Combine {
                        x2: s.import_copy(cy.x2.as_buf_ref()),
                        wts: s.import_copy(cy.wts.as_buf_ref()),
                        aux: cy
                            .aux
                            .iter()
                            .map(|b| s.import_copy(b.as_buf_ref()))
                            .collect(),
                    };
                    let xn = self
                        .blocks
                        .encode_combine(&s, p, k - 1, pv, &combine, ed, t)?;
                    if want.diag_layer == Some(k as u32 - 1) {
                        for (name, b) in [("experts_down", ed), ("x_out", xn)] {
                            reads.push((
                                name.to_string(),
                                s.read_buffer_via_encoder(b, 0, b.len())?,
                            ));
                        }
                    }
                    if want.layer_taps {
                        reads.push((
                            "tap".to_string(),
                            s.read_buffer_via_encoder(xn, 0, xn.len())?,
                        ));
                    }
                    xn
                }
                _ => s.import_copy(x0.as_buf_ref()),
            };
            let mut ids_read = None;
            let mut logits_read = None;
            let next = match &lv {
                Some(v) => {
                    let m = self.blocks.encode_mixer(
                        &s,
                        p,
                        &state.layers[k],
                        state.pos,
                        k,
                        v,
                        x_in,
                        &rope,
                        t,
                        None,
                    )?;
                    ids_read = Some(s.read_buffer_via_encoder(m.ids, 0, m.ids.len())?);
                    if want.diag_layer == Some(k as u32) {
                        for (name, b) in [
                            ("x2", m.combine.x2),
                            ("hn", m.hn),
                            ("ids", m.ids),
                            ("wts", m.wts),
                            ("router_logits", m.logits),
                        ] {
                            reads.push((
                                name.to_string(),
                                s.read_buffer_via_encoder(b, 0, b.len())?,
                            ));
                        }
                    }
                    Some(m)
                }
                None => {
                    if let Some(hv) = &head {
                        let lg = self.blocks.encode_head(
                            &s,
                            p,
                            hv,
                            Act::dense(x_in, hd).at((t - 1) * hd),
                        )?;
                        logits_read = Some(s.read_buffer_via_encoder(lg, 0, lg.len())?);
                    }
                    None
                }
            };
            let new_carry = match next {
                Some(m) => {
                    let mut keep = vec![m.hn, m.combine.x2, m.combine.wts];
                    keep.extend(&m.combine.aux);
                    let (outs, done) = s.submit_many_deferred(&keep);
                    done.await?;
                    let mut it = outs.into_iter();
                    let (hn, x2, wts) = (
                        it.next().expect("carry"),
                        it.next().expect("carry"),
                        it.next().expect("carry"),
                    );
                    Some(Carry {
                        hn,
                        x2,
                        wts,
                        aux: it.collect(),
                    })
                }
                None => {
                    s.submit_void().await?;
                    None
                }
            };
            drop((lv, prev, head));
            self.times.gpu += t_gpu.elapsed().as_secs_f64();
            for (name, r) in reads {
                let v: Vec<f32> = bytemuck::cast_slice(&r.await?).to_vec();
                if name == "tap" {
                    out.layer_taps.push(v);
                } else {
                    out.diag.push((name, v));
                }
            }
            if let Some(r) = logits_read {
                out.logits = take(r.await?);
            }
            carry = match (new_carry, ids_read) {
                (Some(cy), Some(r)) => {
                    let routed: Zeroizing<Vec<u32>> = take(r.await?);
                    if let Some(&bad) = routed.iter().find(|&&e| e >= d.experts) {
                        return Err(LlmError::Weights(format!(
                            "layer {k}: router produced expert id {bad}\n{}",
                            out.summary()
                        )));
                    }
                    Some((cy, routed))
                }
                _ => None,
            };
        }
        Ok(out)
    }

    /// Make the experts `routed` (pairs, row-major `[t, k]`) of `layer`
    /// resident; returns the slot of each pair.
    async fn route_slots(
        &mut self,
        backend: &WgpuBackend,
        layer: usize,
        routed: &[u32],
    ) -> Result<Vec<u32>, LlmError> {
        let mut uniq = routed.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        let slots = self
            .experts
            .ensure(backend, self.residency.source(), layer, &uniq)
            .await?;
        Ok(routed
            .iter()
            .map(|e| slots[uniq.binary_search(e).expect("routed id")])
            .collect())
    }

    /// Routed experts of `li` through per-pair slot tables (wgpu path):
    /// `[t * top_k, hidden]` outputs in pair order.
    fn encode_experts_gather<'s>(
        &self,
        s: &Scope<'s>,
        p: &LlmPipelines,
        li: usize,
        hn: Buf<'s>,
        slots: &[u32],
        t: u32,
    ) -> Result<Buf<'s>, LlmError> {
        let d = self.blocks.dims();
        let (hd, kk, ff) = (d.hidden, d.top_k, d.moe_ff);
        let pairs = t * kk;
        let mut up_tab = Vec::with_capacity(slots.len() * 3);
        let mut down_tab = Vec::with_capacity(slots.len() * 3);
        for (pi, &slot) in slots.iter().enumerate() {
            let pi = pi as u32;
            up_tab.extend_from_slice(&[slot, pi / kk, pi]);
            down_tab.extend_from_slice(&[slot, pi, pi]);
        }
        let ut = s.write_uniform(bytemuck::cast_slice(&up_tab))?;
        let dt = s.write_uniform(bytemuck::cast_slice(&down_tab))?;
        let pool = |k: usize| Mat {
            buf: s.import_copy(BufRef::new(
                self.experts.buffer(k),
                self.experts.buffer_len(k),
            )),
            fmt: self.experts.fmt(li, k),
            slot_bytes: self.experts.slot_stride(k),
        };
        let eg = s.alloc(act_bytes(pairs, ff))?;
        let eu = s.alloc(act_bytes(pairs, ff))?;
        for (k, y) in [(GATE, eg), (UP, eu)] {
            p.gemv_gather(
                s,
                pool(k),
                Act::dense(hn, hd),
                Act::dense(y, ff),
                ut,
                pairs,
                ff,
                hd,
            )?;
        }
        let ea = s.alloc(act_bytes(pairs, ff))?;
        p.glu(s, eg, eu, ea, pairs * ff, d.gelu)?;
        let ed = s.alloc(act_bytes(pairs, hd))?;
        p.gemv_gather(
            s,
            pool(DOWN),
            Act::dense(ea, ff),
            Act::dense(ed, hd),
            dt,
            pairs,
            hd,
            ff,
        )?;
        Ok(ed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_chunks_keep_only_reachable_rows() {
        let w = StateBuf::Window { row: 4, window: 10 };
        // Short chunk: everything.
        assert_eq!(w.kept(0, 6), 0..6);
        // Long chunk: the last `window` (a query at >= end reads > end - 10).
        assert_eq!(w.kept(0, 30), 20..30);
        assert_eq!(w.kept(25, 30), 25..30);
        assert_eq!(w.chunk_bytes(0, 30), 40);
        assert_eq!(StateBuf::Rows(4).chunk_bytes(0, 30), 120);
        assert_eq!(StateBuf::Fixed(7).chunk_bytes(3, 9), 7);
        // Ring = window + MAX_STEP, never more than the context.
        assert_eq!(w.capacity(100), 100);
        assert_eq!(w.capacity(100_000), 10 + MAX_STEP);
        assert_eq!(w.bytes(100_000), 4 * (10 + MAX_STEP) as u64);
    }

    #[test]
    fn ring_segments_wrap() {
        // Linear: one segment at the absolute position.
        assert_eq!(row_segments(5..9, 2, 0), [(10, 8)]);
        // Inside the ring.
        assert_eq!(row_segments(12..15, 2, 10), [(4, 6)]);
        // Wrapping: tail of the ring, then its head.
        assert_eq!(row_segments(17..23, 2, 10), [(14, 6), (0, 6)]);
        // Ending exactly at the ring's end.
        assert_eq!(row_segments(7..10, 2, 10), [(14, 6)]);
        assert!(row_segments(3..3, 2, 10).is_empty());
    }
}
