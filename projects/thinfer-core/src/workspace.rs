use crate::backend::{Backend, Binding, BufRef};
use crate::tensor::GpuBufferId;
use crate::trace;
use crate::weight::WeightId;
use std::cell::{RefCell, RefMut};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// Owned physical GPU allocation. `Drop` calls `backend.free`.
///
/// `OwnedBuffer`s live in two places: inside an outstanding `WsBuf` (rented),
/// or inside the pool's free-list (idle). Either way, when they finally drop,
/// the physical buffer is released back to the backend.
pub struct OwnedBuffer<B: Backend> {
    id: GpuBufferId,
    class: u64,
    backend: Arc<B>,
}

impl<B: Backend> OwnedBuffer<B> {
    pub fn id(&self) -> GpuBufferId {
        self.id
    }
    pub fn class(&self) -> u64 {
        self.class
    }
}

impl<B: Backend> Drop for OwnedBuffer<B> {
    fn drop(&mut self) {
        tracing::info!(
            target: trace::WS,
            op = "drop",
            id = self.id.0,
            class = self.class,
        );
        self.backend.free(self.id);
    }
}

struct WorkspaceInner<B: Backend> {
    free: HashMap<u64, Vec<OwnedBuffer<B>>>,
}

/// Size-classed pool of GPU scratch buffers used for activations.
///
/// `alloc()` returns a `WsBuf` guard. The guard's `Drop` pushes the underlying
/// `OwnedBuffer` back into the pool's free-list (LIFO per class), so the next
/// `alloc()` at the same size class hands out the same physical buffer without
/// a `backend.allocate`. When `Workspace` itself drops, the pool drops, and
/// every idle `OwnedBuffer` drops in turn — releasing the physical GPU memory.
///
/// One buffer per rent: wgpu disallows binding the same buffer object as
/// storage-read and storage-read-write in one dispatch (even at disjoint
/// offsets), so we can't sub-slice one mega-buffer for chained activations.
///
/// Size class is `bytes.next_multiple_of(256).max(256)`. Bucket policy is a
/// knob if traces ever show fragmentation.
///
/// `&self` (not `&mut self`) for `alloc`: the pool is `Arc<Mutex<_>>` inside,
/// so concurrent borrows compose. The mutex is uncontended in single-task
/// usage; it's there to make the `WsBuf::Drop` path safe alongside `alloc`.
pub struct Workspace<B: Backend> {
    backend: Arc<B>,
    inner: Arc<Mutex<WorkspaceInner<B>>>,
}

impl<B: Backend> Workspace<B> {
    pub fn new(backend: Arc<B>) -> Self {
        Self {
            backend,
            inner: Arc::new(Mutex::new(WorkspaceInner {
                free: HashMap::new(),
            })),
        }
    }

    fn size_class(bytes: u64) -> u64 {
        bytes.next_multiple_of(256).max(256)
    }

    /// Rent a scratch buffer at least `bytes` long. Returned `WsBuf.len()` is
    /// the request size (not the rounded-up class), so kernel uniform reads
    /// of `arrayLength` stay accurate via the binding `size`.
    ///
    /// The returned guard must outlive any `submit` whose encoder reads or
    /// writes its binding. In straight-line code Rust scope handles this for
    /// free; helper functions that return an encoder while letting locally
    /// allocated `WsBuf`s drop are the pattern to avoid.
    pub fn alloc(&self, bytes: u64) -> Result<WsBuf<B>, B::Error> {
        let class = Self::size_class(bytes);
        let (owned, reused) = {
            let mut inner = self.inner.lock().unwrap();
            match inner.free.get_mut(&class).and_then(|v| v.pop()) {
                Some(owned) => (owned, true),
                None => {
                    let id = self.backend.allocate(class)?;
                    (
                        OwnedBuffer {
                            id,
                            class,
                            backend: Arc::clone(&self.backend),
                        },
                        false,
                    )
                }
            }
        };
        tracing::info!(
            target: trace::WS,
            op = "alloc",
            id = owned.id.0,
            bytes = bytes,
            class = class,
            reused = reused,
        );
        let view = BufRef {
            id: owned.id,
            offset: 0,
            len: bytes,
        };
        Ok(WsBuf {
            view,
            owned: Some(owned),
            pool: Arc::downgrade(&self.inner),
        })
    }

    /// Open a `BatchScope` — owns one command encoder and a guard vec that
    /// holds every `WsBuf` allocated through `scope.alloc` / `write_uniform`
    /// until the scope is consumed by `submit`. Guarantees no pool reuse of
    /// any in-scope buffer during the lifetime of the encoder being built,
    /// which is what makes the bug class in the worklog unconstructible:
    /// uniforms allocated for one dispatch cannot be recycled into another
    /// dispatch in the same submit.
    pub fn batch(&self) -> BatchScope<'_, B> {
        let encoder = self.backend.create_command_encoder();
        BatchScope {
            workspace: self,
            backend: &self.backend,
            encoder: RefCell::new(Some(encoder)),
            guards: RefCell::new(Vec::new()),
            scope_id: NEXT_SCOPE_ID.fetch_add(1, Ordering::Relaxed),
            _life: PhantomData,
        }
    }

    /// Total idle buffers currently sitting in the pool's free-list. Test /
    /// diagnostics only.
    pub fn free_count(&self) -> usize {
        self.inner
            .lock()
            .unwrap()
            .free
            .values()
            .map(|v| v.len())
            .sum()
    }
}

/// RAII rent-guard for a pooled scratch buffer. On `Drop` the underlying
/// `OwnedBuffer` returns to the pool's free-list (or, if the pool is already
/// gone, the `OwnedBuffer` drops directly, freeing the physical buffer).
pub struct WsBuf<B: Backend> {
    /// Cached `BufRef` view of the owned buffer. Stored separately so `Deref`
    /// can return a `&BufRef` without re-constructing it each call.
    view: BufRef,
    owned: Option<OwnedBuffer<B>>,
    pool: Weak<Mutex<WorkspaceInner<B>>>,
}

impl<B: Backend> WsBuf<B> {
    pub fn id(&self) -> GpuBufferId {
        self.view.id
    }
    pub fn len(&self) -> u64 {
        self.view.len
    }
    pub fn is_empty(&self) -> bool {
        self.view.len == 0
    }
    pub fn as_buf_ref(&self) -> BufRef {
        self.view
    }
    pub fn binding(&self, slot: u32) -> Binding {
        self.view.binding(slot)
    }
}

impl<B: Backend> core::ops::Deref for WsBuf<B> {
    type Target = BufRef;
    fn deref(&self) -> &BufRef {
        &self.view
    }
}

impl<B: Backend> Drop for WsBuf<B> {
    fn drop(&mut self) {
        let owned = self.owned.take().expect("WsBuf double-drop");
        tracing::info!(
            target: trace::WS,
            op = "release",
            id = owned.id.0,
            class = owned.class,
        );
        match self.pool.upgrade() {
            Some(pool) => {
                let class = owned.class;
                pool.lock()
                    .unwrap()
                    .free
                    .entry(class)
                    .or_default()
                    .push(owned);
            }
            None => {
                // Pool already dropped — `owned` drops here, freeing the buffer.
                drop(owned);
            }
        }
    }
}

/// Resolve a `WeightId` to a `BufRef` that's already resident on the device.
/// The runtime guarantees this resolves for every weight declared in
/// `ModuleSignature.weights` before `forward` is invoked.
pub trait WeightTable<B: Backend> {
    fn get(&self, id: &WeightId) -> Option<BufRef>;
}

/// Monotonic counter that mints a unique id for every `BatchScope`. Used by
/// the runtime cross-scope-confusion check: every `BatchBuf` carries the
/// scope_id it was minted from, every dispatch asserts buf.scope_id matches
/// the scope it's being dispatched into. Wraps after 2^64 scopes — not a
/// concern in practice.
static NEXT_SCOPE_ID: AtomicU64 = AtomicU64::new(1);

/// A batched encoder + its scratch lifetime, the only way to allocate
/// uniforms/intermediates without risking pool reuse-into-pending-dispatch.
///
/// Construction: `workspace.batch()`. The scope owns one command encoder and
/// a `Vec<WsBuf>` of every allocation made through it. Every `scope.alloc` /
/// `scope.write_uniform` returns a `BatchBuf` handle (Copy) whose underlying
/// `WsBuf` is held in the scope until `submit` consumes the scope. Imported
/// (caller-owned) buffers flow in via `scope.import(&BufRef)` and are NOT
/// tracked in guards — the caller's borrow lifetime keeps them alive.
///
/// Use:
/// ```ignore
/// let scope = workspace.batch();
/// let dims = scope.write_uniform(&dim_bytes)?;
/// let out  = scope.alloc(out_bytes)?;
/// scope.matmul(
///     pipe, &matmul_op,
///     scope.import(&x_in), scope.import(&w_ref), dims, out,
///     n_rows, out_dim,
/// )?;
/// let owned: WsBuf<_> = scope.submit(out).await?;
/// ```
pub struct BatchScope<'wsp, B: Backend> {
    workspace: &'wsp Workspace<B>,
    backend: &'wsp B,
    encoder: RefCell<Option<B::CommandEncoder>>,
    guards: RefCell<Vec<WsBuf<B>>>,
    scope_id: u64,
    /// Marker to make `BatchScope<'wsp, B>` carry its lifetime even when the
    /// `'wsp` shows up only behind references; not load-bearing for safety.
    _life: PhantomData<&'wsp ()>,
}

/// Handle for an allocation made inside a `BatchScope`. Copy; can ONLY be
/// passed back to the scope's `dispatch_*` methods — there is no exposed
/// way to extract a naked `BufRef` from it, which is what prevents the
/// pool-reuse-into-pending-dispatch bug class from being reconstructible
/// in scope-managed code. Carries a `scope_id` so dispatching a handle
/// into the wrong scope is a debug_assert (cross-scope confusion mitigation
/// for the residual case where two scopes share a `'wsp` lifetime).
///
/// The `'s` lifetime is **invariant** (via `PhantomData<*mut &'s ()>`) so the
/// borrow checker can't silently shrink it — that would let a `BatchBuf` outlive
/// the scope that minted it (worklog risk: covariant would re-enable the leak).
/// Compile-fail test in `tests/batchbuf_invariance.rs` proves this.
#[derive(Clone, Copy, Debug)]
pub struct BatchBuf<'s> {
    buf: BufRef,
    scope_id: u64,
    _phantom: PhantomData<*mut &'s ()>,
}

impl<'s> BatchBuf<'s> {
    pub fn len(&self) -> u64 {
        self.buf.len
    }

    pub fn is_empty(&self) -> bool {
        self.buf.len == 0
    }
}

impl<'wsp, B: Backend> BatchScope<'wsp, B> {
    /// Allocate a scratch buffer of `bytes` from the pool. The `WsBuf` is
    /// retained in `guards` for the life of the scope; the returned handle
    /// is the safe way to bind it to dispatches.
    pub fn alloc(&self, bytes: u64) -> Result<BatchBuf<'wsp>, B::Error> {
        let ws = self.workspace.alloc(bytes)?;
        let buf = ws.as_buf_ref();
        self.guards.borrow_mut().push(ws);
        Ok(BatchBuf {
            buf,
            scope_id: self.scope_id,
            _phantom: PhantomData,
        })
    }

    /// Allocate a uniform-sized scratch buffer and upload `bytes` to it via
    /// `backend.write_buffer`. Common case for kernel uniform parameters.
    pub fn write_uniform(&self, bytes: &[u8]) -> Result<BatchBuf<'wsp>, B::Error> {
        let h = self.alloc(bytes.len() as u64)?;
        self.backend.write_buffer(h.buf.id, 0, bytes)?;
        Ok(h)
    }

    /// Convenience: `[u32; N]` uniform.
    pub fn u32x4_uniform(
        &self,
        a: u32,
        b: u32,
        c: u32,
        d: u32,
    ) -> Result<BatchBuf<'wsp>, B::Error> {
        let mut bytes = [0u8; 16];
        bytes[0..4].copy_from_slice(&a.to_le_bytes());
        bytes[4..8].copy_from_slice(&b.to_le_bytes());
        bytes[8..12].copy_from_slice(&c.to_le_bytes());
        bytes[12..16].copy_from_slice(&d.to_le_bytes());
        self.write_uniform(&bytes)
    }

    /// Bring a caller-owned `BufRef` into the scope without taking ownership.
    /// The borrow must outlive the scope's `'wsp` lifetime — a weak hint
    /// (caller could still drop the source `WsBuf` mid-submit; that's a
    /// runtime invariant we cannot encode today). TODO: redesign `BufRef`
    /// itself to carry a borrow on its `OwnedBuffer` so this constraint is
    /// real, not just a marker.
    pub fn import(&self, buf: &'wsp BufRef) -> BatchBuf<'wsp> {
        BatchBuf {
            buf: *buf,
            scope_id: self.scope_id,
            _phantom: PhantomData,
        }
    }

    /// Internal: resolve a handle back to `BufRef`. Crate-private to keep
    /// naked `BufRef` (Copy) from escaping scope-managed code paths. Asserts
    /// the handle was minted from this scope (cross-scope confusion guard).
    pub(crate) fn resolve(&self, h: BatchBuf<'_>) -> BufRef {
        debug_assert_eq!(
            h.scope_id, self.scope_id,
            "BatchBuf came from a different scope (scope_id mismatch)"
        );
        h.buf
    }

    pub(crate) fn encoder_mut(&self) -> RefMut<'_, B::CommandEncoder> {
        RefMut::map(self.encoder.borrow_mut(), |o| {
            o.as_mut().expect("BatchScope encoder already taken")
        })
    }

    /// Consume the scope: submit the encoder, await GPU completion, then
    /// release scratch guards back to the pool. The `out` handle's backing
    /// `WsBuf` is moved out and returned to the caller — it survives the
    /// scope. (Imported buffers are unaffected; they're caller-owned.)
    ///
    /// Async-cancellation hazard (worklog risk (B)): if this future is
    /// dropped between `backend.submit(enc)` queuing and the `await`
    /// resolving, guards drop while GPU work is still pending → next batch
    /// reuses those buffer ids → write_buffer overwrites them before the GPU
    /// reads them. Mitigation requires `submit_with_guards`: wgpu
    /// `Queue::on_submitted_work_done` callback owns the `Vec<WsBuf>` so
    /// guards live until GPU completion regardless of future state. v1
    /// callers don't cancel submit futures; tracked as a follow-up.
    pub async fn submit(self, out: BatchBuf<'wsp>) -> Result<WsBuf<B>, B::Error> {
        debug_assert_eq!(
            out.scope_id, self.scope_id,
            "submit output BatchBuf came from a different scope"
        );
        let enc = self
            .encoder
            .into_inner()
            .expect("BatchScope encoder already taken (double submit?)");
        let mut guards = self.guards.into_inner();
        // `out` must be one of our owned allocations, not an import. Find it
        // by id (id is unique within a scope — pool reuse is suppressed by
        // the guards vec itself).
        let idx = guards
            .iter()
            .position(|g| g.id() == out.buf.id)
            .expect("submit output BatchBuf was not allocated by this scope");
        let out_ws = guards.swap_remove(idx);
        self.backend.submit(enc).await?;
        Ok(out_ws)
    }

    /// Like `submit` but moves out N owned outputs in one go. Order of the
    /// returned vec matches `outs`. Imports are not allowed (caller already
    /// owns them).
    pub async fn submit_many(self, outs: &[BatchBuf<'wsp>]) -> Result<Vec<WsBuf<B>>, B::Error> {
        for h in outs {
            debug_assert_eq!(h.scope_id, self.scope_id);
        }
        let enc = self
            .encoder
            .into_inner()
            .expect("BatchScope encoder already taken");
        let mut guards = self.guards.into_inner();
        let mut result = Vec::with_capacity(outs.len());
        for h in outs {
            let idx = guards
                .iter()
                .position(|g| g.id() == h.buf.id)
                .expect("submit_many output BatchBuf was not allocated by this scope");
            result.push(guards.swap_remove(idx));
        }
        self.backend.submit(enc).await?;
        Ok(result)
    }

    /// Submit without preserving any output. All guards drop after GPU
    /// completion. Use when the scope's effect is a side-effect on an
    /// imported (caller-owned) destination buffer.
    pub async fn submit_void(self) -> Result<(), B::Error> {
        let enc = self
            .encoder
            .into_inner()
            .expect("BatchScope encoder already taken");
        self.backend.submit(enc).await?;
        Ok(())
    }
}

// Per-op dispatch methods on BatchScope. Thin forwarders to the free
// `dispatch_*` functions in `crate::ops`. The scope methods are the only
// way for model code to drive a dispatch — they assert scope_id match on
// every BatchBuf, build the `Bufs` struct from resolved BufRefs, and
// borrow the encoder for the call. No naked BufRef ever escapes scope-
// managed code; the dispatch helpers retain their original signature for
// raw-buffer use (unit tests).
//
// Convention: arg order on each scope method mirrors the matching `Bufs`
// struct field order, with workgroup-shape u32s trailing.
impl<'wsp, B: Backend> BatchScope<'wsp, B> {
    #[allow(clippy::too_many_arguments)]
    pub fn matmul<O: crate::ops::MatmulOp>(
        &self,
        pipeline: &B::Pipeline,
        op: &O,
        a: BatchBuf<'wsp>,
        b: BatchBuf<'wsp>,
        dims: BatchBuf<'wsp>,
        out: BatchBuf<'wsp>,
        m: u32,
        n: u32,
    ) -> Result<(), B::Error> {
        let a = self.resolve(a);
        let b = self.resolve(b);
        let dims = self.resolve(dims);
        let out = self.resolve(out);
        let bufs = crate::ops::MatmulBufs {
            a: &a,
            b: &b,
            dims: &dims,
            out: &out,
        };
        crate::ops::dispatch_matmul::<O, _>(
            self.backend,
            &mut self.encoder_mut(),
            pipeline,
            op,
            &bufs,
            m,
            n,
        )
    }

    pub fn rmsnorm<O: crate::ops::RmsNormOp>(
        &self,
        pipeline: &B::Pipeline,
        x: BatchBuf<'wsp>,
        w: BatchBuf<'wsp>,
        uniform: BatchBuf<'wsp>,
        out: BatchBuf<'wsp>,
        n_rows: u32,
    ) -> Result<(), B::Error> {
        let x = self.resolve(x);
        let w = self.resolve(w);
        let uniform = self.resolve(uniform);
        let out = self.resolve(out);
        let bufs = crate::ops::RmsNormBufs {
            x: &x,
            w: &w,
            uniform: &uniform,
            out: &out,
        };
        crate::ops::dispatch_rmsnorm::<O, _>(
            self.backend,
            &mut self.encoder_mut(),
            pipeline,
            &bufs,
            n_rows,
        )
    }

    pub fn layernorm<O: crate::ops::LayerNormOp>(
        &self,
        pipeline: &B::Pipeline,
        x: BatchBuf<'wsp>,
        uniform: BatchBuf<'wsp>,
        out: BatchBuf<'wsp>,
        n_rows: u32,
    ) -> Result<(), B::Error> {
        let x = self.resolve(x);
        let uniform = self.resolve(uniform);
        let out = self.resolve(out);
        let bufs = crate::ops::LayerNormBufs {
            x: &x,
            uniform: &uniform,
            out: &out,
        };
        crate::ops::dispatch_layernorm::<O, _>(
            self.backend,
            &mut self.encoder_mut(),
            pipeline,
            &bufs,
            n_rows,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn group_norm<O: crate::ops::GroupNormOp>(
        &self,
        pipeline: &B::Pipeline,
        x: BatchBuf<'wsp>,
        w: BatchBuf<'wsp>,
        bias: BatchBuf<'wsp>,
        uniform: BatchBuf<'wsp>,
        out: BatchBuf<'wsp>,
        n_rows: u32,
    ) -> Result<(), B::Error> {
        let x = self.resolve(x);
        let w = self.resolve(w);
        let bias = self.resolve(bias);
        let uniform = self.resolve(uniform);
        let out = self.resolve(out);
        let bufs = crate::ops::GroupNormBufs {
            x: &x,
            w: &w,
            bias: &bias,
            uniform: &uniform,
            out: &out,
        };
        crate::ops::dispatch_group_norm::<O, _>(
            self.backend,
            &mut self.encoder_mut(),
            pipeline,
            &bufs,
            n_rows,
        )
    }

    pub fn bcast_add<O: crate::ops::BcastAddOp>(
        &self,
        pipeline: &B::Pipeline,
        x: BatchBuf<'wsp>,
        s: BatchBuf<'wsp>,
        uniform: BatchBuf<'wsp>,
        out: BatchBuf<'wsp>,
        n_elems: u32,
    ) -> Result<(), B::Error> {
        let x = self.resolve(x);
        let s = self.resolve(s);
        let uniform = self.resolve(uniform);
        let out = self.resolve(out);
        let bufs = crate::ops::BcastAddBufs {
            x: &x,
            s: &s,
            uniform: &uniform,
            out: &out,
        };
        crate::ops::dispatch_bcast_add::<O, _>(
            self.backend,
            &mut self.encoder_mut(),
            pipeline,
            &bufs,
            n_elems,
        )
    }

    pub fn bcast_affine<O: crate::ops::BcastAffineOp>(
        &self,
        pipeline: &B::Pipeline,
        x: BatchBuf<'wsp>,
        s: BatchBuf<'wsp>,
        uniform: BatchBuf<'wsp>,
        out: BatchBuf<'wsp>,
        n_elems: u32,
    ) -> Result<(), B::Error> {
        let x = self.resolve(x);
        let s = self.resolve(s);
        let uniform = self.resolve(uniform);
        let out = self.resolve(out);
        let bufs = crate::ops::BcastAffineBufs {
            x: &x,
            s: &s,
            uniform: &uniform,
            out: &out,
        };
        crate::ops::dispatch_bcast_affine::<O, _>(
            self.backend,
            &mut self.encoder_mut(),
            pipeline,
            &bufs,
            n_elems,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bcast_fma<O: crate::ops::BcastFmaOp>(
        &self,
        pipeline: &B::Pipeline,
        x: BatchBuf<'wsp>,
        s: BatchBuf<'wsp>,
        y: BatchBuf<'wsp>,
        uniform: BatchBuf<'wsp>,
        out: BatchBuf<'wsp>,
        n_elems: u32,
    ) -> Result<(), B::Error> {
        let x = self.resolve(x);
        let s = self.resolve(s);
        let y = self.resolve(y);
        let uniform = self.resolve(uniform);
        let out = self.resolve(out);
        let bufs = crate::ops::BcastFmaBufs {
            x: &x,
            s: &s,
            y: &y,
            uniform: &uniform,
            out: &out,
        };
        crate::ops::dispatch_bcast_fma::<O, _>(
            self.backend,
            &mut self.encoder_mut(),
            pipeline,
            &bufs,
            n_elems,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope<O: crate::ops::RopeOp>(
        &self,
        pipeline: &B::Pipeline,
        x: BatchBuf<'wsp>,
        freqs: BatchBuf<'wsp>,
        uniform: BatchBuf<'wsp>,
        out: BatchBuf<'wsp>,
        rows: u32,
        heads: u32,
        pairs: u32,
    ) -> Result<(), B::Error> {
        let x = self.resolve(x);
        let freqs = self.resolve(freqs);
        let uniform = self.resolve(uniform);
        let out = self.resolve(out);
        let bufs = crate::ops::RopeBufs {
            x: &x,
            freqs: &freqs,
            uniform: &uniform,
            out: &out,
        };
        crate::ops::dispatch_rope::<O, _>(
            self.backend,
            &mut self.encoder_mut(),
            pipeline,
            &bufs,
            rows,
            heads,
            pairs,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sdpa<O: crate::ops::SdpaOp>(
        &self,
        pipeline: &B::Pipeline,
        q: BatchBuf<'wsp>,
        k: BatchBuf<'wsp>,
        v: BatchBuf<'wsp>,
        mask: BatchBuf<'wsp>,
        uniform: BatchBuf<'wsp>,
        out: BatchBuf<'wsp>,
        b: u32,
        s_q: u32,
        h_q: u32,
    ) -> Result<(), B::Error> {
        let q = self.resolve(q);
        let k = self.resolve(k);
        let v = self.resolve(v);
        let mask = self.resolve(mask);
        let uniform = self.resolve(uniform);
        let out = self.resolve(out);
        let bufs = crate::ops::SdpaBufs {
            q: &q,
            k: &k,
            v: &v,
            mask: &mask,
            uniform: &uniform,
            out: &out,
        };
        crate::ops::dispatch_sdpa::<O, _>(
            self.backend,
            &mut self.encoder_mut(),
            pipeline,
            &bufs,
            b,
            s_q,
            h_q,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv2d<O: crate::ops::Conv2dOp>(
        &self,
        pipeline: &B::Pipeline,
        x: BatchBuf<'wsp>,
        w: BatchBuf<'wsp>,
        bias: BatchBuf<'wsp>,
        uniform: BatchBuf<'wsp>,
        out: BatchBuf<'wsp>,
        n_out_elems: u32,
    ) -> Result<(), B::Error> {
        let x = self.resolve(x);
        let w = self.resolve(w);
        let bias = self.resolve(bias);
        let uniform = self.resolve(uniform);
        let out = self.resolve(out);
        let bufs = crate::ops::Conv2dBufs {
            x: &x,
            w: &w,
            bias: &bias,
            uniform: &uniform,
            out: &out,
        };
        crate::ops::dispatch_conv2d::<O, _>(
            self.backend,
            &mut self.encoder_mut(),
            pipeline,
            &bufs,
            n_out_elems,
        )
    }

    pub fn transpose12<O: crate::ops::Transpose12Op>(
        &self,
        pipeline: &B::Pipeline,
        input: BatchBuf<'wsp>,
        uniform: BatchBuf<'wsp>,
        out: BatchBuf<'wsp>,
        total: u32,
    ) -> Result<(), B::Error> {
        let input = self.resolve(input);
        let uniform = self.resolve(uniform);
        let out = self.resolve(out);
        let bufs = crate::ops::Transpose12Bufs {
            input: &input,
            uniform: &uniform,
            out: &out,
        };
        crate::ops::dispatch_transpose12::<O, _>(
            self.backend,
            &mut self.encoder_mut(),
            pipeline,
            &bufs,
            total,
        )
    }

    pub fn scatter_pad_rows<O: crate::ops::ScatterPadRowsOp>(
        &self,
        pipeline: &B::Pipeline,
        pad: BatchBuf<'wsp>,
        mask: BatchBuf<'wsp>,
        uniform: BatchBuf<'wsp>,
        dst: BatchBuf<'wsp>,
        n_elems: u32,
    ) -> Result<(), B::Error> {
        let pad = self.resolve(pad);
        let mask = self.resolve(mask);
        let uniform = self.resolve(uniform);
        let dst = self.resolve(dst);
        let bufs = crate::ops::ScatterPadRowsBufs {
            pad: &pad,
            mask: &mask,
            uniform: &uniform,
            dst: &dst,
        };
        crate::ops::dispatch_scatter_pad_rows::<O, _>(
            self.backend,
            &mut self.encoder_mut(),
            pipeline,
            &bufs,
            n_elems,
        )
    }

    pub fn upsample2d_nearest<O: crate::ops::Upsample2dNearestOp>(
        &self,
        pipeline: &B::Pipeline,
        x: BatchBuf<'wsp>,
        uniform: BatchBuf<'wsp>,
        out: BatchBuf<'wsp>,
        n_out_elems: u32,
    ) -> Result<(), B::Error> {
        let x = self.resolve(x);
        let uniform = self.resolve(uniform);
        let out = self.resolve(out);
        let bufs = crate::ops::Upsample2dNearestBufs {
            x: &x,
            uniform: &uniform,
            out: &out,
        };
        crate::ops::dispatch_upsample2d_nearest::<O, _>(
            self.backend,
            &mut self.encoder_mut(),
            pipeline,
            &bufs,
            n_out_elems,
        )
    }

    /// Generic elementwise op dispatch (AddF32, MulF32, SiluF32, TanhF32).
    pub fn dispatch_op<O: crate::ops::Op>(
        &self,
        pipeline: &B::Pipeline,
        inputs: &[BatchBuf<'wsp>],
        output: BatchBuf<'wsp>,
    ) -> Result<(), B::Error> {
        for h in inputs {
            debug_assert_eq!(h.scope_id, self.scope_id);
        }
        let input_refs: Vec<BufRef> = inputs.iter().map(|h| h.buf).collect();
        let output_ref = self.resolve(output);
        crate::ops::dispatch_op::<O, _>(
            self.backend,
            &mut self.encoder_mut(),
            pipeline,
            &input_refs,
            output_ref,
        )
    }

    /// Encode a buffer-to-buffer copy inside this scope's encoder.
    pub fn copy_buffer_to_buffer(
        &self,
        src: BatchBuf<'wsp>,
        src_offset: u64,
        dst: BatchBuf<'wsp>,
        dst_offset: u64,
        len: u64,
    ) -> Result<(), B::Error> {
        let src = self.resolve(src);
        let dst = self.resolve(dst);
        self.backend.copy_buffer_to_buffer(
            &mut self.encoder_mut(),
            src.id,
            src_offset + src.offset,
            dst.id,
            dst_offset + dst.offset,
            len,
        )
    }
}

/// Diag-only specialization: forward `WgpuBackend::read_buffer_via_encoder`
/// through the scope's encoder so VAE diag helpers can stage readbacks inside
/// the same submit as compute (avoids the multi-submit driver wedge).
impl<'wsp> BatchScope<'wsp, crate::backend::WgpuBackend> {
    pub fn read_buffer_via_encoder(
        &self,
        buf: BatchBuf<'wsp>,
        offset: u64,
        len: u64,
    ) -> Result<
        impl std::future::Future<Output = Result<Vec<u8>, crate::backend::WgpuError>> + use<>,
        crate::backend::WgpuError,
    > {
        let r = self.resolve(buf);
        self.backend
            .read_buffer_via_encoder(&mut self.encoder_mut(), r.id, offset + r.offset, len)
    }
}

#[cfg(test)]
#[allow(clippy::arc_with_non_send_sync)]
mod tests {
    use super::*;

    struct MockBackend {
        next: std::cell::Cell<u64>,
        allocated: std::cell::RefCell<std::collections::HashSet<GpuBufferId>>,
    }

    impl MockBackend {
        fn new() -> Self {
            Self {
                next: std::cell::Cell::new(1),
                allocated: Default::default(),
            }
        }
        fn live(&self) -> usize {
            self.allocated.borrow().len()
        }
    }

    impl Backend for MockBackend {
        type Error = ();
        type CommandEncoder = ();
        type Pipeline = ();
        fn allocate(&self, _bytes: u64) -> Result<GpuBufferId, ()> {
            let id = GpuBufferId(self.next.get());
            self.next.set(self.next.get() + 1);
            self.allocated.borrow_mut().insert(id);
            Ok(id)
        }
        fn free(&self, id: GpuBufferId) {
            self.allocated.borrow_mut().remove(&id);
        }
        fn write_buffer(&self, _: GpuBufferId, _: u64, _: &[u8]) -> Result<(), ()> {
            Ok(())
        }
        fn create_command_encoder(&self) {}
        fn dispatch(
            &self,
            _: &mut (),
            _: &(),
            _: &[crate::backend::Binding],
            _: [u32; 3],
        ) -> Result<(), ()> {
            Ok(())
        }
        fn copy_buffer_to_buffer(
            &self,
            _: &mut (),
            _: GpuBufferId,
            _: u64,
            _: GpuBufferId,
            _: u64,
            _: u64,
        ) -> Result<(), ()> {
            Ok(())
        }
        async fn submit(&self, _: ()) -> Result<(), ()> {
            Ok(())
        }
        async fn create_pipeline(
            &self,
            _: &str,
            _: &str,
            _: &[crate::backend::BindingLayout],
        ) -> Result<(), ()> {
            Ok(())
        }
        async fn read_buffer(&self, _: GpuBufferId, _: u64, _: u64) -> Result<Vec<u8>, ()> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn each_concurrent_alloc_distinct_buffer() {
        let b = Arc::new(MockBackend::new());
        let ws = Workspace::new(b.clone());
        let r1 = ws.alloc(100).unwrap();
        let r2 = ws.alloc(100).unwrap();
        assert_ne!(r1.id(), r2.id());
        assert_eq!(r1.len(), 100);
        assert_eq!(r1.as_buf_ref().offset, 0);
    }

    #[test]
    fn drop_recycles_same_class() {
        let b = Arc::new(MockBackend::new());
        let ws = Workspace::new(b.clone());
        let (id1, id2) = {
            let r1 = ws.alloc(100).unwrap();
            let r2 = ws.alloc(100).unwrap();
            (r1.id(), r2.id())
        };
        // Both guards dropped; the two physical buffers stayed alive in the pool.
        assert_eq!(b.live(), 2);
        let r3 = ws.alloc(100).unwrap();
        let r4 = ws.alloc(100).unwrap();
        assert!(r3.id() == id1 || r3.id() == id2);
        assert!(r4.id() == id1 || r4.id() == id2);
        assert_eq!(b.live(), 2); // no new allocate
    }

    #[test]
    fn distinct_classes_dont_share() {
        let b = Arc::new(MockBackend::new());
        let ws = Workspace::new(b.clone());
        {
            let _r1 = ws.alloc(100).unwrap(); // class 256
            let _r2 = ws.alloc(4096).unwrap(); // class 4096
        }
        let r3 = ws.alloc(80).unwrap(); // class 256 -> reuse
        let _r4 = ws.alloc(4096).unwrap(); // class 4096 -> reuse
        assert_eq!(b.live(), 2);
        assert_eq!(r3.len(), 80);
        let _r5 = ws.alloc(300).unwrap(); // class 512 -> fresh
        assert_eq!(b.live(), 3);
    }

    #[test]
    fn workspace_drop_frees_pool() {
        let b = Arc::new(MockBackend::new());
        {
            let ws = Workspace::new(b.clone());
            let r1 = ws.alloc(100).unwrap();
            let r2 = ws.alloc(100).unwrap();
            assert_eq!(b.live(), 2);
            drop(r1);
            drop(r2);
            // Guards drop, OwnedBuffers go into the pool's free-list.
            assert_eq!(b.live(), 2);
            assert_eq!(ws.free_count(), 2);
        }
        // Workspace dropped -> Inner dropped -> OwnedBuffers in free-list drop -> backend.free
        assert_eq!(b.live(), 0);
    }

    #[test]
    fn guard_outliving_workspace_still_frees() {
        let b = Arc::new(MockBackend::new());
        let guard;
        {
            let ws = Workspace::new(b.clone());
            guard = ws.alloc(100).unwrap();
            assert_eq!(b.live(), 1);
        }
        // Workspace gone; guard still holds the OwnedBuffer.
        assert_eq!(b.live(), 1);
        drop(guard);
        // Weak-upgrade fails; OwnedBuffer drops directly, freeing.
        assert_eq!(b.live(), 0);
    }
}
