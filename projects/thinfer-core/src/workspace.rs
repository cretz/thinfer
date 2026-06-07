use crate::arbiter::{MemArbiter, MemReclaimer, RECLAIM_IDLE_POOL};
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
        tracing::trace!(
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
    /// Sum of size classes currently held in `free`. Maintained by the
    /// `alloc`, `WsBuf::Drop`, `spill`, and `drain_pool` paths. Read by
    /// the dispatch layer to exclude reusable pool bytes from the "live
    /// workspace" term in the packer budget.
    pool_bytes: u64,
}

/// Pop free-list entries up to `at_least` bytes; entries drop, releasing the
/// underlying GPU buffer (and the MemAccount charge) via `OwnedBuffer::Drop`.
/// Smallest size class first: preserves the largest reusable buffers, since
/// re-allocating those is the expensive case.
fn spill_pool_locked<B: Backend>(inner: &mut WorkspaceInner<B>, at_least: u64) -> u64 {
    if at_least == 0 {
        return 0;
    }
    let mut classes: Vec<u64> = inner.free.keys().copied().collect();
    classes.sort_unstable();
    let mut freed: u64 = 0;
    for class in classes {
        let bufs = inner.free.get_mut(&class).expect("class came from keys()");
        while freed < at_least && bufs.pop().is_some() {
            freed = freed.saturating_add(class);
        }
        if bufs.is_empty() {
            inner.free.remove(&class);
        }
        if freed >= at_least {
            break;
        }
    }
    inner.pool_bytes = inner.pool_bytes.saturating_sub(freed);
    if freed > 0 {
        tracing::debug!(
            target: trace::WS,
            op = "spill",
            at_least = at_least,
            freed = freed,
        );
    }
    freed
}

/// Idle-pool reclaimer for the [`MemArbiter`] chain. Registered by
/// [`Workspace::new`] at `RECLAIM_IDLE_POOL`. Holds a `Weak` to the pool's
/// inner state (not to `Workspace` itself), so a dropped per-generate
/// workspace leaves a dead entry the arbiter prunes on the next `register`.
struct PoolReclaimer<B: Backend> {
    inner: Weak<Mutex<WorkspaceInner<B>>>,
}

impl<B: Backend + Send + Sync> MemReclaimer for PoolReclaimer<B> {
    fn label(&self) -> &'static str {
        "workspace-pool"
    }

    fn alive(&self) -> bool {
        self.inner.strong_count() > 0
    }

    fn reclaim(&self, at_least: u64) -> u64 {
        match self.inner.upgrade() {
            Some(inner) => spill_pool_locked(&mut inner.lock().unwrap(), at_least),
            None => 0,
        }
    }
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
    /// Shared VRAM budget owner. On `alloc` miss the arbiter reclaims (this
    /// pool's idle entries, other clients' caches, evictable weights) until
    /// the pending class fits under the budget, before `backend.allocate`
    /// runs. [`MemArbiter::unlimited`] disables the gate (unit tests that
    /// don't care about VRAM accounting).
    arbiter: Arc<MemArbiter>,
}

impl<B: Backend> Workspace<B> {
    pub fn new(backend: Arc<B>, arbiter: Arc<MemArbiter>) -> Self
    where
        B: Send + Sync,
    {
        let inner = Arc::new(Mutex::new(WorkspaceInner {
            free: HashMap::new(),
            pool_bytes: 0,
        }));
        arbiter.register(
            RECLAIM_IDLE_POOL,
            Box::new(PoolReclaimer {
                inner: Arc::downgrade(&inner),
            }),
        );
        Self {
            backend,
            inner,
            arbiter,
        }
    }

    /// Current sum of idle pool bytes. Cheap. Used by dispatch-layer
    /// callers (e.g. dit.rs packer-budget) to subtract pool from the
    /// "in-flight workspace" reading on `MemAccount` — pool buffers are
    /// reusable, so they shouldn't count against the next phase's budget.
    pub fn pool_bytes(&self) -> u64 {
        self.inner.lock().unwrap().pool_bytes
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
        // Pop outside the arbiter call: the reclaim chain re-enters this
        // pool via `PoolReclaimer`, so the inner lock must not be held when
        // `ensure_headroom` runs (lock order is arbiter -> client).
        let popped = {
            let mut inner = self.inner.lock().unwrap();
            let popped = inner.free.get_mut(&class).and_then(|v| v.pop());
            if popped.is_some() {
                inner.pool_bytes = inner.pool_bytes.saturating_sub(class);
            }
            popped
        };
        let (owned, reused) = match popped {
            Some(owned) => (owned, true),
            None => {
                // Pool miss → about to grow VRAM. The arbiter reclaims
                // (idle pool entries, then evictable weights) until the new
                // class fits under the budget. If even the full chain can't
                // make room, `backend.allocate` overshoots and the test
                // budget assert surfaces the true working-set overrun rather
                // than us silently failing.
                self.arbiter
                    .ensure_headroom(self.backend.mem_account(), class);
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
        };
        tracing::trace!(
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

    /// Drop every idle buffer in the pool, freeing the underlying GPU
    /// allocations back to the backend. Phase-boundary use: call between
    /// text_encode -> DiT -> VAE so size classes from the prior phase don't
    /// sit live in VRAM while the next phase allocates its own working set.
    /// In-flight `WsBuf` rentals are unaffected; only buffers currently in
    /// the free-list drop.
    pub fn drain_pool(&self) {
        let mut inner = self.inner.lock().unwrap();
        // `inner.free` is consumed by drain(); each Vec<OwnedBuffer> drops at
        // the iterator step, then each OwnedBuffer::Drop calls backend.free.
        let total: usize = inner.free.drain().map(|(_, v)| v.len()).sum();
        inner.pool_bytes = 0;
        tracing::debug!(
            target: trace::WS,
            op = "drain_pool",
            freed = total as u64,
        );
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
        tracing::trace!(
            target: trace::WS,
            op = "release",
            id = owned.id.0,
            class = owned.class,
        );
        // Pool alive → return to free-list. Pool gone → `owned` falls out
        // of scope here and OwnedBuffer::Drop calls backend.free.
        if let Some(pool) = self.pool.upgrade() {
            let class = owned.class;
            let mut inner = pool.lock().unwrap();
            inner.pool_bytes = inner.pool_bytes.saturating_add(class);
            inner.free.entry(class).or_default().push(owned);
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

    /// Allocate one scratch buffer of `a_bytes + b_bytes` and return three
    /// views over it: the full fused span, the first `a_bytes` (offset 0),
    /// and the trailing `b_bytes` (offset `a_bytes`). The fused span is
    /// what a kernel binds as one `var<storage>` reading both halves; the
    /// `a` and `b` views are what producers bind when they write each half
    /// independently. All three share one `WsBuf` retained in guards.
    pub fn alloc_pair(
        &self,
        a_bytes: u64,
        b_bytes: u64,
    ) -> Result<(BatchBuf<'wsp>, BatchBuf<'wsp>, BatchBuf<'wsp>), B::Error> {
        let ws = self.workspace.alloc(a_bytes + b_bytes)?;
        let id = ws.id();
        self.guards.borrow_mut().push(ws);
        let fused = BatchBuf {
            buf: BufRef::view(id, 0, a_bytes + b_bytes),
            scope_id: self.scope_id,
            _phantom: PhantomData,
        };
        let a = BatchBuf {
            buf: BufRef::view(id, 0, a_bytes),
            scope_id: self.scope_id,
            _phantom: PhantomData,
        };
        let b = BatchBuf {
            buf: BufRef::view(id, a_bytes, b_bytes),
            scope_id: self.scope_id,
            _phantom: PhantomData,
        };
        Ok((fused, a, b))
    }

    /// Recompose a fused storage view from two BatchBufs that were minted as
    /// adjacent halves of one `alloc_pair` allocation. Asserts shared id and
    /// adjacency. Used to rebuild an `ActBuf::fused` after pack/unpack across
    /// a phase boundary (carry only preserves the data + scale views).
    pub fn fuse_pair(&self, a: BatchBuf<'wsp>, b: BatchBuf<'wsp>) -> BatchBuf<'wsp> {
        debug_assert_eq!(
            a.buf.id, b.buf.id,
            "fuse_pair: views must share a GpuBufferId"
        );
        debug_assert_eq!(
            a.buf.offset + a.buf.len,
            b.buf.offset,
            "fuse_pair: views must be adjacent (a end == b start)"
        );
        BatchBuf {
            buf: BufRef::view(a.buf.id, a.buf.offset, a.buf.len + b.buf.len),
            scope_id: self.scope_id,
            _phantom: PhantomData,
        }
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

    /// Like [`Self::import`] but takes a Copy `BufRef` directly — no
    /// `&'wsp` tether. The caller must keep the backing `WsBuf` alive until
    /// the scope's submit completes; the lifetime can't express that
    /// statically. Used by [`ScopePacker`], which owns carry `WsBuf`s in its
    /// own hold-bag across scope boundaries so no `&'wsp BufRef` is in scope.
    pub fn import_copy(&self, buf: BufRef) -> BatchBuf<'wsp> {
        BatchBuf {
            buf,
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

    /// Deferred variant of [`Self::submit_many`]: queue the encoder, hand the
    /// requested `outs` back as owned `WsBuf`s, and return a completion future
    /// that holds the remaining guards alive until GPU completion. Lets a
    /// caller chain multiple scopes within one logical batch while carrying
    /// specific outputs across the cut — the carry `WsBuf`s outlive the scope
    /// they were allocated in and can be `import`ed into a follow-on scope.
    ///
    /// Same hazard as `submit_deferred`: imported (caller-owned) buffers that
    /// dispatches in this scope reference must outlive the returned future.
    pub fn submit_many_deferred(
        self,
        outs: &[BatchBuf<'wsp>],
    ) -> (
        Vec<WsBuf<B>>,
        impl core::future::Future<Output = Result<(), B::Error>> + use<'wsp, B>,
    ) {
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
                .expect("submit_many_deferred output BatchBuf was not allocated by this scope");
            result.push(guards.swap_remove(idx));
        }
        let fut = self.backend.submit(enc);
        let completion = async move {
            let _g = guards;
            fut.await
        };
        (result, completion)
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

    /// Like `submit_void`, but returns the completion future synchronously
    /// without awaiting it. The caller drives the await on its own schedule
    /// (typically: keep a depth-K ring of in-flight submits to overlap CPU
    /// encoding of block N+1 with GPU execution of block N).
    ///
    /// Critical contract: `backend.submit` already calls `queue.submit`
    /// eagerly in its synchronous prelude, so the GPU work is in-flight as
    /// soon as this function returns. The returned future awaits the
    /// completion fence + error-scope drain. The scope's guards are moved
    /// into the future's frame so pool reuse is suppressed until completion.
    ///
    /// Hazard the caller must respect: any caller-owned `WsBuf` that the
    /// dispatches in this scope still reference (i.e. imported via
    /// `scope.import`) must also be held alive until the returned future
    /// resolves. The scope only tracks buffers it allocated itself.
    pub fn submit_deferred(self) -> impl core::future::Future<Output = Result<(), B::Error>> {
        let enc = self
            .encoder
            .into_inner()
            .expect("BatchScope encoder already taken");
        let guards = self.guards.into_inner();
        // backend.submit's synchronous prelude calls queue.submit; the
        // returned future just awaits completion + error scopes.
        let fut = self.backend.submit(enc);
        async move {
            let _g = guards;
            fut.await
        }
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

    /// Dispatch a quant-weight dequant pass: read Q4_K/Q5_K/Q6_K/Q8_0/Q4_0
    /// blocks from `b_quant` and write a dense bf16-packed `[N, K]` tensor
    /// into `b_dense`. Caller-provided `dims` uniform is `(n, k, _, _)`.
    /// Used by the DiT layer forward to materialize weights once per matmul
    /// site, eliminating the 4x in-matmul re-dequant of the same B columns.
    pub fn dequant(
        &self,
        pipeline: &B::Pipeline,
        scheme: crate::quant::QuantKind,
        b_quant: BatchBuf<'wsp>,
        b_dense: BatchBuf<'wsp>,
        dims: BatchBuf<'wsp>,
        n: u32,
        k: u32,
    ) -> Result<(), B::Error> {
        let b_quant = self.resolve(b_quant);
        let b_dense = self.resolve(b_dense);
        let dims = self.resolve(dims);
        let bufs = crate::ops::DequantBufs {
            b_quant: &b_quant,
            b_dense: &b_dense,
            dims: &dims,
        };
        crate::ops::dispatch_dequant(
            self.backend,
            &mut self.encoder_mut(),
            pipeline,
            scheme,
            &bufs,
            n,
            k,
        )
    }

    /// Dispatch the activation int8 quantizer: read `[M, K]` f16 acts and
    /// write `[M, K]` packed int8 + `[M, K/32]` f32 scales. Caller-provided
    /// `dims` uniform is `(m, k, _, _)`. Pairs with [`Self::matmul_i8`].
    pub fn act_quant(
        &self,
        pipeline: &B::Pipeline,
        a: BatchBuf<'wsp>,
        out_i8: BatchBuf<'wsp>,
        out_params: BatchBuf<'wsp>,
        dims: BatchBuf<'wsp>,
        m: u32,
        k: u32,
    ) -> Result<(), B::Error> {
        let a = self.resolve(a);
        let out_i8 = self.resolve(out_i8);
        let out_params = self.resolve(out_params);
        let dims = self.resolve(dims);
        let bufs = crate::ops::act_quant::ActQuantBufs {
            a: &a,
            out_i8: &out_i8,
            out_params: &out_params,
            dims: &dims,
        };
        crate::ops::act_quant::dispatch_act_quant(
            self.backend,
            &mut self.encoder_mut(),
            pipeline,
            &bufs,
            m,
            k,
        )
    }

    /// Dispatch the I8 dequant pass: read Q-block weights and write
    /// `[N, K]` packed int8 + `[N, K/32]` f32 scales + `[N, K/32]` f32 qsums.
    /// Pairs with [`Self::matmul_i8`].
    #[allow(clippy::too_many_arguments)]
    pub fn dequant_i8(
        &self,
        pipeline: &B::Pipeline,
        scheme: crate::quant::QuantKind,
        b_quant: BatchBuf<'wsp>,
        b_i8: BatchBuf<'wsp>,
        b_scale: BatchBuf<'wsp>,
        b_qsum: BatchBuf<'wsp>,
        dims: BatchBuf<'wsp>,
        n: u32,
        k: u32,
    ) -> Result<(), B::Error> {
        let b_quant = self.resolve(b_quant);
        let b_i8 = self.resolve(b_i8);
        let b_scale = self.resolve(b_scale);
        let b_qsum = self.resolve(b_qsum);
        let dims = self.resolve(dims);
        let bufs = crate::ops::dequant_i8::DequantI8Bufs {
            b_quant: &b_quant,
            b_i8: &b_i8,
            b_scale: &b_scale,
            b_qsum: &b_qsum,
            dims: &dims,
        };
        crate::ops::dequant_i8::dispatch_dequant_i8(
            self.backend,
            &mut self.encoder_mut(),
            pipeline,
            scheme,
            &bufs,
            n,
            k,
        )
    }

    /// DP4A matmul. A and B are `[M, K]` / `[N, K]` packed int8 (4 per u32)
    /// with per-(row, K/32) f32 scale buffers; output is `[M, N]` paired
    /// `vec2<f16>`.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_i8(
        &self,
        pipeline: &B::Pipeline,
        cfg: &crate::ops::matmul_i8::MatMulI8Config,
        a: BatchBuf<'wsp>,
        a_params: BatchBuf<'wsp>,
        b: BatchBuf<'wsp>,
        b_scale: BatchBuf<'wsp>,
        b_qsum: BatchBuf<'wsp>,
        out: BatchBuf<'wsp>,
        dims: BatchBuf<'wsp>,
        dbg_out: BatchBuf<'wsp>,
        dbg: BatchBuf<'wsp>,
        m: u32,
        n: u32,
    ) -> Result<(), B::Error> {
        let a = self.resolve(a);
        let a_params = self.resolve(a_params);
        let b = self.resolve(b);
        let b_scale = self.resolve(b_scale);
        let b_qsum = self.resolve(b_qsum);
        let out = self.resolve(out);
        let dims = self.resolve(dims);
        let dbg_out = self.resolve(dbg_out);
        let dbg = self.resolve(dbg);
        let bufs = crate::ops::matmul_i8::MatMulI8Bufs {
            a: &a,
            a_params: &a_params,
            b: &b,
            b_scale: &b_scale,
            b_qsum: &b_qsum,
            out: &out,
            dims: &dims,
            dbg_out: &dbg_out,
            dbg: &dbg,
        };
        crate::ops::matmul_i8::dispatch_matmul_i8(
            self.backend,
            &mut self.encoder_mut(),
            pipeline,
            cfg,
            &bufs,
            m,
            n,
        )
    }

    /// Dispatch the bf16-block-sum pass: `b_sum[n, t] = Σ_{k in block t}
    /// b[n, k]`. Output is f32 `[N, K/32]`. Pairs with [`Self::matmul_i8_bf16`]
    /// as the asymmetric correction-term factor.
    pub fn bf16_block_sum(
        &self,
        pipeline: &B::Pipeline,
        b: BatchBuf<'wsp>,
        b_sum: BatchBuf<'wsp>,
        dims: BatchBuf<'wsp>,
        n: u32,
        k: u32,
    ) -> Result<(), B::Error> {
        let b = self.resolve(b);
        let b_sum = self.resolve(b_sum);
        let dims = self.resolve(dims);
        let bufs = crate::ops::bf16_block_sum::Bf16BlockSumBufs {
            b: &b,
            b_sum: &b_sum,
            dims: &dims,
        };
        crate::ops::bf16_block_sum::dispatch_bf16_block_sum(
            self.backend,
            &mut self.encoder_mut(),
            pipeline,
            &bufs,
            n,
            k,
        )
    }

    /// Mixed-precision matmul: A is paired packed-i8 (data, per-K=32 f32
    /// scale); B is dense bf16 weights stored K-major as `array<u32>` (2
    /// bf16 elements per word). Output is `[M, N]` paired `vec2<f16>`.
    /// Used by I8-acts × bf16-weight sites (DiT refiners).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_i8_bf16(
        &self,
        pipeline: &B::Pipeline,
        cfg: &crate::ops::matmul_i8_bf16::MatMulI8Bf16Config,
        a: BatchBuf<'wsp>,
        a_params: BatchBuf<'wsp>,
        b: BatchBuf<'wsp>,
        b_sum: BatchBuf<'wsp>,
        out: BatchBuf<'wsp>,
        dims: BatchBuf<'wsp>,
        m: u32,
        n: u32,
    ) -> Result<(), B::Error> {
        let a = self.resolve(a);
        let a_params = self.resolve(a_params);
        let b = self.resolve(b);
        let b_sum = self.resolve(b_sum);
        let out = self.resolve(out);
        let dims = self.resolve(dims);
        let bufs = crate::ops::matmul_i8_bf16::MatMulI8Bf16Bufs {
            a: &a,
            a_params: &a_params,
            b: &b,
            b_sum: &b_sum,
            out: &out,
            dims: &dims,
        };
        crate::ops::matmul_i8_bf16::dispatch_matmul_i8_bf16(
            self.backend,
            &mut self.encoder_mut(),
            pipeline,
            cfg,
            &bufs,
            m,
            n,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sdpa_i8(
        &self,
        pipeline: &B::Pipeline,
        q: BatchBuf<'wsp>,
        k: BatchBuf<'wsp>,
        v: BatchBuf<'wsp>,
        mask: BatchBuf<'wsp>,
        out: BatchBuf<'wsp>,
        uniform: BatchBuf<'wsp>,
        b: u32,
        s_q: u32,
        h_q: u32,
        d: u32,
    ) -> Result<(), B::Error> {
        let q = self.resolve(q);
        let k = self.resolve(k);
        let v = self.resolve(v);
        let mask = self.resolve(mask);
        let out = self.resolve(out);
        let uniform = self.resolve(uniform);
        let bufs = crate::ops::sdpa_i8::SdpaI8Bufs {
            q: &q,
            k: &k,
            v: &v,
            mask: &mask,
            out: &out,
            uniform: &uniform,
        };
        crate::ops::sdpa_i8::dispatch_sdpa_i8(
            self.backend,
            &mut self.encoder_mut(),
            pipeline,
            &bufs,
            b,
            s_q,
            h_q,
            d,
        )
    }

    /// CL-parameterized subgroup sdpa. `cl` must match the value the bound
    /// `pipeline` was built with (it sets BR = WG/CL for the workgroup grid).
    #[allow(clippy::too_many_arguments)]
    pub fn sdpa_sg(
        &self,
        pipeline: &B::Pipeline,
        q: BatchBuf<'wsp>,
        k: BatchBuf<'wsp>,
        v: BatchBuf<'wsp>,
        mask: BatchBuf<'wsp>,
        uniform: BatchBuf<'wsp>,
        out: BatchBuf<'wsp>,
        cl: u32,
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
        crate::ops::dispatch_sdpa_f16_sg(
            self.backend,
            &mut self.encoder_mut(),
            pipeline,
            &bufs,
            cl,
            b,
            s_q,
            h_q,
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
        op: &O,
        x: BatchBuf<'wsp>,
        w: BatchBuf<'wsp>,
        bias: BatchBuf<'wsp>,
        uniform: BatchBuf<'wsp>,
        out: BatchBuf<'wsp>,
        cout: u32,
        m_spatial: u32,
        batch: u32,
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
            op,
            &bufs,
            cout,
            m_spatial,
            batch,
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

    #[allow(clippy::too_many_arguments)]
    pub fn qkv_split<O: crate::ops::QkvSplitOp>(
        &self,
        pipeline: &B::Pipeline,
        input: BatchBuf<'wsp>,
        q: BatchBuf<'wsp>,
        k: BatchBuf<'wsp>,
        v: BatchBuf<'wsp>,
        uniform: BatchBuf<'wsp>,
        n_words: u32,
    ) -> Result<(), B::Error> {
        let input = self.resolve(input);
        let q = self.resolve(q);
        let k = self.resolve(k);
        let v = self.resolve(v);
        let uniform = self.resolve(uniform);
        let bufs = crate::ops::QkvSplitBufs {
            input: &input,
            q: &q,
            k: &k,
            v: &v,
            uniform: &uniform,
        };
        crate::ops::dispatch_qkv_split::<O, _>(
            self.backend,
            &mut self.encoder_mut(),
            pipeline,
            &bufs,
            n_words,
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
    /// Write raw bytes into any scope-visible buffer (honors the view's
    /// offset). Executes before the scope's encoder at submit (queue write
    /// ordering), so it acts as a pre-fill for buffers the dispatches write.
    pub fn write_bytes(
        &self,
        dst: BatchBuf<'wsp>,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), B::Error> {
        let r = self.resolve(dst);
        self.backend.write_buffer(r.id, r.offset + offset, bytes)
    }

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

/// Dynamic per-phase scope packer for VRAM-budget-driven submission.
///
/// Holds at most one open [`BatchScope`] plus a live-bytes counter against a
/// user-supplied `budget_bytes`. Callers drive a sequence of phases; the
/// packer decides per phase whether to keep dispatching into the same scope
/// (cheap, no extra submit) or cut and open a fresh one (frees prior-phase
/// workspace once the in-flight submit completes).
///
/// At small input sizes where every phase fits within the budget under one
/// scope, the packer collapses to a single submit per block — zero overhead
/// vs. the pre-packer shape.
///
/// Lifetime model: carry buffers across cuts are owned by the packer in
/// `carry_holds: Vec<WsBuf>`. The matching `BufRef`s are imported into the
/// new scope via [`BatchScope::import_copy`], which skips the `&'wsp BufRef`
/// tether — the packer's hold-bag is the runtime guarantee that those
/// `WsBuf`s outlive any scope they're imported into. On each cut, the prior
/// cut's carry is folded into the just-submitted scope's completion future
/// so it can release back to the pool as soon as GPU is done.
///
/// Caller flow:
/// ```ignore
/// let mut packer = ScopePacker::new(workspace, budget);
/// // Phase 1
/// let p1_peak = ...;
/// let scope1 = packer.scope_for_phase(p1_peak);
/// let q = scope1.alloc(...)?;
/// // ...dispatches into scope1...
/// // Cut between phases (or keep going in same scope if budget allows).
/// let [q_in_p2] = packer.advance([q], peak_p2)?;
/// let scope2 = packer.scope();
/// // ...dispatches into scope2, q_in_p2 is a valid BatchBuf here...
/// packer.finish().await?;
/// ```
type ScopeSubmitFuture<'wsp, B> = core::pin::Pin<
    Box<dyn core::future::Future<Output = Result<(), <B as Backend>::Error>> + 'wsp>,
>;

pub struct ScopePacker<'wsp, B: Backend> {
    workspace: &'wsp Workspace<B>,
    budget_bytes: u64,
    current: Option<BatchScope<'wsp, B>>,
    current_live: u64,
    pending: Vec<ScopeSubmitFuture<'wsp, B>>,
    carry_holds: Vec<WsBuf<B>>,
}

impl<'wsp, B: Backend> ScopePacker<'wsp, B>
where
    B::Error: 'wsp,
    B: 'wsp,
{
    /// New packer with a single open scope and no live workspace charged.
    pub fn new(workspace: &'wsp Workspace<B>, budget_bytes: u64) -> Self {
        Self {
            workspace,
            budget_bytes,
            current: Some(workspace.batch()),
            current_live: 0,
            pending: Vec::new(),
            carry_holds: Vec::new(),
        }
    }

    /// Current open scope. Panics if a cut consumed it without opening the
    /// next (internal use after `cut`).
    pub fn scope(&self) -> &BatchScope<'wsp, B> {
        self.current
            .as_ref()
            .expect("ScopePacker: no open scope (cut without re-open?)")
    }

    /// Phase entry: caller declares the next phase's estimated peak
    /// workspace. If charging this peak onto the current scope would push
    /// live bytes above the budget AND the current scope already has
    /// workspace charged to it, returns true (caller should cut by calling
    /// [`Self::advance`]). Otherwise returns false and the caller dispatches
    /// into the existing scope.
    ///
    /// Pure query — does not mutate the packer's live counter. Pair with
    /// [`Self::charge`] once the caller commits to dispatching.
    pub fn would_overflow(&self, peak_bytes: u64) -> bool {
        self.current_live > 0 && self.current_live + peak_bytes > self.budget_bytes
    }

    /// Charge `peak_bytes` against the current scope's live counter. Call
    /// after a phase decides to dispatch into the current scope (i.e. no
    /// cut was needed).
    pub fn charge(&mut self, peak_bytes: u64) {
        self.current_live += peak_bytes;
    }

    /// Cut the current scope: queue its submit (deferred), hand back carry
    /// `BatchBuf`s in the freshly opened next scope, and charge `peak_bytes`
    /// against the new scope's live counter.
    ///
    /// `carry_in` are outputs from the current scope to thread into the new
    /// scope. The packer extracts the matching `WsBuf`s from the submitted
    /// scope, moves them into its hold-bag, and imports their `BufRef`s into
    /// the new scope as the returned `BatchBuf`s (same order as `carry_in`).
    ///
    /// The prior cut's carry (now stale w.r.t. fresh scopes) folds into the
    /// just-submitted scope's completion future so it releases to pool when
    /// GPU finishes consuming it.
    pub fn cut(
        &mut self,
        carry_in: &[BatchBuf<'wsp>],
        peak_bytes: u64,
    ) -> Result<Vec<BatchBuf<'wsp>>, B::Error> {
        let scope = self
            .current
            .take()
            .expect("ScopePacker::cut: no open scope");
        let BatchScope {
            workspace: _,
            backend,
            encoder,
            guards,
            scope_id: _,
            _life: _,
        } = scope;
        let enc = encoder
            .into_inner()
            .expect("BatchScope encoder already taken");
        let mut guards = guards.into_inner();
        let mut prior_carry = core::mem::take(&mut self.carry_holds);
        let mut new_carry: Vec<WsBuf<B>> = Vec::with_capacity(carry_in.len());
        // Fused-pair carries (alloc_pair) yield two carry_in BatchBufs that
        // share one WsBuf id (data + scale sub-views over the same allocation).
        // Move the guard exactly once per unique id; the aliased sub-view
        // stays valid because new_carry holds the WsBuf for the new scope.
        for h in carry_in {
            let id = h.buf.id;
            if new_carry.iter().any(|w| w.id() == id) {
                continue;
            }
            if let Some(idx) = guards.iter().position(|g| g.id() == id) {
                new_carry.push(guards.swap_remove(idx));
            } else if let Some(idx) = prior_carry.iter().position(|g| g.id() == id) {
                new_carry.push(prior_carry.swap_remove(idx));
            } else {
                panic!(
                    "ScopePacker::cut: carry BatchBuf id={:?} not found in current scope guards or prior carry",
                    id
                );
            }
        }
        let fut = backend.submit(enc);
        let completion = async move {
            let _g = guards;
            let _p = prior_carry;
            fut.await
        };
        self.pending.push(Box::pin(completion));
        self.carry_holds = new_carry;
        self.current = Some(self.workspace.batch());
        self.current_live = peak_bytes;
        let new_scope = self.current.as_ref().unwrap();
        // Re-mint one BatchBuf per carry_in entry, preserving each input's
        // exact (id, offset, len) sub-view so fused pairs come back as the
        // same two views the caller passed in.
        Ok(carry_in
            .iter()
            .map(|h| new_scope.import_copy(h.buf))
            .collect())
    }

    /// Decide-and-cut helper. If `would_overflow(peak_bytes)` then `cut(...)`
    /// else pass `carry_in` through (still valid in the current scope) and
    /// `charge(peak_bytes)`. Returns the BatchBufs to use in the next phase.
    pub fn advance(
        &mut self,
        carry_in: &[BatchBuf<'wsp>],
        peak_bytes: u64,
    ) -> Result<Vec<BatchBuf<'wsp>>, B::Error> {
        if self.would_overflow(peak_bytes) {
            self.cut(carry_in, peak_bytes)
        } else {
            self.charge(peak_bytes);
            Ok(carry_in.to_vec())
        }
    }

    /// Submit the final scope and return a future that resolves when every
    /// queued submit (including this one) completes. The packer is consumed.
    ///
    /// `outs` survives the final scope as owned `WsBuf`s (same as
    /// `BatchScope::submit_many_deferred`); the rest of the final scope's
    /// guards drop on GPU completion. Hold-bag (carry from the prior cut)
    /// folds into the final completion future.
    pub fn finish(
        mut self,
        outs: &[BatchBuf<'wsp>],
    ) -> (
        Vec<WsBuf<B>>,
        impl core::future::Future<Output = Result<(), B::Error>> + 'wsp,
    ) {
        let scope = self.current.take().expect("ScopePacker::finish: no scope");
        let (out_wsbufs, fut) = scope.submit_many_deferred(outs);
        let prior_carry = self.carry_holds;
        let pending = self.pending;
        let final_fut = async move {
            let _hold = prior_carry;
            for f in pending {
                f.await?;
            }
            fut.await
        };
        (out_wsbufs, final_fut)
    }

    /// Submit-void variant of [`Self::finish`] — no output buffers survive.
    pub fn finish_void(
        mut self,
    ) -> impl core::future::Future<Output = Result<(), B::Error>> + 'wsp {
        let scope = self.current.take().expect("ScopePacker::finish: no scope");
        let fut = scope.submit_deferred();
        let prior_carry = self.carry_holds;
        let pending = self.pending;
        async move {
            let _hold = prior_carry;
            for f in pending {
                f.await?;
            }
            fut.await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct MockBackend {
        next: AtomicU64,
        allocated: Mutex<std::collections::HashSet<GpuBufferId>>,
        mem: Arc<crate::mem::MemAccount>,
    }

    impl MockBackend {
        fn new() -> Self {
            Self {
                next: AtomicU64::new(1),
                allocated: Default::default(),
                mem: crate::mem::MemAccount::new(),
            }
        }
        fn live(&self) -> usize {
            self.allocated.lock().unwrap().len()
        }
    }

    impl Backend for MockBackend {
        type Error = ();
        type CommandEncoder = ();
        type Pipeline = ();
        fn allocate(&self, _bytes: u64) -> Result<GpuBufferId, ()> {
            let id = GpuBufferId(self.next.fetch_add(1, Ordering::Relaxed));
            self.allocated.lock().unwrap().insert(id);
            Ok(id)
        }
        fn free(&self, id: GpuBufferId) {
            self.allocated.lock().unwrap().remove(&id);
        }
        fn mem_account(&self) -> &Arc<crate::mem::MemAccount> {
            &self.mem
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
        let ws = Workspace::new(b.clone(), MemArbiter::unlimited());
        let r1 = ws.alloc(100).unwrap();
        let r2 = ws.alloc(100).unwrap();
        assert_ne!(r1.id(), r2.id());
        assert_eq!(r1.len(), 100);
        assert_eq!(r1.as_buf_ref().offset, 0);
    }

    #[test]
    fn drop_recycles_same_class() {
        let b = Arc::new(MockBackend::new());
        let ws = Workspace::new(b.clone(), MemArbiter::unlimited());
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
        let ws = Workspace::new(b.clone(), MemArbiter::unlimited());
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
            let ws = Workspace::new(b.clone(), MemArbiter::unlimited());
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
            let ws = Workspace::new(b.clone(), MemArbiter::unlimited());
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
