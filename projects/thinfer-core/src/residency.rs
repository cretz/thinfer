//! Weight residency manager. Pages weights disk -> VRAM on `acquire`, evicts
//! LRU on the GPU tier to keep VRAM total under `ResidencyBudget.vram_bytes`.
//! Returned `GpuView` pins the GPU buffer so eviction can't steal it
//! mid-forward.
//!
//! No host-side `Vec<u8>` mirror: the `WeightSource` is mmap-backed, so the OS
//! page cache is the host tier and `ResidencyBudget.ram_bytes` is enforced at
//! the upload-staging level, not here.
//!
//! VRAM budget enforcement lives in the shared [`MemArbiter`]: on a miss the
//! acquire path first recycles the oldest unpinned same-size resident's
//! buffer (streaming steady state, no net-new VRAM), else asks the arbiter
//! for headroom before allocating. The inverse direction (workspace growing
//! while weights are warm) flows through the same arbiter: the
//! [`MemReclaimer`] built by [`WeightResidency::reclaimer`] frees unpinned
//! LRU residents and idle ring slots under pressure from any client.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use crate::arbiter::{MemArbiter, MemReclaimer, MemTier};
use crate::backend::{Backend, BufRef, WeightPrep};
use crate::mem::{RamCategory, RamCharge, VramCategory};
use crate::policy::ResidencyBudget;
use crate::quant::QuantKind;
use crate::tensor::{GpuBufferId, Shape, StorageEncoding};
use crate::weight::{DecodeError, WeightId, WeightReader, WeightSource};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightHandle(u32);

/// Caller-defined ring identity. Handles registered with the same `RingId`
/// share a fixed-size buffer ring: each acquire overwrites one slot via
/// `Queue::write_buffer` instead of going through the pool/alloc path. All
/// handles in a ring must declare the same `storage_bytes` (validated at
/// register time).
///
/// Loader convention: one `RingId` per `(pipeline-set, weight-kind)` pair,
/// e.g. main-layers attn_qkv, main-layers ffn_w1, etc. Refiner / encoder
/// sets get separate `RingId`s because their shapes differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RingId(pub u32);

/// Fixed ring depth. `SUBMIT_DEPTH = 2` (two in-flight submits) plus one
/// slot being written for next submit plus one for `idx+2` prefetch
/// headroom. Caller's working set per kind must not exceed this within
/// any window of in-flight submits or acquire will block waiting for a
/// pinned slot to release.
pub const WEIGHT_RING_SLOTS: usize = 4;

/// Bounded host scratch for source -> GPU streaming. Weight bytes transit
/// this much host memory at most, regardless of tensor size (web hard rule:
/// no tensor-sized wasm allocations). 4-byte multiple (write_buffer
/// alignment).
//
// Chunk size is NOT the bottleneck: a 4x (64 MiB) classifier run was wall-time
// identical to 16 MiB, so the upload cost is throughput-bound, not per-call /
// staging-overhead bound. Kept at 16 MiB (bounds in-flight JS heap to
// READ_QUEUE_DEPTH * 16 MiB).
const UPLOAD_CHUNK_BYTES: u64 = 16 * 1024 * 1024;

/// How many chunk reads the streaming path keeps in flight against the
/// source reader at once. The async reader (web OPFS worker) processes them
/// back-to-back, so the read pipe stays full while the engine thread is busy
/// encoding GPU work — without this lead the worker idles between chunks
/// waiting for the next request and read throughput sags below GPU compute
/// rate (GPU then stalls). No-op on mmap readers (`will_read` is a no-op).
/// In-flight bytes live in the reader's own buffers (JS heap on web), not
/// the wasm scratch below, which still holds one chunk.
const READ_PREFETCH_CHUNKS: u64 = 4;

/// Cap on retired upload scratch buffers kept for reuse. Streaming acquires
/// borrow a chunk-sized host buffer and return it here instead of
/// re-allocating (and re-zeroing) per weight; two can be live at once
/// (`join!` of next-acquire + prefetch), so a small cap covers steady state.
const SCRATCH_POOL_CAP: usize = 4;

/// Target VRAM staging footprint for the banded f32 narrow+transpose prep
/// (`NarrowTransposeF32`). The source is staged one row-band at a time so the
/// transient f32 buffer (2x the bf16 weight) never spikes the whole tensor
/// into VRAM and busts the budget; a band this size is comfortably covered by
/// evicting unpinned residents. Rounded down to whole (even) rows per weight.
const PREP_BAND_BYTES: u64 = 32 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransposePolicy {
    /// 1-D weights, biases, norm gains, pad tokens. No layout change.
    None,
    /// 2-D `nn.Linear` weight `[N, K]` transposed to `[K, N]` at upload. Matmul
    /// kernel convention is `A @ B` with B in `[K, N]`.
    Linear2D,
}

#[derive(Clone, Debug)]
pub struct WeightMeta {
    pub id: WeightId,
    pub shape: Shape,
    pub encoding: StorageEncoding,
    pub on_disk_bytes: u64,
    pub transpose: TransposePolicy,
    /// Load-time requantize target. `Some(kind)`: the file holds bf16
    /// `[N, K]` row-major (`K % 32 == 0`) and `read_for_gpu` encodes it
    /// into the GGUF-native quant block stream (`[N, K]` N-major, no
    /// transpose) so the weight rides the quant matmul path. `None`:
    /// upload at `encoding` as-is.
    pub transcode: Option<QuantKind>,
}

impl WeightMeta {
    pub fn elements(&self) -> u64 {
        self.shape.0.iter().map(|&d| d as u64).product()
    }

    /// Encoding of the bytes that land on GPU. Derived from the file
    /// encoding plus the upload transforms:
    /// - `transcode` set: the quant block stream (bf16 file requantized).
    /// - Quant file + `Linear2D`: dense bf16 `[K, N]`. Dequant-at-upload for
    ///   tensors the engine consumes dense even though the GGUF quantized
    ///   them (AdaLN modulation weights).
    /// - F32 file: bf16. GGUF norms/biases ship F32 upcast from the bf16
    ///   checkpoint; the RNE narrowing is lossless (verified at upload) and
    ///   keeps every kernel on the engine-wide bf16 weight storage.
    /// - else: the file encoding as-is.
    pub fn gpu_encoding(&self) -> StorageEncoding {
        match (self.transcode, self.encoding, self.transpose) {
            (Some(k), _, _) => StorageEncoding::Quant(k),
            (None, StorageEncoding::Quant(_), TransposePolicy::Linear2D) => StorageEncoding::Bf16,
            (None, StorageEncoding::F32, _) => StorageEncoding::Bf16,
            // F16 file (GGUF VAE convs): expand to the engine-wide bf16 weight
            // storage, same as F32. Decode happens host-side at upload.
            (None, StorageEncoding::F16, _) => StorageEncoding::Bf16,
            (None, e, _) => e,
        }
    }

    /// True when the on-disk bytes upload unchanged: no transcode, no
    /// transpose, no encoding change. Such weights stream straight into
    /// their GPU buffer through the bounded scratch.
    fn is_passthrough(&self) -> bool {
        self.transcode.is_none()
            && matches!(self.transpose, TransposePolicy::None)
            && self.gpu_encoding() == self.encoding
    }

    /// Bytes the weight occupies on GPU, derived from `gpu_encoding`. Bf16
    /// packs 2 elements per u32, padded up to u32 alignment; consumed via
    /// kernel `load_*` helpers.
    pub fn storage_bytes(&self) -> u64 {
        match self.gpu_encoding() {
            StorageEncoding::Bf16 => self.elements().div_ceil(2) * 4,
            StorageEncoding::F32 => self.elements() * 4,
            StorageEncoding::Quant(k) => k.bytes_for_elements(self.elements()),
            _ => 0,
        }
    }
}

#[derive(Debug)]
pub enum ResidencyError<SE: core::fmt::Debug, BE: core::fmt::Debug> {
    Source(SE),
    Backend(BE),
    Decode(DecodeError),
    UnknownHandle(WeightHandle),
    SizeMismatch {
        id: WeightId,
        expected: u64,
        got: u64,
    },
    BadRank {
        id: WeightId,
        rank: usize,
        wanted: &'static str,
    },
    /// `WeightReader::Error` is a separate associated type from
    /// `WeightSource::Error`; stringified to avoid a third generic on every
    /// `ResidencyError`.
    Reader(String),
    /// One weight's footprint exceeds its tier budget; no eviction policy can
    /// satisfy. Caller's `ResidencyBudget` is too tight for this model.
    BudgetTooSmall {
        needed: u64,
        have: u64,
        tier: &'static str,
    },
    /// Every slot in a ring is currently pinned by an in-flight view. Caller's
    /// working set per ring exceeds `WEIGHT_RING_SLOTS`. Shouldn't happen at
    /// `SUBMIT_DEPTH=2`; if it does, either the in-flight pinset has leaked
    /// or the ring depth was tuned down.
    RingAllSlotsPinned {
        ring: RingId,
    },
}

/// Returned by `register_in_ring`. Kept separate from `ResidencyError` so
/// call sites in loaders don't need to specify `SE`/`BE` generics; the only
/// failure mode is a size disagreement between handles in the same ring.
#[derive(Debug)]
pub enum RegisterRingError {
    /// A handle larger than the ring's existing `bytes_per_slot` was
    /// registered after a slot had already been allocated on GPU. The
    /// loader must register every handle before any `acquire` runs.
    GrowthAfterAlloc { ring: RingId, old: u64, new: u64 },
}

struct GpuEntry {
    id: GpuBufferId,
    bytes: u64,
    pin_count: u32,
    /// Filled (or prefetch-refreshed) but not yet consumed by a real
    /// `acquire`. Reclaim skips fresh entries: their next access is a
    /// certainty, so evicting them trades a guaranteed re-read for
    /// headroom that LRU victims can provide instead.
    fresh: bool,
}

/// One slot in a `WeightRing`. `id` is `None` until the slot is first
/// populated (lazy allocation). `occupant` is the handle whose bytes the
/// slot currently holds; `pin_count` blocks recycle while a forward holds a
/// view into the slot.
struct RingSlot {
    id: Option<GpuBufferId>,
    occupant: Option<WeightHandle>,
    pin_count: u32,
    /// Filled (or prefetch-refreshed) but not yet consumed by a real
    /// acquire. Reclaim drains stale slots before fresh ones (see
    /// `GpuEntry::fresh`): a prefetched block weight will be read again
    /// this step, so it's the worst eviction choice.
    fresh: bool,
}

struct WeightRing {
    bytes_per_slot: u64,
    slots: [RingSlot; WEIGHT_RING_SLOTS],
    /// Round-robin cursor for the next slot to overwrite on miss.
    next_idx: usize,
    /// Total bytes the ring has actually allocated to wgpu so far
    /// (= populated_slot_count * bytes_per_slot). Tracks against the
    /// VRAM ceiling so eviction predicate counts ring footprint as
    /// reserved.
    allocated_bytes: u64,
}

struct Inner {
    metas: Vec<WeightMeta>,
    gpu: HashMap<WeightHandle, GpuEntry>,
    gpu_bytes: u64,
    gpu_lru: Vec<WeightHandle>, // back == MRU
    /// Per-`RingId` fixed slot rings (see `WEIGHT_RING_SLOTS`). Handles in
    /// `handle_to_ring` bypass the LRU path entirely: acquire overwrites
    /// a slot via `write_buffer` and pins it for the view's lifetime.
    rings: HashMap<RingId, WeightRing>,
    handle_to_ring: HashMap<WeightHandle, RingId>,
    /// Phase pin plan (see [`WeightResidency::set_pin_plan`]). Pinned handles
    /// bypass any ring binding (pool path) and stay resident across acquires:
    /// the same-size recycle scan and the reclaimer skip them (the reclaimer
    /// only as a last resort, so the budget stays a hard ceiling). Nothing
    /// uploads at plan time; residency builds on first touch.
    pinned: HashSet<WeightHandle>,
}

impl Inner {
    fn touch(lru: &mut Vec<WeightHandle>, h: WeightHandle) {
        if let Some(pos) = lru.iter().position(|&x| x == h) {
            lru.remove(pos);
        }
        lru.push(h);
    }
}

pub struct WeightResidency<S: WeightSource> {
    source: S,
    budget: ResidencyBudget,
    inner: Arc<Mutex<Inner>>,
    /// VRAM budget owner shared with every other VRAM client (workspace
    /// pools). Created here because the residency budget is where the
    /// `vram_bytes` ceiling enters the system; handed out via
    /// [`Self::arbiter`].
    arbiter: Arc<MemArbiter>,
    /// Reuse pool for chunk-sized upload scratch (see `SCRATCH_POOL_CAP`).
    /// Avoids a per-weight alloc + full zero of `UPLOAD_CHUNK_BYTES`.
    scratch_pool: Mutex<Vec<Vec<u8>>>,
    /// Persistent VRAM staging band for `NarrowTransposeF32` prep, lazily
    /// allocated on first use (when VRAM is still empty) and reused for every
    /// band. Allocating it once - rather than per-acquire - keeps it from being
    /// a net-new allocation against an already-full budget: charged up front,
    /// the warm weight set self-limits to `budget - PREP_BAND_BYTES` via the
    /// arbiter, so the transient narrow never pushes the true peak over budget.
    /// Freed at phase boundaries by [`Self::evict_all_and_free`].
    prep_staging: Mutex<Option<GpuBufferId>>,
    /// Sticky VRAM reserve the weight-acquire path holds free on every weight
    /// admission, so the streamed weight working set caps at `budget - reserve`
    /// and the in-flight transient envelope (concurrent upload staging + the
    /// forward's workspace) never pushes the true peak past the budget. Budget-
    /// independent (transients don't grow with budget). Zero (default) disables
    /// it. Set per phase via [`Self::set_transient_reserve`].
    transient_reserve: AtomicU64,
}

impl<S: WeightSource> WeightResidency<S> {
    pub fn budget(&self) -> &ResidencyBudget {
        &self.budget
    }

    pub fn new(source: S, budget: ResidencyBudget) -> Self {
        Self {
            source,
            arbiter: MemArbiter::new(MemTier::Vram, budget.vram_bytes),
            budget,
            scratch_pool: Mutex::new(Vec::new()),
            prep_staging: Mutex::new(None),
            transient_reserve: AtomicU64::new(0),
            inner: Arc::new(Mutex::new(Inner {
                metas: Vec::new(),
                gpu: HashMap::new(),
                gpu_bytes: 0,
                gpu_lru: Vec::new(),
                rings: HashMap::new(),
                handle_to_ring: HashMap::new(),
                pinned: HashSet::new(),
            })),
        }
    }

    /// The VRAM budget arbiter. Workspaces are constructed with a clone so
    /// every VRAM client gates net-new allocations through one owner.
    pub fn arbiter(&self) -> &Arc<MemArbiter> {
        &self.arbiter
    }

    /// Build the weights-tier [`MemReclaimer`]: frees unpinned LRU residents
    /// (oldest first), then unpinned ring slots. The wiring layer registers
    /// it on [`Self::arbiter`] at `RECLAIM_EVICTABLE_WEIGHTS` so workspace
    /// growth can apply pressure to prefetch-warmed weights.
    pub fn reclaimer<B: Backend + Send + Sync>(&self, backend: Arc<B>) -> Box<dyn MemReclaimer> {
        Box::new(WeightReclaimer {
            inner: Arc::downgrade(&self.inner),
            backend,
        })
    }

    pub fn source(&self) -> &S {
        &self.source
    }

    pub fn register(&self, meta: WeightMeta) -> WeightHandle {
        let mut g = self.inner.lock().unwrap();
        let h = WeightHandle(g.metas.len() as u32);
        g.metas.push(meta);
        h
    }

    /// Register `meta` and bind its handle to a fixed-size ring. All handles
    /// sharing a `RingId` reuse the same `WEIGHT_RING_SLOTS` GPU buffers via
    /// `write_buffer` overwrite. The ring's `bytes_per_slot` is the *max*
    /// `storage_bytes` across registered handles; smaller handles use
    /// prefix bytes of the slot and the returned `BufRef.len` reflects the
    /// handle's actual size.
    ///
    /// Mixed-quant mismatch is intentional (GGUF Q4_K_M tags the first +
    /// last attn blocks as Q6_K, the rest as Q5_K). Growth is only safe
    /// before any `acquire` populates a slot; the loader is expected to
    /// register every handle before kicking off inference, which holds.
    pub fn register_in_ring(
        &self,
        meta: WeightMeta,
        ring: RingId,
    ) -> Result<WeightHandle, RegisterRingError> {
        let bytes = meta.storage_bytes();
        let mut g = self.inner.lock().unwrap();
        match g.rings.get_mut(&ring) {
            None => {
                g.rings.insert(
                    ring,
                    WeightRing {
                        bytes_per_slot: bytes,
                        slots: std::array::from_fn(|_| RingSlot {
                            id: None,
                            occupant: None,
                            pin_count: 0,
                            fresh: false,
                        }),
                        next_idx: 0,
                        allocated_bytes: 0,
                    },
                );
            }
            Some(r) => {
                if bytes > r.bytes_per_slot {
                    if r.slots.iter().any(|s| s.id.is_some()) {
                        return Err(RegisterRingError::GrowthAfterAlloc {
                            ring,
                            old: r.bytes_per_slot,
                            new: bytes,
                        });
                    }
                    r.bytes_per_slot = bytes;
                }
            }
        }
        let h = WeightHandle(g.metas.len() as u32);
        g.metas.push(meta);
        g.handle_to_ring.insert(h, ring);
        Ok(h)
    }

    /// Install a phase pin plan: walk `priority` front-to-back, marking
    /// handles pinned while their cumulative GPU bytes fit `pin_budget_bytes`
    /// (an entry that doesn't fit is skipped; later smaller ones may still
    /// fit). Pin-on-first-touch: nothing uploads here. A pinned handle
    /// bypasses its ring binding (pool path), and once resident it survives
    /// the same-size recycle scan and the reclaim chain (except as a last
    /// resort), so repeated acquires across steps hit without re-reading the
    /// source. Replaces any previous plan; formerly pinned residents become
    /// ordinary LRU entries. Returns `(pinned_bytes, pinned_count)`.
    ///
    /// Caller computes `pin_budget_bytes` from its budget headroom, typically
    /// `vram_bytes - ring_reserved_bytes() - workspace estimate - staging
    /// reserve`. Over-pinning cannot bust the ceiling (the reclaimer's last
    /// resort frees pinned entries) but starves workspace into reclaim
    /// thrash, so the estimate should be conservative.
    pub fn set_pin_plan(&self, priority: &[WeightHandle], pin_budget_bytes: u64) -> (u64, usize) {
        let mut g = self.inner.lock().unwrap();
        g.pinned.clear();
        let mut bytes = 0u64;
        for &h in priority {
            let Some(meta) = g.metas.get(h.0 as usize) else {
                continue;
            };
            let sz = meta.storage_bytes();
            if bytes.saturating_add(sz) > pin_budget_bytes {
                continue;
            }
            if g.pinned.insert(h) {
                bytes += sz;
            }
        }
        (bytes, g.pinned.len())
    }

    /// Drop the pin plan (phase boundary). Resident entries stay warm but
    /// become ordinary LRU/reclaim candidates, so the next phase's
    /// allocations evict them on demand.
    pub fn clear_pin_plan(&self) {
        self.inner.lock().unwrap().pinned.clear();
    }

    /// Upper bound on the VRAM the rings claim once streaming reaches steady
    /// state: every ring fully populated (`WEIGHT_RING_SLOTS * bytes_per_slot`
    /// each). Pin-plan callers subtract this from the budget headroom.
    pub fn ring_reserved_bytes(&self) -> u64 {
        let g = self.inner.lock().unwrap();
        g.rings
            .values()
            .map(|r| WEIGHT_RING_SLOTS as u64 * r.bytes_per_slot)
            .sum()
    }

    /// Sum of the GPU-resident bytes of every weight registered so far. A phase
    /// that pages a known weight set fully resident (e.g. the VAE decode, whose
    /// weights all fit) diffs this across its `register_*` call to learn that
    /// set's footprint, then reserves exactly it when sizing the non-evictable
    /// workspace -- budget-independent, unlike a budget fraction.
    pub fn total_registered_bytes(&self) -> u64 {
        let g = self.inner.lock().unwrap();
        g.metas.iter().map(|m| m.storage_bytes()).sum()
    }

    /// Largest transient upload-staging footprint a single acquire can take:
    /// the max `on_disk_bytes` across registered weights (GPU-prep acquires
    /// stage the raw bytes in VRAM while the prep kernel runs). Pin-plan
    /// callers subtract this from the budget headroom.
    pub fn staging_reserve_bytes(&self) -> u64 {
        let g = self.inner.lock().unwrap();
        g.metas
            .iter()
            .map(|m| m.on_disk_bytes.next_multiple_of(4))
            .max()
            .unwrap_or(0)
    }

    /// Transient VRAM staging a single acquire of `meta` allocates *on top of*
    /// its weight buffer while uploading. The whole-tensor prep path
    /// (`prep_to_gpu`) stages `on_disk_bytes` in VRAM; the banded
    /// `NarrowTransposeF32` path uses the persistent, self-reserved band; the
    /// passthrough / CPU-fallback paths stage in host RAM, not VRAM. The acquire
    /// path reserves at least this much up front so a single weight and its own
    /// staging never together cross the budget even with no transient envelope
    /// configured. The `+ 16` matches the slack `prep_to_gpu` adds to its own
    /// headroom call so that call stays a no-op.
    fn vram_staging_bytes(meta: &WeightMeta, prep: Option<WeightPrep>) -> u64 {
        match prep {
            Some(WeightPrep::Q8_0FromBf16 { .. } | WeightPrep::TransposeBf16 { .. }) => {
                meta.on_disk_bytes.next_multiple_of(4) + 16
            }
            Some(WeightPrep::NarrowTransposeF32 { .. }) | None => 0,
        }
    }

    /// Largest single VRAM upload-staging buffer across registered weights that
    /// actually stage in VRAM (the whole-tensor `Q8_0FromBf16` / `TransposeBf16`
    /// prep path; passthrough and banded-f32 weights stage in host RAM or the
    /// persistent band, not as a net-new VRAM buffer). Callers scale this by the
    /// prefetch concurrency (the driver can have several uploads in flight) to
    /// size [`Self::set_transient_reserve`].
    pub fn vram_staging_reserve_bytes(&self) -> u64 {
        let g = self.inner.lock().unwrap();
        g.metas
            .iter()
            .map(|m| Self::vram_staging_bytes(m, Self::prep_op(m)))
            .max()
            .unwrap_or(0)
    }

    /// Set the sticky VRAM reserve held free on every weight admission (see
    /// [`Self::transient_reserve`]). Set once per phase; zero disables it.
    pub fn set_transient_reserve(&self, bytes: u64) {
        self.transient_reserve.store(bytes, Ordering::Relaxed);
    }

    /// Page the weight up to GPU. Returns a `GpuView` that pins the buffer for
    /// its lifetime. Drop the view to release the pin; subsequent eviction may
    /// reclaim the buffer.
    pub async fn acquire<'a, B: Backend>(
        &'a self,
        handle: WeightHandle,
        backend: &B,
    ) -> Result<GpuView<'a>, ResidencyError<S::Error, B::Error>> {
        // Ring-bound handles take a separate code path: no pool, no LRU,
        // fixed-size rotating slots overwritten via `write_buffer`. Pinned
        // handles bypass their ring binding: the whole point of the pin is
        // residency across acquires, which a rotating slot can't provide.
        let ring_id = {
            let g = self.inner.lock().unwrap();
            if g.pinned.contains(&handle) {
                None
            } else {
                g.handle_to_ring.get(&handle).copied()
            }
        };
        if let Some(rid) = ring_id {
            return self.acquire_from_ring(handle, rid, backend).await;
        }

        let (meta, gpu_hit) = {
            let mut g = self.inner.lock().unwrap();
            let meta = g
                .metas
                .get(handle.0 as usize)
                .cloned()
                .ok_or(ResidencyError::UnknownHandle(handle))?;
            let hit = g.gpu.get_mut(&handle).map(|e| {
                e.pin_count += 1;
                // First access after a fill consumes the freshness: a
                // prefetch fills cold (fresh), the block's real acquire is
                // this hit, and from here the entry is an ordinary LRU
                // candidate again.
                e.fresh = false;
                BufRef::new(e.id, e.bytes)
            });
            if hit.is_some() {
                Inner::touch(&mut g.gpu_lru, handle);
            }
            (meta, hit)
        };
        if let Some(buf) = gpu_hit {
            tracing::trace!(
                target: crate::trace::RESIDENCY_MOVE,
                handle = handle.0,
                bytes = buf.len,
                "gpu hit"
            );
            return Ok(GpuView {
                inner: &self.inner,
                pin: ViewPin::Pool(handle),
                buf,
            });
        }

        // GPU miss. Stream from the source through a bounded scratch into
        // GPU buffers: no host-side cache, no tensor-sized host allocations
        // (hard rule on web, transient-RAM win on native). Weights matching
        // a `weight_prep` kernel stream raw into GPU staging (the kernel
        // transcodes/transposes during upload); plain weights stream straight
        // into their buffer. Only backends without GPU prep (test mocks)
        // take the whole-tensor CPU path.
        let prep = Self::prep_op(&meta).filter(|&op| backend.supports_weight_prep(op));

        let gpu_size = meta.storage_bytes();
        // Single weight must fit under the absolute budget. Any further
        // headroom claimed by live workspace/staging is dynamic and is
        // the arbiter's concern; here we only reject weights that are
        // structurally too large for the budget at all.
        if gpu_size > self.budget.vram_bytes {
            return Err(ResidencyError::BudgetTooSmall {
                needed: gpu_size,
                have: self.budget.vram_bytes,
                tier: "vram",
            });
        }
        // Headroom available: allocate fresh, building residency (large
        // budgets keep every block warm; later acquires are hits). Under
        // pressure: recycle the buffer of the oldest unpinned resident with
        // the exact same byte size: the streaming steady state, where block
        // N-2's buffer becomes block N+1's with no allocate and no upload of
        // anything that wasn't already due for eviction. Only when neither
        // applies does the arbiter reclaim chain run (idle workspace pool
        // first, then evictable weights).
        let mem = backend.mem_account();
        let recycled = if self.arbiter.has_headroom(mem, gpu_size) {
            None
        } else {
            let mut g = self.inner.lock().unwrap();
            // Prefer a stale same-size victim; only recycle a fresh
            // (prefetched-unconsumed) buffer when no stale one exists.
            // Pinned residents are never recycle victims.
            let victim = g
                .gpu_lru
                .iter()
                .position(|h| {
                    g.gpu.get(h).is_some_and(|e| {
                        e.pin_count == 0 && e.bytes == gpu_size && !e.fresh && !g.pinned.contains(h)
                    })
                })
                .or_else(|| {
                    g.gpu_lru.iter().position(|h| {
                        g.gpu.get(h).is_some_and(|e| {
                            e.pin_count == 0 && e.bytes == gpu_size && !g.pinned.contains(h)
                        })
                    })
                });
            victim.map(|pos| {
                let h = g.gpu_lru.remove(pos);
                let e = g.gpu.remove(&h).expect("victim came from lru scan");
                g.gpu_bytes -= e.bytes;
                tracing::debug!(
                    target: crate::trace::WEIGHT_EVICT,
                    op = "recycle",
                    handle = h.0,
                    bytes = e.bytes,
                );
                e.id
            })
        };
        // Hold the in-flight transient envelope (concurrent upload staging + the
        // forward's workspace) free alongside this weight so the streamed weight
        // set caps below budget and the transients never push the true peak past
        // it (a hard ceiling at any value, not a soft target). Floored by this
        // weight's own staging so a single load is safe even with no envelope
        // configured; reclaiming for staging only after the weight is allocated
        // would let the peak overshoot by one staging buffer.
        let reserve = self
            .transient_reserve
            .load(Ordering::Relaxed)
            .max(Self::vram_staging_bytes(&meta, prep));
        let id = match recycled {
            Some(id) => {
                // Weight buffer reused (still charged to the account); make room
                // for the transient envelope that lands on top of it.
                self.arbiter.ensure_headroom(mem, reserve);
                id
            }
            None => {
                self.arbiter.ensure_headroom(mem, gpu_size + reserve);
                backend
                    .allocate_in(gpu_size, VramCategory::Weights)
                    .map_err(ResidencyError::Backend)?
            }
        };
        match prep {
            Some(op) => {
                // Diag-gated clock: the timing only surfaces via the diag
                // event below, and on wasm each enabled read is a JS roundtrip.
                let t_prep = tracing::enabled!(target: "thinfer::diag", tracing::Level::DEBUG)
                    .then(crate::trace::Instant::now);
                self.prep_to_gpu(backend, &meta, op, id, gpu_size).await?;
                tracing::debug!(
                    target: "thinfer::diag",
                    id = %meta.id.0,
                    mb = (meta.on_disk_bytes as f64) / 1.0e6,
                    prep_ms = t_prep.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0),
                    op = ?op,
                    "gpu prep"
                );
            }
            None if meta.is_passthrough() => {
                // Pure passthrough: stream straight into the weight buffer
                // (storage padding past the on-disk bytes is zero-filled).
                self.stream_source_to_gpu(backend, &meta, id, 0, meta.on_disk_bytes, gpu_size)
                    .await?;
            }
            None => {
                // CPU transform fallback (backend lacks GPU prep, or the
                // layout misses the kernel constraints): whole-tensor read.
                let _upload_charge = RamCharge::new(
                    backend.mem_account().clone(),
                    RamCategory::Upload,
                    gpu_size.max(meta.on_disk_bytes),
                );
                let payload = self.read_for_gpu::<B>(&meta).await?;
                backend.mem_account().add_source_bytes(meta.on_disk_bytes);
                backend
                    .write_buffer(id, 0, payload.as_slice())
                    .map_err(ResidencyError::Backend)?;
            }
        }

        let mut g = self.inner.lock().unwrap();
        g.gpu.insert(
            handle,
            GpuEntry {
                id,
                bytes: gpu_size,
                pin_count: 1,
                fresh: true,
            },
        );
        g.gpu_bytes += gpu_size;
        Inner::touch(&mut g.gpu_lru, handle);
        Ok(GpuView {
            inner: &self.inner,
            pin: ViewPin::Pool(handle),
            buf: BufRef::new(id, gpu_size),
        })
    }

    /// Ring path for `acquire`. Three cases:
    /// 1. A slot already holds `handle`: pin++, return.
    /// 2. No occupant match, an unpinned slot at/after `next_idx` is
    ///    available: reuse its buffer (or allocate if slot was never
    ///    populated), `write_buffer` the new contents, pin, advance.
    /// 3. All slots pinned: error. With `WEIGHT_RING_SLOTS = 4` and
    ///    `SUBMIT_DEPTH = 2`, working sets of two views per ring max
    ///    so this should never fire in practice.
    async fn acquire_from_ring<'a, B: Backend>(
        &'a self,
        handle: WeightHandle,
        ring: RingId,
        backend: &B,
    ) -> Result<GpuView<'a>, ResidencyError<S::Error, B::Error>> {
        let (meta, bytes_per_slot, handle_bytes, slot_idx, needs_alloc) = {
            let mut g = self.inner.lock().unwrap();
            let meta = g
                .metas
                .get(handle.0 as usize)
                .cloned()
                .ok_or(ResidencyError::UnknownHandle(handle))?;
            let handle_bytes = meta.storage_bytes();
            let r = g.rings.get_mut(&ring).expect("ring registered");
            let bytes_per_slot = r.bytes_per_slot;
            if let Some(idx) = r.slots.iter().position(|s| s.occupant == Some(handle)) {
                r.slots[idx].pin_count += 1;
                // Consume freshness: the prefetch filled this slot, this
                // hit is the block's real read (see `GpuEntry::fresh`).
                r.slots[idx].fresh = false;
                let id = r.slots[idx].id.expect("populated occupant has id");
                tracing::trace!(
                    target: crate::trace::RESIDENCY_MOVE,
                    handle = handle.0,
                    ring = ring.0,
                    slot = idx,
                    bytes = handle_bytes,
                    "ring hit"
                );
                return Ok(GpuView {
                    inner: &self.inner,
                    pin: ViewPin::Ring { ring, slot: idx },
                    buf: BufRef::new(id, handle_bytes),
                });
            }
            // Miss: find an unpinned slot starting from next_idx.
            let n = r.slots.len();
            let start = r.next_idx;
            let chosen = (0..n)
                .map(|off| (start + off) % n)
                .find(|&i| r.slots[i].pin_count == 0);
            let idx = chosen.ok_or(ResidencyError::RingAllSlotsPinned { ring })?;
            let needs_alloc = r.slots[idx].id.is_none();
            r.next_idx = (idx + 1) % n;
            // Claim = pin. The fill below awaits off-mutex (web OPFS reads
            // suspend to the event loop), so an interleaved acquire under
            // budget pressure would otherwise see pin_count == 0 and free
            // this slot's buffer mid-fill (UnknownBuffer on the write), and
            // an acquire of the old occupant would ring-hit a slot being
            // overwritten. Unpinned again if the fill fails.
            r.slots[idx].pin_count = 1;
            r.slots[idx].occupant = None;
            (meta, bytes_per_slot, handle_bytes, idx, needs_alloc)
        };

        // Allocate (if needed) and fill the claimed slot off-mutex. Fallible
        // section lives in one block so every failure unpins the claim
        // (a leaked pin would dead-end the ring as RingAllSlotsPinned).
        let fill = async {
            // If we need to allocate, ensure the new slot's footprint fits
            // under the VRAM budget (counting workspace and other rings as
            // already-reserved).
            let id = if needs_alloc {
                if bytes_per_slot > self.budget.vram_bytes {
                    return Err(ResidencyError::BudgetTooSmall {
                        needed: bytes_per_slot,
                        have: self.budget.vram_bytes,
                        tier: "vram",
                    });
                }
                self.arbiter
                    .ensure_headroom(backend.mem_account(), bytes_per_slot);
                let id = backend
                    .allocate_in(bytes_per_slot, VramCategory::Weights)
                    .map_err(ResidencyError::Backend)?;
                let mut g = self.inner.lock().unwrap();
                let r = g.rings.get_mut(&ring).expect("ring registered");
                r.slots[slot_idx].id = Some(id);
                r.allocated_bytes += bytes_per_slot;
                id
            } else {
                let g = self.inner.lock().unwrap();
                g.rings[&ring].slots[slot_idx]
                    .id
                    .expect("populated slot has id")
            };
            if meta.is_passthrough() {
                // Passthrough: stream through the bounded scratch.
                self.stream_source_to_gpu(backend, &meta, id, 0, meta.on_disk_bytes, handle_bytes)
                    .await?;
            } else {
                // CPU transform fallback: whole-tensor read.
                let _upload_charge = RamCharge::new(
                    backend.mem_account().clone(),
                    RamCategory::Upload,
                    handle_bytes,
                );
                let bytes = self.read_for_gpu::<B>(&meta).await?;
                backend.mem_account().add_source_bytes(meta.on_disk_bytes);
                debug_assert_eq!(bytes.len() as u64, handle_bytes);
                backend
                    .write_buffer(id, 0, bytes.as_slice())
                    .map_err(ResidencyError::Backend)?;
            }
            Ok(id)
        };
        let id = match fill.await {
            Ok(id) => id,
            Err(e) => {
                let mut g = self.inner.lock().unwrap();
                let r = g.rings.get_mut(&ring).expect("ring registered");
                // Claim-time pin is ours alone (occupant is None, so no hit
                // can stack a second pin); drop it so the slot stays usable.
                r.slots[slot_idx].pin_count = 0;
                return Err(e);
            }
        };

        let mut g = self.inner.lock().unwrap();
        let r = g.rings.get_mut(&ring).expect("ring registered");
        r.slots[slot_idx].occupant = Some(handle);
        // Just filled, not yet consumed (a prefetch leaves this set until
        // the block's acquire hits and clears it).
        r.slots[slot_idx].fresh = true;
        tracing::trace!(
            target: crate::trace::RESIDENCY_MOVE,
            handle = handle.0,
            ring = ring.0,
            slot = slot_idx,
            bytes = handle_bytes,
            slot_bytes = bytes_per_slot,
            "ring miss"
        );
        Ok(GpuView {
            inner: &self.inner,
            pin: ViewPin::Ring {
                ring,
                slot: slot_idx,
            },
            buf: BufRef::new(id, handle_bytes),
        })
    }

    /// CPU-transform fallback: whole-tensor read into the GPU storage layout
    /// (backends without GPU `weight_prep`, i.e. test mocks; passthrough
    /// layouts stream via `stream_source_to_gpu` instead and never land
    /// here). Transforms: bf16 `Linear2D` block-tiled u16-stride transpose;
    /// `transcode` delegates to `read_transcoded`; F32 narrows to bf16
    /// (lossless round-trip enforced; GGUF F32 tensors are upcast bf16);
    /// Quant + `Linear2D` dequants to dense bf16 `[K, N]` (GGUF-quantized
    /// AdaLN). Output length matches `meta.storage_bytes()`.
    async fn read_for_gpu<B: Backend>(
        &self,
        meta: &WeightMeta,
    ) -> Result<Vec<u8>, ResidencyError<S::Error, B::Error>> {
        if let Some(kind) = meta.transcode {
            return self.read_transcoded::<B>(meta, kind).await;
        }
        let storage_len = meta.storage_bytes() as usize;
        let on_disk_len = meta.on_disk_bytes as usize;
        // Same-encoding flows read straight into a storage-sized buffer (for
        // bf16 with odd element count, storage_bytes pads to u32; the tail
        // stays zero). Encoding-changing flows (F32 narrow, quant dequant)
        // read the tight on-disk bytes and build the storage buffer in the
        // transform below.
        let same_encoding = meta.gpu_encoding() == meta.encoding;
        let mut bytes = vec![
            0u8;
            if same_encoding {
                storage_len
            } else {
                on_disk_len
            }
        ];

        let mut reader = self
            .source
            .open(&meta.id)
            .await
            .map_err(ResidencyError::Source)?;
        let total = reader.len();
        if total != meta.on_disk_bytes {
            return Err(ResidencyError::SizeMismatch {
                id: meta.id.clone(),
                expected: meta.on_disk_bytes,
                got: total,
            });
        }
        // One shot read: source is mmap-backed, so this is a memcpy from the
        // OS page cache. No chunking needed.
        let diag = tracing::enabled!(target: "thinfer::diag", tracing::Level::DEBUG);
        let t_read = diag.then(crate::trace::Instant::now);
        reader
            .read_at(0, &mut bytes[..on_disk_len])
            .await
            .map_err(|e| ResidencyError::Reader(format!("{e:?}")))?;
        let read_ms = t_read.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0);
        let t_xpose = diag.then(crate::trace::Instant::now);

        let out = match (meta.encoding, meta.transpose) {
            (StorageEncoding::Bf16 | StorageEncoding::Quant(_), TransposePolicy::None) => Ok(bytes),
            (StorageEncoding::Bf16, TransposePolicy::Linear2D) => {
                let (n, k) = rank2(meta)?;
                Ok(transpose_bf16_cpu(&bytes, on_disk_len, storage_len, n, k))
            }
            (StorageEncoding::F32, transpose) => {
                // Narrow the tight element bytes only: GGUF pads tensor data to
                // its 32-byte alignment, so `on_disk_len` can exceed the real
                // `elements * 4`.
                let elems = meta.elements() as usize;
                let tight = &bytes[..elems * 4];
                let bf16_len = elems * 2; // tight bf16 byte count
                let narrowed = narrow_f32_to_bf16(
                    tight,
                    match transpose {
                        TransposePolicy::None => storage_len,
                        TransposePolicy::Linear2D => bf16_len,
                    },
                );
                match transpose {
                    TransposePolicy::None => Ok(narrowed),
                    TransposePolicy::Linear2D => {
                        let (n, k) = rank2(meta)?;
                        Ok(transpose_bf16_cpu(&narrowed, bf16_len, storage_len, n, k))
                    }
                }
            }
            (StorageEncoding::F16, transpose) => {
                // GGUF VAE conv weights ship fp16; narrow to the engine-wide
                // bf16 weight storage. Tight element bytes only (see F32 note).
                let elems = meta.elements() as usize;
                let tight = &bytes[..elems * 2];
                let bf16_len = elems * 2;
                let narrowed = narrow_f16_to_bf16(
                    tight,
                    match transpose {
                        TransposePolicy::None => storage_len,
                        TransposePolicy::Linear2D => bf16_len,
                    },
                );
                match transpose {
                    TransposePolicy::None => Ok(narrowed),
                    TransposePolicy::Linear2D => {
                        let (n, k) = rank2(meta)?;
                        Ok(transpose_bf16_cpu(&narrowed, bf16_len, storage_len, n, k))
                    }
                }
            }
            (StorageEncoding::Quant(kind), TransposePolicy::Linear2D) => {
                let (n, k) = rank2(meta)?;
                Ok(dequant_transpose_bf16(kind, &bytes, storage_len, n, k))
            }
            (e, _) => Err(ResidencyError::Decode(DecodeError::UnsupportedEncoding(e))),
        };
        tracing::debug!(
            target: "thinfer::diag",
            id = %meta.id.0,
            mb = (on_disk_len as f64) / 1.0e6,
            read_ms = read_ms,
            transform_ms = t_xpose.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0),
            "plain read"
        );
        out
    }

    /// Load-time requantize: read the bf16 `[N, K]` row-major source and
    /// encode it into the GGUF-native quant block stream. `K % 32 == 0`
    /// (validated at registration) so blocks never straddle rows and no
    /// transpose is involved: the quant layout is `[N, K]` N-major, the
    /// same view the matmul B-side reads for file-native quant weights.
    async fn read_transcoded<B: Backend>(
        &self,
        meta: &WeightMeta,
        kind: QuantKind,
    ) -> Result<Vec<u8>, ResidencyError<S::Error, B::Error>> {
        // Q8_0 only, untransposed (the matmul reads quant B N-major). The source
        // may be bf16 (the i8-DP4A sites) OR f32/f16 (the Wan2.2 GGUF module-level
        // embedders/patch transcoded to match the folded Q8_0 block pipeline);
        // f32/f16 narrow to bf16 first, then the shared Q8_0 encode.
        if kind != QuantKind::Q8_0
            || !matches!(meta.transpose, TransposePolicy::None)
            || !matches!(
                meta.encoding,
                StorageEncoding::Bf16 | StorageEncoding::F16 | StorageEncoding::F32
            )
        {
            return Err(ResidencyError::Decode(DecodeError::UnsupportedEncoding(
                meta.encoding,
            )));
        }
        let mut src = vec![0u8; meta.on_disk_bytes as usize];
        let mut reader = self
            .source
            .open(&meta.id)
            .await
            .map_err(ResidencyError::Source)?;
        if reader.len() != meta.on_disk_bytes {
            return Err(ResidencyError::SizeMismatch {
                id: meta.id.clone(),
                expected: meta.on_disk_bytes,
                got: reader.len(),
            });
        }
        let diag = tracing::enabled!(target: "thinfer::diag", tracing::Level::DEBUG);
        let t_read = diag.then(crate::trace::Instant::now);
        reader
            .read_at(0, &mut src)
            .await
            .map_err(|e| ResidencyError::Reader(format!("{e:?}")))?;
        let read_ms = t_read.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0);
        // Narrow f32/f16 -> bf16 (tight element bytes; GGUF pads to 32B) so the
        // Q8_0 encode has a uniform bf16 input.
        let elems = meta.elements() as usize;
        let bf16: std::borrow::Cow<[u8]> = match meta.encoding {
            StorageEncoding::Bf16 => std::borrow::Cow::Borrowed(&src),
            StorageEncoding::F32 => {
                std::borrow::Cow::Owned(narrow_f32_to_bf16(&src[..elems * 4], elems * 2))
            }
            StorageEncoding::F16 => {
                std::borrow::Cow::Owned(narrow_f16_to_bf16(&src[..elems * 2], elems * 2))
            }
            _ => unreachable!("encoding guarded above"),
        };
        let mut dst = vec![0u8; meta.storage_bytes() as usize];
        let t_enc = diag.then(crate::trace::Instant::now);
        crate::quant::encode_q8_0_from_bf16(&bf16, &mut dst);
        tracing::debug!(
            target: "thinfer::diag",
            id = %meta.id.0,
            mb = (meta.on_disk_bytes as f64) / 1.0e6,
            read_ms = read_ms,
            transcode_ms = t_enc.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0),
            "transcode"
        );
        Ok(dst)
    }

    /// Stream `meta`'s raw on-disk bytes into `dst[0..padded_len)` through a
    /// bounded scratch: host memory never exceeds one chunk regardless of
    /// tensor size (the web hard rule). Bytes past `on_disk_bytes` (storage
    /// padding; `padded_len` is 4-byte aligned for `write_buffer`) are
    /// zero-filled. No transform: callers pair this with GPU `weight_prep`
    /// or use it for passthrough layouts.
    /// The persistent `NarrowTransposeF32` staging band, allocated on first
    /// use. Sized `PREP_BAND_BYTES`; the band loop caps each band to fit (real
    /// linears have `K <= ~16K`, so one band is well under this). Charged to
    /// VRAM once so it reserves its own headroom against the budget.
    fn prep_staging_buf<B: Backend>(
        &self,
        backend: &B,
    ) -> Result<GpuBufferId, ResidencyError<S::Error, B::Error>> {
        let mut slot = self.prep_staging.lock().unwrap();
        if let Some(id) = *slot {
            return Ok(id);
        }
        self.arbiter
            .ensure_headroom(backend.mem_account(), PREP_BAND_BYTES + 16);
        let id = backend
            .allocate_in(PREP_BAND_BYTES, VramCategory::Staging)
            .map_err(ResidencyError::Backend)?;
        *slot = Some(id);
        Ok(id)
    }

    /// Stage the source into a transient VRAM buffer and run `op`'s prep
    /// kernel into `dst`. `NarrowTransposeF32` stages one row band at a time so
    /// the f32 staging stays bounded (`PREP_BAND_BYTES`) instead of spiking the
    /// whole 2x-bf16 tensor into VRAM and busting the budget. The bf16-source
    /// ops (`Q8_0FromBf16` / `TransposeBf16`) stage the whole tensor: their
    /// staging is <= the weight buffer, so no banding is needed.
    async fn prep_to_gpu<B: Backend>(
        &self,
        backend: &B,
        meta: &WeightMeta,
        op: WeightPrep,
        dst: GpuBufferId,
        gpu_size: u64,
    ) -> Result<(), ResidencyError<S::Error, B::Error>> {
        let mem = backend.mem_account();
        if let WeightPrep::NarrowTransposeF32 { n, k, .. } = op {
            // Whole even row-bands, capped to the persistent staging band and
            // to the weight itself (small weights take a single band). N even
            // (prep gate) keeps every band start/length even, so the
            // two-rows-per-output-word writes never straddle a band boundary.
            let row_bytes = k as u64 * 4;
            let cap_rows = (PREP_BAND_BYTES / row_bytes.max(1)) as u32 & !1;
            let band_rows = cap_rows.max(2).min(n);
            let staging = self.prep_staging_buf(backend)?;
            let mut n0 = 0u32;
            while n0 < n {
                let band_n = band_rows.min(n - n0);
                let band_bytes = band_n as u64 * row_bytes;
                self.stream_source_to_gpu(
                    backend,
                    meta,
                    staging,
                    n0 as u64 * row_bytes,
                    band_bytes,
                    band_bytes,
                )
                .await?;
                backend
                    .weight_prep(
                        WeightPrep::NarrowTransposeF32 { n, k, n0, band_n },
                        &BufRef::new(staging, band_bytes),
                        &BufRef::new(dst, gpu_size),
                    )
                    .await
                    .map_err(ResidencyError::Backend)?;
                n0 += band_n;
            }
            return Ok(());
        }
        let staging_len = meta.on_disk_bytes.next_multiple_of(4);
        self.arbiter.ensure_headroom(mem, staging_len + 16);
        let staging = backend
            .allocate_in(staging_len, VramCategory::Staging)
            .map_err(ResidencyError::Backend)?;
        let res = match self
            .stream_source_to_gpu(backend, meta, staging, 0, meta.on_disk_bytes, staging_len)
            .await
        {
            Ok(()) => backend
                .weight_prep(
                    op,
                    &BufRef::new(staging, staging_len),
                    &BufRef::new(dst, gpu_size),
                )
                .await
                .map_err(ResidencyError::Backend),
            Err(e) => Err(e),
        };
        backend.free(staging);
        res
    }

    async fn stream_source_to_gpu<B: Backend>(
        &self,
        backend: &B,
        meta: &WeightMeta,
        dst: GpuBufferId,
        src_offset: u64,
        total: u64,
        padded_len: u64,
    ) -> Result<(), ResidencyError<S::Error, B::Error>> {
        // Streams `total` source bytes starting at `src_offset` into
        // `dst[0..padded_len)` (bytes past `total` zero-filled). `src_offset`
        // is non-zero only for the banded f32 prep path, which feeds one row
        // band at a time so the staging buffer stays bounded; the whole-tensor
        // callers pass `src_offset = 0`, `total = on_disk_bytes`.
        //
        // F32 is allowed here only as raw staging bytes for a GPU prep kernel
        // (NarrowTransposeF32); F32 is never `is_passthrough`, so the only
        // caller streaming f32 is the prep staging path above. Bf16/Quant
        // stream straight into their final buffer.
        if !matches!(
            meta.encoding,
            StorageEncoding::Bf16 | StorageEncoding::Quant(_) | StorageEncoding::F32
        ) {
            return Err(ResidencyError::Decode(DecodeError::UnsupportedEncoding(
                meta.encoding,
            )));
        }
        debug_assert!(padded_len >= total && padded_len.is_multiple_of(4));
        let mut reader = self
            .source
            .open(&meta.id)
            .await
            .map_err(ResidencyError::Source)?;
        if reader.len() != meta.on_disk_bytes || src_offset + total > meta.on_disk_bytes {
            return Err(ResidencyError::SizeMismatch {
                id: meta.id.clone(),
                expected: meta.on_disk_bytes,
                got: reader.len(),
            });
        }
        let scratch_len = padded_len.min(UPLOAD_CHUNK_BYTES) as usize;
        let _upload_charge = RamCharge::new(
            backend.mem_account().clone(),
            RamCategory::Upload,
            scratch_len as u64,
        );
        let mut scratch = {
            let mut pool = self.scratch_pool.lock().unwrap();
            pool.pop().unwrap_or_default()
        };
        // Reused buffers are at least chunk-sized; grow a smaller/empty one.
        // Contents are stale but every chunk fully overwrites `[..n_write]`
        // below, so no up-front zeroing is needed.
        if scratch.len() < scratch_len {
            scratch.resize(scratch_len, 0);
        }
        // Prime the read pipeline: issue the first `READ_PREFETCH_CHUNKS`
        // chunk reads up front so the async reader's IO worker stays
        // saturated and never idles between chunks waiting for the engine
        // thread to ask for the next one.
        let mut primed = 0u64;
        for _ in 0..READ_PREFETCH_CHUNKS {
            if primed >= total {
                break;
            }
            let len = (total - primed).min(UPLOAD_CHUNK_BYTES);
            reader.will_read(src_offset + primed, len);
            primed += UPLOAD_CHUNK_BYTES;
        }
        // Split read (read_at: worker RPC + transfer + copy_to scratch) from
        // upload (write_buffer enqueue, which blocks under queue backpressure)
        // so we can attribute the per-step cost to the OPFS/IO side vs the GPU
        // side. Debug-gated; zero clock reads otherwise.
        let diag = tracing::enabled!(target: "thinfer::diag", tracing::Level::DEBUG);
        let mut read_acc = std::time::Duration::ZERO;
        let mut write_acc = std::time::Duration::ZERO;
        let mut off = 0u64;
        while off < padded_len {
            let n_write = (padded_len - off).min(UPLOAD_CHUNK_BYTES) as usize;
            let n_read = (total.saturating_sub(off)).min(n_write as u64) as usize;
            let t_r = diag.then(crate::trace::Instant::now);
            reader
                .read_at(src_offset + off, &mut scratch[..n_read])
                .await
                .map_err(|e| ResidencyError::Reader(format!("{e:?}")))?;
            if let Some(t) = t_r {
                read_acc += t.elapsed();
            }
            backend.mem_account().add_source_bytes(n_read as u64);
            // Keep the pipeline full: hint the chunk `READ_PREFETCH_CHUNKS`
            // ahead of the one just consumed (the nearer chunks are already
            // queued from priming / prior refills).
            let ahead_off = off + READ_PREFETCH_CHUNKS * UPLOAD_CHUNK_BYTES;
            let ahead_read = total.saturating_sub(ahead_off).min(UPLOAD_CHUNK_BYTES);
            if ahead_read > 0 {
                reader.will_read(src_offset + ahead_off, ahead_read);
            }
            scratch[n_read..n_write].fill(0);
            let t_w = diag.then(crate::trace::Instant::now);
            backend
                .write_buffer(dst, off, &scratch[..n_write])
                .map_err(ResidencyError::Backend)?;
            if let Some(t) = t_w {
                write_acc += t.elapsed();
            }
            off += n_write as u64;
        }
        {
            let mut pool = self.scratch_pool.lock().unwrap();
            if pool.len() < SCRATCH_POOL_CAP {
                pool.push(scratch);
            }
        }
        tracing::debug!(
            target: "thinfer::diag",
            id = %meta.id.0,
            mb = (total as f64) / 1.0e6,
            read_ms = read_acc.as_secs_f64() * 1000.0,
            write_ms = write_acc.as_secs_f64() * 1000.0,
            "stream read"
        );
        Ok(())
    }

    /// GPU-side prep op for this weight, when its layout fits the
    /// `ops::weight_prep` kernels (rank-2 bf16, whole blocks, dispatch
    /// limits). `None` falls back to the CPU `read_for_gpu` path, which
    /// validates and handles every registered shape.
    fn prep_op(meta: &WeightMeta) -> Option<WeightPrep> {
        if meta.shape.0.len() != 2 {
            return None;
        }
        let n = u32::try_from(meta.shape.0[0]).ok()?;
        let k = u32::try_from(meta.shape.0[1]).ok()?;
        // Odd N breaks output word alignment for the transpose kernels (one
        // u32 would span two output rows, racing adjacent threads); real
        // linears are even-N, odd shapes fall back to the CPU path.
        let transpose_ok = n.is_multiple_of(2) && k <= 65535 && n.div_ceil(2).div_ceil(64) <= 65535;
        match (meta.encoding, meta.transcode, meta.transpose) {
            (StorageEncoding::Bf16, Some(QuantKind::Q8_0), TransposePolicy::None) => {
                let blocks = (n as u64) * (k as u64) / 32;
                // Kernel processes block pairs; odd block counts and
                // dispatch-grid overflow fall back to the CPU transcode.
                if !k.is_multiple_of(32) || !blocks.is_multiple_of(2) || blocks / 2 > 65535 * 64 {
                    return None;
                }
                Some(WeightPrep::Q8_0FromBf16 { n, k })
            }
            (StorageEncoding::Bf16, None, TransposePolicy::Linear2D) if transpose_ok => {
                Some(WeightPrep::TransposeBf16 { n, k })
            }
            // f32 (safetensors) linear: fuse the RNE narrow with the upload
            // transpose on the GPU so the source streams raw and the CPU stays
            // off the critical path (the whole-tensor `read_for_gpu` narrow is
            // the cold-load bottleneck). f32 + None (small norm/table tensors)
            // stays on the CPU path: negligible and not worth a kernel.
            (StorageEncoding::F32, None, TransposePolicy::Linear2D) if transpose_ok => {
                // Band window filled per-band by `prep_to_gpu`; this is the
                // whole-tensor descriptor used to pick the kernel.
                Some(WeightPrep::NarrowTransposeF32 {
                    n,
                    k,
                    n0: 0,
                    band_n: n,
                })
            }
            _ => None,
        }
    }

    /// Load `handle` to GPU without pinning. Used to overlap upload work for
    /// block N+2 with the join of `submit(N) || acquire(N+1)`. On completion
    /// the weight is in `Gpu` state at MRU; a later `acquire` is a hit.
    ///
    /// Idempotent for in-flight callers: a second `prefetch` of an already-
    /// resident handle just touches LRU. No fire-and-forget semantics; caller
    /// drives the future via `join!` against work that progresses anyway
    /// (typical: the next `submit_void` await). This keeps the API executor-
    /// neutral - no `spawn` requirement on wasm.
    pub async fn prefetch<B: Backend>(
        &self,
        handle: WeightHandle,
        backend: &B,
    ) -> Result<(), ResidencyError<S::Error, B::Error>> {
        // Same code path as acquire, then drop the view. Net effect: entry
        // is in Gpu state with pin_count=0 at MRU.
        let view = self.acquire(handle, backend).await?;
        drop(view);
        Ok(())
    }

    /// Phase-boundary eviction: free every unpinned GPU-resident weight,
    /// releasing the VRAM back to wgpu. Caller's policy: invoke between
    /// pipeline phases (post text_encode, end of denoise) so peak VRAM is
    /// `max(phase)`, not the sum across phases. Pinned entries (mid-forward)
    /// are skipped; they stay resident.
    pub fn evict_all_and_free<B: Backend>(&self, backend: &B) {
        // A phase boundary ends the previous phase's transient envelope: the
        // sticky reserve is sized for one phase's workspace (e.g. the VAE tile
        // working set holds `budget - weights` free). Carrying it into the next
        // phase would force that phase's weight admissions to keep a phantom
        // multi-GiB reserve free, evicting weights it needs (the DiT block loop
        // streaming right after a VAE encode). The next phase sets its own.
        self.transient_reserve.store(0, Ordering::Relaxed);
        // Release the persistent prep staging band too: the next phase
        // re-allocates it lazily (while VRAM is empty again), so its reserved
        // headroom does not carry across a phase boundary.
        if let Some(id) = self.prep_staging.lock().unwrap().take() {
            backend.free(id);
        }
        let mut g = self.inner.lock().unwrap();
        // A phase boundary ends any pin plan: pinned residents free with the
        // rest (the caller installs a fresh plan for the next phase if any).
        g.pinned.clear();
        let victims: Vec<WeightHandle> = g
            .gpu
            .iter()
            .filter(|(_, e)| e.pin_count == 0)
            .map(|(&h, _)| h)
            .collect();
        for h in victims {
            if let Some(pos) = g.gpu_lru.iter().position(|&x| x == h) {
                g.gpu_lru.remove(pos);
            }
            if let Some(e) = g.gpu.remove(&h) {
                g.gpu_bytes -= e.bytes;
                backend.free(e.id);
            }
        }
        // Drain ring slots whose pin_count is zero. Pinned slots stay (caller
        // is still mid-forward); they'll be reclaimed on the next phase
        // boundary. Allocated_bytes drops accordingly so the eviction
        // predicate frees up budget immediately.
        for (_, r) in g.rings.iter_mut() {
            for slot in r.slots.iter_mut() {
                if slot.pin_count == 0 {
                    if let Some(id) = slot.id.take() {
                        backend.free(id);
                        r.allocated_bytes -= r.bytes_per_slot;
                    }
                    slot.occupant = None;
                    slot.fresh = false;
                }
            }
            r.next_idx = 0;
        }
    }
}

/// Weights-tier reclaimer for the [`MemArbiter`] chain. Frees unpinned LRU
/// residents oldest-first, then unpinned ring slots, directly back to the
/// backend (the requester wants raw headroom, not a warm weight buffer;
/// same-size recycle for the weights' own benefit lives in `acquire`).
/// `Weak` so a dropped `WeightResidency` leaves a dead, prunable entry.
struct WeightReclaimer<B: Backend> {
    inner: Weak<Mutex<Inner>>,
    backend: Arc<B>,
}

impl<B: Backend + Send + Sync> MemReclaimer for WeightReclaimer<B> {
    fn label(&self) -> &'static str {
        "weights"
    }

    fn alive(&self) -> bool {
        self.inner.strong_count() > 0
    }

    fn reclaim(&self, at_least: u64) -> u64 {
        let Some(inner) = self.inner.upgrade() else {
            return 0;
        };
        let mut g = inner.lock().unwrap();
        let mut freed = 0u64;
        // Three passes: stale residents first, then fresh (prefetched-but-
        // unconsumed) ones, then pin-plan residents only if still short. A
        // fresh weight is a guaranteed imminent re-read, so it's a bad
        // eviction; a pinned weight is the phase's whole residency win, so
        // it's the worst - but both stay evictable so a working set that
        // genuinely exceeds budget can't deadlock the chain (the budget is
        // a hard ceiling, not a mode switch).
        for (allow_fresh, allow_pinned) in [(false, false), (true, false), (true, true)] {
            let mut idx = 0;
            while freed < at_least && idx < g.gpu_lru.len() {
                let h = g.gpu_lru[idx];
                if (allow_pinned || !g.pinned.contains(&h))
                    && g.gpu
                        .get(&h)
                        .is_some_and(|e| e.pin_count == 0 && (allow_fresh || !e.fresh))
                {
                    g.gpu_lru.remove(idx);
                    let e = g.gpu.remove(&h).expect("lru entries are resident");
                    g.gpu_bytes -= e.bytes;
                    self.backend.free(e.id);
                    freed += e.bytes;
                    tracing::debug!(
                        target: crate::trace::WEIGHT_EVICT,
                        op = "evict",
                        handle = h.0,
                        bytes = e.bytes,
                    );
                } else {
                    idx += 1;
                }
            }
            if freed >= at_least {
                return freed;
            }
            for r in g.rings.values_mut() {
                for slot in r.slots.iter_mut() {
                    if slot.pin_count == 0
                        && (allow_fresh || !slot.fresh)
                        && let Some(id) = slot.id.take()
                    {
                        self.backend.free(id);
                        r.allocated_bytes -= r.bytes_per_slot;
                        slot.occupant = None;
                        slot.fresh = false;
                        freed += r.bytes_per_slot;
                        if freed >= at_least {
                            return freed;
                        }
                    }
                }
            }
        }
        freed
    }
}

/// Block-tiled u16-stride transpose of bf16 `[n, k]` into `[k, n]`. Bf16
/// elements stored as u16; transpose preserves the bit pattern. n*k must
/// equal the element count (padding only matters when odd, and 2-D linear
/// weights never have an odd product). CPU fallback for
/// `WeightPrep::TransposeBf16`; the GPU kernel must stay bit-identical.
/// `meta.shape` as `(n, k)`, or `BadRank` for non-2-D tensors.
fn rank2<SE: core::fmt::Debug, BE: core::fmt::Debug>(
    meta: &WeightMeta,
) -> Result<(usize, usize), ResidencyError<SE, BE>> {
    if meta.shape.0.len() != 2 {
        return Err(ResidencyError::BadRank {
            id: meta.id.clone(),
            rank: meta.shape.0.len(),
            wanted: "2D",
        });
    }
    Ok((meta.shape.0[0], meta.shape.0[1]))
}

/// Narrow an f32 byte stream to packed bf16 in an `out_len`-byte buffer
/// (`out_len >= 2 * elements`; tail stays zero). Plain RNE narrowing, matching
/// every other f32 -> bf16 path in core (`quant.rs`, `dequant_transpose_bf16`,
/// `weight.rs`).
fn narrow_f32_to_bf16(src: &[u8], out_len: usize) -> Vec<u8> {
    let values: &[f32] = bytemuck::cast_slice(src);
    debug_assert!(out_len >= values.len() * 2);
    let mut out = vec![0u8; out_len];
    let dst: &mut [u16] = bytemuck::cast_slice_mut(&mut out[..values.len() * 2]);
    for (&v, d) in values.iter().zip(dst.iter_mut()) {
        *d = half::bf16::from_f32(v).to_bits();
    }
    out
}

/// Narrow fp16 storage to bf16 (via fp32) into a `out_len`-byte buffer. Same
/// shape as [`narrow_f32_to_bf16`] but the source is 2-byte fp16. Used for the
/// GGUF VAE conv weights, which ship fp16 but upload as bf16.
fn narrow_f16_to_bf16(src: &[u8], out_len: usize) -> Vec<u8> {
    let values: &[u16] = bytemuck::cast_slice(src);
    debug_assert!(out_len >= values.len() * 2);
    let mut out = vec![0u8; out_len];
    let dst: &mut [u16] = bytemuck::cast_slice_mut(&mut out[..values.len() * 2]);
    for (&v, d) in values.iter().zip(dst.iter_mut()) {
        *d = half::bf16::from_f32(half::f16::from_bits(v).to_f32()).to_bits();
    }
    out
}

/// Dequantize a GGUF quant block stream (`[N, K]` row-major, `K` a whole
/// number of blocks) into dense packed-bf16 `[K, N]` (the `Linear2D` matmul
/// B layout). Row at a time: dequant one K-row to f32, RNE to bf16, scatter
/// into column `n` of the output.
fn dequant_transpose_bf16(
    kind: QuantKind,
    src: &[u8],
    storage_len: usize,
    n: usize,
    k: usize,
) -> Vec<u8> {
    let bytes_per_row = kind.bytes_for_elements(k as u64) as usize;
    debug_assert_eq!(bytes_per_row * n, src.len());
    let mut row = vec![0f32; k];
    let mut out = vec![0u8; storage_len];
    let dst: &mut [u16] = bytemuck::cast_slice_mut(&mut out[..n * k * 2]);
    for nn in 0..n {
        crate::quant::dequantize_row(kind, &src[nn * bytes_per_row..][..bytes_per_row], &mut row);
        for (kk, &v) in row.iter().enumerate() {
            dst[kk * n + nn] = half::bf16::from_f32(v).to_bits();
        }
    }
    out
}

fn transpose_bf16_cpu(
    bytes: &[u8],
    on_disk_len: usize,
    storage_len: usize,
    n: usize,
    k: usize,
) -> Vec<u8> {
    const BLOCK: usize = 64;
    debug_assert_eq!(n * k * 2, on_disk_len);
    let mut dst = vec![0u8; storage_len];
    let src_u16: &[u16] = bytemuck::cast_slice(&bytes[..on_disk_len]);
    let dst_u16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..on_disk_len]);
    let mut br = 0usize;
    while br < n {
        let br_end = (br + BLOCK).min(n);
        let mut bc = 0usize;
        while bc < k {
            let bc_end = (bc + BLOCK).min(k);
            for row in br..br_end {
                let src_row = &src_u16[row * k + bc..row * k + bc_end];
                for (col_off, &v) in src_row.iter().enumerate() {
                    dst_u16[(bc + col_off) * n + row] = v;
                }
            }
            bc = bc_end;
        }
        br = br_end;
    }
    dst
}

/// Which storage tier owns the pin: pool entry keyed by handle, or a ring
/// slot keyed by ring + slot index.
enum ViewPin {
    Pool(WeightHandle),
    Ring { ring: RingId, slot: usize },
}

/// Guard pinning a weight's GPU buffer for the view's lifetime. Drop releases
/// the pin; eviction (or ring-slot recycle) can then reclaim the buffer on a
/// later `acquire`.
pub struct GpuView<'a> {
    inner: &'a Mutex<Inner>,
    pin: ViewPin,
    buf: BufRef,
}

impl GpuView<'_> {
    pub fn buf(&self) -> BufRef {
        self.buf
    }
}

impl Drop for GpuView<'_> {
    fn drop(&mut self) {
        let mut g = self.inner.lock().unwrap();
        match self.pin {
            ViewPin::Pool(h) => {
                if let Some(e) = g.gpu.get_mut(&h) {
                    e.pin_count = e.pin_count.saturating_sub(1);
                }
            }
            ViewPin::Ring { ring, slot } => {
                if let Some(r) = g.rings.get_mut(&ring) {
                    r.slots[slot].pin_count = r.slots[slot].pin_count.saturating_sub(1);
                }
            }
        }
    }
}
