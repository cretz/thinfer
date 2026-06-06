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

use std::collections::HashMap;
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

    /// Encoding of the bytes that land on GPU: the transcode target when
    /// set, the file encoding otherwise.
    pub fn gpu_encoding(&self) -> StorageEncoding {
        match self.transcode {
            Some(k) => StorageEncoding::Quant(k),
            None => self.encoding,
        }
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
}

/// One slot in a `WeightRing`. `id` is `None` until the slot is first
/// populated (lazy allocation). `occupant` is the handle whose bytes the
/// slot currently holds; `pin_count` blocks recycle while a forward holds a
/// view into the slot.
struct RingSlot {
    id: Option<GpuBufferId>,
    occupant: Option<WeightHandle>,
    pin_count: u32,
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
            inner: Arc::new(Mutex::new(Inner {
                metas: Vec::new(),
                gpu: HashMap::new(),
                gpu_bytes: 0,
                gpu_lru: Vec::new(),
                rings: HashMap::new(),
                handle_to_ring: HashMap::new(),
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

    /// Page the weight up to GPU. Returns a `GpuView` that pins the buffer for
    /// its lifetime. Drop the view to release the pin; subsequent eviction may
    /// reclaim the buffer.
    pub async fn acquire<'a, B: Backend>(
        &'a self,
        handle: WeightHandle,
        backend: &B,
    ) -> Result<GpuView<'a>, ResidencyError<S::Error, B::Error>> {
        // Ring-bound handles take a separate code path: no pool, no LRU,
        // fixed-size rotating slots overwritten via `write_buffer`.
        let ring_id = {
            let g = self.inner.lock().unwrap();
            g.handle_to_ring.get(&handle).copied()
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

        // GPU miss. Stream from the mmap'd source directly into the upload
        // payload; no host-side cache. The temporary `Vec<u8>` lives only
        // until upload returns - tracked as RAM Upload so the budget sees
        // this transient peak. Weights matching a `weight_prep` kernel read
        // their raw bytes untouched (the GPU transcodes/transposes during
        // upload); everything else takes the CPU `read_for_gpu` path.
        let prep = Self::prep_op(&meta);
        let _upload_charge = RamCharge::new(
            backend.mem_account().clone(),
            RamCategory::Upload,
            meta.storage_bytes().max(meta.on_disk_bytes),
        );
        let payload = match prep {
            Some(_) => self.read_raw::<B>(&meta).await?,
            None => self.read_for_gpu::<B>(&meta).await?,
        };

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
            let victim = g.gpu_lru.iter().position(|h| {
                g.gpu
                    .get(h)
                    .is_some_and(|e| e.pin_count == 0 && e.bytes == gpu_size)
            });
            victim.map(|pos| {
                let h = g.gpu_lru.remove(pos);
                let e = g.gpu.remove(&h).expect("victim came from lru scan");
                g.gpu_bytes -= e.bytes;
                tracing::info!(
                    target: crate::trace::WEIGHT_EVICT,
                    op = "recycle",
                    handle = h.0,
                    bytes = e.bytes,
                );
                e.id
            })
        };
        let id = match recycled {
            Some(id) => id,
            None => {
                self.arbiter.ensure_headroom(mem, gpu_size);
                backend
                    .allocate_in(gpu_size, VramCategory::Weights)
                    .map_err(ResidencyError::Backend)?
            }
        };
        match prep {
            Some(op) => {
                // Transient staging for the raw bytes lives inside
                // `weight_prep`; reserve headroom for it here where the
                // arbiter is in reach (dims uniform is noise, lumped in).
                self.arbiter.ensure_headroom(mem, payload.len() as u64 + 16);
                // Diag-gated clock: `Instant::now` panics on wasm32 and the
                // timing only surfaces via the diag event below.
                let t_prep = tracing::enabled!(target: "thinfer::diag", tracing::Level::INFO)
                    .then(std::time::Instant::now);
                let done = backend
                    .weight_prep(op, &payload, &BufRef::new(id, gpu_size))
                    .await
                    .map_err(ResidencyError::Backend)?;
                if done {
                    tracing::info!(
                        target: "thinfer::diag",
                        id = %meta.id.0,
                        mb = (payload.len() as f64) / 1.0e6,
                        prep_ms = t_prep.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0),
                        op = ?op,
                        "gpu prep"
                    );
                } else {
                    // Backend without GPU prep (test mocks): produce the
                    // same bytes on the CPU.
                    let bytes = match op {
                        WeightPrep::Q8_0FromBf16 { .. } => {
                            let mut dst = vec![0u8; gpu_size as usize];
                            crate::quant::encode_q8_0_from_bf16(&payload, &mut dst);
                            dst
                        }
                        WeightPrep::TransposeBf16 { n, k } => transpose_bf16_cpu(
                            &payload,
                            payload.len(),
                            gpu_size as usize,
                            n as usize,
                            k as usize,
                        ),
                    };
                    backend
                        .write_buffer(id, 0, &bytes)
                        .map_err(ResidencyError::Backend)?;
                }
            }
            None => backend
                .write_buffer(id, 0, payload.as_slice())
                .map_err(ResidencyError::Backend)?,
        }

        let mut g = self.inner.lock().unwrap();
        g.gpu.insert(
            handle,
            GpuEntry {
                id,
                bytes: gpu_size,
                pin_count: 1,
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
            (meta, bytes_per_slot, handle_bytes, idx, needs_alloc)
        };

        // If we need to allocate, ensure the new slot's footprint fits
        // under the VRAM budget (counting workspace and other rings as
        // already-reserved).
        if needs_alloc {
            if bytes_per_slot > self.budget.vram_bytes {
                return Err(ResidencyError::BudgetTooSmall {
                    needed: bytes_per_slot,
                    have: self.budget.vram_bytes,
                    tier: "vram",
                });
            }
            self.arbiter
                .ensure_headroom(backend.mem_account(), bytes_per_slot);
        }

        // Read bytes off-mutex (mmap path may touch async).
        let _upload_charge = RamCharge::new(
            backend.mem_account().clone(),
            RamCategory::Upload,
            handle_bytes,
        );
        let bytes = self.read_for_gpu::<B>(&meta).await?;
        debug_assert_eq!(bytes.len() as u64, handle_bytes);

        // Allocate (if needed), claim the slot, and write the contents.
        let id = if needs_alloc {
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
        backend
            .write_buffer(id, 0, bytes.as_slice())
            .map_err(ResidencyError::Backend)?;

        let mut g = self.inner.lock().unwrap();
        let r = g.rings.get_mut(&ring).expect("ring registered");
        r.slots[slot_idx].occupant = Some(handle);
        r.slots[slot_idx].pin_count = 1;
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

    /// Stream `meta` from source into the GPU storage layout.
    /// Bf16 source: passthrough (bytes are already bf16). `Linear2D` applies a
    /// block-tiled u16-stride transpose. Quant (GGUF Q-block) source: byte
    /// passthrough; GGUF stores `[N, K]` block-major already, so Linear2D is
    /// rejected. F32 source: not implemented on the GPU path (would need a
    /// per-binding kernel flavor; add when a model requires it). Output length
    /// matches `meta.storage_bytes()`.
    async fn read_for_gpu<B: Backend>(
        &self,
        meta: &WeightMeta,
    ) -> Result<Vec<u8>, ResidencyError<S::Error, B::Error>> {
        if !matches!(
            meta.encoding,
            StorageEncoding::Bf16 | StorageEncoding::Quant(_)
        ) {
            return Err(ResidencyError::Decode(DecodeError::UnsupportedEncoding(
                meta.encoding,
            )));
        }
        if matches!(meta.encoding, StorageEncoding::Quant(_))
            && !matches!(meta.transpose, TransposePolicy::None)
        {
            return Err(ResidencyError::Decode(DecodeError::UnsupportedEncoding(
                meta.encoding,
            )));
        }
        if let Some(kind) = meta.transcode {
            return self.read_transcoded::<B>(meta, kind).await;
        }
        let storage_len = meta.storage_bytes() as usize;
        let on_disk_len = meta.on_disk_bytes as usize;
        // For bf16 with odd element count, storage_bytes pads to u32. The disk
        // bytes are tight (no padding); we leave the tail as zero. Quant
        // tensors satisfy `storage_bytes == on_disk_bytes` by construction.
        let mut bytes = vec![0u8; storage_len];

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
        let diag = tracing::enabled!(target: "thinfer::diag", tracing::Level::INFO);
        let t_read = diag.then(std::time::Instant::now);
        reader
            .read_at(0, &mut bytes[..on_disk_len])
            .await
            .map_err(|e| ResidencyError::Reader(format!("{e:?}")))?;
        let read_ms = t_read.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0);
        let t_xpose = diag.then(std::time::Instant::now);

        let out = match meta.transpose {
            TransposePolicy::None => Ok(bytes),
            TransposePolicy::Linear2D => {
                if meta.shape.0.len() != 2 {
                    return Err(ResidencyError::BadRank {
                        id: meta.id.clone(),
                        rank: meta.shape.0.len(),
                        wanted: "2D",
                    });
                }
                let n = meta.shape.0[0];
                let k = meta.shape.0[1];
                Ok(transpose_bf16_cpu(&bytes, on_disk_len, storage_len, n, k))
            }
        };
        tracing::info!(
            target: "thinfer::diag",
            id = %meta.id.0,
            mb = (on_disk_len as f64) / 1.0e6,
            read_ms = read_ms,
            transpose_ms = t_xpose.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0),
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
        if meta.encoding != StorageEncoding::Bf16
            || kind != QuantKind::Q8_0
            || !matches!(meta.transpose, TransposePolicy::None)
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
        let diag = tracing::enabled!(target: "thinfer::diag", tracing::Level::INFO);
        let t_read = diag.then(std::time::Instant::now);
        reader
            .read_at(0, &mut src)
            .await
            .map_err(|e| ResidencyError::Reader(format!("{e:?}")))?;
        let read_ms = t_read.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0);
        let mut dst = vec![0u8; meta.storage_bytes() as usize];
        let t_enc = diag.then(std::time::Instant::now);
        crate::quant::encode_q8_0_from_bf16(&src, &mut dst);
        tracing::info!(
            target: "thinfer::diag",
            id = %meta.id.0,
            mb = (meta.on_disk_bytes as f64) / 1.0e6,
            read_ms = read_ms,
            transcode_ms = t_enc.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0),
            "transcode"
        );
        Ok(dst)
    }

    /// Read `meta`'s raw on-disk bytes untouched (no transcode/transpose).
    /// Feeds `Backend::weight_prep`, which does the munging on the GPU.
    async fn read_raw<B: Backend>(
        &self,
        meta: &WeightMeta,
    ) -> Result<Vec<u8>, ResidencyError<S::Error, B::Error>> {
        let mut raw = vec![0u8; meta.on_disk_bytes as usize];
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
        let t_read = tracing::enabled!(target: "thinfer::diag", tracing::Level::INFO)
            .then(std::time::Instant::now);
        reader
            .read_at(0, &mut raw)
            .await
            .map_err(|e| ResidencyError::Reader(format!("{e:?}")))?;
        tracing::info!(
            target: "thinfer::diag",
            id = %meta.id.0,
            mb = (meta.on_disk_bytes as f64) / 1.0e6,
            read_ms = t_read.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0),
            "raw read"
        );
        Ok(raw)
    }

    /// GPU-side prep op for this weight, when its layout fits the
    /// `ops::weight_prep` kernels (rank-2 bf16, whole blocks, dispatch
    /// limits). `None` falls back to the CPU `read_for_gpu` path, which
    /// validates and handles every registered shape.
    fn prep_op(meta: &WeightMeta) -> Option<WeightPrep> {
        if meta.encoding != StorageEncoding::Bf16 || meta.shape.0.len() != 2 {
            return None;
        }
        let n = u32::try_from(meta.shape.0[0]).ok()?;
        let k = u32::try_from(meta.shape.0[1]).ok()?;
        match (meta.transcode, meta.transpose) {
            (Some(QuantKind::Q8_0), TransposePolicy::None) => {
                let blocks = (n as u64) * (k as u64) / 32;
                // Kernel processes block pairs; odd block counts and
                // dispatch-grid overflow fall back to the CPU transcode.
                if !k.is_multiple_of(32) || !blocks.is_multiple_of(2) || blocks / 2 > 65535 * 64 {
                    return None;
                }
                Some(WeightPrep::Q8_0FromBf16 { n, k })
            }
            (None, TransposePolicy::Linear2D) => {
                // Odd N breaks output word alignment (one u32 would span two
                // output rows, racing adjacent threads); real linears are
                // even-N, odd shapes fall back to the CPU transpose.
                if !n.is_multiple_of(2) || k > 65535 || n.div_ceil(2).div_ceil(64) > 65535 {
                    return None;
                }
                Some(WeightPrep::TransposeBf16 { n, k })
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
        let mut g = self.inner.lock().unwrap();
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
        let mut idx = 0;
        while freed < at_least && idx < g.gpu_lru.len() {
            let h = g.gpu_lru[idx];
            if g.gpu.get(&h).is_some_and(|e| e.pin_count == 0) {
                g.gpu_lru.remove(idx);
                let e = g.gpu.remove(&h).expect("lru entries are resident");
                g.gpu_bytes -= e.bytes;
                self.backend.free(e.id);
                freed += e.bytes;
                tracing::info!(
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
                    && let Some(id) = slot.id.take()
                {
                    self.backend.free(id);
                    r.allocated_bytes -= r.bytes_per_slot;
                    slot.occupant = None;
                    freed += r.bytes_per_slot;
                    if freed >= at_least {
                        return freed;
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
