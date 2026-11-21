//! Routed-expert residency: a fixed VRAM slot pool per expert matrix kind
//! (gate / up / down), filled on demand and recycled least-recently-used. One
//! slot index holds all three matrices of one `(layer, expert)`, so a routed
//! expert is resident or not as a unit.
//!
//! The pool is sized once from the VRAM left after the dense weights, KV cache
//! and workspace, and is charged to the shared `MemAccount` as weights (the
//! budget stays a hard ceiling). Per call, the experts the current layer routes
//! to are pinned: a fill never evicts a slot another selected expert occupies.
//!
//! Two fill paths:
//! * wgpu: misses stream from the weight source through the queue's staging;
//!   kernels index slots through a per-pair table.
//! * accelerator: misses are DMA copies on the accelerator stream out of a
//!   page-locked host slot cache (LRU, sized by the RAM budget, grown in blocks
//!   while the driver grants page-locked memory). A host miss reads the source
//!   into a host slot first. Kernels read each expert's address from a device
//!   table `[layer][kind][expert]`, kept pointing at its VRAM slot.

use std::collections::HashMap;
use std::pin::Pin;

use futures::future::{LocalBoxFuture, MaybeDone};

use thinfer_core::Backend;
use thinfer_core::backend::{Accelerator, BufRef, MappedHost, StagedWrite, WgpuBackend};
use thinfer_core::mem::{RamCategory, RamCharge, VramCategory};
use thinfer_core::ops::llm::LlmWeight;
use thinfer_core::tensor::GpuBufferId;
use thinfer_core::weight::{WeightId, WeightReader, WeightSource};

/// The three routed-expert matrices (gate, up, down) of one MoE layer, in
/// GGUF 3D tensors `[experts, n, k]` (expert = outermost, contiguous). Kinds
/// may share a tensor (gate + up merged): expert `e` of kind `k` is
/// `expert_bytes[k]` at `offsets[k] + e * strides[k]` of tensor `ids[k]`.
#[derive(Clone, Debug)]
pub struct ExpertTensors {
    pub ids: [WeightId; 3],
    pub fmts: [LlmWeight; 3],
    /// Bytes of one expert's matrix per kind.
    pub expert_bytes: [u64; 3],
    pub offsets: [u64; 3],
    pub strides: [u64; 3],
}

impl ExpertTensors {
    /// Gate, up and down in three tensors: `(id, format, bytes per expert)`.
    pub fn separate(t: [(WeightId, LlmWeight, u64); 3]) -> Self {
        let [g, u, d] = t;
        Self {
            fmts: [g.1, u.1, d.1],
            expert_bytes: [g.2, u.2, d.2],
            offsets: [0; 3],
            strides: [g.2, u.2, d.2],
            ids: [g.0, u.0, d.0],
        }
    }

    /// Gate + up merged in one tensor (per expert: the gate rows, then the
    /// up rows; quant rows are independent, so each half is byte exact) and
    /// down in another.
    pub fn merged_gate_up(gu: (WeightId, LlmWeight, u64), d: (WeightId, LlmWeight, u64)) -> Self {
        let half = gu.2 / 2;
        Self {
            fmts: [gu.1, gu.1, d.1],
            expert_bytes: [half, half, d.2],
            offsets: [0, half, 0],
            strides: [gu.2, gu.2, d.2],
            ids: [gu.0.clone(), gu.0, d.0],
        }
    }

    /// Byte offset of expert `e` of kind `k` in its tensor.
    pub fn offset(&self, k: usize, e: u32) -> u64 {
        self.offsets[k] + e as u64 * self.strides[k]
    }

    /// `(expert, kind)` requests grouped per tensor (kinds sharing one are
    /// read in one batch, so a reader can coalesce them), each paired with
    /// its destination from `dsts` (in `experts` x kind order).
    fn batches<T>(
        &self,
        experts: impl Iterator<Item = u32>,
        mut dsts: impl Iterator<Item = T>,
    ) -> Vec<(WeightId, Vec<(u64, T)>)> {
        let mut out: Vec<(WeightId, Vec<(u64, T)>)> = Vec::new();
        for e in experts {
            for k in 0..3 {
                let dst = dsts.next().expect("one destination per (expert, kind)");
                let at = match out.iter().position(|(id, _)| *id == self.ids[k]) {
                    Some(i) => i,
                    None => {
                        out.push((self.ids[k].clone(), Vec::new()));
                        out.len() - 1
                    }
                };
                out[at].1.push((self.offset(k, e), dst));
            }
        }
        out
    }
}

/// A tensor's `(offset, dst)` read requests.
type TensorBatch<'a> = (WeightId, Vec<(u64, &'a mut [u8])>);

/// Read every tensor's `(offset, dst)` batch concurrently.
async fn read_batches<S: WeightSource>(
    source: &S,
    batches: &mut [TensorBatch<'_>],
) -> Result<(), ExpertError> {
    let reads = batches.iter_mut().map(|(id, b)| read_tensor(source, id, b));
    for r in futures::future::join_all(reads).await {
        r?;
    }
    Ok(())
}

pub const GATE: usize = 0;
pub const UP: usize = 1;
pub const DOWN: usize = 2;

#[derive(Debug)]
pub enum ExpertError {
    Source(String),
    Backend(String),
    /// The pool cannot hold the experts one layer routes to at once.
    PoolTooSmall {
        need: usize,
        have: usize,
    },
}

fn src_err(e: impl core::fmt::Debug) -> ExpertError {
    ExpertError::Source(format!("{e:?}"))
}

/// Host slot cache telemetry (accelerator mode).
#[derive(Clone, Debug, Default)]
pub struct HostStats {
    pub slots: usize,
    pub hits: u64,
    /// Experts read from the source into host slots.
    pub reads: u64,
    /// Wall time and bytes of those reads.
    pub read_secs: f64,
    pub read_bytes: u64,
    /// Wall time page-locking host blocks.
    pub grow_secs: f64,
    /// Experts read ahead of their layer, and those reads' start-to-finish
    /// wall time (overlapped with compute; only waits count in `read_secs`).
    pub prefetched: u64,
    pub prefetch_secs: f64,
}

pub struct ExpertCache {
    backend: std::sync::Arc<WgpuBackend>,
    layers: Vec<ExpertTensors>,
    slot_bytes: [u64; 3],
    bufs: [GpuBufferId; 3],
    capacity: usize,
    map: HashMap<(u32, u32), u32>,
    owner: Vec<Option<(u32, u32)>>,
    last_use: Vec<u64>,
    clock: u64,
    /// VRAM misses filled since construction, for telemetry.
    pub fills: u64,
    pub hits: u64,
    /// Experts copied into VRAM one layer ahead ([`Self::prefetch_vram`]),
    /// and those the next layer then routed to.
    pub ahead: u64,
    pub ahead_used: u64,
    /// A layer has VRAM slots filled ahead: its table rows need a push and
    /// its kernels a join on the side copy stream.
    ahead_layer: Option<usize>,
    accel: Option<AccelTier>,
}

/// Accelerator-mode state: the device address table and the host slot cache.
struct AccelTier {
    table: GpuBufferId,
    table_addr: u64,
    /// Page-locked host mirror of the table: entries are edited here and a
    /// layer's rows pushed with one async copy (a pageable write would sync
    /// the stream per entry).
    mirror: Box<dyn MappedHost>,
    slot_addr: [u64; 3],
    experts: u32,
    host: HostSlots,
}

/// Bytes of one page-locked host block (drivers cap single allocations).
const HOST_BLOCK_BYTES: u64 = 256 << 20;

/// One page-locked block. `base` is its host mapping, taken once, so
/// in-flight prefetch reads can write it through a shared `Arc` (which keeps
/// the memory alive until they finish).
struct HostBlock {
    mem: Box<dyn MappedHost>,
    base: *mut u8,
}

// SAFETY: `base` points into `mem` (Send + Sync); writers only ever touch
// disjoint slots, coordinated by `HostSlots` (in-flight slots are never
// handed out).
unsafe impl Send for HostBlock {}
unsafe impl Sync for HostBlock {}

/// A slot's kind region as a raw range (for reads running off-thread).
#[derive(Clone, Copy)]
struct RawRegion(*mut u8, usize);

// SAFETY: see `HostBlock`; a region is written by exactly one read.
unsafe impl Send for RawRegion {}

/// `last_use` of a host slot whose prefetch read is still running: never a
/// victim.
const IN_FLIGHT: u64 = u64::MAX;

/// VRAM slot `last_use` of an expert copied in ahead of its layer
/// ([`ExpertCache::prefetch_vram`]): taken last, claimed or released by that
/// layer's `ensure`.
const RESERVED: u64 = u64::MAX;

/// A host block being page-locked on a worker thread (charged already).
struct NextBlock {
    slots: usize,
    charge: RamCharge,
    alloc: std::thread::JoinHandle<Result<Option<Box<dyn MappedHost>>, String>>,
}

/// LRU cache of experts in page-locked host memory. A slot holds all three
/// matrices of one `(layer, expert)` at `kind_off`. Page-locking costs ~36 ms
/// per 256 MiB block (a cold prefill grows ~40 of them), so the next block is
/// always being allocated on a worker thread while the current one fills.
struct HostSlots {
    backend: std::sync::Arc<WgpuBackend>,
    blocks: Vec<std::sync::Arc<HostBlock>>,
    _charges: Vec<RamCharge>,
    next: Option<NextBlock>,
    /// Slots ever handed out; slots are never released, only reassigned, so
    /// `[used..]` are the free ones.
    used: usize,
    kind_off: [u64; 3],
    slot_bytes: u64,
    per_block: usize,
    max_slots: usize,
    /// The driver refused a block: stop growing.
    capped: bool,
    map: HashMap<(u32, u32), u32>,
    owner: Vec<Option<(u32, u32)>>,
    last_use: Vec<u64>,
    /// Telemetry: host-cache hits, source reads, and the wall time + bytes
    /// of the source read batches.
    hits: u64,
    reads: u64,
    read_secs: f64,
    read_bytes: u64,
    /// Wall time spent allocating (page-locking) host blocks.
    grow_secs: f64,
    /// Experts read ahead by `prefetch`, and those reads' start-to-finish time.
    prefetched: u64,
    prefetch_secs: f64,
}

impl HostSlots {
    fn new(backend: &std::sync::Arc<WgpuBackend>, kind_bytes: [u64; 3], max_slots: usize) -> Self {
        let mut kind_off = [0u64; 3];
        let mut at = 0;
        for k in 0..3 {
            kind_off[k] = at;
            at += kind_bytes[k].next_multiple_of(256);
        }
        let slot_bytes = at;
        Self {
            backend: backend.clone(),
            blocks: Vec::new(),
            _charges: Vec::new(),
            next: None,
            used: 0,
            kind_off,
            slot_bytes,
            per_block: (HOST_BLOCK_BYTES / slot_bytes).max(1) as usize,
            max_slots,
            capped: false,
            map: HashMap::new(),
            owner: Vec::new(),
            last_use: Vec::new(),
            hits: 0,
            reads: 0,
            read_secs: 0.0,
            read_bytes: 0,
            grow_secs: 0.0,
            prefetched: 0,
            prefetch_secs: 0.0,
        }
    }

    fn addr(&self, slot: u32, kind: usize) -> u64 {
        let (b, i) = (
            slot as usize / self.per_block,
            slot as usize % self.per_block,
        );
        self.blocks[b].mem.device_addr() + i as u64 * self.slot_bytes + self.kind_off[kind]
    }

    /// Start page-locking the next block on a worker thread, if the budget
    /// has room for one.
    fn prefetch_block(&mut self) {
        let slots = self
            .per_block
            .min(self.max_slots.saturating_sub(self.owner.len()));
        if self.capped || self.next.is_some() || slots == 0 {
            return;
        }
        let bytes = slots as u64 * self.slot_bytes;
        let charge = RamCharge::new(
            self.backend.mem_account().clone(),
            RamCategory::Other,
            bytes,
        );
        let backend = self.backend.clone();
        let alloc = std::thread::Builder::new()
            .name("thinfer-host-alloc".into())
            .spawn(move || match backend.accelerator() {
                Some(acc) => acc.host_alloc(bytes),
                None => Ok(None),
            });
        match alloc {
            Ok(alloc) => {
                self.next = Some(NextBlock {
                    slots,
                    charge,
                    alloc,
                })
            }
            Err(_) => self.capped = true,
        }
    }

    /// Add the pending block (waiting for it if still in flight) and start
    /// the one after it. `false` = the budget or the driver allows no more.
    fn grow(&mut self) -> bool {
        self.prefetch_block();
        let Some(next) = self.next.take() else {
            return false;
        };
        let t = std::time::Instant::now();
        let block = next.alloc.join();
        self.grow_secs += t.elapsed().as_secs_f64();
        match block {
            Ok(Ok(Some(mut mem))) => {
                // Blocks are full-size in slot arithmetic; a short last block
                // only ever holds `slots` slots.
                let base = mem.bytes().as_mut_ptr();
                self.blocks
                    .push(std::sync::Arc::new(HostBlock { mem, base }));
                self._charges.push(next.charge);
                self.owner.extend((0..next.slots).map(|_| None));
                self.last_use.extend((0..next.slots).map(|_| 0));
                self.prefetch_block();
                true
            }
            _ => {
                self.capped = true;
                false
            }
        }
    }

    /// A free slot (growing if possible), else the least recently used one
    /// whose `last_use` passes `evictable` (in-flight slots never do).
    fn victim(&mut self, evictable: impl Fn(u64) -> bool) -> Option<u32> {
        if self.used < self.owner.len() || self.grow() {
            self.used += 1;
            return Some(self.used as u32 - 1);
        }
        (0..self.owner.len())
            .filter(|&s| self.last_use[s] != IN_FLIGHT && evictable(self.last_use[s]))
            .min_by_key(|&s| self.last_use[s])
            .map(|s| s as u32)
    }

    /// Point `slot` at `key` (dropping its previous owner).
    fn assign(&mut self, slot: u32, key: (u32, u32), tag: u64) {
        if let Some(old) = self.owner[slot as usize].replace(key) {
            self.map.remove(&old);
        }
        self.map.insert(key, slot);
        self.last_use[slot as usize] = tag;
    }

    /// Raw byte ranges of distinct slots' kind regions.
    fn raw_regions(&self, want: &[(u32, usize, u64)]) -> Vec<RawRegion> {
        want.iter()
            .map(|&(slot, kind, len)| {
                let (b, i) = (
                    slot as usize / self.per_block,
                    slot as usize % self.per_block,
                );
                let off = i as u64 * self.slot_bytes + self.kind_off[kind];
                // SAFETY: inside the live block (slot index < its slot count).
                RawRegion(
                    unsafe { self.blocks[b].base.add(off as usize) },
                    len as usize,
                )
            })
            .collect()
    }

    /// Mutable byte ranges of distinct slots' kind regions.
    fn regions(&mut self, want: &[(u32, usize, u64)]) -> Vec<&mut [u8]> {
        self.raw_regions(want)
            .into_iter()
            // SAFETY: `want` names distinct (slot, kind) pairs, i.e. disjoint
            // ranges inside live blocks owned by `self`, borrowed mutably for
            // the returned lifetime; no prefetch writes them (not in flight).
            .map(|r| unsafe { std::slice::from_raw_parts_mut(r.0, r.1) })
            .collect()
    }
}

/// The running reads of a [`Prefetch`].
type Reads<'a> = LocalBoxFuture<'a, Result<(), ExpertError>>;

/// Source reads of experts started ahead of their layer (see
/// [`ExpertCache::prefetch`]); complete with [`ExpertCache::finish`].
pub struct Prefetch<'a> {
    reads: Pin<Box<MaybeDone<Reads<'a>>>>,
    /// `(expert key, host slot)` in flight.
    loads: Vec<((u32, u32), u32)>,
    bytes: u64,
    started: std::time::Instant,
    /// Keeps the written blocks alive while the reads run (dropped after
    /// `reads`: field order).
    _blocks: Vec<std::sync::Arc<HostBlock>>,
}

impl HostSlots {
    /// No slots, the same layout, never growing: what a cache holds once
    /// its tier is taken ([`ExpertCache::take_host`]).
    fn empty_like(&self) -> Self {
        // The kind sizes that lay out to these offsets (already aligned).
        let o = self.kind_off;
        let kinds = [o[1] - o[0], o[2] - o[1], self.slot_bytes - o[2]];
        let mut s = Self::new(&self.backend, kinds, 0);
        s.capped = true;
        s
    }

    /// Settle for keeping between jobs: the speculative next block goes, a
    /// slot whose read never finished is emptied, recency and telemetry
    /// restart (the next cache's clock starts over).
    fn park(&mut self) {
        if let Some(next) = self.next.take() {
            let _ = next.alloc.join();
        }
        for s in 0..self.owner.len() {
            if self.last_use[s] == IN_FLIGHT
                && let Some(key) = self.owner[s].take()
            {
                self.map.remove(&key);
            }
            self.last_use[s] = 0;
        }
        self.hits = 0;
        self.reads = 0;
        self.read_secs = 0.0;
        self.read_bytes = 0;
        self.grow_secs = 0.0;
        self.prefetched = 0;
        self.prefetch_secs = 0.0;
    }

    /// Page-locked bytes held.
    fn bytes(&self) -> u64 {
        // A block is allocated as exactly its slots.
        self.owner.len() as u64 * self.slot_bytes
    }
}

/// A model's page-locked host expert cache, taken out of its
/// [`ExpertCache`] to outlive the model's VRAM ([`ExpertCache::take_host`])
/// and handed to the next cache of the same weights
/// ([`ExpertCache::enable_accel`]). Pristine weight bytes only, keyed by
/// `(layer, expert)`: valid for those weights alone.
pub struct HostTier(HostSlots);

impl HostTier {
    /// Page-locked bytes held.
    pub fn bytes(&self) -> u64 {
        self.0.bytes()
    }
}

impl Drop for HostSlots {
    fn drop(&mut self) {
        // The pending block lands (and is freed) before the backend goes.
        if let Some(next) = self.next.take() {
            let _ = next.alloc.join();
        }
    }
}

impl ExpertCache {
    /// The backend the cache allocates on.
    pub(crate) fn backend(&self) -> &std::sync::Arc<WgpuBackend> {
        &self.backend
    }

    /// The accelerator tier is set up ([`Self::enable_accel`]).
    pub fn accel_enabled(&self) -> bool {
        self.accel.is_some()
    }

    /// Slot bytes per kind: the largest expert matrix of that kind over all
    /// layers, rounded to 256 (storage-binding alignment).
    pub fn slot_bytes(layers: &[ExpertTensors]) -> [u64; 3] {
        let mut s = [0u64; 3];
        for l in layers {
            for (k, b) in l.expert_bytes.iter().enumerate() {
                s[k] = s[k].max(b.next_multiple_of(256));
            }
        }
        s
    }

    /// Allocate a pool of `capacity` slots (clamped to the per-binding limit).
    pub fn new(
        backend: &std::sync::Arc<WgpuBackend>,
        layers: Vec<ExpertTensors>,
        capacity: usize,
    ) -> Result<Self, ExpertError> {
        let slot_bytes = Self::slot_bytes(&layers);
        let limit = backend
            .wgpu_device()
            .limits()
            .max_storage_buffer_binding_size;
        let max_slot = slot_bytes.iter().copied().max().unwrap_or(1).max(1);
        let capacity = capacity.min((limit / max_slot) as usize).max(1);
        let mut bufs = [GpuBufferId(0); 3];
        for k in 0..3 {
            bufs[k] = backend
                .allocate_in(capacity as u64 * slot_bytes[k], VramCategory::Weights)
                .map_err(|e| ExpertError::Backend(format!("{e:?}")))?;
        }
        Ok(Self {
            backend: backend.clone(),
            layers,
            slot_bytes,
            bufs,
            capacity,
            map: HashMap::new(),
            owner: vec![None; capacity],
            last_use: vec![0; capacity],
            clock: 0,
            fills: 0,
            hits: 0,
            ahead: 0,
            ahead_used: 0,
            ahead_layer: None,
            accel: None,
        })
    }

    /// Bytes of every routed expert of every layer.
    pub fn total_bytes(&self, experts: u32) -> u64 {
        self.layers
            .iter()
            .map(|l| l.expert_bytes.iter().sum::<u64>() * experts as u64)
            .sum()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn buffer(&self, kind: usize) -> GpuBufferId {
        self.bufs[kind]
    }

    pub fn buffer_len(&self, kind: usize) -> u64 {
        self.capacity as u64 * self.slot_bytes[kind]
    }

    pub fn slot_stride(&self, kind: usize) -> u32 {
        self.slot_bytes[kind] as u32
    }

    pub fn fmt(&self, layer: usize, kind: usize) -> LlmWeight {
        self.layers[layer].fmts[kind]
    }

    pub fn is_accel(&self) -> bool {
        self.accel.is_some()
    }

    /// Host slot cache telemetry.
    pub fn host_stats(&self) -> HostStats {
        self.accel
            .as_ref()
            .map_or(HostStats::default(), |a| HostStats {
                slots: a.host.owner.len(),
                hits: a.host.hits,
                reads: a.host.reads,
                read_secs: a.host.read_secs,
                read_bytes: a.host.read_bytes,
                grow_secs: a.host.grow_secs,
                prefetched: a.host.prefetched,
                prefetch_secs: a.host.prefetch_secs,
            })
    }

    /// Switch to accelerator fills: the device address table, its host
    /// mirror, and a host slot cache of up to `host_bytes` page-locked RAM
    /// (grown on demand). `reuse`: a tier taken from a cache of the same
    /// weights, adopted when its layout matches and it fits `host_bytes`
    /// (else freed first).
    pub fn enable_accel(
        &mut self,
        backend: &WgpuBackend,
        experts: u32,
        host_bytes: u64,
        reuse: Option<HostTier>,
    ) -> Result<(), ExpertError> {
        let acc = backend
            .accelerator()
            .ok_or_else(|| ExpertError::Backend("no accelerator".into()))?;
        let addr = |b: GpuBufferId, len: u64| {
            backend
                .device_address(&BufRef::new(b, len))
                .ok_or_else(|| ExpertError::Backend("expert buffer not device-addressable".into()))
        };
        let entries = self.layers.len() as u64 * 3 * experts as u64;
        let table = backend
            .allocate_in(entries * 8, VramCategory::Weights)
            .map_err(|e| ExpertError::Backend(format!("{e:?}")))?;
        let mut slot_addr = [0u64; 3];
        for (k, a) in slot_addr.iter_mut().enumerate() {
            *a = addr(self.bufs[k], self.buffer_len(k))?;
        }
        let mirror = acc
            .host_alloc(entries * 8)
            .map_err(ExpertError::Backend)?
            .ok_or_else(|| ExpertError::Backend("no mapped host memory".into()))?;
        let mut kind_bytes = [0u64; 3];
        for l in &self.layers {
            for (kb, &b) in kind_bytes.iter_mut().zip(&l.expert_bytes) {
                *kb = (*kb).max(b);
            }
        }
        let fresh = HostSlots::new(&self.backend, kind_bytes, 0);
        let max_slots = (host_bytes / fresh.slot_bytes) as usize;
        let mut host = match reuse {
            Some(HostTier(h))
                if h.kind_off == fresh.kind_off
                    && h.slot_bytes == fresh.slot_bytes
                    && h.owner.len() <= max_slots =>
            {
                h
            }
            // A mismatched tier is freed here, before the fresh one grows.
            _ => fresh,
        };
        host.max_slots = max_slots;
        host.prefetch_block();
        self.map.clear();
        self.owner.iter_mut().for_each(|o| *o = None);
        self.accel = Some(AccelTier {
            table,
            table_addr: addr(table, entries * 8)?,
            mirror,
            slot_addr,
            experts,
            host,
        });
        Ok(())
    }

    /// Take the page-locked host tier out (accelerator mode) to keep it past
    /// this cache, after queued accelerator work that reads its blocks
    /// finishes. `None`: no tier, or the sync failed (nothing is kept).
    pub fn take_host(&mut self) -> Option<HostTier> {
        let a = self.accel.as_mut()?;
        let acc = self.backend.accelerator()?;
        if let Err(e) = acc.join_ahead().and_then(|()| acc.sync()) {
            tracing::warn!(
                target: thinfer_core::trace::ADAPTER,
                "expert host tier: accelerator sync failed: {e}"
            );
            return None;
        }
        let empty = a.host.empty_like();
        let mut host = std::mem::replace(&mut a.host, empty);
        host.park();
        Some(HostTier(host))
    }

    /// The address-table buffer and the byte range of `(layer, kind)`'s
    /// `[experts]` u64 entries (accelerator mode).
    pub fn table(&self, layer: usize, kind: usize) -> Option<(GpuBufferId, u64, u64)> {
        let a = self.accel.as_ref()?;
        let len = a.experts as u64 * 8;
        Some((a.table, (layer as u64 * 3 + kind as u64) * len, len))
    }

    /// Make every expert in `experts` (unique ids) of `layer` resident and
    /// return its slot, in order. wgpu mode: misses upload through the queue
    /// (ordered before the next submit). Accelerator mode: misses copy on the
    /// accelerator stream and their table entries are rewritten, ordered
    /// before any kernel enqueued after this call.
    pub async fn ensure<S: WeightSource>(
        &mut self,
        backend: &WgpuBackend,
        source: &S,
        layer: usize,
        experts: &[u32],
    ) -> Result<Vec<u32>, ExpertError> {
        if experts.len() > self.capacity {
            return Err(ExpertError::PoolTooSmall {
                need: experts.len(),
                have: self.capacity,
            });
        }
        self.clock += 1;
        let now = self.clock;
        let ahead = self.ahead_layer.take() == Some(layer);
        let mut slots = vec![u32::MAX; experts.len()];
        let mut plan = Vec::new();
        // Hits first, so a slot filled ahead for this layer is claimed before
        // any miss looks for a victim.
        for (i, &e) in experts.iter().enumerate() {
            if let Some(&s) = self.map.get(&(layer as u32, e)) {
                if self.last_use[s as usize] == RESERVED {
                    self.ahead_used += 1;
                }
                self.last_use[s as usize] = now;
                slots[i] = s;
                self.hits += 1;
            }
        }
        // Mispredicted ahead fills: evictable first.
        for u in self.last_use.iter_mut().filter(|u| **u == RESERVED) {
            *u = 0;
        }
        for (i, &e) in experts.iter().enumerate() {
            if slots[i] != u32::MAX {
                continue;
            }
            // A victim slot: a free one, else least recently used not
            // touched by this call.
            let v = (0..self.capacity)
                .filter(|&s| self.last_use[s] != now)
                .min_by_key(|&s| (self.owner[s].is_some(), self.last_use[s]))
                .ok_or(ExpertError::PoolTooSmall {
                    need: experts.len(),
                    have: self.capacity,
                })?;
            if let Some(old) = self.owner[v].take() {
                self.map.remove(&old);
            }
            self.owner[v] = Some((layer as u32, e));
            self.map.insert((layer as u32, e), v as u32);
            self.last_use[v] = now;
            slots[i] = v as u32;
            self.fills += 1;
            plan.push((e, v));
        }
        if ahead {
            let a = self
                .accel
                .as_mut()
                .expect("ahead fills are accelerator-only");
            let acc = backend
                .accelerator()
                .ok_or_else(|| ExpertError::Backend("no accelerator".into()))?;
            // This layer's kernels read the slots the side stream filled.
            acc.join_ahead().map_err(ExpertError::Backend)?;
            if plan.is_empty() {
                // Rows were edited in the mirror when the fills were queued.
                let len = 3 * a.experts as u64 * 8;
                let off = layer as u64 * len;
                acc.copy(a.table_addr + off, a.mirror.device_addr() + off, len)
                    .map_err(ExpertError::Backend)?;
            }
        }
        if plan.is_empty() {
            return Ok(slots);
        }
        let t = &self.layers[layer];
        let Some(a) = self.accel.as_mut() else {
            // wgpu: read straight into the queue's mapped staging (one copy).
            for &(e, v) in &plan {
                for k in 0..3 {
                    let (n, at) = (t.expert_bytes[k], t.offset(k, e));
                    let mut staged = backend
                        .stage_write(self.bufs[k], v as u64 * self.slot_bytes[k], n)
                        .map_err(|e| ExpertError::Backend(format!("{e:?}")))?;
                    let mut r = source.open(&t.ids[k]).await.map_err(src_err)?;
                    r.read_at(at, staged.bytes()).await.map_err(src_err)?;
                }
            }
            return Ok(slots);
        };
        let acc = backend
            .accelerator()
            .ok_or_else(|| ExpertError::Backend("no accelerator".into()))?;
        let host = &mut a.host;
        // Host round: experts whose host copy is ready, and loads pending.
        let mut ready: Vec<(u32, usize, u32)> = Vec::new();
        let mut loads: Vec<(u32, u32)> = Vec::new();
        let mut round = 0u64;
        let mut i = 0;
        while i < plan.len() {
            let (e, v) = plan[i];
            let tag = now * 1024 + round;
            if let Some(&hs) = host.map.get(&(layer as u32, e)) {
                debug_assert_ne!(
                    host.last_use[hs as usize], IN_FLIGHT,
                    "prefetch not finished"
                );
                host.last_use[hs as usize] = tag;
                host.hits += 1;
                ready.push((e, v, hs));
                i += 1;
                continue;
            }
            match host.victim(|u| u != tag) {
                Some(hs) => {
                    host.assign(hs, (layer as u32, e), tag);
                    host.reads += 1;
                    loads.push((e, hs));
                    ready.push((e, v, hs));
                    i += 1;
                }
                None if ready.is_empty() => {
                    return Err(ExpertError::Backend(
                        "no page-locked host memory for expert fills".into(),
                    ));
                }
                None => {
                    // Every host slot is in flight this round: finish it, wait
                    // for its copies, then reuse the slots.
                    Self::flush_round(
                        t,
                        host,
                        acc,
                        source,
                        &a.slot_addr,
                        &self.slot_bytes,
                        &mut loads,
                        &mut ready,
                    )
                    .await?;
                    acc.sync().map_err(ExpertError::Backend)?;
                    round += 1;
                }
            }
        }
        Self::flush_round(
            t,
            host,
            acc,
            source,
            &a.slot_addr,
            &self.slot_bytes,
            &mut loads,
            &mut ready,
        )
        .await?;
        for &(e, v) in &plan {
            for k in 0..3 {
                let entry = ((layer * 3 + k) * a.experts as usize + e as usize) * 8;
                let d = a.slot_addr[k] + v as u64 * self.slot_bytes[k];
                a.mirror.bytes()[entry..entry + 8].copy_from_slice(&d.to_le_bytes());
            }
        }
        // Push this layer's three table rows (after the copies, in order).
        let len = 3 * a.experts as u64 * 8;
        let off = layer as u64 * len;
        acc.copy(a.table_addr + off, a.mirror.device_addr() + off, len)
            .map_err(ExpertError::Backend)?;
        Ok(slots)
    }

    /// Copy the experts of `layer` (unique ids predicted from the previous
    /// layer) that sit in the host cache but not in VRAM into VRAM slots on
    /// the accelerator's side stream, so the transfer overlaps this layer's
    /// compute. Call after this layer's `ensure` (its slots, last used now,
    /// are never taken, and earlier layers' kernels finished at its router
    /// sync). The filled slots are reserved until `layer`'s `ensure`, which
    /// joins the side stream; their host sources stay unevictable until the
    /// next sync (which waits for the side stream too).
    pub fn prefetch_vram(
        &mut self,
        backend: &WgpuBackend,
        layer: usize,
        experts: &[u32],
    ) -> Result<(), ExpertError> {
        let Some(a) = self.accel.as_mut() else {
            return Ok(());
        };
        let acc = backend
            .accelerator()
            .ok_or_else(|| ExpertError::Backend("no accelerator".into()))?;
        let now = self.clock;
        let host_tag = now * 1024;
        let t = &self.layers[layer];
        for &e in experts {
            let key = (layer as u32, e);
            if self.map.contains_key(&key) {
                continue;
            }
            let Some(&hs) = a.host.map.get(&key) else {
                continue;
            };
            if a.host.last_use[hs as usize] == IN_FLIGHT {
                continue;
            }
            let Some(v) = (0..self.capacity)
                .filter(|&s| self.last_use[s] < now)
                .min_by_key(|&s| (self.owner[s].is_some(), self.last_use[s]))
            else {
                break;
            };
            let queued = (0..3).try_fold(true, |ok, k| {
                acc.copy_ahead(
                    a.slot_addr[k] + v as u64 * self.slot_bytes[k],
                    a.host.addr(hs, k),
                    t.expert_bytes[k],
                )
                .map(|q| ok && q)
            });
            if !queued.map_err(ExpertError::Backend)? {
                // No side stream: the regular fill path handles it.
                return Ok(());
            }
            if let Some(old) = self.owner[v].replace(key) {
                self.map.remove(&old);
            }
            self.map.insert(key, v as u32);
            self.last_use[v] = RESERVED;
            a.host.last_use[hs as usize] = a.host.last_use[hs as usize].max(host_tag);
            for k in 0..3 {
                let entry = ((layer * 3 + k) * a.experts as usize + e as usize) * 8;
                let d = a.slot_addr[k] + v as u64 * self.slot_bytes[k];
                a.mirror.bytes()[entry..entry + 8].copy_from_slice(&d.to_le_bytes());
            }
            self.ahead += 1;
            self.ahead_layer = Some(layer);
        }
        Ok(())
    }

    /// Start reading the experts of `layer` (unique ids, e.g. predicted from
    /// the previous layer's hidden state) that are neither in VRAM nor in the
    /// host cache into host slots, and return without waiting: the reads run
    /// on the reader's threads while the caller keeps the GPU busy. Slots
    /// whose host -> VRAM copies may still be queued (the latest `ensure`'s)
    /// are never overwritten. Accelerator mode only; [`Self::finish`] it
    /// before the next `ensure`.
    pub fn prefetch<'a, S: WeightSource>(
        &mut self,
        source: &'a S,
        layer: usize,
        experts: &[u32],
    ) -> Option<Prefetch<'a>> {
        let a = self.accel.as_mut()?;
        let host = &mut a.host;
        // Tags of the latest ensure are >= this (its DMA may be pending).
        let floor = self.clock * 1024;
        let mut loads = Vec::new();
        for &e in experts {
            let key = (layer as u32, e);
            if self.map.contains_key(&key) || host.map.contains_key(&key) {
                continue;
            }
            let Some(hs) = host.victim(|u| u < floor) else {
                break;
            };
            host.assign(hs, key, IN_FLIGHT);
            loads.push((key, hs));
        }
        if loads.is_empty() {
            return None;
        }
        host.reads += loads.len() as u64;
        host.prefetched += loads.len() as u64;
        let t = &self.layers[layer];
        let want: Vec<(u32, usize, u64)> = loads
            .iter()
            .flat_map(|&(_, hs)| (0..3).map(move |k| (hs, k, t.expert_bytes[k])))
            .collect();
        let batches = t.batches(
            loads.iter().map(|&((_, e), _)| e),
            host.raw_regions(&want).into_iter(),
        );
        let reads: Reads<'a> = Box::pin(async move {
            // SAFETY: each region is a distinct in-flight slot's range inside
            // a block `Prefetch` keeps alive; nothing else touches in-flight
            // slots until `finish` returns.
            let mut slices: Vec<TensorBatch<'_>> = batches
                .into_iter()
                .map(|(id, b)| {
                    let b = b
                        .into_iter()
                        .map(|(off, r)| (off, unsafe { std::slice::from_raw_parts_mut(r.0, r.1) }))
                        .collect();
                    (id, b)
                })
                .collect();
            read_batches(source, &mut slices).await
        });
        let mut reads = Box::pin(futures::future::maybe_done(reads));
        // One poll submits every read (they run off-thread from here on).
        let _ = reads
            .as_mut()
            .poll(&mut std::task::Context::from_waker(std::task::Waker::noop()));
        Some(Prefetch {
            reads,
            bytes: want.iter().map(|w| w.2).sum(),
            loads,
            started: std::time::Instant::now(),
            _blocks: host.blocks.clone(),
        })
    }

    /// Wait for a [`Self::prefetch`]: its experts become host-cache hits (or,
    /// if a read failed, are dropped from the cache and the error returned).
    pub async fn finish(&mut self, mut p: Prefetch<'_>) -> Result<(), ExpertError> {
        let t_wait = std::time::Instant::now();
        p.reads.as_mut().await;
        let r = p.reads.as_mut().take_output().expect("prefetch output");
        let host = &mut self
            .accel
            .as_mut()
            .expect("prefetch is accelerator-only")
            .host;
        // Only the wait is on the critical path; the overlapped part is free.
        host.read_secs += t_wait.elapsed().as_secs_f64();
        host.read_bytes += p.bytes;
        host.prefetch_secs += p.started.elapsed().as_secs_f64();
        let tag = self.clock * 1024;
        for &(key, hs) in &p.loads {
            if r.is_ok() {
                host.last_use[hs as usize] = tag;
            } else {
                host.owner[hs as usize] = None;
                host.map.remove(&key);
                host.last_use[hs as usize] = 0;
            }
        }
        r
    }

    /// Read `loads` into their host slots (concurrently), then queue the
    /// host -> VRAM copies of `ready`.
    #[allow(clippy::too_many_arguments)]
    async fn flush_round<S: WeightSource>(
        t: &ExpertTensors,
        host: &mut HostSlots,
        acc: &dyn Accelerator,
        source: &S,
        slot_addr: &[u64; 3],
        slot_bytes: &[u64; 3],
        loads: &mut Vec<(u32, u32)>,
        ready: &mut Vec<(u32, usize, u32)>,
    ) -> Result<(), ExpertError> {
        let want: Vec<(u32, usize, u64)> = loads
            .iter()
            .flat_map(|&(_, hs)| (0..3).map(move |k| (hs, k, t.expert_bytes[k])))
            .collect();
        let t_read = std::time::Instant::now();
        {
            // One batch per tensor, so a reader can coalesce neighbouring
            // experts into large reads.
            let mut batches = t.batches(
                loads.iter().map(|&(e, _)| e),
                host.regions(&want).into_iter(),
            );
            read_batches(source, &mut batches).await?;
        }
        host.read_secs += t_read.elapsed().as_secs_f64();
        host.read_bytes += want.iter().map(|w| w.2).sum::<u64>();
        for &(_, v, hs) in ready.iter() {
            for k in 0..3 {
                acc.copy(
                    slot_addr[k] + v as u64 * slot_bytes[k],
                    host.addr(hs, k),
                    t.expert_bytes[k],
                )
                .map_err(ExpertError::Backend)?;
            }
        }
        loads.clear();
        ready.clear();
        Ok(())
    }
}

/// Read one expert tensor's `(offset, dst)` batch.
async fn read_tensor<S: WeightSource>(
    source: &S,
    id: &WeightId,
    batch: &mut [(u64, &mut [u8])],
) -> Result<(), ExpertError> {
    if batch.is_empty() {
        return Ok(());
    }
    let mut r = source.open(id).await.map_err(src_err)?;
    r.read_batch(batch).await.map_err(src_err)
}

impl Drop for ExpertCache {
    fn drop(&mut self) {
        // The accelerator's contract: sync before memory is recycled. Queued
        // CUDA work still uses this cache's memory (expert kernels on the
        // slots, side-stream copies into them, async pushes from the
        // host-mapped table mirror, kernels reading host blocks in place),
        // and an early exit (cancel, error) can leave some in flight. Wait
        // for all of it before the slots, table, mirror and blocks go.
        if self.accel.is_some()
            && let Some(acc) = self.backend.accelerator()
            && let Err(e) = acc.join_ahead().and_then(|()| acc.sync())
        {
            tracing::warn!(
                target: thinfer_core::trace::ADAPTER,
                "expert cache drop: accelerator sync failed: {e}"
            );
        }
        for b in self.bufs {
            self.backend.free(b);
        }
        if let Some(a) = &self.accel {
            self.backend.free(a.table);
        }
    }
}
