//! Memory-budget arbiter. Single owner of a tier's byte budget: every
//! net-new allocation in the tier asks the arbiter for headroom first; on a
//! would-exceed-budget allocation the arbiter runs its reclaim chain
//! (registered [`MemReclaimer`]s in priority order) until the allocation
//! fits. Reclaimers are caches (idle workspace pool, evictable weight
//! residents) that can release physical memory without breaking correctness.
//!
//! Design notes (pytorch caching-allocator / D3D12 residency-manager shape):
//! - One budget owner per tier; clients never coordinate peer-to-peer.
//! - Lock order is strictly arbiter -> client. Reclaimers must not call back
//!   into the arbiter and must not allocate.
//! - The budget is a ceiling target, not a hard alloc failure: if the chain
//!   can't free enough, the allocation proceeds and the overshoot is traced;
//!   the e2e true-peak assert surfaces a working set that genuinely exceeds
//!   the budget.
//!
//! Only the `Vram` tier is instantiated today. `Ram` is plumbed because the
//! wasm-web weight path (JS-heap weight bytes, no mmap) is a RAM-resident
//! cache that will want the same arbitration; native RAM use is currently
//! transient (upload staging, readback), which budgets via admission
//! control, not reclaim.

use crate::mem::MemAccount;
use crate::trace;
use std::sync::{Arc, Mutex};

/// Which `MemAccount` total the arbiter budgets against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemTier {
    Vram,
    Ram,
}

/// Reclaim priority: lower runs first. Idle pool buffers cost one
/// `backend.allocate` to recreate; evicted weights cost a source re-read +
/// `write_buffer` on the next acquire.
pub const RECLAIM_IDLE_POOL: u32 = 0;
pub const RECLAIM_EVICTABLE_WEIGHTS: u32 = 1;

/// A cache that can release physical memory on demand. Implementations free
/// at least `at_least` bytes if they can (refunding the `MemAccount`) and
/// return the bytes actually freed. Must not allocate and must not call back
/// into the arbiter.
pub trait MemReclaimer: Send + Sync {
    fn reclaim(&self, at_least: u64) -> u64;
    /// Whether the underlying cache still exists. Dead reclaimers (e.g. a
    /// dropped per-generate `Workspace`) are pruned on the next `register`.
    fn alive(&self) -> bool {
        true
    }
    /// Short label for trace output.
    fn label(&self) -> &'static str;
}

/// Single owner of one tier's memory budget. Cheap to share (`Arc`); all
/// state is the reclaimer chain. Current usage is read from the shared
/// [`MemAccount`] passed per call, so the arbiter composes with any backend.
pub struct MemArbiter {
    tier: MemTier,
    budget_bytes: u64,
    /// (priority, reclaimer), kept sorted ascending by priority.
    reclaimers: Mutex<Vec<(u32, Box<dyn MemReclaimer>)>>,
}

impl MemArbiter {
    pub fn new(tier: MemTier, budget_bytes: u64) -> Arc<Self> {
        Arc::new(Self {
            tier,
            budget_bytes,
            reclaimers: Mutex::new(Vec::new()),
        })
    }

    /// Gate disabled: unit tests that don't care about memory accounting.
    pub fn unlimited() -> Arc<Self> {
        Self::new(MemTier::Vram, u64::MAX)
    }

    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    fn tier_current(&self, mem: &MemAccount) -> u64 {
        match self.tier {
            MemTier::Vram => mem.vram_total_current(),
            MemTier::Ram => mem.ram_total_current(),
        }
    }

    /// Whether `bytes` more can be allocated without exceeding the budget.
    pub fn has_headroom(&self, mem: &MemAccount, bytes: u64) -> bool {
        self.budget_bytes == u64::MAX
            || self.tier_current(mem).saturating_add(bytes) <= self.budget_bytes
    }

    /// Add a reclaimer to the chain. Prunes dead entries first, so
    /// per-generate clients (workspace pools) can register each call without
    /// the chain growing.
    pub fn register(&self, priority: u32, r: Box<dyn MemReclaimer>) {
        let mut chain = self.reclaimers.lock().unwrap();
        chain.retain(|(_, r)| r.alive());
        chain.push((priority, r));
        chain.sort_by_key(|(p, _)| *p);
    }

    /// Make room for an upcoming allocation of `bytes`: if the tier's current
    /// total plus `bytes` exceeds the budget, run the reclaim chain until it
    /// fits. Best-effort; traces the overshoot if the chain runs dry.
    pub fn ensure_headroom(&self, mem: &MemAccount, bytes: u64) {
        if self.reclaim_until_fits(mem, bytes) {
            return;
        }
        tracing::info!(
            target: trace::ARBITER,
            op = "overshoot",
            bytes = bytes,
            over = (self.tier_current(mem).saturating_add(bytes)).saturating_sub(self.budget_bytes),
        );
    }

    /// Run the reclaim chain until `bytes` more fits under the budget. Returns
    /// `true` if it now fits (or the budget is disabled), `false` if the chain
    /// ran dry and the allocation would still overshoot. Callers that treat the
    /// budget as a HARD ceiling (e.g. the LTX VAE's adaptive tiler) use the
    /// `false` return to fail the alloc at the budget boundary -- so it shrinks
    /// and retries WITHOUT ever pushing the device into a real OOM (which on a
    /// long-lived process can poison the wgpu device). Soft callers
    /// (`ensure_headroom`) instead proceed and trace the overshoot.
    pub fn reclaim_until_fits(&self, mem: &MemAccount, bytes: u64) -> bool {
        if self.budget_bytes == u64::MAX {
            return true;
        }
        let over = |mem: &MemAccount| {
            (self.tier_current(mem).saturating_add(bytes)).saturating_sub(self.budget_bytes)
        };
        let mut want = over(mem);
        if want == 0 {
            return true;
        }
        let chain = self.reclaimers.lock().unwrap();
        for (_, r) in chain.iter() {
            let freed = r.reclaim(want);
            if freed > 0 {
                // Eviction traffic: debug per the level semantics (info is
                // stage milestones + rollups only).
                tracing::debug!(
                    target: trace::ARBITER,
                    op = "reclaim",
                    source = r.label(),
                    want = want,
                    freed = freed,
                );
            }
            want = over(mem);
            if want == 0 {
                return true;
            }
        }
        false
    }
}
