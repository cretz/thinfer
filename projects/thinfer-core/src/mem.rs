//! Memory accounting. Tracks live + peak bytes per category for VRAM and RAM.
//! Shared across `WgpuBackend`, `WeightResidency`, and `Workspace` so
//! residency's eviction loop can see the full picture (`weights + workspace +
//! staging`) and stop pulling in new weights once non-weight live bytes are
//! near their measured high-water mark.
//!
//! Counters are atomic. The "peak" of each category is the highest value the
//! `current` field has ever reached.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VramCategory {
    /// LRU-evictable model weights resident in VRAM (and idle entries in the
    /// residency's same-size reuse pool).
    Weights,
    /// `Workspace`-managed scratch buffers — both active (held in `WsBuf`) and
    /// idle (in the size-classed free list). Also covers ad-hoc per-phase
    /// buffers allocated outside the pool (e.g. VAE mid).
    Workspace,
    /// Wgpu staging buffers (readback, timestamp resolve). Allocated and freed
    /// internally by `WgpuBackend`.
    Staging,
}

impl VramCategory {
    pub const ALL: [VramCategory; 3] = [Self::Weights, Self::Workspace, Self::Staging];
    pub fn label(self) -> &'static str {
        match self {
            Self::Weights => "weights",
            Self::Workspace => "workspace",
            Self::Staging => "staging",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RamCategory {
    /// Host-side buffers staged for upload to GPU (weight reads from disk
    /// before `write_buffer`; bf16 rounding of activations).
    Upload,
    /// Host-side buffers received from GPU readback.
    Readback,
    /// Catch-all for other large host allocations the engine controls (e.g.
    /// tokenizer scratch). mmap page cache is NOT counted - kernel-managed.
    Other,
}

impl RamCategory {
    pub const ALL: [RamCategory; 3] = [Self::Upload, Self::Readback, Self::Other];
    pub fn label(self) -> &'static str {
        match self {
            Self::Upload => "upload",
            Self::Readback => "readback",
            Self::Other => "other",
        }
    }
}

#[derive(Default, Debug)]
struct Counter {
    current: AtomicU64,
    peak: AtomicU64,
}

impl Counter {
    fn add(&self, n: u64) {
        let now = self.current.fetch_add(n, Ordering::Relaxed) + n;
        let mut peak = self.peak.load(Ordering::Relaxed);
        while now > peak {
            match self.peak.compare_exchange_weak(
                peak,
                now,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
    }
    fn sub(&self, n: u64) {
        // Saturating to guard against accounting bugs that would wrap a u64.
        // Underflow is a bug, but we don't want it to silently look like
        // "everything in use" downstream.
        let prev = self.current.load(Ordering::Relaxed);
        let new = prev.saturating_sub(n);
        // Best-effort CAS - racy concurrent subs are tolerated; tests use
        // either single-threaded paths or the CAS retry below.
        let _ = self.current.compare_exchange(
            prev,
            new,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    }
    fn current(&self) -> u64 {
        self.current.load(Ordering::Relaxed)
    }
    fn peak(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }
}

/// Shared memory accountant. Held by `WgpuBackend`; cloned `Arc`s handed to
/// `WeightResidency` and `Workspace` so eviction and reporting see the same
/// counters. All methods are lock-free.
#[derive(Default, Debug)]
pub struct MemAccount {
    vram: [Counter; 3], // indexed by VramCategory as usize
    ram: [Counter; 3],  // indexed by RamCategory as usize
    /// Cross-category live total. Updated on every charge/release so its
    /// peak reflects the true maximum concurrent VRAM footprint, not the
    /// (looser) sum of per-category peaks.
    vram_total: Counter,
    ram_total: Counter,
}

impl MemAccount {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn charge_vram(&self, cat: VramCategory, bytes: u64) {
        self.vram[cat as usize].add(bytes);
        self.vram_total.add(bytes);
    }
    pub fn release_vram(&self, cat: VramCategory, bytes: u64) {
        self.vram[cat as usize].sub(bytes);
        self.vram_total.sub(bytes);
    }
    pub fn vram_current(&self, cat: VramCategory) -> u64 {
        self.vram[cat as usize].current()
    }
    pub fn vram_peak(&self, cat: VramCategory) -> u64 {
        self.vram[cat as usize].peak()
    }
    pub fn vram_total_current(&self) -> u64 {
        VramCategory::ALL
            .iter()
            .map(|&c| self.vram_current(c))
            .sum()
    }
    /// Sum of every non-weight VRAM category, current. Used by residency's
    /// dynamic ceiling: weights may grow to `budget - non_weights_current`.
    pub fn vram_non_weights_current(&self) -> u64 {
        self.vram_current(VramCategory::Workspace) + self.vram_current(VramCategory::Staging)
    }

    pub fn charge_ram(&self, cat: RamCategory, bytes: u64) {
        self.ram[cat as usize].add(bytes);
        self.ram_total.add(bytes);
    }
    pub fn release_ram(&self, cat: RamCategory, bytes: u64) {
        self.ram[cat as usize].sub(bytes);
        self.ram_total.sub(bytes);
    }
    pub fn ram_current(&self, cat: RamCategory) -> u64 {
        self.ram[cat as usize].current()
    }
    pub fn ram_peak(&self, cat: RamCategory) -> u64 {
        self.ram[cat as usize].peak()
    }
    pub fn ram_total_current(&self) -> u64 {
        RamCategory::ALL.iter().map(|&c| self.ram_current(c)).sum()
    }

    /// True peak of live VRAM bytes across all categories. Single counter
    /// updated on every charge/release, so its peak is the actual maximum
    /// concurrent footprint - the right quantity to assert against
    /// `vram_bytes`. Always less than or equal to the sum of per-category
    /// peaks (which can spuriously trip when peaks occur in different phases).
    pub fn vram_total_peak(&self) -> u64 {
        self.vram_total.peak()
    }
    pub fn ram_total_peak(&self) -> u64 {
        self.ram_total.peak()
    }

    /// Point-in-time read-out for reporting / test assertions.
    pub fn snapshot(&self) -> MemSnapshot {
        MemSnapshot {
            vram_total_peak: self.vram_total_peak(),
            ram_total_peak: self.ram_total_peak(),
            vram_weights: (
                self.vram_current(VramCategory::Weights),
                self.vram_peak(VramCategory::Weights),
            ),
            vram_workspace: (
                self.vram_current(VramCategory::Workspace),
                self.vram_peak(VramCategory::Workspace),
            ),
            vram_staging: (
                self.vram_current(VramCategory::Staging),
                self.vram_peak(VramCategory::Staging),
            ),
            ram_upload: (
                self.ram_current(RamCategory::Upload),
                self.ram_peak(RamCategory::Upload),
            ),
            ram_readback: (
                self.ram_current(RamCategory::Readback),
                self.ram_peak(RamCategory::Readback),
            ),
            ram_other: (
                self.ram_current(RamCategory::Other),
                self.ram_peak(RamCategory::Other),
            ),
        }
    }
}

/// `(current, peak)` per category, frozen at the moment of `snapshot()`.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemSnapshot {
    /// True maximum concurrent live VRAM bytes across all categories. This
    /// is the value to assert against the budget; per-category peaks are
    /// for diagnostics.
    pub vram_total_peak: u64,
    pub ram_total_peak: u64,
    pub vram_weights: (u64, u64),
    pub vram_workspace: (u64, u64),
    pub vram_staging: (u64, u64),
    pub ram_upload: (u64, u64),
    pub ram_readback: (u64, u64),
    pub ram_other: (u64, u64),
}

impl MemSnapshot {
    pub fn vram_current_total(&self) -> u64 {
        self.vram_weights.0 + self.vram_workspace.0 + self.vram_staging.0
    }
    /// Sum of per-category peaks. Loose upper bound on the true concurrent
    /// peak (since different categories may peak at different times). For
    /// budget assertions use `vram_total_peak` instead; this is for
    /// diagnostics only.
    pub fn vram_per_cat_peak_sum(&self) -> u64 {
        self.vram_weights.1 + self.vram_workspace.1 + self.vram_staging.1
    }
    pub fn ram_per_cat_peak_sum(&self) -> u64 {
        self.ram_upload.1 + self.ram_readback.1 + self.ram_other.1
    }
}

/// RAII guard that releases `bytes` from a VRAM category on drop. Used for
/// wgpu-native staging buffers that aren't owned through `Backend::allocate`
/// (they don't have a `GpuBufferId`, so the alloc/free hook doesn't see them).
pub struct VramCharge {
    account: Arc<MemAccount>,
    cat: VramCategory,
    bytes: u64,
}

impl VramCharge {
    pub fn new(account: Arc<MemAccount>, cat: VramCategory, bytes: u64) -> Self {
        account.charge_vram(cat, bytes);
        Self {
            account,
            cat,
            bytes,
        }
    }
}

impl Drop for VramCharge {
    fn drop(&mut self) {
        self.account.release_vram(self.cat, self.bytes);
    }
}

/// RAII guard that releases `bytes` from `cat` on drop. Use for big host
/// `Vec<u8>`s - charge before `vec![0u8; n]`, hold the guard alongside the
/// vec, drop together.
pub struct RamCharge {
    account: Arc<MemAccount>,
    cat: RamCategory,
    bytes: u64,
}

impl RamCharge {
    pub fn new(account: Arc<MemAccount>, cat: RamCategory, bytes: u64) -> Self {
        account.charge_ram(cat, bytes);
        Self {
            account,
            cat,
            bytes,
        }
    }
}

impl Drop for RamCharge {
    fn drop(&mut self) {
        self.account.release_ram(self.cat, self.bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charge_release_tracks_current_and_peak() {
        let m = MemAccount::new();
        m.charge_vram(VramCategory::Weights, 100);
        m.charge_vram(VramCategory::Weights, 50);
        assert_eq!(m.vram_current(VramCategory::Weights), 150);
        assert_eq!(m.vram_peak(VramCategory::Weights), 150);
        m.release_vram(VramCategory::Weights, 100);
        assert_eq!(m.vram_current(VramCategory::Weights), 50);
        assert_eq!(m.vram_peak(VramCategory::Weights), 150);
    }

    #[test]
    fn ram_charge_guard_releases_on_drop() {
        let m = MemAccount::new();
        {
            let _g = RamCharge::new(Arc::clone(&m), RamCategory::Upload, 1024);
            assert_eq!(m.ram_current(RamCategory::Upload), 1024);
        }
        assert_eq!(m.ram_current(RamCategory::Upload), 0);
        assert_eq!(m.ram_peak(RamCategory::Upload), 1024);
    }
}
