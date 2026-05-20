//! Weight residency manager. Pages weights disk -> VRAM on `acquire`, evicts
//! LRU on the GPU tier to keep VRAM total under `ResidencyBudget.vram_bytes`.
//! Returned `GpuView` pins the GPU buffer so eviction can't steal it
//! mid-forward.
//!
//! No host-side `Vec<u8>` mirror: the `WeightSource` is mmap-backed, so the OS
//! page cache is the host tier and `ResidencyBudget.ram_bytes` is enforced at
//! the upload-staging level, not here.
//!
//! Eviction predicate uses the shared `MemAccount` (workspace + staging
//! counters): weights are evicted until
//! `weights_current + needed + max(workspace_reserve, non_weights_current)
//! <= vram_bytes`. This is the "soft floor" that prevents weights from
//! filling the entire budget and forcing every workspace alloc into churn.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::backend::{Backend, BufRef};
use crate::mem::{RamCategory, RamCharge, VramCategory};
use crate::policy::ResidencyBudget;
use crate::tensor::{GpuBufferId, Shape, StorageEncoding};
use crate::weight::{DecodeError, WeightId, WeightReader, WeightSource};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightHandle(u32);

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
}

impl WeightMeta {
    pub fn elements(&self) -> u64 {
        self.shape.0.iter().map(|&d| d as u64).product()
    }

    /// Bytes the weight occupies on GPU, derived from `encoding`. Bf16 packs
    /// 2 elements per u32, padded up to u32 alignment; consumed via kernel
    /// `load_*` helpers.
    pub fn storage_bytes(&self) -> u64 {
        match self.encoding {
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
}

struct GpuEntry {
    id: GpuBufferId,
    bytes: u64,
    pin_count: u32,
}

struct Inner {
    metas: Vec<WeightMeta>,
    gpu: HashMap<WeightHandle, GpuEntry>,
    gpu_bytes: u64,
    gpu_lru: Vec<WeightHandle>, // back == MRU
    /// Size-class free list of GPU buffers reclaimed via eviction. Reused
    /// before calling `backend.allocate` so eviction churn doesn't pay wgpu's
    /// buffer-creation cost. Pytorch caching-allocator parity.
    ///
    /// Pool buffers are still wgpu-allocated and still charged to
    /// `VramCategory::Weights` in the mem account — they're "weights that
    /// just aren't bound to any handle right now". `pool_bytes` mirrors
    /// the sum, so the eviction predicate can include them when computing
    /// VRAM pressure (otherwise the pool grows unbounded under mixed
    /// size classes and the true peak overshoots the budget).
    pool: HashMap<u64, Vec<GpuBufferId>>,
    pool_bytes: u64,
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
    inner: Mutex<Inner>,
}

impl<S: WeightSource> WeightResidency<S> {
    pub fn new(source: S, budget: ResidencyBudget) -> Self {
        Self {
            source,
            budget,
            inner: Mutex::new(Inner {
                metas: Vec::new(),
                gpu: HashMap::new(),
                gpu_bytes: 0,
                gpu_lru: Vec::new(),
                pool: HashMap::new(),
                pool_bytes: 0,
            }),
        }
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

    /// Page the weight up to GPU. Returns a `GpuView` that pins the buffer for
    /// its lifetime. Drop the view to release the pin; subsequent eviction may
    /// reclaim the buffer.
    pub async fn acquire<'a, B: Backend>(
        &'a self,
        handle: WeightHandle,
        backend: &B,
    ) -> Result<GpuView<'a>, ResidencyError<S::Error, B::Error>> {
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
                handle,
                buf,
            });
        }

        // GPU miss. Stream from the mmap'd source directly into the bf16-packed
        // upload layout; no host-side cache. The temporary `Vec<u8>` lives only
        // until `write_buffer` returns - tracked as RAM Upload so the budget
        // sees this transient peak.
        let _upload_charge = RamCharge::new(
            backend.mem_account().clone(),
            RamCategory::Upload,
            meta.storage_bytes(),
        );
        let bytes = self.read_for_gpu::<B>(&meta).await?;

        let gpu_size = bytes.len() as u64;
        // Single weight must fit under the dynamic ceiling (full budget less
        // the workspace reserve). Evict everything else first wouldn't help if
        // even one weight is structurally too large.
        let single_ceiling = self
            .budget
            .vram_bytes
            .saturating_sub(self.budget.workspace_reserve);
        if gpu_size > single_ceiling {
            return Err(ResidencyError::BudgetTooSmall {
                needed: gpu_size,
                have: single_ceiling,
                tier: "vram",
            });
        }
        self.evict_gpu_until_fits::<B>(gpu_size, backend)?;
        let id = {
            let mut g = self.inner.lock().unwrap();
            let popped = g.pool.get_mut(&gpu_size).and_then(Vec::pop);
            if popped.is_some() {
                g.pool_bytes -= gpu_size;
            }
            popped
        };
        let id = match id {
            Some(id) => id,
            None => backend
                .allocate_in(gpu_size, VramCategory::Weights)
                .map_err(ResidencyError::Backend)?,
        };
        backend
            .write_buffer(id, 0, bytes.as_slice())
            .map_err(ResidencyError::Backend)?;

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
            handle,
            buf: BufRef::new(id, gpu_size),
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
        reader
            .read_at(0, &mut bytes[..on_disk_len])
            .await
            .map_err(|e| ResidencyError::Reader(format!("{e:?}")))?;

        match meta.transpose {
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
                // Block-tiled u16-stride transpose. Bf16 elements stored as
                // u16; transpose preserves the bit pattern. n*k must equal the
                // element count (padding only matters when odd, and 2-D linear
                // weights never have an odd product).
                const BLOCK: usize = 64;
                let elts = n * k;
                debug_assert_eq!(elts * 2, on_disk_len);
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
                Ok(dst)
            }
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

    /// Phase-boundary eviction: free every unpinned GPU-resident weight and
    /// drain the pool, releasing the VRAM back to wgpu. Caller's policy:
    /// invoke between pipeline phases (post text_encode, end of denoise) so
    /// peak VRAM is `max(phase)`, not the sum across phases. Pinned entries
    /// (mid-forward) are skipped; they stay resident.
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
        // Drain the within-phase reuse pool too; phase boundaries are when
        // we cash in VRAM, not when we hold extra slots warm.
        for (_, ids) in g.pool.drain() {
            for id in ids {
                backend.free(id);
            }
        }
        g.pool_bytes = 0;
    }

    /// LRU eviction under `vram_bytes` pressure. Evicted buffers are pushed
    /// onto a size-class free list in `Inner.pool`; the next miss of
    /// matching size reuses them. Under further pressure pool buffers are
    /// freed back to wgpu (the pool is a *cache*, not a reservation).
    /// Errors only if every remaining resident weight is pinned AND the
    /// pool is empty (caller's working set exceeds budget).
    ///
    /// Predicate: `gpu_bytes + pool_bytes + needed + non_weights_floor
    /// <= vram_bytes`, where `non_weights_floor =
    /// max(workspace_reserve, mem.vram_non_weights_current())`. Pool
    /// buffers count: they're still wgpu-allocated and still charged
    /// to the weights category in the mem account, so omitting them
    /// lets the true peak overshoot the budget when allocation traffic
    /// fragments across size classes (e.g. Q8 quant weights interleaved
    /// with bf16 fallback tensors).
    fn evict_gpu_until_fits<B: Backend>(
        &self,
        needed: u64,
        backend: &B,
    ) -> Result<(), ResidencyError<S::Error, B::Error>> {
        let mem = backend.mem_account();
        let mut g = self.inner.lock().unwrap();
        let mut idx = 0;
        loop {
            let non_weights_floor = self
                .budget
                .workspace_reserve
                .max(mem.vram_non_weights_current());
            let ceiling = self.budget.vram_bytes.saturating_sub(non_weights_floor);
            if g.gpu_bytes + g.pool_bytes + needed <= ceiling {
                break;
            }
            // Strategy: evict an unpinned LRU resident weight if any
            // remain; that buffer joins the pool so the next match
            // reuses it. If there's no unpinned resident left to
            // evict, free one pool buffer instead.
            let victim = loop {
                if idx >= g.gpu_lru.len() {
                    break None;
                }
                let cand = g.gpu_lru[idx];
                let pinned = g.gpu.get(&cand).is_some_and(|e| e.pin_count > 0);
                if pinned {
                    idx += 1;
                } else {
                    break Some(cand);
                }
            };
            if let Some(v) = victim {
                g.gpu_lru.remove(idx);
                if let Some(e) = g.gpu.remove(&v) {
                    g.gpu_bytes -= e.bytes;
                    g.pool_bytes += e.bytes;
                    g.pool.entry(e.bytes).or_default().push(e.id);
                }
                continue;
            }
            // No unpinned resident weight available; spill a pool entry
            // back to wgpu instead. Pool ordering doesn't matter — pop
            // from whichever size class has any entry. If even the pool
            // is empty, we can't make room.
            let freed = {
                let Some((&size, ids)) = g.pool.iter_mut().find(|(_, v)| !v.is_empty()) else {
                    return Err(ResidencyError::BudgetTooSmall {
                        needed,
                        have: ceiling.saturating_sub(g.gpu_bytes + g.pool_bytes),
                        tier: "vram",
                    });
                };
                let id = ids.pop().expect("filtered to non-empty");
                (size, id)
            };
            g.pool_bytes -= freed.0;
            backend.free(freed.1);
        }
        Ok(())
    }
}

/// Guard pinning a weight's GPU buffer for the view's lifetime. Drop releases
/// the pin; eviction can then reclaim the buffer on a later `acquire`.
pub struct GpuView<'a> {
    inner: &'a Mutex<Inner>,
    handle: WeightHandle,
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
        if let Some(e) = g.gpu.get_mut(&self.handle) {
            e.pin_count = e.pin_count.saturating_sub(1);
        }
    }
}
