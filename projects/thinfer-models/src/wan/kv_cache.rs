//! Windowed KV cache for LongLive-2.0-5B AR (causal/streaming) inference.
//!
//! LongLive denoises video one chunk (`num_frame_per_block` latent frames) at a
//! time; each chunk's self-attention reads a bounded sliding window of prior
//! chunks' keys/values instead of the full O(N) history. Causality is the cache
//! *contents* (the window only ever holds tokens at or before the current
//! chunk), so attention runs mode-0 (no materialized mask) over `q = current
//! chunk` vs `kv = window`. This module owns the window bookkeeping: the
//! roll/evict/insert math and the attended-window layout, ported faithfully
//! from upstream `wan_5b/modules/causal_model.py`
//! (`CausalWanSelfAttention.forward` + `CausalWanModel._apply_cache_updates`)
//! and `pipeline/causal_diffusion_inference.py` (cache init + pin/scene-cut).
//!
//! ## Why host-resident
//!
//! At real dims the cache is large: window `local_attn_size(32) *
//! frame_seq_len(880) = 28160` tokens, K and V each `[28160, inner=3072]` bf16
//! ~= 173 MB -> 346 MB/layer * 30 layers = ~10.4 GB. That cannot co-reside in
//! 8 GB VRAM with the (already residency-paged) ~10 GB of DiT weights. So the
//! backing store is host RAM ([`KvStore`], counted against
//! `ResidencyBudget.ram_bytes`); only the *active* block's window is uploaded
//! to GPU at attention time, paged in lockstep with that block's weights. The
//! [`KvStore`] trait is the seam for future disk offloading (a disk-backed impl
//! swaps in without touching the window logic here).
//!
//! ## Read-only during denoise, commit on the clean pass
//!
//! Within a chunk `current_start` is constant across the 4 UniPC steps + the
//! timestep-0 clean recache pass; every forward re-inserts the chunk's K/V at
//! the same tail slot (overwrite), and only the clean pass's K/V survive into
//! future chunks. So the committed window (prior chunks) is fixed for the whole
//! chunk: [`KvWindowCache::begin_chunk`] does the one-time roll/eviction and
//! returns the [`ChunkPlan`] (committed prefix segments + tail slot), the four
//! denoise steps attend over `prefix (host) ++ this-step chunk K/V (GPU)`
//! without writing back, and [`KvWindowCache::commit_chunk`] writes the clean
//! chunk's K/V into the tail exactly once.

use thinfer_core::policy::ResidencyBudget;

/// Host backing store for per-layer K/V bytes. One `[kv_cache_tokens *
/// token_bytes]` buffer per layer for K and another for V. This is the seam for
/// future disk offloading: a disk-backed impl swaps in without changing the
/// window/roll logic.
pub trait KvStore {
    fn k(&self, layer: usize) -> &[u8];
    fn v(&self, layer: usize) -> &[u8];
    fn k_mut(&mut self, layer: usize) -> &mut [u8];
    fn v_mut(&mut self, layer: usize) -> &mut [u8];
}

/// RAM-resident [`KvStore`]: per-layer `Vec<u8>` on the JS heap (web) / native
/// `Vec` (never WASM linear memory). Zero-initialized to match upstream
/// `torch.zeros` cache init.
#[derive(Debug)]
pub struct RamKvStore {
    k: Vec<Vec<u8>>,
    v: Vec<Vec<u8>>,
}

impl RamKvStore {
    pub fn new(num_layers: usize, bytes_per_layer: usize) -> Self {
        Self {
            k: (0..num_layers)
                .map(|_| vec![0u8; bytes_per_layer])
                .collect(),
            v: (0..num_layers)
                .map(|_| vec![0u8; bytes_per_layer])
                .collect(),
        }
    }

    /// Total host bytes held (K + V across all layers).
    pub fn ram_bytes(&self) -> u64 {
        self.k.iter().map(|b| b.len() as u64).sum::<u64>()
            + self.v.iter().map(|b| b.len() as u64).sum::<u64>()
    }
}

impl KvStore for RamKvStore {
    fn k(&self, layer: usize) -> &[u8] {
        &self.k[layer]
    }
    fn v(&self, layer: usize) -> &[u8] {
        &self.v[layer]
    }
    fn k_mut(&mut self, layer: usize) -> &mut [u8] {
        &mut self.k[layer]
    }
    fn v_mut(&mut self, layer: usize) -> &mut [u8] {
        &mut self.v[layer]
    }
}

/// Geometry of the windowed cache. All sizes that are "in frames" upstream are
/// kept in frames here and converted to tokens via `frame_seq_len`.
#[derive(Clone, Copy, Debug)]
pub struct KvCacheConfig {
    pub num_layers: usize,
    /// Tokens per latent frame (`pph * ppw`). 880 at the LongLive release res.
    pub frame_seq_len: usize,
    /// AR chunk size in latent frames (`num_frame_per_block`, 8).
    pub num_frame_per_block: usize,
    /// Sliding-window size in frames (`local_attn_size`, 32). `usize::MAX` would
    /// be "global" but LongLive is always local; we require a finite window.
    pub local_attn_size: usize,
    /// Per-shot attention-sink frames kept on a scene cut (`sink_size`, 8).
    pub sink_size: usize,
    /// Permanently anchored leading frames (`global_sink_size`): `sink_size`
    /// when `multi_shot_sink` else 0.
    pub global_sink_size: usize,
    /// Per-shot RoPE temporal phase offset (`multi_shot_rope_offset`, 8.0).
    pub multi_shot_rope_offset: f32,
    /// Byte width of one token's K (or V) row = `inner * act_dtype_bytes`.
    pub token_bytes: usize,
}

impl KvCacheConfig {
    /// LongLive-2.0-5B (`configs/inference.yaml`): 30 layers, 880 tokens/frame,
    /// chunk 8, window 32, sink 8, multi-shot sink on (global_sink 8), RoPE
    /// offset 8. `inner`/`act_dtype_bytes` come from the runtime DiT shape.
    pub fn longlive_2_0_5b(num_layers: usize, inner: usize, act_dtype_bytes: usize) -> Self {
        Self {
            num_layers,
            frame_seq_len: 880,
            num_frame_per_block: 8,
            local_attn_size: 32,
            sink_size: 8,
            global_sink_size: 8,
            multi_shot_rope_offset: 8.0,
            token_bytes: inner * act_dtype_bytes,
        }
    }

    /// LongLive geometry with a RUNTIME `frame_seq_len` (`pph * ppw`, i.e.
    /// `h_lat/2 * w_lat/2`). The release `frame_seq_len` is 880 (44x80/4) but the
    /// e2e gate runs smaller grids, so the cache must size to the actual latent.
    /// Window/sink/chunk stay in frames (config-fixed); only the per-frame token
    /// count and the K/V byte width vary with resolution.
    pub fn longlive_runtime(
        num_layers: usize,
        frame_seq_len: usize,
        inner: usize,
        act_dtype_bytes: usize,
    ) -> Self {
        Self {
            frame_seq_len,
            ..Self::longlive_2_0_5b(num_layers, inner, act_dtype_bytes)
        }
    }

    /// AR chunk size in frames (`num_frame_per_block`).
    pub fn chunk_frames(&self) -> usize {
        self.num_frame_per_block
    }

    /// Window capacity in tokens (`local_attn_size * frame_seq_len`). The cache
    /// buffer and the max attention span are both this size in the local path.
    pub fn kv_cache_tokens(&self) -> usize {
        self.local_attn_size * self.frame_seq_len
    }

    /// Host bytes for one layer's K (or V) buffer.
    pub fn bytes_per_layer(&self) -> usize {
        self.kv_cache_tokens() * self.token_bytes
    }

    /// Total host bytes for the whole cache (K + V, all layers).
    pub fn total_ram_bytes(&self) -> u64 {
        2 * self.num_layers as u64 * self.bytes_per_layer() as u64
    }
}

/// One contiguous token range `[start, start + len)` in a per-layer buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seg {
    pub start: usize,
    pub len: usize,
}

impl Seg {
    /// Byte range for a buffer with `token_bytes`-wide rows.
    pub fn byte_range(&self, token_bytes: usize) -> core::ops::Range<usize> {
        self.start * token_bytes..(self.start + self.len) * token_bytes
    }
}

/// Plan for one AR chunk, produced by [`KvWindowCache::begin_chunk`] and reused
/// across the chunk's denoise steps + clean pass.
#[derive(Clone, Debug)]
pub struct ChunkPlan {
    /// Committed (host) token segments the chunk attends to, in order, EXCLUDING
    /// the current chunk. Uploaded per layer per forward straight into the GPU
    /// window buffers (holding all layers' windows resident across a chunk's
    /// forwards would need `2 * num_layers * prefix_bytes` VRAM, over the whole
    /// card at release geometry); the freshly computed chunk K/V (the tail)
    /// concatenates after them.
    pub prefix: Vec<Seg>,
    /// Tail slot for this chunk's K/V in the committed buffer
    /// (`[local_start_index, local_end_index)`). [`KvWindowCache::commit_chunk`]
    /// writes the clean-pass K/V here.
    pub tail: Seg,
    /// `prefix_tokens + tail.len`: total tokens the window attends over.
    pub window_tokens: usize,
    /// Sum of `prefix` segment lengths (== `window_tokens - tail.len`).
    pub prefix_tokens: usize,
    /// Absolute temporal RoPE start frame for the current chunk (`current_start /
    /// frame_seq_len`). In the release `use_relative_rope=False` path both the
    /// query and the chunk's key are RoPE'd at this absolute frame position
    /// (cached prefix keys were already RoPE'd at their own absolute positions
    /// when committed, so the prefix needs no re-rotation).
    pub chunk_start_frame: usize,
    /// Multi-shot RoPE temporal offset for this chunk (`shot_index *
    /// multi_shot_rope_offset`). `multi_shot_rope_offset` is integer (8), so this
    /// is an integer frame shift added to the chunk's temporal positions.
    pub temporal_offset: f32,
}

/// Cache index bookkeeping, shared across layers (every layer receives identical
/// updates upstream, which reads `kv_cache[0]`). Signed to mirror the upstream
/// `pinned_start == -1` sentinel and intermediate index arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IndexState {
    /// Global (unbounded, video-absolute) end token of committed context.
    global_end_index: i64,
    /// Buffer-local (bounded by `kv_cache_tokens`) end token of committed data.
    local_end_index: i64,
    /// Buffer-local start of a pinned multi-shot sink region, or -1 if none.
    pinned_start: i64,
    /// Length of the pinned region in tokens.
    pinned_len: i64,
}

impl IndexState {
    fn reset() -> Self {
        Self {
            global_end_index: 0,
            local_end_index: 0,
            pinned_start: -1,
            pinned_len: 0,
        }
    }
}

/// Error sizing the cache against the RAM budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RamBudgetExceeded {
    pub need: u64,
    pub have: u64,
}

impl core::fmt::Display for RamBudgetExceeded {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "KV cache needs {} host bytes but only {} of the RAM budget remain",
            self.need, self.have
        )
    }
}

impl std::error::Error for RamBudgetExceeded {}

/// Windowed KV cache: the index bookkeeping + roll/window planning. The byte
/// buffers live in a separate [`KvStore`] so the host tier is swappable
/// (RAM now, disk later). One instance drives all layers; per-layer byte moves
/// are applied through the store.
pub struct KvWindowCache {
    cfg: KvCacheConfig,
    idx: IndexState,
    /// Current shot index (incremented on shot boundaries) for the RoPE offset.
    shot_index: usize,
    /// Cached `temporal_offset = shot_index * multi_shot_rope_offset`.
    temporal_offset: f32,
}

impl KvWindowCache {
    pub fn new(cfg: KvCacheConfig) -> Self {
        assert!(cfg.local_attn_size > 0, "LongLive requires a finite window");
        assert!(
            cfg.global_sink_size <= cfg.local_attn_size,
            "global sink ({}) cannot exceed the window ({})",
            cfg.global_sink_size,
            cfg.local_attn_size
        );
        Self {
            cfg,
            idx: IndexState::reset(),
            shot_index: 0,
            temporal_offset: 0.0,
        }
    }

    pub fn config(&self) -> &KvCacheConfig {
        &self.cfg
    }

    /// Allocate the RAM-resident store, checking it fits `ram_available` (the
    /// RAM budget remaining after weights/other host buffers).
    pub fn alloc_ram_store(&self, ram_available: u64) -> Result<RamKvStore, RamBudgetExceeded> {
        let need = self.cfg.total_ram_bytes();
        if need > ram_available {
            return Err(RamBudgetExceeded {
                need,
                have: ram_available,
            });
        }
        Ok(RamKvStore::new(
            self.cfg.num_layers,
            self.cfg.bytes_per_layer(),
        ))
    }

    /// Convenience: size against `budget.ram_bytes` directly. The caller that
    /// also pays for weights out of the same budget should prefer
    /// [`Self::alloc_ram_store`] with the remaining bytes.
    pub fn alloc_ram_store_in_budget(
        &self,
        budget: ResidencyBudget,
    ) -> Result<RamKvStore, RamBudgetExceeded> {
        self.alloc_ram_store(budget.ram_bytes)
    }

    /// Reset to an empty cache for a fresh generation. Does not touch the store
    /// bytes (they are overwritten as chunks commit; stale tail bytes are never
    /// read because the window never extends past `local_end_index`).
    pub fn reset(&mut self) {
        self.idx = IndexState::reset();
        self.shot_index = 0;
        self.temporal_offset = 0.0;
    }

    /// Advance the multi-shot RoPE phase on a shot boundary (call before
    /// [`Self::begin_chunk`] for the first chunk of a new shot).
    pub fn advance_shot(&mut self) {
        self.shot_index += 1;
        self.temporal_offset = self.shot_index as f32 * self.cfg.multi_shot_rope_offset;
    }

    fn frame_seq(&self) -> i64 {
        self.cfg.frame_seq_len as i64
    }

    fn kv_cache_tokens(&self) -> i64 {
        self.cfg.kv_cache_tokens() as i64
    }

    /// Tokens permanently protected at the buffer front during a roll.
    /// Mirrors `effective_sink` (upstream lines 528-534): with a pinned region
    /// merged at the front it is `global + pinned`; otherwise `max(global,
    /// per-shot sink)`.
    fn effective_sink(&self) -> i64 {
        let global = self.cfg.global_sink_size as i64 * self.frame_seq();
        let has_pinned = self.idx.pinned_start >= 0 && self.idx.pinned_len > 0;
        if has_pinned && self.idx.pinned_start == global {
            global + self.idx.pinned_len
        } else if has_pinned {
            global
        } else {
            let shot = self.cfg.sink_size as i64 * self.frame_seq();
            global.max(shot)
        }
    }

    /// Begin an AR chunk at absolute token `current_start` with `num_new_frames`
    /// frames. Applies the one-time roll/eviction to the committed store, sets
    /// the new indices, and returns the [`ChunkPlan`] reused across the chunk's
    /// denoise steps + clean pass.
    ///
    /// Mirrors the `current_end > _cache_global_end` advancing path of
    /// `CausalWanSelfAttention.forward` (upstream is always on this path at a
    /// new chunk's first forward) plus the `roll_and_insert`/`direct_insert`
    /// byte moves of `_apply_cache_updates`, minus the tail insert (deferred to
    /// [`Self::commit_chunk`]).
    pub fn begin_chunk<S: KvStore>(
        &mut self,
        store: &mut S,
        current_start: usize,
        num_new_frames: usize,
    ) -> ChunkPlan {
        let fs = self.frame_seq();
        let num_new_tokens = num_new_frames as i64 * fs;
        let current_start = current_start as i64;
        let current_end = current_start + num_new_tokens;
        let kv_size = self.kv_cache_tokens();
        let local_end = self.idx.local_end_index;
        let global_end = self.idx.global_end_index;
        debug_assert!(
            current_end > global_end,
            "begin_chunk expects an advancing chunk (current_end {current_end} > global_end {global_end})"
        );
        debug_assert!(
            num_new_tokens <= kv_size,
            "chunk ({num_new_tokens} tokens) larger than the window ({kv_size})"
        );

        let effective_sink = self.effective_sink();
        let needs_roll = num_new_tokens + local_end > kv_size;

        let (local_start_index, local_end_index) = if needs_roll {
            let num_evicted = num_new_tokens + local_end - kv_size;
            let num_rolled = local_end - num_evicted - effective_sink;
            debug_assert!(
                num_rolled >= 0,
                "roll underflow: local_end {local_end} - evicted {num_evicted} - sink {effective_sink}"
            );
            let local_end_index = local_end + current_end - global_end - num_evicted;
            let local_start_index = local_end_index - num_new_tokens;

            // Roll committed tokens left, preserving the protected sink prefix:
            // [sink + evicted, sink + evicted + rolled) -> [sink, sink + rolled).
            let sink = effective_sink as usize;
            let evicted = num_evicted as usize;
            let rolled = num_rolled as usize;
            let tb = self.cfg.token_bytes;
            for layer in 0..self.cfg.num_layers {
                roll_left(store.k_mut(layer), sink, evicted, rolled, tb);
                roll_left(store.v_mut(layer), sink, evicted, rolled, tb);
            }

            // A floating pinned region (outside the protected prefix) tracks the
            // same data after the left shift (upstream `pinned_shift`).
            let has_pinned = self.idx.pinned_start >= 0 && self.idx.pinned_len > 0;
            if has_pinned && self.idx.pinned_start >= effective_sink {
                self.idx.pinned_start -= num_evicted;
            }

            (local_start_index, local_end_index)
        } else {
            let local_end_index = local_end + current_end - global_end;
            let local_start_index = local_end_index - num_new_tokens;
            (local_start_index, local_end_index)
        };

        self.idx.global_end_index = current_end;
        self.idx.local_end_index = local_end_index;

        let tail = Seg {
            start: local_start_index as usize,
            len: num_new_tokens as usize,
        };
        let (prefix, prefix_tokens) = self.plan_prefix(local_end_index, tail);

        // RoPE positions (release `use_relative_rope=False` path): q and the
        // chunk's k rotate at the chunk's ABSOLUTE frame position. The committed
        // prefix keys were already RoPE'd at their own absolute positions when
        // committed, so no virtual relayout / prefix re-rotation is needed.
        let chunk_start_frame = (current_start / fs) as usize;

        ChunkPlan {
            prefix,
            tail,
            window_tokens: prefix_tokens + tail.len,
            prefix_tokens,
            chunk_start_frame,
            temporal_offset: self.temporal_offset,
        }
    }

    /// Build the committed (host) window segments that precede the current
    /// chunk. Mirrors the attended-window construction (upstream lines 673-724)
    /// then strips the trailing `tail` (the current chunk, supplied fresh from
    /// GPU rather than read from the committed store).
    fn plan_prefix(&self, local_end_index: i64, tail: Seg) -> (Vec<Seg>, usize) {
        let max_attn = self.kv_cache_tokens();
        let effective_sink = self.effective_sink();
        let window_start = (local_end_index - max_attn).max(0);
        let has_pinned = self.idx.pinned_start >= 0 && self.idx.pinned_len > 0;
        let pinned_start = self.idx.pinned_start;
        let pinned_len = self.idx.pinned_len;

        let prepend_sink = effective_sink > 0 && window_start > 0;
        let prepend_pinned =
            has_pinned && pinned_start >= effective_sink && pinned_start < window_start;

        // Full window segments [start, len) over the committed buffer, in order.
        let mut window: Vec<Seg> = Vec::new();
        let mut push = |start: i64, end: i64| {
            if end > start {
                window.push(Seg {
                    start: start as usize,
                    len: (end - start) as usize,
                });
            }
        };
        if prepend_sink && prepend_pinned {
            let extra = effective_sink + pinned_len;
            let effective_local = max_attn - extra;
            let local_window_start = effective_sink.max(local_end_index - effective_local);
            push(0, effective_sink);
            push(pinned_start, pinned_start + pinned_len);
            push(local_window_start, local_end_index);
        } else if prepend_sink {
            let effective_local = max_attn - effective_sink;
            let local_window_start = effective_sink.max(local_end_index - effective_local);
            push(0, effective_sink);
            push(local_window_start, local_end_index);
        } else if prepend_pinned {
            let effective_local = max_attn - pinned_len;
            let local_window_start = (local_end_index - effective_local).max(0);
            push(pinned_start, pinned_start + pinned_len);
            push(local_window_start, local_end_index);
        } else {
            push(window_start, local_end_index);
        }

        // Strip the trailing `tail` tokens (the current chunk). The tail always
        // sits at the very end of the last segment.
        strip_tail(&mut window, tail.len);
        let prefix_tokens: usize = window.iter().map(|s| s.len).sum();
        (window, prefix_tokens)
    }

    /// Commit the clean-pass K/V for the current chunk into the tail slot. K is
    /// stored already-RoPE'd (upstream `use_relative_rope=False`:
    /// `key_to_cache = roped_key`). `k_bytes`/`v_bytes` are `tail.len *
    /// token_bytes` each.
    pub fn commit_chunk<S: KvStore>(
        &self,
        store: &mut S,
        plan: &ChunkPlan,
        k_bytes: &[&[u8]],
        v_bytes: &[&[u8]],
    ) {
        let tb = self.cfg.token_bytes;
        let range = plan.tail.byte_range(tb);
        debug_assert_eq!(k_bytes.len(), self.cfg.num_layers);
        debug_assert_eq!(v_bytes.len(), self.cfg.num_layers);
        for layer in 0..self.cfg.num_layers {
            store.k_mut(layer)[range.clone()].copy_from_slice(k_bytes[layer]);
            store.v_mut(layer)[range.clone()].copy_from_slice(v_bytes[layer]);
        }
    }

    /// Pin the just-committed chunk as the shot sink for multi-shot (upstream
    /// `_pin_current_chunk`): mark its buffer position so the next roll keeps it
    /// and relocates rolling data around it. No bytes move here. Call after
    /// [`Self::commit_chunk`] on a scene cut.
    pub fn pin_current_chunk(&mut self, num_new_frames: usize) {
        let fs = self.frame_seq();
        let chunk_tokens = num_new_frames as i64 * fs;
        let pin_len = (self.cfg.sink_size as i64 * fs).min(chunk_tokens);
        let chunk_start = self.idx.local_end_index - chunk_tokens;
        self.idx.pinned_start = chunk_start;
        self.idx.pinned_len = pin_len;
    }

    /// Reset the cache to the global sink for a clean recache on a scene cut
    /// (upstream `_zero_kv_data`): keep the global sink prefix, drop everything
    /// else, re-anchor the global token cursor. `current_start` is the new
    /// chunk's absolute start token.
    pub fn zero_for_scene_cut(&mut self, current_start: usize) {
        let global_sink_tokens = self.cfg.global_sink_size as i64 * self.frame_seq();
        self.idx.local_end_index = global_sink_tokens;
        self.idx.global_end_index = current_start as i64;
        self.idx.pinned_start = -1;
        self.idx.pinned_len = 0;
    }
}

/// Shift `[sink + evicted, sink + evicted + rolled)` down to `[sink, sink +
/// rolled)` in a `token_bytes`-strided buffer (overlapping move toward the
/// front; `copy_within` handles the overlap).
fn roll_left(buf: &mut [u8], sink: usize, evicted: usize, rolled: usize, token_bytes: usize) {
    if rolled == 0 || evicted == 0 {
        return;
    }
    let src = (sink + evicted) * token_bytes..(sink + evicted + rolled) * token_bytes;
    let dst = sink * token_bytes;
    buf.copy_within(src, dst);
}

/// Remove the trailing `n` tokens from an ordered segment list (they sit at the
/// end of the final segment).
fn strip_tail(window: &mut Vec<Seg>, mut n: usize) {
    while n > 0 {
        let last = window.last_mut().expect("window shorter than the tail");
        if last.len > n {
            last.len -= n;
            n = 0;
        } else {
            n -= last.len;
            window.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test geometry: tiny frames so a u32 per token tags its global index.
    /// token_bytes = 4 holds one u32 "global token id" per token slot.
    fn cfg(local_attn: usize, sink: usize, global_sink: usize, fpb: usize) -> KvCacheConfig {
        KvCacheConfig {
            num_layers: 2,
            frame_seq_len: 2,
            num_frame_per_block: fpb,
            local_attn_size: local_attn,
            sink_size: sink,
            global_sink_size: global_sink,
            multi_shot_rope_offset: 8.0,
            token_bytes: 4,
        }
    }

    fn tag_bytes(ids: &[i64]) -> Vec<u8> {
        ids.iter().flat_map(|&i| (i as u32).to_le_bytes()).collect()
    }

    fn read_tag(buf: &[u8], seg: Seg) -> Vec<i64> {
        (seg.start..seg.start + seg.len)
            .map(|t| {
                let b = &buf[t * 4..t * 4 + 4];
                u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64
            })
            .collect()
    }

    /// Gather the global-token-ids the committed prefix segments point at.
    fn gathered_prefix(store: &RamKvStore, plan: &ChunkPlan) -> Vec<i64> {
        plan.prefix
            .iter()
            .flat_map(|&seg| read_tag(store.k(0), seg))
            .collect()
    }

    /// Independent reference: a frame deque, capacity `local_attn` frames, with
    /// `effective_sink` leading frames protected. Returns the prior-context
    /// frame ids the next chunk should attend to (excludes the new chunk).
    struct DequeRef {
        cap_frames: usize,
        sink_frames: usize,
        frames: Vec<i64>, // committed frame ids (global)
    }
    impl DequeRef {
        fn new(cap_frames: usize, sink_frames: usize) -> Self {
            Self {
                cap_frames,
                sink_frames,
                frames: Vec::new(),
            }
        }
        /// Evict for a new chunk of `nf` frames; return the surviving prior
        /// frame ids (the window prefix). Then append the chunk's ids.
        fn step(&mut self, first_frame: i64, nf: usize) -> Vec<i64> {
            let need = self.frames.len() + nf;
            if need > self.cap_frames {
                let evict = need - self.cap_frames;
                // Remove `evict` frames after the protected sink prefix.
                self.frames
                    .drain(self.sink_frames..self.sink_frames + evict);
            }
            let prefix = self.frames.clone();
            for j in 0..nf {
                self.frames.push(first_frame + j as i64);
            }
            prefix
        }
    }

    /// Drive a full video through the cache and assert the committed prefix
    /// each chunk matches the deque reference, for several geometries.
    #[test]
    fn window_matches_deque_reference() {
        for &(local_attn, sink, fpb, nblocks) in &[
            (4usize, 1usize, 1usize, 8usize),
            (4, 2, 2, 6),
            (8, 2, 4, 7),
            (6, 3, 1, 12),
        ] {
            let c = cfg(local_attn, sink, sink, fpb);
            let fs = c.frame_seq_len;
            let mut cache = KvWindowCache::new(c);
            let mut store = RamKvStore::new(c.num_layers, c.bytes_per_layer());
            let mut dref = DequeRef::new(local_attn, sink);

            let mut current_start = 0usize;
            let mut first_frame = 0i64;
            for _ in 0..nblocks {
                let plan = cache.begin_chunk(&mut store, current_start, fpb);
                let expect_prefix_frames = dref.step(first_frame, fpb);
                let expect_prefix_tokens: Vec<i64> = expect_prefix_frames
                    .iter()
                    .flat_map(|&f| (0..fs as i64).map(move |t| f * fs as i64 + t))
                    .collect();

                // The committed prefix must point at exactly the reference's
                // prior-context tokens, in order.
                assert_eq!(
                    gathered_prefix(&store, &plan),
                    expect_prefix_tokens,
                    "local_attn={local_attn} sink={sink} fpb={fpb} start={current_start}"
                );
                assert_eq!(plan.prefix_tokens, expect_prefix_tokens.len());
                assert_eq!(plan.window_tokens, plan.prefix_tokens + plan.tail.len);
                assert_eq!(plan.tail.len, fpb * fs);

                // Commit the chunk's clean K/V (tag each token by global id).
                let chunk_ids: Vec<i64> = (0..(fpb * fs) as i64)
                    .map(|t| current_start as i64 + t)
                    .collect();
                let kb = tag_bytes(&chunk_ids);
                let vb = tag_bytes(&chunk_ids);
                let k_refs: Vec<&[u8]> = (0..c.num_layers).map(|_| kb.as_slice()).collect();
                let v_refs: Vec<&[u8]> = (0..c.num_layers).map(|_| vb.as_slice()).collect();
                cache.commit_chunk(&mut store, &plan, &k_refs, &v_refs);

                current_start += fpb * fs;
                first_frame += fpb as i64;
            }
        }
    }

    #[test]
    fn first_chunk_attends_only_itself() {
        let c = cfg(8, 2, 2, 4);
        let mut cache = KvWindowCache::new(c);
        let mut store = RamKvStore::new(c.num_layers, c.bytes_per_layer());
        let plan = cache.begin_chunk(&mut store, 0, 4);
        assert!(plan.prefix.is_empty());
        assert_eq!(plan.prefix_tokens, 0);
        assert_eq!(
            plan.tail,
            Seg {
                start: 0,
                len: 4 * c.frame_seq_len
            }
        );
        assert_eq!(plan.chunk_start_frame, 0);
    }

    #[test]
    fn sink_frames_survive_eviction() {
        // Window 4 frames, sink 1, chunk 1 -> after >4 chunks the first frame
        // (global sink) must remain in every later prefix.
        let c = cfg(4, 1, 1, 1);
        let fs = c.frame_seq_len as i64;
        let mut cache = KvWindowCache::new(c);
        let mut store = RamKvStore::new(c.num_layers, c.bytes_per_layer());
        let mut current_start = 0usize;
        for chunk in 0..10 {
            let plan = cache.begin_chunk(&mut store, current_start, 1);
            if chunk >= 4 {
                // Frame 0's two tokens lead the prefix and are still present.
                let g = gathered_prefix(&store, &plan);
                assert_eq!(&g[..2], &[0i64, 1], "sink lost at chunk {chunk}: {g:?}");
            }
            let ids: Vec<i64> = (0..fs).map(|t| current_start as i64 + t).collect();
            let b = tag_bytes(&ids);
            let r: Vec<&[u8]> = (0..c.num_layers).map(|_| b.as_slice()).collect();
            cache.commit_chunk(&mut store, &plan, &r, &r);
            current_start += c.frame_seq_len;
        }
    }

    #[test]
    fn chunk_start_frame_is_absolute() {
        let c = cfg(8, 2, 2, 2);
        let mut cache = KvWindowCache::new(c);
        let mut store = RamKvStore::new(c.num_layers, c.bytes_per_layer());
        let mut current_start = 0usize;
        // Absolute frame position grows monotonically (current_start / fs),
        // independent of window saturation (release use_relative_rope=False).
        for expect_q in [0usize, 2, 4, 6, 8] {
            let plan = cache.begin_chunk(&mut store, current_start, 2);
            assert_eq!(plan.chunk_start_frame, expect_q, "start={current_start}");
            let ids: Vec<i64> = (0..(2 * c.frame_seq_len) as i64)
                .map(|t| current_start as i64 + t)
                .collect();
            let b = tag_bytes(&ids);
            let r: Vec<&[u8]> = (0..c.num_layers).map(|_| b.as_slice()).collect();
            cache.commit_chunk(&mut store, &plan, &r, &r);
            current_start += 2 * c.frame_seq_len;
        }
    }

    #[test]
    fn ram_budget_sizing() {
        // LongLive real geometry: ~10.4 GB host for the full cache.
        let c = KvCacheConfig::longlive_2_0_5b(30, 3072, 2);
        assert_eq!(c.kv_cache_tokens(), 32 * 880);
        let per_layer = 32 * 880 * 3072 * 2;
        assert_eq!(c.bytes_per_layer(), per_layer);
        assert_eq!(c.total_ram_bytes(), 2 * 30 * per_layer as u64);

        let cache = KvWindowCache::new(c);
        // Under budget: allocates.
        let ok = cache.alloc_ram_store(c.total_ram_bytes());
        assert!(ok.is_ok());
        assert_eq!(ok.unwrap().ram_bytes(), c.total_ram_bytes());
        // Over budget: rejected with the shortfall surfaced.
        let err = cache.alloc_ram_store(c.total_ram_bytes() - 1).unwrap_err();
        assert_eq!(err.need, c.total_ram_bytes());
        assert_eq!(err.have, c.total_ram_bytes() - 1);
    }

    #[test]
    fn scene_cut_zero_keeps_global_sink() {
        let c = cfg(8, 2, 2, 2);
        let fs = c.frame_seq_len as i64;
        let mut cache = KvWindowCache::new(c);
        let mut store = RamKvStore::new(c.num_layers, c.bytes_per_layer());
        // Fill a few chunks.
        let mut current_start = 0usize;
        for _ in 0..3 {
            let plan = cache.begin_chunk(&mut store, current_start, 2);
            let ids: Vec<i64> = (0..(2 * c.frame_seq_len) as i64)
                .map(|t| current_start as i64 + t)
                .collect();
            let b = tag_bytes(&ids);
            let r: Vec<&[u8]> = (0..c.num_layers).map(|_| b.as_slice()).collect();
            cache.commit_chunk(&mut store, &plan, &r, &r);
            current_start += 2 * c.frame_seq_len;
        }
        // Scene cut: local cursor drops to the global sink, global cursor jumps.
        cache.zero_for_scene_cut(current_start);
        assert_eq!(cache.idx.local_end_index, c.global_sink_size as i64 * fs);
        assert_eq!(cache.idx.global_end_index, current_start as i64);
        assert_eq!(cache.idx.pinned_start, -1);

        // Next chunk's prefix == the retained global-sink tokens only.
        let plan = cache.begin_chunk(&mut store, current_start, 2);
        assert_eq!(plan.prefix_tokens, c.global_sink_size * c.frame_seq_len);
    }

    /// A pinned multi-shot sink survives evictions that would otherwise drop it.
    /// Window 4 frames, global sink 1, chunk 1: pin frame 2 at a scene cut, then
    /// roll far enough that (without the pin) frame 2 would be gone, and assert it
    /// is still in the attended prefix while a non-pinned older frame is not.
    #[test]
    fn pinned_chunk_survives_eviction() {
        let c = cfg(4, 1, 1, 1);
        let fs = c.frame_seq_len as i64;
        let mut cache = KvWindowCache::new(c);
        let mut store = RamKvStore::new(c.num_layers, c.bytes_per_layer());

        let commit =
            |cache: &KvWindowCache, store: &mut RamKvStore, plan: &ChunkPlan, start: usize| {
                let ids: Vec<i64> = (0..fs).map(|t| start as i64 + t).collect();
                let b = tag_bytes(&ids);
                let r: Vec<&[u8]> = (0..c.num_layers).map(|_| b.as_slice()).collect();
                cache.commit_chunk(store, plan, &r, &r);
            };

        let mut current_start = 0usize;
        // Chunks 0..3 fill the window; chunk 2 is the scene cut whose chunk we pin.
        for chunk in 0..3 {
            if chunk == 2 {
                cache.advance_shot();
            }
            let plan = cache.begin_chunk(&mut store, current_start, 1);
            commit(&cache, &mut store, &plan, current_start);
            if chunk == 2 {
                cache.pin_current_chunk(1);
            }
            current_start += c.frame_seq_len;
        }

        // Roll well past where frame 2 would normally age out (window 4 frames).
        for _ in 3..9 {
            let plan = cache.begin_chunk(&mut store, current_start, 1);
            let g = gathered_prefix(&store, &plan);
            // Frame 2's tokens (global ids [4, 5]) stay pinned in the prefix...
            assert!(
                g.windows(2).any(|w| w == [4i64, 5]),
                "pinned frame 2 evicted at start={current_start}: {g:?}"
            );
            // ...while a non-pinned older frame (frame 3, ids [6, 7]) is dropped
            // once the rolling window has advanced past it.
            if current_start as i64 / fs >= 8 {
                assert!(
                    !g.windows(2).any(|w| w == [6i64, 7]),
                    "non-pinned frame 3 should have aged out at start={current_start}: {g:?}"
                );
            }
            commit(&cache, &mut store, &plan, current_start);
            current_start += c.frame_seq_len;
        }
    }
}
