# Plan details

Design decisions beyond `orig-plan.md`. Update when decisions change.

## Module: unit of offloadable work

- A `Module` declares: input descriptors, output descriptors, weight references, `forward(ctx, inputs, outputs)`.
- No scheduling, step counts, or loop shape baked in. Diffusion blocks, single-token AR steps, VAE all share the shape.
- Weight refs are handles, not bytes. Resolved lazily at forward time.
- Modules can be leaf or composite (sequence / dynamic-children). Composition is just another module.
- Modules don't carry shader source. Kernels are runtime-owned.
- Pipeline glue (Z-Image, LTX, etc.) lives in `thinfer-models/<model>` and is allowed to be model-specific.

## Tensor / residency typing

- Distinct types per residency tier with explicit moves: `GpuTensor<C>`, `HostTensor`, `OpfsHandle`, etc. Generic over compute dtype `C`.
- Kernels accept only `GpuTensor<C>`. Host→GPU is an explicit `.to_gpu(&runtime).await`.
- High-level `Pipeline` API may hide moves; runtime/kernel layer must not.
- Activation tensors stay GPU-resident across module boundaries. No CPU round-trip between modules.

## Dtype: storage encoding ≠ compute dtype

- Storage encodings (load-time): fp32, fp16, bf16, int8, int4, group-quantized, ...
- Compute dtypes (kernel-time): fp32, fp16 (later fp8).
- Kernels are parametric over compute dtype. Weight loader handles encoding→compute (dequant on upload or on first use).
- Per-module precision policy: a sensitive module can compute in fp32 with weights stored in any encoding. fp32-island detection (post-v1) just changes the policy values.
- bf16 storage is M1 (not M2). Z-Image-Turbo was trained in bf16; an fp16 cast loses magnitude (>65504 -> inf) and produces broken outputs. fp16 is intentionally avoided for Z-Image. bf16 storage was already a v1 commitment for LTX-Video, just pulled forward.
- bf16 v1: expand to fp32 at upload (2x GPU bytes for bf16 weights). All kernels stay fp32 in M1. Provably correct baseline that diffs against PyTorch with a clean tolerance.
- Native fp16 compute (`enable f16;` extension) is intentionally NOT pursued for Z-Image. fp16 safety is input-dependent (activation magnitude varies with prompt, resolution, noise schedule); we cannot statically prove a layer is safe across arbitrary inputs. Probabilistic safety is the wrong contract for an inference engine. The per-module precision policy stays in the trait shape but isn't exercised for Z-Image.

## Weight bytes: never in WASM linear memory

- `WeightBytes` is opaque and uploadable, not slice-readable as Rust bytes.
- Native impl: `Vec<u8>` / mmap / file handle.
- Web impl: `JsValue` holding `ArrayBuffer` / `Uint8Array`. Upload via `web_sys` `GPUQueue.writeBuffer` directly from JS heap.
- Applies to: safetensors loads, OPFS reads, network fetches, eviction-tier caches. No exceptions.

## Weight catalog

- safetensors header parsed once → `WeightCatalog: HashMap<Name, {offset, size, dtype, shape}>`. Persists for the runtime's lifetime.
- Eviction operates on bytes only. Catalog never evicts — small (KB to low MB) and re-parsing safetensors headers on cold paths is painful on web.
- Reload is byte-stream, not parse.
- First-time load: streaming. GPU uploads start as soon as a tensor's bytes arrive (network or OPFS), not after full file is in.

## Pipeline cache

- Lives on the runtime, not the module. Process-lifetime.
- Keyed by `(kernel_id, hint)`. Hint string covers everything that affects WGSL: dtypes, ranks, workgroup, tile, fusions.
- Module load/evict does not touch this cache. Same kernel reused across modules dispatches the cached pipeline.
- Compile is async (`create_compute_pipeline_async` / `createComputePipelineAsync`). Issue compiles for next module concurrent with current module's dispatches.
- Disk-cache the compiled pipelines (browser persistent storage, native fs cache) — deferred, but `PipelineCache` trait must allow it.

## Multi-tier residency

- Tiers (top → bottom): GPU device → GPU host-visible (when distinct) → host RAM (or JS heap on web) → OPFS / disk.
- Host-visible tier is a logical layer; on integrated GPUs it collapses with device (zero-copy, no runtime branch needed at the wgpu API level).
- Eviction flows down a tier, never sideways or up. Bytes evicted from host fall to OPFS; from GPU fall to host.
- OPFS is random-access; treated as the slowest cache, not a cold start.
- Integrated GPU: skip host-visible tier (same physical memory).
- Tier policy: LRU per tier in v1. Frecency / pinning is a knob to add later if traces justify it.
- Eviction reuses size-class slots in a buffer pool. No `destroy/createBuffer` on hot path.
- `WeightResidency<S>` API: `register(meta) -> WeightHandle`, `acquire(handle, &B) -> GpuView<'_>` (async). Backend is passed per-call, not owned. `GpuView` is a pin guard; eviction skips entries with `pin_count > 0`. `ResidencyBudget { ram_bytes, vram_bytes }` is always required - the manager has no "unbounded" mode. Caller subtracts its own workspace/activation estimate from `vram_bytes` before constructing; no sub-budget inside the manager.
- Per-module forwards stay sync and take `bufs: &<X>Bufs` plus a caller-owned `CommandEncoder`. The residency dance (acquire views, encode, submit, await, drop views) lives at the driver layer (`ZImageDit::forward`), one submit per module so pins release between modules. Module structs no longer carry `weights` fields. `<X>Handles` + `<X>Views<'a>` live next to each `<X>Bufs`; `handles.acquire(...)` returns views, `views.bufs()` materializes BufRefs for the encoder build.

## Schedule: dynamic generator

- `Pipeline::next_work(&mut self, &RuntimeState) -> Option<WorkItem>`.
- `WorkItem` = module handle + inputs + completion tag.
- `RuntimeState` exposes: completed-work tags, current step counter, memory budget remaining per tier, output-ready queries (for AR's "did module N's output land yet").
- Runtime peeks ahead K items for weight prefetch + pipeline compile prep. K configurable per execution policy.
- Pipeline can return `Pending` if next decision needs a previous output. Hot path for AR / EOS / conditionals; rare for diffusion.
- Runtime trusts `Pipeline` in release; debug-asserts on duplicate WorkItems, unloaded module handles, and input dtype/shape mismatches.

## wgpu callback pump (`WgpuPoll`)

- wgpu-native needs *something* to call `device.poll` for `map_async` / queue-completion callbacks to fire. wgpu's own helpers don't include a pump. Inline `device.poll(Wait)` on the awaiter blocks the async executor → unacceptable for production.
- Pattern (lifted from cubecl `crates/cubecl-wgpu/src/compute/poll.rs`): one OS thread per `WgpuBackend`, spawned at init, **parked at rest**. An `Arc<()>` sentinel is the wake signal — baseline `strong_count == 2` (one ref in `WgpuPoll`, one in the thread); a `PollGuard` clones to 3+ and `unpark()`s. While guards exist, thread loops on `device.poll(Maintain::Wait)`. When all guards drop, thread re-parks.
- Web: `WgpuPoll` is a no-op stub. Browser event loop drives wgpu callbacks. `PollGuard` is a unit struct.
- Executor-neutral. `read_buffer` future is the same code on tokio, smol, pollster, wasm-bindgen-futures.
- For queue-drain sync without readback, use `queue.on_submitted_work_done` + a `oneshot` (not implemented yet; trait method TBD).
- Rule for new async `Backend` methods on wgpu: if the future depends on a wgpu callback (`map_async`, `on_submitted_work_done`), it must hold a `PollGuard` from before the callback is registered until the callback resolves. Inline `device.poll(Wait)` is banned in async paths.
- Why `Maintain::Wait` (not `Poll` + sleep): blocks until something happens. No busy-wait, no polling-latency floor on small transfers.
- Connection to orig-plan's "generic over async executor": the pump is what makes this real. Without it, every readback would block whatever executor thread it ran on, and the abstraction would be a fiction.
- Single thread per backend in v1. Multi-queue / multi-thread submission later may want a dedicated pump per queue or `Instance::poll_all` from a single pump — revisit when we add a second queue.

## Forward-time scaffolding

- Pipeline-cache inserts happen at module *prepare* time, never inside `forward`. `ForwardCtx` exposes only the read-only `PipelineLookup` half so the hot path can't accidentally trigger compile.
- `Workspace` is a size-classed pool: `alloc(backend, bytes)` returns its own physical `GpuBufferId`; `reset()` returns rented buffers to per-class free lists. Pool reuse is across forwards, not within. Sub-slicing one mega-buffer is NOT viable: wgpu rejects a single dispatch binding the same buffer as both storage-read and storage-read-write, even with disjoint offset/size, so every chained kernel needs distinct buffers.
- Lifetime-aware reuse within a forward (XLA-style buffer assignment) is the v2 move. v1 keeps every activation live to end-of-forward.

## Concurrency on the hot path

- Per-module forward issues all dispatches into one `CommandEncoder`, one submit. No mid-forward awaits on the hot path.
- Cold path: pipeline compile is awaited at module *prepare* time (ahead of dispatch), not inside forward. First forward never blocks on compile.
- Weight upload for module N+1 starts immediately after module N's submit, before module N's completion fence.
- Pipeline compile for module N+1 issued in parallel with module N's dispatch and weight upload.
- No readback between modules. Activations stay on GPU.
- Single wgpu queue in v1.
- Cancellation: stop scheduling new work; in-flight `writeBuffer`/dispatches finish (wgpu can't cancel them); buffers freed when fences resolve.

## Backends

- WebGPU is first-class but not the only target. Backend trait abstracts over wgpu (native + web), with room for CPU SIMD / native compute APIs later.
- WebGPU-specific glue: pipeline compile via `createComputePipelineAsync`; weight upload via `queue.writeBuffer` from JS-side `ArrayBuffer`; OPFS file handles held JS-side.
- Device limits: request `max_storage_buffers_per_shader_stage = 8` (WebGPU baseline) over wgpu's downlevel default of 4. Fused kernels with many bindings need it.

## Model rollout sequencing

The project thesis is "thin inference": low-quant GGUF compute on memory-constrained devices, especially in-browser WebGPU. Quant kernels are the destination, not an optimization phase. M1's fp32 path exists to be the diff-against-PyTorch baseline that validates M2's quant kernels.

- M1: bf16 storage + fp32 compute end-to-end. Z-Image-Turbo, CLI then browser. Provably correct baseline.
- M2: GGUF Q-block compute kernels in WGSL. Block-dequant inside the matmul shader (Q8_0 first - simplest layout, ~lossless; Q4_K next - real VRAM win). Non-matmul ops stay fp32 until profiling justifies more. Reuses every layer except the format parser: catalog, manifest, cache, audit, model wiring, residency tiering, fp32 baseline kernels for diffing. This is the entire point of the engine.
- M3: LTX-Video. Reuses Q-block kernels from M2 (also bf16-trained transformer, similar shapes).

## Crate split (extends orig-plan)

- Per orig-plan: `thinfer-core` (wasm-able, no platform deps), `thinfer-web`, `thinfer-cli`, `thinfer-models`.
- Adds `thinfer-native` (working name; rename TBD): library crate with desktop/native impls of `thinfer-core` traits (mmap weight source, tokio executor, native fs). `thinfer-cli` depends on it.
- wgpu Backend impl and kernels live in `thinfer-core` — wgpu builds on wasm, so this is shared between native and web targets, not duplicated.

## Tracing

- `tracing` crate. Macros compile out via `max_level_*` / `release_max_level_*` features → zero cost when off.
- Categories: `weight.upload`, `weight.evict`, `dispatch`, `compile`, `residency.move`, `fence.await`, `pipeline.next_work`.
- Subscriber configurable via env var (per-category level). Web: `tracing-wasm` to console.

## Test fixtures & pytorch reference

- `projects/thinfer-pytorch-ref/` — uv project. `uv run` generates fixed-seed reference outputs (per-module, not just final) as safetensors. Pinned torch version.
- Small fixtures (per-op + small-model) committed to repo for CI: pure-Rust diff against thinfer outputs, dtype-appropriate tolerances.
- Large fixtures (full Z-Image, LTX) stay on local disk; gated behind a `cargo test` feature for manual integration runs.
- Differential mode (orig-plan) uses the per-module outputs to surface first-divergence at block granularity, not final-image diff.

## Op trait

- Ops declare static metadata + WGSL only: `KERNEL_ID`, `Dtype`, `INPUTS` keys, `OUTPUT` key, `wgsl()`, `layout()`. ~10 lines per op.
- Single generic `dispatch_op<O: Op, B: Backend>` covers every op. Sync, appends to caller-owned encoder. Constraints: input slots `0..N`, output slot `N` (last), all storage buffers in `Op::Dtype`'s storage layout, output element count = max input element count.
- Trait covers elementwise / single-output / storage-only ops. Multi-output / uniforms / weights / dynamic shape get a sibling trait when first hit.
- Pipelines are NOT owned by ops — `PipelineCache` (runtime) holds them. Same kernel reused across modules dispatches the cached pipeline. Op file references its own `WGSL` / `LAYOUT` consts only for compilation input.
- Test harness lives in `tests/common/`, never `src/`. Adding an op = WGSL + metadata + one line in the test registry.

## Open / deferred

- Residency tier set may grow (e.g., HTTP range source as a fifth tier post-v1).
- Frecency vs LRU — LRU first, revisit with traces.
- Pipeline disk cache — trait-shaped now, implementation deferred.
- IR / graph optimizer — deferred per orig-plan.
- Persistent prefetch hints / runtime-learned residency schedules — deferred.
