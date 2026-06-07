# Thinfer

A Rust inference engine designed from the ground up around the assumption that your model doesn't fit in device memory and your weights live behind random-access I/O.

## Why this exists

Existing inference engines either keep all weights resident (Candle, Burn, ONNX Runtime Web) or bolt offloading on top of architectures that weren't designed for it (HuggingFace Accelerate, manual ORT Web splitting). None treat memory-constrained execution and random-access weight sources as first-class design concerns. Thinfer does.

## Primary goals

### Core architecture

- Rust core, async-first, generic over async executor (single-threaded fine in v1, abstraction supports multi-threaded later)
- Single runtime per model — no session boundaries between blocks, no teardown between chunks
- Eager execution in v1; pluggable IR/optimizer architected for but deferred
- Tensor data lives in GPU buffers and JS-owned ArrayBuffers, never in WASM linear memory
- Trait-based abstractions so backends, weight sources, and model formats can be swapped

### Crate structure

- `thinfer-core` — engine logic, no platform-specific dependencies
- `thinfer-cli` — native binary for benchmarking, demos, debugging, model inspection
- `thinfer-web` — WASM bindings, OPFS integration, JS-side weight buffer management
- `thinfer-models` — pipeline glue code in Rust (Z-Image-Turbo for v1, contributions welcome for future models)

### Abstractions (trait-based)

- Model format loader — safetensors v1
- Compute backend — wgpu v1
- Weight source — OPFS, mmap'd file, in-memory v1
- Tokenizer — wraps the `tokenizers` crate
- Async executor — tokio (native), wasm-bindgen-futures (web)

### Precision

- fp32, fp16, bf16 (bf16 implemented via fp32 ops on wgpu given current shader limitations)
- Per-module dtype assignment for mixed-precision policies (sensitive ops in higher precision, bulk compute lower)

### Streaming, residency, and multi-tier memory

- Multi-tier memory hierarchy: GPU device → GPU host-visible (when distinct) → host RAM / JS ArrayBuffer → OPFS / disk
- Bytes flow down through tiers on eviction rather than disappearing — eviction from GPU drops to RAM/JS, not back to OPFS
- Tier-aware allocation: weight bytes cached in RAM/JS so GPU eviction doesn't force OPFS re-read
- Integrated-vs-discrete GPU awareness: skip the host-visible tier on integrated GPUs where it's the same physical memory
- Frecency-based eviction (recency + frequency, aged) at each tier
- Manual pinning for hot weights (embeddings, frequently-reused projections)
- Configurable execution granularity: per-layer, per-block, per-module-group
- Async weight prefetch overlapped with compute (load block N+1 while running block N)
- GPU buffer pool with size-class allocation — no allocator on the hot path
- Persistent shader cache: compile once, dispatch many across all blocks
- Activation residency on GPU between blocks — block N's output is block N+1's input with no copy

### Resource policy

- ExecutionPolicy struct with named presets (`min_memory`, `balanced`, `max_throughput`, `low_latency_first_frame`) and field-level overrides
- Memory budget specification (auto-detect or explicit byte count)
- Proactive eviction stays under the budget — internal "hit-our-limit" cases don't reach users
- Real OOM is fatal: surfaced as a clean error, no internal retry

### Lifecycle

- Cancellation: drop the future, GPU work stops, buffers freed
- Progress reporting: per-step callbacks (denoising step N of M, current latency, current memory)
- Multi-level tracing using the `tracing` crate, debug-gated, configurable verbosity

### Debugging tools

- Feature-gated fp16 overflow / underflow detection (track per-op activation ranges, flag risky ops)
- NaN / inf propagation tracking
- Per-op timing (WebGPU timestamp queries where available, wall-clock fallback)
- Execution trace export
- Differential mode: run thinfer alongside a reference, surface first divergence

### Test harness

- RAM-conscious differential tests against PyTorch (stream reference data from disk, don't load entire reference outputs at once)
- Fixed-seed input generation, reference outputs saved as safetensors
- Precision-appropriate tolerances (`rtol`/`atol` per dtype) — no expectation of bit-exactness with PyTorch
- Per-op tests + end-to-end model tests
- Cosine similarity / PSNR for outputs where bit-comparison isn't meaningful

### Pipelines

- Pipeline glue written in Rust, one module per model in `thinfer-models`
- WASM bindings expose whole pipelines, not individual blocks (end users don't write glue)
- No declarative manifest format in v1

### Demos

- CLI Z-Image-Turbo generation
- Browser Z-Image-Turbo generation
- Memory-constrained run on a phone — works at a target VRAM budget that current browser inference engines can't hit

### Targets

- Mobile browser as a first-class target, not a degraded fallback
- Desktop browser
- Native (Linux, macOS, Windows)
- Performance: explicitly not benchmarked against any specific competitor; success is "works at memory budgets others can't"

## Deferred (post-v1)

- AOT quantization tooling (calibration, sensitivity analysis, layout transforms, fp32-island detection)
- int8 and int4 support
- Graph optimizer (AOT or runtime-learned)
- Persistent metadata store with runtime-learned hints (residency schedules, fp32 islands, calibration data)
- Model manifest format
- LoRA / adapter loading
- HTTP range request weight source
- Multi-threaded native execution
- Deterministic mode and golden file regression tests
- WIT / WebAssembly Component Model pipelines
- Additional model format loaders (GGUF, ONNX, raw state dicts)
- Additional compute backends (CPU SIMD, native CUDA / Metal, Vulkan compute)

## Non-goals

- Training. Inference only.
- Bit-exact reproducibility with PyTorch.
- Beating any specific competitor on raw kernel performance.
- Supporting models that fit comfortably in memory at the expense of streaming-first design.
- Becoming a general-purpose ML framework. This is an inference engine for memory-constrained execution of generative models.