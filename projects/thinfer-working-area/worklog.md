# Worklog

## Current state

Q8 e2e parity passes (numerical clean: step0/step1 `above_tol=0/16384`,
vae_rgb diffs match bf16 baseline). VRAM TRUE_PEAK 1.48 GiB / 2 GiB budget.
Wall: Q8 diffusion ~65 s, bf16 ~47 s on iGPU low-power.

Q8 is slower than bf16 because acts run as F32 (precision-deliberate, not
bf16-packed). For QKV shape at bm=bn=64: tile_a reads ~2.8 GiB vs bf16's
~1.4 GiB. B-bandwidth advantage (~0.7 GiB saved) doesn't cover the
A-bandwidth penalty (~1.4 GiB added). Net, Q8 reads more bytes per
matmul than bf16.

## Locked design decisions (carry forward)

- **Canonical weight schema = upstream Z-Image names.** Engine consumes
  upstream layout (fused `attention.qkv.weight`, `attention.out.weight`,
  `feed_forward.w{1,2,3}.weight`, AdaLN as `adaLN_modulation.0.{weight,bias}`).
  Every external source is a translation problem to canonical: trivial
  name remaps go through `RenamedSource` (`with_passthrough` republishes
  non-renamed entries unchanged); structural transforms (split->fused QKV,
  EMA extraction, dtype repacks) go through dedicated peer adapters.
- **Q8 DiT pinned to `ActDtype::F32`.** Bf16-packed acts on Q8 would lose
  precision at every kernel boundary (7-bit mantissa) and is the wrong
  intermediate format for the dequant path. Shader-f16 is the future
  replacement (10-bit mantissa, halves storage vs f32). Bf16 path stays
  on BF16_PACKED (bit-clean with bf16 weights).
- **Three pipeline-set split.** `block_pipelines` (main DiT), `dit_encoder_block_pipelines`
  (x/t/cap embedders + noise + context refiner + final_layer), `encoder_block_pipelines`
  (Qwen3 text encoder). Refiners/embedders never quantized; AdaLN stays bf16.
- **WGSL templating stays plain `format!()`.** Quant is closed-enum
  `QuantKind`, not `&dyn`.
- **GGUF parser is range-fetch-first.** No whole-file load. Web path
  will use OPFS `FileSystemSyncAccessHandle` through the same `FileOpener`.
- **B viewed `[N, K]` N-major in matmul.** GGUF native layout, no
  transpose at upload. Inner K-loop chunked by `block_size`.
- **`bk % block_size == 0` per scheme.** Q8_0 uses `bk=32`; bigger bk
  hurts iGPU occupancy more than the t-loop reduction saves.
- **shader-f16 is a sibling axis** (gated on `Features::SHADER_F16`).
- **Scope: DiT only.** unsloth Z-Image-Turbo-GGUF is DiT (12.3 GB bf16,
  6B params); Qwen3 + VAE stay safetensors for now.
- **Residency pool counts against the VRAM ceiling.** `evict_gpu_until_fits`
  includes `pool_bytes` in the predicate and spills pool entries back to
  wgpu when the pool exceeds the ceiling. Without this, mixed size
  classes (Q8 quant interleaved with bf16 fallback tensors) cause peak
  weights to drift above the budget non-deterministically.

## Next session: Q8 acts on shader-f16

`Features::SHADER_F16` is a wgpu feature, web + native (Chrome/Edge ship
the browser `"shader-f16"` GPU feature; Firefox WIP). Runtime-gate via
`adapter.features().contains(SHADER_F16)`; WGSL needs `enable f16;` at
top.

Plan:
1. **Add `ActDtype::F16`** sibling of `Bf16`. Packed via WGSL `f16` native
   storage (not the u32-bitcast hack). One f16 per 2 bytes; matmul read
   path widens to f32 inline for compute, write path narrows from f32 to
   f16. f32 accumulators stay f32 (no precision regression vs bf16-packed).
2. **Backend opt-in.** Request `SHADER_F16` when adapter has it; fall back
   to `ActDtype::F32` for Q8 when unavailable. Bf16 path is unaffected.
3. **Pipeline wiring.** Quant arm in `pipeline.rs` picks F16 acts when
   the feature is live. `BlockWgslConfigs` invariant (matmul + ops agree
   on act_dtype) extends naturally.
4. **Op kernel migrations.** Every ops kernel (rmsnorm, layernorm, sdpa,
   rope, qkv_split, silu/silu_mul, bcast_*, scatter_pad_rows) gets an
   F16 variant alongside F32/Bf16 packed. Bulk of the work.
5. **Expected payoff.** Workspace footprint halves (~1.38 GiB → ~700 MiB),
   tile_a bandwidth matches bf16-packed. Combined with Q8's smaller B,
   Q8 should land *under* bf16 wall.

## Backlog (after shader-f16)

- **Asymmetric matmul tiles** for non-square shapes (e.g. `bm=32, bn=128`
  for QKV `M=1024 N=11520`). Reduces tile_a reads at the cost of more
  shared mem (~20 KiB > web baseline 16 KiB; needs per-target tile
  config). Only worth pursuing if shader-f16 doesn't fully close the gap.
- **Operator fusion** (norm+matmul, silu+mul into single kernels) to
  eliminate inter-kernel acts traffic.
- **Sub-allocator** for residency pool: slab-based with free-list
  coalescing for non-uniform sizes (LTX, quantized).
- **Async-cancellation hole:** `submit_with_guards` via
  `on_submitted_work_done` callback.
- **Q4_0 then Q4_K_M.** Q4_0 is a one-bit-unpack-helper change on Q8_0.
  Q4_K is the production target. unsloth ships `_S`/`_M`/`_XL`; `_M` default.
- **Qwen3 to GGUF.** Separate unsloth Qwen3-* GGUF. Confirm variant
  matches Z-Image's tokenizer.
- **Bf16-clean VAE up_blocks.** Audit accumulation order or promote
  select stages to fp32.
- **Conformance fixtures** for `conv2d`, `group_norm`, `upsample2d_nearest`.
- **Per-model gen defaults** via `ModelId::defaults()` when LTX lands.
- **Lift `ZImageRecipe` to engine-agnostic `ComputeRecipe`** when
  LTX/GGUF land.
- **`gelu`** deferred until Z-Image audit (exact-erf vs tanh-approx).
- **e2e_parity NaN-loose assertion bug.** All-NaN currently reports
  `above_tol=0`; should fail loudly.
- **Dedupe `round_f32_to_bf16` / `act_upload_bytes`** between
  `seq.rs` (canonical) and `dit.rs` / `t_embedder.rs` (stale locals).
- **Tighten e2e parity tol** when bf16-clean VAE up_blocks lands.
- **Dispatch overflow regression test:** n > 4M elementwise.
- **Tiled VAE quality knobs** (overlap bump, halo-exchange per-layer).
- **CLI download per-decile progress.**

## Conventions

- Rope: `x [rows, heads, dim]`, `freqs [rows, dim]` interleaved.
  **DiT interleaved `rope`; Qwen3 half-rot `rope_halfrot`.**
- Sdpa: fused, `D <= MAX_D = 128`. Q `[B,S_q,H_q,D]`, K/V `[B,S_k,H_kv,D]`.
- Matmul: kernel is `a@b` with B `[K,N]`. PyTorch `nn.Linear` weights
  `[N,K]` — uploaded transposed at load. **GGUF Q8_0 stores `[N, K]`
  blocks natively; no transpose at load.**
- bcast ops single-batch.

## Carry-forward gotchas

- **VRAM total = weights + workspace + staging + pool.** The residency
  pool is wgpu-allocated and charged to weights; eviction predicate
  must include `pool_bytes` or peak drifts.
- **Per-category peak SUM is NOT the true peak.** Use
  `MemAccount::vram_total_peak()` for hard assertions.
- **`Workspace::drain_pool()` at phase boundaries.**
- **Packed-bf16 act idiom.** WGSL: `array<u32>` storage, two bf16 per
  word, `act_bf16_prelude!` for unpack/pack. Channel-broadcast requires
  C even. `causal_mask_bytes_act` requires seq even.
- **BF16_PACKED uses WeightDtype::Bf16.** Conformance bf16p fixture
  encodes everything as native bf16.
- **WebGPU dispatch caps at 65535 per dim.** Use `linear_workgroups` +
  2D `gid` index pattern.
- **Per-binding cap.** Web baseline 128 MiB < DiT FFN 150 MiB; M2 chunked-matmul.
- **WebGPU workgroup-storage cap.** Downlevel default 16 KiB; we request
  adapter's max. Matmul builds must still fit 16 KiB on web baseline.
- **VAE applies `(z/scaling) + shift` internally.** SCALING=0.3611,
  SHIFT=0.1159.
- **VAE tiled path.** Default tile=64, overlap=8. One submit per tile.
- **VAE diag rule.** <= few KB, in-encoder via
  `scope.read_buffer_via_encoder`. `STAGE_DIAG_MAX_BYTES`=1024.
- **Submit must await before reuse.** Never queue a second submit on
  the same workspace without it.
- **Z-Image text-encoder stops at Qwen3 `hidden_states[-2]`.**
  `embed_tokens.weight` row-by-row via `text_encoder::embed_lookup` -
  never in residency.
- **DiT `decode_image`** passes raw latent `c` to `seq::unpatchify`.
- **SDPA mask** is `[B, S_q, S_k]` additive. Z-Image full-attn:
  `seq::attn_mask_zero_bytes(seq)`; Qwen3 causal:
  `seq::causal_mask_bytes(seq)`. Unmasked: `sdpa_mask_stub` + has_mask=0.
- **fp32 tolerance** is 1e-5.
- **`cfg(test)` is per-crate.** Cross-crate test sharing needs feature
  flag / pub API / separate support crate.
- **Test gating.** `cargo test --workspace` skips uv/torch/wgpu.
  Conformance: `--features conformance`. Z-Image e2e: `--features
  zimage-e2e`.
- **Conformance test invocation:** one test `ops_match_pytorch_reference`
  iterates the registry; name filters do nothing.
- **Parity tests use the engine's exact weights.**
- **WIP branch flow:** `git add ... && git commit --amend --no-edit &&
  git push --force-with-lease`. No new commits on `initial-work` unless
  explicitly requested.
- **Matmul tile tuning (Intel iGPU).** Bigger register-blocked tiles win
  until `tm*tn=32` acc regs/thread. Q8 + 64x64/4x4/bk=32 with cooperative
  dequant (TPB threads per block, see matmul.rs Quant arm) is the
  current setting. Bigger bk hurts on iGPU (occupancy regression
  outweighs t-loop reduction).
- **`THINFER_TRACE=1`** enables tracing collection. `THINFER_TRACE=verbose`
  adds DIAG prints. `[mem]` snapshot gated on the same env var.
- **iGPU is `THINFER_POWER_PREF=low`** (or unset). Discrete is `high`.

## Memorized commands

- Conformance: `cargo test --release -p thinfer-conformance --features
  conformance ops_match_pytorch_reference -- --nocapture`
- e2e_parity (iGPU, both variants):
  `THINFER_POWER_PREF=low THINFER_TRACE=1 THINFER_E2E_PNG_DIR=<dir>
  cargo test --release -p thinfer-conformance --features zimage-e2e
  e2e_parity_for -- --nocapture --test-threads=1`
- CLI smoke: `THINFER_TRACE=1 cargo run -p thinfer-cli --release --
  generate image --prompt "..." --output <path> --height 512 --width 512
  --ram-budget 4G --vram-budget 4G`
