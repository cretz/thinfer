# Worklog

## Current state

Q8 path lit up with native `SHADER_F16` activations:
- `ActDtype::F16` added (sibling of `Bf16`); storage `array<vec2<f16>>`.
- `WgpuBackend::supports_shader_f16()` requests `Features::SHADER_F16`
  opportunistically; falls back to `ActDtype::F32` when unavailable.
- Pipeline.rs Quant arm picks F16 acts when the feature is live.
- All ops (matmul, sdpa small+large-D, rmsnorm, layernorm, rope+halfrot,
  qkv_split, scatter_pad_rows, silu, silu_mul, add, mul, tanh,
  bcast_affine, bcast_fma, bcast_add x{wf32, wbf16}) gained F16 variants.
- Pointwise ops compute in native `vec2<f16>` (silu/silu_mul/add/mul/tanh
  /bcast_fma/qkv_split/scatter_pad_rows). Reductions widen to f32
  internally and narrow on write (matmul / sdpa / rms / layer norm).

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

## Next

F16 NaN bug is fixed. e2e_parity passes for both `gguf_q8_0` and
`safetensors` variants with shader-f16 acts.

Tier 1 matmul perf done: tile_a + tile_b in `vec2<f16>` shared memory,
K-paired inner loop (kk steps by 2, one vec2 load = 2 FMAs). Halves
shared-mem pressure on the F16 path, halves shared-mem read bandwidth
in the inner loop. Accumulator stays f32 so no precision delta. Q8
cooperative dequant writes directly to vec2<f16> tile_b slots; bf16/f32
weight loads also pair along K. Bf16/f32 act paths are unchanged.

Q8+F16 step time on Arc 140T iGPU (low power, 512x512, 2 steps):
- 30.5 s/step -> 14.4 s/step (-53%).
- FFN gpu_disp -62%, QKV gpu_disp -62%, SDPA unchanged.
- Now 2.8x faster than bf16 per step.

Tier 1 conformance + e2e_parity (gguf_q8_0 + safetensors) all pass.

Remaining perf tiers (in priority order):

1. **Tier 2 - asymmetric matmul tiles per shape.** QKV (N=11520) and
   FFN up (N=10240) are wide-N: bm=32,bn=128 likely wins. FFN down
   (K=10240) is wide-K: bm=128,bn=32 likely wins. Per-matmul tile
   in `BlockMatmuls`.
2. **Tier 4 - residency.** Pin small per-block weights (AdaLN, norms,
   biases). Raise per-machine VRAM ceiling above 4 GB so Q8 main DiT
   (~6 GB) goes fully resident. Audit `prefetch_after` actually
   overlaps GPU compute (Vulkan single-queue may serialize).
3. **Tier 3 - kernel fusions.** silu_mul into ffn_w1/w3 writes;
   rmsnorm into following matmul; residual `add` into prior matmul.
4. **Tier 5 - exotics.** Subgroup matrix intrinsics on Arc (XMX);
   persistent kernels. Only if 2-4 don't get us to target.

### The NaN bug (resolved; kept for context)

Fix: saturated-narrow at the matmul F16 storage write site
(`thinfer-core/src/ops/matmul.rs`). The accumulator pair clamps to
`+/- 65504.0` before the `vec2<f16>(vec2<f32>(...))` cast. This stops
the proj-matmul overflow without changing model semantics (padding rows
remain fully-attended per upstream `transformer.py`); padding-row
outputs become saturated-but-finite instead of inf-then-NaN, real image
rows attending to padding incur only a small F16 precision delta.

Per-op taps for ctx_refiner block 0 (`DitTaps::ctx_block0`) are wired
and remain useful for future F16 narrowing; kept in place.

### Original NaN signature (read this before re-debugging)

First narrowing of the NaN source from `THINFER_TRACE=verbose` DIT-TAP:

- `t_emb` -> `AdaLN matmul` -> `AdaLN bcast_add` -> `scale_msa, gate_msa,
  scale_mlp, gate_mlp`: ALL clean (post-matmul, post-bias, post-tanh).
  AdaLN side of the f16 path works.
- `cap_embedded` (post cap_embedder = rmsnorm + matmul + bcast_add):
  CLEAN (no NaN, 32 rows x 3840 dim, magnitudes ~0.1).
- `ctx_refiner_0_out` (after one Block.forward modulation=false on
  cap_embedded): **17 of 32 rows are NaN**, exactly the tail rows
  15..31. Clean rows 0..14. NaN bit pattern is `0xfe00` repeated
  (negative quiet NaN). That's a f32 -NaN narrowed to f16 — somewhere
  a kernel computes -NaN in f32 (`0 * -inf`? `inf - inf`?) and
  `vec2<f16>(...)` narrows it.
- `last_ctx_refiner_out`: 100% NaN (block 1 propagates).
- `unified_in` and `main_layer_0_out`: 100% NaN.

Per-op taps inside ctx_refiner block 0 are NOT instrumented (DitTaps only
captures main block 0). That's the missing diagnostic — the bug is in
exactly one of these ops, but we can't see which:

```
rmsnorm(x) -> matmul(qkv) -> qkv_split -> rmsnorm(q,k) -> rope(q,k)
  -> sdpa -> matmul(proj) -> rmsnorm -> add -> rmsnorm
  -> matmul(w1), matmul(w3) -> silu_mul -> matmul(w2) -> rmsnorm -> add
```

### Verified clean (do NOT re-suspect these)

- Adapter requests `SHADER_F16` and gets it on Intel Arc 140T.
- Pipeline.rs Quant arm picks `ActDtype::F16`.
- `cap_embedded` is clean -> rmsnorm F16+wbf16, matmul F16+wbf16, and
  bcast_add F16+wbf16 ALL produce correct values at cap_embedder shape
  (n_rows=32, dim=3840). Same kernels run inside ctx_refiner block 0.
- t_emb/AdaLN path -> AdaLN matmul (F16+wbf16), bcast_add (F16+wbf16),
  tanh (F16) all clean.

### Root cause (narrowed via ctx_block0 per-op taps)

Per-op taps in `context_refiner.0` (added via `DitTaps::ctx_block0`,
mirrors `block0` for the modulation=false path) pinpoint the first
non-finite as `attn_out` (attention output projection matmul):

- attn_norm1_out, attn_q/k/v, attn_q_norm/k_norm, attn_q_rope/k_rope,
  attn_sdpa: all clean (no nan, no inf).
- attn_out: `nan=0 +inf=17 -inf=17`, max=6.4e4, min=-4.5e4. F16 max is
  65504; proj output saturates to inf at 34 positions, all in the
  padding-row tail (row buckets show finite mass concentrated in
  rows 0..14 and explosion in rows 15..31).
- attn_norm2_out: `nan=34 zeros=65246`. RmsNorm over a row containing
  inf yields `sum_sq=inf -> rsqrt=0 -> finite*0=0` at finite positions
  and `inf*0=NaN` at the inf positions. This is where the NaN bit
  pattern `0xfe00` is born (negative qNaN = the canonical f16 result
  of inf*0). Residual + ffn propagate NaN through rows 15..31.

Why padding rows specifically: cap_embedder maps an all-zero input
(text-encoder padding) to the bias vector identically across all
padding rows (rmsnorm passes zero; matmul of zero is zero; +bias).
Identical input gives identical Q across heads, uniform softmax,
attn_sdpa = mean(values) per head. Combined with the proj weight
matrix this happens to land magnitudes > 65504 for these specific
rows. F32 represents the same values without overflow, which is
why the F32 path looked "fine" even though it was also producing
junk values at padding rows (those rows are not used post-DiT).

Sdpa is exonerated. Don't audit softmax; the bug is downstream.

### Next: fix the F16 overflow

Per-op taps for ctx_refiner block 0 are now wired (`ctx_block0` field
on `DitTaps`, allocation via existing `Block0TapBufs`, prints alongside
`block0.*` in `pipeline.rs`). Use them after any fix to confirm.

Candidates for the fix:

1. **Mask cap padding out of attention** (proper-model-correctness
   fix). `cap_mask` is currently `attn_mask_zero_bytes_act` (full
   attention). If padding key columns are set to -inf, real-query rows
   wouldn't average in padding values, and (this is the bit that
   matters here) padding-query rows would output zero (softmax of all
   -inf is undefined - guard with q-row mask too, or just live with
   garbage on padding-output rows). This is the cleanest fix but
   requires upstream parity: does Z-Image actually mask cap padding?
   The fact that F32 path is bit-clean against the reference suggests
   reference also runs unmasked.
2. **Clamp matmul output to F16 finite range** before f16-narrow on
   write. Cheap, targeted, but a band-aid.
3. **Keep proj-matmul accumulator route as F32-storage out**, then
   narrow only after the subsequent rmsnorm. Requires either fusion
   or a one-shot "rmsnorm-of-f32-buffer-with-f16-weights" variant.
4. **Zero padding rows in the cap stream** after cap_embedder, before
   the refiner. The model can't have learned anything useful from
   constant-bias padding rows, and zeroing them keeps F16 in range.
   Risk: this changes numerics on the legitimate-but-unused rows in a
   way the reference (which keeps them populated) wouldn't, so the
   parity-cap-block test would diverge for those rows. Acceptable if
   we mask them out at compare time.

Recommend: option 4 (zero pad rows after cap_embedder), then verify
parity on non-padding rows only. It's the lowest-blast-radius change
and addresses the root cause (don't process garbage through F16-range-
sensitive ops). Option 1 is "more right" but requires upstream
behavioral parity confirmation.

### Files touched (the F16 surface)

- `thinfer-core/src/ops/mod.rs` — `ActDtype::F16`, `act_f16_prelude!`,
  `WgslConfig::F16_NATIVE_WBF16`, hint() update.
- `thinfer-core/src/backend/wgpu.rs` — `Features::SHADER_F16` request,
  `supports_shader_f16()`.
- `thinfer-core/src/ops/{matmul, rmsnorm, layernorm, sdpa, rope,
  qkv_split, scatter_pad_rows, silu, silu_mul, add, mul, tanh,
  bcast_affine, bcast_fma, bcast_add}.rs` — F16 variants.
- `thinfer-models/src/z_image/pipeline.rs` — Quant arm picks F16 when
  `backend.supports_shader_f16()`.
- `thinfer-models/src/z_image/{dit, seq, t_embedder, block}.rs` —
  upload/readback F16 cases, dispatch counts.
- `thinfer-models/Cargo.toml` — added `half = "2"`.

### Deferred (after F16 works)

- **Matmul tile_a in `vec2<f16>` shared memory.** Currently tile_a stays
  f32 even on the F16 path (only the global load/store narrows). Moving
  tile_a to vec2<f16> halves shared-memory pressure (16 KiB -> 8 KiB at
  bm=bn=64/bk=32), enabling bigger tiles on iGPU and freeing headroom on
  the 16 KiB web baseline. Inner loop widens f16->f32 at register-load
  time; accumulator stays f32.
- **Asymmetric matmul tiles** for non-square shapes (bm=32, bn=128 for
  QKV `M=1024 N=11520`). Only worth pursuing if the shared-mem tile
  shrink above doesn't already close the wall vs bf16.
- **e2e numerical tolerance**: f16 storage is lossier than the bf16
  baseline, so the existing tolerance may need a small bump once the
  NaN is fixed and we get a real comparison.

## Backlog

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
