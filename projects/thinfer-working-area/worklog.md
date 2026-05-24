# Worklog

## Current state (2026-06-05, VAE overhaul: implicit-GEMM conv + parallel GN + f16 acts)

VAE decoder 46.3s -> 3.9s at 768 (4 tiles x ~0.97s, was 11.55s/tile). VAE
is no longer a wall item (~4% of an 8-step run). Three changes:

1. conv2d rewritten as implicit GEMM (`weight[Cout,K] @ im2col[K,M]`,
   im2col never materialized; output lands NCHW, coalesced spatial
   stores/gathers). Configurable `Conv2dConfig` tiles, explicit-scalar
   register blocks. Three regimes in `VaeDecoderPipelines`: default
   64/64/32/4/4, wide (bn=128/tn=8, m>=65536 && cout>=64), small-N
   (bm=4, cout<=4 i.e. conv_out). Old one-thread-per-output kernel gone.
2. group_norm rewritten: one WORKGROUP (256 threads) per (b,group) row,
   shared-mem tree reduction (was 1 thread/row serial scans - the hidden
   up_block.3 killer at 512x512).
3. VAE acts f16 when device has SHADER_F16 (F32 path = fallback).
   Reductions f32 in-kernel; stores saturate +-65504; host edges convert
   via `half` (latent upload, mid/tile readbacks); tile slicing stays in
   raw act bytes. New f16 variants: group_norm/upsample/transpose12;
   mid-block matmul moved to 64/64/16/4/4 tile (old 16x16 default is
   illegal for f16 tiles: tn%2).

- Per tile: up0 0.6->0.07s, up1 2.8->0.30s, up2 3.9->0.33s,
  up3 3.9->0.23s, tail 0.32->0.009s. 768 q4 e2e wall 91.7->46.5s.
- TRUE_PEAK 2.00GiB exact (workspace peak 1.91->1.72GiB from f16).
- Pyref q8 256: PASS; conv_out slope 1.009842 (f32 baseline 1.009814),
  latent slope 1.008233 rel 2.53% - f16 invisible at output. Early-stage
  taps (up0/up1) show f16 rounding above tol counts but wash out by up2.
- Conformance 40/40 incl. 2 new conv2d cases (multitile edges, batch-2).
- Scratch logs: smoke-768-vae-baseline.log (pre-conv-rewrite),
  smoke-768-igemm-baseline.log (igemm f32), smoke.log (GN+f16+regimes).

## Prior state (2026-06-05, matmul_i8 vec4 tile loads)

matmul_i8 round 3 part 1: A/B bindings re-viewed as array<vec4<u32>>;
tile loads are one coalesced 16B load per thread (was 4 scalar u32 loads
with div/mod each). Safe because one K=32 sub-block row is 2 vec4s
(never straddles rows), K%32==0 keeps binding sizes 16B-divisible.

- Main-block matmul GPU (sum, 2 steps, q4_k_m 768): 13153 -> 11798ms
  (-10.3%); ffn2/attn_qkv -12.4%, ffn1 -7.8%, attn_proj -11.1%.
- diffusion_steps wall 22.7 -> 21.2s (-6.7%). TRUE_PEAK 2.00GiB exact.
- Conformance matmul_i8 6/6, matmul_i8_bf16 4/4. Pyref q8 256:
  13/16384 latent cells out of tol, slope 1.0084 rel 2.5% - identical
  to pre-change baseline (numerically transparent).
- Untried from round-3 list: BK=64, tn bump (occupancy risk each).

## Prior state (cross-section submit ring)

dit_forward now runs ONE depth-2 deferred-submit ring across all sections
(noise_refiner, cap glue, context_refiner, main layers) instead of sync
submit_void at every section boundary. Hold-bags are type-erased
(`RingHold` blanket trait); `BatchScope::import_copy` used at deferred
sites (caller's hold-bag keeps views alive). Perf-NEUTRAL (q4_k_m 768:
steps 22.7s vs 21.8s baseline, within session noise; TRUE_PEAK still
2.00GiB; test passes). Kept for structure: proper pipelining, no forced
drains at section boundaries.

Key finding: the "step-boundary burp" (old Next item a) is DEAD. The
section-start gaps were GPU-busy time (196MB/block weight-upload copies
ride the queue ahead of dispatches), not CPU fence waits. What survives:
~260ms/step of queue.submit() blocking at layer-loop start (block.3
ordinals, 50-110ms each, near-empty cmdbufs) - driver/staging-belt
backpressure, not our fences. Steps are otherwise upload-bandwidth +
matmul bound (wb_ms 20-80ms per block, every step, from 2GiB eviction
churn).

## Prior state (refiner Q8_0 transcode)

Refiner matmuls moved off the bf16 path: unsloth GGUFs (Q8_0 AND Q4_K_M)
store all 28 refiner matmul tensors as BF16, so refiners were paying the
untuned bf16 matmul plus 354-362MB/block weight re-uploads. Landed a
load-time bf16 -> Q8_0 transcode (`WeightMeta::transcode` +
`encode_q8_0_from_bf16`, llama.cpp quantize_row_q8_0 semantics) so
refiner qkv/out/w1/w2/w3 ride the DP4A matmul_i8 path:

- e2e 768 q4_k_m 2-step: 94.97 -> 88.30s (~3.3s/step).
- noise_refiner GPU: ~1.04s -> ~474ms/step; context_refiner 45 -> 32ms.
- refiner weight upload: 362 -> 196MB/block (~660MB/step less).
- Pyref parity q8 at 256: 13/16384 latent cells out of tol (was 14),
  slope 1.0084 rel 2.5% (was 1.009 / 2.4%). Conformance 40/40.

Prior round (sdpa subgroup rewrite + matmul_i8 register blocking tm=8):
block 1121 -> 259ms, ~2.9 TFLOPS eff on matmul_i8 (still ~3x from DP4A
ceiling). 768 e2e passes at default 2G/2G (TRUE_PEAK 2047MiB).

## Next (priority order)

1. **DiT round 2** (in expected-value order):
   (a) matmul_i8 round 3: vec4 B-loads, possible BK=64, tn bump.
   (b) queue.submit() stalls at layer-loop start (~260ms/step, 50-110ms
       per call on near-empty cmdbufs): driver/staging backpressure;
       suspect wgpu staging belt or outstanding-copy throttle.
   (c) upload churn: every main block re-uploads ~200MB/step at 2GiB
       budget; structural fix is smaller weights (Q4 main already) or
       smarter residency, not eager-release.
2. **Text encoder to llama.cpp-grade**: Qwen3 ~17.8s (F32 acts + bf16
   weights, untuned matmul) - now the biggest single wall item. Target:
   GGUF-quantized encoder weights + F16 acts + the same DP4A matmul path
   the DiT uses. The refiner transcode machinery applies directly
   (encoder linears are bf16 safetensors; K%32==0 holds).
3. VAE leftovers (low priority now): conv tile sweep (bk, wide-regime
   thresholds), mid-block front still untimed separately, single-shot
   `decoder_forward` is dead code (no callers) - delete or keep?
4. Cleanup/backlog: tighten e2e tolerances to measured baselines;
   revisit sdpa_i8 quality (run the i8_sdpa variant; if noisy,
   K-smoothing a la SageAttention); arbiter eviction churn at 256.

## Invariants (current architecture)

- **VRAM budget has one owner**: `MemArbiter` (thinfer-core/src/arbiter.rs),
  created by `WeightResidency::new`, shared into every `Workspace`. Net-new
  allocs call `ensure_headroom`; the reclaim chain runs in priority order
  (idle workspace pool -> evictable weights -> unpinned ring slots). No
  peer-to-peer spill hooks; lock order strictly arbiter -> client.
- Budget is a ceiling target: chain-dry overshoot is traced
  (target=thinfer::arbiter) and caught by the e2e TRUE_PEAK assert, not a
  hard alloc error. Structural too-big-for-budget weights still error.
- Weight acquire under pressure recycles the oldest unpinned SAME-SIZE
  resident's buffer in place (no free+alloc); with headroom it allocates
  fresh so large budgets build full residency. Residency's size-class pool
  is gone.
- e2e TRUE_PEAK now lands exactly at 2.00GiB/2.00GiB (q8 + q4); parity
  unchanged (q8 latent slope 1.009 rel 2.4%, q4 0.976 rel 9.3%).
- ActDtype::I8 is NEVER a block-wide ops dtype (`BlockWgslConfigs::validate`
  asserts). Residual carry, norms, modulate, FFN glue: dense at act dtype.
- Matmul boundary: `dispatch_matmul_site` accepts dense (act_quant
  transcode inside, DP4A) or paired A-side (sdpa_i8 output -> proj direct).
- **Weight transcode**: `WeightMeta::transcode = Some(Q8_0)` requantizes
  bf16 [N, K] (K%32==0) into GGUF-native Q8_0 blocks at upload, no
  transpose. Applied to refiner matmuls iff any main-layer matmul is
  file-quant (`refiner_transcode_target`); `ZImageModel::load` mirrors the
  decision in the dit-encoder pipeline cfgs. Q8_0 target regardless of the
  main scheme. Embedder/final_layer set split back out to pure bf16.
- `i8_sdpa` (recipe, default off): main DiT blocks only; requires
  SHADER_F16 + quant path. sdpa_i8 I/O slots are fused `[i8 data || scale]`
  pairs via `alloc_pair`; data half is `rows*dim` BYTES (not act_bytes).
- act_quant pipeline is built when `(use_dp4a && any quant site) || i8_sdpa`.
- matmul_i8_bf16 built only for `i8_sdpa && proj weight == Bf16`.
- AdaLN, freqs, masks: plain act dtype (no per-domain overrides remain).
- Surviving i8 kernels: act_quant, dequant_i8, matmul_i8, matmul_i8_bf16,
  bf16_block_sum, sdpa_i8. matmul_i8 keeps the slot-7/8 dbg trace
  bindings (LAYOUT=9; disabled via dbg.enable=0 in production).
- No eprintln in library code: stage timing is tracing::info, diag dumps
  are target=DIAG and tap-gated. CLI without THINFER_TRACE prints nothing.

## Diag instrumentation (opt-in)

- `THINFER_E2E_STEP0_DIAG=1` enables the step-0 localization taps
  (damage-zone per-op taps + block-26 matmul byte audit). Off by default:
  the tap buffers pin ~300 MiB and bust the 2 GiB budget.
- Block-26 audit byte heads are captured INSIDE dispatch_matmul_site_diag
  post-transcode (works for dense and paired A). CPU oracles live in
  e2e_parity.rs (`audit_block26_matmul_i8`, `audit_qkv_segment_slopes`).
- Paired taps exist only for `attn_sdpa` under i8_sdpa (ActTapBuf.scale).

## Locked design decisions

- **i8 activation storage is matmul/sdpa-internal only.** Per-32-block i8
  cannot carry DiT residuals (fixed outlier channels ~3000x median; error
  re-injected per block and compounds). Matches llama.cpp (Q8_1 at matmul
  input only) / TRT-LLM (FP8 in GEMM only) practice. Do not revisit.
- **DP4A matmul** auto-on for `Packed4x8IntegerDotProduct + SHADER_F16 +
  F16 acts + Quant weight`. Tile bm=bn=64, tm=tn=4; subgroup runtime-branch.
- **Sdpa flash-attn small-D**: `SdpaF16Sg` BR=16 BC=32 WG=128 CL=8 (16 KiB
  shared) on F16+subgroups; legacy BR=64/WG=64 one-thread-per-row kernel
  is the non-subgroup / non-F16 / D>128 fallback.
- **Kernel register rule**: no dynamically-indexed local arrays in hot
  loops (naga/NV spill to local memory); unroll via codegen with explicit
  scalars (matmul_i8) or explicit vec4 vars (sdpa).
- **Dequant-once per matmul site** on the non-DP4A fallback.
- **Four pipeline-set split** (main / encoder / dit_encoder / dit_embedder;
  embedder set now identical to dit_encoder).
- **GGUF parser range-fetch-first**, B viewed `[N, K]` N-major.
- **PowerPreference::HighPerformance** CLI/e2e default.
- **VAE per-resnet sub-submits**; **SUBMIT_DEPTH=2** static;
  **WEIGHT_RING_SLOTS=4**; matmul WGSL <= 32 KiB workgroup storage.

## What NOT to do (tested + rejected)

- i8 residual carry / i8 elementwise acts (see locked decision above).
- BC=64 sdpa packed variants (occupancy collapse).
- Mapped-staging same-encoder copy; dedicated submit per upload;
  elastic SUBMIT_DEPTH=3; static workspace reserve; weight prepack.
- Dual-matmul+silu_mul / QKV+RoPE / flash-attn+proj fusions (regressed).
- Larger M tiles (bm=128); LUT dequant; raising VRAM budget as a "fix".
- Q8_0 dequant_i8 pass-through specialization (bit-identical, zero gain).

## Conventions

- Rope: DiT interleaved `rope`; Qwen3 half-rot `rope_halfrot`.
- Matmul `a @ b`, B `[K, N]`; nn.Linear uploaded transposed; GGUF native.
- Z-Image text encoder stops at Qwen3 `hidden_states[-2]`.
- VAE: `(z/scaling) + shift`; SCALING=0.3611, SHIFT=0.1159; tile=64 ovl=8.
- WIP branch flow: `git add ... && git commit --amend --no-edit &&
  git push --force-with-lease`.
- `THINFER_TRACE=1` rollup + fmt stderr; `=verbose` adds span-close.

## Memorized commands

- **e2e parity q8 (with pyref)**: `THINFER_TRACE=verbose
  THINFER_E2E_PNG_DIR=/c/work/personal/thinfer/scratch/png_staging
  THINFER_POWER_PREF=high cargo test --release -p thinfer-conformance
  --features zimage-e2e e2e_parity_for_gguf_q8_0 -- --skip i8_sdpa
  --nocapture --test-threads=1 2>&1 | tee
  /c/work/personal/thinfer/scratch/smoke.log`
  (drop the `--skip` and use `..._i8_sdpa` / `..._q4_k_m` for variants;
  NOTE: `--exact` does NOT work with unqualified test names - it matches
  nothing and exits 0 vacuously. Verify the `e2e-parity[...]: starting`
  line + a non-zero passed count.)
- **Step-0 localization diag**: add `THINFER_E2E_STEP0_DIAG=1` (needs
  budget headroom; busts 2 GiB).
- **Conformance**: `cargo test --release -p thinfer-conformance
  --features conformance -- --test-threads=1`
- **768x768 (no pyref)**: add `THINFER_E2E_SKIP_PYREF=1
  THINFER_E2E_DIMS=768x768`.
- **Perf runs**: ALWAYS add `RUST_LOG="info,thinfer::diag=warn"` - the
  DIAG readback probes gate on the env filter, not the trace level, so
  any THINFER_TRACE setting fires them and serializes every block.
  `THINFER_E2E_BUDGET_GB=N` overrides the 2G default (residency A/B).
- Perf log series in scratch/: smoke-768-ring-baseline.log (submit ring,
  pre-vec4), current smoke.log = vec4 tile loads (q4_k_m 768).
  smoke-pyref-q8.log = post-vec4 pyref parity at 256.
- CLI wall reference: q4 768x768 8-step, 5G/5G budgets: 1m30.5s
  (2026-06-05, post-VAE-overhaul; was 2m11.8s post-vec4, ~2.5m before).
- CLI default model flipped q8 -> q4 (2026-06-05): visual quality
  confirmed acceptable (capybara/lighthouse 768 renders clean).
