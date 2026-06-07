# Worklog

## State (2026-06-06): Z-Image image gen COMPLETE, perf phase closed

Full text-to-image pipeline (Qwen3 encoder -> DiT -> VAE) on wgpu,
q4 default, 2GiB VRAM ceiling, all parity green:

- pyref q8 256: latent slope 1.0068 rel 2.38%, rgb 1.0026 / 2.37%.
- qwen3_parity final hidden 4.90% slope 1.0020 (gate 6% / 1+-0.02).
- TRUE_PEAK 2.00GiB exact at q8 + q4.

Sources: github.com/Tongyi-MAI/Z-Image (model code),
huggingface/diffusers ZImage pipeline, DiT GGUF
hf.co/unsloth/Z-Image-Turbo-GGUF (DiT only; encoder + VAE stay
safetensors from hf.co/tsqn/Z-Image-Turbo_fp32-fp16-bf16_full_and_ema-only).

## Baselines (post GPU weight prep, 2026-06-06)

- CLI gen 768x768 8-step q4: 86.1s in-process. Text encode 7.3s CLI /
  6.7-7.5s e2e. e2e q4 768 2-step: ~35s wall.
- Per-stage (e2e 768 2-step): setup/load ~9.4s, encode ~7.5s, DiT
  17.6s (~8.8s/step), VAE ~3.9s (1s/tile x4).
- DiT step is ~80% real matmul/sdpa GPU compute (~6.8s/step across 30
  blocks; ffn1 gate/up ~41ms each, ~5 TFLOPS effective). Submit stalls
  + upload churn are <5% combined: not worth attacking.
- Encoder remaining ~6.5s is cold-cache disk read at device bandwidth
  floor (~1.1-1.3GB/s, ~7GB/encode). Only fix is fewer bytes (Q8_0
  GGUF encoder checkpoint ~4.3GB vs 8GB bf16): needs user buy-in,
  not approved.

## Next: web phase

1. Simple HTML form for image gen (thinfer-web).
2. Playwright-driven e2e parity in the browser.
3. Re-profile on web: OPFS IO and wasm single-thread change the perf
   landscape; native conclusions (read-ahead rejection, encoder IO
   floor) do not transfer automatically.

Parked (revisit later, not web-blocking):

- CI `pnpm test:web` step is DISABLED (commented in ci.yml, 2026-06-06):
  hangs on all GH-hosted runners right after the Playwright chromium
  download, before any cargo output. Unresolved; suspects: GPU-less
  WebGPU bring-up, pnpm v10 blocking chromedriver's postinstall binary
  download. Run `pnpm test:web` locally before merging web changes (it
  catches real naga-vs-Tint and wasm-runtime divergences: caught 3 on
  2026-06-06). Adapter acquisition now has a 15s loud-failure timeout;
  runner output is always --nocapture; CI-only swiftshader flag wired
  in scripts/test.ts.

- DiT 2x: only real lever is matmul kernel throughput. Subgroup-matrix
  (coop-mat) is experimental/platform-gated and useless in browsers,
  so it conflicts with the browser goal; subgroup-intrinsic Q4_K_M
  dequant is the portable candidate.
- Quantized Qwen3 encoder checkpoint on disk (halves encode read).
- Cleanup: tighten e2e tolerances to measured baselines; sdpa_i8
  K-smoothing if noisy; delete or keep dead `decoder_forward`.

## Invariants (current architecture)

- VRAM budget has one owner: `MemArbiter` (thinfer-core/src/arbiter.rs),
  shared into every `Workspace`. Reclaim chain: idle workspace pool ->
  evictable weights -> unpinned ring slots. Lock order arbiter -> client.
  Budget is a ceiling target; e2e TRUE_PEAK assert catches overshoot.
- ActDtype::I8 is matmul/sdpa-internal ONLY (`BlockWgslConfigs::validate`
  asserts). Residual carry, norms, glue: dense at act dtype.
- `dispatch_matmul_site` accepts dense (act_quant inside, DP4A) or
  paired A-side; `dense_acts` opts a site out at compile time.
- Weight transcode: `WeightMeta::transcode = Some(Q8_0)` requantizes
  bf16 at upload. Qwen3: 6 of 7 sites; `mlp_down` NEVER (massive
  activations). Embedder/final_layer pure bf16.
- GPU weight prep: transcode + Linear2D transpose on-GPU at upload
  (`Backend::weight_prep`); CPU read path is the mock-backend fallback.
  Q8_0 transcode value-equivalent NOT bit-exact vs CPU (+-1 tie flips
  on ~0.03%, llama.cpp-normal).
- act_quant pipeline built when (any i8 site) || i8_sdpa.
- No eprintln in library code; stage timing tracing::info; diag dumps
  target=DIAG tap-gated. No env reads in thinfer-core.

## Locked design decisions

- i8 activation storage matmul/sdpa-internal only (outlier channels).
- Qwen3 `mlp_down` weights stay bf16.
- DP4A matmul auto-on for Packed4x8 + SHADER_F16 + F16 acts + Quant
  weight; tile bm=bn=64 tm=tn=4; subgroup runtime-branch.
- Sdpa flash-attn small-D: `SdpaF16Sg` BR=16 BC=32 WG=128 CL=8.
- Kernel register rule: no dynamically-indexed local arrays in hot
  loops; unroll via codegen.
- Four pipeline-set split (main / encoder / dit_encoder / dit_embedder).
- GGUF parser range-fetch-first; B viewed [N, K] N-major.
- PowerPreference::HighPerformance CLI/e2e default.
- VAE per-resnet sub-submits, single submit per tile; SUBMIT_DEPTH=2;
  WEIGHT_RING_SLOTS=4; matmul WGSL <= 32KiB workgroup storage.

## Tested + rejected (do not retry)

- Kernel read-ahead hints for weight reads: IO is bandwidth-bound, not
  queue-depth-bound; pure overhead. Re-evaluate on web OPFS only.
- `read_at` straight into mapped staging: WC write-combining collapses
  (0.8-1.7GB/s vs 9.5GB/s two-hop). Applies to any direct-from-mmap
  fill of mapped GPU memory.
- BW/dispatch fusions on ALU-bound kernels (dual-matmul+silu_mul,
  QKV+RoPE, flash-attn+proj): all regressed.
- i8 residual carry / i8 elementwise acts; BC=64 sdpa (occupancy);
  bm=128 tiles; LUT dequant; Q8_0 dequant_i8 pass-through.
- Mapped-staging same-encoder copy; dedicated submit per upload;
  SUBMIT_DEPTH=3; static workspace reserve; weight prepack; derived
  on-disk transcode caches (ask first, case-by-case).
- Raising VRAM budget as a perf "fix".

## Conventions

- Rope: DiT interleaved `rope`; Qwen3 half-rot `rope_halfrot`.
- Matmul `a @ b`, B `[K, N]`; nn.Linear uploaded transposed; GGUF native.
- Z-Image text encoder stops at Qwen3 `hidden_states[-2]`; pads odd
  prompts to even seq.
- VAE: `(z/scaling) + shift`; SCALING=0.3611, SHIFT=0.1159; tile=64 ovl=8.
- WIP branch flow: `git add ... && git commit --amend --no-edit &&
  git push --force-with-lease`.
- `THINFER_TRACE=1` rollup + fmt stderr; `=verbose` adds span-close.

## Diag / telemetry (opt-in, use it instead of guessing)

- qwen3_parity self-localizes: per-layer linfits -> suspect layer ->
  auto pyref per-op dumps (`gen_qwen3_parity_ref.py --tap-layers`,
  cached in target/tmp/qwen3_parity/) -> engine `Qwen3Taps::tap_block`.
- Residency acquire emits `thinfer::diag` per-weight read/prep events.
- `THINFER_E2E_STEP0_DIAG=1`: DiT step-0 taps (~300MiB, busts budget).
- Conformance fixtures cached; THINFER_CONFORMANCE_REGEN=1 forces.

## Memorized commands

- **qwen3 parity (encoder localization)**: `THINFER_TRACE=1
  THINFER_POWER_PREF=high cargo test --release -p thinfer-conformance
  --features zimage-e2e qwen3_parity -- --nocapture --test-threads=1
  2>&1 | tee /c/work/personal/thinfer/scratch/qwen3-parity.log`
- **e2e parity q8 (with pyref)**: `THINFER_TRACE=verbose
  THINFER_E2E_PNG_DIR=/c/work/personal/thinfer/scratch/png_staging
  THINFER_POWER_PREF=high cargo test --release -p thinfer-conformance
  --features zimage-e2e e2e_parity_for_gguf_q8_0 -- --skip i8_sdpa
  --nocapture --test-threads=1 2>&1 | tee
  /c/work/personal/thinfer/scratch/smoke.log`
  (variants: `..._q4_k_m`, `..._i8_sdpa`. `--exact` does NOT work with
  unqualified names: matches nothing, exits 0 vacuously. Verify the
  `e2e-parity[...]: starting` line + non-zero passed count.)
- **768x768 (no pyref)**: add `THINFER_E2E_SKIP_PYREF=1
  THINFER_E2E_DIMS=768x768`.
- **Conformance**: `cargo test --release -p thinfer-conformance
  --features conformance -- --test-threads=1` (fixtures cached).
- **Perf runs**: ALWAYS add `RUST_LOG="info,thinfer::diag=warn"` (DIAG
  probes gate on env filter; firing them serializes every block).
  `THINFER_E2E_BUDGET_GB=N` overrides the 2G default.
