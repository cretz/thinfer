# Worklog

## State: Z-Image t2i complete (native + web)

Full text-to-image (Qwen3 encoder -> DiT -> VAE) on wgpu, native + browser
(wasm). q4 default, all parity green. Native q8 256 pyref bit-clean (rgb slope
1.003478 / 2.22% with pinning). Quant variants take the COMPLETE DiT+TE from
GGUFs (q4 9.5GB / q8 14.7GB / bf16 20.5GB). Sources: Tongyi-MAI/Z-Image,
diffusers ZImage pipeline, DiT GGUF unsloth/Z-Image-Turbo-GGUF (Turbo =
distilled few-step), TE GGUF worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF,
VAE safetensors upstream.

## NEXT

DiT per-step time WAS the target; the compute levers are now mined out. The
whole pipeline is COMPUTE-BOUND on every platform (warm step = GPU fence =
matmul/sdpa; uploads + weight-prep hide behind compute on the single queue).
Proven by a pin-vs-no-pin A/B (2026-06-13, a temporary set_pin_disabled knob,
since reverted): forcing every weight to re-stream each step left fence_ms flat
while upload bytes jumped. Desktop q4 768 4G: 1570 -> 3397 MiB/step (2.2x),
fence 3399 -> 3404. So IO is NOT a lever: OPFS read-path work, the JS-heap pin
tier, and bigger budgets are DEAD ENDS. Mobile is just a slow iGPU (VAE 5.4x
desktop) and rides every desktop sdpa/matmul win directly; the only mobile
curve-benders are product-shaped (lower res, compute ~ seq^2; fewer steps,
Turbo is distilled).

SDPA IS MINED OUT (2026-06-13). The ORT `prefer_subgroupshuffle` rewrite was
the last untried compute lever; it was a wash (see rejected). All three sdpa
attempts (FA2 softmax, f16 accum, ORT shuffle) now wash/regress. Per-step
gpu_disp baseline (q4 768 native, post-ORT-matmul): sdpa 1363ms, ffn1 1070,
qkv 726, ffn2 657, proj 225, refiner ffn 119. The image generator is
feature-complete and perf-mined; remaining work is wrap-up and the secondary
below, not another kernel attack.

NEXT ACTION: put a bow on the image generator (cleanup, docs, any final
polish). No more sdpa/matmul kernel attempts - both mined out.

Secondary (still open): web text encode 3436ms (was 2434) on one post-rebuild
run; likely one-off Tint/Dawn recompile (native flat). Confirm on a warm-cache
run. This is the only DiT-adjacent perf item left, and it's a measurement
confirm, not a rewrite.

## Invariants

- VRAM budget single owner `MemArbiter` (arbiter.rs), shared into every
  `Workspace`. Reclaim order: idle pool -> evictable weights -> unpinned ring
  slots. Lock order arbiter -> client. Hard ceiling; e2e TRUE_PEAK asserts it.
- Fill-to-budget pin plan: `set_pin_plan` pins a `pin_priority` prefix up to
  `budget - ring_reserved - workspace_reserve - staging_reserve`, installed in
  `denoise_with` after assemble ("dit pin plan" line). Pin-on-first-touch:
  pinned handles use the pool path, skip recycle + reclaim (freed only as a
  3rd-pass last resort). `evict_all_and_free` drops the plan at denoise end so
  VAE reclaims. `pin_priority` role-major/block-minor, largest roles +
  refiners/embedders first. `set_pin_disabled` forces it off (A/B knob).
- ActDtype::I8 is matmul/sdpa-internal ONLY (`BlockWgslConfigs::validate`);
  residual/norms/glue dense at act dtype. Qwen3 `mlp_down` stays bf16.
- `dispatch_matmul_site`: dense (act_quant inside, DP4A) or paired A-side;
  `dense_acts` opts a site out at compile time.
- Weight transcode `WeightMeta::transcode=Some(Q8_0)` requantizes bf16 at
  upload (Qwen3 6/7 sites; `mlp_down` NEVER). Value-equiv NOT bit-exact vs CPU.
  GPU weight prep (transcode + Linear2D transpose) on-GPU at upload.
- No eprintln in lib; stage timing tracing::info; diag dumps target=DIAG
  tap-gated. No env in thinfer-core (binary edge reads env, passes config).

## Locked design decisions

- DP4A matmul auto-on for Packed4x8 + SHADER_F16 + F16 acts + Quant weight.
  ORT register-resident subtiles (tile=64, 256 threads, lane owns 1 A row + 16
  cols, hardware-broadcast B shared reads, fused scale fold; shared layout
  [kv][row]). q8 256 pyref BIT-IDENTICAL. subgroups OFF: shuffle path kept in
  kernel but ~30% loss on NVIDIA sg=32 (vec4 shuffle = 4 SHFL/dp4a vs free
  broadcast); only an Intel sg=16 per-vendor candidate.
- sdpa flash small-D `SdpaF16Sg`: BR=WG/CL, BC=32, WG=128, CL=min(8,sg_min)
  (native 8, web/mobile 4). The ~2.3x/step lever; needs web subgroups (vendor).
- No dynamically-indexed local arrays in hot loops; unroll via codegen.
- Four pipeline-set split (main / encoder / dit_encoder / dit_embedder).
- GGUF parser range-fetch-first; B viewed [N,K] N-major; matmul `a@b` B [K,N].
- PowerPreference::HighPerformance default (CLI/e2e/web).
- VAE per-resnet sub-submits, single submit per tile. DiT submit ring REMOVED:
  each block awaits its own fence joined with next-block load.
  WEIGHT_RING_SLOTS=4; matmul WGSL <= 32KiB workgroup storage.
- Quants Q4_K_M + Q8_0 (+ bf16 fallback): others no perf win (compute-bound on
  dequant+MAD, not format), Q6_K only a download-size trade; matches ComfyUI.

## Tested + rejected (do not retry)

- sdpa f16-math dot+accumulate: -1.2%/step only AND f16 o-accum compounds over
  keys (blocky grain at 768; pyref-256 can't catch it). NOT ALU-rate-bound.
- sdpa FA2-style tile softmax restructure: +20% native, web flat. Per-key
  softmax bookkeeping is NOT the bottleneck. (Op test kept: tests/sdpa_sg.rs.)
- sdpa ORT `prefer_subgroupshuffle` shape (2026-06-13, reverted): 1 lane = 1
  query row, K/V staged then distributed across the subgroup via subgroupShuffle
  (replacing the CL-cluster D-split + shuffleXor reduce), fp32 accum (NOT ORT's
  f16 - kept fp32 to dodge the f16-grain class above; 768 PNG confirmed clean),
  chunk-wise softmax, head-vec unrolled, chunk width from runtime subgroup_size
  builtin (drops build-time CL). Correct (conformance green; q8-256 slope 1.0044
  vs 1.0035). Perf WASH: q4 768 attn_sdpa 73.7->75.8 ms/block, but EVERY
  unchanged op also drifted (qkv +12%, ffn2 +7%, ffn1 +1%) = global clock/
  thermal drift, not a real sdpa change; sdpa moved less than unchanged qkv.
  CAVEAT: A/B confounded - baseline ab-pin.log was a different session/logging,
  not a same-session old-vs-new. A truly clean verdict would re-run both kernels
  back-to-back; given two prior sdpa washes too, called mined out rather than
  chase it. (Source on the dead branch / git history; ORT template at
  third-party/onnxruntime/.../bert/flash_attention.wgsl.template.)
- matmul bk_step=2 (llama.cpp mmq): +8-12% (shmem vs occupancy).
- matmul 4-bit B end-to-end: +8% (inner shift+mask eats the traffic save).
  Confirms matmul is ALU/occupancy-bound, not B-traffic-bound. (4-bit halves
  transient B VRAM; revisit only if pinning ever needs VRAM, not for speed.)
- matmul subgroup loads (both select() and shuffle forms): ~30% NVIDIA loss.
- BW/dispatch fusions on ALU-bound kernels (dual-matmul+silu_mul, QKV+RoPE,
  flash-attn+proj): all regressed.
- IO: native never disk-bound (page cache serves at memcpy speed; e2e
  read_mbps is ring-consumption-paced + overlap-inflated). `read_at` into
  mapped staging (WC collapse 0.8-1.7 vs 9.5 GB/s). Host-RAM weight cache
  (page cache is free). Host threads / disk caches / JS-heap pin tier on web.
  Kernel read-ahead hints. Raising VRAM budget as a "fix".
- Misc: i8 residual/elementwise; BC=64 sdpa; bm=128 tiles; LUT dequant;
  dedicated submit per upload; SUBMIT_DEPTH ring; static workspace reserve;
  weight prepack; derived on-disk transcode caches (ask first, case-by-case).

## Web specifics

npm lib `thinfer` (TS over wasm) + example `examples/gen-image/` + Playwright
parity. `bindings.rs`/`weight_file.rs` over JS `WeightFile`; `ZImageSource`
single-sources the recipe for CLI/web/e2e. OPFS IO `opfs.ts`/`opfs-worker.ts`:
`OpfsWeightCache {dirName?, io?: auto|worker|inline}`. CSP needs
`worker-src 'self'`.

- No DOM types in lib/API. No lazy downloads (caller downloads, else
  `loadModel` throws). `pnpm build` does NOT run wasm-opt (binaryen absent).
- `setTraceLevel` (default off): info = milestones + rollups; debug = per-weight
  residency + join split; trace = per-dispatch + DIAG readbacks.
- Teardown RAII -> `WgpuBackend::drop` -> `device.destroy()`.
- OPFS `put`: sync-handle coalesced 16 MiB writes, visibility via `<file>#ok`
  marker. Read locks held only during an op (released in `finally` after
  loadModel + each generate). Two tabs at once collide ("locked by another
  context"); accepted. Desktop write ~3 GiB/s, mobile ~1 GiB/s.
- KNOWN FAILURE `pnpm test:web` ("Failed to detect test as having been run"
  since bindings); CI disabled (hangs on GH runners). Run locally before merge.
- Web caps maxBindingSize/workgroup-storage 128MiB / 16KiB (downlevel); matmul
  builds 16KiB-fit. >128MiB single binds (DiT FFN 150MiB) parked (chunking).
- Vendored wgpu facade `projects/vendor/wgpu-29.0.3/` (web subgroups): web-sys
  omits the `subgroups` GpuFeatureName + hardcodes subgroup size to spec floor
  4, both blocking subgroup shaders. Fix = vendor ONLY the `wgpu` facade,
  `[patch.crates-io]`, two `// THINFER-PATCH(web-subgroups #5555)` edits. Our
  `backend.subgroup_enable_directive()` prepends `enable subgroups;` to sdpa
  WGSL on web (Tint needs it, naga rejects). Re-check #5555/#8202 each wgpu
  bump; delete if upstream ships web subgroups. Needed for the sdpa sg lever.

## Conventions

- Rope: DiT interleaved `rope`; Qwen3 half-rot `rope_halfrot`.
- nn.Linear uploaded transposed; GGUF native. TE stops at Qwen3
  `hidden_states[-2]`; pads odd prompts to even seq.
- VAE: `(z/scaling)+shift`; SCALING=0.3611, SHIFT=0.1159; tile=64 ovl=8.
- WIP branch flow: `git commit --amend --no-edit && git push
  --force-with-lease`, terse messages. `THINFER_TRACE=1` rollup; `=verbose`
  adds span-close. fmt + clippy (all warnings) after edits.
- Validation order: op conformance (`--features conformance`) -> q8 256 pyref
  e2e -> q4 768 skip-pyref perf e2e. Serial, NEVER parallel GPU runs.

## Commands (scratch/ is outside the repo, gitignored)

- **web**: `cd thinfer-web && pnpm build` (wasm + tsc, NO wasm-opt), user runs
  the server; report browser numbers back. Debug trace, few steps.
- **e2e parity** (PNG dir env always): `THINFER_TRACE=verbose
  THINFER_E2E_PNG_DIR=scratch/png_staging THINFER_POWER_PREF=high cargo test
  --release -p thinfer-conformance --features zimage-e2e
  e2e_parity_for_gguf_q8_0 -- --skip i8_sdpa --nocapture --test-threads=1`
  (variants `..._q4_k_m`, `..._i8_sdpa`; `--exact` does NOT work; verify the
  `starting` line + non-zero passed count).
- **768 perf (no pyref)**: add `THINFER_E2E_SKIP_PYREF=1
  THINFER_E2E_DIMS=768x768`.
- **perf runs**: add `RUST_LOG="info,thinfer::diag=warn"` (debug serializes
  every block via DIAG probes; use `=debug` only when you need join-split).
  `THINFER_E2E_BUDGET_GB=N` overrides the 2G default.
- **qwen3 parity** (encoder localization): `... qwen3_parity -- --nocapture
  --test-threads=1`.

## Diag / telemetry (opt-in)

- `step done` / `denoise done` / `dit.layers timing` (info): streamed,
  read_mbps, encode/stream ms. debug adds `join split` (fence/acq/pf; fence ~=
  stream + tiny acq/pf = GPU-queue-bound) + per-weight `stream read`
  (overlap-inflated, NOT additive). pf_ms is a finish marker, not a cost.
- `THINFER_E2E_STEP0_DIAG=1`: DiT step-0 taps (~300MiB, busts budget).
- qwen3_parity self-localizes per-layer -> pyref per-op dumps -> engine taps.
- Conformance fixtures cached; `THINFER_CONFORMANCE_REGEN=1` forces.
- NOTE: laptop thermals throttle all clocks uniformly when vents blocked;
  ~2x uniform slowdown is airflow, not code.
