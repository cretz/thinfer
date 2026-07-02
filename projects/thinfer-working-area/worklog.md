# Worklog

Forward-looking only. Git history is the changelog, the code is the record.
Engine-wide design: `plan-details.md`. Per-model plans are separate files (see
Status). Scratch is ephemeral; nothing here depends on a scratch file.

## NOW: causal TI2V (`hunyuan-video-1.5-ti2v`, see `hy15-causal-plan.md`)

Model is SHIPPED + serve-deployed: minWM WorldPlay `HY15/TI2V/dmd`, 4-step
chunked-AR 8B HY1.5. Image OPTIONAL (with = I2V; without = TEXT-ONLY, validated
sharp/coherent at 480p -- this is the fast-T2V play). Health gate green both
modes. Naming follow-up: consider a `-distilled` suffix in the id (user, low).

**>>> BUG FIRST (before perf): progressive AR drift.** User's 77f (5-chunk)
browser run: chunk 1 GORGEOUS (dancing dog, better than the old T2V eyeballs),
then saturation/contrast blows up chunk-over-chunk into chaos by chunks 4-5.
2-chunk runs only hint at it (chunk 2 slightly punchier). Mechanism: the AR
loop feeds each chunk's outputs back through the KV cache, so per-chunk
numerical error COMPOUNDS (the T2V has no feedback loop). **BISECTED: pure
bf16 (`THINFER_HY_I8=0 THINFER_NO_COOPMAT=1`) at 448x256x77f (5 chunks) is
CLEAN through the last frame (`scratch/drift_bf16_png/`), so the amplifier is
QUANTIZATION NOISE (i8 DP4A qkv/ffn_up + Q8-coopmat proj/ffn_down + f16 SDPA)
recycling through the cache -- NOT a structural AR bug.** Fix directions, in
preference order: (1) find WHICH quant site dominates (bisect i8-only vs
coopmat-only vs f16-sdpa-only at 5 chunks) and disable just that site for this
model; (2) recache-forward-only at full precision (the cache entries are what
compound; denoise forwards can stay quant) -- likely the cheapest good fix;
(3) all-bf16 for this model (works but slower + more streaming). Caveat: the
bf16-clean run was tiny-dims; confirm the chosen fix at 832x480x77f. Also
check upstream `stabilization_level > 1` (their knob for this drift class).
Repro frames from the user's mp4: `scratch/crazy_frames/`.

**It is currently WEIGHT-STREAMING BOUND, not compute bound: ~11min for 29f
(2 chunks) at 480p. Each of the `chunks*(4+1)` AR forwards re-streams all 54
blocks (~13.5GB mixed Q8/bf16) for only 6,240 rows of compute; the T2V's single
32,760-row forwards hide the same stream, chunk forwards can't. Attack in this
order:**

1. **Skip the txt-side weights in chunk forwards.** `forward_chunk` acquires
   whole blocks but uses ONLY the img-side (img_mod/q/k/v/proj/fc1/fc2 + norms);
   txt weights are needed once (txt pass). ~2x per-forward traffic for free.
2. **Q8 the modulation linears.** img_mod/txt_mod are `[12288, 2048]` = ~2.7B
   of the 8.3B params, currently dense bf16 (Module site). Dequant-once or
   direct-quant path cuts ~2.7GB/forward more.
3. **Pin a deterministic block prefix resident; stream only the tail.**
   Sequential LRU over a working set bigger than budget = worst-case thrash;
   the arbiter reclaimer (now registered) makes residency safe, but a fixed
   pin-split avoids re-uploading the SAME evicted blocks 25x.
4. **GPU F32->Q8_0 transcode prep kernel.** The 33GB F32 source transcodes on
   CPU per request (per-request isolation, no disk cache) -- minutes of cold
   start. A NarrowQ8 prep kernel (like `NarrowTransposeF32`) removes it.
5. **i8 KV cache** (store + upload i8 instead of bf16): halves the ~14GB host
   KV + its per-forward PCIe. Quality-gate (K/V are post-norm/rope, f16-safe;
   i8 unproven).
6. Gated deviation lever (later): skip the recache forward by caching the last
   denoise step's K/V (-20% forwards, trained-behavior deviation -> eyeball gate).

**T2V (non-causal) perf plan, still open:** perf-harness `gpu_ms`-by-pipeline
rollup on ONE DiT step to nail the SDPA-vs-matmul split, then i8/DP4A SDPA
(`cfgs.i8_sdpa` exists, unproven; gate on dit_parity bands) and DiT step/block
caching (gate). All ship as opt-in user options unless imperceptible.
Windowing stays opt-in (W=3 broke multi-subject coherence; default reverted to
full attention 2026-07-01). Rewrite perf (after DiT): condense the ~5.8k-token
system prompt (option + gate) and causal-aware prefill SDPA (quality-neutral,
default on). Reality check: seconds-at-quality is a HARDWARE ceiling on the
28-SM 8GB mobile 5070; kernel work buys ~2-3x, not 100x.

**Research watch:** no 1-2 step HY1.5 T2V distill exists (CF++ 1/2-step = Wan
2.1-1.3B only, below the quality bar -- user decision; minWM HY15 line is
4-step). Watch thu-ml/Causal-Forcing + MIN-Lab; the AR machinery here is
exactly what a few-step causal HY15 checkpoint would need.

## Lessons / dead-ends (do not retry)

- **Weight transcode must catch every float source encoding.**
  `register_linear_transcode`'s Q8 arm matched Bf16 ONLY; F32/F16 fell through
  to a DENSE registration while the site's quant pipeline read Q8 -> garbage
  (F32 minWM DiT; the T2V "fp16" file is actually BF16 so it never showed).
  Fixed. Diag: garbage q/k look CLEAN after qk-RMSNorm; probe V/proj outputs.
- **Budgets: stream in/out under pressure, never predict reserves** (user,
  firm). Register `residency.reclaimer()` at `RECLAIM_EVICTABLE_WEIGHTS` on the
  arbiter so workspace growth evicts unpinned weights. The old Hunyuan
  carve-out reserves are gone; don't reintroduce.
- **Coopmat is a MATMUL-ONLY win.** Flash-attn coopmat SDPA measured 13x SLOWER
  than `sdpa_sg` (already ~12 TFLOPS-eff). cross-qkv coopmat DEVICE-LOSES
  (un-normed text > f16 65504). Coopmat also device-loses for M < WM(32); the
  `m >= wm` dense-fallback guard in `dispatch_matmul_site_coopmat` stays.
  naga gotcha: `var c: coop;` in a loop null-inits ONCE at fn entry.
- **DiT denoise is at the WGSL matmul ceiling** (latency/occupancy-bound, not
  bandwidth); bigger tiles BACKFIRE; weight-only quant does NOT speed compute.
  Measure via e2e `gpu_ms by pipeline`, NOT microbench.
- **Q8_0 is the quality+perf baseline for big DiTs** (Q4_K per-request fold was
  ~2x SLOWER and broke quality). Q4_K_M = footprint option only.
- **i8 DP4A**: qkv/ffn_up only (normed A-sides). proj/ffn_down carry outliers
  -> f16-cast coopmat OK, i8 acts NOT. Cross-attn-qkv from un-normed text: never.
- **BLOCK-WIDE f16 is a DEAD END** on big-DiT residuals (outlier channels >
  65504); bf16 residual is load-bearing. Mixed-precision f16 SDPA (post-norm
  Q/K/V cast only) is safe and shipped.
- **VAE decode is conv-GPU bound**; do NOT retry the conv3d im2col hoist
  (REVERTED, slower). Tiny (TAEHV) decoders are the fast default where they
  exist; LTX tiles its VAE (activation-bound peaks; seed BELOW budget, balanced
  re-seed on OOM).
- **Never run fp32 CPU pyref above tiny dims** (40GB host at 256x256).
- **LTX**: off-subject output = resolution/adherence regime, NOT text encoding
  (512+ two-stage widescreen is the regime). Gemma encoder MUST run F32 acts.
  The gemma `(1+w)` norm bake is REAL, don't "fix" it. Don't flip the LTX DiT
  to strict budget (relies on overshoot into device slack).

## Carry-forward gotchas (engine-general)

- **Ops reading aux params as f32 must dequant bf16 weights first** (f32
  binding fed bf16 bytes = silent garbage; conformance has no such dtype pair,
  so it passes conformance and fails in the model).
- **Q8_0 subnormal f16 scale bug is FIXED** (regression test exists); affects
  any tiny-weight quant tensor on the bf16-dequant path.
- **umT5 / large-residual encoders MUST run bf16 acts** (f16 overflows ->
  washed blob). Check non-finite, not just NaN.
- **Module-level matmuls need their OWN bf16 `matmul_module` pipeline**; never
  quantize a weight whose matmul site has no dequant step.
- **RoPE**: freqs pack to the act dtype; Wan RoPE3D = interleaved-pair, Qwen3 +
  LTX = half-rot. Hunyuan = interleaved-pair, img tokens only; AR chunks rope
  at ABSOLUTE frame positions.
- **GGUF tensor padding**: F32/F16 narrow arms slice `elements*size`, not
  `on_disk_bytes`.
- **Third-party clones**: `rm -rf <clone>/.claude` right after cloning.
- Video staging: per-frame PNG seq / contact sheet; MP4 in CLI only.

## Status (shipped -- DO NOT DISTURB)

- **hunyuan-video-1.5-ti2v** (causal AR, minWM dmd) -- ACTIVE, see
  `hy15-causal-plan.md`.
- **HunyuanVideo 1.5 T2V** (lightx2v 4-step, 480p) -- shipped; native 4B/8B
  prompt rewriter, tiny-ft VAE, joint windowed SDPA (opt-in), cancel wiring.
- **LTX-2.3 distilled + Sulphur-2** (22B joint AV) -- shipped, `ltx-plan.md`.
- **FastWan2.2-TI2V-5B** -- parity GREEN; UniPC default. PENDING user eyeball
  of a UniPC clip vs the KingNish Space.
- **LongLive-2.0-5B** (AR long/multi-shot) -- shipped. OPEN: AR perf wins
  (upload window prefix once/chunk; cache cross-attn text K/V) -- re-measure
  WARM at 576 first.
- **Face-swap** -- shipped; NEXT = quality (XSeg + GFPGAN), `faceswap-plan.md`.
- **Ideogram-4**, **Z-Image**, **Qwen-Image(-Edit)-Rapid** -- shipped.
- **Wan2.2-T2V-A14B** -- shipped; attn-window default W=3 (long clips).
- **i8 DP4A matmul ON by default** (`--no-i8-matmul` = bf16 reference).

## Crate layout + serve (shipped reference)

- Layering: core < models (dep-clean) < native < app < {cli, serve}; wasm
  `thinfer-web` is its own substrate. serve = separate binary over shared
  `thinfer-app` (`JobExecutor` trait). Web UI is `include_str!`-baked -> web
  edits need a serve rebuild. One fresh wgpu device per JOB.
- **Deploy = I do it myself** (ASK before stopping the server he toys with):
  stop -> `cargo build -p thinfer-serve --release` -> `Start-Process` the exe
  with `-ArgumentList '--config','C:\work\personal\thinfer\scratch\serve.toml'`,
  `-WorkingDirectory <projects>`, stderr -> `scratch/serve.log`, stdout ->
  `scratch/serve.stdout.log`, `-WindowStyle Hidden`. Confirm "listening
  (https)" in the log. NEVER run the exe in a foreground Bash call. rustls is
  TLS-1.3-only; the browser is the check. serve.toml now sets `ram_budget=28G`
  (TI2V host KV cache ~14GB at 77f).
- DEFERRED: serve==CLI byte-parity test; disk-backed SSE ring buffer.

## Running the e2e / measuring

Card = RTX 5070 Laptop (8GB); keep budgets <8GB. All serial
(`--test-threads=1`). Always `THINFER_TRACE=1 THINFER_POWER_PREF=high` +
`THINFER_E2E_PNG_DIR` for staging; read the `gpu_ms by pipeline` rollup first.
- Causal TI2V: `cargo test -p thinfer-conformance --features hunyuan-e2e
  --release i2v_e2e_health -- --nocapture --test-threads=1`. Default 448x256x13
  (one chunk, minutes); `THINFER_E2E_{WIDTH,HEIGHT,FRAMES,VRAM_GB}` scale to
  product dims. `THINFER_I2V_T2V_PROBE=1` = text-only mode.
  `THINFER_AR_DIAG=1` = per-stage stats; `THINFER_HY_I8=0` bisects i8.
- Hunyuan T2V: `t2v_e2e` (pyref parity, tiny dims), `t2v_perf` (engine-only
  per-step bench, `--attn-window` sweep), `dit_parity` taps,
  `rewriter_caption_compare` (`--features qwen3-lm`, `--ignored`).
- FastWan: `video_e2e` (256x256, budget 6GB; `THINFER_E2E_SKIP_PYREF=1` for
  perf-only). LongLive: `longlive_parity`/`longlive_e2e` (256x256).
- LTX: `t2v_e2e_health` (`--features ltx-e2e`, 121f 512x320 6GB); component
  parity gates ONE AT A TIME (12.5GB encoder OOMs a multi-test binary).
- GGUF inspect: `uv run --with gguf python` + `gguf.GGUFReader`.
- CLI run: `THINFER_TRACE=1 THINFER_POWER_PREF=high thinfer generate ...
  --vram-budget 5G --ram-budget 5G` (TI2V needs bigger ram for the KV cache).
