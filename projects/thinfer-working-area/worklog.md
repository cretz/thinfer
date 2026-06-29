# Worklog

Forward-looking only. Git history is the changelog, the code is the record. Past
work appears here ONLY as a one-line lesson or a do-not-retry. Engine-wide design
+ kernel/runtime state: `plan-details.md`. Per-model porting plans are separate
files (see Status). Scratch is ephemeral; nothing here depends on a scratch file.

## NOW / NEXT

**>>> TOP PRIORITY (post-/clear): 81f W=3 is CO-DOMINATED by self-attn AND matmuls;
attack BOTH. FIRST fix the telemetry gap (it misled the analysis once already).** The
81f gpu_ms rollup OMITS `sdpa_sg_win` (telemetry gap below) so it FALSELY reads as
matmul-bound. By wall-clock subtraction (1544s wall - ~893s timed - host) the windowed
self-attn is **~480s = STILL the single biggest term** at W=3. Real ranked picture:
self-attn ~480s > matmul_qkv 250s > coopmat ffn_down+proj 227s > i8 self-qkv+ffn_up
(+dequant) ~175s > VAE ~148s > cross sdpa 19s. So two co-equal levers:
- **(a) matmul: SPLIT cross-attn qkv** (the untried win). `matmul_qkv` 250s is almost
  entirely the cross-attn **Q projection over all 32760 latent rows**, forced DENSE
  bf16 (dense_acts.qkv=true) ONLY because the text K/V are un-normed outliers. But the
  Q is NORMED (i8-safe). Split like `qkv_self` was: i8 the big cross-Q, keep the tiny
  (512-row) cross-K/V dense. i8 DP4A ~5x -> 250s -> ~70s, save ~180s (~3min). (Check:
  module matmuls patch/proj_out are SEPARATE sites, not in matmul_qkv.)
- **(b) self-attn: a tighter window.** Self-attn ~480s scales with (2W+1)/21: W=2
  ~350s (-130s), W=1 ~210s (-270s). REAL e2e lever, gated on quality (W=2 = +-8 output
  frames context; W=1 = +-4, likely too tight for a 5s pan). Needs a quality eyeball
  per W. (User eyeballed W=3 = good; W=2 untested.)
- **REALISTIC FLOOR: (a)+(b@W=2) ~= 26min -> ~20-21min.** NOT minutes. WGSL has no
  tensor cores (lessons) -> ComfyUI/CUDA minutes is unreachable here; ~20min is ~the
  81f@832x480 14B floor on the 8GB card. Set expectations accordingly.

**TEMPORAL WINDOWED SELF-ATTENTION = LANDED + VALIDATED + quality-confirmed
(2026-06-28, committed 229af27 + pushed).** Opt-in `--attn-window W` (latent-frame
radius); breaks the O(frames^2) self-attn wall. Frame-major `(f,h,w)` layout ->
frame = token/period, period = pph*ppw. Conformance bit-exact vs a windowed CPU ref
(incl chunked row_off): `tests/sdpa_sg.rs::sdpa_sg_windowed{,_chunked}`. A/B 256x256/33f
(ppf=9,W=1): self-attn **36.0->11.8 ms/disp = 3.05x** (= F/(2W+1)). **81f@832x480 W=3:
1544s (~25.7min) vs ~40min full = 1.55x e2e, USER EYEBALLED W=3 = GOOD** (clip
scratch/win81_w3.mp4). **Default ON at W=3 for Wan2.2-14B** (`VideoModelId::
default_attn_window`, applied in executor.rs = single source for CLI+serve; explicit
`0` = full attention; long-clip tiled path only). **CLI + serve wire + web UI all
wired + deployed** (server rebuilt + running, attn-window control prefilled 3, revealed
for Wan22).
- **WHY only 1.55x e2e (not the 10x ask): windowing 3x'd self-attn but self-attn was
  ~half the cost, so e2e ~= 1/(0.5/3 + 0.5) = 1.6x** -- matches. At W=3 self-attn is
  STILL the top term (~480s); a tighter W + the cross-Q split (above) stack toward the
  ~20min floor. Order-of-magnitude is NOT on the table in WGSL (no tensor cores).
- **W=1 vs W=3 (projected; only W=3 measured @81f): self-attn 7/21 -> 3/21 = ~480s ->
  ~206s, e2e ~1544s -> ~1270s (~21min). Only ~4.5min between them = diminishing
  returns** (the ~800s matmul+VAE floor doesn't move). W=1 = +-4 output frames context
  (likely seams on a 5s pan) -> W=3 is the right default; don't drop lower blind.
- **TELEMETRY GAP (follow-up): `sdpa_sg_win` is in the per-scope dispatch counts
  (228/block) but ABSENT from the `gpu_ms by pipeline` TIMESTAMP rollup at 81f** (it
  IS present at the small 256x256 run). Likely the chunked windowed self-attn (57
  flush-per-chunk submits at 32760 rows) outruns the timestamp-query capture. So the
  exact 81f self-attn ms is unmeasured directly; the e2e wall + small-scale 3.05x +
  theory triangulate it. FIX before tuning W: instrument DISPATCH_GPU for the chunked
  windowed path (check backend timestamp-query pool vs the per-chunk flush pattern).
- **WIRING**: `build_f16_sg_windowed_wgsl` (sdpa.rs, shares the dense builder via a
  `windowed` bool: +3 uniform fields period/window/row_off, per-wg tile-skip to the
  window's key span, per-key out-of-window fold to -FLT_MAX). `BlockPipelines.
  sdpa_sg_win` built whenever the mixed f16 SDPA is (cheap, always present on 14B);
  `sdpa_uniform_win` (48B); `op_sdpa_f16_win` selects it when `window>0`. Threaded:
  `GenerationParams.attn_window` -> `WanDitInputs.attn_window` -> `forward_block_tiled`
  -> `self_sdpa(period,window)`. CLI `--attn-window`; e2e env THINFER_E2E_ATTN_WINDOW;
  serve `VideoSpec.attn_window` (wire camelCase `attnWindow`) -> api spec_into_request;
  web `#attn-window` input (Wan22-only reveal). Default in `default_attn_window` only.
- **SCOPE LIMITS (by design)**: (1) only the TILED long-clip path honors it (short
  clips / n_tiles==1 run full attention -- windowing is meaningless when ppf is tiny);
  (2) only the bf16 mixed f16-SDPA path = Wan2.2 14B (FastWan is F16-act, different
  branch -- unaffected); (3) AR (LongLive) uses its own windowed KV cache, untouched.
- OTHER LEVERS still open: i8-SDPA on self-attn (same ~12-TFLOPS bar, quality-risky);
  attacking the 3 outlier bf16 matmuls (the other e2e half).

**Coopmat MATMUL = LANDED + VALIDATED (proj+ffn_down on Wan22 14B).** Scope NARROW
(see memory feedback_coopmat_scope). A/B 832x480/5f/seed42 vs THINFER_NO_COOPMAT=1:
**1.31x e2e (346.1s->264.4s); ffn_down 102.9->19.7 ms/disp (5.2x), proj 37.0->6.9
(5.4x).** The 5x (not the synthetic-bench 2.3x) is because dense bf16 on the real
DiT shapes (K=14336, latency-bound) runs FAR below 3 TFLOPS. No OOM/overshoot at 5f.
Clips: scratch/wan22_coopmat.mp4 vs wan22_nocoopmat.mp4 (OPEN: user eyeball quality).
- KEY FACTS: 5070 square 16x16x16 f16->f32 config is DISCRETE-adapter only
  (HighPerformance; Arc iGPU exposes only non-square 8x8x16, unusable -- naga is
  square-only). `coopmat()` filters to square f16/f32. `enable wgpu_cooperative_
  matrix;` is native-only (web needs the future standardized subgroup-matrix
  spelling). `cooperativeMatrixRobustBufferAccess=true` -> ragged M/N need no pad.
  Throughput ceiling ~6.5-7.6 TFLOPS (naga codegen + 1-subgroup-per-wg latency;
  shared-mem staging measured SLOWER; tm/tn past 2x2 spills). Do NOT chase SPIR-V
  passthrough (breaks stays-wgpu / free-web-later).
- FILES (all landed): `thinfer-core/src/ops/matmul_coopmat.rs` (register-tiled GEMM,
  tm2_tn2 best, F32+F16 out, `b_col_major`); backend `wgpu.rs` (coopmat probe,
  Vulkan-preferred adapter, experimental feature, `WgpuConfig.disable_coopmat`);
  `workspace.rs::coopmat`; `common/block.rs` (`CoopmatStep`, `CoopmatSites`,
  `BlockPipelines.coopmat_{proj,ffn_down,qkv}`, `build_coopmat` in compile,
  `dispatch_matmul_site_coopmat`); `wan/dit_block.rs` (`lin` coopmat arg);
  `wan/pipeline.rs` (`block_cfgs` coopmat_acts proj+ffn_down). CLI `THINFER_NO_COOPMAT`.
  conformance `tests/coopmat.rs`. To add a site: f16 dequant (DequantTarget::F16,
  [N,K] nmajor, byte-compat with coopmat `array<f16>` B) + cast A bf16->f16 +
  coopmat(b_col_major,F16 out) + cast out->bf16; all in dispatch_matmul_site_coopmat.

**Serve web UI coopmat toggle = CODE COMPLETE (compiles clean), NOT YET DEPLOYED.**
- DONE (Rust): `wire.rs` `disable_coopmat: Option<bool>` on all 3 specs + `JobSpec::
  disable_coopmat()`; `api.rs` `spec_into_request` captures it (tuple widened to 4)
  + `AppState.coopmat_supported` + `GET /capabilities` ({coopmat:bool}); `job.rs`
  `JobHandle.disable_coopmat` + submit tuple; `worker.rs` layers per-job opt-out
  over `BackendConfig` at device build; `main.rs` startup probe (throwaway backend
  -> `supports_coopmat()` -> AppState). CLI image.rs/video.rs remote specs set
  `disable_coopmat: None`.
- DONE (web): `web/index.html` `#coopmat-row` checkbox (checked = on); `web/app.js`
  sends `spec.disableCoopmat=true` only when unticked + `refreshCapabilities()`
  greys/disables the box when `GET /capabilities` reports `coopmat:false`.
- REMAINING: serve REBUILD + DEPLOY (web is include_str!-baked). ASK before
  stopping the user's running server, then `cargo build -p thinfer-serve --release`
  + relaunch per the deploy note below. Verify in-browser: toggle present, greyed
  on a non-coopmat GPU; a run with it unticked logs the dense path (no coopmat).

**Wan2.2-T2V-A14B (MoE 14B + LightX2V 4-step distill, GGUF Q5_K_M) = numerics
correct + both device-loss modes FIXED; the live blocker is PERF, not crashes.**
Default 832x480 x 33f, wired CLI + serve web UI. Config + wiring all
in code (`WanDitConfig::wan22_14b`, `WanVaeConfig::wan2_1`, `Wan22DistillConfig`;
`open_wan22_source` = two folded GGUF experts; expert switch by step index: <2
high-noise else low; Wan2.1 VAE z16/4x8x8/non-residual/patch1).

- **TOP TWO NEXT: (1) EXTREME perf overhaul -- 81f@832x480 is ~4hr on this 8GB/28-SM
  card and that is the whole user-facing blocker; aim for an order-of-magnitude, not
  a marginal gain; (2) make cancel BLOCK-granular (per-step cancel is useless at
  ~1hr/step).** Both detailed below. 81f is correct but a ~4hr render; usable clips
  today = short (13-17f ~20min) or the 5B FastWan for longer. Only 13f@832x480 is
  validated to completion (rest correct by construction); >13f e2e eyeball still open.
- **FIX 1 -- TDR device-loss: one long SDPA dispatch tripped the 2s Windows GPU
  watchdog** (`TdrDelay` unset = 2s, `TdrLevel` 3 -> nvlddmkm **Event 153** engine
  reset). NOT a shader OOB (the prior note theorizing one was wrong; 4096 rows was
  just where O(rows^2) attention nears 2s). FIX: `common/block.rs::op_sdpa`
  row-chunks queries (`sdpa_chunk_rows(s_k)=10M/s_k`, BR=64 multiple) and flushes
  each chunk to its OWN submit (the watchdog is per-submit). Bit-exact, scales to
  any N. LESSON: a ~2s "device lost" + Event 153 + NO Event 4101 = the COMPUTE TDR;
  check `TdrDelay`/`TdrLevel` + per-op dispatch DURATION before theorizing an OOB;
  bound any dispatch that can exceed ~1.5s and split submits.
- **FIX 2 -- long clips crawled: the residency `transient_reserve` was
  sum-of-phases + an unbounded FFN.** Old `rows*(dff + 8*dim)` (pipeline.rs)
  reserved ~4GB at 81f -- it summed the self-attn (8*dim) and FFN (dff) phases that
  never coexist (q/k/v free before the FFN) and never bounded the rows*dff FFN
  intermediate -- starving the 14B weight cache to <1GB -> constant re-streaming =
  the `arbiter overshoot` storm. FIX: (1) row-tile the FFN (`dit_block.rs`,
  `FFN_TILE_ROWS=8192`; position-wise so BIT-EXACT, short clips = 1 chunk);
  (2) reserve = MAX-of-phase with the tiled FFN -> ~2.6GB at 81f. LESSON: the
  overshoot `bytes` is the RESERVE, not a real buffer, and `ensure_headroom` fires
  from RESIDENCY (weight admission), not just `workspace.alloc` -- a workspace-only
  alloc probe misses it.
- **CANCEL shipped but STEP-granular -> must be BLOCK-granular (open).** wan
  `Cancel` closure threaded generate -> denoise_with, polled per STEP;
  `GenerateError::Cancelled` -> worker emits `JobEvent::Cancelled`; web Cancel
  button POSTs `/jobs/{id}/cancel`. PROBLEM CONFIRMED: at 81f one step is ~1hr, so
  per-step cancel "hangs" (flag set, never polled until the step ends). NEXT: thread
  `Cancel` into `WanDit::forward` and poll at the block loop top (`dit.rs:367`) for
  ~seconds latency. AR (LongLive) cancel also not wired.
- **EXTREME PERF OVERHAUL (2 of 3 levers LANDED + committed): 14B 5f@832x480
  denoise 622s -> 331s (1.88x); 5s clip (81f) projects ~4hr -> ~40min (6.3x),
  quality-neutral (eyeballed clean).** Both fast paths were gated off Wan22 by the
  `act_dtype==F16` coupling; the fix decouples them via a `fast_sdpa` opt-in on the
  bf16 DiT block that builds the bf16<->f16 act casts, so f16-compute kernels run
  while the residual stays bf16. Mechanism (common/block.rs): `BlockWgslConfigs.
  fast_sdpa` (Wan DiT only; umt5/FastWan/others unaffected -- FastWan is F16-act so it
  already had the native path), `fast_mixed` builds `sdpa_sg` + `cast_to_f16/bf16` +
  enables `use_dp4a` for bf16; `op_sdpa_f16` (self-attn only) does the chunked f16 SDPA;
  `dispatch_matmul_site` casts the i8 path's A-side bf16->f16->i8 and its f16 out->bf16.
  `DenseActSites.qkv_self` split out from `.qkv` so self-attn qkv goes i8 while
  cross-attn stays dense.
  1. **f16-SUBGROUP SDPA -- DONE.** self-attn sdpa 204 ms/disp -> 16.3 (12.9x; the old
     bf16 SdpaF32 was the 0.22-TFLOPS spill path). Mirrors the Qwen-Image bf16-residual
     FastSdpa. Cross-attn stays bf16 (un-normed text K/V not f16-proven). cast overhead
     ~0.1%.
  2. **i8 DP4A matmul (qkv_self + ffn_up) -- DONE.** each 7x (67.7s->9.3s, 65s->9.2s
     incl dequant_i8). cast+act_quant overhead ~0.2%. proj + ffn_down + cross-qkv STAY
     bf16 (outlier A-sides: gelu product / attn-out / un-normed text -> per-32 i8 NaNs
     or crushes; the f16-clamp now prevents NaN but quality still suspect -- untested).
  3. **NOT done -- chunked-SDPA submit batching**: sdpa_chunk_rows_f16 already 2x the
     bf16 budget (20M); could batch further. Minor vs levers 1-2.
- **NEXT decision (5s still ~40min = maybe not "usable"; user pushed hard on this):**
  remaining 81f cost is co-dominant sdpa_sg (~1460s) + the 3 outlier-bound bf16 matmuls
  (ffn_down+proj+cross-qkv ~1560s). Levers: (a) **i8-SDPA** (`sdpa_i8`/`i8_sdpa` flag
  exists) on self-attn -> ~1.5-2x the sdpa term -> ~28-30min, but i8 attention =
  real quality risk, must validate. (b) The 3 bf16 matmuls resist i8 (outliers). FLOOR:
  WGSL has NO tensor cores (no wmma) -> minutes (ComfyUI/llama via CUDA) is NOT
  reachable here; ~30min is ~the 14B-5s WGSL floor on this 8GB card. For genuinely fast
  long clips: 5B FastWan (already F16 fast path) or shorter/smaller (attention O(frames^2)).
  Validate numerics each step (eyeball + short parity band) + measure `gpu_ms by pipeline`.
- **Cap** `WAN22_MAX_LATENT_CELLS=32760` = the 81f@832x480 envelope (a
  wall-time/VRAM bound, not a crash guard); `resolve()` errors explicit over-cap
  `--frames`, caps `--duration`/default DOWN with a warning.
- OPERATIONAL: bare `hf download QuantStack/Wan2.2-T2V-A14B-GGUF` pulls EVERY quant
  (150GB+) -- name the file. Wan2.1 VAE MUST be the diffusers-named one
  (`Wan-AI/Wan2.2-T2V-A14B-Diffusers` vae/, `decoder.up_blocks.*`), NOT QuantStack's
  original-named `Wan2.1_VAE.safetensors`. Full-14B pyref infeasible (fp32 ~56GB);
  validation = unit tests + 5B-parity-green components + GPU eyeball. Kill a hung
  `thinfer*.exe` (device-lost, ~12GB) before any GPU run.

**LTX-2.3 + Sulphur-2 = SHIPPED (commit a48a81b).** Done; detail in git +
`ltx-plan.md`. One carry-over to watch on the Wan fold too: the LoRA-fold host-RAM
cache (`LoraFoldSource`) is UNBOUNDED (caches every folded site as Q8 bytes); fine
on big-RAM, bound it before any STACKED fold ships.

## Lessons / dead-ends (do not retry)

- **Coopmat on ATTENTION/cross-qkv = DEAD END (2026-06-28, both reverted).** Two
  attacks, both failed:
  - *Coopmat flash-attn SDPA* (full naga-WGSL QK^T+P@V, conformance-GREEN): drop-in
    for `sdpa_sg` measured **13x SLOWER** (216 vs 16.3 ms/disp, 5f). `sdpa_sg` is
    already ~12-TFLOPS-eff (K/V reuse); naga coopmat ceiling ~7. Coopmat is a
    MATMUL-ONLY win -- never put it on already-compute-efficient ops.
  - *cross-qkv coopmat matmul*: `qkv:true` still DEVICE-LOSES at step 1 (GPU fault,
    fast) even with the `Bf16ToF16` +-65504 clamp (cast_act.rs:45). And coopmat is
    f16-INPUT-only while cross K/V come from un-normed umT5 text > 65504, so the
    input clamp is an inherent quality loss regardless. qkv stays DENSE; proj +
    ffn_down coopmat (normed A-sides) are fine and stay.
  - naga gotcha learned: `var c: coop;` inside a loop is null-inited ONCE at fn
    entry, not per-iteration -- re-zero by copying a fn-scope zero each pass.
- **DiT denoise is at the WGSL matmul ceiling.** ~100% matmul+sdpa GPU time,
  latency/occupancy-bound (NOT bandwidth) on the 28-SM mobile 5070; bigger tiles
  BACKFIRE. Weight-only quant does NOT help perf (dequant->bf16, same FLOPs). Only
  real levers: i8 DP4A (done) + mixed-precision f16 SDPA (done). Measure via e2e
  `gpu_ms by pipeline`, NOT microbench.
- **GGUF quant choice (Qwen lesson, applies to LTX).** A per-request whole-DiT Q4_K
  fold re-quantizes every request -> ~2x SLOWER than Q8_0 AND quality broke. **Q8_0
  is the quality+perf baseline for big DiTs**, not Q4_K. Q4_K_M = compare/footprint
  only, GPU-unvalidated, ZERO speed win (dequants to the same bf16 + same matmul).
- **i8 DP4A**: transcodes ffn_up + self-attn-qkv bf16->Q8_0 at load, `dot4I8Packed`
  path; ~5-6x those ops, quality-neutral (A-sides are normed/modulated, no
  outliers). NOT i8'd: proj + ffn_down (attention-out/gelu has ~16k outliers) +
  cross-attn-qkv from UN-NORMED text (i8 acts overflow f16).
- **Mixed-precision f16 SDPA** (~2x/block, no quality loss): residual + matmuls stay
  bf16/Q8_0; ONLY Q/K/V cast bf16->f16 (post-rmsnorm/rope = f16-safe), f16 subgroup
  SDPA, cast back. `cast_act::{Bf16ToF16,F16ToBf16}` packed-word casts.
- **BLOCK-WIDE f16 is a DEAD END** for these big DiTs: the residual stream has
  large-outlier channels > f16's 65504; full-f16 diverges badly even with clip.
  bf16 residual is load-bearing. (Whole i8-DP4A block path also needs f16 acts ->
  same wall.) Keep residual bf16. NB this is the DiT RESIDUAL only -- f16 acts in
  the VAE decode + the LTX-distilled DiT acts are fine (validated).
- **VAE decode is conv-GPU bound** (~95% conv time; metric = `gpu_disp_ms` for
  `vae_decode`, NOT nvidia-smi). DO NOT retry the conv3d im2col loop-invariant-div
  hoist (REVERTED, slower).
- **Tiny VAE (LightTAE) is the `--vae` default** for Wan (`--vae full` = parity).
  LTX has NO tiny-VAE path (its Quality dropdown is Wan-only).
- **Never run fp32 CPU pyref above tiny dims** (40GB host at 256x256). pyref is
  component-wise, same weights, tiny dims, bf16; never full-pipeline/full-DiT (OOM).
- **LTX-2.3 (do not retry):**
  - Off-subject / garbage LTX output is RESOLUTION + the distilled few-step
    ADHERENCE ceiling, NOT text encoding. 256x256 -> stage1 128x128 is deeply OOD
    (mush); 512+ is sharp/coherent. Distilled uses `SimpleDenoiser` = genuinely
    CFG-free (no neg-prompt/guidance to "fix"). Don't chase the tokenizer/encoder.
  - LTX VAE OOM: TILE (spatial + temporal overlap-blend), don't shrink default
    dims. The decode peak is ACTIVATION-bound, so a strict WEIGHT budget can't cap
    it. f16 acts are a safe default (vae_parity f16 band: slope 0.9998, rel 0.096%);
    bit-exact only at f32 (the parity reference path). Size the first tile BELOW the
    budget (SEED_SAFETY) and re-seed balanced on OOM -- a per-axis halve explodes
    the tile count.
  - Gemma-3 encoder MUST run F32 acts: head_dim 256 needs `large_d_sdpa` (no bf16
    variant) AND the residual overflows f16. All sites dense (no i8).
  - The gemma `(1+w)` norm bake is REAL and STAYS (llama.cpp pre-bakes +1; both
    engine UnitOffset and pyref HF `1+w` add +1 -> match). Don't "fix" it.
  - DON'T flip the DiT to strict budget: it relies on overshooting into device
    slack; strict would reject configs that work today (no adaptive smaller-retry).

## Carry-forward gotchas (engine-general)

- **Ops reading aux params as f32 must dequant bf16 weights first.** An op binding
  declared `array<f32>` (e.g. SnakeBeta alpha/beta, or any per-channel param) fed a
  raw bf16 weight buffer reads garbage -> silently wrong. The conformance harness
  has NO (act-F32, weight-Bf16) dtype, so f32-binding-fed-bf16 is untested -> passes
  conformance, fails in the model. Dequant bf16->f32 on host + upload f32 (or add a
  bf16 binding path). Bit it on the LTX vocoder; bisect model bugs op-by-op when
  conformance is green but the assembled model is decorrelated.
- **Q8_0 subnormal f16 scale (quant.rs `f16_bits_to_f32`)**: a subnormal-branch
  exponent off-by-one HALVED any block whose f16 scale is < 2^-14 (tiny weights,
  e.g. gemma ffn_down). Fixed; regression `q8_0_dequant_once_subnormal_f16_scales`.
  Affects ANY tiny-weight quant tensor on the bf16-dequant path, all models.
- **umT5 / large-residual encoders MUST run bf16 acts** -- residual grows past f16's
  65504 -> inf -> NaN (token-uniform "washed blob", prompt-dependent). Check
  non-finite NOT just NaN (`inf.is_nan()` is false).
- **Module-level matmuls bypass `dispatch_matmul_site`.** Sites like Wan's patch
  embed / condition embedder / proj_out call `scope.matmul` DIRECTLY with a raw
  weight + no dequant pre-pass. They reuse a block pipeline (`matmul_qkv`), which on
  a quant DiT is rebuilt for the dequant WORKSPACE (b_nmajor + workspace dtype) -> a
  raw module weight there reads as garbage (inf/NaN), AND a bf16 weight misreads
  under F16 acts. Rule: module-level dense linears get their OWN bf16 `matmul_module`
  pipeline (block.rs), weights kept bf16. Don't quantize a weight whose matmul site
  has no dequant step. (Wan2.2-14B garbage-output bug, 2026-06-27.)
- **RoPE dtype**: freqs MUST pack to the act dtype (`freqs_upload_bytes`); f32 freqs
  into an f16 kernel -> inf -> NaN. Wan RoPE3D = interleaved-pair; Qwen3 + LTX-2.3 =
  half-rot ("split"). Pick the right one per model.
- **bf16->f16 reinterpret class**: broadcast vectors that are STORED WEIGHTS
  (scale_shift_table, norm affine) use a `weight_dtype`-keyed op, not an act-scale
  op. New broadcast site: check weight vs act.
- **GGUF tensor padding**: F32/F16 narrow arms must slice `elements*size`, not
  `on_disk_bytes` (GGUF pads each tensor to 32B). F16 weight support is in
  `weight::Decoder` + `residency::gpu_encoding` + `register_linear_transcode`.
- **Third-party clones**: `rm -rf <clone>/.claude` right after cloning (they inject
  skills/agents). See memory.
- Video staging: per-frame PNG seq / contact sheet; MP4/WebM in CLI only (openh264).

## Status (shipped -- DO NOT DISTURB)

- **LTX-2.3 distilled + Sulphur-2** (22B joint audio-video) -- shipped, see
  `ltx-plan.md`. Q8 default + Q4_K_M; f16dp4a DiT + f16 VAE; two-stage widescreen
  default; Sulphur distill-LoRA fold.
- **FastWan2.2-TI2V-5B-FullAttn** -- parity GREEN, bf16 acts. UniPC is the default
  sampler (CLI+serve+web); DMD is the parity path. PENDING: user eyeballs a UniPC
  clip vs the KingNish Space; else fall back to GPU upstream-pyref compare.
- **LongLive-2.0-5B** (AR causal long + multi-shot, same Wan base) -- shipped,
  health+parity GREEN (two-tier bands). OPEN: AR perf wins (upload window prefix
  once/chunk; activation-tile AR self-attn; cache cross-attn text K/V; skip head on
  clean recache) -- length/VRAM-bound, not wall-bound. Re-measure WARM at 576 first.
  Multi-shot is only health-tested (a multi-prompt pyref is the durable proof).
- **Face-swap** (`thinfer generate face-swap`) -- shipped. NEXT = quality (XSeg
  occlusion + GFPGAN + HyperSwap mask); see `faceswap-plan.md`.
- **Ideogram-4** (`ideogram4-q8`) -- shipped. Q8_0 enc+DiT, turbotime LoRA folded,
  FLUX.2 KL VAE, i8 DP4A. 512x512/4-step ~79s.
- **Z-Image** -- shipped; `zimage-plan.md` (read only if touching it).
- **Qwen-Image-Edit-Rapid + Qwen-Image-Rapid** (20B MMDiT, 4-step CFG-free; t2i +
  image-edit) -- shipped, deployed to serve. `qwen-image-plan.md`.
- **i8 DP4A matmul ON by default** (`--no-i8-matmul` = bf16 reference path).

## Crate layout + serve (shipped reference)

- Layering: core < models (dep-clean) < native < app < {cli, serve}. `thinfer-web`
  (wasm) is the exception (own browser substrate, calls `thinfer-models` directly;
  cross-language sharing via generated OpenAPI TS, not a shared Rust types crate).
  serve is a SEPARATE binary (not a `thinfer` subcommand) BY DESIGN: it pulls a
  heavy axum/rustls/utoipa/crypto stack the lean clap CLI shouldn't carry; both are
  thin siblings over shared `thinfer-app` (cli vs serve feature).
- `thinfer-app`: `model` (id enums + defaults + manifest), `request`, `wire` DTOs
  (serde + utoipa `ToSchema` under `serve`), `progress` (`Stage`/`ProgressSink`/
  `ProgressEvent`/`ProgressFn`), `download`, `codec` (mp4 faststart/png-frames),
  `executor::{Local,Remote}Executor`, `config`. One `JobExecutor` trait everywhere.
- `thinfer-serve`: axum + utoipa. `POST /jobs`, `GET /jobs/{id}{,/events(SSE),/
  result}`, `POST /jobs/{id}/cancel`, `/openapi.json`. Worker pool (OS threads,
  current-thread runtime; default 1), in-mem metadata, disk artifacts (TTL),
  delete-on-fetch result, opt-in TLS (`tls_self_signed`), RSA-OAEP+AES result
  encryption (needs secure context). Embedded web UI (vanilla JS, fetch-SSE,
  `include_str!`-baked -> web edits need a serve rebuild). **One fresh wgpu device
  per JOB** (worker.rs): nothing resident across jobs, so a bad gen can't leak.
- **Deploy = I do it myself** (stop running server -> `cargo build -p thinfer-serve
  --release` -> restart detached); do NOT hand the user manual steps, but ASK before
  stopping the server he toys with. rustls is TLS-1.3-only -> local PS/.NET/curl can't
  probe; the browser is the check.
  - **EXACT relaunch (config is a FLAG, not positional!):** `Start-Process` the exe
    with `-ArgumentList '--config','C:\work\personal\thinfer\scratch\serve.toml'`,
    `-WorkingDirectory <projects>` (so `artifact_dir=thinfer-artifacts` resolves to
    `projects/thinfer-artifacts` as before), stderr -> `scratch/serve.log` (tracing),
    stdout -> `scratch/serve.stdout.log`, `-WindowStyle Hidden`. Confirm via the log:
    must show `serving HTTPS with a self-signed cert` + `listening (https)`. NEVER run
    the exe in a foreground Bash call (it blocks = "hang").
- DEFERRED: mid-generate cancel (touches shipped generate sigs); serve==CLI
  byte-parity test (weights+GPU gated); disk-backed SSE ring buffer; wasm<->http web
  toggle (server-only by decision).

## Running the e2e / measuring

Card = RTX 5070 Laptop (8GB); keep budgets <8GB (8GB OOMs the device). All serial
(`--test-threads=1`): multiple WgpuBackend in one binary contend on VRAM.
- LTX e2e: `THINFER_POWER_PREF=high THINFER_TRACE=1 THINFER_E2E_FRAMES=121
  THINFER_E2E_HEIGHT=320 THINFER_E2E_WIDTH=512 THINFER_E2E_VRAM_GB=6
  THINFER_E2E_PNG_DIR=<dir> cargo test -p thinfer-conformance --features ltx-e2e
  --release t2v_e2e_health -- --nocapture --test-threads=1`. Component parity gates:
  `vae_parity`, `audio_vae_parity`, `vocoder_parity`, `dit_parity`/`dit_full_parity`/
  `modulation_parity`, `connector_parity`, `encoder_parity` (run ONE AT A TIME -- the
  12.5GB encoder OOMs a multi-test binary). `dit_perf` = engine-only perf bench.
- FastWan parity gate: `THINFER_TRACE=1 THINFER_POWER_PREF=high
  THINFER_E2E_BUDGET_GB=6 THINFER_E2E_WIDTH=256 THINFER_E2E_HEIGHT=256
  THINFER_E2E_PNG_DIR=<dir> cargo test -p thinfer-conformance --features wan-e2e
  --release video_e2e -- --nocapture --test-threads=1`. Perf-only: add
  `THINFER_E2E_SKIP_PYREF=1`.
- LongLive: `longlive_parity` / `longlive_e2e` (use 256x256), `--features wan-e2e`.
- Qwen: `dit_perf` = perf turnaround bench (engine-only); `dit_parity` = cheap
  single-block cached pyref.
- GGUF inspect: `uv run --with gguf python` + `gguf.GGUFReader`; ALL hyperparams come
  from the `config` KV (LTX) or tensor names+shapes. Re-create a dump script as
  needed (scratch is ephemeral).
- CLI run: `THINFER_TRACE=1 THINFER_POWER_PREF=high thinfer generate ...
  --vram-budget 5G --ram-budget 5G`. Read per-op `gpu_ms by pipeline` to localize.
