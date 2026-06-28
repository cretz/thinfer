# Worklog

Forward-looking only. Git history is the changelog, the code is the record. Past
work appears here ONLY as a one-line lesson or a do-not-retry. Engine-wide design
+ kernel/runtime state: `plan-details.md`. Per-model porting plans are separate
files (see Status). Scratch is ephemeral; nothing here depends on a scratch file.

## NOW / NEXT

**ACTIVE: Wan 2.2 14B A14B (MoE) + LightX2V distill, GGUF Q5_K_M.** New target on
the shipped Wan backbone. Numerics CORRECT (coherent, on-prompt). Frame-envelope
cap + per-model `max_frames` + form fixes done; default validated at
512x288x25f. NOT committed (user browser-verifies first).

**ROOT CAUSE CORRECTED: the 832x480x81f default device-loss was NOT a VRAM-budget
overflow -- it is a GPU FAULT in the 14B DiT above a SEQUENCE-LENGTH threshold.**
The earlier "activation envelope overshoots 8GB" framing was wrong. Telemetry
(`nvidia-smi` + Windows System log) is decisive:
- nvlddmkm **Event 153 (GPU engine reset)** at each crash, with NO Event 4101/13
  (TDR "display driver stopped responding") and NO VRAM pinning at 8151MiB. That
  signature = a GPU OOB shader fault (MMU fault -> engine reset -> "device lost" on
  the wgpu poll thread), not a timeout and not a clean alloc-OOM. wgpu's
  uncaptured-error handler never fires (it is a hardware fault, not a CPU-side
  validation error), so the root error stays masked behind "Parent device is lost".
- It is **rows-bound, independent of weight budget and f_lat**: 2G and 5G both
  fault at 6240 rows; 256x256x61f (4096 rows, f_lat 16) COMPLETES. Empirical DiT
  rows (= f_lat*(h/16)*(w/16)): 2048 / 3120 / 4096 complete; 6240 faults. So the
  machine tolerates long GPU work (5f step = 184s, no TDR) -- it is a fault, not a
  budget or a timeout.
- **SHIPPED GUARD (not the real fix): a frame-envelope cap.** `WAN22_MAX_LATENT_CELLS
  = 4096` (confirmed-safe ceiling, ~1.5x under the 6240 fault). Generalized
  `ltx_max_frames` -> per-model `VideoModelId::max_frames(w,h)` (LTX /32 grid +
  6300 cells; Wan /16 grid + 4096). `resolve()` now caps ALL video models: explicit
  over-cap `--frames` = hard error, `--duration`/default cap DOWN with a warning.
  Default res LOWERED to 512x288 (clip length scales as cells/(h/16*w/16), so
  lower res = more frames): default 512x288x25f = 4032 cells ~1.6s. 832x480 stays a
  form option but caps to 5f (~0.3s).
- **REAL FOLLOW-UP (lifts the cap): root-cause the DiT shader OOB at >~4096-6240
  rows.** Needs GPU-assisted validation (DEBUG_PRINTF / robustness) or op-bisection
  of the block forward -- too slow to brute-force (184s/step). Suspects ruled OUT:
  storage-buffer binding (NVIDIA maxStorageBufferRange = 4GB), the bcast/elementwise
  ops (they spill the 1D grid past 65535 via `linear_workgroups`), matmul/SDPA grid
  dims (all < 65535 at these rows), ROPE_MAX_SEQ_LEN 1024 (per-axis, never hit).
  NOT yet checked op-by-op for a hardcoded seq buffer or a Y/Z dispatch dim. Fixing
  it (or row-tiling the DiT forward) would lift the cap and unlock 832x480 at real
  lengths.
- LESSON: pull DRIVER telemetry (Event log Id 153 vs 4101, nvidia-smi peak) BEFORE
  theorizing OOM-vs-TDR-vs-fault; do not bracket by guess-and-check. And never
  redeploy a video model at an unvalidated default res x frames. (Wan2.2-14B,
  2026-06-27.)
- A hung `thinfer.exe`/`thinfer-serve.exe` (device-lost, ~12GB) must be KILLED
  before any GPU run.

**BUG FIXED (the flat/NaN garbage): the module-level matmul sites bypassed the
dequant pre-pass.** The patch embed (`dit.rs:linear_bias`), condition embedder
(`condition_embedder.rs:linear_bias_into`), and proj_out all called `scope.matmul`
on the block's shared `matmul_qkv` pipeline with the RAW weight buffer + no dequant.
On the Q8 GGUF path `matmul_qkv` is rebuilt for the dequant WORKSPACE (b_nmajor=true,
workspace dtype), so feeding it a raw module weight -> inf/NaN. (The earlier bisect
mis-located this inside `dispatch_matmul_site`'s dense-dequant branch; those module
sites never reach that branch.) Fix:
- Module weights now stay DENSE BF16 (`module_transcode` removed; F16/F32 narrow to
  bf16 at upload via residency gpu_encoding). Tiny matmuls, run once/forward -> zero
  perf cost, slightly better quality than Q8.
- New dedicated `matmul_module` pipeline in `common/block.rs` (always bf16 weight,
  `square` 64x64 tiling), wired into the 3 module sites. Decoupled from the block's
  per-site quant pipeline for ALL Wan models (no-op for the bf16 5B path).
- `WanVariant::wan22.act_pref` F32 -> Bf16 (F32 was a diagnostic; bf16 holds the
  14B residual range, ~2x faster, matches pyref dtype).
- All WIP diagnostics removed (MoE step-0 taps, pre-VAE latent stats, residency q8
  transcode-roundtrip log). fmt+clippy clean.
- Perf profile unchanged from the accepted ceiling: bf16 acts, Q8 block weights via
  dequant-once, no DP4A (the 14B residual overflows f16 so the F16-gated i8/sg-SDPA
  paths stay off). `matmul_module` adds ~0.28s total (negligible vs ~80s of block
  matmuls/4 steps).
- Everything else WORKS: VAE name fix (use Wan-AI/Wan2.2-T2V-A14B-Diffusers
  vae/, NOT QuantStack's original-named one), the 2-expert source/load/denoise,
  the step-distill scheduler, wiring. (Default frame cap = the BLOCKED item above.)
- Engine done: `WanDitConfig::wan22_14b`, `WanVaeConfig::wan2_1`, generalized LoRA
  fold (`ltx::lora` discover accepts `lora_down`/`lora_up`), `open_wan22_source`
  (two folded GGUF experts, prefixed `high.`/`low.`, unioned with reused umT5 +
  diffusers Wan2.1 VAE), `WanModel::load_variant` + `WanVariant`, MoE step-distill
  denoise (`Wan22DistillSampler`, expert switch + boundary evict), loader prefix,
  per-config `vae_scale`. Wired: `VideoModelId::Wan22T2vA14b`,
  manifest two-expert `VariantFiles`, executor arm, web dropdown.
- KEY DECISIONS that worked: LoRA covers EVERY block matmul -> fold re-encodes all
  to Q8_0 (uniform). Module-level weights (patch, condition embedder, proj_out) stay
  DENSE BF16 on a dedicated `matmul_module` pipeline -- they are NOT folded by the
  LoRA so quantizing them bought nothing, and they bypass the per-site quant block
  pipeline cleanly. F16/F32 module weights + norms/biases/scale_shift_table
  upload-narrow to bf16 automatically (residency gpu_encoding), no extra work.
- GOTCHA fixed: QuantStack `Wan2.1_VAE.safetensors` is ORIGINAL-Wan naming
  (`decoder.middle.*`, `decoder.conv1`) which the diffusers-named VAE loader can't
  read. Use the diffusers VAE from `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (vae/) ->
  `decoder.up_blocks.*`. VAE_WAN21 points there.
- DOWNLOAD GOTCHA: bare `hf download QuantStack/Wan2.2-T2V-A14B-GGUF` pulls EVERY
  quant (150GB+); always name the specific file.
- PARITY: full-14B pyref INFEASIBLE on this box (fp32 ~56GB, same as the abandoned
  LTX-22B pyref). Validation = unit tests (schedule, fold) + reused-validated
  components (umT5, Wan DiT arch both 5B-parity-green) + the new Wan2.1 VAE
  exercised in the real decode + GPU eyeball. A component Wan2.1-VAE pyref gate is
  a possible follow-up (ask first; GPU).
- Original verified facts below.
- **Two 14B experts (MoE), identical config**: dim 5120 (40 heads x 128), 40
  layers, ffn 13824, in/out 16, freq 256, eps 1e-6, text_len 512, patch (1,2,2),
  qk rms_norm_across_heads, cross_attn_norm, RoPE theta 10000. CONTRAST with our
  5B TI2V (24 heads/30 layers/14336 ffn/48 ch) -- genuinely different, NOT a
  layer-count tweak.
- **Expert switch by noise level**: high-noise expert (index 0) when
  `t >= boundary*1000`, low-noise (index 1) below. T2V boundary 0.875 (I2V 0.900).
  DISTILLED switches by STEP INDEX: `step_index < boundary_step_index(=2)` -> high,
  else low (4 steps = 2 high / 2 low).
- **VAE = Wan2.1** (z_dim 16, 4x8x8, non-residual, patch_size 1), NOT the TI2V-5B
  VAE (z48, 4x16x16, residual). latents_mean/std are 16-vecs (from VAE config).
  Engine `WanVaeConfig` is already parameterized; needs a `wan2_1()` ctor + the
  non-residual/patch1 decode path EXERCISED (5B e2e never hit it).
- **LightX2V distill**: 4-step CFG-free. denoising_step_list [1000,750,500,250],
  sample_shift 5.0, CFG off. TWO separate rank-64 LoRAs (one per expert,
  strength 1.0), name-matched to expert at load. Fold each into its expert GGUF via
  the LoRA-fold machinery (`ltx::lora`, generalize the key map; ideally lift to a
  shared module).
- **Weights (HF cache, downloading)**: QuantStack/Wan2.2-T2V-A14B-GGUF
  (HighNoise/LowNoise Q5_K_M + VAE/Wan2.1_VAE.safetensors); lightx2v/
  Wan2.2-Distill-Loras (t2v high+low rank64 1217). umT5-XXL REUSED from the
  FastWan diffusers bundle (already cached); tokenizer reused.
- **Engine touch-points** (from code map): `WanDitConfig::wan22_14b()` +
  `WanVaeConfig::wan2_1()`; de-hardcode `WanModel::load` 5B config (pipeline.rs:338,
  343) and module consts `VAE_SCALE=16`/`TEMPORAL_SCALE=4` (pipeline.rs:59-60) ->
  per-config (Wan2.1 VAE is 8x spatial). Two-expert dispatch at the DiT forward
  sites (pipeline.rs:1114/1193) keyed on the in-scope `timestep`/step index.
  `VariantFiles` must carry TWO DiT roles (currently single leading DiT). New
  `ModelManifest` + `manifest()` arm; `VideoModelId` variant + web dropdown string.
- **Defaults (upstream)**: 1280x720 (832x480 for the 480p/distill config), 81
  frames (4n+1, the Wan rule), fps 16, flow shift 5.0 (distill). Respect 8GB:
  DiT streams per-block so VRAM is bounded by activations, not weights; size the
  default to the frame envelope like LTX.
- **Plan: don't commit; user verifies in browser.**

**LTX-2.3 + Sulphur-2 family = SHIPPED (commit a48a81b, 2026-06-27).** 22B joint
audio-video. Two-stage widescreen default + per-res frame/duration caps, Q4 Gemma
encoder option, Sulphur distill-LoRA fold. Detail in git + `ltx-plan.md`. Forward
notes retained below; the rest is git history.

## LTX (shipped) -- forward notes only

- Adherence was REGIME, not a bug: low-res single-stage is OOD. Default = in-distrib
  WIDESCREEN TWO-STAGE, CFG=1 (CFG is vetoed: 2x cost). Distill-LoRA choice does NOT
  move photorealism (the illustrative/off-subject skew is the CFG-free regime
  ceiling). condsafe@1.0 is the correct distill artifact; the standalone rank768 is a
  CONTENT lora, not step-distill (mush if folded into the 8-step path).
- **8GB frame envelope = frames dominate VRAM** (stage-2 runs the DiT at full res).
  `LTX_MAX_LATENT_CELLS=6300` + `ltx_max_frames`: 1280x704->49f(~2s),
  1024x576->73f(~3s). Over-budget explicit `--frames` rejected at submit.
- **Stage-2 OOM at high (res x frames) LOSES THE DEVICE** (hard wgpu poll-thread
  panic), not a catchable alloc error -> hung process holding VRAM. No graceful-OOM
  retry yet (follow-up). Clean up hung thinfer.exe before the next run.
- `LTX_VRAM_BUDGET_CAP=2G`: the 22.8GB DiT always streams per-block, so a high weight
  budget only steals device slack stage-2 needs.
- **Sulphur fold host-RAM cache is UNBOUNDED** (`LoraFoldSource` caches every folded
  site as Q8 bytes, ~12-18GB for 22B, not governed by ram_budget). Fine on big-RAM;
  bound it before any stacked fold ships. (Applies to the Wan fold too -- watch it.)

## Lessons / dead-ends (do not retry)

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
