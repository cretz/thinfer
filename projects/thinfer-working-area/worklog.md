# Worklog

Start here. Forward-looking only (git history is the changelog). Engine-wide
design + kernel/runtime state: `plan-details.md`. Active model port:
`wan-plan.md`. Z-Image (shipped, archived): `zimage-plan.md`.

## State

- Z-Image-Turbo t2i complete, native + web, q4 default, parity green, perf mined
  out. Details: `zimage-plan.md`.
- Engine (residency/arbiter, matmul + sdpa kernels, GGUF quant, wgpu native +
  wasm, OPFS) is the reusable substrate for video. Details: `plan-details.md`.
- Wan: SkyReels-V2-DF-1.3B port on branch `video` off `main`. Full pipeline runs
  e2e (umT5 -> DiT -> synchronous-DF UniPC -> 3D causal VAE) within the 2GiB
  budget and is numerically healthy: the **8-step parity gate is green** end to
  end (all `step_post` slopes 0.99-1.00, `vae_rgb` slope 0.994). umT5/text and
  ALL scheduler paths (order-1, order-2 predictor AND corrector) are parity-clean.
- VAE decode at native res (960x544) now fits the budget: watchdog-safe submits
  + **budget-derived spatial tiling** (`wan/vae.rs`, see gotchas). peak_live 5312
  -> 2047 MiB, no TDR, no OOM, no seams. The native peak is now the **DiT
  DENOISE** (~2.0 GiB), not the VAE.
- `thinfer generate video` CLI (t2v) shipped: `thinfer-cli/src/cmd/generate/`
  (`mod` + `image` + `video`). MP4 via openh264 + `mp4`; `--output-format
  png-frames` (--output = dir) for codec-free frame inspection.
- **FIRST REAL FRAME PROVEN** (2026-06-18): CLI t2v with CFG produces a sharp,
  coherent red balloon over a green field (960x544, 5 frames, 25 steps, seed
  random, guidance 6.0). CFG was the missing adherence piece. See CFG below.

## Project: video generation (Wan family)

Port the Wan backbone once, unlock a family. First target SkyReels-V2-DF-1.3B-
540P (low resource + infinite length), then 5B (LongLive-2.0 / Wan2.2-TI2V-5B)
and audio (NAVA). Full plan + sources + rationale in `wan-plan.md`.

## DONE: CFG wired, first real frame proven

SkyReels-V2-DF is NOT guidance-distilled. Confirmed from diffusers source
(`pipeline_skyreels_v2_diffusion_forcing.py`): `guidance_scale` defaults to 6.0
(T2V; 5.0 for I2V), `do_classifier_free_guidance = guidance_scale > 1.0`, default
negative prompt `""`, combine `uncond + gs*(cond - uncond)` (line 909). The pyref
pins `guidance_scale=1.0` purely for parity convenience (one forward = clean
byte-compare), which is why every prior tap was washed out + balloon-less.

Wired (2026-06-18): `GenerationParams.{negative_prompt, guidance_scale}`; default
guidance lives on the model (`manifest::RECIPE.default_guidance_scale = 6.0`), CLI
`--guidance-scale`/`--negative-prompt` override. `denoise_with` encodes both
prompts while umT5 is resident, then runs a second (uncond) DiT forward per step
when `guidance_scale > 1.0`. The e2e pins `guidance_scale = 1.0` at both
`GenerationParams` sites, so `do_cfg=false`, single forward, parity stays
bit-clean.

Result by eye (scratch run, 5 frames / 25 steps / 540P): frames 2-4 are a sharp,
coherent red balloon. Frame 0 is washed-out -- that is latent-frame-0, the causal
VAE anchor (f_lat=2 -> only 2 latent frames; output0 = latent0, outputs1-4 =
latent1 upsampled). Weak-first-frame is a known Wan causal-DF t2v trait, not a
pipeline bug; it dilutes to 1-of-97 at real length. Minor follow-up, not a gate.

## NEXT (lead): DiT denoise perf -- the real frame costs too much

The frame is proven but SLOW: 457s for 5 frames / 25 steps at 540P (~16.7s/step;
CFG doubled it to TWO DiT forwards/step). The 97-frame default (f_lat=25, ~12.5x
tokens, quadratic self-attn) is impractical on this path. This is now the wall
between "proven" and "usable".

NON-NEGOTIABLE: quality stays at full normal-run settings. Do NOT "fix" perf by
cutting steps/frames/guidance -- that degrades the output, it does not optimize
the engine. The fix is making the SAME high-quality run fast.

METHOD (how to attack it):
1. Drive from the e2e (`video_e2e_parity`, SKIP_PYREF + native dims) + the
   `THINFER_TRACE` rollup. Read the per-scope gpu_ms / submit_ms / n_alloc /
   ws_alloc table FIRST; let it point at the hot scope before touching code.
2. If the trace can't localize a cost (e.g. weight-feed vs attention vs
   modulation-broadcast inside a block), ADD telemetry -- but only under the
   trace/diag gate so prod + `generate` pay zero (same discipline as the
   per-step readback sink). Then re-measure.
3. Fix the structural cause, re-run the e2e, confirm the trace moved AND parity
   stayed green. Repeat.

The big lift is wanted and approved: do it right, no shortcuts. Levers below; the
trace hot op at f_lat=2 is `narrow_transpose_f32` (weight-feed), so start by
proving weight-feed-bound vs compute-bound with telemetry, not by guessing.

## DiT denoise is the perf + memory wall (gates real-length + thin HW)

Now that the VAE is tiled, the DENOISE is the single ~2.0 GiB peak AND the time
sink. Both block real runs and thin (<2GiB) hardware.

- Perf: synchronous DF runs FULL O(T^2) self-attention over ALL tokens
  (f_lat*h_lat*w_lat). Measured (960x544, f_lat=2, 25 steps, CFG 6.0, RTX 5070
  laptop): umT5 ~16s, denoise ~16.7s/step, VAE ~22s, total 457s. At the 97-frame
  default (f_lat=25, ~12.5x tokens, quadratic) per-step explodes -> impractical.
  LEVER: async/causal-block DF -> O(T*block).
- CFG doubles per-step cost (cond + uncond = TWO DiT forwards). LEVER: batch the
  two forwards into one `[2*ppf]` rows pass (shared weights, one residency feed,
  better GPU occupancy) instead of two sequential forwards. Or, since CFG is
  expensive, expose a distilled/low-CFG fast mode. Biggest single win on this
  path right now.
- Memory: DF modulation materializes 6+1 per-token `[n_tok, inner]` broadcast
  buffers (~1.9 GB at 540P). FIX: a row-broadcast op so blocks read the compact
  `[f, inner]` form. This is the main thing standing between native decode and a
  <2GiB budget (VAE already tiles to fit).
- Secondary: umT5's flat ~13.5s serial tax (24 layers @ ~0.55s); probe whether
  the q4 GGUF DiT path is weight-feed-bound (f32 safetensors path was; fixed via
  GPU-side `WeightPrep::NarrowTransposeF32`).

## Characterized (not blocking): bf16 DiT-velocity floor

The only numerical residual. fp32-pyref probe verdict: it is a PRECISION FLOOR,
not a block bug -- velocity slope ~1.0 (no systematic under-build) but noisy
(rmse ~0.2 at 2-step), and it diverges similarly vs bf16 AND fp32 references.
Worst at aggressive 2-step (`step1_post` slope 0.927); benign by 8-step (0.997,
the green gate). Scheduler/`conv` is bit-exact (rmse 0.0). No action needed
unless a real frame looks wrong in a way that traces here.

## Smaller follow-ups

- Bump the committed canary default from 2-step to >=4 (`THINFER_E2E_STEPS`
  default) so order-2 predictor AND corrector stay gated and the default sits
  above the 2-step floor. Confirm 4-step green before committing the change.
- Tighten the broken-vs-noisy caps (`CAP_STEP`/`PRE_VAE`/`VAE_RGB` in
  `video_e2e_parity.rs`) to real numbers now that 8-step parity is clean.
- Large-dim pyref + e2e (parity has only run at 64px; coherence lives at native).
  CPU pyref is slow -- may need a GPU/torch-cuda path or a mid-size compromise.
- VAE-ENCODE stage tap (t2v only decodes; encoder unvalidated). `WanModel`
  exposes decode but not encode-stage-diag -- add a thin accessor. Encoder
  `mid_attention` asserts B==T==1; may need general b*t batching if a tap drives
  T>1.

## Running the e2e

`tests/wan/video_e2e_parity.rs` (`video_e2e_parity_safetensors`, feature
`wan-e2e`). Byte-loads pinned `[16,2,8,8]` noise both sides, drives the pyref via
`uv run`, per-stage compares, asserts dims + 2GiB VRAM/RAM, stages PNGs.

`THINFER_TRACE=1 THINFER_POWER_PREF=high THINFER_E2E_PNG_DIR=<dir> cargo test -p
thinfer-conformance --features wan-e2e --release video_e2e_parity -- --nocapture
--test-threads=1`

- Env knobs: `THINFER_E2E_{STEPS,WIDTH,HEIGHT,FRAMES}` (committed default
  2-step/64px; deep accuracy = more steps, perf = larger dims +
  `THINFER_E2E_SKIP_PYREF=1` since the CPU pyref can't scale).
  `THINFER_E2E_PYREF_DTYPE=fp32` for the bf16-floor probe. `SKIP_PYREF=1` keeps
  budget asserts + PNG staging, skips the reference + checks. Authoritative gate
  is STEPS=8; native memory check is SKIP_PYREF=1 WIDTH=960 HEIGHT=544 FRAMES=5.
- Bisection telemetry (gated behind the diag sink; prod/`generate` pass None ->
  zero readbacks): per-step `WanStepDiag` from `denoise_with`; pyref dumps
  `py_dit_out_step{i}` + `py_block{b}_out_step{s}`; test prints velocity linfit +
  scheduler-isolation + per-block slope tables.
- Step-0-only deep harness: `WanModel::diag_step0` + `THINFER_WAN_DIAG=1`. Bringup.

## Module status

Module-complete + fmt/clippy/check clean, exercised via the e2e gate:
- VAE (`wan/vae.rs`): decoder (incl. native-res tiling) + encoder drivers +
  per-stage taps. Pyref `autoencoder_kl_wan.py` (is_residual=False). Encoder
  unvalidated (follow-up above).
- Scheduler (`wan/scheduler.rs`): pinned flow config, synchronous DF only, orders
  1-2 (family pins 2). last_sample = post-corrector sample; all order paths
  proven correct + 8-step e2e green.
- `source.rs`/`manifest.rs`/`loader.rs`/`pipeline.rs`: WanSource (Plain
  safetensors parity path | Quantized umT5-GGUF+DiT-GGUF+VAE-safetensors), GGUF
  rename maps verified from real Q4_K_M dumps. Open for the GGUF path: confirm
  Q4_K_M quantizes patch/proj_out/embedders like the blocks (norms/biases stay
  F32), and `bf16_quant_writes` correctness.

## Carry-forward gotchas

- VAE tiling calibration: if a device still OOMs/overshoots the VAE decode, the
  knobs are `BYTES_PER_LAT_AREA_F16` + `SAFETY_NUM,_DEN` in `vae_tile_dims`
  (`wan/vae.rs`). Live set is ~linear in tile AREA * act bytes; the safe fraction
  of budget is the lever. The encoder has the SAME working-set wall at native res
  -- the decode tiling pattern (`plan_tiles`/`feather_1d`/`decode_tile`) is there
  to copy when the encoder needs it.
- Wan RoPE3D is interleaved-pair, NOT half-rot (opposite of Qwen3). RoPE freqs
  MUST be packed to the act dtype (`freqs_upload_bytes`): f32 freqs into an f16
  kernel decode to inf -> NaN softmax (see `wan/dit.rs`).
- Wan DiT driver takes `text` as host f32 `[text_seq, text_dim]` (umT5 readback +
  reupload), zero-padded to TEXT_SEQ=512, no cross-attn mask (pyref matches).
  Clean e2e seam; revisit if it costs at native.
- Shared-helper layering: Wan DiT modules reach into `z_image::{block, embedders,
  rope_embedder, seq}`. Decision pending: extract a `thinfer-models` common
  module vs leave the reuse. Do before the family grows past Wan; not blocking.
- Video staging: per-frame PNG sequence per side (py_/ours_) + tiled contact
  sheet. MP4/WebM in the CLI only (openh264); the e2e stages raw PNGs.
