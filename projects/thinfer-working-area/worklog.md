# Worklog

Forward-looking only. Git history is the changelog, the code is the record. Past
work appears here ONLY as a one-line lesson or a do-not-retry. Engine-wide design
+ kernel/runtime state: `plan-details.md`. Per-model porting plans are separate
files (see Status). Scratch is ephemeral; nothing here depends on a scratch file.

## NOW / NEXT

**Video session shipped 3 knobs (2026-06-27, assistant on auto, GPU):**
1. **Duration cap is no longer SILENT.** `VideoPlan.warnings` (request.rs) carries a
   notice when a `--duration`/default is capped to the per-res VRAM envelope;
   surfaced via the progress sink in `ltx.rs` + `executor.rs` (CLI stderr + web
   SSE). Explicit over-budget `--frames` stays a hard error. Fixes the "asked 5s,
   got 2s, no idea why" complaint -- at 1280x704 the user now sees "requested ~5.0s
   was capped... lower the resolution (1024x576 ~3s, 768x512 ~5s)". Test
   `ltx_duration_cap_warns_and_caps`. (5s physically cannot fit at 1280x704 on 8GB;
   the cap is real, only the silence was the bug.)
2. **Q4_K_M Gemma text encoder = SHIPPED as a real option (CLI + web + wire),
   APPLIES TO ALL LTX/SULPHUR MODELS.** New role `ENCODER_GGUF_Q4`
   (`gemma-3-12b-it-Q4_K_M.gguf`, 7.3G vs 12.5G) in BOTH manifests. The encoder
   gained a per-site `dense_dequant` map + kind-probe (mirrors `ltx::dit`'s
   mixed-quant path) so it reads the Q4_K/Q6_K mix directly -- the matmul pipeline
   is bf16-reading regardless, only the dequant step varies by probed kind.
   GPU-measured on the PLAIN-LTX manifest (`encoder_perf LTX_ENC_QUANT=q4_k_m`):
   **Q8 forward 15.7s vs Q4 5.6s (~2.8x)**, all 49 states finite. End-to-end smoke
   (`--encoder q4`, 512x320) = coherent, photoreal, on-prompt. Surface: new
   `EncoderQuant {Q8,Q4}` enum (model.rs) -> `VideoRequest.encoder` -> wire
   `encoder: Option<EncoderQuant>` -> CLI `--encoder q8|q4` (default q8) -> serve
   api -> web "Text encoder" dropdown (LTX-only, app.js reveals). Role resolved by
   `VideoRequest::ltx_encoder_role()` (request field; `THINFER_LTX_ENCODER=q4_k_m`
   still forces Q4 as a power-user/test override). Q8 stays the default
   (conditioning quality). Cold/RAM-pressured loads (the 103s case) also benefit
   from the 42%-smaller file + less page-cache evict. **The encoder + cap fix are
   LTX-WIDE (shared encoder, `is_ltx()`-gated, `resolve_ltx`); only the distill
   experiment (#3) was Sulphur-only.** Serve REDEPLOYED with all of this.
3. **Sulphur distill-LoRA experiments = NO win; condsafe@1.0 stays the default.**
   The fold (`ltx::lora`) now stacks N `(lora, strength)` pairs
   (`re-encode(dequant(base) + Σ strength_i·B_i@A_i)`) -- a real, durable upgrade
   from the single-lora fold. Behind `THINFER_SULPHUR_DISTILL`: `stack` =
   `condsafe@0.7 + distilled-lora-384@0.5` (the official ComfyUI distilled-workflow
   recipe; new role `DISTILL_LORA_384`, Lightricks 7.6G); `rank768` = the
   `sulphur_lora_rank_768` standalone. Default unchanged = `[condsafe@1.0]`.
   **GPU-eyeballed all three (1280x704, seed 42, candid-smile prompt):**
   - `rank768` = UNDERCOOKED MUSH (brown blur, faint lattice). Root cause:
     `sulphur_lora_rank_768.safetensors` is NOT a step-distill LoRA -- it is a
     CONTENT lora for the standard CFG workflow (the distilled workflow never loads
     it). Folding a non-distill lora into the 8-step CFG-free path can't converge.
     So "did we pick an old/weak distill?" = **NO**: our `condsafe` IS the correct
     distill artifact. (kept as repro-only.)
   - `stack` = coherent + sharp, but MORE illustrative/anime + still off-prompt
     (3 stylized girls, not the candid woman). Lower distill strength did NOT cut
     the illustrative skew -- it increased it. NOT a win for photoreal.
   - `condsafe@1.0` (run6, the shipped default) = the most realistic of the three.
   **Verdict: distill-LoRA choice does not move photorealism/adherence.** The
   illustrative skew + off-prompt subject is the CFG-free regime ceiling (already
   established), NOT the distill. The Sulphur content is ALREADY in our `sulphur_dev`
   GGUF base (the workflow instead uses generic dev + `sulphur_final` content lora;
   `sulphur_final` isn't in any public repo -- a user-renamed file). COST seen: the
   fold pushes host RAM to ~40GB (rank768) / ~40GB (stack); fine on the 63G box.
- Grid-artifact report (user saw a lattice on 720p): our 1280x704 two-stage CLI
  mp4 is PROVABLY clean (high-pass + Laplacian + 4x-contrast flat regions show only
  subject edges, no VAE-seam/upscaler-checkerboard/macroblock grid). The user's grid
  is browser-export/viewer-side (contact-sheet preview, lower-bitrate re-encode, or
  motion-only shimmer) -- need their actual clip to close.
- **DONE: Q4 encoder promoted to a real CLI/web/wire option (see #2) + serve
  redeployed.** Possible follow-up: default `--encoder q4` for the `*-q4` model
  variants (they already signal a footprint/speed tradeoff) -- not done (Q8 is the
  conditioning-quality default everywhere for now). The distill knobs
  (`stack`/`rank768`) stay env repro-only (no quality win). If chasing photoreal
  further, the lever is the regime/CFG (vetoed) or a different base, NOT the
  distill. The multi-lora-stack fold + per-lora strength is in place for any future
  recipe. Bound the unbounded host fold cache before any stacked fold ships.

**Sulphur-2 = FIXED + served + eyeballed sharp (2026-06-26).** The blurry/
undercooked output is RESOLVED by folding the distill LoRA into the dev DiT at
load. Root cause confirmed from the Sulphur ComfyUI workflows: the published GGUF
(`vantagewithai/Sulphur-2-Base-GGUF`, `sulphur_dev-{Q8_0,Q4_K_M}.gguf`) is the
BASE (`dev`) checkpoint, NOT distilled -- the `t2v distilled` workflow loads dev +
the distill LoRA at strength 1.0 (CFG=1), while `t2v base` (no LoRA) needs ~50
steps + CFG 3.6. Running dev un-distilled through our 8-step CFG-free sampler was
the undercooked denoise. Re-eyeball (512x320, 25f, seed 42, the original prompt):
sharp, coherent, temporally consistent -- the denoise converges. (Style skewed
illustrative; that is prompt-adherence under our CFG-free veto, not the bug.)
- **Fix = `ltx::lora::LoraFoldSource`** (new, self-contained; shipped Ideogram
  fold left untouched per DO-NOT-DISTURB). Wraps the DiT GGUF + LoRA, serves
  `re-encode(dequant(base) + B@A)` at auto-discovered sites, passes the rest
  through. ENCODING-PRESERVING (Q8_0 stays Q8_0, bf16 stays bf16, same [N,K] +
  exact byte size) so it is a byte-shape drop-in for the DiT loader -- no loader/
  register changes. Folded bytes cached in RAM (residency re-acquires per step).
- **LoRA specifics** (`distill_loras/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_
    condsafe.safetensors`, repo `SulphurAI/Sulphur-2-base`): bf16, keys
  `diffusion_model.{X}.lora_{A,B}.weight`; NO `.alpha` -> scale 1.0 (rerank baked
  it in) -> plain B@A. Rank VARIES per tensor (read from `lora_A.shape[0]`: attn
  36, ff 45-72, gate_logits 14-27). Non-zero sites = video/audio self-attn
  (`attn1`/`audio_attn1`, incl `to_gate_logits`) + per-modality FFN +
  patchify/proj_out; cross-attn (`attn2`/`audio_attn2`/`*_to_*_attn`) + all adaLN
  are zeroed in the file (B==0) and auto-skipped. Same STAGE1_SIGMAS 8-step CFG-
  free path; CFG stays vetoed.
- **Wiring**: `SULPHUR_MANIFEST` adds role `DISTILL_LORA` (Sulphur-only;
  `required_files` appends it when `is_sulphur`); `ltx.rs` wraps the DiT source in
  a `DitSource` enum (Plain GGUF | Folded) so the DiT-resident phase stays one
  monomorphic type. LTX distilled models = Plain passthrough (already distilled).
- Tests: `ltx::lora` unit tests (fold_add, encode_rows Q8/bf16 round-trip,
  discover_specs zero-skip + rank read) GREEN. Serve rebuilt --release +
  relaunched (HTTPS up). NEXT (optional, not blocking): if illustrative style is
  unwanted, the only CFG-free lever is a different sigma schedule (the distilled
  workflow uses a shorter manual schedule) -- tune only if user asks.
- **`sulphur-2-q4`**: folded matmul sites re-encode to Q8_0 (`fold_out_enc`), which
  `register_attn/ff` already hint -> no Q4_K-align / Q6_K-requant hazard; non-folded
  tensors stay Q4_K. Should work but GPU-UNVALIDATED (default `sulphur-2` Q8 is the
  validated fix; q4 mirrors the LTX-q4 unvalidated status).

**USER COMPLAINTS post-fold = RESOLVED + shipped (2026-06-27, assistant ran GPU).**
The "coherent but wrong subject/action" (man for a woman, off-script, animated-not-
photoreal) was REGIME: low-res single-stage is OOD. Fixed by defaulting LTX/Sulphur
to the in-distribution WIDESCREEN TWO-STAGE regime. Eyeballed at 1280x704 + 1024x576
(seed 42, the candid-smile-says-Hi prompt): photoreal woman, slow smile to camera,
correct scene, audio "Hi" present. Samples in scratch/ltx-regime (run2 1280x704x49f,
run5 1024x576x73f).
- **New shipped LTX defaults** (validated end-to-end via bare CLI = 1280x704 / 49f /
  two-stage / 11 steps): `video_defaults` LTX -> **1280x704** (was 768x512);
  `two_stage_default()=is_ltx()` so `upscale` defaults ON for LTX in serve/cli/web
  (single-stage at the widescreen default OOMs and low-res single-stage is OOD, so
  two-stage is the only good 8GB regime); web LTX presets are widescreen-first +
  upscale box pre-checked.
- **8GB frame envelope (the real constraint = FRAMES dominate VRAM, not just dims).**
  The two-stage stage-2 denoise runs the DiT at FULL res, and its activation peak +
  per-block streaming alloc/free fragmentation set the ceiling. Empirical (RTX 5070
  8GB, `peak_live`): 25f@1280x704=4.2G, **49f@1280x704=5.4G (safe, the default)**,
  73f@1280x704 DEVICE-LOST, 97f@1280x704 OOM (6.5G at fail); 73f@1024x576=5.2G safe.
  Encoded as `LTX_MAX_LATENT_CELLS=6300` (full-res `f_lat*h/32*w/32`) +
  `ltx_max_frames(w,h)`: `resolve_ltx` defaults frames to `min(121, max)` and
  REJECTS explicit over-budget `--frames` at submit (fail fast, no mid-denoise
  device-loss). Unit test `ltx_frame_cap_tracks_resolution`. So 1280x704->49f(~2s),
  1024x576->73f(~3s), low-res still gets the full 5s.
- **`LTX_VRAM_BUDGET_CAP=2G` (ltx.rs).** The DiT (22.8GB) always streams per-block,
  so a high budget only steals the device slack stage-2 needs -> clamp LTX weight
  residency to 2G (serve's 5G default OOMs stage 2; per-step time unchanged, streaming
  dominates either way). Only lowers an over-large budget; cap point = ltx.rs `run`.
- **GOTCHA: a stage-2 OOM at high (res x frames) loses the DEVICE (hard panic in the
  wgpu poll thread), not a catchable alloc error** -> leaves a hung 22GB process
  holding VRAM (had to `Stop-Process`). The frame cap keeps the default well clear;
  but engine-side, stage-2 has no graceful-OOM-retry like the VAE decode reseed (a
  follow-up: catch + fail the job instead of device-loss).
- **Old NB "assistant must NOT run GPU" is LIFTED for solo/overnight sessions** (user
  asleep, explicitly set me on auto to run GPU). The contention rule only applies
  when the user is actively using the card. Clean up any hung thinfer.exe after a
  device-loss before the next run (`Get-Process thinfer` / `Stop-Process`).
- **WATCH: Sulphur fold host-RAM cache.** `LoraFoldSource` caches every folded site
  (video+audio self-attn + FFN + patchify/proj_out across 48 blocks) as Q8 bytes in
  an unbounded host `HashMap` (NOT governed by `ram_budget`), on top of the mmap'd
  GGUF. For 22B that is ~12-18GB host RAM. Fine on a big-RAM box; revisit (bound the
  cache / evict folded bytes after upload) if host RAM is tight. Not the cause of
  the VRAM OOM above, but a Sulphur-only footprint cost vs plain LTX.

**LTX-2.3 distilled = functionally COMPLETE + served (CLI + thinfer-serve + web).**
22B joint audio-video (Lightricks LTX-2). All components parity GREEN, e2e health
GREEN, eyeballed. Architecture + porting detail live in `ltx-plan.md` + the code.
Models: `ltx-2.3-distilled` (Q8 DiT, default/quality baseline) +
`ltx-2.3-distilled-q4` (Q4_K_M, footprint-only option). Perf defaults: f16dp4a DiT
+ f16 VAE decode.

**PROMPT LOGGING REMOVED (2026-06-26).** Stripped prompt text from all generate
DIAG logs (`executor.rs` x5: 4 image + 1 video `prompts`; `ltx.rs` x1). Structural
fields (model/dims/steps/seed/tokens/budgets) stay; prompt content never logged.
Serve rebuilt + relaunched with this change. See [[feedback_no_prompt_logging]].

**SERVE IS CURRENT (redeployed 2026-06-26).** `thinfer-serve --release` rebuilt +
relaunched; now carries everything: **single-stage distilled DEFAULT** + opt-in
`--upscale` / `upscale` / web "Upscale (hi-res refine)" checkbox (see ltx-plan
two-render-paths decision), f16 VAE decode default, f16dp4a DiT default, per-job
fresh wgpu device (VRAM-leak fix), serve `--config` hardening, web tiny-vae label
fix + q4 dropdown. Both e2e paths GREEN (single-stage @512x320, two-stage @64x64
via `THINFER_E2E_UPSCALE=1`). Web prompt-splitter fixed: only LongLive splits the
box into per-line shots; LTX/Wan send the whole box (blank-line prompts no longer
400). Relaunch recipe under Crate layout.

**PENDING (small, decided not done): rename the knob `upscale`/`--upscale` -> `fast`
/`--fast`** (default off = single-stage quality; on = two-stage half-res+upscale,
faster but lower quality on small dims). User chose "Fast" naming. Pure rename of
the bool already wired across request/wire/api/cli/ltx.rs/e2e/web; help+tooltip
should say "faster; lower quality below ~1024px".

**Open validation (user, in browser):** (1) SUPERSEDED -- 512x320 single-stage is
OOD (off-prompt), confirmed; the default is now 1280x704 two-stage (see RESOLVED
block). (2) deliberate-OOM smoke -> next job runs clean (per-job device); (3) Q4_K_M
DiT GPU finite check + smoke gen (`dit_perf LTX_DIT_QUANT=q4_k_m` was interrupted by
GPU contention).

**LTX adherence RESOLVED (2026-06-26): it was REGIME, not a thinfer bug.** The bad
runs were 512x320 SINGLE-STAGE = out-of-distribution for the distilled model; user
confirmed the character speaks the requested words once dims/regime are right.
Verified faithful to upstream (so none is the cause): Gemma enc, FE V2, connector
KV, AdaLN + `prompt_adaln` KV modulation, full DiT block (cross-attn + per-head
gating + av-cross), sampler -- all bit-tight; product tokenization correct (66 clean
tokens, quoted `Hi` survives); `build_video/audio_positions` (the only un-gated
serve path) match upstream `get_patch_grid_bounds`+`get_pixel_coords`+`/fps`
bit-for-bit. Ecosystem norm for distilled (ComfyUI): TWO-STAGE, WIDESCREEN 16:9/21:9
(portrait/square distort), dims /32, ~1280x704+, 8-step+upscale, **CFG=1 (off; do
NOT add CFG -- user vetoed the 2x cost)**, frames 8n+1.
- Full-e2e-pyref plan ABANDONED: 22B won't fit this box even with offload (bf16
  ~44GB host > ~10GB pyref ceiling) and diffusers has no GGUF LTX2 transformer path.
  thinfer is the only engine here that runs the full 22B pipeline.

**NEXT (forward):**
- **DONE: product defaults no longer push users OOD** -- LTX now defaults to
  1280x704 two-stage with the per-res frame cap (see RESOLVED block above). Possible
  follow-ups: (a) stage-2 graceful-OOM retry so an over-budget config fails the job
  instead of device-losing; (b) a dynamic vram cap keyed to detected card VRAM
  instead of the hardcoded 2G (for >8GB cards); (c) make the web duration placeholder
  recompute per selected preset (now static 2s = the 1280x704 figure).
- **Parity-test tokenizer hole.** encoder/connector pyref tokenizes via
  `AutoTokenizer(gguf_file=...)` = DEGENERATE 145-token char-frags; engine is fed
  those same garbage ids, so the gates validate correct math on the WRONG token
  distribution. Fix = point the pyref at the product `tokenizer.json` (role
  TOKENIZER, 66 clean tokens); re-run encoder_parity + connector_parity to
  re-confirm the bands on real tokens (GPU; ask first).

**Open / not blocking:**
- LTX VAE further speed (ONLY if wanted -- it is < DiT now): split `run_graph` into
  per-up-block submits (bound peak to one block -> fewer/larger tiles), or a
  TAE-style tiny video autoencoder (new weights + decoder, larger scope).
- Smaller Gemma encoder (~16s, stream-bound, ~2% of e2e): BLOCKED on a user call --
  needs a ~7-9GB Q4_K_M/Q5_K_M gemma download + the per-quant-kind dequant dispatch
  ported to the encoder `mm` helper (template = `ltx/dit.rs` AttnDequant/FfDequant)
  + revisits the deliberate Q8 conditioning-quality choice (user rejected Q4 QAT).

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
  the VAE decode + the LTX-distilled DiT acts are fine (validated, see LTX below).
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
    variant) AND the residual overflows f16. All sites dense (no i8: massive-
    activation outliers; once-per-request anyway).
  - The gemma `(1+w)` norm bake is REAL and STAYS: llama.cpp pre-bakes +1 into
    norm.weight; both engine (UnitOffset) and pyref (HF `1+w`) add +1 -> they match.
    Don't "fix" it.
  - DON'T flip the DiT to strict budget: it relies on overshooting into device
    slack at the 768x512 default; strict would reject configs that work today (it
    has no adaptive smaller-retry). DiT soft budget is BY DESIGN.

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

- **LTX-2.3 distilled** (22B joint audio-video) -- shipped, see NOW/NEXT +
  `ltx-plan.md`. Q8 default + Q4_K_M option; f16dp4a DiT + f16 VAE.
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
  image-edit) -- shipped, deployed to serve (CLI+serve+web+OpenAPI), all components
  parity GREEN, e2e health GREEN. `qwen-image-plan.md`. OPEN: user eyeballs a
  browser gen (full-DiT t2i pyref OOMs); opt-in `--ref-size` perf knob (edit-only).
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
    the exe in a foreground Bash call (it blocks = "hang"). (The CLI now rejects a
    bare-positional config instead of silently booting HTTP defaults.)
- DEFERRED: mid-generate cancel (touches shipped generate sigs); serve==CLI
  byte-parity test (weights+GPU gated); disk-backed SSE ring buffer; wasm<->http web
  toggle (server-only by decision).

## Running the e2e / measuring

Card = RTX 5070 Laptop (8GB); keep budgets <8GB (8GB OOMs the device). All serial
(`--test-threads=1`): multiple WgpuBackend in one binary contend on VRAM.
- LTX e2e: `THINFER_POWER_PREF=high THINFER_TRACE=1 THINFER_E2E_FRAMES=121
  THINFER_E2E_HEIGHT=320 THINFER_E2E_WIDTH=512 THINFER_E2E_VRAM_GB=6
  THINFER_E2E_PNG_DIR=<dir> cargo test -p thinfer-conformance --features ltx-e2e
  --release t2v_e2e_health -- --nocapture --test-threads=1` (env dims default to the
  tiny health gate). Component parity gates: `vae_parity`, `audio_vae_parity`,
  `vocoder_parity`, `dit_parity`/`dit_full_parity`/`modulation_parity`,
  `connector_parity`, `encoder_parity` (run ONE AT A TIME -- the 12.5GB encoder OOMs
  a multi-test binary). `dit_perf` = engine-only perf bench (env dims, dumps the
  trace rollup the e2e never does).
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
