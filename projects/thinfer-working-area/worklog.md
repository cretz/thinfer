# Worklog

Forward-looking only. Git history is the changelog, the code is the record.
Engine-wide design: `plan-details.md`. Per-model plans are separate files (see
Status).

## Status (shipped -- DO NOT DISTURB)

- **LTX-2 19B (Phr00t rapid merge)** -- distilled + native I2V. Checkpoint variant
  on the LTX-2.3 pipeline. Net-new: FE V1 text head + caption projection, 2-layer
  ungated 3840 connector, 6-way block modulation, native I2V (frame-0 latent replace
  + denoise-mask blend), reconstructed LTX-2 VAEs, LTX rewriter (concise-EN
  Qwen3-VL). KNOWN: the merge is human/portrait-specialized -> weak compositional
  T2V (single subjects work; multi-subject falls to the human prior). I2V (frame-0
  anchor from a good image model) is the intended fix, not a conditioning lever.
  `ltx2-rapid-plan.md`.
- **LTX-2.3 distilled + Sulphur-2** (22B joint AV) -- `ltx-plan.md`.
- **HunyuanVideo 1.5 T2V** (lightx2v 4-step, 480p) -- shipped.
- **hunyuan-video-1.5-ti2v** (causal AR, minWM dmd) -- `hy15-causal-plan.md`.
  Text-only AR drift at 480p is a regime effect (OOD for the i2v ckpt); I2V anchor
  damps it.
- **Wan2.2-T2V-A14B** -- shipped; attn-window default W=3. Device-loss + perf fixes
  landed (transient-reserve double-count in residency acquire; coopmat ragged-M OOB
  staging under-alloc -> pad to next multiple of WM; cross-block fence carry).
  Background prefold of the LoRA fold cache hides ~40-50s/expert behind encode. 6G
  VRAM budget works (832x480 f33 ~596s). HARD RAM: no wan22/anyflow auto-runs
  (23-30GB host fold cache at any res; batch runs need explicit OK).
- **AnyFlow-Wan2.1-T2V-14B** -- shipped; default 2 steps (quality-neutral); hybrid
  W=3 default; tiny-VAE available; vault LoRA fold end-to-end (community + PEFT key
  conventions). NSCLv1 (noncommercial).
- **FastWan2.2-TI2V-5B** -- Q8 block_transcode; UniPC default. The video CLI default.
- **LongLive-2.0-5B** (AR long/multi-shot) -- shipped. OPEN BUG: `longlive_parity`
  vel_c0s0 FAILS pre-existing at HEAD (gate stale vs weeks of kernel changes);
  root-cause = next LongLive session's first item. Re-measure WARM at 576.
- **Krea 2 Turbo** -- shipped, coopmat+fast_sdpa default (512x512x8 ~99s).
- **DreamID-V-Wan-1.3B-Faster** (diffusion video face-swap) -- ported, parity +
  eyeball PASS, app/serve/web wired w/ encrypted RAM-only video upload + live DWPose
  mask. `dreamidv-plan.md`. Frozen at ~5s/480p.
- **Face-swap** (HyperSwap ONNX) -- shipped. `faceswap-plan.md`. XSeg occlusion mask,
  GFPGAN enhancer, HyperSwap own-mask, strided detection, bitrate + trim (all opt-in,
  CLI+serve+web wired). Perf: CPU/GPU 3-stage pipelining + frame batching B=4 ->
  ~88ms/frame at 1920x1242 (was 207ms serial). Decode robustness: multi-param-set
  annexb + openh264 error concealment.
- **Ideogram-4**, **Z-Image**, **Qwen-Image(-Edit)-Rapid** -- shipped.
  Qwen-Image-Rapid is NOW the image CLI default.
- **User LoRA vault** -- CLI/serve/web (Argon2id + AES-256-GCM, per-entry
  multi-password, model-scoped). Fold wired for Krea2 + Qwen-Image(-Edit) + AnyFlow.
  Pending real-adapter eyeball.
- **i8 DP4A matmul ON by default** (`--no-i8-matmul` = bf16 reference).

## Lessons / dead-ends (do not retry)

- **HyperSwap conv is at the wgpu/naga ceiling** (memory/reuse-bound, not
  ALU-bound). Every kernel-level lever measured flat-or-worse: tiling, i8, bf16,
  32x32-occupancy tile, vec4 128-bit loads, Winograd F(2,2,3,3), double-buffered
  shared loads. The wins were SYSTEM-level: CPU/GPU pipelining + frame batching
  (B=1 forward is ~13x over compute-ideal because each dispatch underfills the GPU).
  Remaining levers are all system-level (batch detection -- BLOCKED by an executor
  batch>1 correctness bug in SCRFD; f16 acts; codec), none conv-kernel.
- **Coopmat is a MATMUL-ONLY win.** Shared-memory staging is a dead end (slower than
  L2-served global coopLoad). Flash-attn coopmat SDPA ~13x slower than `sdpa_sg`.
  cross-qkv coopmat DEVICE-LOSES (un-normed text > f16 65504). Coopmat device-loses
  for M < WM(32) AND fringe-reads past the A-side alloc for ragged M -> pad staging
  to `next_multiple_of(WM)`. naga gotcha: `var c: coop;` in a loop null-inits ONCE.
- **DiT denoise is at the WGSL matmul ceiling** (latency/occupancy-bound); bigger
  tiles BACKFIRE; weight-only quant does NOT speed compute. Measure via e2e
  `gpu_ms by pipeline`, not microbench.
- **i8 DP4A**: qkv/ffn_up only (normed A-sides). proj/ffn_down carry outliers ->
  f16-cast coopmat OK, i8 acts NOT. Cross-attn-qkv from un-normed text: never.
- **Q8_0 is the quality+perf baseline for big DiTs** (Q4_K per-request fold was ~2x
  SLOWER and broke quality). Q4_K_M = footprint option only. quantize_row panics
  for Q5_K/Q6_K (dequant-only), so those are not fold-shrink options.
- **Weight transcode must catch every float source encoding** (a Q8 arm matching
  only Bf16 let F32/F16 fall through to dense while the site read Q8 -> garbage).
- **BLOCK-WIDE f16 is a dead end** on big-DiT residuals (outlier channels > 65504);
  bf16 residual is load-bearing. Mixed-precision f16 SDPA (post-norm Q/K/V cast) is
  safe and shipped.
- **VAE decode is conv-GPU bound**; do NOT retry the conv3d im2col hoist (slower).
  Tiny (TAEHV) decoders are the fast default where they exist; LTX tiles its VAE
  (seed BELOW budget, balanced re-seed on OOM).
- **Never run fp32 CPU pyref above tiny dims** (40GB host at 256x256).
- **Budgets: stream in/out under pressure, never predict reserves.** No carve-out
  reserves. Do NOT pin a weight slice when working set >> VRAM budget (starves
  same-size recycle -> arbiter reclaim mid-compute -> device lost). Pin only when
  the whole DiT fits. `--vram-budget 6G` device-loss was fixed by the residency /
  coopmat fixes above, NOT by eager-destroy.
- **LTX**: off-subject output = resolution/adherence regime, NOT text encoding
  (512+ two-stage widescreen is the regime). Gemma encoder MUST run F32 acts. The
  gemma `(1+w)` norm bake is REAL. Don't flip the LTX DiT to strict budget.
- **Attention windowing (W=3) breaks temporal identity** on multi-subject/tracked
  content. Hybrid (full step 0, windowed after) restores scene stability but still
  drifts attributes; pixel-MAD is blind, eyeball is the gate. Full attention is the
  identity tier.

## Carry-forward gotchas (engine-general)

- **Ops reading aux params as f32 must dequant bf16 weights first** (passes
  conformance -- no such dtype pair there -- fails in the model).
- **Passthrough 1-D F32 weights arrive on GPU as BF16** under bf16 acts; gemma-style
  `(1+w)` norm needs the bf16-weight kernel variant.
- **F16 params the model reads RAW must use `register_raw_param`**;
  `register_linear`'s transpose silently scrambles them (structured artifacts, no NaN).
- **umT5 / large-residual encoders MUST run bf16 acts** (f16 overflows -> washed
  blob). Check non-finite, not just NaN.
- **Module-level matmuls need their OWN bf16 `matmul_module` pipeline.**
- **Shared-single-pipeline encoders can't do per-layer mixed K-quant** (scheme fixed
  at compile from the first probe). Mixed `_K_M` GGUFs bump attn_v/ffn_down to Q6_K.
- **RoPE**: Wan RoPE3D + Hunyuan = interleaved-pair; Qwen3 + LTX = half-rot.
- **GGUF tensor padding**: F32/F16 narrow arms slice `elements*size`, not on-disk bytes.
- **Cargo workspace root is `thinfer/projects/`** (not the repo top). Watch
  tee-pipelines masking the real exit code.
- **Windows**: a running serve LOCKS `thinfer-serve.exe`; `cargo build --release`
  silently leaves the old exe (exit 0). STOP serve before a release rebuild.

## Crate layout + serve (shipped reference)

- Layering: core < models < native < app < {cli, serve}; wasm `thinfer-web` is its
  own substrate. serve = separate binary over shared `thinfer-app`
  (`JobExecutor` trait). Web UI is `include_str!`-baked -> web edits need a serve
  rebuild. One fresh wgpu device per JOB. No cross-request warm state.
- All image inputs ride in RAM (`ImageBytes`, redacting Debug). Only on-disk bytes
  are the encrypted video spill (AES-256-GCM, ephemeral RAM key) + the output
  artifact; both wiped by delete-on-fetch.
- **Deploy**: `cargo build -p thinfer-serve --release`, launch with `--config
  <serve.toml>` (local serve ALWAYS self-signed HTTPS), confirm "listening (https)".
  `ram_budget=28G` (TI2V host KV cache ~14GB at 77f). STOP a running serve before a
  release rebuild (Windows locks the exe; see gotchas).
- CLI/web parity: every generate flag wires through serve wire.rs/api.rs + web UI in
  the same change. Never log prompt text.

## Running the e2e / measuring

Card = RTX 5070 Laptop (8GB); keep budgets <8GB. All serial (`--test-threads=1`).
Always `THINFER_TRACE=1 THINFER_POWER_PREF=high` + `THINFER_E2E_PNG_DIR` for
staging; read the `gpu_ms by pipeline` rollup first. coopmat needs
`THINFER_POWER_PREF=high` (else iGPU, no coopmat / 4x slower).

- LTX: `rapid_t2v_e2e_health` / `t2v_e2e_health` (`--features ltx-e2e`); component
  parity gates ONE AT A TIME (12.5GB encoder OOMs a multi-test binary).
- Causal TI2V: `i2v_e2e_health` (`--features hunyuan-e2e`); `THINFER_I2V_T2V_PROBE=1`
  text-only; `THINFER_AR_DIAG=1` per-stage.
- Hunyuan T2V: `t2v_e2e`, `t2v_perf`, `dit_parity`.
- FastWan: `video_e2e`. LongLive: `longlive_parity`/`longlive_e2e`. AnyFlow:
  `anyflow_e2e`. Wan2.2: `wan22_e2e`.
- CLI run: `THINFER_TRACE=1 THINFER_POWER_PREF=high thinfer generate ...
  --vram-budget 5G --ram-budget 5G` (TI2V needs bigger ram for the KV cache).

## Open / Next (by track)

- **ltx2-rapid**: remaining perf (both large/uncertain) -- two-stage half-res
  upscaler (unproven LTX-2 latent transfer); encoder cold weight-streaming
  (structural; Q4 encoder = quality cost).
- **wan2.2/anyflow**: per-expert LoRA strength knob (CLI+web); fold-cache ram-budget
  design (rides outside the ram budget today); 832x480 quality A/Bs; Q4_K_M ladder
  decision. AnyFlow 512x320 tiny-VAE decode OOM (budget-derived chunk sizing leaves
  no headroom; consider strict-mode shrink+retry like LTX).
- **face-swap**: FF-Android blob-video playback (needs a trusted cert for a
  ServiceWorker, or fMP4/MSE); widen audio passthrough beyond H.264+AAC-LC; SCRFD
  batch>1 executor bug (unlocks detection batching); f16 acts (per-model, the
  bandwidth ceiling-raiser); job-dir artifact TTL/reaper.
- **DreamID-V** 480p footprint/perf: Q8_0 block_transcode + VAE tiling; remove the
  per-request cold fp32->bf16 DiT transcode.
- **User LoRA vault**: real Civitai adapter eyeball end-to-end (Krea2 + qwen-rapid).
