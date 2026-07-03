# Worklog

Forward-looking only. Git history is the changelog, the code is the record.
Engine-wide design: `plan-details.md`. Per-model plans are separate files (see
Status). Scratch is ephemeral; nothing here depends on a scratch file.

## NOW: AnyFlow perf (user directive 2026-07-02: "drive down perf here")

AnyFlow-Wan2.1-T2V-14B is PORTED + VALIDATED + SERVE-DEPLOYED (details in the
track section below). 2-step quality at 480p is GORGEOUS on simple prompts.
The problem is WALL TIME. **User's full serve run, 832x480x81f 2-step, seed
4450053076214346960 (`scratch/browser-run/{log.txt,video.mp4}`): encode 7.5s,
step 1 = 635.5s, step 2 = 609.8s, VAE 169s, total 1424s.**

**ROOT-CAUSED 2026-07-02 (phase telemetry in `forward_block_tiled`, diag
`tiled block phases` event): the 81f block is ~13.2s wall = SDPA 9.0s +
pass-B (FFN+cross) 2.7s + pass-A 1.0s + streaming ~0.9s (overlapped).
`sdpa_sg` at s=32760 full attention is ~22 TFLOP/block and runs at ~2.6
TFLOPS-eff: INSTRUCTION-bound (per-key subgroupShuffleXor reduce chain +
scalar softmax per Q row), NOT DRAM-bound (WG 128->256 halved K/V traffic,
gained only 5%) and NOT streaming-bound (all four old suspects were minor).
The user's "10s at 0% GPU" was Task Manager's 3D graph missing the compute
queue.**

LANDED (uncommitted, all clippy-clean; sdpa_sg conformance 13/13 GREEN):
- Cross-attn Q -> i8 site (`QkvSite::CrossQ` shares the self-attn i8
  pipelines; normed A-side). Was dense-pipeline, ~2.5s/block at 81f (~35% of
  33f DiT GPU). K/V from un-normed text stay dense (unchanged).
- sdpa_sg WG 128->256 (`SG_WG`, web-safe at the default 256 limit).
- Depth-2 submit pipeline in the tiled block loop (pass A/B tiles submit
  ahead one; transient reserve accounts 2 tiles). Neutral at 81f (SDPA
  dominates) but strictly better; keep.
- Web UI: attn-window row now visible for anyflow (blank = full attention).
- Per-block phase telemetry (diag-gated) in `forward_block_tiled`. Needs
  `RUST_LOG=info,thinfer::diag=debug` (THINFER_TRACE alone filters to info).
Result: 81f block 15.9s -> ~12.8s (~16% e2e). W=3 attn-window opt-in cuts
SDPA ~3x more (projected ~7.3s/block, ~750s total) -- USER EYEBALL gate
(multi-subject risk, HY second-cat lesson).

LANDED 2026-07-03 (uncommitted, clippy-clean):
- **Tiny VAE (taew2_1) for AnyFlow, VALIDATED**: new manifest role
  `TINY_VAE_WAN21` (`lightx2v/Autoencoders taew2_1.safetensors`, z16 patch-1,
  same `decoder.{1..22}` keys as lighttaew2_2 -- the shared taehv decoder
  needed ZERO graph changes); per-model tiny-role pick
  (`VideoModelId::wan_tiny_vae_role`) in executor + request; web UI vae
  dropdown un-pinned for anyflow. **CRITICAL latent-space fact: taew2_1
  consumes the DiT's NORMALIZED latents as-is (TAESD convention); only the
  lightx2v `lighttae*` retrains take the full-VAE `z*std+mean` denorm
  (upstream gates on "lighttae" in the filename -- lightx2v pipeline.py).**
  `WanVariant.tiny_vae_normed_latents` carries this; wrong transform =
  washed-gray output (tiny-vs-full frame MAD 0.19 pre-fix -> 0.016 post-fix
  at same latent). Env-gated e2e arm: `THINFER_E2E_TINY_VAE=1 anyflow_e2e`.
  Decode 0.2s vs full 4.0s at 256x256x9f; kills the 169s VAE at 81f.
- **Web bug fixed: anyflow request payload never sent `attnWindow`** (row was
  visible but the spec omitted it -> W=3 silently ran full attention). Also
  sends the vae choice now. NEEDS SERVE REBUILD+REDEPLOY before the user A/B.
- `THINFER_SDPA_SG_CL` env override (models/common block compile) for the
  CL=4 A/B; NOT bit-exact across CL (reduce order), gate like a kernel change.

**DEAD END 2026-07-03: i8 SDPA for wan full attention -- measured ~10x
SLOWER, do not enable.** At 832x480x33f (s=13,320) `sdpa_i8` = 16.6s/block
vs ~1.5s f16 sdpa_sg (O(s^2)-scaled from the 9.0s/81f baseline). The kernel
is thread-per-Q-row scalar unpack+fma (no dot4I8Packed, no subgroup co-op
over D); a DP4A rewrite of the QK dot fixes <=40% of per-key work (the
softmax-V accumulate can't dot4) and SDPA here is instruction-bound, so i8's
halved K/V bytes buy nothing. Only an sdpa_sg-shaped subgroup i8 kernel
could compete, and its only edge over f16 would be bandwidth we don't need.
The wiring stays (off by default): `THINFER_WAN_I8_SDPA=1` env opt-in in
wan/pipeline.rs; sdpa_i8 kernel gained Q-row chunking (`row0`) + dense
f16/bf16 out modes, conformance 7/7 GREEN incl chunk-bit-exactness;
`op_sdpa_impl` i8 branch now gates on paired inputs (dense callers fall
through). Parity note: i8 latent was statistically EQUIVALENT to the f16
baseline on FastWan video_e2e (pre-VAE slope 0.9866 vs 0.9835 baseline);
only the chaotic vae_rgb band missed (0.9275 vs 0.94 floor; baseline itself
passes by 0.0025). Numbers in scratch/{video_e2e_i8,video_e2e_base}.log.

**SDPA kernel-tuning verdicts (2026-07-03, all at 832x480x81f, s=32760,
vs the 9.0s/block CL=8/QR=1/WG256 baseline): the kernel is AT this card's
practical full-attention ceiling. Do not re-try:**
- CL=4 (2 shuffle hops, BR=64): sdpa 10.3s mean over 80 blocks = ~14%
  SLOWER (more per-lane serial D work loses to the saved hop).
- QR=2 (Q-register blocking, landed as `THINFER_SDPA_SG_QR`, default 1):
  sdpa ~9.4s = neutral-to-4%-slower; the doubled q/o register set
  (occupancy) ate the amortized K/V tile loads. Codegen + tests stay
  (bit-exact vs QR=1; sdpa_sg conformance 16/16 GREEN incl
  `sdpa_sg_qr2_bitexact_vs_qr1`); knob may help other adapters.
- i8 SDPA: see DEAD END above (~10x slower).
Remaining full-attention perf comes from changing the WORK, not the
kernel: attn-window W=3 (opt-in, ~3x SDPA) and the steps dial.

DONE 2026-07-03: **post-change 81f e2e GREEN** (832x480x81f 2-step, CL=4 +
tiny VAE, 1183s test: denoise 1169s incl cold transcode, tiny-VAE decode
6.6s (!) vs 169s full, latent std 1.058, video std 0.61, motion MAD 0.24,
frames GORGEOUS -- red-car clip eyeballed at product dims,
`scratch/anyflow_81f_cl4_png/`). At the deployed CL=8 kernel this implies
~1010s denoise -> **~1070s total for 5s 480p with vae=tiny (was 1424s)**.

SERVE REDEPLOYED 2026-07-03 (tiny-VAE + attnWindow/vae web fix live).
**USER EYEBALL 2026-07-03, 832x480x81f 2-step W=3 vae=tiny (browser, seed
10096709036188377068): PERF CONFIRMED -- encode 7.4s, steps 312.5s/299.7s,
tiny VAE 10.7s, total 632s (~10.5 min; was 1424s). QUALITY FAILED: scene +
wardrobe change every ~1-2s == the W=3 window horizon (21 latent frames,
+-3 visible ~= +-0.75s; frames beyond it share no attention path). Plain
all-steps W=3 on AnyFlow = DEAD for temporal identity.** Same mechanism as
the HY second-cat, expressed as drift.

**HYBRID STEP-WINDOWING PROVEN + SHIPPED 2026-07-03 (anyflow default).**
Engine A/B at 832x480x81f 2-step, same seed/prompt vs the full-attn red-car
reference: drift (MAD-vs-frame0 at frame 80) = 0.120 full / 0.216 W3-all /
**0.124 hybrid (step 0 full, W=3 from step 1)** -- hybrid RESTORES
full-attention temporal consistency; frame-80 eyeball scene-stable. Denoise
873s hybrid vs ~1030s full (CL8-normalized) vs 608s W3-all. Shipped as the
AnyFlow arm's default (`step_attn_window(.., default_from=1)` in
pipeline.rs): a user attn-window on anyflow now means "full step 0, then
windowed". Wan2.2 keeps its shipped all-steps W=3 (default_from 0) until
its own eyeball. `THINFER_WAN_WINDOW_FROM_STEP` env overrides both; e2e
takes `THINFER_E2E_ATTN_WINDOW`. No new user flag -> CLI/web parity holds
by construction. PNG evidence: `scratch/anyflow_81f_{cl4,w3all,hybrid}_png`.
**User tier map (5s 480p, tiny VAE): ~10.5 min W3-all (BROKEN quality),
~14 min hybrid W=3, ~17.5 min full attention.**

SERVE REDEPLOYED 2026-07-03 (2nd time): hybrid default + job-lifecycle log
lines (id/dims/seed/elapsed at info, NO prompt -- lost-tab seed recovery;
the engine's diag generate-start stays muted in serve).

LANDED 2026-07-03 PM (uncommitted, user directive: pick the fastest-at-quality
model and drive it down; user vetoed FastWan as primary = quality too low at
5B, so AnyFlow stays the target; FastWan wins banked anyway):
- **FastWan 5B Q8_0 block transcode** (was bf16: ~9.8GB re-streamed per step at
  480p = 3.5x streaming-bound vs its ~13s/step GPU compute; 81f 480p full-VAE
  decode also OOMs at 6GB -> tiny VAE is the product path there). Parity gate
  GREEN at 256 (vae_rgb slope 0.9426 vs 0.94 floor, same band edge as the bf16
  baseline). LongLive split to its own `WanVariant::longlive_2_0_5b()` (keeps
  bf16; its parity bands pin bf16 today).
- **Per-block weight-prep hoist for the tiled Wan path** (`PreparedTileWeights`):
  dispatch_matmul_site re-dequanted each block weight PER ACTIVATION TILE
  (32x redundant at 81f: dequant_i8_qkv_self/ffn_up + dequant_f16_proj/
  ffn_down = ~40s GPU of the 873s hybrid denoise, and the depth-2 tile
  pipeline held two tiles' weight scratch in flight). Now dequanted once per
  block into per-forward persistent buffers, filled alongside cross-K/V.
  Plus act-quant-once for the shared q/k/v A-side (hunyuan qkv_a_side
  pattern). GATED: forced-tiling (THINFER_DIT_TILE_ROWS=64) PNG BYTE-EXACT vs
  pre-change baselines, fastwan(i8) + anyflow(coopmat/f16) legs, twice
  (hoist, then +act-quant-once). MEASURED at 81f hybrid: blocks 872->845.5s
  (pass_b 220->202s, pass_a 70->61s, fill overhead +2s). SDPA now 553s = 65%
  of blocks; the ceiling verdict stands (CL/QR/WG A/Bs triangulate
  issue-bound ALU; packed-f16 math is unsafe for QK products - do not
  reopen). anyflow_e2e gained THINFER_E2E_PROMPT for quality eyeballs.
- **Cold-start note: worklog NEXT item 4 is ALREADY DONE** - residency's
  `prep_op` GPU-transcodes Bf16->Q8_0 whole-tensor at stream-in
  (`WeightPrep::Q8_0FromBf16`, visible as q8_0_from_bf16 in the rollup);
  no CPU cold transcode remains on the anyflow path.
- Phase-split fact (81f hybrid telemetry + per-pipeline gpu_ms): SDPA 541s
  (62%, at kernel ceiling), pass-B 220s (coopmat_ffn_down 73s + coopmat_proj
  50s + i8_ffn_up 33s + dequants 40s), pass-A ~70s. i8 DP4A runs 11-12
  TFLOPS-eff; coopmat 5-5.5 (see dead-end below: that IS its ceiling).
  NB the rollup's sdpa_sg line under-reports (~13s total = impossible);
  trust the W3-vs-full e2e delta (~2.6-2.8 TFLOPS-eff) for SDPA.

**HYBRID EYEBALL DONE 2026-07-03 (agent-eyeballed, 81f 832x480 2-step, same
seed/prompt A/B, drift-prone tracking-shot prompt): hybrid W=3 holds
composition/scene (no W3-all-style cuts) but shows SLOW ATTRIBUTE DRIFT over
5s (hat band appears, sunglasses appear, polka-dot scale grows, dog morphs
breed, pavement dissolves to sand). FULL ATTENTION same seed: identity SOLID
across all 81 frames. The drift is the WINDOW's, not the model's. NB the
pixel-MAD drift metric called them equal (0.124 vs 0.120) -- it is blind to
semantic attribute drift; eyeball remains the deciding gate. Tier map stands:
hybrid ~14 min = stable-scene/soft-identity, full ~17 min = solid identity.
Frames: scratch/anyflow_81f_eyeball{,_full}_png.**

LANDED 2026-07-03 sprint (uncommitted, agents + gates; all clippy-clean):
- **AnyFlow/Wan tiled**: deferred per-block setup (fill_mod+cross_kv+prepared
  fill in ONE submit_deferred scope, SDPA fence chains into pass B: 3 hard
  syncs/block removed, upper bound ~10s at 81f; the double-buffered
  overlap variant was analyzed and DROPPED: submit-interleave hazard +
  ~466MB reserve growth). **Tiled cross-attn K/V step-cache**
  (WanCrossKvCache reused; per-expert caches on MoE; byte replay after step
  0). Both scheduling/replay-only.
- **Hunyuan Q8 mod linears: WIRED BUT DISABLED (gate red, then re-greened
  dense)**: with the adaln Q8 wiring ON, dit_parity (default AND tiled) dies
  in a wgpu device panic (wgpu_core.rs:2253 binding/validation, not a band
  miss) while the tapless t2v_e2e passes -> suspect the taps/diag path or
  the adaln dequant binding. The plumbing (Site::Mod, dequant_adaln,
  reg_lin_q8) is committed but the two flip points are OFF (cfgs note in
  hunyuan_block_cfgs + registration in DitH::register; they MUST flip
  together). All hunyuan gates GREEN with it off. Fix + re-enable = next HY
  session's first item; the prize below stands:
  (worklog attack item 2): img_mod/txt_mod
  -> Q8_0 dequant-once at the adaln site (new Site::Mod; NO i8 acts, weight
  rounding only; ON under the default i8 config, THINFER_HY_I8=0 bisects).
  T2V forward streams 2.37GiB less; AR chunk forwards 1.19GiB less (~10% of
  the 13.5GB stream; ~30GiB less PCIe per 77f run). bf16 lightx2v takes the
  GPU stream-in transcode; F32 minWM stays CPU (known follow-up).
- **Wan full-VAE OOM fix**: temporal streaming already existed (diffusers
  feat_cache form, exact); the bug was the spatial tile sizer seeding at
  100% of budget with no margin/retry. Now: 0.82 seed safety + strict
  budget + balanced re-seed on OOM (LTX pattern) + config-aware calibration
  (Wan2.1 was over-reserving ~2.7x -> bigger tiles on the 169s path).
  THINFER_VAE_MEM=1 telemetry.
- **LTX** (audit agent): act-quant-once (-40% act_quant, -2112 disp/run),
  step-invariant uploads hoisted out of the denoise loop (~45MB/step PCIe),
  sst_out host-cached (2 blocking readbacks/step gone). Confirmed NO
  per-tile weight re-prep bug in LTX; cross-attn K/V legitimately per-step
  (sigma-modulated). Remaining LTX levers (NOT taken, need gates/sign-off):
  dense matmul_qkv sites = 54.9s of ~105s DiT GPU; prompt-encode phase 61s
  of 178s wall (streaming-bound, per-layer prefetch already optimal); VAE
  OOM-retile heuristic.
- **Wan2.2 MoE audit** (no code): LoRA fold = lazy per-tensor, compute-once
  cache, NOT repeated across steps (verified vs step1-step2 delta = 62.5s
  cold fold per expert); fold cache = ~15GB RAM per expert UNACCOUNTED
  (~30GB both, held to request end). Designed-not-implemented: CPU-side
  fold-warm of the low expert during high steps (~60s off step 3) +
  fold-cache clear at the expert switch (frees ~15GB mid-request).

NEXT (in order):
1. USER: pick the anyflow default story given the eyeball verdict above
   (hybrid default is quality-honest for scenes without a tracked subject;
   full attention is the identity tier). Then the 4-step full-attention
   cats/dog composition A/B (decides the quality-ceiling tier).
2. If the anyflow hybrid eyeball is good, run the same A/B for Wan2.2's
   default W=3 (its all-steps window has the same horizon mechanism).
3. Artifact-retention TTL option (delete-on-fetch cost a clip to a closed
   tab 2026-07-03): offered to user, no decision yet.
4. GPU bf16->Q8 transcode prep kernel exists already (`Q8_0FromBf16`); NB
   measured: umT5 + cold transcode + first-block prep = ~12 min wall in
   the e2e harness (serve first-touch overlap is much cheaper, ~26s
   step-1-vs-2 delta) -- localize only if cold-start matters.
5. Consider default vae=tiny for anyflow in web UI (CLI already defaults
   Tiny) -- 81f tiny eyeball GREEN engine-side (red-car frames + user clip
   decode); full stays the parity path.

**Pin-plan lesson (3 device losses 2026-07-02, reverted): do NOT pin a
weight slice when the working set >> VRAM budget.** Pinning starves the
same-size recycle scan -> arbiter reclaim during in-flight compute ->
deferred-destroy zombies -> physical VRAM overshoots the accounted budget ->
device lost (at 4.2G/3.2G/2.1G pins alike, 8GB card). The fully-streamed
steady state (same-size recycle, zero frees) is the stable regime. Z-Image
keeps its pin plan (whole DiT fits). See the NO-pin comment in
`wan/pipeline.rs::denoise_with`.

**Quality note (2-step, hard prompt):** rendering superb, COMPOSITION wrong:
two cats, pianist cat has human hands + sneakered feet, dog limbs smear in
motion (`scratch/browser-run/video.mp4`). Decisive A/B queued: SAME seed +
prompt at 4 steps -- if composition heals, steps is the user's quality dial
(already exposed); if not, it's the model's multi-subject ceiling. Rides the
post-deploy user browser run.

## HY15 TI2V: PARKED (user decision made 2026-07-02)

User picked ANCHOR SYNTHESIS for the TI2V fast-T2V path: text request ->
in-engine t2i makes frame 0 (Z-Image turbo; derive image prompt from the
motion prompt, rewriter machinery is the natural place) -> run the model in
its trained I2V regime. RENAME the model id honestly (it is I2V, optionally
fronted by t2i -- user: "don't call it ti2v"). Prereq: one I2V 77f pixel
validation (latent trend says mild decelerating drift, needs eyeball).
Do AFTER the AnyFlow perf push. TI2V perf items (Q8 mod linears, prefix
pinning) fold into the same engine work above. Drift root-cause record is in
the section below; do not reopen it.

## AnyFlow-Wan2.1-T2V-14B: SHIPPED reference (ported 2026-07-02)

Any-step flow-map distill; steps USER-CONFIGURABLE (CLI+serve+web, done).
LOW-RAM pyref policy honored (component pyref; block-streaming full-DiT pyref
is a follow-up).
FACTS ESTABLISHED (from third-party/AnyFlow + diffusers main
transformer_anyflow.py + HF configs, all verified):
- Arch = stock Wan2.1-14B (= `WanDitConfig::wan22_14b()` exactly; Wan2.1 VAE,
  umT5-xxl, diffusers tensor names -> NO rename layer needed). 1099 tensors.
- ONLY delta: `condition_embedder.delta_embedder.linear_{1,2}` (a second
  TimestepEmbedding fed sincos(r)); rt_emb = 0.75*temb(t) + 0.25*delta(r)
  (gate_value=0.25 config const, deltatime_type="r"); timestep_proj =
  time_proj(silu(rt_emb)); rt_emb also feeds final-layer modulation. Blend =
  BcastMul x2 + add (rows=1).
- Scheduler: linspace(1,0,steps+1), sigma-shift 5 (same formula as ours),
  x -= (sigma_t - sigma_r)*v == our standard Euler; ONLY new input: r =
  next sigma*1000 passed to the DiT per forward. guidance 1 (no CFG). Demo:
  480x832x81f, 4 steps, fps 16, seed via torch Generator.
- Weights: transformer/ 3 bf16 shards IN HF CACHE (27GB). umT5+VAE reuse
  the existing FastWan bundle roles (Wan2.1 VAE in/out=16).
- License NSCLv1 (noncommercial) - user informed 2026-07-02.
IMPLEMENTED (2026-07-02, all compiling clean, uncommitted): WanDitConfig
`delta_embedder` flag + `anyflow_t2v_14b()`; ConditionEmbedder optional
delta MLP + gated blend (GATE_VALUE 0.25, `gated_blend` via BcastMul/Add);
WanDitInputs.r_timestep (None on all legacy paths); AnyFlowSampler
(scheduler.rs, unit-tested incl 2-step grid [1000, 833.3]); pipeline AnyFlow
arm gated on cfg.delta_embedder (steps from params.steps, taps for parity);
WanVariant::anyflow_t2v_14b (Q8_0 block_transcode, bf16 acts, Wan2.1 VAE);
manifest roles DIT_ANYFLOW_1..3 + variant "anyflow-t2v-14b"; VideoModelId::
AnyflowT2v14b + executor arm + defaults (832x480, 16fps, wan22 cell cap);
web UI entry (steps visible, uncapped; vae pinned full). `--steps` already
existed CLI+wire (UniPC's), now feeds AnyFlow too. Conformance:
`tests/wan/anyflow_e2e.rs` (embedder parity vs
`gen_anyflow_embedder_ref.py` fp32 pyref -- low-RAM: reads only
condition_embedder tensors -- + engine health at 256x256x9f 2-step + PNGs).
VALIDATED 2026-07-02: `anyflow_e2e` GREEN (embedder parity slope 1.0008
rel_rmse 2.3e-4 vs fp32 pyref; 256x256x9f 2-step health 70s incl transcode).
832x480x33f 2-step eyeball: GORGEOUS (sharp coherent prompt-faithful car
clip; denoise 375s incl cold bf16->Q8 transcode, VAE 67s;
`scratch/anyflow_480p_png/`). One maiden-run bug found+fixed engine-wide:
dtype probe read the SOURCE encoding while registration transcoded blocks ->
Q8/bf16 pipeline mismatch -> NaN; `load_variant` now derives dit_w from
`variant.block_transcode` first. Serve REDEPLOYED with AnyFlow (steps field
live in web UI). FastWan video_e2e full-parity regression GREEN 2026-07-02
("parity OK vs pyref", all bands) -- legacy Wan models proven untouched.
Follow-ups: block-streaming full-DiT pyref; 81f long-clip quality verdict
(user run in flight at /clear time).

## HY15 TI2V drift: ROOT-CAUSED + CLOSED (2026-07-02, reference record)

Controls at 832x480 (all same seed/prompt, 45f/3-chunk latent-trend harness,
`THINFER_E2E_SKIP_DECODE=1 THINFER_AR_DIAG=1`):
- text-only shipping: latent std/chunk 1.256 -> 1.413 -> 1.580 (+12%/chunk)
- text-only PURE BF16: 1.259 -> 1.417 -> 1.592 (IDENTICAL -> quant innocent;
  f16 SDPA also exonerated by kernel read: all-f32 accumulators)
- I2V (frame-0 anchor): 1.145 -> 1.259 -> 1.335 (+10%, +6%, DECELERATING ->
  anchor damps; commit-KV telemetry: cache K flat, cache V inflates at deep
  blocks, block53 v_std +13% by chunk 2)
Mechanism: unanchored text-only AR rollout at product dims (OOD mode for the
i2v-trained dmd ckpt; 33 txt tokens vs 6240 img tokens/chunk can't anchor).
448x256 is clean (img:txt ratio 3.5x smaller). **Recache-T mitigation DEAD
(2026-07-02): `THINFER_HY_RECACHE_T` (env, plumbed in ar.rs) at 250 gives
1.479/1.653, at 15 gives 1.441/1.620 -- BOTH WORSE than baseline 1.413/1.580.
Upstream's stabilization knob amplifies, never damps, this drift.** No engine
bug: quant/f16-SDPA/faithfulness all exonerated. Remaining product options
(USER DECIDES at wrap-up): (a) ship with guidance (text-only good to ~45f at
480p, I2V for longer), (b) validate I2V 77f pixels (trend says mild) and make
I2V the long-clip recommendation, (c) anchor-synthesis chain (image model
makes frame 0 -> I2V) as a new feature.
Drift metric harness: `scratch/drift_stats.py` (pixels) + in-log per-chunk
latent/commit-KV stats. Perf note: img-only acquire measured ~2.5x on chunk
denoise (23min/5chunks@480p). OOM guard: 6GB VRAM budget OOMs at chunk-5 KV
alloc at 77f/480p; 5GB works.

**>>> BUG FIRST (before perf): progressive AR drift.** User's 77f (5-chunk)
browser run: chunk 1 GORGEOUS (dancing dog, better than the old T2V eyeballs),
then saturation/contrast blows up chunk-over-chunk into chaos by chunks 4-5
(`scratch/crazy_frames/`). **The quant-noise theory is DEAD: the SHIPPING
config (i8+coopmat+f16 SDPA) at 448x256x77f T2V-probe is CLEAN and
statistically identical to the pure-bf16 run (contrast 0.13->0.23, clip
1.5%->14% in BOTH; `scratch/drift_ship_png/` vs `drift_bf16_png/`,
`scratch/drift_stats.py` is the metric). The earlier bisect lacked a positive
control; everything is clean at tiny dims.** Drift is REGIME-dependent:
resolution/KV-length (tokens/frame 390 -> 1560, s_k to ~32k rows where f16
SDPA + bf16 KV cache errors have 4x the accumulation length) and/or
content/prompt. IN FLIGHT: shipping-config repro at 832x480x77f T2V-probe
(duck prompt). If it reproduces -> bisect AT PRODUCT DIMS (i8-only vs
coopmat-only vs no-fast-sdpa). If clean -> content/prompt-dependent; add a
prompt override to the health test and retry with a motion-heavy prompt.
Mild clip%/contrast growth (to ~14% by chunk 5) exists even in pure bf16 and
is visually harmless; the discriminator is growth RATE (crazy frames hit 62%
clip). Upstream `stabilization_level` is 1 in every minWM config (= recache
t=0, what we do); >1 is a mitigation lever, not a faithfulness gap.

**It is currently WEIGHT-STREAMING BOUND, not compute bound: ~11min for 29f
(2 chunks) at 480p. Each of the `chunks*(4+1)` AR forwards re-streams all 54
blocks (~13.5GB mixed Q8/bf16) for only 6,240 rows of compute; the T2V's single
32,760-row forwards hide the same stream, chunk forwards can't. Attack in this
order:**

1. **DONE (uncommitted): skip txt-side weights in chunk forwards**
   (`acquire_img_block` in ar.rs; numerics-neutral). Perf A/B rides the
   in-flight 480p run vs the old 694s/2-chunk measurement.
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

- **Coopmat shared-memory staging is a DEAD END (measured 2026-07-03):** a
  multi-subgroup staged GEMM (ns 2x2..4x2, bk 16-64, llama.cpp shape) ran
  3.2-5.0 TFLOPS vs the direct kernel's 6.3-7.2 at the production site shapes;
  coopLoad from workgroup memory is strictly slower than L2-served global
  coopLoad on naga 29 + NVIDIA. Register-tile sweeps are flat (1x1 ~= 2x2 ~=
  4x4) -> the direct kernel is coopLoad-issue-bound, not reuse-bound; ~6-7
  TFLOPS is naga-coopmat's practical ceiling on this card (i8 DP4A does 11-12).
  Code reverted; only this note remains.

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
- **LongLive-2.0-5B** (AR long/multi-shot) -- shipped. AR perf LANDED
  2026-07-03 (uncommitted): per-request cross-attn text K/V cache
  (`WanCrossKvCache`, byte-replay = bit-identical) + prefix segments written
  directly into the window buffers (kills ~39GB/chunk host gather + GPU-GPU
  copies; upload-once-per-chunk is INFEASIBLE: all-layer prefix K/V = 7.8GB >
  card). Gates: longlive_e2e + multishot GREEN at THINFER_LL_BUDGET_GB=6 (the
  8GB default OOMs on the 8GB card with today's desktop VRAM overhead).
  **DISCOVERED: longlive_parity vel_c0s0 FAILS PRE-EXISTING at HEAD
  (slope 0.9214 vs band 0.030, byte-identical with and without today's
  changes, so today's work is exonerated). The gate is stale vs weeks of
  kernel changes (f16 subgroup SDPA etc.); root-cause = bug-first item for
  the next LongLive session.** OPEN: re-measure WARM at 576.
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
