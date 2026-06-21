# Worklog

Forward-looking only (git history is the changelog). Engine-wide design +
kernel/runtime state: `plan-details.md`. Per-model porting: `wan-plan.md`
(Wan2.2-TI2V-5B line; Wan-backbone / RoPE3D / umT5 / VAE / GGUF lore is
reusable, SkyReels-DF sections obsolete). Z-Image (shipped): `zimage-plan.md`.

## Status

- **FastWan2.2-TI2V-5B-FullAttn SHIPPED.** Parity GREEN at 256x256x5/6GB; coherent
  NaN-free 576x576x97 clips; fits a 2GB budget. bf16 full precision (no quant in
  path). DiT bf16 acts, umT5 bf16 acts. This is the baseline; DO NOT DISTURB it.
- **Active work = LongLive-2.0-5B** (next section). Same base, adds AR/causal
  streaming for long + multi-shot video. AR ENGINE PATH LANDED + e2e health-GREEN
  on the real weights. PYREF PARITY RESOLVED (2026-06-21): engine proven faithful
  (per-forward DiT in band, AR path bit-identical to GREEN FastWan); the AR-depth
  compounding vs the bf16-locked pyref is TOLERATED via a two-tier band (tight
  per-forward `vel_c0s0`, gross floor on compounded tensors). Video CLI-confirmed
  good (512x512x349, 11 chunks, faces stable). MULTI-SHOT LANDED + health-GREEN
  (scene cuts via per-shot prompts + scene_cut_prefix + RoPE-phase advance + chunk
  pin). SELF-ATTN QKV i8 LANDED (qkv site-split: self-attn DP4A, cross-attn bf16;
  parity GREEN, self-attn qkv matmul 6.1x). Remaining = larger-res/warm perf
  (streaming-overlap levers) + multi-shot pyref byte-parity (only health-tested).

## NEXT (active): LongLive-2.0-5B (AR/causal long video)

Direction LOCKED with user (2026-06-20). Same Wan2.2-TI2V-5B base as FastWan; AR/
causal streaming = long + multi-shot video. The "port the backbone once" payoff.

CURRENT STATE: full AR path LANDED + e2e health-GREEN on the real weights; FastWan
byte-untouched. The pieces (read the code, not a re-spec):
- Ingestion: `.pt` WeightSource (`thinfer-core/.../format/pytorch.rs`); `LongLiveSource`
  + `open_longlive_source` + `longlive_dit_renames` (`wan/source.rs`, .pt renamed to
  canonical, unioned over base umT5+VAE safetensors); `longlive-2.0-5b` manifest variant
  (`dit_pt_role`); `WanDitConfig::longlive_2_0_5b()` (== fastwan geometry).
- Sampler: `wan/unipc.rs` `FlowUniPc` (4-step bh2 predict_x0, order [1,2,2,1], pivoted
  2x2 at the sigma=1 edge). Per chunk: 4 steps then a timestep-0 clean recache forward
  (output discarded; only updates the KV cache).
- KV cache: `wan/kv_cache.rs` `KvWindowCache` (`begin_chunk` rolls/evicts once + returns
  the `ChunkPlan{prefix segs, tail, chunk_start_frame, temporal_offset}`; `commit_chunk`
  writes the clean pass; pin/zero/advance_shot for multi-shot). Host-resident
  (`RamKvStore`/`KvStore` offload seam). `longlive_runtime` sizes `frame_seq_len=pph*ppw`.
- GPU AR forward: `WanDitBlock::self_attn_ar` (windowed self-attn: rope q+chunk-k at the
  absolute frame, window = [host prefix ++ chunk], `op_sdpa` mode 0, exports roped-k/v
  for commit); `WanDit::forward_ar` (per-chunk forward, per-layer prefix upload, serial
  residency); `WanModel::denoise_ar`/`generate_ar` (the chunk loop; no separate model
  type -- `WanModel` loads from `LongLiveSource` directly). cross-attn+FFN tail factored
  to `WanDitBlock::cross_ffn` (shared with `forward`, byte-identical).
- CLI: `--model longlive-2.0-5b`. `--frames` must give f_lat % 8 == 0 (29->1 chunk,
  61->2, 125->4). e2e: `longlive_e2e` (conformance, wan-e2e), GREEN at 128x128x61/6GB.
  Run: `THINFER_LONGLIVE_PT=<pt> THINFER_POWER_PREF=high THINFER_LL_BUDGET_GB=6 cargo
  test -p thinfer-conformance --features wan-e2e --release longlive_e2e -- --nocapture
  --test-threads=1`.

REMAINING:
1. PYREF BYTE-COMPARE PARITY -- RESOLVED 2026-06-21 (GREEN with two-tier bands; user
   approved tolerating the AR-depth precision compounding). The engine is proven faithful;
   the failure was AR-depth f16-vs-bf16 compounding vs a bf16-LOCKED reference, NOT a code
   bug. Full diagnosis kept below as the record (do NOT re-open the op-hunt). Harness +
   diagnostic scaffolding retained (gated/opt-in) for the perf + multi-shot work.

   What's built (all UNCOMMITTED, working tree):
   - Pyref `gen_longlive_video_e2e_ref.py` (thinfer-conformance/python/.../wan/). CPU/bf16
     (venv is torch-CPU; fp32 would blow host RAM on the 10GB DiT). Drives the AUTHORITATIVE
     upstream `CausalWanModel` on the real `.pt` (meta-init + assign-load; ckpt["generator"],
     strip "model." -> 825 params exact). Reimplements ONLY the single-prompt T2V AR chunk
     loop (UniPC via upstream FlowUniPCMultistepScheduler, windowed KV cache, clean recache);
     cache/window/RoPE all live in upstream code = truth. umT5+VAE from the shared FastWan
     diffusers base (engine reuses same base for LongLive). Shims (clone is read-only): inject
     `x_clip_loss` into transformers (dropped upstream); replace `flash_attention` with an SDPA
     equivalent (CUDA-only assert otherwise) -- self-attn already uses the SDPA-fallback
     `attention()`; LLV2_TRITON_{ROPE,ADALN}=0 + HF_HUB_OFFLINE. Loads umT5/VAE from
     THINFER_FASTWAN_DIR (Rust sets it = base snapshot dir). Added deps: einops, scipy (uv).
   - Rust test `longlive_parity.rs` (tests/wan/, registered in main.rs). Pins CTHW noise
     (same SplitMix64 as video_e2e), runs pyref via `uv`, byte-compares with slope+rel_rmse
     bands. Pyref noise reshape/permute: CTHW [C,F,h,w] -> [B,C,F,h,w] -> permute [B,F,C,h,w]
     (upstream pipeline is frame-major).
   - Engine diag hooks in `denoise_ar` (pipeline.rs) + `forward_ar` (dit.rs), all opt-in:
     `chunk_diag` (per-chunk post), `vel_diag` (per-step raw velocity), `block_res_diag`
     (per-block residual, chunk0/step0 only). generate_ar passes None. THESE TAPS ARE
     DIAGNOSTIC SCAFFOLDING -- strip (or gate cleanly) before commit.
   - Test env hooks in `longlive_parity.rs` (all uncommitted scaffolding): THINFER_LL_SELFCHECK
     (forward_ar vs forward_velocity_at), THINFER_LL_ACT=fp32|f16|bf16 (force DiT block act
     dtype; default = device f16), THINFER_LL_DUMP_BLOCKRES=<dir> (dump 30 chunk0/step0
     block_res .bin), THINFER_LL_DUMP_SUBOPS=<dir> (call `diag_step_at` on chunk0 -> dump
     front-end taps + 15 block-0 sub-ops + 30 per-block + dit_out).
   - Python A/B tooling in scratch/longlive/ (run via `uv run` from
     thinfer-conformance/python so the venv resolves): `ab_diffusers_upstream.py`
     (diffusers vs upstream, the fact (A) prover -- has `make_pinned_noise` port + the
     longlive->diffusers rename port), `isolate_block.py` (per-block isolation),
     `subop_compare.py` (block-0 sub-op A/B vs upstream -- generalize its block index next),
     `intermediate_mags.py`, `bias_structure.py`, `precision_probe.py` (early dead-end).
   - Run engine parity: `THINFER_LONGLIVE_PT=<pt> THINFER_POWER_PREF=high THINFER_LL_BUDGET_GB=6
     THINFER_LL_WIDTH=256 THINFER_LL_HEIGHT=256 cargo test -p thinfer-conformance --features
     wan-e2e --release longlive_parity -- --nocapture --test-threads=1`. USE 256x256, NOT the
     128 default (128 is pathological per fact (E); 256 = FastWan's GREEN res). Pyref AR loop
     at 256 ~5min CPU; total run ~280s. Logs scratch/logs/ll_*.log; scratch probe
     scratch/longlive/.

   DIAGNOSIS (RE-CORRECTED 2026-06-21, telemetry-driven against the AUTHORITATIVE pyref
   dumps; the prior "find the one op that under-produces outliers at block 17-18" diagnosis
   was ALSO WRONG -- there is NO single buggy op, do not revive the op-hunt). The earlier
   scratch A/B scripts (`chain_compare.py` first pass, `ts_probe.py`) gave UNRELIABLE upstream
   numbers because a hand-built `model(...)` call is fragile (windowed self-attn reads
   `_CURRENT_GRID_META`; mis-set -> garbage, e.g. slope 0.03). The RELIABLE method: compare
   against the pyref's OWN persisted dumps in `target/tmp/wan_longlive_parity/` (py_c0_s0_vel,
   py_c0s0_block{i}); a bf16 replica that mirrors the pyref `run_model` EXACTLY (latents
   permute, context=pe, seq_len=28160, sched.timesteps[0]) is bit-exact to the persisted
   pyref (`fp32_ref.py` sanity slope 1.00000) -- use that, not naive model calls.

   Established facts (authoritative):
   - AR PATH IS NOT THE BUG. THINFER_LL_SELFCHECK at 256: forward_ar chunk0 vel == FastWan
     full-attn forward, BIT-EXACT (rmse 0). diagVSar per-block all 1.00000. So denoise_ar /
     windowed self_attn_ar / KV cache are clean; the gap is in the shared DiT forward.
   - PER-OP / PER-BLOCK-GIVEN-SAME-INPUT IS FAITHFUL. `subop_compare.py` (generalized to any
     THINFER_LL_TAP_BLOCK; engine perblock{N-1} -> upstream block N): every sub-op at blocks
     0,8,16,17,18,20,24,28 matches dev <0.004, rel <1.5e-3. No op under-produces.
   - THE ENGINE CHAIN DIVERGES FROM THE PYREF CHAIN GRADUALLY, ONSET ~block 15 where outlier
     channels SPIKE. Engine perblock{i} vs authoritative py_c0s0_block{i}: slope ~1.0 for
     blocks 0-14, then 0.987(b15) 0.980(b17) 0.971(b18) 0.96-0.97(b20-24) 0.95(b25-28). The
     residual outlier max SPIKES blocks 20-24 (438/648/712/548/324) then COLLAPSES at b25 (76)
     -- catastrophic cancellation of large outlier channels AMPLIFIES the 16-bit rounding
     gap (b25 76 vs 68.6 = 9.7% under). The engine slightly UNDER-produces the spiked outliers.
   - IT IS A TWO-16-BIT-PATHS PRECISION DIVERGENCE, NOT A LOGIC BUG. vel_c0s0 vs pyref scales
     MONOTONICALLY with engine act precision then PLATEAUS: f16 0.97564, bf16 0.98271, fp32
     0.98444 (THINFER_LL_ACT sweep, diag-only). A logic bug would not improve with precision;
     this does, then floors at ~0.984 because the REFERENCE is bf16. The pyref is bf16-LOCKED:
     CPU venv bf16 weights AND `wan_5b/modules/attention.py::attention()` HARDCODES bf16 SDPA
     compute (dtype=torch.bfloat16, line 272) regardless of input dtype -- so self-attn in the
     reference is bf16 by design; a true fp32 reference would require rewriting upstream attn.
   - bf16 ENGINE ACTS DO NOT FIX THE FULL RUN. Full parity at THINFER_LL_ACT=bf16: vel_c0s0
     0.983 (now in band) but vel_c0s1 0.956 (FAILS), compounding to chunk1_post 0.732,
     vae_rgb 0.755. Per-step dev grows SUPER-linearly within chunk0 (0.017->0.044->0.091->
     0.110) = flow-sampling amplification of the per-forward 16-bit gap, not a sampler bug.
   - SAMPLER SCHEDULE IS FAITHFUL. Engine FlowUniPc timesteps [1000,937,833,625] vs upstream
     [999,937,833,624]; steps 1,2 exact, only a 1-unit rounding at the sigma=1 edge (engine
     sigma[0]=1.0 vs upstream 0.99980) and step3. 0.1% effect, not the cause.
   => ROOT: the engine (any 16-bit precision) vs the bf16-locked CPU pyref diverge ~1.6-2.4%
   per forward in the large-outlier residual channels (blocks 15-25, magnitudes 200-700 with a
   cancellation spike), and this COMPOUNDS across 30 blocks x 8 forwards (4 UniPC steps x 2
   chunks) to fail the bands. The engine is arithmetically faithful per-op and bit-clean on the
   AR path; the per-forward gap (vel_c0s0 dev 0.024) is ITSELF within the slope band -- only the
   AR-depth compounding breaks it. FastWan (3 forwards) tolerates the same engine; LongLive
   (8 forwards) does not. This is NOT a fixable single-op/code bug.

   DECISION (made with user 2026-06-21): option (a) -- TOLERATE. Engine stays f16 (perf;
   proven faithful). `longlive_parity.rs` now uses a TWO-TIER band: `vel_c0s0` (single
   forward) holds the TIGHT `TOL_LATENT` (the real per-forward DiT regression catcher);
   the AR-compounded tensors (vel_c0s1+, chunk{c}_post, pre_vae_latent, vae_rgb) use a loose
   GROSS-regression floor `TOL_AR`/`TOL_AR_VAE` (catches sign/scale/structural breaks, not
   byte-parity). Test GREEN at 256. Video CLI-confirmed good. fp32 reference (option b) was
   REJECTED by the user (never want fp32); revisit only if a regression makes the gross floor
   trip. bf16 acts (option c) shown insufficient on its own + costs perf -- not taken.

   Diagnostic scaffolding added this session (UNCOMMITTED, strip/gate before commit):
   `WanDitTaps.tap_block` + `diag_step_at(tap_block)` (pipeline.rs/dit.rs) generalize the
   block-0 sub-op taps to any block; `THINFER_LL_TAP_BLOCKS` (comma list) + `THINFER_LL_DIAG_ONLY`
   in longlive_parity.rs sweep N blocks in one model load and skip denoise_ar. scratch/longlive/
   `chain_compare.py` (engine perblock vs pyref per-block, authoritative), `fp32_ref.py` (the
   bit-exact pyref replica + precision sweep), `ts_probe.py` (timestep probe; note its naive
   model call is unreliable -- kept only as a caution). `cargo fmt`+`clippy` clean.
2. AR PERF. MEASURED 576x576x125 (4 chunks, tiny VAE, 5G, TRACE): 390s wall (DiT
   denoise 376s, tiny VAE 6.4s). COMPUTE-BOUND at 576: GPU sum ~319s vs DiT wall 376s
   => ~57s (~15%) idle (serial-residency weight streaming + submit + host gather/
   readback). Per-op totals over the 20 forwards: ffn_down 80.8s / ffn_up 80.8s /
   qkv 67.5s / sdpa_sg 48.5s / proj 32.8s (matmul ceiling, same as FastWan). workspace
   peaked 4.98GiB at the 5G budget (untiled AR self-attn). Wall is HIGHER than FastWan
   for similar length (FastWan 576x576x97 = 179s) because LongLive runs 20 forwards
   (4 chunks x [4 steps + 1 clean recache]) vs FastWan's 3 -- each re-pays per-block
   weight streaming + cross-attn over the full 512 text rows; the O(N)-streaming win is
   LENGTH/VRAM-bound generation, not shorter wall at a fixed length.
   STATUS: TWO compute wins landed -- i8 DP4A ffn_up AND now self-attn qkv (the
   qkv site-split, see the i8 SHIPPED section below). Measured 256x256x61 CLI
   (tiny VAE, TRACE): self-attn qkv matmul 4237->699ms (6.1x; +430ms dequant net
   ~3.75x), cross-attn qkv UNCHANGED at ~4.9s (stays bf16), wall 63.8->53.8s (-16%).
   The streaming-overlap wins (a/b/d) target idle that is mostly a COLD-START disk
   artifact: warm (page-cache hot) the AR forward at 576 is ~86% compute-bound (GPU
   158s vs denoise wall 183s, ~14% idle that is NOT block-weight streaming).
   Re-measure warm at 576 before chasing any of these.
   WINS, in value order (all quality-neutral, exact):
   (a) PREFETCH OVERLAP in `forward_ar`: mirror `forward`'s `join!(submit,
       next_acquire, prefetch_after)`. TRIED + REVERTED: ~1% warm (noise); the "15%
       idle" was cold-disk, not recoverable per-block streaming. Do NOT re-add without
       a warm before/after showing real gain.
   (b) Upload the window prefix ONCE per chunk, not per forward: it is identical across
       a chunk's 5 forwards (host transfer, not in gpu_ms, but cuts HtoD traffic).
   (c) Activation-tile the AR self-attn (workspace ~= budget at 576): cuts the workspace
       peak, buys prefetch/res headroom; mirror `forward_block_tiled`.
   (d) Cross-attn text K/V cache across a chunk's 5 forwards (same prompt -> identical);
       upstream bypasses it for cudagraphs, but we have no cudagraph, so it is a free win.
   (e) Skip the head (proj_out + unpatchify) on the clean recache pass (velocity
       discarded); small (proj_out is 192 cols).
3. MULTI-SHOT / SCENE-CUT -- LANDED + health-GREEN (2026-06-21). CLI repeats
   `--prompt` (one per shot); `--frames`/`--duration` is one value (split evenly in
   chunk units, mirrors upstream `_even_durations`) or one-per-prompt, else error;
   single prompt -> empty shots (the parity path, byte-unchanged). Engine: `Shot{prompt,
   chunks}` + `shots: &[Shot]` on `generate_ar`/`denoise_ar`; `denoise_ar` builds the
   per-chunk block-prompt list (first chunk of shots>0 gets `SCENE_CUT_PREFIX`), encodes
   each unique prompt once, and at each boundary chunk calls `cache.advance_shot()`
   before `begin_chunk` (RoPE phase, already plumbed to `forward_ar`) + `pin_current_chunk`
   after commit. Release `shot_clean_recache=False`, so NO cache zero on a cut (only
   advance + pin); the `zero_for_scene_cut` primitive stays for if it is ever enabled.
   Tests: `longlive_multishot_e2e` (real weights, 2 shots, NaN-free + variance, PASS 56s);
   `kv_cache::pinned_chunk_survives_eviction` unit test; CLI shot-resolution unit tests.
   OPEN: only health-tested -- a multi-shot pyref byte-parity (extend
   `gen_longlive_video_e2e_ref.py` to the multi-prompt block list) is the durable proof
   if/when multi-shot quality must be guaranteed. `zero_for_scene_cut` path unexercised.

KEY GOTCHA for the AR loop: within a chunk, `current_start` is CONSTANT across all 4
UniPC steps + the clean pass; each forward recomputes the chunk's K/V at the same tail
slot. Only the timestep-0 clean pass's K/V are committed (survive into future chunks).
Convert frames<->tokens with `frame_seq_len = pph*ppw` (= 880 at release res, runtime
elsewhere) everywhere.

CONTEXT (verified against the clone, do not relitigate):
- ROPE: release runs `use_relative_rope=False`, `t_scale=1.0`, `rope_method="linear"`,
  `original_seq_len=None`. ABSOLUTE-position temporal RoPE: q and chunk-k rotate at
  `chunk_start_frame = current_start/frame_seq_len`; cached prefix k stored already-roped
  at its own absolute position (no re-rotation at attention time). `temporal_offset =
  shot_index * 8` is INTEGER, folds into the frame id. `rope3d.rs` `lookup_temporal`.
- For single-prompt T2V the per-frame modulation `e[B,F,6,C]` collapses to FastWan's
  scalar-t `[6,inner]` broadcast (all chunk frames share one timestep), so forward_ar
  reuses the FastWan condition embedder + modulation unchanged; ONLY self-attn differs.
- inference.yaml: chunk 8 / window 32 / sink 8 / multi_shot_sink true (global_sink 8) /
  rope_offset 8 / 4 steps / shift 5.0; shape [1,128,48,44,80] -> frame_seq 880.

GROUND TRUTH IS THE CLONE for the parity work -- re-verify mechanics, don't trust
notes. Upstream `third-party/LongLive` (NVlabs/LongLive, depth-1, sibling of
`thinfer/`): `pipeline/causal_diffusion_inference.py` (the AR loop = the real spec),
`wan_5b/modules/causal_model.py` (self-attn + cache updates), `wan_5b/utils/
fm_solvers_unipc.py` (the sampler), `configs/inference.yaml`. IGNORE the nvfp4 branch.
Weights: HF `Efficient-Large-Model/LongLive-2.0-5B` = `model_bf16.pt` (10GB, 825 bf16
tensors, the COMPLETE merged DiT -- LoRA pre-folded, no separate lora_ckpt, no
safetensors mirror); base Wan2.2-TI2V-5B (umT5+VAE) reused from FastWan.

DECISIONS LOCKED (do not relitigate):
- bf16 full precision, NO quant (NVFP4 upstream variant SKIPPED). Must not regress vs
  FastWan (same 5B/10GB tier -- user's explicit ask).
- RUNTIME `.pt` ingestion, no build step / no committed artifact / no on-disk dup
  (footprint first-class). Reading the real `.pt` keeps parity "same weights" honest.
- DO NOT DISTURB FastWan (GREEN). New attention behavior lives in the AR path only;
  FastWan's full-attention `forward` stays byte-unchanged.
- Track here (no longlive-plan.md).

## SHIPPED: Tiny VAE (LightTAE) -- CLI default

ONE NAME: "tiny" (`VaeChoice::Tiny`, `--vae tiny`, `role::TINY_VAE`). `wan/
vae_tiny.rs` LightTAE (`lighttaew2_2`) is the `--vae` default; `--vae full` is the
parity path. Spec: `scratch/notes/lighttae_spec.md`. Decode tiling LANDED:
temporal-chunk over latent frames sized from VRAM budget; `memcat` carries each
chunk's trailing frame into the next via ping-pong buffer (causal depth 1, no
halo/seam); a clip that fits = one chunk, bit-identical to untiled. Holds 2GB at
576x576x97 (was 4.98GiB untiled). Durable test: `THINFER_E2E_TINY=1` decodes
single- vs multi-chunk, asserts bit-identical + NaN-free + clamp + TRUE_PEAK.
Open follow-ups (low priority): conv tile untuned; `TINY_PEAK_K` anchored to one
point (recalibrate if graph changes); no true Full-vs-Tiny same-latent A/B; no
decode-vs-TAEHV-pyref parity (add when wiring TAEHV pyref).

## SHIPPED: VAE decode VRAM tiling

Decode live set = `FIXED(tout) + area*PER_AREA(tout)` per spatial tile; tile sized
from `budget - reserve` (reserve = real VAE weight footprint + staging, queried
not fractional); `set_transient_reserve` backstop caps weights, degrades to paging
if estimate is off. Constants in `vae_tile_dims` from 4 measured points
(recalibrate if decoder graph changes). DiT weights freed before VAE
(`evict_all_and_free`). At 2GB/576 tile floors at 8 (~87s vs ~55s untiled);
6GB/256 is single-tile/bit-identical. `THINFER_VAE_MEM` gates a per-tile probe.

## SHIPPED: VAE decode perf -- conv-GPU bound (NOT submit-bound)

CORRECTION (2026-06-20): prior "submit/sync bound" read was WRONG. Timestamp
totals at 576x576x49/5GB: VAE wall ~95% pure conv GPU time, ~zero recoverable
idle. Authoritative metric: `gpu_disp_ms` (timestamp) for `vae_decode` vs `busy_ms`,
NOT nvidia-smi util. LANDED: conv tiles tuned (`wan/vae.rs`
`WAN_VAE_CONV3D_TILE`/`CONV2D_TILE` = 128x96x16, tm8 tn6) -- implicit-GEMM convs
are bandwidth-bound, `bm=128` halves weight-side reads; 48 acc / 256 threads is
the occupancy knee. Bit-EXACT (f32 accum, ascending-k). 576x576x49: conv3d
134->95s, conv2d 18.5->12.7s, wall 161->117s (-28%). Sweep: `scratch/sweep_conv.sh`.
NEXT levers if needed (all still conv-GPU bound): budget-aware larger tiles (fewer
halos), packed/vec4 x-gathers (keep f32 accum order). DO NOT retry conv3d index-
math: im2col loop-invariant-div hoist TRIED + REVERTED 2026-06-19 (262->301s).

## SHIPPED: i8 DP4A matmul -- ffn_up + self-attn qkv (opt-in `--i8-matmul`)

CLI `--i8-matmul` (off by default; threaded `WanModel::load(.., i8_matmul)` ->
`WanI8Sites` in `wan/loader.rs`): transcodes the ffn_up AND self-attn qkv weights
bf16->Q8_0 at load and routes those sites through the i8xi8 `matmul_i8` (`dot4I8Packed`)
DP4A path. ffn_up op 42.3->6.9s (~6x: 4x ALU + better occupancy than the latency-bound
bf16 kernel), denoise -20% / total wall -19% at 576x576x61. Pure COMPUTE win (not byte
movement: GPU is 86% busy warm). Quality-neutral, parity GREEN BOTH models: FastWan
`video_e2e` step0/1 vel rel ~0.3% (band <=1.2%); LongLive `longlive_parity` vel_c0s0
slope 0.977 rel 3.1% (band <=6%). i8 error < the f16-vs-bf16 per-forward gap because
these A-sides are layernorm/modulated (no outliers).

QKV SITE-SPLIT (2026-06-21): qkv is now TWO matmul sites so only the safe half goes i8.
self-attn qkv (normed A-side, DP4A-safe) -> `matmul_qkv_self` (i8 when `--i8-matmul`);
cross-attn qkv -> `matmul_qkv` (ALWAYS bf16: its K/V project from UN-NORMED umT5 text and
i8 acts overflow f16, latent -> 65504). The split is a new `matmul_qkv_self` slot in the
shared `BlockWgslConfigs`/`BlockPipelines`/`BlockMatmuls` (== `matmul_qkv` unless pinned,
so FastWan/Z-Image/umT5 byte-identical), a `QkvSite` selector in `wan/dit_block.rs`
(`biased_proj`/`attention` route self vs cross), `WanI8Sites.qkv_self` + the loader
transcoding self-attn qkv only, and `SiteOverride.qkv_self` in `block_cfgs`. Measured
256x256x61: self-attn qkv matmul 4237->699ms (6.1x; +430ms dequant); cross-attn qkv
UNCHANGED ~4.9s; parity vel_c0s0 0.97690 (was 0.977 ffn_up-only -- self qkv i8 adds ~0).
NOT i8'd, kept bf16: proj + ffn_down (A-side = attention output / gelu, ~16k outliers;
per-32 i8 act-quant crushes them); cross-attn qkv (above).
NOTE: UNCOMMITTED on top of commit `1ce85c4` (an earlier env-var-only ffn_up version
the current tree refactors into the flag). Same uncommitted batch also adds CLI
`--duration <s>` (mutually exclusive with `--frames`, fps -> model default 24,
snaps to the legal frame grid; `VideoModelId::{fps,snap_frames,validate_frames}` +
unit tests in `generate/video.rs`).

## SHIPPED: DiT perf -- at the WGSL matmul ceiling (DEAD ENDS, do not retry)

DiT denoise (576x576x97, 3 DMD steps, ~55s/step) is 100% matmul+sdpa GPU time.
WEIGHT-ONLY quant does NOT help (dequant->bf16 matmul, same FLOPs); but i8 DP4A WITH
i8 acts IS a real win -- SHIPPED for ffn_up (section above), 6x on that op.
FFN 49% / attn (qkv+proj+sdpa) 51%. matmul ~2.5-3.4 TFLOP/s = ~20-25%
of the f16 issue ceiling; latency/occupancy bound (NOT bandwidth), so bigger tiles
backfire on the 28-SM mobile 5070. REGRESSED + reverted (tree clean): tile_b
per-kk2 register hoist (36.4->40.4 ffn_down); bk 16->64 (36.4->43.7). Only large
levers left are backend-level, NOT kernel tweaks: tensor-core matmul (5-10x; WGSL/
naga expose no WMMA, likely blocked) or i8 DP4A (DONE for ffn_up, 6x; qkv blocked on
the shared cross-attn site, proj/ffn_down on outliers -- see i8 section above).
Measure via e2e (NOT microbench):
`THINFER_E2E_SKIP_PYREF=1 THINFER_E2E_FRAMES=13`, read `gpu_ms by pipeline`
ms/disp. Baseline (f13): ffn_down 36.4 / ffn_up 26.5 / qkv 5.98 / proj 6.82 /
sdpa_sg 3.40.

## SHIPPED: DiT activation-tiling tier (PROVEN CORRECT)

`wan/dit_block.rs` + `wan/dit.rs`: per-block pass A (row-tiled norm1/qkv/qk-norm/
rope) -> global self-SDPA barrier -> pass B (row-tiled o-proj/residual/cross-attn/
FFN), each its own submit so FFN transients recycle. `DIT_TILE_ROWS=1024`; engages
only above one tile (~real video res/frames). Bit-exact: forcing 2 tiles at 256
stays GREEN vs pyref; 9 tiles at 576x576x13 is bit-IDENTICAL to 2 tiles. Bounds
VRAM at 8100 tok (pegged 5G, no OOM, `sdpa_sg` flash, no materialized [n,n]).

## Carry-forward gotchas (Wan-general; reused by LongLive)

- umT5 MUST run bf16 acts. Its residual stream grows past f16's 65504 by block ~20
  (peak ~67k at 576) -> inf -> NaN in `final_layer_norm` -> token-uniform hidden ->
  mirror-symmetric "washed blob". Magnitude is PROMPT-content dependent (so f16
  blew up only on some prompts). Fix: `pipeline.rs::load_with_act` compiles umT5
  bf16 (matches pyref bf16 text encoder); DiT stays f16 (the umT5->DiT seam is
  host-f32 readback+reupload, dtypes independent). Localize via `video_e2e.rs`
  WAN_DIAG first-NONFINITE walk -- check non-finite NOT just NaN (`inf.is_nan()`
  is false).
- RoPE3D is interleaved-pair, NOT half-rot (opposite of Qwen3). Freqs MUST pack to
  the act dtype (`freqs_upload_bytes`): f32 freqs into an f16 kernel -> inf -> NaN.
- bf16->f16 reinterpret class: broadcast vectors that are STORED WEIGHTS
  (scale_shift_table, norm2 affine) read via a `weight_dtype`-keyed op
  (`bcast_add`/`bcast_mul`), not an act-scale op (`bcast_affine`/`bcast_modulate`).
  New broadcast site: check weight vs act, match the op.
- DiT driver takes `text` as host f32 `[text_seq, text_dim]` (umT5 readback +
  reupload), zero-padded, no cross-attn mask. Clean seam; revisit if it costs.
- umT5 even-pads odd token counts by duplicating EOS; that pad key MUST be masked
  (`wan/umt5.rs`) or bidirectional attention double-counts it.
- VAE decode-tiling pattern (`plan_tiles`/`feather_1d`/`decode_tile`) is what the
  DiT tier mirrors in spirit.
- Shared helpers: Wan DiT reaches into `z_image::{block, embedders,
  rope_embedder, seq}`. Extract a `thinfer-models` common module before the family
  grows (LongLive is the trigger to consider it); not blocking.
- Video staging: per-frame PNG sequence + tiled contact sheet; MP4/WebM in the CLI
  only (openh264; encoder config hardened 2026-06-19).

## Running the e2e / measuring

Test is `video_e2e`. Parity gate (needs HF bundle + `uv`):
`THINFER_TRACE=1 THINFER_POWER_PREF=high THINFER_E2E_BUDGET_GB=6
THINFER_E2E_WIDTH=256 THINFER_E2E_HEIGHT=256 THINFER_E2E_PNG_DIR=<dir> cargo test
-p thinfer-conformance --features wan-e2e --release video_e2e -- --nocapture
--test-threads=1`. Perf/trace only: add `THINFER_E2E_SKIP_PYREF=1`. NEVER run the
fp32 pyref (`REF_DTYPE=fp32`) above tiny dims (~30GB weights, OOMs host). Card is
RTX 5070 Laptop (8GB); keep budgets <8GB (8GB OOMs the device). Per-op `gpu_ms` in
the trace rollup ("gpu_ms by pipeline") localizes perf.

CLI full run: `THINFER_TRACE=1 THINFER_POWER_PREF=high ./target/release/thinfer.exe
generate video --prompt ... --width 576 --height 576 --vram-budget 5G --ram-budget
5G --download-as-needed --output out.mp4` (default 97 frames; CFG-free DMD, no
steps/guidance flags). Rollup + `[mem]` dump at process EXIT. To inspect pixels:
ffmpeg (WinGet) `ffmpeg -i out.mp4 -vf "select=eq(n\,N)" -vframes 1 frame.png`, or
`--output-format png-frames`. Scratch artifacts live in `<repo-root>/scratch/logs/`
(sibling of `thinfer/`, not under working-area).
