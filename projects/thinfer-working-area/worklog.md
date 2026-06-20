# Worklog

Forward-looking only (git history is the changelog). Engine-wide design +
kernel/runtime state: `plan-details.md`. Per-model porting: `wan-plan.md`
(Wan2.2-TI2V-5B line; Wan-backbone / RoPE3D / umT5 / VAE / GGUF lore is
reusable, SkyReels-DF sections obsolete). Z-Image (shipped): `zimage-plan.md`.

## Status

- **Parity GREEN** at 256x256x5 / 6GB (`parity OK vs pyref`). DiT forward is
  bit-clean at high-t (step0/1 slope ~1.000, rel_rmse ~0.003); the final low-t
  step + VAE carry the f16 act precision floor amplified by the stiff low-sigma
  velocity field (slope 0.99->0.96, rel_rmse <= 0.053), proven precision NOT a
  bug. Prod runs f16 acts; fp32 is test-only. Gate metric = slope + rel_rmse
  bands (`CmpTol` in `video_e2e.rs`), not cell-count-over-tol. The parity
  default is 256x256x5/6GB; a tiny grid is a perf/health smoke only (a 2-token
  grid lets one outlier channel dominate the whole-tensor slope -> false red).
- **576x576x5 fits a 2GB budget** (TRUE_PEAK 1.98GiB; thin-hardware goal met)
  and produces coherent NaN-free frames at real resolution.
- **Generation is correct on arbitrary prompts.** The old "washed mirror blob"
  was umT5 f16 overflow, fixed by running umT5 in bf16 acts (see below). A full
  576x576x97 CLI clip is coherent NaN-free (5.00GiB peak @ 5GB budget).

## Tiny VAE (LightTAE) -- SHIPPED, now the CLI default

ONE NAME EVERYWHERE: "tiny" (the ecosystem/TAE name). `VaeChoice::Tiny`, `--vae
tiny`, `role::TINY_VAE`; "turbo" is gone from the VAE surface (Z-Image-Turbo the
model name is unrelated). `wan/vae_tiny.rs` LightTAE (`lighttaew2_2`) is the
`--vae` default; `--vae full` is the opt-in parity path. Arch + bias/dtype map:
`scratch/notes/lighttae_spec.md`. 576x576x97: end-to-end ~179s (DiT-bound, ~93%
denoise); decode ~4.4s.

DECODE TILING -- LANDED. Temporal-chunk over latent frames, sized from the VRAM
budget (`plan_chunk`, `TINY_PEAK_K` calibrated; `THINFER_VAE_TINY_CHUNK` /
`THINFER_VAE_TINY_MEM` for calibration). `memcat` is the ONLY temporal coupling
(causal depth 1), so each chunk carries every MemBlock's trailing input frame
into the next via a ping-pong device buffer; output frames concatenate exactly
(no halo, no seam). `memcat` op extended with a `prev`-frame binding + `has_prev`
flag (`has_prev=0` = the old zero-pad, conformance fixture bit-identical). A clip
that fits runs as one chunk, bit-identical to untiled. VERIFIED: full 576x576x97
holds a 2GB budget (TRUE_PEAK 2.00GiB, cf=7/4 chunks, NaN-free) -- was 4.98GiB
untiled, the case that used to force `--vae full` on thin budgets.

DURABLE TEST -- LANDED. `video_e2e` `THINFER_E2E_TINY=1` loads the tiny decoder
alongside Full (gate untouched) and, post-gate, decodes the same latent single-
vs multi-chunk: asserts bit-identical (proves the carry is exact, no pyref
needed) + NaN-free + clamp range. The whole-run TRUE_PEAK assert now also covers
the tiny decode. Exercises the multi-chunk carry even at the 256 grid.

Follow-ups (open):
- Conv tile is `Conv2dConfig::DEFAULT` (untuned; decode is no longer the
  bottleneck so low priority).
- TINY_PEAK_K is anchored to one measured point (576x576x97). Recalibrate if the
  decoder graph changes or a new res/budget reveals the model is off (the
  `set_transient_reserve` backstop prevents overshoot meanwhile).
- A true same-latent A/B (Full vs Tiny, identical prompt+seed) was NOT run;
  quality judged on the tiny clip alone. Worth one if a regression is suspected.
- No decode-vs-TAEHV-pyref parity (the e2e tiny case is health + tiling-exactness
  only); add one when wiring the TAEHV pyref.

## VRAM: VAE decode tiling (the hog at higher res, not DiT attention)

The decode live set is `FIXED(tout) + area*PER_AREA(tout)` per spatial tile;
each `decode_frame` temporally upsamples one latent frame to `tout` video frames
(`temporal_compression`=4 for groups after the first), so a video group carries
a ~553MiB temporal FIXED floor + ~6MiB/latent-area that tiling shrinks. Tile is
sized from `budget - reserve` where `reserve = real VAE weight footprint
(queried via residency.total_registered_bytes diff) + staging`, NOT a budget
fraction; `set_transient_reserve` (same mechanism DiT uses) is the hard-ceiling
backstop so the arbiter caps weights and never overshoots (degrades to weight
paging if the estimate is off). Model constants in `vae_tile_dims` are
calibrated from 4 measured points; recalibrate if the decoder graph changes.

- Thin-budget cost: at 2GB/576 the tile floors at 8 (64 tiles) -> VAE decode
  ~87s vs ~55s untiled. One-time per clip; only at thin budgets. At 6GB the gate
  (256) is single-tile / bit-identical; 576 tiles only ~2x2.
- 2GB is near the floor for 576 with weights resident (weights 1058 + temporal
  FIXED ~553 + staging). Below ~2GB the backstop pages VAE weights (fits, slower).
- DiT weights are freed before VAE (`denoise_with` -> `evict_all_and_free`); the
  ~1GB resident during decode is the VAE's own weights.
- `THINFER_VAE_MEM` env gates a per-tile workspace probe (off by default).

## Carry-forward: umT5 MUST run bf16 acts (the "odd-token blob" fix)

The "washed mirror blob" was NOT an odd-token masking bug (masking is correct;
DiT-tiling / SDPA-scale / RoPE all exonerated). umT5's residual stream grows
monotonically past f16's 65504 ceiling by block ~20 (peak ~67k at 576) -> inf ->
NaN in `final_layer_norm` (inf*rsqrt(inf) = the `{NaN, 0.0}` hidden, which then
collapses `text_proj` to token-uniform -> mirror-symmetric latent). Magnitude is
PROMPT-CONTENT dependent, so f16 blew up only on some prompts; the even/odd-token
correlation was coincidence (the even gate prompt stayed in range).

Fix: `pipeline.rs::load_with_act` compiles umT5 with bf16 acts (`umt5_act`),
matching the pyref bf16 text encoder. bf16 has f32's exponent range so it holds
~67k; DiT stays f16 (the umT5->DiT seam is host-f32 readback+reupload, so the two
act dtypes are independent). GREEN: corgi parity OK vs pyref (umt5_hidden slope
1.001 / rel_rmse 1.4e-3, was nan=27168/36864); full 576x576x97 clip coherent.

Localize via `video_e2e.rs` WAN_DIAG branch (first-NONFINITE walk over umT5
embeds/layers/ops). GOTCHA: check non-finite, NOT just NaN -- `inf.is_nan()` is
false, so the overflow hid from a nan-only pass until the layer max_abs was
printed.

## VAE decode perf: conv-GPU bound (NOT submit-bound); conv tiles tuned

CORRECTION (2026-06-20): the prior "VAE is SUBMIT/SYNC bound" read was WRONG.
Timestamp totals (true GPU exec, not overlapping wall) at 576x576x49 / 5GB show
the VAE wall is ~95% pure conv GPU time, near-zero idle to recover. The earlier
"sawtooth / half idle" eyeball was misleading (the bandwidth-bound conv reads as
low SM occupancy, not idle). Deepening `ScopePacker::MAX_INFLIGHT` would buy
<=5%; do NOT chase it. Authoritative metric: `gpu_disp_ms` (timestamp) for the
`vae_decode` scope vs scope `busy_ms`, NOT nvidia-smi utilization.

Baseline 576x576x49 / 5GB (`scratch/logs/vae_perf_baseline.log`): VAE wall 161s
= conv3d 134s (72ms/disp) + conv2d 18.5s (119ms/disp) + <4s everything else.

LANDED: VAE conv tiles tuned (`wan/vae.rs` `WAN_VAE_CONV3D_TILE` /
`WAN_VAE_CONV2D_TILE` = 128x96x16, tm8 tn6). The implicit-GEMM convs are
memory-bandwidth bound, so global traffic per output `bk*(1/bm+1/bn)` is the
lever: `bm=128` halves weight-side reads vs the 64x64 `Conv*Config::DEFAULT`.
Sweep found 48 accumulators / 256 threads is the occupancy knee (64 acc =
128x128 REGRESSED on register pressure; >256 threads fails the invocation cap).
Bit-EXACT (f32 accum, ascending-k order is tile-shape-independent) -> 256 parity
gate stays GREEN (vae_rgb slope 0.942 / rel_rmse 0.062, the known precision
floor). Measured at 576x576x49: conv3d 134->95s (-29%), conv2d 18.5->12.7s
(-32%), VAE wall 161->117s (-28%). Extrapolates to ~310->~224s at x97.
Sweep harness: `scratch/sweep_conv.sh` (scratch-only).

NEXT levers if more VAE perf is needed (all still conv-GPU bound, ~95s conv3d):
- Tile-overlap recompute: thin-budget tiling decodes overlapping latent tiles
  with a ~25% halo -> redundant conv on halo pixels. Budget-aware larger tiles
  (fewer/smaller halos) cut conv work directly; trade vs seam quality.
- Packed/vectorized x gathers in the conv kernel (f16x2 / vec4) to lift the
  bandwidth ceiling. Deeper kernel work; keep f32 accum order for bit-exactness.
- `conv3d_small_n` (cout=3/12 convs) still on its own tiny tile; minor.
- DO NOT tweak conv3d kernel MATH. The im2col loop-invariant-div hoist was TRIED
  and REVERTED 2026-06-19: REGRESSED conv3d GPU 262->301s (occupancy loss). Tile
  tuning (above) is the throughput lever, not index-math.

## DiT activation-tiling tier (engages at scale; PROVEN CORRECT)

`wan/dit_block.rs` + `wan/dit.rs`: per-block pass A (row-tiled
norm1/qkv/qk-norm/rope) -> global self-SDPA barrier -> pass B (row-tiled
o-proj/residual/cross-attn/FFN), each movement its own submit so FFN transients
recycle; only qx/kx/v/sa stay resident across the barrier. `DIT_TILE_ROWS=1024`;
engages only above one tile (~1024 tok = real video res/frames).
- VRAM: VERIFIED at 8100 tok -- engages, bounds VRAM (pegged 5G, no OOM), SDPA
  streams (`sdpa_sg` flash, no materialized `[n_tok,n_tok]`).
- CORRECTNESS: bit-exact. `THINFER_DIT_TILE_ROWS=64` forcing 2 tiles at the 256
  parity grid stays GREEN vs pyref; forcing 9 tiles at 576x576x13 is bit-IDENTICAL
  to 2 tiles (FWD_REF rel_rmse 0.0). Tile count does not change output. The
  blocker is umT5, not this tier.

## After FastWan: LongLive-2.0-5B (AR/causal long video)

Same Wan2.2-TI2V-5B base, 4-step DMD, autoregressive/causal. ~90% shared engine;
adds the AR/causal attention regime. GGUF DEFERRED (pin a revision, SCRATCH GGUF
for bringup; `dit_gguf_renames` is Wan-family-general, re-verify vs a real
FastWan GGUF dump; umT5 map is model-agnostic).

## Carry-forward gotchas (Wan-general)

- RoPE3D is interleaved-pair, NOT half-rot (opposite of Qwen3). Freqs MUST pack
  to the act dtype (`freqs_upload_bytes`): f32 freqs into an f16 kernel -> inf ->
  NaN softmax (`wan/dit.rs`).
- bf16->f16 reinterpret class: broadcast vectors that are STORED WEIGHTS
  (scale_shift_table, norm2 affine) read via a `weight_dtype`-keyed op
  (`bcast_add`/`bcast_mul`), not an act-scale op (`bcast_affine`/
  `bcast_modulate`). New broadcast site: check weight vs act and match the op.
- DiT driver takes `text` as host f32 `[text_seq, text_dim]` (umT5 readback +
  reupload), zero-padded, no cross-attn mask. Clean seam; revisit if it costs.
- umT5 even-pads odd token counts by duplicating EOS; that pad key MUST be masked
  (`wan/umt5.rs`) or bidirectional attention double-counts it.
- VAE decode-tiling pattern (`plan_tiles`/`feather_1d`/`decode_tile`) is what the
  DiT tier mirrors in spirit.
- Shared helpers: Wan DiT reaches into `z_image::{block, embedders,
  rope_embedder, seq}`. Extract a `thinfer-models` common module before the
  family grows; not blocking.
- Video staging: per-frame PNG sequence + tiled contact sheet; MP4/WebM in the
  CLI only (openh264).

## Running the e2e / measuring

Test is `video_e2e`. Parity (the gate; needs the HF bundle + `uv`):
`THINFER_TRACE=1 THINFER_POWER_PREF=high THINFER_E2E_BUDGET_GB=6
THINFER_E2E_WIDTH=256 THINFER_E2E_HEIGHT=256 THINFER_E2E_PNG_DIR=<dir> cargo test
-p thinfer-conformance --features wan-e2e --release video_e2e -- --nocapture
--test-threads=1`. Perf/trace only: add `THINFER_E2E_SKIP_PYREF=1`. NEVER run the
fp32 pyref (`REF_DTYPE=fp32`) above tiny dims (~30GB weights, OOMs host). Card is
an RTX 5070 Laptop (8GB); an 8GB budget OOMs the device -- keep budgets <8GB.
Per-op `gpu_ms` in the trace rollup ("gpu_ms by pipeline") localizes perf.

CLI full run (for the large-token blocker): `THINFER_TRACE=1
THINFER_POWER_PREF=high ./target/release/thinfer.exe generate video --prompt ...
--width 576 --height 576 --vram-budget 5G --ram-budget 5G --download-as-needed
--output out.mp4` (default 97 frames = ~4s; CFG-free DMD, no steps/guidance/cfg
flag exists; frames default is good, don't set it). The trace rollup + `[mem]`
dump only at process EXIT. To LOOK at pixels (the codec is fine, inspect the
generation): ffmpeg is installed under WinGet; extract frames with `ffmpeg -i
out.mp4 -vf "select=eq(n\,N)" -vframes 1 frame.png` then read the PNG. Or use
`--output-format png-frames` for the codec-free decode. NB: the mp4 encoder
config was hardened 2026-06-19 (sane bitrate, `enable_skip_frame(false)`,
`max_frame_rate=fps`) replacing openh264's 120kbps+skip default; correct but
incidental (not the bad-video cause). Scratch artifacts live in
`<repo-root>/scratch/logs/` (sibling of `thinfer/`, not under working-area).
