# Worklog

Forward-looking only -- git history is the changelog, the code is the record.
Past work appears here ONLY as a one-line lesson or a do-not-retry. Engine-wide
design + kernel/runtime state: `plan-details.md`. Per-model porting: `wan-plan.md`
(Wan2.2-TI2V-5B line; backbone / RoPE3D / umT5 / VAE / GGUF lore reusable,
SkyReels-DF obsolete). Z-Image: `zimage-plan.md`. Scratch is ephemeral and
clearable -- nothing here depends on a scratch file.

## Status

- **FastWan2.2-TI2V-5B-FullAttn** -- shipped baseline, parity GREEN. bf16 acts
  (DiT + umT5). DO NOT DISTURB.
- **LongLive-2.0-5B** (AR/causal long + multi-shot video, same Wan base) --
  shipped: AR path, multi-shot scene-cuts, self-attn-qkv i8 all landed, e2e
  health-GREEN, parity GREEN (two-tier bands). Remaining = AR perf + a multi-shot
  pyref byte-parity (see the LongLive section).
- **Face-swap** (`thinfer generate face-swap`) -- shipped incl. the 4K / B-frame /
  streaming / audio correctness pass. NEXT = quality (XSeg occlusion + GFPGAN
  enhancer + use HyperSwap's own mask output); see `faceswap-plan.md`.
- **i8 DP4A matmul is ON by default** (opt out `--no-i8-matmul` = bf16 reference
  path); see the i8 lesson below.

## NEXT (planned): `thinfer serve` + OpenAPI + web client

Not started (design converged 2026-06-22). A long-running server exposing
image/video/faceswap generation over a typesafe OpenAPI HTTP API + an example web
page that runs the SAME generation either in-browser (wasm) or on the server, as a
toggle. GOAL: usable from a phone against a `thinfer-serve` box.

CORE ABSTRACTION -- one `Job`, one progress vocabulary, one `JobExecutor` trait,
consumed everywhere: CLI (local/remote), serve (local pool), web (wasm/http).
- `JobExecutor` impls in `thinfer-app`: `LocalExecutor` (download/load/generate on
  this machine) and `RemoteExecutor` (HTTP client to a `thinfer-serve`: POST job,
  stream SSE, render the SAME stderr progress lines, download result). Feature-gate
  `RemoteExecutor` behind a `remote` feature (reqwest/SSE deps).
- `thinfer-cli --remote <url>` selects `RemoteExecutor`; args parse into the same
  request struct either way. Exact mirror of the web app's wasm<->http toggle.

CRATE LAYOUT. `thinfer-app` is a NEW lib (not a rename): holds the `Job`, request
types, progress vocabulary, `JobExecutor` trait + `LocalExecutor`, and the
orchestration currently duplicated across `cmd/generate/{image,video,faceswap}.rs`
(validate -> parse budget -> resolve variant -> resolve/download -> open mmap
openers -> init backend -> generate -> encode). Built on `thinfer-native` +
`thinfer-models`. `thinfer-cli` shrinks to a clap adapter; `thinfer-serve` = axum
adapter. Future native consumers (`-desktop`/`-python`/`-android`) reuse `app` too.
- Layering: core < models (dep-clean) < native < app < {cli, serve, ...}.
- `thinfer-web` (wasm) is the ONE exception -- own browser substrate (fetch/JS file
  handles, no tokio/mmap), CANNOT dep on `app`; it already reimplements the load
  dance and calls `thinfer-models` directly. Cross-language sharing is via OpenAPI
  (generated TS / future python clients), NOT a shared Rust types crate -- so types
  stay in `app`, no separate types crate until a real second consumer needs it.

API SURFACE (no `/v1`, no models-list endpoint; model id is an OpenAPI enum in the
request schema so it cannot drift):
- `POST /jobs` -> `{id, queue_position}`. Large-INPUT jobs (faceswap) return `409`
  if the worker is busy instead of queuing; small-input jobs (image, video t2v)
  queue. Per-request-type rule, single endpoint -- faceswap is still a job.
- `GET /jobs/{id}` -> status snapshot (poll fallback).
- `GET /jobs/{id}/events` -> SSE: `queued{position}/started/progress{phase,i,n,eta}
  /done{result_url}/error`. Disk-backed ring buffer + `Last-Event-ID` reconnect
  (jobs run minutes; clients drop). This is a THIRD sink for the already-structured
  `ProgressEvent`/`ProgressFn` (NOT stdout scraping) alongside CLI stderr + web JS
  fn. Unify the per-model events (z-image TextEncode/Step/VaeDecode vs video/
  faceswap phases) into one vocabulary.
- `GET /jobs/{id}/result` -> streams the artifact from disk (content-type + range;
  NEVER base64 a video into JSON).
- `POST /jobs/{id}/cancel` -> cancel/dequeue; job SURVIVES as `cancelled` (cancel !=
  delete). `DELETE /jobs/{id}` optional/skipped for v1 (TTL sweep handles cleanup).
- Cancellation needs a cancel token threaded into the `generate` signature
  (alongside `progress`, checked between steps) -- decide NOW, it touches the sig.

LARGE DATA / STORAGE: disk for blobs, in-memory for metadata, NO DB. Inputs
(faceswap = disk path-ref, gated to localhost/allowlist -- arbitrary-file-read
footgun), outputs, and the SSE ring buffer live under an artifact dir keyed by job
id. Metadata (state, params, queue pos, progress snapshot, result path, error) in an
in-memory map. ACCEPTED tradeoff: restart loses in-flight/queued metadata (artifacts
orphan -> TTL sweep); fine for single-user. Retain-with-TTL + max-bytes LRU (not
stream-once) so the page can re-download / show a small gallery.

CONCURRENCY: worker POOL of 1..N (one per device), CONFIGURABLE (default 1). Local
N=1 -- DiT denoise is at the matmul ceiling, compute-bound, batch=1 saturates the
28-SM mobile 5070, so concurrency = pure latency tax. Multi-GPU = trivial N workers.
Intra-GPU batching (the vLLM analogy) is NOT v1: vLLM batches bandwidth-bound LLM
DECODE; diffusion is compute-bound + lockstep over a fixed latent (opposite shape),
so batching only pays on UNDER-saturated big GPUs and needs dim/step-compatible
splicing. Job carries dims/steps/model so it is addable later with no schema change.

CONFIG: TOML file for `thinfer-serve` (deployment: bind addr, worker count / device
list, artifact dir, retention TTL, auth token, budgets) + a few flag overrides.
SEPARATE from generation defaults (steps/dims/guidance), which move into the model
registry (`ModelId::defaults()`, the existing `image.rs` TODO) so CLI + API read one
source and cannot drift.

WEB APP (served by `thinfer-serve`): NOT circular at the cargo graph -- `serve` does
not dep the `thinfer-web` crate. Seam is `openapi.json` (emit via a small bin/test,
no GPU). The pnpm/TS app builds against `openapi-typescript`/`openapi-fetch` (server
mode) + the `thinfer-web` wasm npm pkg (wasm mode); the wasm<->http toggle is a
TS-level `Executor` swap behind one interface. Embed built assets behind a feature
flag: release embeds (self-contained binary), dev serves from a dir (edit-reload, no
Rust rebuild). CAVEATS for the toggle: it is per-CAPABILITY -- wasm does image only
(10GB video won't fit an 8GB browser budget; faceswap is native-tool-heavy), so
video/faceswap are server-only in the UI; and output format diverges (server MP4 via
native openh264, wasm PNG-frames / WebCodecs), so the result type carries its format.

STACK: axum + `utoipa`/`utoipa-axum` (OpenAPI 3.1, OpenApiRouter = route+docs), Scalar
docs UI; GPU on dedicated worker thread(s) / spawn_blocking. Feature-gate the utoipa
`ToSchema` derives so they never reach a non-serve build.

BUILD ORDER:
1. `thinfer-app`: `Job`, request types, progress vocabulary, `JobExecutor` +
   `LocalExecutor`; cancel token into `generate`; `ModelId::defaults()`. Re-point
   `thinfer-cli` at it with NO behavior change. Durable proof: serve==CLI byte
   parity test (deterministic engine, same request -> same bytes).
2. `thinfer-serve`: TOML config, worker pool (1 default), queue, SSE + disk ring
   buffer, disk artifacts, `openapi.json` emit.
3. `RemoteExecutor` + `thinfer-cli --remote`.
4. Web executor toggle (wasm<->http) + serve the built app. <- phone goal lands here.

## NEXT (active): LongLive-2.0-5B AR perf

Read the code, not a re-spec (`WanDit::forward_ar`, `WanModel::denoise_ar`/
`generate_ar`, `wan/kv_cache.rs`, `wan/unipc.rs`).

LongLive runs many forwards (chunks x [4 UniPC steps + 1 clean recache]) vs
FastWan's 3, so it is SLOWER at a fixed length -- the O(N)-streaming win is
LENGTH/VRAM-bound generation, not shorter wall. Warm (page-cache hot) the AR
forward at 576 is ~86% compute-bound (matmul ceiling, same ops as FastWan); the
~15% "idle" seen cold is a disk artifact, NOT recoverable per-block streaming.
RE-MEASURE WARM at 576 before chasing any lever. Wins left (quality-neutral, exact),
value order:
- Upload the window prefix ONCE per chunk, not per forward (identical across a
  chunk's forwards; cuts HtoD traffic).
- Activation-tile the AR self-attn (workspace ~= budget at 576; mirror
  `forward_block_tiled`) -- buys prefetch/residency headroom.
- Cache cross-attn text K/V across a chunk's forwards (same prompt; we have no
  cudagraph so it is free).
- Skip the head (proj_out + unpatchify) on the clean recache pass (velocity
  discarded; small).
DEAD END: prefetch-overlap in `forward_ar` (mirror `forward`'s `join!`) -- TRIED +
REVERTED, ~1% warm (noise). Do not re-add without a warm before/after.

Also OPEN: multi-shot is only health-tested. A multi-shot pyref byte-parity (extend
`gen_longlive_video_e2e_ref.py` to the multi-prompt block list) is the durable proof
if multi-shot quality must be guaranteed; the `zero_for_scene_cut` path is unexercised.

AR-loop invariants (do not relitigate):
- Within a chunk, `current_start` is CONSTANT across the 4 UniPC steps + clean pass;
  each forward recomputes the chunk K/V at the same tail slot. Only the timestep-0
  clean pass K/V are committed (survive into future chunks).
- Convert frames<->tokens with `frame_seq_len = pph*ppw` everywhere.
- ABSOLUTE-position temporal RoPE: q + chunk-k rotate at `chunk_start_frame =
  current_start/frame_seq_len`; cached prefix-k stored already-roped (no re-rotation
  at attention time); `temporal_offset = shot_index * 8` folds into the frame id
  (`rope3d.rs lookup_temporal`).
- Single-prompt T2V modulation collapses to FastWan's scalar-t broadcast, so
  forward_ar reuses the FastWan embedder/modulation unchanged; ONLY self-attn differs.

GROUND TRUTH IS THE CLONE (`third-party/LongLive`, NVlabs/LongLive): AR loop
`pipeline/causal_diffusion_inference.py`, self-attn/cache `wan_5b/modules/
causal_model.py`, sampler `wan_5b/utils/fm_solvers_unipc.py`, `configs/inference.yaml`
(chunk 8 / window 32 / sink 8 / 4 steps / shift 5.0). IGNORE nvfp4. Weights: HF
`Efficient-Large-Model/LongLive-2.0-5B` `model_bf16.pt` (10GB, 825 bf16 tensors,
complete merged DiT, LoRA pre-folded); umT5+VAE reused from FastWan.

LESSON -- LongLive AR parity (RESOLVED, do NOT re-open the op-hunt): the engine is
arithmetically faithful per-op AND per-forward (`vel_c0s0` within band; AR path
bit-identical to GREEN FastWan). The only gap is AR-depth compounding of the
16-bit-vs-bf16-locked-pyref per-forward rounding (~2%/forward in large-outlier
residual channels, blocks 15-25) across many forwards -- NOT a code bug, not
precision-fixable (upstream hardcodes bf16 SDPA). Tolerated via the two-tier band in
`longlive_parity.rs` (tight `TOL_LATENT` on the single-forward `vel_c0s0`; loose
gross-regression floor on the AR-compounded tensors).

DECISIONS LOCKED: bf16 full precision, NO quant (NVFP4 skipped); runtime `.pt`
ingestion (no build step / no on-disk dup -- footprint first-class); FastWan `forward`
stays byte-unchanged (AR behavior lives in the AR path only); track here (no
longlive-plan.md).

## Lessons / dead-ends (do not retry)

- **DiT denoise is at the WGSL matmul ceiling.** 100% matmul+sdpa GPU time;
  ~2.5-3.4 TFLOP/s = ~20-25% of the f16 issue ceiling, latency/occupancy bound (NOT
  bandwidth), so bigger tiles BACKFIRE on the 28-SM mobile 5070. Weight-ONLY quant
  does not help (dequant->bf16, same FLOPs). Only real levers are backend-level:
  tensor-core (WGSL/naga expose no WMMA -- likely blocked) or i8 DP4A (done).
  REVERTED: tile_b per-kk2 hoist, bk 16->64. Measure via e2e
  (`THINFER_E2E_SKIP_PYREF=1 THINFER_E2E_FRAMES=13`, read `gpu_ms by pipeline`), NOT
  microbench.
- **i8 DP4A matmul (default on; `--no-i8-matmul` = bf16 reference).** Transcodes
  ffn_up + self-attn-qkv weights bf16->Q8_0 at load, routes the i8xi8 `dot4I8Packed`
  path; ~5-6x those ops, ~-30% wall, quality-neutral (parity GREEN both models; i8
  error < the f16-vs-bf16 per-forward gap because these A-sides are normed/modulated,
  no outliers). NOT i8'd, stays bf16: proj + ffn_down (A-side attention-out/gelu has
  ~16k outliers that per-32 act-quant crushes); cross-attn qkv (K/V project from
  UN-NORMED umT5 text, i8 acts overflow f16). The qkv site is split into
  `matmul_qkv_self` (i8) vs `matmul_qkv` (bf16) in the shared block configs (== each
  other unless pinned, so FastWan/Z-Image/umT5 byte-identical).
- **DiT activation-tiling tier** (`wan/dit_block.rs`/`dit.rs`): pass A (row-tiled
  norm1/qkv/qk-norm/rope) -> global self-SDPA barrier -> pass B (row-tiled
  o-proj/residual/cross-attn/FFN), each its own submit. `DIT_TILE_ROWS=1024`, engages
  above one tile. Bit-exact; bounds VRAM (flash `sdpa_sg`, no materialized [n,n]).
- **VAE decode is conv-GPU bound** (~95% pure conv time; authoritative metric is
  `gpu_disp_ms` for `vae_decode`, NOT nvidia-smi util). Conv tiles tuned (`wan/vae.rs`
  128x96x16) -- implicit-GEMM convs are bandwidth-bound, bm=128 halves weight reads,
  48 acc/256 threads is the occupancy knee; bit-exact (f32 accum). DO NOT retry the
  conv3d im2col loop-invariant-div hoist (REVERTED, 262->301s).
- **VAE decode VRAM tiling**: live set `FIXED(tout) + area*PER_AREA(tout)` per spatial
  tile, sized from `budget - reserve` (reserve queried, not fractional); DiT weights
  freed before VAE (`evict_all_and_free`). Constants in `vae_tile_dims` (recalibrate
  if the decoder graph changes).
- **Tiny VAE (LightTAE) is the `--vae` default** (`--vae full` = parity path).
  Temporal-chunk decode tiling: `memcat` ping-pong carries each chunk's trailing frame
  (causal depth 1, no halo); a clip that fits = one chunk, bit-identical to untiled.
  Durable test `THINFER_E2E_TINY=1`.

## Carry-forward gotchas (Wan-general; reused by LongLive)

- umT5 MUST run bf16 acts. Its residual stream grows past f16's 65504 by block ~20 ->
  inf -> NaN in `final_layer_norm` -> token-uniform "washed blob" (magnitude is
  PROMPT-dependent, so f16 blew up only on some prompts). `pipeline.rs::load_with_act`
  compiles umT5 bf16; DiT stays f16 (the umT5->DiT seam is host-f32 readback). Check
  non-finite NOT just NaN (`inf.is_nan()` is false).
- RoPE3D is interleaved-pair, NOT half-rot (opposite of Qwen3). Freqs MUST pack to the
  act dtype (`freqs_upload_bytes`): f32 freqs into an f16 kernel -> inf -> NaN.
- bf16->f16 reinterpret class: broadcast vectors that are STORED WEIGHTS
  (scale_shift_table, norm2 affine) use a `weight_dtype`-keyed op (`bcast_add`/
  `bcast_mul`), not an act-scale op. New broadcast site: check weight vs act.
- umT5 even-pads odd token counts by duplicating EOS; that pad key MUST be masked
  (`wan/umt5.rs`) or bidirectional attention double-counts it.
- DiT driver takes `text` as host f32 `[text_seq, text_dim]` (umT5 readback + reupload),
  zero-padded, no cross-attn mask.
- Shared helpers: Wan DiT reaches into `z_image::{block, embedders, rope_embedder,
  seq}`. Consider extracting a `thinfer-models` common module as the family grows.
- Video: per-frame PNG sequence / tiled contact sheet for staging; MP4/WebM in the CLI
  only (openh264).

## Running the e2e / measuring

Card is RTX 5070 Laptop (8GB); keep budgets <8GB (8GB OOMs the device).
- FastWan parity gate (needs HF bundle + `uv`): `THINFER_TRACE=1
  THINFER_POWER_PREF=high THINFER_E2E_BUDGET_GB=6 THINFER_E2E_WIDTH=256
  THINFER_E2E_HEIGHT=256 THINFER_E2E_PNG_DIR=<dir> cargo test -p thinfer-conformance
  --features wan-e2e --release video_e2e -- --nocapture --test-threads=1`. Perf/trace
  only: add `THINFER_E2E_SKIP_PYREF=1`. NEVER run the fp32 pyref above tiny dims
  (~30GB host).
- LongLive parity: `longlive_parity` (use 256x256, NOT the 128 default which is
  pathological); LongLive e2e: `longlive_e2e`. Both `--features wan-e2e`.
- CLI full run: `THINFER_TRACE=1 THINFER_POWER_PREF=high thinfer generate video
  --prompt ... --width 576 --height 576 --vram-budget 5G --ram-budget 5G
  --download-as-needed --output out.mp4`. Rollup + `[mem]` at process EXIT; read per-op
  `gpu_ms by pipeline` to localize perf. Inspect pixels via `--output-format png-frames`
  or ffmpeg.
