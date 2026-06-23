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

## NEXT (active): FastWan UniPC sampling path -- LANDED, awaiting visual check

IMPLEMENTED (read the code, do not re-spec). UniPC is now the default FastWan
sampler everywhere (CLI + serve + web UI); DMD stays reachable + is the parity
path. PENDING = the user eyeballs a UniPC clip in the running thinfer-serve web UI
vs the KingNish Space (matched prompt/seed/res). If it looks right, done; if not,
fall back to the GPU upstream-pyref compare. What landed:
- `UniPcConfig::fastwan(steps)` (`wan/unipc.rs`): shift=8.0, 1000 train steps,
  sigma_min 0.001. Verified against the live KingNish `app.py`:
  `UniPCMultistepScheduler.from_config(..., flow_shift=8.0)`, steps default 4 /
  slider 1..=8, `guidance_scale=0` (CFG-free), 896x896 default.
- `VideoSampler { Dmd (default), UniPc{steps} }` on `GenerationParams`; the denoise
  loop in `denoise_with` branches on it. UniPc arm drives the existing `FlowUniPc`,
  no renoise, no guidance; DiT forward byte-identical to DMD (perf unchanged, only
  a tiny host-side latent history added -- no VRAM/residency change). step-diag taps
  stay DMD-only (parity gate uses `VideoSampler::Dmd` explicitly, still GREEN-able).
- Plumbed sampler + steps through `thinfer-app` (`model::VideoSampler`,
  `VIDEO_DEFAULT_STEPS=4`, `VideoRequest`, `wire::VideoSpec`), `serve` (defaults
  UniPC/4), CLI (`--sampler unipc|dmd`, `--steps`), and the web UI (Steps field
  now shown for video, 1..=8 default 4; server defaults the sampler to UniPC).
- Server rebuilt (release) + restarted (default config, 0.0.0.0:8080, embedded web);
  openapi.json carries the new `sampler`/`steps` VideoSpec fields.

ORIGINAL PROBLEM (kept for context): FastWan video looked bad at every res the user
tried (512x288..960x544, tiny AND full VAE). Steps were correct (DMD 3-step
`[1000,757,522]`, matches FastVideo official) and parity is GREEN at 256x256 -- so it
was NOT a kernel/step bug. ROOT CAUSE (found via the reference HF Spaces the user
trusts): they sample the SAME weights with a DIFFERENT denoiser. We used the DMD
re-noise sampler; they use plain UniPC multistep.

CONFIRMED RECIPE (two independent Spaces agree -- `KingNish/wan2-2-fast` app.py and
`rahul7star/Wan2.2-T2V-A14B` app_fast.py, both load `FastVideo/FastWan2.2-TI2V-5B-
FullAttn-Diffusers`): `UniPCMultistepScheduler.from_config(..., flow_shift=8.0)`,
`num_inference_steps=4` (slider 1-8), `guidance_scale=0` (CFG-free). (app_fast also
adds a content LoRA @0.95 -- INCIDENTAL, ignore.) Default res in both is 896x896
square, so square is fine for this model (earlier "avoid square for video" was wrong
for Wan2.2-TI2V).

PLAN (DiT forward stays byte-unchanged; only the sampler around it changes):
- REUSE the existing `FlowUniPc` solver (`wan/unipc.rs`, today only LongLive's). Its
  `UniPcConfig` fields: `sigma_min`, `shift`, `num_train_timesteps`, `sampling_steps`.
  For FastWan: `shift=8.0`, `sampling_steps=4`, `num_train_timesteps=1000`, sigma_min
  as LongLive. NOT touching LongLive.
- Add a NON-AR UniPC denoise loop in `wan/pipeline.rs::generate` as an alternative to
  the `DmdSampler` loop (~line 1046-1149). Per step i: forward DiT at
  `unipc.timestep(i)` -> velocity; `sample = unipc.step(&velocity, &sample)`. NO
  re-noise, NO guidance (CFG-free). Mirrors how `denoise_ar` drives `FlowUniPc`, but
  over the whole latent (no KV window).
- KEEP DMD-3 intact as the parity default so the GREEN gate stays valid. Add a sampler
  choice (DMD | UniPC) + a CONFIGURABLE step count to `VideoRequest`/`wire::VideoSpec`/
  the web UI. DECIDED: step count is a user knob (default 4, range 1-8, per the Spaces),
  NOT a fixed UniPC-4 -- the sparse-distill model is meant to run 3-5 steps. UniPC is
  the serve/UI default sampler for FastWan; DMD-3 stays reachable + is the parity path.
- VALIDATE by eyeballing a UniPC-4 clip vs the Space at matched prompt/seed/res. If it
  looks right, wire it as default; if not, fall back to the GPU upstream-pyref compare.

CARRY-FORWARD (already LIVE in the running `thinfer-serve` -- rebuilt + restarted this
session; do not re-spec, read the code): web UI now has a Size-preset dropdown (trained
aspects, /16 image & /32 video grids, hand-typed = "Custom"), a Quality (VAE) toggle
(tiny default / full), and a Duration(s) field replacing Frames+FPS (sends `durations`).
`default_frames` now = 5s @ model fps (`DEFAULT_DURATION_SECS`), snapping FastWan 121 /
LongLive 125. `ProgressStage::ChunkStep` now serializes camelCase (`numChunks`/`numSteps`)
so the LongLive progress line renders. Server diag probes muted by default
(`info,thinfer::diag=warn`; re-enable via `RUST_LOG`). NB: the user keeps a server
running and toys with it from a browser -- ASK before stopping/restarting it.

## NEXT: `thinfer serve` + OpenAPI + web client

Steps 1-4 LANDED (2026-06-23): the phone goal is reachable. `thinfer-app`
extraction + `thinfer-serve` v1 + `RemoteExecutor`/`--remote` + a server-backed
web UI all build green + clippy-clean; server boots + inits the 5070 worker, the
HTTP surface verified by curl (static UI open + 200, `/jobs*` + openapi gated 401,
valid-token-bad-spec 400 before any download). REMAINING here = the wasm<->http web
toggle + the weights+GPU deferrals below. GOAL (met for server mode): usable from a
phone against a `thinfer-serve` box. Image/video/faceswap over a typesafe OpenAPI
HTTP API + a web page that runs the SAME generation on the server (wasm mode TBD).

DONE THIS PASS (do not re-spec, read the code):
- Wire DTOs live in `thinfer-app::wire` (serde-gated; utoipa `ToSchema` under
  `serve`): `JobSpec`/`{Image,Video,FaceSwap}Spec`, `CreateResponse`, `JobStatus`,
  `JobResult`, `ProgressStage` (+`From<Stage>`/`From<ProgressStage>`),
  `JobStateKind`, `JobEvent` (+`kind`/`is_terminal`). serve re-uses them;
  `spec_into_request` (server-only: artifact path + budget from config) is a free
  fn in `serve::api`. job store/handle/SeqEvent stay server-side.
- `RemoteExecutor` (`thinfer-app::remote`, `remote` feature = reqwest rustls +
  futures-util + serde_json): POST spec -> tail SSE (own fetch-stream frame parser,
  unit-tested) into a `ProgressSink` -> download artifact. Mirrors `LocalExecutor`.
- CLI: `--remote <url>` + `--remote-token` (flattened `RemoteArgs`) on `generate
  image|video`; builds a `JobSpec`, runs via `RemoteExecutor` through the same
  `CliSink` lines. (faceswap remote intentionally absent: server-local path refs.)
- `ServeConfig`: `bind` default `0.0.0.0:8080`; optional `auth_token` (Bearer
  middleware on the `/jobs*`+openapi router only); optional `web_dir`.
- Web UI in `thinfer-serve/web/{index.html,style.css,app.js}`, embedded via
  `include_str!` (self-contained) or served from `web_dir` (dev). Vanilla JS,
  fetch-based SSE so the token rides an `Authorization` header (EventSource can't).
  Mounted outside the auth layer so it can load + prompt for the token. Assets
  served `Cache-Control: no-store` (stale app.js was mis-reading event fields).
- E2E result encryption: browser generates an RSA-OAEP keypair, sends only the
  PUBLIC key in the spec (`public_key`/`publicKey`, optional). Server hybrid-
  encrypts the artifact at rest (random AES-256-GCM key, RSA-OAEP-wrapped;
  `serve::crypto`, ring provider), serves opaque bytes; browser unwraps + AES-
  decrypts to an in-memory blob. No key => plaintext (CLI path). WebCrypto needs
  a SECURE CONTEXT (https/localhost); app.js warns + falls back to plaintext on
  insecure http.
- Result is DELETE-ON-FETCH: `GET /jobs/{id}/result` reads bytes, removes the job
  dir, returns them; second fetch 404s. Browser holds the only lasting copy.
- Opt-in HTTPS for the secure context over LAN: `tls_self_signed` (rcgen self-
  signed at startup, SANs = localhost+127.0.0.1+auto-detected LAN IP+`tls_sans`)
  or BYO `tls_cert`/`tls_key`. axum-server+rustls(ring), no aws-lc C build.
- MP4 faststart (`codec::faststart`): moov relocated before mdat, stco/co64 chunk
  offsets patched; web/strict-player playback + seeking. Unit-tested.
- Web result UI: explicit Download link with a `thinfer.{png,mp4}` filename (a
  bare blob: save dropped the extension -> OS "can't play").

STATE OF THE TREE:
- `thinfer-app` (new lib): `model` (id enums + defaults + frame grid + manifest),
  `request` (`JobRequest` + per-modality structs + shot-plan + `required_files`),
  `progress` (`Stage` + `ProgressSink`), `download`, `codec` (mp4/png-frames/faceswap
  stream), `executor::LocalExecutor`, `config` (`BackendConfig`/budget/mem). Features:
  `cli` (clap ValueEnum), `serde`, `serve` (serde+utoipa ToSchema on the id enums).
- `thinfer-cli`: thin clap adapters -> app requests; CLI keeps env->BackendConfig,
  consent prompt, decile download logging, stamped `CliSink`, mem rollup. Behavior
  preserved (--help defaults match; 10 shot-plan tests moved to app, green).
- `thinfer-serve` (new bin): axum + utoipa. `POST /jobs` (image/video queue; faceswap
  409-if-busy), `GET /jobs/{id}`, `GET /jobs/{id}/events` (SSE, in-mem log replay +
  Last-Event-ID), `GET /jobs/{id}/result` (streams from disk), `POST /jobs/{id}/cancel`,
  `GET /openapi.json` + `--emit-openapi`. Workers = OS threads w/ current-thread
  runtime (avoids Send bound on `!Send` generate futures), one `LocalExecutor` each
  (default 1). In-mem job metadata, on-disk artifacts under `artifact_dir/<id>/`, no
  DB. TOML `ServeConfig`.

DEFERRED (do these in step 2.5 / when running with weights):
- Mid-generate cancel: NOT wired (would touch the shipped z_image/wan `generate`
  signatures = DO-NOT-DISTURB FastWan). v1 cancel only dequeues a QUEUED job; a
  running job finishes. Add a cancel token (or make `ProgressFn` return ControlFlow)
  with a warm before/after, with sign-off.
- serve==CLI byte-parity test: same request via `--remote` vs local -> identical
  bytes (deterministic, fixed seed). The `--remote` vehicle now exists; needs
  weights+GPU, add under `wan-e2e`-style gating. (HTTP surface itself is curl-checked
  without weights; the SSE frame parser has unit tests in `remote.rs`.)
- Disk-backed SSE ring buffer (survive restart) -- v1 keeps the event log in memory.
- Server video is MP4-only (png-frames is a CLI debug format).
- 422 vs 400: a well-formed-JSON-but-wrong-shape body returns axum's 422 (handler's
  own semantic validation returns 400). Fine; revisit if a client needs 400.

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
1. DONE -- `thinfer-app` + `ModelId::defaults()` + `thinfer-cli` re-pointed (cancel
   token deferred, see DEFERRED).
2. DONE (v1) -- `thinfer-serve` (TOML config, worker pool, queue, SSE w/ in-mem log,
   disk artifacts, `openapi.json` + `--emit-openapi`). Disk ring buffer deferred.
3. DONE -- DTO move to `thinfer-app::wire` + `RemoteExecutor` + `thinfer-cli
   --remote`/`--remote-token`. See DONE THIS PASS above.
4. DONE (server mode) -- server-backed web UI served by `thinfer-serve` (embedded /
   `web_dir`). The wasm<->http toggle is NOT built (server-only by decision); when
   adding it, the seam is a TS-level `Executor` swap + the `thinfer-web` wasm pkg
   (image-only in-browser), and the user runs the web dev-server + reports browser
   results.

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
