# Z-Image plan (shipped; archived reference)

Z-Image-Turbo t2i is complete on native + web. This file parks the
Z-Image-specific state and the perf-mining history out of the worklog. Engine
machinery reused across models lives in `engine-reference.md`.

## Shipped state

- Full t2i (Qwen3 encoder -> DiT -> VAE) on wgpu, native + browser (wasm). q4
  default, all parity green. Native q8 256 pyref bit-clean (rgb slope 1.003478 /
  2.22% with pinning).
- Quant variants take the COMPLETE DiT+TE from GGUFs (q4 9.5GB / q8 14.7GB /
  bf16 20.5GB).
- Sources: Tongyi-MAI/Z-Image, diffusers ZImage pipeline, DiT GGUF
  unsloth/Z-Image-Turbo-GGUF (Turbo = distilled few-step), TE GGUF
  worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF, VAE safetensors upstream.

## Z-Image-specific invariants

- Pin plan (`set_pin_plan`): pins a `pin_priority` prefix up to `budget -
  ring_reserved - workspace_reserve - staging_reserve`, installed in
  `denoise_with` after assemble. Pin-on-first-touch; `evict_all_and_free` drops
  the plan at denoise end so VAE reclaims. `pin_priority` role-major/block-minor,
  largest roles + refiners/embedders first.
- ActDtype::I8 matmul/sdpa-internal ONLY; residual/norms/glue dense at act
  dtype. Qwen3 `mlp_down` stays bf16. i8 acts are matmul/sdpa-boundary only
  (block-wide i8 acts removed 2026-06-05, outlier channels).
- Qwen3 RoPE: HF half-rot (k, k+D/2), `rope_halfrot` (was interleaved, fixed).
- VAE: `(z/scaling)+shift`; SCALING=0.3611, SHIFT=0.1159; tile=64 ovl=8. VAE
  per-resnet sub-submits, single submit per tile (consecutive heavy submits hang
  the GPU). MiB-scale read_buffer probes in VAE crash the device; keep probes
  <= few KB and in-encoder.
- DiT diag dims 3840/11520/120/288 (derive, never hardcode; cross-check vs
  dispatch grids).
- TE stops at Qwen3 `hidden_states[-2]`; pads odd prompts to even seq. Qwen3 is
  the 1024 bottleneck (~14s vs ~1.5s/DiT-step); encoder-read win has no derived
  disk cache (case-by-case buy-in only).

## Perf: MINED OUT (2026-06-13)

Whole pipeline is COMPUTE-BOUND on every platform (warm step = GPU fence =
matmul/sdpa; uploads + weight-prep hide behind compute on the single queue).
Proven by pin-vs-no-pin A/B: forcing every weight to re-stream each step left
fence_ms flat while upload bytes jumped (desktop q4 768 4G: 1570 -> 3397
MiB/step, fence 3399 -> 3404). IO is NOT a lever. Mobile is a slow iGPU (VAE
5.4x desktop), rides every desktop sdpa/matmul win; only mobile curve-benders
are product-shaped (lower res, fewer steps).

Per-step gpu_disp baseline (q4 768 native, post-ORT-matmul): sdpa 1363ms, ffn1
1070, qkv 726, ffn2 657, proj 225, refiner ffn 119.

## Tested + rejected (do not retry on this shape class)

- sdpa f16-math dot+accumulate: -1.2%/step AND f16 o-accum compounds over keys
  (blocky grain at 768; pyref-256 misses it). NOT ALU-rate-bound.
- sdpa FA2-style tile softmax restructure: +20% native, web flat. (Op test kept:
  tests/sdpa_sg.rs.)
- sdpa ORT `prefer_subgroupshuffle` shape (reverted): correct (conformance
  green; q8-256 slope 1.0044) but perf WASH, confounded by global clock drift.
  Called mined out after three sdpa washes. ORT template at
  third-party/onnxruntime/.../bert/flash_attention.wgsl.template.
- matmul bk_step=2 (llama.cpp mmq): +8-12% (shmem vs occupancy).
- matmul 4-bit B end-to-end: +8% (inner shift+mask eats the traffic save).
  Confirms matmul is ALU/occupancy-bound, not B-traffic-bound.
- matmul subgroup loads (select() and shuffle): ~30% NVIDIA loss.
- BW/dispatch fusions on ALU-bound kernels (dual-matmul+silu_mul, QKV+RoPE,
  flash-attn+proj): all regressed.
- IO levers (all dead): read_at into mapped staging (WC collapse), host-RAM
  weight cache, host threads, disk caches, JS-heap pin tier, kernel read-ahead,
  raising VRAM budget as a "fix".
- Misc: i8 residual/elementwise; BC=64 sdpa; bm=128 tiles; LUT dequant;
  dedicated submit per upload; SUBMIT_DEPTH ring; static workspace reserve;
  weight prepack; derived on-disk transcode caches.
