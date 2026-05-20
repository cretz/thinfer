# Worklog

## bf16 M1 (landed)

- `WslConfig::BF16_PACKED` live for DiT block_pipelines. Storage 16-bit
  packed in u32 (`array<u32>`, `act_bf16_prelude!` unpack/pack), compute
  fp32, bit-clean parity vs pytorch on `step0.prev_sample` and
  `pre_vae_latent` (0/16384 over tol on iGPU).
- Qwen3 RoPE fixed to half-rot; DiT keeps interleaved.
- `MemAccount` tracks VRAM (Weights/Workspace/Staging) + RAM
  (Upload/Readback/Other) with global cross-category peak.
  `vram_total_peak()` is the only valid hard-ceiling assertion.
- `ResidencyBudget { vram_bytes, ram_bytes, workspace_reserve }`.
  `workspace_reserve` is a soft floor: weights evict to keep
  `weights + needed + max(reserve, non_weights_live) <= vram_bytes`.
- Phase-aware eviction: `evict_all_and_free` at phase boundaries;
  `Workspace::drain_pool` keeps size classes from leaking phase-to-phase.
- Categorized allocation via `Backend::allocate_in(bytes, cat)`;
  `RamCharge` / `VramCharge` RAII guards (held in completion futures).
- DiT loops: K=1 acquire(N+1) + K=2 prefetch(N+2) joined with submit(N)
  in `dit.noise_refiner`, `dit.context_refiner`, `dit.layers`. K=2 was
  marginal (~1-2s of 54s wall); main layers are already at 76% GPU util.
- e2e_parity asserts `vram_total_peak <= 2 GiB` and `ram_total_peak <= 2 GiB`
  with `workspace_reserve = 1 GiB`. VAE diag cap raised 64 -> 96 cells/256
  for iGPU per-stage variance.

State (iGPU, low power pref): **67 s wall total**. text_encode 14.2 s
(Qwen3 GPU util ~20% - massive idle but obsoleted by Qwen3 GGUF later),
diffusion 2 steps 47.4 s (DiT main layers ~76% util), VAE tile 5.5 s.

## Next: GGUF Q8_0 for DiT

Scope: **DiT only**. `unsloth/Z-Image-Turbo-GGUF` is DiT only (bf16 file
12.3 GB matches ~6B DiT params at 2 B/elem; no text encoder, no VAE).
Qwen3 and VAE keep their current safetensors path.

Why Q8_0 first vs jumping to Q4_K_M: simplest dequant-in-shader pattern
(32-elem blocks, one fp16 scale, sign-extended i8), ~lossless. Validates
the dequant kernel infrastructure before tackling Q4_K's super-block
hierarchy. Same kernel shape -> Q4_0 -> Q4_K is incremental.

Expected wall after Q8_0-on-DiT-only: **~50 s** (DiT 47 s -> 25-35 s;
text encoder + VAE unchanged). After Q4_K_M + shader-f16 + Qwen3 GGUF:
~15-20 s estimated, needs measurement.

Work order:

1. **GGUF header parser.** Add to `thinfer-core/src/weight/` alongside
   safetensors. Surfaces a `WeightCatalog` with per-tensor records
   carrying a quant tag (`Q8_0`, later `Q4_0`, `Q4_K`).
2. **`StorageEncoding::Q8_0` variant.** `read_for_gpu` passes Q8_0
   bytes through without transpose (GGUF stores matmul-friendly
   `[N, K]` blocks; kernel reads native).
3. **Matmul-Q8_0 WGSL kernel.** Dequant in the inner loop:
   `f = i8 * fp16_scale` (sign-extend i8 from u32 nibbles; fp16 scale
   bit-decoded to f32 if no shader-f16). `bk` tile multiple of 32
   (Q8_0 block size). f32 accumulator. f32 path first, f16 sibling
   when `Features::SHADER_F16` is requested and granted.
4. **Conformance fixture.** `matmul_q8_0` against pytorch with the
   same dequant function (port `dequantize_row_q8_0` from llama.cpp).
   Tolerance: bit-clean against the dequant ref.
5. **Wire DiT block matmuls** (`attn_to_q/k/v/out`, `ffn_w1/w2/w3`) to
   Q8_0 via WslConfig switch. Norms and biases stay full precision.
6. **Measure.** iGPU + discrete. Verify the ~50 s estimate before
   touching Q4 / shader-f16.

Open design points:

- Per-tensor quant surfacing: `quant: QuantKind` field on `WeightMeta`,
  dispatch site reads it to pick kernel variant. Single `WeightHandle`
  per tensor regardless of quant.
- shader-f16 feature negotiation at adapter init. Backend currently
  doesn't request optional features; add `Features::SHADER_F16` to the
  requested set, fall back when absent.
- Memory budget for Q8_0: ~2.6 GiB DiT residency. The current 2 GiB
  ceiling forces paging; either keep it (proves quant + eviction work
  together) or raise to test the all-resident wall. Probably do both.
- Q4_K dequant requires unpacking super-block 6-bit scale codes; design
  the kernel API so Q8_0 -> Q4_0 -> Q4_K is a swap of the unpack helper,
  not three separate kernels.

## Conventions

- Rope: `x [rows, heads, dim]`, `freqs [rows, dim]` interleaved.
  **DiT interleaved `rope`; Qwen3 half-rot `rope_halfrot`.**
- Sdpa: fused, `D <= MAX_D = 128`. Q `[B,S_q,H_q,D]`, K/V `[B,S_k,H_kv,D]`.
- Matmul: kernel is `a@b` with B `[K,N]`. PyTorch `nn.Linear` weights
  `[N,K]` - uploaded transposed at load. **GGUF Q8_0 stores `[N, K]`
  blocks natively in matmul-friendly layout; no transpose at load.**
- bcast ops single-batch.

## Carry-forward context (gotchas)

- **VRAM total = weights + workspace + staging.** `ResidencyBudget.
  vram_bytes` is the full ceiling. `workspace_reserve` is a soft floor
  the weights side respects; if non-weight live exceeds the reserve,
  the floor becomes the live value. See residency.rs.
- **Per-category peak SUM is NOT the true peak.** Use
  `MemAccount::vram_total_peak()` for hard assertions. `MemSnapshot`
  exposes both.
- **`Workspace::drain_pool()` at phase boundaries.** Without it, size
  classes from text_encode sit live during DiT.
- **Packed-bf16 act idiom.** WGSL: `array<u32>` storage,
  `act_bf16_prelude!` for unpack/pack. One thread = one word = 2 elems.
  Dispatch already word-counted. Channel-broadcast requires C even.
- **BF16_PACKED uses WeightDtype::Bf16.** Conformance bf16p fixture
  encodes everything as native bf16.
- **WebGPU dispatch caps at 65535 per dim.** Use `linear_workgroups`
  + 2D `gid` index pattern for elementwise/conv/upsample.
- **Per-binding cap.** Web baseline 128 MiB < DiT FFN 150 MiB -> M2
  needs chunked-matmul.
- **VAE applies `(z/scaling) + shift` internally.** SCALING=0.3611,
  SHIFT=0.1159.
- **VAE tiled path.** Default tile=64, overlap=8. One submit per tile
  or it TDRs. See [[feedback-no-multi-submit-vae]].
- **VAE diag rule.** <= few KB, in-encoder via
  `scope.read_buffer_via_encoder`. `STAGE_DIAG_MAX_BYTES`=1024.
- **Dtype.** M1: bf16 storage, fp32 compute. M2: Q8_0/Q4_K storage,
  fp32 compute first then shader-f16 sibling.
- **Submit must await before reuse.** `WgpuBackend::submit` does real
  await; never queue a second submit on the same workspace without it.
- **Z-Image text-encoder stops at Qwen3 `hidden_states[-2]`.** No
  `model.norm`/`lm_head`. `embed_tokens.weight` row-by-row via
  `text_encoder::embed_lookup` - never in residency.
- **Qwen3 head shape.** `n_heads*head_dim=4096`, `hidden=2560`.
- **DiT `decode_image`** passes raw latent `c` to `seq::unpatchify`.
- **SDPA mask** is `[B, S_q, S_k]` additive. Z-Image full-attn:
  `seq::attn_mask_zero_bytes(seq)`; Qwen3 causal:
  `seq::causal_mask_bytes(seq)`. Unmasked: `sdpa_mask_stub` +
  has_mask=0.
- **fp32 tolerance** is 1e-5 (silu's `exp` won't match libm
  bit-for-bit).
- **`cfg(test)` is per-crate.** Cross-crate test sharing needs feature
  flag / pub API / separate support crate.
- **Test gating.** `cargo test --workspace` skips uv/torch/wgpu.
  Conformance: `--features conformance`. Z-Image e2e: `--features
  zimage-e2e`.
- **Conformance test invocation:** one test
  `ops_match_pytorch_reference` iterates the registry; name filters
  do nothing. Use:
  `cargo test --release -p thinfer-conformance --features conformance
  ops_match_pytorch_reference -- --nocapture`
- **Parity tests use the engine's exact weights.** See
  [[feedback-parity-same-weights]].
- **WIP branch flow:** `git add ... && git commit --amend --no-edit
  && git push --force-with-lease`. No new commits on `initial-work`
  unless explicitly requested.
- **Matmul tile tuning (Intel iGPU).** Bigger register-blocked tiles
  win until the cliff at `tm*tn=32` acc regs/thread. Square beats
  fat-N at equal reg count. Hard cap: WG threads =
  `(bn/tn)*(bm/tm) <= 256` (WebGPU). Wall has ~1s variance per run;
  trust per-kernel GPU timestamps over wall. **The bf16p tuning won't
  transfer to Q8_0: dequant changes the bandwidth/ALU mix; re-sweep.**
- **`THINFER_TRACE=1`** enables tracing collection. Use
  `THINFER_TRACE=verbose` for fmt-layer DIAG prints. `[mem]` snapshot
  gated on the same env var.
- **iGPU is `THINFER_POWER_PREF=low`** (or unset = `None`). Discrete
  is `THINFER_POWER_PREF=high`. Wall numbers differ; iGPU is the
  primary target since perf wins there scale to web.

Memorized commands:

- Conformance: `cargo test --release -p thinfer-conformance --features
  conformance ops_match_pytorch_reference -- --nocapture`
- e2e_parity (iGPU): `THINFER_POWER_PREF=low THINFER_TRACE=1
  THINFER_E2E_PNG_DIR=<dir> cargo test --release -p thinfer-conformance
  --features zimage-e2e e2e_parity_matches_pytorch -- --nocapture`
- CLI smoke: `THINFER_TRACE=1 cargo run -p thinfer-cli --release --
  generate image --prompt "..." --output <path> --height 512 --width 512
  --ram-budget 4G --vram-budget 4G`

## Backlog

- **Sub-allocator** for residency pool: slab-based with free-list
  coalescing, so non-uniform sizes (LTX, quantized) reuse memory.
  Promoted in priority once Q8_0 lands (varied tensor sizes per quant).
- **Async-cancellation hole:** `submit_with_guards` via
  `on_submitted_work_done` callback. More pressing once prefetch
  futures are in flight at scope drop.
- **Q4_0 then Q4_K_M.** Q4_0 is a one-bit-unpack-helper change on the
  Q8_0 kernel. Q4_K is the production target (super-block layout, 8x
  sub-blocks of 32, 6-bit per-sub-block scale codes).
- **Q4_K_M file selection.** unsloth ships `_S` / `_M` / `_XL` variants
  (mixed per-tensor quant). `_M` is the recommended default.
- **shader-f16 fast path** for Q8_0 / Q4_K matmul. Gated by
  `Features::SHADER_F16` request at adapter init.
- **Qwen3 to GGUF.** Separate unsloth Qwen3-* GGUF. Need to confirm the
  variant matches Z-Image's expected architecture/tokenizer. ~10 s
  text_encode win on top of DiT Q8_0.
- **Bf16-clean VAE up_blocks.** Audit accumulation order or promote
  select stages to fp32. Today drift is ~4600 cells over tol on
  discrete, ~600 on iGPU; caps at 6000 (vae_rgb) and 96 (per-stage).
- **Workspace pool audit** (distinct-slot peak vs concurrent).
- **Dispatch overflow regression test:** n > 4M elementwise.
- **Conformance fixtures** for `conv2d`, `group_norm`,
  `upsample2d_nearest`.
- **Tiled VAE quality knobs** (overlap bump, halo-exchange per-layer).
- **Per-model gen defaults** via `ModelId::defaults()` when LTX lands.
- **CLI download per-decile progress.**
- **Tighten e2e parity tol** when bf16-clean VAE up_blocks lands.
- **Lift `ZImageRecipe` to engine-agnostic `ComputeRecipe`** when
  LTX/GGUF land.
- **`gelu`** deferred until Z-Image audit (exact-erf vs tanh-approx).
- **`ShardedSafetensorsSource` -> `UnionSafetensorsSource`** rename
  TBD.
- **e2e_parity NaN-loose assertion bug.** All-NaN currently reports
  `above_tol=0`; should fail loudly.
- **Dedupe `round_f32_to_bf16` / `act_upload_bytes`** between
  `seq.rs` (canonical) and `dit.rs` / `t_embedder.rs` (stale locals).
- **Raise e2e_parity VRAM budget** to test the all-resident wall once
  Q8_0 lands; keep the tight-budget run too for eviction coverage.
