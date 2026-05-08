# Worklog

## In progress

Matmul tile is `BlockMatmuls` with five `MatMulF32` instances
(qkv/proj/ffn_up/ffn_down share `64x64x16/4x4`; adaln stays DEFAULT
since M=1 makes register blocking pointless). Z-Image 256x256/2-step
wall ~56s.

Next: **bf16 activations as parametric `ActDtype` axis** (see Next #1
below). Biggest remaining lever; expected ~1.3-1.5x end-to-end.

Bf16 activations also unblocks revisiting `text_encode` (still on
untuned matmul path; ~14-15s, ~25% of wall) - bandwidth cut helps it
too.

## Next

1. **bf16 activations as parametric `ActDtype` axis.** Multi-session.
   Real cost is bigger than first-blush. Notes:
   - `bf16_quant_writes` (RNE rounding, f32 storage) and `act_dtype`
     (storage layout) are **orthogonal** axes. Packed-storage implies
     RNE; RNE-without-packing is a valid intermediate. `WgslConfig`
     gets both fields, not a replacement.
   - matmul A as `array<u32>` with `load_a` unpack is straightforward.
   - rmsnorm / silu_mul writing packed `array<u32>` requires geometry
     change: today one thread = one element, but a packed word holds
     two. Either one-thread-per-2-outputs (rewrite the workgroup
     geometry on every elementwise op that produces an activation) or
     `atomicCompareExchangeWeak` half-word stores (slow). Production
     stacks do the geometry change.
   - Workspace buffer sizing changes (half bytes); thread `ActDtype`
     through `block.rs` Bufs and the pipeline-cache keys.
   - Expected ~2x on matmul-heavy block; ~1.3-1.5x end-to-end (denoise
     dominates; VAE decode unchanged in this pass).
2. **Unify VRAM budget across Workspace + WeightResidency** (backlog
   l.203-). Today `vram_bytes` counts weights only; activations + pool
   are uncounted.
3. **Perf reference on real GPU** (torch-directml). Current pytorch-CPU
   baseline isn't a fair perf target. Numerical parity stays CPU bf16.

## Conventions

- Rope: `x [rows, heads, dim]`, `freqs [rows, dim]` interleaved.
  **DiT interleaved `rope`; Qwen3 half-rot `rope_halfrot`.**
- Sdpa: fused, `D <= MAX_D = 128`. Q `[B,S_q,H_q,D]`, K/V `[B,S_k,H_kv,D]`.
- Matmul: kernel is `a@b` with B `[K,N]`. PyTorch `nn.Linear` weights
  `[N,K]` — uploaded transposed at load.
- bcast ops single-batch.

## Carry-forward context (gotchas)

- **WebGPU dispatch caps at 65535 per dim.** Use `linear_workgroups` +
  2D `gid` index pattern for elementwise/conv/upsample.
- **Per-binding cap.** Web baseline 128 MiB < DiT FFN 150 MiB → M2 needs
  chunked-matmul.
- **VAE applies `(z/scaling) + shift` internally.** SCALING=0.3611,
  SHIFT=0.1159.
- **VAE tiled path.** Default tile=64, overlap=8. One submit per tile or
  it TDRs. See [[feedback-no-multi-submit-vae]].
- **VAE diag rule.** ≤ few KB, in-encoder via
  `scope.read_buffer_via_encoder`. `STAGE_DIAG_MAX_BYTES`=1024.
- **Dtype.** M1: bf16 storage, fp32 compute. Expand at GPU upload.
- **Submit must await before reuse.** `WgpuBackend::submit` does real
  await; never queue a second submit on the same workspace without it.
- **Z-Image text-encoder stops at Qwen3 `hidden_states[-2]`.** No
  `model.norm`/`lm_head`. `embed_tokens.weight` row-by-row via
  `text_encoder::embed_lookup` - never in residency.
- **Qwen3 head shape.** `n_heads*head_dim=4096`, `hidden=2560` (decoupled).
- **DiT `decode_image`** passes raw latent `c` to `seq::unpatchify`.
- **SDPA mask** is `[B, S_q, S_k]` additive. Z-Image full-attn:
  `seq::attn_mask_zero_bytes(seq)`; Qwen3 causal:
  `seq::causal_mask_bytes(seq)`. Unmasked: `sdpa_mask_stub` + has_mask=0.
- **fp32 tolerance** is 1e-5 (silu's `exp` won't match libm bit-for-bit).
- **`cfg(test)` is per-crate.** Cross-crate test sharing needs feature
  flag / pub API / separate support crate.
- **Test gating.** `cargo test --workspace` skips uv/torch/wgpu.
  Conformance: `--features conformance`. Z-Image e2e: `--features
  zimage-e2e`.
- **Conformance test invocation:** one test `ops_match_pytorch_reference`
  iterates the registry; name filters do nothing. Use:
  `cargo test --release -p thinfer-conformance --features conformance
  ops_match_pytorch_reference -- --nocapture`
- **Parity tests use the engine's exact weights.** See
  [[feedback-parity-same-weights]].
- **WIP branch flow:** `git add ... && git commit --amend --no-edit &&
  git push --force-with-lease`. No new commits on `initial-work`.
- **Matmul tile tuning (Intel iGPU).** Bigger register-blocked tiles
  win until the cliff at `tm*tn=32` acc regs/thread. Square beats fat-N
  at equal reg count. Hard cap: WG threads = `(bn/tn)*(bm/tm) <= 256`
  (WebGPU). Wall has ~1s variance per run; trust per-kernel GPU
  timestamps over wall. No cross-session wall comparisons.

## Backlog

- **Unify VRAM budget** across Workspace + WeightResidency. Today
  pool/workspace slabs uncounted; on 8 GiB card the 16 GiB e2e budget
  bypasses the LRU but workspace can still OOM.
- **Sub-allocator** for residency pool: slab-based with free-list
  coalescing, so non-uniform sizes (LTX, quantized) reuse memory.
- **Async-cancellation hole:** `submit_with_guards` via
  `on_submitted_work_done` callback.
- **Workspace pool audit** (distinct-slot peak vs concurrent).
- **Dispatch overflow regression test:** n > 4M elementwise.
- **Conformance fixtures** for `conv2d`, `group_norm`, `upsample2d_nearest`.
- **Tiled VAE quality knobs** (overlap bump, halo-exchange per-layer).
- **Per-model gen defaults** via `ModelId::defaults()` when LTX lands.
- **CLI download per-decile progress.**
- **Tighten e2e parity tol** when bf16 activations land. Today
  `vae_rgb` allows 500 cells past 3% tol; VAE diag stages allow 64.
  Drift source is bf16 through VAE up_blocks.
- **Lift `ZImageRecipe` to engine-agnostic `ComputeRecipe`** when
  LTX/GGUF land.
- **`gelu`** deferred until Z-Image audit (exact-erf vs tanh-approx).
- **`ShardedSafetensorsSource` → `UnionSafetensorsSource`** rename TBD.
