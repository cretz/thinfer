# Worklog

## In progress

**bf16 packed activations rollout — DiT clean, VAE drift remains.**
Root-cause of all-NaN: `dit.rs` was uploading RoPE freqs via
`rope.lookup_bytes` (fp32) into the packed-bf16 RoPE kernel, which
reinterprets each u32 word as two bf16 elems -> garbage rotations ->
NaN through SDPA. Fixed by routing freqs through
`seq::act_upload_bytes(pipelines.act_dtype, &rope.lookup(...))`. Also
fixed: x/cap/unified attn masks (`attn_mask_zero_bytes_act`) and the
hardcoded `freq_row = head_dim*4` in the unified-freqs concat copy.

Post-fix e2e_parity: `step0.prev_sample above_tol=0/16384`,
`pre_vae_latent above_tol=0/16384`. Only `vae_rgb` fails:
613/196608 cells over tol (max_abs=0.135, ref_max_abs=1.148). This
is the known backlog item ("Tighten e2e parity tol when bf16
activations land"; VAE up_blocks drift in bf16). DiT side is bit-clean
within tol — perf re-measurement now unblocked.

Done:
- `ActDtype { F32, Bf16 }` enum + `WgslConfig.act_dtype` field +
  `BF16_PACKED` const (pairs with `WeightDtype::Bf16`). `act_bf16_prelude!`
  macro emits shared `unpack_bf16x2` / `pack_bf16x2` / `round_bf16`. Three
  `WgslConfig { ... }` callsites updated (vae.rs / pipeline.rs / dit_scatter
  test).
- Conformance: `Dtype::Bf16Packed` ("bf16p"); native bf16 inputs+outputs in
  the fixture; `DTYPES_ACT_BF16 = [Fp32, Bf16Writes, Bf16Packed]`;
  `bytes_per_elem`; `diff_max_abs` handles 2-byte stride. Python
  `gen_from_spec.py` bf16-rounds inputs and recomputes for bf16p.
- Packed-bf16 WGSL variants + bit-exact conformance for: silu_mul, add,
  mul, silu, tanh, bcast_affine, bcast_fma, bcast_add (last has wf32 and
  wbf16 sub-variants for activation x weight cross-product), rmsnorm,
  layernorm, rope (interleaved + halfrot), sdpa (small-D only), matmul.
  Matmul requires `tn % 2 == 0` and `bn % 2 == 0` when act-packed; pairs
  output columns into one packed word per write (acc stays fp32). A
  read goes through `load_a` (mirrors `load_b`). Conformance registry
  bumped to tn=2 cfg so the bf16p variant has somewhere to land; tn=1
  cfgs (AdaLN matmul) stay on the fp32 / bf16q path by selecting
  `WgslConfig::FP32` / `BF16_QUANT_WRITES` at call sites. Norms
  keep one-thread-per-row geometry; inner reductions loop over `dim/2`
  words and unpack 2 elems each. Rmsnorm bf16-packed pairs with bf16
  weights; act-bf16 + wf32 panics (unused config). Rope halfrot uses
  per-thread = 1 packed word covering 2 consecutive pairs (real and
  imag halves are in different words). Sdpa-packed keeps the fp32
  accumulator `o`; only Q/K/V/Mask/Out storage is packed. Requires `D`
  even (8/128 in real models) and `s_k` even when `has_mask`. Large-D
  sdpa stays fp32 (VAE-only, unchanged this pass).

Per-thread idiom for packed-bf16: each thread emits one u32 word = 2 elems.
`linear_workgroups` count = `output.len/4` (already word count since SIZE=4).
Channel-broadcast ops require C even so a word stays within one row
(`c0 = (w*2) % C`, `c1 = c0+1` always).

## Next

1. **Bisect the all-NaN.** All driver wiring is done; the bug is in one
   of the ops or buffer layouts under packed-bf16. Approach:
   - Re-run e2e_parity with `THINFER_TRACE=verbose
     RUST_LOG=thinfer::diag=info` so engine `OURS-DUMP` lines print
     (RUST_LOG alone isn't enough - the fmt layer in `trace.rs` only
     attaches under `THINFER_TRACE=verbose|v|2`).
   - Or: add a one-shot readback after XEmbedder in `dit.rs::forward`
     (before scatter_pad_rows) using `seq::act_readback_to_f32`. If
     that's already NaN, the suspect set is { host-side `act_upload_bytes`
     byte order, XEmbedder matmul packed variant, bias add packed
     wbf16 variant }. If it's sane, walk forward op by op.
   - Conformance suite already validates each packed op standalone;
     suspect cross-op handoff, buffer-size mismatches, or a host
     upload that wasn't migrated.
2. **Once green: re-measure perf.** Current denoise wall is 17s with
   NaN output - meaningless until correctness is restored. Expected
   ~1.3-1.5x e2e per the foundation worklog.

## Bf16-packed driver state (landed this session)

- `WgslConfig::BF16_PACKED` is live in `z_image/pipeline.rs` for
  `block_pipelines` (DiT side). New sibling `encoder_block_pipelines`
  compiles with the prior `bf16_quant_writes=true, act=F32, w=Bf16`
  config so Qwen3 stays on the untuned matmul path (deferred per the
  prior worklog's "bf16 activations also unblocks revisiting
  text_encode" note - Qwen3 mask path would also need `s_k` even,
  which token_ids.len() doesn't guarantee).
- `dit.rs`: `act_bytes(...)` everywhere including `final_out`;
  `DitForwardLayout` carries `act_dtype`; `decode_image` unpacks bf16
  via match on `act_dtype`. Host uploads of `x_tok` / `cap_in` go
  through `seq::act_upload_bytes` (RNE bf16 when packed).
- `final_layer.rs`, `t_embedder.rs`, `embedders.rs`: all sized allocs
  routed through `pipelines.act_bytes(...)`. `t_embedder::compute_embed`
  emits bf16-packed when `pipelines.act_dtype == Bf16`.
- `text_encoder.rs::forward`: routed through the shared helpers but
  receives `encoder_block_pipelines` (fp32), so it functionally
  behaves as before.
- `seq.rs`: new pub helpers `act_upload_bytes`, `act_readback_to_f32`,
  `round_f32_to_bf16`, `attn_mask_zero_bytes_act`,
  `causal_mask_bytes_act`. `t_embedder.rs` / `dit.rs` still have local
  copies of `round_f32_to_bf16` / `act_upload_bytes` - dedupe later
  (cosmetic, not blocking).
- `pipeline.rs` readback: `row_bytes = oc * layout.act_dtype.bytes_per_elem()`.
- `ops/scatter_pad_rows.rs`: added packed WGSL variant (word-copy
  gated by row mask). `wgsl(cfg)` dispatches on `cfg.act_dtype`. The
  wrapper in `dit.rs::scatter_pad_rows` passes word-count
  `n_rows * dim/2` for the packed path; assertion on dst.len uses
  `act_dtype.bytes_per_elem()`.

## Gotchas surfaced

- The e2e_parity assertion treats all-NaN as "above_tol=0" -- this is
  a test bug worth filing but unrelated to the perf rollout.
- `THINFER_TRACE=1` enables tracing collection but NOT the fmt layer
  that prints DIAG-target events. Use `THINFER_TRACE=verbose`.

Conformance command (memorized):
`cargo test --release -p thinfer-conformance --features conformance
ops_match_pytorch_reference -- --nocapture`

Bf16 activations also unblocks revisiting `text_encode` (still on
untuned matmul path; ~14-15s, ~25% of wall) - bandwidth cut helps it
too.

## Follow-ups after parity is green

1. **Unify VRAM budget across Workspace + WeightResidency** (backlog
   l.203-). Today `vram_bytes` counts weights only; activations + pool
   are uncounted.
2. **Qwen3 text encoder on packed-bf16.** Deferred this session. Need
   `s_k` even for packed sdpa with `has_mask=1`, plus the untuned
   matmul tiles (`DEFAULT bm/bn/bk/tm/tn`) need a packed-bf16 sweep.
3. **Perf reference on real GPU** (torch-directml). Current pytorch-CPU
   baseline isn't a fair perf target. Numerical parity stays CPU bf16.
4. **Dedupe `round_f32_to_bf16` / `act_upload_bytes`** between `seq.rs`
   (canonical) and `dit.rs` / `t_embedder.rs` (stale locals).
5. **e2e_parity NaN-loose assertion is a bug.** All-NaN currently
   reports `above_tol=0`; should fail loudly.

## Conventions

- Rope: `x [rows, heads, dim]`, `freqs [rows, dim]` interleaved.
  **DiT interleaved `rope`; Qwen3 half-rot `rope_halfrot`.**
- Sdpa: fused, `D <= MAX_D = 128`. Q `[B,S_q,H_q,D]`, K/V `[B,S_k,H_kv,D]`.
- Matmul: kernel is `a@b` with B `[K,N]`. PyTorch `nn.Linear` weights
  `[N,K]` — uploaded transposed at load.
- bcast ops single-batch.

## Carry-forward context (gotchas)

- **Packed-bf16 act idiom.** WGSL: `array<u32>` storage, `act_bf16_prelude!`
  for unpack/pack. One thread = one word = 2 elems. Dispatch already
  word-counted (n_elems = output.len/4 = words for bf16 storage since
  SIZE=4). Channel-broadcast requires C even.
- **BF16_PACKED uses WeightDtype::Bf16.** The conformance bf16p fixture
  encodes everything (inputs, weights, outputs) as native bf16. Don't pair
  packed acts with fp32 weights in this config without an explicit
  WGSL_BF16_PACKED_WF32 variant.

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
