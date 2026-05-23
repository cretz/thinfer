# Worklog

## Current state

Live regime: **DP4A int8 matmul** when adapter exposes
`Packed4x8IntegerDotProduct` + `SHADER_F16`; F16-workspace path is the
fallback. Per matmul site the layer forward runs
act_quant -> dequant_i8 -> matmul_i8: A is quantized to packed i8 +
per-(M, K/32) f32 scale; B is dequanted from K-quants / Q8_0 / Q4_0 to
the same packed i8 + scale layout; the inner loop is
`sum_s dot4I8Packed(av, bv)` -> i32 scaled by `a_scale * b_scale` at
every K=32 sub-block.

matmul_i8 now also does:
- **Loop reorder** (`av` hoisted outside the `j` loop, per-(j) dots
  accumulate into a register array, scaled once after `s`). Always-on,
  numerically identical to pre-reorder; cuts shared-mem reads from
  `2*TM*TN*BK_U32` to `TM*BK_U32 + TM*TN*BK_U32`.
- **Subgroup-aware `tile_a` reads** when adapter exposes
  `Features::SUBGROUP`. Universal kernel branches on the runtime
  `subgroup_size` builtin: `subgroupBroadcastFirst` when
  `subgroup_size <= wg_x`, `subgroupShuffle(v, sg_id - lid_x)` when
  `subgroup_size > wg_x`. Correct for any size; on devices with fixed
  size the compiler folds the branch. Workgroup is declared
  `@workgroup_size(THREADS, 1, 1)` because naga 28 rejects
  `subgroup_invocation_id` on multi-dim WGs; `lid_x`, `lid_y` derived
  from `local_invocation_index`.

e2e_parity_for_gguf_q4_k_m VERIFIED PASSING.

Measurements (Q4_K_M, 2 diffusion steps, 256x256):

- **Intel Arc 140T iGPU, `LowPower`**: diffusion_steps 33.7 s ->
  16.86 s/step. Per-block sample ~952 ms. Subgroup size [8, 32]
  (variable). Subgroup + reorder vs prior F16-workspace baseline:
  `-5% diffusion_steps`, `-11% per-block`.
- **NVIDIA RTX 5070 Laptop, `HighPerformance`**: diffusion_steps
  13.54 s -> **6.77 s/step**. Subgroup size 32 (fixed). Per-block
  sample 324 ms.
- `text_encode` unchanged (bf16 safetensors path, not on matmul_i8).

`PowerPreference::None` maps to `LowPower` behavior on Vulkan
(driver treats unset as background-priority hint). CLI + e2e_parity
defaults flipped to `HighPerformance`; explicit `THINFER_POWER_PREF=low`
to exercise thin-hardware path.

Adapter logging surfaces caps at startup whenever
`THINFER_TRACE` is set (any value): adapter name, backend,
shader_f16, packed_int_dot, subgroups, subgroup_size range. Plus a
per-block-build line with the actual `matmul_i8_use_subgroup` flag so
test runs can confirm the optimization path is active.

## Next attack: cooperative matrix (subgroup matrix)

The DP4A inner loop is now memory-coherent (loop reorder) and
subgroup-broadcast-clean. Further wins from the same shape are likely
sub-2%. The next structural step up is **cooperative matrix** ops:
WebGPU's `chromium-experimental-subgroup-matrix` extension exposes
`subgroupMatrixMultiplyAccumulate` (NVIDIA WMMA, Intel XMX, AMD WMMA
underneath) — 4x4xK matrix MACs per instruction vs DP4A's 1x4 dot
product, ~4-8x throughput on hardware that supports it.

Status: behind a Chrome flag today; wgpu doesn't yet surface it
either. When it lands, the dispatch is the new matmul_i8 inner loop
(replace the `s,j` register-blocked DP4A loop with one `coopMatMul`
call per K=32 sub-block). Geometry will need to match the matrix
shape the extension exposes (probably 16x16x16 or 8x8x32 — TBD).

Until then, secondary attacks (all measure-or-die):
1. **Async submit / pipelined finish**. Rollup occasionally shows
   `submit_ms` outliers (~880 ms) while next dispatch sits idle. If we
   can issue the next submit before the previous finishes, sustained
   throughput improves.
2. **Q4_K_M for Qwen3 text encoder**. Encoder is 14 s of ~17 s wall on
   the discrete card (worse share than on iGPU). Same dequant_i8 +
   matmul_i8 path applies; need a Q4_K_M Qwen3 GGUF (find or build).
3. **Q8_0 fast path in `dequant_i8`**. Q8_0 is already i8+scale on disk;
   skip the f32 round-trip. Low priority — dequant cost is small.

## What NOT to do (tested + rejected)

- **Weight prepack `(b_i8, b_scale)` per matmul site.** -0.3% noise;
  6.56 GiB prepacked footprint blew the 2 GiB VRAM ceiling. matmul_i8
  dominates ~89% of block.x — eliminating the dequant_i8 dispatch saves
  nothing.
- **Dual-matmul + silu_mul fusion (FFN super-kernel).** Regressed Q8
  +5.7%, Q4_K_M +3.5%. Doubled tile_b shared-mem + doubled per-thread
  accumulators drops occupancy on Intel iGPU; dispatch saving is sub-1%.
- **QKV+RoPE fused, flash-attn + proj fused.** Same class as above.
- **Larger M tiles (bm=128).** `tm*tn=32` is the register cliff on
  Intel iGPU; 64x128/4x8 doubled FFN ms pre-dequant-once. Re-measure
  under DP4A if you retry; do not extrapolate.
- **LUT dequant.** Per-thread LUT init overwhelms per-elem saves; ~2x
  worse without `state_cache`. Skip.
- **One-shot tile/bk sweeps** without a structural change. DP4A pins
  BK=32 anyway.
- **Further dtype-narrowing on f16-path.** Inner loop is i32-ALU under
  DP4A now; not f32/f16.
- **`HighPerformance` as the only knob.** Picks the discrete GPU when
  present; iGPU is still the thin-hardware target and gets only modest
  wins from the inner-loop work above. Structural wins (coopmat,
  async submit) are device-portable; "use the bigger card" isn't.

## Locked design decisions (carry forward)

- **DP4A matmul (auto-opt-in).** Gated on
  `WgslLanguageFeatures::Packed4x8IntegerDotProduct + SHADER_F16 +
  ActDtype::F16`. Three WGSL kernels: `ops/act_quant.rs`,
  `ops/dequant_i8.rs`, `ops/matmul_i8.rs`. matmul_i8 default tile
  `bm=bn=64, tm=tn=4`. Output paired `vec2<f16>`.
- **`MatMulI8Config.use_subgroup`** auto-on when
  `Features::SUBGROUP` is exposed; runtime-branched shader covers all
  subgroup sizes. Layout change: flat `@workgroup_size(THREADS, 1, 1)`.
- **Dequant-once per matmul site (Quant only).** F16-workspace fallback
  path. Workspace dtype follows `act_dtype`.
- **`WeightDtype::F16`.** Paired `array<vec2<f16>>`; only valid for
  dequant-once workspaces.
- **`MatMulConfig.b_nmajor`.** N-major B-load on bf16-workspace path.
- **K-quant `bk` and `bs` divisor-aligned.** `bk%bs==0 || bs%bk==0`.
- **Quant DiT acts pinned to `ActDtype::F16`** when SHADER_F16
  available, else F32.
- **Three pipeline-set split.** `block_pipelines`,
  `dit_encoder_block_pipelines`, `encoder_block_pipelines`. Refiners
  /embedders never quantized.
- **GGUF parser is range-fetch-first.**
- **B viewed `[N, K]` N-major in matmul.** GGUF native layout.
- **Saturated-narrow at f16 store sites.** Clamp +-65504 before f16 cast.
- **Residency pool counts against VRAM ceiling.**
- **Matmul WGSL fits in 32 KiB workgroup storage.** Hard `assert!`.
- **Pipeline validation errors surface through `PipelineCreate`.**
  `WgpuBackend::create_shader_module` wraps `push_error_scope` /
  `pop_error_scope`.
- **`PowerPreference::HighPerformance` is the CLI/e2e default**;
  `None` aliases to `HighPerformance`. Explicit `low` for thin-hw runs.

## Conformance coverage for DP4A ops

`thinfer-conformance/tests/{act_quant_i8,dequant_i8,matmul_i8}.rs`:
- Per-op pipeline-build tests (catch WGSL parse/validate regressions).
- Numerical round-trip vs scalar Rust reference.
- matmul_i8 has 6 cases: small / multi-block-K / default-tile
  x (use_subgroup={false, true}). The subgroup variants skip cleanly
  when adapter lacks `Features::SUBGROUP`; `try_run` prints adapter
  caps so you can verify which path ran.

## Q4_K_M file encoding map (unsloth z-image-turbo-Q4_K_M.gguf)

- ggml types: 12=Q4_K, 13=Q5_K, 14=Q6_K, 30=bf16
- attn.qkv.weight: 2 Q6_K + 28 Q5_K (special = first+last)
- attn.out.weight: 2 Q5_K + 28 Q4_K
- ffn.w1.weight / ffn.w3.weight: 2 Q5_K + 28 Q4_K
- ffn.w2.weight (FFN-down): 30 Q6_K
- adaLN_modulation.0.weight: filtered by `QuantOnlySource`, bf16 wins

## Conventions

- Rope: DiT interleaved `rope`; Qwen3 half-rot `rope_halfrot`.
- Sdpa: fused, `D <= MAX_D = 128`.
- Matmul kernel: `a @ b` with B `[K, N]`. PyTorch `nn.Linear [N, K]`
  uploaded transposed; **GGUF `[N, K]` natively, no transpose at load.**
- Z-Image text-encoder stops at Qwen3 `hidden_states[-2]`.
- DiT `decode_image` passes raw latent `c` to `seq::unpatchify`.
- SDPA mask `[B, S_q, S_k]` additive.
- VAE: `(z/scaling) + shift` internal; SCALING=0.3611, SHIFT=0.1159.
  Tiled tile=64, overlap=8, one submit per tile.
- fp32 tol 1e-5. `cfg(test)` per-crate.
- WIP branch flow: `git add ... && git commit --amend --no-edit &&
  git push --force-with-lease`. No new commits on `initial-work` unless
  asked.
- WebGPU dispatch caps at 65535/dim.
- `THINFER_TRACE=1` enables tracing rollup + fmt-layer stderr output.
  `THINFER_TRACE=verbose` also adds span-close events.
- `Workspace::drain_pool()` at phase boundaries.

## Perf backlog (deferred)

- Fit DiT fully resident, lower workspace_reserve from `/4` to fixed
  ~512 MiB.
- Sub-allocator for residency pool (slab-based, free-list).
- Bf16-clean VAE up_blocks.
- Per-model gen defaults via `ModelId::defaults()`.
- Lift `ZImageRecipe` to engine-agnostic `ComputeRecipe`.
- e2e_parity NaN-loose assertion bug.
- Tighten e2e parity tol when bf16-clean VAE up_blocks lands.
- VAE tiled quality knobs (overlap bump, halo-exchange per-layer).
- CLI download per-decile progress.
- Q8_0 dequant_i8 near-memcpy fast path.

## Memorized commands

- Conformance: `cargo test --release -p thinfer-conformance --features
  conformance ops_match_pytorch_reference -- --nocapture`
- DP4A op conformance:
  `cargo test --release -p thinfer-conformance --features conformance
   --test act_quant_i8 --test dequant_i8 --test matmul_i8`
- e2e_parity (defaults to HighPerformance now; add `THINFER_POWER_PREF=low`
  to force thin-hw path):
  `THINFER_TRACE=verbose THINFER_E2E_PNG_DIR=<dir> cargo test --release
   -p thinfer-conformance --features zimage-e2e e2e_parity_for_ --
   --nocapture --test-threads=1`
