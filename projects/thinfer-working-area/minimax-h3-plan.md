# MiniMax-H3 port (33B joint audio-video)

ACTIVE. Branch `more-models`, DO NOT COMMIT.

## Scope

- Round 1: **T2VA only** (text -> video + audio). FL2VA checkpoint with zero
  keyframes. This is exactly what the lightx2v turbo LoRA was distilled for
  (`dmd_minimax_h3_t2va4`).
- Deferred: FL2VA keyframes (I2V / first+last frame), Ref2VA, 2K regenerate.
- CFG-distilled: no unconditional pass, no guidance scale, one forward per step.

## Weights

| part | repo / file | size |
| --- | --- | --- |
| DiT | `Abiray/MiniMax-H3-GGUF` `unet/MiniMax-H3-FL2VA-Q4_K_M.gguf` | 18.5 GiB |
| DiT canary | same repo `-Q8_0.gguf` | 33.6 GiB |
| encoder | `Abiray/...` `text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | 13.6 GiB |
| encoder alt | `nif0/...-L0-49-{IQ4_XS,UD-IQ3_XXS}` (pre-truncated) | 12.5 / 9.1 GiB |
| video VAE | `Comfy-Org/MiniMax-H3` `vae/minimax_h3_video_vae_fp16` | 4.85 GiB |
| audio VAE | same `vae/minimax_h3_audio_vae_fp32` | 0.56 GiB |
| turbo LoRA | `lightx2v/Minimax-h3-Turbo` `minimax_h3_fl2v_turbo_4step_v0.1` | 1.29 GiB |

- Turbo LoRA is a **preview** (lightx2v says so). Keep it swappable: fallback is
  `larryvrh/MiniMax-H3-Turbo-Lora` `minimax_h3_turbo_4step_ema_ckpt850` (744 MiB,
  259 pairs, native `blocks.*` naming, includes AdaLN pairs we would ignore).
  Do NOT use `drbaph/*` (AdaLN-stripped, needs 6-10 steps, no upside for us).
- Do NOT use the Comfy-Org `*_pruned_*` DiT. We precompute AdaLN exactly from the
  full checkpoint, which is strictly better than the curve-form approximation.

## Architecture (from `third-party/minimax-h3-mlx`, validated vs diffusers)

- 50 blocks, `hidden 5376`, 56 heads x 128 head_dim = **7168 inner** (wider than
  the residual stream), FFN 14336, `text_dim 5120`, `time_embed_dim 2688`.
- **Single stream.** One packed 1-D sequence holds text + audio + video rows.
  Full self-attention over the whole thing. NO cross-attention, no per-modality
  block weights. Modality only shows up in: the input patch projections, the
  per-row AdaLN tag, and the two output heads.
- Uniform block: `x += gate_msa * attn(mod(norm1(x)))`, `x += gate_mlp * mlp(mod(norm2(x)))`.
- Attention: per-head q/k RMSNorm, `scale = head_dim^-0.5`.
- Token refiner: 2 plain pre-norm blocks over the projected text, no AdaLN, no rope.
- Final layer: shared RMSNorm + modulation, then TWO heads over every row
  (`video_out` -> 96 = 24*1*2*2, `audio_out` -> 32); rows selected after.

### Checkpoint layout quirks (handle at read, do not rewrite weights)

- `attn.qkv_proj` rows are **per-head interleaved**: reshape to `(.., heads, 3, head_dim)`
  and index q/k/v on axis -2. NOT a contiguous [q|k|v] split.
- `mlp.fc1` is a fused `[gate; value]` SwiGLU: `fc2(silu(gate) * value)`.

### RoPE

- **Half-rot** (matches Qwen3/LTX, opposite of Wan/Hunyuan interleaved-pair).
- 3 axes share one 16-freq `inv_freq`; concat 3 axes -> 48, concat with itself
  -> 96 rotated channels; the remaining 32 of each 128-dim head pass through.
- Position ids are **float** `(t, h, w)`, not integers.

## AdaLN precompute (the big win)

- `blocks.{i}.adaln_proj.linear` is `[96768, 2688]` x 50 = **~13B of the 33B**.
- Its only input is the timestep embedding. It does not see the sequence.
- So for a fixed schedule, precompute every modulation tensor once at load and
  drop the projections. `96768 = 6 params x 3 modalities x 5376`.
- Table size: `50 blocks x 6 x (T timesteps x 3 modalities) x 5376`. For a 4-step
  run T is ~5 distinct levels -> ~77 MB. Against ~7 GiB of Q4 weights.
- Row addressing: `adaln_indices = timestep_index * 3 + max(tag, 0)`.
- **Precision: build the table from the f32 timestep embedding through the
  UNQUANTIZED projection.** Every block reads the same temb, so error here biases
  all 50 blocks identically at every step and accumulates coherently. Read the
  Q4 blocks, dequant to f32, project, discard. One pass, ~7 GiB of reads.
- `final_layer.adaln_proj` is only `[10752, 2688]`; cache it too (cheap).
- Keep `TimestepEmbedder` in f32 for the same reason.

## Packed sequence

Row order: `[ text (L) | keyframe conditions (C) | target audio (A) | target video (V) ]`

- Emit **no padding**. Upstream pads to a multiple of 64 and masks; padless needs
  no attention mask at all, which keeps the fast unmasked sdpa path.
- Build the position grid in **f64** on the host, cast to f32 at the end. Video and
  audio share one 40-units-per-second rotary clock and that shared clock IS the
  a/v alignment. f32 accumulation drifts.
- Text rows sit on the time axis at their row index, so text length shifts the
  whole media clock. Media time starts at `num_text`.
- Video temporal spacing: `5/3 * (1,4,4,4,4)` cycling, cumulative.
- Spatial axes: aspect-normalized, `linspace(left, left+ratio, dim//patch,
  endpoint=False) * 32`, `ratio = dim / sqrt(latent_h * latent_w)`.
- Audio rows are channel-major, carry no height coord, pinned to the two extremes
  of the width grid, time = `num_text + arange(num_audio_latents)`.
- Tags: video 0, text 1, audio 2, pad -1. A keyframe's vision block is tagged
  **video**, not text (only its `<Picture i>: ` label stays text).

## Geometry

- 24 fps fixed. Short edge 768, area cap 768*1344, axes rounded to multiple of 32.
- Frames snap up to `17n + 5`; latent frames = `5n + 2`. Spatial compression 16.
- Audio: 40 latents/s, stereo as 2 channel-major blocks, 32 kHz out.
- 16:9 -> 1344x768 -> latent 84x48 -> patch (1,2,2) -> 1008 rows/frame.
  5s = 124 frames = 37 latent frames = **37,296 video rows** + 414 audio + text.
- Reduced canvas for iteration: override h/w (multiples of 32). Off-distribution
  but exercises the whole path at ~1/7 the sequence.

## Schedules

- Rectified-flow Euler, `eta = 0`, exponential shift `s*sig / (1 + (s-1)*sig)`
  over `linspace(1, 0, N)`. Collapse consecutive f32 duplicates (the shift
  compresses near sigma=1 and duplicate collapse changes the step count).
- **Two schedules, one forward**: video shift **12.0**, audio shift **3.0**.
- `timesteps = 1 - sigmas[:-1]`; N sigmas drive N-1 evaluations.
- Velocity is **data-ward**: `x0 = x_t + (1 - t) * v`. Note the `+`, opposite of
  the usual flow-match sign. Then `x_next = r*x_t + (1-r)*x0`, `r = sig_next/sig`.
- Conditioning rows are re-imposed by construction (only generated rows written
  back), never by masking.

## Text encoder

- Qwen3-VL-32B, **layers 0..49 of 64 only**, hidden state taken **before** the
  final norm. `lm_head` and layers 50-63 never evaluated. Vision tower skipped
  entirely for T2VA.
- Presentation: **no chat template, no special tokens** (`add_special_tokens=false`),
  prompt verbatim. Same shape as the LTX finding. Do not add BOS.
- Interleaved M-RoPE with `mrope_section` split (Qwen3-VL native).

## Engine work (LANDED)

Three new ops, each used by BOTH the DiT and the video VAE decoder:

- **`RopeF32HalfRotPartial`** (`core/ops/rope.rs`). Rotates the leading
  `rot_pairs*2` channels, copies the rest through. Pairing is `(k, k+rot_pairs)`
  WITHIN the rotary block, so plain `RopeF32HalfRot` (which pairs `k, k+dim/2`)
  silently mixes a rotated channel with a pass-through one. `rot_pairs` rides in
  the uniform's 4th slot; freqs row width IS the rotary width. **Conformance
  green** fp32/bf16w/bf16p incl. the H3 96-of-128 ratio and a full-width case
  that must reduce to plain half-rot.
- **`QkvSplitHeadwiseF32`** (`core/ops/qkv_split.rs`). Per-head interleaved
  `(rows, heads, 3, head_dim)` split. **NO CONFORMANCE YET.**
- **`SiluMulFusedF32`** (`core/ops/silu_mul_fused.rs`). SwiGLU reading a fused
  `[gate; value]` row in place (the halves are a STRIDED per-row slice, so they
  cannot be two buffer views). **NO CONFORMANCE YET.**
- `BatchBuf::sub(offset, len)` added to `core/workspace.rs` for per-segment
  binding.

**No gather kernel is needed.** Rows sharing an AdaLN table row are contiguous by
construction, so modulation is per-segment `bcast_modulate` / `bcast_fma` against
offset table slices. Every offset is 256B-aligned (5376*4 = 84*256). t2va = 3
segments. `ModulationPlan::segments` + tests pin this.

Reused as-is: `common::block` matmul/sdpa/rmsnorm, `bcast_modulate` (bias=1 folds
the `1+scale`), `bcast_fma` (the gate residual AND the VAE's per-channel
`scale1`/`scale2`), `bcast_add` (matmul biases), weightless `layernorm` +
`bcast_modulate` for the VAE's affine `norm_out`.

## Video VAE: it is a ViT, not a CNN

Decoder only (t2va never encodes; the CNN encoder is keyframe-only, deferred).

- **36-layer ViT at dim 2048**, 32 heads x 64. Every latent voxel is one token;
  `proj_out` expands each token straight to `4*16*16*3 = 3072` values, doing all
  16x spatial + 4x temporal upsampling at once. No conv decoder anywhere.
- Same three quirks as the DiT: `attn.to_qkv` per-head interleaved (WITH bias),
  `ff.w1` fused `[gate; up]`, rope rotates `rope_dim_ratio` 0.75 -> 48 of 64.
- Deltas from the DiT block: every projection has a bias; q/k RMSNorm is
  **unaffine** (no weight in the checkpoint, computed f32); each sublayer output
  is scaled by a learned per-channel `scale1`/`scale2` before the residual.
- 4 learned register tokens + 1 `mask_token` appended, all at position 0, dropped
  before `proj_out`. **`mask_token` is all-zero in the shipped file (verified)**
  but read it rather than assuming.
- rope positions are cell centres on `[-1,1)` scaled by `2*pi`, theta 100.
- **Tiling bounds the sequence**: 256px tiles / 64 overlap -> 16x16 latent x 5
  latent frames + 5 suffix = ~1285 tokens per pass, never the clip's 149k.
- `latents_mean` / `latents_std` ship inside the VAE file.

## Audio VAE: it is a BigVGAN we already have

`ltx/vocoder.rs` is the same family and its `RESBLOCK_KERNELS [3,7,11]` /
`RESBLOCK_DILATIONS [1,3,5]` are IDENTICAL to H3's. conv1d / conv_transpose1d /
snake_beta / replicate_pad1d are already conformance-green. So this is
parameterizing the existing vocoder, not a new implementation.

- H3 decoder: rates 5,5,2,2,2,2,2; kernels 9,9,4,4,4,4,4; dim 1024; 32 latent
  channels; 32 kHz; hop 800 = 40 latents/s.
- Two load details: weight norm is stored as `weight_g`/`weight_v` and must be
  folded to `g*v/||v||` at load; the anti-alias Kaiser-sinc filters are stored as
  buffers but are recomputable.
- Mono model: the two stereo channels run as two batch items.

## Checkpoint naming (verified against the shipped GGUF, 2026-08-07)

DiT GGUF keeps the ORIGINAL checkpoint names (`general.architecture` is
mislabelled `wan` by the comfy converter; ignore it):
`blocks.N.{norm1,norm2,attn.{qkv_proj,q_norm,k_norm,out_proj},mlp.{fc1,fc2},
adaln_proj.linear}`, `token_refiner.blocks.N.*`, `token_refiner.final_norm`,
`{video,audio}_patch_proj`, `condition_proj`, `time_embedder.proj_{in,out}`,
`final_layer.{norm,adaln_proj.linear,video_out,audio_out}`, and `rope.inv_freq
[16]` (use it to verify our computed freqs). Q4_K for the big matmuls, F16 for
the small projections, F32 norms/biases -> per-site dequant, not one scheme.

Encoder GGUF uses HF names `model.layers.N.*` and is ALREADY truncated to 50
layers with no `lm_head` and no final norm.

## Checkpoint traps (verified against the shipped GGUF header, 2026-08-07)

The comfy converter RESHAPES any tensor whose row length is not a multiple of
the 256-element K-quant superblock, recording the truth only in a
`comfy.gguf.orig_shape.*` KV that our GGUF reader does not parse:

| tensor | on disk (catalog, dims reversed) | actually |
| --- | --- | --- |
| `blocks.N.adaln_proj.linear.weight` | `[1016064, 256]` | `[96768, 2688]` |
| `video_patch_proj.weight` | `[2016, 256]` | `[5376, 96]` |

`2688` is 10.5 superblocks, which is why. Element ORDER survives, so reads are
fine, but `TransposePolicy::Linear2D` reads the SHAPE: left alone it transposes
`video_patch_proj` as a `[2016, 256]` matrix and scrambles every video row with
no NaN to catch it. `loader::register_video_patch_proj` restores the real shape.
Any future tensor added here must be checked the same way.

Also: `token_refiner.*` is F16 (unquantized), main blocks Q4_K, norms/biases F32.

## Memory ceiling: the native canvas does not fit

Activations, not weights. At 1344x768 the packed sequence is ~37.8k rows and one
block holds `rows x 2 * ffn_hidden x 4` = 4.3 GiB for the fused SwiGLU alone,
plus 3.25 GiB of qkv. Streaming weights cannot help. `pipeline::MAX_PACKED_ROWS`
(6144) refuses the canvas up front.

FIX (not started): chunk the row-independent sites over rows -- both patch
projections, the whole feed-forward, both output heads. ONLY attention needs the
full sequence resident. Until then H3 runs at a reduced canvas, which is
off-distribution and a quality caveat, not a shipping config.

## State (2026-08-07) and next steps

Green: `cargo fmt` + `clippy -D warnings` on thinfer-models + thinfer-core, 83
`minimax` unit tests, op conformance.

Written: `mod, manifest, packing, rope3d, scheduler, adaln, text_encoder, dit,
video_vae` (the last is config/geometry/handles/CPU helpers only).

NOT written, in order:
1. Video VAE decoder GPU forward. **Its weights are already on disk, so this is
   independently validatable without waiting for the DiT.**
2. Audio VAE (parameterize `ltx/vocoder.rs`).
3. `loader.rs` + the AdaLN precompute pass (load-time GPU matmul: read the Q4
   `adaln_proj`, dequant, project the f32 temb, store the table, drop the
   weights).
4. `pipeline.rs` denoise loop + turbo LoRA fold. The lightx2v LoRA is
   diffusers/PEFT-named (`transformer_blocks.N.attn.to_q.lora_A.default.weight`,
   rank 128, 624 tensors, attn+ff only, NO AdaLN pairs) and must be mapped onto
   the checkpoint's FUSED `qkv_proj` / `fc1`: to_q/to_k/to_v have to be
   re-interleaved per head, and gate/up concatenated. That mapping is the
   fiddliest remaining piece.
5. app/CLI/serve/web wiring (CLI/web parity rule).
6. Conformance for the 2 unvalidated ops; DiT + VAE parity gates; e2e.

**Download blocker.** DiT `unet/MiniMax-H3-FL2VA-Q4_K_M.gguf` (18.5G) was still
streaming at session end. Resume/check with:
```
HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1 hf download Abiray/MiniMax-H3-GGUF \
  unet/MiniMax-H3-FL2VA-Q4_K_M.gguf
```
(`hf_transfer` is installed in `~/.hf-cli/venv`; xet stays disabled per the known
hang. Resumes from the `.incomplete` blob.) Encoder + both VAEs are COMPLETE.

## Budgets / risk

- 8 GB VRAM ceiling holds; DiT streams. Steady-state streamed weights ~11.5 GiB
  per step after the AdaLN drop.
- Compute is the real risk: ~20B active params over ~37.8k rows is ~1.5 PFLOP per
  step at full 768p. Expect minutes per step at full canvas. Validate at reduced
  canvas first; measure before optimizing.
- Attention is 37.8k^2 x 56 heads full self-attention. This is the hot spot and
  the first place to look after correctness.
- HARD: no heavy runs without the user's explicit OK per batch.
