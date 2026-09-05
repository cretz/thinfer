# Worklog

Forward-looking only. Git history is the changelog, the code is the record.
Engine-wide design: `plan-details.md`. Per-model plans are separate files (see
Status).

## In progress -- branch `more-models` (off main; DO NOT COMMIT)

### ACTIVE: MiniMax-H3 port (33B joint AV). See `minimax-h3-plan.md`.

**H3 RUNS END-TO-END (2026-08-07).** `thinfer generate video --model minimax-h3`
writes an MP4 with an AAC track. Product path: CLI -> `app/minimax.rs` ->
encoder -> packed denoise -> ViT video VAE + BigVGAN audio VAE -> H.264+AAC.
384x256 / 22f / 4 steps = **214s** on the 5070. NOT eyeball-confirmed yet.

**THE BIG BUG FOUND (fixed): the DiT read every weight with the WRONG LAYOUT.**
`dit_cfgs` passed `WeightDtype::Bf16` for the matmul config. That flag does NOT
say what the kernel reads -- it selects the B-operand CONVENTION: `Quant(_)` =>
`b_nmajor=true` ("the site pre-dequants to a dense N-major workspace"), anything
else => K-major. H3 pre-dequants per kind (mixed Q4_K/Q6_K) but declared Bf16,
so the matmul transposed every weight. Silent: no NaN, plausible velocity rms.
LTX gets this right (`mm.weight_dtype = Quant(Q8_0)`); H3 now passes
`Quant(Q4_K)`. **Any new model on `common::block` must set this deliberately.**

**LANDED THIS SESSION (fmt + clippy -D warnings clean, 274 model + 40 app tests):**
- `app/minimax.rs` driver + `VideoModelId::MinimaxH3` + `is_minimax()` + resolve
  (`resolve_minimax`: /32 dims, 17n+5 frames, fixed 24 fps) + required_files +
  default dims 512x320 / 56f. Serve rides the shared enum; web UI wired
  (`minimax-h3` in MODELS.video, own /32 preset ladder, Audio toggle, steps/vae/
  upscale/encoder rows hidden). **serve.exe NOT rebuilt** (a running serve locks
  it; stop serve then `cargo build -p thinfer-serve --release`).
- `common/merge.rs` NEW: `MergedSource` unions two weight sources under one
  residency (base + adapter), so the adapter shares one VRAM budget.
- **Turbo LoRA applied at RUNTIME, not folded.** `common::lora`'s fold
  re-encodes Q4_K -> Q8_0, which would take streaming from ~10.8 GiB/step to
  ~20.5 GiB AND park the same again in the unbounded host fold cache (the
  wan22 pathology). Runtime form = +~5% MACs, no requant, footprint-neutral.
  `dit::BlockLoraHandles` + `DitHandles::without_lora()` for the A/B.
- **bf16 acts + `fast_sdpa`** (was f32): DiT SDPA **26s -> 1.0s** via `sdpa_sg`
  (6.6ms/disp vs 228ms). Also halves activation bytes.
- `PixelBlock::to_signed_planar()` (ImageNet undo + channels-last -> planar);
  `pipeline::{denoise, video_rows_to_latents, audio_rows_to_latents}`.
- AAC now accepts 32 kHz (`codec::freq_index_for`) -- H3's rate.
- Conformance `tests/minimax/dit_health.rs` NEW: real Q4_K DiT forward, checks
  finite/plausible, asserts the turbo adapter reaches the output (A/B vs
  `without_lora`), and measures EVERY coopmat site against the dense reference.

**COOPMAT: plumbed, measured, DELIBERATELY OFF.** `build_coopmat_step` (per
quant kind, since `_K_M` is mixed per tensor) + an f32-OUTPUT variant
(`CoopmatOut::F32` + `cast_f32_to_bf16`) because `ffn_up`'s 28672-wide
pre-SwiGLU output overflows f16 -> inf -> NaN (observed: whole decode
non-finite). It roughly halves DiT matmul (96s -> 48s), but `dit_health`
measures **~9% divergence from dense on every site independently** (common-mode,
from the f16 A-side cast). No parity reference exists for H3, so there is
nothing to say which side is right -> `dit::COOPMAT_SITES` all false. Turn on
only after a pyref DiT parity gate. `ffn_down` can never use it: its A-side IS
the SwiGLU output that overflows.

**PERF VERDICT (2026-08-07): H3 IS A BAD FIT FOR THIS CARD. Likely park it.**
Measured 214s for 384x256 / 22f / 4 steps (759 rows). Extrapolates to ~15-20 min
for a 5s clip at 384x256; 5s at 512x320 does NOT fit the 6144-row cap at all.
Against that, shipped `ltx2-rapid` does 5s at 512x320 in **226s** and is ALSO
joint audio-video -- ~5x faster at higher resolution for the same capability.
Best case with every remaining lever (coopmat ~2x on matmul, row-chunking for
canvas) is ~8-12 min at 384x256, still worse than rapid. The floor is
architectural, not tuning: 33B dense on 8GB, and the ONE packed self-attention
sequence makes resolution and clip length trade against each other against a
hard cap. Do not sink more perf work here without a reason to prefer H3's
quality specifically -- and that quality is unverified (see below).

**STILL BROKEN: THE OUTPUT IS NOISE.** User-confirmed, and self-confirmed by
extracting a frame (`ffmpeg -i out.mp4 -vf select=eq(n\,10) -vframes 1 f.png`,
then just READ the png -- fast loop, no round-trip). It is STRUCTURED noise:
per-latent-voxel tiles, i.e. the VAE decodes each token plausibly but the
LATENTS are noise. So the denoise is not denoising.

RULED OUT (do not re-check):
- `QkvSplitHeadwiseF32` + `SiluMulFusedF32`: **conformance-green now** (fp32 and
  bf16-packed exact to 0.0 vs pyref). NB `qkv_split_headwise` is registered
  WITHOUT `bf16w` -- it is a pure data move with no quant-writes variant, so
  that dtype differs by exactly bf16 epsilon and means nothing.
- `RopeF32HalfRotPartial`: already green.
- packing/unpatchify: the row content order is `(c, pt, ph, pw)`, matching the
  reference permute `(0, 2, 4, 6, 1, 3, 5, 7)` verbatim.
- Two REAL bugs found and fixed, neither of which was sufficient:
  1. the B-operand layout (above);
  2. **the step count was off by one**: `Scheduler::new(N)` builds an `N`-sigma
     GRID driving `N - 1` evaluations, so `--steps 4` ran 3 steps against a
     4-step distillation. `DenoisePlan::build` now passes `steps + 1`; pinned by
     `requested_steps_are_model_evaluations`.

STILL SUSPECT: the conditioner output, the AdaLN table application, the block
wiring itself. A pyref DiT parity gate is the way to settle it -- guessing has
now cost several 200s round trips.

**NEXT (ordered):**
1. **Build the pyref DiT parity gate.** It adjudicates the noise bug AND the
   coopmat question, and would have caught the layout bug automatically.
2. **DiT parity vs pyref.** Blocks the coopmat decision AND is the only thing
   that would have caught the layout bug automatically.
3. Conformance for `QkvSplitHeadwiseF32` / `SiluMulFusedF32` (still none).
4. Row-chunk the row-independent sites: raises `MAX_PACKED_ROWS` (6144) toward
   the native 1344x768 canvas AND cuts the block activation peak.
5. Perf after that: `matmul_qkv` is 100s of the 214s run (2384 disp; ~2/3 of
   those dispatches are the LoRA's two extra matmuls per site -- an
   accumulate-into-dst matmul would remove a full-width buffer + an add per
   site). Video VAE `sdpa` is 19.6s at f32 acts (deliberate: its output IS the
   picture; bf16 there is a quality call, not a free win).

User-directed 2026-08-07. Target: t2va end-to-end in the serve web UI, lightx2v
4-step turbo LoRA. NOT finished this session; below is exactly where it stands.

**LANDED (all fmt + clippy -D warnings clean; 102 minimax unit tests green):**
- `models/src/minimax/{mod,manifest,packing,rope3d,scheduler,adaln,text_encoder,
  dit,video_vae,audio_vae,loader,pipeline}.rs`.
- **3 new engine ops**, all needed by BOTH the DiT and the video VAE decoder:
  - `RopeF32HalfRotPartial` (rotate leading N channels, pass the rest through).
    CONFORMANCE GREEN vs pyref across fp32/bf16w/bf16p, incl. the H3 96-of-128
    ratio and a full-width case that must reduce to plain half-rot.
  - `QkvSplitHeadwiseF32` (per-head interleaved `(rows, heads, 3, hd)` split).
    **NO CONFORMANCE YET** -- see the open task; silent-garbage failure mode.
  - `SiluMulFusedF32` (SwiGLU over a fused `[gate; value]` row). **NO
    CONFORMANCE YET** -- same risk.
  - `BatchBuf::sub(offset, len)` added to thinfer-core for per-segment binding.
- **`common/bigvgan.rs` NEW**: the BigVGAN v2 tower extracted out of
  `ltx/vocoder.rs` (pure move; LTX now builds a `BigVganCfg` and keeps only its
  mel/STFT/Hann glue). H3's audio VAE is the SAME family with identical
  `RESBLOCK_KERNELS [3,7,11]` / `DILATIONS [1,3,5]`, so `minimax/audio_vae.rs` is
  ~180 lines of config. NB `VocoderPipelines::compile` now takes a label arg.
  **`ltx vocoder_parity` NOT re-run since the move -- run it before trusting LTX.**
- Video VAE decoder GPU forward + host tiling/chunking/stitch DONE.
- `loader.rs`: host-side AdaLN precompute (streams Q4 rows, f32-exact).

**CHECKPOINT FACTS FOUND THIS SESSION (verified against the shipped GGUF header):**
- **The comfy converter RESHAPES tensors whose row length is not a multiple of
  256** (the K-quant superblock). `blocks.N.adaln_proj.linear.weight` ships as
  `[1016064, 256]` not `[96768, 2688]`; `video_patch_proj.weight` ships as
  `[2016, 256]` not `[5376, 96]`. True shape is ONLY in the
  `comfy.gguf.orig_shape.*` KVs, which our GGUF reader does not parse. Element
  order survives, so reads are fine, but `Linear2D` reads the SHAPE ->
  `loader::register_video_patch_proj` restores it. Missing this scrambles every
  video row with no NaN.
- `token_refiner.*` is F16 (unquantized); main blocks are Q4_K; norms/biases F32.
- Encoder role REPOINTED to the Abiray file already on disk
  (`text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf`), which is the truncated
  50-layer build. Saves a redundant 12.5G nif0 download.
- Reference tiling/blending did NOT match what was in `video_vae.rs`: upstream
  `_split_tiles` distributes slack round-robin in `ratio` steps (not evenly
  spread starts) and `_blend` ramps `i/extent` (not `(i+1)/(extent+1)`). Both
  corrected; wrong versions shift every seam.

**HARD LIMIT FOUND: the native 1344x768 canvas does NOT fit 8GB.** The packed
sequence is ~37.8k rows and the fused SwiGLU intermediate alone is
`rows x 28672 x 4` = 4.3 GiB, before qkv (3.25 GiB). No amount of weight
streaming helps: these are activations. `pipeline::check_sequence_budget` refuses
>6144 rows up front rather than device-losing mid-denoise. **Fix (not started):
chunk the row-independent sites over rows** (both patch projections, the whole
feed-forward, both output heads); only attention needs the full sequence
resident. Until then H3 runs at a REDUCED canvas only (off-distribution).

**KEY DESIGN FINDINGS (do not re-derive):**
- **AdaLN is precomputable.** `blocks.N.adaln_proj.linear` is `[96768, 2688]` x50
  = ~13B of the 33B and its ONLY input is the timestep embedding. Build the whole
  modulation table once at load (~48 MB at 4 steps), then never stream those
  weights: streamed DiT drops 18.5G -> ~11.5G. Build it from the f32 temb through
  the DEQUANTIZED projection (every block reads the same temb, so error here
  biases all 50 blocks coherently).
- **No gather kernel needed.** Rows sharing an AdaLN table row are CONTIGUOUS by
  construction (layout is `[text | cond | audio | video]`), so modulation is
  per-segment `bcast_modulate`/`bcast_fma` against offset table slices. t2va = 3
  segments. Test pins this.
- **The video VAE decoder is a 36-layer ViT at dim 2048, NOT a CNN.** Each latent
  voxel is one token; one output projection does all 16x/4x upsampling. Shares
  all three new ops. Decode is tiled at 256px/64 overlap so one pass is ~1285
  tokens, not the clip's 149k.
- **Encoder is Qwen3-VL-32B truncated to layers 0..49 of 64**, unnormalized
  hidden state, no `lm_head`, no final norm. Abiray's GGUF is ALREADY truncated
  (verified: 50 layers, HF `model.layers.N.*` naming). For TEXT-ONLY requests
  Qwen3-VL M-RoPE reduces exactly to 1-axis rope (all 3 position axes equal), so
  no mrope machinery. NO chat template, NO special tokens, no BOS.
- Checkpoint quirks: `qkv_proj` per-head interleaved; `mlp.fc1` fused
  `[gate; value]`; velocity is DATA-ward (`x0 = x_t + (1-t)*v`, note the `+`).
- GGUF ships `rope.inv_freq [16]` -- verify our computed freqs against it.

**GPU-VALIDATED (real checkpoints, RTX 5070 8GB).** New conformance target
`tests/minimax/` behind feature `minimax-e2e` (skips cleanly when uncached):
- `video_vae_health` **PASS, 21.4s**. Full ViT decode path against the real fp16
  checkpoint: embed -> 36 blocks -> norm_out -> proj_out -> unpatchify -> stitch,
  7 latent frames / 1 tile -> 256x256 x 28 frames. Range **[-1.989, 2.530], mean
  0.149** = exactly the ImageNet-normalized band the decoder should emit, so this
  is stronger than a NaN check. Exercises ALL THREE new H3 kernels against real
  weights (incl. the two with no conformance gate).
- `audio_vae_health` **PASS, 1.3s**. 32000 samples = 2ch x 20 frames x 800 hop,
  range [-0.877, 0.820], in-clamp. Also validates the `common/bigvgan`
  extraction against a SECOND checkpoint (nested `ups.N.0`, interleaved
  `activations.N`, `dec_in_proj`).
- `adaln_precompute` **PASS, 29.5s** against the real 18.5 GiB Q4_K DiT (host,
  no GPU). 50 blocks x 3 timesteps, |max| modulation **16.9** (plausible),
  all finite, varies with timestep. **58.2 MB table replaces ~7 GB of
  `adaln_proj`.** This also proves the superblock-reshape handling is right: a
  misread projection would give garbage magnitudes or non-finite values.
  29.5s is one-time at load and is dominated by reading ~7 GB of Q4.

**DiT GGUF download COMPLETE** (19,864,208,217 bytes). All six runtime roles are
now on disk.

**BUG THE HEALTH TEST CAUGHT (fixed):** in-place `bcast_add` for the bias
(`out = out + bias`) is a wgpu usage conflict -- one buffer cannot be both the
read and the read_write operand of a single dispatch. Fixed in
`video_vae::mm_bias` and the `proj_out` tail by writing to a separate
destination. **`pipeline.rs` and any new bias site must do the same.**

**NOT DONE (ordered). The DiT ITSELF HAS NOT RUN -- everything below is
build+unit-test green only.**
1. lightx2v turbo LoRA fold. Diffusers/PEFT naming
   (`transformer_blocks.N.attn.to_q.lora_A.default.weight`, rank 128, 624
   tensors, attn+ff only, no AdaLN pairs) onto the checkpoint's FUSED
   `qkv_proj` / `fc1`: to_q/k/v must be RE-INTERLEAVED PER HEAD and gate/up
   concatenated. Fiddliest remaining piece. File downloaded.
2. app/CLI/serve/web wiring (CLI/web parity rule applies).
3. First GPU bring-up at a reduced canvas, then browser e2e.
4. Conformance for the 2 unvalidated ops; DiT + VAE parity gates.
5. Row-chunking so the native canvas fits (see the HARD LIMIT above).
6. Re-run `ltx vocoder_parity` after the bigvgan extraction.

**Download: DONE except the DiT** (`unet/MiniMax-H3-FL2VA-Q4_K_M.gguf`, 18.5G)
which was at 17.5G and finishing. Encoder + both VAEs + tokenizer + turbo LoRA
all on disk. Resume with:
`HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1 hf download Abiray/MiniMax-H3-GGUF unet/MiniMax-H3-FL2VA-Q4_K_M.gguf`

### HOT: LTX bad PROMPT ADHERENCE on BOTH rapid AND ltx-2.3-distilled (t2v)
User-reported (2026-07-21): coherent video, but ignores the prompt -- on BOTH
ltx2-rapid (19B) AND ltx-2.3-distilled (22B) t2v. Distilled has strong adherence
in the ecosystem, so a SHARED component is broken. Rapid/distilled differ in FE
(V1 vs V2) + connector, so the shared suspects are: **Gemma-3-12B encoder +
tokenizer + prompt-formatting path**. User's own read: "our ltx prompting or text
encoder is broken." (Do NOT chase the ltx2-rapid portrait-prior weakness -- this
is distilled too, a real bug. IGNORE `scratch/fullgreed*` logs: abandoned expt.)
- **RULED OUT: missing system prompt / chat template.** Upstream ENCODE path
  (`third-party/LTX-2/.../text_encoders/gemma/{tokenizer.py,encoders/base_encoder.py}`)
  uses NEITHER: `encode()` -> `tokenize_with_weights` = `text.strip()` +
  `tokenizer(text, add_special_tokens)` (leading BOS), no template. System prompts
  feed ONLY the enhancer (`enhance_t2v/i2v`), not conditioning. Matches our
  `tokenizer.encode(prompt, true)` in `app/ltx.rs`. So NOT the bug.
- **PRIME SUSPECT: the product tokenize->encoder->FE->connector path is NEVER
  parity-validated.** The encoder-parity gate feeds DUMPED ids built by a
  DEGENERATE pyref tokenizer (`gen_tokens_ref.py` reconstructs from the GGUF: NO
  BOS, NO merges -- see `ltx/manifest.rs:27-35`). So the gate proves only
  "encoder forward matches pyref given degenerate ids", NOT "real product ids ->
  right conditioning". Exactly where a silent adherence bug hides.
- **Upstream specifics to check ours against:** `LTXVGemmaTokenizer` uses
  `max_length=1024`, `truncation=True`, `padding_side=LEFT`, pad=eos; encoder
  returns ALL 49 hidden_states. OUR `text_encoder.rs` runs REAL-TOKENS-ONLY
  (positions 0..n-1, causal mask) and CLAIMS (doc lines 20-23) that is bit-
  identical to the 1024 left-pad "since the 1024 framing + registers happen in
  the connector stage". Plausible for the hidden states (left-pad + mask -> real
  tokens at pos 0..n-1), BUT UNVERIFIED that the FE/connector aggregation over n
  real tokens matches upstream's masked aggregate over 1024.
- **NEXT diagnostic steps (ordered, all RAM-safe / no heavy run):**
  1. Compare product tokenization: our Rust `tokenizers` output vs HF
     `AutoTokenizer(unsloth/gemma-3-12b-it/tokenizer.json)` on sample prompts --
     diff ids (BOS present? merges applied? count? truncation). Tokenizer-only
     Python, no model/GPU. The manifest CLAIMS this matches; verify it.
  2. Read `ltx/connector.rs` `feature_extractor_v2_flatten` / `_v1_flatten` vs
     upstream `feature_extractor.py` masked aggregation -- confirm the n-real-token
     aggregate == upstream's masked-over-1024 aggregate (mask handling, which
     layers, order).
  3. Weak-vs-absent: adherence-bad (coherent but off-prompt) => conditioning is
     WEAK/MISALIGNED not absent; check connector/caption-projection output
     magnitude + how cross-attn KV is consumed in the DiT.
- Key files: `app/ltx.rs` (tokenize), `models/ltx/text_encoder.rs` (encoder),
  `models/ltx/connector.rs` (FE + connector), `conformance/tests/ltx/
  encoder_parity.rs` + `python/.../ltx/gen_tokens_ref.py` (degenerate tokenizer).
  Upstream: `third-party/LTX-2/.../text_encoders/gemma/{tokenizer,feature_extractor,
  encoders/base_encoder}.py`.
- Serve was rebuilt (release) + launched this session then STOPPED (not running
  now). Config = `scratch/serve.toml` (`tls_self_signed=true`); launch:
  `target/release/thinfer-serve.exe --config <that>` -> https :8080. Exe is
  current with this session's edits EXCEPT any code changed after the rebuild
  (rebuild before the next browser test; stop serve first -- Windows exe lock).


### DONE this session (build/clippy green; parity + browser validated). Uncommitted.
- **rapid audio pitch/duration fixed** (`ltx/rapid_vocoder.rs`): `OUTPUT_SR`
  16000 -> **24000**. `prod(UP_RATES)=240` (not 120), mel 100/s -> 24kHz; the mistag
  stretched playback 1.5x (low pitch + long container). Browser-confirmed.
- **rapid perf, quality-neutral** (`ltx/dit.rs`): attn out-proj + ffn_down -> i8 on
  the UNGATED rapid variant (new `I8Site::{Proj,FfnDown}`, `attention.proj_i8`,
  `feed_forward.down_i8`, gated `!gated`, VIDEO stream only; 22B stays dense).
  `rapid_dit_parity` GREEN 0.99%/0.42% (gate 1.2%). Audio-stream i8 REVERTED (blew
  gate 1.57% -- audio residual i8-sensitive even ungated).
- **rapid VRAM budget 2G -> 6G** (`app/ltx.rs` `RAPID_VRAM_BUDGET_CAP`): single-stage
  rapid, so the block LRU keeps ~40% of the 14GB DiT resident across the 8 steps vs
  re-streaming every step. Net (with i8): denoise ~226->~170s.
- **USER MEASURED 343s -> 226s** for a 512x320 5s clip (good video, correct voice).
- Duration placeholder now resolution-aware (`web/app.js`); hid `*-q4` LTX variants
  from the web dropdown (busted; CLI-only). Serve rebuilt/redeployed then KILLED.

### NEXT (open, ordered)
- **SPEECH gibberish = DISTILLATION artifact, CONFIRMED (not a bug).** Ran all three
  gates green (2026-07-21): `audio_vae_parity` rel 0.000%, `vocoder_parity` rel
  0.006%, `connector_parity` video 0.011% / audio 0.016%. Decode/vocode chain +
  22B conditioning tail all bit-clean vs pyref on same weights. Gibberish common to
  BOTH variants + base LTX-2 does dialogue => few-step distillation. (Caveat:
  connector gate covers the 22B 8-layer GATED connector; rapid's 2-layer ungated
  connector is a separate untested path, but shared VAE+vocoder clean + identical
  symptom on both => conclusion holds.) FIX PATH (not a bug hunt): (1) DONE opt-in
  higher-step "audio quality" toggle -- LANDED (build/clippy green, sampler unit
  tests green; UNVALIDATED in browser, needs eyeball). `sampler::refine_sigmas`
  subdivides the coarse distilled sigma jumps into finer Euler substeps (distilled
  anchors kept exactly; cap `AUDIO_HQ_MAX_DSIGMA=0.1` -> 8->16 steps, ~2x denoise).
  Wired single-stage path only (rapid always; 22B when !upscale): `req.audio_hq`
  through CLI `--audio-hq` + serve `audioHq` + web `#audio-hq` (rapid-only row);
  `app/ltx.rs` single-stage branch builds the schedule + `sink.note`s the step bump.
  NEXT for (1): browser eyeball -- does the finer schedule actually improve speech
  intelligibility, and is 0.1/2x the right tier (try 0.075 for finer if weak)?
  (2) R&D cheap AUDIO-ONLY refine steps after the fast joint pass (freeze video +
  cache its a2v/v2a K/V, skip video self/FFN; audio ~5% of work). If (1) helps but
  is too slow at 2x whole-pass, (2) gets most of the gain for ~5% of the cost.
  fast+intelligible+text-only+8GB is at the frontier.
- **mobilewan STILL HUNG / TOO SLOW.** Drain fix landed (`wan/pipeline.rs`
  `drain_pool` per step gated on `cfg.rehyat`) handled CROSS-STEP accumulation.
  Root cause of the remaining hang = PER-BLOCK peak: each block is one
  `scratch.batch()`/`submit_deferred` (`wan/dit.rs:620`), so the F32 island's conv
  casts (w1 h*pdim*hd, w2 h*pdim*pdim ~14MB), c1/g1/c2/r2, s_state, phi_q/phi_k AND
  the whole cross_ffn tail all coexist until the single block submit. FIX LANDED
  (build+clippy green, UNVALIDATED on GPU): split into two submits. `wan/dit_block.rs`
  old `self_attn_mobilewan`/`forward_mobilewan` REPLACED by `mobilewan_phi` (norm+mod+
  qkv+qk-norm+RoPE+F32-island phi maps -> 5 persistent survivors), `mobilewan_attn`
  (chunk loop + out-proj), `mobilewan_attn_post` (attn+gate+cross-ffn). New DiT
  orchestrator `wan/dit.rs::forward_block_mobilewan` allocs persistent qfs/kf/vf/
  phi_q/phi_k + 6 mod bufs, runs phase1 in one `batch()`+`submit_void` (frees
  w1/w2/c1/g1/c2/r2), phase2 in a fresh batch; joined with next_acquire/prefetch in
  the block loop (mirrors the tiled path). Peak now max(phase1, phase2) not sum.
  Byte-identical (verbatim persistent-buffer boundaries; ref = `rapid_vocoder::
  decode`). NEXT: GPU-validate the hang cleared (heavy run -- needs explicit OK).
  ALSO separately too slow (dense `matmul_ffn_down`; scalar `rehyat_attn`) -- perf
  after the hang is confirmed cleared.
  - **RUNS AT 2G/2G, VALIDATED** (832x480/29f/6-step uni-pc): the arbiter streams
    weights at 2G (no pin), fits (peak_live 2073 MiB). The 6G OOM was the fill-to-
    ceiling pathology (weights pinned near budget starved the F32-island phi alloc);
    2G forces streaming and fits. `scratch/mobilewan_val2.mp4`.
  - **PERF: 443.9s -> 274.8s (38% faster)** via two exact-parity kernel rewrites in
    `thinfer-core/src/ops/rehyat.rs` (unit tests green, err ~1e-7): (1) `grouped_matmul`
    naive one-thread-per-output -> 16x16 tiled per-head (wid.z=head) w/ shared-mem
    reuse: 118.2s -> 14.1s (8.4x); dispatch now takes (rows,h,npg) for the 3D grid.
    (2) `rehyat_attn` scalar HD=128 loops -> vec4 128-bit loads + vec4 ALU (q/k/v/
    s_state/out bound as vec4; phiq/z scalar): 140.2s -> 66.6s (2.1x). At 2G compute
    ~= denoise wall (streaming hidden), so compute wins pass through to wall.
  - **REMAINING PERF LEVERS (bigger/uncertain, NOT started):** (a) `matmul_ffn_down`
    66s dense -- coopmat is `None` (mobilewan is bf16 safetensors; coopmat only wired
    for Quant->f16, block.rs:819). Needs a bf16->f16 coopmat dequant OR Q8_0
    block_transcode of the DiT (latter ALSO ~halves the 72GiB/6-step streaming). Both
    quality-affecting, no mobilewan parity gate -> needs eyeball. (b) `rehyat_attn`
    round-2 (66s, occupancy-bound: 256 regs/thread from acc+rnum vec4 arrays) --
    flash-style workgroup-cooperative rewrite of the windowed softmax. (c) qkv_self
    29s / proj 16s dense.
- **ltx-distilled** two-stage 1280x704 eyeball still pending (upscaler-only parity).

### Numbers (rapid 512x320/121f trace, 6G): DiT GPU ~120s/8steps -- qkv DENSE 31.5s
(audio, gate-locked), `sdpa_sg` 29.7s, i8 sites ~49s; streaming ~50s. Structural
remainder (don't quality-cut by default): encoder cold ~61s, rewrite ~22s, VAE ~33s.
Opt-in web levers already exist: `#encoder` Q4 (~-25s), `#rewrite` off (~-22s).

## Status (shipped -- DO NOT DISTURB)

- **LTX-2 19B (Phr00t rapid merge)** -- distilled + native I2V. Checkpoint variant
  on the LTX-2.3 pipeline. Net-new: FE V1 text head + caption projection, 2-layer
  ungated 3840 connector, 6-way block modulation, native I2V (frame-0 latent replace
  + denoise-mask blend), reconstructed LTX-2 VAEs, LTX rewriter (concise-EN
  Qwen3-VL). KNOWN: the merge is human/portrait-specialized -> weak compositional
  T2V (single subjects work; multi-subject falls to the human prior). I2V (frame-0
  anchor from a good image model) is the intended fix, not a conditioning lever.
  `ltx2-rapid-plan.md`.
- **LTX-2.3 distilled + Sulphur-2** (22B joint AV) -- `ltx-plan.md`. Sulphur
  VERIFIED end to end 2026-09-05 (video + audio, eyeballed): 768x512 5s/121f =
  **503s / 509s** with coopmat (was 708s / 744s dense, -30%); 3s/73f = 334s (was
  486s). Total GPU 288s -> 202s on the 3s clip. Weights confirmed current (the Jun-10/13
  base-repo commits touched only `prompt_enhancer_uncensored/`, not the DiT; local
  Q8_0 + distill LoRA match remote sizes/oid). Distill stack = the ecosystem
  default (dev + `...lora-1.1_fro90_ceil72_condsafe` @ 1.0).
- **HunyuanVideo 1.5 T2V** (lightx2v 4-step, 480p) -- shipped.
- **hunyuan-video-1.5-ti2v** (causal AR, minWM dmd) -- `hy15-causal-plan.md`.
  Text-only AR drift at 480p is a regime effect (OOD for the i2v ckpt); I2V anchor
  damps it.
- **Wan2.2-T2V-A14B** -- shipped; attn-window default W=3. Device-loss + perf fixes
  landed (transient-reserve double-count in residency acquire; coopmat ragged-M OOB
  staging under-alloc -> pad to next multiple of WM; cross-block fence carry).
  Background prefold of the LoRA fold cache hides ~40-50s/expert behind encode. 6G
  VRAM budget works (832x480 f33 ~596s). HARD RAM: no wan22/anyflow auto-runs
  (23-30GB host fold cache at any res; batch runs need explicit OK).
- **AnyFlow-Wan2.1-T2V-14B** -- shipped; default 2 steps (quality-neutral); hybrid
  W=3 default; tiny-VAE available; vault LoRA fold end-to-end (community + PEFT key
  conventions). NSCLv1 (noncommercial).
- **FastWan2.2-TI2V-5B** -- Q8 block_transcode; UniPC default. The video CLI default.
- **LongLive-2.0-5B** (AR long/multi-shot) -- shipped. OPEN BUG: `longlive_parity`
  vel_c0s0 FAILS pre-existing at HEAD (gate stale vs weeks of kernel changes);
  root-cause = next LongLive session's first item. Re-measure WARM at 576.
- **Krea 2 Turbo** -- shipped, coopmat+fast_sdpa default (512x512x8 ~99s).
- **DreamID-V-Wan-1.3B-Faster** (diffusion video face-swap) -- ported, parity +
  eyeball PASS, app/serve/web wired w/ encrypted RAM-only video upload + live DWPose
  mask. `dreamidv-plan.md`. Frozen at ~5s/480p.
- **Face-swap** (HyperSwap ONNX) -- shipped. `faceswap-plan.md`. XSeg occlusion mask,
  GFPGAN enhancer, HyperSwap own-mask, strided detection, bitrate + trim (all opt-in,
  CLI+serve+web wired). Perf: CPU/GPU 3-stage pipelining + frame batching B=4 ->
  ~88ms/frame at 1920x1242 (was 207ms serial). Decode robustness: multi-param-set
  annexb + openh264 error concealment.
- **Ideogram-4**, **Z-Image**, **Qwen-Image(-Edit)-Rapid** -- shipped.
  Qwen-Image-Rapid is NOW the image CLI default.
- **User LoRA vault** -- CLI/serve/web (Argon2id + AES-256-GCM, per-entry
  multi-password, model-scoped). Fold wired for Krea2 + Qwen-Image(-Edit) + AnyFlow.
  Pending real-adapter eyeball.
- **i8 DP4A matmul ON by default** (`--no-i8-matmul` = bf16 reference).

## Lessons / dead-ends (do not retry)

- **PAIR every perf run on this box.** Unpaired back-to-back runs swing 10%+ on
  page-cache state alone (the same 3s config measured 485s, 461s, 518s across the
  session with changes that turned out to be neutral). Any A/B from unpaired runs
  is noise. Interleave the conditions and report the spread.
- **A LoRA fold belongs on the background prefold worker.** `spawn_prefold` existed
  but only Wan called it; LTX/Sulphur folded lazily on the streaming path, putting
  ~46s of CPU on the critical path. Building the DiT source BEFORE the encoder and
  prefolding cut Sulphur pre-denoise 102s -> 59s. The fold is byte-identical either
  way: same seed gave an md5-identical MP4 (PSNR inf, MSE 0 on all 73 frames).
  Check any new folded model does the same.

- **HyperSwap conv is at the wgpu/naga ceiling** (memory/reuse-bound, not
  ALU-bound). Every kernel-level lever measured flat-or-worse: tiling, i8, bf16,
  32x32-occupancy tile, vec4 128-bit loads, Winograd F(2,2,3,3), double-buffered
  shared loads. The wins were SYSTEM-level: CPU/GPU pipelining + frame batching
  (B=1 forward is ~13x over compute-ideal because each dispatch underfills the GPU).
  Remaining levers are all system-level (batch detection -- BLOCKED by an executor
  batch>1 correctness bug in SCRFD; f16 acts; codec), none conv-kernel.
- **Coopmat DOES apply to a native f16-act DiT, and is a precision GAIN there.**
  The gate was `fast_mixed` (a bf16-acts concept: it exists to build the bf16<->f16
  casts), which locked out exactly the blocks that need no casts at all. On the
  LTX DiT (`F16Dp4a`) the A-side, residual and coopmat operand are already the same
  f16 values, so the "cast" is just the WM row padding and the output cast vanishes.
  Result: `matmul_qkv` 11.33 -> 5.48 ms/disp, 3s clip 486s -> 334s, and vs pyref
  coopmat is CLOSER than the dense path it replaced (video rel 0.088% vs 0.123%,
  f32 accumulator). H3's ~9% divergence was the f32->f16 A-side cast, not coopmat.
- **A parity gate at tiny dims cannot see coopmat.** `dit_full_parity` runs 4
  video / 4 audio tokens; coopmat needs `M >= WM` (32) and silently falls back to
  dense below it, so that gate was bit-identical with coopmat on or off and proved
  nothing about it. `dit_full_parity_large_m` (64 tokens, own pyref cache) is the
  one that actually covers the kernel. Check M against WM before trusting any
  coopmat result.
- **Coopmat is otherwise a MATMUL-ONLY win.** Shared-memory staging is a dead end (slower than
  L2-served global coopLoad). Flash-attn coopmat SDPA ~13x slower than `sdpa_sg`.
  cross-qkv coopmat DEVICE-LOSES (un-normed text > f16 65504). Coopmat device-loses
  for M < WM(32) AND fringe-reads past the A-side alloc for ragged M -> pad staging
  to `next_multiple_of(WM)`. naga gotcha: `var c: coop;` in a loop null-inits ONCE.
- **DiT denoise is at the WGSL matmul ceiling** (latency/occupancy-bound); bigger
  tiles BACKFIRE; weight-only quant does NOT speed compute. Measure via e2e
  `gpu_ms by pipeline`, not microbench.
- **i8 DP4A**: qkv/ffn_up only (normed A-sides). proj/ffn_down carry outliers ->
  f16-cast coopmat OK, i8 acts NOT. Cross-attn-qkv from un-normed text: never.
- **Q8_0 is the quality+perf baseline for big DiTs** (Q4_K per-request fold was ~2x
  SLOWER and broke quality). Q4_K_M = footprint option only. quantize_row panics
  for Q5_K/Q6_K (dequant-only), so those are not fold-shrink options.
- **Weight transcode must catch every float source encoding** (a Q8 arm matching
  only Bf16 let F32/F16 fall through to dense while the site read Q8 -> garbage).
- **BLOCK-WIDE f16 is a dead end** on big-DiT residuals (outlier channels > 65504);
  bf16 residual is load-bearing. Mixed-precision f16 SDPA (post-norm Q/K/V cast) is
  safe and shipped.
- **VAE decode is conv-GPU bound**; do NOT retry the conv3d im2col hoist (slower).
  Tiny (TAEHV) decoders are the fast default where they exist; LTX tiles its VAE
  (seed BELOW budget, balanced re-seed on OOM).
- **Never run fp32 CPU pyref above tiny dims** (40GB host at 256x256).
- **Budgets: stream in/out under pressure, never predict reserves.** No carve-out
  reserves. Do NOT pin a weight slice when working set >> VRAM budget (starves
  same-size recycle -> arbiter reclaim mid-compute -> device lost). Pin only when
  the whole DiT fits. `--vram-budget 6G` device-loss was fixed by the residency /
  coopmat fixes above, NOT by eager-destroy.
- **LTX**: off-subject output = resolution/adherence regime, NOT text encoding
  (512+ two-stage widescreen is the regime). Gemma encoder MUST run F32 acts. The
  gemma `(1+w)` norm bake is REAL. Don't flip the LTX DiT to strict budget.
- **Attention windowing (W=3) breaks temporal identity** on multi-subject/tracked
  content. Hybrid (full step 0, windowed after) restores scene stability but still
  drifts attributes; pixel-MAD is blind, eyeball is the gate. Full attention is the
  identity tier.

## Carry-forward gotchas (engine-general)

- **Ops reading aux params as f32 must dequant bf16 weights first** (passes
  conformance -- no such dtype pair there -- fails in the model).
- **Passthrough 1-D F32 weights arrive on GPU as BF16** under bf16 acts; gemma-style
  `(1+w)` norm needs the bf16-weight kernel variant.
- **F16 params the model reads RAW must use `register_raw_param`**;
  `register_linear`'s transpose silently scrambles them (structured artifacts, no NaN).
- **umT5 / large-residual encoders MUST run bf16 acts** (f16 overflows -> washed
  blob). Check non-finite, not just NaN.
- **Module-level matmuls need their OWN bf16 `matmul_module` pipeline.**
- **Shared-single-pipeline encoders can't do per-layer mixed K-quant** (scheme fixed
  at compile from the first probe). Mixed `_K_M` GGUFs bump attn_v/ffn_down to Q6_K.
- **RoPE**: Wan RoPE3D + Hunyuan = interleaved-pair; Qwen3 + LTX = half-rot.
- **GGUF tensor padding**: F32/F16 narrow arms slice `elements*size`, not on-disk bytes.
- **Cargo workspace root is `thinfer/projects/`** (not the repo top). Watch
  tee-pipelines masking the real exit code.
- **Windows**: a running serve LOCKS `thinfer-serve.exe`; `cargo build --release`
  silently leaves the old exe (exit 0). STOP serve before a release rebuild.

## Crate layout + serve (shipped reference)

- Layering: core < models < native < app < {cli, serve}; wasm `thinfer-web` is its
  own substrate. serve = separate binary over shared `thinfer-app`
  (`JobExecutor` trait). Web UI is `include_str!`-baked -> web edits need a serve
  rebuild. One fresh wgpu device per JOB. No cross-request warm state.
- All image inputs ride in RAM (`ImageBytes`, redacting Debug). Only on-disk bytes
  are the encrypted video spill (AES-256-GCM, ephemeral RAM key) + the output
  artifact; both wiped by delete-on-fetch.
- **Deploy**: `cargo build -p thinfer-serve --release`, launch with `--config
  <serve.toml>` (local serve ALWAYS self-signed HTTPS), confirm "listening (https)".
  `ram_budget=28G` (TI2V host KV cache ~14GB at 77f). STOP a running serve before a
  release rebuild (Windows locks the exe; see gotchas).
- CLI/web parity: every generate flag wires through serve wire.rs/api.rs + web UI in
  the same change. Never log prompt text.

## Running the e2e / measuring

Card = RTX 5070 Laptop (8GB); keep budgets <8GB. All serial (`--test-threads=1`).
Always `THINFER_TRACE=1 THINFER_POWER_PREF=high` + `THINFER_E2E_PNG_DIR` for
staging; read the `gpu_ms by pipeline` rollup first. coopmat needs
`THINFER_POWER_PREF=high` (else iGPU, no coopmat / 4x slower).

- LTX: `rapid_t2v_e2e_health` / `t2v_e2e_health` (`--features ltx-e2e`); component
  parity gates ONE AT A TIME (12.5GB encoder OOMs a multi-test binary).
- Causal TI2V: `i2v_e2e_health` (`--features hunyuan-e2e`); `THINFER_I2V_T2V_PROBE=1`
  text-only; `THINFER_AR_DIAG=1` per-stage.
- Hunyuan T2V: `t2v_e2e`, `t2v_perf`, `dit_parity`.
- FastWan: `video_e2e`. LongLive: `longlive_parity`/`longlive_e2e`. AnyFlow:
  `anyflow_e2e`. Wan2.2: `wan22_e2e`.
- CLI run: `THINFER_TRACE=1 THINFER_POWER_PREF=high thinfer generate ...
  --vram-budget 5G --ram-budget 5G` (TI2V needs bigger ram for the KV cache).

## Open / Next (by track)

- **sulphur-2 / ltx-2.3 perf.** COOPMAT IS ON and is the big win (see Status).
  Where the remaining 506s 5s run goes, at 768x512 two-stage 8+3: stage2 denoise
  **198s** (3 steps at ~66s), stage1 **138s** (8 at ~19s), VAE decode **97s**,
  setup **71s**. Ranked next levers:
  1. **VAE decode, 97s** and now the largest non-denoise block. `ltx_vae_conv3d`
     is 44.6s of GPU on the 3s clip at 23.6ms/disp. LTX ships no tiny (TAEHV)
     decoder and the conv3d im2col hoist is a known dead end, so this needs a new
     idea, not a retry. Its acts may still be f32 (H3's video VAE deliberately is);
     f16 there is a quality call on the final picture, not a free win.
  2. **The Sulphur LoRA fold is the fixed-cost floor** (~60s, 676 sites, 12.8GB
     host cache). Already fully backgrounded, but only ~23s of encoder + connector
     exists to hide it behind. No pre-distilled Sulphur GGUF exists upstream (only
     `sulphur_dev-*`; the distilled builds are bf16/fp8 safetensors), and both
     obvious shortcuts are barred -- self-quantizing one, or caching the folded
     DiT to disk. So the win has to be a faster fold: GPU dequant/requant, or
     across-site parallelism (rows already fan out within a site).
  3. DiT streaming. Q6_K (18.7GB vs Q8_0 22.8GB) would cut ~18% of the per-step
     bytes and coopmat dequants to f16 anyway, so the source kind no longer
     affects matmul speed -- but the fold re-encodes to Q8_0 regardless, so only
     the unfolded sites shrink. Quality unmeasured. Not attempted.
- **DON'T re-litigate the LTX residency cap.** 2G vs 6G, interleaved A/B at
  256x256/9f: 163.4 / 164.1 / 164.2 / 155.7s. No signal, and it cannot help much
  anyway -- each `WeightResidency` builds its OWN `MemArbiter` over the full
  `vram_bytes`, so the cap is per-residency, not global. `THINFER_LTX_VRAM_CAP`
  overrides it if re-measuring.
- **ltx2-rapid**: remaining perf (both large/uncertain) -- two-stage half-res
  upscaler (unproven LTX-2 latent transfer); encoder cold weight-streaming
  (structural; Q4 encoder = quality cost).
- **wan2.2/anyflow**: per-expert LoRA strength knob (CLI+web); fold-cache ram-budget
  design (rides outside the ram budget today); 832x480 quality A/Bs; Q4_K_M ladder
  decision. AnyFlow 512x320 tiny-VAE decode OOM (budget-derived chunk sizing leaves
  no headroom; consider strict-mode shrink+retry like LTX).
- **face-swap**: FF-Android blob-video playback (needs a trusted cert for a
  ServiceWorker, or fMP4/MSE); widen audio passthrough beyond H.264+AAC-LC; SCRFD
  batch>1 executor bug (unlocks detection batching); f16 acts (per-model, the
  bandwidth ceiling-raiser); job-dir artifact TTL/reaper.
- **DreamID-V** 480p footprint/perf: Q8_0 block_transcode + VAE tiling; remove the
  per-request cold fp32->bf16 DiT transcode.
- **User LoRA vault**: real Civitai adapter eyeball end-to-end (Krea2 + qwen-rapid).
