# HunyuanVideo 1.5 causal TI2V (minWM "worldplay" dmd) port

New model `hunyuan-video-1.5-ti2v` alongside `hunyuan-video-1.5-t2v` (which
stays untouched): prompt [+ optional first-frame image] -> 77-frame 832x480
video, 4-step AR chunked denoise. `--input-image` OPTIONAL: with it = I2V
(SigLIP + VAE-encoded first frame conditioning); WITHOUT it = TEXT-ONLY (the
upstream `mask_type="t2v"` shape: no vision tokens, zero cond/mask channels).
PROBE-VALIDATED 2026-07-01: text-only at 448x256x13f produced a coherent,
detailed, prompt-following clip (rubber duck on rippling water) despite the
i2v-trained checkpoint -- so this model doubles as the FAST T2V path (the
original goal). Text-only runs get the t2v prompt rewriter; image runs skip it
(the text-only rewriter can't see the image).

## Source of truth

- Repo: `MIN-Lab/minWM`, checkpoint `HY15/TI2V/dmd/diffusion_pytorch_model.safetensors`
  (33GB F32, 1793 tensors, diffusers-style names, `ARHunyuanVideo_1_5_DiffusionTransformer`).
- Reference impl: `third-party/minWM` (`HY15/hy15_inference.py` `run_inference_rollout`,
  `hyvideo/models/transformers/worldplay_1_5_transformer.py`).
- Same block arch as our T2V dit.rs: 54 double blocks, dim 2048, heads 16 x 128,
  mlp 8192, qk-norm rms, mod 6*dim. Differences:
  - `img_in.proj [2048, 65, 1,1,1]`: 65 in-ch = 32 latent + 32 cond-latent + 1 mask.
  - `final_layer.linear [32, 2048]` (out 32).
  - `vision_in.*` (SigLIP 1152 -> 2048 projection), `byt5_in.*` (1472 -> 2048 mapper).
  - patch 1x1x1 (no patchify): 1560 tokens/latent-frame at 480x832 (latent 30x52).

## Inference scheme (faithful to run_inference_rollout)

- T = (frames-1)/4+1 latent frames (77 -> 20), chunk = 4 latent frames -> 5 chunks.
- Conditioning: first frame -> HY15-VAE-encode -> cond latent (frame 0, zeros after),
  mask ch (1 at frame 0). hidden = cat(latent 32, cond 32, mask 1) = 65 ch.
- Phase 1 txt: forward txt stream once at t=0 through all 54 blocks, cache per-layer
  txt K/V (~1k tokens). txt tokens = refiner(Qwen2.5-VL embeds) [+ byt5 + vision
  streams with cond_type embeddings 0/1/2].
- Per chunk: 4 euler steps (FlowMatchDiscrete shift 5.0, re-set per chunk, guidance
  1.0 = no CFG) + 1 recache forward at t=0 (stabilization_level-1). SDPA per block:
  q = chunk (roped, absolute frame positions via rope_temporal_size/start_idx);
  kv = txt KV ++ committed vision KV (prior chunks, post-rope) ++ current chunk.
  Dense, no mask (causality = cache contents). Recache K/V appended to vision cache.
- v1 byt5: zero tokens + zero mask (upstream-sanctioned byt5_model=None path);
  skip loading byt5_in. Follow-up if glyph quality matters.
- vision stream: SigLIP (HY1.5 repo `vision_encoder/siglip`) encodes the cond image;
  tokens join the txt stream side (cond_type 2). NEW encoder port (standard ViT).
- No rewriter for v1 (i2v rewrite needs image-aware VL; ours is text-only).

## Memory plan (the hard constraint)

- Vision KV cache = 442KB/token (54L x K+V x 2048 x bf16) -> 13.8GB at 77 frames.
  DOES NOT fit VRAM. LongLive pattern: RAM-staged KV (RamKvStore-like), upload
  prefix per layer per forward (~12s PCIe total per gen), commit chunk K/V once.
- RAM: KV 13.8GB + Q8 weights ~8.8GB. Respect ram-budget: size KV plan from the
  budget; error clearly if frames*budget infeasible (or reduce default frames).
- Weights: F32 safetensors -> Q8_0/bf16 transcode at load (extend transcode F32 arm
  if missing). i8 DP4A on qkv/ffn_up per T2V policy; f16 SDPA (post-norm roped).

## Honest perf expectation

Recache adds ~25% matmul; causal attention saves ~45% of SDPA. Net ~1.3-1.5x vs
our T2V 4-step, PLUS first-chunk streaming preview (~1/5 wall to first frames)
and new I2V capability. Judge in browser.

## State (all built, compiling clean; e2e pending weight download)

- Model id `hunyuan-video-1.5-i2v` (`VideoModelId::Hunyuan15I2v`, `is_hunyuan_i2v`).
  Grid: frames with latent count divisible by 4 -> {13, 29, 45, 61, 77}; default
  77f @ 16fps 832x480 (snap handles it). `--input-image` REQUIRED (resolve
  enforces; other models reject it).
- New code: `hunyuan/dit/ar.rs` (child module of dit.rs: txt pass with per-block
  K/V capture, chunk forward over host-staged KV + ping-pong VRAM prefix bufs,
  recache commit, 4-step euler shift-5 loop), `hunyuan/siglip.rs` (so400m ViT on
  BlockPipelines; head_dim 72 -> dense SdpaF32 fallback), `hunyuan/vae/encode.rs`
  (single-frame encoder half; host rearranges between per-stage scopes;
  distribution MEAN not sample). Manifest roles DIT_I2V (MIN-Lab/minWM) + SIGLIP
  (Comfy-Org/sigclip_vision_384). App driver `hunyuan::run_i2v` (phases like T2V;
  RAM-budget feasibility check on the KV cache: 442KB/token, ~14GB at 77f).
- Wired: CLI (`--input-image`, incl. --remote base64), serve wire `inputImage`
  (base64 -> job-dir file, like image-edit), web (model entry, image picker,
  rewrite/attn-window hidden for i2v). Rewrite NOT available on i2v (text-only
  rewriter; image-aware is a follow-up).
- Test: `i2v_e2e_health` (`--features hunyuan-e2e`): engine-only health gate
  (no pyref: minWM needs CUDA+flash-attn), synthetic first frame, default
  448x256x13f one-chunk; THINFER_E2E_* scales to product dims; PNG staging.
- Known deviations (imperceptible-class, documented in code): byt5 omitted
  (upstream byt5_model=None path), VAE-encode mean vs sample, single 384 resize
  for SigLIP (upstream double-resizes), no first-chunk special schedule (HY15
  uses uniform 4-step everywhere, unlike Wan CF++).

## Health gate: GREEN (2026-07-01, shipping config i8+coopmat)

`i2v_e2e_health` 448x256x13f: frame 0 faithfully reproduces the conditioning
image, motion coherent across frames (disc drifts smoothly), video std 0.336,
temporal MAD 0.11. ar denoise 323s at that grid INCLUDING the cold 33GB F32
read + CPU Q8/bf16 transcode (per-request cost under the no-cross-request-state
policy, like the other transcoding models). Serve redeployed with the model +
the attn-window revert.

**Bug found + fixed on the way (engine-general): `register_linear_transcode`
only honored `transcode` for Bf16 sources; an F32/F16 source silently registered
DENSE (transposed bf16) while the site's quant pipeline read the buffer as a
Q8_0 stream -> garbage.** Never showed before because every transcoding source
was BF16 (the "fp16" lightx2v T2V file is actually BF16). Diag lesson: garbage
q/k LOOK clean post-qk-RMSNorm (it normalizes anything); check V or proj
outputs when hunting this class. `THINFER_AR_DIAG=1` now prints per-stage stats.

## Text-only (T2V) mode: VALIDATED at 480p (2026-07-01)

832x480x29f text-only run: sharp, detailed, temporally coherent (rubber ducks
on rippling water). ar denoise 693.7s for 2 chunks (10 forwards) -> ~29min
projected for 77f... NO: 5 chunks scale super-linearly in KV len; measured 2
chunks = 694s, per-chunk cost grows with the prefix, expect ~30-35min for 77f
denoise + tiny-VAE decode. (448x256x13f = 323s incl cold transcode.) Adherence
at probe = no rewriter; product text-only path RUNS the rewriter. Perf note:
the full-VAE decode at 29f took 461s; tiny-ft stays the practical default.

## Budget policy fix (user directive 2026-07-01, engine-wide)

"Respect budgets by streaming in/out, not by predicting reserves." Both Hunyuan
drivers' predicted-reserve carve-outs REPLACED with the arbiter pattern the
other pipelines use: `residency.arbiter().register(RECLAIM_EVICTABLE_WEIGHTS,
residency.reclaimer(backend))` so workspace growth evicts unpinned weights
under the single budget. See [[feedback_vram_policy]] memory.

## Next

1. Browser eyeball at 832x480 with a real photo (I2V) AND text-only (T2V mode);
   77f needs ~14GB host RAM for the KV cache (serve ram_budget=28G covers it).
2. Follow-ups: image-aware prompt rewrite; i8 KV cache (halves the 14GB + PCIe);
   optional recache-skip perf lever (gated, deviates from reference); GPU prep
   kernel for F32->Q8 transcode (CPU-only today, cold-load cost on the 33GB F32);
   T2V-side perf plan (gpu_ms rollup + i8-SDPA lever) still open in worklog.
