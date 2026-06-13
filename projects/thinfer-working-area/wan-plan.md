# Wan-family video plan (active)

## Thesis

Port the Wan backbone once, unlock a family. Five candidate models share it.
First target = SkyReels-V2-DF-1.3B-540P: lowest resource (1.3B), infinite length
(Diffusion Forcing), commercial license, GGUF already exists, closest to the
existing diffusion denoise loop. Scale to 5B and audio later off the same core.

## Targets (priority order)

1. SkyReels-V2-DF-1.3B-540P. Base Wan2.1-T2V-1.3B. Diffusion Forcing -> infinite
   length. 540P. T2V + I2V variants. License: commercial OK. GGUF:
   wsbagnsv1/SkyReels-V2-DF-1.3B-540P-GGUF (+ ...-I2V-...).
2. LongLive-2.0-5B. Base Wan2.2-TI2V-5B. AR long multi-shot, 720p real-time.
   Quality tier. License: NVIDIA Open Model.
3. NAVA. Wan2.2-TI2V-5B video backbone + LTX2.3 audio VAE/vocoder. Joint
   audio+video MMDiT, 6.3B. License: Apache-2.0. Audio bonus.

Same backbone, cheap adds once Wan is ported: Phantom_Wan-1.3B (subject->video,
identity), LongLive-1.3B (real-time AR, KV-cache).

## New engine work (done once for the family)

- Wan DiT block: DiT, full attention (reuse existing sdpa), cross-attention to
  text, 3D RoPE over (T,H,W) patches, patchify (1,2,2).
- Wan 3D causal video VAE: temporal compression (Wan2.1 VAE ~4x8x8;
  Wan2.2-TI2V 4x16x16). Biggest genuinely-new piece vs Z-Image per-frame VAE.
  Heed the single-heavy-submit VAE rule (plan-details); 3D adds temporal tiles.
- umT5-XXL text encoder: encoder-decoder T5 family, relative-position bias. New
  vs decoder-only Qwen3. Shared across the whole Wan family (and NAVA). ~5.5B;
  quantizable (umt5_xxl fp8/gguf exists in the Wan ecosystem). The recurring
  encoder tax.
- Diffusion Forcing sampler: per-token independent noise levels; extend by
  conditioning on the prior segment's last frames. Stays inside the existing
  diffusion denoise loop. LongLive's AR/KV-cache redesign is deferred.

## Pinned config (SkyReels-V2-DF-1.3B-540P)

Refs (third-party/): convert_skyreelsv2_to_diffusers.py (canonical),
transformer_skyreels_v2.py, autoencoder_kl_wan.py, umt5; GGUF repos.

DiT (SkyReelsV2Transformer3DModel):
- patch (1,2,2); in/out ch 16; heads 12 x head_dim 128 -> inner 1536
- num_layers 30; ffn_dim 8960; freq_dim 256; text_dim 4096; eps 1e-6
- qk_norm rms_norm_across_heads (over full inner, affine); cross_attn_norm True
- added_kv_proj_dim/image_dim None (T2V/DF, no image branch)
- inject_sample_info True (DF-1.3B only: fps_embedding+fps_projection into
  timestep_proj; 14B False)
- DF: per-frame timesteps -> 4D temb broadcast over f*pp_h*pp_w; 6-way
  modulation (shift/scale/gate x self+ffn)
- RoPE3D head_dim 128 -> t=44,h=42,w=42 (h=w=2*(128//6)); theta 10000;
  interleaved-pair (repeat_interleave_real), NOT half-rot (cf Qwen3 fix)

VAE (AutoencoderKLWan / Wan2.1): base_dim 96, z_dim 16, dim_mult [1,2,4,4],
res_blocks 2, temperal_downsample [F,T,T] -> 4x8x8; causal conv3d + feat_cache;
latents_mean/std baked (16-vec). Single-heavy-submit + temporal tiling.

Text enc (umT5-XXL google/umt5-xxl, encoder-only): d_model 4096, d_ff 10240
(gated gelu_new), d_kv 64, heads 64, layers 24, vocab 256384, rel-pos bias
(32 buckets, max 128), ln eps 1e-6. Context 512.

540P (DF defaults 544x960, 97f, 1 block): latent 16x25x68x120 (T=(97-1)/4+1);
DiT tokens 25*34*60=51000 full self-attn; cross-attn ctx 512. >97f stitched via
overlap_history.

Weights (all DiT-only GGUF, Z-Image lesson holds):
- DiT wsbagnsv1/SkyReels-V2-DF-1.3B-540P-GGUF: BF16 2.87 GB, Q4_K_M 992 MB,
  Q8_0 1.55 GB (13 quants)
- umT5 city96/umt5-xxl-encoder-gguf: Q4_K_M 3.66 GB .. Q8_0 6.04 GB .. F16 11.4
- VAE Wan2.1_VAE.pth (Wan-AI/Wan2.1-T2V-14B), safetensors path like Z-Image

## Sources

- SkyReels-V2: github.com/SkyworkAI/SkyReels-V2; arxiv 2504.13074;
  Skywork/SkyReels-V2-DF-1.3B-540P; GGUF wsbagnsv1/SkyReels-V2-DF-1.3B-540P-GGUF.
- Wan2.1: Wan-AI/Wan2.1-T2V-1.3B. Wan2.2-TI2V-5B: Wan-AI + QuantStack GGUF.
- LongLive: arxiv 2509.22622 / Efficient-Large-Model/LongLive-1.3B; LongLive-2.0
  arxiv 2605.18739 / Efficient-Large-Model/LongLive-2.0-5B.
- NAVA: ernie-research.github.io/NAVA; baidu/NAVA (Apache-2.0).
- diffusers SkyReels-V2 pipeline docs.

## umT5 encoder specifics

Mirror `z_image/text_encoder.rs`. Arch deltas vs Qwen3: T5LayerNorm = RMSNorm
(reuse op_rmsnorm); sdpa scale=1.0 (no qk scaling); bidirectional (no causal
mask); no q/k norm, no RoPE; gated-gelu FF (`gelu_new(wi_0)*wi_1 -> wo`, use
`GeluMulF32`); per-layer relative-position bias; runs final_layer_norm. Inner =
heads*d_kv = 4096 = d_model; separate q/k/v (no fused qkv). Weights:
`encoder.block.{i}.layer.0.{layer_norm, SelfAttention.{q,k,v,o,
relative_attention_bias}}`, `.layer.1.{layer_norm, DenseReluDense.{wi_0,wi_1,
wo}}`, `shared`/`encoder.embed_tokens`, `encoder.final_layer_norm`.

Relpos bias (keep the flash kernel generic): SDPA `has_mask` is a MODE -
0=none, 1=shared `[B,S,S]`, 2=per-head `[B,H,S,S]` (only the mask row index
gains `+hq`, no new bindings). A dedicated `relpos_bias` act op expands compact
`table [num_buckets,H]` + CPU `bucket_map [S,S]` (u32) into `[H,S,S]`:
`out[h,i,j] = table[bucket_map[i,j]*H + h]`. bucket = HF
relative_position_bucket (bidirectional, 32 buckets, max_dist 128) of
(key - query); table = relative_attention_bias.weight (HF
Embedding[num_buckets, n_heads]). Identical math to HF's dense `[1,H,S,S]`
bias; compact uploads, no dense materialize.

## e2e parity (pinned design)

Foundation must land before the test (modules are complete but no pipeline yet).
Build order: DF sampler -> WanSource -> WanModel (glue + taps) -> video_e2e_parity
+ gen_video_e2e_parity_ref.py. Mirror `zimage/e2e_parity.rs` + `ZImageSource`/
`ZImageModel` (shared by CLI/web/test); reuse pinned-noise byte-load, per-stage
tap+tolerance+linfit compare, budget asserts, PNG dump.

Pinned config (lowest pyref that still validates both branches everywhere):
- prompt "a red balloon over a green field" (content irrelevant; we byte-compare)
- 64x64, F=5 -> f_lat=2, h_lat=w_lat=8; DiT tokens = 2*4*4 = 32
- steps 2, seed 42, guidance_scale 1.0 (NO CFG: one DiT forward/step)
- noise [16, 2, 8, 8] = 2048 f32, byte-loaded both sides
- WHY F=5/f_lat=2: minimum that hits BOTH causal-VAE branches each side -
  decoder frame0=Rep/no-double + frame1=time_conv double (out 1+4=5 frames);
  encoder chunk0=downsample3d passthrough + chunk1=time_conv-with-cache (4->2->1).
  f_lat=1 would only hit the Rep/passthrough branch. pyref floor = umT5-XXL
  (5.5B, runs once); dims below 64 don't reduce it, so 64 is the sweet spot.
- stages to tap+compare: umt5_out, step{i}.prev_sample, pre_vae_latent,
  vae.<stage>, vae_frames. Add a VAE-ENCODE stage (fixed pinned input video ->
  WanVaeEncoder::encode vs pyref `_encode`) so the encoder is actually validated
  (t2v path only uses the decoder).
- tolerances start broken-vs-noisy, tighten after first clean run (zimage habit).
- staging: per-frame PNG seq ours_/py_ + tiled contact-sheet (worklog carry-fwd).

DF sampler: SYNCHRONOUS mode only for parity (ar_step=0, causal_block_size=1).
`generate_timestep_matrix` then collapses to "all frames share timesteps[i] at
step i", so the loop = standard flow-match denoise over latent video
[16,f_lat,h,w] with per-frame timesteps broadcast equal (DiT already takes
per-frame timesteps). Async/causal-block staggering + overlap_history segment
stitching deferred to long-video (not parity-blocking).

Scheduler DECISION: implement UniPCMultistepScheduler (the shipped
`scheduler_config.json`), NOT Euler. Config: flow_prediction, use_flow_sigmas,
flow_shift 1.0, solver_order 2, solver_type bh2, final_sigmas_type zero,
lower_order_final true, timestep_spacing linspace, num_train_timesteps 1000.
Permanent/production path, reused family-wide; e2e validates the real scheduler.
(At 2 steps it largely degenerates to order-1, but the fixed-shift flow-sigmas +
flow-prediction->x0 conversion still differ from z_image's dynamic-exponential
Euler, so it is its own module: `wan/scheduler.rs`.) pyref must use the bundle's
default UniPC so both sides match.

## Open questions

- umT5-XXL port effort + which quant form to consume.
- Wan 3D VAE tiling vs the single-submit-per-tile rule (temporal tiles added).
- Diffusion Forcing segment length / overlap tuning at 540P.
- License confirm for commercial intent (SkyReels OK; LongLive NVIDIA Open;
  verify Phantom / Wan base).
- SANA-Video (linear-attn, 2B) set aside: research-only license, no GGUF, new
  kernel family. Revisit only if the linear-attention frontier becomes the goal.
