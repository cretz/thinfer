# LTX-2.3 port plan (ACTIVE)

Target: `unsloth/LTX-2.3-GGUF` **distilled-1.1**. A 22B joint audio-video
diffusion model (Lightricks LTX-2). This is the largest port yet: 5-6 net-new
subsystems. Upstream ground truth = `third-party/LTX-2` (cloned) +
`third-party/diffusers` (LTX2 modules). Read those, do not re-derive.

## Objective / non-negotiables

- **Best quality AND highest perf AT THE END.** Intermediate work cost is
  irrelevant. No quality shortcuts (full two-stage, full joint AV, best weights).
- **Q8_0 DiT is the quality+perf baseline** (NOT Q4_K_M). Qwen lesson: a
  per-request whole-DiT Q4_K fold re-quantizes every request -> ~2x slower than
  Q8_0 AND quality broke. Weight-only quant never buys compute (matmul ceiling);
  perf comes from kernel work (i8 DP4A, mixed-precision f16 SDPA). Q4_K_M =
  comparison, Q2_K = footprint-floor experiment only.
- **Two render paths** (upstream `LTX2_DISTILLED` vs `LTX2_TWO_STAGE`). DEFAULT =
  single-stage 8-step denoise at the target res (in-distribution at low res, no
  upscaler model swap). OPT-IN `--upscale` / `upscale` = stage1 8-step half-res ->
  spatial upscale x2 -> stage2 3-step refine, the cheaper route to HIGH res only.
  Decision 2026-06-26: single-stage is default because the two-stage half-res
  stage 1 goes OOD below ~1024-long-side (e.g. 512x320 -> stage1 256x160 = mush;
  symptom was "coherent but wrong subject/action" + a 5x-slow "step 8" that was
  really the upscaler load + 22GB DiT re-stream). `width`x`height` = output res in
  BOTH paths; `--upscale` only changes how it is reached.
- **Joint audio-video.** Runtime opt-out (`--no-audio`) skips ONLY the decode tail
  (audio VAE + vocoder + mux); for best quality the DiT keeps the audio stream
  (a2v/v2a cross-attn slightly affects video). A trained `VideoOnly` mode exists
  (pass audio=None, cross-attn not built) as the cheaper reduced path if wanted.
- **pyref mandatory**: component-wise, SAME GGUF weights, tiny dims, dequant to
  bf16 (never fp32 above tiny = 40GB lesson), never a full-pipeline or full-DiT
  ref (OOMs). **Host RAM ceiling ~10GB** for every pyref (see Gemma P0 risk).
- **Rust honors `--vram-budget` (hard ceiling) + `--ram-budget`**: phase-aware
  eviction, per-block DiT streaming sized to budget, weights mmap'd (never WASM
  linear mem), bounded pin. Proven on Qwen's 20B.
- Conventions: no em dashes; after Rust edits `cargo fmt && cargo clippy` (fix ALL
  warnings); worklog forward-only; surface plan before new ops/files/wiring.

## Weights (all downloaded, in HF cache)

| Role | Repo | File | Size |
|---|---|---|---|
| DiT Q8_0 (baseline) | unsloth/LTX-2.3-GGUF | distilled-1.1/ltx-2.3-22b-distilled-1.1-Q8_0.gguf | 22.8G |
| DiT Q4_K_M (compare) | " | distilled-1.1/...-Q4_K_M.gguf | 14.2G |
| DiT Q2_K (floor) | " | distilled-1.1/...-Q2_K.gguf | 7.94G |
| Text encoder | unsloth/gemma-3-12b-it-qat-GGUF | gemma-3-12b-it-qat-UD-Q4_K_XL.gguf | 7.43G |
| Connector + FE + aggregate embeds | unsloth/LTX-2.3-GGUF | text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors | 2.31G |
| Video VAE | " | vae/ltx-2.3-22b-distilled_video_vae.safetensors | 1.45G |
| Audio VAE + vocoder(BWE) | " | vae/ltx-2.3-22b-distilled_audio_vae.safetensors | 365M |
| Spatial upscaler x2 | Lightricks/LTX-2.3 | ltx-2.3-spatial-upscaler-x2-1.1.safetensors | 996M |

- Audio VAE file = 102 `audio_vae.*` + **1227 `vocoder.*`** tensors (BigVGAN v2 +
  `bwe_generator.*`). Vocoder is NOT a separate download.
- distilled-1.1 reuses the `-distilled_` connector + VAEs (no 1.1-specific
  companions exist).

## Confirmed architecture (GGUF `config` KV = authoritative; re-dump via gguf.GGUFReader)

DiT GGUF: arch=`ltxv`, `AVTransformer3DModel`, 4444 tensors, `transformer_blocks.0..47`.
Tensor names already stripped to `transformer_blocks.N.*` / `audio_*` (minimal/no
rename map vs upstream comfy keys -- verify on load).

- **DiT**: 48 layers. Video heads 32 x head_dim 128 = **4096**. Audio heads 32 x
  head_dim 64 = **2048**. in/out 128. `qk_norm=rms_norm`,
  `standardization_norm=rms_norm`, `norm_elementwise_affine=false`, eps 1e-6.
  `num_embeds_ada_norm=1000`, `timestep_scale_multiplier=1000`. `attention_bias=true`.
  `apply_gated_attention=true` (per-head sigmoid gate). 9 AdaLN mod params
  (`scale_shift_table [*,9]`); separate continuous-sigma prompt AdaLN
  (`prompt_scale_shift_table [*,2]`); av-cross ada-norm (`scale_shift_table_a2v_ca_*
  [*,5]`). `cross_attention_dim=4096` (video text), `audio_cross_attention_dim=2048`.
- **RoPE**: `rope_type="split"` = **HALF-ROT (Qwen3-style, NOT Wan interleaved)**.
  theta 1e4, `frequencies_precision=float64`, video `max_pos=[20,2048,2048]`,
  audio `[20]`, `use_middle_indices_grid=true` (positions are [start,end) bounds,
  middle-of-patch), `causal_temporal_positioning=true`. Positions in PHYSICAL
  units: latent coords -> pixel/second, video temporal / fps, causal_offset=1 on
  frame 0; audio coords in seconds via hop/sr. Net-new position construction;
  rotation kernel reuses Qwen3 half-rot + `op_rope`.
- **Block = 5 attn sublayers**: (1) video self, (2) audio self, (3) video<->text
  cross (`attn2`), (4) audio<->text cross (`audio_attn2`), (5) audio<->video cross
  (`audio_to_video_attn` Q=video/KV=audio + `video_to_audio_attn` Q=audio/KV=video,
  the sync mechanism). Per-modality FFN gelu-approximate. Patchify `patch_size=1`
  (each latent cell = 1 token). Predicts **X0** (denoised), not velocity.
- **Connector** (weights in connector safetensors, config in DiT GGUF):
  `use_embeddings_connector=true`, `connector_num_layers=8` (NB: 8, not the
  diffusers default 2), heads 32 x 128, `max_pos=[4096]`,
  `num_learnable_registers=128`, gated attn, `norm_output=true`,
  `caption_proj_before_connector=true`. Audio connector heads 32 x 64.
  `text_encoder_norm_type=per_token_rms`.
- **Text encoder**: Gemma-3-12B (`Gemma3ForConditionalGeneration`, inner
  `model.model`). Tokenize left-pad `max_length=1024`, no chat template (raw
  prompt, stripped). `output_hidden_states=True` -> ALL layers incl embedding layer
  -> stack `[B,T,D,L]`. **FeatureExtractorV2** (22B path): per-token RMS over D,
  reshape `[B,T,D*L]`, zero pad positions, rescale `sqrt(out/emb)`, two `nn.Linear`
  aggregate embeds (video, audio). Then 8-layer connectors -> video 4096 / audio
  2048. (19B used FE V1 -- DO NOT use.)
- **Video VAE**: `CausalVideoAutoencoder`, dims 3, latent 128ch. scale t8/h32/w32
  (frames must be 8k+1; latent H=H/32, W=W/32). PerChannelRMSNorm (not GroupNorm),
  pixel-shuffle residual down/up-samplers, **timestep-conditioned + noise-injecting
  DECODER** (decoder takes a temb), +1 latent-channel broadcast (conv_out 129 ->
  replicate), reflect pad (decoder) vs zeros (encoder), per-channel
  latents_mean/std (128-vec). ~40-50% net-new vs wan/vae.rs core.
- **Audio VAE decoder (P5; DISK-VERIFIED config + tensors, bf16)**: 2D causal conv
  over mel (NOT 3D; own file `audio_vae.safetensors`, prefix `audio_vae.decoder.*`
  56 tensors + `per_channel_statistics.{mean,std}-of-means [128]`). cfg: ch 128,
  ch_mult [1,2,4], z 8, latent mel_bins 16 -> DECODED mel_bins 64, out_ch 2,
  `norm_type=pixel` (PixelNorm = channel-RMS eps 1e-6, NO affine -> no norm
  tensors), `causality_axis=height`(=time), `mid_block_add_attention=false` +
  `attn_resolutions=[]` -> **NO attention anywhere**. NO temb, NO noise inject
  (temb_ch 0). Graph: denormalize (patchify `b c t f->b t (c f)` 8*16=128 ->
  `x*std+mean` -> unpatchify) -> conv_in Conv2d[512,8,3,3] -> mid(resnet,resnet) ->
  up[2,1,0] each = 3 ResnetBlocks (+ Upsample on levels 2,1) -> norm_out(PixelNorm)
  -> SiLU -> conv_out Conv2d[2,128,3,3]. Channels 512->512->256->128. ResnetBlock =
  pixelnorm->silu->convk3 ->pixelnorm->silu->convk3 + (1x1 nin_shortcut if c
  changes); k1 shortcut keys `...nin_shortcut.conv.weight [128,256,1,1]`. CausalConv2d
  wraps `nn.Conv2d` (extra `.conv.` in keys), pad `(W/2,W-W/2, H_pad, 0)` (k3 ->
  `(1,1,2,0)`: freq symmetric, time fully LEFT/causal). Upsample = nearest x2 (BOTH
  t,f) -> CausalConv2d k3 -> DROP first time frame. Out mel `[2, 4*frames-3, 64]`.
- **Vocoder (P5; DISK-VERIFIED, bf16, prefix `vocoder.*` 1227 tensors)**: mel ->
  48kHz stereo. `VocoderWithBWE` = main BigVGAN + BWE BigVGAN residual + sinc
  resample, all **fp32** (bf16 degrades 40-90% over ~108 convs). Run whole tail f32.
  - **main `vocoder.vocoder` (667 tensors)**: init_ch **1536**, **6** up stages
    rates [5,2,2,2,2,2] (x160) kernels [11,4,4,4,4,4], `use_tanh_at_final=false` ->
    CLAMP, `use_bias_at_final=false` (conv_post no bias). conv_pre Conv1d[1536,128,7]
    (128=2x64) -> 6x {ConvTranspose1d up + mean of 3 AMP resblocks} -> act_post
    SnakeBeta -> conv_post Conv1d[2,24,7]. ch 1536->768->384->192->96->48->24. 18
    resblocks (6 stages x k{3,7,11}). out 16000 Hz.
  - **`vocoder.bwe_generator` (557)**: init 512, **5** stages [6,5,2,2,2] (x120)
    kernels [12,11,4,4,4], `apply_final_activation=false` (RAW residual). conv_pre
    [512,128,7], conv_post [2,16,7]. 15 resblocks. in 16000 -> out 48000 Hz.
  - **`vocoder.mel_stft` (3)**: STFT-as-conv1d (`forward_basis [514,1,512]`,
    `mel_basis [64,257]`; n_fft 512 hop 80 mel 64) recomputes a log-mel from the
    main vocoder's 16k waveform to drive the BWE. `inverse_basis` loaded, UNUSED.
  - **BWE chain**: `clamp(bwe_generator(mel_stft(main(mel))) + resampler(main(mel)),
    -1, 1)[:output_len]`. `resampler` = Hann-window sinc UpSample1d ratio 3
    (48000/16000), `persistent=false` -> NOT in ckpt, regenerate. Anti-alias
    `Activation1d` filters (kaiser sinc, `[1,1,12]`) ARE persistent buffers but
    deterministic. SnakeBeta: `x + (1/exp(beta))*sin(exp(alpha)*x)^2` (alpha/beta
    logscale, eps 1e-9). NET-NEW ops: Conv1d, ConvTranspose1d, SnakeBeta,
    anti-aliased Activation1d (sinc up/down), STFT-as-conv1d. mel layout to vocoder
    `[2,T,64] -> transpose -> rearrange b s c t -> b (s c) t -> [128,T]`.
  - (ltx-plan v0 said 24kHz / init 1024 / rates [6,5,2,2,2] -- WRONG; the on-disk
    `__metadata__.config` corrects: 48kHz out, main init 1536/6-stage, BWE 5-stage.)
- **Spatial upscaler**: `LatentUpsampler` latent-space CNN (init conv -> 4 ResBlocks
  -> PixelShuffle x2 -> 4 ResBlocks -> final conv; in 128, mid 512). Operates on
  UN-normalized video latents: un-normalize -> upsample x2 -> re-normalize (needs
  video VAE per-channel stats). No temporal upscaler in distilled path.
- **Sampler**: X0-prediction + Euler (`euler_denoising_loop`, `EulerDiffusionStep`),
  CFG-free (`SimpleDenoiser`, B=1). Distilled sigma tables (`ltx_pipelines/utils/
  constants.py`):
  - Stage1 (8 steps): `[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]`
  - Stage2 (3 steps): `[0.909375, 0.725, 0.421875, 0.0]` (tail subset)
  - Two-stage flow: stage1 at H/2,W/2 -> upscaler x2 latent -> stage2 re-noises
    upscaled latent to sigma 0.909375 and refines 3 steps. Audio stage2 likewise
    re-noises its stage1 latent. H,W divisible by **64** (two-stage); frames 8k+1.
    Default target 1024x1536, 121 frames @ 24fps.

## P1 locked spec (text conditioning; verified vs upstream ltx-core, NOT diffusers)

Use the `ltx-core` path (GGUF matches it); the diffusers `pipelines/ltx2/
connectors.py` is the OLD variant (single `text_proj_in`, no gate) -- DO NOT use.
Chain: Gemma(49 states) -> FE V2 -> aggregate_embed Linear -> right-pad reorder ->
video_connector -> audio_connector. DiT cross-attn KV = video [B,1024,4096] +
audio [B,1024,2048] (all 1024 positions valid; registers fill the pads).

- **Tokenize**: `max_length=1024`, **LEFT-pad**, `padding="max_length"`, truncation,
  `text.strip()`, NO chat template, pad=eos. Forward inner Gemma backbone with the
  binary attention_mask + `output_hidden_states` -> ALL 49 states `[1,1024,3840]`.
  **Positions = mask cumsum (valid tokens contiguous from 0), NOT naive arange**
  (left-pad!). pad rows masked.
- **FE V2** (`feature_extractor.py:48-69,114-129`): stack -> `[B,T,D=3840,L=49]`;
  per-token RMS over the **D axis per-layer** (eps 1e-6, no weight): `normed =
  enc * rsqrt(mean(enc^2 over D)+eps)`; reshape C-order `[B,T,D,L]->[B,T,188160]`
  so **flat index = d*49 + l** (layer-major within channel); zero padded-token
  rows; then PER HEAD `x * sqrt(out/3840)` (video sqrt(4096/3840), audio
  sqrt(2048/3840)) -> `video_aggregate_embed` Linear(188160->4096,bias) /
  `audio_aggregate_embed` Linear(188160->2048,bias). SAME 188160 vector to both.
- **Right-pad reorder** before connector: stable argsort on the binary mask per
  row -> `[valid..., pad...]`; reuse the same idx for audio.
- **Connector** (8 layers, per modality; `embeddings_connector.py:135-187`):
  - register replace (NOT prepend): assert S%128==0; tile `learnable_registers`
    (stored `[dim,128]` in GGUF = `[128,inner]` logical) to `[S,inner]`; `h =
    mask*h + (1-mask)*registers`; then mask zeroed (full bidirectional).
  - rope = **SPLIT** (half-rot), positions arange(S) over ALL S, theta 1e4,
    max_pos 4096, float64 freq table: `pow=theta^linspace(0,1,inner/2)`,
    `indices=pow*pi/2`, `frac=pos/4096`, `freqs=indices*(frac*2-1)`.
  - block: pre-norm WEIGHTLESS `rms_norm` (inner_dim, eps1e-6) -> attn1 -> +res ->
    weightless rms_norm -> ff -> +res.
  - attn1: to_q/k/v (bias); `q_norm`/`k_norm` = `RMSNorm(inner_dim, eps1e-6, WITH
    weight)` over the **FULL inner_dim** (NOT per-head); rope SPLIT on q,k; sdpa
    scale 1/sqrt(head_dim) non-causal; **gate** = `2*sigmoid(to_gate_logits(
    x_prenorm))` Linear(inner->heads,bias), per-head multiply on attn out BEFORE
    `to_out.0` Linear(inner->inner,bias).
  - ff: `net.0.proj` Linear(inner->4*inner,bias) -> gelu-tanh -> `net.2`
    Linear(4*inner->inner,bias).
  - after 8: weightless `norm_out` rms_norm; registers KEPT; all 1024 valid.
- NB connector qk-norm is full-inner-dim (delta from the DiT's per-head qk-norm).

## Build phases (smallest closed loop first; each has a pyref + budget gate)

- **P0 -- confirms + scaffold.** (a) **Gemma all-layer-hidden-states pyref <=10GB**
  (THE risk -- see below); prove it before P1. (b) `ltx/{mod,manifest,config}.rs`
  scaffold from GGUF config consts; wire lib.rs; build green. (c) Decide DiT GGUF
  rename map (names look pre-stripped; confirm).
- **P1 -- Gemma-3 encoder + connector + FE V2** -> encoder_parity (<=10GB pyref,
  quant-aware). New Gemma-3 block; 8-layer gated connector transformer; per-token-RMS
  FE + aggregate embeds (load from connector safetensors).
- **P2 -- video VAE decode** -> vae_parity (the ~50%-new conv stack: PerChannelRMSNorm,
  pixel-shuffle residual samplers, timestep-conditioned noise-injecting decoder).
  Reuse wan/vae.rs causal-conv3d core where possible.
- **P3 -- dual-stream DiT (full, incl audio stream), 1-block** -> dit_parity
  (split/half-rot rope w/ middle-index physical coords, 5 attn, gating, 9 mod, X0).
  Full 48-block bf16 ref OOMs -> 1-block parity + e2e health only.
- **P4 -- silent-video t2v e2e health = FIRST DELIVERABLE.** Two-stage orchestration
  (stage1 8-step half-res -> spatial upscale -> stage2 3-step refine) -> video VAE
  decode -> frames. Skip decode tail. Component-parity + visual (no full pyref).
- **P5 -- audio decode tail** -> audio VAE decode + vocoder (BigVGAN+BWE, fp32) +
  mux audio into MP4 (openh264 + audio track). Joint AV e2e health.
- **P6 -- VRAM/perf + product wiring.** Phase eviction (Gemma -> evict -> DiT
  streamed-to-budget -> evict -> video VAE -> audio VAE/vocoder); i8 DP4A +
  mixed-precision f16 SDPA on the DiT (watch the residual-outlier dead-end: only
  normed Q/K/V go f16, residual stays bf16); two-stage orchestration perf. Then
  ModelId + CLI + serve (VideoSpec) + web wiring.

## pyref strategy + the Gemma 10GB risk (P0)

- VAE decode/encode, audio VAE, vocoder (fp32), upscaler, DiT 1-block: all small,
  fit <=10GB trivially. Load from the SAME GGUF/safetensors we run.
- **Gemma-3-12B bf16 ~= 24GB > 10GB.** Plan: run the encoder pyref from the SAME Q4
  QAT GGUF via a quant-aware loader (~7.4GB resident) with `output_hidden_states`
  for ALL layers (more faithful to parity = same weights). Confirm a working
  transformers-GGUF (or llama-cpp embeddings) path that exposes all-layer hidden
  states. Fallback: layer-chunked streaming load (one layer resident, bounded RAM).
  The connector/FE/aggregate-embed pyref loads the 2.3GB connector safetensors --
  cheap. Run pyrefs as their own process.

## Reuse map

- **Reuse**: half-rot rope kernel + `op_rope` (Qwen3), SDPA / RMS-QK-norm / AdaLN
  primitives, causal Conv3d core (wan/vae.rs), i8 DP4A matmul, mixed-precision f16
  SDPA (`FastSdpa`), GGUF loader + RenamedSource + F16->bf16 weight path,
  MemArbiter / residency / phase eviction / VRAM tiling, app/serve/web stack.
- **Net-new**: Gemma-3 encoder; 8-layer gated connector + FE V2 + aggregate embeds;
  video VAE deltas (PerChannelRMSNorm, pixel-shuffle samplers, ts-conditioned
  noise-injecting decoder); audio 2D mel VAE; BigVGAN+BWE vocoder + mel front-end;
  dual-stream DiT block (5 attn, a2v/v2a, gating); LatentUpsampler; X0 sampler +
  two-stage orchestration; physical-coord/middle-index rope positions; audio MP4
  muxing.

## Upstream source map (third-party, read these)

- Pipeline/orchestration: `LTX-2/packages/ltx-pipelines/src/ltx_pipelines/`
  `distilled.py`, `utils/constants.py` (sigmas), `utils/blocks.py`
  (VideoUpsampler/AudioDecoder/PromptEncoder), `utils/samplers.py`.
- Core models: `LTX-2/packages/ltx-core/src/ltx_core/model/` `transformer/model.py`
  + `model_configurator.py`, `autoencoder...`, `audio_vae/{audio_vae,vocoder}.py`,
  `upsampler/model.py`; `components/{schedulers,patchifiers}.py`.
- Text encoder: `LTX-2/.../text_encoders/gemma/encoders/base_encoder.py`,
  `feature_extractor.py` (V2), `embeddings_processor.py`.
- diffusers cross-check: `diffusers/src/diffusers/models/transformers/
  transformer_ltx2.py`, `autoencoders/autoencoder_kl_ltx2{,_audio}.py`,
  `pipelines/ltx2/{connectors,vocoder}.py`.
