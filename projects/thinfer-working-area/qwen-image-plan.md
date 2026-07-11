# Qwen-Image-Edit-Rapid-AIO port

ACTIVE per-model port. Ground truth = `third-party/diffusers`
(`models/transformers/transformer_qwenimage.py`,
`pipelines/qwenimage/pipeline_qwenimage_edit.py`,
`models/autoencoders/autoencoder_kl_qwenimage.py`) +
`third-party/transformers/models/qwen2_5_vl` (text encoder + vision tower).
Engine module: `thinfer-models/src/qwen_image/`. Template = `ideogram4`
(GGUF-only image DiT, Qwen-VL encoder tap, KL VAE, folded turbo recipe).

## Model identity

- Phr00t `Qwen-Image-Edit-Rapid-AIO`: 4-step distilled finetune of
  `Qwen-Image-Edit-2511`. CFG-free (guidance_scale 1.0, no uncond branch).
- All-in-one EDIT model: takes a reference image + text instruction -> edited
  image. (NSFW/uncensored finetune; that is what the repo ships.)
- Weights = GGUF only, sourced per Phil2Sat recipe (see manifest.rs):
  - DiT: `Phil2Sat/...-GGUF :: v90/qwen-rapid-nsfw-v9.0-{Q8_0,Q4_K_M}.gguf`
    (latest = v9.0). Q8_0 = parity canary, Q4_K_M = runtime default.
  - Text encoder: `Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf` +
    `.mmproj-f16.gguf` (vision tower).
  - VAE: `calcuis/pig-vae :: pig_qwen_image_vae_fp32-f16.gguf`.
  - Tokenizer/processor: `Qwen/Qwen2.5-VL-7B-Instruct`.
- DiT-only GGUF carries NO TE/VAE (same as ideogram/zimage GGUF scope).

## DiT: MMDiT dual-stream (transformer_qwenimage.py)

Config (QwenImageTransformer2DModel defaults; CONFIRM against GGUF metadata):
- `num_layers=60`, `num_attention_heads=24`, `attention_head_dim=128` ->
  `inner_dim=3072`. `patch_size=2`. `in_channels=64` (= 16 latent ch * 2*2).
  `out_channels=16`. `joint_attention_dim=3584` (Qwen2.5-VL hidden).
- `axes_dims_rope=(16,56,56)` sum 128 = head_dim. `theta=10000`,
  `scale_rope=True`. Rapid-AIO base: `zero_cond_t=False`,
  `guidance_embeds=False`, `use_additional_t_cond=False`,
  `use_layer3d_rope=False` (CONFIRM from GGUF; affects modulation + RoPE).

Top-level forward:
1. `img_in: Linear(64 -> 3072)` on packed image latents.
2. `encoder_hidden_states = txt_in(txt_norm(eh))`; `txt_norm` = RMSNorm(3584),
   `txt_in: Linear(3584 -> 3072)`.
3. `temb = time_text_embed(timestep)`: Timesteps(256, flip_sin_to_cos=True,
   downscale_freq_shift=0, scale=1000) -> TimestepEmbedding(256 -> 3072, SiLU).
4. RoPE: `pos_embed(img_shapes, max_txt_seq_len=text_seq_len)` -> (vid_freqs,
   txt_freqs), complex `[seq, head_dim/2]`.
5. 60x QwenImageTransformerBlock.
6. `norm_out = AdaLayerNormContinuous(3072)` on image stream, then
   `proj_out: Linear(3072 -> 2*2*16=64)`. Unpatchify -> latent.

Block (dual-stream, order = text-then-image in attention):
- `img_mod = Linear(3072 -> 6*3072)` after SiLU; chunk into mod1/mod2 (each
  3*dim: shift,scale,gate). Same for `txt_mod`.
- norm1 = LayerNorm(no affine, eps 1e-6). `_modulate`: `x*(1+scale)+shift`.
- Joint attention (QwenDoubleStreamAttnProcessor2_0):
  - img q/k/v = to_q/to_k/to_v(img_modulated); txt q/k/v =
    add_q/add_k/add_v_proj(txt_modulated). All `Linear(3072->3072, bias)`.
  - QK-RMSNorm per head (norm_q/norm_k, norm_added_q/norm_added_k), eps 1e-6.
  - RoPE applied to img q/k with vid_freqs, txt q/k with txt_freqs.
    `apply_rotary_emb_qwen(use_real=False)` = COMPLEX multiply (interleaved
    pair view_as_complex), NOT half-rot. (Same family as Wan RoPE3D; opposite
    of Qwen3 half-rot. See [[project_qwen3_rope_halfrot]].)
  - concat `[txt, img]` along seq -> SDPA (non-causal, padding mask on text)
    -> split back. img_out = to_out(.), txt_out = to_add_out(.).
  - gate1 residual: `hs += img_gate1 * img_attn_out`; `eh += txt_gate1*...`.
- norm2 + mod2 + MLP (FeedForward gelu-approximate, dim->4*dim->dim) on each
  stream, gate2 residual.
- fp16 clip to +-65504 on both streams at block end (we run bf16 acts; clip is
  the upstream fp16 guard, mirror only if we ever run f16 acts here).

## Edit conditioning (pipeline_qwenimage_edit.py)

Two independent conditioning channels from the SAME reference image:
1. VISION-TOWER channel: prompt wrapped as
   `<|im_start|>system\n...describe input image...<|im_end|>\n<|im_start|>user
   \n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n
   <|im_start|>assistant\n`. The VL model runs image+text jointly; the
   `<|image_pad|>` slots are filled by the vision tower (mmproj). Hidden states
   taken from the LM; `drop_idx = prompt_template_encode_start_idx = 64` tokens
   dropped from the front (the system preamble). Result = `encoder_hidden_states`
   `[txt_seq, 3584]` + mask.
2. VAE-LATENT channel: ref image -> `vae.encode` (argmax/mode, NOT sampled) ->
   normalize `(z - latents_mean)/latents_std` -> pack -> `image_latents`.
   In denoise loop: `latent_model_input = cat([latents, image_latents], dim=1)`
   (concat along image-sequence). `img_shapes` carries BOTH geometries:
   `[(1, H//16, W//16), (1, condH//16, condW//16)]` (vae_scale_factor 8 * patch
   2 = 16). After DiT, keep only the first (noise) span for the velocity;
   `_unpack_latents` uses target H/W.

Sampler: FlowMatchEulerDiscreteScheduler, `calculate_shift` (dynamic mu from
image_seq_len: base_seq 256 / max 4096, base_shift 0.5 / max_shift 1.15).
Rapid-AIO = 4 steps, CFG-free. CONFIRM exact step schedule from the Phr00t
workflow json (fixed-textencode node).

## Text encoder: Qwen2.5-VL-7B (qwen2_5_vl)

- LM (7B): hidden 3584, 28 layers, 28 q-heads / 4 kv-heads (GQA), head_dim 128,
  intermediate 18944, vocab 152064, rms_norm_eps 1e-6, rope_theta 1e6, SwiGLU,
  QKV bias (Qwen2 has attn bias; Qwen3 does not -> NOT identical to
  z_image::Qwen3Block; needs bias + no q/k-norm). MRoPE (text uses 3D sections
  but for pure-text positions collapses). CONFIRM dims from GGUF metadata.
- Vision tower (mmproj-f16.gguf): ViT patch-embed + windowed attention + the
  merger MLP that projects vision tokens to LM hidden 3584. Needed ONLY for the
  edit image channel. Standard Qwen2.5-VL vision config (CONFIRM from mmproj
  GGUF metadata: depth, hidden, heads, patch, spatial_merge_size).
- We consume the LM's last hidden state (not logits). Encoder runs bf16 acts
  (residual-stream overflow guard, same lesson as umT5/Qwen3 [[feedback...]]).

### Vision tower SPEC (verified vs transformers qwen2_5_vl + mmproj GGUF)

GGUF KV (clip arch, 519 tensors): D=1280, depth=32, heads=16 (head_dim 80),
patch 14, image_size 560, FFN 3420 SwiGLU(use_silu), merge=2, projection 3584,
eps 1e-6, n_wa_pattern=8, image_mean/std=CLIP defaults. Tensor keys: `v.blk.N.
{attn_q,k,v,out (w+bias), ffn_gate,up,down (w+bias), ln1, ln2 (w only)}`,
`v.post_ln.weight` (= merger ln_q RMSNorm 1280), `mm.0.{w[5120,5120],b}` GELU
`mm.2.{w[3584,5120],b}`, `v.patch_embd.weight`+`.weight.1` (temporal kernel split
[1280,3,14,14] each, SUM them -- both temporal slots hold identical image data).
- FULL-ATTENTION blocks = {7,15,23,31} (index==7 mod 8); other 28 = windowed.
- window: 112px = 4 merged tokens = 8 raw patches/side; vit_merger_window=4.
- Patchify: image [3,H,W] (smart_resize to mult of 28) -> rows [N=t*gh*gw, 1176]
  (1176=3*2*14*14). t=1 image; gh=H/14, gw=W/14. Row order is MERGE-UNIT-MAJOR
  (permute (0,2,5,3,6,1,4,7)): each consecutive 4 rows = one 2x2 merge unit, NOT
  raster. Normalize (px/255-mean)/std per channel first.
- patch_embed = single Linear [1176->1280] (W=concat weight,weight.1), no bias.
- 2D rope: rot dim head_dim/2=40, inv_freq 1/(1e4^(2i/40)) i=0..19 (20 freqs).
  pos_ids per token = (hpos,wpos) in the SAME merge-unit order. freqs[N,40] =
  [freq(hpos)(20) ++ freq(wpos)(20)]. cos/sin [N,80] = cat(freqs,freqs) =
  [h,w,h,w] 20 each. Apply = ROTATE_HALF (split at 40, NeoX style), fp32, q/k only.
- window reorder: hidden.reshape(N/4,4,D)[window_index].reshape(N,D); reorder
  rope rows the SAME way BEFORE building cos/sin. window_index is a permutation
  over the N/4 merge units. Restore after merger via argsort(window_index).
- block (pre-norm, RMSNorm ln1/ln2 eps1e-6): h+=attn(ln1(h)); h+=mlp(ln2(h)).
  attn: separate q/k/v (bias), rotate_half rope on q/k, segmented SDPA (block-
  diagonal over cu_seqlens: full=[0,N] for fullatt blocks, else per-window
  cu_window_seqlens; raw-patch units), scale 80^-0.5, out proj (bias). mlp =
  down(silu(gate(x))*up(x)) all-bias SwiGLU.
- merger: ln_q RMSNorm(1280) per token -> view(N/4, 5120) (the 4 contiguous
  merge-unit tokens) -> Linear(5120,5120) -> GELU(erf) -> Linear(5120,3584).
  Output [N/4, 3584] = LM embeddings; unsort to raster; scatter into the LM
  input embeds at the <|image_pad|> token positions (count must == N/4).

## VAE: Wan-family 3D causal-conv KL (autoencoder_kl_qwenimage)

CONFIRMED reuse path: diffusers `AutoencoderKLQwenImage` is built from
`QwenImageCausalConv3d / QwenImageResidualBlock / QwenImageResample /
QwenImageMidBlock / QwenImageRMS_norm / QwenImageAttentionBlock` = the EXACT
Wan VAE family already ported in `wan/vae.rs` (parametric `WanVaeConfig` with
base_dim / decoder_base_dim / z_dim / dim_mult / temperal_(up|down)sample /
latents_mean / latents_std, plus full encoder+decoder + loader). Qwen-Image VAE
= Wan2.1 VAE shape: `base_dim=96`, `z_dim=16`, `dim_mult=[1,2,4,4]`,
`num_res_blocks=2`, `temperal_downsample=[F,T,T]` (decoder mirrors). Still
image => frame=1 (one temporal chunk, no halo, like the tiny-VAE single-chunk
path).

Port = THIN ADAPTER, not a new VAE:
1. A `qwen_image_vae() -> WanVaeConfig` constructor (Wan2.1 dims + Qwen's own
   `latents_mean`/`latents_std` 16-vectors). CONFIRM base_dim/decoder_base_dim
   and the mean/std vectors from the calcuis GGUF metadata + diffusers config
   (Wan2.2 splits decoder_base_dim; Qwen-Image likely single base_dim 96).
2. A tensor-name rename map: diffusers `encoder.*/decoder.*/quant_conv/
   post_quant_conv` (or the calcuis pig-gguf key form) -> the keys
   `wan/vae.rs`'s loader expects. CONFIRM exact keys from the GGUF once landed
   (calcuis "pig" packs sometimes rename; may differ from diffusers).
3. Decode entry reuses the wan VAE decode (one heavy submit, VRAM tiling). Encode
   (needed for the EDIT ref-latent channel) reuses the wan encoder + the
   `(z-mean)/std` normalize, `argmax`/mode sampling (NOT random) per the edit
   pipeline.
- latents_mean/std per-channel [16]; used by decode denorm AND edit encode.

### Edit integration SPEC (verified vs diffusers pipeline_qwenimage_edit + qwen2_5_vl)

Single-image edit path (Rapid = this variant, CFG-free).
- TEMPLATE (drop_idx=64): `<|im_start|>system\nDescribe the key features of the
  input image (color, shape, size, texture, objects, background), then explain
  how the user's text instruction should alter or modify the image. Generate a
  new image that meets the user's requirements while maintaining consistency
  with the original input where appropriate.<|im_end|>\n<|im_start|>user\n
  <|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>
  assistant\n`. The single `<|image_pad|>` expands to N/4 placeholders (= post-
  merge vision token count = gh*gw/4 for the ViT grid). Drop the FIRST 64 tokens
  of hidden_states[-1]; image-pad hiddens SURVIVE the drop (feed the DiT text).
- LM MRoPE position_ids (3 channels t,h,w; head_dim sections [16,24,24]): text
  before image = 0..k-1 on all 3 channels; image tokens get t=cur (const),
  h=cur+row, w=cur+col (row-major over the MERGED gh/2 x gw/2 grid); then
  cur += max(gh,gw)//2 and trailing text resumes sequential. (t2i collapses to
  1-axis since no image; edit needs the real 3-axis MRoPE.)
- SCATTER: build inputs_embeds from the token table, overwrite the contiguous
  image_pad block with the vision-tower output rows (raster order), then run the
  28 LM layers + output_norm as usual.
- VAE channel: image_latents = encode(ref).mode() (=mean, ch 0..16); normalize
  (z-mean)/std; pack channel-major 2x2 -> [cond_seq, 64] (SAME packing as noise);
  in the DiT image stream concat [noise_tokens ++ ref_tokens] along seq (noise
  FIRST). img_shapes=[(1,gh,gw) noise, (1,cgh,cgw) ref]; DiT RoPE builds vid
  freqs per block in that SAME order and concatenates. proj_out over the whole
  image stream; VELOCITY = first noise_seq (gh*gw) tokens, drop the ref tail.
- PREPROCESS: same source image. VAE side = calculate_dimensions(1024*1024,
  ratio) rounded to mult of 32. ViT side = smart_resize to mult of 28 (factor
  patch*merge=28), min 56*56 / max 28*28*1280 px. So VAE and ViT see DIFFERENT
  resolutions off the same image (32-grid vs 28-grid). Replicate exactly for
  edit parity.
- CFG-FREE: true_cfg_scale=1.0, no negative branch, guidance=None (Rapid
  guidance_embeds=False; DiT has no guidance embedder tensors). One forward/step.
- CAVEAT: pyref get_rope_index is the newer mm_token_type_ids variant; numeric
  positions match the classic scheme for a single leading image. Validate the
  host-side position-id builder against the encoder-with-image pyref.

## Reuse map

- Module layout mirrors `ideogram4/`: dit, loader, manifest, mrope (3-axis but
  COMPLEX not half-rot -> closer to `wan/rope3d.rs`), packing, pipeline,
  sampler, text_encoder, vae, t_embedder.
- Matmul/SDPA/i8 DP4A, MemArbiter, Workspace, buffer pool: reuse as-is.
- RoPE: complex interleaved-pair -> reuse `wan/rope3d.rs` machinery, NOT the
  ideogram half-rot mrope. 3 axes (frame,h,w) with scale_rope neg/pos split.
- VAE conv stack: reuse `wan/vae.rs` (same Wan 3D causal KL family).
- Encoder block: NEW Qwen2.5 block (bias, GQA, no qk-norm) - adapt
  `z_image::text_encoder` Qwen3 block by adding qkv bias + dropping qk-norm.
- Vision tower: NEW (no precedent in repo).

## Per-step weight streaming (the thesis)

20B DiT at Q4_K_M ~13GB > 8GB VRAM, so the whole DiT cannot co-reside. Blocks
stream through `MemArbiter` residency per step (4 steps => weights move 4x).
This is thinfer's core design, not a workaround. Expect I/O-bound; drive perf
from e2e+TRACE, localize with gated telemetry, attack structurally (prefetch
block N+1 during block N; pin hot small weights; Q4 on-the-fly dequant in the
matmul). Do NOT degrade steps/quality to mask perf.

## Parity strategy

- pyref dequantizes the SAME GGUF the engine loads (ideogram discipline) =>
  isolates kernel correctness from quant loss. Canary = Q8_0, tiny dims
  (256x256), RAM-safe (NEVER fp32 full-res pyref; stream, mode-only VAE).
- Mirror ideogram4 conformance tests: encoder_parity, dit_parity, vae_parity,
  e2e_parity, parity_util. Validation order: op conformance -> q8 256 pyref e2e
  -> q4 default perf e2e. Serial GPU only.

## Build order

1. [in progress] Foundation: manifest, module scaffold, plan, GGUF metadata +
   tensor-name audit (rename maps) once weights land. Confirm all CONFIRM tags.
2. VAE decode (image-only, frame=1) + vae_parity. Smallest closed loop.
3. Text encoder LM (no vision) + encoder_parity (text-only prompt path).
4. MMDiT block + full DiT + dit_parity (feed reference text embeds + latents).
5. t2i pipeline + sampler + e2e_parity (text-only; empty/no edit image).
6. Edit path: VAE encode + vision tower + latent concat; edit e2e_parity.
7. Per-step streaming residency + perf pass. Q4_K_M default.
8. CLI/serve wiring (thinfer-app model id + defaults), web later.

## CONFIRM list (resolve from GGUF metadata before coding each piece)

- DiT: [CONFIRMED from Q8_0 GGUF] `general.architecture=qwen_image`, 1933
  tensors = 60 layers (32/block + 13 top-level). DIM 3072, IN/OUT 64, img_mod
  3072->18432 (6*dim), FFN 12288 (4*dim), norm_out.linear 3072->6144. NO
  guidance/additional_t/layer3d embedder tensors -> all those flags FALSE.
  Tensor keys are 1:1 with diffusers module paths (`transformer_blocks.{i}.
  attn.{to_q,to_k,to_v,to_out.0,add_{q,k,v}_proj,to_add_out,norm_{q,k},
  norm_added_{q,k}}`, `img_mod.1`, `txt_mod.1`, `img_mlp.net.{0.proj,2}`,
  `txt_mlp...`, `img_in`, `txt_in`, `txt_norm`, `norm_out.linear`, `proj_out`,
  `time_text_embed.timestep_embedder.linear_{1,2}`) -> NO rename map needed.
  Quant layout: big matmul weights Q8_0, all biases+norms+img_in/proj_out/
  time_embed F16 (same sensitive-tensor pattern as ideogram).
  NB: GGUF KV header has only 3 keys (architecture/quant/file_type) - NO
  hyperparams; everything is read from tensor names/shapes.
- Encoder: layer count, GQA kv heads, rope_theta, bias presence, tensor keys.
- mmproj: [CONFIRMED] arch=clip/clip-vision, projector=qwen2.5vl_merger; 32
  blocks, hidden 1280, 16 heads (head_dim 80), patch 14, image_size 560, FFN 3420
  SwiGLU (use_silu), ln eps 1e-6, projection_dim 3584, n_wa_pattern=8 (every 8th
  block FULL attention, rest WINDOWED). 519 tensors: v.patch_embd.weight(.1),
  v.blk.N.{attn_q,k,v,out (w+bias), ln1, ln2, ffn_gate,up,down (w+bias)},
  mm.N.{weight,bias} (merger MLP -> 3584). image_mean/std in KV. NEW subsystem
  (~encoder-sized effort): 2D rope, windowed attn, spatial-merge(2x2)->3584.
- VAE ENCODE SOURCE (step-6 BLOCKER, unresolved): the calcuis pig VAE GGUF is
  DECODE-ONLY (no encoder.*). The edit path VAE-encodes the ref image, so it needs
  encoder weights elsewhere -- a full Qwen-Image VAE (diffusers safetensors
  `Qwen/Qwen-Image` vae/, has encoder; NEW download) or an encoder-bearing GGUF.
  wan/vae.rs already has `register_encoder` + the encoder forward, so once a source
  lands it's a thin adapter like decode. DECISION NEEDED (footprint). t2i path does
  NOT need encode (works today).
- VAE: z_dim, base dim, latents_mean/std vectors, tensor keys.
- Sampler: exact 4-step sigma/shift schedule from the Phr00t workflow.
