# ltx2-rapid port plan (ACTIVE)

Model id: `ltx2-rapid`. A community merge (Phr00t/LTX2-Rapid-Merges v62) on the
LTX-2 **19B** line: stock LTX-2 19B + baked distill + detailer + native-I2V LoRAs.
NOT a new arch. It is a checkpoint variant on our shipped LTX-2.3 pipeline
(`ltx-plan.md`). Uncensored line: keep durable text (this file, code, commits)
neutral -- refer to it by version id + URL only.

Weights source = the GGUF conversion (NOT Phr00t's fp8 safetensors; fp8 is
off-limits and we want GGUF anyway). No upstream reference impl exists for a merge,
so validation = component parity of the net-new pieces vs LTX-2 19B ref code + a
ComfyUI-same-GGUF e2e eyeball. The small-dim e2e doubles as the perf bench.

## Weights

| Role | Repo / file | Notes |
|---|---|---|
| DiT Q5_K_M (default) | 3ndetz/LTX2-Rapid-Merges-GGUF `nsfw/ltx2-phr00tmerge-nsfw-v62/...-Q5_K_M.gguf` | 14.2G, mixed-K (attn_v/ffn_down -> Q6_K); our routing handles it |
| DiT Q4_K_M (compare) | " `...-Q4_K_M.gguf` | 12.6G; measure if it buys anything (weight-quant doesn't help compute, only streaming) |
| DiT F16 (canary) | " `...-F16.gguf` | 37.8G; quality reference |
| Connector + FE V1 aggregate embed | `ltx-2-19b-embeddings_connector_distill_bf16.safetensors` | 19B connector; LOCATE repo (Kijai/LTXV2_comfy or Comfy-Org/ltx-2) |
| Video VAE | `LTX2_video_vae_bf16.safetensors` | LOCATE repo (Kijai/LTXV2_comfy). Confirm identical to our LTX-2.3 video VAE |
| Audio VAE + vocoder | `LTX2_audio_vae_bf16.safetensors` | " |
| Text encoder | Gemma-3-12B (cached: unsloth/gemma-3-12b-it-qat GGUF) | SAME as LTX-2.3; caption_channels 3840 |

DiT GGUF confirmed (embedded `config` KV, 3510 tensors, arch `ltxv`, file_type 17 =
Q5_K): `AVTransformer3DModel`, num_layers 48, heads 32x128 video / 32x64 audio,
in/out 128, cross_attn 4096/2048, `qk_norm rms`, `rope_type split`, theta 1e4,
max_pos [20,2048,2048], causal_temporal, middle_indices, `use_embeddings_connector`,
`connector_num_layers 2`, `connector_num_attention_heads 30`,
`connector_num_learnable_registers 128`, `av_cross_ada_norm true`,
`use_audio_video_cross_attention true`. Scheduler RectifiedFlow / LinearQuadratic
(overridden to the distilled 8-step table below).

## Confirmed weight structure (v62 GGUF header, 3510 tensors, all diffusers names)

Per-block (`transformer_blocks.0..47.*`) = SAME module set as 22B: attn1 (video
self), attn2 (video-text cross), audio_attn1, audio_attn2, audio_to_video_attn,
video_to_audio_attn, ff, audio_ff. Deltas vs 22B, from actual shapes:
- `scale_shift_table [4096, 6]` (22B = 9) + `adaln_single.linear [4096, 24576]`
  (6*4096) -> **6-way block modulation** (scale/shift/gate x msa,mlp), no extras.
- **NO `to_gate_logits`** in any attn -> **no gated attention**.
- **NO `prompt_scale_shift_table`** -> **no prompt-AdaLN** (no continuous-sigma
  second modulation).
- `scale_shift_table_a2v_ca_{video[4096,5],audio[2048,5]}` present (av-cross 5-way,
  same as 22B). audio blocks mirror video at width 2048.
- Block matmuls mixed **Q5_K / Q6_K** (to_v + ff.net.2 -> Q6_K); qk/bias/norms F16.
Top-level (video + audio each): `patchify_proj`, `adaln_single.*`
(timestep_embedder 256->4096->4096, linear 4096->24576), `proj_out`,
`scale_shift_table [*,?]` (final 2-way), `av_ca_*_adaln_single.*` (av gates +
scale-shift). PLUS the V1 signature absent in 22B:
- `caption_projection.linear_1 [3840,4096] / linear_2 [4096,4096]` (F16) -- video
  in-transformer PixArtAlpha proj (3840 -> 4096 -> 4096, gelu-tanh between).
- `audio_caption_projection.linear_1 [3840,2048] / linear_2 [2048,2048]` (F16).
**ZERO connector / learnable_registers / aggregate_embed tensors in the GGUF** ->
for 19B those live in the connector safetensors (unlike 22B, where the connector
blocks live in the DiT GGUF). Loader split differs accordingly.

## Deltas from our shipped LTX-2.3 (22B) -- the whole net-new surface

Config-driven in upstream: one `LTXModel` class; 19B vs 22B = config + two branches.

CONNECTOR SAFETENSORS confirmed (Kijai/LTXV2_comfy/text_encoders/
ltx-2-19b-embeddings_connector_distill_bf16.safetensors, 59 tensors, all bf16):
- `text_embedding_projection.aggregate_embed.weight [3840, 188160]` -- single FE V1
  aggregate embed, **NO bias**.
- `model.diffusion_model.{video,audio}_embeddings_connector.*`: TWO connectors, BOTH
  inner **3840** (30 heads x 128; audio falls back to video dims, NOT 2048), **2**
  `transformer_1d_blocks`, ff 15360 (4*3840), each own `learnable_registers
  [128,3840]`. Block = attn1 (to_q/k/v/out + q/k_norm) + ff.net.0.proj/net.2;
  **NO `to_gate_logits` -> UNGATED** (22B connector is gated). Both connectors take
  the shared FE V1 aggregate as input; each frames with its own registers; outputs
  3840; then the in-DiT `{,audio_}caption_projection` maps 3840 -> 4096 / 2048.

1. **FeatureExtractor V1 (largest).** 22B = V2 (per-token RMS -> dual bias'd
   aggregate embeds -> 4096/2048 directly). 19B = V1:
   - per-batch/per-layer masked mean + min/max **range** norm: `8*(x-mean)/(range+eps)`
     (NOT per-token RMS), then flatten `[B,T,3840,49] -> [B,T,3840*49]` (flat = d*49+l).
   - **single** `aggregate_embed` Linear `[3840*49 -> 3840]`, **bias=false**, ONE
     shared stream returned for both video and audio (is_av).
   - caption projection then lives IN THE TRANSFORMER: `PixArtAlphaTextProjection`
     (linear_1 -> gelu-tanh -> linear_2) 3840 -> 4096 (video) and 3840 -> 2048 (audio).
   - Selector: V1 when the four V2 keys are ABSENT from config (v62 omits them).
     Gate on `caption_proj_before_connector == false`.
   - Ref: `ltx-core .../gemma/feature_extractor.py:77-97` (V1), `:12-45` (range norm),
     `text_projection.py:31-38` (PixArtAlpha), `model_configurator.py:178-198`,
     `encoder_configurator.py:65-101,79,146-174`.
2. **DiT block version flags.** v62 config omits `apply_gated_attention` and the
   prompt-AdaLN (`prompt_scale_shift_table`, continuous-sigma prompt modulation)
   that our 22B block hard-wires ON. Make both config-gated (off for 19B). VERIFY
   the real per-block sublayers by dumping GGUF tensor names at load (trust names,
   not the sparse config). Ref: `model_configurator.py:39,41,68,71`,
   diffusers `transformer_ltx2.py:1121-1122,1173`.
3. **Connector config.** Same `Embeddings1DConnector` module; 19B runs 2 layers /
   30 heads / 3840 inner (ours: 8 layers / 32 heads / 4096). Parameterize
   num_layers + inner_dim + heads if we baked the 22B values. `caption_proj` order:
   projection is applied around the connector per `caption_proj_before_connector`.
   Ref: `embeddings_connector.py:72-187,190-250`.
4. **Native I2V conditioning (no adapter).** I2V is standard latent conditioning,
   version-agnostic, reusable for both LTX lines (our shipped path is t2v-only ->
   net-new but shared):
   - frame 0: `VideoConditionByLatentIndex` -- VAE-encode image, patchify, OVERWRITE
     `clean_latent[:, start:stop]`, set `denoise_mask = 1 - strength`.
     (`conditioning/types/latent_cond.py:9-43`)
   - other frames: `VideoConditionByKeyframeIndex` -- APPEND keyframe tokens with
     rope positions offset by frame_idx. (`keyframe_cond.py:10-84`)
   - orchestration `ltx-pipelines/utils/helpers.py:132-166`. I2V Gemma system prompt
     `gemma_i2v_system_prompt.txt` (vs t2v).
5. **`ltx2-rapid` config variant + loader** reading layers/dims/flags from the
   embedded config KV; mixed-K Q5_K routing (already have). RoPE / VAE / audio /
   sampler need NO structural change -- config + file selection only.
6. **Sampler**: distilled 8-step (baked distill LoRA), NOT the LTX-2 base 40-step.
   Sigmas (from the workflow ManualSigmas, = our LTX-2.3 stage-1 table):
   `[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]`.
   Single-stage (workflow has no upscaler). X0 Euler + CFG-off (as shipped).
   OPEN: the workflow's `LTXVNormalizingSampler` per-step factors
   `['1,1,1,1,1,1,1,1', '1,1,0.25,1,1,0.25,1,1']` (0.25 at steps 3,6) -- determine
   whether this is a material guidance/rescale schedule to replicate or a
   ComfyUI-side nicety. Investigate before locking the sampler.

## EXACT 19B forward spec (from LTX-2 source; implement from this)

Block (`transformer.py` BasicAVTransformerBlock, 19B config `cross_attention_adaln
=False`, `apply_gated_attention=False`):
- 6-way `scale_shift_table [dim,6]`; `ada = table + adaln_single(timestep).reshape
  (B,T,6,dim)`, sliced. ORDER (shift-first): `[0]msa_shift [1]msa_scale [2]msa_gate
  [3]mlp_shift [4]mlp_scale [5]mlp_gate`. modulate = `rms_norm(x)*(1+scale)+shift`.
- **attn2 (video-text cross) = RAW, no modulation**: `vx += attn2(rms_norm(vx))`
  (post-self-attn residual normed; no q-mod, no prompt-AdaLN kv-mod). Same for
  `audio_attn2`. (22B's cross-q 3-mod + prompt 2-mod DO NOT EXIST here.)
- **No gated attention**: `to_gate_logits=None`; attn out -> to_out directly.
- av-cross 5-way, SAME as 22B: video table `[0]a2v_scale[1]a2v_shift[2]v2a_scale
  [3]v2a_shift[4]a2v_gate`; audio table same slots, `[4]=v2a_gate`. a2v: `vx +=
  a2v_attn(ada_zero(vx,vid0,vid1), kv=ada_zero(ax,aud0,aud1))*vid4_gate`; v2a mirror.
- Sublayer order: (1) video self [msa] -> (2) video text cross [raw] -> (3) audio
  self [audio 0:3] -> (4) audio text cross [raw] -> (5) a2v -> (6) v2a -> (7) video
  FFN [mlp] -> (8) audio FFN [audio 3:6]. VideoOnly = 1,2,7 only.

FE V1 (`feature_extractor.py` _norm_and_concat_padded_batch): per (batch, layer)
reduce mean/min/max over BOTH token AND hidden axes (masked); `range=max-min`;
`normed = 8*(x-mean)/(range+eps)`, eps 1e-6. Flatten `[B,T,D,L] -> [B,T,D*L]`,
flat index `d*49 + l` (L fastest, L=49, D=3840 -> 188160). SINGLE
`aggregate_embed` Linear 188160->3840, bias=FALSE; is_av -> same tensor to video +
audio. NO sqrt rescale (V2-only).

caption_projection: applied ONCE before the block loop (not per-block).
PixArtAlpha `linear_1(3840->inner) -> gelu(tanh) -> linear_2(inner->inner)`; video
inner 4096, audio 2048. Chain: FE V1 -> connector@3840 -> caption_projection ->
attn2 KV.

I2V frame-0 latent replace: `clean_latent[:,0:ftok] = patchify(vae_enc(image))`;
`denoise_mask[:,0:ftok] = 1-strength`. Per-step (Euler): `timesteps =
denoise_mask*sigma`; `denoised = to_denoised(latent,v,timesteps)`; `denoised =
denoised*denoise_mask + clean*(1-denoise_mask)`; `latent = step(latent,denoised,
sigmas,i)`. Initial noising also `lerp(clean, noised, denoise_mask)`.

## Reuse (shipped LTX-2.3, parity-proven -- do NOT touch)

DiT block (48L dual-stream AV, 5 attn, X0), split/half-rot rope + `op_rope`,
video VAE, audio VAE + BigVGAN/BWE vocoder (fp32), Gemma-3-12B encoder, GGUF loader
+ RenamedSource + mixed-K routing + F16->bf16 path, MemArbiter/residency/phase
eviction/VRAM tiling, i8 DP4A + mixed-prec f16 SDPA, app/serve/web stack.

## Implementation approach (executable; code fully mapped 2026-07-06)

Note: LTX has NO `LtxPipeline::load/generate` in the models crate; the chain
(tokenize -> GemmaEncoder -> FE -> connector -> DitModel denoise -> VAE) is
assembled by the DRIVER (conformance e2e test + thinfer-app executor). Mirror the
existing LTX-2.3 driver for ltx2-rapid.

`LtxVariant` descriptor (new, in mod.rs or dit.rs), threaded through register +
forward; `ltx_2_3_22b()` reproduces current behavior byte-identical:
- `gated_attn` (22B true / 19B false), `cross_adaln` (t/f), `prompt_adaln` (t/f),
  `n_block_mod` (9/6), `caption_proj` (f/t), `fe_v1` (f/t),
  connector spec {layers 8/2, inner 4096/3840, audio_inner 2048/3840, gated t/f,
  from_gguf t / from_safetensors t}.

Edit points:
- `dit::attention()`: add `gated: bool`. When false, skip `to_gate_logits`
  biased_proj + GatedHeadMulF32; `to_out(sa)` directly. (19B AttnHandles have no
  gate_w/gate_b -> make gate handles Option, or a separate register path.)
- `dit::block_forward()`: add variant. 19B: video/audio self = ada_zero(msa) ->
  attention(gated=false) -> gate_residual(msa_gate) [same]. text cross = plain:
  `vca = attention(VIDEO_CROSS, rms_norm(vx1), kv=vtext, gated=false, NO cq/ckv
  op_modulate)`, `vx2 = vx1 + vca` (op_add, not gate_residual). av-cross + FFN
  identical to 22B. Skip all cq_/ckv_ modulate calls when !cross_adaln.
- `cond.rs`: `register_timestep` skip prompt modules when !prompt_adaln;
  `compute_shared_timestep` v_main/a_main = n_block_mod*D, skip prompt forwards;
  `read_block_tables` scale_shift = n_block_mod*D, skip prompt tables;
  `assemble_stream` when n_block_mod==6 fill only msa(0,1,2)+mlp(3,4,5), leave
  cq_/ckv_ empty (block_forward won't read them for 19B).
- `dit::register_block` / `register_attn`: gate handles optional (19B lacks them);
  scale_shift table sized by variant.
- NEW caption_projection (19B only): register `caption_projection.linear_1/2`
  (F16 [3840,4096],[4096,4096]) + `audio_caption_projection.*` ([3840,2048],
  [2048,2048]); apply ONCE (PixArtAlpha: matmul_adaln bf16 -> pipes.gelu (tanh) ->
  matmul) to connector output [1024,3840] -> vtext[1024,4096]/atext[1024,2048].
  Put in dit.rs (needs DitPipelines) or a small new module; DitModel gains
  Option<CaptionProjHandles>.
- `connector.rs`: variant path -- inner 3840, 2 layers, UNGATED block (skip gate),
  weights from the connector SAFETENSORS (bf16, prefix `model.diffusion_model.
  {video,audio}_embeddings_connector.transformer_1d_blocks.N.*` + `.learnable_
  registers`); FE V1 aggregate `text_embedding_projection.aggregate_embed.weight`
  (no bias) -> single stream -> fed to BOTH connectors (each frames w/ own
  registers). Reuse `feature_extractor_v1_flatten` (done). connector out_dim=inner.
- loader/source: DiT GGUF (mixed Q5_K/Q6_K; caption_proj F16) + connector
  safetensors + Kijai VAEs (verify tensor keys vs video_vae/audio_vae; add rename
  if comfy-named). Probe BlockQuantKinds handles mixed-K already.
- driver: a conformance e2e (small dims -> perf rollup) + thinfer-app executor
  arm. Sigmas = the shipped 8-step table. Single-stage (no upscaler). CFG-off.

## Build phases (each: parity gate + budget gate; smallest closed loop first)

- **P0 scaffold + confirms.** Locate + download connector/VAE repos; dump v62 GGUF
  DiT block tensor names (which per-block params exist -> confirm no gate / no
  prompt-AdaLN). Read our `ltx/` rust (config, loader, FE V2, DiT block, connector,
  pipeline). Add `ltx2-rapid` config variant scaffold; build green.
- **P1 FE V1 + caption projection + 2-layer connector** -> encoder_parity vs 19B
  ref (<=10GB pyref, quant-aware Gemma). Range-norm + single aggregate_embed +
  PixArtAlpha proj + connector at 3840/2/30.
- **P2 DiT 1-block with 19B flags** -> dit_parity (gate off gated-attn/prompt-AdaLN
  as the tensor names dictate; everything else reuses).
- **P3 t2v e2e health = first deliverable.** Single-stage 8-step -> video VAE ->
  frames. Component parity + ComfyUI-same-GGUF eyeball. Doubles as perf rollup.
- **P4 native I2V** -> frame-0 latent replacement + keyframe guide; I2V eyeball.
- **P5 audio tail** (reuse) -> joint AV e2e; mux.
- **P6 perf + product wiring.** i8 DP4A + f16 SDPA on the DiT; phase eviction;
  Q4_K_M A/B (measure). Then `VideoModelId::Ltx2Rapid` + executor + CLI + serve
  (VideoSpec, image input for I2V) + web, CLI/web parity. serve redeploy (ASK).

## Upstream source map (third-party, read these)

- Model + config: `third-party/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/`
  `model.py`, `model_configurator.py`, `rope.py`.
- Text cond: `.../text_encoders/gemma/feature_extractor.py`,
  `encoder_configurator.py`, `text_projection.py`, `embeddings_connector.py`,
  `encoders/base_encoder.py` + `prompts/gemma_{i2v,t2v}_system_prompt.txt`.
- I2V conditioning: `.../conditioning/types/{latent_cond,keyframe_cond}.py`;
  `packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py`.
- Sampler/params: `ltx-pipelines/.../utils/constants.py` (LTX_2_PARAMS vs
  LTX_2_3_PARAMS, detect_params on `model_version` metadata).
- ComfyUI workflow (settings/wiring reference): `third-party/LTX2-Rapid-Merges/
  LTXV-DoAlmostEverything-v3-GGUF.json`.
- diffusers cross-check: `third-party/diffusers/.../transformers/transformer_ltx2.py`
  (defaults = the 19B/2.0 shape; note rope_type default interleaved there but the
  v62 config says split).
