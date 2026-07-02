# HunyuanVideo 1.5 port (step-distilled, 8-step, 480p I2V)

ACTIVE per-model plan. Goal: FastWan-class speed, better faces, e2e parity-checked,
RAM+VRAM budget respected, full CLI + serve + web wiring. Reference clone:
`third-party/HunyuanVideo-1.5` (commit 60783e7). Do NOT restart the user's server or
run the GPU without asking.

## Architecture (from per-version config.json, NOT repo Python defaults)

DiT = **dual-stream MMDiT** (Flux/HV1.0 lineage), step-distilled `480p_i2v_step_distilled`:
- hidden 2048, 16 heads, head_dim 128. **54 double-stream blocks, 0 single-stream.**
- in/out 32ch. **patch [1,1,1]** (no patchify; `img_in` is a 1x1x1 Conv3d). VAE does all
  compression.
- RoPE3D **interleaved-pair** (Wan convention, NOT Qwen3 half-rot), axial split
  (t,h,w)=(16,56,56), theta 256. RoPE on IMAGE tokens only (text not rotated).
- qk RMSNorm per-head eps 1e-6, qkv_bias true. mlp ratio 4.0, gelu_tanh.
- adaLN-Zero **6-chunk** modulation per stream (shift/scale/gate x2), `apply_gate`.
- Text enters as **joint tokens** (concat img+txt q/k/v, Flux-style), NOT cross-attn.
  text_states_dim 3584 (Qwen2.5-VL). 2-layer `SingleTokenRefiner` on text first.
- time embed: sinusoidal->MLP. **MeanFlow**: a 2nd timestep `timestep_r` via `time_r_in`
  summed into the modulation `vec`. CFG OFF (guidance_scale 1.0, single forward/step).
- ~8.3B params.

Text/vision encoders:
- **Qwen2.5-VL-7B** (hidden 3584) -- text + I2V image grounding. ALREADY PORTED in
  `qwen_image/{text_encoder,vision}.rs` (same checkpoint). Reuse.
- **ByT5 glyph** (Glyph-SDXL-v2, glyph_byT5_v2) -- in-frame text rendering. New, optional.
- **SigLIP vision** (sigclip_vision_patch14_384) -- I2V semantic tokens. New, optional.

VAE = `AutoencoderKLConv3D`, causal-conv3d (Wan VAE family):
- 32 latent ch, **16x spatial / 4x temporal**, blocks [128,256,512,1024,1024],
  scaling_factor 1.03682, RMS_norm channel-first, PatchCausalConv3d temporal-split tiling.

Scheduler = `FlowMatchDiscreteScheduler`, Euler order-1, `prev = x + v*dt`,
flow_shift 7.0, num_steps 8/12, MeanFlow distill, CFG off.

I2V conditioning (latent-concat, 65ch DiT input):
- ref image -> VAE encode -> cond_latents (*scaling). Broadcast to all frames, then
  ZERO all frames except frame 0. + binary mask channel (1 at frame0, 0 else).
- DiT input = noise(32) + image_cond(32) + mask(1) = 65ch; `img_in` Conv3d extra 33ch
  zero-init. PLUS SigLIP semantic tokens as extra joint tokens.

## VERIFIED SPEC (from checkpoint headers + reference modules, 2026-06-29)

Ground-truthed against the actual safetensors headers (NOT repo Python defaults; a
recon agent that read defaults reported the WRONG version: hidden 3072 / 24 heads /
20+40 blocks / patch[1,2,2] / 9ch / text 4096 -- all wrong, discard). The lightx2v
DiT = `hy1.5_t2v_480p_lightx2v_4step.safetensors`, 1793 BF16 tensors.

DiT tensor names (confirms config.rs exactly):
- `img_in.proj.{weight,bias}` = [2048,65,1,1,1]/[2048] (1x1x1 conv, 65ch in).
- `time_in.mlp.{0,2}` 256->2048->2048. NO time_r_in/guidance/vector_in (meanflow OFF).
- `txt_in`: `input_embedder` [2048,3584]; `t_embedder.mlp.{0,2}`; `c_embedder.{linear_1
  [2048,3584],linear_2 [2048,2048]}`; `individual_token_refiner.blocks.{0,1}.{norm1,
  norm2 (LayerNorm affine, weight+bias, eps1e-6), self_attn_qkv [6144,2048]+bias,
  self_attn_proj, mlp.{fc1 [8192,2048],fc2}, adaLN_modulation.1 [4096,2048] = 2 GATES
  only}`. Refiner has NO q/k norm (Identity). refiner heads = 2048/128 = 16.
- `double_blocks.{0..53}` (54, zero single): per img/txt stream: `{s}_mod.linear
  [12288,2048]+bias` (=6 chunks, ModulateDiT = Linear(SiLU(vec)), order shift_msa,
  scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp); `{s}_attn_{q,k,v} [2048,2048]+bias`
  (SEPARATE, not fused); `{s}_attn_{q,k}_norm.weight [128]` (per-head RMSNorm, weight
  only, eps1e-6); `{s}_attn_proj +bias`; `{s}_mlp.{fc1 [8192,2048],fc2}+bias`
  (gelu_tanh). norm1/norm2 = param-less LayerNorm (no tensors), eps1e-6.
- `cond_type_embedding.weight [3,2048]` (add row 0 to txt tokens). `byt5_in`/`vision_in`
  present but masked-zero for T2V -> OMIT. `final_layer.{norm_final(param-less),
  adaLN_modulation.1 [4096,2048]=shift+scale, linear [32,2048]}`.

modulate(x,shift,scale)=x*(1+scale[:,None])+shift[:,None]; apply_gate=x*gate[:,None].
RoPE (posemb_layers): use_real, interleaved-pair (`rotate_half` on adjacent 2-tuples +
`repeat_interleave(2)`), per-axis freqs concatenated [t16|h56|w56]=128, theta 256, IMG
tokens only. == wan/rope3d.rs convention -> reuse `RopeEmbedder::new(256, [16,56,56], ..)`.
grid order (t outer, w inner), token idx = t*H*W+h*W+w.

VAE (`hunyuanvideo15_vae_fp16.safetensors`, 218 F16 tensors; decoder = 109):
- Names: `decoder.{conv_in.conv, mid.{block_1,attn_1,block_2}, up.{0..4}.block.{0,1,2}.
  {conv1,conv2}.conv, up.{0..3}.upsample.conv.conv, norm_out, conv_out.conv}`. RMS norm
  = `*.{norm1,norm2,norm}.gamma` [C,1,1,1] (channel-first, gamma only, F.normalize ->
  scale dim^0.5). Convs kt=3 causal (front-pad T by 2, replicate), 1x1x1 for attn q/k/v.
- Config maps onto WanVaeConfig: dim_mult [1,2,4,8,8], decoder_base_dim 128, z_dim 32,
  in/out 3, patch_size 1, is_residual true, spatial 16 / temporal 4, num_res_blocks 2.
  dec dims [1024,1024,512,256,128]; temporal-up on stages 0,1; spatial-up on 0,1,2,3.
  conv_in residual: `h = conv_in(z) + z.repeat_interleave(1024/32=32)`.
- **VAE DECODE = FORK, do NOT reuse wan decode_frame.** Two hard divergences: (a) Hunyuan
  `mid.attn_1` is CAUSAL spatio-temporal (frame i attends 0..i over (f*h*w) tokens); wan
  mid-attn is per-frame spatial (asserts T=1, no mask). (b) Hunyuan Upsample = ONE causal
  conv -> out*factor then pixelshuffle rearrange (first latent frame spatial-only + half
  channels; rest full 2*2*2 / spatial 1*2*2) + repeat_interleave residual; wan uses a
  separate time_conv+spatial_conv streaming feat_cache. Write a Hunyuan decoder that
  REUSES only the low-level WGSL ops (conv3d_run, rmsnorm3d, silu, sdpa, transpose12,
  add) with whole-tensor (non-streaming) causal convs for correctness-first parity;
  add tiling/streaming later for perf. Output range [-1,1] -> pipeline does z/scaling
  then *0.5+0.5.clamp.

## VAE Rust impl approach (turnkey; signatures grabbed from wan/vae.rs 2026-06-29)

Write `thinfer-models/src/hunyuan/vae.rs`: WHOLE-TENSOR (non-streaming) decode, NOT the
wan decode_frame/decode_tile/feat_cache streaming harness (that's for per-frame causal
convs + spatial tiling; we want correctness-first at tiny dims). Reuse `WanVaePipelines`
(`::compile(backend)`) + these helpers verbatim: `conv3d_run(scope,pl,x,shape,&ConvBufs,
cout,(kt,kh,kw),(pad_t,pad_h,pad_w),(st,sh,sw))` (front-pads time, symmetric HW -> a
causal kt=3 conv is just `conv3d_run(pad=(2,1,1))` over the WHOLE [C,T,H,W], no
feat_cache), `rmsnorm3d_run` (applies gamma + dim^0.5 scale + eps over C; matches
RMS_norm), `silu_run`, `add_run`, `transpose12_run`, `conv3d_1x1x1` (attn q/k/v/proj +
the 1x1 path).
- Loader: clone the `ConvWeights{weight,bias}` / `RmsWeights{gamma}` / `ResnetWeights` /
  `AttnWeights{norm,to_qkv,proj}` / `MidBlockWeights` / `UpBlockWeights` /
  `VaeDecoderWeights` structs but with HUNYUAN names: `decoder.conv_in.conv`,
  `decoder.mid.{block_1,attn_1,block_2}`, `decoder.up.{i}.block.{j}.{conv1,conv2}.conv`,
  `decoder.up.{i}.upsample.conv.conv`, `decoder.norm_out`, `decoder.conv_out.conv`, norms
  `*.gamma`. NOTE attn q/k/v are SEPARATE (`attn_1.{q,k,v,proj_out}`), NOT a fused to_qkv
  -> 3 ConvBufs. No post_quant_conv. conv_in residual: after conv_in, add
  `z.repeat_interleave(1024/32=32)` (a small host or dupup-style broadcast add).
- Decode forward (per the reference Decoder.forward): conv_in(+residual) -> mid
  (resnet, CAUSAL attn, resnet) -> for stage 0..4: 3 resnets then (if stage<4) upsample
  -> norm_out -> silu -> conv_out. Pre-scale z by 1/1.03682 host-side; output raw [-1,1]
  (pipeline does *0.5+0.5.clamp).
- NEW WGSL OP NEEDED: 3D pixel-shuffle / depth-to-space for the upsample
  `(r1 r2 r3 c) f h w -> c (f r1)(h r2)(w r3)` (r1=2 temporal only on stages 0,1; r2=r3=2
  spatial). First latent frame is special (spatial-only, half channels). Plus the
  repeat_interleave residual shortcut. Add `pixelshuffle3d` WGSL + conformance op test.
  (Confirms the plan's "only VAE needs new ops".)
- CAUSAL mid-attn: engine `scope.sdpa` takes only a SCALAR mask uniform (no additive
  buffer). At f=1 parity dims causal==full -> validate the rest first with the plain
  sdpa. For f>1 correctness add an additive-mask sdpa variant (frame i attends 0..i over
  f*h*w tokens) OR a KV-growing loop. Gate multi-frame VAE parity behind that.
- Parity: `tests/hunyuan/vae_parity.rs` (copy `tests/ltx/vae_parity.rs` + `parity_util.rs`)
  vs `gen_vae_decode_ref.py` (DONE, validated). `hunyuan-e2e` cargo feature. Compare
  conv_in/mid/up_NN/conv_out/video bands (slope ~1, low rel-rmse). Start f=1,h=w=8.

### VAE impl: template + the TWO new-op needs (deep-scoped 2026-06-29)

BEST TEMPLATE = `ltx/video_vae.rs` (NOT wan): it's a from-scratch WHOLE-TENSOR
SINGLE-SUBMIT decoder (DecoderW/DecoderH/DecoderBufs + `conv_w()` name builder +
`DecoderH::register` via `crate::common::loader::register_passthrough` (pub(crate),
usable) + op wrappers all inside one `BatchScope` + `scope.submit_void()` + a
post-submit `read_buffer` tap helper + a tiling fallback). Clone its shape. Its
`LtxVaePipelines` already bundles conv3d / silu / add / concat_time / **depth_to_space**
(`DepthToSpace3dF32`, the pixelshuffle!) but uses `pixel_norm` (LTX) + NO attention.
Hunyuan needs instead: RMS norm (channel-first) + SDPA (causal mid-attn) -> pull
`RmsNorm3dF32` + `SdpaF32LargeD` + `Transpose12F32` WGSL into the Hunyuan pipeline
bundle (all exist in wan). So Hunyuan's pipeline = conv3d + rmsnorm3d + silu + add +
concat_time + depth_to_space + sdpa_large_d + transpose12 (assemble from existing op
structs; mirror LtxVaePipelines::compile_with).

RESUME NOTE (use CORE op structs + own uniform builders, LTX-style; do NOT pub/use
wan's private `conv3d_run`): hunyuan/vae.rs owns its `conv3d_uniform_bytes` etc. (copy
ltx `conv3d_k3` shape) so the conv wrapper writes `pad_mode=1` directly (the U field
landed in conv3d.rs). Conv3dF32 / RmsNorm3dF32 / SiluF32 / AddF32 / ConcatTimeF32 /
DepthToSpace3dF32 / SdpaF32LargeD / Transpose12F32 are the core op structs to compile.
For rmsnorm3d/sdpa/transpose12 uniform byte layouts, copy them from wan/vae.rs
(`rmsnorm3d`/`sdpa_uniform`/`transpose12_uniform`); for conv3d/silu/add/concat_time/
depth_to_space copy from ltx/video_vae.rs. register_passthrough = `crate::common::
loader::register_passthrough`.

NEW-OP / parity blockers (the VAE's hard part):
1. **Replicate padding. SOLVED (approach locked).** Hunyuan `CausalConv3d` =
   `F.pad(mode='replicate')` on ALL of T (front kt-1=2) + H/W (sym 1). conv3d.rs ALWAYS
   zero-pads (im2col gather: OOB -> v=0.0, no mode). FIX = repurpose the conv3d `U`
   struct's spare `_pad0: u32` as `pad_mode` (0=zero default, 1=replicate-edge). In
   build_wgsl's B-tile gather (the `if (ti>=0 && ...)` block ~line 402), when pad_mode==1
   clamp `ti=clamp(ti,0,t_in-1)`, `hi`, `wi` and ALWAYS load (drop the zero branch). One
   edge-clamp serves T-front + H + W replicate exactly (frame0 conv sees [f0,f0,f0];
   causal geometry has no back overflow). NO uniform-size change; existing callers write
   0 -> stay zero-pad. Add a `pad_mode` arg to wan `conv3d_run` (default 0) + a replicate
   conformance case. Hunyuan conv wrapper passes pad_mode=1, pad=(2,1,1) for kt3 /
   (0,0,0) for kt1.
2. **Upsample first-frame split.** Hunyuan Upsample (temporal): conv -> out*8 ch, then
   first latent frame uses `(r2 r3 c)` with HALF channels spatial-only (1 out frame),
   rest use `(r1 r2 r3 c)` full 2*2*2; + repeat_interleave residual. LTX `depth_to_space`
   has `t_drop` (drop leading frame, out.t = T*p0 - t_drop) -> matches the DIMENSION
   (2T-1) but likely NOT the first-frame half-channel VALUE split. VERIFY DepthToSpace3d
   WGSL semantics vs Hunyuan rearrange; if mismatch, extend it or add a hunyuan variant.
   The residual shortcut (repeat_interleave) is separate.

MILESTONE SPLIT (incremental, each parity-green before next):
- 1a: pipeline + loader (conv_in,mid only) + whole-tensor harness + conv_in(+residual)
  + mid (resnet, causal-attn[trivial at f=1], resnet). Compare conv_in.bin + mid.bin.
  REQUIRES new-op #1 (replicate conv) -- it's the first thing conv_in needs.
- 1b: up stages (needs new-op #2) + norm_out + conv_out -> full video.bin parity. Then
  f>1 (causal additive-mask sdpa, task 8).

## Engine analogs / reuse

| Component | Reuse from | Notes |
|---|---|---|
| Qwen2.5-VL text+vision enc | `qwen_image/{text_encoder,vision}.rs` | same checkpoint, done |
| RoPE3D interleaved-pair | `wan/rope3d.rs` | dims 16/56/56, theta 256 |
| causal-conv3d VAE | `wan/vae.rs` OPS ONLY | FORK the decoder fwd (causal mid-attn + diff upsample); reuse conv3d/rmsnorm3d/silu/sdpa/transpose ops |
| MMDiT joint-attn + adaLN mod | `qwen_image/dit.rs` | 6-chunk mod, joint attn |
| flow-match Euler scheduler | `wan` flow-match / `ltx` | + flow_shift 7, MeanFlow |
| I2V latent-concat | `wan` I2V | frame0-only + mask + 65ch |
| MeanFlow 2nd timestep | none (new, small) | timestep_r -> modulation |
| SingleTokenRefiner (2-layer) | none (new, small) | |
| ByT5 glyph / SigLIP | none (new, optional) | extra joint tokens |

## Variant matrix (official tencent/HunyuanVideo-1.5) + strategy

step_distilled exists ONLY for **480p I2V** (8/12-step, CFG-off, flow_shift 7) + the two
SR stages. No 720p step-distill, no T2V step-distill. T2V = base(50) / cfg_distilled(50).
- **TARGET (user-confirmed, BUILD FIRST): 480p T2V, lightx2v 4-step distill** (CFG-off),
  native text->video. Source = `lightx2v/Hy1.5-Distill-Models/hy1.5_t2v_480p_lightx2v_4step
  .safetensors` (fp16, 16.7GB), loaded DIRECT. WHY T2V-first: simplest (no image
  conditioning / t2i front-end / two-phase web), and lets us EYEBALL Hy1.5 faces fast --
  the real unknown. If 4-step T2V faces satisfy, I2V may be unneeded; if not, I2V adds on
  the SAME shared core (T2V = I2V minus the 65ch image conditioning).
- WHY this exists (corrects an earlier wrong claim): Tencent step-distill is I2V-only, BUT
  lightx2v shipped a 4-step CFG-off T2V distill (~25x vs 50-step). The Civitai 325MB file
  is its LoRA form; the HF repo is the full merged checkpoint.
- I2V = PHASE 2 (Tencent `480p_i2v_step_distilled`, 12-step MeanFlow; fp16 already
  downloaded). Its UX (supply-image OR t2i front-end + two-phase web review) is below.
- QUALITY+PERF lever = the official CASCADE, NOT native 720p (50-step at 720p = the 8GB
  VRAM/wall wall): 480p 8-step gen -> **distilled SR** 480->720 (`720p_sr_distilled`,
  6-8 step CFG-off) -> optional 720->1080 (`1080p_sr_distilled`). All step-distilled =
  fast. SR is PHASE 2 (recon running on its arch + conditioning; reuse main DiT if same
  MMDiT). lightx2v fp8 repo = DEAD END (fp8 has no WGSL path; base 720p; no distill).

## Decisions

- **Weights = EXISTING HF repos, loaded DIRECT. NO self-quant, NO making our own GGUF**
  (rule: memory feedback_no_self_quantize). No GGUF exists for any FAST variant (existing
  GGUFs = base/cfg-distilled only = slow; checked the quantized-base-model pages). So:
  - **DiT (T2V)**: `lightx2v/Hy1.5-Distill-Models/hy1.5_t2v_480p_lightx2v_4step
    .safetensors` fp16, loaded direct; engine narrows fp16->bf16 on upload + runtime
    i8-DP4A / f16-SDPA for perf (engine compute path, not a quant artifact). ~16.7GB
    resident (fine on RAM; VRAM bounded by residency budget). fp8 variant = UNUSABLE
    (no WGSL fp8). Phr00t Rapid-AIO = REJECTED (fp8 + unverifiable 50/50 merge).
  - **VAE**: Comfy-Org `hunyuanvideo15_vae_fp16.safetensors` (2.5GB, already downloaded).
  - **Encoder**: reuse the EXISTING qwen_image Qwen2.5-VL-7B port (same checkpoint).
    For Hy1.5, prefer a CLEAN Qwen2.5-VL-7B-Instruct GGUF (not qwen_image's abliterated)
    -- pick an existing published GGUF.
- I2V DiT (phase 2) = Tencent `480p_i2v_step_distilled` fp16 (Comfy-Org repackaged,
  already downloaded), same direct-load rule.
- **Runtime perf**: reuse the Wan22 bf16-residual fast paths (f16 SDPA + i8 DP4A on
  normed sites). MMDiT joint-attn is dense O(N^2) at patch[1,1,1] token counts -- the
  perf lever to watch. Coopmat narrow-scoped to the matmul hot path if it earns it.
- **Validation = FULL e2e parity test** (`hunyuan-e2e`, FastWan `video_e2e` pattern):
  pinned noise + fixed prompt -> compare per-step DiT velocity + post-step latents +
  final VAE pixels vs pyref within calibrated bands. PLUS per-component gates (vae,
  encoder, refiner, dit) at tiny dims (q8 canary) for bisection. The 8.3B DiT + 7B
  Qwen2.5-VL won't co-fit in one pyref process at fp32 (RAM), so SHARD the pyref across
  processes: stage 1 encode -> dump hidden states; stage 2 DiT (consumes dumped
  embeddings + pinned noise) -> dump per-step velocity + latents; stage 3 VAE -> dump
  pixels. Each stage's weights load independently (bf16, tiny dims) -> respects the
  machine while covering the whole pipeline. Engine e2e replays the same pinned noise +
  dumped embeddings and compares end to end. pyref: `projects/thinfer-conformance/python/
  thinfer_pytorch_ref/hunyuan/` (uv; `--with einops --with loguru`).

## UX: I2V + optional t2i front-end (user-confirmed, NOT hidden)

Always presented as I2V. Two input modes, both explicit:
- (a) user supplies the reference image.
- (b) user supplies video prompt + picks an image model + image prompt -> engine
  generates the FIRST FRAME via an existing t2i model (Z-Image-Turbo etc.).
- **Web = two-phase with a human checkpoint**: gen image -> DISPLAY -> user
  approves/regenerates -> THEN animate. Architecturally = TWO ordinary jobs (existing
  t2i job, then new I2V job consuming that image). NO server-side pause/cross-request
  state (respects feedback_no_cross_request_state). Web form orchestrates the two jobs
  + the review step.
- **CLI** = same capability: `--image <path>` (supply) OR `--image-model <id>
  --image-prompt <text>` (engine emits the first frame as an artifact, then animates).

## MeanFlow sampling (parity-critical, from reference; default N=12)

- sigmas = linspace(1,0,N+1); shift sigma' = 7*sigma/(1+6*sigma) (SD3 form, flow_shift 7);
  timesteps = sigma'[:-1]*1000; dt = sigma'[i+1]-sigma'[i] (negative); Euler x += v*dt.
- MeanFlow: per step pass t=sigma'[i]*1000 AND r=sigma'[i+1]*1000 (r=0 on last step);
  embed BOTH via separate TimestepEmbedders, ADD both into the modulation vec. v used
  directly (no x=sample-v*t transform; MeanFlow property is purely in the (t,r) input).
- CFG OFF (guidance_scale 1.0): single forward/step, no neg prompt.
- I2V every step: rebuild concat([noisy(32), img_cond(32 on frame0/0 else), mask(1=frame0)])
  = 65ch; img_in Conv3d is 65->2048. cond block is constant across steps; only the 32
  noisy ch change. VAE decode: /scaling(1.03682), then *0.5+0.5.

## Shared building blocks (refiner + DiT; from reference modules/)

Exact, parity-critical (HV1.5 `modules/{token_refiner,embed_layers,modulate_layers}.py`):
- `modulate(x,shift,scale)` = `x*(1+scale[:,None]) + shift[:,None]`.
- `apply_gate(x,gate)` = `x*gate[:,None]` (plain multiply; tanh=False default).
- `ModulateDiT` = SiLU -> Linear(hidden, factor*hidden), zero-init. Double block
  factor=6 per stream -> (shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp).
- `TimestepEmbedder`: sinusoid `cat([cos(args),sin(args)])` (COS FIRST), freq_dim 256,
  max_period 1e4 -> Linear(256,2048) -> SiLU -> Linear(2048,2048).
- `TextProjection` = Linear(in,h) -> SiLU -> Linear(h,h).

SingleTokenRefiner(in 3584, hidden 2048, heads 16, depth 2, mlp x4 gelu? act=silu default,
qk_norm=False=Identity): `c = t_embedder(t) + c_embedder(mean_masked(x))`;
`x = input_embedder(x)`; then 2x IndividualTokenRefinerBlock(x,c,mask):
- `norm1`=LayerNorm(affine,eps1e-6); qkv Linear(2048,6144,bias); q/k norm=Identity;
  bidirectional SDPA (pad mask, mask[:,0]=True); `x += apply_gate(self_attn_proj(attn),
  gate_msa)`; `x += apply_gate(mlp(norm2(x)), gate_mlp)`. norm2=LayerNorm(affine,eps1e-6).
- adaLN_modulation = SiLU -> Linear(2048,4096) -> chunk2 = (gate_msa, gate_mlp). GATES ONLY
  (no shift/scale in refiner). Tensor names: `txt_in.{input_embedder,t_embedder.mlp.{0,2},
  c_embedder.{linear_1,linear_2}, individual_token_refiner.blocks.{0,1}.{norm1,
  self_attn_qkv,self_attn_proj,norm2,mlp.{fc1,fc2}? ,adaLN_modulation.1}}`.
- NOTE: refiner uses **LayerNorm (affine)**, NOT RMSNorm -> check engine has an affine
  LayerNorm op (else new op). Reuse qwen_image block matmul/sdpa/silu helpers.

## SR cascade (PHASE 2, recon DONE -- reuses main DiT)

SR `720p_sr_distilled`/`1080p_sr_distilled` = SAME MMDiT (2048/16/54, same RoPE/MeanFlow),
only deltas: img_in Conv3d 98->2048 (concat_condition=false, in_channels=98), a 66ch
condition builder (upscaled+flow-noised LQ latents via a small `SRTo{720,1080}pUpsampler`
conv net + t<700 LQ-zeroing), and reuses the SAME VAE+encoders. SR params: 720p 6-step
shift2.0, 1080p 8-step shift2.0, CFG off. DiT weights 33GB bf16 each; upsampler 86MB/201MB.
Build after the 480p I2V core lands + validates.

## Sequencing

STATUS (landed): module scaffold + `config.rs` (all dims/sampling, compiles) + `scheduler.rs`
(FlowMatchSchedule, unit-tested vs reference) DONE; lib.rs registered; fmt+clippy clean.
Confirmed: ONLY the VAE needs new WGSL ops (refiner/DiT use existing LayerNormF32 /
BcastModulateF32 / BcastMulF32 / matmul / QkvSplit / op_sdpa). NEXT: refiner -> encoder
wiring -> DiT -> denoise loop -> VAE -> app/CLI/serve/web -> GPU eyeball (ASK).

T2V-FIRST (shared core = DiT/VAE/encoder/scheduler; T2V = I2V minus image conditioning):
0. Downloads: VAE fp16 DONE; lightx2v T2V 4-step fp16 DOWNLOADING. Recon on T2V config +
   lightx2v 4-step sampling (meanflow? steps/shift/sampler) RUNNING.
1. Scaffold: `VideoModelId::Hunyuan15T2v` (+ name/defaults/manifest), `hunyuan/` module
   skeleton, config consts, register in models lib.rs + app model.rs + executor route.
2. VAE decode (16x/4x/32ch causal-conv3d; Wan VAE family). Parity gate vs pyref.
3. Wire Qwen2.5-VL encoder (reuse qwen_image) + SingleTokenRefiner (2-layer, new).
   Parity gate.
4. DiT: img_in 1x1x1 Conv3d **65->2048** (concat_condition=true even for T2V: input =
   [noise32|zeros32|zeros1]), RoPE3D interleaved [16,56,56]/theta256, 54 double-stream
   blocks (6-chunk adaLN mod + joint img/txt attn + qk-RMSNorm + gelu_tanh FFN), SINGLE
   time embed (NO time_r_in -- meanflow OFF for lightx2v T2V, confirmed via zero time_r*
   tensors), cond_type_embedding[0] added to text tokens, final layer 2048->32. ByT5 +
   SigLIP streams masked-zero -> OMITTED for T2V (bit-faithful). Block + full-DiT parity.
5. Scheduler: 4-step Euler, denoising idx [1000,750,500,250], shift **9.0** (model-card;
   5.0 = A/B fallback). sigmas_full=linspace(1,0,1001)[:-1], sigma'=9s/(1+8s),
   timesteps=[1000,964.29,900,750]. x += (sigma_{i+1}-sigma_i)*v, terminal sigma 0. CFG
   OFF, single forward, no neg prompt. No image concat (cond block all-zero). DiT loads
   as the WHOLE transformer (1793 tensors, standalone, not a LoRA).
6. e2e T2V health + FACE eyeball (GPU -- ASK first). The go/no-go on whether I2V is needed.
7. CLI flags + serve wire/api + web UI (same change, CLI/web parity). Deploy (ASK first).
8. PHASE 2 (if faces need it): I2V (65ch concat + supply-image/t2i-front-end + two-phase
   web review) on the same core; then SR cascade; then ByT5 glyph.

## Open / risks

- MeanFlow `timestep_r` schedule: extract from `hunyuan_video_pipeline.py:1207-1234`.
- Self-quant tooling: HV1.5 DiT arch not in stock llama.cpp gguf -- may need a custom
  name-map convert (city96 ComfyUI-GGUF style), F16 GGUF then llama-quantize.
- Dense attention at patch[1,1,1] = large seq; reference uses STA/SSTA sparse attn.
  May need windowed/tiled attention (reuse Wan windowed self-attn) for perf at 480p.
- SigLIP gated on official path; Comfy-Org repackage is free.
