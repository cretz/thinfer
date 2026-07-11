# DreamID-V port (video face swap on Wan2.1-1.3B) + face-swap web/CLI exposure

Target: `DreamID-V-Wan-1.3B-Faster` (16 steps, lower VRAM; FastWan-speed class).
Ref impl (read-only clone): `third-party/DreamID-V/dreamidv_wan_faster/`.
Base weights reuse: `Wan-AI/Wan2.1-T2V-1.3B` (VAE + no live text encoder needed).

## Two deliverables in this track
1. Expose EXISTING ONNX face-swap (HyperSwap) in web UI + CLI already done. CLI
   exists (`generate face-swap`); WEB is missing -> add. Shares the encrypted
   video-upload path with #2.
2. Port DreamID-V-Faster as a new video model (diffusion face swap).
Video upload from web: stream/RAM; if spilled to disk it MUST be encrypted
(reuse vault AES-GCM). e2e parity + ram/vram budgets + best quality, as usual.

## DreamID-V-Faster architecture (REVERSE-ENGINEERED, confirmed from source)

It is STOCK Wan2.1-1.3B T2V DiT + exactly two structural deltas. Config
(`configs/wan_swapface.py`): model_type i2v, patch (1,2,2), dim 1536, ffn 8960,
freq_dim 256, in_dim 48, num_heads 12, num_layers 30, qk_norm true,
cross_attn_norm true, eps 1e-6, vae_stride (4,8,8). Same block internals as our
Wan2.1 (self-attn RoPE3D interleaved-pair + t2v cross-attn + GELU-tanh FFN +
6-way adaLN `modulation + e`, RMSNorm qk, 2-way head modulation).

Deltas vs stock Wan2.1-1.3B:
- **in_dim 48** (not 16): forward concatenates on CHANNEL dim
  `x_in = cat([noise(16), y], dim=0)` where `y = cat([video_latent(16),
  mask_latent(16)])` = 32ch. So patch_embedding is Conv3d(48 -> 1536, k=(1,2,2),
  s=(1,2,2)). Just a wider conv weight; loader handles.
- **ref_conv**: NEW `Conv2d(16 -> 1536, k=(2,2), s=(2,2))` patchifies the source
  face latent (`img_ref`, 16ch, SINGLE frame) into prefix tokens. Prepended to
  the token sequence; grid F -> F+1; seq_len += ref tokens. RoPE applied over the
  full F+1 grid. After all 30 blocks, `x = x[:, ref_len:]` strips the prefix,
  grid back to F, then head + unpatchify. (ref_conv == a 2x2 spatial patchify =
  linear over 16*2*2 -> 1536; reuse our image patchify/linear.)

NO pose branch / NO projector / NO CLIP in the faster variant (all commented
out; they live only in the regular `dreamidv_wan/` variant). Faster is the clean
minimal one.

Text context: BAKED. `dreamidv_wan_faster/context.pth` (4.2MB) is the fixed
umT5-xxl embedding of the prompt "chang face" (list of tensors, [~L,4096]). No
live text encoder at inference. -> Convert context.pth to our tensor format and
COMMIT as a small asset (or bake into the manifest). text_embedding Linear
(4096->1536)+GELU+Linear still runs on it (weights in the dreamidv ckpt).

## Inference pipeline (`wan_swapface.py::DreamIDV.generate`)
Inputs: target VIDEO (the clip to swap a face into), a face MASK video (DWPose
convex-hull face region, dilated 15x15), a source face IMAGE (512x512 crop).
1. VAE-encode each (Wan2.1 VAE, our encoder exists):
   - video -> `latents_ref["video"]` (16ch)
   - mask  -> `latents_ref["mask"]`  (16ch)  [mask NOT /0.5 normalized; video+img ARE Normalize(0.5,0.5)]
   - image -> `latents_ref["image"]` (16ch, padded white-bg to video aspect)
   Preprocess: NaResize to sqrt(W*H) downsample-only, DivisibleCrop to
   (vae_stride*patch)=(16,16), Normalize(0.5,0.5) for video+image (NOT mask),
   layout c t h w. frames truncated to 4n+1.
2. y = cat([video_latent, mask_latent]) (channel). img_ref = image_latent.
3. Scheduler: FlowUniPCMultistepScheduler, shift=5 (set_timesteps shift arg),
   num_train_timesteps 1000, 16 steps. (We have UniPC for Wan.)
4. Denoise loop per step (CFG on IMAGE ref only):
   - pos_tiv = model(latents, t, context, seq_len, y=[y], img_ref=[img_ref])
   - pos_tv  = model(latents, t, context, seq_len, y=[y], img_ref=[zeros])
   - noise_pred = pos_tiv + guide_scale_img*(pos_tiv - pos_tv), scale=4.0
   - latents = unipc.step(noise_pred, t, latents)
   Two DiT forwards/step (image-CFG). noise seeded fp32.
5. VAE-decode latents -> RGB video (our decoder). value_range (-1,1).
Size default 832*480 (product 1280*720 for best). frame_num 81 (4n+1), fps 24
save (config sample_fps 16). seq_len = ceil(H'*W'/(p*p) * F') incl the +1 ref.

## Mask preprocessing (DWPose) -- DEFERRED to after DiT parity
`pose/extract.py`: yolox_l.onnx (person det) -> dw-ll_ucoco_384.onnx (133
wholebody keypoints) -> face pts [24:92] -> per-frame cv2.convexHull -> fillPoly
white -> dilate(15x15). Both ONNX run through OUR onnx executor (op coverage TBD;
yolox=conv, dwpose=RTMPose SimCC head -> may need new ops). The repo ships a
PRECOMPUTED mask (`assets/test_case/.../a_girl_mask.mp4`) so DiT+VAE parity does
NOT need DWPose. Product/live path DOES (uploaded video -> mask at runtime).

## Weights (HF XuGuo699/DreamID-V)
- dreamidv_faster.pth 5.69GB (the faster DiT ckpt; state_dict, loaded strict=False
  into WanModel). INSPECT tensor names before loader (torch .pth -> need
  safetensors-ish reader OR convert once). Delta tensors: patch_embedding (48ch),
  ref_conv.{weight,bias}. Rest = stock Wan2.1 names (blocks.N.*, text_embedding,
  time_*, head.*).
- Wan2.1_VAE.pth (reuse existing cached Wan2.1 VAE; our encoder+decoder).
- yolox_l.onnx 217MB, dw-ll_ucoco_384.onnx 134MB (DWPose; deferred).
- context.pth 4.2MB (baked text; commit converted).

## Reuse map (confirmed)
- Wan DiT blocks/self-attn/cross-attn/ffn/adaLN/RoPE3D/RMSNorm: `wan/dit.rs`,
  `wan/dit_block.rs` (`WanDitConfig` @ dit_block.rs:101), `wan/rope3d.rs`.
- WanVaeConfig::wan2_1() + encoder (`register_encoder`, VaeEncoder*) + decoder:
  `wan/vae.rs`. ENCODER EXISTS (built for I2V/AR conditioning). Verify encode fn.
- UniPC scheduler: `wan/unipc.rs`. Pipeline pattern: `wan/pipeline.rs`
  (`WanVariant`). Patchify: `wan/patchify.rs`.
- ONNX executor + face-swap: `thinfer-core/src/onnx/`, `thinfer-models/src/
  faceswap/`, `thinfer-cli/.../generate/faceswap.rs`.
- Vault crypto (AES-256-GCM/Argon2id) for encrypted disk spill: `thinfer-app`
  vault module.

## Build order (gated; parity before perf, per house rules)
1. Inspect dreamidv_faster.pth tensor names; decide load path (convert to
   safetensors once, or a torch-pickle reader). Convert context.pth.
2. `wan/` config: add DreamID-V variant (dim1536/ffn8960/L30/H12, in_dim 48,
   ref_conv). WanVariant arm. Loader for the 2 delta tensors + stock Wan names.
3. DiT forward delta: 48ch patch-embed + ref_conv prefix tokens (grid F+1 +
   strip). Reuse the Wan block loop unchanged.
4. Pipeline: VAE-encode(video,mask,image) -> denoise (image-CFG, 2 fwd/step,
   UniPC shift5 16step) -> VAE-decode. Baked context load.
5. e2e parity gate (conformance): pyref = run the faster pipeline in torch at
   TINY dims on the committed a_girl clip + precomputed mask + ref image; diff
   latent/RGB bands. Commit tiny fixtures (clip+mask+image+context).
6. DWPose port: yolox + dwpose ONNX through our executor (op coverage pass);
   convex-hull+dilate mask builder (host). Live path: uploaded video -> mask.
7. App wiring: VideoModelId::DreamIdV + executor run_dreamidv + request (video
   + source-image inputs) + defaults. Encrypted video upload (serve multipart or
   base64 -> RAM/stream; disk spill -> vault AES-GCM). Face-swap web exposure
   too (shares upload). CLI parity.
8. serve redeploy: DreamID-V + face-swap in the web model list.

## Perf targets / levers (after parity)
Wan2.1-1.3B is the SMALL Wan (30 layers) -> should be the fastest video DiT we
have. 2 fwd/step (image-CFG) doubles compute -- consider CFG-cache / single-pass
tricks only if quality-neutral (gate). Reuse coopmat+fast_sdpa (native).
tiny-VAE decode option. 81f full-attn SDPA is still the ceiling (see AnyFlow);
1.3B has 30 vs 40 layers and smaller dim so per-step is cheaper. Report REAL
trace numbers once the e2e gate runs; user knows it is FastWan-speed class, not
HyperSwap-speed.

## CONFIRMED IMPLEMENTATION SPEC (weights inspected 2026-07-05)

dreamidv_faster.pth = 827 tensors, ALL FP32, 1.419B params. Converted ->
`scratch/dreamidv/dit.safetensors` (5.68GB) + `scratch/dreamidv/
context.safetensors` (baked text, ~[4,4096] fp32). Names are ORIGINAL-WAN
convention (not diffusers). Block tensors (x30) match stock Wan exactly:
`blocks.N.{self_attn,cross_attn}.{q,k,v,o}.{weight,bias}`,
`{self_attn,cross_attn}.{norm_q,norm_k}.weight`, `ffn.0.{w,b}`(1536->8960),
`ffn.2.{w,b}`(8960->1536), `modulation`(1,6,1536), `norm3.{w,b}`. (No norm1/norm2
params: WanLayerNorm affine=False; only norm3 has affine b/c cross_attn_norm.)
Top-level:
- patch_embedding.weight (1536,48,1,2,2) + bias  [48-in = the delta]
- ref_conv.weight (1536,16,2,2) + bias           [the delta; Conv2d 16->1536 2x2]
- head.head.weight (64,1536) + bias; head.modulation (1,2,1536)
- text_embedding.0 (1536,4096)+b, text_embedding.2 (1536,1536)+b
- time_embedding.0 (1536,256)+b, time_embedding.2 (1536,1536)+b
- time_projection.1 (9216,1536)+b   [9216 = 6*1536]

LOADER PLAN (small delta): our `register_wan_dit_handles` reads DIFFUSERS keys;
reuse `wan::source::dit_gguf_renames(30)` (original-Wan -> diffusers canonical;
maps modulation->scale_shift_table, self_attn->attn1.to_*, cross_attn->attn2.*,
ffn.0->ffn.net.0.proj, ffn.2->ffn.net.2, norm3->?, time_embedding->
condition_embedder.time_embedder.linear_{1,2}, time_projection->
condition_embedder.time_proj, text_embedding->...text_embedder..., head.head->
proj_out, head.modulation->top scale_shift_table). VERIFY the exact map covers
all our keys (build the union catalog, assert 0 missing like Wan2.2 does).
Source = `RenamedSource::with_passthrough(SafetensorsSource(dit.safetensors),
dit_gguf_renames(30))`. ref_conv has NO diffusers equivalent -> passthrough +
NEW handle (reuse `register_conv_as_linear_bias`, same path patch_embedding
uses; ref_conv = linear over 16*2*2=64 -> 1536). This mirrors LongLive
(`LongLiveDitSide = RenamedSource<PytorchSource>`, source.rs:221).

DiT SURGERY (wan/dit.rs, gated on cfg.ref_conv; other Wan models bit-identical):
- LoadedWanDitHandles gains `ref_conv: Option<LinearBiasHandles>`.
- WanDitInputs gains `img_ref: Option<&[f32]>` = source-face latent [16,1,h,w].
- forward, when ref_conv present:
  1. rows_video = patch tokens as today (grid F,H,W). ALSO compute ref tokens:
     patchify img_ref spatially [16,1,h,w] -> [pph*ppw, 16*2*2=64] -> ref_conv
     linear -> [ref_rows, inner], ref_rows = pph*ppw (ONE frame of tokens).
  2. Allocate residual x at rows = ref_rows + n_tok; write ref tokens at offset
     0, video patch-embed output at offset ref_rows. (concat via sub-region
     binding or a copy op.)
  3. RoPE freqs over grid (ppf+1, pph, ppw): ref = frame 0, video = frames 1..F.
     (rope.lookup takes ppf; pass ppf+1 when ref present.)
  4. block loop over rows (unchanged, just larger rows).
  5. after blocks: head/proj_out + unpatchify operate on x[ref_rows..] only
     (the video rows), grid back to (F,H,W). Strip the prefix.
- patch_in = in_channels(48) * patch(1*2*2) = 192; grid/shape uses in_channels 48
  for patchify but out_channels 16 for unpatchify (WanDitShape already carries c;
  ensure patch uses in, unpatch uses out).

PIPELINE (new; own module, e.g. wan/dreamidv.rs or dreamidv/): reuses WanVae
encode+decode, WanDit::forward (extended), UniPC. Steps:
1. VAE-encode target video (Normalize 0.5), mask video (NO normalize), source
   image (Normalize 0.5, white-pad to video aspect) -> 3 latents (z16). Preprocess
   = NaResize to sqrt(W*H) downsample-only + DivisibleCrop to (16,16).
2. y = channel-cat(video_lat, mask_lat) [32ch]. Per step build 48ch input =
   cat(noise[16], y[32]); img_ref = image_lat.
3. Load baked context (context.safetensors) as inputs.text ([L,4096], DiT pads
   to 512). NO umT5.
4. Denoise (UniPC, shift 5, 16 steps, image-CFG): pos_tiv = fwd(img_ref=face),
   pos_tv = fwd(img_ref=zeros); noise_pred = pos_tiv + 4.0*(pos_tiv - pos_tv);
   unipc.step. 2 DiT forwards/step.
5. VAE-decode -> RGB video (value range -1..1).

PARITY GATE: pyref = run dreamidv_wan_faster torch pipeline at TINY dims on the
committed a_girl clip + precomputed mask (assets/test_case/.../a_girl_mask.mp4) +
ref image (assets/.../an_1.jpg), dump pre-VAE latent + RGB; diff bands. Component
first (VAE-encode of fixture, ref_conv tokens, one block), then full e2e. 1.42B
fp32 may fit tiny-dims torch on the 8GB card; else component-only per house
low-RAM pyref policy. Commit tiny fixtures (clip+mask+image+context.safetensors).

## Open decisions / risks
- .pth (pickle) loading: we have no torch-pickle reader. Cleanest = a one-time
  Python convert (.pth -> safetensors) committed as a scratch tool; runtime loads
  safetensors. Confirm quant: keep fp16/bf16 (no self-quant; runtime i8/coopmat
  at matmul sites like other Wan).
- DWPose ONNX op coverage unknown (SimCC/argmax/attention). If our executor
  can't run dw-ll_ucoco, fallback options: (a) extend executor, (b) a lighter
  face landmarker we already can run (the faceswap SCRFD gives 5pts, not the 68
  the hull wants) -> likely must port dwpose. This is the riskiest sub-item;
  isolated from the DiT (precomputed mask unblocks parity).
- Mask semantics: mask latent is the VAE-encode of a white-on-black face-region
  RGB video (NOT normalized by 0.5). Match exactly.
- Encrypted upload: video can be large (hundreds of MB). Prefer stream-to-swap
  (face-swap) / stream-decode-to-latent (dreamidv) so it never fully lands on
  disk. If it must spill (retry/seek), encrypt with a per-request ephemeral key.
